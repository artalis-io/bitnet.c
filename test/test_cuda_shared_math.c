#include "gpu_cuda.h"
#include "gpu_backend.h"
#include "backend_session.h"
#include "gguf.h"
#include "cuda_bf16_shared_math_reference.h"
#include "cuda_bf16_silu_profiles_reference.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t bf16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static BnGPUResidentView alloc(BnGPUBackend *g, void *r, size_t n) {
    BnGPUResidentView v; assert(!g->resident_alloc(r, BN_GGUF_TENSOR_BF16, n, &v)); return v;
}
static uint64_t digest(BnGPUBackend *g, void *r, BnGPUResidentView v) {
    uint16_t *x = malloc(v.count * 2); assert(x && !g->resident_read(r, v, x));
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < v.count; i++) {
        uint16_t u = x[i];
        if ((u & 0x7f80) == 0x7f80 && (u & 127)) u = 0x7fc0;
        h = (h ^ u) * UINT64_C(1099511628211);
    }
    free(x); return h;
}
static void guards(BnGPUBackend *g, void *r, BnGPUMoESharedBF16ResidentPlan p) {
    BnGPUMoESharedBF16ResidentPlan q;
#define REJECT(change) do { q=p; change; assert(g->moe_shared_bf16_resident(g->ctx,r,&q)==-1); } while(0)
    REJECT(q.n_tokens=0); REJECT(q.dim=0); REJECT(q.n_tokens=INT_MAX);
    REJECT(q.routed.count--); REJECT(q.down.count--); REJECT(q.logits.count--);
    REJECT(q.sigmoid.count--); REJECT(q.shared.count--); REJECT(q.out.count--);
    REJECT(q.logits.type=BN_GGUF_TENSOR_F32);
    REJECT(q.shared=q.down); REJECT(q.out=q.routed); REJECT(q.sigmoid=q.logits);
    REJECT(q.out.offset++);
    BnBackendSession *s=bn_backend_session_create(); assert(s);
    void *peer=bn_gpu_backend_resident_request(g,s); assert(peer);
    REJECT(q.out=alloc(g,peer,p.out.count));
    assert(g->moe_shared_bf16_resident(g->ctx,peer,&p)==-1);
    bn_backend_session_free(s);
#undef REJECT
}
static void hc_edges(BnGPUBackend *g, void *r) {
    const int nt=7,dim=37,streams=4; size_t wide=(size_t)nt*dim*streams;
    uint16_t *x=malloc(wide*2),*gate=malloc(wide*2),*block=malloc(nt*dim*2),*residual=calloc(wide,2),inject[28];
    assert(x&&gate&&block&&residual);
    const float levels[]={-87.5f,-88,-88.5f,0,1,INFINITY,-INFINITY};
    for(size_t i=0;i<wide;i++) { x[i]=bf16(ldexpf(i%2?-1.0f:1.0f,110+(int)(i%12))); gate[i]=bf16(levels[i/(dim*streams)]); }
    for(int i=0;i<nt*dim;i++) block[i]=bf16(ldexpf(i%2?-1.0f:1.0f,110+i%12));
    for(int i=0;i<nt*streams;i++) inject[i]=bf16(levels[i/streams]*4);
    BnGPUHyperMixResidentPlan m={0};m.n_tokens=nt;m.dim=dim;m.streams=streams;
    m.normalized=alloc(g,r,wide);m.gate=alloc(g,r,wide);m.out=alloc(g,r,nt*dim);
    assert(!g->resident_write(r,m.normalized,x)&&!g->resident_write(r,m.gate,gate));
    assert(!g->hyper_connection_mix_bf16_resident(g->ctx,r,&m));
    assert(digest(g,r,m.out)==BN_SHARED_HC_MIXED_FNV);
    BnGPUHyperCombineResidentPlan c={0};c.n_tokens=nt;c.dim=dim;c.streams=streams;
    c.residual=alloc(g,r,wide);c.block_out=alloc(g,r,nt*dim);c.inject=alloc(g,r,nt*streams);c.out=alloc(g,r,wide);
    assert(!g->resident_write(r,c.residual,residual)&&!g->resident_write(r,c.block_out,block)&&!g->resident_write(r,c.inject,inject));
    assert(!g->hyper_connection_combine_bf16_resident(g->ctx,r,&c));
    assert(digest(g,r,c.out)==BN_SHARED_HC_COMBINED_FNV);
    free(x);free(gate);free(block);free(residual);
}
int main(void) {
    BnGPUBackend *g=bn_gpu_cuda_create();assert(g);
    if(!g->moe_shared_bf16_resident){puts("CUDA shared math skipped: optional precision profile unavailable");bn_gpu_cuda_destroy(g);return 0;}
    BnBackendSession *s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);
    const int nt=65536,dim=5;size_t count=(size_t)nt*dim;
    uint16_t *logits=malloc(nt*2),*down=malloc(count*2),*routed=malloc(count*2),*sentinel=malloc((count+16)*2),*actual=malloc((count+16)*2);
    assert(logits&&down&&routed&&sentinel&&actual);
    for(int i=0;i<nt;i++)logits[i]=(uint16_t)i;
    for(size_t i=0;i<count;i++){down[i]=bf16((float)((int)((i*37+19)%509)-254)/128);routed[i]=bf16((float)((int)((i*23+71)%251)-125)/64);}
    for(size_t i=0;i<count+16;i++)sentinel[i]=0x5a5a;
    BnGPUMoESharedBF16ResidentPlan p={0};p.n_tokens=nt;p.dim=dim;
    p.routed=alloc(g,r,count);p.down=alloc(g,r,count);p.logits=alloc(g,r,nt);
    BnGPUResidentView storage[]={alloc(g,r,nt+16),alloc(g,r,count+16),alloc(g,r,count+16)};
    p.sigmoid=bn_gpu_resident_view_slice(storage[0],8,nt);
    p.shared=bn_gpu_resident_view_slice(storage[1],8,count);p.out=bn_gpu_resident_view_slice(storage[2],8,count);
    assert(!g->resident_write(r,p.routed,routed)&&!g->resident_write(r,p.down,down)&&!g->resident_write(r,p.logits,logits));
    for(int i=0;i<3;i++)assert(!g->resident_write(r,storage[i],sentinel));
    guards(g,r,p);
    for(int i=0;i<3;i++){assert(!g->resident_read(r,storage[i],actual));for(size_t j=0;j<storage[i].count;j++)assert(actual[j]==0x5a5a);}
    assert(!g->moe_shared_bf16_resident(g->ctx,r,&p));
    assert(digest(g,r,p.sigmoid)==BN_SHARED_SIGMOID_FNV);
    assert(digest(g,r,p.shared)==BN_SHARED_SHARED_FNV);assert(digest(g,r,p.out)==BN_SHARED_OUT_FNV);
    for(int i=0;i<3;i++){assert(!g->resident_read(r,storage[i],actual));for(int j=0;j<8;j++)assert(actual[j]==0x5a5a&&actual[storage[i].count-8+j]==0x5a5a);}
    assert(!g->resident_read(r,p.logits,actual)&&!memcmp(actual,logits,nt*2));
    assert(!g->resident_read(r,p.down,actual)&&!memcmp(actual,down,count*2));
    assert(!g->resident_read(r,p.routed,actual)&&!memcmp(actual,routed,count*2));
    hc_edges(g,r);
    size_t scaled_count=(size_t)65536*8;
    uint16_t *scaled_input=malloc(scaled_count*2);assert(scaled_input);
    for(size_t i=0;i<scaled_count;i++)scaled_input[i]=(uint16_t)(i/8);
    BnGPUScaledSiluResidentPlan scaled={0};scaled.rows=65536;scaled.cols=8;scaled.divisor=4;
    scaled.x=alloc(g,r,scaled_count);scaled.out=alloc(g,r,scaled_count);
    assert(!g->resident_write(r,scaled.x,scaled_input)&&!g->scaled_silu_bf16_resident(g->ctx,r,&scaled));
    assert(digest(g,r,scaled.out)==BN_SILU_SCALED_FNV);free(scaled_input);
    free(logits);free(down);free(routed);free(sentinel);free(actual);
    bn_backend_session_free(s);bn_gpu_cuda_destroy(g);
    puts("CUDA BF16 shared math PASSED: 65536 sigmoid inputs, shared product/add and HC subnormal edges");
    return 0;
}
