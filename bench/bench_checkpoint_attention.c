/* Optional captured-checkpoint attention subblock gate. Synthetic normalized
 * hidden inputs and independent KV prefixes; excludes outer norm/residual/FFN.
 * Independent reference operations produce all expected files. No timing claim. */
#include "gpu_cuda.h"
#include "gpu_backend.h"
#include "backend_session.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#ifdef BN_ENABLE_CUDA
static void *read_exact(const char *base,const char *name,size_t bytes) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s",base,name);
    if(n<0||(size_t)n>=sizeof(path)){fprintf(stderr,"fixture path too long\n");exit(1);}
    FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"cannot open fixture %s\n",path);exit(1);}
    void *data=malloc(bytes);
    if(!data||fread(data,1,bytes,f)!=bytes||fgetc(f)!=EOF){fprintf(stderr,"wrong fixture size or read failure: %s\n",path);fclose(f);free(data);exit(1);}
    fclose(f);return data;
}
static BnGPUResidentView allocate(BnGPUBackend*g,void*r,size_t n,const void*data) {
    BnGPUResidentView v={0};assert(!g->resident_alloc(r,BN_GGUF_TENSOR_BF16,n,&v));if(data)assert(!g->resident_write(r,v,data));return v;
}
static void compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*base,const char*name,size_t*checked) {
    void *got=malloc(v.count*2),*want=read_exact(base,name,v.count*2);assert(got);assert(!g->resident_read(r,v,got));
    if(memcmp(got,want,v.count*2)) {
        size_t diff=0,first=SIZE_MAX;for(size_t i=0;i<v.count;i++)if(memcmp((char*)got+i*2,(char*)want+i*2,2)){if(first==SIZE_MAX)first=i;diff++;}
        fprintf(stderr,"MISMATCH %s values=%zu first=%zu\n",name,diff,first);abort();
    }
    *checked+=v.count;free(got);free(want);
}
static void *capture(BnGPUBackend*g,const char*base,const char*name,int rows,int cols) {
    size_t bytes=(size_t)rows*cols*2;void*data=read_exact(base,name,bytes);
    void*w=g->buffer_create_quant_only(g->ctx,data,bytes,BN_GGUF_TENSOR_BF16,rows,cols);free(data);assert(w);return w;
}
static void linear(BnGPUBackend*g,void*c,void*w,int type,int rows,int cols,int batch,BnGPUResidentView x,BnGPUResidentView out) {
    BnGPUBF16LinearResidentPlan p={0};p.n_tokens=batch;p.rows=rows;p.cols=cols;p.weight.raw=w;p.weight.type=type;p.x=x;p.out=out;
    assert(!g->linear_bf16_resident(g->ctx,c,&p));
}
#endif
int main(int argc,char**argv) {
#ifndef BN_ENABLE_CUDA
    (void)argc;(void)argv;fprintf(stderr,"requires BN_ENABLE_CUDA=1\n");return 2;
#else
    if(argc!=4){fprintf(stderr,"usage: %s REFERENCE MODEL_GGUF NORM_CAPTURE\n",argv[0]);return 2;}
    BnGPUBackend*g=bn_gpu_cuda_create();assert(g);assert(g->qk_prep_group_bf16_resident&&g->attention_decode_group_bf16_resident);
    BnGGUFFile*f=bn_gguf_open_file(argv[2]);assert(f);void*w[4];
    const char*names[]={"blk.3.attn_q.weight","blk.3.attn_k.weight","blk.3.attn_v.weight","blk.3.attn_output.weight"};
    const int types[]={BN_GGUF_TENSOR_Q5_K,BN_GGUF_TENSOR_Q4_K,BN_GGUF_TENSOR_Q5_K,BN_GGUF_TENSOR_Q6_K};
    const int rows[]={12288,1024,1024,5120},cols[]={5120,5120,5120,6144};
    for(int j=0;j<4;j++) {
        int i=bn_gguf_find_tensor(f,names[j]);assert(i>=0);BnGGUFTensorInfo*t=&f->tensors[i];size_t bytes;
        assert(t->type==(uint32_t)types[j]&&t->dims[0]==(uint64_t)cols[j]&&t->dims[1]==(uint64_t)rows[j]);
        assert(bn_gguf_tensor_size(t->type,(size_t)rows[j]*cols[j],&bytes));
        w[j]=g->buffer_create_quant_only(g->ctx,bn_gguf_tensor_data(f,i),bytes,t->type,rows[j],cols[j]);assert(w[j]);
    }
    void*qw=capture(g,argv[3],"q-weight.bin",1,256),*kw=capture(g,argv[3],"k-weight.bin",1,256),*cs=capture(g,argv[3],"cos-sin.bin",262144,64);
    const int starts[]={0,1,5,33,65,129,257,1047};size_t checked=0;
    for(int batch=1;batch<=8;batch*=2)for(int seed=0;seed<2;seed++) {
        BnBackendSession*co=bn_backend_session_create();assert(co);void*c=bn_gpu_backend_resident_request(g,co);assert(c);
        assert(!g->linear_bf16_reserve(g->ctx,c,(size_t)batch*6144));
        size_t budget=g->qk_prep_group_workspace_size(g->ctx,batch);assert(!g->qk_prep_group_reserve(g->ctx,c,batch,budget));
        budget=g->attention_decode_group_workspace_size(g->ctx,batch);assert(!g->attention_decode_group_reserve(g->ctx,c,batch,budget));
        BnGPUResidentView x=allocate(g,c,(size_t)batch*5120,NULL),qg=allocate(g,c,(size_t)batch*12288,NULL),ki=allocate(g,c,(size_t)batch*1024,NULL),vi=allocate(g,c,(size_t)batch*1024,NULL);
        BnGPUResidentView q=allocate(g,c,(size_t)batch*6144,NULL),k=allocate(g,c,(size_t)batch*1024,NULL),v=allocate(g,c,(size_t)batch*1024,NULL),gate=allocate(g,c,(size_t)batch*6144,NULL);
        BnGPUResidentView attention=allocate(g,c,(size_t)batch*6144,NULL),gated=allocate(g,c,(size_t)batch*6144,NULL),out=allocate(g,c,(size_t)batch*5120,NULL);
        BnBackendSession*s[8];void*r[8];BnGPUQKPrepSequence prep[8]={{0}};BnGPUAttentionDecodeSequence attn[8]={{0}};char name[128];
        for(int i=0;i<batch;i++) {
            s[i]=bn_backend_session_create();assert(s[i]);r[i]=bn_gpu_backend_resident_request(g,s[i]);assert(r[i]);
            int pos=starts[i]+seed*4095,cap=pos+8;assert(!g->attention_bf16_reserve(g->ctx,r[i],1,24,cap));
            prep[i].request=r[i];BnGPUQKPrepResidentPlan*p=&prep[i].plan;
            p->n_tokens=1;p->n_heads=24;p->n_kv_heads=4;p->head_size=256;p->rotary_dim=64;p->position=pos;p->eps=1e-6f;p->weight_offset=1;p->q_weight=qw;p->k_weight=kw;p->cos_sin=cs;
            p->q_gate=bn_gpu_resident_view_slice(qg,(size_t)i*12288,12288);p->k_in=bn_gpu_resident_view_slice(ki,(size_t)i*1024,1024);p->v_in=bn_gpu_resident_view_slice(vi,(size_t)i*1024,1024);
            p->q=bn_gpu_resident_view_slice(q,(size_t)i*6144,6144);p->k=bn_gpu_resident_view_slice(k,(size_t)i*1024,1024);p->v=bn_gpu_resident_view_slice(v,(size_t)i*1024,1024);p->gate=bn_gpu_resident_view_slice(gate,(size_t)i*6144,6144);
            attn[i].request=r[i];BnGPUAttentionResidentPlan*a=&attn[i].plan;
            a->n_tokens=1;a->n_heads=24;a->n_kv_heads=4;a->head_size=256;a->position=pos;a->cache_capacity=cap;a->gated=1;a->scale=.0625f;
            a->q=p->q;a->k=p->k;a->v=p->v;a->gate=p->gate;
            a->attention_out=bn_gpu_resident_view_slice(attention,(size_t)i*6144,6144);a->out=bn_gpu_resident_view_slice(gated,(size_t)i*6144,6144);
            snprintf(name,sizeof(name),"b%d-s%d-r%d-key-initial.bin",batch,seed,i);void*data=read_exact(argv[1],name,(size_t)cap*1024*2);a->key_cache=allocate(g,r[i],(size_t)cap*1024,data);free(data);
            snprintf(name,sizeof(name),"b%d-s%d-r%d-value-initial.bin",batch,seed,i);data=read_exact(argv[1],name,(size_t)cap*1024*2);a->value_cache=allocate(g,r[i],(size_t)cap*1024,data);free(data);
        }
        for(int step=0;step<5;step++) {
            snprintf(name,sizeof(name),"b%d-s%d-t%d-input.bin",batch,seed,step);void*data=read_exact(argv[1],name,(size_t)batch*5120*2);assert(!g->resident_write(c,x,data));free(data);
            linear(g,c,w[0],types[0],12288,5120,batch,x,qg);linear(g,c,w[1],types[1],1024,5120,batch,x,ki);linear(g,c,w[2],types[2],1024,5120,batch,x,vi);
            BnGPUQKPrepGroupPlan qp={batch,prep};assert(!g->qk_prep_group_bf16_resident(g->ctx,c,&qp));
            BnGPUAttentionDecodeGroupPlan ap={batch,attn,BN_GPU_ATTENTION_SPLIT_SINGLETON};assert(!g->attention_decode_group_bf16_resident(g->ctx,c,&ap));
            linear(g,c,w[3],types[3],5120,6144,batch,gated,out);
            /* Entire graph is queued before any validation readback/fence. */
#define CHECK(V,N) do{snprintf(name,sizeof(name),"b%d-s%d-t%d-" N ".bin",batch,seed,step);compare(g,c,V,argv[1],name,&checked);}while(0)
            CHECK(qg,"qg");CHECK(ki,"ki");CHECK(vi,"vi");CHECK(q,"q");CHECK(k,"k");CHECK(v,"v");CHECK(gate,"gate");CHECK(attention,"attention");CHECK(gated,"gated");CHECK(out,"output");
#undef CHECK
            for(int i=0;i<batch;i++) {
                snprintf(name,sizeof(name),"b%d-s%d-t%d-r%d-key.bin",batch,seed,step,i);compare(g,r[i],attn[i].plan.key_cache,argv[1],name,&checked);
                snprintf(name,sizeof(name),"b%d-s%d-t%d-r%d-value.bin",batch,seed,step,i);compare(g,r[i],attn[i].plan.value_cache,argv[1],name,&checked);
                assert(!g->resident_synchronize(r[i]));prep[i].plan.position++;attn[i].plan.position++;
            }
            assert(!g->resident_synchronize(c));printf("b%d s%d step%d exact\n",batch,seed,step);fflush(stdout);
        }
        for(int i=0;i<batch;i++)bn_backend_session_free(s[i]);
        bn_backend_session_free(co);
    }
    for(int i=0;i<4;i++)g->buffer_destroy(g->ctx,w[i]);
    g->buffer_destroy(g->ctx,qw);g->buffer_destroy(g->ctx,kw);g->buffer_destroy(g->ctx,cs);bn_gguf_free(f);bn_gpu_cuda_destroy(g);
    printf("checkpoint projections -> grouped QK/RoPE -> grouped FA2/gate -> Q6_K PASSED: %zu BF16 values\n",checked);return 0;
#endif
}
