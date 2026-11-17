#include "gpu_backend.h"
#include "gpu_cuda.h"
#include "backend_session.h"
#include "gguf.h"
#include "quant.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

static BnGPUResidentView alloc(BnGPUBackend *g, void *r, size_t count) {
    BnGPUResidentView v;
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_BF16, count, &v));
    return v;
}

static void reserve_guards(BnGPUBackend *g, void *r, BnGPUMoEBankConfig c) {
    BnGPUMoEBankConfig bad;
#define REJECT(change) do { bad = c; change; assert(g->moe_bank_reserve(g->ctx, r, &bad) == -1); } while (0)
    REJECT(bad.n_experts = -1);
    REJECT(bad.n_experts = 513);
    REJECT(bad.k = -1);
    REJECT(bad.k = 33);
    REJECT(bad.n_experts = 2;bad.k = 3);
    REJECT(bad.slots = 0);
    REJECT(bad.slots = 5);
    REJECT(bad.experts_per_slot = 0);
    REJECT(bad.experts_per_slot = 65);
    REJECT(bad.dim--);
    REJECT(bad.hidden_dim--);
    REJECT(bad.host_budget_bytes--);
    REJECT(bad.device_budget_bytes--);
    REJECT(bad.dim = INT_MAX - 255; bad.hidden_dim = INT_MAX - 31;
           bad.slots = 4; bad.experts_per_slot = 64; bad.host_budget_bytes = SIZE_MAX; bad.device_budget_bytes = SIZE_MAX);
#undef REJECT
}

static void submit_guards(BnGPUBackend *g, void *r, BnGPUMoEBankResidentPlan p) {
    BnGPUMoEBankResidentPlan bad;
#define REJECT(change) do { bad = p; change; assert(g->moe_bank_submit(g->ctx, r, &bad) == -1); } while (0)
    REJECT(bad.clamped = -1);
    REJECT(bad.clamped = 2);
    REJECT(bad.clamped = 1;bad.clamp_limit = -1);
    REJECT(bad.clamped = 1;bad.clamp_limit = NAN);
    REJECT(bad.clamped = 1;bad.clamp_limit = INFINITY);
    REJECT(bad.lease.request = NULL);
    REJECT(bad.lease.data = NULL);
    REJECT(bad.lease.generation++);
    REJECT(bad.lease.capacity++);
    REJECT(bad.lease.slot = 4);
    REJECT(bad.lease.gate_type = BN_GGUF_TENSOR_Q4_K);
    REJECT(bad.lease.gate_bytes--);
    REJECT(bad.lease.down_bytes--);
    REJECT(bad.lease.expert_bytes--);
    REJECT(bad.total_tokens = 0);
    REJECT(bad.total_tokens = INT_MAX);
    REJECT(bad.n_experts = 0);
    REJECT(bad.n_experts = 3);
    REJECT(bad.experts = NULL);
    REJECT(bad.x.count--);
    REJECT(bad.gate.count--);
    REJECT(bad.up.count--);
    REJECT(bad.out.count--);
    REJECT(bad.up = bad.gate);
    REJECT(bad.out = bad.x);
    REJECT(bad.out.offset++);
    REJECT(bad.grouped = -1);
    REJECT(bad.grouped = 2);
    REJECT(bad.grouped = 1);
    BnGPUMoEBankItem items[2];
    memcpy(items, p.experts, sizeof(items)); p.experts = items;
    REJECT(items[1].expert = 512);
    memcpy(items, (BnGPUMoEBankItem[]){{511, 0, 1}, {256, 1, 1}}, sizeof(items));
    REJECT(items[1].expert = 511);
    items[1].expert = 256;
    REJECT(items[1].first_row = 0);
    items[1].first_row = 1;
    REJECT(items[1].first_row = 10);
    items[1].first_row = 1;
    REJECT(items[1].n_rows = 2);
    items[1].n_rows = 1;
    BnBackendSession *peer = bn_backend_session_create(); assert(peer);
    void *other = bn_gpu_backend_resident_request(g, peer); assert(other);
    REJECT(bad.out = alloc(g, other, p.out.count));
    assert(g->moe_bank_submit(g->ctx, other, &p) == -1);
    assert(g->moe_bank_cancel(other, &p.lease) == -1);
    bn_backend_session_free(peer);
#undef REJECT
}

static uint64_t prepare_input(BnGPUBackend *g, void *r, BnGPUMoEBankResidentPlan bank) {
    BnGPUMoEBankInputPlan p = {0}, bad;
    p.n_tokens = 1; p.x = alloc(g, r, 256);
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_I32, 10, &p.slots));
    assert(!g->resident_alloc(r, BN_GGUF_TENSOR_I32, 1, &p.status));
    uint16_t x[256]; int32_t slots[10], status = 0;
    for (int i = 0; i < 256; i++) x[i] = 0x3f80;
    for (int i = 0; i < 10; i++) slots[i] = 9 - i;
    assert(!g->resident_write(r, p.x, x) && !g->resident_write(r, p.slots, slots));
    assert(!g->resident_write(r, p.status, &status));
    uint64_t generation = 42;
    bank.prepared_input = 1;
    assert(g->moe_bank_submit(g->ctx, r, &bank) == -1);
    assert(g->moe_bank_prepare_input(g->ctx, r, &p, &generation) == -1 && generation == 42);
    assert(g->moe_bank_input_reserve(g->ctx, r, 0, 6336) == -1);
    assert(g->moe_bank_input_reserve(g->ctx, r, INT_MAX, SIZE_MAX) == -1);
    assert(g->moe_bank_input_reserve(g->ctx, r, 2, 6335) == -1);
    assert(!g->moe_bank_input_reserve(g->ctx, r, 2, 6336));
    assert(!g->moe_bank_input_reserve(g->ctx, r, 2, 6336));
    assert(g->moe_bank_input_reserve(g->ctx, r, 1, 6336) == -1);
#define REJECT(change) do { bad = p; change; assert(g->moe_bank_prepare_input(g->ctx, r, &bad, &generation) == -1 && generation == 42); } while (0)
    REJECT(bad.n_tokens = 0);
    REJECT(bad.n_tokens = 3);
    REJECT(bad.x.count--);
    REJECT(bad.slots.count--);
    REJECT(bad.status.count--);
    REJECT(bad.x.type = BN_GGUF_TENSOR_F32);
    REJECT(bad.status = bad.slots; bad.status.count = 1);
    BnBackendSession *peer = bn_backend_session_create(); assert(peer);
    void *other = bn_gpu_backend_resident_request(g, peer); assert(other);
    REJECT(bad.x = alloc(g, other, 256));
    assert(g->moe_bank_prepare_input(g->ctx, other, &p, &generation) == -1 && generation == 42);
    bn_backend_session_free(peer);
#undef REJECT
    for (int malformed = 0; malformed < 2; malformed++) {
        slots[0] = malformed ? 10 : -1;
        status = 0;
        assert(!g->resident_write(r, p.slots, slots) && !g->resident_write(r, p.status, &status));
        assert(!g->moe_bank_prepare_input(g->ctx, r, &p, &generation));
        assert(!g->resident_read(r, p.status, &status) && status == 2);
    }
    slots[0] = 9; status = 0;
    assert(!g->resident_write(r, p.slots, slots) && !g->resident_write(r, p.status, &status));
    assert(!g->moe_bank_prepare_input(g->ctx, r, &p, &generation));
    bank.prepared_input = generation;
    assert(!g->moe_bank_prepare_input(g->ctx, r, &p, &generation));
    assert(g->moe_bank_submit(g->ctx, r, &bank) == -1); /* stale generation */
    bank.prepared_input = generation; bank.total_tokens = 2;
    assert(g->moe_bank_submit(g->ctx, r, &bank) == -1); /* different batch geometry */
    assert(!g->resident_read(r, p.status, &status) && !status);
    BnGPUMoEBankStats stats;
    assert(!g->moe_bank_stats(r, &stats) && stats.reserved_input_bytes == 6336 && stats.input_preparations == 4);
    assert(!stats.upload_bytes && !stats.submitted_banks);
    return generation;
}

static void matrix_tails(BnGPUBackend *g) {
    const int widths[] = {32, 96, 160};
    int cases = 0;
    for (int wi = 0; wi < 3; wi++) for (int gt = 0; gt < 2; gt++) for (int dt = 0; dt < 2; dt++) {
        int hidden = widths[wi], nt = 65, n = nt * 10;
        BnBackendSession *s = bn_backend_session_create(); assert(s);
        void *r = bn_gpu_backend_resident_request(g, s); assert(r);
        size_t raw_payload = (size_t)hidden * 1248; /* Q5_K/Q8_0, two experts. */
        BnGPUMoEBankConfig c = {1, 2, 256, hidden, raw_payload, raw_payload,0,0};
        assert(!g->moe_bank_reserve(g->ctx, r, &c));
        assert(!g->moe_bank_input_reserve(g->ctx, r, nt, (size_t)nt * 256 * 22));
        size_t group_bytes = (size_t)n * (hidden / 32) * 36 + 24;
        assert(!g->moe_bank_grouped_reserve(g->ctx, r, nt, 24, group_bytes));
        assert(!g->linear_a8_bf16_reserve(g->ctx, r, 5 * 256));
        BnGPUMoEBankInputPlan input = {0};
        input.n_tokens = nt; input.x = alloc(g, r, (size_t)nt * 256);
        assert(!g->resident_alloc(r, BN_GGUF_TENSOR_I32, n, &input.slots));
        assert(!g->resident_alloc(r, BN_GGUF_TENSOR_I32, 1, &input.status));
        uint16_t *data = malloc((size_t)n * 256 * 2);
        int32_t *slots = malloc((size_t)n * 4), status = 0;
        assert(data && slots);
        for (int i = 0; i < n * 256; i++) data[i] = 0x3f80;
        for (int i = 0; i < n; i++) slots[i] = n - 1 - i;
        assert(!g->resident_write(r, input.x, data) && !g->resident_write(r, input.slots, slots));
        assert(!g->resident_write(r, input.status, &status));
        uint64_t generation;
        assert(!g->moe_bank_prepare_input(g->ctx, r, &input, &generation));
        BnGPUMoEBankLease lease;
        assert(!g->moe_bank_acquire(r, 0, gt ? BN_GGUF_TENSOR_Q5_K : BN_GGUF_TENSOR_Q4_K,
            dt ? BN_GGUF_TENSOR_Q8_0 : BN_GGUF_TENSOR_Q5_1, &lease));
        memset(lease.data, 0, lease.expert_bytes * 2);
        BnGPUMoEBankItem items[] = {{511, 3, 1}, {256, 17, 5}};
        BnGPUMoEBankResidentPlan p = {0};
        p.lease = lease; p.total_tokens = nt; p.n_experts = 2; p.experts = items;
        p.prepared_input = generation; p.grouped = 1;
        p.x = alloc(g, r, (size_t)n * 256); p.out = alloc(g, r, (size_t)n * 256);
        p.gate = alloc(g, r, (size_t)n * hidden); p.up = alloc(g, r, (size_t)n * hidden);
        assert(!g->resident_write(r, p.x, data));
        for (int i = 0; i < n * 256; i++) data[i] = 0x5a5a;
        assert(!g->resident_write(r, p.out, data) && !g->resident_write(r, p.gate, data) && !g->resident_write(r, p.up, data));
        assert(!g->moe_bank_submit(g->ctx, r, &p));
        BnGPUResidentView views[] = {p.out, p.gate, p.up};
        for (int v = 0; v < 3; v++) {
            int cols = v ? hidden : 256;
            assert(!g->resident_read(r, views[v], data));
            for (int row = 0; row < n; row++) for (int col = 0; col < cols; col++) {
                int active = row == 3 || (row >= 17 && row < 22);
                assert(data[(size_t)row * cols + col] == (active ? 0 : 0x5a5a));
            }
        }
        BnGPUMoEBankStats stats;
        assert(!g->moe_bank_stats(r, &stats) && stats.grouped_banks == 1 && stats.metadata_upload_bytes == 24);
        assert(stats.reserved_group_host_bytes == 24 && stats.reserved_group_device_bytes == group_bytes);
        free(data); free(slots); bn_backend_session_free(s); cases++;
    }
    printf("CUDA MoE grouped tails PASSED: %d cases\n", cases);
}

static void matrix_payload(void *dst,size_t bytes,int type,int seed) {
    const BnQuantFormatOps *f=bn_quant_format_ops(type);assert(f&&bytes%f->bytes_per_block==0);
    memset(dst,0x11+seed%7,bytes);
    for(size_t i=0;i<bytes;i+=f->bytes_per_block) {
        uint16_t d=0x1000,m=type==BN_GGUF_TENSOR_Q5_1?0x8400:0x0400;
        if(type==BN_GGUF_TENSOR_Q6_K){memcpy((unsigned char*)dst+i+f->bytes_per_block-2,&d,2);continue;}
        memcpy((unsigned char*)dst+i,&d,2);
        if(type!=BN_GGUF_TENSOR_Q8_0)memcpy((unsigned char*)dst+i+2,&m,2);
    }
}
/* Nonzero differential against the retained per-expert DP4A path. Uneven
 * groups cross MMA token tiles; inactive output rows retain sentinels. */
static void matrix_nonzero(BnGPUBackend *g) {
    const int widths[]={32,96,160,640,256,512},counts[]={1,7,8,9,23,24,25,63,64,65};int cases=0;
    for(int wi=0;wi<6;wi++)for(int gt=0;gt<2;gt++)for(int dt=0;dt<4;dt++)for(int ni=0;ni<13;ni++) {
        int hidden=widths[wi],nt=ni==12?1:ni==11?257:65,n=nt*10,count=ni==12?1:ni<10?counts[ni]:0;
        if(dt>=2&&hidden%256)continue;
        int experts=ni<10||ni==12?2:8;
        int gate=gt?BN_GGUF_TENSOR_Q5_K:BN_GGUF_TENSOR_Q4_K,down=dt==3?BN_GGUF_TENSOR_Q6_K:dt==2?BN_GGUF_TENSOR_Q5_K:dt?BN_GGUF_TENSOR_Q8_0:BN_GGUF_TENSOR_Q5_1;
        BnBackendSession *s=bn_backend_session_create();assert(s);void *r=bn_gpu_backend_resident_request(g,s);assert(r);
        size_t raw=(size_t)hidden*624*experts,metadata=(size_t)experts*sizeof(BnGPUMoEBankItem);
        BnGPUMoEBankConfig cfg={1,(unsigned)experts,256,hidden,raw,raw,0,0};assert(!g->moe_bank_reserve(g->ctx,r,&cfg));
        assert(!g->moe_bank_input_reserve(g->ctx,r,nt,(size_t)nt*256*22));
        assert(!g->moe_bank_grouped_reserve(g->ctx,r,nt,metadata,(size_t)n*(hidden/32)*36+metadata));
        assert(!g->linear_a8_bf16_reserve(g->ctx,r,(size_t)nt*(hidden>256?hidden:256)));
        size_t cap=(size_t)n*(hidden>256?hidden:256);
        uint16_t *data=malloc(cap*2),*expected[3];int32_t *slots=malloc((size_t)n*4),zero=0;assert(data&&slots);
        BnGPUMoEBankInputPlan input={0};input.n_tokens=nt;input.x=alloc(g,r,(size_t)nt*256);
        assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,n,&input.slots));
        assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,1,&input.status));
        for(int i=0;i<nt*256;i++)data[i]=(uint16_t)((i%5?0x3f80:0xbf80)+(i%13));
        for(int i=0;i<n;i++)slots[i]=n-1-i;
        assert(!g->resident_write(r,input.x,data)&&!g->resident_write(r,input.slots,slots)&&!g->resident_write(r,input.status,&zero));
        uint64_t generation;assert(!g->moe_bank_prepare_input(g->ctx,r,&input,&generation));
        BnGPUMoEBankItem items[8]={{511,3,count},{256,nt*2+3,1}};
        if(experts==8) {
            const int lengths[2][8]={{1,65,2,8,7,64,25,24},{257,1,193,64,129,7,128,65}};
            int first=3;
            for(int i=0;i<8;i++) {
                int entry=7-i;
                items[entry]=(BnGPUMoEBankItem){511-i,first,lengths[ni-10][i]};
                first+=items[entry].n_rows+3;
            }
            assert(first<n);
        }
        BnGPUMoEBankResidentPlan p={0};p.total_tokens=nt;p.n_experts=experts;p.experts=items;p.prepared_input=generation;
        p.x=alloc(g,r,(size_t)n*256);p.gate=alloc(g,r,(size_t)n*hidden);p.up=alloc(g,r,(size_t)n*hidden);p.out=alloc(g,r,(size_t)n*256);
        BnGPUResidentView views[]={p.out,p.gate,p.up};
        for(int pass=0;pass<2;pass++) {
            for(size_t i=0;i<cap;i++)data[i]=0x5a5a;
            for(int v=0;v<3;v++)assert(!g->resident_write(r,views[v],data));
            assert(!g->moe_bank_acquire(r,0,gate,down,&p.lease));
            for(int e=0;e<experts;e++) {
                unsigned char *w=(unsigned char*)p.lease.data+(size_t)e*p.lease.expert_bytes;
                matrix_payload(w,p.lease.gate_bytes,gate,e);
                matrix_payload(w+p.lease.gate_bytes,p.lease.gate_bytes,gate,e+1);
                matrix_payload(w+p.lease.gate_bytes*2,p.lease.down_bytes,down,e+2);
            }
            p.grouped=pass==0;assert(!g->moe_bank_submit(g->ctx,r,&p));
            for(int v=0;v<3;v++) {
                int cols=v?hidden:256,nonzero=0;size_t bytes=(size_t)n*cols*2;
                assert(!g->resident_read(r,views[v],data));
                for(int row=0;row<n;row++)for(int col=0;col<cols;col++) {
                    uint16_t value=data[(size_t)row*cols+col];
                    int active=0;
                    for(int e=0;e<experts;e++)
                        active|=row>=items[e].first_row&&row<items[e].first_row+items[e].n_rows;
                    if(active){assert((value&0x7f80)!=0x7f80);nonzero|=value&0x7fff;}
                    else assert(value==0x5a5a);
                }
                assert(nonzero);
                if(!pass){expected[v]=malloc(bytes);assert(expected[v]);memcpy(expected[v],data,bytes);}
                else{assert(!memcmp(expected[v],data,bytes));free(expected[v]);}
            }
        }
        free(data);free(slots);bn_backend_session_free(s);cases++;
    }
    printf("CUDA grouped MMA nonzero differential PASSED: %d format/width/tile-tail cases\n",cases);
}
static void grouping_profiles(BnGPUBackend*g) {
    int experts[]={1,288,512},ks[]={1,8,32};
    for(int c=0;c<3;++c){int e=experts[c],k=ks[c],nt=2,n=nt*k;
        BnBackendSession*session=bn_backend_session_create();assert(session);void*r=bn_gpu_backend_resident_request(g,session);assert(r);
        int32_t*ids=malloc((size_t)n*4),*offsets=malloc((size_t)(e+1)*4),*slots=malloc((size_t)n*4),*inverse=malloc((size_t)n*4);uint16_t*x=malloc(2*256*2),*gathered=malloc((size_t)n*256*2),*before=malloc((size_t)n*256*2);assert(ids&&offsets&&slots&&inverse&&x&&gathered&&before);
        for(int j=0;j<n;++j)ids[j]=(j*37)%e;
        ids[0]=e-1;
        for(int j=0;j<512;++j)x[j]=j<256?0x3f80:0x4000;
        BnGPUMoEGroupBF16ResidentPlan p={.n_tokens=nt,.n_experts=e,.k=k,.dim=256};
        p.x=alloc(g,r,512);p.gathered=alloc(g,r,(size_t)n*256);assert(!g->resident_write(r,p.x,x));
        assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,n,&p.indices));assert(!g->resident_write(r,p.indices,ids));
        assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,e+1,&p.offsets));assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,e,&p.cursors));
        assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,n,&p.slots));assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,n,&p.inverse));assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,1,&p.status));
        assert(!g->moe_group_bf16_resident(g->ctx,r,&p));int status=-1;assert(!g->resident_read(r,p.status,&status)&&!status);assert(!g->resident_read(r,p.offsets,offsets));assert(!offsets[0]&&offsets[e]==n);
        assert(!g->resident_read(r,p.slots,slots));assert(!g->resident_read(r,p.inverse,inverse));assert(!g->resident_read(r,p.gathered,gathered));
        for(int j=0;j<e;++j)for(int row=offsets[j];row<offsets[j+1];++row){assert(slots[row]>=0&&slots[row]<n&&ids[slots[row]]==j&&inverse[slots[row]]==row);assert(!memcmp(gathered+(size_t)row*256,x+(slots[row]/k)*256,512));}
        float*w=malloc((size_t)n*4);uint16_t y[512];assert(w);for(int j=0;j<n;++j)w[j]=1;
        BnGPUResidentView weights,weighted=alloc(g,r,(size_t)n*256),out=alloc(g,r,512);assert(!g->resident_alloc(r,BN_GGUF_TENSOR_F32,n,&weights));assert(!g->resident_write(r,weights,w));
        BnGPUMoEReduceBF16ResidentPlan red={.n_tokens=nt,.k=k,.dim=256,.grouped=p.gathered,.inverse=p.inverse,.weights=weights,.weighted=weighted,.out=out,.status=p.status};assert(!g->moe_reduce_bf16_resident(g->ctx,r,&red));assert(!g->resident_read(r,out,y));
        uint16_t bits=(uint16_t)(0x3f80+(c==0?0:c==1?3:5)*128);for(int j=0;j<256;++j){assert(y[j]==bits);assert(y[256+j]==bits+128);}free(w);
        memcpy(before,gathered,(size_t)n*256*2);ids[0]=e;assert(!g->resident_write(r,p.indices,ids));assert(!g->moe_group_bf16_resident(g->ctx,r,&p));assert(!g->resident_read(r,p.status,&status)&&status==1);assert(!g->resident_read(r,p.gathered,gathered));assert(!memcmp(before,gathered,(size_t)n*256*2));
        ids[0]=-1;assert(!g->resident_write(r,p.indices,ids));assert(!g->moe_group_bf16_resident(g->ctx,r,&p));assert(!g->resident_read(r,p.status,&status)&&status==1);
        free(ids);free(offsets);free(slots);free(inverse);free(x);free(gathered);free(before);bn_backend_session_free(session);
    }puts("CUDA expert grouping boundary geometries and invalid-ID preservation PASSED");
}
#include "cuda_moe_shared_bank_checks.h"

int main(int argc,char **argv) {
    BnGPUBackend *g = bn_gpu_cuda_create(); assert(g);
    if (!g->moe_bank_reserve) {
        puts("CUDA MoE bank skipped: optional precision profile unavailable");
        bn_gpu_cuda_destroy(g); return 0;
    }
    if(argc==2&&!strcmp(argv[1],"--shared-only")) {
        shared_banks(g);bn_gpu_cuda_destroy(g);return 0;
    }
    assert(argc==1);
    shared_banks(g);
    for (int teardown_pending = 0; teardown_pending < 2; teardown_pending++) {
        BnBackendSession *s = bn_backend_session_create(); assert(s);
        void *r = bn_gpu_backend_resident_request(g, s); assert(r);
        /* Q5_K gate/up: 5632 each, Q8_0 down: 8704, two slots*two experts. */
        BnGPUMoEBankConfig c = {2, 2, 256, 32, 79872, 79872,0,0};
        BnGPUMoEBankStats stats;
        BnGPUMoEBankLease lease, sentinel;
        memset(&sentinel, 0x5a, sizeof(sentinel)); lease = sentinel;
        assert(g->moe_bank_stats(r, &stats) == -1);
        assert(g->moe_bank_acquire(r, 0, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0, &lease) == -1);
        assert(!memcmp(&lease, &sentinel, sizeof(lease)));
        reserve_guards(g, r, c);
        assert(!g->moe_bank_reserve(g->ctx, r, &c) && !g->moe_bank_reserve(g->ctx, r, &c));
        BnGPUMoEBankConfig resize = c; resize.experts_per_slot = 1;
        assert(g->moe_bank_reserve(g->ctx, r, &resize) == -1);
        assert(!g->moe_bank_stats(r, &stats));
        assert(stats.reserved_host_bytes == 79872 && stats.reserved_device_bytes == 79872);
        assert(!stats.upload_bytes && !stats.submitted_banks && !stats.submitted_experts && !stats.busy_acquires);
        assert(g->moe_bank_acquire(r, 2, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0, &lease) == -1);
        assert(g->moe_bank_acquire(r, 0, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0, &lease) == -1);
        assert(g->moe_bank_acquire(r, 0, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K, &lease) == -1);
        assert(!memcmp(&lease, &sentinel, sizeof(lease)));
        assert(!g->moe_bank_acquire(r, 0, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0, &lease));
        BnGPUMoEBankLease stale = lease;
        assert(!g->moe_bank_cancel(r, &lease) && g->moe_bank_cancel(r, &lease) == -1);
        assert(!g->moe_bank_acquire(r, 0, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0, &lease));
        assert(lease.generation == stale.generation + 1 && g->moe_bank_cancel(r, &stale) == -1);
        BnGPUMoEBankLease duplicate = sentinel;
        assert(g->moe_bank_acquire(r, 0, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0, &duplicate) == -1);
        assert(!memcmp(&duplicate, &sentinel, sizeof(duplicate)));
        assert(lease.capacity == 2 && lease.gate_bytes == 5632 && lease.down_bytes == 8704 && lease.expert_bytes == 19968);
        memset(lease.data, 0, lease.expert_bytes * 2);
        BnGPUMoEBankItem items[] = {{511, 0, 1}, {256, 1, 1}};
        BnGPUMoEBankResidentPlan p = {0};
        p.lease = lease; p.total_tokens = 1; p.n_experts = 2; p.experts = items;
        p.x = alloc(g, r, 10 * 256); p.gate = alloc(g, r, 10 * 32);
        p.up = alloc(g, r, 10 * 32); p.out = alloc(g, r, 10 * 256);
        uint16_t data[2560];
        for (int i = 0; i < 2560; i++) data[i] = 0x3f80;
        assert(!g->resident_write(r, p.x, data));
        for (int i = 0; i < 2560; i++) data[i] = 0x5a5a;
        assert(!g->resident_write(r, p.out, data) && !g->resident_write(r, p.gate, data) && !g->resident_write(r, p.up, data));
        /* Missing scratch rejects the entire bank without consuming the lease. */
        assert(g->moe_bank_submit(g->ctx, r, &p) == -1);
        assert(!g->linear_a8_bf16_reserve(g->ctx, r, 256));
        submit_guards(g, r, p);
        if (teardown_pending) {
            p.prepared_input = prepare_input(g, r, p);
            p.grouped = 1;
            assert(g->moe_bank_submit(g->ctx, r, &p) == -1);
            assert(g->moe_bank_grouped_reserve(g->ctx, r, 0, 48, 768) == -1);
            assert(g->moe_bank_grouped_reserve(g->ctx, r, INT_MAX, SIZE_MAX, SIZE_MAX) == -1);
            assert(g->moe_bank_grouped_reserve(g->ctx, r, 2, 47, 768) == -1);
            assert(g->moe_bank_grouped_reserve(g->ctx, r, 2, 48, 767) == -1);
            assert(!g->moe_bank_grouped_reserve(g->ctx, r, 2, 48, 768));
            assert(!g->moe_bank_grouped_reserve(g->ctx, r, 2, 48, 768));
            assert(g->moe_bank_grouped_reserve(g->ctx, r, 1, 48, 768) == -1);
        }
        assert(!g->resident_read(r, p.out, data));
        for (int i = 0; i < 2560; i++) assert(data[i] == 0x5a5a);
        assert(!g->resident_read(r, p.gate, data));
        for (int i = 0; i < 320; i++) assert(data[i] == 0x5a5a);
        assert(!g->resident_read(r, p.up, data));
        for (int i = 0; i < 320; i++) assert(data[i] == 0x5a5a);
        assert(!g->moe_bank_stats(r, &stats) && !stats.upload_bytes && !stats.submitted_banks);
        assert(!g->moe_bank_submit(g->ctx, r, &p));
        assert(g->moe_bank_cancel(r, &lease) == -1 && g->moe_bank_submit(g->ctx, r, &p) == -1);
        size_t bytes = 39936;
        /* Exercise two slots, one-expert tails, changed format pairs and reuse. */
        for (int pass = 0; pass < 6; pass++) {
            int gt = pass & 1 ? BN_GGUF_TENSOR_Q5_K : BN_GGUF_TENSOR_Q4_K;
            int dt = pass & 2 ? BN_GGUF_TENSOR_Q8_0 : BN_GGUF_TENSOR_Q5_1;
            duplicate = sentinel;
            int result;
            do {
                result = g->moe_bank_acquire(r, (unsigned)pass % 2, gt, dt, &duplicate);
                if (result == BN_GPU_MOE_BANK_BUSY) {
                    assert(!memcmp(&duplicate, &sentinel, sizeof(duplicate)));
                    thrd_yield();
                }
            } while (result == BN_GPU_MOE_BANK_BUSY);
            assert(!result);
            assert(g->moe_bank_cancel(r, &lease) == -1);
            memset(duplicate.data, 0, duplicate.expert_bytes);
            p.lease = duplicate; p.n_experts = 1;
            assert(!g->moe_bank_submit(g->ctx, r, &p));
            bytes += duplicate.expert_bytes;
        }
        assert(!g->moe_bank_stats(r, &stats));
        assert(stats.upload_bytes == bytes && stats.submitted_banks == 7 && stats.submitted_experts == 8);
        assert(stats.grouped_banks == (teardown_pending ? 7u : 0u));
        assert(stats.metadata_upload_bytes == (teardown_pending ? 96u : 0u));
        assert(stats.reserved_group_host_bytes == (teardown_pending ? 48u : 0u));
        assert(stats.reserved_group_device_bytes == (teardown_pending ? 768u : 0u));
        if (!teardown_pending) {
            assert(!g->resident_read(r, p.out, data));
            for (int i = 0; i < 2560; i++) assert(data[i] == (i < 512 ? 0 : 0x5a5a));
            assert(!g->moe_bank_acquire(r, 0, BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_1, &duplicate));
            assert(!g->moe_bank_cancel(r, &duplicate));
        }
        /* Second pass deliberately frees a request with work still queued. */
        bn_backend_session_free(s);
    }
    grouping_profiles(g);
    matrix_tails(g);
    matrix_nonzero(g);
    bn_gpu_cuda_destroy(g);
    puts("CUDA MoE bank lifecycle PASSED");
    return 0;
}
