#define main glm_indexer_existing_tests_main
#include "test_cuda_glm_indexer.c"
#undef main
#include "cuda_glm_fp8_cache_reference.h"

static size_t cache_cases(BnGPUBackend *g,const char *directory) {
    size_t elements=0;
    for(size_t ci=0;ci<sizeof(glm_fp8_cache_reference)/sizeof(glm_fp8_cache_reference[0]);++ci) {
        BnGLMFP8CacheReference c=glm_fp8_cache_reference[ci];size_t n=(size_t)c.rows,bytes=(size_t)c.blocks*8448;
        int capacity=c.blocks*64;unsigned char *initial=malloc(bytes);int32_t *slots=malloc((n?n:1)*4),*table=malloc((size_t)c.blocks*4);assert(initial&&slots&&table);memset(initial,0xa5,bytes);
        for(int row=0;row<c.rows;++row){int slot=(row*37+c.seed*11)%capacity;if(row%11==3)slot=-1;if(row%17==5)slot=capacity+7;slots[row]=slot;}
        for(int page=0;page<c.blocks;++page)table[page]=c.blocks-1-page;
        BnBackendSession *s=bn_backend_session_create();assert(s);void *r=bn_gpu_backend_resident_request(g,s);assert(r);
        BnGPUResidentView values=alloc(g,r,BN_GGUF_TENSOR_I8,n*128,NULL),scales=alloc(g,r,BN_GGUF_TENSOR_F32,n,NULL);
        if(n){size_t nt=n*4;uint16_t *key=malloc(nt*128*2),*gate=malloc(nt*128*2),tail[1024]={0};float ape[512];assert(key&&gate);
            for(size_t t=0;t<nt;++t)for(int d=0;d<128;++d){key[t*128+d]=pool_input((int)t,d,c.seed,0);gate[t*128+d]=pool_input((int)t,d,c.seed,1);}
            for(int j=0;j<512;++j)ape[j]=(float)((j*19+c.seed*13)%127-63)/32+0x1p-17f;
            void *aw=bn_gpu_backend_create_quant_only_buffer(g,ape,sizeof(ape),BN_GGUF_TENSOR_F32,4,128);assert(aw);
            BnGPUKPoolFP8ResidentPlan p={.n_tokens=(int)nt,.ape=aw,.key=alloc(g,r,BN_GGUF_TENSOR_BF16,nt*128,key),.gate=alloc(g,r,BN_GGUF_TENSOR_BF16,nt*128,gate),.initial_tail=alloc(g,r,BN_GGUF_TENSOR_BF16,1024,tail),.next_tail=alloc(g,r,BN_GGUF_TENSOR_BF16,1024,NULL),.out=values,.scales=scales};
            assert(!g->kpool_fp8_resident(g->ctx,r,&p));assert(!g->resident_synchronize(r));bn_gpu_backend_destroy_buffer(g,aw);free(key);free(gate);
        }
        elements+=check(g,r,values,1,c.values);elements+=check(g,r,scales,4,c.scales);
        BnGPUResidentView cache=alloc(g,r,BN_GGUF_TENSOR_I8,bytes,initial),sv=alloc(g,r,BN_GGUF_TENSOR_I32,n,slots);int32_t status=0;
        BnGPUIndexerFP8AppendResidentPlan append={.n_rows=c.rows,.num_cache_blocks=c.blocks,.values=values,.scales=scales,.slots=sv,.cache=cache};assert(!g->indexer_fp8_append_resident(g->ctx,r,&append));elements+=check(g,r,cache,1,c.cache);
        BnGPUIndexerFP8GatherResidentPlan gather={.n_rows=c.gather_rows,.num_cache_blocks=c.blocks,.logical_blocks=c.blocks,.cache=cache,.block_table=alloc(g,r,BN_GGUF_TENSOR_I32,c.blocks,table),.out=alloc(g,r,BN_GGUF_TENSOR_I8,(size_t)c.gather_rows*128,NULL),.scales=alloc(g,r,BN_GGUF_TENSOR_F32,c.gather_rows,NULL),.status=alloc(g,r,BN_GGUF_TENSOR_I32,1,&status)};
        assert(!g->indexer_fp8_gather_resident(g->ctx,r,&gather));elements+=check(g,r,gather.out,1,c.gather);elements+=check(g,r,gather.scales,4,c.gather_scales);assert(!g->resident_read(r,gather.status,&status)&&!status);
        char prefix[80];snprintf(prefix,sizeof(prefix),"cache-%d-%d",c.rows,c.seed);compare_file(g,r,values,1,directory,prefix,"-values");compare_file(g,r,scales,4,directory,prefix,"-scales");compare_file(g,r,cache,1,directory,prefix,"-cache");compare_file(g,r,gather.out,1,directory,prefix,"-gather");compare_file(g,r,gather.scales,4,directory,prefix,"-gather_scales");
        check(g,r,cache,1,c.cache);bn_backend_session_free(s);free(initial);free(slots);free(table);
    }
    return elements;
}
static void cache_guards(BnGPUBackend *g) {
    BnBackendSession *s=bn_backend_session_create(),*other=bn_backend_session_create();assert(s&&other);void *r=bn_gpu_backend_resident_request(g,s),*peer=bn_gpu_backend_resident_request(g,other);assert(r&&peer);
    unsigned char cache_bytes[8448],values[128];memset(cache_bytes,0xa5,sizeof(cache_bytes));memset(values,0x5a,sizeof(values));float scale=1;int32_t slot=0,table[2]={0,-1},status=0;
    BnGPUIndexerFP8AppendResidentPlan p={.n_rows=1,.num_cache_blocks=1,.values=alloc(g,r,BN_GGUF_TENSOR_I8,128,values),.scales=alloc(g,r,BN_GGUF_TENSOR_F32,1,&scale),.slots=alloc(g,r,BN_GGUF_TENSOR_I32,1,&slot),.cache=alloc(g,r,BN_GGUF_TENSOR_I8,8448,cache_bytes)},bad;
    BnGPUResidentView raw=alloc(g,r,BN_GGUF_TENSOR_I8,8449,NULL),odd=bn_gpu_resident_view_slice(raw,1,8448);
#define REJECT_A(change) do{bad=p;change;assert(g->indexer_fp8_append_resident(g->ctx,r,&bad)==-1);}while(0)
    REJECT_A(bad.n_rows=-1);REJECT_A(bad.n_rows=INT_MAX);REJECT_A(bad.num_cache_blocks=INT_MAX);REJECT_A(bad.cache.count--);REJECT_A(bad.values.count--);REJECT_A(bad.cache=odd);REJECT_A(bad.values=bn_gpu_resident_view_slice(raw,1,128));REJECT_A(bad.values=bn_gpu_resident_view_slice(p.cache,0,128));REJECT_A(bad.cache.request=peer);
#undef REJECT_A
    check(g,r,p.cache,1,hash(cache_bytes,sizeof(cache_bytes)));
    unsigned char output_canary[65*128];uint32_t scale_canary[65];memset(output_canary,0x69,sizeof(output_canary));
    for(int j=0;j<65;++j)scale_canary[j]=UINT32_C(0x12345678);
    BnGPUIndexerFP8GatherResidentPlan q={.n_rows=65,.num_cache_blocks=1,.logical_blocks=2,.cache=p.cache,.block_table=alloc(g,r,BN_GGUF_TENSOR_I32,2,table),.out=alloc(g,r,BN_GGUF_TENSOR_I8,65u*128,output_canary),.scales=alloc(g,r,BN_GGUF_TENSOR_F32,65,scale_canary),.status=alloc(g,r,BN_GGUF_TENSOR_I32,1,&status)},qb;
#define REJECT_G(change) do{qb=q;change;assert(g->indexer_fp8_gather_resident(g->ctx,r,&qb)==-1);}while(0)
    REJECT_G(qb.n_rows=-1);REJECT_G(qb.logical_blocks=1);REJECT_G(qb.num_cache_blocks=0);REJECT_G(qb.out.count--);REJECT_G(qb.scales.count++);REJECT_G(qb.cache=odd);REJECT_G(qb.out=bn_gpu_resident_view_slice(raw,1,65u*128));REJECT_G(qb.out=bn_gpu_resident_view_slice(p.cache,0,65u*128));REJECT_G(qb.status=bn_gpu_resident_view_slice(q.block_table,0,1));REJECT_G(qb.out.request=peer);
#undef REJECT_G
    check(g,r,q.out,1,hash(output_canary,sizeof(output_canary)));check(g,r,q.scales,4,hash(scale_canary,sizeof(scale_canary)));
    assert(!g->resident_read(r,q.status,&status)&&!status);
    assert(!g->indexer_fp8_gather_resident(g->ctx,r,&q));assert(!g->resident_read(r,q.status,&status)&&status==1);
    unsigned char out[65*128];uint32_t bits[65];assert(!g->resident_read(r,q.out,out));assert(!g->resident_read(r,q.scales,bits));
    for(size_t i=0;i<64u*128;++i)assert(out[i]==0xa5);
    for(size_t i=64u*128;i<sizeof(out);++i)assert(!out[i]);
    assert(bits[64]==0);
    table[1]=1;assert(!g->resident_write(r,q.block_table,table));assert(!g->indexer_fp8_gather_resident(g->ctx,r,&q));assert(!g->resident_read(r,q.scales,bits)&&bits[64]==0);
    BnGPUIndexerFP8GatherResidentPlan empty=q;empty.n_rows=0;empty.out=(BnGPUResidentView){0};empty.scales=(BnGPUResidentView){0};assert(!g->indexer_fp8_gather_resident(g->ctx,r,&empty));
    table[1]=0;assert(!g->resident_write(r,q.block_table,table));assert(!g->indexer_fp8_gather_resident(g->ctx,r,&q));assert(!g->resident_read(r,q.status,&status)&&status==1); /* sticky error */
    /* Raw scale bit patterns survive transport, including signed zero/NaN. */
    uint32_t special=UINT32_C(0x7fc12345);assert(!g->resident_write(r,p.scales,&special));assert(!g->indexer_fp8_append_resident(g->ctx,r,&p));assert(!g->indexer_fp8_gather_resident(g->ctx,r,&q));assert(!g->resident_read(r,q.scales,bits)&&bits[0]==special&&bits[64]==special);
    special=UINT32_C(0x80000000);assert(!g->resident_write(r,p.scales,&special));assert(!g->indexer_fp8_append_resident(g->ctx,r,&p));assert(!g->indexer_fp8_gather_resident(g->ctx,r,&q));assert(!g->resident_read(r,q.scales,bits)&&bits[0]==special);
    bn_backend_session_free(other);bn_backend_session_free(s);
}
int main(int argc,char **argv){assert(argc==1||argc==2);BnGPUBackend *g=bn_gpu_cuda_create();assert(g&&g->indexer_fp8_append_resident&&g->indexer_fp8_gather_resident);cache_guards(g);size_t count=cache_cases(g,argc==2?argv[1]:NULL);bn_gpu_cuda_destroy(g);printf("CUDA GLM FP8 paged cache PASSED: %zu reference elements (%s)\n",count,argc==2?"direct bytes and fixture digests":"fixture digests");return 0;}
