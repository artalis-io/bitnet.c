#include "cuda_native_topk_reference.h"
static uint16_t native_topk_score_word(int row,int col,int kind) {
    unsigned i=(unsigned)col,r=(unsigned)row,idx=i+r*13,u;
    switch(kind) {
    case 0:u=(0x3c00+(idx*17)%1024)|(idx%7==0?0x8000:0);if(idx%19==0)u=0xfc00;if(idx%31==0)u=0;break;
    case 1:u=0;break;
    case 2:u=idx%2?0x8000:0;break;
    case 3:u=0xfc00;break;
    case 4:u=(i*40503+r*2003+7)&65535;break;
    case 5:u=i<7?0x3c00+idx%7:0xfc00;break;
    case 6:u=idx%4096<16?0x4000:idx%3?0x3c00:0x8000;break;
    default:u=(i+r*31)%0x7c00;break;
    }
    return (uint16_t)u;
}
static void native_topk_guard_only(BnGPUBackend*g,void*r,BnGPUResidentView root,size_t n,int type) {
    size_t width=type==BN_GGUF_TENSOR_F16?2:4;unsigned char*b=malloc((n+32)*width);assert(b&&!g->resident_read(r,root,b));
    for(size_t i=0;i<16*width;i++)assert(b[i]==0xa5&&b[(n+16)*width+i]==0xa5);
    free(b);
}
static void run_native_topk_reference(BnGPUBackend*g) {
    assert(g->index_topk_f16_resident);size_t calls=0;
    for(size_t c=0;c<sizeof(native_topk_fixtures)/sizeof(*native_topk_fixtures);c++) {
        int R=native_topk_fixtures[c].R,T=native_topk_fixtures[c].T,stride=native_topk_fixtures[c].stride,k=native_topk_fixtures[c].k,pad=native_topk_fixtures[c].pad,shift=native_topk_fixtures[c].shift;
        size_t sn=(size_t)R*stride;uint16_t*x=malloc((sn+shift)*2);assert(x);for(int j=0;j<shift;j++)x[j]=0xa5a5;
        for(int row=0;row<R;row++)for(int col=0;col<stride;col++)x[shift+(size_t)row*stride+col]=col<T?native_topk_score_word(row,col,native_topk_fixtures[c].kind):0x7fff;
        BnGPUF16IndexTopKWorkspace ws;assert(!bn_gpu_index_topk_workspace(R,T,stride,k,pad,&ws));
        size_t n[]={sn+shift,(size_t)R*pad,ws.indices_count,ws.scores_count,ws.counts_count};int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32};
        void*r[2];BnGPUF16IndexTopKResidentPlan p[2];BnGPUResidentView roots[2][5];
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[5]={{0}};
            for(int j=0;j<5;j++)if(n[j])v[j]=native_router_guard(g,r[q],types[j],n[j],j==0?x:NULL,&roots[q][j]);
            v[0]=bn_gpu_resident_view_slice(v[0],shift,sn);p[q]=(BnGPUF16IndexTopKResidentPlan){R,T,stride,k,pad,v[0],v[1],v[2],v[3],v[4]};BnGPUF16IndexTopKResidentPlan bad;
#define TOPK_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->index_topk_f16_resident(g->ctx,r[q],&bad)==-1);}while(0)
            TOPK_BAD(n_tokens,0);TOPK_BAD(n_keys,-1);TOPK_BAD(score_stride,T-1);TOPK_BAD(k,-1);TOPK_BAD(k,pad+1);TOPK_BAD(scores.count,sn-1);TOPK_BAD(indices.count,n[1]-1);TOPK_BAD(indices.type,BN_GGUF_TENSOR_F16);
            if(ws.split_groups){TOPK_BAD(scratch_counts.count,ws.counts_count-1);TOPK_BAD(indices,bn_gpu_resident_view_slice(p[q].scratch_indices,0,n[1]));}
            else {TOPK_BAD(scratch_counts,p[q].indices);}
#undef TOPK_BAD
            native_router_guard_check(g,r[q],roots[q][1],n[1],types[1],folded_test_unwritten(n[1]*4));
        }
        BnGPUF16IndexTopKResidentPlan bad=p[0];bad.scores=p[1].scores;assert(g->index_topk_f16_resident(g->ctx,r[0],&bad)==-1);
        for(int rep=0;rep<2;rep++)for(int q=0;q<2;q++) {
            assert(!g->index_topk_f16_resident(g->ctx,r[q],p+q));
            native_router_guard_check(g,r[q],roots[q][1],n[1],types[1],native_topk_fixtures[c].hash);
            native_router_guard_check(g,r[q],roots[q][0],n[0],types[0],native_norm_hash(x,n[0]*2));
            for(int j=2;j<5;j++)if(n[j])native_topk_guard_only(g,r[q],roots[q][j],n[j],types[j]);
            calls++;
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}free(x);
    }
    printf("CUDA source F16 index top-k PASSED: %zu calls\n",calls);
}
