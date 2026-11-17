#include "cuda_native_candidates_reference.h"
static uint16_t native_candidate_score_word(int row,int col,int kind) {
    unsigned i=(unsigned)col,r=(unsigned)row,z=i+r*13,u;
    switch(kind) {
    case 0:u=(0x3c00+(z*17)%1024)|(z%7==0?0x8000:0);if(z%19==0)u=0xfc00;if(z%31==0)u=0;break;
    case 1:u=0;break;
    case 2:u=z%2?0x8000:0;break;
    case 3:u=0xfc00;break;
    case 4:u=(((i/8+r)%256)>>(i%8))&1?0x8000:0;break;
    case 5:u=i<7?0x3c00+z%7:0xfc00;break;
    case 6:u=z%4096<16?0x7c00:z%3?0x3c00:0x8000;break;
    default:u=(i+r*31)%0x7c00;break;
    }
    return(uint16_t)u;
}
static void run_native_candidates_reference(BnGPUBackend*g) {
    assert(g->index_candidate_source_f16_resident);size_t calls=0;
    for(size_t c=0;c<sizeof(native_candidate_fixtures)/sizeof(*native_candidate_fixtures);c++) {
        int R=native_candidate_fixtures[c].R,T=native_candidate_fixtures[c].T,stride=native_candidate_fixtures[c].stride,bs=native_candidate_fixtures[c].bs,pos=native_candidate_fixtures[c].pos,rate=native_candidate_fixtures[c].rate,shift=native_candidate_fixtures[c].shift;
        BnGPUIndexCandidateWorkspace ws;assert(!bn_gpu_index_candidate_workspace(R,T,stride,bs,pos,rate,&ws));size_t sn=(size_t)R*stride;
        uint16_t*x=malloc((sn+shift)*2);assert(x);for(int j=0;j<shift;j++)x[j]=0xa5a5;
        for(int row=0;row<R;row++)for(int col=0;col<stride;col++)x[shift+(size_t)row*stride+col]=col<T?native_candidate_score_word(row,col,native_candidate_fixtures[c].kind):0x7fff;
        size_t n[]={sn+shift,ws.block_scores_count,ws.block_indices_count,ws.drop_count,ws.select.indices_count,ws.select.scores_count,ws.select.counts_count};int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32};
        void*r[2];BnGPUResidentView roots[2][7];BnGPUIndexCandidateSourceResidentPlan p[2];
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[7]={{0}};
            for(int j=0;j<7;j++)if(n[j])v[j]=native_router_guard(g,r[q],types[j],n[j],j==0?x:NULL,&roots[q][j]);
            v[0]=bn_gpu_resident_view_slice(v[0],shift,sn);p[q]=(BnGPUIndexCandidateSourceResidentPlan){R,T,stride,bs,pos,rate,v[0],v[1],v[2],v[3],v[4],v[5],v[6]};BnGPUIndexCandidateSourceResidentPlan bad;
#define CAND_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->index_candidate_source_f16_resident(g->ctx,r[q],&bad)==-1);}while(0)
            CAND_BAD(n_tokens,0);CAND_BAD(n_keys,16384);CAND_BAD(score_stride,T-1);CAND_BAD(block_stride,ws.blocks-1);CAND_BAD(pos0,INT_MAX);CAND_BAD(rate,0);CAND_BAD(drop.count,ws.drop_count-1);CAND_BAD(drop.type,BN_GGUF_TENSOR_I32);CAND_BAD(block_scores,bn_gpu_resident_view_slice(p[q].scores,0,ws.block_scores_count));
            if(ws.select.split_groups){CAND_BAD(scratch_counts.count,ws.select.counts_count-1);}else{CAND_BAD(scratch_counts,p[q].block_indices);}
#undef CAND_BAD
            native_router_guard_check(g,r[q],roots[q][1],n[1],types[1],folded_test_unwritten(n[1]*2));native_router_guard_check(g,r[q],roots[q][3],n[3],types[3],folded_test_unwritten(n[3]));
        }
        BnGPUIndexCandidateSourceResidentPlan bad=p[0];bad.scores=p[1].scores;assert(g->index_candidate_source_f16_resident(g->ctx,r[0],&bad)==-1);
        for(int rep=0;rep<2;rep++)for(int q=0;q<2;q++) {
            assert(!g->index_candidate_source_f16_resident(g->ctx,r[q],p+q));
            native_router_guard_check(g,r[q],roots[q][1],n[1],types[1],native_candidate_fixtures[c].hashes[0]);native_router_guard_check(g,r[q],roots[q][3],n[3],types[3],native_candidate_fixtures[c].hashes[1]);
            native_router_guard_check(g,r[q],roots[q][0],n[0],types[0],native_norm_hash(x,n[0]*2));
            for(int j=2;j<7;j++){if(j!=3&&n[j])native_topk_guard_only(g,r[q],roots[q][j],n[j],types[j]);}calls++;
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}free(x);
    }
    printf("CUDA source candidate block mask PASSED: %zu calls\n",calls);
}
