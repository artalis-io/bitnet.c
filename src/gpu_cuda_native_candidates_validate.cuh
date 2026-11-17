#ifndef BN_GPU_CUDA_NATIVE_CANDIDATES_VALIDATE_CUH
#define BN_GPU_CUDA_NATIVE_CANDIDATES_VALIDATE_CUH
#include "gpu_cuda_native_candidates.h"
namespace bn_cuda_native_candidates {
static int run(void*vctx,void*request,const BnGPUIndexCandidateSourceResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    BnGPUIndexCandidateWorkspace ws;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||
       bn_gpu_index_candidate_workspace(p->n_tokens,p->n_keys,p->score_stride,p->block_stride,p->pos0,p->rate,&ws))return -1;
    BnGPUResidentView v[]={p->scores,p->block_scores,p->block_indices,p->drop,p->scratch_indices,p->scratch_scores,p->scratch_counts};
    size_t n[]={(size_t)p->n_tokens*p->score_stride,ws.block_scores_count,ws.block_indices_count,ws.drop_count,ws.select.indices_count,ws.select.scores_count,ws.select.counts_count};
    int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32};void*d[7]={};
    for(int i=0;i<7;i++) {
        if(!n[i]){if(!bn_cuda_index_query::empty(v[i]))return -1;continue;}
        if(v[i].type!=types[i]||v[i].count!=n[i])return -1;
        d[i]=bn_cuda_gdn::view_data(r,v[i]);unsigned mask=i==3?0u:(i==0||i==1||i==5)?1u:3u;
        if(!d[i]||((uintptr_t)d[i]&mask))return -1;
        for(int j=0;j<i;j++)if(n[j]&&bn_cuda_gdn::overlap(v[i],v[j]))return -1;
    }
    if(bn_cuda_gdn::set_device(r))return -1;
    int status=bn_cuda_native_candidates_begin(r->stream,d[0],d[1],d[3],p->n_tokens,p->n_keys,p->score_stride,p->block_stride,p->pos0,p->rate)||
        bn_cuda_native_index_topk(r->stream,d[1],d[2],d[4],d[5],d[6],p->n_tokens,ws.blocks,p->block_stride,2048,2048,ws.select.split_groups)||
        bn_cuda_native_candidates_end(r->stream,d[1],d[2],d[3],p->n_tokens,ws.blocks,p->block_stride);
    if(status)r->failed=1;return status?-1:0;
}
}
#endif
