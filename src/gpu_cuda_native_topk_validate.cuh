#ifndef BN_GPU_CUDA_NATIVE_TOPK_VALIDATE_CUH
#define BN_GPU_CUDA_NATIVE_TOPK_VALIDATE_CUH
#include "gpu_cuda_native_topk.h"
namespace bn_cuda_native_topk {
static int run(void*vctx,void*request,const BnGPUF16IndexTopKResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    BnGPUF16IndexTopKWorkspace ws;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||
       bn_gpu_index_topk_workspace(p->n_tokens,p->n_keys,p->score_stride,p->k,p->k_pad,&ws))return -1;
    BnGPUResidentView v[]={p->scores,p->indices,p->scratch_indices,p->scratch_scores,p->scratch_counts};
    size_t n[]={(size_t)p->n_tokens*p->score_stride,(size_t)p->n_tokens*p->k_pad,ws.indices_count,ws.scores_count,ws.counts_count};
    int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32};void*d[5]={};
    for(int i=0;i<5;i++) {
        if(!n[i]){if(!bn_cuda_index_query::empty(v[i]))return -1;continue;}
        if(v[i].type!=types[i]||v[i].count!=n[i])return -1;
        d[i]=bn_cuda_gdn::view_data(r,v[i]);if(!d[i]||((uintptr_t)d[i]&((i==0||i==3)?1u:3u)))return -1;
        for(int j=0;j<i;j++)if(n[j]&&bn_cuda_gdn::overlap(v[i],v[j]))return -1;
    }
    if(bn_cuda_gdn::set_device(r))return -1;
    int status=bn_cuda_native_index_topk(r->stream,d[0],d[1],d[2],d[3],d[4],p->n_tokens,p->n_keys,p->score_stride,p->k,p->k_pad,ws.split_groups);
    if(status)r->failed=1;return status;
}
}
#endif
