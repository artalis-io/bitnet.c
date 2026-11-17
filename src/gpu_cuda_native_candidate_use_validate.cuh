#ifndef BN_GPU_CUDA_NATIVE_CANDIDATE_USE_VALIDATE_CUH
#define BN_GPU_CUDA_NATIVE_CANDIDATE_USE_VALIDATE_CUH
#include "gpu_cuda_native_candidate_use.h"
namespace bn_cuda_native_candidate_use {
static int run(void*vctx,void*request,const BnGPUIndexCandidateUseResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;BnGPUIndexCandidateUseWorkspace ws;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||
       bn_gpu_index_candidate_use_workspace(p->n_tokens,p->n_keys,p->score_stride,p->pos0,p->rate,&ws)||p->pages<0||p->pages>65535)return -1;
    size_t K=p->n_keys;
    if(p->pages){if((p->entries_per_page!=128&&p->entries_per_page!=256)||p->n_keys>p->pages*p->entries_per_page)return -1;K=(size_t)p->pages*p->entries_per_page;}
    else if(p->entries_per_page||!bn_cuda_index_query::empty(p->block_table))return -1;
    BnGPUResidentView v[]={p->queries,p->weights,p->keys,p->block_table,p->drop,p->entry_ids,p->entry_count,p->gathered_keys,p->scores,p->indices,p->block_counts,p->block_offsets,p->tile_offsets,p->scratch_indices,p->scratch_scores,p->scratch_counts};
    size_t n[]={(size_t)p->n_tokens*4096,(size_t)p->n_tokens*32,K*128,(size_t)p->pages,ws.drop_count,(size_t)ws.capacity,1,ws.gathered_count,ws.scores_count,ws.output_count,(size_t)ws.blocks,(size_t)ws.blocks,(size_t)ws.tiles,ws.select.indices_count,ws.select.scores_count,ws.select.counts_count};
    int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32};void*d[16]={};
    for(int i=0;i<16;i++){if(!n[i]){if(!bn_cuda_index_query::empty(v[i]))return -1;continue;}if(v[i].type!=types[i]||v[i].count!=n[i])return -1;d[i]=bn_cuda_gdn::view_data(r,v[i]);unsigned align=i==4?0u:types[i]==BN_GGUF_TENSOR_F16?((i==0||i==2||i==7)?3u:1u):3u;if(!d[i]||((uintptr_t)d[i]&align))return -1;for(int j=0;j<i;j++)if(n[j]&&bn_cuda_gdn::overlap(v[i],v[j]))return -1;}
    if(bn_cuda_gdn::set_device(r))return -1;
    int status=bn_cuda_native_candidate_gather(r->stream,d[4],d[2],d[3],d[10],d[11],d[12],d[5],d[6],d[7],p->n_tokens,p->n_keys,ws.capacity,p->entries_per_page,ws.tiles)||
        bn_cuda_native_index_score(r->stream,d[0],d[1],d[7],nullptr,d[8],p->n_tokens,ws.capacity,p->score_stride,ws.capacity*p->rate,p->rate,0,d[6])||
        bn_cuda_native_candidate_mask(r->stream,d[4],d[5],d[6],d[8],p->n_tokens,p->n_keys,p->score_stride,p->pos0,p->rate)||
        bn_cuda_native_index_topk(r->stream,d[8],d[9],d[13],d[14],d[15],p->n_tokens,ws.capacity,p->score_stride,512,512,ws.select.split_groups,d[6])||
        bn_cuda_native_candidate_remap(r->stream,d[5],d[6],d[9],p->n_tokens);
    if(status)r->failed=1;return status?-1:0;
}
}
#endif
