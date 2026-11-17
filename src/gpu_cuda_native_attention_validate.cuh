#ifndef BN_GPU_CUDA_NATIVE_ATTENTION_VALIDATE_CUH
#define BN_GPU_CUDA_NATIVE_ATTENTION_VALIDATE_CUH
#include "gpu_cuda_native_attention.h"
namespace bn_cuda_native_attention {
static int run(void*vctx,void*request,const BnGPUF16LatentAttentionResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    BnGPUF16LatentAttentionWorkspace ws;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||
       bn_gpu_latent_attention_workspace(p->n_tokens,p->pool_entries,p->rate,p->dense_pool,p->window,p->pos0,&ws)||
       p->ring_beg<0||p->win_floor<p->ring_beg||p->win_floor>p->pos0||p->ring_rows<0||p->ring_rows>1048576||
       (p->window&&p->win_floor<p->pos0&&p->pos0-p->ring_beg>p->ring_rows)||
       (!p->window&&p->ring_rows)||p->pages<0||p->pages>1048576)return -1;
    size_t capacity=0;
    if(p->pages){if(p->entries_per_page<1||p->entries_per_page>256||256%p->entries_per_page)return -1;
        capacity=(size_t)p->pages*p->entries_per_page;if(capacity>1048576||(size_t)p->pool_entries>capacity)return -1;}
    else if(p->entries_per_page||p->pool_entries)return -1;
    const BnCudaBuffer*sink=(const BnCudaBuffer*)p->sinks;
    if(sink&&(sink->native_raw_owner!=ctx||!sink->data||sink->type!=BN_GGUF_TENSOR_F32||sink->rows!=1||sink->cols!=64||sink->size!=256))return -1;
    BnGPUResidentView v[]={p->queries,p->ring,p->kv,p->pool_c,p->pool_r,p->block_table,p->indices,p->inv_freq,p->out,p->scratch_ml,p->scratch_acc};
    size_t n[]={(size_t)p->n_tokens*64*512,(size_t)p->ring_rows*512,p->window?(size_t)p->n_tokens*512:0,
        capacity*448,capacity*64,(size_t)p->pages,p->dense_pool?0:(size_t)p->n_tokens*512,
        bn_cuda_index_query::empty(p->inv_freq)?0u:32u,(size_t)p->n_tokens*64*512,ws.ml_count,ws.acc_count};
    int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,
        BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32};void*d[11]={};
    for(int i=0;i<11;i++) {
        if(!n[i]){if(!bn_cuda_index_query::empty(v[i]))return -1;continue;}
        if(v[i].type!=types[i]||v[i].count!=n[i])return -1;
        d[i]=bn_cuda_gdn::view_data(r,v[i]);if(!d[i]||((uintptr_t)d[i]&(i==5||i==6||i==7||i>=9?3u:1u)))return -1;
        for(int j=0;j<i;j++)if(n[j]&&bn_cuda_gdn::overlap(v[i],v[j]))return -1;
    }
    if(bn_cuda_gdn::set_device(r))return -1;
    BnCudaNativeLatentAttentionArgs a={(const half*)d[0],(const half*)d[1],(const half*)d[2],(const half*)d[3],(const half*)d[4],
        (const int*)d[5],(const int*)d[6],sink?(const float*)sink->data:NULL,(const float*)d[7],(half*)d[8],(float*)d[9],(float*)d[10],
        p->n_tokens,p->pool_entries,p->rate,p->dense_pool,p->window,p->pos0,p->ring_beg,p->win_floor,p->entries_per_page,ws.splits};
    int status=bn_cuda_native_latent_attention(r->stream,&a);if(status)r->failed=1;return status;
}
}
#endif
