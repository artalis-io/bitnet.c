#ifndef BN_GPU_CUDA_NATIVE_INDEX_VALIDATE_CUH
#define BN_GPU_CUDA_NATIVE_INDEX_VALIDATE_CUH
#include "gpu_cuda_native_index.h"
namespace bn_cuda_index_query {
static int empty(BnGPUResidentView v){return !v.request&&!v.count&&!v.offset&&!v.slot&&!v.type;}
static int validate(void*ctx,void*request,const BnGPUF16IndexQueryResidentPlan*p) {
    if(!p)return -1;void*d[4];int type;
    if(bn_cuda_f16_rows::validate_linear(ctx,request,&p->query,d)||bn_cuda_f16_rows::validate_linear(ctx,request,&p->weights,d)||
       bn_cuda_rope::validate(ctx,request,&p->query_rope,d,&type))return -1;
    if(p->query.n_tokens!=p->weights.n_tokens||p->query.n_tokens!=p->query_rope.n_tokens||
       p->query.out.type!=BN_GGUF_TENSOR_F16||p->weights.out.type!=BN_GGUF_TENSOR_F16||
       p->query.n!=p->query_rope.heads*p->query_rope.head_stride||p->weights.n!=p->query_rope.heads||
       !bn_cuda_exl3::same_view(p->query.out,p->query_rope.x)||p->query_rope.norm||
       p->query_rope.dim!=p->query_rope.rotary_dim||p->query_rope.rotate_offset||
       p->query_rope.slice_offset!=p->query_rope.head_stride-p->query_rope.dim||!empty(p->query_rope.positions))return -1;
    BnGPUResidentView v[]={p->query.x,p->weights.x,p->query.out,p->weights.out,p->query_rope.inv_freq};
    for(int i=0;i<5;i++)for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(v[i],v[j]))return -1;return 0;
}
static int run(void*ctx,void*request,const BnGPUF16IndexQueryResidentPlan*p) {
    if(validate(ctx,request,p))return -1;
    return bn_cuda_f16_rows::linear(ctx,request,&p->query)||bn_cuda_f16_rows::linear(ctx,request,&p->weights)||
        bn_cuda_rope::run(ctx,request,&p->query_rope)?-1:0;
}
}
namespace bn_cuda_index_score {
static int validate(void*vctx,void*request,const BnGPUF16IndexScoreResidentPlan*p,void**data) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||
       p->n_tokens<1||p->n_tokens>65535||p->n_keys<1||p->n_keys>1048576||p->pos0<0||p->pos0>INT_MAX-p->n_tokens||
       p->rate<1||p->rate>256||p->score_stride<((p->n_keys+127)/128)*128||p->score_stride%128||
       (size_t)p->n_tokens>SIZE_MAX/(size_t)p->score_stride/2||p->pages<0||p->pages>65535)return -1;
    size_t K=(size_t)p->n_keys;
    if(p->pages){if((p->entries_per_page!=128&&p->entries_per_page!=256)||p->n_keys>p->pages*p->entries_per_page)return -1;K=(size_t)p->pages*p->entries_per_page;}
    else if(p->entries_per_page||!bn_cuda_index_query::empty(p->block_table))return -1;
    BnGPUResidentView v[]={p->queries,p->weights,p->keys,p->block_table,p->scores};
    size_t n[]={(size_t)p->n_tokens*32*128,(size_t)p->n_tokens*32,K*128,(size_t)p->pages,(size_t)p->n_tokens*p->score_stride};
    for(int i=0;i<5;i++) {
        data[i]=NULL;if(i==3&&!p->pages)continue;
        if(v[i].type!=(i==3?BN_GGUF_TENSOR_I32:BN_GGUF_TENSOR_F16)||v[i].count!=n[i])return -1;
        data[i]=bn_cuda_gdn::view_data(r,v[i]);if(!data[i]||((uintptr_t)data[i]&((i==0||i==2||i==3)?3u:1u)))return -1;
        for(int j=0;j<i;j++)if(n[j]&&bn_cuda_gdn::overlap(v[i],v[j]))return -1;
    }
    return bn_cuda_gdn::set_device(r);
}
static int run(void*ctx,void*request,const BnGPUF16IndexScoreResidentPlan*p) {
    void*d[5];if(validate(ctx,request,p,d))return -1;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    int status=bn_cuda_native_index_score(r->stream,d[0],d[1],d[2],d[3],d[4],p->n_tokens,p->n_keys,p->score_stride,p->pos0,p->rate,p->entries_per_page);
    if(status)r->failed=1;return status;
}
}
namespace bn_cuda_index_query_score {
static int run(void*ctx,void*request,const BnGPUF16IndexQueryScoreResidentPlan*p) {
    if(!p||bn_cuda_index_query::validate(ctx,request,&p->query))return -1;void*d[5];
    if(bn_cuda_index_score::validate(ctx,request,&p->score,d)||
       !bn_cuda_exl3::same_view(p->query.query.out,p->score.queries)||!bn_cuda_exl3::same_view(p->query.weights.out,p->score.weights)||
       p->query.query_rope.pos0!=p->score.pos0||p->query.query.n_tokens!=p->score.n_tokens||
       p->query.query_rope.heads!=32||p->query.query_rope.head_stride!=128)return -1;
    BnGPUResidentView extra[]={p->query.query.x,p->query.weights.x,p->query.query_rope.inv_freq};
    BnGPUResidentView other[]={p->score.keys,p->score.block_table,p->score.scores};
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)if(other[j].request&&bn_cuda_gdn::overlap(extra[i],other[j]))return -1;
    if(bn_cuda_index_query::run(ctx,request,&p->query)||bn_cuda_index_score::run(ctx,request,&p->score))return -1;return 0;
}
}
#endif
