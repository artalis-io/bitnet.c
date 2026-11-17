#ifndef BN_GPU_CUDA_HC4_VALIDATE_CUH
#define BN_GPU_CUDA_HC4_VALIDATE_CUH
#include "gpu_cuda_hc4.h"
namespace bn_cuda_hc4 {
static bool empty(BnGPUResidentView v){return !v.request&&!v.offset&&!v.count&&!v.slot&&!v.type;}
static bool views(bn_cuda_gdn::ResidentRequest *r,const BnGPUResidentView *v,const size_t *n,int count,void **data){
    for(int i=0;i<count;i++){
        if(v[i].type!=BN_GGUF_TENSOR_F32||v[i].count!=n[i])return false;
        data[i]=bn_cuda_gdn::view_data(r,v[i]);
        if(!data[i]||((uintptr_t)data[i]&15u))return false;
        for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(v[i],v[j]))return false;
    }
    return true;
}
static const float *weight(BnCudaCtx *ctx,void *handle,int rows,int cols){
    const BnCudaBuffer *b=(const BnCudaBuffer*)handle;
    if(!b||b->native_raw_owner!=ctx||!b->data||b->type!=BN_GGUF_TENSOR_F32||b->rows!=rows||b->cols!=cols||b->size!=(size_t)rows*cols*4||((uintptr_t)b->data&15u))return NULL;
    return (const float*)b->data;
}
static int mix(void *vctx,void *request,const BnGPUHC4CarryResidentPlan *p){
    BnCudaCtx *ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800)return -1;
    size_t partials=bn_gpu_hc4_partials_count(p->n_tokens,p->dim);
    if(!partials||!isfinite(p->rms_eps)||p->rms_eps<=0||!isfinite(p->hc_eps)||p->hc_eps<=0||p->sinkhorn_iters<1||p->sinkhorn_iters>256)return -1;
    size_t rows=(size_t)p->n_tokens,dim=(size_t)p->dim;
    if(rows>SIZE_MAX/dim/16)return -1;
    const float *fn=weight(ctx,p->fn,24,4*p->dim),*base=weight(ctx,p->base,1,24),*scale=weight(ctx,p->scale,1,3);
    if(!fn||!base||!scale)return -1;
    BnGPUResidentView v[]={p->streams,p->partials,p->post,p->comb,p->collapsed,p->pre_out,p->pre_in};
    size_t n[]={rows*4*dim,partials,rows*4,rows*16,rows*dim,rows*4,rows*4};void *data[7]={0};
    if(!views(r,v,n,empty(p->pre_in)?6:7,data)||bn_cuda_gdn::set_device(r))return -1;
    int result=bn_cuda_hc4_mix_launch(r->stream,(const float*)data[0],fn,base,scale,(float*)data[1],(float*)data[2],(float*)data[3],(float*)data[4],(const float*)data[6],(float*)data[5],p->n_tokens,p->dim,(int)(partials/rows/25),p->rms_eps,p->hc_eps,p->sinkhorn_iters);
    if(result)r->failed=1;return result;
}
static int apply(void *vctx,void *request,const BnGPUHC4ApplyResidentPlan *p){
    BnCudaCtx *ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||!bn_gpu_hc4_partials_count(p->n_tokens,p->dim))return -1;
    size_t rows=(size_t)p->n_tokens,dim=(size_t)p->dim;
    if(rows>SIZE_MAX/dim/16)return -1;
    BnGPUResidentView v[]={p->streams,p->post,p->comb};size_t n[]={rows*4*dim,rows*4,rows*16};void *data[3];
    if(!views(r,v,n,3,data)||p->y.count!=rows*dim||(p->y.type!=BN_GGUF_TENSOR_F16&&p->y.type!=BN_GGUF_TENSOR_F32))return -1;
    void *y=bn_cuda_gdn::view_data(r,p->y);if(!y||((uintptr_t)y&(p->y.type==BN_GGUF_TENSOR_F16?7u:15u)))return -1;
    for(int i=0;i<3;i++)if(bn_cuda_gdn::overlap(p->y,v[i]))return -1;
    if(bn_cuda_gdn::set_device(r))return -1;
    int result=bn_cuda_hc4_apply_launch(r->stream,(float*)data[0],y,(const float*)data[1],(const float*)data[2],p->n_tokens,p->dim,p->y.type==BN_GGUF_TENSOR_F16);
    if(result)r->failed=1;return result;
}
}
#endif
