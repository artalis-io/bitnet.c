#ifndef BN_GPU_CUDA_NATIVE_ROPE_VALIDATE_CUH
#define BN_GPU_CUDA_NATIVE_ROPE_VALIDATE_CUH
#include "gpu_cuda_native_rope.h"
namespace bn_cuda_frequency {
static int run(void*vctx,void*request,const BnGPUF32RopeFrequencyResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||
       bn_gpu_rope_frequency_shape(p->rotary_dim,p->yarn,p->base,p->factor,p->ramp_start,p->ramp_scale)||
       p->out.type!=BN_GGUF_TENSOR_F32||p->negative.type!=BN_GGUF_TENSOR_F32||
       p->out.count!=(size_t)p->rotary_dim/2||p->negative.count!=p->out.count||bn_cuda_gdn::overlap(p->out,p->negative))return -1;
    float*out=(float*)bn_cuda_gdn::view_data(r,p->out),*negative=(float*)bn_cuda_gdn::view_data(r,p->negative);
    if(!out||!negative||((uintptr_t)out&3u)||((uintptr_t)negative&3u)||bn_cuda_gdn::set_device(r))return -1;
    if(bn_cuda_native_frequency(r->stream,out,negative,p->rotary_dim,p->yarn,p->base,p->factor,p->ramp_start,p->ramp_scale)){r->failed=1;return -1;}
    return 0;
}
}
namespace bn_cuda_rope {
static int validate(void *vctx,void *request,const BnGPUGPTJF16ResidentPlan*p,void **data,int *bf16) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||
       p->n_tokens<1||p->n_tokens>65535||p->heads<1||p->heads>65535||
       p->dim<2||p->dim>2048||p->dim%2||p->head_stride<2||p->head_stride%2||
       p->slice_offset<0||p->slice_offset%2||p->slice_offset>p->head_stride||
       p->dim>p->head_stride-p->slice_offset||p->rotary_dim<2||p->rotary_dim%2||
       p->rotate_offset<0||p->rotate_offset%2||p->rotate_offset>p->dim||
       p->rotary_dim>p->dim-p->rotate_offset||p->pos0<0||p->pos0>INT_MAX-(p->n_tokens-1)||
       !isfinite(p->eps)||p->eps<=0||
       (size_t)p->n_tokens>(size_t)INT_MAX/(size_t)p->heads/(size_t)p->head_stride||
       p->x.type!=BN_GGUF_TENSOR_F16||p->x.count!=(size_t)p->n_tokens*p->heads*p->head_stride||
       p->inv_freq.type!=BN_GGUF_TENSOR_F32||p->inv_freq.count!=(size_t)p->rotary_dim/2||
       bn_cuda_gdn::overlap(p->x,p->inv_freq))return -1;
    data[0]=bn_cuda_gdn::view_data(r,p->x);data[1]=bn_cuda_gdn::view_data(r,p->inv_freq);data[2]=NULL;data[3]=NULL;*bf16=0;
    if(!data[0]||!data[1]||((uintptr_t)data[0]&3u)||((uintptr_t)data[1]&3u))return -1;
    if(p->norm) {
        const BnCudaBuffer*w=(const BnCudaBuffer*)p->norm;
        if(w->native_raw_owner!=ctx||!w->data||w->rows!=1||w->cols!=p->dim||
           (w->type!=BN_GGUF_TENSOR_F16&&w->type!=BN_GGUF_TENSOR_BF16)||
           w->size!=(size_t)p->dim*2||((uintptr_t)w->data&3u))return -1;
        data[2]=w->data;*bf16=w->type==BN_GGUF_TENSOR_BF16;
    }
    if(p->positions.request) {
        if(p->positions.type!=BN_GGUF_TENSOR_I32||p->positions.count!=(size_t)p->n_tokens||
           bn_cuda_gdn::overlap(p->positions,p->x)||bn_cuda_gdn::overlap(p->positions,p->inv_freq))return -1;
        data[3]=bn_cuda_gdn::view_data(r,p->positions);if(!data[3]||((uintptr_t)data[3]&3u))return -1;
    } else if(p->positions.count||p->positions.offset||p->positions.slot||p->positions.type)return -1;
    return bn_cuda_gdn::set_device(r);
}
static int run(void*ctx,void*request,const BnGPUGPTJF16ResidentPlan*p) {
    void*data[4];int bf16;if(validate(ctx,request,p,data,&bf16))return -1;
    bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    int status=bn_cuda_native_rope(r->stream,data[0],data[1],data[2],data[3],p->n_tokens,p->heads,
        p->dim,p->head_stride,p->slice_offset,p->rotary_dim,p->rotate_offset,p->pos0,bf16,p->eps);
    if(status)r->failed=1;return status;
}
}
namespace bn_cuda_qkv {
static int run(void*ctx,void*request,const BnGPUF16LatentQKVResidentPlan*p) {
    if(!p)return -1;
    void*linear[2],*norm[3],*rope[4];int type;
    if(bn_cuda_f16_rows::validate_linear(ctx,request,&p->q_a,linear)||
       bn_cuda_f16_rows::validate_linear(ctx,request,&p->q_b,linear)||
       bn_cuda_f16_rows::validate_linear(ctx,request,&p->kv,linear)||
       bn_cuda_native_norm::validate(ctx,request,&p->q_norm,norm,&type)||
       bn_cuda_rope::validate(ctx,request,&p->q_rope,rope,&type)||
       bn_cuda_rope::validate(ctx,request,&p->kv_rope,rope,&type))return -1;
    if(p->q_a.n_tokens!=p->q_b.n_tokens||p->q_a.n_tokens!=p->kv.n_tokens||
       p->q_a.n_tokens!=p->q_norm.n_tokens||p->q_a.n_tokens!=p->q_rope.n_tokens||
       p->q_a.n_tokens!=p->kv_rope.n_tokens||p->q_a.k!=p->kv.k||
       p->q_a.n!=p->q_norm.dim||p->q_a.n!=p->q_b.k||
       p->q_norm.weight_groups!=1||!p->q_norm.weight||
       p->q_norm.constant_bias!=0.f||p->q_norm.constant_scale!=1.f||
       p->q_a.out.type!=BN_GGUF_TENSOR_F16||p->q_b.out.type!=BN_GGUF_TENSOR_F16||
       p->kv.out.type!=BN_GGUF_TENSOR_F16||p->q_norm.out.type!=BN_GGUF_TENSOR_F16||
       !bn_cuda_exl3::same_view(p->q_a.x,p->kv.x)||
       !bn_cuda_exl3::same_view(p->q_a.out,p->q_norm.x)||
       !bn_cuda_exl3::same_view(p->q_norm.out,p->q_b.x)||
       !bn_cuda_exl3::same_view(p->q_b.out,p->q_rope.x)||
       !bn_cuda_exl3::same_view(p->kv.out,p->kv_rope.x)||
       !bn_cuda_exl3::same_view(p->q_rope.inv_freq,p->kv_rope.inv_freq)||
       !bn_cuda_exl3::same_view(p->q_rope.positions,p->kv_rope.positions)||
       p->q_rope.pos0!=p->kv_rope.pos0||p->q_rope.norm||!p->kv_rope.norm||
       p->q_rope.dim!=p->q_rope.rotary_dim||p->q_rope.rotate_offset!=0||
       p->q_rope.slice_offset!=p->q_rope.head_stride-p->q_rope.dim||
       p->q_rope.head_stride!=p->kv_rope.head_stride||p->kv_rope.dim!=p->kv_rope.head_stride||
       p->kv_rope.slice_offset!=0||p->kv_rope.heads!=1||
       p->kv_rope.rotary_dim!=p->q_rope.rotary_dim||
       p->kv_rope.rotate_offset!=p->kv_rope.dim-p->kv_rope.rotary_dim||
       p->q_norm.eps!=p->kv_rope.eps)return -1;
    const BnGPUResidentView live[]={p->q_a.x,p->q_a.out,p->q_norm.out,p->q_b.out,p->kv.out};
    for(int i=0;i<5;i++) {
        for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(live[i],live[j]))return -1;
        if(bn_cuda_gdn::overlap(live[i],p->q_rope.inv_freq)||
           (p->q_rope.positions.request&&bn_cuda_gdn::overlap(live[i],p->q_rope.positions)))return -1;
    }
    if(bn_cuda_f16_rows::linear(ctx,request,&p->q_a)||
       bn_cuda_native_norm::run(ctx,request,&p->q_norm)||
       bn_cuda_f16_rows::linear(ctx,request,&p->q_b)||
       bn_cuda_f16_rows::linear(ctx,request,&p->kv)||
       bn_cuda_rope::run(ctx,request,&p->q_rope)||bn_cuda_rope::run(ctx,request,&p->kv_rope))return -1;
    return 0;
}
}
#endif
