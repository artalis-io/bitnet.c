#ifndef BN_GPU_CUDA_COMPRESS_VALIDATE_CUH
#define BN_GPU_CUDA_COMPRESS_VALIDATE_CUH
#include "gpu_cuda_compress.h"
namespace bn_cuda_compressor {
static int empty(BnGPUResidentView v){return !v.request&&!v.count&&!v.offset&&!v.slot&&!v.type;}
static int validate(void*vctx,void*request,const BnGPUCompressF16ResidentPlan*p,void**data) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||
       p->n_tokens<1||p->n_tokens>65535||p->pos0<0||p->pos0>INT_MAX-p->n_tokens||
       p->rate<1||p->rate>256||p->dim<32||p->dim>1024||p->dim%32||
       p->rotary_dim<2||p->rotary_dim%2||p->rotary_dim>p->dim||
       p->ring_rows<p->rate||p->dest_rows<1||p->split_dim<1||p->split_dim>p->dim||
       (p->relative!=0&&p->relative!=1)||(p->relative&&p->split_dim!=p->dim)||
       !isfinite(p->eps)||p->eps<=0||(size_t)p->ring_rows>(size_t)INT_MAX/(size_t)p->dim)return -1;
    int begin=p->pos0/p->rate,end=(p->pos0+p->n_tokens)/p->rate;
    if(p->dest_rows<(p->relative?end-begin:end))return -1;
    size_t R=(size_t)p->n_tokens,D=(size_t)p->dim,B=(size_t)p->ring_rows,C=(size_t)p->dest_rows;
    BnGPUResidentView v[]={p->kv,p->gate,p->ring_kv,p->ring_gate,p->ape,p->norm,p->inv_freq,p->dest_a,p->dest_b};
    size_t counts[]={R*D,R*D,B*D,B*D,(size_t)p->rate*D,D,(size_t)p->rotary_dim/2,C*p->split_dim,C*(D-p->split_dim)};
    int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16};
    for(int i=0;i<9;i++) {
        data[i]=NULL;
        if(!counts[i]){if(!empty(v[i]))return -1;continue;}
        if(counts[i]>SIZE_MAX/4||v[i].type!=types[i]||v[i].count!=counts[i])return -1;
        data[i]=bn_cuda_gdn::view_data(r,v[i]);if(!data[i]||((uintptr_t)data[i]&3u))return -1;
        for(int j=0;j<i;j++)if(counts[j]&&bn_cuda_gdn::overlap(v[i],v[j])&&
            !(i==1&&j==0&&bn_cuda_exl3::same_view(v[i],v[j])))return -1;
    }
    return bn_cuda_gdn::set_device(r);
}
static int run(void*ctx,void*request,const BnGPUCompressF16ResidentPlan*p) {
    void*data[9];if(validate(ctx,request,p,data))return -1;
    bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    int status=bn_cuda_compress(r->stream,data,p->n_tokens,p->pos0,p->rate,p->dim,p->rotary_dim,
        p->ring_rows,p->split_dim,p->relative,p->eps);
    if(status)r->failed=1;return status;
}
static int projected(void*ctx,void*request,const BnGPUF16CompressorResidentPlan*p) {
    if(!p)return -1;
    void*data[9],*linear[2];
    if(validate(ctx,request,&p->compress,data)||bn_cuda_f16_rows::validate_linear(ctx,request,&p->kv,linear)||
       p->kv.n_tokens!=p->compress.n_tokens||p->kv.n!=p->compress.dim||
       p->kv.out.type!=BN_GGUF_TENSOR_F16||!bn_cuda_exl3::same_view(p->kv.out,p->compress.kv))return -1;
    bool gated=p->compress.rate>1;
    if(gated) {
        if(bn_cuda_f16_rows::validate_linear(ctx,request,&p->gate,linear)||
           p->gate.n_tokens!=p->kv.n_tokens||p->gate.k!=p->kv.k||p->gate.n!=p->kv.n||
           p->gate.out.type!=BN_GGUF_TENSOR_F16||!bn_cuda_exl3::same_view(p->kv.x,p->gate.x)||
           !bn_cuda_exl3::same_view(p->gate.out,p->compress.gate)||
           bn_cuda_gdn::overlap(p->kv.out,p->gate.out))return -1;
    } else if(p->gate.weight||p->gate.k||p->gate.n||p->gate.n_tokens||!empty(p->gate.x)||!empty(p->gate.out)||
              !bn_cuda_exl3::same_view(p->compress.kv,p->compress.gate))return -1;
    BnGPUResidentView other[]={p->compress.ring_kv,p->compress.ring_gate,p->compress.ape,p->compress.norm,
        p->compress.inv_freq,p->compress.dest_a,p->compress.dest_b};
    for(int i=0;i<7;i++)if(other[i].request&&bn_cuda_gdn::overlap(p->kv.x,other[i]))return -1;
    bool bc=gated&&p->kv.n_tokens<=32;
    if(bn_cuda_f16_rows::linear_impl(ctx,request,&p->kv,bc)||
       (gated&&bn_cuda_f16_rows::linear_impl(ctx,request,&p->gate,bc)))return -1;
    return run(ctx,request,&p->compress);
}
}
#endif
