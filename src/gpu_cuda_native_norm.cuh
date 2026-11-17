#ifndef BN_GPU_CUDA_NATIVE_NORM_CUH
#define BN_GPU_CUDA_NATIVE_NORM_CUH
/* Adapted from exllamav3 norm.cu, commit bd1f519172406a9560bd5c90e61000691bb69a07.
 * Copyright (c) 2025 Turboderp. MIT: third_party/exllamav3/LICENSE.
 * FTZ operations are local to this profile; global compiler flags stay intact. */
namespace bn_cuda_native_norm {
static __device__ __forceinline__ float add(float a,float b){float c;asm("add.ftz.f32 %0,%1,%2;":"=f"(c):"f"(a),"f"(b));return c;}
static __device__ __forceinline__ float mul(float a,float b){float c;asm("mul.ftz.f32 %0,%1,%2;":"=f"(c):"f"(a),"f"(b));return c;}
static __device__ __forceinline__ float madd(float a,float b,float c){float d;asm("fma.rn.ftz.f32 %0,%1,%2,%3;":"=f"(d):"f"(a),"f"(b),"f"(c));return d;}
static __device__ __forceinline__ float divide(float a,float b){float c;asm("div.approx.ftz.f32 %0,%1,%2;":"=f"(c):"f"(a),"f"(b));return c;}
static __device__ __forceinline__ float inverse_root(float a){float b;asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(b):"f"(a));return b;}
template<typename T>struct IO;
template<>struct IO<float> {
    static __device__ float4 load(const float *p){return *(const float4*)p;}
    static __device__ void store(float *p,float4 v){*(float4*)p=v;}
    static __device__ float4 input(const float *p){return load(p);}
};
template<>struct IO<half> {
    static __device__ float4 load(const half *p){return make_float4(__half2float(p[0]),__half2float(p[1]),__half2float(p[2]),__half2float(p[3]));}
    static __device__ float4 input(const half *p){float4 v=load(p);v.x=fmaxf(-65504.f,fminf(v.x,65504.f));v.y=fmaxf(-65504.f,fminf(v.y,65504.f));v.z=fmaxf(-65504.f,fminf(v.z,65504.f));v.w=fmaxf(-65504.f,fminf(v.w,65504.f));return v;}
    static __device__ void store(half *p,float4 v){half2 *q=(half2*)p;q[0]=__halves2half2(__float2half_rn(v.x),__float2half_rn(v.y));q[1]=__halves2half2(__float2half_rn(v.z),__float2half_rn(v.w));}
};
template<>struct IO<__nv_bfloat16> {
    static __device__ float4 load(const __nv_bfloat16 *p){return make_float4(__bfloat162float(p[0]),__bfloat162float(p[1]),__bfloat162float(p[2]),__bfloat162float(p[3]));}
};
static __device__ float sum4(float s,float4 v){s=madd(v.x,v.x,s);s=madd(v.y,v.y,s);s=madd(v.z,v.z,s);return madd(v.w,v.w,s);}
static __device__ float reduce(float s){__shared__ float sums[32];int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int d=16;d;d/=2)s=add(s,__shfl_xor_sync(0xffffffff,s,d));
    int warps=blockDim.x/32;if(warps==1)return s;
    if(!lane)sums[warp]=s;__syncthreads();s=lane<warps?sums[lane]:0.f;
    for(int d=16;d;d/=2)s=add(s,__shfl_xor_sync(0xffffffff,s,d));return s;}
template<typename Y,typename W>static __device__ void output(Y *out,float4 x,const W *weight,int col,float rms,float bias){
    if(weight){float4 w=IO<W>::load(weight+(size_t)col*4);if(bias!=0.f){w.x=add(w.x,bias);w.y=add(w.y,bias);w.z=add(w.z,bias);w.w=add(w.w,bias);}x.x=mul(mul(x.x,w.x),rms);x.y=mul(mul(x.y,w.y),rms);x.z=mul(mul(x.z,w.z),rms);x.w=mul(mul(x.w,w.w),rms);}
    else{x.x=mul(x.x,rms);x.y=mul(x.y,rms);x.z=mul(x.z,rms);x.w=mul(x.w,rms);}
    IO<Y>::store(out+(size_t)col*4,x);
}
template<typename X,typename Y,typename W>static __global__ __launch_bounds__(1024)
void kernel(const X *x,Y *out,const W *weight,int rows,int dim,int groups,float eps,float bias,float scale){
    int row=blockIdx.x,t=threadIdx.x,cols=dim/4;size_t offset=(size_t)row*dim;x+=offset;out+=offset;
    if(weight&&groups>1)weight+=(size_t)(row%groups)*dim;
    float sum=0;
    if(cols<=(int)blockDim.x){float4 v=make_float4(0,0,0,0);if(t<cols){v=IO<X>::input(x+(size_t)t*4);sum=sum4(sum,v);}sum=reduce(sum);float rms=mul(inverse_root(add(divide(sum,(float)dim),eps)),scale);if(t<cols)output(out,v,weight,t,rms,bias);}
    else {for(int c=t;c<cols;c+=blockDim.x)sum=sum4(sum,IO<X>::input(x+(size_t)c*4));sum=reduce(sum);float rms=mul(inverse_root(add(divide(sum,(float)dim),eps)),scale);for(int c=t;c<cols;c+=blockDim.x)output(out,IO<X>::input(x+(size_t)c*4),weight,c,rms,bias);}
    (void)rows;
}
static int validate(void *vctx,void *request,const BnGPUVector4RMSNormResidentPlan *p,void **data,int *weight_type){
    BnCudaCtx *ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||p->dim<=0||p->dim%4||p->n_tokens<=0||p->weight_groups<=0||!isfinite(p->eps)||p->eps<=0||!isfinite(p->constant_bias)||!isfinite(p->constant_scale)||
        (size_t)p->n_tokens>SIZE_MAX/(size_t)p->dim/4||
        (p->x.type!=BN_GGUF_TENSOR_F16&&p->x.type!=BN_GGUF_TENSOR_F32)||
        (p->out.type!=BN_GGUF_TENSOR_F16&&p->out.type!=BN_GGUF_TENSOR_F32))return -1;
    size_t count=(size_t)p->n_tokens*p->dim;if(p->x.count!=count||p->out.count!=count)return -1;
    bool same=p->x.request==p->out.request&&p->x.slot==p->out.slot&&p->x.offset==p->out.offset&&p->x.type==p->out.type;
    if(bn_cuda_gdn::overlap(p->x,p->out)&&!same)return -1;
    void *x=bn_cuda_gdn::view_data(r,p->x),*out=bn_cuda_gdn::view_data(r,p->out),*w=NULL;int wt=BN_GGUF_TENSOR_F16;
    if(!x||!out||((uintptr_t)x&(p->x.type==BN_GGUF_TENSOR_F32?15u:7u))||((uintptr_t)out&(p->out.type==BN_GGUF_TENSOR_F32?15u:7u)))return -1;
    if(p->weight){const BnCudaBuffer *b=(const BnCudaBuffer*)p->weight;
        if(b->native_raw_owner!=ctx||!b->data||(b->type!=BN_GGUF_TENSOR_BF16&&b->type!=BN_GGUF_TENSOR_F16)||b->rows!=p->weight_groups||b->cols!=p->dim||
            (size_t)p->weight_groups>SIZE_MAX/(size_t)p->dim/2||b->size!=(size_t)p->weight_groups*p->dim*2||((uintptr_t)b->data&7u))return -1;
        w=b->data;wt=b->type;
    }else if(p->weight_groups!=1)return -1;
    if(bn_cuda_gdn::set_device(r))return -1;
    data[0]=x;data[1]=out;data[2]=w;*weight_type=wt;return 0;
}
static int run(void *vctx,void *request,const BnGPUVector4RMSNormResidentPlan *p){
    void *data[3];int wt;if(validate(vctx,request,p,data,&wt))return -1;
    bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest*)request;
    void *x=data[0],*out=data[1],*w=data[2];
    int threads=((p->dim/4+31)/32)*32;if(threads>1024)threads=1024;
#define BN_NATIVE_NORM_LAUNCH(X,Y,W) kernel<X,Y,W><<<p->n_tokens,threads,0,r->stream>>>((const X*)x,(Y*)out,(const W*)w,p->n_tokens,p->dim,p->weight_groups,p->eps,p->constant_bias,p->constant_scale)
    if(p->x.type==BN_GGUF_TENSOR_F16){if(p->out.type==BN_GGUF_TENSOR_F16){if(wt==BN_GGUF_TENSOR_F16){BN_NATIVE_NORM_LAUNCH(half,half,half);}else{BN_NATIVE_NORM_LAUNCH(half,half,__nv_bfloat16);}}else{if(wt==BN_GGUF_TENSOR_F16){BN_NATIVE_NORM_LAUNCH(half,float,half);}else{BN_NATIVE_NORM_LAUNCH(half,float,__nv_bfloat16);}}}
    else {if(p->out.type==BN_GGUF_TENSOR_F16){if(wt==BN_GGUF_TENSOR_F16){BN_NATIVE_NORM_LAUNCH(float,half,half);}else{BN_NATIVE_NORM_LAUNCH(float,half,__nv_bfloat16);}}else{if(wt==BN_GGUF_TENSOR_F16){BN_NATIVE_NORM_LAUNCH(float,float,half);}else{BN_NATIVE_NORM_LAUNCH(float,float,__nv_bfloat16);}}}
#undef BN_NATIVE_NORM_LAUNCH
    if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
}
}
#endif
