#ifndef BN_GPU_CUDA_NATIVE_PLE_CUH
#define BN_GPU_CUDA_NATIVE_PLE_CUH
/* Numerical sequence adapted from exllamav3 ple.cu, commit bd1f519172406a9560bd5c90e61000691bb69a07.
 * Copyright (c) 2025 Turboderp. MIT: third_party/exllamav3/LICENSE. */
namespace bn_cuda_native_ple {
static __global__ void cast_lookup(const __nv_bfloat16 *x,half *y,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)y[i]=__float2half_rn(__bfloat162float(x[i]));
}
static __global__ void gate_value(const float *dots,const half *value,float *out,
    size_t count,int streams,int dim,float scale) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count/4)return;
    size_t offset=i*4,row=offset/(size_t)dim,col=offset%(size_t)dim;
    float g=bn_cuda_native_norm::mul(dots[row],scale);
    float magnitude=fmaxf(fabsf(g),1e-6f),a;
    asm("sqrt.approx.f32 %0,%1;":"=f"(a):"f"(magnitude));
    float signed_root=g>0.f?a:(g<0.f?-a:0.f);
    float probability=__fdividef(1.f,bn_cuda_native_norm::add(1.f,__expf(-signed_root)));
    const half *v=value+(row/(size_t)streams)*(size_t)dim+col;
    float4 o=make_float4(bn_cuda_native_norm::mul(probability,__half2float(v[0])),
        bn_cuda_native_norm::mul(probability,__half2float(v[1])),bn_cuda_native_norm::mul(probability,__half2float(v[2])),
        bn_cuda_native_norm::mul(probability,__half2float(v[3])));
    *(float4*)(out+offset)=o;
}
static int run(void *vctx,void *request,const BnGPUPLEF16FromBF16ResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)vctx;
    bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||
       p->n_tokens<1||p->n_tokens>16384||p->k<1||p->k>32768||p->dim<4||p->dim>32768||p->dim%4||
       p->streams<1||p->streams>8||!isfinite(p->eps)||p->eps<=0||!isfinite(p->gate_scale)||p->gate_scale<=0||
       !p->weight||!p->norm_key||!p->norm_query||!r->native_f16_cublas||!r->native_f16_workspace||
       !r->trellis_cublas||!r->trellis_workspace)return -1;
    int hd=p->streams*p->dim,width=hd+p->dim;
    size_t rows=(size_t)p->n_tokens,wide=rows*(size_t)hd;
    if(wide>INT_MAX||(size_t)p->k>SIZE_MAX/(size_t)width/2||rows>SIZE_MAX/(size_t)p->k/2||
       rows>SIZE_MAX/(size_t)hd/4)return -1;
    const BnCudaBuffer *w=(const BnCudaBuffer*)p->weight;
    if(w->native_f16_owner!=ctx||w->type!=BN_GGUF_TENSOR_F16||!w->data||
       w->rows!=width||w->cols!=p->k||w->size!=(size_t)width*p->k*2)return -1;
    BnGPUResidentView views[]={p->lookup,p->embedding,p->stream,p->key,p->value,
        p->normalized_key,p->normalized_query,p->dot,p->delta,p->out};
    int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F16,
        BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,
        BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32};
    size_t counts[]={rows*(size_t)p->k,rows*(size_t)p->k,wide,wide,rows*(size_t)p->dim,
        wide,wide,rows*(size_t)p->streams,wide,wide};
    void *data[10];
    for(int i=0;i<10;i++) {
        if(views[i].type!=types[i]||views[i].count!=counts[i]||!(data[i]=bn_cuda_gdn::view_data(r,views[i])))return -1;
        unsigned alignment=(i==0||i==1)?1u:(i==3||i==4)?7u:i==7?3u:15u;
        if((uintptr_t)data[i]&alignment)return -1;
        for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(views[i],views[j]))return -1;
    }
    BnGPUVector4RMSNormResidentPlan key_norm={p->n_tokens*p->streams,p->dim,p->streams,p->eps,0.f,1.f,p->norm_key,p->key,p->normalized_key};
    BnGPUVector4RMSNormResidentPlan query_norm={p->n_tokens*p->streams,p->dim,p->streams,p->eps,0.f,1.f,p->norm_query,p->stream,p->normalized_query};
    void *norm_data[3];int wt;
    /* Preflight BOTH weights/norm plans before any queued cast or projection. */
    if(bn_cuda_native_norm::validate(vctx,request,&key_norm,norm_data,&wt)||wt!=BN_GGUF_TENSOR_BF16||
       bn_cuda_native_norm::validate(vctx,request,&query_norm,norm_data,&wt)||wt!=BN_GGUF_TENSOR_BF16)return -1;
    const float alpha=1.f,beta=0.f;
    cast_lookup<<<(unsigned)((counts[0]+255)/256),256,0,r->stream>>>(
        (const __nv_bfloat16*)data[0],(half*)data[1],counts[0]);
    if(cudaGetLastError()!=cudaSuccess)goto failed;
    if(cublasGemmEx(r->native_f16_cublas,CUBLAS_OP_N,CUBLAS_OP_N,hd,p->n_tokens,p->k,
        &alpha,w->data,CUDA_R_16F,width,data[1],CUDA_R_16F,p->k,&beta,data[3],CUDA_R_16F,hd,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS)goto failed;
    if(bn_cuda_native_norm::run(vctx,request,&key_norm)||bn_cuda_native_norm::run(vctx,request,&query_norm))goto failed;
    /* ATen's singleton matrix strides select TT, not the algebraically
     * equivalent NN form; the latter selects a different reduction kernel. */
    if(p->n_tokens*p->streams==1) {
        if(cublasSgemm(r->native_f16_cublas,CUBLAS_OP_T,CUBLAS_OP_T,1,1,p->dim,
            &alpha,(const float*)data[5],p->dim,(const float*)data[6],1,
            &beta,(float*)data[7],1)!=CUBLAS_STATUS_SUCCESS)goto failed;
    } else if(cublasSgemmStridedBatched(r->native_f16_cublas,CUBLAS_OP_T,CUBLAS_OP_T,1,1,p->dim,
        &alpha,(const float*)data[5],p->dim,p->dim,(const float*)data[6],1,p->dim,
        &beta,(float*)data[7],1,1,p->n_tokens*p->streams)!=CUBLAS_STATUS_SUCCESS)goto failed;
    if(cublasGemmEx(r->native_f16_cublas,CUBLAS_OP_N,CUBLAS_OP_N,p->dim,p->n_tokens,p->k,
        &alpha,(const half*)w->data+hd,CUDA_R_16F,width,data[1],CUDA_R_16F,p->k,&beta,data[4],CUDA_R_16F,p->dim,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS)goto failed;
    gate_value<<<(unsigned)((wide+1023)/1024),256,0,r->stream>>>((const float*)data[7],(const half*)data[4],
        (float*)data[8],wide,p->streams,p->dim,p->gate_scale);
    if(cudaGetLastError()!=cudaSuccess)goto failed;
    {
        BnGPUAddF32ResidentPlan add={wide,p->stream,p->delta,p->out};
        if(bn_cuda_gdn::add_f32(vctx,request,&add))goto failed;
    }
    return 0;
failed:r->failed=1;return -1;
}
}
#endif
