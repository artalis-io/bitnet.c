#ifndef BN_GPU_CUDA_NATIVE_IO_CUH
#define BN_GPU_CUDA_NATIVE_IO_CUH
namespace bn_cuda_native_io {
static __global__ void embedding_kernel(float *out,const int32_t *ids,
    const uint16_t *weight,int *error,int dim,int vocab){
    int c=(int)blockIdx.x*256+(int)threadIdx.x,r=(int)blockIdx.y,d4=dim/4;
    if(c>=d4)return;
    int token=ids[r];float4 v;
    if(token<0||token>=vocab){atomicExch(error,1);v=make_float4(0,0,0,0);}
    else {
        uint2 bits=((const uint2*)weight)[(size_t)token*d4+c];
        /* Bit expansion preserves BF16 subnormals, signed zero and NaN payloads. */
        v=make_float4(__uint_as_float(bits.x<<16),__uint_as_float(bits.x&0xffff0000u),
                     __uint_as_float(bits.y<<16),__uint_as_float(bits.y&0xffff0000u));
    }
    for(int h=0;h<4;h++)((float4*)out)[((size_t)r*4+h)*d4+c]=v;
}
static int embedding(void *vctx,void *request,const BnGPUBF16EmbeddingHC4ResidentPlan *p){
    BnCudaCtx *ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||!r->embedding_error||
        !bn_gpu_hc4_partials_count(p->n_tokens,p->dim)||p->vocab_size<=0)return -1;
    size_t rows=(size_t)p->n_tokens,dim=(size_t)p->dim,vocab=(size_t)p->vocab_size;
    if(rows>SIZE_MAX/dim/16||vocab>SIZE_MAX/dim/2)return -1;
    const BnCudaBuffer*w=(const BnCudaBuffer*)p->weight;
    if(!w||w->native_raw_owner!=ctx||!w->data||w->type!=BN_GGUF_TENSOR_BF16||
        w->rows!=p->vocab_size||w->cols!=p->dim||w->size!=vocab*dim*2||((uintptr_t)w->data&7u)||
        p->indices.type!=BN_GGUF_TENSOR_I32||p->indices.count!=rows||
        p->out.type!=BN_GGUF_TENSOR_F32||p->out.count!=rows*4*dim||
        bn_cuda_gdn::overlap(p->indices,p->out))return -1;
    void*ids=bn_cuda_gdn::view_data(r,p->indices),*out=bn_cuda_gdn::view_data(r,p->out);
    if(!ids||!out||((uintptr_t)ids&3u)||((uintptr_t)out&15u)||bn_cuda_gdn::set_device(r))return -1;
    embedding_kernel<<<dim3((p->dim/4+255)/256,p->n_tokens),256,0,r->stream>>>(
        (float*)out,(const int32_t*)ids,(const uint16_t*)w->data,r->embedding_error,p->dim,p->vocab_size);
    if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
}
static int head(void *vctx,void *request,const BnGPUHC4HeadResidentPlan *p){
    BnCudaCtx *ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||
        !r->trellis_cublas||!r->trellis_workspace||!bn_gpu_hc4_partials_count(p->n_tokens,p->dim))return -1;
    size_t rows=(size_t)p->n_tokens,dim=(size_t)p->dim;
    if(rows>SIZE_MAX/dim/16)return -1;
    BnGPUResidentView views[]={p->streams,p->pre,p->out};
    size_t counts[]={rows*4*dim,rows*4,rows*dim};void*data[3];
    for(int i=0;i<3;i++){
        if(views[i].type!=BN_GGUF_TENSOR_F32||views[i].count!=counts[i])return -1;
        data[i]=bn_cuda_gdn::view_data(r,views[i]);if(!data[i]||((uintptr_t)data[i]&15u))return -1;
        for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(views[i],views[j]))return -1;
    }
    if(bn_cuda_gdn::set_device(r))return -1;
    const float alpha=1,beta=0;
    if(cublasSgemmStridedBatched(r->trellis_cublas,CUBLAS_OP_N,CUBLAS_OP_N,p->dim,1,4,
        &alpha,(const float*)data[0],p->dim,(long long)p->dim*4,
        (const float*)data[1],4,4,&beta,(float*)data[2],p->dim,p->dim,p->n_tokens)==CUBLAS_STATUS_SUCCESS)return 0;
    r->failed=1;return -1;
}
}
#endif
