#ifndef BN_GPU_CUDA_F16_ROWS_CUH
#define BN_GPU_CUDA_F16_ROWS_CUH
namespace bn_cuda_f16_rows {
static __global__ void pack(uint16_t *out,const uint16_t *in,int rows,int cols,int start,int count) {
    __shared__ uint16_t tile[32][33];
    size_t base=(size_t)blockIdx.x*32;
    for(int j=0;j<32;j+=8) {
        int r=(int)threadIdx.y+j;size_t c=base+threadIdx.x;
        if(r<count&&c<(size_t)cols)tile[r][threadIdx.x]=in[(size_t)r*cols+c];
    }
    __syncthreads();
    for(int j=0;j<32;j+=8) {
        size_t k=base+threadIdx.y+j;unsigned n=threadIdx.x;
        if(k<(size_t)cols&&n<(unsigned)count)
            out[k*(size_t)rows+(size_t)start+n]=tile[n][threadIdx.y+j];
    }
}
static void *create(void *vctx,const BnGPUF16RowsPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)vctx;
    if(!ctx||!p||!p->read_rows||p->rows<=0||p->cols<=0)return NULL;
    size_t rows=(size_t)p->rows,cols=(size_t)p->cols,tile_rows=rows<32?rows:32;
    if(rows>SIZE_MAX/cols||rows*cols>SIZE_MAX/2||tile_rows>SIZE_MAX/cols/2)return NULL;
    size_t bytes=rows*cols*2,tile_bytes=tile_rows*cols*2;
    if(bytes>SIZE_MAX-tile_bytes||p->device_budget_bytes<bytes+tile_bytes||
        p->host_budget_bytes<tile_bytes||cuda_ctx_set_device(ctx))return NULL;
    BnCudaBuffer *w=(BnCudaBuffer*)calloc(1,sizeof(*w));
    uint16_t *host=(uint16_t*)malloc(tile_bytes);void *stage=NULL;
    if(!w||!host){free(w);free(host);return NULL;}
    w->size=bytes;w->type=BN_GGUF_TENSOR_F16;w->rows=p->rows;w->cols=p->cols;w->native_f16_owner=ctx;
    cudaError_t error=cudaMalloc(&w->data,bytes);
    if(error==cudaSuccess)error=cudaMalloc(&stage,tile_bytes);
    int rejected=0;
    if(error==cudaSuccess)for(int row=0;row<p->rows;) {
        int count=p->rows-row;if(count>32)count=32;
        if(p->read_rows(p->source,row,count,host,tile_bytes/2)){rejected=1;break;}
        error=cudaMemcpy(stage,host,(size_t)count*cols*2,cudaMemcpyHostToDevice);
        if(error!=cudaSuccess)break;
        pack<<<(unsigned)((cols+31)/32),dim3(32,8)>>>(
            (uint16_t*)w->data,(const uint16_t*)stage,p->rows,p->cols,row,count);
        error=cudaGetLastError();if(error!=cudaSuccess)break;
        row+=count;
    }
    /* Default-stream DMA/packing must be complete before publication or
     * host/source scratch release, including a mid-reader failure. */
    cudaError_t drained=cudaStreamSynchronize(0);
    if(error==cudaSuccess)error=drained;
    if(stage)cudaFree(stage);
    free(host);
    if(error!=cudaSuccess||rejected) {
        if(w->data)cudaFree(w->data);
        free(w);
        if(error==cudaErrorMemoryAllocation)(void)cudaGetLastError();
        return NULL;
    }
    return w;
}
static size_t workspace_size(void *vctx) {
    return bn_cuda_exl3::workspace_size(vctx)?48u*1024u*1024u:0;
}
static int reserve(void *vctx,void *request) {
    /* FP16 torch.matmul and the original F32 hgemm fallback have distinct
     * workspace geometries. Keep the existing 16 MiB EXL3 handle unchanged. */
    if(bn_cuda_exl3::reserve(vctx,request))return -1;
    bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest*)request;
    if(r->native_f16_cublas)return 0;
    if(bn_cuda_gdn::synchronize(r))return -1;
    cublasHandle_t handle=NULL;void *ws=NULL;const size_t bytes=32u*1024u*1024u;
    if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS||
        cudaMalloc(&ws,bytes)!=cudaSuccess||
        cublasSetStream(handle,r->stream)!=CUBLAS_STATUS_SUCCESS||
        cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST)!=CUBLAS_STATUS_SUCCESS||
        cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH)!=CUBLAS_STATUS_SUCCESS||
        cublasSetWorkspace(handle,ws,bytes)!=CUBLAS_STATUS_SUCCESS) {
        if(handle)cublasDestroy(handle);
        cudaFree(ws);return -1;
    }
    r->native_f16_cublas=handle;r->native_f16_workspace=ws;return 0;
}
static int validate_linear(void *vctx,void *request,const BnGPUF16LinearResidentPlan *p,void **data) {
    BnCudaCtx *ctx=(BnCudaCtx*)vctx;
    bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||!p->weight||r->backend_owner!=ctx||ctx->compute_capability<800||
        p->k<=0||p->n<=0||p->n_tokens<=0||
        (size_t)p->k>SIZE_MAX/(size_t)p->n/2||
        (size_t)p->n_tokens>SIZE_MAX/(size_t)p->k/2||
        (size_t)p->n_tokens>SIZE_MAX/(size_t)p->n/4||
        !r->trellis_cublas||!r->trellis_workspace||
        !r->native_f16_cublas||!r->native_f16_workspace)return -1;
    const BnCudaBuffer *w=(const BnCudaBuffer*)p->weight;
    if(w->native_f16_owner!=ctx||w->type!=BN_GGUF_TENSOR_F16||!w->data||
        w->rows!=p->n||w->cols!=p->k||w->size!=(size_t)p->n*p->k*2||
        p->x.type!=BN_GGUF_TENSOR_F16||p->x.count!=(size_t)p->n_tokens*p->k||
        (p->out.type!=BN_GGUF_TENSOR_F16&&p->out.type!=BN_GGUF_TENSOR_F32)||
        p->out.count!=(size_t)p->n_tokens*p->n||
        bn_cuda_gdn::overlap(p->x,p->out))return -1;
    void *x=bn_cuda_gdn::view_data(r,p->x),*out=bn_cuda_gdn::view_data(r,p->out);
    if(!x||!out||((uintptr_t)x&1u)||
        ((uintptr_t)out&(p->out.type==BN_GGUF_TENSOR_F32?3u:1u))||
        bn_cuda_gdn::set_device(r))return -1;
    data[0]=x;data[1]=out;return 0;
}
static int linear_impl(void *vctx,void *request,const BnGPUF16LinearResidentPlan *p,bool source_bc) {
    void *data[2];if(validate_linear(vctx,request,p,data))return -1;
    bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest*)request;
    const BnCudaBuffer *w=(const BnCudaBuffer*)p->weight;
    void *x=data[0],*out=data[1];
    const float alpha=1,beta=0;
    cublasHandle_t handle=p->out.type==BN_GGUF_TENSOR_F16&&!source_bc?r->native_f16_cublas:r->trellis_cublas;
    if(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,
        p->n,p->n_tokens,p->k,&alpha,w->data,CUDA_R_16F,p->n,
        x,CUDA_R_16F,p->k,&beta,out,
        p->out.type==BN_GGUF_TENSOR_F32?CUDA_R_32F:CUDA_R_16F,p->n,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)==CUBLAS_STATUS_SUCCESS)return 0;
    r->failed=1;return -1;
}
static int linear(void *vctx,void *request,const BnGPUF16LinearResidentPlan *p) {return linear_impl(vctx,request,p,false);}
static int grouped_output(void*vctx,void*request,const BnGPUF16GroupedOutputResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    BnGPUF16GroupedOutputWorkspace ws;
    if(!ctx||!r||!p||!p->group_weight||r->backend_owner!=ctx||r->failed||
       bn_gpu_f16_grouped_output_workspace(p->groups,p->k,p->n,p->n_tokens,&ws)||
       p->down.k!=ws.width||p->down.n_tokens!=p->n_tokens)return -1;
    const BnCudaBuffer*w=(const BnCudaBuffer*)p->group_weight;
    if(w->native_f16_owner!=ctx||w->type!=BN_GGUF_TENSOR_F16||!w->data||
       w->rows!=ws.width||w->cols!=p->k||w->size!=(size_t)ws.width*p->k*2||
       p->x.type!=BN_GGUF_TENSOR_F16||p->x.count!=ws.input_count)return -1;
    void*down[2];if(validate_linear(vctx,request,&p->down,down))return -1;
    void*x=bn_cuda_gdn::view_data(r,p->x);
    if(!x||((uintptr_t)x&1u)||bn_cuda_gdn::overlap(p->x,p->down.x)||bn_cuda_gdn::overlap(p->x,p->down.out))return -1;
    const float alpha=1.f,beta=0.f;
    for(int g=0;g<p->groups;g++) {
        const half*weight=(const half*)w->data+(size_t)g*p->n;
        const half*input=(const half*)x+(size_t)g*p->n_tokens*p->k;
        half*out=(half*)down[0]+(size_t)g*p->n;
        if(cublasGemmEx(r->native_f16_cublas,CUBLAS_OP_N,CUBLAS_OP_N,p->n,p->n_tokens,p->k,
            &alpha,weight,CUDA_R_16F,ws.width,input,CUDA_R_16F,p->k,&beta,out,CUDA_R_16F,ws.width,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS){r->failed=1;return -1;}
    }
    return linear(vctx,request,&p->down);
}
static int ffn(void *vctx,void *request,const BnGPUF16FFNResidentPlan *p) {
    if(!p||p->activation_profile!=BN_GPU_F16_FFN_SILU_HALF||
        !isfinite(p->act_limit)||p->act_limit<0||p->act_limit>65504||
        p->gate.k!=p->up.k||p->gate.n!=p->up.n||p->gate.k!=p->down.n||
        p->gate.n!=p->down.k||p->gate.n_tokens!=p->up.n_tokens||
        p->gate.n_tokens!=p->down.n_tokens||(p->gate.n&1)||
        !bn_cuda_exl3::same_view(p->gate.x,p->up.x)||
        p->gate.out.type!=BN_GGUF_TENSOR_F16||p->up.out.type!=BN_GGUF_TENSOR_F16||
        p->down.out.type!=BN_GGUF_TENSOR_F32||
        p->down.x.count/2>(size_t)INT_MAX*256)return -1;
    void *gate[2],*up[2],*down[2];
    if(validate_linear(vctx,request,&p->gate,gate)||
        validate_linear(vctx,request,&p->up,up)||
        validate_linear(vctx,request,&p->down,down))return -1;
    if(((uintptr_t)gate[1]&3u)||((uintptr_t)up[1]&3u)||
        ((uintptr_t)down[0]&3u))return -1;
    const BnGPUResidentView live[]={p->gate.x,p->gate.out,p->up.out,p->down.x,p->down.out};
    for(int i=0;i<5;i++)for(int j=0;j<i;j++)
        if(bn_cuda_gdn::overlap(live[i],live[j]))return -1;
    if(linear(vctx,request,&p->gate)||linear(vctx,request,&p->up))return -1;
    bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest*)request;
    size_t count=p->down.x.count;
    bn_cuda_exl3::silu_mul<<<(unsigned)((count/2+255)/256),256,0,r->stream>>>(
        (half*)down[0],(const half*)gate[1],(const half*)up[1],count,p->act_limit);
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    return linear(vctx,request,&p->down);
}
} // namespace bn_cuda_f16_rows
int bn_gpu_cuda_read_f16_rows(const BnGPUBackend *gpu,void *weight,uint16_t *out,size_t capacity) {
    if(!gpu||gpu->kind!=BN_GPU_BACKEND_CUDA||!gpu->ctx||!weight||!out)return -1;
    BnCudaCtx *ctx=(BnCudaCtx*)gpu->ctx;const BnCudaBuffer *w=(const BnCudaBuffer*)weight;
    if(w->native_f16_owner!=ctx||w->type!=BN_GGUF_TENSOR_F16||!w->data||
        w->rows<=0||w->cols<=0||(size_t)w->rows>SIZE_MAX/(size_t)w->cols||
        (size_t)w->rows*w->cols>SIZE_MAX/2||w->size!=(size_t)w->rows*w->cols*2||
        capacity<w->size/2||cuda_ctx_set_device(ctx))return -1;
    return cudaMemcpy(out,w->data,w->size,cudaMemcpyDeviceToHost)==cudaSuccess?0:-1;
}

namespace bn_cuda_raw_rows {
static size_t width(int type){return type==BN_GGUF_TENSOR_F32?4:
    type==BN_GGUF_TENSOR_F16||type==BN_GGUF_TENSOR_BF16?2:0;}
static void *create(void *vctx,const BnGPURawRowsPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)vctx;
    if(!ctx||!p||!p->read_rows||p->rows<=0||p->cols<=0||!width(p->type))return NULL;
    size_t w=width(p->type),rows=(size_t)p->rows,cols=(size_t)p->cols;
    if(rows>SIZE_MAX/cols/w)return NULL;
    size_t bytes=rows*cols*w,tile=(rows<32?rows:32)*cols*w;
    if(p->device_budget_bytes<bytes||p->host_budget_bytes<tile||cuda_ctx_set_device(ctx))return NULL;
    BnCudaBuffer *buffer=(BnCudaBuffer*)calloc(1,sizeof(*buffer));void *host=malloc(tile);
    if(!buffer||!host){free(buffer);free(host);return NULL;}
    buffer->type=p->type;buffer->rows=p->rows;buffer->cols=p->cols;
    buffer->size=bytes;buffer->native_raw_owner=ctx;
    cudaError_t error=cudaMalloc(&buffer->data,bytes);int rejected=0;
    if(error==cudaSuccess)for(int row=0;row<p->rows;) {
        int count=p->rows-row;if(count>32)count=32;
        if(p->read_rows(p->source,row,count,host,tile)){rejected=1;break;}
        error=cudaMemcpy((char*)buffer->data+(size_t)row*cols*w,host,
            (size_t)count*cols*w,cudaMemcpyHostToDevice);
        if(error!=cudaSuccess)break;
        row+=count;
    }
    cudaError_t drained=cudaStreamSynchronize(0);if(error==cudaSuccess)error=drained;
    free(host);
    if(error!=cudaSuccess||rejected){if(buffer->data)cudaFree(buffer->data);free(buffer);
        if(error==cudaErrorMemoryAllocation)(void)cudaGetLastError();return NULL;}
    return buffer;
}
}
int bn_gpu_cuda_read_raw_rows(const BnGPUBackend *gpu,void *weight,void *out,size_t capacity) {
    if(!gpu||gpu->kind!=BN_GPU_BACKEND_CUDA||!gpu->ctx||!weight||!out)return -1;
    BnCudaCtx *ctx=(BnCudaCtx*)gpu->ctx;const BnCudaBuffer *w=(const BnCudaBuffer*)weight;
    size_t width=bn_cuda_raw_rows::width(w->type);
    if(w->native_raw_owner!=ctx||!w->data||!width||w->rows<=0||w->cols<=0||
        (size_t)w->rows>SIZE_MAX/(size_t)w->cols/width||
        w->size!=(size_t)w->rows*w->cols*width||capacity<w->size||cuda_ctx_set_device(ctx))return -1;
    return cudaMemcpy(out,w->data,w->size,cudaMemcpyDeviceToHost)==cudaSuccess?0:-1;
}
#endif
