#ifndef BN_GPU_CUDA_EXL3_CUH
#define BN_GPU_CUDA_EXL3_CUH
/*
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Wire decoding and Hadamard arithmetic adapted from exllamav3 commit
bd1f519172406a9560bd5c90e61000691bb69a07 quant/{codebook,exl3_dq,hadamard_inner}.cuh
and quant/reconstruct.cu. Backend ownership and validation are native.
*/
namespace bn_cuda_exl3 {

static __global__ void decode(half *out, const uint32_t *packed, int n) {
    int r = threadIdx.x / 16, c = threadIdx.x % 16;
    int lane = (r % 8) / 2 + (c % 8) / 2 * 8 + (c % 2) * 4;
    int j = r % 2 + r / 8 * 2 + c / 8 * 4;
    const uint32_t *words = packed + ((size_t)blockIdx.y * (n / 16) + blockIdx.x) * 24;
    int end_first = (lane * 8 + 257) * 3;
    int first = (end_first - 16) / 32, last = (end_first + 20) / 32;
    int shift = (last + 1) * 32 - (end_first + 21);
    uint64_t merged = ((uint64_t)words[first % 24] << 32) | words[last % 24];
    uint32_t bits = (uint32_t)(merged >> (shift + (j < 4 ? 12 : 0)));
    uint32_t index = (bits >> ((3 - j % 4) * 3)) & 65535u;
    uint32_t product = index * UINT32_C(0x83dcd12d);
    uint32_t sum = __dp4a(product, 0x01010101u, 0x6400u);
    half value = __hfma(__ushort_as_half((uint16_t)sum),
                       __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
    out[((size_t)blockIdx.y * 16 + r) * n + blockIdx.x * 16 + c] = value;
}

template<bool PRE>
static __global__ void had(half *out, const half *in, const half *scale, int dim) {
    int lane = threadIdx.x;
    size_t base = (size_t)blockIdx.x * dim + blockIdx.y * 128 + lane * 4;
    size_t si = blockIdx.y * 128 + lane * 4;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        half x = in[base + j];
        if (PRE) x = __hmul(x, scale[si + j]);
        v[j] = __half2float(x);
    }
    float s0 = __fadd_rn(v[0], v[1]), d0 = __fsub_rn(v[0], v[1]);
    float s1 = __fadd_rn(v[2], v[3]), d1 = __fsub_rn(v[2], v[3]);
    float h[4] = { __fadd_rn(s0,s1), __fadd_rn(d0,d1),
                   __fsub_rn(s0,s1), __fsub_rn(d0,d1) };
#pragma unroll
    for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float peer = __shfl_xor_sync(0xffffffffu, h[j], mask);
            h[j] = __fadd_rn((lane & mask) ? -h[j] : h[j], peer);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        half x = __float2half_rn(__fmul_rn(h[j], 0.088388347648f));
        if (!PRE) x = __hmul(x, scale[si + j]);
        out[base + j] = x;
    }
}

static __global__ void had_post_f32(float *out, const float *in, const half *scale, int dim) {
    int lane = threadIdx.x;
    size_t base = (size_t)blockIdx.x * dim + blockIdx.y * 128 + lane * 4;
    size_t si = blockIdx.y * 128 + lane * 4;
    float s0 = __fadd_rn(in[base],in[base+1]), d0 = __fsub_rn(in[base],in[base+1]);
    float s1 = __fadd_rn(in[base+2],in[base+3]), d1 = __fsub_rn(in[base+2],in[base+3]);
    float h[4] = { __fadd_rn(s0,s1), __fadd_rn(d0,d1),
                   __fsub_rn(s0,s1), __fsub_rn(d0,d1) };
#pragma unroll
    for (int mask=1;mask<32;mask<<=1) {
#pragma unroll
        for (int j=0;j<4;++j) {
            float peer=__shfl_xor_sync(0xffffffffu,h[j],mask);
            h[j]=__fadd_rn((lane&mask)?-h[j]:h[j],peer);
        }
    }
#pragma unroll
    for(int j=0;j<4;++j)
        out[base+j]=__fmul_rn(__fmul_rn(h[j],0.088388347648f),__half2float(scale[si+j]));
}

static int upload_shape(BnCudaCtx *ctx,size_t bytes,int format,int k,int n,uint32_t marker) {
    const BnQuantTrellisOps *ops=bn_quant_format_trellis_ops(format);
    return ctx&&ctx->compute_capability>=800&&ops&&
        ops->projection==BN_QUANT_TRELLIS_PROJECTION_EXL3_MUL1_3_F16_F32&&
        marker==ops->codebook_marker&&k>0&&n>0&&k%128==0&&n%128==0&&
        k<=INT_MAX/n&&(size_t)k*n/8*3+((size_t)k+n)*2==bytes;
}
static BnCudaBuffer *allocate_weight(BnCudaCtx *ctx,size_t bytes,int format,
    int k,int n,uint32_t marker) {
    if(!upload_shape(ctx,bytes,format,k,n,marker)||cuda_ctx_set_device(ctx))return NULL;
    BnCudaBuffer *w=(BnCudaBuffer*)calloc(1,sizeof(*w));if(!w)return NULL;
    w->size=bytes;w->type=format;w->cols=k;w->rows=n;
    w->trellis_owner=ctx;w->trellis_marker=marker;
    cudaError_t error=cudaMalloc(&w->data,bytes);
    if(error!=cudaSuccess) {
        free(w);if(error==cudaErrorMemoryAllocation)(void)cudaGetLastError();return NULL;
    }
    return w;
}
static void *create_weight(void *ctx_ptr,const void *bundle,size_t bytes,
    int format,int k,int n,uint32_t marker) {
    if(!bundle)return NULL;
    BnCudaBuffer *w=allocate_weight((BnCudaCtx*)ctx_ptr,bytes,format,k,n,marker);
    if(!w)return NULL;
    cudaError_t error=cudaMemcpy(w->data,bundle,bytes,cudaMemcpyHostToDevice);
    /* Pageable H2D publication must complete before nonblocking consumers. */
    if(error==cudaSuccess)error=cudaStreamSynchronize(0);
    if(error!=cudaSuccess){cudaFree(w->data);free(w);return NULL;}
    return w;
}

static int write_weight(void *ctx_ptr,void *buffer,const void *bundle,size_t bytes,
    int format,int k,int n,uint32_t marker) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;
    BnCudaBuffer *w=(BnCudaBuffer*)buffer;
    const BnQuantTrellisOps *ops=bn_quant_format_trellis_ops(format);
    if(!ctx || !w || !bundle || !ops || ctx->compute_capability<800 ||
       ops->projection!=BN_QUANT_TRELLIS_PROJECTION_EXL3_MUL1_3_F16_F32 ||
       marker!=ops->codebook_marker || k<=0 || n<=0 || k%128 || n%128 ||
       k>INT_MAX/n || (size_t)k*n/8*3+((size_t)k+n)*2!=bytes ||
       w->trellis_owner!=ctx || !w->data || w->size!=bytes ||
       w->type!=format || w->cols!=k || w->rows!=n ||
       w->trellis_marker!=marker || cuda_ctx_set_device(ctx))return -1;
    cudaError_t error=cudaMemcpy(w->data,bundle,bytes,cudaMemcpyHostToDevice);
    /* Preserve create_weight's publication guarantee across consumer streams. */
    if(error==cudaSuccess)error=cudaStreamSynchronize(0);
    return error==cudaSuccess?0:-1;
}

struct Upload {
    BnCudaCtx *owner;
    void *host;
    size_t bytes;
    cudaStream_t stream;
};
static void *upload_create(void *context,size_t bytes,void **host) {
    BnCudaCtx *ctx=(BnCudaCtx*)context;
    if(!ctx||!host||!bytes||ctx->compute_capability<800||cuda_ctx_set_device(ctx))return NULL;
    Upload *u=(Upload*)calloc(1,sizeof(*u));if(!u)return NULL;
    u->owner=ctx;u->bytes=bytes;
    cudaError_t error=cudaHostAlloc(&u->host,bytes,cudaHostAllocDefault);
    if(error==cudaSuccess)error=cudaStreamCreateWithFlags(&u->stream,cudaStreamNonBlocking);
    if(error!=cudaSuccess) {
        if(u->host)cudaFreeHost(u->host);
        free(u);
        if(error==cudaErrorMemoryAllocation)(void)cudaGetLastError();
        return NULL;
    }
    *host=u->host;return u;
}
static void upload_destroy(void *context,void *state) {
    Upload *u=(Upload*)state;BnCudaCtx *ctx=(BnCudaCtx*)context;
    if(!u||!ctx||u->owner!=ctx||cuda_ctx_set_device(ctx))return;
    (void)cudaStreamSynchronize(u->stream);(void)cudaStreamDestroy(u->stream);
    (void)cudaFreeHost(u->host);free(u);
}
static int upload_batch(void *context,void *state,BnGPUTrellisUploadItem *items,int count) {
    BnCudaCtx *ctx=(BnCudaCtx*)context;Upload *u=(Upload*)state;
    if(!ctx||!u||u->owner!=ctx||!items||count<1||count>3)return -1;
    size_t total=0;BnCudaBuffer *w[3]={NULL,NULL,NULL};bool created[3]={false,false,false};
    for(int p=0;p<count;p++) {
        const BnGPUTrellisUploadItem &v=items[p];
        if(!upload_shape(ctx,v.bytes,v.format,v.k,v.n,v.marker)||
           v.offset>u->bytes||v.bytes>u->bytes-v.offset||v.bytes>u->bytes-total)return -1;
        total+=v.bytes;w[p]=(BnCudaBuffer*)v.buffer;
        if(v.source) {
            uintptr_t source=(uintptr_t)v.source,stage=(uintptr_t)u->host;
            if(v.bytes>UINTPTR_MAX-source||u->bytes>UINTPTR_MAX-stage||
               (source<stage+u->bytes&&stage<source+v.bytes))return -1;
        }
        if(w[p]&&(w[p]->trellis_owner!=ctx||!w[p]->data||w[p]->size!=v.bytes||
           w[p]->type!=v.format||w[p]->cols!=v.k||w[p]->rows!=v.n||w[p]->trellis_marker!=v.marker))return -1;
        for(int j=0;j<p;j++) {
            if(w[p]&&w[p]==w[j])return -1;
            if(v.offset<items[j].offset+items[j].bytes&&
               items[j].offset<v.offset+v.bytes)return -1;
        }
    }
    if(cuda_ctx_set_device(ctx))return -1;
    cudaError_t error=cudaSuccess;
    for(int p=0;p<count;p++)if(!w[p]) {
        const BnGPUTrellisUploadItem &v=items[p];
        w[p]=allocate_weight(ctx,v.bytes,v.format,v.k,v.n,v.marker);
        if(!w[p]){error=cudaErrorMemoryAllocation;break;}created[p]=true;
    }
    bool attempted=false;
    for(int p=0;p<count&&error==cudaSuccess;p++) {
        void *stage=(unsigned char*)u->host+items[p].offset;
        if(items[p].source)memcpy(stage,items[p].source,items[p].bytes);
        attempted=true;
        error=cudaMemcpyAsync(w[p]->data,stage,
            items[p].bytes,cudaMemcpyHostToDevice,u->stream);
    }
    if(attempted) {
        cudaError_t done=cudaStreamSynchronize(u->stream);
        if(error==cudaSuccess)error=done;
    }
    if(error!=cudaSuccess) {
        for(int p=0;p<count;p++)if(created[p]){cudaFree(w[p]->data);free(w[p]);}
        return -1;
    }
    for(int p=0;p<count;p++)items[p].buffer=w[p];
    return 0;
}

static size_t workspace_size(void *ctx_ptr) {
    const BnCudaCtx *ctx = (const BnCudaCtx *)ctx_ptr;
    return ctx && ctx->compute_capability >= 800 ? 16u * 1024u * 1024u : 0;
}
static int reserve(void *ctx_ptr, void *ptr) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    bn_cuda_gdn::ResidentRequest *r = (bn_cuda_gdn::ResidentRequest *)ptr;
    if (!ctx || !r || r->backend_owner != ctx || !workspace_size(ctx) ||
        bn_cuda_gdn::set_device(r)) return -1;
    if (r->trellis_cublas) return 0;
    if (bn_cuda_gdn::synchronize(r)) return -1;
    cublasHandle_t handle = NULL;
    void *ws = NULL;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS ||
        cudaMalloc(&ws, workspace_size(ctx)) != cudaSuccess ||
        cublasSetStream(handle, r->stream) != CUBLAS_STATUS_SUCCESS ||
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(handle, ws, workspace_size(ctx)) != CUBLAS_STATUS_SUCCESS) {
        if (handle) cublasDestroy(handle);
        cudaFree(ws);
        return -1;
    }
    r->trellis_cublas = handle;
    r->trellis_workspace = ws;
    return 0;
}

static int validate(void *ctx_ptr, void *ptr, const BnGPUTrellisProjectionF16ResidentPlan *p, void **data) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    bn_cuda_gdn::ResidentRequest *r = (bn_cuda_gdn::ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->k <= 0 || p->n <= 0 || p->n_tokens <= 0 || p->k % 128 || p->n % 128 ||
        p->k / 16 > 65535 || p->n / 128 > 65535 || p->k > INT_MAX / p->n ||
        p->n_tokens > INT_MAX / p->k || p->n_tokens > INT_MAX / p->n ||
        bn_cuda_gdn::set_device(r)) return -1;
    const BnQuantTrellisOps *ops = bn_quant_format_trellis_ops(p->format);
    if (!ops || ops->projection != BN_QUANT_TRELLIS_PROJECTION_EXL3_MUL1_3_F16_F32 ||
        p->codebook_marker != ops->codebook_marker || !r->trellis_cublas ||
        !r->trellis_workspace) return -1;
    if (p->out.type != BN_GGUF_TENSOR_F16 && p->out.type != BN_GGUF_TENSOR_F32) return -1;
    size_t kn = (size_t)p->k * p->n, nx = (size_t)p->n_tokens * p->k,
           ny = (size_t)p->n_tokens * p->n;
    const BnGPUResidentView views[] = { p->packed,p->suh,p->svh,p->x,p->xh,p->decoded,p->gemm,p->out };
    const size_t counts[] = { kn / 8 * 3,(size_t)p->k,(size_t)p->n,nx,nx,kn,ny,ny };
    int first=0;
    if(p->weight) {
        const BnCudaBuffer *w=(const BnCudaBuffer*)p->weight;
        if(w->trellis_owner!=ctx || !w->data || w->type!=p->format ||
           w->cols!=p->k || w->rows!=p->n || w->trellis_marker!=p->codebook_marker ||
           w->size!=counts[0]+((size_t)p->k+p->n)*2)return -1;
        for(int i=0;i<3;i++)
            if(views[i].request || views[i].count || views[i].offset || views[i].slot || views[i].type)return -1;
        data[0]=w->data;data[1]=(unsigned char*)w->data+counts[0];
        data[2]=(unsigned char*)data[1]+(size_t)p->k*2;
        first=3;
    }
    for (int i = first; i < 8; ++i) {
        data[i] = bn_cuda_gdn::view_data(r, views[i]);
        if (!data[i] || views[i].count != counts[i] ||
            views[i].type != (i == 0 ? BN_GGUF_TENSOR_I8 : i < 6 ? BN_GGUF_TENSOR_F16 : p->out.type) ||
            ((uintptr_t)data[i] & 7u)) return -1;
        for (int j = first; j < i; ++j)
            if (bn_cuda_gdn::overlap(views[i], views[j])) return -1;
    }
    return 0;
}
static int project(void *ctx_ptr, void *ptr, const BnGPUTrellisProjectionF16ResidentPlan *p) {
    void *data[8];
    if (validate(ctx_ptr,ptr,p,data)) return -1;
    bn_cuda_gdn::ResidentRequest *r = (bn_cuda_gdn::ResidentRequest *)ptr;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    had<true><<<dim3(p->n_tokens,p->k/128),32,0,r->stream>>>(
        (half *)data[4],(const half *)data[3],(const half *)data[1],p->k);
    decode<<<dim3(p->n/16,p->k/16),256,0,r->stream>>>(
        (half *)data[5],(const uint32_t *)data[0],p->n);
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    float alpha = 1.0f, beta = 0.0f;
    if (cublasGemmEx(r->trellis_cublas,CUBLAS_OP_N,CUBLAS_OP_N,p->n,p->n_tokens,p->k,
        &alpha,data[5],CUDA_R_16F,p->n,data[4],CUDA_R_16F,p->k,
        &beta,data[6],p->out.type == BN_GGUF_TENSOR_F32 ? CUDA_R_32F : CUDA_R_16F,p->n,CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
        r->failed = 1; return -1;
    }
    if (p->out.type == BN_GGUF_TENSOR_F32)
        had_post_f32<<<dim3(p->n_tokens,p->n/128),32,0,r->stream>>>(
            (float *)data[7],(const float *)data[6],(const half *)data[2],p->n);
    else had<false><<<dim3(p->n_tokens,p->n/128),32,0,r->stream>>>(
        (half *)data[7],(const half *)data[6],(const half *)data[2],p->n);
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1; return -1;
}
static __global__ void silu_mul(half *out,const half *gate,const half *up,size_t count,float limit) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count/2)return;
    half2 g=((const half2*)gate)[i],u=((const half2*)up)[i];
    half2 e=h2exp(__hneg2(g));
    half2 r=h2rcp(__hadd2(__float2half2_rn(1.0f),e));
    g=__hmul2(g,r);
    if(limit!=0.0f) {
        u=__hmax2(u,__float2half2_rn(-limit));
        u=__hmin2(u,__float2half2_rn(limit));
        g=__hmin2(g,__float2half2_rn(limit));
    }
    ((half2*)out)[i]=__hmul2(g,u);
}
static int same_view(BnGPUResidentView a,BnGPUResidentView b) {
    return a.request==b.request && a.slot==b.slot && a.offset==b.offset &&
           a.count==b.count && a.type==b.type;
}
static int ffn(void *ctx_ptr,void *ptr,const BnGPUTrellisFFNResidentPlan *p) {
    if(!p || p->activation_profile!=BN_GPU_TRELLIS_FFN_SILU_HALF ||
       !isfinite(p->act_limit) || p->act_limit<0 || p->act_limit>65504 ||
       p->gate.k!=p->up.k || p->gate.n!=p->up.n || p->gate.k!=p->down.n ||
       p->gate.n!=p->down.k || p->gate.n_tokens!=p->up.n_tokens ||
       p->gate.n_tokens!=p->down.n_tokens || !same_view(p->gate.x,p->up.x) ||
       p->gate.out.type!=BN_GGUF_TENSOR_F16 || p->up.out.type!=BN_GGUF_TENSOR_F16 ||
       p->down.out.type!=BN_GGUF_TENSOR_F32) return -1;
    void *g[8],*u[8],*d[8];
    if(validate(ctx_ptr,ptr,&p->gate,g) || validate(ctx_ptr,ptr,&p->up,u) ||
       validate(ctx_ptr,ptr,&p->down,d))return -1;
    const BnGPUResidentView live[]={p->gate.x,p->gate.out,p->up.out,p->down.x,p->down.out};
    const BnGPUResidentView weights[]={p->gate.packed,p->gate.suh,p->gate.svh,
        p->up.packed,p->up.suh,p->up.svh,p->down.packed,p->down.suh,p->down.svh};
    const BnGPUResidentView scratch[]={p->gate.xh,p->gate.decoded,p->gate.gemm,
        p->up.xh,p->up.decoded,p->up.gemm,p->down.xh,p->down.decoded,p->down.gemm};
    for(int i=0;i<5;i++) {
        for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(live[i],live[j]))return -1;
        for(int j=0;j<9;j++)if(bn_cuda_gdn::overlap(live[i],weights[j]))return -1;
    }
    for(int i=0;i<9;i++) {
        for(int j=0;j<5;j++)if(bn_cuda_gdn::overlap(scratch[i],live[j]))return -1;
        for(int j=0;j<9;j++)if(bn_cuda_gdn::overlap(scratch[i],weights[j]))return -1;
    }
    if(project(ctx_ptr,ptr,&p->gate) || project(ctx_ptr,ptr,&p->up))return -1;
    bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest*)ptr;
    size_t count=p->down.x.count;
    silu_mul<<<(unsigned)((count+511)/512),256,0,r->stream>>>(
        (half*)d[3],(const half*)g[7],(const half*)u[7],count,p->act_limit);
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    return project(ctx_ptr,ptr,&p->down);
}
} // namespace bn_cuda_exl3
#endif
