#ifndef BN_GPU_CUDA_HC_CUH
#define BN_GPU_CUDA_HC_CUH

namespace bn_cuda_hc {
using bn_cuda_gdn::ResidentRequest;
using bn_cuda_gdn::set_device;
using bn_cuda_gdn::view_data;
using bn_cuda_gdn::overlap;

static __device__ float rounded_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

static __device__ float sigmoid_bf16(float value) {
    /* Unary Torch sigmoid preserves BF16 subnormals at -87.5/-88/-88.5.
     * Fast division flushes these otherwise valid outputs to zero. */
    return rounded_bf16(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value))));
}

static __global__ void scaled_silu_kernel(__nv_bfloat16 *out,
    const __nv_bfloat16 *x, size_t count, float inverse) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    float value=rounded_bf16(__fmul_rn(__bfloat162float(x[i]),inverse));
    out[i]=__float2bfloat16_rn(cuda_bf16_silu_float(value));
}

template <int Streams>
static __global__ void mix_kernel(__nv_bfloat16 *out,
    const __nv_bfloat16 *normalized, const __nv_bfloat16 *gate,
    size_t count, int dim) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    size_t base=(i/(size_t)dim)*Streams*dim+i%dim;
    float values[Streams];
#pragma unroll
    for(int s=0;s<Streams;s++) {
        size_t j=base+(size_t)s*dim;
        values[s]=rounded_bf16(__fmul_rn(__bfloat162float(normalized[j]),
                                       sigmoid_bf16(__bfloat162float(gate[j]))));
    }
    float sum;
    if(dim==1) {
        /* A contiguous reduction uses the warp-style tree and zero identity. */
#pragma unroll
        for(int s=0;s<Streams;s++)values[s]=__fadd_rn(0.0f,values[s]);
#pragma unroll
        for(int step=Streams/2;step;step>>=1)
#pragma unroll
            for(int s=0;s<step;s++)values[s]=__fadd_rn(values[s],values[s+step]);
        sum=values[0];
    } else {
        /* Noncontiguous stream means use Torch's four serial accumulators. */
        float acc[4]={0,0,0,0};
#pragma unroll
        for(int s=0;s<Streams;s++)acc[s%4]=__fadd_rn(acc[s%4],values[s]);
        sum=__fadd_rn(__fadd_rn(__fadd_rn(acc[0],acc[1]),acc[2]),acc[3]);
    }
    out[i]=__float2bfloat16_rn(__fmul_rn(sum,1.0f/Streams));
}

static __global__ void combine_kernel(__nv_bfloat16 *out,
    const __nv_bfloat16 *residual, const __nv_bfloat16 *block,
    const __nv_bfloat16 *inject, size_t count, int dim, int streams) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    size_t row=i/dim, token=row/streams;
    float scaled=rounded_bf16(__fmul_rn(__bfloat162float(inject[row]),1.0f/streams));
    float weight=rounded_bf16(__fmul_rn(2.0f,sigmoid_bf16(scaled)));
    float branch=rounded_bf16(__fmul_rn(__bfloat162float(block[token*dim+i%dim]),weight));
    out[i]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(residual[i]),branch));
}

static int streams_valid(int streams) {
    return streams>=2&&streams<=16&&(streams&(streams-1))==0;
}

static int geometry(int tokens, int dim, int streams, size_t *narrow, size_t *wide) {
    if(tokens<=0||dim<=0||!streams_valid(streams)||tokens>INT_MAX/dim||
       (size_t)tokens*dim>(size_t)INT_MAX/streams)return -1;
    *narrow=(size_t)tokens*dim;*wide=*narrow*streams;return 0;
}

static int views(ResidentRequest *r, const BnGPUResidentView *v,
    const size_t *counts, int n, void **data) {
    for(int i=0;i<n;i++) {
        data[i]=view_data(r,v[i]);
        if(!data[i]||!bn_quant_format_is_bf16(v[i].type)||v[i].count!=counts[i])return -1;
        if(i<n-1&&overlap(v[i],v[n-1]))return -1;
    }
    return 0;
}

static int scaled_silu(void *ctx_ptr, void *request, const BnGPUScaledSiluResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||
       p->rows<=0||p->cols<=0||p->rows>INT_MAX/p->cols||
       p->divisor<1||p->divisor>16||(p->divisor&(p->divisor-1))||set_device(r))return -1;
    size_t count=(size_t)p->rows*p->cols,counts[]={count,count};
    BnGPUResidentView v[]={p->x,p->out};void*data[2];
    if(views(r,v,counts,2,data))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    scaled_silu_kernel<<<(unsigned)((count+255)/256),256,0,r->stream>>>(
        (__nv_bfloat16*)data[1],(const __nv_bfloat16*)data[0],count,1.0f/p->divisor);
    if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
}

static int mix(void *ctx_ptr, void *request, const BnGPUHyperMixResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)request;
    size_t narrow,wide;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||
       geometry(p->n_tokens,p->dim,p->streams,&narrow,&wide)||set_device(r))return -1;
    size_t counts[]={wide,wide,narrow};BnGPUResidentView v[]={p->normalized,p->gate,p->out};void*data[3];
    if(views(r,v,counts,3,data))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
#define BN_HC_MIX(S) mix_kernel<S><<<(unsigned)((narrow+255)/256),256,0,r->stream>>>( \
    (__nv_bfloat16*)data[2],(const __nv_bfloat16*)data[0],(const __nv_bfloat16*)data[1],narrow,p->dim)
    if(p->streams==2){BN_HC_MIX(2);}else if(p->streams==4){BN_HC_MIX(4);}
    else if(p->streams==8){BN_HC_MIX(8);}else{BN_HC_MIX(16);}
#undef BN_HC_MIX
    if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
}

static int combine(void *ctx_ptr, void *request, const BnGPUHyperCombineResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)request;
    size_t narrow,wide;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||
       geometry(p->n_tokens,p->dim,p->streams,&narrow,&wide)||set_device(r))return -1;
    size_t counts[]={wide,narrow,(size_t)p->n_tokens*p->streams,wide};
    BnGPUResidentView v[]={p->residual,p->block_out,p->inject,p->out};void*data[4];
    if(views(r,v,counts,4,data))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    combine_kernel<<<(unsigned)((wide+255)/256),256,0,r->stream>>>(
        (__nv_bfloat16*)data[3],(const __nv_bfloat16*)data[0],(const __nv_bfloat16*)data[1],
        (const __nv_bfloat16*)data[2],wide,p->dim,p->streams);
    if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
}
} // namespace bn_cuda_hc
#endif
