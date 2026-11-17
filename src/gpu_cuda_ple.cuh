#ifndef BN_GPU_CUDA_PLE_CUH
#define BN_GPU_CUDA_PLE_CUH
namespace bn_cuda_ple {
using bn_cuda_gdn::ResidentRequest;
using bn_cuda_gdn::set_device;
using bn_cuda_gdn::view_data;
using bn_cuda_gdn::overlap;

static __global__ void conv_kernel(__nv_bfloat16 *out, __nv_bfloat16 *next,
    const __nv_bfloat16 *x, const __nv_bfloat16 *residual, const __nv_bfloat16 *outer,
    const __nv_bfloat16 *initial, const __nv_bfloat16 *weight,
    int nt, int dim, int kernel, int dilation, int history, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int col = i % (size_t)dim;
    if (i < (size_t)nt * dim) {
        int token = i / (size_t)dim;
        float sum = 0;
        for (int k = 0; k < kernel; k++) {
            int source = token - history + k * dilation;
            float tap = __bfloat162float(source < 0
                ? initial[(size_t)(source + history) * dim + col]
                : x[(size_t)source * dim + col]);
            sum = __fmaf_rn(__bfloat162float(weight[(size_t)col * kernel + k]), tap, sum);
        }
        float conv = __bfloat162float(__float2bfloat16_rn(sum));
        float sigmoid = __fdividef(1.0f, __fadd_rn(1.0f, __expf(-conv)));
        float activated = __bfloat162float(__float2bfloat16_rn(__fmul_rn(conv, sigmoid)));
        float branch = __bfloat162float(__float2bfloat16_rn(__fadd_rn(__bfloat162float(residual[i]), activated)));
        out[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(outer[i]), branch));
    }
    if (i < (size_t)history * dim) {
        int source = nt - history + (int)(i / (size_t)dim);
        next[i] = source < 0 ? initial[(size_t)(source + history) * dim + col]
                            : x[(size_t)source * dim + col];
    }
}

static int convolve(void *ctx_ptr, void *request, const BnGPUPLEConvResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)request;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->n_tokens <= 0 || p->dim <= 0 || p->kernel < 2 || p->kernel > 4 ||
        p->dilation < 1 || p->dilation > 8 || p->n_tokens > INT_MAX / p->dim ||
        p->dim > INT_MAX / p->kernel || set_device(r)) return -1;
    int history = (p->kernel - 1) * p->dilation;
    if (p->dim > INT_MAX / history) return -1;
    const BnCudaBuffer *w = (const BnCudaBuffer *)p->weight;
    if (!w || !w->data || !bn_quant_format_is_bf16(w->type) ||
        w->rows != p->dim || w->cols != p->kernel ||
        (size_t)p->dim * p->kernel > SIZE_MAX / sizeof(uint16_t) ||
        w->size < (size_t)p->dim * p->kernel * sizeof(uint16_t)) return -1;
    size_t n = (size_t)p->n_tokens * p->dim, h = (size_t)history * p->dim;
    BnGPUResidentView views[] = {p->x,p->residual,p->outer,p->initial_history,p->next_history,p->out};
    size_t counts[] = {n,n,n,h,h,n}; void *data[6];
    for (int j = 0; j < 6; j++) {
        data[j] = view_data(r, views[j]);
        if (!data[j] || views[j].count != counts[j] || !bn_quant_format_is_bf16(views[j].type)) return -1;
    }
    for (int o = 4; o < 6; o++) for (int j = 0; j < 6; j++)
        if (o != j && overlap(views[o],views[j])) return -1;
    if (cudaGetLastError() != cudaSuccess) {r->failed = 1; return -1;}
    size_t count = n > h ? n : h;
    conv_kernel<<<(count + 255) / 256,256,0,r->stream>>>(
        (__nv_bfloat16 *)data[5],(__nv_bfloat16 *)data[4],(const __nv_bfloat16 *)data[0],
        (const __nv_bfloat16 *)data[1],(const __nv_bfloat16 *)data[2],(const __nv_bfloat16 *)data[3],
        (const __nv_bfloat16 *)w->data,p->n_tokens,p->dim,p->kernel,p->dilation,history,count);
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1; return -1;
}

/* Materialized BF16 products followed by the eager contiguous F32 sum. */
static __global__ void gate_kernel(__nv_bfloat16 *out,
    const __nv_bfloat16 *key, const __nv_bfloat16 *query,
    const __nv_bfloat16 *value, int dim, int streams, int logical, float inv_sqrt) {
    int tid=threadIdx.x;
    size_t base=(size_t)blockIdx.x*dim;
    float acc[4]={0,0,0,0};
    for(int i=tid*4;tid<logical&&i<dim;i+=logical*4)
        for(int j=0;j<4;j++) {
            float product=__bfloat162float(__float2bfloat16_rn(__fmul_rn(
                __bfloat162float(key[base+i+j]),__bfloat162float(query[base+i+j]))));
            acc[j]=__fadd_rn(acc[j],product);
        }
    float sum=__fadd_rn(__fadd_rn(__fadd_rn(acc[0],acc[1]),acc[2]),acc[3]);
    __shared__ float partial[512],factor;
    partial[tid]=sum;
    for(int offset=logical/2;offset>=32;offset>>=1) {
        __syncthreads();
        if(tid<offset){sum=__fadd_rn(sum,partial[tid+offset]);partial[tid]=sum;}
    }
    __syncthreads();
    if(tid<32) {
        for(int offset=16;offset;offset>>=1)
            sum=__fadd_rn(sum,__shfl_down_sync(0xffffffffu,sum,offset));
        if(tid==0) {
            float dot=__bfloat162float(__float2bfloat16_rn(sum));
            float d=__bfloat162float(__float2bfloat16_rn(__fmul_rn(dot,inv_sqrt)));
            float magnitude=__bfloat162float(__float2bfloat16_rn(sqrtf(fmaxf(fabsf(d),1e-6f))));
            float signed_root=d<0?-magnitude:d>0?magnitude:0.0f;
            factor=__bfloat162float(__float2bfloat16_rn(__fdividef(1.0f,1.0f+__expf(-signed_root))));
        }
    }
    __syncthreads();
    size_t value_base=(size_t)(blockIdx.x/streams)*dim;
    for(int i=tid;i<dim;i+=blockDim.x)
        out[base+i]=__float2bfloat16_rn(__fmul_rn(factor,__bfloat162float(value[value_base+i])));
}

static int gate(void *ctx_ptr, void *request, const BnGPUPLEGateResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||
       p->n_tokens<=0||(p->dim!=2560&&p->dim!=4096)||(p->streams!=1&&p->streams!=4)||
       p->n_tokens>INT_MAX/p->streams||p->n_tokens*p->streams>INT_MAX/p->dim||
       !isfinite(p->eps)||p->eps<=0||set_device(r))return -1;
    int rows=p->n_tokens*p->streams;
    size_t n=(size_t)rows*p->dim,nv=(size_t)p->n_tokens*p->dim;
    BnGPUResidentView v[]={p->key,p->value,p->hidden,p->normalized_key,p->normalized_query,p->gated,p->normed};
    size_t counts[]={n,nv,n,n,n,n,n};void *data[7];
    for(int j=0;j<7;j++) {
        data[j]=view_data(r,v[j]);
        if(!data[j]||!bn_quant_format_is_bf16(v[j].type)||v[j].count!=counts[j])return -1;
    }
    for(int o=3;o<7;o++)for(int j=0;j<7;j++)if(o!=j&&overlap(v[o],v[j]))return -1;
    const BnCudaBuffer *weights[]={(const BnCudaBuffer*)p->norm_key,(const BnCudaBuffer*)p->norm_query,(const BnCudaBuffer*)p->norm_conv};
    for(int j=0;j<3;j++) {
        const BnCudaBuffer *w=weights[j];
        if(!w||!w->data||!bn_quant_format_is_bf16(w->type)||w->rows!=p->streams||w->cols!=p->dim||
           w->size/sizeof(uint16_t)<(size_t)p->streams*p->dim)return -1;
    }
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    /* Preflight above covers every operation before any output is modified. */
    for(int j=0;j<2;j++)
        bn_cuda_gdn::launch_contiguous_norm<false,true>((__nv_bfloat16*)data[3+j],NULL,
            (const __nv_bfloat16*)data[j==0?0:2],NULL,(const __nv_bfloat16*)weights[j]->data,
            p->eps,1,rows,p->dim,p->streams,r->stream);
    int logical=bn_cuda_gdn::contiguous_norm_threads(rows),physical=rows<=16?512:logical;
    float inv_sqrt=1.0f/sqrtf((float)p->dim);
    gate_kernel<<<rows,physical,0,r->stream>>>((__nv_bfloat16*)data[5],
        (const __nv_bfloat16*)data[3],(const __nv_bfloat16*)data[4],
        (const __nv_bfloat16*)data[1],p->dim,p->streams,logical,inv_sqrt);
    bn_cuda_gdn::launch_contiguous_norm<false,true>((__nv_bfloat16*)data[6],NULL,
        (const __nv_bfloat16*)data[5],NULL,(const __nv_bfloat16*)weights[2]->data,
        p->eps,1,rows,p->dim,p->streams,r->stream);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

} // namespace bn_cuda_ple
#endif
