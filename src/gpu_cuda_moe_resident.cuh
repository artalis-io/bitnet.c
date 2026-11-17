#ifndef BN_GPU_CUDA_MOE_RESIDENT_CUH
#define BN_GPU_CUDA_MOE_RESIDENT_CUH

namespace bn_cuda_moe_resident {
using bn_cuda_gdn::ResidentRequest;
using bn_cuda_gdn::view_data;
using bn_cuda_gdn::overlap;
using bn_cuda_gdn::set_device;

/* Original E288/K8 single-group sigmoid profile. The reference changes
 * unbiased-score recovery above 1024 tokens; keep that visible boundary. */
static __global__ void biased_sigmoid_f32_kernel(float*weights,int*ids,
    const float*logits,const float*bias,int nt,float route_scale) {
    int t=blockIdx.x,lane=threadIdx.x;float score[9];int index[9];
    for(int j=0;j<9;++j){int e=j*32+lane;float x=logits[t*288+e],b=bias[e];
        float s=fmaf(.5f,tanhf(.5f*x),.5f);
        float v=isfinite(x)&&isfinite(b)?__fadd_rn(s,b):-INFINITY;
        score[j]=v==0.f?0.f:v;index[j]=e;}
    float chosen=-INFINITY;int chosen_id=0;
    for(int rank=0;rank<8;++rank){float best=-INFINITY;int id=INT_MAX;
        for(int j=0;j<9;++j)if(score[j]>best||(score[j]==best&&index[j]<id)){best=score[j];id=index[j];}
        for(int off=16;off;off>>=1){float v=__shfl_xor_sync(0xffffffffu,best,off);int e=__shfl_xor_sync(0xffffffffu,id,off);
            if(v>best||(v==best&&e<id)){best=v;id=e;}}
        if(lane==rank){chosen=best;chosen_id=id;}
        for(int j=0;j<9;++j)if(index[j]==id){score[j]=-INFINITY;index[j]=INT_MAX;}
    }
    bool valid=lane<8&&isfinite(chosen);float value=0;
    if(valid)value=nt<=1024?fmaf(.5f,tanhf(.5f*logits[t*288+chosen_id]),.5f):__fsub_rn(chosen,bias[chosen_id]);
    unsigned finite=__ballot_sync(0xffffffffu,valid);float sum=value;
    for(int off=16;off;off>>=1)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,off));
    float scale=__fdiv_rn(route_scale,__fadd_rn(sum,1e-20f));
    if(lane<8){ids[t*8+lane]=chosen_id;weights[t*8+lane]=finite?__fmul_rn(value,scale):0.125f;}
}
static int biased_sigmoid_f32(void*ctx_ptr,void*ptr,const BnGPUBiasedSigmoidF32ResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||
       p->n_tokens<1||p->n_tokens>INT_MAX/288||p->n_experts!=288||p->k!=8||
       !isfinite(p->route_scale)||p->route_scale<=0||set_device(r))return -1;
    const BnCudaBuffer*b=(const BnCudaBuffer*)p->bias;
    if(!b||!b->data||b->type!=BN_GGUF_TENSOR_F32||b->rows!=1||b->cols!=288||b->size!=288*sizeof(float))return -1;
    BnGPUResidentView v[]={p->logits,p->weights,p->indices};size_t n=p->n_tokens,counts[]={n*288,n*8,n*8};
    int types[]={BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32};void*data[3];
    for(int i=0;i<3;++i){data[i]=view_data(r,v[i]);if(!data[i]||v[i].type!=types[i]||v[i].count!=counts[i])return -1;
        for(int j=0;j<i;++j)if(overlap(v[i],v[j]))return -1;}
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    biased_sigmoid_f32_kernel<<<p->n_tokens,32,0,r->stream>>>((float*)data[1],(int*)data[2],(const float*)data[0],(const float*)b->data,p->n_tokens,p->route_scale);
    if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
}

static __device__ float full_div(float a, float b) {
    float out;
    asm("div.full.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

static __device__ float block_reduce(float value, float *scratch, bool maximum) {
    int tid = threadIdx.x;
    for (int d = 16; d; d >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, value, d);
        value = maximum ? fmaxf(value, other) : __fadd_rn(value, other);
    }
    if (!(tid & 31)) scratch[tid / 32] = value;
    __syncthreads();
    value = scratch[tid & 3];
    for (int d = 2; d; d >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, value, d);
        value = maximum ? fmaxf(value, other) : __fadd_rn(value, other);
    }
    __syncthreads();
    return value;
}

/* SM120 small-batch profile: contiguous four values/thread, four warps.
 * Pairwise local sums and div.full match the pinned sorted softmax kernel. */
static __global__ void sorted_topk(float *weights, int *ids, int *source,
    const __nv_bfloat16 *logits, int nt, int renormalize) {
    int tid = threadIdx.x, token = blockIdx.x;
    __shared__ float scratch[4];
    __shared__ unsigned long long exchange[512];
    float value[4], maximum = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        value[j] = __bfloat162float(logits[(size_t)token * 512 + tid * 4 + j]);
        maximum = fmaxf(maximum, value[j]);
    }
    maximum = block_reduce(maximum, scratch, true);
#pragma unroll
    for (int j = 0; j < 4; j++) value[j] = bn_cuda_gdn::ex(__fsub_rn(value[j], maximum));
    float sum = __fadd_rn(__fadd_rn(value[0], value[2]), __fadd_rn(value[1], value[3]));
    sum = block_reduce(sum, scratch, false);
    unsigned long long key[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float probability = full_div(value[j], sum);
        if (!isfinite(probability)) probability = 0;
        key[j] = ((unsigned long long)__float_as_uint(probability) << 32) |
            (0xffffffffu - (unsigned)(tid * 4 + j));
    }
    for (int size = 2; size <= 512; size *= 2) {
        for (int stride = size / 2; stride; stride /= 2) {
            if (stride >= 128) {
#pragma unroll
                for (int j = 0; j < 4; j++) exchange[tid * 4 + j] = key[j];
                __syncthreads();
            }
            unsigned long long next[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int index = tid * 4 + j;
                unsigned long long other = stride < 4 ? key[j ^ stride] :
                    stride < 128 ? __shfl_xor_sync(0xffffffffu, key[j], stride / 4) :
                    exchange[index ^ stride];
                bool take_max = ((index & size) == 0) == ((index & stride) == 0);
                next[j] = take_max ? (key[j] > other ? key[j] : other) :
                    (key[j] < other ? key[j] : other);
            }
            if (stride >= 128) __syncthreads();
#pragma unroll
            for (int j = 0; j < 4; j++) key[j] = next[j];
        }
    }
    float selected[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
        selected[j] = tid * 4 + j < 10 ? __uint_as_float((unsigned)(key[j] >> 32)) : 0;
    if (renormalize) {
        sum = __fadd_rn(__fadd_rn(selected[0], selected[2]), __fadd_rn(selected[1], selected[3]));
        sum = block_reduce(sum, scratch, false);
        if (!(sum > 0)) sum = 1;
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int rank = tid * 4 + j;
        if (rank < 10) {
            size_t out = (size_t)token * 10 + rank;
            weights[out] = renormalize ? full_div(selected[j], sum) : selected[j];
            ids[out] = (int)(0xffffffffu - (unsigned)key[j]);
            source[out] = rank * nt + token;
        }
    }
}

/* The native large-batch kernel uses two interleaved eight-value loads/lane,
 * libdevice expf, serial local sums and slot-ordered top-k renormalization. */
static __global__ void warp_topk(float *weights, int *ids, int *source,
    const __nv_bfloat16 *logits, int nt, int renormalize) {
    int lane = threadIdx.x & 31, token = blockIdx.x * 4 + threadIdx.x / 32;
    if (token >= nt) return;
    float value[16], maximum = -INFINITY;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        int expert = lane * 8 + j % 8 + (j / 8) * 256;
        value[j] = __bfloat162float(logits[(size_t)token * 512 + expert]);
        maximum = fmaxf(maximum, value[j]);
    }
    for (int d = 16; d; d >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, d));
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        value[j] = expf(__fsub_rn(value[j], maximum));
        sum = __fadd_rn(sum, value[j]);
    }
    for (int d = 16; d; d >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, d));
    float inverse = __fdiv_rn(1.0f, sum);
#pragma unroll
    for (int j = 0; j < 16; j++) {
        value[j] = __fmul_rn(value[j], inverse);
        if (!isfinite(value[j])) value[j] = 0;
    }
    float selected_sum = 0, out_weight = 0;
    int out_id = 0;
    for (int rank = 0; rank < 10; rank++) {
        float best = value[0];
        int best_id = lane * 8;
#pragma unroll
        for (int j = 1; j < 16; j++) {
            int expert = lane * 8 + j % 8 + (j / 8) * 256;
            if (value[j] > best) { best = value[j]; best_id = expert; }
        }
        for (int d = 16; d; d >>= 1) {
            float other = __shfl_xor_sync(0xffffffffu, best, d);
            int other_id = __shfl_xor_sync(0xffffffffu, best_id, d);
            if (other > best || (other == best && other_id < best_id)) { best = other; best_id = other_id; }
        }
        selected_sum = __fadd_rn(selected_sum, best);
        if (lane == rank) { out_weight = best; out_id = best_id; }
        if ((best_id % 256) / 8 == lane) value[(best_id / 256) * 8 + best_id % 8] = -INFINITY;
    }
    if (renormalize) out_weight = __fmul_rn(out_weight, __fdiv_rn(1.0f, selected_sum > 0 ? selected_sum : 1));
    if (lane < 10) {
        size_t out = (size_t)token * 10 + lane;
        weights[out] = out_weight; ids[out] = out_id; source[out] = lane * nt + token;
    }
}

static int topk(void *ctx_ptr, void *ptr, const BnGPUMoETopKBF16ResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability != 1200 ||
        p->n_tokens <= 0 || p->n_tokens > INT_MAX / 512 || p->n_experts != 512 || p->k != 10 ||
        (p->renormalize != 0 && p->renormalize != 1)) return -1;
    BnGPUResidentView v[] = {p->logits, p->weights, p->indices, p->source_rows};
    int types[] = {BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_I32, BN_GGUF_TENSOR_I32};
    void *data[4];
    for (int i = 0; i < 4; i++) {
        data[i] = view_data(r, v[i]);
        if (!data[i] || v[i].type != types[i] || v[i].count != (size_t)p->n_tokens * (i ? 10 : 512) ||
            ((uintptr_t)data[i] & 15u)) return -1;
        for (int j = 0; j < i; j++) if (overlap(v[i], v[j])) return -1;
    }
    if (set_device(r)) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    if (p->n_tokens <= 128)
        sorted_topk<<<p->n_tokens, 128, 0, r->stream>>>((float *)data[1], (int *)data[2], (int *)data[3],
            (const __nv_bfloat16 *)data[0], p->n_tokens, p->renormalize);
    else
        warp_topk<<<(p->n_tokens + 3) / 4, 128, 0, r->stream>>>((float *)data[1], (int *)data[2], (int *)data[3],
            (const __nv_bfloat16 *)data[0], p->n_tokens, p->renormalize);
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}
/* Only small routing metadata is scanned. Activations never leave the device.
 * Atomic cursor order within an expert is deliberately not part of the API;
 * the inverse map preserves original rank order for the final reduction. */
static __global__ void group_reset(int *cursors, int *status, int experts) {
    if(threadIdx.x<experts)cursors[threadIdx.x] = 0;
    if (!threadIdx.x) *status = 0;
}

static __global__ void group_count(const int *ids, int *cursors, int *status, int n, int experts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int expert = ids[i];
    if ((unsigned)expert >= (unsigned)experts) atomicOr(status, 1);
    else atomicAdd(cursors + expert, 1);
}

static __global__ void group_prefix(int *offsets, int *cursors,int experts) {
    __shared__ int sums[512];
    int i = threadIdx.x, count = i<experts?cursors[i]:0;
    sums[i] = count;
    __syncthreads();
    for (int d = 1; d < 512; d *= 2) {
        int other = i >= d ? sums[i - d] : 0;
        __syncthreads();
        sums[i] += other;
        __syncthreads();
    }
    if(i<experts){offsets[i] = sums[i] - count;cursors[i] = sums[i] - count;}
    if(i==experts-1)offsets[experts]=sums[i];
}

static __global__ void group_gather(const int *ids, const __nv_bfloat16 *x,
    int *cursors, int *slots, int *inverse, __nv_bfloat16 *gathered,
    const int *status, int k, int dim) {
    if (*status) return;
    int slot = blockIdx.x;
    __shared__ int row;
    if (!threadIdx.x) {
        row = atomicAdd(cursors + ids[slot], 1);
        slots[row] = slot;
        inverse[slot] = row;
    }
    __syncthreads();
    for (int col = threadIdx.x; col < dim; col += blockDim.x)
        gathered[(size_t)row * dim + col] = x[(size_t)(slot / k) * dim + col];
}

static bool views_validate(ResidentRequest *r, const BnGPUResidentView *v,
    const int *types, const size_t *counts, int n, void **data) {
    for (int i = 0; i < n; i++) {
        data[i] = view_data(r, v[i]);
        if (!data[i] || v[i].type != types[i] || v[i].count != counts[i] ||
            ((uintptr_t)data[i] & 15u)) return false;
        for (int j = 0; j < i; j++) if (overlap(v[i], v[j])) return false;
    }
    return true;
}

static int group(void *ctx_ptr, void *ptr, const BnGPUMoEGroupBF16ResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability != 1200 ||
        p->n_tokens <= 0 || p->n_experts<1||p->n_experts>512||p->k<1||p->k>32||p->k>p->n_experts||
        p->n_tokens > INT_MAX / p->k || p->dim <= 0 || p->dim > INT_MAX / (p->n_tokens * p->k)) return -1;
    size_t n = (size_t)p->n_tokens * p->k;
    BnGPUResidentView v[] = {p->indices, p->x, p->offsets, p->cursors, p->slots,
                            p->inverse, p->gathered, p->status};
    int types[] = {BN_GGUF_TENSOR_I32, BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_I32,
        BN_GGUF_TENSOR_I32, BN_GGUF_TENSOR_I32, BN_GGUF_TENSOR_I32,
        BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_I32};
    size_t counts[] = {n, (size_t)p->n_tokens * p->dim, (size_t)p->n_experts+1, (size_t)p->n_experts, n, n, n * p->dim, 1};
    void *data[8];
    if (!views_validate(r, v, types, counts, 8, data) || set_device(r)) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    group_reset<<<1, 512, 0, r->stream>>>((int *)data[3], (int *)data[7],p->n_experts);
    group_count<<<(unsigned)((n + 255) / 256), 256, 0, r->stream>>>((int *)data[0],
        (int *)data[3], (int *)data[7], (int)n,p->n_experts);
    group_prefix<<<1, 512, 0, r->stream>>>((int *)data[2], (int *)data[3],p->n_experts);
    group_gather<<<(unsigned)n, 256, 0, r->stream>>>((int *)data[0],
        (const __nv_bfloat16 *)data[1], (int *)data[3], (int *)data[4], (int *)data[5],
        (__nv_bfloat16 *)data[6], (const int *)data[7], p->k, p->dim);
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}

static __global__ void route_validate(const int *inverse, int *status, int n) {
    __shared__ int skip;
    if (!threadIdx.x) skip = atomicOr(status, 0);
    __syncthreads();
    if (skip) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && (unsigned)inverse[i] >= (unsigned)n) atomicOr(status, 2);
}

static __global__ void route_weight(const __nv_bfloat16 *grouped, const int *inverse,
    const float *weights, __nv_bfloat16 *weighted, const int *status, int n, int dim) {
    if (*status) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * dim) return;
    int slot = (int)(i / dim), row = inverse[slot];
    weighted[i] = __float2bfloat16_rn(__fmul_rn(
        __bfloat162float(grouped[(size_t)row * dim + i % dim]), weights[slot]));
}

static __global__ void route_sum(const __nv_bfloat16 *weighted,
    __nv_bfloat16 *out, const int *status, int nt, int dim,int k) {
    if (*status) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nt * dim) return;
    size_t base = (i / dim) * k * dim + i % dim;
    float sum = 0;
#pragma unroll
    for (int rank = 0; rank < k; rank++)
        sum = __fadd_rn(sum, __bfloat162float(weighted[base + (size_t)rank * dim]));
    out[i] = __float2bfloat16_rn(sum);
}

static int reduce(void *ctx_ptr, void *ptr, const BnGPUMoEReduceBF16ResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability != 1200 ||
        p->n_tokens <= 0 || p->k<1||p->k>32||p->n_tokens > INT_MAX / p->k ||
        p->dim <= 0 || p->dim > INT_MAX / (p->n_tokens * p->k)) return -1;
    size_t n = (size_t)p->n_tokens * p->k, rows = (size_t)p->n_tokens * p->dim;
    BnGPUResidentView v[] = {p->grouped, p->inverse, p->weights, p->weighted, p->out, p->status};
    int types[] = {BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_I32, BN_GGUF_TENSOR_F32,
                   BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_I32};
    size_t counts[] = {n * p->dim, n, n, n * p->dim, rows, 1};
    void *data[6];
    if (!views_validate(r, v, types, counts, 6, data) || set_device(r)) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    route_validate<<<(unsigned)((n + 255) / 256), 256, 0, r->stream>>>(
        (const int *)data[1], (int *)data[5], (int)n);
    route_weight<<<(unsigned)((n * p->dim + 255) / 256), 256, 0, r->stream>>>(
        (const __nv_bfloat16 *)data[0], (const int *)data[1], (const float *)data[2],
        (__nv_bfloat16 *)data[3], (int *)data[5], (int)n, p->dim);
    route_sum<<<(unsigned)((rows + 255) / 256), 256, 0, r->stream>>>(
        (const __nv_bfloat16 *)data[3], (__nv_bfloat16 *)data[4], (const int *)data[5],
        p->n_tokens, p->dim,p->k);
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}

struct Projection {
    const BnQuantFormatOps *format;
    const BnCudaBuffer *weight;
    const __nv_bfloat16 *x;
    __nv_bfloat16 *out;
    int rows, cols, nt, matrix;
    const BnCudaBlockQ8_1 *quantized_input;
};

static int project_validate(BnCudaCtx *ctx, ResidentRequest *r,
    const BnGPUMoEProjectBF16ResidentPlan *p, Projection *op) {
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability != 1200 ||
        p->n_tokens <= 0 || p->total_tokens < p->n_tokens || p->rows <= 0 || p->cols <= 0 ||
        p->n_tokens > INT_MAX / p->rows || p->n_tokens > INT_MAX / p->cols) return -1;
    const BnQuantFormatOps *format = bn_quant_format_ops(p->type);
    const BnCudaBuffer *w = (const BnCudaBuffer *)p->weight;
    if (!format || format->block_elems <= 0 || !format->bytes_per_block ||
        (format->bf16_moe < BN_QUANT_BF16_MOE_Q4_K || format->bf16_moe > BN_QUANT_BF16_MOE_Q6_K) ||
        p->cols % format->block_elems || !w || !w->data || w->type != p->type ||
        w->rows != p->rows || w->cols != p->cols ||
        (size_t)p->rows > SIZE_MAX / (size_t)(p->cols / format->block_elems) / format->bytes_per_block ||
        w->size < (size_t)p->rows * (p->cols / format->block_elems) * format->bytes_per_block) return -1;
    size_t nx = (size_t)p->n_tokens * p->cols, ny = (size_t)p->n_tokens * p->rows;
    void *x = view_data(r, p->x), *out = view_data(r, p->out);
    if (!x || !out || p->x.type != BN_GGUF_TENSOR_BF16 || p->out.type != BN_GGUF_TENSOR_BF16 ||
        p->x.count != nx || p->out.count != ny || overlap(p->x, p->out) ||
        (((uintptr_t)x | (uintptr_t)out) & 15u) || !r->linear_input ||
        nx / 32 > r->linear_input_bytes / sizeof(BnCudaBlockQ8_1)) return -1;
    op->format = format; op->weight = w;
    op->x = (const __nv_bfloat16 *)x; op->out = (__nv_bfloat16 *)out;
    op->rows = p->rows; op->cols = p->cols; op->nt = p->n_tokens;
    op->matrix = p->total_tokens > 64;
    op->quantized_input = NULL;
    return 0;
}

static int project_launch(ResidentRequest *r, const Projection *op, bool prepare) {
    const BnQuantFormatOps *format = op->format;
    const BnCudaBuffer *w = op->weight;
    const __nv_bfloat16 *x = op->x;
    __nv_bfloat16 *out = op->out;
    int rows = op->rows, cols = op->cols, nt = op->nt;
    size_t nx = (size_t)nt * cols;
    const BnCudaBlockQ8_1 *input = op->quantized_input ? op->quantized_input : r->linear_input;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    if (prepare)
        quantize_bf16_q8_affine_batch_kernel<<<(unsigned)(nx / 32), 32, 0, r->stream>>>(
            r->linear_input, x);
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    __nv_bfloat16 *dest = (__nv_bfloat16 *)out;
    if (!op->matrix) {
        dim3 grid(rows, nt);
        switch (format->bf16_moe) {
        case BN_QUANT_BF16_MOE_Q4_K:
            q4k_bf16_mmvq_kernel<<<grid, 32, 0, r->stream>>>(dest, (const BnBlockQ4K *)w->data, input, rows, cols); break;
        case BN_QUANT_BF16_MOE_Q5_K:
            q5k_bf16_mmvq_kernel<<<grid, 32, 0, r->stream>>>(dest, (const BnBlockQ5K *)w->data, input, rows, cols); break;
        case BN_QUANT_BF16_MOE_Q6_K:
            q6k_bf16_mmvq_kernel<<<grid, 32, 0, r->stream>>>(dest, (const BnBlockQ6K *)w->data, input, rows, cols); break;
        case BN_QUANT_BF16_MOE_Q5_1:
            q5_1_bf16_mmvq_kernel<<<grid, 32, 0, r->stream>>>(dest, (const BnBlockQ5_1 *)w->data, input, rows, cols); break;
        case BN_QUANT_BF16_MOE_Q8_0:
            q8_bf16_mmvq_kernel<<<grid, 32, 0, r->stream>>>(dest, (const BnBlockQ8_0 *)w->data, input, rows, cols); break;
        default: return -1;
        }
    } else {
        unsigned tiles = (unsigned)((((size_t)rows + 127) / 128) * (((size_t)nt + 3) / 4));
        switch (format->bf16_moe) {
        case BN_QUANT_BF16_MOE_Q4_K:
            q4k_bf16_mmq_kernel<<<tiles, 128, 0, r->stream>>>(dest, (const BnBlockQ4K *)w->data, input, rows, cols, nt); break;
        case BN_QUANT_BF16_MOE_Q5_K:
            q5k_bf16_mmq_kernel<<<tiles, 128, 0, r->stream>>>(dest, (const BnBlockQ5K *)w->data, input, rows, cols, nt); break;
        case BN_QUANT_BF16_MOE_Q6_K:
            q6k_bf16_mmq_kernel<<<tiles, 128, 0, r->stream>>>(dest, (const BnBlockQ6K *)w->data, input, rows, cols, nt); break;
        case BN_QUANT_BF16_MOE_Q5_1:
            q5_1_bf16_mmq_kernel<<<tiles, 128, 0, r->stream>>>(dest, (const BnBlockQ5_1 *)w->data, input, rows, cols, nt); break;
        case BN_QUANT_BF16_MOE_Q8_0:
            q8_bf16_mmq_kernel<<<tiles, 128, 0, r->stream>>>(dest, (const BnBlockQ8_0 *)w->data, input, rows, cols, nt); break;
        default: return -1;
        }
    }
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}
static int project(void *ctx_ptr, void *ptr, const BnGPUMoEProjectBF16ResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    Projection op;
    if (project_validate(ctx, r, p, &op) || set_device(r)) return -1;
    return project_launch(r, &op, true);
}

static __global__ void clamped_split_silu(__nv_bfloat16*gate,const __nv_bfloat16*up,size_t n,float limit) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float g=fminf(__bfloat162float(gate[i]),limit),u=fmaxf(-limit,fminf(__bfloat162float(up[i]),limit));
    float s=__bfloat162float(__float2bfloat16_rn(g/(1.f+expf(-g))));
    gate[i]=__float2bfloat16_rn(__fmul_rn(s,__fadd_rn(u,0.f)));
}
static int ffn_launch(ResidentRequest *r, const Projection *ops,int clamped=0,float limit=0) {
    if (project_launch(r, &ops[0], ops[0].quantized_input == NULL) || project_launch(r, &ops[1], false)) return -1;
    size_t full = (size_t)ops[0].nt * ops[0].rows;
    if(clamped)clamped_split_silu<<<(unsigned)((full+255)/256),256,0,r->stream>>>((__nv_bfloat16*)ops[0].out,ops[1].out,full,limit);
    else silu_mul_bf16_kernel<<<(unsigned)((full + 255) / 256), 256, 0, r->stream>>>(
        (__nv_bfloat16 *)ops[0].out, ops[1].out, full);
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    return project_launch(r, &ops[2], true);
}

static int ffn(void *ctx_ptr, void *ptr, const BnGPUMoEFFNBF16ResidentPlan *plan) {
    if (!plan || plan->ffn.activation_profile != BN_GPU_BF16_FFN_SILU_SEPARATE) return -1;
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    const BnGPUFFNResidentPlan *p = &plan->ffn;
    BnGPUResidentView views[] = {p->x, p->gate, p->up, p->out};
    BnGPUBF16LinearWeight weights[] = {p->weights.gate, p->weights.up, p->weights.down};
    Projection ops[3];
    for (int i = 0; i < 3; i++) {
        BnGPUMoEProjectBF16ResidentPlan linear = {};
        linear.n_tokens = p->n_tokens; linear.total_tokens = plan->total_tokens;
        linear.rows = i == 2 ? p->dim : p->hidden_dim;
        linear.cols = i == 2 ? p->hidden_dim : p->dim;
        linear.type = weights[i].type; linear.weight = weights[i].raw;
        linear.x = i == 2 ? p->gate : p->x;
        linear.out = views[i + 1];
        if (project_validate(ctx, r, &linear, &ops[i])) return -1;
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < i; j++) if (overlap(views[i], views[j])) return -1;
    if (set_device(r)) return -1;
    return ffn_launch(r, ops);
}

#include "gpu_cuda_moe_grouped.cuh"
#include "gpu_cuda_moe_joint.cuh"

struct ExpertCache;
struct BankSlot {
    void *host, *device;
    cudaEvent_t ready, completed;
    BnCudaBuffer *weights;
    Projection *ops;
    BnGPUMoEBankLease lease;
    int state; /* 0 idle, 1 acquired, 2 submitted until COMPUTE completes. */
    BnGPUMoEBankItem *host_items, *device_items;
    const unsigned char **host_raw, **device_raw;
};
struct BankState {
    int device;
    BnGPUMoEBankConfig config;
    BnGPUMoEBankStats stats;
    size_t max_expert_bytes;
    cudaStream_t copy_stream;
    BankSlot *slots;
    BnCudaBlockQ8_1 *input, *grouped_input;
    int input_capacity, input_tokens;
    size_t input_budget;
    uint64_t input_generation;
    BnCudaBlockQ8_1 *grouped_down;
    int grouped_capacity;
    size_t grouped_host_budget, grouped_device_budget;
    ResidentRequest *request;
    ExpertCache *cache;
    BnGPUExpertCacheLayer cache_layer;
    BnGPUExpertCacheStats cache_stats;
    int cache_open, cache_prefill, cache_nt, cache_entries[512], cache_seen[512], cache_offsets[513];
    ResidentRequest *cache_group_owner, *cache_group_requests[16];
    int cache_group_count;
    uint64_t cache_group_decode[8];
};

static size_t saturated_add(size_t a, size_t b) { return b > SIZE_MAX - a ? SIZE_MAX : a + b; }

static int bank_pair_bytes(int dim, int hidden, int gate_type, int down_type,
    size_t *gate_bytes, size_t *down_bytes, size_t *expert_bytes) {
    const BnQuantFormatOps *g = bn_quant_format_ops(gate_type), *d = bn_quant_format_ops(down_type);
    if (!g || !d || (g->bf16_moe != BN_QUANT_BF16_MOE_Q4_K && g->bf16_moe != BN_QUANT_BF16_MOE_Q5_K) ||
        (d->bf16_moe != BN_QUANT_BF16_MOE_Q5_1 && d->bf16_moe != BN_QUANT_BF16_MOE_Q8_0 && d->bf16_moe != BN_QUANT_BF16_MOE_Q5_K && d->bf16_moe != BN_QUANT_BF16_MOE_Q6_K) ||
        dim <= 0 || hidden <= 0 || g->block_elems <= 0 || d->block_elems <= 0 ||
        !g->bytes_per_block || !d->bytes_per_block || dim % g->block_elems || hidden % d->block_elems ||
        (size_t)dim > SIZE_MAX / (size_t)hidden) return -1;
    size_t elems = (size_t)dim * hidden;
    if (elems / g->block_elems > SIZE_MAX / g->bytes_per_block ||
        elems / d->block_elems > SIZE_MAX / d->bytes_per_block) return -1;
    *gate_bytes = elems / g->block_elems * g->bytes_per_block;
    *down_bytes = elems / d->block_elems * d->bytes_per_block;
    if (*gate_bytes > (SIZE_MAX - *down_bytes) / 2) return -1;
    *expert_bytes = 2 * *gate_bytes + *down_bytes;
    return 0;
}

#include "gpu_cuda_expert_cache.cuh"

/* Called by request teardown after its compute stream has completed, or by a
 * failed reserve before any work was submitted. Carries its own CUDA context. */
static void bank_destroy(void *ptr) {
    BankState *b = (BankState *)ptr;
    if (!b) return;
    if (b->cache_open||b->cache_group_owner) (void)expert_cache_cancel(b->request);
    if (b->cache) b->cache->bindings--;
    int previous = -1;
    (void)cudaGetDevice(&previous);
    if (cudaSetDevice(b->device) == cudaSuccess) {
        if (b->copy_stream) (void)cudaStreamSynchronize(b->copy_stream);
        if (b->slots) for (unsigned i = 0; i < b->config.slots; i++) {
            BankSlot *s = &b->slots[i];
            if (s->completed) cudaEventDestroy(s->completed);
            if (s->ready) cudaEventDestroy(s->ready);
            if (s->device) cudaFree(s->device);
            if (s->host) cudaFreeHost(s->host);
            if (s->host_items) cudaFreeHost(s->host_items);
            if (s->device_items) cudaFree(s->device_items);
            if (s->host_raw) cudaFreeHost(s->host_raw);
            if (s->device_raw) cudaFree(s->device_raw);
        }
        if (b->copy_stream) cudaStreamDestroy(b->copy_stream);
        cudaFree(b->input); cudaFree(b->grouped_input);
        cudaFree(b->grouped_down);
    }
    if (b->slots) for (unsigned i = 0; i < b->config.slots; i++) {
        free(b->slots[i].weights); free(b->slots[i].ops);
    }
    free(b->slots); free(b);
    if (previous >= 0) (void)cudaSetDevice(previous);
}

static int bank_reserve(void *ctx_ptr, void *ptr, const BnGPUMoEBankConfig *c) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    size_t gb, db, stride;
    if(!c)return -1;
    BnGPUMoEBankConfig normalized=*c;
    if(!normalized.n_experts)normalized.n_experts=512;
    if(!normalized.k)normalized.k=10;
    c=&normalized;
    if(c->n_experts<1||c->n_experts>512||c->k<1||c->k>32||c->k>c->n_experts)return -1;
    if (!ctx || !r || !c || r->backend_owner != ctx || ctx->compute_capability != 1200 ||
        !c->slots || c->slots > 4 || !c->experts_per_slot || c->experts_per_slot > 64 ||
        c->dim <= 0 || c->hidden_dim <= 0 || c->dim % 256 || c->hidden_dim % 32 ||
        bank_pair_bytes(c->dim, c->hidden_dim, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0, &gb, &db, &stride) ||
        stride > SIZE_MAX / c->experts_per_slot / c->slots) return -1;
    size_t slot_bytes = stride * c->experts_per_slot, payload = slot_bytes * c->slots;
    if (payload > c->host_budget_bytes || payload > c->device_budget_bytes) return -1;
    BankState *b = (BankState *)r->moe_banks;
    if (b) {
        const BnGPUMoEBankConfig *old = &b->config;
        return old->slots == c->slots && old->experts_per_slot == c->experts_per_slot &&
            old->dim == c->dim && old->hidden_dim == c->hidden_dim &&
            old->n_experts==c->n_experts&&old->k==c->k&&
            old->host_budget_bytes == c->host_budget_bytes && old->device_budget_bytes == c->device_budget_bytes ? 0 : -1;
    }
    if (set_device(r)) return -1;
    b = (BankState *)calloc(1, sizeof(*b));
    if (!b) return -1;
    b->request = r; b->device = r->device; b->config = *c; b->max_expert_bytes = stride;
    b->stats.reserved_host_bytes = b->stats.reserved_device_bytes = payload;
    b->slots = (BankSlot *)calloc(c->slots, sizeof(BankSlot));
    if (!b->slots || cudaStreamCreateWithFlags(&b->copy_stream, cudaStreamNonBlocking) != cudaSuccess) goto fail;
    for (unsigned i = 0; i < c->slots; i++) {
        BankSlot *s = &b->slots[i];
        s->weights = (BnCudaBuffer *)calloc(c->experts_per_slot * 3, sizeof(BnCudaBuffer));
        s->ops = (Projection *)calloc(c->experts_per_slot * 3, sizeof(Projection));
        if (!s->weights || !s->ops || cudaMallocHost(&s->host, slot_bytes) != cudaSuccess ||
            cudaMalloc(&s->device, slot_bytes) != cudaSuccess ||
            cudaEventCreateWithFlags(&s->ready, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&s->completed, cudaEventDisableTiming) != cudaSuccess) goto fail;
    }
    r->moe_banks = b; r->moe_banks_destroy = bank_destroy;
    return 0;
fail:
    bank_destroy(b);
    (void)cudaGetLastError();
    return -1;
}

static int bank_acquire(void *ptr, unsigned slot, int gate_type, int down_type, BnGPUMoEBankLease *out) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    BankState *b = r ? (BankState *)r->moe_banks : NULL;
    size_t gb, db, stride;
    if (!b || !out || slot >= b->config.slots ||
        bank_pair_bytes(b->config.dim, b->config.hidden_dim, gate_type, down_type, &gb, &db, &stride) ||
        stride > b->max_expert_bytes || set_device(r)) return -1;
    BankSlot *s = &b->slots[slot];
    if (s->state == 1) return -1;
    if (s->state == 2) {
        cudaError_t ready = cudaEventQuery(s->completed);
        if (ready == cudaErrorNotReady) {
            cudaError_t last = cudaGetLastError();
            if (last != cudaSuccess && last != cudaErrorNotReady) { r->failed = 1; return -1; }
            b->stats.busy_acquires = saturated_add(b->stats.busy_acquires, 1);
            return BN_GPU_MOE_BANK_BUSY;
        }
        if (ready != cudaSuccess) { r->failed = 1; return -1; }
        s->state = 0;
    }
    if (s->lease.generation == UINT64_MAX) return -1;
    BnGPUMoEBankLease lease = {};
    lease.request = r; lease.data = s->host; lease.slot = slot; lease.capacity = b->config.experts_per_slot;
    lease.generation = s->lease.generation + 1; lease.gate_type = gate_type; lease.down_type = down_type;
    lease.gate_bytes = gb; lease.down_bytes = db; lease.expert_bytes = stride;
    s->lease = lease; s->state = 1; *out = lease;
    return 0;
}

static BankSlot *bank_lease(ResidentRequest *r, const BnGPUMoEBankLease *l) {
    BankState *b = r ? (BankState *)r->moe_banks : NULL;
    if (!b || !l || l->request != r || l->slot >= b->config.slots) return NULL;
    BankSlot *s = &b->slots[l->slot];
    const BnGPUMoEBankLease *old = &s->lease;
    return s->state == 1 && l->data == old->data && l->generation == old->generation &&
        l->capacity == old->capacity && l->gate_type == old->gate_type && l->down_type == old->down_type &&
        l->gate_bytes == old->gate_bytes && l->down_bytes == old->down_bytes &&
        l->expert_bytes == old->expert_bytes ? s : NULL;
}

static int bank_cancel(void *ptr, const BnGPUMoEBankLease *lease) {
    BankSlot *s = bank_lease((ResidentRequest *)ptr, lease);
    if (!s) return -1;
    s->state = 0;
    return 0;
}

static int bank_submit(void *ctx_ptr, void *ptr, const BnGPUMoEBankResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability != 1200||
        (p->clamped!=0&&p->clamped!=1)||(p->clamped&&(!isfinite(p->clamp_limit)||p->clamp_limit<0))) return -1;
    BankSlot *s = bank_lease(r, &p->lease);
    BankState *b = (BankState *)r->moe_banks;
    if (!s || !p->experts || p->n_experts <= 0 || (unsigned)p->n_experts > b->config.experts_per_slot ||
        p->total_tokens <= 0 || p->total_tokens > INT_MAX / b->config.k || (p->grouped != 0 && p->grouped != 1)) return -1;
    if (p->prepared_input && (p->prepared_input != b->input_generation ||
        p->total_tokens != b->input_tokens || !b->grouped_input)) return -1;
    if (p->grouped && (!p->prepared_input || !b->grouped_down ||
        p->total_tokens > b->grouped_capacity || !s->host_items || !s->device_items)) return -1;
    if(b->cache_group_owner)return -1;
    if (b->cache_open && (!p->grouped || p->total_tokens != b->cache_nt ||
        b->cache->failed || p->lease.gate_type != b->cache->gate_type[b->cache_layer.layer] ||
        p->lease.down_type != b->cache->down_type[b->cache_layer.layer])) return -1;
    int dim = b->config.dim, hidden = b->config.hidden_dim, n = p->total_tokens * b->config.k;
    if (dim > INT_MAX / n || hidden > INT_MAX / n) return -1;
    BnGPUResidentView v[] = {p->x, p->gate, p->up, p->out};
    int types[] = {BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_BF16};
    size_t counts[] = {(size_t)n * dim, (size_t)n * hidden, (size_t)n * hidden, (size_t)n * dim};
    void *data[4];
    if (!views_validate(r, v, types, counts, 4, data)) return -1;
    int max_rows = 0;
    for (int i = 0; i < p->n_experts; i++) {
        const BnGPUMoEBankItem *item = &p->experts[i];
        if ((unsigned)item->expert >= (unsigned)b->config.n_experts || item->n_rows <= 0 || item->n_rows > p->total_tokens ||
            item->first_row < 0 || item->first_row > n - item->n_rows) return -1;
        if (b->cache_open && (b->cache_seen[item->expert] ||
            item->first_row != b->cache_offsets[item->expert] ||
            item->n_rows != b->cache_offsets[item->expert+1]-b->cache_offsets[item->expert])) return -1;
        if (item->n_rows > max_rows) max_rows = item->n_rows;
        for (int j = 0; j < i; j++) {
            const BnGPUMoEBankItem *other = &p->experts[j];
            if (other->expert == item->expert || (item->first_row < other->first_row + other->n_rows &&
                other->first_row < item->first_row + item->n_rows)) return -1;
        }
        unsigned char *raw = (unsigned char *)s->device + (size_t)i * p->lease.expert_bytes;
        for (int role = 0; role < 3; role++) {
            BnCudaBuffer *w = &s->weights[i * 3 + role];
            w->data = raw + (role == 2 ? 2 * p->lease.gate_bytes : role * p->lease.gate_bytes);
            w->size = role == 2 ? p->lease.down_bytes : p->lease.gate_bytes;
            w->type = role == 2 ? p->lease.down_type : p->lease.gate_type;
            w->rows = role == 2 ? dim : hidden; w->cols = role == 2 ? hidden : dim;
            BnGPUMoEProjectBF16ResidentPlan linear = {};
            linear.n_tokens = item->n_rows; linear.total_tokens = p->total_tokens;
            linear.rows = w->rows; linear.cols = w->cols; linear.type = w->type; linear.weight = w;
            linear.x = bn_gpu_resident_view_slice(role == 2 ? p->gate : p->x,
                (size_t)item->first_row * w->cols, (size_t)item->n_rows * w->cols);
            linear.out = bn_gpu_resident_view_slice(v[role + 1],
                (size_t)item->first_row * w->rows, (size_t)item->n_rows * w->rows);
            if (project_validate(ctx, r, &linear, &s->ops[i * 3 + role])) return -1;
            if (p->prepared_input && role != 2)
                s->ops[i * 3 + role].quantized_input = b->grouped_input + (size_t)item->first_row * (dim / 32);
        }
    }
    if (set_device(r)) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    if (p->grouped) memcpy(s->host_items, p->experts, (size_t)p->n_experts * sizeof(BnGPUMoEBankItem));
    s->state = 2; /* Also prevents reuse after a partially queued CUDA failure. */
    if (b->cache_open) {
        if(expert_cache_upload_bank(b,s,&p->lease,p->experts,p->n_experts))goto failed;
    } else {
        size_t bytes=(size_t)p->n_experts*p->lease.expert_bytes;
        if(cudaMemcpyAsync(s->device,s->host,bytes,cudaMemcpyHostToDevice,b->copy_stream)!=cudaSuccess)goto failed;
        b->stats.upload_bytes=saturated_add(b->stats.upload_bytes,bytes);
    }
    if (p->grouped) {
        size_t metadata = (size_t)p->n_experts * sizeof(BnGPUMoEBankItem);
        if (cudaMemcpyAsync(s->device_items, s->host_items, metadata, cudaMemcpyHostToDevice, b->copy_stream) != cudaSuccess) goto failed;
        b->stats.metadata_upload_bytes = saturated_add(b->stats.metadata_upload_bytes, metadata);
    }
    if (cudaEventRecord(s->ready, b->copy_stream) != cudaSuccess ||
        cudaStreamWaitEvent(r->stream, s->ready, 0) != cudaSuccess) goto failed;
    if (p->grouped) {
        GroupedBatch batch = {};
        batch.raw_ptrs = b->cache_open ? s->device_raw : NULL;
        batch.raw = (const unsigned char *)s->device; batch.stride = p->lease.expert_bytes;
        batch.gate_bytes = p->lease.gate_bytes; batch.items = s->device_items;
        batch.input = b->grouped_input; batch.down_input = b->grouped_down;
        batch.gate = (__nv_bfloat16 *)data[1]; batch.up = (__nv_bfloat16 *)data[2]; batch.out = (__nv_bfloat16 *)data[3];
        batch.dim = dim; batch.hidden = hidden; batch.max_rows = max_rows;
        batch.clamped=p->clamped;batch.clamp_limit=p->clamp_limit;
        batch.experts = p->n_experts; batch.matrix = p->total_tokens > 64;
        if (grouped_execute(r, &batch, s->ops[0].format->bf16_moe, s->ops[2].format->bf16_moe)) goto failed;
    } else for (int i = 0; i < p->n_experts; i++) if (ffn_launch(r, &s->ops[i * 3],p->clamped,p->clamp_limit)) goto failed;
    if (cudaEventRecord(s->completed, r->stream) != cudaSuccess) goto failed;
    if(b->cache_open)for(int i=0;i<p->n_experts;i++) {
        int expert=p->experts[i].expert,slot=b->cache_entries[expert];
        b->cache_seen[expert]=1;if(slot>=0)b->cache->entry[slot].valid=1;
    }
    b->stats.submitted_banks = saturated_add(b->stats.submitted_banks, 1);
    b->stats.submitted_experts = saturated_add(b->stats.submitted_experts, (size_t)p->n_experts);
    if (p->grouped) b->stats.grouped_banks = saturated_add(b->stats.grouped_banks, 1);
    return 0;
failed:
    if(b->cache)b->cache->failed=1;
    r->failed = 1;
    return -1;
}

#include "gpu_cuda_moe_shared.cuh"

static int bank_stats(void *ptr, BnGPUMoEBankStats *out) {
    ResidentRequest *r = (ResidentRequest *)ptr;
    BankState *b = r ? (BankState *)r->moe_banks : NULL;
    if (!b || !out) return -1;
    *out = b->stats;
    return 0;
}

static int bank_grouped_reserve(void *ctx_ptr, void *ptr, int max_tokens, size_t host_budget, size_t device_budget) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    BankState *b = r ? (BankState *)r->moe_banks : NULL;
    if (!ctx || !r || !b || r->backend_owner != ctx || max_tokens <= 0 ||
        max_tokens > INT_MAX / b->config.k / b->config.dim || max_tokens > INT_MAX / b->config.k / b->config.hidden_dim) return -1;
    size_t metadata = (size_t)b->config.experts_per_slot * sizeof(BnGPUMoEBankItem);
    size_t all_metadata = metadata * b->config.slots;
    size_t blocks = (size_t)max_tokens * b->config.k * (b->config.hidden_dim / 32);
    if (blocks > (SIZE_MAX - all_metadata) / sizeof(BnCudaBlockQ8_1)) return -1;
    size_t down_bytes = blocks * sizeof(BnCudaBlockQ8_1);
    if (all_metadata > host_budget || down_bytes + all_metadata > device_budget) return -1;
    if (b->grouped_down) return b->grouped_capacity == max_tokens && b->grouped_host_budget == host_budget &&
        b->grouped_device_budget == device_budget ? 0 : -1;
    if (set_device(r)) return -1;
    BnGPUMoEBankItem *host[4] = {}, *device[4] = {};
    BnCudaBlockQ8_1 *down = NULL;
    if (cudaMalloc((void **)&down, down_bytes) != cudaSuccess) goto fail;
    for (unsigned i = 0; i < b->config.slots; i++) {
        if (cudaMallocHost((void **)&host[i], metadata) != cudaSuccess ||
            cudaMalloc((void **)&device[i], metadata) != cudaSuccess) goto fail;
    }
    for (unsigned i = 0; i < b->config.slots; i++) {
        b->slots[i].host_items = host[i]; b->slots[i].device_items = device[i];
    }
    b->grouped_down = down; b->grouped_capacity = max_tokens;
    b->grouped_host_budget = host_budget; b->grouped_device_budget = device_budget;
    b->stats.reserved_group_host_bytes = all_metadata;
    b->stats.reserved_group_device_bytes = down_bytes + all_metadata;
    return 0;
fail:
    for (unsigned i = 0; i < b->config.slots; i++) {
        if (host[i]) cudaFreeHost(host[i]);
        if (device[i]) cudaFree(device[i]);
    }
    cudaFree(down); (void)cudaGetLastError();
    return -1;
}

static int bank_input_reserve(void *ctx_ptr, void *ptr, int max_tokens, size_t budget) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    BankState *b = r ? (BankState *)r->moe_banks : NULL;
    if (!ctx || !r || !b || r->backend_owner != ctx || max_tokens <= 0 ||
        max_tokens > INT_MAX / b->config.k / b->config.dim) return -1;
    size_t blocks = (size_t)max_tokens * (b->config.dim / 32);
    if (blocks > SIZE_MAX / sizeof(BnCudaBlockQ8_1) / (b->config.k+1)) return -1;
    size_t bytes = blocks * sizeof(BnCudaBlockQ8_1);
    if (bytes * (b->config.k+1) > budget) return -1;
    if (b->input) return b->input_capacity == max_tokens && b->input_budget == budget ? 0 : -1;
    if (set_device(r)) return -1;
    BnCudaBlockQ8_1 *original = NULL, *grouped = NULL;
    if (cudaMalloc((void **)&original, bytes) != cudaSuccess ||
        cudaMalloc((void **)&grouped, bytes * b->config.k) != cudaSuccess) {
        cudaFree(original); cudaFree(grouped); (void)cudaGetLastError(); return -1;
    }
    b->input = original; b->grouped_input = grouped;
    b->input_capacity = max_tokens; b->input_budget = budget;
    b->stats.reserved_input_bytes = bytes * (b->config.k+1);
    return 0;
}

static __global__ void bank_map_input(BnCudaBlockQ8_1 *out, const BnCudaBlockQ8_1 *input,
    const int *slots, const int *status, size_t blocks, int row_blocks,int k) {
    if (*status) return;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < blocks) {
        size_t row = i / row_blocks;
        out[i] = input[(size_t)(slots[row] / k) * row_blocks + i % row_blocks];
    }
}

static int bank_prepare_input(void *ctx_ptr, void *ptr, const BnGPUMoEBankInputPlan *p, uint64_t *generation) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    BankState *b = r ? (BankState *)r->moe_banks : NULL;
    if (!ctx || !r || !b || !p || !generation || r->backend_owner != ctx || !b->input ||
        p->n_tokens <= 0 || p->n_tokens > b->input_capacity || b->input_generation == UINT64_MAX) return -1;
    size_t nx = (size_t)p->n_tokens * b->config.dim, slots = (size_t)p->n_tokens * b->config.k;
    BnGPUResidentView v[] = {p->x, p->slots, p->status};
    int types[] = {BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_I32, BN_GGUF_TENSOR_I32};
    size_t counts[] = {nx, slots, 1};
    void *data[3];
    if (!views_validate(r, v, types, counts, 3, data) || set_device(r)) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    route_validate<<<(unsigned)((slots + 255) / 256), 256, 0, r->stream>>>(
        (const int *)data[1], (int *)data[2], (int)slots);
    quantize_bf16_q8_affine_batch_kernel<<<(unsigned)(nx / 32), 32, 0, r->stream>>>(
        b->input, (const __nv_bfloat16 *)data[0]);
    size_t blocks = nx / 32 * b->config.k;
    bank_map_input<<<(unsigned)((blocks + 255) / 256), 256, 0, r->stream>>>(
        b->grouped_input, b->input, (const int *)data[1], (const int *)data[2], blocks, b->config.dim / 32,b->config.k);
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    b->input_tokens = p->n_tokens;
    b->stats.input_preparations = saturated_add(b->stats.input_preparations, 1);
    *generation = ++b->input_generation;
    return 0;
}
static __global__ void shared_kernel(__nv_bfloat16 *out, __nv_bfloat16 *shared_out,
    __nv_bfloat16 *sigmoid_out, const __nv_bfloat16 *routed, const __nv_bfloat16 *down,
    const __nv_bfloat16 *logits, int dim) {
    size_t token = blockIdx.x, base = token * (size_t)dim;
    __shared__ float weight;
    if (!threadIdx.x) {
        weight = bn_cuda_hc::sigmoid_bf16(__bfloat162float(logits[token]));
        sigmoid_out[token] = __float2bfloat16_rn(weight);
    }
    __syncthreads();
    for (size_t col = threadIdx.x; col < (size_t)dim; col += blockDim.x) {
        size_t i = base + col;
        __nv_bfloat16 value = __float2bfloat16_rn(__fmul_rn(weight, __bfloat162float(down[i])));
        shared_out[i] = value;
        out[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(routed[i]), __bfloat162float(value)));
    }
}

static int shared(void *ctx_ptr, void *ptr, const BnGPUMoESharedBF16ResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability != 1200 ||
        p->n_tokens <= 0 || p->dim <= 0 || p->n_tokens > INT_MAX / p->dim) return -1;
    size_t count = (size_t)p->n_tokens * p->dim, nt = (size_t)p->n_tokens;
    BnGPUResidentView v[] = {p->routed, p->down, p->logits, p->sigmoid, p->shared, p->out};
    int types[] = {BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_BF16,
                   BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_BF16};
    size_t counts[] = {count, count, nt, nt, count, count};
    void *data[6];
    if (!views_validate(r, v, types, counts, 6, data) || set_device(r)) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    shared_kernel<<<p->n_tokens, 256, 0, r->stream>>>((__nv_bfloat16 *)data[5],
        (__nv_bfloat16 *)data[4], (__nv_bfloat16 *)data[3], (const __nv_bfloat16 *)data[0],
        (const __nv_bfloat16 *)data[1], (const __nv_bfloat16 *)data[2], p->dim);
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}
} // namespace bn_cuda_moe_resident
#endif
