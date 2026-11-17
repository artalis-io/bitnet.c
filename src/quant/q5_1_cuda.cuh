#ifndef BN_QUANT_Q5_1_CUDA_CUH
#define BN_QUANT_Q5_1_CUDA_CUH

/* Routed affine block32 MMVQ. The backend supplies borrowed buffers and chooses
 * one warp for multi-token routing, four for a single token. BnCudaBlockQ8Mmq
 * stores the original activation sum; BnCudaBlockQ8_1.qsum is not interchangeable. */
static __global__ void q5_1_quantize_mmvq_input_kernel(
    BnCudaBlockQ8Mmq *out, const float *input) {
    int block = blockIdx.x, lane = threadIdx.x;
    float value = input[(size_t)block * 32 + lane];
    float amax = fabsf(value), sum = value;
    for (int offset = 16; offset; offset /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    float d = __fdividef(amax, 127.0f);
    out[block].qs[lane] = d == 0.0f ? 0 : (int8_t)roundf(__fdividef(value, d));
    if (!lane) {
        out[block].d = cuda_fp32_to_fp16_bits(d);
        out[block].original_sum = cuda_fp32_to_fp16_bits(sum);
    }
}

static __device__ __forceinline__ float q5_1_mmvq_dot(
    const BnBlockQ5_1 *weight, const BnCudaBlockQ8Mmq *input, int part) {
    uint32_t high;
    memcpy(&high, weight->qh, sizeof(high));
    int dot = 0;
    for (int j = part * 8; j < part * 8 + 8; j++) {
        int lo = (weight->qs[j] & 15) | (((high >> j) & 1u) << 4);
        int hi = (weight->qs[j] >> 4) | (((high >> (j + 16)) & 1u) << 4);
        dot += lo * input->qs[j] + hi * input->qs[j + 16];
    }
    float scale = cuda_fp16_to_fp32(cuda_fp32_to_fp16_bits(
        cuda_fp16_to_fp32(weight->d) * cuda_fp16_to_fp32(input->d)));
    float offset = cuda_fp16_to_fp32(cuda_fp32_to_fp16_bits(
        cuda_fp16_to_fp32(weight->m) * cuda_fp16_to_fp32(input->original_sum)));
    return fmaf((float)dot, scale, offset * 0.5f);
}

static __global__ void q5_1_routed_mmvq_kernel(
    float *out, const BnBlockQ5_1 *weights, const BnCudaBlockQ8Mmq *input,
    const int *ids, int rows, int cols) {
    __shared__ float partial[3][32];
    int row = blockIdx.x, item = blockIdx.y, tid = threadIdx.x;
    int lane = tid & 31, warp = tid / 32, blocks = cols / 32;
    float sum = 0.0f;
    for (int b = tid / 2; b < blocks; b += blockDim.x / 2) {
        const BnBlockQ5_1 *weight = weights +
            ((size_t)(ids ? ids[item] : 0) * rows + row) * blocks + b;
        sum += q5_1_mmvq_dot(weight, input + (size_t)item * blocks + b, tid & 1);
    }
    if (warp) partial[warp - 1][lane] = sum;
    __syncthreads();
    if (!warp) {
        for (int i = 0; i < (int)blockDim.x / 32 - 1; i++) sum += partial[i][lane];
        for (int offset = 16; offset; offset /= 2)
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        if (!lane) out[(size_t)item * rows + row] = sum;
    }
}
/* MMQ uses a reciprocal scale and original activation sums. Its two FMA
 * updates per block differ from the half-product MMVQ contract above. */
static __global__ void q5_1_routed_mmq_kernel(
    float *out, const BnBlockQ5_1 *weights, const BnCudaBlockQ8Mmq *input,
    const int *map, int rows, int cols, int nt, int experts, int k,
    int jwidth, int grid, int borrowed_expert = -1) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int item = blockIdx.y;
    if (row >= rows) return;
    int expert = borrowed_expert >= 0 ? borrowed_expert : map[item];
    int rank = borrowed_expert >= 0 ? item : map[nt * k + item];
    int nb = cols / 32, xt = (nt - 1) / jwidth + 1;
    int64_t tiles = (int64_t)((rows - 1) / 128 + 1) * xt * experts;
    int64_t tile = ((int64_t)(row / 128) * experts + expert) * xt + rank / jwidth;
    int64_t base = tile * nb, total = tiles * nb;
    int first = grid == tiles ? (int)tile : (int)(base * grid / total) - 2;
    int last = grid == tiles ? (int)tile : (int)((base + nb + 8) * grid / total) + 2;
    if (first < 0) first = 0;
    if (last >= grid) last = grid - 1;
    float tail = 0.0f, prefix = 0.0f;
    bool have = false;
    for (int bid = last; bid >= first; bid--) {
        int64_t begin = (int64_t)bid * total / grid, end = (int64_t)(bid + 1) * total / grid;
        begin -= (begin % nb) % 8; end -= (end % nb) % 8;
        begin -= base; end -= base;
        int lo = begin < 0 ? 0 : (int)begin;
        int hi = end > nb ? nb : (int)end;
        if (lo >= hi) continue;
        float acc = 0.0f;
        for (int b = lo; b < hi; b++) {
            const BnBlockQ5_1 *w = weights + ((size_t)(borrowed_expert >= 0 ? 0 : expert) * rows + row) * nb + b;
            const BnCudaBlockQ8Mmq *x = input + (size_t)item * nb + b;
            uint32_t high; memcpy(&high, w->qh, sizeof(high));
            int dot = 0;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                int low = (w->qs[j] & 15) | (((high >> j) & 1u) << 4);
                int upper = (w->qs[j] >> 4) | (((high >> (j + 16)) & 1u) << 4);
                dot += low * x->qs[j] + upper * x->qs[j + 16];
            }
            float scale = __fmul_rn(cuda_fp16_to_fp32(w->d), cuda_fp16_to_fp32(x->d));
            acc = __fmaf_rn(scale, (float)dot, acc);
            acc = __fmaf_rn(cuda_fp16_to_fp32(w->m), cuda_fp16_to_fp32(x->original_sum), acc);
        }
        if (!have) tail = acc; else prefix = __fadd_rn(prefix, acc);
        have = true;
    }
    out[(size_t)item * rows + row] = __fadd_rn(tail, prefix);
}

/* Explicit single-column quantized-activation graph contract. Legacy
 * floating-input matvec is selected separately by the backend. */
static __global__ void q5_1_matvec_mmvq_kernel(float *out,
    const BnBlockQ5_1 *weights, const BnCudaBlockQ8Mmq *input,
    int rows, int cols, const float *bias, size_t out_offset) {
    __shared__ float partial[3][32];
    int row=blockIdx.x, tid=threadIdx.x, lane=tid&31, warp=tid/32;
    if(row>=rows)return;
    int blocks=cols/32;
    float sum=0.0f;
    for(int b=tid/2;b<blocks;b+=blockDim.x/2)
        sum += q5_1_mmvq_dot(weights+(size_t)row*blocks+b,input+b,tid&1);
    if(warp)partial[warp-1][lane]=sum;
    __syncthreads();
    if(!warp) {
        for(int i=0;i<(int)blockDim.x/32-1;i++)sum+=partial[i][lane];
        for(int offset=16;offset;offset/=2)
            sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
        if(!lane)out[out_offset+row]=bias?__fadd_rn(sum,bias[row]):sum;
    }
}

#endif
