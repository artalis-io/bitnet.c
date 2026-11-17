#ifndef BN_QUANT_Q5_1_BF16_CUDA_CUH
#define BN_QUANT_Q5_1_BF16_CUDA_CUH

/* Layout preparation: GGUF FP32 affine decode, then one BF16 RNE cast.
 * Keep the affine addition separate from the scale multiplication, matching
 * gguf-py dequantization. This is not the Q8 activation MMVQ/MMQ contract. */
static __device__ __forceinline__ __nv_bfloat16 q5_1_decode_bf16_value(
    const BnBlockQ5_1 *weights, size_t i) {
    const BnBlockQ5_1 *w = weights + i / 32;
    int lane = (int)(i % 32);
    int packed = w->qs[lane & 15];
    int low = lane < 16 ? packed & 15 : packed >> 4;
    int high = (w->qh[lane / 8] >> (lane & 7)) & 1;
    float scaled = __fmul_rn(cuda_fp16_to_fp32(w->d), (float)(low | (high << 4)));
    return __float2bfloat16_rn(__fadd_rn(scaled, cuda_fp16_to_fp32(w->m)));
}

static __device__ __forceinline__ uint32_t q5_1_bf16_pack4(
    const BnBlockQ5_1 *w, int base) {
    uint32_t word, high;
    memcpy(&word, w->qs + (base & 15), sizeof(word));
    memcpy(&high, w->qh, sizeof(high));
    word = (word >> (base >= 16 ? 4 : 0)) & 0x0f0f0f0fu;
#pragma unroll
    for (int j = 0; j < 4; j++)
        word |= ((high >> (base + j)) & 1u) << (j * 8 + 4);
    return word;
}

/* Original-sum affine dot. Scale and minimum products round to FP16 before
 * the F32 dot FMA. MMVQ divides the minimum correction over two lane parts. */
static __device__ __forceinline__ float q5_1_bf16_dot(
    const BnBlockQ5_1 *w, const BnCudaBlockQ8_1 *x, int part, bool full) {
    int dot = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        if (full || j / 2 == part || j / 2 == part + 2) {
            int packed;
            memcpy(&packed, x->qs + j * 4, sizeof(packed));
            dot = cuda_dp4a_i32((int)q5_1_bf16_pack4(w, j * 4), packed, dot);
        }
    }
    float scale = cuda_fp16_to_fp32(cuda_fp32_to_fp16_bits(
        __fmul_rn(cuda_fp16_to_fp32(w->d), cuda_fp16_to_fp32(x->d))));
    float offset = cuda_fp16_to_fp32(cuda_fp32_to_fp16_bits(
        __fmul_rn(cuda_fp16_to_fp32(w->m), cuda_fp16_to_fp32((uint16_t)x->qsum))));
    return fmaf((float)dot, scale, full ? offset : __fmul_rn(offset, 0.5f));
}

static __device__ __forceinline__ void q5_1_bf16_mmvq_core(__nv_bfloat16 *out, const BnBlockQ5_1 *weights, const BnCudaBlockQ8_1 *input, int rows, int cols, int row, int token) {
    int lane = threadIdx.x, blocks = cols / 32;
    float sum = 0;
    for (int b = lane / 2; b < blocks; b += 16)
        sum += q5_1_bf16_dot(weights + (size_t)row * blocks + b,
            input + (size_t)token * blocks + b, lane & 1, false);
    for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
    if (!lane) out[(size_t)token * rows + row] = __float2bfloat16_rn(sum);
}

static __global__ void q5_1_bf16_mmvq_kernel(__nv_bfloat16 *out,
    const BnBlockQ5_1 *weights, const BnCudaBlockQ8_1 *input, int rows, int cols) {
    q5_1_bf16_mmvq_core(out, weights, input, rows, cols, blockIdx.x, blockIdx.y);
}

/* Four token columns share each input block. All raw weight reads stay within
 * cols/32 blocks, including the actual 640-column expert down tail. */
static __device__ __forceinline__ void q5_1_bf16_mmq_core(__nv_bfloat16 *out, const BnBlockQ5_1 *weights, const BnCudaBlockQ8_1 *input, int rows, int cols, int nt, int row, int token0, BnCudaBlockQ8_1 activation[4]) {
    int blocks = cols / 32;
    float acc[4] = {0};
    for (int b = 0; b < blocks; b++) {
        if (threadIdx.x < 4) {
            int token = token0 + threadIdx.x;
            activation[threadIdx.x] = input[(size_t)(token < nt ? token : nt - 1) * blocks + b];
        }
        __syncthreads();
        if (row < rows) {
            const BnBlockQ5_1 *w = weights + (size_t)row * blocks + b;
#pragma unroll
            for (int t = 0; t < 4; t++)
                if (token0 + t < nt) acc[t] += q5_1_bf16_dot(w, &activation[t], 0, true);
        }
        __syncthreads();
    }
    if (row < rows)
#pragma unroll
        for (int t = 0; t < 4; t++)
            if (token0 + t < nt) out[(size_t)(token0 + t) * rows + row] = __float2bfloat16_rn(acc[t]);
}

static __global__ void q5_1_bf16_mmq_kernel(__nv_bfloat16 *out,
    const BnBlockQ5_1 *weights, const BnCudaBlockQ8_1 *input, int rows, int cols, int nt) {
    __shared__ BnCudaBlockQ8_1 activation[4];
    int tiles = (nt - 1) / 4 + 1;
    q5_1_bf16_mmq_core(out, weights, input, rows, cols, nt, (blockIdx.x / tiles) * 128 + threadIdx.x, (blockIdx.x % tiles) * 4, activation);
}

#endif
