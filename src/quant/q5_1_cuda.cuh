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
            ((size_t)ids[item] * rows + row) * blocks + b;
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
#endif
