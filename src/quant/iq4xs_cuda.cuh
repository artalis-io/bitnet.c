#ifndef BN_QUANT_IQ4XS_CUDA_CUH
#define BN_QUANT_IQ4XS_CUDA_CUH

/* Backend-owned decode layout: expand the nonlinear codebook once during
 * upload. The original GGUF block remains resident for exact MMQ prefill. */
typedef struct {
    uint16_t d;
    int8_t scales[8];
    int8_t qs[256];
} BnCudaIQ4XSPackedBlock;

static __global__ void iq4xs_pack_decode_kernel(
        BnCudaIQ4XSPackedBlock *out, const BnBlockIQ4XS *in,
        size_t n_blocks) {
    size_t b = blockIdx.x;
    if (b >= n_blocks) return;
    int i = threadIdx.x;
    int group = i / 32, at = i & 31;
    const BnBlockIQ4XS *src = in + b;
    BnCudaIQ4XSPackedBlock *dst = out + b;
    int q = (src->qs[group * 16 + (at & 15)] >>
             (4 * (at >> 4))) & 15;
    dst->qs[i] = bn_kvalues_iq4nl[q];
    if (i < 8) {
        int scale = ((src->scales_l[i / 2] >> (4 * (i & 1))) & 15) |
                    (((src->scales_h >> (2 * i)) & 3) << 4);
        dst->scales[i] = (int8_t)(scale - 32);
    }
    if (i == 0) dst->d = src->d;
}

/* Same block assignment, scale product, and FP32 reduction as the reference
 * MMVQ kernel. Only the integer dot reads the expanded backend layout. */
static __global__ void iq4xs_dot_matvec_packed_kernel(float *out,
        const BnCudaIQ4XSPackedBlock *weights,
        const BnCudaBlockQ8_1 *input, int rows, int cols,
        const float *bias, size_t out_offset) {
    __shared__ float partial[3][32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    int row = blockIdx.x, token = blockIdx.y;
    int blocks = cols / 256, group = tid & 7;
    float sum = 0.0f;
    for (int b = tid / 8; b < blocks; b += blockDim.x / 8) {
        const BnCudaIQ4XSPackedBlock *w =
            weights + (size_t)row * blocks + b;
        const BnCudaBlockQ8_1 *x =
            input + (size_t)token * (cols / 32) + b * 8 + group;
        int dot = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t qv, xv;
            memcpy(&qv, w->qs + group * 32 + j * 4, sizeof(qv));
            memcpy(&xv, x->qs + j * 4, sizeof(xv));
            dot = cuda_dp4a_i32((int)qv, (int)xv, dot);
        }
        dot *= (int)w->scales[group];
        float d = cuda_fp16_to_fp32(w->d) * cuda_fp16_to_fp32(x->d);
        sum = fmaf(d, (float)dot, sum);
    }
    if (warp) partial[warp - 1][lane] = sum;
    __syncthreads();
    if (!warp) {
        for (int i = 0; i < blockDim.x / 32 - 1; i++)
            sum += partial[i][lane];
        for (int i = 16; i > 0; i /= 2)
            sum += __shfl_xor_sync(0xffffffffu, sum, i);
        if (!lane) {
            if (bias) sum += bias[row];
            out[out_offset + (size_t)token * rows + row] = sum;
        }
    }
}

#endif
