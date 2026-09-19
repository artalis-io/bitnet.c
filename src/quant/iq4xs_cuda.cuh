#ifndef BN_QUANT_IQ4XS_CUDA_CUH
#define BN_QUANT_IQ4XS_CUDA_CUH

/* Backend-owned layout: expand the nonlinear codebook once during upload.
 * The original GGUF block remains available for quant-only fallback. */
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

/* Expand eight IQ4_XS nibbles into two signed-byte dot operands using
 * byte permutations. The codebook values are the constants in iq_tables.h. */
static __device__ __forceinline__ int2 iq4xs_expand_codes(uint32_t q4) {
    const uint32_t t0 = 0xbfad9881u, t1 = 0xf6eaddcfu;
    const uint32_t t2 = 0x26190d01u, t3 = 0x71594535u;
    uint32_t selector = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t low0 = __byte_perm(t0, t1, q4);
    uint32_t high0 = __byte_perm(t2, t3, q4);
    uint32_t low1 = __byte_perm(t0, t1, q4 >> 16);
    uint32_t high1 = __byte_perm(t2, t3, q4 >> 16);
    uint32_t words0 = __byte_perm(low0, high0, selector);
    uint32_t words1 = __byte_perm(low1, high1, selector >> 16);
    return make_int2(__byte_perm(words0, words1, 0x6420),
                     __byte_perm(words0, words1, 0x7531));
}

static __global__ void iq4xs_dot_matvec_compact_perm_kernel(float *out,
        const BnBlockIQ4XS *weights, const BnCudaBlockQ8_1 *input,
        int rows, int cols, const float *bias, size_t out_offset) {
    __shared__ float partial[3][32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    int row = blockIdx.x, token = blockIdx.y;
    int blocks = cols / 256, group = tid & 7;
    float sum = 0.0f;
    for (int b = tid / 8; b < blocks; b += blockDim.x / 8) {
        const BnBlockIQ4XS *w = weights + (size_t)row * blocks + b;
        const BnCudaBlockQ8_1 *x =
            input + (size_t)token * (cols / 32) + b * 8 + group;
        int dot = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t codes, u0, u1;
            memcpy(&codes, w->qs + group * 16 + j * 4, sizeof(codes));
            int2 q = iq4xs_expand_codes(codes);
            memcpy(&u0, x->qs + j * 4, sizeof(u0));
            memcpy(&u1, x->qs + 16 + j * 4, sizeof(u1));
            dot = cuda_dp4a_i32(q.x, (int)u0, dot);
            dot = cuda_dp4a_i32(q.y, (int)u1, dot);
        }
        int scale = ((w->scales_l[group / 2] >> (4 * (group & 1))) & 15) |
                    (((w->scales_h >> (2 * group)) & 3) << 4);
        dot *= scale - 32;
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

static __global__ void iq4xs_dot_matvec_compact_perm_pair_kernel(float *out,
        const BnBlockIQ4XS *weights, const BnCudaBlockQ8_1 *input,
        int rows, int cols, const float *bias, size_t out_offset) {
    __shared__ float partial[2][3][32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    int row0 = blockIdx.x * 2, row1 = row0 + 1, token = blockIdx.y;
    int blocks = cols / 256, group = tid & 7;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (int b = tid / 8; b < blocks; b += blockDim.x / 8) {
        const BnBlockIQ4XS *w0 = weights + (size_t)row0 * blocks + b;
        const BnBlockIQ4XS *w1 = row1 < rows
            ? weights + (size_t)row1 * blocks + b : w0;
        const BnCudaBlockQ8_1 *x =
            input + (size_t)token * (cols / 32) + b * 8 + group;
        uint32_t u0[4], u1[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            memcpy(&u0[j], x->qs + j * 4, sizeof(uint32_t));
            memcpy(&u1[j], x->qs + 16 + j * 4, sizeof(uint32_t));
        }
        int dot0 = 0, dot1 = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t codes0, codes1;
            memcpy(&codes0, w0->qs + group * 16 + j * 4, sizeof(codes0));
            memcpy(&codes1, w1->qs + group * 16 + j * 4, sizeof(codes1));
            int2 q0 = iq4xs_expand_codes(codes0);
            int2 q1 = iq4xs_expand_codes(codes1);
            dot0 = cuda_dp4a_i32(q0.x, (int)u0[j], dot0);
            dot0 = cuda_dp4a_i32(q0.y, (int)u1[j], dot0);
            dot1 = cuda_dp4a_i32(q1.x, (int)u0[j], dot1);
            dot1 = cuda_dp4a_i32(q1.y, (int)u1[j], dot1);
        }
        int scale0 = ((w0->scales_l[group / 2] >> (4 * (group & 1))) & 15) |
                     (((w0->scales_h >> (2 * group)) & 3) << 4);
        int scale1 = ((w1->scales_l[group / 2] >> (4 * (group & 1))) & 15) |
                     (((w1->scales_h >> (2 * group)) & 3) << 4);
        dot0 *= scale0 - 32;
        dot1 *= scale1 - 32;
        float xd = cuda_fp16_to_fp32(x->d);
        float d0 = cuda_fp16_to_fp32(w0->d) * xd;
        float d1 = cuda_fp16_to_fp32(w1->d) * xd;
        sum0 = fmaf(d0, (float)dot0, sum0);
        sum1 = fmaf(d1, (float)dot1, sum1);
    }
    if (warp) {
        partial[0][warp - 1][lane] = sum0;
        partial[1][warp - 1][lane] = sum1;
    }
    __syncthreads();
    if (!warp) {
        for (int i = 0; i < blockDim.x / 32 - 1; i++) {
            sum0 += partial[0][i][lane];
            sum1 += partial[1][i][lane];
        }
        for (int i = 16; i > 0; i /= 2) {
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, i);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, i);
        }
        if (!lane) {
            if (bias) {
                sum0 += bias[row0];
                if (row1 < rows) sum1 += bias[row1];
            }
            out[out_offset + (size_t)token * rows + row0] = sum0;
            if (row1 < rows)
                out[out_offset + (size_t)token * rows + row1] = sum1;
        }
    }
}

static __global__ void iq4xs_dot_matvec_compact_perm_quad_kernel(float *out,
        const BnBlockIQ4XS *weights, const BnCudaBlockQ8_1 *input,
        int rows, int cols, const float *bias, size_t out_offset) {
    __shared__ float partial[4][3][32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    int row0 = blockIdx.x * 4, token = blockIdx.y;
    int blocks = cols / 256, group = tid & 7;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int b = tid / 8; b < blocks; b += blockDim.x / 8) {
        const BnCudaBlockQ8_1 *x =
            input + (size_t)token * (cols / 32) + b * 8 + group;
        const BnBlockIQ4XS *ws[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = row0 + r;
            ws[r] = weights + (size_t)(row < rows ? row : row0) * blocks + b;
        }
        int dots[4] = {0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t u0, u1;
            memcpy(&u0, x->qs + j * 4, sizeof(u0));
            memcpy(&u1, x->qs + 16 + j * 4, sizeof(u1));
#pragma unroll
            for (int r = 0; r < 4; r++) {
                uint32_t codes;
                memcpy(&codes, ws[r]->qs + group * 16 + j * 4,
                       sizeof(codes));
                int2 q = iq4xs_expand_codes(codes);
                dots[r] = cuda_dp4a_i32(q.x, (int)u0, dots[r]);
                dots[r] = cuda_dp4a_i32(q.y, (int)u1, dots[r]);
            }
        }
        float xd = cuda_fp16_to_fp32(x->d);
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int scale =
                ((ws[r]->scales_l[group / 2] >> (4 * (group & 1))) & 15) |
                (((ws[r]->scales_h >> (2 * group)) & 3) << 4);
            float d = cuda_fp16_to_fp32(ws[r]->d) * xd;
            sums[r] = fmaf(d, (float)(dots[r] * (scale - 32)), sums[r]);
        }
    }
    if (warp) {
#pragma unroll
        for (int r = 0; r < 4; r++)
            partial[r][warp - 1][lane] = sums[r];
    }
    __syncthreads();
    if (!warp) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
            for (int i = 0; i < blockDim.x / 32 - 1; i++)
                sums[r] += partial[r][i][lane];
            for (int i = 16; i > 0; i /= 2)
                sums[r] += __shfl_xor_sync(0xffffffffu, sums[r], i);
            int row = row0 + r;
            if (!lane && row < rows) {
                if (bias) sums[r] += bias[row];
                out[out_offset + (size_t)token * rows + row] = sums[r];
            }
        }
    }
}

#endif
