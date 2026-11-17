#ifndef BN_QUANT_KQUANT_BF16_CUDA_CUH
#define BN_QUANT_KQUANT_BF16_CUDA_CUH
/* Backend-private 16-row A8 prompt tiles. Integer tensor-core dot
 * products preserve the existing quantized values. The affine two-group
 * reductions and Q6_K outer FMA order are part of the BF16 contract.
 * Shared rows have a 16-byte skew for ldmatrix; inactive rows/tokens are zero.
 * All block threads participate in both block barriers and warp MMA operations.
 * Launch with Tokens*4 threads. Include after the aligned-load and Q6_K unpack helpers. */
#include "int8_mma_cuda.cuh"
static __device__ void cuda_bf16_mma_dot16(const int8_t a[16][272], const int8_t b[64][272],
                                           int warp, int lane, int g, int half, int c[4]) {
#if __CUDA_ARCH__ >= 800
    int av[2], bv;
    unsigned ap = (unsigned)__cvta_generic_to_shared(&a[lane % 16][g * 32 + half * 16]);
    unsigned bp = (unsigned)__cvta_generic_to_shared(&b[warp * 8 + lane % 8][g * 32 + half * 16]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
                 : "=r"(av[0]), "=r"(av[1])
                 : "r"(ap));
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];" : "=r"(bv) : "r"(bp));
    c[0] = c[1] = c[2] = c[3] = 0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
                 "{%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(av[0]), "r"(av[1]), "r"(bv));
#else
    /* This branch keeps lower-architecture compilation valid. The host routes
     * pre-SM80 devices through the existing DP4A kernels. */
    for (int l = 0; l < 4; ++l) {
        int row = lane / 4 + (l / 2) * 8, token = warp * 8 + (lane % 4) * 2 + (l & 1);
        c[l] = 0;
        for (int k = 0; k < 16; ++k)
            c[l] += (int)a[row][g * 32 + half * 16 + k] * (int)b[token][g * 32 + half * 16 + k];
    }
#endif
}
template <typename W, int Tokens>
static __device__ __forceinline__ void cuda_bf16_affine_mma_core(__nv_bfloat16 *out, const W *weights,
                                                   const BnCudaBlockQ8_1 *input, int rows, int cols,
                                                   int nt, int row0, int token0) {
    static_assert(Tokens >= 8 && Tokens <= 64 && Tokens % 8 == 0, "BF16 MMA token tile");
    __shared__ __align__(16) int8_t a[16][272], x[Tokens][272];
    __shared__ int ws[16][8], wmn[16][8];
    __shared__ float wd[16], wm[16], xd[Tokens][8], xs[Tokens][8];
    int tid = threadIdx.x, lane = tid & 31, warp = tid / 32, blocks = cols / 256;
    float acc[4] = {0};
    for (int b = 0; b < blocks; ++b) {
        for (int i = tid * 4; i < 16 * 128; i += Tokens * 16) {
            int r = i / 128, p = i % 128, row = row0 + r;
            int g = 2 * (p / 32), off = p % 32;
            uint2 q = row < rows
                ? cuda_bf16_affine_quant8(weights + (size_t)row * blocks + b, p)
                : make_uint2(0, 0);
            *(uint32_t *)&a[r][g * 32 + off] = q.x;
            *(uint32_t *)&a[r][(g + 1) * 32 + off] = q.y;
        }
        for (int i = tid * 4; i < Tokens * 256; i += Tokens * 16) {
            int t = i / 256, k = i % 256, token = token0 + t;
            uint32_t q =
                token < nt
                    ? __ldg((
                          const uint32_t *)(input[(size_t)token * (cols / 32) + b * 8 + k / 32].qs +
                                            k % 32))
                    : 0;
            *(uint32_t *)&x[t][k] = q;
        }
        if (tid < 16) {
            int row = row0 + tid;
            const W *w = row < rows ? weights + (size_t)row * blocks + b : NULL;
            wd[tid] = w ? cuda_fp16_to_fp32(w->d) : 0;
            wm[tid] = w ? cuda_fp16_to_fp32(w->dmin) : 0;
            for (int g = 0; g < 8; ++g) {
                int s = 0, m = 0;
                if (w)
                    cuda_bf16_kquant_scale_min(w->scales, g, &s, &m);
                ws[tid][g] = s;
                wmn[tid][g] = m;
            }
        }
        for (int i = tid; i < Tokens * 8; i += Tokens * 4) {
            int t = i / 8, g = i % 8, token = token0 + t;
            const BnCudaBlockQ8_1 *q =
                token < nt ? input + (size_t)token * (cols / 32) + b * 8 + g : NULL;
            xd[t][g] = q ? cuda_fp16_to_fp32(q->d) : 0;
            xs[t][g] = q ? cuda_fp16_to_fp32(q->qsum) : 0;
        }
        __syncthreads();
        for (int pair = 0; pair < 8; pair += 2) {
            float ds[4] = {0}, ms[4] = {0};
            for (int j = 0; j < 2; ++j) {
                int g = pair + j, c[4];
                cuda_bf16_mma_dot32(a, x, warp, lane, g, c);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int r = lane / 4 + (l / 2) * 8, t = warp * 8 + (lane % 4) * 2 + (l & 1);
                    ds[l] = fmaf(xd[t][g], (float)(ws[r][g] * c[l]), ds[l]);
                    ms[l] = fmaf(xs[t][g], (float)wmn[r][g], ms[l]);
                }
            }
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                int r = lane / 4 + (l / 2) * 8;
                acc[l] = __fadd_rn(acc[l], fmaf(wd[r], ds[l], -__fmul_rn(wm[r], ms[l])));
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        int r = row0 + lane / 4 + (l / 2) * 8, t = token0 + warp * 8 + (lane % 4) * 2 + (l & 1);
        if (r < rows && t < nt)
            out[(size_t)t * rows + r] = __float2bfloat16_rn(acc[l]);
    }
}
template <typename W, int Tokens>
static __global__ void cuda_bf16_affine_mma_kernel(__nv_bfloat16 *out, const W *weights,
    const BnCudaBlockQ8_1 *input, int rows, int cols, int nt) {
    int tiles=(nt-1)/Tokens+1;
    cuda_bf16_affine_mma_core<W,Tokens>(out,weights,input,rows,cols,nt,
        (blockIdx.x/tiles)*16,(blockIdx.x%tiles)*Tokens);
}
/* Short prompts: four warps compute independent two-group affine terms.
 * Warp zero adds those F32 terms in the original pair/block order. Launch
 * 128 threads for a 16-row by 8-token tile; all threads reach each barrier.
 * Integer dots and intermediate F32 storage preserve the precision contract. */
template <typename W>
static __device__ __forceinline__ void cuda_bf16_affine_pair_split_core(__nv_bfloat16 *out, const W *weights,
                                                   const BnCudaBlockQ8_1 *input, int rows, int cols,
                                                   int nt) {
    const int Rows = 16, Tokens = 8;
    __shared__ __align__(16) int8_t a[Rows][272], x[Tokens][272];
    __shared__ int ws[Rows][8], wmn[Rows][8];
    __shared__ float wd[Rows], wm[Rows], xd[Tokens][8], xs[Tokens][8];
    __shared__ float parts[4][32][4];
    int tid = threadIdx.x, lane = tid & 31, warp = tid / 32, tiles = (nt - 1) / Tokens + 1;
    int row0 = (blockIdx.x / tiles) * Rows, token0 = (blockIdx.x % tiles) * Tokens,
        blocks = cols / 256;
    float acc[4] = {0};
    for (int b = 0; b < blocks; ++b) {
        for (int i = tid * 4; i < Rows * 128; i += blockDim.x * 4) {
            int r = i / 128, p = i % 128, row = row0 + r;
            int g = 2 * (p / 32), off = p % 32;
            uint2 q = row < rows
                ? cuda_bf16_affine_quant8(weights + (size_t)row * blocks + b, p)
                : make_uint2(0, 0);
            *(uint32_t *)&a[r][g * 32 + off] = q.x;
            *(uint32_t *)&a[r][(g + 1) * 32 + off] = q.y;
        }
        for (int i = tid * 4; i < Tokens * 256; i += blockDim.x * 4) {
            int t = i / 256, k = i % 256, token = token0 + t;
            uint32_t q =
                token < nt
                    ? __ldg((
                          const uint32_t *)(input[(size_t)token * (cols / 32) + b * 8 + k / 32].qs +
                                            k % 32))
                    : 0;
            *(uint32_t *)&x[t][k] = q;
        }
        if (tid < Rows) {
            int row = row0 + tid;
            const W *w = row < rows ? weights + (size_t)row * blocks + b : NULL;
            wd[tid] = w ? cuda_fp16_to_fp32(w->d) : 0;
            wm[tid] = w ? cuda_fp16_to_fp32(w->dmin) : 0;
            for (int g = 0; g < 8; ++g) {
                int s = 0, m = 0;
                if (w)
                    cuda_bf16_kquant_scale_min(w->scales, g, &s, &m);
                ws[tid][g] = s;
                wmn[tid][g] = m;
            }
        }
        for (int i = tid; i < Tokens * 8; i += blockDim.x) {
            int t = i / 8, g = i % 8, token = token0 + t;
            const BnCudaBlockQ8_1 *q =
                token < nt ? input + (size_t)token * (cols / 32) + b * 8 + g : NULL;
            xd[t][g] = q ? cuda_fp16_to_fp32(q->d) : 0;
            xs[t][g] = q ? cuda_fp16_to_fp32(q->qsum) : 0;
        }
        __syncthreads();
        float ds[4] = {0}, ms[4] = {0};
        for (int j = 0; j < 2; ++j) {
            int g = warp * 2 + j, c[4];
            cuda_bf16_mma_dot32(a, x, 0, lane, g, c);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                int r = lane / 4 + (l / 2) * 8, t = (lane % 4) * 2 + (l & 1);
                ds[l] = fmaf(xd[t][g], (float)(ws[r][g] * c[l]), ds[l]);
                ms[l] = fmaf(xs[t][g], (float)wmn[r][g], ms[l]);
            }
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            int r = lane / 4 + (l / 2) * 8;
            parts[warp][lane][l] = fmaf(wd[r], ds[l], -__fmul_rn(wm[r], ms[l]));
        }
        __syncthreads();
        if (warp == 0) {
            for (int pair = 0; pair < 4; ++pair)
#pragma unroll
                for (int l = 0; l < 4; ++l)
                    acc[l] = __fadd_rn(acc[l], parts[pair][lane][l]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        int r = row0 + lane / 4 + (l / 2) * 8, t = token0 + (lane % 4) * 2 + (l & 1);
        if (warp == 0 && r < rows && t < nt)
            out[(size_t)t * rows + r] = __float2bfloat16_rn(acc[l]);
    }
}
/* Preserve the unbounded entry point for small grids and other devices.
 * Even a one-block launch bound changes compiler register heuristics, so only
 * the measured large SM120 shape uses the separately bounded entry point. */
template <typename W>
static __global__ void cuda_bf16_affine_pair_split_kernel(__nv_bfloat16 *out, const W *weights,
    const BnCudaBlockQ8_1 *input, int rows, int cols, int nt) {
    cuda_bf16_affine_pair_split_core(out, weights, input, rows, cols, nt);
}
template <typename W>
static __global__ __launch_bounds__(128, 6) void cuda_bf16_affine_pair_resident_kernel(
    __nv_bfloat16 *out, const W *weights, const BnCudaBlockQ8_1 *input,
    int rows, int cols, int nt) {
    cuda_bf16_affine_pair_split_core(out, weights, input, rows, cols, nt);
}
template <int Tokens>
static __device__ __forceinline__ void cuda_bf16_q6_mma_core(__nv_bfloat16 *out, const BnBlockQ6K *weights,
                                           const BnCudaBlockQ8_1 *input, int rows, int cols,
                                           int nt, int row0, int token0) {
    static_assert(Tokens >= 8 && Tokens <= 64 && Tokens % 8 == 0, "BF16 MMA token tile");
    __shared__ __align__(16) int8_t a[16][272], x[Tokens][272];
    __shared__ int ws[16][16];
    __shared__ float wd[16], xd[Tokens][8];
    int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    int blocks = cols / 256;
    float acc[4] = {0};
    for (int b = 0; b < blocks; ++b) {
        for (int i = tid * 4; i < 16 * 128; i += Tokens * 16) {
            int r = i / 128, p = i % 128, row = row0 + r;
            int k = (p / 64) * 128 + ((p % 64) / 32) * 32 + p % 32;
            uint2 q = row < rows
                ? cuda_bf16_q6_quant8(weights + (size_t)row * blocks + b, p)
                : make_uint2(0, 0);
            *(uint32_t *)&a[r][k] = q.x;
            *(uint32_t *)&a[r][k + 64] = q.y;
        }
        for (int i = tid * 4; i < Tokens * 256; i += Tokens * 16) {
            int t = i / 256, k = i % 256, token = token0 + t;
            uint32_t q =
                token < nt
                    ? __ldg((
                          const uint32_t *)(input[(size_t)token * (cols / 32) + b * 8 + k / 32].qs +
                                            k % 32))
                    : 0;
            *(uint32_t *)&x[t][k] = q;
        }
        if (tid < 16) {
            int row = row0 + tid;
            const BnBlockQ6K *w = row < rows ? weights + (size_t)row * blocks + b : NULL;
            wd[tid] = w ? cuda_fp16_to_fp32(w->d) : 0;
            for (int g = 0; g < 16; ++g)
                ws[tid][g] = w ? w->scales[g] : 0;
        }
        for (int i = tid; i < Tokens * 8; i += Tokens * 4) {
            int t = i / 8, g = i % 8, token = token0 + t;
            xd[t][g] = token < nt
                           ? cuda_fp16_to_fp32(input[(size_t)token * (cols / 32) + b * 8 + g].d)
                           : 0;
        }
        __syncthreads();
        float partial[4] = {0};
        for (int g = 0; g < 8; ++g) {
            int c0[4], c1[4];
            cuda_bf16_mma_dot16(a, x, warp, lane, g, 0, c0);
            cuda_bf16_mma_dot16(a, x, warp, lane, g, 1, c1);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                int r = lane / 4 + (l / 2) * 8, t = warp * 8 + (lane % 4) * 2 + (l & 1);
                int dot = c0[l] * ws[r][g * 2] + c1[l] * ws[r][g * 2 + 1];
                partial[l] = fmaf(xd[t][g], (float)dot, partial[l]);
                if (g & 1) {
                    acc[l] = fmaf(wd[r], partial[l], acc[l]);
                    partial[l] = 0;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        int r = row0 + lane / 4 + (l / 2) * 8, t = token0 + warp * 8 + (lane % 4) * 2 + (l & 1);
        if (r < rows && t < nt)
            out[(size_t)t * rows + r] = __float2bfloat16_rn(acc[l]);
    }
}

template <int Tokens>
static __global__ void q6k_bf16_mma_kernel(__nv_bfloat16 *out, const BnBlockQ6K *weights,
    const BnCudaBlockQ8_1 *input, int rows, int cols, int nt) {
    int tiles=(nt-1)/Tokens+1;
    cuda_bf16_q6_mma_core<Tokens>(out,weights,input,rows,cols,nt,
        (blockIdx.x/tiles)*16,(blockIdx.x%tiles)*Tokens);
}

#endif /* BN_QUANT_KQUANT_BF16_CUDA_CUH */
