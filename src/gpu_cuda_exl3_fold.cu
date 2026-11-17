/* Adapted exllamav3 reconstruct.cu/hgemm_f16acc.cu, commit
 * bd1f519172406a9560bd5c90e61000691bb69a07, Copyright 2025 Turboderp.
 * MIT: third_party/exllamav3/LICENSE. Local original fast-math policy. */
#include <cuda_fp16.h>
#include <cstdint>
#include "gpu_cuda_exl3_fold.h"
namespace {
union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};


[[maybe_unused]] static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
#if defined(USE_ROCM)
    return exl3_bfe_u64(value, offset, length);
#else
    uint64_t result64;
    asm ("bfe.u64 %0, %1, %2, %3;"
         : "=l"(result64)
         : "l"(value), "r"(offset), "r"(length));
    return static_cast<uint32_t>(result64);
#endif
}

#if defined(USE_ROCM)
    #define FSHF_IMM(dst, lo, hi, imm) (dst) = exl3_shf_r_wrap((lo), (hi), (imm))
    #define BFE16_IMM(dst, src, imm) (dst) = exl3_bfe_u32_16((src), (imm))
#else
    #define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
    #define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))
#endif

typedef struct __align__(8) half4
{
    half2 x;
    half2 y;
    half4() = default;
    __device__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
    __device__ half4(half h0, half h1, half h2, half h3) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)) {}
}
half4;

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using FragB = Vec<half2, 2>;
__device__ inline half2 shuffle_had_h2x32(half2 v, int lane_id)
{
    for (int i = 1; i < 32; i <<= 1)
    {
        half2 pv = __shfl_xor_sync(0xffffffff, v, i);
        uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80008000);
        v = __hadd2(v, pv);
    }
    return v;
}

#include "gpu_cuda_exl3_fold_dequant.cuh"
#define RH_THREADS 256

template <int K, int cb, bool HALF = false>
__device__ __forceinline__
void reconstruct_had_tile
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    const half* __restrict__ suh,
    const half* __restrict__ svh,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);
    constexpr float r_scale = 0.08838834764831845f;

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int kb = blockIdx.y;
    int nb = blockIdx.x;
    int n = nb * 8;
    int row_len = gridDim.x * 128;

    __shared__ uint32_t s_packed[8][8][packed_size / 2];
    __shared__ half2 stile[128 * 64];

    auto tix = [&] (int R, int q, int p)
    {
        return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p;
    };

    constexpr int j_int4 = packed_size / 8;
    for (int u = t; u < 8 * 8 * j_int4; u += RH_THREADS)
    {
        int j = u / (8 * j_int4);
        int r = u % (8 * j_int4);
        const uint16_t* gp = g_packed +
            ((size_t) ((kb * 8 + j) * packed_blocks_n + packed_n_offset + n)) * packed_size;
        ((int4*) s_packed[j])[r] = ((const int4*) gp)[r];
    }
    __syncthreads();

    for (int jj = 0; jj < 8 * 8 / (RH_THREADS / 32); ++jj)
    {
        int j = (warp_id / 8) * (8 / (RH_THREADS / 256)) + jj;
        int wn = warp_id % 8;
        register FragB frag[2];
        dq_dispatch<K, cb, HALF>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

        if (!(lane_id & 4))
        {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            int r0 = j * 16 + (lane_id % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = lane_id / 8;
            int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = m0;
            stile[tix(r1, q0, p0)] = m1;
            stile[tix(r2, q0, p0)] = m2;
            stile[tix(r3, q0, p0)] = m3;
            stile[tix(r0, q1, p1)] = m4;
            stile[tix(r1, q1, p1)] = m5;
            stile[tix(r2, q1, p1)] = m6;
            stile[tix(r3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    const half2 rs2 = __float2half2_rn(r_scale);
    constexpr int CHUNKS_PW = 32 / (RH_THREADS / 32);
    #pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq)
    {
        int q = warp_id * CHUNKS_PW + qq;
        int qs = q ^ lane_id;
        // a[i]/b[i]: 4 columns x row (4 * lane + i), all four rows share swizzle == lane
        half2 a[4], b[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v = *((const half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2));
            a[i] = v.x;
            b[i] = v.y;
        }
        // butterfly over the 4 rows, two columns at a time, fp16 with pre-applied scale
        #pragma unroll
        for (int x = 0; x < 2; ++x)
        {
            half2* v = x == 0 ? a : b;
            half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
            half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
            v[0] = __hmul2(__hadd2(s0, s1), rs2);
            v[1] = __hmul2(__hadd2(d0, d1), rs2);
            v[2] = __hmul2(__hsub2(s0, s1), rs2);
            v[3] = __hmul2(__hsub2(d0, d1), rs2);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] = shuffle_had_h2x32(v[i], lane_id);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v;
            v.x = a[i];
            v.y = b[i];
            *((half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
        }
    }
    __syncthreads();

    // Row transform fused with the store: after the lane butterfly, lane l holds the
    // FINAL columns 4l..4l+3 of row R. Apply the sign scales in registers and write the
    // coalesced 256-byte row directly.
    constexpr int ROWS_PW = 128 / (RH_THREADS / 32);
    const half4 sv4 = ((const half4*) svh)[nb * 32 + lane_id];
    #pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr)
    {
        int R = warp_id * ROWS_PW + rr;
        int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
        half2 v01 = stile[base];
        half2 v23 = stile[base + 1];
        float v0 = __low2float(v01), v1 = __high2float(v01);
        float v2 = __low2float(v23), v3 = __high2float(v23);
        float s0 = v0 + v1, d0 = v0 - v1;
        float s1 = v2 + v3, d1 = v2 - v3;
        half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
        half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
        h01 = shuffle_had_h2x32(h01, lane_id);
        h23 = shuffle_had_h2x32(h23, lane_id);
        half2 su2 = __half2half2(suh[kb * 128 + R]);
        half4 v;
        v.x = __hmul2(__hmul2(h01, su2), sv4.x);
        v.y = __hmul2(__hmul2(h23, su2), sv4.y);
        *((half4*) (g_unpacked + (size_t) (kb * 128 + R) * row_len + nb * 128 + lane_id * 4)) = v;
    }
}


template <int K, int cb, bool HALF = false>
__global__ __launch_bounds__(RH_THREADS)
void reconstruct_had_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    const half* __restrict__ suh,
    const half* __restrict__ svh,
    int packed_blocks_n,
    int packed_n_offset
)
{
    reconstruct_had_tile<K, cb, HALF>(g_unpacked, g_packed, suh, svh, packed_blocks_n, packed_n_offset);
}

namespace f16acc
{

constexpr int KSLICE = 32;

// Preserve the existing configuration on pre-Blackwell GPUs. The two tuned layouts
// share the 32-term partial-sum interval and differ only in scheduling and storage.
template <int TILE_N, bool TUNED> struct Config
{
    static constexpr int BM = 128, BN = TILE_N, BK = 64, PAD = TUNED ? 0 : 8;
    static constexpr int WARPS_M = 2, WARPS_N = BN / 32;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int STAGES = 2, GROUP_M = TUNED ? 16 : 8;
    static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    static constexpr int MT = WM / 16, NT = WN / 8;
    static constexpr int AS_STRIDE = BK + PAD, BS_STRIDE = BN + PAD;
    static constexpr int A_STAGE = BM * AS_STRIDE, B_STAGE = BK * BS_STRIDE;
    static constexpr size_t SMEM_BYTES = (size_t) STAGES * (A_STAGE + B_STAGE) * sizeof(half);
};

__device__ __forceinline__ void add_half_pair(float& a, float& b, uint32_t h)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // PTX 8.6+: convert an FP16 partial and add it to the FP32 total in one instruction.
    asm volatile("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; "
                 "add.rn.f32.f16 %0, lo, %0; add.rn.f32.f16 %1, hi, %1; }"
                 : "+f"(a), "+f"(b) : "r"(h));
#else
    float2 f = __half22float2(*reinterpret_cast<half2*>(&h));
    a += f.x;
    b += f.y;
#endif
}


__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    int src_size = pred ? 16 : 0;    // 0 -> zero-fill, no global read
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_u32(smem)), "l"(gmem), "r"(src_size));
#else
    (void) smem; (void) gmem; (void) pred;
#endif
}
__device__ __forceinline__ void cp_async_commit()
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_async_wait()
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#else
    (void) N;
#endif
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
#else
    (void) r; (void) p;
#endif
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const void* p)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
#else
    (void) r; (void) p;
#endif
}
__device__ __forceinline__ void mma_f16(uint32_t* c, const uint32_t* a, const uint32_t* b)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void) c; (void) a; (void) b;
#endif
}
__device__ __forceinline__ void mma_f32(float* c, const uint32_t* a, const uint32_t* b)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void) c; (void) a; (void) b;
#endif
}

template <bool OUT_F32, int TILE_N = 128, bool TUNED = false>
__global__ void __launch_bounds__((Config<TILE_N, TUNED>::THREADS), 1)
gemm_kernel
(
    const half* __restrict__ A, const half* __restrict__ B, void* __restrict__ C,
    int M, int N, int K, int ldc,
    long long strideA, long long strideB, long long strideC
)
{
    using CF = Config<TILE_N, TUNED>;
    constexpr int BM = CF::BM, BN = CF::BN, BK = CF::BK, THREADS = CF::THREADS;
    constexpr int STAGES = CF::STAGES, GROUP_M = CF::GROUP_M;
    constexpr int WARPS_N = CF::WARPS_N;
    constexpr int WM = CF::WM, WN = CF::WN, MT = CF::MT, NT = CF::NT;
    constexpr int AS_STRIDE = CF::AS_STRIDE, BS_STRIDE = CF::BS_STRIDE;
    constexpr int A_STAGE = CF::A_STAGE, B_STAGE = CF::B_STAGE;

    extern __shared__ __align__(16) half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * A_STAGE;

    // Grouped raster: consecutive blocks walk GROUP_M M-stripes over one N column before moving
    // on, so the B stripe stays L2-hot across GROUP_M A-stripes
    const int grid_n = gridDim.x, grid_m = gridDim.y;
    const int bid = blockIdx.y * grid_n + blockIdx.x;
    const int group_size = GROUP_M * grid_n;
    const int group = bid / group_size;
    const int first_m = group * GROUP_M;
    const int gm_eff = min(GROUP_M, grid_m - first_m);
    const int in_group = bid - group * group_size;
    const int bm = (first_m + in_group % gm_eff) * BM;
    const int bn = (in_group / gm_eff) * BN;

    const int bz = blockIdx.z;
    A += (long long) bz * strideA;
    B += (long long) bz * strideB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;

    constexpr int A_CPR = BK / 8, B_CPR = BN / 8;
    constexpr int A_ITERS = BM * A_CPR / THREADS, B_ITERS = BK * B_CPR / THREADS;
    auto load_tile = [&](int stage, int kt)
    {
        const int k0 = kt * BK;
        half* as = As + stage * A_STAGE;
        half* bs = Bs + stage * B_STAGE;
        #pragma unroll
        for (int i = 0; i < A_ITERS; ++i)
        {
            int c = tid + i * THREADS;
            int row = c / A_CPR, chunk = c % A_CPR;
            int grow = bm + row;
            bool pred = grow < M;
            const half* src = A + (long long) (pred ? grow : 0) * K + k0 + chunk * 8;
            if constexpr (TUNED) cp_async16(as + row * AS_STRIDE + (chunk ^ (row % A_CPR)) * 8, src, pred);
            else cp_async16(as + row * AS_STRIDE + chunk * 8, src, pred);
        }
        #pragma unroll
        for (int i = 0; i < B_ITERS; ++i)
        {
            int c = tid + i * THREADS;
            int row = c / B_CPR, chunk = c % B_CPR;
            const half* src = B + (long long) (k0 + row) * N + bn + chunk * 8;
            if constexpr (TUNED) cp_async16(bs + row * BS_STRIDE + (chunk ^ (row % B_CPR)) * 8, src, true);
            else cp_async16(bs + row * BS_STRIDE + chunk * 8, src, true);
        }
    };

    float acc[MT][NT][4];
    uint32_t hacc[MT][NT][2];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
        {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
            hacc[i][j][0] = hacc[i][j][1] = 0u;
        }
    auto flush = [&]()
    {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                float2 f0 = __half22float2(*reinterpret_cast<half2*>(&hacc[i][j][0]));
                float2 f1 = __half22float2(*reinterpret_cast<half2*>(&hacc[i][j][1]));
                acc[i][j][0] += f0.x; acc[i][j][1] += f0.y;
                acc[i][j][2] += f1.x; acc[i][j][3] += f1.y;
                hacc[i][j][0] = 0u; hacc[i][j][1] = 0u;
            }
    };

    const int KT = K / BK;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < KT) load_tile(s, s);
        cp_async_commit();
    }

    const int a_lrow = lane & 15, a_lcol = (lane >> 4) * 8;
    const int b_lrow = (lane & 7) + ((lane >> 3) & 1) * 8, b_lcol = (lane >> 4) * 8;
    constexpr int KK = BK / 16;
    static_assert(KSLICE == 16 * 2, "flush cadence below assumes kslice 32 = two k16 steps");

    uint32_t af[2][MT][4];
    uint32_t bf[2][NT][2];
    auto load_frags = [&](int buf, const half* as, const half* bs, int kk)
    {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            if constexpr (TUNED)
                ldmatrix_x4(af[buf][i], as + (i * 16 + a_lrow) * AS_STRIDE +
                            (((kk * 16 + a_lcol) / 8) ^ ((i * 16 + a_lrow) % A_CPR)) * 8);
            else
                ldmatrix_x4(af[buf][i], as + (i * 16 + a_lrow) * AS_STRIDE + kk * 16 + a_lcol);
        #pragma unroll
        for (int j = 0; j < NT; j += 2)
        {
            uint32_t r[4];
            if constexpr (TUNED)
                ldmatrix_x4_trans(r, bs - wn * WN + (kk * 16 + b_lrow) * BS_STRIDE +
                                  (((wn * WN + j * 8 + b_lcol) / 8) ^ ((kk * 16 + b_lrow) % B_CPR)) * 8);
            else
                ldmatrix_x4_trans(r, bs + (kk * 16 + b_lrow) * BS_STRIDE + j * 8 + b_lcol);
            bf[buf][j][0] = r[0]; bf[buf][j][1] = r[1]; bf[buf][j + 1][0] = r[2]; bf[buf][j + 1][1] = r[3];
        }
    };

    for (int kt = 0; kt < KT; ++kt)
    {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KT) load_tile(nk % STAGES, nk);
            cp_async_commit();
        }
        const half* as = As + (kt % STAGES) * A_STAGE + wm * WM * AS_STRIDE;
        const half* bs = Bs + (kt % STAGES) * B_STAGE + wn * WN;

        if constexpr (TUNED)
        {
            // Preserve each pair of k16 MMAs and the FP32 addition order, but flush a
            // fragment immediately so conversions/additions can overlap other MMAs.
            #pragma unroll
            for (int kk = 0; kk < KK; kk += 2)
            {
                load_frags(0, as, bs, kk);
                load_frags(1, as, bs, kk + 1);
                #pragma unroll
                for (int i = 0; i < MT; ++i)
                    #pragma unroll
                    for (int j = 0; j < NT; ++j)
                    {
                        uint32_t h[2] = {};
                        mma_f16(h, af[0][i], bf[0][j]);
                        mma_f16(h, af[1][i], bf[1][j]);
                        add_half_pair(acc[i][j][0], acc[i][j][1], h[0]);
                        add_half_pair(acc[i][j][2], acc[i][j][3], h[1]);
                    }
            }
        }
        else
        {
            load_frags(0, as, bs, 0);
            #pragma unroll
            for (int kk = 0; kk < KK; ++kk)
            {
                if (kk + 1 < KK) load_frags((kk + 1) & 1, as, bs, kk + 1);
                const int cur = kk & 1;
                #pragma unroll
                for (int i = 0; i < MT; ++i)
                    #pragma unroll
                    for (int j = 0; j < NT; ++j)
                        mma_f16(hacc[i][j], af[cur][i], bf[cur][j]);
                if (kk & 1) flush();     // every 32 of K
            }
        }
    }

    const int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
        {
            int col = bn + wn * WN + j * 8 + t * 2;
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                int row = bm + wm * WM + i * 16 + g + h * 8;
                if (row >= M) continue;
                float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
                if constexpr (OUT_F32)
                {
                    float* c = reinterpret_cast<float*>(C) + (long long) bz * strideC + (long long) row * ldc + col;
                    *reinterpret_cast<float2*>(c) = make_float2(v0, v1);
                }
                else
                {
                    half* c = reinterpret_cast<half*>(C) + (long long) bz * strideC + (long long) row * ldc + col;
                    *reinterpret_cast<half2*>(c) = __floats2half2_rn(v0, v1);
                }
            }
        }
}

// Rate probe: independent register-resident MMA chains, no memory traffic
template <bool F16>
__global__ void __launch_bounds__(256) rate_kernel(int iters, float* sink)
{
    uint32_t a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 + 5, b1 = a0 + 7;
    uint32_t a[4] = { a0, a1, a2, a3 }, b[2] = { b0, b1 };
    uint32_t h[8][2] = {};
    float f[8][4] = {};
    for (int i = 0; i < iters; ++i)
    {
        #pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            if constexpr (F16) mma_f16(h[c], a, b);
            else mma_f32(f[c], a, b);
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) s += F16 ? __uint_as_float(h[c][0] ^ h[c][1]) : f[c][0] + f[c][3];
    if (s == 123.456f) sink[threadIdx.x] = s;
}


}

__global__ void folded_batch(half*out,BnCudaFoldPointers ptrs,int packed_n,size_t stride){int b=blockIdx.z;reconstruct_had_tile<3,2,false>(out+(size_t)b*stride,(const uint16_t*)ptrs.packed[b],(const half*)ptrs.pre[b],(const half*)ptrs.post[b],packed_n,0);}
static int measure(bool half_acc,int sms,float*sink,float*ms){cudaEvent_t a=NULL,b=NULL;cudaError_t e=cudaSuccess;
 auto run=[&](){if(half_acc)f16acc::rate_kernel<true><<<sms*4,256>>>(512,sink);else f16acc::rate_kernel<false><<<sms*4,256>>>(512,sink);return cudaGetLastError();};
 e=run();if(e==cudaSuccess)e=cudaEventCreate(&a);if(e==cudaSuccess)e=cudaEventCreate(&b);if(e==cudaSuccess)e=cudaEventRecord(a,0);
 for(int i=0;i<3&&e==cudaSuccess;i++)e=run();if(e==cudaSuccess)e=cudaEventRecord(b,0);if(e==cudaSuccess)e=cudaEventSynchronize(b);if(e==cudaSuccess)e=cudaEventElapsedTime(ms,a,b);
 if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);if(e!=cudaSuccess)cudaStreamSynchronize(0);return e==cudaSuccess?0:-1;
}
}
int bn_cuda_fold_probe(int sms,float*ratio){if(sms<1||!ratio)return -1;float*sink=NULL;if(cudaMalloc(&sink,1024)!=cudaSuccess){(void)cudaGetLastError();return -1;}float t32=0,t16=0;int r=measure(false,sms,sink,&t32);if(!r)r=measure(true,sms,sink,&t16);cudaFree(sink);if(r||t32<=0||t16<=0)return -1;*ratio=t32/t16;return 0;}
int bn_cuda_fold_prepare(){
#define ATTR(F,N,T) if(cudaFuncSetAttribute(f16acc::gemm_kernel<F,N,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,f16acc::Config<N,T>::SMEM_BYTES)!=cudaSuccess)return -1
 ATTR(true,64,true);ATTR(false,64,true);ATTR(true,128,true);ATTR(false,128,true);ATTR(true,128,false);ATTR(false,128,false);
#undef ATTR
 return 0;
}
int bn_cuda_fold_kind(int B,int M,int K,int N,int cc,int sms,int enabled){if(!enabled)return 0;bool narrow=N<=1024||M<128?true:B>1||N>4096?false:M<512||K>N||(M>=1024&&M<4096);bool worth=false;if(cc/100==12&&M>=64&&M*B>=384&&N>=1024&&K>=512){int tile=narrow?64:128;worth=((M+127)/128)*(N/tile)*B*2>=sms;}else if(M>=384)worth=((M+127)/128)*(N/128)*B>=sms;if(!worth)return 0;if(cc==890)return 1;if(cc/100!=12)return 3;return narrow?1:2;}
int bn_cuda_fold_reconstruct(cudaStream_t stream,const BnCudaFoldPointers*p,void*out,int B,int K,int N){folded_batch<<<dim3(N/128,K/128,B),256,0,stream>>>((half*)out,*p,N/16,(size_t)K*N);return cudaGetLastError()==cudaSuccess?0:-1;}
int bn_cuda_fold_gemm(cudaStream_t stream,int kind,const void*x,const void*w,void*out,int B,int M,int K,int N,int f32){
#define RUN(F,TN,T) f16acc::gemm_kernel<F,TN,T><<<dim3(N/TN,(M+127)/128,B),f16acc::Config<TN,T>::THREADS,f16acc::Config<TN,T>::SMEM_BYTES,stream>>>((const half*)x,(const half*)w,out,M,N,K,N,(long long)M*K,(long long)K*N,(long long)M*N)
 if(f32){if(kind==1){RUN(true,64,true);}else if(kind==2){RUN(true,128,true);}else if(kind==3){RUN(true,128,false);}else return -1;}
 else{if(kind==1){RUN(false,64,true);}else if(kind==2){RUN(false,128,true);}else if(kind==3){RUN(false,128,false);}else return -1;}
#undef RUN
 return cudaGetLastError()==cudaSuccess?0:-1;
}
