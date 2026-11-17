/* Adapted exllamav3 routing.cu/routing_gemm.cu, commit
 * bd1f519172406a9560bd5c90e61000691bb69a07, Copyright 2025 Turboderp.
 * MIT: third_party/exllamav3/LICENSE. Fast math is local to this TU. */
#include "gpu_cuda_router.h"
#include <cuda_fp16.h>
#include <stdint.h>
#include <algorithm>
#include <mma.h>
namespace {
#include "gpu_cuda_router_det.cuh"
#define CEIL_DIVIDE(a,b) (((a)+(b)-1)/(b))
#define RG_BM 128
#define RG_BN 64
#define RG_THREADS 512
// One stage on ROCm: its copies are synchronous (rocm/det_gemm_rocm.cuh), so a second stage would overlap
// nothing, and two do not fit RDNA's 64 KB of LDS per workgroup
#if defined(USE_ROCM)
    #define RG_STAGES 1
#else
    #define RG_STAGES 2
#endif
#define RG_KCH 128
#define RG_A_BYTES (2 * RG_BM * RG_KCH)
#define RG_B_BYTES (2 * RG_BN * RG_KCH)
#define RG_S_BYTES (RG_BM * 4)
#define RG_STAGE_BYTES (RG_A_BYTES + RG_B_BYTES + RG_S_BYTES)
#define RG_SMEM (RG_STAGES * RG_STAGE_BYTES)
#define RG_STATIC_ROWS 64

// A (R, K) half -> hi/lo (R, K) int8, sa (R, KC) fp32. One thread per 16 k of one row; the 8
// threads of a 128-chunk share the max by shuffle
__global__ __launch_bounds__(256)
void quant_a_kernel(const half* __restrict__ a, signed char* __restrict__ ahi, signed char* __restrict__ alo,
                    float* __restrict__ sa, const int R, const int K, const int KC)
{
    const int idx = blockIdx.x * 256 + threadIdx.x;
    const int eighth = idx & 7, chunk = (idx >> 3) % KC, row = (idx >> 3) / KC;
    if (row >= R) return;
    const int k = chunk * RG_KCH + eighth * 16;
    float v[16];
    #pragma unroll
    for (int q = 0; q < 2; ++q)
    {
        int4 raw = make_int4(0, 0, 0, 0);
        if (k + q * 8 < K) raw = *(const int4*) (a + (size_t) row * K + k + q * 8);
        const half2* h2 = (const half2*) &raw;
        #pragma unroll
        for (int u = 0; u < 4; ++u) { float2 f = __half22float2(h2[u]); v[q * 8 + u * 2] = f.x; v[q * 8 + u * 2 + 1] = f.y; }
    }
    float m = 0.0f;
    #pragma unroll
    for (int q = 0; q < 16; ++q) m = fmaxf(m, fabsf(v[q]));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
    const float scale = m > 0.0f ? m : 1.0f;
    int4 hi4, lo4;
    det_quant16(v, __fdiv_rn(DET_QMAX, scale), hi4, lo4);
    if (k < K)
    {
        *(int4*) (ahi + (size_t) row * K + k) = hi4;
        *(int4*) (alo + (size_t) row * K + k) = lo4;
    }
    if (eighth == 0) sa[(size_t) row * KC + chunk] = __fdiv_rn(scale, DET_QMAX);
}

__global__ __launch_bounds__(RG_THREADS)
void routing_gemm_i8_kernel
(
    const signed char* __restrict__ ahi, const signed char* __restrict__ alo, const float* __restrict__ sa,   // (R, K), (R, K), (R, KC)
    const signed char* __restrict__ bhi, const signed char* __restrict__ blo, const float* __restrict__ sb,   // (E, K), (E, K), (E)
    half* __restrict__ c, float* __restrict__ part,                                                           // (R, E) half, or (S, R, E) fp32 partials
    const int R, const int E, const int K, const int KC, const int kslice
)
{
    extern __shared__ __align__(128) unsigned char dsm[];
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int wm = warp / 4, wn = warp % 4;
    const int g = lane / 4, tg = lane % 4;
    const int r0 = blockIdx.y * RG_BM, e0 = blockIdx.x * RG_BN;
    const int k_beg = blockIdx.z * kslice, k_end = min(k_beg + kslice, K);
    const int n_chunks = (k_end - k_beg + RG_KCH - 1) / RG_KCH;

    auto st_ahi = [&](int st) { return dsm + st * RG_STAGE_BYTES; };
    auto st_alo = [&](int st) { return dsm + st * RG_STAGE_BYTES + RG_BM * RG_KCH; };
    auto st_bhi = [&](int st) { return dsm + st * RG_STAGE_BYTES + RG_A_BYTES; };
    auto st_blo = [&](int st) { return dsm + st * RG_STAGE_BYTES + RG_A_BYTES + RG_BN * RG_KCH; };
    auto st_sa  = [&](int st) { return (float*) (dsm + st * RG_STAGE_BYTES + RG_A_BYTES + RG_B_BYTES); };

    auto issue = [&](int chunk, int st)
    {
        const int k0 = k_beg + chunk * RG_KCH;
        #pragma unroll
        for (int q = 0; q < 2; ++q)
        {
            const int i = t + q * RG_THREADS, row = i / 8, cc = i % 8;
            const int gr = r0 + row, k = k0 + cc * 16;
            const int bytes = (gr < R) ? max(0, min(16, k_end - k)) : 0;
            const size_t off = bytes ? (size_t) gr * K + k : 0;
            det_cp_async16(det_smem_u32(st_ahi(st) + det_swz8(row, cc)), ahi + off, bytes);
            det_cp_async16(det_smem_u32(st_alo(st) + det_swz8(row, cc)), alo + off, bytes);
        }
        {
            const int row = t / 8, cc = t % 8;
            const int ge = e0 + row, k = k0 + cc * 16;
            const int bytes = (ge < E) ? max(0, min(16, k_end - k)) : 0;
            const size_t off = bytes ? (size_t) ge * K + k : 0;
            det_cp_async16(det_smem_u32(st_bhi(st) + det_swz8(row, cc)), bhi + off, bytes);
            det_cp_async16(det_smem_u32(st_blo(st) + det_swz8(row, cc)), blo + off, bytes);
        }
        if (t < RG_BM / 4)
        {
            float v4[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u)
            {
                const int gr = r0 + t * 4 + u;
                v4[u] = (gr < R) ? sa[(size_t) gr * KC + (k0 / RG_KCH)] : 0.0f;
            }
            *(float4*) (st_sa(st) + t * 4) = make_float4(v4[0], v4[1], v4[2], v4[3]);
        }
    };

#if defined(USE_ROCM)
    DetWmmaTile<2, 1> wt;
    wt.clear();
#else
    float facc[2][2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q) facc[i][j][q] = 0.0f;
#endif

    issue(0, 0); det_cp_async_commit();
    for (int chunk = 0; chunk < n_chunks; ++chunk)
    {
        det_cp_async_wait<0>();
        __syncthreads();
        if (RG_STAGES > 1 && chunk + 1 < n_chunks) issue(chunk + 1, (chunk + 1) % RG_STAGES);
        det_cp_async_commit();
        const int st = chunk % RG_STAGES;
        const unsigned a_hi = det_smem_u32(st_ahi(st)), a_lo = det_smem_u32(st_alo(st));
        const unsigned b_hi = det_smem_u32(st_bhi(st)), b_lo = det_smem_u32(st_blo(st));
        const float* sa_st = st_sa(st);

#if defined(USE_ROCM)
        wt.chunk<128, RG_KCH>(a_hi, a_lo, wm * 32, b_hi, b_lo, wn * 16, sa_st, lane);
#else
        int acc_hh[2][2][4], acc_x[2][2][4];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) { acc_hh[i][j][q] = 0; acc_x[i][j][q] = 0; }

        #pragma unroll
        for (int ks = 0; ks < RG_KCH; ks += 32)
        {
            unsigned ah[2][4], al[2][4], bh[2][2], bl[2][2], r4[4];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                det_load_a(ah[i], a_hi, wm * 32 + i * 16, ks / 16, lane);
                det_load_a(al[i], a_lo, wm * 32 + i * 16, ks / 16, lane);
            }
            det_load_b2(r4, b_hi, wn * 16, ks / 16, lane);
            bh[0][0] = r4[0]; bh[0][1] = r4[1]; bh[1][0] = r4[2]; bh[1][1] = r4[3];
            det_load_b2(r4, b_lo, wn * 16, ks / 16, lane);
            bl[0][0] = r4[0]; bl[0][1] = r4[1]; bl[1][0] = r4[2]; bl[1][1] = r4[3];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    det_mma3(acc_hh[i][j], acc_x[i][j], ah[i], al[i], bh[j], bl[j]);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const float s0 = sa_st[wm * 32 + i * 16 + g], s1 = sa_st[wm * 32 + i * 16 + g + 8];
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q)
                    facc[i][j][q] = det_flush(acc_hh[i][j][q], acc_x[i][j][q], q >= 2 ? s1 : s0, facc[i][j][q]);
        }
#endif
        // Single buffer: the next chunk can only land once everyone is done with this one
        if (RG_STAGES == 1 && chunk + 1 < n_chunks)
        {
            __syncthreads();
            issue(chunk + 1, 0);
        }
    }
    det_cp_async_wait<0>();
    __syncthreads();
    float (*cs_)[RG_BN + 4] = (float (*)[RG_BN + 4]) dsm;
#if defined(USE_ROCM)
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int v = 0; v < 8; ++v)
            cs_[wm * 32 + i * 16 + det_wmma_row(v, lane)][wn * 16 + (lane & 15)] = wt.acc[i][0][v];
#else
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q)
                cs_[wm * 32 + i * 16 + g + (q >= 2 ? 8 : 0)][wn * 16 + j * 8 + tg * 2 + (q & 1)] = facc[i][j][q];
#endif
    __syncthreads();
    if (part) part += (size_t) blockIdx.z * R * E;
    #pragma unroll
    for (int q = 0; q < (RG_BM * RG_BN) / RG_THREADS; ++q)
    {
        const int i = (t + q * RG_THREADS) / RG_BN, j = (t + q * RG_THREADS) % RG_BN;
        if (r0 + i < R && e0 + j < E)
        {
            float v = __fmul_rn(cs_[i][j], sb[e0 + j]);
            if (part) part[(size_t) (r0 + i) * E + e0 + j] = v;
            else c[(size_t) (r0 + i) * E + e0 + j] = __float2half_rn(v);
        }
    }
}

__global__ __launch_bounds__(256)
void routing_gemm_reduce_kernel(const float* __restrict__ part, half* __restrict__ c, const int n, const int S)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    for (int s = 0; s < S; ++s) v = __fadd_rn(v, part[(size_t) s * n + i]);
    c[i] = __float2half_rn(v);
}

// Split-K slices: a function of the shape only
static int rg_slices(int R, int E, int KC)
{
    const int tiles = CEIL_DIVIDE(E, RG_BN) * CEIL_DIVIDE(R, RG_BM);
    int S = 1;
    while (tiles * S < 256 && S < 8 && S * 2 <= KC) S *= 2;
    return S;
}

__global__ __launch_bounds__(256)
void quant_gate_kernel(const half* __restrict__ w, signed char* __restrict__ hi, signed char* __restrict__ lo, float* __restrict__ sb, const int E, const int K)
{
    const int row = blockIdx.x;
    float m = 0.0f;
    for (int k = threadIdx.x; k < K; k += 256) m = fmaxf(m, fabsf(__half2float(w[(size_t) row * K + k])));
    __shared__ float red[8];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    m = red[0];
    #pragma unroll
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    const float scale = m > 0.0f ? m : 1.0f;
    const float inv = __fdiv_rn(DET_QMAX, scale);
    for (int k = threadIdx.x; k < K; k += 256)
    {
        signed char h, l;
        det_quant_split(__half2float(w[(size_t) row * K + k]), inv, h, l);
        hi[(size_t) row * K + k] = h;
        lo[(size_t) row * K + k] = l;
    }
    if (threadIdx.x == 0) sb[row] = __fdiv_rn(scale, DET_QMAX);
}
#define MAX_NUM_EXPERTS 512
#define MAX_K 32
// Deterministic (FMA-only exp, correctly rounded division): the selection scores must agree
// bit for bit across tensor-parallel ranks of different architectures
__device__ __forceinline__
float sigmoid_stable_hf(float xf)
{
    float ez = exp_det(-fabsf(xf));
    float base = __fdiv_rn(ez, __fadd_rn(1.0f, ez));
    return (xf >= 0.0f) ? 1.0f - base : base;
}

// Score activations for the nogroup top-k kernels. Both are strictly increasing, so sorting
// by the raw logit when there is no selection bias remains valid for either
#define ROUTING_ACT_SIGMOID 0
#define ROUTING_ACT_SQRTSP 1

template <int ACT>
__device__ __forceinline__
float routing_act(float xf)
{
    if constexpr (ACT == ROUTING_ACT_SQRTSP)
    {
        // sqrt(softplus(x)), matching torch F.softplus(beta = 1, threshold = 20)
        return __fsqrt_rn(softplus_det(xf));
    }
    else
        return sigmoid_stable_hf(xf);
}


__device__ __forceinline__
void warp_reduce_best_f32(float& key, float& payload, int& idx)
{
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(USE_ROCM)
        // Monotonic unsigned encoding of the float key, hardware max-reduce, then fetch the
        // winner's values from the lowest tied lane
        unsigned int ku = __float_as_uint(key);
        ku = (ku & 0x80000000u) ? ~ku : (ku | 0x80000000u);
        unsigned int m = __reduce_max_sync(0xffffffffu, ku);
        int src = __ffs(__ballot_sync(0xffffffffu, ku == m)) - 1;
        key = __shfl_sync(0xffffffffu, key, src);
        payload = __shfl_sync(0xffffffffu, payload, src);
        idx = __shfl_sync(0xffffffffu, idx, src);
    #else
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_key = __shfl_down_sync(0xffffffffu, key, offset);
            float other_payload = __shfl_down_sync(0xffffffffu, payload, offset);
            int other_idx = __shfl_down_sync(0xffffffffu, idx, offset);
            if (other_key > key)
            {
                key = other_key;
                payload = other_payload;
                idx = other_idx;
            }
        }

        key = __shfl_sync(0xffffffffu, key, 0);
        payload = __shfl_sync(0xffffffffu, payload, 0);
        idx = __shfl_sync(0xffffffffu, idx, 0);
    #endif
}


// Top-K selection by one warp over num_candidates (<= MAX_NUM_EXPERTS) keys in shared memory.
// Each lane keeps its strided share of the keys in registers (at most MAX_NUM_EXPERTS / 32),
// every round picks the lane-local best, the warp reduces the round's winner and the owning
// lane retires it. Lane k ends up holding the round-k winner (payload, position). Ties: lowest
// position within a lane, lowest lane across the warp. One pass of K rounds replaces the
// per-warp pre-selection plus multi-warp merge stages: each stage cost K block-wide rounds, and
// for K > 16 a stage that keeps K of every 32 candidates could not shrink the field at all
__device__ __forceinline__
void warp_topk_shared
(
    const float* sh_key,
    const float* sh_payload,
    const int num_candidates,
    const int K,
    float& out_payload,
    int& out_idx
)
{
    constexpr int MAXC = MAX_NUM_EXPERTS / 32;
    int lane_id = threadIdx.x % 32;
    float ck[MAXC];
    #pragma unroll
    for (int j = 0; j < MAXC; ++j)
    {
        int pos = lane_id + 32 * j;
        ck[j] = pos < num_candidates ? sh_key[pos] : -1.0e30f;
    }
    out_payload = 0.0f;
    out_idx = -1;
    for (int k = 0; k < K; ++k)
    {
        float best_key = -1.0e30f;
        int best_j = -1;
        #pragma unroll
        for (int j = 0; j < MAXC; ++j)
            if (ck[j] > best_key) { best_key = ck[j]; best_j = j; }
        int my_pos = best_j >= 0 ? lane_id + 32 * best_j : -1;
        float best_payload = my_pos >= 0 ? (sh_payload ? sh_payload[my_pos] : best_key) : 0.0f;
        int best_idx = my_pos;
        warp_reduce_best_f32(best_key, best_payload, best_idx);
        if (my_pos >= 0 && my_pos == best_idx)
        {
            // Retire the winner (unrolled select: no dynamic register indexing)
            #pragma unroll
            for (int j = 0; j < MAXC; ++j)
                if (j == best_j) ck[j] = -1.0e30f;
        }
        if (lane_id == k)
        {
            out_payload = best_payload;
            out_idx = best_idx;
        }
    }
}


__device__ inline float warp_reduce_sum_first_k(float v, int K)
{
    int lane_id = threadIdx.x % 32;
    if (lane_id >= K) v = 0.0f;
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        v += other_v;
    }
    v = __shfl_sync(0xffffffffu, v, 0);
    return v;
}

#define RGEMV_WARPS 8

__global__ __launch_bounds__(RGEMV_WARPS * 32)
void routing_gemv_kernel
(
    const half* __restrict__ x,         // (k)
    const half* __restrict__ gate_t,    // (E, k)
    half* __restrict__ scores,          // (E)
    const int k,
    const int E
)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * RGEMV_WARPS + warp;
    if (row >= E) return;

    const half2* x2 = (const half2*) x;
    const half2* w2 = (const half2*) (gate_t + (size_t) row * k);

    float sum = 0.0f;
    for (int j = lane; j < k / 2; j += 32)
    {
        float2 xf = __half22float2(x2[j]);
        float2 wf = __half22float2(w2[j]);
        sum = fmaf(xf.x, wf.x, sum);
        sum = fmaf(xf.y, wf.y, sum);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    if (lane == 0)
        scores[row] = __float2half_rn(sum);
}
template <int ACT>
__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_ds3_nogroup_topk_kernel
(
    const half* __restrict__ scores,
    const half* __restrict__ bias,
    int32_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const float scaling_factor,
    const int num_experts,
    const int K,
    const int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    bool mask = t < num_experts;

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    float* sh_key = reinterpret_cast<float*>(sh);
    float* sh_payload = sh_key + num_experts;

    // Selection key (biased activation, or the raw logit without a bias: the activations are
    // monotonic) and the output payload for every expert
    if (mask)
    {
        float logit = __half2float(scores[t]);
        float act = bias ? routing_act<ACT>(logit) : 0.0f;
        sh_key[t] = bias ? act + __half2float(bias[t]) : logit;
        sh_payload[t] = bias ? act : logit;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float o;
        int out_idx;
        warp_topk_shared(sh_key, sh_payload, num_experts, K, o, out_idx);
        if (lane_id < K && !bias)
            o = routing_act<ACT>(o);

        float sum = warp_reduce_sum_first_k(o, K) + 1e-20f;
        if (lane_id < K)
        {
            topk_indices[lane_id] = (int32_t) out_idx;
            topk_weights[lane_id] = __float2half_rn(o * scaling_factor / sum);
        }
    }
}

}

int bn_cuda_router_prepare_launch(cudaStream_t stream,const void*gate,void*i8,float*sb,int E,int D){
 if(cudaFuncSetAttribute(routing_gemm_i8_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,RG_SMEM)!=cudaSuccess)return -1;
 quant_gate_kernel<<<E,256,0,stream>>>((const half*)gate,(signed char*)i8,(signed char*)i8+(size_t)E*D,sb,E,D);
 return cudaGetLastError()==cudaSuccess?0:-1;
}
int bn_cuda_router_run_launch(cudaStream_t stream,const void*x,const void*gate,const void*i8,const float*sb,const void*bias,void*logits,int32_t*ids,void*weights,void*ahi,void*alo,float*sa,float*part,int R,int D,int E,int K,float scaling){
 if(R==1){routing_gemv_kernel<<<CEIL_DIVIDE(E,8),256,0,stream>>>((const half*)x,(const half*)gate,(half*)logits,D,E);if(cudaGetLastError()!=cudaSuccess)return -1;}
 else{int KC=CEIL_DIVIDE(D,128),S=rg_slices(R,E,KC),ks=CEIL_DIVIDE(KC,S)*128,Se=CEIL_DIVIDE(D,ks);
 quant_a_kernel<<<CEIL_DIVIDE(R*KC*8,256),256,0,stream>>>((const half*)x,(signed char*)ahi,(signed char*)alo,sa,R,D,KC);if(cudaGetLastError()!=cudaSuccess)return -1;
 routing_gemm_i8_kernel<<<dim3(CEIL_DIVIDE(E,64),CEIL_DIVIDE(R,128),Se),512,RG_SMEM,stream>>>((const signed char*)ahi,(const signed char*)alo,sa,(const signed char*)i8,(const signed char*)i8+(size_t)E*D,sb,(half*)logits,Se>1?part:nullptr,R,E,D,KC,ks);if(cudaGetLastError()!=cudaSuccess)return -1;
 if(Se>1){routing_gemm_reduce_kernel<<<CEIL_DIVIDE(R*E,256),256,0,stream>>>(part,(half*)logits,R*E,Se);if(cudaGetLastError()!=cudaSuccess)return -1;}}
 routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SQRTSP><<<R,CEIL_DIVIDE(E,32)*32,E*8,stream>>>((const half*)logits,(const half*)bias,ids,(half*)weights,scaling,E,K,R);
 return cudaGetLastError()==cudaSuccess?0:-1;
}
