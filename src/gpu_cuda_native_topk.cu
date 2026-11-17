/*
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adapted from exllamav3 bd1f519172406a9560bd5c90e61000691bb69a07
exllamav3_ext/dsa_topk.cu. Device arithmetic and compaction preserved.
Host ownership uses explicit request workspace, with no global mutable state.
*/
#include "gpu_cuda_native_topk.h"
#include <cuda_fp16.h>
#include <climits>
#include <cstdint>
namespace bn_cuda_native_topk_detail {
#define TOPK_THREADS 1024

__device__ __forceinline__ uint16_t topk_key_u(const uint16_t u)
{
    return (u & 0x8000) ? (uint16_t) ~u : (uint16_t) (u | 0x8000);
}

// Descending radix bucket search, warp 0 only: find the highest bucket b such that the
// count of entries in buckets ABOVE b is < target <= count including b. Writes res[0] = b
// (-1 if the total is below target) and res[1] = count strictly above b. Lane l owns the
// descending 8-bucket group [255 - 8l .. 248 - 8l]; one shfl scan replaces the serial
// 256-bucket walk (a ~3us latency chain at the head of both histogram passes)
__device__ __forceinline__ void topk_find_bucket(const int* hist, int target, int lane, int* res)
{
    int g0 = 255 - 8 * lane;
    int gs = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) gs += hist[g0 - j];
    int incl = gs;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        int n = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += n;
    }
    int excl = incl - gs;
    bool cross = excl < target && incl >= target;
    unsigned bal = __ballot_sync(0xffffffffu, cross);
    if (!bal)
    {
        if (lane == 31) { res[0] = -1; res[1] = incl; }
        return;
    }
    if (lane == __ffs(bal) - 1)
    {
        int acc = excl;
        int b = g0;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            b = g0 - j;
            if (acc + hist[b] >= target) break;
            acc += hist[b];
        }
        res[0] = b;
        res[1] = acc;
    }
}

__device__ __forceinline__ uint16_t topk_key(const half h)
{
    return topk_key_u(__half_as_ushort(h));
}

template <bool VEC, bool Dynamic = false>
__global__ __launch_bounds__(TOPK_THREADS)
void dsa_topk_kernel
(
    const half* __restrict__ scores,     // (R, s_stride), -inf past each row's causal bound
    int* __restrict__ out,               // (R, k_pad) int32, -1 padded
    int T,
    const int s_stride,
    int k,
    const int k_pad,
    const int* __restrict__ t_ptr,       // device T override (graph modes)
    const int t_seq                      // > 0: t_ptr is per-JOB, T = t_ptr[row / t_seq]
                                         // (rows only scan their own causal region, so no
                                         // -inf backfill of the score buffer is needed)
)
{
    if (t_ptr) { T = t_seq > 0 ? t_ptr[blockIdx.x / t_seq] : *t_ptr; if constexpr(Dynamic) k = min(k, T); }
    constexpr uint16_t KEY_NEG_INF = 0x03ff;   // topk_key(0xfc00)

    const half* row_s = scores + (size_t) blockIdx.x * s_stride;
    int* row_o = out + (size_t) blockIdx.x * k_pad;
    int t = threadIdx.x;
    int lane = t % 32;
    int warp = t / 32;
    constexpr int NUM_WARPS = TOPK_THREADS / 32;

    __shared__ int hist[256];
    __shared__ int warp_cnt[NUM_WARPS];
    __shared__ int warp_off[NUM_WARPS];
    __shared__ int sh_res[2];
    __shared__ int sh_tile_total;

    // Pass 1: histogram of the high key byte. Vectorized: uint4 loads, 8 keys per LDG
    // (rows are 16-byte aligned: the host checks s_stride % 8 == 0), scalar tail
    const uint4* row_s8 = (const uint4*) row_s;
    const int T8 = VEC ? T >> 3 : 0;
    if (t < 256) hist[t] = 0;
    __syncthreads();
    for (int i = t; i < T8; i += TOPK_THREADS)
    {
        uint4 v = row_s8[i];
        uint32_t ws[4] = { v.x, v.y, v.z, v.w };
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint16_t ka = topk_key_u((uint16_t) ws[j]);
            uint16_t kb = topk_key_u((uint16_t) (ws[j] >> 16));
            if (ka > KEY_NEG_INF) atomicAdd(&hist[ka >> 8], 1);
            if (kb > KEY_NEG_INF) atomicAdd(&hist[kb >> 8], 1);
        }
    }
    for (int i = T8 * 8 + t; i < T; i += TOPK_THREADS)
    {
        uint16_t key = topk_key(row_s[i]);
        if (key > KEY_NEG_INF)
            atomicAdd(&hist[key >> 8], 1);
    }
    __syncthreads();
    if (warp == 0)
        topk_find_bucket(hist, k, lane, sh_res);
    __syncthreads();
    int b1 = sh_res[0];          // -1: fewer than k finite entries -> take them all
    int cnt_hi = sh_res[1];      // entries with high byte strictly above b1

    uint16_t thresh;
    int n_eq_take;
    if (b1 < 0)
    {
        // Everything finite is selected: threshold just above -inf, no tie cap
        thresh = KEY_NEG_INF;
        n_eq_take = 0;
    }
    else
    {
        // Pass 2: histogram of the low key byte among entries with high byte == b1
        // (hist reads of pass 1 all happened before the last barrier)
        if (t < 256) hist[t] = 0;
        __syncthreads();
        for (int i = t; i < T8; i += TOPK_THREADS)
        {
            uint4 v = row_s8[i];
            uint32_t ws[4] = { v.x, v.y, v.z, v.w };
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                uint16_t ka = topk_key_u((uint16_t) ws[j]);
                uint16_t kb = topk_key_u((uint16_t) (ws[j] >> 16));
                if (ka > KEY_NEG_INF && (ka >> 8) == b1) atomicAdd(&hist[ka & 0xff], 1);
                if (kb > KEY_NEG_INF && (kb >> 8) == b1) atomicAdd(&hist[kb & 0xff], 1);
            }
        }
        for (int i = T8 * 8 + t; i < T; i += TOPK_THREADS)
        {
            uint16_t key = topk_key(row_s[i]);
            if (key > KEY_NEG_INF && (key >> 8) == b1)
                atomicAdd(&hist[key & 0xff], 1);
        }
        __syncthreads();
        if (warp == 0)
            topk_find_bucket(hist, k - cnt_hi, lane, sh_res);
        __syncthreads();
        int b0 = max(sh_res[0], 0);             // in-band search cannot miss; clamp defensively
        thresh = (uint16_t) ((b1 << 8) | b0);
        n_eq_take = (k - cnt_hi) - sh_res[1];   // ties at v* still needed after all > v*
    }

    // Passes 3a/3b: ordered compaction, 8 elements per thread per tile (uint4 loads).
    // Position of element i is (tile, warp, lane, intra-lane bit). Monotonic in i, so
    // the ascending-index emission order is preserved. Per tile: one intra-warp shfl scan
    // + a warp-0 scan over warp totals; the running output base stays in a REGISTER
    // (block-uniform: everyone adds the same tile total read between the two barriers),
    // so a tile costs 2 barriers for 8 * TOPK_THREADS elements instead of 3 per
    // TOPK_THREADS. The thread whose uint4 slot straddles T picks up the scalar tail
    const int n_slots = (T + 7) >> 3;   // slots above T8 take the scalar path
    const int n_tiles = (n_slots + TOPK_THREADS - 1) / TOPK_THREADS;
    int base = 0;

    #pragma unroll 1
    for (int pass = 0; pass < 2; ++pass)
    {
        // pass 0: keys > thresh (all emitted); pass 1: ties == thresh, capped
        const int eq_start = base;
        const int cap_end = pass == 0 ? INT_MAX : eq_start + n_eq_take;
        for (int t0 = 0; t0 < n_tiles && base < cap_end; ++t0)
        {
            int i8 = t0 * TOPK_THREADS + t;
            unsigned m = 0;
            if (VEC && i8 < T8)
            {
                uint4 v = row_s8[i8];
                uint32_t ws[4] = { v.x, v.y, v.z, v.w };
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    uint16_t ka = topk_key_u((uint16_t) ws[j]);
                    uint16_t kb = topk_key_u((uint16_t) (ws[j] >> 16));
                    bool pa = pass == 0 ? (ka > thresh) : (ka == thresh);
                    bool pb = pass == 0 ? (kb > thresh) : (kb == thresh);
                    if (pa && ka > KEY_NEG_INF) m |= 1u << (2 * j);
                    if (pb && kb > KEY_NEG_INF) m |= 1u << (2 * j + 1);
                }
            }
            else if (i8 >= T8 && i8 < n_slots)
            {
                // Scalar strip: the straddling slot's tail (VEC) or every slot (!VEC).
                // Same 8-element ownership, so the emission order is unchanged
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    int idx = i8 * 8 + j;
                    if (idx >= T) break;
                    uint16_t kx = topk_key(row_s[idx]);
                    bool px = pass == 0 ? (kx > thresh) : (kx == thresh);
                    if (px && kx > KEY_NEG_INF) m |= 1u << j;
                }
            }
            int c = __popc(m);
            int incl = c;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1)
            {
                int n = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += n;
            }
            if (lane == 31) warp_cnt[warp] = incl;
            __syncthreads();
            if (warp == 0)
            {
                int wc = lane < NUM_WARPS ? warp_cnt[lane] : 0;
                int wincl = wc;
                #pragma unroll
                for (int o = 1; o < 32; o <<= 1)
                {
                    int n = __shfl_up_sync(0xffffffffu, wincl, o);
                    if (lane >= o) wincl += n;
                }
                if (lane < NUM_WARPS) warp_off[lane] = wincl - wc;
                if (lane == 31) sh_tile_total = wincl;
            }
            __syncthreads();
            int obase = base + warp_off[warp] + (incl - c);
            int ibase = i8 * 8;
            while (m)
            {
                int j = __ffs((int) m) - 1;
                m &= m - 1;
                if (obase < cap_end)
                    row_o[obase] = ibase + j;
                ++obase;
            }
            base += sh_tile_total;      // uniform: read between the barriers, added by all
        }
        if (pass == 1)
            base = min(base, cap_end);
    }
    int emitted = base;

    // -1 padding
    for (int j = emitted + t; j < k_pad; j += TOPK_THREADS)
        row_o[j] = -1;
}


/*
Split/merge variant for few-row selections (decode): the single-block-per-row kernel is
parallelism-starved when R is small and T is large (three sequential sweeps of T by one
block). Phase 1 selects each of TOPK_SPLIT_G spans' local top-k in parallel (a guaranteed
superset of the row's global top-k) and phase 2 selects over the <= G * k candidates.
Candidates are stored span-major with in-span ascending order, so the merge's ordered
compaction resolves ties in ascending global index order, identical to the legacy kernel.
*/

#define TOPK_SPLIT_G 32
#define TOPK_SPLIT_MIN_T 32768
#define TOPK_SPLIT_MAX_R 16
#define TOPK_SPLIT_MAX_KPAD 4096

template <bool VEC, bool Dynamic = false>
__global__ __launch_bounds__(TOPK_THREADS)
void dsa_topk_split_kernel
(
    const half* __restrict__ scores,
    int* __restrict__ ws_idx,            // (R, G, k_pad) candidate indices, span-major
    half* __restrict__ ws_scr,           // (R, G, k_pad) candidate scores (coalesced merge reads)
    int* __restrict__ ws_cnt,            // (R, G) candidate counts
    int T,
    const int s_stride,
    int k,
    const int k_pad,
    const int* __restrict__ t_ptr,       // device T override (graph modes, t_seq == 0 only)
    const int ws_G,                      // tile mode (ws_slot >= 0): workspace slots per row,
    const int ws_slot,                   //   the slot this launch fills (grid (R, 1)),
    const int idx_offset                 //   and the global index of the tile's first entry
)
{
    if (t_ptr) { T = *t_ptr; if constexpr(Dynamic) k = min(k, T); }
    constexpr uint16_t KEY_NEG_INF = 0x03ff;

    const bool tile = ws_slot >= 0;
    const int g = tile ? ws_slot : blockIdx.y;
    const int G = tile ? ws_G : gridDim.y;
    const int span = tile ? T : (((T + G - 1) / G + 7) & ~7);           // 8-aligned spans
    const int beg = tile ? 0 : g * span;
    const int Tl = min(beg + span, T) - beg;                             // local length
    int* row_ws = ws_idx + ((size_t) blockIdx.x * G + g) * k_pad;
    half* row_scr = ws_scr + ((size_t) blockIdx.x * G + g) * k_pad;
    int* cnt_out = ws_cnt + (size_t) blockIdx.x * G + g;

    int t = threadIdx.x;
    if (Tl <= 0)
    {
        if (t == 0) *cnt_out = 0;
        return;
    }

    const half* row_s = scores + (size_t) blockIdx.x * s_stride + beg;   // 16B aligned (beg % 8 == 0)
    int lane = t % 32;
    int warp = t / 32;
    constexpr int NUM_WARPS = TOPK_THREADS / 32;

    __shared__ int hist[256];
    __shared__ int warp_cnt[NUM_WARPS];
    __shared__ int warp_off[NUM_WARPS];
    __shared__ int sh_res[2];
    __shared__ int sh_tile_total;

    const uint4* row_s8 = (const uint4*) row_s;
    const int T8 = VEC ? Tl >> 3 : 0;
    if (t < 256) hist[t] = 0;
    __syncthreads();
    for (int i = t; i < T8; i += TOPK_THREADS)
    {
        uint4 v = row_s8[i];
        uint32_t w[4] = { v.x, v.y, v.z, v.w };
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint16_t ka = topk_key_u((uint16_t) w[j]);
            uint16_t kb = topk_key_u((uint16_t) (w[j] >> 16));
            if (ka > KEY_NEG_INF) atomicAdd(&hist[ka >> 8], 1);
            if (kb > KEY_NEG_INF) atomicAdd(&hist[kb >> 8], 1);
        }
    }
    for (int i = T8 * 8 + t; i < Tl; i += TOPK_THREADS)
    {
        uint16_t key = topk_key(row_s[i]);
        if (key > KEY_NEG_INF)
            atomicAdd(&hist[key >> 8], 1);
    }
    __syncthreads();
    if (warp == 0)
        topk_find_bucket(hist, k, lane, sh_res);
    __syncthreads();
    int b1 = sh_res[0];
    int cnt_hi = sh_res[1];

    uint16_t thresh;
    int n_eq_take;
    if (b1 < 0)
    {
        thresh = KEY_NEG_INF;
        n_eq_take = 0;
    }
    else
    {
        if (t < 256) hist[t] = 0;
        __syncthreads();
        for (int i = t; i < T8; i += TOPK_THREADS)
        {
            uint4 v = row_s8[i];
            uint32_t w[4] = { v.x, v.y, v.z, v.w };
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                uint16_t ka = topk_key_u((uint16_t) w[j]);
                uint16_t kb = topk_key_u((uint16_t) (w[j] >> 16));
                if (ka > KEY_NEG_INF && (ka >> 8) == b1) atomicAdd(&hist[ka & 0xff], 1);
                if (kb > KEY_NEG_INF && (kb >> 8) == b1) atomicAdd(&hist[kb & 0xff], 1);
            }
        }
        for (int i = T8 * 8 + t; i < Tl; i += TOPK_THREADS)
        {
            uint16_t key = topk_key(row_s[i]);
            if (key > KEY_NEG_INF && (key >> 8) == b1)
                atomicAdd(&hist[key & 0xff], 1);
        }
        __syncthreads();
        if (warp == 0)
            topk_find_bucket(hist, k - cnt_hi, lane, sh_res);
        __syncthreads();
        int b0 = max(sh_res[0], 0);
        thresh = (uint16_t) ((b1 << 8) | b0);
        n_eq_take = (k - cnt_hi) - sh_res[1];
    }

    const int n_slots = (Tl + 7) >> 3;
    const int n_tiles = (n_slots + TOPK_THREADS - 1) / TOPK_THREADS;
    int base = 0;

    #pragma unroll 1
    for (int pass = 0; pass < 2; ++pass)
    {
        const int eq_start = base;
        const int cap_end = pass == 0 ? INT_MAX : eq_start + n_eq_take;
        for (int t0 = 0; t0 < n_tiles && base < cap_end; ++t0)
        {
            int i8 = t0 * TOPK_THREADS + t;
            unsigned m = 0;
            if (VEC && i8 < T8)
            {
                uint4 v = row_s8[i8];
                uint32_t w[4] = { v.x, v.y, v.z, v.w };
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    uint16_t ka = topk_key_u((uint16_t) w[j]);
                    uint16_t kb = topk_key_u((uint16_t) (w[j] >> 16));
                    bool pa = pass == 0 ? (ka > thresh) : (ka == thresh);
                    bool pb = pass == 0 ? (kb > thresh) : (kb == thresh);
                    if (pa && ka > KEY_NEG_INF) m |= 1u << (2 * j);
                    if (pb && kb > KEY_NEG_INF) m |= 1u << (2 * j + 1);
                }
            }
            else if (i8 >= T8 && i8 < n_slots)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    int idx = i8 * 8 + j;
                    if (idx >= Tl) break;
                    uint16_t kx = topk_key(row_s[idx]);
                    bool px = pass == 0 ? (kx > thresh) : (kx == thresh);
                    if (px && kx > KEY_NEG_INF) m |= 1u << j;
                }
            }
            int c = __popc(m);
            int incl = c;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1)
            {
                int n = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += n;
            }
            if (lane == 31) warp_cnt[warp] = incl;
            __syncthreads();
            if (warp == 0)
            {
                int wc = lane < NUM_WARPS ? warp_cnt[lane] : 0;
                int wincl = wc;
                #pragma unroll
                for (int o = 1; o < 32; o <<= 1)
                {
                    int n = __shfl_up_sync(0xffffffffu, wincl, o);
                    if (lane >= o) wincl += n;
                }
                if (lane < NUM_WARPS) warp_off[lane] = wincl - wc;
                if (lane == 31) sh_tile_total = wincl;
            }
            __syncthreads();
            int obase = base + warp_off[warp] + (incl - c);
            int ibase = i8 * 8;
            while (m)
            {
                int j = __ffs((int) m) - 1;
                m &= m - 1;
                if (obase < cap_end)
                {
                    row_ws[obase] = idx_offset + beg + ibase + j;   // global index
                    row_scr[obase] = row_s[ibase + j];
                }
                ++obase;
            }
            base += sh_tile_total;
        }
        if (pass == 1)
            base = min(base, cap_end);
    }
    if (t == 0)
        *cnt_out = base;
}

template<bool Dynamic>
__global__ __launch_bounds__(TOPK_THREADS)
void dsa_topk_merge_kernel
(
    const int* __restrict__ ws_idx,      // (R, G, k_pad)
    const half* __restrict__ ws_scr,     // (R, G, k_pad)
    const int* __restrict__ ws_cnt,      // (R, G)
    int* __restrict__ out,               // (R, k_pad), -1 padded
    const int G,
    int k,
    const int k_pad,
    half* __restrict__ out_scr,          // optional (R, k_pad): scores of the emitted indices
    int* __restrict__ out_cnt,           // optional (R,): number emitted (<= k)
    const int out_stride,                // row stride of out / out_scr (a workspace slot view)
    const int out_cnt_stride,
    const int* active_count
)
{
    const int* row_ws = ws_idx + (size_t) blockIdx.x * G * k_pad;
    const half* row_scr = ws_scr + (size_t) blockIdx.x * G * k_pad;
    const int* row_cnt = ws_cnt + (size_t) blockIdx.x * G;
    int* row_o = out + (size_t) blockIdx.x * out_stride;
    half* row_os = out_scr ? out_scr + (size_t) blockIdx.x * out_stride : nullptr;
    int t = threadIdx.x;
    if constexpr(Dynamic) {
        k = min(k, *active_count);
        if (!k) {
            for (int j=t; j<k_pad; j+=TOPK_THREADS) row_o[j]=-1;
            if(out_cnt && t==0)out_cnt[(size_t)blockIdx.x*out_cnt_stride]=0;
            return;
        }
    }

    int lane = t % 32;
    int warp = t / 32;
    constexpr int NUM_WARPS = TOPK_THREADS / 32;

    __shared__ int hist[256];
    __shared__ int cnt_sh[TOPK_SPLIT_G];
    __shared__ int warp_cnt[NUM_WARPS];
    __shared__ int warp_off[NUM_WARPS];
    __shared__ int sh_res[2];
    __shared__ int sh_tile_total;

    if (t < G) cnt_sh[t] = row_cnt[t];
    if (t < 256) hist[t] = 0;
    __syncthreads();

    int total = 0;
    for (int g = 0; g < G; ++g) total += cnt_sh[g];

    if (total >= k)
    {
        // Pass 1: high-byte histogram over the candidates (all finite by construction)
        for (int g = 0; g < G; ++g)
        {
            int cg = cnt_sh[g];
            for (int i = t; i < cg; i += TOPK_THREADS)
            {
                uint16_t key = topk_key(row_scr[g * k_pad + i]);
                atomicAdd(&hist[key >> 8], 1);
            }
        }
        __syncthreads();
        if (warp == 0)
            topk_find_bucket(hist, k, lane, sh_res);
        __syncthreads();
        int b1 = sh_res[0];
        int cnt_hi = sh_res[1];

        if (t < 256) hist[t] = 0;
        __syncthreads();
        for (int g = 0; g < G; ++g)
        {
            int cg = cnt_sh[g];
            for (int i = t; i < cg; i += TOPK_THREADS)
            {
                uint16_t key = topk_key(row_scr[g * k_pad + i]);
                if ((key >> 8) == b1) atomicAdd(&hist[key & 0xff], 1);
            }
        }
        __syncthreads();
        if (warp == 0)
            topk_find_bucket(hist, k - cnt_hi, lane, sh_res);
        __syncthreads();
        int b0 = max(sh_res[0], 0);
        uint16_t thresh = (uint16_t) ((b1 << 8) | b0);
        int n_eq_take = (k - cnt_hi) - sh_res[1];

        // Ordered compaction over candidates, span-major = ascending global index. One
        // candidate per thread per tile; tie order matches the legacy kernel exactly
        int base = 0;
        #pragma unroll 1
        for (int pass = 0; pass < 2; ++pass)
        {
            const int eq_start = base;
            const int cap_end = pass == 0 ? INT_MAX : eq_start + n_eq_take;
            for (int g = 0; g < G && base < cap_end; ++g)
            {
                int cg = cnt_sh[g];
                for (int c0 = 0; c0 < cg && base < cap_end; c0 += TOPK_THREADS)
                {
                    int i = c0 + t;
                    int gidx = -1;
                    half gscr = __ushort_as_half(0xfc00);
                    bool sel = false;
                    if (i < cg)
                    {
                        gidx = row_ws[g * k_pad + i];
                        gscr = row_scr[g * k_pad + i];
                        uint16_t key = topk_key(gscr);
                        sel = pass == 0 ? (key > thresh) : (key == thresh);
                    }
                    int c = sel ? 1 : 0;
                    int incl = c;
                    #pragma unroll
                    for (int o = 1; o < 32; o <<= 1)
                    {
                        int n = __shfl_up_sync(0xffffffffu, incl, o);
                        if (lane >= o) incl += n;
                    }
                    if (lane == 31) warp_cnt[warp] = incl;
                    __syncthreads();
                    if (warp == 0)
                    {
                        int wc = lane < NUM_WARPS ? warp_cnt[lane] : 0;
                        int wincl = wc;
                        #pragma unroll
                        for (int o = 1; o < 32; o <<= 1)
                        {
                            int n = __shfl_up_sync(0xffffffffu, wincl, o);
                            if (lane >= o) wincl += n;
                        }
                        if (lane < NUM_WARPS) warp_off[lane] = wincl - wc;
                        if (lane == 31) sh_tile_total = wincl;
                    }
                    __syncthreads();
                    if (sel)
                    {
                        int obase = base + warp_off[warp] + (incl - c);
                        if (obase < cap_end)
                        {
                            row_o[obase] = gidx;
                            if (row_os) row_os[obase] = gscr;
                        }
                    }
                    base += sh_tile_total;
                }
            }
            if (pass == 1)
                base = min(base, cap_end);
        }
        for (int j = base + t; j < k_pad; j += TOPK_THREADS)
            row_o[j] = -1;
        if (out_cnt && t == 0) out_cnt[(size_t) blockIdx.x * out_cnt_stride] = base;
    }
    else
    {
        // Fewer candidates than k: emit them all in order, pad the rest
        int base = 0;
        for (int g = 0; g < G; ++g)
        {
            int cg = cnt_sh[g];
            for (int i = t; i < cg; i += TOPK_THREADS)
            {
                row_o[base + i] = row_ws[g * k_pad + i];
                if (row_os) row_os[base + i] = row_scr[g * k_pad + i];
            }
            base += cg;
        }
        for (int j = base + t; j < k_pad; j += TOPK_THREADS)
            row_o[j] = -1;
        if (out_cnt && t == 0) out_cnt[(size_t) blockIdx.x * out_cnt_stride] = base;
    }
}


}
int bn_cuda_native_index_topk(cudaStream_t stream,const void*s,void*out,void*wi,void*ws,void*wc,
                              int R,int T,int stride,int k,int pad,int groups,const void*active) {
    using namespace bn_cuda_native_topk_detail;
    if(!k)return cudaMemsetAsync(out,255,(size_t)R*pad*sizeof(int),stream)==cudaSuccess?0:-1;
    bool vec=stride%8==0&&((uintptr_t)s)%16==0;
#define BN_TOPK_SPLIT(D) dsa_topk_split_kernel<true,D><<<dim3(R,groups),TOPK_THREADS,0,stream>>>((const half*)s,(int*)wi,(half*)ws,(int*)wc,T,stride,k,pad,(const int*)active,0,-1,0)
#define BN_TOPK_MERGE(D) dsa_topk_merge_kernel<D><<<R,TOPK_THREADS,0,stream>>>((const int*)wi,(const half*)ws,(const int*)wc,(int*)out,groups,k,pad,nullptr,nullptr,pad,1,(const int*)active)
#define BN_TOPK_REG(V,D) dsa_topk_kernel<V,D><<<R,TOPK_THREADS,0,stream>>>((const half*)s,(int*)out,T,stride,k,pad,(const int*)active,0)
    if(groups&&vec) {
        if(active){BN_TOPK_SPLIT(true);}else{BN_TOPK_SPLIT(false);}
        if(cudaGetLastError()!=cudaSuccess)return -1;
        if(active){BN_TOPK_MERGE(true);}else{BN_TOPK_MERGE(false);}
    } else if(vec) {
        if(active){BN_TOPK_REG(true,true);}else{BN_TOPK_REG(true,false);}
    } else {
        if(active){BN_TOPK_REG(false,true);}else{BN_TOPK_REG(false,false);}
    }
#undef BN_TOPK_REG
#undef BN_TOPK_MERGE
#undef BN_TOPK_SPLIT
    return cudaGetLastError()==cudaSuccess?0:-1;
}
#undef TOPK_THREADS
#undef TOPK_SPLIT_G
#undef TOPK_SPLIT_MIN_T
#undef TOPK_SPLIT_MAX_R
#undef TOPK_SPLIT_MAX_KPAD
