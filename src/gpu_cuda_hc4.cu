/* Adapted from exllamav3 hc_mix.cu, commit bd1f519172406a9560bd5c90e61000691bb69a07.
 * Copyright (c) 2025 Turboderp. MIT: third_party/exllamav3/LICENSE.
 * This translation unit alone uses the original --use_fast_math policy.
 * F32 model-owned fn is rounded on read for R<=32; no cached half copy. */
#include <cuda_fp16.h>
#include <type_traits>
#include <algorithm>
#include "gpu_cuda_hc4.h"
#define NUM_THREADS 256
#define NUM_THREADS_A 64
namespace {
#include "gpu_cuda_hc4_lanes.cuh"
__device__ __forceinline__ float sigmoidf_(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

template <int H, int M_, bool ROUND_HALF>
__global__ __launch_bounds__(NUM_THREADS_A)
void hc_mix_partials_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const float* __restrict__ fn,         // (M, H * D) F32; ROUND_HALF preserves upstream half arithmetic
    float* __restrict__ partials,        // (R, chunksA, M + 1)
    const int row_len,
    const int chunk_cols                 // multiple of 4 * NUM_THREADS_A
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, row_len);

    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int row_len4 = row_len / 4;

    float acc[M + 1];
    #pragma unroll
    for (int k = 0; k <= M; ++k) acc[k] = 0.0f;

    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS_A)
    {
        float4 s = s4[c];
        acc[M] = fmaf(s.x, s.x, acc[M]);
        acc[M] = fmaf(s.y, s.y, acc[M]);
        acc[M] = fmaf(s.z, s.z, acc[M]);
        acc[M] = fmaf(s.w, s.w, acc[M]);
        #pragma unroll
        for (int j = 0; j < M; ++j)
        {
            float4 w;
            w = ((const float4*) fn)[(size_t) j * row_len4 + c];
            if constexpr (ROUND_HALF) {
                w.x = __half2float(__float2half_rn(w.x));
                w.y = __half2float(__float2half_rn(w.y));
                w.z = __half2float(__float2half_rn(w.z));
                w.w = __half2float(__float2half_rn(w.w));
            }
            float d = fmaf(s.x, w.x, fmaf(s.y, w.y, fmaf(s.z, w.z, s.w * w.w)));
            acc[j] += d;
        }
    }

    // Block reduce M + 1 lanes' accumulators
    __shared__ float red[NUM_THREADS_A / 32][M + 1];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    #pragma unroll
    for (int k = 0; k <= M; ++k)
    {
        float v = hc_warp_sum_lane0(acc[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();

    if (warp == 0)
    {
        float* out = partials + ((size_t) r * gridDim.x + blockIdx.x) * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_THREADS_A / 32; ++w)
                v += red[w][k];
            out[k] = v;
        }
    }
}

template <int H, int M_, bool HEAD, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void hc_mix_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const float* __restrict__ partials,  // (R, chunksA, M + 1)
    const float* __restrict__ base,      // (M)
    const float* __restrict__ scale,     // (3)
    float* __restrict__ post,            // (R, H)
    float* __restrict__ comb,            // (R, H, H)
    void* __restrict__ collapsed,        // (R, D) float or half
    const int D,
    const int chunksA,
    const int chunk_cols_c,              // multiple of 4
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters,
    const float* __restrict__ pre_in,    // (R, H) carried pre weights (carry 1), may be null
    float* __restrict__ pre_out,         // (R, H) this site's own pre weights, written when non-null
    const int carry                     // collapse with 0: own pre, 1: pre_in, 2: stream 0 (first carried site)
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int row_len = H * D;

    // Re-reduce this row's partials (tiny, L2-resident). Every block does this to avoid
    // cross-block dependencies. Warp-split over the chunk axis: the serial loop sits
    // at the head of the kernel's critical path, so with many partials chunks (small
    // NUM_THREADS_A blocks) a single-thread-per-quantity loop is too long
    __shared__ float mix_s[M + 1];
    __shared__ float pre_s[H];
    __shared__ float red_s[NUM_THREADS / 32][M + 1];
    {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane <= M)
        {
            const float* p = partials + (size_t) r * chunksA * (M + 1) + lane;
            float v = 0.0f;
            for (int i = warp; i < chunksA; i += NUM_THREADS / 32)
                v += p[(size_t) i * (M + 1)];
            red_s[warp][lane] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x <= M)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_THREADS / 32; ++w)
            v += red_s[w][threadIdx.x];
        mix_s[threadIdx.x] = v;
    }
    __syncthreads();
    float rmr = rsqrtf(mix_s[M] / (float) row_len + rms_eps);
    if (threadIdx.x < H)
        pre_s[threadIdx.x] = sigmoidf_(fmaf(mix_s[threadIdx.x] * rmr, scale[0], base[threadIdx.x])) + hc_eps;
    __syncthreads();
    // Carried pre weights (mHC with carry_pre): a site collapses with the previous site's pre weights and
    // hands its own on through pre_out
    if (pre_out && blockIdx.x == 0 && threadIdx.x < H)
        pre_out[(size_t) r * H + threadIdx.x] = pre_s[threadIdx.x];

    // Sinkhorn + post/comb writes: chunk-0 block only, one lane per comb element. Runs in
    // a dedicated warp, concurrent with the other warps' phase C: the ~20-iteration
    // normalization is a serial shfl/div latency chain that only needs the M + 1 reduced
    // scalars, so at small R it sets the kernel's critical path; don't stack phase C
    // work in front of it.
    //
    // Lane l = i * H + j; row sums reduce over j (xor 1..H/2), col sums over i (xor H..).
    const bool sink_warp = !HEAD && blockIdx.x == 0 && threadIdx.x < 32;
    if (sink_warp)
    {
        if (threadIdx.x < H)
            post[(size_t) r * H + threadIdx.x] =
                2.0f * sigmoidf_(fmaf(mix_s[H + threadIdx.x] * rmr, scale[1], base[H + threadIdx.x]));

        if (threadIdx.x < H * H)
        {
            const unsigned mask = (H * H == 32) ? 0xffffffffu : ((1u << (H * H)) - 1u);
            float v = fmaf(mix_s[2 * H + threadIdx.x] * rmr, scale[2], base[2 * H + threadIdx.x]);

            // softmax over rows
            float m = hc_xor_max<1, H>(v, mask);
            v = __expf(v - m);
            float s = hc_xor_sum<1, H>(v, mask);
            v = __fdividef(v, s) + hc_eps;

            // column normalize, then (iters - 1) x (row, column)
            float cs = hc_xor_sum<H, H * H>(v, mask);
            v = __fdividef(v, cs + hc_eps);
            for (int it = 0; it < sinkhorn_iters - 1; ++it)
            {
                float rs = hc_xor_sum<1, H>(v, mask);
                v = __fdividef(v, rs + hc_eps);
                cs = hc_xor_sum<H, H * H>(v, mask);
                v = __fdividef(v, cs + hc_eps);
            }
            comb[(size_t) r * H * H + threadIdx.x] = v;
        }
        return;
    }

    // Phase C: collapsed chunk, weighted sum over the H stream rows. In the sinkhorn
    // block the first warp is excluded, so the remaining threads re-cover its lanes
    float pre_r[H];
    #pragma unroll
    for (int h = 0; h < H; ++h)
        pre_r[h] = carry == 0 ? pre_s[h] : carry == 1 ? pre_in[(size_t) r * H + h] : (h == 0 ? 1.0f : 0.0f);

    const bool shrunk = !HEAD && blockIdx.x == 0;
    const int tid = shrunk ? threadIdx.x - 32 : threadIdx.x;
    const int nth = shrunk ? NUM_THREADS - 32 : NUM_THREADS;
    const int c0 = blockIdx.x * chunk_cols_c;
    const int c1 = min(c0 + chunk_cols_c, D);
    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int D4 = D / 4;
    for (int c = c0 / 4 + tid; c < c1 / 4; c += nth)
    {
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            o.x = fmaf(pre_r[h], s.x, o.x);
            o.y = fmaf(pre_r[h], s.y, o.y);
            o.z = fmaf(pre_r[h], s.z, o.z);
            o.w = fmaf(pre_r[h], s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) collapsed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) collapsed + (size_t) r * D))[c] = o;
    }
}
template <int H, typename Y_T, bool HAS_COMB, bool XW>
__global__ __launch_bounds__(NUM_THREADS)
void hc_apply_kernel
(
    float* __restrict__ x,               // (R, H, D), updated in place
    const Y_T* __restrict__ y,           // (R, D) float or half
    const float* __restrict__ post,      // (R, H)
    const float* __restrict__ comb,      // (R, H, H), or null
    const half* __restrict__ wn,         // (H, D) next site's norm weight, or null
    float* __restrict__ xw,              // (R, H, D) weighted copy out, or null
    const int D,
    const int chunk_cols                 // multiple of 4
)
{
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;

    float post_r[H];
    float comb_r[H][H];
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        post_r[h] = __ldg(post + (size_t) r * H + h);
        if (HAS_COMB)
        {
            #pragma unroll
            for (int g = 0; g < H; ++g)
                comb_r[h][g] = __ldg(comb + ((size_t) r * H + h) * H + g);
        }
    }

    float4* x4 = (float4*) (x + (size_t) r * H * D);
    float4* xw4 = XW ? (float4*) (xw + (size_t) r * H * D) : nullptr;
    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS)
    {
        float4 xv[H];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            xv[h] = x4[(size_t) h * D4 + c];

        float4 yv;
        if constexpr (std::is_same_v<Y_T, half>)
        {
            half2 y01 = ((const half2*) (y + (size_t) r * D))[c * 2];
            half2 y23 = ((const half2*) (y + (size_t) r * D))[c * 2 + 1];
            float2 lo = __half22float2(y01);
            float2 hi = __half22float2(y23);
            yv = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        else
            yv = ((const float4*) (y + (size_t) r * D))[c];

        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 o;
            o.x = post_r[h] * yv.x;
            o.y = post_r[h] * yv.y;
            o.z = post_r[h] * yv.z;
            o.w = post_r[h] * yv.w;
            if (HAS_COMB)
            {
                #pragma unroll
                for (int g = 0; g < H; ++g)
                {
                    o.x = fmaf(comb_r[g][h], xv[g].x, o.x);
                    o.y = fmaf(comb_r[g][h], xv[g].y, o.y);
                    o.z = fmaf(comb_r[g][h], xv[g].z, o.z);
                    o.w = fmaf(comb_r[g][h], xv[g].w, o.w);
                }
            }
            else
            {
                o.x += xv[h].x;
                o.y += xv[h].y;
                o.z += xv[h].z;
                o.w += xv[h].w;
            }
            x4[(size_t) h * D4 + c] = o;
            if constexpr (XW)
            {
                const half2* w2 = (const half2*) (wn + (size_t) h * D + c * 4);
                float2 w01 = __half22float2(w2[0]), w23 = __half22float2(w2[1]);
                xw4[(size_t) h * D4 + c] = make_float4(__fmul_rn(o.x, w01.x), __fmul_rn(o.y, w01.y),
                                                       __fmul_rn(o.z, w23.x), __fmul_rn(o.w, w23.y));
            }
        }
    }
}


}
int bn_cuda_hc4_mix_launch(cudaStream_t stream,const float*x,const float*fn,const float*base,const float*scale,float*partials,float*post,float*comb,float*out,const float*pi,float*po,int R,int D,int n,float eps,float hc_eps,int iters){
    int c=((4*D/n+255)/256)*256;
    int nc=std::min(32,std::max(1,256/R));int cc=((D/nc+1023)/1024)*1024;nc=(D+cc-1)/cc;
    if(R<=32)hc_mix_partials_kernel<4,24,true><<<dim3(n,R),64,0,stream>>>(x,fn,partials,4*D,c);
    else hc_mix_partials_kernel<4,24,false><<<dim3(n,R),64,0,stream>>>(x,fn,partials,4*D,c);
    if(cudaGetLastError()!=cudaSuccess)return -1;
    hc_mix_finalize_kernel<4,24,false,false><<<dim3(nc,R),256,0,stream>>>(x,partials,base,scale,post,comb,out,D,n,cc,eps,hc_eps,iters,pi,po,pi?1:2);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
int bn_cuda_hc4_apply_launch(cudaStream_t stream,float*x,const void*y,const float*post,const float*comb,int R,int D,int half_y){
    int nc=std::min(32,std::max(1,256/R));int cc=((D/nc+1023)/1024)*1024;nc=(D+cc-1)/cc;
    if(half_y)hc_apply_kernel<4,half,true,false><<<dim3(nc,R),256,0,stream>>>(x,(const half*)y,post,comb,nullptr,nullptr,D,cc);
    else hc_apply_kernel<4,float,true,false><<<dim3(nc,R),256,0,stream>>>(x,(const float*)y,post,comb,nullptr,nullptr,D,cc);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
