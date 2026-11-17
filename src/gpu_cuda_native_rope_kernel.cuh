/* Original exllamav3 bd1f5191 arithmetic; MIT third_party/exllamav3/LICENSE. */
#pragma once
#define MAX_NUM_THREADS 1024
#define MAX_ROTATE_DIMS 4
#define ROPESTYLE_GPTJ 1
#define ROPESTYLE_NEOX 2
using bfloat16=__nv_bfloat16;using bfloat162=__nv_bfloat162;
__device__ __forceinline__ void sincos_accurate(float x, float* s, float* c)
{
    const float k = rintf(x * 0.15915494309189535f);
    float r = fmaf(-k, 6.28318548202514648f, x);
    r = fmaf(-k, -1.74845553e-7f, r);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __sincosf(r, s, c);
#else
    // Host pass of a .cpp that includes this header: the body is parsed, never run, and the
    // intrinsic is only declared under the device compilers (glibc declares a __sincosf of its
    // own, MSVC's CRT does not)
    *s = sinf(r);
    *c = cosf(r);
#endif
}
__device__ inline float warp_reduce_sum_f(float v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        v += other_v;
    }
    return v;
}
template <int rope_mode, bool norm_bf16>
__global__
void rope_kernel
(
    const half* __restrict__ q,
    half* __restrict__ out_q,
    const half* __restrict__ k,
    half* __restrict__ out_k,
    const float* __restrict__ inv_freq,
    const int bsz,
    const int seq_len,
    const int num_heads_q,
    const int num_heads_k,
    const int head_dim,
    const int q_head_stride,
    const int k_head_stride,
    const int partial_head_dim,
    const int position,
    const uint32_t* __restrict__ positions,
    const uint32_t* __restrict__ position_ids,
    float attn_factor,
    const void* __restrict__ q_norm,
    const void* __restrict__ k_norm,
    const float norm_eps,
    const float norm_constant_bias,
    const bool inv_freq_table,
    const int inv_freq_stride,
    const float llama_4_scaling_beta,
    const int llama_4_scaling_original,
    int position_ids_stride,
    int rotate_dims,
    int rotate_offset
)
{
    int batch = blockIdx.y;
    int token_pos = blockIdx.x;
    int t = threadIdx.x;
    int t_head = threadIdx.y;

    auto get_pos = [&] (int rdim) -> int
    {
        int pos = token_pos + position;
        if (positions)
            pos = token_pos + positions[batch];
        else if (position_ids)
        {
            int idx = (batch * seq_len + token_pos) * position_ids_stride;
            if (position_ids_stride > 1) idx += rdim;
            pos = position_ids[idx];
        }
        return pos;
    };

    auto get_sincos = [&] (int rdim, float &sin, float &cos)
    {
        int pos = get_pos(rdim);
        if (!inv_freq_table)
        {
            float fr = inv_freq[t];
            float pf = __int2float_rn(pos);
            sincos_accurate(fr * pf, &sin, &cos);
            sin *= attn_factor;
            cos *= attn_factor;
        }
        else
        {
            float fr = inv_freq[batch * inv_freq_stride + pos * partial_head_dim / 2 + t];
            sincos_accurate(fr, &sin, &cos);
            sin *= attn_factor;
            cos *= attn_factor;
        }
    };

    // Llama 4 scaling: position-dependent scale on the full query head, queries only (the
    // reference scales query_states after rope; keys enter the cache unscaled)
    float l4_scaling = 1.0f;
    if (llama_4_scaling_beta > 0.0f)
        l4_scaling = 1.0f + llama_4_scaling_beta * __logf(1.0f + (float)(get_pos(0) / llama_4_scaling_original));

    float sin_cache[MAX_ROTATE_DIMS];
    float cos_cache[MAX_ROTATE_DIMS];
    if (t < partial_head_dim / 2)
    {
        for (int rdim = 0; rdim < rotate_dims; ++rdim)
        {
            get_sincos(rdim, sin_cache[rdim], cos_cache[rdim]);
        }
    }

    // Shared buffer
    __shared__ half norm_head[MAX_NUM_THREADS * 2];
    __shared__ float sums[MAX_NUM_THREADS / 32];

    // Prep
    int head_dim_pad = (head_dim + 63) / 64 * 64;

    // Loop over heads
    // Heads are distributed over gridDim.z as well as blockDim.y: at decode (seq 1, bsz 1)
    // the per-token grid is otherwise a single block walking every head serially between
    // block-wide barriers.
    // The head count is rarely a multiple of blockDim.y, so the last round leaves some thread
    // rows without a head. The loop runs the same number of rounds for the whole block (row 0's
    // count) and those rows idle through the round's barriers, so every barrier is reached by
    // every thread
    const int num_heads = num_heads_q + num_heads_k;
    const int head_base = blockIdx.z * blockDim.y;
    const int head_step = gridDim.z * blockDim.y;
    const int head_rounds = head_base < num_heads ? (num_heads - head_base + head_step - 1) / head_step : 0;
    for (int round = 0; round < head_rounds; ++round)
    {
        const int head_idx = head_base + round * head_step + threadIdx.y;
        const bool active = head_idx < num_heads;

        const half* g_head_in_ptr = nullptr;
        half* g_head_out_ptr = nullptr;
        const void* norm_weight = nullptr;
        if (head_idx < num_heads_q)
        {
            g_head_in_ptr = q + ((batch * seq_len + token_pos) * num_heads_q + head_idx) * q_head_stride;
            g_head_out_ptr = out_q + ((batch * seq_len + token_pos) * num_heads_q + head_idx) * q_head_stride;
            norm_weight = q_norm;
        }
        else if (active)
        {
            g_head_in_ptr = k + ((batch * seq_len + token_pos) * num_heads_k + head_idx - num_heads_q) * k_head_stride;
            g_head_out_ptr = out_k + ((batch * seq_len + token_pos) * num_heads_k + head_idx - num_heads_q) * k_head_stride;
            norm_weight = k_norm;
        }

        // Temp storage
        half* sh_head = norm_head + t_head * head_dim_pad;
        auto load_head = [&] ()
        {
            if (active)
            {
                if (t < head_dim / 2)
                    ((half2*) sh_head)[t] = ((half2*)g_head_in_ptr)[t];
                else
                    ((half2*) sh_head)[t] = half2{};
            }
            __syncthreads();
        };
        auto store_head = [&] ()
        {
            if (active && t < head_dim / 2)
                ((half2*) g_head_out_ptr)[t] = ((half2*) sh_head)[t];
            __syncthreads();
        };

        // Apply embeddings
        auto apply_rope = [&] ()
        {
            for (int rdim = 0; rdim < rotate_dims; ++rdim)
            {
                int offset = rotate_offset + partial_head_dim * rdim;
                if (active && t < partial_head_dim / 2)
                {
                    float sin = sin_cache[rdim];
                    float cos = cos_cache[rdim];

                    if constexpr (rope_mode == ROPESTYLE_NEOX)
                    {
                        float v1 = __half2float(sh_head[offset + t]);
                        float v2 = __half2float(sh_head[offset + t + partial_head_dim / 2]);
                        float r1 = v1 * cos - v2 * sin;
                        float r2 = v2 * cos + v1 * sin;
                        sh_head[offset + t] = __float2half_rn(r1);
                        sh_head[offset + t + partial_head_dim / 2] = __float2half_rn(r2);
                    }
                    else if constexpr (rope_mode == ROPESTYLE_GPTJ)
                    {
                        half2 *tptr = (half2*)(sh_head + offset + t * 2);
                        half2 v = *tptr;
                        float v1 = __low2float(v);
                        float v2 = __high2float(v);
                        float r1 = v1 * cos - v2 * sin;
                        float r2 = v2 * cos + v1 * sin;
                        v = __floats2half2_rn(r1, r2);
                        *tptr = v;
                    }
                }
            }
            __syncthreads();
        };

        // RMS Norm
        auto apply_norm = [&] ()
        {
            half2 *tptr = (half2*)(sh_head + t * 2);
            int lane_id = threadIdx.x % 32;
            int warp_id = threadIdx.x / 32;
            int warps = blockDim.x / 32;

            // Sum of squares. A thread row is a whole number of warps, so the warp reduction
            // never mixes active and idle lanes
            half2 v = {};
            float v1 = 0.0f;
            float v2 = 0.0f;
            float sum = 0.0f;
            if (active)
            {
                v = *tptr;
                v1 = __low2float(v);
                v2 = __high2float(v);
                sum = v1 * v1 + v2 * v2;
                sum = warp_reduce_sum_f(sum);
                if (lane_id == 0) sums[warps * t_head + warp_id] = sum;
            }
            __syncthreads();

            if (active)
            {
                sum = sums[warps * t_head];
                for (int i = 1; i < warps; ++i) sum += sums[warps * t_head + i];
            }

            // Normalize and downcast
            float rmf = rsqrtf(sum / (float) head_dim + norm_eps);
            v1 *= rmf;
            v2 *= rmf;

            // Downcast, apply weight and store. Lanes past head_dim / 2 (block rounded up to whole warps)
            // hold padding and must not read the weight
            if (active && t * 2 < head_dim)
            {
                if constexpr (norm_bf16)
                {
                    bfloat162 *wptr = (bfloat162*)(((bfloat16*)norm_weight) + t * 2);
                    float2 w = __bfloat1622float2(*wptr);
                    w.x += norm_constant_bias;
                    w.y += norm_constant_bias;
                    v1 *= w.x;
                    v2 *= w.y;
                    v = __floats2half2_rn(v1, v2);
                }
                else
                {
                    half2 norm_constant_bias_h2 = __float2half2_rn(norm_constant_bias);
                    half2 *wptr = (half2*)(((half*)norm_weight) + t * 2);
                    half2 w = __hadd2(*wptr, norm_constant_bias_h2);
                    v = __floats2half2_rn(v1, v2);
                    v = __hmul2(w, v);
                }

                *tptr = v;
            }
            __syncthreads();
        };

        // Llama 4 scaling. Must be called by the whole block (blockDim.y walks q and k heads
        // concurrently and the lambda syncs), so k heads scale by 1.0
        auto apply_l4_scaling = [&] (float scaling)
        {
            if (active && t < head_dim / 2)
            {
                half2* tptr = ((half2*) sh_head) + t;
                half2 v = *tptr;
                *tptr = __floats2half2_rn(__low2float(v) * scaling, __high2float(v) * scaling);
            }
            __syncthreads();
        };

        // Do the things
        load_head();
        if (q_norm) apply_norm();
        apply_rope();
        if (l4_scaling != 1.0f) apply_l4_scaling(head_idx < num_heads_q ? l4_scaling : 1.0f);
        store_head();
    }
}
