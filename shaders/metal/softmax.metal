#include <metal_stdlib>
using namespace metal;

// Per-head softmax over attention scores
// Dispatch: (n_heads, 1, 1)
kernel void softmax(device float  *att [[buffer(0)]],
                    constant uint *p   [[buffer(1)]],
                    uint3 wid [[threadgroup_position_in_grid]],
                    uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float simd_sums[8];
    uint h = wid.x;
    uint tid = lid.x;
    uint n_heads = p[0], n_kv = p[1], seq_len = p[2];
    uint simd_id = tid / 32;
    uint simd_lane = tid % 32;

    if (h >= n_heads) return;
    uint base = h * seq_len;

    // Phase 1: find max
    float local_max = -3.402823e+38f;
    for (uint i = tid; i < n_kv; i += 256)
        local_max = max(local_max, att[base + i]);

    float partial_max = simd_max(local_max);
    if (simd_lane == 0) simd_sums[simd_id] = partial_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float v = simd_sums[tid];
        v = max(v, simd_shuffle_xor(v, 4));
        v = max(v, simd_shuffle_xor(v, 2));
        v = max(v, simd_shuffle_xor(v, 1));
        if (tid == 0) simd_sums[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = simd_sums[0];

    // Phase 2: exp and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < n_kv; i += 256) {
        float e = exp(att[base + i] - max_val);
        att[base + i] = e;
        local_sum += e;
    }

    float partial_sum = simd_sum(local_sum);
    if (simd_lane == 0) simd_sums[simd_id] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float v = simd_sums[tid];
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (tid == 0) simd_sums[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / simd_sums[0];

    // Phase 3: normalize
    for (uint i = tid; i < n_kv; i += 256)
        att[base + i] *= inv_sum;
}
