#include <metal_stdlib>
using namespace metal;

// Per-head RMS normalization
// Dispatch: (n_heads, 1, 1)
kernel void per_head_rmsnorm(device float       *x      [[buffer(0)]],
                             device const float *weight [[buffer(1)]],
                             constant uint      *p      [[buffer(2)]],
                             uint3 wid [[threadgroup_position_in_grid]],
                             uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float simd_sums[8];
    uint head = wid.x;
    uint tid = lid.x;
    uint hs = p[0];
    float eps = as_type<float>(p[1]);
    uint per_head = p[2];
    uint x_base = head * hs;
    uint w_base = (per_head != 0) ? head * hs : 0;
    uint simd_id = tid / 32;
    uint simd_lane = tid % 32;

    float ss = 0.0f;
    for (uint d = tid; d < hs; d += 256) {
        float v = x[x_base + d];
        ss += v * v;
    }

    // Simdgroup reduction: 256 threads → 8 partial sums → 1 total
    float partial = simd_sum(ss);
    if (simd_lane == 0) simd_sums[simd_id] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float v = simd_sums[tid];
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (tid == 0) simd_sums[0] = 1.0f / sqrt(v / float(hs) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = simd_sums[0];

    for (uint d = tid; d < hs; d += 256)
        x[x_base + d] = x[x_base + d] * weight[w_base + d] * scale;
}
