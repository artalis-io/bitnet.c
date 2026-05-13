#include <metal_stdlib>
using namespace metal;

// Fused residual addition + RMS normalization
// Dispatch: (1, 1, 1)
kernel void residual_rmsnorm(device float       *x      [[buffer(0)]],
                             device const float *r      [[buffer(1)]],
                             device const float *weight [[buffer(2)]],
                             device float       *out    [[buffer(3)]],
                             constant uint      *p      [[buffer(4)]],
                             uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float simd_sums[8];
    uint tid = lid.x;
    uint dim = p[0];
    float eps = as_type<float>(p[1]);

    float sum_sq = 0.0f;
    float comp = 0.0f;
    for (uint i = tid; i < dim; i += 256) {
        float xr = x[i] + r[i];
        x[i] = xr;
        float y = xr * xr - comp;
        float t = sum_sq + y;
        comp = (t - sum_sq) - y;
        sum_sq = t;
    }

    // Simdgroup reduction: 256 threads → 8 partial sums → 1 total
    float partial = simd_sum(sum_sq);
    uint simd_id = tid / 32;
    uint simd_lane = tid % 32;
    if (simd_lane == 0) simd_sums[simd_id] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float v = simd_sums[tid];
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (tid == 0) simd_sums[0] = 1.0f / sqrt(v / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = simd_sums[0];
    for (uint i = tid; i < dim; i += 256)
        out[i] = x[i] * weight[i] * scale;
}
