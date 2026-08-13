#include <metal_stdlib>
using namespace metal;

// RMS normalization: out[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)
// Dispatch: (1, 1, 1)
kernel void rmsnorm(device const float *x      [[buffer(0)]],
                    device const float *weight  [[buffer(1)]],
                    device float       *out     [[buffer(2)]],
                    constant uint      *p       [[buffer(3)]],
                    uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float shared[32];
    uint tid = lid.x;
    uint dim = p[0];
    float eps = as_type<float>(p[1]);
    uint simd_group = tid >> 5;
    uint lane = tid & 31u;

    float sum_sq = 0.0f;
    uint vec_count = dim / 4u;
    for (uint vi = tid; vi < vec_count; vi += 256u) {
        float4 v = *(device const float4 *)(x + vi * 4u);
        sum_sq += dot(v, v);
    }
    for (uint i = vec_count * 4u + tid; i < dim; i += 256u) {
        float v = x[i];
        sum_sq += v * v;
    }

    if (simd_group == 0)
        shared[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_sq = simd_sum(sum_sq);
    if (lane == 0)
        shared[simd_group] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_sq = simd_sum(shared[lane]);

    float scale = 1.0f / sqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += 256) {
        out[i] = (x[i] * scale) * weight[i];
    }
}
