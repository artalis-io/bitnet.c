#include <metal_stdlib>
using namespace metal;

// Diagnostic RMSNorm: match the CPU NEON four-lane accumulation order.
kernel void rmsnorm_cpu_order(device const float *x      [[buffer(0)]],
                              device const float *weight [[buffer(1)]],
                              device float       *out    [[buffer(2)]],
                              constant uint      *p      [[buffer(3)]],
                              uint3 lid [[thread_position_in_threadgroup]]) {
    if (lid.x != 0) return;

    uint dim = p[0];
    float eps = as_type<float>(p[1]);
    float4 sum_sq0 = float4(0.0f);
    float4 sum_sq1 = float4(0.0f);
    uint i = 0;
    for (; i + 7 < dim; i += 8) {
        float4 v0 = *(device const float4 *)(x + i);
        float4 v1 = *(device const float4 *)(x + i + 4);
        sum_sq0 = fma(v0, v0, sum_sq0);
        sum_sq1 = fma(v1, v1, sum_sq1);
    }

    float4 lane_sums = sum_sq0 + sum_sq1;
    float ss = (lane_sums.x + lane_sums.y) +
               (lane_sums.z + lane_sums.w);
    for (; i < dim; i++)
        ss += x[i] * x[i];
    float scale = 1.0f / sqrt(ss / float(dim) + eps);

    for (uint j = 0; j < dim; j++)
        out[j] = (x[j] * weight[j]) * scale;
}
