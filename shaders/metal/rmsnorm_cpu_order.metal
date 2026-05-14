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
    float4 sum_sq = float4(0.0f);
    uint i = 0;
    for (; i + 3 < dim; i += 4) {
        float4 v = *(device const float4 *)(x + i);
        sum_sq = fma(v, v, sum_sq);
    }

    float low = sum_sq.x + sum_sq.z;
    float high = sum_sq.y + sum_sq.w;
    float ss = low + high;
    for (; i < dim; i++)
        ss += x[i] * x[i];
    float scale = 1.0f / sqrt(ss / float(dim) + eps);

    for (uint j = 0; j < dim; j++)
        out[j] = x[j] * scale * weight[j];
}
