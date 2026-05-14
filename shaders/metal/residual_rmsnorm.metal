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
    threadgroup float inv_rms;
    uint tid = lid.x;
    uint dim = p[0];
    float eps = as_type<float>(p[1]);

    for (uint i = tid; i < dim; i += 256) {
        float xr = x[i] + r[i];
        x[i] = xr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
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
        inv_rms = 1.0f / sqrt(ss / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = inv_rms;
    for (uint i = tid; i < dim; i += 256)
        out[i] = x[i] * weight[i] * scale;
}
