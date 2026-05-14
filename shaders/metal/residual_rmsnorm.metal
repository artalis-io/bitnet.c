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
    threadgroup float shared[256];
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

    shared[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        shared[0] = 1.0f / sqrt(shared[0] / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared[0];
    for (uint i = tid; i < dim; i += 256)
        out[i] = x[i] * weight[i] * scale;
}
