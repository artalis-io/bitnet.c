#include <metal_stdlib>
using namespace metal;

// RMS normalization: out[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)
// Dispatch: (1, 1, 1)
kernel void rmsnorm(device const float *x      [[buffer(0)]],
                    device const float *weight  [[buffer(1)]],
                    device float       *out     [[buffer(2)]],
                    constant uint      *p       [[buffer(3)]],
                    uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float shared[256];
    uint tid = lid.x;
    uint dim = p[0];
    float eps = as_type<float>(p[1]);

    float sum_sq = 0.0f;
    float comp = 0.0f;
    for (uint i = tid; i < dim; i += 256) {
        float v = x[i];
        float y = v * v - comp;
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
    for (uint i = tid; i < dim; i += 256) {
        float v = x[i];
        out[i] = v * weight[i] * scale;
    }
}
