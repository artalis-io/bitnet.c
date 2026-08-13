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
    uint input_offset = p[3];
    uint cpu_order = p[4];
    uint x_base = input_offset + head * hs;
    uint w_base = (per_head != 0) ? head * hs : 0;
    uint simd_id = tid / 32;
    uint simd_lane = tid % 32;

    if (cpu_order != 0) {
        if (tid == 0) {
            float4 ss0 = float4(0.0f), ss1 = float4(0.0f);
            for (uint d = 0; d < hs; d += 8) {
                float4 v0 = float4(x[x_base+d+0], x[x_base+d+1], x[x_base+d+2], x[x_base+d+3]);
                float4 v1 = float4(x[x_base+d+4], x[x_base+d+5], x[x_base+d+6], x[x_base+d+7]);
                ss0 = fma(v0, v0, ss0); ss1 = fma(v1, v1, ss1);
            }
            float4 sum = ss0 + ss1;
            float ss = (sum.x + sum.y) + (sum.z + sum.w);
            simd_sums[0] = 1.0f / sqrt(ss / float(hs) + eps);
        }
    } else {
        float ss = 0.0f;
        uint vec_count = hs / 4u;
        for (uint vi = tid; vi < vec_count; vi += 256u) {
            float4 v = *(device const float4 *)(x + x_base + vi * 4u);
            ss += dot(v, v);
        }
        for (uint d = vec_count * 4u + tid; d < hs; d += 256u) {
            float v = x[x_base + d];
            ss += v * v;
        }
        float partial = simd_sum(ss);
        if (simd_lane == 0) simd_sums[simd_id] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8) {
            float v = simd_sums[tid];
            v += simd_shuffle_xor(v, 4);
            v += simd_shuffle_xor(v, 2);
            v += simd_shuffle_xor(v, 1);
            if (tid == 0)
                simd_sums[0] = 1.0f / sqrt(v / float(hs) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = simd_sums[0];

    for (uint d = tid; d < hs; d += 256) {
        x[x_base + d] = (x[x_base + d] * scale) * weight[w_base + d];
    }
}
