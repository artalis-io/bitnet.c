#include <metal_stdlib>
using namespace metal;

// Per-head RMSNorm + SiLU gate
// Dispatch: (num_v_heads, 1, 1)
kernel void ssm_gate(device float       *out    [[buffer(0)]],
                     device const float *z      [[buffer(1)]],
                     device const float *norm_w [[buffer(2)]],
                     constant uint      *p      [[buffer(3)]],
                     uint3 wid [[threadgroup_position_in_grid]],
                     uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float simd_sums[8];
    uint hv_idx = wid.x;
    uint tid = lid.x;
    uint hv = p[0];
    float eps = as_type<float>(p[1]);
    uint base = hv_idx * hv;
    uint simd_id = tid / 32;
    uint simd_lane = tid % 32;

    float ss = 0.0f;
    for (uint d = tid; d < hv; d += 256) {
        float val = out[base + d];
        ss += val * val;
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
        if (tid == 0) simd_sums[0] = 1.0f / sqrt(v / float(hv) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = simd_sums[0];

    for (uint d = tid; d < hv; d += 256) {
        float normed = out[base + d] * inv_rms * norm_w[d];
        float g = z[base + d];
        out[base + d] = normed * (g / (1.0f + exp(-g)));
    }
}
