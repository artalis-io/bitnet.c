#include <metal_stdlib>
using namespace metal;

// Per-head L2 normalization of Q and K vectors
// Dispatch: (num_k_heads, 1, 1)
kernel void ssm_l2norm(device float  *q [[buffer(0)]],
                       device float  *k [[buffer(1)]],
                       constant uint *p [[buffer(2)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float simd_q[8];
    threadgroup float simd_k[8];
    uint head = wid.x;
    uint tid = lid.x;
    uint hd = p[0];
    uint q_base = p[1] + head * hd;
    uint k_base = p[2] + head * hd;
    uint simd_id = tid / 32;
    uint simd_lane = tid % 32;

    float qn = 0.0f, kn = 0.0f;
    for (uint d = tid; d < hd; d += 256) {
        float qv = q[q_base + d], kv = k[k_base + d];
        qn += qv * qv;
        kn += kv * kv;
    }

    // Simdgroup reduction for both Q and K norms in parallel
    float partial_q = simd_sum(qn);
    float partial_k = simd_sum(kn);
    if (simd_lane == 0) {
        simd_q[simd_id] = partial_q;
        simd_k[simd_id] = partial_k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float vq = simd_q[tid];
        float vk = simd_k[tid];
        vq += simd_shuffle_xor(vq, 4);
        vk += simd_shuffle_xor(vk, 4);
        vq += simd_shuffle_xor(vq, 2);
        vk += simd_shuffle_xor(vk, 2);
        vq += simd_shuffle_xor(vq, 1);
        vk += simd_shuffle_xor(vk, 1);
        if (tid == 0) {
            simd_q[0] = vq;
            simd_k[0] = vk;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_qn = 1.0f / (sqrt(simd_q[0]) + 1e-6f);
    float inv_kn = 1.0f / (sqrt(simd_k[0]) + 1e-6f);

    for (uint d = tid; d < hd; d += 256) {
        q[q_base + d] *= inv_qn;
        k[k_base + d] *= inv_kn;
    }
}
