#include <metal_stdlib>
using namespace metal;

// Per-head L2 normalization of Q and K vectors
// Dispatch: (num_k_heads, 1, 1)
kernel void ssm_l2norm(device float  *q [[buffer(0)]],
                       device float  *k [[buffer(1)]],
                       constant uint *p [[buffer(2)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float norms[2];
    uint head = wid.x;
    uint tid = lid.x;
    uint hd = p[0];
    uint q_base = p[1] + head * hd;
    uint k_base = p[2] + head * hd;
    if (tid == 0) {
        float4 qss0 = float4(0.0f), qss1 = float4(0.0f);
        float4 qss2 = float4(0.0f), qss3 = float4(0.0f);
        float4 kss0 = float4(0.0f), kss1 = float4(0.0f);
        float4 kss2 = float4(0.0f), kss3 = float4(0.0f);
        for (uint d = 0; d < hd; d += 16) {
            float4 q0 = float4(q[q_base+d+0], q[q_base+d+1], q[q_base+d+2], q[q_base+d+3]);
            float4 q1 = float4(q[q_base+d+4], q[q_base+d+5], q[q_base+d+6], q[q_base+d+7]);
            float4 q2 = float4(q[q_base+d+8], q[q_base+d+9], q[q_base+d+10], q[q_base+d+11]);
            float4 q3 = float4(q[q_base+d+12], q[q_base+d+13], q[q_base+d+14], q[q_base+d+15]);
            float4 k0 = float4(k[k_base+d+0], k[k_base+d+1], k[k_base+d+2], k[k_base+d+3]);
            float4 k1 = float4(k[k_base+d+4], k[k_base+d+5], k[k_base+d+6], k[k_base+d+7]);
            float4 k2 = float4(k[k_base+d+8], k[k_base+d+9], k[k_base+d+10], k[k_base+d+11]);
            float4 k3 = float4(k[k_base+d+12], k[k_base+d+13], k[k_base+d+14], k[k_base+d+15]);
            qss0 = fma(q0, q0, qss0); qss1 = fma(q1, q1, qss1);
            qss2 = fma(q2, q2, qss2); qss3 = fma(q3, q3, qss3);
            kss0 = fma(k0, k0, kss0); kss1 = fma(k1, k1, kss1);
            kss2 = fma(k2, k2, kss2); kss3 = fma(k3, k3, kss3);
        }
        float4 qs = (qss0 + qss1) + (qss2 + qss3);
        float4 ks = (kss0 + kss1) + (kss2 + kss3);
        norms[0] = (qs.x + qs.z) + (qs.y + qs.w);
        norms[1] = (ks.x + ks.z) + (ks.y + ks.w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float eps = as_type<float>(p[3]);
    float inv_qn = 1.0f / max(sqrt(norms[0]), eps);
    float inv_kn = 1.0f / max(sqrt(norms[1]), eps);

    for (uint d = tid; d < hd; d += 256) {
        q[q_base + d] *= inv_qn;
        k[k_base + d] *= inv_kn;
    }
}
