#include <metal_stdlib>
using namespace metal;

// GQA attention scores: Q·K dot products
// Dispatch: (n_heads, 1, 1)
kernel void gqa_scores(device const float *q         [[buffer(0)]],
                       device const float *key_cache [[buffer(1)]],
                       device float       *att       [[buffer(2)]],
                       constant uint      *p         [[buffer(3)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    uint h = wid.x;
    uint tid = lid.x;
    uint n_heads = p[0], head_size = p[1], n_kv = p[2], kv_mul = p[3];
    uint kv_dim = p[4], seq_len = p[5], loff = p[6];
    float scale = as_type<float>(p[7]);

    if (h >= n_heads) return;

    uint kv_h = h / kv_mul;
    uint q_base = h * head_size;

    for (uint i = tid; i < n_kv; i += 256) {
        uint k_base = loff + i * kv_dim + kv_h * head_size;
        float dot = 0.0f;
        for (uint d = 0; d < head_size; d++)
            dot += q[q_base + d] * key_cache[k_base + d];
        att[h * seq_len + i] = dot * scale;
    }
}
