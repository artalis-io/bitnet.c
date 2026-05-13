#include <metal_stdlib>
using namespace metal;

// GQA attention scores: Q·K dot products
// Dispatch: (n_heads, ceil(n_kv / 8), 1)
// One SIMD group computes one KV score for the current head.
kernel void gqa_scores(device const float *q         [[buffer(0)]],
                       device const float *key_cache [[buffer(1)]],
                       device float       *att       [[buffer(2)]],
                       constant uint      *p         [[buffer(3)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    uint h = wid.x;
    uint simd_id = lid.x >> 5;
    uint lane = lid.x & 31;
    uint i = wid.y * 8 + simd_id;
    uint n_heads = p[0], head_size = p[1], n_kv = p[2], kv_mul = p[3];
    uint kv_dim = p[4], seq_len = p[5], loff = p[6];
    float scale = as_type<float>(p[7]);

    if (h >= n_heads || i >= n_kv) return;

    uint kv_h = h / kv_mul;
    uint q_base = h * head_size;
    uint k_base = loff + i * kv_dim + kv_h * head_size;

    float dot = 0.0f;
    float comp = 0.0f;
    for (uint d = lane; d < head_size; d += 32) {
        float y = q[q_base + d] * key_cache[k_base + d] - comp;
        float t = dot + y;
        comp = (t - dot) - y;
        dot = t;
    }

    dot = simd_sum(dot);
    if (lane == 0)
        att[h * seq_len + i] = dot * scale;
}
