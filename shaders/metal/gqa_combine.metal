#include <metal_stdlib>
using namespace metal;

// GQA weighted sum: xb[h] = sum_i(att[h][i] * V[i])
// Dispatch: (n_heads, 1, 1)
kernel void gqa_combine(device const float *att         [[buffer(0)]],
                        device const float *value_cache [[buffer(1)]],
                        device float       *xb          [[buffer(2)]],
                        constant uint      *p           [[buffer(3)]],
                        uint3 wid [[threadgroup_position_in_grid]],
                        uint3 lid [[thread_position_in_threadgroup]]) {
    uint h = wid.x;
    uint tid = lid.x;
    uint n_heads = p[0], head_size = p[1], n_kv = p[2], kv_mul = p[3];
    uint kv_dim = p[4], seq_len = p[5], loff = p[6];

    if (h >= n_heads) return;

    uint kv_h = h / kv_mul;
    uint out_base = h * head_size;

    for (uint d = tid; d < head_size; d += 256) {
        float acc = 0.0f;
        for (uint i = 0; i < n_kv; i++) {
            uint v_base = loff + i * kv_dim + kv_h * head_size;
            acc += att[h * seq_len + i] * value_cache[v_base + d];
        }
        xb[out_base + d] = acc;
    }
}
