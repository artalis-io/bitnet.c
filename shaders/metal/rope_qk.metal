#include <metal_stdlib>
using namespace metal;

// Fused Rotary Position Encoding: apply RoPE to both Q and K in a single dispatch.
// Dispatch: (n_q_heads + n_kv_heads, 1, 1) — one threadgroup per head total.
// Threadgroups 0..n_kv_heads-1 process K heads, n_kv_heads..end process Q heads.
kernel void rope_qk(device float       *q      [[buffer(0)]],
                    device float       *k      [[buffer(1)]],
                    device const float *freq   [[buffer(2)]],
                    constant uint      *p      [[buffer(3)]],
                    uint3 wid [[threadgroup_position_in_grid]],
                    uint3 lid [[thread_position_in_threadgroup]]) {
    uint h = wid.x;
    uint tid = lid.x;
    uint n_q_heads      = p[0];
    uint head_size       = p[1];
    uint pos             = p[2];
    uint rope_dims       = p[3];
    uint n_kv_heads      = p[4];
    uint k_input_offset  = p[5];

    uint total_heads = n_q_heads + n_kv_heads;
    if (h >= total_heads) return;

    // First n_kv_heads threadgroups handle K, remainder handle Q
    uint base;
    if (h < n_kv_heads) {
        base = k_input_offset + h * head_size;
    } else {
        base = (h - n_kv_heads) * head_size;
    }

    // Select target buffer
    device float *vec = (h < n_kv_heads) ? k : q;

    uint half_rope = rope_dims / 2;

    for (uint i = tid; i < half_rope; i += 256) {
        float angle = float(pos) * freq[i];
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        uint idx = base + i * 2;
        float v0 = vec[idx];
        float v1 = vec[idx + 1];
        vec[idx]     = v0 * cos_a - v1 * sin_a;
        vec[idx + 1] = v0 * sin_a + v1 * cos_a;
    }
}
