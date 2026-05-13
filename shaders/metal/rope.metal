#include <metal_stdlib>
using namespace metal;

// Rotary Position Encoding: rotate Q or K vector by position-dependent cos/sin
// Dispatch: (n_heads, 1, 1)
kernel void rope(device float       *vec   [[buffer(0)]],
                 device const float *freq  [[buffer(1)]],
                 constant uint      *p     [[buffer(2)]],
                 uint3 wid [[threadgroup_position_in_grid]],
                 uint3 lid [[thread_position_in_threadgroup]]) {
    uint h = wid.x;
    uint tid = lid.x;
    uint n_heads = p[0];
    uint head_size = p[1];
    uint pos = p[2];
    uint rope_dims = p[3];
    uint input_offset = p[4];

    if (h >= n_heads) return;

    uint base = input_offset + h * head_size;
    uint half_rope = rope_dims / 2;

    for (uint i = tid; i < half_rope; i += 256) {
        float angle = float(pos) * freq[i];
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        uint idx0 = base + i;
        uint idx1 = idx0 + half_rope;
        float v0 = vec[idx0];
        float v1 = vec[idx1];
        vec[idx0] = v0 * cos_a - v1 * sin_a;
        vec[idx1] = v0 * sin_a + v1 * cos_a;
    }
}
