#include <metal_stdlib>
using namespace metal;

// Q4_0 native-format matvec using prequantized Q8 activation blocks.

#define DQ4_LO(qs, i) float4( \
    float(int((qs)[(i)]     & 0x0Fu) - 8), \
    float(int((qs)[(i) + 1] & 0x0Fu) - 8), \
    float(int((qs)[(i) + 2] & 0x0Fu) - 8), \
    float(int((qs)[(i) + 3] & 0x0Fu) - 8))

#define DQ4_HI(qs, i) float4( \
    float(int((qs)[(i)]     >> 4) - 8), \
    float(int((qs)[(i) + 1] >> 4) - 8), \
    float(int((qs)[(i) + 2] >> 4) - 8), \
    float(int((qs)[(i) + 3] >> 4) - 8))

static inline float q4_q8_dot(device const uchar *qs,
                              device const char4 *xq) {
    return dot(DQ4_LO(qs, 0),  float4(xq[0]))
         + dot(DQ4_LO(qs, 4),  float4(xq[1]))
         + dot(DQ4_LO(qs, 8),  float4(xq[2]))
         + dot(DQ4_LO(qs, 12), float4(xq[3]))
         + dot(DQ4_HI(qs, 0),  float4(xq[4]))
         + dot(DQ4_HI(qs, 4),  float4(xq[5]))
         + dot(DQ4_HI(qs, 8),  float4(xq[6]))
         + dot(DQ4_HI(qs, 12), float4(xq[7]));
}

kernel void q4_native_q8_prequant_matvec(
    device const char  *weights [[buffer(0)]],
    device const char  *x_q     [[buffer(1)]],
    device const float *x_scales [[buffer(2)]],
    device float       *out     [[buffer(3)]],
    constant uint      *p       [[buffer(4)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * 32 : wid.x * 32;
    uint token = (extra > 0) ? 0 : wid.y;

    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint nb = cols >> 5;
    uint x_base = token * cols;
    uint scale_base = token * nb;

    float acc = 0.0f;

    if (global_row < rows) {
        device const char *row_data = weights + (size_t)global_row * nb * 18;

        for (uint b = row_lane; b < nb; b += 8) {
            device const char *block = row_data + (size_t)b * 18;
            float d = float(*(device const half *)block);
            float dx = x_scales[scale_base + b];
            if (dx == 0.0f)
                continue;
            device const uchar *qs = (device const uchar *)(block + 2);
            device const char4 *xqb = (device const char4 *)(x_q + x_base + b * 32);

            acc += d * dx * q4_q8_dot(qs, xqb);
        }
    }

    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows) {
        uint bias_offset = p[4];
        if (bias_offset > 0)
            acc += as_type<float>(((device const uint *)weights)[bias_offset + global_row]);
        out[out_offset + token * rows + global_row] = acc;
    }
}

#undef DQ4_LO
#undef DQ4_HI
