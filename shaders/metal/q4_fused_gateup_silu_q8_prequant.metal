#include <metal_stdlib>
using namespace metal;

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

kernel void q4_fused_gateup_silu_q8_prequant(
    device const char  *weights  [[buffer(0)]],
    device const char  *x_q      [[buffer(1)]],
    device const float *x_scales [[buffer(2)]],
    device float       *out      [[buffer(3)]],
    constant uint      *p        [[buffer(4)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint cols = p[1], gate_rows = p[2];
    uint tile_start = wid.x * 32;
    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint nb = cols >> 5;
    float gate_acc = 0.0f, up_acc = 0.0f;

    if (global_row < gate_rows) {
        device const char *gate_data = weights + (size_t)global_row * nb * 18;
        device const char *up_data = weights + (size_t)(global_row + gate_rows) * nb * 18;

        for (uint b = row_lane; b < nb; b += 8) {
            device const char *gblk = gate_data + (size_t)b * 18;
            device const char *ublk = up_data + (size_t)b * 18;
            float gd = float(*(device const half *)gblk);
            float ud = float(*(device const half *)ublk);
            float dx = x_scales[b];
            if (dx == 0.0f)
                continue;
            device const char4 *xqb = (device const char4 *)(x_q + b * 32);
            gate_acc += gd * dx *
                q4_q8_dot((device const uchar *)(gblk + 2), xqb);
            up_acc += ud * dx *
                q4_q8_dot((device const uchar *)(ublk + 2), xqb);
        }
    }

    gate_acc += simd_shuffle_xor(gate_acc, 1);
    gate_acc += simd_shuffle_xor(gate_acc, 2);
    gate_acc += simd_shuffle_xor(gate_acc, 4);
    up_acc += simd_shuffle_xor(up_acc, 1);
    up_acc += simd_shuffle_xor(up_acc, 2);
    up_acc += simd_shuffle_xor(up_acc, 4);

    if (row_lane == 0 && global_row < gate_rows) {
        float g = gate_acc;
        float u = up_acc;
        uint bias_offset = p[4];
        if (bias_offset > 0) {
            g += as_type<float>(((device const uint *)weights)[bias_offset + global_row]);
            u += as_type<float>(((device const uint *)weights)[bias_offset + global_row + gate_rows]);
        }
        out[global_row] = (g / (1.0f + fast::exp(-g))) * u;
    }
}

#undef DQ4_LO
#undef DQ4_HI
