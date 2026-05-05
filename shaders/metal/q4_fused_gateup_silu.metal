#include <metal_stdlib>
using namespace metal;

// Q4_0 NATIVE fused gate-up SiLU — reads GGUF format (18B/block)
// out[i] = silu(gate[i]·x) * up[i]·x
// Gate rows 0..gate_rows-1, up rows gate_rows..total_rows-1 in stacked buffer.
// Pre-scaling trick. 32 rows/tile, 8 threads/row.
// Dispatch: (ceil(gate_rows/32), 1, 1)

kernel void q4_fused_gateup_silu(device const char  *weights [[buffer(0)]],
                                  device const float *x       [[buffer(1)]],
                                  device float       *out     [[buffer(2)]],
                                  constant uint      *p       [[buffer(3)]],
                                  uint3 wid [[threadgroup_position_in_grid]],
                                  uint3 lid [[thread_position_in_threadgroup]]) {
    uint total_rows = p[0], cols = p[1], gate_rows = p[2];
    uint tile_start = wid.x * 32;
    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint nb = cols >> 5;
    float gate_acc = 0.0f, up_acc = 0.0f;

    if (global_row < gate_rows) {
        device const char *gate_data = weights + (size_t)global_row * nb * 18;
        device const char *up_data   = weights + (size_t)(global_row + gate_rows) * nb * 18;

        for (uint b = row_lane; b < nb; b += 8) {
            device const char *gblk = gate_data + (size_t)b * 18;
            device const char *ublk = up_data   + (size_t)b * 18;
            float gd = float(*(device const half *)gblk);
            float ud = float(*(device const half *)ublk);

            device const uchar *gqs = (device const uchar *)(gblk + 2);
            device const uchar *uqs = (device const uchar *)(ublk + 2);
            uint eb = b * 32;

            device const float4 *xp = (device const float4 *)(x + eb);
            float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            float4 x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];

            float4 g_lo0 = float4(float(gqs[0] & 0x0Fu), float(gqs[1] & 0x0Fu),
                                  float(gqs[2] & 0x0Fu), float(gqs[3] & 0x0Fu)) - 8.0f;
            float4 g_hi0 = float4(float(gqs[0] >> 4), float(gqs[1] >> 4),
                                  float(gqs[2] >> 4), float(gqs[3] >> 4)) - 8.0f;
            float4 g_lo1 = float4(float(gqs[4] & 0x0Fu), float(gqs[5] & 0x0Fu),
                                  float(gqs[6] & 0x0Fu), float(gqs[7] & 0x0Fu)) - 8.0f;
            float4 g_hi1 = float4(float(gqs[4] >> 4), float(gqs[5] >> 4),
                                  float(gqs[6] >> 4), float(gqs[7] >> 4)) - 8.0f;
            float4 g_lo2 = float4(float(gqs[8] & 0x0Fu), float(gqs[9] & 0x0Fu),
                                  float(gqs[10] & 0x0Fu), float(gqs[11] & 0x0Fu)) - 8.0f;
            float4 g_hi2 = float4(float(gqs[8] >> 4), float(gqs[9] >> 4),
                                  float(gqs[10] >> 4), float(gqs[11] >> 4)) - 8.0f;
            float4 g_lo3 = float4(float(gqs[12] & 0x0Fu), float(gqs[13] & 0x0Fu),
                                  float(gqs[14] & 0x0Fu), float(gqs[15] & 0x0Fu)) - 8.0f;
            float4 g_hi3 = float4(float(gqs[12] >> 4), float(gqs[13] >> 4),
                                  float(gqs[14] >> 4), float(gqs[15] >> 4)) - 8.0f;

            float4 u_lo0 = float4(float(uqs[0] & 0x0Fu), float(uqs[1] & 0x0Fu),
                                  float(uqs[2] & 0x0Fu), float(uqs[3] & 0x0Fu)) - 8.0f;
            float4 u_hi0 = float4(float(uqs[0] >> 4), float(uqs[1] >> 4),
                                  float(uqs[2] >> 4), float(uqs[3] >> 4)) - 8.0f;
            float4 u_lo1 = float4(float(uqs[4] & 0x0Fu), float(uqs[5] & 0x0Fu),
                                  float(uqs[6] & 0x0Fu), float(uqs[7] & 0x0Fu)) - 8.0f;
            float4 u_hi1 = float4(float(uqs[4] >> 4), float(uqs[5] >> 4),
                                  float(uqs[6] >> 4), float(uqs[7] >> 4)) - 8.0f;
            float4 u_lo2 = float4(float(uqs[8] & 0x0Fu), float(uqs[9] & 0x0Fu),
                                  float(uqs[10] & 0x0Fu), float(uqs[11] & 0x0Fu)) - 8.0f;
            float4 u_hi2 = float4(float(uqs[8] >> 4), float(uqs[9] >> 4),
                                  float(uqs[10] >> 4), float(uqs[11] >> 4)) - 8.0f;
            float4 u_lo3 = float4(float(uqs[12] & 0x0Fu), float(uqs[13] & 0x0Fu),
                                  float(uqs[14] & 0x0Fu), float(uqs[15] & 0x0Fu)) - 8.0f;
            float4 u_hi3 = float4(float(uqs[12] >> 4), float(uqs[13] >> 4),
                                  float(uqs[14] >> 4), float(uqs[15] >> 4)) - 8.0f;

            gate_acc += gd * (dot(x0, g_lo0) + dot(x4, g_hi0) +
                              dot(x1, g_lo1) + dot(x5, g_hi1) +
                              dot(x2, g_lo2) + dot(x6, g_hi2) +
                              dot(x3, g_lo3) + dot(x7, g_hi3));
            up_acc += ud * (dot(x0, u_lo0) + dot(x4, u_hi0) +
                            dot(x1, u_lo1) + dot(x5, u_hi1) +
                            dot(x2, u_lo2) + dot(x6, u_hi2) +
                            dot(x3, u_lo3) + dot(x7, u_hi3));
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
        out[global_row] = (g / (1.0f + exp(-g))) * u;
    }
}
