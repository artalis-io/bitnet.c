#include <metal_stdlib>
using namespace metal;

// Q4_0 NATIVE fused gate-up SiLU — reads GGUF format (18B/block)
// out[i] = silu(gate[i]·x) * up[i]·x
// Gate rows 0..gate_rows-1, up rows gate_rows..total_rows-1 in stacked buffer.
// Pre-scaling trick. 32 rows/tile, 8 threads/row.
// Dispatch: (ceil(gate_rows/32), 1, 1)

static inline float q4_native_block_sum(device const ushort *qs,
                                        float4 x0, float4 x1, float4 x2, float4 x3,
                                        float4 x4, float4 x5, float4 x6, float4 x7) {
    float sum = 0.0f;

    sum += x0.x          * float(qs[0] & 0x000Fu);
    sum += (x0.y * 0.00390625f)  * float(qs[0] & 0x0F00u);
    sum += (x4.x * 0.0625f)   * float(qs[0] & 0x00F0u);
    sum += (x4.y * 0.000244140625f) * float(qs[0] & 0xF000u);

    sum += x0.z          * float(qs[1] & 0x000Fu);
    sum += (x0.w * 0.00390625f)  * float(qs[1] & 0x0F00u);
    sum += (x4.z * 0.0625f)   * float(qs[1] & 0x00F0u);
    sum += (x4.w * 0.000244140625f) * float(qs[1] & 0xF000u);

    sum += x1.x          * float(qs[2] & 0x000Fu);
    sum += (x1.y * 0.00390625f)  * float(qs[2] & 0x0F00u);
    sum += (x5.x * 0.0625f)   * float(qs[2] & 0x00F0u);
    sum += (x5.y * 0.000244140625f) * float(qs[2] & 0xF000u);

    sum += x1.z          * float(qs[3] & 0x000Fu);
    sum += (x1.w * 0.00390625f)  * float(qs[3] & 0x0F00u);
    sum += (x5.z * 0.0625f)   * float(qs[3] & 0x00F0u);
    sum += (x5.w * 0.000244140625f) * float(qs[3] & 0xF000u);

    sum += x2.x          * float(qs[4] & 0x000Fu);
    sum += (x2.y * 0.00390625f)  * float(qs[4] & 0x0F00u);
    sum += (x6.x * 0.0625f)   * float(qs[4] & 0x00F0u);
    sum += (x6.y * 0.000244140625f) * float(qs[4] & 0xF000u);

    sum += x2.z          * float(qs[5] & 0x000Fu);
    sum += (x2.w * 0.00390625f)  * float(qs[5] & 0x0F00u);
    sum += (x6.z * 0.0625f)   * float(qs[5] & 0x00F0u);
    sum += (x6.w * 0.000244140625f) * float(qs[5] & 0xF000u);

    sum += x3.x          * float(qs[6] & 0x000Fu);
    sum += (x3.y * 0.00390625f)  * float(qs[6] & 0x0F00u);
    sum += (x7.x * 0.0625f)   * float(qs[6] & 0x00F0u);
    sum += (x7.y * 0.000244140625f) * float(qs[6] & 0xF000u);

    sum += x3.z          * float(qs[7] & 0x000Fu);
    sum += (x3.w * 0.00390625f)  * float(qs[7] & 0x0F00u);
    sum += (x7.z * 0.0625f)   * float(qs[7] & 0x00F0u);
    sum += (x7.w * 0.000244140625f) * float(qs[7] & 0xF000u);

    return sum;
}

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

            device const ushort *gqs = (device const ushort *)(gblk + 2);
            device const ushort *uqs = (device const ushort *)(ublk + 2);
            uint eb = b * 32;

            device const float4 *xp = (device const float4 *)(x + eb);
            float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            float4 x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];

            float sumy = (x0.x+x0.y+x0.z+x0.w) + (x1.x+x1.y+x1.z+x1.w)
                       + (x2.x+x2.y+x2.z+x2.w) + (x3.x+x3.y+x3.z+x3.w)
                       + (x4.x+x4.y+x4.z+x4.w) + (x5.x+x5.y+x5.z+x5.w)
                       + (x6.x+x6.y+x6.z+x6.w) + (x7.x+x7.y+x7.z+x7.w);

            gate_acc += gd * (sumy * (-8.0f) +
                              q4_native_block_sum(gqs, x0, x1, x2, x3, x4, x5, x6, x7));
            up_acc += ud * (sumy * (-8.0f) +
                            q4_native_block_sum(uqs, x0, x1, x2, x3, x4, x5, x6, x7));
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
