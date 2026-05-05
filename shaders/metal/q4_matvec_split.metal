#include <metal_stdlib>
using namespace metal;

// Q4_0 NATIVE split matvec — multi-output, reads GGUF format (18B/block)
// Routes output rows to up to 3 buffers based on split points.
// Pre-scaling trick for zero-shift nibble extraction.
// Dispatch: (ceil(total_rows/32), 1, 1)

kernel void q4_matvec_split(device const char  *weights [[buffer(0)]],
                             device const float *x       [[buffer(1)]],
                             device float       *out0    [[buffer(2)]],
                             device float       *out1    [[buffer(3)]],
                             device float       *out2    [[buffer(4)]],
                             constant uint      *p       [[buffer(5)]],
                             uint3 wid [[threadgroup_position_in_grid]],
                             uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1];
    uint split1 = p[2], split2 = p[3];
    uint off0 = p[5], off1 = p[6], off2 = p[7];

    uint tile_start = wid.x * 32;
    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint nb = cols >> 5;
    float acc = 0.0f;

    if (global_row < rows) {
        device const char *row_data = weights + (size_t)global_row * nb * 18;

        for (uint b = row_lane; b < nb; b += 8) {
            device const char *block = row_data + (size_t)b * 18;
            float d = float(*(device const half *)block);
            device const uchar *qs = (device const uchar *)(block + 2);
            uint eb = b * 32;

            device const float4 *xp = (device const float4 *)(x + eb);
            float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            float4 x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];

            float4 lo0 = float4(float(qs[0] & 0x0Fu), float(qs[1] & 0x0Fu),
                                float(qs[2] & 0x0Fu), float(qs[3] & 0x0Fu)) - 8.0f;
            float4 hi0 = float4(float(qs[0] >> 4), float(qs[1] >> 4),
                                float(qs[2] >> 4), float(qs[3] >> 4)) - 8.0f;
            float4 lo1 = float4(float(qs[4] & 0x0Fu), float(qs[5] & 0x0Fu),
                                float(qs[6] & 0x0Fu), float(qs[7] & 0x0Fu)) - 8.0f;
            float4 hi1 = float4(float(qs[4] >> 4), float(qs[5] >> 4),
                                float(qs[6] >> 4), float(qs[7] >> 4)) - 8.0f;
            float4 lo2 = float4(float(qs[8] & 0x0Fu), float(qs[9] & 0x0Fu),
                                float(qs[10] & 0x0Fu), float(qs[11] & 0x0Fu)) - 8.0f;
            float4 hi2 = float4(float(qs[8] >> 4), float(qs[9] >> 4),
                                float(qs[10] >> 4), float(qs[11] >> 4)) - 8.0f;
            float4 lo3 = float4(float(qs[12] & 0x0Fu), float(qs[13] & 0x0Fu),
                                float(qs[14] & 0x0Fu), float(qs[15] & 0x0Fu)) - 8.0f;
            float4 hi3 = float4(float(qs[12] >> 4), float(qs[13] >> 4),
                                float(qs[14] >> 4), float(qs[15] >> 4)) - 8.0f;

            acc += d * (dot(x0, lo0) + dot(x4, hi0) +
                        dot(x1, lo1) + dot(x5, hi1) +
                        dot(x2, lo2) + dot(x6, hi2) +
                        dot(x3, lo3) + dot(x7, hi3));
        }
    }

    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows) {
        uint bias_offset = p[4];
        if (bias_offset > 0)
            acc += as_type<float>(((device const uint *)weights)[bias_offset + global_row]);
        if (split1 > 0 && global_row >= split1) {
            if (split2 > 0 && global_row >= split2)
                out2[off2 + global_row - split2] = acc;
            else
                out1[off1 + global_row - split1] = acc;
        } else {
            out0[off0 + global_row] = acc;
        }
    }
}
