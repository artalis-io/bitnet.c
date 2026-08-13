#include <metal_stdlib>
using namespace metal;

// Q6_K matvec. Each SIMD group computes four rows and reuses every loaded
// activation fragment across those rows.

constant uint TILE_ROWS = 32;
constant uint QK_K = 256;
constant uint BLOCK_BYTES = 210;

kernel void q6k_matvec(device const uchar *weights [[buffer(0)]],
                       device const float *x       [[buffer(1)]],
                       device float       *out     [[buffer(2)]],
                       constant uint      *p       [[buffer(3)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint tile_start = (extra > 0)
        ? (wid.x + wid.y * extra) * TILE_ROWS
        : wid.x * TILE_ROWS;
    uint token = (extra > 0) ? 0 : wid.y;
    uint lane = lid.x & 31u;
    uint simd_group = lid.x >> 5;
    uint first_row = tile_start + simd_group * 4u;

    uint tid = lane >> 1;
    uint block_parity = lane & 1u;
    uint block_half = tid >> 3;
    uint segment = tid & 7u;
    uint l0 = 4u * segment;
    uint scale_base = 8u * block_half + l0 / 16u;
    uint y_offset = 128u * block_half + l0;
    uint ql_offset = 64u * block_half + l0;
    uint qh_offset = 32u * block_half + l0;

    uint n_blocks = cols / QK_K;
    uint x_base = token * cols;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint bi = block_parity; bi < n_blocks; bi += 2u) {
        device const float *yp = x + x_base + bi * QK_K + y_offset;
        float4 y0 = *(device const float4 *)(yp + 0);
        float4 y1 = *(device const float4 *)(yp + 32);
        float4 y2 = *(device const float4 *)(yp + 64);
        float4 y3 = *(device const float4 *)(yp + 96);

        for (uint r = 0; r < 4; ++r) {
            uint row = first_row + r;
            if (row >= rows) continue;
            device const uchar *block = weights +
                ((size_t)row * n_blocks + bi) * BLOCK_BYTES;
            device const uchar *q1 = block + ql_offset;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = block + 128 + qh_offset;
            device const char *sc = (device const char *)(block + 192) +
                                    scale_base;
            float4 sums = float4(0.0f);
            for (uint l = 0; l < 4; ++l) {
                uchar h = qh[l];
                sums[0] += y0[l] *
                    (float((q1[l] & 0x0f) | ((h & 0x03) << 4)) - 32.0f);
                sums[1] += y1[l] *
                    (float((q2[l] & 0x0f) | ((h & 0x0c) << 2)) - 32.0f);
                sums[2] += y2[l] *
                    (float((q1[l] >> 4) | (h & 0x30)) - 32.0f);
                sums[3] += y3[l] *
                    (float((q2[l] >> 4) | ((h & 0xc0) >> 2)) - 32.0f);
            }
            float d = float(*(device const half *)(block + 208));
            acc[r] += d * (sums[0] * float(sc[0]) +
                           sums[1] * float(sc[2]) +
                           sums[2] * float(sc[4]) +
                           sums[3] * float(sc[6]));
        }
    }

    for (uint r = 0; r < 4; ++r) {
        float value = simd_sum(acc[r]);
        uint row = first_row + r;
        if (lane == 0 && row < rows)
            out[out_offset + token * rows + row] = value;
    }
}
