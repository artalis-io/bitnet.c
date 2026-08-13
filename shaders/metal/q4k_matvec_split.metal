#include <metal_stdlib>
using namespace metal;

// Q4_K two-output matvec using the 32-lane reference reduction order.

constant uint QK_K = 256;
constant uint BLOCK_BYTES = 144;
constant uint SIMD_GROUPS_PER_TG = 8;
constant uint ROWS_PER_SIMD_GROUP = 4;
constant uint ROWS_PER_TG = SIMD_GROUPS_PER_TG * ROWS_PER_SIMD_GROUP;

kernel void q4k_matvec_split(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out0 [[buffer(2)]],
    device float *out1 [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint3 wid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], split = p[2];
    uint out0_offset = p[5], out1_offset = p[6];
    uint first_row = wid.x * ROWS_PER_TG +
                     uint(simd_group) * ROWS_PER_SIMD_GROUP;
    if (first_row >= rows) return;

    constexpr ushort mask1 = 0x3f3f;
    constexpr ushort mask2 = 0x0f0f;
    constexpr ushort mask3 = 0xc0c0;
    ushort ix = lane / 8;
    ushort it = lane % 8;
    ushort iq = it / 4;
    ushort ir = it % 4;
    uint n_blocks = cols / QK_K;
    uint row_stride = n_blocks * BLOCK_BYTES;
    device const float *y4 = x + uint(ix) * QK_K + 64u * uint(iq) +
                             8u * uint(ir);
    float sumf[ROWS_PER_SIMD_GROUP] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint n_rows = min(ROWS_PER_SIMD_GROUP, rows - first_row);

    for (uint ib = ix; ib < n_blocks; ib += 4) {
        float yl[16], yh[16];
        float4 sumy = float4(0.0f);
        for (ushort i = 0; i < 8; i++) {
            yl[i] = y4[i];
            sumy[0] += yl[i];
            yl[i + 8] = y4[i + 32];
            sumy[1] += yl[i + 8];
            yh[i] = y4[i + 128];
            sumy[2] += yh[i];
            yh[i + 8] = y4[i + 160];
            sumy[3] += yh[i + 8];
        }
        device const uchar *first_block =
            weights + first_row * row_stride + ib * BLOCK_BYTES;
        device const ushort *sc =
            (device const ushort *)(first_block + 4) + iq;
        device const ushort *q1 =
            (device const ushort *)(first_block + 16) + 16u * uint(iq) +
            4u * uint(ir);
        device const half *dh = (device const half *)first_block;

        for (uint row = 0; row < n_rows; row++) {
            ushort packed[4];
            packed[0] = sc[0] & mask1;
            packed[1] = sc[2] & mask1;
            packed[2] = (sc[4] & mask2) | ((sc[0] & mask3) >> 2);
            packed[3] = ((sc[4] >> 4) & mask2) |
                        ((sc[2] & mask3) >> 2);
            thread const uchar *scale =
                (thread const uchar *)packed;
            device const ushort *q2 = q1 + 32;
            float4 acc1 = float4(0.0f), acc2 = float4(0.0f);
            for (ushort i = 0; i < 4; i++) {
                acc1[0] += yl[2 * i] * float(q1[i] & 0x000f);
                acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0f00);
                acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00f0);
                acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xf000);
                acc2[0] += yh[2 * i] * float(q2[i] & 0x000f);
                acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0f00);
                acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00f0);
                acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xf000);
            }
            float d = float(dh[0]), dmin = float(dh[1]);
            sumf[row] +=
                d * ((acc1[0] + acc1[1] / 256.0f) * float(scale[0]) +
                     (acc1[2] + acc1[3] / 256.0f) * float(scale[1]) /
                         16.0f +
                     (acc2[0] + acc2[1] / 256.0f) * float(scale[4]) +
                     (acc2[2] + acc2[3] / 256.0f) * float(scale[5]) /
                         16.0f) -
                dmin * (sumy[0] * float(scale[2]) +
                        sumy[1] * float(scale[3]) +
                        sumy[2] * float(scale[6]) +
                        sumy[3] * float(scale[7]));
            q1 = (device const ushort *)((device const uchar *)q1 +
                                          row_stride);
            sc = (device const ushort *)((device const uchar *)sc +
                                          row_stride);
            dh = (device const half *)((device const uchar *)dh +
                                        row_stride);
        }
        y4 += 4 * QK_K;
    }

    for (uint row = 0; row < n_rows; row++) {
        float sum = simd_sum(sumf[row]);
        if (lane != 0) continue;
        uint global_row = first_row + row;
        if (global_row < split)
            out0[out0_offset + global_row] = sum;
        else
            out1[out1_offset + global_row - split] = sum;
    }
}
