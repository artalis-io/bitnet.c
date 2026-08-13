#include <metal_stdlib>
using namespace metal;

// Reference Q6_K x Q8_K matvec. One SIMD group owns a row. Each 256-value
// superblock is accumulated completely as int32 before its single float
// conversion, matching the prepared CPU quant contract.
constant uint ROWS_PER_GROUP = 32;
constant uint QK_K = 256;
constant uint BLOCK_BYTES = 210;

static inline int4 q6_load4(device const uchar *qlp,
                            device const uchar *qhp,
                            uint qh_shift,
                            uint ql_high,
                            uint off) {
    uchar4 ql = uchar4(*(device const packed_uchar4 *)(qlp + off));
    uchar4 qh = uchar4(*(device const packed_uchar4 *)(qhp + off));
    uchar4 q = ql_high
        ? (ql >> uchar4(4u)) |
          (((qh >> uchar4(qh_shift)) & uchar4(3u)) << uchar4(4u))
        : (ql & uchar4(0xFu)) |
          (((qh >> uchar4(qh_shift)) & uchar4(3u)) << uchar4(4u));
    return int4(q) - 32;
}

kernel void q6k_q8k_matvec(device const uchar *weights [[buffer(0)]],
                           device const char  *xq      [[buffer(1)]],
                           device const float *xd      [[buffer(2)]],
                           device const short *bsums   [[buffer(3)]],
                           device float       *out     [[buffer(4)]],
                           constant uint      *p       [[buffer(5)]],
                           uint3 wid [[threadgroup_position_in_grid]],
                           uint3 lid3 [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint simd_group = lid3.x >> 5;
    uint lane = lid3.x & 31u;
    uint tile = extra > 0 ? wid.x + wid.y * extra : wid.x;
    uint first_row = tile * ROWS_PER_GROUP + simd_group * 4u;
    uint token = extra > 0 ? 0 : wid.y;
    uint n_blocks = cols / QK_K;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint bi = 0; bi < n_blocks; bi++) {
        for (uint r = 0; r < 4; r++) {
            uint row = first_row + r;
            int partial = 0;
            if (row < rows) {
                uint elem = lane * 8u;
                uint block_half = elem >> 7;
                uint quarter = (elem & 127u) >> 5;
                uint within = elem & 31u;
                uint qh_shift = quarter * 2u;
                uint ql_high = quarter >> 1;
                uint ql_add = (quarter & 1u) * 32u;
                uint scale = elem >> 4;
                device const uchar *block = weights +
                    ((size_t)row * n_blocks + bi) * BLOCK_BYTES;
                device const uchar *qlp =
                    block + block_half * 64u + ql_add;
                device const uchar *qhp =
                    block + 128u + block_half * 32u;
                device const char *sc =
                    (device const char *)(block + 192u);
                uint xoff = token * cols + bi * QK_K + elem;
                int4 q0 = q6_load4(
                    qlp, qhp, qh_shift, ql_high, within);
                int4 q1 = q6_load4(
                    qlp, qhp, qh_shift, ql_high, within + 4u);
                char4 x0 =
                    char4(*(device const packed_char4 *)(xq + xoff));
                char4 x1 = char4(
                    *(device const packed_char4 *)(xq + xoff + 4u));
                int4 ix0 = int4(x0);
                int4 ix1 = int4(x1);
                int sum0 = q0.x * ix0.x + q0.y * ix0.y +
                           q0.z * ix0.z + q0.w * ix0.w;
                int sum1 = q1.x * ix1.x + q1.y * ix1.y +
                           q1.z * ix1.z + q1.w * ix1.w;
                partial = (sum0 + sum1) * int(sc[scale]);
            }
            int block_sum = simd_sum(partial);
            if (lane == 0 && row < rows) {
                device const uchar *block = weights +
                    ((size_t)row * n_blocks + bi) * BLOCK_BYTES;
                float d = float(*(device const half *)(block + 208u));
                acc[r] += (d * xd[token * n_blocks + bi]) *
                          float(block_sum);
            }
        }
    }

    if (lane == 0)
        for (uint r = 0; r < 4; r++) {
            uint row = first_row + r;
            if (row < rows)
                out[out_offset + token * rows + row] = acc[r];
        }
    (void)bsums;
}
