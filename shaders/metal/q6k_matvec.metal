#include <metal_stdlib>
using namespace metal;

// Q6_K TILED matvec — 6-bit k-quant, 256 elements per block, 210 bytes
// Layout: ql[128](0-127), qh[64](128-191), scales[16](192-207), d FP16(208-209)
// Dispatch: (ceil(rows/32), n_tokens, 1)

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint ELEMS_PER_THREAD = 256 / THREADS_PER_ROW;
constant uint QK_K = 256;
constant uint BLOCK_BYTES = 210;

kernel void q6k_matvec(device const uchar *weights [[buffer(0)]],
                       device const float *x       [[buffer(1)]],
                       device float       *out     [[buffer(2)]],
                       constant uint      *p       [[buffer(3)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3];
    uint out_offset = p[5];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * TILE_ROWS : wid.x * TILE_ROWS;
    uint token = (extra > 0) ? 0 : wid.y;
    uint tid = lid.x;

    uint local_row = tid / THREADS_PER_ROW;
    uint local_elem = tid % THREADS_PER_ROW;
    uint global_row = tile_start + local_row;

    uint n_blocks = cols / QK_K;
    uint x_base = token * cols;
    uint row_byte = global_row * n_blocks * BLOCK_BYTES;

    float acc = 0.0f;
    if (global_row < rows) {
        for (uint bi = 0; bi < n_blocks; bi++) {
            device const uchar *block = weights + row_byte + bi * BLOCK_BYTES;
            device const uchar *ql = block;
            device const uchar *qh = block + 128;
            device const char  *sc = (device const char *)(block + 192);
            float d = float(*(device const half *)(block + 208));
            uint elem_base = bi * QK_K;

            uint my_start = local_elem * ELEMS_PER_THREAD;
            uint half_idx = my_start / 128;
            uint quarter = (my_start % 128) / 32;
            uint ql_off = half_idx * 64;
            uint qh_off = half_idx * 32;
            uint sc_off = half_idx * 8;
            uint s_base;
            uint qh_shift;
            uint ql_add;
            uint ql_high;
            switch (quarter) {
                case 0: s_base = 0; qh_shift = 0; ql_add = 0;  ql_high = 0; break;
                case 1: s_base = 2; qh_shift = 2; ql_add = 32; ql_high = 0; break;
                case 2: s_base = 4; qh_shift = 4; ql_add = 0;  ql_high = 1; break;
                default: s_base = 6; qh_shift = 6; ql_add = 32; ql_high = 1; break;
            }
            device const uchar *qlp = ql + ql_off + ql_add;
            device const uchar *qhp = qh + qh_off;
            device const float4 *xp = (device const float4 *)(x + x_base + elem_base + my_start);
            float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            float4 x4 = xp[4], x5 = xp[5], x6v = xp[6], x7 = xp[7];

            float4 q0, q1, q2, q3, q4, q5, q6, q7;
            if (ql_high) {
                q0 = float4(float(((qlp[0] >> 4) | (((qhp[0] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[1] >> 4) | (((qhp[1] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[2] >> 4) | (((qhp[2] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[3] >> 4) | (((qhp[3] >> qh_shift) & 3) << 4))) - 32.0f);
                q1 = float4(float(((qlp[4] >> 4) | (((qhp[4] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[5] >> 4) | (((qhp[5] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[6] >> 4) | (((qhp[6] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[7] >> 4) | (((qhp[7] >> qh_shift) & 3) << 4))) - 32.0f);
                q2 = float4(float(((qlp[8] >> 4) | (((qhp[8] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[9] >> 4) | (((qhp[9] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[10] >> 4) | (((qhp[10] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[11] >> 4) | (((qhp[11] >> qh_shift) & 3) << 4))) - 32.0f);
                q3 = float4(float(((qlp[12] >> 4) | (((qhp[12] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[13] >> 4) | (((qhp[13] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[14] >> 4) | (((qhp[14] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[15] >> 4) | (((qhp[15] >> qh_shift) & 3) << 4))) - 32.0f);
                q4 = float4(float(((qlp[16] >> 4) | (((qhp[16] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[17] >> 4) | (((qhp[17] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[18] >> 4) | (((qhp[18] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[19] >> 4) | (((qhp[19] >> qh_shift) & 3) << 4))) - 32.0f);
                q5 = float4(float(((qlp[20] >> 4) | (((qhp[20] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[21] >> 4) | (((qhp[21] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[22] >> 4) | (((qhp[22] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[23] >> 4) | (((qhp[23] >> qh_shift) & 3) << 4))) - 32.0f);
                q6 = float4(float(((qlp[24] >> 4) | (((qhp[24] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[25] >> 4) | (((qhp[25] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[26] >> 4) | (((qhp[26] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[27] >> 4) | (((qhp[27] >> qh_shift) & 3) << 4))) - 32.0f);
                q7 = float4(float(((qlp[28] >> 4) | (((qhp[28] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[29] >> 4) | (((qhp[29] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[30] >> 4) | (((qhp[30] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[31] >> 4) | (((qhp[31] >> qh_shift) & 3) << 4))) - 32.0f);
            } else {
                q0 = float4(float(((qlp[0] & 0xF) | (((qhp[0] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[1] & 0xF) | (((qhp[1] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[2] & 0xF) | (((qhp[2] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[3] & 0xF) | (((qhp[3] >> qh_shift) & 3) << 4))) - 32.0f);
                q1 = float4(float(((qlp[4] & 0xF) | (((qhp[4] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[5] & 0xF) | (((qhp[5] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[6] & 0xF) | (((qhp[6] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[7] & 0xF) | (((qhp[7] >> qh_shift) & 3) << 4))) - 32.0f);
                q2 = float4(float(((qlp[8] & 0xF) | (((qhp[8] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[9] & 0xF) | (((qhp[9] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[10] & 0xF) | (((qhp[10] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[11] & 0xF) | (((qhp[11] >> qh_shift) & 3) << 4))) - 32.0f);
                q3 = float4(float(((qlp[12] & 0xF) | (((qhp[12] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[13] & 0xF) | (((qhp[13] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[14] & 0xF) | (((qhp[14] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[15] & 0xF) | (((qhp[15] >> qh_shift) & 3) << 4))) - 32.0f);
                q4 = float4(float(((qlp[16] & 0xF) | (((qhp[16] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[17] & 0xF) | (((qhp[17] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[18] & 0xF) | (((qhp[18] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[19] & 0xF) | (((qhp[19] >> qh_shift) & 3) << 4))) - 32.0f);
                q5 = float4(float(((qlp[20] & 0xF) | (((qhp[20] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[21] & 0xF) | (((qhp[21] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[22] & 0xF) | (((qhp[22] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[23] & 0xF) | (((qhp[23] >> qh_shift) & 3) << 4))) - 32.0f);
                q6 = float4(float(((qlp[24] & 0xF) | (((qhp[24] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[25] & 0xF) | (((qhp[25] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[26] & 0xF) | (((qhp[26] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[27] & 0xF) | (((qhp[27] >> qh_shift) & 3) << 4))) - 32.0f);
                q7 = float4(float(((qlp[28] & 0xF) | (((qhp[28] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[29] & 0xF) | (((qhp[29] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[30] & 0xF) | (((qhp[30] >> qh_shift) & 3) << 4))) - 32.0f,
                            float(((qlp[31] & 0xF) | (((qhp[31] >> qh_shift) & 3) << 4))) - 32.0f);
            }
            float scale0 = d * float(sc[sc_off + s_base]);
            float scale1 = d * float(sc[sc_off + s_base + 1]);
            acc += scale0 * (dot(q0, x0) + dot(q1, x1) + dot(q2, x2) + dot(q3, x3)) +
                   scale1 * (dot(q4, x4) + dot(q5, x5) + dot(q6, x6v) + dot(q7, x7));
        }
    }

    // Simdgroup reduction for 8 threads per row (no barriers needed)
    float val = acc;
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    if (local_elem == 0 && global_row < rows)
        out[out_offset + token * rows + global_row] = val;
}
