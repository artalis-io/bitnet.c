#include <metal_stdlib>
using namespace metal;

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint ELEMS_PER_THREAD = 32;
constant uint QK_K = 256;
constant uint BLOCK_BYTES = 210;

static inline float4 q6_load4(device const uchar *qlp,
                              device const uchar *qhp,
                              uint qh_shift,
                              uint ql_high,
                              uint off) {
    if (ql_high) {
        return float4(float(((qlp[off] >> 4) |
                             (((qhp[off] >> qh_shift) & 3u) << 4))) - 32.0f,
                      float(((qlp[off + 1u] >> 4) |
                             (((qhp[off + 1u] >> qh_shift) & 3u) << 4))) - 32.0f,
                      float(((qlp[off + 2u] >> 4) |
                             (((qhp[off + 2u] >> qh_shift) & 3u) << 4))) - 32.0f,
                      float(((qlp[off + 3u] >> 4) |
                             (((qhp[off + 3u] >> qh_shift) & 3u) << 4))) - 32.0f);
    }

    return float4(float(((qlp[off] & 0xFu) |
                         (((qhp[off] >> qh_shift) & 3u) << 4))) - 32.0f,
                  float(((qlp[off + 1u] & 0xFu) |
                         (((qhp[off + 1u] >> qh_shift) & 3u) << 4))) - 32.0f,
                  float(((qlp[off + 2u] & 0xFu) |
                         (((qhp[off + 2u] >> qh_shift) & 3u) << 4))) - 32.0f,
                  float(((qlp[off + 3u] & 0xFu) |
                         (((qhp[off + 3u] >> qh_shift) & 3u) << 4))) - 32.0f);
}

static inline float4 q8_load4(device const char *xq, uint off) {
    return float4(float(xq[off]),
                  float(xq[off + 1u]),
                  float(xq[off + 2u]),
                  float(xq[off + 3u]));
}

kernel void q6k_q8k_matvec(device const uchar *weights [[buffer(0)]],
                           device const char  *xq      [[buffer(1)]],
                           device const float *xd      [[buffer(2)]],
                           device const short *bsums   [[buffer(3)]],
                           device float       *out     [[buffer(4)]],
                           constant uint      *p       [[buffer(5)]],
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
    uint row_byte = global_row * n_blocks * BLOCK_BYTES;
    uint my_start = local_elem * ELEMS_PER_THREAD;
    uint half_idx = my_start / 128u;
    uint quarter = (my_start % 128u) / 32u;
    uint ql_off = half_idx * 64u;
    uint qh_off = half_idx * 32u;
    uint sc_off = half_idx * 8u;
    uint s_base;
    uint qh_shift;
    uint ql_add;
    uint ql_high;
    switch (quarter) {
        case 0: s_base = 0u; qh_shift = 0u; ql_add = 0u;  ql_high = 0u; break;
        case 1: s_base = 2u; qh_shift = 2u; ql_add = 32u; ql_high = 0u; break;
        case 2: s_base = 4u; qh_shift = 4u; ql_add = 0u;  ql_high = 1u; break;
        default: s_base = 6u; qh_shift = 6u; ql_add = 32u; ql_high = 1u; break;
    }

    float acc = 0.0f;
    if (global_row < rows) {
        for (uint bi = 0; bi < n_blocks; bi++) {
            device const uchar *block = weights + row_byte + bi * BLOCK_BYTES;
            device const uchar *qlp = block + ql_off + ql_add;
            device const uchar *qhp = block + 128u + qh_off;
            device const char  *sc = (device const char *)(block + 192u);
            float d = float(*(device const half *)(block + 208u));
            float dx = xd[token * n_blocks + bi];
            uint elem_base = token * cols + bi * QK_K + my_start;

            float4 q0 = q6_load4(qlp, qhp, qh_shift, ql_high, 0u);
            float4 q1 = q6_load4(qlp, qhp, qh_shift, ql_high, 4u);
            float4 q2 = q6_load4(qlp, qhp, qh_shift, ql_high, 8u);
            float4 q3 = q6_load4(qlp, qhp, qh_shift, ql_high, 12u);
            float4 q4 = q6_load4(qlp, qhp, qh_shift, ql_high, 16u);
            float4 q5 = q6_load4(qlp, qhp, qh_shift, ql_high, 20u);
            float4 q6 = q6_load4(qlp, qhp, qh_shift, ql_high, 24u);
            float4 q7 = q6_load4(qlp, qhp, qh_shift, ql_high, 28u);

            float4 x0 = q8_load4(xq, elem_base);
            float4 x1 = q8_load4(xq, elem_base + 4u);
            float4 x2 = q8_load4(xq, elem_base + 8u);
            float4 x3 = q8_load4(xq, elem_base + 12u);
            float4 x4 = q8_load4(xq, elem_base + 16u);
            float4 x5 = q8_load4(xq, elem_base + 20u);
            float4 x6v = q8_load4(xq, elem_base + 24u);
            float4 x7 = q8_load4(xq, elem_base + 28u);

            float scale0 = d * dx * float(sc[sc_off + s_base]);
            float scale1 = d * dx * float(sc[sc_off + s_base + 1u]);
            acc += scale0 * (dot(q0, x0) + dot(q1, x1) + dot(q2, x2) + dot(q3, x3)) +
                   scale1 * (dot(q4, x4) + dot(q5, x5) + dot(q6, x6v) + dot(q7, x7));
        }
    }

    float val = acc;
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    if (local_elem == 0 && global_row < rows)
        out[out_offset + token * rows + global_row] = val;

    (void)bsums;
}
