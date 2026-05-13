#include <metal_stdlib>
using namespace metal;

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint ELEMS_PER_THREAD = 32;
constant uint QK_K = 256;
constant uint BLOCK_BYTES = 210;

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

    float acc = 0.0f;
    if (global_row < rows) {
        for (uint bi = 0; bi < n_blocks; bi++) {
            device const uchar *block = weights + row_byte + bi * BLOCK_BYTES;
            device const uchar *ql = block;
            device const uchar *qh = block + 128;
            device const char  *sc = (device const char *)(block + 192);
            float d = float(*(device const half *)(block + 208));
            float dx = xd[token * n_blocks + bi];
            uint elem_base = token * cols + bi * QK_K;
            uint half_idx = my_start / 128u;
            uint quarter = (my_start % 128u) / 32u;
            uint ql_off = half_idx * 64u + ((quarter & 1u) ? 32u : 0u);
            uint qh_off = half_idx * 32u;
            uint sc_off = half_idx * 8u + (quarter & 1u) * 2u + (quarter >= 2u ? 4u : 0u);
            uint qh_shift = (quarter & 1u) ? 2u : 0u;
            qh_shift += (quarter >= 2u) ? 4u : 0u;
            bool high = quarter >= 2u;

            int sum0 = 0;
            int sum1 = 0;
            for (uint i = 0; i < 16u; i++) {
                uint l = ql_off + i;
                uint q = high
                    ? uint((ql[l] >> 4) | (((qh[qh_off + i] >> qh_shift) & 3u) << 4))
                    : uint((ql[l] & 0xFu) | (((qh[qh_off + i] >> qh_shift) & 3u) << 4));
                sum0 += int(q) * int(xq[elem_base + my_start + i]);
            }
            for (uint i = 0; i < 16u; i++) {
                uint l = ql_off + 16u + i;
                uint q = high
                    ? uint((ql[l] >> 4) | (((qh[qh_off + 16u + i] >> qh_shift) & 3u) << 4))
                    : uint((ql[l] & 0xFu) | (((qh[qh_off + 16u + i] >> qh_shift) & 3u) << 4));
                sum1 += int(q) * int(xq[elem_base + my_start + 16u + i]);
            }

            uint bsum_base = (token * n_blocks + bi) * 16u;
            uint g0 = my_start / 16u;
            int corr = int(sc[sc_off]) * int(bsums[bsum_base + g0]) +
                       int(sc[sc_off + 1u]) * int(bsums[bsum_base + g0 + 1u]);
            int sumi = int(sc[sc_off]) * sum0 + int(sc[sc_off + 1u]) * sum1;
            acc += d * dx * float(sumi - 32 * corr);
        }
    }

    float val = acc;
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    if (local_elem == 0 && global_row < rows)
        out[out_offset + token * rows + global_row] = val;
}
