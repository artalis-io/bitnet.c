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
            for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
                uchar ql_val = ql[ql_off + ql_add + i];
                uchar qh_val = qh[qh_off + i];
                uint ql_bits = ql_high ? (ql_val >> 4) : (ql_val & 0xF);
                int q6 = int(ql_bits | (((qh_val >> qh_shift) & 3) << 4)) - 32;
                uint s_idx = i / 16 + s_base;
                acc += d * float(sc[sc_off + s_idx]) * float(q6) *
                       x[x_base + elem_base + my_start + i];
            }
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
