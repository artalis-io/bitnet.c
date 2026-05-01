#include <metal_stdlib>
using namespace metal;

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint ELEMS_PER_THREAD = 256 / THREADS_PER_ROW;

kernel void tq1_matvec(device const uchar *weights [[buffer(0)]],
                       device const float *x       [[buffer(1)]],
                       device float       *out     [[buffer(2)]],
                       constant uint      *p       [[buffer(3)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * TILE_ROWS : wid.x * TILE_ROWS;
    uint token = (extra > 0) ? 0 : wid.y;
    uint tid = lid.x;
    uint local_row = tid / THREADS_PER_ROW;
    uint local_elem = tid % THREADS_PER_ROW;
    uint global_row = tile_start + local_row;
    uint n_blocks = cols / 256;
    uint x_base = token * cols;

    float acc = 0.0f;
    if (global_row < rows) {
        uint row_byte = global_row * n_blocks * 54;
        for (uint bi = 0; bi < n_blocks; bi++) {
            device const uchar *block = weights + row_byte + bi * 54;
            device const uchar *qs = block;
            device const uchar *qh = block + 48;
            float d = float(*(device const half *)(block + 52));
            uint elem_base = bi * 256;
            uint my_start = local_elem * ELEMS_PER_THREAD;
            for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
                uint elem = my_start + i;
                int trit;
                if (elem < 240) {
                    uint byte_idx = elem / 5;
                    uint pos = elem % 5;
                    uint val = qs[byte_idx];
                    for (uint p2 = 0; p2 < pos; p2++) val /= 3;
                    trit = int(val % 3) - 1;
                } else {
                    uint idx = elem - 240;
                    uint byte_idx = idx / 4;
                    uint pos = idx % 4;
                    trit = int((qh[byte_idx] >> (pos * 2)) & 3) - 1;
                }
                acc += float(trit) * d * x[x_base + elem_base + elem];
            }
        }
    }

    // Simdgroup reduction for 8 threads per row (no barriers needed)
    float val = acc;
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    if (local_elem == 0 && global_row < rows)
        out[token * rows + global_row] = val;
}
