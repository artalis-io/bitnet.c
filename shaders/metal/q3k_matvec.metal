#include <metal_stdlib>
using namespace metal;

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint ELEMS_PER_THREAD = 256 / THREADS_PER_ROW;
constant uint BLOCK_BYTES = 110;

static inline int unpack_q3k_scale(uint j, device const uchar *scales) {
    uint idx = j / 2;
    int raw;
    if (j < 8) {
        raw = int(((j % 2 == 0) ? scales[idx] : (scales[idx] >> 4)) & 0xF);
    } else {
        uint lo = ((j % 2 == 0) ? scales[idx] : (scales[idx] >> 4)) & 0xF;
        uint hi = (scales[8 + (j - 8)] >> ((j % 2) * 4)) & 3;
        raw = int(lo | (hi << 4));
    }
    return raw - 32;
}

kernel void q3k_matvec(device const uchar *weights [[buffer(0)]],
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
        uint row_byte = global_row * n_blocks * BLOCK_BYTES;
        for (uint bi = 0; bi < n_blocks; bi++) {
            device const uchar *block = weights + row_byte + bi * BLOCK_BYTES;
            device const uchar *hmask = block;
            device const uchar *qs = block + 32;
            device const uchar *scales = block + 96;
            float d = float(*(device const half *)(block + 108));
            uint elem_base = bi * 256;
            uint my_start = local_elem * ELEMS_PER_THREAD;
            for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
                uint elem = my_start + i;
                uchar qbyte = qs[elem / 4];
                uint shift = (elem % 4) * 2;
                uint lo2 = (qbyte >> shift) & 3;
                uint hi = (hmask[elem / 8] >> (elem % 8)) & 1;
                int q3 = int(lo2 | (hi << 2)) - 4;
                uint group = elem / 16;
                int sc = unpack_q3k_scale(group, scales);
                acc += d * float(sc) * float(q3) * x[x_base + elem_base + elem];
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
