#include <metal_stdlib>
using namespace metal;

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint ELEMS_PER_THREAD = 256 / THREADS_PER_ROW;
constant uint BLOCK_BYTES = 176;

static inline uint2 get_scale_min_q5k(uint j, device const uchar *scales) {
    uint sc, m;
    if (j < 4) {
        sc = scales[j] & 63;
        m  = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        m  = (scales[j + 4] >> 4)  | ((scales[j] >> 6) << 4);
    }
    return uint2(sc, m);
}

kernel void q5k_matvec(device const uchar *weights [[buffer(0)]],
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
            float d    = float(*(device const half *)block);
            float dmin = float(*(device const half *)(block + 2));
            device const uchar *scales = block + 4;
            device const uchar *qh = block + 16;
            device const uchar *qs = block + 48;
            uint elem_base = bi * 256;
            uint my_start = local_elem * ELEMS_PER_THREAD;
            for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
                uint elem = my_start + i;
                uint group = elem / 32;
                uchar qbyte = qs[elem / 2];
                uint nibble = (elem % 2 == 0) ? (qbyte & 0xF) : (qbyte >> 4);
                uint hi = (qh[elem / 8] >> (elem % 8)) & 1;
                uint q5 = nibble | (hi << 4);
                uint2 sm = get_scale_min_q5k(group, scales);
                acc += (d * float(sm.x) * float(q5) - dmin * float(sm.y)) * x[x_base + elem_base + elem];
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
