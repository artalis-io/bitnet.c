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
    uint out_offset = p[5];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * TILE_ROWS : wid.x * TILE_ROWS;
    uint token = (extra > 0) ? 0 : wid.y;
    uint tid = lid.x;
    uint local_row = tid / THREADS_PER_ROW;
    uint local_elem = tid % THREADS_PER_ROW;
    uint global_row = tile_start + local_row;
    uint n_blocks = cols / 256;
    uint x_base = token * cols;
    uint my_start = local_elem * ELEMS_PER_THREAD;
    uint group = my_start / 64;
    uint is_high = (my_start % 64) / 32;
    uint bit = group * 2 + is_high;
    uint q_off_base = group * 32;

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
            device const uchar *q_off = qs + q_off_base;
            uint2 sm = get_scale_min_q5k(bit, scales);
            for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
                uchar qbyte = q_off[i];
                uint nibble = is_high == 0 ? (qbyte & 0xF) : (qbyte >> 4);
                uint hi = (qh[i] >> bit) & 1;
                uint q5 = nibble | (hi << 4);
                acc += (d * float(sm.x) * float(q5) - dmin * float(sm.y)) *
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
