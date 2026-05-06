#include <metal_stdlib>
using namespace metal;

// Q4_K two-output matvec split for stacked gate+up buffers.
// Layout: d FP16(0-1), dmin FP16(2-3), scales[12](4-15), qs[128](16-143)
// Dispatch: (ceil(rows/32), 1, 1)

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint ELEMS_PER_THREAD = 256 / THREADS_PER_ROW;
constant uint QK_K = 256;
constant uint BLOCK_BYTES = 144;

static inline uint2 get_scale_min(uint j, device const uchar *scales) {
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

kernel void q4k_matvec_split(device const uchar *weights [[buffer(0)]],
                             device const float *x       [[buffer(1)]],
                             device float       *out0    [[buffer(2)]],
                             device float       *out1    [[buffer(3)]],
                             constant uint      *p       [[buffer(4)]],
                             uint3 wid [[threadgroup_position_in_grid]],
                             uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], split = p[2];
    uint tile_start = wid.x * TILE_ROWS;
    uint tid = lid.x;

    uint local_row = tid / THREADS_PER_ROW;
    uint local_elem = tid % THREADS_PER_ROW;
    uint global_row = tile_start + local_row;

    uint n_blocks = cols / QK_K;
    uint row_byte = global_row * n_blocks * BLOCK_BYTES;
    uint my_start = local_elem * ELEMS_PER_THREAD;
    uint group = my_start / 64;
    uint is_high = (my_start % 64) / 32;
    uint sub = group * 2 + is_high;
    uint q_off_base = group * 32;

    float acc = 0.0f;
    if (global_row < rows) {
        for (uint bi = 0; bi < n_blocks; bi++) {
            device const uchar *block = weights + row_byte + bi * BLOCK_BYTES;
            float d    = float(*(device const half *)(block));
            float dmin = float(*(device const half *)(block + 2));
            device const uchar *scales = block + 4;
            device const uchar *qs = block + 16;
            uint elem_base = bi * QK_K;

            uint2 sm = get_scale_min(sub, scales);
            device const uchar *q_off = qs + q_off_base;

            float sum_qx = 0.0f;
            float sum_x = 0.0f;
            for (uint i = 0; i < 32; i++) {
                float xv = x[elem_base + my_start + i];
                uint qbyte = q_off[i];
                float qval = is_high == 0 ? float(qbyte & 0xF) : float(qbyte >> 4);
                sum_qx += qval * xv;
                sum_x += xv;
            }
            acc += d * float(sm.x) * sum_qx - dmin * float(sm.y) * sum_x;
        }
    }

    float val = acc;
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    if (local_elem == 0 && global_row < rows) {
        if (global_row >= split)
            out1[global_row - split] = val;
        else
            out0[global_row] = val;
    }
}
