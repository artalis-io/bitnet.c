#include <metal_stdlib>
using namespace metal;

// I2_S TILED matvec — 2-bit interleaved ternary
// 128 elements per chunk, 32 bytes per chunk, per-tensor scale at end
// Dispatch: (ceil(rows/32), n_tokens, 1)

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint ELEMS_PER_THREAD = 128 / THREADS_PER_ROW;  // 16
constant uint CHUNK_SIZE = 128;

kernel void i2s_matvec(device const uchar *weights [[buffer(0)]],
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

    uint chunks_per_row = cols / CHUNK_SIZE;
    uint x_base = token * cols;

    // Per-tensor scale at end of weight data
    device const uint *w32 = (device const uint *)weights;
    uint scale_offset = rows * cols / 16;
    float scale = as_type<float>(w32[scale_offset]);

    uint row_byte_offset = global_row * chunks_per_row * 32;

    float acc = 0.0f;
    if (global_row < rows) {
        for (uint c = 0; c < chunks_per_row; c++) {
            device const uchar *chunk = weights + row_byte_offset + c * 32;
            uint elem_base = c * CHUNK_SIZE;
            uint my_start = local_elem * ELEMS_PER_THREAD;

            for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
                uint elem = my_start + i;
                uint quarter = elem / 32;
                uint gp = elem % 32;
                uchar byte_val = chunk[gp];
                uint shift = (3 - quarter) * 2;
                uint v = (byte_val >> shift) & 3;
                float dv = (v == 3) ? 0.0f : float(int(v) - 1);
                acc += dv * x[x_base + elem_base + elem];
            }
        }
    }

    // Simdgroup reduction for 8 threads per row (no barriers needed)
    float val = acc;
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    if (local_elem == 0 && global_row < rows)
        out[token * rows + global_row] = val * scale;
}
