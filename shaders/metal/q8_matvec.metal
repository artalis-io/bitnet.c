#include <metal_stdlib>
using namespace metal;

// Q8_0 TILED matvec — 8-bit quantization, 32 elements per block, 34 bytes
// Layout: FP16 scale (2 bytes) + 32 int8 values
// Dispatch: (ceil(rows/32), n_tokens, 1)

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint ELEMS_PER_THREAD = 32 / THREADS_PER_ROW;  // 4

kernel void q8_matvec(device const uchar *weights [[buffer(0)]],
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

    uint blocks_per_row = cols / 32;
    uint x_base = token * cols;
    uint row_byte_base = global_row * blocks_per_row * 34;

    float acc = 0.0f;
    if (global_row < rows) {
        #pragma clang loop unroll_count(4)
        for (uint b = 0; b < blocks_per_row; b++) {
            device const uchar *block = weights + row_byte_base + b * 34;
            float scale = float(*(device const half *)block);
            device const char *qs = (device const char *)(block + 2);
            uint elem_base = b * 32;
            uint my_start = local_elem * ELEMS_PER_THREAD;
            char4 qv = *(device const char4 *)(qs + my_start);
            float4 xv = *(device const float4 *)(x + x_base + elem_base + my_start);
            acc += scale * dot(float4(qv), xv);
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
