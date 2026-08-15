#include <metal_stdlib>
using namespace metal;

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;

static inline float q4_1_block_dot(device const uchar *block,
                                   device const float4 *xp) {
    device const uchar *q = block + 4;
    float dot_q = 0.0f;
    float sum_x = 0.0f;
    for (uint i = 0; i < 4; i++) {
        uint o = i * 4;
        float4 lo = float4(q[o + 0] & 15, q[o + 1] & 15,
                           q[o + 2] & 15, q[o + 3] & 15);
        float4 hi = float4(q[o + 0] >> 4, q[o + 1] >> 4,
                           q[o + 2] >> 4, q[o + 3] >> 4);
        dot_q += dot(lo, xp[i]);
        dot_q += dot(hi, xp[i + 4]);
        sum_x += dot(float4(1.0f), xp[i]);
        sum_x += dot(float4(1.0f), xp[i + 4]);
    }
    float d = float(*(device const half *)block);
    float m = float(*(device const half *)(block + 2));
    return d * dot_q + m * sum_x;
}

kernel void q4_1_matvec(device const uchar *weights [[buffer(0)]],
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
    uint row_lane = tid % THREADS_PER_ROW;
    uint global_row = tile_start + local_row;
    uint blocks_per_row = cols / 32;
    uint x_base = token * cols;

    float acc = 0.0f;
    if (global_row < rows) {
        uint row_byte = global_row * blocks_per_row * 20;
        for (uint b = row_lane; b < blocks_per_row;
             b += THREADS_PER_ROW) {
            device const uchar *block = weights + row_byte + b * 20;
            device const float4 *xp =
                (device const float4 *)(x + x_base + b * 32);
            acc += q4_1_block_dot(block, xp);
        }
    }

    // Simdgroup reduction for 8 threads per row (no barriers needed)
    float val = acc;
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    if (row_lane == 0 && global_row < rows)
        out[out_offset + token * rows + global_row] = val;
}
