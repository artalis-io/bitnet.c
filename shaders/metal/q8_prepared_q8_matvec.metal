#include <metal_stdlib>
using namespace metal;

// Q8_0 weights x blockwise-Q8 activations. Eight lanes cooperate per row.
kernel void q8_prepared_q8_matvec(
    device const uchar *weights [[buffer(0)]],
    device const char *x_q [[buffer(1)]],
    device const float *x_scales [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint tile = (extra > 0) ? wid.x + wid.y * extra : wid.x;
    uint token = (extra > 0) ? 0 : wid.y;
    uint row_lane = lid.x & 7;
    uint row = tile * 32 + (lid.x >> 3);
    uint blocks_per_row = cols >> 5;
    uint x_base = token * cols;
    uint scale_base = token * blocks_per_row;
    float row_sum = 0.0f;

    if (row < rows) {
        uint row_byte_base = row * blocks_per_row * 34;
        for (uint b = 0; b < blocks_per_row; b++) {
            device const uchar *block = weights + row_byte_base + b * 34;
            device const char *wq = (device const char *)(block + 2);
            uint elem = b * 32 + row_lane * 4;
            char4 wv = *(device const char4 *)(wq + row_lane * 4);
            char4 xv = *(device const char4 *)(x_q + x_base + elem);
            int4 wi = int4(wv);
            int4 xi = int4(xv);
            int block_sum = wi.x * xi.x + wi.y * xi.y +
                            wi.z * xi.z + wi.w * xi.w;
            block_sum += simd_shuffle_xor(block_sum, 4);
            block_sum += simd_shuffle_xor(block_sum, 2);
            block_sum += simd_shuffle_xor(block_sum, 1);
            if (row_lane == 0) {
                float dw = float(*(device const half *)block);
                float dx = float(half(x_scales[scale_base + b]));
                row_sum += float(block_sum) * dw * dx;
            }
        }
    }
    if (row_lane == 0 && row < rows)
        out[out_offset + token * rows + row] = row_sum;
}
