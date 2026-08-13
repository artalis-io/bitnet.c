#include <metal_stdlib>
using namespace metal;

static inline float q4_prepared_f32_row_dot(
    device const uchar *weights, device const float *x,
    uint rows, uint cols, uint row, uint row_lane) {
    uint blocks_per_row = cols >> 5;
    uint group_blocks = (rows >> 2) * blocks_per_row;
    device const half *scales = (device const half *)weights;
    device const uchar *qs = weights + group_blocks * 4u * sizeof(half);
    uint group = row >> 2;
    uint row_in_group = row & 3u;
    float sum = 0.0f;
    for (uint b = row_lane; b < blocks_per_row; b += 8u) {
        uint gb = group * blocks_per_row + b;
        device const uchar *qbase = qs + gb * 64u;
        device const float *xb = x + b * 32u;
        float block_sum = 0.0f;
        for (uint ng = 0; ng < 4u; ng++) {
            device const uchar *qrow =
                qbase + ng * 16u + row_in_group * 4u;
            uchar4 raw = *(device const uchar4 *)qrow;
            float4 lo = float4(char4(raw & uchar4(0x0f)) - char4(8));
            float4 hi = float4(char4(raw >> 4) - char4(8));
            block_sum += dot(lo, *(device const float4 *)(xb + ng * 4u));
            block_sum += dot(hi, *(device const float4 *)(xb + 16u + ng * 4u));
        }
        sum = fma(float(scales[gb * 4u + row_in_group]), block_sum, sum);
    }
    return sum;
}

kernel void q4_prepared_f32_split(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out0 [[buffer(2)]],
    device float *out1 [[buffer(3)]],
    device float *out2 [[buffer(4)]],
    constant uint *p [[buffer(5)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint row_lane = lid.x & 7u;
    uint row = wid.x * 32u + (lid.x >> 3);
    uint rows = p[0], cols = p[1], split1 = p[2], split2 = p[3];
    uint bias_offset = p[4], off0 = p[5], off1 = p[6], off2 = p[7];
    float sum = row < rows
        ? q4_prepared_f32_row_dot(weights, x, rows, cols, row, row_lane)
        : 0.0f;
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 4);
    if (row_lane == 0u && row < rows) {
        if (bias_offset > 0u)
            sum += ((device const float *)weights)[bias_offset + row];
        if (split2 > 0u && row >= split2)
            out2[off2 + row - split2] = sum;
        else if (row >= split1)
            out1[off1 + row - split1] = sum;
        else
            out0[off0 + row] = sum;
    }
}
