#include <metal_stdlib>
using namespace metal;

// Four-row prepared Q4_0 layout with an FP32 activation. Eight lanes cooperate
// on each output row; four adjacent rows share the same packed block.
kernel void q4_prepared_f32_matvec(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3];
    uint bias_offset = p[4], out_offset = p[5];
    uint tile_start = extra > 0 ? (wid.x + wid.y * extra) * 32u
                                : wid.x * 32u;
    uint token = extra > 0 ? 0u : wid.y;
    uint row_lane = lid.x & 7u;
    uint row = tile_start + (lid.x >> 3);
    uint blocks_per_row = cols >> 5;
    uint group_blocks = (rows >> 2) * blocks_per_row;
    device const half *scales = (device const half *)weights;
    device const uchar *qs = weights + group_blocks * 4u * sizeof(half);
    float sum = 0.0f;

    if (row < rows) {
        uint group = row >> 2;
        uint row_in_group = row & 3u;
        device const float *xt = x + token * cols;
        for (uint b = row_lane; b < blocks_per_row; b += 8u) {
            uint gb = group * blocks_per_row + b;
            float block_sum = 0.0f;
            device const uchar *qbase = qs + gb * 64u;
            device const float *xb = xt + b * 32u;
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
    }

    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 4);
    if (row_lane == 0u && row < rows) {
        if (bias_offset > 0u)
            sum += ((device const float *)weights)[bias_offset + row];
        out[out_offset + token * rows + row] = sum;
    }
}
