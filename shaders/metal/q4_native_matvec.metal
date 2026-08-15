#include <metal_stdlib>
using namespace metal;

kernel void q4_native_matvec(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint tile_start = extra > 0 ? (wid.x + wid.y * extra) * 4 : wid.x * 4;
    uint token = extra > 0 ? 0 : wid.y;
    uint lane = lid.x & 31;
    uint block_lane = lane >> 1;
    uint half_off = (lane & 1) * 8;
    uint blocks_per_row = cols >> 5;
    uint row_bytes = blocks_per_row * 18;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint b = block_lane; b < blocks_per_row; b += 16) {
        device const float *xb = x + token * cols + b * 32 + half_off;
        float4 x0 = *(device const float4 *)(xb + 0);
        float4 x1 = *(device const float4 *)(xb + 4);
        float4 x2 = *(device const float4 *)(xb + 16);
        float4 x3 = *(device const float4 *)(xb + 20);
        float sumx = (x0.x + x0.y) + (x0.z + x0.w) +
                     (x1.x + x1.y) + (x1.z + x1.w) +
                     (x2.x + x2.y) + (x2.z + x2.w) +
                     (x3.x + x3.y) + (x3.z + x3.w);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 4; r++) {
            uint row = tile_start + r;
            if (row >= rows) continue;
            device const uchar *block = weights + row * row_bytes + b * 18;
            device const uchar *q = block + 2 + half_off;
            uchar4 q0 = *(device const packed_uchar4 *)(q + 0);
            uchar4 q1 = *(device const packed_uchar4 *)(q + 4);
            float dotv = dot(float4(q0 & uchar4(15)), x0) +
                         dot(float4(q1 & uchar4(15)), x1) +
                         dot(float4(q0 >> uchar4(4)), x2) +
                         dot(float4(q1 >> uchar4(4)), x3);
            acc[r] += float(*(device const half *)block) *
                      (dotv - 8.0f * sumx);
        }
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; r++) {
        float total = simd_sum(acc[r]);
        uint row = tile_start + r;
        if (lane == 0 && row < rows)
            out[out_offset + token * rows + row] = total;
    }
}
