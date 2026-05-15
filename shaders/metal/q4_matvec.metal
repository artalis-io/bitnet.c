#include <metal_stdlib>
using namespace metal;

// Q4_0 REPACKED matvec — float4 vectorized, 32 rows/tile, 8 threads/row
//
// GPU buffer layout: [f32 scales: n_blocks][nibble u32s: n_blocks * 4]
//
// 8 threads per row, each processes blocks_per_row/8 blocks.
// Each block: 8 dot(float4) operations instead of 32 scalar multiply-adds.
// float4 loads for x vector (coalesced 16-byte reads).
// Reduction: simd_shuffle_xor (0 barriers).
//
// Dispatch: (ceil(rows/32), n_tokens, 1)

#define UQ4(w, sh) float4( \
    float(((w) >> (sh))        & 0xF), \
    float(((w) >> ((sh) + 4))  & 0xF), \
    float(((w) >> ((sh) + 8))  & 0xF), \
    float(((w) >> ((sh) + 12)) & 0xF))

static inline float q4_block_dot(uint w0, uint w1, uint w2, uint w3,
                                 float s, device const float4 *xp) {
    float4 sx0 = xp[0] + xp[1] + xp[2] + xp[3];
    float4 sx1 = xp[4] + xp[5] + xp[6] + xp[7];
    float4 sx = sx0 + sx1;
    float sumx = (sx.x + sx.y) + (sx.z + sx.w);
    float acc = 0.0f;
    acc += dot(UQ4(w0,  0), xp[0]);
    acc += dot(UQ4(w0, 16), xp[1]);
    acc += dot(UQ4(w1,  0), xp[2]);
    acc += dot(UQ4(w1, 16), xp[3]);
    acc += dot(UQ4(w2,  0), xp[4]);
    acc += dot(UQ4(w2, 16), xp[5]);
    acc += dot(UQ4(w3,  0), xp[6]);
    acc += dot(UQ4(w3, 16), xp[7]);
    return s * (acc - 8.0f * sumx);
}

kernel void q4_matvec(device const uint  *weights [[buffer(0)]],
                      device const float *x       [[buffer(1)]],
                      device float       *out     [[buffer(2)]],
                      constant uint      *p       [[buffer(3)]],
                      uint3 wid [[threadgroup_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3], bias_offset = p[4], out_offset = p[5];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * 32 : wid.x * 32;
    uint token = (extra > 0) ? 0 : wid.y;

    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint blocks_per_row = cols >> 5;
    uint total_blocks = rows * blocks_per_row;
    uint x_base = token * cols;

    float acc = 0.0f;

    if (global_row < rows) {
        uint row_block_base = global_row * blocks_per_row;

        for (uint b = row_lane; b < blocks_per_row; b += 8) {
            uint block_idx = row_block_base + b;
            float s = as_type<float>(weights[block_idx]);

            uint nib_base = total_blocks + block_idx * 4;
            uint w0 = weights[nib_base];
            uint w1 = weights[nib_base + 1];
            uint w2 = weights[nib_base + 2];
            uint w3 = weights[nib_base + 3];

            // float4 loads — coalesced 16-byte reads
            device const float4 *xp = (device const float4 *)(x + x_base + b * 32);

            acc += q4_block_dot(w0, w1, w2, w3, s, xp);
        }
    }

    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows) {
        float result = acc;
        if (bias_offset > 0)
            result += as_type<float>(weights[bias_offset + global_row]);
        out[out_offset + token * rows + global_row] = result;
    }
}

#undef UQ4
