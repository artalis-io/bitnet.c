#include <metal_stdlib>
using namespace metal;

constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint QK_K = 256;
constant uint BLOCK_BYTES = 144;

static inline uint2 get_scale_min(uint j, device const uchar *scales) {
    if (j < 4)
        return uint2(scales[j] & 63, scales[j + 4] & 63);
    return uint2((scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4),
                 (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
}

kernel void q4k_q8k_matvec(device const uchar *weights [[buffer(0)]],
                           device const char  *xq      [[buffer(1)]],
                           device const float *xd      [[buffer(2)]],
                           device const short *bsums   [[buffer(3)]],
                           device float       *out     [[buffer(4)]],
                           constant uint      *p       [[buffer(5)]],
                           uint3 wid [[threadgroup_position_in_grid]],
                           uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint tile = extra > 0 ? wid.x + wid.y * extra : wid.x;
    uint token = extra > 0 ? 0 : wid.y;
    uint lane = lid.x & 7u;
    uint row = tile * TILE_ROWS + (lid.x >> 3u);
    uint n_blocks = cols / QK_K;
    uint group = lane >> 1u;
    uint high = lane & 1u;
    uint sub = group * 2u + high;
    float acc = 0.0f;

    if (row < rows) {
        uint row_base = row * n_blocks * BLOCK_BYTES;
        uint token_block_base = token * n_blocks;
        for (uint b = 0; b < n_blocks; b++) {
            device const uchar *block = weights + row_base + b * BLOCK_BYTES;
            device const uchar *q = block + 16u + group * 32u;
            uint2 sm = get_scale_min(sub, block + 4u);
            uint x_base = token * cols + b * QK_K + lane * 32u;
            int qx = 0;
            for (uint i = 0; i < 32u; i++)
                qx += int(high ? q[i] >> 4 : q[i] & 15) * int(xq[x_base + i]);
            uint sb = token_block_base + b;
            int sx = int(bsums[sb * 16u + lane * 2u]) +
                     int(bsums[sb * 16u + lane * 2u + 1u]);
            float d = float(*(device const half *)block);
            float dmin = float(*(device const half *)(block + 2u));
            acc += xd[sb] * (d * float(sm.x) * float(qx) -
                             dmin * float(sm.y) * float(sx));
        }
    }

    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0 && row < rows)
        out[out_offset + token * rows + row] = acc;
}
