#include <metal_stdlib>
using namespace metal;

constant uint TILE_ROWS = 32;
constant uint QK_K = 256;
constant uint BLOCK_BYTES = 144;

static inline uint2 get_scale_min(uint j, device const uchar *scales) {
    if (j < 4)
        return uint2(scales[j] & 63, scales[j + 4] & 63);
    return uint2((scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4),
                 (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
}

kernel void q4k_q8k_matvec_split(
    device const uchar *weights [[buffer(0)]],
    device const char *xq [[buffer(1)]],
    device const float *xd [[buffer(2)]],
    device const short *bsums [[buffer(3)]],
    device float *out0 [[buffer(4)]],
    device float *out1 [[buffer(5)]],
    device float *out2 [[buffer(6)]],
    constant uint *p [[buffer(7)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], split1 = p[2], split2 = p[3];
    uint off0 = p[5], off1 = p[6], off2 = p[7];
    uint lane = lid.x & 7u;
    uint row = wid.x * TILE_ROWS + (lid.x >> 3u);
    uint n_blocks = cols / QK_K;
    uint group = lane >> 1u;
    uint high = lane & 1u;
    uint sub = group * 2u + high;
    float acc = 0.0f;

    if (row < rows) {
        uint row_base = row * n_blocks * BLOCK_BYTES;
        for (uint b = 0; b < n_blocks; b++) {
            device const uchar *block = weights + row_base + b * BLOCK_BYTES;
            device const uchar *q = block + 16u + group * 32u;
            uint2 sm = get_scale_min(sub, block + 4u);
            uint x_base = b * QK_K + lane * 32u;
            int qx = 0;
            for (uint i = 0; i < 32u; i++)
                qx += int(high ? q[i] >> 4 : q[i] & 15) * int(xq[x_base + i]);
            int sx = int(bsums[b * 16u + lane * 2u]) +
                     int(bsums[b * 16u + lane * 2u + 1u]);
            int scaled_dot = int(sm.x) * qx;
            int min_corr = int(sm.y) * sx;
            scaled_dot += simd_shuffle_xor(scaled_dot, 4);
            min_corr += simd_shuffle_xor(min_corr, 4);
            scaled_dot += simd_shuffle_xor(scaled_dot, 2);
            min_corr += simd_shuffle_xor(min_corr, 2);
            scaled_dot += simd_shuffle_xor(scaled_dot, 1);
            min_corr += simd_shuffle_xor(min_corr, 1);
            float d = float(*(device const half *)block);
            float dmin = float(*(device const half *)(block + 2u));
            float negative_corr = -dmin * float(min_corr);
            float block_dot = fma(d, float(scaled_dot), negative_corr);
            acc = fma(xd[b], block_dot, acc);
        }
    }

    if (lane == 0 && row < rows) {
        if (split2 > 0 && row >= split2)
            out2[off2 + row - split2] = acc;
        else if (row >= split1)
            out1[off1 + row - split1] = acc;
        else
            out0[off0 + row] = acc;
    }
}
