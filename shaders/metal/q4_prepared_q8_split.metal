#include <metal_stdlib>
using namespace metal;

static inline float q4_vec_dot(device const uchar *stored,
                               device const char *x0,
                               device const char *x1) {
    uchar4 raw = *(device const uchar4 *)stored;
    char4 lo = char4(raw & uchar4(0x0F)) - char4(8);
    char4 hi = char4(raw >> 4) - char4(8);
    return dot(float4(lo), float4(*(device const char4 *)x0)) +
           dot(float4(hi), float4(*(device const char4 *)x1));
}

static inline float q4_prepared_row_dot(device const uchar *weights,
                                        device const char *x_q,
                                        device const float *x_scales,
                                        uint rows, uint cols, uint row,
                                        uint row_lane) {
    uint blocks_per_row = cols >> 5;
    uint n_groups = rows >> 2;
    uint n_group_blocks = n_groups * blocks_per_row;
    uint qs_offset = ((n_group_blocks * 4u * 2u) + 3u) & ~3u;
    device const ushort *scales = (device const ushort *)weights;
    device const uchar *qs = weights + qs_offset;
    uint group = row >> 2;
    uint row_in_group = row & 3u;
    float row_sum = 0.0f;
    for (uint b = row_lane; b < blocks_per_row; b += 8u) {
        uint gb = group * blocks_per_row + b;
        float d = float(as_type<half>(scales[gb * 4u + row_in_group]));
        float dx = x_scales[b];
        device const uchar *qbase = qs + gb * 64u;
        device const char *xb = x_q + b * 32u;
        float idot = 0.0f;
        for (uint ng = 0; ng < 4; ng++) {
            device const uchar *qrow = qbase + ng * 16u + row_in_group * 4u;
            uint xlo = ng * 4u;
            uint xhi = 16u + ng * 4u;
            idot += q4_vec_dot(qrow, xb + xlo, xb + xhi);
        }
        row_sum = fma(d * dx, idot, row_sum);
    }
    return row_sum;
}

kernel void q4_prepared_q8_split(
    device const uchar *weights  [[buffer(0)]],
    device const char  *x_q      [[buffer(1)]],
    device const float *x_scales [[buffer(2)]],
    device float       *out0     [[buffer(3)]],
    device float       *out1     [[buffer(4)]],
    device float       *out2     [[buffer(5)]],
    constant uint      *p        [[buffer(6)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint row_lane = lid.x & 7u;
    uint local_row = lid.x >> 3;
    uint rows = p[0], cols = p[1], split1 = p[2], split2 = p[3];
    uint bias_offset = p[4], off0 = p[5], off1 = p[6], off2 = p[7];
    uint row = wid.x * 32u + local_row;
    float acc = 0.0f;
    if (row < rows)
        acc = q4_prepared_row_dot(weights, x_q, x_scales, rows, cols, row, row_lane);
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    if (row_lane == 0 && row < rows) {
        if (bias_offset > 0)
            acc += as_type<float>(((device const uint *)weights)[bias_offset + row]);
        if (split2 > 0 && row >= split2) {
            out2[off2 + row - split2] = acc;
        } else if (row >= split1) {
            out1[off1 + row - split1] = acc;
        } else {
            out0[off0 + row] = acc;
        }
    }
}
