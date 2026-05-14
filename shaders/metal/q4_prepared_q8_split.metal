#include <metal_stdlib>
using namespace metal;

static inline int q4_byte_dot(uchar stored, device const char *x0,
                              device const char *x1) {
    uchar raw = stored ^ uchar(0x88);
    return (int(raw & uchar(0x0F)) - 8) * int(*x0) +
           (int(raw >> 4) - 8) * int(*x1);
}

static inline float q4_prepared_row_dot(device const uchar *weights,
                                        device const char *x_q,
                                        device const float *x_scales,
                                        uint rows, uint cols, uint row) {
    uint blocks_per_row = cols >> 5;
    uint n_groups = rows >> 2;
    uint n_group_blocks = n_groups * blocks_per_row;
    uint qs_offset = ((n_group_blocks * 4u * 2u) + 3u) & ~3u;
    device const ushort *scales = (device const ushort *)weights;
    device const uchar *qs = weights + qs_offset;
    uint group = row >> 2;
    uint row_in_group = row & 3u;
    float row_sum = 0.0f;
    for (uint b = 0; b < blocks_per_row; b++) {
        uint gb = group * blocks_per_row + b;
        float d = float(as_type<half>(scales[gb * 4u + row_in_group]));
        float dx = x_scales[b];
        device const uchar *qbase = qs + gb * 64u;
        device const char *xb = x_q + b * 32u;
        int idot = 0;
        for (uint ng = 0; ng < 4; ng++) {
            device const uchar *qrow = qbase + ng * 16u + row_in_group * 4u;
            uint xlo = ng * 4u;
            uint xhi = 16u + ng * 4u;
            idot += q4_byte_dot(qrow[0], xb + xlo + 0u, xb + xhi + 0u);
            idot += q4_byte_dot(qrow[1], xb + xlo + 1u, xb + xhi + 1u);
            idot += q4_byte_dot(qrow[2], xb + xlo + 2u, xb + xhi + 2u);
            idot += q4_byte_dot(qrow[3], xb + xlo + 3u, xb + xhi + 3u);
        }
        row_sum = fma(d * dx, float(idot), row_sum);
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
    if (lid.x >= 32) return;
    uint rows = p[0], cols = p[1], split1 = p[2], split2 = p[3];
    uint bias_offset = p[4], off0 = p[5], off1 = p[6], off2 = p[7];
    uint row = wid.x * 32u + lid.x;
    if (row >= rows) return;
    float acc = q4_prepared_row_dot(weights, x_q, x_scales, rows, cols, row);
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
