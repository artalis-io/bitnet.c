#include <metal_stdlib>
using namespace metal;

// Q4_0 prepared-layout x Q8 activation matvec.
// Buffer layout: [u16 scales: group_blocks * 4][u8 qs: group_blocks * 64][bias].

static inline int q4_byte_dot(uchar stored, device const char *x0,
                              device const char *x1) {
    uchar raw = stored ^ uchar(0x88);
    return (int(raw & uchar(0x0F)) - 8) * int(*x0) +
           (int(raw >> 4) - 8) * int(*x1);
}

kernel void q4_prepared_q8_matvec(
    device const uchar *weights  [[buffer(0)]],
    device const char  *x_q      [[buffer(1)]],
    device const float *x_scales [[buffer(2)]],
    device float       *out      [[buffer(3)]],
    constant uint      *p        [[buffer(4)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    uint rows = p[0], cols = p[1], extra = p[3], bias_offset = p[4],
         out_offset = p[5];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * 32 : wid.x * 32;
    uint token = (extra > 0) ? 0 : wid.y;
    uint row_lane = lid.x & 7u;
    uint local_row = lid.x >> 3;
    uint row = tile_start + local_row;

    uint blocks_per_row = cols >> 5;
    uint n_groups = rows >> 2;
    uint n_group_blocks = n_groups * blocks_per_row;
    uint qs_offset = ((n_group_blocks * 4u * 2u) + 3u) & ~3u;
    device const ushort *scales = (device const ushort *)weights;
    device const uchar *qs = weights + qs_offset;

    uint group = row >> 2;
    uint row_in_group = row & 3u;
    uint x_base = token * cols;
    uint scale_base = token * blocks_per_row;
    float row_sum = 0.0f;

    if (row < rows) {
        for (uint b = row_lane; b < blocks_per_row; b += 8u) {
            uint gb = group * blocks_per_row + b;
            float d = float(as_type<half>(scales[gb * 4u + row_in_group]));
            float dx = x_scales[scale_base + b];

            device const uchar *qbase = qs + gb * 64u;
            device const char *xb = x_q + x_base + b * 32u;
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
    }
    row_sum += simd_shuffle_xor(row_sum, 1);
    row_sum += simd_shuffle_xor(row_sum, 2);
    row_sum += simd_shuffle_xor(row_sum, 4);

    if (row_lane == 0 && row < rows) {
        if (bias_offset > 0)
            row_sum += as_type<float>(((device const uint *)weights)[bias_offset + row]);
        out[out_offset + token * rows + row] = row_sum;
    }
}
