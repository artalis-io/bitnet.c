#include <metal_stdlib>
using namespace metal;

// Q4_0 prepared-layout x Q8 activation matvec.
// Buffer layout: [u16 scales: group_blocks * 4][u8 qs: group_blocks * 64][bias].

static inline float q4_vec_dot(device const uchar *stored,
                               device const char *x0,
                               device const char *x1) {
    uchar4 raw = *(device const uchar4 *)stored;
    char4 lo = char4(raw & uchar4(0x0F)) - char4(8);
    char4 hi = char4(raw >> 4) - char4(8);
    return dot(float4(lo), float4(*(device const char4 *)x0)) +
           dot(float4(hi), float4(*(device const char4 *)x1));
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
            float idot = 0.0f;
            for (uint ng = 0; ng < 4; ng++) {
                device const uchar *qrow = qbase + ng * 16u + row_in_group * 4u;
                uint xlo = ng * 4u;
                uint xhi = 16u + ng * 4u;
                idot += q4_vec_dot(qrow, xb + xlo, xb + xhi);
            }
            row_sum = fma(d * dx, idot, row_sum);
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

kernel void q4_prepared_q8_matvec_reference(
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

    for (uint base = 0; base < blocks_per_row; base += 8u) {
        uint b = base + row_lane;
        float block_scale = 0.0f;
        float idot = 0.0f;
        if (row < rows && b < blocks_per_row) {
            uint gb = group * blocks_per_row + b;
            float d = float(as_type<half>(scales[gb * 4u + row_in_group]));
            block_scale = d * x_scales[scale_base + b];
            device const uchar *qbase = qs + gb * 64u;
            device const char *xb = x_q + x_base + b * 32u;
            for (uint ng = 0; ng < 4; ng++) {
                device const uchar *qrow =
                    qbase + ng * 16u + row_in_group * 4u;
                uint xlo = ng * 4u;
                uint xhi = 16u + ng * 4u;
                idot += q4_vec_dot(qrow, xb + xlo, xb + xhi);
            }
        }
        for (uint source = 0; source < 8u; source++) {
            float ordered_scale = simd_shuffle(
                block_scale, (local_row & 3u) * 8u + source);
            float ordered_idot = simd_shuffle(
                idot, (local_row & 3u) * 8u + source);
            if (row_lane == 0 && base + source < blocks_per_row)
                row_sum = fma(ordered_scale, ordered_idot, row_sum);
        }
    }
    if (row_lane == 0 && row < rows && bias_offset > 0)
        row_sum += as_type<float>(
            ((device const uint *)weights)[bias_offset + row]);
    if (row_lane == 0 && row < rows)
        out[out_offset + token * rows + row] = row_sum;
}
