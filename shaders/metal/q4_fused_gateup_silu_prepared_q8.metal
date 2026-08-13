#include <metal_stdlib>
using namespace metal;

// Q4_0 f16-scale repacked fused gate/up matvec with prepared Q8 activation blocks.

#define TILE_ROWS 16u

#define DQ4(w, sh) char4( \
    char(int(((w) >> (sh))       & 0xF) - 8), \
    char(int(((w) >> ((sh) + 4)) & 0xF) - 8), \
    char(int(((w) >> ((sh) + 8)) & 0xF) - 8), \
    char(int(((w) >> ((sh) + 12))& 0xF) - 8))

static inline float dot_char4(char4 a, char4 b) {
    return dot(float4(a), float4(b));
}

static inline float2 q4_q8_dot_pair(device const uint *gate,
                                    device const uint *up,
                                    device const char4 *xq) {
    float2 acc = 0.0f;
#define DOT_PAIR(w, sh, xv) do { \
    char4 x4 = (xv); \
    acc.x += dot_char4(DQ4(gate[(w)], (sh)), x4); \
    acc.y += dot_char4(DQ4(up[(w)], (sh)), x4); \
} while (0)
    DOT_PAIR(0,  0, xq[0]);
    DOT_PAIR(0, 16, xq[1]);
    DOT_PAIR(1,  0, xq[2]);
    DOT_PAIR(1, 16, xq[3]);
    DOT_PAIR(2,  0, xq[4]);
    DOT_PAIR(2, 16, xq[5]);
    DOT_PAIR(3,  0, xq[6]);
    DOT_PAIR(3, 16, xq[7]);
#undef DOT_PAIR
    return acc;
}

static inline float bn_fast_exp(float x) {
    const float log2e = 1.4426950409f;
    const float ln2 = 0.6931471806f;
    x = clamp(x, -87.3f, 88.7f);
    float n = floor(fma(x, log2e, 0.5f));
    float r = fma(-n, ln2, x);
    float poly = fma(0.04166664f, r, 0.16666667f);
    poly = fma(poly, r, 0.49999994f);
    poly = fma(poly, r, 1.0f);
    poly = fma(poly, r, 1.0f);
    int e = (int(n) + 127) << 23;
    return poly * as_type<float>(e);
}

static inline float bn_fast_silu(float x) {
    return x / (1.0f + bn_fast_exp(-x));
}

static inline float bn_reference_silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void q4_fused_gateup_silu_prepared_q8(
    device const uint  *weights  [[buffer(0)]],
    device const char  *x_q      [[buffer(1)]],
    device const float *x_scales [[buffer(2)]],
    device float       *out      [[buffer(3)]],
    constant uint      *p        [[buffer(4)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint total_rows = p[0], cols = p[1], gate_rows = p[2];
    uint tile_start = wid.x * TILE_ROWS;
    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint blocks_per_row = cols >> 5;
    uint total_blocks = total_rows * blocks_per_row;
    uint scale_words = (total_blocks + 1) >> 1;
    float gate_acc = 0.0f, up_acc = 0.0f;

    if (global_row < gate_rows) {
        uint gate_row_base = global_row * blocks_per_row;
        uint up_row_base = (global_row + gate_rows) * blocks_per_row;
        for (uint b = row_lane; b < blocks_per_row; b += 8) {
            float dx = x_scales[b];

            uint gate_block = gate_row_base + b;
            uint up_block = up_row_base + b;
            float gate_d = float(((device const half *)weights)[gate_block]);
            float up_d = float(((device const half *)weights)[up_block]);
            uint gate_nib = scale_words + gate_block * 4;
            uint up_nib = scale_words + up_block * 4;
            device const char4 *xqb = (device const char4 *)(x_q + b * 32);

            float2 dots = q4_q8_dot_pair(weights + gate_nib,
                                         weights + up_nib, xqb);
            gate_acc += gate_d * dx * dots.x;
            up_acc += up_d * dx * dots.y;
        }
    }

    gate_acc += simd_shuffle_xor(gate_acc, 1);
    gate_acc += simd_shuffle_xor(gate_acc, 2);
    gate_acc += simd_shuffle_xor(gate_acc, 4);
    up_acc += simd_shuffle_xor(up_acc, 1);
    up_acc += simd_shuffle_xor(up_acc, 2);
    up_acc += simd_shuffle_xor(up_acc, 4);

    if (row_lane == 0 && global_row < gate_rows) {
        float g = gate_acc;
        float u = up_acc;
        uint bias_offset = p[4];
        if (bias_offset > 0) {
            g += as_type<float>(weights[bias_offset + global_row]);
            u += as_type<float>(weights[bias_offset + global_row + gate_rows]);
        }
        float activated = (p[3] & 4u) != 0u
            ? bn_reference_silu(g)
            : bn_fast_silu(g);
        out[global_row] = activated * u;
    }
}

#undef DQ4
#undef TILE_ROWS
