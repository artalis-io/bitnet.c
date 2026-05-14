#include <metal_stdlib>
using namespace metal;

// Q4_0 repacked fused gate/up matvec with prequantized Q8 activation blocks.

#define DQ4(w, sh) char4( \
    char(int(((w) >> (sh))       & 0xF) - 8), \
    char(int(((w) >> ((sh) + 4)) & 0xF) - 8), \
    char(int(((w) >> ((sh) + 8)) & 0xF) - 8), \
    char(int(((w) >> ((sh) + 12))& 0xF) - 8))

static inline float dot_char4(char4 a, char4 b) {
    return dot(float4(a), float4(b));
}

static inline float q4_q8_dot(uint w0, uint w1, uint w2, uint w3,
                              device const char4 *xq) {
    float acc = 0.0f;
    acc += dot_char4(DQ4(w0,  0), xq[0]);
    acc += dot_char4(DQ4(w0, 16), xq[1]);
    acc += dot_char4(DQ4(w1,  0), xq[2]);
    acc += dot_char4(DQ4(w1, 16), xq[3]);
    acc += dot_char4(DQ4(w2,  0), xq[4]);
    acc += dot_char4(DQ4(w2, 16), xq[5]);
    acc += dot_char4(DQ4(w3,  0), xq[6]);
    acc += dot_char4(DQ4(w3, 16), xq[7]);
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

kernel void q4_fused_gateup_silu_q8_prequant(
    device const uint  *weights  [[buffer(0)]],
    device const char  *x_q      [[buffer(1)]],
    device const float *x_scales [[buffer(2)]],
    device float       *out      [[buffer(3)]],
    constant uint      *p        [[buffer(4)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint total_rows = p[0], cols = p[1], gate_rows = p[2];
    uint tile_start = wid.x * 32;
    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint blocks_per_row = cols >> 5;
    uint total_blocks = total_rows * blocks_per_row;
    float gate_acc = 0.0f, up_acc = 0.0f;

    if (global_row < gate_rows) {
        uint gate_row_base = global_row * blocks_per_row;
        uint up_row_base = (global_row + gate_rows) * blocks_per_row;
        for (uint b = row_lane; b < blocks_per_row; b += 8) {
            float dx = x_scales[b];
            if (dx == 0.0f)
                continue;

            uint gate_block = gate_row_base + b;
            uint up_block = up_row_base + b;
            float gate_d = as_type<float>(weights[gate_block]);
            float up_d = as_type<float>(weights[up_block]);
            uint gate_nib = total_blocks + gate_block * 4;
            uint up_nib = total_blocks + up_block * 4;
            device const char4 *xqb = (device const char4 *)(x_q + b * 32);

            float gate_dot = q4_q8_dot(weights[gate_nib], weights[gate_nib + 1],
                                       weights[gate_nib + 2], weights[gate_nib + 3],
                                       xqb);
            float up_dot = q4_q8_dot(weights[up_nib], weights[up_nib + 1],
                                     weights[up_nib + 2], weights[up_nib + 3],
                                     xqb);
            gate_acc += gate_d * dx * gate_dot;
            up_acc += up_d * dx * up_dot;
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
        out[global_row] = bn_fast_silu(g) * u;
    }
}

#undef DQ4
