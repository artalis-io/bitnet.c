#include <metal_stdlib>
using namespace metal;

// Q4_0 repacked fused gate/up matvec with SiLU activation.
// Buffer layout: [f16 scales: n_blocks, u32 aligned][nibble u32s: n_blocks * 4][optional f32 bias].

#define UQ4(w, sh) float4( \
    float(((w) >> (sh))        & 0xF), \
    float(((w) >> ((sh) + 4))  & 0xF), \
    float(((w) >> ((sh) + 8))  & 0xF), \
    float(((w) >> ((sh) + 12)) & 0xF))

static inline float q4_block_sumx(device const float4 *xp) {
    float4 sx0 = xp[0] + xp[1] + xp[2] + xp[3];
    float4 sx1 = xp[4] + xp[5] + xp[6] + xp[7];
    float4 sx = sx0 + sx1;
    return (sx.x + sx.y) + (sx.z + sx.w);
}

static inline float q4_block_dot_sumx(uint w0, uint w1, uint w2, uint w3,
                                      float s, float sumx,
                                      device const float4 *xp) {
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

kernel void q4_fused_gateup_silu(device const uint  *weights [[buffer(0)]],
                                 device const float *x       [[buffer(1)]],
                                 device float       *out     [[buffer(2)]],
                                 constant uint      *p       [[buffer(3)]],
                                 uint3 wid [[threadgroup_position_in_grid]],
                                 uint3 lid [[thread_position_in_threadgroup]]) {
    uint total_rows = p[0], cols = p[1], gate_rows = p[2];
    uint bias_offset = p[4];
    uint tile_start = wid.x * 64;
    uint row_lane = lid.x & 7;
    uint local_pair = lid.x >> 3;
    uint global_row = tile_start + local_pair * 2;

    uint blocks_per_row = cols >> 5;
    uint total_blocks = total_rows * blocks_per_row;
    uint scale_words = (total_blocks + 1) >> 1;
    float2 gate_acc = 0.0f, up_acc = 0.0f;
    if (global_row < gate_rows) {
        uint row1 = min(global_row + 1, gate_rows - 1);
        uint2 gate_row_base = uint2(global_row, row1) * blocks_per_row;
        uint2 up_row_base = uint2(global_row + gate_rows,
                                  row1 + gate_rows) * blocks_per_row;
        for (uint b = row_lane; b < blocks_per_row; b += 8) {
            device const float4 *xp = (device const float4 *)(x + b * 32);
            float sumx = q4_block_sumx(xp);
            for (uint r = 0; r < 2; r++) {
                uint gate_block = gate_row_base[r] + b;
                uint up_block = up_row_base[r] + b;
                float gate_s = float(((device const half *)weights)[gate_block]);
                float up_s = float(((device const half *)weights)[up_block]);
                uint gate_nib = scale_words + gate_block * 4;
                uint up_nib = scale_words + up_block * 4;
                gate_acc[r] += q4_block_dot_sumx(
                    weights[gate_nib], weights[gate_nib + 1],
                    weights[gate_nib + 2], weights[gate_nib + 3],
                    gate_s, sumx, xp);
                up_acc[r] += q4_block_dot_sumx(
                    weights[up_nib], weights[up_nib + 1],
                    weights[up_nib + 2], weights[up_nib + 3],
                    up_s, sumx, xp);
            }
        }
    }

    gate_acc += simd_shuffle_xor(gate_acc, 1);
    gate_acc += simd_shuffle_xor(gate_acc, 2);
    gate_acc += simd_shuffle_xor(gate_acc, 4);
    up_acc += simd_shuffle_xor(up_acc, 1);
    up_acc += simd_shuffle_xor(up_acc, 2);
    up_acc += simd_shuffle_xor(up_acc, 4);

    if (row_lane == 0 && global_row < gate_rows) {
        for (uint r = 0; r < 2 && global_row + r < gate_rows; r++) {
            float gate = gate_acc[r];
            float up = up_acc[r];
            if (bias_offset > 0) {
                gate += as_type<float>(weights[bias_offset + global_row + r]);
                up += as_type<float>(
                    weights[bias_offset + global_row + r + gate_rows]);
            }
            float activated = (p[3] & 4u) != 0u
                ? bn_reference_silu(gate)
                : bn_fast_silu(gate);
            out[global_row + r] = activated * up;
        }
    }
}

#undef UQ4
