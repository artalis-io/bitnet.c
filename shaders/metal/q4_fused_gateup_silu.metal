#include <metal_stdlib>
using namespace metal;

// Q4_0 REPACKED fused gate-up SiLU — float4 vectorized, 32 rows/tile, 8 threads/row
//
// Computes: out[i] = silu(gate[i] . x) * up[i] . x
// where silu(g) = g / (1 + exp(-g))
//
// Gate and up weights are stacked in a single Q4_0 repacked buffer:
//   rows 0..gate_rows-1           = gate weights
//   rows gate_rows..total_rows-1  = up weights
//
// GPU buffer layout: [f32 scales: n_blocks][nibble u32s: n_blocks * 4]
// where n_blocks = total_rows * blocks_per_row
//
// 8 threads per row, each processes blocks_per_row/8 blocks.
// Each block: 8 dot(float4) operations.
// float4 loads for x vector (coalesced 16-byte reads).
// Reduction: simd_shuffle_xor (0 barriers).
//
// Dispatch: (ceil(gate_rows/32), 1, 1)

// Dequantize 4 consecutive nibbles from a u32 word at bit offset sh
#define DQ4(w, sh, s) (s * float4( \
    float(int(((w) >> (sh))       & 0xF) - 8), \
    float(int(((w) >> ((sh) + 4)) & 0xF) - 8), \
    float(int(((w) >> ((sh) + 8)) & 0xF) - 8), \
    float(int(((w) >> ((sh) + 12))& 0xF) - 8)))

kernel void q4_fused_gateup_silu(device const uint  *weights [[buffer(0)]],
                                  device const float *x       [[buffer(1)]],
                                  device float       *out     [[buffer(2)]],
                                  constant uint      *p       [[buffer(3)]],
                                  uint3 wid [[threadgroup_position_in_grid]],
                                  uint3 lid [[thread_position_in_threadgroup]]) {
    uint total_rows = p[0], cols = p[1], gate_rows = p[2];
    uint bias_offset = p[4];
    uint tile_start = wid.x * 32;

    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint blocks_per_row = cols >> 5;
    uint total_blocks = total_rows * blocks_per_row;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    if (global_row < gate_rows) {
        // Gate weight: row = global_row
        uint gate_block_base = global_row * blocks_per_row;
        // Up weight: row = global_row + gate_rows
        uint up_block_base = (global_row + gate_rows) * blocks_per_row;

        for (uint b = row_lane; b < blocks_per_row; b += 8) {
            // --- Gate block ---
            uint g_idx = gate_block_base + b;
            float gs = as_type<float>(weights[g_idx]);

            uint g_nib = total_blocks + g_idx * 4;
            uint gw0 = weights[g_nib];
            uint gw1 = weights[g_nib + 1];
            uint gw2 = weights[g_nib + 2];
            uint gw3 = weights[g_nib + 3];

            // --- Up block ---
            uint u_idx = up_block_base + b;
            float us = as_type<float>(weights[u_idx]);

            uint u_nib = total_blocks + u_idx * 4;
            uint uw0 = weights[u_nib];
            uint uw1 = weights[u_nib + 1];
            uint uw2 = weights[u_nib + 2];
            uint uw3 = weights[u_nib + 3];

            // float4 loads — coalesced 16-byte reads (shared between gate and up)
            device const float4 *xp = (device const float4 *)(x + b * 32);

            // Gate: 8 vectorized dot products
            gate_acc += dot(DQ4(gw0,  0, gs), xp[0]);
            gate_acc += dot(DQ4(gw0, 16, gs), xp[1]);
            gate_acc += dot(DQ4(gw1,  0, gs), xp[2]);
            gate_acc += dot(DQ4(gw1, 16, gs), xp[3]);
            gate_acc += dot(DQ4(gw2,  0, gs), xp[4]);
            gate_acc += dot(DQ4(gw2, 16, gs), xp[5]);
            gate_acc += dot(DQ4(gw3,  0, gs), xp[6]);
            gate_acc += dot(DQ4(gw3, 16, gs), xp[7]);

            // Up: 8 vectorized dot products
            up_acc += dot(DQ4(uw0,  0, us), xp[0]);
            up_acc += dot(DQ4(uw0, 16, us), xp[1]);
            up_acc += dot(DQ4(uw1,  0, us), xp[2]);
            up_acc += dot(DQ4(uw1, 16, us), xp[3]);
            up_acc += dot(DQ4(uw2,  0, us), xp[4]);
            up_acc += dot(DQ4(uw2, 16, us), xp[5]);
            up_acc += dot(DQ4(uw3,  0, us), xp[6]);
            up_acc += dot(DQ4(uw3, 16, us), xp[7]);
        }
    }

    // Reduce gate accumulator across 8 threads in the row
    gate_acc += simd_shuffle_xor(gate_acc, 1);
    gate_acc += simd_shuffle_xor(gate_acc, 2);
    gate_acc += simd_shuffle_xor(gate_acc, 4);

    // Reduce up accumulator across 8 threads in the row
    up_acc += simd_shuffle_xor(up_acc, 1);
    up_acc += simd_shuffle_xor(up_acc, 2);
    up_acc += simd_shuffle_xor(up_acc, 4);

    if (row_lane == 0 && global_row < gate_rows) {
        float g = gate_acc;
        float u = up_acc;
        if (bias_offset > 0) {
            g += as_type<float>(weights[bias_offset + global_row]);
            u += as_type<float>(weights[bias_offset + global_row + gate_rows]);
        }
        // SiLU(gate) * up
        float silu_g = g / (1.0f + exp(-g));
        out[global_row] = silu_g * u;
    }
}

#undef DQ4
