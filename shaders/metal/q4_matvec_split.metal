#include <metal_stdlib>
using namespace metal;

// Q4_0 REPACKED split matvec — multi-output variant of q4_matvec
//
// Routes output rows to up to 3 different buffers based on split points:
//   row < split1  → out0[off0 + row]
//   row < split2  → out1[off1 + (row - split1)]
//   row >= split2 → out2[off2 + (row - split2)]
//
// Use cases:
//   QKV batched:   split1=q_dim, split2=q_dim+kv_dim (Q→out0, K→out1, V→out2)
//   gate+up batch: split1=gate_rows, split2=0         (gate→out0, up→out1)
//   normal:        split1=0, split2=0                  (all→out0)
//
// Dispatch: (ceil(total_rows/32), 1, 1)

#define DQ4(w, sh, s) (s * float4( \
    float(int(((w) >> (sh))       & 0xF) - 8), \
    float(int(((w) >> ((sh) + 4)) & 0xF) - 8), \
    float(int(((w) >> ((sh) + 8)) & 0xF) - 8), \
    float(int(((w) >> ((sh) + 12))& 0xF) - 8)))

kernel void q4_matvec_split(device const uint  *weights [[buffer(0)]],
                             device const float *x       [[buffer(1)]],
                             device float       *out0    [[buffer(2)]],
                             device float       *out1    [[buffer(3)]],
                             device float       *out2    [[buffer(4)]],
                             constant uint      *p       [[buffer(5)]],
                             uint3 wid [[threadgroup_position_in_grid]],
                             uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1];
    uint split1 = p[2], split2 = p[3];
    uint bias_offset = p[4];
    uint off0 = p[5], off1 = p[6], off2 = p[7];

    uint tile_start = wid.x * 32;
    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint blocks_per_row = cols >> 5;
    uint total_blocks = rows * blocks_per_row;

    float acc = 0.0f;

    if (global_row < rows) {
        uint row_block_base = global_row * blocks_per_row;

        for (uint b = row_lane; b < blocks_per_row; b += 8) {
            uint block_idx = row_block_base + b;
            float s = as_type<float>(weights[block_idx]);

            uint nib_base = total_blocks + block_idx * 4;
            uint w0 = weights[nib_base];
            uint w1 = weights[nib_base + 1];
            uint w2 = weights[nib_base + 2];
            uint w3 = weights[nib_base + 3];

            device const float4 *xp = (device const float4 *)(x + b * 32);

            acc += dot(DQ4(w0,  0, s), xp[0]);
            acc += dot(DQ4(w0, 16, s), xp[1]);
            acc += dot(DQ4(w1,  0, s), xp[2]);
            acc += dot(DQ4(w1, 16, s), xp[3]);
            acc += dot(DQ4(w2,  0, s), xp[4]);
            acc += dot(DQ4(w2, 16, s), xp[5]);
            acc += dot(DQ4(w3,  0, s), xp[6]);
            acc += dot(DQ4(w3, 16, s), xp[7]);
        }
    }

    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows) {
        float result = acc;
        if (bias_offset > 0)
            result += as_type<float>(weights[bias_offset + global_row]);

        if (split1 > 0 && global_row >= split1) {
            if (split2 > 0 && global_row >= split2)
                out2[off2 + global_row - split2] = result;
            else
                out1[off1 + global_row - split1] = result;
        } else {
            out0[off0 + global_row] = result;
        }
    }
}

#undef DQ4
