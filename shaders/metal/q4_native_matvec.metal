#include <metal_stdlib>
using namespace metal;

// Q4_0 NATIVE matvec — reads original GGUF format (18 bytes/block)
// Same 32 rows/tile, 8 threads/row layout as original q4_matvec.
// Each thread processes all 32 elements of its assigned blocks.
//
// GGUF Q4_0 block: [FP16 scale (2B)][16 packed nibble bytes]
// qs[j] = (elem[j] & 0xF) | (elem[j+16] << 4)
//
// Dispatch: (ceil(rows/32), n_tokens, 1)

kernel void q4_native_matvec(
    device const char  *weights [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device float       *out     [[buffer(2)]],
    constant uint      *p       [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * 32 : wid.x * 32;
    uint token = (extra > 0) ? 0 : wid.y;

    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint nb = cols >> 5;  // blocks per row
    uint x_base = token * cols;

    float acc = 0.0f;

    if (global_row < rows) {
        device const char *row_data = weights + (size_t)global_row * nb * 18;

        for (uint b = row_lane; b < nb; b += 8) {
            device const char *block = row_data + (size_t)b * 18;
            float d = float(*(device const half *)block);
            device const uchar *qs = (device const uchar *)(block + 2);

            float sum = 0.0f;
            uint elem_base = x_base + b * 32;

            // Process 32 elements: qs[j] lower=elem[j], upper=elem[j+16]
            for (uint j = 0; j < 16; j++) {
                uchar qb = qs[j];
                int lo = (int)(qb & 0xF) - 8;
                int hi = (int)(qb >> 4) - 8;
                sum += (float)lo * x[elem_base + j];
                sum += (float)hi * x[elem_base + j + 16];
            }

            acc += d * sum;
        }
    }

    // Reduction across 8 threads in row
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows)
        out[out_offset + token * rows + global_row] = acc;
}
