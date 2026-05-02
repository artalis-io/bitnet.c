#include <metal_stdlib>
using namespace metal;

// Q4_0 NATIVE matvec — reads original GGUF format (18 bytes/block)
//
// GGUF Q4_0 block: [FP16 scale (2B)][16 packed nibble bytes]
// Nibble packing: qs[j] = (elem[j] & 0xF) | (elem[j+16] << 4)
//
// Pre-scaling trick: pre-divide input vector by position-dependent factors
// to absorb nibble bit positions from uint16 masking. No shift instructions.
// Bias correction: d * (sumy * -8 + acc) per block.
//
// NR=4 rows per SIMD group, NQ=16 blocks per SIMD group per iteration.
// Adapts to threadgroup size via simdgroups_per_threadgroup.
// Reduction: simd_sum (0 barriers).
//
// Dispatch: (ceil(rows/tile_rows), n_tokens, 1) where tile_rows = n_sg * 4

constant uint NR = 4;
constant uint NQ = 16;

kernel void q4_native_matvec(
    device const char  *weights [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device float       *out     [[buffer(2)]],
    constant uint      *p       [[buffer(3)]],
    uint3  wid   [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort ntg   [[simdgroups_per_threadgroup]])
{
    uint rows = p[0], cols = p[1], extra = p[3], out_offset = p[5];
    uint tile_rows = ntg * NR;
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * tile_rows : wid.x * tile_rows;
    uint token = (extra > 0) ? 0u : wid.y;

    uint r0 = tile_start + sgitg * NR;
    uint nb = cols >> 5;  // blocks per row

    // 2 threads per block: ix = block index (0..15), il = byte offset (0 or 8)
    ushort ix = tiisg / 2;
    ushort il = (tiisg % 2) * 8;  // byte offset into nibble data: 0 or 8

    // Weight row pointers
    device const char *ax[NR];
    for (uint row = 0; row < NR; row++)
        ax[row] = weights + (size_t)(r0 + row) * nb * 18;

    float sumf[NR] = {0.f, 0.f, 0.f, 0.f};

    device const float *yb = x + token * cols + (uint)ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        float sumy = 0.f;
        float yl[16];

        // Cache input vector with pre-scaling
        for (ushort i = 0; i < 8; i += 2) {
            sumy     += yb[i + 0] + yb[i + 1] + yb[i + 16] + yb[i + 17];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] / 256.f;
            yl[i + 8] = yb[i + 16] / 16.f;
            yl[i + 9] = yb[i + 17] / 4096.f;
        }

        for (uint row = 0; row < NR; row++) {
            if (r0 + row >= rows) break;

            device const char *block = ax[row] + (size_t)ib * 18;
            float d = float(*(device const half *)block);

            // Read packed nibbles as uint16 — use il as BYTE offset (0 or 8)
            device const ushort *qs = (device const ushort *)(block + 2 + il);

            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            for (ushort i = 0; i < 8; i += 2) {
                acc[0] += yl[i + 0] * float(qs[i / 2] & 0x000Fu);
                acc[1] += yl[i + 1] * float(qs[i / 2] & 0x0F00u);
                acc[2] += yl[i + 8] * float(qs[i / 2] & 0x00F0u);
                acc[3] += yl[i + 9] * float(qs[i / 2] & 0xF000u);
            }

            sumf[row] += d * (sumy * (-8.f) + acc[0] + acc[1] + acc[2] + acc[3]);
        }

        yb += 32 * NQ;  // advance to next block set
    }

    // Reduce across SIMD group and write
    for (uint row = 0; row < NR; row++) {
        float val = simd_sum(sumf[row]);
        uint global_row = r0 + row;
        if (tiisg == 0 && global_row < rows)
            out[out_offset + token * rows + global_row] = val;
    }
}
