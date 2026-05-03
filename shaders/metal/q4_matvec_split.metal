#include <metal_stdlib>
using namespace metal;

// Q4_0 NATIVE split matvec — multi-output, reads GGUF format (18B/block)
// Routes output rows to up to 3 buffers based on split points.
// Pre-scaling trick for zero-shift nibble extraction.
// Dispatch: (ceil(total_rows/32), 1, 1)

kernel void q4_matvec_split(device const char  *weights [[buffer(0)]],
                             device const float *x       [[buffer(1)]],
                             device float       *out0    [[buffer(2)]],
                             device float       *out1    [[buffer(3)]],
                             device float       *out2    [[buffer(4)]],
                             constant uint      *p       [[buffer(5)]],
                             uint3 wid [[threadgroup_position_in_grid]],
                             uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1];
    uint split1 = p[2], split2 = p[3];
    uint off0 = p[5], off1 = p[6], off2 = p[7];

    uint tile_start = wid.x * 32;
    uint row_lane = lid.x & 7;
    uint local_row = lid.x >> 3;
    uint global_row = tile_start + local_row;

    uint nb = cols >> 5;
    float acc = 0.0f;

    if (global_row < rows) {
        device const char *row_data = weights + (size_t)global_row * nb * 18;

        for (uint b = row_lane; b < nb; b += 8) {
            device const char *block = row_data + (size_t)b * 18;
            float d = float(*(device const half *)block);
            device const ushort *qs0 = (device const ushort *)(block + 2);
            device const ushort *qs1 = (device const ushort *)(block + 2 + 8);
            uint eb = b * 32;

            float sumy = 0.f;
            float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (ushort i = 0; i < 8; i += 2) {
                float y0=x[eb+i], y1=x[eb+i+1], y16=x[eb+i+16], y17=x[eb+i+17];
                sumy += y0 + y1 + y16 + y17;
                ushort w = qs0[i/2];
                a[0] += y0         * float(w & 0x000Fu);
                a[1] += (y1/256)   * float(w & 0x0F00u);
                a[2] += (y16/16)   * float(w & 0x00F0u);
                a[3] += (y17/4096) * float(w & 0xF000u);

                float y8=x[eb+8+i], y9=x[eb+8+i+1], y24=x[eb+24+i], y25=x[eb+24+i+1];
                sumy += y8 + y9 + y24 + y25;
                ushort w2 = qs1[i/2];
                a[4] += y8         * float(w2 & 0x000Fu);
                a[5] += (y9/256)   * float(w2 & 0x0F00u);
                a[6] += (y24/16)   * float(w2 & 0x00F0u);
                a[7] += (y25/4096) * float(w2 & 0xF000u);
            }
            acc += d * (sumy*(-8.f) + a[0]+a[1]+a[2]+a[3]+a[4]+a[5]+a[6]+a[7]);
        }
    }

    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);

    if (row_lane == 0 && global_row < rows) {
        if (split1 > 0 && global_row >= split1) {
            if (split2 > 0 && global_row >= split2)
                out2[off2 + global_row - split2] = acc;
            else
                out1[off1 + global_row - split1] = acc;
        } else {
            out0[off0 + global_row] = acc;
        }
    }
}
