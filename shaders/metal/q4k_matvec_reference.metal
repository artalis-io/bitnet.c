#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;
constant uint BLOCK_BYTES = 144;

static inline void q4k_scale_min(device const uchar *scales, uint j,
                                 thread uchar &sc, thread uchar &mn) {
    if (j < 4) {
        sc = scales[j] & 63;
        mn = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4);
        mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

static inline void q4k_acc_16(thread float4 &acc0,
                              thread float4 &acc1,
                              thread float4 &acc2,
                              thread float4 &acc3,
                              device const uchar *packed,
                              device const float *x,
                              float ds, float dm, bool high) {
    float4 w0, w1, w2, w3;
    for (uint i = 0; i < 4; i++) {
        uint shift = high ? 4 : 0;
        w0[i] = float((packed[i] >> shift) & 15);
        w1[i] = float((packed[i + 4] >> shift) & 15);
        w2[i] = float((packed[i + 8] >> shift) & 15);
        w3[i] = float((packed[i + 12] >> shift) & 15);
    }
    acc0 = fma(w0 * ds - dm, float4(x[0], x[1], x[2], x[3]), acc0);
    acc1 = fma(w1 * ds - dm, float4(x[4], x[5], x[6], x[7]), acc1);
    acc2 = fma(w2 * ds - dm, float4(x[8], x[9], x[10], x[11]), acc2);
    acc3 = fma(w3 * ds - dm,
               float4(x[12], x[13], x[14], x[15]), acc3);
}

kernel void q4k_matvec_reference(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3];
    uint tile = extra > 0 ? wid.x + wid.y * extra : wid.x;
    uint token = extra > 0 ? 0 : wid.y;
    uint row = tile * 256 + lid.x;
    if (row >= rows) return;

    uint n_blocks = cols / QK_K;
    device const uchar *block =
        weights + row * n_blocks * BLOCK_BYTES;
    device const float *xb = x + token * cols;
    float row_sum = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        float d = float(*(device const half *)(block + 0));
        float dmin = float(*(device const half *)(block + 2));
        device const uchar *scales = block + 4;
        device const uchar *qs = block + 16;
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f), acc3 = float4(0.0f);
        for (uint j = 0; j < QK_K; j += 64) {
            uint sub = j / 32;
            uchar sc, mn;
            q4k_scale_min(scales, sub, sc, mn);
            q4k_acc_16(acc0, acc1, acc2, acc3, qs, xb + j,
                       d * float(sc), dmin * float(mn), false);
            q4k_acc_16(acc0, acc1, acc2, acc3, qs + 16, xb + j + 16,
                       d * float(sc), dmin * float(mn), false);
            q4k_scale_min(scales, sub + 1, sc, mn);
            q4k_acc_16(acc0, acc1, acc2, acc3, qs, xb + j + 32,
                       d * float(sc), dmin * float(mn), true);
            q4k_acc_16(acc0, acc1, acc2, acc3, qs + 16, xb + j + 48,
                       d * float(sc), dmin * float(mn), true);
            qs += 32;
        }
        float4 sum = (acc0 + acc1) + (acc2 + acc3);
        row_sum += (sum.x + sum.z) + (sum.y + sum.w);
        block += BLOCK_BYTES;
        xb += QK_K;
    }
    out[p[5] + token * rows + row] = row_sum;
}
