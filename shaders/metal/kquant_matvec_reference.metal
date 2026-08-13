#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;

static inline uint2 q5k_scale_min(uint j,
                                  device const uchar *scales) {
    if (j < 4)
        return uint2(scales[j] & 63, scales[j + 4] & 63);
    return uint2((scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4),
                 (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
}

kernel void q5k_matvec_reference(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3];
    uint tile = extra > 0 ? wid.x + wid.y * extra : wid.x;
    uint token = extra > 0 ? 0 : wid.y;
    uint row = tile * 256u + lid3.x;
    if (row >= rows) return;
    uint blocks = cols / QK_K;
    float row_sum = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        device const uchar *block =
            weights + (row * blocks + b) * 176u;
        float d = float(*(device const half *)(block));
        float dmin = float(*(device const half *)(block + 2));
        device const uchar *scales = block + 4;
        device const uchar *qh = block + 16;
        device const uchar *qs = block + 48;
        device const float *xb = x + token * cols + b * QK_K;
        for (uint j = 0; j < QK_K; j += 64) {
            uint sub = j / 32;
            uint group = j / 64;
            uint bit_lo = group * 2;
            uint bit_hi = bit_lo + 1;
            uint2 sm = q5k_scale_min(sub, scales);
            float ds = d * float(sm.x);
            float dm = dmin * float(sm.y);
            for (uint l = 0; l < 32; l++) {
                uint q5 = uint(qs[l] & 15) |
                          (uint((qh[l] >> bit_lo) & 1) << 4);
                row_sum += (ds * float(q5) - dm) * xb[j + l];
            }
            sm = q5k_scale_min(sub + 1, scales);
            ds = d * float(sm.x);
            dm = dmin * float(sm.y);
            for (uint l = 0; l < 32; l++) {
                uint q5 = uint(qs[l] >> 4) |
                          (uint((qh[l] >> bit_hi) & 1) << 4);
                row_sum += (ds * float(q5) - dm) * xb[j + l + 32];
            }
            qs += 32;
        }
    }
    out[p[5] + token * rows + row] = row_sum;
}

// Scalar-order Q6_K matvec for numerically sensitive graph operations.
kernel void q6k_matvec_reference(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3];
    uint tile = extra > 0 ? wid.x + wid.y * extra : wid.x;
    uint token = extra > 0 ? 0 : wid.y;
    uint row = tile * 256u + lid3.x;
    if (row >= rows) return;
    uint blocks = cols / QK_K;
    float row_sum = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        device const uchar *block =
            weights + (row * blocks + b) * 210u;
        float d = float(*(device const half *)(block + 208));
        device const uchar *ql = block;
        device const uchar *qh = block + 128;
        device const char *sc = (device const char *)(block + 192);
        device const float *xb = x + token * cols + b * QK_K;
        for (uint n = 0; n < QK_K; n += 128) {
            for (uint is = 0; is < 2; is++) {
                float sum1 = 0.0f, sum2 = 0.0f;
                float sum3 = 0.0f, sum4 = 0.0f;
                uint l0 = is * 16;
                for (uint i = 0; i < 16; i++) {
                    uint l = l0 + i;
                    uchar h = qh[l];
                    int q1 = int((ql[l] & 15) | ((h & 3) << 4)) - 32;
                    int q2 = int((ql[l + 32] & 15) |
                                 (((h >> 2) & 3) << 4)) - 32;
                    int q3 = int((ql[l] >> 4) |
                                 (((h >> 4) & 3) << 4)) - 32;
                    int q4 = int((ql[l + 32] >> 4) |
                                 (((h >> 6) & 3) << 4)) - 32;
                    sum1 += float(q1) * xb[l];
                    sum2 += float(q2) * xb[l + 32];
                    sum3 += float(q3) * xb[l + 64];
                    sum4 += float(q4) * xb[l + 96];
                }
                row_sum += d * (float(sc[is]) * sum1 +
                                float(sc[is + 2]) * sum2 +
                                float(sc[is + 4]) * sum3 +
                                float(sc[is + 6]) * sum4);
            }
            xb += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    out[p[5] + token * rows + row] = row_sum;
}
