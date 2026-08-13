#include "transformer_math_internal.h"
#include <math.h>
#include <stdint.h>

float bn_transformer_fast_exp_scalar(float x) {
    const float r = 0x1.8p23f;
    float z = fmaf(x, 0x1.715476p+0f, r);
    float n = z - r;
    float b = fmaf(-n, 0x1.7f7d1cp-20f,
                   fmaf(-n, 0x1.62e4p-1f, x));
    union {
        float f;
        uint32_t u;
    } bits = {z};
    uint32_t e = bits.u << 23;
    bits.u = e + UINT32_C(0x3f800000);
    float k = bits.f;
    float u = b * b;
    float j = fmaf(0x1.573e2ep-5f, 1.0f,
                   0x1.0e4020p-7f * b);
    j = fmaf(j, u,
             fmaf(0x1.555e66p-3f, b, 0x1.fffdb6p-2f));
    j = fmaf(j, u, 0x1.ffffecp-1f * b);

    if (fabsf(n) <= 126.0f)
        return fmaf(k, j, k);

    uint32_t d = n <= 0.0f ? UINT32_C(0x82000000) : 0;
    bits.u = d + UINT32_C(0x7f000000);
    float s1 = bits.f;
    bits.u = e - d;
    float s2 = bits.f;
    if (fabsf(n) > 192.0f)
        return s1 * s1;
    return fmaf(s2, j, s2) * s1;
}

void bn_transformer_softmax_scalar(float *x, int size) {
    if (size <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    double sum = 0.0;
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float e0 = bn_transformer_fast_exp_scalar(x[i] - max_val);
        float e1 = bn_transformer_fast_exp_scalar(x[i + 1] - max_val);
        float e2 = bn_transformer_fast_exp_scalar(x[i + 2] - max_val);
        float e3 = bn_transformer_fast_exp_scalar(x[i + 3] - max_val);
        x[i] = e0;
        x[i + 1] = e1;
        x[i + 2] = e2;
        x[i + 3] = e3;
        sum += (double)((e0 + e1) + (e2 + e3));
    }
    if (i < size) {
        float tail[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int n = size - i;
        for (int j = 0; j < n; j++) {
            tail[j] = bn_transformer_fast_exp_scalar(x[i + j] - max_val);
            x[i + j] = tail[j];
        }
        sum += (double)((tail[0] + tail[1]) + (tail[2] + tail[3]));
    }
    float inv = (float)(1.0 / sum);
    for (i = 0; i < size; i++) x[i] *= inv;
}
