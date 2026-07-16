#include "quant_ctx.h"
#include <math.h>
#include <string.h>

float bn_quant_x_to_i8(const float *x, int8_t *x_q, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) {
        memset(x_q, 0, (size_t)n);
        return 0.0f;
    }
    float scale = amax / (float)BN_I8_MAX;
    float inv_scale = (float)BN_I8_MAX / amax;
    for (int i = 0; i < n; i++) {
        int v = (int)roundf(x[i] * inv_scale);
        if (v < -BN_I8_MAX) v = -BN_I8_MAX;
        if (v > BN_I8_MAX) v = BN_I8_MAX;
        x_q[i] = (int8_t)v;
    }
    return scale;
}

void bn_quant_x_to_q8_blocks(const float *x, int8_t *x_q, float *x_scales,
                             int n) {
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + (size_t)b * 32;
        int8_t *qb = x_q + (size_t)b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float a = fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) {
            memset(qb, 0, 32);
            x_scales[b] = 0.0f;
            continue;
        }
        float inv_scale = 127.0f / amax;
        x_scales[b] = bn_fp16_to_fp32(bn_fp32_to_fp16(amax / 127.0f));
        for (int i = 0; i < 32; i++) {
            int v = (int)lrintf(xb[i] * inv_scale);
            if (v < -127) v = -127;
            if (v > 127) v = 127;
            qb[i] = (int8_t)v;
        }
    }
}
