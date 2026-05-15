#include "transformer_rmsnorm_internal.h"
#include <math.h>

void bn_transformer_rmsnorm_scalar(float *out, const float *x, const float *w, int size, float eps) {
    float lane[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int i = 0;
    for (; i + 7 < size; i += 8) {
        lane[0] = fmaf(x[i + 0], x[i + 0], lane[0]);
        lane[1] = fmaf(x[i + 1], x[i + 1], lane[1]);
        lane[2] = fmaf(x[i + 2], x[i + 2], lane[2]);
        lane[3] = fmaf(x[i + 3], x[i + 3], lane[3]);
        lane[4] = fmaf(x[i + 4], x[i + 4], lane[4]);
        lane[5] = fmaf(x[i + 5], x[i + 5], lane[5]);
        lane[6] = fmaf(x[i + 6], x[i + 6], lane[6]);
        lane[7] = fmaf(x[i + 7], x[i + 7], lane[7]);
        out[i + 0] = x[i + 0] * w[i + 0];
        out[i + 1] = x[i + 1] * w[i + 1];
        out[i + 2] = x[i + 2] * w[i + 2];
        out[i + 3] = x[i + 3] * w[i + 3];
        out[i + 4] = x[i + 4] * w[i + 4];
        out[i + 5] = x[i + 5] * w[i + 5];
        out[i + 6] = x[i + 6] * w[i + 6];
        out[i + 7] = x[i + 7] * w[i + 7];
    }
    float s04 = lane[0] + lane[4];
    float s15 = lane[1] + lane[5];
    float s26 = lane[2] + lane[6];
    float s37 = lane[3] + lane[7];
    float ss = (s04 + s15) + (s26 + s37);
    for (; i < size; i++) {
        ss = fmaf(x[i], x[i], ss);
        out[i] = x[i] * w[i];
    }
    ss = 1.0f / sqrtf(ss / size + eps);
    for (i = 0; i < size; i++) out[i] *= ss;
}
