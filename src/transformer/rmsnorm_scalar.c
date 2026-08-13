#include "transformer_rmsnorm_internal.h"
#include <math.h>

void bn_transformer_rmsnorm_scalar(float *out, const float *x, const float *w, int size, float eps) {
    double sum = 0.0;
    for (int i = 0; i < size; i++)
        sum += (double)(x[i] * x[i]);

    float mean = (float)(sum / size);
    float scale = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < size; i++)
        out[i] = x[i] * scale * w[i];
}
