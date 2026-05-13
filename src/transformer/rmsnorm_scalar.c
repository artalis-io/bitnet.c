#include "transformer_rmsnorm_internal.h"
#include <math.h>

void bn_transformer_rmsnorm_scalar(float *out, const float *x, const float *w, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
}
