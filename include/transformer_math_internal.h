#ifndef BN_TRANSFORMER_MATH_INTERNAL_H
#define BN_TRANSFORMER_MATH_INTERNAL_H

#include <math.h>

// Max elements for stack VLAs in backend range functions.
// Prevents stack overflow from malicious model configs.
#ifndef BN_MAX_VLA_ELEMS
#define BN_MAX_VLA_ELEMS 8192
#endif

static inline void bn_transformer_softmax(float *x, int size) {
    if (size <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

#endif // BN_TRANSFORMER_MATH_INTERNAL_H
