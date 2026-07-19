#ifndef BN_TRANSFORMER_MATH_INTERNAL_H
#define BN_TRANSFORMER_MATH_INTERNAL_H

#include <math.h>

// Max elements for stack VLAs in backend range functions.
// Prevents stack overflow from malicious model configs.
#ifndef BN_MAX_VLA_ELEMS
#define BN_MAX_VLA_ELEMS 8192
#endif

void bn_transformer_softmax(float *x, int size);
void bn_transformer_softmax_scalar(float *x, int size);
void bn_transformer_softmax_neon(float *x, int size);

#endif // BN_TRANSFORMER_MATH_INTERNAL_H
