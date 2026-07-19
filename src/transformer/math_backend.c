#include "transformer_math_internal.h"
#include "transformer_cpu_features_internal.h"

void bn_transformer_softmax(float *x, int size) {
#if BN_TRANSFORMER_CPU_HAS_NEON
    bn_transformer_softmax_neon(x, size);
#else
    bn_transformer_softmax_scalar(x, size);
#endif
}
