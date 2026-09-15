#include "transformer_rmsnorm_internal.h"
#include "transformer_cpu_features_internal.h"
#include <math.h>

void bn_transformer_rmsnorm_reference(float *out, const float *x,
    const float *w, int size, float eps) {
    if (size <= 0) return;
    double ss = 0.0;
    for (int i = 0; i < size; i++)
        ss += (double)(x[i] * x[i]);
    float scale = 1.0f / sqrtf((float)(ss / (double)size) + eps);
    for (int i = 0; i < size; i++)
        out[i] = x[i] * scale * w[i];
}

void bn_transformer_rmsnorm_default(float *out,
                                    const float *x,
                                    const float *w,
                                    int size,
                                    float eps) {
#if BN_TRANSFORMER_CPU_HAS_NEON
    bn_transformer_rmsnorm_neon(out, x, w, size, eps);
#elif BN_TRANSFORMER_CPU_HAS_AVX2
    bn_transformer_rmsnorm_avx2(out, x, w, size, eps);
#elif BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
    bn_transformer_rmsnorm_wasm(out, x, w, size, eps);
#else
    bn_transformer_rmsnorm_scalar(out, x, w, size, eps);
#endif
}
