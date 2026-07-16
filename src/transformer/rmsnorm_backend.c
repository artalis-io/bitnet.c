#include "transformer_rmsnorm_internal.h"
#include "transformer_cpu_features_internal.h"

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
