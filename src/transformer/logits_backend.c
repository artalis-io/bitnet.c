#include "transformer_logits_internal.h"
#include "transformer_cpu_features_internal.h"
#include "transformer_rmsnorm_internal.h"

#if BN_TRANSFORMER_CPU_HAS_NEON_FP16_ARITH
static void logits_prepare_f16_x_neon(uint16_t *dst,
                                      const float *src,
                                      int dim) {
    for (int d = 0; d < dim; d += 8) {
        float16x4_t lo = vcvt_f16_f32(vld1q_f32(src + d));
        float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + d + 4));
        vst1q_u16(dst + d, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static const BnLogitsBackendOps BN_LOGITS_BACKEND = {
    bn_transformer_rmsnorm_neon,
#if BN_TRANSFORMER_CPU_HAS_NEON_DOTPROD
    bn_transformer_logits_i8_neon_range,
    1,
    1,
#else
    bn_transformer_logits_i8_scalar_range,
    0,
    0,
#endif
#if BN_TRANSFORMER_CPU_HAS_NEON_FP16_ARITH
    bn_transformer_logits_f16_native_neon_range,
    logits_prepare_f16_x_neon,
#else
    bn_transformer_logits_f16_neon_range,
    NULL,
#endif
};
#elif BN_TRANSFORMER_CPU_HAS_AVX512
static const BnLogitsBackendOps BN_LOGITS_BACKEND = {
    bn_transformer_rmsnorm_avx2,
    bn_transformer_logits_i8_avx2_range,
    1,
    1,
    bn_transformer_logits_f16_avx2_range,
    NULL,
};
#elif BN_TRANSFORMER_CPU_HAS_AVX2
static const BnLogitsBackendOps BN_LOGITS_BACKEND = {
    bn_transformer_rmsnorm_avx2,
    bn_transformer_logits_i8_avx2_range,
    1,
    1,
    bn_transformer_logits_f16_avx2_range,
    NULL,
};
#elif BN_TRANSFORMER_CPU_HAS_WASM_RELAXED_SIMD
static const BnLogitsBackendOps BN_LOGITS_BACKEND = {
    bn_transformer_rmsnorm_wasm,
    bn_transformer_logits_i8_wasm_range,
    1,
    1,
    bn_transformer_logits_f16_wasm_range,
    NULL,
};
#elif BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static const BnLogitsBackendOps BN_LOGITS_BACKEND = {
    bn_transformer_rmsnorm_wasm,
    bn_transformer_logits_i8_scalar_range,
    0,
    0,
    bn_transformer_logits_f16_wasm_range,
    NULL,
};
#else
static const BnLogitsBackendOps BN_LOGITS_BACKEND = {
    bn_transformer_rmsnorm_scalar,
    bn_transformer_logits_i8_scalar_range,
    0,
    0,
    bn_transformer_logits_f16_scalar_range,
    NULL,
};
#endif

const BnLogitsBackendOps *bn_transformer_logits_backend_ops(void) {
    return &BN_LOGITS_BACKEND;
}
