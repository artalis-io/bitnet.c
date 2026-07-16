#include "transformer_kv_internal.h"
#include "transformer_cpu_features_internal.h"
#include "transformer_gqa_internal.h"
#include "quant.h"

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2
static void kv_write_fp16_scalar(uint16_t *kc,
                                 uint16_t *vc,
                                 const float *k_tmp,
                                 const float *v_tmp,
                                 int kv_dim) {
    for (int i = 0; i < kv_dim; i++) {
        kc[i] = bn_fp32_to_fp16(k_tmp[i]);
        vc[i] = bn_fp32_to_fp16(v_tmp[i]);
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static void kv_write_fp16_neon(uint16_t *kc,
                               uint16_t *vc,
                               const float *k_tmp,
                               const float *v_tmp,
                               int kv_dim) {
    for (int i = 0; i < kv_dim; i += 4) {
        vst1_u16(kc + i,
                 vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(k_tmp + i))));
        vst1_u16(vc + i,
                 vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(v_tmp + i))));
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void kv_write_fp16_avx2(uint16_t *kc,
                               uint16_t *vc,
                               const float *k_tmp,
                               const float *v_tmp,
                               int kv_dim) {
    for (int i = 0; i < kv_dim; i += 8) {
        _mm_storeu_si128((__m128i *)(kc + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(k_tmp + i),
                                         _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128((__m128i *)(vc + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(v_tmp + i),
                                         _MM_FROUND_TO_NEAREST_INT));
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static const BnKVCPUOps BN_KV_CPU_OPS = {
    "neon",
    bn_transformer_gqa_tq_neon_range,
    kv_write_fp16_neon,
};
#elif BN_TRANSFORMER_CPU_HAS_AVX512
static const BnKVCPUOps BN_KV_CPU_OPS = {
    "avx512",
    bn_transformer_gqa_tq_scalar_range,
    kv_write_fp16_avx2,
};
#elif BN_TRANSFORMER_CPU_HAS_AVX2
static const BnKVCPUOps BN_KV_CPU_OPS = {
    "avx2",
    bn_transformer_gqa_tq_scalar_range,
    kv_write_fp16_avx2,
};
#else
static const BnKVCPUOps BN_KV_CPU_OPS = {
    "scalar",
    bn_transformer_gqa_tq_scalar_range,
    kv_write_fp16_scalar,
};
#endif

const BnKVCPUOps *bn_transformer_kv_cpu_ops(void) {
    return &BN_KV_CPU_OPS;
}
