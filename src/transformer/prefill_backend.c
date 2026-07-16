#include "transformer_prefill_internal.h"
#include "transformer_cpu_features_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "quant.h"
#include "simd_helpers.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static float prefill_gelu(float x) {
    return 0.5f * x *
           (1.0f + tanhf(0.7978845608028654f * x *
                         (1.0f + 0.044715f * x * x)));
}

static void prefill_ffn_activation_scalar_range(void *ctx, int start, int end) {
    BnPrefillFFNActCtx *c = (BnPrefillFFNActCtx *)ctx;
    int hidden_dim = c->hidden_dim;
    for (int t = start; t < end; t++) {
        float *hb_t = c->hb + (size_t)t * hidden_dim;
        const float *hb2_t = c->hb2 ? c->hb2 + (size_t)t * hidden_dim : NULL;
        if (c->act_type == 1) {
            for (int i = 0; i < hidden_dim; i++) {
                float g = hb_t[i] > 0 ? hb_t[i] : 0;
                hb_t[i] = hb2_t ? g * g * hb2_t[i] : g * g;
            }
        } else if (c->act_type == 2) {
            for (int i = 0; i < hidden_dim; i++) {
                float v = prefill_gelu(hb_t[i]);
                hb_t[i] = hb2_t ? v * hb2_t[i] : v;
            }
        } else {
            for (int i = 0; i < hidden_dim; i++) {
                float v = hb_t[i] / (1.0f + expf(-hb_t[i]));
                hb_t[i] = hb2_t ? v * hb2_t[i] : v;
            }
        }
    }
}

#if BN_TRANSFORMER_CPU_HAS_NEON
static void prefill_ffn_activation_neon_range(void *ctx, int start, int end) {
    BnPrefillFFNActCtx *c = (BnPrefillFFNActCtx *)ctx;
    int hidden_dim = c->hidden_dim;
    for (int t = start; t < end; t++) {
        float *hb_t = c->hb + (size_t)t * hidden_dim;
        const float *hb2_t = c->hb2 ? c->hb2 + (size_t)t * hidden_dim : NULL;
        int i = 0;
        if (c->act_type == 1) {
            float32x4_t zero_v = vdupq_n_f32(0.0f);
            for (; i + 3 < hidden_dim; i += 4) {
                float32x4_t g = vmaxq_f32(vld1q_f32(hb_t + i), zero_v);
                float32x4_t v = vmulq_f32(g, g);
                if (hb2_t)
                    v = vmulq_f32(v, vld1q_f32(hb2_t + i));
                vst1q_f32(hb_t + i, v);
            }
        } else if (c->act_type == 2 && c->fast_approx) {
            for (; i + 3 < hidden_dim; i += 4) {
                float32x4_t v =
                    bn_neon_fast_gelu_f32(vld1q_f32(hb_t + i));
                if (hb2_t)
                    v = vmulq_f32(v, vld1q_f32(hb2_t + i));
                vst1q_f32(hb_t + i, v);
            }
        } else if (c->act_type != 1 && c->act_type != 2 && c->fast_approx) {
            for (; i + 3 < hidden_dim; i += 4) {
                float32x4_t v =
                    bn_neon_fast_silu_f32(vld1q_f32(hb_t + i));
                if (hb2_t)
                    v = vmulq_f32(v, vld1q_f32(hb2_t + i));
                vst1q_f32(hb_t + i, v);
            }
        }
        BnPrefillFFNActCtx tail = {
            hb_t + i, hb2_t ? hb2_t + i : NULL,
            hidden_dim - i, c->act_type, c->fast_approx
        };
        prefill_ffn_activation_scalar_range(&tail, 0, 1);
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void prefill_ffn_activation_avx2_range(void *ctx, int start, int end) {
    BnPrefillFFNActCtx *c = (BnPrefillFFNActCtx *)ctx;
    int hidden_dim = c->hidden_dim;
    for (int t = start; t < end; t++) {
        float *hb_t = c->hb + (size_t)t * hidden_dim;
        const float *hb2_t = c->hb2 ? c->hb2 + (size_t)t * hidden_dim : NULL;
        int i = 0;
        if (c->act_type == 1) {
            __m256 zero_v = _mm256_setzero_ps();
            for (; i + 7 < hidden_dim; i += 8) {
                __m256 g = _mm256_max_ps(_mm256_loadu_ps(hb_t + i), zero_v);
                __m256 v = _mm256_mul_ps(g, g);
                if (hb2_t)
                    v = _mm256_mul_ps(v, _mm256_loadu_ps(hb2_t + i));
                _mm256_storeu_ps(hb_t + i, v);
            }
        } else if (c->act_type == 2 && c->fast_approx) {
            for (; i + 7 < hidden_dim; i += 8) {
                __m256 v =
                    bn_avx2_fast_gelu_ps(_mm256_loadu_ps(hb_t + i));
                if (hb2_t)
                    v = _mm256_mul_ps(v, _mm256_loadu_ps(hb2_t + i));
                _mm256_storeu_ps(hb_t + i, v);
            }
        } else if (c->act_type != 1 && c->act_type != 2 && c->fast_approx) {
            for (; i + 7 < hidden_dim; i += 8) {
                __m256 v =
                    bn_avx2_fast_silu_ps(_mm256_loadu_ps(hb_t + i));
                if (hb2_t)
                    v = _mm256_mul_ps(v, _mm256_loadu_ps(hb2_t + i));
                _mm256_storeu_ps(hb_t + i, v);
            }
        }
        BnPrefillFFNActCtx tail = {
            hb_t + i, hb2_t ? hb2_t + i : NULL,
            hidden_dim - i, c->act_type, c->fast_approx
        };
        prefill_ffn_activation_scalar_range(&tail, 0, 1);
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static int prefill_prepare_preq8k_avx2(int8_t *xq,
                                       float *xd,
                                       int16_t *xbs,
                                       int n_bpr,
                                       const float *x,
                                       int dim,
                                       int n_tokens) {
    if (!xq || !xd || !xbs || !x || dim <= 0 ||
        n_tokens <= 0 || n_bpr <= 0)
        return 0;
    for (int t = 0; t < n_tokens; t++)
        bn_quant_x_to_q8k(x + (size_t)t * dim,
                          xq + (size_t)t * dim,
                          xd + (size_t)t * n_bpr,
                          xbs + (size_t)t * n_bpr * 16, dim);
    return 1;
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static const BnPrefillCPUOps BN_PREFILL_CPU_OPS = {
    "neon",
    bn_transformer_rmsnorm_neon,
    prefill_ffn_activation_neon_range,
    bn_transformer_ssm_conv_silu_neon_range,
    bn_transformer_ssm_l2norm_neon_range,
    bn_transformer_ssm_delta_neon_range,
    bn_transformer_ssm_gate_neon_range,
    NULL,
    0,
};
#elif BN_TRANSFORMER_CPU_HAS_AVX512
static const BnPrefillCPUOps BN_PREFILL_CPU_OPS = {
    "avx512",
    bn_transformer_rmsnorm_avx2,
    prefill_ffn_activation_avx2_range,
    bn_transformer_ssm_conv_silu_avx2_range,
    bn_transformer_ssm_l2norm_avx2_range,
    bn_transformer_ssm_delta_avx2_range,
    bn_transformer_ssm_gate_avx2_range,
    prefill_prepare_preq8k_avx2,
    1,
};
#elif BN_TRANSFORMER_CPU_HAS_AVX2
static const BnPrefillCPUOps BN_PREFILL_CPU_OPS = {
    "avx2",
    bn_transformer_rmsnorm_avx2,
    prefill_ffn_activation_avx2_range,
    bn_transformer_ssm_conv_silu_avx2_range,
    bn_transformer_ssm_l2norm_avx2_range,
    bn_transformer_ssm_delta_avx2_range,
    bn_transformer_ssm_gate_avx2_range,
    prefill_prepare_preq8k_avx2,
    1,
};
#elif BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static const BnPrefillCPUOps BN_PREFILL_CPU_OPS = {
    "wasm",
    bn_transformer_rmsnorm_wasm,
    prefill_ffn_activation_scalar_range,
    bn_transformer_ssm_conv_silu_wasm_range,
    bn_transformer_ssm_l2norm_wasm_range,
    bn_transformer_ssm_delta_wasm_range,
    bn_transformer_ssm_gate_wasm_range,
    NULL,
    0,
};
#else
static const BnPrefillCPUOps BN_PREFILL_CPU_OPS = {
    "scalar",
    bn_transformer_rmsnorm_scalar,
    prefill_ffn_activation_scalar_range,
    bn_transformer_ssm_conv_silu_scalar_range,
    bn_transformer_ssm_l2norm_scalar_range,
    bn_transformer_ssm_delta_scalar_range,
    bn_transformer_ssm_gate_scalar_range,
    NULL,
    0,
};
#endif

const BnPrefillCPUOps *bn_transformer_prefill_cpu_ops(void) {
    return &BN_PREFILL_CPU_OPS;
}

int bn_transformer_prefill_profile_enabled(void) {
    return getenv("BN_PREFILL_PROFILE") != NULL;
}

int bn_transformer_prefill_hybrid_batch_allowed(void) {
    return getenv("BN_PREFILL_ALLOW_HYBRID_BATCH") != NULL;
}

int bn_transformer_prefill_force_token_attention_enabled(void) {
    return getenv("BN_PREFILL_FORCE_TOKEN_ATTN") != NULL;
}
