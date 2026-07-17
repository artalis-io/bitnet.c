#include "transformer_cpu_backend_internal.h"
#include "transformer_cpu_features_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_batched_attn_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "quant.h"
#include "simd_helpers.h"

#include <math.h>

#if !BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_apply_sigmoid_gate_scalar(float *x, const float *gate,
                                          int size);
#endif
static void cpu_apply_rope_heads_scalar(float *buf, int n_heads,
                                        int head_size, int rope_dims,
                                        const float *rc, const float *rs);
#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2 && \
    !BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static void cpu_apply_ffn_activation_scalar(BnRunState *s,
                                            const BnFFNPlan *ffn_plan,
                                            int hidden_dim);
#endif
#if BN_TRANSFORMER_CPU_HAS_NEON
static void cpu_apply_ffn_activation_neon(BnRunState *s,
                                          const BnFFNPlan *ffn_plan,
                                          int hidden_dim);
#endif
#if BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_apply_ffn_activation_avx2(BnRunState *s,
                                          const BnFFNPlan *ffn_plan,
                                          int hidden_dim);
static void cpu_apply_sigmoid_gate_avx2(float *x, const float *gate,
                                        int size);
static void cpu_apply_rope_heads_avx2(float *buf, int n_heads,
                                      int head_size, int rope_dims,
                                      const float *rc, const float *rs);
#endif
#if BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static void cpu_apply_ffn_activation_wasm(BnRunState *s,
                                          const BnFFNPlan *ffn_plan,
                                          int hidden_dim);
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static void cpu_residual_add_neon(float *x, const float *r, int dim) {
    for (int i = 0; i < dim; i += 4)
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(r + i)));
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_residual_add_avx2(float *x, const float *r, int dim) {
    for (int i = 0; i < dim; i += 8)
        _mm256_storeu_ps(x + i,
                         _mm256_add_ps(_mm256_loadu_ps(x + i),
                                       _mm256_loadu_ps(r + i)));
}
#endif

#if BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static void cpu_residual_add_wasm(float *x, const float *r, int dim) {
    for (int i = 0; i < dim; i += 4)
        wasm_v128_store(x + i,
                        wasm_f32x4_add(wasm_v128_load(x + i),
                                       wasm_v128_load(r + i)));
}
#endif

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2 && \
    !BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static void cpu_residual_add_scalar(float *x, const float *r, int dim) {
    for (int i = 0; i < dim; i++)
        x[i] += r[i];
}
#endif

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2 && \
    !BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static float cpu_fast_exp_scalar(float x) {
    const float log2e = 1.4426950409f;
    const float ln2 = 0.6931471806f;
    if (x < -87.3f) x = -87.3f;
    if (x > 88.7f) x = 88.7f;
    float nf = floorf(fmaf(x, log2e, 0.5f));
    int ni = (int)nf;
    float r = fmaf(-nf, ln2, x);
    float poly = fmaf(0.04166664f, r, 0.16666667f);
    poly = fmaf(poly, r, 0.49999994f);
    poly = fmaf(poly, r, 1.0f);
    poly = fmaf(poly, r, 1.0f);
    union {
        uint32_t u;
        float f;
    } e2n = { (uint32_t)(ni + 127) << 23 };
    return poly * e2n.f;
}

static float cpu_fast_silu_scalar(float x) {
    return x / (1.0f + cpu_fast_exp_scalar(-x));
}

static float cpu_fast_tanh_scalar(float x) {
    return 2.0f / (1.0f + cpu_fast_exp_scalar(-2.0f * x)) - 1.0f;
}

static float cpu_fast_gelu_scalar(float x) {
    float inner = 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
    return 0.5f * x * (1.0f + cpu_fast_tanh_scalar(inner));
}
#endif

#if !BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_apply_sigmoid_gate_scalar(float *x, const float *gate,
                                          int size) {
    for (int i = 0; i < size; i++)
        x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}
#endif

static void cpu_apply_rope_heads_scalar(float *buf, int n_heads,
                                        int head_size, int rope_dims,
                                        const float *rc, const float *rs) {
    for (int h = 0; h < n_heads; h++) {
        float *hd = buf + h * head_size;
        int half_rope = rope_dims / 2;
        for (int i = 0; i < half_rope; i++) {
            int j = i + half_rope;
            float v0 = hd[i], v1 = hd[j];
            hd[i] = v0 * rc[i] - v1 * rs[i];
            hd[j] = v0 * rs[i] + v1 * rc[i];
        }
    }
}

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_apply_sigmoid_gate_avx2(float *x, const float *gate,
                                        int size) {
    for (int i = 0; i < size; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 xv = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i,
                         _mm256_mul_ps(xv, bn_avx2_fast_sigmoid_ps(g)));
    }
}

static void cpu_apply_rope_heads_avx2(float *buf, int n_heads,
                                      int head_size, int rope_dims,
                                      const float *rc, const float *rs) {
    if (rope_dims < 8) {
        cpu_apply_rope_heads_scalar(buf, n_heads, head_size, rope_dims, rc, rs);
        return;
    }
    for (int h = 0; h < n_heads; h++) {
        float *hd = buf + h * head_size;
        int half_rope = rope_dims / 2;
        int i = 0;
        for (; i + 7 < half_rope; i += 8) {
            __m256 v0 = _mm256_loadu_ps(hd + i);
            __m256 v1 = _mm256_loadu_ps(hd + half_rope + i);
            __m256 cos_v = _mm256_loadu_ps(rc + i);
            __m256 sin_v = _mm256_loadu_ps(rs + i);
            __m256 out0 = _mm256_fmsub_ps(v0, cos_v,
                                          _mm256_mul_ps(v1, sin_v));
            __m256 out1 = _mm256_fmadd_ps(v0, sin_v,
                                          _mm256_mul_ps(v1, cos_v));
            _mm256_storeu_ps(hd + i, out0);
            _mm256_storeu_ps(hd + half_rope + i, out1);
        }
        for (; i < half_rope; i++) {
            int j = i + half_rope;
            float v0 = hd[i], v1 = hd[j];
            hd[i] = v0 * rc[i] - v1 * rs[i];
            hd[j] = v0 * rs[i] + v1 * rc[i];
        }
    }
}
#endif

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2 && \
    !BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static void cpu_apply_ffn_activation_scalar(BnRunState *s,
                                            const BnFFNPlan *ffn_plan,
                                            int hidden_dim) {
    if (ffn_plan->has_gate) {
        if (ffn_plan->activation == 1) {
            for (int i = 0; i < hidden_dim; i++) {
                float g = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = g * g * s->hb2[i];
            }
        } else if (ffn_plan->activation == 2) {
            for (int i = 0; i < hidden_dim; i++) {
                float x = s->hb[i];
                float g;
                if (ffn_plan->scalar_exact_activation)
                    g = 0.5f * x *
                        (1.0f + tanhf(0.7978845608028654f * x *
                                      (1.0f + 0.044715f * x * x)));
                else
                    g = cpu_fast_gelu_scalar(x);
                s->hb[i] = g * s->hb2[i];
            }
        } else {
            for (int i = 0; i < hidden_dim; i++) {
                float g = s->hb[i];
                if (ffn_plan->scalar_exact_activation)
                    s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
                else
                    s->hb[i] = cpu_fast_silu_scalar(g) * s->hb2[i];
            }
        }
    } else {
        if (ffn_plan->activation == 1) {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = v * v;
            }
        } else if (ffn_plan->activation == 2) {
            for (int i = 0; i < hidden_dim; i++) {
                float x = s->hb[i];
                if (ffn_plan->scalar_exact_activation)
                    s->hb[i] = 0.5f * x *
                               (1.0f + tanhf(0.7978845608028654f * x *
                                             (1.0f + 0.044715f * x * x)));
                else
                    s->hb[i] = cpu_fast_gelu_scalar(x);
            }
        } else {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i];
                if (ffn_plan->scalar_exact_activation)
                    s->hb[i] = v / (1.0f + expf(-v));
                else
                    s->hb[i] = cpu_fast_silu_scalar(v);
            }
        }
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static void cpu_apply_ffn_activation_neon(BnRunState *s,
                                          const BnFFNPlan *ffn_plan,
                                          int hidden_dim) {
    if (ffn_plan->has_gate) {
        if (ffn_plan->activation == 1) {
            float32x4_t zero = vdupq_n_f32(0);
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t g = vmaxq_f32(vld1q_f32(s->hb + i), zero);
                vst1q_f32(s->hb + i, vmulq_f32(vmulq_f32(g, g),
                                                vld1q_f32(s->hb2 + i)));
            }
        } else if (ffn_plan->activation == 2) {
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t g = vld1q_f32(s->hb + i);
                float32x4_t u = vld1q_f32(s->hb2 + i);
                vst1q_f32(s->hb + i,
                          vmulq_f32(bn_neon_fast_gelu_f32(g), u));
            }
        } else if (ffn_plan->scalar_exact_activation) {
            for (int i = 0; i < hidden_dim; i++) {
                float g = s->hb[i];
                s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
            }
        } else {
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t g = vld1q_f32(s->hb + i);
                float32x4_t u = vld1q_f32(s->hb2 + i);
                vst1q_f32(s->hb + i,
                          vmulq_f32(bn_neon_fast_silu_f32(g), u));
            }
        }
    } else {
        if (ffn_plan->activation == 1) {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = v * v;
            }
        } else if (ffn_plan->activation == 2) {
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t v = vld1q_f32(s->hb + i);
                vst1q_f32(s->hb + i, bn_neon_fast_gelu_f32(v));
            }
        } else if (ffn_plan->scalar_exact_activation) {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i];
                s->hb[i] = v / (1.0f + expf(-v));
            }
        } else {
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t v = vld1q_f32(s->hb + i);
                vst1q_f32(s->hb + i, bn_neon_fast_silu_f32(v));
            }
        }
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_apply_ffn_activation_avx2(BnRunState *s,
                                          const BnFFNPlan *ffn_plan,
                                          int hidden_dim) {
    if (ffn_plan->has_gate) {
        if (ffn_plan->activation == 1) {
            __m256 zero = _mm256_setzero_ps();
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 g = _mm256_max_ps(_mm256_loadu_ps(s->hb + i), zero);
                _mm256_storeu_ps(s->hb + i,
                    _mm256_mul_ps(_mm256_mul_ps(g, g),
                                  _mm256_loadu_ps(s->hb2 + i)));
            }
        } else if (ffn_plan->activation == 2) {
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 g = _mm256_loadu_ps(s->hb + i);
                __m256 u = _mm256_loadu_ps(s->hb2 + i);
                _mm256_storeu_ps(s->hb + i,
                                 _mm256_mul_ps(bn_avx2_fast_gelu_ps(g), u));
            }
        } else {
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 g = _mm256_loadu_ps(s->hb + i);
                __m256 u = _mm256_loadu_ps(s->hb2 + i);
                _mm256_storeu_ps(s->hb + i,
                                 _mm256_mul_ps(bn_avx2_fast_silu_ps(g), u));
            }
        }
    } else {
        if (ffn_plan->activation == 1) {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = v * v;
            }
        } else if (ffn_plan->activation == 2) {
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 v = _mm256_loadu_ps(s->hb + i);
                _mm256_storeu_ps(s->hb + i, bn_avx2_fast_gelu_ps(v));
            }
        } else {
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 v = _mm256_loadu_ps(s->hb + i);
                _mm256_storeu_ps(s->hb + i, bn_avx2_fast_silu_ps(v));
            }
        }
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static void cpu_apply_ffn_activation_wasm(BnRunState *s,
                                          const BnFFNPlan *ffn_plan,
                                          int hidden_dim) {
    if (ffn_plan->has_gate && ffn_plan->activation == 1) {
        v128_t zero = wasm_f32x4_splat(0);
        for (int i = 0; i < hidden_dim; i += 4) {
            v128_t g = wasm_f32x4_max(wasm_v128_load(s->hb + i), zero);
            wasm_v128_store(s->hb + i,
                wasm_f32x4_mul(wasm_f32x4_mul(g, g),
                               wasm_v128_load(s->hb2 + i)));
        }
        return;
    }

    if (ffn_plan->has_gate) {
        for (int i = 0; i < hidden_dim; i++) {
            float x = s->hb[i];
            float g;
            if (ffn_plan->activation == 1) {
                g = x > 0 ? x * x : 0.0f;
            } else if (ffn_plan->activation == 2) {
                g = 0.5f * x *
                    (1.0f + tanhf(0.7978845608028654f * x *
                                  (1.0f + 0.044715f * x * x)));
            } else {
                g = x / (1.0f + expf(-x));
            }
            s->hb[i] = g * s->hb2[i];
        }
        return;
    }

    for (int i = 0; i < hidden_dim; i++) {
        float x = s->hb[i];
        if (ffn_plan->activation == 1) {
            float v = x > 0 ? x : 0.0f;
            s->hb[i] = v * v;
        } else if (ffn_plan->activation == 2) {
            s->hb[i] = 0.5f * x *
                       (1.0f + tanhf(0.7978845608028654f * x *
                                     (1.0f + 0.044715f * x * x)));
        } else {
            s->hb[i] = x / (1.0f + expf(-x));
        }
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static const BnCPUBackendOps BN_CPU_BACKEND = {
    "neon",
    bn_transformer_rmsnorm_neon,
    bn_transformer_gqa_neon_range,
    bn_transformer_flash_gqa_neon_range,
    bn_transformer_batched_attn_naive_neon_range,
    bn_transformer_batched_attn_flash_neon_range,
    NULL,
    cpu_residual_add_neon,
    bn_transformer_ssm_conv_silu_neon_range,
    bn_transformer_ssm_l2norm_neon_range,
    bn_transformer_ssm_delta_neon_range,
    bn_transformer_ssm_gate_neon_range,
    cpu_apply_ffn_activation_neon,
    cpu_apply_sigmoid_gate_scalar,
    cpu_apply_rope_heads_scalar,
    0,
    NULL,
};
#elif BN_TRANSFORMER_CPU_HAS_AVX512
static const BnCPUBackendOps BN_CPU_BACKEND = {
    "avx512",
    bn_transformer_rmsnorm_avx2,
    bn_transformer_gqa_avx2_range,
    bn_transformer_flash_gqa_avx2_range,
    bn_transformer_batched_attn_naive_avx2_range,
    bn_transformer_batched_attn_flash_avx2_range,
    bn_transformer_batched_attn_flash_avx2_pair_range,
    cpu_residual_add_avx2,
    bn_transformer_ssm_conv_silu_avx2_range,
    bn_transformer_ssm_l2norm_avx2_range,
    bn_transformer_ssm_delta_avx2_range,
    bn_transformer_ssm_gate_avx2_range,
    cpu_apply_ffn_activation_avx2,
    cpu_apply_sigmoid_gate_avx2,
    cpu_apply_rope_heads_avx2,
    1,
    bn_quant_rmsnorm_q8k_avx2,
};
#elif BN_TRANSFORMER_CPU_HAS_AVX2
static const BnCPUBackendOps BN_CPU_BACKEND = {
    "avx2",
    bn_transformer_rmsnorm_avx2,
    bn_transformer_gqa_avx2_range,
    bn_transformer_flash_gqa_avx2_range,
    bn_transformer_batched_attn_naive_avx2_range,
    bn_transformer_batched_attn_flash_avx2_range,
    bn_transformer_batched_attn_flash_avx2_pair_range,
    cpu_residual_add_avx2,
    bn_transformer_ssm_conv_silu_avx2_range,
    bn_transformer_ssm_l2norm_avx2_range,
    bn_transformer_ssm_delta_avx2_range,
    bn_transformer_ssm_gate_avx2_range,
    cpu_apply_ffn_activation_avx2,
    cpu_apply_sigmoid_gate_avx2,
    cpu_apply_rope_heads_avx2,
    1,
    bn_quant_rmsnorm_q8k_avx2,
};
#elif BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static const BnCPUBackendOps BN_CPU_BACKEND = {
    "wasm",
    bn_transformer_rmsnorm_wasm,
    bn_transformer_gqa_wasm_range,
    bn_transformer_flash_gqa_wasm_range,
    bn_transformer_batched_attn_naive_scalar_range,
    bn_transformer_batched_attn_flash_scalar_range,
    NULL,
    cpu_residual_add_wasm,
    bn_transformer_ssm_conv_silu_wasm_range,
    bn_transformer_ssm_l2norm_wasm_range,
    bn_transformer_ssm_delta_wasm_range,
    bn_transformer_ssm_gate_wasm_range,
    cpu_apply_ffn_activation_wasm,
    cpu_apply_sigmoid_gate_scalar,
    cpu_apply_rope_heads_scalar,
    0,
    NULL,
};
#else
static const BnCPUBackendOps BN_CPU_BACKEND = {
    "scalar",
    bn_transformer_rmsnorm_scalar,
    bn_transformer_gqa_scalar_range,
    bn_transformer_flash_gqa_scalar_range,
    bn_transformer_batched_attn_naive_scalar_range,
    bn_transformer_batched_attn_flash_scalar_range,
    NULL,
    cpu_residual_add_scalar,
    bn_transformer_ssm_conv_silu_scalar_range,
    bn_transformer_ssm_l2norm_scalar_range,
    bn_transformer_ssm_delta_scalar_range,
    bn_transformer_ssm_gate_scalar_range,
    cpu_apply_ffn_activation_scalar,
    cpu_apply_sigmoid_gate_scalar,
    cpu_apply_rope_heads_scalar,
    0,
    NULL,
};
#endif

const BnCPUBackendOps *bn_transformer_cpu_backend_ops(void) {
    return &BN_CPU_BACKEND;
}

BnCPUBackendPlacement bn_transformer_cpu_backend_placement(void) {
#if BN_TRANSFORMER_CPU_HAS_AVX512
    return BN_CPU_BACKEND_AVX512;
#elif BN_TRANSFORMER_CPU_HAS_AVX2
    return BN_CPU_BACKEND_AVX2;
#elif BN_TRANSFORMER_CPU_HAS_NEON
    return BN_CPU_BACKEND_NEON;
#elif BN_TRANSFORMER_CPU_HAS_WASM_RELAXED_SIMD || \
      BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
    return BN_CPU_BACKEND_WASM_SIMD;
#else
    return BN_CPU_BACKEND_SCALAR;
#endif
}
