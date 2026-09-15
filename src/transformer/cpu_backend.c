#include "transformer_cpu_backend_internal.h"
#include "transformer_cpu_features_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_batched_attn_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "backend_quant.h"
#include "quant_dispatch_internal.h"
#include "quant.h"
#include "simd_helpers.h"

#include <math.h>

#if !BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_scaled_residual_add_default(float *x, const float *r,
                                            float scale, int dim,
                                            float *scratch) {
    (void)scratch;
    for (int i = 0; i < dim; i++) x[i] += r[i] * scale;
}
#endif

#if BN_TRANSFORMER_CPU_HAS_NEON
static void cpu_residual_add_reference(float *x, const float *r, int dim) {
    for (int i = 0; i < dim; i++)
        x[i] += r[i];
}
#endif

static float cpu_reference_gelu(float x) {
    if (x <= -10.0f)
        return 0.0f;
    if (x >= 10.0f)
        return x;
    uint16_t rounded_bits = bn_fp32_to_fp16(x);
    if (rounded_bits == 0xbfffu)
        return bn_fp16_to_fp32(0xa9d3u);
    float rounded_x = bn_fp16_to_fp32(rounded_bits);
    float inner = 0.7978845608028654f * rounded_x *
                  (1.0f + 0.044715f * rounded_x * rounded_x);
    float gelu = 0.5f * rounded_x * (1.0f + tanhf(inner));
    return bn_fp16_to_fp32(bn_fp32_to_fp16(gelu));
}

#if !BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_apply_sigmoid_gate_scalar(float *x, const float *gate,
                                          int size);
#endif
#if BN_TRANSFORMER_CPU_HAS_NEON || BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_apply_ffn_reference_activation(BnRunState *s,
                                               const BnFFNPlan *ffn_plan,
                                               int hidden_dim);
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
static inline __m256 cpu_avx2_silu(__m256 x, int batched_prompt_contract) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 ex = batched_prompt_contract
        ? bn_avx2_fast_exp_ps(neg_x)
        : bn_avx2_fast_exp_avx512_ps(neg_x);
    return _mm256_div_ps(x, _mm256_add_ps(one, ex));
}

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

int bn_transformer_cpu_has_native_quant_activation(void) {
    return BN_TRANSFORMER_CPU_HAS_NATIVE_QUANT_ACTIVATION;
}

uint32_t bn_transformer_cpu_ssm_out_matvec_task_flags(void) {
#if defined(__AVX2__) && !defined(__AVX512F__)
    return BN_MATVEC_TASK_REFERENCE_DOT;
#else
    return 0u;
#endif
}

int bn_transformer_cpu_weight_uses_native_quant_activation(
    const BnQWeight *weight) {
    return weight && bn_backend_quant_uses_native_quant(weight->type);
}

void bn_transformer_cpu_prepare_kquant_activation(const float *x,
                                                  int8_t *quantized,
                                                  float *scales,
                                                  int16_t *block_sums,
                                                  int n) {
#if BN_TRANSFORMER_CPU_HAS_NATIVE_QUANT_ACTIVATION
    bn_backend_quant_prepare_kquant_activation(x, quantized, scales,
                                               block_sums, n);
#else
    bn_backend_quant_prepare_kquant_activation_scalar(x, quantized, scales,
                                                      block_sums, n);
#endif
}

int bn_transformer_cpu_quantize_native_logits_refine_activation(
    const float *x,
    int8_t *quantized,
    float *scales,
    int n) {
    if (!bn_transformer_cpu_has_native_quant_activation())
        return -1;
    bn_quant_x_to_q8_blocks(x, quantized, scales, n);
    return 0;
}

static float cpu_quantize_i8_activation_scalar(const float *x,
                                               int8_t *quantized,
                                               int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = x[i] < 0.0f ? -x[i] : x[i];
        if (ax > amax)
            amax = ax;
    }
    float scale = amax / 127.0f;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int i = 0; i < n; i++) {
        int q = (int)(x[i] * inv + (x[i] >= 0.0f ? 0.5f : -0.5f));
        if (q > 127)
            q = 127;
        if (q < -127)
            q = -127;
        quantized[i] = (int8_t)q;
    }
    return scale;
}

float bn_transformer_cpu_quantize_i8_activation(const float *x,
                                                int8_t *quantized,
                                                int n,
                                                int use_standard_quant) {
    if (use_standard_quant)
        return bn_quant_x_to_i8(x, quantized, n);
    return cpu_quantize_i8_activation_scalar(x, quantized, n);
}

void bn_transformer_cpu_prepare_native_logits_refine_activation(
    const float *x,
    int8_t *quantized,
    float *scales,
    int n) {
    bn_quant_x_to_q8_blocks(x, quantized, scales, n);
}

int bn_transformer_cpu_refine_native_logits_row(const BnQWeight *weight,
                                                const int8_t *quantized,
                                                const float *scales,
                                                int row,
                                                float *out) {
    return bn_backend_quant_refine_native_quant_logits_row(
        weight, quantized, scales, row, out);
}

int bn_transformer_cpu_refine_kquant_logits_row(const BnQWeight *weight,
                                                const float *x,
                                                int row,
                                                float *out) {
    return bn_backend_quant_refine_kquant_logits_row(weight, x, row, out);
}

int bn_transformer_cpu_refine_kquant_logits_prepared_activation_row(
    const BnQWeight *weight,
    const int8_t *quantized,
    const float *scales,
    const int16_t *block_sums,
    int row,
    float *out) {
    return bn_backend_quant_refine_kquant_logits_prepared_activation_row(
        weight, quantized, scales, block_sums, row, out);
}

void bn_transformer_cpu_quant_matvec(float *out,
                                     const BnQWeight *weight,
                                     const float *x,
                                     int8_t *quantized_buf,
                                     BnThreadPool *pool) {
    bn_quant_matvec(out, weight, x, quantized_buf, pool);
}

void bn_transformer_cpu_quant_matvec_prepared_flags(
    float *out,
    const BnQWeight *weight,
    const BnPreparedWeight *prepared,
    const float *x,
    int8_t *quantized_buf,
    BnThreadPool *pool,
    uint32_t flags) {
    bn_quant_matvec_prepared_flags(out, weight, prepared, x, quantized_buf,
                                   pool, flags);
}

void bn_transformer_cpu_quant_matvec_batch(const BnMatvecTask *tasks,
                                           int n_tasks,
                                           const float *x,
                                           int8_t *quantized_buf,
                                           BnThreadPool *pool) {
    bn_quant_matvec_batch(tasks, n_tasks, x, quantized_buf, pool);
}

void bn_transformer_cpu_quant_matvec_batch_prepared_kquant_input(
    const BnMatvecTask *tasks,
    int n_tasks,
    const int8_t *quantized,
    const float *scales,
    const int16_t *block_sums,
    const float *x_float,
    BnThreadPool *pool) {
    bn_quant_matvec_batch_prepared_kquant_input(
        tasks, n_tasks, quantized, scales, block_sums, x_float, pool);
}

int bn_transformer_cpu_fused_kquant_gateup_silu(
    float *out,
    const BnQWeight *gate,
    const BnPreparedWeight *gate_prepared,
    const BnQWeight *up,
    const BnPreparedWeight *up_prepared,
    const float *x,
    int8_t *quantized_buf,
    BnThreadPool *pool) {
    return bn_quant_q4_gate_up_silu(out, gate, gate_prepared, up,
                                    up_prepared, x, quantized_buf, pool);
}

#if BN_TRANSFORMER_CPU_HAS_NEON
static void cpu_residual_add_neon(float *x, const float *r, int dim) {
    for (int i = 0; i < dim; i += 4)
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(r + i)));
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX512
static inline __m512 cpu_avx512_reference_silu(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 exp_neg_x = bn_avx512_fast_exp_ps(
        _mm512_sub_ps(_mm512_setzero_ps(), x));
    return _mm512_div_ps(x, _mm512_add_ps(one, exp_neg_x));
}

static void cpu_apply_ffn_activation_avx512(BnRunState *s,
                                            const BnFFNPlan *ffn_plan,
                                            int hidden_dim) {
    if (!bn_transformer_cpu_activation_uses_silu_path(
            ffn_plan->activation)) {
        cpu_apply_ffn_activation_avx2(s, ffn_plan, hidden_dim);
        return;
    }
    int i = 0;
    for (; i + 15 < hidden_dim; i += 16) {
        __m512 input = _mm512_loadu_ps(s->hb + i);
        __m512 value = ffn_plan->reference_activation
            ? cpu_avx512_reference_silu(input)
            : bn_avx512_fast_silu_ps(input);
        if (ffn_plan->has_gate)
            value = _mm512_mul_ps(value, _mm512_loadu_ps(s->hb2 + i));
        _mm512_storeu_ps(s->hb + i, value);
    }
    if (i < hidden_dim) {
        const __mmask16 mask = (__mmask16)((1u << (hidden_dim - i)) - 1u);
        __m512 input = _mm512_maskz_loadu_ps(mask, s->hb + i);
        __m512 value = ffn_plan->reference_activation
            ? cpu_avx512_reference_silu(input)
            : bn_avx512_fast_silu_ps(input);
        if (ffn_plan->has_gate)
            value = _mm512_mul_ps(
                value, _mm512_maskz_loadu_ps(mask, s->hb2 + i));
        _mm512_mask_storeu_ps(s->hb + i, mask, value);
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_residual_add_avx2(float *x, const float *r, int dim) {
    for (int i = 0; i < dim; i += 8)
        _mm256_storeu_ps(x + i,
                         _mm256_add_ps(_mm256_loadu_ps(x + i),
                                       _mm256_loadu_ps(r + i)));
}

static void cpu_scaled_residual_add_avx2(float *x, const float *r,
                                         float scale, int dim,
                                         float *scratch) {
    /* Keep the product rounded before addition even when this helper is
     * inlined. Volatile staging is confined to this exact-arithmetic path. */
    volatile float *rounded = scratch;
    for (int i = 0; i < dim; i++) rounded[i] = r[i] * scale;
    for (int i = 0; i < dim; i++) x[i] += rounded[i];
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
static float cpu_fast_silu_scalar(float x) {
    return x / (1.0f + bn_transformer_fast_exp_scalar(-x));
}

static float cpu_fast_tanh_scalar(float x) {
    return 2.0f / (1.0f + bn_transformer_fast_exp_scalar(-2.0f * x)) -
           1.0f;
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
    // The reference sigmoid operator uses expf, unlike its vector SiLU.
    // Approximation differences can change the following quantized projection.
    for (int i = 0; i < size; i++)
        x[i] *= 1.0f / (1.0f + expf(-gate[i]));
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

#if BN_TRANSFORMER_CPU_HAS_NEON || BN_TRANSFORMER_CPU_HAS_AVX2
static void cpu_apply_ffn_reference_activation(BnRunState *s,
                                               const BnFFNPlan *ffn_plan,
                                               int hidden_dim) {
    for (int i = 0; i < hidden_dim; i++) {
        float x = s->hb[i];
        float value;
        if (bn_transformer_cpu_activation_is_relu2(ffn_plan->activation)) {
            float positive = x > 0.0f ? x : 0.0f;
            value = positive * positive;
        } else if (bn_transformer_cpu_activation_is_gelu(
                       ffn_plan->activation)) {
            value = cpu_reference_gelu(x);
        } else {
            value = x / (1.0f + expf(-x));
        }
        s->hb[i] = ffn_plan->has_gate ? value * s->hb2[i] : value;
    }
}
#endif

#if !BN_TRANSFORMER_CPU_HAS_NEON && !BN_TRANSFORMER_CPU_HAS_AVX2 && \
    !BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static void cpu_apply_ffn_activation_scalar(BnRunState *s,
                                            const BnFFNPlan *ffn_plan,
                                            int hidden_dim) {
    if (ffn_plan->has_gate) {
        if (bn_transformer_cpu_activation_is_relu2(ffn_plan->activation)) {
            for (int i = 0; i < hidden_dim; i++) {
                float g = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = g * g * s->hb2[i];
            }
        } else if (bn_transformer_cpu_activation_is_gelu(ffn_plan->activation)) {
            for (int i = 0; i < hidden_dim; i++) {
                float x = s->hb[i];
                float g;
                if (ffn_plan->reference_activation)
                    g = cpu_reference_gelu(x);
                else
                    g = cpu_fast_gelu_scalar(x);
                s->hb[i] = g * s->hb2[i];
            }
        } else {
            for (int i = 0; i < hidden_dim; i++) {
                float g = s->hb[i];
                if (ffn_plan->reference_activation)
                    s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
                else
                    s->hb[i] = cpu_fast_silu_scalar(g) * s->hb2[i];
            }
        }
    } else {
        if (bn_transformer_cpu_activation_is_relu2(ffn_plan->activation)) {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = v * v;
            }
        } else if (bn_transformer_cpu_activation_is_gelu(ffn_plan->activation)) {
            for (int i = 0; i < hidden_dim; i++) {
                float x = s->hb[i];
                if (ffn_plan->reference_activation)
                    s->hb[i] = cpu_reference_gelu(x);
                else
                    s->hb[i] = cpu_fast_gelu_scalar(x);
            }
        } else {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i];
                if (ffn_plan->reference_activation)
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
    if (ffn_plan->reference_activation) {
        cpu_apply_ffn_reference_activation(s, ffn_plan, hidden_dim);
        return;
    }
    if (ffn_plan->has_gate) {
        if (bn_transformer_cpu_activation_is_relu2(ffn_plan->activation)) {
            float32x4_t zero = vdupq_n_f32(0);
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t g = vmaxq_f32(vld1q_f32(s->hb + i), zero);
                vst1q_f32(s->hb + i, vmulq_f32(vmulq_f32(g, g),
                                                vld1q_f32(s->hb2 + i)));
            }
        } else if (bn_transformer_cpu_activation_is_gelu(ffn_plan->activation)) {
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t g = vld1q_f32(s->hb + i);
                float32x4_t u = vld1q_f32(s->hb2 + i);
                vst1q_f32(s->hb + i,
                          vmulq_f32(bn_neon_fast_gelu_f32(g), u));
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
        if (bn_transformer_cpu_activation_is_relu2(ffn_plan->activation)) {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = v * v;
            }
        } else if (bn_transformer_cpu_activation_is_gelu(ffn_plan->activation)) {
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t v = vld1q_f32(s->hb + i);
                vst1q_f32(s->hb + i, bn_neon_fast_gelu_f32(v));
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
    if (ffn_plan->reference_activation &&
        !bn_transformer_cpu_activation_uses_silu_path(
            ffn_plan->activation)) {
        cpu_apply_ffn_reference_activation(s, ffn_plan, hidden_dim);
        return;
    }
    if (ffn_plan->has_gate) {
        if (bn_transformer_cpu_activation_is_relu2(ffn_plan->activation)) {
            __m256 zero = _mm256_setzero_ps();
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 g = _mm256_max_ps(_mm256_loadu_ps(s->hb + i), zero);
                _mm256_storeu_ps(s->hb + i,
                    _mm256_mul_ps(_mm256_mul_ps(g, g),
                                  _mm256_loadu_ps(s->hb2 + i)));
            }
        } else if (bn_transformer_cpu_activation_is_gelu(ffn_plan->activation)) {
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
                                 _mm256_mul_ps(cpu_avx2_silu(
                                     g, s->batched_prompt_contract), u));
            }
        }
    } else {
        if (bn_transformer_cpu_activation_is_relu2(ffn_plan->activation)) {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = v * v;
            }
        } else if (bn_transformer_cpu_activation_is_gelu(ffn_plan->activation)) {
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 v = _mm256_loadu_ps(s->hb + i);
                _mm256_storeu_ps(s->hb + i, bn_avx2_fast_gelu_ps(v));
            }
        } else {
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 v = _mm256_loadu_ps(s->hb + i);
                _mm256_storeu_ps(s->hb + i, cpu_avx2_silu(
                    v, s->batched_prompt_contract));
            }
        }
    }
}
#endif

#if BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static void cpu_apply_ffn_activation_wasm(BnRunState *s,
                                          const BnFFNPlan *ffn_plan,
                                          int hidden_dim) {
    if (ffn_plan->has_gate && bn_transformer_cpu_activation_is_relu2(ffn_plan->activation)) {
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
            if (bn_transformer_cpu_activation_is_relu2(ffn_plan->activation)) {
                g = x > 0 ? x * x : 0.0f;
            } else if (bn_transformer_cpu_activation_is_gelu(ffn_plan->activation)) {
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
        if (bn_transformer_cpu_activation_is_relu2(ffn_plan->activation)) {
            float v = x > 0 ? x : 0.0f;
            s->hb[i] = v * v;
        } else if (bn_transformer_cpu_activation_is_gelu(ffn_plan->activation)) {
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
    .name = "neon",
    .rmsnorm = bn_transformer_rmsnorm_scalar,
    .gqa = bn_transformer_gqa_neon_range,
    .flash_gqa = bn_transformer_flash_gqa_neon_range,
    .batched_attn_naive = bn_transformer_batched_attn_naive_neon_range,
    .batched_attn_naive_pair = NULL,
    .batched_attn_flash = bn_transformer_batched_attn_flash_neon_range,
    .batched_attn_flash_pair = NULL,
    .residual_add = cpu_residual_add_neon,
    .scaled_residual_add = cpu_scaled_residual_add_default,
    .ssm_conv_silu = bn_transformer_ssm_conv_silu_neon_range,
    .ssm_l2norm = bn_transformer_ssm_l2norm_neon_range,
    .ssm_delta = bn_transformer_ssm_delta_neon_range,
    .ssm_gate = bn_transformer_ssm_gate_neon_range,
    .apply_ffn_activation = cpu_apply_ffn_activation_neon,
    .apply_sigmoid_gate = cpu_apply_sigmoid_gate_scalar,
    .apply_rope_heads = cpu_apply_rope_heads_scalar,
    .supports_prepared_kquant = 0,
    .supports_float_kquant_prefill = 0,
    .prefill_projection_replay =
        BN_CPU_PREFILL_PROJECTION_REPLAY_FLOAT_KQUANT_TAIL,
    .supports_hybrid_batch_prefill = 1,
    .rmsnorm_prepared_kquant = NULL,
};
#elif BN_TRANSFORMER_CPU_HAS_AVX512
static const BnCPUBackendOps BN_CPU_BACKEND = {
    .name = "avx512",
    .supports_hyper_connection_batch_prefill = 1,
    .selects_last_prefill_ffn_row = 1,
    .rmsnorm = bn_transformer_rmsnorm_avx2,
    .gqa = bn_transformer_gqa_avx512_range,
    .flash_gqa = bn_transformer_flash_gqa_avx2_range,
    .batched_attn_naive = bn_transformer_batched_attn_naive_avx2_range,
    .batched_attn_naive_pair = bn_transformer_batched_attn_naive_avx2_pair_range,
    .batched_attn_flash = bn_transformer_batched_attn_flash_avx2_range,
    .batched_attn_flash_pair = bn_transformer_batched_attn_flash_avx2_pair_range,
    .residual_add = cpu_residual_add_avx2,
    .scaled_residual_add = cpu_scaled_residual_add_avx2,
    .ssm_conv_silu = bn_transformer_ssm_conv_silu_x86_range,
    .ssm_l2norm = bn_transformer_ssm_l2norm_avx2_range,
    .ssm_delta = bn_transformer_ssm_delta_avx2_range,
    .ssm_gate = bn_transformer_ssm_gate_x86_range,
    .apply_ffn_activation = cpu_apply_ffn_activation_avx512,
    .apply_sigmoid_gate = cpu_apply_sigmoid_gate_avx2,
    .apply_rope_heads = cpu_apply_rope_heads_avx2,
    .supports_prepared_kquant = 1,
    .supports_float_kquant_prefill = 1,
    .prefill_projection_replay = BN_CPU_PREFILL_PROJECTION_REPLAY_NONE,
    .supports_hybrid_batch_prefill = 1,
    .rmsnorm_prepared_kquant = bn_backend_quant_rmsnorm_prepared_kquant_avx2,
};
#elif BN_TRANSFORMER_CPU_HAS_AVX2
static const BnCPUBackendOps BN_CPU_BACKEND = {
    .name = "avx2",
    .supports_hyper_connection_batch_prefill = 1,
    .selects_last_prefill_ffn_row = 1,
    .rmsnorm = bn_transformer_rmsnorm_avx2,
    .gqa = bn_transformer_gqa_avx2_range,
    .flash_gqa = bn_transformer_flash_gqa_avx2_range,
    .batched_attn_naive = bn_transformer_batched_attn_naive_avx2_range,
    .batched_attn_naive_pair = bn_transformer_batched_attn_naive_avx2_pair_range,
    .batched_attn_flash = bn_transformer_batched_attn_flash_avx2_range,
    .batched_attn_flash_pair = bn_transformer_batched_attn_flash_avx2_pair_range,
    .residual_add = cpu_residual_add_avx2,
    .scaled_residual_add = cpu_scaled_residual_add_avx2,
    .ssm_conv_silu = bn_transformer_ssm_conv_silu_avx2_range,
    .ssm_l2norm = bn_transformer_ssm_l2norm_avx2_range,
    .ssm_delta = bn_transformer_ssm_delta_avx2_range,
    .ssm_gate = bn_transformer_ssm_gate_avx2_range,
    .apply_ffn_activation = cpu_apply_ffn_activation_avx2,
    .apply_sigmoid_gate = cpu_apply_sigmoid_gate_avx2,
    .apply_rope_heads = cpu_apply_rope_heads_avx2,
    .supports_prepared_kquant = 1,
    .supports_float_kquant_prefill = 1,
    .prefill_projection_replay = BN_CPU_PREFILL_PROJECTION_REPLAY_NONE,
    .supports_hybrid_batch_prefill = 1,
    .rmsnorm_prepared_kquant = bn_backend_quant_rmsnorm_prepared_kquant_avx2,
};
#elif BN_TRANSFORMER_CPU_HAS_WASM_SIMD128
static const BnCPUBackendOps BN_CPU_BACKEND = {
    .name = "wasm",
    .rmsnorm = bn_transformer_rmsnorm_wasm,
    .gqa = bn_transformer_gqa_wasm_range,
    .flash_gqa = bn_transformer_flash_gqa_wasm_range,
    .batched_attn_naive = bn_transformer_batched_attn_naive_scalar_range,
    .batched_attn_naive_pair = NULL,
    .batched_attn_flash = bn_transformer_batched_attn_flash_scalar_range,
    .batched_attn_flash_pair = NULL,
    .residual_add = cpu_residual_add_wasm,
    .scaled_residual_add = cpu_scaled_residual_add_default,
    .ssm_conv_silu = bn_transformer_ssm_conv_silu_wasm_range,
    .ssm_l2norm = bn_transformer_ssm_l2norm_wasm_range,
    .ssm_delta = bn_transformer_ssm_delta_wasm_range,
    .ssm_gate = bn_transformer_ssm_gate_wasm_range,
    .apply_ffn_activation = cpu_apply_ffn_activation_wasm,
    .apply_sigmoid_gate = cpu_apply_sigmoid_gate_scalar,
    .apply_rope_heads = cpu_apply_rope_heads_scalar,
    .supports_prepared_kquant = 0,
    .supports_float_kquant_prefill = 0,
    .prefill_projection_replay = BN_CPU_PREFILL_PROJECTION_REPLAY_NONE,
    .supports_hybrid_batch_prefill = 0,
    .rmsnorm_prepared_kquant = NULL,
};
#else
static const BnCPUBackendOps BN_CPU_BACKEND = {
    .name = "scalar",
    .rmsnorm = bn_transformer_rmsnorm_scalar,
    .gqa = bn_transformer_gqa_scalar_range,
    .flash_gqa = bn_transformer_flash_gqa_scalar_range,
    .batched_attn_naive = bn_transformer_batched_attn_naive_scalar_range,
    .batched_attn_naive_pair = NULL,
    .batched_attn_flash = bn_transformer_batched_attn_flash_scalar_range,
    .batched_attn_flash_pair = NULL,
    .residual_add = cpu_residual_add_scalar,
    .scaled_residual_add = cpu_scaled_residual_add_default,
    .ssm_conv_silu = bn_transformer_ssm_conv_silu_scalar_range,
    .ssm_l2norm = bn_transformer_ssm_l2norm_scalar_range,
    .ssm_delta = bn_transformer_ssm_delta_scalar_range,
    .ssm_gate = bn_transformer_ssm_gate_scalar_range,
    .apply_ffn_activation = cpu_apply_ffn_activation_scalar,
    .apply_sigmoid_gate = cpu_apply_sigmoid_gate_scalar,
    .apply_rope_heads = cpu_apply_rope_heads_scalar,
    .supports_prepared_kquant = 0,
    .supports_float_kquant_prefill = 0,
    .prefill_projection_replay = BN_CPU_PREFILL_PROJECTION_REPLAY_NONE,
    .supports_hybrid_batch_prefill = 0,
    .rmsnorm_prepared_kquant = NULL,
};
#endif

const BnCPUBackendOps *bn_transformer_cpu_backend_ops(
    const BnCPURuntimePolicy *runtime) {
#if BN_TRANSFORMER_CPU_HAS_NEON
    static const BnCPUBackendOps reference = {
        .name = "neon-quant-reference-math",
        .rmsnorm = bn_transformer_rmsnorm_scalar,
        .gqa = bn_transformer_gqa_scalar_range,
        .flash_gqa = bn_transformer_flash_gqa_scalar_range,
        .batched_attn_naive = bn_transformer_batched_attn_naive_scalar_range,
        .batched_attn_flash = bn_transformer_batched_attn_flash_scalar_range,
        .batched_attn_flash_pair = NULL,
        .residual_add = cpu_residual_add_reference,
        .scaled_residual_add = cpu_scaled_residual_add_default,
        .ssm_conv_silu = bn_transformer_ssm_conv_silu_neon_range,
        .ssm_l2norm = bn_transformer_ssm_l2norm_neon_range,
        .ssm_delta = bn_transformer_ssm_delta_neon_range,
        .ssm_gate = bn_transformer_ssm_gate_neon_range,
        .apply_ffn_activation = cpu_apply_ffn_reference_activation,
        .apply_sigmoid_gate = cpu_apply_sigmoid_gate_scalar,
        .apply_rope_heads = cpu_apply_rope_heads_scalar,
        .supports_prepared_kquant = 0,
        .supports_float_kquant_prefill = 0,
        .rmsnorm_prepared_kquant = NULL,
    };
    if (bn_transformer_cpu_reference_math_requested(runtime))
        return &reference;
#else
    (void)runtime;
#endif
    return &BN_CPU_BACKEND;
}

bn_tp_fn bn_transformer_cpu_ssm_conv_silu_op(const BnCPUBackendOps *ops) {
    return ops ? ops->ssm_conv_silu : BN_CPU_BACKEND.ssm_conv_silu;
}

bn_tp_fn bn_transformer_cpu_ssm_l2norm_op(const BnCPUBackendOps *ops) {
    return ops ? ops->ssm_l2norm : BN_CPU_BACKEND.ssm_l2norm;
}

bn_tp_fn bn_transformer_cpu_ssm_delta_op(const BnCPUBackendOps *ops) {
    return ops ? ops->ssm_delta : BN_CPU_BACKEND.ssm_delta;
}

bn_tp_fn bn_transformer_cpu_ssm_gate_op(const BnCPUBackendOps *ops) {
    return ops ? ops->ssm_gate : BN_CPU_BACKEND.ssm_gate;
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

int bn_transformer_cpu_backend_supports_float_kquant_prefill(void) {
    return BN_CPU_BACKEND.supports_float_kquant_prefill;
}

BnCPUPrefillProjectionReplayKind
bn_transformer_cpu_backend_prefill_projection_replay(void) {
    return BN_CPU_BACKEND.prefill_projection_replay;
}

int bn_transformer_cpu_backend_supports_hybrid_batch_prefill(void) {
    return BN_CPU_BACKEND.supports_hybrid_batch_prefill;
}

int bn_transformer_cpu_backend_supports_hyper_connection_batch_prefill(void) {
    return BN_CPU_BACKEND.supports_hyper_connection_batch_prefill;
}

int bn_transformer_cpu_backend_selects_last_prefill_ffn_row(void) {
    return BN_CPU_BACKEND.selects_last_prefill_ffn_row;
}
