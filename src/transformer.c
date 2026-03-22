#include "transformer_internal.h"
#include "quant_internal.h"
#include "moe.h"
#include "platform.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdlib.h>

// Per-layer timing instrumentation (compile with -DBN_BENCH_LAYERS)
#ifdef BN_BENCH_LAYERS
static double bl_rmsnorm_us, bl_matvec_qkv_us, bl_rope_us, bl_gqa_us;
static double bl_logits_us, bl_residual_us, bl_ffn_us;
static double bl_ssm_conv_us, bl_ssm_l2norm_us, bl_ssm_delta_us, bl_ssm_gate_us;
static double bl_sigmoid_gate_us;
static int bl_layer_count;

#define BL_START() double _bl_t0 = bn_platform_time_ms()
#define BL_ACC(var) do { (var) += (bn_platform_time_ms() - _bl_t0) * 1000.0; _bl_t0 = bn_platform_time_ms(); } while(0)

static void bl_print_reset(void) {
    if (bl_layer_count == 0) return;
    double total = bl_rmsnorm_us + bl_matvec_qkv_us + bl_rope_us + bl_gqa_us +
                   bl_logits_us + bl_residual_us + bl_ffn_us +
                   bl_ssm_conv_us + bl_ssm_l2norm_us + bl_ssm_delta_us +
                   bl_ssm_gate_us + bl_sigmoid_gate_us;
    fprintf(stderr, "\n=== Per-layer timing (%d layers) ===\n", bl_layer_count);
    fprintf(stderr, "  RMSNorm:       %8.0f us (%.1f%%)\n", bl_rmsnorm_us, 100*bl_rmsnorm_us/total);
    fprintf(stderr, "  Matvec QKV:    %8.0f us (%.1f%%)\n", bl_matvec_qkv_us, 100*bl_matvec_qkv_us/total);
    fprintf(stderr, "  RoPE:          %8.0f us (%.1f%%)\n", bl_rope_us, 100*bl_rope_us/total);
    fprintf(stderr, "  GQA:           %8.0f us (%.1f%%)\n", bl_gqa_us, 100*bl_gqa_us/total);
    fprintf(stderr, "  Sigmoid gate:  %8.0f us (%.1f%%)\n", bl_sigmoid_gate_us, 100*bl_sigmoid_gate_us/total);
    fprintf(stderr, "  FFN:           %8.0f us (%.1f%%)\n", bl_ffn_us, 100*bl_ffn_us/total);
    fprintf(stderr, "  Residual:      %8.0f us (%.1f%%)\n", bl_residual_us, 100*bl_residual_us/total);
    fprintf(stderr, "  SSM conv:      %8.0f us (%.1f%%)\n", bl_ssm_conv_us, 100*bl_ssm_conv_us/total);
    fprintf(stderr, "  SSM l2norm:    %8.0f us (%.1f%%)\n", bl_ssm_l2norm_us, 100*bl_ssm_l2norm_us/total);
    fprintf(stderr, "  SSM delta:     %8.0f us (%.1f%%)\n", bl_ssm_delta_us, 100*bl_ssm_delta_us/total);
    fprintf(stderr, "  SSM gate:      %8.0f us (%.1f%%)\n", bl_ssm_gate_us, 100*bl_ssm_gate_us/total);
    fprintf(stderr, "  Logits:        %8.0f us (%.1f%%)\n", bl_logits_us, 100*bl_logits_us/total);
    fprintf(stderr, "  TOTAL:         %8.0f us\n", total);
    bl_rmsnorm_us = bl_matvec_qkv_us = bl_rope_us = bl_gqa_us = 0;
    bl_logits_us = bl_residual_us = bl_ffn_us = 0;
    bl_ssm_conv_us = bl_ssm_l2norm_us = bl_ssm_delta_us = bl_ssm_gate_us = 0;
    bl_sigmoid_gate_us = 0;
    bl_layer_count = 0;
}
#else
#define BL_START() (void)0
#define BL_ACC(var) (void)0
#endif

// Max elements for stack VLAs (head_size, dim). Prevents stack overflow
// from malicious model configs. 8192 = 32KB of floats, well within stack.
#define BN_MAX_VLA_ELEMS 8192

// Backend-selected rmsnorm
#ifdef __ARM_NEON
#define rmsnorm bn_transformer_rmsnorm_neon
#elif defined(__AVX2__)
#define rmsnorm bn_transformer_rmsnorm_avx2
#elif defined(__wasm_simd128__)
#define rmsnorm bn_transformer_rmsnorm_wasm
#else
#define rmsnorm bn_transformer_rmsnorm_scalar
#endif

// --- Forward pass ---

// Inline helper: add residual xb (or xb2) into x
static inline void residual_add(float *x, const float *r, int dim) {
#ifdef __ARM_NEON
    for (int i = 0; i < dim; i += 4)
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(r + i)));
#elif defined(__AVX2__)
    for (int i = 0; i < dim; i += 8)
        _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(r + i)));
#elif defined(__wasm_simd128__)
    for (int i = 0; i < dim; i += 4)
        wasm_v128_store(x + i, wasm_f32x4_add(wasm_v128_load(x + i), wasm_v128_load(r + i)));
#else
    for (int i = 0; i < dim; i++) x[i] += r[i];
#endif
}

// Apply per-head RoPE using precomputed cos/sin.
// rope_dims = number of dims per head to rotate (rest pass through).
static inline void apply_rope_heads(float *buf, int n_heads, int head_size,
                                    int rope_dims, const float *rc, const float *rs) {
#ifdef __AVX2__
    // AVX2 path: process 4 rotation pairs (8 floats) at a time
    // For each pair (v0,v1): out0 = v0*cos - v1*sin, out1 = v0*sin + v1*cos
    // We need to interleave cos/sin values to match the pair layout
    if (rope_dims >= 8) {
        // Pre-expand cos/sin from half-rate to pair-rate for AVX2
        // rc[i/2] → [rc0,rc0,rc1,rc1,rc2,rc2,rc3,rc3]
        // Process in groups of 4 pairs = 8 floats
        for (int h = 0; h < n_heads; h++) {
            float *hd = buf + h * head_size;
            int i = 0;
            for (; i + 7 < rope_dims; i += 8) {
                int fi = i / 2;
                // Expand cos/sin: duplicate each value for its pair
                __m256 cos_v = _mm256_set_ps(rc[fi+3], rc[fi+3], rc[fi+2], rc[fi+2],
                                              rc[fi+1], rc[fi+1], rc[fi],   rc[fi]);
                __m256 sin_v = _mm256_set_ps(rs[fi+3], rs[fi+3], rs[fi+2], rs[fi+2],
                                              rs[fi+1], rs[fi+1], rs[fi],   rs[fi]);
                __m256 v = _mm256_loadu_ps(hd + i);
                // Rotate: create [v1,v0,v3,v2,v5,v4,v7,v6] and negate odds
                __m256 v_swap = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2,3,0,1));
                // sin_negate: [-sin,sin,-sin,sin,...] for the rotation formula
                __m256 sign_mask = _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f,
                                                  1.0f, -1.0f, 1.0f, -1.0f);
                __m256 sin_neg = _mm256_mul_ps(sin_v, sign_mask);
                __m256 result = _mm256_fmadd_ps(v, cos_v, _mm256_mul_ps(v_swap, sin_neg));
                _mm256_storeu_ps(hd + i, result);
            }
            // Scalar tail
            for (; i < rope_dims; i += 2) {
                int fi2 = i / 2;
                float v0 = hd[i], v1 = hd[i + 1];
                hd[i]     = v0 * rc[fi2] - v1 * rs[fi2];
                hd[i + 1] = v0 * rs[fi2] + v1 * rc[fi2];
            }
        }
        return;
    }
#endif
    for (int h = 0; h < n_heads; h++) {
        float *hd = buf + h * head_size;
        for (int i = 0; i < rope_dims; i += 2) {
            int fi = i / 2;
            float v0 = hd[i], v1 = hd[i + 1];
            hd[i]     = v0 * rc[fi] - v1 * rs[fi];
            hd[i + 1] = v0 * rs[fi] + v1 * rc[fi];
        }
    }
}

// Backend selection for SSM kernels
#ifdef __ARM_NEON
#define ssm_conv_silu bn_transformer_ssm_conv_silu_neon_range
#define ssm_l2norm    bn_transformer_ssm_l2norm_neon_range
#define ssm_delta     bn_transformer_ssm_delta_neon_range
#define ssm_gate      bn_transformer_ssm_gate_neon_range
#elif defined(__AVX2__)
#define ssm_conv_silu bn_transformer_ssm_conv_silu_avx2_range
#define ssm_l2norm    bn_transformer_ssm_l2norm_avx2_range
#define ssm_delta     bn_transformer_ssm_delta_avx2_range
#define ssm_gate      bn_transformer_ssm_gate_avx2_range
#elif defined(__wasm_simd128__)
#define ssm_conv_silu bn_transformer_ssm_conv_silu_wasm_range
#define ssm_l2norm    bn_transformer_ssm_l2norm_wasm_range
#define ssm_delta     bn_transformer_ssm_delta_wasm_range
#define ssm_gate      bn_transformer_ssm_gate_wasm_range
#else
#define ssm_conv_silu bn_transformer_ssm_conv_silu_scalar_range
#define ssm_l2norm    bn_transformer_ssm_l2norm_scalar_range
#define ssm_delta     bn_transformer_ssm_delta_scalar_range
#define ssm_gate      bn_transformer_ssm_gate_scalar_range
#endif

// SSM block: Gated DeltaNet recurrence. Reads s->x, writes s->xb (result for residual).
static void forward_ssm_block(BnModel *m, BnLayerWeights *lw, int l) {
    BnConfig *c = &m->config;
    BnRunState *s = &m->state;
    int dim = c->dim;
    int num_k_heads = c->ssm_group_count;           // 16
    int head_k_dim  = c->ssm_state_size;            // 128
    int num_v_heads = c->ssm_time_step_rank;         // 32
    int head_v_dim  = c->ssm_inner_size / num_v_heads; // 128
    int key_dim     = num_k_heads * head_k_dim;     // 2048
    int value_dim   = c->ssm_inner_size;            // 4096
    int qkv_dim     = key_dim * 2 + value_dim;      // 8192
    int kern        = c->ssm_conv_kernel;           // 4

    // SSM layer index (contiguous among SSM layers)
    int ssm_idx = l - (l + 1) / c->full_attn_interval;
    size_t state_per_layer = (size_t)num_v_heads * head_k_dim * head_v_dim;
    float *state = s->ssm_state + (size_t)ssm_idx * state_per_layer;
    size_t conv_per_layer = (size_t)(kern - 1) * qkv_dim;
    float *conv_state = s->ssm_conv_state + (size_t)ssm_idx * conv_per_layer;

    BL_START();

    // 1. Norm input
    rmsnorm(s->xb, s->x, lw->attn_norm, dim, c->norm_eps);
    BL_ACC(bl_rmsnorm_us);

    // 2. QKV + Z gate projections (both read from s->xb)
    float *qkv = s->hb;   // [hidden_dim] >= qkv_dim
    float *z   = s->hb2;  // [hidden_dim] >= value_dim
    {
        BnMatvecTask tasks[2] = {
            { qkv, &lw->wqkv },
            { z,   &lw->wz   },
        };
        bn_quant_matvec_batch(tasks, 2, s->xb, s->x_q, m->pool);
    }
    BL_ACC(bl_matvec_qkv_us);

    // 3+4. Causal conv1d + SiLU (dispatched over channels)
    {
        BnSSMConvCtx conv_ctx = { qkv, conv_state, lw->ssm_conv1d, qkv_dim, kern };
        BnTPTask conv_task = { ssm_conv_silu, &conv_ctx, qkv_dim };
        bn_tp_dispatch(m->pool, &conv_task, 1);
    }
    BL_ACC(bl_ssm_conv_us);

    // 5. Split xBC: [Q(key_dim), K(key_dim), V(value_dim)]
    float *q_raw = qkv;
    float *k_raw = qkv + key_dim;
    float *v_raw = qkv + 2 * key_dim;

    // 6. L2 normalize Q and K per head (dispatched over heads)
    {
        BnSSML2NormCtx norm_ctx = { q_raw, k_raw, head_k_dim };
        BnTPTask norm_task = { ssm_l2norm, &norm_ctx, num_k_heads };
        bn_tp_dispatch(m->pool, &norm_task, 1);
    }
    BL_ACC(bl_ssm_l2norm_us);

    // 7. Alpha (decay) and Beta (update rate) from normalized input
    if (num_v_heads > BN_MAX_VLA_ELEMS || head_v_dim > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("SSM dimensions too large for stack VLAs");
        return;
    }
    float alpha_arr[num_v_heads], beta_arr[num_v_heads];
    {
        BnMatvecTask ab[2] = {
            { alpha_arr, &lw->ssm_alpha },
            { beta_arr,  &lw->ssm_beta  },
        };
        bn_quant_matvec_batch(ab, 2, s->xb, s->x_q, m->pool);
    }
    for (int h = 0; h < num_v_heads; h++) {
        float dt = alpha_arr[h] + lw->ssm_dt_bias[h];
        float dt_sp = (dt > 20.0f) ? dt : logf(1.0f + expf(dt));
        alpha_arr[h] = expf(dt_sp * lw->ssm_a[h]);
        beta_arr[h] = 1.0f / (1.0f + expf(-beta_arr[h]));
    }
    BL_ACC(bl_matvec_qkv_us);

    // 8. Delta rule recurrence (dispatched over V-heads)
    float *out = s->xb2;
    {
        float q_scale = 1.0f / sqrtf((float)head_k_dim);
        BnSSMDeltaCtx delta_ctx = {
            state, out, q_raw, k_raw, v_raw,
            alpha_arr, beta_arr,
            num_k_heads, head_k_dim, head_v_dim, q_scale
        };
        BnTPTask delta_task = { ssm_delta, &delta_ctx, num_v_heads };
        bn_tp_dispatch(m->pool, &delta_task, 1);
    }
    BL_ACC(bl_ssm_delta_us);

    // 9. Per-head RMSNorm + SiLU gate (dispatched over V-heads)
    {
        BnSSMGateCtx gate_ctx = { out, z, lw->ssm_norm, c->norm_eps, head_v_dim };
        BnTPTask gate_task = { ssm_gate, &gate_ctx, num_v_heads };
        bn_tp_dispatch(m->pool, &gate_task, 1);
    }
    BL_ACC(bl_ssm_gate_us);

    // 10. Output projection: out[value_dim] → xb[dim]
    {
        BnMatvecTask proj[1] = {{ s->xb, &lw->ssm_out }};
        bn_quant_matvec_batch(proj, 1, out, s->x_q, m->pool);
    }
    BL_ACC(bl_matvec_qkv_us);
}

// FFN block: shared by both attention and SSM layers.
// Reads s->x, uses s->xb/hb/hb2/x_q as scratch. Adds result to s->x.
static void forward_ffn_block(BnModel *m, BnLayerWeights *lw) {
    BnConfig *c = &m->config;
    BnRunState *s = &m->state;
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;

    rmsnorm(s->xb, s->x, lw->ffn_norm, dim, c->norm_eps);

    if (c->has_ffn_gate) {
        {
            BnMatvecTask ffn[2] = {
                { s->hb,  &lw->ffn_gate },
                { s->hb2, &lw->ffn_up   },
            };
            bn_quant_matvec_batch(ffn, 2, s->xb, s->x_q, m->pool);
        }

        if (c->act_type == 1) {
#ifdef __ARM_NEON
            float32x4_t zero = vdupq_n_f32(0);
            for (int i = 0; i < hidden_dim; i += 4) {
                float32x4_t g = vmaxq_f32(vld1q_f32(s->hb + i), zero);
                vst1q_f32(s->hb + i, vmulq_f32(vmulq_f32(g, g), vld1q_f32(s->hb2 + i)));
            }
#elif defined(__AVX2__)
            __m256 zero = _mm256_setzero_ps();
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 g = _mm256_max_ps(_mm256_loadu_ps(s->hb + i), zero);
                _mm256_storeu_ps(s->hb + i, _mm256_mul_ps(_mm256_mul_ps(g, g), _mm256_loadu_ps(s->hb2 + i)));
            }
#elif defined(__wasm_simd128__)
            v128_t zero = wasm_f32x4_splat(0);
            for (int i = 0; i < hidden_dim; i += 4) {
                v128_t g = wasm_f32x4_max(wasm_v128_load(s->hb + i), zero);
                wasm_v128_store(s->hb + i, wasm_f32x4_mul(wasm_f32x4_mul(g, g), wasm_v128_load(s->hb2 + i)));
            }
#else
            for (int i = 0; i < hidden_dim; i++) {
                float g = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = g * g * s->hb2[i];
            }
#endif
        } else {
#ifdef __AVX2__
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 g = _mm256_loadu_ps(s->hb + i);
                __m256 u = _mm256_loadu_ps(s->hb2 + i);
                _mm256_storeu_ps(s->hb + i, _mm256_mul_ps(bn_avx2_fast_silu_ps(g), u));
            }
#else
            for (int i = 0; i < hidden_dim; i++) {
                float g = s->hb[i];
                s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
            }
#endif
        }
    } else {
        {
            BnMatvecTask ffn[1] = {{ s->hb, &lw->ffn_up }};
            bn_quant_matvec_batch(ffn, 1, s->xb, s->x_q, m->pool);
        }

        if (c->act_type == 1) {
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i] > 0 ? s->hb[i] : 0;
                s->hb[i] = v * v;
            }
        } else {
#ifdef __AVX2__
            for (int i = 0; i < hidden_dim; i += 8) {
                __m256 v = _mm256_loadu_ps(s->hb + i);
                _mm256_storeu_ps(s->hb + i, bn_avx2_fast_silu_ps(v));
            }
#else
            for (int i = 0; i < hidden_dim; i++) {
                float v = s->hb[i];
                s->hb[i] = v / (1.0f + expf(-v));
            }
#endif
        }
    }

    if (lw->ffn_sub_norm)
        rmsnorm(s->hb, s->hb, lw->ffn_sub_norm, hidden_dim, c->norm_eps);

    {
        BnMatvecTask down[1] = {{ s->xb, &lw->ffn_down }};
        bn_quant_matvec_batch(down, 1, s->hb, s->x_q, m->pool);
    }

    residual_add(s->x, s->xb, dim);
}

// Process a single layer (attention/SSM block + FFN). Reads/writes s->x.
// Returns 0 on success.
static int forward_single_layer(BnModel *m, int l, int pos, int cache_pos,
                                int rope_dims, const float *rope_cos,
                                const float *rope_sin) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &m->state;
    int dim = c->dim;
    int kv_dim = c->kv_dim;
    int kv_mul = c->kv_mul;
    int head_size = c->head_size;
    int n_heads = c->n_heads;
    BnLayerWeights *lw = &w->layers[l];
    int qk_stride = c->qk_norm_per_head ? head_size : 0; // per-head norm offset

    int is_attn = (c->full_attn_interval == 0) ||
                  ((l + 1) % c->full_attn_interval == 0);

    BL_START();

    if (is_attn) {
        // ---- Attention block ----

        // KV cache offset: contiguous among attention layers only
        int attn_idx = (c->full_attn_interval > 0)
            ? (l + 1) / c->full_attn_interval - 1 : l;
        size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;

        // Q projection width detection:
        // q_dim = n_heads * head_size (total Q output elements)
        // Gated Q (Qwen3.5): wq.rows = 2 * q_dim (interleaved [Q, gate] per head)
        // Wide Q (Qwen3 MoE): wq.rows = q_dim > dim (head_size > dim/n_heads)
        // Classic: wq.rows = dim = q_dim
        int q_dim = n_heads * head_size;
        int q_gated = lw->wq.data && (lw->wq.rows > q_dim);
        int q_wide  = !q_gated && lw->wq.data && (lw->wq.rows > dim);

        rmsnorm(s->xb, s->x, lw->attn_norm, dim, c->norm_eps);
        BL_ACC(bl_rmsnorm_us);

        if (q_gated) {
            // --- Gated Q path (Qwen3.5 attention) ---
            float *q_full = s->hb;  // [2*dim]

            if (c->kv_f16) {
                float *k_tmp = s->hb2;
                float *v_tmp = s->hb2 + kv_dim;
                BnMatvecTask q_task[1] = {{ q_full, &lw->wq }};
                bn_quant_matvec_batch(q_task, 1, s->xb, s->x_q, m->pool);
                BnMatvecTask kv[2] = {
                    { k_tmp, &lw->wk },
                    { v_tmp, &lw->wv },
                };
                bn_quant_matvec_batch(kv, 2, s->xb, s->x_q, m->pool);
                BL_ACC(bl_matvec_qkv_us);

                for (int h = 0; h < n_heads; h++)
                    memcpy(s->q + h * head_size,
                           q_full + h * 2 * head_size,
                           head_size * sizeof(float));

                if (lw->q_norm)
                    for (int h = 0; h < n_heads; h++)
                        rmsnorm(s->q + h*head_size, s->q + h*head_size,
                                lw->q_norm + h*qk_stride, head_size, c->norm_eps);
                if (lw->k_norm)
                    for (int h = 0; h < c->n_kv_heads; h++)
                        rmsnorm(k_tmp + h*head_size, k_tmp + h*head_size,
                                lw->k_norm + h*qk_stride, head_size, c->norm_eps);

                apply_rope_heads(s->q, n_heads, head_size,
                                 rope_dims, rope_cos, rope_sin);
                apply_rope_heads(k_tmp, c->n_kv_heads, head_size,
                                 rope_dims, rope_cos, rope_sin);
                BL_ACC(bl_rope_us);

                uint16_t *kc = (uint16_t *)s->key_cache   + loff + (size_t)cache_pos * kv_dim;
                uint16_t *vc = (uint16_t *)s->value_cache + loff + (size_t)cache_pos * kv_dim;
#ifdef __ARM_NEON
                for (int i = 0; i < kv_dim; i += 4) {
                    vst1_u16(kc + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(k_tmp + i))));
                    vst1_u16(vc + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(v_tmp + i))));
                }
#elif defined(__AVX2__)
                for (int i = 0; i < kv_dim; i += 8) {
                    _mm_storeu_si128((__m128i *)(kc + i), _mm256_cvtps_ph(_mm256_loadu_ps(k_tmp + i), _MM_FROUND_TO_NEAREST_INT));
                    _mm_storeu_si128((__m128i *)(vc + i), _mm256_cvtps_ph(_mm256_loadu_ps(v_tmp + i), _MM_FROUND_TO_NEAREST_INT));
                }
#else
                for (int i = 0; i < kv_dim; i++) {
                    kc[i] = bn_fp32_to_fp16(k_tmp[i]);
                    vc[i] = bn_fp32_to_fp16(v_tmp[i]);
                }
#endif
            } else {
                float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_dim;
                float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_dim;
                BnMatvecTask q_task[1] = {{ q_full, &lw->wq }};
                bn_quant_matvec_batch(q_task, 1, s->xb, s->x_q, m->pool);
                BnMatvecTask kv[2] = {
                    { key_cache_row,   &lw->wk },
                    { value_cache_row, &lw->wv },
                };
                bn_quant_matvec_batch(kv, 2, s->xb, s->x_q, m->pool);

                for (int h = 0; h < n_heads; h++)
                    memcpy(s->q + h * head_size,
                           q_full + h * 2 * head_size,
                           head_size * sizeof(float));

                if (lw->q_norm)
                    for (int h = 0; h < n_heads; h++)
                        rmsnorm(s->q + h*head_size, s->q + h*head_size,
                                lw->q_norm + h*qk_stride, head_size, c->norm_eps);
                if (lw->k_norm)
                    for (int h = 0; h < c->n_kv_heads; h++)
                        rmsnorm(key_cache_row + h*head_size,
                                key_cache_row + h*head_size,
                                lw->k_norm + h*qk_stride, head_size, c->norm_eps);

                apply_rope_heads(s->q, n_heads, head_size,
                                 rope_dims, rope_cos, rope_sin);
                apply_rope_heads(key_cache_row, c->n_kv_heads, head_size,
                                 rope_dims, rope_cos, rope_sin);
            }

            // GQA attention
            {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_dim, c->seq_len };
#ifdef __ARM_NEON
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_neon_range : bn_transformer_gqa_neon_range;
#elif defined(__AVX2__)
                bn_tp_fn attn_fn = bn_transformer_gqa_avx2_range;
#elif defined(__wasm_simd128__)
                bn_tp_fn attn_fn = bn_transformer_gqa_wasm_range;
#else
                bn_tp_fn attn_fn = bn_transformer_gqa_scalar_range;
#endif
                BnTPTask gqa = { attn_fn, &gctx, n_heads };
                bn_tp_dispatch(m->pool, &gqa, 1);
            }
            BL_ACC(bl_gqa_us);

            // Sigmoid gate: xb *= sigmoid(gate)
            for (int h = 0; h < n_heads; h++) {
                float *gate_h = q_full + h * 2 * head_size + head_size;
                float *xb_h = s->xb + h * head_size;
#ifdef __AVX2__
                for (int d = 0; d < head_size; d += 8) {
                    __m256 g = _mm256_loadu_ps(gate_h + d);
                    __m256 xv = _mm256_loadu_ps(xb_h + d);
                    _mm256_storeu_ps(xb_h + d, _mm256_mul_ps(xv, bn_avx2_fast_sigmoid_ps(g)));
                }
#else
                for (int d = 0; d < head_size; d++)
                    xb_h[d] *= 1.0f / (1.0f + expf(-gate_h[d]));
#endif
            }
            BL_ACC(bl_sigmoid_gate_us);

            // wo projection + residual
            if (lw->attn_sub_norm)
                rmsnorm(s->xb, s->xb, lw->attn_sub_norm, dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->wo }};
                bn_quant_matvec_batch(wo, 1, s->xb, s->x_q, m->pool);
            }
            BL_ACC(bl_matvec_qkv_us);
            residual_add(s->x, s->xb2, dim);
            BL_ACC(bl_residual_us);

        } else if (q_wide) {
            // --- Wide Q path (Qwen3 MoE: head_size > dim/n_heads, no gate) ---
            // Q projection: dim → q_dim (larger than dim)
            // K/V: dim → kv_dim (uses head_size from attention.key_length)
            float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_dim;
            float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_dim;

            // Q matvec: xb[dim] → q[q_dim]
            {
                BnMatvecTask q_task[1] = {{ s->q, &lw->wq }};
                bn_quant_matvec_batch(q_task, 1, s->xb, s->x_q, m->pool);
            }
            // K/V matvec: xb[dim] → kv_dim
            {
                BnMatvecTask kv[2] = {
                    { key_cache_row,   &lw->wk },
                    { value_cache_row, &lw->wv },
                };
                bn_quant_matvec_batch(kv, 2, s->xb, s->x_q, m->pool);
            }

            if (lw->q_norm)
                for (int h = 0; h < n_heads; h++)
                    rmsnorm(s->q + h*head_size, s->q + h*head_size,
                            lw->q_norm + h*qk_stride, head_size, c->norm_eps);
            if (lw->k_norm)
                for (int h = 0; h < c->n_kv_heads; h++)
                    rmsnorm(key_cache_row + h*head_size, key_cache_row + h*head_size,
                            lw->k_norm + h*qk_stride, head_size, c->norm_eps);

            apply_rope_heads(s->q, n_heads, head_size,
                             rope_dims, rope_cos, rope_sin);
            apply_rope_heads(key_cache_row, c->n_kv_heads, head_size,
                             rope_dims, rope_cos, rope_sin);

            // GQA attention
            {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_dim, c->seq_len };
#ifdef __ARM_NEON
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_neon_range : bn_transformer_gqa_neon_range;
#elif defined(__AVX2__)
                bn_tp_fn attn_fn = bn_transformer_gqa_avx2_range;
#elif defined(__wasm_simd128__)
                bn_tp_fn attn_fn = bn_transformer_gqa_wasm_range;
#else
                bn_tp_fn attn_fn = bn_transformer_gqa_scalar_range;
#endif
                BnTPTask gqa = { attn_fn, &gctx, n_heads };
                bn_tp_dispatch(m->pool, &gqa, 1);
            }

            // No sigmoid gate (unlike gated-Q path)

            // wo projection (q_dim → dim) + residual
            if (lw->attn_sub_norm)
                rmsnorm(s->xb, s->xb, lw->attn_sub_norm, q_dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->wo }};
                bn_quant_matvec_batch(wo, 1, s->xb, s->x_q, m->pool);
            }
            residual_add(s->x, s->xb2, dim);

        } else {
            // --- Classic attention path (existing) ---
            float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_dim;
            float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_dim;

            if (c->kv_f16) {
                float *k_tmp = s->hb, *v_tmp = s->hb2;
                BnMatvecTask qkv[3] = {
                    { s->q,  &lw->wq },
                    { k_tmp, &lw->wk },
                    { v_tmp, &lw->wv },
                };
                bn_quant_matvec_batch(qkv, 3, s->xb, s->x_q, m->pool);
                BL_ACC(bl_matvec_qkv_us);

                if (lw->q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->q_bias[i];
                if (lw->k_bias) for (int i = 0; i < kv_dim; i++) k_tmp[i] += lw->k_bias[i];
                if (lw->v_bias) for (int i = 0; i < kv_dim; i++) v_tmp[i] += lw->v_bias[i];

                if (lw->q_norm)
                    for (int h = 0; h < n_heads; h++)
                        rmsnorm(s->q + h*head_size, s->q + h*head_size,
                                lw->q_norm + h*head_size, head_size, c->norm_eps);
                if (lw->k_norm)
                    for (int h = 0; h < c->n_kv_heads; h++)
                        rmsnorm(k_tmp + h*head_size, k_tmp + h*head_size,
                                lw->k_norm + h*head_size, head_size, c->norm_eps);

                apply_rope_heads(s->q, n_heads, head_size,
                                 rope_dims, rope_cos, rope_sin);
                apply_rope_heads(k_tmp, c->n_kv_heads, head_size,
                                 rope_dims, rope_cos, rope_sin);
                BL_ACC(bl_rope_us);

                uint16_t *kc = (uint16_t *)s->key_cache   + loff + (size_t)cache_pos * kv_dim;
                uint16_t *vc = (uint16_t *)s->value_cache + loff + (size_t)cache_pos * kv_dim;
#ifdef __ARM_NEON
                for (int i = 0; i < kv_dim; i += 4) {
                    float32x4_t kv4 = vld1q_f32(k_tmp + i);
                    float16x4_t kh4 = vcvt_f16_f32(kv4);
                    vst1_u16(kc + i, vreinterpret_u16_f16(kh4));
                    float32x4_t vv4 = vld1q_f32(v_tmp + i);
                    float16x4_t vh4 = vcvt_f16_f32(vv4);
                    vst1_u16(vc + i, vreinterpret_u16_f16(vh4));
                }
#elif defined(__AVX2__)
                for (int i = 0; i < kv_dim; i += 8) {
                    _mm_storeu_si128((__m128i *)(kc + i), _mm256_cvtps_ph(_mm256_loadu_ps(k_tmp + i), _MM_FROUND_TO_NEAREST_INT));
                    _mm_storeu_si128((__m128i *)(vc + i), _mm256_cvtps_ph(_mm256_loadu_ps(v_tmp + i), _MM_FROUND_TO_NEAREST_INT));
                }
#else
                for (int i = 0; i < kv_dim; i++) {
                    kc[i] = bn_fp32_to_fp16(k_tmp[i]);
                    vc[i] = bn_fp32_to_fp16(v_tmp[i]);
                }
#endif
            } else {
                BnMatvecTask qkv[3] = {
                    { s->q,            &lw->wq },
                    { key_cache_row,   &lw->wk },
                    { value_cache_row, &lw->wv },
                };
                bn_quant_matvec_batch(qkv, 3, s->xb, s->x_q, m->pool);
                BL_ACC(bl_matvec_qkv_us);

                if (lw->q_bias) for (int i = 0; i < dim; i++) s->q[i] += lw->q_bias[i];
                if (lw->k_bias) for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->k_bias[i];
                if (lw->v_bias) for (int i = 0; i < kv_dim; i++) value_cache_row[i] += lw->v_bias[i];

                if (lw->q_norm)
                    for (int h = 0; h < n_heads; h++)
                        rmsnorm(s->q + h*head_size, s->q + h*head_size,
                                lw->q_norm + h*head_size, head_size, c->norm_eps);
                if (lw->k_norm)
                    for (int h = 0; h < c->n_kv_heads; h++)
                        rmsnorm(key_cache_row + h*head_size, key_cache_row + h*head_size,
                                lw->k_norm + h*head_size, head_size, c->norm_eps);

                apply_rope_heads(s->q, n_heads, head_size,
                                 rope_dims, rope_cos, rope_sin);
                apply_rope_heads(key_cache_row, c->n_kv_heads, head_size,
                                 rope_dims, rope_cos, rope_sin);
                BL_ACC(bl_rope_us);
            }

            // GQA attention
            {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_dim, c->seq_len };
#ifdef __ARM_NEON
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_neon_range : bn_transformer_gqa_neon_range;
#elif defined(__AVX2__)
                bn_tp_fn attn_fn = bn_transformer_gqa_avx2_range;
#elif defined(__wasm_simd128__)
                bn_tp_fn attn_fn = bn_transformer_gqa_wasm_range;
#else
                bn_tp_fn attn_fn = bn_transformer_gqa_scalar_range;
#endif
                BnTPTask gqa = { attn_fn, &gctx, n_heads };
                bn_tp_dispatch(m->pool, &gqa, 1);
            }
            BL_ACC(bl_gqa_us);

            // Attention sub-norm + wo projection + residual
            if (lw->attn_sub_norm)
                rmsnorm(s->xb, s->xb, lw->attn_sub_norm, dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->wo }};
                bn_quant_matvec_batch(wo, 1, s->xb, s->x_q, m->pool);
            }
            BL_ACC(bl_matvec_qkv_us);
            residual_add(s->x, s->xb2, dim);
            BL_ACC(bl_residual_us);
        }

    } else {
        // ---- SSM block ----
        forward_ssm_block(m, lw, l);
        residual_add(s->x, s->xb, dim);
        BL_ACC(bl_residual_us);
    }

    // ---- FFN block ---- (shared by both layer types)
    if (lw->router_weight) {
        // MoE FFN — route, pread, compute, combine
        bn_moe_forward(m, lw, l);
    } else {
        // Dense FFN
        forward_ffn_block(m, lw);
    }
    BL_ACC(bl_ffn_us);

    (void)is_attn; // used only in debug builds

#ifdef BN_BENCH_LAYERS
    bl_layer_count++;
#endif

    return 0;
}

// Embed + all layers (attention + FFN). Populates KV cache at `pos`.
// Leaves final activation in s->x. Returns 0 on success, -1 on error.
static int forward_layers(BnModel *m, int token, int pos) {
    BnConfig *c = &m->config;
    BnRunState *s = &m->state;
    int head_size = c->head_size;

    // Guard against stack overflow from VLAs sized by model config
    if (head_size > BN_MAX_VLA_ELEMS || c->dim > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("Model dimensions too large for stack VLAs");
        return -1;
    }

    // #9: Validate token bounds
    if (token < 0 || token >= c->vocab_size) {
        SH_LOG_ERROR("Token out of range");
        return -1;
    }

    // #10: Validate pos bounds
    if (pos < 0) {
        SH_LOG_ERROR("Position out of range");
        return -1;
    }

    // Embed the token
    bn_model_embed_token(m, s->x, token);

    // Precompute RoPE cos/sin for this position
    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : head_size;
    int half_rope = rope_dims / 2;
    float rope_cos[half_rope], rope_sin[half_rope];
    for (int i = 0; i < half_rope; i++) {
        float angle = pos * s->rope_freq[i];
        rope_cos[i] = cosf(angle);
        rope_sin[i] = sinf(angle);
    }

    // Process each layer
    int cache_pos = pos % c->seq_len;
    for (int l = 0; l < c->n_layers; l++) {
        if (forward_single_layer(m, l, pos, cache_pos, rope_dims,
                                 rope_cos, rope_sin) != 0)
            return -1;
    }

    return 0;
}

// Final RMSNorm + logits computation. Reads s->x, writes s->logits.
// Returns s->logits.
static float *forward_logits(BnModel *m) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &m->state;
    int dim = c->dim;

    if (dim > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("Model dim too large for stack VLAs");
        return NULL;
    }

    BL_START();

    // Final RMSNorm
    rmsnorm(s->x, s->x, w->output_norm, dim, c->norm_eps);

    // Untied output weight: logits = output_weight @ x
    if (w->output_weight.data && w->output_weight.type == BN_GGUF_TENSOR_F16) {
        int n_rows = w->output_weight.rows;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (w->emb_out_i8) {
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { bn_transformer_logits_i8_neon_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
        } else
#elif defined(__AVX2__)
        if (w->emb_out_i8) {
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { bn_transformer_logits_i8_avx2_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
        } else
#endif
        {
            const uint16_t *emb = (const uint16_t *)w->output_weight.data;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            uint16_t x_f16[dim];
            for (int d = 0; d < dim; d += 8) {
                float16x4_t lo = vcvt_f16_f32(vld1q_f32(s->x + d));
                float16x4_t hi = vcvt_f16_f32(vld1q_f32(s->x + d + 4));
                vst1q_u16(x_f16 + d, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
            }
            BnLogitsCtx lctx = { s->logits, (const float *)(void *)x_f16, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_native_neon_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__ARM_NEON)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_neon_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__AVX2__)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_avx2_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__wasm_simd128__)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_wasm_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#else
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_scalar_range, &lctx, n_rows };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#endif
        }
    }
    else if (w->output_weight.data) {
        bn_quant_matvec(s->logits, &w->output_weight, s->x, s->x_q, m->pool);
    }
    // Tied Q4_0/Q8_0/Q6_K embeddings: use quant matvec
    else if (w->emb_type == BN_GGUF_TENSOR_Q4_0 || w->emb_type == BN_GGUF_TENSOR_Q8_0 ||
             w->emb_type == BN_GGUF_TENSOR_Q6_K) {
        BnQWeight tied = { w->token_embedding, w->emb_type, c->vocab_size, dim, 1.0f, NULL, NULL };
        bn_quant_matvec(s->logits, &tied, s->x, s->x_q, m->pool);
    }
    // Tied F16 embeddings: logits = token_embedding^T @ x
    else if (w->emb_type == BN_GGUF_TENSOR_F16) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (w->emb_out_i8) {
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { bn_transformer_logits_i8_neon_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
        } else
#elif defined(__AVX2__)
        if (w->emb_out_i8) {
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { bn_transformer_logits_i8_avx2_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
        } else
#endif
        {
            const uint16_t *emb = (const uint16_t *)w->token_embedding;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            uint16_t x_f16[dim];
            for (int d = 0; d < dim; d += 8) {
                float16x4_t lo = vcvt_f16_f32(vld1q_f32(s->x + d));
                float16x4_t hi = vcvt_f16_f32(vld1q_f32(s->x + d + 4));
                vst1q_u16(x_f16 + d, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
            }
            BnLogitsCtx lctx = { s->logits, (const float *)(void *)x_f16, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_native_neon_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__ARM_NEON)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_neon_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__AVX2__)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_avx2_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__wasm_simd128__)
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_wasm_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#else
            BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
            BnTPTask logits_task = { bn_transformer_logits_f16_scalar_range, &lctx, c->vocab_size };
            bn_tp_dispatch(m->pool, &logits_task, 1);
#endif
        }
    } else {
        // F32 embeddings
        const float *emb = (const float *)w->token_embedding;
        BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
        BnTPTask logits_task = { bn_transformer_logits_f32_range, &lctx, c->vocab_size };
        bn_tp_dispatch(m->pool, &logits_task, 1);
    }

    BL_ACC(bl_logits_us);
#ifdef BN_BENCH_LAYERS
    bl_print_reset();
#endif

    return s->logits;
}

float *bn_transformer_forward(BnModel *m, int token, int pos) {
    if (forward_layers(m, token, pos) != 0) return NULL;
    return forward_logits(m);
}

// Internal prefill: if all_logits is non-NULL, compute logits at every position.
static float *prefill_internal(BnModel *m, const int *tokens, int n_tokens,
                                int pos0, float *all_logits) {
    if (n_tokens <= 0) return NULL;
    if (n_tokens == 1) return bn_transformer_forward(m, tokens[0], pos0);

    BnConfig *c = &m->config;

    // MoE models use batch MoE prefill (token-expert grouping + per-expert matmul)
    // Falls through to the batched layer loop which calls bn_moe_forward_batch
    BnRunState *s = &m->state;
    int dim = c->dim;
    int head_size = c->head_size;

    if (head_size > BN_MAX_VLA_ELEMS || dim > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("Model dimensions too large for stack VLAs");
        return NULL;
    }

    // Validate all tokens
    for (int t = 0; t < n_tokens; t++) {
        if (tokens[t] < 0 || tokens[t] >= c->vocab_size) {
            SH_LOG_ERROR("Token out of range");
            return NULL;
        }
    }
    if (pos0 < 0) {
        SH_LOG_ERROR("Position out of range");
        return NULL;
    }

    // Activation buffer: n_tokens × dim
    size_t act_elems = (size_t)n_tokens * dim;
    if (act_elems / n_tokens != (size_t)dim) {
        SH_LOG_ERROR("Prefill activation buffer size overflow");
        return NULL;
    }
    float *act = (float *)malloc(act_elems * sizeof(float));
    if (!act) return NULL;

    // Embed all tokens
    for (int t = 0; t < n_tokens; t++)
        bn_model_embed_token(m, act + (size_t)t * dim, tokens[t]);

    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : head_size;
    int half_rope = rope_dims / 2;
    if (half_rope > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("RoPE dimensions too large for stack VLAs");
        free(act);
        return NULL;
    }

    // Allocate batch scratch buffers
    int kv_dim = c->kv_dim;
    int hidden_dim = c->hidden_dim;
    int q_dim = c->n_heads * head_size;
    size_t nt = (size_t)n_tokens;

    // Batch buffers: Xb[nt*dim], Q[nt*q_dim], Hb/Hb2[nt*hidden_dim]
    size_t batch_size = nt * dim          // xb (normed)
                      + nt * (size_t)(q_dim > dim ? q_dim * 2 : dim)  // q (may be wide/gated)
                      + nt * kv_dim * 2   // k_new, v_new
                      + nt * dim          // xb2 (wo output)
                      + nt * hidden_dim   // hb (gate)
                      + nt * hidden_dim;  // hb2 (up)
    float *batch_buf = (float *)malloc(batch_size * sizeof(float));
    if (!batch_buf) { free(act); return NULL; }

    // Carve out pointers
    float *Xb   = batch_buf;
    float *Q_buf = Xb + nt * dim;
    int q_buf_stride = (q_dim > dim ? q_dim * 2 : dim);
    float *K_new = Q_buf + nt * q_buf_stride;
    float *V_new = K_new + nt * kv_dim;
    float *Xb2  = V_new + nt * kv_dim;
    float *Hb   = Xb2 + nt * dim;
    float *Hb2  = Hb + nt * hidden_dim;

    BnWeights *w = &m->weights;

    // Layer-by-layer, batched projections, per-token attention
    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        int is_attn = (c->full_attn_interval == 0) ||
                      ((l + 1) % c->full_attn_interval == 0);

        // --- Attention block (batched projections, per-token GQA) ---
        if (is_attn && lw->wq.data) {
            // Batch RMSNorm
            for (int t = 0; t < n_tokens; t++)
                rmsnorm(Xb + t * dim, act + (size_t)t * dim, lw->attn_norm, dim, c->norm_eps);

            // Batch QKV matmul
            bn_quant_matmul(Q_buf, &lw->wq, Xb, n_tokens, s->x_q, m->pool);
            bn_quant_matmul(K_new, &lw->wk, Xb, n_tokens, s->x_q, m->pool);
            bn_quant_matmul(V_new, &lw->wv, Xb, n_tokens, s->x_q, m->pool);

            // Per-token: RoPE, KV cache write, GQA attention
            int attn_idx = (c->full_attn_interval > 0)
                ? (l + 1) / c->full_attn_interval - 1 : l;
            size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;
            int kv_mul = c->kv_mul;
            int q_gated = lw->wq.rows > q_dim;

            for (int t = 0; t < n_tokens; t++) {
                int pos = pos0 + t;
                int cache_pos = pos % c->seq_len;

                float *q_t = Q_buf + (size_t)t * lw->wq.rows;
                float *k_t = K_new + (size_t)t * kv_dim;
                float *v_t = V_new + (size_t)t * kv_dim;

                // Extract Q for this token (handle gated/wide/classic)
                if (q_gated) {
                    for (int h = 0; h < c->n_heads; h++)
                        memcpy(s->q + h * head_size, q_t + h * 2 * head_size,
                               head_size * sizeof(float));
                } else {
                    memcpy(s->q, q_t, q_dim * sizeof(float));
                }

                // Q/K norms
                if (lw->q_norm) {
                    int qk_stride = c->qk_norm_per_head ? head_size : 0;
                    for (int h = 0; h < c->n_heads; h++)
                        rmsnorm(s->q + h*head_size, s->q + h*head_size,
                                lw->q_norm + h*qk_stride, head_size, c->norm_eps);
                }
                if (lw->k_norm) {
                    int qk_stride = c->qk_norm_per_head ? head_size : 0;
                    for (int h = 0; h < c->n_kv_heads; h++)
                        rmsnorm(k_t + h*head_size, k_t + h*head_size,
                                lw->k_norm + h*qk_stride, head_size, c->norm_eps);
                }

                // Biases
                if (lw->q_bias) for (int i = 0; i < q_dim; i++) s->q[i] += lw->q_bias[i];
                if (lw->k_bias) for (int i = 0; i < kv_dim; i++) k_t[i] += lw->k_bias[i];
                if (lw->v_bias) for (int i = 0; i < kv_dim; i++) v_t[i] += lw->v_bias[i];

                // RoPE
                float rope_cos_t[half_rope], rope_sin_t[half_rope];
                for (int i = 0; i < half_rope; i++) {
                    float angle = pos * s->rope_freq[i];
                    rope_cos_t[i] = cosf(angle);
                    rope_sin_t[i] = sinf(angle);
                }
                apply_rope_heads(s->q, c->n_heads, head_size,
                                 rope_dims, rope_cos_t, rope_sin_t);
                apply_rope_heads(k_t, c->n_kv_heads, head_size,
                                 rope_dims, rope_cos_t, rope_sin_t);

                // Write KV cache
                if (c->kv_f16) {
                    uint16_t *kc = (uint16_t *)s->key_cache + loff + (size_t)cache_pos * kv_dim;
                    uint16_t *vc = (uint16_t *)s->value_cache + loff + (size_t)cache_pos * kv_dim;
                    for (int i = 0; i < kv_dim; i++) {
                        kc[i] = bn_fp32_to_fp16(k_t[i]);
                        vc[i] = bn_fp32_to_fp16(v_t[i]);
                    }
                } else {
                    float *kc = s->key_cache + loff + (size_t)cache_pos * kv_dim;
                    float *vc = s->value_cache + loff + (size_t)cache_pos * kv_dim;
                    memcpy(kc, k_t, kv_dim * sizeof(float));
                    memcpy(vc, v_t, kv_dim * sizeof(float));
                }

                // GQA attention (same as single-token path)
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_dim, c->seq_len };
#ifdef __ARM_NEON
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_neon_range : bn_transformer_gqa_neon_range;
#elif defined(__AVX2__)
                bn_tp_fn attn_fn = bn_transformer_gqa_avx2_range;
#elif defined(__wasm_simd128__)
                bn_tp_fn attn_fn = bn_transformer_gqa_wasm_range;
#else
                bn_tp_fn attn_fn = bn_transformer_gqa_scalar_range;
#endif
                BnTPTask gqa = { attn_fn, &gctx, c->n_heads };
                bn_tp_dispatch(m->pool, &gqa, 1);

                // Sigmoid gate (gated Q only)
                if (q_gated) {
                    for (int h = 0; h < c->n_heads; h++) {
                        float *gate_h = q_t + h * 2 * head_size + head_size;
                        float *xb_h = s->xb + h * head_size;
                        for (int d = 0; d < head_size; d++)
                            xb_h[d] *= 1.0f / (1.0f + expf(-gate_h[d]));
                    }
                }

                // Store attention output for batch Wo
                // For wide-Q: GQA outputs q_dim elements; for classic: dim elements
                int wo_cols = lw->wo.cols;
                memcpy(Q_buf + (size_t)t * wo_cols, s->xb, wo_cols * sizeof(float));
            }

            // Batch sub-norm + Wo matmul (use Q_buf as Wo input, correctly strided)
            {
                int wo_cols = lw->wo.cols;
                if (lw->attn_sub_norm)
                    for (int t = 0; t < n_tokens; t++)
                        rmsnorm(Q_buf + (size_t)t * wo_cols, Q_buf + (size_t)t * wo_cols,
                                lw->attn_sub_norm, wo_cols, c->norm_eps);
                bn_quant_matmul(Xb2, &lw->wo, Q_buf, n_tokens, s->x_q, m->pool);
            }

            // Batch residual add
            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb2[(size_t)t * dim + d];

        } else if (!is_attn) {
            // --- SSM block: must process sequentially ---
            for (int t = 0; t < n_tokens; t++) {
                memcpy(s->x, act + (size_t)t * dim, dim * sizeof(float));
                float rope_cos_t[half_rope], rope_sin_t[half_rope];
                int pos = pos0 + t;
                for (int i = 0; i < half_rope; i++) {
                    float angle = pos * s->rope_freq[i];
                    rope_cos_t[i] = cosf(angle);
                    rope_sin_t[i] = sinf(angle);
                }
                int cache_pos = pos % c->seq_len;
                if (forward_single_layer(m, l, pos, cache_pos, rope_dims,
                                         rope_cos_t, rope_sin_t) != 0) {
                    free(batch_buf); free(act); return NULL;
                }
                memcpy(act + (size_t)t * dim, s->x, dim * sizeof(float));
            }
            continue;  // SSM already did FFN
        }

        // --- FFN block ---
        if (lw->router_weight) {
            // Batch MoE: route all tokens, group by expert, batch matmul
            if (bn_moe_forward_batch(m, lw, l, act, Xb, n_tokens) != 0) {
                free(batch_buf); free(act); return NULL;
            }
        } else if (lw->ffn_up.data) {
            // Dense FFN: batched matmul
            // Batch RMSNorm
            for (int t = 0; t < n_tokens; t++)
                rmsnorm(Xb + t * dim, act + (size_t)t * dim, lw->ffn_norm, dim, c->norm_eps);

            if (c->has_ffn_gate) {
                bn_quant_matmul(Hb, &lw->ffn_gate, Xb, n_tokens, s->x_q, m->pool);
                bn_quant_matmul(Hb2, &lw->ffn_up, Xb, n_tokens, s->x_q, m->pool);

                // Batch activation
                for (int t = 0; t < n_tokens; t++) {
                    float *hb_t = Hb + (size_t)t * hidden_dim;
                    float *hb2_t = Hb2 + (size_t)t * hidden_dim;
                    if (c->act_type == 1) {
                        for (int i = 0; i < hidden_dim; i++) {
                            float g = hb_t[i] > 0 ? hb_t[i] : 0;
                            hb_t[i] = g * g * hb2_t[i];
                        }
                    } else {
                        for (int i = 0; i < hidden_dim; i++) {
                            float g = hb_t[i];
                            hb_t[i] = (g / (1.0f + expf(-g))) * hb2_t[i];
                        }
                    }
                }
            } else {
                bn_quant_matmul(Hb, &lw->ffn_up, Xb, n_tokens, s->x_q, m->pool);
                for (int t = 0; t < n_tokens; t++) {
                    float *hb_t = Hb + (size_t)t * hidden_dim;
                    if (c->act_type == 1) {
                        for (int i = 0; i < hidden_dim; i++) {
                            float v = hb_t[i] > 0 ? hb_t[i] : 0;
                            hb_t[i] = v * v;
                        }
                    } else {
                        for (int i = 0; i < hidden_dim; i++) {
                            float v = hb_t[i];
                            hb_t[i] = v / (1.0f + expf(-v));
                        }
                    }
                }
            }

            if (lw->ffn_sub_norm)
                for (int t = 0; t < n_tokens; t++)
                    rmsnorm(Hb + (size_t)t * hidden_dim, Hb + (size_t)t * hidden_dim,
                            lw->ffn_sub_norm, hidden_dim, c->norm_eps);

            bn_quant_matmul(Xb, &lw->ffn_down, Hb, n_tokens, s->x_q, m->pool);

            // Batch residual add
            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb[(size_t)t * dim + d];
        }
    }

    // Compute logits for all positions (if requested) or just the last
    if (all_logits) {
        int vocab_size = c->vocab_size;
        for (int t = 0; t < n_tokens; t++) {
            memcpy(s->x, act + (size_t)t * dim, dim * sizeof(float));
            float *lg = forward_logits(m);
            if (!lg) { free(batch_buf); free(act); return NULL; }
            memcpy(all_logits + (size_t)t * vocab_size, lg, vocab_size * sizeof(float));
        }
    } else {
        memcpy(s->x, act + (size_t)(n_tokens - 1) * dim, dim * sizeof(float));
    }

    free(batch_buf);
    free(act);
    return forward_logits(m);
}

float *bn_transformer_prefill(BnModel *m, const int *tokens, int n_tokens, int pos0) {
    return prefill_internal(m, tokens, n_tokens, pos0, NULL);
}

int bn_transformer_prefill_all(BnModel *m, const int *tokens, int n_tokens,
                                int pos0, float *all_logits) {
    if (!all_logits || n_tokens <= 0) return -1;

    // Single token: just forward
    if (n_tokens == 1) {
        float *logits = bn_transformer_forward(m, tokens[0], pos0);
        if (!logits) return -1;
        memcpy(all_logits, logits, m->config.vocab_size * sizeof(float));
        return 0;
    }

    float *result = prefill_internal(m, tokens, n_tokens, pos0, all_logits);
    return result ? 0 : -1;
}
