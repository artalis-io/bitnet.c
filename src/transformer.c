#include "transformer_internal.h"
#include "turboquant.h"
#include "quant_internal.h"
#include "gpu_backend.h"
#include "gpu_moe_cache.h"
#include "moe.h"
#include "session.h"
#include "platform.h"
#include "sh_log.h"
#include "sh_arena.h"
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

// TQ KV write: quantize float K/V vectors into compressed TQ cache.
// k_tmp/v_tmp: [kv_dim] float vectors after norms + RoPE.
// attn_idx: layer index among attention layers.
// cache_pos: position within seq_len (modular).
static inline void tq_write_kv(const BnTQState *tq, BnRunState *s,
                                const float *k_tmp, const float *v_tmp,
                                int n_kv_heads, int head_size,
                                int attn_idx, int cache_pos, int seq_len) {
    int key_bytes = bn_tq_key_bytes(tq);
    int val_bytes = bn_tq_value_bytes(tq);
    size_t tq_loff_k = (size_t)attn_idx * seq_len * n_kv_heads * key_bytes;
    size_t tq_loff_v = (size_t)attn_idx * seq_len * n_kv_heads * val_bytes;
    uint8_t *kc_tq = s->key_cache_tq   + tq_loff_k + (size_t)cache_pos * n_kv_heads * key_bytes;
    uint8_t *vc_tq = s->value_cache_tq + tq_loff_v + (size_t)cache_pos * n_kv_heads * val_bytes;
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
        bn_tq_quantize_key(tq, k_tmp + kv_h * head_size, kc_tq + kv_h * key_bytes);
        bn_tq_quantize_value(tq, v_tmp + kv_h * head_size, vc_tq + kv_h * val_bytes);
    }
}

// TQ GQA dispatch: set up context and dispatch TQ attention.
static inline void tq_gqa_dispatch(BnModel *m, BnRunState *s,
                                    int attn_idx, int pos, int n_heads,
                                    int n_kv_heads, int head_size, int kv_mul) {
    const BnConfig *c = &m->config;
    const BnTQState *tq = m->tq_state;
    int key_bytes = bn_tq_key_bytes(tq);
    int val_bytes = bn_tq_value_bytes(tq);
    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;

    size_t tq_loff_k = (size_t)attn_idx * c->seq_len * n_kv_heads * key_bytes;
    size_t tq_loff_v = (size_t)attn_idx * c->seq_len * n_kv_heads * val_bytes;

    BnGQATQCtx tctx = {
        .c = c, .s = s, .tq = tq,
        .tq_keys = s->key_cache_tq + tq_loff_k,
        .tq_values = s->value_cache_tq + tq_loff_v,
        .key_stride = n_kv_heads * key_bytes,
        .val_stride = n_kv_heads * val_bytes,
        .key_bytes = key_bytes,
        .val_bytes = val_bytes,
        .pos = pos, .n_kv = n_kv, .kv_mul = kv_mul,
        .head_size = head_size, .seq_len = c->seq_len,
        .n_kv_heads = n_kv_heads
    };
#ifdef __ARM_NEON
    bn_tp_fn attn_fn = bn_transformer_gqa_tq_neon_range;
#else
    bn_tp_fn attn_fn = bn_transformer_gqa_tq_scalar_range;
#endif
    BnTPTask gqa = { attn_fn, &tctx, n_heads };
    bn_tp_dispatch(m->pool, &gqa, 1);
}

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
static void forward_ssm_block(BnModel *m, BnSession *sess, BnLayerWeights *lw, int l) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
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
        bn_quant_matvec_batch_gpu(tasks, 2, s->xb, s->x_q, m->pool, m->gpu);
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
        bn_quant_matvec_batch_gpu(ab, 2, s->xb, s->x_q, m->pool, m->gpu);
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
        bn_quant_matvec_batch_gpu(proj, 1, out, s->x_q, m->pool, m->gpu);
    }
    BL_ACC(bl_matvec_qkv_us);
}

// FFN block: shared by both attention and SSM layers.
// Reads s->x, uses s->xb/hb/hb2/x_q as scratch. Adds result to s->x.
static void forward_ffn_block(BnModel *m, BnSession *sess, BnLayerWeights *lw) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;

    /* Fused RMSNorm + Q8K quantization on AVX2 for k-quant FFN weights:
     * saves 2 full passes over dim by combining norm scaling with Q8K
     * amax-finding and quantization. Falls back to separate path for GPU
     * or non-k-quant weight types. */
    int fused_gate_up = 0;
#ifdef __AVX2__
    if (c->has_ffn_gate && !m->gpu && dim % BN_QK_K == 0 &&
        (lw->ffn_gate.type == BN_GGUF_TENSOR_Q4_K ||
         lw->ffn_gate.type == BN_GGUF_TENSOR_Q6_K)) {
        int n_sb = dim / BN_QK_K;
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_rmsnorm_q8k_avx2(s->x, lw->ffn_norm, dim, c->norm_eps,
                                    s->xb, s->x_q, q8k_d, q8k_bsums);
        BnMatvecTask ffn[2] = {
            { s->hb,  &lw->ffn_gate },
            { s->hb2, &lw->ffn_up   },
        };
        bn_quant_matvec_batch_preq8k(ffn, 2, s->x_q, q8k_d, q8k_bsums, s->xb, m->pool);
        fused_gate_up = 1;
    }
#endif
    if (!fused_gate_up) {
        rmsnorm(s->xb, s->x, lw->ffn_norm, dim, c->norm_eps);

        if (c->has_ffn_gate) {
            BnMatvecTask ffn[2] = {
                { s->hb,  &lw->ffn_gate },
                { s->hb2, &lw->ffn_up   },
            };
            bn_quant_matvec_batch_gpu(ffn, 2, s->xb, s->x_q, m->pool, m->gpu);
        } else {
            BnMatvecTask ffn[1] = {{ s->hb, &lw->ffn_up }};
            bn_quant_matvec_batch_gpu(ffn, 1, s->xb, s->x_q, m->pool, m->gpu);
        }
    }

    /* Activation function (shared by fused and non-fused paths) */
    if (c->has_ffn_gate) {
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
        bn_quant_matvec_batch_gpu(down, 1, s->hb, s->x_q, m->pool, m->gpu);
    }

    residual_add(s->x, s->xb, dim);
}

// Process a single layer (attention/SSM block + FFN). Reads/writes s->x.
// Returns 0 on success.
static int forward_single_layer(BnModel *m, BnSession *sess, int l, int pos, int cache_pos,
                                int rope_dims, const float *rope_cos,
                                const float *rope_sin) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &sess->state;
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

        /* Fused attn RMSNorm + Q8K: quantize s->xb once, reuse for Q and K+V */
        int attn_preq8k = 0;
#ifdef __AVX2__
        int attn_kquant = (lw->wq.type == BN_GGUF_TENSOR_Q4_K ||
                           lw->wq.type == BN_GGUF_TENSOR_Q6_K) &&
                          !m->gpu && dim % BN_QK_K == 0;
#else
        int attn_kquant = 0;
#endif
        int n_sb_attn = dim / BN_QK_K;
        float attn_q8k_d[n_sb_attn > 0 ? n_sb_attn : 1];
        int16_t attn_q8k_bsums[n_sb_attn > 0 ? n_sb_attn * 16 : 1];
#ifdef __AVX2__
        if (attn_kquant) {
            bn_quant_rmsnorm_q8k_avx2(s->x, lw->attn_norm, dim, c->norm_eps,
                                        s->xb, s->x_q, attn_q8k_d, attn_q8k_bsums);
            attn_preq8k = 1;
        } else
#endif
        {
            rmsnorm(s->xb, s->x, lw->attn_norm, dim, c->norm_eps);
        }
        BL_ACC(bl_rmsnorm_us);

        (void)0;

        if (q_gated) {
            // --- Gated Q path (Qwen3.5 attention) ---
            float *q_full = s->hb;  // [2*dim]
            float *k_tmp = s->hb2;
            float *v_tmp = s->hb2 + kv_dim;

            // Q matvec (reuse cached Q8K if available)
            {
                BnMatvecTask q_task[1] = {{ q_full, &lw->wq }};
                if (attn_preq8k)
                    bn_quant_matvec_batch_preq8k(q_task, 1, s->x_q, attn_q8k_d, attn_q8k_bsums, s->xb, m->pool);
                else
                    bn_quant_matvec_batch_gpu(q_task, 1, s->xb, s->x_q, m->pool, m->gpu);
            }
            // K+V matvecs (reuse same cached Q8K)
            if (!(c->kv_tq_bits > 0 && m->tq_state) && !c->kv_f16) {
                float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_dim;
                float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_dim;
                BnMatvecTask kv[2] = {
                    { key_cache_row,   &lw->wk },
                    { value_cache_row, &lw->wv },
                };
                if (attn_preq8k)
                    bn_quant_matvec_batch_preq8k(kv, 2, s->x_q, attn_q8k_d, attn_q8k_bsums, s->xb, m->pool);
                else
                    bn_quant_matvec_batch_gpu(kv, 2, s->xb, s->x_q, m->pool, m->gpu);
                k_tmp = key_cache_row;
                v_tmp = value_cache_row;
            } else {
                BnMatvecTask kv[2] = {
                    { k_tmp, &lw->wk },
                    { v_tmp, &lw->wv },
                };
                if (attn_preq8k)
                    bn_quant_matvec_batch_preq8k(kv, 2, s->x_q, attn_q8k_d, attn_q8k_bsums, s->xb, m->pool);
                else
                    bn_quant_matvec_batch_gpu(kv, 2, s->xb, s->x_q, m->pool, m->gpu);
            }
            BL_ACC(bl_matvec_qkv_us);

            /* Extract Q from interleaved [Q, gate] and optionally apply Q norm.
             * Fused: copy from q_full stride-2hs directly into rmsnorm if norm exists,
             * avoiding a separate memcpy + reload. */
            if (lw->q_norm) {
                for (int h = 0; h < n_heads; h++)
                    rmsnorm(s->q + h*head_size,
                            q_full + h * 2 * head_size,
                            lw->q_norm + h*qk_stride, head_size, c->norm_eps);
            } else {
                for (int h = 0; h < n_heads; h++)
                    memcpy(s->q + h * head_size,
                           q_full + h * 2 * head_size,
                           head_size * sizeof(float));
            }
            if (lw->k_norm)
                for (int h = 0; h < c->n_kv_heads; h++)
                    rmsnorm(k_tmp + h*head_size, k_tmp + h*head_size,
                            lw->k_norm + h*qk_stride, head_size, c->norm_eps);

            apply_rope_heads(s->q, n_heads, head_size,
                             rope_dims, rope_cos, rope_sin);
            apply_rope_heads(k_tmp, c->n_kv_heads, head_size,
                             rope_dims, rope_cos, rope_sin);
            BL_ACC(bl_rope_us);

            // Write KV + GQA
            if (c->kv_tq_bits > 0 && m->tq_state) {
                tq_write_kv(m->tq_state, s, k_tmp, v_tmp,
                            c->n_kv_heads, head_size, attn_idx, cache_pos, c->seq_len);
                tq_gqa_dispatch(m, s, attn_idx, pos, n_heads,
                                c->n_kv_heads, head_size, kv_mul);
            } else if (c->kv_f16) {
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
            }
            // FP32 path already wrote to cache directly

            if (!(c->kv_tq_bits > 0 && m->tq_state)) {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_dim, c->seq_len };
#ifdef __ARM_NEON
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_neon_range : bn_transformer_gqa_neon_range;
#elif defined(__AVX2__)
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_avx2_range : bn_transformer_gqa_avx2_range;
#elif defined(__wasm_simd128__)
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_wasm_range : bn_transformer_gqa_wasm_range;
#else
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_scalar_range : bn_transformer_gqa_scalar_range;
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
                bn_quant_matvec_batch_gpu(wo, 1, s->xb, s->x_q, m->pool, m->gpu);
            }
            BL_ACC(bl_matvec_qkv_us);
            residual_add(s->x, s->xb2, dim);
            BL_ACC(bl_residual_us);

        } else if (q_wide) {
            // --- Wide Q path (Qwen3 MoE: head_size > dim/n_heads, no gate) ---
            float *k_tmp = s->hb, *v_tmp = s->hb2;

            // Q matvec: xb[dim] → q[q_dim]
            {
                BnMatvecTask q_task[1] = {{ s->q, &lw->wq }};
                bn_quant_matvec_batch_gpu(q_task, 1, s->xb, s->x_q, m->pool, m->gpu);
            }
            // K/V matvec: xb[dim] → kv_dim (always to temp buffers for TQ compat)
            if (c->kv_tq_bits > 0 && m->tq_state) {
                BnMatvecTask kv[2] = {
                    { k_tmp, &lw->wk },
                    { v_tmp, &lw->wv },
                };
                bn_quant_matvec_batch_gpu(kv, 2, s->xb, s->x_q, m->pool, m->gpu);
            } else {
                float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_dim;
                float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_dim;
                BnMatvecTask kv[2] = {
                    { key_cache_row,   &lw->wk },
                    { value_cache_row, &lw->wv },
                };
                bn_quant_matvec_batch_gpu(kv, 2, s->xb, s->x_q, m->pool, m->gpu);
                k_tmp = key_cache_row;
                v_tmp = value_cache_row;
            }

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

            if (c->kv_tq_bits > 0 && m->tq_state) {
                // TQ write + GQA
                tq_write_kv(m->tq_state, s, k_tmp, v_tmp,
                            c->n_kv_heads, head_size, attn_idx, cache_pos, c->seq_len);
                tq_gqa_dispatch(m, s, attn_idx, pos, n_heads,
                                c->n_kv_heads, head_size, kv_mul);
            } else {
                // Standard GQA
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_dim, c->seq_len };
#ifdef __ARM_NEON
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_neon_range : bn_transformer_gqa_neon_range;
#elif defined(__AVX2__)
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_avx2_range : bn_transformer_gqa_avx2_range;
#elif defined(__wasm_simd128__)
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_wasm_range : bn_transformer_gqa_wasm_range;
#else
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_scalar_range : bn_transformer_gqa_scalar_range;
#endif
                BnTPTask gqa = { attn_fn, &gctx, n_heads };
                bn_tp_dispatch(m->pool, &gqa, 1);
            }

            // wo projection (q_dim → dim) + residual
            if (lw->attn_sub_norm)
                rmsnorm(s->xb, s->xb, lw->attn_sub_norm, q_dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->wo }};
                bn_quant_matvec_batch_gpu(wo, 1, s->xb, s->x_q, m->pool, m->gpu);
            }
            residual_add(s->x, s->xb2, dim);

        } else {
            // --- Classic attention path (existing) ---
            float *key_cache_row   = s->key_cache   + loff + (size_t)cache_pos * kv_dim;
            float *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_dim;

            if (c->kv_tq_bits > 0 && m->tq_state) {
                // --- TurboQuant KV path ---
                // Use temp buffers for K/V, then quantize into TQ cache
                float *k_tmp = s->hb, *v_tmp = s->hb2;
                BnMatvecTask qkv[3] = {
                    { s->q,  &lw->wq },
                    { k_tmp, &lw->wk },
                    { v_tmp, &lw->wv },
                };
                bn_quant_matvec_batch_gpu(qkv, 3, s->xb, s->x_q, m->pool, m->gpu);
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

                // Write TQ compressed KV
                tq_write_kv(m->tq_state, s, k_tmp, v_tmp,
                            c->n_kv_heads, head_size, attn_idx, cache_pos, c->seq_len);

                // TQ GQA
                tq_gqa_dispatch(m, s, attn_idx, pos, n_heads,
                                c->n_kv_heads, head_size, kv_mul);
                BL_ACC(bl_gqa_us);

            } else if (c->kv_f16) {
                float *k_tmp = s->hb, *v_tmp = s->hb2;
                BnMatvecTask qkv[3] = {
                    { s->q,  &lw->wq },
                    { k_tmp, &lw->wk },
                    { v_tmp, &lw->wv },
                };
                bn_quant_matvec_batch_gpu(qkv, 3, s->xb, s->x_q, m->pool, m->gpu);
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
                bn_quant_matvec_batch_gpu(qkv, 3, s->xb, s->x_q, m->pool, m->gpu);
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

            // GQA attention (standard path — TQ handled above)
            if (!(c->kv_tq_bits > 0 && m->tq_state)) {
                int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                BnGQACtx gctx = { c, s, loff, pos, n_kv, kv_mul, head_size, kv_dim, c->seq_len };
#ifdef __ARM_NEON
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_neon_range : bn_transformer_gqa_neon_range;
#elif defined(__AVX2__)
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_avx2_range : bn_transformer_gqa_avx2_range;
#elif defined(__wasm_simd128__)
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_wasm_range : bn_transformer_gqa_wasm_range;
#else
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_scalar_range : bn_transformer_gqa_scalar_range;
#endif
                BnTPTask gqa = { attn_fn, &gctx, n_heads };
                bn_tp_dispatch(m->pool, &gqa, 1);
                BL_ACC(bl_gqa_us);
            }

            // Attention sub-norm + wo projection + residual
            if (lw->attn_sub_norm)
                rmsnorm(s->xb, s->xb, lw->attn_sub_norm, dim, c->norm_eps);
            {
                BnMatvecTask wo[1] = {{ s->xb2, &lw->wo }};
                bn_quant_matvec_batch_gpu(wo, 1, s->xb, s->x_q, m->pool, m->gpu);
            }
            BL_ACC(bl_matvec_qkv_us);
            residual_add(s->x, s->xb2, dim);
            BL_ACC(bl_residual_us);
        }

    } else {
        // ---- SSM block ----
        forward_ssm_block(m, sess, lw, l);
        residual_add(s->x, s->xb, dim);
        BL_ACC(bl_residual_us);
    }

    // ---- FFN block ---- (shared by both layer types)
    (void)0;
    if (lw->router_weight) {
        // MoE FFN — route, pread, compute, combine
        bn_moe_forward(m, sess, lw, l);
    } else {
        // Dense FFN
        forward_ffn_block(m, sess, lw);
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
static int forward_layers(BnModel *m, BnSession *sess, int token, int pos) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
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
        if (forward_single_layer(m, sess, l, pos, cache_pos, rope_dims,
                                 rope_cos, rope_sin) != 0)
            return -1;
    }

    return 0;
}

// Final RMSNorm + logits computation. Reads s->x, writes s->logits.
// Returns s->logits.
static float *forward_logits(BnModel *m, BnSession *sess) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &sess->state;
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
#elif defined(__wasm_relaxed_simd__)
        if (w->emb_out_i8) {
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { bn_transformer_logits_i8_wasm_range, &lctx, n_rows };
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
        bn_quant_matvec_gpu(s->logits, &w->output_weight, s->x, s->x_q, m->pool, m->gpu);
    }
    // Tied Q4_0/Q8_0/Q6_K embeddings: use quant matvec
    else if (w->emb_type == BN_GGUF_TENSOR_Q4_0 || w->emb_type == BN_GGUF_TENSOR_Q8_0 ||
             w->emb_type == BN_GGUF_TENSOR_Q6_K) {
        BnQWeight tied = { w->token_embedding, w->emb_type, c->vocab_size, dim, 1.0f, NULL, NULL, NULL };
        bn_quant_matvec_gpu(s->logits, &tied, s->x, s->x_q, m->pool, m->gpu);
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
#elif defined(__wasm_relaxed_simd__)
        if (w->emb_out_i8) {
            float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
            BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                                 s->x_q, x_scale, dim };
            BnTPTask logits_task = { bn_transformer_logits_i8_wasm_range, &lctx, c->vocab_size };
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

// GPU-resident forward pass: one submit per token, reads back logits only.
// Supports classic transformer only (no MoE, no SSM, no gated-Q, no wide-Q,
// no Q/K norms, no sub-norms, no FP16 KV cache).
// Supports attention biases (Qwen2.5) and tied embeddings (BitNet).
// Returns s->logits on success, NULL to fall back to CPU.
static float *forward_gpu(BnModel *m, BnSession *sess, int token, int pos) {
    (void)0;
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &sess->state;
    BnGPUBackend *gpu = m->gpu;

    if (!gpu || !gpu->execute || !gpu->write_activation) return NULL;

    // Bounds checks
    if (token < 0 || token >= c->vocab_size) return NULL;
    if (pos < 0) return NULL;

    // FP16 KV cache not supported on GPU path
    if (c->kv_f16) return NULL;

    int dim = c->dim;
    int kv_dim = c->kv_dim;
    int head_size = c->head_size;
    int n_heads = c->n_heads;
    int hidden_dim = c->hidden_dim;
    int q_dim = n_heads * head_size;
    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : head_size;

    // Embed token on CPU, upload to GPU x buffer
    if (dim > BN_MAX_VLA_ELEMS) return NULL;
    float emb[dim];
    bn_model_embed_token(m, emb, token);
    if (gpu->write_activation(gpu->ctx, BN_GPU_BUF_X, emb,
                              (size_t)dim * sizeof(float), 0) != 0)
        return NULL;

    (void)0;

    // Validation: check for unsupported layer configurations
    // MoE and SSM layers are allowed but handled via CPU fallback per-layer.
    if (!w->output_norm_gpu) return NULL;
    int has_moe = 0, has_ssm = 0;
    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        int is_attn = (c->full_attn_interval == 0) ||
                      ((l + 1) % c->full_attn_interval == 0);
        if (!is_attn) { has_ssm = 1; continue; }
        if (lw->router_weight) { has_moe = 1; }
        if (!lw->wq.data) return NULL;
        // Q-gated: reject entire GPU forward pass — attention on CPU, MoE on CPU
        // The Q-gated attention with head_size=256 on hybrid SSM+attention models
        // produces incorrect GPU output. Root cause unidentified after extensive
        // debugging (deinterleave, norms, sigmoid_gate all verified on CPU side).
        if (lw->wq.rows > q_dim) return NULL;
        if (lw->q_norm && !lw->q_norm_gpu) return NULL;
        if (lw->k_norm && !lw->k_norm_gpu) return NULL;
        if (lw->attn_sub_norm && !lw->attn_sub_norm_gpu) return NULL;
        if (lw->ffn_sub_norm && !lw->ffn_sub_norm_gpu) return NULL;
        if (!lw->attn_norm_gpu || !lw->ffn_norm_gpu) return NULL;
    }
    // Models with SSM or MoE need per-layer CPU-GPU sync via read/write_activation
    if ((has_moe || has_ssm) && (!gpu->read_activation || !gpu->write_activation))
        return NULL;

    // Resolve logits weight
    BnQWeight *ow = &w->output_weight;
    void *logit_gpu_buf = ow->data ? ow->gpu_buf : NULL;
    int logit_type = ow->data ? ow->type : -1;
    int logit_rows = ow->data ? ow->rows : c->vocab_size;
    int logit_cols = ow->data ? ow->cols : dim;
    if (!logit_gpu_buf && w->emb_gpu_buf) {
        logit_gpu_buf = w->emb_gpu_buf;
        logit_type = w->emb_type;
        logit_rows = c->vocab_size;
        logit_cols = dim;
    }
    if (!logit_gpu_buf) return NULL;

    // Precompute eps as uint32
    uint32_t u_eps;
    { float eps = c->norm_eps; memcpy(&u_eps, &eps, 4); }

    // Max ops per batch. MoE/SSM flush between layers, so single-layer max suffices.
    // Max ops: ~32 per layer (26 base + 2 Q/K norms + 2 sub-norms + 2 Q-gated)
    // MoE: 8 experts × 5 ops + shared (5) + residual + rmsnorm + attention (~20) = ~70
    int max_ops = has_moe || has_ssm ? 80 : 34 * c->n_layers + 4;
    BnGPUOp *ops = (BnGPUOp *)malloc((size_t)max_ops * sizeof(BnGPUOp));
    if (!ops) return NULL;
    int n = 0;

    // Helper: flush current ops (no readback), reset counter
    #define GPU_FLUSH() do { \
        if (n > 0) { \
            if (gpu->execute(gpu->ctx, ops, n, -1, NULL, 0) != 0) { free(ops); return NULL; } \
            n = 0; \
        } \
    } while(0)

    // ---- Initial RMSNorm: x -> xb (using layer 0 attn_norm) ----
    ops[n++] =(BnGPUOp){
        .shader = BN_GPU_SHADER_RMSNORM, .type = -1,
        .W_buf = w->layers[0].attn_norm_gpu,
        .buf_in = BN_GPU_BUF_X, .buf_out = BN_GPU_BUF_XB, .buf_aux = -1,
        .rows = 0, .cols = 0,
        .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 }
    };

    (void)0;

    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        int is_attn = (c->full_attn_interval == 0) ||
                      ((l + 1) % c->full_attn_interval == 0);

        // ---- SSM layer: GPU-native if weights uploaded, else CPU fallback ----
        if (!is_attn) {
            if (!lw->ssm_conv1d_gpu || !lw->ssm_dt_bias_gpu || !lw->ssm_a_log_gpu || !lw->ssm_norm_gpu ||
                !lw->wqkv.gpu_buf || !lw->wz.gpu_buf ||
                !lw->ssm_alpha.gpu_buf || !lw->ssm_beta.gpu_buf ||
                !lw->ssm_out.gpu_buf) {
                // CPU fallback: flush GPU, run SSM on CPU, upload result
                GPU_FLUSH();
                if (gpu->read_activation(gpu->ctx, BN_GPU_BUF_X, s->x,
                                          (size_t)dim * sizeof(float), 0) != 0)
                    { free(ops); return NULL; }
                forward_ssm_block(m, sess, lw, l);
                residual_add(s->x, s->xb, dim);
                if (lw->router_weight)
                    bn_moe_forward(m, sess, lw, l);
                else
                    forward_ffn_block(m, sess, lw);
                if (gpu->write_activation(gpu->ctx, BN_GPU_BUF_X, s->x,
                                           (size_t)dim * sizeof(float), 0) != 0)
                    { free(ops); return NULL; }
                void *nn = (l + 1 < c->n_layers) ? w->layers[l + 1].attn_norm_gpu : w->output_norm_gpu;
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_RMSNORM, .type = -1,
                    .W_buf = nn, .buf_in = BN_GPU_BUF_X, .buf_out = BN_GPU_BUF_XB, .buf_aux = -1,
                    .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 } };
                continue;
            }

            // GPU-native SSM path
            int ssm_idx = l - (l + 1) / c->full_attn_interval;
            int num_k_heads = c->ssm_group_count;
            int head_k_dim  = c->ssm_state_size;
            int num_v_heads = c->ssm_time_step_rank;
            int head_v_dim  = c->ssm_inner_size / (num_v_heads > 0 ? num_v_heads : 1);
            int key_dim     = num_k_heads * head_k_dim;
            int value_dim   = c->ssm_inner_size;
            int qkv_dim_ssm = key_dim * 2 + value_dim;
            int kern        = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;
            size_t conv_off = (size_t)ssm_idx * (kern - 1) * qkv_dim_ssm;
            size_t state_per = (size_t)num_v_heads * head_k_dim * head_v_dim;
            size_t state_off = (size_t)ssm_idx * state_per;
            uint32_t u_qscale; { float qs = 1.0f / sqrtf((float)head_k_dim); memcpy(&u_qscale, &qs, 4); }

            // 1. RMSNorm: X -> XB (already done by previous layer's fused resid+norm)
            // 2. QKV matvec: XB -> SSM_QKV
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = lw->wqkv.type,
                .W_buf = lw->wqkv.gpu_buf, .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_SSM_QKV,
                .buf_aux = -1, .rows = lw->wqkv.rows, .cols = lw->wqkv.cols,
                .p = { (uint32_t)lw->wqkv.rows, (uint32_t)lw->wqkv.cols, 1, 0, 0, 0, 0, 0 } };
            // 3. Z matvec: XB -> SSM_Z
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = lw->wz.type,
                .W_buf = lw->wz.gpu_buf, .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_SSM_Z,
                .buf_aux = -1, .rows = lw->wz.rows, .cols = lw->wz.cols,
                .p = { (uint32_t)lw->wz.rows, (uint32_t)lw->wz.cols, 1, 0, 0, 0, 0, 0 } };
            // 4. Conv1d + SiLU
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_SSM_CONV_SILU, .type = -1,
                .W_buf = lw->ssm_conv1d_gpu, .buf_in = BN_GPU_BUF_SSM_QKV, .buf_out = -1, .buf_aux = -1,
                .p = { (uint32_t)qkv_dim_ssm, (uint32_t)kern, (uint32_t)conv_off, (uint32_t)((kern-1)*qkv_dim_ssm), 0, 0, 0, 0 } };
            // 5. Split Q from SSM_QKV -> Q buf
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_COPY, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_SSM_QKV, .buf_out = BN_GPU_BUF_Q, .buf_aux = -1,
                .p = { 0, 0, (uint32_t)key_dim, 0, 0, 0, 0, 0 } };
            // 6. Split K from SSM_QKV -> SCRATCH
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_COPY, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_SSM_QKV, .buf_out = BN_GPU_BUF_SCRATCH, .buf_aux = -1,
                .p = { (uint32_t)key_dim, 0, (uint32_t)key_dim, 0, 0, 0, 0, 0 } };
            // 7. Split V from SSM_QKV -> SSM_V
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_COPY, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_SSM_QKV, .buf_out = BN_GPU_BUF_SSM_V, .buf_aux = -1,
                .p = { (uint32_t)(2*key_dim), 0, (uint32_t)value_dim, 0, 0, 0, 0, 0 } };
            // 8. L2Norm Q (in Q buf) and K (in SCRATCH)
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_SSM_L2NORM, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_Q, .buf_out = -1, .buf_aux = BN_GPU_BUF_SCRATCH,
                .rows = num_k_heads, .p = { (uint32_t)head_k_dim, 0, 0, 0, 0, 0, 0, 0 } };
            // 9. Alpha matvec: XB -> SSM_ALPHA
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = lw->ssm_alpha.type,
                .W_buf = lw->ssm_alpha.gpu_buf, .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_SSM_ALPHA,
                .buf_aux = -1, .rows = lw->ssm_alpha.rows, .cols = lw->ssm_alpha.cols,
                .p = { (uint32_t)lw->ssm_alpha.rows, (uint32_t)lw->ssm_alpha.cols, 1, 0, 0, 0, 0, 0 } };
            // 10. Beta matvec: XB -> SSM_BETA
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = lw->ssm_beta.type,
                .W_buf = lw->ssm_beta.gpu_buf, .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_SSM_BETA,
                .buf_aux = -1, .rows = lw->ssm_beta.rows, .cols = lw->ssm_beta.cols,
                .p = { (uint32_t)lw->ssm_beta.rows, (uint32_t)lw->ssm_beta.cols, 1, 0, 0, 0, 0, 0 } };
            // 11. Alpha/Beta activation (softplus+exp, sigmoid)
            {
                uintptr_t a_ptr = (uintptr_t)lw->ssm_a_log_gpu;
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_SSM_ALPHA_BETA, .type = -1,
                    .W_buf = lw->ssm_dt_bias_gpu, .buf_in = BN_GPU_BUF_SSM_ALPHA,
                    .buf_out = -1, .buf_aux = BN_GPU_BUF_SSM_BETA,
                    .p = { (uint32_t)num_v_heads, 0, 0, 0, 0, 0,
                           (uint32_t)(a_ptr & 0xFFFFFFFF), (uint32_t)((uint64_t)a_ptr >> 32) } };
            }
            // 12. Delta rule recurrence
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_SSM_DELTA, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_Q, .buf_out = BN_GPU_BUF_XB2, .buf_aux = BN_GPU_BUF_SCRATCH,
                .rows = num_v_heads,
                .p = { (uint32_t)head_k_dim, (uint32_t)head_v_dim, (uint32_t)num_k_heads,
                       u_qscale, (uint32_t)(state_off * sizeof(float)),
                       (uint32_t)(state_per * sizeof(float)), 0, 0 } };
            // 13. Gate: per-head RMSNorm + SiLU gate
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_SSM_GATE, .type = -1,
                .W_buf = lw->ssm_norm_gpu, .buf_in = BN_GPU_BUF_XB2, .buf_out = -1,
                .buf_aux = BN_GPU_BUF_SSM_Z, .rows = num_v_heads,
                .p = { (uint32_t)head_v_dim, u_eps, 0, 0, 0, 0, 0, 0 } };
            // 14. Output projection: XB2 -> SCRATCH
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = lw->ssm_out.type,
                .W_buf = lw->ssm_out.gpu_buf, .buf_in = BN_GPU_BUF_XB2, .buf_out = BN_GPU_BUF_SCRATCH,
                .buf_aux = -1, .rows = lw->ssm_out.rows, .cols = lw->ssm_out.cols,
                .p = { (uint32_t)lw->ssm_out.rows, (uint32_t)lw->ssm_out.cols, 1, 0, 0, 0, 0, 0 } };
            // 15. Fused residual + rmsnorm for FFN
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_RESIDUAL_RMSNORM, .type = -1,
                .W_buf = lw->ffn_norm_gpu,
                .buf_in = BN_GPU_BUF_X, .buf_out = BN_GPU_BUF_XB, .buf_aux = BN_GPU_BUF_SCRATCH,
                .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 } };

            // SSM layer's FFN (dense or MoE) — same as attention layer below
            goto ffn_block;
        }

        // KV cache addressing
        int attn_idx = (c->full_attn_interval > 0)
            ? (l + 1) / c->full_attn_interval - 1 : l;
        size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;
        int cache_pos = pos % c->seq_len;
        int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;

        // ---- QKV matvecs ----
        // Q-gated layers must use separate Q/K/V path (stacked doesn't handle interleaved Q+Gate)
        if (lw->qkv_stacked_gpu && lw->wq.rows <= q_dim) {
            // Stacked QKV: single matvec -> QKV buf, then split
            int total_rows = lw->wq.rows + lw->wk.rows + lw->wv.rows;
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_MATVEC, .type = lw->wq.type,
                .W_buf = lw->qkv_stacked_gpu,
                .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_QKV, .buf_aux = -1,
                .rows = total_rows, .cols = lw->wq.cols,
                .p = { (uint32_t)total_rows, (uint32_t)lw->wq.cols, 1, 0, 0, 0, 0, 0 }
            };
            // Copy Q from QKV
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_COPY, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_QKV, .buf_out = BN_GPU_BUF_Q, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { 0, 0, (uint32_t)q_dim, 0, 0, 0, 0, 0 }
            };
            // Q norm (per-head RMSNorm on Q buffer)
            if (lw->q_norm_gpu) {
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_PER_HEAD_RMSNORM, .type = -1,
                    .W_buf = lw->q_norm_gpu, .buf_in = BN_GPU_BUF_Q, .buf_out = -1, .buf_aux = -1,
                    .rows = n_heads,
                    .p = { (uint32_t)head_size, u_eps, (uint32_t)c->qk_norm_per_head, 0, 0, 0, 0, 0 } };
            }
            // Copy K from QKV -> scratch
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_COPY, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_QKV, .buf_out = BN_GPU_BUF_SCRATCH, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)q_dim, 0, (uint32_t)kv_dim, 0, 0, 0, 0, 0 }
            };
            // K norm (per-head RMSNorm on SCRATCH)
            if (lw->k_norm_gpu) {
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_PER_HEAD_RMSNORM, .type = -1,
                    .W_buf = lw->k_norm_gpu, .buf_in = BN_GPU_BUF_SCRATCH, .buf_out = -1, .buf_aux = -1,
                    .rows = c->n_kv_heads,
                    .p = { (uint32_t)head_size, u_eps, (uint32_t)c->qk_norm_per_head, 0, 0, 0, 0, 0 } };
            }
            // RoPE K
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_ROPE, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_SCRATCH, .buf_out = -1, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)c->n_kv_heads, (uint32_t)head_size,
                       (uint32_t)pos, (uint32_t)rope_dims, 0, 0, 0, 0 }
            };
            // K -> key_cache
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_COPY, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_SCRATCH, .buf_out = BN_GPU_BUF_KEY_CACHE, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { 0, (uint32_t)(loff + (size_t)cache_pos * kv_dim),
                       (uint32_t)kv_dim, 0, 0, 0, 0, 0 }
            };
            // Copy V from QKV -> scratch
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_COPY, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_QKV, .buf_out = BN_GPU_BUF_SCRATCH, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)(q_dim + kv_dim), 0, (uint32_t)kv_dim, 0, 0, 0, 0, 0 }
            };
            // V -> value_cache
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_COPY, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_SCRATCH, .buf_out = BN_GPU_BUF_VALUE_CACHE, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { 0, (uint32_t)(loff + (size_t)cache_pos * kv_dim),
                       (uint32_t)kv_dim, 0, 0, 0, 0, 0 }
            };
        } else {
            // Separate Q/K/V matvecs
            int q_gated = (lw->wq.rows > q_dim);
            if (q_gated) {
                // Q-gated: matvec to QKV (2*q_dim), then deinterleave Q
                ops[n++] = (BnGPUOp){
                    .shader = BN_GPU_SHADER_MATVEC, .type = lw->wq.type,
                    .W_buf = lw->wq.gpu_buf,
                    .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_QKV, .buf_aux = -1,
                    .rows = lw->wq.rows, .cols = lw->wq.cols,
                    .p = { (uint32_t)lw->wq.rows, (uint32_t)lw->wq.cols, 1, 0, 0, 0, 0, 0 } };
                ops[n++] = (BnGPUOp){
                    .shader = BN_GPU_SHADER_DEINTERLEAVE_Q, .type = -1, .W_buf = NULL,
                    .buf_in = BN_GPU_BUF_QKV, .buf_out = BN_GPU_BUF_Q, .buf_aux = -1,
                    .p = { (uint32_t)q_dim, (uint32_t)head_size, 0, 0, 0, 0, 0, 0 } };
            } else {
                // Standard Q matvec to Q buffer
                ops[n++] = (BnGPUOp){
                    .shader = BN_GPU_SHADER_MATVEC, .type = lw->wq.type,
                    .W_buf = lw->wq.gpu_buf,
                    .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_Q, .buf_aux = -1,
                    .rows = lw->wq.rows, .cols = lw->wq.cols,
                    .p = { (uint32_t)lw->wq.rows, (uint32_t)lw->wq.cols, 1, 0, 0, 0, 0, 0 } };
            }
            if (lw->q_bias_gpu) {
                ops[n++] =(BnGPUOp){
                    .shader = BN_GPU_SHADER_BIAS_ADD, .type = -1,
                    .W_buf = lw->q_bias_gpu,
                    .buf_in = BN_GPU_BUF_Q, .buf_out = -1, .buf_aux = -1,
                    .rows = 0, .cols = 0,
                    .p = { (uint32_t)q_dim, 0, 0, 0, 0, 0, 0, 0 }
                };
            }
            // Q norm
            if (lw->q_norm_gpu) {
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_PER_HEAD_RMSNORM, .type = -1,
                    .W_buf = lw->q_norm_gpu, .buf_in = BN_GPU_BUF_Q, .buf_out = -1, .buf_aux = -1,
                    .rows = n_heads,
                    .p = { (uint32_t)head_size, u_eps, (uint32_t)c->qk_norm_per_head, 0, 0, 0, 0, 0 } };
            }

            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_MATVEC, .type = lw->wk.type,
                .W_buf = lw->wk.gpu_buf,
                .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_SCRATCH, .buf_aux = -1,
                .rows = lw->wk.rows, .cols = lw->wk.cols,
                .p = { (uint32_t)lw->wk.rows, (uint32_t)lw->wk.cols, 1, 0, 0, 0, 0, 0 }
            };
            if (lw->k_bias_gpu) {
                ops[n++] =(BnGPUOp){
                    .shader = BN_GPU_SHADER_BIAS_ADD, .type = -1,
                    .W_buf = lw->k_bias_gpu,
                    .buf_in = BN_GPU_BUF_SCRATCH, .buf_out = -1, .buf_aux = -1,
                    .rows = 0, .cols = 0,
                    .p = { (uint32_t)kv_dim, 0, 0, 0, 0, 0, 0, 0 }
                };
            }

            // K norm
            if (lw->k_norm_gpu) {
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_PER_HEAD_RMSNORM, .type = -1,
                    .W_buf = lw->k_norm_gpu, .buf_in = BN_GPU_BUF_SCRATCH, .buf_out = -1, .buf_aux = -1,
                    .rows = c->n_kv_heads,
                    .p = { (uint32_t)head_size, u_eps, (uint32_t)c->qk_norm_per_head, 0, 0, 0, 0, 0 } };
            }

            // RoPE K, K -> key_cache
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_ROPE, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_SCRATCH, .buf_out = -1, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)c->n_kv_heads, (uint32_t)head_size,
                       (uint32_t)pos, (uint32_t)rope_dims, 0, 0, 0, 0 }
            };
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_COPY, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_SCRATCH, .buf_out = BN_GPU_BUF_KEY_CACHE, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { 0, (uint32_t)(loff + (size_t)cache_pos * kv_dim),
                       (uint32_t)kv_dim, 0, 0, 0, 0, 0 }
            };

            // V matvec: xb -> scratch
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_MATVEC, .type = lw->wv.type,
                .W_buf = lw->wv.gpu_buf,
                .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_SCRATCH, .buf_aux = -1,
                .rows = lw->wv.rows, .cols = lw->wv.cols,
                .p = { (uint32_t)lw->wv.rows, (uint32_t)lw->wv.cols, 1, 0, 0, 0, 0, 0 }
            };
            if (lw->v_bias_gpu) {
                ops[n++] =(BnGPUOp){
                    .shader = BN_GPU_SHADER_BIAS_ADD, .type = -1,
                    .W_buf = lw->v_bias_gpu,
                    .buf_in = BN_GPU_BUF_SCRATCH, .buf_out = -1, .buf_aux = -1,
                    .rows = 0, .cols = 0,
                    .p = { (uint32_t)kv_dim, 0, 0, 0, 0, 0, 0, 0 }
                };
            }

            // V -> value_cache
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_COPY, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_SCRATCH, .buf_out = BN_GPU_BUF_VALUE_CACHE, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { 0, (uint32_t)(loff + (size_t)cache_pos * kv_dim),
                       (uint32_t)kv_dim, 0, 0, 0, 0, 0 }
            };
        }

        // ---- RoPE Q ----
        ops[n++] =(BnGPUOp){
            .shader = BN_GPU_SHADER_ROPE, .type = -1, .W_buf = NULL,
            .buf_in = BN_GPU_BUF_Q, .buf_out = -1, .buf_aux = -1,
            .rows = 0, .cols = 0,
            .p = { (uint32_t)n_heads, (uint32_t)head_size,
                   (uint32_t)pos, (uint32_t)rope_dims, 0, 0, 0, 0 }
        };

        // ---- GQA: scores, softmax, combine ----
        {
            float inv_sqrt_hs = 1.0f / sqrtf((float)head_size);
            uint32_t u_inv_sqrt_hs;
            memcpy(&u_inv_sqrt_hs, &inv_sqrt_hs, 4);
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_GQA_SCORES, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_Q, .buf_out = -1, .buf_aux = -1,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)n_heads, (uint32_t)head_size, (uint32_t)n_kv,
                       (uint32_t)c->kv_mul, (uint32_t)kv_dim, (uint32_t)c->seq_len,
                       (uint32_t)loff, u_inv_sqrt_hs }
            };
        }
        ops[n++] =(BnGPUOp){
            .shader = BN_GPU_SHADER_SOFTMAX, .type = -1, .W_buf = NULL,
            .buf_in = -1, .buf_out = -1, .buf_aux = -1,
            .rows = 0, .cols = 0,
            .p = { (uint32_t)n_heads, (uint32_t)n_kv, (uint32_t)c->seq_len,
                   0, 0, 0, 0, 0 }
        };
        ops[n++] =(BnGPUOp){
            .shader = BN_GPU_SHADER_GQA_COMBINE, .type = -1, .W_buf = NULL,
            .buf_in = -1, .buf_out = BN_GPU_BUF_XB, .buf_aux = -1,
            .rows = 0, .cols = 0,
            .p = { (uint32_t)n_heads, (uint32_t)head_size, (uint32_t)n_kv,
                   (uint32_t)c->kv_mul, (uint32_t)kv_dim, (uint32_t)c->seq_len,
                   (uint32_t)loff, 0 }
        };

        // ---- Sigmoid gate (Q-gated only): xb *= sigmoid(gate) ----
        {
            int q_gated_l = (lw->wq.rows > q_dim);
            if (q_gated_l) {
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_SIGMOID_GATE, .type = -1, .W_buf = NULL,
                    .buf_in = BN_GPU_BUF_XB, .buf_out = -1, .buf_aux = BN_GPU_BUF_QKV,
                    .p = { (uint32_t)q_dim, (uint32_t)head_size, 0, 0, 0, 0, 0, 0 } };
            }
        }

        // ---- Attention sub-norm (before Wo) ----
        if (lw->attn_sub_norm_gpu) {
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_RMSNORM, .type = -1,
                .W_buf = lw->attn_sub_norm_gpu,
                .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_XB, .buf_aux = -1,
                .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 } };
        }

        // ---- Wo matvec: xb -> xb2 ----
        ops[n++] =(BnGPUOp){
            .shader = BN_GPU_SHADER_MATVEC, .type = lw->wo.type,
            .W_buf = lw->wo.gpu_buf,
            .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_XB2, .buf_aux = -1,
            .rows = lw->wo.rows, .cols = lw->wo.cols,
            .p = { (uint32_t)lw->wo.rows, (uint32_t)lw->wo.cols, 1, 0, 0, 0, 0, 0 }
        };

        // ---- Fused residual + rmsnorm: x += xb2, rmsnorm(x, ffn_norm) -> xb ----
        ops[n++] =(BnGPUOp){
            .shader = BN_GPU_SHADER_RESIDUAL_RMSNORM, .type = -1,
            .W_buf = lw->ffn_norm_gpu,
            .buf_in = BN_GPU_BUF_X, .buf_out = BN_GPU_BUF_XB, .buf_aux = BN_GPU_BUF_XB2,
            .rows = 0, .cols = 0,
            .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 }
        };

        // ---- FFN (MoE or dense) ----
        ffn_block:;
        if (lw->router_weight) {
            // MoE FFN: GPU expert matvecs with CPU-side routing
            GPU_FLUSH();

            // Read back xb (normed input) for CPU routing — small: dim floats
            float xb_cpu[dim];
            if (gpu->read_activation(gpu->ctx, BN_GPU_BUF_XB, xb_cpu,
                                      (size_t)dim * sizeof(float), 0) != 0)
                { free(ops); return NULL; }

            (void)0;
            // CPU routing: select top-K experts
            BnMoEState *ms = sess->moe_state;
            int K = c->n_experts_active;
            bn_moe_route(ms, xb_cpu, lw->router_weight, dim, c->n_experts, K, m->pool);

            // Zero the MoE output accumulator on GPU
            {
                float zeros[dim];
                memset(zeros, 0, sizeof(zeros));
                if (gpu->write_activation(gpu->ctx, BN_GPU_BUF_MOE_OUT, zeros,
                                           (size_t)dim * sizeof(float), 0) != 0)
                    { free(ops); return NULL; }
            }

            // GPU expert dispatch with LRU cache for GPU buffer reuse
            int moe_hidden = c->moe_intermediate_size;
            const BnMoEExpertMap *em = &lw->expert_map;
            BnGPUMoECache *gpu_cache = (BnGPUMoECache *)m->moe_io.gpu_moe_cache;

            for (int k = 0; k < K; k++) {
                int eidx = ms->expert_indices[k];
                if (eidx < 0) continue;
                float ew = ms->expert_weights[k];
                uint32_t u_ew; memcpy(&u_ew, &ew, 4);

                void *gate_gpu, *up_gpu, *down_gpu;
                int cached = bn_gpu_moe_cache_lookup(gpu_cache, l, eidx,
                                                      &gate_gpu, &up_gpu, &down_gpu);
                (void)0;
                if (!cached) {
                    // Cache miss: load from host + upload to GPU
                    const void *gate_data = bn_moe_get_expert_proj(&m->moe_io, ms, em, eidx, 0);
                    const void *up_data   = bn_moe_get_expert_proj(&m->moe_io, ms, em, eidx, 1);
                    const void *down_data = bn_moe_get_expert_proj(&m->moe_io, ms, em, eidx, 2);
                    if (!gate_data || !up_data || !down_data) continue;

                    gate_gpu = gpu->buffer_create(gpu->ctx, gate_data, em->expert_gate_bytes,
                        em->gate_type, em->gate_rows, em->gate_cols);
                    up_gpu = gpu->buffer_create(gpu->ctx, up_data, em->expert_up_bytes,
                        em->up_type, em->up_rows, em->up_cols);
                    down_gpu = gpu->buffer_create(gpu->ctx, down_data, em->expert_down_bytes,
                        em->down_type, em->down_rows, em->down_cols);
                    if (!gate_gpu || !up_gpu || !down_gpu) {
                        if (gate_gpu) gpu->buffer_destroy(gpu->ctx, gate_gpu);
                        if (up_gpu) gpu->buffer_destroy(gpu->ctx, up_gpu);
                        if (down_gpu) gpu->buffer_destroy(gpu->ctx, down_gpu);
                        continue;
                    }

                    if (gpu_cache) {
                        bn_gpu_moe_cache_insert(gpu_cache, l, eidx, gate_gpu, up_gpu, down_gpu);
                    }
                }

                // 5 GPU ops: gate, up, silu_gate, down, weighted_add
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = em->gate_type,
                    .W_buf = gate_gpu, .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_MOE_HB,
                    .buf_aux = -1, .rows = em->gate_rows, .cols = em->gate_cols,
                    .p = { (uint32_t)em->gate_rows, (uint32_t)em->gate_cols, 1, 0, 0, 0, 0, 0 } };
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = em->up_type,
                    .W_buf = up_gpu, .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_MOE_HB2,
                    .buf_aux = -1, .rows = em->up_rows, .cols = em->up_cols,
                    .p = { (uint32_t)em->up_rows, (uint32_t)em->up_cols, 1, 0, 0, 0, 0, 0 } };
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_SILU_GATE, .type = -1, .W_buf = NULL,
                    .buf_in = BN_GPU_BUF_MOE_HB, .buf_out = -1, .buf_aux = BN_GPU_BUF_MOE_HB2,
                    .p = { (uint32_t)moe_hidden, 0, 0, 0, 0, 0, 0, 0 } };
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = em->down_type,
                    .W_buf = down_gpu, .buf_in = BN_GPU_BUF_MOE_HB, .buf_out = BN_GPU_BUF_XB2,
                    .buf_aux = -1, .rows = em->down_rows, .cols = em->down_cols,
                    .p = { (uint32_t)em->down_rows, (uint32_t)em->down_cols, 1, 0, 0, 0, 0, 0 } };
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_WEIGHTED_ADD, .type = -1, .W_buf = NULL,
                    .buf_in = BN_GPU_BUF_MOE_OUT, .buf_out = -1, .buf_aux = BN_GPU_BUF_XB2,
                    .p = { (uint32_t)dim, u_ew, 0, 0, 0, 0, 0, 0 } };

            }

            // Shared expert (if present, weights pre-uploaded at init)
            if (lw->shared_gate.data && lw->shared_gate.gpu_buf) {
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = lw->shared_gate.type,
                    .W_buf = lw->shared_gate.gpu_buf, .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_HB,
                    .buf_aux = -1, .rows = lw->shared_gate.rows, .cols = lw->shared_gate.cols,
                    .p = { (uint32_t)lw->shared_gate.rows, (uint32_t)lw->shared_gate.cols, 1, 0, 0, 0, 0, 0 } };
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = lw->shared_up.type,
                    .W_buf = lw->shared_up.gpu_buf, .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_HB2,
                    .buf_aux = -1, .rows = lw->shared_up.rows, .cols = lw->shared_up.cols,
                    .p = { (uint32_t)lw->shared_up.rows, (uint32_t)lw->shared_up.cols, 1, 0, 0, 0, 0, 0 } };
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_SILU_GATE, .type = -1, .W_buf = NULL,
                    .buf_in = BN_GPU_BUF_HB, .buf_out = -1, .buf_aux = BN_GPU_BUF_HB2,
                    .p = { (uint32_t)lw->shared_gate.rows, 0, 0, 0, 0, 0, 0, 0 } };
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_MATVEC, .type = lw->shared_down.type,
                    .W_buf = lw->shared_down.gpu_buf, .buf_in = BN_GPU_BUF_HB, .buf_out = BN_GPU_BUF_XB2,
                    .buf_aux = -1, .rows = lw->shared_down.rows, .cols = lw->shared_down.cols,
                    .p = { (uint32_t)lw->shared_down.rows, (uint32_t)lw->shared_down.cols, 1, 0, 0, 0, 0, 0 } };
                // TODO: shared expert sigmoid gate (requires dot product shader)
                // For now, add shared expert output with weight=1.0
                uint32_t u_one; { float one = 1.0f; memcpy(&u_one, &one, 4); }
                ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_WEIGHTED_ADD, .type = -1, .W_buf = NULL,
                    .buf_in = BN_GPU_BUF_MOE_OUT, .buf_out = -1, .buf_aux = BN_GPU_BUF_XB2,
                    .p = { (uint32_t)dim, u_one, 0, 0, 0, 0, 0, 0 } };
            }

            // Residual: X += MOE_OUT
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_RESIDUAL_ADD, .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_X, .buf_out = -1, .buf_aux = BN_GPU_BUF_MOE_OUT,
                .p = { (uint32_t)dim, 0, 0, 0, 0, 0, 0, 0 } };

            // RMSNorm for next layer
            void *next_norm = (l + 1 < c->n_layers)
                ? w->layers[l + 1].attn_norm_gpu : w->output_norm_gpu;
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_RMSNORM, .type = -1,
                .W_buf = next_norm, .buf_in = BN_GPU_BUF_X, .buf_out = BN_GPU_BUF_XB, .buf_aux = -1,
                .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 } };
            // Flush all expert ops as ONE GPU submission (instead of per-expert)
            GPU_FLUSH();
            continue;  // skip dense FFN below
        }
        if (c->has_ffn_gate && lw->ffn_gate.data) {
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_MATVEC, .type = lw->ffn_gate.type,
                .W_buf = lw->ffn_gate.gpu_buf,
                .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_HB, .buf_aux = -1,
                .rows = lw->ffn_gate.rows, .cols = lw->ffn_gate.cols,
                .p = { (uint32_t)lw->ffn_gate.rows, (uint32_t)lw->ffn_gate.cols,
                       1, 0, 0, 0, 0, 0 }
            };
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_MATVEC, .type = lw->ffn_up.type,
                .W_buf = lw->ffn_up.gpu_buf,
                .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_HB2, .buf_aux = -1,
                .rows = lw->ffn_up.rows, .cols = lw->ffn_up.cols,
                .p = { (uint32_t)lw->ffn_up.rows, (uint32_t)lw->ffn_up.cols,
                       1, 0, 0, 0, 0, 0 }
            };
            ops[n++] =(BnGPUOp){
                .shader = (c->act_type == 1) ? BN_GPU_SHADER_RELU2_GATE
                                             : BN_GPU_SHADER_SILU_GATE,
                .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_HB, .buf_out = -1,
                .buf_aux = BN_GPU_BUF_HB2,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)hidden_dim, 0, 0, 0, 0, 0, 0, 0 }
            };
        } else {
            ops[n++] =(BnGPUOp){
                .shader = BN_GPU_SHADER_MATVEC, .type = lw->ffn_up.type,
                .W_buf = lw->ffn_up.gpu_buf,
                .buf_in = BN_GPU_BUF_XB, .buf_out = BN_GPU_BUF_HB, .buf_aux = -1,
                .rows = lw->ffn_up.rows, .cols = lw->ffn_up.cols,
                .p = { (uint32_t)lw->ffn_up.rows, (uint32_t)lw->ffn_up.cols,
                       1, 0, 0, 0, 0, 0 }
            };
            ops[n++] =(BnGPUOp){
                .shader = (c->act_type == 1) ? BN_GPU_SHADER_RELU2_GATE
                                             : BN_GPU_SHADER_SILU_GATE,
                .type = -1, .W_buf = NULL,
                .buf_in = BN_GPU_BUF_HB, .buf_out = -1,
                .buf_aux = BN_GPU_BUF_HB,
                .rows = 0, .cols = 0,
                .p = { (uint32_t)hidden_dim, 0, 0, 0, 0, 0, 0, 0 }
            };
        }

        // ---- FFN sub-norm (before down projection) ----
        if (lw->ffn_sub_norm_gpu) {
            ops[n++] = (BnGPUOp){ .shader = BN_GPU_SHADER_RMSNORM, .type = -1,
                .W_buf = lw->ffn_sub_norm_gpu,
                .buf_in = BN_GPU_BUF_HB, .buf_out = BN_GPU_BUF_HB, .buf_aux = -1,
                .p = { (uint32_t)hidden_dim, u_eps, 0, 0, 0, 0, 0, 0 } };
        }

        // ---- Down matvec: hb -> xb2 ----
        ops[n++] =(BnGPUOp){
            .shader = BN_GPU_SHADER_MATVEC, .type = lw->ffn_down.type,
            .W_buf = lw->ffn_down.gpu_buf,
            .buf_in = BN_GPU_BUF_HB, .buf_out = BN_GPU_BUF_XB2, .buf_aux = -1,
            .rows = lw->ffn_down.rows, .cols = lw->ffn_down.cols,
            .p = { (uint32_t)lw->ffn_down.rows, (uint32_t)lw->ffn_down.cols,
                   1, 0, 0, 0, 0, 0 }
        };

        // ---- Fused residual + rmsnorm: x += xb2, rmsnorm(x, next_norm) -> xb ----
        void *next_norm = (l + 1 < c->n_layers)
            ? w->layers[l + 1].attn_norm_gpu : w->output_norm_gpu;
        ops[n++] =(BnGPUOp){
            .shader = BN_GPU_SHADER_RESIDUAL_RMSNORM, .type = -1,
            .W_buf = next_norm,
            .buf_in = BN_GPU_BUF_X, .buf_out = BN_GPU_BUF_XB, .buf_aux = BN_GPU_BUF_XB2,
            .rows = 0, .cols = 0,
            .p = { (uint32_t)dim, u_eps, 0, 0, 0, 0, 0, 0 }
        };
    }

    // ---- Logits matvec: xb -> logits (xb is already normalized) ----
    {
        uint32_t tile_x = (logit_rows > 65535) ? 65535u : 0u;
        ops[n++] =(BnGPUOp){
            .shader = BN_GPU_SHADER_MATVEC,
            .type = logit_type,
            .W_buf = logit_gpu_buf,
            .buf_in = BN_GPU_BUF_XB,
            .buf_out = BN_GPU_BUF_LOGITS,
            .buf_aux = -1,
            .rows = logit_rows, .cols = logit_cols,
            .p = { (uint32_t)logit_rows, (uint32_t)logit_cols, 1, tile_x, 0, 0, 0, 0 }
        };
    }

    // Safety: verify we didn't overflow the ops array
    if (n > max_ops) { free(ops); return NULL; }

    // Execute final batch (logits + any remaining layer ops)
    int rc = gpu->execute(gpu->ctx, ops, n, BN_GPU_BUF_LOGITS,
                          s->logits, c->vocab_size);
    free(ops);
    #undef GPU_FLUSH
    return (rc == 0) ? s->logits : NULL;
}

float *bn_transformer_forward(BnModel *m, BnSession *s, int token, int pos) {
    // Try GPU-resident forward pass first
    float *gpu_logits = forward_gpu(m, s, token, pos);
    if (gpu_logits) {
        (void)0;
        return gpu_logits;
    }

    // Fall back to CPU forward pass
    if (forward_layers(m, s, token, pos) != 0) return NULL;
    return forward_logits(m, s);
}

// Internal prefill: if all_logits is non-NULL, compute logits at every position.
static float *prefill_internal(BnModel *m, BnSession *sess, const int *tokens, int n_tokens,
                                int pos0, float *all_logits) {
    if (n_tokens <= 0) return NULL;
    if (n_tokens == 1) return bn_transformer_forward(m, sess, tokens[0], pos0);

    BnConfig *c = &m->config;

    // MoE models use batch MoE prefill (token-expert grouping + per-expert matmul)
    // Falls through to the batched layer loop which calls bn_moe_forward_batch
    BnRunState *s = &sess->state;
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

    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : head_size;
    int half_rope = rope_dims / 2;
    if (half_rope > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("RoPE dimensions too large for stack VLAs");
        return NULL;
    }

    int kv_dim = c->kv_dim;
    int hidden_dim = c->hidden_dim;
    int q_dim = c->n_heads * head_size;
    size_t nt = (size_t)n_tokens;

    /* Arena for all prefill scratch: single allocation, 32-byte aligned,
     * contiguous layout. Replaces 5 separate mallocs and simplifies cleanup. */
    size_t batch_floats = nt * dim                                            // xb
                        + nt * (size_t)(q_dim > dim ? q_dim * 2 : dim)       // q_buf
                        + nt * kv_dim * 2                                     // k_new, v_new
                        + nt * dim                                            // xb2
                        + nt * hidden_dim * 2;                                // hb, hb2
    size_t arena_size = act_elems * sizeof(float)                             // act
                      + batch_floats * sizeof(float);                         // batch
#ifdef __AVX2__
    int n_bpr_pf = (dim % BN_QK_K == 0) ? dim / BN_QK_K : 0;
    if (n_bpr_pf > 0)
        arena_size += nt * dim                                                // pf_xq (int8)
                    + nt * n_bpr_pf * sizeof(float)                           // pf_xd
                    + nt * n_bpr_pf * 16 * sizeof(int16_t);                   // pf_xbs
#endif

    SHArena *pf_arena = sh_arena_create(arena_size);
    if (!pf_arena) return NULL;

    float *act = (float *)sh_arena_alloc(pf_arena, act_elems * sizeof(float));
    if (!act) { sh_arena_free(pf_arena); return NULL; }

    // Embed all tokens
    for (int t = 0; t < n_tokens; t++)
        bn_model_embed_token(m, act + (size_t)t * dim, tokens[t]);

    float *batch_buf = (float *)sh_arena_alloc(pf_arena, batch_floats * sizeof(float));
    if (!batch_buf) { sh_arena_free(pf_arena); return NULL; }

    /* Q8K scratch from same arena (NULL if non-k-quant or non-AVX2) */
    int8_t *pf_xq = NULL;
    float *pf_xd = NULL;
    int16_t *pf_xbs = NULL;
#ifdef __AVX2__
    if (n_bpr_pf > 0) {
        pf_xq = (int8_t *)sh_arena_alloc(pf_arena, nt * dim);
        pf_xd = (float *)sh_arena_alloc(pf_arena, nt * n_bpr_pf * sizeof(float));
        pf_xbs = (int16_t *)sh_arena_alloc(pf_arena, nt * n_bpr_pf * 16 * sizeof(int16_t));
        if (!pf_xq || !pf_xd || !pf_xbs)
            pf_xq = NULL;  /* fall back to per-matmul quantization */
    }
#endif

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

            // Batch QKV matmul — quantize Xb to Q8K once, reuse for Q, K, V
#ifdef __AVX2__
            if (pf_xq && !m->gpu &&
                (lw->wq.type == BN_GGUF_TENSOR_Q4_K || lw->wq.type == BN_GGUF_TENSOR_Q6_K)) {
                int n_bpr = dim / BN_QK_K;
                for (int t = 0; t < n_tokens; t++)
                    bn_quant_x_to_q8k(Xb + (size_t)t * dim, pf_xq + (size_t)t * dim,
                                       pf_xd + (size_t)t * n_bpr,
                                       pf_xbs + (size_t)t * n_bpr * 16, dim);
                bn_quant_matmul_preq8k(Q_buf, &lw->wq, n_tokens, pf_xq, pf_xd, pf_xbs, Xb, m->pool);
                bn_quant_matmul_preq8k(K_new, &lw->wk, n_tokens, pf_xq, pf_xd, pf_xbs, Xb, m->pool);
                bn_quant_matmul_preq8k(V_new, &lw->wv, n_tokens, pf_xq, pf_xd, pf_xbs, Xb, m->pool);
            } else
#endif
            {
                bn_quant_matmul_gpu(Q_buf, &lw->wq, Xb, n_tokens, s->x_q, m->pool, m->gpu);
                bn_quant_matmul_gpu(K_new, &lw->wk, Xb, n_tokens, s->x_q, m->pool, m->gpu);
                bn_quant_matmul_gpu(V_new, &lw->wv, Xb, n_tokens, s->x_q, m->pool, m->gpu);
            }

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
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_avx2_range : bn_transformer_gqa_avx2_range;
#elif defined(__wasm_simd128__)
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_wasm_range : bn_transformer_gqa_wasm_range;
#else
                bn_tp_fn attn_fn = c->flash_attn ? bn_transformer_flash_gqa_scalar_range : bn_transformer_gqa_scalar_range;
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
                bn_quant_matmul_gpu(Xb2, &lw->wo, Q_buf, n_tokens, s->x_q, m->pool, m->gpu);
            }

            // Batch residual add
            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb2[(size_t)t * dim + d];

        } else if (!is_attn) {
            // --- SSM block: batched projections + sequential recurrence ---
            // SSM config
            int num_k_heads = c->ssm_group_count;
            int head_k_dim  = c->ssm_state_size;
            int num_v_heads = c->ssm_time_step_rank;
            int head_v_dim  = c->ssm_inner_size / (num_v_heads > 0 ? num_v_heads : 1);
            int key_dim_ssm = num_k_heads * head_k_dim;
            int value_dim   = c->ssm_inner_size;
            int qkv_dim_ssm = key_dim_ssm * 2 + value_dim;
            int kern_ssm    = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;

            int ssm_idx = l - (l + 1) / c->full_attn_interval;
            size_t state_per_layer = (size_t)num_v_heads * head_k_dim * head_v_dim;
            float *ssm_state = s->ssm_state + (size_t)ssm_idx * state_per_layer;
            size_t conv_per_layer = (size_t)(kern_ssm - 1) * qkv_dim_ssm;
            float *conv_state = s->ssm_conv_state + (size_t)ssm_idx * conv_per_layer;

            // 1. Batch RMSNorm
            for (int t = 0; t < n_tokens; t++)
                rmsnorm(Xb + (size_t)t * dim, act + (size_t)t * dim, lw->attn_norm, dim, c->norm_eps);

            // 2. Batch matmul: wqkv and wz (the two big projections)
            // QKV_all[n_tokens * qkv_dim_ssm], Z_all[n_tokens * value_dim]
            // Reuse Q_buf and K_new/V_new scratch from batch_buf
            float *QKV_all = Q_buf;   // large enough: q_buf_stride >= qkv_dim_ssm for this model
            float *Z_all   = Xb2;     // [nt * dim] >= [nt * value_dim]
            float *Out_all = Hb;      // [nt * hidden_dim] >= [nt * value_dim]

            bn_quant_matmul_gpu(QKV_all, &lw->wqkv, Xb, n_tokens, s->x_q, m->pool, m->gpu);
            bn_quant_matmul_gpu(Z_all, &lw->wz, Xb, n_tokens, s->x_q, m->pool, m->gpu);

            // 3. Sequential per-token: conv1d + L2norm + alpha/beta + delta + gate
            for (int t = 0; t < n_tokens; t++) {
                float *qkv_t = QKV_all + (size_t)t * lw->wqkv.rows;
                float *z_t   = Z_all + (size_t)t * lw->wz.rows;
                float *out_t = Out_all + (size_t)t * value_dim;
                float *xb_t  = Xb + (size_t)t * dim;  // normed input for alpha/beta

                // Conv1d + SiLU
                {
                    BnSSMConvCtx conv_ctx = { qkv_t, conv_state, lw->ssm_conv1d, qkv_dim_ssm, kern_ssm };
                    BnTPTask conv_task = { ssm_conv_silu, &conv_ctx, qkv_dim_ssm };
                    bn_tp_dispatch(m->pool, &conv_task, 1);
                }

                // Split QKV
                float *q_raw = qkv_t;
                float *k_raw = qkv_t + key_dim_ssm;
                float *v_raw = qkv_t + 2 * key_dim_ssm;

                // L2 normalize Q and K
                {
                    BnSSML2NormCtx norm_ctx = { q_raw, k_raw, head_k_dim };
                    BnTPTask norm_task = { ssm_l2norm, &norm_ctx, num_k_heads };
                    bn_tp_dispatch(m->pool, &norm_task, 1);
                }

                // Alpha + Beta (small matvecs: xb_t → num_v_heads)
                float alpha_arr[num_v_heads > 0 ? num_v_heads : 1];
                float beta_arr[num_v_heads > 0 ? num_v_heads : 1];
                {
                    BnMatvecTask ab[2] = {
                        { alpha_arr, &lw->ssm_alpha },
                        { beta_arr,  &lw->ssm_beta  },
                    };
                    bn_quant_matvec_batch(ab, 2, xb_t, s->x_q, m->pool);
                }
                for (int h = 0; h < num_v_heads; h++) {
                    float dt = alpha_arr[h] + lw->ssm_dt_bias[h];
                    float dt_sp = (dt > 20.0f) ? dt : logf(1.0f + expf(dt));
                    alpha_arr[h] = expf(dt_sp * lw->ssm_a[h]);
                    beta_arr[h] = 1.0f / (1.0f + expf(-beta_arr[h]));
                }

                // Delta rule recurrence
                {
                    float q_scale = 1.0f / sqrtf((float)head_k_dim);
                    BnSSMDeltaCtx delta_ctx = {
                        ssm_state, out_t, q_raw, k_raw, v_raw,
                        alpha_arr, beta_arr,
                        num_k_heads, head_k_dim, head_v_dim, q_scale
                    };
                    BnTPTask delta_task = { ssm_delta, &delta_ctx, num_v_heads };
                    bn_tp_dispatch(m->pool, &delta_task, 1);
                }

                // Per-head RMSNorm + SiLU gate
                {
                    BnSSMGateCtx gate_ctx = { out_t, z_t, lw->ssm_norm, c->norm_eps, head_v_dim };
                    BnTPTask gate_task = { ssm_gate, &gate_ctx, num_v_heads };
                    bn_tp_dispatch(m->pool, &gate_task, 1);
                }
            }

            // 4. Batch matmul: ssm_out (all tokens)
            bn_quant_matmul_gpu(Xb, &lw->ssm_out, Out_all, n_tokens, s->x_q, m->pool, m->gpu);

            // 5. Batch residual add
            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb[(size_t)t * dim + d];

            // Fall through to FFN block (not continue — SSM no longer includes FFN)
        }

        // --- FFN block ---
        if (lw->router_weight) {
            // Batch MoE: route all tokens, group by expert, batch matmul
            if (bn_moe_forward_batch(m, sess, lw, l, act, Xb, n_tokens) != 0) {
                sh_arena_free(pf_arena); return NULL;
            }
        } else if (lw->ffn_up.data) {
            // Dense FFN: batched matmul
            // Batch RMSNorm
            for (int t = 0; t < n_tokens; t++)
                rmsnorm(Xb + t * dim, act + (size_t)t * dim, lw->ffn_norm, dim, c->norm_eps);

            if (c->has_ffn_gate) {
                if (pf_xq && !m->gpu &&
                    (lw->ffn_gate.type == BN_GGUF_TENSOR_Q4_K || lw->ffn_gate.type == BN_GGUF_TENSOR_Q6_K)) {
                    int n_bpr = dim / BN_QK_K;
                    for (int t = 0; t < n_tokens; t++)
                        bn_quant_x_to_q8k(Xb + (size_t)t * dim, pf_xq + (size_t)t * dim,
                                           pf_xd + (size_t)t * n_bpr,
                                           pf_xbs + (size_t)t * n_bpr * 16, dim);
                    bn_quant_matmul_preq8k(Hb, &lw->ffn_gate, n_tokens, pf_xq, pf_xd, pf_xbs, Xb, m->pool);
                    bn_quant_matmul_preq8k(Hb2, &lw->ffn_up, n_tokens, pf_xq, pf_xd, pf_xbs, Xb, m->pool);
                } else {
                    bn_quant_matmul_gpu(Hb, &lw->ffn_gate, Xb, n_tokens, s->x_q, m->pool, m->gpu);
                    bn_quant_matmul_gpu(Hb2, &lw->ffn_up, Xb, n_tokens, s->x_q, m->pool, m->gpu);
                }

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
                bn_quant_matmul_gpu(Hb, &lw->ffn_up, Xb, n_tokens, s->x_q, m->pool, m->gpu);
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

            bn_quant_matmul_gpu(Xb, &lw->ffn_down, Hb, n_tokens, s->x_q, m->pool, m->gpu);

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
            float *lg = forward_logits(m, sess);
            if (!lg) { sh_arena_free(pf_arena); return NULL; }
            memcpy(all_logits + (size_t)t * vocab_size, lg, vocab_size * sizeof(float));
        }
        // Last token's logits already computed and in s->logits
        sh_arena_free(pf_arena);
        return s->logits;
    }

    memcpy(s->x, act + (size_t)(n_tokens - 1) * dim, dim * sizeof(float));
    sh_arena_free(pf_arena);
    return forward_logits(m, sess);
}

float *bn_transformer_prefill(BnModel *m, BnSession *s, const int *tokens, int n_tokens, int pos0) {
    return prefill_internal(m, s, tokens, n_tokens, pos0, NULL);
}

int bn_transformer_prefill_all(BnModel *m, BnSession *s, const int *tokens, int n_tokens,
                                int pos0, float *all_logits) {
    if (!all_logits || n_tokens <= 0) return -1;

    // Single token: just forward
    if (n_tokens == 1) {
        float *logits = bn_transformer_forward(m, s, tokens[0], pos0);
        if (!logits) return -1;
        memcpy(all_logits, logits, m->config.vocab_size * sizeof(float));
        return 0;
    }

    float *result = prefill_internal(m, s, tokens, n_tokens, pos0, all_logits);
    return result ? 0 : -1;
}
