#include "transformer_logits_internal.h"
#include "transformer_plan_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "backend_model.h"
#include "backend_quant.h"
#include "model.h"
#include "session.h"
#include "sh_log.h"
#include "transformer_backend_internal.h"
#include "gpu_backend.h"
#include "quant.h"
#include <stdlib.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

#ifdef __ARM_NEON
#define rmsnorm bn_transformer_rmsnorm_neon
#elif defined(__AVX2__)
#define rmsnorm bn_transformer_rmsnorm_avx2
#elif defined(__wasm_simd128__)
#define rmsnorm bn_transformer_rmsnorm_wasm
#else
#define rmsnorm bn_transformer_rmsnorm_scalar
#endif

#define BN_LOGITS_MAX_VLA_ELEMS 8192
#define BN_LOGITS_REFINE_MAX_SCALE_BLOCKS 512

static int logits_refine_q8_top(float *logits, int n_logits,
                                const BnQWeight *W, const float *x,
                                int8_t *x_q, int top_n) {
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || \
    defined(__AVX2__) || defined(__wasm_relaxed_simd__)
    if (!logits || !W || !W->data || !x || !x_q ||
        W->type != BN_GGUF_TENSOR_Q8_0)
        return 0;
    if (top_n <= 0) return 0;
    if (top_n > 128) top_n = 128;
    if (top_n > n_logits) top_n = n_logits;
    int n_blocks = W->cols / 32;
    if (n_blocks <= 0 || n_blocks > BN_LOGITS_REFINE_MAX_SCALE_BLOCKS)
        return 0;
    int n_blocks_per_row = n_blocks;

    int ids[128];
    float vals[128];
    int n_top = 0;
    for (int i = 0; i < n_logits; i++) {
        float v = logits[i];
        int j = n_top;
        if (j == top_n && v <= vals[j - 1]) continue;
        if (j < top_n) {
            ids[j] = i;
            vals[j] = v;
            n_top++;
        } else {
            j--;
        }
        while (j > 0 && v > vals[j - 1]) {
            ids[j] = ids[j - 1];
            vals[j] = vals[j - 1];
            j--;
        }
        ids[j] = i;
        vals[j] = v;
    }

    float x_scales[BN_LOGITS_REFINE_MAX_SCALE_BLOCKS];
    bn_quant_x_to_q8_blocks(x, x_q, x_scales, W->cols);
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)W->data;
    for (int i = 0; i < n_top; i++) {
        int row = ids[i];
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk =
                &blocks[(size_t)row * (size_t)n_blocks_per_row + (size_t)b];
            const int8_t *xb = x_q + b * 32;
            int32_t sumi = 0;
            for (int j = 0; j < 32; j++)
                sumi += (int32_t)blk->qs[j] * (int32_t)xb[j];
            row_sum += (float)sumi * bn_fp16_to_fp32(blk->d) * x_scales[b];
        }
        logits[row] = row_sum;
    }
    return n_top;
#else
    (void)logits; (void)n_logits; (void)W; (void)x; (void)x_q; (void)top_n;
    return 0;
#endif
}

static int logits_small_qwen_cuda_q8_refine_enabled(const BnModel *m,
                                                    const BnQWeight *W) {
    if (!m || !W || W->type != BN_GGUF_TENSOR_Q8_0)
        return 0;
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnConfig *c = &m->config;
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           (c->arch_flags & BN_MODEL_ARCH_FLAG_QWEN) &&
           c->n_experts <= 0 &&
           c->full_attn_interval <= 0 &&
           c->dim <= 2560 &&
           getenv("BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE") != NULL &&
           getenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE") == NULL;
}

static void logits_refine_small_qwen_cuda_q8(const BnModel *m,
                                             BnRunState *s,
                                             const BnQWeight *W) {
    if (!logits_small_qwen_cuda_q8_refine_enabled(m, W))
        return;
    int refine_top = 16;
    const char *env = getenv("BN_GPU_Q8_REFINE_TOP");
    if (env) refine_top = atoi(env);
    if (refine_top > 0)
        logits_refine_q8_top(s->logits, m->config.vocab_size, W, s->x,
                             s->x_q, refine_top);
}

#if !((defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__) || defined(__wasm_relaxed_simd__))
static float logits_quant_x_to_i8_scalar(const float *x, int8_t *x_q, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = x[i] < 0.0f ? -x[i] : x[i];
        if (ax > amax) amax = ax;
    }
    float scale = amax / 127.0f;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int i = 0; i < n; i++) {
        int q = (int)(x[i] * inv + (x[i] >= 0.0f ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        x_q[i] = (int8_t)q;
    }
    return scale;
}
#endif

static inline void *qweight_backend_buf(const BnBackendModel *backend,
                                        const BnQWeight *w) {
    return bn_backend_model_qweight_buf(backend, w);
}

static void logits_quant_matvec_gpu(const BnModel *m,
                                    float *out,
                                    const BnQWeight *W,
                                    const float *x,
                                    int8_t *x_q_buf) {
    const BnBackendModel *backend = bn_model_backend(m);
    const BnPreparedWeight *prepared =
        bn_backend_model_prepared_qweight(backend, W);
    bn_backend_quant_matvec_gpu_buf_prepared(out, W, prepared,
                                             qweight_backend_buf(backend, W),
                                             x, x_q_buf, bn_model_pool(m),
                                             bn_model_gpu(m));
}

static int logits_i8_dispatch(BnModel *m, BnRunState *s, int rows, int dim) {
    float x_scale;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    bn_tp_fn fn = bn_transformer_logits_i8_neon_range;
#elif defined(__AVX2__)
    bn_tp_fn fn = bn_transformer_logits_i8_avx2_range;
#elif defined(__wasm_relaxed_simd__)
    bn_tp_fn fn = bn_transformer_logits_i8_wasm_range;
    x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
#else
    bn_tp_fn fn = bn_transformer_logits_i8_scalar_range;
    x_scale = logits_quant_x_to_i8_scalar(s->x, s->x_q, dim);
#endif
    BnWeights *w = &m->weights;
    if (!w->emb_out_i8) return 0;
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__)
    x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
#endif
    BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                           s->x_q, x_scale, dim };
    BnTPTask logits_task = { fn, &lctx, rows };
    bn_tp_dispatch(bn_model_pool(m), &logits_task, 1);
    return 1;
}

static void logits_f16_dispatch(BnModel *m,
                                BnRunState *s,
                                const uint16_t *emb,
                                int rows,
                                int dim) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    uint16_t x_f16[dim];
    for (int d = 0; d < dim; d += 8) {
        float16x4_t lo = vcvt_f16_f32(vld1q_f32(s->x + d));
        float16x4_t hi = vcvt_f16_f32(vld1q_f32(s->x + d + 4));
        vst1q_u16(x_f16 + d, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
    }
    BnLogitsCtx lctx = { s->logits, (const float *)(void *)x_f16, emb, dim };
    BnTPTask logits_task = { bn_transformer_logits_f16_native_neon_range, &lctx, rows };
    bn_tp_dispatch(bn_model_pool(m), &logits_task, 1);
#elif defined(__ARM_NEON)
    BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
    BnTPTask logits_task = { bn_transformer_logits_f16_neon_range, &lctx, rows };
    bn_tp_dispatch(bn_model_pool(m), &logits_task, 1);
#elif defined(__AVX2__)
    BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
    BnTPTask logits_task = { bn_transformer_logits_f16_avx2_range, &lctx, rows };
    bn_tp_dispatch(bn_model_pool(m), &logits_task, 1);
#elif defined(__wasm_simd128__)
    BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
    BnTPTask logits_task = { bn_transformer_logits_f16_wasm_range, &lctx, rows };
    bn_tp_dispatch(bn_model_pool(m), &logits_task, 1);
#else
    BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
    BnTPTask logits_task = { bn_transformer_logits_f16_scalar_range, &lctx, rows };
    bn_tp_dispatch(bn_model_pool(m), &logits_task, 1);
#endif
}

float *bn_transformer_forward_logits(BnModel *m, BnSession *sess) {
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &sess->state;
    int dim = c->dim;

    if (dim > BN_LOGITS_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("Model dim too large for stack VLAs");
        return NULL;
    }

    rmsnorm(s->x, s->x, w->output_norm, dim, c->norm_eps);

    if (w->output_weight.data && w->output_weight.type == BN_GGUF_TENSOR_F16) {
        int n_rows = w->output_weight.rows;
        if (!logits_i8_dispatch(m, s, n_rows, dim))
            logits_f16_dispatch(m, s, (const uint16_t *)w->output_weight.data, n_rows, dim);
    } else if (w->output_weight.data) {
        logits_quant_matvec_gpu(m, s->logits, &w->output_weight, s->x, s->x_q);
        logits_refine_small_qwen_cuda_q8(m, s, &w->output_weight);
    } else if (bn_quant_format_supported(w->emb_type) &&
               w->emb_type != BN_GGUF_TENSOR_F16 &&
               w->emb_type != BN_GGUF_TENSOR_F32) {
        const BnQWeight *tied = &w->tied_embedding_weight;
        const BnBackendModel *backend = bn_model_backend(m);
        const BnPreparedWeight *prepared =
            bn_backend_model_prepared_qweight(backend, tied);
        bn_backend_quant_matvec_gpu_buf_prepared(
            s->logits, tied, prepared,
            bn_transformer_backend_handle_or(bn_model_backend(m), -1,
                                             BN_BACKEND_HANDLE_TIED_EMBEDDING),
            s->x, s->x_q, bn_model_pool(m), bn_model_gpu(m));
        logits_refine_small_qwen_cuda_q8(m, s, tied);
    } else if (w->emb_type == BN_GGUF_TENSOR_F16) {
        if (!logits_i8_dispatch(m, s, c->vocab_size, dim))
            logits_f16_dispatch(m, s, (const uint16_t *)w->token_embedding,
                                c->vocab_size, dim);
    } else {
        const float *emb = (const float *)w->token_embedding;
        BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
        BnTPTask logits_task = { bn_transformer_logits_f32_range, &lctx, c->vocab_size };
        bn_tp_dispatch(bn_model_pool(m), &logits_task, 1);
    }

    return s->logits;
}
