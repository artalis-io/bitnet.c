#include "transformer_logits_internal.h"
#include "transformer_plan_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "backend_model.h"
#include "backend_quant.h"
#include "model.h"
#include "model_arch.h"
#include "session.h"
#include "sh_log.h"
#include "transformer_backend_internal.h"
#include "gpu_backend.h"
#include "quant.h"
#include <math.h>
#include <stdlib.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX512F__
#undef __AVX512BW__
#undef __AVX512VNNI__
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

#define BN_LOGITS_MAX_VLA_ELEMS 8192
#define BN_LOGITS_REFINE_MAX_SCALE_BLOCKS 512

typedef struct {
    void (*rmsnorm)(float *out, const float *x, const float *w,
                    int size, float eps);
    bn_tp_fn i8_logits;
    int i8_uses_standard_quant;
    int supports_q8_refine;
    bn_tp_fn f16_logits;
    void (*prepare_f16_x)(uint16_t *dst, const float *src, int dim);
} BnLogitsBackendOps;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
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

#ifdef __ARM_NEON
static const BnLogitsBackendOps BN_LOGITS_BACKEND = {
    bn_transformer_rmsnorm_neon,
#ifdef __ARM_FEATURE_DOTPROD
    bn_transformer_logits_i8_neon_range,
    1,
    1,
#else
    bn_transformer_logits_i8_scalar_range,
    0,
    0,
#endif
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    bn_transformer_logits_f16_native_neon_range,
    logits_prepare_f16_x_neon,
#else
    bn_transformer_logits_f16_neon_range,
    NULL,
#endif
};
#elif defined(__AVX512F__) && defined(__AVX512BW__) && \
      defined(__AVX512VNNI__) && defined(__AVX2__)
static const BnLogitsBackendOps BN_LOGITS_BACKEND = {
    bn_transformer_rmsnorm_avx2,
    bn_transformer_logits_i8_avx2_range,
    1,
    1,
    bn_transformer_logits_f16_avx2_range,
    NULL,
};
#elif defined(__AVX2__)
static const BnLogitsBackendOps BN_LOGITS_BACKEND = {
    bn_transformer_rmsnorm_avx2,
    bn_transformer_logits_i8_avx2_range,
    1,
    1,
    bn_transformer_logits_f16_avx2_range,
    NULL,
};
#elif defined(__wasm_relaxed_simd__)
static const BnLogitsBackendOps BN_LOGITS_BACKEND = {
    bn_transformer_rmsnorm_wasm,
    bn_transformer_logits_i8_wasm_range,
    1,
    1,
    bn_transformer_logits_f16_wasm_range,
    NULL,
};
#elif defined(__wasm_simd128__)
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

static const BnLogitsBackendOps *logits_backend_ops(void) {
    return &BN_LOGITS_BACKEND;
}

static void logits_rmsnorm_model(const BnModel *m, float *out,
                                 const float *x, const float *w,
                                 int size, float eps) {
    if (m &&
        bn_model_arch_rmsnorm_mode(&m->config) ==
            BN_MODEL_ARCH_RMSNORM_LLAMA_SCALAR_ORDER) {
        double ss = 0.0;
        for (int i = 0; i < size; i++)
            ss += (double)(x[i] * x[i]);
        float scale = 1.0f / sqrtf((float)(ss / (double)size) + eps);
        for (int i = 0; i < size; i++)
            out[i] = x[i] * scale * w[i];
        return;
    }
    logits_backend_ops()->rmsnorm(out, x, w, size, eps);
}

static int logits_refine_q8_top(float *logits, int n_logits,
                                const BnQWeight *W, const float *x,
                                int8_t *x_q, int top_n) {
    if (!logits_backend_ops()->supports_q8_refine)
        return 0;
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
}

static int logits_top_ids(const float *logits, int n_logits,
                          int *ids, float *vals, int top_n) {
    if (!logits || !ids || !vals || n_logits <= 0 || top_n <= 0)
        return 0;
    if (top_n > 128) top_n = 128;
    if (top_n > n_logits) top_n = n_logits;

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
    return n_top;
}

static float logits_q6k_row_native(const BnQWeight *W, const float *x,
                                   int row) {
    int n_blocks_per_row = W->cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)W->data;
    float row_sum = 0.0f;

    for (int b = 0; b < n_blocks_per_row; b++) {
        const BnBlockQ6K *blk =
            &blocks[(size_t)row * (size_t)n_blocks_per_row + (size_t)b];
        float d = bn_fp16_to_fp32(blk->d);
        const uint8_t *ql = blk->ql;
        const uint8_t *qh = blk->qh;
        const int8_t *sc = blk->scales;
        const float *xb = x + b * BN_QK_K;

        for (int n = 0; n < BN_QK_K; n += 128) {
            for (int is = 0; is < 2; is++) {
                float sum1 = 0.0f;
                float sum2 = 0.0f;
                float sum3 = 0.0f;
                float sum4 = 0.0f;
                int l0 = is * 16;
                for (int i = 0; i < 16; i++) {
                    int l = l0 + i;
                    int q1 = (int)((ql[l]      & 0xF) |
                                   (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0xF) |
                                   (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l]      >> 4) |
                                   (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4) |
                                   (((qh[l] >> 6) & 3) << 4)) - 32;
                    sum1 += (float)q1 * xb[l +  0];
                    sum2 += (float)q2 * xb[l + 32];
                    sum3 += (float)q3 * xb[l + 64];
                    sum4 += (float)q4 * xb[l + 96];
                }
                row_sum += d * ((float)sc[is + 0] * sum1 +
                                (float)sc[is + 2] * sum2 +
                                (float)sc[is + 4] * sum3 +
                                (float)sc[is + 6] * sum4);
            }
            xb += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return row_sum;
}

static void logits_refine_tied_q6k_top(BnModel *m, BnRunState *s,
                                       const BnQWeight *W) {
    if (!m || !s || !W || W->type != BN_GGUF_TENSOR_Q6_K ||
        !getenv("BN_CPU_TIED_Q6K_REFINE_TOP"))
        return;

    int refine_top = atoi(getenv("BN_CPU_TIED_Q6K_REFINE_TOP"));
    if (refine_top <= 0) return;
    if (refine_top > 128) refine_top = 128;

    int ids[128];
    float vals[128];
    int n_top = logits_top_ids(s->logits, m->config.vocab_size,
                               ids, vals, refine_top);
    for (int i = 0; i < n_top; i++)
        s->logits[ids[i]] = logits_q6k_row_native(W, s->x, ids[i]);
}

static void logits_hybrid_tied_q6k_top(BnModel *m, BnRunState *s,
                                       const BnQWeight *W) {
    const char *env = getenv("BN_CPU_TIED_Q6K_HYBRID_TOP");
    if (!m || !s || !W || W->type != BN_GGUF_TENSOR_Q6_K || !env)
        return;

    int top_n = atoi(env);
    if (top_n < 2) return;
    if (top_n > 128) top_n = 128;

    int ids[128];
    float vals[128];
    int n_top = logits_top_ids(s->logits, m->config.vocab_size,
                               ids, vals, top_n);
    if (n_top < 2) return;

    float native_vals[128];
    int native_best = 0;
    int native_second = 1;
    for (int i = 0; i < n_top; i++) {
        native_vals[i] = logits_q6k_row_native(W, s->x, ids[i]);
        if (i == 0) continue;
        if (native_vals[i] > native_vals[native_best]) {
            native_second = native_best;
            native_best = i;
        } else if (native_second == native_best ||
                   native_vals[i] > native_vals[native_second]) {
            native_second = i;
        }
    }

    if (native_best == 0)
        return;

    float backend_margin = vals[0] - vals[1];
    float native_margin = native_vals[native_best] - native_vals[native_second];
    if (native_margin <= backend_margin)
        return;

    for (int i = 0; i < n_top; i++)
        s->logits[ids[i]] = native_vals[i];
}

static int logits_small_qwen_cuda_q8_refine_enabled(const BnModel *m,
                                                    const BnQWeight *W) {
    if (!m || !W || W->type != BN_GGUF_TENSOR_Q8_0)
        return 0;
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnConfig *c = &m->config;
    return gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
           bn_model_arch_allows_small_cuda_q8_logit_refine(c) &&
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
        if (q < -127) q = -127;
        x_q[i] = (int8_t)q;
    }
    return scale;
}

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
    const BnLogitsBackendOps *ops = logits_backend_ops();
    BnWeights *w = &m->weights;
    if (!w->emb_out_i8) return 0;
    float x_scale = ops->i8_uses_standard_quant
        ? bn_quant_x_to_i8(s->x, s->x_q, dim)
        : logits_quant_x_to_i8_scalar(s->x, s->x_q, dim);
    BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                           s->x_q, x_scale, dim };
    BnTPTask logits_task = { ops->i8_logits, &lctx, rows };
    bn_tp_dispatch(bn_model_pool(m), &logits_task, 1);
    return 1;
}

static void logits_f16_dispatch(BnModel *m,
                                BnRunState *s,
                                const uint16_t *emb,
                                int rows,
                                int dim) {
    const BnLogitsBackendOps *ops = logits_backend_ops();
    const float *x = s->x;
    uint16_t x_f16[dim > 0 ? dim : 1];
    if (ops->prepare_f16_x) {
        ops->prepare_f16_x(x_f16, s->x, dim);
        x = (const float *)(void *)x_f16;
    }
    BnLogitsCtx lctx = { s->logits, x, emb, dim };
    BnTPTask logits_task = { ops->f16_logits, &lctx, rows };
    bn_tp_dispatch(bn_model_pool(m), &logits_task, 1);
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

    logits_rmsnorm_model(m, s->x, s->x, w->output_norm, dim, c->norm_eps);

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
        if (getenv("BN_CPU_NATIVE_TIED_LOGITS") != NULL) {
            bn_quant_matvec_prepared_flags(
                s->logits, tied, prepared, s->x, s->x_q, bn_model_pool(m),
                BN_MATVEC_TASK_NATIVE_QUANT);
        } else {
            bn_backend_quant_matvec_gpu_buf_prepared(
                s->logits, tied, prepared,
                bn_transformer_backend_handle_or(bn_model_backend(m), -1,
                                                 BN_BACKEND_HANDLE_TIED_EMBEDDING),
                s->x, s->x_q, bn_model_pool(m), bn_model_gpu(m));
            logits_hybrid_tied_q6k_top(m, s, tied);
            logits_refine_tied_q6k_top(m, s, tied);
        }
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

    if (c->final_logit_softcap != 0.0f) {
        float cap = c->final_logit_softcap;
        float inv = 1.0f / cap;
        for (int i = 0; i < c->vocab_size; i++)
            s->logits[i] = tanhf(s->logits[i] * inv) * cap;
    }

    return s->logits;
}
