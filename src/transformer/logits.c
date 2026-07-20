#include "transformer_logits_internal.h"
#include "transformer_plan_internal.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "backend_model.h"
#include "model.h"
#include "session.h"
#include "sh_log.h"
#include "transformer_backend_internal.h"
#include "gpu_internal.h"
#include <math.h>

#define BN_LOGITS_MAX_VLA_ELEMS 8192
#define BN_LOGITS_REFINE_MAX_SCALE_BLOCKS 512

static const BnLogitsBackendOps *logits_backend_ops(void) {
    return bn_transformer_logits_backend_ops();
}

static void logits_rmsnorm_model(const BnModel *m, float *out,
                                 const float *x, const float *w,
                                 int size, float eps) {
    if (m && bn_transformer_rmsnorm_uses_reference_order(&m->config)) {
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

static int logits_refine_backend_top(float *logits, int n_logits,
                                     const BnQWeight *W, const float *x,
                                     int8_t *quantized, int top_n) {
    if (!logits || !x || !quantized ||
        !bn_transformer_logits_backend_refine_supported(
            logits_backend_ops(), W))
        return 0;
    if (top_n <= 0) return 0;
    if (top_n > 128) top_n = 128;
    if (top_n > n_logits) top_n = n_logits;
    int n_blocks = W->cols / 32;
    if (n_blocks <= 0 || n_blocks > BN_LOGITS_REFINE_MAX_SCALE_BLOCKS)
        return 0;

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

    float scales[BN_LOGITS_REFINE_MAX_SCALE_BLOCKS];
    bn_transformer_cpu_prepare_native_logits_refine_activation(
        x, quantized, scales, W->cols);
    for (int i = 0; i < n_top; i++) {
        int row = ids[i];
        float row_sum;
        if (bn_transformer_cpu_refine_native_logits_row(
                W, quantized, scales, row, &row_sum) == 0)
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

static void logits_refine_tied_kquant_top(BnModel *m, BnRunState *s,
                                          const BnQWeight *W) {
    if (!m || !s ||
        !bn_transformer_logits_tied_kquant_refine_supported(W))
        return;

    int refine_top = bn_transformer_logits_cpu_tied_kquant_refine_top();
    if (refine_top <= 0) return;

    int ids[128];
    float vals[128];
    int n_top = logits_top_ids(s->logits, m->config.vocab_size,
                               ids, vals, refine_top);
    for (int i = 0; i < n_top; i++) {
        float row_sum;
        if (bn_transformer_cpu_refine_kquant_logits_row(W, s->x, ids[i],
                                                        &row_sum) == 0)
            s->logits[ids[i]] = row_sum;
    }
}

static void logits_hybrid_tied_kquant_top(BnModel *m, BnRunState *s,
                                          const BnQWeight *W) {
    if (!m || !s ||
        !bn_transformer_logits_tied_kquant_refine_supported(W))
        return;

    int top_n = bn_transformer_logits_cpu_tied_kquant_hybrid_top();
    if (top_n <= 0) return;

    int ids[128];
    float vals[128];
    int n_top = logits_top_ids(s->logits, m->config.vocab_size,
                               ids, vals, top_n);
    if (n_top < 2) return;

    float native_vals[128];
    int native_best = 0;
    int native_second = 1;
    for (int i = 0; i < n_top; i++) {
        if (bn_transformer_cpu_refine_kquant_logits_row(W, s->x, ids[i],
                                                        &native_vals[i]) != 0)
            return;
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

static void logits_refine_native_quant(const BnModel *m,
                                        BnRunState *s,
                                        const BnQWeight *W) {
    if (!m || !bn_transformer_logits_native_quant_refine_enabled(
                  bn_model_gpu(m), &m->config, W))
        return;
    int refine_top = bn_transformer_logits_native_quant_refine_top();
    if (refine_top > 0)
        logits_refine_backend_top(s->logits, m->config.vocab_size, W, s->x,
                                  s->x_q, refine_top);
}

static inline void *qweight_backend_buf(const BnBackendModel *backend,
                                        const BnQWeight *w) {
    return bn_backend_model_qweight_buf(backend, w);
}

static void logits_quant_matvec_gpu(const BnModel *m,
                                    float *out,
                                    const BnQWeight *W,
                                    const float *x,
                                    int8_t *quantized_buf) {
    const BnBackendModel *backend = bn_model_backend(m);
    const BnPreparedWeight *prepared =
        bn_backend_model_prepared_qweight(backend, W);
    bn_transformer_logits_quant_matvec_gpu_buffer_prepared(
        out, W, prepared, qweight_backend_buf(backend, W), x, quantized_buf,
        bn_model_pool(m), bn_model_gpu(m));
}

static int logits_i8_dispatch(BnModel *m, BnRunState *s, int rows, int dim) {
    const BnLogitsBackendOps *ops = logits_backend_ops();
    BnWeights *w = &m->weights;
    if (!w->emb_out_i8) return 0;
    float activation_scale =
        bn_transformer_cpu_quantize_i8_activation(
            s->x, s->x_q, dim, ops->i8_uses_standard_quant);
    BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                           s->x_q, activation_scale, dim };
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

    BnLogitsPlan plan;
    bn_transformer_plan_logits(&plan, c, w, bn_model_gpu(m),
                               bn_model_gpu(m) != NULL);

    switch (plan.kind) {
    case BN_LOGITS_UNTIED_F16: {
        int n_rows = w->output_weight.rows;
        if (!logits_i8_dispatch(m, s, n_rows, dim))
            logits_f16_dispatch(m, s, (const uint16_t *)w->output_weight.data, n_rows, dim);
        break;
    }
    case BN_LOGITS_UNTIED_QUANT:
        logits_quant_matvec_gpu(m, s->logits, &w->output_weight, s->x, s->x_q);
        logits_refine_native_quant(m, s, &w->output_weight);
        break;
    case BN_LOGITS_TIED_QUANT: {
        const BnQWeight *tied = &w->tied_embedding_weight;
        const BnBackendModel *backend = bn_model_backend(m);
        const BnPreparedWeight *prepared =
            bn_backend_model_prepared_qweight(backend, tied);
        if (bn_transformer_logits_cpu_native_tied_quant_enabled()) {
            bn_transformer_cpu_quant_matvec_prepared_flags(
                s->logits, tied, prepared, s->x, s->x_q, bn_model_pool(m),
                bn_transformer_logits_native_quant_task_flags(1));
        } else {
            bn_transformer_logits_quant_matvec_gpu_buffer_prepared(
                s->logits, tied, prepared,
                bn_transformer_backend_handle_or(bn_model_backend(m), -1,
                                                 BN_BACKEND_HANDLE_TIED_EMBEDDING),
                s->x, s->x_q, bn_model_pool(m), bn_model_gpu(m));
            logits_hybrid_tied_kquant_top(m, s, tied);
            logits_refine_tied_kquant_top(m, s, tied);
        }
        logits_refine_native_quant(m, s, tied);
        break;
    }
    case BN_LOGITS_TIED_F16:
        if (!logits_i8_dispatch(m, s, c->vocab_size, dim))
            logits_f16_dispatch(m, s, (const uint16_t *)w->token_embedding,
                                c->vocab_size, dim);
        break;
    case BN_LOGITS_TIED_I8:
        if (!logits_i8_dispatch(m, s, c->vocab_size, dim))
            logits_f16_dispatch(m, s, (const uint16_t *)w->token_embedding,
                                c->vocab_size, dim);
        break;
    case BN_LOGITS_TIED_F32: {
        const float *emb = (const float *)w->token_embedding;
        BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
        BnTPTask logits_task = { bn_transformer_logits_f32_range, &lctx, c->vocab_size };
        bn_tp_dispatch(bn_model_pool(m), &logits_task, 1);
        break;
    }
    }

    if (c->final_logit_softcap != 0.0f) {
        float cap = c->final_logit_softcap;
        float inv = 1.0f / cap;
        for (int i = 0; i < c->vocab_size; i++)
            s->logits[i] = tanhf(s->logits[i] * inv) * cap;
    }

    return s->logits;
}
