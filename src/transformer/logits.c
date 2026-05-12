#include "transformer_internal.h"
#include "session.h"
#include "sh_log.h"

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

static inline void *qweight_backend_buf(const BnBackendModel *backend,
                                        const BnQWeight *w) {
    return bn_backend_model_qweight_buf(backend, w);
}

static void logits_quant_matvec_gpu(const BnModel *m,
                                    float *out,
                                    const BnQWeight *W,
                                    const float *x,
                                    int8_t *x_q_buf) {
    bn_quant_matvec_gpu_buf(out, W, qweight_backend_buf(m->backend, W),
                            x, x_q_buf, m->pool, bn_model_gpu(m));
}

static int logits_i8_dispatch(BnModel *m, BnRunState *s, int rows, int dim) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    bn_tp_fn fn = bn_transformer_logits_i8_neon_range;
#elif defined(__AVX2__)
    bn_tp_fn fn = bn_transformer_logits_i8_avx2_range;
#elif defined(__wasm_relaxed_simd__)
    bn_tp_fn fn = bn_transformer_logits_i8_wasm_range;
#else
    (void)m;
    (void)s;
    (void)rows;
    (void)dim;
    return 0;
#endif
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__) || defined(__wasm_relaxed_simd__)
    BnWeights *w = &m->weights;
    if (!w->emb_out_i8) return 0;
    float x_scale = bn_quant_x_to_i8(s->x, s->x_q, dim);
    BnLogitsI8Ctx lctx = { s->logits, w->emb_out_i8, w->emb_out_scales,
                           s->x_q, x_scale, dim };
    BnTPTask logits_task = { fn, &lctx, rows };
    bn_tp_dispatch(m->pool, &logits_task, 1);
    return 1;
#endif
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
    bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__ARM_NEON)
    BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
    BnTPTask logits_task = { bn_transformer_logits_f16_neon_range, &lctx, rows };
    bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__AVX2__)
    BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
    BnTPTask logits_task = { bn_transformer_logits_f16_avx2_range, &lctx, rows };
    bn_tp_dispatch(m->pool, &logits_task, 1);
#elif defined(__wasm_simd128__)
    BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
    BnTPTask logits_task = { bn_transformer_logits_f16_wasm_range, &lctx, rows };
    bn_tp_dispatch(m->pool, &logits_task, 1);
#else
    BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
    BnTPTask logits_task = { bn_transformer_logits_f16_scalar_range, &lctx, rows };
    bn_tp_dispatch(m->pool, &logits_task, 1);
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
    } else if (bn_quant_format_supported(w->emb_type) &&
               w->emb_type != BN_GGUF_TENSOR_F16 &&
               w->emb_type != BN_GGUF_TENSOR_F32) {
        BnQWeight tied = { w->token_embedding, w->emb_type, c->vocab_size, dim, 1.0f, NULL, NULL };
        bn_quant_matvec_gpu_buf(s->logits, &tied,
                                bn_transformer_backend_handle_or(m->backend, -1,
                                                                  BN_BACKEND_HANDLE_TIED_EMBEDDING),
                                s->x, s->x_q, m->pool, bn_model_gpu(m));
    } else if (w->emb_type == BN_GGUF_TENSOR_F16) {
        if (!logits_i8_dispatch(m, s, c->vocab_size, dim))
            logits_f16_dispatch(m, s, (const uint16_t *)w->token_embedding,
                                c->vocab_size, dim);
    } else {
        const float *emb = (const float *)w->token_embedding;
        BnLogitsCtx lctx = { s->logits, s->x, emb, dim };
        BnTPTask logits_task = { bn_transformer_logits_f32_range, &lctx, c->vocab_size };
        bn_tp_dispatch(m->pool, &logits_task, 1);
    }

    return s->logits;
}
