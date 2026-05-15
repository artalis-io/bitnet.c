#include "transformer_logits_internal.h"

void bn_transformer_logits_i8_scalar_range(void *ctx, int v_start, int v_end) {
    BnLogitsI8Ctx *lc = (BnLogitsI8Ctx *)ctx;
    const int8_t *emb_i8 = lc->emb_i8;
    const float *emb_scales = lc->emb_scales;
    const int8_t *x_q = lc->x_q;
    float x_scale = lc->x_scale;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const int8_t *row = emb_i8 + (size_t)v * dim;
        int32_t total = 0;
        for (int d = 0; d < dim; d++)
            total += (int32_t)row[d] * (int32_t)x_q[d];
        lc->logits[v] = (float)total * emb_scales[v] * x_scale;
    }
}

void bn_transformer_logits_f16_scalar_range(void *ctx, int v_start, int v_end) {
    BnLogitsCtx *lc = (BnLogitsCtx *)ctx;
    const uint16_t *emb = (const uint16_t *)lc->emb;
    const float *x = lc->x;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const uint16_t *row = emb + (size_t)v * dim;
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += bn_fp16_to_fp32(row[d]) * x[d];
        }
        lc->logits[v] = sum;
    }
}

void bn_transformer_logits_f32_range(void *ctx, int v_start, int v_end) {
    BnLogitsCtx *lc = (BnLogitsCtx *)ctx;
    const float *emb = (const float *)lc->emb;
    const float *x = lc->x;
    int dim = lc->dim;

    for (int v = v_start; v < v_end; v++) {
        const float *row = emb + (size_t)v * dim;
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += row[d] * x[d];
        }
        lc->logits[v] = sum;
    }
}
