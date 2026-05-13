#include "transformer_logits_internal.h"

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
