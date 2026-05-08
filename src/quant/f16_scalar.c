#include "quant_internal.h"

void bn_quant_f16_scalar_range(void *ctx, int row_start, int row_end) {
    BnF16Ctx *c = (BnF16Ctx *)ctx;
    const uint16_t *data = (const uint16_t *)c->W->data;
    int cols = c->W->cols;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const uint16_t *w = data + (size_t)row * cols;
        float row_sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            row_sum += bn_fp16_to_fp32(w[col]) * x[col];
        }
        c->out[row] = row_sum;
    }
}
