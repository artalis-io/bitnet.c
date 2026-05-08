#include "quant_internal.h"

void bn_quant_f32_scalar_range(void *ctx, int row_start, int row_end) {
    BnF32Ctx *c = (BnF32Ctx *)ctx;
    const float *data = (const float *)c->W->data;
    int cols = c->W->cols;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const float *w = data + (size_t)row * cols;
        float row_sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            row_sum += w[col] * x[col];
        }
        c->out[row] = row_sum;
    }
}
