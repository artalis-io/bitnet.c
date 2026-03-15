#include "quant_internal.h"
#include <string.h>

void bn_quant_bf16_scalar_range(void *ctx, int row_start, int row_end) {
    BnBF16Ctx *c = (BnBF16Ctx *)ctx;
    const uint16_t *data = (const uint16_t *)c->W->data;
    int cols = c->W->cols;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        const uint16_t *w = data + (size_t)row * cols;
        for (int col = 0; col < cols; col++) {
            uint32_t bits = (uint32_t)w[col] << 16;
            float wf;
            memcpy(&wf, &bits, 4);
            row_sum += wf * x[col];
        }
        c->out[row] = row_sum;
    }
}
