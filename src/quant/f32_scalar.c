#include "quant_ctx.h"
#include <math.h>

void bn_quant_f32_scalar_range(void *ctx, int row_start, int row_end) {
    BnF32Ctx *c = (BnF32Ctx *)ctx;
    const float *data = (const float *)c->W->data;
    int cols = c->W->cols;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const float *w = data + (size_t)row * cols;
        float acc[4][4] = {{0.0f}};
        int col = 0;
        for (; col + 15 < cols; col += 16)
            for (int group = 0; group < 4; group++)
                for (int lane = 0; lane < 4; lane++)
                    acc[group][lane] = fmaf(
                        w[col + group * 4 + lane],
                        x[col + group * 4 + lane],
                        acc[group][lane]);

        float lanes[4];
        for (int lane = 0; lane < 4; lane++)
            lanes[lane] = (acc[0][lane] + acc[1][lane]) +
                          (acc[2][lane] + acc[3][lane]);
        float row_sum = (lanes[0] + lanes[2]) +
                        (lanes[1] + lanes[3]);
        for (; col < cols; col++)
            row_sum += w[col] * x[col];
        c->out[row] = row_sum;
    }
}
