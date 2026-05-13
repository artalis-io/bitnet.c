#include "quant_ctx.h"

void bn_quant_i2s_scalar_range(void *ctx, int row_start, int row_end) {
    BnI2SFloatCtx *c = (BnI2SFloatCtx *)ctx;
    int cols = c->W->cols;
    int row_bytes = cols / 4;
    const uint8_t *base = (const uint8_t *)c->W->data;
    float scale = c->W->scale;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        const uint8_t *rd = base + (size_t)row * row_bytes;
        int done = 0;
        const int8_t imap[4] = {-1, 0, 1, 0};
        float sum = 0.0f;
        while (done < cols) {
            for (int gp = 0; gp < 32; gp++) {
                uint8_t b = rd[gp];
                sum += imap[(b >> 6) & 3] * x[done + 0*32 + gp];
                sum += imap[(b >> 4) & 3] * x[done + 1*32 + gp];
                sum += imap[(b >> 2) & 3] * x[done + 2*32 + gp];
                sum += imap[(b >> 0) & 3] * x[done + 3*32 + gp];
            }
            rd += 32;
            done += 128;
        }
        c->out[row] = sum * scale;
    }
}
