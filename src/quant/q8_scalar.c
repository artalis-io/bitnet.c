#include "quant_ctx.h"

void bn_quant_q8_scalar_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        int b = 0;
        for (; b + 3 < n_blocks_per_row; b += 4) {
            float group_sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b + k];
                const int8_t *xb = x_q + (b + k) * 32;
                int32_t sumi = 0;
                for (int i = 0; i < 32; i++)
                    sumi += (int32_t)blk->qs[i] * (int32_t)xb[i];
                group_sum += (float)sumi * bn_fp16_to_fp32(blk->d) * x_scales[b + k];
            }
            row_sum += group_sum;
        }
        for (; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            const int8_t *xb = x_q + b * 32;
            int32_t sumi = 0;
            for (int i = 0; i < 32; i++)
                sumi += (int32_t)blk->qs[i] * (int32_t)xb[i];
            row_sum += (float)sumi * bn_fp16_to_fp32(blk->d) * x_scales[b];
        }
        c->out[row] = row_sum;
    }
}

int bn_quant_q8_logits_refine_row(const BnQWeight *W,
                                  const int8_t *x_q,
                                  const float *x_scales,
                                  int row,
                                  float *out) {
    if (!W || !W->data || !x_q || !x_scales || !out ||
        !bn_quant_format_supports_native_quant_logits_refine(W->type) ||
        row < 0 || row >= W->rows || W->cols <= 0 || (W->cols % 32) != 0)
        return -1;

    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)W->data;
    int n_blocks_per_row = W->cols / 32;
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
    *out = row_sum;
    return 0;
}

void bn_quant_q8_scalar_range(void *ctx, int row_start, int row_end) {
    BnQ8Ctx *c = (BnQ8Ctx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            float block_sum = 0.0f;
            for (int i = 0; i < 32; i++) {
                block_sum += blk->qs[i] * xb[i];
            }
            row_sum += block_sum * d;
        }
        c->out[row] = row_sum;
    }
}

#define Q8_SCALAR_TILE_T 4

void bn_quant_q8_scalar_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantFloatMatmulCtx *c = (BnKQuantFloatMatmulCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->cols / 32;
    int rows = c->W->rows;
    int n_tokens = c->n_tokens;

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < n_tokens; t0 += Q8_SCALAR_TILE_T) {
            int tile_n = t0 + Q8_SCALAR_TILE_T <= n_tokens
                ? Q8_SCALAR_TILE_T : n_tokens - t0;
            float sums[Q8_SCALAR_TILE_T] = {0};

            for (int b = 0; b < n_blocks_per_row; b++) {
                const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
                float d = bn_fp16_to_fp32(blk->d);
                for (int ti = 0; ti < tile_n; ti++) {
                    const float *xb = c->x + (size_t)(t0 + ti) * c->cols + b * 32;
                    float block_sum = 0.0f;
                    for (int i = 0; i < 32; i++)
                        block_sum += (float)blk->qs[i] * xb[i];
                    sums[ti] += block_sum * d;
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] += sums[ti];
        }
    }
}

void bn_quant_q8_scalar_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnQ8MatmulCtx *c = (BnQ8MatmulCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->cols / 32;
    int rows = c->W->rows;
    int n_tokens = c->n_tokens;

    (void)c->prepared;
    for (int row = row_start; row < row_end; row++) {
        for (int t = 0; t < n_tokens; t++) {
            const int8_t *x_q = c->x_q + (size_t)t * c->cols;
            const float *x_scales = c->x_scales + (size_t)t * n_blocks_per_row;
            float row_sum = 0.0f;
            int b = 0;
            for (; b + 3 < n_blocks_per_row; b += 4) {
                float group_sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b + k];
                    const int8_t *xb = x_q + (b + k) * 32;
                    int32_t sumi = 0;
                    for (int i = 0; i < 32; i++)
                        sumi += (int32_t)blk->qs[i] * (int32_t)xb[i];
                    group_sum += (float)sumi * bn_fp16_to_fp32(blk->d) * x_scales[b + k];
                }
                row_sum += group_sum;
            }
            for (; b < n_blocks_per_row; b++) {
                const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
                const int8_t *xb = x_q + b * 32;
                int32_t sumi = 0;
                for (int i = 0; i < 32; i++)
                    sumi += (int32_t)blk->qs[i] * (int32_t)xb[i];
                row_sum += (float)sumi * bn_fp16_to_fp32(blk->d) * x_scales[b];
            }
            c->out[(size_t)t * rows + row] = row_sum;
        }
    }
}
