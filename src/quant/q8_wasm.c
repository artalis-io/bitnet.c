#include "quant_ctx.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

static inline float q8_wasm_block_scale(const BnQWeight *W,
                                        const BnPreparedWeight *prepared,
                                        const BnBlockQ8_0 *blocks,
                                        size_t block_index) {
    return (prepared && prepared->f32_scales)
        ? prepared->f32_scales[block_index]
        : bn_fp16_to_fp32(blocks[block_index].d);
}

void bn_quant_q8_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ8Ctx *c = (BnQ8Ctx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            size_t block_index = (size_t)row * n_blocks_per_row + b;
            const BnBlockQ8_0 *blk = &blocks[block_index];
            float d = q8_wasm_block_scale(c->W, NULL, blocks, block_index);
            const float *xb = x + b * 32;
            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);
            for (int i = 0; i < 2; i++) {
                v128_t w = wasm_v128_load(blk->qs + i * 16);
                v128_t lo16 = wasm_i16x8_extend_low_i8x16(w);
                v128_t hi16 = wasm_i16x8_extend_high_i8x16(w);
                const float *xp = xb + i * 16;
#ifdef __wasm_relaxed_simd__
                acc0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)),  wasm_v128_load(xp),      acc0);
                acc1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xp + 4),  acc1);
                acc2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)),  wasm_v128_load(xp + 8),  acc2);
                acc3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xp + 12), acc3);
#else
                acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), wasm_v128_load(xp)));
                acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xp + 4)));
                acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), wasm_v128_load(xp + 8)));
                acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xp + 12)));
#endif
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum) * d;
        }
        c->out[row] = row_sum;
    }
}

#ifdef __wasm_relaxed_simd__
void bn_quant_q8_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        int base = row * n_blocks_per_row;
        int b = 0;

        for (; b + 3 < n_blocks_per_row; b += 4) {
            const BnBlockQ8_0 *b0 = &blocks[base + b];
            const BnBlockQ8_0 *b1 = &blocks[base + b + 1];
            const BnBlockQ8_0 *b2 = &blocks[base + b + 2];
            const BnBlockQ8_0 *b3 = &blocks[base + b + 3];

            v128_t a0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(b0->qs), wasm_v128_load(x_q + (b * 32)), wasm_i32x4_splat(0));
            a0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(b0->qs + 16), wasm_v128_load(x_q + (b * 32) + 16), a0);

            v128_t a1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(b1->qs), wasm_v128_load(x_q + ((b + 1) * 32)), wasm_i32x4_splat(0));
            a1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(b1->qs + 16), wasm_v128_load(x_q + ((b + 1) * 32) + 16), a1);

            v128_t a2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(b2->qs), wasm_v128_load(x_q + ((b + 2) * 32)), wasm_i32x4_splat(0));
            a2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(b2->qs + 16), wasm_v128_load(x_q + ((b + 2) * 32) + 16), a2);

            v128_t a3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(b3->qs), wasm_v128_load(x_q + ((b + 3) * 32)), wasm_i32x4_splat(0));
            a3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(b3->qs + 16), wasm_v128_load(x_q + ((b + 3) * 32) + 16), a3);

            int32_t s0 = wasm_i32x4_extract_lane(a0, 0) + wasm_i32x4_extract_lane(a0, 1) +
                         wasm_i32x4_extract_lane(a0, 2) + wasm_i32x4_extract_lane(a0, 3);
            int32_t s1 = wasm_i32x4_extract_lane(a1, 0) + wasm_i32x4_extract_lane(a1, 1) +
                         wasm_i32x4_extract_lane(a1, 2) + wasm_i32x4_extract_lane(a1, 3);
            int32_t s2 = wasm_i32x4_extract_lane(a2, 0) + wasm_i32x4_extract_lane(a2, 1) +
                         wasm_i32x4_extract_lane(a2, 2) + wasm_i32x4_extract_lane(a2, 3);
            int32_t s3 = wasm_i32x4_extract_lane(a3, 0) + wasm_i32x4_extract_lane(a3, 1) +
                         wasm_i32x4_extract_lane(a3, 2) + wasm_i32x4_extract_lane(a3, 3);

            row_sum += q8_wasm_block_scale(c->W, c->prepared, blocks, (size_t)base + b) * x_scales[b] * (float)s0
                     + q8_wasm_block_scale(c->W, c->prepared, blocks, (size_t)base + b + 1) * x_scales[b + 1] * (float)s1
                     + q8_wasm_block_scale(c->W, c->prepared, blocks, (size_t)base + b + 2) * x_scales[b + 2] * (float)s2
                     + q8_wasm_block_scale(c->W, c->prepared, blocks, (size_t)base + b + 3) * x_scales[b + 3] * (float)s3;
        }

        for (; b < n_blocks_per_row; b++) {
            const BnBlockQ8_0 *blk = &blocks[base + b];
            float d_w = q8_wasm_block_scale(c->W, c->prepared, blocks, (size_t)base + b);
            float d_x = x_scales[b];
            const int8_t *xb = x_q + b * 32;

            v128_t acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(blk->qs), wasm_v128_load(xb), wasm_i32x4_splat(0));
            acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                wasm_v128_load(blk->qs + 16), wasm_v128_load(xb + 16), acc);

            int32_t total = wasm_i32x4_extract_lane(acc, 0) + wasm_i32x4_extract_lane(acc, 1) +
                            wasm_i32x4_extract_lane(acc, 2) + wasm_i32x4_extract_lane(acc, 3);
            row_sum += d_w * d_x * (float)total;
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q8_wasm_sdot_4row_range(void *ctx, int group_start, int group_end) {
    BnQ8SdotCtx *c = (BnQ8SdotCtx *)ctx;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)c->W->data;
    const float *w_scales = c->prepared ? c->prepared->f32_scales : NULL;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int group = group_start; group < group_end; group++) {
        int row0 = group * 4;
        int rows_left = c->W->rows - row0;
        if (rows_left >= 4) {
            v128_t sum0 = wasm_f32x4_splat(0.0f);
            v128_t sum1 = wasm_f32x4_splat(0.0f);
            v128_t sum2 = wasm_f32x4_splat(0.0f);
            v128_t sum3 = wasm_f32x4_splat(0.0f);

            const BnBlockQ8_0 *row_blocks0 = &blocks[row0 * n_blocks_per_row];
            const BnBlockQ8_0 *row_blocks1 = row_blocks0 + n_blocks_per_row;
            const BnBlockQ8_0 *row_blocks2 = row_blocks1 + n_blocks_per_row;
            const BnBlockQ8_0 *row_blocks3 = row_blocks2 + n_blocks_per_row;

            for (int b = 0; b < n_blocks_per_row; b++) {
                const int8_t *xb = x_q + b * 32;
                v128_t x0 = wasm_v128_load(xb);
                v128_t x1 = wasm_v128_load(xb + 16);
                float dx = x_scales[b];

                const BnBlockQ8_0 *blk0 = &row_blocks0[b];
                size_t idx0 = (size_t)row0 * n_blocks_per_row + b;
                v128_t acc0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(blk0->qs), x0, wasm_i32x4_splat(0));
                acc0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(blk0->qs + 16), x1, acc0);
                v128_t scale0 = wasm_f32x4_splat((w_scales ? w_scales[idx0] : q8_wasm_block_scale(c->W, c->prepared, blocks, idx0)) * dx);
                sum0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(acc0), scale0, sum0);

                const BnBlockQ8_0 *blk1 = &row_blocks1[b];
                size_t idx1 = idx0 + n_blocks_per_row;
                v128_t acc1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(blk1->qs), x0, wasm_i32x4_splat(0));
                acc1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(blk1->qs + 16), x1, acc1);
                v128_t scale1 = wasm_f32x4_splat((w_scales ? w_scales[idx1] : q8_wasm_block_scale(c->W, c->prepared, blocks, idx1)) * dx);
                sum1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(acc1), scale1, sum1);

                const BnBlockQ8_0 *blk2 = &row_blocks2[b];
                size_t idx2 = idx1 + n_blocks_per_row;
                v128_t acc2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(blk2->qs), x0, wasm_i32x4_splat(0));
                acc2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(blk2->qs + 16), x1, acc2);
                v128_t scale2 = wasm_f32x4_splat((w_scales ? w_scales[idx2] : q8_wasm_block_scale(c->W, c->prepared, blocks, idx2)) * dx);
                sum2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(acc2), scale2, sum2);

                const BnBlockQ8_0 *blk3 = &row_blocks3[b];
                size_t idx3 = idx2 + n_blocks_per_row;
                v128_t acc3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(blk3->qs), x0, wasm_i32x4_splat(0));
                acc3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(blk3->qs + 16), x1, acc3);
                v128_t scale3 = wasm_f32x4_splat((w_scales ? w_scales[idx3] : q8_wasm_block_scale(c->W, c->prepared, blocks, idx3)) * dx);
                sum3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(acc3), scale3, sum3);
            }

            c->out[row0] = bn_wasm_hsum_f32x4(sum0);
            c->out[row0 + 1] = bn_wasm_hsum_f32x4(sum1);
            c->out[row0 + 2] = bn_wasm_hsum_f32x4(sum2);
            c->out[row0 + 3] = bn_wasm_hsum_f32x4(sum3);
        } else {
            for (int r = 0; r < rows_left; r++) {
                float sum = 0.0f;
                const BnBlockQ8_0 *row_blocks = &blocks[(row0 + r) * n_blocks_per_row];
                for (int b = 0; b < n_blocks_per_row; b++) {
                    const int8_t *xb = x_q + b * 32;
                    float dx = x_scales[b];
                    const BnBlockQ8_0 *blk = &row_blocks[b];
                    size_t block_index = (size_t)(row0 + r) * n_blocks_per_row + b;
                    v128_t acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                        wasm_v128_load(blk->qs), wasm_v128_load(xb), wasm_i32x4_splat(0));
                    acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                        wasm_v128_load(blk->qs + 16), wasm_v128_load(xb + 16), acc);
                    int32_t total = wasm_i32x4_extract_lane(acc, 0) + wasm_i32x4_extract_lane(acc, 1) +
                                    wasm_i32x4_extract_lane(acc, 2) + wasm_i32x4_extract_lane(acc, 3);
                    sum += (w_scales ? w_scales[block_index] : q8_wasm_block_scale(c->W, c->prepared, blocks, block_index)) * dx * (float)total;
                }
                c->out[row0 + r] = sum;
            }
        }
    }
}

#else
void bn_quant_q8_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    (void)ctx; (void)row_start; (void)row_end;
}

void bn_quant_q8_wasm_sdot_4row_range(void *ctx, int group_start, int group_end) {
    (void)ctx; (void)group_start; (void)group_end;
}
#endif
