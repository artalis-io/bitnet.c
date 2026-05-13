#include "quant_ctx.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_q6k_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ6KCtx *c = (BnQ6KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const float *x = c->x;

    const v128_t mask_lo4 = wasm_i8x16_splat(0xF);
    const v128_t mask_2 = wasm_i8x16_splat(3);
    const v128_t bias32 = wasm_i8x16_splat(32);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const float *xb = x + b * BN_QK_K;

            for (int chunk = 0; chunk < 2; chunk++) {
                v128_t ql0 = wasm_v128_load(ql);
                v128_t ql1 = wasm_v128_load(ql + 16);
                v128_t ql2 = wasm_v128_load(ql + 32);
                v128_t ql3 = wasm_v128_load(ql + 48);
                v128_t qh0 = wasm_v128_load(qh);
                v128_t qh1 = wasm_v128_load(qh + 16);

                v128_t w0a = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(ql0, mask_lo4), wasm_i8x16_shl(wasm_v128_and(qh0, mask_2), 4)), bias32);
                v128_t w0b = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(ql1, mask_lo4), wasm_i8x16_shl(wasm_v128_and(qh1, mask_2), 4)), bias32);
                v128_t w1a = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(ql2, mask_lo4), wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh0, 2), mask_2), 4)), bias32);
                v128_t w1b = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(ql3, mask_lo4), wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh1, 2), mask_2), 4)), bias32);
                v128_t w2a = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(wasm_u8x16_shr(ql0, 4), mask_lo4), wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh0, 4), mask_2), 4)), bias32);
                v128_t w2b = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(wasm_u8x16_shr(ql1, 4), mask_lo4), wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh1, 4), mask_2), 4)), bias32);
                v128_t w3a = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(wasm_u8x16_shr(ql2, 4), mask_lo4), wasm_i8x16_shl(wasm_u8x16_shr(qh0, 6), 4)), bias32);
                v128_t w3b = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(wasm_u8x16_shr(ql3, 4), mask_lo4), wasm_i8x16_shl(wasm_u8x16_shr(qh1, 6), 4)), bias32);

                v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
                v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

                #ifdef __wasm_relaxed_simd__
                #define Q6K_WASM_ACC_16(w_vec, xp, scale_val) do { \
                    v128_t vds = wasm_f32x4_splat(d * (float)(scale_val)); \
                    v128_t lo16 = wasm_i16x8_extend_low_i8x16(w_vec); \
                    v128_t hi16 = wasm_i16x8_extend_high_i8x16(w_vec); \
                    acc0 = wasm_f32x4_relaxed_madd(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), vds), wasm_v128_load(xp), acc0); \
                    acc1 = wasm_f32x4_relaxed_madd(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), vds), wasm_v128_load(xp + 4), acc1); \
                    acc2 = wasm_f32x4_relaxed_madd(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), vds), wasm_v128_load(xp + 8), acc2); \
                    acc3 = wasm_f32x4_relaxed_madd(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), vds), wasm_v128_load(xp + 12), acc3); \
                } while(0)
                #else
                #define Q6K_WASM_ACC_16(w_vec, xp, scale_val) do { \
                    v128_t vds = wasm_f32x4_splat(d * (float)(scale_val)); \
                    v128_t lo16 = wasm_i16x8_extend_low_i8x16(w_vec); \
                    v128_t hi16 = wasm_i16x8_extend_high_i8x16(w_vec); \
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), vds), wasm_v128_load(xp))); \
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), vds), wasm_v128_load(xp + 4))); \
                    acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), vds), wasm_v128_load(xp + 8))); \
                    acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), vds), wasm_v128_load(xp + 12))); \
                } while(0)
                #endif

                Q6K_WASM_ACC_16(w0a, xb +   0, sc[0]);
                Q6K_WASM_ACC_16(w0b, xb +  16, sc[1]);
                Q6K_WASM_ACC_16(w1a, xb +  32, sc[2]);
                Q6K_WASM_ACC_16(w1b, xb +  48, sc[3]);
                Q6K_WASM_ACC_16(w2a, xb +  64, sc[4]);
                Q6K_WASM_ACC_16(w2b, xb +  80, sc[5]);
                Q6K_WASM_ACC_16(w3a, xb +  96, sc[6]);
                Q6K_WASM_ACC_16(w3b, xb + 112, sc[7]);

                #undef Q6K_WASM_ACC_16

                v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
                row_sum += bn_wasm_hsum_f32x4(sum);

                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        c->out[row] = row_sum;
    }
}

#ifdef __wasm_relaxed_simd__

// Q6_K SDOT kernel with Q8_K x quantization:
// - Unsigned 6-bit weights (no bias-32 subtract)
// - Integer accumulation within super-block (one x_d per 256 elements)
// - Bias correction via bsums (integer, outside inner loop)
// - Float conversion once per super-block
void bn_quant_q6k_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const v128_t mask_lo4 = wasm_i8x16_splat(0xF);
    const v128_t mask_2 = wasm_i8x16_splat(3);
    const v128_t zero = wasm_i32x4_splat(0);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            float d  = bn_fp16_to_fp32(blk->d);
            float dx = x_d[b];
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            // Integer accumulation
            v128_t sumi4 = wasm_i32x4_splat(0);
            int32_t bias_corr = 0;

            for (int chunk = 0; chunk < 2; chunk++) {
                v128_t ql0 = wasm_v128_load(ql);
                v128_t ql1 = wasm_v128_load(ql + 16);
                v128_t ql2 = wasm_v128_load(ql + 32);
                v128_t ql3 = wasm_v128_load(ql + 48);
                v128_t qh0 = wasm_v128_load(qh);
                v128_t qh1 = wasm_v128_load(qh + 16);

                // Unpack 8 weight vectors — UNSIGNED (0..63), no bias subtract
                v128_t w0a = wasm_v128_or(wasm_v128_and(ql0, mask_lo4),
                    wasm_i8x16_shl(wasm_v128_and(qh0, mask_2), 4));
                v128_t w0b = wasm_v128_or(wasm_v128_and(ql1, mask_lo4),
                    wasm_i8x16_shl(wasm_v128_and(qh1, mask_2), 4));
                v128_t w1a = wasm_v128_or(wasm_v128_and(ql2, mask_lo4),
                    wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh0, 2), mask_2), 4));
                v128_t w1b = wasm_v128_or(wasm_v128_and(ql3, mask_lo4),
                    wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh1, 2), mask_2), 4));
                v128_t w2a = wasm_v128_or(wasm_u8x16_shr(ql0, 4),
                    wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh0, 4), mask_2), 4));
                v128_t w2b = wasm_v128_or(wasm_u8x16_shr(ql1, 4),
                    wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh1, 4), mask_2), 4));
                v128_t w3a = wasm_v128_or(wasm_u8x16_shr(ql2, 4),
                    wasm_i8x16_shl(wasm_u8x16_shr(qh0, 6), 4));
                v128_t w3b = wasm_v128_or(wasm_u8x16_shr(ql3, 4),
                    wasm_i8x16_shl(wasm_u8x16_shr(qh1, 6), 4));

                // SDOT + integer scale accumulation (4 pairs x 2 sub-blocks each)
                v128_t s0a = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb), w0a, zero);
                v128_t s0b = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + 16), w0b, zero);
                sumi4 = wasm_i32x4_add(sumi4, wasm_i32x4_mul(s0a, wasm_i32x4_splat((int32_t)sc[0])));
                sumi4 = wasm_i32x4_add(sumi4, wasm_i32x4_mul(s0b, wasm_i32x4_splat((int32_t)sc[1])));

                v128_t s1a = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + 32), w1a, zero);
                v128_t s1b = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + 48), w1b, zero);
                sumi4 = wasm_i32x4_add(sumi4, wasm_i32x4_mul(s1a, wasm_i32x4_splat((int32_t)sc[2])));
                sumi4 = wasm_i32x4_add(sumi4, wasm_i32x4_mul(s1b, wasm_i32x4_splat((int32_t)sc[3])));

                v128_t s2a = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + 64), w2a, zero);
                v128_t s2b = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + 80), w2b, zero);
                sumi4 = wasm_i32x4_add(sumi4, wasm_i32x4_mul(s2a, wasm_i32x4_splat((int32_t)sc[4])));
                sumi4 = wasm_i32x4_add(sumi4, wasm_i32x4_mul(s2b, wasm_i32x4_splat((int32_t)sc[5])));

                v128_t s3a = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + 96), w3a, zero);
                v128_t s3b = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + 112), w3b, zero);
                sumi4 = wasm_i32x4_add(sumi4, wasm_i32x4_mul(s3a, wasm_i32x4_splat((int32_t)sc[6])));
                sumi4 = wasm_i32x4_add(sumi4, wasm_i32x4_mul(s3b, wasm_i32x4_splat((int32_t)sc[7])));

                // Bias correction: sum(sc[g] * bsum[g]) for this chunk's 8 groups
                for (int g = 0; g < 8; g++)
                    bias_corr += (int32_t)sc[g] * (int32_t)bsums[chunk * 8 + g];

                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }

            // Single float conversion per super-block
            int32_t sumi = wasm_i32x4_extract_lane(sumi4, 0) + wasm_i32x4_extract_lane(sumi4, 1) +
                           wasm_i32x4_extract_lane(sumi4, 2) + wasm_i32x4_extract_lane(sumi4, 3);
            row_sum += d * dx * (float)(sumi - 32 * bias_corr);
        }
        c->out[row] = row_sum;
    }
}

#else
void bn_quant_q6k_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    (void)ctx; (void)row_start; (void)row_end;
}
#endif
