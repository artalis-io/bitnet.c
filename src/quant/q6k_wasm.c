#include "quant_internal.h"
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

                #define Q6K_WASM_ACC_16(w_vec, xp, scale_val) do { \
                    v128_t vds = wasm_f32x4_splat(d * (float)(scale_val)); \
                    v128_t lo16 = wasm_i16x8_extend_low_i8x16(w_vec); \
                    v128_t hi16 = wasm_i16x8_extend_high_i8x16(w_vec); \
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), vds), wasm_v128_load(xp))); \
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), vds), wasm_v128_load(xp + 4))); \
                    acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), vds), wasm_v128_load(xp + 8))); \
                    acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), vds), wasm_v128_load(xp + 12))); \
                } while(0)

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
