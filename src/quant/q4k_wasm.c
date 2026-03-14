#include "quant_internal.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_q4k_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ4KCtx *c = (BnQ4KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const float *x = c->x;

    const v128_t mask_lo = wasm_i8x16_splat(0xF);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[row * n_blocks_per_row + b];
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const float *xb = x + b * BN_QK_K;

            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;

                #define Q4K_WASM_ACC_16(w_vec, xp, vds, vdm) do { \
                    v128_t lo16 = wasm_i16x8_extend_low_i8x16(w_vec); \
                    v128_t hi16 = wasm_i16x8_extend_high_i8x16(w_vec); \
                    v128_t w0f = wasm_f32x4_sub(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), vds), vdm); \
                    v128_t w1f = wasm_f32x4_sub(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), vds), vdm); \
                    v128_t w2f = wasm_f32x4_sub(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), vds), vdm); \
                    v128_t w3f = wasm_f32x4_sub(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), vds), vdm); \
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(w0f, wasm_v128_load(xp))); \
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(w1f, wasm_v128_load(xp + 4))); \
                    acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(w2f, wasm_v128_load(xp + 8))); \
                    acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(w3f, wasm_v128_load(xp + 12))); \
                } while(0)

                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                v128_t vds = wasm_f32x4_splat(d * sc);
                v128_t vdm = wasm_f32x4_splat(dmin * m);
                v128_t raw0 = wasm_v128_load(qs);
                v128_t raw1 = wasm_v128_load(qs + 16);
                v128_t w0 = wasm_v128_and(raw0, mask_lo);
                v128_t w1 = wasm_v128_and(raw1, mask_lo);
                Q4K_WASM_ACC_16(w0, xb + j, vds, vdm);
                Q4K_WASM_ACC_16(w1, xb + j + 16, vds, vdm);

                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                vds = wasm_f32x4_splat(d * sc);
                vdm = wasm_f32x4_splat(dmin * m);
                w0 = wasm_u8x16_shr(raw0, 4);
                w1 = wasm_u8x16_shr(raw1, 4);
                Q4K_WASM_ACC_16(w0, xb + j + 32, vds, vdm);
                Q4K_WASM_ACC_16(w1, xb + j + 48, vds, vdm);

                #undef Q4K_WASM_ACC_16

                qs += 32;
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum);
        }
        c->out[row] = row_sum;
    }
}
