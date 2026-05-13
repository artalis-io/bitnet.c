#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_q5k_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ5KCtx *c = (BnQ5KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ5K *blocks = (const BnBlockQ5K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ5K *blk = &blocks[row * n_blocks_per_row + b];
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *qs = blk->qs;
            const uint8_t *qh = blk->qh;
            const float *xb = x + b * BN_QK_K;

            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

            for (int j = 0; j < BN_QK_K; j += 64) {
                uint8_t sc, m;
                int sub = j / 32;

                #ifdef __wasm_relaxed_simd__
                #define Q5K_WASM_ACC_16(w_vec, xp, vds, vdm) do { \
                    v128_t lo16 = wasm_i16x8_extend_low_i8x16(w_vec); \
                    v128_t hi16 = wasm_i16x8_extend_high_i8x16(w_vec); \
                    v128_t w0f = wasm_f32x4_sub(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), vds), vdm); \
                    v128_t w1f = wasm_f32x4_sub(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), vds), vdm); \
                    v128_t w2f = wasm_f32x4_sub(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), vds), vdm); \
                    v128_t w3f = wasm_f32x4_sub(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), vds), vdm); \
                    acc0 = wasm_f32x4_relaxed_madd(w0f, wasm_v128_load(xp), acc0); \
                    acc1 = wasm_f32x4_relaxed_madd(w1f, wasm_v128_load(xp + 4), acc1); \
                    acc2 = wasm_f32x4_relaxed_madd(w2f, wasm_v128_load(xp + 8), acc2); \
                    acc3 = wasm_f32x4_relaxed_madd(w3f, wasm_v128_load(xp + 12), acc3); \
                } while(0)
                #else
                #define Q5K_WASM_ACC_16(w_vec, xp, vds, vdm) do { \
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
                #endif

                int group = j / 64;
                int bit_lo = group * 2;      // bits 0,2,4,6
                int bit_hi = group * 2 + 1;  // bits 1,3,5,7
                uint8_t lo0[16], lo1[16], hi0[16], hi1[16];
                for (int l = 0; l < 16; l++) {
                    lo0[l] = (qs[l] & 0xF) | (((qh[l] >> bit_lo) & 1) << 4);
                    lo1[l] = (qs[l+16] & 0xF) | (((qh[l+16] >> bit_lo) & 1) << 4);
                    hi0[l] = (qs[l] >> 4) | (((qh[l] >> bit_hi) & 1) << 4);
                    hi1[l] = (qs[l+16] >> 4) | (((qh[l+16] >> bit_hi) & 1) << 4);
                }

                bn_q4k_get_scale_min(sub, blk->scales, &sc, &m);
                v128_t vds = wasm_f32x4_splat(d * sc);
                v128_t vdm = wasm_f32x4_splat(dmin * m);
                Q5K_WASM_ACC_16(wasm_v128_load(lo0), xb + j, vds, vdm);
                Q5K_WASM_ACC_16(wasm_v128_load(lo1), xb + j + 16, vds, vdm);

                bn_q4k_get_scale_min(sub + 1, blk->scales, &sc, &m);
                vds = wasm_f32x4_splat(d * sc);
                vdm = wasm_f32x4_splat(dmin * m);
                Q5K_WASM_ACC_16(wasm_v128_load(hi0), xb + j + 32, vds, vdm);
                Q5K_WASM_ACC_16(wasm_v128_load(hi1), xb + j + 48, vds, vdm);

                #undef Q5K_WASM_ACC_16

                qs += 32;
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum);
        }
        c->out[row] = row_sum;
    }
}
