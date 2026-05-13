#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_q3k_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ3KCtx *c = (BnQ3KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ3K *blocks = (const BnBlockQ3K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ3K *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);

            uint8_t scales[16];
            bn_q3k_unpack_scales(blk->scales, scales);

            const uint8_t *q  = blk->qs;
            const uint8_t *hm = blk->hmask;
            const float *xb = x + b * BN_QK_K;

            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

            int is = 0;
            uint8_t m = 1;
            for (int n = 0; n < BN_QK_K; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    int8_t tmp0[16], tmp1[16];
                    for (int l = 0; l < 16; l++) {
                        tmp0[l] = (int8_t)(((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                        tmp1[l] = (int8_t)(((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    }

                    #ifdef __wasm_relaxed_simd__
                    #define Q3K_WASM_ACC_16(w_vec, xp, vdl) do { \
                        v128_t lo16 = wasm_i16x8_extend_low_i8x16(w_vec); \
                        v128_t hi16 = wasm_i16x8_extend_high_i8x16(w_vec); \
                        acc0 = wasm_f32x4_relaxed_madd(wasm_f32x4_mul( \
                            wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), vdl), wasm_v128_load(xp), acc0); \
                        acc1 = wasm_f32x4_relaxed_madd(wasm_f32x4_mul( \
                            wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), vdl), wasm_v128_load(xp + 4), acc1); \
                        acc2 = wasm_f32x4_relaxed_madd(wasm_f32x4_mul( \
                            wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), vdl), wasm_v128_load(xp + 8), acc2); \
                        acc3 = wasm_f32x4_relaxed_madd(wasm_f32x4_mul( \
                            wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), vdl), wasm_v128_load(xp + 12), acc3); \
                    } while(0)
                    #else
                    #define Q3K_WASM_ACC_16(w_vec, xp, vdl) do { \
                        v128_t lo16 = wasm_i16x8_extend_low_i8x16(w_vec); \
                        v128_t hi16 = wasm_i16x8_extend_high_i8x16(w_vec); \
                        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_mul( \
                            wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), vdl), wasm_v128_load(xp))); \
                        acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_mul( \
                            wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), vdl), wasm_v128_load(xp + 4))); \
                        acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_mul( \
                            wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), vdl), wasm_v128_load(xp + 8))); \
                        acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_mul( \
                            wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), vdl), wasm_v128_load(xp + 12))); \
                    } while(0)
                    #endif

                    v128_t vdl0 = wasm_f32x4_splat(d * ((int)scales[is++] - 32));
                    Q3K_WASM_ACC_16(wasm_v128_load(tmp0), xb + n + j*32, vdl0);
                    v128_t vdl1 = wasm_f32x4_splat(d * ((int)scales[is++] - 32));
                    Q3K_WASM_ACC_16(wasm_v128_load(tmp1), xb + n + j*32 + 16, vdl1);

                    #undef Q3K_WASM_ACC_16

                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum);
        }
        c->out[row] = row_sum;
    }
}
