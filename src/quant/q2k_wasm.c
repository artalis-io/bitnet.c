#include "quant_internal.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_q2k_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ2KCtx *c = (BnQ2KCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ2K *blocks = (const BnBlockQ2K *)c->W->data;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ2K *blk = &blocks[row * n_blocks_per_row + b];
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            const uint8_t *q = blk->qs;
            const float *xb = x + b * BN_QK_K;

            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

            int is = 0, out_idx = 0;
            for (int n = 0; n < BN_QK_K; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    int8_t tmp0[16], tmp1[16];
                    for (int l = 0; l < 16; l++) {
                        tmp0[l] = (int8_t)((q[l] >> shift) & 3);
                        tmp1[l] = (int8_t)((q[l + 16] >> shift) & 3);
                    }

                    #define Q2K_WASM_ACC_16(w_vec, xp, vds, vdm) do { \
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

                    {
                        uint8_t sc = blk->scales[is++];
                        v128_t vds = wasm_f32x4_splat(d * (sc & 0xF));
                        v128_t vdm = wasm_f32x4_splat(dmin * (sc >> 4));
                        Q2K_WASM_ACC_16(wasm_v128_load(tmp0), xb + out_idx, vds, vdm);
                        out_idx += 16;
                    }
                    {
                        uint8_t sc = blk->scales[is++];
                        v128_t vds = wasm_f32x4_splat(d * (sc & 0xF));
                        v128_t vdm = wasm_f32x4_splat(dmin * (sc >> 4));
                        Q2K_WASM_ACC_16(wasm_v128_load(tmp1), xb + out_idx, vds, vdm);
                        out_idx += 16;
                    }

                    #undef Q2K_WASM_ACC_16

                    shift += 2;
                }
                q += 32;
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum);
        }
        c->out[row] = row_sum;
    }
}
