#include "quant_internal.h"
#include "simd_helpers.h"
#include "iq_tables.h"
#include <wasm_simd128.h>

void bn_quant_iq4xs_wasm_range(void *ctx, int row_start, int row_end) {
    BnIQ4XSCtx *c = (BnIQ4XSCtx *)ctx;
    const BnBlockIQ4XS *blocks = (const BnBlockIQ4XS *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockIQ4XS *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;
            const uint8_t *qs = blk->qs;

            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

            for (int j = 0; j < 8; j++) {
                // Extract 6-bit scale
                int lo = (blk->scales_l[j / 2] >> ((j % 2) * 4)) & 0xF;
                int hi = (blk->scales_h >> (j * 2)) & 3;
                float dl = d * ((lo | (hi << 4)) - 32);
                v128_t vdl = wasm_f32x4_splat(dl);

                // Scalar decode through codebook
                int8_t tmp[32];
                for (int i = 0; i < 16; i++) {
                    uint8_t byte = qs[i];
                    tmp[i]      = bn_kvalues_iq4nl[byte & 0xF];
                    tmp[i + 16] = bn_kvalues_iq4nl[byte >> 4];
                }

                // WASM SIMD: widen int8 to float, multiply by scale, FMA with x
                {
                    v128_t w = wasm_v128_load(tmp);
                    v128_t lo16 = wasm_i16x8_extend_low_i8x16(w);
                    v128_t hi16 = wasm_i16x8_extend_high_i8x16(w);
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), vdl), wasm_v128_load(xb)));
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), vdl), wasm_v128_load(xb + 4)));
                    acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), vdl), wasm_v128_load(xb + 8)));
                    acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), vdl), wasm_v128_load(xb + 12)));
                }
                {
                    v128_t w = wasm_v128_load(tmp + 16);
                    v128_t lo16 = wasm_i16x8_extend_low_i8x16(w);
                    v128_t hi16 = wasm_i16x8_extend_high_i8x16(w);
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), vdl), wasm_v128_load(xb + 16)));
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), vdl), wasm_v128_load(xb + 20)));
                    acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), vdl), wasm_v128_load(xb + 24)));
                    acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), vdl), wasm_v128_load(xb + 28)));
                }

                qs += 16;
                xb += 32;
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum);
        }
        c->out[row] = row_sum;
    }
}
