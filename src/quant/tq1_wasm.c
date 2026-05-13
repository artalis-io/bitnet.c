#include "quant_ctx.h"
#include "simd_helpers.h"
#include <wasm_simd128.h>

void bn_quant_tq1_wasm_range(void *ctx, int row_start, int row_end) {
    BnTQ1Ctx *c = (BnTQ1Ctx *)ctx;
    const BnBlockTQ1 *blocks = (const BnBlockTQ1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    float tensor_scale = c->W->scale;
    const float *x = c->x;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    const v128_t one = wasm_i8x16_splat(1);
    const v128_t three_u8 = wasm_u8x16_splat(3);

    // Widen int8 ternary to float, multiply-accumulate
    #define TQ1_WASM_ACC(ternary, xbase, a0, a1, a2, a3) do { \
        v128_t lo16 = wasm_i16x8_extend_low_i8x16(ternary);  \
        v128_t hi16 = wasm_i16x8_extend_high_i8x16(ternary); \
        v128_t i0_ = wasm_i32x4_extend_low_i16x8(lo16);      \
        v128_t i1_ = wasm_i32x4_extend_high_i16x8(lo16);     \
        v128_t i2_ = wasm_i32x4_extend_low_i16x8(hi16);      \
        v128_t i3_ = wasm_i32x4_extend_high_i16x8(hi16);     \
        a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i0_), wasm_v128_load(xbase)));     \
        a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i1_), wasm_v128_load(xbase + 4))); \
        a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i2_), wasm_v128_load(xbase + 8))); \
        a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i3_), wasm_v128_load(xbase + 12))); \
    } while(0)

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ1 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * BN_QK_K;

            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);

            // First 160 elements: 5 layers × 32 elements from qs[0..31]
            for (int n = 0; n < 5; n++) {
                v128_t pow3_vec = wasm_u8x16_splat(pow3[n]);
                for (int i = 0; i < 2; i++) {
                    v128_t raw = wasm_v128_load(blk->qs + i * 16);
                    // WASM has no i8x16.mul — emulate via widening multiply + narrow
                    v128_t q = wasm_u8x16_narrow_i16x8(
                        wasm_i16x8_mul(wasm_u16x8_extend_low_u8x16(raw), wasm_u16x8_extend_low_u8x16(pow3_vec)),
                        wasm_i16x8_mul(wasm_u16x8_extend_high_u8x16(raw), wasm_u16x8_extend_high_u8x16(pow3_vec)));
                    // xi = (q * 3) >> 8 via widening multiply
                    v128_t prod_lo = wasm_u16x8_extmul_low_u8x16(q, three_u8);
                    v128_t prod_hi = wasm_u16x8_extmul_high_u8x16(q, three_u8);
                    v128_t xi = wasm_u8x16_narrow_i16x8(
                        wasm_u16x8_shr(prod_lo, 8), wasm_u16x8_shr(prod_hi, 8));
                    v128_t ternary = wasm_i8x16_sub(xi, one);
                    TQ1_WASM_ACC(ternary, xb + n*32 + i*16, acc0, acc1, acc2, acc3);
                }
            }

            // Next 80 elements: 5 layers × 16 elements from qs[32..47]
            for (int n = 0; n < 5; n++) {
                v128_t raw = wasm_v128_load(blk->qs + 32);
                v128_t pow3n = wasm_u8x16_splat(pow3[n]);
                v128_t q = wasm_u8x16_narrow_i16x8(
                    wasm_i16x8_mul(wasm_u16x8_extend_low_u8x16(raw), wasm_u16x8_extend_low_u8x16(pow3n)),
                    wasm_i16x8_mul(wasm_u16x8_extend_high_u8x16(raw), wasm_u16x8_extend_high_u8x16(pow3n)));
                v128_t prod_lo = wasm_u16x8_extmul_low_u8x16(q, three_u8);
                v128_t prod_hi = wasm_u16x8_extmul_high_u8x16(q, three_u8);
                v128_t xi = wasm_u8x16_narrow_i16x8(
                    wasm_u16x8_shr(prod_lo, 8), wasm_u16x8_shr(prod_hi, 8));
                v128_t ternary = wasm_i8x16_sub(xi, one);
                TQ1_WASM_ACC(ternary, xb + 160 + n*16, acc0, acc1, acc2, acc3);
            }

            float block_sum = bn_wasm_hsum_f32x4(
                wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3)));

            // Last 16 elements: 4 layers × 4 elements from qh[0..3] (scalar)
            for (int n = 0; n < 4; n++) {
                for (int m = 0; m < 4; m++) {
                    uint8_t q = blk->qh[m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    block_sum += (xi - 1) * xb[240 + n*4 + m];
                }
            }
            row_sum += block_sum * d;
        }
        c->out[row] = row_sum * tensor_scale;
    }
    #undef TQ1_WASM_ACC
}

#ifdef __wasm_relaxed_simd__
void bn_quant_tq1_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    BnTQ1SdotCtx *c = (BnTQ1SdotCtx *)ctx;
    const BnBlockTQ1 *blocks = (const BnBlockTQ1 *)c->W->data;
    int n_blocks_per_row = c->W->cols / BN_QK_K;
    const int8_t *x_q = c->x_q;

    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    const v128_t three_u8 = wasm_u8x16_splat(3);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;

        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockTQ1 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const int8_t *xb = x_q + b * BN_QK_K;

            v128_t iacc = wasm_i32x4_splat(0);
            v128_t xsum_lo = wasm_i16x8_splat(0);
            v128_t xsum_hi = wasm_i16x8_splat(0);

            // Section 1: qs[0..31] → 160 values (5 trit layers × 32 bytes)
            for (int n = 0; n < 5; n++) {
                v128_t pow3_vec = wasm_u8x16_splat(pow3[n]);
                for (int i = 0; i < 2; i++) {
                    v128_t raw = wasm_v128_load(blk->qs + i * 16);
                    // Multiply raw bytes by pow3[n] via 16-bit widening path
                    v128_t q = wasm_u8x16_narrow_i16x8(
                        wasm_i16x8_mul(wasm_u16x8_extend_low_u8x16(raw), wasm_u16x8_extend_low_u8x16(pow3_vec)),
                        wasm_i16x8_mul(wasm_u16x8_extend_high_u8x16(raw), wasm_u16x8_extend_high_u8x16(pow3_vec)));
                    // Decode top trit: xi = (q * 3) >> 8, gives {0,1,2}
                    v128_t prod_lo = wasm_u16x8_extmul_low_u8x16(q, three_u8);
                    v128_t prod_hi = wasm_u16x8_extmul_high_u8x16(q, three_u8);
                    v128_t xi = wasm_u8x16_narrow_i16x8(
                        wasm_u16x8_shr(prod_lo, 8), wasm_u16x8_shr(prod_hi, 8));

                    // xi is unsigned {0,1,2} — safe for relaxed_dot first operand
                    v128_t xv = wasm_v128_load(xb + n * 32 + i * 16);
                    iacc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(xi, xv, iacc);

                    // Accumulate x_sum for correction: vpadalq_s8 equivalent
                    xsum_lo = wasm_i16x8_add(xsum_lo, wasm_i16x8_extend_low_i8x16(xv));
                    xsum_hi = wasm_i16x8_add(xsum_hi, wasm_i16x8_extend_high_i8x16(xv));
                }
            }

            // Section 2: qs[32..47] → 80 values (5 trit layers × 16 bytes)
            for (int n = 0; n < 5; n++) {
                v128_t raw = wasm_v128_load(blk->qs + 32);
                v128_t pow3n = wasm_u8x16_splat(pow3[n]);
                v128_t q = wasm_u8x16_narrow_i16x8(
                    wasm_i16x8_mul(wasm_u16x8_extend_low_u8x16(raw), wasm_u16x8_extend_low_u8x16(pow3n)),
                    wasm_i16x8_mul(wasm_u16x8_extend_high_u8x16(raw), wasm_u16x8_extend_high_u8x16(pow3n)));
                v128_t prod_lo = wasm_u16x8_extmul_low_u8x16(q, three_u8);
                v128_t prod_hi = wasm_u16x8_extmul_high_u8x16(q, three_u8);
                v128_t xi = wasm_u8x16_narrow_i16x8(
                    wasm_u16x8_shr(prod_lo, 8), wasm_u16x8_shr(prod_hi, 8));

                v128_t xv = wasm_v128_load(xb + 160 + n * 16);
                iacc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(xi, xv, iacc);

                xsum_lo = wasm_i16x8_add(xsum_lo, wasm_i16x8_extend_low_i8x16(xv));
                xsum_hi = wasm_i16x8_add(xsum_hi, wasm_i16x8_extend_high_i8x16(xv));
            }

            // Horizontal sum of xsum accumulators (lane index must be constant)
            v128_t xsum_all = wasm_i16x8_add(xsum_lo, xsum_hi);
            int32_t x_sum = (int16_t)wasm_i16x8_extract_lane(xsum_all, 0)
                          + (int16_t)wasm_i16x8_extract_lane(xsum_all, 1)
                          + (int16_t)wasm_i16x8_extract_lane(xsum_all, 2)
                          + (int16_t)wasm_i16x8_extract_lane(xsum_all, 3)
                          + (int16_t)wasm_i16x8_extract_lane(xsum_all, 4)
                          + (int16_t)wasm_i16x8_extract_lane(xsum_all, 5)
                          + (int16_t)wasm_i16x8_extract_lane(xsum_all, 6)
                          + (int16_t)wasm_i16x8_extract_lane(xsum_all, 7);

            // Section 3: qh[0..3] → 16 values (scalar, same as NEON)
            int32_t qh_dot = 0, qh_xsum = 0;
            static const uint8_t pow3s[] = {1, 3, 9, 27};
            for (int n = 0; n < 4; n++) {
                for (int m = 0; m < 4; m++) {
                    uint8_t q = blk->qh[m] * pow3s[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    int8_t xv = xb[240 + n * 4 + m];
                    qh_dot += xi * xv;
                    qh_xsum += xv;
                }
            }

            x_sum += qh_xsum;
            int32_t total = wasm_i32x4_extract_lane(iacc, 0) + wasm_i32x4_extract_lane(iacc, 1) +
                            wasm_i32x4_extract_lane(iacc, 2) + wasm_i32x4_extract_lane(iacc, 3);
            row_sum += d * (float)(total + qh_dot - x_sum);
        }
        c->out[row] = row_sum * c->combined_scale;
    }
}
#else
void bn_quant_tq1_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    (void)ctx; (void)row_start; (void)row_end;
}
#endif
