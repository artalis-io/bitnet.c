#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "simd_helpers.h"
#include <string.h>
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

                #ifdef __wasm_relaxed_simd__
                #define Q4K_WASM_ACC_16(w_vec, xp, vds, vdm) do { \
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
                #endif

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

#ifdef __wasm_relaxed_simd__

// Q4_K SDOT kernel with Q8_K x quantization:
// - Unsigned nibbles (no bias subtract)
// - Integer accumulation within super-block (one x_d per 256 elements)
// - Min correction via bsums (integer, outside inner loop)
// - Float conversion once per super-block
void bn_quant_q4k_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const v128_t mask_lo = wasm_i8x16_splat(0xF);
    const v128_t zero = wasm_i32x4_splat(0);

    // kmask constants for batch scale decode
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            float dx   = x_d[b];
            const uint8_t *qs = blk->qs;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            // Batch-decode all 8 scales and 8 mins (kmask trick)
            uint32_t utmp[3];
            memcpy(utmp, blk->scales, 12);

            // Extract mins before overwriting utmp[1]
            uint32_t m_lo = utmp[1] & kmask1;
            uint32_t m_hi = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);

            // Extract scales
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[0] &= kmask1;
            const uint8_t *sc = (const uint8_t *)utmp;

            // Min correction via bsums (integer)
            uint8_t mins[8];
            memcpy(mins, &m_lo, 4);
            memcpy(mins + 4, &m_hi, 4);

            int32_t bsum_corr = 0;
            for (int j = 0; j < 8; j++)
                bsum_corr += (int32_t)mins[j] * ((int32_t)bsums[2*j] + (int32_t)bsums[2*j + 1]);

            // Integer accumulation: relaxed SDOT for unsigned nibble × signed Q8_K.
            v128_t sumi4 = wasm_i32x4_splat(0);
            for (int j = 0; j < BN_QK_K; j += 64) {
                int sub = j / 32;

                // Low nibbles (sub-block 'sub'): unsigned 0..15
                v128_t raw0 = wasm_v128_load(qs);
                v128_t raw1 = wasm_v128_load(qs + 16);

                v128_t lo0 = wasm_v128_and(raw0, mask_lo);
                v128_t lo1 = wasm_v128_and(raw1, mask_lo);

                v128_t p0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + j), lo0, zero);
                v128_t p1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + j + 16), lo1, zero);
                v128_t psum = wasm_i32x4_add(p0, p1);
                sumi4 = wasm_i32x4_add(
                    sumi4,
                    wasm_i32x4_mul(psum, wasm_i32x4_splat((int32_t)sc[sub])));

                // High nibbles (sub-block 'sub+1'): unsigned 0..15
                v128_t hi0 = wasm_u8x16_shr(raw0, 4);
                v128_t hi1 = wasm_u8x16_shr(raw1, 4);

                p0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + j + 32), hi0, zero);
                p1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                    wasm_v128_load(xb + j + 48), hi1, zero);
                psum = wasm_i32x4_add(p0, p1);
                sumi4 = wasm_i32x4_add(
                    sumi4,
                    wasm_i32x4_mul(psum, wasm_i32x4_splat((int32_t)sc[sub + 1])));

                qs += 32;
            }

            // Single float conversion per super-block
            int32_t sumi = wasm_i32x4_extract_lane(sumi4, 0) + wasm_i32x4_extract_lane(sumi4, 1) +
                           wasm_i32x4_extract_lane(sumi4, 2) + wasm_i32x4_extract_lane(sumi4, 3);
            row_sum += dx * (d * (float)sumi - dmin * (float)bsum_corr);
        }
        c->out[row] = row_sum;
    }
}

#else
void bn_quant_q4k_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    (void)ctx; (void)row_start; (void)row_end;
}
#endif
