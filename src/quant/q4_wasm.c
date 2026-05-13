#include "quant_internal.h"
#include "simd_helpers.h"
#include <math.h>
#include <wasm_simd128.h>

#ifdef __wasm_relaxed_simd__
#define BN_WASM_Q4_LANE_CACHE_MAX_BLOCKS 128

static inline v128_t q4_wasm_fast_exp_f32(v128_t x) {
    const v128_t log2e = wasm_f32x4_splat(1.4426950409f);
    const v128_t ln2 = wasm_f32x4_splat(0.6931471806f);
    const v128_t half = wasm_f32x4_splat(0.5f);
    const v128_t one = wasm_f32x4_splat(1.0f);
    const v128_t p2 = wasm_f32x4_splat(0.49999994f);
    const v128_t p3 = wasm_f32x4_splat(0.16666667f);
    const v128_t p4 = wasm_f32x4_splat(0.04166664f);

    x = wasm_f32x4_max(wasm_f32x4_splat(-87.3f),
                       wasm_f32x4_min(wasm_f32x4_splat(88.7f), x));
    v128_t n = wasm_f32x4_floor(wasm_f32x4_add(wasm_f32x4_mul(x, log2e), half));
    v128_t r = wasm_f32x4_sub(x, wasm_f32x4_mul(n, ln2));

    v128_t poly = wasm_f32x4_add(p3, wasm_f32x4_mul(p4, r));
    poly = wasm_f32x4_add(p2, wasm_f32x4_mul(poly, r));
    poly = wasm_f32x4_add(one, wasm_f32x4_mul(poly, r));
    poly = wasm_f32x4_add(one, wasm_f32x4_mul(poly, r));

    v128_t ni = wasm_i32x4_trunc_sat_f32x4(n);
    v128_t e2n = wasm_i32x4_shl(wasm_i32x4_add(ni, wasm_i32x4_splat(127)), 23);
    return wasm_f32x4_mul(poly, e2n);
}

static inline v128_t q4_wasm_fast_silu_f32(v128_t x) {
    v128_t one = wasm_f32x4_splat(1.0f);
    v128_t ex = q4_wasm_fast_exp_f32(wasm_f32x4_neg(x));
    return wasm_f32x4_div(x, wasm_f32x4_add(one, ex));
}

static inline v128_t q4_wasm_lane0_i8x16(v128_t a) {
    return wasm_i32x4_shuffle(a, a, 0, 0, 0, 0);
}

static inline v128_t q4_wasm_lane1_i8x16(v128_t a) {
    return wasm_i32x4_shuffle(a, a, 1, 1, 1, 1);
}

static inline v128_t q4_wasm_lane2_i8x16(v128_t a) {
    return wasm_i32x4_shuffle(a, a, 2, 2, 2, 2);
}

static inline v128_t q4_wasm_lane3_i8x16(v128_t a) {
    return wasm_i32x4_shuffle(a, a, 3, 3, 3, 3);
}

typedef struct {
    v128_t a00, a01, a02, a03;
    v128_t a10, a11, a12, a13;
} BnWasmQ4XLanes;

static inline void q4_wasm_prepare_x_lanes(BnWasmQ4XLanes *lanes,
                                           const int8_t *x_q,
                                           int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        v128_t a0 = wasm_v128_load(x_q + b * 32);
        v128_t a1 = wasm_v128_load(x_q + b * 32 + 16);
        lanes[b].a00 = q4_wasm_lane0_i8x16(a0);
        lanes[b].a01 = q4_wasm_lane1_i8x16(a0);
        lanes[b].a02 = q4_wasm_lane2_i8x16(a0);
        lanes[b].a03 = q4_wasm_lane3_i8x16(a0);
        lanes[b].a10 = q4_wasm_lane0_i8x16(a1);
        lanes[b].a11 = q4_wasm_lane1_i8x16(a1);
        lanes[b].a12 = q4_wasm_lane2_i8x16(a1);
        lanes[b].a13 = q4_wasm_lane3_i8x16(a1);
    }
}

static inline v128_t q4_repacked_wasm_dot4_xor_lanes(const uint8_t *qbase,
                                                     v128_t a00,
                                                     v128_t a01,
                                                     v128_t a02,
                                                     v128_t a03,
                                                     v128_t a10,
                                                     v128_t a11,
                                                     v128_t a12,
                                                     v128_t a13) {
    const v128_t mask_hi = wasm_i8x16_splat((char)0xF0);
    const v128_t zero = wasm_i32x4_splat(0);

    v128_t raw0 = wasm_v128_load(qbase);
    v128_t raw1 = wasm_v128_load(qbase + 16);
    v128_t raw2 = wasm_v128_load(qbase + 32);
    v128_t raw3 = wasm_v128_load(qbase + 48);
    v128_t lo0 = wasm_i8x16_shl(raw0, 4);
    v128_t lo1 = wasm_i8x16_shl(raw1, 4);
    v128_t lo2 = wasm_i8x16_shl(raw2, 4);
    v128_t lo3 = wasm_i8x16_shl(raw3, 4);
    v128_t hi0 = wasm_v128_and(raw0, mask_hi);
    v128_t hi1 = wasm_v128_and(raw1, mask_hi);
    v128_t hi2 = wasm_v128_and(raw2, mask_hi);
    v128_t hi3 = wasm_v128_and(raw3, mask_hi);

    v128_t acc02 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(lo0, a00, zero);
    v128_t acc13 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(lo1, a01, zero);
    acc02 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(lo2, a02, acc02);
    acc13 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(lo3, a03, acc13);
    v128_t acc46 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(hi0, a10, zero);
    v128_t acc57 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(hi1, a11, zero);
    acc46 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(hi2, a12, acc46);
    acc57 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(hi3, a13, acc57);
    return wasm_i32x4_add(wasm_i32x4_add(acc02, acc13), wasm_i32x4_add(acc46, acc57));
}

static inline v128_t q4_repacked_wasm_dot4_xor(const uint8_t *qbase,
                                               v128_t a0,
                                               v128_t a1) {
    return q4_repacked_wasm_dot4_xor_lanes(qbase,
                                           q4_wasm_lane0_i8x16(a0),
                                           q4_wasm_lane1_i8x16(a0),
                                           q4_wasm_lane2_i8x16(a0),
                                           q4_wasm_lane3_i8x16(a0),
                                           q4_wasm_lane0_i8x16(a1),
                                           q4_wasm_lane1_i8x16(a1),
                                           q4_wasm_lane2_i8x16(a1),
                                           q4_wasm_lane3_i8x16(a1));
}

static inline float q4_wasm_block_scale(const BnQWeight *W,
                                        const BnPreparedWeight *prepared,
                                        const BnBlockQ4_0 *blocks,
                                        int row,
                                        int block) {
    if (prepared && prepared->f32_scales) {
        int n_blocks_per_row = W->cols / 32;
        size_t gb = (size_t)(row >> 2) * n_blocks_per_row + block;
        return prepared->f32_scales[gb * 4 + (row & 3)];
    }
    return bn_fp16_to_fp32(blocks[(size_t)row * (W->cols / 32) + block].d);
}

#endif

void bn_quant_q4_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ4Ctx *c = (BnQ4Ctx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const float *x = c->x;

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            v128_t raw = wasm_v128_load(blk->qs);
            v128_t lo = wasm_i8x16_sub(wasm_v128_and(raw, wasm_i8x16_splat(0xF)), wasm_i8x16_splat(8));
            v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), wasm_i8x16_splat(8));
            v128_t acc0 = wasm_f32x4_splat(0), acc1 = wasm_f32x4_splat(0);
            v128_t acc2 = wasm_f32x4_splat(0), acc3 = wasm_f32x4_splat(0);
            {
                v128_t lo16 = wasm_i16x8_extend_low_i8x16(lo);
                v128_t hi16 = wasm_i16x8_extend_high_i8x16(lo);
#ifdef __wasm_relaxed_simd__
                acc0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)),  wasm_v128_load(xb),      acc0);
                acc1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xb + 4),  acc1);
                acc2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)),  wasm_v128_load(xb + 8),  acc2);
                acc3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xb + 12), acc3);
#else
                acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), wasm_v128_load(xb)));
                acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xb + 4)));
                acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), wasm_v128_load(xb + 8)));
                acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xb + 12)));
#endif
            }
            {
                v128_t lo16 = wasm_i16x8_extend_low_i8x16(hi);
                v128_t hi16 = wasm_i16x8_extend_high_i8x16(hi);
#ifdef __wasm_relaxed_simd__
                acc0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)),  wasm_v128_load(xb + 16), acc0);
                acc1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xb + 20), acc1);
                acc2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)),  wasm_v128_load(xb + 24), acc2);
                acc3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xb + 28), acc3);
#else
                acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16)), wasm_v128_load(xb + 16)));
                acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16)), wasm_v128_load(xb + 20)));
                acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16)), wasm_v128_load(xb + 24)));
                acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16)), wasm_v128_load(xb + 28)));
#endif
            }
            v128_t sum = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
            row_sum += bn_wasm_hsum_f32x4(sum) * d;
        }
        c->out[row] = row_sum;
    }
}

// Relaxed SIMD SDOT path — mirrors NEON SDOT: integer dot product on Q4×Q8 blocks
#ifdef __wasm_relaxed_simd__
void bn_quant_q4_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    const v128_t mask_lo = wasm_i8x16_splat(0xF);
    const v128_t eight = wasm_i8x16_splat(8);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[row * n_blocks_per_row + b];
            float d_q4 = q4_wasm_block_scale(c->W, c->prepared, blocks, row, b);
            float d_q8 = x_scales[b];

            v128_t raw = wasm_v128_load(blk->qs);
            v128_t lo = wasm_i8x16_sub(wasm_v128_and(raw, mask_lo), eight);
            v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), eight);

            const int8_t *xb = x_q + b * 32;
            v128_t acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(lo, wasm_v128_load(xb), wasm_i32x4_splat(0));
            acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(hi, wasm_v128_load(xb + 16), acc);

            // Horizontal sum
            v128_t shuf = wasm_i32x4_shuffle(acc, acc, 2, 3, 0, 1);
            acc = wasm_i32x4_add(acc, shuf);
            shuf = wasm_i32x4_shuffle(acc, acc, 1, 0, 3, 2);
            acc = wasm_i32x4_add(acc, shuf);

            row_sum += d_q4 * d_q8 * (float)wasm_i32x4_extract_lane(acc, 0);
        }
        c->out[row] = row_sum;
    }
}

static inline v128_t q4_wasm_canonical_dot_vec(const BnBlockQ4_0 *blk,
                                               const int8_t *xb) {
    const v128_t mask_lo = wasm_i8x16_splat(0xF);
    const v128_t eight = wasm_i8x16_splat(8);
    v128_t raw = wasm_v128_load(blk->qs);
    v128_t lo = wasm_i8x16_sub(wasm_v128_and(raw, mask_lo), eight);
    v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), eight);
    v128_t acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(lo, wasm_v128_load(xb), wasm_i32x4_splat(0));
    return wasm_i32x4_relaxed_dot_i8x16_i7x16_add(hi, wasm_v128_load(xb + 16), acc);
}

void bn_quant_q4_wasm_sdot_4row_range(void *ctx, int group_start, int group_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int group = group_start; group < group_end; group++) {
        int row = group * 4;
        int rows_left = c->W->rows - row;
        if (rows_left < 4) {
            bn_quant_q4_wasm_sdot_range(ctx, row, c->W->rows);
            continue;
        }

        v128_t sum0 = wasm_f32x4_splat(0.0f);
        v128_t sum1 = wasm_f32x4_splat(0.0f);
        v128_t sum2 = wasm_f32x4_splat(0.0f);
        v128_t sum3 = wasm_f32x4_splat(0.0f);
        const BnBlockQ4_0 *row0 = blocks + (size_t)(row + 0) * n_blocks_per_row;
        const BnBlockQ4_0 *row1 = blocks + (size_t)(row + 1) * n_blocks_per_row;
        const BnBlockQ4_0 *row2 = blocks + (size_t)(row + 2) * n_blocks_per_row;
        const BnBlockQ4_0 *row3 = blocks + (size_t)(row + 3) * n_blocks_per_row;

        for (int b = 0; b < n_blocks_per_row; b++) {
            const int8_t *xb = x_q + b * 32;
            float dx = x_scales[b];

            v128_t d0 = wasm_f32x4_splat(q4_wasm_block_scale(c->W, c->prepared, blocks, row + 0, b) * dx);
            v128_t d1 = wasm_f32x4_splat(q4_wasm_block_scale(c->W, c->prepared, blocks, row + 1, b) * dx);
            v128_t d2 = wasm_f32x4_splat(q4_wasm_block_scale(c->W, c->prepared, blocks, row + 2, b) * dx);
            v128_t d3 = wasm_f32x4_splat(q4_wasm_block_scale(c->W, c->prepared, blocks, row + 3, b) * dx);

            sum0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(row0 + b, xb)), d0, sum0);
            sum1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(row1 + b, xb)), d1, sum1);
            sum2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(row2 + b, xb)), d2, sum2);
            sum3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(row3 + b, xb)), d3, sum3);
        }

        c->out[row + 0] = bn_wasm_hsum_f32x4(sum0);
        c->out[row + 1] = bn_wasm_hsum_f32x4(sum1);
        c->out[row + 2] = bn_wasm_hsum_f32x4(sum2);
        c->out[row + 3] = bn_wasm_hsum_f32x4(sum3);
    }
}

static float q4_wasm_native_row_dot(const BnQWeight *W, int row,
                                    const int8_t *x_q,
                                    const float *x_scales);

void bn_quant_q4_wasm_gate_up_silu_4row_range(void *ctx, int group_start, int group_end) {
    BnQ4GateUpCtx *c = (BnQ4GateUpCtx *)ctx;
    const BnQWeight *gate = c->gate;
    const BnQWeight *up = c->up;
    const BnBlockQ4_0 *gate_blocks = (const BnBlockQ4_0 *)gate->data;
    const BnBlockQ4_0 *up_blocks = (const BnBlockQ4_0 *)up->data;
    int n_blocks_per_row = gate->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    for (int group = group_start; group < group_end; group++) {
        int row = group * 4;
        int rows_left = gate->rows - row;
        if (rows_left < 4) {
            for (int r = row; r < gate->rows; r++) {
                float g = q4_wasm_native_row_dot(gate, r, x_q, x_scales);
                float u = q4_wasm_native_row_dot(up, r, x_q, x_scales);
                c->out[r] = (g / (1.0f + expf(-g))) * u;
            }
            continue;
        }

        v128_t gsum0 = wasm_f32x4_splat(0.0f);
        v128_t gsum1 = wasm_f32x4_splat(0.0f);
        v128_t gsum2 = wasm_f32x4_splat(0.0f);
        v128_t gsum3 = wasm_f32x4_splat(0.0f);
        v128_t usum0 = wasm_f32x4_splat(0.0f);
        v128_t usum1 = wasm_f32x4_splat(0.0f);
        v128_t usum2 = wasm_f32x4_splat(0.0f);
        v128_t usum3 = wasm_f32x4_splat(0.0f);

        const BnBlockQ4_0 *g0 = gate_blocks + (size_t)(row + 0) * n_blocks_per_row;
        const BnBlockQ4_0 *g1 = gate_blocks + (size_t)(row + 1) * n_blocks_per_row;
        const BnBlockQ4_0 *g2 = gate_blocks + (size_t)(row + 2) * n_blocks_per_row;
        const BnBlockQ4_0 *g3 = gate_blocks + (size_t)(row + 3) * n_blocks_per_row;
        const BnBlockQ4_0 *u0 = up_blocks + (size_t)(row + 0) * n_blocks_per_row;
        const BnBlockQ4_0 *u1 = up_blocks + (size_t)(row + 1) * n_blocks_per_row;
        const BnBlockQ4_0 *u2 = up_blocks + (size_t)(row + 2) * n_blocks_per_row;
        const BnBlockQ4_0 *u3 = up_blocks + (size_t)(row + 3) * n_blocks_per_row;

        for (int b = 0; b < n_blocks_per_row; b++) {
            const int8_t *xb = x_q + b * 32;
            float dx = x_scales[b];

            v128_t gd0 = wasm_f32x4_splat(q4_wasm_block_scale(gate, c->gate_prepared, gate_blocks, row + 0, b) * dx);
            v128_t gd1 = wasm_f32x4_splat(q4_wasm_block_scale(gate, c->gate_prepared, gate_blocks, row + 1, b) * dx);
            v128_t gd2 = wasm_f32x4_splat(q4_wasm_block_scale(gate, c->gate_prepared, gate_blocks, row + 2, b) * dx);
            v128_t gd3 = wasm_f32x4_splat(q4_wasm_block_scale(gate, c->gate_prepared, gate_blocks, row + 3, b) * dx);
            v128_t ud0 = wasm_f32x4_splat(q4_wasm_block_scale(up, c->up_prepared, up_blocks, row + 0, b) * dx);
            v128_t ud1 = wasm_f32x4_splat(q4_wasm_block_scale(up, c->up_prepared, up_blocks, row + 1, b) * dx);
            v128_t ud2 = wasm_f32x4_splat(q4_wasm_block_scale(up, c->up_prepared, up_blocks, row + 2, b) * dx);
            v128_t ud3 = wasm_f32x4_splat(q4_wasm_block_scale(up, c->up_prepared, up_blocks, row + 3, b) * dx);

            gsum0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(g0 + b, xb)), gd0, gsum0);
            gsum1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(g1 + b, xb)), gd1, gsum1);
            gsum2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(g2 + b, xb)), gd2, gsum2);
            gsum3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(g3 + b, xb)), gd3, gsum3);
            usum0 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(u0 + b, xb)), ud0, usum0);
            usum1 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(u1 + b, xb)), ud1, usum1);
            usum2 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(u2 + b, xb)), ud2, usum2);
            usum3 = wasm_f32x4_relaxed_madd(wasm_f32x4_convert_i32x4(q4_wasm_canonical_dot_vec(u3 + b, xb)), ud3, usum3);
        }

        v128_t gate4 = wasm_f32x4_make(bn_wasm_hsum_f32x4(gsum0),
                                       bn_wasm_hsum_f32x4(gsum1),
                                       bn_wasm_hsum_f32x4(gsum2),
                                       bn_wasm_hsum_f32x4(gsum3));
        v128_t up4 = wasm_f32x4_make(bn_wasm_hsum_f32x4(usum0),
                                     bn_wasm_hsum_f32x4(usum1),
                                     bn_wasm_hsum_f32x4(usum2),
                                     bn_wasm_hsum_f32x4(usum3));
        wasm_v128_store(c->out + row,
                        wasm_f32x4_mul(q4_wasm_fast_silu_f32(gate4), up4));
    }
}

void bn_quant_q4_repacked_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const uint8_t *rp_qs = c->prepared ? c->prepared->qs : NULL;
    const float *rp_f32_scales = c->prepared ? c->prepared->f32_scales : NULL;
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)c->W->data;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;

    int row = row_start;
    for (; row < row_end && (row & 3); row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            float d_q4 = q4_wasm_block_scale(c->W, c->prepared, blocks, row, b);
            float d_q8 = x_scales[b];

            v128_t raw = wasm_v128_load(blk->qs);
            v128_t lo = wasm_i8x16_sub(wasm_v128_and(raw, wasm_i8x16_splat(0xF)), wasm_i8x16_splat(8));
            v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), wasm_i8x16_splat(8));
            const int8_t *xb = x_q + b * 32;
            v128_t acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(lo, wasm_v128_load(xb), wasm_i32x4_splat(0));
            acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(hi, wasm_v128_load(xb + 16), acc);
            int32_t total = wasm_i32x4_extract_lane(acc, 0) + wasm_i32x4_extract_lane(acc, 1) +
                            wasm_i32x4_extract_lane(acc, 2) + wasm_i32x4_extract_lane(acc, 3);
            row_sum += d_q4 * d_q8 * (float)total;
        }
        c->out[row] = row_sum;
    }

    for (; row + 3 < row_end; row += 4) {
        int group = row >> 2;
        v128_t row_sums = wasm_f32x4_splat(0.0f);

        for (int b = 0; b < n_blocks_per_row; b++) {
            v128_t a0 = wasm_v128_load(x_q + b * 32);
            v128_t a1 = wasm_v128_load(x_q + b * 32 + 16);
            v128_t dx = wasm_f32x4_splat(x_scales[b] * (1.0f / 16.0f));

            size_t gb = (size_t)group * n_blocks_per_row + b;
            v128_t acc = q4_repacked_wasm_dot4_xor(rp_qs + gb * 64, a0, a1);
            v128_t f = wasm_f32x4_convert_i32x4(acc);
            v128_t d = wasm_v128_load(rp_f32_scales + gb * 4);
            row_sums = wasm_f32x4_relaxed_madd(f, wasm_f32x4_mul(d, dx), row_sums);
        }

        wasm_v128_store(&c->out[row], row_sums);
    }

    for (; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            float d_q4 = q4_wasm_block_scale(c->W, c->prepared, blocks, row, b);
            float d_q8 = x_scales[b];

            v128_t raw = wasm_v128_load(blk->qs);
            v128_t lo = wasm_i8x16_sub(wasm_v128_and(raw, wasm_i8x16_splat(0xF)), wasm_i8x16_splat(8));
            v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), wasm_i8x16_splat(8));
            const int8_t *xb = x_q + b * 32;
            v128_t acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(lo, wasm_v128_load(xb), wasm_i32x4_splat(0));
            acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(hi, wasm_v128_load(xb + 16), acc);
            int32_t total = wasm_i32x4_extract_lane(acc, 0) + wasm_i32x4_extract_lane(acc, 1) +
                            wasm_i32x4_extract_lane(acc, 2) + wasm_i32x4_extract_lane(acc, 3);
            row_sum += d_q4 * d_q8 * (float)total;
        }
        c->out[row] = row_sum;
    }
}

void bn_quant_q4_repacked_wasm_sdot_8row_range(void *ctx, int group_start, int group_end) {
    BnQ4SdotCtx *c = (BnQ4SdotCtx *)ctx;
    const uint8_t *rp_qs = c->prepared ? c->prepared->qs : NULL;
    const float *rp_f32_scales = c->prepared ? c->prepared->f32_scales : NULL;
    int n_blocks_per_row = c->W->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;
    BnWasmQ4XLanes x_lanes[BN_WASM_Q4_LANE_CACHE_MAX_BLOCKS];
    int use_lane_cache = n_blocks_per_row <= BN_WASM_Q4_LANE_CACHE_MAX_BLOCKS;
    if (use_lane_cache) {
        q4_wasm_prepare_x_lanes(x_lanes, x_q, n_blocks_per_row);
    }

    for (int group8 = group_start; group8 < group_end; group8++) {
        int row = group8 * 8;
        int rows_left = c->W->rows - row;
        if (rows_left < 8) {
            bn_quant_q4_repacked_wasm_sdot_range(ctx, row, c->W->rows);
            continue;
        }

        int group0 = row >> 2;
        v128_t row_sums0 = wasm_f32x4_splat(0.0f);
        v128_t row_sums1 = wasm_f32x4_splat(0.0f);

        for (int b = 0; b < n_blocks_per_row; b++) {
            v128_t a00, a01, a02, a03, a10, a11, a12, a13;
            if (use_lane_cache) {
                const BnWasmQ4XLanes *xl = &x_lanes[b];
                a00 = xl->a00; a01 = xl->a01; a02 = xl->a02; a03 = xl->a03;
                a10 = xl->a10; a11 = xl->a11; a12 = xl->a12; a13 = xl->a13;
            } else {
                v128_t a0 = wasm_v128_load(x_q + b * 32);
                v128_t a1 = wasm_v128_load(x_q + b * 32 + 16);
                a00 = q4_wasm_lane0_i8x16(a0);
                a01 = q4_wasm_lane1_i8x16(a0);
                a02 = q4_wasm_lane2_i8x16(a0);
                a03 = q4_wasm_lane3_i8x16(a0);
                a10 = q4_wasm_lane0_i8x16(a1);
                a11 = q4_wasm_lane1_i8x16(a1);
                a12 = q4_wasm_lane2_i8x16(a1);
                a13 = q4_wasm_lane3_i8x16(a1);
            }
            v128_t dx = wasm_f32x4_splat(x_scales[b] * (1.0f / 16.0f));

            size_t gb0 = (size_t)group0 * n_blocks_per_row + b;
            v128_t acc0 = q4_repacked_wasm_dot4_xor_lanes(rp_qs + gb0 * 64,
                                                          a00, a01, a02, a03,
                                                          a10, a11, a12, a13);
            v128_t f0 = wasm_f32x4_convert_i32x4(acc0);
            v128_t d0 = wasm_v128_load(rp_f32_scales + gb0 * 4);
            row_sums0 = wasm_f32x4_relaxed_madd(f0, wasm_f32x4_mul(d0, dx), row_sums0);

            size_t gb1 = (size_t)(group0 + 1) * n_blocks_per_row + b;
            v128_t acc1 = q4_repacked_wasm_dot4_xor_lanes(rp_qs + gb1 * 64,
                                                          a00, a01, a02, a03,
                                                          a10, a11, a12, a13);
            v128_t f1 = wasm_f32x4_convert_i32x4(acc1);
            v128_t d1 = wasm_v128_load(rp_f32_scales + gb1 * 4);
            row_sums1 = wasm_f32x4_relaxed_madd(f1, wasm_f32x4_mul(d1, dx), row_sums1);
        }

        wasm_v128_store(c->out + row, row_sums0);
        wasm_v128_store(c->out + row + 4, row_sums1);
    }
}

static float q4_wasm_native_row_dot(const BnQWeight *W, int row,
                                    const int8_t *x_q,
                                    const float *x_scales) {
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)W->data;
    int n_blocks_per_row = W->cols / 32;
    const v128_t mask_lo = wasm_i8x16_splat(0xF);
    const v128_t eight = wasm_i8x16_splat(8);
    float row_sum = 0.0f;

    for (int b = 0; b < n_blocks_per_row; b++) {
        const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_blocks_per_row + b];
        v128_t raw = wasm_v128_load(blk->qs);
        v128_t lo = wasm_i8x16_sub(wasm_v128_and(raw, mask_lo), eight);
        v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), eight);
        const int8_t *xb = x_q + b * 32;
        v128_t acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(lo, wasm_v128_load(xb), wasm_i32x4_splat(0));
        acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(hi, wasm_v128_load(xb + 16), acc);
        int32_t total = wasm_i32x4_extract_lane(acc, 0) + wasm_i32x4_extract_lane(acc, 1) +
                        wasm_i32x4_extract_lane(acc, 2) + wasm_i32x4_extract_lane(acc, 3);
        row_sum += q4_wasm_block_scale(W, NULL, blocks, row, b) * x_scales[b] * (float)total;
    }
    return row_sum;
}

void bn_quant_q4_repacked_gate_up_silu_wasm_range(void *ctx, int row_start, int row_end) {
    BnQ4GateUpCtx *c = (BnQ4GateUpCtx *)ctx;
    const BnQWeight *gate = c->gate;
    const BnQWeight *up = c->up;
    const float *gate_f32_scales = c->gate_prepared ? c->gate_prepared->f32_scales : NULL;
    const float *up_f32_scales = c->up_prepared ? c->up_prepared->f32_scales : NULL;
    const uint8_t *gate_qs = c->gate_prepared ? c->gate_prepared->qs : NULL;
    const uint8_t *up_qs = c->up_prepared ? c->up_prepared->qs : NULL;
    int n_blocks_per_row = gate->cols / 32;
    const int8_t *x_q = c->x_q;
    const float *x_scales = c->x_scales;
    BnWasmQ4XLanes x_lanes[BN_WASM_Q4_LANE_CACHE_MAX_BLOCKS];
    int use_lane_cache = n_blocks_per_row <= BN_WASM_Q4_LANE_CACHE_MAX_BLOCKS;
    if (use_lane_cache) {
        q4_wasm_prepare_x_lanes(x_lanes, x_q, n_blocks_per_row);
    }

    int row = row_start;
    for (; row < row_end && (row & 3); row++) {
        float g = q4_wasm_native_row_dot(gate, row, x_q, x_scales);
        float u = q4_wasm_native_row_dot(up, row, x_q, x_scales);
        c->out[row] = (g / (1.0f + expf(-g))) * u;
    }

    for (; row + 7 < row_end; row += 8) {
        int group = row >> 2;
        v128_t gate_sum0 = wasm_f32x4_splat(0.0f);
        v128_t up_sum0 = wasm_f32x4_splat(0.0f);
        v128_t gate_sum1 = wasm_f32x4_splat(0.0f);
        v128_t up_sum1 = wasm_f32x4_splat(0.0f);

        for (int b = 0; b < n_blocks_per_row; b++) {
            v128_t a00, a01, a02, a03, a10, a11, a12, a13;
            if (use_lane_cache) {
                const BnWasmQ4XLanes *xl = &x_lanes[b];
                a00 = xl->a00; a01 = xl->a01; a02 = xl->a02; a03 = xl->a03;
                a10 = xl->a10; a11 = xl->a11; a12 = xl->a12; a13 = xl->a13;
            } else {
                v128_t a0 = wasm_v128_load(x_q + b * 32);
                v128_t a1 = wasm_v128_load(x_q + b * 32 + 16);
                a00 = q4_wasm_lane0_i8x16(a0);
                a01 = q4_wasm_lane1_i8x16(a0);
                a02 = q4_wasm_lane2_i8x16(a0);
                a03 = q4_wasm_lane3_i8x16(a0);
                a10 = q4_wasm_lane0_i8x16(a1);
                a11 = q4_wasm_lane1_i8x16(a1);
                a12 = q4_wasm_lane2_i8x16(a1);
                a13 = q4_wasm_lane3_i8x16(a1);
            }
            v128_t dx = wasm_f32x4_splat(x_scales[b] * (1.0f / 16.0f));

            size_t gb0 = (size_t)group * n_blocks_per_row + b;
            v128_t g_acc0 = q4_repacked_wasm_dot4_xor_lanes(gate_qs + gb0 * 64,
                                                            a00, a01, a02, a03,
                                                            a10, a11, a12, a13);
            v128_t g_f0 = wasm_f32x4_convert_i32x4(g_acc0);
            v128_t g_d0 = wasm_v128_load(gate_f32_scales + gb0 * 4);
            gate_sum0 = wasm_f32x4_relaxed_madd(g_f0, wasm_f32x4_mul(g_d0, dx), gate_sum0);

            v128_t u_acc0 = q4_repacked_wasm_dot4_xor_lanes(up_qs + gb0 * 64,
                                                            a00, a01, a02, a03,
                                                            a10, a11, a12, a13);
            v128_t u_f0 = wasm_f32x4_convert_i32x4(u_acc0);
            v128_t u_d0 = wasm_v128_load(up_f32_scales + gb0 * 4);
            up_sum0 = wasm_f32x4_relaxed_madd(u_f0, wasm_f32x4_mul(u_d0, dx), up_sum0);

            size_t gb1 = (size_t)(group + 1) * n_blocks_per_row + b;
            v128_t g_acc1 = q4_repacked_wasm_dot4_xor_lanes(gate_qs + gb1 * 64,
                                                            a00, a01, a02, a03,
                                                            a10, a11, a12, a13);
            v128_t g_f1 = wasm_f32x4_convert_i32x4(g_acc1);
            v128_t g_d1 = wasm_v128_load(gate_f32_scales + gb1 * 4);
            gate_sum1 = wasm_f32x4_relaxed_madd(g_f1, wasm_f32x4_mul(g_d1, dx), gate_sum1);

            v128_t u_acc1 = q4_repacked_wasm_dot4_xor_lanes(up_qs + gb1 * 64,
                                                            a00, a01, a02, a03,
                                                            a10, a11, a12, a13);
            v128_t u_f1 = wasm_f32x4_convert_i32x4(u_acc1);
            v128_t u_d1 = wasm_v128_load(up_f32_scales + gb1 * 4);
            up_sum1 = wasm_f32x4_relaxed_madd(u_f1, wasm_f32x4_mul(u_d1, dx), up_sum1);
        }

        wasm_v128_store(c->out + row,
                        wasm_f32x4_mul(q4_wasm_fast_silu_f32(gate_sum0), up_sum0));
        wasm_v128_store(c->out + row + 4,
                        wasm_f32x4_mul(q4_wasm_fast_silu_f32(gate_sum1), up_sum1));
    }

    for (; row + 3 < row_end; row += 4) {
        int group = row >> 2;
        v128_t gate_sum = wasm_f32x4_splat(0.0f);
        v128_t up_sum = wasm_f32x4_splat(0.0f);

        for (int b = 0; b < n_blocks_per_row; b++) {
            v128_t a00, a01, a02, a03, a10, a11, a12, a13;
            if (use_lane_cache) {
                const BnWasmQ4XLanes *xl = &x_lanes[b];
                a00 = xl->a00; a01 = xl->a01; a02 = xl->a02; a03 = xl->a03;
                a10 = xl->a10; a11 = xl->a11; a12 = xl->a12; a13 = xl->a13;
            } else {
                v128_t a0 = wasm_v128_load(x_q + b * 32);
                v128_t a1 = wasm_v128_load(x_q + b * 32 + 16);
                a00 = q4_wasm_lane0_i8x16(a0);
                a01 = q4_wasm_lane1_i8x16(a0);
                a02 = q4_wasm_lane2_i8x16(a0);
                a03 = q4_wasm_lane3_i8x16(a0);
                a10 = q4_wasm_lane0_i8x16(a1);
                a11 = q4_wasm_lane1_i8x16(a1);
                a12 = q4_wasm_lane2_i8x16(a1);
                a13 = q4_wasm_lane3_i8x16(a1);
            }
            v128_t dx = wasm_f32x4_splat(x_scales[b] * (1.0f / 16.0f));
            size_t gb = (size_t)group * n_blocks_per_row + b;

            v128_t g_acc = q4_repacked_wasm_dot4_xor_lanes(gate_qs + gb * 64,
                                                           a00, a01, a02, a03,
                                                           a10, a11, a12, a13);
            v128_t g_f = wasm_f32x4_convert_i32x4(g_acc);
            v128_t g_d = wasm_v128_load(gate_f32_scales + gb * 4);
            gate_sum = wasm_f32x4_relaxed_madd(g_f, wasm_f32x4_mul(g_d, dx), gate_sum);

            v128_t u_acc = q4_repacked_wasm_dot4_xor_lanes(up_qs + gb * 64,
                                                           a00, a01, a02, a03,
                                                           a10, a11, a12, a13);
            v128_t u_f = wasm_f32x4_convert_i32x4(u_acc);
            v128_t u_d = wasm_v128_load(up_f32_scales + gb * 4);
            up_sum = wasm_f32x4_relaxed_madd(u_f, wasm_f32x4_mul(u_d, dx), up_sum);
        }

        wasm_v128_store(c->out + row,
                        wasm_f32x4_mul(q4_wasm_fast_silu_f32(gate_sum), up_sum));
    }

    for (; row < row_end; row++) {
        float g = q4_wasm_native_row_dot(gate, row, x_q, x_scales);
        float u = q4_wasm_native_row_dot(up, row, x_q, x_scales);
        c->out[row] = (g / (1.0f + expf(-g))) * u;
    }
}
#else
void bn_quant_q4_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    (void)ctx; (void)row_start; (void)row_end;
}

void bn_quant_q4_wasm_sdot_4row_range(void *ctx, int group_start, int group_end) {
    (void)ctx; (void)group_start; (void)group_end;
}

void bn_quant_q4_repacked_wasm_sdot_range(void *ctx, int row_start, int row_end) {
    (void)ctx; (void)row_start; (void)row_end;
}

void bn_quant_q4_repacked_wasm_sdot_8row_range(void *ctx, int group_start, int group_end) {
    (void)ctx; (void)group_start; (void)group_end;
}

void bn_quant_q4_wasm_gate_up_silu_4row_range(void *ctx, int group_start, int group_end) {
    (void)ctx; (void)group_start; (void)group_end;
}

void bn_quant_q4_repacked_gate_up_silu_wasm_range(void *ctx, int row_start, int row_end) {
    (void)ctx; (void)row_start; (void)row_end;
}
#endif
