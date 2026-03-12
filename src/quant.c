#include "quant.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

// --- FP16 <-> FP32 conversion ---

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // ±0
        } else {
            // #37: Subnormal: normalize by shifting mantissa left until hidden bit appears.
            // At most 10 shifts (10 mantissa bits), so exp goes from 1 down to at most -9.
            // Result: exp+112 ranges from 103 to 113, always valid for float32 exponent.
            exp = 1;
            while (!(mant & 0x0400)) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            f = sign | ((uint32_t)(exp + 112) << 23) | ((uint32_t)mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | ((uint32_t)mant << 13);  // Inf/NaN
    } else {
        f = sign | ((uint32_t)(exp + 112) << 23) | ((uint32_t)mant << 13);
    }

    float result;
    memcpy(&result, &f, 4);
    return result;
}

uint16_t fp32_to_fp16(float val) {
    uint32_t f;
    memcpy(&f, &val, 4);

    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = ((f >> 23) & 0xFF) - 127;
    uint32_t mant = f & 0x007FFFFF;

    if (exp > 15) {
        return (uint16_t)(sign | 0x7C00);  // Inf
    } else if (exp < -14) {
        return (uint16_t)sign;  // Zero (flush subnormals)
    } else {
        return (uint16_t)(sign | ((uint32_t)(exp + 15) << 10) | (mant >> 13));
    }
}

// --- TQ2_0 dequantization ---
// 2-bit packing: 4 values per byte, map {0,1,2} -> {-1,0,+1}

void dequant_tq2_block(const BlockTQ2 *block, float *out) {
    float d = fp16_to_fp32(block->d);
    int idx = 0;

    // Two groups of 32 bytes
    for (int j = 0; j < 64; j += 32) {
        for (int l = 0; l < 4; l++) {
            for (int m = 0; m < 32; m++) {
                int8_t q = (block->qs[j + m] >> (l * 2)) & 3;
                out[idx++] = (float)(q - 1) * d;
            }
        }
    }
}

// --- TQ1_0 dequantization ---
// Base-3 packing: 5 values per byte in qs (240 values), 4 values per byte in qh (16 values)

void dequant_tq1_block(const BlockTQ1 *block, float *out) {
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    float d = fp16_to_fp32(block->d);
    int idx = 0;

    // Process qs: 48 bytes, in two chunks (32 + 16)
    // First chunk: bytes 0..31, 5 trits each → 160 values
    for (int n = 0; n < 5; n++) {
        for (int m = 0; m < 32; m++) {
            uint8_t q = block->qs[m] * pow3[n];  // uint8 overflow is intentional
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    // Second chunk: bytes 32..47, 5 trits each → 80 values
    for (int n = 0; n < 5; n++) {
        for (int m = 0; m < 16; m++) {
            uint8_t q = block->qs[32 + m] * pow3[n];
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    // Process qh: 4 bytes, 4 trits each → 16 values
    for (int n = 0; n < 4; n++) {
        for (int m = 0; m < 4; m++) {
            uint8_t q = block->qh[m] * pow3[n];
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    // #35: Assert we produced exactly QK_K values (160 + 80 + 16 = 256)
    assert(idx == QK_K);
}

// --- I2_S dequantization (Microsoft BitNet format) ---
// 2-bit ternary, interleaved byte layout, single per-tensor scale
// Each byte: bits 7-6=subrow0, 5-4=subrow1, 3-2=subrow2, 1-0=subrow3
// Processes 128 elements (4 × 32) per 32-byte chunk

// #36: I2_S uses an interleaved byte layout where each byte contains 2-bit values
// from 4 sub-rows of 32 elements. This means each 128-element chunk always uses
// exactly 32 bytes. Model dimensions are always multiples of 128 in practice.
void dequant_i2s_row(const uint8_t *data, float *out, int n, float scale) {
    static const float map2bit[4] = { -1.0f, 0.0f, +1.0f, 0.0f };
    int done = 0;

    while (done < n) {
        int blk_e = (n - done >= 128) ? 128 : (n - done);
        int cols0 = blk_e >= 32  ? 32 : blk_e;
        int cols1 = blk_e >= 64  ? 32 : (blk_e > 32  ? blk_e - 32  : 0);
        int cols2 = blk_e >= 96  ? 32 : (blk_e > 64  ? blk_e - 64  : 0);
        int cols3 = blk_e >= 128 ? 32 : (blk_e > 96  ? blk_e - 96  : 0);

        for (int gp = 0; gp < 32; gp++) {
            uint8_t b = data[gp];
            uint8_t c0 = (b >> 6) & 0x3;
            uint8_t c1 = (b >> 4) & 0x3;
            uint8_t c2 = (b >> 2) & 0x3;
            uint8_t c3 = (b >> 0) & 0x3;

            if (gp < cols0) out[done + 0*32 + gp] = scale * map2bit[c0];
            if (gp < cols1) out[done + 1*32 + gp] = scale * map2bit[c1];
            if (gp < cols2) out[done + 2*32 + gp] = scale * map2bit[c2];
            if (gp < cols3) out[done + 3*32 + gp] = scale * map2bit[c3];
        }

        data += 32;
        done += blk_e;
    }
}

// --- Ternary matrix-vector multiply ---
// out[rows] = W[rows × cols] @ x[cols]

void ternary_matvec(float *out, const QWeight *W, const float *x) {

    if (W->type == 36) {  // I2_S
        // I2_S: flat packed data, one row = cols/4 bytes
        int row_bytes = W->cols / 4;
        const uint8_t *base = (const uint8_t *)W->data;
        // #5: Use malloc instead of __builtin_alloca to avoid stack overflow
        float *row_buf = (float *)malloc((size_t)W->cols * sizeof(float));
        if (!row_buf) return;  // OOM: leave output zeroed

        for (int row = 0; row < W->rows; row++) {
            dequant_i2s_row(base + (size_t)row * row_bytes, row_buf, W->cols, W->scale);
            float sum = 0.0f;
            for (int k = 0; k < W->cols; k++) {
                sum += row_buf[k] * x[k];
            }
            out[row] = sum;
        }
        free(row_buf);
        return;
    }

    // TQ1_0 / TQ2_0: block-based format
    int n_blocks_per_row = W->cols / QK_K;
    float block_out[QK_K];

    for (int row = 0; row < W->rows; row++) {
        float sum = 0.0f;

        for (int b = 0; b < n_blocks_per_row; b++) {
            int block_idx = row * n_blocks_per_row + b;
            int col_offset = b * QK_K;

            if (W->type == 35) {  // TQ2_0
                const BlockTQ2 *blocks = (const BlockTQ2 *)W->data;
                dequant_tq2_block(&blocks[block_idx], block_out);
            } else {  // TQ1_0
                const BlockTQ1 *blocks = (const BlockTQ1 *)W->data;
                dequant_tq1_block(&blocks[block_idx], block_out);
            }

            for (int k = 0; k < QK_K; k++) {
                sum += block_out[k] * x[col_offset + k];
            }
        }

        out[row] = sum * W->scale;
    }
}
