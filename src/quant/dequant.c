#include "quant_ctx.h"
#include "kquant_helpers.h"
#include "iq_tables.h"
#include <assert.h>
#include <string.h>

// --- TQ2_0 dequantization ---

void bn_quant_dequant_tq2(const BnBlockTQ2 *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    int idx = 0;

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

void bn_quant_dequant_tq1(const BnBlockTQ1 *block, float *out) {
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    float d = bn_fp16_to_fp32(block->d);
    int idx = 0;

    for (int n = 0; n < 5; n++) {
        for (int m = 0; m < 32; m++) {
            uint8_t q = block->qs[m] * pow3[n];
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    for (int n = 0; n < 5; n++) {
        for (int m = 0; m < 16; m++) {
            uint8_t q = block->qs[32 + m] * pow3[n];
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    for (int n = 0; n < 4; n++) {
        for (int m = 0; m < 4; m++) {
            uint8_t q = block->qh[m] * pow3[n];
            int16_t xi = ((uint16_t)q * 3) >> 8;
            out[idx++] = (float)(xi - 1) * d;
        }
    }

    assert(idx == BN_QK_K);
}

// --- I2_S dequantization ---

void bn_quant_dequant_i2s(const uint8_t *data, float *out, int n, float scale) {
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

// --- Q8_0 dequantization ---

void bn_quant_dequant_q8_0(const BnBlockQ8_0 *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    for (int i = 0; i < 32; i++) {
        out[i] = block->qs[i] * d;
    }
}

// --- Q4_0 dequantization ---

void bn_quant_dequant_q4_0(const BnBlockQ4_0 *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    for (int i = 0; i < 16; i++) {
        uint8_t b = block->qs[i];
        out[i]      = ((int)(b & 0xF) - 8) * d;
        out[i + 16] = ((int)(b >> 4)  - 8) * d;
    }
}

// --- Q4_1 dequantization ---

void bn_quant_dequant_q4_1(const BnBlockQ4_1 *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    float m = bn_fp16_to_fp32(block->m);
    for (int i = 0; i < 16; i++) {
        uint8_t b = block->qs[i];
        out[i]      = d * (b & 0xF) + m;
        out[i + 16] = d * (b >> 4)  + m;
    }
}

void bn_quant_dequant_q5_1(const BnBlockQ5_1 *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    float m = bn_fp16_to_fp32(block->m);
    uint32_t qh = (uint32_t)block->qh[0] |
                  ((uint32_t)block->qh[1] << 8) |
                  ((uint32_t)block->qh[2] << 16) |
                  ((uint32_t)block->qh[3] << 24);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = block->qs[i];
        out[i] = ((byte & 0xF) | (((qh >> i) & 1u) << 4)) * d + m;
        out[i + 16] = ((byte >> 4) | (((qh >> (i + 16)) & 1u) << 4)) * d + m;
    }
}

// --- Q6_K dequantization ---

void bn_quant_dequant_q6k(const BnBlockQ6K *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t  *sc = block->scales;

    for (int n = 0; n < BN_QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l +  0] = d * sc[is + 0] * q1;
            out[l + 32] = d * sc[is + 2] * q2;
            out[l + 64] = d * sc[is + 4] * q3;
            out[l + 96] = d * sc[is + 6] * q4;
        }
        out += 128;
        ql  += 64;
        qh  += 32;
        sc  += 8;
    }
}

// --- Q8_K dequantization ---

void bn_quant_dequant_q8k(const BnBlockQ8K *block, float *out) {
    float d = block->d;
    for (int i = 0; i < BN_QK_K; i++) {
        out[i] = d * block->qs[i];
    }
}

// --- Q4_K dequantization ---

void bn_quant_dequant_q4k(const BnBlockQ4K *block, float *out) {
    float d    = bn_fp16_to_fp32(block->d);
    float dmin = bn_fp16_to_fp32(block->dmin);
    const uint8_t *qs = block->qs;

    for (int j = 0; j < BN_QK_K; j += 64) {
        uint8_t sc, m;
        int sub = j / 32;
        bn_q4k_get_scale_min(sub, block->scales, &sc, &m);
        float ds  = d * sc;
        float dm  = dmin * m;
        for (int l = 0; l < 32; l++) {
            out[j + l] = ds * (qs[l] & 0xF) - dm;
        }
        bn_q4k_get_scale_min(sub + 1, block->scales, &sc, &m);
        ds = d * sc;
        dm = dmin * m;
        for (int l = 0; l < 32; l++) {
            out[j + l + 32] = ds * (qs[l] >> 4) - dm;
        }
        qs += 32;
    }
}

// --- Q5_K dequantization ---

void bn_quant_dequant_q5k(const BnBlockQ5K *block, float *out) {
    float d    = bn_fp16_to_fp32(block->d);
    float dmin = bn_fp16_to_fp32(block->dmin);
    const uint8_t *qs = block->qs;
    const uint8_t *qh = block->qh;

    for (int j = 0; j < BN_QK_K; j += 64) {
        uint8_t sc, m;
        int sub = j / 32;
        int group = j / 64;  // 0..3
        int bit_lo = group * 2;      // bits 0,2,4,6
        int bit_hi = group * 2 + 1;  // bits 1,3,5,7
        bn_q4k_get_scale_min(sub, block->scales, &sc, &m);
        float ds = d * sc;
        float dm = dmin * m;
        for (int l = 0; l < 32; l++) {
            int q5 = (qs[l] & 0xF) | (((qh[l] >> bit_lo) & 1) << 4);
            out[j + l] = ds * q5 - dm;
        }
        bn_q4k_get_scale_min(sub + 1, block->scales, &sc, &m);
        ds = d * sc;
        dm = dmin * m;
        for (int l = 0; l < 32; l++) {
            int q5 = (qs[l] >> 4) | (((qh[l] >> bit_hi) & 1) << 4);
            out[j + l + 32] = ds * q5 - dm;
        }
        qs += 32;
    }
}

// --- Q2_K dequantization ---

void bn_quant_dequant_q2k(const BnBlockQ2K *block, float *out) {
    float d    = bn_fp16_to_fp32(block->d);
    float dmin = bn_fp16_to_fp32(block->dmin);
    const uint8_t *q = block->qs;

    int is = 0, out_idx = 0;
    for (int n = 0; n < BN_QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc = block->scales[is++];
            float dl = d * (sc & 0xF);
            float ml = dmin * (sc >> 4);
            for (int l = 0; l < 16; l++)
                out[out_idx++] = dl * ((q[l] >> shift) & 3) - ml;
            sc = block->scales[is++];
            dl = d * (sc & 0xF);
            ml = dmin * (sc >> 4);
            for (int l = 0; l < 16; l++)
                out[out_idx++] = dl * ((q[l + 16] >> shift) & 3) - ml;
            shift += 2;
        }
        q += 32;
    }
}

// --- Q3_K dequantization ---

void bn_quant_dequant_q3k(const BnBlockQ3K *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);

    uint8_t scales[16];
    bn_q3k_unpack_scales(block->scales, scales);

    const uint8_t *q  = block->qs;
    const uint8_t *hm = block->hmask;

    int is = 0;
    uint8_t m = 1;
    int out_idx = 0;

    for (int n = 0; n < BN_QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            float dl = d * ((int)scales[is++] - 32);
            for (int l = 0; l < 16; l++) {
                int q3 = ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                out[out_idx++] = dl * q3;
            }
            dl = d * ((int)scales[is++] - 32);
            for (int l = 0; l < 16; l++) {
                int q3 = ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                out[out_idx++] = dl * q3;
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

// --- IQ4_NL dequantization ---

void bn_quant_dequant_iq4nl(const BnBlockIQ4NL *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    for (int i = 0; i < 16; i++) {
        uint8_t b = block->qs[i];
        out[i]      = d * bn_kvalues_iq4nl[b & 0xF];
        out[i + 16] = d * bn_kvalues_iq4nl[b >> 4];
    }
}

// --- IQ4_XS dequantization ---

void bn_quant_dequant_iq4xs(const BnBlockIQ4XS *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    const uint8_t *qs = block->qs;

    for (int j = 0; j < 8; j++) {
        int lo = (block->scales_l[j / 2] >> ((j % 2) * 4)) & 0xF;
        int hi = (block->scales_h >> (j * 2)) & 3;
        float dl = d * ((lo | (hi << 4)) - 32);
        for (int i = 0; i < 16; i++) {
            uint8_t b = qs[i];
            out[j * 32 + i]      = dl * bn_kvalues_iq4nl[b & 0xF];
            out[j * 32 + i + 16] = dl * bn_kvalues_iq4nl[b >> 4];
        }
        qs += 16;
    }
}

// --- IQ3_XXS dequantization ---
// Layout: qs[0..63] = 8 grid indices per sub-block (64 bytes total),
// qs[64..95] = 4-byte scale/sign words per sub-block (32 bytes total).

void bn_quant_dequant_iq3xxs(const BnBlockIQ3XXS *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    const uint8_t *qs = block->qs;
    const uint8_t *scales_and_signs = qs + BN_QK_K / 4;

    for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
        uint32_t aux32;
        memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
        float db = d * (0.5f + (aux32 >> 28)) * 0.5f;

        for (int l = 0; l < 4; l++) {
            const uint8_t signs = bn_ksigns_iq2xs[(aux32 >> (7 * l)) & 0x7F];
            const uint8_t *grid1 = (const uint8_t *)&bn_iq3xxs_grid[qs[2 * l + 0]];
            const uint8_t *grid2 = (const uint8_t *)&bn_iq3xxs_grid[qs[2 * l + 1]];

            for (int j = 0; j < 4; j++) {
                float w1 = (float)grid1[j];
                float w2 = (float)grid2[j];
                if (signs & bn_kmask_iq2xs[j + 0]) w1 = -w1;
                if (signs & bn_kmask_iq2xs[j + 4]) w2 = -w2;
                out[j + 0] = db * w1;
                out[j + 4] = db * w2;
            }
            out += 8;
        }
        qs += 8;
    }
}

// --- IQ3_S dequantization ---

void bn_quant_dequant_iq3s(const BnBlockIQ3S *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    const uint8_t *qs = block->qs;
    const uint8_t *qh = block->qh;
    const uint8_t *signs = block->signs;
    const uint8_t *scales = block->scales;

    for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
        uint8_t sc_byte = scales[ib32 / 2];
        int sc_nib = (sc_byte >> ((ib32 % 2) * 4)) & 0xF;
        float dl = d * (1 + 2 * sc_nib);

        for (int l = 0; l < 8; l++) {
            int idx9 = qs[ib32 * 8 + l] | (((qh[ib32] >> l) & 1) << 8);
            uint32_t grid_val = bn_iq3s_grid[idx9];
            const uint8_t *grid = (const uint8_t *)&grid_val;
            int sign_byte_idx = ib32 * 4 + (l * 4) / 8;
            int sign_bit_base = (l * 4) % 8;
            uint8_t sign_byte = signs[sign_byte_idx];

            for (int k = 0; k < 4; k++) {
                float w = (float)grid[k];
                if ((sign_byte >> (sign_bit_base + k)) & 1) w = -w;
                *out++ = dl * w;
            }
        }
    }
}

// --- IQ2_XXS dequantization ---
// Per GGML: processes 2 sub-blocks at a time. Each uint32 encodes 16 elements:
// bits 0-7: grid index for 8 elements, bits 8-15: grid2 index for 8 elements,
// bits 16-22: 7-bit sign index (applied to grid only, not grid2).
// Scale comes from bits 28-31 of the SECOND uint32 per sub-block pair.

void bn_quant_dequant_iq2xxs(const BnBlockIQ2XXS *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);
    const uint16_t *q = block->qs;

    for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32 += 2) {
        const uint32_t *q32 = (const uint32_t *)q;
        float db1 = d * (0.5f + (q32[1] >> 28));
        float db2 = d * (0.5f + (q32[3] >> 28));

        for (int l = 0; l < 4; l++) {
            const uint8_t *grid  = (const uint8_t *)&bn_iq2xxs_grid[q32[l] & 0xFF];
            const uint8_t *grid2 = (const uint8_t *)&bn_iq2xxs_grid[(q32[l] >> 8) & 0xFF];
            uint8_t signs = bn_ksigns_iq2xs[(q32[l] >> 16) & 127];
            float db = l < 2 ? db1 : db2;

            for (int j = 0; j < 8; j++) {
                *out++ = db * grid[j] * (1 - 2 * (int)(signs & 1));
                signs >>= 1;
            }
            for (int j = 0; j < 8; j++) {
                *out++ = db * grid2[j];
            }
        }
        q += 8;  // advance 16 bytes = 8 uint16_t
    }
}

// --- IQ2_XS dequantization ---
// Each qs[i] is uint16_t: bits 0-8 = 9-bit grid index, bits 9-15 = 7-bit sign index.
// One entry covers 8 elements. 4 entries per 32-element sub-block.

void bn_quant_dequant_iq2xs(const BnBlockIQ2XS *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);

    for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
        uint8_t sc_byte = block->scales[ib32 / 2];
        int sc_nib = (sc_byte >> ((ib32 & 1) * 4)) & 0xF;
        float db = d * (0.5f + sc_nib);

        for (int l = 0; l < 4; l++) {
            uint16_t q2 = block->qs[4 * ib32 + l];
            const uint8_t *grid = (const uint8_t *)&bn_iq2xs_grid[q2 & 511];
            uint8_t signs = bn_ksigns_iq2xs[q2 >> 9];

            for (int j = 0; j < 8; j++) {
                *out++ = db * grid[j] * (1 - 2 * (int)((signs >> j) & 1));
            }
        }
    }
}

// --- IQ2_S dequantization ---
// Grid index: 8 bits from qs[first half] + 2 bits from qh = 10-bit index.
// Signs: 8 bits from qs[second half] (qs[QK_K/8 + offset]).
// Scale: db = d * (1 + 2 * scale_nibble).

void bn_quant_dequant_iq2s(const BnBlockIQ2S *block, float *out) {
    float d = bn_fp16_to_fp32(block->d);

    for (int ib32 = 0; ib32 < BN_QK_K / 32; ib32++) {
        uint8_t sc_byte = block->scales[ib32 / 2];
        int sc_nib = (sc_byte >> ((ib32 & 1) * 4)) & 0xF;
        float db = d * (1 + 2 * sc_nib);

        for (int l = 0; l < 4; l++) {
            int grid_idx = block->qs[4 * ib32 + l] |
                           (((block->qh[ib32] >> (2 * l)) & 3) << 8);
            const uint8_t *grid = (const uint8_t *)&bn_iq2s_grid[grid_idx];
            uint8_t signs = block->qs[BN_QK_K / 8 + 4 * ib32 + l];

            for (int j = 0; j < 8; j++) {
                *out++ = db * grid[j] * (1 - 2 * (int)((signs >> j) & 1));
            }
        }
    }
}
