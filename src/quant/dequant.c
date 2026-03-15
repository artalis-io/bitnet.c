#include "quant_internal.h"
#include <assert.h>

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

    int qhbit = 0;
    for (int j = 0; j < BN_QK_K; j += 64) {
        uint8_t sc, m;
        int sub = j / 32;
        bn_q4k_get_scale_min(sub, block->scales, &sc, &m);
        float ds = d * sc;
        float dm = dmin * m;
        for (int l = 0; l < 32; l++) {
            int q5 = (qs[l] & 0xF) | (((qh[qhbit / 8] >> (qhbit % 8)) & 1) << 4);
            out[j + l] = ds * q5 - dm;
            qhbit++;
        }
        bn_q4k_get_scale_min(sub + 1, block->scales, &sc, &m);
        ds = d * sc;
        dm = dmin * m;
        for (int l = 0; l < 32; l++) {
            int q5 = (qs[l] >> 4) | (((qh[qhbit / 8] >> (qhbit % 8)) & 1) << 4);
            out[j + l + 32] = ds * q5 - dm;
            qhbit++;
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
