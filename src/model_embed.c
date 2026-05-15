#include "model.h"
#include "sh_log.h"
#include <math.h>
#include <string.h>

void bn_model_embed_token(const BnModel *m, float *out, int token) {
    int dim = m->config.dim;

    if (token < 0 || token >= m->config.vocab_size) {
        SH_LOG_ERROR("Token out of range");
        memset(out, 0, dim * sizeof(float));
        return;
    }

    if (m->weights.emb_type == BN_GGUF_TENSOR_F16) {
        const uint16_t *emb = (const uint16_t *)m->weights.token_embedding;
        const uint16_t *row = emb + (size_t)token * dim;
        for (int i = 0; i < dim; i++) {
            out[i] = bn_fp16_to_fp32(row[i]);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q4_0) {
        const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)m->weights.token_embedding;
        int n_blocks_per_row = dim / 32;
        const BnBlockQ4_0 *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q4_0(&row[b], out + b * 32);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q8_0) {
        const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)m->weights.token_embedding;
        int n_blocks_per_row = dim / 32;
        const BnBlockQ8_0 *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q8_0(&row[b], out + b * 32);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q2_K) {
        const BnBlockQ2K *blocks = (const BnBlockQ2K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ2K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q2k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q3_K) {
        const BnBlockQ3K *blocks = (const BnBlockQ3K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ3K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q3k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q4_K) {
        const BnBlockQ4K *blocks = (const BnBlockQ4K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ4K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q4k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q5_K) {
        const BnBlockQ5K *blocks = (const BnBlockQ5K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ5K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q5k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q6_K) {
        const BnBlockQ6K *blocks = (const BnBlockQ6K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ6K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q6k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q8_K) {
        const BnBlockQ8K *blocks = (const BnBlockQ8K *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockQ8K *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q8k(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q4_1) {
        const BnBlockQ4_1 *blocks = (const BnBlockQ4_1 *)m->weights.token_embedding;
        int n_blocks_per_row = dim / 32;
        const BnBlockQ4_1 *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q4_1(&row[b], out + b * 32);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_Q5_1) {
        const BnBlockQ5_1 *blocks = (const BnBlockQ5_1 *)m->weights.token_embedding;
        int n_blocks_per_row = dim / 32;
        const BnBlockQ5_1 *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_q5_1(&row[b], out + b * 32);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_BF16) {
        const uint16_t *emb = (const uint16_t *)m->weights.token_embedding;
        const uint16_t *row = emb + (size_t)token * dim;
        for (int i = 0; i < dim; i++) {
            out[i] = bn_bf16_to_fp32(row[i]);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ4_NL) {
        const BnBlockIQ4NL *blocks = (const BnBlockIQ4NL *)m->weights.token_embedding;
        int n_blocks_per_row = dim / 32;
        const BnBlockIQ4NL *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq4nl(&row[b], out + b * 32);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ4_XS) {
        const BnBlockIQ4XS *blocks = (const BnBlockIQ4XS *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ4XS *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq4xs(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ3_XXS) {
        const BnBlockIQ3XXS *blocks = (const BnBlockIQ3XXS *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ3XXS *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq3xxs(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ3_S) {
        const BnBlockIQ3S *blocks = (const BnBlockIQ3S *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ3S *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq3s(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ2_XXS) {
        const BnBlockIQ2XXS *blocks = (const BnBlockIQ2XXS *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ2XXS *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq2xxs(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ2_XS) {
        const BnBlockIQ2XS *blocks = (const BnBlockIQ2XS *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ2XS *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq2xs(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_IQ2_S) {
        const BnBlockIQ2S *blocks = (const BnBlockIQ2S *)m->weights.token_embedding;
        int n_blocks_per_row = dim / BN_QK_K;
        const BnBlockIQ2S *row = blocks + (size_t)token * n_blocks_per_row;
        for (int b = 0; b < n_blocks_per_row; b++) {
            bn_quant_dequant_iq2s(&row[b], out + b * BN_QK_K);
        }
    } else if (m->weights.emb_type == BN_GGUF_TENSOR_F32) {
        const float *emb = (const float *)m->weights.token_embedding;
        memcpy(out, emb + (size_t)token * dim, dim * sizeof(float));
    } else {
        SH_LOG_ERROR("Unsupported embedding type");
        memset(out, 0, dim * sizeof(float));
        return;
    }

    if (m->config.arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) {
        float scale = sqrtf((float)dim);
        for (int i = 0; i < dim; i++)
            out[i] *= scale;
    }
}
