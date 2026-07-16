#include "model.h"
#include "model_arch.h"
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

    if (bn_quant_dequant_row(m->weights.emb_type, m->weights.token_embedding,
                             token, dim, out) != 0) {
        SH_LOG_ERROR("Unsupported embedding type");
        memset(out, 0, dim * sizeof(float));
        return;
    }

    if (bn_model_arch_uses_per_layer_embedding(&m->config)) {
        float scale = sqrtf((float)dim);
        for (int i = 0; i < dim; i++)
            out[i] *= scale;
    }
}
