#include "quant_ctx.h"
#include <assert.h>
#include <math.h>

void bn_quant_rmsnorm_q8k_avx2(const float *x, const float *w, int dim,
                               float eps, float *xb_out, int8_t *x_q,
                               float *x_d, int16_t *x_bsums) {
    assert(dim % BN_QK_K == 0);

    double sum = 0.0;
    for (int i = 0; i < dim; i++)
        sum += (double)(x[i] * x[i]);

    float scale = 1.0f / sqrtf((float)(sum / dim) + eps);
    for (int i = 0; i < dim; i++)
        xb_out[i] = x[i] * scale * w[i];

    bn_quant_x_to_q8k(xb_out, x_q, x_d, x_bsums, dim);
}
