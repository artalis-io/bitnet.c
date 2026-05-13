#ifndef BN_QUANT_DISPATCH_INTERNAL_H
#define BN_QUANT_DISPATCH_INTERNAL_H

#include "quant_ctx.h"

void bn_quant_matvec_impl(float *out,
                          const BnQWeight *W,
                          const float *x,
                          int8_t *x_q_buf,
                          BnThreadPool *pool,
                          const BnPreparedWeight *prepared);

void bn_quant_x_to_q8k_scalar(const float *x,
                              int8_t *x_q,
                              float *x_d,
                              int16_t *x_bsums,
                              int n);

#endif // BN_QUANT_DISPATCH_INTERNAL_H
