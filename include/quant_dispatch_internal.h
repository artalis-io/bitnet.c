#ifndef BN_QUANT_DISPATCH_INTERNAL_H
#define BN_QUANT_DISPATCH_INTERNAL_H

#include "quant_ctx.h"

void bn_quant_matvec_impl(float *out,
                          const BnQWeight *W,
                          const float *x,
                          int8_t *x_q_buf,
                          BnThreadPool *pool,
                          const BnPreparedWeight *prepared,
                          uint32_t flags);

void bn_quant_x_to_q8k_scalar(const float *x,
                              int8_t *x_q,
                              float *x_d,
                              int16_t *x_bsums,
                              int n);

int bn_quant_policy_avx512_q5k_vnni_enabled(int rows);
int bn_quant_policy_avx2_kquant_float_for_tasks(
    const BnMatvecTask *tasks,
    int n_tasks);
int bn_quant_policy_reference_q4_dot_enabled(uint32_t flags);
int bn_quant_policy_reference_q6_dot_enabled(uint32_t flags);
int bn_quant_policy_batch_reference_q4_dot_enabled(
    const BnMatvecTask *tasks,
    int n_tasks);
int bn_quant_policy_wasm_q4_canonical4_enabled(void);
int bn_quant_policy_q8_0_matmul_batch_enabled(void);

#endif // BN_QUANT_DISPATCH_INTERNAL_H
