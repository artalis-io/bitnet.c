#ifndef BN_BACKEND_LAYOUT_H
#define BN_BACKEND_LAYOUT_H

#include "gpu_backend.h"
#include "quant.h"

typedef enum {
    BN_BACKEND_LAYOUT_OK = 0,
    BN_BACKEND_LAYOUT_NO_GPU,
    BN_BACKEND_LAYOUT_NO_BUFFER_CREATE,
    BN_BACKEND_LAYOUT_NO_BUFFER_CREATE_BIASED,
    BN_BACKEND_LAYOUT_MISSING_WEIGHT,
    BN_BACKEND_LAYOUT_I2S_NOT_STACKABLE,
    BN_BACKEND_LAYOUT_TYPE_MISMATCH,
    BN_BACKEND_LAYOUT_COL_MISMATCH,
    BN_BACKEND_LAYOUT_ZERO_SIZE,
    BN_BACKEND_LAYOUT_ALLOC_FAILED,
    BN_BACKEND_LAYOUT_BIAS_UNSUPPORTED,
} BnBackendLayoutReason;

const char *bn_backend_layout_reason_string(BnBackendLayoutReason reason);

BnBackendLayoutReason bn_backend_layout_stackable_reason(const BnQWeight *a,
                                                         const BnQWeight *b);

int bn_backend_layout_stackable(const BnQWeight *a, const BnQWeight *b);

BnBackendLayoutReason bn_backend_layout_stacked2_reason(const BnGPUBackend *gpu,
                                                        const BnQWeight *a,
                                                        const BnQWeight *b);

BnBackendLayoutReason bn_backend_layout_biased_qweight_reason(const BnGPUBackend *gpu,
                                                              const BnQWeight *w,
                                                              const float *bias);

BnBackendLayoutReason bn_backend_layout_stacked3_qkv_reason(const BnGPUBackend *gpu,
                                                            const BnQWeight *q,
                                                            const BnQWeight *k,
                                                            const BnQWeight *v,
                                                            const float *q_bias,
                                                            const float *k_bias,
                                                            const float *v_bias,
                                                            int q_bias_fused,
                                                            int k_bias_fused,
                                                            int v_bias_fused);

void *bn_backend_layout_upload_stacked2(BnGPUBackend *gpu,
                                        const BnQWeight *a,
                                        const BnQWeight *b);

void *bn_backend_layout_upload_biased_qweight(BnGPUBackend *gpu,
                                              const BnQWeight *w,
                                              const float *bias);

void *bn_backend_layout_upload_stacked3_qkv(BnGPUBackend *gpu,
                                            const BnQWeight *q,
                                            const BnQWeight *k,
                                            const BnQWeight *v,
                                            const float *q_bias,
                                            const float *k_bias,
                                            const float *v_bias,
                                            int q_bias_fused,
                                            int k_bias_fused,
                                            int v_bias_fused);

#endif // BN_BACKEND_LAYOUT_H
