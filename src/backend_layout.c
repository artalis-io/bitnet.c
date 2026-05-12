#include "backend_layout.h"
#include "gguf.h"
#include <stdlib.h>
#include <string.h>

const char *bn_backend_layout_reason_string(BnBackendLayoutReason reason) {
    switch (reason) {
        case BN_BACKEND_LAYOUT_OK: return "ok";
        case BN_BACKEND_LAYOUT_NO_GPU: return "no_gpu";
        case BN_BACKEND_LAYOUT_NO_BUFFER_CREATE: return "no_buffer_create";
        case BN_BACKEND_LAYOUT_NO_BUFFER_CREATE_BIASED: return "no_buffer_create_biased";
        case BN_BACKEND_LAYOUT_MISSING_WEIGHT: return "missing_weight";
        case BN_BACKEND_LAYOUT_I2S_NOT_STACKABLE: return "i2s_not_stackable";
        case BN_BACKEND_LAYOUT_TYPE_MISMATCH: return "type_mismatch";
        case BN_BACKEND_LAYOUT_COL_MISMATCH: return "col_mismatch";
        case BN_BACKEND_LAYOUT_ZERO_SIZE: return "zero_size";
        case BN_BACKEND_LAYOUT_ALLOC_FAILED: return "alloc_failed";
        case BN_BACKEND_LAYOUT_BIAS_UNSUPPORTED: return "bias_unsupported";
        default: return "unknown";
    }
}

BnBackendLayoutReason bn_backend_layout_stackable_reason(const BnQWeight *a,
                                                         const BnQWeight *b) {
    if (!a || !b || !a->data || !b->data) return BN_BACKEND_LAYOUT_MISSING_WEIGHT;
    if (a->type == BN_GGUF_TENSOR_I2_S || b->type == BN_GGUF_TENSOR_I2_S)
        return BN_BACKEND_LAYOUT_I2S_NOT_STACKABLE;
    if (a->type != b->type) return BN_BACKEND_LAYOUT_TYPE_MISMATCH;
    if (a->cols != b->cols) return BN_BACKEND_LAYOUT_COL_MISMATCH;
    return BN_BACKEND_LAYOUT_OK;
}

int bn_backend_layout_stackable(const BnQWeight *a, const BnQWeight *b) {
    return bn_backend_layout_stackable_reason(a, b) == BN_BACKEND_LAYOUT_OK;
}

BnBackendLayoutReason bn_backend_layout_stacked2_reason(const BnGPUBackend *gpu,
                                                        const BnQWeight *a,
                                                        const BnQWeight *b) {
    if (!gpu) return BN_BACKEND_LAYOUT_NO_GPU;
    if (!gpu->buffer_create) return BN_BACKEND_LAYOUT_NO_BUFFER_CREATE;
    BnBackendLayoutReason reason = bn_backend_layout_stackable_reason(a, b);
    if (reason != BN_BACKEND_LAYOUT_OK) return reason;
    if (bn_qweight_data_size(a) == 0 || bn_qweight_data_size(b) == 0)
        return BN_BACKEND_LAYOUT_ZERO_SIZE;
    return BN_BACKEND_LAYOUT_OK;
}

BnBackendLayoutReason bn_backend_layout_biased_qweight_reason(const BnGPUBackend *gpu,
                                                              const BnQWeight *w,
                                                              const float *bias) {
    if (!gpu) return BN_BACKEND_LAYOUT_NO_GPU;
    if (!gpu->buffer_create_biased) return BN_BACKEND_LAYOUT_NO_BUFFER_CREATE_BIASED;
    if (!w || !w->data || !bias) return BN_BACKEND_LAYOUT_MISSING_WEIGHT;
    if (bn_qweight_data_size(w) == 0) return BN_BACKEND_LAYOUT_ZERO_SIZE;
    return BN_BACKEND_LAYOUT_OK;
}

BnBackendLayoutReason bn_backend_layout_stacked3_qkv_reason(const BnGPUBackend *gpu,
                                                            const BnQWeight *q,
                                                            const BnQWeight *k,
                                                            const BnQWeight *v,
                                                            const float *q_bias,
                                                            const float *k_bias,
                                                            const float *v_bias,
                                                            int q_bias_fused,
                                                            int k_bias_fused,
                                                            int v_bias_fused) {
    if (!gpu) return BN_BACKEND_LAYOUT_NO_GPU;
    if (!gpu->buffer_create) return BN_BACKEND_LAYOUT_NO_BUFFER_CREATE;
    BnBackendLayoutReason reason = bn_backend_layout_stackable_reason(q, k);
    if (reason != BN_BACKEND_LAYOUT_OK) return reason;
    reason = bn_backend_layout_stackable_reason(q, v);
    if (reason != BN_BACKEND_LAYOUT_OK) return reason;
    if (bn_qweight_data_size(q) == 0 ||
        bn_qweight_data_size(k) == 0 ||
        bn_qweight_data_size(v) == 0)
        return BN_BACKEND_LAYOUT_ZERO_SIZE;

    int any_bias = q_bias || k_bias || v_bias;
    int all_bias = q_bias && k_bias && v_bias;
    int all_fused = q_bias_fused && k_bias_fused && v_bias_fused;
    if (any_bias && !(all_bias && all_fused && gpu->buffer_create_biased))
        return BN_BACKEND_LAYOUT_BIAS_UNSUPPORTED;
    return BN_BACKEND_LAYOUT_OK;
}

void *bn_backend_layout_upload_stacked2(BnGPUBackend *gpu,
                                        const BnQWeight *a,
                                        const BnQWeight *b) {
    if (bn_backend_layout_stacked2_reason(gpu, a, b) != BN_BACKEND_LAYOUT_OK)
        return NULL;
    size_t a_sz = bn_qweight_data_size(a);
    size_t b_sz = bn_qweight_data_size(b);

    size_t combined_sz = a_sz + b_sz;
    uint8_t *combined = (uint8_t *)malloc(combined_sz);
    if (!combined) return NULL;

    memcpy(combined, a->data, a_sz);
    memcpy(combined + a_sz, b->data, b_sz);

    void *buf = gpu->buffer_create(gpu->ctx, combined, combined_sz,
                                   a->type, a->rows + b->rows, a->cols);
    free(combined);
    return buf;
}

void *bn_backend_layout_upload_biased_qweight(BnGPUBackend *gpu,
                                              const BnQWeight *w,
                                              const float *bias) {
    if (bn_backend_layout_biased_qweight_reason(gpu, w, bias) != BN_BACKEND_LAYOUT_OK)
        return NULL;
    size_t sz = bn_qweight_data_size(w);
    return gpu->buffer_create_biased(gpu->ctx, w->data, sz,
                                     w->type, w->rows, w->cols,
                                     bias, (size_t)w->rows * sizeof(float));
}

void *bn_backend_layout_upload_stacked3_qkv(BnGPUBackend *gpu,
                                            const BnQWeight *q,
                                            const BnQWeight *k,
                                            const BnQWeight *v,
                                            const float *q_bias,
                                            const float *k_bias,
                                            const float *v_bias,
                                            int q_bias_fused,
                                            int k_bias_fused,
                                            int v_bias_fused) {
    if (bn_backend_layout_stacked3_qkv_reason(gpu, q, k, v,
                                              q_bias, k_bias, v_bias,
                                              q_bias_fused, k_bias_fused,
                                              v_bias_fused) != BN_BACKEND_LAYOUT_OK) {
        return NULL;
    }

    size_t q_sz = bn_qweight_data_size(q);
    size_t k_sz = bn_qweight_data_size(k);
    size_t v_sz = bn_qweight_data_size(v);

    int total_rows = q->rows + k->rows + v->rows;
    size_t combined_sz = q_sz + k_sz + v_sz;
    uint8_t *combined = (uint8_t *)malloc(combined_sz);
    if (!combined) return NULL;

    memcpy(combined, q->data, q_sz);
    memcpy(combined + q_sz, k->data, k_sz);
    memcpy(combined + q_sz + k_sz, v->data, v_sz);

    void *buf = NULL;
    int all_biased = q_bias && k_bias && v_bias &&
                     q_bias_fused && k_bias_fused && v_bias_fused;
    int no_bias = !q_bias && !k_bias && !v_bias;

    if (all_biased && gpu->buffer_create_biased) {
        float *cbias = (float *)malloc((size_t)total_rows * sizeof(float));
        if (cbias) {
            memcpy(cbias, q_bias, (size_t)q->rows * sizeof(float));
            memcpy(cbias + q->rows, k_bias, (size_t)k->rows * sizeof(float));
            memcpy(cbias + q->rows + k->rows, v_bias, (size_t)v->rows * sizeof(float));
            buf = gpu->buffer_create_biased(gpu->ctx, combined, combined_sz,
                                            q->type, total_rows, q->cols,
                                            cbias, (size_t)total_rows * sizeof(float));
            free(cbias);
        }
    } else if (no_bias) {
        buf = gpu->buffer_create(gpu->ctx, combined, combined_sz,
                                 q->type, total_rows, q->cols);
    }

    free(combined);
    return buf;
}
