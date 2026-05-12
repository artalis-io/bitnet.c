#ifndef BN_BACKEND_LAYOUT_H
#define BN_BACKEND_LAYOUT_H

#include "gpu_backend.h"
#include "quant.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline int bn_backend_layout_stackable(const BnQWeight *a, const BnQWeight *b) {
    return a && b &&
           a->data && b->data &&
           a->type != BN_GGUF_TENSOR_I2_S &&
           a->type == b->type &&
           a->cols == b->cols;
}

static inline void *bn_backend_layout_upload_stacked2(BnGPUBackend *gpu,
                                                       const BnQWeight *a,
                                                       const BnQWeight *b) {
    if (!gpu || !gpu->buffer_create || !bn_backend_layout_stackable(a, b)) return NULL;

    size_t a_sz = bn_qweight_data_size(a);
    size_t b_sz = bn_qweight_data_size(b);
    if (a_sz == 0 || b_sz == 0) return NULL;

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

static inline void *bn_backend_layout_upload_biased_qweight(BnGPUBackend *gpu,
                                                            const BnQWeight *w,
                                                            const float *bias) {
    if (!gpu || !gpu->buffer_create_biased || !w || !w->data || !bias) return NULL;
    size_t sz = bn_qweight_data_size(w);
    if (sz == 0) return NULL;
    return gpu->buffer_create_biased(gpu->ctx, w->data, sz,
                                     w->type, w->rows, w->cols,
                                     bias, (size_t)w->rows * sizeof(float));
}

static inline void *bn_backend_layout_upload_stacked3_qkv(BnGPUBackend *gpu,
                                                          const BnQWeight *q,
                                                          const BnQWeight *k,
                                                          const BnQWeight *v,
                                                          const float *q_bias,
                                                          const float *k_bias,
                                                          const float *v_bias,
                                                          int q_bias_fused,
                                                          int k_bias_fused,
                                                          int v_bias_fused) {
    if (!gpu || !gpu->buffer_create ||
        !bn_backend_layout_stackable(q, k) ||
        !bn_backend_layout_stackable(q, v)) {
        return NULL;
    }

    size_t q_sz = bn_qweight_data_size(q);
    size_t k_sz = bn_qweight_data_size(k);
    size_t v_sz = bn_qweight_data_size(v);
    if (q_sz == 0 || k_sz == 0 || v_sz == 0) return NULL;

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

#endif // BN_BACKEND_LAYOUT_H
