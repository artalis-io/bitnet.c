#include "backend_layout.h"
#include "backend_quant.h"
#include "gguf.h"
#include <stdlib.h>
#include <string.h>

static void prepared_stats_clear(BnBackendLayoutPreparedStats *stats) {
    if (stats) memset(stats, 0, sizeof(*stats));
}

static void prepared_stats_add(BnBackendLayoutPreparedStats *dst,
                               const BnBackendLayoutPreparedStats *src) {
    if (!dst || !src) return;
    dst->lowbit_repack_bytes += src->lowbit_repack_bytes;
    dst->kquant_scale_table_bytes += src->kquant_scale_table_bytes;
    dst->expanded_kquant_weight_bytes += src->expanded_kquant_weight_bytes;
    dst->f32_scale_table_bytes += src->f32_scale_table_bytes;
}

static void prepared_stats_add_bytes(BnBackendLayoutPreparedStats *stats,
                                     BnPreparedWeightKind kind,
                                     size_t bytes) {
    if (!stats || bytes == 0) return;
    switch (kind) {
        case BN_PREPARED_WEIGHT_Q4_0_REPACK:
            stats->lowbit_repack_bytes += bytes;
            break;
        case BN_PREPARED_WEIGHT_Q4_K_SCALES:
            stats->kquant_scale_table_bytes += bytes;
            break;
        case BN_PREPARED_WEIGHT_Q6_K_EXPANDED:
            stats->expanded_kquant_weight_bytes += bytes;
            break;
        case BN_PREPARED_WEIGHT_Q8_0_F32_SCALES:
            stats->f32_scale_table_bytes += bytes;
            break;
        default:
            break;
    }
}

static void prepared_qweight_size_one(const BnQWeight *w,
                                      BnBackendLayoutPreparedStats *stats) {
    if (!stats) return;
    BnPreparedWeightKind kind = BN_PREPARED_WEIGHT_NONE;
    size_t bytes = bn_quant_prepared_qweight_size(w, &kind);
    prepared_stats_add_bytes(stats, kind, bytes);
}

static void prepared_qweight_size_layer(const BnLayerWeights *lw,
                                        BnBackendLayoutPreparedStats *stats) {
    prepared_qweight_size_one(&lw->attn.wq, stats);
    prepared_qweight_size_one(&lw->attn.wk, stats);
    prepared_qweight_size_one(&lw->attn.wv, stats);
    prepared_qweight_size_one(&lw->attn.wo, stats);
    prepared_qweight_size_one(&lw->ssm.wqkv, stats);
    prepared_qweight_size_one(&lw->ssm.wz, stats);
    prepared_qweight_size_one(&lw->ssm.ssm_out, stats);
    prepared_qweight_size_one(&lw->ffn.ffn_gate, stats);
    prepared_qweight_size_one(&lw->ffn.ffn_up, stats);
    prepared_qweight_size_one(&lw->ffn.ffn_down, stats);
}

static void prepared_qweight_prepare_one(BnBackendModel *backend,
                                         const BnQWeight *w,
                                         SHArena *arena) {
    BnPreparedWeight prepared = { 0 };
    if (bn_quant_prepare_qweight(&prepared, w, arena) == 0)
        (void)bn_backend_model_register_prepared_qweight(backend, w,
                                                         &prepared);
}

static void prepared_qweight_prepare_layer(BnBackendModel *backend,
                                           const BnLayerWeights *lw,
                                           SHArena *arena) {
    prepared_qweight_prepare_one(backend, &lw->attn.wq, arena);
    prepared_qweight_prepare_one(backend, &lw->attn.wk, arena);
    prepared_qweight_prepare_one(backend, &lw->attn.wv, arena);
    prepared_qweight_prepare_one(backend, &lw->attn.wo, arena);
    prepared_qweight_prepare_one(backend, &lw->ssm.wqkv, arena);
    prepared_qweight_prepare_one(backend, &lw->ssm.wz, arena);
    prepared_qweight_prepare_one(backend, &lw->ssm.ssm_out, arena);
    prepared_qweight_prepare_one(backend, &lw->ffn.ffn_gate, arena);
    prepared_qweight_prepare_one(backend, &lw->ffn.ffn_up, arena);
    prepared_qweight_prepare_one(backend, &lw->ffn.ffn_down, arena);
}

static void *upload_stacked_quant_layout(BnGPUBackend *gpu,
                                         const void *data, size_t size,
                                         int type, int rows, int cols) {
    if (!gpu || !data || size == 0)
        return NULL;
    if (gpu->buffer_create_quant_only)
        return gpu->buffer_create_quant_only(gpu->ctx, data, size,
                                             type, rows, cols);
    return gpu->buffer_create(gpu->ctx, data, size, type, rows, cols);
}

const char *bn_backend_layout_reason_string(BnBackendLayoutReason reason) {
    switch (reason) {
        case BN_BACKEND_LAYOUT_OK: return "ok";
        case BN_BACKEND_LAYOUT_NO_GPU: return "no_gpu";
        case BN_BACKEND_LAYOUT_NO_BUFFER_CREATE: return "no_buffer_create";
        case BN_BACKEND_LAYOUT_NO_BUFFER_CREATE_BIASED: return "no_buffer_create_biased";
        case BN_BACKEND_LAYOUT_MISSING_WEIGHT: return "missing_weight";
        case BN_BACKEND_LAYOUT_EMBEDDED_SCALE_NOT_STACKABLE:
            return "embedded_scale_not_stackable";
        case BN_BACKEND_LAYOUT_TYPE_MISMATCH: return "type_mismatch";
        case BN_BACKEND_LAYOUT_COL_MISMATCH: return "col_mismatch";
        case BN_BACKEND_LAYOUT_ZERO_SIZE: return "zero_size";
        case BN_BACKEND_LAYOUT_ALLOC_FAILED: return "alloc_failed";
        case BN_BACKEND_LAYOUT_BIAS_UNSUPPORTED: return "bias_unsupported";
        default: return "unknown";
    }
}

size_t bn_backend_layout_qweight_data_size(const BnQWeight *w) {
    return bn_qweight_data_size(w);
}

BnBackendLayoutReason bn_backend_layout_stackable_reason(const BnQWeight *a,
                                                         const BnQWeight *b) {
    if (!a || !b || !a->data || !b->data) return BN_BACKEND_LAYOUT_MISSING_WEIGHT;
    if (!bn_backend_quant_allows_stacked_layout(a->type) ||
        !bn_backend_quant_allows_stacked_layout(b->type))
        return BN_BACKEND_LAYOUT_EMBEDDED_SCALE_NOT_STACKABLE;
    if (a->type != b->type) return BN_BACKEND_LAYOUT_TYPE_MISMATCH;
    if (a->cols != b->cols) return BN_BACKEND_LAYOUT_COL_MISMATCH;
    return BN_BACKEND_LAYOUT_OK;
}

int bn_backend_layout_stackable(const BnQWeight *a, const BnQWeight *b) {
    return bn_backend_layout_stackable_reason(a, b) == BN_BACKEND_LAYOUT_OK;
}

int bn_backend_layout_stackable3(const BnQWeight *a,
                                 const BnQWeight *b,
                                 const BnQWeight *c) {
    return bn_backend_layout_stackable(a, b) &&
           bn_backend_layout_stackable(a, c);
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

    if (gpu->buffer_create_stacked2) {
        return gpu->buffer_create_stacked2(gpu->ctx, a->data, a_sz,
                                           b->data, b_sz, a->type,
                                           a->rows + b->rows, a->cols);
    }

    size_t combined_sz = a_sz + b_sz;
    uint8_t *combined = (uint8_t *)malloc(combined_sz);
    if (!combined) return NULL;

    memcpy(combined, a->data, a_sz);
    memcpy(combined + a_sz, b->data, b_sz);

    void *buf = upload_stacked_quant_layout(gpu, combined, combined_sz,
                                            a->type, a->rows + b->rows,
                                            a->cols);
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
            if (gpu->buffer_create_stacked3_biased) {
                buf = gpu->buffer_create_stacked3_biased(
                    gpu->ctx, q->data, q_sz, k->data, k_sz, v->data, v_sz,
                    q->type, total_rows, q->cols, cbias,
                    (size_t)total_rows * sizeof(float));
            } else {
                buf = gpu->buffer_create_biased(gpu->ctx, combined,
                                                combined_sz, q->type,
                                                total_rows, q->cols, cbias,
                                                (size_t)total_rows * sizeof(float));
            }
            free(cbias);
        }
    } else if (no_bias) {
        if (gpu->buffer_create_stacked3) {
            buf = gpu->buffer_create_stacked3(gpu->ctx,
                                              q->data, q_sz,
                                              k->data, k_sz,
                                              v->data, v_sz,
                                              q->type, total_rows, q->cols);
        } else {
            buf = upload_stacked_quant_layout(gpu, combined, combined_sz,
                                              q->type, total_rows, q->cols);
        }
    }

    free(combined);
    return buf;
}

size_t bn_backend_layout_prepared_qweights_size(const BnConfig *config,
                                                const BnWeights *weights,
                                                BnBackendLayoutPreparedStats *stats) {
    BnBackendLayoutPreparedStats local = { 0 };
    if (!stats) stats = &local;
    prepared_stats_clear(stats);
    if (!config || !weights) return 0;

    for (int i = 0; i < config->n_layers; i++)
        prepared_qweight_size_layer(&weights->layers[i], stats);
    prepared_qweight_size_one(&weights->output_weight, stats);
    prepared_qweight_size_one(&weights->tied_embedding_weight, stats);

    return stats->lowbit_repack_bytes + stats->kquant_scale_table_bytes +
           stats->expanded_kquant_weight_bytes +
           stats->f32_scale_table_bytes;
}

void bn_backend_layout_prepare_qweights(BnBackendModel *backend,
                                        const BnConfig *config,
                                        const BnWeights *weights,
                                        SHArena *arena,
                                        BnBackendLayoutPreparedStats *stats) {
    BnBackendLayoutPreparedStats local = { 0 };
    if (!backend || !config || !weights || !arena) {
        prepared_stats_clear(stats);
        return;
    }

    (void)bn_backend_layout_prepared_qweights_size(config, weights, &local);
    if (stats) {
        prepared_stats_clear(stats);
        prepared_stats_add(stats, &local);
    }

    for (int i = 0; i < config->n_layers; i++)
        prepared_qweight_prepare_layer(backend, &weights->layers[i], arena);
    prepared_qweight_prepare_one(backend, &weights->output_weight, arena);
    prepared_qweight_prepare_one(backend, &weights->tied_embedding_weight,
                                 arena);
}
