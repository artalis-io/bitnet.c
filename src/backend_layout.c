#include "backend_layout.h"
#include "gguf.h"
#include <stdlib.h>
#include <string.h>

static void prepared_stats_clear(BnBackendLayoutPreparedStats *stats) {
    if (stats) memset(stats, 0, sizeof(*stats));
}

static void prepared_stats_add(BnBackendLayoutPreparedStats *dst,
                               const BnBackendLayoutPreparedStats *src) {
    if (!dst || !src) return;
    dst->q4_repack_bytes += src->q4_repack_bytes;
    dst->q8_scale_bytes += src->q8_scale_bytes;
}

#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__wasm_relaxed_simd__)
static size_t prepared_q4_repack_bytes(const BnQWeight *w) {
    if (w->type != BN_GGUF_TENSOR_Q4_0 || !w->data) return 0;
    if (w->rows % 4 != 0) return 0;
    size_t n_blocks = (size_t)w->rows * (w->cols / 32);
    size_t bytes = n_blocks * 16 + SH_ARENA_ALIGN;
#ifdef __wasm_relaxed_simd__
    bytes += n_blocks * sizeof(float) + SH_ARENA_ALIGN;
#else
    bytes += n_blocks * sizeof(uint16_t) + SH_ARENA_ALIGN;
#endif
    return bytes;
}

static void prepared_q4_repack(BnBackendModel *backend, const BnQWeight *w, SHArena *arena) {
    if (w->type != BN_GGUF_TENSOR_Q4_0 || !w->data) return;
    if (w->rows % 4 != 0) return;
    int n_blocks_per_row = w->cols / 32;
    size_t n_blocks = (size_t)w->rows * n_blocks_per_row;
    BnPreparedWeight prepared = { 0 };

    prepared.qs = (uint8_t *)sh_arena_alloc(arena, n_blocks * 16);
#ifdef __wasm_relaxed_simd__
    prepared.f32_scales = (float *)sh_arena_alloc(arena, n_blocks * sizeof(float));
#else
    prepared.scales = (uint16_t *)sh_arena_alloc(arena, n_blocks * sizeof(uint16_t));
#endif
    if (!prepared.qs) return;
#ifdef __wasm_relaxed_simd__
    if (!prepared.f32_scales) return;
#else
    if (!prepared.scales) return;
#endif

    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)w->data;
    int n_groups = w->rows / 4;
    for (int g = 0; g < n_groups; g++) {
        for (int b = 0; b < n_blocks_per_row; b++) {
            size_t gb = (size_t)g * n_blocks_per_row + b;
            for (int r = 0; r < 4; r++) {
                size_t src = (size_t)(g * 4 + r) * n_blocks_per_row + b;
#ifdef __wasm_relaxed_simd__
                prepared.f32_scales[gb * 4 + r] = bn_fp16_to_fp32(blocks[src].d);
#else
                prepared.scales[gb * 4 + r] = blocks[src].d;
#endif
            }
            uint8_t *dst = prepared.qs + gb * 64;
            for (int ng = 0; ng < 4; ng++) {
                for (int r = 0; r < 4; r++) {
                    size_t src = (size_t)(g * 4 + r) * n_blocks_per_row + b;
                    const uint8_t *qs = blocks[src].qs + ng * 4;
                    uint8_t *dp = dst + ng * 16 + r * 4;
                    for (int j = 0; j < 4; j++)
                        dp[j] = qs[j] ^ 0x88;
                }
            }
        }
    }
    (void)bn_backend_model_register_prepared_qweight(backend, w, &prepared);
}
#else
static size_t prepared_q4_repack_bytes(const BnQWeight *w) {
    (void)w;
    return 0;
}

static void prepared_q4_repack(BnBackendModel *backend, const BnQWeight *w, SHArena *arena) {
    (void)backend;
    (void)w;
    (void)arena;
}
#endif

#ifdef __wasm_relaxed_simd__
static size_t prepared_q8_f32_scale_bytes(const BnQWeight *w) {
    if (w->type != BN_GGUF_TENSOR_Q8_0 || !w->data) return 0;
    if ((w->cols & 31) != 0) return 0;
    size_t n_blocks = (size_t)w->rows * (w->cols / 32);
    return n_blocks * sizeof(float) + SH_ARENA_ALIGN;
}

static void prepared_q8_f32_scales(BnBackendModel *backend, const BnQWeight *w,
                                   SHArena *arena) {
    if (w->type != BN_GGUF_TENSOR_Q8_0 || !w->data) return;
    if ((w->cols & 31) != 0) return;
    int n_blocks_per_row = w->cols / 32;
    size_t n_blocks = (size_t)w->rows * n_blocks_per_row;
    BnPreparedWeight prepared = { 0 };
    prepared.f32_scales = (float *)sh_arena_alloc(arena, n_blocks * sizeof(float));
    if (!prepared.f32_scales) return;

    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)w->data;
    for (size_t i = 0; i < n_blocks; i++)
        prepared.f32_scales[i] = bn_fp16_to_fp32(blocks[i].d);
    (void)bn_backend_model_register_prepared_qweight(backend, w, &prepared);
}
#else
static size_t prepared_q8_f32_scale_bytes(const BnQWeight *w) {
    (void)w;
    return 0;
}

static void prepared_q8_f32_scales(BnBackendModel *backend, const BnQWeight *w,
                                   SHArena *arena) {
    (void)backend;
    (void)w;
    (void)arena;
}
#endif

static void prepared_qweight_size_one(const BnQWeight *w,
                                      BnBackendLayoutPreparedStats *stats) {
    if (!stats) return;
    stats->q4_repack_bytes += prepared_q4_repack_bytes(w);
    stats->q8_scale_bytes += prepared_q8_f32_scale_bytes(w);
}

static void prepared_qweight_size_layer(const BnLayerWeights *lw,
                                        BnBackendLayoutPreparedStats *stats) {
    prepared_qweight_size_one(&lw->wq, stats);
    prepared_qweight_size_one(&lw->wk, stats);
    prepared_qweight_size_one(&lw->wv, stats);
    prepared_qweight_size_one(&lw->wo, stats);
    prepared_qweight_size_one(&lw->wqkv, stats);
    prepared_qweight_size_one(&lw->wz, stats);
    prepared_qweight_size_one(&lw->ssm_out, stats);
    prepared_qweight_size_one(&lw->ffn_gate, stats);
    prepared_qweight_size_one(&lw->ffn_up, stats);
    prepared_qweight_size_one(&lw->ffn_down, stats);
}

static void prepared_qweight_prepare_one(BnBackendModel *backend,
                                         const BnQWeight *w,
                                         SHArena *arena) {
    prepared_q4_repack(backend, w, arena);
    prepared_q8_f32_scales(backend, w, arena);
}

static void prepared_qweight_prepare_layer(BnBackendModel *backend,
                                           const BnLayerWeights *lw,
                                           SHArena *arena) {
    prepared_qweight_prepare_one(backend, &lw->wq, arena);
    prepared_qweight_prepare_one(backend, &lw->wk, arena);
    prepared_qweight_prepare_one(backend, &lw->wv, arena);
    prepared_qweight_prepare_one(backend, &lw->wo, arena);
    prepared_qweight_prepare_one(backend, &lw->wqkv, arena);
    prepared_qweight_prepare_one(backend, &lw->wz, arena);
    prepared_qweight_prepare_one(backend, &lw->ssm_out, arena);
    prepared_qweight_prepare_one(backend, &lw->ffn_gate, arena);
    prepared_qweight_prepare_one(backend, &lw->ffn_up, arena);
    prepared_qweight_prepare_one(backend, &lw->ffn_down, arena);
}

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

    return stats->q4_repack_bytes + stats->q8_scale_bytes;
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
}
