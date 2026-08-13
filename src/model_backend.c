#include "model_internal.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdlib.h>

BnBackendModel *bn_model_backend(const BnModel *model) {
    return (model && model->backend_state) ? model->backend_state->backend : NULL;
}

int bn_model_ensure_backend(BnModel *model) {
    if (!model) return -1;
    if (!model->backend_state) {
        model->backend_state = calloc(1, sizeof(*model->backend_state));
        if (!model->backend_state) return -1;
    }
    if (!model->backend_state->backend)
        model->backend_state->backend = bn_backend_model_create();
    return model->backend_state->backend ? 0 : -1;
}

size_t bn_model_backend_prepared_size(const BnConfig *config,
                                      const BnWeights *weights) {
    return bn_backend_layout_prepared_qweights_size(config, weights, NULL);
}

static void log_prepared_bytes(const char *message, size_t bytes) {
    if (!bytes) return;
    char mb[16];
    snprintf(mb, sizeof(mb), "%.0f", (double)bytes / (1024 * 1024));
    SH_LOG_INFO(message, "MB", mb);
}

void bn_model_backend_prepare(BnModel *model, SHArena *arena) {
    BnBackendLayoutPreparedStats stats = {0};
    bn_backend_layout_prepare_qweights(bn_model_backend(model), &model->config,
                                       &model->weights, arena, &stats);
    log_prepared_bytes("Q4_0 weights repacked", stats.lowbit_repack_bytes);
    log_prepared_bytes("Q8_0 weights repacked", stats.q8_repack_bytes);
    log_prepared_bytes("Q4_K scales prepared", stats.kquant_scale_table_bytes);
    log_prepared_bytes("Q6_K weights expanded", stats.expanded_kquant_weight_bytes);
    log_prepared_bytes("Q8_0 FP32 scales ready", stats.f32_scale_table_bytes);
}

void bn_model_backend_free(BnModel *model) {
    if (!model || !model->backend_state) return;
    bn_backend_model_free(model->backend_state->backend);
}
