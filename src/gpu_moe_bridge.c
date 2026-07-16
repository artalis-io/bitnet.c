#include "gpu_moe_bridge.h"
#include "gpu_backend.h"
#include "gpu_quant_lowering_internal.h"
#include "gpu_moe_cache.h"
#include "moe.h"
#include "quant.h"
#include "transformer/gpu_internal.h"
#include <stdlib.h>
#include <string.h>

static int gpu_moe_can_matvec_split(const BnGPUBackend *gpu, int tensor_type) {
    uint32_t cap = bn_quant_format_gpu_split_cap(tensor_type);
    return gpu && cap && ((gpu->caps & cap) != 0);
}

static int gpu_moe_can_gateup_split(const BnGPUBackend *gpu,
                                    const BnMoEExpertMap *em,
                                    int split_op_code) {
    return split_op_code == BN_GPU_CODE_Q4K_MATVEC_SPLIT &&
           gpu_moe_can_matvec_split(gpu, em->gate_type) &&
           em->up_type == em->gate_type &&
           em->gate_rows == em->up_rows &&
           em->gate_cols == em->up_cols;
}

static void gpu_moe_destroy_partial(BnGPUBackend *gpu,
                                    void *gate,
                                    void *up,
                                    void *down) {
    if (!gpu || !gpu->buffer_destroy) return;
    if (gate) gpu->buffer_destroy(gpu->ctx, gate);
    if (up) gpu->buffer_destroy(gpu->ctx, up);
    if (down) gpu->buffer_destroy(gpu->ctx, down);
}

static void *gpu_moe_create_expert_buffer(BnGPUBackend *gpu,
                                          const void *data,
                                          size_t size,
                                          int type,
                                          int rows,
                                          int cols,
                                          int allow_aux_cache) {
    if (!gpu || !data || size == 0)
        return NULL;
    if (gpu->kind == BN_GPU_BACKEND_CUDA &&
        gpu->buffer_create_quant_only &&
        !allow_aux_cache &&
        getenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE") == NULL &&
        bn_quant_format_cuda_lazy_moe_aux_cache_candidate(type)) {
        return gpu->buffer_create_quant_only(
            gpu->ctx, data, size, type, rows, cols);
    }
    if (bn_quant_format_cuda_moe_prefers_quant_only(type) &&
        gpu->buffer_create_quant_only)
        return gpu->buffer_create_quant_only(
            gpu->ctx, data, size, type, rows, cols);
    return gpu->buffer_create(gpu->ctx, data, size, type, rows, cols);
}

static int gpu_moe_track_temporary(BnGPUMoETemporaryBuffers *temporaries,
                                   void *buffer) {
    if (!temporaries || !buffer)
        return 0;
    int cap = (int)(sizeof(temporaries->buffers) /
                    sizeof(temporaries->buffers[0]));
    if (temporaries->n_buffers < 0 || temporaries->n_buffers >= cap)
        return -1;
    temporaries->buffers[temporaries->n_buffers++] = buffer;
    return 0;
}

int bn_gpu_moe_bridge_get_expert(BnModel *m,
                                  BnSession *sess,
                                  const BnLayerWeights *lw,
                                  int layer,
                                  int expert_idx,
                                  BnGPUMoETemporaryBuffers *temporaries,
                                  BnGPUMoEExpertBuffers *out) {
    if (!m || !sess || !lw || !out) return -1;
    BnGPUBackend *gpu = bn_model_gpu(m);
    BnMoEState *ms = sess->moe_state;
    if (!gpu || !gpu->buffer_create || !ms) return -1;

    const BnMoEExpertMap *em = &lw->moe.expert_map;
    BnGPUMoECache *gpu_cache = (BnGPUMoECache *)bn_model_moe_io(m)->gpu_moe_cache;
    int split_op_code = bn_gpu_quant_split_op_code(em->gate_type);
    int use_split = bn_transformer_gpu_cuda_moe_gateup_split_enabled(
        gpu, gpu_moe_can_gateup_split(gpu, em, split_op_code));

    memset(out, 0, sizeof(*out));
    out->use_gateup_split = use_split;
    out->gateup_split_op_code = split_op_code;

    if (bn_gpu_moe_cache_lookup(gpu_cache, layer, expert_idx,
                                &out->gate, &out->up, &out->down))
        return 0;

    const void *gate_data = bn_moe_get_expert_proj(bn_model_moe_io(m), ms, em,
                                                   expert_idx, 0);
    const void *up_data = bn_moe_get_expert_proj(bn_model_moe_io(m), ms, em,
                                                 expert_idx, 1);
    if (!gate_data || !up_data) return -1;

    if (use_split) {
        size_t gateup_bytes = em->expert_gate_bytes + em->expert_up_bytes;
        if (gpu->buffer_create_stacked2) {
            out->gate = gpu->buffer_create_stacked2(
                gpu->ctx, gate_data, em->expert_gate_bytes,
                up_data, em->expert_up_bytes, em->gate_type,
                em->gate_rows + em->up_rows, em->gate_cols);
        } else {
            uint8_t *gateup_data = (uint8_t *)malloc(gateup_bytes);
            if (!gateup_data) return -1;
            memcpy(gateup_data, gate_data, em->expert_gate_bytes);
            memcpy(gateup_data + em->expert_gate_bytes, up_data,
                   em->expert_up_bytes);
            out->gate = gpu->buffer_create(
                gpu->ctx, gateup_data, gateup_bytes, em->gate_type,
                em->gate_rows + em->up_rows, em->gate_cols);
            free(gateup_data);
        }
        if (!out->gate) return -1;
    } else {
        out->gate = gpu_moe_create_expert_buffer(
            gpu, gate_data, em->expert_gate_bytes, em->gate_type,
            em->gate_rows, em->gate_cols, 0);
        if (!out->gate) return -1;
        out->up = gpu_moe_create_expert_buffer(
            gpu, up_data, em->expert_up_bytes, em->up_type,
            em->up_rows, em->up_cols, 0);
        if (!out->up) {
            gpu_moe_destroy_partial(gpu, out->gate, NULL, NULL);
            memset(out, 0, sizeof(*out));
            return -1;
        }
    }

    const void *down_data = bn_moe_get_expert_proj(bn_model_moe_io(m), ms, em,
                                                   expert_idx, 2);
    if (!down_data) {
        gpu_moe_destroy_partial(gpu, out->gate, out->up, NULL);
        memset(out, 0, sizeof(*out));
        return -1;
    }
    out->down = gpu_moe_create_expert_buffer(
        gpu, down_data, em->expert_down_bytes, em->down_type,
        em->down_rows, em->down_cols, 0);
    if (!out->down) {
        gpu_moe_destroy_partial(gpu, out->gate, out->up, NULL);
        memset(out, 0, sizeof(*out));
        return -1;
    }

    if (gpu_cache) {
        bn_gpu_moe_cache_insert(gpu_cache, layer, expert_idx,
                                out->gate, out->up, out->down);
    } else if (temporaries) {
        if (gpu_moe_track_temporary(temporaries, out->gate) != 0 ||
            gpu_moe_track_temporary(temporaries, out->up) != 0 ||
            gpu_moe_track_temporary(temporaries, out->down) != 0)
            return -1;
    }

    return 0;
}

int bn_gpu_moe_bridge_resolve_resources(BnGPUMoEResources *out,
                                         BnGPUMoEResolvedExpert *expert_storage,
                                         int expert_cap,
                                         BnModel *m,
                                         BnSession *sess,
                                         const BnLayerWeights *lw,
                                         int layer,
                                         BnGPUMoETemporaryBuffers *temporaries) {
    if (!out || !expert_storage || expert_cap < 0 || !m || !sess || !lw ||
        !temporaries)
        return -1;
    BnConfig *c = &m->config;
    BnMoEState *ms = sess->moe_state;
    if (!ms) return -1;

    memset(out, 0, sizeof(*out));
    out->expert_map = &lw->moe.expert_map;
    out->experts = expert_storage;
    out->moe_hidden = c->moe_intermediate_size;
    memset(temporaries, 0, sizeof(*temporaries));

    int K = c->n_experts_active;
    if (K > expert_cap) return -1;
    for (int k = 0; k < K; k++) {
        int eidx = ms->expert_indices[k];
        if (eidx < 0 || eidx >= c->n_experts) continue;
        BnGPUMoEResolvedExpert *expert = &expert_storage[out->n_experts];
        memset(expert, 0, sizeof(*expert));
        if (bn_gpu_moe_bridge_get_expert(m, sess, lw, layer, eidx,
                                         temporaries, &expert->buffers) != 0)
            return -1;
        expert->weight = ms->expert_weights[k];
        out->n_experts++;
    }
    return 0;
}

void bn_gpu_moe_bridge_release_temporaries(
    BnModel *m,
    BnGPUMoETemporaryBuffers *temporaries) {
    if (!m || !temporaries || temporaries->n_buffers <= 0)
        return;
    BnGPUBackend *gpu = bn_model_gpu(m);
    if (!gpu || !gpu->buffer_destroy)
        return;

    for (int i = 0; i < temporaries->n_buffers; i++) {
        if (temporaries->buffers[i])
            gpu->buffer_destroy(gpu->ctx, temporaries->buffers[i]);
        temporaries->buffers[i] = NULL;
    }
    temporaries->n_buffers = 0;
}

int bn_gpu_moe_bridge_preload_all(BnModel *m) {
    if (!m) return -1;
    BnGPUBackend *gpu = bn_model_gpu(m);
    BnGPUMoECache *gpu_cache =
        (BnGPUMoECache *)bn_model_moe_io(m)->gpu_moe_cache;
    if (!gpu || !gpu->buffer_create || !gpu_cache)
        return -1;

    BnMoEState temp_state;
    memset(&temp_state, 0, sizeof(temp_state));
    for (int layer = 0; layer < m->config.n_layers; layer++) {
        const BnLayerWeights *lw = &m->weights.layers[layer];
        if (!lw->moe.router_weight)
            continue;
        const BnMoEExpertMap *em = &lw->moe.expert_map;
        if (em->expert_gate_bytes > temp_state.buf_size)
            temp_state.buf_size = em->expert_gate_bytes;
        if (em->expert_up_bytes > temp_state.buf2_size)
            temp_state.buf2_size = em->expert_up_bytes;
        if (em->expert_down_bytes > temp_state.buf5_size)
            temp_state.buf5_size = em->expert_down_bytes;
    }
    if (temp_state.buf_size > 0) {
        temp_state.buf = (uint8_t *)malloc(temp_state.buf_size);
        if (!temp_state.buf)
            return -1;
    }
    if (temp_state.buf2_size > 0) {
        temp_state.buf2 = (uint8_t *)malloc(temp_state.buf2_size);
        if (!temp_state.buf2) {
            free(temp_state.buf);
            return -1;
        }
    }
    if (temp_state.buf5_size > 0) {
        temp_state.buf5 = (uint8_t *)malloc(temp_state.buf5_size);
        if (!temp_state.buf5) {
            free(temp_state.buf);
            free(temp_state.buf2);
            return -1;
        }
    }

    int loaded = 0;
    for (int layer = 0; layer < m->config.n_layers; layer++) {
        const BnLayerWeights *lw = &m->weights.layers[layer];
        if (!lw->moe.router_weight)
            continue;
        const BnMoEExpertMap *em = &lw->moe.expert_map;
        int split_op_code = bn_gpu_quant_split_op_code(em->gate_type);
        int use_split = bn_transformer_gpu_cuda_moe_gateup_split_enabled(
            gpu, gpu_moe_can_gateup_split(gpu, em, split_op_code));

        for (int expert_idx = 0; expert_idx < m->config.n_experts;
             expert_idx++) {
            const void *gate_data = bn_moe_get_expert_proj(
                bn_model_moe_io(m), &temp_state, em, expert_idx, 0);
            const void *up_data = bn_moe_get_expert_proj(
                bn_model_moe_io(m), &temp_state, em, expert_idx, 1);
            const void *down_data = bn_moe_get_expert_proj(
                bn_model_moe_io(m), &temp_state, em, expert_idx, 2);
            if (!gate_data || !up_data || !down_data) {
                free(temp_state.buf);
                free(temp_state.buf2);
                free(temp_state.buf5);
                return -1;
            }

            void *gate_gpu = NULL;
            void *up_gpu = NULL;
            void *down_gpu = NULL;
            if (use_split) {
                if (gpu->buffer_create_stacked2) {
                    gate_gpu = gpu->buffer_create_stacked2(
                        gpu->ctx, gate_data, em->expert_gate_bytes,
                        up_data, em->expert_up_bytes, em->gate_type,
                        em->gate_rows + em->up_rows, em->gate_cols);
                } else {
                    size_t gateup_bytes =
                        em->expert_gate_bytes + em->expert_up_bytes;
                    uint8_t *gateup_data = (uint8_t *)malloc(gateup_bytes);
                    if (!gateup_data) {
                        free(temp_state.buf);
                        free(temp_state.buf2);
                        free(temp_state.buf5);
                        return -1;
                    }
                    memcpy(gateup_data, gate_data, em->expert_gate_bytes);
                    memcpy(gateup_data + em->expert_gate_bytes, up_data,
                           em->expert_up_bytes);
                    gate_gpu = gpu->buffer_create(
                        gpu->ctx, gateup_data, gateup_bytes, em->gate_type,
                        em->gate_rows + em->up_rows, em->gate_cols);
                    free(gateup_data);
                }
            } else {
                gate_gpu = gpu_moe_create_expert_buffer(
                    gpu, gate_data, em->expert_gate_bytes,
                    em->gate_type, em->gate_rows, em->gate_cols, 1);
                up_gpu = gpu_moe_create_expert_buffer(
                    gpu, up_data, em->expert_up_bytes,
                    em->up_type, em->up_rows, em->up_cols, 1);
            }
            down_gpu = gpu_moe_create_expert_buffer(
                gpu, down_data, em->expert_down_bytes,
                em->down_type, em->down_rows, em->down_cols, 1);
            if (!gate_gpu || (!use_split && !up_gpu) || !down_gpu) {
                gpu_moe_destroy_partial(gpu, gate_gpu, up_gpu, down_gpu);
                free(temp_state.buf);
                free(temp_state.buf2);
                free(temp_state.buf5);
                return -1;
            }
            if (bn_gpu_moe_cache_insert(gpu_cache, layer, expert_idx,
                                        gate_gpu, up_gpu, down_gpu) != 0) {
                gpu_moe_destroy_partial(gpu, gate_gpu, up_gpu, down_gpu);
                free(temp_state.buf);
                free(temp_state.buf2);
                free(temp_state.buf5);
                return -1;
            }
            loaded++;
        }
    }
    free(temp_state.buf);
    free(temp_state.buf2);
    free(temp_state.buf5);
    return loaded;
}
