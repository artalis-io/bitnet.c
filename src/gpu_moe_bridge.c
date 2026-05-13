#include "gpu_moe_bridge.h"
#include "backend_quant.h"
#include "gpu_backend.h"
#include "gpu_moe_cache.h"
#include "moe.h"
#include <stdlib.h>
#include <string.h>

static int gpu_moe_can_matvec_split(const BnGPUBackend *gpu, int tensor_type) {
    uint32_t cap = bn_backend_quant_gpu_split_cap(tensor_type);
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

int bn_gpu_moe_bridge_get_expert(BnModel *m,
                                  BnSession *sess,
                                  const BnLayerWeights *lw,
                                  int layer,
                                  int expert_idx,
                                  void **uncached_bufs,
                                  int *n_uncached,
                                  BnGPUMoEExpertBuffers *out) {
    if (!m || !sess || !lw || !out) return -1;
    BnGPUBackend *gpu = bn_model_gpu(m);
    BnMoEState *ms = sess->moe_state;
    if (!gpu || !gpu->buffer_create || !ms) return -1;

    const BnMoEExpertMap *em = &lw->expert_map;
    BnGPUMoECache *gpu_cache = (BnGPUMoECache *)m->moe_io.gpu_moe_cache;
    int split_op_code = bn_backend_quant_gpu_split_op_code(em->gate_type);
    int use_split = gpu_moe_can_gateup_split(gpu, em, split_op_code);

    memset(out, 0, sizeof(*out));
    out->use_gateup_split = use_split;
    out->gateup_split_op_code = split_op_code;

    if (bn_gpu_moe_cache_lookup(gpu_cache, layer, expert_idx,
                                &out->gate, &out->up, &out->down))
        return 0;

    const void *gate_data = bn_moe_get_expert_proj(&m->moe_io, ms, em,
                                                   expert_idx, 0);
    const void *up_data = bn_moe_get_expert_proj(&m->moe_io, ms, em,
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
        out->gate = gpu->buffer_create(gpu->ctx, gate_data,
            em->expert_gate_bytes, em->gate_type,
            em->gate_rows, em->gate_cols);
        if (!out->gate) return -1;
        out->up = gpu->buffer_create(gpu->ctx, up_data,
            em->expert_up_bytes, em->up_type,
            em->up_rows, em->up_cols);
        if (!out->up) {
            gpu_moe_destroy_partial(gpu, out->gate, NULL, NULL);
            memset(out, 0, sizeof(*out));
            return -1;
        }
    }

    const void *down_data = bn_moe_get_expert_proj(&m->moe_io, ms, em,
                                                   expert_idx, 2);
    if (!down_data) {
        gpu_moe_destroy_partial(gpu, out->gate, out->up, NULL);
        memset(out, 0, sizeof(*out));
        return -1;
    }
    out->down = gpu->buffer_create(gpu->ctx, down_data,
        em->expert_down_bytes, em->down_type,
        em->down_rows, em->down_cols);
    if (!out->down) {
        gpu_moe_destroy_partial(gpu, out->gate, out->up, NULL);
        memset(out, 0, sizeof(*out));
        return -1;
    }

    if (gpu_cache) {
        bn_gpu_moe_cache_insert(gpu_cache, layer, expert_idx,
                                out->gate, out->up, out->down);
    } else if (uncached_bufs && n_uncached) {
        uncached_bufs[(*n_uncached)++] = out->gate;
        if (out->up) uncached_bufs[(*n_uncached)++] = out->up;
        uncached_bufs[(*n_uncached)++] = out->down;
    }

    return 0;
}
