#include "gpu_moe_bridge.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "model_internal.h"
#include "gpu_backend.h"
#include "gpu_moe_cache.h"
#include "gpu_policy.h"
#include "moe.h"
#include "moe_internal.h"
#include "transformer/gpu_internal.h"
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int gpu_moe_backend_can_upload_expert_buffers(
    const BnGPUBackend *gpu) {
    return bn_gpu_backend_can_create_buffer(gpu);
}

static void gpu_moe_destroy_partial(BnGPUBackend *gpu,
                                    void *gate,
                                    void *up,
                                    void *down) {
    bn_gpu_backend_destroy_buffer(gpu, gate);
    bn_gpu_backend_destroy_buffer(gpu, up);
    bn_gpu_backend_destroy_buffer(gpu, down);
}

static void *gpu_moe_create_expert_buffer(BnGPUBackend *gpu,
                                          const void *data,
                                          size_t size,
                                          int type,
                                          int rows,
                                          int cols,
                                          int allow_borrowed,
                                          int allow_aux_cache) {
    if (!gpu || !data || size == 0)
        return NULL;
    if (allow_borrowed &&
        bn_gpu_backend_can_create_native_matvec_borrowed_buffer(gpu)) {
        void *borrowed =
            bn_gpu_backend_create_native_matvec_borrowed_buffer(
                gpu, data, size, type, rows, cols);
        if (borrowed)
            return borrowed;
    }
    if (bn_gpu_backend_can_create_quant_only_buffer(gpu) &&
        bn_transformer_gpu_moe_quant_only_without_aux_cache(
            gpu, type, allow_aux_cache)) {
        return bn_gpu_backend_create_quant_only_buffer(
            gpu, data, size, type, rows, cols);
    }
    if (bn_gpu_policy_moe_prefers_quant_only(gpu, type) &&
        bn_gpu_backend_can_create_quant_only_buffer(gpu))
        return bn_gpu_backend_create_quant_only_buffer(
            gpu, data, size, type, rows, cols);
    return bn_gpu_backend_create_buffer(gpu, data, size, type, rows, cols);
}

static void *gpu_moe_create_gateup_split_buffer(BnGPUBackend *gpu,
                                                const void *gate_data,
                                                size_t gate_bytes,
                                                const void *up_data,
                                                size_t up_bytes,
                                                int tensor_type,
                                                int rows,
                                                int cols,
                                                int quant_only) {
    if (!gpu_moe_backend_can_upload_expert_buffers(gpu) ||
        !gate_data || !up_data ||
        gate_bytes == 0 || up_bytes == 0 || gate_bytes > SIZE_MAX - up_bytes)
        return NULL;
    quant_only = quant_only && bn_gpu_backend_can_create_quant_only_buffer(gpu);
    if (!quant_only && bn_gpu_backend_can_create_stacked2_buffer(gpu))
        return bn_gpu_backend_create_stacked2_buffer(
            gpu, gate_data, gate_bytes, up_data, up_bytes, tensor_type,
            rows, cols);

    size_t gateup_bytes = gate_bytes + up_bytes;
    uint8_t *gateup_data = (uint8_t *)malloc(gateup_bytes);
    if (!gateup_data)
        return NULL;
    memcpy(gateup_data, gate_data, gate_bytes);
    memcpy(gateup_data + gate_bytes, up_data, up_bytes);
    void *out = quant_only
        ? bn_gpu_backend_create_quant_only_buffer(gpu, gateup_data, gateup_bytes,
                                                  tensor_type, rows, cols)
        : bn_gpu_backend_create_buffer(gpu, gateup_data, gateup_bytes,
                                        tensor_type, rows, cols);
    free(gateup_data);
    return out;
}

static int gpu_moe_use_gateup_split(const BnGPUBackend *gpu,
                                    const BnMoEExpertMap *em) {
    int split_op_code = bn_transformer_gpu_matvec_split_op_code(em->gate_type);
    int use_split = bn_transformer_gpu_moe_gateup_split_enabled(
        gpu, bn_transformer_gpu_moe_gateup_split_supported(
                 gpu, em, split_op_code));
    if (bn_gpu_backend_native_matvec_borrowed_supported(
            gpu, em->expert_gate_bytes, em->gate_type,
            em->gate_rows, em->gate_cols) &&
        bn_gpu_backend_native_matvec_borrowed_supported(
            gpu, em->expert_up_bytes, em->up_type,
            em->up_rows, em->up_cols))
        use_split = 0;
    return use_split;
}

/* Bound all persistent layouts, including ones materialized lazily by prefill.
 * Mapped native layouts may have backend-specific smaller charges. */
static size_t gpu_moe_projection_charge(const BnGPUBackend *gpu,
    size_t bytes, int type, int rows, int cols, int borrowed) {
    if (borrowed && bn_gpu_backend_native_matvec_borrowed_supported(
            gpu, bytes, type, rows, cols))
        return bn_gpu_backend_buffer_cache_charge(gpu, bytes, type, rows, cols);
    size_t extra = gpu->buffer_f16_cache_extra_bytes
        ? bn_gpu_backend_f16_cache_extra_bytes(gpu, type, rows, cols) : 0;
    return extra > SIZE_MAX - bytes ? SIZE_MAX : bytes + extra;
}

size_t bn_gpu_moe_bridge_cache_entry_bytes(const BnModel *m,
                                           const BnGPUBackend *gpu) {
    if (!m || !gpu || !m->weights.layers) return 0;
    size_t maximum = 0;
    const BnMoEIO *io = bn_model_moe_io_const(m);
    for (int l = 0; l < m->config.n_layers; l++) {
        const BnLayerWeights *lw = &m->weights.layers[l];
        if (!bn_moe_policy_layer_has_router(lw)) continue;
        const BnMoEExpertMap *em = &lw->moe.expert_map;
        size_t gate = gpu_moe_projection_charge(gpu, em->expert_gate_bytes,
            em->gate_type, em->gate_rows, em->gate_cols,
            bn_moe_mmap_base_for_proj(io, em, 0) != NULL);
        size_t up = gpu_moe_projection_charge(gpu, em->expert_up_bytes,
            em->up_type, em->up_rows, em->up_cols,
            bn_moe_mmap_base_for_proj(io, em, 1) != NULL);
        if (gpu_moe_use_gateup_split(gpu, em)) {
            if (em->expert_gate_bytes > SIZE_MAX - em->expert_up_bytes ||
                em->gate_rows > INT_MAX - em->up_rows) return 0;
            gate = gpu_moe_projection_charge(gpu,
                em->expert_gate_bytes + em->expert_up_bytes, em->gate_type,
                em->gate_rows + em->up_rows, em->gate_cols, 0);
            up = 0;
        }
        size_t down = gpu_moe_projection_charge(gpu, em->expert_down_bytes,
            em->down_type, em->down_rows, em->down_cols,
            bn_moe_mmap_base_for_proj(io, em, 2) != NULL);
        if (gate > SIZE_MAX - up || gate + up > SIZE_MAX - down) return 0;
        size_t entry = gate + up + down;
        if (entry > maximum) maximum = entry;
    }
    return maximum;
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

static int gpu_moe_get_expert(BnModel *m,
                                  BnSession *sess,
                                  const BnLayerWeights *lw,
                                  int layer,
                                  int expert_idx,
                                  BnGPUMoETemporaryBuffers *temporaries,
                                  BnGPUMoEExpertBuffers *out, int admit_cache, int gathered_prefill) {
    if (!m || !sess || !lw || !out) return -1;
    BnGPUBackend *gpu = bn_model_gpu(m);
    BnMoEState *ms = sess->moe_state;
    if (!gpu_moe_backend_can_upload_expert_buffers(gpu) || !ms) return -1;

    const BnMoEExpertMap *em = &lw->moe.expert_map;
    const BnMoEIO *io = bn_model_moe_io_const(m);
    BnGPUMoECache *gpu_cache = (BnGPUMoECache *)bn_model_moe_io(m)->gpu_moe_cache;
    int split_op_code = bn_transformer_gpu_matvec_split_op_code(
        em->gate_type);
    int use_split = gpu_moe_use_gateup_split(gpu, em);

    memset(out, 0, sizeof(*out));
    out->use_gateup_split = use_split;
    out->gateup_split_op_code = split_op_code;

    int hit = admit_cache
        ? bn_gpu_moe_cache_lookup(gpu_cache, layer, expert_idx,
                                 &out->gate, &out->up, &out->down)
        : bn_gpu_moe_cache_lookup_prefill(gpu_cache, layer, expert_idx,
                                         &out->gate, &out->up, &out->down);
    if (hit) return 0;
    /* A prefill miss must have an owner without entering decode's cache.
     * Reserve tracking capacity before creating any backend handles. */
    if (!admit_cache && (!temporaries || temporaries->n_buffers < 0 ||
        temporaries->n_buffers > BN_MAX_MOE_K * 3 - 3)) return -1;

    BnMoEHostExpert host;
    if (bn_moe_host_expert_acquire(bn_model_moe_io(m), ms, em,
            layer, expert_idx, admit_cache, &host) != 0) return -1;
    const void *gate_data = host.projection[0];
    const void *up_data = host.projection[1];
    if (use_split) {
        out->gate = gpu_moe_create_gateup_split_buffer(
            gpu, gate_data, em->expert_gate_bytes, up_data,
            em->expert_up_bytes, em->gate_type, em->gate_rows + em->up_rows,
            em->gate_cols,
            !admit_cache && gathered_prefill &&
            bn_gpu_policy_moe_gathered_prefill_quant_only_enabled(
                gpu, em->gate_type, em->up_type, em->down_type));
        if (!out->gate) goto fail;
    } else {
        out->gate = gpu_moe_create_expert_buffer(
            gpu, gate_data, em->expert_gate_bytes, em->gate_type,
            em->gate_rows, em->gate_cols,
            bn_moe_mmap_base_for_proj(io, em, 0) != NULL, 0);
        if (!out->gate) goto fail;
        out->up = gpu_moe_create_expert_buffer(
            gpu, up_data, em->expert_up_bytes, em->up_type,
            em->up_rows, em->up_cols,
            bn_moe_mmap_base_for_proj(io, em, 1) != NULL, 0);
        if (!out->up) {
            gpu_moe_destroy_partial(gpu, out->gate, NULL, NULL);
            memset(out, 0, sizeof(*out));
            goto fail;
        }
    }

    const void *down_data = host.projection[2];
    out->down = gpu_moe_create_expert_buffer(
        gpu, down_data, em->expert_down_bytes, em->down_type,
        em->down_rows, em->down_cols,
        bn_moe_mmap_base_for_proj(io, em, 2) != NULL, 0);
    if (!out->down) {
        gpu_moe_destroy_partial(gpu, out->gate, out->up, NULL);
        memset(out, 0, sizeof(*out));
        goto fail;
    }

    bn_moe_host_expert_release(&host);
    int cached = gpu_cache && admit_cache &&
        bn_gpu_moe_cache_insert(gpu_cache, layer, expert_idx,
                                out->gate, out->up, out->down) == 0;
    if (!cached && temporaries) {
        if (temporaries->n_buffers < 0 || temporaries->n_buffers > BN_MAX_MOE_K * 3 - 3) {
            gpu_moe_destroy_partial(gpu, out->gate, out->up, out->down);
            memset(out, 0, sizeof(*out));
            return -1;
        }
        if (gpu_moe_track_temporary(temporaries, out->gate) != 0 ||
            gpu_moe_track_temporary(temporaries, out->up) != 0 ||
            gpu_moe_track_temporary(temporaries, out->down) != 0)
            return -1;
    } else if (!cached && gpu_cache && admit_cache) {
        gpu_moe_destroy_partial(gpu, out->gate, out->up, out->down);
        memset(out, 0, sizeof(*out));
        return -1;
    }

    return 0;

fail:
    bn_moe_host_expert_release(&host);
    return -1;
}

int bn_gpu_moe_bridge_get_expert(BnModel *m, BnSession *sess,
    const BnLayerWeights *lw, int layer, int expert,
    BnGPUMoETemporaryBuffers *temporaries, BnGPUMoEExpertBuffers *out) {
    return gpu_moe_get_expert(m, sess, lw, layer, expert, temporaries, out,
                              !(sess && sess->moe_cache_prefill), 0);
}

int bn_gpu_moe_bridge_get_prefill_expert(BnModel *m, BnSession *sess,
    const BnLayerWeights *lw, int layer, int expert,
    BnGPUMoETemporaryBuffers *temporaries, BnGPUMoEExpertBuffers *out) {
    /* Gathered K-quant consumers read raw rows. Ordinary tokenwise forwards
     * retain their packed layouts even while a prompt cache scope is active. */
    return gpu_moe_get_expert(m, sess, lw, layer, expert, temporaries, out, 0, 1);
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
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&m->config);
    BnMoEState *ms = sess->moe_state;
    if (!ms) return -1;

    memset(out, 0, sizeof(*out));
    out->expert_map = &lw->moe.expert_map;
    out->experts = expert_storage;
    out->moe_hidden = route_policy.expert_hidden_dim;
    memset(temporaries, 0, sizeof(*temporaries));

    int K = route_policy.active_experts;
    if (K > expert_cap) return -1;
    for (int k = 0; k < K; k++) {
        int eidx = ms->expert_indices[k];
        if (eidx < 0 || eidx >= route_policy.total_experts) continue;
        BnGPUMoEResolvedExpert *expert = &expert_storage[out->n_experts];
        memset(expert, 0, sizeof(*expert));
        if (bn_gpu_moe_bridge_get_expert(m, sess, lw, layer, eidx,
                                         temporaries, &expert->buffers) != 0)
            return -1;
        expert->weight = ms->expert_weights[k];
        expert->output_scale = bn_moe_expert_weight_scale(lw, eidx);
        expert->has_output_scale = lw->moe.expert_down_scale != NULL;
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

    for (int i = 0; i < temporaries->n_buffers; i++) {
        bn_gpu_backend_destroy_buffer(gpu, temporaries->buffers[i]);
        temporaries->buffers[i] = NULL;
    }
    temporaries->n_buffers = 0;
}

int bn_gpu_moe_bridge_preload_all(BnModel *m) {
    if (!m) return -1;
    BnGPUBackend *gpu = bn_model_gpu(m);
    BnGPUMoECache *gpu_cache =
        (BnGPUMoECache *)bn_model_moe_io(m)->gpu_moe_cache;
    if (!gpu_moe_backend_can_upload_expert_buffers(gpu) || !gpu_cache)
        return -1;
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&m->config);
    int allow_aux_cache =
        bn_transformer_gpu_moe_lazy_aux_cache_enabled(gpu);

    BnMoEState temp_state;
    memset(&temp_state, 0, sizeof(temp_state));
    for (int layer = 0; layer < m->config.n_layers; layer++) {
        const BnLayerWeights *lw = &m->weights.layers[layer];
        BnTransformerGPULayerKindPolicy layer_kind =
            bn_transformer_gpu_layer_kind_policy(lw);
        if (!layer_kind.uses_moe)
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
        BnTransformerGPULayerKindPolicy layer_kind =
            bn_transformer_gpu_layer_kind_policy(lw);
        if (!layer_kind.uses_moe)
            continue;
        const BnMoEExpertMap *em = &lw->moe.expert_map;
        const BnMoEIO *io = bn_model_moe_io_const(m);
        int use_split = gpu_moe_use_gateup_split(gpu, em);

        for (int expert_idx = 0; expert_idx < route_policy.total_experts;
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
                gate_gpu = gpu_moe_create_gateup_split_buffer(
                    gpu, gate_data, em->expert_gate_bytes, up_data,
                    em->expert_up_bytes, em->gate_type,
                    em->gate_rows + em->up_rows, em->gate_cols, 0);
            } else {
                gate_gpu = gpu_moe_create_expert_buffer(
                    gpu, gate_data, em->expert_gate_bytes,
                    em->gate_type, em->gate_rows, em->gate_cols,
                    bn_moe_mmap_base_for_proj(io, em, 0) != NULL,
                    allow_aux_cache);
                up_gpu = gpu_moe_create_expert_buffer(
                    gpu, up_data, em->expert_up_bytes,
                    em->up_type, em->up_rows, em->up_cols,
                    bn_moe_mmap_base_for_proj(io, em, 1) != NULL,
                    allow_aux_cache);
            }
            down_gpu = gpu_moe_create_expert_buffer(
                gpu, down_data, em->expert_down_bytes,
                em->down_type, em->down_rows, em->down_cols,
                bn_moe_mmap_base_for_proj(io, em, 2) != NULL,
                allow_aux_cache);
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

/* Native placement uses the same host LRU and device reservation machinery
 * as ordinary experts. Only the immutable checkpoint bundle layout differs. */
typedef struct {
    void *weights[3];
    BnTrellisLayout layouts[3];
} TrellisTemporary;
typedef struct {
    const BnModel *model;
    const BnMoEExpertReader *reader;
    BnGPUBackend *gpu;
    void *ram, *staging, *upload;
    void *projection[3];
    BnGPUMoECache *decode, *temporary;
    TrellisTemporary *reusable;
    BnTrellisLayout (*decode_layouts)[3];
    BnGPUMoETrellisStats stats;
    int ram_admit_misses;
} TrellisStore;

static int trellis_layout_matches(const BnTrellisLayout *a,const BnTrellisLayout *b) {
    return a->format==b->format&&a->k==b->k&&a->n==b->n&&a->marker==b->marker&&
        a->packed_bytes==b->packed_bytes&&a->pre_offset==b->pre_offset&&
        a->post_offset==b->post_offset&&a->bundle_bytes==b->bundle_bytes;
}
static void trellis_temporary_clear(TrellisStore *s,TrellisTemporary *slot) {
    for(int p=0;p<3;p++)if(slot->weights[p]) {
        bn_gpu_backend_destroy_buffer(s->gpu,slot->weights[p]);
        s->stats.temporary_allocated_payload-=slot->layouts[p].bundle_bytes;
    }
    memset(slot,0,sizeof(*slot));
}
static int trellis_temporary_matches(const TrellisTemporary *slot,
    const BnTrellisLayout *const layouts[3]) {
    for(int p=0;p<3;p++)if(slot->weights[p]) {
        const BnTrellisLayout *a=&slot->layouts[p],*b=layouts[p];
        if(!trellis_layout_matches(a,b))return 0;
    }
    return 1;
}
static void trellis_store_free(TrellisStore *s) {
    if (!s) return;
    if(s->reusable)for(int i=0;i<s->temporary->n_slots;i++)
        trellis_temporary_clear(s,&s->reusable[i]);
    free(s->reusable);
    free(s->decode_layouts);
    bn_gpu_moe_cache_free(s->temporary);
    bn_gpu_moe_cache_free(s->decode);
    bn_moe_cache_free(s->ram);
    if(s->upload)s->gpu->trellis_upload_destroy(s->gpu->ctx,s->upload);
    else free(s->staging);
    free(s);
}

int bn_gpu_moe_trellis_store_init(BnGPUMoETrellisStore *store,
    const BnModel *model, BnGPUBackend *gpu, size_t ram_budget,
    size_t decode_budget, size_t temporary_budget, size_t staging_budget) {
    return bn_gpu_moe_trellis_store_init_pinned(store,model,gpu,ram_budget,
        decode_budget,temporary_budget,staging_budget,0);
}
int bn_gpu_moe_trellis_store_init_pinned(BnGPUMoETrellisStore *store,
    const BnModel *model,BnGPUBackend *gpu,size_t ram_budget,
    size_t decode_budget,size_t temporary_budget,size_t staging_budget,
    size_t pinned_budget) {
    return bn_gpu_moe_trellis_store_init_policy(store,model,gpu,ram_budget,
        decode_budget,temporary_budget,staging_budget,pinned_budget,NULL);
}
int bn_gpu_moe_trellis_store_init_policy(BnGPUMoETrellisStore *store,
    const BnModel *model,BnGPUBackend *gpu,size_t ram_budget,
    size_t decode_budget,size_t temporary_budget,size_t staging_budget,
    size_t pinned_budget,const BnGPUMoETrellisPolicy *policy) {
    int ram_admit=policy?policy->ram_admit_misses:0;
    if((ram_admit!=0&&ram_admit!=1)||(ram_admit&&!ram_budget))return -1;
    BnMoECachePagePolicy pages=policy?policy->ram_pages:BN_MOE_CACHE_PAGES_HUGE_HINT;
    if((pages!=BN_MOE_CACHE_PAGES_HUGE_HINT&&pages!=BN_MOE_CACHE_PAGES_SYSTEM)||
       (pages==BN_MOE_CACHE_PAGES_SYSTEM&&!ram_budget))return -1;
    const BnMoEExpertReader *reader = bn_model_trellis_expert_reader(model);
    if (!store || store->state || !reader || !gpu ||
        !gpu->trellis_buffer_create || !gpu->buffer_destroy ||
        reader->n_layers <= 0 || reader->n_experts <= 0 ||
        reader->n_layers > INT_MAX / reader->n_experts) return -1;
    size_t entry = 0;
    for (int p = 0; p < 3; p++) {
        if (!reader->bytes[p] || reader->bytes[p] > SIZE_MAX - entry) return -1;
        entry += reader->bytes[p];
    }
    if (staging_budget < entry || (pinned_budget&&(pinned_budget<entry||
         !gpu->trellis_upload_create||!gpu->trellis_upload_batch||!gpu->trellis_upload_destroy)) ||
        (!decode_budget && !temporary_budget) ||
        (ram_budget && ram_budget < entry) ||
        (decode_budget && decode_budget < entry) ||
        (temporary_budget && temporary_budget < entry)) return -1;
    TrellisStore *s = calloc(1, sizeof(*s));
    if (!s) return -1;
    s->model = model; s->reader = reader; s->gpu = gpu;s->ram_admit_misses=ram_admit;
    s->stats.ram_pages=pages;
    int entries = reader->n_layers * reader->n_experts;
    if (ram_budget) {
        s->ram = bn_moe_cache_create_reader_policy(ram_budget,reader,pages);
        if (!s->ram) goto fail;
    }
    if (decode_budget) {
        s->decode = bn_gpu_moe_cache_create(decode_budget, entry, entries, gpu);
        if (!s->decode) goto fail;
        if(gpu->trellis_buffer_write) {
            s->decode_layouts=calloc((size_t)s->decode->n_slots,sizeof(*s->decode_layouts));
            if(!s->decode_layouts)goto fail;
        }
    }
    if (temporary_budget) {
        s->temporary = bn_gpu_moe_cache_create(temporary_budget, entry, entries, gpu);
        if (!s->temporary) goto fail;
        if(gpu->trellis_buffer_write) {
            s->reusable=calloc((size_t)s->temporary->n_slots,sizeof(*s->reusable));
            if(!s->reusable)goto fail;
        }
    }
    if(pinned_budget) {
        void *host=NULL;
        s->upload=gpu->trellis_upload_create(gpu->ctx,entry,&host);
        if(!s->upload||!host)goto fail;
        s->staging=host;s->stats.pinned_staging_payload=entry;
    } else s->staging = malloc(entry);
    if (!s->staging) goto fail;
    size_t offset = 0;
    for (int p = 0; p < 3; p++) {
        s->projection[p] = (unsigned char *)s->staging + offset;
        offset += reader->bytes[p];
    }
    s->stats.staging_payload = entry;
    store->state = s;
    return 0;
fail:
    trellis_store_free(s);
    return -1;
}

int bn_gpu_moe_trellis_acquire(BnGPUMoETrellisStore *store, int layer,
    int expert, int admit, BnGPUMoETrellisLease *out) {
    TrellisStore *s = store ? store->state : NULL;
    if (!s || !out || out->owner || out->cached.cache || out->temporary.cache ||
        out->buffers.gate || out->buffers.up || out->buffers.down ||
        (admit != 0 && admit != 1) ||
        bn_model_trellis_expert_reader(s->model) != s->reader ||
        layer < 0 || layer >= s->reader->n_layers ||
        expert < 0 || expert >= s->reader->n_experts ||
        s->stats.live_leases == SIZE_MAX) return -1;
    BnGPUMoETrellisLease lease = {0};
    if (s->decode) {
        int rc = bn_gpu_moe_cache_acquire(s->decode, layer, expert,
                                         admit, &lease.cached);
        if (rc < 0) return -1;
        if (rc) {
            lease.buffers.gate = lease.cached.gate;
            lease.buffers.up = lease.cached.up;
            lease.buffers.down = lease.cached.down;
            lease.owner = s;
            s->stats.live_leases++;
            *out = lease;
            return 0;
        }
    }
    BnGPUMoECache *cache = admit ? s->decode : s->temporary;
    if (!cache) return BN_GPU_MOE_CACHE_BUSY;
    BnGPUMoEReservation pending = {0};
    int rc=admit&&s->decode_layouts?
        bn_gpu_moe_cache_reserve_reuse(cache,layer,expert,&pending):
        bn_gpu_moe_cache_reserve(cache,layer,expert,&pending);
    if (rc) return rc;
    BnMoEHostExpert host = {0};
    void *weights[3] = {0};
    int retained=0;
    TrellisTemporary *reusable=!admit&&s->reusable?&s->reusable[pending.slot]:NULL;
    const BnTrellisLayout *layouts[3];
    if(admit&&s->decode_layouts) {
        retained=bn_gpu_moe_cache_reserved_buffers(&pending,&weights[0],&weights[1],&weights[2]);
        if(retained<0)goto fail;
    }
    if (bn_moe_host_expert_acquire_reader(s->reader, s->ram, &s->stats.host,
        layer, expert, admit||s->ram_admit_misses, s->projection, s->reader->bytes, &host)) goto fail;
    for (int p = 0; p < 3; p++) {
        layouts[p] = bn_trellis_source_layout(
            bn_model_trellis_expert_source(s->model, layer, expert, p));
        if (!layouts[p] || layouts[p]->bundle_bytes != s->reader->bytes[p]) goto fail;
    }
    if(reusable&&!trellis_temporary_matches(reusable,layouts))
        trellis_temporary_clear(s,reusable);
    if(retained)for(int p=0;p<3;p++)
        if(!trellis_layout_matches(&s->decode_layouts[pending.slot][p],layouts[p])) {
            if(bn_gpu_moe_cache_discard_reserved_buffers(&pending))goto fail;
            retained=0;memset(weights,0,sizeof(weights));break;
        }
    if(s->upload) {
        BnGPUTrellisUploadItem items[3];int created[3];size_t offset=0;
        for(int p=0;p<3;p++) {
            const BnTrellisLayout *layout=layouts[p];
            void *buffer=retained?weights[p]:reusable?reusable->weights[p]:NULL;
            items[p]=(BnGPUTrellisUploadItem){buffer,offset,layout->bundle_bytes,
                layout->format,layout->k,layout->n,layout->marker,
                host.projection[p]!=s->projection[p]?host.projection[p]:NULL};
            created[p]=buffer==NULL;offset+=layout->bundle_bytes;
        }
        double start=bn_platform_time_ms();
        int status=s->gpu->trellis_upload_batch(s->gpu->ctx,s->upload,items,3);
        s->stats.upload_time_ms+=bn_platform_time_ms()-start;
        if(status)goto fail;
        for(int p=0;p<3;p++) {
            weights[p]=items[p].buffer;
            s->stats.buffer_create_count+=created[p];s->stats.buffer_reuse_count+=!created[p];
            s->stats.upload_count++;s->stats.upload_bytes+=layouts[p]->bundle_bytes;
            if(reusable) {
                reusable->weights[p]=weights[p];reusable->layouts[p]=*layouts[p];
                if(created[p])s->stats.temporary_allocated_payload+=layouts[p]->bundle_bytes;
            }
        }
        if(s->stats.temporary_allocated_payload>s->stats.temporary_peak_payload)
            s->stats.temporary_peak_payload=s->stats.temporary_allocated_payload;
        s->stats.upload_batches++;
    } else for (int p = 0; p < 3; p++) {
        const BnTrellisLayout *layout=layouts[p];
        double start = bn_platform_time_ms();
        int status=0;
        if(retained||(reusable&&reusable->weights[p])) {
            if(!retained)weights[p]=reusable->weights[p];
            status=s->gpu->trellis_buffer_write(s->gpu->ctx,weights[p],
                host.projection[p],layout->bundle_bytes,layout->format,
                layout->k,layout->n,layout->marker);
            if(!status)s->stats.buffer_reuse_count++;
        } else {
            weights[p] = s->gpu->trellis_buffer_create(s->gpu->ctx,
                host.projection[p], layout->bundle_bytes, layout->format,
                layout->k, layout->n, layout->marker);
            if(weights[p]) {
                s->stats.buffer_create_count++;
                if(reusable) {
                    reusable->weights[p]=weights[p];reusable->layouts[p]=*layout;
                    s->stats.temporary_allocated_payload+=layout->bundle_bytes;
                    if(s->stats.temporary_allocated_payload>s->stats.temporary_peak_payload)
                        s->stats.temporary_peak_payload=s->stats.temporary_allocated_payload;
                }
            }
        }
        s->stats.upload_time_ms += bn_platform_time_ms() - start;
        if (!weights[p]||status) goto fail;
        s->stats.upload_count++;
        s->stats.upload_bytes += layout->bundle_bytes;
    }
    /* The backend upload contract completes host consumption before return. */
    bn_moe_host_expert_release(&host);
    if (admit) {
        int slot=pending.slot;
        if (bn_gpu_moe_cache_commit(&pending, weights[0], weights[1],
                                   weights[2], &lease.cached)) goto fail;
        if(s->decode_layouts)for(int p=0;p<3;p++)s->decode_layouts[slot][p]=*layouts[p];
    } else {
        /* Leave a charged pending reservation until the temporary is released;
         * a second consumer can share this lease but cannot publish this key. */
        lease.temporary = pending;
    }
    lease.buffers.gate = weights[0];
    lease.buffers.up = weights[1];
    lease.buffers.down = weights[2];
    lease.owner = s;
    s->stats.live_leases++;
    *out = lease;
    return 0;
fail:
    bn_moe_host_expert_release(&host);
    if(reusable)trellis_temporary_clear(s,reusable);
    else if(!retained)gpu_moe_destroy_partial(s->gpu, weights[0], weights[1], weights[2]);
    bn_gpu_moe_cache_abort(&pending);
    return -1;
}

int bn_gpu_moe_trellis_release(BnGPUMoETrellisStore *store,
    BnGPUMoETrellisLease *lease) {
    TrellisStore *s = store ? store->state : NULL;
    if (!s || !lease) return -1;
    if (!lease->owner) {
        return lease->cached.cache || lease->temporary.cache ||
            lease->buffers.gate || lease->buffers.up || lease->buffers.down ? -1 : 0;
    }
    if (lease->owner != s || !s->stats.live_leases) return -1;
    if (lease->cached.cache) {
        if (lease->cached.cache != s->decode || lease->temporary.cache ||
            bn_gpu_moe_cache_release(&lease->cached)) return -1;
    } else {
        BnGPUMoEReservation *r = &lease->temporary;
        if (r->cache != s->temporary || !r->cache || r->slot < 0 ||
            r->slot >= r->cache->n_slots) return -1;
        const BnGPUMoECacheEntry *entry = &r->cache->entries[r->slot];
        if (!entry->pending || entry->published || entry->pins ||
            entry->generation != r->generation) return -1;
        if(s->reusable) {
            const TrellisTemporary *slot=&s->reusable[r->slot];
            if(slot->weights[0]!=lease->buffers.gate||
               slot->weights[1]!=lease->buffers.up||
               slot->weights[2]!=lease->buffers.down)return -1;
        } else gpu_moe_destroy_partial(s->gpu, lease->buffers.gate,
                                      lease->buffers.up, lease->buffers.down);
        if (bn_gpu_moe_cache_abort(&lease->temporary)) return -1;
    }
    s->stats.live_leases--;
    memset(lease, 0, sizeof(*lease));
    return 0;
}

int bn_gpu_moe_trellis_stats(const BnGPUMoETrellisStore *store,
    BnGPUMoETrellisStats *out) {
    const TrellisStore *s = store ? store->state : NULL;
    if (!s || !out) return -1;
    BnGPUMoETrellisStats stats = s->stats;
    if ((s->decode && bn_gpu_moe_cache_stats(s->decode, &stats.decode)) ||
        (s->temporary && bn_gpu_moe_cache_stats(s->temporary, &stats.temporary)))
        return -1;
    *out = stats;
    return 0;
}

int bn_gpu_moe_trellis_store_close(BnGPUMoETrellisStore *store) {
    if (!store) return -1;
    TrellisStore *s = store->state;
    if (!s) return 0;
    if (s->stats.live_leases) return BN_GPU_MOE_CACHE_BUSY;
    trellis_store_free(s);
    store->state = NULL;
    return 0;
}

static const char native_trellis_model_owner=0;
static void trellis_model_store_release(void *context) {
    BnGPUMoETrellisStore *store=context;
    if(bn_gpu_moe_trellis_store_close(store)) {
        fprintf(stderr,"Native expert store teardown requires released leases\n");
        return;
    }
    free(store);
}
BnGPUMoETrellisStore *bn_gpu_moe_trellis_model_store(const BnModel *model) {
    BnGPUMoETrellisStore *store=bn_backend_layout_native_execution(bn_model_backend(model),
        &native_trellis_model_owner);
    const TrellisStore *state=store?store->state:NULL;
    return state&&state->model==model&&
        state->reader==bn_model_trellis_expert_reader(model)?store:NULL;
}
int bn_gpu_moe_trellis_prepare_model(BnModel *model,size_t ram_budget,
    size_t decode_budget,size_t temporary_budget,size_t staging_budget) {
    return bn_gpu_moe_trellis_prepare_model_pinned(model,ram_budget,decode_budget,
        temporary_budget,staging_budget,0);
}
int bn_gpu_moe_trellis_prepare_model_pinned(BnModel *model,size_t ram_budget,
    size_t decode_budget,size_t temporary_budget,size_t staging_budget,
    size_t pinned_budget) {
    return bn_gpu_moe_trellis_prepare_model_policy(model,ram_budget,decode_budget,
        temporary_budget,staging_budget,pinned_budget,NULL);
}
int bn_gpu_moe_trellis_prepare_model_policy(BnModel *model,size_t ram_budget,
    size_t decode_budget,size_t temporary_budget,size_t staging_budget,
    size_t pinned_budget,const BnGPUMoETrellisPolicy *policy) {
    BnBackendModel *backend=bn_model_backend(model);
    BnGPUBackend *gpu=bn_backend_model_gpu(backend);
    if(!gpu||!bn_backend_layout_native_core_stats(backend)||
        bn_gpu_moe_trellis_model_store(model))return -1;
    BnGPUMoETrellisStore *store=calloc(1,sizeof(*store));
    if(!store)return -1;
    if(bn_gpu_moe_trellis_store_init_policy(store,model,gpu,ram_budget,decode_budget,
        temporary_budget,staging_budget,pinned_budget,policy)||
       bn_backend_layout_attach_native_execution(backend,&native_trellis_model_owner,
                                                 store,trellis_model_store_release)) {
        bn_gpu_moe_trellis_store_close(store);
        free(store);
        return -1;
    }
    return 0;
}
