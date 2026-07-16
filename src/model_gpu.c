#include "model.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "gpu_backend.h"
#include "gpu_policy.h"
#include "moe_internal.h"
#include "quant.h"
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

BnGPUBackend *bn_model_gpu(const BnModel *model) {
    return model ? bn_backend_model_gpu(bn_model_backend(model)) : NULL;
}

void bn_model_set_gpu_disabled(BnModel *model, int disabled) {
    if (!model) return;
    bn_backend_model_set_gpu_disabled(bn_model_backend(model), disabled);
}

static void *upload_qweight_logits(BnGPUBackend *gpu, BnQWeight *w) {
    if (!w->data) return NULL;
    size_t sz = bn_qweight_data_size(w);
    if (sz == 0) return NULL;
    if (bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(gpu, w->type)) {
        return gpu->buffer_create_q6_f32_cache(
            gpu->ctx, w->data, sz, w->type, w->rows, w->cols);
    }
    if (bn_gpu_policy_cuda_logits_f16_cache_enabled(gpu)) {
        return gpu->buffer_create_f16_cache(
            gpu->ctx, w->data, sz, w->type, w->rows, w->cols);
    }
    return gpu->buffer_create(gpu->ctx, w->data, sz, w->type, w->rows, w->cols);
}

static void *upload_qweight_mode(BnGPUBackend *gpu, BnQWeight *w,
                                 int quant_only) {
    if (!w->data) return NULL;
    size_t sz = bn_qweight_data_size(w);
    if (sz == 0) return NULL;
    if (quant_only && gpu->buffer_create_quant_only)
        return gpu->buffer_create_quant_only(gpu->ctx, w->data, sz, w->type,
                                             w->rows, w->cols);
    return gpu->buffer_create(gpu->ctx, w->data, sz, w->type, w->rows,
                              w->cols);
}

static int upload_qweight_owned_mode(BnModel *model, BnBackendModel *backend,
                                     BnGPUBackend *gpu, BnQWeight *w,
                                     int quant_only) {
    (void)model;
    void *handle = upload_qweight_mode(gpu, w, quant_only);
    if (!w->data) return 0;
    if (!handle) return -1;
    if (bn_backend_model_register_qweight(backend, w, handle) != 0) {
        gpu->buffer_destroy(gpu->ctx, handle);
        return -1;
    }
    return 0;
}

static int register_gpu_handle(BnModel *model, int layer,
                               BnBackendHandleRole role, void *handle) {
    if (!handle) return 0;
    return bn_backend_model_register_handle(bn_model_backend(model), layer, role, handle);
}

static void *upload_f32_buf(BnGPUBackend *gpu, const float *data, int n_elems) {
    if (!data || n_elems <= 0) return NULL;
    return gpu->buffer_create(gpu->ctx, data, (size_t)n_elems * sizeof(float),
                              -1, n_elems, 1);
}

static void *upload_moe_router_diff2(BnGPUBackend *gpu,
                                     const float *router_weight,
                                     int dim) {
    if (!gpu || !router_weight || dim <= 0) return NULL;
    float *diff = (float *)malloc((size_t)dim * sizeof(float));
    if (!diff) return NULL;
    const float *r0 = router_weight;
    const float *r1 = router_weight + dim;
    for (int i = 0; i < dim; i++)
        diff[i] = r0[i] - r1[i];
    void *handle = gpu->buffer_create(gpu->ctx, diff,
                                      (size_t)dim * sizeof(float),
                                      bn_quant_format_gpu_float_buffer_type(),
                                      1, dim);
    free(diff);
    return handle;
}

static void *upload_moe_all_proj(BnModel *model,
                                 BnGPUBackend *gpu,
                                 const BnMoEExpertMap *em,
                                 int proj,
                                 int n_experts,
                                 int q8_f16_cache) {
    if (!model || !gpu || !em || n_experts <= 0)
        return NULL;
    size_t offset = 0;
    size_t expert_bytes = 0;
    if (bn_moe_proj_info(em, 0, proj, &offset, &expert_bytes) != 0 ||
        expert_bytes == 0)
        return NULL;
    size_t stride = 0;
    int type = 0;
    int rows = 0;
    int cols = 0;
    switch (proj) {
        case 0:
            stride = em->gate_stride ? em->gate_stride : em->expert_gate_bytes;
            type = em->gate_type;
            rows = em->gate_rows;
            cols = em->gate_cols;
            break;
        case 1:
            stride = em->up_stride ? em->up_stride : em->expert_up_bytes;
            type = em->up_type;
            rows = em->up_rows;
            cols = em->up_cols;
            break;
        case 2:
            stride = em->down_stride ? em->down_stride : em->expert_down_bytes;
            type = em->down_type;
            rows = em->down_rows;
            cols = em->down_cols;
            break;
        default:
            return NULL;
    }
    const uint8_t *base = bn_moe_mmap_base_for_proj(
        bn_model_moe_io(model), em, proj);
    if (!base)
        return NULL;
    size_t total_bytes = 0;
    if (checked_mul_size(expert_bytes, (size_t)n_experts, &total_bytes) != 0)
        return NULL;
    if ((size_t)n_experts > (size_t)INT_MAX / (size_t)rows)
        return NULL;
    int force_f16_cache =
        bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
            gpu, type, q8_f16_cache);
    int prefer_q6_f32_cache =
        proj == 2 &&
        bn_quant_format_cuda_moe_down_q6_f32_cache_supported(type) &&
        bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(cols) &&
        !force_f16_cache &&
        bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(gpu);
    int prefer_q4_f32_cache =
        proj == 2 &&
        !force_f16_cache &&
        bn_gpu_policy_cuda_moe_down_q4_f32_cache_enabled(gpu, type);
    int force_full_buffer =
        force_f16_cache ||
        prefer_q6_f32_cache ||
        prefer_q4_f32_cache ||
        (proj == 2 &&
         bn_quant_format_cuda_moe_down_q6_f32_cache_supported(type) &&
         bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced());
    void *(*create_buffer)(void *, const void *, size_t, int, int, int) =
        force_f16_cache
            ? gpu->buffer_create_f16_cache :
        (prefer_q6_f32_cache || prefer_q4_f32_cache)
            ? gpu->buffer_create_q6_f32_cache :
        ((!force_full_buffer ||
          bn_quant_format_cuda_moe_quant_only_after_cache(type,
                                                           q8_f16_cache)) &&
         gpu->buffer_create_quant_only)
            ? gpu->buffer_create_quant_only
            : gpu->buffer_create;
    if (stride != expert_bytes) {
        uint8_t *packed = (uint8_t *)malloc(total_bytes);
        if (!packed)
            return NULL;
        for (int e = 0; e < n_experts; e++) {
            memcpy(packed + (size_t)e * expert_bytes,
                   base + offset + (size_t)e * stride,
                   expert_bytes);
        }
        void *handle = create_buffer(gpu->ctx, packed, total_bytes,
                                     type, rows * n_experts, cols);
        free(packed);
        return handle;
    }
    return create_buffer(gpu->ctx, base + offset, total_bytes,
                         type, rows * n_experts, cols);
}

static int can_use_cuda_moe_routed_ffn(const BnConfig *c,
                                       const BnLayerWeights *lw) {
    if (!c || !lw || !lw->moe.router_weight)
        return 0;
    const BnMoEExpertMap *em = &lw->moe.expert_map;
    int is_q4 = bn_quant_format_supports_moe_q4_down_route(
        em->gate_type, em->up_type, em->down_type, 1);
    int is_q8 = bn_quant_format_supports_moe_q8_route(
        em->gate_type, em->up_type, em->down_type);
    return (is_q4 || is_q8) &&
           em->gate_rows == c->moe_intermediate_size &&
           em->up_rows == c->moe_intermediate_size &&
           em->gate_cols == c->dim &&
           em->up_cols == c->dim &&
           em->down_rows == c->dim &&
           em->down_cols == c->moe_intermediate_size;
}

static int can_use_cuda_moe_routed_ffn_model(const BnConfig *c,
                                             const BnWeights *w) {
    if (!c || !w)
        return 0;
    int moe_layers = 0;
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        if (!lw->moe.router_weight)
            continue;
        moe_layers++;
        if (!can_use_cuda_moe_routed_ffn(c, lw))
            return 0;
    }
    return moe_layers > 0;
}

static size_t env_mb_or_default(const char *name, size_t def) {
    const char *s = getenv(name);
    if (!s || !*s)
        return def;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0')
        return def;
    return (size_t)v;
}

static size_t qweight_pair_upload_bytes(const BnQWeight *a,
                                        const BnQWeight *b);
static size_t qweight_triple_upload_bytes(const BnQWeight *a,
                                          const BnQWeight *b,
                                          const BnQWeight *c);

static int optional_layout_fits_memory(BnGPUBackend *gpu, size_t bytes,
                                       int layer, const char *name) {
    if (bytes == 0)
        return 0;
    if (!gpu || !gpu->memory_info)
        return 1;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (gpu->memory_info(gpu->ctx, &free_bytes, &total_bytes) != 0)
        return 1;
    size_t reserve_mb = env_mb_or_default("BN_CUDA_LAYOUT_RESERVE_MB", 512);
    size_t reserve = reserve_mb > SIZE_MAX / (1024u * 1024u)
        ? SIZE_MAX
        : reserve_mb * 1024u * 1024u;
    if (free_bytes > bytes && free_bytes - bytes >= reserve)
        return 1;

    static int skipped_logs = 0;
    if (skipped_logs < 8) {
        fprintf(stderr,
                "[bn:gpu] skipping optional CUDA layout %s layer=%d: "
                "need=%.1f MiB free=%.1f MiB total=%.1f MiB "
                "reserve=%.1f MiB\n",
                name ? name : "unknown", layer, bytes / 1048576.0,
                free_bytes / 1048576.0, total_bytes / 1048576.0,
                reserve / 1048576.0);
        skipped_logs++;
    }
    return 0;
}

static size_t qweight_pair_bytes(const BnQWeight *a, const BnQWeight *b) {
    if (!a || !b)
        return 0;
    size_t a_sz = bn_qweight_data_size(a);
    size_t b_sz = bn_qweight_data_size(b);
    if (a_sz > SIZE_MAX - b_sz)
        return SIZE_MAX;
    return a_sz + b_sz;
}

static int qweight_pair_stackable(const BnQWeight *a, const BnQWeight *b) {
    return a && b && a->rows > 0 && b->rows > 0 &&
           bn_backend_layout_stackable(a, b);
}

static size_t qweight_triple_bytes(const BnQWeight *a, const BnQWeight *b,
                                   const BnQWeight *c) {
    size_t ab = qweight_pair_bytes(a, b);
    size_t c_sz = c ? bn_qweight_data_size(c) : 0;
    if (ab > SIZE_MAX - c_sz)
        return SIZE_MAX;
    return ab + c_sz;
}

static int add_size_checked(size_t *total, size_t add) {
    if (!total || *total > SIZE_MAX - add)
        return -1;
    *total += add;
    return 0;
}

static int mul3_size(size_t a, size_t b, size_t c, size_t *out) {
    size_t ab = 0;
    if (checked_mul_size(a, b, &ab) != 0 ||
        checked_mul_size(ab, c, out) != 0)
        return -1;
    return 0;
}

static size_t cuda_q6_down_f32_cache_bytes(const BnGPUBackend *gpu,
                                           const BnMoEExpertMap *em,
                                           int n_experts) {
    if (!bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(gpu) || !em ||
        !bn_quant_format_cuda_moe_down_q6_f32_cache_supported(
            em->down_type) ||
        n_experts <= 0)
        return 0;
    if (!bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(
            em->down_cols))
        return 0;

    size_t elems = 0;
    size_t bytes = 0;
    if (mul3_size((size_t)n_experts, (size_t)em->down_rows,
                  (size_t)em->down_cols, &elems) != 0 ||
        checked_mul_size(elems, sizeof(float), &bytes) != 0)
        return SIZE_MAX;

    if (bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced())
        return bytes;

    int max_mb = bn_gpu_policy_cuda_cublas_cache_max_mb(512, 0);
    if (max_mb <= 0)
        return bytes;
    size_t max_bytes = (size_t)max_mb * 1024u * 1024u;
    return bytes <= max_bytes ? bytes : 0;
}

static int add_qweight_bytes(size_t *total, const BnQWeight *w) {
    return add_size_checked(total, bn_qweight_data_size(w));
}

static size_t qweight_default_aux_cache_bytes(const BnQWeight *w) {
    if (!w || !w->data || w->rows <= 0 || w->cols <= 0 ||
        (w->cols & 31) != 0 ||
        !bn_gpu_policy_cuda_cublas_matmul_enabled())
        return 0;
    if (!bn_quant_format_cuda_aux_cache_supported(w->type))
        return 0;
    int q6_as_f16 = bn_quant_format_cuda_aux_cache_can_use_f16(w->type) &&
                    bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled();
    size_t bytes = 0;
    if (mul3_size((size_t)w->rows, (size_t)w->cols,
                  bn_quant_format_cuda_aux_cache_uses_f32(w->type, q6_as_f16)
                      ? sizeof(float) : sizeof(uint16_t),
                  &bytes) != 0)
        return SIZE_MAX;

    int max_mb = bn_gpu_policy_cuda_cublas_cache_max_mb(
        128, bn_quant_format_cuda_aux_cache_prefers_large_budget(w->type));
    if (max_mb > 0 && bytes > (size_t)max_mb * 1024u * 1024u)
        return 0;
    return bytes;
}

static size_t qweight_stacked_aux_cache_bytes(const BnQWeight *a,
                                              int rows) {
    if (!a || rows <= 0)
        return 0;
    BnQWeight stacked = *a;
    stacked.rows = rows;
    return qweight_default_aux_cache_bytes(&stacked);
}

static size_t qweight_pair_upload_bytes(const BnQWeight *a,
                                        const BnQWeight *b) {
    size_t raw = qweight_pair_bytes(a, b);
    if (raw == SIZE_MAX)
        return SIZE_MAX;
    if (!a || !b || a->type != b->type || a->cols != b->cols)
        return raw;
    size_t aux = qweight_stacked_aux_cache_bytes(a, a->rows + b->rows);
    if (aux == SIZE_MAX || raw > SIZE_MAX - aux)
        return SIZE_MAX;
    return raw + aux;
}

static size_t qweight_triple_upload_bytes(const BnQWeight *a,
                                          const BnQWeight *b,
                                          const BnQWeight *c) {
    size_t raw = qweight_triple_bytes(a, b, c);
    if (raw == SIZE_MAX)
        return SIZE_MAX;
    if (!a || !b || !c ||
        a->type != b->type || a->type != c->type ||
        a->cols != b->cols || a->cols != c->cols)
        return raw;
    size_t aux = qweight_stacked_aux_cache_bytes(
        a, a->rows + b->rows + c->rows);
    if (aux == SIZE_MAX || raw > SIZE_MAX - aux)
        return SIZE_MAX;
    return raw + aux;
}

static int add_qweight_upload_bytes(size_t *total, const BnQWeight *w) {
    size_t aux = qweight_default_aux_cache_bytes(w);
    if (add_qweight_bytes(total, w) != 0 || aux == SIZE_MAX)
        return -1;
    return add_size_checked(total, aux);
}

static int add_f32_bytes(size_t *total, const float *data, int n_elems) {
    size_t bytes = 0;
    if (!data || n_elems <= 0)
        return 0;
    if (checked_mul_size((size_t)n_elems, sizeof(float), &bytes) != 0)
        return -1;
    return add_size_checked(total, bytes);
}

static size_t estimate_cuda_base_model_bytes(const BnConfig *c,
                                             const BnWeights *w) {
    if (!c || !w)
        return 0;
    size_t total = 0;
    if (w->output_weight.data) {
        if (add_qweight_upload_bytes(&total, &w->output_weight) != 0)
            return SIZE_MAX;
    } else if (w->token_embedding) {
        size_t elems = 0;
        size_t bytes = 0;
        if (checked_mul_size((size_t)c->vocab_size, (size_t)c->dim,
                             &elems) != 0 ||
            !bn_gguf_tensor_size((uint32_t)w->emb_type, (uint64_t)elems,
                                 &bytes) ||
            add_size_checked(&total, bytes) != 0)
            return SIZE_MAX;
    }
    if (add_f32_bytes(&total, w->output_norm, c->dim) != 0)
        return SIZE_MAX;

    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        const BnQWeight *weights[] = {
            &lw->attn.wq, &lw->attn.wk, &lw->attn.wv, &lw->attn.wo,
            &lw->ffn.ffn_gate, &lw->ffn.ffn_up, &lw->ffn.ffn_down,
            &lw->ssm.wqkv, &lw->ssm.wz,
            &lw->ssm.ssm_alpha, &lw->ssm.ssm_beta, &lw->ssm.ssm_out,
            &lw->shared.shared_gate, &lw->shared.shared_up,
            &lw->shared.shared_down,
        };
        int n_weights = (int)(sizeof(weights) / sizeof(weights[0]));
        for (int i = 0; i < n_weights; i++) {
            if (add_qweight_upload_bytes(&total, weights[i]) != 0)
                return SIZE_MAX;
        }
        if (add_f32_bytes(&total, lw->norm.attn_norm, c->dim) != 0 ||
            add_f32_bytes(&total, lw->norm.ffn_norm, c->dim) != 0 ||
            add_f32_bytes(&total, lw->moe.router_weight,
                          c->n_experts * c->dim) != 0 ||
            add_f32_bytes(&total, lw->shared.shared_expert_gate,
                          c->dim) != 0)
            return SIZE_MAX;
        if (c->n_experts == 2 && c->n_experts_active == 2 &&
            lw->moe.router_weight &&
            add_f32_bytes(&total, lw->moe.router_weight, c->dim) != 0)
            return SIZE_MAX;
        if (add_f32_bytes(&total, lw->attn.q_bias, lw->attn.wq.rows) != 0 ||
            add_f32_bytes(&total, lw->attn.k_bias, lw->attn.wk.rows) != 0 ||
            add_f32_bytes(&total, lw->attn.v_bias, lw->attn.wv.rows) != 0)
            return SIZE_MAX;
        if (lw->attn.q_norm) {
            int q_norm_size = c->qk_norm_per_head
                ? c->n_heads * c->head_size : c->head_size;
            if (add_f32_bytes(&total, lw->attn.q_norm, q_norm_size) != 0)
                return SIZE_MAX;
        }
        if (lw->attn.k_norm) {
            int k_norm_size = c->qk_norm_per_head ? c->kv_dim : c->head_size;
            if (add_f32_bytes(&total, lw->attn.k_norm, k_norm_size) != 0)
                return SIZE_MAX;
        }
        if (add_f32_bytes(&total, lw->norm.attn_sub_norm, c->dim) != 0 ||
            add_f32_bytes(&total, lw->norm.ffn_sub_norm, c->hidden_dim) != 0)
            return SIZE_MAX;
        if (lw->ssm.ssm_conv1d) {
            int num_v_heads = c->ssm_time_step_rank;
            int head_k_dim = c->ssm_state_size;
            int key_dim = c->ssm_group_count * head_k_dim;
            int qkv_dim = key_dim * 2 + c->ssm_inner_size;
            int kern = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;
            if (add_f32_bytes(&total, lw->ssm.ssm_conv1d,
                              kern * qkv_dim) != 0 ||
                add_f32_bytes(&total, lw->ssm.ssm_dt_bias,
                              num_v_heads) != 0 ||
                add_f32_bytes(&total, lw->ssm.ssm_a, num_v_heads) != 0)
                return SIZE_MAX;
            if (lw->ssm.ssm_norm) {
                int head_v_dim = num_v_heads > 0
                    ? c->ssm_inner_size / num_v_heads : c->ssm_inner_size;
                if (add_f32_bytes(&total, lw->ssm.ssm_norm,
                                  head_v_dim) != 0)
                    return SIZE_MAX;
            }
        }
    }
    return total;
}

static size_t estimate_cuda_moe_layer_bytes(const BnConfig *c,
                                            const BnMoEExpertMap *em,
                                            const BnGPUBackend *gpu,
                                            int q8_f16_cache) {
    if (!c || !em || c->n_experts <= 0)
        return 0;
    size_t total = 0;
    size_t proj = 0;
    if (checked_mul_size(em->expert_gate_bytes,
                         (size_t)c->n_experts, &proj) != 0)
        return SIZE_MAX;
    total += proj;
    if (checked_mul_size(em->expert_up_bytes,
                         (size_t)c->n_experts, &proj) != 0 ||
        add_size_checked(&total, proj) != 0)
        return SIZE_MAX;
    if (checked_mul_size(em->expert_down_bytes,
                         (size_t)c->n_experts, &proj) != 0 ||
        add_size_checked(&total, proj) != 0)
        return SIZE_MAX;
    size_t aux = cuda_q6_down_f32_cache_bytes(gpu, em, c->n_experts);
    if (aux == SIZE_MAX || add_size_checked(&total, aux) != 0)
        return SIZE_MAX;
    if (bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
            gpu, em->gate_type, q8_f16_cache)) {
        if (mul3_size((size_t)c->n_experts,
                      (size_t)em->gate_rows,
                      (size_t)em->gate_cols, &aux) != 0 ||
            checked_mul_size(aux, sizeof(uint16_t), &aux) != 0 ||
            add_size_checked(&total, aux) != 0)
            return SIZE_MAX;
    }
    if (bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
            gpu, em->up_type, q8_f16_cache)) {
        if (mul3_size((size_t)c->n_experts,
                      (size_t)em->up_rows,
                      (size_t)em->up_cols, &aux) != 0 ||
            checked_mul_size(aux, sizeof(uint16_t), &aux) != 0 ||
            add_size_checked(&total, aux) != 0)
            return SIZE_MAX;
    }
    if (bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
            gpu, em->down_type, q8_f16_cache)) {
        if (mul3_size((size_t)c->n_experts,
                      (size_t)em->down_rows,
                      (size_t)em->down_cols, &aux) != 0 ||
            checked_mul_size(aux, sizeof(uint16_t), &aux) != 0 ||
            add_size_checked(&total, aux) != 0)
            return SIZE_MAX;
    }
    return total;
}

static size_t estimate_cuda_moe_gateup_f16_all_bytes(const BnConfig *c,
                                                     const BnWeights *w,
                                                     const BnGPUBackend *gpu) {
    if (!c || !w || c->n_experts <= 0)
        return 0;
    size_t total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        const BnMoEExpertMap *em = &w->layers[l].moe.expert_map;
        size_t layer = estimate_cuda_moe_layer_bytes(c, em, gpu, 0);
        size_t aux = 0;
        if (layer == SIZE_MAX)
            return SIZE_MAX;
        if (bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
                gpu, em->gate_type, 1)) {
            if (mul3_size((size_t)c->n_experts,
                          (size_t)em->gate_rows,
                          (size_t)em->gate_cols, &aux) != 0 ||
                checked_mul_size(aux, sizeof(uint16_t), &aux) != 0 ||
                add_size_checked(&layer, aux) != 0)
                return SIZE_MAX;
        }
        if (bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
                gpu, em->up_type, 1)) {
            if (mul3_size((size_t)c->n_experts,
                          (size_t)em->up_rows,
                          (size_t)em->up_cols, &aux) != 0 ||
                checked_mul_size(aux, sizeof(uint16_t), &aux) != 0 ||
                add_size_checked(&layer, aux) != 0)
                return SIZE_MAX;
        }
        if (add_size_checked(&total, layer) != 0)
            return SIZE_MAX;
    }
    return total;
}

static size_t estimate_cuda_moe_all_bytes(const BnConfig *c,
                                          const BnWeights *w,
                                          const BnGPUBackend *gpu,
                                          int q8_f16_cache) {
    if (!c || !w || c->n_experts <= 0)
        return 0;
    size_t total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        const BnMoEExpertMap *em = &w->layers[l].moe.expert_map;
        size_t layer =
            estimate_cuda_moe_layer_bytes(c, em, gpu, q8_f16_cache);
        if (layer == SIZE_MAX)
            return SIZE_MAX;
        if (add_size_checked(&total, layer) != 0)
            return SIZE_MAX;
    }
    return total;
}

static int cuda_moe_all_fits_memory(BnGPUBackend *gpu,
                                    const BnConfig *c,
                                    const BnWeights *w,
                                    int q8_f16_cache) {
    if (!gpu || !gpu->memory_info)
        return 1;
    size_t need = estimate_cuda_moe_all_bytes(c, w, gpu, q8_f16_cache);
    if (need == 0)
        return 0;
    if (need == SIZE_MAX)
        return 0;
    size_t base = estimate_cuda_base_model_bytes(c, w);
    if (base == SIZE_MAX)
        return 0;
    size_t projected = 0;
    if (add_size_checked(&projected, need) != 0 ||
        add_size_checked(&projected, base) != 0)
        return 0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (gpu->memory_info(gpu->ctx, &free_bytes, &total_bytes) != 0)
        return 1;
    size_t reserve_mb = env_mb_or_default("BN_CUDA_MOE_FULL_RESERVE_MB", 512);
    size_t reserve = reserve_mb > SIZE_MAX / (1024u * 1024u)
        ? SIZE_MAX
        : reserve_mb * 1024u * 1024u;
    if (free_bytes > projected && free_bytes - projected >= reserve)
        return 1;
    fprintf(stderr,
            "[bn:gpu] skipping full CUDA MoE residency: moe=%.1f GiB "
            "base=%.1f GiB projected=%.1f GiB "
            "free=%.1f GiB total=%.1f GiB reserve=%.1f GiB "
            "(%s)\n",
            need / 1073741824.0, base / 1073741824.0,
            projected / 1073741824.0, free_bytes / 1073741824.0,
            total_bytes / 1073741824.0, reserve / 1073741824.0,
            q8_f16_cache
                ? "will try quant-only full residency"
                : "using lazy GPU expert cache");
    return 0;
}

static int cuda_moe_gateup_f16_fits_memory(BnGPUBackend *gpu,
                                           const BnConfig *c,
                                           const BnWeights *w) {
    if (!gpu || !gpu->memory_info)
        return 1;
    size_t need = estimate_cuda_moe_gateup_f16_all_bytes(c, w, gpu);
    if (need == 0 || need == SIZE_MAX)
        return 0;
    size_t base = estimate_cuda_base_model_bytes(c, w);
    if (base == SIZE_MAX)
        return 0;
    size_t projected = 0;
    if (add_size_checked(&projected, need) != 0 ||
        add_size_checked(&projected, base) != 0)
        return 0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (gpu->memory_info(gpu->ctx, &free_bytes, &total_bytes) != 0)
        return 1;
    size_t reserve_mb = env_mb_or_default("BN_CUDA_MOE_FULL_RESERVE_MB", 512);
    size_t reserve = reserve_mb > SIZE_MAX / (1024u * 1024u)
        ? SIZE_MAX
        : reserve_mb * 1024u * 1024u;
    if (free_bytes > projected && free_bytes - projected >= reserve)
        return 1;
    if (bn_gpu_policy_cuda_moe_fit_debug_enabled())
        fprintf(stderr,
                "[bn:gpu] skipping gate/up F16 CUDA MoE aux cache: "
                "moe=%.1f GiB base=%.1f GiB projected=%.1f GiB "
                "free=%.1f GiB total=%.1f GiB reserve=%.1f GiB\n",
                need / 1073741824.0, base / 1073741824.0,
                projected / 1073741824.0, free_bytes / 1073741824.0,
                total_bytes / 1073741824.0, reserve / 1073741824.0);
    return 0;
}

static int cuda_moe_f16_cache_layers_that_fit(BnGPUBackend *gpu,
                                              const BnConfig *c,
                                              const BnWeights *w) {
    if (!gpu || !gpu->memory_info || !c || !w)
        return 0;
    size_t quant_need = estimate_cuda_moe_all_bytes(c, w, gpu, 0);
    size_t base = estimate_cuda_base_model_bytes(c, w);
    if (quant_need == 0 || quant_need == SIZE_MAX || base == SIZE_MAX)
        return 0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (gpu->memory_info(gpu->ctx, &free_bytes, &total_bytes) != 0)
        return 0;
    size_t reserve_mb = env_mb_or_default("BN_CUDA_MOE_FULL_RESERVE_MB", 512);
    size_t reserve = reserve_mb > SIZE_MAX / (1024u * 1024u)
        ? SIZE_MAX
        : reserve_mb * 1024u * 1024u;
    size_t used = 0;
    if (add_size_checked(&used, base) != 0 ||
        add_size_checked(&used, quant_need) != 0 ||
        add_size_checked(&used, reserve) != 0 ||
        used >= free_bytes)
        return 0;
    size_t extra_budget = free_bytes - used;
    int layers = 0;
    for (int l = 0; l < c->n_layers; l++) {
        const BnMoEExpertMap *em = &w->layers[l].moe.expert_map;
        size_t quant_layer = estimate_cuda_moe_layer_bytes(c, em, gpu, 0);
        size_t f16_layer = estimate_cuda_moe_layer_bytes(c, em, gpu, 1);
        if (quant_layer == SIZE_MAX || f16_layer == SIZE_MAX)
            break;
        if (f16_layer <= quant_layer) {
            layers++;
            continue;
        }
        size_t extra = f16_layer - quant_layer;
        if (extra > extra_budget)
            break;
        extra_budget -= extra;
        layers++;
    }
    return layers;
}

int bn_model_upload_weights(BnModel *model, BnGPUBackend *gpu) {
    if (!model || !gpu || !gpu->buffer_create) return -1;
    if (bn_model_ensure_backend(model) != 0) return -1;
    BnBackendModel *backend = bn_model_backend(model);
    bn_backend_model_bind_gpu(backend, gpu);

    BnWeights *w = &model->weights;
    BnConfig *c = &model->config;
    int n_layers = c->n_layers;
    int upload_moe_all_model = bn_gpu_policy_cuda_moe_routed_ffn_enabled(
        can_use_cuda_moe_routed_ffn_model(c, w));
    int upload_moe_all_q8_f16_cache = 0;
    int upload_moe_all_gateup_f16_cache = 0;
    int upload_moe_all_f16_cache_layers = 0;
    if (upload_moe_all_model) {
        int force_f16_cache =
            bn_gpu_policy_cuda_moe_all_f16_cache_forced();
        int auto_f16_cache =
            c->n_experts > 2 ||
            (c->n_experts == 2 && c->n_experts_active == 2);
        if ((force_f16_cache || auto_f16_cache) &&
            cuda_moe_all_fits_memory(gpu, c, w, 1)) {
            upload_moe_all_q8_f16_cache = 1;
            upload_moe_all_f16_cache_layers = c->n_layers;
        } else if (!force_f16_cache &&
                   cuda_moe_all_fits_memory(gpu, c, w, 0)) {
            upload_moe_all_q8_f16_cache = 0;
            if (bn_gpu_policy_cuda_moe_gateup_f16_cache_enabled(
                    auto_f16_cache) &&
                cuda_moe_gateup_f16_fits_memory(gpu, c, w)) {
                upload_moe_all_gateup_f16_cache = 1;
                fprintf(stderr,
                        "[bn:gpu] using gate/up F16 aux CUDA MoE cache; "
                        "down experts remain quantized\n");
            }
            if (bn_gpu_policy_cuda_partial_moe_f16_cache_enabled(
                    auto_f16_cache))
                upload_moe_all_f16_cache_layers =
                    cuda_moe_f16_cache_layers_that_fit(gpu, c, w);
            if (upload_moe_all_f16_cache_layers > 0) {
                fprintf(stderr,
                        "[bn:gpu] using partial F16 aux CUDA MoE cache for "
                        "%d/%d layers; remaining resident experts are "
                        "quant-only\n",
                        upload_moe_all_f16_cache_layers, c->n_layers);
            } else {
                fprintf(stderr,
                        "[bn:gpu] using quant-only full CUDA MoE residency; "
                        "F16 aux expert cache is disabled or does not fit "
                        "current VRAM\n");
            }
        } else {
            upload_moe_all_model = 0;
        }
    }

    if (w->output_weight.data) {
        void *output_weight_gpu = upload_qweight_logits(gpu, &w->output_weight);
        if (!output_weight_gpu ||
            bn_backend_model_register_qweight(backend, &w->output_weight,
                                              output_weight_gpu) != 0) {
            if (output_weight_gpu)
                gpu->buffer_destroy(gpu->ctx, output_weight_gpu);
            bn_model_release_gpu(model);
            return -1;
        }
    }

    if (!w->output_weight.data && w->token_embedding) {
        size_t nelements = 0;
        size_t emb_size = 0;
        if (checked_mul_size((size_t)c->vocab_size, (size_t)c->dim, &nelements) == 0 &&
            bn_gguf_tensor_size((uint32_t)w->emb_type, (uint64_t)nelements, &emb_size)) {
            BnQWeight tied = {
                .data = w->token_embedding,
                .type = w->emb_type,
                .rows = c->vocab_size,
                .cols = c->dim,
                .scale = 1.0f,
            };
            void *emb_gpu_buf = upload_qweight_logits(gpu, &tied);
            if (register_gpu_handle(model, -1, BN_BACKEND_HANDLE_TIED_EMBEDDING,
                                    emb_gpu_buf) != 0) {
                if (emb_gpu_buf) gpu->buffer_destroy(gpu->ctx, emb_gpu_buf);
                bn_model_release_gpu(model);
                return -1;
            }
        }
    }

    void *output_norm_gpu = upload_f32_buf(gpu, w->output_norm, c->dim);
    if (register_gpu_handle(model, -1, BN_BACKEND_HANDLE_OUTPUT_NORM,
                            output_norm_gpu) != 0) {
        if (output_norm_gpu) gpu->buffer_destroy(gpu->ctx, output_norm_gpu);
        bn_model_release_gpu(model);
        return -1;
    }

    for (int l = 0; l < n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        BnQWeight *weights[] = {
            &lw->attn.wq, &lw->attn.wk, &lw->attn.wv, &lw->attn.wo,
            &lw->ffn.ffn_gate, &lw->ffn.ffn_up, &lw->ffn.ffn_down,
            &lw->ssm.wqkv, &lw->ssm.wz,
            &lw->ssm.ssm_alpha, &lw->ssm.ssm_beta, &lw->ssm.ssm_out,
            &lw->shared.shared_gate, &lw->shared.shared_up,
            &lw->shared.shared_down,
        };
        int quant_only_individual[15] = {0};
        if (bn_gpu_policy_cuda_individual_upload_quant_only_enabled(gpu)) {
            if (lw->ssm.wqkv.data &&
                qweight_pair_stackable(&lw->ffn.ffn_gate,
                                       &lw->ffn.ffn_up)) {
                quant_only_individual[4] = 1;
                quant_only_individual[5] = 1;
            }
            if (qweight_pair_stackable(&lw->ssm.wqkv, &lw->ssm.wz)) {
                quant_only_individual[7] = 1;
                quant_only_individual[8] = 1;
            }
        }
        int n_weights = (int)(sizeof(weights) / sizeof(weights[0]));
        for (int i = 0; i < n_weights; i++) {
            if (upload_qweight_owned_mode(model, backend, gpu, weights[i],
                                          quant_only_individual[i]) != 0) {
                bn_model_release_gpu(model);
                return -1;
            }
        }

        void *attn_norm_gpu = upload_f32_buf(gpu, lw->norm.attn_norm, c->dim);
        void *ffn_norm_gpu  = upload_f32_buf(gpu, lw->norm.ffn_norm, c->dim);
        void *moe_router_diff_gpu =
            (c->n_experts == 2 && c->n_experts_active == 2)
                ? upload_moe_router_diff2(gpu, lw->moe.router_weight, c->dim)
                : NULL;
        void *moe_router_gpu = lw->moe.router_weight
            ? gpu->buffer_create(
                gpu->ctx, lw->moe.router_weight,
                (size_t)c->n_experts * (size_t)c->dim * sizeof(float),
                bn_quant_format_gpu_float_buffer_type(),
                c->n_experts, c->dim)
            : NULL;
        int upload_moe_all = upload_moe_all_model &&
                             can_use_cuda_moe_routed_ffn(c, lw);
        int upload_layer_f16_cache =
            upload_moe_all_q8_f16_cache ||
            (upload_moe_all_f16_cache_layers > 0 &&
             l < upload_moe_all_f16_cache_layers);
        int upload_layer_gateup_f16_cache =
            upload_layer_f16_cache || upload_moe_all_gateup_f16_cache;
        void *moe_gate_all_gpu = upload_moe_all
            ? upload_moe_all_proj(model, gpu, &lw->moe.expert_map, 0,
                                  c->n_experts,
                                  upload_layer_gateup_f16_cache)
            : NULL;
        void *moe_up_all_gpu = upload_moe_all
            ? upload_moe_all_proj(model, gpu, &lw->moe.expert_map, 1,
                                  c->n_experts,
                                  upload_layer_gateup_f16_cache)
            : NULL;
        void *moe_down_all_gpu = upload_moe_all
            ? upload_moe_all_proj(model, gpu, &lw->moe.expert_map, 2,
                                  c->n_experts,
                                  upload_layer_f16_cache)
            : NULL;
        if (upload_moe_all &&
            (!moe_gate_all_gpu || !moe_up_all_gpu || !moe_down_all_gpu)) {
            if (moe_gate_all_gpu)
                gpu->buffer_destroy(gpu->ctx, moe_gate_all_gpu);
            if (moe_up_all_gpu)
                gpu->buffer_destroy(gpu->ctx, moe_up_all_gpu);
            if (moe_down_all_gpu)
                gpu->buffer_destroy(gpu->ctx, moe_down_all_gpu);
            moe_gate_all_gpu = NULL;
            moe_up_all_gpu = NULL;
            moe_down_all_gpu = NULL;
            if (upload_layer_gateup_f16_cache) {
                fprintf(stderr,
                        "[bn:gpu] full CUDA MoE residency aux-cache upload "
                        "failed at layer=%d; retrying quant-only resident "
                        "experts for this and later layers\n", l);
                upload_moe_all_q8_f16_cache = 0;
                upload_moe_all_gateup_f16_cache = 0;
                upload_moe_all_f16_cache_layers = l;
                moe_gate_all_gpu =
                    upload_moe_all_proj(model, gpu, &lw->moe.expert_map, 0,
                                        c->n_experts, 0);
                moe_up_all_gpu =
                    upload_moe_all_proj(model, gpu, &lw->moe.expert_map, 1,
                                        c->n_experts, 0);
                moe_down_all_gpu =
                    upload_moe_all_proj(model, gpu, &lw->moe.expert_map, 2,
                                        c->n_experts, 0);
            }
            if (!moe_gate_all_gpu || !moe_up_all_gpu || !moe_down_all_gpu) {
                if (moe_gate_all_gpu)
                    gpu->buffer_destroy(gpu->ctx, moe_gate_all_gpu);
                if (moe_up_all_gpu)
                    gpu->buffer_destroy(gpu->ctx, moe_up_all_gpu);
                if (moe_down_all_gpu)
                    gpu->buffer_destroy(gpu->ctx, moe_down_all_gpu);
                fprintf(stderr,
                        "[bn:gpu] failed full CUDA MoE residency upload at "
                        "layer=%d; aborting upload instead of mixing "
                        "resident and fallback MoE paths\n", l);
                bn_model_release_gpu(model);
                return -1;
            }
        }
        void *shared_expert_gate_gpu = lw->shared.shared_expert_gate
            ? gpu->buffer_create(
                gpu->ctx, lw->shared.shared_expert_gate,
                (size_t)c->dim * sizeof(float),
                bn_quant_format_gpu_float_buffer_type(), 1, c->dim)
            : NULL;
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_ATTN_NORM,
                                attn_norm_gpu) != 0 ||
            register_gpu_handle(model, l, BN_BACKEND_HANDLE_FFN_NORM,
                                ffn_norm_gpu) != 0 ||
            register_gpu_handle(model, l, BN_BACKEND_HANDLE_MOE_ROUTER_DIFF,
                                moe_router_diff_gpu) != 0 ||
            register_gpu_handle(model, l, BN_BACKEND_HANDLE_MOE_ROUTER,
                                moe_router_gpu) != 0 ||
            register_gpu_handle(model, l, BN_BACKEND_HANDLE_MOE_GATE_ALL,
                                moe_gate_all_gpu) != 0 ||
            register_gpu_handle(model, l, BN_BACKEND_HANDLE_MOE_UP_ALL,
                                moe_up_all_gpu) != 0 ||
            register_gpu_handle(model, l, BN_BACKEND_HANDLE_MOE_DOWN_ALL,
                                moe_down_all_gpu) != 0 ||
            register_gpu_handle(model, l, BN_BACKEND_HANDLE_SHARED_EXPERT_GATE,
                                shared_expert_gate_gpu) != 0) {
            if (attn_norm_gpu) gpu->buffer_destroy(gpu->ctx, attn_norm_gpu);
            if (ffn_norm_gpu) gpu->buffer_destroy(gpu->ctx, ffn_norm_gpu);
            if (moe_router_diff_gpu)
                gpu->buffer_destroy(gpu->ctx, moe_router_diff_gpu);
            if (moe_router_gpu)
                gpu->buffer_destroy(gpu->ctx, moe_router_gpu);
            if (moe_gate_all_gpu)
                gpu->buffer_destroy(gpu->ctx, moe_gate_all_gpu);
            if (moe_up_all_gpu)
                gpu->buffer_destroy(gpu->ctx, moe_up_all_gpu);
            if (moe_down_all_gpu)
                gpu->buffer_destroy(gpu->ctx, moe_down_all_gpu);
            if (shared_expert_gate_gpu)
                gpu->buffer_destroy(gpu->ctx, shared_expert_gate_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *q_bias_gpu = NULL;
        void *k_bias_gpu = NULL;
        void *v_bias_gpu = NULL;
        if (gpu->buffer_create_biased) {
            struct { BnQWeight *w; float *bias; void **bias_gpu; } qkv_bias[] = {
                { &lw->attn.wq, lw->attn.q_bias, &q_bias_gpu },
                { &lw->attn.wk, lw->attn.k_bias, &k_bias_gpu },
                { &lw->attn.wv, lw->attn.v_bias, &v_bias_gpu },
            };
            for (int i = 0; i < 3; i++) {
                void *old_handle = bn_backend_model_qweight_buf(backend, qkv_bias[i].w);
                if (!qkv_bias[i].bias || !old_handle) continue;
                void *fused = bn_backend_layout_upload_biased_qweight(
                    gpu, qkv_bias[i].w, qkv_bias[i].bias);
                if (fused) {
                    gpu->buffer_destroy(gpu->ctx, old_handle);
                    if (bn_backend_model_register_qweight(backend,
                                                          qkv_bias[i].w,
                                                          fused) != 0) {
                        bn_model_release_gpu(model);
                        return -1;
                    }
                } else {
                    *qkv_bias[i].bias_gpu = upload_f32_buf(gpu, qkv_bias[i].bias,
                                                            qkv_bias[i].w->rows);
                }
            }
        } else {
            if (lw->attn.q_bias)
                q_bias_gpu = upload_f32_buf(gpu, lw->attn.q_bias, c->dim);
            if (lw->attn.k_bias)
                k_bias_gpu = upload_f32_buf(gpu, lw->attn.k_bias, c->kv_dim);
            if (lw->attn.v_bias)
                v_bias_gpu = upload_f32_buf(gpu, lw->attn.v_bias, c->kv_dim);
        }
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_Q_BIAS, q_bias_gpu) != 0 ||
            register_gpu_handle(model, l, BN_BACKEND_HANDLE_K_BIAS, k_bias_gpu) != 0 ||
            register_gpu_handle(model, l, BN_BACKEND_HANDLE_V_BIAS, v_bias_gpu) != 0) {
            if (q_bias_gpu) gpu->buffer_destroy(gpu->ctx, q_bias_gpu);
            if (k_bias_gpu) gpu->buffer_destroy(gpu->ctx, k_bias_gpu);
            if (v_bias_gpu) gpu->buffer_destroy(gpu->ctx, v_bias_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *qkv_stacked_gpu = NULL;
        if (optional_layout_fits_memory(
                gpu, qweight_triple_upload_bytes(&lw->attn.wq, &lw->attn.wk,
                                                  &lw->attn.wv),
                l, "qkv_stacked")) {
            qkv_stacked_gpu = bn_backend_layout_upload_stacked3_qkv(
                gpu, &lw->attn.wq, &lw->attn.wk, &lw->attn.wv,
                lw->attn.q_bias, lw->attn.k_bias, lw->attn.v_bias,
                lw->attn.q_bias && !q_bias_gpu,
                lw->attn.k_bias && !k_bias_gpu,
                lw->attn.v_bias && !v_bias_gpu);
        }
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_QKV_STACKED,
                                qkv_stacked_gpu) != 0) {
            if (qkv_stacked_gpu) gpu->buffer_destroy(gpu->ctx, qkv_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *qk_stacked_gpu = NULL;
        if (optional_layout_fits_memory(
                gpu, qweight_pair_upload_bytes(&lw->attn.wq, &lw->attn.wk),
                l, "qk_stacked")) {
            qk_stacked_gpu =
                bn_backend_layout_upload_stacked2(gpu, &lw->attn.wq,
                                                  &lw->attn.wk);
        }
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_QK_STACKED,
                                qk_stacked_gpu) != 0) {
            if (qk_stacked_gpu) gpu->buffer_destroy(gpu->ctx, qk_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *gateup_stacked_gpu = NULL;
        if (optional_layout_fits_memory(
                gpu,
                qweight_pair_upload_bytes(&lw->ffn.ffn_gate, &lw->ffn.ffn_up),
                l, "gateup_stacked")) {
            gateup_stacked_gpu =
                bn_backend_layout_upload_stacked2(gpu, &lw->ffn.ffn_gate,
                                                  &lw->ffn.ffn_up);
        }
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_GATEUP_STACKED,
                                gateup_stacked_gpu) != 0) {
            if (gateup_stacked_gpu) gpu->buffer_destroy(gpu->ctx, gateup_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *shared_gateup_stacked_gpu = NULL;
        if (optional_layout_fits_memory(
                gpu, qweight_pair_upload_bytes(&lw->shared.shared_gate,
                                                &lw->shared.shared_up),
                l, "shared_gateup_stacked")) {
            shared_gateup_stacked_gpu =
                bn_backend_layout_upload_stacked2(
                    gpu, &lw->shared.shared_gate, &lw->shared.shared_up);
        }
        if (register_gpu_handle(model, l,
                                BN_BACKEND_HANDLE_SHARED_GATEUP_STACKED,
                                shared_gateup_stacked_gpu) != 0) {
            if (shared_gateup_stacked_gpu)
                gpu->buffer_destroy(gpu->ctx, shared_gateup_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *ssm_qkvz_stacked_gpu = NULL;
        if (optional_layout_fits_memory(
                gpu, qweight_pair_upload_bytes(&lw->ssm.wqkv, &lw->ssm.wz),
                l, "ssm_qkvz_stacked")) {
            ssm_qkvz_stacked_gpu =
                bn_backend_layout_upload_stacked2(gpu, &lw->ssm.wqkv,
                                                  &lw->ssm.wz);
        }
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_QKVZ_STACKED,
                                ssm_qkvz_stacked_gpu) != 0) {
            if (ssm_qkvz_stacked_gpu) gpu->buffer_destroy(gpu->ctx, ssm_qkvz_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *ssm_ab_stacked_gpu = NULL;
        if (optional_layout_fits_memory(
                gpu, qweight_pair_upload_bytes(&lw->ssm.ssm_alpha,
                                                &lw->ssm.ssm_beta),
                l, "ssm_ab_stacked")) {
            ssm_ab_stacked_gpu =
                bn_backend_layout_upload_stacked2(gpu, &lw->ssm.ssm_alpha,
                                                  &lw->ssm.ssm_beta);
        }
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_AB_STACKED,
                                ssm_ab_stacked_gpu) != 0) {
            if (ssm_ab_stacked_gpu) gpu->buffer_destroy(gpu->ctx, ssm_ab_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        if (lw->attn.q_norm) {
            int q_norm_size = c->qk_norm_per_head ? (c->n_heads * c->head_size) : c->head_size;
            void *q_norm_gpu = upload_f32_buf(gpu, lw->attn.q_norm, q_norm_size);
            if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_Q_NORM,
                                    q_norm_gpu) != 0) {
                if (q_norm_gpu) gpu->buffer_destroy(gpu->ctx, q_norm_gpu);
                bn_model_release_gpu(model);
                return -1;
            }
        }
        if (lw->attn.k_norm) {
            int k_norm_size = c->qk_norm_per_head ? c->kv_dim : c->head_size;
            void *k_norm_gpu = upload_f32_buf(gpu, lw->attn.k_norm, k_norm_size);
            if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_K_NORM,
                                    k_norm_gpu) != 0) {
                if (k_norm_gpu) gpu->buffer_destroy(gpu->ctx, k_norm_gpu);
                bn_model_release_gpu(model);
                return -1;
            }
        }
        if (lw->norm.attn_sub_norm) {
            void *attn_sub_norm_gpu = upload_f32_buf(gpu, lw->norm.attn_sub_norm, c->dim);
            if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_ATTN_SUB_NORM,
                                    attn_sub_norm_gpu) != 0) {
                if (attn_sub_norm_gpu) gpu->buffer_destroy(gpu->ctx, attn_sub_norm_gpu);
                bn_model_release_gpu(model);
                return -1;
            }
        }
        if (lw->norm.ffn_sub_norm) {
            void *ffn_sub_norm_gpu = upload_f32_buf(gpu, lw->norm.ffn_sub_norm, c->hidden_dim);
            if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_FFN_SUB_NORM,
                                    ffn_sub_norm_gpu) != 0) {
                if (ffn_sub_norm_gpu) gpu->buffer_destroy(gpu->ctx, ffn_sub_norm_gpu);
                bn_model_release_gpu(model);
                return -1;
            }
        }

        if (lw->ssm.ssm_conv1d) {
            int num_v_heads = c->ssm_time_step_rank;
            int head_k_dim  = c->ssm_state_size;
            int key_dim     = c->ssm_group_count * head_k_dim;
            int qkv_dim     = key_dim * 2 + c->ssm_inner_size;
            int kern        = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;
            void *ssm_conv1d_gpu = upload_f32_buf(gpu, lw->ssm.ssm_conv1d, kern * qkv_dim);
            if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_CONV1D,
                                    ssm_conv1d_gpu) != 0) {
                if (ssm_conv1d_gpu) gpu->buffer_destroy(gpu->ctx, ssm_conv1d_gpu);
                bn_model_release_gpu(model);
                return -1;
            }

            if (lw->ssm.ssm_dt_bias && num_v_heads > 0) {
                void *ssm_dt_bias_gpu = upload_f32_buf(gpu, lw->ssm.ssm_dt_bias, num_v_heads);
                if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_DT_BIAS,
                                        ssm_dt_bias_gpu) != 0) {
                    if (ssm_dt_bias_gpu) gpu->buffer_destroy(gpu->ctx, ssm_dt_bias_gpu);
                    bn_model_release_gpu(model);
                    return -1;
                }
            }
            if (lw->ssm.ssm_a && num_v_heads > 0) {
                void *ssm_a_log_gpu = upload_f32_buf(gpu, lw->ssm.ssm_a, num_v_heads);
                if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_A_LOG,
                                        ssm_a_log_gpu) != 0) {
                    if (ssm_a_log_gpu) gpu->buffer_destroy(gpu->ctx, ssm_a_log_gpu);
                    bn_model_release_gpu(model);
                    return -1;
                }
            }

            if (lw->ssm.ssm_norm) {
                int head_v_dim = num_v_heads > 0
                    ? c->ssm_inner_size / num_v_heads : c->ssm_inner_size;
                void *ssm_norm_gpu = upload_f32_buf(gpu, lw->ssm.ssm_norm, head_v_dim);
                if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_NORM,
                                        ssm_norm_gpu) != 0) {
                    if (ssm_norm_gpu) gpu->buffer_destroy(gpu->ctx, ssm_norm_gpu);
                    bn_model_release_gpu(model);
                    return -1;
                }
            }
        }
    }

    return 0;
}

void bn_model_release_gpu(BnModel *model) {
    if (!model) return;
    BnBackendModel *backend = bn_model_backend(model);

    if (backend)
        bn_backend_model_release_gpu(backend);
}
