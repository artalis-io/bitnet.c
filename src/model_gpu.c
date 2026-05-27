#include "model.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "gpu_backend.h"
#include "moe_internal.h"
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

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

static void *upload_qweight(BnGPUBackend *gpu, BnQWeight *w) {
    if (!w->data) return NULL;
    size_t sz = bn_qweight_data_size(w);
    if (sz == 0) return NULL;
    return gpu->buffer_create(gpu->ctx, w->data, sz, w->type, w->rows, w->cols);
}

static int upload_qweight_owned(BnModel *model, BnBackendModel *backend,
                                BnGPUBackend *gpu, BnQWeight *w) {
    (void)model;
    void *handle = upload_qweight(gpu, w);
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
                                      BN_GGUF_TENSOR_F32, 1, dim);
    free(diff);
    return handle;
}

static void *upload_moe_all_proj(BnModel *model,
                                 BnGPUBackend *gpu,
                                 const BnMoEExpertMap *em,
                                 int proj,
                                 int n_experts) {
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
    if (stride != expert_bytes)
        return NULL;
    const uint8_t *base = bn_moe_mmap_base_for_proj(
        bn_model_moe_io(model), em, proj);
    if (!base)
        return NULL;
    size_t total_bytes = 0;
    if (checked_mul_size(expert_bytes, (size_t)n_experts, &total_bytes) != 0)
        return NULL;
    if ((size_t)n_experts > (size_t)INT_MAX / (size_t)rows)
        return NULL;
    return gpu->buffer_create(gpu->ctx, base + offset, total_bytes,
                              type, rows * n_experts, cols);
}

static int can_use_cuda_moe_routed_ffn(const BnConfig *c,
                                       const BnLayerWeights *lw) {
    if (!c || !lw || !lw->moe.router_weight)
        return 0;
    const BnMoEExpertMap *em = &lw->moe.expert_map;
    return !c->has_shared_expert &&
           em->gate_type == BN_GGUF_TENSOR_Q4_K &&
           em->up_type == BN_GGUF_TENSOR_Q4_K &&
           em->down_type == BN_GGUF_TENSOR_Q6_K &&
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

int bn_model_upload_weights(BnModel *model, BnGPUBackend *gpu) {
    if (!model || !gpu || !gpu->buffer_create) return -1;
    if (bn_model_ensure_backend(model) != 0) return -1;
    BnBackendModel *backend = bn_model_backend(model);
    bn_backend_model_bind_gpu(backend, gpu);

    BnWeights *w = &model->weights;
    BnConfig *c = &model->config;
    int n_layers = c->n_layers;
    int upload_moe_all_model = getenv("BN_CUDA_ENABLE_MOE_ROUTED_FFN") &&
                               can_use_cuda_moe_routed_ffn_model(c, w);

    if (w->output_weight.data) {
        void *output_weight_gpu = upload_qweight(gpu, &w->output_weight);
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
            void *emb_gpu_buf = gpu->buffer_create(gpu->ctx, w->token_embedding,
                emb_size, w->emb_type, c->vocab_size, c->dim);
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
            &lw->shared.shared_gate, &lw->shared.shared_up, &lw->shared.shared_down,
        };
        int n_weights = (int)(sizeof(weights) / sizeof(weights[0]));
        for (int i = 0; i < n_weights; i++) {
            if (upload_qweight_owned(model, backend, gpu, weights[i]) != 0) {
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
                BN_GGUF_TENSOR_F32, c->n_experts, c->dim)
            : NULL;
        int upload_moe_all = upload_moe_all_model &&
                             can_use_cuda_moe_routed_ffn(c, lw);
        void *moe_gate_all_gpu = upload_moe_all
            ? upload_moe_all_proj(model, gpu, &lw->moe.expert_map, 0,
                                  c->n_experts)
            : NULL;
        void *moe_up_all_gpu = upload_moe_all
            ? upload_moe_all_proj(model, gpu, &lw->moe.expert_map, 1,
                                  c->n_experts)
            : NULL;
        void *moe_down_all_gpu = upload_moe_all
            ? upload_moe_all_proj(model, gpu, &lw->moe.expert_map, 2,
                                  c->n_experts)
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
        }
        void *shared_expert_gate_gpu = lw->shared.shared_expert_gate
            ? gpu->buffer_create(
                gpu->ctx, lw->shared.shared_expert_gate,
                (size_t)c->dim * sizeof(float), BN_GGUF_TENSOR_F32,
                1, c->dim)
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

        void *qkv_stacked_gpu = bn_backend_layout_upload_stacked3_qkv(
            gpu, &lw->attn.wq, &lw->attn.wk, &lw->attn.wv,
            lw->attn.q_bias, lw->attn.k_bias, lw->attn.v_bias,
            lw->attn.q_bias && !q_bias_gpu,
            lw->attn.k_bias && !k_bias_gpu,
            lw->attn.v_bias && !v_bias_gpu);
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_QKV_STACKED,
                                qkv_stacked_gpu) != 0) {
            if (qkv_stacked_gpu) gpu->buffer_destroy(gpu->ctx, qkv_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *qk_stacked_gpu =
            bn_backend_layout_upload_stacked2(gpu, &lw->attn.wq, &lw->attn.wk);
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_QK_STACKED,
                                qk_stacked_gpu) != 0) {
            if (qk_stacked_gpu) gpu->buffer_destroy(gpu->ctx, qk_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *gateup_stacked_gpu =
            bn_backend_layout_upload_stacked2(gpu, &lw->ffn.ffn_gate, &lw->ffn.ffn_up);
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_GATEUP_STACKED,
                                gateup_stacked_gpu) != 0) {
            if (gateup_stacked_gpu) gpu->buffer_destroy(gpu->ctx, gateup_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *shared_gateup_stacked_gpu =
            bn_backend_layout_upload_stacked2(
                gpu, &lw->shared.shared_gate, &lw->shared.shared_up);
        if (register_gpu_handle(model, l,
                                BN_BACKEND_HANDLE_SHARED_GATEUP_STACKED,
                                shared_gateup_stacked_gpu) != 0) {
            if (shared_gateup_stacked_gpu)
                gpu->buffer_destroy(gpu->ctx, shared_gateup_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *ssm_qkvz_stacked_gpu =
            bn_backend_layout_upload_stacked2(gpu, &lw->ssm.wqkv, &lw->ssm.wz);
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_QKVZ_STACKED,
                                ssm_qkvz_stacked_gpu) != 0) {
            if (ssm_qkvz_stacked_gpu) gpu->buffer_destroy(gpu->ctx, ssm_qkvz_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *ssm_ab_stacked_gpu =
            bn_backend_layout_upload_stacked2(gpu, &lw->ssm.ssm_alpha, &lw->ssm.ssm_beta);
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
