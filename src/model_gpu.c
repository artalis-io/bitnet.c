#include "model.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "gpu_backend.h"
#include <stdint.h>

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

BnGPUBackend *bn_model_gpu(const BnModel *model) {
    return model ? bn_backend_model_gpu(model->backend) : NULL;
}

void bn_model_set_gpu_disabled(BnModel *model, int disabled) {
    if (!model) return;
    bn_backend_model_set_gpu_disabled(model->backend, disabled);
}

static void *upload_qweight(BnGPUBackend *gpu, BnQWeight *w) {
    if (!w->data) return NULL;
    size_t sz = bn_qweight_data_size(w);
    if (sz == 0) return NULL;
    return gpu->buffer_create(gpu->ctx, w->data, sz, w->type, w->rows, w->cols);
}

static int upload_qweight_owned(BnBackendModel *backend, BnGPUBackend *gpu, BnQWeight *w) {
    void *handle = upload_qweight(gpu, w);
    if (!w->data) return 0;
    if (!handle) return -1;
    if (bn_backend_model_register_qweight(backend, w, handle) != 0) {
        gpu->buffer_destroy(gpu->ctx, handle);
        return -1;
    }
    return 0;
}

static void *upload_f32_buf(BnGPUBackend *gpu, const float *data, int n_elems) {
    if (!data || n_elems <= 0) return NULL;
    return gpu->buffer_create(gpu->ctx, data, (size_t)n_elems * sizeof(float),
                              -1, n_elems, 1);
}

static int register_gpu_handle(BnModel *model, int layer,
                               BnBackendHandleRole role, void *handle) {
    if (!handle) return 0;
    return bn_backend_model_register_handle(model->backend, layer, role, handle);
}

int bn_model_upload_weights(BnModel *model, BnGPUBackend *gpu) {
    if (!model || !gpu || !gpu->buffer_create) return -1;
    if (!model->backend) {
        model->backend = bn_backend_model_create();
        if (!model->backend) return -1;
    }
    bn_backend_model_bind_gpu(model->backend, gpu);

    BnWeights *w = &model->weights;
    BnConfig *c = &model->config;
    int n_layers = c->n_layers;

    if (upload_qweight_owned(model->backend, gpu, &w->output_weight) != 0) {
        bn_model_release_gpu(model);
        return -1;
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
            &lw->wq, &lw->wk, &lw->wv, &lw->wo,
            &lw->ffn_gate, &lw->ffn_up, &lw->ffn_down,
            &lw->wqkv, &lw->wz,
            &lw->ssm_alpha, &lw->ssm_beta, &lw->ssm_out,
            &lw->shared_gate, &lw->shared_up, &lw->shared_down,
        };
        int n_weights = (int)(sizeof(weights) / sizeof(weights[0]));
        for (int i = 0; i < n_weights; i++) {
            if (upload_qweight_owned(model->backend, gpu, weights[i]) != 0) {
                bn_model_release_gpu(model);
                return -1;
            }
        }

        void *attn_norm_gpu = upload_f32_buf(gpu, lw->attn_norm, c->dim);
        void *ffn_norm_gpu  = upload_f32_buf(gpu, lw->ffn_norm, c->dim);
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_ATTN_NORM,
                                attn_norm_gpu) != 0 ||
            register_gpu_handle(model, l, BN_BACKEND_HANDLE_FFN_NORM,
                                ffn_norm_gpu) != 0) {
            if (attn_norm_gpu) gpu->buffer_destroy(gpu->ctx, attn_norm_gpu);
            if (ffn_norm_gpu) gpu->buffer_destroy(gpu->ctx, ffn_norm_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *q_bias_gpu = NULL;
        void *k_bias_gpu = NULL;
        void *v_bias_gpu = NULL;
        if (gpu->buffer_create_biased) {
            struct { BnQWeight *w; float *bias; void **bias_gpu; } qkv_bias[] = {
                { &lw->wq, lw->q_bias, &q_bias_gpu },
                { &lw->wk, lw->k_bias, &k_bias_gpu },
                { &lw->wv, lw->v_bias, &v_bias_gpu },
            };
            for (int i = 0; i < 3; i++) {
                void *old_handle = bn_backend_model_qweight_buf(model->backend, qkv_bias[i].w);
                if (!qkv_bias[i].bias || !old_handle) continue;
                void *fused = bn_backend_layout_upload_biased_qweight(
                    gpu, qkv_bias[i].w, qkv_bias[i].bias);
                if (fused) {
                    gpu->buffer_destroy(gpu->ctx, old_handle);
                    if (bn_backend_model_register_qweight(model->backend,
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
            if (lw->q_bias)
                q_bias_gpu = upload_f32_buf(gpu, lw->q_bias, c->dim);
            if (lw->k_bias)
                k_bias_gpu = upload_f32_buf(gpu, lw->k_bias, c->kv_dim);
            if (lw->v_bias)
                v_bias_gpu = upload_f32_buf(gpu, lw->v_bias, c->kv_dim);
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
            gpu, &lw->wq, &lw->wk, &lw->wv,
            lw->q_bias, lw->k_bias, lw->v_bias,
            lw->q_bias && !q_bias_gpu,
            lw->k_bias && !k_bias_gpu,
            lw->v_bias && !v_bias_gpu);
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_QKV_STACKED,
                                qkv_stacked_gpu) != 0) {
            if (qkv_stacked_gpu) gpu->buffer_destroy(gpu->ctx, qkv_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *gateup_stacked_gpu =
            bn_backend_layout_upload_stacked2(gpu, &lw->ffn_gate, &lw->ffn_up);
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_GATEUP_STACKED,
                                gateup_stacked_gpu) != 0) {
            if (gateup_stacked_gpu) gpu->buffer_destroy(gpu->ctx, gateup_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *ssm_qkvz_stacked_gpu =
            bn_backend_layout_upload_stacked2(gpu, &lw->wqkv, &lw->wz);
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_QKVZ_STACKED,
                                ssm_qkvz_stacked_gpu) != 0) {
            if (ssm_qkvz_stacked_gpu) gpu->buffer_destroy(gpu->ctx, ssm_qkvz_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        void *ssm_ab_stacked_gpu =
            bn_backend_layout_upload_stacked2(gpu, &lw->ssm_alpha, &lw->ssm_beta);
        if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_AB_STACKED,
                                ssm_ab_stacked_gpu) != 0) {
            if (ssm_ab_stacked_gpu) gpu->buffer_destroy(gpu->ctx, ssm_ab_stacked_gpu);
            bn_model_release_gpu(model);
            return -1;
        }

        if (lw->q_norm) {
            int q_norm_size = c->qk_norm_per_head ? (c->n_heads * c->head_size) : c->head_size;
            void *q_norm_gpu = upload_f32_buf(gpu, lw->q_norm, q_norm_size);
            if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_Q_NORM,
                                    q_norm_gpu) != 0) {
                if (q_norm_gpu) gpu->buffer_destroy(gpu->ctx, q_norm_gpu);
                bn_model_release_gpu(model);
                return -1;
            }
        }
        if (lw->k_norm) {
            int k_norm_size = c->qk_norm_per_head ? c->kv_dim : c->head_size;
            void *k_norm_gpu = upload_f32_buf(gpu, lw->k_norm, k_norm_size);
            if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_K_NORM,
                                    k_norm_gpu) != 0) {
                if (k_norm_gpu) gpu->buffer_destroy(gpu->ctx, k_norm_gpu);
                bn_model_release_gpu(model);
                return -1;
            }
        }
        if (lw->attn_sub_norm) {
            void *attn_sub_norm_gpu = upload_f32_buf(gpu, lw->attn_sub_norm, c->dim);
            if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_ATTN_SUB_NORM,
                                    attn_sub_norm_gpu) != 0) {
                if (attn_sub_norm_gpu) gpu->buffer_destroy(gpu->ctx, attn_sub_norm_gpu);
                bn_model_release_gpu(model);
                return -1;
            }
        }
        if (lw->ffn_sub_norm) {
            void *ffn_sub_norm_gpu = upload_f32_buf(gpu, lw->ffn_sub_norm, c->hidden_dim);
            if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_FFN_SUB_NORM,
                                    ffn_sub_norm_gpu) != 0) {
                if (ffn_sub_norm_gpu) gpu->buffer_destroy(gpu->ctx, ffn_sub_norm_gpu);
                bn_model_release_gpu(model);
                return -1;
            }
        }

        if (lw->ssm_conv1d) {
            int num_v_heads = c->ssm_time_step_rank;
            int head_k_dim  = c->ssm_state_size;
            int key_dim     = c->ssm_group_count * head_k_dim;
            int qkv_dim     = key_dim * 2 + c->ssm_inner_size;
            int kern        = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;
            void *ssm_conv1d_gpu = upload_f32_buf(gpu, lw->ssm_conv1d, kern * qkv_dim);
            if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_CONV1D,
                                    ssm_conv1d_gpu) != 0) {
                if (ssm_conv1d_gpu) gpu->buffer_destroy(gpu->ctx, ssm_conv1d_gpu);
                bn_model_release_gpu(model);
                return -1;
            }

            if (lw->ssm_dt_bias && num_v_heads > 0) {
                void *ssm_dt_bias_gpu = upload_f32_buf(gpu, lw->ssm_dt_bias, num_v_heads);
                if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_DT_BIAS,
                                        ssm_dt_bias_gpu) != 0) {
                    if (ssm_dt_bias_gpu) gpu->buffer_destroy(gpu->ctx, ssm_dt_bias_gpu);
                    bn_model_release_gpu(model);
                    return -1;
                }
            }
            if (lw->ssm_a && num_v_heads > 0) {
                void *ssm_a_log_gpu = upload_f32_buf(gpu, lw->ssm_a, num_v_heads);
                if (register_gpu_handle(model, l, BN_BACKEND_HANDLE_SSM_A_LOG,
                                        ssm_a_log_gpu) != 0) {
                    if (ssm_a_log_gpu) gpu->buffer_destroy(gpu->ctx, ssm_a_log_gpu);
                    bn_model_release_gpu(model);
                    return -1;
                }
            }

            if (lw->ssm_norm) {
                int head_v_dim = num_v_heads > 0
                    ? c->ssm_inner_size / num_v_heads : c->ssm_inner_size;
                void *ssm_norm_gpu = upload_f32_buf(gpu, lw->ssm_norm, head_v_dim);
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
    BnBackendModel *backend = model->backend;

    if (backend)
        bn_backend_model_release_gpu(backend);
}
