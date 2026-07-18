#include "transformer_internal.h"
#include "transformer_cpu_internal.h"
#include "transformer_batched_attn_internal.h"
#include "gpu_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_prefill_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "backend_model.h"
#include "model_arch.h"
#include "moe.h"
#include "session.h"
#include "sh_arena.h"
#include "sh_log.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>

#define BN_MAX_VLA_ELEMS 8192

typedef struct {
    int enabled;
    double embed_ms;
    double attn_norm_ms;
    double qkv_ms;
    double attn_cpu_ms;
    double wo_ms;
    double ffn_norm_ms;
    double ffn_ms;
    double ffn_gateup_ms;
    double ffn_act_ms;
    double ffn_down_ms;
    double residual_ms;
    double logits_ms;
} BnPrefillProfile;

static inline double prefill_profile_now(const BnPrefillProfile *p) {
    return p && p->enabled ? bn_platform_time_ms() : 0.0;
}

static inline void prefill_profile_add(double *dst, double start) {
    if (start > 0.0)
        *dst += bn_platform_time_ms() - start;
}

static inline void *prefill_qweight_backend_buf(const BnBackendModel *backend,
                                                const BnQWeight *w) {
    return bn_backend_model_qweight_buf(backend, w);
}

static inline void *prefill_backend_role_or_qweight(
        const BnBackendModel *backend, int layer, BnBackendHandleRole role,
        const BnQWeight *w) {
    void *buf = backend ? bn_backend_model_handle(backend, layer, role) : NULL;
    return buf ? buf : prefill_qweight_backend_buf(backend, w);
}

static float *prefill_decode_tokens(BnModel *m, BnSession *sess,
                                    const int *tokens, int n_tokens,
                                    int pos0, float *all_logits,
                                    int need_last_logits) {
    BnConfig *c = &m->config;
    float *logits = NULL;
    if (all_logits) {
        for (int t = 0; t < n_tokens; t++) {
            logits = bn_transformer_forward(m, sess, tokens[t], pos0 + t);
            if (!logits)
                return NULL;
            memcpy(all_logits + (size_t)t * c->vocab_size, logits,
                   (size_t)c->vocab_size * sizeof(float));
        }
        if (bn_model_gpu(m))
            sess->gpu_kv_direct_valid = 1;
        return logits;
    }
    if (need_last_logits) {
        for (int t = 0; t + 1 < n_tokens; t++)
            if (bn_transformer_forward_no_logits(
                    m, sess, tokens[t], pos0 + t) != 0)
                return NULL;
        logits = bn_transformer_forward(m, sess, tokens[n_tokens - 1],
                                        pos0 + n_tokens - 1);
        if (logits && bn_model_gpu(m))
            sess->gpu_kv_direct_valid = 1;
        return logits;
    }
    for (int t = 0; t < n_tokens; t++)
        if (bn_transformer_forward_no_logits(m, sess, tokens[t],
                                             pos0 + t) != 0)
            return NULL;
    if (bn_model_gpu(m))
        sess->gpu_kv_direct_valid = 1;
    return sess->state.x;
}

static int prefill_force_float_kquant(const BnModel *m) {
    return m && bn_transformer_cpu_prefill_force_float_kquant_enabled(
        &m->config);
}

static int prefill_qweight_is_kquant(const BnQWeight *w) {
    return w && bn_transformer_prefill_uses_float_kquant_fallback(w->type);
}

static int prefill_all_kquant(const BnQWeight *const *W, int n) {
    if (n <= 0)
        return 0;
    for (int i = 0; i < n; i++) {
        if (!prefill_qweight_is_kquant(W[i]))
            return 0;
    }
    return 1;
}

static void prefill_quant_matmul_forced_kquant(
        const BnModel *m, float **out, const BnQWeight **W, int n,
        const float *X, int n_tokens, int8_t *x_q_buf) {
    const BnBackendModel *backend = bn_model_backend(m);
    BnMatvecTask tasks[4];
    if (n > 4)
        return;
    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < n; i++) {
            tasks[i] = (BnMatvecTask){
                out[i] + (size_t)t * W[i]->rows,
                W[i],
                bn_backend_model_prepared_qweight(backend, W[i]),
                BN_MATVEC_TASK_FORCE_FLOAT_KQUANT
            };
        }
        bn_quant_matvec_batch(tasks, n, X + (size_t)t * W[0]->cols,
                              x_q_buf, bn_model_pool(m));
    }
}

static void prefill_quant_matmul_gpu(const BnModel *m,
                                     float *out,
                                     const BnQWeight *W,
                                     const float *X,
                                     int n_tokens,
                                     int8_t *x_q_buf) {
    if (!bn_model_gpu(m)) {
        if (prefill_force_float_kquant(m) && prefill_qweight_is_kquant(W)) {
            float *outs[1] = { out };
            const BnQWeight *weights[1] = { W };
            prefill_quant_matmul_forced_kquant(
                m, outs, weights, 1, X, n_tokens, x_q_buf);
            return;
        }
        const BnBackendModel *backend = bn_model_backend(m);
        bn_quant_matmul_prepared(out, W,
                                 bn_backend_model_prepared_qweight(backend, W),
                                 X, n_tokens, x_q_buf, bn_model_pool(m));
        return;
    }
    bn_transformer_prefill_quant_matmul_gpu_buffer(
        out, W, prefill_qweight_backend_buf(bn_model_backend(m), W), X,
        n_tokens, x_q_buf, bn_model_pool(m), bn_model_gpu(m));
}

static int prefill_quant_matmul_gpu_buf(const BnModel *m,
                                        float *out,
                                        const BnQWeight *W,
                                        void *buf,
                                        const float *X,
                                        int n_tokens) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    if (!gpu || !gpu->matmul || !out || !W || !buf || !X)
        return -1;
    return gpu->matmul(gpu->ctx, out, buf, X, W->rows, W->cols,
                       n_tokens, W->type);
}

static void prefill_quant_matmul_multi(const BnModel *m,
                                       float **out,
                                       const BnQWeight **W,
                                       int n,
                                       const float *X,
                                       int n_tokens,
                                       int8_t *x_q_buf) {
    if (!bn_model_gpu(m)) {
        if (n <= 4 && prefill_force_float_kquant(m) &&
            prefill_all_kquant(W, n)) {
            prefill_quant_matmul_forced_kquant(
                m, out, W, n, X, n_tokens, x_q_buf);
            return;
        }
        const BnBackendModel *backend = bn_model_backend(m);
        const BnPreparedWeight *prepared[4] = { NULL, NULL, NULL, NULL };
        if (n > 4) {
            for (int i = 0; i < n; i++)
                prefill_quant_matmul_gpu(m, out[i], W[i], X, n_tokens, x_q_buf);
            return;
        }
        for (int i = 0; i < n; i++)
            prepared[i] = bn_backend_model_prepared_qweight(backend, W[i]);
        bn_quant_matmul_prepared_multi(out, W, prepared, n, X, n_tokens,
                                       x_q_buf, bn_model_pool(m));
        return;
    }
    if (n > 1 && n <= 16 && bn_model_gpu(m)->matmul_batch) {
        const BnBackendModel *backend = bn_model_backend(m);
        BnMatvecTask tasks[16];
        const void *bufs[16];
        int all_bufs = backend != NULL;
        for (int i = 0; i < n; i++) {
            tasks[i] = (BnMatvecTask){ out[i], W[i], NULL, 0 };
            bufs[i] = backend ? prefill_qweight_backend_buf(backend, W[i]) : NULL;
            if (!bufs[i]) all_bufs = 0;
        }
        if (all_bufs) {
            bn_transformer_prefill_quant_matmul_batch_gpu_buffers(
                tasks, bufs, n, X, n_tokens, W[0]->cols, x_q_buf,
                bn_model_pool(m), bn_model_gpu(m));
            return;
        }
    }
    for (int i = 0; i < n; i++)
        prefill_quant_matmul_gpu(m, out[i], W[i], X, n_tokens, x_q_buf);
}

static int prefill_qk_stacked_gpu(const BnModel *m,
                                  const BnLayerWeights *lw,
                                  float *q_tmp,
                                  float *k_out,
                                  const float *X,
                                  int n_tokens,
                                  int q_stride,
                                  int dim,
                                  int layer) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    if (!gpu || !gpu->matmul || !backend || !lw || !q_tmp || !k_out || !X)
        return -1;
    if (!bn_transformer_prefill_stacked_pair_same_format(
            lw->attn.wq.type, lw->attn.wk.type) ||
        lw->attn.wq.cols != dim || lw->attn.wk.cols != dim ||
        q_stride < lw->attn.wq.rows + lw->attn.wk.rows)
        return -1;
    void *qk_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_QK_STACKED);
    if (!qk_buf) return -1;
    int rows = lw->attn.wq.rows + lw->attn.wk.rows;
    if (gpu->matmul(gpu->ctx, q_tmp, qk_buf, X, rows, dim, n_tokens,
                    lw->attn.wq.type) != 0)
        return -1;
    for (int t = n_tokens - 1; t >= 0; t--) {
        float *src = q_tmp + (size_t)t * rows;
        memcpy(k_out + (size_t)t * lw->attn.wk.rows,
               src + lw->attn.wq.rows,
               (size_t)lw->attn.wk.rows * sizeof(float));
        memmove(q_tmp + (size_t)t * q_stride, src,
                (size_t)lw->attn.wq.rows * sizeof(float));
    }
    return 0;
}

static int prefill_qkv_stacked_batch_gpu(const BnModel *m,
                                         const BnLayerWeights *lw,
                                         float *q_tmp,
                                         float *k_out,
                                         float *v_out,
                                         const float *X,
                                         int n_tokens,
                                         int q_stride,
                                         int dim,
                                         int layer) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    if (!gpu || !gpu->matmul_batch || !backend || !lw || !q_tmp ||
        !k_out || !v_out || !X)
        return -1;
    if (!bn_transformer_prefill_stacked_pair_same_format(
            lw->attn.wq.type, lw->attn.wk.type) ||
        lw->attn.wq.cols != dim || lw->attn.wk.cols != dim ||
        lw->attn.wv.cols != dim ||
        q_stride < lw->attn.wq.rows + lw->attn.wk.rows)
        return -1;
    void *qk_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_QK_STACKED);
    void *wv_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_WV_PREFILL);
    if (!qk_buf || !wv_buf) return -1;

    int qk_rows = lw->attn.wq.rows + lw->attn.wk.rows;
    BnGPUMatvecOp ops[2] = {
        {
            .out = q_tmp,
            .W_buf = qk_buf,
            .rows = qk_rows,
            .cols = dim,
            .type = lw->attn.wq.type,
        },
        {
            .out = v_out,
            .W_buf = wv_buf,
            .rows = lw->attn.wv.rows,
            .cols = dim,
            .type = lw->attn.wv.type,
        },
    };
    if (gpu->matmul_batch(gpu->ctx, ops, 2, X, n_tokens, dim) != 0)
        return -1;

    for (int t = n_tokens - 1; t >= 0; t--) {
        float *src = q_tmp + (size_t)t * qk_rows;
        memcpy(k_out + (size_t)t * lw->attn.wk.rows,
               src + lw->attn.wq.rows,
               (size_t)lw->attn.wk.rows * sizeof(float));
        memmove(q_tmp + (size_t)t * q_stride, src,
                (size_t)lw->attn.wq.rows * sizeof(float));
    }
    return 0;
}

static int prefill_dense_ffn_gpu_batch(const BnModel *m,
                                       float *out,
                                       const BnLayerWeights *lw,
                                       const float *X,
                                       int n_tokens,
                                       int dim,
                                       int hidden_dim,
                                       int act_type,
                                       int layer,
                                       void *norm_buf,
                                       float norm_eps,
                                       int add_residual) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    if (!gpu || !gpu->dense_ffn_batch || !backend ||
        !lw->ffn.ffn_gate.data || !lw->ffn.ffn_up.data ||
        !lw->ffn.ffn_down.data)
        return -1;

    void *gateup_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_GATEUP_STACKED);
    void *gate_buf = NULL;
    void *up_buf = NULL;
    if (gateup_buf &&
        bn_transformer_prefill_stacked_pair_same_format(
            lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type)) {
        gate_buf = gateup_buf;
    } else {
        gate_buf = prefill_qweight_backend_buf(backend, &lw->ffn.ffn_gate);
        up_buf = prefill_qweight_backend_buf(backend, &lw->ffn.ffn_up);
    }
    void *down_buf = prefill_qweight_backend_buf(backend, &lw->ffn.ffn_down);
    if (!down_buf)
        down_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_FFN_DOWN_PREFILL);
    if (!gate_buf || !down_buf)
        return -1;

    if (norm_buf && add_residual && gpu->dense_ffn_batch_norm_resid) {
        return gpu->dense_ffn_batch_norm_resid(
            gpu->ctx, out, gate_buf, up_buf, down_buf, norm_buf, X,
            n_tokens, dim, hidden_dim, lw->ffn.ffn_gate.type,
            lw->ffn.ffn_up.type, lw->ffn.ffn_down.type, act_type,
            norm_eps);
    }
    if (norm_buf && gpu->dense_ffn_batch_norm) {
        return gpu->dense_ffn_batch_norm(
            gpu->ctx, out, gate_buf, up_buf, down_buf, norm_buf, X,
            n_tokens, dim, hidden_dim, lw->ffn.ffn_gate.type,
            lw->ffn.ffn_up.type, lw->ffn.ffn_down.type, act_type,
            norm_eps);
    }

    return gpu->dense_ffn_batch(
        gpu->ctx, out, gate_buf, up_buf, down_buf, X, n_tokens,
        dim, hidden_dim, lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type,
        lw->ffn.ffn_down.type, act_type);
}

static int prefill_dense_layer_gpu_batch(const BnModel *m,
                                         float *out,
                                         const BnLayerWeights *lw,
                                         const float *X,
                                         float *K_out,
                                         float *V_out,
                                         int n_tokens,
                                         int dim,
                                         int hidden_dim,
                                         int n_heads,
                                         int n_kv_heads,
                                         int head_size,
                                         int kv_mul,
                                         int kv_dim,
                                         int rope_dims,
                                         int layer,
                                         int pos0,
                                         uint32_t kv_cache_off,
                                         int kv_cache_stride,
                                         float attention_scale) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    int q_dim = n_heads * head_size;
    int has_split_qkv =
        lw->attn.wq.data && lw->attn.wk.data && lw->attn.wv.data;
    int has_packed_qkv =
        lw->ssm.wqkv.data && lw->ssm.wqkv.cols == dim &&
        lw->ssm.wqkv.rows == q_dim + 2 * kv_dim;
    if (!gpu || !gpu->prefill_dense_layer || !backend ||
        (!has_split_qkv && !has_packed_qkv) ||
        !lw->attn.wo.data || !lw->ffn.ffn_gate.data ||
        !lw->ffn.ffn_up.data || !lw->ffn.ffn_down.data)
        return -1;

    void *qk_buf = NULL;
    void *wv_buf = NULL;
    int qk_rows = 0;
    int qk_type = 0;
    int wv_rows = 0;
    int wv_type = 0;
    if (has_packed_qkv) {
        qk_buf = prefill_qweight_backend_buf(backend, &lw->ssm.wqkv);
        qk_rows = lw->ssm.wqkv.rows;
        qk_type = lw->ssm.wqkv.type;
        wv_rows = 0;
        wv_type = qk_type;
    } else {
        qk_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_QK_STACKED);
        wv_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_WV_PREFILL);
        if (!wv_buf)
            wv_buf = prefill_qweight_backend_buf(backend, &lw->attn.wv);
        qk_rows = lw->attn.wq.rows + lw->attn.wk.rows;
        qk_type = lw->attn.wq.type;
        wv_rows = lw->attn.wv.rows;
        wv_type = lw->attn.wv.type;
    }
    void *wo_buf = prefill_backend_role_or_qweight(
        backend, layer, BN_BACKEND_HANDLE_WO_PREFILL, &lw->attn.wo);
    void *gateup_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_GATEUP_STACKED);
    void *gate_buf = NULL;
    void *up_buf = NULL;
    if (gateup_buf &&
        bn_transformer_prefill_stacked_pair_same_format(
            lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type)) {
        gate_buf = gateup_buf;
    } else {
        gate_buf = prefill_qweight_backend_buf(backend, &lw->ffn.ffn_gate);
        up_buf = prefill_qweight_backend_buf(backend, &lw->ffn.ffn_up);
    }
    void *down_buf = prefill_qweight_backend_buf(backend, &lw->ffn.ffn_down);
    if (!down_buf)
        down_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_FFN_DOWN_PREFILL);
    void *attn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_ATTN_NORM);
    void *ffn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_FFN_NORM);
    void *q_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_Q_NORM);
    void *k_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_K_NORM);
    void *q_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_Q_BIAS);
    void *k_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_K_BIAS);
    void *v_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_V_BIAS);
    if (!qk_buf || (!has_packed_qkv && !wv_buf) || !wo_buf || !gate_buf || !down_buf ||
        !attn_norm_buf || !ffn_norm_buf ||
        (lw->attn.q_bias && !q_bias_buf) ||
        (lw->attn.k_bias && !k_bias_buf) ||
        (lw->attn.v_bias && !v_bias_buf))
        return -1;

    return gpu->prefill_dense_layer(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, gate_buf, up_buf, down_buf,
        attn_norm_buf, ffn_norm_buf, q_norm_buf, k_norm_buf,
        q_bias_buf, k_bias_buf, v_bias_buf, X, K_out, V_out, n_tokens, dim,
        hidden_dim, n_heads, n_kv_heads, head_size, kv_mul, kv_dim, qk_rows,
        qk_type, wv_rows, wv_type, lw->attn.wo.rows, lw->attn.wo.cols,
        lw->attn.wo.type, lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type,
        lw->ffn.ffn_down.type, m->config.act_type,
        m->config.qk_norm_per_head, m->config.norm_eps, pos0, rope_dims,
        kv_cache_off, kv_cache_stride, attention_scale);
}

static int prefill_moe_layer_gpu_batch(const BnModel *m,
                                       float *out,
                                       const BnLayerWeights *lw,
                                       const float *X,
                                       float *K_out,
                                       float *V_out,
                                       int n_tokens,
                                       int dim,
                                       int moe_hidden_dim,
                                       int n_heads,
                                       int n_kv_heads,
                                       int head_size,
                                       int kv_mul,
                                       int kv_dim,
                                       int rope_dims,
                                       int layer,
                                       int pos0,
                                       uint32_t kv_cache_off,
                                       int kv_cache_stride,
                                       float attention_scale) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    if (!bn_transformer_gpu_prefill_moe_layer_backend_available(
            gpu, &m->config, &lw->moe.expert_map, dim, 0) ||
        !backend ||
        !lw->moe.router_weight || !lw->attn.wq.data ||
        !lw->attn.wk.data || !lw->attn.wv.data || !lw->attn.wo.data)
        return -1;

    void *qk_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_QK_STACKED);
    void *wv_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_WV_PREFILL);
    if (!wv_buf)
        wv_buf = prefill_qweight_backend_buf(backend, &lw->attn.wv);
    void *wo_buf = prefill_backend_role_or_qweight(
        backend, layer, BN_BACKEND_HANDLE_WO_PREFILL, &lw->attn.wo);
    void *router_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER);
    void *gate_all_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_GATE_ALL);
    void *up_all_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_UP_ALL);
    void *down_all_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
    void *shared_gate_buf = NULL;
    void *shared_up_buf = NULL;
    void *shared_down_buf = NULL;
    void *shared_gate_weight_buf = NULL;
    int shared_hidden_dim = 0;
    int shared_gate_type = 0;
    int shared_up_type = 0;
    int shared_down_type = 0;
    if (m->config.has_shared_expert && lw->shared.shared_gate.data) {
        shared_gate_buf =
            prefill_qweight_backend_buf(backend, &lw->shared.shared_gate);
        shared_up_buf =
            prefill_qweight_backend_buf(backend, &lw->shared.shared_up);
        shared_down_buf =
            prefill_qweight_backend_buf(backend, &lw->shared.shared_down);
        shared_gate_weight_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_SHARED_EXPERT_GATE);
        shared_hidden_dim = m->config.shared_expert_intermediate_size;
        shared_gate_type = lw->shared.shared_gate.type;
        shared_up_type = lw->shared.shared_up.type;
        shared_down_type = lw->shared.shared_down.type;
        if (!shared_gate_buf || !shared_up_buf || !shared_down_buf)
            return -1;
    }
    void *attn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_ATTN_NORM);
    void *ffn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_FFN_NORM);
    void *q_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_Q_NORM);
    void *k_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_K_NORM);
    void *q_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_Q_BIAS);
    void *k_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_K_BIAS);
    void *v_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_V_BIAS);
    if (!qk_buf || !wv_buf || !wo_buf || !router_buf || !gate_all_buf ||
        !up_all_buf || !down_all_buf || !attn_norm_buf || !ffn_norm_buf ||
        (lw->attn.q_bias && !q_bias_buf) ||
        (lw->attn.k_bias && !k_bias_buf) ||
        (lw->attn.v_bias && !v_bias_buf))
        return -1;

    return gpu->prefill_moe_layer(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, router_buf, gate_all_buf,
        up_all_buf, down_all_buf, shared_gate_buf, shared_up_buf,
        shared_down_buf, shared_gate_weight_buf, attn_norm_buf, ffn_norm_buf,
        q_norm_buf, k_norm_buf, q_bias_buf, k_bias_buf, v_bias_buf,
        X, K_out, V_out, n_tokens, dim, moe_hidden_dim,
        m->config.n_experts, m->config.n_experts_active, n_heads,
        n_kv_heads, head_size, kv_mul, kv_dim,
        lw->attn.wq.rows + lw->attn.wk.rows, lw->attn.wq.type,
        lw->attn.wv.rows, lw->attn.wv.type, lw->attn.wo.rows,
        lw->attn.wo.cols, lw->attn.wo.type,
        lw->moe.expert_map.gate_type, lw->moe.expert_map.up_type,
        lw->moe.expert_map.down_type, m->config.act_type,
        shared_hidden_dim, shared_gate_type, shared_up_type,
        shared_down_type,
        m->config.qk_norm_per_head, m->config.norm_eps, pos0, rope_dims,
        kv_cache_off, kv_cache_stride,
        attention_scale, m->config.moe_norm_topk_prob,
        m->config.moe_expert_weights_scale);
}

static int prefill_moe_ffn_gpu_batch(const BnModel *m,
                                     float *out,
                                     const BnLayerWeights *lw,
                                     const float *X,
                                     int n_tokens,
                                     int dim,
                                     int layer) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    const BnConfig *c = &m->config;
    if (!bn_transformer_gpu_cuda_prefill_moe_ffn_batch_available(
            gpu, c, &lw->moe.expert_map, dim, 0) ||
        !backend ||
        !lw->moe.router_weight || n_tokens <= 0 || dim <= 0)
        return -1;

    void *router_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER);
    void *gate_all_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_GATE_ALL);
    void *up_all_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_UP_ALL);
    void *down_all_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
    void *ffn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_FFN_NORM);
    if (!router_buf || !gate_all_buf || !up_all_buf || !down_all_buf ||
        !ffn_norm_buf)
        return -1;

    void *shared_gate_buf = NULL;
    void *shared_up_buf = NULL;
    void *shared_down_buf = NULL;
    void *shared_gate_weight_buf = NULL;
    int shared_hidden_dim = 0;
    int shared_gate_type = 0;
    int shared_up_type = 0;
    int shared_down_type = 0;
    if (c->has_shared_expert && lw->shared.shared_gate.data) {
        shared_gate_buf =
            prefill_qweight_backend_buf(backend, &lw->shared.shared_gate);
        shared_up_buf =
            prefill_qweight_backend_buf(backend, &lw->shared.shared_up);
        shared_down_buf =
            prefill_qweight_backend_buf(backend, &lw->shared.shared_down);
        shared_gate_weight_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_SHARED_EXPERT_GATE);
        if (!shared_gate_buf || !shared_up_buf || !shared_down_buf)
            return -1;
        shared_hidden_dim = c->shared_expert_intermediate_size;
        shared_gate_type = lw->shared.shared_gate.type;
        shared_up_type = lw->shared.shared_up.type;
        shared_down_type = lw->shared.shared_down.type;
    }

    return gpu->moe_route_routed_ffn_batch_norm_resid(
        gpu->ctx, out, router_buf, gate_all_buf, up_all_buf, down_all_buf,
        shared_gate_buf, shared_up_buf, shared_down_buf,
        shared_gate_weight_buf, ffn_norm_buf, X, n_tokens, dim,
        c->moe_intermediate_size, c->n_experts, c->n_experts_active,
        lw->moe.expert_map.gate_type, lw->moe.expert_map.up_type,
        lw->moe.expert_map.down_type, c->act_type,
        shared_hidden_dim, shared_gate_type, shared_up_type,
        shared_down_type, c->norm_eps, c->moe_norm_topk_prob,
        c->moe_expert_weights_scale);
}

static int prefill_ssm_moe_layer_chain_ready(const BnModel *m,
                                             const BnLayerWeights *lw,
                                             int layer,
                                             int n_tokens) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    const BnConfig *c = &m->config;
    if (!bn_transformer_gpu_cuda_prefill_ssm_moe_chain_available(
            gpu, c, lw ? &lw->moe.expert_map : NULL, c ? c->dim : 0, 0,
            n_tokens) ||
        !backend ||
        !lw || !lw->moe.router_weight ||
        !lw->ssm.wqkv.data || !lw->ssm.wz.data ||
        !lw->ssm.ssm_alpha.data || !lw->ssm.ssm_beta.data ||
        !lw->ssm.ssm_out.data || !lw->norm.attn_norm ||
        !lw->ssm.ssm_conv1d || !lw->ssm.ssm_dt_bias ||
        !lw->ssm.ssm_a || !lw->ssm.ssm_norm ||
        lw->norm.ffn_sub_norm || lw->norm.layer_output_scale ||
        (bn_transformer_attention_uses_post_norm(c) &&
         (lw->norm.attn_post_norm || lw->norm.ffn_post_norm)) ||
        c->ssm_time_step_rank <= 0 || c->ssm_state_size <= 0 ||
        c->ssm_inner_size <= 0 || c->ssm_group_count <= 0)
        return 0;

    void *wqkv_buf = prefill_qweight_backend_buf(backend, &lw->ssm.wqkv);
    void *wz_buf = prefill_qweight_backend_buf(backend, &lw->ssm.wz);
    void *alpha_buf = prefill_qweight_backend_buf(backend,
                                                  &lw->ssm.ssm_alpha);
    void *beta_buf = prefill_qweight_backend_buf(backend,
                                                 &lw->ssm.ssm_beta);
    void *out_buf = prefill_qweight_backend_buf(backend, &lw->ssm.ssm_out);
    void *attn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_ATTN_NORM);
    void *conv1d_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_CONV1D);
    void *dt_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_DT_BIAS);
    void *a_log_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_A_LOG);
    void *ssm_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_NORM);
    void *router_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER);
    void *gate_all_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_GATE_ALL);
    void *up_all_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_UP_ALL);
    void *down_all_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
    void *ffn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_FFN_NORM);

    if (!(wqkv_buf && wz_buf && alpha_buf && beta_buf && out_buf &&
          attn_norm_buf && conv1d_buf && dt_bias_buf && a_log_buf &&
          ssm_norm_buf && router_buf && gate_all_buf && up_all_buf &&
          down_all_buf && ffn_norm_buf))
        return 0;

    if (c->has_shared_expert && lw->shared.shared_gate.data) {
        if (!prefill_qweight_backend_buf(backend, &lw->shared.shared_gate) ||
            !prefill_qweight_backend_buf(backend, &lw->shared.shared_up) ||
            !prefill_qweight_backend_buf(backend, &lw->shared.shared_down))
            return 0;
    }
    return 1;
}

static int prefill_dense_layer_chain_ready(const BnModel *m,
                                           const BnLayerWeights *lw,
                                           const BnLayerShapePlan *plan,
                                           int layer,
                                           int n_tokens,
                                           float layer_rope_theta) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    const BnConfig *c = &m->config;
    if (!gpu || !gpu->prefill_dense_layer || !backend ||
        bn_model_tq_state(m) != NULL ||
        n_tokens <
            bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(c, gpu) ||
        layer_rope_theta != c->rope_theta ||
        !plan->is_attn ||
        lw->moe.router_weight || !c->has_ffn_gate || !lw->ffn.ffn_up.data ||
        !lw->attn.wo.data || !lw->ffn.ffn_gate.data ||
        !lw->ffn.ffn_down.data ||
        lw->norm.attn_sub_norm || lw->norm.ffn_sub_norm ||
        lw->norm.layer_output_scale ||
        (bn_transformer_attention_uses_post_norm(c) &&
         (lw->norm.attn_post_norm || lw->norm.ffn_post_norm)))
        return 0;
    int q_dim = plan->q_dim > 0 ? plan->q_dim : c->n_heads * plan->head_size;
    int has_split_qkv =
        lw->attn.wq.data && lw->attn.wk.data && lw->attn.wv.data;
    int has_packed_qkv =
        lw->ssm.wqkv.data && lw->ssm.wqkv.cols == c->dim &&
        lw->ssm.wqkv.rows == q_dim + 2 * plan->kv_dim;
    if (!has_split_qkv && !has_packed_qkv)
        return 0;

    void *qk_buf = NULL;
    void *wv_buf = NULL;
    if (has_packed_qkv) {
        qk_buf = prefill_qweight_backend_buf(backend, &lw->ssm.wqkv);
    } else {
        qk_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_QK_STACKED);
        wv_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_WV_PREFILL);
        if (!wv_buf)
            wv_buf = prefill_qweight_backend_buf(backend, &lw->attn.wv);
    }
    void *wo_buf = prefill_backend_role_or_qweight(
        backend, layer, BN_BACKEND_HANDLE_WO_PREFILL, &lw->attn.wo);
    void *gateup_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_GATEUP_STACKED);
    void *gate_buf = gateup_buf &&
                     bn_transformer_prefill_stacked_pair_same_format(
                         lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type)
                         ? gateup_buf
                         : prefill_qweight_backend_buf(backend,
                                                       &lw->ffn.ffn_gate);
    void *up_buf = gateup_buf &&
                   bn_transformer_prefill_stacked_pair_same_format(
                       lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type)
                       ? NULL
                       : prefill_qweight_backend_buf(backend, &lw->ffn.ffn_up);
    void *down_buf = prefill_qweight_backend_buf(backend, &lw->ffn.ffn_down);
    if (!down_buf)
        down_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_FFN_DOWN_PREFILL);
    void *attn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_ATTN_NORM);
    void *ffn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_FFN_NORM);
    void *q_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_Q_BIAS);
    void *k_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_K_BIAS);
    void *v_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_V_BIAS);
    return qk_buf && (has_packed_qkv || wv_buf) && wo_buf &&
           gate_buf && (gateup_buf || up_buf) &&
           down_buf && attn_norm_buf && ffn_norm_buf &&
           (!lw->attn.q_bias || q_bias_buf) &&
           (!lw->attn.k_bias || k_bias_buf) &&
           (!lw->attn.v_bias || v_bias_buf);
}

static int prefill_moe_layer_chain_ready(const BnModel *m,
                                         const BnLayerWeights *lw,
                                         const BnLayerShapePlan *plan,
                                         int layer,
                                         int n_tokens,
                                         float layer_rope_theta) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    const BnConfig *c = &m->config;
    if (!bn_transformer_gpu_prefill_moe_layer_chain_available(
            gpu, c, lw ? &lw->moe.expert_map : NULL, c ? c->dim : 0, 0,
            n_tokens) ||
        !backend ||
        bn_model_tq_state(m) != NULL ||
        layer_rope_theta != c->rope_theta ||
        !plan->is_attn || !lw->moe.router_weight ||
        lw->norm.attn_sub_norm ||
        lw->norm.layer_output_scale) {
        if (bn_transformer_gpu_cuda_prefill_moe_chain_debug_enabled())
            fprintf(stderr,
                    "[bn:prefill:moe-chain] reject layer=%d basic gpu=%d hook=%d backend=%d tq=%d toks=%d min=%d theta=%d attn=%d moe=%d shared=%d bias=%d subnorm=%d scale=%d\n",
                    layer, gpu != NULL,
                    gpu && gpu->prefill_moe_layer ? 1 : 0,
                    backend != NULL, bn_model_tq_state(m) != NULL,
                    n_tokens,
                    bn_transformer_gpu_cuda_prefill_moe_chain_min_tokens(
                        c, gpu),
                    layer_rope_theta == c->rope_theta, plan->is_attn,
                    lw->moe.router_weight != NULL, c->has_shared_expert,
                    lw->attn.q_bias || lw->attn.k_bias || lw->attn.v_bias,
                    lw->norm.attn_sub_norm != NULL,
                    lw->norm.layer_output_scale != NULL);
        return 0;
    }
    if (!lw->attn.wq.data || !lw->attn.wk.data || !lw->attn.wv.data ||
        !lw->attn.wo.data)
        return 0;
    void *wv_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_WV_PREFILL);
    if (!wv_buf)
        wv_buf = prefill_qweight_backend_buf(backend, &lw->attn.wv);
    int ready =
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_QK_STACKED) &&
        wv_buf &&
        prefill_backend_role_or_qweight(
            backend, layer, BN_BACKEND_HANDLE_WO_PREFILL, &lw->attn.wo) &&
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER) &&
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_GATE_ALL) &&
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_UP_ALL) &&
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_DOWN_ALL) &&
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_ATTN_NORM) &&
        bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_FFN_NORM) &&
        (!lw->attn.q_bias ||
         bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_Q_BIAS)) &&
        (!lw->attn.k_bias ||
         bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_K_BIAS)) &&
        (!lw->attn.v_bias ||
         bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_V_BIAS));
    if (ready && c->has_shared_expert && lw->shared.shared_gate.data) {
        ready =
            prefill_qweight_backend_buf(backend, &lw->shared.shared_gate) &&
            prefill_qweight_backend_buf(backend, &lw->shared.shared_up) &&
            prefill_qweight_backend_buf(backend, &lw->shared.shared_down);
    }
    if (!ready && bn_transformer_gpu_cuda_prefill_moe_chain_debug_enabled())
        fprintf(stderr,
                "[bn:prefill:moe-chain] reject layer=%d handles qk=%d wv=%d wo=%d router=%d gate=%d up=%d down=%d anorm=%d fnorm=%d\n",
                layer,
                bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_QK_STACKED) != NULL,
                wv_buf != NULL,
                prefill_backend_role_or_qweight(
                    backend, layer, BN_BACKEND_HANDLE_WO_PREFILL,
                    &lw->attn.wo) != NULL,
                bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_ROUTER) != NULL,
                bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_GATE_ALL) != NULL,
                bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_UP_ALL) != NULL,
                bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_MOE_DOWN_ALL) != NULL,
                bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_ATTN_NORM) != NULL,
                bn_backend_model_handle(backend, layer, BN_BACKEND_HANDLE_FFN_NORM) != NULL);
    return ready;
}

static int prefill_ssm_layer_gpu(const BnModel *m,
                                 float *out,
                                 const BnLayerWeights *lw,
                                 const float *X,
                                 int n_tokens,
                                 int dim,
                                 int qkv_dim,
                                 int inner_dim,
                                 int num_k_heads,
                                 int head_k_dim,
                                 int num_v_heads,
                                 int head_v_dim,
                                 int conv_kernel,
                                 int ssm_idx,
                                 int layer,
                                 int fuse_ffn,
                                 float norm_eps,
                                 int *did_ffn) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    if (!bn_transformer_gpu_prefill_ssm_layer_backend_available(gpu) ||
        !backend || !lw ||
        !lw->ssm.wqkv.data || !lw->ssm.wz.data ||
        !lw->ssm.ssm_alpha.data || !lw->ssm.ssm_beta.data ||
        !lw->ssm.ssm_out.data || !lw->norm.attn_norm ||
        !lw->ssm.ssm_conv1d || !lw->ssm.ssm_dt_bias ||
        !lw->ssm.ssm_a || !lw->ssm.ssm_norm)
        return -1;
    void *wqkv_buf = prefill_qweight_backend_buf(backend, &lw->ssm.wqkv);
    void *wz_buf = prefill_qweight_backend_buf(backend, &lw->ssm.wz);
    void *alpha_buf = prefill_qweight_backend_buf(backend, &lw->ssm.ssm_alpha);
    void *beta_buf = prefill_qweight_backend_buf(backend, &lw->ssm.ssm_beta);
    void *qkvz_stacked_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_QKVZ_STACKED);
    void *ab_stacked_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_AB_STACKED);
    void *out_buf = prefill_qweight_backend_buf(backend, &lw->ssm.ssm_out);
    void *attn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_ATTN_NORM);
    void *conv1d_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_CONV1D);
    void *dt_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_DT_BIAS);
    void *a_log_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_A_LOG);
    void *ssm_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_NORM);
    void *gateup_buf = NULL;
    void *gate_buf = NULL;
    void *up_buf = NULL;
    void *down_buf = NULL;
    void *ffn_norm_buf = NULL;
    if (fuse_ffn &&
        bn_transformer_gpu_cuda_prefill_ssm_ffn_fuse_allowed() &&
        lw->ffn.ffn_gate.data && lw->ffn.ffn_up.data &&
        lw->ffn.ffn_down.data && m->config.has_ffn_gate &&
        !lw->norm.ffn_sub_norm &&
        !lw->norm.layer_output_scale &&
        !(bn_transformer_ffn_uses_post_norm(&m->config) &&
          lw->norm.ffn_post_norm)) {
        gateup_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_GATEUP_STACKED);
        if (gateup_buf &&
            bn_transformer_prefill_stacked_pair_same_format(
                lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type)) {
            gate_buf = gateup_buf;
        } else {
            gate_buf = prefill_qweight_backend_buf(backend,
                                                   &lw->ffn.ffn_gate);
            up_buf = prefill_qweight_backend_buf(backend, &lw->ffn.ffn_up);
        }
        down_buf = prefill_qweight_backend_buf(backend, &lw->ffn.ffn_down);
        if (!down_buf)
            down_buf = bn_backend_model_handle(
                backend, layer, BN_BACKEND_HANDLE_FFN_DOWN_PREFILL);
        ffn_norm_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_FFN_NORM);
        if (!gate_buf || !down_buf || !ffn_norm_buf) {
            gate_buf = NULL;
            up_buf = NULL;
            down_buf = NULL;
            ffn_norm_buf = NULL;
        }
    }
    if (!wqkv_buf || !wz_buf || !alpha_buf || !beta_buf || !out_buf ||
        !attn_norm_buf || !conv1d_buf || !dt_bias_buf || !a_log_buf ||
        !ssm_norm_buf)
        return -1;
    if (did_ffn)
        *did_ffn = 0;
    return gpu->prefill_ssm_layer(
        gpu->ctx, out, wqkv_buf, wz_buf, alpha_buf, beta_buf,
        qkvz_stacked_buf, ab_stacked_buf, out_buf, attn_norm_buf, conv1d_buf,
        dt_bias_buf, a_log_buf, ssm_norm_buf, gate_buf, up_buf, down_buf,
        ffn_norm_buf, X, n_tokens, dim, qkv_dim, inner_dim,
        num_k_heads, head_k_dim, num_v_heads, head_v_dim,
        conv_kernel, ssm_idx, lw->ssm.wqkv.type, lw->ssm.wz.type,
        lw->ssm.ssm_alpha.type, lw->ssm.ssm_beta.type, lw->ssm.ssm_out.type,
        lw->ffn.ffn_down.cols, lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type,
        lw->ffn.ffn_down.type, m->config.act_type, norm_eps, did_ffn);
}

static int prefill_ssm_layer_chain_ready(const BnModel *m,
                                         const BnLayerWeights *lw,
                                         int layer,
                                         int n_tokens) {
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    const BnConfig *c = &m->config;
    if (!bn_transformer_gpu_cuda_prefill_ssm_dense_chain_available(
            gpu, c, n_tokens) ||
        !backend ||
        !lw || !lw->ssm.wqkv.data || !lw->ssm.wz.data ||
        !lw->ssm.ssm_alpha.data || !lw->ssm.ssm_beta.data ||
        !lw->ssm.ssm_out.data || !lw->norm.attn_norm ||
        !lw->ssm.ssm_conv1d || !lw->ssm.ssm_dt_bias ||
        !lw->ssm.ssm_a || !lw->ssm.ssm_norm ||
        !lw->ffn.ffn_gate.data || !lw->ffn.ffn_up.data ||
        !lw->ffn.ffn_down.data || !c->has_ffn_gate ||
        lw->moe.router_weight || lw->norm.ffn_sub_norm ||
        lw->norm.layer_output_scale ||
        (bn_transformer_attention_uses_post_norm(c) &&
         (lw->norm.attn_post_norm || lw->norm.ffn_post_norm)) ||
        c->ssm_time_step_rank <= 0 || c->ssm_state_size <= 0 ||
        c->ssm_inner_size <= 0 || c->ssm_group_count <= 0)
        return 0;

    void *wqkv_buf = prefill_qweight_backend_buf(backend, &lw->ssm.wqkv);
    void *wz_buf = prefill_qweight_backend_buf(backend, &lw->ssm.wz);
    void *alpha_buf = prefill_qweight_backend_buf(backend,
                                                  &lw->ssm.ssm_alpha);
    void *beta_buf = prefill_qweight_backend_buf(backend,
                                                 &lw->ssm.ssm_beta);
    void *out_buf = prefill_qweight_backend_buf(backend, &lw->ssm.ssm_out);
    void *attn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_ATTN_NORM);
    void *conv1d_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_CONV1D);
    void *dt_bias_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_DT_BIAS);
    void *a_log_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_A_LOG);
    void *ssm_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_SSM_NORM);
    void *gateup_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_GATEUP_STACKED);
    void *gate_buf = gateup_buf &&
                     bn_transformer_prefill_stacked_pair_same_format(
                         lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type)
                         ? gateup_buf
                         : prefill_qweight_backend_buf(backend,
                                                       &lw->ffn.ffn_gate);
    void *up_buf = gateup_buf &&
                   bn_transformer_prefill_stacked_pair_same_format(
                       lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type)
                       ? NULL
                       : prefill_qweight_backend_buf(backend, &lw->ffn.ffn_up);
    void *down_buf = prefill_qweight_backend_buf(backend, &lw->ffn.ffn_down);
    if (!down_buf)
        down_buf = bn_backend_model_handle(
            backend, layer, BN_BACKEND_HANDLE_FFN_DOWN_PREFILL);
    void *ffn_norm_buf = bn_backend_model_handle(
        backend, layer, BN_BACKEND_HANDLE_FFN_NORM);

    return wqkv_buf && wz_buf && alpha_buf && beta_buf && out_buf &&
           attn_norm_buf && conv1d_buf && dt_bias_buf && a_log_buf &&
           ssm_norm_buf && gate_buf && (gateup_buf || up_buf) &&
           down_buf && ffn_norm_buf;
}

static void prefill_quant_matmul_preq8k_multi(const BnModel *m,
                                              float **out,
                                              const BnQWeight **W,
                                              int n,
                                              int n_tokens,
                                              const int8_t *x_q,
                                              const float *x_d,
                                              const int16_t *x_bsums,
                                              const float *x_float) {
    const BnBackendModel *backend = bn_model_backend(m);
    const BnPreparedWeight *prepared[4] = { NULL, NULL, NULL, NULL };
    if (n > 4) {
        bn_quant_matmul_preq8k_multi(out, W, NULL, n, n_tokens, x_q, x_d,
                                     x_bsums, x_float, bn_model_pool(m));
        return;
    }
    for (int i = 0; i < n; i++)
        prepared[i] = bn_backend_model_prepared_qweight(backend, W[i]);
    bn_quant_matmul_preq8k_multi(out, W, prepared, n, n_tokens, x_q, x_d,
                                 x_bsums, x_float, bn_model_pool(m));
}

typedef struct {
    int8_t *xq;
    float *xd;
    int16_t *xbs;
    int n_bpr;
} BnPrefillPreQ8KBuffers;

static const BnPrefillCPUOps *prefill_cpu_ops(void) {
    return bn_transformer_prefill_cpu_ops();
}

static size_t prefill_preq8k_arena_bytes(int dim, int n_tokens) {
    if (!prefill_cpu_ops()->supports_preq8k)
        return 0;
    if (dim <= 0 || n_tokens <= 0 || dim % BN_QK_K != 0)
        return 0;
    int n_bpr = dim / BN_QK_K;
    return (size_t)n_tokens * (size_t)dim +
           (size_t)n_tokens * (size_t)n_bpr * sizeof(float) +
           (size_t)n_tokens * (size_t)n_bpr * 16 * sizeof(int16_t);
}

static BnPrefillPreQ8KBuffers prefill_alloc_preq8k_buffers(SHArena *arena,
                                                           int dim,
                                                           int n_tokens) {
    BnPrefillPreQ8KBuffers b = { NULL, NULL, NULL, 0 };
    if (!prefill_cpu_ops()->supports_preq8k)
        return b;
    if (!arena || dim <= 0 || n_tokens <= 0 || dim % BN_QK_K != 0)
        return b;
    b.n_bpr = dim / BN_QK_K;
    b.xq = (int8_t *)sh_arena_alloc(arena, (size_t)n_tokens * (size_t)dim);
    b.xd = (float *)sh_arena_alloc(
        arena, (size_t)n_tokens * (size_t)b.n_bpr * sizeof(float));
    b.xbs = (int16_t *)sh_arena_alloc(
        arena, (size_t)n_tokens * (size_t)b.n_bpr * 16 * sizeof(int16_t));
    if (!b.xq || !b.xd || !b.xbs) {
        b.xq = NULL;
        b.xd = NULL;
        b.xbs = NULL;
        b.n_bpr = 0;
    }
    return b;
}

static int prefill_prepare_preq8k(BnPrefillPreQ8KBuffers *b,
                                  const float *x,
                                  int dim,
                                  int n_tokens) {
    const BnPrefillCPUOps *ops = prefill_cpu_ops();
    if (!b || !b->xq || !b->xd || !b->xbs || !x || dim <= 0 ||
        n_tokens <= 0 || b->n_bpr <= 0 || !ops->prepare_preq8k)
        return 0;
    return ops->prepare_preq8k(b->xq, b->xd, b->xbs, b->n_bpr,
                               x, dim, n_tokens);
}

static int prefill_try_preq8k_multi(const BnModel *m,
                                    BnPrefillPreQ8KBuffers *b,
                                    float **out,
                                    const BnQWeight **W,
                                    int n,
                                    const float *X,
                                    int dim,
                                    int n_tokens) {
    const BnPrefillCPUOps *ops = prefill_cpu_ops();
    if (!m || !b || !b->xq || !out || !W || !X || n <= 0 || n > 4)
        return 0;
    for (int i = 0; i < n; i++) {
        if (!bn_transformer_prefill_route_preq8k_type_enabled(
                ops, bn_model_gpu(m), prefill_force_float_kquant(m), dim,
                W[i]->type))
            return 0;
    }
    if (!prefill_prepare_preq8k(b, X, dim, n_tokens))
        return 0;
    prefill_quant_matmul_preq8k_multi(m, out, W, n, n_tokens,
                                      b->xq, b->xd, b->xbs, X);
    return 1;
}

static int prefill_try_preq8k_matvec_batch(const BnModel *m,
                                           BnPrefillPreQ8KBuffers *b,
                                           BnMatvecTask *tasks,
                                           int n,
                                           const float *x,
                                           int dim,
                                           int token_index) {
    const BnPrefillCPUOps *ops = prefill_cpu_ops();
    if (!m || !b || !b->xq || !tasks || !x || n <= 0 || token_index < 0)
        return 0;
    for (int i = 0; i < n; i++) {
        if (!bn_transformer_prefill_route_preq8k_type_enabled(
                ops, bn_model_gpu(m), prefill_force_float_kquant(m), dim,
                tasks[i].W->type))
            return 0;
    }
    bn_quant_matvec_batch_preq8k(
        tasks, n, b->xq + (size_t)token_index * dim,
        b->xd + (size_t)token_index * b->n_bpr,
        b->xbs + (size_t)token_index * b->n_bpr * 16,
        x, bn_model_pool(m));
    return 1;
}

static float *prefill_logits(BnModel *m, BnSession *sess) {
    return bn_transformer_forward_logits(m, sess);
}

static int prefill_use_shared_all2_decode_fallback(const BnModel *m) {
    if (!m)
        return 0;
    if (!bn_model_arch_uses_all_active_two_expert_moe(&m->config,
                                                      m->config.dim) ||
        !m->config.has_shared_expert)
        return 0;
    return bn_model_gpu(m) == NULL;
}

static int prefill_layer_rope_dims(const BnConfig *c, int layer_head_size) {
    int use_swa_rope = c->rope_theta_swa > 0.0f && layer_head_size < c->head_size;
    int base_rope_dims = c->rope_text_dims > 0
        ? c->rope_text_dims
        : (c->rope_dim_count > 0 ? c->rope_dim_count : layer_head_size);
    int rope_dims = use_swa_rope && c->rope_dim_count_swa > 0
        ? c->rope_dim_count_swa
        : base_rope_dims;
    return rope_dims > layer_head_size ? layer_head_size : rope_dims;
}

static float prefill_layer_rope_theta(const BnConfig *c, int layer_head_size) {
    return (c->rope_theta_swa > 0.0f && layer_head_size < c->head_size)
        ? c->rope_theta_swa : c->rope_theta;
}

static float prefill_attention_scale(const BnConfig *c, int head_size) {
    return bn_transformer_attention_scale(c, head_size);
}

static void prefill_rmsnorm_unit(float *out, const float *x, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++)
        ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)size + eps);
    for (int i = 0; i < size; i++)
        out[i] = x[i] * ss;
}

static void prefill_rmsnorm_unit_heads(float *x, int n_heads,
                                       int head_size, float eps) {
    for (int h = 0; h < n_heads; h++)
        prefill_rmsnorm_unit(x + h * head_size, x + h * head_size,
                             head_size, eps);
}

static void prefill_fill_rope(float *rope_cos_buf, float *rope_sin_buf,
                              int rope_stride, int n_tokens, int pos0,
                              int rope_dims, float theta) {
    int half_rope = rope_dims / 2;
    for (int t = 0; t < n_tokens; t++) {
        int pos = pos0 + t;
        for (int i = 0; i < half_rope; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)rope_dims);
            float angle = pos * freq;
            rope_cos_buf[(size_t)t * rope_stride + i] = cosf(angle);
            rope_sin_buf[(size_t)t * rope_stride + i] = sinf(angle);
        }
    }
}

static int prefill_prepare_q_for_gpu_attention(BnBatchedAttnCtx *b) {
    if (!b || b->q_gated || !b->Q_buf || !b->rope_cos || !b->rope_sin)
        return -1;
    if (b->pos0 != 0)
        return -1;
    int head_size = b->head_size;
    int n_heads = b->n_heads;
    int n_tokens = b->n_tokens;
    int q_row_stride = b->q_row_stride > 0 ? b->q_row_stride : b->wq_rows;
    int rope_stride = b->rope_stride > 0 ? b->rope_stride : b->rope_dims / 2;
    if (head_size <= 0 || n_heads <= 0 || n_tokens <= 1 ||
        q_row_stride < n_heads * head_size)
        return -1;

    for (int t = 0; t < n_tokens; t++) {
        float *row = b->Q_buf + (size_t)t * q_row_stride;
        if (b->q_bias) {
            for (int i = 0; i < n_heads * head_size; i++)
                row[i] += b->q_bias[i];
        }
        if (b->q_norm) {
            int stride = b->qk_norm_per_head ? head_size : 0;
            for (int h = 0; h < n_heads; h++)
                prefill_cpu_ops()->rmsnorm(row + h * head_size,
                                           row + h * head_size,
                                           b->q_norm + h * stride, head_size,
                                           b->norm_eps);
        }
        bn_transformer_cpu_apply_rope_heads(
            row, n_heads, head_size, b->rope_dims,
            b->rope_cos + (size_t)t * rope_stride,
            b->rope_sin + (size_t)t * rope_stride);
        if (q_row_stride != n_heads * head_size)
            memmove(b->Q_buf + (size_t)t * n_heads * head_size, row,
                    (size_t)n_heads * (size_t)head_size * sizeof(float));
    }
    return 0;
}

static float *prefill_internal(BnModel *m, BnSession *sess, const int *tokens,
                               int n_tokens, int pos0, float *all_logits,
                               int need_last_logits) {
    if (n_tokens <= 0) return NULL;
    if (sess) sess->gpu_kv_direct_valid = 0;
    if (n_tokens == 1) {
        float *logits = bn_transformer_forward(m, sess, tokens[0], pos0);
        return need_last_logits ? logits : (logits ? sess->state.x : NULL);
    }

    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    int dim = c->dim;
    int head_size = c->head_size;
    BnPrefillProfile prof = {0};
    prof.enabled = bn_transformer_prefill_profile_enabled();
    double t_prof = prefill_profile_now(&prof);

    if (head_size > BN_MAX_VLA_ELEMS || dim > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("Model dimensions too large for stack VLAs");
        return NULL;
    }

    for (int t = 0; t < n_tokens; t++) {
        if (tokens[t] < 0 || tokens[t] >= c->vocab_size) {
            SH_LOG_ERROR("Token out of range");
            return NULL;
        }
    }
    if (pos0 < 0) {
        SH_LOG_ERROR("Position out of range");
        return NULL;
    }

    BnGPUBackend *prefill_gpu = bn_model_gpu(m);
    int gpu_hybrid_prefill =
        bn_transformer_gpu_hybrid_prefill_chain_applicable(prefill_gpu, c);
    int gpu_moe_prefill =
        bn_transformer_gpu_moe_prefill_chain_applicable(prefill_gpu, c);
    int cuda_small_dense_prefill_chain =
        bn_transformer_gpu_cuda_small_dense_prefill_chain_applicable(
            prefill_gpu, c);
    if (gpu_moe_prefill &&
        (!bn_transformer_gpu_cuda_moe_prefill_enabled() ||
         n_tokens <
             bn_transformer_gpu_cuda_prefill_moe_chain_min_tokens(
                 c, prefill_gpu))) {
        return prefill_decode_tokens(m, sess, tokens, n_tokens, pos0,
                                     all_logits, need_last_logits);
    }
    if (cuda_small_dense_prefill_chain &&
        n_tokens <
            bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(
                c, prefill_gpu)) {
        return prefill_decode_tokens(m, sess, tokens, n_tokens, pos0,
                                     all_logits, need_last_logits);
    }
    if (gpu_hybrid_prefill &&
        bn_model_arch_uses_large_dense_hybrid_ssm(c) &&
        bn_transformer_gpu_cuda_large_hybrid_prefill_disabled()) {
        return prefill_decode_tokens(m, sess, tokens, n_tokens, pos0,
                                     all_logits, need_last_logits);
    }
    if (bn_model_arch_uses_hybrid_ssm(c) &&
        !gpu_hybrid_prefill &&
        !bn_transformer_prefill_hybrid_batch_allowed()) {
        float *logits = NULL;
        for (int t = 0; t < n_tokens; t++) {
            logits = bn_transformer_forward(m, sess, tokens[t], pos0 + t);
            if (!logits)
                return NULL;
            if (all_logits)
                memcpy(all_logits + (size_t)t * c->vocab_size, logits,
                       (size_t)c->vocab_size * sizeof(float));
        }
        return need_last_logits ? logits : (logits ? sess->state.x : NULL);
    }

    size_t act_elems = (size_t)n_tokens * dim;
    if (act_elems / n_tokens != (size_t)dim) {
        SH_LOG_ERROR("Prefill activation buffer size overflow");
        return NULL;
    }

    int rope_dims = c->rope_text_dims > 0
        ? c->rope_text_dims
        : (c->rope_dim_count > 0 ? c->rope_dim_count : head_size);
    int half_rope = rope_dims / 2;
    if (half_rope > BN_MAX_VLA_ELEMS) {
        SH_LOG_ERROR("RoPE dimensions too large for stack VLAs");
        return NULL;
    }

    int kv_dim = c->kv_dim;
    int hidden_dim = c->hidden_dim;
    int q_dim = c->n_heads * head_size;
    int q_buf_stride = (q_dim > dim ? q_dim * 2 : dim);
    int xb2_stride = dim;
    int hb_stride = hidden_dim;
    int hb2_stride = hidden_dim;
    if (bn_model_arch_uses_hybrid_ssm(c)) {
        int ssm_qkv_dim = c->ssm_group_count * c->ssm_state_size * 2 +
                          c->ssm_inner_size;
        if (ssm_qkv_dim > q_buf_stride)
            q_buf_stride = ssm_qkv_dim;
        if (c->ssm_inner_size > xb2_stride)
            xb2_stride = c->ssm_inner_size;
        if (c->ssm_inner_size > hb_stride)
            hb_stride = c->ssm_inner_size;
        if (c->ssm_inner_size > hb2_stride)
            hb2_stride = c->ssm_inner_size;
    }
    size_t nt = (size_t)n_tokens;

    size_t batch_floats = nt * dim
                        + nt * (size_t)q_buf_stride
                        + nt * kv_dim * 2
                        + nt * (size_t)xb2_stride
                        + nt * (size_t)hb_stride
                        + nt * (size_t)hb2_stride;
    size_t arena_size = act_elems * sizeof(float)
                      + batch_floats * sizeof(float)
                      + nt * half_rope * 2 * sizeof(float)
                      + 512;
    arena_size += prefill_preq8k_arena_bytes(dim, n_tokens);

    SHArena *pf_arena = sh_arena_create(arena_size);
    if (!pf_arena) return NULL;

    float *act = (float *)sh_arena_alloc(pf_arena, act_elems * sizeof(float));
    if (!act) { sh_arena_free(pf_arena); return NULL; }

    for (int t = 0; t < n_tokens; t++)
        bn_model_embed_token(m, act + (size_t)t * dim, tokens[t]);
    prefill_profile_add(&prof.embed_ms, t_prof);

    float *batch_buf = (float *)sh_arena_alloc(pf_arena, batch_floats * sizeof(float));
    if (!batch_buf) { sh_arena_free(pf_arena); return NULL; }

    BnPrefillPreQ8KBuffers preq8k =
        prefill_alloc_preq8k_buffers(pf_arena, dim, n_tokens);

    float *Xb = batch_buf;
    float *Q_buf = Xb + nt * dim;
    float *K_new = Q_buf + nt * q_buf_stride;
    float *V_new = K_new + nt * kv_dim;
    float *Xb2 = V_new + nt * kv_dim;
    float *Hb = Xb2 + nt * xb2_stride;
    float *Hb2 = Hb + nt * hb_stride;

    float *rope_cos_buf = (float *)sh_arena_alloc(pf_arena, nt * half_rope * sizeof(float));
    float *rope_sin_buf = (float *)sh_arena_alloc(pf_arena, nt * half_rope * sizeof(float));
    if (!rope_cos_buf || !rope_sin_buf) { sh_arena_free(pf_arena); return NULL; }

    BnWeights *w = &m->weights;
    int rope_cache_dims = -1;
    float rope_cache_theta = 0.0f;

    if (bn_transformer_gpu_cuda_prefill_dense_chain_enabled() &&
        bn_model_gpu(m) && pos0 == 0 && c->n_layers > 0) {
        int chain_ready = 1;
        for (int l = 0; l < c->n_layers; l++) {
            BnLayerWeights *lw = &w->layers[l];
            BnLayerShapePlan plan;
            bn_transformer_plan_layer_shape(
                &plan, c, lw, l, bn_model_tq_state(m) != NULL);
            int layer_head_size = plan.head_size;
            float layer_rope_theta =
                prefill_layer_rope_theta(c, layer_head_size);
            if (!prefill_moe_layer_chain_ready(
                    m, lw, &plan, l, n_tokens, layer_rope_theta) &&
                !prefill_dense_layer_chain_ready(
                    m, lw, &plan, l, n_tokens, layer_rope_theta)) {
                chain_ready = 0;
                break;
            }
        }
        if (chain_ready) {
            int direct_gpu_kv =
                bn_transformer_gpu_cuda_prefill_direct_kv_allowed(
                    c, w, bn_model_gpu(m), pos0, n_tokens);
            sess->gpu_kv_direct_valid = 0;
            t_prof = prefill_profile_now(&prof);
            for (int l = 0; l < c->n_layers; l++) {
                BnLayerWeights *lw = &w->layers[l];
                BnLayerShapePlan plan;
                bn_transformer_plan_layer_shape(
                    &plan, c, lw, l, bn_model_tq_state(m) != NULL);
                int layer_head_size = plan.head_size;
                int layer_kv_dim = plan.kv_dim;
                int layer_n_kv_heads = plan.n_kv_heads;
                int layer_kv_mul = plan.kv_mul;
                int layer_rope_dims =
                    prefill_layer_rope_dims(c, layer_head_size);
                float layer_rope_theta =
                    prefill_layer_rope_theta(c, layer_head_size);
                if (rope_cache_dims != layer_rope_dims ||
                    rope_cache_theta != layer_rope_theta) {
                    prefill_fill_rope(rope_cos_buf, rope_sin_buf, half_rope,
                                      n_tokens, pos0, layer_rope_dims,
                                      layer_rope_theta);
                    rope_cache_dims = layer_rope_dims;
                    rope_cache_theta = layer_rope_theta;
                }
                float *layer_out = (l == c->n_layers - 1) ? act : NULL;
                const float *layer_in = (l == 0) ? act : NULL;
                size_t loff = (size_t)plan.attn_idx * c->seq_len * kv_dim;
                uint32_t kv_cache_off =
                    (uint32_t)(loff + (size_t)pos0 * (size_t)kv_dim);
                BnTransformerPrefillLayerKindPolicy layer_kind =
                    bn_transformer_prefill_layer_kind_policy(
                        lw->moe.router_weight);
                int layer_rc = layer_kind.uses_moe
                    ? prefill_moe_layer_gpu_batch(
                          m, layer_out, lw, layer_in,
                          direct_gpu_kv ? NULL : K_new,
                          direct_gpu_kv ? NULL : V_new,
                          n_tokens, dim, c->moe_intermediate_size,
                          c->n_heads, layer_n_kv_heads, layer_head_size,
                          layer_kv_mul, layer_kv_dim, layer_rope_dims,
                          l, pos0, kv_cache_off, kv_dim,
                          prefill_attention_scale(c, layer_head_size))
                    : prefill_dense_layer_gpu_batch(
                          m, layer_out, lw, layer_in,
                          direct_gpu_kv ? NULL : K_new,
                          direct_gpu_kv ? NULL : V_new, n_tokens,
                          dim, hidden_dim, c->n_heads, layer_n_kv_heads,
                          layer_head_size, layer_kv_mul, layer_kv_dim,
                          layer_rope_dims, l, pos0, kv_cache_off, kv_dim,
                          prefill_attention_scale(c, layer_head_size));
                if (layer_rc != 0) {
                    sh_arena_free(pf_arena);
                    return NULL;
                }
                if (!direct_gpu_kv) {
                    for (int t = 0; t < n_tokens; t++) {
                        int pos = pos0 + t;
                        int cache_pos = pos % c->seq_len;
                        float *k_t = K_new + (size_t)t * layer_kv_dim;
                        float *v_t = V_new + (size_t)t * layer_kv_dim;
                        if (c->kv_f16)
                            bn_transformer_write_kv_fp16(s, loff, cache_pos,
                                                         kv_dim, k_t, v_t,
                                                         layer_kv_dim);
                        else
                            bn_transformer_write_kv_fp32(s, loff, cache_pos,
                                                         kv_dim, k_t, v_t,
                                                         layer_kv_dim);
                    }
                }
            }
            if (direct_gpu_kv)
                sess->gpu_kv_direct_valid = 1;
            prefill_profile_add(&prof.qkv_ms, t_prof);
            goto prefill_layers_done;
        }
    }

    if (bn_transformer_gpu_cuda_prefill_hybrid_chain_enabled(prefill_gpu, c) &&
        gpu_hybrid_prefill && pos0 == 0 && c->n_layers > 0 &&
        bn_model_tq_state(m) == NULL) {
        int chain_ready = 1;
        for (int l = 0; l < c->n_layers; l++) {
            BnLayerWeights *lw = &w->layers[l];
            BnLayerShapePlan plan;
            bn_transformer_plan_layer_shape(
                &plan, c, lw, l, bn_model_tq_state(m) != NULL);
            if (plan.is_attn) {
                int layer_head_size = plan.head_size;
                float layer_rope_theta =
                    prefill_layer_rope_theta(c, layer_head_size);
                if (!prefill_moe_layer_chain_ready(
                        m, lw, &plan, l, n_tokens, layer_rope_theta) &&
                    !prefill_dense_layer_chain_ready(
                        m, lw, &plan, l, n_tokens, layer_rope_theta)) {
                    if (bn_transformer_gpu_cuda_prefill_hybrid_chain_debug_enabled())
                        fprintf(stderr,
                                "[bn:prefill:hybrid-chain] reject attn layer=%d\n",
                                l);
                    chain_ready = 0;
                    break;
                }
            } else if (!prefill_ssm_moe_layer_chain_ready(m, lw, l, n_tokens) &&
                       !prefill_ssm_layer_chain_ready(m, lw, l, n_tokens)) {
                if (bn_transformer_gpu_cuda_prefill_hybrid_chain_debug_enabled())
                    fprintf(stderr,
                            "[bn:prefill:hybrid-chain] reject ssm layer=%d\n",
                            l);
                chain_ready = 0;
                break;
            }
        }
        if (chain_ready) {
            int direct_gpu_kv =
                bn_transformer_gpu_cuda_prefill_direct_kv_allowed(
                    c, w, bn_model_gpu(m), pos0, n_tokens);
            sess->gpu_kv_direct_valid = 0;
            t_prof = prefill_profile_now(&prof);
            for (int l = 0; l < c->n_layers; l++) {
                BnLayerWeights *lw = &w->layers[l];
                BnLayerShapePlan plan;
                bn_transformer_plan_layer_shape(
                    &plan, c, lw, l, bn_model_tq_state(m) != NULL);
                float *layer_out = (l == c->n_layers - 1) ? act : NULL;
                const float *layer_in = (l == 0) ? act : NULL;
                if (plan.is_attn) {
                    int layer_head_size = plan.head_size;
                    int layer_kv_dim = plan.kv_dim;
                    int layer_n_kv_heads = plan.n_kv_heads;
                    int layer_kv_mul = plan.kv_mul;
                    int layer_rope_dims =
                        prefill_layer_rope_dims(c, layer_head_size);
                    size_t loff =
                        (size_t)plan.attn_idx * c->seq_len * kv_dim;
                    uint32_t kv_cache_off =
                        (uint32_t)(loff + (size_t)pos0 * (size_t)kv_dim);
                    BnTransformerPrefillLayerKindPolicy layer_kind =
                        bn_transformer_prefill_layer_kind_policy(
                            lw->moe.router_weight);
                    int layer_rc = layer_kind.uses_moe
                        ? prefill_moe_layer_gpu_batch(
                              m, layer_out, lw, layer_in,
                              direct_gpu_kv ? NULL : K_new,
                              direct_gpu_kv ? NULL : V_new,
                              n_tokens, dim, c->moe_intermediate_size,
                              c->n_heads, layer_n_kv_heads, layer_head_size,
                              layer_kv_mul, layer_kv_dim, layer_rope_dims, l,
                              pos0, kv_cache_off, kv_dim,
                              prefill_attention_scale(c, layer_head_size))
                        : prefill_dense_layer_gpu_batch(
                              m, layer_out, lw, layer_in,
                              direct_gpu_kv ? NULL : K_new,
                              direct_gpu_kv ? NULL : V_new, n_tokens,
                              dim, hidden_dim, c->n_heads, layer_n_kv_heads,
                              layer_head_size, layer_kv_mul, layer_kv_dim,
                              layer_rope_dims, l, pos0, kv_cache_off, kv_dim,
                              prefill_attention_scale(c, layer_head_size));
                    if (layer_rc != 0) {
                        sh_arena_free(pf_arena);
                        return NULL;
                    }
                    if (!direct_gpu_kv) {
                        for (int t = 0; t < n_tokens; t++) {
                            int pos = pos0 + t;
                            int cache_pos = pos % c->seq_len;
                            float *k_t = K_new + (size_t)t * layer_kv_dim;
                            float *v_t = V_new + (size_t)t * layer_kv_dim;
                            if (c->kv_f16)
                                bn_transformer_write_kv_fp16(
                                    s, loff, cache_pos, kv_dim, k_t, v_t,
                                    layer_kv_dim);
                            else
                                bn_transformer_write_kv_fp32(
                                    s, loff, cache_pos, kv_dim, k_t, v_t,
                                    layer_kv_dim);
                        }
                    }
                } else {
                    int num_k_heads = c->ssm_group_count;
                    int head_k_dim = c->ssm_state_size;
                    int num_v_heads = c->ssm_time_step_rank;
                    int head_v_dim =
                        c->ssm_inner_size /
                        (num_v_heads > 0 ? num_v_heads : 1);
                    int key_dim_ssm = num_k_heads * head_k_dim;
                    int value_dim = c->ssm_inner_size;
                    int qkv_dim_ssm = key_dim_ssm * 2 + value_dim;
                    int kern_ssm =
                        c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;
                    BnTransformerPrefillLayerKindPolicy layer_kind =
                        bn_transformer_prefill_layer_kind_policy(
                            lw->moe.router_weight);
                    int r_is_moe = layer_kind.uses_moe;
                    float *ssm_out =
                        (!r_is_moe && l == c->n_layers - 1) ? act : NULL;
                    int ssm_did_ffn = 0;
                    if (prefill_ssm_layer_gpu(
                            m, ssm_out, lw, layer_in, n_tokens, dim,
                            qkv_dim_ssm, value_dim, num_k_heads,
                            head_k_dim, num_v_heads, head_v_dim, kern_ssm,
                            plan.ssm_idx, l, !r_is_moe, c->norm_eps,
                            &ssm_did_ffn) != 0 || (!r_is_moe && !ssm_did_ffn)) {
                        sh_arena_free(pf_arena);
                        return NULL;
                    }
                    if (r_is_moe &&
                        prefill_moe_ffn_gpu_batch(
                            m, layer_out, lw, NULL, n_tokens, dim, l) != 0) {
                        sh_arena_free(pf_arena);
                        return NULL;
                    }
                }
            }
            if (direct_gpu_kv)
                sess->gpu_kv_direct_valid = 1;
            prefill_profile_add(&prof.qkv_ms, t_prof);
            goto prefill_layers_done;
        }
    }

    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        BnLayerShapePlan plan;
        bn_transformer_plan_layer_shape(&plan, c, lw, l, bn_model_tq_state(m) != NULL);
        int is_attn = plan.is_attn;

        if (is_attn && lw->attn.wq.data) {
            int layer_head_size = plan.head_size;
            int layer_kv_dim = plan.kv_dim;
            int layer_n_kv_heads = plan.n_kv_heads;
            int layer_kv_mul = plan.kv_mul;
            int layer_q_dim = plan.q_dim;
            int layer_rope_dims = prefill_layer_rope_dims(c, layer_head_size);
            float layer_rope_theta = prefill_layer_rope_theta(c, layer_head_size);
            if (rope_cache_dims != layer_rope_dims ||
                rope_cache_theta != layer_rope_theta) {
                prefill_fill_rope(rope_cos_buf, rope_sin_buf, half_rope, n_tokens, pos0,
                                  layer_rope_dims, layer_rope_theta);
                rope_cache_dims = layer_rope_dims;
                rope_cache_theta = layer_rope_theta;
            }
            BnTransformerPrefillLayerKindPolicy layer_kind =
                bn_transformer_prefill_layer_kind_policy(
                    lw->moe.router_weight);
            BnTransformerPrefillDenseLayerBatchPolicy dense_layer_batch =
                bn_transformer_prefill_dense_layer_batch_policy(
                    bn_model_gpu(m) != NULL,
                    bn_model_tq_state(m) != NULL,
                    bn_transformer_gpu_cuda_prefill_dense_chain_enabled(),
                    n_tokens,
                    bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(
                        c, bn_model_gpu(m)),
                    pos0, layer_rope_theta, c->rope_theta, layer_kind,
                    c->has_ffn_gate, lw->ffn.ffn_up.data != NULL,
                    lw->attn.q_bias != NULL, lw->attn.k_bias != NULL,
                    lw->attn.v_bias != NULL,
                    lw->norm.attn_sub_norm != NULL,
                    lw->norm.ffn_sub_norm != NULL,
                    lw->norm.layer_output_scale != NULL,
                    bn_transformer_attention_uses_post_norm(c),
                    lw->norm.attn_post_norm != NULL,
                    lw->norm.ffn_post_norm != NULL);
            if (dense_layer_batch.enabled) {
                t_prof = prefill_profile_now(&prof);
                if (prefill_dense_layer_gpu_batch(
                        m, act, lw, act, K_new, V_new, n_tokens, dim,
                        hidden_dim, c->n_heads, layer_n_kv_heads,
                        layer_head_size, layer_kv_mul, layer_kv_dim,
                        layer_rope_dims, l, pos0,
                        (uint32_t)((size_t)plan.attn_idx * c->seq_len *
                                   kv_dim + (size_t)pos0 * kv_dim),
                        kv_dim,
                        prefill_attention_scale(c, layer_head_size)) == 0) {
                    prefill_profile_add(&prof.qkv_ms, t_prof);
                    int attn_idx = plan.attn_idx;
                    size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;
                    for (int t = 0; t < n_tokens; t++) {
                        int pos = pos0 + t;
                        int cache_pos = pos % c->seq_len;
                        float *k_t = K_new + (size_t)t * layer_kv_dim;
                        float *v_t = V_new + (size_t)t * layer_kv_dim;
                        if (c->kv_f16)
                            bn_transformer_write_kv_fp16(s, loff, cache_pos,
                                                         kv_dim, k_t, v_t,
                                                         layer_kv_dim);
                        else
                            bn_transformer_write_kv_fp32(s, loff, cache_pos,
                                                         kv_dim, k_t, v_t,
                                                         layer_kv_dim);
                    }
                    continue;
                }
            }
            int q_read_stride = lw->attn.wq.rows;
            int attn_idx = plan.attn_idx;
            size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;
            int q_gated = plan.q_gated;
            int wo_cols_attn = lw->attn.wo.cols;
            int used_fused_attn_wo = 0;
            int used_raw_prefill_attn_wo = 0;
            int used_attn_residual = 0;
            BnGPUBackend *gpu = bn_model_gpu(m);
            const BnBackendModel *backend = bn_model_backend(m);
            void *attn_norm_buf = backend ? bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_ATTN_NORM) : NULL;
            BnTransformerPrefillRawAttentionPolicy raw_attn_policy =
                bn_transformer_prefill_raw_attention_policy(
                    gpu != NULL,
                    gpu && gpu->prefill_qkv_attention_wo,
                    gpu && gpu->prefill_qkv_attention_wo_norm_resid,
                    attn_norm_buf != NULL,
                    bn_model_tq_state(m) != NULL,
                    q_gated,
                    pos0,
                    n_tokens,
                    bn_transformer_gpu_cuda_prefill_attention_min_tokens(),
                    layer_rope_theta,
                    c->rope_theta,
                    lw->attn.q_bias != NULL,
                    lw->attn.k_bias != NULL,
                    lw->attn.v_bias != NULL,
                    lw->norm.attn_sub_norm != NULL,
                    bn_transformer_attention_uses_post_norm(c),
                    lw->norm.attn_post_norm != NULL);
            int attn_norm_ready = 0;
            if (!raw_attn_policy.fuses_input_norm) {
                t_prof = prefill_profile_now(&prof);
                for (int t = 0; t < n_tokens; t++)
                    prefill_cpu_ops()->rmsnorm(Xb + t * dim,
                                               act + (size_t)t * dim,
                                               lw->norm.attn_norm, dim,
                                               c->norm_eps);
                prefill_profile_add(&prof.attn_norm_ms, t_prof);
                attn_norm_ready = 1;
            }
            if (raw_attn_policy.eligible) {
                void *qk_buf = backend ? bn_backend_model_handle(
                    backend, l, BN_BACKEND_HANDLE_QK_STACKED) : NULL;
                void *wv_buf = backend ? bn_backend_model_handle(
                    backend, l, BN_BACKEND_HANDLE_WV_PREFILL) : NULL;
                if (!wv_buf)
                    wv_buf = prefill_qweight_backend_buf(backend,
                                                         &lw->attn.wv);
                void *wo_buf = prefill_backend_role_or_qweight(
                    backend, l, BN_BACKEND_HANDLE_WO_PREFILL, &lw->attn.wo);
                void *q_norm_buf = backend ? bn_backend_model_handle(
                    backend, l, BN_BACKEND_HANDLE_Q_NORM) : NULL;
                void *k_norm_buf = backend ? bn_backend_model_handle(
                    backend, l, BN_BACKEND_HANDLE_K_NORM) : NULL;
                t_prof = prefill_profile_now(&prof);
                int qkv_fused_rc = -1;
                if (qk_buf && wv_buf && wo_buf &&
                    raw_attn_policy.fuses_input_norm) {
                    qkv_fused_rc = gpu->prefill_qkv_attention_wo_norm_resid(
                        gpu->ctx, act, qk_buf, wv_buf, wo_buf,
                        attn_norm_buf, q_norm_buf, k_norm_buf, act, K_new,
                        V_new, n_tokens, dim, c->n_heads,
                        layer_n_kv_heads, layer_head_size, layer_kv_mul,
                        kv_dim, lw->attn.wq.rows + lw->attn.wk.rows,
                        lw->attn.wq.type, lw->attn.wv.rows,
                        lw->attn.wv.type, lw->attn.wo.rows,
                        lw->attn.wo.cols, lw->attn.wo.type,
                        c->qk_norm_per_head, c->norm_eps, pos0,
                        layer_rope_dims,
                        prefill_attention_scale(c, layer_head_size));
                    if (qkv_fused_rc == 0)
                        used_attn_residual = 1;
                    if (qkv_fused_rc != 0) {
                        t_prof = prefill_profile_now(&prof);
                        for (int t = 0; t < n_tokens; t++)
                            prefill_cpu_ops()->rmsnorm(
                                Xb + t * dim, act + (size_t)t * dim,
                                lw->norm.attn_norm, dim, c->norm_eps);
                        prefill_profile_add(&prof.attn_norm_ms, t_prof);
                        attn_norm_ready = 1;
                        t_prof = prefill_profile_now(&prof);
                    }
                }
                if (qkv_fused_rc != 0 && qk_buf && wv_buf && wo_buf) {
                    qkv_fused_rc = gpu->prefill_qkv_attention_wo(
                        gpu->ctx, Xb2, qk_buf, wv_buf, wo_buf,
                        q_norm_buf, k_norm_buf, Xb, K_new, V_new,
                        n_tokens, dim, c->n_heads, layer_n_kv_heads,
                        layer_head_size, layer_kv_mul, kv_dim,
                        lw->attn.wq.rows + lw->attn.wk.rows,
                        lw->attn.wq.type, lw->attn.wv.rows,
                        lw->attn.wv.type, lw->attn.wo.rows,
                        lw->attn.wo.cols, lw->attn.wo.type,
                        c->qk_norm_per_head, c->norm_eps, pos0,
                        layer_rope_dims,
                        prefill_attention_scale(c, layer_head_size));
                }
                if (qkv_fused_rc == 0) {
                    used_raw_prefill_attn_wo = 1;
                    used_fused_attn_wo = 1;
                    prefill_profile_add(&prof.qkv_ms, t_prof);
                }
            }
            if (!used_raw_prefill_attn_wo) {
                if (!attn_norm_ready) {
                    t_prof = prefill_profile_now(&prof);
                    for (int t = 0; t < n_tokens; t++)
                        prefill_cpu_ops()->rmsnorm(Xb + t * dim,
                                                   act + (size_t)t * dim,
                                                   lw->norm.attn_norm, dim,
                                                   c->norm_eps);
                    prefill_profile_add(&prof.attn_norm_ms, t_prof);
                    attn_norm_ready = 1;
                }
            {
                t_prof = prefill_profile_now(&prof);
                float *qkv_out[3] = { Q_buf, K_new, V_new };
                const BnQWeight *qkv_w[3] = {
                    &lw->attn.wq, &lw->attn.wk, &lw->attn.wv
                };
                int used_preq8k =
                    bn_transformer_prefill_route_preq8k_triple_enabled(
                        prefill_cpu_ops(), bn_model_gpu(m),
                        prefill_force_float_kquant(m), dim,
                        lw->attn.wq.type, lw->attn.wk.type,
                        lw->attn.wv.type) &&
                    prefill_try_preq8k_multi(m, &preq8k, qkv_out, qkv_w,
                                             3, Xb, dim, n_tokens);
                if (used_preq8k) {
                    /* handled by preq8k helper */
                } else if (prefill_qkv_stacked_batch_gpu(
                        m, lw, Q_buf, K_new, V_new, Xb, n_tokens,
                        q_buf_stride, dim, l) == 0) {
                    q_read_stride = q_buf_stride;
                } else if (prefill_qk_stacked_gpu(m, lw, Q_buf, K_new, Xb,
                                                  n_tokens, q_buf_stride, dim,
                                                  l) == 0) {
                    q_read_stride = q_buf_stride;
                    void *wv_buf = backend ? bn_backend_model_handle(
                        backend, l, BN_BACKEND_HANDLE_WV_PREFILL) : NULL;
                    if (prefill_quant_matmul_gpu_buf(
                            m, V_new, &lw->attn.wv, wv_buf, Xb,
                            n_tokens) != 0)
                        prefill_quant_matmul_gpu(m, V_new, &lw->attn.wv,
                                                 Xb, n_tokens, s->x_q);
                } else {
                    prefill_quant_matmul_multi(m, qkv_out, qkv_w, 3, Xb,
                                               n_tokens, s->x_q);
                }
                prefill_profile_add(&prof.qkv_ms, t_prof);
            }
            }

            int force_token_attn =
                bn_transformer_prefill_force_token_attention_enabled() ||
                gpu_hybrid_prefill;
            if (bn_model_tq_state(m) == NULL && !force_token_attn) {
                // Phase 1: prepare K/V (bias, norm, RoPE) and write to cache
                t_prof = prefill_profile_now(&prof);
                if (!used_raw_prefill_attn_wo) {
                    for (int t = 0; t < n_tokens; t++) {
                        int pos = pos0 + t;
                        int cache_pos = pos % c->seq_len;
                        float *k_t = K_new + (size_t)t * layer_kv_dim;
                        float *v_t = V_new + (size_t)t * layer_kv_dim;
                        float *rc = rope_cos_buf + t * half_rope;
                        float *rs = rope_sin_buf + t * half_rope;

                        if (lw->attn.k_bias)
                            for (int i = 0; i < layer_kv_dim; i++) k_t[i] += lw->attn.k_bias[i];
                        if (lw->attn.v_bias)
                            for (int i = 0; i < layer_kv_dim; i++) v_t[i] += lw->attn.v_bias[i];
                        if (lw->attn.k_norm) {
                            int qk_stride = c->qk_norm_per_head ? layer_head_size : 0;
                            for (int h = 0; h < layer_n_kv_heads; h++)
                                prefill_cpu_ops()->rmsnorm(
                                    k_t + h * layer_head_size,
                                    k_t + h * layer_head_size,
                                    lw->attn.k_norm + h * qk_stride,
                                    layer_head_size, c->norm_eps);
                        }
                        if (bn_transformer_attention_value_shares_key(c))
                            prefill_rmsnorm_unit_heads(v_t, layer_n_kv_heads,
                                                       layer_head_size, c->norm_eps);
                        bn_transformer_cpu_apply_rope_heads(k_t, layer_n_kv_heads,
                                                            layer_head_size,
                                                            layer_rope_dims, rc, rs);

                        if (c->kv_f16)
                            bn_transformer_write_kv_fp16(s, loff, cache_pos, kv_dim,
                                                         k_t, v_t, layer_kv_dim);
                        else
                            bn_transformer_write_kv_fp32(s, loff, cache_pos, kv_dim,
                                                         k_t, v_t, layer_kv_dim);
                    }
                } else {
                    for (int t = 0; t < n_tokens; t++) {
                        int pos = pos0 + t;
                        int cache_pos = pos % c->seq_len;
                        float *k_t = K_new + (size_t)t * layer_kv_dim;
                        float *v_t = V_new + (size_t)t * layer_kv_dim;
                        if (c->kv_f16)
                            bn_transformer_write_kv_fp16(s, loff, cache_pos, kv_dim,
                                                         k_t, v_t, layer_kv_dim);
                        else
                            bn_transformer_write_kv_fp32(s, loff, cache_pos, kv_dim,
                                                         k_t, v_t, layer_kv_dim);
                    }
                }

                // Phase 2: batched attention (Q processing + attention, parallel over heads)
                BnBatchedAttnCtx bctx = {
                    .c = c, .s = s,
                    .Q_buf = Q_buf, .K_new = K_new, .V_new = V_new,
                    .out = q_gated ? Xb2 : Q_buf,
                    .loff = loff, .pos0 = pos0, .n_tokens = n_tokens,
                    .n_heads = c->n_heads, .n_kv_heads = layer_n_kv_heads,
                    .head_size = layer_head_size, .kv_dim = kv_dim,
                    .kv_mul = layer_kv_mul, .seq_len = c->seq_len,
                    .rope_dims = layer_rope_dims, .rope_freq = s->rope_freq,
                    .rope_cos = rope_cos_buf, .rope_sin = rope_sin_buf,
                    .rope_stride = half_rope,
                    .attention_scale = prefill_attention_scale(c, layer_head_size),
                    .q_norm = lw->attn.q_norm, .k_norm = lw->attn.k_norm,
                    .q_bias = lw->attn.q_bias, .k_bias = lw->attn.k_bias,
                    .v_bias = lw->attn.v_bias,
                    .qk_norm_per_head = c->qk_norm_per_head,
                    .norm_eps = c->norm_eps,
                    .q_gated = q_gated,
                    .wq_rows = lw->attn.wq.rows,
                    .q_row_stride = q_read_stride,
                    .wo_cols = wo_cols_attn,
                };
                t_prof = prefill_profile_now(&prof);
                int used_gpu_attn = used_raw_prefill_attn_wo;
                if (!used_raw_prefill_attn_wo &&
                    gpu && gpu->prefill_attention &&
                    bn_transformer_gpu_cuda_prefill_attention_enabled() &&
                    n_tokens >=
                        bn_transformer_gpu_cuda_prefill_attention_min_tokens() &&
                    prefill_prepare_q_for_gpu_attention(&bctx) == 0) {
                    void *wo_buf = prefill_backend_role_or_qweight(
                        backend, l, BN_BACKEND_HANDLE_WO_PREFILL,
                        &lw->attn.wo);
                    if (gpu->prefill_attention_wo && wo_buf &&
                        !lw->norm.attn_sub_norm &&
                        !(bn_transformer_attention_uses_post_norm(c) &&
                          lw->norm.attn_post_norm) &&
                        gpu->prefill_attention_wo(
                            gpu->ctx, Xb2, wo_buf, Q_buf, K_new, V_new,
                            n_tokens, c->n_heads, layer_n_kv_heads,
                            layer_head_size, layer_kv_mul, kv_dim,
                            lw->attn.wo.rows, lw->attn.wo.cols,
                            lw->attn.wo.type,
                            prefill_attention_scale(c, layer_head_size)) == 0) {
                        used_gpu_attn = 1;
                        used_fused_attn_wo = 1;
                    } else if (gpu->prefill_attention(
                            gpu->ctx, Q_buf, Q_buf, K_new, V_new, n_tokens,
                            c->n_heads, layer_n_kv_heads, layer_head_size,
                            layer_kv_mul, kv_dim,
                            prefill_attention_scale(c, layer_head_size)) == 0) {
                        used_gpu_attn = 1;
                    } else {
                        sh_arena_free(pf_arena);
                        return NULL;
                    }
                }
                if (!used_gpu_attn)
                    bn_transformer_batched_attn_dispatch(m, &bctx);
                if (!used_gpu_attn && q_gated) {
                    for (int t = 0; t < n_tokens; t++)
                        memcpy(Q_buf + (size_t)t * wo_cols_attn,
                               Xb2 + (size_t)t * wo_cols_attn,
                               (size_t)wo_cols_attn * sizeof(float));
                }
                prefill_profile_add(&prof.attn_cpu_ms, t_prof);
            } else {
                t_prof = prefill_profile_now(&prof);
                for (int t = 0; t < n_tokens; t++) {
                    int pos = pos0 + t;
                    int cache_pos = pos % c->seq_len;
                    float *q_t = Q_buf + (size_t)t * q_read_stride;
                    float *k_t = K_new + (size_t)t * layer_kv_dim;
                    float *v_t = V_new + (size_t)t * layer_kv_dim;

                    if (q_gated) {
                        for (int h = 0; h < c->n_heads; h++)
                            memcpy(s->q + h * layer_head_size,
                                   q_t + h * 2 * layer_head_size,
                                   layer_head_size * sizeof(float));
                    } else {
                        memcpy(s->q, q_t, layer_q_dim * sizeof(float));
                    }

                    if (lw->attn.q_norm) {
                        int qk_stride = c->qk_norm_per_head ? layer_head_size : 0;
                        for (int h = 0; h < c->n_heads; h++)
                            prefill_cpu_ops()->rmsnorm(
                                s->q + h * layer_head_size,
                                s->q + h * layer_head_size,
                                lw->attn.q_norm + h * qk_stride,
                                layer_head_size, c->norm_eps);
                    }
                    if (lw->attn.k_norm) {
                        int qk_stride = c->qk_norm_per_head ? layer_head_size : 0;
                        for (int h = 0; h < layer_n_kv_heads; h++)
                            prefill_cpu_ops()->rmsnorm(
                                k_t + h * layer_head_size,
                                k_t + h * layer_head_size,
                                lw->attn.k_norm + h * qk_stride,
                                layer_head_size, c->norm_eps);
                    }
                    if (bn_transformer_attention_value_shares_key(c))
                        prefill_rmsnorm_unit_heads(v_t, layer_n_kv_heads,
                                                   layer_head_size, c->norm_eps);

                    if (lw->attn.q_bias) for (int i = 0; i < layer_q_dim; i++) s->q[i] += lw->attn.q_bias[i];
                    if (lw->attn.k_bias) for (int i = 0; i < layer_kv_dim; i++) k_t[i] += lw->attn.k_bias[i];
                    if (lw->attn.v_bias) for (int i = 0; i < layer_kv_dim; i++) v_t[i] += lw->attn.v_bias[i];

                    float *rc = rope_cos_buf + t * half_rope;
                    float *rs = rope_sin_buf + t * half_rope;
                    bn_transformer_cpu_apply_rope_heads(s->q, c->n_heads,
                                                        layer_head_size,
                                                        layer_rope_dims, rc, rs);
                    bn_transformer_cpu_apply_rope_heads(k_t, layer_n_kv_heads,
                                                        layer_head_size,
                                                        layer_rope_dims, rc, rs);

                    if (c->kv_f16)
                        bn_transformer_write_kv_fp16(s, loff, cache_pos, kv_dim,
                                                     k_t, v_t, layer_kv_dim);
                    else
                        bn_transformer_write_kv_fp32(s, loff, cache_pos, kv_dim,
                                                     k_t, v_t, layer_kv_dim);

                    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
                    BnGQACtx gctx = { c, s, loff, pos, n_kv, layer_kv_mul,
                                      layer_head_size, kv_dim, c->seq_len,
                                      prefill_attention_scale(c, layer_head_size) };
                    bn_transformer_cpu_gqa_dispatch(m, &gctx, c->n_heads, layer_kv_mul);

                    if (q_gated) {
                        for (int h = 0; h < c->n_heads; h++) {
                            float *gate_h = q_t + h * 2 * layer_head_size + layer_head_size;
                            float *xb_h = s->xb + h * layer_head_size;
                            for (int d = 0; d < layer_head_size; d++)
                                xb_h[d] *= 1.0f / (1.0f + expf(-gate_h[d]));
                        }
                    }

                    memcpy(Q_buf + (size_t)t * wo_cols_attn, s->xb,
                           wo_cols_attn * sizeof(float));
            }
                prefill_profile_add(&prof.attn_cpu_ms, t_prof);
                }

            {
                int wo_cols = lw->attn.wo.cols;
                t_prof = prefill_profile_now(&prof);
                if (!used_fused_attn_wo && lw->norm.attn_sub_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_cpu_ops()->rmsnorm(
                            Q_buf + (size_t)t * wo_cols,
                            Q_buf + (size_t)t * wo_cols,
                            lw->norm.attn_sub_norm, wo_cols, c->norm_eps);
                if (!used_fused_attn_wo) {
                    void *wo_buf = prefill_backend_role_or_qweight(
                        backend, l, BN_BACKEND_HANDLE_WO_PREFILL,
                        &lw->attn.wo);
                    if (prefill_quant_matmul_gpu_buf(
                            m, Xb2, &lw->attn.wo, wo_buf, Q_buf,
                            n_tokens) != 0)
                        prefill_quant_matmul_gpu(m, Xb2, &lw->attn.wo,
                                                 Q_buf, n_tokens, s->x_q);
                }
                if (!used_fused_attn_wo &&
                    bn_transformer_attention_uses_post_norm(c) &&
                    lw->norm.attn_post_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_cpu_ops()->rmsnorm(Xb2 + (size_t)t * dim,
                                                   Xb2 + (size_t)t * dim,
                                                   lw->norm.attn_post_norm,
                                                   dim, c->norm_eps);
                prefill_profile_add(&prof.wo_ms, t_prof);
            }

            t_prof = prefill_profile_now(&prof);
            if (!used_attn_residual)
                for (int t = 0; t < n_tokens; t++)
                    for (int d = 0; d < dim; d++)
                        act[(size_t)t * dim + d] += Xb2[(size_t)t * dim + d];
            prefill_profile_add(&prof.residual_ms, t_prof);

        } else if (!is_attn) {
            int num_k_heads = c->ssm_group_count;
            int head_k_dim = c->ssm_state_size;
            int num_v_heads = c->ssm_time_step_rank;
            int head_v_dim = c->ssm_inner_size / (num_v_heads > 0 ? num_v_heads : 1);
            int key_dim_ssm = num_k_heads * head_k_dim;
            int value_dim = c->ssm_inner_size;
            int qkv_dim_ssm = key_dim_ssm * 2 + value_dim;
            int kern_ssm = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;
            int ssm_idx = plan.ssm_idx;

            if (bn_transformer_gpu_cuda_prefill_ssm_run_chain_enabled() &&
                gpu_hybrid_prefill &&
                (prefill_ssm_layer_chain_ready(m, lw, l, n_tokens) ||
                 prefill_ssm_moe_layer_chain_ready(m, lw, l, n_tokens))) {
                int run_end = l + 1;
                while (run_end < c->n_layers) {
                    BnLayerWeights *rlw = &w->layers[run_end];
                    BnLayerShapePlan rplan;
                    bn_transformer_plan_layer_shape(
                        &rplan, c, rlw, run_end,
                        bn_model_tq_state(m) != NULL);
                    if (rplan.is_attn ||
                        (!prefill_ssm_layer_chain_ready(
                             m, rlw, run_end, n_tokens) &&
                         !prefill_ssm_moe_layer_chain_ready(
                             m, rlw, run_end, n_tokens)))
                        break;
                    run_end++;
                }
                if (run_end - l > 1) {
                    for (int rl = l; rl < run_end; rl++) {
                        BnLayerWeights *rlw = &w->layers[rl];
                        BnLayerShapePlan rplan;
                        bn_transformer_plan_layer_shape(
                            &rplan, c, rlw, rl,
                            bn_model_tq_state(m) != NULL);
                        int r_did_ffn = 0;
                        int r_is_moe = rlw->moe.router_weight != NULL;
                        float *layer_out =
                            (!r_is_moe && rl == run_end - 1) ? act : NULL;
                        const float *layer_in = (rl == l) ? act : NULL;
                        if (prefill_ssm_layer_gpu(
                                m, layer_out, rlw, layer_in, n_tokens, dim,
                                qkv_dim_ssm, value_dim, num_k_heads,
                                head_k_dim, num_v_heads, head_v_dim,
                                kern_ssm, rplan.ssm_idx, rl, !r_is_moe,
                                c->norm_eps, &r_did_ffn) != 0 ||
                            (!r_is_moe && !r_did_ffn)) {
                            sh_arena_free(pf_arena);
                            return NULL;
                        }
                        if (r_is_moe) {
                            float *moe_out =
                                (rl == run_end - 1) ? act : NULL;
                            if (prefill_moe_ffn_gpu_batch(
                                    m, moe_out, rlw, NULL, n_tokens, dim,
                                    rl) != 0) {
                                sh_arena_free(pf_arena);
                                return NULL;
                            }
                        }
                    }
                    l = run_end - 1;
                    goto prefill_layer_done;
                }
            }

            size_t state_per_layer = (size_t)num_v_heads * head_k_dim * head_v_dim;
            float *ssm_state = s->ssm_state + (size_t)ssm_idx * state_per_layer;
            size_t conv_per_layer = (size_t)(kern_ssm - 1) * qkv_dim_ssm;
            float *conv_state = s->ssm_conv_state + (size_t)ssm_idx * conv_per_layer;

            int ssm_did_ffn = 0;
            if (prefill_ssm_layer_gpu(m, act, lw, act, n_tokens, dim,
                                      qkv_dim_ssm, value_dim, num_k_heads,
                                      head_k_dim, num_v_heads, head_v_dim,
                                      kern_ssm, ssm_idx, l,
                                      n_tokens >=
                                          bn_transformer_gpu_cuda_prefill_moe_chain_min_tokens(
                                              c, bn_model_gpu(m)),
                                      c->norm_eps, &ssm_did_ffn) == 0) {
                if (ssm_did_ffn)
                    goto prefill_layer_done;
                goto prefill_ssm_done;
            }

            for (int t = 0; t < n_tokens; t++)
                prefill_cpu_ops()->rmsnorm(Xb + (size_t)t * dim,
                                           act + (size_t)t * dim,
                                           lw->norm.attn_norm, dim,
                                           c->norm_eps);

            if (q_buf_stride < qkv_dim_ssm) { sh_arena_free(pf_arena); return NULL; }
            float *QKV_all = Q_buf;
            float *Z_all = Xb2;
            float *Out_all = Hb;
            const BnPrefillCPUOps *ssm_cpu_ops = prefill_cpu_ops();

            int ssm_preq8k = 0;
            {
                float *qz_out[2] = { QKV_all, Z_all };
                const BnQWeight *qz_w[2] = { &lw->ssm.wqkv, &lw->ssm.wz };
                ssm_preq8k =
                    bn_transformer_prefill_route_preq8k_pair_enabled(
                        prefill_cpu_ops(), bn_model_gpu(m),
                        prefill_force_float_kquant(m), dim,
                        lw->ssm.wqkv.type, lw->ssm.wz.type) &&
                    prefill_try_preq8k_multi(m, &preq8k, qz_out, qz_w,
                                             2, Xb, dim, n_tokens);
            }
            if (!ssm_preq8k) {
                prefill_quant_matmul_gpu(m, QKV_all, &lw->ssm.wqkv, Xb,
                                         n_tokens, s->x_q);
                prefill_quant_matmul_gpu(m, Z_all, &lw->ssm.wz, Xb,
                                         n_tokens, s->x_q);
            }

            for (int t = 0; t < n_tokens; t++) {
                float *qkv_t = QKV_all + (size_t)t * lw->ssm.wqkv.rows;
                float *z_t = Z_all + (size_t)t * lw->ssm.wz.rows;
                float *out_t = Out_all + (size_t)t * value_dim;
                float *xb_t = Xb + (size_t)t * dim;

                BnSSMConvCtx conv_ctx = { qkv_t, conv_state, lw->ssm.ssm_conv1d,
                                          qkv_dim_ssm, kern_ssm };
                BnTPTask conv_task = {
                    bn_transformer_prefill_ssm_conv_silu_op(c, ssm_cpu_ops),
                    &conv_ctx, qkv_dim_ssm
                };
                bn_tp_dispatch(bn_model_pool(m), &conv_task, 1);

                float *q_raw = qkv_t;
                float *k_raw = qkv_t + key_dim_ssm;
                float *v_raw = qkv_t + 2 * key_dim_ssm;

                BnSSML2NormCtx norm_ctx = { q_raw, k_raw, c->norm_eps, head_k_dim };
                BnTPTask norm_task = {
                    bn_transformer_prefill_ssm_l2norm_op(c, ssm_cpu_ops),
                    &norm_ctx, num_k_heads
                };
                bn_tp_dispatch(bn_model_pool(m), &norm_task, 1);

                if (num_v_heads > BN_MAX_VLA_ELEMS) continue;
                float alpha_arr[num_v_heads > 0 ? num_v_heads : 1];
                float beta_arr[num_v_heads > 0 ? num_v_heads : 1];
                BnMatvecTask ab[2] = {
                    { alpha_arr, &lw->ssm.ssm_alpha, NULL, 0 },
                    { beta_arr,  &lw->ssm.ssm_beta, NULL, 0 },
                };
                if (!ssm_preq8k ||
                    !bn_transformer_prefill_route_preq8k_pair_enabled(
                        prefill_cpu_ops(), bn_model_gpu(m),
                        prefill_force_float_kquant(m), dim,
                        lw->ssm.ssm_alpha.type, lw->ssm.ssm_beta.type) ||
                    !prefill_try_preq8k_matvec_batch(m, &preq8k, ab, 2,
                                                     xb_t, dim, t)) {
                    bn_quant_matvec_batch(ab, 2, xb_t, s->x_q,
                                          bn_model_pool(m));
                }
                for (int h = 0; h < num_v_heads; h++) {
                    float dt = alpha_arr[h] + lw->ssm.ssm_dt_bias[h];
                    float dt_sp = (dt > 20.0f) ? dt : logf(1.0f + expf(dt));
                    alpha_arr[h] = expf(dt_sp * lw->ssm.ssm_a[h]);
                    beta_arr[h] = 1.0f / (1.0f + expf(-beta_arr[h]));
                }

                float q_scale = 1.0f / sqrtf((float)head_k_dim);
                BnSSMDeltaCtx delta_ctx = {
                    ssm_state, out_t, q_raw, k_raw, v_raw,
                    alpha_arr, beta_arr,
                    num_k_heads, head_k_dim, head_v_dim, q_scale
                };
                BnTPTask delta_task = {
                    bn_transformer_prefill_ssm_delta_op(c, ssm_cpu_ops),
                    &delta_ctx, num_v_heads
                };
                bn_tp_dispatch(bn_model_pool(m), &delta_task, 1);

                BnSSMGateCtx gate_ctx = { out_t, z_t, lw->ssm.ssm_norm,
                                          c->norm_eps, head_v_dim };
                BnTPTask gate_task = {
                    bn_transformer_prefill_ssm_gate_op(c, ssm_cpu_ops),
                    &gate_ctx, num_v_heads
                };
                bn_tp_dispatch(bn_model_pool(m), &gate_task, 1);
            }

            prefill_quant_matmul_gpu(m, Xb, &lw->ssm.ssm_out, Out_all, n_tokens, s->x_q);

            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb[(size_t)t * dim + d];
        }
prefill_ssm_done:

        if (lw->moe.router_weight) {
            if (bn_moe_forward_batch(m, sess, lw, l, act, Xb, n_tokens) != 0) {
                sh_arena_free(pf_arena);
                return NULL;
            }
        } else if (lw->ffn.ffn_up.data) {
            int used_gpu_batch_ffn = 0;
            int used_ffn_residual = 0;
            t_prof = prefill_profile_now(&prof);
            BnGPUBackend *gpu_ffn = bn_model_gpu(m);
            const BnBackendModel *backend_ffn = bn_model_backend(m);
            int dense_ffn_batch_min_tokens =
                bn_transformer_gpu_cuda_prefill_dense_chain_min_tokens(
                    c, gpu_ffn);
            int can_use_dense_ffn_batch =
                bn_transformer_gpu_dense_ffn_batch_tokens_allowed(
                    gpu_ffn, c, n_tokens);
            void *ffn_norm_buf =
                backend_ffn ? bn_backend_model_handle(
                    backend_ffn, l, BN_BACKEND_HANDLE_FFN_NORM) : NULL;
            if (c->has_ffn_gate && !lw->norm.ffn_sub_norm &&
                !(bn_transformer_ffn_uses_post_norm(c) &&
                  lw->norm.ffn_post_norm) &&
                can_use_dense_ffn_batch &&
                gpu_ffn && gpu_ffn->dense_ffn_batch_norm_resid &&
                n_tokens >= dense_ffn_batch_min_tokens &&
                ffn_norm_buf &&
                prefill_dense_ffn_gpu_batch(m, act, lw, act, n_tokens,
                                            dim, hidden_dim,
                                            c->act_type, l, ffn_norm_buf,
                                            c->norm_eps, 1) == 0) {
                used_gpu_batch_ffn = 1;
                used_ffn_residual = 1;
            }
            if (!used_gpu_batch_ffn) {
                prefill_profile_add(&prof.ffn_ms, t_prof);
                t_prof = prefill_profile_now(&prof);
                for (int t = 0; t < n_tokens; t++)
                    prefill_cpu_ops()->rmsnorm(Xb + t * dim,
                                               act + (size_t)t * dim,
                                               lw->norm.ffn_norm, dim,
                                               c->norm_eps);
                prefill_profile_add(&prof.ffn_norm_ms, t_prof);

                t_prof = prefill_profile_now(&prof);
                if (c->has_ffn_gate && !lw->norm.ffn_sub_norm &&
                    !(bn_transformer_ffn_uses_post_norm(c) &&
                      lw->norm.ffn_post_norm) &&
                    can_use_dense_ffn_batch &&
                    (!bn_model_arch_uses_hybrid_layer_layout(c) ||
                     n_tokens >= dense_ffn_batch_min_tokens) &&
                    prefill_dense_ffn_gpu_batch(m, Xb, lw, Xb, n_tokens,
                                                dim, hidden_dim,
                                                c->act_type, l, NULL,
                                                0.0f, 0) == 0) {
                    used_gpu_batch_ffn = 1;
                } else if (c->has_ffn_gate) {
                    double t_ffn_step = prefill_profile_now(&prof);
                    float *gu_out[2] = { Hb, Hb2 };
                    const BnQWeight *gu_w[2] = {
                        &lw->ffn.ffn_gate, &lw->ffn.ffn_up
                    };
                    if (!bn_transformer_prefill_route_preq8k_pair_enabled(
                            prefill_cpu_ops(), bn_model_gpu(m),
                            prefill_force_float_kquant(m), dim,
                            lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type) ||
                        !prefill_try_preq8k_multi(m, &preq8k, gu_out, gu_w,
                                                  2, Xb, dim, n_tokens))
                        prefill_quant_matmul_multi(m, gu_out, gu_w, 2, Xb,
                                                   n_tokens, s->x_q);
                prefill_profile_add(&prof.ffn_gateup_ms, t_ffn_step);

                t_ffn_step = prefill_profile_now(&prof);
                BnPrefillFFNActCtx act_ctx = {
                    Hb, Hb2, hidden_dim, c->act_type,
                    bn_transformer_prefill_uses_exact_activation(c)
                };
                BnTPTask act_task = {
                    prefill_cpu_ops()->ffn_activation, &act_ctx, n_tokens
                };
                bn_tp_dispatch(bn_model_pool(m), &act_task, 1);
                prefill_profile_add(&prof.ffn_act_ms, t_ffn_step);
            } else {
                double t_ffn_step = prefill_profile_now(&prof);
                prefill_quant_matmul_gpu(m, Hb, &lw->ffn.ffn_up, Xb, n_tokens, s->x_q);
                prefill_profile_add(&prof.ffn_gateup_ms, t_ffn_step);
                t_ffn_step = prefill_profile_now(&prof);
                BnPrefillFFNActCtx act_ctx = {
                    Hb, NULL, hidden_dim, c->act_type,
                    bn_transformer_prefill_uses_exact_activation(c)
                };
                BnTPTask act_task = {
                    prefill_cpu_ops()->ffn_activation, &act_ctx, n_tokens
                };
                bn_tp_dispatch(bn_model_pool(m), &act_task, 1);
                prefill_profile_add(&prof.ffn_act_ms, t_ffn_step);
                }
            }

            if (!used_gpu_batch_ffn) {
                if (lw->norm.ffn_sub_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_cpu_ops()->rmsnorm(
                            Hb + (size_t)t * hidden_dim,
                            Hb + (size_t)t * hidden_dim,
                            lw->norm.ffn_sub_norm, hidden_dim, c->norm_eps);

                double t_ffn_step = prefill_profile_now(&prof);
                prefill_quant_matmul_gpu(m, Xb, &lw->ffn.ffn_down, Hb,
                                         n_tokens, s->x_q);
                prefill_profile_add(&prof.ffn_down_ms, t_ffn_step);
                if (bn_transformer_ffn_uses_post_norm(c) &&
                    lw->norm.ffn_post_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_cpu_ops()->rmsnorm(Xb + (size_t)t * dim,
                                                   Xb + (size_t)t * dim,
                                                   lw->norm.ffn_post_norm,
                                                   dim, c->norm_eps);
            }
            prefill_profile_add(&prof.ffn_ms, t_prof);

            if (used_ffn_residual)
                continue;

            t_prof = prefill_profile_now(&prof);
            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb[(size_t)t * dim + d];
            prefill_profile_add(&prof.residual_ms, t_prof);
        }

        if (bn_transformer_uses_layer_output_scale(c) && lw->norm.layer_output_scale) {
            float scale = lw->norm.layer_output_scale[0];
            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] *= scale;
        }
prefill_layer_done:
        ;
    }

prefill_layers_done:
    if (all_logits) {
        int vocab_size = c->vocab_size;
        t_prof = prefill_profile_now(&prof);
        for (int t = 0; t < n_tokens; t++) {
            memcpy(s->x, act + (size_t)t * dim, dim * sizeof(float));
            float *lg = prefill_logits(m, sess);
            if (!lg) { sh_arena_free(pf_arena); return NULL; }
            memcpy(all_logits + (size_t)t * vocab_size, lg,
                   vocab_size * sizeof(float));
        }
        prefill_profile_add(&prof.logits_ms, t_prof);
        if (prof.enabled) {
            fprintf(stderr,
                    "[bn:prefill:profile] tokens=%d layers=%d embed=%.3f attn_norm=%.3f qkv=%.3f attn_cpu=%.3f wo=%.3f ffn_norm=%.3f ffn=%.3f gateup=%.3f act=%.3f down=%.3f residual=%.3f logits=%.3f\n",
                    n_tokens, c->n_layers, prof.embed_ms, prof.attn_norm_ms,
                    prof.qkv_ms, prof.attn_cpu_ms, prof.wo_ms,
                    prof.ffn_norm_ms, prof.ffn_ms, prof.ffn_gateup_ms,
                    prof.ffn_act_ms, prof.ffn_down_ms, prof.residual_ms,
                    prof.logits_ms);
        }
        sh_arena_free(pf_arena);
        return s->logits;
    }

    if (prof.enabled) {
        fprintf(stderr,
                "[bn:prefill:profile] tokens=%d layers=%d embed=%.3f attn_norm=%.3f qkv=%.3f attn_cpu=%.3f wo=%.3f ffn_norm=%.3f ffn=%.3f gateup=%.3f act=%.3f down=%.3f residual=%.3f logits=%.3f\n",
                n_tokens, c->n_layers, prof.embed_ms, prof.attn_norm_ms,
                prof.qkv_ms, prof.attn_cpu_ms, prof.wo_ms,
                prof.ffn_norm_ms, prof.ffn_ms, prof.ffn_gateup_ms,
                prof.ffn_act_ms, prof.ffn_down_ms, prof.residual_ms,
                prof.logits_ms);
    }
    memcpy(s->x, act + (size_t)(n_tokens - 1) * dim, dim * sizeof(float));
    sh_arena_free(pf_arena);
    if (need_last_logits)
        return prefill_logits(m, sess);
    return s->x;
}

float *bn_transformer_prefill(BnModel *m, BnSession *s, const int *tokens,
                              int n_tokens, int pos0) {
    if (prefill_use_shared_all2_decode_fallback(m)) {
        for (int i = 0; i + 1 < n_tokens; i++)
            if (bn_transformer_forward_no_logits(m, s, tokens[i], pos0 + i) != 0)
                return NULL;
        return n_tokens > 0
            ? bn_transformer_forward(m, s, tokens[n_tokens - 1],
                                     pos0 + n_tokens - 1)
            : NULL;
    }
    return prefill_internal(m, s, tokens, n_tokens, pos0, NULL, 1);
}

int bn_transformer_prefill_no_logits(BnModel *m, BnSession *s, const int *tokens,
                                     int n_tokens, int pos0) {
    if (prefill_use_shared_all2_decode_fallback(m)) {
        for (int i = 0; i < n_tokens; i++)
            if (bn_transformer_forward_no_logits(m, s, tokens[i], pos0 + i) != 0)
                return -1;
        return 0;
    }
    return prefill_internal(m, s, tokens, n_tokens, pos0, NULL, 0) ? 0 : -1;
}

int bn_transformer_prefill_all(BnModel *m, BnSession *s, const int *tokens,
                               int n_tokens, int pos0, float *all_logits) {
    if (!all_logits || n_tokens <= 0) return -1;

    if (prefill_use_shared_all2_decode_fallback(m)) {
        size_t row_bytes = (size_t)m->config.vocab_size * sizeof(float);
        for (int i = 0; i < n_tokens; i++) {
            float *logits = bn_transformer_forward(m, s, tokens[i], pos0 + i);
            if (!logits) return -1;
            memcpy(all_logits + (size_t)i * m->config.vocab_size,
                   logits, row_bytes);
        }
        return 0;
    }

    if (n_tokens == 1) {
        float *logits = bn_transformer_forward(m, s, tokens[0], pos0);
        if (!logits) return -1;
        memcpy(all_logits, logits, m->config.vocab_size * sizeof(float));
        return 0;
    }

    float *result = prefill_internal(m, s, tokens, n_tokens, pos0, all_logits, 1);
    return result ? 0 : -1;
}
