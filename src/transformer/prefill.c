#include "transformer_internal.h"
#include "transformer_cpu_internal.h"
#include "transformer_batched_attn_internal.h"
#include "simd_helpers.h"
#include "transformer_gqa_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "backend_model.h"
#include "backend_quant.h"
#include "moe.h"
#include "session.h"
#include "sh_arena.h"
#include "sh_log.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

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

static void prefill_quant_matmul_gpu(const BnModel *m,
                                     float *out,
                                     const BnQWeight *W,
                                     const float *X,
                                     int n_tokens,
                                     int8_t *x_q_buf) {
    if (!bn_model_gpu(m)) {
        const BnBackendModel *backend = bn_model_backend(m);
        bn_quant_matmul_prepared(out, W,
                                 bn_backend_model_prepared_qweight(backend, W),
                                 X, n_tokens, x_q_buf, bn_model_pool(m));
        return;
    }
    bn_backend_quant_matmul_gpu_buf(out, W,
                                    prefill_qweight_backend_buf(bn_model_backend(m), W),
                                    X, n_tokens, x_q_buf, bn_model_pool(m),
                                    bn_model_gpu(m));
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
            bn_backend_quant_matmul_batch_gpu_buf(
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
    if (lw->attn.wq.type != lw->attn.wk.type ||
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
    if (lw->attn.wq.type != lw->attn.wk.type ||
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
    if (gateup_buf && lw->ffn.ffn_gate.type == lw->ffn.ffn_up.type) {
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

static int prefill_gpu_attention_min_tokens(void);
static int prefill_dense_chain_min_tokens(const BnConfig *c,
                                          const BnGPUBackend *gpu);

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
    if (gateup_buf && lw->ffn.ffn_gate.type == lw->ffn.ffn_up.type) {
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
    if (!qk_buf || (!has_packed_qkv && !wv_buf) || !wo_buf || !gate_buf || !down_buf ||
        !attn_norm_buf || !ffn_norm_buf)
        return -1;

    return gpu->prefill_dense_layer(
        gpu->ctx, out, qk_buf, wv_buf, wo_buf, gate_buf, up_buf, down_buf,
        attn_norm_buf, ffn_norm_buf, q_norm_buf, k_norm_buf, X, K_out, V_out,
        n_tokens, dim, hidden_dim, n_heads, n_kv_heads, head_size, kv_mul,
        kv_dim, qk_rows, qk_type, wv_rows, wv_type, lw->attn.wo.rows,
        lw->attn.wo.cols, lw->attn.wo.type, lw->ffn.ffn_gate.type,
        lw->ffn.ffn_up.type, lw->ffn.ffn_down.type, m->config.act_type,
        m->config.qk_norm_per_head, m->config.norm_eps, pos0, rope_dims,
        kv_cache_off, kv_cache_stride, attention_scale);
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
        n_tokens < prefill_dense_chain_min_tokens(c, gpu) ||
        layer_rope_theta != c->rope_theta ||
        !plan->is_attn || plan->q_gated ||
        lw->moe.router_weight || !c->has_ffn_gate || !lw->ffn.ffn_up.data ||
        !lw->attn.wo.data || !lw->ffn.ffn_gate.data ||
        !lw->ffn.ffn_down.data ||
        lw->attn.q_bias || lw->attn.k_bias || lw->attn.v_bias ||
        lw->norm.attn_sub_norm || lw->norm.ffn_sub_norm ||
        lw->norm.layer_output_scale ||
        ((c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) &&
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
                     lw->ffn.ffn_gate.type == lw->ffn.ffn_up.type
                         ? gateup_buf
                         : prefill_qweight_backend_buf(backend,
                                                       &lw->ffn.ffn_gate);
    void *up_buf = gateup_buf &&
                   lw->ffn.ffn_gate.type == lw->ffn.ffn_up.type
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
    return qk_buf && (has_packed_qkv || wv_buf) && wo_buf &&
           gate_buf && (gateup_buf || up_buf) &&
           down_buf && attn_norm_buf && ffn_norm_buf;
}

#ifdef __AVX2__
static int prefill_quant_can_preq8k_pair(int a, int b) {
    return bn_quant_format_can_preq8k(a) && bn_quant_format_can_preq8k(b);
}

static int prefill_quant_can_preq8k_triple(int a, int b, int c) {
    return prefill_quant_can_preq8k_pair(a, b) && bn_quant_format_can_preq8k(c);
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
#endif

#ifdef __ARM_NEON
#define prefill_rmsnorm bn_transformer_rmsnorm_neon
#elif defined(__AVX2__)
#define prefill_rmsnorm bn_transformer_rmsnorm_avx2
#elif defined(__wasm_simd128__)
#define prefill_rmsnorm bn_transformer_rmsnorm_wasm
#else
#define prefill_rmsnorm bn_transformer_rmsnorm_scalar
#endif

#ifdef __ARM_NEON
#define prefill_ssm_conv_silu bn_transformer_ssm_conv_silu_neon_range
#define prefill_ssm_l2norm    bn_transformer_ssm_l2norm_neon_range
#define prefill_ssm_delta     bn_transformer_ssm_delta_neon_range
#define prefill_ssm_gate      bn_transformer_ssm_gate_neon_range
#elif defined(__AVX2__)
#define prefill_ssm_conv_silu bn_transformer_ssm_conv_silu_avx2_range
#define prefill_ssm_l2norm    bn_transformer_ssm_l2norm_avx2_range
#define prefill_ssm_delta     bn_transformer_ssm_delta_avx2_range
#define prefill_ssm_gate      bn_transformer_ssm_gate_avx2_range
#elif defined(__wasm_simd128__)
#define prefill_ssm_conv_silu bn_transformer_ssm_conv_silu_wasm_range
#define prefill_ssm_l2norm    bn_transformer_ssm_l2norm_wasm_range
#define prefill_ssm_delta     bn_transformer_ssm_delta_wasm_range
#define prefill_ssm_gate      bn_transformer_ssm_gate_wasm_range
#else
#define prefill_ssm_conv_silu bn_transformer_ssm_conv_silu_scalar_range
#define prefill_ssm_l2norm    bn_transformer_ssm_l2norm_scalar_range
#define prefill_ssm_delta     bn_transformer_ssm_delta_scalar_range
#define prefill_ssm_gate      bn_transformer_ssm_gate_scalar_range
#endif

static float *prefill_logits(BnModel *m, BnSession *sess) {
    return bn_transformer_forward_logits(m, sess);
}

static int prefill_layer_rope_dims(const BnConfig *c, int layer_head_size) {
    int use_swa_rope = c->rope_theta_swa > 0.0f && layer_head_size < c->head_size;
    int rope_dims = use_swa_rope && c->rope_dim_count_swa > 0
        ? c->rope_dim_count_swa
        : (c->rope_dim_count > 0 ? c->rope_dim_count : layer_head_size);
    return rope_dims > layer_head_size ? layer_head_size : rope_dims;
}

static float prefill_layer_rope_theta(const BnConfig *c, int layer_head_size) {
    return (c->rope_theta_swa > 0.0f && layer_head_size < c->head_size)
        ? c->rope_theta_swa : c->rope_theta;
}

static float prefill_attention_scale(const BnConfig *c, int head_size) {
    return (c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4)
        ? 1.0f
        : 1.0f / sqrtf((float)head_size);
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

static float prefill_gelu(float x) {
    return 0.5f * x *
           (1.0f + tanhf(0.7978845608028654f * x *
                         (1.0f + 0.044715f * x * x)));
}

typedef struct {
    float *hb;
    const float *hb2;
    int hidden_dim;
    int act_type;
} BnPrefillFFNActCtx;

static void prefill_ffn_activation_range(void *ctx, int start, int end) {
    BnPrefillFFNActCtx *c = (BnPrefillFFNActCtx *)ctx;
    int hidden_dim = c->hidden_dim;
    for (int t = start; t < end; t++) {
        float *hb_t = c->hb + (size_t)t * hidden_dim;
        const float *hb2_t = c->hb2 ? c->hb2 + (size_t)t * hidden_dim : NULL;
        if (c->act_type == 1) {
            int i = 0;
#ifdef __ARM_NEON
            float32x4_t zero_v = vdupq_n_f32(0.0f);
            for (; i + 3 < hidden_dim; i += 4) {
                float32x4_t g = vmaxq_f32(vld1q_f32(hb_t + i), zero_v);
                float32x4_t v = vmulq_f32(g, g);
                if (hb2_t)
                    v = vmulq_f32(v, vld1q_f32(hb2_t + i));
                vst1q_f32(hb_t + i, v);
            }
#endif
#ifdef __AVX2__
            __m256 zero_v = _mm256_setzero_ps();
            for (; i + 7 < hidden_dim; i += 8) {
                __m256 g = _mm256_max_ps(_mm256_loadu_ps(hb_t + i), zero_v);
                __m256 v = _mm256_mul_ps(g, g);
                if (hb2_t)
                    v = _mm256_mul_ps(v, _mm256_loadu_ps(hb2_t + i));
                _mm256_storeu_ps(hb_t + i, v);
            }
#endif
            for (; i < hidden_dim; i++) {
                float g = hb_t[i] > 0 ? hb_t[i] : 0;
                hb_t[i] = hb2_t ? g * g * hb2_t[i] : g * g;
            }
        } else if (c->act_type == 2) {
            int i = 0;
#ifdef __ARM_NEON
            for (; i + 3 < hidden_dim; i += 4) {
                float32x4_t v = bn_neon_fast_gelu_f32(vld1q_f32(hb_t + i));
                if (hb2_t)
                    v = vmulq_f32(v, vld1q_f32(hb2_t + i));
                vst1q_f32(hb_t + i, v);
            }
#endif
#ifdef __AVX2__
            for (; i + 7 < hidden_dim; i += 8) {
                __m256 v = bn_avx2_fast_gelu_ps(_mm256_loadu_ps(hb_t + i));
                if (hb2_t)
                    v = _mm256_mul_ps(v, _mm256_loadu_ps(hb2_t + i));
                _mm256_storeu_ps(hb_t + i, v);
            }
#endif
            for (; i < hidden_dim; i++) {
                float v = prefill_gelu(hb_t[i]);
                hb_t[i] = hb2_t ? v * hb2_t[i] : v;
            }
        } else {
            int i = 0;
#ifdef __ARM_NEON
            for (; i + 3 < hidden_dim; i += 4) {
                float32x4_t v = bn_neon_fast_silu_f32(vld1q_f32(hb_t + i));
                if (hb2_t)
                    v = vmulq_f32(v, vld1q_f32(hb2_t + i));
                vst1q_f32(hb_t + i, v);
            }
#endif
#ifdef __AVX2__
            for (; i + 7 < hidden_dim; i += 8) {
                __m256 v = bn_avx2_fast_silu_ps(_mm256_loadu_ps(hb_t + i));
                if (hb2_t)
                    v = _mm256_mul_ps(v, _mm256_loadu_ps(hb2_t + i));
                _mm256_storeu_ps(hb_t + i, v);
            }
#endif
            for (; i < hidden_dim; i++) {
                float v = hb_t[i] / (1.0f + expf(-hb_t[i]));
                hb_t[i] = hb2_t ? v * hb2_t[i] : v;
            }
        }
    }
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
                prefill_rmsnorm(row + h * head_size, row + h * head_size,
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

static int prefill_gpu_attention_min_tokens(void) {
    const char *env = getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    if (!env || !*env) return 64;
    int n = atoi(env);
    return n > 0 ? n : 64;
}

static int prefill_dense_chain_min_tokens(const BnConfig *c,
                                          const BnGPUBackend *gpu) {
    const char *env = getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    if (env && *env)
        return prefill_gpu_attention_min_tokens();
    if (gpu && gpu->kind == BN_GPU_BACKEND_CUDA && c && c->dim >= 2048)
        return 16;
    return prefill_gpu_attention_min_tokens();
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
    prof.enabled = getenv("BN_PREFILL_PROFILE") != NULL;
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

    size_t act_elems = (size_t)n_tokens * dim;
    if (act_elems / n_tokens != (size_t)dim) {
        SH_LOG_ERROR("Prefill activation buffer size overflow");
        return NULL;
    }

    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : head_size;
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
    if (c->full_attn_interval > 0 && c->ssm_inner_size > 0) {
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
#ifdef __AVX2__
    int n_bpr_pf = (dim % BN_QK_K == 0) ? dim / BN_QK_K : 0;
    if (n_bpr_pf > 0)
        arena_size += nt * dim
                    + nt * n_bpr_pf * sizeof(float)
                    + nt * n_bpr_pf * 16 * sizeof(int16_t);
#endif

    SHArena *pf_arena = sh_arena_create(arena_size);
    if (!pf_arena) return NULL;

    float *act = (float *)sh_arena_alloc(pf_arena, act_elems * sizeof(float));
    if (!act) { sh_arena_free(pf_arena); return NULL; }

    for (int t = 0; t < n_tokens; t++)
        bn_model_embed_token(m, act + (size_t)t * dim, tokens[t]);
    prefill_profile_add(&prof.embed_ms, t_prof);

    float *batch_buf = (float *)sh_arena_alloc(pf_arena, batch_floats * sizeof(float));
    if (!batch_buf) { sh_arena_free(pf_arena); return NULL; }

#ifdef __AVX2__
    int8_t *pf_xq = NULL;
    float *pf_xd = NULL;
    int16_t *pf_xbs = NULL;
    if (n_bpr_pf > 0) {
        pf_xq = (int8_t *)sh_arena_alloc(pf_arena, nt * dim);
        pf_xd = (float *)sh_arena_alloc(pf_arena, nt * n_bpr_pf * sizeof(float));
        pf_xbs = (int16_t *)sh_arena_alloc(pf_arena, nt * n_bpr_pf * 16 * sizeof(int16_t));
        if (!pf_xq || !pf_xd || !pf_xbs)
            pf_xq = NULL;
    }
#endif

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

    if (!getenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN") &&
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
            if (!prefill_dense_layer_chain_ready(
                    m, lw, &plan, l, n_tokens, layer_rope_theta)) {
                chain_ready = 0;
                break;
            }
        }
        if (chain_ready) {
            int direct_gpu_kv =
                !getenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV") &&
                bn_model_gpu(m) &&
                bn_model_gpu(m)->kind == BN_GPU_BACKEND_CUDA &&
                !c->kv_f16 &&
                pos0 >= 0 && pos0 + n_tokens <= c->seq_len;
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
                if (prefill_dense_layer_gpu_batch(
                        m, layer_out, lw, layer_in,
                        direct_gpu_kv ? NULL : K_new,
                        direct_gpu_kv ? NULL : V_new, n_tokens,
                        dim, hidden_dim, c->n_heads, layer_n_kv_heads,
                        layer_head_size, layer_kv_mul, layer_kv_dim,
                        layer_rope_dims, l, pos0, kv_cache_off, kv_dim,
                        prefill_attention_scale(c, layer_head_size)) != 0) {
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
            if (bn_model_gpu(m) &&
                bn_model_tq_state(m) == NULL &&
                n_tokens >= prefill_gpu_attention_min_tokens() &&
                pos0 == 0 &&
                layer_rope_theta == c->rope_theta &&
                !lw->moe.router_weight &&
                c->has_ffn_gate &&
                lw->ffn.ffn_up.data &&
                !plan.q_gated &&
                !lw->attn.q_bias && !lw->attn.k_bias && !lw->attn.v_bias &&
                !lw->norm.attn_sub_norm && !lw->norm.ffn_sub_norm &&
                !lw->norm.layer_output_scale &&
                !((c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) &&
                  (lw->norm.attn_post_norm || lw->norm.ffn_post_norm))) {
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
            int can_fuse_attn_input_norm =
                gpu && gpu->prefill_qkv_attention_wo_norm_resid &&
                attn_norm_buf &&
                n_tokens >= prefill_gpu_attention_min_tokens() &&
                bn_model_tq_state(m) == NULL &&
                !q_gated && pos0 == 0 &&
                layer_rope_theta == c->rope_theta &&
                !lw->attn.q_bias && !lw->attn.k_bias && !lw->attn.v_bias &&
                !lw->norm.attn_sub_norm &&
                !((c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) &&
                  lw->norm.attn_post_norm);
            int attn_norm_ready = 0;
            if (!can_fuse_attn_input_norm) {
                t_prof = prefill_profile_now(&prof);
                for (int t = 0; t < n_tokens; t++)
                    prefill_rmsnorm(Xb + t * dim, act + (size_t)t * dim,
                                    lw->norm.attn_norm, dim, c->norm_eps);
                prefill_profile_add(&prof.attn_norm_ms, t_prof);
                attn_norm_ready = 1;
            }
            if (gpu && gpu->prefill_qkv_attention_wo &&
                bn_model_tq_state(m) == NULL &&
                !q_gated && pos0 == 0 &&
                layer_rope_theta == c->rope_theta &&
                !lw->attn.q_bias && !lw->attn.k_bias && !lw->attn.v_bias &&
                !lw->norm.attn_sub_norm &&
                !((c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) &&
                  lw->norm.attn_post_norm)) {
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
                if (qk_buf && wv_buf && wo_buf && can_fuse_attn_input_norm) {
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
                            prefill_rmsnorm(
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
                        prefill_rmsnorm(Xb + t * dim,
                                        act + (size_t)t * dim,
                                        lw->norm.attn_norm, dim,
                                        c->norm_eps);
                    prefill_profile_add(&prof.attn_norm_ms, t_prof);
                    attn_norm_ready = 1;
                }
#ifdef __AVX2__
            if (pf_xq && !bn_model_gpu(m) &&
                prefill_quant_can_preq8k_triple(lw->attn.wq.type, lw->attn.wk.type, lw->attn.wv.type)) {
                int n_bpr = dim / BN_QK_K;
                for (int t = 0; t < n_tokens; t++)
                    bn_quant_x_to_q8k(Xb + (size_t)t * dim,
                                      pf_xq + (size_t)t * dim,
                                      pf_xd + (size_t)t * n_bpr,
                                      pf_xbs + (size_t)t * n_bpr * 16, dim);
                {
                    float *qkv_out[3] = { Q_buf, K_new, V_new };
                    const BnQWeight *qkv_w[3] = { &lw->attn.wq, &lw->attn.wk, &lw->attn.wv };
                    prefill_quant_matmul_preq8k_multi(m, qkv_out, qkv_w, 3,
                                                       n_tokens, pf_xq, pf_xd,
                                                       pf_xbs, Xb);
                }
            } else
#endif
            {
                t_prof = prefill_profile_now(&prof);
                if (prefill_qkv_stacked_batch_gpu(
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
                    float *qkv_out[3] = { Q_buf, K_new, V_new };
                    const BnQWeight *qkv_w[3] = {
                        &lw->attn.wq, &lw->attn.wk, &lw->attn.wv
                    };
                    prefill_quant_matmul_multi(m, qkv_out, qkv_w, 3, Xb,
                                               n_tokens, s->x_q);
                }
                prefill_profile_add(&prof.qkv_ms, t_prof);
            }
            }

            if (bn_model_tq_state(m) == NULL) {
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
                                prefill_rmsnorm(k_t + h * layer_head_size, k_t + h * layer_head_size,
                                                lw->attn.k_norm + h * qk_stride,
                                                layer_head_size, c->norm_eps);
                        }
                        if (c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4)
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
                    .out = Q_buf,
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
                    !getenv("BN_CUDA_DISABLE_PREFILL_ATTN") &&
                    n_tokens >= prefill_gpu_attention_min_tokens() &&
                    prefill_prepare_q_for_gpu_attention(&bctx) == 0) {
                    void *wo_buf = prefill_backend_role_or_qweight(
                        backend, l, BN_BACKEND_HANDLE_WO_PREFILL,
                        &lw->attn.wo);
                    if (gpu->prefill_attention_wo && wo_buf &&
                        !lw->norm.attn_sub_norm &&
                        !((c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) &&
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
                            prefill_rmsnorm(s->q + h * layer_head_size,
                                            s->q + h * layer_head_size,
                                            lw->attn.q_norm + h * qk_stride,
                                            layer_head_size, c->norm_eps);
                    }
                    if (lw->attn.k_norm) {
                        int qk_stride = c->qk_norm_per_head ? layer_head_size : 0;
                        for (int h = 0; h < layer_n_kv_heads; h++)
                            prefill_rmsnorm(k_t + h * layer_head_size,
                                            k_t + h * layer_head_size,
                                            lw->attn.k_norm + h * qk_stride,
                                            layer_head_size, c->norm_eps);
                    }
                    if (c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4)
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
                        prefill_rmsnorm(Q_buf + (size_t)t * wo_cols,
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
                    (c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) &&
                    lw->norm.attn_post_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_rmsnorm(Xb2 + (size_t)t * dim,
                                        Xb2 + (size_t)t * dim,
                                        lw->norm.attn_post_norm, dim, c->norm_eps);
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
            size_t state_per_layer = (size_t)num_v_heads * head_k_dim * head_v_dim;
            float *ssm_state = s->ssm_state + (size_t)ssm_idx * state_per_layer;
            size_t conv_per_layer = (size_t)(kern_ssm - 1) * qkv_dim_ssm;
            float *conv_state = s->ssm_conv_state + (size_t)ssm_idx * conv_per_layer;

            for (int t = 0; t < n_tokens; t++)
                prefill_rmsnorm(Xb + (size_t)t * dim, act + (size_t)t * dim,
                                lw->norm.attn_norm, dim, c->norm_eps);

            if (q_buf_stride < qkv_dim_ssm) { sh_arena_free(pf_arena); return NULL; }
            float *QKV_all = Q_buf;
            float *Z_all = Xb2;
            float *Out_all = Hb;

            prefill_quant_matmul_gpu(m, QKV_all, &lw->ssm.wqkv, Xb, n_tokens, s->x_q);
            prefill_quant_matmul_gpu(m, Z_all, &lw->ssm.wz, Xb, n_tokens, s->x_q);

            for (int t = 0; t < n_tokens; t++) {
                float *qkv_t = QKV_all + (size_t)t * lw->ssm.wqkv.rows;
                float *z_t = Z_all + (size_t)t * lw->ssm.wz.rows;
                float *out_t = Out_all + (size_t)t * value_dim;
                float *xb_t = Xb + (size_t)t * dim;

                BnSSMConvCtx conv_ctx = { qkv_t, conv_state, lw->ssm.ssm_conv1d,
                                          qkv_dim_ssm, kern_ssm };
                BnTPTask conv_task = { prefill_ssm_conv_silu, &conv_ctx, qkv_dim_ssm };
                bn_tp_dispatch(bn_model_pool(m), &conv_task, 1);

                float *q_raw = qkv_t;
                float *k_raw = qkv_t + key_dim_ssm;
                float *v_raw = qkv_t + 2 * key_dim_ssm;

                BnSSML2NormCtx norm_ctx = { q_raw, k_raw, head_k_dim };
                BnTPTask norm_task = { prefill_ssm_l2norm, &norm_ctx, num_k_heads };
                bn_tp_dispatch(bn_model_pool(m), &norm_task, 1);

                if (num_v_heads > BN_MAX_VLA_ELEMS) continue;
                float alpha_arr[num_v_heads > 0 ? num_v_heads : 1];
                float beta_arr[num_v_heads > 0 ? num_v_heads : 1];
                BnMatvecTask ab[2] = {
                    { alpha_arr, &lw->ssm.ssm_alpha, NULL, 0 },
                    { beta_arr,  &lw->ssm.ssm_beta, NULL, 0 },
                };
                bn_quant_matvec_batch(ab, 2, xb_t, s->x_q, bn_model_pool(m));
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
                BnTPTask delta_task = { prefill_ssm_delta, &delta_ctx, num_v_heads };
                bn_tp_dispatch(bn_model_pool(m), &delta_task, 1);

                BnSSMGateCtx gate_ctx = { out_t, z_t, lw->ssm.ssm_norm,
                                          c->norm_eps, head_v_dim };
                BnTPTask gate_task = { prefill_ssm_gate, &gate_ctx, num_v_heads };
                bn_tp_dispatch(bn_model_pool(m), &gate_task, 1);
            }

            prefill_quant_matmul_gpu(m, Xb, &lw->ssm.ssm_out, Out_all, n_tokens, s->x_q);

            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] += Xb[(size_t)t * dim + d];
        }

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
            void *ffn_norm_buf =
                backend_ffn ? bn_backend_model_handle(
                    backend_ffn, l, BN_BACKEND_HANDLE_FFN_NORM) : NULL;
            if (c->has_ffn_gate && !lw->norm.ffn_sub_norm &&
                !((c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) &&
                  lw->norm.ffn_post_norm) &&
                n_tokens >= prefill_gpu_attention_min_tokens() &&
                gpu_ffn && gpu_ffn->dense_ffn_batch_norm_resid &&
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
                    prefill_rmsnorm(Xb + t * dim, act + (size_t)t * dim,
                                    lw->norm.ffn_norm, dim, c->norm_eps);
                prefill_profile_add(&prof.ffn_norm_ms, t_prof);

                t_prof = prefill_profile_now(&prof);
                if (c->has_ffn_gate && !lw->norm.ffn_sub_norm &&
                    !((c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) &&
                      lw->norm.ffn_post_norm) &&
                    prefill_dense_ffn_gpu_batch(m, Xb, lw, Xb, n_tokens,
                                                dim, hidden_dim,
                                                c->act_type, l, NULL,
                                                0.0f, 0) == 0) {
                    used_gpu_batch_ffn = 1;
                } else if (c->has_ffn_gate) {
#ifdef __AVX2__
                if (pf_xq && !bn_model_gpu(m) &&
                    prefill_quant_can_preq8k_pair(lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type)) {
                    int n_bpr = dim / BN_QK_K;
                    for (int t = 0; t < n_tokens; t++)
                        bn_quant_x_to_q8k(Xb + (size_t)t * dim,
                                          pf_xq + (size_t)t * dim,
                                          pf_xd + (size_t)t * n_bpr,
                                          pf_xbs + (size_t)t * n_bpr * 16, dim);
                    {
                        float *gu_out[2] = { Hb, Hb2 };
                        const BnQWeight *gu_w[2] = { &lw->ffn.ffn_gate, &lw->ffn.ffn_up };
                        prefill_quant_matmul_preq8k_multi(m, gu_out, gu_w, 2,
                                                           n_tokens, pf_xq,
                                                           pf_xd, pf_xbs, Xb);
                    }
                } else
#endif
                {
                    float *gu_out[2] = { Hb, Hb2 };
                    const BnQWeight *gu_w[2] = {
                        &lw->ffn.ffn_gate, &lw->ffn.ffn_up
                    };
                    prefill_quant_matmul_multi(m, gu_out, gu_w, 2, Xb,
                                               n_tokens, s->x_q);
                }

                BnPrefillFFNActCtx act_ctx = { Hb, Hb2, hidden_dim, c->act_type };
                BnTPTask act_task = { prefill_ffn_activation_range, &act_ctx, n_tokens };
                bn_tp_dispatch(bn_model_pool(m), &act_task, 1);
            } else {
                prefill_quant_matmul_gpu(m, Hb, &lw->ffn.ffn_up, Xb, n_tokens, s->x_q);
                BnPrefillFFNActCtx act_ctx = { Hb, NULL, hidden_dim, c->act_type };
                BnTPTask act_task = { prefill_ffn_activation_range, &act_ctx, n_tokens };
                bn_tp_dispatch(bn_model_pool(m), &act_task, 1);
                }
            }

            if (!used_gpu_batch_ffn) {
                if (lw->norm.ffn_sub_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_rmsnorm(Hb + (size_t)t * hidden_dim,
                                        Hb + (size_t)t * hidden_dim,
                                        lw->norm.ffn_sub_norm, hidden_dim,
                                        c->norm_eps);

                prefill_quant_matmul_gpu(m, Xb, &lw->ffn.ffn_down, Hb,
                                         n_tokens, s->x_q);
                if ((c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) &&
                    lw->norm.ffn_post_norm)
                    for (int t = 0; t < n_tokens; t++)
                        prefill_rmsnorm(Xb + (size_t)t * dim,
                                        Xb + (size_t)t * dim,
                                        lw->norm.ffn_post_norm, dim,
                                        c->norm_eps);
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

        if ((c->arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4) && lw->norm.layer_output_scale) {
            float scale = lw->norm.layer_output_scale[0];
            for (int t = 0; t < n_tokens; t++)
                for (int d = 0; d < dim; d++)
                    act[(size_t)t * dim + d] *= scale;
        }
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
                    "[bn:prefill:profile] tokens=%d layers=%d embed=%.3f attn_norm=%.3f qkv=%.3f attn_cpu=%.3f wo=%.3f ffn_norm=%.3f ffn=%.3f residual=%.3f logits=%.3f\n",
                    n_tokens, c->n_layers, prof.embed_ms, prof.attn_norm_ms,
                    prof.qkv_ms, prof.attn_cpu_ms, prof.wo_ms,
                    prof.ffn_norm_ms, prof.ffn_ms, prof.residual_ms,
                    prof.logits_ms);
        }
        sh_arena_free(pf_arena);
        return s->logits;
    }

    if (prof.enabled) {
        fprintf(stderr,
                "[bn:prefill:profile] tokens=%d layers=%d embed=%.3f attn_norm=%.3f qkv=%.3f attn_cpu=%.3f wo=%.3f ffn_norm=%.3f ffn=%.3f residual=%.3f logits=%.3f\n",
                n_tokens, c->n_layers, prof.embed_ms, prof.attn_norm_ms,
                prof.qkv_ms, prof.attn_cpu_ms, prof.wo_ms,
                prof.ffn_norm_ms, prof.ffn_ms, prof.residual_ms,
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
    return prefill_internal(m, s, tokens, n_tokens, pos0, NULL, 1);
}

int bn_transformer_prefill_no_logits(BnModel *m, BnSession *s, const int *tokens,
                                     int n_tokens, int pos0) {
    return prefill_internal(m, s, tokens, n_tokens, pos0, NULL, 0) ? 0 : -1;
}

int bn_transformer_prefill_all(BnModel *m, BnSession *s, const int *tokens,
                               int n_tokens, int pos0, float *all_logits) {
    if (!all_logits || n_tokens <= 0) return -1;

    if (n_tokens == 1) {
        float *logits = bn_transformer_forward(m, s, tokens[0], pos0);
        if (!logits) return -1;
        memcpy(all_logits, logits, m->config.vocab_size * sizeof(float));
        return 0;
    }

    float *result = prefill_internal(m, s, tokens, n_tokens, pos0, all_logits, 1);
    return result ? 0 : -1;
}
