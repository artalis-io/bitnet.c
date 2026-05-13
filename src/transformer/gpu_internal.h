#ifndef BN_TRANSFORMER_GPU_INTERNAL_H
#define BN_TRANSFORMER_GPU_INTERNAL_H

#include "backend_session.h"
#include "backend_model.h"
#include "gpu_backend.h"
#include "gpu_graph_ir.h"
#include "gpu_moe_bridge.h"
#include "model.h"
#include "session.h"
#include "transformer_plan_internal.h"

#define BN_TRANSFORMER_GPU_MAX_VLA_ELEMS 8192

typedef struct {
    void *lowered_ops;
    int n;
    int cap;
    BnGPUValueGraph graph_storage;
    BnGPUValueGraph *graph;
    void *lowering_values;
    int cap_lowering_values;
    int owns_graph_storage;
    int owns_lowering_values;
} BnTransformerGPUEmitContext;

typedef struct {
    const BnGPUBackend *gpu;
    void *gateup_stacked;
    void *ffn_sub_norm;
    void *ffn_gate;
    void *ffn_up;
    void *ffn_down;
} BnTransformerGPUDenseFFNResources;

typedef struct {
    const BnGPUBackend *gpu;
    void *q_bias;
    void *k_bias;
    void *v_bias;
    void *q_norm;
    void *k_norm;
    void *qkv_stacked;
    void *packed_qkv;
    void *wq;
    void *wk;
    void *wv;
} BnTransformerGPUQKVResources;

typedef struct {
    const BnGPUBackend *gpu;
    void *k_bias;
    void *attn_sub_norm;
    void *ffn_norm;
    void *wo;
} BnTransformerGPUAttentionResources;

typedef struct {
    const BnGPUBackend *gpu;
    void *ssm_qkvz_stacked;
    void *ssm_ab_stacked;
    void *ssm_conv1d;
    void *ssm_dt_bias;
    void *ssm_a_log;
    void *ssm_norm;
    void *ffn_norm;
    void *wqkv;
    void *wz;
    void *ssm_alpha;
    void *ssm_beta;
    void *ssm_out;
} BnTransformerGPUSSMResources;

typedef struct {
    void *shared_gate;
    void *shared_up;
    void *shared_down;
} BnTransformerGPUMoESharedResources;

typedef struct {
    void *attn_norm;
    void *ffn_norm;
    void *q_norm;
    void *k_norm;
    void *attn_sub_norm;
    void *ffn_sub_norm;
} BnTransformerGPULayerValidationResources;

typedef struct {
    void *gpu_buf;
    int type;
    int rows;
    int cols;
    BnQWeight *cpu_weight;
    BnQWeight tied_weight;
} BnTransformerGPULogitResources;

typedef struct {
    void *output_norm;
    BnTransformerGPULogitResources logits;
    int has_moe;
    int has_ssm;
} BnTransformerGPUForwardPolicy;

int bn_transformer_gpu_validate_forward(
    BnTransformerGPUForwardPolicy *out,
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnConfig *c,
    const BnWeights *w,
    int token,
    int pos,
    const char **reject_reason);
int bn_transformer_gpu_graph_op_capacity(const BnConfig *c);
int bn_transformer_gpu_logits_needs_cpu_fallback(
    const BnGPUBackend *gpu,
    const BnTransformerGPULogitResources *logits);
void bn_transformer_gpu_report_fallback(const char *reason);
float *bn_transformer_gpu_reject_forward(
    BnTransformerGPUEmitContext *emit,
    const char *reason);
void *bn_transformer_gpu_resolve_output_norm(
    const BnBackendModel *backend);
void *bn_transformer_gpu_resolve_initial_norm(
    const BnBackendModel *backend);
void *bn_transformer_gpu_resolve_next_norm(
    const BnBackendModel *backend,
    int layer,
    int n_layers,
    void *output_norm);
BnTransformerGPULayerValidationResources
bn_transformer_gpu_resolve_layer_validation_resources(
    const BnBackendModel *backend,
    int layer);
void bn_transformer_gpu_resolve_logit_resources(
    BnTransformerGPULogitResources *out,
    const BnBackendModel *backend,
    const BnConfig *c,
    const BnWeights *w);
BnTransformerGPUDenseFFNResources
bn_transformer_gpu_resolve_dense_ffn_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer);
BnTransformerGPUQKVResources bn_transformer_gpu_resolve_qkv_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer);
BnTransformerGPUAttentionResources
bn_transformer_gpu_resolve_attention_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer);
BnTransformerGPUSSMResources bn_transformer_gpu_resolve_ssm_resources(
    const BnGPUBackend *gpu,
    const BnBackendModel *backend,
    const BnLayerWeights *lw,
    int layer);
BnTransformerGPUMoESharedResources
bn_transformer_gpu_resolve_moe_shared_resources(
    const BnBackendModel *backend,
    const BnLayerWeights *lw);

void bn_transformer_gpu_finalize_op_kinds(void *ops, int n);
void bn_transformer_gpu_emit_context_init(BnTransformerGPUEmitContext *ctx,
                                          void *lowered_ops,
                                          int cap);
int bn_transformer_gpu_emit_context_init_session(
    BnTransformerGPUEmitContext *ctx,
    BnBackendSession *backend,
    void *lowered_ops,
    int cap,
    int cap_values,
    int cap_ops);
int bn_transformer_gpu_emit_context_reserve(
    BnTransformerGPUEmitContext *ctx,
    int cap_values,
    int cap_ops);
void bn_transformer_gpu_emit_context_free(BnTransformerGPUEmitContext *ctx);
int bn_transformer_gpu_emit_context_lower_pending(
    BnTransformerGPUEmitContext *ctx);
int bn_transformer_gpu_emit_context_execute(
    BnTransformerGPUEmitContext *ctx,
    const BnGPUBackend *gpu,
    int readback_buf,
    float *readback,
    int readback_count);
int bn_transformer_gpu_emit_context_flush(
    BnTransformerGPUEmitContext *ctx,
    const BnGPUBackend *gpu);
int bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
    BnTransformerGPUEmitContext *ctx,
    void *norm_gpu,
    int dim,
    uint32_t u_eps);
int bn_transformer_gpu_emit_context_execute_logits(
    BnTransformerGPUEmitContext *ctx,
    const BnGPUBackend *gpu,
    float *logits,
    int vocab_size);
int bn_transformer_gpu_emit_context_rmsnorm(BnTransformerGPUEmitContext *ctx,
                                            void *norm_gpu,
                                            int buf_in,
                                            int buf_out,
                                            int dim,
                                            uint32_t u_eps);
int bn_transformer_gpu_emit_context_logits(BnTransformerGPUEmitContext *ctx,
                                           void *logit_gpu_buf,
                                           int logit_type,
                                           int logit_rows,
                                           int logit_cols);
int bn_transformer_gpu_emit_context_copy(BnTransformerGPUEmitContext *ctx,
                                         int buf_in,
                                         int buf_out,
                                         int src_offset,
                                         int dst_offset,
                                         int count);
int bn_transformer_gpu_emit_context_residual_add(
    BnTransformerGPUEmitContext *ctx,
    int buf_in,
    int buf_aux,
    int count);
int bn_transformer_gpu_emit_context_activation(
    BnTransformerGPUEmitContext *ctx,
    int buf_in,
    int buf_aux,
    int count,
    int param1,
    BnGPUIRActivationKind kind);
int bn_transformer_gpu_emit_context_matvec(BnTransformerGPUEmitContext *ctx,
                                           int type,
                                           void *weight_buf,
                                           int buf_in,
                                           int buf_out,
                                           int rows,
                                           int cols,
                                           int output_offset);
int bn_transformer_gpu_emit_context_fused_gateup_silu(
    BnTransformerGPUEmitContext *ctx,
    int type,
    void *weight_buf,
    int buf_in,
    int buf_out,
    int gate_rows,
    int up_rows,
    int cols);
int bn_transformer_gpu_fallback_ssm_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    uint32_t u_eps,
    void *next_norm);
int bn_transformer_gpu_fallback_moe_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    uint32_t u_eps,
    void *next_norm);
int bn_transformer_gpu_fallback_logits(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    const BnTransformerGPULogitResources *logits,
    int dim);
int bn_transformer_gpu_execute_ops(const BnGPUBackend *gpu,
                                   void *ops,
                                   int n,
                                   int readback_buf,
                                   float *readback,
                                   int readback_count);
int bn_transformer_gpu_write_x(const BnGPUBackend *gpu,
                               const float *x,
                               size_t size_bytes);
int bn_transformer_gpu_read_x(const BnGPUBackend *gpu,
                              float *x,
                              size_t size_bytes);
int bn_transformer_gpu_read_xb(const BnGPUBackend *gpu,
                               float *xb,
                               size_t size_bytes);
void bn_transformer_gpu_emit_context_dense_ffn(
    BnTransformerGPUEmitContext *ctx,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnFFNPlan *ffn_plan,
    const BnTransformerGPUDenseFFNResources *res,
    int dim,
    uint32_t u_eps,
    void *next_norm);
void bn_transformer_gpu_emit_context_attention(
    BnTransformerGPUEmitContext *ctx,
    const BnConfig *c,
    const BnLayerWeights *lw,
    const BnTransformerGPUAttentionResources *res,
    int pos,
    int dim,
    int q_dim,
    int head_size,
    int n_heads,
    int kv_dim,
    int rope_dims,
    int n_kv,
    size_t loff,
    uint32_t kv_cache_off,
    int has_moe,
    uint32_t u_eps);
void bn_transformer_gpu_emit_context_qkv(BnTransformerGPUEmitContext *ctx,
                                         const BnConfig *c,
                                         const BnLayerWeights *lw,
                                         const BnLayerShapePlan *plan,
                                         const BnTransformerGPUQKVResources *res,
                                         int pos,
                                         int q_dim,
                                         int head_size,
                                         int n_heads,
                                         int kv_dim,
                                         int rope_dims,
                                         uint32_t kv_cache_off,
                                         uint32_t u_eps);
void bn_transformer_gpu_emit_context_ssm(BnTransformerGPUEmitContext *ctx,
                                         const BnConfig *c,
                                         const BnLayerWeights *lw,
                                         const BnLayerShapePlan *plan,
                                         const BnTransformerGPUSSMResources *res,
                                         int dim,
                                         uint32_t u_eps);
void bn_transformer_gpu_emit_context_moe(BnTransformerGPUEmitContext *ctx,
                                         const BnGPUMoEResources *moe,
                                         const BnTransformerGPUMoESharedResources *shared,
                                         const BnLayerWeights *lw,
                                         int dim,
                                         uint32_t u_eps,
                                         void *next_norm);

#endif // BN_TRANSFORMER_GPU_INTERNAL_H
