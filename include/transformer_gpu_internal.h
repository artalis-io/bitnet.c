#ifndef BN_TRANSFORMER_GPU_INTERNAL_H
#define BN_TRANSFORMER_GPU_INTERNAL_H

#include "backend_model.h"
#include "gpu_backend.h"
#include "gpu_shader_ir.h"
#include "model.h"
#include "session.h"
#include "transformer_plan_internal.h"

void bn_transformer_gpu_finalize_op_kinds(BnGPUOp *ops, int n);
void bn_transformer_gpu_emit_rmsnorm(BnGPUOp *ops, int *n,
                                     void *norm_gpu,
                                     int buf_in,
                                     int buf_out,
                                     int dim,
                                     uint32_t u_eps);
void bn_transformer_gpu_emit_logits(BnGPUOp *ops, int *n,
                                    void *logit_gpu_buf,
                                    int logit_type,
                                    int logit_rows,
                                    int logit_cols);
void bn_transformer_gpu_emit_dense_ffn(BnGPUOp *ops, int *n,
                                       const BnConfig *c,
                                       const BnLayerWeights *lw,
                                       const BnFFNPlan *ffn_plan,
                                       const BnGPUBackend *gpu,
                                       const BnBackendModel *backend,
                                       int layer,
                                       int dim,
                                       uint32_t u_eps,
                                       void *next_norm);
void bn_transformer_gpu_emit_attention(BnGPUOp *ops, int *n,
                                       const BnConfig *c,
                                       const BnLayerWeights *lw,
                                       const BnGPUBackend *gpu,
                                       const BnBackendModel *backend,
                                       int layer,
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
void bn_transformer_gpu_emit_qkv(BnGPUOp *ops, int *n,
                                 const BnConfig *c,
                                 const BnLayerWeights *lw,
                                 const BnLayerShapePlan *plan,
                                 const BnGPUBackend *gpu,
                                 const BnBackendModel *backend,
                                 int layer,
                                 int pos,
                                 int q_dim,
                                 int head_size,
                                 int n_heads,
                                 int kv_dim,
                                 int rope_dims,
                                 uint32_t kv_cache_off,
                                 uint32_t u_eps);
void bn_transformer_gpu_emit_ssm(BnGPUOp *ops, int *n,
                                 const BnConfig *c,
                                 const BnLayerWeights *lw,
                                 const BnLayerShapePlan *plan,
                                 const BnGPUBackend *gpu,
                                 const BnBackendModel *backend,
                                 int layer,
                                 int dim,
                                 uint32_t u_eps);
void bn_transformer_gpu_emit_moe(BnGPUOp *ops, int *n,
                                 BnModel *m,
                                 BnSession *sess,
                                 const BnLayerWeights *lw,
                                 int layer,
                                 int dim,
                                 uint32_t u_eps,
                                 void *next_norm,
                                 void **uncached_bufs,
                                 int *n_uncached);

#endif // BN_TRANSFORMER_GPU_INTERNAL_H
