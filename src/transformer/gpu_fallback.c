#include "gpu_internal.h"
#include "transformer_cpu_internal.h"
#include "moe.h"
#include "quant.h"

static int gpu_fallback_flush(BnTransformerGPUEmitContext *emit,
                              const BnGPUBackend *gpu) {
    if (emit->n == 0 && emit->graph.n_ops == 0)
        return 0;
    return bn_transformer_gpu_emit_context_execute(emit, gpu, -1, NULL, 0);
}

int bn_transformer_gpu_fallback_ssm_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (gpu_fallback_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    bn_transformer_cpu_forward_ssm_block(m, sess, lw, layer);
    bn_transformer_cpu_residual_add(s->x, s->xb, dim);
    if (lw->moe.router_weight)
        bn_moe_forward(m, sess, lw, layer);
    else
        bn_transformer_cpu_forward_ffn_block(m, sess, lw, NULL);
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_rmsnorm(
        emit, next_norm, BN_GPU_VALUE_X, BN_GPU_VALUE_XB, dim, u_eps);
}

int bn_transformer_gpu_fallback_moe_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (gpu_fallback_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    bn_moe_forward(m, sess, lw, layer);
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_rmsnorm(
        emit, next_norm, BN_GPU_VALUE_X, BN_GPU_VALUE_XB, dim, u_eps);
}

int bn_transformer_gpu_fallback_logits(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    const BnTransformerGPULogitResources *logits,
    int dim) {
    BnRunState *s = &sess->state;
    if (gpu_fallback_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_xb(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    bn_quant_matvec(s->logits, logits->cpu_weight, s->x, s->x_q,
                    bn_model_pool(m));
    return 0;
}
