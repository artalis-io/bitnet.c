#ifndef BN_TRANSFORMER_CPU_BACKEND_INTERNAL_H
#define BN_TRANSFORMER_CPU_BACKEND_INTERNAL_H

#include "gpu_backend.h"
#include "model_run_state.h"
#include "quant.h"
#include "threadpool.h"
#include "transformer_plan_internal.h"
#include <stdint.h>

typedef struct {
    const char *name;
    void (*rmsnorm)(float *out, const float *x, const float *w,
                    int size, float eps);
    bn_tp_fn gqa;
    bn_tp_fn flash_gqa;
    bn_tp_fn batched_attn_naive;
    bn_tp_fn batched_attn_flash;
    bn_tp_fn batched_attn_flash_pair;
    void (*residual_add)(float *x, const float *r, int dim);
    bn_tp_fn ssm_conv_silu;
    bn_tp_fn ssm_l2norm;
    bn_tp_fn ssm_delta;
    bn_tp_fn ssm_gate;
    void (*apply_ffn_activation)(BnRunState *s, const BnFFNPlan *ffn_plan,
                                 int hidden_dim);
    void (*apply_sigmoid_gate)(float *x, const float *gate, int size);
    void (*apply_rope_heads)(float *buf, int n_heads, int head_size,
                             int rope_dims, const float *rc,
                             const float *rs);
    int supports_prepared_kquant;
    int supports_float_kquant_prefill;
    void (*rmsnorm_prepared_kquant)(const float *x, const float *w, int dim,
                                    float eps, float *out,
                                    int8_t *quantized, float *scales,
                                    int16_t *block_sums);
} BnCPUBackendOps;

typedef struct {
    int apply;
} BnTransformerCPUPostNormPolicy;

typedef struct {
    int apply;
} BnTransformerCPULayerOutputScalePolicy;

const BnCPUBackendOps *bn_transformer_cpu_backend_ops(void);
int bn_transformer_cpu_prepared_qweights_enabled(void);
const char *bn_transformer_cpu_debug_dump_path(void);
int bn_transformer_cpu_debug_dump_pos_selected(int pos);
int bn_transformer_cpu_debug_dump_heads_enabled(void);
int bn_transformer_cpu_fused_kquant_gateup_silu_allowed(void);
int bn_transformer_cpu_can_fused_kquant_gateup_silu(int gate_type, int up_type);
int bn_transformer_cpu_can_prepared_kquant_pair(const BnCPUBackendOps *ops,
                                       int left_type,
                                       int right_type);
int bn_transformer_cpu_can_prepared_kquant_triple(const BnCPUBackendOps *ops,
                                         int first_type,
                                         int second_type,
                                         int third_type);
int bn_transformer_cpu_route_prepared_kquant_pair_enabled(
    const BnCPUBackendOps *ops,
    const BnGPUBackend *gpu,
    int dim,
    int left_type,
    int right_type);
int bn_transformer_cpu_route_prepared_kquant_triple_enabled(
    const BnCPUBackendOps *ops,
    const BnGPUBackend *gpu,
    int dim,
    int first_type,
    int second_type,
    int third_type);
int bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
    const BnGPUBackend *gpu,
    const BnFFNPlan *ffn_plan,
    int dim,
    int gate_type,
    int up_type);
int bn_transformer_cpu_gpu_dense_ffn_fast_path_available(
    const BnGPUBackend *gpu,
    const BnFFNPlan *ffn_plan);
int bn_transformer_cpu_activation_is_relu2(int activation);
int bn_transformer_cpu_activation_is_gelu(int activation);
int bn_transformer_cpu_activation_uses_silu_path(int activation);
uint32_t bn_transformer_cpu_float_kquant_task_flags(int enabled);
BnTransformerCPUPostNormPolicy
bn_transformer_cpu_attention_post_norm_policy(int uses_attention_post_norm,
                                              int has_attn_post_norm);
BnTransformerCPUPostNormPolicy
bn_transformer_cpu_ffn_post_norm_policy(int uses_ffn_post_norm,
                                        int has_ffn_post_norm);
BnTransformerCPULayerOutputScalePolicy
bn_transformer_cpu_layer_output_scale_policy(int uses_layer_output_scale,
                                             int has_layer_output_scale);
bn_tp_fn bn_transformer_cpu_ssm_conv_silu_op(const BnConfig *c,
                                             const BnCPUBackendOps *ops);
bn_tp_fn bn_transformer_cpu_ssm_l2norm_op(const BnConfig *c,
                                          const BnCPUBackendOps *ops);
bn_tp_fn bn_transformer_cpu_ssm_delta_op(const BnConfig *c,
                                         const BnCPUBackendOps *ops);
bn_tp_fn bn_transformer_cpu_ssm_gate_op(const BnConfig *c,
                                        const BnCPUBackendOps *ops);
int bn_transformer_cpu_backend_supports_float_kquant_prefill(void);
int bn_transformer_cpu_backend_supports_mixed_shared_gateup_batch(void);
int bn_transformer_cpu_has_native_q8x_quant(void);
void bn_transformer_cpu_prepare_kquant_activation(const float *x,
                                                  int8_t *quantized,
                                                  float *scales,
                                                  int16_t *block_sums,
                                                  int n);
int bn_transformer_cpu_quantize_q8_blocks_native(const float *x,
                                                 int8_t *quantized,
                                                 float *scales,
                                                 int n);
float bn_transformer_cpu_quantize_i8_activation(const float *x,
                                                int8_t *quantized,
                                                int n,
                                                int use_standard_quant);
void bn_transformer_cpu_prepare_q8_logits_refine_activation(const float *x,
                                                            int8_t *quantized,
                                                            float *scales,
                                                            int n);
int bn_transformer_cpu_refine_q8_logits_row(const BnQWeight *weight,
                                            const int8_t *quantized,
                                            const float *scales,
                                            int row,
                                            float *out);
int bn_transformer_cpu_refine_q6_logits_row(const BnQWeight *weight,
                                            const float *x,
                                            int row,
                                            float *out);
void bn_transformer_cpu_quant_matvec(float *out,
                                     const BnQWeight *weight,
                                     const float *x,
                                     int8_t *quantized_buf,
                                     BnThreadPool *pool);
void bn_transformer_cpu_quant_matvec_prepared_flags(
    float *out,
    const BnQWeight *weight,
    const BnPreparedWeight *prepared,
    const float *x,
    int8_t *quantized_buf,
    BnThreadPool *pool,
    uint32_t flags);
void bn_transformer_cpu_quant_matvec_batch_gpu_buffers(
    const BnMatvecTask *tasks,
    const void **buffers,
    int n_tasks,
    const float *x,
    int8_t *quantized_buf,
    BnThreadPool *pool,
    BnGPUBackend *gpu);

#endif // BN_TRANSFORMER_CPU_BACKEND_INTERNAL_H
