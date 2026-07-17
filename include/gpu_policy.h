#ifndef BN_GPU_POLICY_H
#define BN_GPU_POLICY_H

#include "gpu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

int bn_gpu_policy_cuda_moe_routed_ffn_enabled(int eligible);
int bn_gpu_policy_float_buffer_type(void);
int bn_gpu_policy_cuda_moe_resident_routed_ffn_quant_eligible(
    int gate_type,
    int up_type,
    int down_type);
int bn_gpu_policy_cuda_moe_all_f16_cache_forced(void);
int bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
    const BnGPUBackend *gpu,
    int tensor_type,
    int q8_f16_cache);
int bn_gpu_policy_cuda_moe_gateup_f16_cache_enabled(int eligible);
int bn_gpu_policy_cuda_partial_moe_f16_cache_enabled(int eligible);
int bn_gpu_policy_cuda_moe_fit_debug_enabled(void);
int bn_gpu_policy_cuda_keep_individual_f16_cache_enabled(void);
int bn_gpu_policy_cuda_individual_upload_quant_only_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type);
int bn_gpu_policy_cuda_logits_f16_cache_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced(void);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(int cols);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_preferred(
    const BnGPUBackend *gpu,
    int tensor_type,
    int cols,
    int force_f16_cache);
size_t bn_gpu_policy_cuda_moe_down_q6_f32_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols,
    int n_experts);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_requires_full_buffer(
    int tensor_type);
int bn_gpu_policy_cuda_moe_down_q4_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type);
int bn_gpu_policy_cuda_moe_quant_only_after_cache(int tensor_type,
                                                  int q8_f16_cache);
int bn_gpu_policy_cuda_moe_prefers_quant_only(int tensor_type);
int bn_gpu_policy_cuda_matvec_disabled(void);
int bn_gpu_policy_cuda_matvec_type_disabled(int tensor_type);
size_t bn_gpu_policy_cuda_layout_reserve_bytes(void);
size_t bn_gpu_policy_cuda_moe_full_reserve_bytes(void);
int bn_gpu_policy_cuda_cublas_matmul_enabled(void);
int bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled(void);
int bn_gpu_policy_cuda_q8_0_quant_matmul_enabled(void);
int bn_gpu_policy_cuda_f16_q8_0_matmul_enabled(void);
int bn_gpu_policy_cuda_q8_0_preq_split_enabled(void);
int bn_gpu_policy_cuda_decode_logits_cache_enabled(int gpu_logits_need_cpu);
int bn_gpu_policy_cuda_moe_decode_cache_enabled(void);
int bn_gpu_policy_cuda_moe_decode_cache_disabled(void);
int bn_gpu_policy_cuda_decode_cache_disabled(void);
int bn_gpu_policy_cuda_q4_q8_decode_cache_disabled(void);
int bn_gpu_policy_cuda_logits_argmax_disabled(void);
int bn_gpu_policy_cuda_dense_logits_argmax_enabled(void);
int bn_gpu_policy_cuda_moe_logits_mmvq_argmax_enabled(void);
int bn_gpu_policy_cuda_moe_logits_mmvq_argmax_disabled(void);
int bn_gpu_policy_cuda_prefill_ssm_layer_disabled(void);
int bn_gpu_policy_cuda_q5k_fused_gateup_enabled(void);
int bn_gpu_policy_cuda_shared_q4_q8_dot_enabled(void);
int bn_gpu_policy_cuda_shared_expert_gate_enabled(void);
int bn_gpu_policy_cuda_prefill_attention_min_tokens_configured(void);
int bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(
    int default_tokens);
int bn_gpu_policy_cuda_prefill_dense_chain_enabled(void);
int bn_gpu_policy_cuda_prefill_hybrid_chain_enabled(void);
int bn_gpu_policy_cuda_prefill_attention_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_run_chain_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_ffn_fuse_allowed(void);
int bn_gpu_policy_cuda_prefill_moe_chain_debug_enabled(void);
int bn_gpu_policy_cuda_prefill_hybrid_chain_debug_enabled(void);
int bn_gpu_policy_cuda_moe_prefill_enabled(void);
int bn_gpu_policy_cuda_moe_prefill_min_tokens_configured(void);
int bn_gpu_policy_cuda_moe_prefill_min_tokens_or_default(
    int default_tokens);
int bn_gpu_policy_cuda_moe_cache_prefill_enabled(void);
int bn_gpu_policy_cuda_moe_prefill_shared_fuse_enabled(void);
int bn_gpu_policy_cuda_moe_route_batch_debug_enabled(void);
int bn_gpu_policy_cuda_large_hybrid_attention_enabled(void);
int bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_enabled(void);
int bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_disabled(void);
int bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_forced(void);
int bn_gpu_policy_cuda_large_hybrid_prefill_enabled(void);
int bn_gpu_policy_cuda_large_hybrid_prefill_chain_enabled(void);
int bn_gpu_policy_cuda_large_hybrid_prefill_disabled(void);
int bn_gpu_policy_cuda_large_hybrid_argmax_enabled(void);
int bn_gpu_policy_cuda_cublas_cache_max_mb(int default_mb,
                                           int large_budget);
int bn_gpu_policy_cuda_cublas_aux_cache_max_mb(int tensor_type,
                                               int force_q6_f32,
                                               int force_f16);
int bn_gpu_policy_cuda_q6k_f16_cache_adds_f32_down_cache(void);
size_t bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols);
size_t bn_gpu_policy_cuda_aux_cache_bytes(int tensor_type,
                                          int rows,
                                          int cols);
int bn_gpu_policy_moe_auto_resident_enabled(void);
int bn_gpu_policy_auto_caps_sequence(int webgpu,
                                     int cuda,
                                     int metal,
                                     int has_moe,
                                     int model_seq_len,
                                     int cap_seq_len);
int bn_gpu_policy_cuda_duplicate_moe_cache_enabled(void);
int bn_gpu_policy_metal_mmap_zero_copy_enabled(void);
void bn_gpu_policy_metal_apply_q4_q8_default(void);
int bn_gpu_policy_metal_q4_q8_enabled(void);
int bn_gpu_policy_metal_q4_prepared_enabled(void);
int bn_gpu_policy_metal_q4_prepared_upload_enabled(void);
int bn_gpu_policy_metal_shared_weights_enabled(void);
int bn_gpu_policy_metal_q6_q8k_enabled(void);
int bn_gpu_policy_metal_q8_barriers_enabled(void);
int bn_gpu_policy_metal_cpu_order_rmsnorm_enabled(void);
int bn_gpu_policy_metal_full_barriers_enabled(void);
int bn_gpu_policy_metal_barriers_enabled(void);
int bn_gpu_policy_metal_barriers_disabled(void);
int bn_gpu_policy_fused_gateup_enabled(void);
int bn_gpu_policy_q4_q8_fused_gateup_enabled(void);
int bn_gpu_policy_gateup_split_enabled(void);
int bn_gpu_policy_q4_q8_ffn_down_enabled(void);
int bn_gpu_policy_qkv_split_enabled(void);
int bn_gpu_policy_qkv_split_debug_enabled(void);
int bn_gpu_policy_ssm_qkvz_split_enabled(void);
int bn_gpu_policy_ssm_ab_stack_enabled(void);
int bn_gpu_policy_split_residual_rmsnorm_enabled(void);
int bn_gpu_policy_argmax_debug_enabled(void);
int bn_gpu_policy_profile_level(void);

#ifdef __cplusplus
}
#endif

#endif // BN_GPU_POLICY_H
