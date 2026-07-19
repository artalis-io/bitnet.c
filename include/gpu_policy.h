#ifndef BN_GPU_POLICY_H
#define BN_GPU_POLICY_H

#include "backend_placement.h"
#include "gguf.h"
#include "gpu_backend.h"
#include "model_config.h"

#ifdef __cplusplus
extern "C" {
#endif

int bn_gpu_policy_cuda_moe_routed_ffn_enabled(int eligible);
int bn_gpu_policy_moe_resident_routed_ffn_enabled(int eligible);
int bn_gpu_policy_backend_is_cuda(const BnGPUBackend *gpu);
BnBackendPlacement bn_gpu_policy_backend_placement(const BnGPUBackend *gpu);
int bn_gpu_policy_float_buffer_type(void);
int bn_gpu_policy_attention_layer_count(const BnConfig *c);
int bn_gpu_policy_ssm_layer_count(const BnConfig *c);
int bn_gpu_policy_uses_hybrid_ssm(const BnConfig *c);
int bn_gpu_policy_uses_hybrid_moe(const BnConfig *c);
int bn_gpu_policy_uses_moe(const BnConfig *c);
int bn_gpu_policy_moe_router_diff2_upload_enabled(const BnConfig *c);
int bn_gpu_policy_cuda_moe_f16_aux_cache_auto_enabled(const BnConfig *c);
int bn_gpu_policy_moe_f16_aux_cache_auto_enabled(const BnConfig *c);
int bn_gpu_policy_cuda_moe_resident_routed_ffn_quant_eligible(
    int gate_type,
    int up_type,
    int down_type);
int bn_gpu_policy_moe_resident_routed_ffn_quant_eligible(
    int gate_type,
    int up_type,
    int down_type);
int bn_gpu_policy_cuda_moe_all_f16_cache_forced(void);
int bn_gpu_policy_moe_all_f16_cache_forced(void);
int bn_gpu_policy_cuda_moe_all_f16_cache_enabled_for_type(
    const BnGPUBackend *gpu,
    int tensor_type,
    int q8_f16_cache);
int bn_gpu_policy_moe_all_f16_cache_enabled_for_type(
    const BnGPUBackend *gpu,
    int tensor_type,
    int q8_f16_cache);
int bn_gpu_policy_cuda_moe_gateup_f16_cache_enabled(int eligible);
int bn_gpu_policy_moe_gateup_f16_cache_enabled(int eligible);
int bn_gpu_policy_cuda_partial_moe_f16_cache_enabled(int eligible);
int bn_gpu_policy_partial_moe_f16_cache_enabled(int eligible);
int bn_gpu_policy_cuda_moe_fit_debug_enabled(void);
int bn_gpu_policy_moe_residency_fit_debug_enabled(void);
int bn_gpu_policy_cuda_keep_individual_f16_cache_enabled(void);
int bn_gpu_policy_moe_lazy_aux_cache_enabled(void);
int bn_gpu_policy_cuda_moe_lazy_aux_cache_enabled(void);
int bn_gpu_policy_cuda_individual_upload_quant_only_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_individual_upload_quant_only_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type);
int bn_gpu_policy_logits_q6_f32_cache_enabled(const BnGPUBackend *gpu,
                                              int tensor_type);
int bn_gpu_policy_cuda_logits_f16_cache_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_logits_f16_cache_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_cublas_logits_enabled(void);
int bn_gpu_policy_cuda_f32_logits_matvec_enabled(void);
int bn_gpu_policy_cuda_f16_logits_matvec_enabled(void);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced(void);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols(int cols);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_preferred(
    const BnGPUBackend *gpu,
    int tensor_type,
    int cols,
    int force_f16_cache);
int bn_gpu_policy_moe_down_q6_f32_cache_preferred(
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
size_t bn_gpu_policy_moe_down_q6_f32_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols,
    int n_experts);
int bn_gpu_policy_cuda_moe_down_q6_f32_cache_requires_full_buffer(
    int tensor_type);
int bn_gpu_policy_moe_down_q6_f32_cache_requires_full_buffer(
    int tensor_type);
int bn_gpu_policy_cuda_moe_down_q4_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type);
int bn_gpu_policy_moe_down_q4_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type);
int bn_gpu_policy_cuda_moe_quant_only_after_cache(int tensor_type,
                                                  int q8_f16_cache);
int bn_gpu_policy_moe_quant_only_after_cache(int tensor_type,
                                             int q8_f16_cache);
int bn_gpu_policy_moe_prefers_quant_only(const BnGPUBackend *gpu,
                                         int tensor_type);
int bn_gpu_policy_cuda_moe_prefers_quant_only(int tensor_type);
int bn_gpu_policy_cuda_matvec_disabled(void);
int bn_gpu_policy_cuda_matvec_type_disabled(int tensor_type);
int bn_gpu_policy_cuda_matvec_type_supported(int tensor_type);
int bn_gpu_policy_cuda_matmul_batch_enabled(void);
int bn_gpu_policy_cuda_matvec_batch_enabled(void);
int bn_gpu_policy_small_kquant_native_enabled(int force_float_kquant);
int bn_gpu_policy_small_kquant_native_disabled(void);
int bn_gpu_policy_cuda_small_kquant_native_enabled(int force_float_kquant);
int bn_gpu_policy_cuda_small_kquant_native_disabled(void);
size_t bn_gpu_policy_max_storage_binding_bytes(size_t backend_limit);
size_t bn_gpu_policy_cuda_layout_reserve_bytes(void);
size_t bn_gpu_policy_layout_reserve_bytes(void);
size_t bn_gpu_policy_cuda_moe_full_reserve_bytes(void);
size_t bn_gpu_policy_moe_full_reserve_bytes(void);
int bn_gpu_policy_cuda_cublas_matmul_enabled(void);
int bn_gpu_policy_cuda_cublas_gemm_algo_index_or_default(
    int default_index);
int bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled(void);
int bn_gpu_policy_cuda_q8_0_quant_matmul_enabled(void);
int bn_gpu_policy_cuda_f16_q8_0_matmul_enabled(void);
int bn_gpu_policy_cuda_q8_0_preq_split_enabled(void);
int bn_gpu_policy_cuda_q8_preq_all_enabled(void);
int bn_gpu_policy_cuda_q8_preq_logits_disabled(void);
int bn_gpu_policy_cuda_q8_preq_logits_default_enabled(
    int preq_logits_disabled);
int bn_gpu_policy_cuda_q8k_input_cache_enabled(void);
int bn_gpu_policy_cuda_force_quant_matmul_for_type(
    int tensor_type,
    int f16_q8_0_matmul_enabled);
int bn_gpu_policy_cuda_q6k_4warp_long_enabled(int rows, int cols);
int bn_gpu_policy_cuda_q6k_5warp_exact_enabled(int rows, int cols);
int bn_gpu_policy_cuda_q6k_3warp_exact_enabled(int rows, int cols);
int bn_gpu_policy_cuda_q6k_2warp_long_enabled(int rows, int cols);
int bn_gpu_policy_cuda_q6k_matvec4_shape_disabled(int rows, int cols);
int bn_gpu_policy_cuda_q6k_moe_quant_down_preferred(int routed_q4,
                                                    int down_type,
                                                    int hidden_dim,
                                                    int n_experts,
                                                    int k);
int bn_gpu_policy_cuda_q6k_moe_down_f32_cache_path_enabled(
    int routed_q4,
    int down_type,
    int has_f32_data,
    int prefer_quant_down,
    int dim,
    int hidden_dim,
    int n_experts,
    int k);
int bn_gpu_policy_cuda_moe_down_4row_enabled(int hidden_dim);
int bn_gpu_policy_cuda_moe_down_8row_enabled(int hidden_dim);
int bn_gpu_policy_cuda_q6k_moe_down_halfwarp_enabled(
    int down_type,
    int prefer_quant_down,
    int n_experts,
    int k);
int bn_gpu_policy_cuda_q6k_moe_down_split4_enabled(
    int down_type,
    int use_halfwarp,
    int n_experts,
    int k);
int bn_gpu_policy_cuda_q6k_moe_down_scatter_enabled(
    int down_type,
    int use_halfwarp,
    int use_split4);
int bn_gpu_policy_cuda_q6k_moe_down_scatter_16row_enabled(
    int use_scatter,
    int hidden_dim);
int bn_gpu_policy_cuda_q6k_moe_float_down_enabled(void);
int bn_gpu_policy_cuda_q6k_moe_pair_down_enabled(
    int f32_down_default,
    int pair_down_f32_layer,
    int all2_disable_pair_down);
int bn_gpu_policy_cuda_q6k_moe_prefer_f32_down(
    int has_f32_data,
    int hidden_dim,
    int all2_q4q6,
    int all2_f32_down);
int bn_gpu_policy_cuda_q6k_moe_down_f32_pair2_enabled(int n_experts,
                                                      int k);
int bn_gpu_policy_cuda_q6k_moe_down_f32_pair2_4row_enabled(void);
int bn_gpu_policy_cuda_q6k_moe_down_q8k_all2_accum_enabled(int all2_q4q6);
int bn_gpu_policy_cuda_q6k_moe_down_q8k_pair4_sum_enabled(int all2_q4q6);
int bn_gpu_policy_cuda_q6k_moe_down_q8k_k8_4row_sum_enabled(
    int all2_q4q6,
    int k,
    int hidden_dim);
int bn_gpu_policy_cuda_q6k_moe_down_q8k_k8_8row_sum_enabled(
    int k8_4row_sum,
    int hidden_dim);
int bn_gpu_policy_cuda_q6k_moe_down_q8k_all2_fixed_enabled(int all2_q4q6);
int bn_gpu_policy_cuda_q6k_moe_down_resid_rmsnorm_fuse_enabled(void);
int bn_gpu_policy_cuda_q6k_moe_down_q8k_k8_exact_2048_768_enabled(
    int dim,
    int hidden_dim,
    int k);
int bn_gpu_policy_cuda_q6k_moe_down_q8k_all2_accum_4row_enabled(void);
int bn_gpu_policy_cuda_q6k_moe_down_q8k_pair_4row_enabled(void);
int bn_gpu_policy_cuda_q6k_moe_down_f32_cache_enabled(
    int has_f32_data,
    int all2_disable_f32_cache);
int bn_gpu_policy_cuda_q6k_moe_down_f16_cache_enabled(int has_f16_data);
int bn_gpu_policy_cuda_q4k_moe_down_f32_cache_enabled(int has_f32_data);
int bn_gpu_policy_cuda_q4k_moe_pair_down_enabled(int n_experts,
                                                 int k,
                                                 int hidden_dim);
int bn_gpu_policy_cuda_q4k_moe_down_8row_enabled(int hidden_dim);
int bn_gpu_policy_cuda_q4k_q8k_moe_gateup_enabled(int n_tokens,
                                                  int dim,
                                                  int allow_small_dim);
int bn_gpu_policy_cuda_q4k_moe_gateup_8row_enabled(int dim);
int bn_gpu_policy_cuda_q4k_moe_gateup_split_enabled(int dim,
                                                    int n_experts);
int bn_gpu_policy_cuda_moe_route_q8k_prequant_enabled(int dim,
                                                      int all2_q4q6);
int bn_gpu_policy_cuda_moe_route_q8_1_prequant_enabled(int dim,
                                                       int all2_q4q6,
                                                       int exact_silu);
int bn_gpu_policy_cuda_moe_router_fused_topk_enabled(int n_experts,
                                                     int route_block);
int bn_gpu_policy_cuda_moe_router_warp_disabled(int route_block);
int bn_gpu_policy_cuda_moe_router_4warp_enabled(int dim);
int bn_gpu_policy_cuda_moe_router_2warp_enabled(int dim);
int bn_gpu_policy_cuda_moe_router_warp_topk_enabled(int n_experts);
int bn_gpu_policy_cuda_q8_moe_q8_1_batch_enabled(int routed_q8);
int bn_gpu_policy_cuda_q8_moe_q8x_enabled(void);
int bn_gpu_policy_cuda_q8_moe_gateup_2row_enabled(int hidden_dim);
int bn_gpu_policy_cuda_q8_moe_down_4row_enabled(int hidden_dim);
int bn_gpu_policy_cuda_q8_moe_down_2row_enabled(int hidden_dim);
int bn_gpu_policy_cuda_moe_all2_fast_enabled(int all2_q4_or_q6);
int bn_gpu_policy_cuda_moe_q4k_q8k_dot_enabled(int use_all2_q8k_default,
                                               int fast_q8k_gateup,
                                               int all2_q4q6,
                                               int hidden_dim,
                                               int dim);
int bn_gpu_policy_cuda_moe_internal_profile_enabled(int profile);
int bn_gpu_policy_cuda_moe_q4k_all2_fixed_4row_enabled(
    int prequantized_q8k,
    int all2_fast_enabled);
int bn_gpu_policy_cuda_moe_q4k_gateup_4row_disabled(void);
int bn_gpu_policy_cuda_decode_logits_cache_enabled(int gpu_logits_need_cpu);
int bn_gpu_policy_decode_logits_cache_enabled(int gpu_logits_need_cpu);
int bn_gpu_policy_cuda_moe_decode_cache_enabled(void);
int bn_gpu_policy_moe_decode_cache_enabled(void);
int bn_gpu_policy_cuda_moe_decode_cache_disabled(void);
int bn_gpu_policy_moe_decode_cache_disabled(void);
int bn_gpu_policy_cuda_decode_cache_disabled(void);
int bn_gpu_policy_decode_cache_disabled(void);
int bn_gpu_policy_cuda_q4_q8_decode_cache_disabled(void);
int bn_gpu_policy_q4_q8_decode_cache_disabled(void);
int bn_gpu_policy_cuda_logits_argmax_disabled(void);
int bn_gpu_policy_logits_argmax_disabled(void);
int bn_gpu_policy_cuda_dense_logits_argmax_enabled(void);
int bn_gpu_policy_dense_logits_argmax_enabled(void);
int bn_gpu_policy_cuda_moe_logits_mmvq_argmax_enabled(void);
int bn_gpu_policy_moe_logits_mmvq_argmax_enabled(void);
int bn_gpu_policy_cuda_moe_logits_mmvq_argmax_disabled(void);
int bn_gpu_policy_moe_logits_mmvq_argmax_disabled(void);
int bn_gpu_policy_cuda_moe_logits_mmvq_argmax_path_enabled(int rows,
                                                           int cols);
int bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_enabled(int use_mmvq,
                                                           int rows,
                                                           int cols);
int bn_gpu_policy_cuda_moe_logits_mmvq_1warp16_1536_enabled(
    int use_1warp8);
int bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_unroll_enabled(
    int use_1warp8,
    int use_1warp16);
int bn_gpu_policy_cuda_argmax_fast_enabled(void);
int bn_gpu_policy_cuda_optimistic_argmax_penalty_enabled(void);
int bn_gpu_policy_cuda_q5_matvec4_enabled(void);
int bn_gpu_policy_cuda_q5_warp_enabled(void);
int bn_gpu_policy_cuda_q5k_deint_pair_matvec_enabled(void);
int bn_gpu_policy_cuda_q5k_4warp_enabled(int cols);
int bn_gpu_policy_cuda_q5k_split_4warp_enabled(int cols);
int bn_gpu_policy_cuda_q5k_gateup_2warp_enabled(void);
int bn_gpu_policy_cuda_q4k_dot_enabled(void);
int bn_gpu_policy_cuda_q5k_dot_enabled(void);
int bn_gpu_policy_cuda_q4k_4warp_enabled(void);
int bn_gpu_policy_cuda_q4k_4warp_shape_enabled(int rows, int cols);
int bn_gpu_policy_cuda_q4k_out_residual_rmsnorm_fuse_enabled(void);
int bn_gpu_policy_cuda_q4k_qkv_mixed_fuse_enabled(int tensor_type);
int bn_gpu_policy_cuda_q4k_split_k_rope_cache_fuse_enabled(void);
int bn_gpu_policy_cuda_q4k_split_qk_rope_cache_fuse_enabled(void);
int bn_gpu_policy_cuda_q4k_split_4warp_enabled(int cols);
int bn_gpu_policy_cuda_q4k_split_5warp_enabled(int cols);
int bn_gpu_policy_cuda_q4k_split_value_rows(int total_rows, int cols);
int bn_gpu_policy_cuda_q4k_split_value_fuse_enabled(int value_rows);
int bn_gpu_policy_cuda_q4k_gateup_q8k_path_enabled(int q8k_flag);
int bn_gpu_policy_cuda_q4k_gateup_qwarp4_enabled(int cols);
int bn_gpu_policy_cuda_q4k_gateup_5warp_enabled(int enable_q4k_4warp,
                                                int cols);
int bn_gpu_policy_cuda_q4k_gateup_2warp_enabled(int enable_q4k_4warp,
                                                int cols);
int bn_gpu_policy_cuda_q4k_gateup_4warp_enabled(int enable_q4k_4warp,
                                                int cols);
int bn_gpu_policy_cuda_q8_warp_disabled(void);
int bn_gpu_policy_cuda_q8_0_ssm_matvec_enabled(void);
int bn_gpu_policy_cuda_q8_0_ssm_preq_enabled(void);
int bn_gpu_policy_cuda_q8_mixed_preq_enabled(int type_a,
                                             int type_b,
                                             int cols);
int bn_gpu_policy_cuda_f16_q8_0_ssm_matvec_enabled(void);
int bn_gpu_policy_cuda_f16_q8_0_matvec_enabled(void);
int bn_gpu_policy_cuda_f16_q5k_matvec_enabled(void);
int bn_gpu_policy_cuda_q4k_pair_matvec_enabled(void);
int bn_gpu_policy_cuda_q4k_q8k_dot_enabled(void);
int bn_gpu_policy_cuda_q4k_q8k_dot_forced(void);
int bn_gpu_policy_cuda_q4k_q8k_matvec4_enabled(int cols);
int bn_gpu_policy_cuda_q4k_matmul8_enabled(void);
int bn_gpu_policy_cuda_q4k_sharedx_enabled(void);
int bn_gpu_policy_cuda_q4k_batch_sharedx_enabled(void);
int bn_gpu_policy_cuda_q6k_dot_enabled(void);
int bn_gpu_policy_cuda_q6k_dot_forced(void);
int bn_gpu_policy_cuda_q6k_warp_enabled(void);
int bn_gpu_policy_cuda_q6k_q4k_pair_matvec_enabled(int cols);
int bn_gpu_policy_cuda_q6k_q8_1_dot_enabled(int is_logits_op);
int bn_gpu_policy_cuda_q6k_mmvq_enabled(int rows,
                                        int cols,
                                        int is_logits_op,
                                        int exact_q6k);
int bn_gpu_policy_cuda_q6k_mmvq_2warp_logits_enabled(int rows,
                                                     int cols,
                                                     int is_logits_op);
int bn_gpu_policy_cuda_q6k_down_residual_rmsnorm_fuse_enabled(void);
int bn_gpu_policy_cuda_f16_q6k_matvec_enabled(int rows,
                                              int cols,
                                              int exact_q6k);
int bn_gpu_policy_cuda_q6k_matmul8_enabled(void);
int bn_gpu_policy_cuda_q6k_matmul4_enabled(void);
int bn_gpu_policy_cuda_q6k_matvec4_enabled(void);
int bn_gpu_policy_cuda_q6k_batch_warp_enabled(void);
int bn_gpu_policy_cuda_fuse_bias_enabled(void);
int bn_gpu_policy_cuda_rope_flash_fuse_enabled(void);
int bn_gpu_policy_cuda_bias_rope_flash_fuse_enabled(void);
int bn_gpu_policy_cuda_qk_norm_rope_flash_fuse_enabled(void);
int bn_gpu_policy_cuda_qk_norm_rope_fuse_enabled(void);
int bn_gpu_policy_cuda_weighted_add_sigmoid_residual_rmsnorm_fuse_enabled(void);
int bn_gpu_policy_cuda_weighted_add_sigmoid_residual_fuse_enabled(void);
int bn_gpu_policy_cuda_readback_debug_enabled(void);
int bn_gpu_policy_cuda_cublas_cache_debug_enabled(void);
int bn_gpu_policy_cuda_nan_verbose_debug_enabled(void);
int bn_gpu_policy_cuda_stream_exec_enabled(void);
int bn_gpu_policy_cuda_profile_enabled(void);
int bn_gpu_policy_cuda_wall_profile_enabled(void);
int bn_gpu_policy_cuda_profile_shapes_enabled(void);
const char *bn_gpu_policy_cuda_device_selector(void);
int bn_gpu_policy_cuda_exec_fail_debug_enabled(void);
int bn_gpu_policy_cuda_sync_each_op_debug_enabled(void);
int bn_gpu_policy_cuda_nan_debug_enabled(void);
int bn_gpu_policy_cuda_dump_ops_enabled(void);
int bn_gpu_policy_cuda_dump_ops_every_enabled(void);
int bn_gpu_policy_cuda_prefill_moe_layer_disabled(void);
int bn_gpu_policy_cuda_prefill_dense_layer_disabled(void);
int bn_gpu_policy_cuda_prefill_dense_debug_enabled(void);
int bn_gpu_policy_cuda_prefill_dense_profile_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_layer_disabled(void);
int bn_gpu_policy_prefill_ssm_layer_disabled(void);
int bn_gpu_policy_cuda_prefill_fused_q4k_gateup_batch_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_fused_q4k_gateup_batch_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_profile_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_stacked_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_stream_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_input_alias_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_f32_ab_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_scan_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_delta_128_warp_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_ffn_profile_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_enabled(void);
int bn_gpu_policy_explicit_q5k_fused_gateup_enabled(void);
int bn_gpu_policy_fused_gateup_silu_allowed(const BnGPUBackend *gpu,
                                            int tensor_type);
int bn_gpu_policy_cuda_shared_q4_q8_dot_enabled(void);
int bn_gpu_policy_shared_q4_q8_dot_enabled(void);
int bn_gpu_policy_cuda_shared_expert_gate_enabled(void);
int bn_gpu_policy_shared_expert_gate_enabled(void);
int bn_gpu_policy_cuda_prefill_attention_min_tokens_configured(void);
int bn_gpu_policy_prefill_attention_min_tokens_configured(void);
int bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(
    int default_tokens);
int bn_gpu_policy_prefill_attention_min_tokens_or_default(
    int default_tokens);
int bn_gpu_policy_cuda_prefill_gemm_attention_min_tokens_or_default(
    int default_tokens);
int bn_gpu_policy_cuda_prefill_gemm_attention_enabled(int n_tokens,
                                                      int max_tokens);
int bn_gpu_policy_cuda_prefill_attention_wo_enabled(void);
int bn_gpu_policy_cuda_prefill_qkv_attention_wo_enabled(void);
int bn_gpu_policy_cuda_prefill_batched_gemm_enabled(void);
int bn_gpu_policy_cuda_prefill_gemm_debug_enabled(void);
int bn_gpu_policy_cuda_prefill_dense_chain_enabled(void);
int bn_gpu_policy_prefill_dense_chain_enabled(void);
int bn_gpu_policy_cuda_prefill_hybrid_chain_enabled(void);
int bn_gpu_policy_prefill_hybrid_chain_enabled(void);
int bn_gpu_policy_cuda_prefill_attention_enabled(void);
int bn_gpu_policy_prefill_attention_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_run_chain_enabled(void);
int bn_gpu_policy_prefill_ssm_run_chain_enabled(void);
int bn_gpu_policy_cuda_prefill_ssm_ffn_fuse_allowed(void);
int bn_gpu_policy_prefill_ssm_ffn_fuse_allowed(void);
int bn_gpu_policy_cuda_prefill_moe_chain_debug_enabled(void);
int bn_gpu_policy_prefill_moe_chain_debug_enabled(void);
int bn_gpu_policy_cuda_prefill_hybrid_chain_debug_enabled(void);
int bn_gpu_policy_prefill_hybrid_chain_debug_enabled(void);
int bn_gpu_policy_cuda_moe_prefill_enabled(void);
int bn_gpu_policy_moe_prefill_enabled(void);
int bn_gpu_policy_cuda_moe_prefill_min_tokens_configured(void);
int bn_gpu_policy_moe_prefill_min_tokens_configured(void);
int bn_gpu_policy_cuda_moe_prefill_min_tokens_or_default(
    int default_tokens);
int bn_gpu_policy_moe_prefill_min_tokens_or_default(int default_tokens);
int bn_gpu_policy_cuda_dense_ffn_enabled(void);
int bn_gpu_policy_cuda_dense_ffn_batch_enabled(void);
int bn_gpu_policy_cuda_moe_cublas_gateup_f16_out_enabled(void);
int bn_gpu_policy_cuda_moe_cublas_grouped_variable_enabled(void);
int bn_gpu_policy_cuda_moe_cublas_grouped_enabled(int routed_q8,
                                                  int routed_q4,
                                                  int gate_f16,
                                                  int up_f16,
                                                  int down_f16,
                                                  int n_experts,
                                                  int k,
                                                  int route_items);
int bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(int use_grouped,
                                                      int routed_q8,
                                                      int routed_q4,
                                                      int gate_f16,
                                                      int up_f16,
                                                      int down_f16,
                                                      int n_tokens);
int bn_gpu_policy_cuda_moe_cublas_all2_fixed_enabled(int use_grouped,
                                                     int n_experts,
                                                     int k);
int bn_gpu_policy_cuda_moe_sorted_slots_enabled(int routed_q4,
                                                int routed_q8,
                                                int n_tokens,
                                                int use_all2_fixed,
                                                int use_grouped,
                                                int use_gateup_only);
int bn_gpu_policy_cuda_moe_prefill_internal_profile_enabled(void);
int bn_gpu_policy_cuda_moe_prefill_direct_resid_out_enabled(
    int add_norm_resid,
    int out_provided,
    int has_shared,
    int init_out_with_residual);
int bn_gpu_policy_cuda_moe_batch_fused_route_topk_enabled(int n_experts);
int bn_gpu_policy_cuda_moe_route_dist_profile_enabled(void);
int bn_gpu_policy_cuda_moe_route_dist_profile_every_or_default(
    int default_every);
int bn_gpu_policy_cuda_moe_cublas_grouped_debug_enabled(void);
int bn_gpu_policy_cuda_moe_cublas_gateup_debug_enabled(void);
int bn_gpu_policy_cuda_moe_ffn_batch_enabled(void);
int bn_gpu_policy_cuda_moe_ffn_batch_profile_enabled(void);
int bn_gpu_policy_cuda_moe_cache_prefill_enabled(void);
int bn_gpu_policy_moe_cache_prefill_enabled(void);
int bn_gpu_policy_cuda_moe_prefill_shared_fuse_enabled(void);
int bn_gpu_policy_moe_prefill_shared_fuse_enabled(void);
int bn_gpu_policy_cuda_moe_route_batch_enabled(void);
int bn_gpu_policy_moe_route_batch_enabled(void);
int bn_gpu_policy_cuda_moe_route_batch_debug_enabled(void);
int bn_gpu_policy_moe_route_batch_debug_enabled(void);
int bn_gpu_policy_cuda_large_hybrid_attention_enabled(void);
int bn_gpu_policy_large_hybrid_attention_enabled(void);
int bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_enabled(void);
int bn_gpu_policy_large_hybrid_cpu_attention_safe_enabled(void);
int bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_disabled(void);
int bn_gpu_policy_large_hybrid_cpu_attention_safe_disabled(void);
int bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe_forced(void);
int bn_gpu_policy_large_hybrid_cpu_attention_safe_forced(void);
int bn_gpu_policy_cuda_large_hybrid_prefill_enabled(void);
int bn_gpu_policy_large_hybrid_prefill_enabled(void);
int bn_gpu_policy_cuda_large_hybrid_prefill_chain_enabled(void);
int bn_gpu_policy_large_hybrid_prefill_chain_enabled(void);
int bn_gpu_policy_cuda_large_hybrid_prefill_disabled(void);
int bn_gpu_policy_large_hybrid_prefill_disabled(void);
int bn_gpu_policy_cuda_large_hybrid_argmax_enabled(void);
int bn_gpu_policy_large_hybrid_argmax_enabled(void);
int bn_gpu_policy_prefill_matmul_disabled(void);
int bn_gpu_policy_prefill_matmul_enabled(void);
int bn_gpu_policy_cuda_prefill_direct_kv_disabled(void);
int bn_gpu_policy_prefill_direct_kv_disabled(void);
int bn_gpu_policy_cuda_prefill_direct_kv_with_cpu_fallback_enabled(void);
int bn_gpu_policy_prefill_direct_kv_with_cpu_fallback_enabled(void);
int bn_gpu_policy_cpu_decode_fallback_requested(void);
int bn_gpu_policy_cpu_fallback_layer_or_default(int default_layer);
int bn_gpu_policy_cpu_fallback_from_layer_or_default(int default_layer);
int bn_gpu_policy_cpu_attention_layer_or_default(int default_layer);
int bn_gpu_policy_cpu_attention_from_layer_or_default(int default_layer);
int bn_gpu_policy_cpu_ffn_layer_or_default(int default_layer);
int bn_gpu_policy_cpu_ffn_from_layer_or_default(int default_layer);
int bn_gpu_policy_cpu_ffn_down_from_layer_or_default(int default_layer);
int bn_gpu_policy_ssm_graph_disabled(void);
int bn_gpu_policy_cuda_ssm_graph_disabled(void);
int bn_gpu_policy_cuda_qkv_mixed_fuse_disabled(void);
int bn_gpu_policy_cuda_qkv_key_cache_fuse_enabled(void);
int bn_gpu_policy_cuda_qkv_kpair_opt_enabled(void);
int bn_gpu_policy_cuda_q5_gateup_warp_disabled(void);
int bn_gpu_policy_cuda_q8_gateup_warp_disabled(void);
int bn_gpu_policy_cuda_graph_exec_requested(void);
int bn_gpu_policy_cuda_moe_graph_max_experts_or_default(
    int default_experts);
int bn_gpu_policy_cuda_decode_graph_default_enabled(int moe_graph,
                                                    int default_moe_graph);
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
size_t bn_gpu_policy_moe_down_aux_cache_bytes(const BnGPUBackend *gpu,
                                              int tensor_type,
                                              int rows,
                                              int cols);
size_t bn_gpu_policy_cuda_aux_cache_bytes(int tensor_type,
                                          int rows,
                                          int cols);
size_t bn_gpu_policy_aux_cache_bytes(int tensor_type,
                                     int rows,
                                     int cols);
int bn_gpu_policy_cuda_cublas_aux_cache_supported(int tensor_type,
                                                  int cols);
int bn_gpu_policy_moe_auto_resident_enabled(void);
size_t bn_gpu_policy_moe_cache_reserve_bytes(void);
int bn_gpu_policy_auto_caps_sequence(int webgpu,
                                     int cuda,
                                     int metal,
                                     int has_moe,
                                     int model_seq_len,
                                     int cap_seq_len);
int bn_gpu_policy_auto_caps_gguf_sequence(int webgpu,
                                          int cuda,
                                          int metal,
                                          BnGGUFFile *gf,
                                          int cap_seq_len);
int bn_gpu_policy_cuda_duplicate_moe_cache_enabled(void);
int bn_gpu_policy_duplicate_moe_cache_enabled(void);
int bn_gpu_policy_webgpu_repacked_buffer_supported(int tensor_type);
int bn_gpu_policy_webgpu_repacked_bias_supported(int tensor_type);
int bn_gpu_policy_metal_mmap_zero_copy_enabled(void);
void bn_gpu_policy_apply_metal_barrier_disable_override(void);
void bn_gpu_policy_apply_specialized_q6_q8k_override(void);
void bn_gpu_policy_apply_q4_q8_prepared_override(void);
void bn_gpu_policy_apply_metal_q4_q8_default_disable_override(void);
void bn_gpu_policy_apply_metal_private_weights_override(void);
void bn_gpu_policy_metal_apply_q4_q8_default(void);
int bn_gpu_policy_metal_q4_q8_enabled(void);
int bn_gpu_policy_metal_q4_prepared_enabled(void);
int bn_gpu_policy_q4_q8_prepared_layer_default_enabled(void);
int bn_gpu_policy_metal_q4_prepared_upload_enabled(void);
int bn_gpu_policy_metal_repacked_buffer_supported(int tensor_type);
int bn_gpu_policy_metal_repacked_buffer_type(int tensor_type);
int bn_gpu_policy_metal_prepared_stacked_upload_blocked(int tensor_type);
int bn_gpu_policy_metal_shared_weights_enabled(void);
int bn_gpu_policy_metal_q6_q8k_enabled(void);
int bn_gpu_policy_specialized_q6_q8k_path_enabled(void);
int bn_gpu_policy_metal_q8_barriers_enabled(void);
int bn_gpu_policy_metal_q4_q8_matvec_supported(int tensor_type,
                                               int q4_q8_enabled,
                                               int q4_prepared,
                                               int has_q8_quant_pipeline,
                                               int has_q4_q8_pipeline,
                                               int has_q4_prepared_q8_pipeline);
int bn_gpu_policy_metal_q4_q8_graph_path_supported(
    int tensor_type,
    int q4_q8_enabled,
    int q4_prepared,
    int prepared_path,
    int has_q8_quant_pipeline,
    int has_pipeline);
int bn_gpu_policy_metal_q6_q8k_matvec_supported(int tensor_type,
                                                int cols,
                                                int has_q8k_quant_pipeline,
                                                int has_q6_q8k_pipeline);
int bn_gpu_policy_metal_cpu_order_rmsnorm_enabled(void);
int bn_gpu_policy_metal_full_barriers_enabled(void);
int bn_gpu_policy_metal_barriers_enabled(void);
int bn_gpu_policy_metal_barriers_disabled(void);
int bn_gpu_policy_fused_gateup_enabled(void);
int bn_gpu_policy_q4_q8_fused_gateup_enabled(void);
int bn_gpu_policy_q4_q8_attn_only_enabled(void);
int bn_gpu_policy_q4_q8_ffn_only_enabled(void);
int bn_gpu_policy_q4_q8_from_layer_or_default(int n_layers);
int bn_gpu_policy_q4_q8_to_layer_or_default(int n_layers,
                                            int metal_q4_prepared);
int bn_gpu_policy_gateup_split_enabled(void);
int bn_gpu_policy_q4_q8_ffn_down_enabled(void);
int bn_gpu_policy_qkv_split_enabled(void);
int bn_gpu_policy_qkv_split_debug_enabled(void);
int bn_gpu_policy_ssm_qkvz_split_enabled(void);
int bn_gpu_policy_ssm_ab_stack_enabled(void);
int bn_gpu_policy_split_residual_rmsnorm_enabled(void);
int bn_gpu_policy_debug_fallback_enabled(void);
int bn_gpu_policy_force_graph_enabled(void);
int bn_gpu_policy_flash_min_kv_or_default(int default_min_kv);
int bn_gpu_policy_backend_flash_max_kv_or_default(const BnGPUBackend *gpu,
                                                  int default_max_kv);
int bn_gpu_policy_backend_flash_default_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_backend_large_graph_native_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_small_dense_native_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_all2_q4q6_moe_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_cpu_attention_fallback_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_small_dense_exact_q4_q8_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_prefill_decode_fallback_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_prefill_chain_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_matvec_fallback_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_dense_batch_prefill_shape_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_lazy_moe_aux_cache_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_small_dense_q8_logits_refine_default_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_all2_q4q6_moe_q6_logits_refine_default_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_decode_graph_cache_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_moe_exact_attention_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_ssm_graph_supported(const BnGPUBackend *gpu);
int bn_gpu_policy_backend_large_hybrid_argmax_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_all2_moe_direct_route_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_resident_moe_ffn_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_moe_gateup_split_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_flash_max_kv_or_default(int cuda_backend,
                                          int default_max_kv);
int bn_gpu_policy_argmax_debug_enabled(void);
int bn_gpu_policy_cpu_logits_enabled(void);
int bn_gpu_policy_compare_logits_enabled(void);
int bn_gpu_policy_debug_argmax_compare_enabled(void);
int bn_gpu_policy_backend_q6_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int q6_refine_default);
int bn_gpu_policy_q6_logits_refine_enabled(int cuda_backend,
                                           int q6_refine_default);
int bn_gpu_policy_q6_logits_refine_top_or_default(int default_top);
int bn_gpu_policy_backend_q8_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int q8_refine_default);
int bn_gpu_policy_q8_logits_refine_enabled(int cuda_backend,
                                           int q8_refine_default);
int bn_gpu_policy_q8_logits_refine_top_or_default(int default_top);
int bn_gpu_policy_moe_ffn_disabled(void);
int bn_gpu_policy_moe_router_topk_enabled(int eligible);
int bn_gpu_policy_q8_moe_cpu_route_resident_enabled(int eligible);
int bn_gpu_policy_moe_router_gpu_enabled(void);
int bn_gpu_policy_moe_router_diff2_enabled(void);
int bn_gpu_policy_moe_routed_ffn_batch_enabled(void);
int bn_gpu_policy_moe_routed_ffn_batch_allowed(int large_moe);
int bn_gpu_policy_moe_cpu_actual_override_enabled(void);
int bn_gpu_policy_cuda_moe_ffn_disabled(void);
int bn_gpu_policy_cuda_moe_router_topk_enabled(int eligible);
int bn_gpu_policy_cuda_q8_moe_cpu_route_resident_enabled(int eligible);
int bn_gpu_policy_cuda_moe_router_gpu_enabled(void);
int bn_gpu_policy_cuda_moe_router_diff2_enabled(void);
int bn_gpu_policy_cuda_moe_routed_ffn_batch_enabled(void);
int bn_gpu_policy_cuda_moe_routed_ffn_batch_allowed(int large_moe);
int bn_gpu_policy_cuda_moe_cpu_actual_override_enabled(void);
int bn_gpu_policy_small_dense_q8_cpu_attention_safe_disabled(void);
int bn_gpu_policy_small_dense_exact_q4_q8_disabled(void);
int bn_gpu_policy_small_dense_exact_ffn_down_enabled(void);
int bn_gpu_policy_small_dense_prefill_disabled(void);
int bn_gpu_policy_small_dense_q8_logits_refine_enabled(void);
int bn_gpu_policy_small_dense_q8_logits_refine_disabled(void);
int bn_gpu_policy_all2_q4q6_moe_fast_ffn_enabled(void);
int bn_gpu_policy_all2_q4q6_moe_fast_graph_disabled(void);
int bn_gpu_policy_all2_q4q6_moe_cublas_decode_enabled(void);
int bn_gpu_policy_cuda_moe_cublas_decode_enabled(void);
int bn_gpu_policy_cuda_moe_cublas_decode_debug_enabled(void);
int bn_gpu_policy_all2_q4q6_moe_all2_fast_enabled(void);
int bn_gpu_policy_all2_q4q6_moe_q8k_default_disabled(void);
int bn_gpu_policy_all2_q4q6_route_q8k_default_disabled(void);
int bn_gpu_policy_all2_q4q6_route_q8_1_prequant_enabled(void);
int bn_gpu_policy_all2_q4q6_fast_q8k_gateup_enabled(void);
int bn_gpu_policy_all2_q4q6_fast_q8k_gateup_disabled(void);
int bn_gpu_policy_all2_q4q6_q6k_pair_down_enabled(void);
int bn_gpu_policy_all2_q4q6_q6k_pair_down_f32_layers_disabled(void);
int bn_gpu_policy_all2_q4q6_q6k_pair_down_f32_layer_selected(int layer);
int bn_gpu_policy_all2_q4q6_q6k_ordered_down_enabled(void);
int bn_gpu_policy_all2_q4q6_q6k_ordered_down_disabled(void);
int bn_gpu_policy_all2_q4q6_q6k_f32_down_default_enabled(void);
int bn_gpu_policy_all2_q4q6_q6k_f32_down_default_disabled(void);
int bn_gpu_policy_all2_q4q6_q6k_f32_all2_down_disabled(void);
int bn_gpu_policy_all2_q4q6_q6k_f32_cache_enabled(void);
int bn_gpu_policy_all2_q4q6_q6k_float_4row_down_default_disabled(void);
int bn_gpu_policy_all2_q4q6_q6k_float_4row_down_disabled(void);
int bn_gpu_policy_all2_q4q6_q6k_f32_exact_4row_down_layer_selected(
    int layer);
int bn_gpu_policy_all2_q4q6_q6k_f32_exact_4row_down_default_disabled(void);
int bn_gpu_policy_all2_q4q6_q6k_f32_exact_4row_down_disabled(void);
float bn_gpu_policy_all2_q4q6_down_skip_eps_or_default(float default_eps);
int bn_gpu_policy_all2_q4q6_moe_cpu_attention_safe_disabled(void);
int bn_gpu_policy_all2_q4q6_moe_q6_logits_refine_disabled(void);
int bn_gpu_policy_all2_q4q6_moe_cpu_moe_safe_disabled(void);
int bn_gpu_policy_all2_q4q6_moe_exact_attention_disabled(void);
int bn_gpu_policy_all2_q4q6_moe_cpu_route_resident_disabled(void);
int bn_gpu_policy_all2_q4q6_moe_exact_gpu_route_requested(void);
int bn_gpu_policy_all2_q4q6_moe_exact_gpu_route_disabled(void);
int bn_gpu_policy_all2_q4q6_moe_route_selection_enabled(void);
void bn_gpu_policy_all2_q4q6_moe_route_layer_range(int *from_layer,
                                                   int *to_layer);
int bn_gpu_policy_moe_compare_layer_selected(int layer, int pos);
int bn_gpu_policy_moe_compare_input_norm_enabled(void);
int bn_gpu_policy_moe_compare_actual_enabled(void);
int bn_gpu_policy_moe_compare_route_enabled(void);
int bn_gpu_policy_moe_compare_raw_enabled(void);
int bn_gpu_policy_moe_compare_mid_enabled(void);
int bn_gpu_policy_moe_compare_parts_enabled(void);
int bn_gpu_policy_moe_compare_shared_mid_enabled(void);
int bn_gpu_policy_moe_compare_shared_down_enabled(void);
int bn_gpu_policy_moe_compare_norm_enabled(void);
int bn_gpu_policy_compare_attention_layer_or_default(int default_layer);
int bn_gpu_policy_compare_attention_pos_or_default(int default_pos);
int bn_gpu_policy_compare_gqa_layer_or_default(int default_layer);
int bn_gpu_policy_compare_gqa_pos_or_default(int default_pos);
int bn_gpu_policy_compare_qkv_layer_or_default(int default_layer);
int bn_gpu_policy_compare_qkv_pos_or_default(int default_pos);
int bn_gpu_policy_compare_ffn_down_layer_or_default(int default_layer);
int bn_gpu_policy_compare_ffn_down_pos_or_default(int default_pos);
int bn_gpu_policy_compare_ffn_state_layer_or_default(int default_layer);
int bn_gpu_policy_compare_ffn_state_pos_or_default(int default_pos);
int bn_gpu_policy_moe_shared_cpu_fallback_enabled(int eligible);
int bn_gpu_policy_moe_gateup_split_enabled(int can_split);
int bn_gpu_policy_cuda_moe_shared_cpu_fallback_enabled(int eligible);
int bn_gpu_policy_cuda_moe_gateup_split_enabled(int can_split);
int bn_gpu_policy_moe_route_profile_enabled(void);
int bn_gpu_policy_moe_route_profile_every_or_default(int default_every);
int bn_gpu_policy_profile_level(void);

#ifdef __cplusplus
}
#endif

#endif // BN_GPU_POLICY_H
