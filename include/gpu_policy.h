#ifndef BN_GPU_POLICY_H
#define BN_GPU_POLICY_H

#include "backend_placement.h"
#include "gpu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int total_experts;
    int active_experts;
    int expert_hidden_dim;
} BnGPUMoERouteShape;

int bn_gpu_policy_moe_resident_routed_ffn_enabled(
    const BnGPUBackend *gpu, int eligible);
BnBackendPlacement bn_gpu_policy_backend_placement(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_resident_routed_ffn_quant_eligible(
    int gate_type,
    int up_type,
    int down_type);
int bn_gpu_policy_backend_moe_resident_routed_ffn_eligible(
    const BnGPUBackend *gpu,
    int standard_quant_eligible,
    int metal_quant_eligible,
    int layout_eligible);
int bn_gpu_policy_moe_all_f16_cache_forced(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_all_f16_cache_enabled_for_type(
    const BnGPUBackend *gpu,
    int tensor_type,
    int native_quant_f16_cache);
int bn_gpu_policy_moe_gateup_f16_cache_enabled(const BnGPUBackend *gpu,
                                                int eligible);
int bn_gpu_policy_partial_moe_f16_cache_enabled(const BnGPUBackend *gpu,
                                                 int eligible);
int bn_gpu_policy_moe_residency_fit_debug_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_lazy_aux_cache_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_individual_upload_quant_only_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_logits_kquant_f32_cache_enabled(const BnGPUBackend *gpu,
                                                  int tensor_type);
int bn_gpu_policy_logits_f16_cache_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_cublas_logits_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_f32_logits_matvec_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_f16_logits_matvec_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_down_kquant_f32_cache_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_moe_down_kquant_f32_cache_forced(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_down_kquant_f32_cache_default_for_cols(const BnBackendRuntimePolicy *policy, int cols);
int bn_gpu_policy_moe_down_kquant_f32_cache_preferred(
    const BnGPUBackend *gpu,
    int tensor_type,
    int cols,
    int force_f16_cache);
size_t bn_gpu_policy_moe_down_kquant_f32_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols,
    int n_experts);
int bn_gpu_policy_moe_down_kquant_f32_cache_requires_full_buffer(
    const BnBackendRuntimePolicy *policy,
    int tensor_type);
int bn_gpu_policy_moe_down_small_expert_f32_cache_enabled(
    const BnGPUBackend *gpu,
    int tensor_type);
int bn_gpu_policy_moe_quant_only_after_cache(int tensor_type,
                                             int native_quant_f16_cache);
int bn_gpu_policy_moe_prefers_quant_only(const BnGPUBackend *gpu,
                                         int tensor_type);
int bn_gpu_policy_matvec_disabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_matvec_type_disabled(
    const BnBackendRuntimePolicy *policy, int tensor_type);
int bn_gpu_policy_matvec_type_supported(
    const BnBackendRuntimePolicy *policy, int tensor_type);
int bn_gpu_policy_matmul_batch_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_matvec_batch_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_small_state_native_quant_enabled(
    const BnBackendRuntimePolicy *policy,
    int uses_float_kquant_fallback);
int bn_gpu_policy_small_state_native_quant_disabled(
    const BnBackendRuntimePolicy *policy);
size_t bn_gpu_policy_max_storage_binding_bytes(
    const BnBackendRuntimePolicy *policy, size_t backend_limit);
size_t bn_gpu_policy_layout_reserve_bytes(
    const BnBackendRuntimePolicy *policy);
size_t bn_gpu_policy_moe_full_reserve_bytes(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_cublas_matmul_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_cublas_gemm_algo_index_or_default(
    const BnBackendRuntimePolicy *policy, int default_index);
int bn_gpu_policy_cuda_down_kquant_cublas_f16_cache_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_native_quant_matmul_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_f16_native_quant_matmul_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_native_quant_prepared_input_split_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_native_quant_prepared_input_all_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_native_quant_prepared_input_logits_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_native_quant_prepared_input_logits_default_enabled(
    const BnBackendRuntimePolicy *policy,
    int prepared_input_logits_disabled);
int bn_gpu_policy_prepared_kquant_input_cache_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_quant_matmul_preferred_for_type(
    const BnBackendRuntimePolicy *policy,
    int tensor_type,
    int f16_native_quant_matmul_enabled);
int bn_gpu_policy_cuda_down_kquant_4warp_long_enabled(const BnBackendRuntimePolicy *policy, int rows, int cols);
int bn_gpu_policy_cuda_down_kquant_5warp_shape_enabled(const BnBackendRuntimePolicy *policy, int rows, int cols);
int bn_gpu_policy_cuda_down_kquant_3warp_shape_enabled(const BnBackendRuntimePolicy *policy, int rows, int cols);
int bn_gpu_policy_cuda_down_kquant_2warp_long_enabled(const BnBackendRuntimePolicy *policy, int rows, int cols);
int bn_gpu_policy_cuda_down_kquant_matvec4_shape_disabled(const BnBackendRuntimePolicy *policy, int rows, int cols);
int bn_gpu_policy_moe_route_all_active_two(int n_experts, int k);
int bn_gpu_policy_moe_route_expanded_topk(int n_experts, int k);
int bn_gpu_policy_moe_route_all_active_two_large_hidden(int n_experts,
                                                        int k,
                                                        int hidden_dim);
int bn_gpu_policy_cuda_moe_down_quant_path_preferred(
    const BnBackendRuntimePolicy *policy,
    int routed_asymmetric_kquant,
    int down_type,
    int hidden_dim,
    int n_experts,
    int k);
int bn_gpu_policy_cuda_moe_down_f32_cache_path_enabled(
    const BnBackendRuntimePolicy *policy,
    int routed_asymmetric_kquant,
    int down_type,
    int has_f32_data,
    int prefer_quant_down,
    int dim,
    int hidden_dim,
    int n_experts,
    int k);
int bn_gpu_policy_cuda_moe_down_4row_enabled(
    const BnBackendRuntimePolicy *policy, int hidden_dim);
int bn_gpu_policy_cuda_moe_down_8row_enabled(
    const BnBackendRuntimePolicy *policy, int hidden_dim);
int bn_gpu_policy_cuda_moe_down_halfwarp_enabled(
    const BnBackendRuntimePolicy *policy,
    int down_type,
    int prefer_quant_down,
    int n_experts,
    int k);
int bn_gpu_policy_cuda_moe_down_split4_enabled(
    const BnBackendRuntimePolicy *policy,
    int down_type,
    int use_halfwarp,
    int n_experts,
    int k);
int bn_gpu_policy_cuda_moe_down_scatter_enabled(
    const BnBackendRuntimePolicy *policy,
    int down_type,
    int use_halfwarp,
    int use_split4);
int bn_gpu_policy_cuda_moe_down_scatter_16row_enabled(
    const BnBackendRuntimePolicy *policy,
    int use_scatter,
    int hidden_dim);
int bn_gpu_policy_cuda_moe_down_float_path_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_down_pair_path_enabled(
    const BnBackendRuntimePolicy *policy,
    int f32_down_default,
    int pair_down_f32_layer,
    int all_active_two_disable_pair_down);
int bn_gpu_policy_cuda_moe_down_prefers_f32_cache(
    const BnBackendRuntimePolicy *policy,
    int has_f32_data,
    int hidden_dim,
    int all_active_two_kquant,
    int all_active_two_f32_down);
int bn_gpu_policy_cuda_moe_down_f32_pair2_enabled(const BnBackendRuntimePolicy *policy, int n_experts,
                                                      int k);
int bn_gpu_policy_cuda_moe_down_f32_pair2_4row_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_accum_enabled(
    const BnBackendRuntimePolicy *policy,
    int all_active_two_kquant);
int bn_gpu_policy_all_active_two_kquant_moe_down_pair4_sum_enabled(const BnBackendRuntimePolicy *policy, int all_active_two_kquant);
int bn_gpu_policy_cuda_moe_down_prepared_native_quant_4row_sum_enabled(
    const BnBackendRuntimePolicy *policy,
    int all_active_two_kquant,
    int k,
    int hidden_dim);
int bn_gpu_policy_cuda_moe_down_prepared_native_quant_8row_sum_enabled(
    const BnBackendRuntimePolicy *policy,
    int prepared_native_quant_4row_sum,
    int hidden_dim);
int bn_gpu_policy_all_active_two_kquant_moe_down_fixed_enabled(
    const BnBackendRuntimePolicy *policy,
    int all_active_two_kquant);
int bn_gpu_policy_cuda_moe_down_resid_rmsnorm_fuse_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_down_prepared_native_quant_shape_2048_768_enabled(
    const BnBackendRuntimePolicy *policy,
    int dim,
    int hidden_dim,
    int k);
int bn_gpu_policy_all_active_two_kquant_moe_down_accum_4row_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_down_prepared_pair_4row_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_down_f32_cache_enabled(
    const BnBackendRuntimePolicy *policy,
    int has_f32_data,
    int all_active_two_disable_f32_cache);
int bn_gpu_policy_cuda_moe_down_f16_cache_enabled(const BnBackendRuntimePolicy *policy, int has_f16_data);
int bn_gpu_policy_cuda_moe_down_aux_f32_cache_enabled(const BnBackendRuntimePolicy *policy, int has_f32_data);
int bn_gpu_policy_cuda_moe_down_prepared_pair8_enabled(const BnBackendRuntimePolicy *policy, int n_experts,
                                                       int k,
                                                       int hidden_dim);
int bn_gpu_policy_cuda_moe_down_prepared_8row_enabled(const BnBackendRuntimePolicy *policy, int hidden_dim);
int bn_gpu_policy_cuda_moe_gateup_prepared_dot_enabled(const BnBackendRuntimePolicy *policy, int n_tokens,
                                                       int dim,
                                                       int allow_small_dim);
int bn_gpu_policy_cuda_moe_gateup_prepared_8row_enabled(
    const BnBackendRuntimePolicy *policy, int dim);
int bn_gpu_policy_cuda_moe_gateup_prepared_split_enabled(
                                                         const BnBackendRuntimePolicy *policy,
                                                         int dim,
                                                         int n_experts);
int bn_gpu_policy_cuda_moe_route_dot_prepared_input_enabled(
    const BnBackendRuntimePolicy *policy,
    int dim,
    int all_active_two_kquant);
int bn_gpu_policy_cuda_moe_route_block_prepared_input_enabled(
    const BnBackendRuntimePolicy *policy,
    int dim,
    int all_active_two_kquant,
    int uses_reference_silu);
int bn_gpu_policy_cuda_moe_router_fused_topk_enabled(
    const BnBackendRuntimePolicy *policy, int n_experts, int route_block);
int bn_gpu_policy_cuda_moe_router_warp_disabled(
    const BnBackendRuntimePolicy *policy, int route_block);
int bn_gpu_policy_cuda_moe_router_4warp_enabled(
    const BnBackendRuntimePolicy *policy, int dim);
int bn_gpu_policy_cuda_moe_router_2warp_enabled(
    const BnBackendRuntimePolicy *policy, int dim);
int bn_gpu_policy_cuda_moe_router_warp_topk_enabled(
    const BnBackendRuntimePolicy *policy, int n_experts);
int bn_gpu_policy_cuda_moe_block_prepared_batch_enabled(const BnBackendRuntimePolicy *policy, int routed_native_quant);
int bn_gpu_policy_cuda_moe_block_prepared_decode_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_gateup_block_2row_enabled(const BnBackendRuntimePolicy *policy, int hidden_dim);
int bn_gpu_policy_cuda_moe_down_block_4row_enabled(const BnBackendRuntimePolicy *policy, int hidden_dim);
int bn_gpu_policy_cuda_moe_down_block_2row_enabled(const BnBackendRuntimePolicy *policy, int hidden_dim);
int bn_gpu_policy_cuda_moe_all_active_two_fast_enabled(
    const BnBackendRuntimePolicy *policy,
    int all_active_two_graph_kquant);
int bn_gpu_policy_cuda_moe_prepared_dot_enabled(
    const BnBackendRuntimePolicy *policy,
    int use_all_active_two_prepared_default,
    int fast_prepared_gateup,
    int all_active_two_kquant,
    int hidden_dim,
    int dim);
int bn_gpu_policy_cuda_moe_internal_profile_enabled(
    const BnBackendRuntimePolicy *policy, int profile);
int bn_gpu_policy_cuda_moe_all_active_two_fixed_prepared_4row_enabled(
    const BnBackendRuntimePolicy *policy,
    int prepared_dot_input,
    int all_active_two_fast_enabled);
int bn_gpu_policy_cuda_moe_gateup_prepared_4row_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_decode_logits_cache_enabled(
    const BnGPUBackend *gpu, int gpu_logits_need_cpu);
int bn_gpu_policy_moe_decode_cache_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_decode_cache_disabled(const BnGPUBackend *gpu);
int bn_gpu_policy_decode_cache_disabled(const BnGPUBackend *gpu);
int bn_gpu_policy_native_quant_decode_cache_disabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_logits_argmax_disabled(const BnGPUBackend *gpu);
int bn_gpu_policy_dense_logits_argmax_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_logits_mmvq_argmax_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_logits_mmvq_argmax_disabled(const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_moe_logits_mmvq_argmax_path_enabled(
    const BnBackendRuntimePolicy *policy, int rows, int cols);
int bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_enabled(
    const BnBackendRuntimePolicy *policy, int use_mmvq, int rows, int cols);
int bn_gpu_policy_cuda_moe_logits_mmvq_1warp16_1536_enabled(
    const BnBackendRuntimePolicy *policy, int use_1warp8);
int bn_gpu_policy_cuda_moe_logits_mmvq_1warp8_1536_unroll_enabled(
    const BnBackendRuntimePolicy *policy, int use_1warp8, int use_1warp16);
int bn_gpu_policy_cuda_argmax_fast_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_optimistic_argmax_penalty_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_legacy_block_matvec4_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_legacy_block_warp_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_deinterleaved_kquant_pair_matvec_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_deinterleaved_kquant_4warp_enabled(const BnBackendRuntimePolicy *policy, int cols);
int bn_gpu_policy_cuda_deinterleaved_kquant_split_4warp_enabled(const BnBackendRuntimePolicy *policy, int cols);
int bn_gpu_policy_cuda_deinterleaved_kquant_gateup_2warp_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_symmetric_kquant_dot_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_deinterleaved_kquant_dot_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_asymmetric_kquant_4warp_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_asymmetric_kquant_4warp_shape_enabled(const BnBackendRuntimePolicy *policy, int rows,
                                                             int cols);
int bn_gpu_policy_cuda_asymmetric_kquant_out_residual_rmsnorm_fuse_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_asymmetric_kquant_qkv_mixed_fuse_enabled(
    const BnBackendRuntimePolicy *policy, int tensor_type);
int bn_gpu_policy_cuda_asymmetric_kquant_split_k_rope_cache_fuse_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_asymmetric_kquant_split_qk_rope_cache_fuse_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_asymmetric_kquant_split_4warp_enabled(const BnBackendRuntimePolicy *policy, int cols);
int bn_gpu_policy_cuda_asymmetric_kquant_split_5warp_enabled(const BnBackendRuntimePolicy *policy, int cols);
int bn_gpu_policy_cuda_asymmetric_kquant_split_value_rows(const BnBackendRuntimePolicy *policy, int total_rows,
                                                          int cols);
int bn_gpu_policy_cuda_asymmetric_kquant_split_value_fuse_enabled(
    const BnBackendRuntimePolicy *policy, int value_rows);
int bn_gpu_policy_kquant_gateup_prepared_path_enabled(
    const BnBackendRuntimePolicy *policy, int uses_prepared_kquant_input);
int bn_gpu_policy_cuda_asymmetric_kquant_gateup_qwarp4_enabled(const BnBackendRuntimePolicy *policy, int cols);
int bn_gpu_policy_cuda_asymmetric_kquant_gateup_5warp_enabled(
    const BnBackendRuntimePolicy *policy,
    int enable_asymmetric_kquant_4warp,
    int cols);
int bn_gpu_policy_cuda_asymmetric_kquant_gateup_2warp_enabled(
    const BnBackendRuntimePolicy *policy,
    int enable_asymmetric_kquant_4warp,
    int cols);
int bn_gpu_policy_cuda_asymmetric_kquant_gateup_4warp_enabled(
    int enable_asymmetric_kquant_4warp,
    int cols);
int bn_gpu_policy_cuda_native_quant_warp_disabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_native_quant_ssm_matvec_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_native_quant_ssm_prepared_input_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_native_quant_mixed_prepared_input_enabled(
    const BnBackendRuntimePolicy *policy,
    int type_a,
    int type_b,
    int cols);
int bn_gpu_policy_cuda_f16_native_quant_ssm_matvec_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_f16_native_quant_matvec_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_f16_packed_kquant_matvec_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_symmetric_kquant_pair_matvec_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_kquant_dot_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_kquant_dot_forced(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_kquant_matvec4_enabled(const BnBackendRuntimePolicy *policy, int cols);
int bn_gpu_policy_cuda_asymmetric_kquant_matmul8_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_asymmetric_kquant_sharedx_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_asymmetric_kquant_batch_sharedx_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_down_kquant_dot_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_down_kquant_dot_forced(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_down_kquant_warp_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_asymmetric_kquant_pair_matvec_enabled(const BnBackendRuntimePolicy *policy, int cols);
int bn_gpu_policy_cuda_down_kquant_prepared_dot_enabled(const BnBackendRuntimePolicy *policy, int is_logits_op);
int bn_gpu_policy_cuda_down_kquant_mmvq_enabled(const BnBackendRuntimePolicy *policy, int rows,
                                                int cols,
                                                int is_logits_op,
                                                int uses_reference_kquant_matvec);
int bn_gpu_policy_cuda_down_kquant_mmvq_2warp_logits_enabled(const BnBackendRuntimePolicy *policy, int rows,
                                                             int cols,
                                                             int is_logits_op);
int bn_gpu_policy_cuda_down_kquant_residual_rmsnorm_fuse_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_f16_down_kquant_matvec_enabled(const BnBackendRuntimePolicy *policy, int rows,
                                                      int cols,
                                                      int uses_reference_kquant_matvec);
int bn_gpu_policy_cuda_down_kquant_matmul8_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_down_kquant_matmul4_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_down_kquant_matvec4_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_down_kquant_batch_warp_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_fuse_bias_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_rope_flash_fuse_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_bias_rope_flash_fuse_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_qk_norm_rope_flash_fuse_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_qk_norm_rope_fuse_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_weighted_add_sigmoid_residual_rmsnorm_fuse_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_weighted_add_sigmoid_residual_fuse_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_readback_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_cublas_cache_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_cublas_cache_reserve_mb_or_default(
    const BnBackendRuntimePolicy *policy, int default_mb);
int bn_gpu_policy_cuda_cublas_workspace_mb_or_default(
    const BnBackendRuntimePolicy *policy, int default_mb);
int bn_gpu_policy_cuda_nan_verbose_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_stream_exec_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_profile_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_wall_profile_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every);
int bn_gpu_policy_cuda_wall_profile_detail_limit_or_default(
    const BnBackendRuntimePolicy *policy, int default_limit);
int bn_gpu_policy_cuda_wall_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every);
int bn_gpu_policy_cuda_profile_shapes_enabled(
    const BnBackendRuntimePolicy *policy);
const char *bn_gpu_policy_cuda_device_selector(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_exec_fail_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_sync_each_op_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_nan_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_dump_ops_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_dump_ops_every_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_dump_ops_limit_or_default(
    const BnBackendRuntimePolicy *policy, int default_limit);
int bn_gpu_policy_cuda_prefill_moe_layer_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_dense_layer_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_dense_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_dense_profile_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_dense_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every);
int bn_gpu_policy_cuda_prefill_ssm_layer_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_prefill_ssm_layer_disabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_prefill_fused_asymmetric_kquant_gateup_batch_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_ssm_fused_asymmetric_kquant_gateup_batch_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_ssm_profile_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_ssm_stacked_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_ssm_stream_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_ssm_input_alias_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_ssm_f32_ab_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_ssm_scan_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_ssm_delta_128_warp_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_ssm_ffn_profile_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_ssm_ffn_gateup_f16_out_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_backend_opt_in_fused_gateup_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_fused_gateup_silu_allowed(const BnGPUBackend *gpu,
                                            int tensor_type);
int bn_gpu_policy_shared_kquant_dot_enabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_shared_expert_gate_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_attention_min_tokens_configured(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_prefill_attention_min_tokens_configured(
    const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_prefill_attention_min_tokens_or_default(
    const BnBackendRuntimePolicy *policy, int default_tokens);
int bn_gpu_policy_prefill_attention_min_tokens_or_default(
    const BnGPUBackend *gpu, int default_tokens);
int bn_gpu_policy_cuda_prefill_gemm_attention_min_tokens_or_default(
    const BnBackendRuntimePolicy *policy, int default_tokens);
int bn_gpu_policy_cuda_prefill_gemm_attention_enabled(
    const BnBackendRuntimePolicy *policy, int n_tokens, int max_tokens);
int bn_gpu_policy_cuda_prefill_attention_wo_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_qkv_attention_wo_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_batched_gemm_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_prefill_gemm_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_prefill_dense_chain_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_prefill_hybrid_chain_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_prefill_attention_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_prefill_attention_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_prefill_ssm_run_chain_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_prefill_ssm_ffn_fuse_allowed(const BnGPUBackend *gpu);
int bn_gpu_policy_prefill_moe_chain_debug_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_prefill_hybrid_chain_debug_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_moe_prefill_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_prefill_min_tokens_configured(
    const BnGPUBackend *gpu);
int bn_gpu_policy_moe_prefill_min_tokens_or_default(
    const BnGPUBackend *gpu, int default_tokens);
int bn_gpu_policy_cuda_dense_ffn_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_dense_ffn_batch_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_cublas_gateup_f16_out_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_cublas_grouped_variable_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_cublas_grouped_enabled(
                                                  const BnBackendRuntimePolicy *policy,
                                                  int routed_native_quant,
                                                  int routed_asymmetric_kquant,
                                                  int gate_f16,
                                                  int up_f16,
                                                  int down_f16,
                                                  int n_experts,
                                                  int k,
                                                  int route_items);
int bn_gpu_policy_cuda_moe_cublas_gateup_only_enabled(
                                                      const BnBackendRuntimePolicy *policy,
                                                      int use_grouped,
                                                      int routed_native_quant,
                                                      int routed_asymmetric_kquant,
                                                      int gate_f16,
                                                      int up_f16,
                                                      int down_f16,
                                                      int n_tokens);
int bn_gpu_policy_cuda_moe_cublas_all_active_two_fixed_enabled(
    const BnBackendRuntimePolicy *policy,
    int use_grouped,
    int n_experts,
    int k);
int bn_gpu_policy_cuda_moe_cublas_all_active_two_decode_enabled(
    const BnBackendRuntimePolicy *policy,
    int n_tokens,
    int routed_asymmetric_kquant,
    int down_type,
    int hidden_dim,
    int n_experts,
    int k,
    int gate_f16,
    int up_f16,
    int down_f16);
int bn_gpu_policy_cuda_moe_sorted_slots_enabled(
                                                const BnBackendRuntimePolicy *policy,
                                                int routed_asymmetric_kquant,
                                                int routed_native_quant,
                                                int n_tokens,
                                                int use_all_active_two_fixed,
                                                int use_grouped,
                                                int use_gateup_only);
int bn_gpu_policy_cuda_moe_prefill_internal_profile_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_prefill_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every);
int bn_gpu_policy_cuda_moe_prefill_direct_resid_out_enabled(
    const BnBackendRuntimePolicy *policy,
    int add_norm_resid,
    int out_provided,
    int has_shared,
    int init_out_with_residual);
int bn_gpu_policy_cuda_moe_batch_fused_route_topk_enabled(
    const BnBackendRuntimePolicy *policy, int n_experts);
int bn_gpu_policy_cuda_moe_route_dist_profile_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_route_dist_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every);
int bn_gpu_policy_cuda_moe_cublas_grouped_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_cublas_gateup_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_ffn_batch_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_ffn_batch_profile_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_ffn_batch_profile_every_or_default(
    const BnBackendRuntimePolicy *policy, int default_every);
int bn_gpu_policy_moe_cache_prefill_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_prefill_shared_fuse_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_route_batch_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_route_batch_debug_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_large_hybrid_attention_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_large_hybrid_cpu_attention_safe_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_large_hybrid_cpu_attention_safe_disabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_large_hybrid_cpu_attention_safe_forced(
    const BnGPUBackend *gpu);
int bn_gpu_policy_large_hybrid_prefill_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_large_hybrid_prefill_chain_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_large_hybrid_prefill_disabled(const BnGPUBackend *gpu);
int bn_gpu_policy_large_hybrid_argmax_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_prefill_matmul_disabled(const BnGPUBackend *gpu);
int bn_gpu_policy_prefill_matmul_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_prefill_direct_kv_disabled(const BnGPUBackend *gpu);
int bn_gpu_policy_prefill_direct_kv_with_cpu_fallback_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_cpu_decode_fallback_requested(const BnGPUBackend *gpu);
int bn_gpu_policy_cpu_fallback_layer_or_default(
    const BnGPUBackend *gpu, int default_layer);
int bn_gpu_policy_cpu_fallback_from_layer_or_default(
    const BnGPUBackend *gpu, int default_layer);
int bn_gpu_policy_cpu_attention_layer_or_default(
    const BnGPUBackend *gpu, int default_layer);
int bn_gpu_policy_cpu_attention_from_layer_or_default(
    const BnGPUBackend *gpu, int default_layer);
int bn_gpu_policy_cpu_ffn_layer_or_default(
    const BnGPUBackend *gpu, int default_layer);
int bn_gpu_policy_cpu_ffn_from_layer_or_default(
    const BnGPUBackend *gpu, int default_layer);
int bn_gpu_policy_cpu_ffn_down_from_layer_or_default(
    const BnGPUBackend *gpu, int default_layer);
int bn_gpu_policy_ssm_graph_disabled(const BnGPUBackend *gpu);
int bn_gpu_policy_cuda_qkv_mixed_fuse_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_qkv_key_cache_fuse_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_qkv_kpair_opt_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_legacy_block_gateup_warp_disabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_native_quant_gateup_warp_disabled(const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_graph_exec_requested(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_graph_max_experts_or_default(
    const BnBackendRuntimePolicy *policy, int default_experts);
int bn_gpu_policy_cuda_decode_graph_default_enabled(
    const BnBackendRuntimePolicy *policy, int moe_graph,
    int default_moe_graph);
int bn_gpu_policy_cuda_cublas_cache_max_mb(
    const BnBackendRuntimePolicy *policy, int default_mb, int large_budget);
int bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
                                               const BnBackendRuntimePolicy *policy,
                                               int tensor_type,
                                               int force_down_kquant_f32,
                                               int force_f16);
int bn_gpu_policy_cuda_down_kquant_f16_cache_adds_f32_down_cache(const BnBackendRuntimePolicy *policy);
size_t bn_gpu_policy_cuda_moe_down_cublas_cache_bytes(
    const BnGPUBackend *gpu,
    int tensor_type,
    int rows,
    int cols);
size_t bn_gpu_policy_moe_down_aux_cache_bytes(const BnGPUBackend *gpu,
                                              int tensor_type,
                                              int rows,
                                              int cols);
size_t bn_gpu_policy_aux_cache_bytes(const BnGPUBackend *gpu,
                                     int tensor_type,
                                     int rows,
                                     int cols);
int bn_gpu_policy_cuda_cublas_aux_cache_supported(
                                                  const BnBackendRuntimePolicy *policy,
                                                  int tensor_type,
                                                  int cols);
int bn_gpu_policy_moe_auto_resident_enabled(const BnGPUBackend *gpu);
size_t bn_gpu_policy_moe_cache_reserve_bytes(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_auto_caps_sequence(int webgpu,
                                     int cuda,
                                     int metal,
                                     int has_moe,
                                     int model_seq_len,
                                     int cap_seq_len);
int bn_gpu_policy_duplicate_moe_cache_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_webgpu_repacked_buffer_supported(int tensor_type);
int bn_gpu_policy_webgpu_repacked_bias_supported(int tensor_type);
int bn_gpu_policy_metal_mmap_zero_copy_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_apply_metal_barrier_disable_override(
    BnBackendRuntimePolicy *policy);
int bn_gpu_policy_apply_specialized_native_quant_decode_override(
    BnBackendRuntimePolicy *policy);
int bn_gpu_policy_apply_specialized_native_quant_decode_disable_override(
    BnBackendRuntimePolicy *policy);
int bn_gpu_policy_apply_native_quant_prepared_override(
    BnBackendRuntimePolicy *policy);
int bn_gpu_policy_apply_metal_small_dense_native_quant_default_disable_override(
    BnBackendRuntimePolicy *policy);
int bn_gpu_policy_apply_metal_private_weights_override(
    BnBackendRuntimePolicy *policy);
int bn_gpu_policy_apply_metal_cpu_route_resident_moe_override(
    BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_apply_small_dense_native_quant_default(
    BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_small_dense_native_quant_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_native_quant_prepared_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_prepared_f32_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_small_dense_native_quant_prepared_layer_default_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_native_quant_prepared_upload_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_repacked_buffer_supported(int tensor_type);
int bn_gpu_policy_metal_repacked_buffer_type(int tensor_type);
int bn_gpu_policy_metal_prepared_stacked_upload_blocked(
    const BnBackendRuntimePolicy *policy, int tensor_type);
int bn_gpu_policy_metal_shared_weights_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_specialized_native_quant_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_specialized_native_quant_decode_path_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_native_quant_barriers_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_small_dense_native_quant_matvec_supported(int tensor_type,
                                               int small_dense_native_quant_enabled,
                                               int native_quant_prepared,
                                               int has_native_quant_pipeline,
                                               int has_native_quant_pipeline_unprepared,
                                               int has_native_quant_prepared_pipeline);
int bn_gpu_policy_metal_small_dense_native_quant_graph_path_supported(
    int tensor_type,
    int small_dense_native_quant_enabled,
    int native_quant_prepared,
    int native_quant_prepared_path,
    int has_native_quant_pipeline,
    int has_pipeline);
int bn_gpu_policy_metal_block_q8_activation_graph_path_supported(
    int tensor_type,
    int block_q8_activation_enabled,
    int has_activation_quant_pipeline,
    int has_matvec_pipeline);
int bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
    const BnBackendRuntimePolicy *policy,
    int tensor_type,
    int cols,
    int has_prepared_activation_pipeline,
    int has_specialized_native_pipeline);
int bn_gpu_policy_metal_specialized_native_quant_shape_default_enabled(
    const BnBackendRuntimePolicy *policy,
    int tensor_type, int rows, int cols);
int bn_gpu_policy_metal_cpu_order_rmsnorm_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_full_barriers_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_route_history_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_barriers_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_barriers_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_fused_gateup_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_small_dense_native_quant_fused_gateup_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_small_dense_native_quant_attn_only_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_small_dense_native_quant_ffn_only_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_small_dense_native_quant_from_layer_or_default(
    const BnBackendRuntimePolicy *policy, int n_layers);
int bn_gpu_policy_small_dense_native_quant_to_layer_or_default(
    const BnBackendRuntimePolicy *policy,
    int n_layers,
    int native_quant_prepared);
int bn_gpu_policy_gateup_split_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_small_dense_native_quant_ffn_down_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_qkv_split_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_qkv_split_debug_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_ssm_qkvz_split_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_ssm_ab_stack_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_split_residual_rmsnorm_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_debug_fallback_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_metal_cpu_route_resident_moe_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_dense_residual_graph_diagnostic_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_shared_mmap_buffer_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_force_graph_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_flash_min_kv_or_default(const BnGPUBackend *gpu,
                                          int default_min_kv);
int bn_gpu_policy_backend_flash_max_kv_or_default(const BnGPUBackend *gpu,
                                                  int default_max_kv);
int bn_gpu_policy_backend_flash_default_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_backend_large_graph_native_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_small_dense_native_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_all_active_two_kquant_moe_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_cpu_attention_fallback_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_reference_attention_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_reference_attention_native_graph_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_reference_attention_token_fallback_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_reference_recurrent_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_reference_attention_fallback_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_small_dense_native_quant_supported(
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
int bn_gpu_policy_backend_native_quant_logits_refine_default_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_all_active_two_kquant_moe_logits_refine_default_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_decode_graph_cache_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_moe_reference_attention_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_ssm_graph_supported(const BnGPUBackend *gpu);
int bn_gpu_policy_backend_large_hybrid_argmax_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_all_active_two_moe_direct_route_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_resident_moe_ffn_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_moe_expert_graph_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_weighted_add_sigmoid_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_backend_moe_route_topk_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_metal_routed_moe_decode_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_moe_expert_graph_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_backend_moe_gateup_split_supported(
    const BnGPUBackend *gpu);
int bn_gpu_policy_argmax_debug_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_cpu_logits_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_compare_logits_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_debug_argmax_compare_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_backend_kquant_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int kquant_refine_default);
int bn_gpu_policy_kquant_logits_refine_top_or_default(
    const BnBackendRuntimePolicy *policy, int default_top);
int bn_gpu_policy_backend_native_quant_logits_refine_enabled(
    const BnGPUBackend *gpu,
    int native_quant_refine_default);
int bn_gpu_policy_native_quant_logits_refine_top_or_default(
    const BnBackendRuntimePolicy *policy, int default_top);
int bn_gpu_policy_moe_ffn_disabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_router_topk_enabled(const BnGPUBackend *gpu,
                                          int eligible);
int bn_gpu_policy_native_quant_moe_cpu_route_resident_enabled(
    const BnGPUBackend *gpu, int eligible);
int bn_gpu_policy_lowbit_block32_moe_cpu_route_resident_enabled(
    const BnGPUBackend *gpu, int eligible);
int bn_gpu_policy_moe_router_gpu_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_router_diff2_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_routed_ffn_batch_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_routed_ffn_batch_allowed(
                                                const BnBackendRuntimePolicy *policy,
                                                int large_moe);
int bn_gpu_policy_moe_cpu_actual_override_enabled(
    const BnGPUBackend *gpu);
int bn_gpu_policy_small_dense_native_quant_cpu_attention_safe_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_small_dense_native_quant_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_small_dense_native_quant_ffn_down_requested(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_small_dense_prefill_disabled(const BnGPUBackend *gpu);
int bn_gpu_policy_native_quant_logits_refine_requested(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_native_quant_logits_refine_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_fast_ffn_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_fast_graph_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_cublas_decode_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_cublas_decode_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_cuda_moe_cublas_decode_debug_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_fast_route_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_dot_prepared_input_default_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_route_dot_prepared_input_default_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_route_block_prepared_input_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_fast_prepared_gateup_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layers_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_pair_path_f32_layer_selected(
    const BnBackendRuntimePolicy *policy, int layer);
int bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_ordered_quant_path_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_default_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_f32_all_active_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_f32_cache_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_default_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_float_4row_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_f32_4row_layer_selected(
    const BnBackendRuntimePolicy *policy, int layer);
int bn_gpu_policy_all_active_two_kquant_moe_down_f32_4row_default_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_down_f32_4row_disabled(
    const BnBackendRuntimePolicy *policy);
float bn_gpu_policy_all_active_two_kquant_down_skip_eps_or_default(
    const BnBackendRuntimePolicy *policy, float default_eps);
int bn_gpu_policy_all_active_two_kquant_moe_cpu_attention_safe_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_logits_refine_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_cpu_moe_safe_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_reference_attention_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_cpu_route_resident_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_reference_gpu_route_requested(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_reference_gpu_route_disabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_all_active_two_kquant_moe_route_selection_enabled(
    const BnBackendRuntimePolicy *policy);
void bn_gpu_policy_all_active_two_kquant_moe_route_layer_range(
    const BnBackendRuntimePolicy *policy,
    int *from_layer,
    int *to_layer);
int bn_gpu_policy_moe_compare_layer_selected(
    const BnBackendRuntimePolicy *policy, int layer, int pos);
int bn_gpu_policy_moe_compare_input_norm_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_compare_actual_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_compare_route_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_compare_raw_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_compare_mid_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_compare_parts_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_compare_shared_mid_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_compare_shared_down_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_compare_norm_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_compare_attention_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer);
int bn_gpu_policy_compare_attention_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos);
int bn_gpu_policy_compare_gqa_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer);
int bn_gpu_policy_compare_gqa_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos);
int bn_gpu_policy_compare_qkv_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer);
int bn_gpu_policy_compare_qkv_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos);
int bn_gpu_policy_compare_ffn_down_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer);
int bn_gpu_policy_compare_ffn_down_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos);
int bn_gpu_policy_compare_ffn_state_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer);
int bn_gpu_policy_compare_ffn_state_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos);
int bn_gpu_policy_compare_ssm_layer_or_default(
    const BnBackendRuntimePolicy *policy, int default_layer);
int bn_gpu_policy_compare_ssm_pos_or_default(
    const BnBackendRuntimePolicy *policy, int default_pos);
int bn_gpu_policy_metal_reference_attention_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_prepared_native_quant_attention_enabled(
    const BnBackendRuntimePolicy *policy);
uint32_t bn_gpu_policy_metal_reference_attention_stage_mask(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_metal_ssm_graph_enabled(
    const BnBackendRuntimePolicy *policy);
int bn_gpu_policy_moe_shared_cpu_fallback_enabled(
    const BnGPUBackend *gpu, int eligible);
int bn_gpu_policy_moe_gateup_split_enabled(
    const BnGPUBackend *gpu, int can_split);
int bn_gpu_policy_moe_route_profile_enabled(const BnGPUBackend *gpu);
int bn_gpu_policy_moe_route_profile_every_or_default(
    const BnGPUBackend *gpu, int default_every);
int bn_gpu_policy_profile_level(const BnBackendRuntimePolicy *policy);

#ifdef __cplusplus
}
#endif

#endif // BN_GPU_POLICY_H
