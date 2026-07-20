#ifndef BN_BACKEND_QUANT_H
#define BN_BACKEND_QUANT_H

#include <stdint.h>
#include "gguf.h"
#include "gpu_backend.h"
#include "model_weights.h"
#include "quant.h"

#define BN_BACKEND_QUANT_GPU_MATVEC_FLAG_Q8K_DOT BN_QUANT_GPU_MATVEC_FLAG_Q8K_DOT
#define BN_BACKEND_QUANT_GPU_MATVEC_FLAG_EXACT_Q6K BN_QUANT_GPU_MATVEC_FLAG_EXACT_Q6K

static inline uint32_t bn_backend_quant_gpu_split_cap(int type) {
    return bn_quant_format_gpu_split_cap(type);
}

static inline int bn_backend_quant_can_gpu_split(int type) {
    return bn_quant_format_can_gpu_split(type);
}

static inline int bn_backend_quant_can_gpu_native(int type) {
    return bn_quant_format_can_gpu_native(type);
}

static inline int bn_backend_quant_can_gpu_repack(int type) {
    return bn_quant_format_can_gpu_repack(type);
}

static inline int bn_backend_quant_gpu_supports_repacked_bias(int type) {
    return bn_quant_format_gpu_supports_repacked_bias(type);
}

static inline int bn_backend_quant_dense_graph_supported(int type) {
    return bn_quant_format_supports_gpu_dense_graph(type);
}

static inline int bn_backend_quant_dense_graph_native_quant_supported(int type) {
    return bn_quant_format_supports_gpu_dense_graph_native_quant(type);
}

static inline int bn_backend_quant_dense_graph_weight_supported(
    const BnQWeight *w,
    int native_quant_only) {
    if (!w || !w->data)
        return 1;
    return native_quant_only
        ? bn_backend_quant_dense_graph_native_quant_supported(w->type)
        : bn_backend_quant_dense_graph_supported(w->type);
}

static inline int bn_backend_quant_dense_graph_tensor_supported(
    int tensor_type,
    int native_quant_only) {
    return native_quant_only
        ? bn_backend_quant_dense_graph_native_quant_supported(tensor_type)
        : bn_backend_quant_dense_graph_supported(tensor_type);
}

static inline int bn_backend_quant_dense_graph_model_supported(
    const BnWeights *w,
    const BnConfig *c,
    int native_quant_only) {
    if (!w || !c)
        return 0;
    if (w->output_weight.data) {
        if (!bn_backend_quant_dense_graph_weight_supported(&w->output_weight,
                                                           native_quant_only))
            return 0;
    } else if (!bn_backend_quant_dense_graph_tensor_supported(
                   w->emb_type, native_quant_only)) {
        return 0;
    }
    for (int l = 0; l < c->n_layers; l++) {
        const BnLayerWeights *lw = &w->layers[l];
        const BnQWeight *weights[] = {
            &lw->attn.wq, &lw->attn.wk, &lw->attn.wv, &lw->attn.wo,
            &lw->ffn.ffn_gate, &lw->ffn.ffn_up, &lw->ffn.ffn_down,
        };
        int n_weights = (int)(sizeof(weights) / sizeof(weights[0]));
        for (int i = 0; i < n_weights; i++) {
            if (!bn_backend_quant_dense_graph_weight_supported(
                    weights[i], native_quant_only))
                return 0;
        }
    }
    return 1;
}

static inline int bn_backend_quant_is_q4k(int type) {
    return bn_quant_format_is_q4k(type);
}

static inline int bn_backend_quant_is_q5k(int type) {
    return bn_quant_format_is_q5k(type);
}

static inline int bn_backend_quant_is_q8k(int type) {
    return bn_quant_format_is_q8k(type);
}

static inline int bn_backend_quant_is_q8_0(int type) {
    return bn_quant_format_is_q8_0(type);
}

static inline int bn_backend_quant_is_q5_0(int type) {
    return bn_quant_format_is_q5_0(type);
}

static inline int bn_backend_quant_is_bf16(int type) {
    return bn_quant_format_is_bf16(type);
}

static inline int bn_backend_quant_is_q3k(int type) {
    return bn_quant_format_is_q3k(type);
}

static inline int bn_backend_quant_requires_float_kquant_fallback(int type) {
    return bn_quant_format_is_float_kquant_fallback_candidate(type);
}

static inline int bn_backend_quant_supports_prepared_kquant(int type) {
    return bn_quant_format_supports_prepared_kquant(type);
}

static inline int bn_backend_quant_supports_native_quant_logits_refine(int type) {
    return bn_quant_format_supports_q8_logits_refine(type);
}

static inline int bn_backend_quant_logits_uses_f16_path(int type) {
    return bn_quant_format_uses_f16_logits_path(type);
}

static inline int bn_backend_quant_tied_logits_uses_quant_path(int type) {
    return bn_quant_format_tied_logits_uses_quant_path(type);
}

static inline int bn_backend_quant_logits_i8_cache_supported(int type) {
    return bn_quant_format_supports_logits_i8_cache(type);
}

static inline int bn_backend_quant_tied_logits_uses_f16_path(int type) {
    return bn_quant_format_tied_logits_uses_f16_path(type);
}

static inline int bn_backend_quant_tied_logits_i8_weight_type(void) {
    return bn_quant_format_tied_logits_i8_weight_type();
}

static inline int bn_backend_quant_tied_logits_f16_weight_type(void) {
    return bn_quant_format_tied_logits_f16_weight_type();
}

static inline int bn_backend_quant_tied_logits_f32_weight_type(void) {
    return bn_quant_format_tied_logits_f32_weight_type();
}

static inline int bn_backend_quant_dense_f32_type(void) {
    return bn_quant_format_dense_f32_type();
}

static inline int bn_backend_quant_gpu_float_buffer_type(void) {
    return bn_quant_format_gpu_float_buffer_type();
}

static inline int bn_backend_quant_already_f32(int type) {
    return bn_quant_format_is_f32(type);
}

static inline int bn_backend_quant_can_convert_dense_to_f32(int type) {
    return bn_quant_format_can_convert_dense_to_f32(type);
}

static inline int bn_backend_quant_convert_dense_to_f32(
    int type, const void *src, float *dst, int n) {
    return bn_quant_format_convert_dense_to_f32(type, src, dst, n);
}

static inline int bn_backend_quant_gpu_requires_exact_silu(int type) {
    return bn_quant_format_gpu_requires_exact_silu(type);
}

static inline int bn_backend_quant_gpu_prefers_gateup_split(int type) {
    return bn_quant_format_gpu_prefers_gateup_split(type);
}

static inline int bn_backend_quant_moe_route_asymmetric_kquant_down(
    int gate_type,
    int up_type,
    int down_type,
    int allow_asymmetric_kquant_down) {
    return bn_quant_format_supports_moe_q4_down_route(gate_type,
                                                      up_type,
                                                      down_type,
                                                      allow_asymmetric_kquant_down);
}

static inline int bn_backend_quant_moe_routed_asymmetric_kquant(
    int gate_type,
    int up_type,
    int down_type) {
    return bn_backend_quant_moe_route_asymmetric_kquant_down(gate_type,
                                                             up_type,
                                                             down_type,
                                                             1);
}

static inline int bn_backend_quant_moe_routed_kquant_gateup(int gate_type,
                                                            int up_type) {
    return bn_quant_format_supports_moe_q4_gateup(gate_type, up_type);
}

static inline int bn_backend_quant_cpu_fused_kquant_gateup_silu(int gate_type,
                                                                int up_type) {
    return bn_quant_format_supports_cpu_fused_q4_gateup_silu(gate_type,
                                                             up_type);
}

static inline int bn_backend_quant_stacked_pair_same_format(int left_type,
                                                            int right_type) {
    return bn_quant_format_pair_same_format(left_type, right_type);
}

static inline int bn_backend_quant_shared_gateup_batch_type_supported(
    int shared_gate_type, int shared_up_type, int batch_type) {
    return bn_quant_format_supports_shared_gateup_batch(shared_gate_type,
                                                        shared_up_type,
                                                        batch_type);
}

static inline int bn_backend_quant_allows_stacked_layout(int type) {
    return bn_quant_format_allows_stacked_layout(type);
}

static inline int bn_backend_quant_has_embedded_tensor_scale(int type) {
    return bn_quant_format_has_embedded_tensor_scale(type);
}

static inline size_t bn_backend_quant_embedded_tensor_scale_offset(
    int type, int rows, int cols) {
    return bn_quant_embedded_tensor_scale_offset(type, rows, cols);
}

static inline int bn_backend_quant_moe_route_native_quant(int gate_type,
                                                int up_type,
                                                int down_type) {
    return bn_quant_format_supports_moe_q8_route(gate_type,
                                                 up_type,
                                                 down_type);
}

static inline int bn_backend_quant_moe_routed_native_quant(int gate_type,
                                                 int up_type,
                                                 int down_type) {
    return bn_backend_quant_moe_route_native_quant(gate_type, up_type,
                                                   down_type);
}

static inline int bn_backend_quant_moe_routed_op_uses_native_quant(int type) {
    return bn_backend_quant_is_q8_0(type);
}

static inline int bn_backend_quant_moe_down_uses_down_kquant(int down_type) {
    return bn_quant_format_is_q6k(down_type);
}

static inline int bn_backend_quant_moe_down_uses_asymmetric_kquant(
    int down_type) {
    return bn_quant_format_is_q4k(down_type);
}

static inline int bn_backend_quant_moe_down_uses_graph_kquant(int down_type) {
    return bn_backend_quant_moe_down_uses_asymmetric_kquant(down_type) ||
           bn_backend_quant_moe_down_uses_down_kquant(down_type);
}

static inline int bn_backend_quant_gpu_graph_gateup_needs_prepared_input_scratch(
    int type) {
    return bn_backend_quant_is_q4k(type) ||
           bn_backend_quant_is_q5k(type) ||
           bn_backend_quant_is_q8_0(type);
}

static inline int bn_backend_quant_gpu_graph_matvec_needs_prepared_input_scratch(
    int type) {
    return bn_backend_quant_gpu_graph_gateup_needs_prepared_input_scratch(type);
}

static inline int bn_backend_quant_gpu_graph_matvec_down_kquant_needs_dot_scratch(
    int type) {
    return bn_backend_quant_moe_down_uses_down_kquant(type);
}

static inline int bn_backend_quant_gpu_graph_matvec_asymmetric_kquant_needs_dot_scratch(
    int type) {
    return bn_backend_quant_is_q4k(type);
}

static inline int bn_backend_quant_deinterleaved_kquant_pair_matvec(
    int first_type, int second_type) {
    return bn_backend_quant_is_q5k(first_type) &&
           bn_backend_quant_is_q5k(second_type);
}

static inline int bn_backend_quant_asymmetric_kquant_pair_matvec(
    int first_type, int second_type) {
    return bn_backend_quant_moe_down_uses_down_kquant(first_type) &&
           bn_backend_quant_is_q4k(second_type);
}

static inline int bn_backend_quant_symmetric_kquant_pair_matvec(
    int first_type,
    int second_type) {
    return bn_backend_quant_is_q4k(first_type) &&
           bn_backend_quant_is_q4k(second_type);
}

static inline int bn_backend_quant_native_quant_small_state_matvec_candidate(
    int type) {
    return bn_backend_quant_is_q8_0(type);
}

static inline int bn_backend_quant_native_quant_f16_cache_matvec_candidate(
    int type) {
    return bn_backend_quant_is_q8_0(type);
}

static inline int bn_backend_quant_f16_float_cache_matvec_candidate(
    int type) {
    return bn_quant_format_is_f16_float_cache_matvec_candidate(type);
}

static inline int bn_backend_quant_packed_kquant_f16_cache_matvec_candidate(
    int type) {
    return bn_backend_quant_is_q5k(type);
}

static inline int bn_backend_quant_down_kquant_f16_cache_matvec_candidate(
    int type) {
    return bn_backend_quant_moe_down_uses_down_kquant(type);
}

static inline int bn_backend_quant_kquant_logits_cache_matvec_candidate(
    int type) {
    return bn_backend_quant_moe_down_uses_down_kquant(type);
}

static inline int bn_backend_quant_legacy_block_matvec_candidate(int type) {
    return bn_quant_format_is_q5_0(type);
}

static inline int bn_backend_quant_down_kquant_dot_matvec_candidate(int type) {
    return bn_backend_quant_moe_down_uses_down_kquant(type);
}

static inline int bn_backend_quant_down_kquant_warp_matvec_candidate(int type) {
    return bn_backend_quant_moe_down_uses_down_kquant(type);
}

static inline int bn_backend_quant_asymmetric_kquant_dot_matvec_candidate(int type) {
    return bn_backend_quant_is_q4k(type);
}

static inline int bn_backend_quant_asymmetric_kquant_prepared_input_matvec_candidate(int type) {
    return bn_backend_quant_is_q4k(type);
}

static inline int bn_backend_quant_deinterleaved_kquant_prepared_input_matvec_candidate(int type) {
    return bn_backend_quant_is_q5k(type);
}

static inline int bn_backend_quant_native_quant_prepared_input_matvec_candidate(int type) {
    return bn_backend_quant_is_q8_0(type);
}

static inline int bn_backend_quant_native_quant_warp_matvec_candidate(int type) {
    return bn_backend_quant_is_q8_0(type);
}

static inline int bn_backend_quant_asymmetric_kquant_dot_matmul_candidate(int type) {
    return bn_backend_quant_is_q4k(type);
}

static inline int bn_backend_quant_asymmetric_kquant_prepared_input_matmul_candidate(int type) {
    return bn_backend_quant_is_q4k(type);
}

static inline int bn_backend_quant_deinterleaved_kquant_prepared_input_matmul_candidate(int type) {
    return bn_backend_quant_is_q5k(type);
}

static inline int bn_backend_quant_down_kquant_dot_matmul_candidate(int type) {
    return bn_backend_quant_moe_down_uses_down_kquant(type);
}

static inline int bn_backend_quant_native_quant_matmul_candidate(int type) {
    return bn_backend_quant_is_q8_0(type);
}

static inline int bn_backend_quant_legacy_block_matmul_candidate(int type) {
    return bn_quant_format_is_q5_0(type);
}

static inline int bn_backend_quant_legacy_block_fused_gateup_candidate(
    int type) {
    return bn_backend_quant_legacy_block_matmul_candidate(type);
}

static inline int bn_backend_quant_native_quant_fused_gateup_candidate(
    int type) {
    return bn_backend_quant_native_quant_matmul_candidate(type);
}

static inline int bn_backend_quant_asymmetric_kquant_dot_fused_gateup_candidate(
    int type) {
    return bn_backend_quant_asymmetric_kquant_dot_matvec_candidate(type);
}

static inline int bn_backend_quant_asymmetric_kquant_prepared_input_fused_gateup_candidate(
    int type) {
    return bn_backend_quant_asymmetric_kquant_prepared_input_matvec_candidate(type);
}

static inline int bn_backend_quant_deinterleaved_kquant_prepared_input_fused_gateup_candidate(
    int type) {
    return bn_backend_quant_deinterleaved_kquant_prepared_input_matvec_candidate(type);
}

static inline int bn_backend_quant_matvec_allows_fused_bias(int type) {
    return !bn_backend_quant_native_quant_warp_matvec_candidate(type);
}

static inline int bn_backend_quant_split_allows_fused_bias(int type) {
    return bn_backend_quant_matvec_allows_fused_bias(type);
}

static inline int bn_backend_quant_asymmetric_kquant_dot_split_candidate(int type) {
    return bn_backend_quant_asymmetric_kquant_dot_matvec_candidate(type);
}

static inline int bn_backend_quant_asymmetric_kquant_prepared_input_split_candidate(int type) {
    return bn_backend_quant_asymmetric_kquant_prepared_input_matvec_candidate(type);
}

static inline int bn_backend_quant_deinterleaved_kquant_prepared_input_split_candidate(int type) {
    return bn_backend_quant_deinterleaved_kquant_prepared_input_matvec_candidate(type);
}

static inline int bn_backend_quant_native_quant_split_candidate(int type) {
    return bn_backend_quant_native_quant_warp_matvec_candidate(type);
}

static inline int bn_backend_quant_split_value_4warp_dot_candidate(
    int type) {
    return bn_backend_quant_asymmetric_kquant_prepared_input_matvec_candidate(type);
}

static inline int bn_backend_quant_split_value_mmvq_candidate(
    int type) {
    return bn_backend_quant_down_kquant_warp_matvec_candidate(type);
}

static inline int bn_backend_quant_split_value_fuse_candidate(int type) {
    return bn_backend_quant_split_value_4warp_dot_candidate(type) ||
           bn_backend_quant_split_value_mmvq_candidate(type);
}

static inline int bn_backend_quant_legacy_block_pair_matmul(
    int first_type, int second_type) {
    return bn_backend_quant_legacy_block_matmul_candidate(first_type) &&
           bn_backend_quant_legacy_block_matmul_candidate(second_type);
}

static inline int bn_backend_quant_native_quant_pair_matmul(
    int first_type, int second_type) {
    return bn_backend_quant_native_quant_matmul_candidate(first_type) &&
           bn_backend_quant_native_quant_matmul_candidate(second_type);
}

static inline int bn_backend_quant_asymmetric_kquant_pair_matmul(
    int first_type, int second_type) {
    return bn_backend_quant_asymmetric_kquant_prepared_input_matmul_candidate(first_type) &&
           bn_backend_quant_asymmetric_kquant_prepared_input_matmul_candidate(second_type);
}

static inline int bn_backend_quant_deinterleaved_kquant_pair_matmul(
    int first_type, int second_type) {
    return bn_backend_quant_deinterleaved_kquant_prepared_input_matmul_candidate(first_type) &&
           bn_backend_quant_deinterleaved_kquant_prepared_input_matmul_candidate(second_type);
}

static inline int bn_backend_quant_kquant_logits_argmax_candidate(int type) {
    return bn_backend_quant_moe_down_uses_down_kquant(type);
}

static inline int bn_backend_quant_moe_route_all_active_two(int n_experts,
                                                            int k) {
    return n_experts == 2 && k == 2;
}

static inline int bn_backend_quant_moe_all_active_two_kquant_shape(int n_experts,
                                                       int k,
                                                       int down_type,
                                                       int hidden_dim,
                                                       int dim) {
    return bn_backend_quant_moe_route_all_active_two(n_experts, k) &&
           bn_backend_quant_moe_down_uses_down_kquant(down_type) &&
           hidden_dim >= 4096 && dim <= 2048;
}

static inline int bn_backend_quant_moe_all_active_two_kquant_routed_op(int op_type,
                                                           int n_experts,
                                                           int k,
                                                           int down_type,
                                                           int hidden_dim,
                                                           int dim) {
    return bn_backend_quant_is_q4k(op_type) &&
           bn_backend_quant_moe_all_active_two_kquant_shape(n_experts, k,
                                                            down_type,
                                                            hidden_dim, dim);
}

static inline int bn_backend_quant_moe_all_active_two_graph_kquant_shape(
    int n_experts,
    int k,
    int down_type,
    int hidden_dim,
    int dim) {
    return bn_backend_quant_moe_route_all_active_two(n_experts, k) &&
           bn_backend_quant_moe_down_uses_graph_kquant(down_type) &&
           hidden_dim >= 4096 && dim <= 2048;
}

static inline int bn_backend_quant_supports_kquant_logits_refine(int type) {
    return bn_quant_format_supports_q6_logits_refine(type);
}

int bn_backend_quant_cpu_tied_kquant_refine_top(void);
int bn_backend_quant_cpu_tied_kquant_hybrid_top(void);

static inline int bn_backend_quant_logits_q6_f32_cache_supported(int type) {
    return bn_quant_format_logits_q6_f32_cache_supported(type);
}

static inline int bn_backend_quant_moe_all_f16_cache_supported(int type) {
    return bn_quant_format_moe_all_f16_cache_supported(type);
}

static inline int bn_backend_quant_moe_down_q6_f32_cache_supported(int type) {
    return bn_quant_format_moe_down_q6_f32_cache_supported(type);
}

static inline int bn_backend_quant_moe_down_cublas_cache_supported(int type) {
    return bn_quant_format_moe_down_cublas_cache_supported(type);
}

static inline int bn_backend_quant_moe_down_cublas_cache_elem_bytes(
    int type, int down_kquant_f16_cache) {
    return bn_quant_format_moe_down_cublas_cache_elem_bytes(
        type, down_kquant_f16_cache);
}

static inline int bn_backend_quant_moe_down_q4_f32_cache_supported(int type) {
    return bn_quant_format_moe_down_q4_f32_cache_supported(type);
}

static inline int bn_backend_quant_moe_quant_only_after_cache(
    int type, int native_quant_f16_cache) {
    return bn_quant_format_moe_quant_only_after_cache(
        type, native_quant_f16_cache);
}

static inline int bn_backend_quant_lazy_moe_aux_cache_candidate(int type) {
    return bn_quant_format_lazy_moe_aux_cache_candidate(type);
}

static inline int bn_backend_quant_lazy_moe_aux_cache_dequant_block(
    int type, const void *blocks, size_t block_idx, float *out) {
    return bn_quant_dequant_lazy_aux_cache_block(type, blocks, block_idx, out);
}

static inline int bn_backend_quant_moe_prefers_quant_only(int type) {
    return bn_quant_format_moe_prefers_quant_only(type);
}

static inline int bn_backend_quant_aux_cache_supported(int type) {
    return bn_quant_format_aux_cache_supported(type);
}

static inline int bn_backend_quant_aux_cache_can_use_f16(int type) {
    return bn_quant_format_aux_cache_can_use_f16(type);
}

static inline int bn_backend_quant_aux_cache_uses_f32(
    int type,
    int down_kquant_f16_cache) {
    return bn_quant_format_aux_cache_uses_f32(type, down_kquant_f16_cache);
}

static inline int bn_backend_quant_aux_cache_prefers_large_budget(int type) {
    return bn_quant_format_aux_cache_prefers_large_budget(type);
}

static inline int bn_backend_quant_aux_cache_force_q4_f32(int type,
                                                          int force_f32) {
    return force_f32 && bn_backend_quant_moe_down_uses_asymmetric_kquant(type);
}

static inline int bn_backend_quant_aux_cache_q6_can_use_f16(
    int type, int force_f16, int force_q6_f32) {
    return bn_backend_quant_moe_down_uses_down_kquant(type) &&
           (force_f16 || !force_q6_f32);
}

static inline int bn_backend_quant_aux_cache_add_q6_f32(
    int type, int force_f16) {
    return force_f16 && bn_backend_quant_moe_down_uses_down_kquant(type);
}

static inline int bn_backend_quant_aux_cache_f32_storage(
    int type, int force_q4_f32, int down_kquant_f16_cache) {
    return force_q4_f32 ||
           bn_backend_quant_aux_cache_uses_f32(type, down_kquant_f16_cache);
}

typedef enum BnBackendQuantAuxCacheDequant {
    BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_NONE = 0,
    BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_BF16_TO_F16,
    BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q8_0_TO_F16,
    BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q5_0_TO_F16,
    BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q3K_TO_F16,
    BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q4K_TO_F32,
    BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q4K_TO_F16,
    BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q5K_TO_F16,
    BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q6K_TO_F16,
    BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q6K_TO_F32,
} BnBackendQuantAuxCacheDequant;

static inline BnBackendQuantAuxCacheDequant
bn_backend_quant_aux_cache_dequant_route(int type,
                                         int force_q4_f32,
                                         int down_kquant_f16_cache) {
    if (bn_backend_quant_is_bf16(type))
        return BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_BF16_TO_F16;
    if (bn_backend_quant_is_q8_0(type))
        return BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q8_0_TO_F16;
    if (bn_backend_quant_is_q5_0(type))
        return BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q5_0_TO_F16;
    if (bn_backend_quant_is_q3k(type))
        return BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q3K_TO_F16;
    if (bn_backend_quant_is_q4k(type))
        return force_q4_f32
            ? BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q4K_TO_F32
            : BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q4K_TO_F16;
    if (bn_backend_quant_is_q5k(type))
        return BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q5K_TO_F16;
    if (bn_backend_quant_moe_down_uses_down_kquant(type))
        return down_kquant_f16_cache
            ? BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q6K_TO_F16
            : BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_Q6K_TO_F32;
    return BN_BACKEND_QUANT_AUX_CACHE_DEQUANT_NONE;
}

static inline int bn_backend_quant_eager_aux_cache_supported(int type) {
    return bn_quant_format_eager_aux_cache_supported(type);
}

static inline int bn_backend_quant_metal_q4_q8_matvec_supported(int type) {
    return bn_quant_format_metal_q4_q8_matvec_supported(type);
}

static inline int bn_backend_quant_metal_q6_q8k_matvec_supported(int type) {
    return bn_quant_format_metal_q6_q8k_matvec_supported(type);
}

static inline int bn_backend_quant_gpu_matvec_supported(int type) {
    return bn_quant_format_gpu_matvec_supported(type);
}

static inline int bn_backend_quant_avoids_quant_matmul_on_f16_input(
    int type) {
    return bn_quant_format_avoids_quant_matmul_on_f16_input(type);
}

static inline int bn_backend_quant_force_q4k_quant_matmul_candidate(
    int type) {
    return bn_backend_quant_is_q4k(type) &&
           bn_quant_format_force_quant_matmul_candidate(type);
}

static inline int bn_backend_quant_force_q6k_quant_matmul_candidate(
    int type) {
    return bn_backend_quant_moe_down_uses_down_kquant(type) &&
           bn_quant_format_force_quant_matmul_candidate(type);
}

static inline uint32_t bn_backend_quant_gpu_fused_gateup_silu_cap(int type) {
    return bn_quant_format_gpu_fused_gateup_silu_cap(type);
}

static inline int bn_backend_quant_gpu_fused_gateup_requires_backend_opt_in(
    int type) {
    return bn_quant_format_gpu_fused_gateup_requires_backend_opt_in(type);
}

static inline int bn_backend_quant_can_gpu_gateup_split_activation(int type,
                                                                  int act_type) {
    return bn_quant_format_gpu_allows_gateup_split_activation(type, act_type);
}

static inline uint32_t bn_backend_quant_gpu_matvec_kquant_dot_flag(
    int type,
    int enabled) {
    return bn_quant_format_gpu_matvec_q8k_dot_flag(type, enabled);
}

static inline uint32_t bn_backend_quant_gpu_matvec_exact_kquant_flag(
    int type,
    int enabled) {
    return bn_quant_format_gpu_matvec_exact_q6k_flag(type, enabled);
}

void bn_backend_quant_matvec_gpu(float *out, const BnQWeight *W,
                                 const float *x, int8_t *x_q_buf,
                                 BnThreadPool *pool, BnGPUBackend *gpu);
void bn_backend_quant_matvec_gpu_buf(float *out, const BnQWeight *W,
                                     void *W_buf, const float *x,
                                     int8_t *x_q_buf, BnThreadPool *pool,
                                     BnGPUBackend *gpu);
void bn_backend_quant_matvec_gpu_buf_prepared(float *out, const BnQWeight *W,
                                              const BnPreparedWeight *prepared,
                                              void *W_buf, const float *x,
                                              int8_t *x_q_buf,
                                              BnThreadPool *pool,
                                              BnGPUBackend *gpu);
void bn_backend_quant_matvec_batch_gpu(const BnMatvecTask *tasks, int n_tasks,
                                       const float *x, int8_t *x_q_buf,
                                       BnThreadPool *pool, BnGPUBackend *gpu);
void bn_backend_quant_matvec_batch_gpu_buf(const BnMatvecTask *tasks,
                                           const void *const *W_bufs,
                                           int n_tasks, const float *x,
                                           int8_t *x_q_buf,
                                           BnThreadPool *pool,
                                           BnGPUBackend *gpu);
void bn_backend_quant_matmul_gpu(float *out, const BnQWeight *W,
                                 const float *X, int n_tokens,
                                 int8_t *x_q_buf, BnThreadPool *pool,
                                 BnGPUBackend *gpu);
void bn_backend_quant_matmul_gpu_buf(float *out, const BnQWeight *W,
                                     void *W_buf, const float *X,
                                     int n_tokens, int8_t *x_q_buf,
                                     BnThreadPool *pool, BnGPUBackend *gpu);
void bn_backend_quant_matmul_batch_gpu_buf(const BnMatvecTask *tasks,
                                           const void *const *W_bufs,
                                           int n_tasks, const float *X,
                                           int n_tokens, int x_cols,
                                           int8_t *x_q_buf,
                                           BnThreadPool *pool,
                                           BnGPUBackend *gpu);

#endif // BN_BACKEND_QUANT_H
