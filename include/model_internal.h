#ifndef BN_MODEL_INTERNAL_H
#define BN_MODEL_INTERNAL_H

#include "model.h"
#include "model_run_state.h"
#include "moe_types.h"
#include "platform.h"
#include "sh_arena.h"
#include <stddef.h>

typedef struct BnBackendModel BnBackendModel;
typedef struct BnThreadPool BnThreadPool;
typedef struct BnTQState BnTQState;

struct BnModelRuntime {
    BnThreadPool *pool;
    int owns_pool;
    SHArena *weight_arena;
    BnTQState *tq_state;
    int owns_tq_state;
};

struct BnModelIO {
    BnMappedFile file;
    BnMoEIO moe_io;
};

struct BnModelBackendState {
    BnBackendModel *backend;
};

BnBackendModel *bn_model_backend(const BnModel *model);
int bn_model_ensure_backend(BnModel *model);
int bn_model_dequant_qweight_row(const BnQWeight *weight,
                                 int row,
                                 int n,
                                 float *out);
void bn_model_set_file(BnModel *model, BnMappedFile file);
BnThreadPool *bn_model_pool(const BnModel *model);
void bn_model_set_thread_pool(BnModel *model, BnThreadPool *pool, int owned);
SHArena *bn_model_weight_arena(const BnModel *model);
BnTQState *bn_model_tq_state(const BnModel *model);
void bn_model_set_tq_state(BnModel *model, BnTQState *state, int owned);
int bn_model_has_tq(const BnModel *model);
BnMoEIO *bn_model_moe_io(BnModel *model);
const BnMoEIO *bn_model_moe_io_const(const BnModel *model);
void bn_model_set_moe_mmap_base(BnModel *model, const uint8_t *base);
void bn_model_set_moe_mmap_shards(BnModel *model, const uint8_t **bases,
                                  size_t n_bases);
void bn_model_set_moe_fd(BnModel *model, int fd);
void bn_model_set_moe_madvise(BnModel *model, int enabled);
void bn_model_set_moe_cache(BnModel *model, void *cache);
void *bn_model_moe_cache(const BnModel *model);
void bn_model_set_gpu_moe_cache(BnModel *model, void *cache);
void *bn_model_gpu_moe_cache(const BnModel *model);
size_t bn_model_session_arena_size(const BnConfig *config,
                                   const BnWeights *weights);
int bn_model_alloc_session_buffers(const BnConfig *config,
                                   const BnWeights *weights,
                                   SHArena *arena,
                                   BnRunState *state,
                                   BnMoEState **moe_out);

int bn_model_activation_is_relu2(int activation);
int bn_model_activation_is_gelu(int activation);
int bn_model_activation_uses_silu_path(int activation);
int bn_model_gguf_uses_moe(BnGGUFFile *file);
int bn_model_gguf_context_length(BnGGUFFile *file);
int bn_model_load_policy_uses_moe(const BnConfig *config);
int bn_model_load_policy_loads_extra_metadata(const BnConfig *config);
int bn_model_load_policy_uses_hybrid_layer_layout(const BnConfig *config);
int bn_model_load_policy_moe_total_experts(const BnConfig *config);
int bn_model_load_policy_moe_active_experts(const BnConfig *config);
int bn_model_load_policy_moe_expert_hidden_dim(const BnConfig *config);
int bn_model_load_policy_moe_route_shape_valid(const BnConfig *config);
int bn_model_load_policy_loads_per_layer_input_weights(
    const BnConfig *config);
int bn_model_load_policy_layer_reuses_kv(const BnConfig *config,
                                         int layer);
int bn_model_load_policy_kv_reuse_layer(const BnConfig *config,
                                        int layer);
int bn_model_load_policy_uses_ffn_post_norm(const BnConfig *config);
int bn_model_load_policy_loads_extra_ffn_post_norms(
    const BnConfig *config);
int bn_model_load_policy_moe_uses_scaled_router_input(
    const BnConfig *config);
int bn_model_load_policy_moe_uses_dense_residual_branch(
    const BnConfig *config);
int bn_model_load_policy_has_shared_expert(const BnConfig *config);
int bn_model_load_policy_weight_type_supported(int type);
int bn_model_load_policy_weight_uses_embedded_block_scale(int type);
int bn_model_load_policy_weight_has_embedded_tensor_scale(int type);
size_t bn_model_load_policy_weight_embedded_tensor_scale_offset(int type,
                                                                int rows,
                                                                int cols);
int bn_model_load_policy_tied_logits_uses_quant_path(int type);
int bn_model_load_policy_logits_i8_cache_supported(int type);
void bn_model_load_policy_prepare_logits_i8_cache(const uint16_t *src,
                                                  int8_t *dst,
                                                  float *scales,
                                                  int rows,
                                                  int dim);
int bn_model_load_policy_shared_expert_gate_uses_dense_float(int type);
int bn_model_load_policy_can_convert_shared_expert_gate_to_dense_float(
    int type);
int bn_model_load_policy_convert_shared_expert_gate_to_dense_float(
    int type,
    const void *src,
    float *dst,
    int n);
int bn_model_load_policy_dense_float_weight_type(void);
int bn_model_prompt_cache_attention_layer_count(const BnConfig *config);
int bn_model_prompt_cache_supports_kv_snapshot(const BnConfig *config);
int bn_model_session_policy_attention_layer_count(const BnConfig *config);
int bn_model_session_policy_ssm_layer_count(const BnConfig *config);
int bn_model_session_policy_uses_hybrid_layer_layout(
    const BnConfig *config);
int bn_model_session_policy_shared_expert_hidden_dim(
    const BnConfig *config);
int bn_model_session_policy_uses_moe(const BnConfig *config);
int bn_model_session_policy_per_layer_embedding_dim(
    const BnConfig *config);
void bn_model_session_policy_init_rope_frequencies(const BnConfig *config,
                                                   float *freqs,
                                                   int capacity_pairs);
int bn_model_embed_policy_scales_token_embedding(const BnConfig *config);
int bn_model_embed_policy_dequant_row(int type,
                                      const void *data,
                                      int row,
                                      int n,
                                      float *out);
int bn_model_moe_policy_requires_float_kquant_gateup_fallback(
    const BnConfig *config);
int bn_model_moe_policy_uses_scaled_router_input(const BnConfig *config);
int bn_model_moe_policy_uses_dense_residual_branch(const BnConfig *config);
int bn_model_moe_policy_uses_reference_silu(const BnConfig *config);
int bn_model_moe_policy_activation(const BnConfig *config);
float bn_model_moe_policy_norm_epsilon(const BnConfig *config);
int bn_model_moe_policy_prefill_requires_matvec(const BnConfig *config);
int bn_model_moe_policy_uses_grouped_expert_route(const BnConfig *config);
int bn_model_moe_policy_total_experts(const BnConfig *config);
int bn_model_moe_policy_active_experts(const BnConfig *config);
int bn_model_moe_policy_expert_hidden_dim(const BnConfig *config);
int bn_model_moe_policy_normalizes_topk_route_weights(
    const BnConfig *config);
float bn_model_moe_policy_expert_weights_scale(const BnConfig *config);
int bn_model_moe_policy_uses_expert_weights(const BnConfig *config);
int bn_model_moe_policy_uses_all_active_two_expert_set(
    const BnConfig *config);
int bn_model_moe_policy_uses_all_active_two_expert_route(
    const BnConfig *config,
    int dim);
int bn_model_moe_policy_has_shared_expert(const BnConfig *config);
int bn_model_moe_policy_shared_expert_hidden_dim(
    const BnConfig *config);
int bn_model_transformer_policy_is_attention_layer(const BnConfig *config,
                                                   int layer);
int bn_model_transformer_policy_attention_layer_index(
    const BnConfig *config,
    int layer);
int bn_model_transformer_policy_ssm_layer_index(const BnConfig *config,
                                                int layer);
int bn_model_transformer_policy_attention_layer_count(
    const BnConfig *config);
int bn_model_transformer_policy_ssm_layer_count(const BnConfig *config);
int bn_model_transformer_policy_uses_hybrid_layer_layout(
    const BnConfig *config);
int bn_model_transformer_policy_uses_hybrid_ssm(const BnConfig *config);
int bn_model_transformer_policy_uses_hybrid_moe(const BnConfig *config);
int bn_model_transformer_policy_uses_large_dense_hybrid_ssm(
    const BnConfig *config);
int bn_model_transformer_policy_uses_non_hybrid_moe(
    const BnConfig *config);
int bn_model_transformer_policy_uses_moe(const BnConfig *config);
int bn_model_transformer_policy_uses_dense_attention_only(
    const BnConfig *config);
int bn_model_transformer_policy_uses_small_dense_shape(
    const BnConfig *config);
int bn_model_transformer_policy_uses_large_dense_shape(
    const BnConfig *config);
int bn_model_transformer_policy_uses_large_gpu_graph_fallback_shape(
    const BnConfig *config);
int bn_model_transformer_policy_uses_small_dense_native_quant_shape(
    const BnConfig *config);
int bn_model_transformer_policy_allows_small_dense_native_quant(
    const BnConfig *config);
int bn_model_transformer_policy_small_dense_native_quant_to_layer(
    const BnConfig *config);
int bn_model_transformer_policy_allows_small_dense_prefill_decode_fallback(
    const BnConfig *config);
int bn_model_transformer_policy_small_dense_prefill_min_tokens(
    const BnConfig *config);
int bn_model_transformer_policy_dense_batch_prefill_shape_allowed(
    const BnConfig *config,
    int supports_large_dense_batch_prefill);
int bn_model_transformer_policy_dense_logits_argmax_shape_allowed(
    const BnConfig *config,
    int logits_rows);
int bn_model_transformer_policy_moe_logits_mmvq_argmax_shape_allowed(
    const BnConfig *config,
    int logits_cols);
int bn_model_transformer_policy_allows_small_dense_native_logit_refine(
    const BnConfig *config);
int bn_model_transformer_policy_moe_prefers_reference_gpu_attention(
    const BnConfig *config);
float bn_model_transformer_policy_norm_epsilon(const BnConfig *config);
int bn_model_transformer_policy_requires_float_kquant_fallback(
    const BnConfig *config);
int bn_model_transformer_policy_activation(const BnConfig *config);
int bn_model_transformer_policy_has_ffn_gate(const BnConfig *config);
float bn_model_transformer_policy_final_logit_softcap(
    const BnConfig *config);
int bn_model_transformer_policy_attention_flash_requested(
    const BnConfig *config);
int bn_model_transformer_policy_attention_qk_norm_stride(
    const BnConfig *config,
    int head_size);
int bn_model_transformer_policy_attention_uses_per_head_qk_norm(
    const BnConfig *config);
int bn_model_transformer_policy_prefill_uses_decode_for_parity(
    const BnConfig *config);
int bn_model_transformer_policy_rmsnorm_uses_reference_order(
    const BnConfig *config);
float bn_model_transformer_policy_attention_scale(const BnConfig *config,
                                                  int head_size);
int bn_model_transformer_policy_attention_value_shares_key(
    const BnConfig *config);
int bn_model_transformer_policy_uses_attention_post_norm(
    const BnConfig *config);
int bn_model_transformer_policy_uses_ffn_post_norm(
    const BnConfig *config);
int bn_model_transformer_policy_uses_layer_output_scale(
    const BnConfig *config);
int bn_model_transformer_policy_per_layer_embedding_dim(
    const BnConfig *config);
int bn_model_transformer_policy_uses_per_layer_embedding(
    const BnConfig *config);
int bn_model_transformer_policy_divides_rope_freqs(const BnConfig *config,
                                                   int layer);
int bn_model_transformer_policy_rope_dims_for_head(const BnConfig *config,
                                                   int layer_head_size);
float bn_model_transformer_policy_rope_theta_for_head(
    const BnConfig *config,
    int layer_head_size);
float bn_model_transformer_policy_rope_base_theta(const BnConfig *config);
int bn_model_transformer_policy_rope_uses_base_frequency(
    const BnConfig *config,
    int layer_head_size);
int bn_model_transformer_policy_uses_reference_hybrid_ssm(
    const BnConfig *config);
int bn_model_transformer_policy_prefill_uses_reference_activation(
    const BnConfig *config);
int bn_model_transformer_policy_ffn_uses_reference_activation(
    const BnConfig *config);
int bn_model_gpu_policy_attention_layer_count(const BnConfig *config);
int bn_model_gpu_policy_ssm_layer_count(const BnConfig *config);
int bn_model_gpu_policy_uses_hybrid_ssm(const BnConfig *config);
int bn_model_gpu_policy_uses_hybrid_moe(const BnConfig *config);
int bn_model_gpu_policy_uses_moe(const BnConfig *config);
int bn_model_gpu_policy_rope_dims_for_head(const BnConfig *config,
                                           int layer_head_size);
void bn_model_gpu_policy_init_rope_frequencies(const BnConfig *config,
                                               float *freqs,
                                               int capacity_pairs);

#endif // BN_MODEL_INTERNAL_H
