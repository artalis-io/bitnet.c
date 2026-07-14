#ifndef BN_MODEL_WEIGHTS_H
#define BN_MODEL_WEIGHTS_H

#include "model_config.h"
#include "quant.h"
#include "moe_types.h"
#include <stdint.h>

typedef enum {
    BN_LAYER_BLOCK_ATTENTION = 0,
    BN_LAYER_BLOCK_SSM = 1,
} BnLayerBlockKind;

typedef enum {
    BN_LAYER_FFN_DENSE = 0,
    BN_LAYER_FFN_MOE = 1,
} BnLayerFFNKind;

typedef struct {
    float *attn_norm, *attn_sub_norm;       // RMSNorm weights [dim]
    float *attn_post_norm;                  // post-attention RMSNorm weights [dim]
    float *ffn_norm, *ffn_sub_norm;         // RMSNorm weights
    float *ffn_post_norm;                   // post-FFN RMSNorm weights [dim]
    float *ffn_post_norm_1;                 // optional Gemma4 dense-branch post norm [dim]
    float *ffn_post_norm_2;                 // optional second post-FFN norm [dim]
    float *layer_output_scale;              // optional scalar [1]
} BnLayerNormWeights;

typedef struct {
    BnQWeight wq, wk, wv, wo;               // attention projection weights (NULL for SSM layers)
    int head_size, kv_dim, kv_mul, n_kv_heads, q_dim;
    int has_kv, kv_reuse_layer;
    float *q_bias, *k_bias, *v_bias;        // attention biases (NULL if not present)
    float *q_norm, *k_norm;                 // per-head Q/K RMSNorm (NULL if absent)
} BnAttentionWeights;

typedef struct {
    BnQWeight ffn_gate, ffn_up, ffn_down;   // FFN weights
} BnFFNWeights;

typedef struct {
    BnQWeight inp_gate, proj;                // Gemma4 per-layer embedding adapters
    float *post_norm;                        // [dim]
} BnPerLayerEmbeddingWeights;

typedef struct {
    BnQWeight wqkv;                         // fused QKV [dim, qkv_dim]
    BnQWeight wz;                           // Z gate projection [dim, value_dim]
    float *ssm_a;                           // [num_v_heads] F32 - A_log
    BnQWeight ssm_alpha;                    // [dim, num_v_heads] - decay projection
    BnQWeight ssm_beta;                     // [dim, num_v_heads] - update rate projection
    float *ssm_conv1d;                      // [conv_kernel, conv_dim] F32
    float *ssm_dt_bias;                     // [num_v_heads] F32
    float *ssm_norm;                        // [head_v_dim] F32
    BnQWeight ssm_out;                      // [value_dim, dim]
} BnSSMWeights;

typedef struct {
    float *router_weight;                   // [n_experts * dim] F32 - routing gate (always resident)
    float *router_scale;                    // optional [dim] Gemma4 router input scale
    float *expert_down_scale;               // optional [n_experts] Gemma4 down projection scale
    BnMoEExpertMap expert_map;              // file offsets for gate/up/down expert tensors
} BnMoEWeights;

typedef struct {
    BnQWeight shared_gate, shared_up, shared_down;
    float *shared_expert_gate;              // [dim] sigmoid gate for shared expert output (NULL if absent)
    int shared_expert_gate_type;            // GGUF tensor type for shared_expert_gate
} BnSharedExpertWeights;

typedef struct BnLayerWeights {
    BnLayerBlockKind block_kind;
    BnLayerFFNKind ffn_kind;
    BnLayerNormWeights norm;
    BnAttentionWeights attn;
    BnSSMWeights ssm;
    BnFFNWeights ffn;
    BnPerLayerEmbeddingWeights per_layer;
    BnMoEWeights moe;
    BnSharedExpertWeights shared;
} BnLayerWeights;

typedef struct {
    const void *token_embedding;  // raw embedding data (F16, Q4_0, Q8_0, etc.)
    int emb_type;                 // tensor type (F16, Q4_0, Q8_0, etc.)
    int8_t *emb_out_i8;           // [vocab_size * dim] INT8 copy for logits (NULL if unused)
    float  *emb_out_scales;       // [vocab_size] per-row scales (NULL if unused)
    BnQWeight tied_embedding_weight; // stable quant descriptor for tied logits
    BnQWeight output_weight;      // untied output projection (data=NULL if tied)
    float *output_norm;           // [dim]
    float *rope_freqs;            // optional Gemma4 full-attention RoPE factors [head_size/2]
    BnQWeight per_layer_model_proj; // optional Gemma4 per-layer input projection
    BnQWeight per_layer_token_embd; // optional Gemma4 per-layer token embedding
    float *per_layer_proj_norm;   // [gemma4_per_layer_dim]
    BnLayerWeights *layers;       // [n_layers]
} BnWeights;

#endif // BN_MODEL_WEIGHTS_H
