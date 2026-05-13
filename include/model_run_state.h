#ifndef BN_MODEL_RUN_STATE_H
#define BN_MODEL_RUN_STATE_H

#include <stdint.h>

typedef struct {
    float *x, *xb, *xb2;          // [dim] activation buffers
    float *hb, *hb2;              // [hidden_dim]
    float *q;                     // [dim] query buffer
    float *att;                   // [n_heads * seq_len] attention scores
    float *logits;                // [vocab_size]
    float *key_cache;             // [n_attn_layers * seq_len * kv_dim]
    float *value_cache;           // [n_attn_layers * seq_len * kv_dim]
    int8_t *x_q;                  // [max(dim, hidden_dim)] scratch for int8 quantized x
    float *rope_freq;             // [head_size/2] precomputed RoPE frequencies
    // TurboQuant compressed KV cache (NULL if kv_tq_bits == 0)
    uint8_t *key_cache_tq;        // [n_attn_layers * seq_len * n_kv_heads * key_bytes]
    uint8_t *value_cache_tq;      // [n_attn_layers * seq_len * n_kv_heads * val_bytes]
    float *q_rotated;             // [n_heads * head_size] scratch for rotated queries
    // SSM state (NULL if no SSM layers)
    float *ssm_state;             // [n_ssm * num_v_heads * head_k_dim * head_v_dim]
    float *ssm_conv_state;        // [n_ssm * (conv_kernel-1) * conv_dim]
} BnRunState;

#endif // BN_MODEL_RUN_STATE_H
