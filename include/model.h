#ifndef MODEL_H
#define MODEL_H

#include "platform.h"
#include "gguf.h"
#include "quant.h"

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads;
    int vocab_size, seq_len;
    float rope_theta, norm_eps;
    int head_size, kv_dim, kv_mul;  // derived
    int has_ffn_gate, act_type;     // 0=SiLU, 1=ReLU²
} Config;

typedef struct {
    float *attn_norm, *attn_sub_norm;       // RMSNorm weights [dim]
    QWeight wq, wk, wv, wo;                 // ternary attention weights
    float *ffn_norm, *ffn_sub_norm;         // RMSNorm weights
    QWeight ffn_gate, ffn_up, ffn_down;     // ternary FFN weights
} LayerWeights;

typedef struct {
    const void *token_embedding;  // raw F16 data (dequant on demand)
    int emb_type;                 // tensor type (F16, Q6_K, etc.)
    float *output_norm;           // [dim]
    LayerWeights *layers;         // [n_layers]
} Weights;

typedef struct {
    float *x, *xb, *xb2;         // [dim] activation buffers
    float *hb, *hb2;             // [hidden_dim]
    float *q;                     // [dim] query buffer
    float *att;                   // [n_heads * seq_len] attention scores
    float *logits;                // [vocab_size]
    float *key_cache;             // [n_layers * seq_len * kv_dim]
    float *value_cache;           // [n_layers * seq_len * kv_dim]
} RunState;

typedef struct {
    Config config;
    Weights weights;
    RunState state;
    MappedFile file;  // keeps mmap/buffer alive
} Model;

int  model_load(Model *m, GGUFFile *f, int max_seq_len);
void model_free(Model *m);
void model_embed_token(const Model *m, float *out, int token);

#endif // MODEL_H
