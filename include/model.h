#ifndef BN_MODEL_H
#define BN_MODEL_H

#include "platform.h"
#include "gguf.h"
#include "quant.h"
#include "threadpool.h"
#include "sh_arena.h"

#define BN_DEFAULT_ROPE_THETA  10000.0f
#define BN_DEFAULT_NORM_EPS    1e-5f

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads;
    int vocab_size, seq_len;
    float rope_theta, norm_eps;
    int head_size, kv_dim, kv_mul;  // derived
    int has_ffn_gate, act_type;     // 0=SiLU, 1=ReLU²
    int flash_attn;                 // use flash attention (online softmax)
} BnConfig;

typedef struct {
    float *attn_norm, *attn_sub_norm;       // RMSNorm weights [dim]
    BnQWeight wq, wk, wv, wo;                 // ternary attention weights
    float *ffn_norm, *ffn_sub_norm;         // RMSNorm weights
    BnQWeight ffn_gate, ffn_up, ffn_down;     // ternary FFN weights
} BnLayerWeights;

typedef struct {
    const void *token_embedding;  // raw F16 data (dequant on demand)
    int emb_type;                 // tensor type (F16, Q6_K, etc.)
    int8_t *emb_out_i8;          // [vocab_size * dim] INT8 copy for logits (NULL if unused)
    float  *emb_out_scales;      // [vocab_size] per-row scales (NULL if unused)
    float *output_norm;           // [dim]
    BnLayerWeights *layers;         // [n_layers]
} BnWeights;

typedef struct {
    float *x, *xb, *xb2;         // [dim] activation buffers
    float *hb, *hb2;             // [hidden_dim]
    float *q;                     // [dim] query buffer
    float *att;                   // [n_heads * seq_len] attention scores
    float *logits;                // [vocab_size]
    float *key_cache;             // [n_layers * seq_len * kv_dim]
    float *value_cache;           // [n_layers * seq_len * kv_dim]
    int8_t *x_q;                  // [max(dim, hidden_dim)] scratch for int8 quantized x
    float *rope_freq;             // [head_size/2] precomputed RoPE frequencies
} BnRunState;

typedef struct {
    BnConfig config;
    BnWeights weights;
    BnRunState state;
    BnMappedFile file;  // keeps mmap/buffer alive
    BnThreadPool *pool; // thread pool for parallel dispatch
    SHArena *arena;     // arena for all RunState buffers
} BnModel;

int  bn_model_load(BnModel *m, BnGGUFFile *f, int max_seq_len);
void bn_model_free(BnModel *m);
void bn_model_embed_token(const BnModel *m, float *out, int token);

#endif // BN_MODEL_H
