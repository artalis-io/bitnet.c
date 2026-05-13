#ifndef BN_MODEL_CONFIG_H
#define BN_MODEL_CONFIG_H

#include <stdint.h>

#define BN_DEFAULT_ROPE_THETA  10000.0f
#define BN_DEFAULT_NORM_EPS    1e-5f

#define BN_MODEL_ARCH_FLAG_GEMMA4 1u
#define BN_MODEL_ARCH_FLAG_LARGE_GPU_GRAPH_FALLBACK 2u
#define BN_MODEL_ARCH_FLAG_QWEN 4u
#define BN_MODEL_ARCH_FLAG_BITNET 8u

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads;
    int vocab_size, seq_len;
    float rope_theta, norm_eps;
    int head_size, kv_dim, kv_mul;  // derived
    int has_ffn_gate, act_type;     // 0=SiLU, 1=ReLU²
    uint32_t arch_flags;            // BnModelArchOps flags for planner/backend constraints
    int qk_norm_per_head;           // 1 = per-head separate norms [dim], 0 = shared [head_size]
    int flash_attn;                 // use flash attention (online softmax)
    int kv_f16;                     // store KV cache in FP16 (halves attention DRAM bandwidth)
    // Hybrid SSM + Attention (all zero = pure attention, backward compatible)
    int rope_dim_count;             // partial RoPE dim count (0 = full head_size)
    int rope_text_dims;             // MROPE: dims for text section only (0 = use rope_dim_count)
    int full_attn_interval;         // 0 = all attention, N = every Nth layer is attention
    int ssm_state_size;             // head_k_dim (128)
    int ssm_conv_kernel;            // conv kernel size (4)
    int ssm_inner_size;             // value_dim = num_v_heads * head_v_dim (4096)
    int ssm_time_step_rank;         // num_v_heads (32)
    int ssm_group_count;            // num_k_heads (16)
    // MoE config (all zero = dense FFN, backward compatible)
    int n_experts;              // total experts per layer (e.g. 256 for Qwen3.5-35B)
    int n_experts_active;       // top-K active per token (e.g. 8)
    int moe_intermediate_size;  // per-expert hidden dim
    int has_shared_expert;      // 1 if shared expert exists
    int shared_expert_intermediate_size; // shared expert hidden dim
    // TurboQuant KV compression (0=disabled, 2-4 = bits)
    int kv_tq_bits;
} BnConfig;

#endif // BN_MODEL_CONFIG_H
