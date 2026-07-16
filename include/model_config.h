#ifndef BN_MODEL_CONFIG_H
#define BN_MODEL_CONFIG_H

#include <stdint.h>

#define BN_DEFAULT_ROPE_THETA  10000.0f
#define BN_DEFAULT_NORM_EPS    1e-5f

#define BN_MODEL_ARCH_POLICY_UNIT_ATTENTION_SCALE              (1u << 0)
#define BN_MODEL_ARCH_POLICY_LARGE_GPU_GRAPH_FALLBACK          (1u << 1)
#define BN_MODEL_ARCH_POLICY_SCALAR_HYBRID_SSM_CPU             (1u << 2)
#define BN_MODEL_ARCH_POLICY_CPU_FLOAT_KQUANT                  (1u << 3)
#define BN_MODEL_ARCH_POLICY_MOE_EXACT_SILU                    (1u << 4)
#define BN_MODEL_ARCH_POLICY_LLAMA_RMSNORM_ORDER               (1u << 5)
#define BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY        (1u << 6)
#define BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT                   (1u << 7)
#define BN_MODEL_ARCH_POLICY_ATTENTION_POST_NORM               (1u << 8)
#define BN_MODEL_ARCH_POLICY_FFN_POST_NORM                     (1u << 9)
#define BN_MODEL_ARCH_POLICY_LAYER_OUTPUT_SCALE                (1u << 10)
#define BN_MODEL_ARCH_POLICY_CPU_PREFILL_DECODE_PARITY         (1u << 11)
#define BN_MODEL_ARCH_POLICY_SMALL_CUDA_PREFILL_DECODE_FALLBACK (1u << 12)
#define BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP           (1u << 13)
#define BN_MODEL_ARCH_POLICY_MOE_CUDA_EXACT_ATTENTION          (1u << 14)
#define BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT           (1u << 15)
#define BN_MODEL_ARCH_POLICY_MOE_DENSE_RESIDUAL_BRANCH         (1u << 16)
#define BN_MODEL_ARCH_POLICY_SMALL_CUDA_DENSE_EXACT_Q4_Q8      (1u << 17)
#define BN_MODEL_ARCH_POLICY_SMALL_CUDA_Q8_LOGIT_REFINE        (1u << 18)
#define BN_MODEL_ARCH_POLICY_PREFILL_EXACT_ACTIVATION          (1u << 19)
#define BN_MODEL_ARCH_POLICY_EXACT_SCALAR_FFN_ACTIVATION       (1u << 20)

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads;
    int vocab_size, seq_len;
    float rope_theta, norm_eps;
    float rope_theta_swa;
    int head_size, kv_dim, kv_mul;  // derived
    int has_ffn_gate, act_type;     // 0=SiLU, 1=ReLU², 2=GELU
    uint32_t policy_flags;          // BnModelArchOps behavior policies for planner/backend constraints
    int qk_norm_per_head;           // 1 = per-head separate norms [dim], 0 = shared [head_size]
    int flash_attn;                 // use flash attention (online softmax)
    int kv_f16;                     // store KV cache in FP16 (halves attention DRAM bandwidth)
    // Hybrid SSM + Attention (all zero = pure attention, backward compatible)
    int rope_dim_count;             // partial RoPE dim count (0 = full head_size)
    int rope_dim_count_swa;         // sliding-window/local-attention RoPE dim count
    int rope_text_dims;             // MROPE: dims for text section only (0 = use rope_dim_count)
    int full_attn_interval;         // 0 = all attention, N = every Nth layer is attention
    int ssm_state_size;             // head_k_dim (128)
    int ssm_conv_kernel;            // conv kernel size (4)
    int ssm_inner_size;             // value_dim = num_v_heads * head_v_dim (4096)
    int ssm_time_step_rank;         // num_v_heads (32)
    int ssm_group_count;            // num_k_heads (16)
    // MoE config (all zero = dense FFN, backward compatible)
    int n_experts;              // total experts per layer
    int n_experts_active;       // top-K active per token
    int moe_intermediate_size;  // per-expert hidden dim
    int moe_norm_topk_prob;     // normalize selected expert weights to sum 1
    int moe_exact_silu;         // use exact SiLU in MoE FFN for parity-sensitive archs
    float moe_expert_weights_scale; // optional post-routing expert weight scale
    int has_shared_expert;      // 1 if shared expert exists
    int shared_expert_intermediate_size; // shared expert hidden dim
    // Shared-KV / per-layer input metadata (zero = disabled)
    int kv_unique_layer_count;  // first N layers own KV cache, later layers reuse
    int sliding_window_pattern[128];
    int per_layer_input_dim;
    float final_logit_softcap;
    // TurboQuant KV compression (0=disabled, 2-4 = bits)
    int kv_tq_bits;
} BnConfig;

#endif // BN_MODEL_CONFIG_H
