/* exllamav3 MIT; see third_party/exllamav3/LICENSE. */
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include "exl3_kernel_map.cuh"   // EXL3_GEMM_BASE_THREADS, SMEM_MAX

#define MOE_ACT_SILU 0
#define MOE_ACT_GELU 1
#define MOE_ACT_RELU2_NOGATE 2  // non-gated relu2 (NemotronH): gate GEMM and staging skipped

#define MOE_SMS_PER_EXPERT 8       // default/minimum group width, also sets max concurrency (buffer count)
#define MOE_MAX_SMS_PER_EXPERT 32  // widest expert group when few experts are active
#define MOE_TILESIZE_K 32
#define MOE_TILESIZE_M 16          // default row tile; the 32 / 64 row tiles are separate kernel instances
#define MOE_SH_STAGES 3
#define MOE_FRAG_STAGES 3

#define BN_EXL3_MOE_ARGS(POINTER_DECL)            \
    const half* __restrict__ hidden_state,      \
    half* __restrict__ temp_state_g,            \
    half* __restrict__ temp_state_u,            \
    half* __restrict__ temp_intermediate_g,     \
    half* __restrict__ temp_intermediate_u,     \
    float* __restrict__ output_state,           \
                                                \
    POINTER_DECL,                              \
    const int* __restrict__ expert_count,   \
    const int* __restrict__ token_sorted,   \
    const half* __restrict__ weight_sorted,     \
                                                \
    const int hidden_dim,                       \
    const int intermediate_dim,                 \
    const int num_experts,                      \
    const int num_experts_per_tok,              \
    const int max_tokens_per_expert,            \
    const int concurrency,                      \
    const float act_limit,                      \
    const int act_function,                     \
    const int K_gate,                           \
    const int K_up,                             \
    const int K_down,                           \
                                                \
    int* __restrict__ locks,                    \
                                                \
    float* __restrict__ output_scratch,         \
    const int* __restrict__ fused_base,     \
                                                \
    const int count_lo,                         \
    const int count_hi,                        \
    const int input_tokens,                     \
    const int slot_rows

#define EXL3_MOE_KERNEL_ARGS BN_EXL3_MOE_ARGS(const BnCudaFusedEXL3Pointers pointers)
