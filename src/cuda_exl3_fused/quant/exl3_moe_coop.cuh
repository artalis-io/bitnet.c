/* exllamav3 bd1f5191 MIT; plain private launch parameters, no ATen. */
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Fused decode-shaped MoE path (bsz <= MAX_BSZN): two launches per layer run every (token, expert)
// slot through gate/up, activation and down, replacing the per-projection mgemm CUDA graphs of
// BC_BlockSparseMLP. See exl3_moe_coop_kernel.cuh / exl3_moe_coop.cu.

#define MOE_COOP_ACT_SILU 0
#define MOE_COOP_ACT_GELU 1
#define MOE_COOP_ACT_RELU2 2
#define MOE_COOP_ACT_SILU_OAI 3

#define MOE_COOP_THREADS 512
#define MOE_COOP_WNT 2                  // adjacent 16-column tiles per warp (32-column groups per block)
#define MOE_COOP_COLS (MOE_COOP_WNT * 16)

struct MoeCoopParams
{
    // Inputs
    const half* x;              // (bsz, H), row stride x_stride
    int x_stride;
    const int64_t* sel;         // (bsz, topk) global expert indices
    const half* rw;             // (bsz, topk) routing weights
    int bsz;
    int topk;
    int H;                      // x width (the experts' input width before padding)
    int Hi;                     // gate/up in_features, padded (multiple of 128)
    int I;                      // intermediate width, padded (multiple of 128)
    int Ho;                     // down out_features, padded (multiple of 128)
    int H_out;                  // output width (<= Ho)
    int min_expert;             // -1: no expert-range filtering; else sel in [min, max) is local
    int max_expert;

    // Per-expert pointer tables (int64 device arrays indexed by local expert)
    const int64_t* g_trellis;   const int64_t* g_suh;   const int64_t* g_svh;
    const int64_t* u_trellis;   const int64_t* u_suh;   const int64_t* u_svh;
    const int64_t* d_trellis;   const int64_t* d_suh;   const int64_t* d_svh;
    const int64_t* g_bias;      // nullable
    const int64_t* u_bias;
    const int64_t* d_bias;
    int act;                    // MOE_COOP_ACT_*
    float act_limit;
    bool gated;                 // false: no gate projection, activation is relu(u) * u

    // Scratch (leading dim >= bsz * topk)
    half* had_g;                // (slots, Hi) rotated gate input (bsz > 1, written by the rot kernel)
    half* had_u;                // (slots, Hi) rotated up input
    bool a_global;              // true: the GEMV reads had_g / had_u; false (bsz 1): rotated in-block
    void* gu_g;                 // (slots, I) gate GEMV output, half or float (gu_f32)
    void* gu_u;                 // (slots, I)
    bool gu_f32;
    half* act_out;              // (slots, I) activation, rotated for the down projection
    float* d_out;               // (slots, Ho) down GEMV output before the output rotation
    int* ctr_a;                 // (slots_max, I / 128) completion counters of the gate/up stage
    int* ctr_b;                 // (MAX_BSZN, Ho / 128) completion counters of the down stage
    int ctr_a_len;
    int ctr_b_len;
    int ksplit_a;               // split-k blocks per chunk per stage (launcher; partial rows at slot + q * slots)
    int ksplit_b;
    int dbg;                    // diagnostics (EXL3_MOE_COOP_DBG): 1 skip A epilogue, 2 skip B reduction
    int* runs;                  // run table (bsz > 1): [n_runs, -, run_start[slots_max + 1], order[slots_max]]
    int slots_max;
    int rows_max;               // token rows the output scratch holds
    int n_local;                // entries in the pointer tables
    int sh_gate_n;              // shared gate weight length (checked against H per call)

    // Output
    float* out;                 // (bsz, H_out), row stride out_stride
    int out_stride;

    // Shared expert (nullable): out += gate * sh_out, gate = sigmoid(x . sh_gate_w) or 1
    const float* sh_out;        // (bsz, H), row stride H
    const half* sh_gate_w;      // (H,) or null
};

