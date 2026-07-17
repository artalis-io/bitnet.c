#ifndef BN_QUANT_H
#define BN_QUANT_H

#include <stdint.h>
#include <stddef.h>
#include "gguf.h"
#include "threadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BN_QK_K 256

// FP16 bit field constants
#define BN_FP16_SIGN_MASK    0x8000
#define BN_FP16_EXP_MASK     0x1F
#define BN_FP16_MANT_MASK    0x03FF
#define BN_FP16_HIDDEN_BIT   0x0400
#define BN_FP16_INF          0x7C00
#define BN_FP16_EXP_REBIAS   112        // FP32 bias (127) - FP16 bias (15)
#define BN_FP32_EXP_INF      0x7F800000
#define BN_FP32_MANT_MASK    0x007FFFFF

#define BN_I8_MAX             127
#define BN_I2S_BLOCK_ELEMS   128         // elements per I2S interleaved block
#define BN_I2S_SUBROW_SIZE   32          // elements per sub-row
#define BN_QUANT_GPU_MATVEC_FLAG_Q8K_DOT 1u
#define BN_QUANT_GPU_MATVEC_FLAG_EXACT_Q6K 8u

// TQ1_0: base-3 ternary packing, 256 weights per block
// qs packs 240 values (5 per byte in base-3), qh packs remaining 16 (4 per byte)
// Total: 48 + 4 + 2 = 54 bytes per 256-element block (1.6875 bpw)
typedef struct {
    uint8_t  qs[(BN_QK_K - 4 * BN_QK_K / 64) / 5];  // 48 bytes: (256-16)/5
    uint8_t  qh[BN_QK_K / 64];                    // 4 bytes
    uint16_t d;                                  // FP16 scale
} BnBlockTQ1;

// TQ2_0: 2-bit ternary packing, 256 weights per block
// 64 bytes qs (4 weights per byte), 2 bytes scale per block
typedef struct {
    uint8_t  qs[BN_QK_K / 4];  // 64 bytes
    uint16_t d;              // FP16 scale
} BnBlockTQ2;

// Q8_0: 8-bit quantization, 32 elements per block
// FP16 per-block scale + 32 int8 quantized values = 34 bytes
typedef struct {
    uint16_t d;       // FP16 scale
    int8_t   qs[32];  // quantized values
} BnBlockQ8_0;

// Q4_0: 4-bit quantization, 32 elements per block
// FP16 per-block scale + 16 packed nibble bytes = 18 bytes
typedef struct {
    uint16_t d;       // FP16 scale
    uint8_t  qs[16];  // packed nibbles (2 values per byte)
} BnBlockQ4_0;

// MXFP4: 32 E2M1 values sharing one E8M0 power-of-two scale.
typedef struct {
    uint8_t e;
    uint8_t qs[16];
} BnBlockMXFP4;

// Q5_0: 5-bit quantization, 32 elements per block
// FP16 per-block scale + 4 high-bit bytes + 16 packed nibble bytes = 22 bytes
typedef struct {
    uint16_t d;
    uint8_t  qh[4];
    uint8_t  qs[16];
} BnBlockQ5_0;

// Q4_1: 4-bit quantization with min, 32 elements per block
// FP16 scale + FP16 min + 16 packed nibble bytes = 20 bytes
typedef struct {
    uint16_t d;       // FP16 scale (delta)
    uint16_t m;       // FP16 min
    uint8_t  qs[16];  // packed nibbles (2 values per byte)
} BnBlockQ4_1;

// Q5_1: 5-bit quantization with min, 32 elements per block
// FP16 scale + FP16 min + 4 high-bit bytes + 16 packed nibble bytes = 24 bytes
typedef struct {
    uint16_t d;
    uint16_t m;
    uint8_t  qh[4];
    uint8_t  qs[16];
} BnBlockQ5_1;

// Q2_K: 2-bit k-quant, 256 elements per block
// 16 bytes scales + 64 bytes qs + 2 bytes d + 2 bytes dmin = 84 bytes
typedef struct {
    uint8_t  scales[BN_QK_K / 16]; // 16 bytes: 4-bit scale + 4-bit min per 16-element group
    uint8_t  qs[BN_QK_K / 4];     // 64 bytes: 2-bit quants (4 per byte)
    uint16_t d;                     //  2 bytes: FP16 super-block scale
    uint16_t dmin;                  //  2 bytes: FP16 super-block min
} BnBlockQ2K;                       // 84 bytes total

// Q3_K: 3-bit k-quant, 256 elements per block
// 32 bytes hmask + 64 bytes qs + 12 bytes scales + 2 bytes d = 110 bytes
typedef struct {
    uint8_t  hmask[BN_QK_K / 8]; // 32 bytes: high bit of each 3-bit quant
    uint8_t  qs[BN_QK_K / 4];   // 64 bytes: low 2 bits of each quant
    uint8_t  scales[12];         // 12 bytes: 16 packed 6-bit scales
    uint16_t d;                  //  2 bytes: FP16 super-block scale
} BnBlockQ3K;                    // 110 bytes total

// Q4_K: 4-bit k-quant, 256 elements per block
// 2 bytes d + 2 bytes dmin + 12 bytes scales + 128 bytes qs = 144 bytes
typedef struct {
    uint16_t d;                  //  2 bytes: FP16 super-block scale
    uint16_t dmin;               //  2 bytes: FP16 super-block min
    uint8_t  scales[12];         // 12 bytes: 8 packed 6-bit scales + 8 packed 6-bit mins
    uint8_t  qs[BN_QK_K / 2];   // 128 bytes: 4-bit quants (0-15)
} BnBlockQ4K;                    // 144 bytes total

// Runtime Q4_K layout interleaving 8 output rows for CPU prompt matmul.
// This mirrors llama.cpp's Q4_Kx8 repack and is not a GGUF disk format.
typedef struct {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[96];
    uint8_t qs[1024];
} BnBlockQ4Kx8;

// Q5_K: 5-bit k-quant, 256 elements per block
// 2 bytes d + 2 bytes dmin + 12 bytes scales + 32 bytes qh + 128 bytes qs = 176 bytes
typedef struct {
    uint16_t d;                  //  2 bytes: FP16 super-block scale
    uint16_t dmin;               //  2 bytes: FP16 super-block min
    uint8_t  scales[12];         // 12 bytes: same packing as Q4_K
    uint8_t  qh[BN_QK_K / 8];   // 32 bytes: high bit of each 5-bit quant
    uint8_t  qs[BN_QK_K / 2];   // 128 bytes: low 4 bits of each quant
} BnBlockQ5K;                    // 176 bytes total

// Q6_K: 6-bit k-quant, 256 elements per block
// 128 bytes ql (lower 4 bits) + 64 bytes qh (upper 2 bits) + 16 int8 scales + FP16 d = 210 bytes
typedef struct {
    uint8_t ql[BN_QK_K / 2];    // 128 bytes: lower 4 bits of quants
    uint8_t qh[BN_QK_K / 4];    //  64 bytes: upper 2 bits of quants
    int8_t  scales[BN_QK_K / 16]; // 16 bytes: 8-bit sub-block scales
    uint16_t d;                  //   2 bytes: FP16 super-block scale
} BnBlockQ6K;                    // 210 bytes total

typedef struct {
    float d;
    int8_t scales[BN_QK_K / 16];
    uint8_t qs[BN_QK_K];
} BnBlockQ6KPrepared;

// Q8_K: 8-bit k-quant, 256 elements per block
// 4 bytes d (float32!) + 256 bytes qs + 32 bytes bsums = 292 bytes
typedef struct {
    float    d;                        //  4 bytes: float32 scale (NOT FP16!)
    int8_t   qs[BN_QK_K];             // 256 bytes: signed int8 quants
    int16_t  bsums[BN_QK_K / 16];     //  32 bytes: sum of quants in groups of 16
} BnBlockQ8K;                          // 292 bytes total

// IQ4_NL: 4-bit non-linear quantization, 32 elements per block
// Same layout as Q4_0 but values are indices into a 16-entry codebook
typedef struct {
    uint16_t d;       // FP16 scale
    uint8_t  qs[16];  // packed 4-bit codebook indices
} BnBlockIQ4NL;       // 18 bytes (same as Q4_0)

// IQ4_XS: 4-bit non-linear with sub-block scales, 256 elements per block
typedef struct {
    uint16_t d;                     //  2 bytes: FP16 super-block scale
    uint16_t scales_h;              //  2 bytes: high 2 bits of 8 6-bit scales
    uint8_t  scales_l[BN_QK_K/64]; //  4 bytes: low 4 bits of 8 scales (nibble-packed)
    uint8_t  qs[BN_QK_K / 2];      // 128 bytes: packed 4-bit IQ4_NL indices
} BnBlockIQ4XS;                     // 136 bytes total

// IQ3_XXS: 3-bit codebook quantization, 256 elements per block
typedef struct {
    uint16_t d;                         //  2 bytes: FP16 super-block scale
    uint8_t  qs[3 * BN_QK_K / 8];      // 96 bytes: grid indices + packed signs/scales
} BnBlockIQ3XXS;                        // 98 bytes total

// IQ3_S: 3-bit codebook with separate signs/scales, 256 elements per block
typedef struct {
    uint16_t d;                     //  2 bytes: FP16 super-block scale
    uint8_t  qs[BN_QK_K / 4];      // 64 bytes: 8-bit grid indices
    uint8_t  qh[BN_QK_K / 32];     //  8 bytes: high bits for 9-bit grid indices
    uint8_t  signs[BN_QK_K / 8];   // 32 bytes: sign bits (1 per element)
    uint8_t  scales[BN_QK_K / 32]; //  8 bytes: 4-bit sub-block scales (nibble-packed)
} BnBlockIQ3S;                      // 114 bytes total

// IQ2_XXS: 2-bit codebook quantization, 256 elements per block
typedef struct {
    uint16_t d;                     //  2 bytes: FP16 super-block scale
    uint16_t qs[BN_QK_K / 8];      // 64 bytes: packed grid indices + signs/scales
} BnBlockIQ2XXS;                    // 66 bytes total

// IQ2_XS: 2-bit codebook with explicit scales, 256 elements per block
typedef struct {
    uint16_t d;                     //  2 bytes: FP16 super-block scale
    uint16_t qs[BN_QK_K / 8];      // 64 bytes: packed grid indices + signs
    uint8_t  scales[BN_QK_K / 32]; //  8 bytes: 4-bit sub-block scales
} BnBlockIQ2XS;                     // 74 bytes total

// IQ2_S: 2-bit codebook (1024-entry grid), 256 elements per block
typedef struct {
    uint16_t d;                     //  2 bytes: FP16 super-block scale
    uint8_t  qs[BN_QK_K / 4];      // 64 bytes: 8-bit grid indices
    uint8_t  qh[BN_QK_K / 32];     //  8 bytes: high bits for grid indices
    uint8_t  scales[BN_QK_K / 32]; //  8 bytes: 4-bit sub-block scales
} BnBlockIQ2S;                      // 82 bytes total

// I2_S: Microsoft BitNet 2-bit ternary, no per-block scale
// Interleaved byte layout: each byte packs 4 values from 4 sub-rows of 32
// Single per-tensor scale stored at offset nelements/4 in the data
// Encoding: 0=-1, 1=0, 2=+1

typedef struct SHArena SHArena;

typedef enum {
    BN_PREPARED_WEIGHT_NONE = 0,
    BN_PREPARED_WEIGHT_Q4_0_REPACK = 1,
    BN_PREPARED_WEIGHT_Q8_0_F32_SCALES = 2,
    BN_PREPARED_WEIGHT_Q4_K_SCALES = 3,
    BN_PREPARED_WEIGHT_Q6_K_EXPANDED = 4,
} BnPreparedWeightKind;

// Backend/runtime-prepared layout for a quantized weight tensor.
typedef struct BnPreparedWeight {
    uint16_t *scales;
    uint8_t *qs;
    uint8_t *aux;
    float *f32_scales;
    size_t aux_size;
    BnPreparedWeightKind kind;
} BnPreparedWeight;

// Quantized weight tensor descriptor (zero-copy into GGUF buffer)
typedef struct BnQWeight {
    const void *data;   // packed weight data
    int type;           // GGUF tensor type
    int rows, cols;
    float scale;        // per-tensor scale (from .scale tensor or embedded in data)
} BnQWeight;

static inline int bn_quant_format_has_embedded_tensor_scale(int type) {
    return type == BN_GGUF_TENSOR_I2_S;
}

static inline size_t bn_quant_embedded_tensor_scale_offset(int type,
                                                          int rows,
                                                          int cols) {
    if (!bn_quant_format_has_embedded_tensor_scale(type))
        return 0;
    return (size_t)rows * (size_t)cols / 4;
}

static inline int bn_quant_format_allows_stacked_layout(int type) {
    return !bn_quant_format_has_embedded_tensor_scale(type);
}

typedef enum {
    BN_QUANT_LAYOUT_DENSE = 0,
    BN_QUANT_LAYOUT_BLOCK32 = 1,
    BN_QUANT_LAYOUT_BLOCK256 = 2,
    BN_QUANT_LAYOUT_I2S = 3,
} BnQuantLayout;

typedef enum {
    BN_QUANT_CAP_LOADABLE       = 1u << 0,
    BN_QUANT_CAP_EMBEDDED_SCALE = 1u << 1,
    BN_QUANT_CAP_CPU_MATVEC     = 1u << 4,
    BN_QUANT_CAP_CPU_BATCH      = 1u << 5,
    BN_QUANT_CAP_CPU_MATMUL     = 1u << 6,
    BN_QUANT_CAP_CPU_PREQ8K     = 1u << 7,
    BN_QUANT_CAP_CPU_REPACKED   = 1u << 8,
    BN_QUANT_CAP_GPU_SMALL_DENSE = 1u << 9,
    BN_QUANT_CAP_GPU_SMALL_DENSE_Q8 = 1u << 10,
    BN_QUANT_CAP_FLOAT_KQUANT_FALLBACK = 1u << 11,
    BN_QUANT_CAP_Q8_LOGITS_REFINE = 1u << 12,
    BN_QUANT_CAP_Q6_LOGITS_REFINE = 1u << 13,
} BnQuantCapability;

typedef void (*BnQuantMatvecFn)(float *out, const BnQWeight *W, const float *x,
                                int8_t *x_q_buf, BnThreadPool *pool);
typedef void (*BnQuantMatmulFn)(float *out, const BnQWeight *W, const float *X,
                                int n_tokens, int8_t *x_q_buf,
                                BnThreadPool *pool);

typedef struct {
    int type;
    const char *name;
    BnQuantLayout layout;
    int block_elems;
    size_t bytes_per_block;
    uint32_t caps;
    BnQuantMatvecFn matvec;
    BnQuantMatmulFn matmul;
} BnQuantFormatOps;

const BnQuantFormatOps *bn_quant_format_ops(int type);
int      bn_quant_format_has_cap(int type, uint32_t cap);
int      bn_quant_format_supported(int type);
int      bn_quant_format_uses_embedded_scale(int type);
int      bn_quant_format_has_cpu_matvec(int type);
int      bn_quant_format_has_cpu_batch(int type);
int      bn_quant_format_has_cpu_matmul(int type);
int      bn_quant_format_can_preq8k(int type);
int      bn_quant_format_can_cpu_repack(int type);
int      bn_quant_format_can_gpu_native(int type);
int      bn_quant_format_can_gpu_repack(int type);
int      bn_quant_format_supports_gpu_small_dense(int type);
int      bn_quant_format_supports_gpu_small_dense_q8(int type);
int      bn_quant_format_is_float_kquant_fallback_candidate(int type);
int      bn_quant_format_supports_q8_logits_refine(int type);
int      bn_quant_format_supports_q6_logits_refine(int type);
int      bn_quant_q8_logits_refine_row(const BnQWeight *W,
                                        const int8_t *x_q,
                                        const float *x_scales,
                                        int row,
                                        float *out);
int      bn_quant_q6_logits_refine_row(const BnQWeight *W,
                                        const float *x,
                                        int row,
                                        float *out);
int      bn_quant_q6_logits_refine_q8k_row(const BnQWeight *W,
                                            const int8_t *x_q,
                                            const float *x_d,
                                            const int16_t *x_bsums,
                                            int row,
                                            float *out);
uint32_t bn_quant_format_gpu_split_cap(int type);
int      bn_quant_format_can_gpu_split(int type);
int      bn_quant_format_gpu_requires_exact_silu(int type);
int      bn_quant_format_gpu_prefers_gateup_split(int type);
uint32_t bn_quant_format_gpu_fused_gateup_silu_cap(int type);
int      bn_quant_format_gpu_fused_gateup_requires_cuda_opt_in(int type);
int      bn_quant_format_gpu_allows_gateup_split_activation(int type,
                                                            int act_type);
uint32_t bn_quant_format_gpu_matvec_q8k_dot_flag(int type, int enabled);
uint32_t bn_quant_format_gpu_matvec_exact_q6k_flag(int type, int enabled);
int      bn_quant_format_dense_f32_type(void);
int      bn_quant_format_gpu_float_buffer_type(void);
int      bn_quant_format_is_f32(int type);
int      bn_quant_format_can_convert_dense_to_f32(int type);
int      bn_quant_format_convert_dense_to_f32(int type, const void *src,
                                              float *dst, int n);
int      bn_quant_format_cuda_logits_q6_f32_cache_supported(int type);
int      bn_quant_format_cuda_moe_all_f16_cache_supported(int type);
int      bn_quant_format_cuda_moe_down_q6_f32_cache_supported(int type);
int      bn_quant_format_cuda_moe_down_cublas_cache_supported(int type);
int      bn_quant_format_cuda_moe_down_cublas_cache_elem_bytes(int type,
                                                               int q6_as_f16);
int      bn_quant_format_cuda_moe_down_q4_f32_cache_supported(int type);
int      bn_quant_format_cuda_moe_quant_only_after_cache(int type,
                                                         int q8_f16_cache);
int      bn_quant_format_cuda_lazy_moe_aux_cache_candidate(int type);
int      bn_quant_format_cuda_moe_prefers_quant_only(int type);
int      bn_quant_format_cuda_aux_cache_supported(int type);
int      bn_quant_format_cuda_aux_cache_can_use_f16(int type);
int      bn_quant_format_cuda_aux_cache_uses_f32(int type, int q6_as_f16);
int      bn_quant_format_cuda_aux_cache_prefers_large_budget(int type);
int      bn_quant_policy_cuda_matvec_type_disabled(int type);
int      bn_quant_format_uses_f16_logits_path(int type);
int      bn_quant_format_tied_logits_uses_quant_path(int type);
int      bn_quant_format_supports_logits_i8_cache(int type);
int      bn_quant_format_tied_logits_uses_f16_path(int type);
int      bn_quant_format_tied_logits_i8_weight_type(void);
int      bn_quant_format_tied_logits_f16_weight_type(void);
int      bn_quant_format_tied_logits_f32_weight_type(void);
int      bn_quant_format_supports_moe_q4_down_route(int gate_type,
                                                     int up_type,
                                                     int down_type,
                                                     int allow_q4_down);
int      bn_quant_format_supports_moe_q4_gateup(int gate_type, int up_type);
int      bn_quant_format_supports_cpu_fused_q4_gateup_silu(int gate_type,
                                                           int up_type);
int      bn_quant_format_pair_same_format(int left_type, int right_type);
int      bn_quant_format_supports_moe_q8_route(int gate_type,
                                               int up_type,
                                               int down_type);
BnQuantMatvecFn bn_quant_format_matvec(int type);
BnQuantMatmulFn bn_quant_format_matmul(int type);
size_t   bn_quant_format_data_size(int type, int rows, int cols);

// Compute raw data size in bytes for a quantized weight tensor.
size_t bn_qweight_data_size(const BnQWeight *w);

float    bn_fp16_to_fp32(uint16_t h);
uint16_t bn_fp32_to_fp16(float f);
void     bn_quant_dequant_tq1(const BnBlockTQ1 *block, float *out);
void     bn_quant_dequant_tq2(const BnBlockTQ2 *block, float *out);
void     bn_quant_dequant_q8_0(const BnBlockQ8_0 *block, float *out);
void     bn_quant_dequant_q4_0(const BnBlockQ4_0 *block, float *out);
void     bn_quant_dequant_q5_0(const BnBlockQ5_0 *block, float *out);
void     bn_quant_dequant_q4_1(const BnBlockQ4_1 *block, float *out);
void     bn_quant_dequant_q5_1(const BnBlockQ5_1 *block, float *out);
void     bn_quant_dequant_q2k(const BnBlockQ2K *block, float *out);
void     bn_quant_dequant_q3k(const BnBlockQ3K *block, float *out);
void     bn_quant_dequant_q4k(const BnBlockQ4K *block, float *out);
void     bn_quant_dequant_q5k(const BnBlockQ5K *block, float *out);
void     bn_quant_dequant_q6k(const BnBlockQ6K *block, float *out);
void     bn_quant_dequant_q8k(const BnBlockQ8K *block, float *out);
void     bn_quant_dequant_iq4nl(const BnBlockIQ4NL *block, float *out);
void     bn_quant_dequant_iq4xs(const BnBlockIQ4XS *block, float *out);
void     bn_quant_dequant_iq3xxs(const BnBlockIQ3XXS *block, float *out);
void     bn_quant_dequant_iq3s(const BnBlockIQ3S *block, float *out);
void     bn_quant_dequant_iq2xxs(const BnBlockIQ2XXS *block, float *out);
void     bn_quant_dequant_iq2xs(const BnBlockIQ2XS *block, float *out);
void     bn_quant_dequant_iq2s(const BnBlockIQ2S *block, float *out);
float    bn_bf16_to_fp32(uint16_t h);
void     bn_quant_dequant_i2s(const uint8_t *data, float *out, int n, float scale);
int      bn_quant_dequant_row(int type, const void *data, int row, int n,
                              float *out);
void     bn_quant_matvec(float *out, const BnQWeight *W, const float *x,
                         int8_t *x_q_buf, BnThreadPool *pool);
void     bn_quant_matvec_prepared(float *out, const BnQWeight *W,
                                  const BnPreparedWeight *prepared,
                                  const float *x, int8_t *x_q_buf,
                                  BnThreadPool *pool);
void     bn_quant_matvec_prepared_flags(float *out, const BnQWeight *W,
                                        const BnPreparedWeight *prepared,
                                        const float *x, int8_t *x_q_buf,
                                        BnThreadPool *pool, uint32_t flags);

// Batch matvec: run multiple independent matvecs with a single dispatch
#define BN_MATVEC_TASK_FORCE_FLOAT_KQUANT 1u
#define BN_MATVEC_TASK_REFERENCE_DOT      2u
#define BN_MATVEC_TASK_LLAMA_DOT          BN_MATVEC_TASK_REFERENCE_DOT
#define BN_MATVEC_TASK_NATIVE_QUANT       4u

typedef struct {
    float *out;
    const BnQWeight *W;
    const BnPreparedWeight *prepared;
    uint32_t flags;
} BnMatvecTask;

void bn_quant_matvec_batch(const BnMatvecTask *tasks, int n_tasks,
                           const float *x, int8_t *x_q_buf, BnThreadPool *pool);

// Batch matvec with pre-quantized Q8_K input (skips redundant quantization).
// Used after fused rmsnorm+q8k to avoid re-quantizing the same activation.
void bn_quant_matvec_batch_preq8k(const BnMatvecTask *tasks, int n_tasks,
                                    const int8_t *x_q, const float *x_d,
                                    const int16_t *x_bsums, const float *x_float,
                                    BnThreadPool *pool);

// Multi-input matvec: K independent (W, x) pairs dispatched in a single cycle.
// Each task has its own weight matrix and input vector.
// x_q_bufs must be [n_tasks * max_cols] int8 (pre-allocated scratch).
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
    const BnPreparedWeight *prepared;
} BnMatvecMultiTask;

void bn_quant_matvec_multi(const BnMatvecMultiTask *tasks, int n_tasks,
                            int8_t *x_q_bufs, BnThreadPool *pool);

size_t bn_quant_prepared_qweight_size(const BnQWeight *W,
                                      BnPreparedWeightKind *kind);
int bn_quant_prepare_qweight(BnPreparedWeight *prepared, const BnQWeight *W,
                             SHArena *arena);

// Fused Q4_0 gate/up SiLU fast path. Returns 0 if dispatched, -1 if unsupported.
int bn_quant_q4_gate_up_silu(float *out,
                             const BnQWeight *gate,
                             const BnPreparedWeight *gate_prepared,
                             const BnQWeight *up,
                             const BnPreparedWeight *up_prepared,
                             const float *x,
                             int8_t *x_q_buf,
                             BnThreadPool *pool);

// Matrix-matrix multiply: Out[n_tokens][rows] = W[rows][cols] @ X[n_tokens][cols]^T
// Processes n_tokens input vectors against the same weight matrix.
// out must be [n_tokens * W->rows] floats, X is [n_tokens * W->cols] floats.
void bn_quant_matmul(float *out, const BnQWeight *W, const float *X,
                     int n_tokens, int8_t *x_q_buf, BnThreadPool *pool);
void bn_quant_matmul_prepared(float *out, const BnQWeight *W,
                              const BnPreparedWeight *prepared,
                              const float *X, int n_tokens,
                              int8_t *x_q_buf, BnThreadPool *pool);
void bn_quant_matmul_prepared_multi(float **out, const BnQWeight **W,
                                    const BnPreparedWeight **prepared, int n,
                                    const float *X, int n_tokens,
                                    int8_t *x_q_buf, BnThreadPool *pool);

// Matmul with pre-quantized Q8_K input (avoids redundant re-quantization).
// x_q/x_d/x_bsums must be [n_tokens * cols] / [n_tokens * n_bpr] / [n_tokens * n_bpr * 16].
// x_float is the original float input for fallback (non-k-quant types).
void bn_quant_matmul_preq8k(float *out, const BnQWeight *W, int n_tokens,
                              const int8_t *x_q, const float *x_d,
                              const int16_t *x_bsums, const float *x_float,
                              BnThreadPool *pool);

// Dispatch up to n matmuls in a single threadpool cycle.
// All weight matrices must share the same pre-quantized input.
// out/W arrays must have n entries. Falls back to sequential if types differ.
void bn_quant_matmul_preq8k_multi(float **out, const BnQWeight **W,
                                  const BnPreparedWeight **prepared, int n,
                                  int n_tokens, const int8_t *x_q,
                                  const float *x_d, const int16_t *x_bsums,
                                  const float *x_float, BnThreadPool *pool);

// Get platform-optimal kernel for float-x quant types (K-quants, BF16, IQ*, Q4_1, Q8_K).
// Returns NULL if the type requires int8 quantized x (I2_S, Q4_0, Q8_0, TQ1, TQ2).
bn_tp_fn bn_quant_get_float_kernel(int type);

// Quantize float vector to int8, returns scale = amax/127.
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || \
    defined(__AVX2__) || defined(__wasm_relaxed_simd__)
float bn_quant_x_to_i8(const float *x, int8_t *x_q, int n);

// Quantize float vector to per-block Q8_0: 32-element blocks with per-block scales.
void bn_quant_x_to_q8_blocks(const float *x, int8_t *x_q, float *x_scales, int n);
void bn_quant_x_to_q8k(const float *x, int8_t *x_q, float *x_d,
                         int16_t *x_bsums, int n);

// Quantize F16 rows to INT8 + per-row scales for INT8 logits kernel.
void bn_quant_f16_rows_to_i8(const uint16_t *f16, int8_t *i8_out,
                              float *scales_out, int n_rows, int dim);

// Fused RMSNorm + Q8_K quantization (AVX2 only). Saves 2 passes over dim.
// xb_out receives the scaled x*w (float), x_q/x_d/x_bsums receive Q8_K result.
#ifdef __AVX2__
void bn_quant_rmsnorm_q8k_avx2(const float *x, const float *w, int dim, float eps,
                                  float *xb_out, int8_t *x_q, float *x_d,
                                  int16_t *x_bsums);
#endif
#endif

// Quantize F16 rows to INT8 + per-row scales, using the native quant backend
// when available and a scalar fallback otherwise.
void bn_quant_f16_rows_to_i8_dispatch(const uint16_t *f16, int8_t *i8_out,
                                      float *scales_out, int n_rows, int dim);

#ifdef __cplusplus
}
#endif

#endif // BN_QUANT_H
