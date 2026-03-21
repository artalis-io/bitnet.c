#ifndef BN_QUANT_H
#define BN_QUANT_H

#include <stdint.h>
#include "threadpool.h"

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

// Q4_1: 4-bit quantization with min, 32 elements per block
// FP16 scale + FP16 min + 16 packed nibble bytes = 20 bytes
typedef struct {
    uint16_t d;       // FP16 scale (delta)
    uint16_t m;       // FP16 min
    uint8_t  qs[16];  // packed nibbles (2 values per byte)
} BnBlockQ4_1;

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

// Quantized weight tensor descriptor (zero-copy into GGUF buffer)
typedef struct {
    const void *data;   // packed weight data
    int type;           // BN_GGUF_TENSOR_TQ1_0, TQ2_0, or I2_S
    int rows, cols;
    float scale;        // per-tensor scale (from .scale tensor or embedded in data)
    float *rp_scales;   // repacked: pre-converted FP32 per-block scales (NULL if not repacked)
    uint8_t *rp_qs;     // repacked: contiguous quant data (NULL if not repacked)
} BnQWeight;

float    bn_fp16_to_fp32(uint16_t h);
uint16_t bn_fp32_to_fp16(float f);
void     bn_quant_dequant_tq1(const BnBlockTQ1 *block, float *out);
void     bn_quant_dequant_tq2(const BnBlockTQ2 *block, float *out);
void     bn_quant_dequant_q8_0(const BnBlockQ8_0 *block, float *out);
void     bn_quant_dequant_q4_0(const BnBlockQ4_0 *block, float *out);
void     bn_quant_dequant_q4_1(const BnBlockQ4_1 *block, float *out);
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
void     bn_quant_matvec(float *out, const BnQWeight *W, const float *x,
                         int8_t *x_q_buf, BnThreadPool *pool);

// Batch matvec: run multiple independent matvecs with a single dispatch
typedef struct {
    float *out;
    const BnQWeight *W;
} BnMatvecTask;

void bn_quant_matvec_batch(const BnMatvecTask *tasks, int n_tasks,
                           const float *x, int8_t *x_q_buf, BnThreadPool *pool);

// Get platform-optimal kernel for float-x quant types (K-quants, BF16, IQ*, Q4_1, Q8_K).
// Returns NULL if the type requires int8 quantized x (I2_S, Q4_0, Q8_0, TQ1, TQ2).
bn_tp_fn bn_quant_get_float_kernel(int type);

// Quantize float vector to int8, returns scale = amax/127.
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__) || defined(__wasm_relaxed_simd__)
float bn_quant_x_to_i8(const float *x, int8_t *x_q, int n);

// Quantize float vector to per-block Q8_0: 32-element blocks with per-block scales.
void bn_quant_x_to_q8_blocks(const float *x, int8_t *x_q, float *x_scales, int n);
void bn_quant_x_to_q8k(const float *x, int8_t *x_q, float *x_d,
                         int16_t *x_bsums, int n);

#if !defined(__wasm_relaxed_simd__)
// Quantize F16 rows to INT8 + per-row scales for INT8 logits kernel.
void bn_quant_f16_rows_to_i8(const uint16_t *f16, int8_t *i8_out,
                              float *scales_out, int n_rows, int dim);
#endif
#endif

#endif // BN_QUANT_H
