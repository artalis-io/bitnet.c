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

// I2_S: Microsoft BitNet 2-bit ternary, no per-block scale
// Interleaved byte layout: each byte packs 4 values from 4 sub-rows of 32
// Single per-tensor scale stored at offset nelements/4 in the data
// Encoding: 0=-1, 1=0, 2=+1

// Ternary weight tensor descriptor (zero-copy into GGUF buffer)
typedef struct {
    const void *data;   // packed weight data
    int type;           // BN_GGUF_TENSOR_TQ1_0, TQ2_0, or I2_S
    int rows, cols;
    float scale;        // per-tensor scale (from .scale tensor or embedded in data)
} BnQWeight;

float    bn_fp16_to_fp32(uint16_t h);
uint16_t bn_fp32_to_fp16(float f);
void     bn_quant_dequant_tq1(const BnBlockTQ1 *block, float *out);
void     bn_quant_dequant_tq2(const BnBlockTQ2 *block, float *out);
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

// Quantize float vector to int8, returns scale = amax/127.
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
float bn_quant_x_to_i8(const float *x, int8_t *x_q, int n);
#endif

#endif // BN_QUANT_H
