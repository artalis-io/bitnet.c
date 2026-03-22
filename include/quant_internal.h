#ifndef BN_QUANT_INTERNAL_H
#define BN_QUANT_INTERNAL_H

#include "quant.h"
#include "gguf.h"
#include <string.h>

// --- Context structs for range functions ---

// I2_S integer dot product context (NEON SDOT / AVX2 DPBUSD)
typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    float combined_scale;
} BnI2SCtx;

// I2_S float activation context (NEON float / WASM / scalar)
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnI2SFloatCtx;

// TQ2_0 context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnTQ2Ctx;

// TQ2_0 DPBUSD context (AVX2)
typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    float combined_scale;
} BnTQ2SdotCtx;

// TQ1_0 float context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnTQ1Ctx;

// TQ1_0 SDOT context
typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    float combined_scale;
} BnTQ1SdotCtx;

// Q8_0 context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnQ8Ctx;

// Q8_0 DPBUSD context (AVX2)
typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_scales;
} BnQ8SdotCtx;

// Q4_0 float context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnQ4Ctx;

// Q4_0 SDOT/DPBUSD context
typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_scales;
} BnQ4SdotCtx;

// Q6_K context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnQ6KCtx;

// K-quant SDOT context (Q8_K x quantization: 256-element super-blocks)
// Shared by Q4_K and Q6_K SDOT kernels.
typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_d;          // one scale per 256-element super-block
    const int16_t *x_bsums;   // sum per 16-element group (16 per super-block)
} BnKQuantSdotCtx;

typedef BnKQuantSdotCtx BnQ6KSdotCtx;  // backward compat

typedef BnKQuantSdotCtx BnQ4KSdotCtx;  // backward compat

// K-quant matmul context (fused: weight loaded once for all tokens)
typedef struct {
    float *out;                // [n_tokens * W->rows]
    const BnQWeight *W;
    const int8_t *x_q;        // [n_tokens * cols]
    const float *x_d;         // [n_tokens * n_bpr] (one per token per super-block)
    const int16_t *x_bsums;   // [n_tokens * n_bpr * 16]
    int n_tokens;
    int cols;                  // cached W->cols for stride computation
} BnKQuantMatmulCtx;

// Generic float-x context: { out, W, x } — shared by all float-input kernels.
// All K-quant, IQ, BF16, and Q4_1 float-x kernels use this identical layout.
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnFloatXCtx;

typedef BnFloatXCtx BnQ8KCtx;
typedef BnFloatXCtx BnQ4KCtx;
typedef BnFloatXCtx BnQ5KCtx;
typedef BnFloatXCtx BnQ4_1Ctx;
typedef BnFloatXCtx BnBF16Ctx;
typedef BnFloatXCtx BnIQ4NLCtx;
typedef BnFloatXCtx BnIQ4XSCtx;
typedef BnFloatXCtx BnIQ3XXSCtx;
typedef BnFloatXCtx BnIQ3SCtx;
typedef BnFloatXCtx BnIQ2XXSCtx;
typedef BnFloatXCtx BnIQ2XSCtx;
typedef BnFloatXCtx BnIQ2SCtx;
typedef BnFloatXCtx BnQ2KCtx;
typedef BnFloatXCtx BnQ3KCtx;

// --- Range function declarations ---

// I2_S kernels
void bn_quant_i2s_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_i2s_neon_range(void *ctx, int start, int end);
void bn_quant_i2s_avx2_range(void *ctx, int start, int end);
void bn_quant_i2s_avx2_4row_range(void *ctx, int group_start, int group_end);
void bn_quant_i2s_wasm_range(void *ctx, int start, int end);
void bn_quant_i2s_wasm_sdot_range(void *ctx, int start, int end);
void bn_quant_i2s_scalar_range(void *ctx, int start, int end);

// TQ2_0 kernels
void bn_quant_tq2_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_tq2_neon_range(void *ctx, int start, int end);
void bn_quant_tq2_avx2_range(void *ctx, int start, int end);
void bn_quant_tq2_wasm_range(void *ctx, int start, int end);
void bn_quant_tq2_scalar_range(void *ctx, int start, int end);

// TQ1_0 kernels
void bn_quant_tq1_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_tq1_neon_range(void *ctx, int start, int end);
void bn_quant_tq1_avx2_range(void *ctx, int start, int end);
void bn_quant_tq1_wasm_range(void *ctx, int start, int end);
void bn_quant_tq1_scalar_range(void *ctx, int start, int end);

// Q8_0 kernels
void bn_quant_q8_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q8_neon_range(void *ctx, int start, int end);
void bn_quant_q8_avx2_range(void *ctx, int start, int end);
void bn_quant_q8_wasm_range(void *ctx, int start, int end);
void bn_quant_q8_scalar_range(void *ctx, int start, int end);

// Q4_0 kernels
void bn_quant_q4_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q4_repacked_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q4_neon_range(void *ctx, int start, int end);
void bn_quant_q4_avx2_range(void *ctx, int start, int end);
void bn_quant_q4_avx2_4row_range(void *ctx, int group_start, int group_end);
void bn_quant_q4_wasm_range(void *ctx, int start, int end);
void bn_quant_q4_wasm_sdot_range(void *ctx, int start, int end);
void bn_quant_q4_scalar_range(void *ctx, int start, int end);

// Q6_K kernels
void bn_quant_q6k_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q6k_neon_range(void *ctx, int start, int end);
void bn_quant_q6k_avx2_range(void *ctx, int start, int end);
void bn_quant_q6k_wasm_range(void *ctx, int start, int end);
void bn_quant_q6k_scalar_range(void *ctx, int start, int end);

// Q8_K kernels
void bn_quant_q8k_neon_range(void *ctx, int start, int end);
void bn_quant_q8k_avx2_range(void *ctx, int start, int end);
void bn_quant_q8k_wasm_range(void *ctx, int start, int end);
void bn_quant_q8k_scalar_range(void *ctx, int start, int end);

// Q4_K kernels
void bn_quant_q4k_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q4k_neon_sdot_matmul_range(void *ctx, int row_start, int row_end);
void bn_quant_q6k_neon_sdot_matmul_range(void *ctx, int row_start, int row_end);
void bn_quant_q4k_avx2_sdot_range(void *ctx, int start, int end);
void bn_quant_q6k_avx2_sdot_range(void *ctx, int start, int end);
void bn_quant_q4k_neon_range(void *ctx, int start, int end);
void bn_quant_q4k_avx2_range(void *ctx, int start, int end);
void bn_quant_q4k_wasm_range(void *ctx, int start, int end);
void bn_quant_q4k_scalar_range(void *ctx, int start, int end);

// Q5_K kernels
void bn_quant_q5k_neon_range(void *ctx, int start, int end);
void bn_quant_q5k_avx2_range(void *ctx, int start, int end);
void bn_quant_q5k_wasm_range(void *ctx, int start, int end);
void bn_quant_q5k_scalar_range(void *ctx, int start, int end);

// Q4_1 kernels
void bn_quant_q4_1_neon_range(void *ctx, int start, int end);
void bn_quant_q4_1_avx2_range(void *ctx, int start, int end);
void bn_quant_q4_1_wasm_range(void *ctx, int start, int end);
void bn_quant_q4_1_scalar_range(void *ctx, int start, int end);

// BF16 weight kernels
void bn_quant_bf16_neon_range(void *ctx, int start, int end);
void bn_quant_bf16_avx2_range(void *ctx, int start, int end);
void bn_quant_bf16_wasm_range(void *ctx, int start, int end);
void bn_quant_bf16_scalar_range(void *ctx, int start, int end);

// IQ4_NL kernels
void bn_quant_iq4nl_neon_range(void *ctx, int start, int end);
void bn_quant_iq4nl_avx2_range(void *ctx, int start, int end);
void bn_quant_iq4nl_wasm_range(void *ctx, int start, int end);
void bn_quant_iq4nl_scalar_range(void *ctx, int start, int end);

// IQ4_XS kernels
void bn_quant_iq4xs_neon_range(void *ctx, int start, int end);
void bn_quant_iq4xs_avx2_range(void *ctx, int start, int end);
void bn_quant_iq4xs_wasm_range(void *ctx, int start, int end);
void bn_quant_iq4xs_scalar_range(void *ctx, int start, int end);

// IQ3_XXS kernels
void bn_quant_iq3xxs_neon_range(void *ctx, int start, int end);
void bn_quant_iq3xxs_avx2_range(void *ctx, int start, int end);
void bn_quant_iq3xxs_wasm_range(void *ctx, int start, int end);
void bn_quant_iq3xxs_scalar_range(void *ctx, int start, int end);

// IQ3_S kernels
void bn_quant_iq3s_neon_range(void *ctx, int start, int end);
void bn_quant_iq3s_avx2_range(void *ctx, int start, int end);
void bn_quant_iq3s_wasm_range(void *ctx, int start, int end);
void bn_quant_iq3s_scalar_range(void *ctx, int start, int end);

// IQ2_XXS kernels
void bn_quant_iq2xxs_neon_range(void *ctx, int start, int end);
void bn_quant_iq2xxs_avx2_range(void *ctx, int start, int end);
void bn_quant_iq2xxs_wasm_range(void *ctx, int start, int end);
void bn_quant_iq2xxs_scalar_range(void *ctx, int start, int end);

// IQ2_XS kernels
void bn_quant_iq2xs_neon_range(void *ctx, int start, int end);
void bn_quant_iq2xs_avx2_range(void *ctx, int start, int end);
void bn_quant_iq2xs_wasm_range(void *ctx, int start, int end);
void bn_quant_iq2xs_scalar_range(void *ctx, int start, int end);

// IQ2_S kernels
void bn_quant_iq2s_neon_range(void *ctx, int start, int end);
void bn_quant_iq2s_avx2_range(void *ctx, int start, int end);
void bn_quant_iq2s_wasm_range(void *ctx, int start, int end);
void bn_quant_iq2s_scalar_range(void *ctx, int start, int end);

// Q2_K kernels
void bn_quant_q2k_neon_range(void *ctx, int start, int end);
void bn_quant_q2k_avx2_range(void *ctx, int start, int end);
void bn_quant_q2k_wasm_range(void *ctx, int start, int end);
void bn_quant_q2k_scalar_range(void *ctx, int start, int end);

// Q3_K kernels
void bn_quant_q3k_neon_range(void *ctx, int start, int end);
void bn_quant_q3k_avx2_range(void *ctx, int start, int end);
void bn_quant_q3k_wasm_range(void *ctx, int start, int end);
void bn_quant_q3k_scalar_range(void *ctx, int start, int end);

// --- Inline helpers ---

// Q4_K/Q5_K shared: unpack 6-bit scale/min from 12-byte packed array
static inline void bn_q4k_get_scale_min(int j, const uint8_t *q,
                                         uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

// Q3_K: unpack 12-byte packed scales to 16 6-bit values
static inline void bn_q3k_unpack_scales(const uint8_t *scales, uint8_t *out) {
    uint32_t aux[4];
    memcpy(aux, scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0fu)         | (((tmp >> 0) & 0x03030303u) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0fu)         | (((tmp >> 2) & 0x03030303u) << 4);
    memcpy(out, aux, 16);
}

#endif // BN_QUANT_INTERNAL_H
