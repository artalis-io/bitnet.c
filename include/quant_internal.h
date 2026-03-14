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

// Q8_K context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnQ8KCtx;

// Q4_K context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnQ4KCtx;

// Q5_K context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnQ5KCtx;

// Q3_K context
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnQ3KCtx;

// --- Range function declarations ---

// I2_S kernels
void bn_quant_i2s_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_i2s_neon_range(void *ctx, int start, int end);
void bn_quant_i2s_avx2_range(void *ctx, int start, int end);
void bn_quant_i2s_wasm_range(void *ctx, int start, int end);
void bn_quant_i2s_scalar_range(void *ctx, int start, int end);

// TQ2_0 kernels
void bn_quant_tq2_neon_range(void *ctx, int start, int end);
void bn_quant_tq2_scalar_range(void *ctx, int start, int end);

// TQ1_0 kernels
void bn_quant_tq1_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_tq1_neon_range(void *ctx, int start, int end);
void bn_quant_tq1_scalar_range(void *ctx, int start, int end);

// Q8_0 kernels
void bn_quant_q8_neon_range(void *ctx, int start, int end);
void bn_quant_q8_avx2_range(void *ctx, int start, int end);
void bn_quant_q8_wasm_range(void *ctx, int start, int end);
void bn_quant_q8_scalar_range(void *ctx, int start, int end);

// Q4_0 kernels
void bn_quant_q4_neon_sdot_range(void *ctx, int start, int end);
void bn_quant_q4_neon_range(void *ctx, int start, int end);
void bn_quant_q4_avx2_range(void *ctx, int start, int end);
void bn_quant_q4_wasm_range(void *ctx, int start, int end);
void bn_quant_q4_scalar_range(void *ctx, int start, int end);

// Q6_K kernels
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
void bn_quant_q4k_neon_range(void *ctx, int start, int end);
void bn_quant_q4k_avx2_range(void *ctx, int start, int end);
void bn_quant_q4k_wasm_range(void *ctx, int start, int end);
void bn_quant_q4k_scalar_range(void *ctx, int start, int end);

// Q5_K kernels
void bn_quant_q5k_neon_range(void *ctx, int start, int end);
void bn_quant_q5k_avx2_range(void *ctx, int start, int end);
void bn_quant_q5k_wasm_range(void *ctx, int start, int end);
void bn_quant_q5k_scalar_range(void *ctx, int start, int end);

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
