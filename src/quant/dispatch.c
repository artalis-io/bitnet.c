#include "quant_ctx.h"
#include "quant_dispatch_internal.h"
#include "quant_kernels_scalar.h"
#include "quant_kernels_neon.h"
#include "quant_kernels_avx512.h"
#include "quant_kernels_avx2.h"
#include "quant_kernels_wasm.h"
#include "threadpool.h"
#include "gguf.h"
#include <stdlib.h>
#include <string.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX512F__
#undef __AVX512BW__
#undef __AVX512VNNI__
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

// Max VLA elements for stack-allocated scale arrays
#define BN_MAX_SCALE_BLOCKS 8192

// Max tasks in a single batch dispatch (supports MoE K=8 gate+up = 16)
#define BN_MAX_BATCH 24

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
static int bn_quant_use_avx512_q5k_vnni(int rows) {
    const char *v = getenv("BN_AVX512_Q5K_VNNI");
    if (v)
        return v[0] != '\0' && v[0] != '0';
    return rows >= 4096;
}
#endif

#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__wasm_relaxed_simd__)
static void quant_x_to_q8_blocks_scalar(const float *x, int8_t *x_q,
                                        float *x_scales, int n) {
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + b * 32;
        int8_t *xqb = x_q + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float ax = xb[i] < 0.0f ? -xb[i] : xb[i];
            if (ax > amax) amax = ax;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        x_scales[b] = d;
        for (int i = 0; i < 32; i++) {
            float v = xb[i] * id;
            int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            xqb[i] = (int8_t)q;
        }
    }
}
#endif

void bn_quant_x_to_q8k_scalar(const float *x, int8_t *x_q, float *x_d,
                              int16_t *x_bsums, int n) {
    int n_bpr = n / BN_QK_K;
    for (int b = 0; b < n_bpr; b++) {
        const float *xb = x + b * BN_QK_K;
        int8_t *xqb = x_q + b * BN_QK_K;
        float amax = 0.0f;
        for (int i = 0; i < BN_QK_K; i++) {
            float ax = xb[i] < 0.0f ? -xb[i] : xb[i];
            if (ax > amax) amax = ax;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        x_d[b] = d;
        for (int g = 0; g < 16; g++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) {
                float v = xb[g * 16 + i] * id;
                int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                xqb[g * 16 + i] = (int8_t)q;
                sum += q;
            }
            x_bsums[b * 16 + g] = (int16_t)sum;
        }
    }
}

// --- Quantized matrix-vector multiply ---
// out[rows] = W[rows x cols] @ x[cols]

void bn_quant_matvec_impl(float *out, const BnQWeight *W, const float *x,
                          int8_t *x_q_buf, BnThreadPool *pool,
                          const BnPreparedWeight *prepared) {

    if (W->type == BN_GGUF_TENSOR_I2_S) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        BnI2SCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        BnTPTask task = { bn_quant_i2s_neon_sdot_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__ARM_NEON)
        (void)x_q_buf;
        BnI2SFloatCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_i2s_neon_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__AVX2__)
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        BnI2SCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        // Use 4-row kernel for better bandwidth utilization on DDR4
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_i2s_avx2_4row_range, &ctx, n_groups };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__wasm_relaxed_simd__)
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        BnI2SCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        BnTPTask task = { bn_quant_i2s_wasm_sdot_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        BnI2SFloatCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_i2s_wasm_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#else
        (void)x_q_buf;
        BnI2SFloatCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_i2s_scalar_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#endif
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q8_0) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ8SdotCtx ctx = { out, W, x_q_buf, x_scales, prepared };
        BnTPTask task = { bn_quant_q8_neon_sdot_range, &ctx, W->rows };
#elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ8SdotCtx ctx = { out, W, x_q_buf, x_scales, prepared };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q8_avx512_vnni_4row_range, &ctx, n_groups };
#elif defined(__AVX2__)
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ8SdotCtx ctx = { out, W, x_q_buf, x_scales, prepared };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q8_avx2_4row_range, &ctx, n_groups };
#elif defined(__wasm_relaxed_simd__)
        int n_blocks = W->cols / 32;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ8SdotCtx ctx = { out, W, x_q_buf, x_scales, prepared };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q8_wasm_sdot_4row_range, &ctx, n_groups };
#else
#ifdef __ARM_NEON
        (void)x_q_buf;
        BnQ8Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q8_neon_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        BnQ8Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q8_wasm_range, &ctx, W->rows };
#else
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        quant_x_to_q8_blocks_scalar(x, x_q_buf, x_scales, W->cols);
        BnQ8SdotCtx ctx = { out, W, x_q_buf, x_scales, prepared };
        BnTPTask task = { bn_quant_q8_scalar_sdot_range, &ctx, W->rows };
#endif
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q4_0) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ4SdotCtx ctx = { out, W, x_q_buf, x_scales, prepared };
        void (*fn)(void *, int, int) = (prepared && prepared->scales)
            ? bn_quant_q4_repacked_neon_sdot_range
            : bn_quant_q4_neon_sdot_range;
        BnTPTask task = { fn, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ4SdotCtx ctx = { out, W, x_q_buf, x_scales, prepared };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q4_avx512_vnni_4row_range, &ctx, n_groups };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__AVX2__)
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ4SdotCtx ctx = { out, W, x_q_buf, x_scales, prepared };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q4_avx2_4row_range, &ctx, n_groups };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__wasm_relaxed_simd__)
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ4SdotCtx ctx = { out, W, x_q_buf, x_scales, prepared };
        if (getenv("BN_WASM_Q4_CANONICAL4")) {
            int n_groups = (W->rows + 3) / 4;
            BnTPTask task = { bn_quant_q4_wasm_sdot_4row_range, &ctx, n_groups };
            bn_tp_dispatch(pool, &task, 1);
        } else if (prepared && prepared->qs) {
            int n_groups = (W->rows + 7) / 8;
            BnTPTask task = { bn_quant_q4_repacked_wasm_sdot_8row_range, &ctx, n_groups };
            bn_tp_dispatch(pool, &task, 1);
        } else {
            BnTPTask task = { bn_quant_q4_wasm_sdot_range, &ctx, W->rows };
            bn_tp_dispatch(pool, &task, 1);
        }
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        BnQ4Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q4_wasm_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#else
        (void)x_q_buf;
        BnQ4Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q4_scalar_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#endif
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q6_K) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        int n_sb = W->cols / BN_QK_K;
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) return;
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, W->cols);
        BnQ6KSdotCtx ctx = { out, W, x_q_buf, q8k_d, q8k_bsums };
        BnTPTask task = { bn_quant_q6k_neon_sdot_range, &ctx, W->rows };
#elif defined(__ARM_NEON)
        (void)x_q_buf;
        BnQ6KCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q6k_neon_range, &ctx, W->rows };
#elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        int n_sb_q6k = W->cols / BN_QK_K;
        if (n_sb_q6k < 1 || n_sb_q6k > BN_MAX_SCALE_BLOCKS / 8) return;
        float q6k_d[n_sb_q6k];
        int16_t q6k_bsums[n_sb_q6k * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q6k_d, q6k_bsums, W->cols);
        BnKQuantSdotCtx ctx = { out, W, x_q_buf, q6k_d, q6k_bsums };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q6k_avx512_vnni_4row_range, &ctx, n_groups };
#elif defined(__AVX2__)
        int n_sb_q6k = W->cols / BN_QK_K;
        if (n_sb_q6k < 1 || n_sb_q6k > BN_MAX_SCALE_BLOCKS / 8) return;
        float q6k_d[n_sb_q6k];
        int16_t q6k_bsums[n_sb_q6k * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q6k_d, q6k_bsums, W->cols);
        BnKQuantSdotCtx ctx = { out, W, x_q_buf, q6k_d, q6k_bsums };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q6k_avx2_4row_range, &ctx, n_groups };
#elif defined(__wasm_relaxed_simd__)
        int n_sb_q6k_w = W->cols / BN_QK_K;
        if (n_sb_q6k_w < 1 || n_sb_q6k_w > BN_MAX_SCALE_BLOCKS / 8) return;
        float q6k_d_w[n_sb_q6k_w];
        int16_t q6k_bsums_w[n_sb_q6k_w * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q6k_d_w, q6k_bsums_w, W->cols);
        BnKQuantSdotCtx ctx = { out, W, x_q_buf, q6k_d_w, q6k_bsums_w };
        BnTPTask task = { bn_quant_q6k_wasm_sdot_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        BnQ6KCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q6k_wasm_range, &ctx, W->rows };
#else
        int n_sb_q6k_s = W->cols / BN_QK_K;
        if (n_sb_q6k_s < 1 || n_sb_q6k_s > BN_MAX_SCALE_BLOCKS / 8) return;
        float q6k_d_s[n_sb_q6k_s];
        int16_t q6k_bsums_s[n_sb_q6k_s * 16];
        bn_quant_x_to_q8k_scalar(x, x_q_buf, q6k_d_s, q6k_bsums_s, W->cols);
        BnKQuantSdotCtx ctx = { out, W, x_q_buf, q6k_d_s, q6k_bsums_s };
        BnTPTask task = { bn_quant_q6k_scalar_sdot_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q8_K) {
        (void)x_q_buf;
        BnQ8KCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_q8k_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_q8k_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_q8k_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_q8k_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q4_K) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        int n_sb = W->cols / BN_QK_K;
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) return;
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, W->cols);
        BnQ4KSdotCtx ctx = { out, W, x_q_buf, q8k_d, q8k_bsums };
        BnTPTask task = { bn_quant_q4k_neon_sdot_range, &ctx, W->rows };
#elif defined(__ARM_NEON)
        (void)x_q_buf;
        BnQ4KCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q4k_neon_range, &ctx, W->rows };
#elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        int n_sb_q4k = W->cols / BN_QK_K;
        if (n_sb_q4k < 1 || n_sb_q4k > BN_MAX_SCALE_BLOCKS / 8) return;
        float q4k_d[n_sb_q4k];
        int16_t q4k_bsums[n_sb_q4k * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q4k_d, q4k_bsums, W->cols);
        BnKQuantSdotCtx ctx = { out, W, x_q_buf, q4k_d, q4k_bsums };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q4k_avx512_vnni_4row_range, &ctx, n_groups };
#elif defined(__AVX2__)
        int n_sb_q4k = W->cols / BN_QK_K;
        if (n_sb_q4k < 1 || n_sb_q4k > BN_MAX_SCALE_BLOCKS / 8) return;
        float q4k_d[n_sb_q4k];
        int16_t q4k_bsums[n_sb_q4k * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q4k_d, q4k_bsums, W->cols);
        BnKQuantSdotCtx ctx = { out, W, x_q_buf, q4k_d, q4k_bsums };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q4k_avx2_4row_range, &ctx, n_groups };
#elif defined(__wasm_relaxed_simd__)
        int n_sb_q4k_w = W->cols / BN_QK_K;
        if (n_sb_q4k_w < 1 || n_sb_q4k_w > BN_MAX_SCALE_BLOCKS / 8) return;
        float q4k_d_w[n_sb_q4k_w];
        int16_t q4k_bsums_w[n_sb_q4k_w * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q4k_d_w, q4k_bsums_w, W->cols);
        BnKQuantSdotCtx ctx = { out, W, x_q_buf, q4k_d_w, q4k_bsums_w };
        BnTPTask task = { bn_quant_q4k_wasm_sdot_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        BnQ4KCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q4k_wasm_range, &ctx, W->rows };
#else
        int n_sb_q4k_s = W->cols / BN_QK_K;
        if (n_sb_q4k_s < 1 || n_sb_q4k_s > BN_MAX_SCALE_BLOCKS / 8) return;
        float q4k_d_s[n_sb_q4k_s];
        int16_t q4k_bsums_s[n_sb_q4k_s * 16];
        bn_quant_x_to_q8k_scalar(x, x_q_buf, q4k_d_s, q4k_bsums_s, W->cols);
        BnKQuantSdotCtx ctx = { out, W, x_q_buf, q4k_d_s, q4k_bsums_s };
        BnTPTask task = { bn_quant_q4k_scalar_sdot_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q5_K) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        int n_sb = W->cols / BN_QK_K;
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) return;
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, W->cols);
        BnQ5KSdotCtx ctx = { out, W, x_q_buf, q8k_d, q8k_bsums };
        BnTPTask task = { bn_quant_q5k_neon_sdot_range, &ctx, W->rows };
#else
#if defined(__ARM_NEON)
        (void)x_q_buf;
        BnQ5KCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q5k_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        if (bn_quant_use_avx512_q5k_vnni(W->rows)) {
            int n_sb = W->cols / BN_QK_K;
            if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) return;
            float q8k_d[n_sb];
            int16_t q8k_bsums[n_sb * 16];
            bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, W->cols);
            BnQ5KSdotCtx ctx = { out, W, x_q_buf, q8k_d, q8k_bsums };
            int n_groups = (W->rows + 3) / 4;
            BnTPTask task = { bn_quant_q5k_avx512_vnni_4row_range, &ctx, n_groups };
            bn_tp_dispatch(pool, &task, 1);
            return;
        }
#endif
        (void)x_q_buf;
        BnQ5KCtx ctx = { out, W, x };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q5k_avx2_4row_range, &ctx, n_groups };
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        BnQ5KCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q5k_wasm_range, &ctx, W->rows };
#else
        int n_sb = W->cols / BN_QK_K;
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) return;
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_x_to_q8k_scalar(x, x_q_buf, q8k_d, q8k_bsums, W->cols);
        BnQ5KSdotCtx ctx = { out, W, x_q_buf, q8k_d, q8k_bsums };
        BnTPTask task = { bn_quant_q5k_scalar_sdot_range, &ctx, W->rows };
#endif
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q2_K) {
        (void)x_q_buf;
        BnQ2KCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_q2k_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_q2k_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_q2k_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_q2k_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q3_K) {
        (void)x_q_buf;
        BnQ3KCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_q3k_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_q3k_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_q3k_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_q3k_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q4_1) {
        (void)x_q_buf;
        BnQ4_1Ctx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_q4_1_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_q4_1_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_q4_1_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_q4_1_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_BF16) {
        (void)x_q_buf;
        BnBF16Ctx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_bf16_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_bf16_avx2_4row_range, &ctx, n_groups };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_bf16_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_bf16_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_F32) {
        (void)x_q_buf;
        BnF32Ctx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_f32_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_f32_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_f32_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_f32_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_F16) {
        (void)x_q_buf;
        BnF16Ctx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_f16_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_f16_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_f16_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_f16_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_IQ4_NL) {
        (void)x_q_buf;
        BnIQ4NLCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_iq4nl_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_iq4nl_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_iq4nl_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_iq4nl_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_IQ4_XS) {
        (void)x_q_buf;
        BnIQ4XSCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_iq4xs_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_iq4xs_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_iq4xs_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_iq4xs_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_IQ3_XXS) {
        (void)x_q_buf;
        BnIQ3XXSCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_iq3xxs_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_iq3xxs_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_iq3xxs_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_iq3xxs_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_IQ3_S) {
        (void)x_q_buf;
        BnIQ3SCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_iq3s_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_iq3s_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_iq3s_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_iq3s_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_IQ2_XXS) {
        (void)x_q_buf;
        BnIQ2XXSCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_iq2xxs_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_iq2xxs_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_iq2xxs_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_iq2xxs_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_IQ2_XS) {
        (void)x_q_buf;
        BnIQ2XSCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_iq2xs_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_iq2xs_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_iq2xs_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_iq2xs_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_IQ2_S) {
        (void)x_q_buf;
        BnIQ2SCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_iq2s_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_iq2s_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_iq2s_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_iq2s_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_TQ2_0) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        BnTQ2SdotCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        BnTPTask task = { bn_quant_tq2_neon_sdot_range, &ctx, W->rows };
#elif defined(__AVX2__)
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        BnTQ2SdotCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        BnTPTask task = { bn_quant_tq2_avx2_range, &ctx, W->rows };
#elif defined(__ARM_NEON)
        (void)x_q_buf;
        BnTQ2Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_tq2_neon_range, &ctx, W->rows };
#elif defined(__wasm_relaxed_simd__)
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        BnTQ2SdotCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        BnTPTask task = { bn_quant_tq2_wasm_sdot_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        BnTQ2Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_tq2_wasm_range, &ctx, W->rows };
#else
        (void)x_q_buf;
        BnTQ2Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_tq2_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    // TQ1_0
    {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        BnTQ1SdotCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        BnTPTask task = { bn_quant_tq1_neon_sdot_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__ARM_NEON)
        (void)x_q_buf;
        BnTQ1Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_tq1_neon_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__AVX2__)
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        BnTQ1SdotCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        BnTPTask task = { bn_quant_tq1_avx2_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__wasm_relaxed_simd__)
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, W->cols);
        BnTQ1SdotCtx ctx = { out, W, x_q_buf, W->scale * x_scale };
        BnTPTask task = { bn_quant_tq1_wasm_sdot_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        BnTQ1Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_tq1_wasm_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#else
        (void)x_q_buf;
        BnTQ1Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_tq1_scalar_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#endif
    }
}

void bn_quant_matvec(float *out, const BnQWeight *W, const float *x,
                     int8_t *x_q_buf, BnThreadPool *pool) {
    bn_quant_matvec_impl(out, W, x, x_q_buf, pool, NULL);
}

// --- Data size computation ---

size_t bn_qweight_data_size(const BnQWeight *w) {
    if (!w || !w->data) return 0;
    return bn_quant_format_data_size(w->type, w->rows, w->cols);
}
