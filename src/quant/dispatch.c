#include "quant_internal.h"
#include "threadpool.h"

// Max VLA elements for stack-allocated scale arrays
#define BN_MAX_SCALE_BLOCKS 8192

// Max tasks in a single batch dispatch (supports MoE K=8 gate+up = 16)
#define BN_MAX_BATCH 16

// --- Quantized matrix-vector multiply ---
// out[rows] = W[rows x cols] @ x[cols]

void bn_quant_matvec(float *out, const BnQWeight *W, const float *x,
                     int8_t *x_q_buf, BnThreadPool *pool) {

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
        BnQ8SdotCtx ctx = { out, W, x_q_buf, x_scales };
        BnTPTask task = { bn_quant_q8_neon_sdot_range, &ctx, W->rows };
#elif defined(__AVX2__)
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ8SdotCtx ctx = { out, W, x_q_buf, x_scales };
        BnTPTask task = { bn_quant_q8_avx2_range, &ctx, W->rows };
#else
        (void)x_q_buf;
#ifdef __ARM_NEON
        BnQ8Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q8_neon_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnQ8Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q8_wasm_range, &ctx, W->rows };
#else
        BnQ8Ctx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q8_scalar_range, &ctx, W->rows };
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
        BnQ4SdotCtx ctx = { out, W, x_q_buf, x_scales };
        void (*fn)(void *, int, int) = W->rp_scales
            ? bn_quant_q4_repacked_neon_sdot_range
            : bn_quant_q4_neon_sdot_range;
        BnTPTask task = { fn, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__AVX2__)
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ4SdotCtx ctx = { out, W, x_q_buf, x_scales };
        int n_groups = (W->rows + 3) / 4;
        BnTPTask task = { bn_quant_q4_avx2_4row_range, &ctx, n_groups };
        bn_tp_dispatch(pool, &task, 1);
#elif defined(__wasm_relaxed_simd__)
        int n_blocks = W->cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, W->cols);
        BnQ4SdotCtx ctx = { out, W, x_q_buf, x_scales };
        BnTPTask task = { bn_quant_q4_wasm_sdot_range, &ctx, W->rows };
        bn_tp_dispatch(pool, &task, 1);
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
#elif defined(__AVX2__)
        int n_sb_q6k = W->cols / BN_QK_K;
        if (n_sb_q6k < 1 || n_sb_q6k > BN_MAX_SCALE_BLOCKS / 8) return;
        float q6k_d[n_sb_q6k];
        int16_t q6k_bsums[n_sb_q6k * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q6k_d, q6k_bsums, W->cols);
        BnKQuantSdotCtx ctx = { out, W, x_q_buf, q6k_d, q6k_bsums };
        BnTPTask task = { bn_quant_q6k_avx2_sdot_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        BnQ6KCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q6k_wasm_range, &ctx, W->rows };
#else
        (void)x_q_buf;
        BnQ6KCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q6k_scalar_range, &ctx, W->rows };
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
#elif defined(__AVX2__)
        int n_sb_q4k = W->cols / BN_QK_K;
        if (n_sb_q4k < 1 || n_sb_q4k > BN_MAX_SCALE_BLOCKS / 8) return;
        float q4k_d[n_sb_q4k];
        int16_t q4k_bsums[n_sb_q4k * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q4k_d, q4k_bsums, W->cols);
        BnKQuantSdotCtx ctx = { out, W, x_q_buf, q4k_d, q4k_bsums };
        BnTPTask task = { bn_quant_q4k_avx2_sdot_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        (void)x_q_buf;
        BnQ4KCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q4k_wasm_range, &ctx, W->rows };
#else
        (void)x_q_buf;
        BnQ4KCtx ctx = { out, W, x };
        BnTPTask task = { bn_quant_q4k_scalar_range, &ctx, W->rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        return;
    }

    if (W->type == BN_GGUF_TENSOR_Q5_K) {
        (void)x_q_buf;
        BnQ5KCtx ctx = { out, W, x };
#ifdef __ARM_NEON
        BnTPTask task = { bn_quant_q5k_neon_range, &ctx, W->rows };
#elif defined(__AVX2__)
        BnTPTask task = { bn_quant_q5k_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_q5k_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_q5k_scalar_range, &ctx, W->rows };
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
        BnTPTask task = { bn_quant_bf16_avx2_range, &ctx, W->rows };
#elif defined(__wasm_simd128__)
        BnTPTask task = { bn_quant_bf16_wasm_range, &ctx, W->rows };
#else
        BnTPTask task = { bn_quant_bf16_scalar_range, &ctx, W->rows };
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

// --- Float-x kernel selection ---
// Returns platform-optimal kernel for types that use float x (no int8 quantization).

bn_tp_fn bn_quant_get_float_kernel(int type) {
    switch (type) {
#ifdef __ARM_NEON
    case BN_GGUF_TENSOR_Q4_K:    return bn_quant_q4k_neon_range;
    case BN_GGUF_TENSOR_Q5_K:    return bn_quant_q5k_neon_range;
    case BN_GGUF_TENSOR_Q6_K:    return bn_quant_q6k_neon_range;
    case BN_GGUF_TENSOR_Q3_K:    return bn_quant_q3k_neon_range;
    case BN_GGUF_TENSOR_Q2_K:    return bn_quant_q2k_neon_range;
    case BN_GGUF_TENSOR_Q8_K:    return bn_quant_q8k_neon_range;
    case BN_GGUF_TENSOR_BF16:    return bn_quant_bf16_neon_range;
    case BN_GGUF_TENSOR_Q4_1:    return bn_quant_q4_1_neon_range;
    case BN_GGUF_TENSOR_IQ4_NL:  return bn_quant_iq4nl_neon_range;
    case BN_GGUF_TENSOR_IQ4_XS:  return bn_quant_iq4xs_neon_range;
    case BN_GGUF_TENSOR_IQ3_XXS: return bn_quant_iq3xxs_neon_range;
    case BN_GGUF_TENSOR_IQ3_S:   return bn_quant_iq3s_neon_range;
    case BN_GGUF_TENSOR_IQ2_XXS: return bn_quant_iq2xxs_neon_range;
    case BN_GGUF_TENSOR_IQ2_XS:  return bn_quant_iq2xs_neon_range;
    case BN_GGUF_TENSOR_IQ2_S:   return bn_quant_iq2s_neon_range;
#elif defined(__AVX2__)
    case BN_GGUF_TENSOR_Q4_K:    return bn_quant_q4k_avx2_range;
    case BN_GGUF_TENSOR_Q5_K:    return bn_quant_q5k_avx2_range;
    case BN_GGUF_TENSOR_Q6_K:    return bn_quant_q6k_avx2_range;
    case BN_GGUF_TENSOR_Q3_K:    return bn_quant_q3k_avx2_range;
    case BN_GGUF_TENSOR_Q2_K:    return bn_quant_q2k_avx2_range;
    case BN_GGUF_TENSOR_Q8_K:    return bn_quant_q8k_avx2_range;
    case BN_GGUF_TENSOR_BF16:    return bn_quant_bf16_avx2_range;
    case BN_GGUF_TENSOR_Q4_1:    return bn_quant_q4_1_avx2_range;
    case BN_GGUF_TENSOR_IQ4_NL:  return bn_quant_iq4nl_avx2_range;
    case BN_GGUF_TENSOR_IQ4_XS:  return bn_quant_iq4xs_avx2_range;
    case BN_GGUF_TENSOR_IQ3_XXS: return bn_quant_iq3xxs_avx2_range;
    case BN_GGUF_TENSOR_IQ3_S:   return bn_quant_iq3s_avx2_range;
    case BN_GGUF_TENSOR_IQ2_XXS: return bn_quant_iq2xxs_avx2_range;
    case BN_GGUF_TENSOR_IQ2_XS:  return bn_quant_iq2xs_avx2_range;
    case BN_GGUF_TENSOR_IQ2_S:   return bn_quant_iq2s_avx2_range;
#elif defined(__wasm_simd128__)
    case BN_GGUF_TENSOR_Q4_K:    return bn_quant_q4k_wasm_range;
    case BN_GGUF_TENSOR_Q5_K:    return bn_quant_q5k_wasm_range;
    case BN_GGUF_TENSOR_Q6_K:    return bn_quant_q6k_wasm_range;
    case BN_GGUF_TENSOR_Q3_K:    return bn_quant_q3k_wasm_range;
    case BN_GGUF_TENSOR_Q2_K:    return bn_quant_q2k_wasm_range;
    case BN_GGUF_TENSOR_Q8_K:    return bn_quant_q8k_wasm_range;
    case BN_GGUF_TENSOR_BF16:    return bn_quant_bf16_wasm_range;
    case BN_GGUF_TENSOR_Q4_1:    return bn_quant_q4_1_wasm_range;
    case BN_GGUF_TENSOR_IQ4_NL:  return bn_quant_iq4nl_wasm_range;
    case BN_GGUF_TENSOR_IQ4_XS:  return bn_quant_iq4xs_wasm_range;
    case BN_GGUF_TENSOR_IQ3_XXS: return bn_quant_iq3xxs_wasm_range;
    case BN_GGUF_TENSOR_IQ3_S:   return bn_quant_iq3s_wasm_range;
    case BN_GGUF_TENSOR_IQ2_XXS: return bn_quant_iq2xxs_wasm_range;
    case BN_GGUF_TENSOR_IQ2_XS:  return bn_quant_iq2xs_wasm_range;
    case BN_GGUF_TENSOR_IQ2_S:   return bn_quant_iq2s_wasm_range;
#else
    case BN_GGUF_TENSOR_Q4_K:    return bn_quant_q4k_scalar_range;
    case BN_GGUF_TENSOR_Q5_K:    return bn_quant_q5k_scalar_range;
    case BN_GGUF_TENSOR_Q6_K:    return bn_quant_q6k_scalar_range;
    case BN_GGUF_TENSOR_Q3_K:    return bn_quant_q3k_scalar_range;
    case BN_GGUF_TENSOR_Q2_K:    return bn_quant_q2k_scalar_range;
    case BN_GGUF_TENSOR_Q8_K:    return bn_quant_q8k_scalar_range;
    case BN_GGUF_TENSOR_BF16:    return bn_quant_bf16_scalar_range;
    case BN_GGUF_TENSOR_Q4_1:    return bn_quant_q4_1_scalar_range;
    case BN_GGUF_TENSOR_IQ4_NL:  return bn_quant_iq4nl_scalar_range;
    case BN_GGUF_TENSOR_IQ4_XS:  return bn_quant_iq4xs_scalar_range;
    case BN_GGUF_TENSOR_IQ3_XXS: return bn_quant_iq3xxs_scalar_range;
    case BN_GGUF_TENSOR_IQ3_S:   return bn_quant_iq3s_scalar_range;
    case BN_GGUF_TENSOR_IQ2_XXS: return bn_quant_iq2xxs_scalar_range;
    case BN_GGUF_TENSOR_IQ2_XS:  return bn_quant_iq2xs_scalar_range;
    case BN_GGUF_TENSOR_IQ2_S:   return bn_quant_iq2s_scalar_range;
#endif
    default: return NULL;
    }
}

// --- Batch matvec ---
// Runs multiple independent matvecs with a single dispatch.

void bn_quant_matvec_batch(const BnMatvecTask *tasks, int n_tasks,
                           const float *x, int8_t *x_q_buf, BnThreadPool *pool) {
    if (n_tasks <= 0) return;

    int cols = tasks[0].W->cols;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int all_i2s = 1, all_q4 = 1, all_tq1 = 1, all_tq2 = 1, all_q8 = 1;
    int all_q4k = 1, all_q6k = 1;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].W->type != BN_GGUF_TENSOR_I2_S) all_i2s = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_0) all_q4 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_TQ1_0) all_tq1 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_TQ2_0) all_tq2 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q8_0) all_q8 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_K) all_q4k = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q6_K) all_q6k = 0;
        if (!all_i2s && !all_q4 && !all_tq1 && !all_tq2 && !all_q8 &&
            !all_q4k && !all_q6k) break;
    }

    if (all_i2s) {
        if (n_tasks > 4) {
            for (int t = 0; t < n_tasks; t++)
                bn_quant_matvec(tasks[t].out, tasks[t].W, x, x_q_buf, pool);
            return;
        }

        float x_scale = bn_quant_x_to_i8(x, x_q_buf, cols);

        BnI2SCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnI2SCtx){ tasks[t].out, tasks[t].W, x_q_buf,
                                tasks[t].W->scale * x_scale };
            tp_tasks[t] = (BnTPTask){ bn_quant_i2s_neon_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q4 && n_tasks <= 4) {
        int n_blocks = cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, cols);

        BnQ4SdotCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            void (*fn)(void *, int, int) = tasks[t].W->rp_scales
                ? bn_quant_q4_repacked_neon_sdot_range
                : bn_quant_q4_neon_sdot_range;
            ctxs[t] = (BnQ4SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales };
            tp_tasks[t] = (BnTPTask){ fn, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_tq1 && n_tasks <= 4) {
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, cols);

        BnTQ1SdotCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnTQ1SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf,
                                     tasks[t].W->scale * x_scale };
            tp_tasks[t] = (BnTPTask){ bn_quant_tq1_neon_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_tq2 && n_tasks <= 4) {
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, cols);

        BnTQ2SdotCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnTQ2SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf,
                                     tasks[t].W->scale * x_scale };
            tp_tasks[t] = (BnTPTask){ bn_quant_tq2_neon_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q8 && n_tasks <= 4) {
        int n_blocks = cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, cols);

        BnQ8SdotCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnQ8SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales };
            tp_tasks[t] = (BnTPTask){ bn_quant_q8_neon_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q6k && n_tasks <= BN_MAX_BATCH) {
        int n_sb = cols / BN_QK_K;
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) { for (int t = 0; t < n_tasks; t++) bn_quant_matvec(tasks[t].out, tasks[t].W, x, x_q_buf, pool); return; }
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, cols);

        BnQ6KSdotCtx ctxs[BN_MAX_BATCH];
        BnTPTask tp_tasks[BN_MAX_BATCH];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnQ6KSdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, q8k_d, q8k_bsums };
            tp_tasks[t] = (BnTPTask){ bn_quant_q6k_neon_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q4k && n_tasks <= BN_MAX_BATCH) {
        int n_sb = cols / BN_QK_K;
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) { for (int t = 0; t < n_tasks; t++) bn_quant_matvec(tasks[t].out, tasks[t].W, x, x_q_buf, pool); return; }
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, cols);

        BnQ4KSdotCtx ctxs[BN_MAX_BATCH];
        BnTPTask tp_tasks[BN_MAX_BATCH];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnQ4KSdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, q8k_d, q8k_bsums };
            tp_tasks[t] = (BnTPTask){ bn_quant_q4k_neon_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }
#elif defined(__AVX2__)
    int all_i2s = 1, all_q4 = 1, all_tq1 = 1, all_tq2 = 1;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].W->type != BN_GGUF_TENSOR_I2_S) all_i2s = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_0) all_q4 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_TQ1_0) all_tq1 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_TQ2_0) all_tq2 = 0;
        if (!all_i2s && !all_q4 && !all_tq1 && !all_tq2) break;
    }

    if (all_i2s) {
        if (n_tasks > 4) {
            for (int t = 0; t < n_tasks; t++)
                bn_quant_matvec(tasks[t].out, tasks[t].W, x, x_q_buf, pool);
            return;
        }

        float x_scale = bn_quant_x_to_i8(x, x_q_buf, cols);

        BnI2SCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnI2SCtx){ tasks[t].out, tasks[t].W, x_q_buf,
                                tasks[t].W->scale * x_scale };
            int n_groups = (tasks[t].W->rows + 3) / 4;
            tp_tasks[t] = (BnTPTask){ bn_quant_i2s_avx2_4row_range, &ctxs[t], n_groups };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q4 && n_tasks <= 4) {
        int n_blocks = cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, cols);

        BnQ4SdotCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnQ4SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales };
            int n_groups = (tasks[t].W->rows + 3) / 4;
            tp_tasks[t] = (BnTPTask){ bn_quant_q4_avx2_4row_range, &ctxs[t], n_groups };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_tq1 && n_tasks <= 4) {
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, cols);

        BnTQ1SdotCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnTQ1SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf,
                                     tasks[t].W->scale * x_scale };
            tp_tasks[t] = (BnTPTask){ bn_quant_tq1_avx2_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_tq2 && n_tasks <= 4) {
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, cols);

        BnTQ2SdotCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnTQ2SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf,
                                     tasks[t].W->scale * x_scale };
            tp_tasks[t] = (BnTPTask){ bn_quant_tq2_avx2_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }
#elif defined(__wasm_relaxed_simd__)
    int all_i2s = 1, all_q4 = 1;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].W->type != BN_GGUF_TENSOR_I2_S) all_i2s = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_0) all_q4 = 0;
        if (!all_i2s && !all_q4) break;
    }

    if (all_i2s && n_tasks <= 4) {
        float x_scale = bn_quant_x_to_i8(x, x_q_buf, cols);

        BnI2SCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnI2SCtx){ tasks[t].out, tasks[t].W, x_q_buf,
                                tasks[t].W->scale * x_scale };
            tp_tasks[t] = (BnTPTask){ bn_quant_i2s_wasm_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q4 && n_tasks <= 4) {
        int n_blocks = cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, cols);

        BnQ4SdotCtx ctxs[4];
        BnTPTask tp_tasks[4];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnQ4SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales };
            tp_tasks[t] = (BnTPTask){ bn_quant_q4_wasm_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }
#else
    (void)x_q_buf;
    (void)cols;
#endif

    // Generic batch for float-x types (K-quants, BF16, IQ*, Q4_1, Q8_K).
    // All share identical ctx layout { out, W, x } — no int8 quantization.
    if (n_tasks <= BN_MAX_BATCH) {
        int batch_type = tasks[0].W->type;
        int all_same = 1;
        for (int t = 1; t < n_tasks; t++) {
            if (tasks[t].W->type != batch_type) { all_same = 0; break; }
        }
        if (all_same) {
            bn_tp_fn kernel = bn_quant_get_float_kernel(batch_type);
            if (kernel) {
                // All float-x ctx types share { out, W, x } layout
                BnQ4KCtx ctxs[BN_MAX_BATCH];
                BnTPTask tp_tasks[BN_MAX_BATCH];
                for (int t = 0; t < n_tasks; t++) {
                    ctxs[t] = (BnQ4KCtx){ tasks[t].out, tasks[t].W, x };
                    tp_tasks[t] = (BnTPTask){ kernel, &ctxs[t], tasks[t].W->rows };
                }
                bn_tp_dispatch(pool, tp_tasks, n_tasks);
                return;
            }
        }
    }

    // Fallback: use existing per-task matvec
    for (int t = 0; t < n_tasks; t++) {
        bn_quant_matvec(tasks[t].out, tasks[t].W, x, x_q_buf, pool);
    }
}
