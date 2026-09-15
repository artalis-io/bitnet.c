#include "quant_ctx.h"
#include "quant.h"
#include "quant_dispatch_internal.h"
#include "quant_kernels_avx512.h"
#include "quant_kernels_neon.h"
#include "quant_kernels_avx2.h"
#include "quant_kernels_scalar.h"
#include "threadpool.h"
#include "gguf.h"
#include <stdlib.h>
#include <string.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX2__
#undef __AVX512F__
#undef __AVX512BW__
#undef __AVX512VNNI__
#endif

#define BN_MAX_SCALE_BLOCKS 8192

#if !defined(__AVX2__) && !(defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
static void matmul_quant_x_to_q8_blocks_scalar(const float *x, int8_t *x_q,
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
        float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        x_scales[b] = d;
        for (int i = 0; i < 32; i++) {
            float v = xb[i] * id;
            int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            xqb[i] = (int8_t)q;
        }
    }
}
#endif

static inline void matmul_quant_x_to_q8k(const float *x, int8_t *x_q,
                                         float *x_d, int16_t *x_bsums,
                                         int n) {
#if defined(BN_FORCE_SCALAR)
    bn_quant_x_to_q8k_scalar(x, x_q, x_d, x_bsums, n);
#else
    bn_quant_x_to_q8k(x, x_q, x_d, x_bsums, n);
#endif
}

#ifdef __AVX2__
static void matmul_q4k_gemv_rows(void *ctx, int start, int end);
static void matmul_q4k_gemm8_rows(void *ctx, int start, int end);
static void matmul_q6k_rows(void *ctx, int start, int end);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
static void matmul_q4k_gemm16_rows(void *ctx, int start, int end);
static void matmul_q8_vnni_rows(void *ctx, int start, int end) {
    /* Keep pool chunks in physical rows and assign each four-row group
     * exactly once, including when chunk boundaries cut through a group. */
    int first = start / 4 + (start % 4 != 0);
    int last = end / 4 + (end % 4 != 0);
    if (first < last)
        bn_quant_q8_avx512_vnni_matmul_4row_range(ctx, first, last);
}
#endif
static BnBlockQ8Kx4 *pack_q8k_x4(const int8_t *x_q, const float *x_d,
                                 const float *x_float, int n_tokens,
                                 int cols);
#endif

/* A supplied panel is borrowed. Tail-only canonical inputs are valid only
 * when every matrix is guaranteed to select a packed Q4_K GEMM kernel. */
static void matmul_prepared_kquant_input_multi(
    float **out, const BnQWeight **W, const BnPreparedWeight **prepared,
    int n, int n_tokens, const int8_t *x_q, const float *x_d,
    const int16_t *x_bsums, const float *x_float, BnThreadPool *pool,
    const BnBlockQ8Kx4 *packed_input);

#ifdef __AVX2__
static int q4k_uses_packed_input(const BnPreparedWeight *prepared, int rows,
                                 BnThreadPool *pool) {
    return prepared && prepared->kind == BN_PREPARED_WEIGHT_Q4_K_SCALES &&
           prepared->aux && rows % 8 == 0 &&
           bn_quant_policy_native_matmul_batch_enabled(bn_tp_quant_policy(pool));
}
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void q4_pack_q8_panel4(const int8_t *xq_all, const float *xs_all,
                              int n_tokens, int cols, int n_blocks,
                              int8_t *xq4, float *xs4) {
    int n_panels = n_tokens / 4;
    for (int p = 0; p < n_panels; p++) {
        int t0 = p * 4;
        for (int b = 0; b < n_blocks; b++) {
            int8_t *dst = xq4 + ((size_t)p * n_blocks + b) * 128;
            for (int half = 0; half < 2; half++) {
                for (int g = 0; g < 4; g++) {
                    int off = half * 16 + g * 4;
                    int dst_base = half * 64 + g * 16;
                    for (int t = 0; t < 4; t++) {
                        const int8_t *src = xq_all + (size_t)(t0 + t) * cols + b * 32 + off;
                        memcpy(dst + dst_base + t * 4, src, 4);
                    }
                }
            }
            float *sd = xs4 + ((size_t)p * n_blocks + b) * 4;
            for (int t = 0; t < 4; t++)
                sd[t] = xs_all[(size_t)(t0 + t) * n_blocks + b];
        }
    }
}
#endif

void bn_quant_matmul_prepared(float *out, const BnQWeight *W,
                              const BnPreparedWeight *prepared,
                              const float *X, int n_tokens,
                              int8_t *x_q_buf, BnThreadPool *pool) {
#if !defined(__AVX2__) && !(defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    (void)prepared;
#endif
    int rows = W->rows;
    int cols = W->cols;

    if (n_tokens <= 1) {
        bn_quant_matvec_prepared(out, W, prepared, X, x_q_buf, pool);
        return;
    }

#ifdef __AVX2__
    if (W->type == BN_GGUF_TENSOR_Q5_1 &&
        bn_quant_policy_native_matmul_batch_enabled(bn_tp_quant_policy(pool))) {
        BnKQuantFloatMatmulCtx ctx = {out, W, X, n_tokens, cols};
        BnTPTask task = {bn_quant_q5_1_avx2_matmul_range, &ctx, rows};
        bn_tp_dispatch(pool, &task, 1);
        return;
    }
    if (W->type == BN_GGUF_TENSOR_F32) {
        BnKQuantFloatMatmulCtx ctx = { out, W, X, n_tokens, cols };
        BnTPTask task = { bn_quant_f32_avx2_matmul_range, &ctx, rows };
        if (bn_quant_policy_f32_matmul_fine(
                rows, cols, n_tokens, bn_tp_num_threads(pool)))
            bn_tp_dispatch_fine(pool, &task, 1);
        else
            bn_tp_dispatch(pool, &task, 1);
        return;
    }
#endif

#if defined(__AVX2__) || (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    if (W->type == BN_GGUF_TENSOR_Q8_0
#ifdef __AVX2__
        || W->type == BN_GGUF_TENSOR_MXFP4
#endif
    ) {
        if (!bn_quant_policy_native_matmul_batch_enabled(
                bn_tp_quant_policy(pool)))
            goto fallback_loop;
        int n_blocks = cols / 32;
        if (n_blocks < 1 || n_blocks > BN_MAX_SCALE_BLOCKS) goto fallback_loop;
        size_t xq_size = (size_t)n_tokens * cols;
        if (n_tokens > 0 && xq_size / n_tokens != (size_t)cols) goto fallback_loop;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xs_all = (float *)malloc((size_t)n_tokens * n_blocks * sizeof(float));
        if (!xq_all || !xs_all) {
            free(xq_all);
            free(xs_all);
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            bn_quant_x_to_q8_blocks(X + (size_t)t * cols,
                                    xq_all + (size_t)t * cols,
                                    xs_all + (size_t)t * n_blocks, cols);
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
#ifdef __AVX2__
        BnQ4MatmulCtx ctx = {
            out, W, xq_all, xs_all, prepared, n_tokens, cols, NULL, NULL, 0
        };
        if (W->type == BN_GGUF_TENSOR_MXFP4) {
            BnTPTask task = { bn_quant_mxfp4_avx2_matmul_range, &ctx, rows };
            bn_tp_dispatch(pool, &task, 1);
            free(xq_all);
            free(xs_all);
            return;
        }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        BnTPTask task = {
            matmul_q8_vnni_rows,
            &ctx,
            rows
        };
#else
        BnTPTask task = { bn_quant_q8_avx2_matmul_range, &ctx, rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        free(xq_all);
        free(xs_all);
        return;
#else
        free(xq_all);
        free(xs_all);
        goto fallback_loop;
#endif
    }

    if (W->type == BN_GGUF_TENSOR_Q4_0) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__AVX2__)
        goto fallback_loop;
#endif
        int n_blocks = cols / 32;
        if (n_blocks < 1 || n_blocks > BN_MAX_SCALE_BLOCKS) goto fallback_loop;
        size_t xq_size = (size_t)n_tokens * cols;
        if (n_tokens > 0 && xq_size / n_tokens != (size_t)cols) goto fallback_loop;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xs_all = (float *)malloc((size_t)n_tokens * n_blocks * sizeof(float));
        if (!xq_all || !xs_all) {
            free(xq_all);
            free(xs_all);
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            bn_quant_x_to_q8_blocks(X + (size_t)t * cols,
                                    xq_all + (size_t)t * cols,
                                    xs_all + (size_t)t * n_blocks, cols);
        int8_t *xq4 = NULL;
        float *xs4 = NULL;
        int n_panels = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        int use_panel4 = prepared && prepared->qs && prepared->scales &&
                         (rows % 4) == 0 && n_tokens >= 4;
        if (use_panel4) {
            n_panels = n_tokens / 4;
            xq4 = (int8_t *)malloc((size_t)n_panels * n_blocks * 128);
            xs4 = (float *)malloc((size_t)n_panels * n_blocks * 4 * sizeof(float));
            if (!xq4 || !xs4) {
                free(xq4);
                free(xs4);
                xq4 = NULL;
                xs4 = NULL;
                n_panels = 0;
                use_panel4 = 0;
            } else {
                q4_pack_q8_panel4(xq_all, xs_all, n_tokens, cols, n_blocks, xq4, xs4);
            }
        }
#endif
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
        BnQ4MatmulCtx ctx = {
            out, W, xq_all, xs_all, prepared, n_tokens, cols, NULL, NULL, 0
        };
        ctx.x_q4 = xq4;
        ctx.x_scales4 = xs4;
        ctx.n_token_panels = n_panels;
#ifdef __AVX2__
        BnTPTask task = { bn_quant_q4_avx2_matmul_range, &ctx, rows };
#else
        int use_group_range = prepared && prepared->qs && prepared->scales &&
                              (rows % 4) == 0;
        BnTPTask task = {
            use_panel4
                ? bn_quant_q4_repacked_neon_sdot_matmul_panel4_range
                : (use_group_range
                ? bn_quant_q4_repacked_neon_sdot_matmul_group_range
                : ((prepared && prepared->qs && prepared->scales)
                    ? bn_quant_q4_repacked_neon_sdot_matmul_range
                    : bn_quant_q4_neon_sdot_matmul_range)),
            &ctx,
            use_group_range ? rows / 4 : rows
        };
#endif
        bn_tp_dispatch(pool, &task, 1);
        free(xq4);
        free(xs4);
        free(xq_all);
        free(xs_all);
        return;
    }
#endif

#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__)
    if (W->type == BN_GGUF_TENSOR_Q4_K) {
        int n_bpr = cols / BN_QK_K;
        if (n_bpr < 1 || n_bpr > BN_MAX_SCALE_BLOCKS / 8) goto fallback_loop;
        size_t xq_size = (size_t)n_tokens * cols;
        if (n_tokens > 0 && xq_size / n_tokens != (size_t)cols) goto fallback_loop;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xd_all = (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
        int16_t *xbs_all = (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
        if (!xq_all || !xd_all || !xbs_all) {
            free(xq_all);
            free(xd_all);
            free(xbs_all);
            for (int t = 0; t < n_tokens; t++)
                bn_quant_matvec_prepared(
                    out + (size_t)t * rows, W, prepared,
                    X + (size_t)t * cols, x_q_buf, pool);
            return;
        }
        int canonical_start = 0;
#ifdef __AVX2__
        BnBlockQ8Kx4 *x_q8k_x4 = NULL;
        if (prepared && prepared->aux && (rows % 8) == 0 &&
            n_tokens >= 4)
            x_q8k_x4 = pack_q8k_x4(NULL, NULL, X, n_tokens, cols);
        if (x_q8k_x4 && q4k_uses_packed_input(prepared, rows, pool))
            canonical_start = n_tokens - n_tokens % 4;
#endif
        for (int t = canonical_start; t < n_tokens; t++)
            matmul_quant_x_to_q8k(X + (size_t)t * cols,
                              xq_all + (size_t)t * cols,
                              xd_all + (size_t)t * n_bpr,
                              xbs_all + (size_t)t * n_bpr * 16, cols);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
#endif

        BnKQuantMatmulCtx ctx = {
            out, W, xq_all, xd_all, xbs_all, n_tokens, cols, prepared,
#ifdef __AVX2__
            x_q8k_x4
#else
            NULL
#endif
        };
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        BnTPTask task = { bn_quant_q4k_neon_sdot_matmul_range, &ctx, rows };
#elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        BnTPTask task = !bn_quant_policy_native_matmul_batch_enabled(
                              bn_tp_quant_policy(pool))
            ? (BnTPTask){ bn_quant_q4k_avx2_sdot_matmul_exact_range,
                          &ctx, rows }
            : x_q8k_x4 && (rows % 16) == 0
            ? (BnTPTask){ matmul_q4k_gemm16_rows, &ctx, rows }
            : x_q8k_x4
            ? (BnTPTask){ matmul_q4k_gemm8_rows, &ctx, rows }
            : (prepared && prepared->aux && (rows % 8) == 0)
            ? (BnTPTask){ matmul_q4k_gemv_rows, &ctx, rows }
            : (BnTPTask){ bn_quant_q4k_avx2_sdot_matmul_exact_range,
                          &ctx, rows };
#else
        BnTPTask task = !bn_quant_policy_native_matmul_batch_enabled(
                              bn_tp_quant_policy(pool))
            ? (BnTPTask){ bn_quant_q4k_avx2_sdot_matmul_exact_range,
                          &ctx, rows }
            : x_q8k_x4
            ? (BnTPTask){ matmul_q4k_gemm8_rows, &ctx, rows }
            : (prepared && prepared->aux && (rows % 8) == 0)
            ? (BnTPTask){ matmul_q4k_gemv_rows, &ctx, rows }
            : (BnTPTask){ bn_quant_q4k_avx2_sdot_matmul_range, &ctx, rows };
#endif
        bn_tp_dispatch(pool, &task, 1);

#ifdef __AVX2__
        free(x_q8k_x4);
#endif
        free(xq_all);
        free(xd_all);
        free(xbs_all);
        return;
    }
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (W->type == BN_GGUF_TENSOR_Q5_K) {
        int n_bpr = cols / BN_QK_K;
        if (n_bpr < 1 || n_bpr > BN_MAX_SCALE_BLOCKS / 8) goto fallback_loop;
        size_t xq_size = (size_t)n_tokens * cols;
        if (n_tokens > 0 && xq_size / n_tokens != (size_t)cols) goto fallback_loop;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xd_all = (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
        int16_t *xbs_all = (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
        if (!xq_all || !xd_all || !xbs_all) {
            free(xq_all);
            free(xd_all);
            free(xbs_all);
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            matmul_quant_x_to_q8k(X + (size_t)t * cols,
                              xq_all + (size_t)t * cols,
                              xd_all + (size_t)t * n_bpr,
                              xbs_all + (size_t)t * n_bpr * 16, cols);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
#endif
        BnKQuantMatmulCtx ctx = {
            out, W, xq_all, xd_all, xbs_all, n_tokens, cols, NULL, NULL
        };
        BnTPTask task = { bn_quant_q5k_neon_sdot_matmul_range, &ctx, rows };
        bn_tp_dispatch(pool, &task, 1);

        free(xq_all);
        free(xd_all);
        free(xbs_all);
        return;
    }
#elif defined(__AVX2__)
    if (W->type == BN_GGUF_TENSOR_Q5_K) {
        int n_bpr = cols / BN_QK_K;
        if (n_bpr < 1 || n_bpr > BN_MAX_SCALE_BLOCKS / 8) goto fallback_loop;
        size_t xq_size = (size_t)n_tokens * cols;
        if (n_tokens > 0 && xq_size / n_tokens != (size_t)cols) goto fallback_loop;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xd_all = (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
        int16_t *xbs_all = (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
        if (!xq_all || !xd_all || !xbs_all) {
            free(xq_all);
            free(xd_all);
            free(xbs_all);
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            matmul_quant_x_to_q8k(X + (size_t)t * cols,
                              xq_all + (size_t)t * cols,
                              xd_all + (size_t)t * n_bpr,
                              xbs_all + (size_t)t * n_bpr * 16, cols);
        BnKQuantMatmulCtx ctx = {
            out, W, xq_all, xd_all, xbs_all, n_tokens, cols, prepared, NULL
        };
        BnTPTask task = {
            bn_quant_q5k_avx2_sdot_matmul_range,
            &ctx,
            rows
        };
        bn_tp_dispatch(pool, &task, 1);
        free(xq_all);
        free(xd_all);
        free(xbs_all);
        return;
    }
#endif
    if (W->type == BN_GGUF_TENSOR_Q6_K) {
        int n_bpr = cols / BN_QK_K;
        if (n_bpr < 1 || n_bpr > BN_MAX_SCALE_BLOCKS / 8) goto fallback_loop;
        size_t xq_size = (size_t)n_tokens * cols;
        if (n_tokens > 0 && xq_size / n_tokens != (size_t)cols) goto fallback_loop;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xd_all = (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
        int16_t *xbs_all = (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
        if (!xq_all || !xd_all || !xbs_all) {
            free(xq_all);
            free(xd_all);
            free(xbs_all);
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            matmul_quant_x_to_q8k(X + (size_t)t * cols,
                              xq_all + (size_t)t * cols,
                              xd_all + (size_t)t * n_bpr,
                              xbs_all + (size_t)t * n_bpr * 16, cols);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
#endif
        BnKQuantMatmulCtx ctx = {
            out, W, xq_all, xd_all, xbs_all, n_tokens, cols, prepared, NULL
        };
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        BnTPTask task = { bn_quant_q6k_neon_sdot_matmul_range, &ctx, rows };
#else
        BnTPTask task = {
            matmul_q6k_rows,
            &ctx,
            rows
        };
#endif
        bn_tp_dispatch(pool, &task, 1);
        free(xq_all);
        free(xd_all);
        free(xbs_all);
        return;
    }
fallback_loop:
#endif

#if !defined(__AVX2__) && !(defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    if (W->type == BN_GGUF_TENSOR_Q8_0) {
        int n_blocks = cols / 32;
        if (n_blocks < 1 || n_blocks > BN_MAX_SCALE_BLOCKS) goto fallback_loop;
        size_t xq_size = (size_t)n_tokens * cols;
        if (n_tokens > 0 && xq_size / n_tokens != (size_t)cols) goto fallback_loop;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xs_all = (float *)malloc((size_t)n_tokens * n_blocks * sizeof(float));
        if (!xq_all || !xs_all) {
            free(xq_all);
            free(xs_all);
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            matmul_quant_x_to_q8_blocks_scalar(
                X + (size_t)t * cols, xq_all + (size_t)t * cols,
                xs_all + (size_t)t * n_blocks, cols);
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
        BnQ8MatmulCtx ctx = {
            out, W, xq_all, xs_all, n_tokens, cols, prepared
        };
        BnTPTask task = { bn_quant_q8_scalar_sdot_matmul_range, &ctx, rows };
        bn_tp_dispatch(pool, &task, 1);
        free(xq_all);
        free(xs_all);
        return;
    }
    if (W->type == BN_GGUF_TENSOR_Q4_K) {
        int n_bpr = cols / BN_QK_K;
        if (n_bpr < 1 || n_bpr > BN_MAX_SCALE_BLOCKS / 8) goto fallback_loop;
        size_t xq_size = (size_t)n_tokens * cols;
        if (n_tokens > 0 && xq_size / n_tokens != (size_t)cols) goto fallback_loop;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xd_all = (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
        int16_t *xbs_all = (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
        if (!xq_all || !xd_all || !xbs_all) {
            free(xq_all);
            free(xd_all);
            free(xbs_all);
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            matmul_quant_x_to_q8k(X + (size_t)t * cols,
                              xq_all + (size_t)t * cols,
                              xd_all + (size_t)t * n_bpr,
                              xbs_all + (size_t)t * n_bpr * 16, cols);
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
        BnKQuantMatmulCtx ctx = {
            .out = out,
            .W = W,
            .x_q = xq_all,
            .x_d = xd_all,
            .x_bsums = xbs_all,
            .n_tokens = n_tokens,
            .cols = cols,
        };
        BnTPTask task = { bn_quant_q4k_scalar_sdot_matmul_range, &ctx, rows };
        bn_tp_dispatch(pool, &task, 1);
        free(xq_all);
        free(xd_all);
        free(xbs_all);
        return;
    }
    if (W->type == BN_GGUF_TENSOR_Q5_K) {
        int n_bpr = cols / BN_QK_K;
        if (n_bpr < 1 || n_bpr > BN_MAX_SCALE_BLOCKS / 8) goto fallback_loop;
        size_t xq_size = (size_t)n_tokens * cols;
        if (n_tokens > 0 && xq_size / n_tokens != (size_t)cols) goto fallback_loop;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xd_all = (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
        int16_t *xbs_all = (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
        if (!xq_all || !xd_all || !xbs_all) {
            free(xq_all);
            free(xd_all);
            free(xbs_all);
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            matmul_quant_x_to_q8k(X + (size_t)t * cols,
                                  xq_all + (size_t)t * cols,
                                  xd_all + (size_t)t * n_bpr,
                                  xbs_all + (size_t)t * n_bpr * 16, cols);
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
        BnKQuantMatmulCtx ctx = {
            .out = out,
            .W = W,
            .x_q = xq_all,
            .x_d = xd_all,
            .x_bsums = xbs_all,
            .n_tokens = n_tokens,
            .cols = cols,
        };
        BnTPTask task = { bn_quant_q5k_scalar_sdot_matmul_range, &ctx, rows };
        bn_tp_dispatch(pool, &task, 1);
        free(xq_all);
        free(xd_all);
        free(xbs_all);
        return;
    }
    if (W->type == BN_GGUF_TENSOR_Q6_K) {
        int n_bpr = cols / BN_QK_K;
        if (n_bpr < 1 || n_bpr > BN_MAX_SCALE_BLOCKS / 8) goto fallback_loop;
        size_t xq_size = (size_t)n_tokens * cols;
        if (n_tokens > 0 && xq_size / n_tokens != (size_t)cols) goto fallback_loop;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xd_all = (float *)malloc((size_t)n_tokens * n_bpr * sizeof(float));
        int16_t *xbs_all = (int16_t *)malloc((size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
        if (!xq_all || !xd_all || !xbs_all) {
            free(xq_all);
            free(xd_all);
            free(xbs_all);
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            matmul_quant_x_to_q8k(X + (size_t)t * cols,
                              xq_all + (size_t)t * cols,
                              xd_all + (size_t)t * n_bpr,
                              xbs_all + (size_t)t * n_bpr * 16, cols);
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
        BnKQuantMatmulCtx ctx = {
            .out = out,
            .W = W,
            .x_q = xq_all,
            .x_d = xd_all,
            .x_bsums = xbs_all,
            .n_tokens = n_tokens,
            .cols = cols,
        };
        BnTPTask task = { bn_quant_q6k_scalar_sdot_matmul_range, &ctx, rows };
        bn_tp_dispatch(pool, &task, 1);
        free(xq_all);
        free(xd_all);
        free(xbs_all);
        return;
    }
fallback_loop:
#endif

#if defined(__AVX2__)
    if (W->type == BN_GGUF_TENSOR_IQ4_NL && n_tokens >= 4 &&
        (rows % 8) == 0) {
        int n_blocks = cols / 32;
        size_t xq_size = (size_t)n_tokens * cols;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xs_all = (float *)malloc(
            (size_t)n_tokens * n_blocks * sizeof(float));
        if (!xq_all || !xs_all) {
            free(xq_all);
            free(xs_all);
            goto iq4nl_fallback;
        }
        for (int t = 0; t < n_tokens; t++)
            bn_quant_x_to_q8_blocks(
                X + (size_t)t * cols, xq_all + (size_t)t * cols,
                xs_all + (size_t)t * n_blocks, cols);
        BnQ8MatmulCtx ctx = {
            out, W, xq_all, xs_all, n_tokens, cols, prepared
        };
        BnTPTask task = {
            bn_quant_iq4nl_avx2_q8_matmul_x8_range, &ctx, rows / 8
        };
        bn_tp_dispatch(pool, &task, 1);
        free(xq_all);
        free(xs_all);
        return;
    }
iq4nl_fallback:
    if (W->type == BN_GGUF_TENSOR_IQ3_XXS ||
        W->type == BN_GGUF_TENSOR_IQ3_S ||
        W->type == BN_GGUF_TENSOR_IQ4_XS) {
        int n_bpr = cols / BN_QK_K;
        size_t xq_size = (size_t)n_tokens * cols;
        int8_t *xq_all = (int8_t *)malloc(xq_size);
        float *xd_all = (float *)malloc(
            (size_t)n_tokens * n_bpr * sizeof(float));
        int16_t *xbs_all = (int16_t *)malloc(
            (size_t)n_tokens * n_bpr * 16 * sizeof(int16_t));
        if (!xq_all || !xd_all || !xbs_all) {
            free(xq_all);
            free(xd_all);
            free(xbs_all);
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            matmul_quant_x_to_q8k(
                X + (size_t)t * cols, xq_all + (size_t)t * cols,
                xd_all + (size_t)t * n_bpr,
                xbs_all + (size_t)t * n_bpr * 16, cols);
        if (n_tokens >= 8 && (rows % 8) == 0) {
            BnKQuantMatmulCtx ctx = {
                out, W, xq_all, xd_all, xbs_all, n_tokens, cols,
                prepared, NULL
            };
            BnTPTask task = {
                bn_quant_iq_panel_avx2_matmul_range, &ctx, rows / 8
            };
            bn_tp_dispatch(pool, &task, 1);
        } else for (int t = 0; t < n_tokens; t++) {
            BnKQuantSdotCtx ctx = {
                out + (size_t)t * rows, W,
                xq_all + (size_t)t * cols,
                xd_all + (size_t)t * n_bpr,
                xbs_all + (size_t)t * n_bpr * 16, prepared
            };
            BnTPTask task = {
                W->type == BN_GGUF_TENSOR_IQ3_XXS
                    ? bn_quant_iq3xxs_avx2_q8k_range
                    : W->type == BN_GGUF_TENSOR_IQ3_S
                    ? bn_quant_iq3s_avx2_q8k_range
                    : bn_quant_iq4xs_avx2_q8k_range,
                &ctx, rows
            };
            bn_tp_dispatch(pool, &task, 1);
        }
        free(xq_all);
        free(xd_all);
        free(xbs_all);
        return;
    }
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if (W->type == BN_GGUF_TENSOR_IQ4_XS) {
        BnKQuantFloatMatmulCtx ctx = { out, W, X, n_tokens, cols };
        BnTPTask task = { bn_quant_iq4xs_avx2_matmul_range, &ctx, rows };
        bn_tp_dispatch(pool, &task, 1);
        return;
    }
    if (bn_quant_format_cpu_matvec_uses_float_input(W->type)) {
        bn_quant_matmul_float_x(out, W, X, n_tokens, pool);
        return;
    }
#endif

    for (int t = 0; t < n_tokens; t++) {
        bn_quant_matvec_prepared(out + (size_t)t * rows, W, prepared, X + (size_t)t * cols,
                        x_q_buf, pool);
    }
}

void bn_quant_matmul(float *out, const BnQWeight *W, const float *X,
                     int n_tokens, int8_t *x_q_buf, BnThreadPool *pool) {
    bn_quant_matmul_prepared(out, W, NULL, X, n_tokens, x_q_buf, pool);
}

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *X;
    int n_tokens;
    bn_tp_fn kernel;
} BnFloatXMatmulDispatchCtx;

static void float_x_matmul_range(void *vctx, int row_start, int row_end) {
    BnFloatXMatmulDispatchCtx *c = (BnFloatXMatmulDispatchCtx *)vctx;
    int rows = c->W->rows;
    int cols = c->W->cols;
    for (int row = row_start; row < row_end; row++) {
        for (int token = 0; token < c->n_tokens; token++) {
            BnFloatXCtx row_ctx = {
                c->out + (size_t)token * rows,
                c->W,
                c->X + (size_t)token * cols
            };
            c->kernel(&row_ctx, row, row + 1);
        }
    }
}

void bn_quant_matmul_float_x(float *out, const BnQWeight *W,
                             const float *X, int n_tokens,
                             BnThreadPool *pool) {
    if (!out || !W || !X || n_tokens <= 0) return;
    bn_tp_fn kernel = bn_quant_get_float_kernel(W->type);
    if (!kernel) return;
    BnFloatXMatmulDispatchCtx ctx = { out, W, X, n_tokens, kernel };
    BnTPTask task = { float_x_matmul_range, &ctx, W->rows };
    bn_tp_dispatch(pool, &task, 1);
}

#define BN_MAX_PREPARED_MULTI_MATMUL 4

#ifdef __AVX2__
static void matmul_q4k_unpacked_gemv_rows(void *ctx, int start, int end) {
    /* Chunk boundaries are physical rows; assign each four-row group once. */
    int first = start / 4 + (start % 4 != 0);
    int last = end / 4 + (end % 4 != 0);
    /* Reuse decoded weights across tokens without changing GEMV order. */
    if (first < last)
        bn_quant_q4k_avx2_sdot_matmul_4row_range(ctx, first, last);
}

static void matmul_q6k_rows(void *ctx, int start, int end) {
    // Keep pool chunking in physical rows without splitting a kernel group.
    int first = start / 4 + (start % 4 != 0);
    int last = end / 4 + (end % 4 != 0);
    if (first < last)
        bn_quant_q6k_avx2_sdot_matmul_4row_range(ctx, first, last);
}

static void matmul_q4k_gemv_rows(void *ctx, int start, int end) {
    // Schedule physical rows so the pool's minimum chunk does not become
    // 256 rows merely because the kernel consumes eight-row groups.
    // Ceil both boundaries: even one-row chunks assign each group once.
    int first = start / 8 + (start % 8 != 0);
    int last = end / 8 + (end % 8 != 0);
    if (first < last)
        bn_quant_q4k_avx2_x8_matmul_range(ctx, first, last);
}
static void matmul_q4k_gemm8_rows(void *ctx, int start, int end) {
    int first = start / 8 + (start % 8 != 0);
    int last = end / 8 + (end % 8 != 0);
    if (first < last)
        bn_quant_q4k_avx2_x8_gemm_range(ctx, first, last);
}
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
static void matmul_q4k_gemm16_rows(void *ctx, int start, int end) {
    int first = start / 16 + (start % 16 != 0);
    int last = end / 16 + (end % 16 != 0);
    if (first < last)
        bn_quant_q4k_avx512_x16_gemm_range(ctx, first, last);
}
#endif
#endif

void bn_quant_matmul_prepared_multi_gemv(
    float **out, const BnQWeight **W, const BnPreparedWeight **prepared,
    int n, const float *X, int n_tokens, int8_t *x_q_buf, BnThreadPool *pool) {
    if (n <= 0 || n_tokens <= 0) return;
    int cols = W[0]->cols;
#ifdef __AVX2__
    int native_gemv_order = cols > 0 && cols % 32 == 0;
    for (int i = 0; i < n && native_gemv_order; i++) {
        if (W[i]->cols != cols || !bn_quant_format_has_cap(
                W[i]->type, BN_QUANT_CAP_CPU_X86_GEMV_ORDER_MATMUL))
            native_gemv_order = 0;
        if (W[i]->type == BN_GGUF_TENSOR_Q5_K &&
            (cols % BN_QK_K != 0 ||
             bn_quant_policy_avx2_q5k_float_matvec_enabled(bn_tp_quant_policy(pool)) ||
             !bn_quant_policy_native_matmul_batch_enabled(bn_tp_quant_policy(pool))))
            native_gemv_order = 0;
    }
    if (native_gemv_order) {
        // These kernels retain GEMV arithmetic while reusing weights across
        // tokens. The normal matmul route owns policy and allocation fallback.
        for (int i = 0; i < n; i++)
            bn_quant_matmul_prepared(out[i], W[i],
                prepared ? prepared[i] : NULL, X, n_tokens, x_q_buf, pool);
        return;
    }
    /* Share Q8_K inputs and one dispatch while retaining zero-flag GEMV
     * arithmetic. Mixed/prepared layouts keep their existing route. */
    int unpacked_q4k = n <= BN_MAX_PREPARED_MULTI_MATMUL && n_tokens > 1 &&
        cols > 0 && cols % BN_QK_K == 0 &&
        cols / BN_QK_K <= BN_MAX_SCALE_BLOCKS / 8 &&
        bn_quant_policy_native_matmul_batch_enabled(bn_tp_quant_policy(pool)) &&
        !bn_quant_policy_avx2_kquant_float_for_tasks(
            bn_tp_quant_policy(pool), NULL, 0);
    for (int i = 0; i < n && unpacked_q4k; i++)
        if (W[i]->type != BN_GGUF_TENSOR_Q4_K || W[i]->cols != cols ||
            (prepared && prepared[i]))
            unpacked_q4k = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if (bn_quant_policy_reference_q4k_dot_enabled(bn_tp_quant_policy(pool), 0))
        unpacked_q4k = 0;
#endif
    int packed_q4k = n <= BN_MAX_PREPARED_MULTI_MATMUL &&
        cols > 0 && cols % BN_QK_K == 0 &&
        !bn_quant_policy_reference_q4k_dot_enabled(bn_tp_quant_policy(pool), 0);
    for (int i = 0; i < n && packed_q4k; i++)
        if (W[i]->type != BN_GGUF_TENSOR_Q4_K || W[i]->cols != cols ||
            W[i]->rows % 8 != 0 || !prepared || !prepared[i] ||
            prepared[i]->kind != BN_PREPARED_WEIGHT_Q4_K_SCALES ||
            !prepared[i]->aux)
            packed_q4k = 0;
    /* Q8_K scales and sums each occupy fewer bytes than the quantized rows.
     * Bound the largest allocation before multiplying any of their sizes. */
    if ((packed_q4k || unpacked_q4k) &&
        (size_t)n_tokens <= SIZE_MAX / (size_t)cols) {
        int nb = cols / BN_QK_K;
        int8_t *xq = malloc((size_t)n_tokens * cols);
        float *xd = malloc((size_t)n_tokens * nb * sizeof(*xd));
        int16_t *bsums = malloc((size_t)n_tokens * nb * 16 * sizeof(*bsums));
        if (xq && xd && bsums) {
            for (int t = 0; t < n_tokens; t++)
                matmul_quant_x_to_q8k(X + (size_t)t * cols,
                    xq + (size_t)t * cols, xd + (size_t)t * nb,
                    bsums + (size_t)t * nb * 16, cols);
            BnKQuantMatmulCtx contexts[BN_MAX_PREPARED_MULTI_MATMUL];
            BnTPTask tasks[BN_MAX_PREPARED_MULTI_MATMUL];
            for (int i = 0; i < n; i++) {
                contexts[i] = (BnKQuantMatmulCtx){out[i], W[i], xq, xd,
                    bsums, n_tokens, cols, prepared ? prepared[i] : NULL, NULL};
                tasks[i] = (BnTPTask){unpacked_q4k
                    ? matmul_q4k_unpacked_gemv_rows : matmul_q4k_gemv_rows,
                    &contexts[i], W[i]->rows};
            }
            bn_tp_dispatch(pool, tasks, n);
        }
        int computed = xq && xd && bsums;
        free(xq);
        free(xd);
        free(bsums);
        if (computed) return;
    }
#endif
    // Keep allocation failures and unsupported layouts on the same GEMV
    // arithmetic contract; never fall back to a native GEMM here.
    for (int t = 0; t < n_tokens; t++) {
        for (int first = 0; first < n; first += BN_MAX_PREPARED_MULTI_MATMUL) {
            BnMatvecTask tasks[BN_MAX_PREPARED_MULTI_MATMUL];
            int count = n - first;
            if (count > BN_MAX_PREPARED_MULTI_MATMUL)
                count = BN_MAX_PREPARED_MULTI_MATMUL;
            for (int j = 0; j < count; j++) {
                int i = first + j;
                tasks[j] = (BnMatvecTask){out[i] + (size_t)t * W[i]->rows,
                    W[i], prepared ? prepared[i] : NULL, 0};
            }
            bn_quant_matvec_batch(tasks, count, X + (size_t)t * cols,
                                  x_q_buf, pool);
        }
    }
}

void bn_quant_matmul_prepared_multi(float **out, const BnQWeight **W,
                                    const BnPreparedWeight **prepared, int n,
                                    const float *X, int n_tokens,
                                    int8_t *x_q_buf, BnThreadPool *pool) {
    if (n <= 0 || n > BN_MAX_PREPARED_MULTI_MATMUL) {
        for (int i = 0; i < n; i++)
            bn_quant_matmul_prepared(out[i], W[i],
                                     prepared ? prepared[i] : NULL,
                                     X, n_tokens, x_q_buf, pool);
        return;
    }

    if (n_tokens <= 1) {
        for (int i = 0; i < n; i++)
            bn_quant_matmul_prepared(out[i], W[i],
                                     prepared ? prepared[i] : NULL,
                                     X, n_tokens, x_q_buf, pool);
        return;
    }

#ifdef __AVX2__
    /* Homogeneous K-quant projections can share activation quantization,
     * panel packing, and dispatch without changing each matrix's kernel. */
    {
        int cols = W[0]->cols;
        int all_q4k = cols > 0 && cols % BN_QK_K == 0;
        for (int i = 0; i < n; i++)
            if (!W[i] || W[i]->type != BN_GGUF_TENSOR_Q4_K ||
                W[i]->cols != cols)
                all_q4k = 0;
        int nb = cols / BN_QK_K;
        if (all_q4k && nb <= BN_MAX_SCALE_BLOCKS / 8) {
            size_t xq_size = (size_t)n_tokens * cols;
            if (xq_size / (size_t)n_tokens != (size_t)cols)
                goto fallback_loop;
            int8_t *xq = malloc(xq_size);
            float *xd = malloc((size_t)n_tokens * nb * sizeof(*xd));
            int16_t *bsums = malloc((size_t)n_tokens * nb * 16 * sizeof(*bsums));
            if (xq && xd && bsums) {
                int all_packed = n_tokens >= 4;
                for (int i = 0; i < n; i++)
                    if (!q4k_uses_packed_input(prepared ? prepared[i] : NULL,
                                               W[i]->rows, pool))
                        all_packed = 0;
                BnBlockQ8Kx4 *packed = all_packed
                    ? pack_q8k_x4(NULL, NULL, X, n_tokens, cols) : NULL;
                /* On a panel allocation failure every canonical row is
                 * initialized before the normal dispatch/fallback path. */
                int canonical_start = packed ? n_tokens - n_tokens % 4 : 0;
                for (int t = canonical_start; t < n_tokens; t++)
                    matmul_quant_x_to_q8k(X + (size_t)t * cols,
                        xq + (size_t)t * cols, xd + (size_t)t * nb,
                        bsums + (size_t)t * nb * 16, cols);
                matmul_prepared_kquant_input_multi(
                    out, W, prepared, n, n_tokens, xq, xd, bsums, X, pool, packed);
                free(packed);
                free(xq);
                free(xd);
                free(bsums);
                return;
            }
            free(xq);
            free(xd);
            free(bsums);
            goto fallback_loop;
        }
    }
#endif

#if defined(__AVX2__) || (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    {
        int cols = W[0]->cols;
        int all_q4 = cols > 0 && cols % 32 == 0;
        for (int i = 0; i < n; i++) {
            if (!W[i] || W[i]->type != BN_GGUF_TENSOR_Q4_0 ||
                W[i]->cols != cols) {
                all_q4 = 0;
                break;
            }
        }

        if (all_q4) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(__AVX2__)
            goto fallback_loop;
#endif
            int n_blocks = cols / 32;
            if (n_blocks < 1 || n_blocks > BN_MAX_SCALE_BLOCKS)
                goto fallback_loop;
            size_t xq_size = (size_t)n_tokens * cols;
            if (xq_size / (size_t)n_tokens != (size_t)cols)
                goto fallback_loop;
            int8_t *xq_all = (int8_t *)malloc(xq_size);
            float *xs_all = (float *)malloc((size_t)n_tokens * n_blocks * sizeof(float));
            if (!xq_all || !xs_all) {
                free(xq_all);
                free(xs_all);
                goto fallback_loop;
            }

            for (int t = 0; t < n_tokens; t++)
                bn_quant_x_to_q8_blocks(X + (size_t)t * cols,
                                        xq_all + (size_t)t * cols,
                                        xs_all + (size_t)t * n_blocks, cols);
            int8_t *xq4 = NULL;
            float *xs4 = NULL;
            int n_panels = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            int can_panel4 = n_tokens >= 4;
            if (can_panel4) {
                n_panels = n_tokens / 4;
                xq4 = (int8_t *)malloc((size_t)n_panels * n_blocks * 128);
                xs4 = (float *)malloc((size_t)n_panels * n_blocks * 4 * sizeof(float));
                if (!xq4 || !xs4) {
                    free(xq4);
                    free(xs4);
                    xq4 = NULL;
                    xs4 = NULL;
                    n_panels = 0;
                    can_panel4 = 0;
                } else {
                    q4_pack_q8_panel4(xq_all, xs_all, n_tokens, cols, n_blocks, xq4, xs4);
                }
            }
#endif

            BnQ4MatmulCtx ctxs[BN_MAX_PREPARED_MULTI_MATMUL];
            BnTPTask tasks[BN_MAX_PREPARED_MULTI_MATMUL];
            for (int i = 0; i < n; i++) {
                memset(out[i], 0, (size_t)n_tokens * W[i]->rows * sizeof(float));
                ctxs[i] = (BnQ4MatmulCtx){
                    out[i], W[i], xq_all, xs_all,
                    (W[i]->type != BN_GGUF_TENSOR_Q6_K && prepared)
                        ? prepared[i] : NULL,
                    n_tokens, cols, NULL, NULL, 0
                };
                ctxs[i].x_q4 = xq4;
                ctxs[i].x_scales4 = xs4;
                ctxs[i].n_token_panels = n_panels;
#ifdef __AVX2__
                tasks[i] = (BnTPTask){ bn_quant_q4_avx2_matmul_range,
                                       &ctxs[i], W[i]->rows };
#else
                int use_group_range = prepared && prepared[i] &&
                                      prepared[i]->qs && prepared[i]->scales &&
                                      (W[i]->rows % 4) == 0;
                int use_panel4 = can_panel4 && use_group_range;
                tasks[i] = (BnTPTask){
                    use_panel4
                        ? bn_quant_q4_repacked_neon_sdot_matmul_panel4_range
                        : (use_group_range
                        ? bn_quant_q4_repacked_neon_sdot_matmul_group_range
                        : ((prepared && prepared[i] &&
                            prepared[i]->qs && prepared[i]->scales)
                            ? bn_quant_q4_repacked_neon_sdot_matmul_range
                            : bn_quant_q4_neon_sdot_matmul_range)),
                    &ctxs[i],
                    use_group_range ? W[i]->rows / 4 : W[i]->rows
                };
#endif
            }
            bn_tp_dispatch(pool, tasks, n);
            free(xq4);
            free(xs4);
            free(xq_all);
            free(xs_all);
            return;
        }
    }
fallback_loop:
#endif
    for (int i = 0; i < n; i++)
        bn_quant_matmul_prepared(out[i], W[i], prepared ? prepared[i] : NULL,
                                 X, n_tokens, x_q_buf, pool);
}

void bn_quant_matmul_prepared_kquant_input(float *out,
                                           const BnQWeight *W,
                                           int n_tokens,
                                           const int8_t *x_q,
                                           const float *x_d,
                                           const int16_t *x_bsums,
                                           const float *x_float,
                                           BnThreadPool *pool) {
    int rows = W->rows;
    int cols = W->cols;

    if (n_tokens <= 1) {
        BnMatvecTask task = { out, W, NULL, 0 };
        bn_quant_matvec_batch_prepared_kquant_input(
            &task, 1, x_q, x_d, x_bsums, x_float, pool);
        return;
    }

#ifdef __AVX2__
    if (W->type == BN_GGUF_TENSOR_Q4_K ||
        W->type == BN_GGUF_TENSOR_Q5_K ||
        W->type == BN_GGUF_TENSOR_Q6_K) {
        BnKQuantMatmulCtx ctx = { out, W, (int8_t *)x_q, (float *)x_d,
                                  (int16_t *)x_bsums, n_tokens, cols, NULL,
                                  NULL };
        bn_tp_fn fn;
        int units;
        if (W->type == BN_GGUF_TENSOR_Q4_K) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
            fn = (bn_tp_fn)bn_quant_q4k_avx2_sdot_matmul_exact_range;
            units = rows;
#else
            fn = (bn_tp_fn)bn_quant_q4k_avx2_sdot_matmul_range;
            units = rows;
#endif
        } else if (W->type == BN_GGUF_TENSOR_Q5_K) {
            fn = (bn_tp_fn)bn_quant_q5k_avx2_sdot_matmul_range;
            units = rows;
        } else {
            fn = matmul_q6k_rows;
            units = rows;
        }
        BnTPTask task = { fn, &ctx, units };
        bn_tp_dispatch(pool, &task, 1);
        return;
    }
#endif

    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(out + (size_t)t * rows, W, x_float + (size_t)t * cols,
                        (int8_t *)x_q, pool);
}

#define BN_MAX_MULTI_MATMUL 4

#ifdef __AVX2__
static BnBlockQ8Kx4 *pack_q8k_x4(const int8_t *x_q, const float *x_d,
                                 const float *x_float, int n_tokens,
                                 int cols) {
    int nb = cols / BN_QK_K;
    size_t count = (size_t)(n_tokens / 4) * (size_t)nb;
    BnBlockQ8Kx4 *packed = malloc(count * sizeof(*packed));
    if (!packed)
        return NULL;

    if (x_float) {
        bn_quant_q8k_avx2_pack_x4(packed, x_float,
                                  n_tokens - n_tokens % 4, cols);
        return packed;
    }

    for (int panel = 0; panel < n_tokens / 4; panel++) {
        for (int b = 0; b < nb; b++) {
            BnBlockQ8Kx4 *dst = &packed[(size_t)panel * nb + b];
            memset(dst->bsums, 0, sizeof(dst->bsums));
            for (int t = 0; t < 4; t++)
                dst->d[t] = x_d[((size_t)panel * 4 + t) * nb + b];

            for (int j = 0; j < BN_QK_K * 4; j++) {
                int src_offset = (j / 32) * 8 + j % 8;
                int src_id = (j % 32) / 8;
                int index = (((j & 31) >> 3) << 2) +
                            ((j >> 8) << 4) + ((j >> 6) & 3);
                const int8_t *src = x_q +
                    ((size_t)panel * 4 + src_id) * cols + b * BN_QK_K;
                dst->qs[j] = src[src_offset];
                dst->bsums[index] += dst->qs[j];
            }
        }
    }
    return packed;
}
#endif

static void matmul_prepared_kquant_input_multi(
    float **out,
    const BnQWeight **W,
    const BnPreparedWeight **prepared,
    int n,
    int n_tokens,
    const int8_t *x_q,
    const float *x_d,
    const int16_t *x_bsums,
    const float *x_float,
    BnThreadPool *pool,
    const BnBlockQ8Kx4 *packed_input) {
#ifndef __AVX2__
    (void)prepared;
    (void)packed_input;
#endif
    if (n <= 0 || n > BN_MAX_MULTI_MATMUL) {
        for (int i = 0; i < n; i++)
            bn_quant_matmul_prepared_kquant_input(
                out[i], W[i], n_tokens, x_q, x_d, x_bsums, x_float, pool);
        return;
    }

    if (n_tokens <= 1) {
        for (int i = 0; i < n; i++)
            bn_quant_matmul_prepared_kquant_input(
                out[i], W[i], n_tokens, x_q, x_d, x_bsums, x_float, pool);
        return;
    }

#ifdef __AVX2__
    {
        int all_kquant = 1;
        for (int i = 0; i < n; i++) {
            if (W[i]->type != BN_GGUF_TENSOR_Q4_K &&
                W[i]->type != BN_GGUF_TENSOR_Q5_K &&
                W[i]->type != BN_GGUF_TENSOR_Q6_K) {
                all_kquant = 0;
                break;
            }
        }

        if (all_kquant) {
            BnKQuantMatmulCtx ctxs[BN_MAX_MULTI_MATMUL];
            BnTPTask tasks[BN_MAX_MULTI_MATMUL];
            int cols = W[0]->cols;
            BnBlockQ8Kx4 *owned_panel = NULL;
            const BnBlockQ8Kx4 *x_q8k_x4 = packed_input;
            if (!x_q8k_x4 && n_tokens >= 4) {
                for (int i = 0; i < n; i++) {
                    if ((W[i]->type == BN_GGUF_TENSOR_Q4_K ||
                         W[i]->type == BN_GGUF_TENSOR_Q5_K) && prepared &&
                        prepared[i] && prepared[i]->aux &&
                        (W[i]->rows % 8) == 0) {
                        owned_panel = pack_q8k_x4(x_q, x_d, x_float,
                                              n_tokens, cols);
                        x_q8k_x4 = owned_panel;
                        break;
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                ctxs[i] = (BnKQuantMatmulCtx){
                    out[i], W[i], (int8_t *)x_q, (float *)x_d,
                    (int16_t *)x_bsums, n_tokens, cols,
                    prepared ? prepared[i] : NULL,
                    x_q8k_x4
                };
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                bn_tp_fn fn;
                int units;
                if (W[i]->type == BN_GGUF_TENSOR_Q4_K) {
                    if (!bn_quant_policy_native_matmul_batch_enabled(
                            bn_tp_quant_policy(pool))) {
                        fn = (bn_tp_fn)bn_quant_q4k_avx2_sdot_matmul_exact_range;
                        units = W[i]->rows;
                    } else if (prepared && prepared[i] && prepared[i]->aux &&
                        (W[i]->rows % 8) == 0) {
                        if (x_q8k_x4 && (W[i]->rows % 16) == 0) {
                            fn = matmul_q4k_gemm16_rows;
                            units = W[i]->rows;
                        } else {
                            fn = x_q8k_x4
                                ? matmul_q4k_gemm8_rows
                                : matmul_q4k_gemv_rows;
                            units = W[i]->rows;
                        }
                    } else {
                        fn = (bn_tp_fn)bn_quant_q4k_avx2_sdot_matmul_exact_range;
                        units = W[i]->rows;
                    }
                } else if (W[i]->type == BN_GGUF_TENSOR_Q5_K) {
                    fn = (bn_tp_fn)bn_quant_q5k_avx2_sdot_matmul_range;
                    units = W[i]->rows;
                } else {
                    /* llama.cpp keeps Q6_K in its native x86 layout. */
                    fn = matmul_q6k_rows;
                    units = W[i]->rows;
                }
#else
                bn_tp_fn fn;
                int units;
                if (W[i]->type == BN_GGUF_TENSOR_Q4_K) {
                    if (!bn_quant_policy_native_matmul_batch_enabled(
                            bn_tp_quant_policy(pool))) {
                        fn = (bn_tp_fn)bn_quant_q4k_avx2_sdot_matmul_exact_range;
                        units = W[i]->rows;
                    } else if (prepared && prepared[i] && prepared[i]->aux &&
                        (W[i]->rows % 8) == 0) {
                        fn = x_q8k_x4
                            ? matmul_q4k_gemm8_rows
                            : matmul_q4k_gemv_rows;
                        units = W[i]->rows;
                    } else {
                        fn = (bn_tp_fn)bn_quant_q4k_avx2_sdot_matmul_range;
                        units = W[i]->rows;
                    }
                } else if (W[i]->type == BN_GGUF_TENSOR_Q5_K) {
                    fn = (bn_tp_fn)bn_quant_q5k_avx2_sdot_matmul_range;
                    units = W[i]->rows;
                } else {
                    /* llama.cpp keeps Q6_K in its native x86 layout. */
                    fn = matmul_q6k_rows;
                    units = W[i]->rows;
                }
#endif
                tasks[i] = (BnTPTask){ fn, &ctxs[i], units };
            }
            bn_tp_dispatch(pool, tasks, n);
            free(owned_panel);
            return;
        }
    }
#endif

    for (int i = 0; i < n; i++)
        bn_quant_matmul_prepared_kquant_input(
            out[i], W[i], n_tokens, x_q, x_d, x_bsums, x_float, pool);
}

void bn_quant_matmul_prepared_kquant_input_multi(
    float **out, const BnQWeight **W, const BnPreparedWeight **prepared,
    int n, int n_tokens, const int8_t *x_q, const float *x_d,
    const int16_t *x_bsums, const float *x_float, BnThreadPool *pool) {
    matmul_prepared_kquant_input_multi(out, W, prepared, n, n_tokens,
        x_q, x_d, x_bsums, x_float, pool, NULL);
}
