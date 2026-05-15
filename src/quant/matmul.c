#include "quant_ctx.h"
#include "quant_kernels_neon.h"
#include "quant_kernels_avx2.h"
#include "threadpool.h"
#include "gguf.h"
#include <stdlib.h>
#include <string.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#endif

#define BN_MAX_SCALE_BLOCKS 8192

void bn_quant_matmul_prepared(float *out, const BnQWeight *W,
                              const BnPreparedWeight *prepared,
                              const float *X, int n_tokens,
                              int8_t *x_q_buf, BnThreadPool *pool) {
    int rows = W->rows;
    int cols = W->cols;

    if (n_tokens <= 1) {
        bn_quant_matvec(out, W, X, x_q_buf, pool);
        return;
    }

#if defined(__AVX2__) || (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
    if (W->type == BN_GGUF_TENSOR_Q4_0) {
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
        BnQ4MatmulCtx ctx = { out, W, xq_all, xs_all, prepared, n_tokens, cols };
#ifdef __AVX2__
        BnTPTask task = { bn_quant_q4_avx2_matmul_range, &ctx, rows };
#else
        BnTPTask task = {
            (prepared && prepared->qs && prepared->scales)
                ? bn_quant_q4_repacked_neon_sdot_matmul_range
                : bn_quant_q4_neon_sdot_matmul_range,
            &ctx,
            rows
        };
#endif
        bn_tp_dispatch(pool, &task, 1);
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
            goto fallback_loop;
        }
        for (int t = 0; t < n_tokens; t++)
            bn_quant_x_to_q8k(X + (size_t)t * cols,
                              xq_all + (size_t)t * cols,
                              xd_all + (size_t)t * n_bpr,
                              xbs_all + (size_t)t * n_bpr * 16, cols);

        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));

        BnKQuantMatmulCtx ctx = { out, W, xq_all, xd_all, xbs_all, n_tokens, cols };
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        BnTPTask task = { bn_quant_q4k_neon_sdot_matmul_range, &ctx, rows };
#else
        BnTPTask task = { bn_quant_q4k_avx2_sdot_matmul_range, &ctx, rows };
#endif
        bn_tp_dispatch(pool, &task, 1);

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
            bn_quant_x_to_q8k(X + (size_t)t * cols,
                              xq_all + (size_t)t * cols,
                              xd_all + (size_t)t * n_bpr,
                              xbs_all + (size_t)t * n_bpr * 16, cols);

        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
        BnKQuantMatmulCtx ctx = { out, W, xq_all, xd_all, xbs_all, n_tokens, cols };
        BnTPTask task = { bn_quant_q5k_neon_sdot_matmul_range, &ctx, rows };
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
            bn_quant_x_to_q8k(X + (size_t)t * cols,
                              xq_all + (size_t)t * cols,
                              xd_all + (size_t)t * n_bpr,
                              xbs_all + (size_t)t * n_bpr * 16, cols);
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
        BnKQuantMatmulCtx ctx = { out, W, xq_all, xd_all, xbs_all, n_tokens, cols };
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        BnTPTask task = { bn_quant_q6k_neon_sdot_matmul_range, &ctx, rows };
#else
        BnTPTask task = { bn_quant_q6k_avx2_sdot_matmul_range, &ctx, rows };
#endif
        bn_tp_dispatch(pool, &task, 1);
        free(xq_all);
        free(xd_all);
        free(xbs_all);
        return;
    }
fallback_loop:
#endif
    for (int t = 0; t < n_tokens; t++) {
        bn_quant_matvec(out + (size_t)t * rows, W, X + (size_t)t * cols,
                        x_q_buf, pool);
    }
}

void bn_quant_matmul(float *out, const BnQWeight *W, const float *X,
                     int n_tokens, int8_t *x_q_buf, BnThreadPool *pool) {
    bn_quant_matmul_prepared(out, W, NULL, X, n_tokens, x_q_buf, pool);
}

void bn_quant_matmul_preq8k(float *out, const BnQWeight *W, int n_tokens,
                            const int8_t *x_q, const float *x_d,
                            const int16_t *x_bsums, const float *x_float,
                            BnThreadPool *pool) {
    int rows = W->rows;
    int cols = W->cols;

    if (n_tokens <= 1) {
        BnMatvecTask task = { out, W, NULL, 0 };
        bn_quant_matvec_batch_preq8k(&task, 1, x_q, x_d, x_bsums, x_float, pool);
        return;
    }

#ifdef __AVX2__
    if (W->type == BN_GGUF_TENSOR_Q4_K || W->type == BN_GGUF_TENSOR_Q6_K) {
        memset(out, 0, (size_t)n_tokens * rows * sizeof(float));
        BnKQuantMatmulCtx ctx = { out, W, (int8_t *)x_q, (float *)x_d,
                                  (int16_t *)x_bsums, n_tokens, cols };
        bn_tp_fn fn = (W->type == BN_GGUF_TENSOR_Q4_K)
            ? (bn_tp_fn)bn_quant_q4k_avx2_sdot_matmul_range
            : (bn_tp_fn)bn_quant_q6k_avx2_sdot_matmul_range;
        BnTPTask task = { fn, &ctx, rows };
        bn_tp_dispatch(pool, &task, 1);
        return;
    }
#endif

    for (int t = 0; t < n_tokens; t++)
        bn_quant_matvec(out + (size_t)t * rows, W, x_float + (size_t)t * cols,
                        (int8_t *)x_q, pool);
}

#define BN_MAX_MULTI_MATMUL 4

void bn_quant_matmul_preq8k_multi(float **out, const BnQWeight **W, int n,
                                  int n_tokens, const int8_t *x_q,
                                  const float *x_d, const int16_t *x_bsums,
                                  const float *x_float, BnThreadPool *pool) {
    if (n <= 0 || n > BN_MAX_MULTI_MATMUL) {
        for (int i = 0; i < n; i++)
            bn_quant_matmul_preq8k(out[i], W[i], n_tokens, x_q, x_d, x_bsums,
                                   x_float, pool);
        return;
    }

    if (n_tokens <= 1) {
        for (int i = 0; i < n; i++)
            bn_quant_matmul_preq8k(out[i], W[i], n_tokens, x_q, x_d, x_bsums,
                                   x_float, pool);
        return;
    }

#ifdef __AVX2__
    {
        int all_kquant = 1;
        for (int i = 0; i < n; i++) {
            if (W[i]->type != BN_GGUF_TENSOR_Q4_K &&
                W[i]->type != BN_GGUF_TENSOR_Q6_K) {
                all_kquant = 0;
                break;
            }
        }

        if (all_kquant) {
            BnKQuantMatmulCtx ctxs[BN_MAX_MULTI_MATMUL];
            BnTPTask tasks[BN_MAX_MULTI_MATMUL];
            int cols = W[0]->cols;

            for (int i = 0; i < n; i++) {
                memset(out[i], 0, (size_t)n_tokens * W[i]->rows * sizeof(float));
                ctxs[i] = (BnKQuantMatmulCtx){
                    out[i], W[i], (int8_t *)x_q, (float *)x_d,
                    (int16_t *)x_bsums, n_tokens, cols
                };
                bn_tp_fn fn = (W[i]->type == BN_GGUF_TENSOR_Q4_K)
                    ? (bn_tp_fn)bn_quant_q4k_avx2_sdot_matmul_range
                    : (bn_tp_fn)bn_quant_q6k_avx2_sdot_matmul_range;
                tasks[i] = (BnTPTask){ fn, &ctxs[i], W[i]->rows };
            }
            bn_tp_dispatch(pool, tasks, n);
            return;
        }
    }
#endif

    for (int i = 0; i < n; i++)
        bn_quant_matmul_preq8k(out[i], W[i], n_tokens, x_q, x_d, x_bsums,
                               x_float, pool);
}
