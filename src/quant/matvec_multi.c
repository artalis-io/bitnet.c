#include "quant_ctx.h"
#include "quant_dispatch_internal.h"
#include "quant_kernels_scalar.h"
#include "quant_kernels_neon.h"
#include "quant_kernels_avx2.h"
#include "quant_kernels_wasm.h"
#include "threadpool.h"
#include "gguf.h"
#include <stdlib.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

#define BN_MAX_SCALE_BLOCKS 8192
#define BN_MAX_BATCH 24

// Multi-input matvec: K independent (W, x) pairs in a single dispatch.
// Pre-quantizes each x, builds per-task contexts, dispatches all at once.
void bn_quant_matvec_multi(const BnMatvecMultiTask *tasks, int n_tasks,
                           int8_t *x_q_bufs, BnThreadPool *pool) {
    if (n_tasks <= 0) return;
    if (n_tasks == 1) {
        bn_quant_matvec(tasks[0].out, tasks[0].W, tasks[0].x, x_q_bufs, pool);
        return;
    }

    int cols = tasks[0].W->cols;

#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__) || defined(__wasm_relaxed_simd__)
    // Determine common type (all tasks should have same type for efficient batching)
    int type0 = tasks[0].W->type;

    // SDOT path: quantize each x independently, then dispatch all tasks
    int all_same_type = 1;
    for (int t = 1; t < n_tasks; t++)
        if (tasks[t].W->type != type0) { all_same_type = 0; break; }

    if (all_same_type && type0 == BN_GGUF_TENSOR_I2_S && n_tasks <= BN_MAX_BATCH) {
        float x_scales[BN_MAX_BATCH];
        for (int t = 0; t < n_tasks; t++)
            x_scales[t] = bn_quant_x_to_i8(tasks[t].x, x_q_bufs + (size_t)t * cols, cols);

        BnI2SCtx ctxs[BN_MAX_BATCH];
        BnTPTask tp_tasks[BN_MAX_BATCH];
        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnI2SCtx){
                tasks[t].out, tasks[t].W,
                x_q_bufs + (size_t)t * cols,
                tasks[t].W->scale * x_scales[t]
            };
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            tp_tasks[t] = (BnTPTask){ bn_quant_i2s_neon_sdot_range, &ctxs[t], tasks[t].W->rows };
#elif defined(__AVX2__)
            int n_groups = (tasks[t].W->rows + 3) / 4;
            tp_tasks[t] = (BnTPTask){ bn_quant_i2s_avx2_4row_range, &ctxs[t], n_groups };
#else
            tp_tasks[t] = (BnTPTask){ bn_quant_i2s_wasm_sdot_range, &ctxs[t], tasks[t].W->rows };
#endif
        }
        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    // K-quant SDOT: quantize to Q8_K per task
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__) || defined(__wasm_relaxed_simd__)
    if (all_same_type && (type0 == BN_GGUF_TENSOR_Q4_K ||
                          type0 == BN_GGUF_TENSOR_Q6_K
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
                          || type0 == BN_GGUF_TENSOR_Q5_K
#endif
                          ) && n_tasks <= BN_MAX_BATCH) {
        int n_bpr = cols / BN_QK_K;
        if (n_bpr >= 1 && n_bpr <= BN_MAX_SCALE_BLOCKS / 8) {
            // VLAs sized by actual n_bpr (not worst-case BN_MAX_SCALE_BLOCKS)
            float q8k_d[n_tasks * n_bpr];
            int16_t q8k_bsums[n_tasks * n_bpr * 16];
            for (int t = 0; t < n_tasks; t++)
                bn_quant_x_to_q8k(tasks[t].x, x_q_bufs + (size_t)t * cols,
                                  q8k_d + t * n_bpr, q8k_bsums + t * n_bpr * 16, cols);

            BnKQuantSdotCtx ctxs[BN_MAX_BATCH];
            BnTPTask tp_tasks[BN_MAX_BATCH];
            for (int t = 0; t < n_tasks; t++) {
                ctxs[t] = (BnKQuantSdotCtx){
                    tasks[t].out, tasks[t].W,
                    x_q_bufs + (size_t)t * cols,
                    q8k_d + t * n_bpr,
                    q8k_bsums + t * n_bpr * 16
                };
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
                void (*fn)(void *, int, int);
                if (type0 == BN_GGUF_TENSOR_Q4_K)
                    fn = bn_quant_q4k_neon_sdot_range;
                else if (type0 == BN_GGUF_TENSOR_Q5_K)
                    fn = bn_quant_q5k_neon_sdot_range;
                else
                    fn = bn_quant_q6k_neon_sdot_range;
#elif defined(__AVX2__)
                void (*fn)(void *, int, int) = (type0 == BN_GGUF_TENSOR_Q4_K)
                    ? bn_quant_q4k_avx2_4row_range : bn_quant_q6k_avx2_4row_range;
#else
                void (*fn)(void *, int, int) = (type0 == BN_GGUF_TENSOR_Q4_K)
                    ? bn_quant_q4k_wasm_sdot_range : bn_quant_q6k_wasm_sdot_range;
#endif
                int n_items = tasks[t].W->rows;
#ifdef __AVX2__
                n_items = (tasks[t].W->rows + 3) / 4;
#endif
                tp_tasks[t] = (BnTPTask){ fn, &ctxs[t], n_items };
            }
            bn_tp_dispatch(pool, tp_tasks, n_tasks);
            return;
        }
    }

#if defined(__AVX2__)
    int all_kquant = 1;
    for (int t = 0; t < n_tasks; t++) {
        int ty = tasks[t].W->type;
        if (ty != BN_GGUF_TENSOR_Q4_K && ty != BN_GGUF_TENSOR_Q6_K) {
            all_kquant = 0;
            break;
        }
    }
    if (!all_same_type && all_kquant && n_tasks <= BN_MAX_BATCH) {
        int n_bpr = cols / BN_QK_K;
        if (n_bpr >= 1 && n_bpr <= BN_MAX_SCALE_BLOCKS / 8) {
            float q8k_d[n_tasks * n_bpr];
            int16_t q8k_bsums[n_tasks * n_bpr * 16];
            for (int t = 0; t < n_tasks; t++)
                bn_quant_x_to_q8k(tasks[t].x, x_q_bufs + (size_t)t * cols,
                                  q8k_d + t * n_bpr,
                                  q8k_bsums + t * n_bpr * 16, cols);

            BnKQuantSdotCtx ctxs[BN_MAX_BATCH];
            BnTPTask tp_tasks[BN_MAX_BATCH];
            for (int t = 0; t < n_tasks; t++) {
                ctxs[t] = (BnKQuantSdotCtx){
                    tasks[t].out, tasks[t].W,
                    x_q_bufs + (size_t)t * cols,
                    q8k_d + t * n_bpr,
                    q8k_bsums + t * n_bpr * 16
                };
                void (*fn)(void *, int, int) =
                    (tasks[t].W->type == BN_GGUF_TENSOR_Q4_K)
                    ? bn_quant_q4k_avx2_4row_range
                    : bn_quant_q6k_avx2_4row_range;
                int n_groups = (tasks[t].W->rows + 3) / 4;
                tp_tasks[t] = (BnTPTask){ fn, &ctxs[t], n_groups };
            }
            bn_tp_dispatch(pool, tp_tasks, n_tasks);
            return;
        }
    }
#endif
#endif

    // Q8_0/Q4_0 SDOT path
    if (all_same_type && (type0 == BN_GGUF_TENSOR_Q8_0 || type0 == BN_GGUF_TENSOR_Q4_0)
        && n_tasks <= BN_MAX_BATCH) {
        int n_blocks = cols / 32;
        if (n_blocks <= BN_MAX_SCALE_BLOCKS) {
            float x_scales_all[n_tasks * n_blocks];  // VLA sized by actual dims
            for (int t = 0; t < n_tasks; t++)
                bn_quant_x_to_q8_blocks(tasks[t].x, x_q_bufs + (size_t)t * cols,
                                        x_scales_all + t * n_blocks, cols);

            if (type0 == BN_GGUF_TENSOR_Q4_0) {
                BnQ4SdotCtx ctxs[BN_MAX_BATCH];
                BnTPTask tp_tasks[BN_MAX_BATCH];
                for (int t = 0; t < n_tasks; t++) {
                    ctxs[t] = (BnQ4SdotCtx){
                        tasks[t].out, tasks[t].W,
                        x_q_bufs + (size_t)t * cols,
                        x_scales_all + t * n_blocks,
                        tasks[t].prepared
                    };
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
                    void (*fn)(void *, int, int) = (tasks[t].prepared && tasks[t].prepared->scales)
                        ? bn_quant_q4_repacked_neon_sdot_range
                        : bn_quant_q4_neon_sdot_range;
#elif defined(__AVX2__)
                    void (*fn)(void *, int, int) = bn_quant_q4_avx2_4row_range;
#elif defined(__wasm_relaxed_simd__)
                    void (*fn)(void *, int, int) = getenv("BN_WASM_Q4_CANONICAL4")
                        ? bn_quant_q4_wasm_sdot_4row_range
                        : (tasks[t].prepared && tasks[t].prepared->qs
                           ? bn_quant_q4_repacked_wasm_sdot_8row_range
                           : bn_quant_q4_wasm_sdot_range);
#else
                    void (*fn)(void *, int, int) = bn_quant_q4_scalar_range;
#endif
                    int n_items = tasks[t].W->rows;
#if defined(__AVX2__)
                    n_items = (tasks[t].W->rows + 3) / 4;
#elif defined(__wasm_relaxed_simd__)
                    if (getenv("BN_WASM_Q4_CANONICAL4"))
                        n_items = (tasks[t].W->rows + 3) / 4;
                    else if (tasks[t].prepared && tasks[t].prepared->qs)
                        n_items = (tasks[t].W->rows + 7) / 8;
#endif
                    tp_tasks[t] = (BnTPTask){ fn, &ctxs[t], n_items };
                }
                bn_tp_dispatch(pool, tp_tasks, n_tasks);
                return;
            }
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__)
            if (type0 == BN_GGUF_TENSOR_Q8_0) {
                BnQ8SdotCtx ctxs[BN_MAX_BATCH];
                BnTPTask tp_tasks[BN_MAX_BATCH];
                for (int t = 0; t < n_tasks; t++) {
                    ctxs[t] = (BnQ8SdotCtx){
                        tasks[t].out, tasks[t].W,
                        x_q_bufs + (size_t)t * cols,
                        x_scales_all + t * n_blocks,
                        tasks[t].prepared
                    };
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
                    tp_tasks[t] = (BnTPTask){ bn_quant_q8_neon_sdot_range, &ctxs[t], tasks[t].W->rows };
#else
                    int n_groups = (tasks[t].W->rows + 3) / 4;
                    tp_tasks[t] = (BnTPTask){ bn_quant_q8_avx2_4row_range, &ctxs[t], n_groups };
#endif
                }
                bn_tp_dispatch(pool, tp_tasks, n_tasks);
                return;
            }
#endif
        }
    }
#endif

#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__wasm_relaxed_simd__)
    if (n_tasks <= BN_MAX_BATCH) {
        int all_kquant = 1;
        for (int t = 0; t < n_tasks; t++) {
            int ty = tasks[t].W->type;
            if (ty != BN_GGUF_TENSOR_Q4_K && ty != BN_GGUF_TENSOR_Q6_K) {
                all_kquant = 0;
                break;
            }
        }
        if (all_kquant) {
            int n_bpr = cols / BN_QK_K;
            if (n_bpr >= 1 && n_bpr <= BN_MAX_SCALE_BLOCKS / 8) {
                float q8k_d[n_tasks * n_bpr];
                int16_t q8k_bsums[n_tasks * n_bpr * 16];
                for (int t = 0; t < n_tasks; t++)
                    bn_quant_x_to_q8k_scalar(tasks[t].x, x_q_bufs + (size_t)t * cols,
                                             q8k_d + t * n_bpr,
                                             q8k_bsums + t * n_bpr * 16, cols);

                BnKQuantSdotCtx ctxs[BN_MAX_BATCH];
                BnTPTask tp_tasks[BN_MAX_BATCH];
                for (int t = 0; t < n_tasks; t++) {
                    ctxs[t] = (BnKQuantSdotCtx){
                        tasks[t].out, tasks[t].W,
                        x_q_bufs + (size_t)t * cols,
                        q8k_d + t * n_bpr,
                        q8k_bsums + t * n_bpr * 16
                    };
                    bn_tp_fn fn = (tasks[t].W->type == BN_GGUF_TENSOR_Q4_K)
                        ? bn_quant_q4k_scalar_sdot_range
                        : bn_quant_q6k_scalar_sdot_range;
                    tp_tasks[t] = (BnTPTask){ fn, &ctxs[t], tasks[t].W->rows };
                }
                bn_tp_dispatch(pool, tp_tasks, n_tasks);
                return;
            }
        }
    }
#endif

#ifdef __AVX2__
    if (all_same_type && type0 == BN_GGUF_TENSOR_Q5_K && n_tasks <= BN_MAX_BATCH) {
        BnQ5KCtx ctxs[BN_MAX_BATCH];
        BnTPTask tp_tasks[BN_MAX_BATCH];
        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnQ5KCtx){ tasks[t].out, tasks[t].W, tasks[t].x };
            int n_groups = (tasks[t].W->rows + 3) / 4;
            tp_tasks[t] = (BnTPTask){ bn_quant_q5k_avx2_4row_range, &ctxs[t], n_groups };
        }
        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_same_type && type0 == BN_GGUF_TENSOR_BF16 && n_tasks <= BN_MAX_BATCH) {
        BnBF16Ctx ctxs[BN_MAX_BATCH];
        BnTPTask tp_tasks[BN_MAX_BATCH];
        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnBF16Ctx){ tasks[t].out, tasks[t].W, tasks[t].x };
            int n_groups = (tasks[t].W->rows + 3) / 4;
            tp_tasks[t] = (BnTPTask){ bn_quant_bf16_avx2_4row_range, &ctxs[t], n_groups };
        }
        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }
#endif

    // Fallback: sequential matvec
    for (int t = 0; t < n_tasks; t++)
        bn_quant_matvec_impl(tasks[t].out, tasks[t].W, tasks[t].x,
                             x_q_bufs + (size_t)t * cols, pool,
                             tasks[t].prepared);
}
