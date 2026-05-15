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

#define BN_MAX_SCALE_BLOCKS 8192
#define BN_MAX_BATCH 24

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
static int bn_quant_batch_use_avx512_q5k_vnni(int rows) {
    const char *v = getenv("BN_AVX512_Q5K_VNNI");
    if (v)
        return v[0] != '\0' && v[0] != '0';
    return rows >= 4096;
}
#endif

#if (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)) || defined(__AVX2__)
static int bn_quant_batch_force_avx2_kquant_float(const BnMatvecTask *tasks,
                                                  int n_tasks) {
    const char *v = getenv("BN_AVX2_KQUANT_FLOAT");
    if (v && v[0] != '\0' && v[0] != '0')
        return 1;
    for (int i = 0; i < n_tasks; i++) {
        if (tasks[i].flags & BN_MATVEC_TASK_FORCE_FLOAT_KQUANT)
            return 1;
    }
    return 0;
}
#endif

// --- Batch matvec ---
// Runs multiple independent matvecs with a single dispatch.

void bn_quant_matvec_batch(const BnMatvecTask *tasks, int n_tasks,
                           const float *x, int8_t *x_q_buf, BnThreadPool *pool) {
    if (n_tasks <= 0) return;

    int cols = tasks[0].W->cols;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int all_i2s = 1, all_q4 = 1, all_tq1 = 1, all_tq2 = 1, all_q8 = 1;
    int all_q4k = 1, all_q5k = 1, all_q6k = 1;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].W->type != BN_GGUF_TENSOR_I2_S) all_i2s = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_0) all_q4 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_TQ1_0) all_tq1 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_TQ2_0) all_tq2 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q8_0) all_q8 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_K) all_q4k = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q5_K) all_q5k = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q6_K) all_q6k = 0;
        if (!all_i2s && !all_q4 && !all_tq1 && !all_tq2 && !all_q8 &&
            !all_q4k && !all_q5k && !all_q6k) break;
    }

    if (all_i2s) {
        if (n_tasks > 4) {
            for (int t = 0; t < n_tasks; t++)
                bn_quant_matvec_impl(tasks[t].out, tasks[t].W, x, x_q_buf, pool, tasks[t].prepared);
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
            void (*fn)(void *, int, int) = (tasks[t].prepared && tasks[t].prepared->scales)
                ? bn_quant_q4_repacked_neon_sdot_range
                : bn_quant_q4_neon_sdot_range;
            ctxs[t] = (BnQ4SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales, tasks[t].prepared };
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
            ctxs[t] = (BnQ8SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales, tasks[t].prepared };
            tp_tasks[t] = (BnTPTask){ bn_quant_q8_neon_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q6k && n_tasks <= BN_MAX_BATCH) {
        int n_sb = cols / BN_QK_K;
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) { for (int t = 0; t < n_tasks; t++) bn_quant_matvec_impl(tasks[t].out, tasks[t].W, x, x_q_buf, pool, tasks[t].prepared); return; }
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
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) { for (int t = 0; t < n_tasks; t++) bn_quant_matvec_impl(tasks[t].out, tasks[t].W, x, x_q_buf, pool, tasks[t].prepared); return; }
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

    if (all_q5k && n_tasks <= BN_MAX_BATCH) {
        int n_sb = cols / BN_QK_K;
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) { for (int t = 0; t < n_tasks; t++) bn_quant_matvec_impl(tasks[t].out, tasks[t].W, x, x_q_buf, pool, tasks[t].prepared); return; }
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, cols);

        BnQ5KSdotCtx ctxs[BN_MAX_BATCH];
        BnTPTask tp_tasks[BN_MAX_BATCH];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnQ5KSdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, q8k_d, q8k_bsums };
            tp_tasks[t] = (BnTPTask){ bn_quant_q5k_neon_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }
#elif (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)) || defined(__AVX2__)
    int all_i2s = 1, all_q4 = 1, all_q8 = 1, all_tq1 = 1, all_tq2 = 1;
    int all_kquant = 1;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].W->type != BN_GGUF_TENSOR_I2_S) all_i2s = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_0) all_q4 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q8_0) all_q8 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_TQ1_0) all_tq1 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_TQ2_0) all_tq2 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_K &&
            tasks[t].W->type != BN_GGUF_TENSOR_Q6_K) all_kquant = 0;
        if (!all_i2s && !all_q4 && !all_q8 && !all_tq1 && !all_tq2 &&
            !all_kquant) break;
    }

    if (all_i2s) {
        if (n_tasks > 4) {
            for (int t = 0; t < n_tasks; t++)
                bn_quant_matvec_impl(tasks[t].out, tasks[t].W, x, x_q_buf, pool, tasks[t].prepared);
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
            ctxs[t] = (BnQ4SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales, tasks[t].prepared };
            int n_groups = (tasks[t].W->rows + 3) / 4;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
            tp_tasks[t] = (BnTPTask){ bn_quant_q4_avx512_vnni_4row_range, &ctxs[t], n_groups };
#else
            tp_tasks[t] = (BnTPTask){ bn_quant_q4_avx2_4row_range, &ctxs[t], n_groups };
#endif
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q8 && n_tasks <= BN_MAX_BATCH) {
        int n_blocks = cols / 32;
        if (n_blocks > BN_MAX_SCALE_BLOCKS) return;
        float x_scales[n_blocks];
        bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, cols);

        BnQ8SdotCtx ctxs[BN_MAX_BATCH];
        BnTPTask tp_tasks[BN_MAX_BATCH];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnQ8SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales, tasks[t].prepared };
            int n_groups = (tasks[t].W->rows + 3) / 4;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
            tp_tasks[t] = (BnTPTask){ bn_quant_q8_avx512_vnni_4row_range, &ctxs[t], n_groups };
#else
            tp_tasks[t] = (BnTPTask){ bn_quant_q8_avx2_4row_range, &ctxs[t], n_groups };
#endif
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

    /* Q4_K / Q6_K batch: quantize x to Q8_K ONCE, reuse across all tasks.
     * Uses 4-row kernels for bandwidth amortization. */
    if (all_kquant && n_tasks <= BN_MAX_BATCH &&
        !bn_quant_batch_force_avx2_kquant_float(tasks, n_tasks)) {
        int n_bpr = cols / BN_QK_K;
        if (n_bpr >= 1 && n_bpr <= BN_MAX_SCALE_BLOCKS / 8) {
            float q8k_d[n_bpr];
            int16_t q8k_bsums[n_bpr * 16];
            bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, cols);

            BnKQuantSdotCtx ctxs[BN_MAX_BATCH];
            BnTPTask tp_tasks[BN_MAX_BATCH];
            for (int t = 0; t < n_tasks; t++) {
                ctxs[t] = (BnKQuantSdotCtx){ tasks[t].out, tasks[t].W,
                                              x_q_buf, q8k_d, q8k_bsums };
                bn_tp_fn fn;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                fn = (tasks[t].W->type == BN_GGUF_TENSOR_Q4_K)
                    ? (bn_tp_fn)bn_quant_q4k_avx512_vnni_4row_range
                    : (bn_tp_fn)bn_quant_q6k_avx512_vnni_4row_range;
#else
                fn = (tasks[t].W->type == BN_GGUF_TENSOR_Q4_K)
                    ? (bn_tp_fn)bn_quant_q4k_avx2_4row_range
                    : (bn_tp_fn)bn_quant_q6k_avx2_4row_range;
#endif
                int n_groups = (tasks[t].W->rows + 3) / 4;
                tp_tasks[t] = (BnTPTask){ fn, &ctxs[t], n_groups };
            }
            bn_tp_dispatch(pool, tp_tasks, n_tasks);
            return;
        }
    }
#elif defined(__wasm_relaxed_simd__)
    int all_i2s = 1, all_q4 = 1, all_tq1 = 1, all_tq2 = 1;
    int all_q4k = 1, all_q6k = 1;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].W->type != BN_GGUF_TENSOR_I2_S) all_i2s = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_0) all_q4 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_TQ1_0) all_tq1 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_TQ2_0) all_tq2 = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q4_K) all_q4k = 0;
        if (tasks[t].W->type != BN_GGUF_TENSOR_Q6_K) all_q6k = 0;
        if (!all_i2s && !all_q4 && !all_tq1 && !all_tq2 &&
            !all_q4k && !all_q6k) break;
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
            ctxs[t] = (BnQ4SdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, x_scales, tasks[t].prepared };
            if (getenv("BN_WASM_Q4_CANONICAL4")) {
                int n_groups = (tasks[t].W->rows + 3) / 4;
                tp_tasks[t] = (BnTPTask){ bn_quant_q4_wasm_sdot_4row_range, &ctxs[t], n_groups };
            } else if (tasks[t].prepared && tasks[t].prepared->qs) {
                int n_groups = (tasks[t].W->rows + 7) / 8;
                tp_tasks[t] = (BnTPTask){ bn_quant_q4_repacked_wasm_sdot_8row_range, &ctxs[t], n_groups };
            } else {
                tp_tasks[t] = (BnTPTask){ bn_quant_q4_wasm_sdot_range, &ctxs[t], tasks[t].W->rows };
            }
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
            tp_tasks[t] = (BnTPTask){ bn_quant_tq1_wasm_sdot_range, &ctxs[t], tasks[t].W->rows };
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
            tp_tasks[t] = (BnTPTask){ bn_quant_tq2_wasm_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q6k && n_tasks <= BN_MAX_BATCH) {
        int n_sb = cols / BN_QK_K;
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) { for (int t = 0; t < n_tasks; t++) bn_quant_matvec_impl(tasks[t].out, tasks[t].W, x, x_q_buf, pool, tasks[t].prepared); return; }
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, cols);

        BnQ6KSdotCtx ctxs[BN_MAX_BATCH];
        BnTPTask tp_tasks[BN_MAX_BATCH];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnQ6KSdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, q8k_d, q8k_bsums };
            tp_tasks[t] = (BnTPTask){ bn_quant_q6k_wasm_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }

    if (all_q4k && n_tasks <= BN_MAX_BATCH) {
        int n_sb = cols / BN_QK_K;
        if (n_sb < 1 || n_sb > BN_MAX_SCALE_BLOCKS / 8) { for (int t = 0; t < n_tasks; t++) bn_quant_matvec_impl(tasks[t].out, tasks[t].W, x, x_q_buf, pool, tasks[t].prepared); return; }
        float q8k_d[n_sb];
        int16_t q8k_bsums[n_sb * 16];
        bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, cols);

        BnQ4KSdotCtx ctxs[BN_MAX_BATCH];
        BnTPTask tp_tasks[BN_MAX_BATCH];

        for (int t = 0; t < n_tasks; t++) {
            ctxs[t] = (BnQ4KSdotCtx){ tasks[t].out, tasks[t].W, x_q_buf, q8k_d, q8k_bsums };
            tp_tasks[t] = (BnTPTask){ bn_quant_q4k_wasm_sdot_range, &ctxs[t], tasks[t].W->rows };
        }

        bn_tp_dispatch(pool, tp_tasks, n_tasks);
        return;
    }
#else
    (void)x_q_buf;
    (void)cols;
#endif

#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__wasm_relaxed_simd__)
    if (n_tasks <= BN_MAX_BATCH) {
        int all_kquant = 1;
        for (int t = 0; t < n_tasks; t++) {
            int type = tasks[t].W->type;
            if (type != BN_GGUF_TENSOR_Q4_K && type != BN_GGUF_TENSOR_Q6_K) {
                all_kquant = 0;
                break;
            }
        }
        if (all_kquant) {
            int n_bpr = cols / BN_QK_K;
            if (n_bpr >= 1 && n_bpr <= BN_MAX_SCALE_BLOCKS / 8) {
                float q8k_d[n_bpr];
                int16_t q8k_bsums[n_bpr * 16];
                bn_quant_x_to_q8k_scalar(x, x_q_buf, q8k_d, q8k_bsums, cols);

                BnKQuantSdotCtx ctxs[BN_MAX_BATCH];
                BnTPTask tp_tasks[BN_MAX_BATCH];
                for (int t = 0; t < n_tasks; t++) {
                    ctxs[t] = (BnKQuantSdotCtx){ tasks[t].out, tasks[t].W,
                                                  x_q_buf, q8k_d, q8k_bsums };
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

#if defined(__AVX2__)
    if (bn_quant_batch_force_avx2_kquant_float(tasks, n_tasks)) {
        int all_kquant_float = 1;
        for (int t = 0; t < n_tasks; t++) {
            int ty = tasks[t].W->type;
            if (ty != BN_GGUF_TENSOR_Q4_K && ty != BN_GGUF_TENSOR_Q6_K) {
                all_kquant_float = 0;
                break;
            }
        }
        if (all_kquant_float) {
            int n_bpr = cols / BN_QK_K;
            if (n_bpr >= 1 && n_bpr <= BN_MAX_SCALE_BLOCKS / 8) {
                float q8k_d[n_bpr];
                int16_t q8k_bsums[n_bpr * 16];
                bn_quant_x_to_q8k_scalar(x, x_q_buf, q8k_d, q8k_bsums, cols);

                BnKQuantSdotCtx ctxs[BN_MAX_BATCH];
                BnTPTask tp_tasks[BN_MAX_BATCH];
                for (int t = 0; t < n_tasks; t++) {
                    ctxs[t] = (BnKQuantSdotCtx){ tasks[t].out, tasks[t].W,
                                                  x_q_buf, q8k_d, q8k_bsums };
                    bn_tp_fn kernel = (tasks[t].W->type == BN_GGUF_TENSOR_Q4_K)
                        ? bn_quant_q4k_scalar_sdot_range
                        : bn_quant_q6k_scalar_sdot_range;
                    tp_tasks[t] = (BnTPTask){ kernel, &ctxs[t], tasks[t].W->rows };
                }
                bn_tp_dispatch(pool, tp_tasks, n_tasks);
                return;
            }
        }
    }
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
#ifdef __AVX2__
            if (batch_type == BN_GGUF_TENSOR_Q5_K) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                if (bn_quant_batch_use_avx512_q5k_vnni(tasks[0].W->rows)) {
                    int n_bpr = cols / BN_QK_K;
                    if (n_bpr >= 1 && n_bpr <= BN_MAX_SCALE_BLOCKS / 8) {
                        float q8k_d[n_bpr];
                        int16_t q8k_bsums[n_bpr * 16];
                        bn_quant_x_to_q8k(x, x_q_buf, q8k_d, q8k_bsums, cols);
                        BnKQuantSdotCtx ctxs[BN_MAX_BATCH];
                        BnTPTask tp_tasks[BN_MAX_BATCH];
                        for (int t = 0; t < n_tasks; t++) {
                            ctxs[t] = (BnKQuantSdotCtx){ tasks[t].out, tasks[t].W,
                                                          x_q_buf, q8k_d, q8k_bsums };
                            int n_groups = (tasks[t].W->rows + 3) / 4;
                            tp_tasks[t] = (BnTPTask){ bn_quant_q5k_avx512_vnni_4row_range,
                                                      &ctxs[t], n_groups };
                        }
                        bn_tp_dispatch(pool, tp_tasks, n_tasks);
                        return;
                    }
                }
#endif
                BnQ5KCtx ctxs[BN_MAX_BATCH];
                BnTPTask tp_tasks[BN_MAX_BATCH];
                for (int t = 0; t < n_tasks; t++) {
                    ctxs[t] = (BnQ5KCtx){ tasks[t].out, tasks[t].W, x };
                    int n_groups = (tasks[t].W->rows + 3) / 4;
                    tp_tasks[t] = (BnTPTask){ bn_quant_q5k_avx2_4row_range, &ctxs[t], n_groups };
                }
                bn_tp_dispatch(pool, tp_tasks, n_tasks);
                return;
            }
            if (batch_type == BN_GGUF_TENSOR_BF16) {
                BnBF16Ctx ctxs[BN_MAX_BATCH];
                BnTPTask tp_tasks[BN_MAX_BATCH];
                for (int t = 0; t < n_tasks; t++) {
                    ctxs[t] = (BnBF16Ctx){ tasks[t].out, tasks[t].W, x };
                    int n_groups = (tasks[t].W->rows + 3) / 4;
                    tp_tasks[t] = (BnTPTask){ bn_quant_bf16_avx2_4row_range, &ctxs[t], n_groups };
                }
                bn_tp_dispatch(pool, tp_tasks, n_tasks);
                return;
            }
#endif
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
        bn_quant_matvec_impl(tasks[t].out, tasks[t].W, x, x_q_buf, pool, tasks[t].prepared);
    }
}
