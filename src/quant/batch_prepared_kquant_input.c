#include "quant_ctx.h"
#include "quant_dispatch_internal.h"
#include "quant_kernels_scalar.h"
#include "quant_kernels_neon.h"
#include "quant_kernels_avx512.h"
#include "quant_kernels_avx2.h"
#include "quant_kernels_wasm.h"
#include "threadpool.h"
#include "gguf.h"

#ifdef BN_FORCE_SCALAR
#undef __AVX512F__
#undef __AVX512BW__
#undef __AVX512VNNI__
#undef __AVX2__
#endif

#define BN_MAX_BATCH 24

// Batch matvec with prepared K-quant input. Dispatches 4-row kernels
// directly without re-quantizing. Falls back to float path for non-k-quant types.
void bn_quant_matvec_batch_prepared_kquant_input(
    const BnMatvecTask *tasks,
    int n_tasks,
    const int8_t *x_q,
    const float *x_d,
    const int16_t *x_bsums,
    const float *x_float,
    BnThreadPool *pool) {
    if (n_tasks <= 0) return;

#if defined(__AVX2__)
    if (x_float &&
        bn_quant_policy_avx2_q5k_float_matvec_enabled(
            bn_tp_quant_policy(pool))) {
        for (int t = 0; t < n_tasks; t++) {
            if (tasks[t].W->type == BN_GGUF_TENSOR_Q5_K) {
                for (int i = 0; i < n_tasks; i++)
                    bn_quant_matvec_impl(
                        tasks[i].out, tasks[i].W, x_float,
                        (int8_t *)x_q, pool, tasks[i].prepared,
                        tasks[i].flags);
                return;
            }
        }
    }
#endif

#if (defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)) || defined(__AVX2__)
    if (n_tasks <= BN_MAX_BATCH) {
        int all_kquant = 1;
        for (int t = 0; t < n_tasks; t++) {
            int type = tasks[t].W->type;
            if (type != BN_GGUF_TENSOR_Q4_K &&
                type != BN_GGUF_TENSOR_Q5_K &&
                type != BN_GGUF_TENSOR_Q6_K) {
                all_kquant = 0;
                break;
            }
        }
        if (all_kquant) {
            BnKQuantSdotCtx ctxs[BN_MAX_BATCH];
            BnTPTask tp_tasks[BN_MAX_BATCH];
            for (int t = 0; t < n_tasks; t++) {
                ctxs[t] = (BnKQuantSdotCtx){
                    tasks[t].out, tasks[t].W,
                    (int8_t *)x_q, (float *)x_d, (int16_t *)x_bsums,
                    tasks[t].prepared
                };
                bn_tp_fn fn;
                int reference_dot =
                    tasks[t].W->type == BN_GGUF_TENSOR_Q4_K
                    ? bn_quant_policy_reference_q4k_dot_enabled(
                          bn_tp_quant_policy(pool), tasks[t].flags)
                    : tasks[t].W->type == BN_GGUF_TENSOR_Q6_K
                    ? bn_quant_policy_reference_q6_dot_enabled(
                          bn_tp_quant_policy(pool), tasks[t].flags)
                    : 0;
                int use_q4k_x8 =
                    tasks[t].W->type == BN_GGUF_TENSOR_Q4_K &&
                    !reference_dot &&
                    tasks[t].prepared && tasks[t].prepared->aux &&
                    (tasks[t].W->rows % 8) == 0;
                if (use_q4k_x8) {
                    fn = (bn_tp_fn)bn_quant_q4k_avx2_x8_matvec_range;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                } else if (reference_dot) {
                    fn = tasks[t].W->type == BN_GGUF_TENSOR_Q4_K
                        ? (bn_tp_fn)bn_quant_q4k_avx2_sdot_range
                        : (bn_tp_fn)bn_quant_q6k_avx2_sdot_range;
#endif
                } else if (tasks[t].W->type == BN_GGUF_TENSOR_Q5_K) {
                    fn = (bn_tp_fn)bn_quant_q5k_avx2_sdot_range;
                } else {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                    if (tasks[t].W->type == BN_GGUF_TENSOR_Q4_K)
                        fn = (bn_tp_fn)bn_quant_q4k_avx512_vnni_4row_range;
                    else
                        fn = (bn_tp_fn)bn_quant_q6k_avx2_4row_range;
#else
                    if (tasks[t].W->type == BN_GGUF_TENSOR_Q4_K)
                        fn = (bn_tp_fn)bn_quant_q4k_avx2_4row_range;
                    else
                        fn = (bn_tp_fn)bn_quant_q6k_avx2_4row_range;
#endif
                }
                int n_groups = use_q4k_x8
                    ? tasks[t].W->rows / 8
                    : reference_dot
                        ? tasks[t].W->rows
                    : tasks[t].W->type == BN_GGUF_TENSOR_Q5_K
                        ? tasks[t].W->rows
                    : tasks[t].W->type == BN_GGUF_TENSOR_Q4_K
                        ? tasks[t].W->rows
                        : (tasks[t].W->rows + 3) / 4;
#if defined(__AVX2__) && \
    !(defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__))
                if (reference_dot &&
                    tasks[t].W->type == BN_GGUF_TENSOR_Q4_K)
                    n_groups = (tasks[t].W->rows + 3) / 4;
#endif
                tp_tasks[t] = (BnTPTask){ fn, &ctxs[t], n_groups };
            }
            bn_tp_dispatch(pool, tp_tasks, n_tasks);
            return;
        }
    }
#else
    (void)x_d;
    (void)x_bsums;
#endif

    // Fallback: use float path (x_float must be provided)
    if (x_float) {
        for (int t = 0; t < n_tasks; t++) {
            bn_quant_matvec_impl(tasks[t].out, tasks[t].W, x_float,
                                 (int8_t *)x_q, pool, tasks[t].prepared,
                                 tasks[t].flags);
        }
    }
}
