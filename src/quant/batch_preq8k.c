#include "quant_ctx.h"
#include "quant_dispatch_internal.h"
#include "quant_kernels_scalar.h"
#include "quant_kernels_neon.h"
#include "quant_kernels_avx2.h"
#include "quant_kernels_wasm.h"
#include "threadpool.h"
#include "gguf.h"

#ifdef BN_FORCE_SCALAR
#undef __AVX2__
#endif

#define BN_MAX_BATCH 24

// Batch matvec with pre-quantized Q8_K input. Dispatches 4-row kernels
// directly without re-quantizing. Falls back to float path for non-k-quant types.
void bn_quant_matvec_batch_preq8k(const BnMatvecTask *tasks, int n_tasks,
                                  const int8_t *x_q, const float *x_d,
                                  const int16_t *x_bsums, const float *x_float,
                                  BnThreadPool *pool) {
    if (n_tasks <= 0) return;

#ifdef __AVX2__
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
            BnKQuantSdotCtx ctxs[BN_MAX_BATCH];
            BnTPTask tp_tasks[BN_MAX_BATCH];
            for (int t = 0; t < n_tasks; t++) {
                ctxs[t] = (BnKQuantSdotCtx){
                    tasks[t].out, tasks[t].W,
                    (int8_t *)x_q, (float *)x_d, (int16_t *)x_bsums
                };
                bn_tp_fn fn = (tasks[t].W->type == BN_GGUF_TENSOR_Q4_K)
                    ? (bn_tp_fn)bn_quant_q4k_avx2_4row_range
                    : (bn_tp_fn)bn_quant_q6k_avx2_4row_range;
                int n_groups = (tasks[t].W->rows + 3) / 4;
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
                                 (int8_t *)x_q, pool, tasks[t].prepared);
        }
    }
}
