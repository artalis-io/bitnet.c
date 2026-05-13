#include "backend_quant.h"
#include <stdlib.h>

void bn_backend_quant_matmul_gpu_buf(float *out, const BnQWeight *W,
                                     void *W_buf, const float *X,
                                     int n_tokens, int8_t *x_q_buf,
                                     BnThreadPool *pool, BnGPUBackend *gpu) {
    if (gpu && gpu->matmul && W_buf && n_tokens > 1) {
        if (gpu->matmul(gpu->ctx, out, W_buf, X,
                        W->rows, W->cols, n_tokens, W->type) == 0)
            return;
    }
    bn_quant_matmul(out, W, X, n_tokens, x_q_buf, pool);
}

void bn_backend_quant_matmul_gpu(float *out, const BnQWeight *W,
                                 const float *X, int n_tokens,
                                 int8_t *x_q_buf, BnThreadPool *pool,
                                 BnGPUBackend *gpu) {
    bn_backend_quant_matmul_gpu_buf(out, W, NULL, X, n_tokens, x_q_buf,
                                    pool, gpu);
}

void bn_backend_quant_matvec_gpu_buf(float *out, const BnQWeight *W,
                                     void *W_buf, const float *x,
                                     int8_t *x_q_buf, BnThreadPool *pool,
                                     BnGPUBackend *gpu) {
    if (gpu && W_buf && gpu->matvec) {
        if (gpu->matvec(gpu->ctx, out, W_buf, x,
                        W->rows, W->cols, W->type) == 0)
            return;
    }
    bn_quant_matvec(out, W, x, x_q_buf, pool);
}

void bn_backend_quant_matvec_gpu(float *out, const BnQWeight *W,
                                 const float *x, int8_t *x_q_buf,
                                 BnThreadPool *pool, BnGPUBackend *gpu) {
    bn_backend_quant_matvec_gpu_buf(out, W, NULL, x, x_q_buf, pool, gpu);
}

void bn_backend_quant_matvec_batch_gpu_buf(const BnMatvecTask *tasks,
                                           const void *const *W_bufs,
                                           int n_tasks, const float *x,
                                           int8_t *x_q_buf,
                                           BnThreadPool *pool,
                                           BnGPUBackend *gpu) {
    if (gpu) {
        int all_gpu = 1;
        for (int t = 0; t < n_tasks; t++) {
            if (!W_bufs || !W_bufs[t]) { all_gpu = 0; break; }
        }
        if (all_gpu) {
            if (gpu->matvec_batch && n_tasks <= 16) {
                BnGPUMatvecOp ops[16];
                for (int t = 0; t < n_tasks; t++) {
                    ops[t] = (BnGPUMatvecOp){
                        .out = tasks[t].out,
                        .W_buf = (void *)W_bufs[t],
                        .rows = tasks[t].W->rows,
                        .cols = tasks[t].W->cols,
                        .type = tasks[t].W->type,
                    };
                }
                if (gpu->matvec_batch(gpu->ctx, ops, n_tasks, x,
                                       tasks[0].W->cols) == 0)
                    return;
            }
            if (gpu->matvec) {
                for (int t = 0; t < n_tasks; t++) {
                    const BnQWeight *W = tasks[t].W;
                    if (gpu->matvec(gpu->ctx, tasks[t].out,
                                    (void *)W_bufs[t], x,
                                    W->rows, W->cols, W->type) != 0) {
                        bn_quant_matvec_batch(tasks, n_tasks, x, x_q_buf, pool);
                        return;
                    }
                }
                return;
            }
        }
    }
    bn_quant_matvec_batch(tasks, n_tasks, x, x_q_buf, pool);
}

void bn_backend_quant_matvec_batch_gpu(const BnMatvecTask *tasks, int n_tasks,
                                       const float *x, int8_t *x_q_buf,
                                       BnThreadPool *pool,
                                       BnGPUBackend *gpu) {
    const void *bufs_inline[16];
    const void **bufs = bufs_inline;
    const void **heap_bufs = NULL;
    if (n_tasks > 16) {
        heap_bufs = (const void **)malloc((size_t)n_tasks * sizeof(void *));
        bufs = heap_bufs;
    }
    if (bufs) {
        for (int t = 0; t < n_tasks; t++)
            bufs[t] = NULL;
    }
    bn_backend_quant_matvec_batch_gpu_buf(tasks, bufs, n_tasks, x, x_q_buf,
                                          pool, gpu);
    free(heap_bufs);
}
