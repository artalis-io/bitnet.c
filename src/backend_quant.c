#include "backend_quant.h"
#include "quant_dispatch_internal.h"
#include <math.h>
#include <stdlib.h>

static int backend_quant_env_top_n(const char *name, int min_value) {
    const char *env = getenv(name);
    if (!env)
        return 0;
    int top_n = atoi(env);
    if (top_n < min_value)
        return 0;
    return top_n > 128 ? 128 : top_n;
}

int bn_backend_quant_cpu_tied_kquant_refine_top(void) {
    return backend_quant_env_top_n("BN_CPU_TIED_Q6K_REFINE_TOP", 1);
}

int bn_backend_quant_cpu_tied_kquant_hybrid_top(void) {
    return backend_quant_env_top_n("BN_CPU_TIED_Q6K_HYBRID_TOP", 2);
}

void bn_backend_quant_prepare_kquant_activation(const float *x,
                                                int8_t *quantized,
                                                float *scales,
                                                int16_t *block_sums,
                                                int n) {
    bn_quant_x_to_q8k(x, quantized, scales, block_sums, n);
}

void bn_backend_quant_prepare_kquant_activation_scalar(const float *x,
                                                       int8_t *quantized,
                                                       float *scales,
                                                       int16_t *block_sums,
                                                       int n) {
    bn_quant_x_to_q8k_scalar(x, quantized, scales, block_sums, n);
}

void bn_backend_quant_rmsnorm_prepared_kquant_avx2(
    const float *x,
    const float *w,
    int dim,
    float eps,
    float *out,
    int8_t *quantized,
    float *scales,
    int16_t *block_sums) {
#ifdef __AVX2__
    bn_quant_rmsnorm_q8k_avx2(x, w, dim, eps, out, quantized, scales,
                              block_sums);
#else
    float ss = 0.0f;
    for (int i = 0; i < dim; i++)
        ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int i = 0; i < dim; i++)
        out[i] = w[i] * (x[i] * inv);
    bn_backend_quant_prepare_kquant_activation_scalar(
        out, quantized, scales, block_sums, dim);
#endif
}

int bn_backend_quant_refine_kquant_logits_prepared_activation_row(
    const BnQWeight *weight,
    const int8_t *quantized,
    const float *scales,
    const int16_t *block_sums,
    int row,
    float *out) {
    return bn_quant_q6_logits_refine_q8k_row(weight, quantized, scales,
                                             block_sums, row, out);
}

int bn_backend_quant_refine_native_quant_logits_row(
    const BnQWeight *weight,
    const int8_t *quantized,
    const float *scales,
    int row,
    float *out) {
    return bn_quant_q8_logits_refine_row(weight, quantized, scales, row, out);
}

int bn_backend_quant_refine_kquant_logits_row(const BnQWeight *weight,
                                              const float *x,
                                              int row,
                                              float *out) {
    return bn_quant_q6_logits_refine_row(weight, x, row, out);
}

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

static void bn_backend_quant_matmul_batch_cpu(const BnMatvecTask *tasks,
                                              int n_tasks,
                                              const float *X,
                                              int n_tokens,
                                              int8_t *x_q_buf,
                                              BnThreadPool *pool) {
    for (int i = 0; i < n_tasks; i++)
        bn_quant_matmul(tasks[i].out, tasks[i].W, X, n_tokens, x_q_buf,
                        pool);
}

void bn_backend_quant_matmul_batch_gpu_buf(const BnMatvecTask *tasks,
                                           const void *const *W_bufs,
                                           int n_tasks, const float *X,
                                           int n_tokens, int x_cols,
                                           int8_t *x_q_buf,
                                           BnThreadPool *pool,
                                           BnGPUBackend *gpu) {
    if (gpu && gpu->matmul_batch && W_bufs && n_tasks > 1 && n_tasks <= 16 &&
        n_tokens > 1) {
        BnGPUMatvecOp ops[16];
        int all_gpu = 1;
        for (int i = 0; i < n_tasks; i++) {
            if (!W_bufs[i] || tasks[i].W->cols != x_cols) {
                all_gpu = 0;
                break;
            }
            ops[i] = (BnGPUMatvecOp){
                .out = tasks[i].out,
                .W_buf = (void *)W_bufs[i],
                .rows = tasks[i].W->rows,
                .cols = tasks[i].W->cols,
                .type = tasks[i].W->type,
            };
        }
        if (all_gpu &&
            gpu->matmul_batch(gpu->ctx, ops, n_tasks, X, n_tokens,
                              x_cols) == 0)
            return;
    }
    bn_backend_quant_matmul_batch_cpu(tasks, n_tasks, X, n_tokens, x_q_buf,
                                      pool);
}

void bn_backend_quant_matmul_gpu(float *out, const BnQWeight *W,
                                 const float *X, int n_tokens,
                                 int8_t *x_q_buf, BnThreadPool *pool,
                                 BnGPUBackend *gpu) {
    bn_backend_quant_matmul_gpu_buf(out, W, NULL, X, n_tokens, x_q_buf,
                                    pool, gpu);
}

void bn_backend_quant_matvec_gpu_buf_prepared(float *out, const BnQWeight *W,
                                              const BnPreparedWeight *prepared,
                                              void *W_buf, const float *x,
                                              int8_t *x_q_buf,
                                              BnThreadPool *pool,
                                              BnGPUBackend *gpu) {
    if (gpu && W_buf && gpu->matvec) {
        if (gpu->matvec(gpu->ctx, out, W_buf, x,
                        W->rows, W->cols, W->type) == 0)
            return;
    }
    bn_quant_matvec_prepared(out, W, prepared, x, x_q_buf, pool);
}

void bn_backend_quant_matvec_gpu_buf(float *out, const BnQWeight *W,
                                     void *W_buf, const float *x,
                                     int8_t *x_q_buf, BnThreadPool *pool,
                                     BnGPUBackend *gpu) {
    bn_backend_quant_matvec_gpu_buf_prepared(out, W, NULL, W_buf, x, x_q_buf,
                                             pool, gpu);
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
