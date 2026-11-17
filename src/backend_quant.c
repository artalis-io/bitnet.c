#include "backend_quant.h"
#include "quant_dispatch_internal.h"
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

static int backend_quant_ranges_overlap(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (na > UINTPTR_MAX - x || nb > UINTPTR_MAX - y) return 1;
    return x < y + nb && y < x + na;
}

size_t bn_backend_quant_host_row_stride(const BnQWeight *w) {
    const BnQuantFormatOps *f = w ? bn_quant_format_ops(w->type) : NULL;
    if (!f || !w->data || w->rows <= 0 || w->cols <= 0 || w->scale != 1.0f ||
        f->layout == BN_QUANT_LAYOUT_I2S || f->block_elems <= 0 ||
        !f->bytes_per_block || w->cols % f->block_elems ||
        (size_t)(w->cols / f->block_elems) > (SIZE_MAX - 15) / f->bytes_per_block) return 0;
    size_t row = (size_t)(w->cols / f->block_elems) * f->bytes_per_block;
    if ((size_t)w->rows > SIZE_MAX / row) return 0;
    return (row + 15) & ~(size_t)15;
}

int bn_backend_quant_host_rows_bf16_available(const BnQWeight *w) {
    const BnQuantFormatOps *f = w ? bn_quant_format_ops(w->type) : NULL;
    return f && f->bf16_decode == BN_QUANT_BF16_DECODE_IQ4_NL &&
        bn_backend_quant_host_row_stride(w) != 0;
}

int bn_backend_quant_gather_host_rows(const BnQWeight *w,
    const int32_t *ids, size_t n, void *out, size_t bytes) {
    size_t stride = bn_backend_quant_host_row_stride(w);
    if (!stride || !ids || !n || !out || n > SIZE_MAX / stride ||
        n > SIZE_MAX / sizeof(*ids) || bytes != n * stride) return -1;
    const BnQuantFormatOps *f = bn_quant_format_ops(w->type);
    size_t row = (size_t)(w->cols / f->block_elems) * f->bytes_per_block;
    if (backend_quant_ranges_overlap(out, bytes, w->data, (size_t)w->rows * row) ||
        backend_quant_ranges_overlap(out, bytes, ids, n * sizeof(*ids)) ||
        backend_quant_ranges_overlap(out, bytes, w, sizeof(*w))) return -1;
    for (size_t i = 0; i < n; i++) if (ids[i] < 0 || ids[i] >= w->rows) return -1;
    for (size_t i = 0; i < n; i++) {
        unsigned char *dst = (unsigned char *)out + i * stride;
        memcpy(dst, (const unsigned char *)w->data + (size_t)ids[i] * row, row);
        memset(dst + row, 0, stride - row);
    }
    return 0;
}

int bn_backend_quant_bf16_matmul_plan(int type, int rows, int cols,
    int n_tokens, BnBackendBF16MatmulPlan *plan) {
    const BnQuantFormatOps *format = bn_quant_format_ops(type);
    if (!plan || !format || rows <= 0 || cols <= 0 || n_tokens <= 0 ||
        n_tokens > INT_MAX / rows || n_tokens > INT_MAX / cols ||
        format->block_elems <= 0 || cols % format->block_elems ||
        (format->bf16_linear != BN_QUANT_BF16_LINEAR_DENSE &&
         format->bf16_linear != BN_QUANT_BF16_LINEAR_IQ4 &&
         format->bf16_linear != BN_QUANT_BF16_LINEAR_IQ3_S &&
         format->bf16_linear != BN_QUANT_BF16_LINEAR_Q3_K &&
         format->bf16_linear != BN_QUANT_BF16_LINEAR_Q8_0 &&
         format->bf16_linear != BN_QUANT_BF16_LINEAR_Q5_K &&
         format->bf16_linear != BN_QUANT_BF16_LINEAR_Q4_K &&
         format->bf16_linear != BN_QUANT_BF16_LINEAR_Q6_K) ||
        (size_t)rows > SIZE_MAX / sizeof(uint16_t) / (size_t)cols)
        return -1;
    BnBackendBF16MatmulPlan result;
    int a8 = format->bf16_linear == BN_QUANT_BF16_LINEAR_Q3_K ||
        format->bf16_linear == BN_QUANT_BF16_LINEAR_Q8_0 ||
        format->bf16_linear == BN_QUANT_BF16_LINEAR_Q5_K ||
        format->bf16_linear == BN_QUANT_BF16_LINEAR_Q4_K ||
        format->bf16_linear == BN_QUANT_BF16_LINEAR_Q6_K;
    int mmvq_limit = a8 ? (rows > 5120 ? 2 : 6) : (rows > 5120 ? 8 : 16);
    if (format->bf16_linear == BN_QUANT_BF16_LINEAR_DENSE) {
        result.kind = BN_BACKEND_BF16_MATMUL_DECODED_GEMM;
        result.decoded_weight_bytes = (size_t)rows * cols * sizeof(uint16_t);
    } else if (n_tokens <= mmvq_limit) {
        result.kind = BN_BACKEND_BF16_MATMUL_A8_MMVQ;
        result.decoded_weight_bytes = 0;
    } else if (a8) {
        result.kind = BN_BACKEND_BF16_MATMUL_A8_MMQ;
        result.decoded_weight_bytes = 0;
    } else {
        result.kind = BN_BACKEND_BF16_MATMUL_DECODED_GEMM;
        result.decoded_weight_bytes = (size_t)rows * cols * sizeof(uint16_t);
    }
    *plan = result;
    return 0;
}

void bn_backend_quant_prepare_kquant_activation(const float *x,
                                                int8_t *quantized,
                                                float *scales,
                                                int16_t *block_sums,
                                                int n) {
#ifdef BN_FORCE_SCALAR
    bn_quant_x_to_q8k_scalar(x, quantized, scales, block_sums, n);
#else
    bn_quant_x_to_q8k(x, quantized, scales, block_sums, n);
#endif
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
    if (W_buf && (n_tokens > 1 ||
                  (n_tokens == 1 && bn_gpu_backend_is_cuda(gpu)))) {
        if (bn_gpu_backend_matmul(gpu, out, W_buf, X, W->rows, W->cols,
                                  n_tokens, W->type) == 0)
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
    if (bn_gpu_backend_can_matmul_batch(gpu) && W_bufs && n_tasks > 1 &&
        n_tasks <= 16 &&
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
            bn_gpu_backend_matmul_batch(gpu, ops, n_tasks, X, n_tokens,
                                        x_cols) == 0)
            return;
    }
    if (gpu && W_bufs) {
        /* A declined batch does not imply its individual projections are
         * unsupported. Preserve backend arithmetic wherever possible. */
        for (int i = 0; i < n_tasks; i++) {
            void *buffer = tasks[i].W->cols == x_cols
                ? (void *)W_bufs[i] : NULL;
            bn_backend_quant_matmul_gpu_buf(tasks[i].out, tasks[i].W,
                                             buffer, X, n_tokens, x_q_buf,
                                             pool, gpu);
        }
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
    if (W_buf) {
        if (bn_gpu_backend_matvec(gpu, out, W_buf, x, W->rows, W->cols,
                                  W->type) == 0)
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
            if (bn_gpu_backend_can_matvec_batch(gpu) && n_tasks <= 16) {
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
                if (bn_gpu_backend_matvec_batch(gpu, ops, n_tasks, x,
                                                tasks[0].W->cols) == 0)
                    return;
            }
            if (bn_gpu_backend_can_matvec(gpu)) {
                for (int t = 0; t < n_tasks; t++) {
                    const BnQWeight *W = tasks[t].W;
                    if (bn_gpu_backend_matvec(gpu, tasks[t].out,
                                              (void *)W_bufs[t], x,
                                              W->rows, W->cols,
                                              W->type) != 0) {
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
