#include "threadpool.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <limits.h>
#include <assert.h>
#include <string.h>

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

// Chunk size for atomic work-stealing.
// Large chunks preserve memory locality (contiguous row access per thread).
// Stealing only kicks in for the last chunk when threads finish at different times.
#define TP_CHUNK_MIN 32
#define TP_CHUNK_MIN_LARGE 128
#if defined(__APPLE__) && defined(__aarch64__)
#define TP_POLL_ITERS_LARGE 50000
#define TP_POLL_ITERS_SMALL 50000
#define TP_CPU_DECODE_POLL_ITERS 200000
#else
#define TP_POLL_ITERS_LARGE 5000
#define TP_POLL_ITERS_SMALL 5000
#define TP_CPU_DECODE_POLL_ITERS 5000
#endif

typedef struct {
    BnThreadPool *pool;
    int tid;
} WorkerArg;

#define TP_MAX_TASKS 32  // max concurrent tasks per dispatch

struct BnThreadPool {
    pthread_t    *threads;
    int           n_workers;   // background threads
    int           n_threads;   // n_workers + 1 (main)
    BnTPTask     *tasks;
    int           n_tasks;
    _Atomic int   cursors[TP_MAX_TASKS];  // atomic work-stealing cursors
    pthread_mutex_t mtx;
    pthread_cond_t  work_cond;
    pthread_cond_t  done_cond;
    _Atomic int64_t generation;
    _Atomic int   n_done;
    int           shutdown;
    _Atomic int   poll_iters;
    int           poll_iters_large;
    int           poll_iters_small;
    _Atomic int   dispatching; // reentrancy guard (main-thread-only, atomic for safety)
    BnQuantRuntimePolicy quant_policy;
    BnCPURuntimePolicy cpu_policy;
};

static int tp_env_enabled(const char *a, const char *b, const char *c) {
    return (a && getenv(a)) || (b && getenv(b)) || (c && getenv(c));
}

void bn_quant_runtime_policy_from_env(BnQuantRuntimePolicy *policy) {
    if (!policy) return;
    *policy = (BnQuantRuntimePolicy){ .avx512_kquant_vnni = -1 };
    policy->reference_dot =
        tp_env_enabled("BN_CPU_REFERENCE_DOT", "BN_CPU_LLAMA_DOT", NULL);
    policy->reference_q4_dot = tp_env_enabled(
        "BN_CPU_REFERENCE_BLOCK_QUANT_DOT", "BN_CPU_REFERENCE_Q4_DOT",
        "BN_CPU_LLAMA_Q4_DOT");
    policy->reference_q6_dot = tp_env_enabled(
        "BN_CPU_REFERENCE_KQUANT_DOT", "BN_CPU_REFERENCE_Q6_DOT",
        "BN_CPU_LLAMA_Q6_DOT");
    policy->disable_q4_dot = tp_env_enabled(
        "BN_CPU_DISABLE_Q4_DOT", "BN_CPU_BLOCK_QUANT_FLOAT", NULL);
    policy->disable_q6_dot = tp_env_enabled(
        "BN_CPU_DISABLE_Q6_DOT", "BN_CPU_KQUANT_FLOAT", NULL);
    const char *vnni = getenv("BN_AVX512_KQUANT_VNNI");
    if (!vnni) vnni = getenv("BN_AVX512_Q5K_VNNI");
    if (vnni) policy->avx512_kquant_vnni = vnni[0] != '\0' && vnni[0] != '0';
    const char *avx2 = getenv("BN_AVX2_KQUANT_FLOAT");
    policy->avx2_kquant_float = avx2 && avx2[0] != '\0' && avx2[0] != '0';
    policy->q4_scalar_dot = tp_env_enabled(
        "BN_CPU_Q4_SCALAR_DOT", "BN_CPU_REFERENCE_Q4_SCALAR_DOT", NULL);
    policy->wasm_q4_canonical4 = tp_env_enabled(
        "BN_WASM_BLOCK_QUANT_CANONICAL4", "BN_WASM_Q4_CANONICAL4", NULL);
    policy->disable_native_quant_matmul_batch = tp_env_enabled(
        "BN_DISABLE_NATIVE_QUANT_MATMUL_BATCH",
        "BN_DISABLE_Q8_0_MATMUL_BATCH", NULL);
}

const BnQuantRuntimePolicy *bn_tp_quant_policy(const BnThreadPool *pool) {
    return pool ? &pool->quant_policy : NULL;
}

static void tp_copy_env(char *dst, size_t capacity, const char *name) {
    const char *value = getenv(name);
    if (!dst || capacity == 0) return;
    if (!value || !value[0]) {
        dst[0] = '\0';
        return;
    }
    size_t n = strlen(value);
    if (n >= capacity) n = capacity - 1;
    memcpy(dst, value, n);
    dst[n] = '\0';
}

static int tp_env_int(const char *name, int *is_set) {
    const char *value = getenv(name);
    if (is_set) *is_set = value && value[0];
    return value && value[0] ? atoi(value) : 0;
}

static int tp_compat_top_n(const char *name, const char *compat_name,
                           int min_value) {
    const char *value = getenv(name);
    if (!value) value = getenv(compat_name);
    if (!value) return 0;
    int top_n = atoi(value);
    if (top_n < min_value) return 0;
    return top_n > 128 ? 128 : top_n;
}

void bn_cpu_runtime_policy_from_env(BnCPURuntimePolicy *policy) {
    if (!policy) return;
    *policy = (BnCPURuntimePolicy){ .prepared_qweights = 1 };
    policy->prepared_qweights = !tp_env_enabled(
        "BN_CPU_DISABLE_PREPARED_QWEIGHTS", NULL, NULL);
    policy->reference_math = tp_env_enabled(
        "BN_CPU_REFERENCE_MATH", "BN_CPU_SCALAR_TRANSFORMER_MATH", NULL);
    policy->reference_dot = tp_env_enabled(
        "BN_CPU_REFERENCE_DOT", "BN_CPU_LLAMA_DOT", NULL);
    policy->reference_kquant_dot = tp_env_enabled(
        "BN_CPU_REFERENCE_BLOCK_QUANT_DOT", "BN_CPU_REFERENCE_Q4_DOT",
        "BN_CPU_LLAMA_Q4_DOT");
    policy->native_tied_quant_logits = tp_env_enabled(
        "BN_CPU_ENABLE_NATIVE_QUANT_TIED_LOGITS",
        "BN_CPU_NATIVE_TIED_LOGITS", NULL);
    policy->tied_kquant_refine_top = tp_compat_top_n(
        "BN_CPU_TIED_KQUANT_REFINE_TOP", "BN_CPU_TIED_Q6K_REFINE_TOP", 1);
    policy->tied_kquant_hybrid_top = tp_compat_top_n(
        "BN_CPU_TIED_KQUANT_HYBRID_TOP", "BN_CPU_TIED_Q6K_HYBRID_TOP", 2);
    policy->prefill_profile = tp_env_enabled("BN_PREFILL_PROFILE", NULL, NULL);
    policy->prefill_hybrid_batch = tp_env_enabled(
        "BN_PREFILL_ALLOW_HYBRID_BATCH", NULL, NULL);
    policy->prefill_force_token_attention = tp_env_enabled(
        "BN_PREFILL_FORCE_TOKEN_ATTN", NULL, NULL);
    policy->debug_dump_heads = tp_env_enabled("BN_DUMP_ALL_HEADS", NULL, NULL);
    policy->debug_dump_pos = tp_env_int(
        "BN_DUMP_LAYER_POS", &policy->debug_dump_pos_set);
    policy->debug_binary_layer = tp_env_int(
        "BN_DUMP_BINARY_LAYER", &policy->debug_binary_layer_set);
    tp_copy_env(policy->debug_dump_path, sizeof(policy->debug_dump_path),
                "BN_DUMP_LAYER_INP");
    tp_copy_env(policy->debug_binary_path, sizeof(policy->debug_binary_path),
                "BN_DUMP_BINARY_PATH");
    tp_copy_env(policy->debug_binary_tag, sizeof(policy->debug_binary_tag),
                "BN_DUMP_BINARY_TAG");
}

const BnCPURuntimePolicy *bn_tp_cpu_policy(const BnThreadPool *pool) {
    return pool ? &pool->cpu_policy : NULL;
}

static inline void tp_cpu_relax(void) {
#if defined(__i386__) || defined(__x86_64__)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield" ::: "memory");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

// Execute all tasks via atomic work-stealing with adaptive chunk size.
// Chunk = n / (4 * n_threads) — mostly static, stealing for tail imbalance.
static void tp_execute(BnThreadPool *pool, int task_offset) {
    int nt = pool->n_threads;
    for (int i = 0; i < pool->n_tasks; i++) {
        int t = (i + task_offset) % pool->n_tasks;
        BnTPTask *task = &pool->tasks[t];
        int n = task->n;
        int nt4 = nt <= INT_MAX / 4 ? nt * 4 : nt;  // avoid overflow
        int chunk = n / nt4;
        int min_chunk = (n >= 4096) ? TP_CHUNK_MIN_LARGE : TP_CHUNK_MIN;
        if (n <= nt4) {
            chunk = 1;
        } else if (chunk < min_chunk) {
            chunk = min_chunk;
        }
        for (;;) {
            int start = atomic_fetch_add_explicit(&pool->cursors[t], chunk,
                                                   memory_order_relaxed);
            if (start >= n) break;
            int end = start + chunk;
            if (end > n) end = n;
            task->fn(task->ctx, start, end);
        }
    }
}

static void *worker_loop(void *arg) {
    WorkerArg *wa = (WorkerArg *)arg;
    BnThreadPool *pool = wa->pool;
    int tid = wa->tid;
    free(wa);

#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    int64_t my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&pool->mtx);
        while (atomic_load_explicit(&pool->generation, memory_order_acquire) == my_gen && !pool->shutdown) {
            pthread_cond_wait(&pool->work_cond, &pool->mtx);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mtx);
            return NULL;
        }
        my_gen = atomic_load_explicit(&pool->generation, memory_order_acquire);
        pthread_mutex_unlock(&pool->mtx);

        for (;;) {
            // Do work
            tp_execute(pool, tid);

            // Signal completion
            int done = atomic_fetch_add_explicit(&pool->n_done, 1, memory_order_acq_rel) + 1;
            if (done == pool->n_workers) {
                pthread_mutex_lock(&pool->mtx);
                pthread_cond_signal(&pool->done_cond);
                pthread_mutex_unlock(&pool->mtx);
            }

            int picked_up = 0;
            int poll_iters = atomic_load_explicit(&pool->poll_iters, memory_order_relaxed);
            for (int spin = 0; spin < poll_iters; spin++) {
                int64_t next_gen = atomic_load_explicit(&pool->generation, memory_order_acquire);
                if (next_gen != my_gen) {
                    my_gen = next_gen;
                    picked_up = 1;
                    break;
                }
                tp_cpu_relax();
            }
            if (!picked_up) break;
        }
    }
}

BnThreadPool *bn_tp_create(int n_workers) {
    if (n_workers <= 0) return NULL;

    BnThreadPool *pool = (BnThreadPool *)calloc(1, sizeof(BnThreadPool));
    if (!pool) return NULL;

    pool->n_workers = n_workers;
    pool->n_threads = n_workers + 1;
    pool->poll_iters_large = TP_POLL_ITERS_LARGE;
    pool->poll_iters_small = TP_POLL_ITERS_SMALL;
    bn_quant_runtime_policy_from_env(&pool->quant_policy);
    bn_cpu_runtime_policy_from_env(&pool->cpu_policy);

    pthread_mutex_init(&pool->mtx, NULL);
    pthread_cond_init(&pool->work_cond, NULL);
    pthread_cond_init(&pool->done_cond, NULL);

    pool->threads = (pthread_t *)calloc(n_workers, sizeof(pthread_t));
    if (!pool->threads) {
        pthread_mutex_destroy(&pool->mtx);
        pthread_cond_destroy(&pool->work_cond);
        pthread_cond_destroy(&pool->done_cond);
        free(pool);
        return NULL;
    }

    int created = 0;
    for (int i = 0; i < n_workers; i++) {
        WorkerArg *wa = (WorkerArg *)malloc(sizeof(WorkerArg));
        if (!wa) goto fail;
        wa->pool = pool;
        wa->tid = i + 1;  // main thread is tid 0
        if (pthread_create(&pool->threads[i], NULL, worker_loop, wa) != 0) {
            free(wa);
            goto fail;
        }
        created++;
    }

    return pool;

fail:
    // Shut down already-created threads
    pthread_mutex_lock(&pool->mtx);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->mtx);
    for (int i = 0; i < created; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    pthread_mutex_destroy(&pool->mtx);
    pthread_cond_destroy(&pool->work_cond);
    pthread_cond_destroy(&pool->done_cond);
    free(pool->threads);
    free(pool);
    return NULL;
}

void bn_tp_free(BnThreadPool *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mtx);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->mtx);

    for (int i = 0; i < pool->n_workers; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    pthread_mutex_destroy(&pool->mtx);
    pthread_cond_destroy(&pool->work_cond);
    pthread_cond_destroy(&pool->done_cond);
    free(pool->threads);
    free(pool);
}

void bn_tp_dispatch(BnThreadPool *pool, BnTPTask *tasks, int n_tasks) {
    if (n_tasks <= 0) return;

    // Serial fallback when no pool
    if (!pool) {
        for (int t = 0; t < n_tasks; t++) {
            if (tasks[t].n > 0) {
                tasks[t].fn(tasks[t].ctx, 0, tasks[t].n);
            }
        }
        return;
    }

    assert(!pool->dispatching && "bn_tp_dispatch is not reentrant");
    pool->dispatching = 1;

    // Initialize atomic cursors (pool-internal storage)
    assert(n_tasks <= TP_MAX_TASKS && "too many tasks for pool cursor array");
    for (int t = 0; t < n_tasks; t++)
        atomic_store_explicit(&pool->cursors[t], 0, memory_order_relaxed);

    // Set up work and wake workers
    pthread_mutex_lock(&pool->mtx);
    pool->tasks = tasks;
    pool->n_tasks = n_tasks;
    atomic_store_explicit(&pool->n_done, 0, memory_order_release);
    int max_n = 0;
    for (int t = 0; t < n_tasks; t++) {
        if (tasks[t].n > max_n) max_n = tasks[t].n;
    }
    atomic_store_explicit(&pool->poll_iters,
                          max_n >= 1024 ? pool->poll_iters_large
                                        : pool->poll_iters_small,
                          memory_order_relaxed);
    atomic_fetch_add_explicit(&pool->generation, 1, memory_order_release);
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->mtx);

    // Main thread does its share
    tp_execute(pool, pool->n_workers);

    // Wait for workers to finish
    int poll_iters = atomic_load_explicit(&pool->poll_iters, memory_order_relaxed);
    for (int spin = 0; spin < poll_iters; spin++) {
        if (atomic_load_explicit(&pool->n_done, memory_order_acquire) >= pool->n_workers) {
            pool->dispatching = 0;
            return;
        }
        tp_cpu_relax();
    }

    pthread_mutex_lock(&pool->mtx);
    while (atomic_load_explicit(&pool->n_done, memory_order_acquire) < pool->n_workers) {
        pthread_cond_wait(&pool->done_cond, &pool->mtx);
    }
    pthread_mutex_unlock(&pool->mtx);

    pool->dispatching = 0;
}

int bn_tp_num_threads(const BnThreadPool *pool) {
    return pool ? pool->n_threads : 1;
}

void bn_tp_set_poll_iters(BnThreadPool *pool, int large, int small) {
    if (!pool || large < 0 || small < 0) return;
    assert(!pool->dispatching && "threadpool policy changed during dispatch");
    pool->poll_iters_large = large;
    pool->poll_iters_small = small;
}

void bn_tp_set_cpu_decode_policy(BnThreadPool *pool) {
    bn_tp_set_poll_iters(pool, TP_CPU_DECODE_POLL_ITERS,
                         TP_CPU_DECODE_POLL_ITERS);
}
