#include "threadpool.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <limits.h>
#include <assert.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

// Chunk size for atomic work-stealing.
// Large chunks preserve memory locality (contiguous row access per thread).
// Stealing only kicks in for the last chunk when threads finish at different times.
#define TP_CHUNK_MIN 16

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
    int64_t       generation;
    int           n_done;
    int           shutdown;
    _Atomic int   dispatching; // reentrancy guard (main-thread-only, atomic for safety)
};

// Execute all tasks via atomic work-stealing with adaptive chunk size.
// Chunk = n / (4 * n_threads) — mostly static, stealing for tail imbalance.
static void tp_execute(BnThreadPool *pool) {
    int nt = pool->n_threads;
    for (int t = 0; t < pool->n_tasks; t++) {
        BnTPTask *task = &pool->tasks[t];
        int n = task->n;
        int nt4 = nt <= INT_MAX / 4 ? nt * 4 : nt;  // avoid overflow
        int chunk = n / nt4;
        if (chunk < TP_CHUNK_MIN) chunk = TP_CHUNK_MIN;
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
    (void)wa->tid;
    free(wa);

#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    int64_t my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&pool->mtx);
        while (pool->generation == my_gen && !pool->shutdown) {
            pthread_cond_wait(&pool->work_cond, &pool->mtx);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mtx);
            return NULL;
        }
        my_gen = pool->generation;
        pthread_mutex_unlock(&pool->mtx);

        // Do work
        tp_execute(pool);

        // Signal completion
        pthread_mutex_lock(&pool->mtx);
        pool->n_done++;
        if (pool->n_done == pool->n_workers) {
            pthread_cond_signal(&pool->done_cond);
        }
        pthread_mutex_unlock(&pool->mtx);
    }
}

BnThreadPool *bn_tp_create(int n_workers) {
    if (n_workers <= 0) return NULL;

    BnThreadPool *pool = (BnThreadPool *)calloc(1, sizeof(BnThreadPool));
    if (!pool) return NULL;

    pool->n_workers = n_workers;
    pool->n_threads = n_workers + 1;

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
    pool->n_done = 0;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->mtx);

    // Main thread does its share
    tp_execute(pool);

    // Wait for workers to finish
    pthread_mutex_lock(&pool->mtx);
    while (pool->n_done < pool->n_workers) {
        pthread_cond_wait(&pool->done_cond, &pool->mtx);
    }
    pthread_mutex_unlock(&pool->mtx);

    pool->dispatching = 0;
}

int bn_tp_num_threads(const BnThreadPool *pool) {
    return pool ? pool->n_threads : 1;
}
