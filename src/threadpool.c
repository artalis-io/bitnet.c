#include "threadpool.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

typedef struct {
    ThreadPool *pool;
    int tid;
} WorkerArg;

struct ThreadPool {
    pthread_t    *threads;
    int           n_workers;   // background threads
    int           n_threads;   // n_workers + 1 (main)
    const TPTask *tasks;
    int           n_tasks;
    pthread_mutex_t mtx;
    pthread_cond_t  work_cond;
    pthread_cond_t  done_cond;
    int64_t       generation;
    int           n_done;
    int           shutdown;
    int           dispatching; // reentrancy guard
};

// Execute all tasks for a given thread id
static void tp_execute(const ThreadPool *pool, int tid) {
    int nt = pool->n_threads;
    for (int t = 0; t < pool->n_tasks; t++) {
        int n = pool->tasks[t].n;
        int start = tid * n / nt;
        int end   = (tid + 1) * n / nt;
        if (start < end) {
            pool->tasks[t].fn(pool->tasks[t].ctx, start, end);
        }
    }
}

static void *worker_loop(void *arg) {
    WorkerArg *wa = (WorkerArg *)arg;
    ThreadPool *pool = wa->pool;
    int tid = wa->tid;
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
        tp_execute(pool, tid);

        // Signal completion
        pthread_mutex_lock(&pool->mtx);
        pool->n_done++;
        if (pool->n_done == pool->n_workers) {
            pthread_cond_signal(&pool->done_cond);
        }
        pthread_mutex_unlock(&pool->mtx);
    }
}

ThreadPool *tp_create(int n_workers) {
    if (n_workers <= 0) return NULL;

    ThreadPool *pool = (ThreadPool *)calloc(1, sizeof(ThreadPool));
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

void tp_free(ThreadPool *pool) {
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

void tp_dispatch(ThreadPool *pool, const TPTask *tasks, int n_tasks) {
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

    assert(!pool->dispatching && "tp_dispatch is not reentrant");
    pool->dispatching = 1;

    // Set up work and wake workers
    pthread_mutex_lock(&pool->mtx);
    pool->tasks = tasks;
    pool->n_tasks = n_tasks;
    pool->n_done = 0;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->mtx);

    // Main thread does its share (tid 0)
    tp_execute(pool, 0);

    // Wait for workers to finish
    pthread_mutex_lock(&pool->mtx);
    while (pool->n_done < pool->n_workers) {
        pthread_cond_wait(&pool->done_cond, &pool->mtx);
    }
    pthread_mutex_unlock(&pool->mtx);

    pool->dispatching = 0;
}

int tp_num_threads(const ThreadPool *pool) {
    return pool ? pool->n_threads : 1;
}
