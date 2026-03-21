#ifndef BN_THREADPOOL_H
#define BN_THREADPOOL_H

// Persistent pthread thread pool with atomic work-stealing dispatch.
// Threads grab chunks of rows via atomic_fetch_add for load balancing.

typedef void (*bn_tp_fn)(void *ctx, int start, int end);

typedef struct {
    bn_tp_fn fn;    // range function: called with [start, end)
    void *ctx;      // opaque context pointer
    int   n;        // iteration count
} BnTPTask;

typedef struct BnThreadPool BnThreadPool;

// Create a thread pool with n_workers background threads.
// Main thread participates as thread 0 (not counted in n_workers).
BnThreadPool *bn_tp_create(int n_workers);

// Destroy the thread pool, joining all worker threads.
void bn_tp_free(BnThreadPool *pool);

// Dispatch tasks to the pool. Blocks until all tasks complete.
// If pool is NULL, runs serially on the calling thread.
// Threads steal work in chunks via atomic counters for load balancing.
void bn_tp_dispatch(BnThreadPool *pool, BnTPTask *tasks, int n_tasks);

// Returns total thread count (n_workers + 1 for main thread).
int bn_tp_num_threads(const BnThreadPool *pool);

#endif // BN_THREADPOOL_H
