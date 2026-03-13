#ifndef THREADPOOL_H
#define THREADPOOL_H

// Persistent pthread thread pool for parallel dispatch.
// Replaces OpenMP fork/join with ~2us condvar dispatch.

typedef void (*tp_fn)(void *ctx, int start, int end);

typedef struct {
    tp_fn fn;    // range function: called with [start, end)
    void *ctx;   // opaque context pointer
    int   n;     // iteration count
} TPTask;

typedef struct ThreadPool ThreadPool;

// Create a thread pool with n_workers background threads.
// Main thread participates as thread 0 (not counted in n_workers).
ThreadPool *tp_create(int n_workers);

// Destroy the thread pool, joining all worker threads.
void tp_free(ThreadPool *pool);

// Dispatch tasks to the pool. Blocks until all tasks complete.
// If pool is NULL, runs serially on the calling thread.
// Multi-task dispatch (2-3 tasks) uses a single wake/wait cycle.
void tp_dispatch(ThreadPool *pool, const TPTask *tasks, int n_tasks);

// Returns total thread count (n_workers + 1 for main thread).
int tp_num_threads(const ThreadPool *pool);

#endif // THREADPOOL_H
