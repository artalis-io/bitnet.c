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

typedef struct {
    int reference_dot;
    int reference_q4_dot;
    int reference_q6_dot;
    int disable_q4_dot;
    int disable_q6_dot;
    int avx512_kquant_vnni; /* -1 = shape default, 0/1 = override */
    int avx2_kquant_float;
    int q4_scalar_dot;
    int wasm_q4_canonical4;
    int disable_native_quant_matmul_batch;
} BnQuantRuntimePolicy;

#define BN_RUNTIME_POLICY_PATH_CAP 512
#define BN_RUNTIME_POLICY_TAG_CAP 96

typedef struct BnCPURuntimePolicy {
    int prepared_qweights;
    int reference_math;
    int reference_dot;
    int reference_kquant_dot;
    int native_tied_quant_logits;
    int tied_kquant_refine_top;
    int tied_kquant_hybrid_top;
    int prefill_profile;
    int prefill_hybrid_batch;
    int prefill_force_token_attention;
    int debug_dump_pos_set;
    int debug_dump_pos;
    int debug_dump_heads;
    int debug_binary_layer_set;
    int debug_binary_layer;
    char debug_dump_path[BN_RUNTIME_POLICY_PATH_CAP];
    char debug_binary_path[BN_RUNTIME_POLICY_PATH_CAP];
    char debug_binary_tag[BN_RUNTIME_POLICY_TAG_CAP];
} BnCPURuntimePolicy;

// Capture process configuration once at runtime construction. Computational
// quant policy consumes this immutable value and never reads the environment.
void bn_quant_runtime_policy_from_env(BnQuantRuntimePolicy *policy);
const BnQuantRuntimePolicy *bn_tp_quant_policy(const BnThreadPool *pool);
void bn_cpu_runtime_policy_from_env(BnCPURuntimePolicy *policy);
const BnCPURuntimePolicy *bn_tp_cpu_policy(const BnThreadPool *pool);

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

// Configure bounded busy-wait windows used between dispatches. Call only while
// the pool is idle. CPU-only decode benefits from longer windows; GPU runtimes
// should keep the defaults to avoid competing with backend submission work.
void bn_tp_set_poll_iters(BnThreadPool *pool, int large, int small);

// Select the platform-owned busy-wait policy for CPU token decode.
void bn_tp_set_cpu_decode_policy(BnThreadPool *pool);

#endif // BN_THREADPOOL_H
