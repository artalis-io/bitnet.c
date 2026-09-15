#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdatomic.h>

typedef struct {
    _Atomic int hits[50];
    _Atomic int max_span;
    int n;
} FineRangeCtx;

static void record_fine_range(void *ctx, int start, int end) {
    FineRangeCtx *c = ctx;
    assert(start >= 0 && start < end && end <= c->n);
    int span = end - start;
    int old = atomic_load(&c->max_span);
    while (old < span &&
           !atomic_compare_exchange_weak(&c->max_span, &old, span)) {}
    for (int i = start; i < end; i++) atomic_fetch_add(&c->hits[i + 1], 1);
}

static void test_fine_dispatch_reset(void) {
    printf("test_fine_dispatch_reset... ");
    const int workers[] = {-1, 0, 3, 7};
    for (size_t w = 0; w < sizeof(workers) / sizeof(workers[0]); w++) {
        BnThreadPool *pool = workers[w] < 0 ? NULL : bn_tp_create(workers[w]);
        assert(workers[w] < 0 || pool);
        FineRangeCtx c[2];
        for (int j = 0; j < 2; j++) {
            c[j].n = j ? 17 : 48;
            atomic_init(&c[j].max_span, 0);
            for (int i = 0; i < 50; i++) atomic_init(&c[j].hits[i], 0);
        }
        BnTPTask tasks[] = {{record_fine_range, &c[0], 48},
            {record_fine_range, &c[1], 17}, {record_fine_range, &c[0], 0}};
        for (int round = 0; round < 32; round++) {
            int fine = !(round % 2);
            for (int j = 0; j < 2; j++) {
                atomic_store(&c[j].max_span, 0);
                for (int i = 0; i < 50; i++) atomic_store(&c[j].hits[i], 0);
            }
            /* Empty dispatches must not execute a callback or leak policy. */
            bn_tp_dispatch_fine(pool, NULL, 0);
            if (fine) bn_tp_dispatch_fine(pool, tasks, 3);
            else bn_tp_dispatch(pool, tasks, 3);
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 50; i++)
                    assert(atomic_load(&c[j].hits[i]) == (i > 0 && i <= c[j].n));
                int span = atomic_load(&c[j].max_span);
                if (!pool) assert(span == c[j].n);
                else if (fine) assert(span == 1);
                else if (j == 0) assert(span > 1);
            }
        }
        bn_tp_free(pool);
    }
    printf("PASSED\n");
}

// --- Test serial dispatch (pool=NULL) ---

static void add_one(void *ctx, int start, int end) {
    int *arr = (int *)ctx;
    for (int i = start; i < end; i++) arr[i] += 1;
}

static void test_serial_dispatch(void) {
    printf("test_serial_dispatch... ");

    int arr[100];
    memset(arr, 0, sizeof(arr));

    BnTPTask task = { add_one, arr, 100 };
    bn_tp_dispatch(NULL, &task, 1);

    for (int i = 0; i < 100; i++) {
        assert(arr[i] == 1);
    }

    printf("PASSED\n");
}

// --- Test single-task dispatch with various thread counts ---

static void test_threaded_single_task(void) {
    printf("test_threaded_single_task... ");

    for (int nw = 1; nw <= 4; nw++) {
        BnThreadPool *pool = bn_tp_create(nw);
        assert(pool != NULL);
        assert(bn_tp_num_threads(pool) == nw + 1);

        int arr[256];
        memset(arr, 0, sizeof(arr));

        BnTPTask task = { add_one, arr, 256 };
        bn_tp_dispatch(pool, &task, 1);

        // Verify all elements were incremented exactly once
        for (int i = 0; i < 256; i++) {
            assert(arr[i] == 1);
        }

        bn_tp_free(pool);
    }

    printf("PASSED\n");
}

// --- Test multi-task dispatch ---

typedef struct {
    float *out;
    const float *a;
    const float *b;
    int len;
} VecAddCtx;

static void vec_add_range(void *ctx, int start, int end) {
    VecAddCtx *c = (VecAddCtx *)ctx;
    for (int i = start; i < end; i++) {
        c->out[i] = c->a[i] + c->b[i];
    }
}

static void vec_mul_range(void *ctx, int start, int end) {
    VecAddCtx *c = (VecAddCtx *)ctx;
    for (int i = start; i < end; i++) {
        c->out[i] = c->a[i] * c->b[i];
    }
}

static void test_multi_task_dispatch(void) {
    printf("test_multi_task_dispatch... ");

    BnThreadPool *pool = bn_tp_create(3);

    float a[128], b[128], sum_out[128], prod_out[128];
    for (int i = 0; i < 128; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    VecAddCtx sum_ctx = { sum_out, a, b, 128 };
    VecAddCtx prod_ctx = { prod_out, a, b, 128 };

    BnTPTask tasks[2] = {
        { vec_add_range, &sum_ctx, 128 },
        { vec_mul_range, &prod_ctx, 128 },
    };
    bn_tp_dispatch(pool, tasks, 2);

    for (int i = 0; i < 128; i++) {
        assert(sum_out[i] == (float)(i + i * 2));
        assert(prod_out[i] == (float)(i * i * 2));
    }

    bn_tp_free(pool);
    printf("PASSED\n");
}

// --- Test rapid successive dispatches (stress test generation counter) ---

static void test_rapid_dispatch(void) {
    printf("test_rapid_dispatch... ");

    BnThreadPool *pool = bn_tp_create(3);

    int arr[64];
    memset(arr, 0, sizeof(arr));

    BnTPTask task = { add_one, arr, 64 };

    // Dispatch 100 times rapidly
    for (int round = 0; round < 100; round++) {
        bn_tp_dispatch(pool, &task, 1);
    }

    for (int i = 0; i < 64; i++) {
        assert(arr[i] == 100);
    }

    bn_tp_free(pool);
    printf("PASSED\n");
}

// --- Test bn_tp_num_threads ---

static void test_num_threads(void) {
    printf("test_num_threads... ");

    assert(bn_tp_num_threads(NULL) == 1);

    BnThreadPool *pool = bn_tp_create(7);
    assert(bn_tp_num_threads(pool) == 8);
    bn_tp_free(pool);

    printf("PASSED\n");
}

static void test_poll_policy(void) {
    printf("test_poll_policy... ");

    BnThreadPool *pool = bn_tp_create(2);
    assert(pool != NULL);
    int arr[2048];
    memset(arr, 0, sizeof(arr));
    BnTPTask task = { add_one, arr, 2048 };

    bn_tp_set_poll_iters(pool, 0, 0);
    bn_tp_dispatch(pool, &task, 1);
    bn_tp_set_poll_iters(pool, 50000, 50000);
    bn_tp_dispatch(pool, &task, 1);
    bn_tp_set_cpu_decode_policy(pool);
    bn_tp_dispatch(pool, &task, 1);

    for (int i = 0; i < 2048; i++)
        assert(arr[i] == 3);
    bn_tp_free(pool);

    printf("PASSED\n");
}

static void test_runtime_policy_snapshot(void) {
    printf("test_runtime_policy_snapshot... ");

    setenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS", "1", 1);
    setenv("BN_DUMP_LAYER_INP", "/tmp/bn-policy-before", 1);
    BnThreadPool *pool = bn_tp_create(1);
    assert(pool != NULL);
    const BnCPURuntimePolicy *policy = bn_tp_cpu_policy(pool);
    assert(policy && !policy->prepared_qweights);
    assert(strcmp(policy->debug_dump_path, "/tmp/bn-policy-before") == 0);

    unsetenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS");
    setenv("BN_DUMP_LAYER_INP", "/tmp/bn-policy-after", 1);
    assert(!policy->prepared_qweights);
    assert(strcmp(policy->debug_dump_path, "/tmp/bn-policy-before") == 0);

    bn_tp_free(pool);
    unsetenv("BN_DUMP_LAYER_INP");
    printf("PASSED\n");
}

int main(void) {
    setenv("BN_CPU_REFERENCE_MATH", "1", 1);
    BnThreadPool *serial_pool = bn_tp_create(0);
    assert(serial_pool != NULL);
    assert(bn_tp_num_threads(serial_pool) == 1);
    assert(bn_tp_cpu_policy(serial_pool)->reference_math);
    int serial_values[17] = {0};
    BnTPTask serial_task = { add_one, serial_values, 17 };
    bn_tp_dispatch(serial_pool, &serial_task, 1);
    for (int i = 0; i < 17; i++) assert(serial_values[i] == 1);
    bn_tp_free(serial_pool);
    unsetenv("BN_CPU_REFERENCE_MATH");

    printf("=== ThreadPool Tests ===\n");
    test_serial_dispatch();
    test_fine_dispatch_reset();
    test_threaded_single_task();
    test_multi_task_dispatch();
    test_rapid_dispatch();
    test_num_threads();
    test_poll_policy();
    test_runtime_policy_snapshot();
    printf("All threadpool tests passed!\n");
    return 0;
}
