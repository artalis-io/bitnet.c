#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// --- Test serial dispatch (pool=NULL) ---

static void add_one(void *ctx, int start, int end) {
    int *arr = (int *)ctx;
    for (int i = start; i < end; i++) arr[i] += 1;
}

static void test_serial_dispatch(void) {
    printf("test_serial_dispatch... ");

    int arr[100];
    memset(arr, 0, sizeof(arr));

    TPTask task = { add_one, arr, 100 };
    tp_dispatch(NULL, &task, 1);

    for (int i = 0; i < 100; i++) {
        assert(arr[i] == 1);
    }

    printf("PASSED\n");
}

// --- Test single-task dispatch with various thread counts ---

static void test_threaded_single_task(void) {
    printf("test_threaded_single_task... ");

    for (int nw = 1; nw <= 4; nw++) {
        ThreadPool *pool = tp_create(nw);
        assert(pool != NULL);
        assert(tp_num_threads(pool) == nw + 1);

        int arr[256];
        memset(arr, 0, sizeof(arr));

        TPTask task = { add_one, arr, 256 };
        tp_dispatch(pool, &task, 1);

        // Verify all elements were incremented exactly once
        for (int i = 0; i < 256; i++) {
            assert(arr[i] == 1);
        }

        tp_free(pool);
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

    ThreadPool *pool = tp_create(3);

    float a[128], b[128], sum_out[128], prod_out[128];
    for (int i = 0; i < 128; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    VecAddCtx sum_ctx = { sum_out, a, b, 128 };
    VecAddCtx prod_ctx = { prod_out, a, b, 128 };

    TPTask tasks[2] = {
        { vec_add_range, &sum_ctx, 128 },
        { vec_mul_range, &prod_ctx, 128 },
    };
    tp_dispatch(pool, tasks, 2);

    for (int i = 0; i < 128; i++) {
        assert(sum_out[i] == (float)(i + i * 2));
        assert(prod_out[i] == (float)(i * i * 2));
    }

    tp_free(pool);
    printf("PASSED\n");
}

// --- Test rapid successive dispatches (stress test generation counter) ---

static void test_rapid_dispatch(void) {
    printf("test_rapid_dispatch... ");

    ThreadPool *pool = tp_create(3);

    int arr[64];
    memset(arr, 0, sizeof(arr));

    TPTask task = { add_one, arr, 64 };

    // Dispatch 100 times rapidly
    for (int round = 0; round < 100; round++) {
        tp_dispatch(pool, &task, 1);
    }

    for (int i = 0; i < 64; i++) {
        assert(arr[i] == 100);
    }

    tp_free(pool);
    printf("PASSED\n");
}

// --- Test tp_num_threads ---

static void test_num_threads(void) {
    printf("test_num_threads... ");

    assert(tp_num_threads(NULL) == 1);

    ThreadPool *pool = tp_create(7);
    assert(tp_num_threads(pool) == 8);
    tp_free(pool);

    printf("PASSED\n");
}

int main(void) {
    printf("=== ThreadPool Tests ===\n");
    test_serial_dispatch();
    test_threaded_single_task();
    test_multi_task_dispatch();
    test_rapid_dispatch();
    test_num_threads();
    printf("All threadpool tests passed!\n");
    return 0;
}
