#include "sh_arena.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

static void test_create_free(void) {
    SHArena *a = sh_arena_create(1024);
    assert(a != NULL);
    assert(sh_arena_remaining(a) == 1024);
    assert(sh_arena_used(a) == 0);
    sh_arena_free(a);

    // NULL arena is safe
    sh_arena_free(NULL);
    assert(sh_arena_remaining(NULL) == 0);
    assert(sh_arena_used(NULL) == 0);
    printf("test_create_free... PASSED\n");
}

static void test_alloc_basic(void) {
    SHArena *a = sh_arena_create(256);

    void *p1 = sh_arena_alloc(a, 32);
    assert(p1 != NULL);
    assert(sh_arena_used(a) >= 32);

    void *p2 = sh_arena_alloc(a, 64);
    assert(p2 != NULL);
    assert((char *)p2 > (char *)p1);  // monotonically increasing

    // Returned pointers should be 8-byte aligned
    assert(((uintptr_t)p1 % SH_ARENA_ALIGN) == 0);
    assert(((uintptr_t)p2 % SH_ARENA_ALIGN) == 0);

    sh_arena_free(a);
    printf("test_alloc_basic... PASSED\n");
}

static void test_calloc_zeroed(void) {
    SHArena *a = sh_arena_create(256);

    int *arr = (int *)sh_arena_calloc(a, 16, sizeof(int));
    assert(arr != NULL);
    for (int i = 0; i < 16; i++) {
        assert(arr[i] == 0);
    }

    sh_arena_free(a);
    printf("test_calloc_zeroed... PASSED\n");
}

static void test_capacity_exhaustion(void) {
    SHArena *a = sh_arena_create(64);

    // Should succeed
    void *p1 = sh_arena_alloc(a, 32);
    assert(p1 != NULL);

    // Should fail — not enough space (32 used + alignment)
    void *p2 = sh_arena_alloc(a, 64);
    assert(p2 == NULL);

    // Small alloc should still work if space remains
    void *p3 = sh_arena_alloc(a, 8);
    // May or may not succeed depending on alignment, just don't crash
    (void)p3;

    sh_arena_free(a);
    printf("test_capacity_exhaustion... PASSED\n");
}

static void test_reset(void) {
    SHArena *a = sh_arena_create(128);

    sh_arena_alloc(a, 100);
    assert(sh_arena_used(a) >= 100);

    sh_arena_reset(a);
    assert(sh_arena_used(a) == 0);
    assert(sh_arena_remaining(a) == 128);

    // Can allocate again after reset
    void *p = sh_arena_alloc(a, 64);
    assert(p != NULL);

    sh_arena_free(a);

    // Reset on NULL is safe
    sh_arena_reset(NULL);
    printf("test_reset... PASSED\n");
}

static void test_alignment(void) {
    SHArena *a = sh_arena_create(256);

    // Allocate odd sizes, verify alignment is maintained
    for (int i = 0; i < 10; i++) {
        void *p = sh_arena_alloc(a, 1 + i);  // sizes 1..10
        if (!p) break;
        assert(((uintptr_t)p % SH_ARENA_ALIGN) == 0);
    }

    sh_arena_free(a);
    printf("test_alignment... PASSED\n");
}

static void test_null_arena(void) {
    // All operations on NULL arena should be safe
    assert(sh_arena_alloc(NULL, 32) == NULL);
    assert(sh_arena_calloc(NULL, 4, 8) == NULL);
    printf("test_null_arena... PASSED\n");
}

static void test_overflow_guards(void) {
    SHArena *huge = sh_arena_create(SIZE_MAX);
    assert(huge == NULL);

    SHArena *a = sh_arena_create(128);
    assert(a != NULL);
    assert(sh_arena_alloc(a, SIZE_MAX) == NULL);
    assert(sh_arena_calloc(a, SIZE_MAX, 2) == NULL);
    sh_arena_free(a);

    printf("test_overflow_guards... PASSED\n");
}

int main(void) {
    printf("=== Arena Tests ===\n");
    test_create_free();
    test_alloc_basic();
    test_calloc_zeroed();
    test_capacity_exhaustion();
    test_reset();
    test_alignment();
    test_null_arena();
    test_overflow_guards();
    printf("All arena tests passed!\n");
    return 0;
}
