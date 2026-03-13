/*
 * sh_arena.h - Arena allocator for OTTO Platform
 *
 * Simple bump allocator for allocating many arrays with the same lifetime.
 * Benefits:
 * - Single malloc/free instead of 20+ individual allocations
 * - No fragmentation from repeated alloc/free cycles
 * - Simpler error handling (one check, not 20)
 * - Cache-friendly contiguous memory layout
 *
 * Usage:
 *   SHArena *arena = sh_arena_create(1024 * 1024);  // 1MB arena
 *   double *arr1 = sh_arena_alloc(arena, 100 * sizeof(double));
 *   int *arr2 = sh_arena_calloc(arena, 50, sizeof(int));
 *   // ... use arrays ...
 *   sh_arena_reset(arena);  // Reuse for next batch (no free needed)
 *   // ... or ...
 *   sh_arena_free(arena);   // Free all memory
 */

#ifndef SH_ARENA_H
#define SH_ARENA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Alignment for allocations (8 bytes for double/pointer) */
#define SH_ARENA_ALIGN 8

/* Arena allocator structure */
typedef struct SHArena {
    char *buffer;       /* Pre-allocated memory block */
    size_t capacity;    /* Total size of buffer */
    size_t used;        /* Currently used bytes */
} SHArena;

/*
 * Create arena with given capacity.
 * Returns NULL on allocation failure.
 */
SHArena *sh_arena_create(size_t capacity);

/*
 * Allocate from arena with 8-byte alignment.
 * Returns NULL if arena is NULL or out of space.
 * Memory is NOT initialized.
 */
void *sh_arena_alloc(SHArena *arena, size_t size);

/*
 * Allocate and zero-initialize from arena.
 * Returns NULL if arena is NULL or out of space.
 */
void *sh_arena_calloc(SHArena *arena, size_t count, size_t size);

/*
 * Reset arena for reuse.
 * Does not free memory, just resets position to beginning.
 * All previous allocations become invalid.
 */
void sh_arena_reset(SHArena *arena);

/*
 * Free arena and all memory.
 */
void sh_arena_free(SHArena *arena);

/*
 * Get remaining capacity in arena.
 */
size_t sh_arena_remaining(const SHArena *arena);

/*
 * Get current used bytes in arena.
 */
size_t sh_arena_used(const SHArena *arena);

#ifdef __cplusplus
}
#endif

#endif /* SH_ARENA_H */
