/*
 * sh_arena.c - Arena allocator implementation
 */

#include "sh_arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>  /* For SIZE_MAX */

SHArena *sh_arena_create(size_t capacity)
{
    SHArena *arena = malloc(sizeof(SHArena));
    if (!arena) return NULL;

    arena->buffer = malloc(capacity);
    if (!arena->buffer) {
        free(arena);
        return NULL;
    }

    arena->capacity = capacity;
    arena->used = 0;
    return arena;
}

void *sh_arena_alloc(SHArena *arena, size_t size)
{
    if (!arena || !arena->buffer) return NULL;

    /* Align to SH_ARENA_ALIGN bytes for double/pointer alignment */
    size = (size + SH_ARENA_ALIGN - 1) & ~(size_t)(SH_ARENA_ALIGN - 1);

    if (arena->used + size > arena->capacity) {
        return NULL;  /* Out of space */
    }

    void *ptr = arena->buffer + arena->used;
    arena->used += size;
    return ptr;
}

void *sh_arena_calloc(SHArena *arena, size_t count, size_t size)
{
    /* Check for integer overflow before multiplication */
    if (size > 0 && count > SIZE_MAX / size) {
        return NULL;
    }
    size_t total = count * size;
    void *ptr = sh_arena_alloc(arena, total);
    if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
}

void sh_arena_reset(SHArena *arena)
{
    if (arena) {
        arena->used = 0;
    }
}

void sh_arena_free(SHArena *arena)
{
    if (arena) {
        free(arena->buffer);
        free(arena);
    }
}

size_t sh_arena_remaining(const SHArena *arena)
{
    if (!arena) return 0;
    return arena->capacity - arena->used;
}

size_t sh_arena_used(const SHArena *arena)
{
    if (!arena) return 0;
    return arena->used;
}
