#include "quant.h"
#include "sh_arena.h"
#include "gguf.h"
#include <string.h>

size_t bn_quant_prepared_qweight_size(const BnQWeight *w,
                                      BnPreparedWeightKind *kind) {
    if (kind) *kind = BN_PREPARED_WEIGHT_NONE;
    if (!w || !w->data) return 0;

#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__wasm_relaxed_simd__)
    if (w->type == BN_GGUF_TENSOR_Q4_0) {
        if (w->rows % 4 != 0) return 0;
        size_t n_blocks = (size_t)w->rows * (w->cols / 32);
        size_t bytes = n_blocks * 16 + SH_ARENA_ALIGN;
#ifdef __wasm_relaxed_simd__
        bytes += n_blocks * sizeof(float) + SH_ARENA_ALIGN;
#else
        bytes += n_blocks * sizeof(uint16_t) + SH_ARENA_ALIGN;
#endif
        if (kind) *kind = BN_PREPARED_WEIGHT_Q4_0_REPACK;
        return bytes;
    }
#endif

#ifdef __wasm_relaxed_simd__
    if (w->type == BN_GGUF_TENSOR_Q8_0) {
        if ((w->cols & 31) != 0) return 0;
        if (kind) *kind = BN_PREPARED_WEIGHT_Q8_0_F32_SCALES;
        return (size_t)w->rows * (w->cols / 32) * sizeof(float) +
               SH_ARENA_ALIGN;
    }
#endif

#if defined(__AVX2__)
    if (w->type == BN_GGUF_TENSOR_Q4_K) {
        int n_blocks_per_row = w->cols / BN_QK_K;
        if (n_blocks_per_row <= 0) return 0;
        size_t n_blocks = (size_t)w->rows * n_blocks_per_row;
        if (kind) *kind = BN_PREPARED_WEIGHT_Q4_K_SCALES;
        return n_blocks * 16 + SH_ARENA_ALIGN +
               n_blocks * 2 * sizeof(float) + SH_ARENA_ALIGN;
    }
#endif

    return 0;
}

static int prepare_q4_0(BnPreparedWeight *prepared, const BnQWeight *w,
                        SHArena *arena) {
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__wasm_relaxed_simd__)
    if (w->type != BN_GGUF_TENSOR_Q4_0 || !w->data) return -1;
    if (w->rows % 4 != 0) return -1;
    int n_blocks_per_row = w->cols / 32;
    size_t n_blocks = (size_t)w->rows * n_blocks_per_row;

    prepared->kind = BN_PREPARED_WEIGHT_Q4_0_REPACK;
    prepared->qs = (uint8_t *)sh_arena_alloc(arena, n_blocks * 16);
#ifdef __wasm_relaxed_simd__
    prepared->f32_scales =
        (float *)sh_arena_alloc(arena, n_blocks * sizeof(float));
#else
    prepared->scales =
        (uint16_t *)sh_arena_alloc(arena, n_blocks * sizeof(uint16_t));
#endif
    if (!prepared->qs) return -1;
#ifdef __wasm_relaxed_simd__
    if (!prepared->f32_scales) return -1;
#else
    if (!prepared->scales) return -1;
#endif

    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)w->data;
    int n_groups = w->rows / 4;
    for (int g = 0; g < n_groups; g++) {
        for (int b = 0; b < n_blocks_per_row; b++) {
            size_t gb = (size_t)g * n_blocks_per_row + b;
            for (int r = 0; r < 4; r++) {
                size_t src = (size_t)(g * 4 + r) * n_blocks_per_row + b;
#ifdef __wasm_relaxed_simd__
                prepared->f32_scales[gb * 4 + r] =
                    bn_fp16_to_fp32(blocks[src].d);
#else
                prepared->scales[gb * 4 + r] = blocks[src].d;
#endif
            }
            uint8_t *dst = prepared->qs + gb * 64;
            for (int ng = 0; ng < 4; ng++) {
                for (int r = 0; r < 4; r++) {
                    size_t src = (size_t)(g * 4 + r) * n_blocks_per_row + b;
                    const uint8_t *qs = blocks[src].qs + ng * 4;
                    uint8_t *dp = dst + ng * 16 + r * 4;
                    for (int j = 0; j < 4; j++)
                        dp[j] = qs[j] ^ 0x88;
                }
            }
        }
    }
    return 0;
#else
    (void)prepared;
    (void)w;
    (void)arena;
    return -1;
#endif
}

static int prepare_q8_0(BnPreparedWeight *prepared, const BnQWeight *w,
                        SHArena *arena) {
#ifdef __wasm_relaxed_simd__
    if (w->type != BN_GGUF_TENSOR_Q8_0 || !w->data) return -1;
    if ((w->cols & 31) != 0) return -1;
    int n_blocks_per_row = w->cols / 32;
    size_t n_blocks = (size_t)w->rows * n_blocks_per_row;

    prepared->kind = BN_PREPARED_WEIGHT_Q8_0_F32_SCALES;
    prepared->f32_scales =
        (float *)sh_arena_alloc(arena, n_blocks * sizeof(float));
    if (!prepared->f32_scales) return -1;

    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)w->data;
    for (size_t i = 0; i < n_blocks; i++)
        prepared->f32_scales[i] = bn_fp16_to_fp32(blocks[i].d);
    return 0;
#else
    (void)prepared;
    (void)w;
    (void)arena;
    return -1;
#endif
}

static int prepare_q4_k(BnPreparedWeight *prepared, const BnQWeight *w,
                        SHArena *arena) {
#if defined(__AVX2__)
    if (w->type != BN_GGUF_TENSOR_Q4_K || !w->data) return -1;
    int n_blocks_per_row = w->cols / BN_QK_K;
    if (n_blocks_per_row <= 0) return -1;
    size_t n_blocks = (size_t)w->rows * n_blocks_per_row;

    prepared->kind = BN_PREPARED_WEIGHT_Q4_K_SCALES;
    prepared->qs = (uint8_t *)sh_arena_alloc(arena, n_blocks * 16);
    prepared->f32_scales =
        (float *)sh_arena_alloc(arena, n_blocks * 2 * sizeof(float));
    if (!prepared->qs || !prepared->f32_scales) return -1;

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)w->data;

    for (size_t i = 0; i < n_blocks; i++) {
        const BnBlockQ4K *blk = &blocks[i];
        uint32_t utmp[3];
        memcpy(utmp, blk->scales, 12);
        uint32_t m_lo = utmp[1] & kmask1;
        uint32_t m_hi = ((utmp[2] >> 4) & kmask2) |
                        (((utmp[1] >> 6) & kmask3) << 4);
        utmp[1] = (utmp[2] & kmask2) |
                  (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;
        memcpy(prepared->qs + i * 16, utmp, 8);
        memcpy(prepared->qs + i * 16 + 8, &m_lo, 4);
        memcpy(prepared->qs + i * 16 + 12, &m_hi, 4);
        prepared->f32_scales[i * 2] = bn_fp16_to_fp32(blk->d);
        prepared->f32_scales[i * 2 + 1] = bn_fp16_to_fp32(blk->dmin);
    }
    return 0;
#else
    (void)prepared;
    (void)w;
    (void)arena;
    return -1;
#endif
}

int bn_quant_prepare_qweight(BnPreparedWeight *prepared, const BnQWeight *w,
                             SHArena *arena) {
    if (!prepared || !w || !arena) return -1;
    memset(prepared, 0, sizeof(*prepared));

    BnPreparedWeightKind kind = BN_PREPARED_WEIGHT_NONE;
    if (bn_quant_prepared_qweight_size(w, &kind) == 0) return -1;

    switch (kind) {
        case BN_PREPARED_WEIGHT_Q4_0_REPACK:
            return prepare_q4_0(prepared, w, arena);
        case BN_PREPARED_WEIGHT_Q8_0_F32_SCALES:
            return prepare_q8_0(prepared, w, arena);
        case BN_PREPARED_WEIGHT_Q4_K_SCALES:
            return prepare_q4_k(prepared, w, arena);
        default:
            return -1;
    }
}
