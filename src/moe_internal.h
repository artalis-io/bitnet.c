#ifndef BN_MOE_INTERNAL_H
#define BN_MOE_INTERNAL_H

#include "model.h"
#include "moe.h"
#include "session.h"
#include "platform.h"
#include "quant.h"
#include "simd_helpers.h"
#include "sh_log.h"
#include "bn_alloc.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

#ifndef __EMSCRIPTEN__
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
extern void bn_transformer_rmsnorm_neon(float *out, const float *x, const float *w, int size, float eps);
#define bn_moe_rmsnorm bn_transformer_rmsnorm_neon
#elif defined(__AVX2__)
extern void bn_transformer_rmsnorm_avx2(float *out, const float *x, const float *w, int size, float eps);
#define bn_moe_rmsnorm bn_transformer_rmsnorm_avx2
#elif defined(__wasm_simd128__)
extern void bn_transformer_rmsnorm_wasm(float *out, const float *x, const float *w, int size, float eps);
#define bn_moe_rmsnorm bn_transformer_rmsnorm_wasm
#else
extern void bn_transformer_rmsnorm_scalar(float *out, const float *x, const float *w, int size, float eps);
#define bn_moe_rmsnorm bn_transformer_rmsnorm_scalar
#endif

typedef struct BnMoECache BnMoECache;
typedef struct BnMoEPrefetch BnMoEPrefetch;

typedef struct {
    float *hb;
    const float *gate;
    const float *up;
} BnSwiGLUCtx;

int bn_moe_checked_mul_size(size_t a, size_t b, size_t *out);
int bn_moe_proj_info(const BnMoEExpertMap *map, int expert_idx, int proj,
                     size_t *offset, size_t *proj_bytes);
#if !defined(__EMSCRIPTEN__)
void bn_moe_madvise_experts(const BnMoEIO *io, const BnMoEExpertMap *map,
                            const int *indices, int n, int advice, int proj_mask);
#endif
const void *bn_moe_load_expert_proj_into(const BnMoEIO *io, BnMoEStats *stats,
                                         const BnMoEExpertMap *map,
                                         int expert_idx, int proj,
                                         uint8_t *buf, size_t buf_size);
const void *bn_moe_load_expert_proj(const BnMoEIO *io, BnMoEState *ms,
                                    const BnMoEExpertMap *map,
                                    int expert_idx, int proj);
BnQWeight bn_moe_make_qweight(const void *data, int type, int rows, int cols);
void bn_moe_swiglu_range(void *ctx, int start, int end);
void bn_moe_swiglu(float *hb, const float *gate, const float *up, int n);
double bn_moe_time_ms(void);
void bn_moe_weighted_add(float *dst, const float *src, float weight, int n);
void bn_moe_residual_add(float *x, const float *r, int n);

const uint8_t *bn_moe_cache_lookup_internal(BnMoECache *c, int layer, int expert_idx);
uint8_t *bn_moe_cache_insert_internal(BnMoECache *c, int layer, int expert_idx);
size_t bn_moe_cache_gate_bytes(const BnMoECache *c);
size_t bn_moe_cache_up_bytes(const BnMoECache *c);

#if !defined(__EMSCRIPTEN__)
BnMoEPrefetch *bn_moe_prefetch_init_internal(int fd);
void bn_moe_prefetch_free_internal(BnMoEPrefetch *pf);
void bn_moe_prefetch_start2_internal(BnMoEPrefetch *pf,
                                     uint8_t *buf1, size_t size1, off_t off1,
                                     uint8_t *buf2, size_t size2, off_t off2);
void bn_moe_prefetch_start1_internal(BnMoEPrefetch *pf,
                                     uint8_t *buf1, size_t size1, off_t off1);
int bn_moe_prefetch_wait_internal(BnMoEPrefetch *pf);
void bn_moe_prefetch_collect_stats(BnMoEPrefetch *pf, BnMoEStats *stats);
#endif

#endif // BN_MOE_INTERNAL_H
