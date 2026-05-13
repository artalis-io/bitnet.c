#include "transformer_gqa_internal.h"
#include "turboquant.h"

#ifdef __ARM_NEON

// NEON TurboQuant GQA: benefits from NEON-optimized FWHT, popcount,
// and FMA in turboquant.c. Uses same precomputed-QJL flow as scalar.
void bn_transformer_gqa_tq_neon_range(void *ctx, int start, int end) {
    bn_transformer_gqa_tq_scalar_range(ctx, start, end);
}

#endif // __ARM_NEON
