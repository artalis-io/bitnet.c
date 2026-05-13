#ifndef BN_TRANSFORMER_SIMD_INTERNAL_H
#define BN_TRANSFORMER_SIMD_INTERNAL_H

#include "simd_helpers.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
static inline float bn_transformer_neon_hsum_f32(float32x4_t v) {
    float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}
#endif

#endif // BN_TRANSFORMER_SIMD_INTERNAL_H
