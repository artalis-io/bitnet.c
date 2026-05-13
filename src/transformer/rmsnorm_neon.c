#include "transformer_rmsnorm_internal.h"
#include "transformer_simd_internal.h"
#include <math.h>

#ifdef __ARM_NEON

void bn_transformer_rmsnorm_neon(float *out, const float *x, const float *w, int size, float eps) {
    float32x4_t sum_sq = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        sum_sq = vmlaq_f32(sum_sq, xv, xv);
    }
    float ss = bn_transformer_neon_hsum_f32(sum_sq);
    for (; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    float32x4_t ss_v = vdupq_n_f32(ss);
    for (i = 0; i + 3 < size; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(xv, ss_v), wv));
    }
    for (; i < size; i++) out[i] = x[i] * ss * w[i];
}

#endif // __ARM_NEON
