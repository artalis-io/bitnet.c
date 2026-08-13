#include "transformer_rmsnorm_internal.h"
#include "transformer_simd_internal.h"
#include <math.h>

#ifdef __ARM_NEON

void bn_transformer_rmsnorm_neon(float *out, const float *x, const float *w, int size, float eps) {
    double sum = 0.0;
    for (int i = 0; i < size; i++)
        sum += (double)(x[i] * x[i]);

    float mean = (float)(sum / size);
    float scale = 1.0f / sqrtf(mean + eps);
    float32x4_t scale_v = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t scaled = vmulq_f32(vld1q_f32(x + i), scale_v);
        vst1q_f32(out + i, vmulq_f32(scaled, vld1q_f32(w + i)));
    }
    for (; i < size; i++)
        out[i] = x[i] * scale * w[i];
}

#endif // __ARM_NEON
