#include "transformer_rmsnorm_internal.h"
#include "transformer_simd_internal.h"
#include <math.h>

#ifdef __AVX2__

void bn_transformer_rmsnorm_avx2(float *out, const float *x, const float *w, int size, float eps) {
    double sum = 0.0;
    for (int i = 0; i < size; i++)
        sum += (double)(x[i] * x[i]);

    float scale = 1.0f / sqrtf((float)(sum / size) + eps);
    __m256 scale_v = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 wv = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(out + i,
                         _mm256_mul_ps(_mm256_mul_ps(xv, scale_v), wv));
    }
    for (; i < size; i++)
        out[i] = x[i] * scale * w[i];
}

#endif // __AVX2__
