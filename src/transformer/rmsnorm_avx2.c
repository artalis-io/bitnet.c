#include "transformer_internal.h"

#ifdef __AVX2__

void bn_transformer_rmsnorm_avx2(float *out, const float *x, const float *w, int size, float eps) {
    /* Fused two-pass: pass 1 computes sum(x²) AND stores x*w (unscaled).
     * Pass 2 scales out[] by ss — loads only out[], avoids reloading x[] and w[]. */
    __m256 sum_sq = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 wv = _mm256_loadu_ps(w + i);
        sum_sq = _mm256_fmadd_ps(xv, xv, sum_sq);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(xv, wv));  /* unscaled x*w */
    }
    float ss = bn_avx2_hsum_ps(sum_sq);
    for (; i < size; i++) {
        ss += x[i] * x[i];
        out[i] = x[i] * w[i];  /* unscaled */
    }
    ss = 1.0f / sqrtf(ss / size + eps);

    /* Pass 2: scale out[] — loads 1 array instead of 2 */
    __m256 ss_v = _mm256_set1_ps(ss);
    for (i = 0; i + 7 < size; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), ss_v));
    for (; i < size; i++) out[i] *= ss;
}

#endif // __AVX2__
