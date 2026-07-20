#include "sampler_backend_internal.h"
#include "transformer_cpu_features_internal.h"

#if BN_TRANSFORMER_CPU_HAS_NEON
#include <arm_neon.h>
#endif

#if BN_TRANSFORMER_CPU_HAS_AVX2 || BN_TRANSFORMER_CPU_HAS_AVX512
#include <immintrin.h>
#endif

int bn_sampler_argmax(const float *v, int n) {
    if (n <= 0) return 0;
#if BN_TRANSFORMER_CPU_HAS_NEON
    int i = 0;
    int best = 0;
    float best_val = v[0];

    if (n >= 4) {
        float32x4_t maxv = vld1q_f32(v);
        int32x4_t maxi = {0, 1, 2, 3};
        int32x4_t idx = {4, 5, 6, 7};
        int32x4_t four = vdupq_n_s32(4);

        for (i = 4; i + 3 < n; i += 4) {
            float32x4_t x = vld1q_f32(v + i);
            uint32x4_t gt = vcgtq_f32(x, maxv);
            maxv = vbslq_f32(gt, x, maxv);
            maxi = vbslq_s32(gt, idx, maxi);
            idx = vaddq_s32(idx, four);
        }

        float vals[4];
        int ids[4];
        vst1q_f32(vals, maxv);
        vst1q_s32(ids, maxi);
        best_val = vals[0];
        best = ids[0];
        for (int k = 1; k < 4; k++) {
            if (vals[k] > best_val) {
                best_val = vals[k];
                best = ids[k];
            }
        }
    }

    for (; i < n; i++) {
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
#elif BN_TRANSFORMER_CPU_HAS_AVX512
    int i = 0;
    int best = 0;
    float best_val = v[0];

    if (n >= 16) {
        __m512 maxv = _mm512_loadu_ps(v);
        __m512i maxi = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                         8, 9, 10, 11, 12, 13, 14, 15);
        __m512i idx = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23,
                                        24, 25, 26, 27, 28, 29, 30, 31);
        __m512i step = _mm512_set1_epi32(16);

        for (i = 16; i + 15 < n; i += 16) {
            __m512 x = _mm512_loadu_ps(v + i);
            __mmask16 gt = _mm512_cmp_ps_mask(x, maxv, _CMP_GT_OQ);
            maxv = _mm512_mask_blend_ps(gt, maxv, x);
            maxi = _mm512_mask_blend_epi32(gt, maxi, idx);
            idx = _mm512_add_epi32(idx, step);
        }

        float vals[16];
        int ids[16];
        _mm512_storeu_ps(vals, maxv);
        _mm512_storeu_si512((__m512i *)ids, maxi);
        best_val = vals[0];
        best = ids[0];
        for (int k = 1; k < 16; k++) {
            if (vals[k] > best_val) {
                best_val = vals[k];
                best = ids[k];
            }
        }
    }

    for (; i < n; i++) {
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
#elif BN_TRANSFORMER_CPU_HAS_AVX2
    int i = 0;
    int best = 0;
    float best_val = v[0];

    if (n >= 8) {
        __m256 maxv = _mm256_loadu_ps(v);
        __m256i maxi = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i idx = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
        __m256i eight = _mm256_set1_epi32(8);

        for (i = 8; i + 7 < n; i += 8) {
            __m256 x = _mm256_loadu_ps(v + i);
            __m256 gt = _mm256_cmp_ps(x, maxv, _CMP_GT_OQ);
            maxv = _mm256_blendv_ps(maxv, x, gt);
            maxi = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(maxi), _mm256_castsi256_ps(idx), gt));
            idx = _mm256_add_epi32(idx, eight);
        }

        float vals[8];
        int ids[8];
        _mm256_storeu_ps(vals, maxv);
        _mm256_storeu_si256((__m256i *)ids, maxi);
        best_val = vals[0];
        best = ids[0];
        for (int k = 1; k < 8; k++) {
            if (vals[k] > best_val) {
                best_val = vals[k];
                best = ids[k];
            }
        }
    }

    for (; i < n; i++) {
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
#else
    int best = 0;
    float best_val = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
#endif
}
