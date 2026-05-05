#ifndef BN_SIMD_HELPERS_H
#define BN_SIMD_HELPERS_H

// Shared SIMD helper functions for NEON, AVX2, and WASM SIMD128.
// Used by quant.c and transformer.c.

#ifdef __ARM_NEON
#include <arm_neon.h>

// Fast vectorized exp approximation using range reduction + polynomial.
// Accurate enough for activation functions; not used for probability math.
static inline float32x4_t bn_neon_fast_exp_f32(float32x4_t x) {
    const float32x4_t log2e = vdupq_n_f32(1.4426950409f);
    const float32x4_t ln2   = vdupq_n_f32(0.6931471806f);
    const float32x4_t half  = vdupq_n_f32(0.5f);
    const float32x4_t one   = vdupq_n_f32(1.0f);
    const float32x4_t p2    = vdupq_n_f32(0.49999994f);
    const float32x4_t p3    = vdupq_n_f32(0.16666667f);
    const float32x4_t p4    = vdupq_n_f32(0.04166664f);

    x = vmaxq_f32(vdupq_n_f32(-87.3f), vminq_f32(vdupq_n_f32(88.7f), x));

    float32x4_t n = vrndmq_f32(vmlaq_f32(half, x, log2e));
    int32x4_t ni = vcvtq_s32_f32(n);
    float32x4_t r = vmlsq_f32(x, n, ln2);

    float32x4_t poly = vmlaq_f32(p3, p4, r);
    poly = vmlaq_f32(p2, poly, r);
    poly = vmlaq_f32(one, poly, r);
    poly = vmlaq_f32(one, poly, r);

    int32x4_t e2n = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
    return vmulq_f32(poly, vreinterpretq_f32_s32(e2n));
}

static inline float32x4_t bn_neon_fast_sigmoid_f32(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t ex = bn_neon_fast_exp_f32(vnegq_f32(x));
    return vdivq_f32(one, vaddq_f32(one, ex));
}

static inline float32x4_t bn_neon_fast_silu_f32(float32x4_t x) {
    return vmulq_f32(x, bn_neon_fast_sigmoid_f32(x));
}
#endif // __ARM_NEON

#ifdef __AVX2__
#include <immintrin.h>

// Horizontal sum of 8 int32 lanes → scalar int32.
static inline int32_t bn_avx2_hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    __m128i shuf = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2));
    sum128 = _mm_add_epi32(sum128, shuf);
    shuf = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 1, 0, 1));
    sum128 = _mm_add_epi32(sum128, shuf);
    return _mm_cvtsi128_si32(sum128);
}

// Horizontal sum of 8 floats → scalar float.
static inline float bn_avx2_hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);        // [1,1,3,3]
    sum128 = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sum128);            // [2,3,...]
    sum128 = _mm_add_ss(sum128, shuf);
    return _mm_cvtss_f32(sum128);
}

// Horizontal max of 8 floats → scalar float.
static inline float bn_avx2_hmax_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(max128);
    max128 = _mm_max_ps(max128, shuf);
    shuf = _mm_movehl_ps(shuf, max128);
    max128 = _mm_max_ss(max128, shuf);
    return _mm_cvtss_f32(max128);
}

// Signed×signed byte dot product accumulate (no VNNI needed).
// Uses the sign trick: maddubs requires unsigned×signed, so we
// compute abs(a)×sign(a,b) where sign(a,b) flips b's sign where a<0.
// Then _mm256_maddubs_epi16 (u8×s8→s16 pairs) + _mm256_madd_epi16 (→s32).
static inline __m256i bn_avx2_dpbusd(__m256i acc, __m256i a, __m256i b) {
    __m256i sign_a = _mm256_sign_epi8(a, a);       // abs(a) — treats 0x80 as negative
    __m256i sign_b = _mm256_sign_epi8(b, a);       // b with sign of a applied
    __m256i prod16 = _mm256_maddubs_epi16(sign_a, sign_b);  // u8×s8 → s16 pairs
    return _mm256_add_epi32(acc, _mm256_madd_epi16(prod16, _mm256_set1_epi16(1)));
}

// Fast vectorized exp approximation using range reduction + polynomial.
// Accurate to ~1e-5 relative error over [-87, 88], sufficient for inference.
// Method: exp(x) = 2^n * exp(r) where n = floor(x/ln2), r = x - n*ln2.
// exp(r) approximated by degree-4 polynomial on [-0.5*ln2, 0.5*ln2].
static inline __m256 bn_avx2_fast_exp_ps(__m256 x) {
    const __m256 log2e   = _mm256_set1_ps(1.4426950409f);   // 1/ln(2)
    const __m256 ln2     = _mm256_set1_ps(0.6931471806f);   // ln(2)
    const __m256 half    = _mm256_set1_ps(0.5f);
    const __m256 one     = _mm256_set1_ps(1.0f);
    // Minimax coefficients for exp(r) on [-ln2/2, ln2/2]
    const __m256 p2      = _mm256_set1_ps(0.49999994f);     // ~1/2!
    const __m256 p3      = _mm256_set1_ps(0.16666667f);     // ~1/3!
    const __m256 p4      = _mm256_set1_ps(0.04166664f);     // ~1/4!

    // Clamp to prevent overflow/underflow
    x = _mm256_max_ps(_mm256_set1_ps(-87.3f), _mm256_min_ps(_mm256_set1_ps(88.7f), x));

    // n = round(x / ln2) = floor(x * log2e + 0.5)
    __m256 t   = _mm256_fmadd_ps(x, log2e, half);
    __m256 n   = _mm256_floor_ps(t);
    __m256i ni = _mm256_cvtps_epi32(n);    // integer n

    // r = x - n * ln2 (reduced argument, |r| <= ln2/2)
    __m256 r = _mm256_fnmadd_ps(n, ln2, x);

    // Polynomial: exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4! (Horner form)
    __m256 poly = _mm256_fmadd_ps(p4, r, p3);
    poly = _mm256_fmadd_ps(poly, r, p2);
    poly = _mm256_fmadd_ps(poly, r, one);
    poly = _mm256_fmadd_ps(poly, r, one);

    // Scale by 2^n: add n to IEEE754 exponent field (bias=127)
    __m256i e2n = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(poly, _mm256_castsi256_ps(e2n));
}

// Fast vectorized sigmoid: 1 / (1 + exp(-x))
static inline __m256 bn_avx2_fast_sigmoid_ps(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 ex = bn_avx2_fast_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(one, _mm256_add_ps(one, ex));
}

// Fast vectorized SiLU: x * sigmoid(x)
static inline __m256 bn_avx2_fast_silu_ps(__m256 x) {
    return _mm256_mul_ps(x, bn_avx2_fast_sigmoid_ps(x));
}

#endif // __AVX2__

#ifdef __wasm_simd128__
#include <wasm_simd128.h>

// Horizontal sum of 4 floats → scalar float.
static inline float bn_wasm_hsum_f32x4(v128_t v) {
    v128_t shuf = wasm_i32x4_shuffle(v, v, 2, 3, 0, 1);
    v128_t sum = wasm_f32x4_add(v, shuf);
    shuf = wasm_i32x4_shuffle(sum, sum, 1, 0, 3, 2);
    sum = wasm_f32x4_add(sum, shuf);
    return wasm_f32x4_extract_lane(sum, 0);
}

// Horizontal max of 4 floats → scalar float.
static inline float bn_wasm_hmax_f32x4(v128_t v) {
    v128_t shuf = wasm_i32x4_shuffle(v, v, 2, 3, 0, 1);
    v128_t mx = wasm_f32x4_max(v, shuf);
    shuf = wasm_i32x4_shuffle(mx, mx, 1, 0, 3, 2);
    mx = wasm_f32x4_max(mx, shuf);
    return wasm_f32x4_extract_lane(mx, 0);
}

#endif // __wasm_simd128__

#endif // BN_SIMD_HELPERS_H
