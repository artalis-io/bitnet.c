#ifndef BN_TRANSFORMER_CPU_FEATURES_INTERNAL_H
#define BN_TRANSFORMER_CPU_FEATURES_INTERNAL_H

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX512F__
#undef __AVX512BW__
#undef __AVX512VNNI__
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

#ifdef __ARM_NEON
#define BN_TRANSFORMER_CPU_HAS_NEON 1
#else
#define BN_TRANSFORMER_CPU_HAS_NEON 0
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define BN_TRANSFORMER_CPU_HAS_NEON_DOTPROD 1
#else
#define BN_TRANSFORMER_CPU_HAS_NEON_DOTPROD 0
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define BN_TRANSFORMER_CPU_HAS_NEON_FP16_ARITH 1
#else
#define BN_TRANSFORMER_CPU_HAS_NEON_FP16_ARITH 0
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    defined(__AVX512VNNI__) && defined(__AVX2__)
#define BN_TRANSFORMER_CPU_HAS_AVX512 1
#else
#define BN_TRANSFORMER_CPU_HAS_AVX512 0
#endif

#ifdef __AVX2__
#define BN_TRANSFORMER_CPU_HAS_AVX2 1
#else
#define BN_TRANSFORMER_CPU_HAS_AVX2 0
#endif

#ifdef __wasm_simd128__
#define BN_TRANSFORMER_CPU_HAS_WASM_SIMD128 1
#else
#define BN_TRANSFORMER_CPU_HAS_WASM_SIMD128 0
#endif

#ifdef __wasm_relaxed_simd__
#define BN_TRANSFORMER_CPU_HAS_WASM_RELAXED_SIMD 1
#else
#define BN_TRANSFORMER_CPU_HAS_WASM_RELAXED_SIMD 0
#endif

#endif // BN_TRANSFORMER_CPU_FEATURES_INTERNAL_H
