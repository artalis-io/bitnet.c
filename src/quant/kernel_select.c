#include "quant_ctx.h"
#include "quant_kernels_scalar.h"
#include "quant_kernels_neon.h"
#include "quant_kernels_avx2.h"
#include "quant_kernels_wasm.h"
#include "gguf.h"

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#endif

bn_tp_fn bn_quant_get_float_kernel(int type) {
    switch (type) {
#ifdef __ARM_NEON
    case BN_GGUF_TENSOR_Q4_K:    return bn_quant_q4k_neon_range;
    case BN_GGUF_TENSOR_Q5_K:    return bn_quant_q5k_neon_range;
    case BN_GGUF_TENSOR_Q6_K:    return bn_quant_q6k_neon_range;
    case BN_GGUF_TENSOR_Q3_K:    return bn_quant_q3k_neon_range;
    case BN_GGUF_TENSOR_Q2_K:    return bn_quant_q2k_neon_range;
    case BN_GGUF_TENSOR_Q8_K:    return bn_quant_q8k_neon_range;
    case BN_GGUF_TENSOR_F32:     return bn_quant_f32_neon_range;
    case BN_GGUF_TENSOR_F16:     return bn_quant_f16_neon_range;
    case BN_GGUF_TENSOR_BF16:    return bn_quant_bf16_neon_range;
    case BN_GGUF_TENSOR_Q4_1:    return bn_quant_q4_1_neon_range;
    case BN_GGUF_TENSOR_Q5_1:    return bn_quant_q5_1_scalar_range;
    case BN_GGUF_TENSOR_IQ4_NL:  return bn_quant_iq4nl_neon_range;
    case BN_GGUF_TENSOR_IQ4_XS:  return bn_quant_iq4xs_neon_range;
    case BN_GGUF_TENSOR_IQ3_XXS: return bn_quant_iq3xxs_neon_range;
    case BN_GGUF_TENSOR_IQ3_S:   return bn_quant_iq3s_neon_range;
    case BN_GGUF_TENSOR_IQ2_XXS: return bn_quant_iq2xxs_neon_range;
    case BN_GGUF_TENSOR_IQ2_XS:  return bn_quant_iq2xs_neon_range;
    case BN_GGUF_TENSOR_IQ2_S:   return bn_quant_iq2s_neon_range;
#elif defined(__AVX2__)
    case BN_GGUF_TENSOR_Q4_K:    return bn_quant_q4k_avx2_range;
    case BN_GGUF_TENSOR_Q5_K:    return bn_quant_q5k_avx2_range;
    case BN_GGUF_TENSOR_Q6_K:    return bn_quant_q6k_avx2_range;
    case BN_GGUF_TENSOR_Q3_K:    return bn_quant_q3k_avx2_range;
    case BN_GGUF_TENSOR_Q2_K:    return bn_quant_q2k_avx2_range;
    case BN_GGUF_TENSOR_Q8_K:    return bn_quant_q8k_avx2_range;
    case BN_GGUF_TENSOR_F32:     return bn_quant_f32_avx2_range;
    case BN_GGUF_TENSOR_F16:     return bn_quant_f16_avx2_range;
    case BN_GGUF_TENSOR_BF16:    return bn_quant_bf16_avx2_range;
    case BN_GGUF_TENSOR_Q4_1:    return bn_quant_q4_1_avx2_range;
    case BN_GGUF_TENSOR_Q5_1:    return bn_quant_q5_1_scalar_range;
    case BN_GGUF_TENSOR_IQ4_NL:  return bn_quant_iq4nl_avx2_range;
    case BN_GGUF_TENSOR_IQ4_XS:  return bn_quant_iq4xs_avx2_range;
    case BN_GGUF_TENSOR_IQ3_XXS: return bn_quant_iq3xxs_avx2_range;
    case BN_GGUF_TENSOR_IQ3_S:   return bn_quant_iq3s_avx2_range;
    case BN_GGUF_TENSOR_IQ2_XXS: return bn_quant_iq2xxs_avx2_range;
    case BN_GGUF_TENSOR_IQ2_XS:  return bn_quant_iq2xs_avx2_range;
    case BN_GGUF_TENSOR_IQ2_S:   return bn_quant_iq2s_avx2_range;
#elif defined(__wasm_simd128__)
    case BN_GGUF_TENSOR_Q4_K:    return bn_quant_q4k_wasm_range;
    case BN_GGUF_TENSOR_Q5_K:    return bn_quant_q5k_wasm_range;
    case BN_GGUF_TENSOR_Q6_K:    return bn_quant_q6k_wasm_range;
    case BN_GGUF_TENSOR_Q3_K:    return bn_quant_q3k_wasm_range;
    case BN_GGUF_TENSOR_Q2_K:    return bn_quant_q2k_wasm_range;
    case BN_GGUF_TENSOR_Q8_K:    return bn_quant_q8k_wasm_range;
    case BN_GGUF_TENSOR_F32:     return bn_quant_f32_wasm_range;
    case BN_GGUF_TENSOR_F16:     return bn_quant_f16_wasm_range;
    case BN_GGUF_TENSOR_BF16:    return bn_quant_bf16_wasm_range;
    case BN_GGUF_TENSOR_Q4_1:    return bn_quant_q4_1_wasm_range;
    case BN_GGUF_TENSOR_Q5_1:    return bn_quant_q5_1_scalar_range;
    case BN_GGUF_TENSOR_IQ4_NL:  return bn_quant_iq4nl_wasm_range;
    case BN_GGUF_TENSOR_IQ4_XS:  return bn_quant_iq4xs_wasm_range;
    case BN_GGUF_TENSOR_IQ3_XXS: return bn_quant_iq3xxs_wasm_range;
    case BN_GGUF_TENSOR_IQ3_S:   return bn_quant_iq3s_wasm_range;
    case BN_GGUF_TENSOR_IQ2_XXS: return bn_quant_iq2xxs_wasm_range;
    case BN_GGUF_TENSOR_IQ2_XS:  return bn_quant_iq2xs_wasm_range;
    case BN_GGUF_TENSOR_IQ2_S:   return bn_quant_iq2s_wasm_range;
#else
    case BN_GGUF_TENSOR_Q4_K:    return bn_quant_q4k_scalar_range;
    case BN_GGUF_TENSOR_Q5_K:    return bn_quant_q5k_scalar_range;
    case BN_GGUF_TENSOR_Q6_K:    return bn_quant_q6k_scalar_range;
    case BN_GGUF_TENSOR_Q3_K:    return bn_quant_q3k_scalar_range;
    case BN_GGUF_TENSOR_Q2_K:    return bn_quant_q2k_scalar_range;
    case BN_GGUF_TENSOR_Q8_K:    return bn_quant_q8k_scalar_range;
    case BN_GGUF_TENSOR_F32:     return bn_quant_f32_scalar_range;
    case BN_GGUF_TENSOR_F16:     return bn_quant_f16_scalar_range;
    case BN_GGUF_TENSOR_BF16:    return bn_quant_bf16_scalar_range;
    case BN_GGUF_TENSOR_Q4_1:    return bn_quant_q4_1_scalar_range;
    case BN_GGUF_TENSOR_Q5_1:    return bn_quant_q5_1_scalar_range;
    case BN_GGUF_TENSOR_IQ4_NL:  return bn_quant_iq4nl_scalar_range;
    case BN_GGUF_TENSOR_IQ4_XS:  return bn_quant_iq4xs_scalar_range;
    case BN_GGUF_TENSOR_IQ3_XXS: return bn_quant_iq3xxs_scalar_range;
    case BN_GGUF_TENSOR_IQ3_S:   return bn_quant_iq3s_scalar_range;
    case BN_GGUF_TENSOR_IQ2_XXS: return bn_quant_iq2xxs_scalar_range;
    case BN_GGUF_TENSOR_IQ2_XS:  return bn_quant_iq2xs_scalar_range;
    case BN_GGUF_TENSOR_IQ2_S:   return bn_quant_iq2s_scalar_range;
#endif
    default: return NULL;
    }
}
