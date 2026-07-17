#include "transformer_prefill_internal.h"
#include "backend_quant.h"

#include <stdlib.h>

int bn_transformer_prefill_profile_enabled(void) {
    return getenv("BN_PREFILL_PROFILE") != NULL;
}

int bn_transformer_prefill_hybrid_batch_allowed(void) {
    return getenv("BN_PREFILL_ALLOW_HYBRID_BATCH") != NULL;
}

int bn_transformer_prefill_force_token_attention_enabled(void) {
    return getenv("BN_PREFILL_FORCE_TOKEN_ATTN") != NULL;
}

int bn_transformer_prefill_can_preq8k_type(const BnPrefillCPUOps *ops,
                                           int tensor_type) {
    return ops && ops->supports_preq8k &&
           bn_backend_quant_can_preq8k(tensor_type);
}

int bn_transformer_prefill_can_preq8k_pair(const BnPrefillCPUOps *ops,
                                           int left_type,
                                           int right_type) {
    return bn_transformer_prefill_can_preq8k_type(ops, left_type) &&
           bn_backend_quant_can_preq8k(right_type);
}

int bn_transformer_prefill_can_preq8k_triple(const BnPrefillCPUOps *ops,
                                             int first_type,
                                             int second_type,
                                             int third_type) {
    return bn_transformer_prefill_can_preq8k_pair(ops, first_type,
                                                  second_type) &&
           bn_backend_quant_can_preq8k(third_type);
}

int bn_transformer_prefill_stacked_pair_same_format(int left_type,
                                                    int right_type) {
    return bn_backend_quant_stacked_pair_same_format(left_type, right_type);
}

int bn_transformer_prefill_uses_float_kquant_fallback(int tensor_type) {
    return bn_backend_quant_is_kquant_float_fallback_candidate(tensor_type);
}
