#include "transformer_logits_internal.h"

#include <stdlib.h>

static int logits_env_top_n(const char *name, int min_value) {
    const char *env = getenv(name);
    if (!env)
        return 0;
    int top_n = atoi(env);
    if (top_n < min_value)
        return 0;
    return top_n > 128 ? 128 : top_n;
}

int bn_transformer_logits_cpu_tied_q6k_refine_top(void) {
    return logits_env_top_n("BN_CPU_TIED_Q6K_REFINE_TOP", 1);
}

int bn_transformer_logits_cpu_tied_q6k_hybrid_top(void) {
    return logits_env_top_n("BN_CPU_TIED_Q6K_HYBRID_TOP", 2);
}

int bn_transformer_logits_cpu_native_tied_quant_enabled(void) {
    return getenv("BN_CPU_NATIVE_TIED_LOGITS") != NULL;
}
