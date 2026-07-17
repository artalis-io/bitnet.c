#include "quant_dispatch_internal.h"
#include <stdlib.h>

int bn_quant_policy_avx512_q5k_vnni_enabled(int rows) {
    const char *v = getenv("BN_AVX512_Q5K_VNNI");
    if (v)
        return v[0] != '\0' && v[0] != '0';
    return rows >= 4096;
}

int bn_quant_policy_avx2_kquant_float_for_tasks(
    const BnMatvecTask *tasks,
    int n_tasks) {
    const char *v = getenv("BN_AVX2_KQUANT_FLOAT");
    if (v && v[0] != '\0' && v[0] != '0')
        return 1;
    for (int i = 0; i < n_tasks; i++) {
        if (tasks[i].flags & BN_MATVEC_TASK_FORCE_FLOAT_KQUANT)
            return 1;
    }
    return 0;
}

int bn_quant_policy_llama_q4_dot_enabled(uint32_t flags) {
    return !(flags & BN_MATVEC_TASK_NATIVE_QUANT) &&
           ((flags & BN_MATVEC_TASK_LLAMA_DOT) ||
            getenv("BN_CPU_LLAMA_DOT") != NULL ||
            getenv("BN_CPU_LLAMA_Q4_DOT") != NULL);
}

int bn_quant_policy_llama_q6_dot_enabled(uint32_t flags) {
    return !(flags & BN_MATVEC_TASK_NATIVE_QUANT) &&
           ((flags & BN_MATVEC_TASK_LLAMA_DOT) ||
            getenv("BN_CPU_LLAMA_DOT") != NULL ||
            getenv("BN_CPU_LLAMA_Q4_DOT") != NULL ||
            getenv("BN_CPU_LLAMA_Q6_DOT") != NULL);
}

int bn_quant_policy_batch_llama_q4_dot_enabled(
    const BnMatvecTask *tasks,
    int n_tasks) {
    int llama_dot = getenv("BN_CPU_LLAMA_DOT") != NULL ||
                    getenv("BN_CPU_LLAMA_Q4_DOT") != NULL;
    for (int t = 0; t < n_tasks; t++)
        llama_dot = llama_dot ||
                    ((tasks[t].flags & BN_MATVEC_TASK_LLAMA_DOT) != 0);
    for (int t = 0; t < n_tasks; t++)
        if (tasks[t].flags & BN_MATVEC_TASK_NATIVE_QUANT)
            llama_dot = 0;
    return llama_dot;
}

int bn_quant_policy_wasm_q4_canonical4_enabled(void) {
    return getenv("BN_WASM_Q4_CANONICAL4") != NULL;
}

int bn_quant_policy_q8_0_matmul_batch_enabled(void) {
    return getenv("BN_DISABLE_Q8_0_MATMUL_BATCH") == NULL;
}
