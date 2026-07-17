#include "transformer_cpu_backend_internal.h"

#include <stdlib.h>

int bn_transformer_cpu_prepared_qweights_enabled(void) {
    return getenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS") == NULL;
}

const char *bn_transformer_cpu_debug_dump_path(void) {
    const char *path = getenv("BN_DUMP_LAYER_INP");
    return path && path[0] ? path : NULL;
}

int bn_transformer_cpu_debug_dump_pos_selected(int pos) {
    const char *pos_env = getenv("BN_DUMP_LAYER_POS");
    return !pos_env || !pos_env[0] || atoi(pos_env) == pos;
}

int bn_transformer_cpu_debug_dump_heads_enabled(void) {
    const char *enabled = getenv("BN_DUMP_ALL_HEADS");
    return enabled && enabled[0];
}

int bn_transformer_cpu_fused_q4_gateup_silu_allowed(void) {
    return getenv("BN_CPU_LLAMA_DOT") == NULL &&
           getenv("BN_CPU_LLAMA_Q4_DOT") == NULL;
}
