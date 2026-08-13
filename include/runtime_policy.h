#ifndef BN_RUNTIME_POLICY_H
#define BN_RUNTIME_POLICY_H

#include <stddef.h>

typedef struct {
    char **entries;
    size_t count;
} BnBackendRuntimePolicy;

int bn_backend_runtime_policy_init(BnBackendRuntimePolicy *policy,
                                   char *const *environment);
int bn_backend_runtime_policy_clone(BnBackendRuntimePolicy *policy,
                                    const BnBackendRuntimePolicy *source);
int bn_backend_runtime_policy_set(BnBackendRuntimePolicy *policy,
                                  const char *name,
                                  const char *value,
                                  int overwrite);
void bn_backend_runtime_policy_unset(BnBackendRuntimePolicy *policy,
                                     const char *name);
void bn_backend_runtime_policy_free(BnBackendRuntimePolicy *policy);
const char *bn_backend_runtime_policy_get(
    const BnBackendRuntimePolicy *policy, const char *name);
int bn_backend_runtime_policy_enabled(
    const BnBackendRuntimePolicy *policy, const char *name);

#endif // BN_RUNTIME_POLICY_H
