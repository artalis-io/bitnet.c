#include "runtime_policy.h"

#include <stdlib.h>
#include <string.h>

static int runtime_entry_is_backend_policy(const char *entry) {
    return entry &&
           (strncmp(entry, "BN_GPU_", 7) == 0 ||
            strncmp(entry, "BN_METAL_", 9) == 0 ||
            strncmp(entry, "BN_CUDA_", 8) == 0);
}

static char *runtime_entry_copy(const char *entry) {
    size_t n = strlen(entry) + 1;
    char *copy = (char *)malloc(n);
    if (copy) memcpy(copy, entry, n);
    return copy;
}

static int runtime_entry_compare(const void *lhs, const void *rhs) {
    const char *const a = *(const char *const *)lhs;
    const char *const b = *(const char *const *)rhs;
    return strcmp(a, b);
}

static int runtime_policy_name_allowed(const char *name) {
    return name && !strchr(name, '=') &&
           (strncmp(name, "BN_GPU_", 7) == 0 ||
            strncmp(name, "BN_METAL_", 9) == 0 ||
            strncmp(name, "BN_CUDA_", 8) == 0);
}

static size_t runtime_policy_find(const BnBackendRuntimePolicy *policy,
                                  const char *name) {
    if (!policy || !name) return (size_t)-1;
    size_t n = strlen(name);
    for (size_t i = 0; i < policy->count; i++) {
        const char *entry = policy->entries[i];
        if (strncmp(entry, name, n) == 0 && entry[n] == '=') return i;
    }
    return (size_t)-1;
}

int bn_backend_runtime_policy_init(BnBackendRuntimePolicy *policy,
                                   char *const *environment) {
    if (!policy) return -1;
    *policy = (BnBackendRuntimePolicy){0};
    if (!environment) return 0;

    size_t count = 0;
    for (size_t i = 0; environment[i]; i++)
        count += runtime_entry_is_backend_policy(environment[i]);
    if (count == 0) return 0;

    policy->entries = (char **)calloc(count, sizeof(*policy->entries));
    if (!policy->entries) return -1;
    for (size_t i = 0; environment[i]; i++) {
        if (!runtime_entry_is_backend_policy(environment[i])) continue;
        char *copy = runtime_entry_copy(environment[i]);
        if (!copy) {
            bn_backend_runtime_policy_free(policy);
            return -1;
        }
        policy->entries[policy->count++] = copy;
    }
    qsort(policy->entries, policy->count, sizeof(*policy->entries),
          runtime_entry_compare);
    return 0;
}

int bn_backend_runtime_policy_clone(BnBackendRuntimePolicy *policy,
                                    const BnBackendRuntimePolicy *source) {
    if (!policy) return -1;
    *policy = (BnBackendRuntimePolicy){0};
    if (!source || source->count == 0) return 0;

    policy->entries =
        (char **)calloc(source->count, sizeof(*policy->entries));
    if (!policy->entries) return -1;
    for (size_t i = 0; i < source->count; i++) {
        char *copy = runtime_entry_copy(source->entries[i]);
        if (!copy) {
            bn_backend_runtime_policy_free(policy);
            return -1;
        }
        policy->entries[policy->count++] = copy;
    }
    return 0;
}

int bn_backend_runtime_policy_set(BnBackendRuntimePolicy *policy,
                                  const char *name,
                                  const char *value,
                                  int overwrite) {
    if (!policy || !runtime_policy_name_allowed(name)) return -1;
    if (!value) value = "1";
    size_t index = runtime_policy_find(policy, name);
    if (index != (size_t)-1 && !overwrite) return 0;

    size_t name_len = strlen(name);
    size_t value_len = strlen(value);
    char *entry = (char *)malloc(name_len + value_len + 2);
    if (!entry) return -1;
    memcpy(entry, name, name_len);
    entry[name_len] = '=';
    memcpy(entry + name_len + 1, value, value_len + 1);

    if (index != (size_t)-1) {
        free(policy->entries[index]);
        policy->entries[index] = entry;
    } else {
        char **entries = (char **)realloc(
            policy->entries, (policy->count + 1) * sizeof(*policy->entries));
        if (!entries) {
            free(entry);
            return -1;
        }
        policy->entries = entries;
        policy->entries[policy->count++] = entry;
    }
    qsort(policy->entries, policy->count, sizeof(*policy->entries),
          runtime_entry_compare);
    return 0;
}

void bn_backend_runtime_policy_unset(BnBackendRuntimePolicy *policy,
                                     const char *name) {
    size_t index = runtime_policy_find(policy, name);
    if (index == (size_t)-1) return;
    free(policy->entries[index]);
    if (index + 1 < policy->count) {
        memmove(policy->entries + index, policy->entries + index + 1,
                (policy->count - index - 1) * sizeof(*policy->entries));
    }
    policy->count--;
    if (policy->count == 0) {
        free(policy->entries);
        policy->entries = NULL;
    }
}

void bn_backend_runtime_policy_free(BnBackendRuntimePolicy *policy) {
    if (!policy) return;
    for (size_t i = 0; i < policy->count; i++) free(policy->entries[i]);
    free(policy->entries);
    *policy = (BnBackendRuntimePolicy){0};
}

const char *bn_backend_runtime_policy_get(
    const BnBackendRuntimePolicy *policy, const char *name) {
    if (!policy || !name || !name[0]) return NULL;
    size_t n = strlen(name);
    size_t lo = 0;
    size_t hi = policy->count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        const char *entry = policy->entries[mid];
        int cmp = strncmp(entry, name, n);
        if (cmp == 0) {
            unsigned char suffix = (unsigned char)entry[n];
            if (suffix == '=') return entry + n + 1;
            cmp = suffix < (unsigned char)'=' ? -1 : 1;
        }
        if (cmp < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return NULL;
}

int bn_backend_runtime_policy_enabled(
    const BnBackendRuntimePolicy *policy, const char *name) {
    return bn_backend_runtime_policy_get(policy, name) != NULL;
}
