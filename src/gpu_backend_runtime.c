#include "gpu_backend.h"
#include "platform.h"
#include "backend_session.h"

void *bn_gpu_backend_resident_request(const BnGPUBackend *gpu,
                                      BnBackendSession *session) {
    if (!bn_gpu_backend_can_resident_request(gpu) || !gpu->ctx || !session)
        return NULL;
    void *request = bn_backend_session_gpu_resources(session, gpu->ctx);
    if (request) return request;
    request = gpu->resident_request_create(gpu->ctx);
    if (!request) return NULL;
    if (bn_backend_session_attach_gpu_resources(session, gpu->ctx, request,
                                               gpu->resident_request_destroy)) {
        gpu->resident_request_destroy(request);
        return NULL;
    }
    return request;
}

int bn_gpu_backend_runtime_policy_init(BnBackendRuntimePolicy *policy) {
    return bn_backend_runtime_policy_init(policy, bn_platform_environment());
}

int bn_gpu_backend_capture_runtime_policy(BnGPUBackend *gpu) {
    if (!gpu) return -1;
    bn_backend_runtime_policy_free(&gpu->runtime_policy);
    return bn_gpu_backend_runtime_policy_init(&gpu->runtime_policy);
}

int bn_gpu_backend_capture_runtime_policy_from(
    BnGPUBackend *gpu, const BnBackendRuntimePolicy *policy) {
    if (!gpu) return -1;
    if (!policy) return bn_gpu_backend_capture_runtime_policy(gpu);
    bn_backend_runtime_policy_free(&gpu->runtime_policy);
    return bn_backend_runtime_policy_clone(&gpu->runtime_policy, policy);
}

void bn_gpu_backend_release_runtime_policy(BnGPUBackend *gpu) {
    if (gpu) bn_backend_runtime_policy_free(&gpu->runtime_policy);
}
