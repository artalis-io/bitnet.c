#ifndef BN_BACKEND_SESSION_H
#define BN_BACKEND_SESSION_H

typedef struct BnBackendSession BnBackendSession;

BnBackendSession *bn_backend_session_create(void);
void bn_backend_session_free(BnBackendSession *backend);
void *bn_backend_session_gpu_graph(const BnBackendSession *backend);
void *bn_backend_session_ensure_gpu_graph(BnBackendSession *backend, int cap_ops);
void bn_backend_session_set_gpu_graph(BnBackendSession *backend, void *graph);
void bn_backend_session_release_gpu_graph(BnBackendSession *backend);

#endif // BN_BACKEND_SESSION_H
