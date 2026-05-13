#ifndef BN_BACKEND_MODEL_H
#define BN_BACKEND_MODEL_H

#ifndef BN_GPU_BACKEND_DECLARED
#define BN_GPU_BACKEND_DECLARED
typedef struct BnGPUBackend BnGPUBackend;
#endif

typedef struct BnBackendModel BnBackendModel;
typedef struct BnBackendSession BnBackendSession;
typedef struct BnQWeight BnQWeight;
typedef struct BnPreparedWeight BnPreparedWeight;

typedef enum {
    BN_BACKEND_HANDLE_OUTPUT_NORM = 1,
    BN_BACKEND_HANDLE_TIED_EMBEDDING = 2,
    BN_BACKEND_HANDLE_ATTN_NORM = 3,
    BN_BACKEND_HANDLE_FFN_NORM = 4,
    BN_BACKEND_HANDLE_Q_BIAS = 5,
    BN_BACKEND_HANDLE_K_BIAS = 6,
    BN_BACKEND_HANDLE_V_BIAS = 7,
    BN_BACKEND_HANDLE_QKV_STACKED = 8,
    BN_BACKEND_HANDLE_GATEUP_STACKED = 9,
    BN_BACKEND_HANDLE_SSM_AB_STACKED = 10,
    BN_BACKEND_HANDLE_SSM_QKVZ_STACKED = 11,
    BN_BACKEND_HANDLE_Q_NORM = 12,
    BN_BACKEND_HANDLE_K_NORM = 13,
    BN_BACKEND_HANDLE_ATTN_SUB_NORM = 14,
    BN_BACKEND_HANDLE_FFN_SUB_NORM = 15,
    BN_BACKEND_HANDLE_SSM_CONV1D = 16,
    BN_BACKEND_HANDLE_SSM_DT_BIAS = 17,
    BN_BACKEND_HANDLE_SSM_A_LOG = 18,
    BN_BACKEND_HANDLE_SSM_NORM = 19,
} BnBackendHandleRole;

BnBackendModel *bn_backend_model_create(void);
void bn_backend_model_free(BnBackendModel *backend);
BnGPUBackend *bn_backend_model_gpu(const BnBackendModel *backend);
BnGPUBackend *bn_backend_model_raw_gpu(const BnBackendModel *backend);
void bn_backend_model_bind_gpu(BnBackendModel *backend, BnGPUBackend *gpu);
void bn_backend_model_clear_gpu(BnBackendModel *backend);
void bn_backend_model_release_gpu(BnBackendModel *backend);
void bn_backend_model_set_gpu_disabled(BnBackendModel *backend, int disabled);
int bn_backend_model_register_handle(BnBackendModel *backend,
                                     int layer,
                                     BnBackendHandleRole role,
                                     void *handle);
void *bn_backend_model_handle(const BnBackendModel *backend,
                              int layer,
                              BnBackendHandleRole role);
int bn_backend_model_register_qweight(BnBackendModel *backend,
                                      const BnQWeight *weight,
                                      void *gpu_buf);
void *bn_backend_model_qweight_buf(const BnBackendModel *backend,
                                   const BnQWeight *weight);
int bn_backend_model_register_prepared_qweight(BnBackendModel *backend,
                                               const BnQWeight *weight,
                                               const BnPreparedWeight *prepared);
const BnPreparedWeight *bn_backend_model_prepared_qweight(
    const BnBackendModel *backend,
    const BnQWeight *weight);

BnBackendSession *bn_backend_session_create(void);
void bn_backend_session_free(BnBackendSession *backend);
void *bn_backend_session_gpu_graph(const BnBackendSession *backend);
void *bn_backend_session_ensure_gpu_graph(BnBackendSession *backend, int cap_ops);
void bn_backend_session_set_gpu_graph(BnBackendSession *backend, void *graph);
void bn_backend_session_release_gpu_graph(BnBackendSession *backend);

#endif // BN_BACKEND_MODEL_H
