#ifndef BN_GPU_CUDA_MLA_ATTENTION_H
#define BN_GPU_CUDA_MLA_ATTENTION_H
#include <cuda_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Private raw-pointer ABI; upstream C++ types stay inside the adapter. */
typedef struct {
    const void *query, *cache;
    const int *indices;
    void *partial, *out;
    float *partial_lse, *lse;
    int n_tokens, multiprocessors;
    cudaStream_t stream;
} BnCudaMLAAttentionCall;
/* Configure this device's kernels before request execution/capture. */
int bn_cuda_mla_attention_prepare(void);
int bn_cuda_mla_attention_forward(const BnCudaMLAAttentionCall *call);
#ifdef __cplusplus
}
#endif
#endif
