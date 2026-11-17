#ifndef BN_GPU_CUDA_FA2_H
#define BN_GPU_CUDA_FA2_H

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Private CUDA ABI. No Torch or upstream C++ types escape this adapter. */
typedef struct {
    void *q, *k, *v, *out;
    float *lse, *partial_lse, *partial_out;
    int n_tokens, n_heads, n_kv_heads, kv_tokens, cache_capacity, multiprocessors;
    float scale;
    cudaStream_t stream;
    /* Zero retains singleton selection. Only the host partition heuristic
     * uses cohort geometry; actual KV bounds remain request-local. */
    int split_batch_size, split_kv_tokens;
    /* Zero inherits this row's actual query length. Nonzero selects the
     * cohort's FA2 mode only; query/KV extents and storage stay row-local. */
    int max_query_tokens;
} BnCudaFA2Call;

/* Private indirect decode row: all pointers remain device-resident. */
typedef struct {
    BnCudaFA2Call call;
    const void *kin, *vin, *gate;
    void *gated_out;
    int gate_profile, splits; /* -1 disables gating. */
} BnCudaFA2DecodeRow;

int bn_cuda_fa2_decode_splits(const BnCudaFA2Call *call);
int bn_cuda_fa2_decode_group(const BnCudaFA2DecodeRow *device_rows, int count,
    int max_heads, int max_splits, int max_combine_blocks, cudaStream_t stream);
int bn_cuda_fa2_prepare(void);
int bn_cuda_fa2_forward(const BnCudaFA2Call *call);

#ifdef __cplusplus
}
#endif
#endif
