#ifndef BN_GPU_MOE_BRIDGE_H
#define BN_GPU_MOE_BRIDGE_H

#include "model.h"
#include "session.h"

typedef struct {
    void *gate;
    void *up;
    void *down;
    int use_gateup_split;
    int gateup_split_op_code;
} BnGPUMoEExpertBuffers;

typedef struct {
    BnGPUMoEExpertBuffers buffers;
    float weight;
} BnGPUMoEResolvedExpert;

typedef struct {
    const BnMoEExpertMap *expert_map;
    const BnGPUMoEResolvedExpert *experts;
    int n_experts;
    int moe_hidden;
} BnGPUMoEResources;

typedef struct {
    void *buffers[BN_MAX_MOE_K * 3];
    int n_buffers;
} BnGPUMoETemporaryBuffers;

int bn_gpu_moe_bridge_get_expert(BnModel *m,
                                  BnSession *sess,
                                  const BnLayerWeights *lw,
                                  int layer,
                                  int expert_idx,
                                  BnGPUMoETemporaryBuffers *temporaries,
                                  BnGPUMoEExpertBuffers *out);

int bn_gpu_moe_bridge_resolve_resources(BnGPUMoEResources *out,
                                         BnGPUMoEResolvedExpert *expert_storage,
                                         int expert_cap,
                                         BnModel *m,
                                         BnSession *sess,
                                         const BnLayerWeights *lw,
                                         int layer,
                                         BnGPUMoETemporaryBuffers *temporaries);

void bn_gpu_moe_bridge_release_temporaries(
    BnModel *m,
    BnGPUMoETemporaryBuffers *temporaries);

#endif // BN_GPU_MOE_BRIDGE_H
