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

int bn_gpu_moe_bridge_get_expert(BnModel *m,
                                  BnSession *sess,
                                  const BnLayerWeights *lw,
                                  int layer,
                                  int expert_idx,
                                  void **uncached_bufs,
                                  int *n_uncached,
                                  BnGPUMoEExpertBuffers *out);

int bn_gpu_moe_bridge_resolve_resources(BnGPUMoEResources *out,
                                         BnGPUMoEResolvedExpert *expert_storage,
                                         int expert_cap,
                                         BnModel *m,
                                         BnSession *sess,
                                         const BnLayerWeights *lw,
                                         int layer,
                                         void **uncached_bufs,
                                         int *n_uncached);

#endif // BN_GPU_MOE_BRIDGE_H
