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

int bn_gpu_moe_bridge_get_expert(BnModel *m,
                                  BnSession *sess,
                                  const BnLayerWeights *lw,
                                  int layer,
                                  int expert_idx,
                                  void **uncached_bufs,
                                  int *n_uncached,
                                  BnGPUMoEExpertBuffers *out);

#endif // BN_GPU_MOE_BRIDGE_H
