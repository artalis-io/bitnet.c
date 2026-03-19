#ifndef BN_MOE_H
#define BN_MOE_H

#include "model.h"

// Router: compute top-K expert indices and weights from hidden state.
// Writes to ms->expert_indices and ms->expert_weights.
void bn_moe_route(BnMoEState *ms, const float *x, const float *router_w,
                  int dim, int n_experts, int k);

// Full MoE FFN block: route -> load -> compute -> combine.
// Reads from s->x (after norm), writes result to s->xb for residual add.
void bn_moe_forward(BnModel *m, BnLayerWeights *lw, int l);

#endif // BN_MOE_H
