#ifndef BN_TRANSFORMER_CPU_INTERNAL_H
#define BN_TRANSFORMER_CPU_INTERNAL_H

#include "model.h"
#include "session.h"
#include "transformer_plan_internal.h"

void bn_transformer_cpu_residual_add(float *x, const float *r, int dim);
void bn_transformer_cpu_apply_ffn_activation(BnRunState *s,
                                             const BnFFNPlan *ffn_plan,
                                             int hidden_dim,
                                             int already_activated);
void bn_transformer_cpu_forward_ssm_block(BnModel *m,
                                          BnSession *sess,
                                          BnLayerWeights *lw,
                                          int layer);
void bn_transformer_cpu_forward_ffn_block(BnModel *m,
                                          BnSession *sess,
                                          BnLayerWeights *lw,
                                          const BnFFNPlan *ffn_plan);
int bn_transformer_cpu_forward_layer(BnModel *m,
                                     BnSession *sess,
                                     int layer,
                                     int pos,
                                     int cache_pos,
                                     int rope_dims,
                                     const float *rope_cos,
                                     const float *rope_sin);

#endif // BN_TRANSFORMER_CPU_INTERNAL_H
