#ifndef BN_TRANSFORMER_CPU_INTERNAL_H
#define BN_TRANSFORMER_CPU_INTERNAL_H

#include "model.h"
#include "model_run_state.h"
#include "session.h"
#include "transformer_plan_internal.h"

void bn_transformer_cpu_residual_add(const BnCPURuntimePolicy *runtime,
                                     float *x, const float *r, int dim);
/* scratch holds dim floats and must not overlap x or r. On x86, stage
 * the scaled values to preserve the separate multiply/add rounding. */
void bn_transformer_cpu_scaled_residual_add(const BnCPURuntimePolicy *runtime,
                                            float *x, const float *r,
                                            float scale, int dim,
                                            float *scratch);
int bn_transformer_cpu_apply_positional_layer_embedding_host(
    BnModel *model, BnSession *session, BnLayerWeights *layer, int pos);
void bn_transformer_cpu_apply_ffn_activation(const BnCPURuntimePolicy *runtime,
                                             BnRunState *s,
                                             const BnFFNPlan *ffn_plan,
                                             int hidden_dim,
                                             int already_activated);
int bn_transformer_cpu_hyper_connection_mix(
    BnModel *m, BnSession *sess, const BnHyperConnectionWeights *hc,
    int produce_inject);
void bn_transformer_cpu_hyper_connection_combine(
    BnModel *m, BnSession *sess, const float *block_out);
int bn_transformer_cpu_apply_positional_layer_embedding(
    BnModel *m, BnSession *sess, BnLayerWeights *lw, int pos);
int bn_transformer_cpu_prepare_positional_layer_embedding(
    BnModel *m, BnSession *sess, BnLayerWeights *lw, int pos);
int bn_transformer_cpu_apply_positional_layer_embedding_projected(
    BnModel *m, BnSession *sess, BnLayerWeights *lw, int pos,
    const float *key, const float *value);
void bn_transformer_cpu_forward_ssm_block(BnModel *m,
                                          BnSession *sess,
                                          BnLayerWeights *lw,
                                          int layer,
                                          int pos);
void bn_transformer_cpu_forward_ffn_block(BnModel *m,
                                          BnSession *sess,
                                          BnLayerWeights *lw,
                                          int layer,
                                          int pos,
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
