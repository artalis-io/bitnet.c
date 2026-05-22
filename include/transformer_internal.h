#ifndef BN_TRANSFORMER_INTERNAL_H
#define BN_TRANSFORMER_INTERNAL_H

// Internal header for top-level transformer orchestration.
// Not part of the public API.

#include "transformer.h"
#include "model.h"

float *bn_transformer_forward_logits(BnModel *m, BnSession *sess);
float *bn_transformer_gpu_forward(BnModel *m,
                                  BnSession *sess,
                                  int token,
                                  int pos);
float *bn_transformer_gpu_forward_no_logits(BnModel *m,
                                            BnSession *sess,
                                            int token,
                                            int pos);
int bn_transformer_gpu_forward_argmax(BnModel *m,
                                      BnSession *sess,
                                      int token,
                                      int pos,
                                      const int *penalty_tokens,
                                      int n_penalty_tokens,
                                      float repeat_penalty,
                                      int *out_token);
int bn_transformer_gpu_upload_kv_cache(BnModel *m, BnSession *sess,
                                       int pos0, int n_tokens);
int bn_transformer_gpu_upload_ssm_state(BnModel *m, BnSession *sess);

#endif // BN_TRANSFORMER_INTERNAL_H
