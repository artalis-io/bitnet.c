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

#endif // BN_TRANSFORMER_INTERNAL_H
