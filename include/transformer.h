#ifndef BN_TRANSFORMER_H
#define BN_TRANSFORMER_H

#include "model.h"

// Run one token through the transformer, returns pointer to logits
float *bn_transformer_forward(BnModel *m, int token, int pos);

// Process n_tokens starting at pos0, computing logits only for the last token.
// Returns logits pointer (same as bn_transformer_forward), or NULL on error.
float *bn_transformer_prefill(BnModel *m, const int *tokens, int n_tokens, int pos0);

#endif // BN_TRANSFORMER_H
