#ifndef BN_TRANSFORMER_H
#define BN_TRANSFORMER_H

#include "model.h"

// Run one token through the transformer, returns pointer to logits
float *bn_transformer_forward(BnModel *m, int token, int pos);

// Process n_tokens starting at pos0, computing logits only for the last token.
// Returns logits pointer (same as bn_transformer_forward), or NULL on error.
float *bn_transformer_prefill(BnModel *m, const int *tokens, int n_tokens, int pos0);

// Process n_tokens and compute logits at EVERY position.
// all_logits must be [n_tokens * vocab_size] floats (caller-allocated).
// Returns 0 on success, -1 on error. Last token's logits also in m->state.logits.
int bn_transformer_prefill_all(BnModel *m, const int *tokens, int n_tokens,
                                int pos0, float *all_logits);

#endif // BN_TRANSFORMER_H
