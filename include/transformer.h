#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "model.h"

// Run one token through the transformer, returns pointer to logits
float *transformer_forward(Model *m, int token, int pos);

#endif // TRANSFORMER_H
