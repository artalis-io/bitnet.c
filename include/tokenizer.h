#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf.h"

typedef struct {
    char  **vocab;
    float  *scores;
    int     vocab_size;
    int     bos_id, eos_id, eot_id;
    int     max_token_length;
    // internal: sorted index for binary search during encoding
    int    *sorted_indices;
} Tokenizer;

int         tokenizer_init(Tokenizer *t, GGUFFile *f);
void        tokenizer_free(Tokenizer *t);
int         tokenizer_encode(const Tokenizer *t, const char *text, int add_bos,
                             int *tokens, int max_tokens);
const char *tokenizer_decode(const Tokenizer *t, int token);

#endif // TOKENIZER_H
