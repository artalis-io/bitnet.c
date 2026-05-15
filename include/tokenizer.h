#ifndef BN_TOKENIZER_H
#define BN_TOKENIZER_H

#include "gguf.h"

#define BN_BPE_UNICODE_OFFSET  0x100    // non-printable bytes → U+0100..U+0143
#define BN_BPE_UNICODE_END     0x143

typedef struct {
    char  **vocab;
    float  *scores;
    int     vocab_size;
    int     bos_id, eos_id, eot_id;
    int     im_start_id, im_end_id; // ChatML tokens (-1 if absent)
    int     add_bos;                // whether to prepend BOS (from GGUF metadata)
    int     chatml;                 // 1 if model uses ChatML template
    int     metaspace;              // 1 for tokenizers that encode spaces as U+2581
    int     max_token_length;
    // internal: sorted index for binary search during encoding
    int    *sorted_indices;
} BnTokenizer;

int         bn_tokenizer_init(BnTokenizer *t, BnGGUFFile *f);
void        bn_tokenizer_free(BnTokenizer *t);
int         bn_tokenizer_encode(const BnTokenizer *t, const char *text, int add_bos,
                             int *tokens, int max_tokens);
const char *bn_tokenizer_decode(const BnTokenizer *t, int token);
int         bn_tokenizer_lookup(const BnTokenizer *t, const char *str);

#endif // BN_TOKENIZER_H
