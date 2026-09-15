#ifndef BN_MODEL_TOKENIZER_POLICY_H
#define BN_MODEL_TOKENIZER_POLICY_H

typedef enum {
    BN_TOKENIZER_PRE_NONE,
    BN_TOKENIZER_PRE_LETTERS,
    BN_TOKENIZER_PRE_LETTERS_MARKS,
    BN_TOKENIZER_PRE_NEWLINES,
} BnTokenizerPre;

BnTokenizerPre bn_model_tokenizer_pretokenizer(const char *tokenizer_pre);

int bn_model_tokenizer_uses_metaspace(const char *tokenizer_model);
int bn_model_tokenizer_default_add_bos(const char *tokenizer_pre,
                                       int has_bos_token);

#endif // BN_MODEL_TOKENIZER_POLICY_H
