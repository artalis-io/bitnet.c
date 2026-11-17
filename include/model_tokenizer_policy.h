#ifndef BN_MODEL_TOKENIZER_POLICY_H
#define BN_MODEL_TOKENIZER_POLICY_H

typedef enum {
    BN_TOKENIZER_PRE_NONE,
    BN_TOKENIZER_PRE_LETTERS,
    BN_TOKENIZER_PRE_LETTERS_MARKS,
    BN_TOKENIZER_PRE_NEWLINES,
    BN_TOKENIZER_PRE_GLM4,
} BnTokenizerPre;

BnTokenizerPre bn_model_tokenizer_pretokenizer(const char *tokenizer_pre);

const char *bn_model_tokenizer_extra_eog(const char *tokenizer_pre, int index);

int bn_model_tokenizer_normalizes_nfc(const char *tokenizer_pre);

int bn_model_tokenizer_uses_metaspace(const char *tokenizer_model);
int bn_model_tokenizer_default_add_bos(const char *tokenizer_pre,
                                       int has_bos_token);

/* V4.1: Unicode9 NFKD/mark removal, per-character Unicode17 lowercase,
 * source ASCII whitespace collapse, Unicode whitespace trim and PUA escape.
 * Replacement-token and empty-normalization fallbacks precede compression. */
typedef enum {BN_TOKENIZER_MAP_NONE=0,BN_TOKENIZER_MAP_ENGRAM_V41=1} BnTokenizerMapProfile;
typedef struct {
    BnTokenizerMapProfile profile;
    int vocab_size,compressed_vocab,pad_token; /* pad_token is the original ID. */
} BnTokenizerMapSpec;

#endif // BN_MODEL_TOKENIZER_POLICY_H
