#ifndef BN_MODEL_TOKENIZER_POLICY_H
#define BN_MODEL_TOKENIZER_POLICY_H

int bn_model_tokenizer_uses_metaspace(const char *tokenizer_model);
int bn_model_tokenizer_default_add_bos(const char *tokenizer_pre,
                                       int has_bos_token);

#endif // BN_MODEL_TOKENIZER_POLICY_H
