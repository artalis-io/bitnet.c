#ifndef BN_JSON_H
#define BN_JSON_H
#include <stddef.h>
#include <stdint.h>
#define BN_JSON_MAX_BYTES (16u * 1024u * 1024u)
#define BN_JSON_MAX_NODES 65536u
#define BN_JSON_MAX_DEPTH 64u
/* Immutable, owned configuration document. Caller owns a zeroed handle.
 * Input may be discarded after init; failure leaves handle unchanged. No
 * copying initialized handles. Rejects duplicate decoded keys, invalid UTF-8,
 * NUL strings, trailing input, and limits above. Number lexemes are preserved;
 * no locale-dependent or floating point conversion is performed. */
typedef struct { void *state; } BnJSON;
typedef struct BnJSONValue BnJSONValue;
typedef enum { BN_JSON_INVALID=0, BN_JSON_NULL, BN_JSON_BOOL, BN_JSON_NUMBER,
    BN_JSON_STRING, BN_JSON_ARRAY, BN_JSON_OBJECT } BnJSONType;
typedef struct {size_t max_bytes,max_nodes;} BnJSONLimits;
/* Explicit per-document caps for large vocabulary/configuration documents.
 * Depth remains BN_JSON_MAX_DEPTH. Caps bound serialized input/value nodes,
 * not allocator overhead. Default init retains the constants above. */
int bn_json_init_limits(BnJSON *document,const void *data,size_t bytes,const BnJSONLimits *limits);
int bn_json_init(BnJSON *document, const void *data, size_t bytes);
void bn_json_free(BnJSON *document);
const BnJSONValue *bn_json_root(const BnJSON *document);
BnJSONType bn_json_type(const BnJSONValue *value);
const BnJSONValue *bn_json_get(const BnJSONValue *object, const char *key);
const BnJSONValue *bn_json_at(const BnJSONValue *array, size_t index);
/* Object traversal is sorted by decoded key, independent of source order.
 * Returned key/value borrow the document. Invalid access preserves *key. */
const BnJSONValue *bn_json_member(const BnJSONValue *object,size_t index,const char **key);
size_t bn_json_size(const BnJSONValue *value);
const char *bn_json_text(const BnJSONValue *value); /* String or number only. */
int bn_json_u64(const BnJSONValue *value, uint64_t *out); /* Unsigned integer lexeme only. */
int bn_json_bool(const BnJSONValue *value, int *out);
#endif
