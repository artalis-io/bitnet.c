#ifndef BN_TOKENIZER_H
#define BN_TOKENIZER_H

#include "gguf.h"
#include "json.h"
#include "model_tokenizer_policy.h"

#define BN_BPE_UNICODE_OFFSET  0x100    // non-printable bytes → U+0100..U+0143
#define BN_BPE_UNICODE_END     0x143
#define BN_TOKENIZER_MAX_EOG   16

typedef struct {
    char  **vocab;
    float  *scores;
    int     vocab_size;
    int     bos_id, eos_id, eot_id;
    int     im_start_id, im_end_id; // ChatML tokens (-1 if absent)
    int     eog_ids[BN_TOKENIZER_MAX_EOG];
    int     n_eog;
    int     add_bos;                // whether to prepend BOS (from GGUF metadata)
    int     chatml;                 // 1 if model uses ChatML template
    int     metaspace;              // 1 for tokenizers that encode spaces as U+2581
    int     max_token_length;
    BnTokenizerPre pretokenizer;
    int    *added_ids;             // owned GGUF control/user-defined tokens
    int     n_added;
    unsigned char added_prefix[256];
    int     normalize_nfc;          // model-owned, pinned Unicode normalization policy
    int     ranked_bpe;
    int     n_merges;
    struct BnTokenizerMerge *merges; // owned, immutable pair ranks after init
    // internal: sorted index for binary search during encoding
    int    *sorted_indices;
} BnTokenizer;

/* Immutable ID-indexed BPE/ByteLevel vocabulary view; decoding only. Borrows
 * the initialized JSON document, which must remain unchanged and outlive all
 * operations. No copying live handles. One bounded metadata allocation at init;
 * strings remain document-owned. Caller owns a zeroed handle. */
typedef struct {void *state;} BnTokenizerVocab;
typedef struct {size_t bytes;int contains_replacement;} BnTokenizerVocabDecodeInfo;
int bn_tokenizer_vocab_plan(int vocab_size,size_t *host_bytes);
int bn_tokenizer_vocab_init_json(BnTokenizerVocab *vocab,const BnJSON *document,int vocab_size,size_t host_budget);
void bn_tokenizer_vocab_free(BnTokenizerVocab *vocab);
int bn_tokenizer_vocab_size(const BnTokenizerVocab *vocab);
const char *bn_tokenizer_vocab_raw(const BnTokenizerVocab *vocab,int token);
/* Per-token UTF-8 lossy decode, matching decode([token], skip_special_tokens=false).
 * Literal U+FFFD also sets the flag. This view does not encode text or retain
 * a multi-token streaming decoder state.
 * Output may contain embedded NUL; bytes are counted, never NUL-terminated.
 * Decode allocates nothing. Invalid ID/capacity preserves outputs. Direct
 * overlap with metadata, current raw piece or written is rejected.
 * Output and written must not alias each other or vocabulary/document storage. */
int bn_tokenizer_vocab_decode_info(const BnTokenizerVocab *vocab,int token,BnTokenizerVocabDecodeInfo *out);
int bn_tokenizer_vocab_decode(const BnTokenizerVocab *vocab,int token,unsigned char *out,size_t capacity,size_t *written);

/* Immutable ByteLevel decode model; owns metadata, borrows JSON strings.
 * Document must outlive model use, and model must outlive its streams.
 * Added entries require a Boolean special flag; the per-token vocabulary API
 * above is unchanged. Caller owns zeroed handles. Budget excludes JSON storage.
 */
typedef struct {void *state;} BnTokenizerDecoder;
int bn_tokenizer_decoder_plan(int vocabulary,size_t *host_bytes);
int bn_tokenizer_decoder_init_json(BnTokenizerDecoder *decoder,const BnJSON *document,int vocabulary,size_t host_budget);
/* ByteLevel GGUF vocabulary bridge. Borrows tokenizer strings, which must
 * outlive decoder/streams. Rejects metaspace models. GGUF CONTROL(type3)
 * values are special; USER_DEFINED(type4) values remain ordinary output.
 * Missing token_type metadata means no skip flags. No GGUF storage borrow is
 * retained after init. Budget and stream contracts match the JSON decoder. */
int bn_tokenizer_decoder_init_gguf(BnTokenizerDecoder *decoder,const BnTokenizer *tokenizer,
    const BnGGUFFile *file,size_t host_budget);
void bn_tokenizer_decoder_free(BnTokenizerDecoder *decoder);
/* Request-local value state. Do not edit fields. Quiescent value copies are
 * allowed for checkpoints; each active request needs independent state.
 * Returned states retain at most three unfinished UTF-8 bytes, matching Python
 * incremental replacement (including its deferred surrogate prefix). No heap
 * allocation. init requires a zeroed/unbound stream.
 */
typedef struct {
    const void *owner;
    unsigned char pending[4];
    unsigned count;
    int skip_special,finished;
} BnTokenizerDecodeStream;
int bn_tokenizer_decode_stream_init(BnTokenizerDecodeStream *stream,const BnTokenizerDecoder *decoder,int skip_special);
int bn_tokenizer_decode_stream_reset(BnTokenizerDecodeStream *stream);
void bn_tokenizer_decode_stream_free(BnTokenizerDecodeStream *stream);
/* Append-safe, complete UTF-8 chunks, length-counted including NUL. Combined
 * output plus finish matches Rust full ByteLevel decode, NOT its DecodeStream
 * chunk timing/replacement heuristic. finish flushes pending bytes using UTF-8
 * replacement and makes the stream terminal until reset. Bad IDs/capacity
 * preserve state/output/written. Output, written and input IDs must be disjoint
 * from mutable state; output/written also from each other and model/document
 * storage. Direct metadata/input/raw-piece overlaps reject. Valid accessible
 * caller memory is required. Every call, including failure, allocates nothing.
 */
int bn_tokenizer_decode_stream_push(const BnTokenizerDecoder *decoder,BnTokenizerDecodeStream *stream,
    const int *tokens,size_t count,int finish,unsigned char *out,size_t capacity,size_t *written);

/* Immutable compressed token map for an architecture-selected key profile.
 * One owned block includes IDs, counted canonical keys and hash metadata;
 * input vocabulary/document may be freed after successful initialization. */
typedef struct {void *state;} BnTokenizerMap;
typedef struct {size_t host_bytes,key_bytes,hash_slots;} BnTokenizerMapWorkspace;
typedef struct {const unsigned char *data;size_t bytes;} BnTokenizerMapKey;
typedef struct {int vocab_size,compressed_vocab;uint32_t mapped_pad;size_t host_bytes,key_bytes;} BnTokenizerMapInfo;
int bn_tokenizer_map_plan(const BnTokenizerVocab *vocab,const BnTokenizerMapSpec *spec,BnTokenizerMapWorkspace *out);
int bn_tokenizer_map_init(BnTokenizerMap *map,const BnTokenizerVocab *vocab,const BnTokenizerMapSpec *spec,size_t host_budget);
void bn_tokenizer_map_free(BnTokenizerMap *map);
const uint32_t *bn_tokenizer_map_ids(const BnTokenizerMap *map);
int bn_tokenizer_map_info(const BnTokenizerMap *map,BnTokenizerMapInfo *out);
int bn_tokenizer_map_key(const BnTokenizerMap *map,int token,BnTokenizerMapKey *out);

/* Immutable native JSON byte BPE encoder. Supports the captured ordered
 * number/CJK/letter-punctuation Split + ByteLevel profile, no normalization,
 * and non-stripping added tokens. Unsupported transformations reject setup.
 * Borrows document strings: document must remain unchanged and outlive encoder.
 * Host budget counts owned metadata, excludes the borrowed JSON document.
 * Raw encoding adds no BOS/EOS/chat template. Inputs are counted valid UTF-8
 * (including NUL). Caller owns zeroed handles; initialized handles may not be copied. */
typedef struct {void *state;} BnTokenizerEncoder;
typedef struct {void *state;} BnTokenizerEncodeScratch;
typedef struct {size_t host_bytes,trie_nodes;int vocabulary,merges;} BnTokenizerEncoderPlan;
int bn_tokenizer_encoder_plan(const BnJSON *document,int vocabulary,BnTokenizerEncoderPlan *out);
int bn_tokenizer_encoder_init_json(BnTokenizerEncoder *encoder,const BnJSON *document,int vocabulary,size_t host_budget);
void bn_tokenizer_encoder_free(BnTokenizerEncoder *encoder);
int bn_tokenizer_encode_scratch_plan(size_t input_capacity,size_t *host_bytes);
int bn_tokenizer_encode_scratch_init(BnTokenizerEncodeScratch *scratch,size_t input_capacity,size_t host_budget);
void bn_tokenizer_encode_scratch_free(BnTokenizerEncodeScratch *scratch);
/* Caller supplies independent scratch for concurrent calls. All operations
 * allocate nothing after initialization. Rejection preserves output/written.
 * Output/written must be disjoint from handles, metadata, scratch, input and
 * ALL borrowed document storage. Input must not alias mutable scratch. */
int bn_tokenizer_encoder_encode(const BnTokenizerEncoder *encoder,BnTokenizerEncodeScratch *scratch,
    const unsigned char *input,size_t bytes,int *tokens,size_t capacity,size_t *written);
/* Diagnostic model pre-tokenizer only (no added-token isolation). Returns
 * cumulative UTF-8 byte ends, matching the source pre_tokenize_str boundaries.
 * Two allocation-free passes preserve outputs on insufficient capacity. */
int bn_tokenizer_encoder_pretokenize(const BnTokenizerEncoder *encoder,const unsigned char *input,
    size_t bytes,uint32_t *ends,size_t capacity,size_t *written);

int         bn_tokenizer_init(BnTokenizer *t, BnGGUFFile *f);
void        bn_tokenizer_free(BnTokenizer *t);
int         bn_tokenizer_encode(const BnTokenizer *t, const char *text, int add_bos,
                             int *tokens, int max_tokens);
/* Reusable GGUF raw BPE workspace, including the pinned NFC policy and added
 * token isolation. Metaspace requires the original encoder and rejects this plan. Binds an
 * immutable tokenizer that outlives scratch; initialized handles may not be
 * copied. Payload includes owned workspace/control storage, not allocator
 * bookkeeping. No allocations during encode; insufficient input/output
 * capacity returns -1 without truncating or changing output. Input/output
 * must be valid, disjoint caller storage and must not alias scratch/tokenizer.
 * One serialized caller per scratch. */
typedef struct {void *state;} BnTokenizerGGUFScratch;
int bn_tokenizer_gguf_scratch_plan(const BnTokenizer *t,size_t input_capacity,size_t *host_bytes);
int bn_tokenizer_gguf_scratch_init(BnTokenizerGGUFScratch *scratch,const BnTokenizer *t,
    size_t input_capacity,size_t host_budget);
void bn_tokenizer_gguf_scratch_free(BnTokenizerGGUFScratch *scratch);
int bn_tokenizer_gguf_encode(const BnTokenizer *t,BnTokenizerGGUFScratch *scratch,
    const char *text,int add_bos,int *tokens,int max_tokens);
const char *bn_tokenizer_decode(const BnTokenizer *t, int token);
int         bn_tokenizer_lookup(const BnTokenizer *t, const char *str);
int         bn_tokenizer_is_eog(const BnTokenizer *t, int token);

#endif // BN_TOKENIZER_H
