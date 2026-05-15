#ifndef BN_GENERATE_H
#define BN_GENERATE_H

#include "tokenizer.h"
#include "sampler.h"
#include "bn_alloc.h"

typedef struct BnModel BnModel;
// Forward declaration — full definition in session.h
typedef struct BnSession BnSession;

// Callback for streaming token output. Return non-zero to stop generation.
typedef int (*bn_token_callback)(const char *piece, int token_id, void *user_data);

// Chat template format
typedef enum {
    BN_CHAT_AUTO,    // auto-detect from tokenizer (ChatML if im_start/im_end present, else LLaMA)
    BN_CHAT_CHATML,  // <|im_start|>role\n{content}<|im_end|>\n
    BN_CHAT_LLAMA,   // Role: {content}<|eot_id|>
    BN_CHAT_RAW,     // no wrapping — encode content directly (caller handles template)
} BnChatFormat;

// Chat message roles
typedef enum {
    BN_ROLE_SYSTEM,
    BN_ROLE_USER,
    BN_ROLE_ASSISTANT,
} BnChatRole;

// A single chat message (role + content).
typedef struct {
    BnChatRole role;
    const char *content;
} BnChatMessage;

// Stop string configuration for generation.
typedef struct {
    const char **strings;  // array of stop strings (NULL-terminated content, not token IDs)
    int n;                 // number of stop strings
} BnStopStrings;

// Generate tokens autoregressively from pre-computed logits.
// The model must have logits ready (from bn_prefill or bn_transformer_forward).
// pos is updated to reflect the new position after generation.
// stop: optional stop strings (NULL to disable). Generation halts when any
//       stop string appears in the output. The stop string is NOT included
//       in the callback output.
// alloc is used for internal scratch buffers (NULL = stdlib default).
// Returns: number of tokens generated, -1 on loop detected, -2 on error,
//          -3 on stop string match.
int bn_generate(BnModel *model, BnSession *s, BnTokenizer *tok, BnSampler *sampler,
                int max_tokens, int *pos,
                bn_token_callback cb, void *user_data,
                const BnStopStrings *stop,
                BnAllocator *alloc);

// Speculative decoding: draft K tokens with small model, verify with target.
// Both models must have logits ready from prefill. Greedy only (temperature=0).
// alloc is used for verify_logits buffer (NULL = stdlib default).
// pos is updated. Returns: n_generated, -1 on loop, -2 on error.
int bn_generate_speculative(BnModel *target, BnSession *ts,
                            BnModel *draft, BnSession *ds, int draft_k,
                            BnTokenizer *tok, BnSampler *sampler,
                            int max_tokens, int *pos,
                            bn_token_callback cb, void *user_data,
                            BnAllocator *alloc);

// Prefill prompt tokens through the model. Returns logits for the last token,
// or NULL on error. pos is set to pos0 + n_tokens after return.
// If no_prefill is set, runs tokens one at a time (for debugging).
float *bn_prefill(BnModel *model, BnSession *s, const int *tokens, int n_tokens,
                  int pos0, int no_prefill);
int bn_prefill_no_logits(BnModel *model, BnSession *s, const int *tokens,
                         int n_tokens, int pos0, int no_prefill);

// Encode text into tokens. Returns number of tokens written.
// alloc is used for scratch buffer (NULL = stdlib default).
int bn_count_tokens(const BnTokenizer *tok, const char *text,
                    BnAllocator *alloc);

// Format a single user message into a chat turn (legacy convenience wrapper).
// fmt=BN_CHAT_AUTO uses tokenizer's detected format.
// alloc is used for message formatting buffer (NULL = stdlib default).
// Writes encoded tokens into out_tokens[0..max_tokens-1].
// Returns number of tokens written.
int bn_chat_format_turn(const BnTokenizer *tok, BnChatFormat fmt,
                        const char *user_msg,
                        int *out_tokens, int max_tokens,
                        BnAllocator *alloc);

// Format a multi-turn conversation into tokens.
// Encodes all messages in order, appends assistant prompt at the end.
// fmt=BN_CHAT_AUTO uses tokenizer's detected format.
// alloc is used for formatting buffers (NULL = stdlib default).
// Writes encoded tokens into out_tokens[0..max_tokens-1].
// Returns number of tokens written.
int bn_chat_format_messages(const BnTokenizer *tok, BnChatFormat fmt,
                            const BnChatMessage *messages, int n_messages,
                            int *out_tokens, int max_tokens,
                            BnAllocator *alloc);

// Return the end-of-turn token ID for the given format.
// Used to feed into KV cache after assistant response completes.
// Returns -1 if no end-of-turn token exists for the format.
int bn_chat_turn_end_id(const BnTokenizer *tok, BnChatFormat fmt);

// --- Logprobs API ---

#define BN_LOGPROBS_MAX_TOP_K 20

// A single token's log probability entry.
typedef struct {
    int   token_id;
    float logprob;    // natural log probability (ln)
    const char *text; // decoded token text (pointer into tokenizer vocab, not owned)
} BnLogprobEntry;

// Logprobs result for one generated token.
typedef struct {
    BnLogprobEntry chosen;                        // the sampled token
    BnLogprobEntry top[BN_LOGPROBS_MAX_TOP_K];    // top-K alternatives (sorted by logprob, descending)
    int top_k;                                     // number of valid entries in top[]
} BnLogprobs;

// Compute logprobs from raw logits.
// logits: [vocab_size] raw logits from the forward pass.
// chosen_token: the token that was sampled (its logprob is returned in result->chosen).
// top_k: number of top alternatives to return (clamped to BN_LOGPROBS_MAX_TOP_K).
//        Pass 0 to only compute the chosen token's logprob.
// tok: tokenizer for decoding token text (may be NULL, text fields will be NULL).
// result: output struct, filled by this function.
void bn_logprobs_compute(const float *logits, int vocab_size,
                         int chosen_token, int top_k,
                         const BnTokenizer *tok,
                         BnLogprobs *result);

// --- SSE streaming format (OpenAI-compatible) ---

// Format a token piece into an OpenAI-compatible SSE "data:" line.
// Writes to buf (including trailing \n\n). Returns bytes written (excluding NUL).
// piece: token text (NULL for finish chunks with empty delta).
// id: request ID string (NULL -> "chatcmpl-0").
// model: model name (NULL -> "bitnet").
// finish_reason: NULL for normal chunks, "stop"/"length" for final chunk.
// created: unix timestamp (0 to omit from output).
// Returns -1 if buf_size is insufficient.
int bn_format_sse_chunk(char *buf, int buf_size,
                        const char *piece, const char *id,
                        const char *model, const char *finish_reason,
                        long long created);

// Format the SSE stream terminator: "data: [DONE]\n\n"
// Returns bytes written (excluding NUL), or -1 on insufficient buffer.
int bn_format_sse_done(char *buf, int buf_size);

#endif // BN_GENERATE_H
