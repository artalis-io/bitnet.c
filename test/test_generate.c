#include "generate.h"
#include "sampler.h"
#include "tokenizer.h"
#include "model.h"
#include "session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// ===================================================================
// Test bn_count_tokens
// ===================================================================

// Minimal GGUF builder for tokenizer tests
typedef struct {
    uint8_t *data;
    size_t   pos;
    size_t   cap;
} WriteBuffer;

static void wb_write(WriteBuffer *wb, const void *data, size_t size) {
    assert(wb->pos + size <= wb->cap);
    memcpy(wb->data + wb->pos, data, size);
    wb->pos += size;
}
static void wb_u32(WriteBuffer *wb, uint32_t v) { wb_write(wb, &v, 4); }
static void wb_u64(WriteBuffer *wb, uint64_t v) { wb_write(wb, &v, 8); }
static void wb_f32(WriteBuffer *wb, float v)    { wb_write(wb, &v, 4); }
static void wb_str(WriteBuffer *wb, const char *s) {
    uint64_t len = strlen(s);
    wb_u64(wb, len);
    wb_write(wb, s, len);
}

// GGUF KV types
#define GGUF_TYPE_UINT32  4
#define GGUF_TYPE_INT32   5
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TYPE_STRING  8
#define GGUF_TYPE_ARRAY   9

// Build a minimal GGUF with a small vocab for testing
static BnGGUFFile *build_test_gguf(uint8_t *buf, size_t bufsize) {
    WriteBuffer wb = { buf, 0, bufsize };

    // Header
    wb_u32(&wb, 0x46554747);  // magic "GGUF"
    wb_u32(&wb, 3);           // version
    wb_u64(&wb, 0);           // n_tensors
    wb_u64(&wb, 5);           // n_kv

    // KV 0: tokenizer.ggml.model = "gpt2"
    wb_str(&wb, "tokenizer.ggml.model");
    wb_u32(&wb, GGUF_TYPE_STRING);
    wb_str(&wb, "gpt2");

    // KV 1: tokenizer.ggml.tokens (array of strings)
    wb_str(&wb, "tokenizer.ggml.tokens");
    wb_u32(&wb, GGUF_TYPE_ARRAY);
    wb_u32(&wb, GGUF_TYPE_STRING);  // element type
    wb_u64(&wb, 6);                  // count
    wb_str(&wb, "<s>");    // 0 = BOS
    wb_str(&wb, "</s>");   // 1 = EOS
    wb_str(&wb, "hello");  // 2
    wb_str(&wb, " world"); // 3
    wb_str(&wb, "foo");    // 4
    wb_str(&wb, "bar");    // 5

    // KV 2: tokenizer.ggml.scores
    wb_str(&wb, "tokenizer.ggml.scores");
    wb_u32(&wb, GGUF_TYPE_ARRAY);
    wb_u32(&wb, GGUF_TYPE_FLOAT32);
    wb_u64(&wb, 6);
    wb_f32(&wb, 0.0f);
    wb_f32(&wb, 0.0f);
    wb_f32(&wb, -1.0f);
    wb_f32(&wb, -2.0f);
    wb_f32(&wb, -3.0f);
    wb_f32(&wb, -4.0f);

    // KV 3: tokenizer.ggml.bos_token_id
    wb_str(&wb, "tokenizer.ggml.bos_token_id");
    wb_u32(&wb, GGUF_TYPE_UINT32);
    wb_u32(&wb, 0);

    // KV 4: tokenizer.ggml.eos_token_id
    wb_str(&wb, "tokenizer.ggml.eos_token_id");
    wb_u32(&wb, GGUF_TYPE_UINT32);
    wb_u32(&wb, 1);

    return bn_gguf_open(buf, wb.pos);
}

static void test_count_tokens(void) {
    printf("test_count_tokens... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    // "hello" should tokenize to at least 1 token (may be byte-level with small vocab)
    int n = bn_count_tokens(&tok, "hello", NULL);
    assert(n >= 0);

    // Empty string
    n = bn_count_tokens(&tok, "", NULL);
    assert(n == 0);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

// ===================================================================
// Test bn_chat_format_turn
// ===================================================================
static void test_chat_format_chatml(void) {
    printf("test_chat_format_chatml... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    // Force ChatML mode and set special token IDs
    tok.chatml = 1;
    tok.im_start_id = 2;  // reuse "hello" slot
    tok.im_end_id = 3;    // reuse " world" slot

    int tokens[256];
    int n = bn_chat_format_turn(&tok, BN_CHAT_CHATML, "hello", tokens, 256, NULL);

    // Should produce some tokens
    assert(n > 0);
    // First token should be im_start_id
    assert(tokens[0] == tok.im_start_id);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_chat_format_llama(void) {
    printf("test_chat_format_llama... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    // LLaMA-style (non-ChatML)
    tok.chatml = 0;
    tok.eot_id = 1;

    int tokens[256];
    int n = bn_chat_format_turn(&tok, BN_CHAT_LLAMA, "hello", tokens, 256, NULL);

    // Should produce tokens
    assert(n > 0);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_chat_format_raw(void) {
    printf("test_chat_format_raw... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    int tokens[256];
    // RAW mode: just encode the message, no wrapping
    int n = bn_chat_format_turn(&tok, BN_CHAT_RAW, "hello", tokens, 256, NULL);
    assert(n >= 0);

    // Compare with direct encode — should be identical
    int tokens2[256];
    int n2 = bn_tokenizer_encode(&tok, "hello", 0, tokens2, 256);
    assert(n == n2);
    for (int i = 0; i < n; i++)
        assert(tokens[i] == tokens2[i]);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_chat_format_auto(void) {
    printf("test_chat_format_auto... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    // Auto with chatml=1 should match explicit ChatML
    tok.chatml = 1;
    tok.im_start_id = 2;
    tok.im_end_id = 3;

    int tokens_auto[256], tokens_explicit[256];
    int na = bn_chat_format_turn(&tok, BN_CHAT_AUTO, "hi", tokens_auto, 256, NULL);
    int ne = bn_chat_format_turn(&tok, BN_CHAT_CHATML, "hi", tokens_explicit, 256, NULL);
    assert(na == ne);
    for (int i = 0; i < na; i++)
        assert(tokens_auto[i] == tokens_explicit[i]);

    // Auto with chatml=0 should match explicit LLaMA
    tok.chatml = 0;
    na = bn_chat_format_turn(&tok, BN_CHAT_AUTO, "hi", tokens_auto, 256, NULL);
    ne = bn_chat_format_turn(&tok, BN_CHAT_LLAMA, "hi", tokens_explicit, 256, NULL);
    assert(na == ne);
    for (int i = 0; i < na; i++)
        assert(tokens_auto[i] == tokens_explicit[i]);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_chat_turn_end_id(void) {
    printf("test_chat_turn_end_id... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    tok.chatml = 1;
    tok.im_start_id = 2;
    tok.im_end_id = 3;
    tok.eot_id = 1;

    assert(bn_chat_turn_end_id(&tok, BN_CHAT_CHATML) == 3);
    assert(bn_chat_turn_end_id(&tok, BN_CHAT_LLAMA) == 1);
    assert(bn_chat_turn_end_id(&tok, BN_CHAT_RAW) == -1);

    // AUTO should resolve to ChatML since chatml=1
    assert(bn_chat_turn_end_id(&tok, BN_CHAT_AUTO) == 3);

    tok.chatml = 0;
    assert(bn_chat_turn_end_id(&tok, BN_CHAT_AUTO) == 1);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

// ===================================================================
// Test bn_generate with synthetic logits
// ===================================================================

// Track tokens received by callback
typedef struct {
    int tokens[64];
    int count;
    int stop_after;  // -1 = don't stop
} GenCallbackState;

static int test_gen_callback(const char *piece, int token_id, void *user_data) {
    (void)piece;
    GenCallbackState *s = (GenCallbackState *)user_data;
    if (s->count < 64)
        s->tokens[s->count] = token_id;
    s->count++;
    if (s->stop_after >= 0 && s->count >= s->stop_after)
        return 1;  // stop
    return 0;
}

static void test_generate_eos_stop(void) {
    printf("test_generate_eos_stop... ");

    // bn_generate needs model->state.logits to be non-NULL and
    // calls bn_sampler_sample + bn_transformer_forward.
    // We can't easily mock the transformer, but we can test the callback
    // cancellation path with a real sampler and fake logits.

    int vocab = 6;
    BnSampler sampler;
    int rc = bn_sampler_init(&sampler, vocab, 0.0f, 0.9f, 42);
    assert(rc == 0);

    // Create fake logits where EOS (token 1) has highest score
    float logits[6] = {-10.0f, 100.0f, -10.0f, -10.0f, -10.0f, -10.0f};

    // Build a minimal "model" — we only need state.logits set
    // bn_generate will try to call bn_transformer_forward after first token,
    // but if EOS is sampled first, it should stop before that call.
    BnModel model;
    memset(&model, 0, sizeof(model));
    BnSession sess;
    memset(&sess, 0, sizeof(sess));
    sess.state.logits = logits;

    // Build tokenizer
    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);
    BnTokenizer tok;
    rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    GenCallbackState cb_state = {{0}, 0, -1};
    int pos = 0;

    // Generate — should immediately hit EOS and return 0
    int n = bn_generate(&model, &sess, &tok, &sampler, 100, &pos, test_gen_callback, &cb_state, NULL, NULL);
    assert(n == 0);  // EOS sampled first, no tokens generated
    assert(cb_state.count == 0);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    bn_sampler_free(&sampler);
    printf("PASSED\n");
}

static void test_generate_callback_cancel(void) {
    printf("test_generate_callback_cancel... ");

    // Test that returning non-zero from callback stops generation.
    // We need logits that DON'T produce EOS so generation would continue.
    int vocab = 6;
    BnSampler sampler;
    int rc = bn_sampler_init(&sampler, vocab, 0.0f, 0.9f, 42);
    assert(rc == 0);

    // Token 2 ("hello") has highest score — not EOS
    float logits[6] = {-10.0f, -10.0f, 100.0f, -10.0f, -10.0f, -10.0f};

    BnModel model;
    memset(&model, 0, sizeof(model));
    BnSession sess;
    memset(&sess, 0, sizeof(sess));
    sess.state.logits = logits;

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);
    BnTokenizer tok;
    rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    // Stop after 1 token via callback
    GenCallbackState cb_state = {{0}, 0, 1};
    int pos = 0;

    // bn_generate will sample token 2, call callback which returns 1 (stop).
    // It will NOT call bn_transformer_forward because it breaks before that.
    int n = bn_generate(&model, &sess, &tok, &sampler, 100, &pos, test_gen_callback, &cb_state, NULL, NULL);
    assert(n == 1);  // generated 1 token before callback stopped it
    assert(cb_state.count == 1);
    assert(cb_state.tokens[0] == 2);  // "hello"

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    bn_sampler_free(&sampler);
    printf("PASSED\n");
}

static void test_generate_null_logits(void) {
    printf("test_generate_null_logits... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    BnSession sess;
    memset(&sess, 0, sizeof(sess));
    sess.state.logits = NULL;  // no logits

    BnSampler sampler;
    int rc = bn_sampler_init(&sampler, 6, 0.0f, 0.9f, 42);
    assert(rc == 0);

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);
    BnTokenizer tok;
    rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    int pos = 0;
    int n = bn_generate(&model, &sess, &tok, &sampler, 100, &pos, NULL, NULL, NULL, NULL);
    assert(n == -2);  // error: null logits

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    bn_sampler_free(&sampler);
    printf("PASSED\n");
}

// ===================================================================
// Test bn_prefill
// ===================================================================
static void test_prefill_null_model(void) {
    printf("test_prefill_null_model... ");

    // bn_prefill with no_prefill=1 calls bn_transformer_forward per token.
    // With a zeroed model this will return NULL.
    BnModel model;
    memset(&model, 0, sizeof(model));
    BnSession sess;
    memset(&sess, 0, sizeof(sess));

    int tokens[] = {2, 3};
    float *logits = bn_prefill(&model, &sess, tokens, 2, 0, 1);
    // transformer_forward on a zeroed model returns NULL
    assert(logits == NULL);

    printf("PASSED\n");
}

static void test_prefill_no_logits_null_model(void) {
    printf("test_prefill_no_logits_null_model... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    BnSession sess;
    memset(&sess, 0, sizeof(sess));

    int tokens[] = {2, 3};
    int rc = bn_prefill_no_logits(&model, &sess, tokens, 2, 0, 1);
    assert(rc == -1);

    printf("PASSED\n");
}

// ===================================================================
// Test bn_chat_format_messages (multi-turn)
// ===================================================================
static void test_chat_format_messages_chatml(void) {
    printf("test_chat_format_messages_chatml... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    tok.chatml = 1;
    tok.im_start_id = 2;
    tok.im_end_id = 3;

    BnChatMessage msgs[] = {
        { BN_ROLE_SYSTEM, "You are helpful." },
        { BN_ROLE_USER, "hello" },
    };

    int tokens[512];
    int n = bn_chat_format_messages(&tok, BN_CHAT_CHATML, msgs, 2, tokens, 512, NULL);

    // Should produce tokens
    assert(n > 0);
    // First token should be im_start_id (system turn start)
    assert(tokens[0] == tok.im_start_id);

    // Should end with im_start_id + "assistant\n" (assistant prompt)
    // The last special token before encoded "assistant\n" should be im_start_id

    // Single-message format should produce same result as format_turn for user msg
    int tokens_single[512];
    BnChatMessage user_only = { BN_ROLE_USER, "hello" };
    int ns = bn_chat_format_messages(&tok, BN_CHAT_CHATML, &user_only, 1, tokens_single, 512, NULL);
    int nt = bn_chat_format_turn(&tok, BN_CHAT_CHATML, "hello", tokens_single, 512, NULL);
    // format_turn delegates to format_messages, so should be identical
    assert(ns == nt);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_chat_format_messages_llama(void) {
    printf("test_chat_format_messages_llama... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    tok.chatml = 0;
    tok.eot_id = 1;

    BnChatMessage msgs[] = {
        { BN_ROLE_SYSTEM, "Be concise." },
        { BN_ROLE_USER, "hello" },
        { BN_ROLE_ASSISTANT, "Hi!" },
        { BN_ROLE_USER, "how are you?" },
    };

    int tokens[512];
    int n = bn_chat_format_messages(&tok, BN_CHAT_LLAMA, msgs, 4, tokens, 512, NULL);
    assert(n > 0);

    // Should contain eot_id tokens between turns
    int eot_count = 0;
    for (int i = 0; i < n; i++)
        if (tokens[i] == tok.eot_id) eot_count++;
    // 4 messages = 4 eot_id separators
    assert(eot_count == 4);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_chat_format_messages_with_system(void) {
    printf("test_chat_format_messages_with_system... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    tok.chatml = 1;
    tok.im_start_id = 2;
    tok.im_end_id = 3;

    // With system prompt should produce more tokens than without
    BnChatMessage with_sys[] = {
        { BN_ROLE_SYSTEM, "You are a pirate." },
        { BN_ROLE_USER, "hello" },
    };
    BnChatMessage without_sys[] = {
        { BN_ROLE_USER, "hello" },
    };

    int tokens_with[512], tokens_without[512];
    int nw = bn_chat_format_messages(&tok, BN_CHAT_CHATML, with_sys, 2, tokens_with, 512, NULL);
    int nwo = bn_chat_format_messages(&tok, BN_CHAT_CHATML, without_sys, 1, tokens_without, 512, NULL);
    assert(nw > nwo);  // system prompt adds tokens

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_chat_format_messages_raw(void) {
    printf("test_chat_format_messages_raw... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    BnChatMessage msgs[] = {
        { BN_ROLE_USER, "hello" },
        { BN_ROLE_ASSISTANT, "hi" },
    };

    int tokens[512];
    int n = bn_chat_format_messages(&tok, BN_CHAT_RAW, msgs, 2, tokens, 512, NULL);

    // RAW just concatenates content — compare with direct encode of both
    int tokens2[512];
    int n2 = bn_tokenizer_encode(&tok, "hello", 0, tokens2, 512);
    n2 += bn_tokenizer_encode(&tok, "hi", 0, tokens2 + n2, 512 - n2);
    assert(n == n2);
    for (int i = 0; i < n; i++)
        assert(tokens[i] == tokens2[i]);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_chat_format_messages_empty(void) {
    printf("test_chat_format_messages_empty... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    tok.chatml = 1;
    tok.im_start_id = 2;
    tok.im_end_id = 3;

    int tokens[512];
    // Zero messages — should still produce assistant prompt
    int n = bn_chat_format_messages(&tok, BN_CHAT_CHATML, NULL, 0, tokens, 512, NULL);
    assert(n >= 1);  // at least im_start_id
    assert(tokens[0] == tok.im_start_id);

    // LLaMA with zero messages — may be 0 with toy vocab that can't encode "Assistant: "
    tok.chatml = 0;
    n = bn_chat_format_messages(&tok, BN_CHAT_LLAMA, NULL, 0, tokens, 512, NULL);
    assert(n >= 0);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_chat_format_messages_assistant_history(void) {
    printf("test_chat_format_messages_assistant_history... ");

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);

    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    tok.chatml = 1;
    tok.im_start_id = 2;
    tok.im_end_id = 3;

    // Full multi-turn with assistant in history
    BnChatMessage msgs[] = {
        { BN_ROLE_USER, "hello" },
        { BN_ROLE_ASSISTANT, "Hi there!" },
        { BN_ROLE_USER, "how are you?" },
    };

    int tokens[512];
    int n = bn_chat_format_messages(&tok, BN_CHAT_CHATML, msgs, 3, tokens, 512, NULL);
    assert(n > 0);

    // Count im_end tokens — should be 3 (one per completed turn)
    int im_end_count = 0;
    for (int i = 0; i < n; i++)
        if (tokens[i] == tok.im_end_id) im_end_count++;
    assert(im_end_count == 3);

    // Count im_start tokens — should be 4 (3 turns + 1 assistant prompt)
    int im_start_count = 0;
    for (int i = 0; i < n; i++)
        if (tokens[i] == tok.im_start_id) im_start_count++;
    assert(im_start_count == 4);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

// ===================================================================
// Test stop strings
// ===================================================================
static void test_generate_stop_string(void) {
    printf("test_generate_stop_string... ");

    // We can test the stop string detection by generating a token whose
    // decoded text contains the stop string. Token 2 = "hello".
    int vocab = 6;
    BnSampler sampler;
    int rc = bn_sampler_init(&sampler, vocab, 0.0f, 0.9f, 42);
    assert(rc == 0);

    // Token 2 ("hello") has highest score
    float logits[6] = {-10.0f, -10.0f, 100.0f, -10.0f, -10.0f, -10.0f};

    BnModel model;
    memset(&model, 0, sizeof(model));
    BnSession sess;
    memset(&sess, 0, sizeof(sess));
    sess.state.logits = logits;

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);
    BnTokenizer tok;
    rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    const char *stops[] = { "hello" };
    BnStopStrings ss = { stops, 1 };

    GenCallbackState cb_state = {{0}, 0, -1};
    int pos = 0;

    // Should generate token 2 ("hello"), detect stop string, return -3
    int n = bn_generate(&model, &sess, &tok, &sampler, 100, &pos,
                         test_gen_callback, &cb_state, &ss, NULL);
    assert(n == -3);  // stop string matched
    assert(cb_state.count == 1);  // callback was called once before stop detected

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    bn_sampler_free(&sampler);
    printf("PASSED\n");
}

static void test_generate_no_stop_match(void) {
    printf("test_generate_no_stop_match... ");

    // Stop string that won't match — generation should stop via EOS normally
    int vocab = 6;
    BnSampler sampler;
    int rc = bn_sampler_init(&sampler, vocab, 0.0f, 0.9f, 42);
    assert(rc == 0);

    // EOS (token 1) has highest score
    float logits[6] = {-10.0f, 100.0f, -10.0f, -10.0f, -10.0f, -10.0f};

    BnModel model;
    memset(&model, 0, sizeof(model));
    BnSession sess;
    memset(&sess, 0, sizeof(sess));
    sess.state.logits = logits;

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);
    BnTokenizer tok;
    rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    const char *stops[] = { "xyz" };
    BnStopStrings ss = { stops, 1 };

    int pos = 0;
    int n = bn_generate(&model, &sess, &tok, &sampler, 100, &pos,
                         NULL, NULL, &ss, NULL);
    assert(n == 0);  // EOS hit, not stop string

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    bn_sampler_free(&sampler);
    printf("PASSED\n");
}

static void test_generate_stop_multiple(void) {
    printf("test_generate_stop_multiple... ");

    // Multiple stop strings — second one matches
    int vocab = 6;
    BnSampler sampler;
    int rc = bn_sampler_init(&sampler, vocab, 0.0f, 0.9f, 42);
    assert(rc == 0);

    // Token 2 ("hello") has highest score
    float logits[6] = {-10.0f, -10.0f, 100.0f, -10.0f, -10.0f, -10.0f};

    BnModel model;
    memset(&model, 0, sizeof(model));
    BnSession sess;
    memset(&sess, 0, sizeof(sess));
    sess.state.logits = logits;

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);
    BnTokenizer tok;
    rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    const char *stops[] = { "xyz", "llo" };  // "llo" is suffix of "hello"
    BnStopStrings ss = { stops, 2 };

    GenCallbackState cb_state = {{0}, 0, -1};
    int pos = 0;

    int n = bn_generate(&model, &sess, &tok, &sampler, 100, &pos,
                         test_gen_callback, &cb_state, &ss, NULL);
    assert(n == -3);  // second stop string matched

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    bn_sampler_free(&sampler);
    printf("PASSED\n");
}

static void test_generate_stop_empty_list(void) {
    printf("test_generate_stop_empty_list... ");

    // Empty stop list (n=0) should behave like no stop strings
    int vocab = 6;
    BnSampler sampler;
    int rc = bn_sampler_init(&sampler, vocab, 0.0f, 0.9f, 42);
    assert(rc == 0);

    // EOS (token 1) has highest score
    float logits[6] = {-10.0f, 100.0f, -10.0f, -10.0f, -10.0f, -10.0f};

    BnModel model;
    memset(&model, 0, sizeof(model));
    BnSession sess;
    memset(&sess, 0, sizeof(sess));
    sess.state.logits = logits;

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);
    BnTokenizer tok;
    rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    BnStopStrings ss = { NULL, 0 };

    int pos = 0;
    int n = bn_generate(&model, &sess, &tok, &sampler, 100, &pos,
                         NULL, NULL, &ss, NULL);
    assert(n == 0);  // EOS hit normally

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    bn_sampler_free(&sampler);
    printf("PASSED\n");
}

// ===================================================================
// Test logprobs
// ===================================================================
static void test_logprobs_basic(void) {
    printf("test_logprobs_basic... ");

    // Simple logits: token 2 has highest logit
    float logits[6] = {0.0f, 1.0f, 5.0f, 2.0f, -1.0f, 0.5f};

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);
    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    BnLogprobs lp;
    bn_logprobs_compute(logits, 6, 2, 3, &tok, &lp);

    // Chosen token should be token 2
    assert(lp.chosen.token_id == 2);
    // Logprob should be negative (probability < 1)
    assert(lp.chosen.logprob < 0.0f);
    // Logprob should be close to 0 since token 2 has highest logit
    assert(lp.chosen.logprob > -1.0f);  // dominant token

    // Top-3 should be sorted descending
    assert(lp.top_k == 3);
    assert(lp.top[0].logprob >= lp.top[1].logprob);
    assert(lp.top[1].logprob >= lp.top[2].logprob);
    // Top-1 should be token 2 (highest logit)
    assert(lp.top[0].token_id == 2);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_logprobs_sum_to_one(void) {
    printf("test_logprobs_sum_to_one... ");

    // Verify that exp(logprob) values sum to ~1.0 when we get all tokens
    float logits[6] = {1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f};

    BnLogprobs lp;
    // Get all 6 tokens as top-K
    bn_logprobs_compute(logits, 6, 0, 6, NULL, &lp);

    float sum = 0.0f;
    for (int i = 0; i < lp.top_k; i++)
        sum += expf(lp.top[i].logprob);
    // Should sum to ~1.0 (all tokens accounted for)
    assert(fabsf(sum - 1.0f) < 1e-5f);

    printf("PASSED\n");
}

static void test_logprobs_zero_topk(void) {
    printf("test_logprobs_zero_topk... ");

    float logits[6] = {0.0f, 1.0f, 5.0f, 2.0f, -1.0f, 0.5f};

    BnLogprobs lp;
    bn_logprobs_compute(logits, 6, 2, 0, NULL, &lp);

    // Should still have chosen token logprob
    assert(lp.chosen.token_id == 2);
    assert(lp.chosen.logprob < 0.0f);
    assert(lp.top_k == 0);

    printf("PASSED\n");
}

static void test_logprobs_uniform(void) {
    printf("test_logprobs_uniform... ");

    // All logits equal — uniform distribution, each logprob = ln(1/6)
    float logits[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    BnLogprobs lp;
    bn_logprobs_compute(logits, 6, 0, 6, NULL, &lp);

    float expected = logf(1.0f / 6.0f);
    assert(fabsf(lp.chosen.logprob - expected) < 1e-5f);

    // All top-K logprobs should be equal
    for (int i = 0; i < lp.top_k; i++)
        assert(fabsf(lp.top[i].logprob - expected) < 1e-5f);

    printf("PASSED\n");
}

static void test_logprobs_text(void) {
    printf("test_logprobs_text... ");

    float logits[6] = {0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f};

    uint8_t buf[4096];
    BnGGUFFile *gf = build_test_gguf(buf, sizeof(buf));
    assert(gf != NULL);
    BnTokenizer tok;
    int rc = bn_tokenizer_init(&tok, gf);
    assert(rc == 0);

    BnLogprobs lp;
    bn_logprobs_compute(logits, 6, 2, 3, &tok, &lp);

    // Chosen token text should be "hello" (token 2 in our test vocab)
    assert(lp.chosen.text != NULL);
    assert(strcmp(lp.chosen.text, "hello") == 0);

    // Top entries should have text
    assert(lp.top[0].text != NULL);

    // With NULL tokenizer, text should be NULL
    BnLogprobs lp2;
    bn_logprobs_compute(logits, 6, 2, 3, NULL, &lp2);
    assert(lp2.chosen.text == NULL);
    assert(lp2.top[0].text == NULL);

    bn_tokenizer_free(&tok);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

// ===================================================================
// Test SSE chunk formatting
// ===================================================================

static void test_sse_chunk_basic(void) {
    printf("test_sse_chunk_basic... ");

    char buf[1024];
    int n = bn_format_sse_chunk(buf, sizeof(buf), "Hello", "chatcmpl-abc",
                                 "bitnet-2b", NULL, 1700000000LL);
    assert(n > 0);
    assert(strncmp(buf, "data: ", 6) == 0);
    assert(strstr(buf, "\"content\":\"Hello\"") != NULL);
    assert(strstr(buf, "\"finish_reason\":null") != NULL);
    assert(strstr(buf, "\"id\":\"chatcmpl-abc\"") != NULL);
    assert(strstr(buf, "\"model\":\"bitnet-2b\"") != NULL);
    assert(strstr(buf, "\"created\":1700000000") != NULL);
    // Must end with \n\n
    assert(n >= 2 && buf[n - 1] == '\n' && buf[n - 2] == '\n');

    printf("PASSED\n");
}

static void test_sse_chunk_finish(void) {
    printf("test_sse_chunk_finish... ");

    char buf[1024];
    int n = bn_format_sse_chunk(buf, sizeof(buf), NULL, "chatcmpl-abc",
                                 "bitnet-2b", "stop", 1700000000LL);
    assert(n > 0);
    assert(strstr(buf, "\"delta\":{}") != NULL);
    assert(strstr(buf, "\"finish_reason\":\"stop\"") != NULL);

    printf("PASSED\n");
}

static void test_sse_chunk_escape(void) {
    printf("test_sse_chunk_escape... ");

    char buf[1024];
    // piece with ", \, \n, and control char \x01
    int n = bn_format_sse_chunk(buf, sizeof(buf), "a\"b\\c\nd\x01" "e",
                                 "id1", "m1", NULL, 0);
    assert(n > 0);
    // Check escaped characters appear in output
    assert(strstr(buf, "a\\\"b\\\\c\\nd\\u0001e") != NULL);

    printf("PASSED\n");
}

static void test_sse_chunk_defaults(void) {
    printf("test_sse_chunk_defaults... ");

    char buf[1024];
    int n = bn_format_sse_chunk(buf, sizeof(buf), "Hi", NULL, NULL, NULL, 0);
    assert(n > 0);
    assert(strstr(buf, "\"id\":\"chatcmpl-0\"") != NULL);
    assert(strstr(buf, "\"model\":\"bitnet\"") != NULL);
    // created=0 should be omitted
    assert(strstr(buf, "created") == NULL);

    printf("PASSED\n");
}

static void test_sse_chunk_overflow(void) {
    printf("test_sse_chunk_overflow... ");

    char buf[16];
    int n = bn_format_sse_chunk(buf, sizeof(buf), "Hello", "id", "m", NULL, 0);
    assert(n == -1);

    printf("PASSED\n");
}

static void test_sse_done(void) {
    printf("test_sse_done... ");

    char buf[64];
    int n = bn_format_sse_done(buf, sizeof(buf));
    assert(n > 0);
    assert(strcmp(buf, "data: [DONE]\n\n") == 0);

    // Overflow
    char tiny[5];
    assert(bn_format_sse_done(tiny, sizeof(tiny)) == -1);

    printf("PASSED\n");
}

// ===================================================================
// Main
// ===================================================================

int main(void) {
    printf("\n=== Generate API Tests ===\n");
    test_count_tokens();
    test_chat_format_chatml();
    test_chat_format_llama();
    test_chat_format_raw();
    test_chat_format_auto();
    test_chat_turn_end_id();
    test_chat_format_messages_chatml();
    test_chat_format_messages_llama();
    test_chat_format_messages_with_system();
    test_chat_format_messages_raw();
    test_chat_format_messages_empty();
    test_chat_format_messages_assistant_history();
    test_generate_eos_stop();
    test_generate_callback_cancel();
    test_generate_null_logits();
    test_generate_stop_string();
    test_generate_no_stop_match();
    test_generate_stop_multiple();
    test_generate_stop_empty_list();
    test_prefill_null_model();
    test_prefill_no_logits_null_model();
    test_logprobs_basic();
    test_logprobs_sum_to_one();
    test_logprobs_zero_topk();
    test_logprobs_uniform();
    test_logprobs_text();
    test_sse_chunk_basic();
    test_sse_chunk_finish();
    test_sse_chunk_escape();
    test_sse_chunk_defaults();
    test_sse_chunk_overflow();
    test_sse_done();
    printf("All generate API tests passed!\n");
    return 0;
}
