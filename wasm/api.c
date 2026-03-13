#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"

#include <stdlib.h>
#include <string.h>

static BnModel     g_model;
static BnGGUFFile *g_gguf;
static BnTokenizer g_tokenizer;
static BnSampler   g_sampler;
static int       g_initialized = 0;

EMSCRIPTEN_KEEPALIVE
int bitnet_init(const uint8_t *data, size_t size) {
    if (g_initialized) return -1;

    BnMappedFile mf = bn_platform_load_buffer(data, size);

    g_gguf = bn_gguf_open(mf.data, mf.size);
    if (!g_gguf) return -1;

    if (bn_model_load(&g_model, g_gguf, 2048, 0) != 0) {
        bn_gguf_free(g_gguf);
        return -1;
    }
    g_model.file = mf;

    if (bn_tokenizer_init(&g_tokenizer, g_gguf) != 0) {
        bn_model_free(&g_model);
        bn_gguf_free(g_gguf);
        return -1;
    }

    bn_sampler_init(&g_sampler, g_model.config.vocab_size, 0.0f, 0.9f, 42);
    g_initialized = 1;

    return 0;
}

EMSCRIPTEN_KEEPALIVE
int bitnet_forward_token(int token, int pos) {
    if (!g_initialized) return -1;
    float *logits = bn_transformer_forward(&g_model, token, pos);
    return logits ? 0 : -1;
}

EMSCRIPTEN_KEEPALIVE
float *bitnet_get_logits(void) {
    if (!g_initialized) return NULL;
    return g_model.state.logits;
}

EMSCRIPTEN_KEEPALIVE
int bitnet_sample(float temperature, float topp) {
    if (!g_initialized) return -1;
    g_sampler.temperature = temperature;
    g_sampler.topp = topp;
    return bn_sampler_sample(&g_sampler, g_model.state.logits);
}

EMSCRIPTEN_KEEPALIVE
int bitnet_encode(const char *text, int bos, int *buf, int max) {
    if (!g_initialized) return 0;
    return bn_tokenizer_encode(&g_tokenizer, text, bos, buf, max);
}

EMSCRIPTEN_KEEPALIVE
const char *bitnet_decode(int token) {
    if (!g_initialized) return "";
    return bn_tokenizer_decode(&g_tokenizer, token);
}

EMSCRIPTEN_KEEPALIVE
int bitnet_vocab_size(void) {
    if (!g_initialized) return 0;
    return g_model.config.vocab_size;
}

EMSCRIPTEN_KEEPALIVE
int bitnet_bos_id(void) {
    if (!g_initialized) return -1;
    return g_tokenizer.bos_id;
}

EMSCRIPTEN_KEEPALIVE
int bitnet_eos_id(void) {
    if (!g_initialized) return -1;
    return g_tokenizer.eos_id;
}

EMSCRIPTEN_KEEPALIVE
void bitnet_free(void) {
    if (!g_initialized) return;
    bn_tokenizer_free(&g_tokenizer);
    bn_model_free(&g_model);
    bn_gguf_free(g_gguf);
    g_initialized = 0;
}

// --- Chat state machine ---

#include <stdio.h>

#define CHAT_MAX_TOKENS  (4096 + 64)
#define CHAT_MAX_TURNS   2
#define CHAT_LOOP_BUF    32
#define CHAT_LOOP_NGRAM  4

static int g_chat_pos;
static int g_chat_turn_count;
static int g_chat_tokens[CHAT_MAX_TOKENS];
static int g_loop_buf[CHAT_LOOP_BUF];
static int g_loop_idx;
static int g_gen_count;

EMSCRIPTEN_KEEPALIVE
void bitnet_chat_reset(void) {
    if (!g_initialized) return;
    g_chat_pos = 0;
    g_chat_turn_count = 0;
    g_gen_count = 0;
    g_loop_idx = 0;
    memset(g_loop_buf, -1, sizeof(g_loop_buf));
    bn_sampler_reset_recent(&g_sampler);
    bn_transformer_forward(&g_model, g_tokenizer.bos_id, g_chat_pos);
    g_chat_pos++;
}

EMSCRIPTEN_KEEPALIVE
void bitnet_chat_init(float temp, float topp, float repeat_penalty) {
    if (!g_initialized) return;
    g_sampler.temperature = temp;
    g_sampler.topp = topp;
    if (repeat_penalty > 1.0f)
        bn_sampler_set_repeat_penalty(&g_sampler, repeat_penalty, 64);
    bitnet_chat_reset();
}

EMSCRIPTEN_KEEPALIVE
int bitnet_chat_submit(const char *text) {
    if (!g_initialized) return -1;

    // Encode "User: {text}"
    char user_buf[4096 + 16];
    snprintf(user_buf, sizeof(user_buf), "User: %s", text);
    int n = bn_tokenizer_encode(&g_tokenizer, user_buf, 0,
                                 g_chat_tokens, CHAT_MAX_TOKENS);

    // Append eot_id
    if (n < CHAT_MAX_TOKENS)
        g_chat_tokens[n++] = g_tokenizer.eot_id;

    // Encode "Assistant: "
    int n2 = bn_tokenizer_encode(&g_tokenizer, "Assistant: ", 0,
                                  g_chat_tokens + n, CHAT_MAX_TOKENS - n);
    n += n2;

    // Prefill all prompt tokens
    if (n > 1) {
        bn_transformer_prefill(&g_model, g_chat_tokens, n, g_chat_pos);
        g_chat_pos += n;
    } else if (n == 1) {
        bn_transformer_forward(&g_model, g_chat_tokens[0], g_chat_pos);
        g_chat_pos++;
    }

    // Reset loop state for new generation
    g_gen_count = 0;
    g_loop_idx = 0;
    memset(g_loop_buf, -1, sizeof(g_loop_buf));

    return n;
}

EMSCRIPTEN_KEEPALIVE
int bitnet_chat_next(void) {
    if (!g_initialized) return -2;

    float *logits = g_model.state.logits;
    if (!logits) return -2;

    int next = bn_sampler_sample(&g_sampler, logits);

    // End of turn
    if (next == g_tokenizer.eot_id || next == g_tokenizer.eos_id)
        return -1;

    // Loop detection
    g_loop_buf[g_loop_idx] = next;
    g_loop_idx = (g_loop_idx + 1) % CHAT_LOOP_BUF;
    g_gen_count++;

    if (g_gen_count >= 2 * CHAT_LOOP_NGRAM) {
        int looping = 1;
        for (int k = 0; k < CHAT_LOOP_NGRAM; k++) {
            int a = g_loop_buf[((g_loop_idx - 1 - k) % CHAT_LOOP_BUF + CHAT_LOOP_BUF) % CHAT_LOOP_BUF];
            int b = g_loop_buf[((g_loop_idx - 1 - k - CHAT_LOOP_NGRAM) % CHAT_LOOP_BUF + CHAT_LOOP_BUF) % CHAT_LOOP_BUF];
            if (a != b) { looping = 0; break; }
        }
        if (looping) return -2;
    }

    bn_sampler_accept(&g_sampler, next);

    // Forward pass for next token
    bn_transformer_forward(&g_model, next, g_chat_pos);
    g_chat_pos++;

    return next;
}

EMSCRIPTEN_KEEPALIVE
int bitnet_chat_end_turn(void) {
    if (!g_initialized) return 0;

    // Feed EOT to close assistant turn
    bn_transformer_forward(&g_model, g_tokenizer.eot_id, g_chat_pos);
    g_chat_pos++;

    g_chat_turn_count++;
    if (g_chat_turn_count >= CHAT_MAX_TURNS) {
        bitnet_chat_reset();
        return 1;
    }
    return 0;
}
