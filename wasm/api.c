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

static Model     g_model;
static GGUFFile *g_gguf;
static Tokenizer g_tokenizer;
static Sampler   g_sampler;
static int       g_initialized = 0;

EMSCRIPTEN_KEEPALIVE
int bitnet_init(const uint8_t *data, size_t size) {
    if (g_initialized) return -1;

    MappedFile mf = platform_load_buffer(data, size);

    g_gguf = gguf_open(mf.data, mf.size);
    if (!g_gguf) return -1;

    if (model_load(&g_model, g_gguf, 2048) != 0) {
        gguf_free(g_gguf);
        return -1;
    }
    g_model.file = mf;

    if (tokenizer_init(&g_tokenizer, g_gguf) != 0) {
        model_free(&g_model);
        gguf_free(g_gguf);
        return -1;
    }

    sampler_init(&g_sampler, g_model.config.vocab_size, 0.0f, 0.9f, 42);
    g_initialized = 1;

    return 0;
}

EMSCRIPTEN_KEEPALIVE
int bitnet_forward_token(int token, int pos) {
    if (!g_initialized) return -1;
    transformer_forward(&g_model, token, pos);
    return 0;
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
    return sampler_sample(&g_sampler, g_model.state.logits);
}

EMSCRIPTEN_KEEPALIVE
int bitnet_encode(const char *text, int bos, int *buf, int max) {
    if (!g_initialized) return 0;
    return tokenizer_encode(&g_tokenizer, text, bos, buf, max);
}

EMSCRIPTEN_KEEPALIVE
const char *bitnet_decode(int token) {
    if (!g_initialized) return "";
    return tokenizer_decode(&g_tokenizer, token);
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
    tokenizer_free(&g_tokenizer);
    model_free(&g_model);
    gguf_free(g_gguf);
    g_initialized = 0;
}
