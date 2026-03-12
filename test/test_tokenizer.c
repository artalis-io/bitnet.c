#include "platform.h"
#include "gguf.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Build a minimal GGUF with tokenizer metadata
static size_t build_tokenizer_gguf(uint8_t *buf, size_t buf_size) {
    // We'll build a minimal GGUF with:
    // - tokenizer.ggml.tokens: string array ["<bos>", "<eos>", "h", "e", "l", "o", "he", "hel", "lo", "hello"]
    // - tokenizer.ggml.scores: float array [0,0, 0,0,0,0, 1,2,3,4]
    // - tokenizer.ggml.bos_token_id: 0
    // - tokenizer.ggml.eos_token_id: 1

    typedef struct { uint8_t *data; size_t pos; size_t cap; } WB;
    WB wb = { buf, 0, buf_size };

    #define WB_WRITE(wb, d, sz) do { memcpy((wb).data + (wb).pos, d, sz); (wb).pos += sz; } while(0)
    #define WB_U32(wb, v) do { uint32_t _v = v; WB_WRITE(wb, &_v, 4); } while(0)
    #define WB_U64(wb, v) do { uint64_t _v = v; WB_WRITE(wb, &_v, 8); } while(0)
    #define WB_F32(wb, v) do { float _v = v; WB_WRITE(wb, &_v, 4); } while(0)
    #define WB_STR(wb, s) do { uint64_t _l = strlen(s); WB_U64(wb, _l); WB_WRITE(wb, s, _l); } while(0)

    // Header
    WB_U32(wb, 0x46554747);  // magic
    WB_U32(wb, 3);           // version
    WB_U64(wb, 0);           // n_tensors
    WB_U64(wb, 4);           // n_kv

    // KV 1: tokenizer.ggml.tokens (string array)
    WB_STR(wb, "tokenizer.ggml.tokens");
    WB_U32(wb, 9);           // GGUF_TYPE_ARRAY
    WB_U32(wb, 8);           // elem_type = STRING
    WB_U64(wb, 10);          // count
    const char *tokens[] = {"<bos>", "<eos>", "h", "e", "l", "o", "he", "hel", "lo", "hello"};
    for (int i = 0; i < 10; i++) WB_STR(wb, tokens[i]);

    // KV 2: tokenizer.ggml.scores (float array)
    WB_STR(wb, "tokenizer.ggml.scores");
    WB_U32(wb, 9);           // GGUF_TYPE_ARRAY
    WB_U32(wb, 6);           // elem_type = FLOAT32
    WB_U64(wb, 10);          // count
    float scores[] = {0,0, 0,0,0,0, 1,2,3,4};
    WB_WRITE(wb, scores, sizeof(scores));

    // KV 3: bos_token_id
    WB_STR(wb, "tokenizer.ggml.bos_token_id");
    WB_U32(wb, 4);           // GGUF_TYPE_UINT32
    WB_U32(wb, 0);

    // KV 4: eos_token_id
    WB_STR(wb, "tokenizer.ggml.eos_token_id");
    WB_U32(wb, 4);           // GGUF_TYPE_UINT32
    WB_U32(wb, 1);

    #undef WB_WRITE
    #undef WB_U32
    #undef WB_U64
    #undef WB_F32
    #undef WB_STR

    return wb.pos;
}

static void test_tokenizer_init(void) {
    printf("test_tokenizer_init... ");

    uint8_t buf[8192];
    size_t size = build_tokenizer_gguf(buf, sizeof(buf));

    GGUFFile *gf = gguf_open(buf, size);
    assert(gf != NULL);

    Tokenizer t;
    int rc = tokenizer_init(&t, gf);
    assert(rc == 0);
    assert(t.vocab_size == 10);
    assert(t.bos_id == 0);
    assert(t.eos_id == 1);
    assert(strcmp(t.vocab[0], "<bos>") == 0);
    assert(strcmp(t.vocab[9], "hello") == 0);

    tokenizer_free(&t);
    gguf_free(gf);
    printf("PASSED\n");
}

static void test_tokenizer_decode(void) {
    printf("test_tokenizer_decode... ");

    uint8_t buf[8192];
    size_t size = build_tokenizer_gguf(buf, sizeof(buf));
    GGUFFile *gf = gguf_open(buf, size);
    Tokenizer t;
    tokenizer_init(&t, gf);

    assert(strcmp(tokenizer_decode(&t, 0), "<bos>") == 0);
    assert(strcmp(tokenizer_decode(&t, 2), "h") == 0);
    assert(strcmp(tokenizer_decode(&t, 9), "hello") == 0);

    tokenizer_free(&t);
    gguf_free(gf);
    printf("PASSED\n");
}

static void test_tokenizer_encode(void) {
    printf("test_tokenizer_encode... ");

    uint8_t buf[8192];
    size_t size = build_tokenizer_gguf(buf, sizeof(buf));
    GGUFFile *gf = gguf_open(buf, size);
    Tokenizer t;
    tokenizer_init(&t, gf);

    int tokens[32];
    int n;

    // Encode "hello" with BOS
    n = tokenizer_encode(&t, "hello", 1, tokens, 32);
    assert(n >= 2);  // at least BOS + "hello"
    assert(tokens[0] == 0);  // BOS

    // The BPE should merge h+e→he, he+l→hel, hel+lo→hello (score 4 is highest)
    // But exact merge order depends on scores. "hello" has score 4 (highest),
    // so it should eventually merge to a single token
    // tokens[1] should be 9 ("hello")
    assert(tokens[1] == 9);

    tokenizer_free(&t);
    gguf_free(gf);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Tokenizer Tests ===\n");
    test_tokenizer_init();
    test_tokenizer_decode();
    test_tokenizer_encode();
    printf("All tokenizer tests passed!\n");
    return 0;
}
