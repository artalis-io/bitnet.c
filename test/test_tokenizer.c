#include "platform.h"
#include "gguf.h"
#include "tokenizer.h"
#include "model_tokenizer_policy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Build a minimal GGUF with tokenizer metadata
static size_t build_tokenizer_gguf_scores(uint8_t *buf, size_t buf_size, uint32_t score_elem_type) {
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
    WB_U32(wb, 9);           // BN_GGUF_TYPE_ARRAY
    WB_U32(wb, 8);           // elem_type = STRING
    WB_U64(wb, 10);          // count
    const char *tokens[] = {"<bos>", "<eos>", "h", "e", "l", "o", "he", "hel", "lo", "hello"};
    for (int i = 0; i < 10; i++) WB_STR(wb, tokens[i]);

    // KV 2: tokenizer.ggml.scores (float array)
    WB_STR(wb, "tokenizer.ggml.scores");
    WB_U32(wb, 9);           // BN_GGUF_TYPE_ARRAY
    WB_U32(wb, score_elem_type);
    WB_U64(wb, 10);          // count
    if (score_elem_type == BN_GGUF_TYPE_FLOAT32) {
        float scores[] = {0,0, 0,0,0,0, 1,2,3,4};
        WB_WRITE(wb, scores, sizeof(scores));
    } else {
        uint32_t scores[] = {0,0, 0,0,0,0, 1,2,3,4};
        WB_WRITE(wb, scores, sizeof(scores));
    }

    // KV 3: bos_token_id
    WB_STR(wb, "tokenizer.ggml.bos_token_id");
    WB_U32(wb, 4);           // BN_GGUF_TYPE_UINT32
    WB_U32(wb, 0);

    // KV 4: eos_token_id
    WB_STR(wb, "tokenizer.ggml.eos_token_id");
    WB_U32(wb, 4);           // BN_GGUF_TYPE_UINT32
    WB_U32(wb, 1);

    #undef WB_WRITE
    #undef WB_U32
    #undef WB_U64
    #undef WB_F32
    #undef WB_STR

    return wb.pos;
}

static size_t build_tokenizer_gguf(uint8_t *buf, size_t buf_size) {
    return build_tokenizer_gguf_scores(buf, buf_size, BN_GGUF_TYPE_FLOAT32);
}

static size_t build_gemma4_tokenizer_gguf(uint8_t *buf, size_t buf_size) {
    typedef struct { uint8_t *data; size_t pos; size_t cap; } WB;
    WB wb = { buf, 0, buf_size };

    #define WB_WRITE(wb, d, sz) do { memcpy((wb).data + (wb).pos, d, sz); (wb).pos += sz; } while(0)
    #define WB_U32(wb, v) do { uint32_t _v = v; WB_WRITE(wb, &_v, 4); } while(0)
    #define WB_U64(wb, v) do { uint64_t _v = v; WB_WRITE(wb, &_v, 8); } while(0)
    #define WB_F32(wb, v) do { float _v = v; WB_WRITE(wb, &_v, 4); } while(0)
    #define WB_STR(wb, s) do { uint64_t _l = strlen(s); WB_U64(wb, _l); WB_WRITE(wb, s, _l); } while(0)

    const char *tokens[] = {
        "<pad>", "<eos>", "<bos>", "T", "h", "e", "Th", "The",
        "▁", "i", "s", "▁i", "▁is"
    };
    float scores[] = {
        -1000, -1000, -1000, 0, 0, 0, 10, 20, 0, 0, 0, 30, 40
    };
    uint64_t n_vocab = sizeof(tokens) / sizeof(tokens[0]);

    WB_U32(wb, 0x46554747);
    WB_U32(wb, 3);
    WB_U64(wb, 0);
    WB_U64(wb, 5);

    WB_STR(wb, "tokenizer.ggml.model");
    WB_U32(wb, BN_GGUF_TYPE_STRING);
    WB_STR(wb, "gemma4");

    WB_STR(wb, "tokenizer.ggml.tokens");
    WB_U32(wb, BN_GGUF_TYPE_ARRAY);
    WB_U32(wb, BN_GGUF_TYPE_STRING);
    WB_U64(wb, n_vocab);
    for (uint64_t i = 0; i < n_vocab; i++) WB_STR(wb, tokens[i]);

    WB_STR(wb, "tokenizer.ggml.scores");
    WB_U32(wb, BN_GGUF_TYPE_ARRAY);
    WB_U32(wb, BN_GGUF_TYPE_FLOAT32);
    WB_U64(wb, n_vocab);
    WB_WRITE(wb, scores, sizeof(scores));

    WB_STR(wb, "tokenizer.ggml.bos_token_id");
    WB_U32(wb, BN_GGUF_TYPE_UINT32);
    WB_U32(wb, 2);

    WB_STR(wb, "tokenizer.ggml.eos_token_id");
    WB_U32(wb, BN_GGUF_TYPE_UINT32);
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

    BnGGUFFile *gf = bn_gguf_open(buf, size);
    assert(gf != NULL);

    BnTokenizer t;
    int rc = bn_tokenizer_init(&t, gf);
    assert(rc == 0);
    assert(t.vocab_size == 10);
    assert(t.bos_id == 0);
    assert(t.eos_id == 1);
    assert(bn_tokenizer_is_eog(&t, t.eos_id));
    assert(!bn_tokenizer_is_eog(&t, 2));
    assert(strcmp(t.vocab[0], "<bos>") == 0);
    assert(bn_model_tokenizer_default_add_bos(NULL, 1));
    assert(!bn_model_tokenizer_default_add_bos("qwen35", 1));
    assert(bn_model_tokenizer_pretokenizer(NULL) == BN_TOKENIZER_PRE_NONE);
    assert(bn_model_tokenizer_pretokenizer("unknown") == BN_TOKENIZER_PRE_NONE);
    assert(bn_model_tokenizer_pretokenizer("qwen2") == BN_TOKENIZER_PRE_LETTERS);
    assert(bn_model_tokenizer_pretokenizer("qwen35") == BN_TOKENIZER_PRE_LETTERS_MARKS);
    assert(strcmp(t.vocab[9], "hello") == 0);

    bn_tokenizer_free(&t);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_tokenizer_decode(void) {
    printf("test_tokenizer_decode... ");

    uint8_t buf[8192];
    size_t size = build_tokenizer_gguf(buf, sizeof(buf));
    BnGGUFFile *gf = bn_gguf_open(buf, size);
    BnTokenizer t;
    bn_tokenizer_init(&t, gf);

    assert(strcmp(bn_tokenizer_decode(&t, 0), "<bos>") == 0);
    assert(strcmp(bn_tokenizer_decode(&t, 2), "h") == 0);
    assert(strcmp(bn_tokenizer_decode(&t, 9), "hello") == 0);

    bn_tokenizer_free(&t);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_tokenizer_encode(void) {
    printf("test_tokenizer_encode... ");

    uint8_t buf[8192];
    size_t size = build_tokenizer_gguf(buf, sizeof(buf));
    BnGGUFFile *gf = bn_gguf_open(buf, size);
    BnTokenizer t;
    bn_tokenizer_init(&t, gf);

    int tokens[32];
    int n;

    // Encode "hello" with BOS
    n = bn_tokenizer_encode(&t, "hello", 1, tokens, 32);
    assert(n >= 2);  // at least BOS + "hello"
    assert(tokens[0] == 0);  // BOS

    // The BPE should merge h+e→he, he+l→hel, hel+lo→hello (score 4 is highest)
    // But exact merge order depends on scores. "hello" has score 4 (highest),
    // so it should eventually merge to a single token
    // tokens[1] should be 9 ("hello")
    assert(tokens[1] == 9);

    bn_tokenizer_free(&t);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_tokenizer_encode_empty(void) {
    printf("test_tokenizer_encode_empty... ");

    uint8_t buf[8192];
    size_t size = build_tokenizer_gguf(buf, sizeof(buf));
    BnGGUFFile *gf = bn_gguf_open(buf, size);
    BnTokenizer t;
    bn_tokenizer_init(&t, gf);

    int tokens[32];
    int n;

    // Empty string with BOS should return just BOS
    n = bn_tokenizer_encode(&t, "", 1, tokens, 32);
    assert(n == 1);
    assert(tokens[0] == 0);  // BOS

    // Empty string without BOS
    n = bn_tokenizer_encode(&t, "", 0, tokens, 32);
    assert(n == 0);

    bn_tokenizer_free(&t);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_tokenizer_encode_max_tokens(void) {
    printf("test_tokenizer_encode_max_tokens... ");

    uint8_t buf[8192];
    size_t size = build_tokenizer_gguf(buf, sizeof(buf));
    BnGGUFFile *gf = bn_gguf_open(buf, size);
    BnTokenizer t;
    bn_tokenizer_init(&t, gf);

    int tokens[2];
    int n;

    // Encode with very small buffer — should not overflow
    n = bn_tokenizer_encode(&t, "hello", 1, tokens, 2);
    assert(n <= 2);
    assert(tokens[0] == 0);  // BOS

    bn_tokenizer_free(&t);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_tokenizer_decode_oob(void) {
    printf("test_tokenizer_decode_oob... ");

    uint8_t buf[8192];
    size_t size = build_tokenizer_gguf(buf, sizeof(buf));
    BnGGUFFile *gf = bn_gguf_open(buf, size);
    BnTokenizer t;
    bn_tokenizer_init(&t, gf);

    // Out-of-bounds decode should return empty string
    const char *piece = bn_tokenizer_decode(&t, -1);
    assert(piece != NULL);
    assert(strlen(piece) == 0);

    piece = bn_tokenizer_decode(&t, 999);
    assert(piece != NULL);
    assert(strlen(piece) == 0);

    bn_tokenizer_free(&t);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_tokenizer_scores_wrong_type(void) {
    printf("test_tokenizer_scores_wrong_type... ");

    uint8_t buf[8192];
    size_t size = build_tokenizer_gguf_scores(buf, sizeof(buf), BN_GGUF_TYPE_UINT32);
    BnGGUFFile *gf = bn_gguf_open(buf, size);
    assert(gf != NULL);

    BnTokenizer t;
    int rc = bn_tokenizer_init(&t, gf);
    assert(rc == 0);
    for (int i = 0; i < t.vocab_size; i++) {
        assert(t.scores[i] == 0.0f);
    }

    bn_tokenizer_free(&t);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

static void test_tokenizer_gemma4_metaspace(void) {
    printf("test_tokenizer_gemma4_metaspace... ");

    uint8_t buf[8192];
    size_t size = build_gemma4_tokenizer_gguf(buf, sizeof(buf));
    BnGGUFFile *gf = bn_gguf_open(buf, size);
    assert(gf != NULL);

    BnTokenizer t;
    int rc = bn_tokenizer_init(&t, gf);
    assert(rc == 0);
    assert(t.metaspace == 1);

    int tokens[8];
    int n = bn_tokenizer_encode(&t, "The is", 1, tokens, 8);
    assert(n == 3);
    assert(tokens[0] == 2);
    assert(tokens[1] == 7);
    assert(tokens[2] == 12);
    assert(strcmp(bn_tokenizer_decode(&t, 12), " is") == 0);

    bn_tokenizer_free(&t);
    bn_gguf_free(gf);
    printf("PASSED\n");
}

/* Deliberately admit cross-boundary vocabulary entries and misleading scores.
 * Only pretokenization and the explicit pair table may authorize a merge. */
static BnGGUFFile *build_ranked_typed_fixture(uint8_t *buf, const char *model, const char *pre,
                                      const char *const *vocab, size_t nv,
                                      const char *const *merges, size_t nm, const int32_t *types) {
    size_t pos = 0;
    #define PUT(d,n) do { memcpy(buf + pos, (d), (n)); pos += (n); } while (0)
    #define U32(v) do { uint32_t x_ = (v); PUT(&x_, 4); } while (0)
    #define U64(v) do { uint64_t x_ = (v); PUT(&x_, 8); } while (0)
    #define STR(s) do { const char *s_ = (s); size_t n_ = strlen(s_); U64(n_); PUT(s_, n_); } while (0)
    U32(0x46554747); U32(3); U64(0); U64((pre ? 4 : 3) + (types ? 1 : 0));
    STR("tokenizer.ggml.model"); U32(BN_GGUF_TYPE_STRING); STR(model);
    if (pre) { STR("tokenizer.ggml.pre"); U32(BN_GGUF_TYPE_STRING); STR(pre); }
    STR("tokenizer.ggml.tokens"); U32(BN_GGUF_TYPE_ARRAY);
    U32(BN_GGUF_TYPE_STRING); U64(nv);
    for (size_t i = 0; i < nv; i++) STR(vocab[i]);
    STR("tokenizer.ggml.merges"); U32(BN_GGUF_TYPE_ARRAY);
    U32(BN_GGUF_TYPE_STRING); U64(nm);
    for (size_t i = 0; i < nm; i++) STR(merges[i]);
    if (types) {
        STR("tokenizer.ggml.token_type"); U32(BN_GGUF_TYPE_ARRAY);
        U32(BN_GGUF_TYPE_INT32); U64(nv);
        for (size_t i = 0; i < nv; i++) U32(types[i]);
    }
    #undef PUT
    #undef U32
    #undef U64
    #undef STR
    return bn_gguf_open(buf, pos);
}

static BnGGUFFile *build_ranked_fixture(uint8_t *buf, const char *model, const char *pre,
                                      const char *const *vocab, size_t nv,
                                      const char *const *merges, size_t nm) {
    return build_ranked_typed_fixture(buf, model, pre, vocab, nv, merges, nm, NULL);
}

static void assert_encoding(BnTokenizer *t, const char *text,
                            const int *expected, int count) {
    int actual[128];
    int n = bn_tokenizer_encode(t, text, 0, actual, 128);
    assert(n == count);
    assert(memcmp(actual, expected, (size_t)count * sizeof(int)) == 0);
    if(!t->metaspace) {
        BnTokenizerGGUFScratch scratch={0};size_t bytes,capacity=strlen(text);if(!capacity)capacity=1;
        assert(!bn_tokenizer_gguf_scratch_plan(t,capacity,&bytes));
        assert(!bn_tokenizer_gguf_scratch_init(&scratch,t,capacity,bytes));
        for(int reuse=0;reuse<3;reuse++) {
            memset(actual,0x5a,sizeof(actual));
            assert(bn_tokenizer_gguf_encode(t,&scratch,text,0,actual,128)==count);
            assert(!memcmp(actual,expected,(size_t)count*sizeof(int)));
        }
        if(count>1) {
            memset(actual,0x5a,sizeof(actual));int before[128];memcpy(before,actual,sizeof(actual));
            assert(bn_tokenizer_gguf_encode(t,&scratch,text,0,actual,count-1)==-1&&!memcmp(actual,before,sizeof(actual)));
        }
        bn_tokenizer_gguf_scratch_free(&scratch);
    }
    /* Truncation must preserve the prefix and caller's guard. */
    for (int cap = 1; cap <= count; cap++) {
        actual[cap] = -123;
        n = bn_tokenizer_encode(t, text, 0, actual, cap);
        assert(n == cap && actual[cap] == -123);
        assert(memcmp(actual, expected, (size_t)cap * sizeof(int)) == 0);
    }
}

static void test_tokenizer_ranked_bpe(void) {
    printf("test_tokenizer_ranked_bpe... ");
    const char *vocab[] = {"a", "b", "c", "ab", "bc", "abc", "aa"};
    const char *merges[] = {"b c", "a b", "a a"};
    uint8_t buf[8192];
    BnGGUFFile *f = build_ranked_fixture(buf, "gpt2", "qwen2", vocab, 7, merges, 3);
    assert(f);
    BnTokenizer t;
    assert(bn_tokenizer_init(&t, f) == 0);
    bn_gguf_free(f); /* The tokenizer owns its merge table. */
    t.scores[3] = 100; t.scores[5] = 1000;
    const int abc[] = {0, 4}, aaa[] = {6, 0};
    assert_encoding(&t, "abc", abc, 2);
    assert_encoding(&t, "aaa", aaa, 2);
    bn_tokenizer_free(&t);
    f = build_ranked_fixture(buf, "gpt2", "qwen2", vocab, 7, NULL, 0);
    assert(f && bn_tokenizer_init(&t, f) == 0);
    const int no_merges[] = {0, 1, 2};
    assert_encoding(&t, "abc", no_merges, 3);
    bn_tokenizer_free(&t); bn_gguf_free(f);
    printf("PASSED\n");
}

static void test_tokenizer_qwen_boundaries(void) {
    printf("test_tokenizer_qwen_boundaries... ");
    const char *vocab[] = {
        "Ġ", "ĠĠ", "ĠĠĠ", "ĠĠĠĠ", "x", "Ġx", "ĠĠĠĠx", // 0..6
        "1", "2", "12", "'", "s", "'s", "x'", "x's",     // 7..14
        "!", "Ċ", "!Ċ", "ĉ", "ĉĉ", "ĉx",                 // 15..20
        "a", "Ì", "ģ", "Ìģ", "aÌģ", "b", "aÌģb", "Ìģb", // 21..28
        "Â", "ł", "Âł", "ÂłÂł", "Âłx"                   // 29..33
    };
    const char *merges[] = {
        "Ġ Ġ", "ĠĠ Ġ", "ĠĠ ĠĠ", "ĠĠĠ Ġ", "Ġ x", "ĠĠĠĠ x",
        "1 2", "x '", "' s", "x' s", "! Ċ", "ĉ ĉ", "ĉ x",
        "Ì ģ", "a Ìģ", "aÌģ b", "Ìģ b", "Â ł", "Âł Âł", "Âł x"
    };
    for (int version = 1; version <= 2; version++) {
        uint8_t buf[8192];
        BnGGUFFile *f = build_ranked_fixture(buf, "gpt2", version == 1 ? "qwen2" : "qwen35",
            vocab, sizeof(vocab)/sizeof(vocab[0]), merges, sizeof(merges)/sizeof(merges[0]));
        assert(f);
        BnTokenizer t;
        assert(bn_tokenizer_init(&t, f) == 0);
        const int indent[] = {2, 5}, digits[] = {7, 8}, contraction[] = {4, 12};
        const int newline[] = {17, 2, 5}, tabs[] = {18, 20}, trailing[] = {3};
        const int nbsp[] = {31, 33};
        assert_encoding(&t, "    x", indent, 2);
        assert_encoding(&t, "12", digits, 2);
        assert_encoding(&t, "x's", contraction, 2);
        assert_encoding(&t, "!\n    x", newline, 3);
        assert_encoding(&t, "\t\tx", tabs, 2);
        assert_encoding(&t, "    ", trailing, 1);
        assert_encoding(&t, "\xc2\xa0\xc2\xa0x", nbsp, 2);
        /* Isolate the existing regex boundary contract. Complete qwen35 NFC
         * encoding is independently checked by the Unicode conformance suite. */
        assert(t.normalize_nfc == (version == 2));
        t.normalize_nfc = 0;
        const int qwen2_marks[] = {21, 28}, qwen35_marks[] = {27};
        assert_encoding(&t, "a\xcc\x81" "b", version == 1 ? qwen2_marks : qwen35_marks,
                        version == 1 ? 2 : 1);
        bn_tokenizer_free(&t); bn_gguf_free(f);
    }
    printf("PASSED\n");
}

static void test_tokenizer_raw_bpe(void) {
    printf("test_tokenizer_raw_bpe... ");
    const char *vocab[] = {"a", "b", "c", "ab", "bc", "abc", "▁", "▁a",
        "\n", "\n\n", "a\n", "\na", "<0x0D>", "<0xF0>", "<0x9F>", "<0x99>", "<0x82>"};
    const char *merges[] = {"b c", "a b", "▁ a"};
    uint8_t buf[8192];
    /* Real metadata may identify the tokenizer model without a pre key. */
    BnGGUFFile *f = build_ranked_fixture(buf, "gemma4", NULL, vocab, 17, merges, 3);
    assert(f);
    BnTokenizer t;
    assert(bn_tokenizer_init(&t, f) == 0);
    assert(t.pretokenizer == BN_TOKENIZER_PRE_NEWLINES && t.ranked_bpe);
    t.scores[5] = t.scores[10] = t.scores[11] = 1000;
    const int abc[] = {0, 4}, space[] = {7}, lines[] = {0, 9, 7};
    const int unlisted_run[] = {0, 8, 8, 8, 7}, bytes[] = {12, 13, 14, 15, 16};
    assert_encoding(&t, "abc", abc, 2);
    assert_encoding(&t, " a", space, 1);
    assert_encoding(&t, "a\n\n a", lines, 3);
    assert_encoding(&t, "a\n\n\n a", unlisted_run, 5);
    assert_encoding(&t, "\r🙂", bytes, 5);
    const unsigned char expected[] = {13, 0xf0, 0x9f, 0x99, 0x82};
    for (int i = 0; i < 5; i++) {
        const unsigned char *piece = (const unsigned char *)bn_tokenizer_decode(&t, bytes[i]);
        assert(piece[0] == expected[i] && piece[1] == 0);
    }
    bn_tokenizer_free(&t); bn_gguf_free(f);

    /* Normalization can expand an unknown space to three fallback bytes. */
    const char *fallback[] = {"<0xE2>", "<0x96>", "<0x81>"};
    f = build_ranked_fixture(buf, "gemma4", NULL, fallback, 3, NULL, 0);
    assert(f && bn_tokenizer_init(&t, f) == 0);
    const int expanded[] = {0, 1, 2, 0, 1, 2};
    assert_encoding(&t, "  ", expanded, 6);
    bn_tokenizer_free(&t); bn_gguf_free(f);
    printf("PASSED\n");
}

static void test_tokenizer_glm4(void) {
    const char *vocab[] = {"a", "b", "c", "ab", "bc", "abc", "<S>", "<S>x", "<U>",
        "Ġ", "1", "2", "3", "4", "12", "123", "34", "1234", "<|observation|>"};
    const char *merges[] = {"a b", "ab c", "1 2", "12 3", "3 4", "123 4"};
    int32_t types[19];
    for (int i = 0; i < 19; i++) types[i] = 1;
    types[6] = 3; types[7] = 4; types[8] = 3; types[18] = 3;
    uint8_t buf[8192];
    BnGGUFFile *f = build_ranked_typed_fixture(buf, "gpt2", "glm4", vocab, 19, merges, 6, types);
    BnTokenizer t;
    assert(f && bn_tokenizer_init(&t, f) == 0);
    assert(t.pretokenizer == BN_TOKENIZER_PRE_GLM4 && t.n_added == 4);
    assert(bn_tokenizer_is_eog(&t, 18));
    const int a[] = {5,7,8,5}, b[] = {5,6,8,5}, c[] = {6,7}, d[] = {6,0};
    const int digits[] = {15,13}, spaced[] = {9,14};
    assert_encoding(&t, "abc<S>x<U>abc", a, 4);
    assert_encoding(&t, "abc<S><U>abc", b, 4);
    assert_encoding(&t, "<S><S>x", c, 2);
    assert_encoding(&t, "<S>a", d, 2);
    assert_encoding(&t, "1234", digits, 2);
    assert_encoding(&t, " 12", spaced, 2);
    bn_tokenizer_free(&t);
    int idx = bn_gguf_find_key(f, "tokenizer.ggml.token_type");
    f->kvs[idx].value.arr.n = 1;
    assert(bn_tokenizer_init(&t, f) == -1);
    f->kvs[idx].value.arr.n = 19;
    f->kvs[idx].value.arr.elem_type = BN_GGUF_TYPE_FLOAT32;
    assert(bn_tokenizer_init(&t, f) == -1);
    bn_gguf_free(f);
    assert(!bn_model_tokenizer_extra_eog(NULL, 0));
    assert(!bn_model_tokenizer_extra_eog("qwen2", 0));
    assert(!bn_model_tokenizer_extra_eog("glm4", -1));
    assert(!bn_model_tokenizer_extra_eog("glm4", 2));
    printf("test_tokenizer_glm4... PASSED\n");
}

#include "tokenizer_nfc_cases.h"

static void test_tokenizer_nfc(void) {
    char byte_vocab[256][3];
    const char *vocab[260];
    unsigned next = 256;
    for (unsigned byte = 0; byte < 256; byte++) {
        unsigned cp = (byte >= 33 && byte <= 126) ||
            (byte >= 161 && byte <= 172) || byte >= 174 ? byte : next++;
        if (cp < 128) { byte_vocab[byte][0] = (char)cp; byte_vocab[byte][1] = 0; }
        else {
            byte_vocab[byte][0] = (char)(0xc0 | (cp >> 6));
            byte_vocab[byte][1] = (char)(0x80 | (cp & 63));
            byte_vocab[byte][2] = 0;
        }
        vocab[byte] = byte_vocab[byte];
    }
    uint8_t buf[8192];
    BnGGUFFile *f = build_ranked_fixture(buf, "gpt2", "qwen35", vocab, 256, NULL, 0);
    BnTokenizer t;
    assert(f && bn_tokenizer_init(&t, f) == 0 && t.normalize_nfc);
    size_t cases = sizeof(tokenizer_nfc_cases) / sizeof(tokenizer_nfc_cases[0]);
    for (size_t i = 0; i < cases; i++) {
        for (int column = 0; column < 5; column++) {
            const char *expected = tokenizer_nfc_cases[i][column < 3 ? 1 : 3];
            int ids[256];
            size_t count = strlen(expected);
            assert(count < 256);
            ids[count] = -123;
            int n = bn_tokenizer_encode(&t, tokenizer_nfc_cases[i][column], 0, ids, (int)count + 1);
            assert(n == (int)count && ids[count] == -123);
            for (int j = 0; j < n; j++) assert(ids[j] == (unsigned char)expected[j]);
        }
    }
    const int acute[] = {195,161}, hangul[] = {234,176,129};
    const int expanded[] = {204,136,204,129}, malformed[] = {97,255,204,129};
    assert_encoding(&t, "a\xcc\x81", acute, 2);
    assert_encoding(&t, "\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8", hangul, 3);
    assert_encoding(&t, "\xcd\x84", expanded, 4);
    assert_encoding(&t, "a\xff\xcc\x81", malformed, 4);
    int with_bos[4] = {-1,-1,-1,-123};
    assert(bn_tokenizer_encode(&t, "a\xcc\x81", 1, with_bos, 3) == 3);
    assert(with_bos[0] == t.bos_id && with_bos[1] == 195 && with_bos[2] == 161 && with_bos[3] == -123);
    /* Independently checked against the pinned original NFC normalizer. */
    const int future_barrier[] = {97,240,158,139,172,204,167};
    assert_encoding(&t, "a\xf0\x9e\x8b\xac\xcc\xa7", future_barrier, 7);
    size_t repeats = 4096, length = 1 + repeats * 6;
    char *long_text = malloc(length + 1);
    int *long_ids = malloc((length + 1) * sizeof(int));
    assert(long_text && long_ids);
    long_text[0] = 'a';
    for (size_t i = 0; i < repeats; i++)
        memcpy(long_text + 1 + i * 6, "\xcc\x95\xcc\x80\xcc\xa7", 6);
    long_text[length] = 0;
    long_ids[length] = -123;
    int long_count = bn_tokenizer_encode(&t, long_text, 0, long_ids, (int)length);
    assert(long_count == (int)length - 1 && long_ids[length] == -123);
    assert(long_ids[0] == 195 && long_ids[1] == 160);
    size_t offset = 2;
    for (size_t group = 0; group < 3; group++) {
        size_t count = group == 1 ? repeats - 1 : repeats;
        int second = group == 0 ? 167 : group == 1 ? 128 : 149;
        for (size_t i = 0; i < count; i++) {
            assert(long_ids[offset++] == 204);
            assert(long_ids[offset++] == second);
        }
    }
    assert(offset == (size_t)long_count);
    BnTokenizerGGUFScratch retained={0};size_t retained_bytes;
    assert(!bn_tokenizer_gguf_scratch_plan(&t,length,&retained_bytes));
    assert(!bn_tokenizer_gguf_scratch_init(&retained,&t,length,retained_bytes));
    int *retained_ids=malloc((length+1)*sizeof(int));assert(retained_ids);
    for(int reuse=0;reuse<2;reuse++) {
        retained_ids[length]=-123;
        assert(bn_tokenizer_gguf_encode(&t,&retained,long_text,0,retained_ids,(int)length)==long_count);
        assert(retained_ids[length]==-123&&!memcmp(retained_ids,long_ids,(size_t)long_count*sizeof(int)));
    }
    bn_tokenizer_gguf_scratch_free(&retained);free(retained_ids);
    free(long_text); free(long_ids);
    bn_tokenizer_free(&t); bn_gguf_free(f);

    /* Added tokens marked normalized=false are extracted before NFC. NFC must
     * neither change them nor create a second round of added-token matches. */
    vocab[256] = "a\xcc\x81"; vocab[257] = "<K>";
    vocab[258] = "\xcc\x81"; vocab[259] = "\xc3\xa1";
    int32_t types[260];
    for (int i = 0; i < 260; i++) types[i] = i < 256 ? 1 : 3;
    f = build_ranked_typed_fixture(buf, "gpt2", "qwen35", vocab, 260, NULL, 0, types);
    assert(f && bn_tokenizer_init(&t, f) == 0);
    const int added[] = {256}, barrier[] = {101,258}, original_tag[] = {257};
    const int normalized_tag[] = {60,75,62};
    assert_encoding(&t, "a\xcc\x81", added, 1);
    assert_encoding(&t, "e\xcc\x81", barrier, 2);
    assert_encoding(&t, "<K>", original_tag, 1);
    assert_encoding(&t, "<\xe2\x84\xaa>", normalized_tag, 3);
    assert_encoding(&t, "a\xcd\x81", acute, 2);
    bn_tokenizer_free(&t); bn_gguf_free(f);
    assert(bn_model_tokenizer_normalizes_nfc("qwen35"));
    assert(!bn_model_tokenizer_normalizes_nfc("qwen2"));
    assert(!bn_model_tokenizer_normalizes_nfc("glm4"));
    assert(!bn_model_tokenizer_normalizes_nfc(NULL));
    printf("test_tokenizer_nfc... PASSED (%zu independent Unicode checks)\n", cases * 5);
}

#include "native_vocab_checks.h"
#include "native_encode_checks.h"
#include "native_stream_checks.h"
#include "native_tokenmap_checks.h"
static void test_gguf_scratch(void) {
    uint8_t data[8192];size_t n=build_tokenizer_gguf(data,sizeof(data));BnGGUFFile *f=bn_gguf_open(data,n);assert(f);
    BnTokenizer t;assert(!bn_tokenizer_init(&t,f));BnTokenizerGGUFScratch scratch={0};size_t bytes=0;
    assert(!bn_tokenizer_gguf_scratch_plan(&t,10,&bytes)&&bytes>80);
    assert(bn_tokenizer_gguf_scratch_init(&scratch,&t,10,bytes-1)==-1&&!scratch.state);
    assert(!bn_tokenizer_gguf_scratch_init(&scratch,&t,10,bytes));void *state=scratch.state;
    assert(bn_tokenizer_gguf_scratch_init(&scratch,&t,10,bytes)==-1&&scratch.state==state);
    int out[4]={99,98,97,96};
    assert(bn_tokenizer_gguf_encode(&t,&scratch,"hellohello",0,out,4)==2&&out[0]==9&&out[1]==9&&out[2]==97);
    out[0]=99;out[1]=98;
    assert(bn_tokenizer_gguf_encode(&t,&scratch,"hello",1,out,1)==-1&&out[0]==99&&out[1]==98);
    assert(bn_tokenizer_gguf_encode(&t,&scratch,"hello",1,out,4)==2&&out[0]==0&&out[1]==9);
    out[0]=99;assert(bn_tokenizer_gguf_encode(&t,&scratch,"hellohellohello",0,out,4)==-1&&out[0]==99);
    BnTokenizerGGUFScratch copy=scratch;assert(bn_tokenizer_gguf_encode(&t,&copy,"hello",0,out,4)==-1);
    bn_tokenizer_gguf_scratch_free(&copy);assert(scratch.state==state);
    BnTokenizer other=t;assert(bn_tokenizer_gguf_encode(&other,&scratch,"hello",0,out,4)==-1);
    assert(!bn_tokenizer_gguf_encode(&t,&scratch,"",0,out,4)&&out[0]==99);
    assert(bn_tokenizer_gguf_encode(&t,&scratch,"",1,out,4)==1&&out[0]==0);
    bn_tokenizer_gguf_scratch_free(&scratch);bn_tokenizer_gguf_scratch_free(&scratch);
    size_t before=7;assert(bn_tokenizer_gguf_scratch_plan(&t,SIZE_MAX,&before)==-1&&before==7);
    t.metaspace=1;
    assert(bn_tokenizer_gguf_scratch_plan(&t,10,&before)==-1&&before==7);t.metaspace=0;t.normalize_nfc=1;
    assert(!bn_tokenizer_gguf_scratch_plan(&t,10,&before)&&before>bytes);
    assert(!bn_tokenizer_gguf_scratch_init(&scratch,&t,10,before));
    assert(bn_tokenizer_gguf_encode(&t,&scratch,"hello",0,out,4)==1&&out[0]==9);
    bn_tokenizer_gguf_scratch_free(&scratch);t.normalize_nfc=0;
    bn_tokenizer_free(&t);bn_gguf_free(f);
    puts("GGUF scratch: exact BPE reuse, budget preflight, immutable ownership and whole-output capacity rejection PASSED");
}
int main(void) {
    test_gguf_scratch();
    test_native_stream();
    if(getenv("BN_TEST_NATIVE_STREAM_ONLY"))return 0;
    test_native_encode();
    if(getenv("BN_TEST_NATIVE_ENCODE_ONLY"))return 0;
    test_native_tokenmap();
    if(getenv("BN_TEST_NATIVE_TOKENMAP_ONLY"))return 0;
    test_native_vocab();
    if(getenv("BN_TEST_NATIVE_VOCAB_ONLY"))return 0;
    printf("=== Tokenizer Tests ===\n");
    test_tokenizer_nfc();
    test_tokenizer_glm4();
    test_tokenizer_raw_bpe();
    test_tokenizer_ranked_bpe();
    test_tokenizer_qwen_boundaries();
    test_tokenizer_init();
    test_tokenizer_decode();
    test_tokenizer_encode();
    test_tokenizer_encode_empty();
    test_tokenizer_encode_max_tokens();
    test_tokenizer_decode_oob();
    test_tokenizer_scores_wrong_type();
    test_tokenizer_gemma4_metaspace();
    printf("All tokenizer tests passed!\n");
    return 0;
}
