#include "platform.h"
#include "gguf.h"
#include "quant.h"
#include "sampler.h"
#include "tokenizer.h"
#include "model.h"
#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// --- GGUF write helpers (shared with test_gguf.c) ---

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
static void wb_str(WriteBuffer *wb, const char *s) {
    uint64_t len = strlen(s);
    wb_u64(wb, len);
    wb_write(wb, s, len);
}

// ===================================================================
// Test: Truncated GGUF file should return NULL (#12)
// ===================================================================
static void test_gguf_truncated(void) {
    printf("test_gguf_truncated... ");

    // Valid magic but truncated header (only 8 bytes, need 24)
    uint8_t buf[8];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_u32(&wb, 0x46554747);  // magic
    wb_u32(&wb, 3);           // version

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f == NULL);  // should fail: truncated

    printf("PASSED\n");
}

// ===================================================================
// Test: GGUF with n_dims > 4 should fail (#11)
// ===================================================================
static void test_gguf_bad_ndims(void) {
    printf("test_gguf_bad_ndims... ");

    uint8_t buf[4096];
    WriteBuffer wb = { buf, 0, sizeof(buf) };

    // Header
    wb_u32(&wb, 0x46554747);  // magic
    wb_u32(&wb, 3);           // version
    wb_u64(&wb, 1);           // n_tensors
    wb_u64(&wb, 0);           // n_kv

    // Tensor with n_dims = 5 (invalid, max is 4)
    wb_str(&wb, "bad_tensor");
    wb_u32(&wb, 5);           // n_dims = 5 → should be rejected
    for (int d = 0; d < 5; d++) wb_u64(&wb, 256);
    wb_u32(&wb, 0);           // type
    wb_u64(&wb, 0);           // offset

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f == NULL);  // should fail: n_dims > 4

    printf("PASSED\n");
}

// ===================================================================
// Test: bn_gguf_get_u32 with wrong type returns 0 (#21)
// ===================================================================
static void test_gguf_type_mismatch(void) {
    printf("test_gguf_type_mismatch... ");

    uint8_t buf[4096];
    WriteBuffer wb = { buf, 0, sizeof(buf) };

    // Header
    wb_u32(&wb, 0x46554747);
    wb_u32(&wb, 3);
    wb_u64(&wb, 0);  // n_tensors
    wb_u64(&wb, 1);  // n_kv

    // KV: string value, but we'll try to read as u32
    wb_str(&wb, "test.key");
    wb_u32(&wb, 8);  // type = STRING
    wb_str(&wb, "hello");

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f != NULL);

    // Key exists but is STRING, not UINT32 → should return 0
    uint32_t val = bn_gguf_get_u32(f, "test.key");
    assert(val == 0);

    // bn_gguf_get_str should work
    const char *str = bn_gguf_get_str(f, "test.key");
    assert(str != NULL);
    assert(strcmp(str, "hello") == 0);

    bn_gguf_free(f);
    printf("PASSED\n");
}

// ===================================================================
// Test: bn_gguf_tensor_data with bad offset returns NULL (#13)
// ===================================================================
static void test_gguf_tensor_data_oob(void) {
    printf("test_gguf_tensor_data_oob... ");

    uint8_t buf[4096];
    WriteBuffer wb = { buf, 0, sizeof(buf) };

    // Header
    wb_u32(&wb, 0x46554747);
    wb_u32(&wb, 3);
    wb_u64(&wb, 1);  // n_tensors
    wb_u64(&wb, 0);  // n_kv

    // Tensor with huge offset
    wb_str(&wb, "test.tensor");
    wb_u32(&wb, 1);           // n_dims
    wb_u64(&wb, 256);         // dim[0]
    wb_u32(&wb, 0);           // type = F32
    wb_u64(&wb, 999999999);   // offset way beyond buffer

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f != NULL);

    int ti = bn_gguf_find_tensor(f, "test.tensor");
    assert(ti >= 0);

    void *data = bn_gguf_tensor_data(f, ti);
    assert(data == NULL);  // should be NULL: offset beyond buffer

    // Negative index should also return NULL
    assert(bn_gguf_tensor_data(f, -1) == NULL);

    bn_gguf_free(f);
    printf("PASSED\n");
}

// ===================================================================
// Test: bn_gguf_get_arr_str with negative idx returns NULL (#34)
// ===================================================================
static void test_gguf_arr_str_negative_idx(void) {
    printf("test_gguf_arr_str_negative_idx... ");

    uint8_t buf[4096];
    WriteBuffer wb = { buf, 0, sizeof(buf) };

    // Header
    wb_u32(&wb, 0x46554747);
    wb_u32(&wb, 3);
    wb_u64(&wb, 0);  // n_tensors
    wb_u64(&wb, 1);  // n_kv

    // KV: string array with 2 elements
    wb_str(&wb, "test.arr");
    wb_u32(&wb, 9);  // type = ARRAY
    wb_u32(&wb, 8);  // elem_type = STRING
    wb_u64(&wb, 2);  // n = 2
    wb_str(&wb, "hello");
    wb_str(&wb, "world");

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f != NULL);

    // Valid indices
    assert(bn_gguf_get_arr_str(f, "test.arr", 0) != NULL);
    assert(bn_gguf_get_arr_str(f, "test.arr", 1) != NULL);

    // Negative index → NULL
    assert(bn_gguf_get_arr_str(f, "test.arr", -1) == NULL);

    // Out of range → NULL
    assert(bn_gguf_get_arr_str(f, "test.arr", 2) == NULL);

    bn_gguf_free(f);
    printf("PASSED\n");
}

// ===================================================================
// Test: sampler edge cases (#28, #29)
// ===================================================================
static void test_sampler_edge_cases(void) {
    printf("test_sampler_edge_cases... ");

    // Test sampler with temp=0 (argmax)
    BnSampler s;
    bn_sampler_init(&s, 4, 0.0f, 0.9f, 42);

    float logits[] = {1.0f, 3.0f, 2.0f, 0.5f};
    int tok = bn_sampler_sample(&s, logits);
    assert(tok == 1);  // argmax should pick index 1
    bn_sampler_free(&s);

    // Test with vocab_size=1
    bn_sampler_init(&s, 1, 0.5f, 0.9f, 42);
    float logits1[] = {5.0f};
    tok = bn_sampler_sample(&s, logits1);
    assert(tok == 0);
    bn_sampler_free(&s);

    printf("PASSED\n");
}

// ===================================================================
// Test: sampler repeat penalty + top-p
// ===================================================================
static void test_sampler_repeat_penalty(void) {
    printf("test_sampler_repeat_penalty... ");

    BnSampler s;
    bn_sampler_init(&s, 4, 0.0f, 0.9f, 42);
    bn_sampler_set_repeat_penalty(&s, 1.5f, 4);

    // Accept token 1 multiple times
    bn_sampler_accept(&s, 1);
    bn_sampler_accept(&s, 1);
    bn_sampler_accept(&s, 1);

    // Token 1 has highest logit but should be penalized
    float logits[] = {2.0f, 2.1f, 2.0f, 2.0f};
    int tok = bn_sampler_sample(&s, logits);
    // With penalty 1.5, logit[1]=2.1 becomes 2.1/1.5=1.4, so token 0 or 2 should win
    assert(tok != 1);

    // Reset and verify penalty is cleared
    bn_sampler_reset_recent(&s);
    float logits2[] = {1.0f, 5.0f, 1.0f, 1.0f};
    tok = bn_sampler_sample(&s, logits2);
    assert(tok == 1);  // no penalty, token 1 should win

    bn_sampler_free(&s);
    printf("PASSED\n");
}

static void test_sampler_topp(void) {
    printf("test_sampler_topp... ");

    BnSampler s;
    bn_sampler_init(&s, 4, 1.0f, 0.5f, 42);

    // Run multiple samples to verify top-p constrains to high-prob tokens
    int counts[4] = {0};
    for (int trial = 0; trial < 100; trial++) {
        float logits[] = {10.0f, 0.0f, 0.0f, 0.0f};  // token 0 dominates
        int tok = bn_sampler_sample(&s, logits);
        assert(tok >= 0 && tok < 4);
        counts[tok]++;
    }
    // Token 0 should get the vast majority with topp=0.5
    assert(counts[0] > 80);

    bn_sampler_free(&s);
    printf("PASSED\n");
}

// ===================================================================
// Test: bn_model_embed_token with out-of-range token (#8)
// ===================================================================
static void test_embed_token_oob(void) {
    printf("test_embed_token_oob... ");

    // Create a minimal model with fake embedding
    BnModel m;
    memset(&m, 0, sizeof(m));
    m.config.dim = 4;
    m.config.vocab_size = 3;
    m.weights.emb_type = 0;  // F32

    float fake_emb[12] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    m.weights.token_embedding = fake_emb;

    float out[4];

    // Valid token
    bn_model_embed_token(&m, out, 0);
    assert(out[0] == 1.0f);

    // Negative token → should zero out
    bn_model_embed_token(&m, out, -1);
    assert(out[0] == 0.0f && out[1] == 0.0f);

    // Token beyond vocab → should zero out
    bn_model_embed_token(&m, out, 999);
    assert(out[0] == 0.0f && out[1] == 0.0f);

    printf("PASSED\n");
}

// ===================================================================
// Test: bn_transformer_forward with bad token/pos returns NULL (#9, #10)
// ===================================================================
static void test_transformer_bounds(void) {
    printf("test_transformer_bounds... ");

    // Create a minimal model (just enough for bounds checks)
    BnModel m;
    memset(&m, 0, sizeof(m));
    m.config.vocab_size = 100;
    m.config.seq_len = 512;
    m.config.dim = 64;
    m.config.n_layers = 0;  // skip layer processing

    // bn_transformer_forward should reject invalid token
    float *logits = bn_transformer_forward(&m, -1, 0);
    assert(logits == NULL);

    logits = bn_transformer_forward(&m, 100, 0);  // == vocab_size
    assert(logits == NULL);

    // bn_transformer_forward should reject invalid pos
    logits = bn_transformer_forward(&m, 0, -1);
    assert(logits == NULL);

    printf("PASSED\n");
}

// ===================================================================
// Test: bn_quant_dequant_tq1 produces exactly BN_QK_K values (#35)
// ===================================================================
static void test_dequant_tq1_count(void) {
    printf("test_dequant_tq1_count... ");

    BnBlockTQ1 block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3C00;  // scale = 1.0

    // Fill with known pattern
    for (int i = 0; i < 48; i++) block.qs[i] = 0;
    for (int i = 0; i < 4; i++) block.qh[i] = 0;

    float out[BN_QK_K];
    memset(out, 0x7F, sizeof(out));  // fill with garbage

    // Should produce exactly 256 values (assertion inside bn_quant_dequant_tq1)
    bn_quant_dequant_tq1(&block, out);

    // All values should be -1.0 (all-zero packed = trit value 0 = mapped to -1)
    for (int i = 0; i < BN_QK_K; i++) {
        assert(fabsf(out[i] - (-1.0f)) < 1e-5f);
    }

    printf("PASSED\n");
}

// ===================================================================
// Test: I2_S dequantization correctness (#36)
// ===================================================================
static void test_i2s_dequant(void) {
    printf("test_i2s_dequant... ");

    // Test with n=128 (one full chunk)
    // 0xAA = 10_10_10_10 → all sub-rows map to +1
    uint8_t data[32];
    memset(data, 0xAA, sizeof(data));

    float out[128];
    memset(out, 0, sizeof(out));

    bn_quant_dequant_i2s(data, out, 128, 1.0f);

    // All 128 values should be +1.0
    for (int i = 0; i < 128; i++) {
        assert(fabsf(out[i] - 1.0f) < 1e-5f);
    }

    // Test with n=256 (two chunks)
    uint8_t data2[64];
    memset(data2, 0x55, sizeof(data2));  // 0x55 = 01_01_01_01 → all 0

    float out2[256];
    bn_quant_dequant_i2s(data2, out2, 256, 1.0f);

    for (int i = 0; i < 256; i++) {
        assert(fabsf(out2[i] - 0.0f) < 1e-5f);
    }

    printf("PASSED\n");
}

// ===================================================================
// Test: bn_platform_load_buffer with is_mmap=2 (#19)
// ===================================================================
static void test_platform_load_buffer(void) {
    printf("test_platform_load_buffer... ");

    uint8_t stack_buf[32] = {1, 2, 3};
    BnMappedFile mf = bn_platform_load_buffer(stack_buf, sizeof(stack_buf));

    assert(mf.data == stack_buf);
    assert(mf.size == sizeof(stack_buf));
    assert(mf.is_mmap == 2);  // externally owned

    // Unloading should NOT free the stack buffer
    bn_platform_unload_file(&mf);
    assert(mf.data == NULL);

    // Original buffer should still be valid (stack_buf[0] == 1)
    assert(stack_buf[0] == 1);

    printf("PASSED\n");
}

int main(void) {
    printf("=== Safety Regression Tests ===\n");

    // GGUF safety
    test_gguf_truncated();
    test_gguf_bad_ndims();
    test_gguf_type_mismatch();
    test_gguf_tensor_data_oob();
    test_gguf_arr_str_negative_idx();

    // Sampler safety
    test_sampler_edge_cases();
    test_sampler_repeat_penalty();
    test_sampler_topp();

    // Model safety
    test_embed_token_oob();

    // Transformer safety
    test_transformer_bounds();

    // Quant safety
    test_dequant_tq1_count();
    test_i2s_dequant();

    // Platform safety
    test_platform_load_buffer();

    printf("All safety regression tests passed!\n");
    return 0;
}
