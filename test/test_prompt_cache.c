#include "prompt_cache.h"
#include "model.h"
#include "session.h"
#include "turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Minimal model config for prompt cache tests
static void init_test_config(BnConfig *c) {
    memset(c, 0, sizeof(*c));
    c->dim = 64;
    c->hidden_dim = 128;
    c->n_layers = 4;
    c->n_heads = 4;
    c->n_kv_heads = 4;
    c->vocab_size = 100;
    c->seq_len = 64;
    c->head_size = 16;
    c->kv_dim = 64;
    c->kv_mul = 1;
    c->rope_theta = 10000.0f;
    c->norm_eps = 1e-5f;
}

// Write a known pattern into session KV cache at given positions
static void write_kv_pattern(BnSession *s, const BnConfig *c, int pos_start, int n_pos, float base) {
    int n_attn = c->n_layers;  // all attention for non-hybrid
    int kv_dim = c->kv_dim;
    for (int a = 0; a < n_attn; a++) {
        for (int p = pos_start; p < pos_start + n_pos; p++) {
            float *k = s->state.key_cache + (size_t)a * c->seq_len * kv_dim + (size_t)p * kv_dim;
            float *v = s->state.value_cache + (size_t)a * c->seq_len * kv_dim + (size_t)p * kv_dim;
            for (int d = 0; d < kv_dim; d++) {
                k[d] = base + a * 1000.0f + p * 10.0f + d * 0.01f;
                v[d] = base + a * 1000.0f + p * 10.0f + d * 0.01f + 0.5f;
            }
        }
    }
}

// Verify KV pattern at given positions
static int verify_kv_pattern(const BnSession *s, const BnConfig *c, int pos_start, int n_pos, float base) {
    int n_attn = c->n_layers;
    int kv_dim = c->kv_dim;
    for (int a = 0; a < n_attn; a++) {
        for (int p = pos_start; p < pos_start + n_pos; p++) {
            const float *k = s->state.key_cache + (size_t)a * c->seq_len * kv_dim + (size_t)p * kv_dim;
            const float *v = s->state.value_cache + (size_t)a * c->seq_len * kv_dim + (size_t)p * kv_dim;
            for (int d = 0; d < kv_dim; d++) {
                float expect_k = base + a * 1000.0f + p * 10.0f + d * 0.01f;
                float expect_v = base + a * 1000.0f + p * 10.0f + d * 0.01f + 0.5f;
                if (k[d] != expect_k || v[d] != expect_v) return 0;
            }
        }
    }
    return 1;
}

// ===================================================================
// Test: create and free cache
// ===================================================================
static void test_create_free(void) {
    printf("test_create_free... ");
    BnPromptCache *pc = bn_prompt_cache_create(0, NULL);
    assert(pc != NULL);
    assert(bn_prompt_cache_count(pc) == 0);
    assert(bn_prompt_cache_bytes(pc) == 0);
    bn_prompt_cache_free(pc);
    printf("PASSED\n");
}

// ===================================================================
// Test: store and restore roundtrip
// ===================================================================
static void test_store_restore(void) {
    printf("test_store_restore... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    BnSession *s1 = bn_session_create(&model, NULL);
    assert(s1);

    // Write pattern and set pos
    int tokens[] = {10, 20, 30, 40, 50};
    int n = 5;
    write_kv_pattern(s1, &model.config, 0, n, 1.0f);
    s1->pos = n;

    // Store into cache
    BnPromptCache *pc = bn_prompt_cache_create(0, NULL);
    int rc = bn_prompt_cache_store(pc, &model, s1, tokens, n);
    assert(rc == 0);
    assert(bn_prompt_cache_count(pc) == 1);

    // Restore into a fresh session
    BnSession *s2 = bn_session_create(&model, NULL);
    assert(s2);
    int restored = bn_prompt_cache_restore(pc, &model, s2, tokens, n);
    assert(restored == n);
    assert(s2->pos == n);

    // Verify KV data matches
    assert(verify_kv_pattern(s2, &model.config, 0, n, 1.0f));

    bn_session_free(s1, NULL);
    bn_session_free(s2, NULL);
    bn_prompt_cache_free(pc);
    printf("PASSED\n");
}

// ===================================================================
// Test: longest prefix match
// ===================================================================
static void test_prefix_match(void) {
    printf("test_prefix_match... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    BnSession *s1 = bn_session_create(&model, NULL);
    assert(s1);

    // Store [A,B,C,D] with 4 tokens of KV data
    int tokens_full[] = {10, 20, 30, 40};
    write_kv_pattern(s1, &model.config, 0, 4, 2.0f);
    s1->pos = 4;
    BnPromptCache *pc = bn_prompt_cache_create(0, NULL);
    assert(bn_prompt_cache_store(pc, &model, s1, tokens_full, 4) == 0);

    // Query [A,B,C,E] — should match first 3 tokens
    BnSession *s2 = bn_session_create(&model, NULL);
    int tokens_query[] = {10, 20, 30, 99};
    int restored = bn_prompt_cache_restore(pc, &model, s2, tokens_query, 4);
    assert(restored == 3);
    assert(s2->pos == 3);

    // Verify only the first 3 positions were copied
    assert(verify_kv_pattern(s2, &model.config, 0, 3, 2.0f));

    bn_session_free(s1, NULL);
    bn_session_free(s2, NULL);
    bn_prompt_cache_free(pc);
    printf("PASSED\n");
}

// ===================================================================
// Test: cache miss
// ===================================================================
static void test_cache_miss(void) {
    printf("test_cache_miss... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    BnSession *s1 = bn_session_create(&model, NULL);
    assert(s1);

    int tokens_stored[] = {10, 20, 30};
    write_kv_pattern(s1, &model.config, 0, 3, 1.0f);
    s1->pos = 3;
    BnPromptCache *pc = bn_prompt_cache_create(0, NULL);
    assert(bn_prompt_cache_store(pc, &model, s1, tokens_stored, 3) == 0);

    // Query completely different tokens
    BnSession *s2 = bn_session_create(&model, NULL);
    int tokens_query[] = {99, 98, 97};
    int restored = bn_prompt_cache_restore(pc, &model, s2, tokens_query, 3);
    assert(restored == 0);
    assert(s2->pos == 0);

    bn_session_free(s1, NULL);
    bn_session_free(s2, NULL);
    bn_prompt_cache_free(pc);
    printf("PASSED\n");
}

// ===================================================================
// Test: FP16 KV cache
// ===================================================================
static void test_fp16_roundtrip(void) {
    printf("test_fp16_roundtrip... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);
    model.config.kv_f16 = 1;

    BnSession *s1 = bn_session_create(&model, NULL);
    assert(s1);

    // Write raw bytes into KV cache (FP16 mode uses uint16_t layout)
    int n = 3;
    int kv_dim = model.config.kv_dim;
    int n_attn = model.config.n_layers;
    for (int a = 0; a < n_attn; a++) {
        uint16_t *k = (uint16_t *)s1->state.key_cache + (size_t)a * model.config.seq_len * kv_dim;
        uint16_t *v = (uint16_t *)s1->state.value_cache + (size_t)a * model.config.seq_len * kv_dim;
        for (int p = 0; p < n; p++) {
            for (int d = 0; d < kv_dim; d++) {
                k[p * kv_dim + d] = (uint16_t)(a * 1000 + p * 10 + d);
                v[p * kv_dim + d] = (uint16_t)(a * 1000 + p * 10 + d + 5000);
            }
        }
    }
    s1->pos = n;

    int tokens[] = {1, 2, 3};
    BnPromptCache *pc = bn_prompt_cache_create(0, NULL);
    assert(bn_prompt_cache_store(pc, &model, s1, tokens, n) == 0);

    // Restore and verify
    BnSession *s2 = bn_session_create(&model, NULL);
    int restored = bn_prompt_cache_restore(pc, &model, s2, tokens, n);
    assert(restored == n);

    for (int a = 0; a < n_attn; a++) {
        uint16_t *k = (uint16_t *)s2->state.key_cache + (size_t)a * model.config.seq_len * kv_dim;
        uint16_t *v = (uint16_t *)s2->state.value_cache + (size_t)a * model.config.seq_len * kv_dim;
        for (int p = 0; p < n; p++) {
            for (int d = 0; d < kv_dim; d++) {
                assert(k[p * kv_dim + d] == (uint16_t)(a * 1000 + p * 10 + d));
                assert(v[p * kv_dim + d] == (uint16_t)(a * 1000 + p * 10 + d + 5000));
            }
        }
    }

    bn_session_free(s1, NULL);
    bn_session_free(s2, NULL);
    bn_prompt_cache_free(pc);
    printf("PASSED\n");
}

// ===================================================================
// Test: eviction when max_bytes exceeded
// ===================================================================
static void test_eviction(void) {
    printf("test_eviction... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    // Set a tiny budget: one entry's KV is n_attn*n_tokens*kv_dim*4 bytes * 2 (key+value)
    // For 2 tokens: 4 * 2 * 64 * 4 * 2 = 4096 bytes + token bytes
    // Set budget to fit ~1.5 entries so second store evicts the first
    size_t budget = 5000;
    BnPromptCache *pc = bn_prompt_cache_create(budget, NULL);

    BnSession *s = bn_session_create(&model, NULL);
    assert(s);

    // Store entry 1
    int tok1[] = {1, 2};
    write_kv_pattern(s, &model.config, 0, 2, 1.0f);
    s->pos = 2;
    assert(bn_prompt_cache_store(pc, &model, s, tok1, 2) == 0);
    assert(bn_prompt_cache_count(pc) == 1);

    // Store entry 2 — should evict entry 1
    int tok2[] = {3, 4};
    write_kv_pattern(s, &model.config, 0, 2, 2.0f);
    s->pos = 2;
    assert(bn_prompt_cache_store(pc, &model, s, tok2, 2) == 0);
    assert(bn_prompt_cache_count(pc) == 1);  // only entry 2 remains

    // Entry 1 should miss
    BnSession *s2 = bn_session_create(&model, NULL);
    assert(bn_prompt_cache_restore(pc, &model, s2, tok1, 2) == 0);

    // Entry 2 should hit
    int restored = bn_prompt_cache_restore(pc, &model, s2, tok2, 2);
    assert(restored == 2);
    assert(verify_kv_pattern(s2, &model.config, 0, 2, 2.0f));

    bn_session_free(s, NULL);
    bn_session_free(s2, NULL);
    bn_prompt_cache_free(pc);
    printf("PASSED\n");
}

// ===================================================================
// Test: hybrid model rejection
// ===================================================================
static void test_hybrid_reject(void) {
    printf("test_hybrid_reject... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);
    model.config.full_attn_interval = 4;  // hybrid SSM+Attention

    BnSession *s = bn_session_create(&model, NULL);
    assert(s);
    s->pos = 3;

    int tokens[] = {1, 2, 3};
    BnPromptCache *pc = bn_prompt_cache_create(0, NULL);

    // Store should fail with -1
    assert(bn_prompt_cache_store(pc, &model, s, tokens, 3) == -1);
    assert(bn_prompt_cache_count(pc) == 0);

    // Restore should return 0
    assert(bn_prompt_cache_restore(pc, &model, s, tokens, 3) == 0);

    bn_session_free(s, NULL);
    bn_prompt_cache_free(pc);
    printf("PASSED\n");
}

// ===================================================================
// Test: clear
// ===================================================================
static void test_clear(void) {
    printf("test_clear... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    BnSession *s = bn_session_create(&model, NULL);
    assert(s);

    BnPromptCache *pc = bn_prompt_cache_create(0, NULL);

    int tok1[] = {1, 2};
    write_kv_pattern(s, &model.config, 0, 2, 1.0f);
    s->pos = 2;
    assert(bn_prompt_cache_store(pc, &model, s, tok1, 2) == 0);

    int tok2[] = {3, 4, 5};
    write_kv_pattern(s, &model.config, 0, 3, 2.0f);
    s->pos = 3;
    assert(bn_prompt_cache_store(pc, &model, s, tok2, 3) == 0);
    assert(bn_prompt_cache_count(pc) == 2);

    bn_prompt_cache_clear(pc);
    assert(bn_prompt_cache_count(pc) == 0);
    assert(bn_prompt_cache_bytes(pc) == 0);

    bn_session_free(s, NULL);
    bn_prompt_cache_free(pc);
    printf("PASSED\n");
}

// ===================================================================
// Test: multiple entries, best prefix wins
// ===================================================================
static void test_best_prefix(void) {
    printf("test_best_prefix... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    BnPromptCache *pc = bn_prompt_cache_create(0, NULL);
    BnSession *s = bn_session_create(&model, NULL);
    assert(s);

    // Store short prefix [1,2]
    int tok_short[] = {1, 2};
    write_kv_pattern(s, &model.config, 0, 2, 1.0f);
    s->pos = 2;
    assert(bn_prompt_cache_store(pc, &model, s, tok_short, 2) == 0);

    // Store longer prefix [1,2,3,4]
    int tok_long[] = {1, 2, 3, 4};
    write_kv_pattern(s, &model.config, 0, 4, 2.0f);
    s->pos = 4;
    assert(bn_prompt_cache_store(pc, &model, s, tok_long, 4) == 0);
    assert(bn_prompt_cache_count(pc) == 2);

    // Query [1,2,3,4,5] — should match the longer entry (4 tokens)
    BnSession *s2 = bn_session_create(&model, NULL);
    int tok_query[] = {1, 2, 3, 4, 5};
    int restored = bn_prompt_cache_restore(pc, &model, s2, tok_query, 5);
    assert(restored == 4);
    assert(s2->pos == 4);
    assert(verify_kv_pattern(s2, &model.config, 0, 4, 2.0f));

    bn_session_free(s, NULL);
    bn_session_free(s2, NULL);
    bn_prompt_cache_free(pc);
    printf("PASSED\n");
}

// ===================================================================
// Test: custom allocator
// ===================================================================
static void test_custom_alloc(void) {
    printf("test_custom_alloc... ");

    BnAllocator alloc = bn_allocator_default();
    BnPromptCache *pc = bn_prompt_cache_create(0, &alloc);
    assert(pc != NULL);
    assert(bn_prompt_cache_count(pc) == 0);

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    BnSession *s = bn_session_create(&model, NULL);
    assert(s);
    int tokens[] = {10, 20};
    write_kv_pattern(s, &model.config, 0, 2, 3.0f);
    s->pos = 2;
    assert(bn_prompt_cache_store(pc, &model, s, tokens, 2) == 0);
    assert(bn_prompt_cache_count(pc) == 1);

    bn_session_free(s, NULL);
    bn_prompt_cache_free(pc);
    printf("PASSED\n");
}

// ===================================================================
// Test: TurboQuant KV cache roundtrip
// ===================================================================
static void test_tq_roundtrip(void) {
    printf("test_tq_roundtrip... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);
    model.config.kv_tq_bits = 3;
    model.config.head_size = 16;  // small for test
    model.config.n_kv_heads = 4;

    // Initialize TQ state on the model
    BnTQState tq;
    assert(bn_tq_init(&tq, model.config.head_size, 3, 0x5451303042ULL) == 0);
    bn_model_set_tq_state(&model, &tq, 0);

    BnSession *s1 = bn_session_create(&model, NULL);
    assert(s1);
    assert(s1->state.key_cache_tq != NULL);
    assert(s1->state.value_cache_tq != NULL);

    int n = 3;
    int n_attn = model.config.n_layers;
    int n_kv_heads = model.config.n_kv_heads;
    int kb = bn_tq_key_bytes(&tq);
    int vb = bn_tq_value_bytes(&tq);

    // Write a known pattern into TQ packed caches
    for (int a = 0; a < n_attn; a++) {
        for (int p = 0; p < n; p++) {
            for (int h = 0; h < n_kv_heads; h++) {
                size_t k_off = (size_t)a * model.config.seq_len * n_kv_heads * kb
                             + (size_t)p * n_kv_heads * kb + (size_t)h * kb;
                size_t v_off = (size_t)a * model.config.seq_len * n_kv_heads * vb
                             + (size_t)p * n_kv_heads * vb + (size_t)h * vb;
                for (int b = 0; b < kb; b++)
                    s1->state.key_cache_tq[k_off + b] = (uint8_t)((a + p + h + b) & 0xFF);
                for (int b = 0; b < vb; b++)
                    s1->state.value_cache_tq[v_off + b] = (uint8_t)((a + p + h + b + 128) & 0xFF);
            }
        }
    }
    s1->pos = n;

    int tokens[] = {10, 20, 30};
    BnPromptCache *pc = bn_prompt_cache_create(0, NULL);
    int rc = bn_prompt_cache_store(pc, &model, s1, tokens, n);
    assert(rc == 0);
    assert(bn_prompt_cache_count(pc) == 1);

    // Verify entry has TQ format
    // (used_bytes should reflect TQ sizes, not FP32 sizes)
    size_t expected_key = (size_t)n_attn * n * n_kv_heads * kb;
    size_t expected_val = (size_t)n_attn * n * n_kv_heads * vb;
    size_t expected_total = expected_key + expected_val + (size_t)n * sizeof(int);
    assert(bn_prompt_cache_bytes(pc) == expected_total);

    // Restore into fresh session
    BnSession *s2 = bn_session_create(&model, NULL);
    assert(s2);
    int restored = bn_prompt_cache_restore(pc, &model, s2, tokens, n);
    assert(restored == n);
    assert(s2->pos == n);

    // Verify TQ packed data matches byte-for-byte
    for (int a = 0; a < n_attn; a++) {
        for (int p = 0; p < n; p++) {
            for (int h = 0; h < n_kv_heads; h++) {
                size_t k_off = (size_t)a * model.config.seq_len * n_kv_heads * kb
                             + (size_t)p * n_kv_heads * kb + (size_t)h * kb;
                size_t v_off = (size_t)a * model.config.seq_len * n_kv_heads * vb
                             + (size_t)p * n_kv_heads * vb + (size_t)h * vb;
                for (int b = 0; b < kb; b++)
                    assert(s2->state.key_cache_tq[k_off + b] == (uint8_t)((a + p + h + b) & 0xFF));
                for (int b = 0; b < vb; b++)
                    assert(s2->state.value_cache_tq[v_off + b] == (uint8_t)((a + p + h + b + 128) & 0xFF));
            }
        }
    }

    // Verify FP32/FP16 entry doesn't match TQ entry
    BnModel model_fp32;
    memset(&model_fp32, 0, sizeof(model_fp32));
    init_test_config(&model_fp32.config);
    model_fp32.config.kv_tq_bits = 0;  // no TQ

    BnSession *s3 = bn_session_create(&model_fp32, NULL);
    assert(s3);
    int no_match = bn_prompt_cache_restore(pc, &model_fp32, s3, tokens, n);
    assert(no_match == 0);  // format mismatch: TQ entry vs FP32 query

    bn_session_free(s1, NULL);
    bn_session_free(s2, NULL);
    bn_session_free(s3, NULL);
    bn_prompt_cache_free(pc);
    bn_tq_free(&tq);
    printf("PASSED\n");
}

int main(void) {
    test_create_free();
    test_store_restore();
    test_prefix_match();
    test_cache_miss();
    test_fp16_roundtrip();
    test_eviction();
    test_hybrid_reject();
    test_clear();
    test_best_prefix();
    test_custom_alloc();
    test_tq_roundtrip();
    printf("\nAll prompt cache tests passed!\n");
    return 0;
}
