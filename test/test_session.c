#include "session.h"
#include "model.h"
#include "backend_session.h"
#include "gpu_backend.h"
#include "../src/gpu_shader_ir_internal.h"
#include "turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Minimal model config for session tests (no real weights needed)
static void init_test_config(BnConfig *c) {
    memset(c, 0, sizeof(*c));
    c->dim = 64;
    c->hidden_dim = 128;
    c->n_layers = 2;
    c->n_heads = 4;
    c->n_kv_heads = 4;
    c->vocab_size = 100;
    c->seq_len = 32;
    c->head_size = 16;
    c->kv_dim = 64;
    c->kv_mul = 1;
    c->rope_theta = 10000.0f;
    c->norm_eps = 1e-5f;
}

// Test: create and free two sessions from the same model
static void test_session_create_free(void) {
    printf("test_session_create_free... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);
    assert(bn_model_backend(&model) == NULL);

    BnSession *s1 = bn_session_create(&model, NULL);
    assert(s1 != NULL);
    assert(bn_model_backend(&model) == NULL);
    assert(s1->backend != NULL);
    assert(s1->state.x != NULL);
    assert(s1->state.logits != NULL);
    assert(s1->state.key_cache != NULL);
    assert(s1->pos == 0);

    BnSession *s2 = bn_session_create(&model, NULL);
    assert(s2 != NULL);
    assert(bn_model_backend(&model) == NULL);
    assert(s2->backend != NULL);
    assert(s2->state.x != NULL);

    // Sessions should have independent buffers
    assert(s1->state.x != s2->state.x);
    assert(s1->state.logits != s2->state.logits);
    assert(s1->state.key_cache != s2->state.key_cache);

    bn_session_free(s1, NULL);
    bn_session_free(s2, NULL);
    printf("PASSED\n");
}

// Test: KV cache isolation between sessions
static void test_session_kv_isolation(void) {
    printf("test_session_kv_isolation... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    BnSession *s1 = bn_session_create(&model, NULL);
    BnSession *s2 = bn_session_create(&model, NULL);
    assert(s1 && s2);

    int kv_dim = model.config.kv_dim;

    // Write pattern A to s1's KV cache
    for (int i = 0; i < kv_dim; i++) {
        s1->state.key_cache[i] = 1.0f;
        s1->state.value_cache[i] = 2.0f;
    }

    // Write pattern B to s2's KV cache
    for (int i = 0; i < kv_dim; i++) {
        s2->state.key_cache[i] = 3.0f;
        s2->state.value_cache[i] = 4.0f;
    }

    // Verify s1 still has pattern A
    for (int i = 0; i < kv_dim; i++) {
        assert(s1->state.key_cache[i] == 1.0f);
        assert(s1->state.value_cache[i] == 2.0f);
    }

    // Verify s2 has pattern B
    for (int i = 0; i < kv_dim; i++) {
        assert(s2->state.key_cache[i] == 3.0f);
        assert(s2->state.value_cache[i] == 4.0f);
    }

    bn_session_free(s1, NULL);
    bn_session_free(s2, NULL);
    printf("PASSED\n");
}

// Test: session reset clears KV cache and pos
static void test_session_reset(void) {
    printf("test_session_reset... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    BnSession *s = bn_session_create(&model, NULL);
    assert(s);

    // Write data to KV cache and advance pos
    int kv_dim = model.config.kv_dim;
    for (int i = 0; i < kv_dim; i++) {
        s->state.key_cache[i] = 42.0f;
        s->state.value_cache[i] = 42.0f;
    }
    s->pos = 10;

    // Reset
    bn_session_reset(s, &model);

    // Verify zeroed
    assert(s->pos == 0);
    for (int i = 0; i < kv_dim; i++) {
        assert(s->state.key_cache[i] == 0.0f);
        assert(s->state.value_cache[i] == 0.0f);
    }

    bn_session_free(s, NULL);
    printf("PASSED\n");
}

// Test: pos is per-session
static void test_session_pos_tracking(void) {
    printf("test_session_pos_tracking... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    BnSession *s1 = bn_session_create(&model, NULL);
    BnSession *s2 = bn_session_create(&model, NULL);
    assert(s1 && s2);

    s1->pos = 5;
    s2->pos = 10;

    assert(s1->pos == 5);
    assert(s2->pos == 10);

    bn_session_free(s1, NULL);
    bn_session_free(s2, NULL);
    printf("PASSED\n");
}

// Test: create N sessions from same model
static void test_multiple_sessions(void) {
    printf("test_multiple_sessions... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    #define N_SESSIONS 8
    BnSession *sessions[N_SESSIONS];

    for (int i = 0; i < N_SESSIONS; i++) {
        sessions[i] = bn_session_create(&model, NULL);
        assert(sessions[i] != NULL);
        sessions[i]->pos = i;
    }

    // Verify all independent
    for (int i = 0; i < N_SESSIONS; i++) {
        assert(sessions[i]->pos == i);
        // Write to each session's activation buffer
        sessions[i]->state.x[0] = (float)i;
    }

    // Verify no cross-contamination
    for (int i = 0; i < N_SESSIONS; i++) {
        assert(sessions[i]->state.x[0] == (float)i);
    }

    for (int i = 0; i < N_SESSIONS; i++)
        bn_session_free(sessions[i], NULL);

    printf("PASSED\n");
}

// Test: session reset clears TQ caches
static void test_session_reset_tq(void) {
    printf("test_session_reset_tq... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);
    model.config.kv_tq_bits = 3;

    BnTQState tq;
    assert(bn_tq_init(&tq, model.config.head_size, 3, 0x5451303042ULL) == 0);
    bn_model_set_tq_state(&model, &tq, 0);

    BnSession *s = bn_session_create(&model, NULL);
    assert(s);
    assert(s->state.key_cache_tq != NULL);
    assert(s->state.value_cache_tq != NULL);

    // Write non-zero data into TQ caches
    int kb = bn_tq_key_bytes(&tq);
    int vb = bn_tq_value_bytes(&tq);
    int n_attn = model.config.n_layers;
    size_t tq_key_total = (size_t)n_attn * (size_t)model.config.seq_len *
                          (size_t)model.config.n_kv_heads * (size_t)kb;
    size_t tq_val_total = (size_t)n_attn * (size_t)model.config.seq_len *
                          (size_t)model.config.n_kv_heads * (size_t)vb;
    memset(s->state.key_cache_tq, 0xAA, tq_key_total);
    memset(s->state.value_cache_tq, 0xBB, tq_val_total);
    s->pos = 10;

    // Reset
    bn_session_reset(s, &model);

    // Verify zeroed
    assert(s->pos == 0);
    for (size_t i = 0; i < tq_key_total; i++)
        assert(s->state.key_cache_tq[i] == 0);
    for (size_t i = 0; i < tq_val_total; i++)
        assert(s->state.value_cache_tq[i] == 0);

    bn_session_free(s, NULL);
    bn_tq_free(&tq);
    printf("PASSED\n");
}

// Test: cached GPU op graphs are per-session, not model-owned
static void test_session_gpu_graph_isolation(void) {
    printf("test_session_gpu_graph_isolation... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);

    BnSession *s1 = bn_session_create(&model, NULL);
    BnSession *s2 = bn_session_create(&model, NULL);
    assert(s1 && s2);
    assert(bn_backend_session_gpu_graph(s1->backend) == NULL);
    assert(bn_backend_session_gpu_graph(s2->backend) == NULL);

    BnGPUGraph *g1 =
        (BnGPUGraph *)bn_backend_session_ensure_gpu_graph(s1->backend, 4);
    BnGPUGraph *g2 =
        (BnGPUGraph *)bn_backend_session_ensure_gpu_graph(s2->backend, 8);
    assert(g1 && g2);
    assert(g1->ops && g2->ops);
    g1->cap = 4;
    g2->cap = 8;
    assert(bn_backend_session_gpu_graph(s1->backend) !=
           bn_backend_session_gpu_graph(s2->backend));
    assert(bn_backend_session_ensure_gpu_graph(s1->backend, 2) == g1);
    BnGPUGraph *g1_grown =
        (BnGPUGraph *)bn_backend_session_ensure_gpu_graph(s1->backend, 16);
    assert(g1_grown != NULL);
    assert(g1_grown->cap == 16);
    assert(g1_grown->ops != NULL);
    bn_backend_session_release_gpu_graph(s1->backend);
    assert(bn_backend_session_gpu_graph(s1->backend) == NULL);

    bn_session_free(s1, NULL);
    bn_session_free(s2, NULL);
    printf("PASSED\n");
}

int main(void) {
    test_session_create_free();
    test_session_kv_isolation();
    test_session_reset();
    test_session_pos_tracking();
    test_multiple_sessions();
    test_session_reset_tq();
    test_session_gpu_graph_isolation();
    printf("\nAll session tests passed!\n");
    return 0;
}
