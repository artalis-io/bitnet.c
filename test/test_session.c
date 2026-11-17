#include "session.h"
#include "session_internal.h"
#include "model.h"
#include "model_internal.h"
#include "backend_session.h"
#include "backend_model.h"
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

static int reset_activation_calls;

static int mock_reset_activations(void *ctx) {
    (void)ctx;
    reset_activation_calls++;
    return 0;
}

typedef struct {
    BnBackendSession *session;
    const void *owner;
    int *release_count;
    float recurrent_state;
} MockGPUResources;

static void mock_release_gpu_resources(void *ptr) {
    MockGPUResources *resources = (MockGPUResources *)ptr;
    assert(bn_backend_session_gpu_resources(resources->session,
                                            resources->owner) == NULL);
    ++*resources->release_count;
    free(resources);
}

static MockGPUResources *mock_gpu_resources(BnSession *session,
    const void *owner, int *release_count) {
    MockGPUResources *resources = (MockGPUResources *)calloc(1, sizeof(*resources));
    assert(resources);
    resources->session = bn_session_backend(session);
    resources->owner = owner;
    resources->release_count = release_count;
    return resources;
}

static void test_session_gpu_resource_lifetime(void) {
    printf("test_session_gpu_resource_lifetime... ");
    BnModel model = {0};
    init_test_config(&model.config);
    BnSession *s1 = bn_session_create(&model, NULL);
    BnSession *s2 = bn_session_create(&model, NULL);
    assert(s1 && s2);
    BnBackendSession *b1 = bn_session_backend(s1);
    BnBackendSession *b2 = bn_session_backend(s2);
    int owner1 = 0, owner2 = 0, releases1 = 0, releases2 = 0;
    MockGPUResources *r1 = mock_gpu_resources(s1, &owner1, &releases1);
    MockGPUResources *r2 = mock_gpu_resources(s2, &owner1, &releases2);
    assert(bn_backend_session_attach_gpu_resources(NULL, &owner1, r1, mock_release_gpu_resources) == -1);
    assert(bn_backend_session_attach_gpu_resources(b1, NULL, r1, mock_release_gpu_resources) == -1);
    assert(bn_backend_session_attach_gpu_resources(b1, &owner1, NULL, mock_release_gpu_resources) == -1);
    assert(bn_backend_session_attach_gpu_resources(b1, &owner1, r1, NULL) == -1);
    assert(bn_backend_session_gpu_resources(b1, &owner1) == NULL);
    assert(bn_backend_session_gpu_resources(NULL, &owner1) == NULL);
    assert(releases1 == 0);
    assert(bn_backend_session_attach_gpu_resources(b1, &owner1, r1, mock_release_gpu_resources) == 0);
    assert(bn_backend_session_attach_gpu_resources(b2, &owner1, r2, mock_release_gpu_resources) == 0);
    assert(bn_backend_session_gpu_resources(b1, NULL) == NULL);
    assert(bn_backend_session_gpu_resources(b1, &owner2) == NULL);
    assert(bn_backend_session_gpu_resources(b1, &owner1) == r1);
    assert(bn_backend_session_gpu_resources(b2, &owner1) == r2);
    r1->recurrent_state = 17.0f;
    r2->recurrent_state = 29.0f;
    /* A rejected replacement remains caller-owned, including another owner. */
    MockGPUResources *replacement = mock_gpu_resources(s1, &owner2, &releases1);
    assert(bn_backend_session_attach_gpu_resources(b1, &owner1, r1, mock_release_gpu_resources) == -1);
    assert(bn_backend_session_attach_gpu_resources(b1, &owner2, replacement, mock_release_gpu_resources) == -1);
    assert(releases1 == 0 && r1->recurrent_state == 17.0f);
    free(replacement);
    bn_session_reset(s1, &model);
    assert(releases1 == 1 && releases2 == 0);
    assert(bn_backend_session_gpu_resources(b1, &owner1) == NULL);
    assert(bn_backend_session_gpu_resources(b2, &owner1) == r2);
    assert(r2->recurrent_state == 29.0f);
    bn_session_reset(s1, &model);
    assert(releases1 == 1);
    r1 = mock_gpu_resources(s1, &owner2, &releases1);
    assert(bn_backend_session_attach_gpu_resources(b1, &owner2, r1, mock_release_gpu_resources) == 0);
    bn_session_free(s2, NULL);
    assert(releases1 == 1 && releases2 == 1);
    bn_session_free(s1, NULL);
    assert(releases1 == 2 && releases2 == 1);
    bn_backend_session_release_gpu_resources(NULL);
    printf("PASSED\n");
}

typedef struct {
    BnBackendSession *backend;
    const void *owner;
    int *gpu_releases;
    int expected_gpu_releases;
    int *host_releases;
} ExecutionProbe;

static void release_execution_probe(void *ptr) {
    ExecutionProbe *p = ptr;
    assert(*p->gpu_releases == p->expected_gpu_releases);
    assert(!bn_backend_session_has_gpu_resources(p->backend));
    assert(!bn_backend_session_has_execution_resources(p->backend));
    assert(bn_backend_session_execution_resources(p->backend, p->owner) == NULL);
    ++*p->host_releases;
    free(p);
}

static void test_session_execution_metadata_lifetime(void) {
    BnModel model = {0}; init_test_config(&model.config);
    BnSession *s = bn_session_create(&model, NULL); assert(s);
    BnBackendSession *b = bn_session_backend(s);
    int owner = 0, other = 0, gpu_releases = 0, host_releases = 0;
    ExecutionProbe *p = malloc(sizeof(*p)); assert(p);
    *p = (ExecutionProbe){b, &owner, &gpu_releases, 1, &host_releases};
    assert(!bn_backend_session_has_gpu_resources(NULL));
    assert(!bn_backend_session_has_execution_resources(NULL));
    assert(bn_backend_session_attach_execution_resources(NULL, &owner, p, release_execution_probe) == -1);
    assert(bn_backend_session_attach_execution_resources(b, NULL, p, release_execution_probe) == -1);
    assert(bn_backend_session_attach_execution_resources(b, &owner, NULL, release_execution_probe) == -1);
    assert(bn_backend_session_attach_execution_resources(b, &owner, p, NULL) == -1);
    assert(bn_backend_session_execution_resources(NULL, &owner) == NULL);
    MockGPUResources *gpu = mock_gpu_resources(s, &owner, &gpu_releases);
    assert(bn_backend_session_attach_gpu_resources(b, &owner, gpu, mock_release_gpu_resources) == 0);
    assert(bn_backend_session_has_gpu_resources(b));
    assert(bn_backend_session_attach_execution_resources(b, &owner, p, release_execution_probe) == 0);
    assert(bn_backend_session_has_execution_resources(b));
    assert(bn_backend_session_execution_resources(b, &owner) == p);
    assert(bn_backend_session_execution_resources(b, &other) == NULL);
    assert(bn_backend_session_execution_resources(b, NULL) == NULL);
    assert(bn_backend_session_attach_execution_resources(b, &owner, p, release_execution_probe) == -1);
    assert(bn_backend_session_attach_execution_resources(b, &other, p, release_execution_probe) == -1);
    bn_session_reset(s, &model);
    assert(gpu_releases == 1 && host_releases == 1);
    bn_session_reset(s, &model);
    assert(gpu_releases == 1 && host_releases == 1);
    /* Host-only partial setup is released even without a GPU container. */
    p = malloc(sizeof(*p)); assert(p);
    *p = (ExecutionProbe){b, &owner, &gpu_releases, 1, &host_releases};
    assert(bn_backend_session_attach_execution_resources(b, &owner, p, release_execution_probe) == 0);
    bn_session_free(s, NULL);
    assert(gpu_releases == 1 && host_releases == 2);
    puts("test_session_execution_metadata_lifetime... PASSED");
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
    assert(bn_session_matches_model(s1, &model));
    assert(!bn_session_matches_model(NULL, &model));
    assert(!bn_session_matches_model(s1, NULL));
    BnModel copy = model;
    assert(!bn_session_matches_model(s1, &copy));
    s1->pos = 7;
    bn_session_reset(s1, &copy);
    assert(s1->pos == 7);
    bn_session_reset(s1, &model);
    assert(!s1->pos);
    model.config.vocab_size++;
    assert(!bn_session_matches_model(s1, &model));
    model.config.vocab_size--;
    model.config.seq_len++;
    assert(!bn_session_matches_model(s1, &model));
    model.config.seq_len--;
    assert(bn_session_matches_model(s1, &model));
    assert(bn_model_backend(&model) == NULL);
    assert(bn_session_backend(s1) != NULL);
    assert(s1->state.x != NULL);
    assert(s1->state.logits != NULL);
    assert(s1->state.key_cache != NULL);
    assert(s1->pos == 0);
    assert(s1->state.batched_prompt_contract == 0);

    BnSession *s2 = bn_session_create(&model, NULL);
    assert(s2 != NULL);
    assert(bn_model_backend(&model) == NULL);
    assert(bn_session_backend(s2) != NULL);
    assert(s2->state.x != NULL);

    assert(!bn_backend_session_gpu_decode_ssm_pending(bn_session_backend(s1)));
    assert(!bn_backend_session_gpu_decode_ssm_pending(bn_session_backend(s2)));
    bn_backend_session_set_gpu_decode_ssm_pending(bn_session_backend(s1), 1);
    assert(bn_backend_session_gpu_decode_ssm_pending(bn_session_backend(s1)));
    assert(!bn_backend_session_gpu_decode_ssm_pending(bn_session_backend(s2)));

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
    s->state.batched_prompt_contract = 1;
    s->moe_cache_prefill = 1;
    bn_backend_session_set_gpu_decode_ssm_pending(bn_session_backend(s), 1);

    // Reset
    bn_session_reset(s, &model);

    // Verify zeroed
    assert(s->pos == 0);
    assert(s->state.batched_prompt_contract == 0);
    assert(s->moe_cache_prefill == 0);
    assert(!bn_backend_session_gpu_decode_ssm_pending(bn_session_backend(s)));
    for (int i = 0; i < kv_dim; i++) {
        assert(s->state.key_cache[i] == 0.0f);
        assert(s->state.value_cache[i] == 0.0f);
    }

    bn_session_free(s, NULL);
    printf("PASSED\n");
}

static void test_session_reset_resets_backend_activations(void) {
    printf("test_session_reset_resets_backend_activations... ");
    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);
    assert(bn_model_ensure_backend(&model) == 0);
    BnGPUBackend gpu = {0};
    gpu.reset_activations = mock_reset_activations;
    bn_backend_model_bind_gpu(bn_model_backend(&model), &gpu);
    BnSession *session = bn_session_create(&model, NULL);
    assert(session != NULL);
    reset_activation_calls = 0;
    bn_session_reset(session, &model);
    assert(reset_activation_calls == 1);
    bn_session_free(session, NULL);
    bn_backend_model_bind_gpu(bn_model_backend(&model), NULL);
    bn_model_free(&model);
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

static void test_session_ple_state_reset(void) {
    printf("test_session_ple_state_reset... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);
    model.config.hyper_connection_count = 4;
    model.config.hyper_connection_rank = 16;
    model.config.ple_head_count = 16;
    model.config.ple_head_dim = 4;
    model.config.ple_ngram_size = 3;
    model.config.ple_conv_kernel = 4;

    BnSession *s = bn_session_create(&model, NULL);
    assert(s != NULL);
    assert(s->state.token_history != NULL);
    assert(s->state.ple_conv_state != NULL);

    size_t wide = (size_t)model.config.hyper_connection_count *
                  model.config.dim;
    size_t history = (size_t)(model.config.ple_conv_kernel - 1) *
                     model.config.ple_ngram_size * wide;
    s->state.token_history[3] = 42;
    s->state.ple_conv_state[history - 1] = 3.0f;
    bn_session_reset(s, &model);
    assert(s->state.token_history[3] == 0);
    assert(s->state.ple_conv_state[history - 1] == 0.0f);

    bn_session_free(s, NULL);
    printf("PASSED\n");
}

// Test: shared expert policy sizes the reusable hidden buffers
static void test_session_shared_expert_hidden_buffers(void) {
    printf("test_session_shared_expert_hidden_buffers... ");

    BnModel base;
    memset(&base, 0, sizeof(base));
    init_test_config(&base.config);

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);
    model.config.has_shared_expert = 1;
    model.config.shared_expert_intermediate_size = 256;

    size_t base_size = bn_model_session_arena_size(&base.config,
                                                   &base.weights);
    size_t shared_size = bn_model_session_arena_size(&model.config,
                                                     &model.weights);
    size_t expected_growth = 2 * (size_t)(256 - base.config.hidden_dim) *
                             sizeof(float);
    assert(base_size > 0);
    assert(shared_size == base_size + expected_growth);

    BnSession *s = bn_session_create(&model, NULL);
    assert(s != NULL);
    s->state.hb[255] = 3.0f;
    s->state.hb2[255] = 7.0f;
    assert(s->state.hb[255] == 3.0f);
    assert(s->state.hb2[255] == 7.0f);

    bn_session_free(s, NULL);
    printf("PASSED\n");
}

// Test: routed MoE policy sizes request-local expert buffers
static void test_session_moe_route_buffers(void) {
    printf("test_session_moe_route_buffers... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);
    model.config.n_experts = 4;
    model.config.n_experts_active = 3;
    model.config.moe_intermediate_size = 192;

    BnSession *s = bn_session_create(&model, NULL);
    assert(s != NULL);
    assert(s->moe_state != NULL);

    BnMoEState *ms = s->moe_state;
    assert(ms->router_logits != NULL);
    assert(ms->expert_weights != NULL);
    assert(ms->expert_indices != NULL);
    assert(ms->expert_hb != NULL);
    assert(ms->expert_hb2 != NULL);
    assert(ms->expert_hb_batch[2] != NULL);
    assert(ms->expert_hb2_batch[2] != NULL);
    assert(ms->expert_down_batch[2] != NULL);
    assert(ms->down_x_q_bufs != NULL);

    ms->router_logits[3] = 1.0f;
    ms->expert_weights[2] = 2.0f;
    ms->expert_indices[2] = 3;
    ms->expert_hb[191] = 4.0f;
    ms->expert_hb2[191] = 5.0f;
    ms->expert_hb_batch[2][191] = 6.0f;
    ms->expert_hb2_batch[2][191] = 7.0f;
    ms->expert_down_batch[2][63] = 8.0f;
    ms->down_x_q_bufs[3 * 192 - 1] = 9;

    assert(ms->router_logits[3] == 1.0f);
    assert(ms->expert_weights[2] == 2.0f);
    assert(ms->expert_indices[2] == 3);
    assert(ms->expert_hb[191] == 4.0f);
    assert(ms->expert_hb2[191] == 5.0f);
    assert(ms->expert_hb_batch[2][191] == 6.0f);
    assert(ms->expert_hb2_batch[2][191] == 7.0f);
    assert(ms->expert_down_batch[2][63] == 8.0f);
    assert(ms->down_x_q_bufs[3 * 192 - 1] == 9);

    bn_session_free(s, NULL);
    printf("PASSED\n");
}

// Test: model RoPE policy initializes reusable session frequencies
static void test_session_rope_frequency_policy(void) {
    printf("test_session_rope_frequency_policy... ");

    BnModel model;
    memset(&model, 0, sizeof(model));
    init_test_config(&model.config);
    model.config.rope_dim_count = 8;
    model.config.rope_text_dims = 4;

    BnSession *s = bn_session_create(&model, NULL);
    assert(s != NULL);
    assert(s->state.rope_freq != NULL);
    assert(s->state.rope_freq[0] == 1.0f);
    assert(s->state.rope_freq[2] == 0.0f);
    assert(s->state.rope_freq[3] == 0.0f);

    bn_session_free(s, NULL);
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
    bn_model_free(&model);
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
    assert(bn_backend_session_gpu_graph(bn_session_backend(s1)) == NULL);
    assert(bn_backend_session_gpu_graph(bn_session_backend(s2)) == NULL);

    BnGPUGraph *g1 =
        (BnGPUGraph *)bn_backend_session_ensure_gpu_graph(
            bn_session_backend(s1), 4);
    BnGPUGraph *g2 =
        (BnGPUGraph *)bn_backend_session_ensure_gpu_graph(
            bn_session_backend(s2), 8);
    assert(g1 && g2);
    assert(g1->ops && g2->ops);
    g1->cap = 4;
    g2->cap = 8;
    assert(bn_backend_session_gpu_graph(bn_session_backend(s1)) !=
           bn_backend_session_gpu_graph(bn_session_backend(s2)));
    assert(bn_backend_session_ensure_gpu_graph(
               bn_session_backend(s1), 2) == g1);
    BnGPUGraph *g1_grown =
        (BnGPUGraph *)bn_backend_session_ensure_gpu_graph(
            bn_session_backend(s1), 16);
    assert(g1_grown != NULL);
    assert(g1_grown->cap == 16);
    assert(g1_grown->ops != NULL);
    bn_backend_session_release_gpu_graph(bn_session_backend(s1));
    assert(bn_backend_session_gpu_graph(bn_session_backend(s1)) == NULL);

    bn_session_free(s1, NULL);
    bn_session_free(s2, NULL);
    printf("PASSED\n");
}

static void test_backend_session_host_storage(void) {
    printf("test_backend_session_host_storage... ");
    BnModel model = {0};
    init_test_config(&model.config);
    BnSession *a = bn_session_create_backend(&model, NULL);
    BnSession *b = bn_session_create_backend(&model, NULL);
    assert(a && b && a->backend_only && b->backend_only);
    assert(a->arena->capacity < 4096);
    assert(a->state.logits && a->state.token_history);
    assert(a->state.logits != b->state.logits);
    assert(!a->state.x && !a->state.key_cache && !a->state.value_cache);
    assert(!a->state.ssm_state && !a->state.ssm_conv_state && !a->moe_state);
    a->state.logits[0] = 7;
    a->state.token_history[0] = 23;
    a->pos = 4;
    int owner = 0, releases = 0;
    MockGPUResources *r = mock_gpu_resources(a, &owner, &releases);
    assert(!bn_backend_session_attach_gpu_resources(bn_session_backend(a),
        &owner, r, mock_release_gpu_resources));
    bn_session_reset(a, &model);
    assert(releases == 1 && a->pos == 0 && a->backend_only);
    assert(a->state.token_history[0] == 0 && b->state.logits[0] == 0);
    assert(!bn_backend_session_has_gpu_resources(bn_session_backend(a)));
    bn_session_free(a, NULL);
    bn_session_free(b, NULL);
    assert(releases == 1);
    assert(!bn_session_create_backend(NULL, NULL));
    model.config.vocab_size = 0;
    assert(!bn_session_create_backend(&model, NULL));
    model.config.vocab_size = 100;
    model.config.seq_len = -1;
    assert(!bn_session_create_backend(&model, NULL));
    init_test_config(&model.config);
    BnModelRuntime runtime={0};model.runtime=&runtime;
    assert(!bn_json_init(&runtime.native_config,"{}",2));
    /* A native document without prepared/source-valid anatomy cannot use
     * otherwise valid legacy host dimensions. */
    assert(!bn_session_create_backend(&model,NULL));
    assert(!bn_session_create(&model,NULL));
    bn_json_free(&runtime.native_config);model.runtime=NULL;
    puts("PASSED");
}

int main(void) {
    test_backend_session_host_storage();
    test_session_create_free();
    test_session_gpu_resource_lifetime();
    test_session_execution_metadata_lifetime();
    test_session_kv_isolation();
    test_session_reset();
    test_session_reset_resets_backend_activations();
    test_session_pos_tracking();
    test_multiple_sessions();
    test_session_ple_state_reset();
    test_session_shared_expert_hidden_buffers();
    test_session_moe_route_buffers();
    test_session_rope_frequency_policy();
    test_session_reset_tq();
    test_session_gpu_graph_isolation();
    printf("\nAll session tests passed!\n");
    return 0;
}
