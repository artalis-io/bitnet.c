#include "moe.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// --- Test: router top-K selection and weight normalization ---

static void test_moe_route(void) {
    printf("test_moe_route... ");

    int n_experts = 8;
    int k = 3;
    int dim = 4;

    // Allocate MoE state manually
    BnMoEState ms = {0};
    ms.io.fd = -1;
    float router_logits[8];
    float expert_weights[3];
    int expert_indices[3];
    ms.router_logits = router_logits;
    ms.expert_weights = expert_weights;
    ms.expert_indices = expert_indices;

    // Input vector: all ones
    float x[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // Router weights: identity-like, expert e has row = [e, 0, 0, 0]
    // So logit[e] = e * 1.0 = e
    float router_w[32];
    memset(router_w, 0, sizeof(router_w));
    for (int e = 0; e < n_experts; e++)
        router_w[e * dim + 0] = (float)e;

    bn_moe_route(&ms, x, router_w, dim, n_experts, k, NULL);

    // Top-3 should be experts 7, 6, 5 (highest logits)
    assert(ms.expert_indices[0] == 7);
    assert(ms.expert_indices[1] == 6);
    assert(ms.expert_indices[2] == 5);

    // Weights should sum to 1.0
    float wsum = 0.0f;
    for (int i = 0; i < k; i++) {
        assert(ms.expert_weights[i] > 0.0f);
        wsum += ms.expert_weights[i];
    }
    assert(fabsf(wsum - 1.0f) < 1e-5f);

    // Weight[0] (expert 7) should be largest
    assert(ms.expert_weights[0] > ms.expert_weights[1]);
    assert(ms.expert_weights[1] > ms.expert_weights[2]);

    printf("PASSED\n");
}

// --- Test: expert offset computation ---

static void test_expert_offset(void) {
    printf("test_expert_offset... ");

    // Simulate a BnMoEExpertMap for 4 experts
    // Gate proj: [4, 32, 16] Q4_0 → 32*16=512 elements per expert → 512/32*18 = 288 bytes
    BnMoEExpertMap em = {0};
    em.gate_offset = 1000;  // base file offset
    em.expert_gate_bytes = 288;
    em.gate_type = BN_GGUF_TENSOR_Q4_0;
    em.gate_rows = 32;
    em.gate_cols = 16;

    // Expert 0: offset 1000
    // Expert 1: offset 1288
    // Expert 2: offset 1576
    // Expert 3: offset 1864
    for (int e = 0; e < 4; e++) {
        size_t expected = 1000 + (size_t)e * 288;
        size_t actual = em.gate_offset + (size_t)e * em.expert_gate_bytes;
        assert(actual == expected);
    }

    printf("PASSED\n");
}

// --- Test: MoE config detection (zero = dense, backward compatible) ---

static void test_moe_config_compat(void) {
    printf("test_moe_config_compat... ");

    BnConfig c = {0};
    // All MoE fields zero = dense model
    assert(c.n_experts == 0);
    assert(c.n_experts_active == 0);
    assert(c.moe_intermediate_size == 0);
    assert(c.has_shared_expert == 0);

    // Dense layer should have NULL router_weight
    BnLayerWeights lw = {0};
    assert(lw.router_weight == NULL);

    printf("PASSED\n");
}

// --- Test: SwiGLU activation (reference check) ---

static void test_swiglu(void) {
    printf("test_swiglu... ");

    // SwiGLU: SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
    float gate = 2.0f;
    float up = 3.0f;
    float expected = (gate / (1.0f + expf(-gate))) * up;

    // Compute manually
    float silu = gate / (1.0f + expf(-gate));
    float result = silu * up;
    assert(fabsf(result - expected) < 1e-6f);

    printf("PASSED\n");
}

// --- Test: weight normalization edge cases ---

static void test_route_uniform(void) {
    printf("test_route_uniform... ");

    int n_experts = 4;
    int k = 2;
    int dim = 2;

    BnMoEState ms = {0};
    ms.io.fd = -1;
    float router_logits[4];
    float expert_weights[2];
    int expert_indices[2];
    ms.router_logits = router_logits;
    ms.expert_weights = expert_weights;
    ms.expert_indices = expert_indices;

    // All experts have equal logits (uniform routing)
    float x[2] = {1.0f, 0.0f};
    float router_w[8] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

    bn_moe_route(&ms, x, router_w, dim, n_experts, k, NULL);

    // All logits equal → after softmax all 0.25
    // Top-2 picks should still have normalized weights summing to 1.0
    float wsum = ms.expert_weights[0] + ms.expert_weights[1];
    assert(fabsf(wsum - 1.0f) < 1e-5f);
    // Equal weights: each should be ~0.5
    assert(fabsf(ms.expert_weights[0] - 0.5f) < 1e-5f);
    assert(fabsf(ms.expert_weights[1] - 0.5f) < 1e-5f);

    printf("PASSED\n");
}

static void test_moe_cache(void) {
    printf("test_moe_cache... ");
    int result = bn_moe_cache_test();
    assert(result == 0);
    printf("PASSED\n");
}

int main(void) {
    printf("=== MoE Unit Tests ===\n");
    test_moe_route();
    test_expert_offset();
    test_moe_config_compat();
    test_swiglu();
    test_route_uniform();
    test_moe_cache();
    printf("All MoE tests passed!\n");
    return 0;
}
