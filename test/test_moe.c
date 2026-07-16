#include "moe.h"
#include "model.h"
#include "model_arch.h"
#include "quant.h"
#include "../src/moe_internal.h"
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
    // fd is now on BnMoEIO (shared on BnModel), not per-session MoE state
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

    bn_moe_route(&ms, x, router_w, dim, n_experts, k, 1, 0.0f, NULL);

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

    bn_moe_route(&ms, x, router_w, dim, n_experts, k, 0, 2.0f, NULL);
    wsum = 0.0f;
    for (int i = 0; i < k; i++)
        wsum += ms.expert_weights[i];
    assert(wsum < 2.0f);
    assert(wsum > 1.0f);

    printf("PASSED\n");
}

// --- Test: expert map loading from GGUF tensor metadata ---

static BnGGUFTensorInfo make_tensor(char *name, uint32_t type,
                                    uint64_t cols, uint64_t rows,
                                    uint64_t experts, uint64_t offset) {
    BnGGUFTensorInfo info = {0};
    info.name = name;
    info.type = type;
    info.n_dims = 3;
    info.dims[0] = cols;
    info.dims[1] = rows;
    info.dims[2] = experts;
    info.offset = offset;
    return info;
}

static void test_expert_map_split(void) {
    printf("test_expert_map_split... ");

    BnGGUFTensorInfo tensors[3];
    tensors[0] = make_tensor("blk.0.ffn_gate_exps.weight", BN_GGUF_TENSOR_Q4_0, 16, 32, 4, 10);
    tensors[1] = make_tensor("blk.0.ffn_up_exps.weight", BN_GGUF_TENSOR_Q4_0, 16, 32, 4, 1000);
    tensors[2] = make_tensor("blk.0.ffn_down_exps.weight", BN_GGUF_TENSOR_Q4_0, 32, 16, 4, 2000);

    BnGGUFFile f = {0};
    f.n_tensors = 3;
    f.tensors = tensors;
    f.data_offset = 4096;

    BnMoEExpertTensorNames names = {
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_gate_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    };
    BnMoEExpertMap em = {0};
    assert(bn_moe_load_expert_map(&f, &names, 4, 32, &em) == 0);

    size_t gate_bytes = 0, down_bytes = 0;
    assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_Q4_0, 32 * 16, &gate_bytes));
    assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_Q4_0, 16 * 32, &down_bytes));
    assert(em.gate_offset == 4096 + 10);
    assert(em.up_offset == 4096 + 1000);
    assert(em.down_offset == 4096 + 2000);
    assert(em.expert_gate_bytes == gate_bytes);
    assert(em.expert_up_bytes == gate_bytes);
    assert(em.expert_down_bytes == down_bytes);
    assert(em.gate_stride == gate_bytes);
    assert(em.up_stride == gate_bytes);
    assert(em.down_stride == down_bytes);
    assert(em.gate_rows == 32 && em.gate_cols == 16);
    assert(em.down_rows == 16 && em.down_cols == 32);

    printf("PASSED\n");
}

static void test_expert_map_fused_gate_up(void) {
    printf("test_expert_map_fused_gate_up... ");

    BnGGUFTensorInfo tensors[2];
    tensors[0] = make_tensor("blk.0.ffn_gate_up_exps.weight", BN_GGUF_TENSOR_F32, 16, 64, 4, 30);
    tensors[1] = make_tensor("blk.0.ffn_down_exps.weight", BN_GGUF_TENSOR_F32, 32, 16, 4, 5000);

    BnGGUFFile f = {0};
    f.n_tensors = 2;
    f.tensors = tensors;
    f.data_offset = 8192;

    BnMoEExpertTensorNames names = {
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_gate_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    };
    BnMoEExpertMap em = {0};
    assert(bn_moe_load_expert_map(&f, &names, 4, 32, &em) == 0);

    size_t one_proj_bytes = 0, fused_bytes = 0, down_bytes = 0;
    assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_F32, 32 * 16, &one_proj_bytes));
    assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_F32, 64 * 16, &fused_bytes));
    assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_F32, 16 * 32, &down_bytes));
    assert(em.gate_offset == 8192 + 30);
    assert(em.up_offset == em.gate_offset + one_proj_bytes);
    assert(em.down_offset == 8192 + 5000);
    assert(em.expert_gate_bytes == one_proj_bytes);
    assert(em.expert_up_bytes == one_proj_bytes);
    assert(em.expert_down_bytes == down_bytes);
    assert(em.gate_stride == fused_bytes);
    assert(em.up_stride == fused_bytes);
    assert(em.down_stride == down_bytes);
    assert(em.gate_rows == 32 && em.up_rows == 32);
    assert(em.gate_cols == 16 && em.up_cols == 16);

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
    assert(lw.moe.router_weight == NULL);

    printf("PASSED\n");
}

static BnGGUFKeyValue make_u32_kv(char *key, uint32_t value) {
    BnGGUFKeyValue kv = {0};
    kv.key = key;
    kv.type = BN_GGUF_TYPE_UINT32;
    kv.value.u32 = value;
    return kv;
}

static void test_qwen2moe_arch_config(void) {
    printf("test_qwen2moe_arch_config... ");

    BnGGUFKeyValue kvs[3];
    kvs[0] = make_u32_kv("qwen2moe.expert_count", 2);
    kvs[1] = make_u32_kv("qwen2moe.expert_used_count", 2);
    kvs[2] = make_u32_kv("qwen2moe.expert_feed_forward_length", 8960);

    BnGGUFFile f = {0};
    f.n_kv = 3;
    f.kvs = kvs;

    BnConfig c = {0};
    const BnModelArchOps *ops = bn_model_arch_ops_for("qwen2moe");
    assert(ops != NULL);
    bn_model_arch_load_moe_config(&c, &f, ops, "qwen2moe");

    assert(c.n_experts == 2);
    assert(c.n_experts_active == 2);
    assert(c.moe_intermediate_size == 8960);
    assert(c.moe_norm_topk_prob == 0);
    assert(c.moe_exact_silu == 1);
    assert((c.policy_flags & BN_MODEL_ARCH_POLICY_MOE_EXACT_SILU) != 0);
    assert(bn_model_arch_moe_forces_float_kquant_gateup(&c));
    assert(bn_model_arch_moe_prefers_cuda_exact_attention(&c));
    assert(bn_model_arch_cpu_prefill_uses_decode_for_parity(&c));

    printf("PASSED\n");
}

// --- Test: SwiGLU activation (reference check) ---

static void test_swiglu(void) {
    printf("test_swiglu... ");

    // SwiGLU: SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
    float gate[8] = {-5.0f, -2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 5.0f, 9.0f};
    float up[8] = {1.0f, -3.0f, 0.25f, 4.0f, -2.0f, 3.0f, 0.5f, -1.0f};
    float out[8];

    bn_moe_swiglu(out, gate, up, 8, 1);
    for (int i = 0; i < 8; i++) {
        float expected = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
        assert(fabsf(out[i] - expected) < 1e-6f);
    }

    printf("PASSED\n");
}

// --- Test: weight normalization edge cases ---

static void test_route_uniform(void) {
    printf("test_route_uniform... ");

    int n_experts = 4;
    int k = 2;
    int dim = 2;

    BnMoEState ms = {0};
    // fd is now on BnMoEIO (shared on BnModel), not per-session MoE state
    float router_logits[4];
    float expert_weights[2];
    int expert_indices[2];
    ms.router_logits = router_logits;
    ms.expert_weights = expert_weights;
    ms.expert_indices = expert_indices;

    // All experts have equal logits (uniform routing)
    float x[2] = {1.0f, 0.0f};
    float router_w[8] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

    bn_moe_route(&ms, x, router_w, dim, n_experts, k, 1, 0.0f, NULL);

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
    test_expert_map_split();
    test_expert_map_fused_gate_up();
    test_moe_config_compat();
    test_qwen2moe_arch_config();
    test_swiglu();
    test_route_uniform();
    test_moe_cache();
    printf("All MoE tests passed!\n");
    return 0;
}
