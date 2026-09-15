#include "moe.h"
#include "backend_model.h"
#include "gpu_backend.h"
#include "model.h"
#include "model_internal.h"
#include "model_arch.h"
#include "quant.h"
#include "../src/moe_internal.h"
#if defined(__ARM_NEON) || defined(__AVX2__)
#include "simd_helpers.h"
#endif
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

    bn_moe_route(&ms, x, router_w, dim, n_experts, k, 1, 0.0f, 0, NULL);

    // Top-3 should be experts 7, 6, 5 (highest logits)
    assert(ms.expert_indices[0] == 7);
    assert(ms.expert_indices[1] == 6);
    assert(ms.expert_indices[2] == 5);

    // Routing must preserve raw scores for diagnostics and reuse.
    for (int e = 0; e < n_experts; e++)
        assert(ms.router_logits[e] == (float)e);

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

    bn_moe_route(&ms, x, router_w, dim, n_experts, k, 0, 2.0f, 0, NULL);
    wsum = 0.0f;
    for (int i = 0; i < k; i++)
        wsum += ms.expert_weights[i];
    assert(wsum < 2.0f);
    assert(wsum > 1.0f);

    printf("PASSED\n");
}

static void test_moe_reference_router_accumulation(void) {
    printf("test_moe_reference_router_accumulation... ");
    enum { dim = 67, experts = 8, k = 2 };
    float x[dim], router_w[experts * dim], logits[experts];
    float expert_weights[k];
    int expert_indices[k];
    for (int d = 0; d < dim; d++)
        x[d] = cosf((float)d * 0.17f);
    for (int i = 0; i < experts * dim; i++)
        router_w[i] = sinf((float)i * 0.13f);

    bn_moe_route_buffers(logits, expert_indices, expert_weights, x, router_w,
                         dim, experts, k, 1, 1.0f, 1, NULL);
    for (int e = 0; e < experts; e++) {
        float expected = bn_moe_dot_row_reference(
            router_w + (size_t)e * dim, x, dim);
        assert(logits[e] == expected);
    }
    printf("PASSED\n");
}

static void test_moe_projection_buffer_layout(void) {
    printf("test_moe_projection_buffer_layout... ");
    BnConfig c = {0};
    BnWeights w = {0};
    BnLayerWeights layers[3] = {0};
    c.n_layers = 3;
    w.layers = layers;
    layers[0].moe.expert_map.expert_gate_bytes = 64;
    layers[0].moe.expert_map.expert_up_bytes = 80;
    layers[0].moe.expert_map.expert_down_bytes = 96;
    layers[1].moe.expert_map.expert_gate_bytes = 128;
    layers[1].moe.expert_map.expert_up_bytes = 72;
    layers[1].moe.expert_map.expert_down_bytes = 112;
    layers[2].moe.expert_map.expert_gate_bytes = 48;
    layers[2].moe.expert_map.expert_up_bytes = 144;
    layers[2].moe.expert_map.expert_down_bytes = 160;

    BnMoEProjectionBufferLayout layout =
        bn_moe_projection_buffer_layout(&c, &w);
    assert(layout.gate_bytes == 128);
    assert(layout.up_bytes == 144);
    assert(layout.down_bytes == 160);
    assert(layout.max_bytes == 160);
    assert(bn_moe_projection_buffer_layout(NULL, &w).max_bytes == 0);
    assert(bn_moe_projection_buffer_layout(&c, NULL).max_bytes == 0);
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
    assert(!em.gate_up_fused);
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
    assert(em.gate_up_fused);
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

static BnGGUFKeyValue make_str_kv(char *key, char *value) {
    BnGGUFKeyValue kv = {0};
    kv.key = key;
    kv.type = BN_GGUF_TYPE_STRING;
    kv.value.str.str = value;
    kv.value.str.len = strlen(value);
    return kv;
}

static void test_model_arch_gguf_uses_moe(void) {
    printf("test_model_arch_gguf_uses_moe... ");

    BnGGUFKeyValue qwen_moe_kvs[2];
    qwen_moe_kvs[0] = make_str_kv("general.architecture", "qwen35moe");
    qwen_moe_kvs[1] = make_u32_kv("qwen35moe.expert_count", 4);
    BnGGUFFile qwen_moe = {0};
    qwen_moe.n_kv = 2;
    qwen_moe.kvs = qwen_moe_kvs;
    assert(bn_model_arch_gguf_uses_moe(&qwen_moe));

    BnGGUFKeyValue gemma_dense_kvs[2];
    gemma_dense_kvs[0] = make_str_kv("general.architecture", "gemma4");
    gemma_dense_kvs[1] = make_u32_kv("gemma4.expert_count", 0);
    BnGGUFFile gemma_dense = {0};
    gemma_dense.n_kv = 2;
    gemma_dense.kvs = gemma_dense_kvs;
    assert(!bn_model_arch_gguf_uses_moe(&gemma_dense));

    BnGGUFKeyValue fallback_kvs[1];
    fallback_kvs[0] = make_u32_kv("llama.expert_count", 2);
    BnGGUFFile fallback = {0};
    fallback.n_kv = 1;
    fallback.kvs = fallback_kvs;
    assert(bn_model_arch_gguf_uses_moe(&fallback));

    printf("PASSED\n");
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
    assert(c.moe_uses_reference_silu == 1);
    assert((c.policy_flags & BN_MODEL_ARCH_POLICY_MOE_REFERENCE_SILU) != 0);
    assert(bn_model_arch_moe_requires_float_kquant_gateup_fallback(&c));
    assert(bn_moe_float_kquant_gateup_fallback_task_flags(&c) ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);
    assert(bn_model_arch_moe_requires_reference_attention(&c));
    assert(bn_model_arch_prefill_uses_decode_for_parity(&c));

    BnConfig dense = {0};
    assert(bn_moe_float_kquant_gateup_fallback_task_flags(&dense) == 0);

    printf("PASSED\n");
}

static void test_moe_execution_policy(void) {
    printf("test_moe_execution_policy... ");

    BnConfig c = {0};
    c.moe_uses_reference_silu = 1;
    c.act_type = 2;
    c.norm_eps = 1.0e-5f;
    BnMoEExecutionPolicy policy = bn_moe_execution_policy(&c);
    assert(!policy.uses_scaled_router_input);
    assert(!policy.uses_dense_residual_branch);
    assert(policy.uses_reference_silu == 1);
    assert(policy.activation == 2);
    assert(policy.norm_eps == 1.0e-5f);
    assert(bn_moe_policy_uses_reference_silu(&c) == 1);

    c.policy_flags = BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT;
    policy = bn_moe_execution_policy(&c);
    assert(policy.uses_scaled_router_input);
    assert(!policy.uses_dense_residual_branch);
    assert(policy.uses_reference_silu == 1);
    assert(bn_moe_policy_uses_reference_silu(&c) == 1);

    c.policy_flags = BN_MODEL_ARCH_POLICY_MOE_DENSE_RESIDUAL_BRANCH;
    policy = bn_moe_execution_policy(&c);
    assert(!policy.uses_scaled_router_input);
    assert(policy.uses_dense_residual_branch);
    assert(policy.uses_reference_silu == -1);
    assert(policy.norm_eps == bn_model_moe_policy_norm_epsilon(&c));
    assert(bn_moe_policy_uses_reference_silu(&c) == -1);

    policy = bn_moe_execution_policy(NULL);
    assert(!policy.uses_scaled_router_input);
    assert(!policy.uses_dense_residual_branch);
    assert(policy.uses_reference_silu == -1);
    assert(policy.activation == 0);
    assert(policy.norm_eps == 0.0f);
    assert(bn_model_moe_policy_norm_epsilon(NULL) == 0.0f);
    assert(bn_moe_policy_uses_reference_silu(NULL) == -1);

    printf("PASSED\n");
}

static void test_moe_prefill_policy(void) {
    printf("test_moe_prefill_policy... ");

    BnConfig c = {0};
    BnMoEPrefillPolicy policy = bn_moe_prefill_policy(&c);
    assert(!policy.requires_matvec_prefill);
    assert(!policy.uses_grouped_expert_route);
    assert(!policy.uses_matvec_router);
    assert(!bn_model_arch_moe_prefill_uses_matvec_router(NULL));
    c.policy_flags = bn_model_arch_ops_for("gemma4")->policy_flags;
    assert(!bn_moe_prefill_policy(&c).uses_matvec_router);
    c.policy_flags = bn_model_arch_ops_for("qwen3")->policy_flags |
                     bn_model_arch_ops_for("qwen3")->moe_policy_flags;
    assert(bn_moe_route_policy(&c).uses_reference_router_accumulation);
    assert(!bn_moe_prefill_policy(&c).uses_matvec_router);
    c.policy_flags = 0;

    c.n_experts = 4;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 128;
    c.moe_norm_topk_prob = 1;
    c.moe_expert_weights_scale = 0.5f;
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&c);
    assert(!route_policy.uses_reference_router_accumulation);
    assert(route_policy.total_experts == 4);
    assert(route_policy.active_experts == 2);
    assert(route_policy.expert_hidden_dim == 128);
    assert(route_policy.norm_topk_prob == 1);
    assert(route_policy.expert_weights_scale == 0.5f);
    c.policy_flags |=
        BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION;
    route_policy = bn_moe_route_policy(&c);
    assert(route_policy.uses_reference_router_accumulation);
    c.policy_flags &=
        ~BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION;
    assert(bn_moe_policy_normalizes_topk_route_weights(&c));
    c.moe_norm_topk_prob = 0;
    assert(!bn_moe_policy_normalizes_topk_route_weights(&c));
    c.moe_norm_topk_prob = 1;
    assert(!bn_moe_policy_normalizes_topk_route_weights(NULL));
    assert(bn_model_moe_policy_total_experts(&c) == 4);
    assert(bn_model_moe_policy_active_experts(&c) == 2);
    assert(bn_model_moe_policy_expert_hidden_dim(&c) == 128);
    assert(bn_model_arch_moe_route_shape_valid(&c));
    assert(bn_model_moe_policy_total_experts(NULL) == 0);
    assert(bn_model_moe_policy_active_experts(NULL) == 0);
    assert(bn_model_moe_policy_expert_hidden_dim(NULL) == 0);

    assert(!bn_model_arch_moe_route_shape_valid(NULL));
    c.n_experts_active = 0;
    assert(!bn_model_arch_moe_route_shape_valid(&c));
    c.n_experts_active = 2;
    c.moe_intermediate_size = 0;
    assert(!bn_model_arch_moe_route_shape_valid(&c));
    c.moe_intermediate_size = 128;
    route_policy = bn_moe_route_policy(NULL);
    assert(route_policy.total_experts == 0);
    assert(route_policy.active_experts == 0);
    assert(route_policy.expert_hidden_dim == 0);
    assert(route_policy.norm_topk_prob == 0);
    assert(route_policy.expert_weights_scale == 0.0f);
    c.has_shared_expert = 1;
    policy = bn_moe_prefill_policy(&c);
    assert(!policy.requires_matvec_prefill);
    assert(policy.uses_grouped_expert_route);

    c.n_experts = 2;
    c.n_experts_active = 2;
    c.has_shared_expert = 0;
    policy = bn_moe_prefill_policy(&c);
    assert(!policy.requires_matvec_prefill);
    assert(!policy.uses_grouped_expert_route);

    c.has_shared_expert = 1;
    policy = bn_moe_prefill_policy(&c);
    assert(policy.requires_matvec_prefill);
    assert(!policy.uses_grouped_expert_route);

    policy = bn_moe_prefill_policy(NULL);
    assert(!policy.requires_matvec_prefill);
    assert(!policy.uses_grouped_expert_route);

    assert(bn_moe_policy_uses_expert_weights(&c));
    assert(bn_moe_policy_uses_all_active_two_expert_set(&c));
    assert(!bn_moe_policy_uses_all_active_two_expert_route(&c, 0));
    c.moe_intermediate_size = 4096;
    assert(bn_moe_policy_uses_all_active_two_expert_route(&c, 0));
    BnMoEAllActiveTwoRouteResourcePolicy all_active_two_resources =
        bn_moe_all_active_two_route_resource_policy(&c);
    assert(all_active_two_resources.enabled);
    assert(all_active_two_resources.total_experts == 2);
    assert(all_active_two_resources.expert_hidden_dim == 4096);
    assert(all_active_two_resources.complement_route_from_expert == 1);
    assert(!bn_moe_policy_uses_grouped_expert_route(&c));
    BnConfig dense = {0};
    assert(!bn_moe_policy_uses_expert_weights(&dense));
    assert(!bn_moe_policy_uses_all_active_two_expert_set(&dense));
    assert(!bn_moe_policy_uses_all_active_two_expert_route(&dense, 0));
    all_active_two_resources =
        bn_moe_all_active_two_route_resource_policy(&dense);
    assert(!all_active_two_resources.enabled);
    assert(all_active_two_resources.total_experts == 0);
    assert(all_active_two_resources.expert_hidden_dim == 0);
    assert(all_active_two_resources.complement_route_from_expert == 0);
    assert(!bn_moe_policy_uses_grouped_expert_route(&dense));
    assert(!bn_moe_policy_uses_expert_weights(NULL));
    assert(!bn_moe_policy_uses_all_active_two_expert_set(NULL));
    assert(!bn_moe_policy_uses_all_active_two_expert_route(NULL, 0));
    assert(!bn_moe_policy_uses_grouped_expert_route(NULL));
    assert(!bn_moe_policy_layer_has_router(NULL));
    BnLayerWeights routed_lw = {0};
    assert(!bn_moe_policy_layer_has_router(&routed_lw));
    routed_lw.moe.router_weight = (float *)1;
    assert(bn_moe_policy_layer_has_router(&routed_lw));

    BnBackendModel *backend = bn_backend_model_create();
    assert(backend);
    BnMoEPrefillRoutedGPUResourcePolicy routed_resources =
        bn_moe_prefill_routed_gpu_resource_policy(NULL, 0);
    assert(!routed_resources.router);
    assert(!routed_resources.sub_norm);
    assert(!routed_resources.router_scale);
    assert(!routed_resources.routed_valid);
    assert(!routed_resources.norm_resid_valid);
    BnMoEPrefillResidentGPUResourcePolicy resident_resources =
        bn_moe_prefill_resident_gpu_resource_policy(NULL, 0);
    assert(!resident_resources.valid);
    int router_handle;
    int gate_all_handle;
    int up_all_handle;
    int down_all_handle;
    int norm_handle;
    int sub_norm_handle;
    int router_scale_handle;
    assert(bn_backend_model_register_handle(
               backend, 0, BN_BACKEND_HANDLE_MOE_ROUTER,
               &router_handle) == 0);
    assert(bn_backend_model_register_handle(
               backend, 0, BN_BACKEND_HANDLE_MOE_GATE_ALL,
               &gate_all_handle) == 0);
    assert(bn_backend_model_register_handle(
               backend, 0, BN_BACKEND_HANDLE_MOE_UP_ALL,
               &up_all_handle) == 0);
    assert(bn_backend_model_register_handle(
               backend, 0, BN_BACKEND_HANDLE_MOE_DOWN_ALL,
               &down_all_handle) == 0);
    BnBackendModelMoEPrefillRoutedResources backend_routed_resources =
        bn_backend_model_moe_prefill_routed_resources(backend, 0);
    assert(backend_routed_resources.router == &router_handle);
    assert(backend_routed_resources.gate_all == &gate_all_handle);
    assert(backend_routed_resources.up_all == &up_all_handle);
    assert(backend_routed_resources.down_all == &down_all_handle);
    assert(backend_routed_resources.routed_valid);
    assert(!backend_routed_resources.norm_resid_valid);
    BnBackendModelMoEPrefillResidentResources backend_resident_resources =
        bn_backend_model_moe_prefill_resident_resources(backend, 0);
    assert(backend_resident_resources.gate_all == &gate_all_handle);
    assert(backend_resident_resources.up_all == &up_all_handle);
    assert(backend_resident_resources.down_all == &down_all_handle);
    assert(backend_resident_resources.valid);
    routed_resources =
        bn_moe_prefill_routed_gpu_resource_policy(backend, 0);
    assert(routed_resources.router == &router_handle);
    assert(routed_resources.gate_all == &gate_all_handle);
    assert(routed_resources.up_all == &up_all_handle);
    assert(routed_resources.down_all == &down_all_handle);
    assert(routed_resources.routed_valid);
    assert(!routed_resources.norm_resid_valid);
    resident_resources =
        bn_moe_prefill_resident_gpu_resource_policy(backend, 0);
    assert(resident_resources.gate_all == &gate_all_handle);
    assert(resident_resources.up_all == &up_all_handle);
    assert(resident_resources.down_all == &down_all_handle);
    assert(resident_resources.valid);
    assert(bn_backend_model_register_handle(
               backend, 0, BN_BACKEND_HANDLE_FFN_NORM,
               &norm_handle) == 0);
    backend_routed_resources =
        bn_backend_model_moe_prefill_routed_resources(backend, 0);
    assert(backend_routed_resources.norm == &norm_handle);
    assert(backend_routed_resources.norm_resid_valid);
    routed_resources =
        bn_moe_prefill_routed_gpu_resource_policy(backend, 0);
    assert(routed_resources.norm == &norm_handle);
    assert(routed_resources.norm_resid_valid);
    assert(!routed_resources.sub_norm);
    assert(bn_backend_model_register_handle(
               backend, 0, BN_BACKEND_HANDLE_FFN_SUB_NORM,
               &sub_norm_handle) == 0);
    backend_routed_resources =
        bn_backend_model_moe_prefill_routed_resources(backend, 0);
    routed_resources = bn_moe_prefill_routed_gpu_resource_policy(backend, 0);
    assert(backend_routed_resources.sub_norm == &sub_norm_handle);
    assert(routed_resources.sub_norm == &sub_norm_handle);
    assert(routed_resources.norm == &norm_handle);
    assert(!bn_moe_prefill_routed_gpu_resource_policy(backend, 1).sub_norm);
    assert(bn_backend_model_register_handle(backend, 0,
        BN_BACKEND_HANDLE_MOE_ROUTER_SCALE, &router_scale_handle) == 0);
    assert(bn_backend_model_moe_prefill_routed_resources(backend, 0).router_scale
        == &router_scale_handle);
    assert(bn_moe_prefill_routed_gpu_resource_policy(backend, 0).router_scale
        == &router_scale_handle);
    assert(!bn_moe_prefill_routed_gpu_resource_policy(backend, 1).router_scale);
    bn_backend_model_free(backend);

    c.n_experts = 4;
    c.n_experts_active = 2;
    assert(!bn_moe_policy_uses_all_active_two_expert_set(&c));
    assert(!bn_moe_policy_uses_all_active_two_expert_route(&c, 0));
    assert(bn_moe_policy_uses_grouped_expert_route(&c));

    BnLayerWeights lw = {0};
    c.has_shared_expert = 1;
    c.shared_expert_intermediate_size = 256;
    assert(bn_moe_policy_has_shared_expert(&c, &lw));
    assert(bn_moe_policy_shared_expert_hidden_dim(&c) == 256);
    c.has_shared_expert = 0;
    assert(!bn_moe_policy_has_shared_expert(&c, &lw));
    assert(bn_moe_policy_shared_expert_hidden_dim(&c) == 0);
    lw.shared.shared_expert_gate = (float *)1;
    assert(bn_moe_policy_has_shared_expert(&c, &lw));
    assert(bn_moe_policy_shared_expert_hidden_dim(&c) == 0);
    lw.shared.shared_expert_gate = NULL;
    c.has_shared_expert = 1;
    assert(bn_moe_policy_shared_expert_hidden_dim(&c) == 256);
    c.shared_expert_intermediate_size = 0;
    assert(bn_moe_policy_shared_expert_hidden_dim(&c) == 0);
    c.shared_expert_intermediate_size = 256;
    assert(!bn_moe_policy_has_shared_expert(NULL, &lw));
    assert(!bn_moe_policy_has_shared_expert_gate_vector(&lw));
    assert(!bn_moe_shared_expert_gate_vector(&lw));
    lw.shared.shared_expert_gate = (float *)1;
    assert(bn_moe_policy_has_shared_expert_gate_vector(&lw));
    assert(bn_moe_shared_expert_gate_vector(&lw) ==
           lw.shared.shared_expert_gate);
    lw.shared.shared_expert_gate = NULL;
    assert(fabsf(bn_moe_shared_expert_gate_weight(&lw, NULL, 0) -
                 1.0f) < 1e-6f);
    float gate_vec[2] = {1.0f, -2.0f};
    float gate_x[2] = {3.0f, 1.0f};
    lw.shared.shared_expert_gate = gate_vec;
    float gate = bn_moe_shared_expert_gate_weight(&lw, gate_x, 2);
    assert(fabsf(gate - (1.0f / (1.0f + expf(-1.0f)))) < 1e-6f);
    lw.shared.shared_expert_gate = NULL;
    assert(!bn_moe_policy_has_loaded_shared_gate_projection(&lw));
    lw.shared.shared_gate.data = (void *)1;
    assert(bn_moe_policy_has_loaded_shared_gate_projection(&lw));
    lw.shared.shared_gate.data = NULL;
    assert(!bn_moe_policy_has_loaded_shared_expert_path(&c, &lw));
    lw.shared.shared_gate.data = (void *)1;
    assert(bn_moe_policy_has_loaded_shared_expert_path(&c, &lw));
    c.has_shared_expert = 0;
    assert(!bn_moe_policy_has_loaded_shared_expert_path(&c, &lw));
    lw.shared.shared_expert_gate = (float *)1;
    assert(bn_moe_policy_has_loaded_shared_expert_path(&c, &lw));
    lw.shared.shared_expert_gate = NULL;
    c.has_shared_expert = 1;
    lw.shared.shared_gate.data = NULL;
    assert(!bn_moe_policy_has_loaded_shared_expert(&c, &lw));
    lw.shared.shared_gate.data = (void *)1;
    assert(bn_moe_policy_has_loaded_shared_expert(&c, &lw));
    BnMoELoadedSharedExpertPolicy shared_policy =
        bn_moe_loaded_shared_expert_policy(&c, &lw);
    assert(shared_policy.has_loaded_path);
    assert(shared_policy.hidden_dim == 256);
    c.has_shared_expert = 0;
    assert(!bn_moe_policy_has_loaded_shared_expert(&c, &lw));
    shared_policy = bn_moe_loaded_shared_expert_policy(&c, &lw);
    assert(!shared_policy.has_loaded_path);
    assert(shared_policy.hidden_dim == 0);
    assert(!bn_moe_policy_has_loaded_shared_expert(NULL, &lw));
    c.has_shared_expert = 1;
    assert(!bn_moe_policy_has_loaded_shared_expert(&c, NULL));

    printf("PASSED\n");
}

static void test_moe_quant_policy_helpers(void) {
    printf("test_moe_quant_policy_helpers... ");

    assert(!bn_moe_quant_uses_embedded_tensor_scale(BN_GGUF_TENSOR_F32));
    assert(bn_moe_quant_embedded_tensor_scale_offset(BN_GGUF_TENSOR_F32,
                                                     4, 32) == 0);
    assert(bn_moe_quant_uses_embedded_tensor_scale(BN_GGUF_TENSOR_I2_S));
    assert(bn_moe_quant_embedded_tensor_scale_offset(BN_GGUF_TENSOR_I2_S,
                                                     4, 32) == 32);
    assert(bn_moe_policy_supports_shared_gateup_batch_type(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_moe_policy_supports_shared_gateup_batch_type(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q4_0));
    assert(bn_moe_policy_supports_shared_gateup_batch_type(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_moe_policy_supports_shared_gateup_batch_type(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K, BN_GGUF_TENSOR_Q8_K));
    assert(bn_moe_policy_supports_shared_gateup_batch_type_on_cpu(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0, 0));
    assert(!bn_moe_policy_supports_shared_gateup_batch_type_on_cpu(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q4_0, 0));
    assert(!bn_moe_policy_supports_shared_gateup_batch_type_on_cpu(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K, BN_GGUF_TENSOR_Q5_K, 0));
    assert(bn_moe_policy_supports_shared_gateup_batch_type_on_cpu(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K, BN_GGUF_TENSOR_Q5_K, 1));
    BnMoESharedGateupBatchPolicy shared_batch_policy =
        bn_moe_shared_gateup_batch_policy(
            BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0,
            BN_GGUF_TENSOR_Q4_0, 0);
    assert(shared_batch_policy.can_batch);
    shared_batch_policy = bn_moe_shared_gateup_batch_policy(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q8_0,
        BN_GGUF_TENSOR_Q4_0, 1);
    assert(!shared_batch_policy.can_batch);
    shared_batch_policy = bn_moe_shared_gateup_batch_policy(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K,
        BN_GGUF_TENSOR_Q5_K, 0);
    assert(!shared_batch_policy.can_batch);
    shared_batch_policy = bn_moe_shared_gateup_batch_policy(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K,
        BN_GGUF_TENSOR_Q5_K, 1);
    assert(shared_batch_policy.can_batch);

    BnLayerWeights lw = {0};
    float shared_gate_data[4] = {0};
    float shared_up_data[4] = {0};
    float shared_down_data[4] = {0};
    float gate_out[4] = {0};
    float up_out[4] = {0};
    BnMatvecTask shared_tasks[2];
    BnMoESharedExpertWeights shared_weights;
    assert(!bn_moe_policy_can_batch_loaded_shared_gateup(NULL, 0, &lw));
    assert(bn_moe_shared_expert_gateup_tasks(
               shared_tasks, gate_out, up_out, &lw, 123u) == 0);
    assert(bn_moe_shared_expert_down_weight(&lw) == NULL);
    assert(!bn_moe_shared_expert_projection_weights(
        &shared_weights, &lw));

    lw.shared.shared_gate.data = shared_gate_data;
    lw.shared.shared_gate.type = BN_GGUF_TENSOR_Q4_0;
    lw.shared.shared_up.data = shared_up_data;
    lw.shared.shared_up.type = BN_GGUF_TENSOR_Q4_0;
    lw.shared.shared_down.data = shared_down_data;
    lw.shared.shared_down.type = BN_GGUF_TENSOR_Q4_0;

    BnQWeight routed = {0};
    routed.type = BN_GGUF_TENSOR_Q4_0;
    BnMatvecTask routed_tasks[1] = {{ gate_out, &routed, NULL, 0 }};
    assert(bn_moe_policy_can_batch_loaded_shared_gateup(
        routed_tasks, 1, &lw));
    assert(bn_moe_shared_expert_gateup_tasks(
               shared_tasks, gate_out, up_out, &lw, 123u) == 2);
    assert(shared_tasks[0].out == gate_out);
    assert(shared_tasks[0].W == &lw.shared.shared_gate);
    assert(shared_tasks[0].flags == 123u);
    assert(shared_tasks[1].out == up_out);
    assert(shared_tasks[1].W == &lw.shared.shared_up);
    assert(shared_tasks[1].flags == 123u);
    assert(bn_moe_shared_expert_down_weight(&lw) ==
           &lw.shared.shared_down);
    assert(bn_moe_shared_expert_projection_weights(
        &shared_weights, &lw));
    assert(shared_weights.gate == &lw.shared.shared_gate);
    assert(shared_weights.up == &lw.shared.shared_up);
    assert(shared_weights.down == &lw.shared.shared_down);

    BnMoEExpertMap em = {0};
    float gate_proj[8] = {0};
    float up_proj[8] = {0};
    float down_proj[8] = {0};
    em.gate_type = BN_GGUF_TENSOR_Q4_0;
    em.gate_rows = 2;
    em.gate_cols = 4;
    em.up_type = BN_GGUF_TENSOR_Q5_0;
    em.up_rows = 3;
    em.up_cols = 5;
    em.down_type = BN_GGUF_TENSOR_Q8_0;
    em.down_rows = 4;
    em.down_cols = 6;
    BnQWeight expert_w = {0};
    assert(bn_moe_expert_projection_weight(&expert_w, gate_proj,
                                           &em, 0));
    assert(expert_w.data == gate_proj);
    assert(expert_w.type == BN_GGUF_TENSOR_Q4_0);
    assert(expert_w.rows == 2);
    assert(expert_w.cols == 4);
    assert(bn_moe_expert_projection_weight(&expert_w, up_proj,
                                           &em, 1));
    assert(expert_w.data == up_proj);
    assert(expert_w.type == BN_GGUF_TENSOR_Q5_0);
    assert(expert_w.rows == 3);
    assert(expert_w.cols == 5);
    assert(bn_moe_expert_projection_weight(&expert_w, down_proj,
                                           &em, 2));
    assert(expert_w.data == down_proj);
    assert(expert_w.type == BN_GGUF_TENSOR_Q8_0);
    assert(expert_w.rows == 4);
    assert(expert_w.cols == 6);
    assert(!bn_moe_expert_projection_weight(&expert_w, down_proj,
                                            &em, 3));
    assert(!bn_moe_expert_projection_weight(NULL, down_proj, &em, 2));
    assert(!bn_moe_expert_projection_weight(&expert_w, NULL, &em, 2));
    assert(!bn_moe_expert_projection_weight(&expert_w, down_proj, NULL, 2));
    BnMoERoutedExpertProjectionTypes expert_types = {0};
    assert(bn_moe_routed_expert_projection_types(&expert_types, &em));
    assert(expert_types.gate_type == BN_GGUF_TENSOR_Q4_0);
    assert(expert_types.up_type == BN_GGUF_TENSOR_Q5_0);
    assert(expert_types.down_type == BN_GGUF_TENSOR_Q8_0);
    assert(!bn_moe_routed_expert_projection_types(NULL, &em));
    assert(!bn_moe_routed_expert_projection_types(&expert_types, NULL));
    BnMoERoutedExpertProjectionLayout expert_layout = {0};
    assert(bn_moe_routed_expert_projection_layout(&expert_layout, &em));
    assert(expert_layout.gate_type == BN_GGUF_TENSOR_Q4_0);
    assert(expert_layout.gate_rows == 2);
    assert(expert_layout.gate_cols == 4);
    assert(expert_layout.up_type == BN_GGUF_TENSOR_Q5_0);
    assert(expert_layout.up_rows == 3);
    assert(expert_layout.up_cols == 5);
    assert(expert_layout.down_type == BN_GGUF_TENSOR_Q8_0);
    assert(expert_layout.down_rows == 4);
    assert(expert_layout.down_cols == 6);
    assert(!bn_moe_routed_expert_projection_layout(NULL, &em));
    assert(!bn_moe_routed_expert_projection_layout(&expert_layout, NULL));

    printf("PASSED\n");
}

static void test_moe_resident_routed_ffn_layout_policy(void) {
    printf("test_moe_resident_routed_ffn_layout_policy... ");

    BnConfig c = {0};
    c.dim = 16;
    c.moe_intermediate_size = 32;

    BnMoEExpertMap em = {0};
    em.gate_rows = 32;
    em.up_rows = 32;
    em.down_rows = 16;
    em.gate_cols = 16;
    em.up_cols = 16;
    em.down_cols = 32;
    assert(bn_moe_policy_supports_resident_routed_ffn_shape(16, 32, &em));
    assert(bn_moe_policy_supports_resident_routed_ffn_layout(&c, &em));
    assert(bn_moe_policy_supports_gateup_split_layout(&em));

    em.up_cols = 15;
    assert(!bn_moe_policy_supports_resident_routed_ffn_shape(16, 32, &em));
    assert(!bn_moe_policy_supports_resident_routed_ffn_layout(&c, &em));
    assert(!bn_moe_policy_supports_gateup_split_layout(&em));
    em.up_cols = 16;
    em.up_rows = 31;
    assert(!bn_moe_policy_supports_gateup_split_layout(&em));
    em.up_rows = 32;
    em.down_cols = 31;
    assert(!bn_moe_policy_supports_resident_routed_ffn_shape(16, 32, &em));
    assert(!bn_moe_policy_supports_resident_routed_ffn_layout(&c, &em));
    assert(!bn_moe_policy_supports_resident_routed_ffn_shape(16, 32, NULL));
    assert(!bn_moe_policy_supports_resident_routed_ffn_layout(NULL, &em));
    assert(!bn_moe_policy_supports_resident_routed_ffn_layout(&c, NULL));
    assert(!bn_moe_policy_supports_gateup_split_layout(NULL));

    printf("PASSED\n");
}

// --- Test: SwiGLU activation (reference check) ---

static void test_swiglu(void) {
    printf("test_swiglu... ");

    // SwiGLU: SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
    float gate[8] = {-5.0f, -2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 5.0f, 9.0f};
    float up[8] = {1.0f, -3.0f, 0.25f, 4.0f, -2.0f, 3.0f, 0.5f, -1.0f};
    float out[8];

    bn_moe_swiglu(out, gate, up, 8, 1, 0);
    for (int i = 0; i < 8; i++) {
        float expected = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
        assert(fabsf(out[i] - expected) < 1e-6f);
    }

    // Standard GEGLU keeps the FP32 tanh approximation.
    float gelu_gate[4] = {-1.2345f, -0.12345f, 0.12345f, 1.2345f};
    float gelu_up[4] = {0.5f, -2.0f, 3.0f, -0.75f};
    bn_moe_swiglu(out, gelu_gate, gelu_up, 4, -1, 0);
    for (int i = 0; i < 4; i++) {
        float x = gelu_gate[i];
        float inner = 0.7978845608028654f * x *
                      (1.0f + 0.044715f * x * x);
        float expected = 0.5f * x * (1.0f + tanhf(inner)) * gelu_up[i];
        assert(fabsf(out[i] - expected) < 1e-7f);
    }

    // Reference activation policy rounds the GELU boundary through FP16.
    bn_moe_swiglu(out, gelu_gate, gelu_up, 4, -1, 1);
    for (int i = 0; i < 4; i++) {
        float x = bn_fp16_to_fp32(bn_fp32_to_fp16(gelu_gate[i]));
        float inner = 0.7978845608028654f * x *
                      (1.0f + 0.044715f * x * x);
        float gelu = 0.5f * x * (1.0f + tanhf(inner));
        float rounded_gelu = bn_fp16_to_fp32(bn_fp32_to_fp16(gelu));
        assert(out[i] == rounded_gelu * gelu_up[i]);
    }

    printf("PASSED\n");
}

static void test_reference_gelu_limits(void) {
    printf("test_reference_gelu_limits... ");
    // Pinned llama.cpp CPU GELU: table lookup below 10, original FP32
    // input at/above 10. Include a real Gemma4 expert gate and a value
    // that would overflow FP16 if the limit were checked after conversion.
    const float gate[8] = {
        -10.001f, -10.0f, -9.999f, 9.999f,
        10.0f, 10.001f, 16.4173756f, 70000.0f
    };
    const float up[8] = {1.0f, 2.0f, 0.5f, -2.0f,
                         0.25f, 1.0f, 1.0f, 0.5f};
    const float expected[8] = {
        0.0f, 0.0f, 0.0f, -20.0f,
        2.5f, 10.001f, 16.4173756f, 35000.0f
    };
    float direct[8], ranged[8];
    bn_moe_swiglu(direct, gate, up, 8, -1, 1);
    BnSwiGLUCtx ctx = {
        .hb = ranged, .gate = gate, .up = up,
        .uses_reference_silu = -1, .uses_reference_ffn_activation = 1
    };
    bn_moe_swiglu_range(&ctx, 0, 5);
    bn_moe_swiglu_range(&ctx, 5, 8);
    for (int i = 0; i < 8; i++) {
        assert(direct[i] == expected[i]);
        assert(ranged[i] == expected[i]);
    }
    printf("PASSED\n");
}

static void test_reference_gelu_table_entry(void) {
    printf("test_reference_gelu_table_entry... ");
    // Independent pinned GGML CPU outputs for FP16 bffe/bfff/c000,
    // plus the real Gemma4 expert input that rounds to bfff.
    const float gate[4] = {-0x1.ff8p+0f, -0x1.ffcp+0f,
                           -0x1p+1f, -1.99877179f};
    const float up[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float expected[4] = {-0x1.754p-5f, -0x1.74cp-5f,
                               -0x1.74p-5f, -0x1.74cp-5f};
    float direct[4], ranged[4];
    bn_moe_swiglu(direct, gate, up, 4, -1, 1);
    BnSwiGLUCtx ctx = {
        .hb = ranged, .gate = gate, .up = up,
        .uses_reference_silu = -1, .uses_reference_ffn_activation = 1
    };
    bn_moe_swiglu_range(&ctx, 0, 2);
    bn_moe_swiglu_range(&ctx, 2, 4);
    for (int i = 0; i < 4; i++) {
        assert(direct[i] == expected[i]);
        assert(ranged[i] == expected[i]);
    }
    printf("PASSED\n");
}

// --- Test: weight normalization edge cases ---

// Golden values from the pinned GGML CPU RMS_NORM/SCALE/MUL and
// SCALE/SCALE/ADD graphs; these arrays do not recompute the implementation.
static const float router_scaled_reference[32] = {
0x1.3ffdf6p-2f,
0x1.75fd9cp-11f,
0x1.1ffe2ap-12f,
-0x1.35fe06p-11f,
-0x1.edfcdap-10f,
0x1.67fdb4p-9f,
0x1.9ffd58p-10f,
0x0p+0f,
-0x1.0ffe44p-11f,
-0x1.7ffd8cp-10f,
0x1.92fd7p-10f,
0x1.7bfd94p-11f,
-0x1.0dfe48p-11f,
-0x1.1dfe2cp-9f,
0x1.67fdb4p-11f,
0x1.53fdd6p-11f,
0x1.7ffd8cp-13f,
-0x1.73fda2p-11f,
-0x1.09fe4ep-9f,
0x1.517dd8p-9f,
0x1.6bfdaep-10f,
-0x1.3ffdf6p-15f,
-0x1.31fe0cp-11f,
-0x1.97fd66p-10f,
0x1.73fda2p-10f,
0x1.2ffe1p-11f,
-0x1.67fdb4p-11f,
-0x1.37fe04p-9f,
0x1.53fdd4p-11f,
0x1.31fe0cp-11f,
0x1.7ffd8cp-14f,
-0x1.b1fd3ap-11f
};
static const float router_unscaled_reference[32] = {
0x1.fffcbcp-1f,
0x1.5ffdcp-10f,
0x1.7ffd8ep-12f,
-0x1.3ffdf6p-11f,
-0x1.9ffd5ap-10f,
0x1.fffcbcp-10f,
0x1.fffcbcp-11f,
0x0p+0f,
-0x1.fffcbcp-11f,
-0x1.fffcbcp-10f,
0x1.9ffd5ap-10f,
0x1.3ffdf6p-11f,
-0x1.7ffd8ep-12f,
-0x1.5ffdcp-10f,
0x1.1ffe2ap-9f,
0x1.3ffdf6p-10f,
0x1.fffcbcp-13f,
-0x1.7ffd8ep-11f,
-0x1.bffd24p-10f,
0x1.dffcfp-10f,
0x1.bffd24p-11f,
-0x1.fffcbcp-14f,
-0x1.1ffe2ap-10f,
-0x1.0ffe44p-9f,
0x1.7ffd8ep-10f,
0x1.fffcbcp-12f,
-0x1.fffcbcp-12f,
-0x1.7ffd8ep-10f,
0x1.0ffe44p-9f,
0x1.1ffe2ap-10f,
0x1.fffcbcp-14f,
-0x1.bffd24p-11f
};
static const float expert_scaled_reference[17] = {
-0x1.dac408p-1f,
-0x1.6173dap-1f,
-0x1.d04754p-2f,
-0x1.788ddap-1f,
-0x1.fe7b54p-2f,
-0x1.0bdaf8p-2f,
-0x1.1657acp-1f,
-0x1.3a0efap-2f,
-0x1.1dba78p-4f,
-0x1.6842fcp-2f,
-0x1.d68a8p-4f,
0x1.f3f6f4p-4f,
-0x1.47ad44p-3f,
0x1.3b26ecp-4f,
0x1.416a18p-2f,
0x1.04adcp-5f,
0x1.133618p-2f
};

static void test_scaled_router_input(void) {
    printf("test_scaled_router_input... ");
    float x[32], scale[32], out[32], inplace[32];
    for (int i = 0; i < 32; i++) {
        x[i] = i ? ((i * 29) % 37 - 18) * 0.125f : 1024.0f;
        scale[i] = 0.3125f + (i % 7) * 0.21875f;
    }
    bn_moe_scaled_router_input(out, x, scale, 32, 1e-6f);
    assert(memcmp(out, router_scaled_reference, sizeof(out)) == 0);
    memcpy(inplace, x, sizeof(x));
    bn_moe_scaled_router_input(inplace, inplace, scale, 32, 1e-6f);
    assert(memcmp(inplace, router_scaled_reference, sizeof(inplace)) == 0);
    bn_moe_scaled_router_input(out, x, NULL, 32, 1e-6f);
    assert(memcmp(out, router_unscaled_reference, sizeof(out)) == 0);
    printf("PASSED\n");
}

static void test_expert_scale_order(void) {
    printf("test_expert_scale_order... ");
    float down[17], out[17];
    for (int i = 0; i < 17; i++) {
        down[i] = i * 0.3125f - 2.71875f;
        out[i] = (i % 3) * 0.173f - 0.371f;
    }
    bn_moe_scale_expert_output(down, 1.731f, 17);
    bn_moe_scale_expert_output(down, 0.118201949f, 17);
    bn_moe_residual_add(out, down, 17);
    assert(memcmp(out, expert_scaled_reference, sizeof(out)) == 0);
    printf("PASSED\n");
}

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

    bn_moe_route(&ms, x, router_w, dim, n_experts, k, 1, 0.0f, 0, NULL);

    // All logits equal → after softmax all 0.25
    // Top-2 picks should still have normalized weights summing to 1.0
    float wsum = ms.expert_weights[0] + ms.expert_weights[1];
    assert(fabsf(wsum - 1.0f) < 1e-5f);
    // Equal weights: each should be ~0.5
    assert(fabsf(ms.expert_weights[0] - 0.5f) < 1e-5f);
    assert(fabsf(ms.expert_weights[1] - 0.5f) < 1e-5f);

    printf("PASSED\n");
}

static void test_router_softmax_backend(void) {
    printf("test_router_softmax_backend... ");

    float logits[8] = {
        -6.15005159f, -5.27897453f, -2.91207242f, -5.36358547f,
        -4.85903406f, -5.60444450f, -6.29297018f, -2.38901019f,
    };
    float probs[8];
    double sum = bn_moe_softmax_exp(probs, logits, 8, logits[7]);
    assert(sum > 1.0);
    for (int i = 0; i < 8; i++) {
        assert(isfinite(probs[i]));
        assert(probs[i] > 0.0f);
    }
    assert(probs[7] == 1.0f);

#if defined(__ARM_NEON) && defined(__aarch64__)
    float expected[8];
    double expected_sum = 0.0;
    float32x4_t maxv = vdupq_n_f32(logits[7]);
    for (int i = 0; i < 8; i += 4) {
        float32x4_t values = bn_neon_fast_exp_f32(
            vsubq_f32(vld1q_f32(logits + i), maxv));
        vst1q_f32(expected + i, values);
        expected_sum += (double)vaddvq_f32(values);
    }
    assert(memcmp(probs, expected, sizeof(probs)) == 0);
    assert(sum == expected_sum);
#elif defined(__AVX2__)
    float logits16[16] = {
        -6.15005159f, -5.27897453f, -2.91207242f, -5.36358547f,
        -4.85903406f, -5.60444450f, -6.29297018f, -2.38901019f,
        -1.125f, 0.0f, -9.25f, -3.75f, 1.125f, -0.125f, -7.0f, -2.0f,
    };
    float probs16[16], expected16[16];
    double sum16 = bn_moe_softmax_exp(probs16, logits16, 16, 1.125f);
    double expected_sum16 = 0.0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 maxv = _mm512_set1_ps(1.125f);
    __m512 values = bn_avx512_fast_exp_ps(
        _mm512_sub_ps(_mm512_loadu_ps(logits16), maxv));
    _mm512_storeu_ps(expected16, values);
    expected_sum16 = (double)_mm512_reduce_add_ps(values);
#else
    __m256 maxv = _mm256_set1_ps(1.125f);
    for (int i = 0; i < 16; i += 8) {
        __m256 values = bn_avx2_fast_exp_avx512_ps(
            _mm256_sub_ps(_mm256_loadu_ps(logits16 + i), maxv));
        _mm256_storeu_ps(expected16 + i, values);
    }
    float half[8], quarter[4];
    for (int i = 0; i < 8; i++)
        half[i] = expected16[i + 8] + expected16[i];
    for (int i = 0; i < 4; i++) quarter[i] = half[i + 4] + half[i];
    expected_sum16 = (double)((quarter[0] + quarter[2]) +
                              (quarter[1] + quarter[3]));
#endif
    assert(memcmp(probs16, expected16, sizeof(probs16)) == 0);
    assert(sum16 == expected_sum16);
#endif

    printf("PASSED\n");
}

static void test_moe_native_single_dot(void) {
#if defined(__AVX2__)
    printf("test_moe_native_single_dot... ");
    float w[2052], x[2052];
    const int dims[] = {1, 31, 32, 33, 63, 64, 65, 128, 131, 2048, 2051};
    for (int i = 0; i < 2052; i++) {
        w[i] = sinf((float)i * 0.13f);
        x[i] = cosf((float)i * 0.17f);
    }
    for (size_t c = 0; c < sizeof(dims) / sizeof(dims[0]); c++) {
        int dim = dims[c], d = 0;
#if defined(__AVX512F__)
        __m512 a[4] = {0};
        for (; d + 63 < dim; d += 64)
            for (int j = 0; j < 4; j++)
                a[j] = _mm512_fmadd_ps(_mm512_loadu_ps(w + 1 + d + j * 16),
                    _mm512_loadu_ps(x + 1 + d + j * 16), a[j]);
        float expected = _mm512_reduce_add_ps(_mm512_add_ps(
            _mm512_add_ps(a[0], a[2]), _mm512_add_ps(a[1], a[3])));
#else
        float a[4][16] = {{0}};
        for (; d + 63 < dim; d += 64)
            for (int group = 0; group < 4; group++)
                for (int lane = 0; lane < 16; lane++) {
                    int i = d + group * 16 + lane;
                    a[group][lane] = fmaf(w[1 + i], x[1 + i],
                                          a[group][lane]);
                }
        float merged[16], half[8], quarter[4];
        for (int lane = 0; lane < 16; lane++)
            merged[lane] = (a[0][lane] + a[2][lane]) +
                           (a[1][lane] + a[3][lane]);
        for (int lane = 0; lane < 8; lane++)
            half[lane] = merged[lane + 8] + merged[lane];
        for (int lane = 0; lane < 4; lane++)
            quarter[lane] = half[lane + 4] + half[lane];
        float expected = (quarter[0] + quarter[2]) +
                         (quarter[1] + quarter[3]);
#endif
        for (; d < dim; d++) expected += w[d + 1] * x[d + 1];
        assert(bn_moe_dot_row(w + 1, x + 1, dim) == expected);
        BnLayerWeights layer = {0};
        layer.shared.shared_expert_gate = w + 1;
        float gate = 1.0f / (1.0f + expf(-expected));
        assert(bn_moe_shared_expert_gate_weight(&layer, x + 1, dim) == gate);
    }
    printf("PASSED\n");
#endif
}

static void test_router_native_dot4(void) {
#if defined(__AVX512F__)
    printf("test_router_native_dot4... ");
    enum { max_dim = 131, rows = 8 };
    float weights[rows * max_dim], x[max_dim], actual[4];
    const int dimensions[] = {64, 128, 131};
    for (int i = 0; i < rows * max_dim; i++)
        weights[i] = sinf((float)i * 0.13f);
    for (int i = 0; i < max_dim; i++) x[i] = cosf((float)i * 0.17f);
    for (int n = 0; n < 3; n++) {
        int dim = dimensions[n];
        assert(bn_moe_dot4_rows(actual, weights, x, dim, 4));
        for (int r = 0; r < 4; r++) {
            const float *w = weights + (r + 4) * dim;
            __m512 sums[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
                             _mm512_setzero_ps(), _mm512_setzero_ps()};
            int d = 0;
            for (; d + 63 < dim; d += 64)
                for (int j = 0; j < 4; j++)
                    sums[j] = _mm512_fmadd_ps(_mm512_loadu_ps(w + d + j * 16),
                        _mm512_loadu_ps(x + d + j * 16), sums[j]);
            float expected = _mm512_reduce_add_ps(_mm512_add_ps(
                _mm512_add_ps(sums[0], sums[2]), _mm512_add_ps(sums[1], sums[3])));
            for (; d < dim; d++) expected += w[d] * x[d];
            assert(actual[r] == expected);
        }
    }
    printf("PASSED\n");
#endif
}

static void test_router_batch_logits(void) {
    printf("test_router_batch_logits... ");
    enum { dim = 35, experts = 9, tokens = 5, k = 3 };
    float weights[experts * dim], x[tokens * dim];
    float actual[tokens * experts], expected[tokens * experts];
    for (int i = 0; i < experts * dim; i++)
        weights[i] = (float)((i * 13) % 37 - 18) / 19.0f;
    for (int i = 0; i < tokens * dim; i++)
        x[i] = (float)((i * 7) % 29 - 14) / 17.0f;
    BnQWeight w = {weights, BN_GGUF_TENSOR_F32, experts, dim, 1.0f};
    BnThreadPool *workers = bn_tp_create(3);
    assert(workers);
    for (int threaded = 0; threaded < 2; threaded++) {
        BnThreadPool *pool = threaded ? workers : NULL;
        for (int n = 1; n <= tokens; n++) {
            for (int i = 0; i < tokens * experts; i++)
                actual[i] = 123.0f;
            int used = bn_moe_router_batch_logits(
                actual, weights, x, n, dim, experts, pool);
#if defined(__AVX2__)
            assert(used == (n > 1));
#else
            assert(!used);
#endif
            if (used) {
                bn_quant_matmul(expected, &w, x, n, NULL, pool);
                assert(memcmp(actual, expected,
                              (size_t)n * experts * sizeof(float)) == 0);
                for (int t = 0; t < n; t++) {
                    int indices[k];
                    float probabilities[k];
                    bn_moe_route_logits(actual + t * experts, indices,
                        probabilities, experts, k, 1, 2.0f);
                    float sum = 0.0f;
                    for (int j = 0; j < k; j++) {
                        assert(indices[j] >= 0 && indices[j] < experts);
                        assert(probabilities[j] > 0.0f);
                        if (j) assert(probabilities[j - 1] >= probabilities[j]);
                        sum += probabilities[j];
                    }
                    assert(fabsf(sum - 2.0f) < 1e-6f);
                }
                assert(memcmp(actual, expected,
                              (size_t)n * experts * sizeof(float)) == 0);
            }
            for (int i = used ? n * experts : 0; i < tokens * experts; i++)
                assert(actual[i] == 123.0f);
        }
    }
    bn_tp_free(workers);
    printf("PASSED\n");
}

static void test_moe_cache(void) {
    printf("test_moe_cache... ");
    int result = bn_moe_cache_test();
    assert(result == 0);
    printf("PASSED\n");
}

#if defined(__AVX2__)
typedef struct {
    BnBackendModel *backend;
    BnQWeight weight;
    size_t budget;
    int projections;
    int checked;
} PreparedDisabledObserve;

static void observe_disabled_preparation(void *ctx, BnMoEObservePoint point,
                                         int token, int slot, int expert,
                                         const float *values, int n_values) {
    (void)token; (void)slot; (void)expert; (void)values; (void)n_values;
    PreparedDisabledObserve *check = ctx;
    if (point != BN_MOE_OBSERVE_ROUTED_ACTIVATION || check->checked)
        return;
    /* Active prepared projections would remain pinned across this callback.
     * With preparation disabled, all warmed entries must be evictable. */
    bn_backend_model_set_cpu_prepared_cache_budget(check->backend, 0);
    bn_backend_model_set_cpu_prepared_cache_budget(check->backend, check->budget);
    for (int p = 0; p < check->projections; p++) {
        BnQWeight weight = check->weight;
        weight.data = (const BnBlockQ4K *)weight.data + p * weight.rows;
        const BnPreparedWeight *cached =
            bn_backend_model_acquire_cached_cpu_prepared(check->backend, &weight);
        assert(cached == NULL);
    }
    check->checked = 1;
}
#endif
static void test_moe_prefill_prepared_cache_tokens(int tokens) {
#if defined(__AVX2__)
    printf("test_moe_prefill_prepared_cache(tokens=%d)... ", tokens);
    enum { dim = 256, experts = 2, projections = experts * 3 };
    BnBlockQ4K *blocks = calloc((projections + 1) * dim, sizeof(*blocks));
    assert(blocks);
    for (int i = 0; i < (projections + 1) * dim; i++) {
        blocks[i].d = bn_fp32_to_fp16(0.001f * (1 + i % 5));
        blocks[i].dmin = bn_fp32_to_fp16(0.0003f);
        memset(blocks[i].scales, 1 + i % 7, sizeof(blocks[i].scales));
        for (size_t j = 0; j < sizeof(blocks[i].qs); j++)
            blocks[i].qs[j] = (uint8_t)(i * 17 + (int)j * 31);
    }
    const size_t projection_bytes = dim * sizeof(*blocks);
    float router[experts * dim] = {0};
    float input[tokens * dim], reference[tokens * dim];
    float actual[tokens * dim], scratch[tokens * dim];
    for (int i = 0; i < tokens * dim; i++)
        input[i] = sinf((float)i * 0.13f);
    BnLayerWeights layer = {0};
    layer.moe.router_weight = router;
    layer.moe.expert_map = (BnMoEExpertMap){
        .gate_offset = 0, .up_offset = experts * projection_bytes,
        .down_offset = experts * 2 * projection_bytes,
        .expert_gate_bytes = projection_bytes, .expert_up_bytes = projection_bytes,
        .expert_down_bytes = projection_bytes,
        .gate_stride = projection_bytes, .up_stride = projection_bytes,
        .down_stride = projection_bytes,
        .gate_type = BN_GGUF_TENSOR_Q4_K, .up_type = BN_GGUF_TENSOR_Q4_K,
        .down_type = BN_GGUF_TENSOR_Q4_K,
        .gate_rows = dim, .gate_cols = dim, .up_rows = dim, .up_cols = dim,
        .down_rows = dim, .down_cols = dim
    };
    BnModelIO io = {0};
    io.moe_io.mmap_base = (const uint8_t *)blocks;
    BnModelBackendState backend_state = {bn_backend_model_create()};
    assert(backend_state.backend);
    BnModel model = {0};
    model.io = &io;
    model.backend_state = &backend_state;
    model.config.dim = dim;
    model.config.n_experts = experts;
    model.config.n_experts_active = experts;
    model.config.moe_intermediate_size = dim;
    model.config.moe_norm_topk_prob = 1;
    BnMoEState moe_state = {0};
    BnSession session = {0};
    session.moe_state = &moe_state;
    BnQWeight weight = {blocks, BN_GGUF_TENSOR_Q4_K, dim, dim, 1.0f};
    size_t bytes = bn_quant_prepared_qweight_size(&weight, NULL);
    assert(bytes > 0);
    memcpy(reference, input, sizeof(input));
    assert(bn_moe_forward_batch_observed(&model, &session, &layer, 0,
        reference, scratch, tokens, BN_MOE_BATCH_INPUT_PRENORMALIZED,
        NULL, NULL) == 0);
    assert(memcmp(reference, input, sizeof(input)) != 0);
    /* Compare cold raw/local fallback and warm reuse, including a one-entry
     * cache too small to retain all expert projections. */
    for (int capacity = 1; capacity <= projections; capacity += projections - 1) {
        bn_backend_model_set_cpu_prepared_cache_budget(backend_state.backend,
                                                       bytes * capacity);
        for (int repeat = 0; repeat < 2; repeat++) {
            if (repeat == 1) for (int p = 0; p < projections; p++) {
                BnQWeight cached_weight = weight;
                cached_weight.data = blocks + p * dim;
                const BnPreparedWeight *prepared =
                    bn_backend_model_acquire_cpu_prepared(
                        backend_state.backend, &cached_weight);
                assert(prepared);
                bn_backend_model_release_cpu_prepared(backend_state.backend, prepared);
            }
            memcpy(actual, input, sizeof(input));
            assert(bn_moe_forward_batch_observed(&model, &session, &layer, 0,
                actual, scratch, tokens, BN_MOE_BATCH_INPUT_PRENORMALIZED,
                NULL, NULL) == 0);
            assert(memcmp(actual, reference, sizeof(actual)) == 0);
        }
        /* All batch acquisitions must have been released: shrinking to zero
         * must leave enough room to acquire an unrelated projection. */
        bn_backend_model_set_cpu_prepared_cache_budget(backend_state.backend, 0);
        bn_backend_model_set_cpu_prepared_cache_budget(backend_state.backend, bytes);
        BnQWeight unrelated = weight;
        unrelated.data = blocks + projections * dim;
        const BnPreparedWeight *p =
            bn_backend_model_acquire_cpu_prepared(backend_state.backend, &unrelated);
        assert(p);
        bn_backend_model_release_cpu_prepared(backend_state.backend, p);
    }
    /* Disabled preparation must bypass warm cached layouts too. */
    const char *prior_env = getenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS");
    char *saved_env = prior_env ? malloc(strlen(prior_env) + 1) : NULL;
    if (prior_env) { assert(saved_env); strcpy(saved_env, prior_env); }
    assert(setenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS", "1", 1) == 0);
    BnThreadPool *raw_pool = bn_tp_create(1);
    assert(raw_pool && !bn_tp_cpu_policy(raw_pool)->prepared_qweights);
    if (saved_env) {
        assert(setenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS", saved_env, 1) == 0);
        free(saved_env);
    } else {
        assert(unsetenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS") == 0);
    }
    BnModelRuntime raw_runtime = {.pool = raw_pool};
    model.runtime = &raw_runtime;
    bn_backend_model_set_cpu_prepared_cache_budget(backend_state.backend, 0);
    bn_backend_model_set_cpu_prepared_cache_budget(backend_state.backend,
                                                   bytes * projections);
    for (int p = 0; p < projections; p++) {
        BnQWeight warm = weight;
        warm.data = blocks + p * dim;
        const BnPreparedWeight *prepared =
            bn_backend_model_acquire_cpu_prepared(backend_state.backend, &warm);
        assert(prepared);
        bn_backend_model_release_cpu_prepared(backend_state.backend, prepared);
    }
    PreparedDisabledObserve check = {
        backend_state.backend, weight, bytes * projections, projections, 0
    };
    memcpy(actual, input, sizeof(input));
    assert(bn_moe_forward_batch_observed(&model, &session, &layer, 0,
        actual, scratch, tokens, BN_MOE_BATCH_INPUT_PRENORMALIZED,
        observe_disabled_preparation, &check) == 0);
    assert(check.checked);
    for (int p = 0; p < projections; p++) {
        BnQWeight uncached = weight;
        uncached.data = blocks + p * dim;
        assert(bn_backend_model_acquire_cached_cpu_prepared(
            backend_state.backend, &uncached) == NULL);
    }
    for (int i = 0; i < tokens * dim; i++)
        assert(fabsf(actual[i] - reference[i]) < 1e-4f);
    model.runtime = NULL;
    bn_tp_free(raw_pool);

    io.moe_io.mmap_base = NULL;
    assert(bn_moe_acquire_prepared_projection(&model, &weight) == NULL);
    const uint8_t *shard = (const uint8_t *)blocks;
    io.moe_io.mmap_bases = &shard;
    io.moe_io.n_mmap_bases = 1;
    const BnPreparedWeight *prime =
        bn_backend_model_acquire_cpu_prepared(backend_state.backend, &weight);
    assert(prime);
    bn_backend_model_release_cpu_prepared(backend_state.backend, prime);
    const BnPreparedWeight *sharded =
        bn_moe_acquire_prepared_projection(&model, &weight);
    assert(sharded);
    bn_moe_release_prepared_projection(&model, sharded);
    bn_backend_model_free(backend_state.backend);
    free(blocks);
    printf("PASSED\n");
#else
    (void)tokens;
#endif
}

static void test_moe_prefill_prepared_cache(void) {
    test_moe_prefill_prepared_cache_tokens(9);
    test_moe_prefill_prepared_cache_tokens(127);
    test_moe_prefill_prepared_cache_tokens(128);
}

static void test_backend_cpu_prepared_cache_pinning(void) {
    printf("test_backend_cpu_prepared_cache_pinning... ");
#if defined(__AVX2__)
    BnBlockQ4K blocks_a[8] = {0};
    BnBlockQ4K blocks_b[8] = {0};
    BnQWeight weight_a = {
        blocks_a, BN_GGUF_TENSOR_Q4_K, 8, BN_QK_K, 1.0f
    };
    BnQWeight weight_b = {
        blocks_b, BN_GGUF_TENSOR_Q4_K, 8, BN_QK_K, 1.0f
    };
    size_t bytes = bn_quant_prepared_qweight_size(&weight_a, NULL);
    assert(bytes > 0);
    BnBackendModel *backend = bn_backend_model_create();
    assert(backend != NULL);
    bn_backend_model_set_cpu_prepared_cache_budget(backend, bytes);

    assert(bn_backend_model_acquire_cached_cpu_prepared(backend, &weight_a) == NULL);

    const BnPreparedWeight *a =
        bn_backend_model_acquire_cpu_prepared(backend, &weight_a);
    assert(a != NULL && a->aux != NULL);
    const BnPreparedWeight *cached_a =
        bn_backend_model_acquire_cached_cpu_prepared(backend, &weight_a);
    assert(cached_a == a);
    assert(bn_backend_model_acquire_cached_cpu_prepared(backend, &weight_b) == NULL);
    bn_backend_model_release_cpu_prepared(backend, cached_a);
    assert(bn_backend_model_acquire_cpu_prepared(backend, &weight_b) == NULL);
    bn_backend_model_release_cpu_prepared(backend, a);

    const BnPreparedWeight *b =
        bn_backend_model_acquire_cpu_prepared(backend, &weight_b);
    assert(b != NULL && b->aux != NULL);
    bn_backend_model_release_cpu_prepared(backend, b);
    bn_backend_model_free(backend);
#endif
    printf("PASSED\n");
}


/* A scalar F32 oracle checks the full two-branch graph, including nontrivial
 * router and expert scales. Three tokens exercise grouping across experts. */
static void moe_test_norm(float *out, const float *x, const float *w, int n,
                          float eps) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float r = 1.0f / sqrtf(sum / n + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] * r) * (w ? w[i] : 1.0f);
}

static void moe_test_ffn(float *out, const float *x, const float *gate,
                         const float *up, const float *down, int n) {
    float h[32];
    assert(n == 32);
    for (int i = 0; i < n; i++) {
        float g = 0.0f, u = 0.0f;
        for (int j = 0; j < n; j++) {
            g += gate[i * n + j] * x[j];
            u += up[i * n + j] * x[j];
        }
        h[i] = 0.5f * g * (1.0f + tanhf(0.7978845608028654f * g *
                         (1.0f + 0.044715f * g * g))) * u;
    }
    for (int i = 0; i < n; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < n; j++) out[i] += down[i * n + j] * h[j];
    }
}

typedef struct {
    float logits[3][3];
    float weights[3][2];
    int indices[3][2];
    const float *scales;
    int seen;
    int gpu_calls;
    int reduction_calls;
    int composition_calls;
    int composition_fail_at;
} MoEBatchRouterCheck;

static void moe_test_batch_router_observe(void *ctx, BnMoEObservePoint point,
                                          int token, int slot, int expert,
                                          const float *values, int n) {
    (void)slot;
    (void)expert;
    if (point != BN_MOE_OBSERVE_ROUTER_LOGITS) return;
    MoEBatchRouterCheck *check = ctx;
    assert(token >= 0 && token < 3 && n == 3);
    assert(memcmp(values, check->logits[token], (size_t)n * sizeof(float)) == 0);
    check->seen++;
}

static int moe_test_reject_gpu_batch(void *ctx, float *out,
    void *gate, void *up, void *down, const int *indices, const float *weights,
    const float *output_scales, const float *x,
    int tokens, int dim, int hidden, int experts, int k,
    int gate_type, int up_type, int down_type, int activation) {
    (void)out; (void)gate; (void)up; (void)down; (void)x;
    (void)gate_type; (void)up_type; (void)down_type; (void)activation;
    MoEBatchRouterCheck *check = ctx;
    assert(dim == 32 && hidden == 32 && experts == 3 && k == 2);
    for (int t = 0; t < tokens; t++)
        for (int slot = 0; slot < k; slot++) {
            int expert = check->indices[t][slot];
            assert(indices[t * k + slot] == expert);
            assert(weights[t * k + slot] == check->weights[t][slot]);
            assert(output_scales[t * k + slot] == check->scales[expert]);
        }
    check->gpu_calls++;
    // The CPU fallback must still receive the original route coefficients.
    return -1;
}

static int moe_test_reject_gpu_reduction(void *ctx, float *out,
    const float *raw, const float *weights, const float *scales,
    int nt, int k, int dim) {
    MoEBatchRouterCheck *check = ctx;
    assert(out && raw && k == 2 && dim == 32);
    for (int t = 0; t < nt; t++) for (int slot = 0; slot < k; slot++) {
        assert(weights[t*k+slot] == check->weights[t][slot]);
        assert(scales[t*k+slot] == check->scales[check->indices[t][slot]]);
    }
    check->reduction_calls++;
    return -1;
}

/* Deliberately write scratch even when rejecting. The runtime must retain
 * both original branches until every composition operation succeeds. */
static int moe_test_composition_step(void *ctx, float *out, int n, float value) {
    MoEBatchRouterCheck *check = ctx;
    check->composition_calls++;
    for (int i = 0; i < n; i++) out[i] = value;
    return check->composition_calls == check->composition_fail_at ? -1 : 0;
}

static int moe_test_composition_norm(void *ctx, float *out, void *norm,
    const float *input, int tokens, int dim, float eps) {
    assert(norm && input && eps > 0.0f);
    return moe_test_composition_step(ctx, out, tokens * dim, 0.625f);
}

static int moe_test_composition_residual(void *ctx, float *out, void *norm,
    const float *input, const float *residual, int tokens, int dim, float eps) {
    assert(norm && input && residual && eps > 0.0f);
    return moe_test_composition_step(ctx, out, tokens * dim, 0.75f);
}

static int moe_test_composition_ffn(void *ctx, float *out, void *gate,
    void *up, void *down, const float *input, int tokens, int dim, int hidden,
    int gate_type, int up_type, int down_type, int activation) {
    assert(gate && up && down && input && hidden == dim);
    assert(gate_type == BN_GGUF_TENSOR_F32 && up_type == gate_type &&
           down_type == gate_type);
    (void)activation;
    return moe_test_composition_step(ctx, out, tokens * dim, 0.125f);
}

static void test_moe_prefill_dense_residual(void) {
    printf("test_moe_prefill_dense_residual... ");
    enum { dim = 32, experts = 3, active = 2, tokens = 3, matrix = dim * dim };
    float *weights = malloc(3 * experts * matrix * sizeof(float));
    float *dense = malloc(3 * matrix * sizeof(float));
    assert(weights && dense);
    for (int i = 0; i < 3 * experts * matrix; i++)
        weights[i] = 0.045f * sinf((float)i * 0.31f);
    for (int i = 0; i < 3 * matrix; i++)
        dense[i] = 0.035f * cosf((float)i * 0.17f);
    float input[tokens * dim], actual[tokens * dim], scratch[tokens * dim];
    float router[experts * dim], scale[dim], norms[5][dim];
    float expert_scale[experts] = {0.3f, 1.7f, 0.8f};
    for (int i = 0; i < tokens * dim; i++)
        input[i] = sinf((float)i * 0.23f) + 0.3f * cosf((float)i * 0.37f);
    for (int i = 0; i < experts * dim; i++) router[i] = cosf((float)i * 0.29f);
    for (int d = 0; d < dim; d++) {
        scale[d] = 0.4f + (d % 7) * 0.3f;
        for (int k = 0; k < 5; k++)
            norms[k][d] = 0.3f + ((d * (k + 1) + k) % 11) * 0.11f;
    }
    BnLayerWeights layer = {0};
    layer.norm.ffn_norm = norms[0];
    layer.norm.ffn_sub_norm = norms[1];
    layer.moe.router_weight = router;
    layer.moe.router_scale = scale;
    layer.moe.expert_down_scale = expert_scale;
    layer.ffn.ffn_gate = bn_quant_f32_weight(dense, dim, dim);
    layer.ffn.ffn_up = bn_quant_f32_weight(dense + matrix, dim, dim);
    layer.ffn.ffn_down = bn_quant_f32_weight(dense + 2 * matrix, dim, dim);
    size_t bytes = matrix * sizeof(float);
    layer.moe.expert_map = (BnMoEExpertMap){
        .gate_offset = 0, .up_offset = experts * bytes,
        .down_offset = 2 * experts * bytes,
        .expert_gate_bytes = bytes, .expert_up_bytes = bytes, .expert_down_bytes = bytes,
        .gate_stride = bytes, .up_stride = bytes, .down_stride = bytes,
        .gate_type = BN_GGUF_TENSOR_F32, .up_type = BN_GGUF_TENSOR_F32,
        .down_type = BN_GGUF_TENSOR_F32,
        .gate_rows = dim, .gate_cols = dim, .up_rows = dim, .up_cols = dim,
        .down_rows = dim, .down_cols = dim
    };
    BnModelIO io = {0};
    io.moe_io.mmap_base = (const uint8_t *)weights;
    BnModel model = {0};
    model.io = &io;
    model.config.dim = dim;
    model.config.hidden_dim = dim;
    model.config.n_experts = experts;
    model.config.n_experts_active = active;
    model.config.moe_intermediate_size = dim;
    model.config.moe_norm_topk_prob = 1;
    model.config.norm_eps = 1e-5f;
    model.config.policy_flags = BN_MODEL_ARCH_POLICY_MOE_DENSE_RESIDUAL_BRANCH |
                                BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT |
                                BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION |
                                BN_MODEL_ARCH_POLICY_MOE_PREFILL_MATVEC_ROUTER;
    BnMoEState state = {0};
    BnSession session = {0};
    session.moe_state = &state;
    MoEBatchRouterCheck check = {.scales = expert_scale};
    BnGPUBackend gpu = {.kind = BN_GPU_BACKEND_CUDA, .ctx = &check,
                       .moe_routed_ffn_batch = moe_test_reject_gpu_batch,
                       .moe_reduce_batch = moe_test_reject_gpu_reduction};
    BnModelBackendState backend_state = {.backend = bn_backend_model_create()};
    assert(backend_state.backend);
    model.backend_state = &backend_state;
    bn_backend_model_bind_gpu(backend_state.backend, &gpu);
    assert(bn_backend_model_register_handle(backend_state.backend, 0,
        BN_BACKEND_HANDLE_MOE_GATE_ALL, weights) == 0);
    assert(bn_backend_model_register_handle(backend_state.backend, 0,
        BN_BACKEND_HANDLE_MOE_UP_ALL, weights + experts * matrix) == 0);
    assert(bn_backend_model_register_handle(backend_state.backend, 0,
        BN_BACKEND_HANDLE_MOE_DOWN_ALL, weights + 2 * experts * matrix) == 0);
    for (int t = 0; t < tokens; t++) {
        float router_input[dim];
        bn_moe_scaled_router_input(router_input, input + t * dim, scale, dim, 1e-5f);
        bn_moe_route_buffers(check.logits[t], check.indices[t], check.weights[t],
            router_input, router, dim, experts, active, 1, 0.0f, 1, NULL);
    }
    assert(bn_backend_model_register_qweight(backend_state.backend,
        &layer.ffn.ffn_gate, dense) == 0);
    assert(bn_backend_model_register_qweight(backend_state.backend,
        &layer.ffn.ffn_up, dense + matrix) == 0);
    assert(bn_backend_model_register_qweight(backend_state.backend,
        &layer.ffn.ffn_down, dense + 2 * matrix) == 0);
    const BnBackendHandleRole norm_roles[4] = {
        BN_BACKEND_HANDLE_FFN_NORM, BN_BACKEND_HANDLE_FFN_POST_NORM_1,
        BN_BACKEND_HANDLE_FFN_POST_NORM_2, BN_BACKEND_HANDLE_FFN_POST_NORM
    };
    for (int i = 0; i < 4; i++)
        assert(bn_backend_model_register_handle(backend_state.backend, 0,
            norm_roles[i], norms[i == 0 ? 0 : i + 1]) == 0);
    float cpu_results[2][2][2][tokens*dim];
    for (int mode = 0; mode < 8; mode++) {
    int gpu_enabled = mode != 0;
    gpu.rmsnorm_batch = mode >= 2 ? moe_test_composition_norm : NULL;
    gpu.rmsnorm_residual_batch = mode >= 2 ? moe_test_composition_residual : NULL;
    gpu.dense_ffn_batch = mode >= 2 ? moe_test_composition_ffn : NULL;
    check.composition_fail_at = mode >= 2 && mode <= 6 ? mode - 1 : 0;
    bn_backend_model_set_gpu_disabled(backend_state.backend, !gpu_enabled);
    for (int count = 1; count <= tokens; count += tokens - 1) {
    for (int normalized = 0; normalized < 2; normalized++) {
        layer.norm.ffn_post_norm_1 = normalized ? norms[2] : NULL;
        layer.norm.ffn_post_norm_2 = normalized ? norms[3] : NULL;
        layer.norm.ffn_post_norm = normalized ? norms[4] : NULL;
        for (int raw = 0; raw < 2; raw++) {
            memcpy(actual, input, sizeof(input));
            check.seen = 0;
            check.gpu_calls = 0;
            check.reduction_calls = 0;
            check.composition_calls = 0;
            assert(bn_moe_forward_batch_observed(&model, &session, &layer, 0,
                actual, scratch, count, raw ? BN_MOE_BATCH_OUTPUT_RAW : 0,
                moe_test_batch_router_observe, &check) == 0);
            assert(check.seen == count);
            assert(check.gpu_calls == gpu_enabled);
            assert(check.reduction_calls == gpu_enabled);
            assert(check.composition_calls == (normalized && mode >= 2
                ? (mode == 7 ? 5 : mode - 1) : 0));
            if (normalized && mode == 7) {
                for (int i = 0; i < count * dim; i++)
                    assert(actual[i] == (raw ? 0.625f : 0.75f));
                continue;
            }
            float *cpu_result = cpu_results[count == 1 ? 0 : 1][normalized][raw];
            if (!gpu_enabled) memcpy(cpu_result, actual, (size_t)count*dim*sizeof(float));
            else assert(memcmp(cpu_result, actual, (size_t)count*dim*sizeof(float)) == 0);
            for (int t = 0; t < count; t++) {
                const float *x = input + t * dim;
                float router_x[dim], routed_x[dim], dense_x[dim];
                float logits[experts], probabilities[experts], output[dim] = {0};
                moe_test_norm(router_x, x, NULL, dim, 1e-5f);
                moe_test_norm(routed_x, x, norms[1], dim, 1e-5f);
                moe_test_norm(dense_x, x, norms[0], dim, 1e-5f);
                for (int d = 0; d < dim; d++) router_x[d] *= scale[d] / sqrtf(dim);
                for (int e = 0; e < experts; e++) {
                    logits[e] = 0.0f;
                    for (int d = 0; d < dim; d++) logits[e] += router[e * dim + d] * router_x[d];
                    probabilities[e] = expf(logits[e]);
                }
                int chosen[active];
                for (int k = 0; k < active; k++) {
                    int best = 0;
                    for (int e = 1; e < experts; e++) if (logits[e] > logits[best]) best = e;
                    chosen[k] = best;
                    logits[best] = -INFINITY;
                }
                float denominator = probabilities[chosen[0]] + probabilities[chosen[1]];
                for (int k = 0; k < active; k++) {
                    int e = chosen[k];
                    float expert[dim];
                    moe_test_ffn(expert, routed_x, weights + e * matrix,
                        weights + (experts + e) * matrix,
                        weights + (2 * experts + e) * matrix, dim);
                    float w = probabilities[e] / denominator * expert_scale[e];
                    for (int d = 0; d < dim; d++) output[d] += w * expert[d];
                }
                if (normalized) moe_test_norm(output, output, norms[3], dim, 1e-5f);
                float shared[dim];
                moe_test_ffn(shared, dense_x, dense, dense + matrix, dense + 2 * matrix, dim);
                if (normalized) moe_test_norm(shared, shared, norms[2], dim, 1e-5f);
                for (int d = 0; d < dim; d++) output[d] += shared[d];
                if (normalized) moe_test_norm(output, output, norms[4], dim, 1e-5f);
                for (int d = 0; d < dim; d++) {
                    float expected = output[d] + (raw ? 0.0f : x[d]);
                    assert(fabsf(actual[t * dim + d] - expected) < 3e-4f);
                }
            }
        }
    }
    }
    }
    bn_backend_model_free(backend_state.backend);
    free(weights);
    free(dense);
    printf("PASSED\n");
}

int main(void) {
    printf("=== MoE Unit Tests ===\n");
    test_moe_prefill_dense_residual();
    test_moe_route();
    test_moe_reference_router_accumulation();
    test_moe_projection_buffer_layout();
    test_expert_map_split();
    test_expert_map_fused_gate_up();
    test_moe_config_compat();
    test_model_arch_gguf_uses_moe();
    test_qwen2moe_arch_config();
    test_moe_execution_policy();
    test_moe_prefill_policy();
    test_moe_quant_policy_helpers();
    test_moe_resident_routed_ffn_layout_policy();
    test_swiglu();
    test_reference_gelu_limits();
    test_reference_gelu_table_entry();
    test_scaled_router_input();
    test_expert_scale_order();
    test_route_uniform();
    test_router_softmax_backend();
    test_router_batch_logits();
    test_router_native_dot4();
    test_moe_native_single_dot();
    test_moe_cache();
    test_backend_cpu_prepared_cache_pinning();
    test_moe_prefill_prepared_cache();
    printf("All MoE tests passed!\n");
    return 0;
}
