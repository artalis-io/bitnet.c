#ifndef BN_GLM_ANATOMY_CHECKS_H
#define BN_GLM_ANATOMY_CHECKS_H
#include "glm_anatomy_reference.h"
#include "../src/transformer/bf16_internal.h"

static void *glm_topology_legacy_buffer(void *ctx, const void *data, size_t size,
                                        int type, int rows, int cols) {
    (void)ctx; (void)data; (void)size; (void)type; (void)rows; (void)cols;
    assert(!"topology binding must reject legacy GPU upload before allocation");
    return NULL;
}

static void expect_glm_anatomy_rejected(BnGGUFFile *f) {
    BnModelAnatomy *out = malloc(sizeof(*out)), *before = malloc(sizeof(*before));
    assert(out && before);
    memset(out, 0x5a, sizeof(*out)); memcpy(before, out, sizeof(*out));
    assert(bn_model_arch_plan_anatomy(f, out) == -1);
    assert(!memcmp(out, before, sizeof(*out)));
    free(out); free(before);
}

static void test_glm_model_anatomy(void) {
    printf("test_glm_model_anatomy... ");
    BnGGUFFile f = {0};
    f.kvs = glm_anatomy_reference_kvs;
    f.n_kv = sizeof(glm_anatomy_reference_kvs) / sizeof(glm_anatomy_reference_kvs[0]);
    f.tensors = glm_anatomy_reference_tensors;
    f.n_tensors = sizeof(glm_anatomy_reference_tensors) / sizeof(glm_anatomy_reference_tensors[0]);
    const BnModelArchOps *ops = bn_model_arch_ops_for("glm5next");
    assert(ops && !strcmp(ops->name, "glm5next") && ops->plan_anatomy);
    assert(!ops->is_ssm_layer && ops->bf16_layout == BN_MODEL_BF16_LAYOUT_GLM_MHC);
    assert(ops->ssm_qk_norm == BN_SSM_QK_NORM_L2_SUM_EPS);
    assert(bn_model_gguf_bf16_layout(NULL) == BN_MODEL_BF16_LAYOUT_NONE);
    assert(bn_model_gguf_bf16_layout(&f) == BN_MODEL_BF16_LAYOUT_GLM_MHC);
    assert(!bn_model_arch_ops_for("qwen4exp")->plan_anatomy);
    char name[128];
    assert(bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0, BN_MODEL_TENSOR_SSM_QKV) == -1);
    assert(bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0, BN_MODEL_TENSOR_HC_ATTN_NORM) == -1);
    assert(bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0, BN_MODEL_TENSOR_INDEXER_Q_NORM) == -1);
    assert(bn_model_arch_tensor_name_for(ops, name, 2, 0, BN_MODEL_TENSOR_SSM_F_A) == -1);
    BnModelAnatomy *p = malloc(sizeof(*p)); assert(p);
    assert(bn_model_arch_plan_anatomy(&f, p) == 0);
    assert(p->stored_layers == 46 && p->n_layers == 45 && p->nextn_layers == 1);
    assert(p->dim == 4096 && p->vocab_size == 154880 && p->context_length == 1048576);
    assert(p->attention_heads == 64 && p->mla_key_dim == 256 && p->mla_value_dim == 256);
    assert(p->q_rank == 1536 && p->kv_rank == 512 && p->rope_dim == 0);
    assert(p->kda_heads == 64 && p->kda_head_dim == 128 && p->kda_gate_rank == 128);
    assert(p->conv_kernel == 4 && p->kda_parameter_is_negative_exp);
    assert(p->kda_lower_bound == -5 && p->kda_qk_eps == 1e-6f && p->kda_output_eps == 1e-5f);
    assert(p->mhc_streams == 4 && p->mhc_iterations == 20 && p->mhc_eps == 1e-6f);
    assert(p->indexer_heads == 32 && p->indexer_dim == 128 && p->indexer_top_k == 2048 && p->indexer_pool == 4);
    assert(p->dense_hidden == 12288 && p->leading_dense_layers == 3);
    assert(p->experts == 288 && p->active_experts == 8 && p->expert_hidden == 2048);
    assert(p->shared_experts == 1 && p->shared_hidden == 2048);
    assert(p->normalize_expert_weights && p->expert_weights_scale == 2.5f);
    unsigned char *used = calloc((size_t)f.n_tensors, 1); assert(used);
    used[p->embedding_tensor]++; used[p->output_tensor]++; used[p->output_norm_tensor]++;
    int kda = 0, mla = 0, bindings = 3;
    for (int l = 0; l < p->n_layers; l++) {
        const BnModelLayerAnatomy *a = &p->layers[l];
        assert(a->uses_moe == (l >= 3));
        assert(a->swiglu_clamp_exp == 10 && a->swiglu_clamp_shexp == 10);
        if ((l+1)%4) { assert(a->kind == BN_MODEL_ANATOMY_KDA); kda++; }
        else { assert(a->kind == BN_MODEL_ANATOMY_INDEXED_MLA); mla++; }
        for (int r = 0; r < BN_MODEL_TENSOR_ROLE_COUNT; r++) if (a->tensor[r] >= 0) {
            int index = a->tensor[r]; assert((uint64_t)index < f.n_tensors && !used[index]);
            used[index]++; bindings++;
        }
        assert(a->tensor[BN_MODEL_TENSOR_HC_ATTN_NORM] == -1);
        assert(a->tensor[BN_MODEL_TENSOR_SSM_QKV] == -1);
    }
    assert(kda == 34 && mla == 11 && bindings == 1383);
    int auxiliary = 0;
    for (uint64_t i = 0; i < f.n_tensors; i++) if (!used[i]) {
        assert(!strncmp(f.tensors[i].name, "blk.45.", 7)); auxiliary++;
    }
    assert(auxiliary == 29 && p->layers[45].kind == BN_MODEL_ANATOMY_NONE);
    for (int r = 0; r < BN_MODEL_TENSOR_ROLE_COUNT; r++) assert(p->layers[45].tensor[r] == -1);
    free(used);
    static const struct { int layer; BnModelTensorRole role; const char *name; } roles[] = {
        {0, BN_MODEL_TENSOR_ATTN_Q, "blk.0.attn_q.weight"},
        {0, BN_MODEL_TENSOR_ATTN_K, "blk.0.attn_k.weight"},
        {0, BN_MODEL_TENSOR_ATTN_V, "blk.0.attn_v.weight"},
        {0, BN_MODEL_TENSOR_SSM_OUT, "blk.0.attn_output.weight"},
        {0, BN_MODEL_TENSOR_SSM_F_A, "blk.0.ssm_f_a.weight"},
        {0, BN_MODEL_TENSOR_SSM_G_A, "blk.0.ssm_g_a.weight"},
        {0, BN_MODEL_TENSOR_SSM_F_B, "blk.0.ssm_f_b.weight"},
        {0, BN_MODEL_TENSOR_SSM_G_B, "blk.0.ssm_g_b.weight"},
        {0, BN_MODEL_TENSOR_MHC_ATTN_FN, "blk.0.hc_attn_fn.weight"},
        {0, BN_MODEL_TENSOR_MHC_FFN_FN, "blk.0.hc_ffn_fn.weight"},
        {3, BN_MODEL_TENSOR_MLA_K_B, "blk.3.attn_k_b.weight"},
        {3, BN_MODEL_TENSOR_MLA_V_B, "blk.3.attn_v_b.weight"},
        {3, BN_MODEL_TENSOR_INDEXER_Q, "blk.3.indexer.attn_q_b.weight"},
        {3, BN_MODEL_TENSOR_INDEXER_HEAD_WEIGHTS, "blk.3.indexer.proj.weight"},
        {3, BN_MODEL_TENSOR_MOE_ROUTER_BIAS, "blk.3.exp_probs_b.bias"},
    };
    for (size_t i = 0; i < sizeof(roles)/sizeof(roles[0]); i++)
        assert(!strcmp(f.tensors[p->layers[roles[i].layer].tensor[roles[i].role]].name, roles[i].name));
    int exp_key = bn_gguf_find_key(&f, "glm5next.swiglu_clamp_exp");
    int shared_key = bn_gguf_find_key(&f, "glm5next.swiglu_clamp_shexp");
    float *ce = f.kvs[exp_key].value.arr.data, *cs = f.kvs[shared_key].value.arr.data;
    ce[44] = 7; cs[4] = 3.5f;
    assert(bn_model_arch_plan_anatomy(&f, p) == 0);
    assert(p->layers[44].swiglu_clamp_exp == 7 && p->layers[43].swiglu_clamp_exp == 10);
    assert(p->layers[4].swiglu_clamp_shexp == 3.5f && p->layers[3].swiglu_clamp_shexp == 10);
    ce[44] = cs[4] = 10;
    const char *bad_keys[] = {"block_count", "nextn_predict_layers", "attention.head_count",
        "hyper_connection.count", "attention.indexer.kpool", "expert_used_count",
        "attention.layer_norm_rms_epsilon", "expert_group_count"};
    for (size_t i = 0; i < sizeof(bad_keys)/sizeof(bad_keys[0]); i++) {
        snprintf(name, sizeof(name), "glm5next.%s", bad_keys[i]);
        int index = bn_gguf_find_key(&f, name); assert(index >= 0);
        BnGGUFKeyValue saved = f.kvs[index];
        if (saved.type == BN_GGUF_TYPE_FLOAT32) f.kvs[index].value.f32 = NAN;
        else f.kvs[index].value.u32 = UINT32_MAX;
        expect_glm_anatomy_rejected(&f); f.kvs[index] = saved;
    }
    int layout = bn_gguf_find_key(&f, "glm5next.attention.head_count_kv");
    BnGGUFKeyValue layout_saved = f.kvs[layout];
    f.kvs[layout].value.arr.n--; expect_glm_anatomy_rejected(&f); f.kvs[layout] = layout_saved;
    f.kvs[layout].value.arr.elem_type = BN_GGUF_TYPE_FLOAT32;
    expect_glm_anatomy_rejected(&f); f.kvs[layout] = layout_saved;
    uint32_t *kinds = f.kvs[layout].value.arr.data;
    kinds[44] = 1; expect_glm_anatomy_rejected(&f); kinds[44] = 0;
    ce[44] = INFINITY; expect_glm_anatomy_rejected(&f); ce[44] = 10;
    cs[45] = -1; expect_glm_anatomy_rejected(&f); cs[45] = 10;
    int late = bn_gguf_find_tensor(&f, "blk.44.ssm_g_b.weight"); assert(late >= 0);
    BnGGUFTensorInfo saved = f.tensors[late];
    f.tensors[late].dims[0]++; expect_glm_anatomy_rejected(&f); f.tensors[late] = saved;
    f.tensors[late].type = UINT32_MAX; expect_glm_anatomy_rejected(&f); f.tensors[late] = saved;
    f.tensors[late].name = "missing"; expect_glm_anatomy_rejected(&f); f.tensors[late] = saved;
    BnGGUFTensorInfo tail_saved = f.tensors[f.n_tensors-1];
    f.tensors[f.n_tensors-1] = saved; expect_glm_anatomy_rejected(&f);
    f.tensors[f.n_tensors-1] = tail_saved;
    int fw = bn_gguf_find_tensor(&f, "blk.3.indexer.proj.weight");
    uint32_t type = f.tensors[fw].type; f.tensors[fw].type = BN_GGUF_TENSOR_Q8_0;
    expect_glm_anatomy_rejected(&f); f.tensors[fw].type = type;
    assert(bn_model_arch_plan_anatomy(&f, p) == 0);
    assert(bn_model_arch_plan_anatomy(NULL, p) == -1);
    assert(bn_model_arch_plan_anatomy(&f, NULL) == -1);
    BnGGUFKeyValue *extra = malloc(((size_t)f.n_kv+1)*sizeof(*extra)); assert(extra);
    memcpy(extra, f.kvs, (size_t)f.n_kv*sizeof(*extra));
    BnGGUFFile duplicate = f; duplicate.kvs = extra; duplicate.n_kv++;
    extra[f.n_kv] = f.kvs[bn_gguf_find_key(&f, "glm5next.kda.gate_lower_bound")];
    expect_glm_anatomy_rejected(&duplicate);
    extra[f.n_kv] = f.kvs[bn_gguf_find_key(&f, "general.architecture")];
    expect_glm_anatomy_rejected(&duplicate); free(extra);
    BnModel model;
    assert(bn_model_load_with_cpu_preparation(&model, &f, 8, 0, 0, 0) == -1);
    assert(!model.weights.layers); bn_model_free(&model);
    assert(!bn_model_anatomy(NULL));
    size_t count = 99;
    assert(!bn_model_topology_experts(NULL, &count) && !count);
    assert(!bn_model_load_topology(&model, &f, 8));
    const BnModelAnatomy *owned = bn_model_anatomy(&model);
    assert(owned && owned != p && !memcmp(owned, p, sizeof(*p)));
    assert(bn_model_topology_file(&model) == &f);
    const BnMoEExpertMap *maps = bn_model_topology_experts(&model, &count);
    assert(maps && count == 45 && model.config.seq_len == 8);
    assert(model.config.dim == 4096 && model.config.vocab_size == 154880);
    assert(bn_model_uses_moe(&model) && model.config.n_experts == 288);
    assert(!model.weights.layers && !model.weights.token_embedding);
    assert(!model.config.n_heads && !model.config.kv_dim);
    for (int l = 0; l < 45; ++l) {
        if (l < 3) { assert(!maps[l].expert_gate_bytes); continue; }
        assert(maps[l].gate_rows == 2048 && maps[l].gate_cols == 4096);
        assert(maps[l].down_rows == 4096 && maps[l].down_cols == 2048);
        assert(maps[l].gate_type == (l == 11 ? BN_GGUF_TENSOR_Q5_K : BN_GGUF_TENSOR_Q4_K));
        assert(maps[l].down_type == (l == 11 || l == 12 || l == 44 ? BN_GGUF_TENSOR_Q6_K : BN_GGUF_TENSOR_Q5_K));
        assert(!maps[l].repacked);
    }
    BnGPUBackend legacy = {.buffer_create = glm_topology_legacy_buffer};
    assert(bn_model_upload_weights(&model, &legacy) == -1);
    assert(!bn_model_gpu(&model));
    assert(!bn_session_create(&model, NULL));
    BnSession *session = bn_session_create_backend(&model, NULL);
    assert(session && session->backend_only && session->state.logits);
    assert(bn_transformer_bf16_glm_prepare_context(NULL, session, 1, 8, 1, 1) == -1);
    assert(bn_transformer_bf16_glm_prepare_context(&model, NULL, 1, 8, 1, 1) == -1);
    assert(bn_transformer_bf16_glm_prepare_context(&model, session, 1, 8, 1, 1) == -1);
    assert(bn_transformer_bf16_position(bn_session_backend(session)) == -1);
    assert(!bn_transformer_bf16_max_tokens(bn_session_backend(session)));
    assert(!bn_transformer_bf16_capacity(bn_session_backend(session)));
    assert(!bn_transformer_forward(&model, session, 0, 0));
    bn_session_free(session, NULL);
    bn_model_free(&model);
    assert(!bn_model_anatomy(&model) && !bn_model_topology_file(&model));
    assert(!bn_model_topology_experts(&model, &count) && !count);
    /* Malformed input must not publish a partially bound model. */
    BnModel before; memset(&before, 0x5a, sizeof(before)); model = before;
    BnGGUFFile invalid = f; invalid.n_tensors = 1;
    assert(bn_model_load_topology(&model, &invalid, 8) == -1);
    assert(!memcmp(&model, &before, sizeof(model)));
    assert(bn_model_load_topology(&model, &f, -1) == -1);
    assert(!memcmp(&model, &before, sizeof(model)));
    free(p);
    printf("PASSED (1383 bindings, 34 KDA + 11 MLA, 29 auxiliary tensors excluded)\n");
}
#endif
