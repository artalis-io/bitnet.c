#include "gguf.h"
#include "model.h"
#include "model_internal.h"
#include "model_arch.h"
#include "session.h"
#include "transformer.h"
#include "transformer_prefill_internal.h"
#include "ple_ids_reference.h"
#if defined(BN_QWEN36_TEST_CUDA) && defined(BN_ENABLE_CUDA)
#include "gpu_cuda.h"
#endif
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t *data;
    size_t pos;
    size_t cap;
} WriteBuffer;

typedef struct {
    char name[96];
    uint32_t type;
    uint32_t n_dims;
    uint64_t dims[4];
    uint64_t offset;
    size_t bytes;
} TensorSpec;

typedef struct {
    TensorSpec tensors[256];
    int n_tensors;
    const char *arch;
    int moe;
} ModelSpec;

static void wb_write(WriteBuffer *wb, const void *data, size_t size) {
    assert(wb->pos + size <= wb->cap);
    memcpy(wb->data + wb->pos, data, size);
    wb->pos += size;
}

static void wb_zero(WriteBuffer *wb, size_t size) {
    assert(wb->pos + size <= wb->cap);
    memset(wb->data + wb->pos, 0, size);
    wb->pos += size;
}

static void wb_u32(WriteBuffer *wb, uint32_t v) { wb_write(wb, &v, sizeof(v)); }
static void wb_u64(WriteBuffer *wb, uint64_t v) { wb_write(wb, &v, sizeof(v)); }
static void wb_f32(WriteBuffer *wb, float v) { wb_write(wb, &v, sizeof(v)); }

static void wb_str(WriteBuffer *wb, const char *s) {
    uint64_t len = (uint64_t)strlen(s);
    wb_u64(wb, len);
    wb_write(wb, s, (size_t)len);
}

static size_t align32(size_t v) {
    return (v + 31u) & ~(size_t)31u;
}

static size_t tensor_bytes(uint32_t type, uint64_t n) {
    assert(type == BN_GGUF_TENSOR_F32);
    return (size_t)n * sizeof(float);
}

static void add_tensor(ModelSpec *s, const char *name, uint32_t type,
                       uint64_t d0, uint64_t d1, uint64_t d2) {
    assert(s->n_tensors < (int)(sizeof(s->tensors) / sizeof(s->tensors[0])));
    TensorSpec *t = &s->tensors[s->n_tensors++];
    memset(t, 0, sizeof(*t));
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->type = type;
    t->dims[0] = d0;
    t->n_dims = 1;
    uint64_t n = d0;
    if (d1 > 0) {
        t->dims[1] = d1;
        t->n_dims = 2;
        n *= d1;
    }
    if (d2 > 0) {
        t->dims[2] = d2;
        t->n_dims = 3;
        n *= d2;
    }
    t->bytes = tensor_bytes(type, n);
}

static void add_common_tensors(ModelSpec *s) {
    const int dim = 128;
    const int vocab = 8;
    const int layers = 4;
    const int hidden = s->moe ? 0 : 256;
    const int moe_hidden = 64;
    const int n_experts = 4;
    const int n_heads = 2;
    const int n_kv_heads = 1;
    const int head_size = 64;
    const int q_dim = n_heads * head_size;
    const int kv_dim = n_kv_heads * head_size;
    const int ssm_state = 16;
    const int ssm_groups = 1;
    const int ssm_inner = 64;
    const int ssm_rank = 1;
    const int ssm_qkv = ssm_groups * ssm_state * 2 + ssm_inner;
    char name[96];

    add_tensor(s, "token_embd.weight", BN_GGUF_TENSOR_F32, dim, vocab, 0);
    add_tensor(s, "output_norm.weight", BN_GGUF_TENSOR_F32, dim, 0, 0);

    for (int l = 0; l < layers; l++) {
        int is_ssm = ((l + 1) % 4) != 0;
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, 0, 0);

        if (is_ssm) {
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, ssm_qkv, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, ssm_inner, 0);
            snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, ssm_rank, 0, 0);
            snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, ssm_rank, 0);
            snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, ssm_rank, 0);
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, 4 * ssm_qkv, 0, 0);
            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, ssm_rank, 0, 0);
            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, ssm_inner / ssm_rank, 0, 0);
            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, ssm_inner, dim, 0);
        } else {
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, 2 * q_dim, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, kv_dim, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, kv_dim, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, q_dim, dim, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
            /* Qwen shares this norm across heads; its length is one head,
             * not the concatenated query width (nor the gated Q width). */
            add_tensor(s, name, BN_GGUF_TENSOR_F32, head_size, 0, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, kv_dim, 0, 0);
        }

        /* Exercise both GGUF spellings for the same FFN input role. */
        snprintf(name, sizeof(name), l % 2 ? "blk.%d.post_attention_norm.weight"
                                         : "blk.%d.ffn_norm.weight", l);
        add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, 0, 0);

        if (s->moe) {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, n_experts, 0);
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, moe_hidden, n_experts);
            snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, moe_hidden, n_experts);
            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, moe_hidden, dim, n_experts);
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, moe_hidden, 0);
            snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, moe_hidden, 0);
            snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, moe_hidden, dim, 0);
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, 0, 0);
        } else {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, hidden, 0);
            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, dim, hidden, 0);
            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, hidden, dim, 0);
        }
    }
}

static void write_model_kvs(WriteBuffer *wb, const ModelSpec *s) {
    char key[96];

    wb_str(wb, "general.architecture");
    wb_u32(wb, BN_GGUF_TYPE_STRING);
    wb_str(wb, s->arch);

    wb_str(wb, "tokenizer.ggml.tokens");
    wb_u32(wb, BN_GGUF_TYPE_ARRAY);
    wb_u32(wb, BN_GGUF_TYPE_STRING);
    wb_u64(wb, 8);
    for (int i = 0; i < 8; i++) {
        char tok[8];
        snprintf(tok, sizeof(tok), "t%d", i);
        wb_str(wb, tok);
    }

#define KV_U32(suffix, value) do { \
    snprintf(key, sizeof(key), "%s.%s", s->arch, suffix); \
    wb_str(wb, key); \
    wb_u32(wb, BN_GGUF_TYPE_UINT32); \
    wb_u32(wb, (uint32_t)(value)); \
} while (0)
#define KV_F32(suffix, value) do { \
    snprintf(key, sizeof(key), "%s.%s", s->arch, suffix); \
    wb_str(wb, key); \
    wb_u32(wb, BN_GGUF_TYPE_FLOAT32); \
    wb_f32(wb, (float)(value)); \
} while (0)

    KV_U32("embedding_length", 128);
    KV_U32("feed_forward_length", s->moe ? 0 : 256);
    KV_U32("block_count", 4);
    KV_U32("attention.head_count", 2);
    KV_U32("attention.head_count_kv", 1);
    KV_U32("context_length", 16);
    KV_F32("rope.freq_base", 10000000.0f);
    KV_F32("attention.layer_norm_rms_epsilon", 1e-6f);
    KV_U32("attention.key_length", 64);
    KV_U32("rope.dimension_count", 64);
    KV_U32("full_attention_interval", 4);
    KV_U32("ssm.state_size", 16);
    KV_U32("ssm.conv_kernel", 4);
    KV_U32("ssm.inner_size", 64);
    KV_U32("ssm.time_step_rank", 1);
    KV_U32("ssm.group_count", 1);
    if (s->moe) {
        KV_U32("expert_count", 4);
        KV_U32("expert_used_count", 2);
        KV_U32("expert_feed_forward_length", 64);
        KV_U32("expert_shared_feed_forward_length", 64);
    }

#undef KV_U32
#undef KV_F32
}

static BnGGUFFile *build_qwen36_gguf(uint8_t *buf, size_t cap,
                                      const char *arch, int moe) {
    ModelSpec spec = {0};
    spec.arch = arch;
    spec.moe = moe;
    add_common_tensors(&spec);

    int n_kv = moe ? 22 : 18;
    WriteBuffer wb = { buf, 0, cap };
    wb_u32(&wb, BN_GGUF_MAGIC);
    wb_u32(&wb, 3);
    wb_u64(&wb, (uint64_t)spec.n_tensors);
    wb_u64(&wb, (uint64_t)n_kv);

    write_model_kvs(&wb, &spec);

    size_t data_cursor = 0;
    for (int i = 0; i < spec.n_tensors; i++) {
        TensorSpec *t = &spec.tensors[i];
        data_cursor = align32(data_cursor);
        t->offset = (uint64_t)data_cursor;
        data_cursor += t->bytes;

        wb_str(&wb, t->name);
        wb_u32(&wb, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++)
            wb_u64(&wb, t->dims[d]);
        wb_u32(&wb, t->type);
        wb_u64(&wb, t->offset);
    }

    wb.pos = align32(wb.pos);
    size_t data_offset = wb.pos;
    wb_zero(&wb, data_cursor);

    for (int i = 0; i < spec.n_tensors; i++) {
        TensorSpec *t = &spec.tensors[i];
        float *dst = (float *)(buf + data_offset + (size_t)t->offset);
        size_t n = t->bytes / sizeof(float);
        int is_norm = strstr(t->name, "_norm.weight") != NULL ||
                      strcmp(t->name, "output_norm.weight") == 0;
        for (size_t j = 0; j < n; j++)
            dst[j] = is_norm ? 1.0f : 0.0f;
    }

    return bn_gguf_open(buf, wb.pos);
}

static void assert_forward_finite(BnModel *model) {
    assert(!bn_model_arch_uses_attention_post_norm(&model->config));
    for (int l = 0; l < model->config.n_layers; l++) {
        assert(model->weights.layers[l].norm.ffn_norm != NULL);
        assert(model->weights.layers[l].norm.attn_post_norm == NULL);
    }
    BnSession *session = bn_session_create(model, NULL);
    assert(session != NULL);
    float *logits = bn_transformer_forward(model, session, 0, 0);
    assert(logits != NULL);
    for (int i = 0; i < model->config.vocab_size; i++)
        assert(isfinite(logits[i]));
    bn_session_free(session, NULL);
}

#if defined(BN_QWEN36_TEST_CUDA) && defined(BN_ENABLE_CUDA)
static void assert_forward_finite_cuda(BnModel *model) {
    BnGPUBackend *gpu = bn_gpu_cuda_create();
    if (!gpu) {
        printf("SKIPPED_CUDA ");
        return;
    }
    assert(bn_model_upload_weights(model, gpu) == 0);
    assert(gpu->init_activations);
    assert(bn_model_init_gpu_activations(model, gpu) == 0);
    assert_forward_finite(model);
    bn_model_free(model);
    bn_gpu_cuda_destroy(gpu);
}
#endif

static void test_qwen36_dense(void) {
    printf("test_qwen36_dense... ");

    uint8_t *buf = (uint8_t *)calloc(1, 4 * 1024 * 1024);
    assert(buf != NULL);
    BnGGUFFile *gf = build_qwen36_gguf(buf, 4 * 1024 * 1024, "qwen35", 0);
    assert(gf != NULL);
    assert(strcmp(bn_model_arch_prefix("qwen35"), "qwen35") == 0);
    assert(bn_model_arch_activation("qwen35") == 0);
    assert(bn_model_arch_attention_value_shares_key("qwen35") == 0);

    BnModel model;
    assert(bn_model_load(&model, gf, 8, 0, 0) == 0);
    assert(bn_model_backend(&model) != NULL);
    assert(bn_model_gpu(&model) == NULL);
    assert(model.config.n_experts == 0);
    assert(model.config.full_attn_interval == 4);
    assert(model.config.n_layers == 4);
    assert(model.config.seq_len == 8);
    assert(model.config.head_size == 64);
    assert(model.config.rope_dim_count == 64);
    assert(bn_model_arch_separates_rope_norm(&model.config));
    assert(model.weights.layers[0].block_kind == BN_LAYER_BLOCK_SSM);
    assert(model.weights.layers[3].block_kind == BN_LAYER_BLOCK_ATTENTION);
    assert(model.weights.layers[3].ffn_kind == BN_LAYER_FFN_DENSE);
    assert(model.weights.layers[0].ssm.wqkv.data != NULL);
    assert(model.weights.layers[3].attn.wq.rows == 256);
    assert(model.weights.layers[3].ffn.ffn_gate.data != NULL);
#if defined(BN_QWEN36_TEST_CUDA) && defined(BN_ENABLE_CUDA)
    assert_forward_finite_cuda(&model);
#else
    assert_forward_finite(&model);
    bn_model_free(&model);
#endif

    bn_gguf_free(gf);
    free(buf);
    printf("PASSED\n");
}

static void test_qwen36_moe(void) {
    printf("test_qwen36_moe... ");

    uint8_t *buf = (uint8_t *)calloc(1, 4 * 1024 * 1024);
    assert(buf != NULL);
    BnGGUFFile *gf = build_qwen36_gguf(buf, 4 * 1024 * 1024, "qwen35moe", 1);
    assert(gf != NULL);
    const BnModelArchOps *ops = bn_model_arch_ops_for("qwen36");
    assert(ops);
    assert(bn_model_arch_infer_moe_hidden(gf, ops) == 64);
    assert(bn_model_arch_has_shared_expert(gf, ops) == 1);
    assert(bn_model_arch_infer_shared_expert_hidden(gf, ops) == 64);

    BnModel model;
    assert(bn_model_load(&model, gf, 8, 0, 0) == 0);
    assert(bn_model_backend(&model) != NULL);
    assert(bn_model_gpu(&model) == NULL);
    bn_model_set_moe_mmap_base(&model, gf->raw);
    assert(model.config.n_experts == 4);
    assert(model.config.n_experts_active == 2);
    assert(model.config.moe_intermediate_size == 64);
    assert(model.config.has_shared_expert == 1);
    assert(model.weights.layers[0].ffn_kind == BN_LAYER_FFN_MOE);
    assert(model.weights.layers[0].moe.expert_map.expert_gate_bytes == 128 * 64 * sizeof(float));
    assert(model.weights.layers[0].shared.shared_expert_gate != NULL);
    assert(model.weights.layers[3].attn.wq.rows == 256);
#if defined(BN_QWEN36_TEST_CUDA) && defined(BN_ENABLE_CUDA)
    assert_forward_finite_cuda(&model);
#else
    assert_forward_finite(&model);
    bn_model_free(&model);
#endif

    bn_gguf_free(gf);
    free(buf);
    printf("PASSED\n");
}

static void test_qwen36_explicit_dense(void) {
    printf("test_qwen36_explicit_dense... ");

    uint8_t *buf = (uint8_t *)calloc(1, 4 * 1024 * 1024);
    assert(buf != NULL);
    BnGGUFFile *gf = build_qwen36_gguf(buf, 4 * 1024 * 1024, "qwen36", 0);
    assert(gf != NULL);
    assert(strcmp(bn_model_arch_prefix("qwen36"), "qwen36") == 0);

    BnModel model;
    assert(bn_model_load(&model, gf, 8, 0, 0) == 0);
    assert(model.config.n_experts == 0);
    assert(model.weights.layers[0].block_kind == BN_LAYER_BLOCK_SSM);
    assert(model.weights.layers[3].block_kind == BN_LAYER_BLOCK_ATTENTION);
#if defined(BN_QWEN36_TEST_CUDA) && defined(BN_ENABLE_CUDA)
    assert_forward_finite_cuda(&model);
#else
    assert_forward_finite(&model);
    bn_model_free(&model);
#endif

    bn_gguf_free(gf);
    free(buf);
    printf("PASSED\n");
}

static void test_qwen36_explicit_moe(void) {
    printf("test_qwen36_explicit_moe... ");

    uint8_t *buf = (uint8_t *)calloc(1, 4 * 1024 * 1024);
    assert(buf != NULL);
    BnGGUFFile *gf = build_qwen36_gguf(buf, 4 * 1024 * 1024, "qwen36moe", 1);
    assert(gf != NULL);
    const BnModelArchOps *ops = bn_model_arch_ops_for("qwen36");
    assert(ops);
    assert(bn_model_arch_infer_moe_hidden(gf, ops) == 64);
    assert(bn_model_arch_has_shared_expert(gf, ops) == 1);

    BnModel model;
    assert(bn_model_load(&model, gf, 8, 0, 0) == 0);
    bn_model_set_moe_mmap_base(&model, gf->raw);
    assert(model.config.n_experts == 4);
    assert(model.weights.layers[0].ffn_kind == BN_LAYER_FFN_MOE);
#if defined(BN_QWEN36_TEST_CUDA) && defined(BN_ENABLE_CUDA)
    assert_forward_finite_cuda(&model);
#else
    assert_forward_finite(&model);
    bn_model_free(&model);
#endif

    bn_gguf_free(gf);
    free(buf);
    printf("PASSED\n");
}

static void test_prefill_output_rows_state_for_pool(int n_workers,
                                                    float results[4][16]) {
    enum { dim = 128, streams = 2, rank = 16, wide = dim * streams };
    enum { hc_size = wide + 2 * rank * wide + streams * wide };
    for (int use_hc = 0; use_hc < 2; use_hc++) {
        for (int final_ssm = 0; final_ssm < 2; final_ssm++) {
            uint8_t *buf = calloc(1, 4 * 1024 * 1024);
            assert(buf);
            BnGGUFFile *gf = build_qwen36_gguf(
                buf, 4 * 1024 * 1024, "qwen35", use_hc);
            assert(gf);
            for (uint64_t t = 0; t < gf->n_tensors; t++) {
                const BnGGUFTensorInfo *info = &gf->tensors[t];
                size_t count = 1;
                for (uint32_t d = 0; d < info->n_dims; d++) count *= info->dims[d];
                float *data = bn_gguf_tensor_data(gf, (int)t);
                assert(data);
                if (strstr(info->name, "norm")) continue;
                for (size_t j = 0; j < count; j++)
                    data[j] = 0.03f * sinf((float)(j + 17 * t) * 0.13f);
            }
            BnModel model;
            assert(bn_model_load(&model, gf, 8, 0, 0) == 0);
            /* Keep the synthetic gated-attention anatomy self-consistent.
             * A wrong shared norm length can otherwise make token/head
             * output regions overlap and mask the error in serial runs. */
            const BnLayerWeights *attn_layer = &model.weights.layers[3];
            assert(attn_layer->attn.head_size == 64);
            assert(attn_layer->attn.wo.cols == model.config.n_heads * 64);
            assert(attn_layer->attn.wq.rows == 2 * model.config.n_heads * 64);
            if (n_workers >= 0) {
                BnThreadPool *pool = bn_tp_create(n_workers);
                assert(pool);
                bn_model_set_thread_pool(&model, pool, 1);
            }
            bn_model_set_moe_mmap_base(&model, gf->raw);
            if (final_ssm) model.config.n_layers = 3;
            float *hc_data = NULL;
            float *ple_data = NULL;
            if (use_hc) {
                model.config.policy_flags |= BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
                if (final_ssm)
                    model.config.policy_flags |=
                        BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER;
                model.config.hyper_connection_count = streams;
                model.config.hyper_connection_rank = rank;
                hc_data = calloc(9 * hc_size, sizeof(float));
                assert(hc_data);
                for (int h = 0; h < 9; h++) {
                    BnHyperConnectionWeights *hc = h == 8 ? &model.weights.hc_output :
                        h % 2 ? &model.weights.layers[h / 2].hc_ffn :
                                &model.weights.layers[h / 2].hc_attn;
                    float *p = hc_data + h * hc_size;
                    hc->norm = p;
                    for (int i = 0; i < wide; i++) p[i] = 1;
                    p += wide;
                    hc->down = (BnQWeight){p, BN_GGUF_TENSOR_F32, rank, wide, 1};
                    p += rank * wide;
                    hc->up = (BnQWeight){p, BN_GGUF_TENSOR_F32, wide, rank, 1};
                    p += rank * wide;
                    hc->inject = (BnQWeight){p, BN_GGUF_TENSOR_F32, streams, wide, 1};
                    for (int i = wide; i < hc_size; i++)
                        hc_data[h * hc_size + i] = 0.01f * sinf((float)i * 0.17f);
                }
                /* Put recurrent PLE in the final layer to catch premature
                 * output-row selection before persistent history updates. */
                model.config.ple_layer = model.config.n_layers - 1;
                model.config.ple_head_count = 2;
                model.config.ple_head_dim = 64;
                model.config.ple_heads_per_ngram = 2;
                model.config.ple_ngram_size = 2;
                model.config.ple_conv_kernel = 3;
                model.config.ple_eos_token_id = 7;
                model.config.ple_layer_multipliers[0] = 17;
                model.config.ple_layer_multipliers[1] = 31;
                model.config.ple_head_vocab_sizes[0] = 4;
                model.config.ple_head_vocab_sizes[1] = 4;
                model.config.ple_head_offsets[1] = 4;
                enum { ple_size = 8 * 64 + wide * dim + dim * dim + 6 * wide };
                ple_data = calloc(ple_size, sizeof(float));
                assert(ple_data);
                for (int i = 0; i < ple_size; i++)
                    ple_data[i] = 0.02f * sinf((float)i * 0.11f);
                float *p = ple_data;
                model.weights.ple_token_embd =
                    (BnQWeight){p, BN_GGUF_TENSOR_F32, 8, 64, 1};
                p += 8 * 64;
                BnPositionalLayerEmbeddingWeights *ple =
                    &model.weights.layers[model.config.ple_layer].ple;
                ple->key = (BnQWeight){p, BN_GGUF_TENSOR_F32, wide, dim, 1};
                p += wide * dim;
                ple->value = (BnQWeight){p, BN_GGUF_TENSOR_F32, dim, dim, 1};
                p += dim * dim;
                for (int i = 0; i < 3 * wide; i++) p[i] = 1;
                ple->norm_key = p; p += wide;
                ple->norm_query = p; p += wide;
                ple->norm_conv = p; p += wide;
                ple->conv1d = p;
            }
            BnSession *sessions[3];
            int history[3][8] = {{0}};
            for (int j = 0; j < 3; j++) {
                sessions[j] = bn_session_create(&model, NULL);
                assert(sessions[j]);
                sessions[j]->state.token_history = history[j];
                sessions[j]->pos = 1; /* Transformer calls leave caller position alone. */
                assert(bn_transformer_forward(&model, sessions[j], 0, 0));
            }
            const int tokens[] = {1, 2, 3, 4};
            float all[4 * 8], last[8];
            for (int i = 0; i < 32; i++) all[i] = NAN;
            assert(bn_transformer_prefill_all(&model, sessions[0], tokens, 4, 1, all) == 0);
            float *logits = bn_transformer_prefill(&model, sessions[1], tokens, 4, 1);
            assert(logits);
            memcpy(last, logits, sizeof(last));
            assert(bn_transformer_prefill_no_logits(&model, sessions[2], tokens, 4, 1) == 0);
            for (int i = 0; i < 32; i++) assert(isfinite(all[i]));
            for (int i = 0; i < 8; i++) {
                assert(isfinite(last[i]));
                assert(fabsf(last[i] - all[24 + i]) < 1e-4f);
            }
            for (int j = 1; j < 3; j++) {
                const BnRunState *a = &sessions[0]->state, *b = &sessions[j]->state;
                assert(memcmp(a->ssm_state, b->ssm_state, 3 * 64 * 16 * sizeof(float)) == 0);
                assert(memcmp(a->ssm_conv_state, b->ssm_conv_state, 3 * 3 * 96 * sizeof(float)) == 0);
                if (use_hc)
                    assert(memcmp(a->ple_conv_state, b->ple_conv_state,
                                  4 * wide * sizeof(float)) == 0);
                if (!final_ssm) {
                    assert(memcmp(a->key_cache, b->key_cache, 8 * 64 * sizeof(float)) == 0);
                    assert(memcmp(a->value_cache, b->value_cache, 8 * 64 * sizeof(float)) == 0);
                }
                assert(memcmp(history[0], history[j], sizeof(history[0])) == 0);
            }
            for (int t = 0; t < 4; t++) assert(history[0][1 + t] == tokens[t]);
            float continuation[8];
            for (int j = 0; j < 3; j++) {
                assert(sessions[j]->pos == 1);
                logits = bn_transformer_forward(&model, sessions[j], 5, 5);
                assert(logits);
                if (j == 0) memcpy(continuation, logits, sizeof(continuation));
                else assert(memcmp(continuation, logits, sizeof(continuation)) == 0);
                sessions[j]->state.token_history = NULL;
                bn_session_free(sessions[j], NULL);
            }
            memcpy(results[use_hc * 2 + final_ssm], last, sizeof(last));
            memcpy(results[use_hc * 2 + final_ssm] + 8, continuation,
                   sizeof(continuation));
            bn_model_free(&model);
            free(hc_data);
            free(ple_data);
            bn_gguf_free(gf);
            free(buf);
        }
    }
}

static void test_prefill_output_rows_state(void) {
    printf("test_prefill_output_rows_state... ");
    float serial[4][16], pooled[4][16];
    test_prefill_output_rows_state_for_pool(-1, serial);
    const int worker_counts[] = {0, 3, 7};
    for (size_t p = 0; p < sizeof(worker_counts) / sizeof(worker_counts[0]); p++) {
        test_prefill_output_rows_state_for_pool(worker_counts[p], pooled);
        assert(memcmp(serial, pooled, sizeof(serial)) == 0);
    }
    printf("PASSED\n");
}

static void test_prefill_projection_failure(void) {
    printf("test_prefill_projection_failure... ");
    uint8_t *buf = calloc(1, 4*1024*1024); assert(buf);
    BnGGUFFile *gf = build_qwen36_gguf(buf, 4*1024*1024, "qwen35", 0); assert(gf);
    BnModel model;
    assert(bn_model_load(&model, gf, 8, 0, 0) == 0);
    // Force this test through batch prefill even on scalar-only builds.
    const char *old = getenv("BN_PREFILL_ALLOW_HYBRID_BATCH");
    char *saved = old ? malloc(strlen(old)+1) : NULL;
    if (old) { assert(saved); memcpy(saved, old, strlen(old)+1); }
    assert(setenv("BN_PREFILL_ALLOW_HYBRID_BATCH", "1", 1) == 0);
    BnThreadPool *pool = bn_tp_create(0); assert(pool);
    if (saved) { assert(setenv("BN_PREFILL_ALLOW_HYBRID_BATCH", saved, 1) == 0); free(saved); }
    else assert(unsetenv("BN_PREFILL_ALLOW_HYBRID_BATCH") == 0);
    bn_model_set_thread_pool(&model, pool, 1);
    const int tokens[] = {1,2,3,4};
    BnSession *control = bn_session_create(&model, NULL); assert(control);
    assert(bn_transformer_prefill(&model, control, tokens, 4, 0));
    bn_session_free(control, NULL);
    int layers = bn_model_arch_ssm_layer_count(&model.config);
    size_t ns = (size_t)layers * model.config.ssm_inner_size * model.config.ssm_state_size;
    size_t qkv = 2u * model.config.ssm_group_count * model.config.ssm_state_size + model.config.ssm_inner_size;
    size_t nc = (size_t)layers * (model.config.ssm_conv_kernel-1) * qkv;
    float *state = malloc(ns*sizeof(float)), *conv = malloc(nc*sizeof(float)); assert(state && conv);
    int saved_type = model.weights.layers[0].ssm.wz.type;
    // QKV succeeds, then the unsupported Z format must abort before convolution.
    model.weights.layers[0].ssm.wz.type = -1;
    for (int request = 0; request < 3; request++) {
        BnSession *session = bn_session_create(&model, NULL); assert(session);
        memcpy(state, session->state.ssm_state, ns*sizeof(float));
        memcpy(conv, session->state.ssm_conv_state, nc*sizeof(float));
        float logits[32]; for (int i=0;i<32;i++) logits[i]=NAN;
        if (request == 0) assert(!bn_transformer_prefill(&model, session, tokens, 4, 0));
        else if (request == 1) assert(bn_transformer_prefill_no_logits(&model, session, tokens, 4, 0) == -1);
        else assert(bn_transformer_prefill_all(&model, session, tokens, 4, 0, logits) == -1);
        assert(session->pos == 0);
        assert(memcmp(state, session->state.ssm_state, ns*sizeof(float)) == 0);
        assert(memcmp(conv, session->state.ssm_conv_state, nc*sizeof(float)) == 0);
        for (int i=0;i<32;i++) assert(isnan(logits[i]));
        bn_session_free(session, NULL);
    }
    model.weights.layers[0].ssm.wz.type = saved_type;
    free(conv); free(state); bn_model_free(&model); bn_gguf_free(gf); free(buf);
    printf("PASSED\n");
}

static void test_ple_batch_ids(void) {
    size_t words = 0;
    for (size_t ci = 0; ci < sizeof(ple_ids_reference) / sizeof(ple_ids_reference[0]); ci++) {
        const BnPLEIdsReference *r = &ple_ids_reference[ci];
        BnConfig c = {0}; c.ple_ngram_size = r->ng; c.ple_heads_per_ngram = r->hp;
        c.ple_head_count = (r->ng - 1) * r->hp; c.ple_eos_token_id = 248044;
        for (int j = 0; j < r->ng; j++)
            c.ple_layer_multipliers[j] = r->profile == 0 ? UINT64_C(23703573157769) + j * UINT64_C(1000003) :
                (r->profile == 1 ? UINT64_MAX - (uint64_t)j * 2 : (UINT64_C(1) << 63) + j * UINT64_C(1000003));
        for (int h = 0; h < c.ple_head_count; h++) {
            c.ple_head_vocab_sizes[h] = 101 + 2 * h; c.ple_head_offsets[h] = 1000 * h;
        }
        int32_t tokens[129], starts[5] = {0}, context[28], result[129 * 56 + 2];
        for (int i = 0; i < r->nt; i++) tokens[i] = i % 11 == 5 ? 248044 : (i * 37 + r->seed) % 248000;
        starts[r->nr] = r->nt;
        if (r->nr == 4) { starts[1] = 0; starts[2] = r->nt / 3; starts[3] = r->nt / 2; }
        size_t nc = (size_t)r->nr * (r->ng - 1), ny = (size_t)r->nt * c.ple_head_count;
        for (size_t i = 0; i < nc; i++) context[i] = i % 3 == 0 ? 248044 : (int32_t)((i * 19 + r->seed) % 251);
        for (size_t i = 0; i < ny + 2; i++) result[i] = 0x5a5a;
        assert(bn_transformer_prefill_ple_ids(&c, tokens, r->nt, starts, r->nr, context, nc - 1, result + 1, ny) == -1);
        assert(bn_transformer_prefill_ple_ids(&c, tokens, r->nt, starts, r->nr, context, nc, result + 1, ny - 1) == -1);
        tokens[r->nt - 1] = -1;
        assert(bn_transformer_prefill_ple_ids(&c, tokens, r->nt, starts, r->nr, context, nc, result + 1, ny) == -1);
        tokens[r->nt - 1] = (r->nt - 1) % 11 == 5 ? 248044 : ((r->nt - 1) * 37 + r->seed) % 248000;
        for (size_t i = 0; i < ny + 2; i++) assert(result[i] == 0x5a5a);
        assert(!bn_transformer_prefill_ple_ids(&c, tokens, r->nt, starts, r->nr, context, nc, result + 1, ny));
        uint64_t hash = UINT64_C(14695981039346656037);
        for (size_t i = 0; i < ny; i++) for (int b = 0; b < 4; b++)
            hash = (hash ^ (((uint32_t)result[i + 1] >> (b * 8)) & 255)) * UINT64_C(1099511628211);
        assert(hash == r->hash);
        assert(result[0] == 0x5a5a && result[ny + 1] == 0x5a5a); words += ny;
    }
    printf("test_ple_batch_ids... PASSED (%zu exact independent IDs)\n", words);
}

static void test_native_expert_component_names(void) {
    char out[128],saved[128];
    const BnModelTensorRole roles[]={BN_MODEL_TENSOR_MOE_GATE_EXPS,
        BN_MODEL_TENSOR_MOE_UP_EXPS,BN_MODEL_TENSOR_MOE_DOWN_EXPS};
    const char *w[]={"w1","w3","w2"};
    const char *suffix[]={"trellis","suh","svh","mul1"};
    for(int p=0;p<3;p++)for(int c=0;c<4;c++) {
        char expected[128];
        snprintf(expected,sizeof(expected),"layers.39.ffn.experts.383.%s.%s",w[p],suffix[c]);
        assert(!bn_model_arch_expert_component_name("deepseek_v41",out,
            sizeof(out),39,383,roles[p],(BnModelExpertComponent)c));
        assert(!strcmp(out,expected));
        size_t needed=strlen(expected)+1;
        assert(!bn_model_arch_expert_component_name("deepseek_v41",out,
            needed,39,383,roles[p],(BnModelExpertComponent)c));
        memset(out,0xa5,sizeof(out));memcpy(saved,out,sizeof(out));
        assert(bn_model_arch_expert_component_name("deepseek_v41",out,
            needed-1,39,383,roles[p],(BnModelExpertComponent)c)==-1);
        assert(!memcmp(out,saved,sizeof(out)));
    }
    const char *bad_arch[]={NULL,"","deepseek_v4","llama","DeepseekV41ForCausalLM"};
    for(size_t i=0;i<sizeof(bad_arch)/sizeof(bad_arch[0]);i++) {
        assert(bn_model_arch_expert_component_name(bad_arch[i],out,sizeof(out),
            0,0,roles[0],BN_MODEL_EXPERT_COMPONENT_PACKED)==-1);
        assert(!memcmp(out,saved,sizeof(out)));
    }
    assert(bn_model_arch_expert_component_name("deepseek_v41",NULL,128,0,0,roles[0],0)==-1);
    assert(bn_model_arch_expert_component_name("deepseek_v41",out,0,0,0,roles[0],0)==-1);
    assert(bn_model_arch_expert_component_name("deepseek_v41",out,sizeof(out),-1,0,roles[0],0)==-1);
    assert(bn_model_arch_expert_component_name("deepseek_v41",out,sizeof(out),0,-1,roles[0],0)==-1);
    assert(bn_model_arch_expert_component_name("deepseek_v41",out,sizeof(out),0,0,BN_MODEL_TENSOR_FFN_GATE,0)==-1);
    assert(bn_model_arch_expert_component_name("deepseek_v41",out,sizeof(out),0,0,roles[0],(BnModelExpertComponent)-1)==-1);
    assert(bn_model_arch_expert_component_name("deepseek_v41",out,sizeof(out),0,0,roles[0],(BnModelExpertComponent)4)==-1);
    assert(!memcmp(out,saved,sizeof(out)));
    puts("Native architecture-owned expert component names PASSED");
}

int main(void) {
    test_native_expert_component_names();
    printf("=== Qwen3.6 Architecture Tests ===\n");
    test_ple_batch_ids();
    test_qwen36_dense();
    test_qwen36_moe();
    test_qwen36_explicit_dense();
    test_qwen36_explicit_moe();
    test_prefill_output_rows_state();
    test_prefill_projection_failure();
    printf("All Qwen3.6 architecture tests passed!\n");
    return 0;
}
