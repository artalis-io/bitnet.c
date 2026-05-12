#include "gguf.h"
#include "model.h"
#include "model_arch.h"
#include "session.h"
#include "transformer.h"
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
            add_tensor(s, name, BN_GGUF_TENSOR_F32, q_dim, 0, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
            add_tensor(s, name, BN_GGUF_TENSOR_F32, kv_dim, 0, 0);
        }

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
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
    BnSession *session = bn_session_create(model, NULL);
    assert(session != NULL);
    float *logits = bn_transformer_forward(model, session, 0, 0);
    assert(logits != NULL);
    for (int i = 0; i < model->config.vocab_size; i++)
        assert(isfinite(logits[i]));
    bn_session_free(session, NULL);
}

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
    assert(model.config.n_experts == 0);
    assert(model.config.full_attn_interval == 4);
    assert(model.config.n_layers == 4);
    assert(model.config.seq_len == 8);
    assert(model.config.head_size == 64);
    assert(model.config.rope_dim_count == 64);
    assert(model.weights.layers[0].wqkv.data != NULL);
    assert(model.weights.layers[3].wq.rows == 256);
    assert(model.weights.layers[3].ffn_gate.data != NULL);
    assert_forward_finite(&model);

    bn_model_free(&model);
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
    assert(bn_model_arch_infer_moe_hidden(gf) == 64);
    assert(bn_model_arch_has_shared_expert(gf) == 1);
    assert(bn_model_arch_infer_shared_expert_hidden(gf) == 64);

    BnModel model;
    assert(bn_model_load(&model, gf, 8, 0, 0) == 0);
    model.moe_io.mmap_base = gf->raw;
    assert(model.config.n_experts == 4);
    assert(model.config.n_experts_active == 2);
    assert(model.config.moe_intermediate_size == 64);
    assert(model.config.has_shared_expert == 1);
    assert(model.weights.layers[0].expert_map.expert_gate_bytes == 128 * 64 * sizeof(float));
    assert(model.weights.layers[0].shared_expert_gate != NULL);
    assert(model.weights.layers[3].wq.rows == 256);
    assert_forward_finite(&model);

    bn_model_free(&model);
    bn_gguf_free(gf);
    free(buf);
    printf("PASSED\n");
}

int main(void) {
    printf("=== Qwen3.6 Architecture Tests ===\n");
    test_qwen36_dense();
    test_qwen36_moe();
    printf("All Qwen3.6 architecture tests passed!\n");
    return 0;
}
