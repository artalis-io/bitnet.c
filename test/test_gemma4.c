#include "gguf.h"
#include "model.h"
#include "model_arch.h"
#include "session.h"
#include "transformer.h"
#if defined(BN_GEMMA4_TEST_WEBGPU) && defined(BN_ENABLE_WEBGPU)
#include "gpu_wgpu.h"
#endif
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint8_t *data; size_t pos, cap; } WriteBuffer;
typedef struct {
    char name[96];
    uint32_t type, n_dims;
    uint64_t dims[4], offset;
    size_t bytes;
} TensorSpec;
typedef struct { TensorSpec tensors[128]; int n_tensors, moe; } ModelSpec;

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
static void wb_u32(WriteBuffer *wb, uint32_t v) { wb_write(wb, &v, 4); }
static void wb_u64(WriteBuffer *wb, uint64_t v) { wb_write(wb, &v, 8); }
static void wb_f32(WriteBuffer *wb, float v) { wb_write(wb, &v, 4); }
static void wb_str(WriteBuffer *wb, const char *s) {
    wb_u64(wb, (uint64_t)strlen(s));
    wb_write(wb, s, strlen(s));
}
static size_t align32(size_t v) { return (v + 31u) & ~(size_t)31u; }

static void add_tensor(ModelSpec *s, const char *name, uint64_t d0, uint64_t d1, uint64_t d2) {
    TensorSpec *t = &s->tensors[s->n_tensors++];
    memset(t, 0, sizeof(*t));
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->type = BN_GGUF_TENSOR_F32;
    t->dims[0] = d0;
    t->n_dims = 1;
    uint64_t n = d0;
    if (d1) { t->dims[1] = d1; t->n_dims = 2; n *= d1; }
    if (d2) { t->dims[2] = d2; t->n_dims = 3; n *= d2; }
    t->bytes = (size_t)n * sizeof(float);
}

static void add_tensors(ModelSpec *s) {
    const int dim = 128, vocab = 8, layers = 2, hidden = 256, moe_hidden = 32, experts = 4;
    char name[96];
    add_tensor(s, "token_embd.weight", dim, vocab, 0);
    add_tensor(s, "output_norm.weight", dim, 0, 0);
    for (int l = 0; l < layers; l++) {
        int hs = l == 0 ? 64 : 32;
        int nkv = s->moe ? (l == 0 ? 2 : 1) : 1;
        int qdim = 4 * hs;
        int kvdim = nkv * hs;
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        add_tensor(s, name, dim, 0, 0);
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        add_tensor(s, name, dim, qdim, 0);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        add_tensor(s, name, dim, kvdim, 0);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        add_tensor(s, name, dim, kvdim, 0);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        add_tensor(s, name, qdim, dim, 0);
        snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
        add_tensor(s, name, hs, 0, 0);
        snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
        add_tensor(s, name, hs, 0, 0);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        add_tensor(s, name, dim, 0, 0);
        if (s->moe) {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
            add_tensor(s, name, dim, experts, 0);
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_up_exps.weight", l);
            add_tensor(s, name, dim, 2 * moe_hidden, experts);
            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
            add_tensor(s, name, moe_hidden, dim, experts);
        } else {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
            add_tensor(s, name, dim, hidden, 0);
            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
            add_tensor(s, name, dim, hidden, 0);
            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
            add_tensor(s, name, hidden, dim, 0);
        }
    }
}

static void kv_u32(WriteBuffer *wb, const char *suffix, uint32_t v) {
    char key[96]; snprintf(key, sizeof(key), "gemma4.%s", suffix);
    wb_str(wb, key); wb_u32(wb, BN_GGUF_TYPE_UINT32); wb_u32(wb, v);
}
static void kv_f32(WriteBuffer *wb, const char *suffix, float v) {
    char key[96]; snprintf(key, sizeof(key), "gemma4.%s", suffix);
    wb_str(wb, key); wb_u32(wb, BN_GGUF_TYPE_FLOAT32); wb_f32(wb, v);
}
static void kv_i32_arr(WriteBuffer *wb, const char *suffix, const int32_t *v, uint64_t n) {
    char key[96]; snprintf(key, sizeof(key), "gemma4.%s", suffix);
    wb_str(wb, key); wb_u32(wb, BN_GGUF_TYPE_ARRAY);
    wb_u32(wb, BN_GGUF_TYPE_INT32); wb_u64(wb, n);
    wb_write(wb, v, (size_t)n * sizeof(int32_t));
}

static void write_kvs(WriteBuffer *wb, int moe) {
    wb_str(wb, "general.architecture"); wb_u32(wb, BN_GGUF_TYPE_STRING); wb_str(wb, "gemma4");
    wb_str(wb, "tokenizer.ggml.tokens"); wb_u32(wb, BN_GGUF_TYPE_ARRAY);
    wb_u32(wb, BN_GGUF_TYPE_STRING); wb_u64(wb, 8);
    for (int i = 0; i < 8; i++) { char tok[8]; snprintf(tok, sizeof(tok), "t%d", i); wb_str(wb, tok); }
    kv_u32(wb, "embedding_length", 128);
    int32_t ffn[2] = {256, 256};
    if (moe) kv_u32(wb, "feed_forward_length", 256); else kv_i32_arr(wb, "feed_forward_length", ffn, 2);
    kv_u32(wb, "block_count", 2);
    kv_u32(wb, "attention.head_count", 4);
    int32_t nkv[2] = {2, 1};
    if (moe) kv_i32_arr(wb, "attention.head_count_kv", nkv, 2); else kv_u32(wb, "attention.head_count_kv", 1);
    kv_u32(wb, "context_length", 16);
    kv_f32(wb, "rope.freq_base", 1000000.0f);
    kv_f32(wb, "attention.layer_norm_rms_epsilon", 1e-6f);
    kv_u32(wb, "attention.key_length", 64);
    kv_u32(wb, "rope.dimension_count", 64);
    if (moe) {
        kv_u32(wb, "expert_count", 4);
        kv_u32(wb, "expert_used_count", 2);
        kv_u32(wb, "expert_feed_forward_length", 32);
    }
}

static BnGGUFFile *build_gemma4(uint8_t *buf, size_t cap, int moe) {
    ModelSpec spec = {0}; spec.moe = moe; add_tensors(&spec);
    WriteBuffer wb = {buf, 0, cap};
    wb_u32(&wb, BN_GGUF_MAGIC); wb_u32(&wb, 3);
    wb_u64(&wb, (uint64_t)spec.n_tensors);
    wb_u64(&wb, moe ? 15 : 12);
    write_kvs(&wb, moe);
    size_t cursor = 0;
    for (int i = 0; i < spec.n_tensors; i++) {
        TensorSpec *t = &spec.tensors[i];
        cursor = align32(cursor); t->offset = cursor; cursor += t->bytes;
        wb_str(&wb, t->name); wb_u32(&wb, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) wb_u64(&wb, t->dims[d]);
        wb_u32(&wb, t->type); wb_u64(&wb, t->offset);
    }
    wb.pos = align32(wb.pos);
    size_t data_offset = wb.pos;
    wb_zero(&wb, cursor);
    for (int i = 0; i < spec.n_tensors; i++) {
        TensorSpec *t = &spec.tensors[i];
        float *dst = (float *)(buf + data_offset + (size_t)t->offset);
        size_t n = t->bytes / sizeof(float);
        int norm = strstr(t->name, "_norm.weight") || strcmp(t->name, "output_norm.weight") == 0;
        for (size_t j = 0; j < n; j++) dst[j] = norm ? 1.0f : 0.0f;
    }
    return bn_gguf_open(buf, wb.pos);
}

static void assert_forward_finite(BnModel *model) {
    BnSession *s = bn_session_create(model, NULL);
    assert(s);
    float *logits = bn_transformer_forward(model, s, 0, 0);
    assert(logits);
    for (int i = 0; i < model->config.vocab_size; i++) assert(isfinite(logits[i]));
    bn_session_free(s, NULL);
}

#if defined(BN_GEMMA4_TEST_WEBGPU) && defined(BN_ENABLE_WEBGPU)
static int assert_forward_finite_webgpu(BnModel *model) {
    BnGPUBackend *gpu = bn_gpu_wgpu_create("shaders");
    if (!gpu) {
        printf("SKIPPED_WEBGPU ");
        return 0;
    }
    assert(bn_model_upload_weights(model, gpu) == 0);
    assert_forward_finite(model);
    bn_model_free(model);
    bn_gpu_wgpu_destroy(gpu);
    return 1;
}
#endif

static void test_gemma4_dense(void) {
    printf("test_gemma4_dense... ");
    uint8_t *buf = calloc(1, 4 * 1024 * 1024); assert(buf);
    BnGGUFFile *gf = build_gemma4(buf, 4 * 1024 * 1024, 0); assert(gf);
    int32_t sections[3] = {16, 24, 24};
    assert(bn_model_arch_is_gemma4("gemma4") == 1);
    assert(bn_model_arch_attention_value_shares_key("gemma4") == 1);
    assert(bn_model_arch_rope_text_dims(64, sections, 3) == 32);
    BnModel m; assert(bn_model_load(&m, gf, 8, 0, 0) == 0);
    assert(m.backend == NULL);
    assert(bn_model_gpu(&m) == NULL);
    assert(m.config.arch_flags & BN_MODEL_ARCH_FLAG_GEMMA4);
    assert(bn_model_arch_requires_large_gpu_graph_fallback(&m.config));
    assert(m.config.head_size == 64);
    assert(m.config.kv_dim == 64);
    assert(m.weights.layers[0].q_dim == 256);
    assert(m.weights.layers[1].head_size == 32);
    assert(m.weights.layers[1].kv_dim == 32);
    assert_forward_finite(&m);
    bn_model_free(&m); bn_gguf_free(gf); free(buf);
    printf("PASSED\n");
}

static void test_gemma4_moe_fused_gate_up(void) {
    printf("test_gemma4_moe_fused_gate_up... ");
    uint8_t *buf = calloc(1, 4 * 1024 * 1024); assert(buf);
    BnGGUFFile *gf = build_gemma4(buf, 4 * 1024 * 1024, 1); assert(gf);
    const BnModelArchOps *ops = bn_model_arch_ops_for("gemma4");
    assert(ops);
    assert(bn_model_arch_infer_moe_hidden(gf, ops) == 32);
    assert(bn_model_arch_has_shared_expert(gf, ops) == 0);
    BnModel m; assert(bn_model_load(&m, gf, 8, 0, 0) == 0);
    assert(m.backend == NULL);
    assert(bn_model_gpu(&m) == NULL);
    m.moe_io.mmap_base = gf->raw;
    BnMoEExpertMap *em = &m.weights.layers[0].expert_map;
    assert(m.config.n_kv_heads == 2);
    assert(m.config.kv_dim == 128);
    assert(m.weights.layers[1].n_kv_heads == 1);
    assert(m.weights.layers[1].kv_dim == 32);
    assert(em->expert_gate_bytes == 128 * 32 * sizeof(float));
    assert(em->up_offset == em->gate_offset + em->expert_gate_bytes);
    assert(em->gate_stride == 128 * 64 * sizeof(float));
    assert_forward_finite(&m);
    bn_model_free(&m); bn_gguf_free(gf); free(buf);
    printf("PASSED\n");
}

#if defined(BN_GEMMA4_TEST_WEBGPU) && defined(BN_ENABLE_WEBGPU)
static void test_gemma4_dense_webgpu(void) {
    printf("test_gemma4_dense_webgpu... ");
    uint8_t *buf = calloc(1, 4 * 1024 * 1024); assert(buf);
    BnGGUFFile *gf = build_gemma4(buf, 4 * 1024 * 1024, 0); assert(gf);
    BnModel m; assert(bn_model_load(&m, gf, 8, 0, 0) == 0);
    int ran = assert_forward_finite_webgpu(&m);
    if (!ran) bn_model_free(&m);
    bn_gguf_free(gf); free(buf);
    printf("PASSED\n");
}

static void test_gemma4_moe_webgpu(void) {
    printf("test_gemma4_moe_webgpu... ");
    uint8_t *buf = calloc(1, 4 * 1024 * 1024); assert(buf);
    BnGGUFFile *gf = build_gemma4(buf, 4 * 1024 * 1024, 1); assert(gf);
    BnModel m; assert(bn_model_load(&m, gf, 8, 0, 0) == 0);
    m.moe_io.mmap_base = gf->raw;
    int ran = assert_forward_finite_webgpu(&m);
    if (!ran) bn_model_free(&m);
    bn_gguf_free(gf); free(buf);
    printf("PASSED\n");
}
#endif

int main(void) {
    printf("=== Gemma4 Architecture Tests ===\n");
    test_gemma4_dense();
    test_gemma4_moe_fused_gate_up();
#if defined(BN_GEMMA4_TEST_WEBGPU) && defined(BN_ENABLE_WEBGPU)
    test_gemma4_dense_webgpu();
    test_gemma4_moe_webgpu();
#endif
    printf("All Gemma4 architecture tests passed!\n");
    return 0;
}
