#include "transformer_kv_internal.h"
#include "transformer_gqa_internal.h"
#include "model.h"
#include "turboquant.h"

static const BnKVCPUOps *kv_cpu_ops(void) {
    return bn_transformer_kv_cpu_ops();
}

BnKVMode bn_transformer_kv_mode(const BnConfig *c, int tq_enabled) {
    if (c && c->kv_tq_bits > 0 && tq_enabled) return BN_KV_TQ;
    if (c && c->kv_f16) return BN_KV_FP16;
    return BN_KV_FP32;
}

int bn_transformer_kv_mode_stores_host_float_rows(BnKVMode mode) {
    return mode == BN_KV_FP32;
}

int bn_transformer_kv_mode_uses_turboquant(BnKVMode mode) {
    return mode == BN_KV_TQ;
}

int bn_transformer_kv_mode_uses_fp16(BnKVMode mode) {
    return mode == BN_KV_FP16;
}

int bn_transformer_kv_mode_uses_cpu_gqa_cache(BnKVMode mode) {
    return mode != BN_KV_TQ;
}

int bn_transformer_kv_host_float_cache_rows_available(const BnConfig *c) {
    return c && bn_transformer_kv_mode_stores_host_float_rows(
        bn_transformer_kv_mode(c, 1));
}

int bn_transformer_kv_host_cache_uses_fp16_rows(const BnConfig *c) {
    return bn_transformer_kv_mode_uses_fp16(bn_transformer_kv_mode(c, 0));
}

int bn_transformer_kv_requires_gpu_cache_write_staging(const BnConfig *c) {
    return bn_transformer_kv_host_cache_uses_fp16_rows(c);
}

size_t bn_transformer_kv_host_cache_element_size(const BnConfig *c) {
    return bn_transformer_kv_host_cache_uses_fp16_rows(c) ? sizeof(uint16_t)
                                                         : sizeof(float);
}

void bn_transformer_tq_write_kv(const BnTQState *tq,
                                BnRunState *s,
                                const float *k_tmp,
                                const float *v_tmp,
                                int n_kv_heads,
                                int head_size,
                                int attn_idx,
                                int cache_pos,
                                int seq_len) {
    int key_bytes = bn_tq_key_bytes(tq);
    int val_bytes = bn_tq_value_bytes(tq);
    size_t tq_loff_k = (size_t)attn_idx * seq_len * n_kv_heads * key_bytes;
    size_t tq_loff_v = (size_t)attn_idx * seq_len * n_kv_heads * val_bytes;
    uint8_t *kc_tq = s->key_cache_tq + tq_loff_k +
                     (size_t)cache_pos * n_kv_heads * key_bytes;
    uint8_t *vc_tq = s->value_cache_tq + tq_loff_v +
                     (size_t)cache_pos * n_kv_heads * val_bytes;
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
        bn_tq_quantize_key(tq, k_tmp + kv_h * head_size,
                           kc_tq + kv_h * key_bytes);
        bn_tq_quantize_value(tq, v_tmp + kv_h * head_size,
                             vc_tq + kv_h * val_bytes);
    }
}

void bn_transformer_tq_gqa_dispatch(BnModel *m,
                                    BnRunState *s,
                                    int attn_idx,
                                    int pos,
                                    int n_heads,
                                    int n_kv_heads,
                                    int head_size,
                                    int kv_mul) {
    const BnConfig *c = &m->config;
    const BnTQState *tq = bn_model_tq_state(m);
    int key_bytes = bn_tq_key_bytes(tq);
    int val_bytes = bn_tq_value_bytes(tq);
    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;

    size_t tq_loff_k = (size_t)attn_idx * c->seq_len * n_kv_heads * key_bytes;
    size_t tq_loff_v = (size_t)attn_idx * c->seq_len * n_kv_heads * val_bytes;

    BnGQATQCtx tctx = {
        .c = c, .s = s, .tq = tq,
        .tq_keys = s->key_cache_tq + tq_loff_k,
        .tq_values = s->value_cache_tq + tq_loff_v,
        .key_stride = n_kv_heads * key_bytes,
        .val_stride = n_kv_heads * val_bytes,
        .key_bytes = key_bytes,
        .val_bytes = val_bytes,
        .pos = pos, .n_kv = n_kv, .kv_mul = kv_mul,
        .head_size = head_size, .seq_len = c->seq_len,
        .n_kv_heads = n_kv_heads
    };
    BnTPTask gqa = { kv_cpu_ops()->tq_gqa, &tctx, n_heads };
    bn_tp_dispatch(bn_model_pool(m), &gqa, 1);
}

void bn_transformer_write_kv_fp16(BnRunState *s,
                                  size_t loff,
                                  int cache_pos,
                                  int kv_cache_stride,
                                  const float *k_tmp,
                                  const float *v_tmp,
                                  int kv_dim) {
    uint16_t *kc = (uint16_t *)s->key_cache +
                   loff + (size_t)cache_pos * kv_cache_stride;
    uint16_t *vc = (uint16_t *)s->value_cache +
                   loff + (size_t)cache_pos * kv_cache_stride;
    kv_cpu_ops()->write_kv_fp16(kc, vc, k_tmp, v_tmp, kv_dim);
}

void bn_transformer_kv_cache_rows(BnRunState *s,
                                  size_t loff,
                                  int cache_pos,
                                  int kv_cache_stride,
                                  float **key_cache_row,
                                  float **value_cache_row) {
    *key_cache_row = s->key_cache + loff + (size_t)cache_pos * kv_cache_stride;
    *value_cache_row = s->value_cache + loff + (size_t)cache_pos * kv_cache_stride;
}

void bn_transformer_write_kv_fp32(BnRunState *s,
                                  size_t loff,
                                  int cache_pos,
                                  int kv_cache_stride,
                                  const float *k_tmp,
                                  const float *v_tmp,
                                  int kv_dim) {
    float *kc, *vc;
    bn_transformer_kv_cache_rows(s, loff, cache_pos, kv_cache_stride, &kc, &vc);
    memcpy(kc, k_tmp, (size_t)kv_dim * sizeof(float));
    memcpy(vc, v_tmp, (size_t)kv_dim * sizeof(float));
}

int bn_transformer_write_host_kv_cache_row(BnRunState *s,
                                           BnKVMode mode,
                                           size_t loff,
                                           int cache_pos,
                                           int kv_cache_stride,
                                           const float *k_tmp,
                                           const float *v_tmp,
                                           int kv_dim) {
    if (bn_transformer_kv_mode_uses_turboquant(mode))
        return -1;
    if (bn_transformer_kv_mode_uses_fp16(mode)) {
        bn_transformer_write_kv_fp16(s, loff, cache_pos, kv_cache_stride,
                                     k_tmp, v_tmp, kv_dim);
        return 0;
    }
    bn_transformer_write_kv_fp32(s, loff, cache_pos, kv_cache_stride,
                                 k_tmp, v_tmp, kv_dim);
    return 0;
}
