#include "transformer_kv_internal.h"
#include "transformer_gqa_internal.h"
#include "model.h"
#include "turboquant.h"

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __AVX2__
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

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
    const BnTQState *tq = m->tq_state;
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
#ifdef __ARM_NEON
    bn_tp_fn attn_fn = bn_transformer_gqa_tq_neon_range;
#else
    bn_tp_fn attn_fn = bn_transformer_gqa_tq_scalar_range;
#endif
    BnTPTask gqa = { attn_fn, &tctx, n_heads };
    bn_tp_dispatch(m->pool, &gqa, 1);
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
#ifdef __ARM_NEON
    for (int i = 0; i < kv_dim; i += 4) {
        vst1_u16(kc + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(k_tmp + i))));
        vst1_u16(vc + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(v_tmp + i))));
    }
#elif defined(__AVX2__)
    for (int i = 0; i < kv_dim; i += 8) {
        _mm_storeu_si128((__m128i *)(kc + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(k_tmp + i),
                                         _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128((__m128i *)(vc + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(v_tmp + i),
                                         _MM_FROUND_TO_NEAREST_INT));
    }
#else
    for (int i = 0; i < kv_dim; i++) {
        kc[i] = bn_fp32_to_fp16(k_tmp[i]);
        vc[i] = bn_fp32_to_fp16(v_tmp[i]);
    }
#endif
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
