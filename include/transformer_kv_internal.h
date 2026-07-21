#ifndef BN_TRANSFORMER_KV_INTERNAL_H
#define BN_TRANSFORMER_KV_INTERNAL_H

#include "model_config.h"
#include "model_run_state.h"
#include "threadpool.h"
#include <stddef.h>
#include <stdint.h>

typedef struct BnModel BnModel;
typedef struct BnTQState BnTQState;

typedef struct {
    const char *name;
    bn_tp_fn tq_gqa;
    void (*write_kv_fp16)(uint16_t *kc, uint16_t *vc,
                          const float *k_tmp, const float *v_tmp,
                          int kv_dim);
} BnKVCPUOps;

const BnKVCPUOps *bn_transformer_kv_cpu_ops(void);
int bn_transformer_kv_host_float_cache_rows_available(const BnConfig *c);

void bn_transformer_tq_write_kv(const BnTQState *tq,
                                BnRunState *s,
                                const float *k_tmp,
                                const float *v_tmp,
                                int n_kv_heads,
                                int head_size,
                                int attn_idx,
                                int cache_pos,
                                int seq_len);
void bn_transformer_tq_gqa_dispatch(BnModel *m,
                                    BnRunState *s,
                                    int attn_idx,
                                    int pos,
                                    int n_heads,
                                    int n_kv_heads,
                                    int head_size,
                                    int kv_mul);
void bn_transformer_write_kv_fp16(BnRunState *s,
                                  size_t loff,
                                  int cache_pos,
                                  int kv_cache_stride,
                                  const float *k_tmp,
                                  const float *v_tmp,
                                  int kv_dim);
void bn_transformer_kv_cache_rows(BnRunState *s,
                                  size_t loff,
                                  int cache_pos,
                                  int kv_cache_stride,
                                  float **key_cache_row,
                                  float **value_cache_row);
void bn_transformer_write_kv_fp32(BnRunState *s,
                                  size_t loff,
                                  int cache_pos,
                                  int kv_cache_stride,
                                  const float *k_tmp,
                                  const float *v_tmp,
                                  int kv_dim);

#endif // BN_TRANSFORMER_KV_INTERNAL_H
