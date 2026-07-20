#ifndef BN_MODEL_INTERNAL_H
#define BN_MODEL_INTERNAL_H

#include "model.h"
#include "platform.h"
#include <stddef.h>

struct BnModelRuntime {
    BnThreadPool *pool;
    int owns_pool;
    SHArena *weight_arena;
    BnTQState *tq_state;
    int owns_tq_state;
};

struct BnModelIO {
    BnMappedFile file;
    BnMoEIO moe_io;
};

struct BnModelBackendState {
    BnBackendModel *backend;
};

int bn_model_quant_type_supported(int type);
int bn_model_quant_uses_embedded_block_scale(int type);
int bn_model_quant_uses_embedded_tensor_scale(int type);
size_t bn_model_quant_embedded_tensor_scale_offset(int type,
                                                   int rows,
                                                   int cols);
int bn_model_quant_tied_logits_uses_quant_path(int type);
int bn_model_quant_logits_i8_cache_supported(int type);
void bn_model_quant_prepare_logits_i8_cache(const uint16_t *src,
                                            int8_t *dst,
                                            float *scales,
                                            int rows,
                                            int dim);
int bn_model_quant_is_dense_f32(int type);
int bn_model_quant_can_convert_dense_to_f32(int type);
int bn_model_quant_convert_dense_to_f32(int type,
                                        const void *src,
                                        float *dst,
                                        int n);
int bn_model_quant_dense_f32_type(void);
int bn_model_quant_dequant_row(int type,
                               const void *data,
                               int row,
                               int n,
                               float *out);
int bn_model_activation_is_relu2(int activation);
int bn_model_activation_is_gelu(int activation);
int bn_model_activation_uses_silu_path(int activation);
int bn_model_gguf_uses_moe(BnGGUFFile *file);
int bn_model_gguf_context_length(BnGGUFFile *file);

#endif // BN_MODEL_INTERNAL_H
