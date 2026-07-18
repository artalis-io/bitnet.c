#include "moe_internal.h"
#include "backend_quant.h"
#include "model_arch.h"

uint32_t bn_moe_gateup_task_flags(const BnConfig *c) {
    return bn_model_arch_moe_forces_float_kquant_gateup(c)
        ? BN_MATVEC_TASK_FORCE_FLOAT_KQUANT
        : 0u;
}

int bn_moe_quant_supports_prepared_q8k(int type) {
    return bn_backend_quant_can_preq8k(type);
}

int bn_moe_quant_uses_embedded_tensor_scale(int type) {
    return bn_backend_quant_has_embedded_tensor_scale(type);
}

size_t bn_moe_quant_embedded_tensor_scale_offset(int type, int rows, int cols) {
    return bn_backend_quant_embedded_tensor_scale_offset(type, rows, cols);
}

void bn_moe_quant_matvec_gateup_gpu_buffers(BnMatvecTask *tasks,
                                            const void **buffers,
                                            int n_tasks,
                                            const float *x,
                                            int8_t *x_q_buf,
                                            BnThreadPool *pool,
                                            BnGPUBackend *gpu) {
    bn_backend_quant_matvec_batch_gpu_buf(tasks, buffers, n_tasks, x, x_q_buf,
                                          pool, gpu);
}

void bn_moe_quant_matvec_down_gpu_buffer(float *out,
                                         const BnQWeight *W,
                                         void *W_buf,
                                         const float *x,
                                         int8_t *x_q_buf,
                                         BnThreadPool *pool,
                                         BnGPUBackend *gpu) {
    bn_backend_quant_matvec_gpu_buf(out, W, W_buf, x, x_q_buf, pool, gpu);
}
