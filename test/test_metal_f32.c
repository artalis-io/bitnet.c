#include "gpu_backend.h"
#include "gpu_metal.h"
#include "model.h"
#include "model_config.h"
#include "quant.h"
#include "sh_arena.h"
#include "../src/gpu_shader_ir_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    enum { rows = 64, cols = 2048 };
    float *weights = malloc((size_t)rows * cols * sizeof(float));
    float *x = malloc((size_t)cols * sizeof(float));
    float *expected = calloc(rows, sizeof(float));
    float *actual = calloc(rows, sizeof(float));
    if (!weights || !x || !expected || !actual)
        return 2;

    for (int c = 0; c < cols; c++)
        x[c] = (float)((c % 29) - 14) / 29.0f;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float w = (float)(((r * 17 + c * 3) % 31) - 15) / 31.0f;
            weights[(size_t)r * cols + c] = w;
            expected[r] += w * x[c];
        }
    }

    BnGPUBackend *gpu = bn_gpu_metal_create("shaders/metal/");
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    int memory_info_ok = gpu &&
        bn_gpu_backend_query_memory(gpu, &free_bytes, &total_bytes) == 0 &&
        total_bytes > 0 && free_bytes <= total_bytes;
    void *buffer = gpu ? bn_gpu_backend_create_buffer(
        gpu, weights, (size_t)rows * cols * sizeof(float),
        BN_GGUF_TENSOR_F32, rows, cols) : NULL;
    int rc = buffer ? bn_gpu_backend_matvec(
        gpu, actual, buffer, x, rows, cols, BN_GGUF_TENSOR_F32) : -1;
    float max_diff = 0.0f;
    int max_row = -1;
    int nonzero_rows = 0;
    for (int r = 0; r < rows; r++) {
        if (actual[r] != 0.0f)
            nonzero_rows++;
        float diff = fabsf(actual[r] - expected[r]);
        if (diff > max_diff) {
            max_diff = diff;
            max_row = r;
        }
    }
    printf("Metal F32 matvec max_diff=%.9g row=%d expected=%.9g actual=%.9g nonzero=%d/%d\n",
           max_diff, max_row, expected[max_row], actual[max_row],
           nonzero_rows, rows);

    if (buffer)
        bn_gpu_backend_destroy_buffer(gpu, buffer);

    enum { q4_rows = 256, q4_cols = 10240 };
    size_t q4_blocks_count =
        (size_t)q4_rows * (size_t)(q4_cols / 32);
    BnBlockQ4_0 *q4_blocks = calloc(q4_blocks_count,
                                     sizeof(BnBlockQ4_0));
    float *q4_expected = calloc(q4_rows, sizeof(float));
    float *q4_actual = calloc(q4_rows, sizeof(float));
    float *q4_quant_only_actual = calloc(q4_rows, sizeof(float));
    float *q4_x = malloc((size_t)q4_cols * sizeof(float));
    int8_t *x_q = malloc(q4_cols);
    BnQWeight q4_weight = {
        .data = q4_blocks,
        .type = BN_GGUF_TENSOR_Q4_0,
        .rows = q4_rows,
        .cols = q4_cols,
        .scale = 1.0f,
    };
    size_t prepared_size = bn_quant_prepared_qweight_size(&q4_weight, NULL);
    SHArena *arena = sh_arena_create(prepared_size);
    BnPreparedWeight prepared = { 0 };
    int q4_rc = -1;
    int q4_quant_only_rc = -1;
    float q4_max_diff = INFINITY;
    float q4_quant_only_max_diff = INFINITY;
    float routed_max_diff = INFINITY;
    int routed_rc = -1;
    int q4_max_row = -1;
    if (q4_blocks && q4_expected && q4_actual && q4_quant_only_actual &&
        q4_x && x_q && arena) {
        for (int c = 0; c < q4_cols; c++)
            q4_x[c] = sinf((float)c * 0.017f) * 0.4f +
                      cosf((float)c * 0.0031f) * 0.15f;
        for (size_t b = 0; b < q4_blocks_count; b++) {
            float scale = 0.015625f + (float)(b % 13) / 4096.0f;
            q4_blocks[b].d = bn_fp32_to_fp16(scale);
            for (int i = 0; i < 16; i++) {
                uint8_t lo = (uint8_t)((b * 5 + (size_t)i * 3) & 15u);
                uint8_t hi = (uint8_t)((b * 7 + (size_t)i * 11 + 1) & 15u);
                q4_blocks[b].qs[i] = (uint8_t)(lo | (hi << 4));
            }
        }
        if (bn_quant_prepare_qweight(&prepared, &q4_weight, arena) == 0) {
            bn_quant_matvec_prepared(q4_expected, &q4_weight, &prepared,
                                     q4_x, x_q, NULL);
            bn_gpu_backend_configure_prepared_native_quant(gpu, 1);
            void *q4_buffer = bn_gpu_backend_create_buffer(
                gpu, q4_blocks,
                q4_blocks_count * sizeof(BnBlockQ4_0),
                BN_GGUF_TENSOR_Q4_0, q4_rows, q4_cols);
            q4_rc = q4_buffer ? bn_gpu_backend_matvec(
                gpu, q4_actual, q4_buffer, q4_x, q4_rows, q4_cols,
                BN_GGUF_TENSOR_Q4_0) : -1;
            q4_max_diff = 0.0f;
            for (int r = 0; r < q4_rows; r++) {
                float diff = fabsf(q4_actual[r] - q4_expected[r]);
                if (diff > q4_max_diff) {
                    q4_max_diff = diff;
                    q4_max_row = r;
                }
            }
            printf("Metal prepared Q4_0 max_diff=%.9g row=%d expected=%.9g actual=%.9g\n",
                   q4_max_diff, q4_max_row, q4_expected[q4_max_row],
                   q4_actual[q4_max_row]);
            if (q4_buffer)
                bn_gpu_backend_destroy_buffer(gpu, q4_buffer);

            void *q4_quant_only_buffer =
                bn_gpu_backend_create_quant_only_buffer(
                    gpu, q4_blocks,
                    q4_blocks_count * sizeof(BnBlockQ4_0),
                    BN_GGUF_TENSOR_Q4_0, q4_rows, q4_cols);
            BnConfig config = {
                .dim = q4_cols,
                .hidden_dim = q4_cols,
                .n_layers = 1,
                .n_heads = 1,
                .n_kv_heads = 1,
                .vocab_size = 1,
                .seq_len = 1,
                .head_size = q4_cols,
                .kv_dim = q4_cols,
                .kv_mul = 1,
            };
            BnModel model = { .config = config };
            BnGPUOp op = {
                .op_kind = BN_GPU_OP_MATVEC,
                .op_code = BN_GPU_CODE_MATVEC,
                .type = BN_GGUF_TENSOR_Q4_0,
                .W_buf = q4_quant_only_buffer,
                .buf_in = BN_GPU_VALUE_X,
                .buf_out = BN_GPU_VALUE_XB,
                .buf_aux = -1,
                .rows = q4_rows,
                .cols = q4_cols,
                .flags = BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION,
                .p = { q4_rows, q4_cols, 1, 0, 0, 0, 1, 0 },
            };
            if (q4_quant_only_buffer &&
                bn_model_init_gpu_activations(&model, gpu) == 0 &&
                bn_gpu_backend_write_activation(
                    gpu, BN_GPU_VALUE_X, q4_x,
                    (size_t)q4_cols * sizeof(float), 0) == 0) {
                q4_quant_only_rc = bn_gpu_backend_execute(
                    gpu, &op, 1, BN_GPU_VALUE_XB,
                    q4_quant_only_actual, q4_rows);
            }
            q4_quant_only_max_diff = 0.0f;
            int q4_quant_only_max_row = -1;
            for (int r = 0; r < q4_rows; r++) {
                float diff = fabsf(q4_quant_only_actual[r] - q4_expected[r]);
                if (diff > q4_quant_only_max_diff) {
                    q4_quant_only_max_diff = diff;
                    q4_quant_only_max_row = r;
                }
            }
            printf("Metal quant-only Q4_0 Q8-input max_diff=%.9g row=%d",
                   q4_quant_only_max_diff, q4_quant_only_max_row);
            if (q4_quant_only_max_row >= 0)
                printf(" expected=%.9g actual=%.9g",
                       q4_expected[q4_quant_only_max_row],
                       q4_quant_only_actual[q4_quant_only_max_row]);
            printf("\n");
            bn_gpu_backend_free_activations(gpu);
            if (q4_quant_only_buffer)
                bn_gpu_backend_destroy_buffer(gpu, q4_quant_only_buffer);
        }
    }

    enum { moe_dim = 64, moe_hidden = 64, moe_experts = 2, moe_k = 2 };
    const size_t moe_gate_blocks =
        (size_t)moe_experts * moe_hidden * (moe_dim / 32);
    const size_t moe_down_blocks =
        (size_t)moe_experts * moe_dim * (moe_hidden / 32);
    BnBlockQ4_0 *moe_gate = calloc(moe_gate_blocks, sizeof(*moe_gate));
    BnBlockQ4_0 *moe_up = calloc(moe_gate_blocks, sizeof(*moe_up));
    BnBlockQ4_0 *moe_down = calloc(moe_down_blocks, sizeof(*moe_down));
    float *moe_x = malloc((size_t)moe_dim * sizeof(*moe_x));
    float *moe_expected = calloc(moe_dim, sizeof(*moe_expected));
    float *moe_actual = calloc(moe_dim, sizeof(*moe_actual));
    float *moe_g = malloc((size_t)moe_hidden * sizeof(*moe_g));
    float *moe_u = malloc((size_t)moe_hidden * sizeof(*moe_u));
    float *moe_mid = malloc((size_t)moe_hidden * sizeof(*moe_mid));
    float *moe_part = malloc((size_t)moe_dim * sizeof(*moe_part));
    int8_t *moe_xq = malloc(moe_hidden > moe_dim ? moe_hidden : moe_dim);
    if (moe_gate && moe_up && moe_down && moe_x && moe_expected &&
        moe_actual && moe_g && moe_u && moe_mid && moe_part && moe_xq) {
        for (int i = 0; i < moe_dim; i++)
            moe_x[i] = sinf((float)i * 0.13f) * 0.7f;
        for (size_t b = 0; b < moe_gate_blocks; b++) {
            BnBlockQ4_0 *blocks[2] = { &moe_gate[b], &moe_up[b] };
            for (int which = 0; which < 2; which++) {
                blocks[which]->d = bn_fp32_to_fp16(
                    0.0125f + (float)((b + which * 3) % 7) * 0.001f);
                for (int i = 0; i < 16; i++) {
                    uint8_t lo = (uint8_t)((b * 3 + i * 5 + which) & 15u);
                    uint8_t hi = (uint8_t)((b * 7 + i * 2 + which + 1) & 15u);
                    blocks[which]->qs[i] = (uint8_t)(lo | (hi << 4));
                }
            }
        }
        for (size_t b = 0; b < moe_down_blocks; b++) {
            moe_down[b].d = bn_fp32_to_fp16(
                0.011f + (float)(b % 9) * 0.00075f);
            for (int i = 0; i < 16; i++) {
                uint8_t lo = (uint8_t)((b * 11 + i * 3) & 15u);
                uint8_t hi = (uint8_t)((b * 5 + i * 7 + 2) & 15u);
                moe_down[b].qs[i] = (uint8_t)(lo | (hi << 4));
            }
        }
        const int indices[moe_k] = { 1, 0 };
        const float route_weights[moe_k] = { 0.625f, 0.375f };
        const size_t gate_expert_blocks =
            (size_t)moe_hidden * (moe_dim / 32);
        const size_t down_expert_blocks =
            (size_t)moe_dim * (moe_hidden / 32);
        for (int slot = 0; slot < moe_k; slot++) {
            int expert = indices[slot];
            BnQWeight gate_weight = {
                .data = moe_gate + (size_t)expert * gate_expert_blocks,
                .type = BN_GGUF_TENSOR_Q4_0,
                .rows = moe_hidden, .cols = moe_dim, .scale = 1.0f,
            };
            BnQWeight up_weight = gate_weight;
            up_weight.data = moe_up + (size_t)expert * gate_expert_blocks;
            BnQWeight down_weight = {
                .data = moe_down + (size_t)expert * down_expert_blocks,
                .type = BN_GGUF_TENSOR_Q4_0,
                .rows = moe_dim, .cols = moe_hidden, .scale = 1.0f,
            };
            bn_quant_matvec_prepared_flags(
                moe_g, &gate_weight, NULL, moe_x, moe_xq, NULL,
                BN_MATVEC_TASK_REFERENCE_DOT);
            bn_quant_matvec_prepared_flags(
                moe_u, &up_weight, NULL, moe_x, moe_xq, NULL,
                BN_MATVEC_TASK_REFERENCE_DOT);
            for (int i = 0; i < moe_hidden; i++) {
                float g = moe_g[i];
                float inner = 0.7978845608028654f * g *
                              (1.0f + 0.044715f * g * g);
                float activated = g <= -10.0f ? 0.0f
                    : g >= 10.0f ? g
                    : 0.5f * g * (1.0f + tanhf(inner));
                moe_mid[i] = activated * moe_u[i];
            }
            bn_quant_matvec_prepared_flags(
                moe_part, &down_weight, NULL, moe_mid, moe_xq, NULL,
                BN_MATVEC_TASK_REFERENCE_DOT);
            for (int i = 0; i < moe_dim; i++)
                moe_expected[i] += route_weights[slot] * moe_part[i];
        }
        void *gate_buf = bn_gpu_backend_create_quant_only_buffer(
            gpu, moe_gate, moe_gate_blocks * sizeof(*moe_gate),
            BN_GGUF_TENSOR_Q4_0, moe_hidden, moe_dim);
        void *up_buf = bn_gpu_backend_create_quant_only_buffer(
            gpu, moe_up, moe_gate_blocks * sizeof(*moe_up),
            BN_GGUF_TENSOR_Q4_0, moe_hidden, moe_dim);
        void *down_buf = bn_gpu_backend_create_quant_only_buffer(
            gpu, moe_down, moe_down_blocks * sizeof(*moe_down),
            BN_GGUF_TENSOR_Q4_0, moe_dim, moe_hidden);
        BnConfig moe_config = {
            .dim = moe_dim, .hidden_dim = moe_hidden, .n_layers = 1,
            .n_heads = 1, .n_kv_heads = 1, .vocab_size = 1, .seq_len = 1,
            .head_size = moe_dim, .kv_dim = moe_dim, .kv_mul = 1,
            .n_experts = moe_experts, .n_experts_active = moe_k,
            .moe_intermediate_size = moe_hidden,
        };
        BnModel moe_model = { .config = moe_config };
        if (gate_buf && up_buf && down_buf &&
            bn_model_init_gpu_activations(&moe_model, gpu) == 0) {
            float route[2 * moe_k];
            for (int slot = 0; slot < moe_k; slot++) {
                route[slot] = route_weights[slot];
                route[moe_k + slot] = (float)indices[slot];
            }
            BnGPUOp routed_op = {
                .op_kind = BN_GPU_OP_FFN,
                .op_code = BN_GPU_CODE_MOE_ROUTED_FFN,
                .type = BN_GGUF_TENSOR_Q4_0,
                .W_buf = gate_buf, .W_buf2 = up_buf, .W_buf3 = down_buf,
                .buf_in = BN_GPU_VALUE_XB,
                .buf_out = BN_GPU_VALUE_MOE_OUT,
                .buf_aux = BN_GPU_VALUE_MOE_HB2,
                .rows = moe_hidden, .cols = moe_dim,
                .p = { moe_hidden, moe_experts, moe_k,
                       BN_MODEL_ACTIVATION_GELU, BN_GPU_VALUE_MOE_HB,
                       0,
                       moe_dim * (moe_hidden / 32) * sizeof(BnBlockQ4_0),
                       moe_hidden * (moe_dim / 32) * sizeof(BnBlockQ4_0) },
            };
            if (bn_gpu_backend_write_activation(
                    gpu, BN_GPU_VALUE_XB, moe_x,
                    (size_t)moe_dim * sizeof(*moe_x), 0) == 0 &&
                bn_gpu_backend_write_activation(
                    gpu, BN_GPU_VALUE_MOE_HB2, route,
                    sizeof(route), 0) == 0)
                routed_rc = bn_gpu_backend_execute(
                    gpu, &routed_op, 1, BN_GPU_VALUE_MOE_OUT,
                    moe_actual, moe_dim);
            routed_max_diff = 0.0f;
            for (int i = 0; i < moe_dim; i++) {
                float diff = fabsf(moe_actual[i] - moe_expected[i]);
                if (diff > routed_max_diff) routed_max_diff = diff;
            }
            printf("Metal routed Q4_0 FFN max_diff=%.9g\n", routed_max_diff);
            bn_gpu_backend_free_activations(gpu);
        }
        if (gate_buf) bn_gpu_backend_destroy_buffer(gpu, gate_buf);
        if (up_buf) bn_gpu_backend_destroy_buffer(gpu, up_buf);
        if (down_buf) bn_gpu_backend_destroy_buffer(gpu, down_buf);
    }
    if (gpu)
        bn_gpu_metal_destroy(gpu);
    free(weights);
    free(x);
    free(expected);
    free(actual);
    sh_arena_free(arena);
    free(x_q);
    free(q4_x);
    free(q4_quant_only_actual);
    free(q4_actual);
    free(q4_expected);
    free(q4_blocks);
    free(moe_xq);
    free(moe_part);
    free(moe_mid);
    free(moe_u);
    free(moe_g);
    free(moe_actual);
    free(moe_expected);
    free(moe_x);
    free(moe_down);
    free(moe_up);
    free(moe_gate);
    return memory_info_ok && rc == 0 && max_diff < 1e-4f &&
           q4_rc == 0 && q4_max_diff < 1e-3f &&
           q4_quant_only_rc == 0 && q4_quant_only_max_diff < 1e-3f &&
           routed_rc == 0 && routed_max_diff < 1e-2f ? 0 : 1;
}
