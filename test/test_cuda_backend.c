#include "gpu_cuda.h"
#include "gguf.h"
#include "model_config.h"
#include "quant.h"
#include "../src/gpu_shader_ir_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void expect_close(const float *got, const float *ref, int n) {
    for (int i = 0; i < n; i++)
        assert(fabsf(got[i] - ref[i]) < 1e-4f);
}

static void run_matvec_case(BnGPUBackend *gpu, const void *data,
                            size_t data_size, int type, int rows, int cols,
                            const float *x, const float *ref) {
    float out[8] = { 0 };
    assert(rows <= (int)(sizeof(out) / sizeof(out[0])));
    void *buf = gpu->buffer_create(gpu->ctx, data, data_size, type, rows, cols);
    assert(buf != NULL);
    assert(gpu->matvec(gpu->ctx, out, buf, x, rows, cols, type) == 0);
    gpu->buffer_destroy(gpu->ctx, buf);
    expect_close(out, ref, rows);
}

int main(void) {
#ifndef BN_ENABLE_CUDA
    printf("CUDA backend test skipped: BN_ENABLE_CUDA not set\n");
    return 0;
#else
    BnGPUBackend *gpu = bn_gpu_cuda_create();
    if (!gpu) {
        printf("CUDA backend test skipped: no CUDA device\n");
        return 0;
    }

    assert(gpu->init_activations != NULL);
    assert(gpu->write_activation != NULL);
    assert(gpu->read_activation != NULL);
    assert(gpu->free_activations != NULL);
    BnConfig cfg = {0};
    cfg.dim = 4;
    cfg.hidden_dim = 8;
    cfg.n_layers = 1;
    cfg.n_heads = 1;
    cfg.n_kv_heads = 1;
    cfg.vocab_size = 16;
    cfg.seq_len = 8;
    cfg.rope_theta = BN_DEFAULT_ROPE_THETA;
    cfg.head_size = 4;
    cfg.kv_dim = 4;
    cfg.kv_mul = 1;
    assert(gpu->init_activations(gpu->ctx, &cfg) == 0);
    {
        float in[4] = { 1.0f, -2.0f, 3.0f, -4.0f };
        float out[4] = { 0 };
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, in,
                                     sizeof(in), 0) == 0);
        assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_X, out,
                                    sizeof(out), 0) == 0);
        expect_close(out, in, 4);
    }
    gpu->free_activations(gpu->ctx);

    const int rows = 3;
    const int cols = 4;
    const float W_f32[12] = {
        1.0f,  2.0f,  3.0f,  4.0f,
       -1.0f,  0.5f,  2.0f, -0.5f,
        0.25f, 1.5f, -2.0f,  3.0f,
    };
    const float x[4] = { 0.5f, -1.0f, 2.0f, 0.25f };
    float ref_f32[3] = { 0 };
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++)
            ref_f32[r] += W_f32[r * cols + c] * x[c];
    }
    run_matvec_case(gpu, W_f32, sizeof(W_f32), BN_GGUF_TENSOR_F32,
                    rows, cols, x, ref_f32);

    float xk[BN_QK_K];
    for (int i = 0; i < BN_QK_K; i++)
        xk[i] = (float)((i % 7) - 3) * 0.25f;

    BnBlockQ8_0 q8_0[8];
    memset(q8_0, 0, sizeof(q8_0));
    float ref_q8_0 = 0.0f;
    for (int b = 0; b < 8; b++) {
        q8_0[b].d = bn_fp32_to_fp16(0.5f);
        for (int i = 0; i < 32; i++) {
            q8_0[b].qs[i] = (int8_t)((i % 9) - 4);
            ref_q8_0 += 0.5f * (float)q8_0[b].qs[i] * xk[b * 32 + i];
        }
    }
    run_matvec_case(gpu, q8_0, sizeof(q8_0), BN_GGUF_TENSOR_Q8_0,
                    1, BN_QK_K, xk, &ref_q8_0);

    BnBlockQ4_0 q4_0[8];
    memset(q4_0, 0, sizeof(q4_0));
    float ref_q4_0 = 0.0f;
    for (int b = 0; b < 8; b++) {
        q4_0[b].d = bn_fp32_to_fp16(0.25f);
        for (int i = 0; i < 16; i++) {
            uint8_t lo = (uint8_t)(i & 15);
            uint8_t hi = (uint8_t)((15 - i) & 15);
            q4_0[b].qs[i] = (uint8_t)(lo | (hi << 4));
            ref_q4_0 += 0.25f * ((int)lo - 8) * xk[b * 32 + i];
            ref_q4_0 += 0.25f * ((int)hi - 8) * xk[b * 32 + 16 + i];
        }
    }
    run_matvec_case(gpu, q4_0, sizeof(q4_0), BN_GGUF_TENSOR_Q4_0,
                    1, BN_QK_K, xk, &ref_q4_0);

    BnBlockQ5_0 q5_0[8];
    memset(q5_0, 0, sizeof(q5_0));
    float ref_q5_0 = 0.0f;
    for (int b = 0; b < 8; b++) {
        q5_0[b].d = bn_fp32_to_fp16(0.125f);
        for (int i = 0; i < 16; i++) {
            uint8_t lo = (uint8_t)((i + b) & 15);
            uint8_t hi = (uint8_t)((31 - i - b) & 15);
            int q0 = (int)lo - 16;
            int q1 = (int)(hi | 16) - 16;
            q5_0[b].qs[i] = (uint8_t)(lo | (hi << 4));
            q5_0[b].qh[(i + 16) / 8] |= (uint8_t)(1u << ((i + 16) & 7));
            ref_q5_0 += 0.125f * (float)q0 * xk[b * 32 + i];
            ref_q5_0 += 0.125f * (float)q1 * xk[b * 32 + 16 + i];
        }
    }
    run_matvec_case(gpu, q5_0, sizeof(q5_0), BN_GGUF_TENSOR_Q5_0,
                    1, BN_QK_K, xk, &ref_q5_0);

    BnBlockQ4K q4k;
    memset(&q4k, 0, sizeof(q4k));
    q4k.d = bn_fp32_to_fp16(0.25f);
    q4k.dmin = bn_fp32_to_fp16(0.0f);
    q4k.scales[0] = 3;
    q4k.scales[1] = 5;
    q4k.scales[2] = 7;
    q4k.scales[3] = 11;
    float ref_q4k = 0.0f;
    for (int pair = 0; pair < 2; pair++) {
        float scale_lo = pair == 0 ? 3.0f : 7.0f;
        float scale_hi = pair == 0 ? 5.0f : 11.0f;
        for (int i = 0; i < 32; i++) {
            uint8_t lo = (uint8_t)((i + pair) & 15);
            uint8_t hi = (uint8_t)((15 - i + pair) & 15);
            int base = pair * 64;
            q4k.qs[pair * 32 + i] = (uint8_t)(lo | (hi << 4));
            ref_q4k += 0.25f * scale_lo * (float)lo * xk[base + i];
            ref_q4k += 0.25f * scale_hi * (float)hi * xk[base + 32 + i];
        }
    }
    run_matvec_case(gpu, &q4k, sizeof(q4k), BN_GGUF_TENSOR_Q4_K,
                    1, BN_QK_K, xk, &ref_q4k);

    BnBlockQ5K q5k;
    memset(&q5k, 0, sizeof(q5k));
    q5k.d = bn_fp32_to_fp16(0.125f);
    q5k.dmin = bn_fp32_to_fp16(0.0f);
    q5k.scales[0] = 2;
    float ref_q5k = 0.0f;
    for (int i = 0; i < 32; i++) {
        uint8_t q = (uint8_t)(i & 15);
        q5k.qs[i] = q;
        ref_q5k += 0.125f * 2.0f * (float)q * xk[i];
    }
    run_matvec_case(gpu, &q5k, sizeof(q5k), BN_GGUF_TENSOR_Q5_K,
                    1, BN_QK_K, xk, &ref_q5k);

    BnBlockQ6K q6k;
    memset(&q6k, 0, sizeof(q6k));
    q6k.d = bn_fp32_to_fp16(0.5f);
    q6k.scales[0] = 3;
    float ref_q6k = 0.0f;
    for (int i = 0; i < 16; i++) {
        uint8_t q = (uint8_t)(32 + (i % 7));
        q6k.ql[i] = (uint8_t)(q & 15);
        q6k.qh[i] = (uint8_t)((q >> 4) & 3);
        ref_q6k += 0.5f * 3.0f * (float)((int)q - 32) * xk[i];
    }
    run_matvec_case(gpu, &q6k, sizeof(q6k), BN_GGUF_TENSOR_Q6_K,
                    1, BN_QK_K, xk, &ref_q6k);

    BnBlockQ8K q8k;
    memset(&q8k, 0, sizeof(q8k));
    q8k.d = 0.25f;
    float ref_q8k = 0.0f;
    for (int i = 0; i < BN_QK_K; i++) {
        q8k.qs[i] = (int8_t)((i % 11) - 5);
        ref_q8k += 0.25f * (float)q8k.qs[i] * xk[i];
    }
    run_matvec_case(gpu, &q8k, sizeof(q8k), BN_GGUF_TENSOR_Q8_K,
                    1, BN_QK_K, xk, &ref_q8k);

    bn_gpu_cuda_destroy(gpu);

    printf("CUDA backend test PASSED\n");
    return 0;
#endif
}
