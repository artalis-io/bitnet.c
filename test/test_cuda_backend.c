#include "cuda_prefix_attention_reference.h"
#include "cuda_hc_combine_reference.h"
#include "cuda_hc_mixer_reference.h"
#include "cuda_hc_rmsnorm_reference.h"
#include "cuda_signed_sqrt_gate_reference.h"
#include "cuda_gelu_reference.h"
#include "cuda_ssm_reference.h"
#include "gpu_cuda.h"
#include "gguf.h"
#include "model_config.h"
#include "moe_types.h"
#include "quant.h"
#include "transformer_math_internal.h"
#include "../src/gpu_shader_ir_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

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
    for (int i = 0; i < rows; i++) {
        if (fabsf(out[i] - ref[i]) >= 1e-4f)
            fprintf(stderr,
                    "CUDA matvec mismatch: type=%d row=%d got=%g ref=%g diff=%g\n",
                    type, i, out[i], ref[i], fabsf(out[i] - ref[i]));
    }
    expect_close(out, ref, rows);
}

static void run_wide_q4k_matmul_case(BnGPUBackend *gpu, int n_tokens) {
    enum {
        rows = 130,
        cols = 512,
        n_blocks = rows * cols / BN_QK_K,
    };
    BnBlockQ4K *weights =
        (BnBlockQ4K *)calloc((size_t)n_blocks, sizeof(*weights));
    float *x = (float *)malloc(
        (size_t)n_tokens * cols * sizeof(*x));
    float *out = (float *)malloc(
        (size_t)n_tokens * rows * sizeof(*out));
    float *ref = (float *)malloc(
        (size_t)n_tokens * rows * sizeof(*ref));
    assert(weights && x && out && ref);
    for (int b = 0; b < n_blocks; b++) {
        weights[b].d = bn_fp32_to_fp16(0.00390625f * (float)(1 + b % 5));
        weights[b].dmin = bn_fp32_to_fp16(0.001953125f * (float)(b % 3));
        for (int i = 0; i < 12; i++)
            weights[b].scales[i] = (uint8_t)(3 + ((b * 17 + i * 11) & 31));
        for (int i = 0; i < 128; i++)
            weights[b].qs[i] = (uint8_t)(b * 29 + i * 13);
    }
    for (int i = 0; i < n_tokens * cols; i++)
        x[i] = sinf((float)(i % 257) * 0.03125f);
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (int b = 0; b < cols / BN_QK_K; b++) {
                float dequant[BN_QK_K];
                bn_quant_dequant_q4k(
                    &weights[r * (cols / BN_QK_K) + b], dequant);
                for (int k = 0; k < BN_QK_K; k++)
                    sum += dequant[k] * x[t * cols + b * BN_QK_K + k];
            }
            ref[t * rows + r] = sum;
        }
    }

    void *buf = gpu->buffer_create(
        gpu->ctx, weights, (size_t)n_blocks * sizeof(*weights),
        BN_GGUF_TENSOR_Q4_K, rows, cols);
    assert(buf != NULL);
    assert(bn_gpu_backend_matmul(gpu, out, buf, x, rows, cols,
                                 n_tokens, BN_GGUF_TENSOR_Q4_K) == 0);
    for (int i = 0; i < n_tokens * rows; i++) {
        if (fabsf(out[i] - ref[i]) >= 1.0f)
            fprintf(stderr, "wide matmul tokens=%d i=%d out=%g ref=%g\n", n_tokens, i, out[i], ref[i]);
        assert(fabsf(out[i] - ref[i]) < 1.0f);
    }

    gpu->buffer_destroy(gpu->ctx, buf);
    free(ref);
    free(out);
    free(x);
    free(weights);
}

static void run_wide_q5k_matmul_case(BnGPUBackend *gpu, int n_tokens) {
    enum {
        rows = 130,
        cols = 512,
        n_blocks = rows * cols / BN_QK_K,
    };
    BnBlockQ5K *weights =
        (BnBlockQ5K *)calloc((size_t)n_blocks, sizeof(*weights));
    float *x = (float *)malloc((size_t)n_tokens * cols * sizeof(*x));
    float *out = (float *)malloc((size_t)n_tokens * rows * sizeof(*out));
    float *ref = (float *)malloc((size_t)n_tokens * rows * sizeof(*ref));
    assert(weights && x && out && ref);

    for (int b = 0; b < n_blocks; b++) {
        weights[b].d = bn_fp32_to_fp16(0.00390625f * (float)(1 + b % 5));
        weights[b].dmin = bn_fp32_to_fp16(0.001953125f * (float)(b % 3));
        for (int i = 0; i < 12; i++)
            weights[b].scales[i] = (uint8_t)(3 + ((b * 17 + i * 11) & 31));
        for (int i = 0; i < 128; i++)
            weights[b].qs[i] = (uint8_t)(b * 29 + i * 13);
        for (int i = 0; i < 32; i++)
            weights[b].qh[i] = (uint8_t)(b * 19 + i * 7);
    }
    for (int i = 0; i < n_tokens * cols; i++)
        x[i] = sinf((float)(i % 257) * 0.03125f);
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (int b = 0; b < cols / BN_QK_K; b++) {
                float dequant[BN_QK_K];
                bn_quant_dequant_q5k(
                    &weights[r * (cols / BN_QK_K) + b], dequant);
                for (int k = 0; k < BN_QK_K; k++)
                    sum += dequant[k] * x[t * cols + b * BN_QK_K + k];
            }
            ref[t * rows + r] = sum;
        }
    }

    void *buf = gpu->buffer_create(
        gpu->ctx, weights, (size_t)n_blocks * sizeof(*weights),
        BN_GGUF_TENSOR_Q5_K, rows, cols);
    assert(buf != NULL);
    assert(bn_gpu_backend_matmul(gpu, out, buf, x, rows, cols,
                                 n_tokens, BN_GGUF_TENSOR_Q5_K) == 0);
    for (int i = 0; i < n_tokens * rows; i++) {
        if (fabsf(out[i] - ref[i]) >= 1.0f)
            fprintf(stderr, "wide matmul tokens=%d i=%d out=%g ref=%g\n", n_tokens, i, out[i], ref[i]);
        assert(fabsf(out[i] - ref[i]) < 1.0f);
    }

    gpu->buffer_destroy(gpu->ctx, buf);
    free(ref);
    free(out);
    free(x);
    free(weights);
}

static void run_wide_q6k_prefill_case(BnGPUBackend *gpu, int n_tokens) {
    enum { rows = 512, cols = 8192, blocks_per_row = cols / BN_QK_K };
    const int selected[] = {0, blocks_per_row / 2, blocks_per_row - 1};
    size_t n_blocks = (size_t)rows * blocks_per_row;
    BnBlockQ6K *weights = (BnBlockQ6K *)calloc(n_blocks, sizeof(*weights));
    float *x = (float *)malloc((size_t)n_tokens * cols * sizeof(*x));
    float *out = (float *)malloc((size_t)n_tokens * rows * sizeof(*out));
    assert(weights && x && out);
    for (int r = 0; r < rows; r++) {
        for (size_t s = 0; s < sizeof(selected) / sizeof(selected[0]); s++) {
            int b = selected[s];
            BnBlockQ6K *block = &weights[r * blocks_per_row + b];
            block->d = bn_fp32_to_fp16(0.00390625f * (float)(1 + r % 3));
            for (int i = 0; i < 128; i++)
                block->ql[i] = (uint8_t)(r * 29 + b * 7 + i * 13);
            for (int i = 0; i < 64; i++)
                block->qh[i] = (uint8_t)(r * 11 + b * 17 + i * 3);
            for (int i = 0; i < 16; i++)
                block->scales[i] = (int8_t)((r + b + i * 5) % 15 - 7);
        }
    }
    for (int i = 0; i < n_tokens * cols; i++)
        x[i] = 0.5f * sinf((float)(i % 257) * 0.03125f);

    void *buf = gpu->buffer_create(gpu->ctx, weights,
        n_blocks * sizeof(*weights), BN_GGUF_TENSOR_Q6_K, rows, cols);
    assert(buf);
    assert(bn_gpu_backend_matmul(gpu, out, buf, x, rows, cols,
                                 n_tokens, BN_GGUF_TENSOR_Q6_K) == 0);
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            double expected = 0.0;
            for (size_t s = 0; s < sizeof(selected) / sizeof(selected[0]); s++) {
                int b = selected[s];
                float dequant[BN_QK_K];
                bn_quant_dequant_q6k(&weights[r * blocks_per_row + b],
                                     dequant);
                for (int i = 0; i < BN_QK_K; i++)
                    expected += (double)dequant[i] *
                        x[(size_t)t * cols + b * BN_QK_K + i];
            }
            assert(isfinite(out[(size_t)t * rows + r]));
            assert(fabs((double)out[(size_t)t * rows + r] - expected) < 0.5);
        }
    }
    gpu->buffer_destroy(gpu->ctx, buf);
    free(out);
    free(x);
    free(weights);
}

static void run_q4k_matmul_batch_case(BnGPUBackend *gpu) {
    enum { cols = BN_QK_K, n_tokens = 5, rows0 = 3, rows1 = 5 };
    BnBlockQ4K weights0[rows0];
    BnBlockQ4K weights1[rows1];
    float x[n_tokens * cols];
    float ref0[n_tokens * rows0];
    float ref1[n_tokens * rows1];
    float out0[n_tokens * rows0];
    float out1[n_tokens * rows1];

    for (int t = 0; t < n_tokens; t++)
        for (int i = 0; i < cols; i++)
            x[t * cols + i] = sinf((float)(t * 37 + i) * 0.03125f);
    for (int r = 0; r < rows0 + rows1; r++) {
        BnBlockQ4K *block = r < rows0 ? &weights0[r] : &weights1[r - rows0];
        memset(block, 0, sizeof(*block));
        block->d = bn_fp32_to_fp16(0.015625f * (float)(r + 1));
        for (int i = 0; i < 12; i++)
            block->scales[i] = (uint8_t)(3 + ((r * 11 + i * 7) & 31));
        for (int i = 0; i < 128; i++)
            block->qs[i] = (uint8_t)(r * 29 + i * 13);
    }

    void *buf0 = gpu->buffer_create(gpu->ctx, weights0, sizeof(weights0),
                                    BN_GGUF_TENSOR_Q4_K, rows0, cols);
    void *buf1 = gpu->buffer_create(gpu->ctx, weights1, sizeof(weights1),
                                    BN_GGUF_TENSOR_Q4_K, rows1, cols);
    assert(buf0 && buf1);
    assert(bn_gpu_backend_matmul(gpu, ref0, buf0, x, rows0, cols,
                                 n_tokens, BN_GGUF_TENSOR_Q4_K) == 0);
    assert(bn_gpu_backend_matmul(gpu, ref1, buf1, x, rows1, cols,
                                 n_tokens, BN_GGUF_TENSOR_Q4_K) == 0);
    BnGPUMatvecOp ops[2] = {
        { .out = out0, .W_buf = buf0, .rows = rows0, .cols = cols,
          .type = BN_GGUF_TENSOR_Q4_K },
        { .out = out1, .W_buf = buf1, .rows = rows1, .cols = cols,
          .type = BN_GGUF_TENSOR_Q4_K },
    };
    assert(bn_gpu_backend_matmul_batch(gpu, ops, 2, x, n_tokens, cols) == 0);
    for (int i = 0; i < n_tokens * rows0; i++)
        assert(fabsf(out0[i] - ref0[i]) < 2.0f);
    for (int i = 0; i < n_tokens * rows1; i++)
        assert(fabsf(out1[i] - ref1[i]) < 2.0f);
    gpu->buffer_destroy(gpu->ctx, buf1);
    gpu->buffer_destroy(gpu->ctx, buf0);
}

static void run_moe_prefill_asymmetric_residual_case(BnGPUBackend *gpu) {
    enum {
        n_tokens = 3,
        dim = BN_QK_K,
        hidden_dim = BN_QK_K,
        n_experts = 2,
        k = 1,
        expert_values = n_experts * dim * hidden_dim,
        expert_blocks = expert_values / BN_QK_K,
    };
    float router[n_experts * dim];
    float norm[dim];
    float input[n_tokens * dim];
    float output[n_tokens * dim];
    BnBlockQ4K *gate =
        (BnBlockQ4K *)calloc(expert_blocks, sizeof(*gate));
    BnBlockQ4K *up =
        (BnBlockQ4K *)calloc(expert_blocks, sizeof(*up));
    BnBlockQ6K *down =
        (BnBlockQ6K *)calloc(expert_blocks, sizeof(*down));
    assert(gate && up && down);
    memset(router, 0, sizeof(router));
    for (int i = 0; i < dim; i++) norm[i] = 1.0f;
    for (int i = 0; i < n_tokens * dim; i++)
        input[i] = 0.125f * (float)((i % 17) - 8);

    void *router_buf = gpu->buffer_create(
        gpu->ctx, router, sizeof(router), BN_GGUF_TENSOR_F32,
        n_experts, dim);
    void *gate_buf = gpu->buffer_create(
        gpu->ctx, gate, (size_t)expert_blocks * sizeof(*gate),
        BN_GGUF_TENSOR_Q4_K, n_experts * hidden_dim, dim);
    void *up_buf = gpu->buffer_create(
        gpu->ctx, up, (size_t)expert_blocks * sizeof(*up),
        BN_GGUF_TENSOR_Q4_K, n_experts * hidden_dim, dim);
    void *down_buf = gpu->buffer_create(
        gpu->ctx, down, (size_t)expert_blocks * sizeof(*down),
        BN_GGUF_TENSOR_Q6_K, n_experts * dim, hidden_dim);
    void *norm_buf = gpu->buffer_create(
        gpu->ctx, norm, sizeof(norm), BN_GGUF_TENSOR_F32, 1, dim);
    assert(router_buf && gate_buf && up_buf && down_buf && norm_buf);

    assert(bn_gpu_backend_moe_route_routed_ffn_batch_norm_resid(
               gpu, output, router_buf, gate_buf, up_buf, down_buf,
               NULL, NULL, NULL, NULL, norm_buf, input, n_tokens, dim,
               hidden_dim, n_experts, k, BN_GGUF_TENSOR_Q4_K,
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q6_K,
               BN_MODEL_ACTIVATION_SILU, 0, 0, 0, 0, 1e-6f, 1,
               1.0f) == 0);
    expect_close(output, input, n_tokens * dim);

    gpu->buffer_destroy(gpu->ctx, norm_buf);
    gpu->buffer_destroy(gpu->ctx, down_buf);
    gpu->buffer_destroy(gpu->ctx, up_buf);
    gpu->buffer_destroy(gpu->ctx, gate_buf);
    gpu->buffer_destroy(gpu->ctx, router_buf);
    free(down);
    free(up);
    free(gate);
}

static uint32_t f32_bits(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static int init_test_activations(BnGPUBackend *gpu, const BnConfig *cfg);

static void run_prefill_attention_case(BnGPUBackend *gpu, int kv_f16) {
    enum {
        n_tokens = 21,
        n_heads = 32,
        n_kv_heads = 16,
        head_size = 256,
        kv_mul = n_heads / n_kv_heads,
        kv_dim = n_kv_heads * head_size,
        q_values = n_tokens * n_heads * head_size,
        kv_values = n_tokens * kv_dim,
    };
    BnConfig cfg = {0};
    cfg.dim = cfg.hidden_dim = n_heads * head_size;
    cfg.vocab_size = 32; cfg.n_layers = 1; cfg.seq_len = n_tokens;
    cfg.n_heads = n_heads; cfg.head_size = head_size; cfg.kv_dim = kv_dim;
    cfg.kv_f16 = kv_f16; cfg.rope_theta = 10000.0f;
    assert(init_test_activations(gpu, &cfg) == 0);
    float *q = (float *)malloc((size_t)q_values * sizeof(float));
    float *k = (float *)malloc((size_t)kv_values * sizeof(float));
    float *v = (float *)malloc((size_t)kv_values * sizeof(float));
    float *got = (float *)malloc((size_t)q_values * sizeof(float));
    float *ref = (float *)calloc((size_t)q_values, sizeof(float));
    assert(q && k && v && got && ref);

    for (int i = 0; i < q_values; i++)
        q[i] = sinf((float)(i * 17 + 3));
    for (int i = 0; i < kv_values; i++) {
        k[i] = cosf((float)(i * 13 + 5));
        v[i] = 2.0f * sinf((float)(i * 7 + 11));
    }

    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < n_heads; h++) {
            const int kv_h = h / kv_mul;
            const float *qh = q + ((size_t)t * n_heads + h) * head_size;
            float scores[256];
            for (int j = 0; j <= t; j++) {
                const float *kh = k + (size_t)j * kv_dim +
                                  (size_t)kv_h * head_size;
                float score = 0.0f;
                for (int d = 0; d < head_size; d++)
                    score = fmaf(qh[d], kv_f16 ? bn_fp16_to_fp32(bn_fp32_to_fp16(kh[d])) : kh[d], score);
                scores[j] = score;
            }
            for (int j = t + 1; j < 256; j++) scores[j] = -INFINITY;
            bn_transformer_softmax_avx2(scores, 256);
            float *dst = ref + ((size_t)t * n_heads + h) * head_size;
            for (int j = 0; j <= t; j++) {
                const float *vh = v + (size_t)j * kv_dim +
                                  (size_t)kv_h * head_size;
                for (int d = 0; d < head_size; d++)
                    dst[d] = fmaf(scores[j], kv_f16 ? bn_fp16_to_fp32(bn_fp32_to_fp16(vh[d])) : vh[d], dst[d]);
            }
        }
    }

    assert(bn_gpu_backend_prefill_attention(
               gpu, got, q, k, v, n_tokens, n_heads, n_kv_heads,
               head_size, kv_mul, kv_dim, 1.0f, 0) == 0);
    float max_diff = 0.0f;
    double mse = 0.0;
    for (int i = 0; i < q_values; i++) {
        float diff = fabsf(got[i] - ref[i]);
        if (diff > max_diff) max_diff = diff;
        mse += (double)diff * (double)diff;
    }
    fprintf(stderr, "CUDA prefill attention kv_f16=%d: max_diff=%g rms=%g\n",
            kv_f16, max_diff, sqrt(mse / (double)q_values));
    if (kv_f16) {
        /* Independent llama.cpp CUDA flash-attention oracle, 3d3d7c8181,
         * SM120. FP16 MMA intentionally differs from scalar FP32 attention. */
        uint64_t hash = UINT64_C(14695981039346656037);
        for (int i = 0; i < q_values; i++)
            hash = (hash ^ f32_bits(got[i])) * UINT64_C(1099511628211);
        assert(hash == UINT64_C(0xe59860896f8c5e2b));
    } else {
        assert(max_diff < 1e-4f);
    }

    free(ref);
    free(got);
    free(v);
    free(k);
    free(q);
    gpu->free_activations(gpu->ctx);
}

static int init_test_activations(BnGPUBackend *gpu, const BnConfig *cfg) {
    int rope_count = cfg->head_size / 2;
    float rope[256];
    assert(rope_count <= (int)(sizeof(rope) / sizeof(rope[0])));
    for (int i = 0; i < rope_count; i++)
        rope[i] = powf(cfg->rope_theta, -2.0f * (float)i /
                                       (float)cfg->head_size);
    BnGPUActivationPlan plan = {
        .dim = cfg->dim,
        .n_layers = cfg->n_layers,
        .seq_len = cfg->seq_len,
        .kv_dim = cfg->kv_dim,
        .n_heads = cfg->n_heads,
        .head_size = cfg->head_size,
        .vocab_size = cfg->vocab_size,
        .hyper_connection_count = cfg->hyper_connection_count,
        .hyper_connection_rank = cfg->hyper_connection_rank,
        .kv_f16 = cfg->kv_f16,
        .attention_layer_count = cfg->n_layers,
        .xb2_elements = cfg->dim,
        .hb_elements = cfg->hidden_dim,
        .rope_frequencies = rope,
        .rope_frequency_count = rope_count,
    };
    return bn_gpu_backend_init_activations(gpu, &plan);
}

static void run_mmq_input_rounding_case(BnGPUBackend *gpu) {
    /* llama.cpp 3d3d7c8181 CUDA MMQ oracle, SM120: x = amax/2 must
     * quantize to +/-64. Correctly-rounded division instead gives +/-63
     * for these inputs. Sparse weights isolate that integer decision. */
    enum { cols = BN_QK_K, max_tokens = 21 };
    const uint32_t maxima[] = { UINT32_C(0x41515330), UINT32_C(0x417a1b5b) };
    const uint32_t half_maxima[] = { UINT32_C(0x40d15330), UINT32_C(0x40fa1b5b) };
    const uint32_t scales_f32[] = { UINT32_C(0x3dd2f921), UINT32_C(0x3dfc1381) };
    const uint16_t scales_f16[] = { 0x2e98, 0x2fe1 };
    const int batch_sizes[] = { 9, 16, 21 };
    BnBlockQ4K q4 = {0};
    BnBlockQ5K q5 = {0};
    BnBlockQ6K q6 = {0};
    q4.d = q5.d = q6.d = bn_fp32_to_fp16(1.0f);
    q4.scales[0] = q5.scales[0] = 1;
    q4.qs[0] = q5.qs[0] = 1;
    /* Q6_K stores signed zero as 32 (high bits 2, low bits 0). */
    memset(q6.qh, 0xaa, sizeof(q6.qh));
    q6.ql[0] = 1;
    q6.scales[0] = 1;
    const void *weights[] = { &q4, &q5, &q6 };
    const size_t sizes[] = { sizeof(q4), sizeof(q5), sizeof(q6) };
    const int types[] = { BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K,
                          BN_GGUF_TENSOR_Q6_K };
    float input[max_tokens * cols] = {0};
    for (int t = 0; t < max_tokens; t++) {
        int c = t % 5;
        if (c == 4) continue;
        int index = c / 2;
        memcpy(&input[t * cols], &half_maxima[index], sizeof(float));
        memcpy(&input[t * cols + 1], &maxima[index], sizeof(float));
        if (c & 1) input[t * cols] = -input[t * cols];
    }
    for (int f = 0; f < 3; f++) {
        void *weight = gpu->buffer_create_quant_only(
            gpu->ctx, weights[f], sizes[f], types[f], 1, cols);
        assert(weight);
        for (size_t b = 0; b < sizeof(batch_sizes) / sizeof(batch_sizes[0]); b++) {
            int tokens = batch_sizes[b];
            float output[max_tokens + 2];
            output[0] = 12345.0f;
            output[tokens + 1] = -67890.0f;
            assert(gpu->matmul(gpu->ctx, output + 1, weight, input,
                               1, cols, tokens, types[f]) == 0);
            assert(output[0] == 12345.0f && output[tokens + 1] == -67890.0f);
            for (int t = 0; t < tokens; t++) {
                int c = t % 5;
                float expected = 0.0f;
                if (c != 4) {
                    float scale;
                    if (f == 2) memcpy(&scale, &scales_f32[c / 2], sizeof(scale));
                    else scale = bn_fp16_to_fp32(scales_f16[c / 2]);
                    expected = (c & 1 ? -64.0f : 64.0f) * scale;
                }
                assert(isfinite(output[t + 1]));
                assert(fabsf(output[t + 1] - expected) < 1e-6f);
            }
        }
        gpu->buffer_destroy(gpu->ctx, weight);
    }
    printf("CUDA MMQ input rounding cases PASSED\n");
}

static void run_q4k_mmq_original_sum_case(BnGPUBackend *gpu) {
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_Q4_K, BN_QUANT_CAP_GPU_MMQ_ORIGINAL_SUM));
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_Q5_K, BN_QUANT_CAP_GPU_MMQ_ORIGINAL_SUM));
    assert(!bn_quant_format_has_cap(BN_GGUF_TENSOR_Q6_K, BN_QUANT_CAP_GPU_MMQ_ORIGINAL_SUM));
    enum { rows = 9, cols = 512, tokens = 16, blocks_per_row = cols / BN_QK_K };
    BnBlockQ4K weights[rows * blocks_per_row];
    float input[tokens * cols], output[tokens * rows];
    for (int b = 0; b < rows * blocks_per_row; b++) {
        weights[b].d = bn_fp32_to_fp16(0.0013f * (float)(1 + b % 5));
        weights[b].dmin = bn_fp32_to_fp16(0.0007f * (float)(1 + b % 3));
        for (int j = 0; j < 12; j++) weights[b].scales[j] = (uint8_t)(b * 13 + j * 7);
        for (int j = 0; j < 128; j++) weights[b].qs[j] = (uint8_t)(b * 31 + j * 17);
    }
    for (int i = 0; i < tokens * cols; i++)
        input[i] = 0.7f * sinf((float)(i * 13 + 5)) + 0.013f;
    void *weight = gpu->buffer_create_quant_only(gpu->ctx, weights, sizeof(weights),
                                                BN_GGUF_TENSOR_Q4_K, rows, cols);
    assert(weight);
    assert(gpu->matmul(gpu->ctx, output, weight, input, rows, cols,
                       tokens, BN_GGUF_TENSOR_Q4_K) == 0);
    for (int t = 0; t < tokens; t++) {
        for (int r = 0; r < rows; r++) {
            double expected = 0.0;
            for (int b = 0; b < blocks_per_row; b++) {
                const BnBlockQ4K *w = &weights[r * blocks_per_row + b];
                for (int g = 0; g < 8; g++) {
                    const float *x = input + t * cols + b * BN_QK_K + g * 32;
                    float amax = 0.0f, sum = 0.0f;
                    for (int j = 0; j < 32; j++) {
                        amax = fmaxf(amax, fabsf(x[j]));
                        sum += x[j];
                    }
                    float inv = 127.0f / amax;
                    float d = bn_fp16_to_fp32(bn_fp32_to_fp16(1.0f / inv));
                    sum = bn_fp16_to_fp32(bn_fp32_to_fp16(sum));
                    int sc = g < 4 ? w->scales[g] & 63
                        : (w->scales[g + 4] & 15) | ((w->scales[g - 4] >> 6) << 4);
                    int mn = g < 4 ? w->scales[g + 4] & 63
                        : (w->scales[g + 4] >> 4) | ((w->scales[g] >> 6) << 4);
                    float wd = bn_fp16_to_fp32(bn_fp32_to_fp16(bn_fp16_to_fp32(w->d) * sc));
                    float wm = bn_fp16_to_fp32(bn_fp32_to_fp16(bn_fp16_to_fp32(w->dmin) * mn));
                    int dot = 0;
                    for (int j = 0; j < 32; j++) {
                        int q = (w->qs[(g / 2) * 32 + j] >> ((g & 1) * 4)) & 15;
                        dot += q * (int)roundf(x[j] * inv);
                    }
                    expected += (double)wd * d * dot - (double)wm * sum;
                }
            }
            assert(isfinite(output[t * rows + r]));
            assert(fabs(output[t * rows + r] - expected) < 1e-5);
        }
    }
    float batched[2][tokens * rows];
    BnGPUMatvecOp ops[2] = {
        { .out = batched[0], .W_buf = weight, .rows = rows, .cols = cols,
          .type = BN_GGUF_TENSOR_Q4_K },
        { .out = batched[1], .W_buf = weight, .rows = rows, .cols = cols,
          .type = BN_GGUF_TENSOR_Q4_K },
    };
    assert(bn_gpu_backend_matmul_batch(gpu, ops, 2, input, tokens, cols) == 0);
    expect_close(batched[0], output, tokens * rows);
    expect_close(batched[1], output, tokens * rows);
    /* Short prompts use the padded packed MMQ tile as well. Their rows must
     * agree with the independently checked full tile. */
    float short_output[14 * rows];
    const int short_counts[] = {8, 14};
    for (size_t i = 0; i < sizeof(short_counts) / sizeof(short_counts[0]); i++) {
        int n = short_counts[i];
        assert(gpu->matmul(gpu->ctx, short_output, weight, input, rows, cols,
                           n, BN_GGUF_TENSOR_Q4_K) == 0);
        expect_close(short_output, output, n * rows);
    }
    gpu->buffer_destroy(gpu->ctx, weight);
}

/* Hash complete FP32 tensors by bits, independent of host byte order. */
static uint64_t cuda_test_float_bits_hash(const float *values, size_t n) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, values + i, sizeof(bits));
        hash = (hash ^ bits) * UINT64_C(1099511628211);
    }
    return hash;
}


/* Reassemble independent reference outputs while checking untouched storage.
 * Layouts include two/three outputs, nonzero offsets, and shared buffers. */
static void run_kquant_split_decode_reference(BnGPUBackend *gpu, void *buffer,
                                              const float *input, int cols,
                                              int type, uint64_t reference_hash) {
    enum { rows = 37, cap = rows + 32 };
    const int buffers[3] = { BN_GPU_VALUE_HB, BN_GPU_VALUE_HB2, BN_GPU_VALUE_XB2 };
    float sentinel[cap], values[3][cap], packed[rows];
    for (int j = 0; j < cap; j++) sentinel[j] = -12345.0f;
    for (int layout = 0; layout < 6; layout++) {
        int split0 = layout == 0 ? 1 : rows / 3;
        int split1 = layout % 2 ? 2 * rows / 3 : 0;
        int offset1 = layout >= 4 ? split0 + 7 : layout >= 2 ? 7 : 0;
        int offset2 = layout >= 4 ? split1 + 19 : layout >= 2 ? 19 : 0;
        for (unsigned flags = 0; flags <= 1; flags++) {
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X,
                                         input, (size_t)cols * 4, 0) == 0);
            for (int k = 0; k < 3; k++)
                assert(gpu->write_activation(gpu->ctx, buffers[k],
                                             sentinel, sizeof(sentinel), 0) == 0);
            BnGPUOp op = {0};
            op.op_code = BN_GPU_CODE_MATVEC_SPLIT;
            op.buf_in = BN_GPU_VALUE_X;
            op.buf_out = buffers[0];
            op.buf_aux = layout >= 4 ? buffers[0] : buffers[1];
            op.rows = layout >= 4 ? buffers[0] : buffers[2];
            op.W_buf = buffer; op.type = type; op.cols = cols; op.flags = flags;
            op.p[0] = rows; op.p[1] = cols;
            op.p[2] = split0; op.p[3] = split1;
            op.p[6] = offset1; op.p[7] = offset2;
            assert(gpu->execute(gpu->ctx, &op, 1, buffers[0], values[0], cap) == 0);
            for (int k = 0; k < 3; k++)
                assert(gpu->read_activation(gpu->ctx, buffers[k], values[k],
                                            sizeof(values[k]), 0) == 0);
            unsigned char written[3][cap] = {{0}};
            for (int k = 0; k < 3; k++) {
                int dest = layout >= 4 ? 0 : k;
                int first = k == 0 ? 0 : k == 1 ? split0 : split1;
                int count = k == 0 ? split0 : k == 1
                    ? (split1 ? split1 : rows) - split0 : split1 ? rows - split1 : 0;
                int offset = k == 0 ? 0 : k == 1 ? offset1 : offset2;
                for (int j = 0; j < count; j++) {
                    assert(!written[dest][offset + j]);
                    written[dest][offset + j] = 1;
                    packed[first + j] = values[dest][offset + j];
                }
            }
            for (int k = 0; k < 3; k++)
                for (int j = 0; j < cap; j++)
                    if (!written[k][j]) assert(values[k][j] == sentinel[j]);
            uint64_t hash = cuda_test_float_bits_hash(packed, rows);
            if (hash != reference_hash)
                fprintf(stderr, "split type=%d cols=%d layout=%d flags=%u hash=%016llx expected=%016llx\n",
                        type, cols, layout, flags, (unsigned long long)hash,
                        (unsigned long long)reference_hash);
            assert(hash == reference_hash);
        }
    }
}

static void fill_norm_residual_reference(float *x, float *w, float *r, float *nw,
                                         int n, unsigned seed) {
    unsigned state = seed + 17u;
    for (int i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        x[i] = (float)((int)((state >> 8) % 20001) - 10000) / 8192.0f;
        state = state * 1664525u + 1013904223u;
        r[i] = (float)((int)((state >> 8) % 20001) - 10000) / 4096.0f;
        w[i] = (float)(i % 19 + 1) / 16.0f;
        nw[i] = (float)(i % 13 + 1) / 8.0f;
    }
}

static void run_norm_residual_reference_case(BnGPUBackend *gpu) {
    /* Independent fused/unfused GGML CUDA graphs, llama.cpp commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e, SM120. */
    const struct { int n; unsigned seed; uint64_t fused[2], unfused[2]; } cases[] = {
        {32, 0, {UINT64_C(0x6acfb1fe0ee9d2bd), UINT64_C(0x899e984bc0875366)}, {UINT64_C(0xe92e206015590b5d), UINT64_C(0x92a6ec00c914e014)}},
        {32, 23, {UINT64_C(0xa0e3c6bbaa0df369), UINT64_C(0xa00dde1f24f3fd15)}, {UINT64_C(0xa2a024dbf60cb561), UINT64_C(0xe511f5bd5fd71faf)}},
        {128, 0, {UINT64_C(0x8887959174770dbd), UINT64_C(0xa6d94ae6fbd9f0f7)}, {UINT64_C(0x60bd462ee34af76b), UINT64_C(0xf71cbfac6a2d7b1f)}},
        {128, 23, {UINT64_C(0xf35c164530e7674f), UINT64_C(0xa68571e7dbe4b4df)}, {UINT64_C(0x889588ef320deecf), UINT64_C(0xe2a1fc382667dff6)}},
        {256, 0, {UINT64_C(0x59e9d8a7bc5443be), UINT64_C(0xba5bb029cf730b85)}, {UINT64_C(0x28790fe1231ceebe), UINT64_C(0x13606235f26ce967)}},
        {256, 23, {UINT64_C(0x0c3b3c485c5dd82e), UINT64_C(0x7c92c24a616f1599)}, {UINT64_C(0x5884a504eaff7a33), UINT64_C(0x7efc536346d9ce2f)}},
        {512, 0, {UINT64_C(0xe3e686fa0f38e33f), UINT64_C(0x878bba7a6e6d9871)}, {UINT64_C(0xeb550a2b72f8e648), UINT64_C(0x23ef3a1bfd3a4aa3)}},
        {512, 23, {UINT64_C(0x048acdd464e2714b), UINT64_C(0x42f419fa99e5f320)}, {UINT64_C(0x6da8900eb4bebd06), UINT64_C(0x12bb60079bb6c8a4)}},
        {768, 0, {UINT64_C(0x848e7cf7c082c398), UINT64_C(0x0c7fd531a0dd4b94)}, {UINT64_C(0x9f0bdf0c2aea3320), UINT64_C(0xffd2cb294b42f7aa)}},
        {768, 23, {UINT64_C(0x125501f4af5487b5), UINT64_C(0x912e486a145287c1)}, {UINT64_C(0x0fc4bd0e2bc6c919), UINT64_C(0xd5c0be536d61d7f2)}},
        {1024, 0, {UINT64_C(0xedbc1584cba5da06), UINT64_C(0x83fa2328e663a34b)}, {UINT64_C(0x0eadc3d55090d1c2), UINT64_C(0x7cab3d08ad59498b)}},
        {1024, 23, {UINT64_C(0xbd61c867b3ad9eb4), UINT64_C(0x0a9c57b424a9f600)}, {UINT64_C(0xc6733bfc50aa71d6), UINT64_C(0x643365288d9f7fcb)}},
        {1536, 0, {UINT64_C(0x46e660f14c7b585b), UINT64_C(0xb3818d2cda88a77a)}, {UINT64_C(0x5987b4126af84b25), UINT64_C(0xa98aceb70f99dfd4)}},
        {1536, 23, {UINT64_C(0xa1412b26401fd18a), UINT64_C(0x7d469747c866d44e)}, {UINT64_C(0xd7ba1fac1c2cc2dd), UINT64_C(0x500afca2f482d8da)}},
        {2048, 0, {UINT64_C(0x011194d16b321134), UINT64_C(0x3c18058737e0af9b)}, {UINT64_C(0xfc412506fdb148f8), UINT64_C(0x95747cff76cb457f)}},
        {2048, 23, {UINT64_C(0xaa940729f3908af9), UINT64_C(0xc42ef65a8ffa1dcd)}, {UINT64_C(0xa1c619fab0623787), UINT64_C(0xb8d4503a79b82d85)}},
        {4096, 0, {UINT64_C(0xb2decdb11d16db91), UINT64_C(0x0643bf527b7f8812)}, {UINT64_C(0x7855be5e04d0e414), UINT64_C(0x29251e5f1c3db1c5)}},
        {4096, 23, {UINT64_C(0x59a032582b4ab251), UINT64_C(0xb7adf60217cf14a0)}, {UINT64_C(0x4d0d1bc1a2a90f98), UINT64_C(0xcae0c56f153fb099)}},
        {5376, 0, {UINT64_C(0xce08629f4f7c7475), UINT64_C(0xc2257c1e9a498358)}, {UINT64_C(0xf044ca59d16edad5), UINT64_C(0x4f9571f2dc160f2e)}},
        {5376, 23, {UINT64_C(0xb770ea71cf19fbaa), UINT64_C(0x6d6457de48d4406c)}, {UINT64_C(0x72cd654512e15f63), UINT64_C(0x569320be2875ccc7)}},
        {8192, 0, {UINT64_C(0xd773c84a7db7de97), UINT64_C(0x744b79fd9d055d8e)}, {UINT64_C(0xf5d6d36e01217783), UINT64_C(0x8edf13883fe05659)}},
        {8192, 23, {UINT64_C(0x2985d5f07abcd433), UINT64_C(0x42fb413347ba4eac)}, {UINT64_C(0xce12317495ecd66f), UINT64_C(0x6a4cccd94019abc4)}},
        {12288, 0, {UINT64_C(0x4d9f6571f1b3abb6), UINT64_C(0x75e95554b783669b)}, {UINT64_C(0xb1c975dfdb03d106), UINT64_C(0xa85cf5fd25489c8c)}},
        {12288, 23, {UINT64_C(0x1bf2b6bde887d8e5), UINT64_C(0x359f282f7d3a793a)}, {UINT64_C(0x4a815d5c53fee6f4), UINT64_C(0x0f78aec9e7175b9a)}},
    };
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int n = cases[c].n;
        float *data = malloc((size_t)(n * 8 + 2) * sizeof(float));
        assert(data);
        float *x = data, *w = x + n, *r = w + n, *nw = r + n;
        float *output = nw + n, *saved = output + n + 2, *intermediate = saved + n;
        fill_norm_residual_reference(x, w, r, nw, n, cases[c].seed);
        float eps = cases[c].seed ? 1e-5f : 1e-6f;
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = n;
        plan.vocab_size = 32;
        plan.n_layers = plan.seq_len = plan.n_heads = plan.head_size = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        void *wb = gpu->buffer_create(gpu->ctx, w, (size_t)n * 4,
                                     BN_GGUF_TENSOR_F32, n, 1);
        void *nb = gpu->buffer_create(gpu->ctx, nw, (size_t)n * 4,
                                     BN_GGUF_TENSOR_F32, n, 1);
        assert(wb && nb);
        for (int norm = 0; norm < 2; norm++) {
            BnGPUOp ops[3] = {{0}};
            ops[0].op_code = BN_GPU_CODE_RMSNORM;
            ops[0].buf_in = BN_GPU_VALUE_XB2;
            ops[0].buf_out = BN_GPU_VALUE_SCRATCH;
            ops[0].W_buf = wb; ops[0].p[0] = n;
            memcpy(&ops[0].p[1], &eps, 4);
            ops[1].op_code = norm ? BN_GPU_CODE_RESIDUAL_RMSNORM : BN_GPU_CODE_RESIDUAL_ADD;
            ops[1].buf_in = BN_GPU_VALUE_X; ops[1].buf_aux = BN_GPU_VALUE_SCRATCH;
            ops[1].buf_out = BN_GPU_VALUE_XB; ops[1].W_buf = nb; ops[1].p[0] = n;
            memcpy(&ops[1].p[1], &eps, 4);
            int result = norm ? BN_GPU_VALUE_XB : BN_GPU_VALUE_X;
            /* Fused, requested intermediate, and explicitly separate calls. */
            for (int path = 0; path < 3; path++) {
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB2, x, (size_t)n * 4, 0) == 0);
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, r, (size_t)n * 4, 0) == 0);
                output[0] = output[n + 1] = -12345.0f;
                if (path == 2) {
                    assert(gpu->execute(gpu->ctx, ops, 1, BN_GPU_VALUE_SCRATCH, output + 1, n) == 0);
                    assert(memcmp(output + 1, intermediate, (size_t)n * 4) == 0);
                    assert(gpu->execute(gpu->ctx, ops + 1, 1, result, output + 1, n) == 0);
                } else {
                    assert(gpu->execute(gpu->ctx, ops, 2,
                        path ? BN_GPU_VALUE_SCRATCH : result, output + 1, n) == 0);
                    if (path) {
                        memcpy(intermediate, output + 1, (size_t)n * 4);
                        assert(gpu->read_activation(gpu->ctx, result, output + 1, (size_t)n * 4, 0) == 0);
                    }
                }
                assert(cuda_test_float_bits_hash(output + 1, n) ==
                       (path ? cases[c].unfused[norm] : cases[c].fused[norm]));
                assert(output[0] == -12345.0f && output[n + 1] == -12345.0f);
            }
            /* Valid aliases and a later consumer must retain sequential behavior. */
            for (int alias = 0; alias < 3; alias++) {
                ops[1].buf_in = alias == 0 ? BN_GPU_VALUE_XB2
                    : alias == 1 ? BN_GPU_VALUE_SCRATCH : BN_GPU_VALUE_X;
                ops[2].op_code = BN_GPU_CODE_COPY;
                ops[2].buf_in = BN_GPU_VALUE_SCRATCH;
                ops[2].buf_out = BN_GPU_VALUE_HB;
                ops[2].p[2] = n;
                int count = alias == 2 ? 3 : 2;
                int readback = alias == 2 ? BN_GPU_VALUE_HB : norm ? BN_GPU_VALUE_XB : ops[1].buf_in;
                for (int combined = 0; combined < 2; combined++) {
                    for (int j = 0; j < n; j++) output[j + 1] = -12345.0f;
                    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SCRATCH,
                                                 output + 1, (size_t)n * 4, 0) == 0);
                    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB2, x, (size_t)n * 4, 0) == 0);
                    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, r, (size_t)n * 4, 0) == 0);
                    if (combined) {
                        assert(gpu->execute(gpu->ctx, ops, count, readback, output + 1, n) == 0);
                        assert(memcmp(output + 1, saved, (size_t)n * 4) == 0);
                    } else {
                        assert(gpu->execute(gpu->ctx, ops, 1, BN_GPU_VALUE_SCRATCH, output + 1, n) == 0);
                        assert(gpu->execute(gpu->ctx, ops + 1, count - 1, readback, saved, n) == 0);
                    }
                }
            }
        }
        gpu->buffer_destroy(gpu->ctx, wb); gpu->buffer_destroy(gpu->ctx, nb);
        free(data);
    }
    printf("CUDA norm residual reference PASSED\n");
}

static void fill_q4_decode_reference(BnBlockQ4K *w, float *x,
                                     int rows, int cols, unsigned seed) {
    unsigned state = seed + 17u;
    for (int b = 0; b < rows * (cols / BN_QK_K); b++) {
        w[b].d = (uint16_t)(0x2400 + (b % 29) * 13);
        w[b].dmin = (uint16_t)(0x2000 + (b % 17) * 19);
        for (int i = 0; i < 12; i++) {
            state = state * 1664525u + 1013904223u;
            w[b].scales[i] = (uint8_t)(state >> 24);
        }
        for (int i = 0; i < 128; i++) {
            state = state * 1664525u + 1013904223u;
            w[b].qs[i] = (uint8_t)(state >> 24);
        }
    }
    for (int i = 0; i < cols; i++) {
        state = state * 1664525u + 1013904223u;
        x[i] = (float)((int)((state >> 8) % 20001) - 10000) / 8192.0f;
    }
}

#include "cuda_kquant_reference.h"

#include "cuda_signed_mmq_fixture.h"
#include "cuda_signed_mmq_reference.h"
#include "cuda_iq3s_reference.h"
#include "cuda_iq4xs_reference.h"
#include "cuda_iq4nl_reference.h"
#include "cuda_mxfp4_fixture.h"
#include "cuda_mxfp4_reference.h"
#include "cuda_q3k_reference.h"
#include "cuda_q8_reference.h"
#include "cuda_q4_reference.h"

static void run_q4_original_sum_reference_case(BnGPUBackend *gpu) {
    for (size_t c=0;c<sizeof(cuda_q4_reference)/sizeof(cuda_q4_reference[0]);c++) {
        const BnCudaSignedMmqReference *r=&cuda_q4_reference[c];
        int rows=r->rows,cols=r->cols,nt=r->tokens;
        size_t nb=(size_t)rows*cols/32,nx=(size_t)cols*nt,ny=(size_t)rows*nt;
        BnBlockQ4_0 *w=malloc(nb*sizeof(*w));
        float *x=malloc(nx*sizeof(float)),*y=malloc(ny*sizeof(float));
        assert(w&&x&&y);
        for(size_t b=0;b<nb;b++) {
            w[b].d=(uint16_t)(0x2400+((b*13+r->seed)%20)*128);
            if(b%3==0)w[b].d|=0x8000;
            for(int j=0;j<16;j++)w[b].qs[j]=(uint8_t)(((b*7+j*3+r->seed)&15)|(((b*11+j*5+r->seed+1)&15)<<4));
        }
        for(size_t i=0;i<nx;i++)x[i]=(float)((int)((i*37+(unsigned)r->seed*11)%257)-128)/31.7f;
        for(int raw=0;raw<2;raw++) {
            void *buffer=raw?gpu->buffer_create_quant_only(gpu->ctx,w,nb*sizeof(*w),r->type,rows,cols):gpu->buffer_create(gpu->ctx,w,nb*sizeof(*w),r->type,rows,cols);
            assert(buffer);
            assert(gpu->matmul(gpu->ctx,y,buffer,x,rows,cols,nt,r->type)==0);
            if(cuda_test_float_bits_hash(y,ny)!=r->hash)
                fprintf(stderr,"Q4 case=%zu rows=%d cols=%d nt=%d raw=%d got=%016llx ref=%016llx first=%.9g\n",c,rows,cols,nt,raw,(unsigned long long)cuda_test_float_bits_hash(y,ny),(unsigned long long)r->hash,y[0]);
            assert(cuda_test_float_bits_hash(y,ny)==r->hash);
            if(nt==1) {
                assert(gpu->matvec(gpu->ctx,y,buffer,x,rows,cols,r->type)==0);
                assert(cuda_test_float_bits_hash(y,ny)==r->hash);
                BnGPUActivationPlan plan={0};
                plan.dim=cols;plan.hb_elements=rows+7;plan.xb2_elements=cols;
                plan.vocab_size=32;plan.n_layers=plan.seq_len=plan.n_heads=plan.head_size=1;
                assert(gpu->init_activations(gpu->ctx,&plan)==0);
                assert(gpu->write_activation(gpu->ctx,BN_GPU_VALUE_X,x,nx*sizeof(float),0)==0);
                BnGPUOp op={0};op.op_code=BN_GPU_CODE_MATVEC;op.W_buf=buffer;
                op.type=r->type;op.rows=rows;op.cols=cols;op.buf_in=BN_GPU_VALUE_X;op.buf_out=BN_GPU_VALUE_HB;
                for(int replay=0;replay<2;replay++) {
                    assert(gpu->execute(gpu->ctx,&op,1,BN_GPU_VALUE_HB,y,rows)==0);
                    assert(cuda_test_float_bits_hash(y,ny)==r->hash);
                }
                gpu->free_activations(gpu->ctx);
            }
            gpu->buffer_destroy(gpu->ctx,buffer);
        }
        free(w);free(x);free(y);
    }
    printf("CUDA Q4 original-sum reference PASSED\n");
}

static void run_signed_quant_reference_cases(BnGPUBackend *gpu,
        const BnCudaSignedMmqReference *cases, size_t n_cases) {
    for (size_t c = 0; c < n_cases; c++) {
        const BnCudaSignedMmqReference *ref = &cases[c];
        if (ref->type == BN_GGUF_TENSOR_MXFP4) {
            assert(bn_quant_format_has_cap(ref->type, BN_QUANT_CAP_GPU_MMQ_BLOCK32_FP4));
        } else if (ref->type == BN_GGUF_TENSOR_Q8_0) {
            assert(bn_quant_format_supports_gpu_dense_graph_native_quant(ref->type));
        } else {
        assert(bn_quant_format_has_cap(ref->type, BN_QUANT_CAP_GPU_MMQ_F32_SCALE));
        assert(bn_quant_format_has_cap(ref->type, ref->type == BN_GGUF_TENSOR_Q3_K
            ? BN_QUANT_CAP_GPU_MMQ_SUBBLOCK16 : BN_QUANT_CAP_GPU_MMQ_SUBBLOCK32));
        }
        size_t count = (size_t)ref->rows * ref->tokens;
        size_t nx = (size_t)ref->cols * ref->tokens;
        size_t capacity = count > nx ? count : nx;
        size_t elements = (size_t)ref->rows * ref->cols;
        size_t bytes = ref->type == BN_GGUF_TENSOR_MXFP4 ? elements / 32 * 17 :
            signed_mmq_fixture_bytes(ref->type, elements);
        void *weights = malloc(bytes);
        float *input = malloc(nx * sizeof(float));
        float *output = malloc((capacity + 2) * sizeof(float));
        assert(weights && input && output);
        if (ref->type == BN_GGUF_TENSOR_MXFP4)
            mxfp4_fixture_weights(weights, elements, ref->seed);
        else signed_mmq_fixture_weights(weights, ref->type, elements, ref->seed);
        if (ref->type == BN_GGUF_TENSOR_MXFP4) mxfp4_fixture_input(input, nx, ref->seed);
        else signed_mmq_fixture_input(input, nx, ref->seed);
        for (int quant_only = 0; quant_only < 2; quant_only++) {
            void *buffer = quant_only
                ? gpu->buffer_create_quant_only(gpu->ctx, weights, bytes,
                    ref->type, ref->rows, ref->cols)
                : gpu->buffer_create(gpu->ctx, weights, bytes,
                    ref->type, ref->rows, ref->cols);
            assert(buffer);
            for (int alias = 0; alias < 2; alias++) {
                for (size_t i = 0; i < capacity + 2; i++) output[i] = -12345.0f;
                if (alias) memcpy(output + 1, input, nx * sizeof(float));
                assert(gpu->matmul(gpu->ctx, output + 1, buffer,
                    alias ? output + 1 : input, ref->rows, ref->cols,
                    ref->tokens, ref->type) == 0);
                uint64_t hash = cuda_test_float_bits_hash(output + 1, count);
                if (hash != ref->hash)
                    fprintf(stderr, "signed MMQ type=%d rows=%d cols=%d nt=%d seed=%d quant_only=%d alias=%d: %016llx != %016llx\n",
                        ref->type, ref->rows, ref->cols, ref->tokens, ref->seed,
                        quant_only, alias, (unsigned long long)hash,
                        (unsigned long long)ref->hash);
                assert(hash == ref->hash);
                assert(output[0] == -12345.0f && output[capacity + 1] == -12345.0f);
                for (size_t i = count; i < capacity; i++)
                    assert(output[i + 1] == (alias && i < nx ? input[i] : -12345.0f));
            }
            if (ref->type == BN_GGUF_TENSOR_MXFP4 && ref->tokens == 1) {
                assert(gpu->matvec(gpu->ctx, output + 1, buffer, input,
                                   ref->rows, ref->cols, ref->type) == 0);
                assert(cuda_test_float_bits_hash(output + 1, count) == ref->hash);
            }
            gpu->buffer_destroy(gpu->ctx, buffer);
        }
        free(output); free(input); free(weights);
    }
}

static void run_signed_mmq_reference_case(BnGPUBackend *gpu) {
    run_signed_quant_reference_cases(gpu, cuda_signed_mmq_reference,
        sizeof(cuda_signed_mmq_reference) / sizeof(cuda_signed_mmq_reference[0]));
    printf("CUDA signed MMQ reference PASSED\n");
}

static void run_signed_quant_graph_reference_cases_offset(BnGPUBackend *gpu,
        const BnCudaSignedMmqReference *cases, size_t n_cases, int output_slot,
        int output_offset) {
    for (size_t c = 0; c < n_cases; c++) {
        const BnCudaSignedMmqReference *ref = &cases[c];
        if (ref->tokens != 1) continue;
        int rows = ref->rows, cols = ref->cols, capacity = rows + 7;
        size_t elements = (size_t)rows * cols;
        size_t bytes = ref->type == BN_GGUF_TENSOR_MXFP4 ? elements / 32 * 17 :
            signed_mmq_fixture_bytes(ref->type, elements);
        void *weights = malloc(bytes);
        float *input = malloc((size_t)cols * sizeof(float));
        float *output = malloc((size_t)capacity * sizeof(float));
        assert(weights && input && output);
        if (ref->type == BN_GGUF_TENSOR_MXFP4)
            mxfp4_fixture_weights(weights, elements, ref->seed);
        else signed_mmq_fixture_weights(weights, ref->type, elements, ref->seed);
        if (ref->type == BN_GGUF_TENSOR_MXFP4) mxfp4_fixture_input(input, cols, ref->seed);
        else signed_mmq_fixture_input(input, cols, ref->seed);
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = cols > capacity ? cols : capacity;
        plan.vocab_size = capacity;
        plan.n_layers = plan.seq_len = plan.n_heads = plan.head_size = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        for (int quant_only = 0; quant_only < 2; quant_only++) {
            void *buffer = quant_only
                ? gpu->buffer_create_quant_only(gpu->ctx, weights, bytes, ref->type, rows, cols)
                : gpu->buffer_create(gpu->ctx, weights, bytes, ref->type, rows, cols);
            assert(buffer);
            BnGPUOp ops[32] = {{0}};
            ops[0].op_code = BN_GPU_CODE_MATVEC;
            ops[0].buf_in = BN_GPU_VALUE_X;
            ops[0].buf_out = output_slot;
            ops[0].W_buf = buffer; ops[0].type = ref->type;
            ops[0].rows = rows; ops[0].cols = cols; ops[0].p[5] = output_offset;
            int previous = output_slot;
            for (int i = 1; i < 32; i++) {
                ops[i].op_code = BN_GPU_CODE_COPY;
                ops[i].buf_in = previous;
                ops[i].buf_out = i & 1 ? BN_GPU_VALUE_HB : BN_GPU_VALUE_XB;
                ops[i].p[0] = i == 1 ? output_offset : 0;
                ops[i].p[2] = rows;
                previous = ops[i].buf_out;
            }
            for (int repeat = 0; repeat < 4; repeat++) {
                for (int i = 0; i < capacity; i++) output[i] = -12345.0f;
                assert(gpu->write_activation(gpu->ctx, output_slot, output,
                    (size_t)capacity * sizeof(float), 0) == 0);
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, input,
                    (size_t)cols * sizeof(float), 0) == 0);
                assert(gpu->execute(gpu->ctx, ops, 32, previous, output, rows) == 0);
                assert(cuda_test_float_bits_hash(output, rows) == ref->hash);
                assert(gpu->read_activation(gpu->ctx, output_slot, output,
                    (size_t)capacity * sizeof(float), 0) == 0);
                assert(cuda_test_float_bits_hash(output + output_offset, rows) == ref->hash);
                for (int i = 0; i < output_offset; i++) assert(output[i] == -12345.0f);
                for (int i = rows + output_offset; i < capacity; i++) assert(output[i] == -12345.0f);
            }
            gpu->buffer_destroy(gpu->ctx, buffer);
        }
        free(weights); free(input); free(output);
    }
}

static void run_signed_quant_graph_reference_cases(BnGPUBackend *gpu,
        const BnCudaSignedMmqReference *cases, size_t n_cases, int output_slot) {
    run_signed_quant_graph_reference_cases_offset(gpu, cases, n_cases, output_slot, 3);
}

static void run_iq3s_reference_case(BnGPUBackend *gpu) {
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_IQ3_S,
        BN_QUANT_CAP_GPU_MMVQ_SUBBLOCK32_INT_SCALE));
    run_signed_quant_reference_cases(gpu, cuda_iq3s_reference,
        sizeof(cuda_iq3s_reference) / sizeof(cuda_iq3s_reference[0]));
    run_signed_quant_graph_reference_cases(gpu, cuda_iq3s_reference,
        sizeof(cuda_iq3s_reference) / sizeof(cuda_iq3s_reference[0]), BN_GPU_VALUE_LOGITS);
    printf("CUDA IQ3_S reference PASSED\n");
}

static void run_iq4xs_reference_case(BnGPUBackend *gpu) {
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_IQ4_XS,
        BN_QUANT_CAP_GPU_MMVQ_SUBBLOCK32_INT_SCALE));
    run_signed_quant_reference_cases(gpu, cuda_iq4xs_reference,
        sizeof(cuda_iq4xs_reference) / sizeof(cuda_iq4xs_reference[0]));
    run_signed_quant_graph_reference_cases(gpu, cuda_iq4xs_reference,
        sizeof(cuda_iq4xs_reference) / sizeof(cuda_iq4xs_reference[0]), BN_GPU_VALUE_LOGITS);
    /* Cached attention projections use a non-logits destination at offset zero.
     * Their arithmetic must agree with quant-only buffers and the reference. */
    run_signed_quant_graph_reference_cases_offset(gpu, cuda_iq4xs_reference,
        sizeof(cuda_iq4xs_reference) / sizeof(cuda_iq4xs_reference[0]), BN_GPU_VALUE_XB2, 0);
    printf("CUDA IQ4_XS reference PASSED\n");
}

#include "cuda_mxfp4_routed_reference.h"
#include "cuda_q5q6_routed_reference.h"
#include "cuda_q451_routed_reference.h"
#include "cuda_router512_reference.h"
#ifdef BN_CUDA_MXFP4_SM120
static void run_mxfp4_combined_case(BnGPUBackend *gpu,
    const BnCudaRoutedReference *r, void *g, void *u, void *d,
    float *input, float *output, int *ids, float *route) {
    int nt=r->nt, dim=r->dim, hidden=r->hidden, experts=r->experts, k=8;
    size_t n=(size_t)nt*dim;
    float *router_data=calloc((size_t)experts*dim,sizeof(float));
    float *normalized=malloc(n*sizeof(float)), *expected=malloc(n*sizeof(float));
    float *norm_data=malloc((size_t)dim*sizeof(float));
    assert(router_data && normalized && expected && norm_data);
    for(int i=0;i<dim;i++)norm_data[i]=1.0f+(float)(i%7)/32.0f;
    /* Nonzero logits below the fused top-k boundary; uniform logits above it
     * keep standalone and graph softmax layouts numerically equivalent. */
    if(nt<=8)for(int e=0;e<experts;e++)for(int i=0;i<dim;i++)
        router_data[(size_t)e*dim+i]=(float)((e*13+i*17)%257-128)/4096.0f;
    void *router=gpu->buffer_create(gpu->ctx,router_data,
        (size_t)experts*dim*sizeof(float),BN_GGUF_TENSOR_F32,experts,dim);
    void *norm=gpu->buffer_create(gpu->ctx,norm_data,
        (size_t)dim*sizeof(float),BN_GGUF_TENSOR_F32,1,dim);
    assert(router && norm);
    enum { sh=256 };
    size_t shared_blocks=(size_t)dim*sh/32;
    BnBlockQ8_0 *shared_data=malloc(shared_blocks*sizeof(*shared_data));
    float *shared_result=malloc(n*sizeof(float));
    float *shared_weight=calloc((size_t)dim,sizeof(float));
    assert(shared_data && shared_result && shared_weight);
    for(size_t block=0;block<shared_blocks;block++) {
        shared_data[block].d=bn_fp32_to_fp16(1.0f/512.0f);
        for(int j=0;j<32;j++)shared_data[block].qs[j]=(int8_t)((block*7+j*13)%17-8);
    }
    void *sg=gpu->buffer_create(gpu->ctx,shared_data,shared_blocks*sizeof(*shared_data),
        BN_GGUF_TENSOR_Q8_0,sh,dim);
    void *sd=gpu->buffer_create(gpu->ctx,shared_data,shared_blocks*sizeof(*shared_data),
        BN_GGUF_TENSOR_Q8_0,dim,sh);
    void *sw=gpu->buffer_create(gpu->ctx,shared_weight,(size_t)dim*sizeof(float),
        BN_GGUF_TENSOR_F32,1,dim);
    assert(sg && sd && sw);
    for(int normalized_residual=0;normalized_residual<4;normalized_residual++) {
        mxfp4_fixture_input(input+1,n,r->seed);
        if(normalized_residual)
            assert(gpu->rmsnorm_batch(gpu->ctx,normalized,norm,input+1,nt,dim,1e-6f)==0);
        else memcpy(normalized,input+1,n*sizeof(float));
        assert(gpu->moe_route_batch(gpu->ctx,ids,route,router,normalized,
            nt,dim,experts,k,1,1.0f)==0);
        assert(gpu->moe_routed_ffn_batch(gpu->ctx,expected,g,u,d,ids,route,NULL,normalized,
            nt,dim,hidden,experts,k,BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_MXFP4,
            BN_GGUF_TENSOR_Q5_K,BN_MODEL_ACTIVATION_SILU)==0);
        if(normalized_residual==1) {
            BnBackendRuntimePolicy policy={0};
            BnGPUBackend *resident=bn_gpu_cuda_create_with_policy(&policy);
            assert(resident && (resident->caps & BN_GPU_CAP_MOE_ROUTED_E8M0));
            assert(resident->rmsnorm_batch(resident->ctx,normalized,norm,input+1,nt,dim,1e-6f)==0);
            assert(resident->moe_route_routed_ffn_batch(resident->ctx,output+1,router,g,u,d,NULL,
                nt,dim,hidden,experts,k,BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_MXFP4,
                BN_GGUF_TENSOR_Q5_K,BN_MODEL_ACTIVATION_SILU,1,1.0f)==0);
            assert(memcmp(output+1,expected,n*sizeof(float))==0);
            assert(resident->rmsnorm_batch(resident->ctx,normalized,norm,input+1,nt,dim,1e-6f)==0);
            assert(resident->moe_route_routed_ffn_batch(resident->ctx,NULL,router,g,u,d,NULL,
                nt,dim,hidden,experts,k,BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_MXFP4,
                BN_GGUF_TENSOR_Q5_K,BN_MODEL_ACTIVATION_SILU,1,1.0f)==0);
            assert(gpu->moe_route_routed_ffn_batch(gpu->ctx,shared_result,router,g,u,d,expected,
                nt,dim,hidden,experts,k,BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_MXFP4,
                BN_GGUF_TENSOR_Q5_K,BN_MODEL_ACTIVATION_SILU,1,1.0f)==0);
            assert(resident->moe_route_routed_ffn_batch(resident->ctx,output+1,router,g,u,d,NULL,
                nt,dim,hidden,experts,k,BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_MXFP4,
                BN_GGUF_TENSOR_Q5_K,BN_MODEL_ACTIVATION_SILU,1,1.0f)==0);
            assert(memcmp(output+1,shared_result,n*sizeof(float))==0);
            bn_gpu_cuda_destroy(resident);
        }
        if(normalized_residual>=2) {
            assert(gpu->dense_ffn_batch(gpu->ctx,shared_result,sg,sg,sd,normalized,
                nt,dim,sh,BN_GGUF_TENSOR_Q8_0,BN_GGUF_TENSOR_Q8_0,
                BN_GGUF_TENSOR_Q8_0,BN_MODEL_ACTIVATION_SILU)==0);
            for(size_t i=0;i<n;i++)expected[i]+=shared_result[i]*(normalized_residual==3?0.5f:1.0f);
        }
        if(normalized_residual)for(size_t i=0;i<n;i++)expected[i]+=input[i+1];
        for(int alias=0;alias<2;alias++) {
            mxfp4_fixture_input(input+1,n,r->seed);
            input[0]=input[n+1]=output[0]=output[n+1]=12345.0f;
            float *out=alias?input+1:output+1;
            int rc=normalized_residual ?
                gpu->moe_route_routed_ffn_batch_norm_resid(gpu->ctx,out,router,g,u,d,
                    normalized_residual>=2?sg:NULL,normalized_residual>=2?sg:NULL,
                    normalized_residual>=2?sd:NULL,normalized_residual==3?sw:NULL,
                    norm,input+1,nt,dim,hidden,experts,k,
                    BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_Q5_K,
                    BN_MODEL_ACTIVATION_SILU,normalized_residual>=2?sh:0,
                    BN_GGUF_TENSOR_Q8_0,BN_GGUF_TENSOR_Q8_0,BN_GGUF_TENSOR_Q8_0,
                    1e-6f,1,1.0f) :
                gpu->moe_route_routed_ffn_batch(gpu->ctx,out,router,g,u,d,input+1,
                    nt,dim,hidden,experts,k,BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_MXFP4,
                    BN_GGUF_TENSOR_Q5_K,BN_MODEL_ACTIVATION_SILU,1,1.0f);
            if(rc || memcmp(out,expected,n*sizeof(float)))
                fprintf(stderr,"MXFP4 combined mismatch nt=%d dim=%d experts=%d norm=%d alias=%d rc=%d\n",
                    nt,dim,experts,normalized_residual,alias,rc);
            assert(rc==0 && memcmp(out,expected,n*sizeof(float))==0);
            assert(input[0]==12345.0f && input[n+1]==12345.0f);
            assert(output[0]==12345.0f && output[n+1]==12345.0f);
        }
    }
    gpu->buffer_destroy(gpu->ctx,sg);gpu->buffer_destroy(gpu->ctx,sd);
    gpu->buffer_destroy(gpu->ctx,sw);free(shared_data);free(shared_result);free(shared_weight);
    gpu->buffer_destroy(gpu->ctx,router);gpu->buffer_destroy(gpu->ctx,norm);
    free(router_data);free(normalized);free(expected);free(norm_data);
}

static void mxfp4_graph_backend_init(BnGPUBackend **backends) {
    for (int mode=0;mode<3;mode++) {
        BnBackendRuntimePolicy policy={0};
        assert(bn_backend_runtime_policy_set(&policy,mode ?
            "BN_CUDA_ENABLE_GRAPH_EXEC" : "BN_CUDA_DISABLE_GRAPH_EXEC","1",1)==0);
        if (!mode) assert(bn_backend_runtime_policy_set(&policy,
            "BN_CUDA_DISABLE_STREAM_EXEC","1",1)==0);
        BnGPUBackend *gpu=bn_gpu_cuda_create_with_policy(&policy);assert(gpu);
        bn_backend_runtime_policy_free(&policy);
        float rope=1.0f;
        BnGPUActivationPlan plan={0};
        plan.dim=plan.vocab_size=plan.xb2_elements=3072;plan.hb_elements=8192;
        plan.n_layers=plan.n_heads=plan.attention_layer_count=1;
        plan.seq_len=plan.kv_dim=plan.head_size=2;
        plan.rope_frequencies=&rope;plan.rope_frequency_count=1;
        plan.moe_total_experts=256;plan.moe_active_experts=8;
        plan.moe_expert_hidden_dim=1024;
        assert(gpu->init_activations(gpu->ctx,&plan)==0);
        backends[mode]=gpu;
    }
}

static void run_routed_graph_case(BnGPUBackend **backends, BnGPUBackend *owner,
    const BnCudaRoutedReference *r, const uint8_t *gate, const uint8_t *up,
    const uint8_t *down, void *g, void *u, void *d, float *input, float *output, int gate_type, int down_type,
    const BnCudaRoutedReference *references, size_t reference_count) {
    enum { k=8, small_dim=256, small_hidden=512, small_experts=16 };
    void *sg=owner->buffer_create(owner->ctx,gate,bn_quant_format_data_size(gate_type,small_experts*small_hidden,small_dim),
        gate_type,small_experts*small_hidden,small_dim);
    void *su=owner->buffer_create(owner->ctx,up,bn_quant_format_data_size(gate_type,small_experts*small_hidden,small_dim),
        gate_type,small_experts*small_hidden,small_dim);
    void *sd=owner->buffer_create(owner->ctx,down,bn_quant_format_data_size(down_type,small_experts*small_dim,small_hidden),
        down_type,small_experts*small_dim,small_hidden);
    assert(sg && su && sd);
    uint64_t small_hash=0;
    for(size_t i=0;i<reference_count;i++) {
        const BnCudaRoutedReference *q=&references[i];
        if(q->dim==small_dim && q->hidden==small_hidden && q->experts==small_experts &&
           q->nt==1 && q->seed==r->seed) small_hash=q->hash;
    }
    assert(small_hash);
    uint64_t reference_hash = 0;
    uint64_t small_reference_hash = 0;
    for(int mode=0;mode<3;mode++) {
        BnGPUBackend *gpu=backends[mode];
        BnGPUOp ops[13]={0};
        for(int j=0;j<11;j++) {
            ops[j].op_code=BN_GPU_CODE_MOE_ROUTED_FFN;
            ops[j].W_buf=g;ops[j].W_buf2=u;ops[j].W_buf3=d;
            ops[j].type=gate_type;ops[j].cols=r->dim;
            ops[j].buf_in=BN_GPU_VALUE_X;ops[j].buf_out=BN_GPU_VALUE_MOE_OUT;
            ops[j].buf_aux=BN_GPU_VALUE_XB2;ops[j].p[0]=r->hidden;
            ops[j].p[1]=r->experts;ops[j].p[2]=k;ops[j].p[3]=down_type;
            ops[j].p[4]=BN_GPU_VALUE_MOE_HB;
        }
        ops[0].W_buf=sg;ops[0].W_buf2=su;ops[0].W_buf3=sd;
        ops[0].cols=small_dim;ops[0].p[0]=small_hidden;ops[0].p[1]=small_experts;
        ops[0].buf_aux=BN_GPU_VALUE_HB;ops[0].buf_out=BN_GPU_VALUE_XB;
        if(mode==2)ops[10].buf_out=BN_GPU_VALUE_LOGITS;
        /* Supply valid attention runtime parameters to require real graph replay. */
        ops[11].op_code=BN_GPU_CODE_ROPE;ops[11].buf_in=BN_GPU_VALUE_Q;
        ops[11].p[0]=1;ops[11].p[1]=2;ops[11].p[3]=2;
        ops[12].op_code=BN_GPU_CODE_FLASH_ATTN;
        ops[12].buf_in=BN_GPU_VALUE_Q;ops[12].buf_out=BN_GPU_VALUE_HB2;
        ops[12].p[0]=1;ops[12].p[1]=2;ops[12].p[2]=1;
        ops[12].p[3]=1;ops[12].p[4]=2;ops[12].p[5]=2;
        ops[12].p[7]=f32_bits(1.0f);
        float zero[4]={0};
        assert(gpu->write_activation(gpu->ctx,BN_GPU_VALUE_Q,zero,2*sizeof(float),0)==0);
        assert(gpu->write_activation(gpu->ctx,BN_GPU_VALUE_KEY_CACHE,zero,sizeof(zero),0)==0);
        assert(gpu->write_activation(gpu->ctx,BN_GPU_VALUE_VALUE_CACHE,zero,sizeof(zero),0)==0);
        for(int repeat=0;repeat<4;repeat++) {
            float route[2*k], small_route[2*k], small_output[small_dim];
            for(int j=0;j<11;j++)
                ops[j].flags = repeat >= 2
                    ? BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION : 0;
            for(int j=0;j<k;j++) {
                route[j]=small_route[j]=repeat==3 ? 0.0f : (float)(j+1)/36.0f;
                route[k+j]=repeat==3 ? (j%2 ? NAN : (float)(r->experts+10)) : (float)(j*3%r->experts);
                small_route[k+j]=repeat==3 ? -1.0f : (float)(j*3%small_experts);
            }
            mxfp4_fixture_input(input+1,r->dim,r->seed);
            if(repeat==1)memset(input+1,0,(size_t)r->dim*sizeof(float));
            assert(gpu->write_activation(gpu->ctx,BN_GPU_VALUE_X,input+1,
                (size_t)r->dim*sizeof(float),0)==0);
            assert(gpu->write_activation(gpu->ctx,BN_GPU_VALUE_XB2,route,sizeof(route),0)==0);
            assert(gpu->write_activation(gpu->ctx,BN_GPU_VALUE_HB,small_route,sizeof(small_route),0)==0);
            output[0]=output[r->dim+1]=12345.0f;
            assert(gpu->execute(gpu->ctx,ops,13,ops[10].buf_out,output+1,r->dim)==0);
            assert(gpu->read_activation(gpu->ctx,BN_GPU_VALUE_XB,small_output,sizeof(small_output),0)==0);
            if(repeat==0) {
                assert(cuda_test_float_bits_hash(output+1,r->dim)==r->hash);
                assert(cuda_test_float_bits_hash(small_output,small_dim)==small_hash);
            } else if(repeat==2) {
                uint64_t actual_hash = cuda_test_float_bits_hash(output+1,r->dim);
                uint64_t actual_small_hash = cuda_test_float_bits_hash(small_output,small_dim);
                if(mode==0) {
                    reference_hash = actual_hash;
                    small_reference_hash = actual_small_hash;
                    assert(reference_hash != r->hash);
                    assert(small_reference_hash != small_hash);
                } else {
                    assert(actual_hash == reference_hash);
                    assert(actual_small_hash == small_reference_hash);
                }
            } else {
                for(int j=0;j<r->dim;j++)assert(output[j+1]==0.0f);
                for(int j=0;j<small_dim;j++)assert(small_output[j]==0.0f);
            }
            assert(output[0]==12345.0f && output[r->dim+1]==12345.0f);
        }
        for(int j=0;j<r->dim;j++)output[j+1]=12345.0f;
        uint32_t hidden=ops[10].p[0];ops[10].p[0]=0;
        assert(gpu->execute(gpu->ctx,ops,13,ops[10].buf_out,output+1,r->dim)!=0);
        for(int j=0;j<r->dim;j++)assert(output[j+1]==12345.0f);
        ops[10].p[0]=hidden;
        assert(gpu->execute(gpu->ctx,ops,8193,ops[10].buf_out,output+1,r->dim)!=0);
        for(int j=0;j<r->dim;j++)assert(output[j+1]==12345.0f);
    }
    owner->buffer_destroy(owner->ctx,sg);owner->buffer_destroy(owner->ctx,su);
    owner->buffer_destroy(owner->ctx,sd);
}
#endif

static void run_mxfp4_routed_reference_cases(BnGPUBackend *gpu, int graph_only) {
#ifdef BN_CUDA_MXFP4_SM120
    BnGPUBackend *graph_backends[3]={0};
    if(graph_only==1)mxfp4_graph_backend_init(graph_backends);
    for (size_t c = 0; c < sizeof(cuda_mxfp4_routed_reference)/sizeof(cuda_mxfp4_routed_reference[0]); c++) {
        const BnCudaRoutedReference *r = &cuda_mxfp4_routed_reference[c];
        if(graph_only==1 && r->nt!=1)continue;
        int dim=r->dim, hidden=r->hidden, experts=r->experts, nt=r->nt, k=8;
        size_t elements=(size_t)experts*hidden*dim;
        size_t gate_bytes=elements/32*17, down_bytes=elements/256*176;
        uint8_t *gate=malloc(gate_bytes), *up=malloc(gate_bytes), *down=malloc(down_bytes);
        size_t n=(size_t)nt*dim, items=(size_t)nt*k;
        float *input=malloc((n+2)*sizeof(float)), *output=malloc((n+2)*sizeof(float));
        float *route=malloc(items*sizeof(float));int *ids=malloc(items*sizeof(int));
        assert(gate && up && down && input && output && route && ids);
        mxfp4_fixture_weights(gate,elements,r->seed);
        mxfp4_fixture_weights(up,elements,r->seed+19);
        for(size_t b=0;b<elements/32;b++){gate[b*17]-=8;up[b*17]-=8;}
        int seed=r->seed+37;
        for(size_t b=0;b<elements/256;b++) {
            uint8_t *w=down+b*176;
            uint16_t d=(uint16_t)(0x1800+((b+seed)%17)*37);
            uint16_t m=(uint16_t)(0x1400+((b*3+seed)%19)*29);
            memcpy(w,&d,2);memcpy(w+2,&m,2);
            for(int j=0;j<12;j++)w[4+j]=(uint8_t)(b*7+j*13+seed);
            for(int j=0;j<32;j++)w[16+j]=(uint8_t)(b*11+j*17+seed);
            for(int j=0;j<128;j++)w[48+j]=(uint8_t)(b*19+j*23+seed);
        }
        for(int t=0;t<nt;t++)for(int j=0;j<k;j++) {
            ids[t*k+j]=(t*7+j*3)%experts;route[t*k+j]=(float)(j+1)/36.0f;
        }
        void *g=gpu->buffer_create(gpu->ctx,gate,gate_bytes,BN_GGUF_TENSOR_MXFP4,experts*hidden,dim);
        void *u=gpu->buffer_create(gpu->ctx,up,gate_bytes,BN_GGUF_TENSOR_MXFP4,experts*hidden,dim);
        void *d=gpu->buffer_create(gpu->ctx,down,down_bytes,BN_GGUF_TENSOR_Q5_K,experts*dim,hidden);
        assert(g && u && d);
        if(graph_only==2) {
            run_mxfp4_combined_case(gpu,r,g,u,d,input,output,ids,route);
        } else if(graph_only==1) {
            run_routed_graph_case(graph_backends,gpu,r,gate,up,down,g,u,d,input,output,
                BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_Q5_K,cuda_mxfp4_routed_reference,
                sizeof(cuda_mxfp4_routed_reference)/sizeof(cuda_mxfp4_routed_reference[0]));
        } else for(int repeat=0;repeat<2;repeat++)for(int alias=0;alias<2;alias++) {
            input[0]=input[n+1]=output[0]=output[n+1]=12345.0f;
            mxfp4_fixture_input(input+1,n,r->seed);
            float *out=alias?input+1:output+1;
            assert(gpu->moe_routed_ffn_batch(gpu->ctx,out,g,u,d,ids,route,NULL,input+1,
                nt,dim,hidden,experts,k,BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_MXFP4,
                BN_GGUF_TENSOR_Q5_K,BN_MODEL_ACTIVATION_SILU)==0);
            if(cuda_test_float_bits_hash(out,n)!=r->hash)
                fprintf(stderr,"routed MXFP4 case=%zu alias=%d repeat=%d mismatch\n",c,alias,repeat);
            assert(cuda_test_float_bits_hash(out,n)==r->hash);
            int saved=ids[0];
            const int invalid[]={-1,experts,ids[1]};
            for(size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++) {
                ids[0]=invalid[i];
                assert(gpu->moe_routed_ffn_batch(gpu->ctx,out,g,u,d,ids,route,NULL,input+1,
                    nt,dim,hidden,experts,k,BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_MXFP4,
                    BN_GGUF_TENSOR_Q5_K,BN_MODEL_ACTIVATION_SILU)!=0);
                assert(cuda_test_float_bits_hash(out,n)==r->hash);
            }
            ids[0]=saved;
            assert(gpu->moe_routed_ffn_batch(gpu->ctx,out,g,u,d,ids,route,NULL,input+1,
                2049,dim,hidden,experts,k,BN_GGUF_TENSOR_MXFP4,BN_GGUF_TENSOR_MXFP4,
                BN_GGUF_TENSOR_Q5_K,BN_MODEL_ACTIVATION_SILU)!=0);
            assert(cuda_test_float_bits_hash(out,n)==r->hash);
            assert(input[0]==12345.0f && input[n+1]==12345.0f);
            assert(output[0]==12345.0f && output[n+1]==12345.0f);
        }
        gpu->buffer_destroy(gpu->ctx,g);gpu->buffer_destroy(gpu->ctx,u);gpu->buffer_destroy(gpu->ctx,d);
        free(gate);free(up);free(down);free(input);free(output);free(route);free(ids);
    }
    if(graph_only==1)for(int mode=0;mode<3;mode++)bn_gpu_cuda_destroy(graph_backends[mode]);
    printf("CUDA routed MXFP4 %sreference PASSED\n",graph_only==2?"combined ":graph_only?"graph ":"");
#else
    (void)gpu; (void)graph_only;
    printf("CUDA routed MXFP4 reference skipped: requires SM120 native FP4 build\n");
#endif
}

static void run_q5q6_routed_reference_cases(BnGPUBackend *gpu, int graph_only) {
#ifdef BN_CUDA_MXFP4_SM120
    BnGPUBackend *graph_backends[3]={0};
    if(graph_only==1)mxfp4_graph_backend_init(graph_backends);
    for (size_t c = 0; c < sizeof(cuda_q5q6_routed_reference)/sizeof(cuda_q5q6_routed_reference[0]); c++) {
        const BnCudaRoutedReference *r = &cuda_q5q6_routed_reference[c];
        if(graph_only==1 && r->nt!=1)continue;
        int dim=r->dim, hidden=r->hidden, experts=r->experts, nt=r->nt, k=8;
        size_t elements=(size_t)experts*hidden*dim;
        size_t gate_bytes=elements/256*176, down_bytes=elements/256*210;
        uint8_t *gate=malloc(gate_bytes), *up=malloc(gate_bytes), *down=malloc(down_bytes);
        size_t n=(size_t)nt*dim, items=(size_t)nt*k;
        float *input=malloc((n+2)*sizeof(float)), *output=malloc((n+2)*sizeof(float));
        float *route=malloc(items*sizeof(float));int *ids=malloc(items*sizeof(int));
        assert(gate && up && down && input && output && route && ids);
        for(int matrix=0;matrix<2;matrix++) {
            uint8_t *data=matrix?up:gate;int seed=r->seed+matrix*19;
            for(size_t b=0;b<elements/256;b++) {
                uint8_t*w=data+b*176;
                uint16_t scale=(uint16_t)(0x1800+((b+seed)%17)*37);
                uint16_t minimum=(uint16_t)(0x1400+((b*3+seed)%19)*29);
                memcpy(w,&scale,2);memcpy(w+2,&minimum,2);
                for(int j=0;j<12;j++)w[4+j]=(uint8_t)(b*7+j*13+seed);
                for(int j=0;j<32;j++)w[16+j]=(uint8_t)(b*11+j*17+seed);
                for(int j=0;j<128;j++)w[48+j]=(uint8_t)(b*19+j*23+seed);
            }
        }
        int seed=r->seed+37;
        for(size_t b=0;b<elements/256;b++) {
            uint8_t*w=down+b*210;
            for(int j=0;j<192;j++)w[j]=(uint8_t)(b*19+j*23+seed);
            for(int j=0;j<16;j++)w[192+j]=(uint8_t)((int)((b*3+j*5+seed)%31)-15);
            uint16_t scale=(uint16_t)(0x1400+((b+seed)%17)*37);memcpy(w+208,&scale,2);
        }
        for(int t=0;t<nt;t++)for(int j=0;j<k;j++) {
            ids[t*k+j]=(t*7+j*3)%experts;route[t*k+j]=(float)(j+1)/36.0f;
        }
        void *g=gpu->buffer_create(gpu->ctx,gate,gate_bytes,BN_GGUF_TENSOR_Q5_K,experts*hidden,dim);
        void *u=gpu->buffer_create(gpu->ctx,up,gate_bytes,BN_GGUF_TENSOR_Q5_K,experts*hidden,dim);
        void *d=gpu->buffer_create(gpu->ctx,down,down_bytes,BN_GGUF_TENSOR_Q6_K,experts*dim,hidden);
        assert(g && u && d);
        if(graph_only==1) {
            run_routed_graph_case(graph_backends,gpu,r,gate,up,down,g,u,d,input,output,
                BN_GGUF_TENSOR_Q5_K,BN_GGUF_TENSOR_Q6_K,cuda_q5q6_routed_reference,
                sizeof(cuda_q5q6_routed_reference)/sizeof(cuda_q5q6_routed_reference[0]));
        } else for(int repeat=0;repeat<2;repeat++)for(int alias=0;alias<2;alias++) {
            input[0]=input[n+1]=output[0]=output[n+1]=12345.0f;
            mxfp4_fixture_input(input+1,n,r->seed);
            float *out=alias?input+1:output+1;
            assert(gpu->moe_routed_ffn_batch(gpu->ctx,out,g,u,d,ids,route,NULL,input+1,
                nt,dim,hidden,experts,k,BN_GGUF_TENSOR_Q5_K,BN_GGUF_TENSOR_Q5_K,
                BN_GGUF_TENSOR_Q6_K,BN_MODEL_ACTIVATION_SILU)==0);
            if(cuda_test_float_bits_hash(out,n)!=r->hash)
                fprintf(stderr,"routed Q5_K/Q6_K case=%zu alias=%d repeat=%d mismatch\n",c,alias,repeat);
            assert(cuda_test_float_bits_hash(out,n)==r->hash);
            int saved=ids[0];
            const int invalid[]={-1,experts,ids[1]};
            for(size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++) {
                ids[0]=invalid[i];
                assert(gpu->moe_routed_ffn_batch(gpu->ctx,out,g,u,d,ids,route,NULL,input+1,
                    nt,dim,hidden,experts,k,BN_GGUF_TENSOR_Q5_K,BN_GGUF_TENSOR_Q5_K,
                    BN_GGUF_TENSOR_Q6_K,BN_MODEL_ACTIVATION_SILU)!=0);
                assert(cuda_test_float_bits_hash(out,n)==r->hash);
            }
            ids[0]=saved;
            assert(gpu->moe_routed_ffn_batch(gpu->ctx,out,g,u,d,ids,route,NULL,input+1,
                2049,dim,hidden,experts,k,BN_GGUF_TENSOR_Q5_K,BN_GGUF_TENSOR_Q5_K,
                BN_GGUF_TENSOR_Q6_K,BN_MODEL_ACTIVATION_SILU)!=0);
            assert(cuda_test_float_bits_hash(out,n)==r->hash);
            assert(input[0]==12345.0f && input[n+1]==12345.0f);
            assert(output[0]==12345.0f && output[n+1]==12345.0f);
        }
        gpu->buffer_destroy(gpu->ctx,g);gpu->buffer_destroy(gpu->ctx,u);gpu->buffer_destroy(gpu->ctx,d);
        free(gate);free(up);free(down);free(input);free(output);free(route);free(ids);
    }
    if(graph_only==1)for(int mode=0;mode<3;mode++)bn_gpu_cuda_destroy(graph_backends[mode]);
    printf("CUDA routed Q5_K/Q6_K %sreference PASSED\n",graph_only?"graph ":"");
#else
    (void)gpu; (void)graph_only;
    printf("CUDA routed Q5_K/Q6_K reference skipped: requires SM120 native FP4 build\n");
#endif
}

static void run_q451_routed_reference_cases(BnGPUBackend *gpu, int graph_only) {
#ifdef BN_CUDA_MXFP4_SM120
    BnGPUBackend *graph_backends[3]={0};
    if(graph_only==1)mxfp4_graph_backend_init(graph_backends);
    for (size_t c = 0; c < sizeof(cuda_q451_routed_reference)/sizeof(cuda_q451_routed_reference[0]); c++) {
        const BnCudaRoutedReference *r = &cuda_q451_routed_reference[c];
        if(graph_only==1 && r->nt!=1)continue;
        int dim=r->dim, hidden=r->hidden, experts=r->experts, nt=r->nt, k=8;
        size_t elements=(size_t)experts*hidden*dim;
        size_t gate_bytes=elements/256*144, down_bytes=elements/32*24;
        uint8_t *gate=malloc(gate_bytes), *up=malloc(gate_bytes), *down=malloc(down_bytes);
        size_t n=(size_t)nt*dim, items=(size_t)nt*k;
        float *input=malloc((n+2)*sizeof(float)), *output=malloc((n+2)*sizeof(float));
        float *route=malloc(items*sizeof(float));int *ids=malloc(items*sizeof(int));
        assert(gate && up && down && input && output && route && ids);
        for(int matrix=0;matrix<2;matrix++) {
            uint8_t *data=matrix?up:gate;int seed=r->seed+matrix*19;
            for(size_t b=0;b<elements/256;b++) {
                uint8_t*w=data+b*144;
                uint16_t scale=(uint16_t)(0x1800+((b+seed)%17)*37);
                uint16_t minimum=(uint16_t)(0x1400+((b*3+seed)%19)*29);
                memcpy(w,&scale,2);memcpy(w+2,&minimum,2);
                for(int j=0;j<12;j++)w[4+j]=(uint8_t)(b*7+j*13+seed);
                for(int j=0;j<128;j++)w[16+j]=(uint8_t)(b*19+j*23+seed);
            }
        }
        int seed=r->seed+37;
        for(size_t b=0;b<elements/32;b++) {
            uint8_t *w=down+b*24;
            uint16_t scale=(uint16_t)(0x1800+((b+seed)%17)*37);
            uint16_t minimum=(uint16_t)(0x9400+((b*3+seed)%19)*29);
            memcpy(w,&scale,2);memcpy(w+2,&minimum,2);
            for(int j=4;j<24;j++)w[j]=(uint8_t)(b*19+j*23+seed);
        }
        for(int t=0;t<nt;t++)for(int j=0;j<k;j++) {
            ids[t*k+j]=(t*7+j*3)%experts;route[t*k+j]=(float)(j+1)/36.0f;
        }
        void *g=gpu->buffer_create(gpu->ctx,gate,gate_bytes,BN_GGUF_TENSOR_Q4_K,experts*hidden,dim);
        void *u=gpu->buffer_create(gpu->ctx,up,gate_bytes,BN_GGUF_TENSOR_Q4_K,experts*hidden,dim);
        void *d=gpu->buffer_create(gpu->ctx,down,down_bytes,BN_GGUF_TENSOR_Q5_1,experts*dim,hidden);
        assert(g && u && d);
        if(graph_only==1) {
            run_routed_graph_case(graph_backends,gpu,r,gate,up,down,g,u,d,input,output,
                BN_GGUF_TENSOR_Q4_K,BN_GGUF_TENSOR_Q5_1,cuda_q451_routed_reference,
                sizeof(cuda_q451_routed_reference)/sizeof(cuda_q451_routed_reference[0]));
        } else for(int repeat=0;repeat<2;repeat++)for(int alias=0;alias<2;alias++) {
            input[0]=input[n+1]=output[0]=output[n+1]=12345.0f;
            mxfp4_fixture_input(input+1,n,r->seed);
            float *out=alias?input+1:output+1;
            assert(gpu->moe_routed_ffn_batch(gpu->ctx,out,g,u,d,ids,route,NULL,input+1,
                nt,dim,hidden,experts,k,BN_GGUF_TENSOR_Q4_K,BN_GGUF_TENSOR_Q4_K,
                BN_GGUF_TENSOR_Q5_1,BN_MODEL_ACTIVATION_SILU)==0);
            if(cuda_test_float_bits_hash(out,n)!=r->hash)
                fprintf(stderr,"routed Q4_K/Q5_1 case=%zu alias=%d repeat=%d mismatch\n",c,alias,repeat);
            assert(cuda_test_float_bits_hash(out,n)==r->hash);
            int saved=ids[0];
            const int invalid[]={-1,experts,ids[1]};
            for(size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++) {
                ids[0]=invalid[i];
                assert(gpu->moe_routed_ffn_batch(gpu->ctx,out,g,u,d,ids,route,NULL,input+1,
                    nt,dim,hidden,experts,k,BN_GGUF_TENSOR_Q4_K,BN_GGUF_TENSOR_Q4_K,
                    BN_GGUF_TENSOR_Q5_1,BN_MODEL_ACTIVATION_SILU)!=0);
                assert(cuda_test_float_bits_hash(out,n)==r->hash);
            }
            ids[0]=saved;
            assert(input[0]==12345.0f && input[n+1]==12345.0f);
            assert(output[0]==12345.0f && output[n+1]==12345.0f);
        }
        gpu->buffer_destroy(gpu->ctx,g);gpu->buffer_destroy(gpu->ctx,u);gpu->buffer_destroy(gpu->ctx,d);
        free(gate);free(up);free(down);free(input);free(output);free(route);free(ids);
    }
    if(graph_only==1)for(int mode=0;mode<3;mode++)bn_gpu_cuda_destroy(graph_backends[mode]);
    printf("CUDA routed Q4_K/Q5_1 %sreference PASSED\n",graph_only?"graph ":"");
#else
    (void)gpu; (void)graph_only;
    printf("CUDA routed Q4_K/Q5_1 reference skipped: requires SM120 native FP4 build\n");
#endif
}

static void run_mxfp4_reference_case(BnGPUBackend *gpu) {
#ifdef BN_CUDA_MXFP4_SM120
    size_t count = sizeof(cuda_mxfp4_reference) / sizeof(cuda_mxfp4_reference[0]);
    run_signed_quant_reference_cases(gpu, cuda_mxfp4_reference, count);
    run_signed_quant_graph_reference_cases(gpu, cuda_mxfp4_reference,
                                           count, BN_GPU_VALUE_LOGITS);
    run_signed_quant_graph_reference_cases_offset(gpu, cuda_mxfp4_reference,
                                                  count, BN_GPU_VALUE_XB2, 0);
    printf("CUDA MXFP4 reference PASSED\n");
#else
    (void)gpu;
    printf("CUDA MXFP4 reference skipped: requires SM120 native FP4 build\n");
#endif
}

static void run_iq4nl_reference_case(BnGPUBackend *gpu) {
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_IQ4_NL,
        BN_QUANT_CAP_GPU_MMVQ_BLOCK32_CODEBOOK_FP16));
    run_signed_quant_reference_cases(gpu, cuda_iq4nl_reference,
        sizeof(cuda_iq4nl_reference) / sizeof(cuda_iq4nl_reference[0]));
    run_signed_quant_graph_reference_cases(gpu, cuda_iq4nl_reference,
        sizeof(cuda_iq4nl_reference) / sizeof(cuda_iq4nl_reference[0]), BN_GPU_VALUE_LOGITS);
    printf("CUDA IQ4_NL reference PASSED\n");
}

static void run_q3k_reference_case(BnGPUBackend *gpu) {
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_Q3_K,
        BN_QUANT_CAP_GPU_MMVQ_SUBBLOCK16_INPUT_SCALE));
    size_t n = sizeof(cuda_q3k_reference) / sizeof(cuda_q3k_reference[0]);
    run_signed_quant_reference_cases(gpu, cuda_q3k_reference, n);
    run_signed_quant_graph_reference_cases(gpu, cuda_q3k_reference, n, BN_GPU_VALUE_LOGITS);
    run_signed_quant_graph_reference_cases(gpu, cuda_q3k_reference, n, BN_GPU_VALUE_XB2);
    printf("CUDA Q3_K reference PASSED\n");
}

/* Split projections must retain the unsplit GGML result, including when the
 * prepared input permits the first output to reuse the input allocation. */
static void run_q8_split_reference_case(BnGPUBackend *gpu) {
    size_t n = sizeof(cuda_q8_reference) / sizeof(cuda_q8_reference[0]);
    for (size_t c = 0; c < n; c++) {
        const BnCudaSignedMmqReference *ref = &cuda_q8_reference[c];
        if (ref->tokens != 1 || ref->rows < 3) continue;
        int rows = ref->rows, cols = ref->cols;
        int capacity = (cols > rows ? cols : rows) + 11;
        size_t elements = (size_t)rows * cols;
        size_t bytes = signed_mmq_fixture_bytes(ref->type, elements);
        void *weights = malloc(bytes);
        float *input = malloc((size_t)cols * sizeof(float));
        float *output = malloc((size_t)capacity * sizeof(float));
        float *joined = malloc((size_t)rows * sizeof(float));
        float *chain = malloc((size_t)rows * sizeof(float));
        assert(weights && input && output && joined && chain);
        signed_mmq_fixture_weights(weights, ref->type, elements, ref->seed);
        signed_mmq_fixture_input(input, cols, ref->seed);
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = capacity;
        plan.vocab_size = capacity;
        plan.n_layers = plan.seq_len = plan.n_heads = plan.head_size = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        for (int quant_only = 0; quant_only < 2; quant_only++) {
            void *buffer = quant_only
                ? gpu->buffer_create_quant_only(gpu->ctx, weights, bytes, ref->type, rows, cols)
                : gpu->buffer_create(gpu->ctx, weights, bytes, ref->type, rows, cols);
            assert(buffer);
            for (int parts = 2; parts <= 3; parts++) {
                int split = rows / parts;
                for (int alias = 0; alias < 2; alias++) {
                    int slots[3] = {alias ? BN_GPU_VALUE_X : BN_GPU_VALUE_HB,
                                    BN_GPU_VALUE_HB2, BN_GPU_VALUE_XB2};
                    int offsets[3] = {0, 3, 5};
                    BnGPUOp ops[32] = {{0}};
                    ops[0].op_code = BN_GPU_CODE_MATVEC_SPLIT;
                    ops[0].type = ref->type; ops[0].W_buf = buffer;
                    ops[0].buf_in = BN_GPU_VALUE_X;
                    ops[0].buf_out = slots[0]; ops[0].buf_aux = slots[1];
                    ops[0].rows = slots[2];
                    ops[0].p[0] = rows; ops[0].p[1] = cols;
                    ops[0].p[2] = split; ops[0].p[3] = parts == 3 ? 2 * split : 0;
                    ops[0].p[6] = offsets[1]; ops[0].p[7] = offsets[2];
                    int previous = slots[0];
                    for (int i = 1; i < 32; i++) {
                        ops[i].op_code = BN_GPU_CODE_COPY;
                        ops[i].buf_in = previous;
                        ops[i].buf_out = i & 1 ? BN_GPU_VALUE_LOGITS : BN_GPU_VALUE_XB;
                        ops[i].p[2] = split;
                        previous = ops[i].buf_out;
                    }
                    for (int repeat = 0; repeat < 4; repeat++) {
                        for (int i = 0; i < capacity; i++) output[i] = -12345.0f;
                        for (int part = 0; part < parts; part++)
                            assert(gpu->write_activation(gpu->ctx, slots[part], output,
                                (size_t)capacity * sizeof(float), 0) == 0);
                        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, input,
                            (size_t)cols * sizeof(float), 0) == 0);
                        assert(gpu->execute(gpu->ctx, ops, 32, previous, chain, split) == 0);
                        int first = 0;
                        for (int part = 0; part < parts; part++) {
                            int count = part == parts - 1 ? rows - first : split;
                            int offset = offsets[part];
                            assert(gpu->read_activation(gpu->ctx, slots[part], output,
                                (size_t)capacity * sizeof(float), 0) == 0);
                            memcpy(joined + first, output + offset, (size_t)count * sizeof(float));
                            for (int i = 0; i < capacity; i++) {
                                if (i >= offset && i < offset + count) continue;
                                float expected = alias && part == 0 && i < cols
                                    ? input[i] : -12345.0f;
                                assert(output[i] == expected);
                            }
                            first += count;
                        }
                        uint64_t hash = cuda_test_float_bits_hash(joined, rows);
                        if (hash != ref->hash)
                            fprintf(stderr, "Q8 split rows=%d cols=%d seed=%d quant=%d parts=%d alias=%d repeat=%d\n",
                                rows, cols, ref->seed, quant_only, parts, alias, repeat);
                        assert(hash == ref->hash);
                        assert(memcmp(chain, joined, (size_t)split * sizeof(float)) == 0);
                    }
                }
            }
            gpu->buffer_destroy(gpu->ctx, buffer);
        }
        free(weights); free(input); free(output); free(joined); free(chain);
    }
    printf("CUDA Q8_0 split reference PASSED\n");
}

static void run_q8_reference_case(BnGPUBackend *gpu) {
    assert(bn_quant_format_supports_gpu_dense_graph_native_quant(BN_GGUF_TENSOR_Q8_0));
    size_t n = sizeof(cuda_q8_reference) / sizeof(cuda_q8_reference[0]);
    run_signed_quant_reference_cases(gpu, cuda_q8_reference, n);
    run_signed_quant_graph_reference_cases(gpu, cuda_q8_reference, n, BN_GPU_VALUE_LOGITS);
    run_signed_quant_graph_reference_cases(gpu, cuda_q8_reference, n, BN_GPU_VALUE_XB2);
    printf("CUDA Q8_0 reference PASSED\n");
}

static void run_kquant_mmq_reference_case(BnGPUBackend *gpu) {
    for (size_t c = 0; c < sizeof(cuda_kquant_reference) /
                                sizeof(cuda_kquant_reference[0]); c++) {
        const BnCudaKQuantReferenceCase *ref = &cuda_kquant_reference[c];
        int rows = ref->rows, cols = ref->cols, nt = ref->tokens;
        size_t blocks = (size_t)rows * cols / BN_QK_K;
        size_t bytes = blocks * kquant_fixture_block_size(ref->type);
        size_t count = (size_t)rows * nt, nx = (size_t)cols * nt;
        size_t capacity = count > nx ? count : nx;
        void *weights = malloc(bytes);
        float *input = malloc(nx * sizeof(float));
        float *output = malloc((capacity + 2) * sizeof(float));
        assert(weights && input && output);
        kquant_fixture_weights(weights, ref->type, blocks);
        kquant_fixture_input(input, nx);
        for (int quant_only = 0; quant_only < 2; quant_only++) {
            void *buffer = quant_only
                ? gpu->buffer_create_quant_only(gpu->ctx, weights, bytes, ref->type, rows, cols)
                : gpu->buffer_create(gpu->ctx, weights, bytes, ref->type, rows, cols);
            assert(buffer);
            for (int alias = 0; alias < 2; alias++) {
                for (size_t i = 0; i < capacity + 2; i++) output[i] = -12345.0f;
                if (alias) memcpy(output + 1, input, nx * sizeof(float));
                assert(gpu->matmul(gpu->ctx, output + 1, buffer,
                    alias ? output + 1 : input, rows, cols, nt, ref->type) == 0);
                uint64_t hash = cuda_test_float_bits_hash(output + 1, count);
                if (hash != ref->hash)
                    fprintf(stderr, "K-quant MMQ type=%d rows=%d cols=%d nt=%d quant_only=%d alias=%d: %016llx != %016llx\n",
                        ref->type, rows, cols, nt, quant_only, alias,
                        (unsigned long long)hash, (unsigned long long)ref->hash);
                assert(hash == ref->hash);
                assert(output[0] == -12345.0f && output[capacity + 1] == -12345.0f);
                size_t untouched = alias && nx > count ? nx : count;
                for (size_t i = untouched; i < capacity; i++) assert(output[i + 1] == -12345.0f);
            }
            gpu->buffer_destroy(gpu->ctx, buffer);
        }
        free(output); free(input); free(weights);
    }
    printf("CUDA K-quant MMQ reference PASSED\n");
}

#include "cuda_q6_reference.h"

static void run_q6_small_reference_case(BnGPUBackend *gpu) {
    for (size_t c = 0; c < sizeof(cuda_q6_reference) / sizeof(cuda_q6_reference[0]); c++) {
        const BnCudaQ6ReferenceCase *ref = &cuda_q6_reference[c];
        int rows = ref->rows, cols = ref->cols, nt = ref->tokens;
        size_t blocks = (size_t)rows * cols / BN_QK_K;
        size_t bytes = blocks * sizeof(BnBlockQ6K);
        size_t count = (size_t)rows * nt, nx = (size_t)cols * nt;
        size_t capacity = count > nx ? count : nx;
        void *weights = malloc(bytes);
        float *input = malloc(nx * sizeof(float));
        float *out = malloc((capacity + 2) * sizeof(float));
        assert(weights && input && out);
        kquant_fixture_weights(weights, BN_GGUF_TENSOR_Q6_K, blocks);
        for (size_t i = 0; i < nx; i++)
            input[i] = (float)((int)((i * 37 + ref->seed) % 257) - 128) / 256.0f;
        void *buffer = gpu->buffer_create(gpu->ctx, weights, bytes,
            BN_GGUF_TENSOR_Q6_K, rows, cols);
        assert(buffer);
        for (int alias = 0; alias < 2; alias++) {
            for (size_t i = 0; i < capacity + 2; i++) out[i] = -12345.0f;
            if (alias) memcpy(out + 1, input, nx * sizeof(float));
            assert(gpu->matmul(gpu->ctx, out + 1, buffer, alias ? out + 1 : input,
                rows, cols, nt, BN_GGUF_TENSOR_Q6_K) == 0);
            uint64_t hash = cuda_test_float_bits_hash(out + 1, count);
            if (hash != ref->hash)
                fprintf(stderr, "Q6 reference rows=%d cols=%d nt=%d seed=%d alias=%d: %016llx != %016llx\n",
                    rows, cols, nt, ref->seed, alias,
                    (unsigned long long)hash, (unsigned long long)ref->hash);
            assert(hash == ref->hash);
            assert(out[0] == -12345.0f && out[capacity + 1] == -12345.0f);
        }
        if (nt == 1) {
            assert(gpu->matvec(gpu->ctx, out + 1, buffer, input, rows, cols,
                BN_GGUF_TENSOR_Q6_K) == 0);
            assert(cuda_test_float_bits_hash(out + 1, rows) == ref->hash);
            float frequency[16] = {0};
            BnGPUActivationPlan plan = {0};
            plan.dim = plan.xb2_elements = cols;
            plan.hb_elements = rows > cols ? rows : cols; plan.vocab_size = rows + 2;
            plan.n_layers = plan.attention_layer_count = 1; plan.n_heads = 1;
            plan.seq_len = 1; plan.kv_dim = plan.head_size = 32;
            plan.rope_frequencies = frequency; plan.rope_frequency_count = 16;
            assert(gpu->init_activations(gpu->ctx, &plan) == 0);
            for (int flags = 0; flags < 2; flags++) {
                BnGPUOp ops[32] = {{0}};
                ops[0].op_code = BN_GPU_CODE_MATVEC; ops[0].op_kind = BN_GPU_OP_MATVEC;
                ops[0].buf_in = BN_GPU_VALUE_XB; ops[0].buf_out = BN_GPU_VALUE_LOGITS;
                ops[0].buf_aux = -1; ops[0].W_buf = buffer;
                ops[0].rows = rows; ops[0].cols = cols; ops[0].type = BN_GGUF_TENSOR_Q6_K;
                ops[0].flags = flags ? BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT : 0;
                ops[0].p[5] = 1;
                for (int i = 1; i < 32; i++) {
                    ops[i].op_code = BN_GPU_CODE_COPY;
                    ops[i].buf_in = ops[i].buf_out = BN_GPU_VALUE_XB; ops[i].p[2] = 1;
                }
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, input, nx * sizeof(float), 0) == 0);
                for (int graph = 0; graph < 2; graph++) {
                    for (int replay = 0; replay < (graph ? 2 : 1); replay++) {
                        for (int i = 0; i < rows + 2; i++) out[i] = -12345.0f;
                        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_LOGITS, out,
                            (size_t)(rows + 2) * sizeof(float), 0) == 0);
                        assert(gpu->execute(gpu->ctx, ops, graph ? 32 : 1,
                            BN_GPU_VALUE_LOGITS, out, rows + 2) == 0);
                        uint64_t hash = cuda_test_float_bits_hash(out + 1, rows);
                        if (hash != ref->hash)
                            fprintf(stderr, "Q6 graph rows=%d cols=%d seed=%d flags=%d graph=%d replay=%d: %016llx != %016llx\n",
                                rows, cols, ref->seed, flags, graph, replay,
                                (unsigned long long)hash, (unsigned long long)ref->hash);
                        assert(hash == ref->hash);
                        assert(out[0] == -12345.0f && out[rows + 1] == -12345.0f);
                    }
                }
            }
        }
        gpu->buffer_destroy(gpu->ctx, buffer);
        free(out); free(input); free(weights);
    }
    printf("CUDA Q6 small batch and decode reference PASSED\n");
}

#include "cuda_dense_prefill_reference.h"

static void run_dense_prefill_reference_case(BnGPUBackend *gpu) {
    const float scales[] = {1.0f, 0.3f, -0.7f, 0.0f, 1e-8f};
    for (size_t c = 0; c < sizeof(cuda_dense_prefill_reference) /
                                sizeof(cuda_dense_prefill_reference[0]); c++) {
        const BnCudaDensePrefillReferenceCase *ref = &cuda_dense_prefill_reference[c];
        int dim = ref->dim, nt = ref->tokens;
        size_t count = (size_t)dim * nt, kv_count = (size_t)nt * 32;
        float *x = malloc(count * sizeof(float));
        float *out = malloc((count + 2) * sizeof(float));
        float *key = malloc((kv_count + 2) * sizeof(float));
        float *value = malloc((kv_count + 2) * sizeof(float));
        float *identity = calloc((size_t)dim * dim, sizeof(float));
        float *qk = calloc((size_t)64 * dim, sizeof(float));
        float *wv = calloc((size_t)32 * dim, sizeof(float));
        float *wo = calloc((size_t)dim * 32, sizeof(float));
        float *weights = malloc((size_t)dim * 4 * sizeof(float));
        assert(x && out && key && value && identity && qk && wv && wo && weights);
        float *w = weights, *fw = w + dim, *aw = fw + dim, *pw = aw + dim;
        for (int i = 0; i < dim; i++) {
            identity[(size_t)i * dim + i] = 1.0f;
            w[i] = 1.0f; fw[i] = (float)(i % 13 + 1) / 8.0f;
            aw[i] = (float)(i % 19 + 1) / 16.0f;
            pw[i] = (float)(i % 17 + 1) / 16.0f;
            wo[(size_t)i * 32 + i % 32] = (ref->postnorms & 4)
                ? 0.0f : (float)(i % 5 + 1) / 8.0f;
            for (int t = 0; t < nt; t++)
                x[(size_t)t * dim + i] = (float)((i * 17 + t * 13) % 257 - 128) / 128.0f;
        }
        float vb[32], frequency[16] = {0};
        for (int i = 0; i < 32; i++) vb[i] = 1.0f;
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = dim;
        plan.vocab_size = 32; plan.n_layers = plan.attention_layer_count = 1;
        plan.n_heads = 1; plan.seq_len = nt > 32 ? nt : 32; plan.kv_dim = plan.head_size = 32;
        plan.rope_frequencies = frequency; plan.rope_frequency_count = 16;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        float *data[] = {qk, wv, wo, identity, w, fw, aw, pw, vb};
        int rows[] = {64, 32, dim, dim, dim, dim, dim, dim, 32};
        int cols[] = {dim, dim, 32, dim, 1, 1, 1, 1, 1};
        void *buf[9];
        for (int i = 0; i < 9; i++) {
            buf[i] = gpu->buffer_create(gpu->ctx, data[i],
                (size_t)rows[i] * cols[i] * sizeof(float), 0, rows[i], cols[i]);
            assert(buf[i]);
        }
        for (int alias = 0; alias < 2; alias++) {
            out[0] = out[count + 1] = -12345.0f;
            key[0] = key[kv_count + 1] = value[0] = value[kv_count + 1] = -12345.0f;
            for (size_t i = 0; i < kv_count; i++) key[i + 1] = value[i + 1] = -999.0f;
            if (alias) memcpy(out + 1, x, count * sizeof(float));
            else for (size_t i = 0; i < count; i++) out[i + 1] = -999.0f;
            assert(gpu->prefill_dense_layer(gpu->ctx, out + 1,
                buf[0], buf[1], buf[2], buf[3], buf[3], buf[3], buf[4], buf[5],
                ref->postnorms & 1 ? buf[6] : NULL,
                ref->postnorms & 2 ? buf[7] : NULL,
                NULL, NULL, NULL, NULL, buf[8], alias ? out + 1 : x,
                key + 1, value + 1, nt, dim, dim, 1, 1, 32, 1, 32, 64,
                0, 32, 0, dim, 32, 0, 0, 0, 0, BN_MODEL_ACTIVATION_GELU,
                0, 0, 1e-6f, 0, 32, 0, 0, 32, 1.0f, scales[ref->scale], 0, 0) == 0);
            uint64_t hash = cuda_test_float_bits_hash(out + 1, count);
            if (hash != ref->hash)
                fprintf(stderr, "dense prefill reference dim=%d nt=%d postnorms=%d scale=%d alias=%d: %016llx != %016llx\n",
                    dim, nt, ref->postnorms, ref->scale, alias,
                    (unsigned long long)hash, (unsigned long long)ref->hash);
            assert(hash == ref->hash);
            assert(out[0] == -12345.0f && out[count + 1] == -12345.0f);
            assert(key[0] == -12345.0f && key[kv_count + 1] == -12345.0f);
            assert(value[0] == -12345.0f && value[kv_count + 1] == -12345.0f);
            for (size_t i = 0; i < kv_count; i++) {
                assert(key[i + 1] == 0.0f);
                assert(value[i + 1] == 1.0f);
            }
        }
        for (int i = 0; i < 9; i++) gpu->buffer_destroy(gpu->ctx, buf[i]);
        free(weights); free(wo); free(wv); free(qk); free(identity);
        free(value); free(key); free(out); free(x);
    }
    printf("CUDA dense prefill reference PASSED\n");
}

#include "cuda_dense_projection_reference.h"

static void run_dense_gateup_reference_case(BnGPUBackend *gpu) {
    const float scales[] = {1.0f, 0.3f, -0.7f, 0.0f, 1e-8f};
    for (size_t c = 0; c < sizeof(cuda_dense_projection_reference) /
                                sizeof(cuda_dense_projection_reference[0]); c++) {
        const BnCudaDenseProjectionReferenceCase *ref = &cuda_dense_projection_reference[c];
        if (ref->kind != 0) continue;
        int dim = ref->dim, nt = ref->tokens;
        size_t count = (size_t)dim * nt, kv_count = (size_t)nt * 32;
        float *x = malloc(count * sizeof(float));
        float *out = malloc((count + 2) * sizeof(float));
        float *key = malloc((kv_count + 2) * sizeof(float));
        float *value = malloc((kv_count + 2) * sizeof(float));
        float *identity = calloc((size_t)dim * dim, sizeof(float));
        float *qk = calloc((size_t)64 * dim, sizeof(float));
        float *wv = calloc((size_t)32 * dim, sizeof(float));
        float *wo = calloc((size_t)dim * 32, sizeof(float));
        float *weights = malloc((size_t)dim * 4 * sizeof(float));
        assert(x && out && key && value && identity && qk && wv && wo && weights);
        float *w = weights, *fw = w + dim, *aw = fw + dim, *pw = aw + dim;
        for (int i = 0; i < dim; i++) {
            identity[(size_t)i * dim + i] = 1.0f;
            w[i] = 1.0f; fw[i] = (float)(i % 13 + 1) / 8.0f;
            aw[i] = (float)(i % 19 + 1) / 16.0f;
            pw[i] = (float)(i % 17 + 1) / 16.0f;
            wo[(size_t)i * 32 + i % 32] = (float)(i % 5 + 1) / 8.0f;
            for (int t = 0; t < nt; t++)
                x[(size_t)t * dim + i] = (float)((i * 17 + t * 13) % 257 - 128) / 128.0f;
        }
        float vb[32], frequency[16] = {0};
        for (int i = 0; i < 32; i++) vb[i] = 1.0f;
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = dim;
        plan.vocab_size = 32; plan.n_layers = plan.attention_layer_count = 1;
        plan.n_heads = 1; plan.seq_len = 32; plan.kv_dim = plan.head_size = 32;
        plan.rope_frequencies = frequency; plan.rope_frequency_count = 16;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        float *data[] = {qk, wv, wo, identity, w, fw, aw, pw, vb};
        int rows[] = {64, 32, dim, dim, dim, dim, dim, dim, 32};
        int cols[] = {dim, dim, 32, dim, 1, 1, 1, 1, 1};
        void *buf[9];
        for (int i = 0; i < 9; i++) {
            buf[i] = gpu->buffer_create(gpu->ctx, data[i],
                (size_t)rows[i] * cols[i] * sizeof(float), 0, rows[i], cols[i]);
            assert(buf[i]);
        }
        size_t gw_bytes = (size_t)2 * dim * dim / BN_QK_K * kquant_fixture_block_size(ref->type);
        void *gw = malloc(gw_bytes);
        assert(gw);
        kquant_fixture_weights(gw, ref->type, (size_t)2 * dim * dim / BN_QK_K);
        void *gb = gpu->buffer_create(gpu->ctx, gw, gw_bytes, ref->type, dim * 2, dim);
        void *g = gpu->buffer_create(gpu->ctx, gw, gw_bytes / 2, ref->type, dim, dim);
        void *u = gpu->buffer_create(gpu->ctx, (char *)gw + gw_bytes / 2, gw_bytes / 2, ref->type, dim, dim);
        assert(gb && g && u);
        for (int variant = 0; variant < 4; variant++) {
            int alias = variant & 1, stacked = variant < 2;
            size_t ncases = sizeof(cuda_dense_projection_reference) / sizeof(cuda_dense_projection_reference[0]);
            if (variant == 3 && (c + 1 == ncases || cuda_dense_projection_reference[c + 1].kind != 0)) {
                /* Exercise graph-to-prefill ordering without a host readback. */
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, x, sizeof(float), 0) == 0);
                BnGPUOp copy = {0};
                copy.op_code = BN_GPU_CODE_COPY;
                copy.buf_in = BN_GPU_VALUE_X; copy.buf_out = BN_GPU_VALUE_SCRATCH;
                copy.p[2] = 1;
                assert(gpu->execute(gpu->ctx, &copy, 1, -1, NULL, 0) == 0);
            }
            out[0] = out[count + 1] = -12345.0f;
            key[0] = key[kv_count + 1] = value[0] = value[kv_count + 1] = -12345.0f;
            for (size_t i = 0; i < kv_count; i++) key[i + 1] = value[i + 1] = -999.0f;
            if (alias) memcpy(out + 1, x, count * sizeof(float));
            else for (size_t i = 0; i < count; i++) out[i + 1] = -999.0f;
            assert(gpu->prefill_dense_layer(gpu->ctx, out + 1,
                buf[0], buf[1], buf[2], stacked ? gb : g, stacked ? NULL : u, buf[3], buf[4], buf[5],
                ref->postnorms & 1 ? buf[6] : NULL,
                ref->postnorms & 2 ? buf[7] : NULL,
                NULL, NULL, NULL, NULL, buf[8], alias ? out + 1 : x,
                key + 1, value + 1, nt, dim, dim, 1, 1, 32, 1, 32, 64,
                0, 32, 0, dim, 32, 0, ref->type, ref->type, 0, BN_MODEL_ACTIVATION_GELU,
                0, 0, 1e-6f, 0, 32, 0, 0, 32, 1.0f, scales[ref->scale], 0, 0) == 0);
            uint64_t hash = cuda_test_float_bits_hash(out + 1, count);
            if (hash != ref->hash)
                fprintf(stderr, "dense prefill reference dim=%d nt=%d postnorms=%d scale=%d alias=%d: %016llx != %016llx\n",
                    dim, nt, ref->postnorms, ref->scale, alias,
                    (unsigned long long)hash, (unsigned long long)ref->hash);
            assert(hash == ref->hash);
            assert(out[0] == -12345.0f && out[count + 1] == -12345.0f);
            assert(key[0] == -12345.0f && key[kv_count + 1] == -12345.0f);
            assert(value[0] == -12345.0f && value[kv_count + 1] == -12345.0f);
            for (size_t i = 0; i < kv_count; i++) {
                assert(key[i + 1] == 0.0f);
                assert(value[i + 1] == 1.0f);
            }
        }
        gpu->buffer_destroy(gpu->ctx, gb); gpu->buffer_destroy(gpu->ctx, g);
        gpu->buffer_destroy(gpu->ctx, u); free(gw);
        for (int i = 0; i < 9; i++) gpu->buffer_destroy(gpu->ctx, buf[i]);
        free(weights); free(wo); free(wv); free(qk); free(identity);
        free(value); free(key); free(out); free(x);
    }
    printf("CUDA dense gate/up reference PASSED\n");
}

static void run_dense_qk_reference_case(BnGPUBackend *gpu) {
    for (size_t c = 0; c < sizeof(cuda_dense_projection_reference) /
                                sizeof(cuda_dense_projection_reference[0]); c++) {
        const BnCudaDenseProjectionReferenceCase *ref = &cuda_dense_projection_reference[c];
        if (ref->kind != 1) continue;
        int dim = ref->dim, nt = ref->tokens;
        size_t count = (size_t)dim * nt, nk = (size_t)nt * 32;
        size_t qw_bytes = (size_t)2080 * dim / BN_QK_K * kquant_fixture_block_size(ref->type);
        void *qw = malloc(qw_bytes);
        float *x = malloc(count * sizeof(float)), *out = malloc((count + 2) * sizeof(float));
        float *key = malloc((nk + 2) * sizeof(float)), *value = malloc((nk + 2) * sizeof(float));
        float *zero = calloc((size_t)2048 * dim, sizeof(float));
        float *norm = malloc((size_t)dim * sizeof(float));
        assert(qw && x && out && key && value && zero && norm);
        kquant_fixture_weights(qw, ref->type, (size_t)2080 * dim / BN_QK_K);
        for (int i = 0; i < dim; i++) {
            norm[i] = 1.0f;
            for (int t = 0; t < nt; t++)
                x[(size_t)t * dim + i] = (float)((i * 17 + t * 13) % 257 - 128) / 128.0f;
        }
        float frequency[16] = {0}, vb[32];
        for (int i = 0; i < 32; i++) vb[i] = 1.0f;
        BnGPUActivationPlan plan = {0};
        plan.dim = dim; plan.xb2_elements = dim > 2048 ? dim : 2048; plan.hb_elements = 256;
        plan.vocab_size = 32; plan.n_layers = plan.attention_layer_count = 1;
        plan.n_heads = 64; plan.seq_len = 32; plan.kv_dim = 32; plan.head_size = 32;
        plan.rope_frequencies = frequency; plan.rope_frequency_count = 16;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        void *qb = gpu->buffer_create(gpu->ctx, qw, qw_bytes, ref->type, 2080, dim);
        void *v = gpu->buffer_create(gpu->ctx, zero, (size_t)32 * dim * 4, 0, 32, dim);
        void *wo = gpu->buffer_create(gpu->ctx, zero, (size_t)dim * 2048 * 4, 0, dim, 2048);
        void *gu = gpu->buffer_create(gpu->ctx, zero, (size_t)256 * dim * 4, 0, 256, dim);
        void *down = gpu->buffer_create(gpu->ctx, zero, (size_t)dim * 256 * 4, 0, dim, 256);
        void *nw = gpu->buffer_create(gpu->ctx, norm, (size_t)dim * 4, 0, dim, 1);
        void *bias = gpu->buffer_create(gpu->ctx, vb, sizeof(vb), 0, 32, 1);
        assert(qb && v && wo && gu && down && nw && bias);
        /* Preserve the independently generated K weights while inserting
         * a second query-sized block for gated Q. The reference K hash must
         * not depend on the neighboring query projection's row count. */
        size_t row_bytes = qw_bytes / 2080;
        size_t gated_bytes = (size_t)4128 * row_bytes;
        void *gated_weights = malloc(gated_bytes);
        assert(gated_weights);
        memcpy(gated_weights, qw, (size_t)2048 * row_bytes);
        memcpy((char *)gated_weights + (size_t)2048 * row_bytes,
            qw, (size_t)2048 * row_bytes);
        memcpy((char *)gated_weights + (size_t)4096 * row_bytes,
            (char *)qw + (size_t)2048 * row_bytes, (size_t)32 * row_bytes);
        void *gated_qb = gpu->buffer_create(gpu->ctx, gated_weights,
            gated_bytes, ref->type, 4128, dim);
        assert(gated_qb);
        for (int variant = 0; variant < 4; variant++) {
            int alias = variant & 1, gated = variant >= 2;
            for (size_t i = 0; i < count + 2; i++) out[i] = -12345.0f;
            for (size_t i = 0; i < nk + 2; i++) key[i] = value[i] = -12345.0f;
            if (alias) memcpy(out + 1, x, count * sizeof(float));
            assert(gpu->prefill_dense_layer(gpu->ctx, out + 1,
                gated ? gated_qb : qb, v, wo, gu, gu, down, nw, nw, NULL, NULL, NULL, NULL,
                NULL, NULL, bias, alias ? out + 1 : x, key + 1, value + 1,
                nt, dim, 256, 64, 1, 32, 64, 32, gated ? 4128 : 2080, ref->type, 32, 0,
                dim, 2048, 0, 0, 0, 0, BN_MODEL_ACTIVATION_GELU, gated, 0,
                1e-6f, 0, 32, 0, 0, 32, 1.0f, 1.0f, 0, 0) == 0);
            uint64_t hash = cuda_test_float_bits_hash(key + 1, nk);
            if (hash != ref->hash)
                fprintf(stderr, "dense Q/K type=%d dim=%d nt=%d alias=%d: %016llx != %016llx\n",
                    ref->type, dim, nt, alias, (unsigned long long)hash,
                    (unsigned long long)ref->hash);
            assert(hash == ref->hash);
            if (memcmp(out + 1, x, count * sizeof(float)) != 0) {
                size_t different = 0, nonfinite = 0;
                for (size_t i = 0; i < count; i++) {
                    different += memcmp(out + 1 + i, x + i, sizeof(float)) != 0;
                    nonfinite += !isfinite(out[1 + i]);
                }
                fprintf(stderr, "dense Q/K residual type=%d dim=%d nt=%d gated=%d alias=%d diff=%zu nonfinite=%zu\n",
                    ref->type, dim, nt, gated, alias, different, nonfinite);
            }
            assert(memcmp(out + 1, x, count * sizeof(float)) == 0);
            assert(out[0] == -12345.0f && out[count + 1] == -12345.0f);
            assert(key[0] == -12345.0f && key[nk + 1] == -12345.0f);
            assert(value[0] == -12345.0f && value[nk + 1] == -12345.0f);
            for (size_t i = 0; i < nk; i++) assert(value[i + 1] == 1.0f);
        }
        gpu->buffer_destroy(gpu->ctx, gated_qb);
        free(gated_weights);
        void *buffers[] = {qb, v, wo, gu, down, nw, bias};
        for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++)
            gpu->buffer_destroy(gpu->ctx, buffers[i]);
        free(norm); free(zero); free(value); free(key); free(out); free(x); free(qw);
    }
    printf("CUDA dense Q/K reference PASSED\n");
}

#include "cuda_decode_attention_reference.h"
#include "cuda_decode_attention_f32_reference.h"
#include "cuda_window_attention_reference.h"

#include "cuda_prefill_window_reference.h"

static float cuda_test_tf32_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x0fff) + ((bits >> 13) & 1u);
    bits &= UINT32_C(0xffffe000);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void run_prefill_window_paths_case(BnGPUBackend *gpu) {
    enum { dim = 256, hs = 32, nh = 8, kd = 32, qr = dim + kd };
    const int lengths[] = {129, 257, 1025}, windows[] = {0, 3, 128, 1024};
    const int f32 = BN_GGUF_TENSOR_F32, q8 = BN_GGUF_TENSOR_Q8_0;
    const float eps = 1e-6f;
    float *qkw = calloc((size_t)qr * dim, sizeof(float));
    float *vw = calloc((size_t)kd * dim, sizeof(float));
    float *ow = calloc((size_t)dim * dim, sizeof(float));
    float *zero = calloc((size_t)dim * dim, sizeof(float));
    BnBlockQ8_0 *zero8 = calloc(dim, sizeof(*zero8));
    float norm[dim], frequency[hs / 2] = {0};
    assert(qkw && vw && ow && zero && zero8);
    for (int i = 0; i < dim; i++) { norm[i] = 1; ow[(size_t)i * dim + i] = 1; }
    for (int i = 0; i < kd; i++) vw[(size_t)i * dim + i] = 1;
    void *b[] = {
        gpu->buffer_create(gpu->ctx, qkw, (size_t)qr * dim * 4, f32, qr, dim),
        gpu->buffer_create(gpu->ctx, vw, (size_t)kd * dim * 4, f32, kd, dim),
        gpu->buffer_create(gpu->ctx, ow, (size_t)dim * dim * 4, f32, dim, dim),
        gpu->buffer_create(gpu->ctx, norm, sizeof(norm), f32, 1, dim),
        gpu->buffer_create(gpu->ctx, zero, (size_t)dim * dim * 4, f32, dim, dim),
        gpu->buffer_create(gpu->ctx, zero, (size_t)dim * 4, f32, 1, dim),
        gpu->buffer_create(gpu->ctx, zero8, (size_t)dim * sizeof(*zero8), q8, 32, dim),
        gpu->buffer_create(gpu->ctx, zero8, (size_t)dim * sizeof(*zero8), q8, dim, 32),
    };
    for (int i = 0; i < 8; i++) assert(b[i]);
    for (int li = 0; li < 3; li++) for (int half = 0; half < 2; half++) {
        int nt = lengths[li]; size_t nq = (size_t)nt * dim, nk = (size_t)nt * kd;
        float *x = calloc(nq, sizeof(float)), *q = calloc(nq, sizeof(float));
        float *k = calloc(nk, sizeof(float)), *v = calloc(nk, sizeof(float));
        float *out = malloc((nq + 2) * sizeof(float));
        float *ko = malloc((nk + 2) * sizeof(float)), *vo = malloc((nk + 2) * sizeof(float));
        assert(x && q && k && v && out && ko && vo);
        float normalized = 8.0f / sqrtf(64.0f / dim + eps);
        float cached = half ? bn_fp16_to_fp32(bn_fp32_to_fp16(normalized)) : normalized;
        for (int t = 0; t < nt; t++) {
            x[(size_t)t * dim + t % dim] = 8.0f;
            if (t % dim < kd) v[(size_t)t * kd + t % dim] = cached;
        }
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = dim;
        plan.n_layers = plan.attention_layer_count = 1; plan.vocab_size = 32;
        plan.n_heads = nh; plan.seq_len = nt; plan.head_size = hs;
        plan.kv_dim = kd; plan.kv_f16 = half;
        plan.rope_frequencies = frequency; plan.rope_frequency_count = hs / 2;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        BnGPUOp graph_ops[32] = {{0}};
        graph_ops[0].op_code = BN_GPU_CODE_ROPE; graph_ops[0].buf_in = BN_GPU_VALUE_Q;
        graph_ops[0].p[0] = nh; graph_ops[0].p[1] = hs; graph_ops[0].p[3] = 2;
        graph_ops[1].op_code = BN_GPU_CODE_FLASH_ATTN;
        graph_ops[1].buf_in = BN_GPU_VALUE_Q; graph_ops[1].buf_out = BN_GPU_VALUE_XB;
        graph_ops[1].p[0] = nh; graph_ops[1].p[1] = hs; graph_ops[1].p[2] = 1;
        graph_ops[1].p[3] = nh; graph_ops[1].p[4] = kd; graph_ops[1].p[5] = nt;
        graph_ops[1].p[7] = f32_bits(1.0f);
        for (int i = 2; i < 32; i++) {
            graph_ops[i].op_code = BN_GPU_CODE_COPY;
            graph_ops[i].buf_in = BN_GPU_VALUE_HB; graph_ops[i].buf_out = BN_GPU_VALUE_HB2;
            graph_ops[i].p[2] = 1;
        }
        for (int w = 0; w < 4; w++) for (int mode = 0; mode < 7; mode++)
        for (int alias = 0; alias < 2; alias++) {
            int window = windows[w], rc = -1;
            /* Leave decode work asynchronous to verify the stream handoff. */
            assert(gpu->execute(gpu->ctx, graph_ops, 32, -1, NULL, 0) == 0);
            out[0] = out[nq + 1] = ko[0] = ko[nk + 1] = vo[0] = vo[nk + 1] = -12345.0f;
            memcpy(out + 1, x, nq * sizeof(float));
            const float *input = alias ? out + 1 : x;
            if (mode == 0)
                rc = gpu->prefill_attention(gpu->ctx, out + 1, alias ? out + 1 : q,
                    k, v, nt, nh, 1, hs, nh, kd, 1.0f, window);
            else if (mode == 1)
                rc = gpu->prefill_attention_wo(gpu->ctx, out + 1, b[2], q, k, v,
                    nt, nh, 1, hs, nh, kd, dim, dim, f32, 1.0f, window);
            else if (mode == 2)
                rc = gpu->prefill_qkv_attention_wo(gpu->ctx, out + 1, b[0], b[1], b[2],
                    NULL, NULL, input, ko + 1, vo + 1, nt, dim, nh, 1, hs, nh, kd,
                    qr, f32, kd, f32, dim, dim, f32, 0, eps, 0, 2, 1.0f, window);
            else if (mode == 3)
                rc = gpu->prefill_qkv_attention_wo_norm(gpu->ctx, out + 1, b[0], b[1], b[2],
                    b[3], NULL, NULL, input, ko + 1, vo + 1, nt, dim, nh, 1, hs, nh, kd,
                    qr, f32, kd, f32, dim, dim, f32, 0, eps, 0, 2, 1.0f, window);
            else if (mode == 4)
                rc = gpu->prefill_qkv_attention_wo_norm_resid(gpu->ctx, out + 1, b[0], b[1], b[2],
                    b[3], NULL, NULL, input, ko + 1, vo + 1, nt, dim, nh, 1, hs, nh, kd,
                    qr, f32, kd, f32, dim, dim, f32, 0, eps, 0, 2, 1.0f, window);
            else if (mode == 5)
                rc = gpu->prefill_dense_layer(gpu->ctx, out + 1, b[0], b[1], b[2],
                    b[4], b[4], b[4], b[3], b[3], NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, input, ko + 1, vo + 1, nt, dim, dim, nh, 1, hs,
                    nh, kd, qr, f32, kd, f32, dim, dim, f32, f32, f32, f32,
                    0, 0, 0, eps, 0, 2, 0, 0, kd, 1.0f, 1.0f, window, 0);
            else
                rc = gpu->prefill_moe_layer(gpu->ctx, out + 1, b[0], b[1], b[2],
                    b[5], b[6], b[6], b[7], NULL, NULL, NULL, NULL,
                    b[3], b[3], NULL, NULL, NULL, NULL, NULL, input, ko + 1, vo + 1,
                    nt, dim, 32, 1, 1, nh, 1, hs, nh, kd, qr, f32, kd, f32,
                    dim, dim, f32, q8, q8, q8, 0, 0, 0, 0, 0,
                    0, eps, 0, 2, 0, kd, 1.0f, 1, 1.0f, window);
            if (rc) fprintf(stderr, "prefill window path rejected nt=%d half=%d mode=%d window=%d alias=%d\n",
                nt, half, mode, window, alias);
            assert(rc == 0);
            for (int t = 0; t < nt; t++) {
                int first = window > 0 && t + 1 > window ? t + 1 - window : 0;
                for (int d = 0; d < dim; d++) {
                    int count = 0;
                    for (int j = d % hs; j <= t; j += dim) count += j >= first;
                    /* F32 projection GEMMs use TF32 operands. The sparse
                     * identity matrices isolate that rounding from masking. */
                    double factor = mode == 2 ? 8.0 : mode >= 3
                        ? cuda_test_tf32_round(normalized) : cached;
                    double expected = factor * count / (t + 1 - first);
                    if (mode > 0) expected = cuda_test_tf32_round((float)expected);
                    if (mode >= 4) expected += x[(size_t)t * dim + d];
                    float actual = out[1 + (size_t)t * dim + d];
                    if (!isfinite(actual) || fabs(actual - expected) > 0.0002)
                        fprintf(stderr, "prefill window path nt=%d half=%d mode=%d window=%d alias=%d t=%d d=%d got=%.9g expected=%.9g\n",
                            nt, half, mode, window, alias, t, d, actual, expected);
                    assert(isfinite(actual) && fabs(actual - expected) <= 0.0002);
                }
            }
            assert(out[0] == -12345.0f && out[nq + 1] == -12345.0f);
            assert(ko[0] == -12345.0f && ko[nk + 1] == -12345.0f);
            assert(vo[0] == -12345.0f && vo[nk + 1] == -12345.0f);
        }
        gpu->free_activations(gpu->ctx);
        free(vo); free(ko); free(out); free(v); free(k); free(q); free(x);
    }
    for (int i = 0; i < 8; i++) gpu->buffer_destroy(gpu->ctx, b[i]);
    free(zero8); free(zero); free(ow); free(vw); free(qkw);
    printf("CUDA prefill window paths PASSED\n");
}

static void run_prefill_window_reference_cases(BnGPUBackend *gpu, int long_only) {
    for (size_t c = 0; c < sizeof(cuda_prefill_window_reference) /
                                sizeof(cuda_prefill_window_reference[0]); c++) {
        const BnCudaPrefillWindowReference *r = &cuda_prefill_window_reference[c];
        if ((long_only == 1 || long_only == 2) && r->n_tokens <= 32) continue;
        if (long_only == 2 && r->head_size != 512) continue;
        if (long_only == 3 && r->gqa != 6) continue;
        int hs = r->head_size, nk = r->kv_heads, nh = nk * r->gqa;
        int nt = r->n_tokens, dim = nh * hs, kd = nk * hs;
        size_t nq = (size_t)nt * dim, nkeys = (size_t)nt * kd;
        float *q = malloc(nq * sizeof(float));
        float *k = malloc(nkeys * sizeof(float));
        float *v = malloc(nkeys * sizeof(float));
        float *out = malloc((nq + 2) * sizeof(float));
        float *kout = malloc((nkeys + 2) * sizeof(float));
        assert(q && k && v && out && kout);
        float mag = r->seed ? 3.0f : 1.0f;
        for (size_t i = 0; i < nq; i++)
            q[i] = sinf((float)(i * 17 + 3 + r->seed)) * (1.31f * mag);
        for (size_t i = 0; i < nkeys; i++) {
            k[i] = bn_fp16_to_fp32(bn_fp32_to_fp16(sinf((float)(i * 11 + 9 + r->seed)) * (1.7f * mag)));
            v[i] = bn_fp16_to_fp32(bn_fp32_to_fp16(cosf((float)(i * 7 + 5 + r->seed)) * (0.73f * mag)));
        }
        float freq[256] = {0};
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = dim;
        plan.n_layers = plan.attention_layer_count = 1; plan.vocab_size = 32;
        plan.n_heads = nh; plan.seq_len = (nt + 255) / 256 * 256; plan.head_size = hs;
        plan.kv_dim = kd; plan.kv_f16 = 1;
        plan.rope_frequencies = freq; plan.rope_frequency_count = hs / 2;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        float scale = r->unit_scale ? 1.0f : 1.0f / sqrtf((float)hs);
        for (int mode = 0; mode < 4; mode++) {
            out[0] = out[nq + 1] = kout[0] = kout[nkeys + 1] = -12345.0f;
            memcpy(out + 1, q, nq * sizeof(float));
            const float *input = mode & 1 ? out + 1 : q;
            if (mode < 2) {
                assert(gpu->prefill_attention(gpu->ctx, out + 1, input, k, v,
                    nt, nh, nk, hs, r->gqa, kd, scale, r->window) == 0);
            } else {
                BnGPUAttentionPrefillPlan p = {0};
                p.n_tokens = nt; p.n_heads = nh; p.n_kv_heads = nk;
                p.head_size = hs; p.q_row_stride = dim; p.rope_dims = 2;
                p.norm_eps = 1e-6f; p.attention_scale = scale;
                p.attention_window = r->window;
                assert(gpu->prefill_attention_prepared(gpu->ctx, out + 1,
                    kout + 1, input, k, v, NULL, NULL, &p) == 0);
                assert(memcmp(kout + 1, k, nkeys * sizeof(float)) == 0);
            }
            uint64_t actual = cuda_test_float_bits_hash(out + 1, nq);
            if (actual != r->hash)
                fprintf(stderr, "prefill window case=%zu mode=%d width=%d keys=%d window=%d hash=%016llx expected=%016llx\n",
                    c, mode, hs, nt, r->window, (unsigned long long)actual,
                    (unsigned long long)r->hash);
            assert(actual == r->hash);
            assert(out[0] == -12345.0f && out[nq + 1] == -12345.0f);
            assert(kout[0] == -12345.0f && kout[nkeys + 1] == -12345.0f);
        }
        gpu->free_activations(gpu->ctx);
        free(kout); free(out); free(v); free(k); free(q);
    }
    if (!long_only) run_prefill_window_paths_case(gpu);
    printf("CUDA prefill window reference PASSED\n");
}

static void run_graph_teardown_case(void) {
    /* Destroy an owner with a live captured graph. Calling free_activations
     * first would hide duplicate graph destruction in the backend destructor. */
    for (int cycle = 0; cycle < 4; cycle++) {
        BnGPUBackend *gpu = bn_gpu_cuda_create();
        assert(gpu);
        float frequency[128] = {0};
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = 512;
        plan.vocab_size = 32; plan.n_layers = plan.attention_layer_count = 1;
        plan.n_heads = 2; plan.head_size = plan.kv_dim = 256;
        plan.seq_len = 512; plan.kv_f16 = 1;
        plan.rope_frequencies = frequency; plan.rope_frequency_count = 128;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        BnGPUOp ops[32] = {{0}};
        ops[0].op_code = BN_GPU_CODE_ROPE; ops[0].buf_in = BN_GPU_VALUE_Q;
        ops[0].p[0] = 2; ops[0].p[1] = ops[0].p[3] = 256;
        ops[1].op_code = BN_GPU_CODE_FLASH_ATTN;
        ops[1].buf_in = BN_GPU_VALUE_Q; ops[1].buf_out = BN_GPU_VALUE_XB;
        ops[1].p[0] = 2; ops[1].p[1] = 256; ops[1].p[2] = 257;
        ops[1].p[3] = 2; ops[1].p[4] = 256; ops[1].p[5] = 512;
        ops[1].p[7] = f32_bits(1.0f);
        for (int i = 2; i < 32; i++) {
            ops[i].op_code = BN_GPU_CODE_COPY;
            ops[i].buf_in = BN_GPU_VALUE_HB; ops[i].buf_out = BN_GPU_VALUE_HB2;
            ops[i].p[2] = 1;
        }
        for (int replay = 0; replay < 2; replay++)
            assert(gpu->execute(gpu->ctx, ops, 32, -1, NULL, 0) == 0);
        if (cycle & 1) {
            float out[512];
            assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_XB, out, sizeof(out), 0) == 0);
            for (int i = 0; i < 512; i++) assert(out[i] == 0.0f);
        }
        bn_gpu_cuda_destroy(gpu);
    }
    printf("CUDA graph teardown PASSED\n");
}

static void run_decode_attention_partition_replay_case(BnGPUBackend *gpu, int head_size, int gqa) {
    /* Keep one graph and cache allocation while crossing a partition boundary
     * in both directions. The reference inputs are prefixes of the full cache. */
    const int counts512[] = {1504, 1505, 1536, 2048, 1024, 1505};
    const int counts256[] = {257, 512, 513, 1025, 2048, 512};
    const int *valid_counts = head_size == 256 ? counts256 : counts512;
    const int nk = head_size == 256 ? (gqa == 16 ? 2 : 16) : 8;
    const int nh = nk * gqa, seq = 8192;
    const size_t dim = (size_t)nh * head_size, kv_dim = (size_t)nk * head_size;
    const size_t kv_values = (size_t)2048 * kv_dim;
    float *q = malloc(dim * sizeof(float));
    uint16_t *k = malloc(kv_values * sizeof(uint16_t));
    uint16_t *v = malloc(kv_values * sizeof(uint16_t));
    float *out = malloc((dim + 2) * sizeof(float));
    assert(q && k && v && out);
    for (size_t i = 0; i < dim; i++)
        q[i] = sinf((float)(i * 17 + 26)) * (1.31f * 3.0f);
    for (size_t i = 0; i < kv_values; i++) {
        k[i] = bn_fp32_to_fp16(sinf((float)(i * 11 + 32)) * (1.7f * 3.0f));
        v[i] = bn_fp32_to_fp16(cosf((float)(i * 7 + 28)) * (0.73f * 3.0f));
    }
    float frequency[256] = {0};
    for (int unit = 0; unit < 2; unit++) {
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = (int)dim;
        plan.vocab_size = 32; plan.n_layers = plan.attention_layer_count = 1;
        plan.n_heads = nh; plan.seq_len = seq; plan.head_size = head_size;
        plan.kv_dim = (int)kv_dim; plan.kv_f16 = 1;
        plan.rope_frequencies = frequency; plan.rope_frequency_count = head_size / 2;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE, k, kv_values * 2, 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_VALUE_CACHE, v, kv_values * 2, 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB, q, 4, 0) == 0);
        BnGPUOp ops[32] = {{0}};
        ops[0].op_code = BN_GPU_CODE_ROPE; ops[0].buf_in = BN_GPU_VALUE_Q;
        ops[0].p[0] = nh; ops[0].p[1] = head_size; ops[0].p[3] = head_size;
        ops[1].op_code = BN_GPU_CODE_FLASH_ATTN;
        ops[1].buf_in = BN_GPU_VALUE_Q; ops[1].buf_out = BN_GPU_VALUE_XB;
        ops[1].p[0] = nh; ops[1].p[1] = head_size; ops[1].p[3] = gqa;
        ops[1].p[4] = (uint32_t)kv_dim; ops[1].p[5] = seq;
        float scale = unit ? 1.0f : 1.0f / sqrtf((float)head_size);
        memcpy(&ops[1].p[7], &scale, 4);
        for (int i = 2; i < 32; i++) {
            ops[i].op_code = BN_GPU_CODE_COPY;
            ops[i].buf_in = BN_GPU_VALUE_HB; ops[i].buf_out = BN_GPU_VALUE_HB2;
            ops[i].p[2] = 1;
        }
        for (size_t i = 0; i < sizeof(counts512) / sizeof(counts512[0]); i++) {
            int valid = valid_counts[i];
            uint64_t expected = 0;
            if (head_size == 256) {
                for (size_t c = 0; c < sizeof(cuda_window_attention_reference) /
                                            sizeof(cuda_window_attention_reference[0]); c++) {
                    const BnCudaWindowAttentionReferenceCase *r = &cuda_window_attention_reference[c];
                    if (r->head_size == head_size && r->gqa == gqa && r->valid == valid && r->kv_heads == nk &&
                        r->seed == 23 && r->unit_scale == unit && r->window == 0)
                        expected = r->hash;
                }
            } else {
                for (size_t c = 0; c < sizeof(cuda_decode_attention_reference) /
                                            sizeof(cuda_decode_attention_reference[0]); c++) {
                    const BnCudaDecodeAttentionReferenceCase *r = &cuda_decode_attention_reference[c];
                    if (r->valid == valid && r->kv_heads == nk && r->seed == 23 && r->unit_scale == unit)
                        expected = r->hash;
                }
            }
            assert(expected);
            ops[0].p[2] = valid - 1; ops[1].p[2] = valid;
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_Q, q, dim * 4, 0) == 0);
            out[0] = out[dim + 1] = -12345.0f;
            assert(gpu->execute(gpu->ctx, ops, 32, -1, NULL, 0) == 0);
            assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_XB, out + 1, dim * 4, 0) == 0);
            assert(cuda_test_float_bits_hash(out + 1, dim) == expected);
            assert(out[0] == -12345.0f && out[dim + 1] == -12345.0f);
        }
        gpu->free_activations(gpu->ctx);
    }
    free(out); free(v); free(k); free(q);
}

static void run_attention_window_replay_case(BnGPUBackend *gpu) {
    const int widths[] = {32, 256, 512};
    const int windows[] = {0, 3, 16, 0, 1, 3};
    const int valid = 257, seq = 512, nh = 8;
    for (int wi = 0; wi < 3; wi++) for (int half = 0; half < 2; half++) {
        int width = widths[wi], dim = nh * width;
        size_t kv_count = (size_t)valid * width;
        float *q = calloc((size_t)dim, sizeof(float));
        float *k = calloc(kv_count, sizeof(float));
        float *v = malloc(kv_count * sizeof(float));
        uint16_t *kh = calloc(kv_count, sizeof(uint16_t));
        uint16_t *vh = malloc(kv_count * sizeof(uint16_t));
        float *out = malloc(((size_t)dim + 2) * sizeof(float));
        assert(q && k && v && kh && vh && out);
        for (int t = 0; t < valid; t++) for (int d = 0; d < width; d++) {
            size_t i = (size_t)t * width + d;
            v[i] = (float)(t % 8) * 0.125f;
            vh[i] = bn_fp32_to_fp16(v[i]);
        }
        float frequency[256] = {0};
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = dim;
        plan.vocab_size = 32; plan.n_layers = plan.attention_layer_count = 1;
        plan.n_heads = nh; plan.seq_len = seq; plan.head_size = width;
        plan.kv_dim = width; plan.kv_f16 = half;
        plan.rope_frequencies = frequency; plan.rope_frequency_count = width / 2;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE,
            half ? (void *)kh : (void *)k, kv_count * (half ? 2 : 4), 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_VALUE_CACHE,
            half ? (void *)vh : (void *)v, kv_count * (half ? 2 : 4), 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB, q, 4, 0) == 0);
        float one = 1.0f;
        void *logits_weight = gpu->buffer_create(gpu->ctx, &one, sizeof(one),
            BN_GGUF_TENSOR_F32, 1, 1);
        assert(logits_weight);
        for (int mode = 0; mode < 4; mode++) {
            int split = mode & 1, graph = mode >= 2;
            BnGPUOp ops[32] = {{0}};
            int ai = graph ? 1 : 0;
            BnGPUOp *attn = &ops[ai];
            attn->op_code = split ? BN_GPU_CODE_GQA_SCORES : BN_GPU_CODE_FLASH_ATTN;
            attn->buf_in = BN_GPU_VALUE_Q; attn->buf_out = BN_GPU_VALUE_XB;
            attn->p[0] = nh; attn->p[1] = width; attn->p[2] = valid;
            attn->p[3] = nh; attn->p[4] = width; attn->p[5] = seq;
            attn->p[7] = f32_bits(1.0f);
            if (ai) {
                ops[0].op_code = BN_GPU_CODE_ROPE; ops[0].buf_in = BN_GPU_VALUE_Q;
                ops[0].p[0] = nh; ops[0].p[1] = width;
                ops[0].p[2] = valid - 1; ops[0].p[3] = width;
            }
            int count = ai + 1;
            if (split) {
                ops[ai + 1].op_code = BN_GPU_CODE_SOFTMAX;
                ops[ai + 1].p[0] = nh; ops[ai + 1].p[1] = valid; ops[ai + 1].p[2] = seq;
                ops[ai + 2] = *attn; ops[ai + 2].op_code = BN_GPU_CODE_GQA_COMBINE;
                count = ai + 3;
            }
            for (int i = count; i < 32; i++) {
                ops[i].op_code = BN_GPU_CODE_COPY;
                ops[i].buf_in = BN_GPU_VALUE_HB; ops[i].buf_out = BN_GPU_VALUE_HB2;
                ops[i].p[2] = 1;
            }
            /* A real logits operation selects the runtime-parameter capture
             * path used during model decode, including static kernel args. */
            ops[31] = (BnGPUOp){0};
            ops[31].op_kind = BN_GPU_OP_LOGITS; ops[31].op_code = BN_GPU_CODE_MATVEC;
            ops[31].W_buf = logits_weight; ops[31].type = BN_GGUF_TENSOR_F32;
            ops[31].rows = ops[31].cols = 1;
            ops[31].buf_in = BN_GPU_VALUE_HB; ops[31].buf_out = BN_GPU_VALUE_LOGITS;
            /* Change a static mask with an unchanged graph size, then replay.
             * Zero scores make the expected result the visible V mean. */
            for (size_t w = 0; w < sizeof(windows) / sizeof(windows[0]); w++) {
                int window = windows[w], first = window ? valid - window : 0;
                double expected = 0;
                for (int t = first; t < valid; t++) expected += (t % 8) * 0.125;
                expected /= valid - first;
                attn->attention_window = window;
                for (int replay = 0; replay < 2; replay++) {
                    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_Q,
                        q, (size_t)dim * 4, 0) == 0);
                    out[0] = out[dim + 1] = -12345.0f;
                    assert(gpu->execute(gpu->ctx, ops, graph ? 32 : count,
                        graph ? -1 : BN_GPU_VALUE_XB, graph ? NULL : out + 1,
                        graph ? 0 : dim) == 0);
                    if (graph) assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_XB,
                        out + 1, (size_t)dim * 4, 0) == 0);
                    for (int i = 0; i < dim; i++) {
                        if (!isfinite(out[i + 1]) || fabs(out[i + 1] - expected) >= 0.002)
                            fprintf(stderr, "window replay width=%d half=%d mode=%d window=%d replay=%d index=%d got=%.9g expected=%.9g\n",
                                width, half, mode, window, replay, i, out[i + 1], expected);
                        assert(isfinite(out[i + 1]) && fabs(out[i + 1] - expected) < 0.002);
                    }
                    assert(out[0] == -12345.0f && out[dim + 1] == -12345.0f);
                }
            }
        }
        gpu->buffer_destroy(gpu->ctx, logits_weight);
        gpu->free_activations(gpu->ctx);
        free(out); free(vh); free(kh); free(v); free(k); free(q);
    }
    printf("CUDA attention window replay PASSED\n");
}

static void run_decode_attention_f32_reference_cases(BnGPUBackend *gpu) {
    const int d = 256, gqa = 6, nk = 4, nh = nk * gqa, seq = 256;
    const size_t dim = (size_t)nh * d, kv_dim = (size_t)nk * d;
    float *q = malloc(dim * sizeof(float));
    float *cache_k = malloc((size_t)seq * kv_dim * sizeof(float));
    float *cache_v = malloc((size_t)seq * kv_dim * sizeof(float));
    float *out = malloc(dim * sizeof(float));
    assert(q && cache_k && cache_v && out);
    BnGPUActivationPlan plan = {0};
    plan.dim = plan.hb_elements = plan.xb2_elements = (int)dim;
    plan.vocab_size = 32; plan.n_layers = plan.attention_layer_count = 1;
    plan.n_heads = nh; plan.seq_len = seq; plan.head_size = d;
    plan.kv_dim = (int)kv_dim; plan.kv_f16 = 0;
    assert(gpu->init_activations(gpu->ctx, &plan) == 0);
    for (size_t c = 0; c < sizeof(cuda_decode_attention_f32_reference) /
                                sizeof(cuda_decode_attention_f32_reference[0]); c++) {
        const BnCudaDecodeAttentionF32ReferenceCase *ref =
            &cuda_decode_attention_f32_reference[c];
        float mag = ref->seed ? 3.0f : 1.0f;
        for (size_t i = 0; i < dim; i++)
            q[i] = sinf((float)(i * 17 + 3 + ref->seed)) * (1.31f * mag);
        memset(cache_k, 0, (size_t)seq * kv_dim * sizeof(float));
        memset(cache_v, 0, (size_t)seq * kv_dim * sizeof(float));
        /* llama's oracle tensors are [head_size, padded_keys, kv_heads].
         * Publish the same semantic values in the runtime's token-major cache. */
        for (int h = 0; h < nk; h++) for (int t = 0; t < ref->valid; t++)
            for (int j = 0; j < d; j++) {
                size_t oracle_i = ((size_t)h * seq + t) * d + j;
                size_t cache_i = ((size_t)t * nk + h) * d + j;
                cache_k[cache_i] = sinf((float)(oracle_i * 11 + 9 + ref->seed)) * (1.7f * mag);
                cache_v[cache_i] = cosf((float)(oracle_i * 7 + 5 + ref->seed)) * (0.73f * mag);
            }
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_Q, q, dim * 4, 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE,
            cache_k, (size_t)seq * kv_dim * 4, 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_VALUE_CACHE,
            cache_v, (size_t)seq * kv_dim * 4, 0) == 0);
        BnGPUOp ops[3] = {{0}};
        ops[0].op_code = BN_GPU_CODE_FLASH_ATTN; ops[0].buf_in = BN_GPU_VALUE_Q;
        ops[0].buf_out = BN_GPU_VALUE_XB;
        ops[0].p[0]=nh; ops[0].p[1]=d; ops[0].p[2]=ref->valid; ops[0].p[3]=gqa;
        ops[0].p[4]=(uint32_t)kv_dim; ops[0].p[5]=seq;
        ops[0].p[7]=f32_bits(1.0f/sqrtf((float)d));
        assert(gpu->execute(gpu->ctx, ops, 1, BN_GPU_VALUE_XB, out, (int)dim) == 0);
        uint64_t actual = cuda_test_float_bits_hash(out, dim);
        if (actual != ref->hash) {
            fprintf(stderr, "F32 decode attention keys=%d seed=%d hash=%016llx expected=%016llx\n",
                ref->valid, ref->seed, (unsigned long long)actual,
                (unsigned long long)ref->hash);
            for (int h = 0; h < nh; h++) fprintf(stderr, " h%d=%a", h, out[(size_t)h*d]);
            fputc('\n', stderr);
        }
        assert(actual == ref->hash);
        ops[0].op_code = BN_GPU_CODE_GQA_SCORES;
        ops[1].op_code = BN_GPU_CODE_SOFTMAX;
        ops[1].p[0] = nh; ops[1].p[1] = ref->valid; ops[1].p[2] = seq;
        ops[2] = ops[0]; ops[2].op_code = BN_GPU_CODE_GQA_COMBINE;
        ops[2].buf_out = BN_GPU_VALUE_XB;
        assert(gpu->execute(gpu->ctx, ops, 3, BN_GPU_VALUE_XB,
                            out, (int)dim) == 0);
        actual = cuda_test_float_bits_hash(out, dim);
        if (actual != ref->hash)
            fprintf(stderr, "F32 split decode attention keys=%d seed=%d hash=%016llx expected=%016llx\n",
                ref->valid, ref->seed, (unsigned long long)actual,
                (unsigned long long)ref->hash);
        assert(actual == ref->hash);
    }
    gpu->free_activations(gpu->ctx);
    free(out); free(cache_v); free(cache_k); free(q);
    printf("CUDA F32 decode attention reference PASSED\n");
}

static void run_decode_attention_reference_cases(BnGPUBackend *gpu, int head_filter, int gqa_filter) {
    const size_t full_count = sizeof(cuda_decode_attention_reference) /
                              sizeof(cuda_decode_attention_reference[0]);
    const size_t window_count = sizeof(cuda_window_attention_reference) /
                                sizeof(cuda_window_attention_reference[0]);
    for (size_t c = 0; c < full_count + window_count; c++) {
        BnCudaDecodeAttentionReferenceCase selected;
        int head_size = 512, gqa = 8, window = 0;
        if (c < full_count) {
            selected = cuda_decode_attention_reference[c];
        } else {
            const BnCudaWindowAttentionReferenceCase *w =
                &cuda_window_attention_reference[c - full_count];
            selected = (BnCudaDecodeAttentionReferenceCase){
                w->valid, w->kv_heads, w->seed, w->unit_scale, w->hash};
            head_size = w->head_size; gqa = w->gqa; window = w->window;
        }
        if (head_filter && head_size != head_filter) continue;
        if (gqa_filter && gqa != gqa_filter) continue;
        const BnCudaDecodeAttentionReferenceCase *ref = &selected;
        int valid = ref->valid, nk = ref->kv_heads, nh = nk * gqa;
        int seq = ((valid + 255) / 256) * 256;
        size_t dim = (size_t)nh * head_size, kv_dim = (size_t)nk * head_size;
        size_t kv_values = (size_t)valid * kv_dim;
        float *q = malloc(dim * sizeof(float));
        uint16_t *k = malloc(kv_values * sizeof(uint16_t));
        uint16_t *v = malloc(kv_values * sizeof(uint16_t));
        float *out = malloc((dim + 2) * sizeof(float));
        assert(q && k && v && out);
        float mag = ref->seed ? 3.0f : 1.0f;
        for (size_t i = 0; i < dim; i++)
            q[i] = sinf((float)(i * 17 + 3 + ref->seed)) * (1.31f * mag);
        for (size_t i = 0; i < kv_values; i++) {
            k[i] = bn_fp32_to_fp16(sinf((float)(i * 11 + 9 + ref->seed)) * (1.7f * mag));
            v[i] = bn_fp32_to_fp16(cosf((float)(i * 7 + 5 + ref->seed)) * (0.73f * mag));
        }
        float frequency[256] = {0};
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = (int)dim;
        plan.vocab_size = 32; plan.n_layers = plan.attention_layer_count = 2;
        plan.n_heads = nh; plan.seq_len = seq; plan.head_size = head_size;
        plan.kv_dim = (int)kv_dim; plan.kv_f16 = 1;
        plan.rope_frequencies = frequency; plan.rope_frequency_count = 256;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        float scale = ref->unit_scale ? 1.0f : 1.0f / sqrtf((float)head_size);
        /* Plain, in-place, fused RoPE, and runtime graph build/replay. Zero
         * frequencies isolate attention while exercising both RoPE hooks. */
        for (int mode = 0; mode < 4; mode++) {
            size_t layer_offset = (mode & 1) ? (size_t)seq * kv_dim : 0;
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE,
                k, kv_values * 2, layer_offset * 2) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_VALUE_CACHE,
                v, kv_values * 2, layer_offset * 2) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB, q, 4, 0) == 0);
            BnGPUOp ops[32] = {{0}};
            int ai = mode >= 2 ? 1 : 0;
            BnGPUOp *attn = &ops[ai];
            attn->op_code = BN_GPU_CODE_FLASH_ATTN;
            attn->attention_window = window;
            attn->buf_in = BN_GPU_VALUE_Q;
            attn->buf_out = mode == 1 ? BN_GPU_VALUE_Q : BN_GPU_VALUE_XB;
            attn->p[0] = nh; attn->p[1] = head_size; attn->p[2] = valid;
            attn->p[3] = gqa; attn->p[4] = (uint32_t)kv_dim;
            attn->p[5] = seq; attn->p[6] = (uint32_t)layer_offset;
            memcpy(&attn->p[7], &scale, 4);
            if (ai) {
                ops[0].op_code = BN_GPU_CODE_ROPE; ops[0].buf_in = BN_GPU_VALUE_Q;
                ops[0].p[0] = nh; ops[0].p[1] = head_size;
                ops[0].p[2] = valid - 1; ops[0].p[3] = head_size;
            }
            for (int i = ai + 1; i < 32; i++) {
                ops[i].op_code = BN_GPU_CODE_COPY;
                ops[i].buf_in = BN_GPU_VALUE_HB; ops[i].buf_out = BN_GPU_VALUE_HB2;
                ops[i].p[2] = 1;
            }
            for (int replay = 0; replay < (mode == 3 ? 2 : 1); replay++) {
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_Q, q, dim * 4, 0) == 0);
                out[0] = out[dim + 1] = -12345.0f;
                if (mode == 3) {
                    assert(gpu->execute(gpu->ctx, ops, 32, -1, NULL, 0) == 0);
                    assert(gpu->read_activation(gpu->ctx, attn->buf_out, out + 1, dim * 4, 0) == 0);
                } else {
                    assert(gpu->execute(gpu->ctx, ops, ai + 1, attn->buf_out, out + 1, (int)dim) == 0);
                }
                uint64_t hash = cuda_test_float_bits_hash(out + 1, dim);
                if (hash != ref->hash)
                    fprintf(stderr, "decode attention keys=%d heads=%d seed=%d unit=%d mode=%d replay=%d width=%d window=%d hash=%016llx expected=%016llx\n",
                        valid, nk, ref->seed, ref->unit_scale, mode, replay, head_size, window,
                        (unsigned long long)hash, (unsigned long long)ref->hash);
                assert(hash == ref->hash);
                assert(out[0] == -12345.0f && out[dim + 1] == -12345.0f);
            }
        }
        gpu->free_activations(gpu->ctx);
        free(out); free(v); free(k); free(q);
    }
    if (!head_filter) run_decode_attention_partition_replay_case(gpu, 512, 8);
    run_decode_attention_partition_replay_case(gpu, 256, 2);
    run_decode_attention_partition_replay_case(gpu, 256, 16);
    run_attention_window_replay_case(gpu);
    printf("CUDA decode attention reference PASSED\n");
}

static void run_q4_decode_reference_case(BnGPUBackend *gpu) {
    /* Independent ggml_mul_mat CUDA graphs, llama.cpp commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e, SM120. Hash every output
     * word to detect both per-block FMA and inter-warp reduction changes. */
    const struct { int cols; unsigned seed; uint64_t hash; } cases[] = {
        {256, 0, UINT64_C(0x14ec2cc988637465)},
        {256, 23, UINT64_C(0xc6c2147a333a8f71)},
        {512, 0, UINT64_C(0x885f06d7ed3757f9)},
        {512, 23, UINT64_C(0xf82fefe7767b974f)},
        {768, 0, UINT64_C(0x75741e41d5afbd00)},
        {768, 23, UINT64_C(0x7ece4e1998234987)},
        {1024, 0, UINT64_C(0x406d9faff96fb4d8)},
        {1024, 23, UINT64_C(0xfd243c38931d09de)},
        {1536, 0, UINT64_C(0xb8170ef97ab8fa2c)},
        {1536, 23, UINT64_C(0x3eedc3853bb9f22b)},
        {2048, 0, UINT64_C(0x33dff56574d6ba58)},
        {2048, 23, UINT64_C(0xf35ccd68d44f56af)},
        {3072, 0, UINT64_C(0xb0eac8033d9107ce)},
        {3072, 23, UINT64_C(0x0d51d98a3eabb2a4)},
        {4096, 0, UINT64_C(0xaa5fd4b04e6e826e)},
        {4096, 23, UINT64_C(0x485a882bd2e9001b)},
        {5376, 0, UINT64_C(0x0fb464427aba4c78)},
        {5376, 23, UINT64_C(0x7f327be4e1eb1853)},
        {8192, 0, UINT64_C(0xaf97f19993a3a67c)},
        {8192, 23, UINT64_C(0xf9dcdf1a350a5dd2)},
        {8448, 0, UINT64_C(0x219d2fe4db91a51f)},
        {8448, 23, UINT64_C(0x83936fe979810333)},
        {8960, 0, UINT64_C(0xd4749eab004b59a5)},
        {8960, 23, UINT64_C(0xe4c529140e6225ca)},
        {9728, 0, UINT64_C(0xfa1c259a72f36e6b)},
        {9728, 23, UINT64_C(0x19c5483a84939f5e)},
        {12288, 0, UINT64_C(0x9c082ec83bc4cd7d)},
        {12288, 23, UINT64_C(0xf1a471433853514f)},
        {14336, 0, UINT64_C(0x0b24af691a97789f)},
        {14336, 23, UINT64_C(0x1811887ccfda68d7)},
        {16384, 0, UINT64_C(0xb9ba038fdc3ea04f)},
        {16384, 23, UINT64_C(0x9a8dba95d581bb65)},
        {21504, 0, UINT64_C(0x43f45aa427d59aec)},
        {21504, 23, UINT64_C(0xa94c3e320b0f8699)},
        {28672, 0, UINT64_C(0x4413afe51d2d0234)},
        {28672, 23, UINT64_C(0x6a88d99679937510)},
        {32768, 0, UINT64_C(0x1181d2c61001553e)},
        {32768, 23, UINT64_C(0xfb64548ed6e677da)},
        {65536, 0, UINT64_C(0xd23dfc7d8d85a30f)},
        {65536, 23, UINT64_C(0x0870fde583212bc3)},
        {2560, 0, UINT64_C(0x6e4d62bca1ce9197)},
        {2560, 23, UINT64_C(0x2792e8b85fc1fd4f)},
    };
    const int rows = 37;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int cols = cases[c].cols;
        size_t bytes = (size_t)rows * (cols / BN_QK_K) * sizeof(BnBlockQ4K);
        BnBlockQ4K *weights = (BnBlockQ4K *)malloc(bytes);
        float *input = (float *)malloc((size_t)cols * sizeof(float));
        float output[39];
        assert(weights && input);
        fill_q4_decode_reference(weights, input, rows, cols, cases[c].seed);
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = cols;
        plan.vocab_size = rows;
        plan.n_layers = plan.seq_len = plan.n_heads = plan.head_size = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        void *buffer = gpu->buffer_create(gpu->ctx, weights, bytes,
                                         BN_GGUF_TENSOR_Q4_K, rows, cols);
        assert(buffer);
        for (unsigned flags = 0; flags <= 1; flags++) {
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X,
                                         input, (size_t)cols * 4, 0) == 0);
            BnGPUOp op = {0};
            op.op_code = BN_GPU_CODE_MATVEC;
            op.buf_in = BN_GPU_VALUE_X;
            op.buf_out = BN_GPU_VALUE_LOGITS;
            op.W_buf = buffer;
            op.type = BN_GGUF_TENSOR_Q4_K;
            op.rows = rows;
            op.cols = cols;
            op.flags = flags;
            output[0] = output[rows + 1] = -12345.0f;
            assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_LOGITS,
                                output + 1, rows) == 0);
            uint64_t hash = cuda_test_float_bits_hash(output + 1, rows);
            if (hash != cases[c].hash)
                fprintf(stderr, "Q4 decode cols=%d seed=%u flags=%u hash=%016llx expected=%016llx\n",
                        cols, cases[c].seed, flags, (unsigned long long)hash,
                        (unsigned long long)cases[c].hash);
            assert(hash == cases[c].hash);
            assert(output[0] == -12345.0f && output[rows + 1] == -12345.0f);
        }
        /* The host matvec and one-token matmul APIs share decode arithmetic. */
        for (int matmul = 0; matmul <= 1; matmul++) {
            output[0] = output[rows + 1] = -12345.0f;
            int rc = matmul
                ? gpu->matmul(gpu->ctx, output + 1, buffer, input,
                              rows, cols, 1, BN_GGUF_TENSOR_Q4_K)
                : gpu->matvec(gpu->ctx, output + 1, buffer, input,
                              rows, cols, BN_GGUF_TENSOR_Q4_K);
            assert(rc == 0);
            assert(cuda_test_float_bits_hash(output + 1, rows) == cases[c].hash);
            assert(output[0] == -12345.0f && output[rows + 1] == -12345.0f);
        }
        run_kquant_split_decode_reference(gpu, buffer, input, cols,
                                          BN_GGUF_TENSOR_Q4_K, cases[c].hash);
        if (cols == 8448 && cases[c].seed == 0) {
            /* Preserve the explicit one-warp diagnostic override. This hash
             * is from the pre-change backend, not the parity reference. */
            BnBackendRuntimePolicy policy = {0};
            assert(bn_backend_runtime_policy_clone(&policy, &gpu->runtime_policy) == 0);
            assert(bn_backend_runtime_policy_set(&policy,
                "BN_CUDA_DISABLE_ASYMMETRIC_KQUANT_4WARP", "1", 1) == 0);
            BnGPUBackend *disabled = bn_gpu_cuda_create_with_policy(&policy);
            bn_backend_runtime_policy_free(&policy);
            assert(disabled);
            void *disabled_buffer = disabled->buffer_create(disabled->ctx,
                weights, bytes, BN_GGUF_TENSOR_Q4_K, rows, cols);
            assert(disabled_buffer);
            for (int matmul = 0; matmul <= 1; matmul++) {
                output[0] = output[rows + 1] = -12345.0f;
                int rc = matmul
                    ? disabled->matmul(disabled->ctx, output + 1, disabled_buffer,
                                       input, rows, cols, 1, BN_GGUF_TENSOR_Q4_K)
                    : disabled->matvec(disabled->ctx, output + 1, disabled_buffer,
                                       input, rows, cols, BN_GGUF_TENSOR_Q4_K);
                assert(rc == 0);
                assert(cuda_test_float_bits_hash(output + 1, rows) ==
                       UINT64_C(0xc2bdeaa12cd3a4fa));
                assert(output[0] == -12345.0f && output[rows + 1] == -12345.0f);
            }
            disabled->buffer_destroy(disabled->ctx, disabled_buffer);
            bn_gpu_cuda_destroy(disabled);
        }
        gpu->buffer_destroy(gpu->ctx, buffer);
        free(input);
        free(weights);
    }
    printf("CUDA Q4 decode reference PASSED\n");
}

static void fill_q5_decode_reference(BnBlockQ5K *w, float *x,
                                     int rows, int cols, unsigned seed) {
    unsigned state = seed + 17u;
    for (int b = 0; b < rows * (cols / BN_QK_K); b++) {
        w[b].d = (uint16_t)(0x2400 + (b % 29) * 13);
        w[b].dmin = (uint16_t)(0x2000 + (b % 17) * 19);
        for (int i = 0; i < 12; i++) {
            state = state * 1664525u + 1013904223u;
            w[b].scales[i] = (uint8_t)(state >> 24);
        }
        for (int i = 0; i < 32; i++) {
            state = state * 1664525u + 1013904223u;
            w[b].qh[i] = (uint8_t)(state >> 24);
        }
        for (int i = 0; i < 128; i++) {
            state = state * 1664525u + 1013904223u;
            w[b].qs[i] = (uint8_t)(state >> 24);
        }
    }
    for (int i = 0; i < cols; i++) {
        state = state * 1664525u + 1013904223u;
        x[i] = (float)((int)((state >> 8) % 20001) - 10000) / 8192.0f;
    }
}

static void run_q5_decode_reference_case(BnGPUBackend *gpu) {
    /* Independent ggml_mul_mat CUDA graphs, llama.cpp commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e, SM120. Hash every output
     * word to detect both per-block FMA and inter-warp reduction changes. */
    const struct { int cols; unsigned seed; uint64_t hash; } cases[] = {
        {256, 0, UINT64_C(0xdd1eab2a9f2554d4)},
        {256, 23, UINT64_C(0x22f20fe0c744395e)},
        {512, 0, UINT64_C(0x67bf0f52103fe609)},
        {512, 23, UINT64_C(0x852288e9534554d5)},
        {768, 0, UINT64_C(0x85cb1d0cd842e160)},
        {768, 23, UINT64_C(0x58e6827d2723a022)},
        {1024, 0, UINT64_C(0x316ba1d46bed36b1)},
        {1024, 23, UINT64_C(0xfa3b7e7230143df4)},
        {1536, 0, UINT64_C(0x9906b6893835d4d1)},
        {1536, 23, UINT64_C(0x9a0c56bd6e0e0494)},
        {2048, 0, UINT64_C(0x92d1eb924deb18f1)},
        {2048, 23, UINT64_C(0x5331fe93b5fe8c7c)},
        {3072, 0, UINT64_C(0x306ee211eb6591b8)},
        {3072, 23, UINT64_C(0xaaf0d57a36e4f44a)},
        {4096, 0, UINT64_C(0xf0e51f2eae7d2690)},
        {4096, 23, UINT64_C(0x5fe8cd58afc37495)},
        {5120, 0, UINT64_C(0x143d776bc2113c42)},
        {5120, 23, UINT64_C(0x4e6c71f48dd13158)},
        {5376, 0, UINT64_C(0x9c114e79b73ccde7)},
        {5376, 23, UINT64_C(0x0bf660e238afb530)},
        {6144, 0, UINT64_C(0x936be6b127771765)},
        {6144, 23, UINT64_C(0xa58d6ad65b595bc3)},
        {8192, 0, UINT64_C(0x4a1230e169a628cd)},
        {8192, 23, UINT64_C(0xdf1304312214717b)},
        {8448, 0, UINT64_C(0x791e4c0b66e43b22)},
        {8448, 23, UINT64_C(0xa9a1d88c3d45cbfd)},
        {8960, 0, UINT64_C(0xcfd2e972885ff3db)},
        {8960, 23, UINT64_C(0x65ffa4d591c0e6f0)},
        {9728, 0, UINT64_C(0x9204bc103dc165bc)},
        {9728, 23, UINT64_C(0x1a0c7afd1d415ab6)},
        {10240, 0, UINT64_C(0x22adb9e55b06c5be)},
        {10240, 23, UINT64_C(0x388d05f5e33c8374)},
        {12288, 0, UINT64_C(0xe3810a71fa47e5c8)},
        {12288, 23, UINT64_C(0x978c46d372a727c3)},
        {14336, 0, UINT64_C(0x26d4af716d2035e1)},
        {14336, 23, UINT64_C(0x97204a2c580fcfab)},
        {16384, 0, UINT64_C(0x8a86fff1edbce49e)},
        {16384, 23, UINT64_C(0x708da1a476e0fd61)},
        {17408, 0, UINT64_C(0xad0a29f950dc0f1b)},
        {17408, 23, UINT64_C(0xf3b24faf2b7b4849)},
        {21504, 0, UINT64_C(0x08117181ccf0ae54)},
        {21504, 23, UINT64_C(0x8221e1f039432473)},
        {28672, 0, UINT64_C(0x3dcfb7a894579867)},
        {28672, 23, UINT64_C(0x6bac8d2feb1017cb)},
        {32768, 0, UINT64_C(0x56ca783b303987cc)},
        {32768, 23, UINT64_C(0xd606b3bc2e3d975e)},
        {65536, 0, UINT64_C(0x2b93eccbca141ffd)},
        {65536, 23, UINT64_C(0x4c7b70c2893d15bd)},
        {2560, 0, UINT64_C(0x36a99b64f4d38b73)},
        {2560, 23, UINT64_C(0xacd114ae7aad9429)},
    };
    const int rows = 37;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int cols = cases[c].cols;
        size_t bytes = (size_t)rows * (cols / BN_QK_K) * sizeof(BnBlockQ5K);
        BnBlockQ5K *weights = (BnBlockQ5K *)malloc(bytes);
        float *input = (float *)malloc((size_t)cols * sizeof(float));
        float output[39];
        assert(weights && input);
        fill_q5_decode_reference(weights, input, rows, cols, cases[c].seed);
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = cols;
        plan.vocab_size = rows;
        plan.n_layers = plan.seq_len = plan.n_heads = plan.head_size = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        void *buffer = gpu->buffer_create(gpu->ctx, weights, bytes,
                                         BN_GGUF_TENSOR_Q5_K, rows, cols);
        assert(buffer);
        for (unsigned flags = 0; flags <= 1; flags++) {
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X,
                                         input, (size_t)cols * 4, 0) == 0);
            BnGPUOp op = {0};
            op.op_code = BN_GPU_CODE_MATVEC;
            op.buf_in = BN_GPU_VALUE_X;
            op.buf_out = BN_GPU_VALUE_LOGITS;
            op.W_buf = buffer;
            op.type = BN_GGUF_TENSOR_Q5_K;
            op.rows = rows;
            op.cols = cols;
            op.flags = flags;
            output[0] = output[rows + 1] = -12345.0f;
            assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_LOGITS,
                                output + 1, rows) == 0);
            uint64_t hash = cuda_test_float_bits_hash(output + 1, rows);
            if (hash != cases[c].hash)
                fprintf(stderr, "Q5 decode cols=%d seed=%u flags=%u hash=%016llx expected=%016llx\n",
                        cols, cases[c].seed, flags, (unsigned long long)hash,
                        (unsigned long long)cases[c].hash);
            assert(hash == cases[c].hash);
            assert(output[0] == -12345.0f && output[rows + 1] == -12345.0f);
        }
        /* The host matvec and one-token matmul APIs share decode arithmetic. */
        for (int matmul = 0; matmul <= 1; matmul++) {
            output[0] = output[rows + 1] = -12345.0f;
            int rc = matmul
                ? gpu->matmul(gpu->ctx, output + 1, buffer, input,
                              rows, cols, 1, BN_GGUF_TENSOR_Q5_K)
                : gpu->matvec(gpu->ctx, output + 1, buffer, input,
                              rows, cols, BN_GGUF_TENSOR_Q5_K);
            assert(rc == 0);
            assert(cuda_test_float_bits_hash(output + 1, rows) == cases[c].hash);
            assert(output[0] == -12345.0f && output[rows + 1] == -12345.0f);
        }
        run_kquant_split_decode_reference(gpu, buffer, input, cols,
                                          BN_GGUF_TENSOR_Q5_K, cases[c].hash);
        if (cols == 8448 && cases[c].seed == 0) {
            /* Preserve the explicit one-warp diagnostic override. This hash
             * is from the pre-change backend, not the parity reference. */
            BnBackendRuntimePolicy policy = {0};
            assert(bn_backend_runtime_policy_clone(&policy, &gpu->runtime_policy) == 0);
            assert(bn_backend_runtime_policy_set(&policy,
                "BN_CUDA_DISABLE_DEINTERLEAVED_KQUANT_4WARP", "1", 1) == 0);
            BnGPUBackend *disabled = bn_gpu_cuda_create_with_policy(&policy);
            bn_backend_runtime_policy_free(&policy);
            assert(disabled);
            void *disabled_buffer = disabled->buffer_create(disabled->ctx,
                weights, bytes, BN_GGUF_TENSOR_Q5_K, rows, cols);
            assert(disabled_buffer);
            for (int matmul = 0; matmul <= 1; matmul++) {
                output[0] = output[rows + 1] = -12345.0f;
                int rc = matmul
                    ? disabled->matmul(disabled->ctx, output + 1, disabled_buffer,
                                       input, rows, cols, 1, BN_GGUF_TENSOR_Q5_K)
                    : disabled->matvec(disabled->ctx, output + 1, disabled_buffer,
                                       input, rows, cols, BN_GGUF_TENSOR_Q5_K);
                assert(rc == 0);
                assert(cuda_test_float_bits_hash(output + 1, rows) ==
                       UINT64_C(0x91e4a62098631fff));
                assert(output[0] == -12345.0f && output[rows + 1] == -12345.0f);
            }
            disabled->buffer_destroy(disabled->ctx, disabled_buffer);
            bn_gpu_cuda_destroy(disabled);
        }
        gpu->buffer_destroy(gpu->ctx, buffer);
        free(input);
        free(weights);
    }
    printf("CUDA Q5 decode reference PASSED\n");
}

static void run_qk_rope_reference_contract(BnGPUBackend *gpu,
                                           int separate_rope_norm) {
    /* Independent CUDA RMSNorm/multiply/NeoX-RoPE graphs from llama.cpp
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e. Include nonzero positions,
     * partial rotary dimensions and both normalization reduction widths. */
    static const int head_sizes[] = {128, 256, 512, 1024};
    static const int rotary_dims[] = {128, 64, 128, 128};
    static const int tokens[] = {2, 9, 16};
    static const uint64_t hashes[] = {
        UINT64_C(0x8b37a656c5f8780d),
        UINT64_C(0x9ee332e74643a222),
        UINT64_C(0x1a21343f9e66c67c),
        UINT64_C(0x90ea5c5f9b8865dd),
        UINT64_C(0x1ff3677cbf881ba9),
        UINT64_C(0xd09627bc37952a6a),
        UINT64_C(0x997e6b2704777c61),
        UINT64_C(0xfd2c34e55cf064f0),
        UINT64_C(0x88940f5d429d0536),
        UINT64_C(0x12e856759fdcf047),
        UINT64_C(0x1b3eb1f0e693ba34),
        UINT64_C(0xf3c4ce80646d9bb1),
        UINT64_C(0xca4faab824c8ca1b),
        UINT64_C(0x6f1629500d004c82),
        UINT64_C(0xab628283b7d50f00),
        UINT64_C(0x5af8315bbd272465),
        UINT64_C(0x28dc3a73f0975be3),
        UINT64_C(0x34e62564cf341642),
        UINT64_C(0xd90a70fd060632d1),
        UINT64_C(0xbc26dc8664ed8cf4),
        UINT64_C(0x7248af23aaf2e674),
        UINT64_C(0xe5b4b3400309a0c5),
        UINT64_C(0xcae67ab9513a9f39),
        UINT64_C(0x74652fafb0d06006),
        UINT64_C(0xab16300aa0088d4d),
        UINT64_C(0x13268b7ac8ecccfa),
        UINT64_C(0x4cfa93dc57696080),
        UINT64_C(0x3709bf26054285bc),
        UINT64_C(0x2d6238599c944ef6),
        UINT64_C(0x3ecd7f527a344559),
        UINT64_C(0x1f437af6ad064ee1),
        UINT64_C(0xecb353f31d036ca7),
        UINT64_C(0xa8b3d8282e1c757e),
        UINT64_C(0xae967277684cc713),
        UINT64_C(0x7e3f64c74b9504ba),
        UINT64_C(0xf149a6080b59b3d5),
        UINT64_C(0xf0baa534e3aaf7c0),
        UINT64_C(0x47235a67fcb81f3c),
        UINT64_C(0xa3406426f3083815),
        UINT64_C(0xd29b33ae1aa9d22b),
        UINT64_C(0xed253dddff8d2126),
        UINT64_C(0x8d52cbab5188f5ca),
        UINT64_C(0x73fe24cd1d9f0194),
        UINT64_C(0x5b232e43bf1c7d72),
        UINT64_C(0x36e6e11dba993bc1),
        UINT64_C(0x8d5c65660a0c004c),
        UINT64_C(0x668218870a2550c3),
        UINT64_C(0xfccbaf7ef420c59b),
    };
    /* Separate normalization and RoPE graphs: no reference fusion crosses
     * the boundary. Keep the original fused-graph hashes above as well. */
    static const uint64_t unfused_hashes[] = {
        UINT64_C(0x7778d97befb6bc70),
        UINT64_C(0x6c6102e79bc9e731),
        UINT64_C(0x7def90b9bde59f33),
        UINT64_C(0xd72d8aae2420cc6f),
        UINT64_C(0xf16588c348fda38b),
        UINT64_C(0x45bcecb0f45ebeef),
        UINT64_C(0xfe786509d28b20d7),
        UINT64_C(0xeb3d0820637d2567),
        UINT64_C(0x496f9b6508c643d1),
        UINT64_C(0x6313ed77367f221f),
        UINT64_C(0x5b066f1fd72116bd),
        UINT64_C(0x4497a4b43be310bb),
        UINT64_C(0xdb8b49db269a1307),
        UINT64_C(0x1c3451b1ee309c66),
        UINT64_C(0x3916b23f2ad929c2),
        UINT64_C(0x9cdd629a4e019416),
        UINT64_C(0xd0655b1e09f24d5a),
        UINT64_C(0x266b99dc21aeef68),
        UINT64_C(0xdfb4290b7b9d4b31),
        UINT64_C(0xe80d8d5d0f82049e),
        UINT64_C(0x8470975f11781098),
        UINT64_C(0xcd13d1003682cd97),
        UINT64_C(0x0edb4da7e4ece5b9),
        UINT64_C(0x7e28038e3d7ddf3c),
        UINT64_C(0x3d6157271e7b2a89),
        UINT64_C(0xc89e49417d4e7495),
        UINT64_C(0x3a293ee2259fd447),
        UINT64_C(0x5f91d3acedb6ff14),
        UINT64_C(0x4d2e777f2c8572d1),
        UINT64_C(0x50f1e0d57171a4ac),
        UINT64_C(0x969d560d5f5ae323),
        UINT64_C(0x6d725b1c7b601331),
        UINT64_C(0x94739f57bfde2a2a),
        UINT64_C(0xfc36dc4e8cc90de0),
        UINT64_C(0x79f2ec1619062dfc),
        UINT64_C(0x85f159391969c79d),
        UINT64_C(0x06e23bbb40a3160e),
        UINT64_C(0xd244df73475251bc),
        UINT64_C(0x2e3ccfcd27c97203),
        UINT64_C(0x70c32d6e0a99095a),
        UINT64_C(0x7c4eeb22183e6835),
        UINT64_C(0x1c7688b43da90cfc),
        UINT64_C(0x25862f0b9ccc1094),
        UINT64_C(0xc2b2c86af15845ed),
        UINT64_C(0xaf2ad257ca962b97),
        UINT64_C(0x1897b3407348f110),
        UINT64_C(0x85db4de299bd9d26),
        UINT64_C(0x98853e9e750785c1),
    };
    BnBackendRuntimePolicy policy = {0};
    assert(bn_backend_runtime_policy_clone(&policy, &gpu->runtime_policy) == 0);
    assert(bn_backend_runtime_policy_set(&policy,
        "BN_CUDA_DISABLE_QK_NORM_ROPE_FUSE", "1", 1) == 0);
    BnGPUBackend *disabled = bn_gpu_cuda_create_with_policy(&policy);
    bn_backend_runtime_policy_free(&policy);
    assert(disabled);
    const uint64_t *adjacent_hashes =
        separate_rope_norm ? unfused_hashes : hashes;
    int case_index = 0;
    for (int hi = 0; hi < 4; hi++) {
        int hs = head_sizes[hi], rd = rotary_dims[hi];
        float frequencies[64] = {0};
        BnGPURopeFrequencyPlan recipe = {
            0, rd / 2, rd, 10000000.0f, NULL, BN_GPU_ROPE_FACTOR_NONE
        };
        BnGPUActivationPlan plan = {0};
        plan.separate_rope_norm = separate_rope_norm;
        plan.dim = plan.hb_elements = plan.xb2_elements = 16 * hs;
        plan.n_layers = 1; plan.n_heads = 16; plan.seq_len = 2048;
        plan.attention_layer_count = 1;
        plan.head_size = hs; plan.kv_dim = 2 * hs; plan.vocab_size = 32;
        plan.rope_frequencies = frequencies; plan.rope_frequency_count = rd / 2;
        plan.rope_frequency_plans = &recipe; plan.rope_frequency_plan_count = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        assert(disabled->init_activations(disabled->ctx, &plan) == 0);
        for (int ti = 0; ti < 3; ti++) {
            int nt = tokens[ti];
            for (int per = 0; per < 2; per++) {
                int pos = per ? 127 : 0;
                float *q_input = NULL, *q_reference = NULL;
                void *q_weight = NULL;
                for (int kind = 0; kind < 2; kind++, case_index++) {
                    int nh = kind == 0 ? 16 : 2, n = nh * hs;
                    size_t count = (size_t)nt * n;
                    int weight_count = (per ? nh : 1) * hs;
                    float *input = (float *)malloc(count * sizeof(float));
                    float *weight = (float *)malloc((size_t)weight_count * sizeof(float));
                    float *output = (float *)malloc((count + 2) * sizeof(float));
                    assert(input && weight && output);
                    for (size_t i = 0; i < count; i++)
                        input[i] = kind == 0 ? sinf((float)(i * 17 + 3)) * 0.3f
                                             : cosf((float)(i * 11 + 7)) * 0.7f;
                    for (int i = 0; i < weight_count; i++)
                        weight[i] = kind == 0 ? 0.5f + (float)(i % 19) / 16.0f
                                              : 0.4f + (float)(i % 13) / 8.0f;
                    void *wb = gpu->buffer_create(gpu->ctx, weight,
                        (size_t)weight_count * sizeof(float), BN_GGUF_TENSOR_F32,
                        per ? nh : 1, hs);
                    assert(wb);
                    BnGPUOp ops[2] = {0};
                    float eps = 1e-6f;
                    ops[0].op_code = BN_GPU_CODE_PER_HEAD_RMSNORM;
                    ops[0].buf_in = BN_GPU_VALUE_Q; ops[0].rows = nh;
                    ops[0].W_buf = wb; ops[0].p[0] = hs; ops[0].p[2] = per;
                    memcpy(&ops[0].p[1], &eps, sizeof(eps));
                    ops[1].op_code = BN_GPU_CODE_ROPE;
                    ops[1].buf_in = BN_GPU_VALUE_Q; ops[1].buf_aux = -1;
                    ops[1].p[0] = nh; ops[1].p[1] = hs; ops[1].p[3] = rd;
                    output[0] = output[count + 1] = 12345.0f;
                    float *adjacent = malloc(count * sizeof(float));
                    assert(adjacent);
                    for (int t = 0; t < nt; t++) {
                        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_Q,
                            input + (size_t)t * n, (size_t)n * sizeof(float), 0) == 0);
                        ops[1].p[2] = pos + t;
                        assert(gpu->execute(gpu->ctx, ops, 2, BN_GPU_VALUE_Q,
                            adjacent + (size_t)t * n, n) == 0);
                        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_Q,
                            input + (size_t)t * n, (size_t)n * sizeof(float), 0) == 0);
                        assert(gpu->execute(gpu->ctx, ops, 1, BN_GPU_VALUE_Q,
                            output + 1 + (size_t)t * n, n) == 0);
                        assert(gpu->execute(gpu->ctx, ops + 1, 1, BN_GPU_VALUE_Q,
                            output + 1 + (size_t)t * n, n) == 0);
                    }
                    uint64_t adjacent_hash = cuda_test_float_bits_hash(adjacent, count);
                    if (adjacent_hash != adjacent_hashes[case_index])
                        fprintf(stderr,
                            "qk rope separate=%d case=%d got=%016llx split=%016llx expected=%016llx\n",
                            separate_rope_norm, case_index,
                            (unsigned long long)adjacent_hash,
                            (unsigned long long)cuda_test_float_bits_hash(output + 1, count),
                            (unsigned long long)adjacent_hashes[case_index]);
                    assert(adjacent_hash == adjacent_hashes[case_index]);
                    /* Materializing gated Q must preserve the declared
                     * arithmetic contract; layout alone does not select it. */
                    if (kind == 0) {
                        float *gated = malloc((size_t)2 * n * sizeof(float));
                        assert(gated);
                        BnGPUOp gated_ops[3] = {{0}, ops[0], ops[1]};
                        gated_ops[0].op_code = BN_GPU_CODE_DEINTERLEAVE_Q;
                        gated_ops[0].buf_in = BN_GPU_VALUE_QKV;
                        gated_ops[0].buf_out = BN_GPU_VALUE_Q;
                        gated_ops[0].p[0] = (uint32_t)n;
                        gated_ops[0].p[1] = (uint32_t)hs;
                        for (int t = 0; t < nt; t++) {
                            for (int h = 0; h < nh; h++) {
                                memcpy(gated + (size_t)h * 2 * hs,
                                       input + (size_t)t * n + h * hs,
                                       (size_t)hs * sizeof(float));
                                for (int j = 0; j < hs; j++)
                                    gated[(size_t)h * 2 * hs + hs + j] = 0.25f;
                            }
                            gated_ops[2].p[2] = pos + t;
                            assert(gpu->write_activation(gpu->ctx,
                                BN_GPU_VALUE_QKV, gated,
                                (size_t)2 * n * sizeof(float), 0) == 0);
                            assert(gpu->execute(gpu->ctx, gated_ops, 3,
                                BN_GPU_VALUE_Q, adjacent + (size_t)t * n,
                                n) == 0);
                        }
                        assert(cuda_test_float_bits_hash(adjacent, count) ==
                               adjacent_hashes[case_index]);
                        free(gated);
                    }

                    void *disabled_wb = disabled->buffer_create(disabled->ctx,
                        weight, (size_t)weight_count * sizeof(float),
                        BN_GGUF_TENSOR_F32, per ? nh : 1, hs);
                    assert(disabled_wb);
                    BnGPUOp disabled_ops[2] = {ops[0], ops[1]};
                    disabled_ops[0].W_buf = disabled_wb;
                    for (int t = 0; t < nt; t++) {
                        disabled_ops[1].p[2] = pos + t;
                        assert(disabled->write_activation(disabled->ctx,
                            BN_GPU_VALUE_Q, input + (size_t)t * n,
                            (size_t)n * sizeof(float), 0) == 0);
                        assert(disabled->execute(disabled->ctx, disabled_ops, 2,
                            BN_GPU_VALUE_Q, adjacent + (size_t)t * n, n) == 0);
                    }
                    assert(cuda_test_float_bits_hash(adjacent, count) ==
                           unfused_hashes[case_index]);
                    disabled->buffer_destroy(disabled->ctx, disabled_wb);
                    free(adjacent);
                    uint64_t hash = cuda_test_float_bits_hash(output + 1, count);
                    if (hash != unfused_hashes[case_index])
                        fprintf(stderr, "QK RoPE case=%d hash=%llx expected=%llx\n",
                            case_index, (unsigned long long)hash,
                            (unsigned long long)unfused_hashes[case_index]);
                    assert(hash == unfused_hashes[case_index]);
                    assert(output[0] == 12345.0f && output[count + 1] == 12345.0f);
                    free(weight);
                    if (kind == 0) {
                        q_input = input; q_reference = output; q_weight = wb;
                    } else {
                        /* Joint Q/K preparation obeys the same declared
                         * contraction for both Q and K. */
                        int qn = 16 * hs;
                        float *fused = (float *)malloc((size_t)nt * (size_t)(qn + n) * sizeof(float));
                        assert(fused);
                        BnGPUOp fused_ops[3] = {ops[0], ops[0], ops[1]};
                        fused_ops[0].W_buf = q_weight; fused_ops[0].rows = 16;
                        fused_ops[1].buf_in = BN_GPU_VALUE_KEY_CACHE;
                        fused_ops[2].op_code = BN_GPU_CODE_ROPE_QK;
                        fused_ops[2].buf_aux = BN_GPU_VALUE_KEY_CACHE;
                        fused_ops[2].p[0] = 16; fused_ops[2].p[4] = nh;
                        for (int t = 0; t < nt; t++) {
                            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_Q,
                                q_input + (size_t)t * qn, (size_t)qn * sizeof(float), 0) == 0);
                            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE,
                                input + (size_t)t * n, (size_t)n * sizeof(float), 0) == 0);
                            fused_ops[2].p[2] = pos + t;
                            assert(gpu->execute(gpu->ctx, fused_ops, 3, BN_GPU_VALUE_Q,
                                fused + (size_t)t * qn, qn) == 0);
                            assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE,
                                fused + (size_t)nt * qn + (size_t)t * n,
                                (size_t)n * sizeof(float), 0) == 0);
                        }
                        assert(cuda_test_float_bits_hash(fused, (size_t)nt * qn) ==
                            adjacent_hashes[case_index - 1]);
                        assert(cuda_test_float_bits_hash(fused + (size_t)nt * qn,
                            (size_t)nt * n) == adjacent_hashes[case_index]);
                        free(fused);
                        gpu->buffer_destroy(gpu->ctx, q_weight);
                        gpu->buffer_destroy(gpu->ctx, wb);
                        free(q_reference); free(q_input);
                        free(output); free(input);
                    }
                }
            }
        }
    }
    assert(case_index == 48);
    bn_gpu_cuda_destroy(disabled);

}

static void run_qk_rope_reference_case(BnGPUBackend *gpu) {
    run_qk_rope_reference_contract(gpu, 0);
    run_qk_rope_reference_contract(gpu, 1);
    printf("CUDA QK RoPE reference PASSED\n");
}

static void run_standalone_rope_reference_case(BnGPUBackend *gpu) {
    /* Independent ggml CUDA standalone NeoX RoPE, reference commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e. Normalization is deliberately
     * absent: its fused reference kernel has a different contraction order. */
    const struct { int head, rotary, pos; uint64_t q, k; } cases[] = {
        {128, 32, 12, UINT64_C(0x329c816dc6c58d11), UINT64_C(0x1e71d9eb28ff4e14)},
        {128, 32, 127, UINT64_C(0x47825fb7bca5a896), UINT64_C(0x044f0dc5c77e5255)},
        {256, 64, 12, UINT64_C(0x1eb9908ae8a14b1a), UINT64_C(0xd2b79434fe6ae255)},
        {256, 64, 127, UINT64_C(0x72b82fc2ebf6f21f), UINT64_C(0x7e6eb576c538195c)},
        {512, 128, 12, UINT64_C(0xe094e27c97ce99e1), UINT64_C(0xe57b23647c296e19)},
        {512, 128, 127, UINT64_C(0x345040159c3987e0), UINT64_C(0x642811dc36068d80)},
        {1024, 128, 12, UINT64_C(0xb608f1972aa13151), UINT64_C(0xb71ff01f022c68ea)},
        {1024, 128, 127, UINT64_C(0xc17417f0c30816a8), UINT64_C(0xa9edda0706e6c09d)},
    };
    const int nt = 3, nh = 16, nk = 2, key_offset = 19;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int hs = cases[c].head, rd = cases[c].rotary;
        int qdim = nh * hs, kdim = nk * hs;
        size_t nq = (size_t)nt * qdim, nkey = (size_t)nt * kdim;
        float *q = malloc(nq * sizeof(float)), *k = malloc(nkey * sizeof(float));
        float *qo = malloc((nq + 2) * sizeof(float));
        float *ko = malloc((nkey + 2) * sizeof(float));
        assert(q && k && qo && ko);
        for (size_t i = 0; i < nq; i++) q[i] = sinf((float)(i * 17 + 3)) * 0.3f;
        for (size_t i = 0; i < nkey; i++) k[i] = cosf((float)(i * 11 + 7)) * 0.7f;
        float frequencies[64] = {0};
        BnGPURopeFrequencyPlan recipe = {
            0, rd / 2, rd, 10000000.0f, NULL, BN_GPU_ROPE_FACTOR_NONE
        };
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = qdim;
        plan.n_layers = plan.attention_layer_count = 1;
        plan.n_heads = nh; plan.seq_len = 2048; plan.head_size = hs;
        plan.kv_dim = kdim; plan.vocab_size = 32;
        plan.rope_frequencies = frequencies; plan.rope_frequency_count = rd / 2;
        plan.rope_frequency_plans = &recipe; plan.rope_frequency_plan_count = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        for (int paired = 0; paired < 2; paired++) {
            BnGPUOp op = {0};
            op.op_code = paired ? BN_GPU_CODE_ROPE_QK : BN_GPU_CODE_ROPE;
            op.buf_in = BN_GPU_VALUE_Q;
            op.buf_aux = paired ? BN_GPU_VALUE_KEY_CACHE : -1;
            op.p[0] = nh; op.p[1] = hs; op.p[3] = rd;
            op.p[4] = paired ? nk : 0; op.p[5] = paired ? key_offset : 0;
            qo[0] = qo[nq + 1] = ko[0] = ko[nkey + 1] = 12345.0f;
            for (int t = 0; t < nt; t++) {
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_Q,
                    q + (size_t)t * qdim, (size_t)qdim * sizeof(float), 0) == 0);
                if (paired)
                    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE,
                        k + (size_t)t * kdim, (size_t)kdim * sizeof(float),
                        key_offset * sizeof(float)) == 0);
                op.p[2] = cases[c].pos + t;
                assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_Q,
                    qo + 1 + (size_t)t * qdim, qdim) == 0);
                if (paired)
                    assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE,
                        ko + 1 + (size_t)t * kdim, (size_t)kdim * sizeof(float),
                        key_offset * sizeof(float)) == 0);
            }
            assert(cuda_test_float_bits_hash(qo + 1, nq) == cases[c].q);
            if (paired) assert(cuda_test_float_bits_hash(ko + 1, nkey) == cases[c].k);
            assert(qo[0] == 12345.0f && qo[nq + 1] == 12345.0f);
            assert(ko[0] == 12345.0f && ko[nkey + 1] == 12345.0f);
        }
        gpu->free_activations(gpu->ctx);
        free(q); free(k); free(qo); free(ko);
    }
    printf("CUDA standalone RoPE reference PASSED\n");
}

static void run_raw_attention_mma_reference_case(BnGPUBackend *shared_gpu) {
    /* Independent ggml CUDA graph: input RMSNorm, packed Q/K and V
     * projections, head RMSNorm/RoPE, F16-cache flash attention, optional
     * sigmoid gate, output projection and residual. Reference commit:
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e. */
    const struct { int tokens, gated; uint64_t output, key, value; } cases[] = {
        {2, 0, UINT64_C(0x2bfe2c39de6c7e8a), UINT64_C(0x170a29880fced325), UINT64_C(0xbf113ebf76c03325)},
        {2, 1, UINT64_C(0xedd927582663aa02), UINT64_C(0x516ebb5ffcf23325), UINT64_C(0xbf113ebf76c03325)},
        {7, 0, UINT64_C(0x8368c4bf680b668c), UINT64_C(0xc907e74234ecbb25), UINT64_C(0x2adbf7a772113b25)},
        {7, 1, UINT64_C(0x6b092d7db2b8c29b), UINT64_C(0x4902d7de11c41b25), UINT64_C(0x2adbf7a772113b25)},
        {9, 0, UINT64_C(0x5e8197cf35ca38c1), UINT64_C(0x787de73d79c5ab25), UINT64_C(0x633ce06d8cb38b25)},
        {9, 1, UINT64_C(0xf606cd3d1dc24666), UINT64_C(0x14f8cf99e613eb25), UINT64_C(0x633ce06d8cb38b25)},
        {16, 0, UINT64_C(0xcdebd50a8f65d251), UINT64_C(0xec69ae1829ee6325), UINT64_C(0x1f460a3431774325)},
        {16, 1, UINT64_C(0x4072ee4bc2992e53), UINT64_C(0xee5a705596b4e325), UINT64_C(0x1f460a3431774325)},
        {17, 0, UINT64_C(0xb0cf0abe7ead1c67), UINT64_C(0x9325de9e71216b25), UINT64_C(0x1a93bfd79673cb25)},
        {17, 1, UINT64_C(0xaa687a6979fbf3d1), UINT64_C(0x812c5475b7120b25), UINT64_C(0x1a93bfd79673cb25)},
        {28, 0, UINT64_C(0x7ae94ec81aedbd42), UINT64_C(0x003a6d75f3d5c325), UINT64_C(0x00754940c04c8325)},
        {28, 1, UINT64_C(0x98b0dc3fef06a768), UINT64_C(0xba1060b123abc325), UINT64_C(0x00754940c04c8325)},
        {29, 0, UINT64_C(0x50dc24e4d057aa47), UINT64_C(0x6eb40ce4c7882b25), UINT64_C(0xd4bfaf9fb4e12b25)},
        {29, 1, UINT64_C(0x14764e84408fbac4), UINT64_C(0x95d53b96f329cb25), UINT64_C(0xd4bfaf9fb4e12b25)},
        {32, 0, UINT64_C(0x132aea02eb6cc61f), UINT64_C(0xf22214ece83f0325), UINT64_C(0xf337cb16de378325)},
        {32, 1, UINT64_C(0xe8ec7893de14ddd8), UINT64_C(0xb8b4a07f85ec8325), UINT64_C(0xf337cb16de378325)},
    };
    BnBackendRuntimePolicy policy = {0};
    assert(bn_backend_runtime_policy_clone(&policy, &shared_gpu->runtime_policy) == 0);
    if (bn_gpu_backend_has_cap(shared_gpu, BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL)) {
        /* The declared prefill contract must accept short prompts by default. */
        bn_backend_runtime_policy_unset(&policy, "BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    } else {
        assert(bn_backend_runtime_policy_set(&policy,
            "BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "1", 1) == 0);
    }
    BnGPUBackend *gpu = bn_gpu_cuda_create_with_policy(&policy);
    bn_backend_runtime_policy_free(&policy);
    assert(gpu);
    const int dim = 32, hs = 256, nh = 16, nk = 2, qd = nh * hs, kd = nk * hs;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int nt = cases[c].tokens, gated = cases[c].gated;
        int rows = qd * (gated ? 2 : 1) + kd;
        size_t nx = (size_t)nt * dim, nkv = (size_t)nt * kd;
        size_t nqkw = (size_t)dim * rows, nvw = (size_t)dim * kd;
        size_t now = (size_t)dim * qd;
        float *input = malloc(nx * sizeof(float));
        float *qkw = malloc(nqkw * sizeof(float));
        float *vw = malloc(nvw * sizeof(float));
        float *ow = malloc(now * sizeof(float));
        float *output = malloc((nx + 2) * sizeof(float));
        float *key = malloc((nkv + 2) * sizeof(float));
        float *value = malloc((nkv + 2) * sizeof(float));
        float norm[32], qnorm[256], knorm[256], table[128] = {0};
        assert(input && qkw && vw && ow && output && key && value);
        for (size_t i = 0; i < nx; i++)
            input[i] = (float)((int)((i * 13 + 5) % 97) - 48) / 64.0f;
        for (int i = 0; i < dim; i++) norm[i] = 0.5f + (float)(i % 19) / 16.0f;
        for (size_t i = 0; i < nqkw; i++)
            qkw[i] = (float)((int)((i * 17 + 3) % 101) - 50) / 512.0f;
        for (size_t i = 0; i < nvw; i++)
            vw[i] = (float)((int)((i * 11 + 7) % 103) - 51) / 512.0f;
        for (size_t i = 0; i < now; i++)
            ow[i] = (float)((int)((i * 7 + 5) % 107) - 53) / 4096.0f;
        for (int i = 0; i < hs; i++) {
            qnorm[i] = 0.5f + (float)(i % 19) / 16.0f;
            knorm[i] = 0.4f + (float)(i % 13) / 8.0f;
        }
        BnGPURopeFrequencyPlan recipe = {
            0, 32, 64, 1e7f, NULL, BN_GPU_ROPE_FACTOR_NONE
        };
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = dim; plan.xb2_elements = qd;
        plan.vocab_size = 32; plan.n_layers = 1; plan.n_heads = nh;
        plan.seq_len = 256; plan.head_size = hs; plan.kv_dim = kd; plan.kv_f16 = 1;
        plan.rope_frequencies = table; plan.rope_frequency_count = 128;
        plan.rope_frequency_plans = &recipe; plan.rope_frequency_plan_count = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        const int type = BN_GGUF_TENSOR_F32;
        void *buffers[] = {
            gpu->buffer_create(gpu->ctx, qkw, nqkw * sizeof(float), type, rows, dim),
            gpu->buffer_create(gpu->ctx, vw, nvw * sizeof(float), type, kd, dim),
            gpu->buffer_create(gpu->ctx, ow, now * sizeof(float), type, dim, qd),
            gpu->buffer_create(gpu->ctx, norm, sizeof(norm), type, 1, dim),
            gpu->buffer_create(gpu->ctx, qnorm, sizeof(qnorm), type, 1, hs),
            gpu->buffer_create(gpu->ctx, knorm, sizeof(knorm), type, 1, hs),
        };
        for (size_t b = 0; b < sizeof(buffers) / sizeof(buffers[0]); b++) assert(buffers[b]);
        output[0] = output[nx + 1] = key[0] = key[nkv + 1] =
            value[0] = value[nkv + 1] = 123.0f;
        for (int alias = 0; alias < 2; alias++) {
            float *out = alias ? input : output + 1;
            assert(gpu->prefill_qkv_attention_wo_norm_resid(
                gpu->ctx, out, buffers[0], buffers[1], buffers[2], buffers[3],
                buffers[4], buffers[5], input, key + 1, value + 1,
                nt, dim, nh, nk, hs, 8, kd, rows, type, kd, type,
                dim, qd, type, 0, 1e-6f, 0, 64, 0.0625f, 0) == 0);
            assert(cuda_test_float_bits_hash(out, nx) == cases[c].output);
            assert(cuda_test_float_bits_hash(key + 1, nkv) == cases[c].key);
            assert(cuda_test_float_bits_hash(value + 1, nkv) == cases[c].value);
            assert(output[0] == 123.0f && output[nx + 1] == 123.0f);
            assert(key[0] == 123.0f && key[nkv + 1] == 123.0f);
            assert(value[0] == 123.0f && value[nkv + 1] == 123.0f);
        }
        for (size_t b = 0; b < sizeof(buffers) / sizeof(buffers[0]); b++)
            gpu->buffer_destroy(gpu->ctx, buffers[b]);
        gpu->free_activations(gpu->ctx);
        free(value); free(key); free(output); free(ow); free(vw); free(qkw); free(input);
    }
    bn_gpu_cuda_destroy(gpu);
    printf("CUDA raw attention MMA reference PASSED\n");
}

static void run_model_attention_reference_case(BnGPUBackend *shared_gpu) {
    /* Independent llama.cpp CUDA reference, commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e: separate Q/K projections,
     * head normalization copied into standalone RoPE graph inputs, F16
     * attention, optional query gate, WO and residual. A zero MoE branch
     * isolates attention through the full model-layer API; nonzero MoE
     * arithmetic has independent regressions below. Preserve the existing
     * fused/packed attention oracle in run_raw_attention_mma_reference_case. */
    const struct { int dim, type, tokens, gated; uint64_t output, key, value; } cases[] = {
        {256, 0, 2, 0, UINT64_C(0x839b61597cbc706f), UINT64_C(0xa2e0ea0802041325), UINT64_C(0x106b3a5d5a5bb325)},
        {256, 0, 2, 1, UINT64_C(0x4bd77de6081eec13), UINT64_C(0x444bca9aa93b5325), UINT64_C(0x106b3a5d5a5bb325)},
        {256, 0, 7, 0, UINT64_C(0x5b08a420a2c71628), UINT64_C(0x89a5e7f9b74e5b25), UINT64_C(0x6d7272be9abbfb25)},
        {256, 0, 7, 1, UINT64_C(0x0a3aeed6beda14f3), UINT64_C(0x4d2f751fb11b1b25), UINT64_C(0x6d7272be9abbfb25)},
        {256, 0, 8, 0, UINT64_C(0x539ab253ab3c8462), UINT64_C(0x4017b6ff4f9e4325), UINT64_C(0xaf58bc0683a80325)},
        {256, 0, 8, 1, UINT64_C(0x2dd58c45a726c72f), UINT64_C(0x0c15638f49a14325), UINT64_C(0xaf58bc0683a80325)},
        {256, 0, 9, 0, UINT64_C(0xd45511207afbd6aa), UINT64_C(0x7d8ee4ed4e73cb25), UINT64_C(0x101719db3cc9ab25)},
        {256, 0, 9, 1, UINT64_C(0xa8aed1ddaea74d2e), UINT64_C(0x41ffd5de51a76b25), UINT64_C(0x101719db3cc9ab25)},
        {256, 0, 17, 0, UINT64_C(0x2ce6f64a658880e2), UINT64_C(0xce350f64080e0b25), UINT64_C(0x6bc3867f06fb8b25)},
        {256, 0, 17, 1, UINT64_C(0xf831731c8ad8fa4e), UINT64_C(0xf3c828e7a6fc8b25), UINT64_C(0x6bc3867f06fb8b25)},
        {256, 0, 29, 0, UINT64_C(0xffdf3afc38e82de2), UINT64_C(0xb4ad39e83ad4eb25), UINT64_C(0xc61a905fd44cab25)},
        {256, 0, 29, 1, UINT64_C(0xe220b1d5bf9ff9e2), UINT64_C(0x6c36dfb47e338b25), UINT64_C(0xc61a905fd44cab25)},
        {256, 0, 32, 0, UINT64_C(0xad9a8731287f651a), UINT64_C(0x27740e922b4f0325), UINT64_C(0xbf9fd0cfcd470325)},
        {256, 0, 32, 1, UINT64_C(0x85763770d5244d06), UINT64_C(0xb7c9ab9193b5c325), UINT64_C(0xbf9fd0cfcd470325)},
        {256, 8, 2, 0, UINT64_C(0x720bf1b4789aa4df), UINT64_C(0xb5c8eaaeaa999325), UINT64_C(0x106b3a5d5a5bb325)},
        {256, 8, 2, 1, UINT64_C(0x9905e3996a31463d), UINT64_C(0xd54709bc5c60f325), UINT64_C(0x106b3a5d5a5bb325)},
        {256, 8, 7, 0, UINT64_C(0x91d6a6c3f09b54a0), UINT64_C(0x18182bbbab3d5b25), UINT64_C(0x6d7272be9abbfb25)},
        {256, 8, 7, 1, UINT64_C(0xc5834c6d18b1d1b9), UINT64_C(0xb78925b8b3169b25), UINT64_C(0x6d7272be9abbfb25)},
        {256, 8, 8, 0, UINT64_C(0x9922e770ae29e848), UINT64_C(0x4c8e769bcb39e325), UINT64_C(0xaf58bc0683a80325)},
        {256, 8, 8, 1, UINT64_C(0x40dfecdb26c3dcec), UINT64_C(0x8e052926ab834325), UINT64_C(0xaf58bc0683a80325)},
        {256, 8, 9, 0, UINT64_C(0xf0c47113e356c667), UINT64_C(0x5e475572d830cb25), UINT64_C(0x101719db3cc9ab25)},
        {256, 8, 9, 1, UINT64_C(0x65995290ac742700), UINT64_C(0xcc153f82a1e60b25), UINT64_C(0x101719db3cc9ab25)},
        {256, 8, 17, 0, UINT64_C(0x89949632ca43a3be), UINT64_C(0x7bb91cf38ff08b25), UINT64_C(0x6bc3867f06fb8b25)},
        {256, 8, 17, 1, UINT64_C(0xbb61636f0e2aa371), UINT64_C(0x0f97a35bd4d0ab25), UINT64_C(0x6bc3867f06fb8b25)},
        {256, 8, 29, 0, UINT64_C(0x3890459e81c198d5), UINT64_C(0xe61f4b03b9650b25), UINT64_C(0xc61a905fd44cab25)},
        {256, 8, 29, 1, UINT64_C(0x693860490ddd2b45), UINT64_C(0x871cc905051d0b25), UINT64_C(0xc61a905fd44cab25)},
        {256, 8, 32, 0, UINT64_C(0xaf252206e23b1108), UINT64_C(0x4729c50dd20be325), UINT64_C(0xbf9fd0cfcd470325)},
        {256, 8, 32, 1, UINT64_C(0x597dc2cec1749475), UINT64_C(0xfd1753ae7d486325), UINT64_C(0xbf9fd0cfcd470325)},
        {2048, 0, 2, 0, UINT64_C(0x88618f0bd9697a2c), UINT64_C(0xee276153f42ab325), UINT64_C(0x4d4f5c2cea4db325)},
        {2048, 0, 2, 1, UINT64_C(0xe469574ecb7c8f8b), UINT64_C(0xaa7ec68d3b7cd325), UINT64_C(0x4d4f5c2cea4db325)},
        {2048, 0, 7, 0, UINT64_C(0x907bee7c59c3d108), UINT64_C(0xffa59bbd093b9b25), UINT64_C(0x5bc91672f1e3db25)},
        {2048, 0, 7, 1, UINT64_C(0x4998b30345f84dc8), UINT64_C(0x813f16d762001b25), UINT64_C(0x5bc91672f1e3db25)},
        {2048, 0, 8, 0, UINT64_C(0xf99ebf1400ce7621), UINT64_C(0x33af72e42047c325), UINT64_C(0xcd15f9ab49f94325)},
        {2048, 0, 8, 1, UINT64_C(0x2de11201347a70f7), UINT64_C(0xe28668ced0fbc325), UINT64_C(0xcd15f9ab49f94325)},
        {2048, 0, 9, 0, UINT64_C(0x03891c8669c3dbf6), UINT64_C(0xf331c1f5385ccb25), UINT64_C(0x709310b3b4760b25)},
        {2048, 0, 9, 1, UINT64_C(0xd1cc8ef73fda9b52), UINT64_C(0x76acc992a55c8b25), UINT64_C(0x709310b3b4760b25)},
        {2048, 0, 17, 0, UINT64_C(0x44bfb612569755df), UINT64_C(0xb0324f1c3aceeb25), UINT64_C(0xf87b2a9083bcab25)},
        {2048, 0, 17, 1, UINT64_C(0xa2d5789d882c6aae), UINT64_C(0x2b3003e6f6e1ab25), UINT64_C(0xf87b2a9083bcab25)},
        {2048, 0, 29, 0, UINT64_C(0x9e9a4ff677fb5ded), UINT64_C(0x46229a7d9a566b25), UINT64_C(0x8bba7e549bc42b25)},
        {2048, 0, 29, 1, UINT64_C(0xb745eda7e7f065b2), UINT64_C(0x5f5d1f55b2ec4b25), UINT64_C(0x8bba7e549bc42b25)},
        {2048, 0, 32, 0, UINT64_C(0x70f280bd125bad82), UINT64_C(0xbdc273a2d31ac325), UINT64_C(0xdf1124888b2a6325)},
        {2048, 0, 32, 1, UINT64_C(0x1458374930c21623), UINT64_C(0x623264d786fd8325), UINT64_C(0xdf1124888b2a6325)},
        {2048, 8, 2, 0, UINT64_C(0x01343b2aa830c5b8), UINT64_C(0x0ad182cf489e3325), UINT64_C(0x4d4f5c2cea4db325)},
        {2048, 8, 2, 1, UINT64_C(0x75939fe809df61bd), UINT64_C(0x3d72702aaf6c1325), UINT64_C(0x4d4f5c2cea4db325)},
        {2048, 8, 7, 0, UINT64_C(0x478835758df25ea7), UINT64_C(0x924a7ab933853b25), UINT64_C(0x5bc91672f1e3db25)},
        {2048, 8, 7, 1, UINT64_C(0x34e1e3bf0dada31a), UINT64_C(0xed3fc9fe51535b25), UINT64_C(0x5bc91672f1e3db25)},
        {2048, 8, 8, 0, UINT64_C(0xc8bd56923c65291b), UINT64_C(0xadf13417225e0325), UINT64_C(0xcd15f9ab49f94325)},
        {2048, 8, 8, 1, UINT64_C(0x5fe6243a4072fb0c), UINT64_C(0x7500745681e0a325), UINT64_C(0xcd15f9ab49f94325)},
        {2048, 8, 9, 0, UINT64_C(0xe42cc63ddf51ee02), UINT64_C(0x0fc09bfe582d2b25), UINT64_C(0x709310b3b4760b25)},
        {2048, 8, 9, 1, UINT64_C(0xa5455656e59cb293), UINT64_C(0xc2309c7c0b3a4b25), UINT64_C(0x709310b3b4760b25)},
        {2048, 8, 17, 0, UINT64_C(0x03be9e17bf032dc2), UINT64_C(0x3b8504163e534b25), UINT64_C(0xf87b2a9083bcab25)},
        {2048, 8, 17, 1, UINT64_C(0x2638f5f766a7e1bd), UINT64_C(0xdd575b14ab1a4b25), UINT64_C(0xf87b2a9083bcab25)},
        {2048, 8, 29, 0, UINT64_C(0x3921c9d94e798556), UINT64_C(0x79be24ec57434b25), UINT64_C(0x8bba7e549bc42b25)},
        {2048, 8, 29, 1, UINT64_C(0x48b063405867f9a8), UINT64_C(0xae2a29823540eb25), UINT64_C(0x8bba7e549bc42b25)},
        {2048, 8, 32, 0, UINT64_C(0xa8214ecf14b0131a), UINT64_C(0xb5bdeed6770ca325), UINT64_C(0xdf1124888b2a6325)},
        {2048, 8, 32, 1, UINT64_C(0x4a1466e70616e92d), UINT64_C(0x427df809554ca325), UINT64_C(0xdf1124888b2a6325)},
    };
    BnBackendRuntimePolicy policy = {0};
    assert(bn_backend_runtime_policy_clone(&policy, &shared_gpu->runtime_policy) == 0);
    assert(bn_backend_runtime_policy_set(&policy,
        "BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "1", 1) == 0);
    const char *flags[] = {
        "BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT",
        "BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT",
        "BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE",
        "BN_CUDA_DISABLE_MOE_NATIVE_QUANT_GATEUP_2ROW",
        "BN_CUDA_DISABLE_MOE_NATIVE_QUANT_DOWN_2ROW"
    };
    for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); i++)
        assert(bn_backend_runtime_policy_set(&policy, flags[i], "1", 1) == 0);
    BnGPUBackend *gpu = bn_gpu_cuda_create_with_policy(&policy);
    bn_backend_runtime_policy_free(&policy);
    assert(gpu);
    const int hs = 256, nh = 16, nk = 2, qd = nh * hs, kd = nk * hs;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int dim = cases[c].dim, qtype = cases[c].type;
        int nt = cases[c].tokens, gated = cases[c].gated;
        int rows = qd * (gated ? 2 : 1) + kd;
        size_t nx = (size_t)nt * dim, nkv = (size_t)nt * kd;
        size_t nqkw = (size_t)dim * rows, nvw = (size_t)dim * kd;
        size_t now = (size_t)dim * qd;
        float *input = malloc(nx * sizeof(float));
        float *qkw = malloc(nqkw * sizeof(float));
        float *vw = malloc(nvw * sizeof(float));
        float *ow = malloc(now * sizeof(float));
        float *output = malloc((nx + 2) * sizeof(float));
        float *key = malloc((nkv + 2) * sizeof(float));
        float *value = malloc((nkv + 2) * sizeof(float));
        float *norm = malloc((size_t)dim * sizeof(float));
        float *router = calloc((size_t)dim, sizeof(float));
        BnBlockQ8_0 *q8 = malloc(nqkw / 32 * sizeof(*q8));
        BnBlockQ8_0 *zero = calloc((size_t)dim, sizeof(*zero));
        float qnorm[256], knorm[256], table[128] = {0};
        assert(norm && router && q8 && zero);
        for (size_t b = 0; b < nqkw / 32; b++) {
            q8[b].d = (uint16_t)(0x1800 + (b % 7) * 17);
            for (int j = 0; j < 32; j++)
                q8[b].qs[j] = (int8_t)((int)((b * 17 + j * 13) % 255) - 127);
        }
        assert(input && qkw && vw && ow && output && key && value);
        for (size_t i = 0; i < nx; i++)
            input[i] = (float)((int)((i * 13 + 5) % 97) - 48) / 64.0f;
        for (int i = 0; i < dim; i++) norm[i] = 0.5f + (float)(i % 19) / 16.0f;
        for (size_t i = 0; i < nqkw; i++)
            qkw[i] = (float)((int)((i * 17 + 3) % 101) - 50) / 512.0f;
        for (size_t i = 0; i < nvw; i++)
            vw[i] = (float)((int)((i * 11 + 7) % 103) - 51) / 512.0f;
        for (size_t i = 0; i < now; i++)
            ow[i] = (float)((int)((i * 7 + 5) % 107) - 53) / 4096.0f;
        for (int i = 0; i < hs; i++) {
            qnorm[i] = 0.5f + (float)(i % 19) / 16.0f;
            knorm[i] = 0.4f + (float)(i % 13) / 8.0f;
        }
        BnGPURopeFrequencyPlan recipe = {
            0, 32, 64, 1e7f, NULL, BN_GPU_ROPE_FACTOR_NONE
        };
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = dim; plan.xb2_elements = qd;
        plan.vocab_size = 32; plan.n_layers = 1; plan.n_heads = nh;
        plan.seq_len = 256; plan.head_size = hs; plan.kv_dim = kd; plan.kv_f16 = 1;
        plan.rope_frequencies = table; plan.rope_frequency_count = 128;
        plan.rope_frequency_plans = &recipe; plan.rope_frequency_plan_count = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        const int type = BN_GGUF_TENSOR_F32;
        void *buffers[] = {
            gpu->buffer_create(gpu->ctx, qtype == BN_GGUF_TENSOR_Q8_0
                ? (const void *)q8 : (const void *)qkw,
                qtype == BN_GGUF_TENSOR_Q8_0 ? nqkw / 32 * sizeof(*q8)
                : nqkw * sizeof(float), qtype, rows, dim),
            gpu->buffer_create(gpu->ctx, vw, nvw * sizeof(float), type, kd, dim),
            gpu->buffer_create(gpu->ctx, ow, now * sizeof(float), type, dim, qd),
            gpu->buffer_create(gpu->ctx, norm, (size_t)dim * sizeof(float), type, 1, dim),
            gpu->buffer_create(gpu->ctx, qnorm, sizeof(qnorm), type, 1, hs),
            gpu->buffer_create(gpu->ctx, knorm, sizeof(knorm), type, 1, hs),
            gpu->buffer_create(gpu->ctx, router, (size_t)dim * sizeof(float), type, 1, dim),
            gpu->buffer_create(gpu->ctx, zero, (size_t)dim * sizeof(*zero),
                BN_GGUF_TENSOR_Q8_0, 32, dim),
            gpu->buffer_create(gpu->ctx, zero, (size_t)dim * sizeof(*zero),
                BN_GGUF_TENSOR_Q8_0, dim, 32),
        };
        for (size_t b = 0; b < sizeof(buffers) / sizeof(buffers[0]); b++) assert(buffers[b]);
        output[0] = output[nx + 1] = key[0] = key[nkv + 1] =
            value[0] = value[nkv + 1] = 123.0f;
        for (int alias = 0; alias < 2; alias++) {
            float *out = alias ? input : output + 1;
            assert(gpu->prefill_moe_layer(gpu->ctx, out,
                buffers[0], buffers[1], buffers[2], buffers[6],
                buffers[7], buffers[7], buffers[8], NULL, NULL, NULL, NULL,
                buffers[3], buffers[3], buffers[4], buffers[5],
                NULL, NULL, NULL, input, key + 1, value + 1,
                nt, dim, 32, 1, 1, nh, nk, hs, 8, kd, rows, qtype,
                kd, type, dim, qd, type, BN_GGUF_TENSOR_Q8_0,
                BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0, 0, 0, 0, 0, 0,
                0, 1e-6f, 0, 64, 0u, kd, 0.0625f, 1, 1.0f, 0) == 0);
            uint64_t output_hash = cuda_test_float_bits_hash(out, nx);
            uint64_t key_hash = cuda_test_float_bits_hash(key + 1, nkv);
            uint64_t value_hash = cuda_test_float_bits_hash(value + 1, nkv);
            if (output_hash != cases[c].output || key_hash != cases[c].key ||
                value_hash != cases[c].value)
                fprintf(stderr, "CUDA model attention mismatch: case=%zu dim=%d type=%d tokens=%d gated=%d out=%016llx/%016llx key=%016llx/%016llx value=%016llx/%016llx\n",
                    c, dim, qtype, nt, gated,
                    (unsigned long long)output_hash,
                    (unsigned long long)cases[c].output,
                    (unsigned long long)key_hash,
                    (unsigned long long)cases[c].key,
                    (unsigned long long)value_hash,
                    (unsigned long long)cases[c].value);
            assert(output_hash == cases[c].output);
            assert(key_hash == cases[c].key);
            assert(value_hash == cases[c].value);
            assert(output[0] == 123.0f && output[nx + 1] == 123.0f);
            assert(key[0] == 123.0f && key[nkv + 1] == 123.0f);
            assert(value[0] == 123.0f && value[nkv + 1] == 123.0f);
        }
        for (size_t b = 0; b < sizeof(buffers) / sizeof(buffers[0]); b++)
            gpu->buffer_destroy(gpu->ctx, buffers[b]);
        gpu->free_activations(gpu->ctx);
        free(zero); free(q8); free(router); free(norm);
        free(value); free(key); free(output); free(ow); free(vw); free(qkw); free(input);
    }
    bn_gpu_cuda_destroy(gpu);
    printf("CUDA model attention reference PASSED\n");
}

#include "cuda_rmsnorm_batch_reference.h"
static void run_rmsnorm_batch_reference_case(BnGPUBackend *gpu) {
    assert(bn_gpu_backend_can_rmsnorm_batch(gpu));
    float frequencies[16] = {0};
    BnGPUActivationPlan plan = {0};
    plan.dim = plan.xb2_elements = plan.hb_elements = plan.vocab_size = 32;
    plan.n_layers = plan.attention_layer_count = plan.n_heads = plan.seq_len = 1;
    plan.kv_dim = plan.head_size = 32;
    plan.rope_frequencies = frequencies; plan.rope_frequency_count = 16;
    assert(gpu->init_activations(gpu->ctx, &plan) == 0);
    plan.separate_rope_norm = 2;
    assert(gpu->init_activations(gpu->ctx, &plan) == -1);
    BnGPUOp graph[32] = {{0}};
    float marker = 1.25f;
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, &marker, sizeof(marker), 0) == 0);
    for (int i = 0; i < 32; i++) {
        graph[i].op_code = BN_GPU_CODE_COPY;
        graph[i].buf_in = BN_GPU_VALUE_X; graph[i].buf_out = BN_GPU_VALUE_SCRATCH;
        graph[i].p[2] = 1;
    }
    for (size_t c = 0; c < sizeof(cuda_rmsnorm_batch_reference) /
                                  sizeof(cuda_rmsnorm_batch_reference[0]); c++) {
        const BnCudaRmsnormBatchReference *r = &cuda_rmsnorm_batch_reference[c];
        size_t count = (size_t)r->dim * r->tokens;
        float *x = malloc(count * sizeof(float));
        float *w = malloc((size_t)r->dim * sizeof(float));
        float *out = malloc((count + 2) * sizeof(float));
        assert(x && w && out);
        for (size_t i = 0; i < count; i++)
            x[i] = (float)((int)((i * 37 + r->seed * 11) % 509) - 254) / 259.0f;
        for (int i = 0; i < r->dim; i++)
            w[i] = r->seed ? (float)((i * 7) % 19 - 9) / 16.0f : 0.25f + (float)(i % 13) / 16.0f;
        for (int raw = 0; raw < 2; raw++) {
            void *norm = gpu->buffer_create(gpu->ctx, w, (size_t)r->dim * sizeof(float),
                raw ? -1 : BN_GGUF_TENSOR_F32, 1, r->dim);
            assert(norm);
            for (int alias = 0; alias < 2; alias++) {
                out[0] = out[count + 1] = -12345.0f;
                memcpy(out + 1, x, count * sizeof(float));
                if (alias) assert(gpu->execute(gpu->ctx, graph, 32, -1, NULL, 0) == 0);
                assert(bn_gpu_backend_rmsnorm_batch(gpu, out + 1, norm,
                    alias ? out + 1 : x, r->tokens, r->dim, 1e-6f) == 0);
                assert(cuda_test_float_bits_hash(out + 1, count) == r->hash);
                assert(out[0] == -12345.0f && out[count + 1] == -12345.0f);
            }
            gpu->buffer_destroy(gpu->ctx, norm);
        }
        free(x); free(w); free(out);
    }
    float w[2] = {1.0f, 1.0f}, x[2] = {1.0f, 2.0f}, guard[4];
    void *norm = gpu->buffer_create(gpu->ctx, w, sizeof(w), BN_GGUF_TENSOR_F32, 1, 2);
    void *short_norm = gpu->buffer_create(gpu->ctx, w, sizeof(float), BN_GGUF_TENSOR_F32, 1, 1);
    uint16_t half_w[2] = {0x3c00, 0x3c00};
    void *half_norm = gpu->buffer_create(gpu->ctx, half_w, sizeof(half_w), BN_GGUF_TENSOR_F16, 1, 2);
    assert(norm && short_norm && half_norm);
    for (int i = 0; i < 4; i++) guard[i] = -12345.0f;
    assert(bn_gpu_backend_rmsnorm_batch(NULL, guard + 1, norm, x, 1, 2, 1e-6f) == -1);
    BnGPUBackend absent = {0};
    assert(bn_gpu_backend_rmsnorm_batch(&absent, guard + 1, norm, x, 1, 2, 1e-6f) == -1);
    assert(bn_gpu_backend_rmsnorm_batch(gpu, guard + 1, NULL, x, 1, 2, 1e-6f) == -1);
    assert(bn_gpu_backend_rmsnorm_batch(gpu, guard + 1, short_norm, x, 1, 2, 1e-6f) == -1);
    assert(bn_gpu_backend_rmsnorm_batch(gpu, guard + 1, half_norm, x, 1, 2, 1e-6f) == -1);
    assert(bn_gpu_backend_rmsnorm_batch(gpu, guard + 1, norm, NULL, 1, 2, 1e-6f) == -1);
    assert(bn_gpu_backend_rmsnorm_batch(gpu, NULL, norm, x, 1, 2, 1e-6f) == -1);
    assert(bn_gpu_backend_rmsnorm_batch(gpu, guard + 1, norm, x, 0, 2, 1e-6f) == -1);
    assert(bn_gpu_backend_rmsnorm_batch(gpu, guard + 1, norm, x, 1, 0, 1e-6f) == -1);
    assert(bn_gpu_backend_rmsnorm_batch(gpu, guard + 1, norm, x, INT_MAX, 2, 1e-6f) == -1);
    assert(bn_gpu_backend_rmsnorm_batch(gpu, guard + 1, norm, x, 1, 2, NAN) == -1);
    assert(bn_gpu_backend_rmsnorm_batch(gpu, guard + 1, norm, x, 1, 2, -1.0f) == -1);
    for (int i = 0; i < 4; i++) assert(guard[i] == -12345.0f);
    gpu->buffer_destroy(gpu->ctx, norm); gpu->buffer_destroy(gpu->ctx, short_norm);
    gpu->buffer_destroy(gpu->ctx, half_norm);
    gpu->free_activations(gpu->ctx);
    printf("CUDA RMSNorm batch reference PASSED\n");
}



#include "cuda_norm_residual_reference.h"
static void run_norm_residual_reference_cases(BnGPUBackend *gpu) {
    assert(bn_gpu_backend_can_rmsnorm_residual_batch(gpu));
    for(size_t c=0;c<sizeof(cuda_norm_residual_reference)/sizeof(cuda_norm_residual_reference[0]);c++) {
        const BnCudaNormResidualReference *r=&cuda_norm_residual_reference[c];
        int dim=r->dim,nt=r->nt,seed=r->seed;size_t n=(size_t)dim*nt;
        float *x=malloc((n+2)*sizeof(float)),*res=malloc((n+2)*sizeof(float));
        float *out=malloc((n+2)*sizeof(float)),*w=malloc((size_t)dim*sizeof(float));
        assert(x&&res&&out&&w);
        for(int i=0;i<dim;i++)w[i]=(float)(i%17-8)/7.0f;
        for(int raw=0;raw<2;raw++) {
            void *norm=gpu->buffer_create(gpu->ctx,w,(size_t)dim*sizeof(float),raw?-1:BN_GGUF_TENSOR_F32,dim,1);
            assert(norm);
            for(int alias=0;alias<3;alias++) {
                for(size_t i=0;i<n;i++) {
                    x[i+1]=(float)((int)((i*37+seed*11)%509)-254)/259.0f;
                    res[i+1]=(float)((int)((i*13+seed*7)%257)-128)/31.7f;
                }
                x[0]=x[n+1]=res[0]=res[n+1]=out[0]=out[n+1]=12345.0f;
                float *y=alias==1?x+1:alias==2?res+1:out+1;
                assert(bn_gpu_backend_rmsnorm_residual_batch(gpu,y,norm,x+1,res+1,nt,dim,1e-6f)==0);
                assert(cuda_test_float_bits_hash(y,n)==r->hash);
                assert(bn_gpu_backend_rmsnorm_residual_batch(gpu,y,norm,x+1,NULL,nt,dim,1e-6f)!=0);
                assert(bn_gpu_backend_rmsnorm_residual_batch(gpu,y,norm,x+1,res+1,nt,0,1e-6f)!=0);
                assert(cuda_test_float_bits_hash(y,n)==r->hash);
                assert(x[0]==12345.0f&&x[n+1]==12345.0f&&res[0]==12345.0f&&res[n+1]==12345.0f&&out[0]==12345.0f&&out[n+1]==12345.0f);
            }
            gpu->buffer_destroy(gpu->ctx,norm);
        }
        free(x);free(res);free(out);free(w);
    }
    printf("CUDA norm residual reference PASSED\n");
}

#include "cuda_prepared_v_reference.h"
static void run_prepared_v_reference_cases(BnGPUBackend *gpu) {
    assert(bn_gpu_backend_can_prefill_attention_prepared_v(gpu));
    for (size_t c = 0; c < sizeof(cuda_prepared_v_reference) / sizeof(cuda_prepared_v_reference[0]); c++) {
        const BnCudaPreparedVReference *r = &cuda_prepared_v_reference[c];
        int hs = r->hs, nh = r->nh, nk = r->nk, nt = r->nt;
        int dim = nh * hs, kd = nk * hs, stride = dim * (r->gated ? 2 : 1) + 7;
        size_t nq = (size_t)nt * dim, nkeys = (size_t)nt * kd, nr = (size_t)nt * stride;
        int nwq = (r->per ? nh : 1) * hs, nwk = (r->per ? nk : 1) * hs;
        float table[45] = {0};
        BnGPURopeFrequencyPlan recipe = {13,32,64,10000000.0f,NULL,BN_GPU_ROPE_FACTOR_NONE};
        BnGPUActivationPlan ap = {0};
        ap.dim = ap.hb_elements = ap.xb2_elements = dim;
        ap.vocab_size = 32; ap.n_layers = ap.attention_layer_count = 1;
        ap.n_heads = nh; ap.seq_len = nt > 256 ? nt : 256; ap.head_size = hs; ap.kv_dim = kd; ap.kv_f16 = 1;
        ap.rope_frequencies = table; ap.rope_frequency_count = 45;
        ap.rope_frequency_plans = &recipe; ap.rope_frequency_plan_count = 1;
        assert(gpu->init_activations(gpu->ctx, &ap) == 0);
        unsigned char cache_guard[32], cache_got[32];
        memset(cache_guard,0x5a,sizeof(cache_guard));
        assert(gpu->write_activation(gpu->ctx,BN_GPU_VALUE_KEY_CACHE,cache_guard,32,0)==0);
        assert(gpu->write_activation(gpu->ctx,BN_GPU_VALUE_VALUE_CACHE,cache_guard,32,0)==0);
        float *wq = malloc((size_t)nwq * sizeof(float)), *wk = malloc((size_t)nwk * sizeof(float));
        float *raw = malloc((nr + 2) * sizeof(float)), *key = malloc((nkeys + 2) * sizeof(float));
        float *value = malloc((nkeys + 2) * sizeof(float)), *out = malloc((nq + 2) * sizeof(float));
        float *ko = malloc((nkeys + 2) * sizeof(float)), *vo = malloc((nkeys + 2) * sizeof(float));
        assert(wq && wk && raw && key && value && out && ko && vo);
        for(int i=0;i<nwq;i++)wq[i]=0.5f+(float)(i%19)/16.0f;
        for(int i=0;i<nwk;i++)wk[i]=0.4f+(float)(i%13)/8.0f;
        for(int upload=0;upload<2;upload++) {
            void *qw = gpu->buffer_create(gpu->ctx,wq,(size_t)nwq*4,upload?-1:BN_GGUF_TENSOR_F32,nwq,1);
            void *kw = gpu->buffer_create(gpu->ctx,wk,(size_t)nwk*4,upload?-1:BN_GGUF_TENSOR_F32,nwk,1);
            assert(qw && kw);
            for(int alias=0;alias<2;alias++) {
                for(size_t i=0;i<nr+2;i++)raw[i]=12345.0f;
                key[0]=key[nkeys+1]=value[0]=value[nkeys+1]=12345.0f;
                out[0]=out[nq+1]=ko[0]=ko[nkeys+1]=vo[0]=vo[nkeys+1]=12345.0f;
                for(int t=0;t<nt;t++)for(int h=0;h<nh;h++)for(int d=0;d<hs;d++) {
                    size_t i=((size_t)t*nh+h)*hs+d, dst=(size_t)t*stride+h*hs*(r->gated?2:1)+d;
                    raw[1+dst]=sinf((float)(i*17+3))*0.3f;
                    if(r->gated)raw[1+dst+hs]=cosf((float)(i*5+7))*0.8f;
                }
                for(size_t i=0;i<nkeys;i++){key[i+1]=cosf((float)(i*11+7))*0.7f;value[i+1]=sinf((float)(i*13+5))*0.9f;}
                BnGPUAttentionPrefillPlan p = {0};
                p.n_tokens=nt;p.n_heads=nh;p.n_kv_heads=nk;p.head_size=hs;p.q_row_stride=stride;
                p.q_gated=r->gated;p.qk_norm_per_head=r->per;p.rope_dims=64;
                p.rope_freq_offset=13;p.norm_eps=1e-6f;p.attention_scale=0.0625f;
                float *y=alias?raw+1:out+1, *kout=alias?key+1:ko+1, *vout=alias?value+1:vo+1;
                assert(bn_gpu_backend_prefill_attention_prepared_v(gpu,y,kout,vout,
                    raw+1,key+1,value+1,(r->norm&1)?qw:NULL,(r->norm&2)?kw:NULL,&p)==0);
                uint64_t got_attention=cuda_test_float_bits_hash(y,nq);
                uint64_t got_keys=cuda_test_float_bits_hash(kout,nkeys);
                uint64_t got_values=cuda_test_float_bits_hash(vout,nkeys);
                if(got_attention!=r->attention || got_keys!=r->keys || got_values!=r->values)
                    fprintf(stderr,"prepared V case=%zu upload=%d alias=%d mismatch attention=%016llx/%016llx keys=%016llx/%016llx values=%016llx/%016llx\n",c,upload,alias,
                        (unsigned long long)got_attention,(unsigned long long)r->attention,
                        (unsigned long long)got_keys,(unsigned long long)r->keys,
                        (unsigned long long)got_values,(unsigned long long)r->values);
                assert(cuda_test_float_bits_hash(y,nq)==r->attention);
                assert(cuda_test_float_bits_hash(kout,nkeys)==r->keys);
                assert(cuda_test_float_bits_hash(vout,nkeys)==r->values);
                p.pos0=1;
                assert(bn_gpu_backend_prefill_attention_prepared_v(gpu,y,kout,vout,
                    raw+1,key+1,value+1,qw,kw,&p)!=0);
                assert(cuda_test_float_bits_hash(y,nq)==r->attention);
                assert(cuda_test_float_bits_hash(kout,nkeys)==r->keys);
                assert(cuda_test_float_bits_hash(vout,nkeys)==r->values);
                p.pos0=0;
                p.n_tokens=2049;
                assert(bn_gpu_backend_prefill_attention_prepared_v(gpu,y,kout,vout,
                    raw+1,key+1,value+1,qw,kw,&p)!=0);
                assert(cuda_test_float_bits_hash(y,nq)==r->attention);
                assert(cuda_test_float_bits_hash(kout,nkeys)==r->keys);
                assert(cuda_test_float_bits_hash(vout,nkeys)==r->values);
                p.n_tokens=nt;
                assert(bn_gpu_backend_prefill_attention_prepared_v(gpu,y,kout,NULL,
                    raw+1,key+1,value+1,qw,kw,&p)!=0);
                assert(raw[0]==12345.0f && raw[nr+1]==12345.0f && key[0]==12345.0f && key[nkeys+1]==12345.0f);
                assert(value[0]==12345.0f && value[nkeys+1]==12345.0f && out[0]==12345.0f && out[nq+1]==12345.0f);
                assert(ko[0]==12345.0f && ko[nkeys+1]==12345.0f && vo[0]==12345.0f && vo[nkeys+1]==12345.0f);
            }
            gpu->buffer_destroy(gpu->ctx,qw);gpu->buffer_destroy(gpu->ctx,kw);
        }
        assert(gpu->read_activation(gpu->ctx,BN_GPU_VALUE_KEY_CACHE,cache_got,32,0)==0);
        assert(memcmp(cache_got,cache_guard,32)==0);
        assert(gpu->read_activation(gpu->ctx,BN_GPU_VALUE_VALUE_CACHE,cache_got,32,0)==0);
        assert(memcmp(cache_got,cache_guard,32)==0);
        free(wq);free(wk);free(raw);free(key);free(value);free(out);free(ko);free(vo);
        gpu->free_activations(gpu->ctx);
    }
    printf("CUDA prepared unit V reference PASSED\n");
}

#include "cuda_prepared_rope_reference.h"
static void run_prepared_rope_reference_case(BnGPUBackend *gpu) {
    for (size_t c = 0; c < sizeof(cuda_prepared_rope_reference) /
                                  sizeof(cuda_prepared_rope_reference[0]); c++) {
        const BnCudaPreparedRopeReference *r = &cuda_prepared_rope_reference[c];
        int nt = r->tokens, nh = r->heads, nk = r->kv_heads, hs = 256;
        int dim = nh * hs, stride = dim * (r->gated ? 2 : 1) + 7;
        size_t nq = (size_t)nt * dim, nkeys = (size_t)nt * nk * hs;
        size_t nr = (size_t)nt * stride;
        float frequencies[45] = {0};
        BnGPURopeFrequencyPlan recipe = {13, 32, 64, 1e7f, NULL, BN_GPU_ROPE_FACTOR_NONE};
        BnGPUActivationPlan plan = {0};
        plan.dim = plan.hb_elements = plan.xb2_elements = dim;
        plan.vocab_size = 32; plan.n_layers = plan.attention_layer_count = 1;
        plan.n_heads = nh; plan.seq_len = 256; plan.head_size = hs;
        plan.kv_dim = nk * hs; plan.kv_f16 = 1;
        plan.rope_frequencies = frequencies; plan.rope_frequency_count = 45;
        plan.rope_frequency_plans = &recipe; plan.rope_frequency_plan_count = 1;
        plan.separate_rope_norm = r->separate;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        BnGPUOp graph[32] = {{0}};
        float marker = 1.25f;
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, &marker, sizeof(marker), 0) == 0);
        for (int i = 0; i < 32; i++) {
            graph[i].op_code = BN_GPU_CODE_COPY;
            graph[i].buf_in = BN_GPU_VALUE_X; graph[i].buf_out = BN_GPU_VALUE_SCRATCH;
            graph[i].p[2] = 1;
        }
        int wqn = (r->per_head ? nh : 1) * hs, wkn = (r->per_head ? nk : 1) * hs;
        float *wq = malloc((size_t)wqn * sizeof(float));
        float *wk = malloc((size_t)wkn * sizeof(float));
        float *raw_guard = malloc((nr + 2) * sizeof(float));
        float *key_guard = malloc((nkeys + 2) * sizeof(float));
        float *value = malloc(nkeys * sizeof(float));
        float *out = malloc((nq + 2) * sizeof(float));
        float *kout = malloc((nkeys + 2) * sizeof(float));
        assert(wq && wk && raw_guard && key_guard && value && out && kout);
        float *raw = raw_guard + 1, *key = key_guard + 1;
        for (int i = 0; i < wqn; i++) wq[i] = 0.5f + (float)(i % 19) / 16.0f;
        for (int i = 0; i < wkn; i++) wk[i] = 0.4f + (float)(i % 13) / 8.0f;
        for (int raw_norm = 0; raw_norm < 2; raw_norm++) {
            void *qw = gpu->buffer_create(gpu->ctx, wq, (size_t)wqn * sizeof(float),
                raw_norm ? -1 : BN_GGUF_TENSOR_F32, 1, wqn);
            void *kw = gpu->buffer_create(gpu->ctx, wk, (size_t)wkn * sizeof(float),
                raw_norm ? -1 : BN_GGUF_TENSOR_F32, 1, wkn);
            assert(qw && kw);
            for (int alias = 0; alias < 2; alias++) {
                for (size_t i = 0; i < nr; i++) raw[i] = 99.0f;
                for (int t = 0; t < nt; t++) for (int h = 0; h < nh; h++)
                    for (int d = 0; d < hs; d++) {
                        size_t src = ((size_t)t * nh + h) * hs + d;
                        size_t dst = (size_t)t * stride + h * hs * (r->gated ? 2 : 1) + d;
                        raw[dst] = sinf((float)(src * 17 + 3)) * 0.3f;
                        if (r->gated) raw[dst + hs] = cosf((float)(src * 5 + 7)) * 0.8f;
                    }
                for (size_t i = 0; i < nkeys; i++) {
                    key[i] = cosf((float)(i * 11 + 7)) * 0.7f;
                    value[i] = sinf((float)(i * 13 + 5)) * 0.9f;
                }
                raw_guard[0] = raw_guard[nr + 1] = key_guard[0] = key_guard[nkeys + 1] = -12345.0f;
                out[0] = out[nq + 1] = kout[0] = kout[nkeys + 1] = -12345.0f;
                if (alias) assert(gpu->execute(gpu->ctx, graph, 32, -1, NULL, 0) == 0);
                BnGPUAttentionPrefillPlan p = {0};
                p.n_tokens = nt; p.n_heads = nh; p.n_kv_heads = nk; p.head_size = hs;
                p.q_row_stride = stride; p.q_gated = r->gated; p.qk_norm_per_head = r->per_head;
                p.rope_dims = 64; p.rope_freq_offset = 13; p.norm_eps = 1e-6f; p.attention_scale = 0.0625f;
                float *result = alias ? raw : out + 1, *keys = alias ? key : kout + 1;
                assert(bn_gpu_backend_prefill_attention_prepared(gpu, result, keys, raw, key, value,
                    (r->norm & 1) ? qw : NULL, (r->norm & 2) ? kw : NULL, &p) == 0);
                uint64_t got = cuda_test_float_bits_hash(result, nq);
                uint64_t got_keys = cuda_test_float_bits_hash(keys, nkeys);
                if (got != r->attention || got_keys != r->keys)
                    fprintf(stderr, "prepared RoPE case=%zu raw=%d alias=%d mismatch\n", c, raw_norm, alias);
                assert(got == r->attention && got_keys == r->keys);
                assert(raw_guard[0] == -12345.0f && raw_guard[nr + 1] == -12345.0f);
                assert(key_guard[0] == -12345.0f && key_guard[nkeys + 1] == -12345.0f);
                assert(out[0] == -12345.0f && out[nq + 1] == -12345.0f);
                assert(kout[0] == -12345.0f && kout[nkeys + 1] == -12345.0f);
            }
            gpu->buffer_destroy(gpu->ctx, qw); gpu->buffer_destroy(gpu->ctx, kw);
        }
        free(wq); free(wk); free(raw_guard); free(key_guard); free(value); free(out); free(kout);
        gpu->free_activations(gpu->ctx);
    }
    printf("CUDA fused and separate RoPE reference PASSED\n");
}

static void run_prefix_attention_reference_cases(BnGPUBackend *gpu, int smoke) {
    if (!bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_PREFILL_PREFIX_KV)) {
        printf("CUDA prefix attention reference skipped: capability unavailable\n");
        return;
    }
    size_t words = 0, tested = 0;
    const size_t count = sizeof(cuda_prefix_attention_refs) / sizeof(cuda_prefix_attention_refs[0]);
    for (size_t ci = 0; ci < count; ci++) {
        const BnCUDAPrefixAttentionReference *r = &cuda_prefix_attention_refs[ci];
        if (smoke && ci != 0 && !(r->pos==512 && r->nk==2 &&
                r->seed==23 && !r->unit && !r->window &&
                (r->nt==2 || r->nt==3 || r->nt==4 || r->nt==34 || r->nt==512)))
            continue;
        const int hs = 256, nh = r->nk * 16, dim = nh * hs, kd = r->nk * hs;
        const int keys = r->pos + r->nt, seq = keys + 3, stride = kd + 16;
        const size_t base = (size_t)seq * stride + 7;
        const size_t nc = (size_t)2 * seq * stride, nq = (size_t)r->nt * dim;
        const size_t nk = (size_t)r->nt * kd;
        float zero_frequency = 0.0f;
        BnGPUActivationPlan ap = {0};
        ap.dim = ap.hb_elements = ap.xb2_elements = dim;
        ap.vocab_size = 32; ap.n_layers = ap.attention_layer_count = 2;
        ap.n_heads = nh; ap.seq_len = seq; ap.head_size = hs;
        ap.kv_dim = stride; ap.kv_f16 = 1;
        ap.rope_frequencies = &zero_frequency; ap.rope_frequency_count = 1;
        assert(gpu->init_activations(gpu->ctx, &ap) == 0);
        float *q = malloc(nq * sizeof(float)), *k = malloc(nk * sizeof(float));
        float *v = malloc(nk * sizeof(float)), *out = malloc((nq + 2) * sizeof(float));
        float *ko = malloc((nk + 2) * sizeof(float));
        float *vo = malloc((nk + 2) * sizeof(float));
        uint16_t *kc = malloc(nc * sizeof(uint16_t)), *vc = malloc(nc * sizeof(uint16_t));
        uint16_t *got = malloc(nc * sizeof(uint16_t));
        assert(q && k && v && out && ko && vo && kc && vc && got);
        for (size_t i = 0; i < nc; i++) kc[i] = vc[i] = 0x3555;
        const float mag = r->seed ? 3.0f : 1.0f;
        for (size_t i = 0; i < nq; i++)
            q[i] = sinf((float)(i * 17 + 3 + r->seed)) * (1.31f * mag);
        for (int t = 0; t < keys; t++) for (int d = 0; d < kd; d++) {
            const size_t i = (size_t)t * kd + d, at = base + (size_t)t * stride + d;
            uint16_t a = bn_fp32_to_fp16(sinf((float)(i * 11 + 9 + r->seed)) * (1.7f * mag));
            uint16_t b = bn_fp32_to_fp16(cosf((float)(i * 7 + 5 + r->seed)) * (0.73f * mag));
            if (t < r->pos) { kc[at] = a; vc[at] = b; }
            else { k[(size_t)(t-r->pos)*kd+d] = bn_fp16_to_fp32(a);
                   v[(size_t)(t-r->pos)*kd+d] = bn_fp16_to_fp32(b); }
        }
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE, kc, nc*2, 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_VALUE_CACHE, vc, nc*2, 0) == 0);
        BnGPUAttentionPrefillPlan p = {0};
        p.n_tokens=r->nt; p.n_heads=nh; p.n_kv_heads=r->nk; p.head_size=hs;
        p.q_row_stride=dim; p.pos0=r->pos; p.rope_dims=2; p.norm_eps=1e-6f;
        p.attention_scale=r->unit ? 1.0f : 0.0625f; p.attention_window=r->window;
        p.kv_cache_off=base+(size_t)r->pos*stride; p.kv_cache_stride=stride;
        assert(gpu->prefill_attention_prefix_supported);
        assert(gpu->prefill_attention_prefix_supported(gpu->ctx,&p)==1);
        out[0]=out[nq+1]=ko[0]=ko[nk+1]=vo[0]=vo[nk+1]=12345.0f;
        int rc = r->normalize_v
            ? bn_gpu_backend_prefill_attention_prepared_v(gpu,out+1,ko+1,vo+1,q,k,v,NULL,NULL,&p)
            : bn_gpu_backend_prefill_attention_prepared(gpu,out+1,ko+1,q,k,v,NULL,NULL,&p);
        assert(rc==0);
        if (r->normalize_v) assert(cuda_test_float_bits_hash(vo+1,nk)==r->vhash);
        assert(vo[0]==12345.0f && vo[nk+1]==12345.0f);
        uint64_t hash = cuda_test_float_bits_hash(out+1,nq);
        if (hash != r->hash) fprintf(stderr,"prefix case=%zu nt=%d pos=%d hash=%016llx expected=%016llx\n",
            ci,r->nt,r->pos,(unsigned long long)hash,(unsigned long long)r->hash);
        assert(hash==r->hash);
        assert(memcmp(ko+1,k,nk*sizeof(float))==0);
        assert(out[0]==12345.0f && out[nq+1]==12345.0f && ko[0]==12345.0f && ko[nk+1]==12345.0f);
        if (ci == 0 || ci == 1440) {
            for (int bad=0; bad<8; bad++) {
                BnGPUAttentionPrefillPlan reject=p;
                switch(bad) {
                case 0: reject.kv_cache_off=SIZE_MAX; break;
                case 1: reject.kv_cache_off=(size_t)p.pos0*stride-1; break;
                case 2: reject.kv_cache_stride=kd-1; break;
                case 3: reject.pos0=INT_MAX; break;
                case 4: reject.pos0=1; reject.n_tokens=2; break;
                case 5: reject.n_tokens=1; break;
                case 6: reject.n_heads=nh/2; break;
                default: reject.kv_cache_off=nc; break;
                }
                assert(gpu->prefill_attention_prefix_supported(gpu->ctx,&reject)==0);
                rc = r->normalize_v
                    ? bn_gpu_backend_prefill_attention_prepared_v(gpu,out+1,ko+1,vo+1,q,k,v,NULL,NULL,&reject)
                    : bn_gpu_backend_prefill_attention_prepared(gpu,out+1,ko+1,q,k,v,NULL,NULL,&reject);
                assert(rc==-1);
                if(r->normalize_v)assert(cuda_test_float_bits_hash(vo+1,nk)==r->vhash);
                assert(cuda_test_float_bits_hash(out+1,nq)==r->hash);
                assert(memcmp(ko+1,k,nk*sizeof(float))==0);
            }
        }
        assert(gpu->read_activation(gpu->ctx,BN_GPU_VALUE_KEY_CACHE,got,nc*2,0)==0);
        assert(memcmp(got,kc,nc*2)==0);
        assert(gpu->read_activation(gpu->ctx,BN_GPU_VALUE_VALUE_CACHE,got,nc*2,0)==0);
        assert(memcmp(got,vc,nc*2)==0);
        if (ci==0 || r->normalize_v) {
            rc = r->normalize_v
                ? bn_gpu_backend_prefill_attention_prepared_v(gpu,q,k,v,q,k,v,NULL,NULL,&p)
                : bn_gpu_backend_prefill_attention_prepared(gpu,q,k,q,k,v,NULL,NULL,&p);
            assert(rc==0 && cuda_test_float_bits_hash(q,nq)==r->hash);
            assert(memcmp(k,ko+1,nk*sizeof(float))==0);
            if(r->normalize_v)assert(cuda_test_float_bits_hash(v,nk)==r->vhash);
        }
        if (ci == 0) {
            BnGPUAttentionPrefillPlan no_frequency=p;
            no_frequency.rope_freq_offset=SIZE_MAX;
            assert(gpu->prefill_attention_prefix_supported(gpu->ctx,&no_frequency)==0);
            assert(gpu->prefill_attention_prefix_supported(gpu->ctx,NULL)==0);
            assert(gpu->prefill_attention_prefix_supported(NULL,&p)==0);
            no_frequency=p;
            no_frequency.rope_dims=3;
            assert(gpu->prefill_attention_prefix_supported(gpu->ctx,&no_frequency)==0);
            ap.kv_f16=0;
            assert(gpu->init_activations(gpu->ctx,&ap)==0);
            assert(gpu->prefill_attention_prefix_supported(gpu->ctx,&p)==0);
            assert(bn_gpu_backend_prefill_attention_prepared(gpu,out+1,ko+1,q,k,v,NULL,NULL,&p)==-1);
            assert(cuda_test_float_bits_hash(out+1,nq)==r->hash);
            gpu->free_activations(gpu->ctx);
            assert(gpu->prefill_attention_prefix_supported(gpu->ctx,&p)==0);
            assert(bn_gpu_backend_prefill_attention_prepared(gpu,out+1,ko+1,q,k,v,NULL,NULL,&p)==-1);
            assert(cuda_test_float_bits_hash(out+1,nq)==r->hash);
        }
        tested++;
        words+=nq;
        free(got); free(vc); free(kc); free(vo); free(ko); free(out); free(v); free(k); free(q);
        gpu->free_activations(gpu->ctx);
    }
    printf("CUDA prefix attention reference PASSED: %zu cases, %zu words\n",tested,words);
}

static void run_prepared_attention_reference_case(BnGPUBackend *gpu) {
    /* Independent llama.cpp norm/RoPE/flash-attention/sigmoid graphs at
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e; keys precede F16 casting. */
    static const uint64_t hashes[48][2] = {
        {UINT64_C(0x62b05f6c61f85f2a), UINT64_C(0x90e390346933c499)},
        {UINT64_C(0xb4ce4e4a8ac276a1), UINT64_C(0x90e390346933c499)},
        {UINT64_C(0x24712d3fd30c73ec), UINT64_C(0x6f1629500d004c82)},
        {UINT64_C(0xef783f81a0618260), UINT64_C(0x6f1629500d004c82)},
        {UINT64_C(0x62b05f6c61f85f2a), UINT64_C(0x90e390346933c499)},
        {UINT64_C(0x0d913483d022761b), UINT64_C(0x90e390346933c499)},
        {UINT64_C(0xa0c28f65f285da0c), UINT64_C(0x3339d886df632b98)},
        {UINT64_C(0x6ada8b2cce46661d), UINT64_C(0x3339d886df632b98)},
        {UINT64_C(0x8597c69792e7797f), UINT64_C(0x90e390346933c499)},
        {UINT64_C(0x5e25fff58e61f4b1), UINT64_C(0x90e390346933c499)},
        {UINT64_C(0x04845e577b186f68), UINT64_C(0x6f1629500d004c82)},
        {UINT64_C(0x2bc8d4d8481aeed9), UINT64_C(0x6f1629500d004c82)},
        {UINT64_C(0x8597c69792e7797f), UINT64_C(0x90e390346933c499)},
        {UINT64_C(0x716478c1ab74d08c), UINT64_C(0x90e390346933c499)},
        {UINT64_C(0xbea63a60196eb80c), UINT64_C(0x3339d886df632b98)},
        {UINT64_C(0x5256571ad7e47926), UINT64_C(0x3339d886df632b98)},
        {UINT64_C(0x8430aee5f749c4a5), UINT64_C(0x882da6e69d6ad6b7)},
        {UINT64_C(0xd3bd913667a0ba70), UINT64_C(0x882da6e69d6ad6b7)},
        {UINT64_C(0xf5fec63ffc04a09d), UINT64_C(0x34e62564cf341642)},
        {UINT64_C(0x0db213cb021083cc), UINT64_C(0x34e62564cf341642)},
        {UINT64_C(0x8430aee5f749c4a5), UINT64_C(0x882da6e69d6ad6b7)},
        {UINT64_C(0x33db683c3b0b7f01), UINT64_C(0x882da6e69d6ad6b7)},
        {UINT64_C(0x8d0f355bf3818160), UINT64_C(0xfd7762f121e9bfcb)},
        {UINT64_C(0x1c96320c8beaaef9), UINT64_C(0xfd7762f121e9bfcb)},
        {UINT64_C(0x61c494e10e5f2278), UINT64_C(0x882da6e69d6ad6b7)},
        {UINT64_C(0xc5805ec85969deba), UINT64_C(0x882da6e69d6ad6b7)},
        {UINT64_C(0x3ddcc56a95a19df4), UINT64_C(0x34e62564cf341642)},
        {UINT64_C(0xd59f667594035b28), UINT64_C(0x34e62564cf341642)},
        {UINT64_C(0x61c494e10e5f2278), UINT64_C(0x882da6e69d6ad6b7)},
        {UINT64_C(0x49d33d3ad43204f6), UINT64_C(0x882da6e69d6ad6b7)},
        {UINT64_C(0x0114772c10acf607), UINT64_C(0xfd7762f121e9bfcb)},
        {UINT64_C(0x28b714ecb225fc98), UINT64_C(0xfd7762f121e9bfcb)},
        {UINT64_C(0x23bafbb6250c60df), UINT64_C(0x10a7ffb1799112b1)},
        {UINT64_C(0x0580989a4542fdeb), UINT64_C(0x10a7ffb1799112b1)},
        {UINT64_C(0x3d9944e2a94c534b), UINT64_C(0xe5b4b3400309a0c5)},
        {UINT64_C(0x957e0d42db00b290), UINT64_C(0xe5b4b3400309a0c5)},
        {UINT64_C(0x23bafbb6250c60df), UINT64_C(0x10a7ffb1799112b1)},
        {UINT64_C(0x8e0ca3e04713d5f5), UINT64_C(0x10a7ffb1799112b1)},
        {UINT64_C(0x9f70316c5d17c50a), UINT64_C(0x74f7f3f527035b24)},
        {UINT64_C(0x031fde3c2d671638), UINT64_C(0x74f7f3f527035b24)},
        {UINT64_C(0xc1fb502edeb6360a), UINT64_C(0x10a7ffb1799112b1)},
        {UINT64_C(0xcfbdebbb94ac0852), UINT64_C(0x10a7ffb1799112b1)},
        {UINT64_C(0x519bd69a918faa08), UINT64_C(0xe5b4b3400309a0c5)},
        {UINT64_C(0x03ad1734cc562bc6), UINT64_C(0xe5b4b3400309a0c5)},
        {UINT64_C(0xc1fb502edeb6360a), UINT64_C(0x10a7ffb1799112b1)},
        {UINT64_C(0x254936fd45dcba3e), UINT64_C(0x10a7ffb1799112b1)},
        {UINT64_C(0x369879e13ae0147a), UINT64_C(0x74f7f3f527035b24)},
        {UINT64_C(0xde7a4cd71f4d179a), UINT64_C(0x74f7f3f527035b24)},
    };
    static const int counts[] = {2, 9, 16};
    enum { hs = 256, nh = 16, nk = 2, rd = 64, dim = nh * hs };
    float frequencies[45] = {0};
    BnGPURopeFrequencyPlan recipe = {13, 32, rd, 10000000.0f, NULL, BN_GPU_ROPE_FACTOR_NONE};
    BnGPUActivationPlan plan = {0};
    plan.dim = plan.hb_elements = plan.xb2_elements = dim;
    plan.vocab_size = 32; plan.n_layers = plan.attention_layer_count = 1;
    plan.n_heads = nh; plan.seq_len = 256; plan.head_size = hs;
    plan.kv_dim = nk * hs; plan.kv_f16 = 1;
    plan.rope_frequencies = frequencies; plan.rope_frequency_count = 45;
    plan.rope_frequency_plans = &recipe; plan.rope_frequency_plan_count = 1;
    assert(gpu->init_activations(gpu->ctx, &plan) == 0);
    unsigned char cache_guard[32], cache_got[32];
    memset(cache_guard, 0x5a, sizeof(cache_guard));
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE,
        cache_guard, sizeof(cache_guard), 0) == 0);
    for (int raw_norm = 0; raw_norm < 2; raw_norm++) {
        int ci = 0;
        for (int ti = 0; ti < 3; ti++) {
            int nt = counts[ti]; size_t nq = (size_t)nt * dim, nkeys = (size_t)nt * nk * hs;
            for (int gated = 0; gated < 2; gated++) {
                int stride = dim * (gated ? 2 : 1) + 7;
                size_t nr = (size_t)nt * stride;
                for (int per = 0; per < 2; per++) {
                    int wqn = (per ? nh : 1) * hs, wkn = (per ? nk : 1) * hs;
                    float *wq = (float *)malloc((size_t)wqn * sizeof(float));
                    float *wk = (float *)malloc((size_t)wkn * sizeof(float));
                    assert(wq && wk);
                    for (int i = 0; i < wqn; i++) wq[i] = 0.5f + (float)(i % 19) / 16.0f;
                    for (int i = 0; i < wkn; i++) wk[i] = 0.4f + (float)(i % 13) / 8.0f;
                    void *qw = gpu->buffer_create(gpu->ctx, wq, (size_t)wqn * sizeof(float),
                        raw_norm ? -1 : BN_GGUF_TENSOR_F32,
                        raw_norm ? wqn : (per ? nh : 1), raw_norm ? 1 : hs);
                    void *kw = gpu->buffer_create(gpu->ctx, wk, (size_t)wkn * sizeof(float),
                        raw_norm ? -1 : BN_GGUF_TENSOR_F32,
                        raw_norm ? wkn : (per ? nk : 1), raw_norm ? 1 : hs);
                    assert(qw && kw);
                    for (int norm = 0; norm < 4; norm++, ci++) {
                        float *raw = (float *)malloc(nr * sizeof(float));
                        float *key = (float *)malloc(nkeys * sizeof(float));
                        float *value = (float *)malloc(nkeys * sizeof(float));
                        float *out = (float *)malloc((nq + 2) * sizeof(float));
                        float *kout = (float *)malloc((nkeys + 2) * sizeof(float));
                        assert(raw && key && value && out && kout);
                        for (size_t i = 0; i < nr; i++) raw[i] = 99.0f;
                        for (int t = 0; t < nt; t++) for (int h = 0; h < nh; h++) {
                            for (int d = 0; d < hs; d++) {
                                size_t src = ((size_t)t * nh + h) * hs + d;
                                size_t dst = (size_t)t * stride + h * hs * (gated ? 2 : 1) + d;
                                raw[dst] = sinf((float)(src * 17 + 3)) * 0.3f;
                                if (gated) raw[dst + hs] = cosf((float)(src * 5 + 7)) * 0.8f;
                            }
                        }
                        for (size_t i = 0; i < nkeys; i++) {
                            key[i] = cosf((float)(i * 11 + 7)) * 0.7f;
                            value[i] = sinf((float)(i * 13 + 5)) * 0.9f;
                        }
                        BnGPUAttentionPrefillPlan p = {
                            nt, nh, nk, hs, stride, gated, per, 0,
                            0, rd, 0, 13, 1e-6f, 0.0625f, 0, 0
                        };
                        out[0] = out[nq + 1] = kout[0] = kout[nkeys + 1] = 12345.0f;
                        assert(bn_gpu_backend_prefill_attention_prepared(gpu,
                            out + 1, kout + 1, raw, key, value,
                            (norm & 1) ? qw : NULL, (norm & 2) ? kw : NULL, &p) == 0);
                        assert(cuda_test_float_bits_hash(out + 1, nq) == hashes[ci][0]);
                        assert(cuda_test_float_bits_hash(kout + 1, nkeys) == hashes[ci][1]);
                        assert(out[0] == 12345.0f && out[nq + 1] == 12345.0f &&
                               kout[0] == 12345.0f && kout[nkeys + 1] == 12345.0f);
                        p.pos0 = 1;
                        assert(bn_gpu_backend_prefill_attention_prepared(gpu,
                            out + 1, kout + 1, raw, key, value, qw, kw, &p) == -1);
                        p.pos0 = 0; p.rope_freq_offset = SIZE_MAX;
                        assert(bn_gpu_backend_prefill_attention_prepared(gpu,
                            out + 1, kout + 1, raw, key, value, qw, kw, &p) == -1);
                        assert(cuda_test_float_bits_hash(out + 1, nq) == hashes[ci][0]);
                        assert(cuda_test_float_bits_hash(kout + 1, nkeys) == hashes[ci][1]);
                        p.rope_freq_offset = 13;
                        assert(bn_gpu_backend_prefill_attention_prepared(gpu,
                            raw, key, raw, key, value,
                            (norm & 1) ? qw : NULL, (norm & 2) ? kw : NULL, &p) == 0);
                        assert(cuda_test_float_bits_hash(raw, nq) == hashes[ci][0]);
                        assert(cuda_test_float_bits_hash(key, nkeys) == hashes[ci][1]);
                        free(kout); free(out); free(value); free(key); free(raw);
                    }
                    gpu->buffer_destroy(gpu->ctx, qw); gpu->buffer_destroy(gpu->ctx, kw);
                    free(wk); free(wq);
                }
            }
        }
        assert(ci == 48);
    }
    assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE,
        cache_got, sizeof(cache_got), 0) == 0);
    assert(memcmp(cache_got, cache_guard, sizeof(cache_guard)) == 0);
    printf("CUDA prepared attention reference PASSED\n");
}

static void run_gqa2_attention_reference_case(BnGPUBackend *gpu) {
    /* Independent llama.cpp 3d3d7c8181 CUDA flash-attention oracle, SM120.
     * GQA2, head256, F16 KV, causal mask; standard and Gemma unit scales. */
    static const uint64_t hashes[] = {
        UINT64_C(0x85a09c28ab38758f),
        UINT64_C(0x8707f42e5ac353be),
        UINT64_C(0x4375562825ac2fd7),
        UINT64_C(0xfbc165cfa8234d54),
        UINT64_C(0x988740b90e7ae11a),
        UINT64_C(0x035b6916c35bb314),
        UINT64_C(0xaed46eb8632d2f73),
        UINT64_C(0xdc188e7908318767),
        UINT64_C(0x4e233d787932a92c),
        UINT64_C(0xea635e08438a227b),
        UINT64_C(0x9dca24415f0d97c6),
        UINT64_C(0x340009594cb255ba),
        UINT64_C(0x860ce4e7a3b6d483),
        UINT64_C(0x1b496eaa9410dcaa),
        UINT64_C(0x35b14084f4d3e283),
        UINT64_C(0x34c6b08df7944e39),
        UINT64_C(0xe74d64ed6d37016f),
        UINT64_C(0x94a0d7a1c7d07293),
        UINT64_C(0x9c1cee6ca19af419),
        UINT64_C(0x0b02e3dfb3c8028d),
        UINT64_C(0xaff3a9972c970159),
        UINT64_C(0xd9c76c0e50669f77),
        UINT64_C(0x92986118f8d999f5),
        UINT64_C(0x016d8bef4c2b2bbf),
        UINT64_C(0x076f6ff31c0ed27e),
        UINT64_C(0x7c98e60fcd26e4f9),
        UINT64_C(0x514ca86ae72e2608),
        UINT64_C(0x9ec0c82a9ed2874f),
        UINT64_C(0xce5afae3fb2487f0),
        UINT64_C(0x98e308c190e5e2e2),
        UINT64_C(0x5e9fbf6a071de26c),
        UINT64_C(0x1090229d920c354a),
        UINT64_C(0x237371d5ffa41eaf),
        UINT64_C(0xb5b3a294706b9078),
        UINT64_C(0xa1413c65888e097e),
        UINT64_C(0xb104d3a0d4defdce),
        UINT64_C(0x6f11985578e1b430),
        UINT64_C(0xafb67dfe7294c925),
        UINT64_C(0x1c52e361767ac40e),
        UINT64_C(0xb44e9f3063e0dafb),
        UINT64_C(0xce07ff650e9ff342),
        UINT64_C(0x5515fbf67af4207f),
        UINT64_C(0xf2c81f8aa9382454),
        UINT64_C(0xb9ee8b9682da5a6d),
        UINT64_C(0x8ac387f2e6259450),
        UINT64_C(0xb1429c07579216dc),
        UINT64_C(0x80ddfde448e7d616),
        UINT64_C(0x65c8be13dc801fa0),
        UINT64_C(0x8f78d727161ea52a),
        UINT64_C(0x0c5f01152d3b63b2),
        UINT64_C(0x27ed62de6f877fbb),
        UINT64_C(0xbafc033fa136218f),
        UINT64_C(0x03b2096afa134634),
        UINT64_C(0x5022d38dddcc7063),
        UINT64_C(0x9d1687dfce6e3e3b),
        UINT64_C(0x8802cdc9e61c663a),
        UINT64_C(0x159159b313d8ad9b),
        UINT64_C(0x6e6715a3eb71dffe),
        UINT64_C(0xf2680546700aba49),
        UINT64_C(0x080a5e24a549242f),
        UINT64_C(0x8062e9bcda3a180a),
        UINT64_C(0xef4dfccede15c2ba),
        UINT64_C(0xd64fcb34a2398d4c),
        UINT64_C(0xe2ad90f3f575cb0b),
        UINT64_C(0xdf1fdbc7f90b3e24),
        UINT64_C(0x4964a43f0a58435c),
        UINT64_C(0x9a12ed494b1e904e),
        UINT64_C(0x6c1ed8262dd3a3e1),
        UINT64_C(0xd786494715b31029),
        UINT64_C(0xd6fb775d2d83f918),
        UINT64_C(0x25655016bef8887d),
        UINT64_C(0xc46a34f05f7aefc7),
    };
    const int tokens[] = {2, 8, 9, 16, 21, 32};
    const int kv_heads[] = {1, 2, 16};
    size_t case_index = 0;
    for (int unit_scale = 0; unit_scale < 2; unit_scale++) {
        float scale = unit_scale ? 1.0f : 0.0625f;
        for (size_t ti = 0; ti < sizeof(tokens) / sizeof(tokens[0]); ti++) {
            int nt = tokens[ti];
            for (size_t ki = 0; ki < sizeof(kv_heads) / sizeof(kv_heads[0]); ki++) {
                int nk = kv_heads[ki], nh = nk * 2, d = 256, dim = nh * d;
                BnConfig cfg = {0};
                cfg.dim = cfg.hidden_dim = dim; cfg.vocab_size = 32;
                cfg.n_layers = 1; cfg.seq_len = 256;
                cfg.n_heads = nh; cfg.head_size = d; cfg.kv_dim = nk * d;
                cfg.kv_f16 = 1; cfg.rope_theta = 10000.0f;
                assert(init_test_activations(gpu, &cfg) == 0);
                size_t nq = (size_t)nt * dim, nkv = (size_t)nt * nk * d;
                float *q = malloc(nq * sizeof(float));
                float *k = malloc(nkv * sizeof(float));
                float *v = malloc(nkv * sizeof(float));
                float *out = malloc((nq + 2) * sizeof(float));
                assert(q && k && v && out);
                for (int si = 0; si < 2; si++, case_index++) {
                    int seed = si ? 23 : 0;
                    float mag = si ? 3.0f : 1.0f;
                    for (size_t i = 0; i < nq; i++)
                        q[i] = sinf((float)(i * 17 + 3 + seed)) * (1.31f * mag);
                    for (size_t i = 0; i < nkv; i++) {
                        k[i] = bn_fp16_to_fp32(bn_fp32_to_fp16(
                            sinf((float)(i * 11 + 9 + seed)) * (1.7f * mag)));
                        v[i] = bn_fp16_to_fp32(bn_fp32_to_fp16(
                            cosf((float)(i * 7 + 5 + seed)) * (0.73f * mag)));
                    }
                    out[0] = out[nq + 1] = 12345.0f;
                    assert(bn_gpu_backend_prefill_attention(gpu, out + 1,
                        q, k, v, nt, nh, nk, d, 2, nk * d, scale, 0) == 0);
                    assert(cuda_test_float_bits_hash(out + 1, nq) == hashes[case_index]);
                    assert(out[0] == 12345.0f && out[nq + 1] == 12345.0f);
                    memcpy(out + 1, q, nq * sizeof(float));
                    assert(bn_gpu_backend_prefill_attention(gpu, out + 1,
                        out + 1, k, v, nt, nh, nk, d, 2, nk * d, scale, 0) == 0);
                    assert(cuda_test_float_bits_hash(out + 1, nq) == hashes[case_index]);
                    assert(out[0] == 12345.0f && out[nq + 1] == 12345.0f);
                }
                free(out); free(v); free(k); free(q);
                gpu->free_activations(gpu->ctx);
            }
        }
    }
    assert(case_index == sizeof(hashes) / sizeof(hashes[0]));
    printf("CUDA GQA2 attention reference PASSED\n");
}

static void run_head512_attention_reference_case(BnGPUBackend *gpu) {
    /* Independent llama.cpp 3d3d7c8181 CUDA flash-attention oracle, SM120.
     * GQA8, head512, F16 KV, causal mask; standard and Gemma unit scales. */
    static const uint64_t hashes[] = {
        UINT64_C(0x9e46f380e9ea2e5c),
        UINT64_C(0xadaef257e005eace),
        UINT64_C(0x0341ffd53c464350),
        UINT64_C(0x6022016d4eea6f12),
        UINT64_C(0xd29c4db267bed19f),
        UINT64_C(0xf6018557ff59eee4),
        UINT64_C(0x68680bd82d6064aa),
        UINT64_C(0x82fd4dbaecc114b3),
        UINT64_C(0x68eadc20681245e8),
        UINT64_C(0x9a40035cf789870c),
        UINT64_C(0x2eb1b6297fd1fcf3),
        UINT64_C(0xafde41bfa9ceb316),
        UINT64_C(0x4ad05c03a62598d6),
        UINT64_C(0xf63eb33ec12cf1a8),
        UINT64_C(0x9c19226b7a9d0efd),
        UINT64_C(0x6c39ca1b3b8f140b),
        UINT64_C(0x715440fcce940d07),
        UINT64_C(0x44f11c1c719d7997),
        UINT64_C(0x30a67f6579137c3f),
        UINT64_C(0xe76ccfbb75919faf),
        UINT64_C(0x2425022a33abcfe1),
        UINT64_C(0x056e0d9384692e7f),
        UINT64_C(0xd7fc2f464f74682f),
        UINT64_C(0xaac12c7a02ef673d),
        UINT64_C(0x5670a5c9aa35dcbc),
        UINT64_C(0xea03955d8c70194a),
        UINT64_C(0xa01675b852177a3e),
        UINT64_C(0x69ddb4c3fa3a28a6),
        UINT64_C(0x5e5916068ce265d0),
        UINT64_C(0x7cf3f3f3e8cf2310),
        UINT64_C(0x52f24f0f751900b2),
        UINT64_C(0xa2943280b3cad3c0),
        UINT64_C(0x285f9f5562d33c1d),
        UINT64_C(0x2ec6d5612d917c25),
        UINT64_C(0x7a82a7144ad9ca31),
        UINT64_C(0xbf380491b113b369),
        UINT64_C(0x49cba95a4932159d),
        UINT64_C(0x7217144adf917298),
        UINT64_C(0x5d261ed93e18812a),
        UINT64_C(0xfc6393ee1d8e5d10),
        UINT64_C(0x281763a64b0eb78a),
        UINT64_C(0x26e6ed796ea34cf0),
        UINT64_C(0x9ff363ed8921f884),
        UINT64_C(0x3c515b4af9cfb726),
        UINT64_C(0x68f229d79816a845),
        UINT64_C(0xc57fda65d35d5b76),
        UINT64_C(0x54c528d2ae2907da),
        UINT64_C(0x9ed5a201376913b9),
        UINT64_C(0xe324beaba9ad65a2),
        UINT64_C(0xaa0220bf8e8f9a4a),
        UINT64_C(0xf2b6b097316bd07c),
        UINT64_C(0x21501240f3bfa46a),
        UINT64_C(0x3d04dbf618ccc2bd),
        UINT64_C(0x3e05bb1a1ee0490b),
        UINT64_C(0xb865a76a14b8fffd),
        UINT64_C(0x4e6a157d6b384341),
        UINT64_C(0xcddb786a05e3d22b),
        UINT64_C(0x7ee4b8ecd0a23d7c),
        UINT64_C(0x4217ee5ba8f8627c),
        UINT64_C(0x4db0193f6ed53359),
        UINT64_C(0x72936893078a5ad9),
        UINT64_C(0x869662d411dd0a4e),
        UINT64_C(0x8c58d33352aa2a71),
        UINT64_C(0x2205c20e63988790),
        UINT64_C(0x0ed828ca890dce3c),
        UINT64_C(0xe2b98b0495c2c29c),
        UINT64_C(0xdd13e5287f292b78),
        UINT64_C(0x7b36ed887385fe09),
        UINT64_C(0x1a1961026255a299),
        UINT64_C(0xffd8d32ea3f7649f),
        UINT64_C(0xe73fe1cd2a16c69c),
        UINT64_C(0xa762d7734f0f5cbc),
        UINT64_C(0xf226c9de6fc6a574),
        UINT64_C(0xeabbfd0b7e744f2c),
        UINT64_C(0xd5c529dbc588f9c4),
        UINT64_C(0x12047539bc492729),
        UINT64_C(0x1d3a94b352fc671f),
        UINT64_C(0x145061f4f6e018ad),
        UINT64_C(0x0b33cd8fb452735e),
        UINT64_C(0xafbec3d2208ba045),
        UINT64_C(0xe2c24270a05e8a14),
        UINT64_C(0xa6aaf5b6dc2e4788),
        UINT64_C(0xe1edfa9ec91fc556),
        UINT64_C(0xd8a923502e53d73a),
        UINT64_C(0x4cde159d27143b35),
        UINT64_C(0x4506d0f1f3bf8f6d),
        UINT64_C(0xd461e020a242bb21),
        UINT64_C(0xc5d2d9b05167e958),
        UINT64_C(0xa94f49186de46034),
        UINT64_C(0x5a59ad016deffbde),
        UINT64_C(0x0063f3a9916941ca),
        UINT64_C(0x6610d5bb9d458e5b),
        UINT64_C(0x007691df97133567),
        UINT64_C(0x38473c8c833b1b45),
        UINT64_C(0xe36e48b803a7ee75),
        UINT64_C(0x3a111afbf535dc8d),
    };
    const int tokens[] = {2, 3, 4, 8, 9, 16, 21, 32};
    const int kv_heads[] = {1, 2, 4};
    size_t case_index = 0;
    for (int unit_scale = 0; unit_scale < 2; unit_scale++) {
        float scale = unit_scale ? 1.0f : 1.0f / sqrtf(512.0f);
        for (size_t ti = 0; ti < sizeof(tokens) / sizeof(tokens[0]); ti++) {
            int nt = tokens[ti];
            for (size_t ki = 0; ki < sizeof(kv_heads) / sizeof(kv_heads[0]); ki++) {
                int nk = kv_heads[ki], nh = nk * 8, d = 512, dim = nh * d;
                BnConfig cfg = {0};
                cfg.dim = cfg.hidden_dim = dim; cfg.vocab_size = 32;
                cfg.n_layers = 1; cfg.seq_len = 256;
                cfg.n_heads = nh; cfg.head_size = d; cfg.kv_dim = nk * d;
                cfg.kv_f16 = 1; cfg.rope_theta = 10000.0f;
                assert(init_test_activations(gpu, &cfg) == 0);
                size_t nq = (size_t)nt * dim, nkv = (size_t)nt * nk * d;
                float *q = malloc(nq * sizeof(float));
                float *k = malloc(nkv * sizeof(float));
                float *v = malloc(nkv * sizeof(float));
                float *out = malloc((nq + 2) * sizeof(float));
                assert(q && k && v && out);
                for (int si = 0; si < 2; si++, case_index++) {
                    int seed = si ? 23 : 0;
                    float mag = si ? 3.0f : 1.0f;
                    for (size_t i = 0; i < nq; i++)
                        q[i] = sinf((float)(i * 17 + 3 + seed)) * (1.31f * mag);
                    for (size_t i = 0; i < nkv; i++) {
                        k[i] = bn_fp16_to_fp32(bn_fp32_to_fp16(
                            sinf((float)(i * 11 + 9 + seed)) * (1.7f * mag)));
                        v[i] = bn_fp16_to_fp32(bn_fp32_to_fp16(
                            cosf((float)(i * 7 + 5 + seed)) * (0.73f * mag)));
                    }
                    out[0] = out[nq + 1] = 12345.0f;
                    assert(bn_gpu_backend_prefill_attention(gpu, out + 1,
                        q, k, v, nt, nh, nk, d, 8, nk * d, scale, 0) == 0);
                    assert(cuda_test_float_bits_hash(out + 1, nq) == hashes[case_index]);
                    assert(out[0] == 12345.0f && out[nq + 1] == 12345.0f);
                    memcpy(out + 1, q, nq * sizeof(float));
                    assert(bn_gpu_backend_prefill_attention(gpu, out + 1,
                        out + 1, k, v, nt, nh, nk, d, 8, nk * d, scale, 0) == 0);
                    assert(cuda_test_float_bits_hash(out + 1, nq) == hashes[case_index]);
                    assert(out[0] == 12345.0f && out[nq + 1] == 12345.0f);
                }
                free(out); free(v); free(k); free(q);
                gpu->free_activations(gpu->ctx);
            }
        }
    }
    assert(case_index == sizeof(hashes) / sizeof(hashes[0]));
    printf("CUDA head512 attention reference PASSED\n");
}

static void run_small_attention_reference_case(BnGPUBackend *gpu) {
    /* Independent llama.cpp CUDA flash-attention graphs, reference commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e. F16 KV, F32 Q, causal mask.
     * Cover both query tiles, padded rows, head groups, and input magnitudes. */
    static const int tokens[] = {2, 3, 4, 5, 8, 9, 12, 16};
    static const int seeds[] = {0, 7, 31};
    static const float magnitudes[] = {1.0f, 0.1f, 3.0f};
    static const uint64_t hashes[] = {
        UINT64_C(0xb88c7c04bb7cabc3),
        UINT64_C(0xce1c8f13daa7631f),
        UINT64_C(0xb8b9013b0cac6f78),
        UINT64_C(0x17644f7a5c584236),
        UINT64_C(0x97abad453c165c05),
        UINT64_C(0x9d5d9169e9731430),
        UINT64_C(0x4ba770099b06cf23),
        UINT64_C(0xb92573b1e53b9bbb),
        UINT64_C(0xd03fe3475c1f12c7),
        UINT64_C(0x13cbb54f3802bade),
        UINT64_C(0xa87cbba7fa7bdfe2),
        UINT64_C(0x88434c2faac54e1e),
        UINT64_C(0x96af9e373f842b3c),
        UINT64_C(0xb26fec1890233ab9),
        UINT64_C(0x7fc07d49af8a3f33),
        UINT64_C(0xf4f02c1de933fab0),
        UINT64_C(0xad2f57fba4ca0d7e),
        UINT64_C(0xa6bc80f01d5fa2ad),
        UINT64_C(0x1c3cc06c51daa84e),
        UINT64_C(0xac7f9a2b13c4d451),
        UINT64_C(0x2f396d4f1f11a50d),
        UINT64_C(0x4793b39c55a95077),
        UINT64_C(0x7a2a8b5d7acc63e2),
        UINT64_C(0xf5e4815718209385),
        UINT64_C(0x15419c22a2bf6297),
        UINT64_C(0x53ec8d2de8fa89fc),
        UINT64_C(0xf2ba28999ec14722),
        UINT64_C(0x190a34eea9523cda),
        UINT64_C(0x1922c9b6dae69b7f),
        UINT64_C(0x0f3c4d8e2beeb55f),
        UINT64_C(0x37adab58b89e6d4c),
        UINT64_C(0x92c2dba90439c901),
        UINT64_C(0xb2daa5834b9287df),
        UINT64_C(0xbdeef8951c1ba65f),
        UINT64_C(0x8f2a31c9b01013f4),
        UINT64_C(0xfe9b005371852ba8),
        UINT64_C(0x74e8f460210568fe),
        UINT64_C(0x1e11b0cb5a32aa31),
        UINT64_C(0xe67909eab30bf151),
        UINT64_C(0x3926aa4a7f795443),
        UINT64_C(0x6b35ca8403c8c29d),
        UINT64_C(0xc2e750c1e099ac74),
        UINT64_C(0x787cb029795de180),
        UINT64_C(0x997f49799557c47d),
        UINT64_C(0x28e4f5aaf7bc69ba),
        UINT64_C(0xaca79e0c33c9035c),
        UINT64_C(0xf82e1a997fc4123a),
        UINT64_C(0x09b166a7bc52b2d0),
        UINT64_C(0x31040bceb1757a54),
        UINT64_C(0x711186718aade64d),
        UINT64_C(0xd17ab03342f3599a),
        UINT64_C(0xa7a3a04e33221608),
        UINT64_C(0x3e31b59d2e2d4de8),
        UINT64_C(0x6b48b4d155d19ba4),
        UINT64_C(0xd43e3d8f2c2964c3),
        UINT64_C(0x1803a17588283a7f),
        UINT64_C(0xf49ef853a16cae1e),
        UINT64_C(0x73d20b0bea533864),
        UINT64_C(0xe6da657c3be18cb6),
        UINT64_C(0xa047455e5b342055),
        UINT64_C(0x76b7c29ca84cc858),
        UINT64_C(0x2341daf0b672fb10),
        UINT64_C(0x8c0206df552ea3c1),
        UINT64_C(0x2aba1d39de101c62),
        UINT64_C(0xc691cd34c05cd54b),
        UINT64_C(0x450486ed305e0490),
        UINT64_C(0x715c0efb2b2ac282),
        UINT64_C(0x69fa00a985e07620),
        UINT64_C(0x4ef5a302739c92a3),
        UINT64_C(0x3fbe077a00c11d63),
        UINT64_C(0x965c8c02a2b8f343),
        UINT64_C(0xfab29521907dd24e),
    };
    static const uint64_t projected_hashes[] = {
        UINT64_C(0xd2f72648885fe064),
        UINT64_C(0xd0efc639e1085704),
        UINT64_C(0x6e248e85a915a05a),
        UINT64_C(0x4e53f99d3440e226),
        UINT64_C(0x66ce91f1aaca7d5a),
        UINT64_C(0x69c70d4912c685b9),
        UINT64_C(0xd72631494f1a55a5),
        UINT64_C(0x256e6a0fad7a75a5),
        UINT64_C(0x196662df9b1a95a5),
        UINT64_C(0xbc44dc3b1cea5c45),
        UINT64_C(0xffb058bb61d9fc45),
        UINT64_C(0xd1d51241f9771c45),
        UINT64_C(0xe9647c14270dc825),
        UINT64_C(0x5f71e33ab298c825),
        UINT64_C(0xc8b4e1e3fa520825),
        UINT64_C(0x701e97571ea1dec5),
        UINT64_C(0xd36027767ed43ec5),
        UINT64_C(0xa9fcf736e1239ec5),
        UINT64_C(0xdef06d3b83251aa5),
        UINT64_C(0x782f2822ca2c5aa5),
        UINT64_C(0x03d39ef615d57aa5),
        UINT64_C(0x6dd760f3236b0d25),
        UINT64_C(0x2d5d9588a8ffed25),
        UINT64_C(0xdebd70189cbe6d25),
        UINT64_C(0xd421affdbc1e1a6c),
        UINT64_C(0xb5a1c13b7f4f192c),
        UINT64_C(0x597749de69945321),
        UINT64_C(0xd14c1a146ecd08f7),
        UINT64_C(0xcb5edab0d779ddda),
        UINT64_C(0x2286981ce4afcbec),
        UINT64_C(0xf27a95620aa315a5),
        UINT64_C(0xa0b66f172c2075a5),
        UINT64_C(0x4d23eb3c59a1f5a5),
        UINT64_C(0x403cacb66a775c45),
        UINT64_C(0xcd5ed70862f41c45),
        UINT64_C(0xd13f42232330bc45),
        UINT64_C(0xb14027e85c3a6825),
        UINT64_C(0x151a677d96cf8825),
        UINT64_C(0xbcf368757d016825),
        UINT64_C(0x9c1ae50085e61ec5),
        UINT64_C(0x543c7643f1d15ec5),
        UINT64_C(0xbff9b2bff6733ec5),
        UINT64_C(0xae2fed9a30451aa5),
        UINT64_C(0x65e97c61b49a9aa5),
        UINT64_C(0x31c5b1b581f8baa5),
        UINT64_C(0xacafd0fdd69c0d25),
        UINT64_C(0x21764fa773a6ad25),
        UINT64_C(0x95e1c1c84e434d25),
        UINT64_C(0x3d6dfc11a737aedb),
        UINT64_C(0x8557724da9eda8fa),
        UINT64_C(0x00d9a129f7f19e2c),
        UINT64_C(0xea060936eb334138),
        UINT64_C(0xce93531990996665),
        UINT64_C(0xbab56cdbadff25cb),
        UINT64_C(0x9b979cf322f155a5),
        UINT64_C(0x4bcebb17568815a5),
        UINT64_C(0x08e0949d7c9c95a5),
        UINT64_C(0xdee2d8d04d505c45),
        UINT64_C(0xbc0e3e51bfe41c45),
        UINT64_C(0x0b2a0ce8068b1c45),
        UINT64_C(0xced70f877727e825),
        UINT64_C(0x08d2d206e50e0825),
        UINT64_C(0xdd953b2c7da8c825),
        UINT64_C(0x042363fc0703bec5),
        UINT64_C(0x9d08a6c06ca77ec5),
        UINT64_C(0x2686b4d6a4c1bec5),
        UINT64_C(0xce7ff8d6f0159aa5),
        UINT64_C(0x3cafe4ccaeb07aa5),
        UINT64_C(0x7b5d97e670e23aa5),
        UINT64_C(0xc2e248f9373f0d25),
        UINT64_C(0x2a0f25e8aba38d25),
        UINT64_C(0x263a39095c5d8d25),
    };
    int case_index = 0;
    for (int nk = 1; nk <= 4; nk *= 2) {
        int nh = nk * 8, d = 256, dim = nh * d;
        BnConfig cfg = {0};
        cfg.dim = cfg.hidden_dim = dim; cfg.vocab_size = 32;
        cfg.n_layers = 1; cfg.seq_len = 256;
        cfg.n_heads = nh; cfg.head_size = d; cfg.kv_dim = nk * d;
        cfg.kv_f16 = 1; cfg.rope_theta = 10000.0f;
        assert(init_test_activations(gpu, &cfg) == 0);
        for (int ti = 0; ti < 8; ti++) {
            int nt = tokens[ti];
            size_t nq = (size_t)nt * dim, nkv = (size_t)nt * nk * d;
            float *q = (float *)malloc(nq * sizeof(float));
            float *k = (float *)malloc(nkv * sizeof(float));
            float *v = (float *)malloc(nkv * sizeof(float));
            float *out = (float *)malloc((nq + 2) * sizeof(float));
            assert(q && k && v && out);
            for (int si = 0; si < 3; si++, case_index++) {
                int seed = seeds[si]; float mag = magnitudes[si];
                for (size_t i = 0; i < nq; i++)
                    q[i] = sinf((float)(i * 17 + 3 + seed)) * (1.31f * mag);
                for (size_t i = 0; i < nkv; i++) {
                    k[i] = bn_fp16_to_fp32(bn_fp32_to_fp16(
                        sinf((float)(i * 11 + 9 + seed)) * (1.7f * mag)));
                    v[i] = bn_fp16_to_fp32(bn_fp32_to_fp16(
                        cosf((float)(i * 7 + 5 + seed)) * (0.73f * mag)));
                }
                out[0] = out[nq + 1] = 12345.0f;
                assert(bn_gpu_backend_prefill_attention(gpu, out + 1, q, k, v,
                    nt, nh, nk, d, 8, nk * d, 1.0f / sqrtf((float)d), 0) == 0);
                uint64_t hash = cuda_test_float_bits_hash(out + 1, nq);
                if (hash != hashes[case_index])
                    fprintf(stderr, "small attention case=%d hash=%llx expected=%llx\n",
                        case_index, (unsigned long long)hash,
                        (unsigned long long)hashes[case_index]);
                assert(hash == hashes[case_index]);
                assert(out[0] == 12345.0f && out[nq + 1] == 12345.0f);

                /* Compare the fused route to an independent reference graph.
                 * Even one-hot projections retain matrix-path rounding. */
                enum { rows = 8 };
                float *weight = (float *)calloc((size_t)rows * dim, sizeof(float));
                float projected[16 * rows + 2];
                assert(weight);
                for (int r = 0; r < rows; r++)
                    weight[(size_t)r * dim + (r * 257) % dim] = 1.0f;
                void *wb = gpu->buffer_create(gpu->ctx, weight,
                    (size_t)rows * dim * sizeof(float), BN_GGUF_TENSOR_F32, rows, dim);
                assert(wb);
                projected[0] = projected[nt * rows + 1] = 12345.0f;
                assert(gpu->prefill_attention_wo(gpu->ctx, projected + 1, wb,
                    q, k, v, nt, nh, nk, d, 8, nk * d, rows, dim,
                    BN_GGUF_TENSOR_F32, 1.0f / sqrtf((float)d), 0) == 0);
                assert(cuda_test_float_bits_hash(projected + 1, (size_t)nt * rows) ==
                       projected_hashes[case_index]);
                assert(projected[0] == 12345.0f &&
                       projected[nt * rows + 1] == 12345.0f);
                gpu->buffer_destroy(gpu->ctx, wb); free(weight);
            }
            free(out); free(v); free(k); free(q);
        }
    }
    assert(case_index == 72);
    printf("CUDA small attention reference PASSED\n");
}

static void run_split_attention_reference_case(BnGPUBackend *gpu) {
    /* Independent llama.cpp CUDA flash-attention graphs, commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e, SM120. These cross the
     * 32-key boundary and require independent FP16 partials + F32 fixup. */
    static const int tokens[] = {33, 40, 51, 60, 64};
    static const int seeds[] = {0, 23};
    static const float magnitudes[] = {1.0f, 3.0f};
    static const uint64_t hashes[] = {
        UINT64_C(0x2f75651cddc9045d),
        UINT64_C(0x2605d2d6ff5e820f),
        UINT64_C(0xc363cabd077e459f),
        UINT64_C(0xb8a57ba06459a19b),
        UINT64_C(0x746320360a3c3416),
        UINT64_C(0x7c1536361cfd2986),
        UINT64_C(0xbbaac99e882e8318),
        UINT64_C(0xeb52f6091e8f72c8),
        UINT64_C(0x2d5b553ff40df4b1),
        UINT64_C(0x8252981d495bbed6),
        UINT64_C(0x98a5ee1528b20b5d),
        UINT64_C(0xc4622f011db59127),
        UINT64_C(0x9dcae6a2087cbb31),
        UINT64_C(0xfb1f475a7908a8f3),
        UINT64_C(0x8f62f923859bd05c),
        UINT64_C(0x457e1edf09326ca2),
        UINT64_C(0xcab17262dfc54165),
        UINT64_C(0x08944d6f88aef057),
        UINT64_C(0xffc1fd58dbc21bc2),
        UINT64_C(0x9be1162bc7e5f3b4),
        UINT64_C(0xe6c9e628a2c01043),
        UINT64_C(0xda825abe6502d658),
        UINT64_C(0xc40f70f158ba735a),
        UINT64_C(0xb216f3fb7a4ecd3d),
        UINT64_C(0x501c65239fa0406e),
        UINT64_C(0xacef971deee76745),
        UINT64_C(0xada102fd40df960c),
        UINT64_C(0xb59f1e8aff2f3623),
        UINT64_C(0xee1ec097ab4883d4),
        UINT64_C(0xe5e93ad8fad7922f),
    };
    int case_index = 0;
    for (int ti = 0; ti < 5; ti++) {
        int nt = tokens[ti];
        for (int nk = 1; nk <= 4; nk *= 2) {
            int nh = nk * 8, d = 256, dim = nh * d;
            BnConfig cfg = {0};
            cfg.dim = cfg.hidden_dim = dim; cfg.vocab_size = 32;
            cfg.n_layers = 1; cfg.seq_len = 256;
            cfg.n_heads = nh; cfg.head_size = d; cfg.kv_dim = nk * d;
            cfg.kv_f16 = 1; cfg.rope_theta = 10000.0f;
            assert(init_test_activations(gpu, &cfg) == 0);
            size_t nq = (size_t)nt * dim, nkv = (size_t)nt * nk * d;
            float *q = (float *)malloc((nq + 2) * sizeof(float));
            float *k = (float *)malloc(nkv * sizeof(float));
            float *v = (float *)malloc(nkv * sizeof(float));
            float *out = (float *)malloc((nq + 2) * sizeof(float));
            assert(q && k && v && out);
            for (int si = 0; si < 2; si++, case_index++) {
                int seed = seeds[si]; float mag = magnitudes[si];
                for (size_t i = 0; i < nq; i++)
                    q[i + 1] = sinf((float)(i * 17 + 3 + seed)) * (1.31f * mag);
                for (size_t i = 0; i < nkv; i++) {
                    k[i] = bn_fp16_to_fp32(bn_fp32_to_fp16(
                        sinf((float)(i * 11 + 9 + seed)) * (1.7f * mag)));
                    v[i] = bn_fp16_to_fp32(bn_fp32_to_fp16(
                        cosf((float)(i * 7 + 5 + seed)) * (0.73f * mag)));
                }
                out[0] = out[nq + 1] = q[0] = q[nq + 1] = 12345.0f;
                assert(bn_gpu_backend_prefill_attention(gpu, out + 1, q + 1, k, v,
                    nt, nh, nk, d, 8, nk * d, 1.0f / sqrtf((float)d), 0) == 0);
                assert(cuda_test_float_bits_hash(out + 1, nq) == hashes[case_index]);
                assert(out[0] == 12345.0f && out[nq + 1] == 12345.0f);
                assert(bn_gpu_backend_prefill_attention(gpu, q + 1, q + 1, k, v,
                    nt, nh, nk, d, 8, nk * d, 1.0f / sqrtf((float)d), 0) == 0);
                assert(cuda_test_float_bits_hash(q + 1, nq) == hashes[case_index]);
                assert(q[0] == 12345.0f && q[nq + 1] == 12345.0f);
            }
            free(out); free(v); free(k); free(q);
        }
    }
    assert(case_index == 30);
    printf("CUDA split attention reference PASSED\n");
}

static void *ssm_gateup_test_buffer(BnGPUBackend *gpu, int rows, int cols, float fill) {
    size_t n=(size_t)rows*cols; float *x=malloc(n*sizeof(float)); assert(x);
    for(size_t i=0;i<n;i++)x[i]=fill;
    void *b=gpu->buffer_create(gpu->ctx,x,n*sizeof(float),0,rows,cols);free(x);assert(b);return b;
}
/* Independent llama.cpp CUDA graph, commit
 * 3d3d7c81813067fc8c185da017e0af03b4269b1e, Blackwell. Zero Q/K/V with
 * nonzero initial state isolates alpha projection/decay arithmetic. Output
 * residuals alone would conceal this error. These cases do not test beta
 * sensitivity or quantized QKV/Z arithmetic. */
static void run_ssm_projection_state_reference_case(BnGPUBackend *gpu) {
    const struct { int dim, heads, tokens; uint64_t hash; } cases[] = {
        {256, 4, 2, UINT64_C(0x2d4b66d808e7ff54)},
        {256, 4, 9, UINT64_C(0xc46416fcdcf65ac5)},
        {2048, 32, 3, UINT64_C(0x077a57d287bcf73e)},
        {2048, 32, 8, UINT64_C(0x6055b3012bde8cd3)},
        {5120, 48, 2, UINT64_C(0xacd557486f146349)},
        {5120, 48, 3, UINT64_C(0xc2942c2802bbd87f)},
        {5120, 48, 8, UINT64_C(0x38b42b01221d6170)},
        {5120, 48, 9, UINT64_C(0xa519c61c30feddee)},
        {5120, 48, 16, UINT64_C(0x1bd80e17afd4a935)},
        {5120, 64, 8, UINT64_C(0x55c7657312cfdf2c)},
        {5120, 64, 17, UINT64_C(0x858af8792b7d5ded)},
        {8192, 64, 9, UINT64_C(0x49cf1a641e3b27f4)},
    };
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int dim = cases[c].dim, hv = cases[c].heads, nt = cases[c].tokens;
        int h = 128, inner = h * hv, qkv = 2 * h + inner;
        size_t ns = (size_t)h * inner, nw = (size_t)dim * hv;
        float *x = malloc((size_t)dim * nt * sizeof(float));
        float *ab = malloc(2 * nw * sizeof(float));
        float *bias = malloc((size_t)hv * sizeof(float));
        float *al = malloc((size_t)hv * sizeof(float));
        float *state = malloc(ns * sizeof(float));
        float *got = malloc((ns + 2) * sizeof(float));
        float *out = malloc((size_t)dim * nt * sizeof(float));
        float *zeros = calloc((size_t)qkv * 3, sizeof(float));
        assert(x && ab && bias && al && state && got && out && zeros);
        for (int t = 0; t < nt; t++)
            for (int d = 0; d < dim; d++)
                x[(size_t)t * dim + d] = (float)((d * 37 + t * 19) % 251 - 125) / 259.0f;
        for (int r = 0; r < hv; r++) {
            bias[r] = (float)(r % 7 - 3) / 64.0f;
            al[r] = -0.1f - (float)(r % 13) / 100.0f;
            for (int d = 0; d < dim; d++) {
                ab[(size_t)r * dim + d] = (float)((r * 13 + d * 17) % 257 - 128) / 257.0f;
                ab[nw + (size_t)r * dim + d] = (float)((r * 11 + d * 19) % 251 - 125) / 259.0f;
            }
        }
        for (size_t i = 0; i < ns; i++)
            state[i] = (float)((int)((i * 13) % 257) - 128) / 4096.0f;
        BnGPUActivationPlan plan = {0};
        float rope = 1;
        plan.dim = plan.xb2_elements = dim; plan.hb_elements = inner;
        plan.vocab_size = 32; plan.n_layers = plan.n_heads = 1;
        plan.seq_len = nt + 1; plan.head_size = plan.kv_dim = 2;
        plan.rope_frequencies = &rope; plan.rope_frequency_count = 1;
        plan.uses_hybrid_ssm = plan.ssm_layer_count = 1;
        plan.ssm_state_size = h; plan.ssm_inner_size = inner;
        plan.ssm_time_step_rank = hv; plan.ssm_group_count = 1;
        plan.ssm_conv_kernel = 4;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        void *buffers[] = {
            ssm_gateup_test_buffer(gpu, qkv, dim, 0),
            ssm_gateup_test_buffer(gpu, inner, dim, 0),
            gpu->buffer_create(gpu->ctx, ab, nw * sizeof(float), BN_GGUF_TENSOR_F32, hv, dim),
            gpu->buffer_create(gpu->ctx, ab + nw, nw * sizeof(float), BN_GGUF_TENSOR_F32, hv, dim),
            ssm_gateup_test_buffer(gpu, dim, inner, 0),
            ssm_gateup_test_buffer(gpu, dim, 1, 1),
            ssm_gateup_test_buffer(gpu, qkv, 4, 0),
            gpu->buffer_create(gpu->ctx, bias, (size_t)hv * sizeof(float), BN_GGUF_TENSOR_F32, hv, 1),
            gpu->buffer_create(gpu->ctx, al, (size_t)hv * sizeof(float), BN_GGUF_TENSOR_F32, hv, 1),
            ssm_gateup_test_buffer(gpu, h, 1, 1),
            gpu->buffer_create(gpu->ctx, ab, 2 * nw * sizeof(float), BN_GGUF_TENSOR_F32, 2 * hv, dim)
        };
        for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++) assert(buffers[i]);
        for (int stacked = 0; stacked < 2; stacked++) {
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_STATE,
                state, ns * sizeof(float), 0) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_CONV_STATE,
                zeros, (size_t)qkv * 3 * sizeof(float), 0) == 0);
            int did_ffn = 0;
            assert(gpu->prefill_ssm_layer(gpu->ctx, out,
                buffers[0], buffers[1], buffers[2], buffers[3], NULL,
                stacked ? buffers[10] : NULL, buffers[4], buffers[5],
                buffers[6], buffers[7], buffers[8], buffers[9],
                NULL, NULL, NULL, NULL, x, nt, dim, qkv, inner,
                1, h, hv, h, 4, 0,
                BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32,
                BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32, 0, 0, 0, 0,
                0, 0, 1e-6f, &did_ffn) == 0);
            assert(!did_ffn);
            got[0] = got[ns + 1] = -12345.0f;
            assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_SSM_STATE,
                got + 1, ns * sizeof(float), 0) == 0);
            assert(cuda_test_float_bits_hash(got + 1, ns) == cases[c].hash);
            assert(got[0] == -12345.0f && got[ns + 1] == -12345.0f);
        }
        for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++)
            gpu->buffer_destroy(gpu->ctx, buffers[i]);
        gpu->free_activations(gpu->ctx);
        free(zeros); free(out); free(got); free(state); free(al); free(bias); free(ab); free(x);
    }
    printf("CUDA SSM projection state reference PASSED\n");
}

/* Zero SSM output isolates its fused FFN. Stacked storage must preserve
 * the same logical projections as separate storage. This shape crosses
 * a reference MMQ scheduling boundary (the old path differed in 42887
 * of 46080 output words). No real model is needed. */
static void run_ssm_gateup_layout_type(BnGPUBackend *gpu, int type) {
    enum {d=5120,hid=17408,nt=9,h=128,qkv=384};
    BnGPUActivationPlan plan={0}; float rope=1;
    plan.dim=d;plan.vocab_size=32;plan.hb_elements=hid;plan.xb2_elements=d;
    plan.n_layers=plan.n_heads=1;plan.seq_len=32;plan.kv_dim=plan.head_size=2;
    plan.rope_frequencies=&rope;plan.rope_frequency_count=1;
    plan.uses_hybrid_ssm=plan.ssm_layer_count=1;plan.ssm_time_step_rank=1;
    plan.ssm_state_size=h;plan.ssm_inner_size=h;plan.ssm_group_count=1;plan.ssm_conv_kernel=4;
    assert(gpu->init_activations(gpu->ctx,&plan)==0);
    /* Session state is caller-initialized; reused allocations are not zeroed. */
    float *state_zero = calloc((size_t)h * h, sizeof(float));
    assert(state_zero);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_STATE,
        state_zero, (size_t)h * h * sizeof(float), 0) == 0);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_CONV_STATE,
        state_zero, (size_t)qkv * 3 * sizeof(float), 0) == 0);
    free(state_zero);
    void *q=ssm_gateup_test_buffer(gpu,qkv,d,0),*z=ssm_gateup_test_buffer(gpu,h,d,0),*ab=ssm_gateup_test_buffer(gpu,1,d,0);
    void *so=ssm_gateup_test_buffer(gpu,d,h,0),*norm=ssm_gateup_test_buffer(gpu,d,1,1),*conv=ssm_gateup_test_buffer(gpu,qkv,4,0);
    void *bias=ssm_gateup_test_buffer(gpu,1,1,0),*sn=ssm_gateup_test_buffer(gpu,h,1,1);
    size_t blocks=(size_t)hid*d/BN_QK_K;
    size_t bytes = type == BN_GGUF_TENSOR_Q5_K
        ? blocks * kquant_fixture_block_size(type)
        : signed_mmq_fixture_bytes(type, blocks * BN_QK_K);
    void *raw=malloc(bytes*2);assert(raw);
    if (type == BN_GGUF_TENSOR_Q5_K) kquant_fixture_weights(raw, type, blocks * 2);
    else signed_mmq_fixture_weights(raw, type, blocks * BN_QK_K * 2, 23);
    void *stack=gpu->buffer_create(gpu->ctx,raw,bytes*2,type,hid*2,d);
    void *gate=gpu->buffer_create(gpu->ctx,raw,bytes,type,hid,d);
    void *up=gpu->buffer_create(gpu->ctx,(char*)raw+bytes,bytes,type,hid,d);free(raw);
    bytes=blocks*kquant_fixture_block_size(BN_GGUF_TENSOR_Q6_K);raw=malloc(bytes);assert(raw);
    kquant_fixture_weights(raw,BN_GGUF_TENSOR_Q6_K,blocks);
    void *down=gpu->buffer_create(gpu->ctx,raw,bytes,BN_GGUF_TENSOR_Q6_K,d,hid);free(raw);
    assert(stack&&gate&&up&&down);size_t n=(size_t)d*nt;
    float *x=malloc(n*sizeof(float)), *guard=malloc((n+2)*sizeof(float));
    float *ref=malloc(n*sizeof(float)); assert(x && guard && ref);
    float *out=guard+1;
    for(size_t i=0;i<n;i++)x[i]=(float)((int)((i*17)%257)-128)/128.0f;
    /* A preceding graph selects the nondefault execution stream. */
    BnGPUOp graph[32] = {0};
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, x,
        sizeof(float), 0) == 0);
    for (int i = 0; i < 32; i++) {
        graph[i].op_code = BN_GPU_CODE_COPY;
        graph[i].buf_in = BN_GPU_VALUE_X;
        graph[i].buf_out = BN_GPU_VALUE_SCRATCH;
        graph[i].p[2] = 1;
    }
    assert(gpu->execute(gpu->ctx, graph, 32, -1, NULL, 0) == 0);
    for(int variant=0;variant<4;variant++) {
        int stacked=variant&1, alias=variant>=2;
        guard[0]=guard[n+1]=-12345.0f;
        if(alias)memcpy(out,x,n*sizeof(float));
        int did=0;
        assert(gpu->prefill_ssm_layer(gpu->ctx,out,q,z,ab,ab,NULL,NULL,so,norm,conv,bias,bias,sn,
            stacked?stack:gate,stacked?NULL:up,down,norm,alias?out:x,nt,d,qkv,h,1,h,1,h,4,0,
            0,0,0,0,0,hid,type,type,BN_GGUF_TENSOR_Q6_K,
            BN_MODEL_ACTIVATION_SILU,0,1e-6f,&did)==0);assert(did);
        if(!variant)memcpy(ref,out,n*sizeof(float));
        else {
            if (memcmp(ref, out, n * sizeof(float)) != 0) {
                size_t different = 0, nonfinite = 0;
                for (size_t i = 0; i < n; i++) {
                    different += memcmp(ref + i, out + i, sizeof(float)) != 0;
                    nonfinite += !isfinite(ref[i]) || !isfinite(out[i]);
                }
                fprintf(stderr, "SSM layout type=%d variant=%d different=%zu nonfinite=%zu ref=%016llx out=%016llx\n",
                    type, variant, different, nonfinite,
                    (unsigned long long)cuda_test_float_bits_hash(ref, n),
                    (unsigned long long)cuda_test_float_bits_hash(out, n));
            }
            assert(memcmp(ref, out, n * sizeof(float)) == 0);
        }
        assert(guard[0]==-12345.0f && guard[n+1]==-12345.0f);
    }
    free(x);free(guard);free(ref);
    void *bufs[]={q,z,ab,so,norm,conv,bias,sn,stack,gate,up,down};for(size_t i=0;i<sizeof(bufs)/sizeof(bufs[0]);i++)gpu->buffer_destroy(gpu->ctx,bufs[i]);
    gpu->free_activations(gpu->ctx);
    printf("CUDA SSM gate/up layout type=%d PASSED\n", type);
}

static void run_ssm_gateup_layout_case(BnGPUBackend *gpu) {
    const int types[] = { BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q3_K,
        BN_GGUF_TENSOR_IQ4_NL, BN_GGUF_TENSOR_IQ4_XS };
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++)
        run_ssm_gateup_layout_type(gpu, types[i]);
}

static void run_ssm_conv_reference_case(BnGPUBackend *gpu) {
    enum { n = 512, nt = 4, kern = 4, steps = nt + kern - 1 };
    float raw[steps * n], weight[kern * n], out[nt * n], state[(kern - 1) * n];
    for (int i = 0; i < steps * n; i++) raw[i] = (float)((i * 13) % 97 - 48) / 16.0f;
    for (int i = 0; i < kern * n; i++) weight[i] = (float)((i * 7) % 31 - 15) / 32.0f;
    BnGPUActivationPlan plan = {0};
    float rope = 1.0f;
    plan.dim = plan.vocab_size = plan.hb_elements = plan.xb2_elements = n;
    plan.n_layers = plan.n_heads = 1; plan.seq_len = nt;
    plan.kv_dim = plan.head_size = 2;
    plan.rope_frequencies = &rope; plan.rope_frequency_count = 1;
    plan.uses_hybrid_ssm = plan.ssm_layer_count = 1;
    plan.ssm_time_step_rank = 2; plan.ssm_state_size = 128;
    plan.ssm_inner_size = 256; plan.ssm_group_count = 1; plan.ssm_conv_kernel = kern;
    assert(gpu->init_activations(gpu->ctx, &plan) == 0);
    void *w = gpu->buffer_create(gpu->ctx, weight, sizeof(weight), BN_GGUF_TENSOR_F32, n, kern);
    assert(w);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_CONV_STATE,
                                 raw, sizeof(state), 0) == 0);
    for (int t = 0; t < nt; t++) {
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_QKV,
                                     raw + (t + kern - 1) * n, n * sizeof(float), 0) == 0);
        BnGPUOp op = {0};
        op.op_code = BN_GPU_CODE_SSM_CONV_SILU; op.W_buf = w;
        op.buf_in = BN_GPU_VALUE_SSM_QKV; op.p[0] = n; op.p[1] = kern;
        assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_SSM_QKV, out + t * n, n) == 0);
        assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_SSM_CONV_STATE,
                                    state, sizeof(state), 0) == 0);
        assert(memcmp(state, raw + (t + 1) * n, sizeof(state)) == 0);
    }
    /* ggml_ssm_conv + SiLU, llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e,
       CUDA 13.2 sm_120a -use_fast_math. Hash covers every output value. */
    assert(cuda_test_float_bits_hash(out, nt * n) == UINT64_C(0x52a0c70a2caf49c1));
    gpu->buffer_destroy(gpu->ctx, w);
    gpu->free_activations(gpu->ctx);
}

static void run_ssm_delta_reference_case(BnGPUBackend *gpu) {
    enum { h = 128, hk = 2, hv = 4, nt = 4, n = h * hv, ns = h * n };
    /* llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e,
       ggml_gated_delta_net, CUDA 13.2 sm_120a -use_fast_math.
       g[t,h] = -0.125 * (1+h); decay below is reference CUDA exp(g).
       Expected hashes cover all four outputs and the entire final state. */
    const float decay[hv] = {0x1.c3d6a2p-1f, 0x1.8ebefap-1f,
                              0x1.5fe460p-1f, 0x1.368b30p-1f};
    float q[h * hk], k[h * hk], v[n], beta[hv];
    float *state = malloc((size_t)ns * sizeof(float));
    float *out = malloc((size_t)nt * n * sizeof(float));
    assert(state && out);
    for (int i = 0; i < ns; i++) state[i] = (float)((i * 7) % 101 - 50) / 512.0f;
    for (int i = 0; i < hv; i++) beta[i] = 0.25f + 0.125f * i;
    BnGPUActivationPlan plan = {0};
    float rope = 1.0f;
    plan.dim = plan.vocab_size = plan.hb_elements = plan.xb2_elements = n;
    plan.n_layers = plan.n_heads = 1; plan.seq_len = nt;
    plan.kv_dim = plan.head_size = 2;
    plan.rope_frequencies = &rope; plan.rope_frequency_count = 1;
    plan.uses_hybrid_ssm = plan.ssm_layer_count = 1;
    plan.ssm_time_step_rank = hv; plan.ssm_state_size = h;
    plan.ssm_inner_size = n; plan.ssm_group_count = hk; plan.ssm_conv_kernel = 4;
    assert(gpu->init_activations(gpu->ctx, &plan) == 0);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_STATE,
                                 state, (size_t)ns * sizeof(float), 0) == 0);
    for (int t = 0; t < nt; t++) {
        for (int j = 0; j < h * hk; j++) {
            int i = t * h * hk + j;
            q[j] = (float)((i * 13) % 97 - 48) / 512.0f;
            k[j] = (float)((i * 17) % 89 - 44) / 512.0f;
        }
        for (int j = 0; j < n; j++) v[j] = (float)(((t * n + j) * 19) % 83 - 41) / 32.0f;
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, q, sizeof(q), 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB, k, sizeof(k), 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_V, v, sizeof(v), 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_ALPHA, decay, sizeof(decay), 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_BETA, beta, sizeof(beta), 0) == 0);
        BnGPUOp op = {0};
        op.op_code = BN_GPU_CODE_SSM_DELTA; op.buf_in = BN_GPU_VALUE_XB;
        op.buf_aux = BN_GPU_VALUE_HB; op.buf_out = BN_GPU_VALUE_LOGITS;
        op.rows = hv; op.p[0] = op.p[1] = h; op.p[2] = hk;
        op.p[3] = f32_bits(0x1.6a09e6p-4f);
        assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_LOGITS, out + t * n, n) == 0);
    }
    assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_SSM_STATE,
                                state, (size_t)ns * sizeof(float), 0) == 0);
    assert(cuda_test_float_bits_hash(out, nt * n) == UINT64_C(0x3715bfb3e57a6742));
    assert(cuda_test_float_bits_hash(state, ns) == UINT64_C(0x8ba6c6b976be5454));
    gpu->free_activations(gpu->ctx);
    free(state); free(out);
}

static void run_ssm_l2_epsilon_case(BnGPUBackend *gpu) {
    const int widths[] = {128, 1024};
    const float epsilons[] = {1e-8f, 1e-6f, 1e-3f};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        int h = widths[wi], n = 4 * h;
        BnGPUActivationPlan plan = {0};
        float rope = 1.0f;
        plan.dim = plan.vocab_size = plan.hb_elements = plan.xb2_elements = n;
        plan.n_layers = plan.n_heads = 1; plan.seq_len = 2;
        plan.kv_dim = plan.head_size = 2;
        plan.rope_frequencies = &rope; plan.rope_frequency_count = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        float *q = calloc((size_t)n, sizeof(float));
        float *k = calloc((size_t)n, sizeof(float));
        float *out = malloc((size_t)n * sizeof(float));
        assert(q && k && out);
        for (size_t ei = 0; ei < sizeof(epsilons) / sizeof(epsilons[0]); ei++) {
            float eps = epsilons[ei];
            memset(q, 0, (size_t)n * sizeof(float));
            q[h] = 0.5f * eps; q[2 * h] = 2.0f * eps; q[3 * h] = -0.5f;
            for (int i = 0; i < n; i++) k[i] = -q[i];
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, q,
                                         (size_t)n * sizeof(float), 0) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB, k,
                                         (size_t)n * sizeof(float), 0) == 0);
            BnGPUOp op = {0};
            op.op_code = BN_GPU_CODE_SSM_L2NORM;
            op.buf_in = BN_GPU_VALUE_XB; op.buf_aux = BN_GPU_VALUE_HB;
            op.rows = 4; op.p[0] = h; op.p[3] = f32_bits(eps);
            assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_XB, out, n) == 0);
            for (int which = 0; which < 2; which++) {
                if (which)
                    assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_HB, out,
                                                (size_t)n * sizeof(float), 0) == 0);
                for (int i = 0; i < n; i++) {
                    float expected = i == h ? 0.5f : i == 2 * h ? 1.0f : i == 3 * h ? -1.0f : 0.0f;
                    if (which) expected = -expected;
                    assert(isfinite(out[i]));
                    assert(fabsf(out[i] - expected) < 1e-6f);
                }
            }
        }
        free(q); free(k); free(out);
        gpu->free_activations(gpu->ctx);
    }
}

static void run_ssm_gate_composition_case(BnGPUBackend *gpu) {
    const int widths[] = {32, 64, 128, 256, 512, 1024, 2048};
    for (size_t c = 0; c < sizeof(widths) / sizeof(widths[0]); c++) {
        int n = widths[c];
        BnGPUActivationPlan plan = {0};
        float rope = 1.0f;
        plan.dim = plan.vocab_size = plan.hb_elements = plan.xb2_elements = n;
        plan.n_layers = plan.n_heads = 1; plan.seq_len = 2;
        plan.kv_dim = plan.head_size = 2;
        plan.rope_frequencies = &rope; plan.rope_frequency_count = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        float *x = malloc((size_t)n * sizeof(float));
        float *z = malloc((size_t)n * sizeof(float));
        float *w = malloc((size_t)n * sizeof(float));
        float *reference = malloc((size_t)n * sizeof(float));
        float *out = malloc((size_t)n * sizeof(float));
        assert(x && z && w && reference && out);
        for (int i = 0; i < n; i++) {
            x[i] = sinf((float)i * 0.17f) * 3.0f;
            z[i] = -12.0f + 24.0f * i / (n - 1);
            w[i] = 0.8f + 0.01f * (i % 23);
        }
        void *weight = gpu->buffer_create(gpu->ctx, w, (size_t)n * sizeof(float),
                                           BN_GGUF_TENSOR_F32, 1, n);
        assert(weight);
        for (int sigmoid = 0; sigmoid < 2; sigmoid++) {
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, x,
                                         (size_t)n * sizeof(float), 0) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB, z,
                                         (size_t)n * sizeof(float), 0) == 0);
            BnGPUOp ops[2] = {{0}};
            ops[0].op_code = BN_GPU_CODE_RMSNORM; ops[0].W_buf = weight;
            ops[0].buf_in = ops[0].buf_out = BN_GPU_VALUE_XB;
            ops[0].p[0] = n; ops[0].p[1] = f32_bits(1e-6f);
            ops[1].op_code = sigmoid ? BN_GPU_CODE_SIGMOID_GATE : BN_GPU_CODE_SILU_GATE;
            ops[1].buf_in = BN_GPU_VALUE_HB; ops[1].buf_aux = BN_GPU_VALUE_XB;
            ops[1].buf_out = BN_GPU_VALUE_HB; ops[1].p[0] = n;
            assert(gpu->execute(gpu->ctx, ops, 2, BN_GPU_VALUE_HB, reference, n) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, x,
                                         (size_t)n * sizeof(float), 0) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB, z,
                                         (size_t)n * sizeof(float), 0) == 0);
            BnGPUOp gate = {0};
            gate.op_code = BN_GPU_CODE_SSM_GATE; gate.W_buf = weight;
            gate.buf_in = BN_GPU_VALUE_XB; gate.buf_aux = BN_GPU_VALUE_HB;
            gate.rows = 1; gate.p[0] = n; gate.p[1] = f32_bits(1e-6f);
            gate.p[2] = sigmoid;
            assert(gpu->execute(gpu->ctx, &gate, 1, BN_GPU_VALUE_XB, out, n) == 0);
            assert(memcmp(out, reference, (size_t)n * sizeof(float)) == 0);
        }
        gpu->buffer_destroy(gpu->ctx, weight);
        free(x); free(z); free(w); free(reference); free(out);
        gpu->free_activations(gpu->ctx);
    }
}

static void run_ssm_alpha_beta_reference_case(BnGPUBackend *gpu) {
    enum { n = 128 };
    BnGPUActivationPlan plan = {0};
    float rope = 1.0f;
    plan.dim = plan.vocab_size = plan.hb_elements = plan.xb2_elements = 256;
    plan.n_layers = 1; plan.seq_len = 2;
    plan.kv_dim = plan.head_size = 2; plan.n_heads = 1;
    plan.rope_frequencies = &rope; plan.rope_frequency_count = 1;
    plan.uses_hybrid_ssm = plan.ssm_layer_count = 1;
    plan.ssm_time_step_rank = n; plan.ssm_state_size = 2;
    plan.ssm_inner_size = n * 2; plan.ssm_group_count = 1;
    plan.ssm_conv_kernel = 4;
    assert(gpu->init_activations(gpu->ctx, &plan) == 0);
    float input[2 * n], bias[n], decay[n], output[n];
    for (int i = 0; i < n; i++) {
        input[i] = -20.0f + 40.0f * i / (n - 1);
        input[n + i] = -12.0f + 24.0f * i / (n - 1);
        bias[i] = 0.1f * (i % 7 - 3);
        decay[i] = -0.1f - 0.01f * (i % 19);
    }
    void *bias_buf = gpu->buffer_create(gpu->ctx, bias, sizeof(bias),
                                        BN_GGUF_TENSOR_F32, n, 1);
    void *decay_buf = gpu->buffer_create(gpu->ctx, decay, sizeof(decay),
                                         BN_GGUF_TENSOR_F32, n, 1);
    assert(bias_buf && decay_buf);
    for (int split = 0; split < 2; split++) {
        for (int beta = 0; beta < 2; beta++) {
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB,
                                         input, sizeof(input), 0) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_ALPHA,
                                         input, n * sizeof(float), 0) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_SSM_BETA,
                                         input + n, n * sizeof(float), 0) == 0);
            BnGPUOp op = {0};
            op.op_code = split ? BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT
                               : BN_GPU_CODE_SSM_ALPHA_BETA;
            op.W_buf = bias_buf; op.buf_in = BN_GPU_VALUE_XB;
            op.p[0] = n; op.p[1] = n;
            uintptr_t raw = (uintptr_t)decay_buf;
            op.p[6] = (uint32_t)raw; op.p[7] = (uint32_t)(raw >> 32);
            assert(gpu->execute(gpu->ctx, &op, 1,
                                beta ? BN_GPU_VALUE_SSM_BETA : BN_GPU_VALUE_SSM_ALPHA,
                                output, n) == 0);
            const float *ref = beta ? cuda_ssm_beta_reference : cuda_ssm_alpha_reference;
            assert(memcmp(output, ref, sizeof(output)) == 0);
        }
    }
    gpu->buffer_destroy(gpu->ctx, bias_buf);
    gpu->buffer_destroy(gpu->ctx, decay_buf);
    gpu->free_activations(gpu->ctx);
}

static void run_silu_graph_accuracy_case(BnGPUBackend *gpu) {
    enum { n = 512 };
    BnConfig cfg = {0};
    cfg.dim = cfg.hidden_dim = cfg.vocab_size = n;
    cfg.n_layers = 1;
    cfg.seq_len = 8;
    cfg.n_heads = 4;
    cfg.head_size = cfg.kv_dim = 128;
    cfg.rope_theta = 10000.0f;
    assert(init_test_activations(gpu, &cfg) == 0);
    float input[n], up[n], output[n], native_output[n];
    for (int i = 0; i < n; i++) {
        input[i] = -12.0f + 24.0f * (float)i / (float)(n - 1);
        up[i] = 0.25f + (float)(i % 17) * 0.125f;
    }
    for (int gated = 0; gated < 2; gated++) {
        for (int reference = 0; reference < 2; reference++) {
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB,
                                         input, sizeof(input), 0) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB2,
                                         up, sizeof(up), 0) == 0);
            BnGPUOp op = {0};
            op.op_code = gated ? BN_GPU_CODE_SILU_GATE : BN_GPU_CODE_SILU_ACT;
            op.buf_in = BN_GPU_VALUE_HB;
            op.buf_out = -1;
            op.buf_aux = gated ? BN_GPU_VALUE_HB2 : -1;
            op.p[0] = n;
            op.flags = reference ? BN_GPU_OP_FLAG_REFERENCE_SILU : 0;
            assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_HB,
                                output, n) == 0);
            if (!reference)
                memcpy(native_output, output, sizeof(output));
            else
                assert(memcmp(native_output, output, sizeof(output)) == 0);
            for (int i = 0; i < n; i++) {
                double expected = input[i] / (1.0 + exp(-(double)input[i]));
                if (gated) expected *= up[i];
                assert(isfinite(output[i]));
                assert(fabs(output[i] - expected) <=
                       3e-7 * (1.0 + fabs(expected)));
            }
        }
    }
    gpu->free_activations(gpu->ctx);
}

static void run_gelu_gate_accuracy_case(BnGPUBackend *gpu) {
    assert(gpu->caps & BN_GPU_CAP_FP32_GELU);
    enum { n = 512 };
    BnConfig cfg = {0};
    cfg.dim = cfg.hidden_dim = cfg.vocab_size = 2 * n;
    cfg.n_layers = 1; cfg.seq_len = 8; cfg.n_heads = 8;
    cfg.head_size = 64; cfg.kv_dim = n;
    cfg.rope_theta = 10000.0f;
    assert(init_test_activations(gpu, &cfg) == 0);
    float input[n], aux[2 * n], output[n];
    for (int i = 0; i < n; i++) {
        input[i] = -8.0f + 16.0f * (float)i / (float)(n - 1);
        aux[i] = -100.0f;
        aux[n + i] = 0.25f + (float)(i % 17) * 0.125f;
    }
    for (int reference = 0; reference < 2; reference++) {
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB,
                                     input, sizeof(input), 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB2,
                                     aux, sizeof(aux), 0) == 0);
        BnGPUOp op = {0};
        op.op_code = BN_GPU_CODE_GELU_GATE;
        op.buf_in = BN_GPU_VALUE_HB; op.buf_aux = BN_GPU_VALUE_HB2;
        op.buf_out = -1; op.p[0] = n; op.p[1] = n;
        op.flags = reference ? BN_GPU_OP_FLAG_REFERENCE_ACTIVATION : 0;
        assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_HB,
                            output, n) == 0);
        for (int i = 0; i < n; i++) {
            double expected = (double)cuda_gelu_reference[i] * aux[n + i];
            assert(isfinite(output[i]));
            assert(fabs(output[i] - expected) <= 3e-7 * (1.0 + fabs(expected)));
        }
    }
    gpu->free_activations(gpu->ctx);
}

static void run_sigmoid_gate_accuracy_case(BnGPUBackend *gpu) {
    enum { n = 512, head_size = 64 };
    BnConfig cfg = {0};
    cfg.dim = cfg.hidden_dim = cfg.vocab_size = 2 * n;
    cfg.n_layers = 1; cfg.seq_len = 8; cfg.n_heads = 8;
    cfg.head_size = head_size; cfg.kv_dim = n;
    cfg.rope_theta = 10000.0f;
    assert(init_test_activations(gpu, &cfg) == 0);
    float gates[n], values[n], interleaved[2 * n], output[n];
    for (int i = 0; i < n; i++) {
        gates[i] = -16.0f + 32.0f * (float)i / (float)(n - 1);
        values[i] = sinf((float)(i * 3 + 1)) * 2.0f;
        int h = i / head_size, d = i % head_size;
        interleaved[h * 2 * head_size + d] = 100.0f;
        interleaved[h * 2 * head_size + head_size + d] = gates[i];
    }
    for (int packed = 0; packed < 2; packed++) {
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB,
            packed ? values : gates, sizeof(gates), 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB2,
            packed ? interleaved : values,
            packed ? sizeof(interleaved) : sizeof(values), 0) == 0);
        BnGPUOp op = {0};
        op.op_code = BN_GPU_CODE_SIGMOID_GATE;
        op.buf_in = BN_GPU_VALUE_HB; op.buf_aux = BN_GPU_VALUE_HB2;
        op.buf_out = -1; op.p[0] = n;
        op.p[1] = packed ? head_size : 0;
        assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_HB,
                            output, n) == 0);
        for (int i = 0; i < n; i++) {
            double expected = values[i] / (1.0 + exp(-(double)gates[i]));
            assert(isfinite(output[i]));
            assert(fabs(output[i] - expected) <= 3e-7 * (1.0 + fabs(expected)));
        }
    }
    gpu->free_activations(gpu->ctx);
}

static void run_q6k_block32_graph_case(BnGPUBackend *gpu) {
    enum { rows = 1024, cols = 2560, blocks_per_row = cols / BN_QK_K };
    BnBlockQ6K *weights = calloc((size_t)rows * blocks_per_row,
                                 sizeof(*weights));
    float *input = malloc(cols * sizeof(float));
    float *quantized = malloc(cols * sizeof(float));
    float *output = malloc(rows * sizeof(float));
    assert(weights && input && quantized && output);
    for (int i = 0; i < cols; i++)
        input[i] = 0.3f * sinf((float)(i * 17 + 3));
    /* Independent block32 activation reference: scale is stored as FP16,
     * while integer rounding uses the original FP32 scale. */
    for (int b = 0; b < cols; b += 32) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++)
            amax = fmaxf(amax, fabsf(input[b + j]));
        float scale = amax / 127.0f;
        float stored_scale = bn_fp16_to_fp32(bn_fp32_to_fp16(scale));
        for (int j = 0; j < 32; j++)
            quantized[b + j] = stored_scale * roundf(input[b + j] / scale);
    }
    for (int b = 0; b < rows * blocks_per_row; b++) {
        weights[b].d = bn_fp32_to_fp16(0.0001f * (float)(1 + b % 5));
        for (int j = 0; j < 128; j++)
            weights[b].ql[j] = (uint8_t)(b * 29 + j * 13);
        for (int j = 0; j < 64; j++)
            weights[b].qh[j] = (uint8_t)(b * 7 + j * 17);
        for (int j = 0; j < 16; j++)
            weights[b].scales[j] = (int8_t)((b + j * 11) % 31 - 15);
    }
    BnConfig cfg = {0};
    cfg.dim = cols; cfg.hidden_dim = rows; cfg.vocab_size = rows;
    cfg.n_layers = 1; cfg.seq_len = 1; cfg.n_heads = 1;
    cfg.head_size = 2; cfg.kv_dim = 2; cfg.rope_theta = 10000.0f;
    assert(init_test_activations(gpu, &cfg) == 0);
    void *weight = gpu->buffer_create(gpu->ctx, weights,
        (size_t)rows * blocks_per_row * sizeof(*weights),
        BN_GGUF_TENSOR_Q6_K, rows, cols);
    assert(weight);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, input,
                                cols * sizeof(float), 0) == 0);
    BnGPUOp op = {0};
    op.op_code = BN_GPU_CODE_MATVEC; op.op_kind = BN_GPU_OP_MATVEC;
    op.W_buf = weight; op.type = BN_GGUF_TENSOR_Q6_K;
    op.rows = rows; op.cols = cols;
    op.buf_in = BN_GPU_VALUE_XB; op.buf_out = BN_GPU_VALUE_LOGITS;
    op.buf_aux = -1;
    assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_LOGITS,
                        output, rows) == 0);
    for (int host = 0; host < 2; host++) {
        if (host)
            assert(gpu->matvec(gpu->ctx, output, weight, input,
                               rows, cols, BN_GGUF_TENSOR_Q6_K) == 0);
        for (int r = 0; r < rows; r++) {
            double expected = 0.0;
            for (int b = 0; b < blocks_per_row; b++) {
                float values[BN_QK_K];
                bn_quant_dequant_q6k(&weights[r * blocks_per_row + b], values);
                for (int j = 0; j < BN_QK_K; j++)
                    expected += (double)values[j] * quantized[b * BN_QK_K + j];
            }
            assert(isfinite(output[r]));
            assert(fabs((double)output[r] - expected) < 1e-5);
        }
    }
    assert(bn_quant_format_has_cap(BN_GGUF_TENSOR_Q6_K, BN_QUANT_CAP_GPU_MMQ_F32_SCALE));
    const int batch_sizes[] = {1, 5, 7, 9};
    for (size_t case_i = 0; case_i < sizeof(batch_sizes) / sizeof(batch_sizes[0]); case_i++) {
        int tokens = batch_sizes[case_i];
        float *batch = malloc((size_t)tokens * cols * sizeof(float));
        float *batch_q = malloc((size_t)tokens * cols * sizeof(float));
        float *batch_out = malloc((size_t)tokens * rows * sizeof(float));
        assert(batch && batch_q && batch_out);
        for (int i = 0; i < tokens * cols; i++)
            batch[i] = i < 32 ? 0.0f : sinf((float)(i * 7 + 3)) * 0.31f;
        for (int b = 0; b < tokens * cols; b += 32) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(batch[b + j]));
            float inv = amax == 0.0f ? 0.0f : 127.0f / amax;
            float scale = inv == 0.0f ? 0.0f : 1.0f / inv;
            if (tokens <= 7) {
                float d = amax / 127.0f;
                float stored = bn_fp16_to_fp32(bn_fp32_to_fp16(d));
                for (int j = 0; j < 32; j++)
                    batch_q[b + j] = d == 0.0f ? 0.0f :
                        stored * roundf(batch[b + j] / d);
            } else {
                for (int j = 0; j < 32; j++)
                    batch_q[b + j] = scale * roundf(batch[b + j] * inv);
            }
        }
        assert(gpu->matmul(gpu->ctx, batch_out, weight, batch,
                           rows, cols, tokens, BN_GGUF_TENSOR_Q6_K) == 0);
        for (int t = 0; t < tokens; t++) {
            for (int r = 0; r < rows; r++) {
                double expected = 0.0;
                for (int b = 0; b < blocks_per_row; b++) {
                    float values[BN_QK_K];
                    bn_quant_dequant_q6k(&weights[r * blocks_per_row + b], values);
                    for (int j = 0; j < BN_QK_K; j++)
                        expected += (double)values[j] * batch_q[t * cols + b * BN_QK_K + j];
                }
                assert(isfinite(batch_out[t * rows + r]));
                assert(fabs(batch_out[t * rows + r] - expected) < 1e-5);
            }
        }
        free(batch); free(batch_q); free(batch_out);
    }
    gpu->buffer_destroy(gpu->ctx, weight);
    gpu->free_activations(gpu->ctx);
    free(output); free(quantized); free(input); free(weights);
}

static void run_hyper_connection_ops_case(BnGPUBackend *gpu) {
    enum { dim = 4, streams = 2, rank = 3 };
    BnConfig cfg = {0};
    cfg.dim = dim;
    cfg.hidden_dim = 8;
    cfg.n_layers = 1;
    cfg.n_heads = 1;
    cfg.n_kv_heads = 1;
    cfg.vocab_size = 16;
    cfg.seq_len = 8;
    cfg.rope_theta = BN_DEFAULT_ROPE_THETA;
    cfg.head_size = dim;
    cfg.kv_dim = dim;
    cfg.kv_mul = 1;
    cfg.hyper_connection_count = streams;
    cfg.hyper_connection_rank = rank;
    assert(init_test_activations(gpu, &cfg) == 0);

    const float residual[streams * dim] = {
        1.0f, -2.0f, 3.0f, -4.0f,
        0.5f, 1.5f, -2.5f, 3.5f,
    };
    const float weight[streams * dim] = {
        1.0f, 0.5f, 1.5f, -0.5f,
        0.25f, 2.0f, -1.0f, 0.75f,
    };
    float norm_ref[streams * dim];
    for (int s = 0; s < streams; s++) {
        float ss = 0.0f;
        for (int i = 0; i < dim; i++)
            ss += residual[s * dim + i] * residual[s * dim + i];
        float scale = 1.0f / sqrtf(ss / (float)dim + 1e-5f);
        for (int i = 0; i < dim; i++)
            norm_ref[s * dim + i] = residual[s * dim + i] * scale *
                                     weight[s * dim + i];
    }
    void *w = gpu->buffer_create(gpu->ctx, weight, sizeof(weight),
                                 BN_GGUF_TENSOR_F32, streams, dim);
    assert(w != NULL);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HC_RESIDUAL,
                                 residual, sizeof(residual), 0) == 0);
    BnGPUOp norm = {0};
    norm.op_code = BN_GPU_CODE_HC_STREAM_RMSNORM;
    norm.W_buf = w;
    norm.buf_in = BN_GPU_VALUE_HC_RESIDUAL;
    norm.buf_out = BN_GPU_VALUE_HC_NORM;
    norm.buf_aux = -1;
    norm.p[0] = dim;
    norm.p[1] = streams;
    norm.p[2] = f32_bits(1e-5f);
    float got[streams * dim] = {0};
    assert(gpu->execute(gpu->ctx, &norm, 1, BN_GPU_VALUE_HC_NORM,
                        got, streams * dim) == 0);
    expect_close(got, norm_ref, streams * dim);
    gpu->buffer_destroy(gpu->ctx, w);

    const float low_rank[rank] = {-3.0f, 0.5f, 4.0f};
    float silu_ref[rank];
    for (int i = 0; i < rank; i++) {
        float v = low_rank[i] / (float)streams;
        silu_ref[i] = v / (1.0f + expf(-v));
    }
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HC_LOW_RANK,
                                 low_rank, sizeof(low_rank), 0) == 0);
    BnGPUOp silu = {0};
    silu.op_code = BN_GPU_CODE_HC_SCALE_SILU;
    silu.buf_in = BN_GPU_VALUE_HC_LOW_RANK;
    silu.buf_out = -1;
    silu.buf_aux = -1;
    silu.p[0] = rank;
    silu.p[1] = f32_bits(1.0f / (float)streams);
    float silu_got[rank] = {0};
    assert(gpu->execute(gpu->ctx, &silu, 1, BN_GPU_VALUE_HC_LOW_RANK,
                        silu_got, rank) == 0);
    expect_close(silu_got, silu_ref, rank);
    float silu_host[rank];
    memcpy(silu_host, low_rank, sizeof(silu_host));
    assert(bn_gpu_backend_hyper_connection_scaled_silu_batch(
               gpu, silu_host, rank, 1.0f / (float)streams) == 0);
    expect_close(silu_host, silu_ref, rank);

    const float gate[streams * dim] = {
        -1.0f, 0.0f, 1.0f, 2.0f,
        2.5f, -2.0f, 0.25f, -0.5f,
    };
    float reduce_ref[dim] = {0};
    for (int s = 0; s < streams; s++)
        for (int i = 0; i < dim; i++)
            reduce_ref[i] += norm_ref[s * dim + i] /
                (1.0f + expf(-gate[s * dim + i])) / (float)streams;
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HC_GATE,
                                 gate, sizeof(gate), 0) == 0);
    BnGPUOp reduce = {0};
    reduce.op_code = BN_GPU_CODE_HC_GATED_REDUCE;
    reduce.buf_in = BN_GPU_VALUE_HC_NORM;
    reduce.buf_out = BN_GPU_VALUE_X;
    reduce.buf_aux = BN_GPU_VALUE_HC_GATE;
    reduce.p[0] = dim;
    reduce.p[1] = streams;
    float reduce_got[dim] = {0};
    assert(gpu->execute(gpu->ctx, &reduce, 1, BN_GPU_VALUE_X,
                        reduce_got, dim) == 0);
    expect_close(reduce_got, reduce_ref, dim);
    float reduce_host[dim] = {0};
    assert(bn_gpu_backend_hyper_connection_mix_batch(
               gpu, reduce_host, norm_ref, gate, 1, dim, streams) == 0);
    expect_close(reduce_host, reduce_ref, dim);

    const float block_out[dim] = {0.25f, -0.75f, 1.25f, -1.5f};
    const float inject[streams] = {1.0f, -2.0f};
    float combine_ref[streams * dim];
    memcpy(combine_ref, residual, sizeof(combine_ref));
    for (int s = 0; s < streams; s++) {
        float scatter = 2.0f / (1.0f + expf(-inject[s] / streams));
        for (int i = 0; i < dim; i++)
            combine_ref[s * dim + i] += block_out[i] * scatter;
    }
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HC_RESIDUAL,
                                 residual, sizeof(residual), 0) == 0);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB,
                                 block_out, sizeof(block_out), 0) == 0);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HC_INJECT,
                                 inject, sizeof(inject), 0) == 0);
    BnGPUOp combine = {0};
    combine.op_code = BN_GPU_CODE_HC_COMBINE;
    combine.buf_in = BN_GPU_VALUE_HC_RESIDUAL;
    combine.buf_out = -1;
    combine.buf_aux = BN_GPU_VALUE_XB;
    combine.p[0] = dim;
    combine.p[1] = streams;
    float combine_got[streams * dim] = {0};
    assert(gpu->execute(gpu->ctx, &combine, 1,
                        BN_GPU_VALUE_HC_RESIDUAL, combine_got,
                        streams * dim) == 0);
    expect_close(combine_got, combine_ref, streams * dim);

    gpu->free_activations(gpu->ctx);
}

static void run_hc_combine_reference_cases(BnGPUBackend *gpu) {
    size_t words = 0;
    const size_t cases = sizeof(cuda_hc_combine_reference) /
                         sizeof(cuda_hc_combine_reference[0]);
    for (size_t ci = 0; ci < cases; ci++) {
        const BnCudaHCCombineReference *r = &cuda_hc_combine_reference[ci];
        int dim = r->dim, streams = r->streams, count = dim * streams;
        BnConfig cfg = {0};
        cfg.dim = dim; cfg.hidden_dim = dim; cfg.n_layers = 1;
        cfg.n_heads = dim / 64; cfg.n_kv_heads = 1; cfg.head_size = 64;
        cfg.kv_dim = 64; cfg.seq_len = 8; cfg.vocab_size = 16;
        cfg.hyper_connection_count = streams; cfg.hyper_connection_rank = 3;
        cfg.rope_theta = BN_DEFAULT_ROPE_THETA;
        assert(init_test_activations(gpu, &cfg) == 0);
        float *residual = malloc((size_t)count * sizeof(float));
        float *block = malloc((size_t)dim * sizeof(float));
        float *inject = malloc((size_t)streams * sizeof(float));
        float *guard = malloc(((size_t)count + 2) * sizeof(float));
        assert(residual && block && inject && guard);
        float *got = guard + 1;
        guard[0] = guard[count + 1] = 12345.0f;
        for (int i = 0; i < count; i++)
            residual[i] = sinf((float)(i * 13 + r->seed)) * r->magnitude;
        for (int i = 0; i < dim; i++)
            block[i] = cosf((float)(i * 7 + 3 + r->seed)) * r->magnitude;
        for (int i = 0; i < streams; i++)
            inject[i] = sinf((float)(i * 5 + 1 + r->seed)) * 3.0f;
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB,
                                    block, (size_t)dim * sizeof(float), 0) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HC_INJECT,
                                    inject, (size_t)streams * sizeof(float), 0) == 0);
        BnGPUOp ops[32] = {0};
        for (int i = 0; i < 31; i++) {
            ops[i].op_code = BN_GPU_CODE_COPY;
            ops[i].buf_in = BN_GPU_VALUE_XB;
            ops[i].buf_out = BN_GPU_VALUE_SCRATCH;
            ops[i].p[2] = dim;
        }
        ops[31].op_code = BN_GPU_CODE_HC_COMBINE;
        ops[31].buf_in = BN_GPU_VALUE_HC_RESIDUAL;
        ops[31].buf_out = -1;
        ops[31].buf_aux = BN_GPU_VALUE_XB;
        ops[31].p[0] = dim; ops[31].p[1] = streams;
        for (int pass = 0; pass < 3; pass++) {
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HC_RESIDUAL,
                        residual, (size_t)count * sizeof(float), 0) == 0);
            assert(gpu->execute(gpu->ctx, pass ? ops : ops + 31,
                        pass ? 32 : 1, BN_GPU_VALUE_HC_RESIDUAL, got, count) == 0);
            assert(cuda_test_float_bits_hash(got, count) == r->hash);
            assert(guard[0] == 12345.0f && guard[count + 1] == 12345.0f);
            words += (size_t)count;
        }
        free(guard); free(inject); free(block); free(residual);
        gpu->free_activations(gpu->ctx);
    }
    printf("CUDA HC combine reference PASSED: %zu cases, %zu words\n", cases, words);
}

static void run_hc_mixer_reference_cases(BnGPUBackend *gpu) {
    size_t words = 0;
    const size_t cases = sizeof(cuda_hc_mixer_reference) /
                         sizeof(cuda_hc_mixer_reference[0]);
    for (size_t ci = 0; ci < cases; ci++) {
        const BnCudaHCMixerReference *r = &cuda_hc_mixer_reference[ci];
        int dim = r->dim, streams = r->streams;
        int count = r->mode ? dim * streams : dim;
        BnConfig cfg = {0};
        cfg.dim = dim; cfg.hidden_dim = dim; cfg.n_layers = 1;
        cfg.n_heads = dim / 64; cfg.n_kv_heads = 1; cfg.head_size = 64;
        cfg.kv_dim = 64; cfg.seq_len = 8; cfg.vocab_size = 16;
        cfg.hyper_connection_count = streams; cfg.hyper_connection_rank = dim;
        cfg.rope_theta = BN_DEFAULT_ROPE_THETA;
        assert(init_test_activations(gpu, &cfg) == 0);
        float *input = malloc((size_t)count * sizeof(float));
        float *gate = malloc((size_t)count * sizeof(float));
        float *guard = malloc(((size_t)dim + 2) * sizeof(float));
        assert(input && gate && guard);
        float *got = guard + 1;
        guard[0] = guard[dim + 1] = 12345.0f;
        for (int i = 0; i < count; i++) {
            input[i] = sinf((float)(i * 13 + r->seed)) * r->magnitude;
            gate[i] = cosf((float)(i * 7 + 3 + r->seed)) * 3.0f;
        }
        int in = r->mode ? BN_GPU_VALUE_HC_NORM : BN_GPU_VALUE_HC_LOW_RANK;
        int out = r->mode ? BN_GPU_VALUE_X : in;
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HC_GATE,
                                    gate, (size_t)count * sizeof(float), 0) == 0);
        BnGPUOp ops[32] = {0};
        for (int i = 0; i < 31; i++) {
            ops[i].op_code = BN_GPU_CODE_COPY;
            ops[i].buf_in = in;
            ops[i].buf_out = BN_GPU_VALUE_SCRATCH;
            ops[i].p[2] = dim;
        }
        ops[31].op_code = r->mode ? BN_GPU_CODE_HC_GATED_REDUCE
                                  : BN_GPU_CODE_HC_SCALE_SILU;
        ops[31].buf_in = in;
        ops[31].buf_out = r->mode ? out : -1;
        ops[31].buf_aux = r->mode ? BN_GPU_VALUE_HC_GATE : -1;
        ops[31].p[0] = dim;
        ops[31].p[1] = r->mode ? (uint32_t)streams : f32_bits(1.0f / streams);
        for (int pass = 0; pass < 3; pass++) {
            assert(gpu->write_activation(gpu->ctx, in, input,
                        (size_t)count * sizeof(float), 0) == 0);
            assert(gpu->execute(gpu->ctx, pass ? ops : ops + 31,
                        pass ? 32 : 1, out, got, dim) == 0);
            assert(cuda_test_float_bits_hash(got, dim) == r->hash);
            assert(guard[0] == 12345.0f && guard[dim + 1] == 12345.0f);
            words += (size_t)dim;
        }
        free(guard); free(gate); free(input);
        gpu->free_activations(gpu->ctx);
    }
    printf("CUDA HC mixer reference PASSED: %zu cases, %zu words\n", cases, words);
}

static void run_hc_rmsnorm_reference_cases(BnGPUBackend *gpu) {
    size_t words = 0;
    const size_t cases = sizeof(cuda_hc_rmsnorm_reference) /
                         sizeof(cuda_hc_rmsnorm_reference[0]);
    for (size_t ci = 0; ci < cases; ci++) {
        const BnCudaHCRMSNormReference *r = &cuda_hc_rmsnorm_reference[ci];
        size_t count = (size_t)r->dim * r->streams;
        float *input = malloc(count * sizeof(float));
        float *weight = malloc(count * sizeof(float));
        float *guard = malloc((count + 2) * sizeof(float));
        assert(input && weight && guard);
        for (size_t i = 0; i < count; i++) {
            input[i] = sinf((float)(i * 13 + r->seed)) * r->magnitude;
            weight[i] = cosf((float)(i * 7 + r->seed + 3)) * 1.5f;
        }
        BnConfig cfg = {0};
        cfg.dim = r->dim;
        cfg.hidden_dim = r->dim;
        cfg.n_layers = 1;
        cfg.n_heads = r->dim / 64;
        cfg.n_kv_heads = 1;
        cfg.head_size = 64;
        cfg.kv_dim = 64;
        cfg.seq_len = 1;
        cfg.vocab_size = r->dim;
        cfg.hyper_connection_count = r->streams;
        cfg.hyper_connection_rank = r->dim;
        cfg.rope_theta = BN_DEFAULT_ROPE_THETA;
        assert(init_test_activations(gpu, &cfg) == 0);
        void *weight_buf = gpu->buffer_create(
            gpu->ctx, weight, count * sizeof(float), BN_GGUF_TENSOR_F32,
            r->streams, r->dim);
        assert(weight_buf);
        BnGPUOp op = {0};
        op.op_code = BN_GPU_CODE_HC_STREAM_RMSNORM;
        op.W_buf = weight_buf;
        op.buf_in = BN_GPU_VALUE_HC_RESIDUAL;
        op.buf_out = BN_GPU_VALUE_HC_NORM;
        op.buf_aux = -1;
        op.p[0] = (uint32_t)r->dim;
        op.p[1] = (uint32_t)r->streams;
        op.p[2] = f32_bits(1e-6f);
        guard[0] = guard[count + 1] = 12345.0f;
        for (int pass = 0; pass < 3; pass++) {
            assert(gpu->write_activation(
                gpu->ctx, BN_GPU_VALUE_HC_RESIDUAL, input,
                count * sizeof(float), 0) == 0);
            assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_HC_NORM,
                                guard + 1, (int)count) == 0);
            assert(cuda_test_float_bits_hash(guard + 1, count) == r->hash);
            assert(guard[0] == 12345.0f && guard[count + 1] == 12345.0f);
            words += count;
        }
        gpu->buffer_destroy(gpu->ctx, weight_buf);
        gpu->free_activations(gpu->ctx);
        free(guard);
        free(weight);
        free(input);
    }
    printf("CUDA HC RMSNorm reference PASSED: %zu cases, %zu words\n",
           cases, words);
}

static void run_signed_sqrt_gate_reference_cases(BnGPUBackend *gpu) {
    const int dim = 2560;
    const int streams = 4;
    size_t words = 0;
    size_t cases = sizeof(cuda_signed_sqrt_gate_reference) /
                   sizeof(cuda_signed_sqrt_gate_reference[0]);
    assert(gpu->signed_sqrt_gate);
    for (size_t ci = 0; ci < cases; ci++) {
        const BnCudaSignedSqrtGateReference *r =
            &cuda_signed_sqrt_gate_reference[ci];
        size_t wide = (size_t)r->tokens * streams * dim;
        size_t value_count = (size_t)r->tokens * dim;
        float *key = malloc(wide * sizeof(float));
        float *query = malloc(wide * sizeof(float));
        float *value = malloc(value_count * sizeof(float));
        float *gate = malloc((size_t)r->tokens * streams * sizeof(float));
        float *gated = malloc(wide * sizeof(float));
        assert(key && query && value && gate && gated);
        for (size_t i = 0; i < wide; i++) {
            float v = sinf((float)(i % 10007) * 0.113f);
            key[i] = r->pattern == 0 ? 0.0f :
                     r->pattern == 1 ? v * 1e-5f : v * 2.0f;
            query[i] = r->pattern == 0 ? 1.0f :
                       r->pattern == 1
                           ? cosf((float)(i % 7919) * 0.27f) * 1e-5f
                           : r->pattern == 2 ? -key[i] : key[i];
        }
        for (size_t i = 0; i < value_count; i++)
            value[i] = cosf((float)(i % 65521) * 0.123f) * 3.0f;
        for (int token = 0; token < r->tokens; token++) {
            size_t key_off = (size_t)token * streams * dim;
            size_t value_off = (size_t)token * dim;
            assert(gpu->signed_sqrt_gate(
                gpu->ctx, gate + (size_t)token * streams,
                gated + key_off, key + key_off, query + key_off,
                value + value_off, dim, streams, r->reduction_threads) == 0);
        }
        assert(cuda_test_float_bits_hash(
                   gate, (size_t)r->tokens * streams) == r->gate_hash);
        assert(cuda_test_float_bits_hash(gated, wide) == r->gated_hash);
        words += (size_t)r->tokens * streams + wide;
        free(gated);
        free(gate);
        free(value);
        free(query);
        free(key);
    }
    printf("CUDA signed-root gate reference PASSED: %zu cases, %zu words\n",
           cases, words);
}

static void run_rope_preparation_case(BnGPUBackend *gpu) {
    float table[192], factors[64], got[192];
    for (int i = 0; i < 192; i++) table[i] = -99;
    for (int i = 0; i < 64; i++) factors[i] = 0.5f + (float)i / 32;
    BnGPURopeFrequencyPlan recipes[3] = {
        {0, 64, 128, 1000000.0f, NULL, BN_GPU_ROPE_FACTOR_NONE},
        {64, 32, 64, 10000.0f, factors, BN_GPU_ROPE_FACTOR_MULTIPLY},
        {128, 64, 128, 1000000.0f, factors, BN_GPU_ROPE_FACTOR_DIVIDE},
    };
    BnGPUActivationPlan plan = {0};
    plan.dim = 128; plan.n_layers = 3; plan.seq_len = 2;
    plan.kv_dim = 128; plan.n_heads = 1; plan.head_size = 128;
    plan.vocab_size = 128; plan.attention_layer_count = 3;
    plan.rope_frequencies = table; plan.rope_frequency_count = 192;
    plan.rope_frequency_plans = recipes; plan.rope_frequency_plan_count = 3;
    assert(gpu->init_activations(gpu->ctx, &plan) == 0);
    assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_ROPE_FREQ,
        got, sizeof(got), 0) == 0);
    for (int r = 0; r < 3; r++) {
        float scale = powf(recipes[r].theta, -2.0f / recipes[r].rotary_dims);
        for (int i = 0; i < recipes[r].pair_count; i++) {
            double expected = pow((double)scale, i);
            if (r == 1) expected *= factors[i];
            if (r == 2) expected /= factors[i];
            /* Device power uses approximate log/exp; this checks recipe
             * placement and factors. Direct CUDA-reference probes check parity. */
            assert(fabs(got[recipes[r].offset + i] - expected) <= 1e-5 * expected);
        }
    }
    for (int i = 96; i < 128; i++) assert(got[i] == -99);
    gpu->free_activations(gpu->ctx);
    recipes[2].offset = 192;
    assert(gpu->init_activations(gpu->ctx, &plan) == -1);
    plan.rope_frequency_plans = NULL; plan.rope_frequency_plan_count = 0;
    assert(gpu->init_activations(gpu->ctx, &plan) == 0);
    assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_ROPE_FREQ,
        got, sizeof(got), 0) == 0);
    assert(memcmp(got, table, sizeof(table)) == 0);
    gpu->free_activations(gpu->ctx);
}

static void run_activation_upload_readiness_case(BnGPUBackend *gpu) {
    /* A raw table has no preparation kernel to synchronize its upload.
     * Verify immediate readback on the backend execution stream. */
    const int layers = 16384, head_size = 128;
    size_t count = (size_t)layers * (head_size / 2);
    size_t bytes = count * sizeof(float);
    float *table = malloc(bytes), *got = malloc(bytes);
    assert(table && got);
    BnGPUActivationPlan plan = {0};
    plan.dim = head_size; plan.n_layers = layers; plan.seq_len = 1;
    plan.kv_dim = head_size; plan.n_heads = 1; plan.head_size = head_size;
    plan.vocab_size = head_size; plan.attention_layer_count = layers;
    plan.rope_frequencies = table; plan.rope_frequency_count = (int)count;
    for (int trial = 0; trial < 8; trial++) {
        for (size_t i = 0; i < count; i++)
            table[i] = (float)((int)(i % 257) - 128 + trial) / 64.0f;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_ROPE_FREQ,
                                    got, bytes, 0) == 0);
        assert(memcmp(got, table, bytes) == 0);
        gpu->free_activations(gpu->ctx);
    }
    free(got); free(table);
    printf("CUDA activation upload readiness PASSED\n");
}

#include "cuda_gelu_prefill_reference.h"
static void *gelu_prefill_reference_weights(size_t elements, int seed) {
    size_t blocks = elements / 32;
    BnBlockQ4_0 *w = malloc(blocks * sizeof(*w));
    assert(w);
    for (size_t b = 0; b < blocks; b++) {
        w[b].d = (uint16_t)(0x2400 + ((b * 13 + seed) % 20) * 128);
        if (b % 3 == 0) w[b].d |= 0x8000;
        for (int j = 0; j < 16; j++)
            w[b].qs[j] = (uint8_t)(((b * 7 + j * 3 + seed) & 15) |
                (((b * 11 + j * 5 + seed + 1) & 15) << 4));
    }
    return w;
}

static void run_gelu_prefill_reference_cases(BnGPUBackend *gpu) {
    for (size_t c = 0; c < sizeof(cuda_gelu_prefill_reference) /
                                  sizeof(cuda_gelu_prefill_reference[0]); c++) {
        const BnCudaGeluPrefillReference *r = &cuda_gelu_prefill_reference[c];
        int dim = r->dim, hidden = r->hidden, nt = r->tokens;
        size_t elements = (size_t)dim * hidden, bytes = elements / 32 * sizeof(BnBlockQ4_0);
        size_t count = (size_t)dim * nt;
        void *weights[3] = {gelu_prefill_reference_weights(elements, r->seed),
            gelu_prefill_reference_weights(elements, r->seed + 7),
            gelu_prefill_reference_weights(elements, r->seed + 13)};
        unsigned char *stacked = malloc(2 * bytes);
        float *input = malloc(count * sizeof(float));
        float *output = malloc((count + 2) * sizeof(float));
        float *norm = malloc((size_t)dim * sizeof(float));
        assert(stacked && input && output && norm);
        memcpy(stacked, weights[0], bytes); memcpy(stacked + bytes, weights[1], bytes);
        for (size_t i = 0; i < count; i++)
            input[i] = (float)((int)((i * 37 + (unsigned)r->seed * 11) % 257) - 128) / 31.7f;
        for (int i = 0; i < dim; i++) norm[i] = 0.25f + (float)(i % 13) / 16.0f;
        void *nw = gpu->buffer_create(gpu->ctx, norm, (size_t)dim * sizeof(float),
                                      BN_GGUF_TENSOR_F32, 1, dim);
        assert(nw);
        for (int raw = 0; raw < 2; raw++) for (int stack = 0; stack < 2; stack++) {
            void *buffers[3];
            for (int i = 0; i < 3; i++) {
                const void *data = i == 0 && stack ? stacked : weights[i];
                size_t size = i == 0 && stack ? 2 * bytes : bytes;
                int rows = i == 2 ? dim : (i == 0 && stack ? 2 * hidden : hidden);
                int cols = i == 2 ? hidden : dim;
                buffers[i] = raw ? gpu->buffer_create_quant_only(gpu->ctx, data, size,
                    BN_GGUF_TENSOR_Q4_0, rows, cols) : gpu->buffer_create(gpu->ctx,
                    data, size, BN_GGUF_TENSOR_Q4_0, rows, cols);
                assert(buffers[i]);
            }
            output[0] = output[count + 1] = -12345.0f;
            void *up = stack ? NULL : buffers[1];
            int rc;
            if (r->mode == 0)
                rc = gpu->dense_ffn_batch(gpu->ctx, output + 1, buffers[0], up,
                    buffers[2], input, nt, dim, hidden, BN_GGUF_TENSOR_Q4_0,
                    BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0, BN_MODEL_ACTIVATION_GELU);
            else if (r->mode == 1)
                rc = gpu->dense_ffn_batch_norm(gpu->ctx, output + 1, buffers[0], up,
                    buffers[2], nw, input, nt, dim, hidden, BN_GGUF_TENSOR_Q4_0,
                    BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0, BN_MODEL_ACTIVATION_GELU, 1e-6f);
            else
                rc = gpu->dense_ffn_batch_norm_resid(gpu->ctx, output + 1, buffers[0], up,
                    buffers[2], nw, input, nt, dim, hidden, BN_GGUF_TENSOR_Q4_0,
                    BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0, BN_MODEL_ACTIVATION_GELU, 1e-6f);
            assert(rc == 0);
            assert(cuda_test_float_bits_hash(output + 1, count) == r->hash);
            assert(output[0] == -12345.0f && output[count + 1] == -12345.0f);
            for (int i = 0; i < 3; i++) gpu->buffer_destroy(gpu->ctx, buffers[i]);
        }
        gpu->buffer_destroy(gpu->ctx, nw);
        for (int i = 0; i < 3; i++) free(weights[i]);
        free(stacked); free(input); free(output); free(norm);
    }
    printf("CUDA Q4 GELU prefill reference PASSED\n");
}


#include "cuda_routed_q4_reference.h"
#include "cuda_merged_routed_q4_reference.h"
static void *merged_routed_reference_weights(size_t elements, int seed) {
    unsigned char *data = gelu_prefill_reference_weights(elements, seed);
    // Fine mantissas expose changes in stream-K reduction order.
    for (size_t b = 0; b < elements / 32; b++) {
        uint16_t d = (uint16_t)(0x1801 + ((b * 73 + seed * 29) % 4093));
        if (b % 3 == 0) d |= 0x8000;
        memcpy(data + b * sizeof(BnBlockQ4_0), &d, sizeof(d));
    }
    return data;
}

static void run_routed_q4_reference_cases_impl(BnGPUBackend *gpu,
    const BnCudaRoutedQ4Reference *refs, size_t n_cases, int merged) {
    assert(gpu->moe_expert_ffn_batch);
    for (size_t c = 0; c < n_cases; c++) {
        const BnCudaRoutedQ4Reference *r = &refs[c];
        int dim = r->dim, hidden = r->hidden, nt = r->tokens, k = 8;
        size_t elements = (size_t)dim * hidden, bytes = elements / 32 * sizeof(BnBlockQ4_0);
        size_t count = (size_t)dim * nt * k;
        void *(*make_weights)(size_t, int) = merged
            ? merged_routed_reference_weights : gelu_prefill_reference_weights;
        unsigned char *w[3] = {
            make_weights(elements * r->experts, r->seed),
            make_weights(elements * r->experts, r->seed + 7),
            make_weights(elements * r->experts, r->seed + 13)};
        float *input = malloc((size_t)dim * nt * sizeof(float));
        float *gather = malloc((size_t)dim * nt * sizeof(float));
        float *expert_out = malloc(((size_t)dim * nt + 2) * sizeof(float));
        float *output = malloc(count * sizeof(float));
        unsigned char *stacked = malloc(2 * bytes);
        int *slots = malloc((size_t)nt * sizeof(int));
        assert(input && gather && expert_out && output && stacked && slots);
        for (size_t i = 0; i < (size_t)dim * nt; i++)
            input[i] = (float)((int)((i * 37 + (unsigned)r->seed * 11) % 257) - 128) / 31.7f;
        for (int stack = 0; stack < 2; stack++) {
            for (int e = 0; e < r->experts; e++) {
                int tokens = 0;
                for (int t = 0; t < nt; t++) for (int j = 0; j < k; j++) {
                    int selected = (t * 3 + j * 5 + r->seed) % r->experts;
                    // Regression: the merged 704-row halves straddle tiles for
                    // an expert used by only one token in a 14-token batch.
                    if (merged && r->experts == 128 && nt == 14) {
                        if (selected == 51) selected = 52;
                        if (t == 11 && j == 7) selected = 51;
                    }
                    if (selected != e) continue;
                    assert(tokens < nt);
                    slots[tokens] = t * k + j;
                    memcpy(gather + (size_t)tokens * dim, input + (size_t)t * dim, dim * sizeof(float));
                    tokens++;
                }
                if (!tokens) continue;
                memcpy(stacked, w[0] + (size_t)e * bytes, bytes);
                memcpy(stacked + bytes, w[1] + (size_t)e * bytes, bytes);
                void *buffers[3];
                for (int i = 0; i < 3; i++) {
                    buffers[i] = gpu->buffer_create_quant_only(gpu->ctx,
                        i == 0 && stack ? stacked : w[i] + (size_t)e * bytes,
                        i == 0 && stack ? 2 * bytes : bytes, BN_GGUF_TENSOR_Q4_0,
                        i == 2 ? dim : (i == 0 && stack ? 2 * hidden : hidden),
                        i == 2 ? hidden : dim);
                    assert(buffers[i]);
                }
                BnGPUMoEExpertBatchPlan plan = {nt, r->experts, e, merged};
                size_t n = (size_t)tokens * dim;
                expert_out[0] = expert_out[n + 1] = -12345.0f;
                assert(gpu->moe_expert_ffn_batch(gpu->ctx, expert_out + 1,
                    buffers[0], stack ? NULL : buffers[1], buffers[2], gather,
                    tokens, dim, hidden, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0,
                    BN_GGUF_TENSOR_Q4_0, BN_MODEL_ACTIVATION_GELU, &plan) == 0);
                assert(expert_out[0] == -12345.0f && expert_out[n + 1] == -12345.0f);
                for (int t = 0; t < tokens; t++)
                    memcpy(output + (size_t)slots[t] * dim, expert_out + 1 + (size_t)t * dim, dim * sizeof(float));
                plan.gate_up_fused = 2;
                assert(gpu->moe_expert_ffn_batch(gpu->ctx, expert_out,
                    buffers[0], stack ? NULL : buffers[1], buffers[2], gather,
                    tokens, dim, hidden, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0,
                    BN_GGUF_TENSOR_Q4_0, BN_MODEL_ACTIVATION_GELU, &plan) != 0);
                assert(expert_out[0] == -12345.0f);
                plan.gate_up_fused = merged;
                plan.total_tokens = tokens - 1;
                assert(gpu->moe_expert_ffn_batch(gpu->ctx, expert_out,
                    buffers[0], stack ? NULL : buffers[1], buffers[2], gather,
                    tokens, dim, hidden, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0,
                    BN_GGUF_TENSOR_Q4_0, BN_MODEL_ACTIVATION_GELU, &plan) != 0);
                assert(expert_out[0] == -12345.0f);
                for (int i = 0; i < 3; i++) gpu->buffer_destroy(gpu->ctx, buffers[i]);
            }
            assert(cuda_test_float_bits_hash(output, count) == r->hash);
        }
        for (int i = 0; i < 3; i++) free(w[i]);
        free(input); free(gather); free(expert_out); free(output); free(stacked); free(slots);
    }
    printf("CUDA %s routed Q4 reference PASSED\n", merged ? "merged" : "independent");
}

static void run_routed_q4_reference_cases(BnGPUBackend *gpu) {
    run_routed_q4_reference_cases_impl(gpu, cuda_routed_q4_reference,
        sizeof(cuda_routed_q4_reference)/sizeof(cuda_routed_q4_reference[0]), 0);
}

static void run_merged_routed_q4_reference_cases(BnGPUBackend *gpu) {
    run_routed_q4_reference_cases_impl(gpu, cuda_merged_routed_q4_reference,
        sizeof(cuda_merged_routed_q4_reference)/sizeof(cuda_merged_routed_q4_reference[0]), 1);
}

#include "cuda_expert_reduction_reference.h"
static void run_expert_batch_reduction_reference_cases(BnGPUBackend *gpu) {
    assert(bn_gpu_backend_can_moe_reduce_batch(gpu));
    for (size_t c = 0; c < sizeof(cuda_expert_reduction_reference) /
                                  sizeof(cuda_expert_reduction_reference[0]); c++) {
        const BnCudaExpertReductionReference *r = &cuda_expert_reduction_reference[c];
        int n = r->dim, k = r->experts, nt = r->tokens;
        size_t rows = (size_t)nt*k, count = rows*n, outputs = (size_t)nt*n;
        float *input = malloc((count + 2)*sizeof(float));
        float *output = malloc((outputs + 2)*sizeof(float));
        float *scales = malloc(rows*sizeof(float));
        float *weights = malloc(rows*sizeof(float));
        assert(input && output && scales && weights);
        for (size_t row = 0; row < rows; row++) {
            scales[row] = (row + r->seed) % 7 == 0 ? 0.0f
                : ((float)((row*13 + r->seed)%47) - 23.0f)/17.0f;
            weights[row] = ((float)((row*7 + r->seed)%29) + 1.0f)/31.0f;
        }
        for (int alias = 0; alias < 2; alias++) for (int replay = 0; replay < 2; replay++) {
            input[0] = input[count+1] = output[0] = output[outputs+1] = 123.0f;
            for (size_t i = 0; i < count; i++)
                input[i+1] = (float)((int)((i*37 + r->seed*11)%509) - 254)/259.0f;
            float *out = alias ? input+1 : output+1;
            assert(bn_gpu_backend_moe_reduce_batch(gpu, out, input+1,
                weights, scales, nt, k, n) == 0);
            assert(cuda_test_float_bits_hash(out, outputs) == r->hash);
            assert(input[0] == 123.0f && input[count+1] == 123.0f);
            assert(output[0] == 123.0f && output[outputs+1] == 123.0f);
        }
        free(input); free(output); free(scales); free(weights);
    }
    float x[4] = {1,2,3,4}, coef[2] = {0.3f,0.7f};
    float guard[4] = {123,8,9,123}, saved[4]; memcpy(saved,guard,sizeof(guard));
    BnGPUBackend absent = {0};
    assert(bn_gpu_backend_moe_reduce_batch(NULL,guard+1,x,coef,coef,1,2,2) == -1);
    assert(bn_gpu_backend_moe_reduce_batch(&absent,guard+1,x,coef,coef,1,2,2) == -1);
    assert(bn_gpu_backend_moe_reduce_batch(gpu,guard+1,NULL,coef,coef,1,2,2) == -1);
    assert(bn_gpu_backend_moe_reduce_batch(gpu,guard+1,x,NULL,coef,1,2,2) == -1);
    assert(bn_gpu_backend_moe_reduce_batch(gpu,guard+1,x,coef,NULL,1,2,2) == -1);
    assert(bn_gpu_backend_moe_reduce_batch(gpu,guard+1,x,coef,coef,0,2,2) == -1);
    assert(bn_gpu_backend_moe_reduce_batch(gpu,guard+1,x,coef,coef,1,0,2) == -1);
    assert(bn_gpu_backend_moe_reduce_batch(gpu,guard+1,x,coef,coef,1,17,2) == -1);
    assert(bn_gpu_backend_moe_reduce_batch(gpu,guard+1,x,coef,coef,1,2,0) == -1);
    assert(bn_gpu_backend_moe_reduce_batch(gpu,guard+1,x,coef,coef,2147483647,2,2) == -1);
    assert(memcmp(guard,saved,sizeof(guard)) == 0);
    printf("CUDA expert batch reduction reference PASSED\n");
}

static void run_expert_reduction_reference_cases(BnGPUBackend *gpu) {
    assert(gpu->caps & BN_GPU_CAP_WEIGHTED_ADD_SEPARATE_SCALE);
    for (size_t c = 0; c < sizeof(cuda_expert_reduction_reference) /
                                  sizeof(cuda_expert_reduction_reference[0]); c++) {
        const BnCudaExpertReductionReference *r = &cuda_expert_reduction_reference[c];
        int n = r->dim, k = r->experts;
        size_t count = (size_t)n * k * r->tokens;
        BnConfig cfg = {0};
        cfg.dim = n; cfg.hidden_dim = n; cfg.n_layers = 1;
        cfg.n_heads = 1; cfg.n_kv_heads = 1; cfg.head_size = 2;
        cfg.kv_dim = 2; cfg.kv_mul = 1; cfg.seq_len = 2;
        cfg.vocab_size = n; cfg.rope_theta = BN_DEFAULT_ROPE_THETA;
        assert(init_test_activations(gpu, &cfg) == 0);
        float *input = malloc(count * sizeof(float));
        float *output = malloc((size_t)n * r->tokens * sizeof(float));
        assert(input && output);
        for (size_t i = 0; i < count; i++)
            input[i] = (float)((int)((i * 37 + r->seed * 11) % 509) - 254) / 259.0f;
        for (int replay = 0; replay < 2; replay++) {
            for (int t = 0; t < r->tokens; t++) {
                for (int e = 0; e < k; e++) {
                    int row = t * k + e;
                    float scale = (row + r->seed) % 7 == 0 ? 0.0f
                        : ((float)((row * 13 + r->seed) % 47) - 23.0f) / 17.0f;
                    float weight = ((float)((row * 7 + r->seed) % 29) + 1.0f) / 31.0f;
                    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB,
                        input + (size_t)row * n, (size_t)n * sizeof(float), 0) == 0);
                    BnGPUOp op = {0};
                    op.op_code = BN_GPU_CODE_WEIGHTED_ADD;
                    op.op_kind = BN_GPU_OP_RESIDUAL;
                    op.buf_in = BN_GPU_VALUE_LOGITS; op.buf_aux = BN_GPU_VALUE_XB;
                    op.p[0] = n; op.p[1] = f32_bits(weight); op.p[2] = e == 0;
                    op.p[3] = f32_bits(scale);
                    op.p[4] = BN_GPU_WEIGHTED_ADD_SEPARATE_SCALE;
                    if (k >= 2 && k <= 15) op.p[4] |= BN_GPU_WEIGHTED_ADD_FMA;
                    int last = e == k - 1;
                    assert(gpu->execute(gpu->ctx, &op, 1,
                        last ? BN_GPU_VALUE_LOGITS : -1,
                        last ? output + (size_t)t * n : NULL, last ? n : 0) == 0);
                }
            }
            assert(cuda_test_float_bits_hash(output, (size_t)n * r->tokens) == r->hash);
        }
        gpu->free_activations(gpu->ctx);
        free(output); free(input);
    }
    printf("CUDA scaled expert reduction reference PASSED\n");
}

#include "cuda_scaled_rmsnorm_reference.h"
static void run_scaled_rmsnorm_reference_cases(BnGPUBackend *gpu) {
    assert(gpu->caps & BN_GPU_CAP_RMSNORM_SEPARATE_SCALE);
    for (size_t c = 0; c < sizeof(cuda_scaled_rms_reference) /
                                  sizeof(cuda_scaled_rms_reference[0]); c++) {
        const BnCudaScaledRmsReference *r = &cuda_scaled_rms_reference[c];
        int n = r->dim;
        size_t count = (size_t)n * r->tokens;
        BnConfig cfg = {0};
        cfg.dim = n; cfg.hidden_dim = n; cfg.n_layers = 1;
        cfg.n_heads = 1; cfg.n_kv_heads = 1; cfg.head_size = 2;
        cfg.kv_dim = 2; cfg.kv_mul = 1; cfg.seq_len = 2;
        cfg.vocab_size = n; cfg.rope_theta = BN_DEFAULT_ROPE_THETA;
        assert(init_test_activations(gpu, &cfg) == 0);
        float *input = malloc(count * sizeof(float));
        float *weight = malloc((size_t)n * sizeof(float));
        float *output = malloc(count * sizeof(float));
        assert(input && weight && output);
        for (size_t i = 0; i < count; i++)
            input[i] = (float)((int)((i * 37 + r->seed * 11) % 509) - 254) / 259.0f;
        for (int i = 0; i < n; i++)
            weight[i] = r->seed ? (float)((i * 7) % 19 - 9) / 16.0f
                               : 0.25f + (float)(i % 13) / 16.0f;
        for (int raw = 0; raw < 2; raw++) {
            void *w = gpu->buffer_create(gpu->ctx, weight,
                (size_t)n * sizeof(float), raw ? -1 : BN_GGUF_TENSOR_F32, 1, n);
            assert(w);
            for (int inplace = 0; inplace < 2; inplace++) {
                BnGPUOp op = {0};
                op.op_code = BN_GPU_CODE_RMSNORM;
                op.op_kind = BN_GPU_OP_RMSNORM;
                op.W_buf = w; op.buf_in = BN_GPU_VALUE_XB;
                op.buf_out = inplace ? BN_GPU_VALUE_XB : BN_GPU_VALUE_LOGITS;
                op.flags = BN_GPU_OP_FLAG_RMSNORM_SEPARATE_SCALE;
                op.p[0] = n; op.p[1] = f32_bits(1e-6f);
                op.p[2] = f32_bits(1.0f / sqrtf((float)n));
                for (int replay = 0; replay < 2; replay++) {
                    for (int t = 0; t < r->tokens; t++) {
                        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB,
                            input + (size_t)t * n, (size_t)n * sizeof(float), 0) == 0);
                        assert(gpu->execute(gpu->ctx, &op, 1, op.buf_out,
                            output + (size_t)t * n, n) == 0);
                    }
                    assert(cuda_test_float_bits_hash(output, count) == r->hash);
                }
            }
            gpu->buffer_destroy(gpu->ctx, w);
        }
        gpu->free_activations(gpu->ctx);
        free(output); free(weight); free(input);
    }
    printf("CUDA separately scaled RMSNorm reference PASSED\n");
}

static void run_scaled_rmsnorm_batch_reference_cases(BnGPUBackend *gpu) {
    assert(bn_gpu_backend_can_rmsnorm_scaled_batch(gpu));
    for (size_t c = 0; c < sizeof(cuda_scaled_rms_reference) /
                                  sizeof(cuda_scaled_rms_reference[0]); c++) {
        const BnCudaScaledRmsReference *r = &cuda_scaled_rms_reference[c];
        int n = r->dim;
        size_t count = (size_t)n * r->tokens;
        float *input = malloc((count + 2) * sizeof(float));
        float *output = malloc((count + 2) * sizeof(float));
        float *weight = malloc((size_t)n * sizeof(float));
        assert(input && output && weight);
        for (int i = 0; i < n; i++)
            weight[i] = r->seed ? (float)((i * 7) % 19 - 9) / 16.0f
                               : 0.25f + (float)(i % 13) / 16.0f;
        for (int raw = 0; raw < 2; raw++) {
            void *w = gpu->buffer_create(gpu->ctx, weight,
                (size_t)n * sizeof(float), raw ? -1 : BN_GGUF_TENSOR_F32, 1, n);
            assert(w);
            for (int alias = 0; alias < 2; alias++) {
                for (int replay = 0; replay < 2; replay++) {
                    input[0] = input[count + 1] = output[0] = output[count + 1] = 123.0f;
                    for (size_t i = 0; i < count; i++)
                        input[i + 1] = (float)((int)((i * 37 + r->seed * 11) % 509) - 254) / 259.0f;
                    float *out = alias ? input + 1 : output + 1;
                    assert(bn_gpu_backend_rmsnorm_scaled_batch(gpu, out, w,
                        input + 1, r->tokens, n, 1e-6f,
                        1.0f / sqrtf((float)n)) == 0);
                    assert(cuda_test_float_bits_hash(out, count) == r->hash);
                    assert(input[0] == 123.0f && input[count + 1] == 123.0f);
                    assert(output[0] == 123.0f && output[count + 1] == 123.0f);
                }
            }
            gpu->buffer_destroy(gpu->ctx, w);
        }
        free(weight); free(output); free(input);
    }
    float x[2] = {1.0f, -2.0f};
    float guard[4] = {123.0f, 7.0f, 8.0f, 123.0f};
    const float saved[4] = {123.0f, 7.0f, 8.0f, 123.0f};
    BnGPUBackend absent = {0};
    void *norm = gpu->buffer_create(gpu->ctx, x, sizeof(x), BN_GGUF_TENSOR_F32, 1, 2);
    void *short_norm = gpu->buffer_create(gpu->ctx, x, sizeof(float), BN_GGUF_TENSOR_F32, 1, 1);
    assert(norm && short_norm);
    assert(bn_gpu_backend_rmsnorm_scaled_batch(NULL, guard+1, norm, x, 1, 2, 1e-6f, 1.0f) == -1);
    assert(bn_gpu_backend_rmsnorm_scaled_batch(&absent, guard+1, norm, x, 1, 2, 1e-6f, 1.0f) == -1);
    assert(bn_gpu_backend_rmsnorm_scaled_batch(gpu, guard+1, NULL, x, 1, 2, 1e-6f, 1.0f) == -1);
    assert(bn_gpu_backend_rmsnorm_scaled_batch(gpu, guard+1, short_norm, x, 1, 2, 1e-6f, 1.0f) == -1);
    assert(bn_gpu_backend_rmsnorm_scaled_batch(gpu, guard+1, norm, x, 0, 2, 1e-6f, 1.0f) == -1);
    assert(bn_gpu_backend_rmsnorm_scaled_batch(gpu, guard+1, norm, x, 1, 0, 1e-6f, 1.0f) == -1);
    assert(bn_gpu_backend_rmsnorm_scaled_batch(gpu, guard+1, norm, x, 1, 2, NAN, 1.0f) == -1);
    assert(bn_gpu_backend_rmsnorm_scaled_batch(gpu, guard+1, norm, x, 1, 2, 1e-6f, INFINITY) == -1);
    assert(memcmp(guard, saved, sizeof(guard)) == 0);
    gpu->buffer_destroy(gpu->ctx, short_norm);
    gpu->buffer_destroy(gpu->ctx, norm);
    printf("CUDA scaled RMSNorm batch reference PASSED\n");
}

static void run_rmsnorm_width_cases(BnGPUBackend *gpu) {
    const int widths[] = {32, 1000, 1024, 2560, 5376};
    for (size_t c = 0; c < sizeof(widths) / sizeof(widths[0]); c++) {
        int n = widths[c];
        BnConfig cfg = {0};
        cfg.dim = n; cfg.hidden_dim = n; cfg.n_layers = 1;
        cfg.n_heads = 1; cfg.n_kv_heads = 1; cfg.head_size = 2;
        cfg.kv_dim = 2; cfg.kv_mul = 1; cfg.seq_len = 2;
        cfg.vocab_size = n; cfg.rope_theta = BN_DEFAULT_ROPE_THETA;
        assert(init_test_activations(gpu, &cfg) == 0);
        float *input = malloc((size_t)n * sizeof(float));
        float *weight = malloc((size_t)n * sizeof(float));
        float *output = malloc((size_t)n * sizeof(float));
        assert(input && weight && output);
        double sum = 0.0;
        double reference_sum = 0.0;
        for (int i = 0; i < n; i++) {
            input[i] = sinf((float)(i * 17 + 3)) * (1 + i % 11);
            weight[i] = 0.5f + (float)(i % 13) / 16.0f;
            sum += (double)input[i] * input[i];
            reference_sum += (double)(input[i] * input[i]);
        }
        double scale = 1.0 / sqrt(sum / n + 1e-6);
        float reference_scale = 1.0f /
            sqrtf((float)(reference_sum / (double)n) + 1e-6f);
        void *w = gpu->buffer_create(gpu->ctx, weight,
            (size_t)n * sizeof(float), BN_GGUF_TENSOR_F32, 1, n);
        assert(w);
        for (int reference = 0; reference < 2; reference++) {
            for (int inplace = 0; inplace < 2; inplace++) {
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB,
                    input, (size_t)n * sizeof(float), 0) == 0);
                BnGPUOp op = {0};
                op.op_code = BN_GPU_CODE_RMSNORM;
                op.W_buf = w; op.buf_in = BN_GPU_VALUE_XB;
                op.buf_out = inplace ? BN_GPU_VALUE_XB : BN_GPU_VALUE_LOGITS;
                op.p[0] = n; op.p[1] = f32_bits(1e-6f);
                op.flags = reference ?
                    BN_GPU_OP_FLAG_RMSNORM_REFERENCE_ORDER : 0;
                assert(gpu->execute(gpu->ctx, &op, 1,
                                    op.buf_out, output, n) == 0);
                for (int i = 0; i < n; i++) {
                    float reference_expected =
                        input[i] * reference_scale * weight[i];
                    double expected = reference ? reference_expected :
                        input[i] * scale * weight[i];
                    assert(isfinite(output[i]));
                    if (reference)
                        assert(output[i] == reference_expected);
                    else
                        assert(fabs(output[i] - expected) <
                               1e-6 * (1 + fabs(expected)));
                }
            }
        }
        gpu->buffer_destroy(gpu->ctx, w);
        gpu->free_activations(gpu->ctx);
        free(output); free(weight); free(input);
    }
}

static void run_per_head_rmsnorm_case(BnGPUBackend *gpu) {
    BnConfig cfg = {0};
    cfg.dim = 8;
    cfg.hidden_dim = 8;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_kv_heads = 2;
    cfg.vocab_size = 16;
    cfg.seq_len = 8;
    cfg.rope_theta = BN_DEFAULT_ROPE_THETA;
    cfg.head_size = 4;
    cfg.kv_dim = 8;
    cfg.kv_mul = 1;
    assert(init_test_activations(gpu, &cfg) == 0);

    float in[8] = { 1.0f, -2.0f, 3.0f, -4.0f, 2.0f, 1.0f, -1.0f, 0.5f };
    float weight[4] = { 1.0f, 0.5f, 1.5f, -2.0f };
    float out[8] = { 0 };
    float ref[8] = { 0 };
    for (int h = 0; h < 2; h++) {
        float ss = 0.0f;
        for (int i = 0; i < 4; i++)
            ss += in[h * 4 + i] * in[h * 4 + i];
        float scale = 1.0f / sqrtf(ss / 4.0f + 1e-5f);
        for (int i = 0; i < 4; i++)
            ref[h * 4 + i] = in[h * 4 + i] * scale * weight[i];
    }

    void *w = gpu->buffer_create(gpu->ctx, weight, sizeof(weight),
                                 BN_GGUF_TENSOR_F32, 1, 4);
    assert(w != NULL);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_Q, in,
                                 sizeof(in), 0) == 0);
    BnGPUOp op = {0};
    op.op_code = BN_GPU_CODE_PER_HEAD_RMSNORM;
    op.W_buf = w;
    op.buf_in = BN_GPU_VALUE_Q;
    op.rows = 2;
    op.p[0] = 4;
    op.p[1] = f32_bits(1e-5f);
    op.p[2] = 0;
    assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_Q, out, 8) == 0);
    expect_close(out, ref, 8);

    gpu->buffer_destroy(gpu->ctx, w);
    gpu->free_activations(gpu->ctx);
}

static void run_gated_q_utility_case(BnGPUBackend *gpu) {
    BnConfig cfg = {0};
    cfg.dim = 8;
    cfg.hidden_dim = 8;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_kv_heads = 2;
    cfg.vocab_size = 16;
    cfg.seq_len = 8;
    cfg.rope_theta = BN_DEFAULT_ROPE_THETA;
    cfg.head_size = 4;
    cfg.kv_dim = 8;
    cfg.kv_mul = 1;
    assert(init_test_activations(gpu, &cfg) == 0);

    float qkv[16] = {
        1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 1.0f, 2.0f,
        5.0f, 6.0f, 7.0f, 8.0f, -2.0f, -1.0f, 0.5f, 1.5f,
    };
    float q_ref[8] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
    };
    float q_out[8] = {0};
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_QKV, qkv,
                                 sizeof(qkv), 0) == 0);
    BnGPUOp deint = {0};
    deint.op_code = BN_GPU_CODE_DEINTERLEAVE_Q;
    deint.buf_in = BN_GPU_VALUE_QKV;
    deint.buf_out = BN_GPU_VALUE_Q;
    deint.p[0] = 8;
    deint.p[1] = 4;
    assert(gpu->execute(gpu->ctx, &deint, 1, BN_GPU_VALUE_Q, q_out, 8) == 0);
    expect_close(q_out, q_ref, 8);

    float attn[8] = { 2.0f, -4.0f, 1.0f, 0.5f, -1.5f, 3.0f, 0.25f, -2.0f };
    float gated_ref[8] = {0};
    for (int h = 0; h < 2; h++) {
        for (int d = 0; d < 4; d++) {
            int i = h * 4 + d;
            float gate = qkv[h * 8 + 4 + d];
            gated_ref[i] = attn[i] / (1.0f + expf(-gate));
        }
    }
    float gated_out[8] = {0};
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, attn,
                                 sizeof(attn), 0) == 0);
    BnGPUOp gate = {0};
    gate.op_code = BN_GPU_CODE_SIGMOID_GATE;
    gate.buf_in = BN_GPU_VALUE_XB;
    gate.buf_aux = BN_GPU_VALUE_QKV;
    gate.p[0] = 8;
    gate.p[1] = 4;
    assert(gpu->execute(gpu->ctx, &gate, 1, BN_GPU_VALUE_XB,
                        gated_out, 8) == 0);
    expect_close(gated_out, gated_ref, 8);

    gpu->free_activations(gpu->ctx);
}

static void run_vector_attention_reference_case(BnGPUBackend *gpu) {
    enum { heads = 16, kv_heads = 2, width = 256, capacity = 256,
           q_count = heads * width, kv_count = capacity * kv_heads * width };
    /* Exact output fingerprints from llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e,
     * CUDA sm_120a, flash_attn_ext_vec, F16 KV padded to 256 keys.
     * Hash each F32 bit pattern as a word; no host byte-order dependency.
     * Cases straddle both warp-group and online-rescaling boundaries. */
    const int lengths[] = {1, 32, 128, 129, 256};
    const uint64_t expected[] = {UINT64_C(0xf58b37f34d0cbbc5), UINT64_C(0x37f04e38807126bd), UINT64_C(0xa98e93d005a3ef09), UINT64_C(0x47a697c0e546926e), UINT64_C(0xeeb4b7031d09f24d)};
    BnConfig cfg = {0};
    cfg.dim = cfg.hidden_dim = q_count; cfg.vocab_size = 32;
    cfg.n_layers = 1; cfg.seq_len = capacity; cfg.n_heads = heads;
    cfg.head_size = width; cfg.kv_dim = kv_heads * width;
    cfg.kv_f16 = 1; cfg.rope_theta = 10000.0f;
    assert(init_test_activations(gpu, &cfg) == 0);
    float q[q_count], got[q_count];
    uint16_t *k = malloc(kv_count * sizeof(*k));
    uint16_t *v = malloc(kv_count * sizeof(*v));
    assert(k && v);
    for (int i = 0; i < q_count; i++) q[i] = sinf((float)(i * 17 + 6)) * 1.31f;
    for (int i = 0; i < kv_count; i++) {
        k[i] = bn_fp32_to_fp16(sinf((float)(i * 11 + 12)) * 1.7f);
        v[i] = bn_fp32_to_fp16(cosf((float)(i * 7 + 8)) * 0.73f);
    }
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_Q, q, sizeof(q), 0) == 0);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_KEY_CACHE,
                                 k, kv_count * sizeof(*k), 0) == 0);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_VALUE_CACHE,
                                 v, kv_count * sizeof(*v), 0) == 0);
    for (size_t c = 0; c < sizeof(lengths) / sizeof(lengths[0]); c++) {
        BnGPUOp op = {0};
        op.op_code = BN_GPU_CODE_FLASH_ATTN;
        op.buf_in = BN_GPU_VALUE_Q; op.buf_out = BN_GPU_VALUE_XB;
        op.p[0] = heads; op.p[1] = width; op.p[2] = lengths[c];
        op.p[3] = heads / kv_heads; op.p[4] = kv_heads * width;
        op.p[5] = capacity;
        float scale = 1.0f / 16.0f;
        memcpy(&op.p[7], &scale, sizeof(scale));
        assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_XB, got, q_count) == 0);
        uint64_t hash = UINT64_C(14695981039346656037);
        for (int i = 0; i < q_count; i++) {
            uint32_t bits;
            memcpy(&bits, &got[i], sizeof(bits));
            hash = (hash ^ bits) * UINT64_C(1099511628211);
        }
        if (hash != expected[c])
            fprintf(stderr, "CUDA vector attention reference mismatch: keys=%d\n", lengths[c]);
        assert(hash == expected[c]);
    }
    free(k); free(v);
    gpu->free_activations(gpu->ctx);
}

static void run_q8_prepared_reference_case(void) {
    const int widths[] = {256, 512, 1024, 2048, 4096, 8192};
    /* llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e CUDA MMVQ,
     * exact F32-word fingerprints. The two widest cases also distinguish
     * fast input division from correctly rounded division at quantization. */
    const uint64_t expected[] = {
        UINT64_C(0x23c76f2c7706ce54), UINT64_C(0xf8d1f14e67c39f1e),
        UINT64_C(0x55aa860539687c7f), UINT64_C(0x0277588be044bb7d),
        UINT64_C(0x3225690918c5ef2a), UINT64_C(0x64cff01e9cfc8f7d)
    };
    BnBackendRuntimePolicy policy = {0};
    assert(bn_backend_runtime_policy_set(&policy,
        "BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT", "1", 1) == 0);
    BnGPUBackend *gpu = bn_gpu_cuda_create_with_policy(&policy);
    assert(gpu);
    bn_backend_runtime_policy_free(&policy);
    for (size_t c = 0; c < sizeof(widths) / sizeof(widths[0]); c++) {
        const int cols = widths[c], rows = 17;
        size_t count = (size_t)rows * cols / 32;
        BnBlockQ8_0 *weights = malloc(count * sizeof(*weights));
        float *input = malloc((size_t)cols * sizeof(*input));
        float got[17];
        assert(weights && input);
        for (size_t b = 0; b < count; b++) {
            weights[b].d = (uint16_t)(0x2c00 + (b % 17) * 37);
            for (int i = 0; i < 32; i++)
                weights[b].qs[i] = (int8_t)((int)((b * 17 + i * 13) % 255) - 127);
        }
        for (int i = 0; i < cols; i++)
            input[i] = (float)((i * 37) % 257 - 128) * 0.00390625f;
        void *buffer = gpu->buffer_create(gpu->ctx, weights,
            count * sizeof(*weights), BN_GGUF_TENSOR_Q8_0, rows, cols);
        assert(buffer);
        BnConfig cfg = {0};
        cfg.dim = cols; cfg.hidden_dim = rows; cfg.vocab_size = rows;
        cfg.n_layers = 1; cfg.seq_len = 1; cfg.n_heads = 1;
        cfg.head_size = cfg.kv_dim = 2; cfg.rope_theta = 10000.0f;
        assert(init_test_activations(gpu, &cfg) == 0);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB,
                                     input, (size_t)cols * sizeof(float), 0) == 0);
        for (int split = 0; split < 2; split++) {
            BnGPUOp op = {0};
            op.op_kind = BN_GPU_OP_MATVEC;
            op.W_buf = buffer; op.type = BN_GGUF_TENSOR_Q8_0;
            op.buf_in = BN_GPU_VALUE_XB;
            if (split) {
                op.op_code = BN_GPU_CODE_MATVEC_SPLIT;
                op.buf_out = BN_GPU_VALUE_XB2; op.buf_aux = BN_GPU_VALUE_HB;
                op.p[0] = rows; op.p[1] = cols; op.p[2] = 7;
                assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_XB2, got, 7) == 0);
                assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_HB,
                                             got + 7, 10 * sizeof(float), 0) == 0);
            } else {
                op.op_code = BN_GPU_CODE_MATVEC;
                op.rows = rows; op.cols = cols;
                op.buf_out = BN_GPU_VALUE_LOGITS; op.buf_aux = -1;
                assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_LOGITS, got, rows) == 0);
            }
            uint64_t hash = cuda_test_float_bits_hash(got, rows);
            if (hash != expected[c])
                fprintf(stderr, "CUDA Q8 prepared mismatch: cols=%d split=%d\n", cols, split);
            assert(hash == expected[c]);
        }
        gpu->free_activations(gpu->ctx);
        gpu->buffer_destroy(gpu->ctx, buffer);
        free(input); free(weights);
    }
    bn_gpu_cuda_destroy(gpu);
}

static void run_cuda_execution_policy_isolation(void) {
    enum { cols = 2048, rows = 17, blocks = rows * cols / 32 };
    BnBlockQ8_0 *weights = malloc(blocks * sizeof(*weights));
    float input[cols], got[rows], disabled_output[rows];
    assert(weights);
    for (int b = 0; b < blocks; b++) {
        weights[b].d = (uint16_t)(0x2c00 + (b % 17) * 37);
        for (int i = 0; i < 32; i++)
            weights[b].qs[i] = (int8_t)(((b * 17 + i * 13) % 255) - 127);
    }
    for (int i = 0; i < cols; i++) input[i] = (float)((i * 37) % 257 - 128) / 256.0f;
    BnConfig cfg = {0};
    cfg.dim = cols; cfg.hidden_dim = rows; cfg.vocab_size = rows;
    cfg.n_layers = cfg.seq_len = cfg.n_heads = 1;
    cfg.head_size = cfg.kv_dim = 2; cfg.rope_theta = 10000.0f;
    for (int reverse = 0; reverse < 2; reverse++) {
        BnGPUBackend *backends[2];
        void *buffers[2];
        for (int j = 0; j < 2; j++) {
            int enabled = j ^ reverse;
            BnBackendRuntimePolicy policy = {0};
            if (enabled)
                assert(bn_backend_runtime_policy_set(&policy,
                    "BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT", "1", 1) == 0);
            else
                assert(bn_backend_runtime_policy_set(&policy,
                    "BN_CUDA_DISABLE_NATIVE_QUANT_PREPARED_INPUT", "1", 1) == 0);
            backends[enabled] = bn_gpu_cuda_create_with_policy(&policy);
            assert(backends[enabled]);
            bn_backend_runtime_policy_free(&policy);
            BnGPUBackend *gpu = backends[enabled];
            buffers[enabled] = gpu->buffer_create(gpu->ctx, weights, sizeof(*weights) * blocks,
                                                   BN_GGUF_TENSOR_Q8_0, rows, cols);
            assert(buffers[enabled]);
            assert(init_test_activations(gpu, &cfg) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, input, sizeof(input), 0) == 0);
        }
        for (int step = 0; step < 4; step++) {
            int enabled = (step & 1) ^ reverse;
            BnGPUBackend *gpu = backends[enabled];
            BnGPUOp op = {0};
            op.op_code = BN_GPU_CODE_MATVEC; op.op_kind = BN_GPU_OP_MATVEC;
            op.W_buf = buffers[enabled]; op.type = BN_GGUF_TENSOR_Q8_0;
            op.rows = rows; op.cols = cols; op.buf_in = BN_GPU_VALUE_XB;
            op.buf_out = BN_GPU_VALUE_HB; op.buf_aux = -1;
            assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_HB, got, rows) == 0);
            uint64_t hash = cuda_test_float_bits_hash(got, rows);
            if (enabled) {
                assert(hash == UINT64_C(0x0277588be044bb7d));
            } else {
                assert(hash != UINT64_C(0x0277588be044bb7d));
                if (step < 2) memcpy(disabled_output, got, sizeof(got));
                else assert(memcmp(disabled_output, got, sizeof(got)) == 0);
            }
        }
        for (int j = 0; j < 2; j++) {
            backends[j]->buffer_destroy(backends[j]->ctx, buffers[j]);
            bn_gpu_cuda_destroy(backends[j]);
        }
    }
    free(weights);
    printf("CUDA execution policy isolation PASSED\n");
}

static void run_cuda_diagnostics_isolation(void) {
    for (int backend = 0; backend < 2; backend++) {
        BnBackendRuntimePolicy policy = {0};
        const char *flags[] = {"BN_CUDA_DUMP_OPS", "BN_CUDA_PROFILE",
            "BN_CUDA_PROFILE_SHAPES", "BN_CUDA_PROFILE_WALL",
            "BN_CUDA_PROFILE_WALL_EVERY"};
        for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); i++)
            assert(bn_backend_runtime_policy_set(&policy, flags[i], "1", 1) == 0);
        BnGPUBackend *gpu = bn_gpu_cuda_create_with_policy(&policy);
        assert(gpu);
        bn_backend_runtime_policy_free(&policy);
        BnConfig cfg = {0};
        cfg.dim = cfg.hidden_dim = cfg.vocab_size = 4;
        cfg.n_layers = cfg.seq_len = cfg.n_heads = 1;
        cfg.head_size = cfg.kv_dim = 2; cfg.rope_theta = 10000.0f;
        assert(init_test_activations(gpu, &cfg) == 0);
        float input[] = {1, -2, 3, -4}, got[4];
        float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        void *weight = gpu->buffer_create(gpu->ctx, identity, sizeof(identity), BN_GGUF_TENSOR_F32, 4, 4);
        assert(weight);
        BnGPUOp op = {0};
        op.op_code = BN_GPU_CODE_MATVEC; op.op_kind = BN_GPU_OP_MATVEC;
        op.W_buf = weight; op.type = BN_GGUF_TENSOR_F32; op.rows = op.cols = 4;
        op.buf_in = BN_GPU_VALUE_HB; op.buf_out = BN_GPU_VALUE_XB; op.buf_aux = -1;
        op.p[0] = 4;
        for (int call = 0; call < 2; call++) {
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB, input, sizeof(input), 0) == 0);
            assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_XB, got, 4) == 0);
            for (int i = 0; i < 4; i++) assert(got[i] == input[i]);
        }
        gpu->buffer_destroy(gpu->ctx, weight);
        bn_gpu_cuda_destroy(gpu);
    }
}

static void run_q8_small_batch_reference_case(BnGPUBackend *gpu) {
    const int widths[] = {256, 4096, 8192};
    /* Full-output F32-word hashes from llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e,
     * CUDA Q8 MMVQ, token counts 1..8. Covers the 4-to-2 logical-warp
     * transition, quantization boundaries and a partial output-row tile. */
    const uint64_t expected[3][8] = {
        {UINT64_C(0x23c76f2c7706ce54), UINT64_C(0xd581717350a9ab4e), UINT64_C(0xf74c6c50ce485c8a), UINT64_C(0x540e946827d31658), UINT64_C(0x6db798ea25d37bf0), UINT64_C(0x76262888fdef2e12), UINT64_C(0x364a5d307df224ca), UINT64_C(0x7ee6a9781ec66319)},
        {UINT64_C(0x3225690918c5ef2a), UINT64_C(0x19316f62b88436d9), UINT64_C(0xb233c1a9a4ae3e5b), UINT64_C(0x904022ea36004b37), UINT64_C(0x99a137428724ea09), UINT64_C(0x8c2778dc216be477), UINT64_C(0x607421d5a0ab3dbc), UINT64_C(0x59e1144c37657841)},
        {UINT64_C(0x64cff01e9cfc8f7d), UINT64_C(0x3671ae3d58886182), UINT64_C(0x8f05722fdeed8501), UINT64_C(0xb4deea0337ee3798), UINT64_C(0xad34378c40149844), UINT64_C(0x29edbdd714a9a02e), UINT64_C(0x61629ad4251b69de), UINT64_C(0x95bdb7cbfd20406a)}
    };
    for (size_t c = 0; c < sizeof(widths) / sizeof(widths[0]); c++) {
        int cols = widths[c], rows = 17;
        size_t blocks = (size_t)rows * cols / 32;
        BnBlockQ8_0 *weights = malloc(blocks * sizeof(*weights));
        float *input = malloc((size_t)cols * 8 * sizeof(*input));
        float output[2][17 * 8 + 2];
        assert(weights && input);
        for (size_t b = 0; b < blocks; b++) {
            weights[b].d = (uint16_t)(0x2c00 + (b % 17) * 37);
            for (int i = 0; i < 32; i++)
                weights[b].qs[i] = (int8_t)((int)((b * 17 + i * 13) % 255) - 127);
        }
        for (int i = 0; i < cols * 8; i++)
            input[i] = (float)((i * 37) % 257 - 128) / 256.0f;
        void *buffer = gpu->buffer_create(gpu->ctx, weights,
            blocks * sizeof(*weights), BN_GGUF_TENSOR_Q8_0, rows, cols);
        assert(buffer);
        for (int tokens = 1; tokens <= 8; tokens++) {
            int count = rows * tokens;
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 17 * 8 + 2; j++) output[i][j] = -12345.0f;
            assert(bn_gpu_backend_matmul(gpu, output[0] + 1, buffer, input,
                                         rows, cols, tokens, BN_GGUF_TENSOR_Q8_0) == 0);
            assert(cuda_test_float_bits_hash(output[0] + 1, count) == expected[c][tokens - 1]);
            assert(output[0][0] == -12345.0f);
            for (int j = count + 1; j < 17 * 8 + 2; j++) assert(output[0][j] == -12345.0f);
            /* The optional CUDA multi-projection callback requires at least
             * two tokens; one-token matmul is covered above. */
            if (tokens == 1) continue;
            BnGPUMatvecOp ops[2] = {
                {output[0] + 1, buffer, rows, cols, BN_GGUF_TENSOR_Q8_0},
                {output[1] + 1, buffer, rows, cols, BN_GGUF_TENSOR_Q8_0}
            };
            assert(bn_gpu_backend_matmul_batch(gpu, ops, 2, input, tokens, cols) == 0);
            for (int i = 0; i < 2; i++) {
                assert(cuda_test_float_bits_hash(output[i] + 1, count) == expected[c][tokens - 1]);
                assert(output[i][0] == -12345.0f);
                for (int j = count + 1; j < 17 * 8 + 2; j++) assert(output[i][j] == -12345.0f);
            }
        }
        gpu->buffer_destroy(gpu->ctx, buffer);
        free(input); free(weights);
    }
}

static void run_buffer_upload_readiness_case(BnGPUBackend *gpu) {
    /* A returned weight handle must be ready for the nonblocking execution
     * stream. Reuse allocations and change every row so a pending default-
     * stream upload cannot be hidden by old contents or scratch allocation. */
    enum { rows = 1024, cols = 2048 };
    float *weights = malloc((size_t)rows*cols*sizeof(float));
    float input[cols], output[rows];
    assert(weights);
    for (int i = 0; i < cols; i++) input[i] = 0.5f;
    BnConfig cfg = {0};
    cfg.dim = cols; cfg.hidden_dim = cfg.vocab_size = rows;
    cfg.n_layers = cfg.seq_len = cfg.n_heads = 1;
    cfg.head_size = cfg.kv_dim = 2; cfg.rope_theta = 10000.0f;
    assert(init_test_activations(gpu, &cfg) == 0);
    for (int pass = 0; pass < 8; pass++) {
        for (int row = 0; row < rows; row++) {
            float value = (float)(1 + (row+pass)%7)/1024.0f;
            for (int col = 0; col < cols; col++) weights[(size_t)row*cols+col] = value;
        }
        void *buffer = gpu->buffer_create(gpu->ctx, weights,
            (size_t)rows*cols*sizeof(float), BN_GGUF_TENSOR_F32, rows, cols);
        assert(buffer);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, input, sizeof(input), 0) == 0);
        BnGPUOp op = {0}; op.op_code = BN_GPU_CODE_MATVEC; op.op_kind = BN_GPU_OP_MATVEC;
        op.W_buf = buffer; op.type = BN_GGUF_TENSOR_F32;
        op.rows = rows; op.cols = cols; op.buf_in = BN_GPU_VALUE_XB;
        op.buf_out = BN_GPU_VALUE_LOGITS; op.buf_aux = -1;
        assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_LOGITS, output, rows) == 0);
        for (int row = 0; row < rows; row++) assert(output[row] == (float)(1+(row+pass)%7));
        gpu->buffer_destroy(gpu->ctx, buffer);
    }
    free(weights); gpu->free_activations(gpu->ctx);
    printf("CUDA buffer upload readiness PASSED\n");
}

static void run_f32_matrix_reference_case(BnGPUBackend *gpu) {
    /* Independent llama.cpp CUDA matrix graphs, commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e, Blackwell. */
    const struct { int cols, rows, tokens; uint64_t hash; } cases[] = {
        {256, 1, 1, UINT64_C(0x2da03ad1518281f9)},
        {256, 1, 3, UINT64_C(0xa9646b2791264224)},
        {256, 1, 4, UINT64_C(0x13a9a979dd73f808)},
        {256, 1, 8, UINT64_C(0xa477aff8f950ac14)},
        {256, 1, 9, UINT64_C(0x587e7d5572ed5532)},
        {256, 1, 16, UINT64_C(0x7aa1ff9a60180c66)},
        {256, 1, 17, UINT64_C(0x275490288a8be2f4)},
        {256, 1, 129, UINT64_C(0x1df1ec29b2d18c4b)},
        {768, 17, 1, UINT64_C(0x2d189ff940826dc6)},
        {768, 17, 3, UINT64_C(0x01fade6253860e94)},
        {768, 17, 4, UINT64_C(0xdd25f33816dfd152)},
        {768, 17, 8, UINT64_C(0xedd2ddcd91c1a2c5)},
        {768, 17, 9, UINT64_C(0xe6c4342d6a039070)},
        {768, 17, 16, UINT64_C(0x2e9ecf29addfccef)},
        {768, 17, 17, UINT64_C(0x7b1a5019873f6149)},
        {768, 17, 129, UINT64_C(0xa5fb8be2d7bc4467)},
        {2048, 32, 1, UINT64_C(0xaecff47b805d8ca8)},
        {2048, 32, 3, UINT64_C(0x4b37b09c6a9c87e6)},
        {2048, 32, 4, UINT64_C(0x461e049a0d4646f3)},
        {2048, 32, 8, UINT64_C(0xe09391bd50cd95f9)},
        {2048, 32, 9, UINT64_C(0xa4e94c6361558d66)},
        {2048, 32, 16, UINT64_C(0xf0ffd05ae66500c3)},
        {2048, 32, 17, UINT64_C(0xd186c214c6017cbb)},
        {2048, 32, 129, UINT64_C(0xc83f9b559d01f7e1)},
        {4096, 129, 1, UINT64_C(0xf3207a10f2f10e7f)},
        {4096, 129, 3, UINT64_C(0xf389e00daf4d174a)},
        {4096, 129, 4, UINT64_C(0x2b7162fce9fc49ca)},
        {4096, 129, 8, UINT64_C(0x6945e692d36fc9e9)},
        {4096, 129, 9, UINT64_C(0x36e37a8b1077cabc)},
        {4096, 129, 16, UINT64_C(0x83703ac337be3714)},
        {4096, 129, 17, UINT64_C(0xfdcd33fc3e3bb0af)},
        {4096, 129, 129, UINT64_C(0x17c9ac4f30a66c95)},
        {8192, 256, 1, UINT64_C(0x7bd0c1833277bf9e)},
        {8192, 256, 3, UINT64_C(0x8b8de0ae96d38dbd)},
        {8192, 256, 4, UINT64_C(0xb596e097e7848c02)},
        {8192, 256, 8, UINT64_C(0x7469df3365b58fe0)},
        {8192, 256, 9, UINT64_C(0xbece288ba4d9b7a7)},
        {8192, 256, 16, UINT64_C(0x976d882c3eb179a1)},
        {8192, 256, 17, UINT64_C(0x39c012ae7a42dbde)},
        {8192, 256, 129, UINT64_C(0x8b996df616fe89d6)},
        {257, 33, 1, UINT64_C(0x9ad45dd7d78981fb)},
        {257, 33, 3, UINT64_C(0xd480e4af90c873f7)},
        {257, 33, 4, UINT64_C(0x70b4b46912910cb0)},
        {257, 33, 8, UINT64_C(0xe782d74ccbbafd86)},
        {257, 33, 9, UINT64_C(0x19d894ea9dd74a8c)},
        {257, 33, 16, UINT64_C(0x20ca70aa9f6a0459)},
        {257, 33, 17, UINT64_C(0x5eea3408ac10f6fb)},
        {257, 33, 129, UINT64_C(0x413c6ef9852d3f92)},
    };
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        int cols = cases[c].cols, rows = cases[c].rows, tokens = cases[c].tokens;
        size_t count = (size_t)rows*tokens;
        float *w = malloc((size_t)rows*cols*sizeof(float));
        float *input = malloc((size_t)tokens*cols*sizeof(float));
        float *out = malloc((count+2)*sizeof(float));
        assert(w && input && out);
        for (int r = 0; r < rows; r++)
            for (int d = 0; d < cols; d++)
                w[(size_t)r*cols+d] = (float)((r*13+d*17)%257-128)/257.0f;
        for (int t = 0; t < tokens; t++)
            for (int d = 0; d < cols; d++)
                input[(size_t)t*cols+d] = (float)((d*37+t*19)%251-125)/259.0f;
        void *buffer = gpu->buffer_create(gpu->ctx, w, (size_t)rows*cols*sizeof(float),
            BN_GGUF_TENSOR_F32, rows, cols);
        assert(buffer); out[0] = out[count+1] = -12345.0f;
        assert(gpu->matmul(gpu->ctx, out+1, buffer, input,
            rows, cols, tokens, BN_GGUF_TENSOR_F32) == 0);
        assert(cuda_test_float_bits_hash(out+1,count) == cases[c].hash);
        assert(out[0] == -12345.0f && out[count+1] == -12345.0f);
        if (tokens == 1 && cols % 2 == 0) {
            /* Decode must use the same independent F32 MMVF reference. */
            BnGPUActivationPlan plan = {0};
            plan.dim = plan.hb_elements = plan.xb2_elements = cols;
            plan.vocab_size = rows;
            plan.n_layers = plan.seq_len = plan.n_heads = plan.head_size = 1;
            assert(gpu->init_activations(gpu->ctx, &plan) == 0);
            assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X,
                input, (size_t)cols * sizeof(float), 0) == 0);
            BnGPUOp op = {0};
            op.op_code = BN_GPU_CODE_MATVEC;
            op.buf_in = BN_GPU_VALUE_X;
            op.buf_out = BN_GPU_VALUE_LOGITS;
            op.W_buf = buffer;
            op.type = BN_GGUF_TENSOR_F32;
            op.rows = rows;
            op.cols = cols;
            assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_LOGITS,
                out + 1, rows) == 0);
            assert(cuda_test_float_bits_hash(out + 1, count) == cases[c].hash);
            assert(out[0] == -12345.0f && out[count + 1] == -12345.0f);
            gpu->free_activations(gpu->ctx);
        }
        if (tokens > 1) {
            float *second = malloc((count+2)*sizeof(float));
            assert(second);
            second[0] = second[count+1] = -12345.0f;
            BnGPUMatvecOp ops[2] = {{0}};
            for (int i = 0; i < 2; i++) {
                ops[i].W_buf = buffer; ops[i].rows = rows; ops[i].cols = cols;
                ops[i].type = BN_GGUF_TENSOR_F32;
                ops[i].out = i ? second+1 : out+1;
            }
            assert(gpu->matmul_batch(gpu->ctx, ops, 2, input, tokens, cols) == 0);
            assert(cuda_test_float_bits_hash(out+1,count) == cases[c].hash);
            assert(cuda_test_float_bits_hash(second+1,count) == cases[c].hash);
            assert(out[0] == -12345.0f && out[count+1] == -12345.0f);
            assert(second[0] == -12345.0f && second[count+1] == -12345.0f);
            free(second);
        }
        gpu->buffer_destroy(gpu->ctx, buffer); free(w); free(input); free(out);
    }
    printf("CUDA F32 matrix reference PASSED\n");
}

static void run_f32_ffn_entry_reference_case(BnGPUBackend *gpu) {
    /* Full independent llama.cpp CUDA FFN graphs at commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e, Blackwell. Mode0 is plain,
     * mode1 includes input RMSNorm, mode2 also adds the input residual.
     * Stacked and separate projections may legitimately have different
     * reference hashes because matrix shapes select different arithmetic. */
    const struct { int dim, hidden, tokens; uint64_t hash[3][2]; } cases[] = {
        {256, 17, 1, {
            {UINT64_C(0xd6bdef37576a99c1), UINT64_C(0xd6bdef37576a99c1)},
            {UINT64_C(0xc467de7fab686635), UINT64_C(0xc467de7fab686635)},
            {UINT64_C(0xbd59088f2693b15e), UINT64_C(0xbd59088f2693b15e)},
        }},
        {256, 17, 2, {
            {UINT64_C(0x70988e8371fbc073), UINT64_C(0x70988e8371fbc073)},
            {UINT64_C(0x446b768120350483), UINT64_C(0x446b768120350483)},
            {UINT64_C(0x335f3d0b2b744ceb), UINT64_C(0x335f3d0b2b744ceb)},
        }},
        {256, 17, 3, {
            {UINT64_C(0xd72e9d1fc5d17e17), UINT64_C(0xd72e9d1fc5d17e17)},
            {UINT64_C(0x57709d2da1ea397d), UINT64_C(0x57709d2da1ea397d)},
            {UINT64_C(0x3a8e921ee5e496ba), UINT64_C(0x3a8e921ee5e496ba)},
        }},
        {256, 17, 4, {
            {UINT64_C(0x40514ad84167a9cb), UINT64_C(0x40514ad84167a9cb)},
            {UINT64_C(0x18cb2df1ef44c485), UINT64_C(0x18cb2df1ef44c485)},
            {UINT64_C(0x2677d210fb13931a), UINT64_C(0x2677d210fb13931a)},
        }},
        {256, 17, 8, {
            {UINT64_C(0x445767d463ec405c), UINT64_C(0x445767d463ec405c)},
            {UINT64_C(0x27ca156323967447), UINT64_C(0x27ca156323967447)},
            {UINT64_C(0xe875016cad14107c), UINT64_C(0xe875016cad14107c)},
        }},
        {256, 17, 9, {
            {UINT64_C(0x69586f61d1193698), UINT64_C(0x69586f61d1193698)},
            {UINT64_C(0xaed0f732a12d46a2), UINT64_C(0xaed0f732a12d46a2)},
            {UINT64_C(0x3b62c28832ed588c), UINT64_C(0x3b62c28832ed588c)},
        }},
        {256, 17, 16, {
            {UINT64_C(0xd45cd96799414e03), UINT64_C(0xd45cd96799414e03)},
            {UINT64_C(0x5dcce30113c687ce), UINT64_C(0x5dcce30113c687ce)},
            {UINT64_C(0x7a0087479b080533), UINT64_C(0x7a0087479b080533)},
        }},
        {256, 17, 17, {
            {UINT64_C(0x2694fafb972fdce5), UINT64_C(0x2694fafb972fdce5)},
            {UINT64_C(0x3b838621a8a6f4c0), UINT64_C(0x3b838621a8a6f4c0)},
            {UINT64_C(0x7c6b011b7d677509), UINT64_C(0x7c6b011b7d677509)},
        }},
        {256, 17, 129, {
            {UINT64_C(0x82776006efc0ce7e), UINT64_C(0x82776006efc0ce7e)},
            {UINT64_C(0x06a7012816c1ef9f), UINT64_C(0x06a7012816c1ef9f)},
            {UINT64_C(0xeb2ee1818373b552), UINT64_C(0xeb2ee1818373b552)},
        }},
        {2048, 32, 1, {
            {UINT64_C(0xc562a9b25f4aa705), UINT64_C(0xc562a9b25f4aa705)},
            {UINT64_C(0x10cfd9ba266105db), UINT64_C(0x10cfd9ba266105db)},
            {UINT64_C(0x621c5d9b644486b8), UINT64_C(0x621c5d9b644486b8)},
        }},
        {2048, 32, 2, {
            {UINT64_C(0x648a21ae06d26bae), UINT64_C(0x648a21ae06d26bae)},
            {UINT64_C(0xa7d652ca659e8cae), UINT64_C(0xa7d652ca659e8cae)},
            {UINT64_C(0x6dfae2011a1f1ba2), UINT64_C(0x6dfae2011a1f1ba2)},
        }},
        {2048, 32, 3, {
            {UINT64_C(0x4bd87081d2450f11), UINT64_C(0x4bd87081d2450f11)},
            {UINT64_C(0x4d9bbd9d85c34c67), UINT64_C(0x4d9bbd9d85c34c67)},
            {UINT64_C(0x1abbbd3d01d5ba65), UINT64_C(0x1abbbd3d01d5ba65)},
        }},
        {2048, 32, 4, {
            {UINT64_C(0xb0e2a3c4c614cbcd), UINT64_C(0xb0e2a3c4c614cbcd)},
            {UINT64_C(0x842a36a222585300), UINT64_C(0x842a36a222585300)},
            {UINT64_C(0xf0207aec6923189d), UINT64_C(0xf0207aec6923189d)},
        }},
        {2048, 32, 8, {
            {UINT64_C(0xcdb3746584165627), UINT64_C(0xcdb3746584165627)},
            {UINT64_C(0x77c5c2d5ba68971f), UINT64_C(0x77c5c2d5ba68971f)},
            {UINT64_C(0xcb6702c1a7f0e5bf), UINT64_C(0xcb6702c1a7f0e5bf)},
        }},
        {2048, 32, 9, {
            {UINT64_C(0x86f9ebe2d9615e52), UINT64_C(0x86f9ebe2d9615e52)},
            {UINT64_C(0x48b08191398aeae2), UINT64_C(0x48b08191398aeae2)},
            {UINT64_C(0xa461fabf343d8c78), UINT64_C(0xa461fabf343d8c78)},
        }},
        {2048, 32, 16, {
            {UINT64_C(0xf0f029862b472015), UINT64_C(0xf0f029862b472015)},
            {UINT64_C(0x829a51e26b4fde17), UINT64_C(0x829a51e26b4fde17)},
            {UINT64_C(0x3b39e798c6fa940c), UINT64_C(0x3b39e798c6fa940c)},
        }},
        {2048, 32, 17, {
            {UINT64_C(0x2f3f94e235f206d3), UINT64_C(0x2f3f94e235f206d3)},
            {UINT64_C(0x422d5626f9c81c57), UINT64_C(0x422d5626f9c81c57)},
            {UINT64_C(0xf9534a6a26fcbdf5), UINT64_C(0xf9534a6a26fcbdf5)},
        }},
        {2048, 32, 129, {
            {UINT64_C(0xa701a63b4386e6e8), UINT64_C(0x1d79a3d4918a3e35)},
            {UINT64_C(0xe170bae851d6c7cf), UINT64_C(0xec8f88743ba6b662)},
            {UINT64_C(0x39986d8ad64925e3), UINT64_C(0xed31599561ce5f57)},
        }},
        {4096, 64, 1, {
            {UINT64_C(0xc8169ad4df5ad383), UINT64_C(0xc8169ad4df5ad383)},
            {UINT64_C(0x664cfb190c3efe64), UINT64_C(0x664cfb190c3efe64)},
            {UINT64_C(0xb9dbf4f976f3e5f6), UINT64_C(0xb9dbf4f976f3e5f6)},
        }},
        {4096, 64, 2, {
            {UINT64_C(0xec31f21792901d81), UINT64_C(0xec31f21792901d81)},
            {UINT64_C(0xf16f0b0d04c73016), UINT64_C(0xf16f0b0d04c73016)},
            {UINT64_C(0x9a245907a85eea39), UINT64_C(0x9a245907a85eea39)},
        }},
        {4096, 64, 3, {
            {UINT64_C(0xb8a40dc872a41e45), UINT64_C(0xb8a40dc872a41e45)},
            {UINT64_C(0xf453b8a058ac142c), UINT64_C(0xf453b8a058ac142c)},
            {UINT64_C(0x9c48254368aa8ad4), UINT64_C(0x9c48254368aa8ad4)},
        }},
        {4096, 64, 4, {
            {UINT64_C(0x61f23bedaacdb00e), UINT64_C(0x61f23bedaacdb00e)},
            {UINT64_C(0x96e53fcb27a54c7b), UINT64_C(0x96e53fcb27a54c7b)},
            {UINT64_C(0x34c3886bbc2021d3), UINT64_C(0x34c3886bbc2021d3)},
        }},
        {4096, 64, 8, {
            {UINT64_C(0x1895ee1dfe998339), UINT64_C(0x1895ee1dfe998339)},
            {UINT64_C(0x1370d437345196e6), UINT64_C(0x1370d437345196e6)},
            {UINT64_C(0xf884a84a62000d40), UINT64_C(0xf884a84a62000d40)},
        }},
        {4096, 64, 9, {
            {UINT64_C(0x55c4e0beef379301), UINT64_C(0x55c4e0beef379301)},
            {UINT64_C(0x95a258cd0b6e8d3c), UINT64_C(0x95a258cd0b6e8d3c)},
            {UINT64_C(0xe04a8ddec3233f23), UINT64_C(0xe04a8ddec3233f23)},
        }},
        {4096, 64, 16, {
            {UINT64_C(0xc983ec5589832a86), UINT64_C(0xc983ec5589832a86)},
            {UINT64_C(0xe97db54e6e4d1991), UINT64_C(0xe97db54e6e4d1991)},
            {UINT64_C(0xac79e540c4de87dc), UINT64_C(0xac79e540c4de87dc)},
        }},
        {4096, 64, 17, {
            {UINT64_C(0xf8ed874edefd858b), UINT64_C(0xf8ed874edefd858b)},
            {UINT64_C(0xdfc4485d2599f10a), UINT64_C(0xdfc4485d2599f10a)},
            {UINT64_C(0xaf137175acac33d4), UINT64_C(0xaf137175acac33d4)},
        }},
        {4096, 64, 129, {
            {UINT64_C(0x0371d62e2521b85d), UINT64_C(0x38ed3d52adcbf9e4)},
            {UINT64_C(0x400a3c2d3fec82be), UINT64_C(0x84689f1bcb2c67ef)},
            {UINT64_C(0x23fbba709616db6b), UINT64_C(0xf17dc18f630e6bae)},
        }},
    };
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        int dim = cases[c].dim, hidden = cases[c].hidden, tokens = cases[c].tokens;
        size_t matrix = (size_t)dim*hidden, count = (size_t)dim*tokens;
        float *gate_up = malloc(2*matrix*sizeof(float));
        float *down = malloc(matrix*sizeof(float));
        float *norm = malloc(dim*sizeof(float));
        float *input = malloc(count*sizeof(float));
        float *output = malloc((count+2)*sizeof(float));
        assert(gate_up && down && norm && input && output);
        for (int r = 0; r < hidden; r++)
            for (int d = 0; d < dim; d++) {
                gate_up[(size_t)r*dim+d] = (float)((r*13+d*17)%257-128)/2048.0f;
                gate_up[matrix+(size_t)r*dim+d] = (float)((r*19+d*11)%251-125)/1024.0f;
            }
        for (int r = 0; r < dim; r++)
            for (int d = 0; d < hidden; d++)
                down[(size_t)r*hidden+d] = (float)((r*7+d*11)%113-56)/512.0f;
        for (int d = 0; d < dim; d++) norm[d] = (float)(d%7+1)/8.0f;
        for (int t = 0; t < tokens; t++)
            for (int d = 0; d < dim; d++)
                input[(size_t)t*dim+d] = (float)((t*19+d*37)%251-125)/259.0f;
        void *gate = gpu->buffer_create(gpu->ctx, gate_up, matrix*sizeof(float),
            BN_GGUF_TENSOR_F32, hidden, dim);
        void *up = gpu->buffer_create(gpu->ctx, gate_up+matrix, matrix*sizeof(float),
            BN_GGUF_TENSOR_F32, hidden, dim);
        void *stacked = gpu->buffer_create(gpu->ctx, gate_up, 2*matrix*sizeof(float),
            BN_GGUF_TENSOR_F32, 2*hidden, dim);
        void *down_buf = gpu->buffer_create(gpu->ctx, down, matrix*sizeof(float),
            BN_GGUF_TENSOR_F32, dim, hidden);
        void *norm_buf = gpu->buffer_create(gpu->ctx, norm, dim*sizeof(float),
            BN_GGUF_TENSOR_F32, 1, dim);
        assert(gate && up && stacked && down_buf && norm_buf);
        for (int mode = 0; mode < 3; mode++) {
            for (int stack = 0; stack < 2; stack++) {
                void *g = stack ? stacked : gate, *u = stack ? NULL : up;
                output[0] = output[count+1] = -12345.0f;
                int rc;
                if (mode == 0)
                    rc = gpu->dense_ffn_batch(gpu->ctx, output+1, g, u, down_buf,
                        input, tokens, dim, hidden, BN_GGUF_TENSOR_F32,
                        BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32, 0);
                else if (mode == 1)
                    rc = gpu->dense_ffn_batch_norm(gpu->ctx, output+1, g, u, down_buf,
                        norm_buf, input, tokens, dim, hidden, BN_GGUF_TENSOR_F32,
                        BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32, 0, 1e-6f);
                else
                    rc = gpu->dense_ffn_batch_norm_resid(gpu->ctx, output+1, g, u,
                        down_buf, norm_buf, input, tokens, dim, hidden,
                        BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32,
                        0, 1e-6f);
                assert(rc == 0);
                assert(cuda_test_float_bits_hash(output+1,count) == cases[c].hash[mode][stack]);
                assert(output[0] == -12345.0f && output[count+1] == -12345.0f);
            }
        }
        gpu->buffer_destroy(gpu->ctx, gate); gpu->buffer_destroy(gpu->ctx, up);
        gpu->buffer_destroy(gpu->ctx, stacked); gpu->buffer_destroy(gpu->ctx, down_buf);
        gpu->buffer_destroy(gpu->ctx, norm_buf);
        free(gate_up); free(down); free(norm); free(input); free(output);
    }
    printf("CUDA F32 FFN entry reference PASSED\n");
}

#include "cuda_standalone_ffn_fixture.h"
#include "cuda_standalone_ffn_reference.h"
static void run_standalone_ffn_reference_case(BnGPUBackend *gpu) {
    float frequencies[16] = {0};
    BnGPUActivationPlan plan = {0};
    plan.dim = plan.xb2_elements = plan.hb_elements = plan.vocab_size = 256;
    plan.n_layers = plan.attention_layer_count = plan.n_heads = plan.seq_len = 1;
    plan.kv_dim = plan.head_size = 32;
    plan.rope_frequencies = frequencies; plan.rope_frequency_count = 16;
    assert(gpu->init_activations(gpu->ctx, &plan) == 0);
    BnGPUOp graph[32] = {{0}};
    float marker = 1.25f;
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, &marker, sizeof(marker), 0) == 0);
    for (int i = 0; i < 32; i++) {
        graph[i].op_code = BN_GPU_CODE_COPY;
        graph[i].buf_in = BN_GPU_VALUE_X; graph[i].buf_out = BN_GPU_VALUE_SCRATCH;
        graph[i].p[2] = 1;
    }
    for (size_t c = 0; c < sizeof(cuda_standalone_ffn_reference) /
                                  sizeof(cuda_standalone_ffn_reference[0]); c++) {
        const BnCudaStandaloneFFNReference *r = &cuda_standalone_ffn_reference[c];
        int dim = r->dim, hidden = r->hidden, nt = r->tokens;
        size_t matrix = (size_t)dim * hidden, count = (size_t)dim * nt;
        size_t gb = ffn_bytes(r->gate_type, matrix);
        size_t ub = ffn_bytes(r->up_type, matrix);
        size_t db = ffn_bytes(r->down_type, matrix);
        char *gate = malloc(gb * 2), *up = malloc(ub), *down = malloc(db);
        float *norm = malloc((size_t)dim * sizeof(float));
        float *input = malloc(count * sizeof(float));
        float *output = malloc((count + 2) * sizeof(float));
        assert(gate && up && down && norm && input && output);
        ffn_weights(gate, r->gate_type, matrix, 0);
        ffn_weights(up, r->up_type, matrix, 23);
        ffn_weights(down, r->down_type, matrix, 7);
        if (r->stacked) { assert(gb == ub); memcpy(gate + gb, up, ub); }
        for (int d = 0; d < dim; d++) norm[d] = (float)(d % 7 + 1) / 8.0f;
        for (int t = 0; t < nt; t++)
            for (int d = 0; d < dim; d++)
                input[(size_t)t * dim + d] = (float)((t * 19 + d * 37) % 251 - 125) / 259.0f;
        for (int quant_only = 0; quant_only < 2; quant_only++) {
            void *(*create)(void *, const void *, size_t, int, int, int) =
                quant_only ? gpu->buffer_create_quant_only : gpu->buffer_create;
            void *bg = create(gpu->ctx, gate, gb * (r->stacked ? 2 : 1),
                r->gate_type, hidden * (r->stacked ? 2 : 1), dim);
            void *bu = r->stacked ? NULL : create(gpu->ctx, up, ub, r->up_type, hidden, dim);
            void *bd = create(gpu->ctx, down, db, r->down_type, dim, hidden);
            void *bn = gpu->buffer_create(gpu->ctx, norm, (size_t)dim * sizeof(float),
                BN_GGUF_TENSOR_F32, 1, dim);
            assert(bg && bd && bn && (r->stacked || bu));
            for (int alias = 0; alias < 2; alias++) {
                output[0] = output[count + 1] = -12345.0f;
                memcpy(output + 1, input, count * sizeof(float));
                const float *x = alias ? output + 1 : input;
                /* Alternate direct calls and calls following a decode graph. */
                if (alias)
                    assert(gpu->execute(gpu->ctx, graph, 32, -1, NULL, 0) == 0);
                int rc;
                if (r->mode == 0)
                    rc = gpu->dense_ffn_batch(gpu->ctx, output + 1, bg, bu, bd,
                        x, nt, dim, hidden, r->gate_type, r->up_type, r->down_type, 0);
                else if (r->mode == 1)
                    rc = gpu->dense_ffn_batch_norm(gpu->ctx, output + 1, bg, bu, bd, bn,
                        x, nt, dim, hidden, r->gate_type, r->up_type, r->down_type, 0, 1e-6f);
                else
                    rc = gpu->dense_ffn_batch_norm_resid(gpu->ctx, output + 1, bg, bu, bd, bn,
                        x, nt, dim, hidden, r->gate_type, r->up_type, r->down_type, 0, 1e-6f);
                assert(rc == 0);
                uint64_t got = cuda_test_float_bits_hash(output + 1, count);
                if (got != r->hash) fprintf(stderr,
                    "standalone FFN case=%zu quant_only=%d alias=%d hash=%016llx expected=%016llx\n",
                    c, quant_only, alias, (unsigned long long)got, (unsigned long long)r->hash);
                assert(got == r->hash);
                assert(output[0] == -12345.0f && output[count + 1] == -12345.0f);
            }
            gpu->buffer_destroy(gpu->ctx, bg);
            if (bu) gpu->buffer_destroy(gpu->ctx, bu);
            gpu->buffer_destroy(gpu->ctx, bd); gpu->buffer_destroy(gpu->ctx, bn);
        }
        free(gate); free(up); free(down); free(norm); free(input); free(output);
    }
    gpu->free_activations(gpu->ctx);
    printf("CUDA standalone quant FFN reference PASSED\n");
}

static void run_moe_residual_order_reference_case(BnGPUBackend *gpu) {
    /* Independent ggml CUDA RMSNorm -> routed Q8_0 FFN + gated F32 shared
     * FFN -> residual graphs, llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e.
     * Both expert branches are nonzero: zero routed weights hide reordering
     * of the residual and shared branch. */
    const struct { int tokens; uint64_t hash; } cases[] = {
        {2, UINT64_C(0x29044c849c12ccb5)},
        {7, UINT64_C(0x1ea536235237f8b1)},
        {9, UINT64_C(0xea9f45dd7c5b6d49)},
        {16, UINT64_C(0x116de52b8aa10bec)},
        {17, UINT64_C(0x75c627b8db9abf01)},
    };
    const int dim = 2048, hidden = 32;
    const int f32 = BN_GGUF_TENSOR_F32, q8 = BN_GGUF_TENSOR_Q8_0;
    size_t matrix_values = (size_t)dim * hidden;
    float *norm = malloc((size_t)dim * sizeof(float));
    float *shared_gate = calloc(matrix_values, sizeof(float));
    float *shared_down = calloc(matrix_values, sizeof(float));
    float *shared_weight = malloc((size_t)dim * sizeof(float));
    float *router = calloc((size_t)dim, sizeof(float));
    BnBlockQ8_0 *gate = calloc(matrix_values / 32, sizeof(*gate));
    BnBlockQ8_0 *down = calloc(matrix_values / 32, sizeof(*down));
    assert(norm && shared_gate && shared_down && shared_weight && router &&
           gate && down);
    for (int i = 0; i < dim; i++) {
        norm[i] = 0.5f + (float)(i % 19) / 16.0f;
        shared_weight[i] = (float)(i % 7 - 3) / 2048.0f;
        shared_down[(size_t)i * hidden] = 0.375f;
        down[i].d = bn_fp32_to_fp16(0.125f);
        down[i].qs[0] = 1;
    }
    shared_gate[0] = 1.0f;
    gate[0].d = bn_fp32_to_fp16(1.0f);
    gate[0].qs[0] = 1;
    void *buffers[] = {
        gpu->buffer_create(gpu->ctx, gate,
            matrix_values / 32 * sizeof(*gate), q8, hidden, dim),
        gpu->buffer_create(gpu->ctx, down,
            matrix_values / 32 * sizeof(*down), q8, dim, hidden),
        gpu->buffer_create(gpu->ctx, shared_gate,
            matrix_values * sizeof(float), f32, hidden, dim),
        gpu->buffer_create(gpu->ctx, shared_down,
            matrix_values * sizeof(float), f32, dim, hidden),
        gpu->buffer_create(gpu->ctx, shared_weight,
            (size_t)dim * sizeof(float), f32, 1, dim),
        gpu->buffer_create(gpu->ctx, norm,
            (size_t)dim * sizeof(float), f32, 1, dim),
        gpu->buffer_create(gpu->ctx, router,
            (size_t)dim * sizeof(float), f32, 1, dim),
    };
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++)
        assert(buffers[i]);
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int nt = cases[c].tokens;
        size_t count = (size_t)nt * dim;
        float *input = malloc(count * sizeof(float));
        float *output = malloc((count + 2) * sizeof(float));
        assert(input && output);
        for (size_t i = 0; i < count; i++)
            input[i] = (float)((int)((i * 13 + 5) % 97) - 48) / 64.0f;
        output[0] = output[count + 1] = 123.0f;
        for (int alias = 0; alias < 2; alias++) {
            float *out = alias ? input : output + 1;
            assert(gpu->moe_route_routed_ffn_batch_norm_resid(
                gpu->ctx, out, buffers[6], buffers[0], buffers[0], buffers[1],
                buffers[2], buffers[2], buffers[3], buffers[4], buffers[5],
                input, nt, dim, hidden, 1, 1, q8, q8, q8, 0,
                hidden, f32, f32, f32, 1e-6f, 1, 1.0f) == 0);
            assert(cuda_test_float_bits_hash(out, count) == cases[c].hash);
            assert(output[0] == 123.0f && output[count + 1] == 123.0f);
        }
        free(output);
        free(input);
    }
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++)
        gpu->buffer_destroy(gpu->ctx, buffers[i]);
    free(down); free(gate); free(router); free(shared_weight);
    free(shared_down); free(shared_gate); free(norm);
    printf("CUDA MoE residual order reference cases PASSED\n");
}

static void run_shared_batch_reference_case(BnGPUBackend *shared_gpu) {
    /* Independent full FFN/shared-gate graphs at the same reference commit.
     * Combined cases use a zero routed expert and nonzero shared expert.
     * Mode2 routes just one token with zero weight while the shared expert
     * processes the full batch. Fresh contexts expose undersized scratch
     * hidden by earlier allocations. */
    const struct { int dim, tokens, mode; uint64_t hash; } cases[] = {
        {256, 1, 0, UINT64_C(0x82893c5d03fd5725)},
        {256, 2, 0, UINT64_C(0xb1d000b539e8db25)},
        {256, 3, 0, UINT64_C(0x0f85b383b3170f25)},
        {256, 4, 0, UINT64_C(0x2f5094f4116b3725)},
        {256, 5, 0, UINT64_C(0xb70e4b3231c3cf25)},
        {256, 8, 0, UINT64_C(0x20b326f3d9c2de25)},
        {256, 9, 0, UINT64_C(0xe87b4fa380f1e125)},
        {256, 16, 0, UINT64_C(0xaa06ef959b89e025)},
        {256, 17, 0, UINT64_C(0xfaad73ca11c89f25)},
        {256, 32, 0, UINT64_C(0x78ededb721ba8725)},
        {256, 129, 0, UINT64_C(0xe3de2b1f1bb9f325)},
        {256, 512, 0, UINT64_C(0x92949ff4f604e525)},
        {768, 1, 0, UINT64_C(0x63d0649f6c49ee25)},
        {768, 2, 0, UINT64_C(0x32796109f1eb5a25)},
        {768, 3, 0, UINT64_C(0xaf054d1ed41da625)},
        {768, 4, 0, UINT64_C(0x4b461c693ce8ef25)},
        {768, 5, 0, UINT64_C(0xad3f38b5bc04ff25)},
        {768, 8, 0, UINT64_C(0x7a987140e8716f25)},
        {768, 9, 0, UINT64_C(0xe0276d0a69710e25)},
        {768, 16, 0, UINT64_C(0xaadefb09c73d9925)},
        {768, 17, 0, UINT64_C(0xe977c823dd97f525)},
        {768, 32, 0, UINT64_C(0x95f37f51ef631925)},
        {768, 129, 0, UINT64_C(0x907b8fa9cd58a525)},
        {768, 512, 0, UINT64_C(0x31ddc33d68d25025)},
        {2048, 1, 0, UINT64_C(0x9d99f13b62f8bb25)},
        {2048, 2, 0, UINT64_C(0xa8e8c014999a2325)},
        {2048, 3, 0, UINT64_C(0x4013bd1892bf4325)},
        {2048, 4, 0, UINT64_C(0xf6d248404b31fb25)},
        {2048, 5, 0, UINT64_C(0x399fb3a8d2bd1b25)},
        {2048, 8, 0, UINT64_C(0xbbf08f8f646c4b25)},
        {2048, 9, 0, UINT64_C(0xe9e30c0a05a2fb25)},
        {2048, 16, 0, UINT64_C(0xa99df2bfadfabb25)},
        {2048, 17, 0, UINT64_C(0x52f99806a45b1325)},
        {2048, 32, 0, UINT64_C(0xce2cc6e3bb1f4325)},
        {2048, 129, 0, UINT64_C(0xb4de1036f1bd4b25)},
        {2048, 512, 0, UINT64_C(0xb54b4c0129f72b25)},
        {8192, 1, 0, UINT64_C(0x659f7b11311ca325)},
        {8192, 2, 0, UINT64_C(0x3412a536082f2325)},
        {8192, 3, 0, UINT64_C(0xb798b6804095a325)},
        {8192, 4, 0, UINT64_C(0xc4e53dcdfdbfc325)},
        {8192, 5, 0, UINT64_C(0xe095d97af8026325)},
        {8192, 8, 0, UINT64_C(0x3be4d30c32ab6325)},
        {8192, 9, 0, UINT64_C(0x11896df6b7042325)},
        {8192, 16, 0, UINT64_C(0xf29ab014ffc1a325)},
        {8192, 17, 0, UINT64_C(0xd266773d59e22325)},
        {8192, 32, 0, UINT64_C(0xc6ddc87582f84325)},
        {8192, 129, 0, UINT64_C(0x983bf87b2f692325)},
        {8192, 512, 0, UINT64_C(0x1c2fbcbf43226325)},
        {2048, 1, 1, UINT64_C(0xa22f77fd00150818)},
        {2048, 3, 1, UINT64_C(0x06a1cc59b701624f)},
        {2048, 4, 1, UINT64_C(0xcd5f616e8e9298b2)},
        {2048, 8, 1, UINT64_C(0xf2ede69e7c1eeb88)},
        {2048, 9, 1, UINT64_C(0x4715cab71da077db)},
        {2048, 16, 1, UINT64_C(0xe70fa1e17e48cea1)},
        {2048, 17, 1, UINT64_C(0x8a9d0b8982539bae)},
        {2048, 129, 1, UINT64_C(0x1293d94c351a37cb)},
        {2048, 512, 1, UINT64_C(0x8e8de6cbb62f27f1)},
        {2048, 3, 2, UINT64_C(0xa382fe8ad1b37325)},
        {2048, 4, 2, UINT64_C(0x8134cf67decf1b25)},
        {2048, 9, 2, UINT64_C(0x3d3b1ef18de06b25)},
        {2048, 129, 2, UINT64_C(0x9b70b623429c8b25)},
        {2048, 512, 2, UINT64_C(0x5fcbef52877d9325)},
        {8192, 17, 2, UINT64_C(0xe09841631d972325)},
    };
    enum { hidden = 32 };
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        BnGPUBackend *gpu = cases[c].mode ? bn_gpu_cuda_create() : shared_gpu;
        assert(gpu);
        int dim = cases[c].dim, tokens = cases[c].tokens;
        size_t count = (size_t)dim*tokens, matrix_count = (size_t)dim*hidden;
        float *gu = calloc(matrix_count, sizeof(float));
        float *down = calloc(matrix_count, sizeof(float));
        float *gate = malloc(dim*sizeof(float));
        float *input = malloc(count*sizeof(float));
        float *output = malloc(count*sizeof(float));
        assert(gu && down && gate && input && output);
        gu[0] = 1.0f;
        for (int d = 0; d < dim; d++) {
            down[(size_t)d*hidden] = 1.0f;
            gate[d] = (float)((d*17)%257-128)/257.0f;
        }
        for (int t = 0; t < tokens; t++)
            for (int d = 0; d < dim; d++)
                input[(size_t)t*dim+d] = d ? (float)((d*37+t*19)%251-125)/259.0f : 1.0f;
        void *bg = gpu->buffer_create(gpu->ctx, gu, matrix_count*sizeof(float),
            BN_GGUF_TENSOR_F32, hidden, dim);
        void *bd = gpu->buffer_create(gpu->ctx, down, matrix_count*sizeof(float),
            BN_GGUF_TENSOR_F32, dim, hidden);
        void *bs = gpu->buffer_create(gpu->ctx, gate, dim*sizeof(float),
            BN_GGUF_TENSOR_F32, 1, dim);
        assert(bg && bd && bs);
        if (cases[c].mode != 1) {
            int *ids = malloc(tokens*sizeof(int));
            float *weights = malloc(tokens*sizeof(float));
            assert(ids && weights);
            for (int t = 0; t < tokens; t++) { ids[t] = t; weights[t] = cases[c].mode == 2 ? 0.0f : 1.0f; }
            BnGPUMoEPrefillExpert expert = {0};
            expert.gate_buf = expert.up_buf = bg; expert.down_buf = bd;
            int offset = 0, expert_count = cases[c].mode == 2 ? 1 : tokens;
            assert(gpu->moe_ffn_batch(gpu->ctx, output, &expert, 1,
                &offset, &expert_count, ids, weights, input, tokens, dim, hidden,
                BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32, 0,
                bg, bg, bd, bs, hidden, BN_GGUF_TENSOR_F32,
                BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32) == 0);
            free(ids); free(weights);
        } else {
            size_t blocks = matrix_count/32;
            BnBlockQ8_0 *zero = calloc(blocks, sizeof(*zero));
            float *router = calloc(dim, sizeof(float));
            float *norm = malloc(dim*sizeof(float));
            assert(zero && router && norm);
            for (int d = 0; d < dim; d++) norm[d] = 1.0f;
            void *qg = gpu->buffer_create(gpu->ctx, zero, blocks*sizeof(*zero),
                BN_GGUF_TENSOR_Q8_0, hidden, dim);
            void *qd = gpu->buffer_create(gpu->ctx, zero, blocks*sizeof(*zero),
                BN_GGUF_TENSOR_Q8_0, dim, hidden);
            void *br = gpu->buffer_create(gpu->ctx, router, dim*sizeof(float),
                BN_GGUF_TENSOR_F32, 1, dim);
            void *bn = gpu->buffer_create(gpu->ctx, norm, dim*sizeof(float),
                BN_GGUF_TENSOR_F32, 1, dim);
            assert(qg && qd && br && bn);
            assert(gpu->moe_route_routed_ffn_batch_norm_resid(gpu->ctx, output,
                br, qg, qg, qd, bg, bg, bd, bs, bn, input, tokens, dim, hidden, 1, 1,
                BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0, 0,
                hidden, BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32,
                1e-6f, 1, 1.0f) == 0);
            gpu->buffer_destroy(gpu->ctx, qg); gpu->buffer_destroy(gpu->ctx, qd);
            gpu->buffer_destroy(gpu->ctx, br); gpu->buffer_destroy(gpu->ctx, bn);
            free(zero); free(router); free(norm);
        }
        uint64_t actual = cuda_test_float_bits_hash(output,count);
        if (actual != cases[c].hash)
            fprintf(stderr, "shared batch case=%zu dim=%d tokens=%d mode=%d hash=%016llx expected=%016llx first=%g\n",
                c, dim, tokens, cases[c].mode, (unsigned long long)actual,
                (unsigned long long)cases[c].hash, output[0]);
        assert(actual == cases[c].hash);
        gpu->buffer_destroy(gpu->ctx, bg); gpu->buffer_destroy(gpu->ctx, bd);
        gpu->buffer_destroy(gpu->ctx, bs);
        free(gu); free(down); free(gate); free(input); free(output);
        if (cases[c].mode) bn_gpu_cuda_destroy(gpu);
    }
    printf("CUDA shared batch reference PASSED\n");
}

static void run_shared_gate_reference_case(BnGPUBackend *gpu) {
    /* Independent llama.cpp CUDA graphs at commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e, Blackwell. Full-output hashes
     * cover reset/complement, residual add, fused RMSNorm and large widths. */
    const struct { int dim; uint64_t hash[3][4]; } cases[] = {
        {256, {
            {UINT64_C(0xb1bac076e3256124), UINT64_C(0x038b8aae730e97d0), UINT64_C(0xd199f30f02d98702), UINT64_C(0xca553dbc4f65fdca)},
            {UINT64_C(0xc83cef11488b167c), UINT64_C(0xee12ba55b143eac0), UINT64_C(0x2791f5e6b67cc358), UINT64_C(0x83899a18ab08e4fd)},
            {UINT64_C(0xd3c709e16c9bf291), UINT64_C(0x24cfd3538fd7207c), UINT64_C(0x435a60371bb95c2d), UINT64_C(0x74560b10d54f2d0d)},
        }},
        {768, {
            {UINT64_C(0xdcaf509f4be07f85), UINT64_C(0x67377dcd8cd85fea), UINT64_C(0xaa13a6e0d405fbb6), UINT64_C(0xda87b16ed4ee9341)},
            {UINT64_C(0xdf947dce37b27f4c), UINT64_C(0xac8100f3e83c453a), UINT64_C(0xc27f92486ee8c41c), UINT64_C(0x6cdaf94f2878dbf5)},
            {UINT64_C(0xf18d62759af285d7), UINT64_C(0xeb2faa7af8dfff7c), UINT64_C(0x7d9108b7778475fd), UINT64_C(0x13ae2f480453c427)},
        }},
        {1536, {
            {UINT64_C(0xf8b3c764499f1067), UINT64_C(0xdda25c6a375a4f57), UINT64_C(0x8cb96a00ed291560), UINT64_C(0x2f137b7ecad51537)},
            {UINT64_C(0x0277d8462c73d0fa), UINT64_C(0x61764d67a2fb84f7), UINT64_C(0xea324b19f1410f20), UINT64_C(0x397886fed3eeb575)},
            {UINT64_C(0x09a65482910c7df9), UINT64_C(0xb8f9bf89e14f07aa), UINT64_C(0x90faf57588f5937f), UINT64_C(0xb9d26c1d188d2b1d)},
        }},
        {2048, {
            {UINT64_C(0x7b0e338564431a7a), UINT64_C(0x54f5b4d5cb9446a9), UINT64_C(0xf583cdcc98a26eb8), UINT64_C(0x10244376074291da)},
            {UINT64_C(0xde16320c28fa9a72), UINT64_C(0x1c55eaffa15d1a0e), UINT64_C(0xc927f38b9b693d83), UINT64_C(0x24fcad651e89b979)},
            {UINT64_C(0xdcbde2f9fa118fc1), UINT64_C(0xd2de0c7161b5106e), UINT64_C(0x28ee97077f7ef86b), UINT64_C(0x493b261e41292c09)},
        }},
        {4096, {
            {UINT64_C(0xd8ef68026d32a3f5), UINT64_C(0xe60739bdff1550aa), UINT64_C(0x85d485db0255b878), UINT64_C(0x7e4b360e8886598b)},
            {UINT64_C(0xbb3e2b542865745a), UINT64_C(0xee7ce6646f0ed679), UINT64_C(0xa104b8d0ed933f39), UINT64_C(0xeb5e4ad941b12961)},
            {UINT64_C(0x008d370410beffd7), UINT64_C(0x266c66a56aabaf27), UINT64_C(0xa4e2f8927e56b6a9), UINT64_C(0x9ce63376253df019)},
        }},
        {8192, {
            {UINT64_C(0x21af599afdfcbc21), UINT64_C(0x2304a591684ba9d6), UINT64_C(0x045839b34550f9c4), UINT64_C(0xb56b34dbaa37beb1)},
            {UINT64_C(0x3c18393d1a3d74a2), UINT64_C(0xd257521f1d72856e), UINT64_C(0xde75eedb4804d088), UINT64_C(0x7437688969be6be0)},
            {UINT64_C(0x2c8a5ade7834b119), UINT64_C(0xca5ffab9b60c3800), UINT64_C(0xb2c8331d20a5f1f4), UINT64_C(0x30d26cdabb51e118)},
        }},
        {16384, {
            {UINT64_C(0x861dd893a1be5cb8), UINT64_C(0xfe9fdae782550e3a), UINT64_C(0x902c6594c9c352a7), UINT64_C(0xdb37f874675010b4)},
            {UINT64_C(0x96023aa2cb5b975c), UINT64_C(0xbaef81ec21c11d5c), UINT64_C(0x95c9c5f06883384e), UINT64_C(0x4206285b7a392d04)},
            {UINT64_C(0x11bdd22274f67e11), UINT64_C(0x9145198f80b0e27f), UINT64_C(0xa352de2c26a47da4), UINT64_C(0x642d3ab0b46b653b)},
        }},
    };
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        int n = cases[c].dim;
        float *data = malloc((size_t)n*8*sizeof(float));
        assert(data);
        float *gate = data, *input = gate+n, *branch = input+n, *base = branch+n;
        float *residual = base+n, *norm = residual+n, *output = norm+n;
        for (int i = 0; i < n; i++) {
            gate[i] = (float)((i*17)%257-128)/257.0f;
            input[i] = (float)((i*37)%251-125)/259.0f;
            branch[i] = (float)((i*19)%113-56)/31.0f;
            base[i] = (float)((i*23)%127-63)/61.0f;
            residual[i] = (float)((i*7)%113-56)/61.0f;
            norm[i] = (float)(i%7+1)/8.0f;
        }
        BnGPUActivationPlan plan = {0}; float rope = 1.0f;
        plan.dim = plan.vocab_size = plan.xb2_elements = plan.hb_elements = n;
        plan.n_layers = plan.n_heads = plan.attention_layer_count = 1;
        plan.seq_len = plan.kv_dim = plan.head_size = 2;
        plan.rope_frequencies = &rope; plan.rope_frequency_count = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        void *gate_buffer = gpu->buffer_create(gpu->ctx, gate, n*sizeof(float),
            BN_GGUF_TENSOR_F32, 1, n);
        void *norm_buffer = gpu->buffer_create(gpu->ctx, norm, n*sizeof(float),
            BN_GGUF_TENSOR_F32, 1, n);
        assert(gate_buffer && norm_buffer);
        for (int mode = 0; mode < 3; mode++) {
            for (int flags = 0; flags < 4; flags++) {
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, input, n*sizeof(float), 0) == 0);
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB2, branch, n*sizeof(float), 0) == 0);
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_HB, base, n*sizeof(float), 0) == 0);
                assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, residual, n*sizeof(float), 0) == 0);
                BnGPUOp ops[2] = {{0}};
                ops[0].op_code = BN_GPU_CODE_WEIGHTED_ADD_SIGMOID;
                ops[0].W_buf = gate_buffer;
                ops[0].buf_in = BN_GPU_VALUE_HB; ops[0].buf_aux = BN_GPU_VALUE_XB2;
                ops[0].p[0] = ops[0].p[3] = n;
                ops[0].p[2] = flags/2; ops[0].p[4] = flags%2;
                ops[1].op_code = mode == 2 ? BN_GPU_CODE_RESIDUAL_RMSNORM : BN_GPU_CODE_RESIDUAL_ADD;
                ops[1].buf_in = BN_GPU_VALUE_X; ops[1].buf_aux = BN_GPU_VALUE_HB;
                ops[1].buf_out = BN_GPU_VALUE_HB; ops[1].p[0] = n;
                ops[1].W_buf = norm_buffer;
                float eps = 1e-6f; memcpy(&ops[1].p[1], &eps, sizeof(eps));
                assert(gpu->execute(gpu->ctx, ops, mode ? 2 : 1,
                    mode == 1 ? BN_GPU_VALUE_X : BN_GPU_VALUE_HB, output, n) == 0);
                assert(cuda_test_float_bits_hash(output,n) == cases[c].hash[mode][flags]);
            }
        }
        gpu->buffer_destroy(gpu->ctx, gate_buffer); gpu->buffer_destroy(gpu->ctx, norm_buffer);
        gpu->free_activations(gpu->ctx); free(data);
    }
    printf("CUDA shared gate reference PASSED\n");
}

static void run_router512_reference_cases(BnGPUBackend *gpu) {
#ifdef BN_CUDA_MXFP4_SM120
    enum { dim = 32, experts = 512, batch = 8, max_k = 16 };
    float router[experts * dim], input[batch * dim];
    for (size_t c = 0; c < sizeof(cuda_router512_reference) / sizeof(cuda_router512_reference[0]); c++) {
        const BnCudaRouter512Reference *r = &cuda_router512_reference[c];
        memset(router, 0, sizeof(router));
        memset(input, 0, sizeof(input));
        for (int i = 0; i < experts; i++)
            router[i * dim] = sinf((float)(i * 13 + r->seed)) * r->magnitude;
        for (int t = 0; t < batch; t++) input[t * dim] = 1.0f;
        BnGPUActivationPlan plan = {0};
        float rope = 1.0f;
        plan.dim = dim;
        plan.n_layers = plan.n_heads = plan.seq_len = plan.attention_layer_count = 1;
        plan.kv_dim = plan.head_size = 2;
        plan.vocab_size = plan.hb_elements = experts;
        plan.xb2_elements = 2 * max_k;
        plan.rope_frequencies = &rope;
        plan.rope_frequency_count = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        void *buffer = gpu->buffer_create(gpu->ctx, router, sizeof(router),
            BN_GGUF_TENSOR_F32, experts, dim);
        assert(buffer);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, input,
            dim * sizeof(float), 0) == 0);
        BnGPUOp ops[16] = {0};
        for (int i = 0; i < 16; i++) {
            ops[i].op_code = BN_GPU_CODE_MOE_ROUTE_TOPK;
            ops[i].W_buf = buffer;
            ops[i].cols = dim;
            ops[i].buf_in = BN_GPU_VALUE_XB;
            ops[i].buf_out = BN_GPU_VALUE_XB2;
            ops[i].buf_aux = BN_GPU_VALUE_HB;
            ops[i].p[0] = experts;
            ops[i].p[1] = r->k;
            ops[i].p[2] = f32_bits(1.0f);
            if (!r->normalize) ops[i].flags |= BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM;
        }
        for (int pass = 0; r->nt == 1 && pass < 3; pass++) {
            float output[2 * max_k + 2];
            output[0] = output[2 * r->k + 1] = 12345.0f;
            assert(gpu->execute(gpu->ctx, ops, pass ? 16 : 1,
                BN_GPU_VALUE_XB2, output + 1, 2 * r->k) == 0);
            if (cuda_test_float_bits_hash(output + 1, r->k) != r->weights)
                fprintf(stderr, "router512 case=%zu pass=%d weight mismatch\n", c, pass);
            assert(cuda_test_float_bits_hash(output + 1, r->k) == r->weights);
            uint64_t hash = UINT64_C(14695981039346656037);
            for (int i = 0; i < r->k; i++)
                hash = (hash ^ (uint32_t)(int)output[1 + r->k + i]) * UINT64_C(1099511628211);
            assert(hash == r->ids);
            assert(output[0] == 12345.0f && output[2 * r->k + 1] == 12345.0f);
        }
        {
            int nt = r->nt;
            int ids[batch * max_k + 2];
            float weights[batch * max_k + 2];
            ids[0] = ids[nt * r->k + 1] = -12345;
            weights[0] = weights[nt * r->k + 1] = 12345.0f;
            assert(gpu->moe_route_batch(gpu->ctx, ids + 1, weights + 1,
                buffer, input, nt, dim, experts, r->k, r->normalize, 1.0f) == 0);
            if (cuda_test_float_bits_hash(weights + 1, nt * r->k) != r->weights)
                fprintf(stderr, "router512 case=%zu nt=%d weight mismatch\n", c, nt);
            assert(cuda_test_float_bits_hash(weights + 1, nt * r->k) == r->weights);
            uint64_t hash = UINT64_C(14695981039346656037);
            for (int i = 0; i < nt * r->k; i++)
                hash = (hash ^ (uint32_t)ids[1 + i]) * UINT64_C(1099511628211);
            assert(hash == r->ids);
            assert(ids[0] == -12345 && ids[nt * r->k + 1] == -12345);
            assert(weights[0] == 12345.0f && weights[nt * r->k + 1] == 12345.0f);
        }
        gpu->buffer_destroy(gpu->ctx, buffer);
    }
    gpu->free_activations(gpu->ctx);
    printf("CUDA router512 reference PASSED: 480 cases, 17316 weights and IDs\n");
#else
    (void)gpu;
    printf("CUDA router512 reference skipped: requires SM120 reference build\n");
#endif
}

static void run_router_reference_case(BnGPUBackend *gpu) {
    /* Independent llama.cpp CUDA softmax/top-k/get_rows consumer graphs,
     * commit3d3d7c81813067fc8c185da017e0af03b4269b1e. Near-equal logits
     * round to tied probabilities: raw-logit top-k selects different IDs.
     * Mode2 puts every selected expert in the same normalization lane. */
    const struct {
        int experts, k, mode, normalize; float scale;
        uint64_t logits, weights, ids;
    } cases[] = {
        {256,8,0,1,1,UINT64_C(0x61fee5ab7c03b725),UINT64_C(0x002fece5483bb655),UINT64_C(0x21c2051d967d0de9)},
        {256,8,1,1,1,UINT64_C(0xad4b41cd86c3ac67),UINT64_C(0xdf3bcf5e681a39c5),UINT64_C(0xdb46b02ad604020d)},
        {256,8,2,1,1,UINT64_C(0x0b43248f89abb725),UINT64_C(0x083907dc83f44d31),UINT64_C(0xa65989e17d9e2dcd)},
        {128,16,0,1,1,UINT64_C(0xa23838fa0834ed25),UINT64_C(0x3607060fb49df1c9),UINT64_C(0x079258e711ea1f35)},
        {128,8,3,1,1,UINT64_C(0x8421ae126c7ced25),UINT64_C(0xdf3bcf5e681a39c5),UINT64_C(0xa4dc49e2b28ecb7d)},
        {32,8,1,1,1,UINT64_C(0xa72b7652a2db5996),UINT64_C(0xdf3bcf5e681a39c5),UINT64_C(0xdb46b02ad604020d)},
        {64,8,0,1,1,UINT64_C(0x3b59daa5be750825),UINT64_C(0x6c78c1d29ee525d2),UINT64_C(0xaef512a06b6b856d)},
        {256,8,4,1,1,UINT64_C(0x43894bf39ae900b5),UINT64_C(0x58f01d8bfc3e5c9e),UINT64_C(0xea9db483d9a48e99)},
        {256,8,0,0,1,UINT64_C(0x61fee5ab7c03b725),UINT64_C(0xc28416ddffc5a24e),UINT64_C(0x21c2051d967d0de9)},
        {256,8,1,0,2.5f,UINT64_C(0xad4b41cd86c3ac67),UINT64_C(0x744e80ad335a39c5),UINT64_C(0xdb46b02ad604020d)},
        {256,8,2,1,2.5f,UINT64_C(0x0b43248f89abb725),UINT64_C(0xc6e06b3f7ec75409),UINT64_C(0xa65989e17d9e2dcd)}
    };
    enum { dim = 2048, batch = 3 };
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        int n = cases[c].experts, k = cases[c].k, mode = cases[c].mode;
        float *router = calloc((size_t)n*dim, sizeof(float));
        float *input = calloc(batch*dim, sizeof(float));
        assert(router && input);
        for (int e = 0; e < n; e++) {
            router[(size_t)e*dim+e] = 1.0f;
            input[e] = mode == 0 ? (float)((e*37)%257-128)/64.0f :
                       mode == 1 ? (float)(e%13)*1e-8f :
                       mode == 2 ? (e%32 == 3 ? (float)e/128.0f : -8.0f) : 0.0f;
        }
        if (mode == 4) {
            for (int e = 0; e < n; e++)
                for (int d = 0; d < dim; d++)
                    router[(size_t)e*dim+d] = (float)((e*13+d*17)%257-128)/256.0f;
            for (int d = 0; d < dim; d++) input[d] = (float)((d*37)%257-128)/259.0f;
        }
        for (int t = 1; t < batch; t++) memcpy(input+t*dim, input, dim*sizeof(float));
        BnGPUActivationPlan plan = {0}; float rope = 1.0f;
        plan.dim = dim; plan.n_layers = plan.n_heads = plan.seq_len = plan.attention_layer_count = 1;
        plan.kv_dim = plan.head_size = 2; plan.vocab_size = plan.hb_elements = n;
        plan.xb2_elements = 2*k; plan.rope_frequencies = &rope; plan.rope_frequency_count = 1;
        assert(gpu->init_activations(gpu->ctx, &plan) == 0);
        void *buffer = gpu->buffer_create(gpu->ctx, router, (size_t)n*dim*sizeof(float),
                                          BN_GGUF_TENSOR_F32, n, dim);
        assert(buffer);
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, input, dim*sizeof(float), 0) == 0);
        BnGPUOp op = {0}; op.op_code = BN_GPU_CODE_MOE_ROUTE_TOPK;
        op.W_buf = buffer; op.cols = dim; op.buf_in = BN_GPU_VALUE_XB;
        op.buf_out = BN_GPU_VALUE_XB2; op.buf_aux = BN_GPU_VALUE_HB;
        op.p[0] = n; op.p[1] = k; memcpy(&op.p[2], &cases[c].scale, sizeof(float));
        if (!cases[c].normalize) op.flags |= BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM;
        float output[2*BN_MAX_MOE_K], logits[256];
        assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_XB2, output, 2*k) == 0);
        assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_HB, logits, n*sizeof(float), 0) == 0);
        uint64_t logits_hash = cuda_test_float_bits_hash(logits,n);
        if (logits_hash != cases[c].logits)
            fprintf(stderr, "router reference case=%zu mode=%d logits=%016llx expected=%016llx first=%g\n",
                    c, mode, (unsigned long long)logits_hash,
                    (unsigned long long)cases[c].logits, logits[0]);
        assert(logits_hash == cases[c].logits);
        assert(cuda_test_float_bits_hash(output,k) == cases[c].weights);
        uint64_t id_hash = UINT64_C(14695981039346656037);
        for (int i = 0; i < k; i++)
            id_hash = (id_hash ^ (uint32_t)(int)output[k+i])*UINT64_C(1099511628211);
        assert(id_hash == cases[c].ids);
        /* Check all MMVF batch sizes, including the dense dot products. */
        for (int tokens = 1; tokens <= batch; tokens++) {
            int ids[batch*BN_MAX_MOE_K]; float weights[batch*BN_MAX_MOE_K];
            assert(gpu->moe_route_batch(gpu->ctx, ids, weights, buffer, input,
                tokens, dim, n, k, cases[c].normalize, cases[c].scale) == 0);
            for (int t = 0; t < tokens; t++) {
                assert(cuda_test_float_bits_hash(weights+t*k,k) == cases[c].weights);
                id_hash = UINT64_C(14695981039346656037);
                for (int i = 0; i < k; i++)
                    id_hash = (id_hash ^ (uint32_t)ids[t*k+i])*UINT64_C(1099511628211);
                assert(id_hash == cases[c].ids);
            }
        }
        gpu->buffer_destroy(gpu->ctx, buffer); free(input); free(router);
    }
    gpu->free_activations(gpu->ctx);
    printf("CUDA router softmax/top-k reference PASSED\n");
}

static void run_router_mmvf_reference_case(BnGPUBackend *gpu) {
    /* Independent CUDA graphs from the same llama.cpp commit as the router
     * test above. Widths exercise logical thread-count and FMA-loop changes. */
    const struct { int dim; uint64_t weights, ids; } cases[] = {
        {256, UINT64_C(0xabf4fffbbac3e434), UINT64_C(0x6e2b7c6dbddcfff0)},
        {512, UINT64_C(0x4aa95ca4220599c6), UINT64_C(0xf1663800aa7ae0d5)},
        {768, UINT64_C(0x7dd67ea4fa8254c8), UINT64_C(0xa422fe04759ec6ed)},
        {1024, UINT64_C(0xe067fb7d262bca21), UINT64_C(0x8698909f446e5959)},
        {1536, UINT64_C(0x3189ff0b644ee237), UINT64_C(0x243ecf433bc0b63a)},
        {2048, UINT64_C(0x58f01d8bfc3e5c9e), UINT64_C(0xea9db483d9a48e99)},
        {2560, UINT64_C(0xcc39f480a3b709a7), UINT64_C(0xea9db483d9a48e99)},
        {4096, UINT64_C(0x8d9d893007a9ca0e), UINT64_C(0xe704f345c2ff0656)},
        {8192, UINT64_C(0x9a5cfbaf6406048d), UINT64_C(0x5ac968ea6854e01c)}
    };
    enum { experts = 256, k = 8, batch = 3 };
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        int dim = cases[c].dim;
        float *router = malloc((size_t)experts*dim*sizeof(float));
        float *input = malloc((size_t)batch*dim*sizeof(float));
        assert(router && input);
        for (int e = 0; e < experts; e++)
            for (int d = 0; d < dim; d++)
                router[(size_t)e*dim+d] = (float)((e*13+d*17)%257-128)/256.0f;
        for (int t = 0; t < batch; t++)
            for (int d = 0; d < dim; d++)
                input[(size_t)t*dim+d] = (float)((d*37)%257-128)/259.0f;
        void *buffer = gpu->buffer_create(gpu->ctx, router,
            (size_t)experts*dim*sizeof(float), BN_GGUF_TENSOR_F32, experts, dim);
        assert(buffer);
        for (int tokens = 1; tokens <= batch; tokens++) {
            int ids[batch*k]; float weights[batch*k];
            assert(gpu->moe_route_batch(gpu->ctx, ids, weights, buffer,
                input, tokens, dim, experts, k, 1, 1.0f) == 0);
            for (int t = 0; t < tokens; t++) {
                assert(cuda_test_float_bits_hash(weights+t*k,k) == cases[c].weights);
                uint64_t hash = UINT64_C(14695981039346656037);
                for (int i = 0; i < k; i++)
                    hash = (hash ^ (uint32_t)ids[t*k+i])*UINT64_C(1099511628211);
                assert(hash == cases[c].ids);
            }
        }
        gpu->buffer_destroy(gpu->ctx, buffer); free(router); free(input);
    }
    printf("CUDA router MMVF reference PASSED\n");
}

static void run_router_matrix_reference_case(BnGPUBackend *gpu) {
    /* Independent llama.cpp CUDA consumer graphs, commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e, Blackwell. Cover MMVF,
     * native TF32, cuBLAS, odd-width fallback and changing expert counts.
     * Each token has a different input; hashes cover all weights and IDs. */
    const struct { int dim, experts, tokens; uint64_t weights, ids; } cases[] = {
        {256, 32, 4, UINT64_C(0x3f771c4c2f2ae9ab), UINT64_C(0x209b8a9a8102400c)},
        {256, 32, 16, UINT64_C(0xd2db485e307b1f6e), UINT64_C(0x8cf6ab2f456cf8e2)},
        {256, 32, 17, UINT64_C(0x99f1b4de8ab7f07a), UINT64_C(0xd85afa39aca7c7df)},
        {768, 64, 4, UINT64_C(0x1536fa32bb5c3516), UINT64_C(0x4a9e809301dbc6b6)},
        {768, 64, 16, UINT64_C(0x882ae9acb65df16d), UINT64_C(0x45b5d3bb0dbcad52)},
        {768, 64, 17, UINT64_C(0x3a9a9c0c8fd01b2e), UINT64_C(0xab79e79126d3ff87)},
        {2048, 256, 3, UINT64_C(0x413535b8dae3ab71), UINT64_C(0x635b4a2cb3ec4220)},
        {2048, 256, 4, UINT64_C(0x8cb4ce379fba2020), UINT64_C(0xf966263ce9347c50)},
        {2048, 256, 5, UINT64_C(0x958c4ad6e7ae8877), UINT64_C(0xa1813d8305123f60)},
        {2048, 256, 8, UINT64_C(0x9c7d7f45df94c6d2), UINT64_C(0x225c9b748d3fd394)},
        {2048, 256, 9, UINT64_C(0xf066bb7ac3361981), UINT64_C(0x75e2757fda38067b)},
        {2048, 256, 16, UINT64_C(0xd3ebc80149af4835), UINT64_C(0x2514e7b40bb7068f)},
        {2048, 256, 17, UINT64_C(0x2d99bb4d6e09adee), UINT64_C(0x7b9d31baf634465f)},
        {2048, 256, 32, UINT64_C(0x8b593709aece1af7), UINT64_C(0xdd61ee1d6a8c48b0)},
        {2048, 256, 129, UINT64_C(0x43cfa2f6b7159ad7), UINT64_C(0x6007fc11e981a38e)},
        {4096, 128, 4, UINT64_C(0xf319be31a94d1af2), UINT64_C(0xccd9b326c67c7175)},
        {4096, 128, 16, UINT64_C(0x032f0bbb38b6757f), UINT64_C(0x09d1b2622768be13)},
        {4096, 128, 17, UINT64_C(0x3f2ec1a7e56ce086), UINT64_C(0x50dd38fde07061a1)},
        {8192, 256, 4, UINT64_C(0x50c808fde2432e0e), UINT64_C(0x5ee33e42310f79cd)},
        {8192, 256, 16, UINT64_C(0x9e098994ad99421b), UINT64_C(0xeb092e12e06da05a)},
        {8192, 256, 17, UINT64_C(0xa046ec0150a84c44), UINT64_C(0x86dab6f25a7dd2d7)},
        {257, 256, 4, UINT64_C(0x3f5602e656133be7), UINT64_C(0x3a328860b3eb35c1)},
        {257, 256, 16, UINT64_C(0x4a346b35f2b6a8c4), UINT64_C(0xa414fbff17da184b)},
        {257, 256, 17, UINT64_C(0x42d37a91a2394bd7), UINT64_C(0x131aa6de5efceb97)},
        {2048, 256, 6, UINT64_C(0x229d93e7e840c031), UINT64_C(0xe3fd53ec0e1efed4)},
        {2048, 256, 7, UINT64_C(0x648388b53a380db5), UINT64_C(0x0d546cc1a8c151b8)},
        {2048, 256, 10, UINT64_C(0x33c6e5ccb8220a5c), UINT64_C(0x82bb158ee6fd66ce)},
        {2048, 256, 11, UINT64_C(0x439264d643ad5cff), UINT64_C(0x66a535c23fe52af1)},
        {2048, 256, 12, UINT64_C(0x521648f0118689d7), UINT64_C(0x68595674064598e4)},
        {2048, 256, 13, UINT64_C(0x4d3d81a059533bea), UINT64_C(0x8c5249b0539d8564)},
        {2048, 256, 14, UINT64_C(0xc2f121f968aeac07), UINT64_C(0x72d2f8794a100c4e)},
        {2048, 256, 15, UINT64_C(0x3906fb6540114df2), UINT64_C(0x406fe7f112ab705b)},
        {2048, 256, 256, UINT64_C(0x365a3e2e51d3fe81), UINT64_C(0x90ed95fd717d72c6)},
        {2048, 256, 512, UINT64_C(0x03ee7eb77caa4cc9), UINT64_C(0xfdd242cc971d6e40)},
        {2048, 256, 1024, UINT64_C(0x694096d44765c8ad), UINT64_C(0x1f7c71ec845db81e)},
    };
    enum { k = 8 };
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        int dim = cases[c].dim, experts = cases[c].experts, tokens = cases[c].tokens;
        size_t count = (size_t)tokens*k;
        float *router = malloc((size_t)experts*dim*sizeof(float));
        float *input = malloc((size_t)tokens*dim*sizeof(float));
        float *weights = malloc(count*sizeof(float));
        int *ids = malloc(count*sizeof(int));
        assert(router && input && weights && ids);
        for (int e = 0; e < experts; e++)
            for (int d = 0; d < dim; d++)
                router[(size_t)e*dim+d] = (float)((e*13+d*17)%257-128)/257.0f;
        for (int t = 0; t < tokens; t++)
            for (int d = 0; d < dim; d++)
                input[(size_t)t*dim+d] = (float)((d*37+t*19)%257-128)/259.0f;
        void *buffer = gpu->buffer_create(gpu->ctx, router,
            (size_t)experts*dim*sizeof(float), BN_GGUF_TENSOR_F32, experts, dim);
        assert(buffer);
        assert(gpu->moe_route_batch(gpu->ctx, ids, weights, buffer,
            input, tokens, dim, experts, k, 1, 1.0f) == 0);
        assert(cuda_test_float_bits_hash(weights,count) == cases[c].weights);
        uint64_t hash = UINT64_C(14695981039346656037);
        for (size_t i = 0; i < count; i++)
            hash = (hash ^ (uint32_t)ids[i])*UINT64_C(1099511628211);
        assert(hash == cases[c].ids);
        gpu->buffer_destroy(gpu->ctx, buffer);
        free(router); free(input); free(weights); free(ids);
    }
    printf("CUDA router matrix reference PASSED\n");
}

static void run_combined_router_reference_case(void) {
    /* Independent llama.cpp CUDA graphs, commit
     * 3d3d7c81813067fc8c185da017e0af03b4269b1e, Blackwell. Compute router
     * softmax first, copy probabilities into a separate graph, then run
     * argsort/normalization and nonzero routed Q8 FFN. The graph boundary
     * guarantees the unfused routing contract without an observer.
     * One output row per expert makes tied expert selection observable. */
    const struct { int experts, tokens, k, mode; uint64_t hash; } cases[] = {
        {32, 9, 4, 0, UINT64_C(0xed3c7a2a0e69d2f0)},
        {32, 9, 4, 1, UINT64_C(0xe66fe58e9ab0ce11)},
        {32, 9, 4, 2, UINT64_C(0xcd06143199068aae)},
        {32, 9, 8, 0, UINT64_C(0xc56d88cd2252123c)},
        {32, 9, 8, 1, UINT64_C(0x0ce47abe22e632c5)},
        {32, 9, 8, 2, UINT64_C(0xba8d5ed66110beed)},
        {32, 17, 4, 0, UINT64_C(0x71b983f48121f02b)},
        {32, 17, 4, 1, UINT64_C(0x033001fe30943681)},
        {32, 17, 4, 2, UINT64_C(0xb68c6d19ebf87fa3)},
        {32, 17, 8, 0, UINT64_C(0xba427fbd42f35a04)},
        {32, 17, 8, 1, UINT64_C(0x18b51db24532d98f)},
        {32, 17, 8, 2, UINT64_C(0x03d397b81fce9f8b)},
        {32, 29, 4, 0, UINT64_C(0x3adaa1fe3bcb1e63)},
        {32, 29, 4, 1, UINT64_C(0x5124f0c09e0475a1)},
        {32, 29, 4, 2, UINT64_C(0x68dbdf4ba54667d1)},
        {32, 29, 8, 0, UINT64_C(0xe13ed5400d024b5d)},
        {32, 29, 8, 1, UINT64_C(0x7113499a4e86f6c1)},
        {32, 29, 8, 2, UINT64_C(0x325ba5988f3516a9)},
        {128, 9, 4, 0, UINT64_C(0xdddf2a58cbb5468e)},
        {128, 9, 4, 1, UINT64_C(0xe66fe58e9ab0ce11)},
        {128, 9, 4, 2, UINT64_C(0xb58c30916f8e6541)},
        {128, 9, 8, 0, UINT64_C(0x379a435ec96e4a1e)},
        {128, 9, 8, 1, UINT64_C(0x0ce47abe22e632c5)},
        {128, 9, 8, 2, UINT64_C(0x48d27ff48232c744)},
        {128, 17, 4, 0, UINT64_C(0xd8ee48b163954184)},
        {128, 17, 4, 1, UINT64_C(0x033001fe30943681)},
        {128, 17, 4, 2, UINT64_C(0x767470ea8cd8b836)},
        {128, 17, 8, 0, UINT64_C(0x5b1abcfac70f07b7)},
        {128, 17, 8, 1, UINT64_C(0x18b51db24532d98f)},
        {128, 17, 8, 2, UINT64_C(0x706a5ef1328e8d01)},
        {128, 29, 4, 0, UINT64_C(0x749b552d098be9f6)},
        {128, 29, 4, 1, UINT64_C(0x5124f0c09e0475a1)},
        {128, 29, 4, 2, UINT64_C(0x15d5f023dcfa7936)},
        {128, 29, 8, 0, UINT64_C(0xb15cd078c8e7c603)},
        {128, 29, 8, 1, UINT64_C(0x7113499a4e86f6c1)},
        {128, 29, 8, 2, UINT64_C(0xcd7959893375437c)},
        {256, 9, 4, 0, UINT64_C(0xc0f1c16ea1ff0ce1)},
        {256, 9, 4, 1, UINT64_C(0xe66fe58e9ab0ce11)},
        {256, 9, 4, 2, UINT64_C(0xc94bcf4524b181bb)},
        {256, 9, 8, 0, UINT64_C(0x5122fd8bfb5c80fb)},
        {256, 9, 8, 1, UINT64_C(0x0ce47abe22e632c5)},
        {256, 9, 8, 2, UINT64_C(0x69acc2da2f26bf33)},
        {256, 17, 4, 0, UINT64_C(0x848105516b353dfc)},
        {256, 17, 4, 1, UINT64_C(0x033001fe30943681)},
        {256, 17, 4, 2, UINT64_C(0xdcc7541d6bea6941)},
        {256, 17, 8, 0, UINT64_C(0x49338f089b3c13c3)},
        {256, 17, 8, 1, UINT64_C(0x18b51db24532d98f)},
        {256, 17, 8, 2, UINT64_C(0xe8bce1ae1aae1d9b)},
        {256, 29, 4, 0, UINT64_C(0xb707b440c912add5)},
        {256, 29, 4, 1, UINT64_C(0x5124f0c09e0475a1)},
        {256, 29, 4, 2, UINT64_C(0xd984473a788efbca)},
        {256, 29, 8, 0, UINT64_C(0x2b706466e132d9ed)},
        {256, 29, 8, 1, UINT64_C(0x7113499a4e86f6c1)},
        {256, 29, 8, 2, UINT64_C(0xa349b42456b519a0)},
    };
    BnBackendRuntimePolicy policy = {0};
    const char *flags[] = {
        "BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT",
        "BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT_SPLIT",
        "BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE",
        "BN_CUDA_DISABLE_MOE_NATIVE_QUANT_GATEUP_2ROW",
        "BN_CUDA_DISABLE_MOE_NATIVE_QUANT_DOWN_2ROW"
    };
    for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); i++)
        assert(bn_backend_runtime_policy_set(&policy, flags[i], "1", 1) == 0);
    BnGPUBackend *gpu = bn_gpu_cuda_create_with_policy(&policy);
    assert(gpu);
    bn_backend_runtime_policy_free(&policy);
    enum { dim = 256, hidden = 32 };
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int experts = cases[c].experts, nt = cases[c].tokens;
        int k = cases[c].k, mode = cases[c].mode;
        size_t count = (size_t)nt * dim;
        size_t blocks = (size_t)experts * dim * hidden / 32;
        float *input = malloc(count * sizeof(float));
        float *output = malloc((count + 2) * sizeof(float));
        float *router = calloc((size_t)experts * dim, sizeof(float));
        BnBlockQ8_0 *gate = calloc(blocks, sizeof(*gate));
        BnBlockQ8_0 *down = calloc(blocks, sizeof(*down));
        assert(input && output && router && gate && down);
        for (int t = 0; t < nt; t++)
            for (int e = 0; e < dim; e++)
                input[(size_t)t * dim + e] = mode == 0
                    ? (float)((e * 37 + t * 19) % 257 - 128) / 64.0f
                    : mode == 1 ? (float)((e + t) % 13) * 1e-8f
                    : (e % 32 == 3 ? (float)(e + t) / 128.0f : -8.0f);
        for (int e = 0; e < experts; e++) {
            router[(size_t)e * dim + e] = 1.0f;
            gate[(size_t)e * dim * hidden / 32].d = bn_fp32_to_fp16(1.0f);
            gate[(size_t)e * dim * hidden / 32].qs[0] = 1;
            down[(size_t)e * dim + e].d = bn_fp32_to_fp16(0.125f);
            down[(size_t)e * dim + e].qs[0] = 1;
        }
        void *buffers[] = {
            gpu->buffer_create(gpu->ctx, gate, blocks * sizeof(*gate),
                BN_GGUF_TENSOR_Q8_0, experts * hidden, dim),
            gpu->buffer_create(gpu->ctx, down, blocks * sizeof(*down),
                BN_GGUF_TENSOR_Q8_0, experts * dim, hidden),
            gpu->buffer_create(gpu->ctx, router,
                (size_t)experts * dim * sizeof(float),
                BN_GGUF_TENSOR_F32, experts, dim),
        };
        for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++)
            assert(buffers[i]);
        output[0] = output[count + 1] = 123.0f;
        for (int alias = 0; alias < 2; alias++) {
            float *out = alias ? input : output + 1;
            assert(gpu->moe_route_routed_ffn_batch(gpu->ctx, out, buffers[2],
                buffers[0], buffers[0], buffers[1], input, nt, dim, hidden,
                experts, k, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0,
                BN_GGUF_TENSOR_Q8_0, 0, 1, 1.0f) == 0);
            uint64_t hash = cuda_test_float_bits_hash(out, count);
            if (hash != cases[c].hash)
                fprintf(stderr, "combined routing case=%zu alias=%d hash=%016llx expected=%016llx\n",
                    c, alias, (unsigned long long)hash,
                    (unsigned long long)cases[c].hash);
            assert(hash == cases[c].hash);
            assert(output[0] == 123.0f && output[count + 1] == 123.0f);
        }
        for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++)
            gpu->buffer_destroy(gpu->ctx, buffers[i]);
        free(down); free(gate); free(router); free(output); free(input);
    }
    bn_gpu_cuda_destroy(gpu);
    printf("CUDA combined routing reference PASSED\n");
}

static void run_q8_mmq_reference_case(BnGPUBackend *gpu) {
    /* Full-output hashes from independent ggml_mul_mat CUDA graphs at
     * llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e, Blackwell.
     * Covers partial row tiles, the >128-token transition, schedule
     * replacement and nonzero output offsets in multi-projection calls. */
    const struct { int rows, cols, tokens; uint64_t hash; } cases[] = {
        {17, 256, 9, UINT64_C(0x31c4e672729eeefd)},
        {17, 256, 129, UINT64_C(0x79aa17219b476e41)},
        {17, 4096, 9, UINT64_C(0x3b817ad13d613e2d)},
        {17, 4096, 129, UINT64_C(0xd5fe6754898b6c32)},
        {129, 4096, 256, UINT64_C(0x14a5993558890409)},
        {512, 8192, 160, UINT64_C(0x184ea84670d6fc6d)},
        {17, 256, 9, UINT64_C(0x31c4e672729eeefd)}
    };
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        int rows = cases[c].rows, cols = cases[c].cols, tokens = cases[c].tokens;
        size_t blocks = (size_t)rows*cols/32, count = (size_t)rows*tokens;
        BnBlockQ8_0 *weights = malloc(blocks*sizeof(*weights));
        float *input = malloc((size_t)cols*tokens*sizeof(*input));
        float *output[2] = {malloc((count+2)*sizeof(float)), malloc((count+2)*sizeof(float))};
        assert(weights && input && output[0] && output[1]);
        for (size_t b = 0; b < blocks; b++) {
            weights[b].d = (uint16_t)(0x2c00 + (b%17)*37);
            for (int i = 0; i < 32; i++)
                weights[b].qs[i] = (int8_t)((int)((b*17+i*13)%255)-127);
        }
        for (size_t i = 0; i < (size_t)cols*tokens; i++)
            input[i] = (float)((int)((i*37)%257)-128)/256.0f;
        void *buffer = gpu->buffer_create(gpu->ctx, weights, blocks*sizeof(*weights),
                                          BN_GGUF_TENSOR_Q8_0, rows, cols);
        assert(buffer);
        for (int call = 0; call < 2; call++) {
            for (int i = 0; i < 2; i++)
                for (size_t j = 0; j < count+2; j++) output[i][j] = -12345.0f;
            if (call == 0) {
                assert(gpu->matmul(gpu->ctx, output[0]+1, buffer, input,
                                   rows, cols, tokens, BN_GGUF_TENSOR_Q8_0) == 0);
            } else {
                BnGPUMatvecOp ops[2] = {
                    {output[0]+1, buffer, rows, cols, BN_GGUF_TENSOR_Q8_0},
                    {output[1]+1, buffer, rows, cols, BN_GGUF_TENSOR_Q8_0}
                };
                assert(gpu->matmul_batch(gpu->ctx, ops, 2, input, tokens, cols) == 0);
            }
            for (int i = 0; i <= call; i++) {
                assert(cuda_test_float_bits_hash(output[i]+1, (int)count) == cases[c].hash);
                assert(output[i][0] == -12345.0f && output[i][count+1] == -12345.0f);
            }
        }
        /* Zero activation blocks must remain finite under reciprocal
         * quantization, including after reuse of nonzero scratch. */
        memset(input, 0, (size_t)cols*tokens*sizeof(*input));
        assert(gpu->matmul(gpu->ctx, output[0]+1, buffer, input,
                           rows, cols, tokens, BN_GGUF_TENSOR_Q8_0) == 0);
        for (size_t i = 0; i < count; i++) assert(output[0][i+1] == 0.0f);
        assert(output[0][0] == -12345.0f && output[0][count+1] == -12345.0f);
        gpu->buffer_destroy(gpu->ctx, buffer);
        free(output[1]); free(output[0]); free(input); free(weights);
    }
    printf("CUDA dense Q8 MMQ reference PASSED\n");
}

static void run_q8_routed_reference_case(int calibrated) {
    /* llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e CUDA:
     * mul_mat_id -> swiglu -> mul_mat_id -> weighted expert reduction.
     * Non-power-of-two route weights distinguish fused FMA from separate
     * multiplication/addition. Hashes cover every output float.
     * Larger batches cover Blackwell MMQ schedules, the 128-token tile
     * boundary, growth/shrinkage and reuse of backend-owned scratch. */
    const int tokens[] = {1, 2, 3, 4, 5, 8, 9, 32, 128, 129, 160, 256, 129, 129, 9};
    const uint64_t expected[] = {
        UINT64_C(0x5ccde7298b908dcb), UINT64_C(0x4c73a402ac2e34f8),
        UINT64_C(0x7b0eb047dac968f5), UINT64_C(0xb4f663bc8c710ac7),
        UINT64_C(0xf04166559e466b88), UINT64_C(0xf88a0ff6f0eb82b7),
        UINT64_C(0x3f9b6a3412d306e1), UINT64_C(0x94979a2030ebce66),
        UINT64_C(0x0fff6802e6af7f22), UINT64_C(0x110eb6d89b3205cf),
        UINT64_C(0xd3c1f31289c9a78d), UINT64_C(0xac15629a7e0f004f),
        UINT64_C(0x110eb6d89b3205cf), UINT64_C(0x110eb6d89b3205cf),
        UINT64_C(0x3f9b6a3412d306e1)
    };
    enum { dim = 2048, hidden = 512, experts = 8, k = 4, max_tokens = 256 };
    BnBackendRuntimePolicy policy = {0};
    const char *flags[] = {
        "BN_CUDA_ENABLE_NATIVE_QUANT_PREPARED_INPUT",
        "BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE",
        "BN_CUDA_DISABLE_MOE_NATIVE_QUANT_GATEUP_2ROW",
        "BN_CUDA_DISABLE_MOE_NATIVE_QUANT_DOWN_2ROW"
    };
    /* Default combined routing and arithmetic must match the same
     * independent goldens without calibration overrides. */
    if (calibrated)
        for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); i++)
            assert(bn_backend_runtime_policy_set(&policy, flags[i], "1", 1) == 0);
    BnGPUBackend *gpu = bn_gpu_cuda_create_with_policy(&policy);
    assert(gpu);
    bn_backend_runtime_policy_free(&policy);
    float rope = 1.0f;
    BnGPUActivationPlan plan = {0};
    plan.dim = plan.vocab_size = plan.xb2_elements = dim;
    plan.hb_elements = k * hidden;
    plan.n_layers = plan.n_heads = plan.attention_layer_count = 1;
    plan.seq_len = plan.kv_dim = plan.head_size = 2;
    plan.rope_frequencies = &rope; plan.rope_frequency_count = 1;
    plan.moe_total_experts = experts; plan.moe_active_experts = k;
    plan.moe_expert_hidden_dim = hidden;
    assert(gpu->init_activations(gpu->ctx, &plan) == 0);
    size_t blocks = (size_t)experts * hidden * dim / 32;
    BnBlockQ8_0 *weights = malloc(blocks * sizeof(*weights));
    float *input = malloc(max_tokens * dim * sizeof(*input));
    float *output = malloc((max_tokens * dim + 2) * sizeof(*output));
    float *mid = malloc(k * hidden * sizeof(*mid));
    assert(weights && input && output && mid);
    void *buffers[3];
    for (int matrix = 0; matrix < 3; matrix++) {
        int seed = matrix * 19;
        for (size_t b = 0; b < blocks; b++) {
            weights[b].d = (uint16_t)(0x1800 + ((b + seed) % 17) * 37);
            for (int i = 0; i < 32; i++)
                weights[b].qs[i] = (int8_t)((int)((b * 17 + i * 13 + seed) % 255) - 127);
        }
        buffers[matrix] = gpu->buffer_create(gpu->ctx, weights,
            blocks * sizeof(*weights), BN_GGUF_TENSOR_Q8_0,
            experts * (matrix == 2 ? dim : hidden), matrix == 2 ? hidden : dim);
        assert(buffers[matrix]);
    }
    for (int i = 0; i < max_tokens * dim; i++)
        input[i] = (float)((i * 37) % 257 - 128) / 256.0f;
    int indices[max_tokens * k];
    float route[max_tokens * k];
    for (int i = 0; i < max_tokens * k; i++) {
        indices[i] = (i * 2 + 1) % experts;
        route[i] = (float)(i % k + 1) / 13.0f;
    }
    for (size_t c = 0; c < sizeof(tokens) / sizeof(tokens[0]); c++) {
        int count = tokens[c] * dim;
        for (int i = 0; i < max_tokens * dim + 2; i++) output[i] = -12345.0f;
        assert(gpu->moe_routed_ffn_batch(gpu->ctx, output + 1,
            buffers[0], buffers[1], buffers[2], indices, route, NULL, input,
            tokens[c], dim, hidden, experts, k, BN_GGUF_TENSOR_Q8_0,
            BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0, 0) == 0);
        assert(cuda_test_float_bits_hash(output + 1, count) == expected[c]);
        assert(output[0] == -12345.0f);
        for (int i = count + 1; i < max_tokens * dim + 2; i++)
            assert(output[i] == -12345.0f);
    }
    float packed_route[2 * k];
    for (int i = 0; i < k; i++) {
        packed_route[i] = route[i]; packed_route[k + i] = (float)indices[i];
    }
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB, input, dim * sizeof(float), 0) == 0);
    assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_XB2, packed_route, sizeof(packed_route), 0) == 0);
    BnGPUOp op = {0};
    op.op_code = BN_GPU_CODE_MOE_ROUTED_FFN;
    op.W_buf = buffers[0]; op.W_buf2 = buffers[1]; op.W_buf3 = buffers[2];
    op.type = BN_GGUF_TENSOR_Q8_0; op.cols = dim;
    op.buf_in = BN_GPU_VALUE_XB; op.buf_out = BN_GPU_VALUE_MOE_OUT;
    op.buf_aux = BN_GPU_VALUE_XB2;
    op.p[0] = hidden; op.p[1] = experts; op.p[2] = k;
    op.p[3] = BN_GGUF_TENSOR_Q8_0; op.p[4] = BN_GPU_VALUE_MOE_HB;
    assert(gpu->execute(gpu->ctx, &op, 1, BN_GPU_VALUE_MOE_OUT, output, dim) == 0);
    assert(cuda_test_float_bits_hash(output, dim) == expected[0]);
    assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_MOE_HB, mid,
                               k * hidden * sizeof(float), 0) == 0);
    assert(cuda_test_float_bits_hash(mid, k * hidden) == UINT64_C(0xe886144622a85ac4));
    /* Compare combined device routing/FFN with explicit route results,
     * including nonzero logits and both router dispatch boundaries. */
    float *router_data = calloc((size_t)experts * dim, sizeof(float));
    float *combined = malloc((size_t)max_tokens * dim * sizeof(float));
    assert(router_data && combined);
    for (int e = 0; e < experts; e++)
        for (int d = 0; d < dim; d++)
            router_data[(size_t)e*dim+d] = (float)((e*13+d*17)%257-128)/257.0f;
    void *router = gpu->buffer_create(gpu->ctx, router_data,
        (size_t)experts * dim * sizeof(float), BN_GGUF_TENSOR_F32, experts, dim);
    assert(router);
    const int route_batches[] = {3, 4, 9, 16, 17, 129};
    for (size_t pass = 0; pass < sizeof(route_batches)/sizeof(route_batches[0]); pass++) {
        int nt = route_batches[pass];
        assert(gpu->moe_route_batch(gpu->ctx, indices, route, router, input,
                                   nt, dim, experts, k, 1, 1.0f) == 0);
        assert(gpu->moe_routed_ffn_batch(gpu->ctx, output,
            buffers[0], buffers[1], buffers[2], indices, route, NULL, input,
            nt, dim, hidden, experts, k, BN_GGUF_TENSOR_Q8_0,
            BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0, 0) == 0);
        assert(gpu->moe_route_routed_ffn_batch(gpu->ctx, combined, router,
            buffers[0], buffers[1], buffers[2], input, nt, dim, hidden,
            experts, k, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0,
            BN_GGUF_TENSOR_Q8_0, 0, 1, 1.0f) == 0);
        assert(memcmp(output, combined, (size_t)nt * dim * sizeof(float)) == 0);
    }
    gpu->buffer_destroy(gpu->ctx, router);
    free(combined); free(router_data);
    for (int i = 0; i < 3; i++) gpu->buffer_destroy(gpu->ctx, buffers[i]);
    free(mid); free(output); free(input); free(weights);
    bn_gpu_cuda_destroy(gpu);
    printf("CUDA routed Q8 reference (%s) PASSED\n",
           calibrated ? "calibrated" : "default");
}

static void run_moe_dense_residual_composition_case(BnGPUBackend *gpu) {
    enum { nt = 3, dim = 32, hidden = 32, count = nt * dim };
    float gate[dim * hidden], up[dim * hidden], down[dim * hidden];
    float norm[dim], dense_norm[dim], routed_norm[dim], output_norm[dim];
    float act[count], routed[count], input[count], dense[count];
    float combined[count], expected[count], actual[count];
    for (int i = 0; i < dim * hidden; i++) {
        gate[i] = (float)((i * 13) % 37 - 18) / 256.0f;
        up[i] = (float)((i * 17) % 41 - 20) / 256.0f;
        down[i] = (float)((i * 19) % 43 - 21) / 256.0f;
    }
    for (int i = 0; i < dim; i++) {
        norm[i] = 0.6f + 0.01f * (float)(i % 11);
        dense_norm[i] = 0.7f + 0.01f * (float)(i % 7);
        routed_norm[i] = 0.8f + 0.01f * (float)(i % 5);
        output_norm[i] = 0.9f + 0.01f * (float)(i % 3);
    }
    for (int i = 0; i < count; i++) {
        act[i] = (float)((i * 23) % 59 - 29) / 32.0f;
        routed[i] = (float)((i * 29) % 61 - 30) / 64.0f;
    }
    void *bg = gpu->buffer_create(gpu->ctx, gate, sizeof(gate),
                                   BN_GGUF_TENSOR_F32, hidden, dim);
    void *bu = gpu->buffer_create(gpu->ctx, up, sizeof(up),
                                   BN_GGUF_TENSOR_F32, hidden, dim);
    void *bd = gpu->buffer_create(gpu->ctx, down, sizeof(down),
                                   BN_GGUF_TENSOR_F32, dim, hidden);
    void *bn = gpu->buffer_create(gpu->ctx, norm, sizeof(norm),
                                   BN_GGUF_TENSOR_F32, 1, dim);
    void *bdn = gpu->buffer_create(gpu->ctx, dense_norm,
                                    sizeof(dense_norm), BN_GGUF_TENSOR_F32,
                                    1, dim);
    void *brn = gpu->buffer_create(gpu->ctx, routed_norm,
                                    sizeof(routed_norm), BN_GGUF_TENSOR_F32,
                                    1, dim);
    void *bon = gpu->buffer_create(gpu->ctx, output_norm,
                                    sizeof(output_norm), BN_GGUF_TENSOR_F32,
                                    1, dim);
    assert(bg && bu && bd && bn && bdn && brn && bon);
    assert(gpu->moe_dense_residual_batch);
    for (int raw = 0; raw < 2; raw++) {
        assert(gpu->rmsnorm_batch(gpu->ctx, input, bn, act,
                                  nt, dim, 1e-6f) == 0);
        assert(gpu->dense_ffn_batch(gpu->ctx, dense, bg, bu, bd, input,
                                     nt, dim, hidden, BN_GGUF_TENSOR_F32,
                                     BN_GGUF_TENSOR_F32, BN_GGUF_TENSOR_F32,
                                     BN_MODEL_ACTIVATION_SILU) == 0);
        assert(gpu->rmsnorm_batch(gpu->ctx, dense, bdn, dense,
                                  nt, dim, 1e-6f) == 0);
        assert(gpu->rmsnorm_residual_batch(gpu->ctx, combined, brn,
                                            routed, dense, nt, dim,
                                            1e-6f) == 0);
        if (raw)
            assert(gpu->rmsnorm_batch(gpu->ctx, expected, bon, combined,
                                      nt, dim, 1e-6f) == 0);
        else
            assert(gpu->rmsnorm_residual_batch(gpu->ctx, expected, bon,
                                                combined, act, nt, dim,
                                                1e-6f) == 0);
        memcpy(actual, routed, sizeof(actual));
        BnGPUMoEDenseResidualBatch batch = {
            .out = actual, .act = act, .routed = actual,
            .gate_buf = bg, .up_buf = bu, .down_buf = bd,
            .input_norm_buf = bn, .dense_post_norm_buf = bdn,
            .routed_post_norm_buf = brn, .output_norm_buf = bon,
            .n_tokens = nt, .dim = dim, .hidden_dim = hidden,
            .gate_type = BN_GGUF_TENSOR_F32,
            .up_type = BN_GGUF_TENSOR_F32,
            .down_type = BN_GGUF_TENSOR_F32,
            .act_type = BN_MODEL_ACTIVATION_SILU,
            .norm_eps = 1e-6f, .raw_output = raw,
        };
        assert(bn_gpu_backend_moe_dense_residual_batch(gpu, &batch) == 0);
        expect_close(actual, expected, count);
        batch.hidden_dim = -1;
        assert(bn_gpu_backend_moe_dense_residual_batch(gpu, &batch) != 0);
        expect_close(actual, expected, count);
    }
    gpu->buffer_destroy(gpu->ctx, bg);
    gpu->buffer_destroy(gpu->ctx, bu);
    gpu->buffer_destroy(gpu->ctx, bd);
    gpu->buffer_destroy(gpu->ctx, bn);
    gpu->buffer_destroy(gpu->ctx, bdn);
    gpu->buffer_destroy(gpu->ctx, brn);
    gpu->buffer_destroy(gpu->ctx, bon);
    printf("CUDA MoE dense-residual composition PASSED\n");
}

static void run_moe_routed_dense_residual_case(BnGPUBackend *gpu) {
    enum { nt = 3, dim = 32, hidden = 32, experts = 2, k = 2,
           count = nt * dim, blocks = experts * dim * hidden / 32 };
    BnBlockQ4_0 weights[3][blocks];
    float act[count], input[count], routed[count], expected[count], actual[count];
    float normalized[count], dense[count], combined[count];
    float norm[dim], dense_norm[dim], routed_norm[dim], output_norm[dim];
    int indices[nt * k] = {0, 1, 1, 0, 0, 1};
    float route[nt * k] = {0.7f, 0.3f, 0.6f, 0.4f, 0.8f, 0.2f};
    float scales[nt * k] = {1.1f, 0.9f, 0.9f, 1.1f, 1.1f, 0.9f};
    for (int m = 0; m < 3; m++)
        for (int b = 0; b < blocks; b++) {
            weights[m][b].d = bn_fp32_to_fp16(0.03f + 0.001f * m);
            for (int j = 0; j < 16; j++) {
                int lo = (b * 7 + j * 3 + m) & 15;
                int hi = (b * 11 + j * 5 + m) & 15;
                weights[m][b].qs[j] = (uint8_t)(lo | (hi << 4));
            }
        }
    for (int i = 0; i < count; i++) {
        act[i] = (float)((i * 13) % 41 - 20) / 32.0f;
        input[i] = (float)((i * 17) % 47 - 23) / 32.0f;
    }
    for (int i = 0; i < dim; i++) {
        norm[i] = 0.7f + 0.01f * (i % 5);
        dense_norm[i] = 0.8f + 0.01f * (i % 7);
        routed_norm[i] = 0.9f + 0.01f * (i % 3);
        output_norm[i] = 1.0f + 0.01f * (i % 11);
    }
    void *rg = gpu->buffer_create(gpu->ctx, weights[0], sizeof(weights[0]),
                                    BN_GGUF_TENSOR_Q4_0, experts * hidden, dim);
    void *ru = gpu->buffer_create(gpu->ctx, weights[1], sizeof(weights[1]),
                                    BN_GGUF_TENSOR_Q4_0, experts * hidden, dim);
    void *rd = gpu->buffer_create(gpu->ctx, weights[2], sizeof(weights[2]),
                                    BN_GGUF_TENSOR_Q4_0, experts * dim, hidden);
    void *dg = gpu->buffer_create(gpu->ctx, weights[0],
                                    sizeof(weights[0]) / experts,
                                    BN_GGUF_TENSOR_Q4_0, hidden, dim);
    void *du = gpu->buffer_create(gpu->ctx, weights[1],
                                    sizeof(weights[1]) / experts,
                                    BN_GGUF_TENSOR_Q4_0, hidden, dim);
    void *dd = gpu->buffer_create(gpu->ctx, weights[2],
                                    sizeof(weights[2]) / experts,
                                    BN_GGUF_TENSOR_Q4_0, dim, hidden);
    void *n = gpu->buffer_create(gpu->ctx, norm, sizeof(norm),
                                  BN_GGUF_TENSOR_F32, 1, dim);
    void *dn = gpu->buffer_create(gpu->ctx, dense_norm, sizeof(dense_norm),
                                   BN_GGUF_TENSOR_F32, 1, dim);
    void *rn = gpu->buffer_create(gpu->ctx, routed_norm, sizeof(routed_norm),
                                   BN_GGUF_TENSOR_F32, 1, dim);
    void *on = gpu->buffer_create(gpu->ctx, output_norm, sizeof(output_norm),
                                   BN_GGUF_TENSOR_F32, 1, dim);
    assert(rg && ru && rd && dg && du && dd && n && dn && rn && on);
    assert(gpu->moe_routed_dense_residual_batch);
    for (int raw = 0; raw < 2; raw++) {
        assert(gpu->moe_routed_ffn_batch(gpu->ctx, routed, rg, ru, rd,
            indices, route, scales, input, nt, dim, hidden, experts, k,
            BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0,
            BN_GGUF_TENSOR_Q4_0, BN_MODEL_ACTIVATION_GELU) == 0);
        assert(gpu->rmsnorm_batch(gpu->ctx, normalized, n, act,
                                  nt, dim, 1e-6f) == 0);
        assert(gpu->dense_ffn_batch(gpu->ctx, dense, dg, du, dd, normalized,
            nt, dim, hidden, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0,
            BN_GGUF_TENSOR_Q4_0, BN_MODEL_ACTIVATION_GELU) == 0);
        assert(gpu->rmsnorm_batch(gpu->ctx, dense, dn, dense,
                                  nt, dim, 1e-6f) == 0);
        assert(gpu->rmsnorm_residual_batch(gpu->ctx, combined, rn, routed,
                                            dense, nt, dim, 1e-6f) == 0);
        if (raw)
            assert(gpu->rmsnorm_batch(gpu->ctx, expected, on, combined,
                                      nt, dim, 1e-6f) == 0);
        else
            assert(gpu->rmsnorm_residual_batch(gpu->ctx, expected, on,
                                                combined, act, nt, dim,
                                                1e-6f) == 0);
        BnGPUMoERoutedDenseResidualBatch batch = {0};
        batch.dense = (BnGPUMoEDenseResidualBatch) {
            .out = actual, .act = act,
            .gate_buf = dg, .up_buf = du, .down_buf = dd,
            .input_norm_buf = n, .dense_post_norm_buf = dn,
            .routed_post_norm_buf = rn, .output_norm_buf = on,
            .n_tokens = nt, .dim = dim, .hidden_dim = hidden,
            .gate_type = BN_GGUF_TENSOR_Q4_0,
            .up_type = BN_GGUF_TENSOR_Q4_0,
            .down_type = BN_GGUF_TENSOR_Q4_0,
            .act_type = BN_MODEL_ACTIVATION_GELU,
            .norm_eps = 1e-6f, .raw_output = raw,
        };
        batch.input = input;
        batch.indices = indices;
        batch.weights = route;
        batch.output_scales = scales;
        batch.routed_gate_buf = rg;
        batch.routed_up_buf = ru;
        batch.routed_down_buf = rd;
        batch.moe_hidden_dim = hidden;
        batch.n_experts = experts;
        batch.k = k;
        batch.routed_gate_type = BN_GGUF_TENSOR_Q4_0;
        batch.routed_up_type = BN_GGUF_TENSOR_Q4_0;
        batch.routed_down_type = BN_GGUF_TENSOR_Q4_0;
        assert(bn_gpu_backend_moe_routed_dense_residual_batch(gpu, &batch) == 0);
        expect_close(actual, expected, count);
        batch.n_experts = 0;
        assert(bn_gpu_backend_moe_routed_dense_residual_batch(gpu, &batch) != 0);
        expect_close(actual, expected, count);
    }
    void *buffers[] = {rg, ru, rd, dg, du, dd, n, dn, rn, on};
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++)
        gpu->buffer_destroy(gpu->ctx, buffers[i]);
    printf("CUDA MoE routed dense-residual composition PASSED\n");
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
#ifndef BN_ENABLE_CUDA
    printf("CUDA backend test skipped: BN_ENABLE_CUDA not set\n");
    return 0;
#else
    BnGPUBackend *gpu = bn_gpu_cuda_create();
    if (!gpu) {
        printf("CUDA backend test skipped: no CUDA device\n");
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--moe-dense-residual-composition") == 0) {
        run_moe_dense_residual_composition_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--moe-routed-dense-residual") == 0) {
        run_moe_routed_dense_residual_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--iq4nl-reference") == 0) {
        run_iq4nl_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--iq4xs-reference") == 0) {
        run_iq4xs_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--q5-decode-reference") == 0) {
        run_q5_decode_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--iq3s-reference") == 0) {
        run_iq3s_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--gemma4-decode-attention-reference") == 0) {
        run_decode_attention_partition_replay_case(gpu, 256, 2);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--q4k-short-mmq-reference") == 0) {
        run_q4k_mmq_original_sum_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--q5-wide-prefill") == 0) {
        run_wide_q5k_matmul_case(gpu, 127);
        run_wide_q5k_matmul_case(gpu, 128);
        run_wide_q5k_matmul_case(gpu, 129);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--q6-wide-prefill") == 0) {
        run_wide_q6k_prefill_case(gpu, 129);
        run_wide_q6k_prefill_case(gpu, 257);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    assert(gpu->caps & BN_GPU_CAP_REFERENCE_ATTENTION);
    assert(!(gpu->caps & BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH));
    assert(gpu->caps & BN_GPU_CAP_REFERENCE_ATTENTION_FALLBACK);
    assert(!(gpu->caps & BN_GPU_CAP_DECODE_GRAPH_CACHE));

    if (argc == 2 && strcmp(argv[1], "--q4-original-sum-reference") == 0) {
        run_q4_original_sum_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "--routed-mxfp4-combined-reference") == 0) {
        run_mxfp4_routed_reference_cases(gpu,2);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--routed-mxfp4-graph-reference") == 0) {
        run_mxfp4_routed_reference_cases(gpu,1);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--router512-reference") == 0) {
        run_router512_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--routed-q451-reference") == 0) {
        run_q451_routed_reference_cases(gpu,0);
        run_q451_routed_reference_cases(gpu,1);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--routed-q5q6-reference") == 0) {
        run_q5q6_routed_reference_cases(gpu,0);
        run_q5q6_routed_reference_cases(gpu,1);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--routed-mxfp4-reference") == 0) {
        run_mxfp4_routed_reference_cases(gpu,0);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--norm-residual-reference") == 0) {
        run_norm_residual_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--prepared-v-reference") == 0) {
        run_prepared_v_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--merged-routed-q4-reference") == 0) {
        run_merged_routed_q4_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--routed-q4-reference") == 0) {
        run_routed_q4_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--gelu-prefill-reference") == 0) {
        run_gelu_prefill_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--gelu-gate") == 0) {
        run_gelu_gate_accuracy_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--expert-reduction-reference") == 0) {
        run_expert_reduction_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--hc-combine-reference") == 0) {
        run_hc_combine_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--hc-mixer-reference") == 0) {
        run_hc_mixer_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--hc-rmsnorm-reference") == 0) {
        run_hc_rmsnorm_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--signed-sqrt-gate-reference") == 0) {
        run_signed_sqrt_gate_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--expert-batch-reduction-reference") == 0) {
        run_expert_batch_reduction_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--scaled-rmsnorm-batch-reference") == 0) {
        run_scaled_rmsnorm_batch_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--scaled-rmsnorm-reference") == 0) {
        run_scaled_rmsnorm_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--rmsnorm-widths") == 0) {
        run_rmsnorm_width_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--mixed-attention-reference") == 0) {
        run_rmsnorm_batch_reference_case(gpu);
        run_prepared_rope_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--standalone-ffn-reference") == 0) {
        run_standalone_ffn_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--q4-reference") == 0) {
        run_q4_original_sum_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--mxfp4-reference") == 0) {
        run_mxfp4_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--q3k-reference") == 0) {
        run_q3k_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--q8-split-reference") == 0) {
        run_q8_split_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--q8-reference") == 0) {
        run_q8_reference_case(gpu);
        run_q8_split_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--signed-mmq-reference") == 0) {
        run_signed_mmq_reference_case(gpu);
        run_ssm_gateup_layout_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--gqa6-prefill-reference") == 0) {
        run_prefill_window_reference_cases(gpu, 3);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--ssm-gateup-layout") == 0) {
        run_ssm_gateup_layout_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && (strcmp(argv[1], "--dense-projections-reference") == 0 ||
                      strcmp(argv[1], "--dense-gateup-reference") == 0 ||
                      strcmp(argv[1], "--dense-qk-reference") == 0)) {
        if (strcmp(argv[1], "--dense-qk-reference") != 0) run_dense_gateup_reference_case(gpu);
        if (strcmp(argv[1], "--dense-gateup-reference") != 0) run_dense_qk_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--q6-small-reference") == 0) {
        run_q6_small_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--kquant-mmq-reference") == 0) {
        run_kquant_mmq_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--dense-prefill-reference") == 0) {
        run_dense_prefill_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--prefill-window-paths") == 0) {
        run_prefill_window_paths_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--prefill512-long-reference") == 0) {
        run_prefill_window_reference_cases(gpu, 2);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--prefill-long-reference") == 0) {
        run_prefill_window_reference_cases(gpu, 1);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--prefill-window-reference") == 0) {
        run_prefill_window_reference_cases(gpu, 0);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--attention-window-replay") == 0) {
        run_attention_window_replay_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--graph-teardown") == 0) {
        run_graph_teardown_case();
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--decode-f32-attention-reference") == 0) {
        run_decode_attention_f32_reference_cases(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--decode16-attention-reference") == 0) {
        run_decode_attention_reference_cases(gpu, 256, 16);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--decode256-attention-reference") == 0) {
        run_decode_attention_reference_cases(gpu, 256, 0);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--decode-attention-reference") == 0) {
        run_decode_attention_reference_cases(gpu, 0, 0);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--norm-residual-reference") == 0) {
        run_norm_residual_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--qk-rope-reference") == 0) {
        run_qk_rope_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--ssm-projection-state-reference") == 0) {
        run_ssm_projection_state_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--f32-matrix-reference") == 0) {
        run_f32_matrix_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--q4-decode-reference") == 0) {
        run_q4_decode_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--gqa2-attention-reference") == 0) {
        run_gqa2_attention_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--head512-attention-reference") == 0) {
        run_head512_attention_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--combined-router-reference") == 0) {
        bn_gpu_cuda_destroy(gpu);
        run_combined_router_reference_case();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--standalone-rope-reference") == 0) {
        run_standalone_rope_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--model-attention-reference") == 0) {
        run_model_attention_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--raw-attention-reference") == 0) {
        run_raw_attention_mma_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "--prefix-attention-smoke") == 0) {
        run_prefix_attention_reference_cases(gpu, 1);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--prefix-attention-reference") == 0) {
        run_prefix_attention_reference_cases(gpu, 0);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--prepared-attention-reference") == 0) {
        run_prepared_attention_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "--qk-rope-reference") == 0) {
        run_qk_rope_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "--split-attention-reference") == 0) {
        run_split_attention_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "--small-attention-reference") == 0) {
        run_small_attention_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "--activation-upload-readiness") == 0) {
        run_rope_preparation_case(gpu);
        run_activation_upload_readiness_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "--q8-routed-reference") == 0) {
        run_q8_routed_reference_case(0);
        run_q8_routed_reference_case(1);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--moe-residual-order-reference") == 0) {
        run_rmsnorm_batch_reference_case(gpu);
    run_prepared_rope_reference_case(gpu);
    run_standalone_ffn_reference_case(gpu);
    run_moe_residual_order_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "--shared-batch-reference") == 0) {
        run_shared_batch_reference_case(gpu);
        bn_gpu_cuda_destroy(gpu);
        return 0;
    }

    if (argc == 2 && strcmp(argv[1], "--diagnostic-isolation") == 0) {
        bn_gpu_cuda_destroy(gpu);
        run_cuda_diagnostics_isolation();
        return 0;
    }

    assert(bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_MOE_EXPERT_GRAPH));
    assert(bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_REFERENCE_RECURRENT));
    assert(bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_KQUANT_BLOCK32_LOGITS));
    if (bn_gpu_backend_has_cap(gpu, BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL))
        assert(gpu->prefill_ssm_layer != NULL);
    assert(gpu->init_activations != NULL);
    assert(gpu->write_activation != NULL);
    assert(gpu->read_activation != NULL);
    assert(gpu->free_activations != NULL);
    assert(gpu->buffer_f16_cache_extra_bytes != NULL);
    size_t cache_elements = (size_t)1024 * 2560;
    assert(bn_gpu_backend_f16_cache_extra_bytes(
               gpu, BN_GGUF_TENSOR_Q6_K, 1024, 2560) ==
           cache_elements * (sizeof(uint16_t) + sizeof(float)));
    assert(bn_gpu_backend_f16_cache_extra_bytes(
               gpu, BN_GGUF_TENSOR_Q4_K, 1024, 2560) >
           cache_elements * sizeof(uint16_t));
    assert(bn_gpu_backend_f16_cache_extra_bytes(
               gpu, BN_GGUF_TENSOR_Q6_K, INT_MAX, INT_MAX / 256 * 256) ==
           SIZE_MAX);
    assert(bn_gpu_backend_f16_cache_extra_bytes(
               gpu, BN_GGUF_TENSOR_Q6_K, 0, 2560) == 0);
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
    cfg.hyper_connection_count = 2;
    cfg.hyper_connection_rank = 3;
    assert(init_test_activations(gpu, &cfg) == 0);
    {
        float in[4] = { 1.0f, -2.0f, 3.0f, -4.0f };
        float out[4] = { 0 };
        assert(gpu->write_activation(gpu->ctx, BN_GPU_VALUE_X, in,
                                     sizeof(in), 0) == 0);
        assert(gpu->read_activation(gpu->ctx, BN_GPU_VALUE_X, out,
                                    sizeof(out), 0) == 0);
        expect_close(out, in, 4);
        assert(gpu->write_activation(gpu->ctx,
                                     BN_GPU_VALUE_HC_RESIDUAL, in,
                                     sizeof(in), sizeof(in)) == 0);
        memset(out, 0, sizeof(out));
        assert(gpu->read_activation(gpu->ctx,
                                    BN_GPU_VALUE_HC_RESIDUAL, out,
                                    sizeof(out), sizeof(out)) == 0);
        expect_close(out, in, 4);
    }
    gpu->free_activations(gpu->ctx);

    run_hyper_connection_ops_case(gpu);
    run_hc_combine_reference_cases(gpu);
    run_hc_mixer_reference_cases(gpu);
    run_hc_rmsnorm_reference_cases(gpu);
    run_signed_sqrt_gate_reference_cases(gpu);

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
    float xk_block32[BN_QK_K];
    for (int b = 0; b < BN_QK_K; b += 32) {
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(xk[b + j]));
        float d = amax / 127.0f;
        float stored = bn_fp16_to_fp32(bn_fp32_to_fp16(d));
        for (int j = 0; j < 32; j++)
            xk_block32[b + j] = d == 0 ? 0 : stored * roundf(xk[b + j] / d);
    }

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

    BnBlockQ5_1 q5_1[8];
    memset(q5_1, 0, sizeof(q5_1));
    float ref_q5_1 = 0.0f;
    for (int b = 0; b < 8; b++) {
        q5_1[b].d = bn_fp32_to_fp16(0.125f);
        q5_1[b].m = bn_fp32_to_fp16(-0.75f);
        for (int i = 0; i < 16; i++) {
            int q0 = (i + b) & 31;
            int q1 = (31 - i + b) & 31;
            q5_1[b].qs[i] = (uint8_t)((q0 & 15) | ((q1 & 15) << 4));
            q5_1[b].qh[i / 8] |= (uint8_t)(((q0 >> 4) & 1) << (i & 7));
            q5_1[b].qh[(i + 16) / 8] |=
                (uint8_t)(((q1 >> 4) & 1) << ((i + 16) & 7));
            ref_q5_1 += (0.125f * (float)q0 - 0.75f) *
                        xk[b * 32 + i];
            ref_q5_1 += (0.125f * (float)q1 - 0.75f) *
                        xk[b * 32 + i + 16];
        }
    }
    run_matvec_case(gpu, q5_1, sizeof(q5_1), BN_GGUF_TENSOR_Q5_1,
                    1, BN_QK_K, xk, &ref_q5_1);

    BnBlockQ4K q4k;
    memset(&q4k, 0, sizeof(q4k));
    q4k.d = bn_fp32_to_fp16(0.25f);
    q4k.dmin = bn_fp32_to_fp16(0.0f);
    q4k.scales[0] = 3;
    q4k.scales[1] = 5;
    q4k.scales[2] = 7;
    q4k.scales[3] = 11;
    for (int pair = 0; pair < 2; pair++) {
        for (int i = 0; i < 32; i++) {
            uint8_t lo = (uint8_t)((i + pair) & 15);
            uint8_t hi = (uint8_t)((15 - i + pair) & 15);
            q4k.qs[pair * 32 + i] = (uint8_t)(lo | (hi << 4));
        }
    }
    float q4k_values[BN_QK_K];
    float ref_q4k = 0.0f;
    bn_quant_dequant_q4k(&q4k, q4k_values);
    for (int i = 0; i < BN_QK_K; i++)
        ref_q4k += q4k_values[i] * xk_block32[i];
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
        ref_q5k += 0.125f * 2.0f * (float)q * xk_block32[i];
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
        ref_q6k += 0.5f * 3.0f * (float)((int)q - 32) * xk_block32[i];
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

    BnBlockQ3K q3k = {0};
    q3k.d = bn_fp32_to_fp16(0.125f);
    for (int i = 0; i < 32; i++) q3k.hmask[i] = (uint8_t)(0xa5u ^ i);
    for (int i = 0; i < 64; i++) q3k.qs[i] = (uint8_t)(i * 29u + 7u);
    for (int i = 0; i < 12; i++) q3k.scales[i] = (uint8_t)(i * 17u + 3u);
    float q3k_values[BN_QK_K];
    bn_quant_dequant_q3k(&q3k, q3k_values);
    float ref_q3k = 0.0f;
    for (int i = 0; i < BN_QK_K; i++) ref_q3k += q3k_values[i] * xk[i];
    run_matvec_case(gpu, &q3k, sizeof(q3k), BN_GGUF_TENSOR_Q3_K,
                    1, BN_QK_K, xk, &ref_q3k);

    float x32[32];
    BnBlockIQ4NL iq4nl;
    memset(&iq4nl, 0, sizeof(iq4nl));
    iq4nl.d = bn_fp32_to_fp16(0.25f);
    static const int8_t iq4nl_values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113,
    };
    float ref_iq4nl = 0.0f;
    for (int i = 0; i < 32; i++) {
        x32[i] = (float)(i - 15) * 0.125f;
        int q = i & 15;
        if (i < 16)
            iq4nl.qs[i] = (uint8_t)q;
        else
            iq4nl.qs[i - 16] |= (uint8_t)(q << 4);
        ref_iq4nl += 0.25f * (float)iq4nl_values[q] * x32[i];
    }
    run_matvec_case(gpu, &iq4nl, sizeof(iq4nl),
                    BN_GGUF_TENSOR_IQ4_NL, 1, 32, x32, &ref_iq4nl);

    BnBlockIQ4XS iq4xs[5];
    float ref_iq4xs[5] = { 0 };
    memset(iq4xs, 0, sizeof(iq4xs));
    for (int r = 0; r < 5; r++) {
        float d = (float)(r + 1) / 262144.0f;
        iq4xs[r].d = bn_fp32_to_fp16(d);
        for (int group = 0; group < 8; group++) {
            int scale = (r * 11 + group * 7) & 63;
            iq4xs[r].scales_l[group / 2] |=
                (uint8_t)((scale & 15) << ((group & 1) * 4));
            iq4xs[r].scales_h |=
                (uint16_t)((scale >> 4) << (group * 2));
            float dl = d * (float)(scale - 32);
            for (int i = 0; i < 16; i++) {
                int q0 = (r + group + i) & 15;
                int q1 = (r + group * 3 + 15 - i) & 15;
                iq4xs[r].qs[group * 16 + i] =
                    (uint8_t)(q0 | (q1 << 4));
                ref_iq4xs[r] += dl *
                    ((float)iq4nl_values[q0] * xk[group * 32 + i] +
                     (float)iq4nl_values[q1] * xk[group * 32 + i + 16]);
            }
        }
    }
    run_matvec_case(gpu, iq4xs, sizeof(iq4xs),
                    BN_GGUF_TENSOR_IQ4_XS, 5, BN_QK_K, xk, ref_iq4xs);

    BnBlockIQ3S iq3s;
    memset(&iq3s, 0, sizeof(iq3s));
    iq3s.d = bn_fp32_to_fp16(0.5f);
    float ref_iq3s = 0.0f;
    for (int i = 0; i < BN_QK_K; i++) {
        int ib32 = i / 32;
        int in32 = i % 32;
        int l = in32 / 4;
        int k = in32 % 4;
        if (i & 1)
            iq3s.signs[ib32 * 4 + l / 2] |=
                (uint8_t)(1u << ((l & 1) * 4 + k));
        ref_iq3s += (i & 1 ? -0.5f : 0.5f) * xk[i];
    }
    run_matvec_case(gpu, &iq3s, sizeof(iq3s),
                    BN_GGUF_TENSOR_IQ3_S, 1, BN_QK_K, xk, &ref_iq3s);

    run_vector_attention_reference_case(gpu);
    run_cuda_execution_policy_isolation();
    run_q8_prepared_reference_case();
    run_q8_small_batch_reference_case(gpu);
    run_q8_routed_reference_case(0);
    run_q8_routed_reference_case(1);
    run_combined_router_reference_case();
    run_q8_mmq_reference_case(gpu);
    run_f32_matrix_reference_case(gpu);
    run_f32_ffn_entry_reference_case(gpu);
    run_moe_dense_residual_composition_case(gpu);
    run_moe_routed_dense_residual_case(gpu);
    run_shared_batch_reference_case(gpu);
    run_rmsnorm_batch_reference_case(gpu);
    run_standalone_ffn_reference_case(gpu);
    run_moe_residual_order_reference_case(gpu);
    run_shared_gate_reference_case(gpu);
    run_router_reference_case(gpu);
    run_router512_reference_cases(gpu);
    run_router_mmvf_reference_case(gpu);
    run_router_matrix_reference_case(gpu);
    run_buffer_upload_readiness_case(gpu);
    run_activation_upload_readiness_case(gpu);
    run_rope_preparation_case(gpu);
    run_norm_residual_reference_cases(gpu);
    run_prepared_v_reference_cases(gpu);
    run_routed_q4_reference_cases(gpu);
    run_merged_routed_q4_reference_cases(gpu);
    run_gelu_prefill_reference_cases(gpu);
    run_expert_reduction_reference_cases(gpu);
    run_expert_batch_reduction_reference_cases(gpu);
    run_scaled_rmsnorm_reference_cases(gpu);
    run_scaled_rmsnorm_batch_reference_cases(gpu);
    run_rmsnorm_width_cases(gpu);
    run_per_head_rmsnorm_case(gpu);
    run_mmq_input_rounding_case(gpu);
    run_q4k_mmq_original_sum_case(gpu);
    run_silu_graph_accuracy_case(gpu);
    run_ssm_gateup_layout_case(gpu);
    run_ssm_projection_state_reference_case(gpu);
    run_ssm_alpha_beta_reference_case(gpu);
    run_ssm_gate_composition_case(gpu);
    run_ssm_delta_reference_case(gpu);
    run_ssm_conv_reference_case(gpu);
    run_ssm_l2_epsilon_case(gpu);
    run_sigmoid_gate_accuracy_case(gpu);
    run_gelu_gate_accuracy_case(gpu);
    run_q6k_block32_graph_case(gpu);
    run_gated_q_utility_case(gpu);
    run_qk_rope_reference_case(gpu);
    run_prefix_attention_reference_cases(gpu, 0);
    run_prepared_attention_reference_case(gpu);
    run_standalone_rope_reference_case(gpu);
    run_raw_attention_mma_reference_case(gpu);
    run_model_attention_reference_case(gpu);
    run_small_attention_reference_case(gpu);
    run_gqa2_attention_reference_case(gpu);
    run_q6_small_reference_case(gpu);
    run_kquant_mmq_reference_case(gpu);
    run_signed_mmq_reference_case(gpu);
    run_iq3s_reference_case(gpu);
    run_iq4xs_reference_case(gpu);
    run_iq4nl_reference_case(gpu);
    run_q4_original_sum_reference_case(gpu);
    run_mxfp4_routed_reference_cases(gpu,2);
    run_mxfp4_routed_reference_cases(gpu,1);
    run_q5q6_routed_reference_cases(gpu,0);
    run_q5q6_routed_reference_cases(gpu,1);
    run_q451_routed_reference_cases(gpu,0);
    run_q451_routed_reference_cases(gpu,1);
    run_mxfp4_routed_reference_cases(gpu,0);
    run_mxfp4_reference_case(gpu);
    run_q3k_reference_case(gpu);
    run_q8_reference_case(gpu);
    run_q8_split_reference_case(gpu);
    run_dense_gateup_reference_case(gpu);
    run_dense_qk_reference_case(gpu);
    run_dense_prefill_reference_case(gpu);
    run_graph_teardown_case();
    run_decode_attention_f32_reference_cases(gpu);
    run_decode_attention_reference_cases(gpu, 0, 0);
    run_norm_residual_reference_case(gpu);
    run_q5_decode_reference_case(gpu);
    run_q4_decode_reference_case(gpu);
    run_head512_attention_reference_case(gpu);
    run_split_attention_reference_case(gpu);
    run_prefill_window_reference_cases(gpu, 0);
    run_prefill_attention_case(gpu, 1);
    run_prefill_attention_case(gpu, 0);
    run_moe_prefill_asymmetric_residual_case(gpu);

    /* Quant-only upload omits optional layouts; batched execution must
     * still produce correct output using the quantized buffers. */
    BnGPUBackend quant_only_gpu = *gpu;
    quant_only_gpu.buffer_create = gpu->buffer_create_quant_only;
    assert(quant_only_gpu.buffer_create);
    run_wide_q4k_matmul_case(&quant_only_gpu, 17);
    run_wide_q4k_matmul_case(&quant_only_gpu, 65);
    run_wide_q5k_matmul_case(&quant_only_gpu, 17);

    bn_gpu_cuda_destroy(gpu);

    setenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_NATIVE_DOT", "1", 1);
    gpu = bn_gpu_cuda_create();
    assert(gpu != NULL);
    run_q4k_matmul_batch_case(gpu);
    bn_gpu_cuda_destroy(gpu);
    unsetenv("BN_CUDA_ENABLE_ASYMMETRIC_KQUANT_NATIVE_DOT");

    setenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT", "1", 1);
    gpu = bn_gpu_cuda_create();
    assert(gpu != NULL);
    run_wide_q4k_matmul_case(gpu, 17);
    run_wide_q4k_matmul_case(gpu, 33);
    run_wide_q4k_matmul_case(gpu, 65);
    run_wide_q5k_matmul_case(gpu, 17);
    run_wide_q5k_matmul_case(gpu, 33);
    run_wide_q5k_matmul_case(gpu, 65);
    run_wide_q5k_matmul_case(gpu, 128);
    run_wide_q5k_matmul_case(gpu, 129);
    bn_gpu_cuda_destroy(gpu);
    unsetenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT");

    printf("CUDA backend test PASSED\n");
    return 0;
#endif
}
