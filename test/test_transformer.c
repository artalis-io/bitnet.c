#include "transformer.h"
#include "transformer_cpu_internal.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_batched_attn_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_logits_internal.h"
#include "transformer_math_internal.h"
#include "transformer_prefill_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "../src/transformer/gpu_internal.h"
#include "../src/gpu_shader.h"
#include "transformer_plan_internal.h"
#include "backend_model.h"
#include "backend_session.h"
#include "gpu_policy.h"
#include "model_arch.h"
#include "model_internal.h"
#include "session_internal.h"
#include "quant.h"
#include "simd_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <limits.h>

typedef struct {
    float x[4];
    float rope[2];
    float per_layer[2];
    int fail_buf;
} MockTokenStaging;

static int mock_token_staging_write(void *ctx, int buf_idx,
                                    const void *data, size_t size,
                                    size_t offset) {
    MockTokenStaging *state = (MockTokenStaging *)ctx;
    if (buf_idx == state->fail_buf) return -1;
    assert(offset == 0);
    float *dst = NULL;
    size_t capacity = 0;
    switch (buf_idx) {
        case BN_GPU_VALUE_X:
            dst = state->x; capacity = sizeof(state->x); break;
        case BN_GPU_VALUE_ROPE_FREQ:
            dst = state->rope; capacity = sizeof(state->rope); break;
        case BN_GPU_VALUE_PER_LAYER_INPUT:
            dst = state->per_layer; capacity = sizeof(state->per_layer); break;
        default: assert(0); return -1;
    }
    assert(size == capacity);
    memcpy(dst, data, size);
    return 0;
}

static void test_gpu_token_staging_preserves_prepared_rope(void) {
    float embeddings[] = {1, 2, 3, 4, -1, -2, -3, -4};
    float per_layer[] = {5, 6};
    /* Backend preparation may round differently from CPU frequency setup. */
    const float prepared_rope[] = {1.0f, 0.010000001f};
    BnLayerWeights layer = {0};
    BnModel model = {0};
    BnSession session = {0};
    MockTokenStaging state = {.fail_buf = -1};
    BnGPUBackend gpu = {0};
    model.config.dim = 4;
    model.config.vocab_size = 2;
    model.config.n_layers = 1;
    model.config.head_size = 4;
    model.config.rope_dim_count = 4;
    model.config.rope_theta = 10000.0f;
    model.weights.layers = &layer;
    model.weights.emb_type = BN_GGUF_TENSOR_F32;
    model.weights.token_embedding = embeddings;
    session.state.per_layer_input = per_layer;
    gpu.ctx = &state;
    gpu.write_activation = mock_token_staging_write;
    memcpy(state.rope, prepared_rope, sizeof(prepared_rope));
    for (int use_per_layer = 0; use_per_layer < 2; use_per_layer++) {
        model.config.policy_flags = use_per_layer
            ? BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT : 0;
        model.config.per_layer_input_dim = use_per_layer ? 2 : 0;
        for (int token = 0; token < 2; token++) {
            per_layer[0] = (float)(token + 5);
            assert(bn_transformer_gpu_stage_token_input(
                       &gpu, &model, &session, token) == 0);
            for (int i = 0; i < 4; i++)
                assert(state.x[i] == embeddings[token * 4 + i] *
                       (use_per_layer ? 2.0f : 1.0f));
            assert(memcmp(state.rope, prepared_rope, sizeof(prepared_rope)) == 0);
            if (use_per_layer)
                assert(memcmp(state.per_layer, per_layer, sizeof(per_layer)) == 0);
        }
    }
    state.fail_buf = BN_GPU_VALUE_X;
    assert(bn_transformer_gpu_stage_token_input(&gpu, &model, &session, 0) == -1);
    state.fail_buf = BN_GPU_VALUE_PER_LAYER_INPUT;
    assert(bn_transformer_gpu_stage_token_input(&gpu, &model, &session, 0) == -1);
    state.fail_buf = -1;
    session.state.per_layer_input = NULL;
    assert(bn_transformer_gpu_stage_token_input(&gpu, &model, &session, 0) == -1);
    assert(memcmp(state.rope, prepared_rope, sizeof(prepared_rope)) == 0);
    printf("test_gpu_token_staging_preserves_prepared_rope... PASSED\n");
}

static void test_scaled_silu_order(void) {
    printf("test_scaled_silu_order... ");
    bn_transformer_scaled_silu(NULL, 1.0f, 0);
    const int sizes[] = {1, 7, 8, 15, 16, 17, 31, 32, 33, 320, 321};
    const float scales[] = {0.25f, 1.0f / 3.0f, -0.7f, 0.0f};
    float actual[323], expected[321];
    for (size_t s = 0; s < sizeof(scales) / sizeof(scales[0]); s++) {
        for (size_t z = 0; z < sizeof(sizes) / sizeof(sizes[0]); z++) {
            int n = sizes[z];
            for (int i = 0; i < 323; i++) actual[i] = NAN;
            for (int i = 0; i < n; i++) {
                actual[i + 1] = sinf((float)i * 0.37f) * 25.0f;
                expected[i] = actual[i + 1] * scales[s];
            }
            int i = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && !defined(BN_FORCE_SCALAR)
            for (; i + 15 < n; i += 16)
                _mm512_storeu_ps(expected + i,
                    bn_avx512_fast_silu_ps(_mm512_loadu_ps(expected + i)));
#elif defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
            for (; i + 7 < n; i += 8) {
                __m256 value = _mm256_loadu_ps(expected + i);
                __m256 exp_neg = bn_avx2_fast_exp_avx512_ps(
                    _mm256_sub_ps(_mm256_setzero_ps(), value));
                _mm256_storeu_ps(expected + i, _mm256_div_ps(
                    value, _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg)));
            }
#endif
            for (; i < n; i++)
                expected[i] = expected[i] / (1.0f + expf(-expected[i]));
            bn_transformer_scaled_silu(actual + 1, scales[s], n);
            assert(memcmp(actual + 1, expected, (size_t)n * sizeof(float)) == 0);
            assert(isnan(actual[0]));
            for (i = n + 1; i < 323; i++) assert(isnan(actual[i]));
        }
    }
    printf("PASSED\n");
}

static void test_dilated_conv_and_branch_order(void) {
    printf("test_dilated_conv_and_branch_order... ");
    bn_transformer_dilated_conv_silu(NULL, NULL, NULL, NULL, 0, 1, 1);
    bn_transformer_scaled_branch_add(NULL, NULL, 1, NULL, 0);
    enum { max_n = 33, max_hist = 9 * max_n };
    const int sizes[] = {1, 7, 8, 15, 16, 17, 32, 33};
    float current[max_n + 2], out[max_n + 2], expected[max_n];
    float weights[4 * max_n], history[max_hist], saved[max_hist];
    for (size_t z = 0; z < sizeof(sizes) / sizeof(sizes[0]); z++) {
        int n = sizes[z];
        for (int kernel = 1; kernel <= 4; kernel++) {
            for (int dilation = 1; dilation <= 3; dilation++) {
                for (int i = 0; i < max_hist; i++) history[i] = NAN;
                for (int i = 0; i < max_n + 2; i++)
                    current[i] = out[i] = NAN;
                for (int i = 0; i < n; i++) {
                    current[i + 1] = (float)(i % 11 - 5) / 7;
                    for (int k = 0; k < kernel; k++) {
                        weights[i * kernel + k] = (float)((i + k) % 7 - 3) / 5;
                        if (k < kernel - 1)
                            history[k * dilation * n + i] =
                                (float)((i + 3 * k) % 13 - 6) / 9;
                    }
                }
                memcpy(saved, history, sizeof(history));
                for (int i = 0; i < n; i++) {
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
                    float sum = 0;
                    for (int k = 0; k < kernel; k++) {
                        float v = k == kernel - 1 ? current[i + 1] :
                            history[k * dilation * n + i];
                        volatile float p = v * weights[i * kernel + k];
                        sum = k == 0 ? p : sum + p;
                    }
#else
                    float sum = current[i + 1] * weights[i * kernel + kernel - 1];
                    for (int k = 0; k < kernel - 1; k++)
                        sum += history[k * dilation * n + i] * weights[i * kernel + k];
#endif
                    expected[i] = sum / (1.0f + expf(-sum));
                }
                bn_transformer_dilated_conv_silu(out + 1, current + 1,
                    history, weights, n, kernel, dilation);
                for (int i = 0; i < n; i++)
                    assert(fabsf(out[i + 1] - expected[i]) < 2e-6f);
                assert(isnan(out[0]) && isnan(out[n + 1]));
                assert(memcmp(saved, history, sizeof(history)) == 0);
                bn_transformer_dilated_conv_silu(current + 1, current + 1,
                    history, weights, n, kernel, dilation);
                assert(memcmp(out + 1, current + 1, (size_t)n * sizeof(float)) == 0);
                assert(isnan(current[0]) && isnan(current[n + 1]));
            }
        }
    }
    // Distinguish oldest-first from current-first convolution accumulation.
    for (int i = 0; i < max_n; i++) {
        current[i + 1] = 1;
        history[i] = 0x1p24f;
        history[max_n + i] = -0x1p24f;
        for (int k = 0; k < 3; k++) weights[i * 3 + k] = 1;
    }
    bn_transformer_dilated_conv_silu(out + 1, current + 1, history,
        weights, max_n, 3, 1);
    for (int i = 0; i < max_n; i++) {
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
        assert(fabsf(out[i + 1] - 1.0f / (1.0f + expf(-1))) < 2e-6f);
#else
        assert(out[i + 1] == 0);
#endif
    }
    // Distinguish residual grouping; guards also cover unaligned tail lengths.
    for (int i = 0; i < max_n; i++) {
        out[i + 1] = 0x1p25f;
        current[i + 1] = -0x1p25f;
        expected[i] = 1;
    }
    bn_transformer_scaled_branch_add(out + 1, current + 1, 1, expected, max_n);
    for (int i = 0; i < max_n; i++) {
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
        assert(out[i + 1] == 0);
#else
        assert(out[i + 1] == 1);
#endif
    }
    assert(isnan(out[0]) && isnan(out[max_n + 1]));
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
    // A fused multiply-add would preserve a spurious low-order term.
    out[1] = 0;
    current[1] = 0x1.000002p0f;
    expected[0] = -0x1.000004p0f;
    bn_transformer_scaled_branch_add(out + 1, current + 1,
        0x1.000002p0f, expected, 1);
    assert(out[1] == 0);
#endif
    printf("PASSED\n");
}

static void test_sum_products_rounding(void) {
    printf("test_sum_products_rounding... ");
    assert(bn_transformer_sum_products(NULL, NULL, 0) == 0.0f);
    const float a[] = {1, 2, 3}, b[] = {4, 5, 6};
    assert(bn_transformer_sum_products(a, b, 3) == 32.0f);
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
    const float cancel[] = {0x1p24f, 1, -0x1p24f};
    const float ones[] = {1, 1, 1};
    assert(bn_transformer_sum_products(cancel, ones, 3) == 1.0f);
    const float rounded_a[] = {0x1.000002p0f, -0x1.000004p0f};
    const float rounded_b[] = {0x1.000002p0f, 1};
    assert(bn_transformer_sum_products(rounded_a, rounded_b, 2) == 0.0f);
    assert((double)rounded_a[0] * rounded_b[0] + rounded_a[1] != 0.0);
    const int sizes[] = {1, 3, 7, 8, 15, 16, 31, 32, 33, 320, 2560};
    float x[2562], y[2562];
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++) {
        int n = sizes[k];
        x[0] = y[0] = x[n + 1] = y[n + 1] = NAN;
        double expected = 0.0;
        for (int i = 0; i < n; i++) {
            x[i + 1] = (float)((i * 31) % 127 - 63) / 17.0f;
            y[i + 1] = (float)((i * 13) % 97 - 48) / 19.0f;
            volatile float product = x[i + 1] * y[i + 1];
            expected += (double)product;
        }
        assert(bn_transformer_sum_products(x + 1, y + 1, n) == (float)expected);
        assert(isnan(x[0]) && isnan(y[0]));
        assert(isnan(x[n + 1]) && isnan(y[n + 1]));
    }
#endif
    printf("PASSED\n");
}

static void test_scaled_residual_rounding(void) {
    printf("test_scaled_residual_rounding... ");
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
    enum { max_dim = 33, streams = 4 };
    const int dims[] = {1, 7, 8, 15, 16, 17, 32, 33};
    float input[max_dim + 2], residual[streams * max_dim + 2];
    float scratch[max_dim + 2], expected[streams * max_dim];
    float scale = 0x1.000002p0f;
    for (int i = 0; i < max_dim + 2; i++) input[i] = scale;
    assert(fmaf(scale, scale, -0x1.000004p0f) != 0.0f);
    for (size_t d = 0; d < sizeof(dims) / sizeof(dims[0]); d++) {
        int dim = dims[d];
        for (int i = 0; i < streams * max_dim + 2; i++) residual[i] = NAN;
        for (int i = 0; i < max_dim + 2; i++) scratch[i] = NAN;
        for (int i = 0; i < dim; i++) residual[i + 1] = -0x1.000004p0f;
        bn_transformer_cpu_scaled_residual_add(NULL, residual + 1,
            input + 1, scale, dim, scratch + 1);
        for (int i = 0; i < dim; i++) assert(residual[i + 1] == 0.0f);
        assert(isnan(residual[0]) && isnan(residual[dim + 1]));
        assert(isnan(scratch[0]) && isnan(scratch[dim + 1]));

        BnModel m = {0};
        BnSession sess = {0};
        float inject[streams] = {-1.3f, 0.001f, 2.7f, -0.72f};
        m.config.dim = dim;
        m.config.hyper_connection_count = streams;
        sess.state.hc_inject = inject;
        sess.state.hc_norm = scratch + 1;
        sess.state.hc_residual = residual + 1;
        for (int j = 0; j < streams; j++) {
            float scatter = 2.0f / (1.0f + expf(-inject[j] * 0.25f));
            for (int i = 0; i < dim; i++) {
                volatile float product = input[i + 1] * scatter;
                residual[1 + j * dim + i] = -product;
                expected[j * dim + i] = 0.0f;
            }
        }
        bn_transformer_cpu_hyper_connection_combine(&m, &sess, input + 1);
        assert(memcmp(residual + 1, expected, (size_t)streams * dim * sizeof(float)) == 0);
        assert(isnan(residual[0]) && isnan(residual[streams * dim + 1]));
    }
#endif
    printf("PASSED\n");
}

static const BnCPURuntimePolicy *test_cpu_policy(void) {
    static BnCPURuntimePolicy policy;
    bn_cpu_runtime_policy_from_env(&policy);
    return &policy;
}

static void test_gpu_runtime_refresh(BnGPUBackend *gpu) {
    bn_backend_runtime_policy_free(&gpu->runtime_policy);
    assert(bn_gpu_backend_runtime_policy_init(&gpu->runtime_policy) == 0);
}

static BnBackendRuntimePolicy *test_current_backend_runtime(void) {
    static BnBackendRuntimePolicy policy;
    static int initialized;
    if (initialized) bn_backend_runtime_policy_free(&policy);
    assert(bn_gpu_backend_runtime_policy_init(&policy) == 0);
    initialized = 1;
    return &policy;
}

// Test the helper functions that would be internal to transformer.c
// We re-implement them here for testing since they're static in transformer.c

static void rmsnorm(float *out, const float *x, const float *w, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static int mock_gpu_execute(void *ctx, const void *ops, int n_ops,
                            int readback_buf, float *out_host,
                            int out_len) {
    (void)ctx;
    (void)ops;
    (void)n_ops;
    (void)readback_buf;
    (void)out_host;
    (void)out_len;
    return 0;
}

static int mock_gpu_write_activation(void *ctx, int buf_idx,
                                     const void *data, size_t n_bytes,
                                     size_t offset_bytes) {
    (void)ctx;
    (void)buf_idx;
    (void)data;
    (void)n_bytes;
    (void)offset_bytes;
    return 0;
}

typedef struct {
    const float *logits;
    const float *xb;
} MockGPURefineReadback;

static int mock_gpu_refine_read_activation(
    void *ctx, int buf_idx, void *out, size_t size, size_t offset) {
    MockGPURefineReadback *readback = (MockGPURefineReadback *)ctx;
    const float *src = buf_idx == BN_GPU_VALUE_LOGITS
        ? readback->logits
        : buf_idx == BN_GPU_VALUE_XB ? readback->xb : NULL;
    if (!src)
        return -1;
    memcpy(out, (const unsigned char *)src + offset, size);
    return 0;
}

static int mock_argmax_activation(void *ctx, int buf_idx, int n,
                                  const int *penalty_tokens,
                                  int n_penalty_tokens,
                                  float repeat_penalty,
                                  int *out_token) {
    (void)ctx;
    (void)buf_idx;
    (void)n;
    (void)penalty_tokens;
    (void)n_penalty_tokens;
    (void)repeat_penalty;
    (void)out_token;
    return 0;
}

static int mock_matvec_argmax_activation(
    void *ctx, void *W_buf, int type, int rows, int cols, int buf_idx,
    const int *penalty_tokens, int n_penalty_tokens, float repeat_penalty,
    int *out_token) {
    (void)ctx; (void)W_buf; (void)type; (void)rows; (void)cols;
    (void)buf_idx; (void)penalty_tokens; (void)n_penalty_tokens;
    (void)repeat_penalty; (void)out_token;
    return 0;
}

static int mock_gpu_matmul(void *ctx, float *out, void *W_buf,
                           const float *X, int rows, int cols,
                           int n_tokens, int type) {
    (void)ctx; (void)out; (void)W_buf; (void)X;
    (void)rows; (void)cols; (void)n_tokens; (void)type;
    return 0;
}

static int mock_gpu_matmul_batch(void *ctx, const BnGPUMatvecOp *ops,
                                 int n_ops, const float *X, int n_tokens,
                                 int cols) {
    (void)ctx; (void)ops; (void)n_ops; (void)X;
    (void)n_tokens; (void)cols;
    return 0;
}

static int mock_dense_ffn_batch(
    void *ctx, float *out, void *gate_buf, void *up_buf, void *down_buf,
    const float *X, int n_tokens, int dim, int hidden_dim, int gate_type,
    int up_type, int down_type, int act_type) {
    (void)ctx; (void)out; (void)gate_buf; (void)up_buf;
    (void)down_buf; (void)X; (void)n_tokens; (void)dim;
    (void)hidden_dim; (void)gate_type; (void)up_type;
    (void)down_type; (void)act_type;
    return 0;
}

static int mock_routed_expert_batch(
    void *ctx, float *out, void *gate, void *up, void *down,
    const float *X, int nt, int dim, int hidden, int gate_type,
    int up_type, int down_type, int act_type,
    const BnGPUMoEExpertBatchPlan *plan) {
    (void)out; (void)gate; (void)up; (void)down; (void)X;
    (void)dim; (void)hidden; (void)gate_type; (void)up_type;
    (void)down_type; (void)act_type;
    assert(plan == ctx);
    assert(nt == 2 && plan->total_tokens == 17);
    assert(plan->n_experts == 128 && plan->expert_index == 31);
    return -7;
}

static int test_scaled_norm_callback(void *ctx, float *out, void *norm,
    const float *x, int nt, int dim, float eps, float post_scale) {
    int *calls = ctx;
    (*calls)++;
    assert(norm == x && nt == 1 && dim == 1);
    assert(eps == 1e-6f && post_scale == 0.5f);
    *out = 17.0f;
    return 0;
}

static void test_prefill_scaled_norm_raw_weight_contract(void) {
    int calls = 0;
    float x = 1.0f, out = 9.0f;
    BnGPUBackend gpu = {0};
    gpu.ctx = &calls;
    gpu.rmsnorm_scaled_batch = test_scaled_norm_callback;
    assert(bn_transformer_gpu_prefill_scaled_rmsnorm_backend_run(
        &gpu, &out, &x, &x, 1, 1, 1e-6f, 0.5f) == -1);
    assert(calls == 0 && out == 9.0f);
    gpu.caps = BN_GPU_CAP_RMSNORM_SEPARATE_SCALE;
    assert(bn_transformer_gpu_prefill_scaled_rmsnorm_backend_run(
        &gpu, &out, &x, &x, 1, 1, 1e-6f, 0.5f) == 0);
    assert(calls == 1 && out == 17.0f);
    gpu.rmsnorm_scaled_batch = NULL;
    assert(bn_transformer_gpu_prefill_scaled_rmsnorm_backend_run(
        &gpu, &out, &x, &x, 1, 1, 1e-6f, 0.5f) == -1);
    printf("test_prefill_scaled_norm_raw_weight_contract PASSED\n");
}

static void test_prefill_logical_projection_rows(void) {
    BnGPUBackend gpu = {0};
    assert(bn_transformer_gpu_prefill_stacked_projection_allowed(&gpu, BN_GGUF_TENSOR_Q4_0, 14));
    gpu.caps = BN_GPU_CAP_PREFILL_LOGICAL_PROJECTION_ROWS;
    assert(bn_transformer_gpu_prefill_stacked_projection_allowed(&gpu, BN_GGUF_TENSOR_Q4_0, 8));
    assert(!bn_transformer_gpu_prefill_stacked_projection_allowed(&gpu, BN_GGUF_TENSOR_Q4_0, 9));
    assert(!bn_transformer_gpu_prefill_stacked_projection_allowed(&gpu, BN_GGUF_TENSOR_Q4_0, 33));
    assert(bn_transformer_gpu_prefill_stacked_projection_allowed(&gpu, BN_GGUF_TENSOR_F32, 14));
    assert(bn_transformer_gpu_prefill_stacked_projection_allowed(NULL, BN_GGUF_TENSOR_Q4_0, 14));
    printf("test_prefill_logical_projection_rows PASSED\n");
}

static void test_prefill_expert_batch_geometry(void) {
    BnGPUMoEExpertBatchPlan plan = {17, 128, 31, 0};
    BnGPUBackend gpu = {0};
    gpu.ctx = &plan;
    gpu.dense_ffn_batch = mock_dense_ffn_batch;
    gpu.moe_expert_ffn_batch = mock_routed_expert_batch;
    // A rejected routed call must not silently retry with dense geometry.
    assert(bn_transformer_gpu_prefill_expert_ffn_batch_backend_run(&gpu,
        NULL, NULL, NULL, NULL, NULL, 2, 96, 64, BN_GGUF_TENSOR_Q4_0,
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0,
        BN_MODEL_ACTIVATION_GELU, &plan) == -7);
    printf("test_prefill_expert_batch_geometry PASSED\n");
}

static int mock_prefill_qkv_attention_wo(
    void *ctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
    void *q_norm_buf, void *k_norm_buf, const float *X, float *K_out,
    float *V_out, int n_tokens, int dim, int n_heads, int n_kv_heads,
    int head_size, int kv_mul, int kv_dim, int qk_rows, int qk_type,
    int wv_rows, int wv_type, int wo_rows, int wo_cols, int wo_type,
    int qk_norm_per_head, float norm_eps, int pos0, int rope_dims,
    float attention_scale, int attention_window) {
    (void)attention_window;
    (void)ctx; (void)out; (void)qk_buf; (void)wv_buf; (void)wo_buf;
    (void)q_norm_buf; (void)k_norm_buf; (void)X; (void)K_out;
    (void)V_out; (void)n_tokens; (void)dim; (void)n_heads;
    (void)n_kv_heads; (void)head_size; (void)kv_mul; (void)kv_dim;
    (void)qk_rows; (void)qk_type; (void)wv_rows; (void)wv_type;
    (void)wo_rows; (void)wo_cols; (void)wo_type; (void)qk_norm_per_head;
    (void)norm_eps; (void)pos0; (void)rope_dims; (void)attention_scale;
    return 0;
}

static int mock_prefill_qkv_attention_wo_norm_resid(
    void *ctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
    void *attn_norm_buf, void *q_norm_buf, void *k_norm_buf,
    const float *X, float *K_out, float *V_out, int n_tokens, int dim,
    int n_heads, int n_kv_heads, int head_size, int kv_mul, int kv_dim,
    int qk_rows, int qk_type, int wv_rows, int wv_type, int wo_rows,
    int wo_cols, int wo_type, int qk_norm_per_head, float norm_eps,
    int pos0, int rope_dims, float attention_scale, int attention_window) {
    (void)attention_window;
    (void)ctx; (void)out; (void)qk_buf; (void)wv_buf; (void)wo_buf;
    (void)attn_norm_buf; (void)q_norm_buf; (void)k_norm_buf;
    (void)X; (void)K_out; (void)V_out; (void)n_tokens; (void)dim;
    (void)n_heads; (void)n_kv_heads; (void)head_size; (void)kv_mul;
    (void)kv_dim; (void)qk_rows; (void)qk_type; (void)wv_rows;
    (void)wv_type; (void)wo_rows; (void)wo_cols; (void)wo_type;
    (void)qk_norm_per_head; (void)norm_eps; (void)pos0;
    (void)rope_dims; (void)attention_scale;
    return 0;
}

static int mock_prefill_attention(void *ctx, float *out, const float *Q,
                                  const float *K, const float *V,
                                  int n_tokens, int n_heads,
                                  int n_kv_heads, int head_size,
                                  int kv_mul, int kv_dim,
                                  float attention_scale, int attention_window) {
    (void)attention_window;
    (void)ctx; (void)out; (void)Q; (void)K; (void)V;
    (void)n_tokens; (void)n_heads; (void)n_kv_heads;
    (void)head_size; (void)kv_mul; (void)kv_dim;
    (void)attention_scale;
    return 0;
}

typedef struct {
    float *out, *k_out;
    const float *q, *k, *v;
    void *q_norm, *k_norm;
    const BnGPUAttentionPrefillPlan *plan;
    int calls;
} MockPreparedAttention;

static int mock_prefill_attention_prepared(void *ctx, float *out, float *k_out,
    const float *q, const float *k, const float *v,
    void *q_norm, void *k_norm, const BnGPUAttentionPrefillPlan *plan) {
    MockPreparedAttention *expected = (MockPreparedAttention *)ctx;
    assert(out == expected->out && k_out == expected->k_out);
    assert(q == expected->q && k == expected->k && v == expected->v);
    assert(q_norm == expected->q_norm && k_norm == expected->k_norm);
    assert(plan == expected->plan);
    expected->calls++;
    return -7;
}

static int mock_prefill_attention_wo(
    void *ctx, float *out, void *wo_buf, const float *Q, const float *K,
    const float *V, int n_tokens, int n_heads, int n_kv_heads,
    int head_size, int kv_mul, int kv_dim, int wo_rows, int wo_cols,
    int wo_type, float attention_scale, int attention_window) {
    (void)attention_window;
    (void)ctx; (void)out; (void)wo_buf; (void)Q; (void)K; (void)V;
    (void)n_tokens; (void)n_heads; (void)n_kv_heads; (void)head_size;
    (void)kv_mul; (void)kv_dim; (void)wo_rows; (void)wo_cols;
    (void)wo_type; (void)attention_scale;
    return 0;
}

static int mock_prefill_dense_layer(
    void *ctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
    void *gate_buf, void *up_buf, void *down_buf, void *attn_norm_buf,
    void *ffn_norm_buf, void *attn_post_norm_buf,
    void *ffn_post_norm_buf, void *q_norm_buf, void *k_norm_buf,
    void *q_bias_buf, void *k_bias_buf, void *v_bias_buf,
    const float *X, float *K_out, float *V_out, int n_tokens, int dim,
    int hidden_dim, int n_heads, int n_kv_heads, int head_size,
    int kv_mul, int kv_dim, int qk_rows, int qk_type, int wv_rows,
    int wv_type, int wo_rows, int wo_cols, int wo_type, int gate_type,
    int up_type, int down_type, int act_type, int qk_norm_per_head,
    int normalize_v, float norm_eps, int pos0, int rope_dims,
    size_t rope_freq_offset,
    uint32_t kv_cache_off, int kv_cache_stride, float attention_scale,
    float layer_output_scale, int attention_window,
    int final_ffn_last_row_only) {
    (void)attention_window;
    (void)ctx; (void)out; (void)qk_buf; (void)wv_buf; (void)wo_buf;
    (void)gate_buf; (void)up_buf; (void)down_buf; (void)attn_norm_buf;
    (void)ffn_norm_buf; (void)attn_post_norm_buf;
    (void)ffn_post_norm_buf; (void)q_norm_buf; (void)k_norm_buf;
    (void)q_bias_buf; (void)k_bias_buf; (void)v_bias_buf; (void)X;
    (void)K_out; (void)V_out; (void)n_tokens; (void)dim;
    (void)hidden_dim; (void)n_heads; (void)n_kv_heads; (void)head_size;
    (void)kv_mul; (void)kv_dim; (void)qk_rows; (void)qk_type;
    (void)wv_rows; (void)wv_type; (void)wo_rows; (void)wo_cols;
    (void)wo_type; (void)gate_type; (void)up_type; (void)down_type;
    (void)act_type; (void)qk_norm_per_head; (void)norm_eps;
    (void)normalize_v;
    (void)pos0; (void)rope_dims; (void)kv_cache_off;
    (void)rope_freq_offset; (void)kv_cache_stride; (void)attention_scale;
    (void)layer_output_scale;
    (void)final_ffn_last_row_only;
    return 0;
}

static int mock_dense_ffn(
    void *ctx, float *out, void *gate_buf, void *up_buf, void *down_buf,
    const float *x, int dim, int hidden_dim, int gate_type, int up_type,
    int down_type, int act_type) {
    (void)ctx; (void)out; (void)gate_buf; (void)up_buf;
    (void)down_buf; (void)x; (void)dim; (void)hidden_dim;
    (void)gate_type; (void)up_type; (void)down_type; (void)act_type;
    return 0;
}

static int mock_moe_route_routed_ffn_batch_norm_resid(
    void *ctx, float *out, void *router_buf, void *gate_all_buf,
    void *up_all_buf, void *down_all_buf, void *shared_gate_buf,
    void *shared_up_buf, void *shared_down_buf, void *shared_gate_weight_buf,
    void *norm_buf, const float *X, int n_tokens, int dim, int hidden_dim,
    int n_experts, int k, int gate_type, int up_type, int down_type,
    int act_type, int shared_hidden_dim, int shared_gate_type,
    int shared_up_type, int shared_down_type, float norm_eps,
    int norm_topk_prob, float expert_weights_scale) {
    (void)ctx; (void)out; (void)router_buf; (void)gate_all_buf;
    (void)up_all_buf; (void)down_all_buf; (void)shared_gate_buf;
    (void)shared_up_buf; (void)shared_down_buf;
    (void)shared_gate_weight_buf; (void)norm_buf; (void)X;
    (void)n_tokens; (void)dim; (void)hidden_dim; (void)n_experts;
    (void)k; (void)gate_type; (void)up_type; (void)down_type;
    (void)act_type; (void)shared_hidden_dim; (void)shared_gate_type;
    (void)shared_up_type; (void)shared_down_type; (void)norm_eps;
    (void)norm_topk_prob; (void)expert_weights_scale;
    return 0;
}

static int mock_moe_route_routed_ffn_batch(
    void *ctx, float *out, void *router_buf, void *gate_all_buf,
    void *up_all_buf, void *down_all_buf, const float *X, int n_tokens,
    int dim, int hidden_dim, int n_experts, int k, int gate_type,
    int up_type, int down_type, int act_type, int norm_topk_prob,
    float expert_weights_scale) {
    (void)ctx; (void)out; (void)router_buf; (void)gate_all_buf;
    (void)up_all_buf; (void)down_all_buf; (void)X; (void)n_tokens;
    (void)dim; (void)hidden_dim; (void)n_experts; (void)k;
    (void)gate_type; (void)up_type; (void)down_type; (void)act_type;
    (void)norm_topk_prob; (void)expert_weights_scale;
    return 0;
}

static int mock_moe_route_batch(
    void *ctx, int *indices, float *weights, void *router_buf,
    const float *X, int n_tokens, int dim, int n_experts, int k,
    int norm_topk_prob, float expert_weights_scale) {
    (void)ctx; (void)indices; (void)weights; (void)router_buf;
    (void)X; (void)n_tokens; (void)dim; (void)n_experts; (void)k;
    (void)norm_topk_prob; (void)expert_weights_scale;
    return 0;
}

static int mock_moe_routed_ffn_batch(
    void *ctx, float *out, void *gate_all_buf, void *up_all_buf,
    void *down_all_buf, const int *indices, const float *weights,
    const float *output_scales,
    const float *X, int n_tokens, int dim, int hidden_dim,
    int n_experts, int k, int gate_type, int up_type, int down_type,
    int act_type) {
    (void)ctx; (void)out; (void)gate_all_buf; (void)up_all_buf;
    (void)down_all_buf; (void)indices; (void)weights;
    (void)output_scales; (void)X;
    (void)n_tokens; (void)dim; (void)hidden_dim; (void)n_experts;
    (void)k; (void)gate_type; (void)up_type; (void)down_type;
    (void)act_type;
    return 0;
}

static int mock_moe_ffn_batch(
    void *ctx, float *out, const BnGPUMoEPrefillExpert *experts,
    int n_experts, const int *expert_offsets, const int *expert_counts,
    const int *token_ids, const float *weights, const float *X,
    int n_tokens, int dim, int hidden_dim, int gate_type, int up_type,
    int down_type, int act_type, void *shared_gate_buf, void *shared_up_buf,
    void *shared_down_buf, void *shared_gate_weight_buf,
    int shared_hidden_dim, int shared_gate_type, int shared_up_type,
    int shared_down_type) {
    (void)ctx; (void)out; (void)experts; (void)n_experts;
    (void)expert_offsets; (void)expert_counts; (void)token_ids;
    (void)weights; (void)X; (void)n_tokens; (void)dim;
    (void)hidden_dim; (void)gate_type; (void)up_type; (void)down_type;
    (void)act_type; (void)shared_gate_buf; (void)shared_up_buf;
    (void)shared_down_buf; (void)shared_gate_weight_buf;
    (void)shared_hidden_dim; (void)shared_gate_type;
    (void)shared_up_type; (void)shared_down_type;
    return 0;
}

static int mock_prefill_moe_layer(
    void *ctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
    void *router_buf, void *gate_all_buf, void *up_all_buf,
    void *down_all_buf, void *shared_gate_buf, void *shared_up_buf,
    void *shared_down_buf, void *shared_gate_weight_buf,
    void *attn_norm_buf, void *ffn_norm_buf, void *q_norm_buf,
    void *k_norm_buf, void *q_bias_buf, void *k_bias_buf,
    void *v_bias_buf, const float *X, float *K_out, float *V_out,
    int n_tokens, int dim, int moe_hidden_dim, int n_experts,
    int experts_active, int n_heads, int n_kv_heads, int head_size,
    int kv_mul, int kv_dim, int qk_rows, int qk_type, int wv_rows,
    int wv_type, int wo_rows, int wo_cols, int wo_type, int gate_type,
    int up_type, int down_type, int act_type, int shared_hidden_dim,
    int shared_gate_type, int shared_up_type, int shared_down_type,
    int qk_norm_per_head, float norm_eps, int pos0, int rope_dims,
    uint32_t kv_cache_off, int kv_cache_stride, float attention_scale,
    int norm_topk_prob, float expert_weights_scale, int attention_window) {
    (void)attention_window;
    (void)ctx; (void)out; (void)qk_buf; (void)wv_buf; (void)wo_buf;
    (void)router_buf; (void)gate_all_buf; (void)up_all_buf;
    (void)down_all_buf; (void)shared_gate_buf; (void)shared_up_buf;
    (void)shared_down_buf; (void)shared_gate_weight_buf;
    (void)attn_norm_buf; (void)ffn_norm_buf; (void)q_norm_buf;
    (void)k_norm_buf; (void)q_bias_buf; (void)k_bias_buf;
    (void)v_bias_buf; (void)X; (void)K_out; (void)V_out;
    (void)n_tokens; (void)dim; (void)moe_hidden_dim;
    (void)n_experts; (void)experts_active; (void)n_heads;
    (void)n_kv_heads; (void)head_size; (void)kv_mul; (void)kv_dim;
    (void)qk_rows; (void)qk_type; (void)wv_rows; (void)wv_type;
    (void)wo_rows; (void)wo_cols; (void)wo_type; (void)gate_type;
    (void)up_type; (void)down_type; (void)act_type;
    (void)shared_hidden_dim; (void)shared_gate_type;
    (void)shared_up_type; (void)shared_down_type;
    (void)qk_norm_per_head; (void)norm_eps; (void)pos0;
    (void)rope_dims; (void)kv_cache_off; (void)kv_cache_stride;
    (void)attention_scale; (void)norm_topk_prob;
    (void)expert_weights_scale;
    return 0;
}

static int mock_prefill_ssm_layer(
    void *ctx, float *out, void *wqkv_buf, void *wz_buf, void *alpha_buf,
    void *beta_buf, void *qkvz_stacked_buf, void *ab_stacked_buf,
    void *ssm_out_buf, void *attn_norm_buf, void *conv1d_buf,
    void *dt_bias_buf, void *a_log_buf, void *ssm_norm_buf,
    void *ffn_gate_buf, void *ffn_up_buf, void *ffn_down_buf,
    void *ffn_norm_buf, const float *X, int n_tokens, int dim,
    int qkv_dim, int inner_dim, int num_k_heads, int head_k_dim,
    int num_v_heads, int head_v_dim, int conv_kernel, int ssm_idx,
    int wqkv_type, int wz_type, int alpha_type, int beta_type,
    int out_type, int hidden_dim, int ffn_gate_type, int ffn_up_type,
    int ffn_down_type, int act_type, int sigmoid_gate, float norm_eps,
    int *did_ffn) {
    (void)ctx; (void)out; (void)wqkv_buf; (void)wz_buf;
    (void)alpha_buf; (void)beta_buf; (void)qkvz_stacked_buf;
    (void)ab_stacked_buf; (void)ssm_out_buf; (void)attn_norm_buf;
    (void)conv1d_buf; (void)dt_bias_buf; (void)a_log_buf;
    (void)ssm_norm_buf; (void)ffn_gate_buf; (void)ffn_up_buf;
    (void)ffn_down_buf; (void)ffn_norm_buf; (void)X;
    (void)n_tokens; (void)dim; (void)qkv_dim; (void)inner_dim;
    (void)num_k_heads; (void)head_k_dim; (void)num_v_heads;
    (void)head_v_dim; (void)conv_kernel; (void)ssm_idx;
    (void)wqkv_type; (void)wz_type; (void)alpha_type; (void)beta_type;
    (void)out_type; (void)hidden_dim; (void)ffn_gate_type;
    (void)ffn_up_type; (void)ffn_down_type; (void)act_type;
    (void)sigmoid_gate;
    (void)norm_eps; (void)did_ffn;
    return 0;
}

static void rope(float *vec, int dim, int head_size, int pos, float theta) {
    for (int h = 0; h < dim; h += head_size) {
        int half_rope = head_size / 2;
        for (int i = 0; i < half_rope; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_size);
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            int j = i + half_rope;
            float v0 = vec[h + i];
            float v1 = vec[h + j];
            vec[h + i] = v0 * cos_a - v1 * sin_a;
            vec[h + j] = v0 * sin_a + v1 * cos_a;
        }
    }
}

// --- Tests ---

static void test_rmsnorm_reference_contract(void) {
    printf("test_rmsnorm_reference_contract... ");
    bn_transformer_rmsnorm_reference(NULL, NULL, NULL, 0, 1e-6f);
    const int sizes[] = {1, 7, 8, 15, 16, 17, 33, 257};
    float x[259], w[259], out[259], expected[257];
    for (size_t t = 0; t < sizeof(sizes) / sizeof(sizes[0]); t++) {
        int n = sizes[t];
        for (int i = 0; i < 259; i++) x[i] = w[i] = out[i] = NAN;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            x[i + 1] = sinf((float)i * 0.23f) * 7 + 0.125f;
            w[i + 1] = (float)(i % 11 - 5) / 7;
            volatile float square = x[i + 1] * x[i + 1];
            sum += (double)square;
        }
        float scale = 1.0f / sqrtf((float)(sum / n) + 1e-6f);
        for (int i = 0; i < n; i++)
            expected[i] = (x[i + 1] * scale) * w[i + 1];
        bn_transformer_rmsnorm_reference(out + 1, x + 1, w + 1, n, 1e-6f);
        assert(memcmp(out + 1, expected, (size_t)n * sizeof(float)) == 0);
        bn_transformer_rmsnorm_reference(x + 1, x + 1, w + 1, n, 1e-6f);
        assert(memcmp(x + 1, expected, (size_t)n * sizeof(float)) == 0);
        assert(isnan(x[0]) && isnan(out[0]) && isnan(w[0]));
        for (int i = n + 1; i < 259; i++)
            assert(isnan(x[i]) && isnan(out[i]) && isnan(w[i]));
    }
    printf("PASSED\n");
}

static void test_rmsnorm(void) {
    printf("test_rmsnorm... ");

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];

    rmsnorm(out, x, w, 4, 1e-5f);

    // RMSNorm: x * 1/rms(x), where rms = sqrt(mean(x^2))
    // mean(x^2) = (1+4+9+16)/4 = 7.5
    // rms = sqrt(7.5) ≈ 2.7386
    // scale = 1/rms ≈ 0.3651
    float rms = sqrtf(7.5f + 1e-5f);
    float scale = 1.0f / rms;

    for (int i = 0; i < 4; i++) {
        float expected = x[i] * scale;
        assert(fabsf(out[i] - expected) < 1e-5f);
    }

    printf("PASSED\n");
}

static void test_rmsnorm_simd_matches_scalar_order(void) {
    printf("test_rmsnorm_simd_matches_scalar_order... ");

#if defined(__AVX2__) || defined(__ARM_NEON)
    enum { N = 2560 };
    float x[N], w[N], out_scalar[N];
    for (int i = 0; i < N; i++) {
        x[i] = sinf((float)i * 0.017f) * 3.0f + cosf((float)i * 0.031f);
        w[i] = 0.75f + 0.25f * sinf((float)i * 0.013f);
    }

    bn_transformer_rmsnorm_scalar(out_scalar, x, w, N, 1e-6f);
#ifdef __AVX2__
    float out_avx2[N];
    bn_transformer_rmsnorm_avx2(out_avx2, x, w, N, 1e-6f);

    for (int i = 0; i < N; i++)
        assert(fabsf(out_scalar[i] - out_avx2[i]) < 1e-6f);
#endif
#ifdef __ARM_NEON
    float out_neon[N];
    bn_transformer_rmsnorm_neon(out_neon, x, w, N, 1e-6f);

    for (int i = 0; i < N; i++)
        assert(out_scalar[i] == out_neon[i]);
#endif
#endif

    printf("PASSED\n");
}

static void test_softmax(void) {
    printf("test_softmax... ");

    float x[] = {1.0f, 2.0f, 3.0f};
    softmax(x, 3);

    // Check probabilities sum to 1
    float sum = x[0] + x[1] + x[2];
    assert(fabsf(sum - 1.0f) < 1e-5f);

    // Check monotonicity
    assert(x[0] < x[1]);
    assert(x[1] < x[2]);

    // Check specific values
    // softmax([1,2,3]) = exp([1,2,3]) / sum(exp([1,2,3]))
    float e1 = expf(1), e2 = expf(2), e3 = expf(3);
    float esum = e1 + e2 + e3;
    assert(fabsf(x[0] - e1/esum) < 1e-5f);
    assert(fabsf(x[1] - e2/esum) < 1e-5f);
    assert(fabsf(x[2] - e3/esum) < 1e-5f);

    printf("PASSED\n");
}

static void test_runtime_softmax(void) {
    printf("test_runtime_softmax... ");

    float actual[] = {3.25f, -1.0f, 0.5f, 8.0f, -4.0f, 2.0f, 1.25f};
    float expected[sizeof(actual) / sizeof(actual[0])];
    float max_value = actual[0];
    for (size_t i = 1; i < sizeof(actual) / sizeof(actual[0]); i++)
        if (actual[i] > max_value) max_value = actual[i];

    double sum = 0.0;
    for (size_t i = 0; i < sizeof(actual) / sizeof(actual[0]); i++) {
        expected[i] = expf(actual[i] - max_value);
        sum += expected[i];
    }
    for (size_t i = 0; i < sizeof(actual) / sizeof(actual[0]); i++)
        expected[i] = (float)(expected[i] / sum);

    bn_transformer_softmax(actual,
                           (int)(sizeof(actual) / sizeof(actual[0])));
    float actual_sum = 0.0f;
    for (size_t i = 0; i < sizeof(actual) / sizeof(actual[0]); i++) {
        assert(fabsf(actual[i] - expected[i]) < 2e-6f);
        actual_sum += actual[i];
    }
    assert(fabsf(actual_sum - 1.0f) < 2e-6f);

#ifdef __ARM_NEON
    float scalar[] = {3.25f, -1.0f, 0.5f, 8.0f, -4.0f, 2.0f, 1.25f};
    float neon[sizeof(scalar) / sizeof(scalar[0])];
    memcpy(neon, scalar, sizeof(scalar));
    bn_transformer_softmax_scalar(
        scalar, (int)(sizeof(scalar) / sizeof(scalar[0])));
    bn_transformer_softmax_neon(
        neon, (int)(sizeof(neon) / sizeof(neon[0])));
    for (size_t i = 0; i < sizeof(scalar) / sizeof(scalar[0]); i++)
        assert(scalar[i] == neon[i]);
#endif

    printf("PASSED\n");
}

static void test_rope(void) {
    printf("test_rope... ");

    // Test that RoPE at pos=0 is identity
    float vec[] = {1.0f, 0.0f, 0.0f, 1.0f};
    rope(vec, 4, 4, 0, 10000.0f);

    // At pos=0, angle=0, cos=1, sin=0, so output should equal input
    assert(fabsf(vec[0] - 1.0f) < 1e-5f);
    assert(fabsf(vec[1] - 0.0f) < 1e-5f);
    assert(fabsf(vec[2] - 0.0f) < 1e-5f);
    assert(fabsf(vec[3] - 1.0f) < 1e-5f);

    // Test that RoPE preserves vector magnitude
    float vec2[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float mag_before = 0;
    for (int i = 0; i < 4; i++) mag_before += vec2[i] * vec2[i];

    rope(vec2, 4, 4, 5, 10000.0f);

    float mag_after = 0;
    for (int i = 0; i < 4; i++) mag_after += vec2[i] * vec2[i];

    assert(fabsf(mag_before - mag_after) < 1e-4f);

    printf("PASSED\n");
}

static void test_fp16_embed(void) {
    printf("test_fp16_embed... ");

    // Test FP16 → F32 conversion for embedding lookup
    uint16_t fp16_vals[] = {0x3C00, 0x4000, 0xBC00, 0x0000};  // 1.0, 2.0, -1.0, 0.0
    float f32_vals[4];
    for (int i = 0; i < 4; i++) {
        f32_vals[i] = bn_fp16_to_fp32(fp16_vals[i]);
    }

    assert(fabsf(f32_vals[0] - 1.0f) < 1e-6f);
    assert(fabsf(f32_vals[1] - 2.0f) < 1e-6f);
    assert(fabsf(f32_vals[2] - (-1.0f)) < 1e-6f);
    assert(fabsf(f32_vals[3] - 0.0f) < 1e-6f);

    printf("PASSED\n");
}

static void test_fast_silu(void) {
    printf("test_fast_silu... ");

#ifdef __ARM_NEON
    float vals[12] = {
        -8.0f, -4.0f, -1.5f, -0.25f,
         0.0f,  0.25f, 1.0f,  2.0f,
         4.0f,  8.0f, 12.0f, -12.0f
    };
    float out[12];
    for (int i = 0; i < 12; i += 4) {
        float32x4_t v = vld1q_f32(vals + i);
        vst1q_f32(out + i, bn_neon_fast_silu_f32(v));
    }
    for (int i = 0; i < 12; i++) {
        float exact = vals[i] / (1.0f + expf(-vals[i]));
        assert(fabsf(out[i] - exact) < 2e-3f);
    }
#elif defined(__AVX2__)
    float vals[8] = {-8.0f, -4.0f, -1.5f, -0.25f, 0.25f, 1.0f, 4.0f, 8.0f};
    float out[8];
    __m256 v = _mm256_loadu_ps(vals);
    _mm256_storeu_ps(out, bn_avx2_fast_silu_ps(v));
    for (int i = 0; i < 8; i++) {
        float exact = vals[i] / (1.0f + expf(-vals[i]));
        assert(fabsf(out[i] - exact) < 2e-3f);
    }
#endif

    printf("PASSED\n");
}

static void test_cpu_execution_helpers(void) {
    printf("test_cpu_execution_helpers... ");
#if defined(__AVX2__) && !defined(__AVX512F__)
    assert(bn_transformer_cpu_ssm_out_matvec_task_flags() ==
           BN_MATVEC_TASK_REFERENCE_DOT);
#else
    assert(bn_transformer_cpu_ssm_out_matvec_task_flags() == 0u);
#endif

    float x[8] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
    float r[8] = {0.5f, 0.5f, -1.0f, -1.0f, 2.0f, 2.0f, -3.0f, -3.0f};
    bn_transformer_cpu_residual_add(test_cpu_policy(), x, r, 8);
    float expected_residual[8] = {1.5f, -1.5f, 2.0f, -5.0f, 7.0f, -4.0f, 4.0f, -11.0f};
    for (int i = 0; i < 8; i++)
        assert(fabsf(x[i] - expected_residual[i]) < 1e-6f);

    float rope_buf[8] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
    float rc[2] = {0.0f, 1.0f};
    float rs[2] = {1.0f, 0.0f};
    bn_transformer_cpu_apply_rope_heads(
        test_cpu_policy(), rope_buf, 2, 4, 4, rc, rs);
    float expected_rope[8] = {-3.0f, 2.0f, 1.0f, 4.0f, 3.0f, -2.0f, -1.0f, -4.0f};
    for (int i = 0; i < 8; i++)
        assert(fabsf(rope_buf[i] - expected_rope[i]) < 1e-6f);

    float hb[8] = {-2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f};
    float hb2[8] = {1.0f, 0.5f, 2.0f, 3.0f, -1.0f, 1.5f, -0.5f, 2.0f};
    BnRunState s;
    memset(&s, 0, sizeof(s));
    s.hb = hb;
    s.hb2 = hb2;
    BnFFNPlan ffn;
    memset(&ffn, 0, sizeof(ffn));
    ffn.has_gate = 1;
    ffn.activation = 0;
    bn_transformer_cpu_apply_ffn_activation(
        test_cpu_policy(), &s, &ffn, 8, 0);
    for (int i = 0; i < 8; i++) {
        float g = (float[]){-2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f}[i];
        float u = (float[]){1.0f, 0.5f, 2.0f, 3.0f, -1.0f, 1.5f, -0.5f, 2.0f}[i];
        float expected = (g / (1.0f + expf(-g))) * u;
        assert(fabsf(hb[i] - expected) < 2e-3f);
    }

    float reference_hb[8] = {
        -2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f
    };
    s.hb = reference_hb;
    ffn.reference_activation = 1;
    bn_transformer_cpu_apply_ffn_activation(
        test_cpu_policy(), &s, &ffn, 8, 0);
#ifdef __AVX512F__
    float reference_expected[16];
    __m512 reference_g = _mm512_setr_ps(
        -2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    __m512 reference_den = _mm512_add_ps(
        _mm512_set1_ps(1.0f), bn_avx512_fast_exp_ps(
            _mm512_sub_ps(_mm512_setzero_ps(), reference_g)));
    __m512 reference_up = _mm512_setr_ps(
        1.0f, 0.5f, 2.0f, 3.0f, -1.0f, 1.5f, -0.5f, 2.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    _mm512_storeu_ps(reference_expected, _mm512_mul_ps(
        _mm512_div_ps(reference_g, reference_den), reference_up));
#elif defined(__AVX2__)
    float reference_expected[8];
    __m256 reference_g = _mm256_setr_ps(
        -2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f);
    __m256 reference_den = _mm256_add_ps(
        _mm256_set1_ps(1.0f), bn_avx2_fast_exp_avx512_ps(
            _mm256_sub_ps(_mm256_setzero_ps(), reference_g)));
    _mm256_storeu_ps(reference_expected, _mm256_mul_ps(
        _mm256_div_ps(reference_g, reference_den), _mm256_loadu_ps(hb2)));
#endif
    for (int i = 0; i < 8; i++) {
#if defined(__AVX512F__) || defined(__AVX2__)
        float expected = reference_expected[i];
#else
        float g = (float[]){
            -2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f
        }[i];
        float expected = (g / (1.0f + expf(-g))) * hb2[i];
#endif
        assert(reference_hb[i] == expected);
    }
    ffn.reference_activation = 0;

    float relu_hb[8] = {-2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f};
    s.hb = relu_hb;
    ffn.has_gate = 0;
    ffn.activation = 1;
    bn_transformer_cpu_apply_ffn_activation(
        test_cpu_policy(), &s, &ffn, 8, 0);
    float expected_relu2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0625f, 1.0f, 4.0f, 16.0f};
    for (int i = 0; i < 8; i++)
        assert(fabsf(relu_hb[i] - expected_relu2[i]) < 1e-6f);

    float unchanged[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    s.hb = unchanged;
    bn_transformer_cpu_apply_ffn_activation(
        test_cpu_policy(), &s, &ffn, 8, 1);
    for (int i = 0; i < 8; i++)
        assert(fabsf(unchanged[i] - (float)(i + 1)) < 1e-6f);

    BnConfig kv = {0};
    assert(bn_transformer_kv_host_float_cache_rows_available(&kv));
    assert(!bn_transformer_kv_host_cache_uses_fp16_rows(&kv));
    assert(!bn_transformer_kv_requires_gpu_cache_write_staging(&kv));
    assert(bn_transformer_kv_host_cache_element_size(&kv) == sizeof(float));
    assert(bn_transformer_kv_mode(&kv, 0) == BN_KV_FP32);
    assert(bn_transformer_kv_mode_stores_host_float_rows(BN_KV_FP32));
    assert(!bn_transformer_kv_mode_uses_turboquant(BN_KV_FP32));
    assert(!bn_transformer_kv_mode_uses_fp16(BN_KV_FP32));
    assert(bn_transformer_kv_mode_uses_cpu_gqa_cache(BN_KV_FP32));
    kv.kv_f16 = 1;
    assert(!bn_transformer_kv_host_float_cache_rows_available(&kv));
    assert(bn_transformer_kv_host_cache_uses_fp16_rows(&kv));
    assert(bn_transformer_kv_requires_gpu_cache_write_staging(&kv));
    assert(bn_transformer_kv_host_cache_element_size(&kv) == sizeof(uint16_t));
    assert(bn_transformer_kv_mode(&kv, 0) == BN_KV_FP16);
    assert(!bn_transformer_kv_mode_stores_host_float_rows(BN_KV_FP16));
    assert(!bn_transformer_kv_mode_uses_turboquant(BN_KV_FP16));
    assert(bn_transformer_kv_mode_uses_fp16(BN_KV_FP16));
    assert(bn_transformer_kv_mode_uses_cpu_gqa_cache(BN_KV_FP16));
    kv.kv_f16 = 0;
    kv.kv_tq_bits = 3;
    assert(!bn_transformer_kv_host_float_cache_rows_available(&kv));
    assert(bn_transformer_kv_mode(&kv, 1) == BN_KV_TQ);
    assert(!bn_transformer_kv_mode_stores_host_float_rows(BN_KV_TQ));
    assert(bn_transformer_kv_mode_uses_turboquant(BN_KV_TQ));
    assert(!bn_transformer_kv_mode_uses_fp16(BN_KV_TQ));
    assert(!bn_transformer_kv_mode_uses_cpu_gqa_cache(BN_KV_TQ));
    assert(!bn_transformer_kv_host_float_cache_rows_available(NULL));

    float key_cache[8] = {0};
    float value_cache[8] = {0};
    float k_src[2] = {3.0f, 4.0f};
    float v_src[2] = {5.0f, 6.0f};
    s.key_cache = key_cache;
    s.value_cache = value_cache;
    assert(bn_transformer_write_host_kv_cache_row(
               &s, BN_KV_FP32, 2, 1, 2, k_src, v_src, 2) == 0);
    assert(fabsf(key_cache[4] - 3.0f) < 1e-6f);
    assert(fabsf(key_cache[5] - 4.0f) < 1e-6f);
    assert(fabsf(value_cache[4] - 5.0f) < 1e-6f);
    assert(fabsf(value_cache[5] - 6.0f) < 1e-6f);

    uint16_t key_cache16[2] = {0};
    uint16_t value_cache16[2] = {0};
    s.key_cache = (float *)key_cache16;
    s.value_cache = (float *)value_cache16;
    assert(bn_transformer_write_host_kv_cache_row(
               &s, BN_KV_FP16, 0, 0, 2, k_src, v_src, 2) == 0);
    assert(key_cache16[0] != 0);
    assert(value_cache16[0] != 0);
    assert(bn_transformer_write_host_kv_cache_row(
               &s, BN_KV_TQ, 0, 0, 2, k_src, v_src, 2) != 0);

    printf("PASSED\n");
}

static void test_gpu_capability_routing(void) {
    printf("test_gpu_capability_routing... ");

    BnGPUBackend gpu;
    memset(&gpu, 0, sizeof(gpu));

    assert(!bn_transformer_gpu_has_cap(NULL, BN_GPU_CAP_FLASH_ATTN));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q8_0));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q5_0));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_transformer_gpu_can_native_quant_qkv(BN_GGUF_TENSOR_Q4_0,
                                              BN_GGUF_TENSOR_Q4_0,
                                              BN_GGUF_TENSOR_TQ1_0));

    gpu.caps = BN_GPU_CAP_LOWBIT_BLOCK32_MATVEC_SPLIT |
               BN_GPU_CAP_MIDBIT_BLOCK32_MATVEC_SPLIT |
               BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT |
               BN_GPU_CAP_NATIVE_QUANT_MATVEC_SPLIT |
               BN_GPU_CAP_DEINTERLEAVED_KQUANT_MATVEC_SPLIT |
               BN_GPU_CAP_LOWBIT_BLOCK32_FUSED_GATEUP_SILU |
               BN_GPU_CAP_MIDBIT_BLOCK32_FUSED_GATEUP_SILU |
               BN_GPU_CAP_FLASH_ATTN;
    gpu.kind = BN_GPU_BACKEND_METAL;

    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q4_0));
    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q8_0));
    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q5_0));
    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q5_K));
    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_F16));
    assert(bn_transformer_gpu_can_native_quant_qkv(BN_GGUF_TENSOR_Q4_0,
                                             BN_GGUF_TENSOR_Q4_0,
                                             BN_GGUF_TENSOR_Q4_0));
    assert(bn_transformer_gpu_can_stack_same_quant_format_qk(BN_GGUF_TENSOR_Q4_K,
                                                 BN_GGUF_TENSOR_Q4_K));
    assert(!bn_transformer_gpu_can_stack_same_quant_format_qk(BN_GGUF_TENSOR_Q4_K,
                                                  BN_GGUF_TENSOR_Q5_K));
    BnQWeight q = {0};
    BnQWeight k = {0};
    q.type = BN_GGUF_TENSOR_Q4_K;
    k.type = BN_GGUF_TENSOR_Q4_K;
    q.rows = 64;
    k.rows = 32;
    q.cols = k.cols = 128;
    assert(bn_transformer_gpu_can_stack_same_quant_format_qk_weights(&q, &k, 64, 32));
    k.cols = 64;
    assert(!bn_transformer_gpu_can_stack_same_quant_format_qk_weights(&q, &k, 64, 32));
    k.cols = 128;
    k.rows = 64;
    assert(!bn_transformer_gpu_can_stack_same_quant_format_qk_weights(&q, &k, 64, 32));
    k.rows = 32;
    k.type = BN_GGUF_TENSOR_Q5_K;
    assert(!bn_transformer_gpu_can_stack_same_quant_format_qk_weights(&q, &k, 64, 32));
    BnQWeight gate = {0};
    BnQWeight up = {0};
    gate.type = BN_GGUF_TENSOR_Q4_0;
    up.type = BN_GGUF_TENSOR_Q4_0;
    gate.rows = up.rows = 64;
    gate.cols = up.cols = 128;
    gate.data = &gate;
    up.data = &up;
    assert(bn_transformer_gpu_can_stack_same_quant_format_gateup(&gate, &up));
    up.cols = 64;
    assert(!bn_transformer_gpu_can_stack_same_quant_format_gateup(&gate, &up));
    up.cols = 128;
    up.type = BN_GGUF_TENSOR_Q4_K;
    assert(!bn_transformer_gpu_can_stack_same_quant_format_gateup(&gate, &up));
    gate.type = BN_GGUF_TENSOR_Q4_K;
    assert(bn_transformer_gpu_can_stack_same_quant_format_gateup(&gate, &up));

    assert(bn_transformer_gpu_can_fused_gateup_silu(&gpu, BN_GGUF_TENSOR_Q4_0, 0));
    assert(!bn_transformer_gpu_can_fused_gateup_silu(&gpu, BN_GGUF_TENSOR_Q4_0, 1));
    assert(bn_transformer_gpu_can_fused_gateup_silu(&gpu, BN_GGUF_TENSOR_Q5_0, 0));
    assert(!bn_transformer_gpu_can_fused_gateup_silu(&gpu, BN_GGUF_TENSOR_Q8_0, 0));
    assert(bn_transformer_gpu_can_fused_gateup_silu_pair(
        &gpu, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0, 0));
    assert(!bn_transformer_gpu_can_fused_gateup_silu_pair(
        &gpu, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q5_0, 0));
    assert(bn_transformer_gpu_can_gateup_split_activation(
        &gpu, BN_GGUF_TENSOR_Q4_0, 0));
    assert(!bn_transformer_gpu_can_gateup_split_activation(
        &gpu, BN_GGUF_TENSOR_F16, 0));
    assert(bn_transformer_gpu_matvec_kquant_dot_flags(
               BN_GGUF_TENSOR_Q4_K, 1) ==
           BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT);
    assert(bn_transformer_gpu_matvec_kquant_dot_flags(
               BN_GGUF_TENSOR_Q4_K, 0) == 0);
    assert(bn_transformer_gpu_matvec_kquant_dot_flags(
               BN_GGUF_TENSOR_Q8_0, 1) == 0);
    assert(bn_transformer_gpu_matvec_quant_dot_flags(
               BN_GGUF_TENSOR_Q4_0, 1) ==
           BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT);
    assert(bn_transformer_gpu_matvec_quant_dot_flags(
               BN_GGUF_TENSOR_Q4_K, 1) ==
           BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT);
    assert(bn_transformer_gpu_matvec_quant_dot_flags(
               BN_GGUF_TENSOR_F32, 1) == 0);
    assert(bn_transformer_gpu_moe_route_raw_compare_matvec_flags(
               BN_GGUF_TENSOR_Q4_K) ==
           BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT);
    assert(bn_transformer_gpu_moe_route_raw_compare_matvec_flags(
               BN_GGUF_TENSOR_F32) == 0);
    BnMoEExpertMap routed_projection_flags;
    memset(&routed_projection_flags, 0, sizeof(routed_projection_flags));
    routed_projection_flags.gate_type = BN_GGUF_TENSOR_Q4_0;
    routed_projection_flags.up_type = BN_GGUF_TENSOR_Q8_0;
    routed_projection_flags.down_type = BN_GGUF_TENSOR_Q4_0;
    assert(bn_transformer_gpu_moe_expert_projection_matvec_flags(
               &routed_projection_flags, 0, 1) ==
           BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT);
    assert(bn_transformer_gpu_moe_expert_projection_matvec_flags(
               &routed_projection_flags, 1, 1) == 0);
    assert(bn_transformer_gpu_moe_expert_projection_matvec_flags(
               &routed_projection_flags, 2, 1) ==
           BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT);
    assert(bn_transformer_gpu_moe_expert_projection_matvec_flags(
               &routed_projection_flags, 0, 0) == 0);
    assert(bn_transformer_gpu_moe_expert_projection_matvec_flags(
               &routed_projection_flags, 3, 1) == 0);
    assert(bn_transformer_gpu_moe_expert_projection_matvec_flags(
               NULL, 0, 1) == 0);
    assert(bn_transformer_gpu_moe_shared_down_matvec_flags(
               &gpu, BN_GGUF_TENSOR_Q5_K, 0, 1, 1) ==
           BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_moe_shared_down_matvec_flags(
               &gpu, BN_GGUF_TENSOR_Q5_K, 0, 1, 1) == 0);
    assert(bn_transformer_gpu_moe_shared_down_matvec_flags(
               &gpu, BN_GGUF_TENSOR_Q8_0, 1, 1, 1) ==
           BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT);
    assert(bn_transformer_gpu_moe_shared_down_matvec_flags(
               &gpu, BN_GGUF_TENSOR_Q8_0, 1, 1, 0) ==
           (BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT |
            BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION));
    assert(bn_transformer_gpu_moe_dense_residual_down_matvec_flags(
               &gpu, BN_GGUF_TENSOR_Q8_0, 1, 1) ==
           BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_moe_dense_residual_down_matvec_flags(
               &gpu, BN_GGUF_TENSOR_Q8_0, 1, 1) ==
           (BN_GPU_OP_FLAG_MATVEC_KQUANT_DOT |
            BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_moe_shared_down_matvec_flags(
               &gpu, BN_GGUF_TENSOR_Q4_K, 0, 1, 1) ==
           BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION);
    assert(bn_transformer_gpu_moe_shared_down_matvec_flags(
               &gpu, BN_GGUF_TENSOR_Q5_K, 1, 1, 1) ==
           BN_GPU_OP_FLAG_MATVEC_REFERENCE_KQUANT);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_matvec_reference_kquant_flags(
               BN_GGUF_TENSOR_Q6_K, 1) ==
           BN_GPU_OP_FLAG_MATVEC_REFERENCE_KQUANT);
    assert(bn_transformer_gpu_matvec_reference_kquant_flags(
               BN_GGUF_TENSOR_Q6_K, 0) == 0);
    assert(bn_transformer_gpu_matvec_reference_kquant_flags(
               BN_GGUF_TENSOR_Q4_K, 1) ==
           BN_GPU_OP_FLAG_MATVEC_REFERENCE_KQUANT);
    assert(bn_transformer_gpu_float_buffer_type() == BN_GGUF_TENSOR_F32);
    assert(bn_transformer_gpu_reference_silu_flags(
               BN_GGUF_TENSOR_Q8_0, 1) == BN_GPU_OP_FLAG_REFERENCE_SILU);
    assert(bn_transformer_gpu_reference_silu_flags(
               BN_GGUF_TENSOR_Q8_0, 0) == 0);
    assert(bn_transformer_gpu_reference_silu_flags(
               BN_GGUF_TENSOR_Q4_0, 1) == 0);
    assert(bn_transformer_gpu_reference_silu_active_flags(1) ==
           BN_GPU_OP_FLAG_REFERENCE_SILU);
    assert(bn_transformer_gpu_reference_silu_active_flags(0) == 0);
    assert(bn_transformer_gpu_reference_silu_active_flags(-1) == 0);
    assert(bn_transformer_gpu_reference_activation_flags(1) ==
           BN_GPU_OP_FLAG_REFERENCE_ACTIVATION);
    assert(bn_transformer_gpu_reference_activation_flags(0) == 0);
    assert(bn_transformer_gpu_reference_block_accumulation_flags(1) ==
           BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION);
    assert(bn_transformer_gpu_reference_block_accumulation_flags(0) == 0);
    assert(bn_transformer_gpu_prefers_gateup_split(BN_GGUF_TENSOR_Q8_0));
    assert(!bn_transformer_gpu_prefers_gateup_split(BN_GGUF_TENSOR_Q4_0));
    assert(bn_transformer_gpu_same_quant_format_pair_stackable(
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_transformer_gpu_same_quant_format_pair_stackable(
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K));
    assert(bn_transformer_gpu_shared_kquant_gateup_dot_eligible(
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K, 256));
    assert(!bn_transformer_gpu_shared_kquant_gateup_dot_eligible(
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K, 128));
    assert(!bn_transformer_gpu_shared_kquant_gateup_dot_eligible(
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, 256));

    BnLayerWeights shared_gateup_lw;
    memset(&shared_gateup_lw, 0, sizeof(shared_gateup_lw));
    BnTransformerGPUMoESharedResources shared_gateup_resources;
    memset(&shared_gateup_resources, 0, sizeof(shared_gateup_resources));
    shared_gateup_lw.shared.shared_gate.type = BN_GGUF_TENSOR_Q4_0;
    shared_gateup_lw.shared.shared_up.type = BN_GGUF_TENSOR_Q4_0;
    shared_gateup_lw.shared.shared_gate.rows = 64;
    shared_gateup_lw.shared.shared_up.rows = 64;
    shared_gateup_lw.shared.shared_gate.cols = 128;
    shared_gateup_lw.shared.shared_up.cols = 128;
    shared_gateup_lw.shared.shared_down.type = BN_GGUF_TENSOR_Q4_0;
    shared_gateup_lw.shared.shared_down.rows = 128;
    shared_gateup_lw.shared.shared_down.cols = 64;
    shared_gateup_lw.shared.shared_gate.data = (void *)1;
    BnTransformerGPUMoESharedProjectionInfo shared_info;
    assert(bn_transformer_gpu_resolve_moe_shared_projection_info(
        &shared_info, &shared_gateup_lw));
    assert(shared_info.gate_type == BN_GGUF_TENSOR_Q4_0);
    assert(shared_info.up_type == BN_GGUF_TENSOR_Q4_0);
    assert(shared_info.down_type == BN_GGUF_TENSOR_Q4_0);
    assert(shared_info.gate_rows == 64);
    assert(shared_info.up_rows == 64);
    assert(shared_info.down_rows == 128);
    assert(shared_info.gate_cols == 128);
    assert(shared_info.up_cols == 128);
    assert(shared_info.down_cols == 64);
    shared_gateup_resources.gpu = &gpu;
    shared_gateup_resources.shared_gate = (void *)1;
    shared_gateup_resources.shared_gateup_stacked = (void *)1;
    BnTransformerGPUSharedExpertGateupPolicy shared_gateup =
        bn_transformer_gpu_shared_expert_gateup_policy(
            &gpu, &shared_gateup_lw, &shared_gateup_resources);
    assert(!shared_gateup.use_kquant_dot);
    assert(shared_gateup.use_fused_gateup);
    assert(!shared_gateup.use_gateup_split);

    shared_gateup_lw.shared.shared_gate.type = BN_GGUF_TENSOR_Q8_0;
    shared_gateup_lw.shared.shared_up.type = BN_GGUF_TENSOR_Q8_0;
    shared_gateup = bn_transformer_gpu_shared_expert_gateup_policy(
        &gpu, &shared_gateup_lw, &shared_gateup_resources);
    assert(!shared_gateup.use_kquant_dot);
    assert(!shared_gateup.use_fused_gateup);
    assert(shared_gateup.use_gateup_split);

    shared_gateup_lw.shared.shared_gate.type = BN_GGUF_TENSOR_Q4_K;
    shared_gateup_lw.shared.shared_up.type = BN_GGUF_TENSOR_Q4_K;
    shared_gateup_lw.shared.shared_gate.cols = 256;
    shared_gateup_lw.shared.shared_up.cols = 256;
    shared_gateup = bn_transformer_gpu_shared_expert_gateup_policy(
        &gpu, &shared_gateup_lw, &shared_gateup_resources);
    assert(shared_gateup.use_kquant_dot);

    setenv("BN_GPU_DISABLE_FUSED_GATEUP", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_fused_gateup_silu_policy_allows(
        &gpu, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_transformer_gpu_can_fused_gateup_silu(
        &gpu, BN_GGUF_TENSOR_Q4_0, 0));
    unsetenv("BN_GPU_DISABLE_FUSED_GATEUP");
    test_gpu_runtime_refresh(&gpu);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    gpu.caps |= BN_GPU_CAP_DEINTERLEAVED_KQUANT_FUSED_GATEUP_SILU;
    unsetenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP");
    assert(!bn_transformer_gpu_fused_gateup_silu_policy_allows(
        &gpu, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_transformer_gpu_can_fused_gateup_silu(
        &gpu, BN_GGUF_TENSOR_Q5_K, 0));
    setenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_fused_gateup_silu_policy_allows(
        &gpu, BN_GGUF_TENSOR_Q5_K));
    assert(bn_transformer_gpu_can_fused_gateup_silu(
        &gpu, BN_GGUF_TENSOR_Q5_K, 0));
    unsetenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP");
    test_gpu_runtime_refresh(&gpu);

    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_DISABLE_GATEUP");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_small_dense_native_quant_fused_gateup_enabled(&gpu, 0));
    assert(bn_transformer_gpu_small_dense_native_quant_fused_gateup_enabled(&gpu, 1));
    setenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_DISABLE_GATEUP", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_small_dense_native_quant_fused_gateup_enabled(&gpu, 1));
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_DISABLE_GATEUP");

    unsetenv("BN_GPU_DISABLE_GATEUP_SPLIT");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_gateup_split_enabled(&gpu));
    setenv("BN_GPU_DISABLE_GATEUP_SPLIT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_gateup_split_enabled(&gpu));
    unsetenv("BN_GPU_DISABLE_GATEUP_SPLIT");

    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_DISABLE_FFN_DOWN");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_small_dense_native_quant_down_enabled(&gpu, 0));
    assert(bn_transformer_gpu_small_dense_native_quant_down_enabled(&gpu, 1));
    setenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_DISABLE_FFN_DOWN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_small_dense_native_quant_down_enabled(&gpu, 1));
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_DISABLE_FFN_DOWN");

    unsetenv("BN_GPU_DISABLE_QKV_SPLIT");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_qkv_split_enabled(&gpu, 0));
    assert(bn_transformer_gpu_qkv_split_enabled(&gpu, 1));
    assert(bn_transformer_gpu_qk_split_enabled(&gpu));
    setenv("BN_GPU_DISABLE_QKV_SPLIT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_qkv_split_enabled(&gpu, 0));
    assert(!bn_transformer_gpu_qk_split_enabled(&gpu));
    unsetenv("BN_GPU_DISABLE_QKV_SPLIT");

    unsetenv("BN_GPU_DEBUG_QKV_SPLIT");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_qkv_split_debug_enabled(&gpu));
    setenv("BN_GPU_DEBUG_QKV_SPLIT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_qkv_split_debug_enabled(&gpu));
    unsetenv("BN_GPU_DEBUG_QKV_SPLIT");

    unsetenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_ssm_qkvz_split_enabled(&gpu));
    setenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_ssm_qkvz_split_enabled(&gpu));
    unsetenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT");

    unsetenv("BN_GPU_DISABLE_SSM_AB_STACK");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_ssm_ab_stack_enabled(&gpu));
    setenv("BN_GPU_DISABLE_SSM_AB_STACK", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_ssm_ab_stack_enabled(&gpu));
    unsetenv("BN_GPU_DISABLE_SSM_AB_STACK");

    unsetenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_split_residual_rmsnorm_enabled(&gpu));
    setenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_split_residual_rmsnorm_enabled(&gpu));
    unsetenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM");

    unsetenv("BN_GPU_DEBUG_FALLBACK");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_debug_fallback_enabled(&gpu));
    setenv("BN_GPU_DEBUG_FALLBACK", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_debug_fallback_enabled(&gpu));
    unsetenv("BN_GPU_DEBUG_FALLBACK");

    unsetenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_shared_kquant_dot_enabled(&gpu, 0));
    assert(bn_transformer_gpu_shared_kquant_dot_enabled(&gpu, 1));
    setenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_shared_kquant_dot_enabled(&gpu, 1));
    unsetenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT");

    unsetenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE");
    assert(!bn_transformer_gpu_shared_expert_gate_enabled(&gpu, 0));
    assert(bn_transformer_gpu_shared_expert_gate_enabled(&gpu, 1));
    setenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_shared_expert_gate_enabled(&gpu, 1));
    unsetenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE");

    BnLayerWeights shared_path_lw;
    memset(&shared_path_lw, 0, sizeof(shared_path_lw));
    BnTransformerGPUMoESharedResources shared_resources;
    memset(&shared_resources, 0, sizeof(shared_resources));
    shared_path_lw.shared.shared_gate.data = (void *)1;
    assert(!bn_transformer_gpu_shared_expert_path_available(
        &shared_path_lw, &shared_resources));
    shared_resources.shared_gate = (void *)1;
    assert(bn_transformer_gpu_shared_expert_path_available(
        &shared_path_lw, &shared_resources));
    assert(!bn_transformer_gpu_shared_expert_gate_available(
        &shared_path_lw, &shared_resources));
    shared_path_lw.shared.shared_expert_gate = (float *)1;
    BnTransformerGPUSharedExpertGatePolicy shared_gate_policy =
        bn_transformer_gpu_shared_expert_gate_policy(&shared_path_lw);
    assert(shared_gate_policy.has_gate_vector);
    assert(!bn_transformer_gpu_shared_expert_gate_available(
        &shared_path_lw, &shared_resources));
    shared_resources.shared_expert_gate = (void *)1;
    assert(bn_transformer_gpu_shared_expert_gate_available(
        &shared_path_lw, &shared_resources));
    assert(!bn_transformer_gpu_shared_expert_path_available(
        NULL, &shared_resources));
    assert(!bn_transformer_gpu_shared_expert_path_available(
        &shared_path_lw, NULL));
    assert(!bn_transformer_gpu_shared_expert_gate_available(
        NULL, &shared_resources));
    assert(!bn_transformer_gpu_shared_expert_gate_available(
        &shared_path_lw, NULL));
    shared_path_lw.shared.shared_expert_gate = NULL;
    shared_gate_policy =
        bn_transformer_gpu_shared_expert_gate_policy(&shared_path_lw);
    assert(!shared_gate_policy.has_gate_vector);

    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_can_flash_attn(&gpu));
    assert(!bn_transformer_gpu_can_layerwise_rope(&gpu));
    gpu.caps |= BN_GPU_CAP_LAYERWISE_ROPE;
    assert(bn_transformer_gpu_can_layerwise_rope(&gpu));

    printf("PASSED\n");
}

static int test_prefill_prefix_support(void *ctx,
                                     const BnGPUAttentionPrefillPlan *plan) {
    assert(plan);
    return *(const int *)ctx;
}

static void test_prefill_microbatch_policy(void) {
    BnGPUBackend gpu = {0};
    BnConfig c = {0};
    BnGPUAttentionPrefillPlan span = {0};
    int supported = 1;
    assert(!bn_gpu_backend_can_check_prefill_prefix(NULL));
    assert(!bn_gpu_backend_prefill_prefix_supported(&gpu, &span));
    gpu.ctx = &supported;
    gpu.prefill_attention_prefix_supported = test_prefill_prefix_support;
    assert(bn_gpu_backend_can_check_prefill_prefix(&gpu));
    assert(bn_gpu_backend_prefill_prefix_supported(&gpu, &span));
    assert(!bn_gpu_backend_prefill_prefix_supported(&gpu, NULL));
    supported = 0;
    assert(!bn_gpu_backend_prefill_prefix_supported(&gpu, &span));
    supported = -1;
    assert(!bn_gpu_backend_prefill_prefix_supported(&gpu, &span));
    c.seq_len=2048; c.n_layers=48; c.full_attn_interval=4;
    c.ssm_state_size=128; c.n_experts=256; c.n_experts_active=8; c.kv_f16=1;
    assert(bn_transformer_prefill_microbatch_tokens(NULL)==0);
    assert(bn_transformer_prefill_microbatch_tokens(&gpu)==0);
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,1058,0));
    gpu.caps=BN_GPU_CAP_PREFILL_PREFIX_KV|BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL;
    assert(bn_transformer_prefill_microbatch_tokens(&gpu)==512);
    assert(bn_backend_runtime_policy_set(&gpu.runtime_policy,"BN_GPU_PREFILL_FULL_PROMPT","1",1)==0);
    assert(bn_transformer_prefill_microbatch_tokens(&gpu)==0);
    /* Backend boolean policies are presence-based, including the value "0". */
    assert(bn_backend_runtime_policy_set(&gpu.runtime_policy,"BN_GPU_PREFILL_FULL_PROMPT","0",1)==0);
    assert(bn_transformer_prefill_microbatch_tokens(&gpu)==0);
    bn_backend_runtime_policy_free(&gpu.runtime_policy);
    assert(bn_transformer_prefill_microbatch_tokens(&gpu)==512);
    assert(bn_transformer_prefill_prefix_request_allowed(&c,&gpu,512,0));
    assert(bn_transformer_prefill_prefix_request_allowed(&c,&gpu,512,512));
    assert(bn_transformer_prefill_prefix_request_allowed(&c,&gpu,1,512));
    assert(bn_transformer_prefill_prefix_request_allowed(&c,&gpu,34,1024));
    assert(bn_transformer_prefill_prefix_request_allowed(&c,&gpu,512,1536));
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,513,1536));
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,512,INT_MAX));
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,0,0));
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,2,1));
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,512,-1));
    c.seq_len=1024;
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,34,1024));
    c.seq_len=2048; c.kv_f16=0;
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,512,0));
    c.kv_f16=1; c.n_experts=0;
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,512,0));
    c.n_experts=256; c.kv_tq_bits=3;
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,512,0));
    c.kv_tq_bits=0; c.hyper_connection_count=4;
    assert(!bn_transformer_prefill_prefix_request_allowed(&c,&gpu,512,0));
    BnTransformerPrefillHybridModelChainPolicy chain =
        bn_transformer_prefill_hybrid_model_chain_policy(1,1,512,48,0,1);
    assert(chain.enabled);
    chain=bn_transformer_prefill_hybrid_model_chain_policy(1,1,512,48,0,0);
    assert(!chain.enabled);
    chain=bn_transformer_prefill_hybrid_model_chain_policy(1,1,-1,48,0,1);
    assert(!chain.enabled);
    chain=bn_transformer_prefill_hybrid_model_chain_policy(1,1,512,48,1,1);
    assert(!chain.enabled);
    printf("prefill microbatch policy PASSED\n");
}

static void test_gpu_policy_helpers(void) {
    printf("test_gpu_policy_helpers... ");

    BnConfig c;
    memset(&c, 0, sizeof(c));
    c.n_layers = 4;
    BnConfig reference_qkv = {0};
    BnGPUBackend reference_qkv_gpu = {0};
    reference_qkv.n_layers = 30;
    reference_qkv.n_experts = 128;
    assert(bn_transformer_gpu_reference_qkv_layer_enabled(
        &reference_qkv_gpu, 29, &reference_qkv));
    assert(!bn_transformer_gpu_reference_qkv_layer_enabled(
        &reference_qkv_gpu, 30, &reference_qkv));
    assert(bn_transformer_gpu_graph_op_capacity(&c) >
           80 * c.n_layers);
    assert(bn_transformer_gpu_uses_small_dense_shape(&c));
    assert(!bn_transformer_gpu_uses_large_dense_shape(&c));
    assert(!bn_transformer_gpu_uses_per_layer_embedding(&c));
    assert(bn_transformer_gpu_uses_dense_attention_only(&c));
    assert(!bn_transformer_gpu_uses_hybrid_ssm(&c));
    assert(!bn_transformer_gpu_uses_moe(&c));
    BnGPUBackend cuda_reference = {
        .kind = BN_GPU_BACKEND_CUDA,
        .caps = BN_GPU_CAP_REFERENCE_ATTENTION,
    };
    c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION;
    assert(bn_transformer_gpu_reference_dense_ffn_decode_accumulation_enabled(
        &cuda_reference, &c));
    c.full_attn_interval = 4;
    c.ssm_inner_size = 64;
    assert(!bn_transformer_gpu_reference_dense_ffn_decode_accumulation_enabled(
        &cuda_reference, &c));
    c.policy_flags |= BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
    assert(bn_transformer_gpu_reference_dense_ffn_decode_accumulation_enabled(
        &cuda_reference, &c));
    c.policy_flags = 0;
    c.full_attn_interval = 0;
    c.ssm_inner_size = 0;
    BnLayerWeights gpu_dense_lw = {0};
    BnLayerWeights gpu_moe_lw = {0};
    gpu_moe_lw.moe.router_weight = (float *)1;
    BnTransformerGPULayerKindPolicy gpu_layer_kind =
        bn_transformer_gpu_layer_kind_policy(NULL);
    assert(!gpu_layer_kind.uses_moe);
    gpu_layer_kind = bn_transformer_gpu_layer_kind_policy(&gpu_dense_lw);
    assert(!gpu_layer_kind.uses_moe);
    gpu_layer_kind = bn_transformer_gpu_layer_kind_policy(&gpu_moe_lw);
    assert(gpu_layer_kind.uses_moe);
    BnTransformerGPULayerResources projection_resources = {0};
    assert(!bn_transformer_gpu_layer_projection_resources_available(
        NULL, &projection_resources));
    assert(!bn_transformer_gpu_layer_projection_resources_available(
        &gpu_dense_lw, NULL));
    assert(bn_transformer_gpu_layer_projection_resources_available(
        &gpu_dense_lw, &projection_resources));
    gpu_dense_lw.attn.wq.data = (void *)1;
    gpu_dense_lw.attn.has_kv = 1;
    assert(!bn_transformer_gpu_layer_projection_resources_available(
        &gpu_dense_lw, &projection_resources));
    projection_resources.qkv.wq = (void *)1;
    assert(!bn_transformer_gpu_layer_projection_resources_available(
        &gpu_dense_lw, &projection_resources));
    projection_resources.qkv.wk = (void *)2;
    assert(!bn_transformer_gpu_layer_projection_resources_available(
        &gpu_dense_lw, &projection_resources));
    projection_resources.qkv.wv = (void *)3;
    assert(bn_transformer_gpu_layer_projection_resources_available(
        &gpu_dense_lw, &projection_resources));
    gpu_dense_lw.attn.has_kv = 0;
    projection_resources.qkv.wk = NULL;
    projection_resources.qkv.wv = NULL;
    assert(bn_transformer_gpu_layer_projection_resources_available(
        &gpu_dense_lw, &projection_resources));
    gpu_dense_lw.attn.wo.data = (void *)1;
    assert(!bn_transformer_gpu_layer_projection_resources_available(
        &gpu_dense_lw, &projection_resources));
    projection_resources.attention.wo = (void *)4;
    gpu_dense_lw.ffn.ffn_up.data = (void *)1;
    gpu_dense_lw.ffn.ffn_down.data = (void *)1;
    assert(!bn_transformer_gpu_layer_projection_resources_available(
        &gpu_dense_lw, &projection_resources));
    projection_resources.dense_ffn.ffn_up = (void *)5;
    projection_resources.dense_ffn.ffn_down = (void *)6;
    assert(bn_transformer_gpu_layer_projection_resources_available(
        &gpu_dense_lw, &projection_resources));
    assert(bn_transformer_gpu_layer_projection_resources_available(
        &gpu_moe_lw, &projection_resources));
    BnGPUMoEResources resolved_all_active_two;
    BnGPUMoEResolvedExpert all_active_two_storage[2];
    BnGPUMoETemporaryBuffers all_active_two_temporaries = {0};
    BnModel all_active_two_model = {0};
    BnSession all_active_two_session = {0};
    assert(bn_transformer_gpu_resolve_all_active_two_moe_resources(
               NULL, all_active_two_storage, &all_active_two_model,
               &all_active_two_session, &gpu_moe_lw, 0, (void *)1,
               &all_active_two_temporaries) == -1);
    assert(bn_transformer_gpu_resolve_all_active_two_moe_resources(
               &resolved_all_active_two, all_active_two_storage,
               &all_active_two_model, &all_active_two_session, &gpu_moe_lw,
               0, (void *)1, &all_active_two_temporaries) == -1);
    assert(bn_transformer_gpu_resolve_routed_moe_resources(
               NULL, all_active_two_storage, &all_active_two_model,
               &all_active_two_session, &gpu_moe_lw, 0,
               &all_active_two_temporaries) == -1);
    assert(bn_transformer_gpu_resolve_profiled_routed_moe_resources(
               NULL, all_active_two_storage, &all_active_two_model,
               &all_active_two_session, &gpu_moe_lw, 0,
               &all_active_two_temporaries, 0, 2, 2,
               0.0, 0.0, 0.0) == -1);
    bn_transformer_gpu_release_moe_temporaries(
        &all_active_two_model, &all_active_two_temporaries);
    float debug_norm_in[2] = {3.0f, 4.0f};
    float debug_norm_weight[2] = {1.0f, 1.0f};
    float debug_norm_out[2] = {0.0f, 0.0f};
    float debug_norm_expected[2] = {0.0f, 0.0f};
    bn_transformer_gpu_debug_rmsnorm(
        debug_norm_out, debug_norm_in, debug_norm_weight, 2, 1e-6f);
    bn_transformer_rmsnorm_scalar(
        debug_norm_expected, debug_norm_in, debug_norm_weight, 2, 1e-6f);
    assert(memcmp(debug_norm_out, debug_norm_expected,
                  sizeof(debug_norm_out)) == 0);
    bn_transformer_gpu_debug_compare_vec(NULL, 0, 0, NULL, NULL, 0);
    bn_transformer_gpu_moe_route_profile_add(
        NULL, 2, 2, 1.0, 2.0, 3.0, 4.0);
    bn_transformer_gpu_debug_compare_argmax(NULL, 0, NULL, 0, 1.0f, 0);
    bn_transformer_gpu_debug_compare_logits(
        NULL, NULL, NULL, NULL, 0, 0);
    c.policy_flags = BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT;
    assert(bn_transformer_gpu_uses_per_layer_embedding(&c));
    c.policy_flags = 0;
    c.dim = 4096;
    c.full_attn_interval = 4;
    assert(!bn_transformer_gpu_uses_small_dense_shape(&c));
    assert(bn_transformer_gpu_uses_large_dense_shape(&c));
    assert(!bn_transformer_gpu_uses_dense_attention_only(&c));
    c.ssm_inner_size = 64;
    assert(bn_transformer_gpu_uses_hybrid_ssm(&c));
    assert(bn_transformer_gpu_uses_large_dense_hybrid_ssm(&c));
    c.n_experts = 2;
    assert(bn_transformer_gpu_uses_moe(&c));
    assert(!bn_transformer_gpu_uses_non_hybrid_moe(&c));
    c.full_attn_interval = 0;
    assert(bn_transformer_gpu_uses_non_hybrid_moe(&c));
    assert(!bn_transformer_gpu_uses_large_dense_hybrid_ssm(&c));
    c.ssm_inner_size = 0;
    c.n_experts = 0;
    c.dim = 2048;
    assert(bn_transformer_gpu_uses_small_dense_native_quant_shape(&c));
    assert(!bn_transformer_gpu_requires_float_kquant(&c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK;
    assert(bn_transformer_gpu_requires_float_kquant(&c));
    c.policy_flags = 0;
    assert(bn_transformer_gpu_dense_batch_prefill_shape_allowed_for_backend(
        &c, 0));
    c.dim = 4096;
    assert(!bn_transformer_gpu_dense_batch_prefill_shape_allowed_for_backend(
        &c, 0));
    assert(bn_transformer_gpu_dense_batch_prefill_shape_allowed_for_backend(
        &c, 1));
    c.dim = 2048;
    assert(bn_transformer_gpu_dense_logits_argmax_shape_allowed(&c, 300000));
    c.n_experts = 128;
    assert(bn_transformer_gpu_moe_logits_mmvq_argmax_shape_allowed(&c,
                                                                    1536));
    assert(!bn_transformer_gpu_dense_logits_argmax_shape_allowed(&c,
                                                                 300000));
    c.n_experts = 0;
    assert(bn_transformer_gpu_dense_logits_argmax_shape_allowed(&c, 300000));
    assert(!bn_transformer_gpu_moe_logits_mmvq_argmax_shape_allowed(&c,
                                                                    1536));
    c.dim = 0;

    BnGPUBackend gpu;
    BnTransformerGPULogitResources logits;
    BnQWeight W;
    memset(&gpu, 0, sizeof(gpu));
    memset(&logits, 0, sizeof(logits));
    memset(&W, 0, sizeof(W));
    assert(bn_transformer_gpu_refine_kquant_logits_top(
               NULL, 0, NULL, NULL, NULL, 0) == 0);
    assert(bn_transformer_gpu_refine_native_quant_logits_top(
               NULL, 0, NULL, NULL, NULL, 0) == 0);
    assert(bn_transformer_gpu_try_refined_argmax(
               NULL, NULL, NULL, NULL, NULL, 0, 0,
               NULL, 0, 1.0f, NULL) == 0);
    bn_transformer_gpu_refine_output_logits(
        NULL, NULL, NULL, NULL, NULL, 0, 0);
    assert(bn_transformer_gpu_resolve_moe_route(
               NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
               0, 0, 0, 0, NULL) == -1);
    assert(bn_transformer_gpu_prepare_routed_moe_route(
               NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
               0, 0, 0, NULL) == -1);
    assert(bn_transformer_gpu_debug_compare_routed_moe_raw(
               NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
               NULL, 0, 0, 0) == 0);
    assert(bn_transformer_gpu_debug_compare_routed_moe_mid(
               NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0) == 0);
    assert(bn_transformer_gpu_prepare_routed_moe_debug_state(
               NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
               0, 0, 0, 0.0f) == -1);
    assert(bn_transformer_gpu_prepare_routed_moe_parts_comparison(
               NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0) ==
           -1);
    bn_transformer_gpu_compare_routed_moe_shared_part(
        NULL, NULL, NULL, 0, 0, 0);
    bn_transformer_gpu_debug_compare_routed_moe_post_layer(
        NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0.0f);
    assert(bn_transformer_gpu_complete_routed_moe_debug_state(
               NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
               0, 0, 0, 0, 0.0f) == -1);
    bn_transformer_gpu_discard_routed_moe_debug_state(NULL);
    bn_transformer_gpu_discard_routed_moe_parts_comparison(NULL);
    assert(bn_transformer_gpu_prepare_moe_layer_comparison(
               NULL, NULL, NULL, NULL, NULL, NULL, 0, 0) == -1);
    assert(bn_transformer_gpu_complete_moe_layer_comparison(
               NULL, NULL, NULL, 0, 0, 0, 0.0f) == -1);
    bn_transformer_gpu_discard_moe_layer_comparison(NULL);
    assert(bn_transformer_gpu_debug_snapshot_attention_state(
               NULL, NULL, NULL, 0) == -1);
    assert(bn_transformer_gpu_debug_snapshot_ffn_state(
               NULL, NULL, NULL, 0) == -1);
    assert(bn_transformer_gpu_capture_logits_refine_state(
               NULL, NULL, NULL, 0) == -1);
    assert(bn_transformer_gpu_flush_and_release_moe_temporaries(
               NULL, NULL, NULL, NULL) == -1);
    assert(bn_transformer_gpu_stage_token_input(
               NULL, NULL, NULL, 0) == -1);
    float gpu_logits[] = {4.0f, 3.0f, 2.0f};
    float gpu_xb[] = {1.0f, -1.0f};
    float host_logits[3] = {0};
    float host_xb[2] = {0};
    int8_t host_xq[2] = {0};
    int penalty_token = 0;
    int refined_argmax = -1;
    MockGPURefineReadback refine_readback = {gpu_logits, gpu_xb};
    BnModel refine_model;
    BnSession refine_session;
    BnTransformerGPULogitsRefinePolicy mock_refine_policy;
    memset(&refine_model, 0, sizeof(refine_model));
    memset(&refine_session, 0, sizeof(refine_session));
    memset(&mock_refine_policy, 0, sizeof(mock_refine_policy));
    refine_model.config.vocab_size = 3;
    refine_session.state.logits = host_logits;
    refine_session.state.xb = host_xb;
    refine_session.state.x_q = host_xq;
    logits.cpu_weight = &W;
    mock_refine_policy.native_quant_captures_xb = 1;
    mock_refine_policy.native_quant_refine_top = 1;
    gpu.ctx = &refine_readback;
    gpu.read_activation = mock_gpu_refine_read_activation;
    assert(bn_transformer_gpu_try_refined_argmax(
               &gpu, &refine_model, &refine_session, &logits,
               &mock_refine_policy, 2, 0, &penalty_token, 1, 2.0f,
               &refined_argmax) == 1);
    assert(refined_argmax == 1);
    mock_refine_policy.native_quant_captures_xb = 0;
    mock_refine_policy.native_quant_refine_top = 0;
    mock_refine_policy.kquant_captures_xb = 1;
    mock_refine_policy.kquant_refine_top = 1;
    refined_argmax = -1;
    assert(bn_transformer_gpu_try_refined_argmax(
               &gpu, &refine_model, &refine_session, &logits,
               &mock_refine_policy, 2, 1, &penalty_token, 1, 2.0f,
               &refined_argmax) == 1);
    assert(refined_argmax == 1);
    assert(bn_transformer_gpu_fallback_shared_expert_mid(
               NULL, NULL, NULL, NULL, NULL) == -1);
    assert(bn_transformer_gpu_fallback_shared_expert_output(
               NULL, NULL, NULL, 0, NULL, NULL) == -1);
    assert(bn_transformer_gpu_fallback_shared_expert_residual(
               NULL, NULL, NULL, NULL, NULL, 0) == -1);
    assert(bn_transformer_gpu_fallback_shared_expert_down(
               NULL, NULL, NULL, 0, NULL, NULL) == -1);
    assert(bn_transformer_gpu_fallback_moe_output_from_state(
               NULL, NULL, NULL, 0, 0, NULL) == -1);
    assert(bn_transformer_gpu_fallback_moe_mid(
               NULL, NULL, NULL, NULL, NULL) == -1);
    assert(bn_transformer_gpu_fallback_moe_raw_gate_up(
               NULL, NULL, NULL, NULL, NULL, NULL) == -1);
    assert(bn_transformer_gpu_fallback_moe_output(
               NULL, NULL, NULL, 0, NULL, NULL, NULL) == -1);
    assert(bn_transformer_gpu_fallback_moe_parts(
               NULL, NULL, NULL, 0, NULL, NULL, NULL) == -1);
    BnTransformerGPUMoEExecutionPolicy moe_execution =
        bn_transformer_gpu_moe_execution_policy(NULL);
    assert(moe_execution.total_experts == 0);
    assert(moe_execution.active_experts == 0);
    assert(moe_execution.expert_hidden_dim == 0);
    assert(!bn_transformer_gpu_moe_projection_policy(NULL).valid);
    c.n_experts = 4;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 64;
    c.moe_norm_topk_prob = 1;
    c.moe_expert_weights_scale = 0.5f;
    moe_execution = bn_transformer_gpu_moe_execution_policy(&c);
    assert(moe_execution.total_experts == 4);
    assert(moe_execution.active_experts == 2);
    assert(moe_execution.expert_hidden_dim == 64);
    assert(moe_execution.normalize_topk);
    assert(moe_execution.expert_weights_scale == 0.5f);
    c.n_experts = 0;
    c.n_experts_active = 0;
    gpu.caps |= BN_GPU_CAP_DECODE_GRAPH_CACHE;
    c.moe_intermediate_size = 0;
    c.moe_norm_topk_prob = 0;
    c.moe_expert_weights_scale = 0.0f;
    BnMoEExpertMap projection_map = {0};
    projection_map.gate_type = BN_GGUF_TENSOR_Q4_K;
    projection_map.up_type = BN_GGUF_TENSOR_Q6_K;
    projection_map.down_type = BN_GGUF_TENSOR_F16;
    BnTransformerGPUMoEProjectionPolicy projection_policy =
        bn_transformer_gpu_moe_projection_policy(&projection_map);
    assert(projection_policy.valid);
    assert(projection_policy.gate_type == BN_GGUF_TENSOR_Q4_K);
    assert(projection_policy.up_type == BN_GGUF_TENSOR_Q6_K);
    assert(projection_policy.down_type == BN_GGUF_TENSOR_F16);
    assert(bn_transformer_gpu_backend_placement(&gpu) ==
           BN_BACKEND_GPU_UNKNOWN);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_backend_placement(&gpu) ==
           BN_BACKEND_CUDA);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_backend_placement(&gpu) ==
           BN_BACKEND_METAL);
    gpu.kind = BN_GPU_BACKEND_WEBGPU;
    assert(bn_transformer_gpu_backend_placement(&gpu) ==
           BN_BACKEND_WEBGPU);
    gpu.kind = BN_GPU_BACKEND_UNKNOWN;
    assert(bn_transformer_backend_placement(&gpu, BN_EXEC_GPU) ==
           BN_BACKEND_GPU_UNKNOWN);
    assert(bn_transformer_backend_placement(&gpu, BN_EXEC_CPU) ==
           BN_BACKEND_CPU);

    {
        BnTransformerGPUForwardPolicy forward_policy;
        BnWeights weights;
        const char *reject_reason = NULL;
        float rope_freqs[1] = {1.0f};
        memset(&weights, 0, sizeof(weights));
        weights.rope_freqs = rope_freqs;
        c.vocab_size = 8;
        c.dim = 2560;
        c.policy_flags = BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT;
        c.per_layer_input_dim = 128;
        gpu.kind = BN_GPU_BACKEND_METAL;
        gpu.execute = mock_gpu_execute;
        gpu.write_activation = mock_gpu_write_activation;
        gpu.caps = 0;
        c.full_attn_interval = 4;
        c.ssm_inner_size = 128;
        assert(bn_transformer_gpu_validate_forward(
                   &forward_policy, &gpu, NULL, &c, &weights, 0, 0,
                   &reject_reason) != 0);
        assert(reject_reason &&
               strcmp(reject_reason,
                      "hybrid ssm graph unsupported by gpu backend") == 0);
        gpu.caps = BN_GPU_CAP_SSM_GRAPH;
        c.n_experts = 8;
        c.n_experts_active = 2;
        reject_reason = NULL;
        assert(bn_transformer_gpu_validate_forward(
                   &forward_policy, &gpu, NULL, &c, &weights, 0, 0,
                   &reject_reason) != 0);
        assert(reject_reason &&
               strcmp(reject_reason,
                      "combined hybrid ssm/moe graph unsupported by gpu backend") == 0);
        gpu.caps |= BN_GPU_CAP_HYBRID_SSM_MOE_GRAPH |
                    BN_GPU_CAP_LARGE_GRAPH_NATIVE;
        c.policy_flags |= BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
        reject_reason = NULL;
        assert(bn_transformer_gpu_validate_forward(
                   &forward_policy, &gpu, NULL, &c, &weights, 0, 0,
                   &reject_reason) != 0);
        assert(reject_reason &&
               strcmp(reject_reason,
                      "hyper-connection graph unsupported by gpu backend") == 0);
        c.policy_flags &= ~BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
        reject_reason = NULL;
        assert(bn_transformer_gpu_validate_forward(
                   &forward_policy, &gpu, NULL, &c, &weights, 0, 0,
                   &reject_reason) != 0);
        assert(reject_reason &&
               strcmp(reject_reason,
                      "layerwise rope unsupported by gpu backend") == 0);
        c.n_experts = 0;
        c.n_experts_active = 0;
        c.full_attn_interval = 0;
        c.ssm_inner_size = 0;
        reject_reason = NULL;
        assert(bn_transformer_gpu_requires_layerwise_rope(&c, &weights));
        assert(bn_transformer_gpu_validate_forward(
                   &forward_policy, &gpu, NULL, &c, &weights, 0, 0,
                   &reject_reason) != 0);
        assert(reject_reason &&
               strcmp(reject_reason,
                      "layerwise rope unsupported by gpu backend") == 0);
        gpu.caps = BN_GPU_CAP_LAYERWISE_ROPE;
        reject_reason = NULL;
        assert(bn_transformer_gpu_validate_forward(
                   &forward_policy, &gpu, NULL, &c, &weights, 0, 0,
                   &reject_reason) != 0);
        assert(reject_reason &&
               strcmp(reject_reason,
                      "per-layer input graph unsupported by gpu backend") == 0);
        gpu.caps |= BN_GPU_CAP_PER_LAYER_INPUT_GRAPH;
        reject_reason = NULL;
        assert(bn_transformer_gpu_validate_forward(
                   &forward_policy, &gpu, NULL, &c, &weights, 0, 0,
                   &reject_reason) != 0);
        assert(forward_policy.gpu == &gpu);
        assert(reject_reason &&
               strcmp(reject_reason, "output norm not uploaded") == 0);
        memset(&gpu, 0, sizeof(gpu));
    }

    c.policy_flags = BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
                     BN_MODEL_ARCH_POLICY_PREFILL_REFERENCE_ACTIVATION;
    assert(!bn_transformer_prefill_small_dense_chain_applicable(
        &gpu, &c));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_prefill_small_dense_chain_applicable(
        &gpu, &c));
    c.full_attn_interval = 4;
    assert(!bn_transformer_prefill_small_dense_chain_applicable(
        &gpu, &c));
    c.full_attn_interval = 0;
    gpu.kind = BN_GPU_BACKEND_UNKNOWN;

    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_prefill_ssm_layer_disabled(&gpu, NULL));
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_prefill_ssm_layer_disabled(&gpu, NULL));
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    test_gpu_runtime_refresh(&gpu);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    c.policy_flags |= BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
    c.hyper_connection_count = 4;
    assert(bn_transformer_gpu_prefill_ssm_layer_disabled(&gpu, &c));
    assert(bn_transformer_gpu_cpu_batch_prefill_fallback_enabled(&gpu, &c));
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
    c.hyper_connection_count = 0;
    gpu.kind = BN_GPU_BACKEND_UNKNOWN;

    unsetenv("BN_GPU_PROFILE");
    assert(bn_transformer_gpu_profile_level(&gpu) == 0);
    setenv("BN_GPU_PROFILE", "3", 1);
    assert(bn_backend_runtime_policy_set(&gpu.runtime_policy,
                                         "BN_GPU_PROFILE", "3", 1) == 0);
    assert(bn_transformer_gpu_profile_level(&gpu) == 3);
    bn_backend_runtime_policy_free(&gpu.runtime_policy);
    unsetenv("BN_GPU_PROFILE");

    unsetenv("BN_GPU_MOE_ROUTE_PROFILE");
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY");
    assert(!bn_transformer_gpu_moe_route_profile_enabled(&gpu));
    assert(bn_transformer_gpu_moe_route_profile_every(&gpu) == 28);
    setenv("BN_GPU_MOE_ROUTE_PROFILE", "1", 1);
    setenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY", "5", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_route_profile_enabled(&gpu));
    assert(bn_transformer_gpu_moe_route_profile_every(&gpu) == 5);
    setenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY", "0", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_route_profile_every(&gpu) == 28);
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE");
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY");

    assert(!bn_gpu_policy_auto_caps_sequence(0, 0, 0, 0, 8193, 4096));
    assert(!bn_gpu_policy_auto_caps_sequence(1, 0, 0, 0, 4096, 4096));
    assert(bn_gpu_policy_auto_caps_sequence(1, 0, 0, 0, 4097, 4096));
    assert(bn_gpu_policy_auto_caps_sequence(0, 1, 0, 0, 4097, 4096));
    assert(!bn_gpu_policy_auto_caps_sequence(0, 0, 1, 0, 4097, 4096));
    assert(bn_gpu_policy_auto_caps_sequence(0, 0, 1, 1, 4097, 4096));

    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_DISABLE_MOE_FFN");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_ffn_disabled(&gpu));
    setenv("BN_CUDA_DISABLE_MOE_FFN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_ffn_disabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_moe_ffn_disabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_UNKNOWN;
    unsetenv("BN_CUDA_DISABLE_MOE_FFN");
    test_gpu_runtime_refresh(&gpu);

    unsetenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    unsetenv("BN_GPU_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_cpu_actual_override_enabled(&gpu, 0));
    assert(bn_transformer_gpu_moe_cpu_actual_override_enabled(&gpu, 1));
    setenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_moe_cpu_actual_override_enabled(&gpu, 0));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_moe_cpu_actual_override_enabled(&gpu, 0));
    setenv("BN_GPU_OVERRIDE_MOE_WITH_CPU_ACTUAL", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_cpu_actual_override_enabled(&gpu, 0));
    unsetenv("BN_GPU_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    gpu.kind = BN_GPU_BACKEND_UNKNOWN;
    unsetenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    unsetenv("BN_GPU_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    test_gpu_runtime_refresh(&gpu);

    unsetenv("BN_GPU_COMPARE_MOE_LAYER");
    unsetenv("BN_GPU_COMPARE_MOE_POS");
    assert(!bn_gpu_policy_moe_compare_layer_selected(test_current_backend_runtime(), 3, 7));
    setenv("BN_GPU_COMPARE_MOE_LAYER", "3", 1);
    assert(bn_gpu_policy_moe_compare_layer_selected(test_current_backend_runtime(), 3, 7));
    assert(!bn_gpu_policy_moe_compare_layer_selected(test_current_backend_runtime(), 4, 7));
    setenv("BN_GPU_COMPARE_MOE_POS", "7", 1);
    assert(bn_gpu_policy_moe_compare_layer_selected(test_current_backend_runtime(), 3, 7));
    assert(!bn_gpu_policy_moe_compare_layer_selected(test_current_backend_runtime(), 3, 8));
    setenv("BN_GPU_COMPARE_MOE_LAYER", "all", 1);
    assert(bn_gpu_policy_moe_compare_layer_selected(test_current_backend_runtime(), 3, 7));
    assert(bn_gpu_policy_moe_compare_layer_selected(test_current_backend_runtime(), 19, 7));
    assert(!bn_gpu_policy_moe_compare_layer_selected(test_current_backend_runtime(), 19, 8));
    unsetenv("BN_GPU_COMPARE_MOE_LAYER");
    unsetenv("BN_GPU_COMPARE_MOE_POS");

    unsetenv("BN_GPU_COMPARE_MOE_INPUT_NORM");
    unsetenv("BN_GPU_COMPARE_MOE_ACTUAL");
    unsetenv("BN_GPU_COMPARE_MOE_ROUTE");
    unsetenv("BN_GPU_COMPARE_MOE_RAW");
    unsetenv("BN_GPU_COMPARE_MOE_MID");
    unsetenv("BN_GPU_COMPARE_MOE_PARTS");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_MID");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_DOWN");
    unsetenv("BN_GPU_COMPARE_MOE_NORM");
    assert(!bn_gpu_policy_moe_compare_input_norm_enabled(test_current_backend_runtime()));
    assert(!bn_gpu_policy_moe_compare_actual_enabled(test_current_backend_runtime()));
    assert(!bn_gpu_policy_moe_compare_route_enabled(test_current_backend_runtime()));
    assert(!bn_gpu_policy_moe_compare_raw_enabled(test_current_backend_runtime()));
    assert(!bn_gpu_policy_moe_compare_mid_enabled(test_current_backend_runtime()));
    assert(!bn_gpu_policy_moe_compare_parts_enabled(test_current_backend_runtime()));
    assert(!bn_gpu_policy_moe_compare_shared_mid_enabled(test_current_backend_runtime()));
    assert(!bn_gpu_policy_moe_compare_shared_down_enabled(test_current_backend_runtime()));
    assert(!bn_gpu_policy_moe_compare_norm_enabled(test_current_backend_runtime()));
    setenv("BN_GPU_COMPARE_MOE_INPUT_NORM", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_ACTUAL", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_ROUTE", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_RAW", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_MID", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_PARTS", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_SHARED_MID", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_SHARED_DOWN", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_NORM", "1", 1);
    assert(bn_gpu_policy_moe_compare_input_norm_enabled(test_current_backend_runtime()));
    assert(bn_gpu_policy_moe_compare_actual_enabled(test_current_backend_runtime()));
    assert(bn_gpu_policy_moe_compare_route_enabled(test_current_backend_runtime()));
    assert(bn_gpu_policy_moe_compare_raw_enabled(test_current_backend_runtime()));
    assert(bn_gpu_policy_moe_compare_mid_enabled(test_current_backend_runtime()));
    assert(bn_gpu_policy_moe_compare_parts_enabled(test_current_backend_runtime()));
    assert(bn_gpu_policy_moe_compare_shared_mid_enabled(test_current_backend_runtime()));
    assert(bn_gpu_policy_moe_compare_shared_down_enabled(test_current_backend_runtime()));
    assert(bn_gpu_policy_moe_compare_norm_enabled(test_current_backend_runtime()));
    unsetenv("BN_GPU_COMPARE_MOE_INPUT_NORM");
    unsetenv("BN_GPU_COMPARE_MOE_ACTUAL");
    unsetenv("BN_GPU_COMPARE_MOE_ROUTE");
    unsetenv("BN_GPU_COMPARE_MOE_RAW");
    unsetenv("BN_GPU_COMPARE_MOE_MID");
    unsetenv("BN_GPU_COMPARE_MOE_PARTS");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_MID");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_DOWN");
    unsetenv("BN_GPU_COMPARE_MOE_NORM");

    unsetenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    BnTransformerGPUMoEDebugPolicy moe_debug =
        bn_transformer_gpu_moe_debug_policy(&gpu, 0, 0);
    assert(!moe_debug.override_cpu_actual);
    assert(!moe_debug.compare_layer);
    assert(!moe_debug.compare_route);
    assert(!moe_debug.compare_input_norm);
    assert(!moe_debug.compare_actual);
    assert(!moe_debug.compare_raw);
    assert(!moe_debug.compare_mid);
    assert(!moe_debug.compare_parts);
    assert(!moe_debug.compare_shared_mid);
    assert(!moe_debug.compare_shared_down);
    assert(!moe_debug.compare_norm);
    moe_debug = bn_transformer_gpu_moe_debug_policy(&gpu, 1, 0);
    assert(moe_debug.override_cpu_actual);
    assert(!moe_debug.compare_layer);
    moe_debug = bn_transformer_gpu_moe_debug_policy(&gpu, 0, 1);
    assert(!moe_debug.override_cpu_actual);
    assert(moe_debug.compare_layer);
    assert(!moe_debug.compare_route);
    assert(!moe_debug.compare_input_norm);
    assert(!moe_debug.compare_actual);
    assert(!moe_debug.compare_raw);
    assert(!moe_debug.compare_mid);
    assert(!moe_debug.compare_parts);
    assert(!moe_debug.compare_shared_mid);
    assert(!moe_debug.compare_shared_down);
    assert(!moe_debug.compare_norm);
    setenv("BN_GPU_COMPARE_MOE_LAYER", "2", 1);
    test_gpu_runtime_refresh(&gpu);
    moe_debug = bn_transformer_gpu_moe_decode_debug_policy(
        &gpu, &c, NULL, 2, 0);
    assert(!moe_debug.override_cpu_actual);
    assert(moe_debug.compare_layer);
    unsetenv("BN_GPU_COMPARE_MOE_LAYER");
    setenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_ROUTE", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_INPUT_NORM", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_ACTUAL", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_RAW", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_MID", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_PARTS", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_SHARED_MID", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_SHARED_DOWN", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_NORM", "1", 1);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    moe_debug = bn_transformer_gpu_moe_debug_policy(&gpu, 0, 0);
    assert(moe_debug.override_cpu_actual);
    assert(!moe_debug.compare_layer);
    assert(!moe_debug.compare_route);
    assert(!moe_debug.compare_input_norm);
    assert(!moe_debug.compare_actual);
    assert(!moe_debug.compare_raw);
    assert(!moe_debug.compare_mid);
    assert(!moe_debug.compare_parts);
    assert(!moe_debug.compare_shared_mid);
    assert(!moe_debug.compare_shared_down);
    assert(!moe_debug.compare_norm);
    moe_debug = bn_transformer_gpu_moe_debug_policy(&gpu, 0, 1);
    assert(moe_debug.override_cpu_actual);
    assert(moe_debug.compare_layer);
    assert(moe_debug.compare_route);
    assert(moe_debug.compare_input_norm);
    assert(moe_debug.compare_actual);
    assert(moe_debug.compare_raw);
    assert(moe_debug.compare_mid);
    assert(moe_debug.compare_parts);
    assert(moe_debug.compare_shared_mid);
    assert(moe_debug.compare_shared_down);
    assert(moe_debug.compare_norm);
    unsetenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    unsetenv("BN_GPU_COMPARE_MOE_ROUTE");
    unsetenv("BN_GPU_COMPARE_MOE_INPUT_NORM");
    unsetenv("BN_GPU_COMPARE_MOE_ACTUAL");
    unsetenv("BN_GPU_COMPARE_MOE_RAW");
    unsetenv("BN_GPU_COMPARE_MOE_MID");
    unsetenv("BN_GPU_COMPARE_MOE_PARTS");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_MID");
    unsetenv("BN_GPU_COMPARE_MOE_SHARED_DOWN");
    unsetenv("BN_GPU_COMPARE_MOE_NORM");

    unsetenv("BN_GPU_CPU_LOGITS");
    unsetenv("BN_GPU_DEBUG_ARGMAX_COMPARE");
    unsetenv("BN_GPU_COMPARE_LOGITS");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_cpu_logits_enabled(&gpu, 0));
    assert(bn_transformer_gpu_cpu_logits_enabled(&gpu, 1));
    assert(!bn_transformer_gpu_debug_argmax_compare_enabled(&gpu));
    assert(!bn_transformer_gpu_compare_logits_enabled(&gpu));
    setenv("BN_GPU_CPU_LOGITS", "1", 1);
    setenv("BN_GPU_DEBUG_ARGMAX_COMPARE", "1", 1);
    setenv("BN_GPU_COMPARE_LOGITS", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_cpu_logits_enabled(&gpu, 0));
    assert(bn_transformer_gpu_debug_argmax_compare_enabled(&gpu));
    assert(bn_transformer_gpu_compare_logits_enabled(&gpu));
    unsetenv("BN_GPU_CPU_LOGITS");
    unsetenv("BN_GPU_DEBUG_ARGMAX_COMPARE");
    unsetenv("BN_GPU_COMPARE_LOGITS");

    unsetenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK");
    unsetenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK");
    BnLayerWeights shared_layer;
    memset(&shared_layer, 0, sizeof(shared_layer));
    c.has_shared_expert = 1;
    assert(!bn_transformer_gpu_moe_has_loaded_shared_expert(
        &c, &shared_layer));
    shared_layer.shared.shared_gate.data = (void *)1;
    assert(bn_transformer_gpu_moe_has_loaded_shared_expert(
        &c, &shared_layer));
    c.has_shared_expert = 0;
    assert(!bn_transformer_gpu_moe_has_loaded_shared_expert(
        &c, &shared_layer));
    shared_layer.shared.shared_expert_gate = (float *)1;
    assert(bn_transformer_gpu_moe_has_loaded_shared_expert(
        &c, &shared_layer));
    shared_layer.shared.shared_expert_gate = NULL;
    shared_layer.shared.shared_gate.data = NULL;
    c.has_shared_expert = 1;
    BnTransformerGPUMoESharedCPUFallbackPolicy shared_fallback =
        bn_transformer_gpu_moe_shared_cpu_fallback_policy(
            &gpu, &c, &shared_layer);
    assert(!shared_fallback.enabled);
    shared_layer.shared.shared_gate.data = (void *)1;
    shared_fallback = bn_transformer_gpu_moe_shared_cpu_fallback_policy(
        &gpu, &c, &shared_layer);
    assert(!shared_fallback.enabled);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(!bn_transformer_gpu_moe_shared_cpu_fallback_enabled(&gpu, 0));
    assert(!bn_transformer_gpu_moe_shared_cpu_fallback_enabled(&gpu, 1));
    setenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy,
               "BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK", "1", 1) == 0);
    shared_fallback = bn_transformer_gpu_moe_shared_cpu_fallback_policy(
        &gpu, &c, &shared_layer);
    assert(shared_fallback.enabled);
    c.has_shared_expert = 0;
    shared_fallback = bn_transformer_gpu_moe_shared_cpu_fallback_policy(
        &gpu, &c, &shared_layer);
    assert(!shared_fallback.enabled);
    c.has_shared_expert = 1;
    assert(!bn_transformer_gpu_moe_shared_cpu_fallback_enabled(&gpu, 0));
    assert(bn_transformer_gpu_moe_shared_cpu_fallback_enabled(&gpu, 1));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_moe_shared_cpu_fallback_enabled(&gpu, 1));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    setenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy,
               "BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK", "1", 1) == 0);
    shared_fallback = bn_transformer_gpu_moe_shared_cpu_fallback_policy(
        &gpu, &c, &shared_layer);
    assert(!shared_fallback.enabled);
    assert(!bn_transformer_gpu_moe_shared_cpu_fallback_enabled(&gpu, 1));
    unsetenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK");
    unsetenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK");
    bn_backend_runtime_policy_free(&gpu.runtime_policy);
    c.has_shared_expert = 0;

    setenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_decode_cacheable(&gpu, &c, NULL, NULL));
    unsetenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE");
    test_gpu_runtime_refresh(&gpu);

    BnModel model;
    BnLayerWeights layer;
    memset(&model, 0, sizeof(model));
    memset(&layer, 0, sizeof(layer));
    model.config.dim = 2048;
    model.config.n_layers = 1;
    model.config.policy_flags =
        BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK;
    model.weights.layers = &layer;
    model.weights.emb_type = BN_GGUF_TENSOR_Q8_0;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    gpu.execute = mock_gpu_execute;
    unsetenv("BN_CUDA_ENABLE_SMALL_STATE_NATIVE_QUANT");
    unsetenv("BN_CUDA_DISABLE_SMALL_STATE_NATIVE_QUANT");
    unsetenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE");
    unsetenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    BnTransformerGPUMatvecFallbackPolicy matvec_fallback =
        bn_transformer_gpu_matvec_fallback_policy(&model, &gpu);
    assert(matvec_fallback.keep_backend_matvec);
    assert(!matvec_fallback.disable_backend_matvec);
    model.weights.emb_type = BN_GGUF_TENSOR_Q4_K;
    assert(!bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    matvec_fallback =
        bn_transformer_gpu_matvec_fallback_policy(&model, &gpu);
    assert(!matvec_fallback.keep_backend_matvec);
    assert(matvec_fallback.disable_backend_matvec);
    setenv("BN_CUDA_ENABLE_SMALL_STATE_NATIVE_QUANT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    matvec_fallback =
        bn_transformer_gpu_matvec_fallback_policy(&model, &gpu);
    assert(matvec_fallback.keep_backend_matvec);
    assert(!matvec_fallback.disable_backend_matvec);
    unsetenv("BN_CUDA_ENABLE_SMALL_STATE_NATIVE_QUANT");
    test_gpu_runtime_refresh(&gpu);
    model.weights.emb_type = BN_GGUF_TENSOR_Q8_0;
    layer.attn.wq.data = (void *)1;
    layer.attn.wq.type = BN_GGUF_TENSOR_Q4_K;
    assert(!bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    layer.attn.wq.type = BN_GGUF_TENSOR_Q8_0;
    assert(bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    model.config.n_experts = 1;
    assert(!bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    assert(!bn_transformer_gpu_backend_cpu_operations_kept(&model, &gpu));
    assert(bn_model_ensure_backend(&model) == 0);
    BnBackendModel *fallback_backend = bn_model_backend(&model);
    bn_backend_model_bind_gpu(fallback_backend, &gpu);
    gpu.moe_routed_ffn_batch = mock_moe_routed_ffn_batch;
    layer.moe.router_weight = (float *)1;
    int resident_gate, resident_up, resident_down;
    assert(bn_backend_model_register_handle(
               fallback_backend, 0, BN_BACKEND_HANDLE_MOE_GATE_ALL,
               &resident_gate) == 0);
    assert(bn_backend_model_register_handle(
               fallback_backend, 0, BN_BACKEND_HANDLE_MOE_UP_ALL,
               &resident_up) == 0);
    assert(bn_backend_model_register_handle(
               fallback_backend, 0, BN_BACKEND_HANDLE_MOE_DOWN_ALL,
               &resident_down) == 0);
    assert(bn_transformer_gpu_backend_cpu_operations_kept(&model, &gpu));
    matvec_fallback =
        bn_transformer_gpu_matvec_fallback_policy(&model, &gpu);
    assert(!matvec_fallback.keep_backend_matvec);
    assert(matvec_fallback.keep_backend_operations);
    assert(!matvec_fallback.disable_backend_matvec);
    layer.moe.router_weight = NULL;
    gpu.moe_routed_ffn_batch = NULL;
    model.config.n_experts = 0;
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    gpu.execute = NULL;
    assert(!bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    unsetenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE");

    unsetenv("BN_GPU_DISABLE_PREFILL_MATMUL");
    unsetenv("BN_GPU_PREFILL_MATMUL");
    memset(&c, 0, sizeof(c));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    c.dim = 8192;
    assert(bn_transformer_gpu_dense_batch_prefill_shape_allowed(&gpu, &c));
    assert(bn_transformer_gpu_batch_prefill_enabled(&gpu, &c));
    c.dim = 8193;
    assert(!bn_transformer_gpu_dense_batch_prefill_shape_allowed(&gpu, &c));
    assert(!bn_transformer_gpu_batch_prefill_enabled(&gpu, &c));
    gpu.kind = BN_GPU_BACKEND_METAL;
    c.dim = 2560;
    assert(bn_transformer_gpu_dense_batch_prefill_shape_allowed(&gpu, &c));
    assert(bn_transformer_gpu_batch_prefill_enabled(&gpu, &c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY;
    assert(!bn_transformer_gpu_batch_prefill_enabled(&gpu, &c));
    setenv("BN_GPU_PREFILL_MATMUL", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_batch_prefill_enabled(&gpu, &c));
    unsetenv("BN_GPU_PREFILL_MATMUL");
    test_gpu_runtime_refresh(&gpu);
    c.policy_flags = 0;
    c.dim = 2561;
    assert(!bn_transformer_gpu_dense_batch_prefill_shape_allowed(&gpu, &c));
    assert(!bn_transformer_gpu_batch_prefill_enabled(&gpu, &c));
    assert(!bn_transformer_gpu_dense_batch_prefill_shape_allowed(&gpu, NULL));
    c.dim = 9000;
    setenv("BN_GPU_PREFILL_MATMUL", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_batch_prefill_enabled(&gpu, &c));
    unsetenv("BN_GPU_PREFILL_MATMUL");

    assert(bn_transformer_gpu_moe_gateup_task_flags(&c) == 0);
    c.policy_flags |= BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP_FALLBACK;
    assert(bn_transformer_gpu_moe_gateup_task_flags(&c) ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);
    c.policy_flags = 0;
    assert(!bn_transformer_gpu_moe_activation_policy(&c).uses_reference_silu);
    assert(!bn_transformer_gpu_moe_activation_policy(&c)
                .uses_reference_ffn_activation);
    c.policy_flags = BN_MODEL_ARCH_POLICY_MOE_REFERENCE_SILU;
    c.moe_uses_reference_silu = 1;
    assert(bn_transformer_gpu_moe_activation_policy(&c).uses_reference_silu);
    assert(!bn_transformer_gpu_moe_activation_policy(&c)
                .uses_reference_ffn_activation);
    c.policy_flags |= BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION;
    assert(bn_transformer_gpu_moe_activation_policy(&c)
               .uses_reference_ffn_activation);
    c.policy_flags = 0;
    c.moe_uses_reference_silu = 0;
    c.has_shared_expert = 1;
    c.shared_expert_intermediate_size = 2048;
    assert(bn_transformer_gpu_moe_shared_expert_shape_policy(&c).hidden_dim ==
           2048);
    c.has_shared_expert = 0;
    assert(bn_transformer_gpu_moe_shared_expert_shape_policy(&c).hidden_dim ==
           0);
    memset(&c, 0, sizeof(c));
    c.dim = 2048;
    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    c.moe_norm_topk_prob = 1;
    assert(bn_transformer_moe_uses_all_active_two_route(&c, c.dim));
    assert(bn_transformer_moe_uses_all_active_two_expert_set(&c));
    assert(bn_transformer_moe_uses_configured_all_active_two_route(&c));
    assert(!bn_transformer_moe_uses_grouped_route(&c));
    assert(bn_transformer_moe_normalizes_topk_route_weights(&c));
    BnMoEExpertMap resident_map = {0};
    resident_map.gate_rows = 4096;
    resident_map.up_rows = 4096;
    resident_map.gate_cols = 2048;
    resident_map.up_cols = 2048;
    resident_map.down_rows = 2048;
    resident_map.down_cols = 4096;
    assert(bn_transformer_moe_supports_resident_routed_ffn_shape(
        &c, &resident_map, c.dim));
    assert(bn_transformer_moe_supports_resident_routed_ffn_layout(
        &c, &resident_map));
    resident_map.down_cols = 4095;
    assert(!bn_transformer_moe_supports_resident_routed_ffn_shape(
        &c, &resident_map, c.dim));
    assert(!bn_transformer_moe_supports_resident_routed_ffn_layout(
        &c, &resident_map));
    c.moe_norm_topk_prob = 0;
    assert(!bn_transformer_moe_normalizes_topk_route_weights(&c));
    assert(!bn_transformer_prefill_shared_all_active_two_decode_fallback_policy(
                &c, 0).enabled);
    c.has_shared_expert = 1;
    assert(!bn_transformer_prefill_shared_all_active_two_decode_fallback_policy(
                &c, 0).enabled);
    assert(!bn_transformer_prefill_shared_all_active_two_decode_fallback_policy(
                &c, 1).enabled);
    BnTransformerPrefillEntryDispatchPolicy prefill_entry =
        bn_transformer_prefill_entry_dispatch_policy(
            &c, 0, BN_TRANSFORMER_PREFILL_REQUEST_LAST_LOGITS);
    assert(!prefill_entry.uses_decode_fallback);
    assert(prefill_entry.path == BN_TRANSFORMER_PREFILL_ENTRY_BATCH);
    prefill_entry = bn_transformer_prefill_entry_dispatch_policy(
        &c, 0, BN_TRANSFORMER_PREFILL_REQUEST_NO_LOGITS);
    assert(!prefill_entry.uses_decode_fallback);
    assert(prefill_entry.path == BN_TRANSFORMER_PREFILL_ENTRY_BATCH);
    prefill_entry = bn_transformer_prefill_entry_dispatch_policy(
        &c, 0, BN_TRANSFORMER_PREFILL_REQUEST_ALL_LOGITS);
    assert(!prefill_entry.uses_decode_fallback);
    assert(prefill_entry.path == BN_TRANSFORMER_PREFILL_ENTRY_BATCH);
    prefill_entry = bn_transformer_prefill_entry_dispatch_policy(
        &c, 1, BN_TRANSFORMER_PREFILL_REQUEST_LAST_LOGITS);
    assert(!prefill_entry.uses_decode_fallback);
    assert(prefill_entry.path == BN_TRANSFORMER_PREFILL_ENTRY_BATCH);
    prefill_entry = bn_transformer_prefill_entry_dispatch_policy(
        NULL, 0, BN_TRANSFORMER_PREFILL_REQUEST_LAST_LOGITS);
    assert(!prefill_entry.uses_decode_fallback);
    assert(prefill_entry.path == BN_TRANSFORMER_PREFILL_ENTRY_BATCH);
    c.dim = 2049;
    assert(!bn_transformer_moe_uses_all_active_two_route(&c, c.dim));
    assert(!bn_transformer_moe_uses_configured_all_active_two_route(&c));
    assert(!bn_transformer_prefill_shared_all_active_two_decode_fallback_policy(
                &c, 0).enabled);
    c.n_experts = 4;
    c.n_experts_active = 2;
    assert(!bn_transformer_moe_uses_all_active_two_expert_set(&c));
    assert(bn_transformer_moe_uses_grouped_route(&c));
    memset(&c, 0, sizeof(c));
    assert(!bn_transformer_moe_uses_all_active_two_route(NULL, 0));
    assert(!bn_transformer_moe_uses_all_active_two_expert_set(NULL));
    assert(!bn_transformer_moe_uses_configured_all_active_two_route(NULL));
    assert(!bn_transformer_moe_uses_grouped_route(NULL));
    assert(!bn_transformer_moe_normalizes_topk_route_weights(NULL));
    assert(!bn_transformer_moe_supports_resident_routed_ffn_shape(
        NULL, &resident_map, 0));
    assert(!bn_transformer_moe_supports_resident_routed_ffn_layout(
        NULL, &resident_map));
    assert(!bn_transformer_moe_supports_resident_routed_ffn_layout(
        &c, NULL));
    assert(bn_transformer_prefill_float_kquant_fallback_task_flags(0) == 0);
    assert(bn_transformer_prefill_float_kquant_fallback_task_flags(1) ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);
    c.policy_flags = BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK;
    BnTransformerPrefillFloatKQuantFallbackPolicy prefill_kquant_fallback =
        bn_transformer_prefill_float_kquant_fallback_policy(&c);
    assert(prefill_kquant_fallback.enabled ==
           bn_transformer_cpu_prefill_uses_float_kquant_fallback(&c));
    assert(prefill_kquant_fallback.task_flags ==
           bn_transformer_prefill_float_kquant_fallback_task_flags(
               prefill_kquant_fallback.enabled));
    c.policy_flags = 0;

    BnTransformerPrefillQuantMatmulDispatchPolicy matmul_dispatch =
        bn_transformer_prefill_quant_matmul_dispatch_policy(
            0, 4, 0, 0, 0, 0, 0);
    assert(!matmul_dispatch.valid);
    matmul_dispatch = bn_transformer_prefill_quant_matmul_dispatch_policy(
        2, 4, 0, 0, 0, 1, 1);
    assert(matmul_dispatch.valid);
    assert(matmul_dispatch.path ==
           BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_FLOAT_KQUANT_FALLBACK);
    matmul_dispatch = bn_transformer_prefill_quant_matmul_dispatch_policy(
        2, 4, 0, 0, 0, 1, 0);
    assert(matmul_dispatch.valid);
    assert(matmul_dispatch.path ==
           BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_PREPARED_MULTI);
    matmul_dispatch = bn_transformer_prefill_quant_matmul_dispatch_policy(
        5, 4, 0, 0, 0, 1, 1);
    assert(matmul_dispatch.valid);
    assert(matmul_dispatch.path ==
           BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_SINGLE);
    matmul_dispatch = bn_transformer_prefill_quant_matmul_dispatch_policy(
        3, 4, 1, 1, 1, 1, 1);
    assert(matmul_dispatch.valid);
    assert(matmul_dispatch.path ==
           BN_TRANSFORMER_PREFILL_QUANT_MATMUL_GPU_BATCH);
    matmul_dispatch = bn_transformer_prefill_quant_matmul_dispatch_policy(
        3, 4, 1, 1, 0, 1, 1);
    assert(matmul_dispatch.valid);
    assert(matmul_dispatch.path ==
           BN_TRANSFORMER_PREFILL_QUANT_MATMUL_GPU_SINGLE);

    BnTransformerPrefillQuantMatmulResourcePolicy matmul_resources =
        bn_transformer_prefill_quant_matmul_resource_policy(
            test_cpu_policy(), NULL, NULL, 1, 4);
    assert(!matmul_resources.valid);
    BnQWeight q4k_weight = {0};
    q4k_weight.type = BN_GGUF_TENSOR_Q4_K;
    BnQWeight f32_weight = {0};
    f32_weight.type = BN_GGUF_TENSOR_F32;
    const BnQWeight *prefill_weights[2] = {
        &q4k_weight,
        &q4k_weight
    };
    matmul_resources =
        bn_transformer_prefill_quant_matmul_resource_policy(
            test_cpu_policy(), NULL, prefill_weights, 2, 4);
    assert(matmul_resources.valid);
    assert(matmul_resources.n_tasks == 2);
    assert(matmul_resources.prepared[0] == NULL);
    assert(matmul_resources.gpu_buffers[0] == NULL);
    assert(!matmul_resources.all_gpu_buffers_available);

    BnBackendModel *prefill_backend = bn_backend_model_create();
    assert(prefill_backend);
    BnPreparedWeight prefill_prepared = {0};
    prefill_prepared.kind = BN_PREPARED_WEIGHT_Q4_K_SCALES;
    int q4k_gpu_buf;
    assert(bn_backend_model_register_qweight(
               prefill_backend, &q4k_weight, &q4k_gpu_buf) == 0);
    assert(bn_backend_model_register_prepared_qweight(
               prefill_backend, &q4k_weight, &prefill_prepared) == 0);
    matmul_resources =
        bn_transformer_prefill_quant_matmul_resource_policy(
            test_cpu_policy(), prefill_backend, prefill_weights, 2, 4);
    assert(matmul_resources.valid);
    assert(matmul_resources.n_tasks == 2);
    assert(matmul_resources.prepared[0] != NULL);
    assert(matmul_resources.prepared[0]->kind == prefill_prepared.kind);
    assert(matmul_resources.gpu_buffers[0] == &q4k_gpu_buf);
    assert(matmul_resources.all_gpu_buffers_available);
    assert(bn_transformer_prefill_qweight_gpu_buffer_policy(
               prefill_backend, &q4k_weight) == &q4k_gpu_buf);
    int prefill_role_buf;
    BnLayerWeights prefill_role_lw = {0};
    prefill_role_lw.attn.wo = q4k_weight;
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_role_lw.attn.wo,
               &q4k_gpu_buf) == 0);
    BnTransformerGPUAttentionResources prefill_role_attn_res =
        bn_transformer_gpu_resolve_attention_resources(NULL, prefill_backend,
                                                       &prefill_role_lw, 0);
    assert(prefill_role_attn_res.wo_prefill == NULL);
    assert(prefill_role_attn_res.wo == &q4k_gpu_buf);
    assert(bn_backend_model_register_handle(
               prefill_backend, 0, BN_BACKEND_HANDLE_WO_PREFILL,
               &prefill_role_buf) == 0);
    prefill_role_attn_res =
        bn_transformer_gpu_resolve_attention_resources(NULL, prefill_backend,
                                                       &prefill_role_lw, 0);
    assert(prefill_role_attn_res.wo_prefill == &prefill_role_buf);
    assert(prefill_role_attn_res.wo == &q4k_gpu_buf);
    BnLayerWeights prefill_attn_lw = {0};
    BnQWeight prefill_wv_weight = {0};
    BnQWeight prefill_wo_weight = {0};
    BnQWeight prefill_wqkv_weight = {0};
    prefill_wv_weight.type = BN_GGUF_TENSOR_Q6_K;
    prefill_wo_weight.type = BN_GGUF_TENSOR_Q4_K;
    prefill_wqkv_weight.type = BN_GGUF_TENSOR_Q5_K;
    prefill_attn_lw.attn.wv = prefill_wv_weight;
    prefill_attn_lw.attn.wo = prefill_wo_weight;
    prefill_attn_lw.ssm.wqkv = prefill_wqkv_weight;
    int prefill_qk_handle;
    int prefill_wv_handle;
    int prefill_wv_buf;
    int prefill_wo_buf;
    int prefill_wqkv_buf;
    assert(bn_backend_model_register_handle(
               prefill_backend, 1, BN_BACKEND_HANDLE_QK_STACKED,
               &prefill_qk_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 1, BN_BACKEND_HANDLE_WV_PREFILL,
               &prefill_wv_handle) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_attn_lw.attn.wv,
               &prefill_wv_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_attn_lw.attn.wo,
               &prefill_wo_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_attn_lw.ssm.wqkv,
               &prefill_wqkv_buf) == 0);
    BnTransformerGPUAttentionResources prefill_attn_resolved =
        bn_transformer_gpu_resolve_attention_resources(NULL, prefill_backend,
                                                       &prefill_attn_lw, 1);
    assert(prefill_attn_resolved.qk_stacked == &prefill_qk_handle);
    assert(prefill_attn_resolved.wv_prefill == &prefill_wv_handle);
    assert(prefill_attn_resolved.wv == &prefill_wv_buf);
    assert(prefill_attn_resolved.wo_prefill == NULL);
    assert(prefill_attn_resolved.wo == &prefill_wo_buf);
    BnTransformerPrefillAttentionProjectionTypes prefill_attn_resource_types =
        {0};
    prefill_attn_resource_types.q_type = BN_GGUF_TENSOR_Q4_K;
    prefill_attn_resource_types.q_rows = 64;
    prefill_attn_resource_types.k_rows = 16;
    prefill_attn_resource_types.v_type = BN_GGUF_TENSOR_Q6_K;
    prefill_attn_resource_types.v_rows = 16;
    BnTransformerPrefillSSMProjectionTypes prefill_ssm_resource_types = {0};
    prefill_ssm_resource_types.qkv_type = BN_GGUF_TENSOR_Q5_K;
    prefill_ssm_resource_types.qkv_rows = 80;
    BnTransformerPrefillAttentionGPUResourcePolicy prefill_attn_resources =
        bn_transformer_prefill_attention_gpu_resource_policy(
            prefill_backend, 1, &prefill_attn_lw, 0,
            prefill_attn_resource_types, prefill_ssm_resource_types);
    assert(prefill_attn_resources.valid);
    assert(!prefill_attn_resources.uses_packed_qkv);
    assert(prefill_attn_resources.qk == &prefill_qk_handle);
    assert(prefill_attn_resources.wv == &prefill_wv_handle);
    assert(prefill_attn_resources.wo == &prefill_wo_buf);
    assert(prefill_attn_resources.qk_rows == 80);
    assert(prefill_attn_resources.qk_type == BN_GGUF_TENSOR_Q4_K);
    assert(prefill_attn_resources.wv_rows == 16);
    assert(prefill_attn_resources.wv_type == BN_GGUF_TENSOR_Q6_K);
    BnTransformerPrefillStackedAttentionGPUResourcePolicy
        stacked_attn_resources =
            bn_transformer_prefill_stacked_attention_gpu_resource_policy(
                prefill_backend, 1, prefill_attn_resource_types);
    assert(stacked_attn_resources.qk_valid);
    assert(stacked_attn_resources.qkv_valid);
    assert(stacked_attn_resources.qk == &prefill_qk_handle);
    assert(stacked_attn_resources.wv == &prefill_wv_handle);
    assert(stacked_attn_resources.qk_rows == 80);
    assert(stacked_attn_resources.qk_type == BN_GGUF_TENSOR_Q4_K);
    assert(stacked_attn_resources.wv_rows == 16);
    assert(stacked_attn_resources.wv_type == BN_GGUF_TENSOR_Q6_K);
    int prefill_raw_attn_norm_handle;
    int prefill_raw_q_norm_handle;
    int prefill_raw_k_norm_handle;
    assert(bn_backend_model_register_handle(
               prefill_backend, 1, BN_BACKEND_HANDLE_ATTN_NORM,
               &prefill_raw_attn_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 1, BN_BACKEND_HANDLE_Q_NORM,
               &prefill_raw_q_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 1, BN_BACKEND_HANDLE_K_NORM,
               &prefill_raw_k_norm_handle) == 0);
    BnTransformerGPULayerValidationResources raw_layer_res =
        bn_transformer_gpu_resolve_layer_validation_resources(prefill_backend,
                                                              1);
    assert(raw_layer_res.attn_norm == &prefill_raw_attn_norm_handle);
    assert(raw_layer_res.q_norm == &prefill_raw_q_norm_handle);
    assert(raw_layer_res.k_norm == &prefill_raw_k_norm_handle);
    BnTransformerPrefillRawAttentionGPUResourcePolicy raw_attn_resources =
        bn_transformer_prefill_raw_attention_gpu_resource_policy(
            prefill_backend, 1, &prefill_attn_lw,
            prefill_attn_resource_types);
    assert(raw_attn_resources.valid);
    assert(raw_attn_resources.qk == &prefill_qk_handle);
    assert(raw_attn_resources.wv == &prefill_wv_handle);
    assert(raw_attn_resources.wo == &prefill_wo_buf);
    assert(raw_attn_resources.attn_norm == &prefill_raw_attn_norm_handle);
    assert(raw_attn_resources.q_norm == &prefill_raw_q_norm_handle);
    assert(raw_attn_resources.k_norm == &prefill_raw_k_norm_handle);
    assert(raw_attn_resources.qk_rows == 80);
    assert(raw_attn_resources.qk_type == BN_GGUF_TENSOR_Q4_K);
    assert(raw_attn_resources.wv_rows == 16);
    assert(raw_attn_resources.wv_type == BN_GGUF_TENSOR_Q6_K);
    assert(bn_backend_model_register_handle(
               prefill_backend, 2, BN_BACKEND_HANDLE_QK_STACKED,
               &prefill_qk_handle) == 0);
    prefill_attn_resources =
        bn_transformer_prefill_attention_gpu_resource_policy(
            prefill_backend, 2, &prefill_attn_lw, 0,
            prefill_attn_resource_types, prefill_ssm_resource_types);
    assert(prefill_attn_resources.valid);
    assert(prefill_attn_resources.wv == &prefill_wv_buf);
    stacked_attn_resources =
        bn_transformer_prefill_stacked_attention_gpu_resource_policy(
            prefill_backend, 2, prefill_attn_resource_types);
    assert(stacked_attn_resources.qk_valid);
    assert(!stacked_attn_resources.qkv_valid);
    assert(stacked_attn_resources.qk == &prefill_qk_handle);
    assert(stacked_attn_resources.wv == NULL);
    raw_attn_resources =
        bn_transformer_prefill_raw_attention_gpu_resource_policy(
            prefill_backend, 2, &prefill_attn_lw,
            prefill_attn_resource_types);
    assert(raw_attn_resources.valid);
    assert(raw_attn_resources.wv == &prefill_wv_buf);
    assert(raw_attn_resources.attn_norm == NULL);
    prefill_attn_resources =
        bn_transformer_prefill_attention_gpu_resource_policy(
            prefill_backend, 2, &prefill_attn_lw, 1,
            prefill_attn_resource_types, prefill_ssm_resource_types);
    assert(prefill_attn_resources.valid);
    assert(prefill_attn_resources.uses_packed_qkv);
    assert(prefill_attn_resources.qk == &prefill_wqkv_buf);
    assert(prefill_attn_resources.wv == NULL);
    assert(prefill_attn_resources.qk_rows == 80);
    assert(prefill_attn_resources.qk_type == BN_GGUF_TENSOR_Q5_K);
    assert(prefill_attn_resources.wv_rows == 0);
    assert(prefill_attn_resources.wv_type == BN_GGUF_TENSOR_Q5_K);
    prefill_attn_resources =
        bn_transformer_prefill_attention_gpu_resource_policy(
            prefill_backend, 2, NULL, 0,
            prefill_attn_resource_types, prefill_ssm_resource_types);
    assert(!prefill_attn_resources.valid);
    raw_attn_resources =
        bn_transformer_prefill_raw_attention_gpu_resource_policy(
            prefill_backend, 2, NULL, prefill_attn_resource_types);
    assert(!raw_attn_resources.valid);

    BnQWeight prefill_dense_gate = {0};
    BnQWeight prefill_dense_up = {0};
    BnQWeight prefill_dense_down = {0};
    prefill_dense_gate.type = BN_GGUF_TENSOR_Q4_K;
    prefill_dense_up.type = BN_GGUF_TENSOR_Q4_K;
    prefill_dense_down.type = BN_GGUF_TENSOR_Q6_K;
    prefill_attn_lw.ffn.ffn_gate = prefill_dense_gate;
    prefill_attn_lw.ffn.ffn_up = prefill_dense_up;
    prefill_attn_lw.ffn.ffn_down = prefill_dense_down;
    int prefill_dense_gate_buf;
    int prefill_dense_up_buf;
    int prefill_dense_down_buf;
    int prefill_dense_gateup_handle;
    int prefill_dense_qk_handle;
    int prefill_dense_attn_norm_handle;
    int prefill_dense_ffn_norm_handle;
    int prefill_dense_q_norm_handle;
    int prefill_dense_k_norm_handle;
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_attn_lw.ffn.ffn_gate,
               &prefill_dense_gate_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_attn_lw.ffn.ffn_up,
               &prefill_dense_up_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_attn_lw.ffn.ffn_down,
               &prefill_dense_down_buf) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 7, BN_BACKEND_HANDLE_QK_STACKED,
               &prefill_dense_qk_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 7, BN_BACKEND_HANDLE_GATEUP_STACKED,
               &prefill_dense_gateup_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 7, BN_BACKEND_HANDLE_ATTN_NORM,
               &prefill_dense_attn_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 7, BN_BACKEND_HANDLE_FFN_NORM,
               &prefill_dense_ffn_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 7, BN_BACKEND_HANDLE_Q_NORM,
               &prefill_dense_q_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 7, BN_BACKEND_HANDLE_K_NORM,
               &prefill_dense_k_norm_handle) == 0);
    BnTransformerGPULayerValidationResources dense_layer_validation =
        bn_transformer_gpu_resolve_layer_validation_resources(prefill_backend,
                                                              7);
    assert(dense_layer_validation.attn_norm ==
           &prefill_dense_attn_norm_handle);
    assert(dense_layer_validation.ffn_norm == &prefill_dense_ffn_norm_handle);
    assert(dense_layer_validation.q_norm == &prefill_dense_q_norm_handle);
    assert(dense_layer_validation.k_norm == &prefill_dense_k_norm_handle);
    BnTransformerPrefillFFNProjectionTypes prefill_dense_ffn_types = {0};
    prefill_dense_ffn_types.gate_type = BN_GGUF_TENSOR_Q4_K;
    prefill_dense_ffn_types.up_type = BN_GGUF_TENSOR_Q4_K;
    prefill_dense_ffn_types.down_type = BN_GGUF_TENSOR_Q6_K;
    BnTransformerPrefillDenseLayerGPUResourcePolicy dense_layer_resources =
        bn_transformer_prefill_dense_layer_gpu_resource_policy(
            prefill_backend, 7, &prefill_attn_lw, 0,
            prefill_attn_resource_types, prefill_ssm_resource_types,
            prefill_dense_ffn_types);
    assert(dense_layer_resources.valid);
    assert(dense_layer_resources.qk == &prefill_dense_qk_handle);
    assert(dense_layer_resources.wv == &prefill_wv_buf);
    assert(dense_layer_resources.wo == &prefill_wo_buf);
    assert(dense_layer_resources.gate == &prefill_dense_gateup_handle);
    assert(dense_layer_resources.up == NULL);
    assert(dense_layer_resources.down == &prefill_dense_down_buf);
    assert(dense_layer_resources.attn_norm ==
           &prefill_dense_attn_norm_handle);
    assert(dense_layer_resources.ffn_norm == &prefill_dense_ffn_norm_handle);
    assert(dense_layer_resources.q_norm == &prefill_dense_q_norm_handle);
    assert(dense_layer_resources.k_norm == &prefill_dense_k_norm_handle);
    assert(dense_layer_resources.q_bias == NULL);
    assert(dense_layer_resources.k_bias == NULL);
    assert(dense_layer_resources.v_bias == NULL);
    assert(dense_layer_resources.qk_rows == 80);
    assert(dense_layer_resources.qk_type == BN_GGUF_TENSOR_Q4_K);
    assert(dense_layer_resources.wv_rows == 16);
    assert(dense_layer_resources.wv_type == BN_GGUF_TENSOR_Q6_K);

    int prefill_dense_q_bias_handle;
    int prefill_dense_k_bias_handle;
    int prefill_dense_v_bias_handle;
    float prefill_dense_q_bias_weight;
    float prefill_dense_k_bias_weight;
    float prefill_dense_v_bias_weight;
    prefill_attn_lw.attn.q_bias = &prefill_dense_q_bias_weight;
    prefill_attn_lw.attn.k_bias = &prefill_dense_k_bias_weight;
    prefill_attn_lw.attn.v_bias = &prefill_dense_v_bias_weight;
    dense_layer_resources =
        bn_transformer_prefill_dense_layer_gpu_resource_policy(
            prefill_backend, 7, &prefill_attn_lw, 0,
            prefill_attn_resource_types, prefill_ssm_resource_types,
            prefill_dense_ffn_types);
    assert(!dense_layer_resources.valid);
    assert(bn_backend_model_register_handle(
               prefill_backend, 7, BN_BACKEND_HANDLE_Q_BIAS,
               &prefill_dense_q_bias_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 7, BN_BACKEND_HANDLE_K_BIAS,
               &prefill_dense_k_bias_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 7, BN_BACKEND_HANDLE_V_BIAS,
               &prefill_dense_v_bias_handle) == 0);
    BnTransformerGPUQKVResources dense_qkv_resources =
        bn_transformer_gpu_resolve_qkv_resources(NULL, prefill_backend,
                                                 &prefill_attn_lw, 7);
    assert(dense_qkv_resources.q_bias == &prefill_dense_q_bias_handle);
    assert(dense_qkv_resources.k_bias == &prefill_dense_k_bias_handle);
    assert(dense_qkv_resources.v_bias == &prefill_dense_v_bias_handle);
    dense_layer_resources =
        bn_transformer_prefill_dense_layer_gpu_resource_policy(
            prefill_backend, 7, &prefill_attn_lw, 0,
            prefill_attn_resource_types, prefill_ssm_resource_types,
            prefill_dense_ffn_types);
    assert(dense_layer_resources.valid);
    assert(dense_layer_resources.q_bias == &prefill_dense_q_bias_handle);
    assert(dense_layer_resources.k_bias == &prefill_dense_k_bias_handle);
    assert(dense_layer_resources.v_bias == &prefill_dense_v_bias_handle);
    dense_layer_resources =
        bn_transformer_prefill_dense_layer_gpu_resource_policy(
            prefill_backend, 8, &prefill_attn_lw, 0,
            prefill_attn_resource_types, prefill_ssm_resource_types,
            prefill_dense_ffn_types);
    assert(!dense_layer_resources.valid);
    dense_layer_resources =
        bn_transformer_prefill_dense_layer_gpu_resource_policy(
            prefill_backend, 7, NULL, 0, prefill_attn_resource_types,
            prefill_ssm_resource_types, prefill_dense_ffn_types);
    assert(!dense_layer_resources.valid);
    prefill_attn_lw.attn.q_bias = NULL;
    prefill_attn_lw.attn.k_bias = NULL;
    prefill_attn_lw.attn.v_bias = NULL;

    int prefill_moe_qk_handle;
    int prefill_moe_wv_handle;
    int prefill_moe_router_handle;
    int prefill_moe_gate_all_handle;
    int prefill_moe_up_all_handle;
    int prefill_moe_down_all_handle;
    int prefill_moe_attn_norm_handle;
    int prefill_moe_ffn_norm_handle;
    int prefill_moe_q_bias_handle;
    int prefill_moe_k_bias_handle;
    int prefill_moe_v_bias_handle;
    float prefill_moe_q_bias_weight;
    float prefill_moe_k_bias_weight;
    float prefill_moe_v_bias_weight;
    prefill_attn_lw.attn.q_bias = &prefill_moe_q_bias_weight;
    prefill_attn_lw.attn.k_bias = &prefill_moe_k_bias_weight;
    prefill_attn_lw.attn.v_bias = &prefill_moe_v_bias_weight;
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_QK_STACKED,
               &prefill_moe_qk_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_WV_PREFILL,
               &prefill_moe_wv_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_MOE_ROUTER,
               &prefill_moe_router_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_MOE_GATE_ALL,
               &prefill_moe_gate_all_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_MOE_UP_ALL,
               &prefill_moe_up_all_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_MOE_DOWN_ALL,
               &prefill_moe_down_all_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_ATTN_NORM,
               &prefill_moe_attn_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_FFN_NORM,
               &prefill_moe_ffn_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_Q_BIAS,
               &prefill_moe_q_bias_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_K_BIAS,
               &prefill_moe_k_bias_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 3, BN_BACKEND_HANDLE_V_BIAS,
               &prefill_moe_v_bias_handle) == 0);
    BnTransformerPrefillMoELayerGPUResourcePolicy prefill_moe_resources =
        bn_transformer_prefill_moe_layer_gpu_resource_policy(
            prefill_backend, &c, 3, &prefill_attn_lw,
            prefill_attn_resource_types);
    BnTransformerGPUMoEPrefillFFNResources prefill_moe_ffn_resolved =
        bn_transformer_gpu_resolve_moe_prefill_ffn_resources(prefill_backend,
                                                             3);
    assert(prefill_moe_ffn_resolved.resident_valid);
    assert(prefill_moe_ffn_resolved.router == &prefill_moe_router_handle);
    assert(prefill_moe_ffn_resolved.gate_all ==
           &prefill_moe_gate_all_handle);
    assert(prefill_moe_ffn_resolved.up_all == &prefill_moe_up_all_handle);
    assert(prefill_moe_ffn_resolved.down_all ==
           &prefill_moe_down_all_handle);
    assert(prefill_moe_ffn_resolved.ffn_norm ==
           &prefill_moe_ffn_norm_handle);
    assert(prefill_moe_resources.valid);
    assert(prefill_moe_resources.qk == &prefill_moe_qk_handle);
    assert(prefill_moe_resources.wv == &prefill_moe_wv_handle);
    assert(prefill_moe_resources.wo == &prefill_wo_buf);
    assert(prefill_moe_resources.router == &prefill_moe_router_handle);
    assert(prefill_moe_resources.gate_all == &prefill_moe_gate_all_handle);
    assert(prefill_moe_resources.up_all == &prefill_moe_up_all_handle);
    assert(prefill_moe_resources.down_all == &prefill_moe_down_all_handle);
    assert(prefill_moe_resources.attn_norm == &prefill_moe_attn_norm_handle);
    assert(prefill_moe_resources.ffn_norm == &prefill_moe_ffn_norm_handle);
    assert(prefill_moe_resources.q_bias == &prefill_moe_q_bias_handle);
    assert(prefill_moe_resources.k_bias == &prefill_moe_k_bias_handle);
    assert(prefill_moe_resources.v_bias == &prefill_moe_v_bias_handle);
    assert(prefill_moe_resources.qk_rows == 80);
    assert(prefill_moe_resources.qk_type == BN_GGUF_TENSOR_Q4_K);
    assert(prefill_moe_resources.wv_rows == 16);
    assert(prefill_moe_resources.wv_type == BN_GGUF_TENSOR_Q6_K);

    BnTransformerPrefillMoEFFNGPUResourcePolicy prefill_moe_ffn_resources =
        bn_transformer_prefill_moe_ffn_gpu_resource_policy(
            prefill_backend, &c, 3, &prefill_attn_lw);
    assert(prefill_moe_ffn_resources.valid);
    assert(prefill_moe_ffn_resources.router == &prefill_moe_router_handle);
    assert(prefill_moe_ffn_resources.gate_all ==
           &prefill_moe_gate_all_handle);
    assert(prefill_moe_ffn_resources.up_all == &prefill_moe_up_all_handle);
    assert(prefill_moe_ffn_resources.down_all ==
           &prefill_moe_down_all_handle);
    assert(prefill_moe_ffn_resources.ffn_norm ==
           &prefill_moe_ffn_norm_handle);
    assert(prefill_moe_ffn_resources.shared_gate == NULL);
    assert(prefill_moe_ffn_resources.shared_up == NULL);
    assert(prefill_moe_ffn_resources.shared_down == NULL);
    assert(prefill_moe_ffn_resources.shared_gate_weight == NULL);

    assert(bn_backend_model_register_handle(
               prefill_backend, 4, BN_BACKEND_HANDLE_QK_STACKED,
               &prefill_moe_qk_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 4, BN_BACKEND_HANDLE_MOE_ROUTER,
               &prefill_moe_router_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 4, BN_BACKEND_HANDLE_MOE_GATE_ALL,
               &prefill_moe_gate_all_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 4, BN_BACKEND_HANDLE_MOE_UP_ALL,
               &prefill_moe_up_all_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 4, BN_BACKEND_HANDLE_MOE_DOWN_ALL,
               &prefill_moe_down_all_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 4, BN_BACKEND_HANDLE_ATTN_NORM,
               &prefill_moe_attn_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 4, BN_BACKEND_HANDLE_FFN_NORM,
               &prefill_moe_ffn_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 4, BN_BACKEND_HANDLE_Q_BIAS,
               &prefill_moe_q_bias_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 4, BN_BACKEND_HANDLE_K_BIAS,
               &prefill_moe_k_bias_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 4, BN_BACKEND_HANDLE_V_BIAS,
               &prefill_moe_v_bias_handle) == 0);
    prefill_moe_resources =
        bn_transformer_prefill_moe_layer_gpu_resource_policy(
            prefill_backend, &c, 4, &prefill_attn_lw,
            prefill_attn_resource_types);
    assert(prefill_moe_resources.valid);
    assert(prefill_moe_resources.wv == &prefill_wv_buf);

    assert(bn_backend_model_register_handle(
               prefill_backend, 5, BN_BACKEND_HANDLE_QK_STACKED,
               &prefill_moe_qk_handle) == 0);
    prefill_moe_ffn_resolved =
        bn_transformer_gpu_resolve_moe_prefill_ffn_resources(prefill_backend,
                                                             5);
    assert(!prefill_moe_ffn_resolved.resident_valid);
    prefill_moe_resources =
        bn_transformer_prefill_moe_layer_gpu_resource_policy(
            prefill_backend, &c, 5, &prefill_attn_lw,
            prefill_attn_resource_types);
    assert(!prefill_moe_resources.valid);
    prefill_moe_ffn_resources =
        bn_transformer_prefill_moe_ffn_gpu_resource_policy(
            prefill_backend, &c, 5, &prefill_attn_lw);
    assert(!prefill_moe_ffn_resources.valid);
    prefill_moe_ffn_resources =
        bn_transformer_prefill_moe_ffn_gpu_resource_policy(
            prefill_backend, &c, 3, NULL);
    assert(!prefill_moe_ffn_resources.valid);
    prefill_moe_resources =
        bn_transformer_prefill_moe_layer_gpu_resource_policy(
            prefill_backend, &c, 3, NULL, prefill_attn_resource_types);
    assert(!prefill_moe_resources.valid);
    prefill_attn_lw.attn.q_bias = NULL;
    prefill_attn_lw.attn.k_bias = NULL;
    prefill_attn_lw.attn.v_bias = NULL;

    BnLayerWeights prefill_ffn_lw;
    memset(&prefill_ffn_lw, 0, sizeof(prefill_ffn_lw));
    BnLayerWeights prefill_ffn_fallback_lw;
    memset(&prefill_ffn_fallback_lw, 0, sizeof(prefill_ffn_fallback_lw));
    BnQWeight prefill_ffn_down_fallback = {0};
    prefill_ffn_lw.ffn.ffn_gate.type = BN_GGUF_TENSOR_Q4_K;
    prefill_ffn_lw.ffn.ffn_up.type = BN_GGUF_TENSOR_Q4_K;
    prefill_ffn_lw.ffn.ffn_down.type = BN_GGUF_TENSOR_Q6_K;
    prefill_ffn_down_fallback.type = BN_GGUF_TENSOR_Q6_K;
    prefill_ffn_fallback_lw.ffn.ffn_gate.type = BN_GGUF_TENSOR_Q4_K;
    prefill_ffn_fallback_lw.ffn.ffn_up.type = BN_GGUF_TENSOR_Q4_K;
    prefill_ffn_fallback_lw.ffn.ffn_down = prefill_ffn_down_fallback;
    int prefill_gate_buf;
    int prefill_up_buf;
    int prefill_down_buf;
    int prefill_gateup_buf;
    int prefill_down_handle;
    int prefill_ffn_norm_handle;
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ffn_lw.ffn.ffn_gate,
               &prefill_gate_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ffn_lw.ffn.ffn_up,
               &prefill_up_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ffn_lw.ffn.ffn_down,
               &prefill_down_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ffn_fallback_lw.ffn.ffn_gate,
               &prefill_gate_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ffn_fallback_lw.ffn.ffn_up,
               &prefill_up_buf) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 0, BN_BACKEND_HANDLE_GATEUP_STACKED,
               &prefill_gateup_buf) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 0, BN_BACKEND_HANDLE_FFN_DOWN_PREFILL,
               &prefill_down_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 0, BN_BACKEND_HANDLE_FFN_NORM,
               &prefill_ffn_norm_handle) == 0);
    BnTransformerGPUDenseFFNResources prefill_ffn_resolved =
        bn_transformer_gpu_resolve_dense_ffn_resources(NULL, prefill_backend,
                                                       &prefill_ffn_lw, 0);
    assert(prefill_ffn_resolved.gateup_stacked == &prefill_gateup_buf);
    assert(prefill_ffn_resolved.ffn_gate == &prefill_gate_buf);
    assert(prefill_ffn_resolved.ffn_up == &prefill_up_buf);
    assert(prefill_ffn_resolved.ffn_down == &prefill_down_buf);
    assert(prefill_ffn_resolved.ffn_down_prefill == &prefill_down_handle);
    BnTransformerGPULayerValidationResources prefill_ffn_layer_res =
        bn_transformer_gpu_resolve_layer_validation_resources(prefill_backend,
                                                              0);
    assert(prefill_ffn_layer_res.ffn_norm == &prefill_ffn_norm_handle);
    BnTransformerPrefillFFNProjectionTypes prefill_resource_types = {0};
    prefill_resource_types.gate_type = BN_GGUF_TENSOR_Q4_K;
    prefill_resource_types.up_type = BN_GGUF_TENSOR_Q4_K;
    BnTransformerPrefillDenseFFNGPUResourcePolicy prefill_ffn_resources =
        bn_transformer_prefill_dense_ffn_gpu_resource_policy(
            prefill_backend, 0, &prefill_ffn_lw, prefill_resource_types);
    assert(prefill_ffn_resources.valid);
    assert(prefill_ffn_resources.uses_stacked_gateup);
    assert(prefill_ffn_resources.gate == &prefill_gateup_buf);
    assert(prefill_ffn_resources.up == NULL);
    assert(prefill_ffn_resources.down == &prefill_down_buf);
    assert(prefill_ffn_resources.ffn_norm == &prefill_ffn_norm_handle);
    prefill_resource_types.up_type = BN_GGUF_TENSOR_Q5_K;
    prefill_ffn_resources =
        bn_transformer_prefill_dense_ffn_gpu_resource_policy(
            prefill_backend, 0, &prefill_ffn_lw, prefill_resource_types);
    assert(prefill_ffn_resources.valid);
    assert(!prefill_ffn_resources.uses_stacked_gateup);
    assert(prefill_ffn_resources.gate == &prefill_gate_buf);
    assert(prefill_ffn_resources.up == &prefill_up_buf);
    assert(prefill_ffn_resources.down == &prefill_down_buf);
    assert(prefill_ffn_resources.ffn_norm == &prefill_ffn_norm_handle);
    prefill_ffn_resources =
        bn_transformer_prefill_dense_ffn_gpu_resource_policy(
            prefill_backend, 0, &prefill_ffn_fallback_lw,
            prefill_resource_types);
    assert(prefill_ffn_resources.valid);
    assert(prefill_ffn_resources.down == &prefill_down_handle);
    prefill_ffn_resources =
        bn_transformer_prefill_dense_ffn_gpu_resource_policy(
            prefill_backend, 0, NULL, prefill_resource_types);
    assert(!prefill_ffn_resources.valid);
    prefill_resource_types.up_type = BN_GGUF_TENSOR_Q4_K;

    BnLayerWeights prefill_ssm_lw = {0};
    BnQWeight prefill_ssm_wqkv = {0};
    BnQWeight prefill_ssm_wz = {0};
    BnQWeight prefill_ssm_alpha = {0};
    BnQWeight prefill_ssm_beta = {0};
    BnQWeight prefill_ssm_out = {0};
    prefill_ssm_wqkv.type = BN_GGUF_TENSOR_Q4_K;
    prefill_ssm_wz.type = BN_GGUF_TENSOR_Q5_K;
    prefill_ssm_alpha.type = BN_GGUF_TENSOR_Q6_K;
    prefill_ssm_beta.type = BN_GGUF_TENSOR_Q8_0;
    prefill_ssm_out.type = BN_GGUF_TENSOR_Q4_K;
    prefill_ssm_lw.ssm.wqkv = prefill_ssm_wqkv;
    prefill_ssm_lw.ssm.wz = prefill_ssm_wz;
    prefill_ssm_lw.ssm.ssm_alpha = prefill_ssm_alpha;
    prefill_ssm_lw.ssm.ssm_beta = prefill_ssm_beta;
    prefill_ssm_lw.ssm.ssm_out = prefill_ssm_out;
    prefill_ssm_lw.ffn.ffn_gate.type = BN_GGUF_TENSOR_Q4_K;
    prefill_ssm_lw.ffn.ffn_up.type = BN_GGUF_TENSOR_Q4_K;
    prefill_ssm_lw.ffn.ffn_down.type = BN_GGUF_TENSOR_Q6_K;
    int prefill_ssm_wqkv_buf;
    int prefill_ssm_wz_buf;
    int prefill_ssm_alpha_buf;
    int prefill_ssm_beta_buf;
    int prefill_ssm_out_buf;
    int prefill_ssm_ffn_down_buf;
    int prefill_ssm_qkvz_handle;
    int prefill_ssm_ab_handle;
    int prefill_ssm_attn_norm_handle;
    int prefill_ssm_conv_handle;
    int prefill_ssm_dt_bias_handle;
    int prefill_ssm_a_log_handle;
    int prefill_ssm_norm_handle;
    int prefill_ssm_ffn_norm_handle;
    int prefill_ssm_gateup_handle;
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ssm_lw.ssm.wqkv,
               &prefill_ssm_wqkv_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ssm_lw.ssm.wz,
               &prefill_ssm_wz_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ssm_lw.ssm.ssm_alpha,
               &prefill_ssm_alpha_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ssm_lw.ssm.ssm_beta,
               &prefill_ssm_beta_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ssm_lw.ssm.ssm_out,
               &prefill_ssm_out_buf) == 0);
    assert(bn_backend_model_register_qweight(
               prefill_backend, &prefill_ssm_lw.ffn.ffn_down,
               &prefill_ssm_ffn_down_buf) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 6, BN_BACKEND_HANDLE_SSM_QKVZ_STACKED,
               &prefill_ssm_qkvz_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 6, BN_BACKEND_HANDLE_SSM_AB_STACKED,
               &prefill_ssm_ab_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 6, BN_BACKEND_HANDLE_ATTN_NORM,
               &prefill_ssm_attn_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 6, BN_BACKEND_HANDLE_SSM_CONV1D,
               &prefill_ssm_conv_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 6, BN_BACKEND_HANDLE_SSM_DT_BIAS,
               &prefill_ssm_dt_bias_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 6, BN_BACKEND_HANDLE_SSM_A_LOG,
               &prefill_ssm_a_log_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 6, BN_BACKEND_HANDLE_SSM_NORM,
               &prefill_ssm_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 6, BN_BACKEND_HANDLE_FFN_NORM,
               &prefill_ssm_ffn_norm_handle) == 0);
    assert(bn_backend_model_register_handle(
               prefill_backend, 6, BN_BACKEND_HANDLE_GATEUP_STACKED,
               &prefill_ssm_gateup_handle) == 0);
    BnTransformerGPUSSMResources ssm_resolved =
        bn_transformer_gpu_resolve_ssm_resources(NULL, prefill_backend,
                                                 &prefill_ssm_lw, 6);
    assert(ssm_resolved.ssm_qkvz_stacked == &prefill_ssm_qkvz_handle);
    assert(ssm_resolved.ssm_ab_stacked == &prefill_ssm_ab_handle);
    assert(ssm_resolved.ssm_conv1d == &prefill_ssm_conv_handle);
    assert(ssm_resolved.ssm_dt_bias == &prefill_ssm_dt_bias_handle);
    assert(ssm_resolved.ssm_a_log == &prefill_ssm_a_log_handle);
    assert(ssm_resolved.ssm_norm == &prefill_ssm_norm_handle);
    assert(ssm_resolved.ffn_norm == &prefill_ssm_ffn_norm_handle);
    assert(ssm_resolved.wqkv == &prefill_ssm_wqkv_buf);
    assert(ssm_resolved.wz == &prefill_ssm_wz_buf);
    assert(ssm_resolved.ssm_alpha == &prefill_ssm_alpha_buf);
    assert(ssm_resolved.ssm_beta == &prefill_ssm_beta_buf);
    assert(ssm_resolved.ssm_out == &prefill_ssm_out_buf);
    BnTransformerGPULayerValidationResources ssm_layer_res =
        bn_transformer_gpu_resolve_layer_validation_resources(prefill_backend,
                                                              6);
    assert(ssm_layer_res.attn_norm == &prefill_ssm_attn_norm_handle);
    BnTransformerPrefillSSMGPUResourcePolicy ssm_resources =
        bn_transformer_prefill_ssm_gpu_resource_policy(
            prefill_backend, 6, &prefill_ssm_lw, 1,
            prefill_resource_types);
    assert(ssm_resources.valid);
    assert(ssm_resources.fuses_ffn);
    assert(ssm_resources.wqkv == &prefill_ssm_wqkv_buf);
    assert(ssm_resources.wz == &prefill_ssm_wz_buf);
    assert(ssm_resources.alpha == &prefill_ssm_alpha_buf);
    assert(ssm_resources.beta == &prefill_ssm_beta_buf);
    assert(ssm_resources.qkvz_stacked == &prefill_ssm_qkvz_handle);
    assert(ssm_resources.ab_stacked == &prefill_ssm_ab_handle);
    assert(ssm_resources.out == &prefill_ssm_out_buf);
    assert(ssm_resources.attn_norm == &prefill_ssm_attn_norm_handle);
    assert(ssm_resources.conv1d == &prefill_ssm_conv_handle);
    assert(ssm_resources.dt_bias == &prefill_ssm_dt_bias_handle);
    assert(ssm_resources.a_log == &prefill_ssm_a_log_handle);
    assert(ssm_resources.ssm_norm == &prefill_ssm_norm_handle);
    assert(ssm_resources.gate == &prefill_ssm_gateup_handle);
    assert(ssm_resources.up == NULL);
    assert(ssm_resources.down == &prefill_ssm_ffn_down_buf);
    assert(ssm_resources.ffn_norm == &prefill_ssm_ffn_norm_handle);
    ssm_resources = bn_transformer_prefill_ssm_gpu_resource_policy(
        prefill_backend, 6, &prefill_ssm_lw, 0, prefill_resource_types);
    assert(ssm_resources.valid);
    assert(!ssm_resources.fuses_ffn);
    assert(ssm_resources.gate == NULL);
    assert(ssm_resources.ffn_norm == NULL);
    ssm_resources = bn_transformer_prefill_ssm_gpu_resource_policy(
        prefill_backend, 7, &prefill_ssm_lw, 1, prefill_resource_types);
    assert(!ssm_resources.valid);
    ssm_resources = bn_transformer_prefill_ssm_gpu_resource_policy(
        prefill_backend, 6, NULL, 1, prefill_resource_types);
    assert(!ssm_resources.valid);

    setenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS", "1", 1);
    matmul_resources =
        bn_transformer_prefill_quant_matmul_resource_policy(
            test_cpu_policy(), prefill_backend, prefill_weights, 2, 4);
    assert(matmul_resources.valid);
    assert(matmul_resources.prepared[0] == NULL);
    assert(matmul_resources.gpu_buffers[0] == &q4k_gpu_buf);
    assert(matmul_resources.all_gpu_buffers_available);
    unsetenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS");
    prefill_weights[1] = &f32_weight;
    matmul_resources =
        bn_transformer_prefill_quant_matmul_resource_policy(
            test_cpu_policy(), prefill_backend, prefill_weights, 2, 4);
    assert(matmul_resources.valid);
    assert(!matmul_resources.all_gpu_buffers_available);
    bn_backend_model_free(prefill_backend);

    prefill_weights[1] = &q4k_weight;
    c.policy_flags = BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK;
    matmul_dispatch =
        bn_transformer_prefill_quant_matmul_dispatch_policy_for(
            &c, prefill_weights, 2, 4, 0, 0, 0);
    assert(matmul_dispatch.valid);
    assert(matmul_dispatch.path ==
           (bn_transformer_cpu_prefill_uses_float_kquant_fallback(&c)
                ? BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_FLOAT_KQUANT_FALLBACK
                : BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_PREPARED_MULTI));
    prefill_weights[1] = &f32_weight;
    matmul_dispatch =
        bn_transformer_prefill_quant_matmul_dispatch_policy_for(
            &c, prefill_weights, 2, 4, 0, 0, 0);
    assert(matmul_dispatch.valid);
    assert(matmul_dispatch.path ==
           BN_TRANSFORMER_PREFILL_QUANT_MATMUL_CPU_PREPARED_MULTI);
    prefill_weights[1] = &q4k_weight;
    matmul_dispatch =
        bn_transformer_prefill_quant_matmul_dispatch_policy_for(
            &c, prefill_weights, 2, 4, 1, 1, 1);
    assert(matmul_dispatch.valid);
    assert(matmul_dispatch.path ==
           BN_TRANSFORMER_PREFILL_QUANT_MATMUL_GPU_BATCH);
    c.policy_flags = 0;

    BnMoEExpertMap expert_map;
    memset(&expert_map, 0, sizeof(expert_map));
    expert_map.gate_type = BN_GGUF_TENSOR_Q4_K;
    expert_map.up_type = BN_GGUF_TENSOR_Q4_K;
    expert_map.gate_rows = 32;
    expert_map.up_rows = 32;
    expert_map.gate_cols = 64;
    expert_map.up_cols = 64;
    assert(bn_transformer_moe_supports_gateup_split_layout(&expert_map));
    gpu.caps = BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT;
    assert(bn_transformer_gpu_moe_gateup_split_layout_policy(
               &expert_map).supported);
    assert(bn_transformer_gpu_moe_gateup_split_supported(
        &gpu, &expert_map, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_moe_gateup_split_supported(
        NULL, &expert_map, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_moe_gateup_split_supported(
        &gpu, NULL, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_moe_gateup_split_supported(
        &gpu, &expert_map, BN_GPU_CODE_Q5K_MATVEC_SPLIT));
    expert_map.up_type = BN_GGUF_TENSOR_Q5_K;
    assert(!bn_transformer_gpu_moe_gateup_split_supported(
        &gpu, &expert_map, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    expert_map.up_type = BN_GGUF_TENSOR_Q4_K;
    expert_map.up_rows = 16;
    assert(!bn_transformer_moe_supports_gateup_split_layout(&expert_map));
    assert(!bn_transformer_gpu_moe_gateup_split_layout_policy(
                &expert_map).supported);
    assert(!bn_transformer_gpu_moe_gateup_split_supported(
        &gpu, &expert_map, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    expert_map.up_rows = 32;
    expert_map.up_cols = 32;
    assert(!bn_transformer_moe_supports_gateup_split_layout(&expert_map));
    assert(!bn_transformer_gpu_moe_gateup_split_layout_policy(
                &expert_map).supported);
    assert(!bn_transformer_gpu_moe_gateup_split_supported(
        &gpu, &expert_map, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    expert_map.up_cols = 64;
    gpu.caps = 0;
    assert(!bn_transformer_gpu_moe_gateup_split_supported(
        &gpu, &expert_map, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_moe_gateup_split_enabled(&gpu, 1));
    setenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy, "BN_CUDA_DISABLE_MOE_GATEUP_SPLIT",
               "1", 1) == 0);
    assert(!bn_transformer_gpu_moe_gateup_split_enabled(&gpu, 1));
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_CUDA_DISABLE_MOE_GATEUP_SPLIT");
    gpu.kind = BN_GPU_BACKEND_WEBGPU;
    assert(!bn_transformer_gpu_moe_gateup_split_enabled(&gpu, 1));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(!bn_transformer_gpu_moe_gateup_split_enabled(&gpu, 0));

    BnQWeight gate_w;
    BnQWeight up_w;
    memset(&gate_w, 0, sizeof(gate_w));
    memset(&up_w, 0, sizeof(up_w));
    gate_w.type = BN_GGUF_TENSOR_Q4_K;
    gate_w.rows = 32;
    gate_w.cols = 64;
    up_w.type = BN_GGUF_TENSOR_Q4_K;
    up_w.rows = 32;
    up_w.cols = 64;
    gpu.caps = BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT;
    assert(bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
        &gpu, &gate_w, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
        NULL, &gate_w, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
        &gpu, NULL, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
        &gpu, &gate_w, NULL, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
        &gpu, &gate_w, &up_w, 1, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
        &gpu, &gate_w, &up_w, BN_MODEL_ACTIVATION_GELU,
        BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
        &gpu, &gate_w, &up_w, 0, BN_GPU_CODE_Q5K_MATVEC_SPLIT));
    up_w.rows = 16;
    assert(!bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
        &gpu, &gate_w, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    up_w.rows = 32;
    up_w.cols = 32;
    assert(!bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
        &gpu, &gate_w, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    up_w.cols = 64;
    gpu.caps = 0;
    assert(!bn_transformer_gpu_dense_gateup_reference_activation_split_supported(
        &gpu, &gate_w, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));

    BnQWeight q_w;
    BnQWeight k_w;
    BnQWeight packed_qkv_w;
    memset(&q_w, 0, sizeof(q_w));
    memset(&k_w, 0, sizeof(k_w));
    memset(&packed_qkv_w, 0, sizeof(packed_qkv_w));
    q_w.type = BN_GGUF_TENSOR_Q4_0;
    q_w.rows = 32;
    q_w.cols = 64;
    k_w.type = BN_GGUF_TENSOR_Q4_0;
    k_w.rows = 16;
    k_w.cols = 64;
    packed_qkv_w.type = BN_GGUF_TENSOR_Q5_K;
    packed_qkv_w.rows = 64;
    packed_qkv_w.cols = 64;
    gpu.caps = BN_GPU_CAP_LOWBIT_BLOCK32_MATVEC_SPLIT |
               BN_GPU_CAP_DEINTERLEAVED_KQUANT_MATVEC_SPLIT |
               BN_GPU_CAP_NATIVE_QUANT_MATVEC_SPLIT;
    assert(bn_transformer_gpu_packed_qkv_split_supported(
        &gpu, &packed_qkv_w, 1, 0, BN_GPU_CODE_Q5K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_packed_qkv_split_supported(
        &gpu, &packed_qkv_w, 0, 0, BN_GPU_CODE_Q5K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_packed_qkv_split_supported(
        &gpu, &packed_qkv_w, 1, 1, BN_GPU_CODE_Q5K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_packed_qkv_split_supported(
        &gpu, &packed_qkv_w, 1, 0, BN_GPU_CODE_Q8_MATVEC_SPLIT));
    assert(bn_transformer_gpu_qkv_split_standard_supported(
        &gpu, &q_w, BN_GPU_CODE_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_qkv_split_standard_supported(
        &gpu, &q_w, BN_GPU_CODE_Q8_MATVEC_SPLIT));
    q_w.type = BN_GGUF_TENSOR_Q8_0;
    assert(bn_transformer_gpu_qkv_split_native_quant_supported(
        &gpu, &q_w, BN_GPU_CODE_Q8_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_qkv_split_native_quant_supported(
        &gpu, &q_w, BN_GPU_CODE_MATVEC_SPLIT));
    q_w.type = BN_GGUF_TENSOR_Q5_K;
    assert(bn_transformer_gpu_qkv_split_packed_kquant_supported(
        &gpu, &q_w, BN_GPU_CODE_Q5K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_qkv_split_packed_kquant_supported(
        &gpu, &q_w, BN_GPU_CODE_Q8_MATVEC_SPLIT));
    q_w.type = BN_GGUF_TENSOR_Q4_0;
    assert(bn_transformer_gpu_qk_split_supported(
        &gpu, &q_w, &k_w, 32, 16, BN_GPU_CODE_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_qk_split_supported(
        &gpu, &q_w, &k_w, 32, 16, BN_GPU_CODE_UNKNOWN));
    k_w.cols = 32;
    assert(!bn_transformer_gpu_qk_split_supported(
        &gpu, &q_w, &k_w, 32, 16, BN_GPU_CODE_MATVEC_SPLIT));
    k_w.cols = 64;
    k_w.type = BN_GGUF_TENSOR_Q5_0;
    assert(!bn_transformer_gpu_qk_split_supported(
        &gpu, &q_w, &k_w, 32, 16, BN_GPU_CODE_MATVEC_SPLIT));
    k_w.type = BN_GGUF_TENSOR_Q4_0;
    gpu.caps = 0;
    assert(!bn_transformer_gpu_qk_split_supported(
        &gpu, &q_w, &k_w, 32, 16, BN_GPU_CODE_MATVEC_SPLIT));
    gpu.caps = BN_GPU_CAP_LOWBIT_BLOCK32_MATVEC_SPLIT;
    assert(bn_transformer_gpu_ssm_qkvz_split_supported(
        &gpu, &q_w, BN_GPU_CODE_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_ssm_qkvz_split_supported(
        &gpu, &q_w, BN_GPU_CODE_UNKNOWN));
    assert(!bn_transformer_gpu_ssm_qkvz_split_supported(
        NULL, &q_w, BN_GPU_CODE_MATVEC_SPLIT));
    BnQWeight alpha_w;
    BnQWeight beta_w;
    memset(&alpha_w, 0, sizeof(alpha_w));
    memset(&beta_w, 0, sizeof(beta_w));
    alpha_w.type = BN_GGUF_TENSOR_Q4_K;
    beta_w.type = BN_GGUF_TENSOR_Q4_K;
    alpha_w.rows = beta_w.rows = 16;
    alpha_w.cols = beta_w.cols = 64;
    assert(bn_transformer_gpu_can_stack_same_quant_format_alpha_beta(&alpha_w, &beta_w));
    beta_w.cols = 32;
    assert(!bn_transformer_gpu_can_stack_same_quant_format_alpha_beta(&alpha_w, &beta_w));
    beta_w.cols = 64;
    beta_w.rows = 32;
    assert(!bn_transformer_gpu_can_stack_same_quant_format_alpha_beta(&alpha_w, &beta_w));
    beta_w.rows = 16;
    beta_w.type = BN_GGUF_TENSOR_Q5_K;
    assert(!bn_transformer_gpu_can_stack_same_quant_format_alpha_beta(&alpha_w, &beta_w));

    W.type = BN_GGUF_TENSOR_Q4_0;
    W.rows = 32;
    W.cols = 32;
    W.data = (void *)1;
    logits.cpu_weight = &W;

    gpu.max_storage_binding_size = 0;
    assert(!bn_transformer_gpu_logits_needs_cpu_fallback(&gpu, &logits));

    gpu.max_storage_binding_size = bn_qweight_data_size(&W);
    assert(!bn_transformer_gpu_logits_needs_cpu_fallback(&gpu, &logits));

    gpu.max_storage_binding_size = bn_qweight_data_size(&W) - 1;
    assert(bn_transformer_gpu_logits_needs_cpu_fallback(&gpu, &logits));

    logits.cpu_weight = NULL;
    assert(!bn_transformer_gpu_logits_needs_cpu_fallback(&gpu, &logits));

    unsetenv("BN_GPU_NATIVE_QUANT_LOGITS_REFINE_TOP");
    unsetenv("BN_GPU_Q8_REFINE_TOP");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_native_quant_logits_refine_top(&gpu, 1) == 16);
    setenv("BN_GPU_NATIVE_QUANT_LOGITS_REFINE_TOP", "5", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_native_quant_logits_refine_top(&gpu, 1) == 5);
    unsetenv("BN_GPU_NATIVE_QUANT_LOGITS_REFINE_TOP");

    W.type = BN_GGUF_TENSOR_Q6_K;
    W.rows = 1024;
    W.cols = 2048;
    logits.type = BN_GGUF_TENSOR_Q6_K;
    logits.rows = W.rows;
    logits.cols = W.cols;
    logits.cpu_weight = &W;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_GPU_ENABLE_KQUANT_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_KQUANT_LOGITS_REFINE");
    unsetenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 0));
    assert(bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 1));
    gpu.caps |= BN_GPU_CAP_KQUANT_BLOCK32_LOGITS;
    assert(!bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 0));
    assert(bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 1));
    setenv("BN_GPU_ENABLE_KQUANT_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 0));
    setenv("BN_GPU_DISABLE_KQUANT_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 1));
    unsetenv("BN_GPU_DISABLE_KQUANT_LOGITS_REFINE");
    gpu.caps &= ~BN_GPU_CAP_KQUANT_BLOCK32_LOGITS;
    unsetenv("BN_GPU_ENABLE_KQUANT_LOGITS_REFINE");
    test_gpu_runtime_refresh(&gpu);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 0));
    setenv("BN_GPU_DISABLE_KQUANT_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 0));
    unsetenv("BN_GPU_DISABLE_KQUANT_LOGITS_REFINE");
    assert(bn_transformer_gpu_kquant_logits_refine_captures_xb(
        &logits, 1));
    assert(!bn_transformer_gpu_kquant_logits_refine_captures_xb(
        &logits, 0));
    logits.cpu_weight = NULL;
    assert(!bn_transformer_gpu_kquant_logits_refine_captures_xb(
        &logits, 1));
    logits.cpu_weight = &W;

    unsetenv("BN_GPU_KQUANT_LOGITS_REFINE_TOP");
    unsetenv("BN_GPU_Q6_Q8K_REFINE_TOP");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_kquant_logits_refine_top(&gpu, 1) == 64);
    assert(bn_transformer_gpu_kquant_logits_refine_top(&gpu, 0) == 8);
    setenv("BN_GPU_KQUANT_LOGITS_REFINE_TOP", "11", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_kquant_logits_refine_top(&gpu, 1) == 11);
    unsetenv("BN_GPU_KQUANT_LOGITS_REFINE_TOP");
    assert(bn_transformer_gpu_kquant_logits_refine_blocks_per_row(
               BN_QK_K * 3) == 3);
    assert(bn_transformer_gpu_kquant_logits_refine_blocks_per_row(
               BN_QK_K - 1) == 0);
    assert(bn_transformer_gpu_kquant_logits_refine_block_sums_per_row(3) ==
           48);

    W.type = BN_GGUF_TENSOR_Q8_0;
    logits.type = BN_GGUF_TENSOR_Q8_0;
    logits.cpu_weight = &W;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_GPU_ENABLE_NATIVE_QUANT_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_NATIVE_QUANT_LOGITS_REFINE");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_native_quant_logits_refine_active(&gpu, 0));
    assert(bn_transformer_gpu_native_quant_logits_refine_active(&gpu, 1));
    setenv("BN_GPU_ENABLE_NATIVE_QUANT_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_native_quant_logits_refine_active(&gpu, 0));
    unsetenv("BN_GPU_ENABLE_NATIVE_QUANT_LOGITS_REFINE");
    test_gpu_runtime_refresh(&gpu);
    gpu.kind = BN_GPU_BACKEND_WEBGPU;
    assert(bn_transformer_gpu_native_quant_logits_refine_active(&gpu, 0));
    setenv("BN_GPU_DISABLE_NATIVE_QUANT_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_native_quant_logits_refine_active(&gpu, 0));
    unsetenv("BN_GPU_DISABLE_NATIVE_QUANT_LOGITS_REFINE");
    assert(bn_transformer_gpu_native_quant_logits_refine_captures_xb(&logits, 1));
    logits.cpu_weight = NULL;
    assert(!bn_transformer_gpu_native_quant_logits_refine_captures_xb(&logits, 1));
    logits.cpu_weight = &W;

    memset(&c, 0, sizeof(c));
    c.dim = 2048;
    c.policy_flags = 0;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    test_gpu_runtime_refresh(&gpu);
    BnTransformerGPULogitsRefinePolicy refine_policy =
        bn_transformer_gpu_logits_refine_policy(&gpu, &c, NULL, &logits, 1);
    assert(!refine_policy.kquant_default);
    assert(refine_policy.kquant_enabled);
    assert(!refine_policy.kquant_captures_xb);
    assert(refine_policy.kquant_refine_top == 8);
    assert(refine_policy.native_quant_default);
    assert(refine_policy.native_quant_enabled);
    assert(refine_policy.native_quant_captures_xb);
    assert(refine_policy.native_quant_refine_top == 16);
    setenv("BN_GPU_KQUANT_LOGITS_REFINE_TOP", "13", 1);
    setenv("BN_GPU_NATIVE_QUANT_LOGITS_REFINE_TOP", "7", 1);
    test_gpu_runtime_refresh(&gpu);
    refine_policy =
        bn_transformer_gpu_logits_refine_policy(&gpu, &c, NULL, &logits, 1);
    assert(refine_policy.kquant_refine_top == 13);
    assert(refine_policy.native_quant_refine_top == 7);
    unsetenv("BN_GPU_KQUANT_LOGITS_REFINE_TOP");
    unsetenv("BN_GPU_NATIVE_QUANT_LOGITS_REFINE_TOP");
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");

    BnLayerWeights refine_layer;
    BnWeights refine_weights;
    memset(&refine_layer, 0, sizeof(refine_layer));
    memset(&refine_weights, 0, sizeof(refine_weights));
    c.n_layers = 1;
    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    c.policy_flags = 0;
    refine_layer.moe.router_weight = (void *)1;
    refine_layer.moe.expert_map.gate_type = BN_GGUF_TENSOR_Q4_K;
    refine_layer.moe.expert_map.up_type = BN_GGUF_TENSOR_Q4_K;
    refine_layer.moe.expert_map.down_type = BN_GGUF_TENSOR_Q6_K;
    refine_weights.layers = &refine_layer;
    W.type = BN_GGUF_TENSOR_Q6_K;
    logits.type = BN_GGUF_TENSOR_Q6_K;
    logits.cpu_weight = &W;
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_LOGITS_REFINE");
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    refine_policy = bn_transformer_gpu_logits_refine_policy(
        &gpu, &c, &refine_weights, &logits, 0);
    assert(refine_policy.kquant_default);
    assert(refine_policy.kquant_enabled);
    assert(refine_policy.kquant_captures_xb);
    assert(!refine_policy.native_quant_default);
    assert(!refine_policy.native_quant_captures_xb);
    BnTransformerGPULogitsRefineSnapshotPolicy snapshot_policy =
        bn_transformer_gpu_logits_refine_snapshot_policy(
            1, 0, &refine_policy);
    assert(snapshot_policy.snapshot_before_logits);
    assert(snapshot_policy.snapshot_satisfies_kquant_refine);
    snapshot_policy = bn_transformer_gpu_logits_refine_snapshot_policy(
        0, 0, &refine_policy);
    assert(!snapshot_policy.snapshot_before_logits);
    assert(!snapshot_policy.snapshot_satisfies_kquant_refine);
    snapshot_policy = bn_transformer_gpu_logits_refine_snapshot_policy(
        1, 1, &refine_policy);
    assert(snapshot_policy.snapshot_before_logits);
    assert(snapshot_policy.snapshot_satisfies_kquant_refine);
    refine_policy.kquant_captures_xb = 0;
    snapshot_policy = bn_transformer_gpu_logits_refine_snapshot_policy(
        1, 0, &refine_policy);
    assert(!snapshot_policy.snapshot_before_logits);
    assert(!snapshot_policy.snapshot_satisfies_kquant_refine);
    snapshot_policy = bn_transformer_gpu_logits_refine_snapshot_policy(
        1, 0, NULL);
    assert(!snapshot_policy.snapshot_before_logits);
    assert(!snapshot_policy.snapshot_satisfies_kquant_refine);
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");

    unsetenv("BN_CUDA_DISABLE_SSM_FFN_FUSE");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_prefill_ssm_ffn_fuse_allowed(&gpu));
    setenv("BN_CUDA_DISABLE_SSM_FFN_FUSE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_prefill_ssm_ffn_fuse_allowed(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_prefill_ssm_ffn_fuse_allowed(&gpu));
    unsetenv("BN_CUDA_DISABLE_SSM_FFN_FUSE");

    unsetenv("BN_CUDA_ENABLE_MOE_PREFILL");
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_prefill_moe_enabled(&gpu));
    setenv("BN_CUDA_DISABLE_MOE_PREFILL", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_prefill_moe_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_prefill_moe_enabled(&gpu));
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL");
    unsetenv("BN_CUDA_ENABLE_MOE_PREFILL");
    memset(&c, 0, sizeof(c));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(!bn_transformer_prefill_moe_chain_applicable(&gpu, &c));
    c.n_experts = 4;
    c.n_experts_active = 2;
    assert(bn_transformer_prefill_moe_chain_applicable(&gpu, &c));
    c.full_attn_interval = 1;
    assert(!bn_transformer_prefill_moe_chain_applicable(&gpu, &c));

    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_prefill_min_tokens(&gpu) == 1);
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "0", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_prefill_min_tokens(&gpu) == 1);
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "9", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_prefill_min_tokens(&gpu) == 9);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_moe_prefill_min_tokens(&gpu) == 1);
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");

    unsetenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_moe_cache_prefill_enabled(&gpu));
    setenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy, "BN_CUDA_DISABLE_MOE_CACHE_PREFILL",
               "1", 1) == 0);
    assert(!bn_transformer_gpu_moe_cache_prefill_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_moe_cache_prefill_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_CUDA_DISABLE_MOE_CACHE_PREFILL");

    memset(&c, 0, sizeof(c));
    c.n_experts = 2;
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ROUTE_RESIDENT");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &gpu, &c, 0, 0, 1, 0));
    c.n_experts = 3;
    assert(bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &gpu, &c, 0, 0, 1, 0));
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &gpu, &c, 0, 1, 1, 0));
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &gpu, &c, 0, 0, 0, 0));
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &gpu, NULL, 0, 0, 1, 0));
    setenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CPU_ROUTE_RESIDENT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &gpu, &c, 0, 0, 1, 0));
    unsetenv("BN_CUDA_DISABLE_NATIVE_QUANT_MOE_CPU_ROUTE_RESIDENT");
    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    c.dim = 2048;
    assert(bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &gpu, &c, 1, 0, 0, 0));
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ROUTE_RESIDENT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &gpu, &c, 1, 0, 0, 0));
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ROUTE_RESIDENT");

    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_moe_prefill_shared_fuse_enabled(&gpu));
    setenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy,
               "BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE", "1", 1) == 0);
    assert(!bn_transformer_gpu_moe_prefill_shared_fuse_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_moe_prefill_shared_fuse_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");

    unsetenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(!bn_transformer_gpu_moe_route_batch_debug_enabled(&gpu));
    setenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy, "BN_CUDA_DEBUG_MOE_ROUTE_BATCH",
               "1", 1) == 0);
    assert(bn_transformer_gpu_moe_route_batch_debug_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_moe_route_batch_debug_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_CUDA_DEBUG_MOE_ROUTE_BATCH");

    unsetenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_lazy_aux_cache_enabled(&gpu));
    setenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_lazy_aux_cache_enabled(&gpu));
    unsetenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE");

    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(!bn_transformer_gpu_large_hybrid_prefill_disabled(&gpu));
    assert(!bn_transformer_prefill_large_hybrid_disabled(&gpu));
    setenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy, "BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL",
               "1", 1) == 0);
    assert(bn_transformer_gpu_large_hybrid_prefill_disabled(&gpu));
    assert(bn_transformer_prefill_large_hybrid_disabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_large_hybrid_prefill_disabled(&gpu));
    assert(!bn_transformer_prefill_large_hybrid_disabled(&gpu));
    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL");

    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_prefill_dense_chain_enabled(&gpu));
    setenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_prefill_dense_chain_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_prefill_dense_chain_enabled(&gpu));
    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");

    memset(&c, 0, sizeof(c));
    c.dim = 2048;
    c.full_attn_interval = 1;
    c.ssm_inner_size = 128;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_prefill_hybrid_chain_applicable(&gpu, &c));
    assert(!bn_transformer_gpu_large_hybrid_prefill_decode_fallback_default(
        &gpu, &c));
    assert(bn_transformer_prefill_hybrid_chain_enabled(&gpu, &c));
    assert(bn_transformer_gpu_prefill_hybrid_chain_enabled(&gpu, &c));
    setenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_prefill_hybrid_chain_enabled(&gpu, &c));
    assert(!bn_transformer_gpu_prefill_hybrid_chain_enabled(&gpu, &c));
    unsetenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN");
    test_gpu_runtime_refresh(&gpu);
    c.ssm_inner_size = 0;
    assert(!bn_transformer_prefill_hybrid_chain_applicable(&gpu, &c));
    c.ssm_inner_size = 128;
    c.dim = 4096;
    assert(!bn_transformer_gpu_large_hybrid_prefill_decode_fallback_default(
        &gpu, &c));
    assert(!bn_transformer_gpu_large_hybrid_prefill_chain_disabled_default(
        &gpu, &c));
    assert(bn_transformer_prefill_hybrid_chain_enabled(&gpu, &c));
    setenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_large_hybrid_prefill_decode_fallback_default(
        &gpu, &c));
    assert(bn_transformer_gpu_large_hybrid_prefill_chain_disabled_default(
        &gpu, &c));
    assert(!bn_transformer_prefill_hybrid_chain_enabled(&gpu, &c));
    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL");
    test_gpu_runtime_refresh(&gpu);

    unsetenv("BN_CUDA_DISABLE_PREFILL_ATTN");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_prefill_attention_enabled(&gpu));
    assert(bn_transformer_gpu_prefill_attention_enabled(&gpu));
    setenv("BN_CUDA_DISABLE_PREFILL_ATTN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_prefill_attention_enabled(&gpu));
    assert(!bn_transformer_gpu_prefill_attention_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_prefill_attention_enabled(&gpu));
    assert(bn_transformer_gpu_prefill_attention_enabled(&gpu));
    unsetenv("BN_CUDA_DISABLE_PREFILL_ATTN");
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_prefill_attention_min_tokens(NULL, &gpu) == 16);
    assert(bn_transformer_prefill_attention_min_tokens(NULL, &gpu) == 16);
    setenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "11", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_prefill_attention_min_tokens(NULL, &gpu) == 11);
    assert(bn_transformer_prefill_attention_min_tokens(NULL, &gpu) == 11);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_prefill_attention_min_tokens(NULL, &gpu) == 16);
    assert(bn_transformer_prefill_attention_min_tokens(NULL, &gpu) == 16);
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");

    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_prefill_ssm_run_chain_enabled(&gpu));
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_prefill_ssm_run_chain_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_prefill_ssm_run_chain_enabled(&gpu));
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN");

    unsetenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_prefill_moe_chain_debug_enabled(&gpu));
    setenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_prefill_moe_chain_debug_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_prefill_moe_chain_debug_enabled(&gpu));
    unsetenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");

    unsetenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_prefill_hybrid_chain_debug_enabled(&gpu));
    assert(!bn_transformer_gpu_prefill_hybrid_chain_debug_enabled(&gpu));
    setenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_prefill_hybrid_chain_debug_enabled(&gpu));
    assert(bn_transformer_gpu_prefill_hybrid_chain_debug_enabled(&gpu));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_prefill_hybrid_chain_debug_enabled(&gpu));
    assert(!bn_transformer_gpu_prefill_hybrid_chain_debug_enabled(&gpu));
    unsetenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN");

    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    unsetenv("BN_GPU_CPU_FALLBACK_FROM_LAYER");
    unsetenv("BN_GPU_CPU_ATTN_LAYER");
    unsetenv("BN_GPU_CPU_ATTN_FROM_LAYER");
    unsetenv("BN_GPU_CPU_FFN_LAYER");
    unsetenv("BN_GPU_CPU_FFN_FROM_LAYER");
    unsetenv("BN_GPU_CPU_FFN_DOWN_FROM_LAYER");
    test_gpu_runtime_refresh(&gpu);
    BnTransformerGPUCPUFallbackPolicy fallback_policy =
        bn_transformer_gpu_cpu_fallback_policy(&gpu);
    assert(fallback_policy.layer == -1);
    assert(fallback_policy.from_layer == -1);
    assert(fallback_policy.attn_layer == -1);
    assert(fallback_policy.attn_from_layer == -1);
    assert(fallback_policy.ffn_layer == -1);
    assert(fallback_policy.ffn_from_layer == -1);
    assert(fallback_policy.ffn_down_from_layer == -1);
    setenv("BN_GPU_CPU_FALLBACK_LAYER", "2", 1);
    setenv("BN_GPU_CPU_FALLBACK_FROM_LAYER", "3", 1);
    setenv("BN_GPU_CPU_ATTN_LAYER", "4", 1);
    setenv("BN_GPU_CPU_ATTN_FROM_LAYER", "5", 1);
    setenv("BN_GPU_CPU_FFN_LAYER", "6", 1);
    setenv("BN_GPU_CPU_FFN_FROM_LAYER", "7", 1);
    setenv("BN_GPU_CPU_FFN_DOWN_FROM_LAYER", "8", 1);
    test_gpu_runtime_refresh(&gpu);
    fallback_policy = bn_transformer_gpu_cpu_fallback_policy(&gpu);
    assert(fallback_policy.layer == 2);
    assert(fallback_policy.from_layer == 3);
    assert(fallback_policy.attn_layer == 4);
    assert(fallback_policy.attn_from_layer == 5);
    assert(fallback_policy.ffn_layer == 6);
    assert(fallback_policy.ffn_from_layer == 7);
    assert(fallback_policy.ffn_down_from_layer == 8);
    assert(!bn_transformer_gpu_cpu_fallback_layer_selected(1, 2, -1));
    assert(bn_transformer_gpu_cpu_fallback_layer_selected(2, 2, -1));
    assert(!bn_transformer_gpu_cpu_fallback_layer_selected(2, -1, 3));
    assert(bn_transformer_gpu_cpu_fallback_layer_selected(3, -1, 3));
    assert(bn_transformer_gpu_cpu_fallback_layer_selected(8, -1, 3));
    assert(bn_transformer_gpu_cpu_fallback_layer_selected(2, 2, 4));
    assert(bn_transformer_gpu_cpu_fallback_layer_selected(4, 2, 4));
    assert(!bn_transformer_gpu_cpu_fallback_layer_selected(1, -1, -1));
    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    unsetenv("BN_GPU_CPU_FALLBACK_FROM_LAYER");
    unsetenv("BN_GPU_CPU_ATTN_LAYER");
    unsetenv("BN_GPU_CPU_ATTN_FROM_LAYER");
    unsetenv("BN_GPU_CPU_FFN_LAYER");
    unsetenv("BN_GPU_CPU_FFN_FROM_LAYER");
    unsetenv("BN_GPU_CPU_FFN_DOWN_FROM_LAYER");

    BnWeights hyper_weights = {0};
    c.hyper_connection_count = 4;
    c.policy_flags |= BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
    c.n_layers = 48;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    fallback_policy = (BnTransformerGPUCPUFallbackPolicy)
        {-1, -1, -1, -1, -1, -1, -1};
    fallback_policy = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        fallback_policy, &gpu, &c, &hyper_weights);
    assert(fallback_policy.attn_layer == 47);
    assert(fallback_policy.attn_from_layer == -1);
    fallback_policy = (BnTransformerGPUCPUFallbackPolicy)
        {-1, -1, 3, -1, -1, -1, -1};
    fallback_policy = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        fallback_policy, &gpu, &c, &hyper_weights);
    assert(fallback_policy.attn_layer == 3);
    c.hyper_connection_count = 0;
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
    c.n_layers = 4;

    unsetenv("BN_GPU_COMPARE_ATTENTION_LAYER");
    unsetenv("BN_GPU_COMPARE_ATTENTION_POS");
    unsetenv("BN_GPU_COMPARE_GQA_LAYER");
    unsetenv("BN_GPU_COMPARE_GQA_POS");
    unsetenv("BN_GPU_COMPARE_QKV_LAYER");
    unsetenv("BN_GPU_COMPARE_QKV_POS");
    unsetenv("BN_GPU_COMPARE_FFN_DOWN_LAYER");
    unsetenv("BN_GPU_COMPARE_FFN_DOWN_POS");
    unsetenv("BN_GPU_COMPARE_FFN_STATE_LAYER");
    unsetenv("BN_GPU_COMPARE_FFN_STATE_POS");
    unsetenv("BN_GPU_COMPARE_SSM_LAYER");
    unsetenv("BN_GPU_COMPARE_SSM_POS");
    BnTransformerGPUComparePolicy compare_policy =
        bn_transformer_gpu_compare_policy(&gpu);
    assert(compare_policy.attention_layer == -1);
    assert(compare_policy.attention_pos == -1);
    assert(compare_policy.gqa_layer == -1);
    assert(compare_policy.gqa_pos == -1);
    assert(compare_policy.qkv_layer == -1);
    assert(compare_policy.qkv_pos == -1);
    assert(compare_policy.ffn_down_layer == -1);
    assert(compare_policy.ffn_down_pos == -1);
    assert(compare_policy.ffn_state_layer == -1);
    assert(compare_policy.ffn_state_pos == -1);
    assert(compare_policy.ssm_layer == -1);
    assert(compare_policy.ssm_pos == -1);
    setenv("BN_GPU_COMPARE_ATTENTION_LAYER", "1", 1);
    setenv("BN_GPU_COMPARE_ATTENTION_POS", "2", 1);
    setenv("BN_GPU_COMPARE_GQA_LAYER", "3", 1);
    setenv("BN_GPU_COMPARE_GQA_POS", "4", 1);
    setenv("BN_GPU_COMPARE_QKV_LAYER", "5", 1);
    setenv("BN_GPU_COMPARE_QKV_POS", "6", 1);
    setenv("BN_GPU_COMPARE_FFN_DOWN_LAYER", "7", 1);
    setenv("BN_GPU_COMPARE_FFN_DOWN_POS", "8", 1);
    setenv("BN_GPU_COMPARE_FFN_STATE_LAYER", "9", 1);
    setenv("BN_GPU_COMPARE_FFN_STATE_POS", "10", 1);
    setenv("BN_GPU_COMPARE_SSM_LAYER", "11", 1);
    setenv("BN_GPU_COMPARE_SSM_POS", "12", 1);
    test_gpu_runtime_refresh(&gpu);
    compare_policy = bn_transformer_gpu_compare_policy(&gpu);
    assert(compare_policy.attention_layer == 1);
    assert(compare_policy.attention_pos == 2);
    assert(compare_policy.gqa_layer == 3);
    assert(compare_policy.gqa_pos == 4);
    assert(compare_policy.qkv_layer == 5);
    assert(compare_policy.qkv_pos == 6);
    assert(compare_policy.ffn_down_layer == 7);
    assert(compare_policy.ffn_down_pos == 8);
    assert(compare_policy.ffn_state_layer == 9);
    assert(compare_policy.ffn_state_pos == 10);
    assert(compare_policy.ssm_layer == 11);
    assert(compare_policy.ssm_pos == 12);
    unsetenv("BN_GPU_COMPARE_ATTENTION_LAYER");
    unsetenv("BN_GPU_COMPARE_ATTENTION_POS");
    unsetenv("BN_GPU_COMPARE_GQA_LAYER");
    unsetenv("BN_GPU_COMPARE_GQA_POS");
    unsetenv("BN_GPU_COMPARE_QKV_LAYER");
    unsetenv("BN_GPU_COMPARE_QKV_POS");
    unsetenv("BN_GPU_COMPARE_FFN_DOWN_LAYER");
    unsetenv("BN_GPU_COMPARE_FFN_DOWN_POS");
    unsetenv("BN_GPU_COMPARE_FFN_STATE_LAYER");
    unsetenv("BN_GPU_COMPARE_FFN_STATE_POS");
    unsetenv("BN_GPU_COMPARE_SSM_LAYER");
    unsetenv("BN_GPU_COMPARE_SSM_POS");

    memset(&c, 0, sizeof(c));
    c.n_layers = 40;
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT");
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_FROM_LAYER");
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_TO_LAYER");
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_TAIL_NATIVE");
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_ATTN_ONLY");
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_FFN_ONLY");
    unsetenv("BN_METAL_NATIVE_QUANT_PREPARED");
    unsetenv("BN_METAL_Q4_PREPARED");
    test_gpu_runtime_refresh(&gpu);
    BnTransformerGPUSmallDenseNativeQuantLayerPolicy small_dense_native_quant_policy =
        bn_transformer_gpu_small_dense_native_quant_layer_policy_for_backend(
            &gpu, &c);
    assert(small_dense_native_quant_policy.from_layer == -1);
    assert(small_dense_native_quant_policy.to_layer == -1);
    assert(!small_dense_native_quant_policy.attn_only);
    assert(!small_dense_native_quant_policy.ffn_only);
    setenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    small_dense_native_quant_policy =
        bn_transformer_gpu_small_dense_native_quant_layer_policy_for_backend(
            &gpu, &c);
    assert(small_dense_native_quant_policy.from_layer == 39);
    assert(small_dense_native_quant_policy.to_layer == 0);
    setenv("BN_METAL_NATIVE_QUANT_PREPARED", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    small_dense_native_quant_policy =
        bn_transformer_gpu_small_dense_native_quant_layer_policy_for_backend(
            &gpu, &c);
    assert(small_dense_native_quant_policy.from_layer == 39);
    assert(small_dense_native_quant_policy.to_layer == -1);
    unsetenv("BN_METAL_NATIVE_QUANT_PREPARED");
    setenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_FROM_LAYER", "10", 1);
    setenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_TO_LAYER", "20", 1);
    setenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_ATTN_ONLY", "1", 1);
    setenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_FFN_ONLY", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    small_dense_native_quant_policy =
        bn_transformer_gpu_small_dense_native_quant_layer_policy_for_backend(
            &gpu, &c);
    assert(small_dense_native_quant_policy.from_layer == 10);
    assert(small_dense_native_quant_policy.to_layer == 20);
    assert(small_dense_native_quant_policy.attn_only);
    assert(small_dense_native_quant_policy.ffn_only);
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_TO_LAYER");
    setenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_TAIL_NATIVE", "4", 1);
    test_gpu_runtime_refresh(&gpu);
    small_dense_native_quant_policy =
        bn_transformer_gpu_small_dense_native_quant_layer_policy_for_backend(
            &gpu, &c);
    assert(small_dense_native_quant_policy.to_layer == 35);
    setenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_TAIL_NATIVE", "100", 1);
    test_gpu_runtime_refresh(&gpu);
    small_dense_native_quant_policy =
        bn_transformer_gpu_small_dense_native_quant_layer_policy_for_backend(
            &gpu, &c);
    assert(small_dense_native_quant_policy.to_layer == -1);
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT");
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_FROM_LAYER");
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_TAIL_NATIVE");
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_ATTN_ONLY");
    unsetenv("BN_GPU_SMALL_DENSE_NATIVE_QUANT_FFN_ONLY");
    unsetenv("BN_METAL_NATIVE_QUANT_PREPARED");
    unsetenv("BN_METAL_Q4_PREPARED");

    BnTransformerGPUSmallDenseNativeQuantLayerPolicy manual_small_dense_native_quant_policy = {
        .from_layer = 2,
        .to_layer = 4,
        .attn_only = 0,
        .ffn_only = 0,
    };
    c.policy_flags = 0;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    BnTransformerGPUSmallDenseNativeQuantLayerUsePolicy small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy, 1, 0, -1);
    assert(!small_dense_native_quant_use.use_layer);
    assert(!small_dense_native_quant_use.use_attention);
    assert(!small_dense_native_quant_use.use_ffn);
    assert(!small_dense_native_quant_use.use_ffn_down);
    small_dense_native_quant_use = bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
        &gpu, &c, &manual_small_dense_native_quant_policy, 2, 0, -1);
    assert(small_dense_native_quant_use.use_layer);
    assert(small_dense_native_quant_use.use_attention);
    assert(small_dense_native_quant_use.use_ffn);
    assert(small_dense_native_quant_use.use_ffn_down);
    c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT |
                     BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
    small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy, 2, 0, -1);
    assert(small_dense_native_quant_use.use_hc_attention);
    c.policy_flags = 0;
    manual_small_dense_native_quant_policy.attn_only = 1;
    small_dense_native_quant_use = bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
        &gpu, &c, &manual_small_dense_native_quant_policy, 2, 0, -1);
    assert(small_dense_native_quant_use.use_attention);
    assert(!small_dense_native_quant_use.use_ffn);
    assert(!small_dense_native_quant_use.use_ffn_down);
    manual_small_dense_native_quant_policy.attn_only = 0;
    manual_small_dense_native_quant_policy.ffn_only = 1;
    small_dense_native_quant_use = bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
        &gpu, &c, &manual_small_dense_native_quant_policy, 2, 0, -1);
    assert(!small_dense_native_quant_use.use_attention);
    assert(small_dense_native_quant_use.use_ffn);
    assert(small_dense_native_quant_use.use_ffn_down);
    manual_small_dense_native_quant_policy.from_layer = -1;
    manual_small_dense_native_quant_policy.to_layer = -1;
    manual_small_dense_native_quant_policy.ffn_only = 0;
    c.policy_flags = BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ATTENTION;
    small_dense_native_quant_use = bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
        &gpu, &c, &manual_small_dense_native_quant_policy, 0, 0, -1);
    assert(!small_dense_native_quant_use.use_layer);
    assert(!small_dense_native_quant_use.use_attention);
    assert(!small_dense_native_quant_use.use_ffn);
    manual_small_dense_native_quant_policy.ffn_only = 1;
    small_dense_native_quant_use = bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
        &gpu, &c, &manual_small_dense_native_quant_policy, 0, 0, -1);
    assert(!small_dense_native_quant_use.use_attention);
    manual_small_dense_native_quant_policy.ffn_only = 0;
    gpu.kind = BN_GPU_BACKEND_METAL;
    gpu.caps |= BN_GPU_CAP_REFERENCE_ATTENTION;
    c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION;
    small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy, 0, 0, -1);
    assert(!small_dense_native_quant_use.use_layer);
    assert(!small_dense_native_quant_use.use_attention);
    assert(!small_dense_native_quant_use.use_ffn);
    manual_small_dense_native_quant_policy.ffn_only = 1;
    small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy, 0, 0, -1);
    assert(!small_dense_native_quant_use.use_attention);
    assert(!small_dense_native_quant_use.use_ffn);
    manual_small_dense_native_quant_policy.ffn_only = 0;
    gpu.caps |= BN_GPU_CAP_PREPARED_NATIVE_QUANT;
    c.policy_flags |= BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT;
    small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy, 0, 0, -1);
    assert(!small_dense_native_quant_use.use_layer);
    assert(!small_dense_native_quant_use.use_attention);
    assert(!small_dense_native_quant_use.use_ffn);
    gpu.caps |= BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN;
    small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy, 0, 0, -1);
    assert(small_dense_native_quant_use.use_layer);
    assert(!small_dense_native_quant_use.use_attention);
    assert(small_dense_native_quant_use.use_ffn);
    assert(!small_dense_native_quant_use.use_ffn_down);
    gpu.caps |= BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN_DOWN;
    small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy, 0, 0, -1);
    assert(small_dense_native_quant_use.use_ffn_down);
    gpu.caps |= BN_GPU_CAP_PREPARED_NATIVE_QUANT_ATTENTION;
    small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy, 0, 0, -1);
    assert(small_dense_native_quant_use.use_attention);
    gpu.caps |= BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH;
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT;
    small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy, 0, 0, -1);
    assert(small_dense_native_quant_use.use_layer);
    assert(small_dense_native_quant_use.use_attention);
    gpu.caps &= ~BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH;
    c.policy_flags |= BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT;
    gpu.caps &= ~BN_GPU_CAP_PREPARED_NATIVE_QUANT_ATTENTION;
    gpu.caps &= ~BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN_DOWN;
    gpu.caps &= ~BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN;
    gpu.caps &= ~BN_GPU_CAP_PREPARED_NATIVE_QUANT;
    gpu.caps &= ~BN_GPU_CAP_REFERENCE_ATTENTION;
    c.policy_flags = 0;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    c.policy_flags = 0;
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_FFN_DOWN");
    small_dense_native_quant_use = bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
        &gpu, &c, &manual_small_dense_native_quant_policy, 4, 1, 3);
    assert(!small_dense_native_quant_use.use_layer);
    assert(!small_dense_native_quant_use.small_dense_native_quant_path);
    small_dense_native_quant_use = bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
        &gpu, &c, &manual_small_dense_native_quant_policy, 3, 1, 3);
    assert(small_dense_native_quant_use.use_layer);
    assert(small_dense_native_quant_use.small_dense_native_quant_path);
    assert(small_dense_native_quant_use.use_attention);
    assert(small_dense_native_quant_use.use_ffn);
    assert(!small_dense_native_quant_use.use_ffn_down);
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_FFN_DOWN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    small_dense_native_quant_use = bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
        &gpu, &c, &manual_small_dense_native_quant_policy, 3, 1, 3);
    assert(small_dense_native_quant_use.use_ffn_down);
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_FFN_DOWN");
    test_gpu_runtime_refresh(&gpu);
    c.policy_flags = 0;

    gpu.caps |= BN_GPU_CAP_PREPARED_NATIVE_QUANT |
                BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN;
    c.policy_flags = BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT;
    manual_small_dense_native_quant_policy.attn_only = 1;
    manual_small_dense_native_quant_policy.ffn_only = 1;
    small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy,
            0, 0, -1);
    assert(small_dense_native_quant_use.use_layer);
    assert(!small_dense_native_quant_use.use_attention);
    assert(small_dense_native_quant_use.use_ffn);
    assert(!small_dense_native_quant_use.use_ffn_down);
    gpu.caps |= BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN_DOWN;
    small_dense_native_quant_use =
        bn_transformer_gpu_small_dense_native_quant_layer_use_policy(
            &gpu, &c, &manual_small_dense_native_quant_policy,
            0, 0, -1);
    assert(small_dense_native_quant_use.use_ffn_down);
    gpu.caps &= ~BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN_DOWN;
    gpu.caps &= ~BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN;
    gpu.caps &= ~BN_GPU_CAP_PREPARED_NATIVE_QUANT;
    manual_small_dense_native_quant_policy.attn_only = 0;
    manual_small_dense_native_quant_policy.ffn_only = 0;
    c.policy_flags = 0;

    BnTransformerGPUCachedDecodePolicy cached_decode =
        bn_transformer_gpu_cached_decode_policy(0, 1, 0, 0);
    assert(!cached_decode.use_cache);
    assert(!cached_decode.clear_cache);
    cached_decode = bn_transformer_gpu_cached_decode_policy(4, 0, 0, 0);
    assert(cached_decode.use_cache);
    assert(!cached_decode.clear_cache);
    cached_decode = bn_transformer_gpu_cached_decode_policy(4, 1, 1, 0);
    assert(cached_decode.use_cache);
    assert(!cached_decode.clear_cache);
    cached_decode = bn_transformer_gpu_cached_decode_policy(4, 1, 0, 1);
    assert(cached_decode.use_cache);
    assert(!cached_decode.clear_cache);
    cached_decode = bn_transformer_gpu_cached_decode_policy(4, 1, 0, 0);
    assert(!cached_decode.use_cache);
    assert(cached_decode.clear_cache);
    assert(bn_transformer_gpu_patch_cached_decode_ops(
               NULL, 0, NULL, 0) == -1);
    BnConfig cached_config = {0};
    cached_config.seq_len = 8;
    cached_config.kv_dim = 4;
    BnGPUOp cached_ops[10] = {0};
    cached_ops[0].op_code = BN_GPU_CODE_MATVEC;
    cached_ops[0].buf_out = BN_GPU_VALUE_KEY_CACHE;
    cached_ops[0].rows = 2;
    cached_ops[0].p[5] = 37;
    cached_ops[1].op_code = BN_GPU_CODE_MATVEC_SPLIT;
    cached_ops[1].buf_aux = BN_GPU_VALUE_KEY_CACHE;
    cached_ops[1].rows = BN_GPU_VALUE_VALUE_CACHE;
    cached_ops[1].p[0] = 8;
    cached_ops[1].p[2] = 6;
    cached_ops[1].p[3] = 8;
    cached_ops[1].p[6] = 36;
    cached_ops[1].p[7] = 70;
    cached_ops[2].op_code = BN_GPU_CODE_ROPE_QK;
    cached_ops[2].buf_aux = BN_GPU_VALUE_KEY_CACHE;
    cached_ops[2].p[5] = 35;
    cached_ops[3].op_code = BN_GPU_CODE_FLASH_ATTN;
    cached_ops[4].op_code = BN_GPU_CODE_PER_HEAD_RMSNORM;
    cached_ops[4].buf_in = BN_GPU_VALUE_KEY_CACHE;
    cached_ops[4].p[0] = 1;
    cached_ops[4].p[3] = 34;
    cached_ops[5].op_code = BN_GPU_CODE_PER_HEAD_RMSNORM;
    cached_ops[5].buf_in = BN_GPU_VALUE_VALUE_CACHE;
    cached_ops[5].p[0] = 1;
    cached_ops[5].p[3] = 66;
    cached_ops[6].op_code = BN_GPU_CODE_COPY;
    cached_ops[6].buf_out = BN_GPU_VALUE_VALUE_CACHE;
    cached_ops[6].p[1] = 65;
    cached_ops[6].p[2] = 2;
    cached_ops[7].op_code = BN_GPU_CODE_GQA_SCORES;
    cached_ops[7].p[2] = 1;
    cached_ops[8].op_code = BN_GPU_CODE_SOFTMAX;
    cached_ops[8].p[1] = 1;
    cached_ops[9].op_code = BN_GPU_CODE_GQA_COMBINE;
    cached_ops[9].p[2] = 1;
    assert(bn_transformer_gpu_patch_cached_decode_ops(
               cached_ops, 10, &cached_config, 11) == 0);
    assert(cached_ops[0].p[5] == 44);
    assert(cached_ops[1].p[6] == 44);
    assert(cached_ops[1].p[7] == 76);
    assert(cached_ops[2].p[2] == 11);
    assert(cached_ops[2].p[5] == 44);
    assert(cached_ops[3].p[2] == 8);
    assert(cached_ops[4].p[3] == 44);
    assert(cached_ops[5].p[3] == 76);
    assert(cached_ops[6].p[1] == 76);
    assert(cached_ops[7].p[2] == 8);
    assert(cached_ops[8].p[1] == 8);
    assert(cached_ops[9].p[2] == 8);

    BnBackendSession *decode_backend = bn_backend_session_create();
    assert(decode_backend);
    BnTransformerGPUDecodeSessionResources decode_session_resources;
    assert(bn_transformer_gpu_resolve_decode_session_resources(
               NULL, decode_backend, 4, 0) == -1);
    assert(bn_transformer_gpu_resolve_decode_session_resources(
               &decode_session_resources, NULL, 4, 0) == -1);
    assert(bn_transformer_gpu_resolve_decode_session_resources(
               &decode_session_resources, decode_backend, 0, 0) == -1);
    assert(bn_transformer_gpu_resolve_decode_session_resources(
               &decode_session_resources, decode_backend, 4, 0) == 0);
    assert(decode_session_resources.command_buffer != NULL);
    assert(decode_session_resources.command_cap >= 4);
    assert(decode_session_resources.cached_op_count == 0);
    assert(!decode_session_resources.cached_has_logits);
    BnSessionBackendState decode_backend_state = {
        .backend = decode_backend,
    };
    BnSession decode_session = {
        .backend_state = &decode_backend_state,
    };
    assert(bn_transformer_gpu_resolve_session_decode_resources(
               &decode_session_resources, &decode_session, 4, 0) == 0);
    assert(bn_transformer_gpu_resolve_session_decode_resources(
               &decode_session_resources, NULL, 4, 0) == -1);
    BnGPUOp decode_ops[4];
    BnTransformerGPUEmitContext decode_emit;
    assert(bn_transformer_gpu_emit_context_init_decode_session(
               &decode_emit, NULL, decode_ops, 4, 8, 4) == -1);
    assert(bn_transformer_gpu_emit_context_init_decode_session(
               &decode_emit, &decode_session, decode_ops, 4, 8, 4) == 0);
    assert(decode_emit.graph != NULL);
    assert(decode_emit.lowering_values != NULL);
    bn_transformer_gpu_emit_context_free(&decode_emit);
    bn_transformer_gpu_store_decode_session_cache(decode_backend, 3, 1);
    assert(bn_transformer_gpu_resolve_decode_session_resources(
               &decode_session_resources, decode_backend, 4, 1) == 0);
    assert(decode_session_resources.cached_op_count == 3);
    assert(decode_session_resources.cached_has_logits);
    bn_transformer_gpu_clear_session_decode_cache(&decode_session);
    assert(bn_transformer_gpu_resolve_decode_session_resources(
               &decode_session_resources, decode_backend, 4, 1) == 0);
    assert(decode_session_resources.cached_op_count == 0);
    assert(!decode_session_resources.cached_has_logits);
    bn_transformer_gpu_store_session_decode_cache(&decode_session, 2, 0);
    assert(bn_transformer_gpu_resolve_session_decode_resources(
               &decode_session_resources, &decode_session, 4, 1) == 0);
    assert(decode_session_resources.cached_op_count == 2);
    assert(!decode_session_resources.cached_has_logits);
    bn_backend_session_free(decode_backend);

    setenv("BN_GPU_FLASH_MIN_KV", "0", 1);
    setenv("BN_GPU_FLASH_MAX_KV", "2048", 1);
    gpu.caps = BN_GPU_CAP_FLASH_ATTN;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_flash_attention_enabled(&gpu, 0, 0, 128));
    assert(!bn_transformer_gpu_flash_attention_enabled(&gpu, 0, 0, 4096));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_flash_attention_enabled(&gpu, 0, 0, 128));
    assert(bn_transformer_gpu_flash_attention_enabled(&gpu, 1, 0, 128));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    BnConfig flash_config = {0};
    flash_config.policy_flags =
        BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION;
    assert(!bn_transformer_gpu_model_flash_attention_enabled(
        &gpu, &flash_config, 0, 128));
    flash_config.flash_attn = 1;
    assert(bn_transformer_gpu_model_flash_attention_enabled(
        &gpu, &flash_config, 0, 128));
    flash_config.flash_attn = 0;
    assert(bn_transformer_gpu_model_flash_attention_enabled(
        &gpu, &flash_config, 1, 128));

    BnMoEExpertMap map;
    memset(&map, 0, sizeof(map));
    map.gate_type = BN_GGUF_TENSOR_Q4_K;
    map.up_type = BN_GGUF_TENSOR_Q4_K;
    map.down_type = BN_GGUF_TENSOR_Q6_K;
    map.gate_rows = 4096;
    map.gate_cols = 2048;
    map.up_rows = 4096;
    map.up_cols = 2048;
    map.down_rows = 2048;
    map.down_cols = 4096;
    assert(bn_transformer_gpu_moe_routed_kquant_down_allowed(&map, 0));
    assert(bn_transformer_gpu_moe_routed_kquant_down(&map));
    assert(!bn_transformer_gpu_moe_routed_native_quant(&map));

    BnWeights moe_w;
    BnLayerWeights moe_layers[1];
    memset(&moe_w, 0, sizeof(moe_w));
    memset(moe_layers, 0, sizeof(moe_layers));
    moe_w.layers = moe_layers;
    moe_layers[0].moe.router_weight = (void *)1;
    moe_layers[0].moe.expert_map = map;
    c.n_layers = 1;
    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    c.dim = 2048;
    c.moe_norm_topk_prob = 1;
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_fallback_enabled(
        &gpu, &c, &moe_w));
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_layer_enabled(
        &gpu, &c, &moe_layers[0], c.dim));
    assert(bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 0, -1, -1));
    c.n_experts = 256;
    c.n_experts_active = 8;
    gpu.max_moe_route_experts = 128;
    gpu.caps |= BN_GPU_CAP_MOE_EXPERT_GRAPH;
    assert(!bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 0, -1, -1));
    gpu.caps &= ~BN_GPU_CAP_MOE_EXPERT_GRAPH;
    gpu.max_moe_route_experts = 0;
    c.n_experts = 2;
    c.n_experts_active = 2;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    gpu.caps |= BN_GPU_CAP_MOE_ROUTED_FFN |
                BN_GPU_CAP_MOE_ROUTED_KQUANT_DOWN_CACHE |
                BN_GPU_CAP_MOE_ROUTED_NATIVE_QUANT;
    assert(bn_transformer_gpu_all_active_two_kquant_moe_model(&c, &moe_w));
    assert(bn_transformer_gpu_all_active_two_kquant_moe_layer(
        &c, &moe_layers[0], c.dim));
    unsetenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE");
    BnBackendModel *resident_backend = bn_backend_model_create();
    assert(resident_backend);
    BnTransformerGPUMoEDecodeResources decode_resources =
        bn_transformer_gpu_resolve_moe_decode_resources(NULL, 0);
    assert(!decode_resources.has_router);
    assert(!decode_resources.resident_valid);
    decode_resources =
        bn_transformer_gpu_resolve_moe_decode_resources(resident_backend, 0);
    assert(!decode_resources.has_router);
    assert(!decode_resources.resident_valid);
    assert(bn_backend_model_register_handle(
               resident_backend, 0, BN_BACKEND_HANDLE_MOE_ROUTER,
               (void *)2) == 0);
    assert(bn_backend_model_register_handle(
               resident_backend, 0, BN_BACKEND_HANDLE_MOE_GATE_ALL,
               (void *)3) == 0);
    assert(bn_backend_model_register_handle(
               resident_backend, 0, BN_BACKEND_HANDLE_MOE_UP_ALL,
               (void *)4) == 0);
    assert(bn_backend_model_register_handle(
               resident_backend, 0, BN_BACKEND_HANDLE_MOE_DOWN_ALL,
               (void *)5) == 0);
    decode_resources =
        bn_transformer_gpu_resolve_moe_decode_resources(resident_backend, 0);
    assert(decode_resources.router == (void *)2);
    assert(decode_resources.gate_all == (void *)3);
    assert(decode_resources.up_all == (void *)4);
    assert(decode_resources.down_all == (void *)5);
    assert(decode_resources.has_router);
    assert(decode_resources.resident_valid);
    assert(bn_transformer_gpu_moe_decode_cacheable(&gpu,
        &c, &moe_w, resident_backend));
    c.policy_flags |= BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT;
    assert(!bn_transformer_gpu_moe_decode_cacheable(&gpu,
        &c, &moe_w, resident_backend));
    assert(bn_backend_model_register_handle(
               resident_backend, 0, BN_BACKEND_HANDLE_MOE_ROUTER_SCALE,
               (void *)6) == 0);
    assert(bn_transformer_gpu_moe_decode_cacheable(&gpu,
        &c, &moe_w, resident_backend));
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT;
    moe_layers[0].moe.expert_map.down_cols = 4095;
    assert(!bn_transformer_gpu_moe_decode_cacheable(&gpu,
        &c, &moe_w, resident_backend));
    moe_layers[0].moe.expert_map.down_cols = 4096;
    BnBackendModel *router_diff_backend = bn_backend_model_create();
    assert(router_diff_backend);
    assert(bn_backend_model_register_handle(
               router_diff_backend, 0, BN_BACKEND_HANDLE_MOE_ROUTER_DIFF,
               (void *)6) == 0);
    assert(bn_backend_model_register_handle(
               router_diff_backend, 0, BN_BACKEND_HANDLE_MOE_GATE_ALL,
               (void *)7) == 0);
    assert(bn_backend_model_register_handle(
               router_diff_backend, 0, BN_BACKEND_HANDLE_MOE_UP_ALL,
               (void *)8) == 0);
    assert(bn_backend_model_register_handle(
               router_diff_backend, 0, BN_BACKEND_HANDLE_MOE_DOWN_ALL,
               (void *)9) == 0);
    decode_resources =
        bn_transformer_gpu_resolve_moe_decode_resources(
            router_diff_backend, 0);
    assert(!decode_resources.router);
    assert(decode_resources.router_diff == (void *)6);
    assert(decode_resources.has_router);
    assert(decode_resources.resident_valid);
    assert(bn_transformer_gpu_moe_decode_cacheable(&gpu,
        &c, &moe_w, router_diff_backend));
    bn_backend_model_free(router_diff_backend);
    bn_backend_model_free(resident_backend);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_safe_default(
        &gpu, &c, &moe_w));
    assert(bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_fallback_enabled(
        &gpu, &c, &moe_w));
    BnTransformerGPUCPUFallbackPolicy fallback =
        {-1, -1, -1, -1, -1, -1, -1};
    fallback = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        fallback, &gpu, &c, &moe_w);
    assert(fallback.attn_from_layer == 0);
    fallback = (BnTransformerGPUCPUFallbackPolicy)
        {-1, -1, 3, -1, -1, -1, -1};
    fallback = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        fallback, &gpu, &c, &moe_w);
    assert(fallback.attn_layer == 3);
    assert(fallback.attn_from_layer == -1);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_layer_enabled(
        &gpu, &c, &moe_layers[0], c.dim));
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_TOPK");
    unsetenv("BN_GPU_ENABLE_MOE_ROUTER_GPU");
    BnTransformerGPUMoERouteLayerPolicy route_layers = {-1, -1};
    BnTransformerGPUMoEDecodeRoutePolicy route_policy =
        bn_transformer_gpu_moe_decode_route_policy(
            &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
            (void *)2, (void *)3, (void *)4, (void *)5, (void *)6);
    assert(route_policy.all_active_two_kquant_moe);
    assert(!route_policy.route_layer_selected);
    assert(!route_policy.reference_gpu_route);
    assert(route_policy.router == (void *)2);
    assert(!route_policy.gpu_route_topk);
    assert(!route_policy.gpu_routed_ffn);
    assert(!bn_transformer_gpu_moe_route_topk_enabled(
        &gpu, (void *)2, 1, 0));
    assert(bn_transformer_gpu_moe_routed_ffn_enabled(
        &gpu, 0, 1, (void *)4, (void *)5, (void *)6, &map, &c, 2048));
    assert(route_policy.route_flags == 0);
    assert(bn_transformer_gpu_moe_route_normalization_flags(&c) == 0);
    BnTransformerGPUMoEDecodeDispatchPolicy dispatch_policy =
        bn_transformer_gpu_moe_decode_dispatch_policy(
            &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
            (void *)2, (void *)3, (void *)4, (void *)5, (void *)6);
    assert(!dispatch_policy.direct_route.enabled);
    assert(dispatch_policy.requires_session_state);
    assert(!dispatch_policy.route_profile_enabled);
    assert(!dispatch_policy.decode_route.gpu_routed_ffn);
    assert(dispatch_policy.decode_route.router == (void *)2);
    setenv("BN_GPU_MOE_ROUTE_PROFILE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    dispatch_policy = bn_transformer_gpu_moe_decode_dispatch_policy(
        &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
        (void *)2, (void *)3, (void *)4, (void *)5, (void *)6);
    assert(!dispatch_policy.direct_route.enabled);
    assert(dispatch_policy.requires_session_state);
    assert(dispatch_policy.route_profile_enabled);
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE");
    test_gpu_runtime_refresh(&gpu);
    c.moe_norm_topk_prob = 0;
    route_policy = bn_transformer_gpu_moe_decode_route_policy(
        &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
        (void *)2, (void *)3, (void *)4, (void *)5, (void *)6);
    assert(bn_transformer_gpu_moe_route_normalization_flags(&c) ==
           BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM);
    assert(route_policy.route_flags ==
           bn_transformer_gpu_moe_route_normalization_flags(&c));
    assert(bn_transformer_gpu_moe_route_normalization_flags(NULL) ==
           BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM);
    c.moe_norm_topk_prob = 1;
    setenv("BN_GPU_ENABLE_MOE_ROUTER_GPU", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    route_policy = bn_transformer_gpu_moe_decode_route_policy(
        &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
        (void *)2, (void *)3, (void *)4, (void *)5, (void *)6);
    assert(route_policy.route_layer_selected);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_route_layer_selected(
        &gpu, 0, -1, -1));
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_reference_gpu_route_enabled(
        &gpu, 1, 1));
    assert(route_policy.router == (void *)3);
    assert(route_policy.gpu_route_topk);
    assert(!route_policy.cpu_route_resident_ffn);
    assert(route_policy.gpu_routed_ffn);
    gpu.max_moe_route_experts = 1;
    route_policy = bn_transformer_gpu_moe_decode_route_policy(
        &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
        (void *)2, (void *)3, (void *)4, (void *)5, (void *)6);
    assert(!route_policy.gpu_route_topk);
    assert(route_policy.cpu_route_resident_ffn);
    assert(!route_policy.gpu_routed_ffn);
    gpu.max_moe_route_experts = 0;
    c.policy_flags |= BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT;
    route_policy = bn_transformer_gpu_moe_decode_route_policy(
        &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
        (void *)2, (void *)3, (void *)4, (void *)5, (void *)6);
    assert(!route_policy.gpu_route_topk);
    assert(route_policy.cpu_route_resident_ffn);
    assert(route_policy.gpu_routed_ffn);
    float router_scale = 1.0f;
    moe_layers[0].moe.router_scale = &router_scale;
    route_policy = bn_transformer_gpu_moe_decode_route_policy(
        &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
        (void *)2, (void *)3, (void *)4, (void *)5, (void *)6);
    assert(route_policy.uses_scaled_router_input);
    assert(route_policy.gpu_route_topk);
    assert(!route_policy.cpu_route_resident_ffn);
    assert(route_policy.gpu_routed_ffn);
    moe_layers[0].moe.router_scale = NULL;
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT;
    assert(bn_transformer_gpu_moe_route_topk_enabled(
        &gpu, (void *)2, 1, 1));
    assert(bn_transformer_gpu_moe_routed_ffn_enabled(
        &gpu, 1, 0, (void *)4, (void *)5, (void *)6, &map, &c, 2048));
    unsetenv("BN_GPU_ENABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    unsetenv("BN_CUDA_DISABLE_MOE_FFN");
    fallback = (BnTransformerGPUCPUFallbackPolicy)
        {-1, -1, -1, -1, -1, -1, -1};
    BnTransformerGPUMoEFFNFallbackPolicy moe_ffn_fallback =
        bn_transformer_gpu_moe_ffn_fallback_policy(
            &gpu, &c, &map, c.dim, 1, 0, &fallback);
    assert(moe_ffn_fallback.use_cpu);
    assert(bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 0, -1, -1));
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_reference_gpu_route_enabled(
        &gpu, 1, 1));
    moe_ffn_fallback = bn_transformer_gpu_moe_ffn_fallback_policy(
        &gpu, &c, &map, c.dim, 1, 0, &fallback);
    assert(!moe_ffn_fallback.use_cpu);
    assert(!bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 0, -1, -1));
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    test_gpu_runtime_refresh(&gpu);
    fallback.ffn_layer = 3;
    moe_ffn_fallback = bn_transformer_gpu_moe_ffn_fallback_policy(
        &gpu, &c, &map, c.dim, 1, 3, &fallback);
    assert(moe_ffn_fallback.use_cpu);
    assert(bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 3, 3, -1));
    fallback.ffn_layer = -1;
    fallback.ffn_from_layer = 2;
    moe_ffn_fallback = bn_transformer_gpu_moe_ffn_fallback_policy(
        &gpu, &c, &map, c.dim, 1, 3, &fallback);
    assert(moe_ffn_fallback.use_cpu);
    assert(bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 3, -1, 2));
    setenv("BN_CUDA_DISABLE_MOE_FFN", "1", 1);
    assert(bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 0, -1, -1));
    unsetenv("BN_CUDA_DISABLE_MOE_FFN");
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ATTN_SAFE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_safe_default(
        &gpu, &c, &moe_w));
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_fallback_enabled(
        &gpu, &c, &moe_w));
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_ATTN_SAFE");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_cpu_moe_safe_default(
        &gpu, &c, &moe_w));
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_moe_safe_default(
        &gpu, &c, &moe_w));
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    test_gpu_runtime_refresh(&gpu);
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_MOE_SAFE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_moe_safe_default(
        &gpu, &c, &moe_w));
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_CPU_MOE_SAFE");
    test_gpu_runtime_refresh(&gpu);
    c.policy_flags = BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ATTENTION;
    assert(bn_transformer_gpu_moe_reference_attention_enabled(&gpu, &c));
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_REFERENCE_ATTN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_reference_attention_enabled(&gpu, &c));
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_REFERENCE_ATTN");
    test_gpu_runtime_refresh(&gpu);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_moe_reference_attention_enabled(&gpu, &c));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    c.policy_flags = 0;

    gpu.moe_route_routed_ffn_batch_norm_resid =
        mock_moe_route_routed_ffn_batch_norm_resid;
    gpu.moe_route_routed_ffn_batch = mock_moe_route_routed_ffn_batch;
    gpu.moe_route_batch = mock_moe_route_batch;
    gpu.moe_routed_ffn_batch = mock_moe_routed_ffn_batch;
    gpu.moe_ffn_batch = mock_moe_ffn_batch;
    gpu.dense_ffn_batch = mock_dense_ffn_batch;
    gpu.prefill_moe_layer = mock_prefill_moe_layer;
    gpu.prefill_ssm_layer = mock_prefill_ssm_layer;
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE");
    test_gpu_runtime_refresh(&gpu);
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_routed_ffn_batch_allowed(&gpu, &c, &map));
    assert(bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(&gpu, &c, &map));
    assert(!bn_transformer_gpu_moe_prefill_route_batch_available(
        &gpu, &c, 1));
    assert(!bn_transformer_gpu_moe_prefill_routed_ffn_batch_available(
        &gpu, &c, &map, c.dim, 0));
    assert(!bn_transformer_gpu_moe_prefill_resident_expert_batch_available(
        &gpu, &c, &map, c.dim, 0, 0));
    assert(!bn_transformer_gpu_moe_prefill_split_expert_batch_available(
        &gpu, &c, &map, c.dim, 0, 0));
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_prefill_routed_ffn_batch_available(
        &gpu, &c, &map, c.dim, 0));
    unsetenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL");
    assert(bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
        &gpu, &c, 1));
    assert(!bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
        &gpu, &c, 0));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
        &gpu, &c, 1));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    c.n_experts_active = 1;
    assert(!bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
        &gpu, &c, 1));
    c.n_experts_active = 2;
    setenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy, "BN_CUDA_DISABLE_MOE_CACHE_PREFILL",
               "1", 1) == 0);
    assert(!bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
        &gpu, &c, 1));
    unsetenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_CUDA_DISABLE_MOE_CACHE_PREFILL");
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
    assert(bn_transformer_gpu_moe_prefill_shared_batch_available(
        &gpu, 1, 1));
    assert(!bn_transformer_gpu_moe_prefill_shared_batch_available(
        &gpu, 1, 0));
    gpu.dense_ffn_batch = NULL;
    assert(!bn_transformer_gpu_moe_prefill_shared_batch_available(
        &gpu, 1, 1));
    gpu.dense_ffn_batch = mock_dense_ffn_batch;
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_moe_prefill_shared_batch_available(
        &gpu, 1, 1));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "4", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_prefill_shared_batch_available(
        &gpu, 3, 1));
    assert(bn_transformer_gpu_moe_prefill_shared_batch_available(
        &gpu, 4, 1));
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    test_gpu_runtime_refresh(&gpu);
    setenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy,
               "BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE", "1", 1) == 0);
    assert(!bn_transformer_gpu_moe_prefill_shared_batch_available(
        &gpu, 1, 1));
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
    c.has_shared_expert = 1;
    moe_layers[0].shared.shared_gate.data = (void *)7;
    assert(bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
        &gpu, &c, &moe_layers[0], 1));
    assert(!bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
        &gpu, &c, &moe_layers[0], 0));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
        &gpu, &c, &moe_layers[0], 1));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    c.has_shared_expert = 0;
    assert(!bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
        &gpu, &c, &moe_layers[0], 1));
    c.has_shared_expert = 1;
    moe_layers[0].shared.shared_gate.data = NULL;
    assert(!bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
        &gpu, &c, &moe_layers[0], 1));
    moe_layers[0].shared.shared_gate.data = (void *)7;
    setenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy,
               "BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE", "1", 1) == 0);
    assert(!bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
        &gpu, &c, &moe_layers[0], 1));
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
    c.has_shared_expert = 0;
    moe_layers[0].shared.shared_gate.data = NULL;
    assert(bn_transformer_gpu_moe_prefill_resident_expert_batch_available(
        &gpu, &c, &map, c.dim, 0, 0));
    assert(!bn_transformer_gpu_moe_prefill_resident_expert_batch_available(
        &gpu, &c, &map, c.dim, 0, 1));
    assert(bn_transformer_gpu_moe_prefill_split_expert_batch_available(
        &gpu, &c, &map, c.dim, 0, 0));
    assert(!bn_transformer_gpu_moe_prefill_split_expert_batch_available(
        &gpu, &c, &map, c.dim, 0, 1));
    assert(bn_transformer_gpu_moe_prefill_single_expert_batch_available(
        &gpu, 1));
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "4", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_prefill_single_expert_batch_available(
        &gpu, 3));
    assert(bn_transformer_gpu_moe_prefill_single_expert_batch_available(
        &gpu, 4));
    gpu.dense_ffn_batch = NULL;
    assert(!bn_transformer_gpu_moe_prefill_single_expert_batch_available(
        &gpu, 4));
    gpu.dense_ffn_batch = mock_dense_ffn_batch;
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_moe_prefill_single_expert_batch_available(
        &gpu, 4));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    test_gpu_runtime_refresh(&gpu);
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    c.n_experts = 3;
    assert(!bn_transformer_gpu_moe_routed_ffn_batch_allowed(&gpu, &c, &map));
    gpu.caps |= BN_GPU_CAP_MOE_COMBINED_PREFILL_DEFAULT;
    assert(!bn_transformer_gpu_moe_routed_ffn_batch_allowed(&gpu, &c, &map));
    assert(!bn_transformer_gpu_prefill_moe_layer_backend_available(
        &gpu, &c, &map, c.dim, 0));
    BnMoEExpertMap native_map = map;
    native_map.gate_type = native_map.up_type = native_map.down_type = BN_GGUF_TENSOR_Q8_0;
    assert(bn_transformer_gpu_moe_routed_ffn_batch_allowed(&gpu, &c, &native_map));
    assert(bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(
        &gpu, &c, &native_map));
    gpu.caps &= ~BN_GPU_CAP_MOE_COMBINED_PREFILL_DEFAULT;
    assert(bn_transformer_gpu_moe_prefill_route_batch_available(
        &gpu, &c, 1));
    assert(!bn_transformer_gpu_moe_prefill_route_batch_available(
        &gpu, &c, 0));
    assert(!bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(&gpu, &c, &map));
    setenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_moe_routed_ffn_batch_allowed(&gpu, &c, &map));
    assert(bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(&gpu, &c, &map));
    assert(bn_transformer_gpu_moe_prefill_routed_ffn_batch_available(
        &gpu, &c, &map, c.dim, 0));
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_moe_routed_ffn_batch_allowed(&gpu, &c, &map));
    assert(!bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(&gpu, &c, &map));
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE");
    test_gpu_runtime_refresh(&gpu);
    c.n_experts = 2;
    assert(bn_transformer_gpu_prefill_ssm_layer_backend_available(&gpu));
    assert(!bn_transformer_gpu_prefill_moe_ffn_batch_available(
        &gpu, &c, &map, c.dim, 0));
    assert(!bn_transformer_prefill_moe_layer_backend_available(
        &gpu, &c, &map, c.dim, 0));
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_prefill_moe_ffn_batch_available(
        &gpu, &c, &map, c.dim, 0));
    assert(bn_transformer_prefill_moe_layer_backend_available(
        &gpu, &c, &map, c.dim, 0));
    setenv("BN_CUDA_DISABLE_MOE_PREFILL", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_prefill_moe_layer_backend_available(
        &gpu, &c, &map, c.dim, 0));
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_prefill_moe_layer_backend_available(
        &gpu, &c, &map, c.dim, 0));
    setenv("BN_GPU_DISABLE_PREFILL_MATMUL", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_prefill_moe_layer_backend_available(
        &gpu, &c, &map, c.dim, 0));
    unsetenv("BN_GPU_DISABLE_PREFILL_MATMUL");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_prefill_moe_layer_chain_available(
        &gpu, &c, &map, c.dim, 0, 0));
    assert(bn_transformer_gpu_prefill_moe_layer_chain_available(
        &gpu, &c, &map, c.dim, 0, 1));
    assert(!bn_transformer_gpu_prefill_ssm_moe_chain_available(
        &gpu, &c, &map, c.dim, 0, 0));
    assert(bn_transformer_gpu_prefill_ssm_moe_chain_available(
        &gpu, &c, &map, c.dim, 0, 1));
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_prefill_ssm_moe_chain_available(
        &gpu, &c, &map, c.dim, 0, 1));
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_prefill_moe_ffn_batch_available(
        &gpu, &c, &map, c.dim, 0));
    assert(bn_transformer_prefill_moe_layer_backend_available(
        &gpu, &c, &map, c.dim, 0));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    gpu.prefill_moe_layer = NULL;
    assert(!bn_transformer_prefill_moe_layer_backend_available(
        &gpu, &c, &map, c.dim, 0));
    gpu.prefill_moe_layer = mock_prefill_moe_layer;
    gpu.prefill_ssm_layer = NULL;
    assert(!bn_transformer_gpu_prefill_ssm_layer_backend_available(&gpu));
    gpu.prefill_ssm_layer = mock_prefill_ssm_layer;
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");

    BnWeights dense_w;
    BnLayerWeights dense_layers[1];
    memset(&dense_w, 0, sizeof(dense_w));
    memset(dense_layers, 0, sizeof(dense_layers));
    dense_w.layers = dense_layers;
    dense_w.emb_type = BN_GGUF_TENSOR_Q8_0;
    c.n_experts = 0;
    c.n_experts_active = 0;
    c.moe_intermediate_size = 0;
    c.dim = 2048;
    c.policy_flags = BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY;
    gpu.kind = BN_GPU_BACKEND_METAL;
    gpu.caps |= BN_GPU_CAP_REFERENCE_ATTENTION_FALLBACK;
    assert(!bn_transformer_gpu_reference_attention_cpu_fallback_enabled(
        &gpu, &c));
    c.policy_flags |= BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION;
    assert(bn_transformer_gpu_reference_attention_cpu_fallback_enabled(
        &gpu, &c));
    c.nextn_predict_layers = 1;
    assert(!bn_transformer_gpu_reference_attention_cpu_fallback_enabled(
        &gpu, &c));
    assert(bn_model_transformer_policy_has_auxiliary_prediction_blocks(&c));
    assert(!bn_transformer_gpu_can_borrowed_pair_gateup_silu(
        &gpu, &c, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0,
        BN_MODEL_ACTIVATION_SILU));
    gpu.caps |= BN_GPU_CAP_NATIVE_QUANT_FUSED_GATEUP_SILU;
    assert(bn_transformer_gpu_can_borrowed_pair_gateup_silu(
        &gpu, &c, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0,
        BN_MODEL_ACTIVATION_SILU));
    assert(!bn_transformer_gpu_can_borrowed_pair_gateup_silu(
        &gpu, &c, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0,
        BN_MODEL_ACTIVATION_SILU));
    c.nextn_predict_layers = 0;
    assert(!bn_model_transformer_policy_has_auxiliary_prediction_blocks(&c));
    assert(!bn_transformer_gpu_can_borrowed_pair_gateup_silu(
        &gpu, &c, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0,
        BN_MODEL_ACTIVATION_SILU));
    assert(!bn_transformer_gpu_reference_attention_exact_enabled(&gpu, &c));
    gpu.caps |= BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH;
    assert(!bn_transformer_gpu_reference_attention_cpu_fallback_enabled(
        &gpu, &c));
    assert(bn_transformer_gpu_reference_attention_no_logits_cpu_fallback_enabled(
        &gpu, &c, 0));
    assert(!bn_transformer_gpu_reference_attention_no_logits_cpu_fallback_enabled(
        &gpu, &c, 1));
    assert(!bn_transformer_gpu_reference_attention_exact_enabled(&gpu, &c));
    gpu.caps &= ~BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH;
    gpu.caps |= BN_GPU_CAP_REFERENCE_ATTENTION_TOKEN_FALLBACK;
    assert(!bn_transformer_gpu_reference_attention_cpu_fallback_enabled(
        &gpu, &c));
    BnTransformerGPUDecodeEntryPolicy reference_entry =
        bn_transformer_gpu_decode_entry_policy(&gpu, &c, &dense_w, 0);
    assert(reference_entry.block_forward);
    gpu.caps |= BN_GPU_CAP_REFERENCE_ATTENTION;
    reference_entry =
        bn_transformer_gpu_decode_entry_policy(&gpu, &c, &dense_w, 0);
    assert(reference_entry.block_forward);
    assert(!bn_transformer_gpu_reference_attention_exact_enabled(&gpu, &c));
    c.policy_flags |= BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY;
    reference_entry =
        bn_transformer_gpu_decode_entry_policy(&gpu, &c, &dense_w, 0);
    assert(reference_entry.block_forward);
    assert(!bn_transformer_gpu_reference_attention_exact_enabled(&gpu, &c));
    gpu.caps |= BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH;
    reference_entry =
        bn_transformer_gpu_decode_entry_policy(&gpu, &c, &dense_w, 0);
    assert(!reference_entry.block_forward);
    assert(bn_transformer_gpu_reference_attention_exact_enabled(&gpu, &c));
    int saved_n_layers = c.n_layers;
    c.n_layers = 36;
    c.n_experts = 0;
    c.policy_flags |= BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION_LAST_THIRD;
    BnGPUBackendKind saved_backend_kind = gpu.kind;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_tokenwise_reference_attention_from_layer(
               &gpu, &c, 0) == 24);
    assert(bn_transformer_gpu_tokenwise_reference_attention_from_layer(
               &gpu, &c, 1) == -1);
    gpu.kind = saved_backend_kind;
    c.n_layers = saved_n_layers;
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION_LAST_THIRD;
    assert(!bn_transformer_gpu_reference_attention_no_logits_cpu_fallback_enabled(
        &gpu, &c, 0));
    gpu.caps &= ~BN_GPU_CAP_REFERENCE_ATTENTION;
    gpu.caps &= ~BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH;
    gpu.caps &= ~BN_GPU_CAP_REFERENCE_ATTENTION_TOKEN_FALLBACK;
    c.policy_flags |= BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    gpu.caps |= BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL;
    assert(!bn_transformer_gpu_reference_attention_cpu_fallback_enabled(
        &gpu, &c));
    c.policy_flags |= BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
    fallback = (BnTransformerGPUCPUFallbackPolicy)
        {-1, -1, -1, -1, -1, -1, -1};
    fallback = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        fallback, &gpu, &c, &dense_w);
    assert(fallback.attn_layer == -1);
    assert(fallback.attn_from_layer == -1);
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
    gpu.caps &= ~BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL;
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_reference_recurrent_exact_enabled(&gpu, &c));
    gpu.caps |= BN_GPU_CAP_REFERENCE_RECURRENT;
    assert(bn_transformer_gpu_reference_recurrent_exact_enabled(&gpu, &c));
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT;
    assert(!bn_transformer_gpu_reference_recurrent_exact_enabled(&gpu, &c));
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY;
    fallback = (BnTransformerGPUCPUFallbackPolicy)
        {-1, -1, -1, -1, -1, -1, -1};
    fallback = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        fallback, &gpu, &c, &dense_w);
    assert(fallback.attn_from_layer == 0);
    gpu.caps |= BN_GPU_CAP_REFERENCE_ATTENTION;
    assert(bn_transformer_gpu_reference_attention_cpu_fallback_enabled(
        &gpu, &c));
    assert(!bn_transformer_gpu_reference_attention_exact_enabled(&gpu, &c));
    c.policy_flags |= BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY;
    assert(bn_transformer_gpu_reference_attention_cpu_fallback_enabled(
        &gpu, &c));
    assert(!bn_transformer_gpu_reference_attention_exact_enabled(&gpu, &c));
    gpu.caps &= ~BN_GPU_CAP_REFERENCE_ATTENTION;
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION;
    assert(!bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
        &gpu, &c, &dense_w));
    assert(!bn_transformer_gpu_small_dense_native_quant_default(
        &gpu, &c, -1));
    assert(!bn_transformer_gpu_small_dense_native_quant_ffn_down_enabled(
        &gpu, &c));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
        &gpu, &c, &dense_w));
    fallback = (BnTransformerGPUCPUFallbackPolicy)
        {-1, -1, -1, -1, -1, -1, -1};
    fallback = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        fallback, &gpu, &c, &dense_w);
    assert(fallback.attn_from_layer == 0);
    assert(bn_transformer_gpu_small_dense_native_quant_default(
        &gpu, &c, -1));
    BnTransformerGPUSmallDenseNativeQuantLayerPolicy small_dense_native_quant_layer =
        {.from_layer = -1, .to_layer = -1};
    BnTransformerGPUSmallDenseNativeQuantDecodePolicy small_dense_native_quant_decode =
        bn_transformer_gpu_small_dense_native_quant_decode_policy(&gpu, &c, &small_dense_native_quant_layer);
    assert(small_dense_native_quant_decode.small_dense_native_quant_default);
    c.n_layers = 61;
    assert(bn_transformer_small_dense_native_quant_to_layer(&c) == 27);
    small_dense_native_quant_decode =
        bn_transformer_gpu_small_dense_native_quant_decode_policy(&gpu, &c, &small_dense_native_quant_layer);
    assert(small_dense_native_quant_decode.small_dense_native_quant_default);
    assert(small_dense_native_quant_decode.small_dense_native_quant_to_layer == 27);
    assert(bn_transformer_gpu_small_dense_native_quant_to_layer(
               &c, 1, -1) == 27);
    small_dense_native_quant_layer.to_layer = 9;
    small_dense_native_quant_decode =
        bn_transformer_gpu_small_dense_native_quant_decode_policy(&gpu, &c, &small_dense_native_quant_layer);
    assert(small_dense_native_quant_decode.small_dense_native_quant_to_layer == 9);
    assert(bn_transformer_gpu_small_dense_native_quant_to_layer(
               &c, 1, 9) == 9);
    small_dense_native_quant_layer.to_layer = -1;
    c.n_layers = 33;
    assert(bn_transformer_small_dense_native_quant_to_layer(&c) == -1);
    small_dense_native_quant_decode =
        bn_transformer_gpu_small_dense_native_quant_decode_policy(&gpu, &c, &small_dense_native_quant_layer);
    assert(small_dense_native_quant_decode.small_dense_native_quant_to_layer == -1);
    assert(bn_transformer_gpu_small_dense_native_quant_to_layer(
               &c, 1, -1) == -1);
    c.n_layers = 0;
    assert(bn_transformer_gpu_small_dense_native_quant_to_layer(
               &c, 0, -1) == -1);
    assert(!bn_transformer_gpu_small_dense_native_quant_default(
        &gpu, &c, 0));
    small_dense_native_quant_layer.from_layer = 0;
    small_dense_native_quant_decode =
        bn_transformer_gpu_small_dense_native_quant_decode_policy(&gpu, &c, &small_dense_native_quant_layer);
    assert(!small_dense_native_quant_decode.small_dense_native_quant_default);
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_CPU_ATTN_SAFE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
        &gpu, &c, &dense_w));
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_CPU_ATTN_SAFE");
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_small_dense_native_quant_default(
        &gpu, &c, -1));
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT");
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_FFN_DOWN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_small_dense_native_quant_ffn_down_enabled(
        &gpu, &c));
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_FFN_DOWN");
    test_gpu_runtime_refresh(&gpu);
    c.policy_flags |= BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK;
    assert(!bn_transformer_gpu_small_dense_prefill_decode_fallback_requested(
        &gpu, &c));
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_PREFILL", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_small_dense_prefill_decode_fallback_requested(
        &gpu, &c));
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_PREFILL");
    assert(!bn_transformer_gpu_native_quant_logits_refine_enabled(
        &gpu, &c, BN_GGUF_TENSOR_Q8_0));
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_native_quant_logits_refine_enabled(
        &gpu, &c, BN_GGUF_TENSOR_Q8_0));
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_native_quant_logits_refine_enabled(
        &gpu, &c, BN_GGUF_TENSOR_Q8_0));
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");
    c.policy_flags = 0;

    BnLayerWeights logits_refine_layer;
    memset(&logits_refine_layer, 0, sizeof(logits_refine_layer));
    BnWeights logits_refine_weights;
    memset(&logits_refine_weights, 0, sizeof(logits_refine_weights));
    logits_refine_weights.layers = &logits_refine_layer;
    c.dim = 2048;
    c.n_layers = 1;
    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    c.moe_norm_topk_prob = 1;
    logits_refine_layer.moe.router_weight = (void *)1;
    logits_refine_layer.moe.expert_map.gate_type = BN_GGUF_TENSOR_Q4_K;
    logits_refine_layer.moe.expert_map.up_type = BN_GGUF_TENSOR_Q4_K;
    logits_refine_layer.moe.expert_map.down_type = BN_GGUF_TENSOR_Q6_K;
    logits_refine_layer.moe.expert_map.gate_rows = c.moe_intermediate_size;
    logits_refine_layer.moe.expert_map.gate_cols = c.dim;
    logits_refine_layer.moe.expert_map.up_rows = c.moe_intermediate_size;
    logits_refine_layer.moe.expert_map.up_cols = c.dim;
    logits_refine_layer.moe.expert_map.down_rows = c.dim;
    logits_refine_layer.moe.expert_map.down_cols = c.moe_intermediate_size;
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_logits_refine_default(
        &gpu, &c, &logits_refine_weights));
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_logits_refine_default(
        &gpu, &c, &logits_refine_weights));
    setenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_logits_refine_default(
        &gpu, &c, &logits_refine_weights));
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    unsetenv("BN_CUDA_DISABLE_ALL_ACTIVE_TWO_KQUANT_MOE_LOGITS_REFINE");
    test_gpu_runtime_refresh(&gpu);
    c.dim = 2048;
    c.n_layers = 0;
    c.n_experts = 0;
    c.n_experts_active = 0;
    c.moe_intermediate_size = 0;
    c.moe_norm_topk_prob = 0;

    BnWeights hybrid_w;
    memset(&hybrid_w, 0, sizeof(hybrid_w));
    c.dim = 4096;
    c.full_attn_interval = 4;
    c.ssm_inner_size = 128;
    setenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy,
               "BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE", "1", 1) == 0);
    gpu.kind = BN_GPU_BACKEND_METAL;
    gpu.caps &= ~(BN_GPU_CAP_LARGE_GRAPH_NATIVE | BN_GPU_CAP_SSM_GRAPH);
    assert(!bn_transformer_gpu_large_hybrid_cpu_attn_safe_fallback_enabled(
        &gpu, &c, &hybrid_w));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    gpu.caps |= BN_GPU_CAP_LARGE_GRAPH_NATIVE | BN_GPU_CAP_SSM_GRAPH;
    assert(bn_transformer_gpu_large_hybrid_cpu_attn_safe_default(
        &gpu, &c, &hybrid_w));
    assert(bn_transformer_gpu_large_hybrid_cpu_attn_safe_fallback_enabled(
        &gpu, &c, &hybrid_w));
    BnTransformerGPUDecodeEntryPolicy decode_entry =
        bn_transformer_gpu_decode_entry_policy(&gpu, &c, &hybrid_w, 1);
    assert(decode_entry.block_argmax);
    decode_entry =
        bn_transformer_gpu_decode_entry_policy(&gpu, &c, &hybrid_w, 0);
    assert(!decode_entry.block_argmax);
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy, "BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX",
               "1", 1) == 0);
    decode_entry =
        bn_transformer_gpu_decode_entry_policy(&gpu, &c, &hybrid_w, 1);
    assert(!decode_entry.block_argmax);
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX");
    fallback = (BnTransformerGPUCPUFallbackPolicy)
        {-1, -1, -1, -1, -1, -1, -1};
    fallback = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        fallback, &gpu, &c, &hybrid_w);
    assert(fallback.attn_from_layer == 0);
    unsetenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy,
        "BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE");
    assert(!bn_transformer_gpu_ssm_cpu_fallback_required(&gpu));
    BnTransformerGPUSSMFallbackPolicy ssm_fallback =
        bn_transformer_gpu_ssm_fallback_policy(&gpu);
    assert(!ssm_fallback.use_cpu);
    setenv("BN_CUDA_DISABLE_SSM_GRAPH", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy, "BN_CUDA_DISABLE_SSM_GRAPH", "1", 1) ==
           0);
    assert(bn_transformer_gpu_ssm_cpu_fallback_required(&gpu));
    ssm_fallback = bn_transformer_gpu_ssm_fallback_policy(&gpu);
    assert(ssm_fallback.use_cpu);
    unsetenv("BN_CUDA_DISABLE_SSM_GRAPH");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_CUDA_DISABLE_SSM_GRAPH");
    gpu.kind = BN_GPU_BACKEND_METAL;
    setenv("BN_CUDA_DISABLE_SSM_GRAPH", "1", 1);
    assert(!bn_transformer_gpu_ssm_cpu_fallback_required(&gpu));
    setenv("BN_GPU_DISABLE_SSM_GRAPH", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy, "BN_GPU_DISABLE_SSM_GRAPH", "1", 1) ==
           0);
    assert(bn_transformer_gpu_ssm_cpu_fallback_required(&gpu));
    unsetenv("BN_GPU_DISABLE_SSM_GRAPH");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_GPU_DISABLE_SSM_GRAPH");
    unsetenv("BN_CUDA_DISABLE_SSM_GRAPH");
    gpu.caps &= ~BN_GPU_CAP_SSM_GRAPH;
    assert(bn_transformer_gpu_ssm_cpu_fallback_required(&gpu));
    ssm_fallback = bn_transformer_gpu_ssm_fallback_policy(&gpu);
    assert(ssm_fallback.use_cpu);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    c.full_attn_interval = 0;
    c.ssm_inner_size = 0;
    c.dim = 2048;

    c.seq_len = 32;
    c.kv_f16 = 0;
    unsetenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV");
    unsetenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK");
    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    assert(bn_transformer_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    setenv("BN_GPU_CPU_FALLBACK_LAYER", "0", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    assert(!bn_transformer_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    setenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    assert(bn_transformer_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    unsetenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK");
    test_gpu_runtime_refresh(&gpu);
    c.kv_f16 = 1;
    assert(!bn_transformer_gpu_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    assert(!bn_transformer_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    c.kv_f16 = 0;
    assert(!bn_transformer_gpu_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 24, 16));
    assert(!bn_transformer_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 24, 16));
    setenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    assert(!bn_transformer_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    unsetenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV");

    W.type = BN_GGUF_TENSOR_Q6_K;
    W.rows = 300000;
    W.cols = 2048;
    logits.type = BN_GGUF_TENSOR_Q6_K;
    logits.rows = W.rows;
    logits.cols = W.cols;
    logits.cpu_weight = &W;
    gpu.matvec_argmax_activation = mock_matvec_argmax_activation;
    gpu.argmax_activation = mock_argmax_activation;
    gpu.max_storage_binding_size = bn_qweight_data_size(&W);
    BnTransformerGPUGenerateArgmaxPolicy generate_argmax =
        bn_transformer_gpu_generate_argmax_policy(&gpu, 0, 0.0f, 1.0f);
    assert(generate_argmax.enabled);
    setenv("BN_GPU_CPU_LOGITS", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    generate_argmax =
        bn_transformer_gpu_generate_argmax_policy(&gpu, 0, 0.0f, 1.0f);
    assert(!generate_argmax.enabled);
    unsetenv("BN_GPU_CPU_LOGITS");
    test_gpu_runtime_refresh(&gpu);
    generate_argmax =
        bn_transformer_gpu_generate_argmax_policy(NULL, 0, 0.0f, 1.0f);
    assert(!generate_argmax.enabled);
    gpu.argmax_activation = NULL;
    generate_argmax =
        bn_transformer_gpu_generate_argmax_policy(&gpu, 0, 0.0f, 1.0f);
    assert(!generate_argmax.enabled);
    gpu.argmax_activation = mock_argmax_activation;
    generate_argmax =
        bn_transformer_gpu_generate_argmax_policy(&gpu, 1, 0.0f, 1.0f);
    assert(!generate_argmax.enabled);
    generate_argmax =
        bn_transformer_gpu_generate_argmax_policy(&gpu, 0, 0.7f, 1.0f);
    assert(!generate_argmax.enabled);
    generate_argmax =
        bn_transformer_gpu_generate_argmax_policy(&gpu, 0, 0.0f, 0.9f);
    assert(!generate_argmax.enabled);
    assert(bn_transformer_gpu_argmax_available(&gpu, 0));
    assert(bn_transformer_gpu_argmax_available(&gpu, 1));
    assert(bn_transformer_gpu_argmax_available(NULL, 0));
    gpu.argmax_activation = NULL;
    assert(!bn_transformer_gpu_argmax_available(&gpu, 1));
    assert(!bn_transformer_gpu_argmax_available(NULL, 1));
    gpu.argmax_activation = mock_argmax_activation;
    float write_tmp[2] = {1.0f, 2.0f};
    assert(bn_transformer_gpu_write_activation_buf(
        NULL, BN_GPU_VALUE_X, write_tmp, sizeof(write_tmp)) != 0);
    gpu.write_activation = mock_gpu_write_activation;
    assert(bn_transformer_gpu_write_activation_buf(
        &gpu, BN_GPU_VALUE_X, write_tmp, sizeof(write_tmp)) == 0);
    gpu.write_activation = NULL;
    assert(bn_transformer_gpu_write_activation_buf(
        &gpu, BN_GPU_VALUE_X, write_tmp, sizeof(write_tmp)) != 0);
    gpu.write_activation = mock_gpu_write_activation;
    assert(bn_transformer_gpu_write_activation_buf(
        &gpu, BN_GPU_VALUE_X, NULL, sizeof(write_tmp)) != 0);
    assert(bn_transformer_gpu_write_activation_buf_offset(
        NULL, BN_GPU_VALUE_X, write_tmp, sizeof(write_tmp), 0) != 0);
    assert(bn_transformer_gpu_write_activation_buf_offset(
        &gpu, BN_GPU_VALUE_X, write_tmp, sizeof(write_tmp), 4) == 0);
    gpu.write_activation = NULL;
    assert(bn_transformer_gpu_write_activation_buf_offset(
        &gpu, BN_GPU_VALUE_X, write_tmp, sizeof(write_tmp), 4) != 0);
    gpu.write_activation = mock_gpu_write_activation;

    c.n_experts = 0;
    unsetenv("BN_GPU_CPU_LOGITS");
    unsetenv("BN_CUDA_DISABLE_LOGITS_ARGMAX");
    unsetenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX");
    unsetenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX");
    unsetenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    BnTransformerGPULogitsDispatchPolicy logits_dispatch =
        bn_transformer_gpu_logits_dispatch_policy(
            &gpu, &c, &logits, 1, 0);
    assert(!logits_dispatch.needs_cpu_fallback);
    assert(!logits_dispatch.cpu_logits_enabled);
    assert(logits_dispatch.use_matvec_argmax);
    logits_dispatch = bn_transformer_gpu_logits_dispatch_policy(
        &gpu, &c, &logits, 1, 1);
    assert(!logits_dispatch.needs_cpu_fallback);
    assert(!logits_dispatch.cpu_logits_enabled);
    assert(!logits_dispatch.use_matvec_argmax);
    W.rows = 1024;
    logits.rows = W.rows;
    assert(!bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    setenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    unsetenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX");
    test_gpu_runtime_refresh(&gpu);
    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    assert(bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    c.moe_intermediate_size = 4095;
    assert(!bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    c.moe_intermediate_size = 4096;
    assert(!bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 1, 0));
    c.n_experts_active = 1;
    c.moe_intermediate_size = 4095;
    logits.cols = 1536;
    assert(bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    logits.cols = 2048;
    assert(!bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    setenv("BN_GPU_CPU_LOGITS", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    logits_dispatch = bn_transformer_gpu_logits_dispatch_policy(
        &gpu, &c, &logits, 1, 0);
    assert(!logits_dispatch.needs_cpu_fallback);
    assert(logits_dispatch.cpu_logits_enabled);
    assert(!logits_dispatch.use_matvec_argmax);
    unsetenv("BN_GPU_CPU_LOGITS");
    test_gpu_runtime_refresh(&gpu);
    size_t saved_max_storage = gpu.max_storage_binding_size;
    gpu.max_storage_binding_size = bn_qweight_data_size(&W) - 1;
    logits_dispatch = bn_transformer_gpu_logits_dispatch_policy(
        &gpu, &c, &logits, 1, 0);
    assert(logits_dispatch.needs_cpu_fallback);
    assert(logits_dispatch.cpu_logits_enabled);
    assert(!logits_dispatch.use_matvec_argmax);
    gpu.max_storage_binding_size = saved_max_storage;
    setenv("BN_CUDA_DISABLE_LOGITS_ARGMAX", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    unsetenv("BN_CUDA_DISABLE_LOGITS_ARGMAX");
    gpu.matvec_argmax_activation = NULL;
    assert(!bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    gpu.matvec_argmax_activation = mock_matvec_argmax_activation;
    c.n_experts = 0;
    c.n_experts_active = 0;

    setenv("BN_CUDA_ENABLE_LOGITS_CACHE", "1", 1);
    unsetenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_DECODE_CACHE");
    unsetenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE");
    unsetenv("BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT");
    unsetenv("BN_METAL_ENABLE_Q6_Q8K");
    setenv("BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    gpu.caps |= BN_GPU_CAP_DECODE_GRAPH_CACHE;
    BnTransformerGPULogitsRefinePolicy decode_refine = {0};
    BnTransformerGPUCPUFallbackPolicy decode_fallback =
        {-1, -1, -1, -1, -1, -1, -1};
    BnTransformerGPUComparePolicy decode_compare = {
        .attention_layer = -1, .attention_pos = -1,
        .gqa_layer = -1, .gqa_pos = -1,
        .qkv_layer = -1, .qkv_pos = -1,
        .ffn_down_layer = -1, .ffn_down_pos = -1,
        .ffn_state_layer = -1, .ffn_state_pos = -1,
        .ssm_layer = -1, .ssm_pos = -1,
    };
    BnTransformerGPUDecodeCacheabilityPolicy decode_cacheability =
        bn_transformer_gpu_decode_cacheability_policy(
            &gpu, &c, NULL, NULL, 1, 0, 0, 0, &decode_refine, 0,
            &decode_fallback, &decode_compare);
    assert(!decode_cacheability.resident_moe);
    assert(decode_cacheability.graph_cacheable);
    assert(bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    decode_fallback.layer = 0;
    decode_cacheability = bn_transformer_gpu_decode_cacheability_policy(
        &gpu, &c, NULL, NULL, 1, 0, 0, 0, &decode_refine, 0,
        &decode_fallback, &decode_compare);
    assert(!decode_cacheability.graph_cacheable);
    decode_fallback.layer = -1;
    unsetenv("BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT");
    setenv("BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy,
               "BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT", "1", 1) == 0);
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    unsetenv("BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_METAL_ENABLE_SPECIALIZED_NATIVE_QUANT");
    setenv("BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT", "1", 1);
    assert(bn_backend_runtime_policy_set(
               &gpu.runtime_policy,
               "BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT", "1", 1) == 0);
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 1, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    setenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 1, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    unsetenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE");
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 1, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 1, 0, 0, 0, 1, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 0, 0, 0,
        0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    setenv("BN_CUDA_DISABLE_DECODE_CACHE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    unsetenv("BN_CUDA_DISABLE_DECODE_CACHE");
    unsetenv("BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT");
    bn_backend_runtime_policy_unset(
        &gpu.runtime_policy, "BN_METAL_DISABLE_SPECIALIZED_NATIVE_QUANT");
    gpu.kind = BN_GPU_BACKEND_METAL;
    gpu.caps &= ~BN_GPU_CAP_DECODE_GRAPH_CACHE;
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_ENABLE_LOGITS_CACHE");

    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    test_gpu_runtime_refresh(&gpu);
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 16));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(!bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 15));
    assert(bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 16));
    c.policy_flags |= BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT;
    gpu.caps &= ~BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL;
    assert(!bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 16));
    gpu.caps |= BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL;
    assert(bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 16));
    gpu.caps &= ~BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL;
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT;
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 16));
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    test_gpu_runtime_refresh(&gpu);
    gpu.prefill_ssm_layer = NULL;
    assert(!bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 16));
    gpu.prefill_ssm_layer = mock_prefill_ssm_layer;

    map.down_type = BN_GGUF_TENSOR_Q4_K;
    assert(!bn_transformer_gpu_moe_routed_kquant_down_allowed(&map, 0));
    assert(bn_transformer_gpu_moe_routed_kquant_down_allowed(&map, 1));
    assert(bn_transformer_gpu_moe_routed_kquant_down(&map));

    map.gate_type = BN_GGUF_TENSOR_Q8_0;
    map.up_type = BN_GGUF_TENSOR_Q8_0;
    map.down_type = BN_GGUF_TENSOR_Q8_0;
    assert(!bn_transformer_gpu_moe_routed_kquant_down(&map));
    assert(bn_transformer_gpu_moe_routed_native_quant(&map));
    assert(!bn_transformer_gpu_moe_routed_lowbit_block32(&map));

    map.gate_type = BN_GGUF_TENSOR_Q4_0;
    map.up_type = BN_GGUF_TENSOR_Q4_0;
    map.down_type = BN_GGUF_TENSOR_Q4_0;
    assert(!bn_transformer_gpu_moe_routed_native_quant(&map));
    assert(bn_transformer_gpu_moe_routed_lowbit_block32(&map));

    int route_from = 0;
    int route_to = 0;
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_TO_LAYER");
    bn_transformer_gpu_all_active_two_kquant_moe_route_layer_range(
        &gpu, &route_from, &route_to);
    assert(route_from == -1);
    assert(route_to == -1);

    setenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_FROM_LAYER", "2", 1);
    setenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_TO_LAYER", "6", 1);
    test_gpu_runtime_refresh(&gpu);
    bn_transformer_gpu_all_active_two_kquant_moe_route_layer_range(
        &gpu, &route_from, &route_to);
    assert(route_from == 2);
    assert(route_to == 6);
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_ALL_ACTIVE_TWO_KQUANT_MOE_GPU_ROUTE_TO_LAYER");
    test_gpu_runtime_refresh(&gpu);
    BnTransformerGPUMoERouteLayerPolicy route_layer_policy =
        bn_transformer_gpu_moe_route_layer_policy(&gpu);
    assert(route_layer_policy.from_layer == -1);
    assert(route_layer_policy.to_layer == -1);

    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    c.dim = 2048;
    c.moe_norm_topk_prob = 1;
    assert(bn_transformer_gpu_uses_configured_all_active_two_kquant_moe_route(
        &c));
    unsetenv("BN_GPU_DISABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2");
    setenv("BN_GPU_ENABLE_MOE_ROUTER_GPU", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
        &gpu, &c, (void *)1, NULL));
    assert(bn_transformer_gpu_moe_route_normalization_flags(&c) == 0u);
    c.moe_norm_topk_prob = 0;
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
        &gpu, &c, (void *)1, NULL));
    assert(bn_transformer_gpu_moe_route_normalization_flags(&c) ==
           BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM);
    c.moe_norm_topk_prob = 1;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    BnTransformerGPUMoEDirectRoutePolicy direct_route =
        bn_transformer_gpu_moe_direct_route_policy(&gpu, &c, (void *)1, NULL);
    assert(direct_route.enabled);
    assert(direct_route.router_diff == (void *)1);
    BnTransformerGPUMoEDecodeDispatchPolicy direct_dispatch_policy =
        bn_transformer_gpu_moe_decode_dispatch_policy(
            &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
            (void *)2, (void *)1, NULL, (void *)5, (void *)6);
    assert(direct_dispatch_policy.direct_route.enabled);
    assert(direct_dispatch_policy.direct_route.router_diff == (void *)1);
    assert(!direct_dispatch_policy.requires_session_state);
    assert(!direct_dispatch_policy.route_profile_enabled);
    assert(!direct_dispatch_policy.decode_route.gpu_routed_ffn);
    gpu.kind = BN_GPU_BACKEND_METAL;
    direct_route =
        bn_transformer_gpu_moe_direct_route_policy(&gpu, &c, (void *)1, NULL);
    assert(!direct_route.enabled);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    direct_route =
        bn_transformer_gpu_moe_direct_route_policy(&gpu, &c, NULL, NULL);
    assert(!direct_route.enabled);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_router(
        &gpu, &c, (void *)2, (void *)1, 1, 0) == (void *)1);
    c.n_experts_active = 1;
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
        &gpu, &c, (void *)1, NULL));
    assert(bn_transformer_gpu_all_active_two_kquant_moe_router(
        &gpu, &c, (void *)2, (void *)1, 1, 0) == (void *)2);
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4095;
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
        &gpu, &c, (void *)1, NULL));
    c.moe_intermediate_size = 4096;
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
        &gpu, &c, (void *)1, (void *)3));
    BnTransformerGPUMoEAllActiveTwoResourcePolicy all_active_two_resources =
        bn_transformer_gpu_moe_all_active_two_resource_policy(&c);
    assert(all_active_two_resources.enabled);
    assert(all_active_two_resources.total_experts == 2);
    assert(all_active_two_resources.expert_hidden_dim == 4096);
    assert(all_active_two_resources.complement_route_from_expert == 1);
    c.dim = 2049;
    assert(!bn_transformer_gpu_uses_configured_all_active_two_kquant_moe_route(
        &c));
    all_active_two_resources =
        bn_transformer_gpu_moe_all_active_two_resource_policy(&c);
    assert(!all_active_two_resources.enabled);
    assert(all_active_two_resources.total_experts == 0);
    assert(all_active_two_resources.expert_hidden_dim == 0);
    assert(all_active_two_resources.complement_route_from_expert == 0);
    c.dim = 2048;
    c.n_experts_active = 1;
    all_active_two_resources =
        bn_transformer_gpu_moe_all_active_two_resource_policy(&c);
    assert(!all_active_two_resources.enabled);
    c.n_experts_active = 2;
    unsetenv("BN_GPU_ENABLE_MOE_ROUTER_GPU");

    printf("PASSED\n");
}

static void test_logits_policy_helpers(void) {
    printf("test_logits_policy_helpers... ");

    unsetenv("BN_CPU_TIED_KQUANT_REFINE_TOP");
    unsetenv("BN_CPU_TIED_Q6K_REFINE_TOP");
    assert(bn_transformer_logits_cpu_tied_kquant_refine_top(
        test_cpu_policy()) == 0);
    setenv("BN_CPU_TIED_KQUANT_REFINE_TOP", "0", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_refine_top(
        test_cpu_policy()) == 0);
    setenv("BN_CPU_TIED_KQUANT_REFINE_TOP", "7", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_refine_top(
        test_cpu_policy()) == 7);
    setenv("BN_CPU_TIED_KQUANT_REFINE_TOP", "200", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_refine_top(
        test_cpu_policy()) == 128);
    unsetenv("BN_CPU_TIED_KQUANT_REFINE_TOP");
    setenv("BN_CPU_TIED_Q6K_REFINE_TOP", "7", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_refine_top(
        test_cpu_policy()) == 7);
    unsetenv("BN_CPU_TIED_Q6K_REFINE_TOP");

    unsetenv("BN_CPU_TIED_KQUANT_HYBRID_TOP");
    unsetenv("BN_CPU_TIED_Q6K_HYBRID_TOP");
    assert(bn_transformer_logits_cpu_tied_kquant_hybrid_top(
        test_cpu_policy()) == 0);
    setenv("BN_CPU_TIED_KQUANT_HYBRID_TOP", "1", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_hybrid_top(
        test_cpu_policy()) == 0);
    setenv("BN_CPU_TIED_KQUANT_HYBRID_TOP", "9", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_hybrid_top(
        test_cpu_policy()) == 9);
    setenv("BN_CPU_TIED_KQUANT_HYBRID_TOP", "200", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_hybrid_top(
        test_cpu_policy()) == 128);
    unsetenv("BN_CPU_TIED_KQUANT_HYBRID_TOP");
    setenv("BN_CPU_TIED_Q6K_HYBRID_TOP", "9", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_hybrid_top(
        test_cpu_policy()) == 9);
    unsetenv("BN_CPU_TIED_Q6K_HYBRID_TOP");

    unsetenv("BN_CPU_ENABLE_NATIVE_QUANT_TIED_LOGITS");
    unsetenv("BN_CPU_NATIVE_TIED_LOGITS");
    assert(!bn_transformer_logits_cpu_native_tied_quant_enabled(
        test_cpu_policy()));
    setenv("BN_CPU_ENABLE_NATIVE_QUANT_TIED_LOGITS", "1", 1);
    assert(bn_transformer_logits_cpu_native_tied_quant_enabled(
        test_cpu_policy()));
    unsetenv("BN_CPU_ENABLE_NATIVE_QUANT_TIED_LOGITS");
    setenv("BN_CPU_NATIVE_TIED_LOGITS", "1", 1);
    assert(bn_transformer_logits_cpu_native_tied_quant_enabled(
        test_cpu_policy()));
    unsetenv("BN_CPU_ENABLE_NATIVE_QUANT_TIED_LOGITS");
    unsetenv("BN_CPU_NATIVE_TIED_LOGITS");

    assert(bn_transformer_logits_untied_uses_f16_path(BN_GGUF_TENSOR_F16));
    assert(!bn_transformer_logits_untied_uses_f16_path(BN_GGUF_TENSOR_Q4_K));
    assert(bn_transformer_logits_tied_uses_quant_path(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_transformer_logits_tied_uses_quant_path(BN_GGUF_TENSOR_F32));
    assert(bn_transformer_logits_tied_uses_f16_path(BN_GGUF_TENSOR_F16));
    assert(!bn_transformer_logits_tied_uses_f16_path(BN_GGUF_TENSOR_Q6_K));
    assert(bn_transformer_logits_tied_i8_weight_type() == BN_GGUF_TENSOR_Q8_0);
    assert(bn_transformer_logits_tied_f16_weight_type() == BN_GGUF_TENSOR_F16);
    assert(bn_transformer_logits_tied_dense_float_weight_type() == BN_GGUF_TENSOR_F32);
    assert(bn_transformer_logits_native_quant_task_flags(0) == 0);
    assert(bn_transformer_logits_native_quant_task_flags(1) ==
           BN_MATVEC_TASK_NATIVE_QUANT);
    assert(bn_transformer_logits_final_softcap(NULL) == 0.0f);
    BnConfig softcap_config = {0};
    softcap_config.norm_eps = 1.0e-5f;
    assert(bn_model_arch_norm_epsilon(&softcap_config) == 1.0e-5f);
    BnLogitsExecutionPolicy logits_exec =
        bn_transformer_logits_execution_policy(&softcap_config);
    assert(logits_exec.norm_eps == 1.0e-5f);
    assert(logits_exec.norm_eps ==
           bn_model_arch_norm_epsilon(&softcap_config));
    assert(logits_exec.final_softcap == 0.0f);
    assert(bn_transformer_logits_final_softcap(&softcap_config) == 0.0f);
    softcap_config.final_logit_softcap = 30.0f;
    assert(bn_model_arch_final_logit_softcap(&softcap_config) == 30.0f);
    assert(bn_transformer_logits_final_softcap(&softcap_config) == 30.0f);
    logits_exec = bn_transformer_logits_execution_policy(&softcap_config);
    assert(logits_exec.norm_eps == 1.0e-5f);
    assert(logits_exec.final_softcap == 30.0f);
    float softcap_logits[] = { -60.0f, 0.0f, 60.0f };
    bn_transformer_logits_apply_final_softcap(
        softcap_logits, 3, logits_exec.final_softcap);
    assert(fabsf(softcap_logits[0] + 30.0f * tanhf(2.0f)) < 1.0e-6f);
    assert(softcap_logits[1] == 0.0f);
    assert(fabsf(softcap_logits[2] - 30.0f * tanhf(2.0f)) < 1.0e-6f);
    logits_exec = bn_transformer_logits_execution_policy(NULL);
    assert(logits_exec.norm_eps == 0.0f);
    assert(bn_model_arch_norm_epsilon(NULL) == 0.0f);
    assert(bn_model_arch_final_logit_softcap(NULL) == 0.0f);
    assert(logits_exec.final_softcap == 0.0f);

    BnLogitsTiedQuantDispatchPolicy tied_quant_dispatch =
        bn_transformer_logits_tied_quant_dispatch_policy(0, 1, 8, 5, 1);
    assert(tied_quant_dispatch.valid);
    assert(tied_quant_dispatch.matvec_path ==
           BN_LOGITS_TIED_QUANT_BACKEND_PREPARED);
    assert(tied_quant_dispatch.run_tied_kquant_hybrid_refine);
    assert(tied_quant_dispatch.run_tied_kquant_refine);
    assert(tied_quant_dispatch.run_native_quant_refine);
    tied_quant_dispatch =
        bn_transformer_logits_tied_quant_dispatch_policy(1, 1, 8, 5, 1);
    assert(tied_quant_dispatch.valid);
    assert(tied_quant_dispatch.matvec_path ==
           BN_LOGITS_TIED_QUANT_CPU_NATIVE);
    assert(!tied_quant_dispatch.run_tied_kquant_hybrid_refine);
    assert(!tied_quant_dispatch.run_tied_kquant_refine);
    assert(tied_quant_dispatch.run_native_quant_refine);
    tied_quant_dispatch =
        bn_transformer_logits_tied_quant_dispatch_policy(0, 1, 1, 0, 0);
    assert(tied_quant_dispatch.valid);
    assert(tied_quant_dispatch.matvec_path ==
           BN_LOGITS_TIED_QUANT_BACKEND_PREPARED);
    assert(!tied_quant_dispatch.run_tied_kquant_hybrid_refine);
    assert(!tied_quant_dispatch.run_tied_kquant_refine);
    assert(!tied_quant_dispatch.run_native_quant_refine);
    tied_quant_dispatch =
        bn_transformer_logits_tied_quant_dispatch_policy(0, 0, 8, 5, 1);
    assert(tied_quant_dispatch.valid);
    assert(!tied_quant_dispatch.run_tied_kquant_hybrid_refine);
    assert(!tied_quant_dispatch.run_tied_kquant_refine);
    assert(tied_quant_dispatch.run_native_quant_refine);

    unsetenv("BN_CPU_ENABLE_NATIVE_QUANT_TIED_LOGITS");
    unsetenv("BN_CPU_NATIVE_TIED_LOGITS");
    unsetenv("BN_CPU_TIED_KQUANT_HYBRID_TOP");
    unsetenv("BN_CPU_TIED_KQUANT_REFINE_TOP");
    BnQWeight tied_q6 = {0};
    tied_q6.type = BN_GGUF_TENSOR_Q6_K;
    tied_q6.data = (void *)1;
    setenv("BN_CPU_TIED_KQUANT_HYBRID_TOP", "8", 1);
    setenv("BN_CPU_TIED_KQUANT_REFINE_TOP", "5", 1);
    tied_quant_dispatch =
        bn_transformer_logits_tied_quant_dispatch_policy_for(
            test_cpu_policy(), NULL, NULL, &tied_q6);
    assert(tied_quant_dispatch.valid);
    assert(tied_quant_dispatch.matvec_path ==
           BN_LOGITS_TIED_QUANT_BACKEND_PREPARED);
    assert(tied_quant_dispatch.run_tied_kquant_hybrid_refine);
    assert(tied_quant_dispatch.run_tied_kquant_refine);
    assert(!tied_quant_dispatch.run_native_quant_refine);
    setenv("BN_CPU_NATIVE_TIED_LOGITS", "1", 1);
    tied_quant_dispatch =
        bn_transformer_logits_tied_quant_dispatch_policy_for(
            test_cpu_policy(), NULL, NULL, &tied_q6);
    assert(tied_quant_dispatch.valid);
    assert(tied_quant_dispatch.matvec_path ==
           BN_LOGITS_TIED_QUANT_CPU_NATIVE);
    assert(!tied_quant_dispatch.run_tied_kquant_hybrid_refine);
    assert(!tied_quant_dispatch.run_tied_kquant_refine);
    assert(!tied_quant_dispatch.run_native_quant_refine);
    unsetenv("BN_CPU_NATIVE_TIED_LOGITS");
    unsetenv("BN_CPU_TIED_KQUANT_HYBRID_TOP");
    unsetenv("BN_CPU_TIED_KQUANT_REFINE_TOP");

    BnLogitsTiedQuantExecutionPolicy tied_quant_exec =
        bn_transformer_logits_tied_quant_execution_policy_for(
            test_cpu_policy(), NULL, NULL, NULL, NULL);
    assert(!tied_quant_exec.valid);
    assert(tied_quant_exec.weight == NULL);
    assert(tied_quant_exec.prepared == NULL);
    assert(tied_quant_exec.backend_handle == NULL);

    tied_quant_exec =
        bn_transformer_logits_tied_quant_execution_policy_for(
            test_cpu_policy(), NULL, NULL, NULL, &tied_q6);
    assert(tied_quant_exec.valid);
    assert(tied_quant_exec.weight == &tied_q6);
    assert(tied_quant_exec.prepared == NULL);
    assert(tied_quant_exec.backend_handle == NULL);
    assert(tied_quant_exec.dispatch.valid);
    assert(tied_quant_exec.dispatch.matvec_path ==
           BN_LOGITS_TIED_QUANT_BACKEND_PREPARED);

    BnBackendModel *backend = bn_backend_model_create();
    assert(backend);
    BnPreparedWeight prepared = {0};
    prepared.kind = BN_PREPARED_WEIGHT_Q6_K_EXPANDED;
    BnLogitsQuantResources quant_resources =
        bn_transformer_logits_quant_resources(NULL, NULL);
    assert(quant_resources.prepared == NULL);
    assert(quant_resources.gpu_buffer == NULL);
    assert(bn_backend_model_register_prepared_qweight(
               backend, &tied_q6, &prepared) == 0);
    int quant_gpu_buffer;
    assert(bn_backend_model_register_qweight(
               backend, &tied_q6, &quant_gpu_buffer) == 0);
    quant_resources =
        bn_transformer_logits_quant_resources(backend, &tied_q6);
    assert(quant_resources.prepared != NULL);
    assert(quant_resources.prepared->kind == prepared.kind);
    assert(quant_resources.gpu_buffer == &quant_gpu_buffer);
    int tied_handle;
    assert(bn_backend_model_register_handle(
               backend, -1, BN_BACKEND_HANDLE_TIED_EMBEDDING,
               &tied_handle) == 0);
    assert(bn_transformer_gpu_resolve_tied_embedding(backend) ==
           &tied_handle);
    tied_quant_exec =
        bn_transformer_logits_tied_quant_execution_policy_for(
            test_cpu_policy(), NULL, NULL, backend, &tied_q6);
    assert(tied_quant_exec.valid);
    assert(tied_quant_exec.weight == &tied_q6);
    assert(tied_quant_exec.prepared != NULL);
    assert(tied_quant_exec.prepared->kind == prepared.kind);
    assert(tied_quant_exec.uses_prepared_weight);
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
    BnQWeight tied_q4 = tied_q6;
    tied_q4.type = BN_GGUF_TENSOR_Q4_0;
    BnPreparedWeight packed_q4 = {.kind = BN_PREPARED_WEIGHT_Q4_0_X8};
    assert(bn_backend_model_register_prepared_qweight(backend, &tied_q4, &packed_q4) == 0);
    BnLogitsTiedQuantExecutionPolicy tied_q4_exec =
        bn_transformer_logits_tied_quant_execution_policy_for(
            test_cpu_policy(), NULL, NULL, backend, &tied_q4);
    assert(tied_q4_exec.valid && !tied_q4_exec.uses_prepared_weight);
    assert(tied_q4_exec.prepared == NULL);
#endif
    assert(tied_quant_exec.backend_handle == &tied_handle);
    assert(tied_quant_exec.dispatch.valid);
    bn_backend_model_free(backend);

    BnGPUBackend gpu = {0};
    unsetenv("BN_GPU_NATIVE_QUANT_LOGITS_REFINE_TOP");
    unsetenv("BN_GPU_Q8_REFINE_TOP");
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_logits_native_quant_refine_top(&gpu) == 16);
    setenv("BN_GPU_NATIVE_QUANT_LOGITS_REFINE_TOP", "6", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(bn_transformer_logits_native_quant_refine_top(&gpu) == 6);
    unsetenv("BN_GPU_NATIVE_QUANT_LOGITS_REFINE_TOP");

    BnConfig c = {0};
    BnQWeight q8 = {0};
    q8.type = BN_GGUF_TENSOR_Q8_0;
    assert(!bn_transformer_logits_native_quant_refine_enabled(
        NULL, &c, &q8));
    gpu.kind = BN_GPU_BACKEND_METAL;
    c.dim = 2048;
    c.policy_flags = 0;
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_logits_native_quant_refine_enabled(
        &gpu, &c, &q8));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_logits_native_quant_refine_enabled(
        &gpu, &c, &q8));
    q8.type = BN_GGUF_TENSOR_Q4_0;
    assert(!bn_transformer_logits_native_quant_refine_enabled(
        &gpu, &c, &q8));
    q8.type = BN_GGUF_TENSOR_Q8_0;
    c.policy_flags = 0;
    assert(bn_transformer_logits_native_quant_refine_enabled(
        &gpu, &c, &q8));
    c.policy_flags = 0;
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    assert(!bn_transformer_logits_native_quant_refine_enabled(
        &gpu, &c, &q8));
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_NATIVE_QUANT_LOGITS_REFINE");

    printf("PASSED\n");
}

static void test_gpu_staged_value_normalization(void) {
    printf("test_gpu_staged_value_normalization... ");
    for (int fp16 = 0; fp16 < 2; fp16++) {
        for (int normalize = 0; normalize < 2; normalize++) {
            BnConfig c = {0}; c.dim = 32; c.kv_f16 = fp16;
            BnLayerWeights lw = {0};
            lw.attn.wq.type = lw.attn.wk.type = lw.attn.wv.type = BN_GGUF_TENSOR_F32;
            lw.attn.wq.rows = 32; lw.attn.wk.rows = lw.attn.wv.rows = 16;
            lw.attn.wq.cols = lw.attn.wk.cols = lw.attn.wv.cols = 32;
            BnLayerShapePlan plan = {0};
            plan.q_dim = 32; plan.kv_dim = 16; plan.head_size = 8;
            plan.n_heads = 4; plan.n_kv_heads = 2;
            plan.value_shares_key = normalize;
            BnTransformerGPUQKVResources res = {0};
            res.wq = (void *)1; res.wk = (void *)2; res.wv = (void *)3;
            res.v_unit_norm = (void *)4;
            BnGPUOp ops[32]; BnTransformerGPUEmitContext ctx;
            bn_transformer_gpu_emit_context_init(&ctx, ops, 32);
            bn_transformer_gpu_emit_context_qkv(&ctx, &c, &lw, &plan,
                                                &res, 2, 8, 32, 0, 0, 0);
            assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
            int found = 0;
            for (int i = 0; i < ctx.n; i++) {
                if (ops[i].op_code != BN_GPU_CODE_PER_HEAD_RMSNORM ||
                    ops[i].W_buf != res.v_unit_norm) continue;
                found++;
                assert(ops[i].rows == 2 && ops[i].p[0] == 8);
                assert(ops[i].buf_in == (fp16 ? BN_GPU_VALUE_SCRATCH : BN_GPU_VALUE_VALUE_CACHE));
                assert(ops[i].p[3] == (fp16 ? 0u : 32u));
                if (fp16) {
                    assert(i + 1 < ctx.n);
                    assert(ops[i + 1].buf_out == BN_GPU_VALUE_VALUE_CACHE);
                }
            }
            assert(found == normalize);
            bn_transformer_gpu_emit_context_free(&ctx);
        }
    }
    printf("PASSED\n");
}

static void test_gpu_op_kind_mapping(void) {
    printf("test_gpu_op_kind_mapping... ");

    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_MATVEC) == BN_GPU_OP_MATVEC);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_Q8_MATVEC_SPLIT) == BN_GPU_OP_MATVEC);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_RMSNORM) == BN_GPU_OP_RMSNORM);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_ROPE_QK) == BN_GPU_OP_ROPE);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_FLASH_ATTN) == BN_GPU_OP_ATTENTION);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_SILU_ACT) == BN_GPU_OP_ACTIVATION);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_RESIDUAL_ADD) == BN_GPU_OP_RESIDUAL);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_COPY) == BN_GPU_OP_COPY);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_FUSED_GATEUP_SILU) == BN_GPU_OP_FFN);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_SSM_DELTA) == BN_GPU_OP_SSM);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_HC_STREAM_RMSNORM) ==
           BN_GPU_OP_RMSNORM);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_HC_SCALE_SILU) ==
           BN_GPU_OP_ACTIVATION);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_HC_GATED_REDUCE) ==
           BN_GPU_OP_RESIDUAL);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_HC_COMBINE) ==
           BN_GPU_OP_RESIDUAL);
    assert(bn_gpu_op_kind_from_code(99999) == BN_GPU_OP_UNKNOWN);
    assert(bn_gpu_op_kind_from_code(BN_GPU_CODE_FLASH_ATTN) == BN_GPU_OP_ATTENTION);
    assert(bn_gpu_op_code_is_matvec(BN_GPU_CODE_MATVEC));
    assert(!bn_gpu_op_code_is_matvec(BN_GPU_CODE_MATVEC_SPLIT));
    assert(bn_gpu_op_code_is_split_matvec(BN_GPU_CODE_MATVEC_SPLIT));
    assert(bn_gpu_op_code_is_split_matvec(BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(bn_gpu_op_code_is_split_matvec(BN_GPU_CODE_Q8_MATVEC_SPLIT));
    assert(bn_gpu_op_code_is_split_matvec(BN_GPU_CODE_Q5K_MATVEC_SPLIT));
    assert(!bn_gpu_op_code_is_split_matvec(BN_GPU_CODE_MATVEC));
    assert(bn_gpu_op_code_is_rope(BN_GPU_CODE_ROPE));
    assert(bn_gpu_op_code_is_rope(BN_GPU_CODE_ROPE_QK));
    assert(!bn_gpu_op_code_is_rope(BN_GPU_CODE_FLASH_ATTN));
    assert(bn_gpu_op_code_is_rope_qk(BN_GPU_CODE_ROPE_QK));
    assert(!bn_gpu_op_code_is_rope_qk(BN_GPU_CODE_ROPE));
    assert(bn_gpu_op_code_is_flash_attention(BN_GPU_CODE_FLASH_ATTN));
    assert(!bn_gpu_op_code_is_flash_attention(BN_GPU_CODE_GQA_SCORES));
    assert(bn_gpu_op_code_is_per_head_rmsnorm(BN_GPU_CODE_PER_HEAD_RMSNORM));
    assert(!bn_gpu_op_code_is_per_head_rmsnorm(BN_GPU_CODE_RMSNORM));
    assert(bn_gpu_op_code_is_copy(BN_GPU_CODE_COPY));
    assert(!bn_gpu_op_code_is_copy(BN_GPU_CODE_DEINTERLEAVE_Q));
    assert(bn_transformer_gpu_matvec_split_op_code(BN_GGUF_TENSOR_Q4_0) ==
           BN_GPU_CODE_MATVEC_SPLIT);
    assert(bn_transformer_gpu_matvec_split_op_code(BN_GGUF_TENSOR_Q8_0) ==
           BN_GPU_CODE_Q8_MATVEC_SPLIT);
    assert(bn_transformer_gpu_matvec_split_op_code(BN_GGUF_TENSOR_Q5_K) ==
           BN_GPU_CODE_Q5K_MATVEC_SPLIT);
    assert(bn_transformer_gpu_matvec_split_op_code(BN_GGUF_TENSOR_Q4_K) ==
           BN_GPU_CODE_Q4K_MATVEC_SPLIT);
    assert(bn_transformer_gpu_matvec_split_op_code(BN_GGUF_TENSOR_I2_S) == 0);
    assert(bn_transformer_gpu_activation_uses_silu_path(
        BN_MODEL_ACTIVATION_SILU));
    assert(!bn_transformer_gpu_activation_uses_silu_path(
        BN_MODEL_ACTIVATION_RELU2));
    assert(bn_transformer_gpu_ffn_activation_kind(
               BN_MODEL_ACTIVATION_SILU) == BN_GPU_IR_ACTIVATION_SILU);
    assert(bn_transformer_gpu_ffn_activation_kind(
               BN_MODEL_ACTIVATION_RELU2) == BN_GPU_IR_ACTIVATION_RELU2);
    assert(bn_transformer_gpu_ffn_activation_kind(
               BN_MODEL_ACTIVATION_GELU) == BN_GPU_IR_ACTIVATION_GELU);
    assert(bn_transformer_gpu_activation_is_relu2(
        BN_MODEL_ACTIVATION_RELU2));
    assert(!bn_transformer_gpu_activation_is_relu2(
        BN_MODEL_ACTIVATION_SILU));
    BnConfig gpu_norm_config = {0};
    gpu_norm_config.norm_eps = 1.0e-5f;
    assert(bn_transformer_gpu_norm_epsilon(&gpu_norm_config) == 1.0e-5f);
    assert(bn_transformer_gpu_norm_epsilon(NULL) == 0.0f);

    BnGPUOp op;
    memset(&op, 0, sizeof(op));
    op.op_code = BN_GPU_CODE_MATVEC;
    assert(bn_gpu_op_kind(&op) == BN_GPU_OP_MATVEC);
    op.op_kind = BN_GPU_OP_LOGITS;
    assert(bn_gpu_op_kind(&op) == BN_GPU_OP_LOGITS);
    memset(&op, 0, sizeof(op));
    op.op_code = BN_GPU_CODE_SSM_DELTA;
    assert(bn_gpu_op_kind(&op) == BN_GPU_OP_SSM);

    uint32_t reads = 0;
    uint32_t writes = 0;
    BnGPUOp dep_op;
    memset(&dep_op, 0, sizeof(dep_op));
    dep_op.op_code = BN_GPU_CODE_FLASH_ATTN;
    dep_op.buf_in = BN_GPU_VALUE_Q;
    dep_op.buf_out = BN_GPU_VALUE_XB;
    assert(bn_gpu_shader_access_masks(
               &dep_op, bn_gpu_shader_from_op_code(dep_op.op_code),
               &reads, &writes) == 0);
    assert(reads == ((1u << BN_GPU_VALUE_Q) |
                     (1u << BN_GPU_VALUE_KEY_CACHE) |
                     (1u << BN_GPU_VALUE_VALUE_CACHE)));
    assert(writes == (1u << BN_GPU_VALUE_XB));

    memset(&dep_op, 0, sizeof(dep_op));
    dep_op.op_code = BN_GPU_CODE_Q5K_MATVEC_SPLIT;
    dep_op.buf_in = BN_GPU_VALUE_XB;
    dep_op.buf_out = BN_GPU_VALUE_Q;
    dep_op.buf_aux = BN_GPU_VALUE_KEY_CACHE;
    dep_op.rows = BN_GPU_VALUE_VALUE_CACHE;
    assert(bn_gpu_shader_access_masks(
               &dep_op, bn_gpu_shader_from_op_code(dep_op.op_code),
               &reads, &writes) == 0);
    assert(reads == (1u << BN_GPU_VALUE_XB));
    assert(writes == ((1u << BN_GPU_VALUE_Q) |
                      (1u << BN_GPU_VALUE_KEY_CACHE) |
                      (1u << BN_GPU_VALUE_VALUE_CACHE)));

    memset(&op, 0, sizeof(op));
    op.op_code = BN_GPU_CODE_ROPE_QK;
    bn_transformer_gpu_finalize_op_kinds(&op, 1);
    assert(op.op_kind == BN_GPU_OP_ROPE);

    BnGPUOp ctx_ops[2];
    BnTransformerGPUEmitContext ctx;
    bn_transformer_gpu_emit_context_init(&ctx, ctx_ops, 2);
    assert(bn_transformer_gpu_emit_context_rmsnorm(
               &ctx, (void *)3, BN_GPU_VALUE_X, BN_GPU_VALUE_XB,
               32, 0) == 0);
    assert(ctx.n == 0);
    assert(ctx.graph->n_ops == 1);
    assert(bn_transformer_gpu_emit_context_logits(
               &ctx, (void *)4, BN_GGUF_TENSOR_Q8_0, 50, 32) == 0);
    assert(ctx.n == 0);
    assert(ctx.graph->n_ops == 2);
    assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
    assert(ctx.n == 2);
    assert(ctx.graph->n_ops == 0);
    assert(ctx_ops[0].op_kind == BN_GPU_OP_RMSNORM);
    assert(ctx_ops[0].op_code == BN_GPU_CODE_RMSNORM);
    assert(ctx_ops[1].p[6] == 1);
    assert(ctx_ops[1].flags == BN_QUANT_GPU_MATVEC_FLAG_KQUANT_DOT);
    assert(ctx_ops[0].W_buf == (void *)3);
    assert(ctx_ops[1].op_kind == BN_GPU_OP_LOGITS);
    assert(ctx_ops[1].op_code == BN_GPU_CODE_MATVEC);
    assert(ctx_ops[1].type == BN_GGUF_TENSOR_Q8_0);
    assert(ctx_ops[1].W_buf == (void *)4);
    bn_transformer_gpu_emit_context_free(&ctx);

    /* Logits arithmetic composes quant metadata with backend capability;
     * neither backend identity nor model family selects the contract. */
    for (int block32 = 0; block32 < 2; block32++) {
        BnGPUBackend logits_gpu = {0};
        logits_gpu.caps = block32 ? BN_GPU_CAP_KQUANT_BLOCK32_LOGITS : 0;
        bn_transformer_gpu_emit_context_init(&ctx, ctx_ops, 2);
        ctx.gpu = &logits_gpu;
        assert(bn_transformer_gpu_emit_context_logits(
                   &ctx, (void *)4, BN_GGUF_TENSOR_Q6_K, 50, 256) == 0);
        assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
        assert(ctx_ops[0].flags == (block32 ? 0u :
                   BN_GPU_OP_FLAG_MATVEC_REFERENCE_KQUANT));
        bn_transformer_gpu_emit_context_free(&ctx);
    }

    BnGPUOp ctx_ops2[6];
    bn_transformer_gpu_emit_context_init(&ctx, ctx_ops2, 6);
    assert(bn_transformer_gpu_emit_context_copy(
               &ctx, BN_GPU_VALUE_QKV, BN_GPU_VALUE_Q, 0, 0, 16) == 0);
    assert(bn_transformer_gpu_emit_context_residual_add(
               &ctx, BN_GPU_VALUE_X, BN_GPU_VALUE_XB2, 32) == 0);
    assert(bn_transformer_gpu_emit_context_activation(
               &ctx, BN_GPU_VALUE_HB, BN_GPU_VALUE_HB2, 32, 0,
               BN_GPU_IR_ACTIVATION_SILU) == 0);
    assert(bn_transformer_gpu_emit_context_matvec(
               &ctx, BN_GGUF_TENSOR_Q4_0, (void *)5, BN_GPU_VALUE_XB,
               BN_GPU_VALUE_HB, 64, 32, 0) == 0);
    assert(bn_transformer_gpu_emit_context_fused_gateup_silu(
               &ctx, BN_GGUF_TENSOR_Q4_K, (void *)6, BN_GPU_VALUE_XB,
               BN_GPU_VALUE_HB, 64, 64, 32, 0, 0) == 0);
    assert(ctx.n == 0);
    assert(ctx.graph->n_ops == 5);
    assert(bn_transformer_gpu_emit_context_fused_gateup_silu_pair(
               &ctx, BN_GGUF_TENSOR_Q4_0, (void *)7, (void *)8,
               BN_GPU_VALUE_XB, BN_GPU_VALUE_HB, 64, 32, 4) == 0);
    assert(ctx.n == 6);
    assert(ctx.graph->n_ops == 0);
    assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
    assert(ctx.n == 6);
    assert(ctx_ops2[0].op_code == BN_GPU_CODE_COPY);
    assert(ctx_ops2[1].op_code == BN_GPU_CODE_RESIDUAL_ADD);
    assert(ctx_ops2[2].op_code == BN_GPU_CODE_SILU_GATE);
    assert(ctx_ops2[3].op_code == BN_GPU_CODE_MATVEC);
    assert(ctx_ops2[4].op_code == BN_GPU_CODE_FUSED_GATEUP_SILU);
    assert(ctx_ops2[5].op_kind == BN_GPU_OP_FFN);
    assert(ctx_ops2[5].op_code == BN_GPU_CODE_FUSED_GATEUP_SILU);
    assert(ctx_ops2[5].type == BN_GGUF_TENSOR_Q4_0);
    assert(ctx_ops2[5].W_buf == (void *)7);
    assert(ctx_ops2[5].W_buf2 == (void *)8);
    assert(ctx_ops2[5].rows == 64);
    assert(ctx_ops2[5].cols == 32);
    assert(ctx_ops2[5].flags == 4);
    bn_transformer_gpu_emit_context_free(&ctx);

    BnGPUOp route_ops[4];
    BnGPUBackend route_gpu = {0};
    BnModel route_model = {0};
    BnSession route_session = {0};
    BnLayerWeights route_layer = {0};
    BnTransformerGPUMoEExecutionPolicy route_execution = {
        .total_experts = 256,
        .active_experts = 8,
        .expert_weights_scale = 1.0f,
    };
    BnTransformerGPUMoEDecodeRoutePolicy scaled_route = {
        .router = (void *)7,
        .router_scale = (void *)8,
        .uses_scaled_router_input = 1,
        .gpu_route_topk = 1,
        .gpu_routed_ffn = 1,
    };
    BnTransformerGPUMoEDebugPolicy route_debug = {0};
    route_model.config.norm_eps = 1.0e-6f;
    route_session.moe_state = (BnMoEState *)1;
    const char *route_reason = NULL;
    bn_transformer_gpu_emit_context_init(&ctx, route_ops, 4);
    assert(bn_transformer_gpu_prepare_routed_moe_route(
               &ctx, &route_gpu, &route_model, &route_session,
               &route_layer, &route_execution, &scaled_route,
               &route_debug, 0, 0, 32, &route_reason) == 0);
    assert(ctx.n == 2);
    assert(ctx.graph->n_ops == 0);
    assert(route_ops[0].op_code == BN_GPU_CODE_RMSNORM);
    assert(route_ops[0].W_buf == (void *)8);
    assert(route_ops[0].buf_in == BN_GPU_VALUE_X);
    assert(route_ops[0].buf_out == BN_GPU_VALUE_MOE_OUT);
    assert(route_ops[1].op_code == BN_GPU_CODE_MOE_ROUTE_TOPK);
    assert(route_ops[1].W_buf == (void *)7);
    assert(route_ops[1].buf_in == BN_GPU_VALUE_MOE_OUT);
    assert(route_ops[1].buf_out == BN_GPU_VALUE_MOE_HB2);
    assert(route_ops[1].buf_aux == BN_GPU_VALUE_MOE_HB);
    bn_transformer_gpu_emit_context_free(&ctx);

    BnGPUBackend split_gpu = {
        .caps = BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT,
    };
    BnConfig split_config = {.dim = 32};
    BnLayerWeights split_layer = {0};
    split_layer.ffn.ffn_gate = (BnQWeight){
        .data = (void *)1, .type = BN_GGUF_TENSOR_Q4_K,
        .rows = 64, .cols = 32,
    };
    split_layer.ffn.ffn_up = (BnQWeight){
        .data = (void *)2, .type = BN_GGUF_TENSOR_Q4_K,
        .rows = 64, .cols = 32,
    };
    BnFFNPlan split_plan = {
        .hidden_dim = 64,
        .has_gate = 1,
        .activation = BN_MODEL_ACTIVATION_SILU,
    };
    BnTransformerGPUDenseFFNResources split_resources = {
        .gpu = &split_gpu,
        .gateup_stacked = (void *)3,
    };
    BnGPUOp split_ops[2];
    bn_transformer_gpu_emit_context_init(&ctx, split_ops, 2);
    bn_transformer_gpu_emit_context_dense_ffn(
        &ctx, &split_config, &split_layer, &split_plan, &split_resources,
        32, 0, NULL, 1, NULL, 0, 0, 0, 0);
    assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
    assert(ctx.n == 2);
    assert(split_ops[0].op_code == BN_GPU_CODE_Q4K_MATVEC_SPLIT);
    assert(split_ops[1].op_code == BN_GPU_CODE_SILU_GATE);
    bn_transformer_gpu_emit_context_free(&ctx);

    BnConfig residual_config = {.dim = 32};
    BnLayerWeights residual_layer = {0};
    residual_layer.norm.ffn_norm = (float *)1;
    residual_layer.ffn.ffn_gate = (BnQWeight){
        .data = (void *)2, .type = BN_GGUF_TENSOR_Q4_0,
        .rows = 64, .cols = 32,
    };
    residual_layer.ffn.ffn_up = (BnQWeight){
        .data = (void *)3, .type = BN_GGUF_TENSOR_Q4_0,
        .rows = 64, .cols = 32,
    };
    residual_layer.ffn.ffn_down = (BnQWeight){
        .data = (void *)4, .type = BN_GGUF_TENSOR_Q4_0,
        .rows = 32, .cols = 64,
    };
    BnFFNPlan residual_plan = {
        .hidden_dim = 64,
        .has_gate = 1,
        .activation = BN_MODEL_ACTIVATION_GELU,
    };
    BnTransformerGPUDenseFFNResources residual_resources = {
        .ffn_norm = (void *)5,
        .ffn_gate = (void *)6,
        .ffn_up = (void *)7,
        .ffn_down = (void *)8,
        .ffn_down_prefill = (void *)10,
    };
    BnGPUOp residual_ops[32];
    bn_transformer_gpu_emit_context_init(&ctx, residual_ops, 32);
    assert(bn_transformer_gpu_emit_context_dense_residual_moe(
               &ctx, &residual_config, &residual_layer, &residual_plan,
               &residual_resources, 32, 0, (void *)9, 1, 1) == 0);
    assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
    int residual_q4_matvecs = 0;
    for (int i = 0; i < ctx.n; i++) {
        if (residual_ops[i].op_code != BN_GPU_CODE_MATVEC ||
            residual_ops[i].type != BN_GGUF_TENSOR_Q4_0)
            continue;
        assert((residual_ops[i].flags &
                BN_QUANT_GPU_MATVEC_FLAG_KQUANT_DOT) != 0);
        assert((residual_ops[i].flags &
                BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION) != 0);
        if (residual_ops[i].rows != 64)
            assert((residual_ops[i].flags &
                    BN_GPU_OP_FLAG_MATVEC_BLOCK_Q8_ACTIVATION) != 0);
        assert(residual_ops[i].p[6] == 1);
        residual_q4_matvecs++;
    }
    assert(residual_q4_matvecs == 3);
    bn_transformer_gpu_emit_context_free(&ctx);

    bn_transformer_gpu_emit_context_init(&ctx, residual_ops, 32);
    assert(bn_transformer_gpu_emit_context_dense_residual_moe(
               &ctx, &residual_config, &residual_layer, &residual_plan,
               &residual_resources, 32, 0, (void *)9, 1, 0) == 0);
    assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
    int found_quant_down = 0;
    for (int i = 0; i < ctx.n; i++) {
        if (residual_ops[i].op_code == BN_GPU_CODE_MATVEC &&
            residual_ops[i].rows == 32 && residual_ops[i].cols == 64) {
            assert(residual_ops[i].W_buf == (void *)10);
            assert((residual_ops[i].flags &
                    BN_GPU_OP_FLAG_MATVEC_BLOCK_Q8_ACTIVATION) != 0);
            assert((residual_ops[i].flags &
                    BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION) != 0);
            assert(residual_ops[i].p[6] == 1);
            found_quant_down = 1;
        }
    }
    assert(found_quant_down);
    bn_transformer_gpu_emit_context_free(&ctx);

    residual_resources.ffn_post_norm_1 = (void *)11;
    residual_resources.ffn_post_norm_2 = (void *)12;
    residual_resources.ffn_post_norm = (void *)13;
    bn_transformer_gpu_emit_context_init(&ctx, residual_ops, 32);
    assert(bn_transformer_gpu_emit_context_dense_residual_moe(
               &ctx, &residual_config, &residual_layer, &residual_plan,
               &residual_resources, 32, 0, (void *)9, 1, 0) == 0);
    assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
    int dense_norm = -1, routed_norm = -1;
    for (int i = 0; i < ctx.n; i++) {
        if (residual_ops[i].op_code != BN_GPU_CODE_RMSNORM) continue;
        if (residual_ops[i].W_buf == (void *)11) dense_norm = i;
        if (residual_ops[i].W_buf == (void *)12) routed_norm = i;
    }
    assert(dense_norm >= 0 && routed_norm > dense_norm);
    assert(routed_norm + 1 < ctx.n);
    /* The reference fuses routed normalization with the branch sum. */
    assert(residual_ops[routed_norm + 1].op_code == BN_GPU_CODE_RESIDUAL_ADD);
    assert(residual_ops[routed_norm + 1].buf_aux == residual_ops[routed_norm].buf_out);
    assert(residual_ops[routed_norm + 1].buf_in == residual_ops[dense_norm].buf_out);
    bn_transformer_gpu_emit_context_free(&ctx);

    printf("PASSED\n");
}

static void test_model_arch_registry(void) {
    printf("test_model_arch_registry... ");

    size_t count = 0;
    const BnModelArchOps *registry = bn_model_arch_registry(&count);
    assert(registry);
    assert(count >= 4);

    const BnModelArchOps *gemma = bn_model_arch_ops_for("gemma4");
    assert(gemma);
    assert(strcmp(gemma->name, "gemma4") == 0);
    assert(gemma->policy_flags & BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY);
    assert(gemma->policy_flags & BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT);
    assert(gemma->policy_flags & BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION);
    assert(gemma->policy_flags &
           BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER);
    assert(gemma->policy_flags &
           BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION);
    assert(strcmp(gemma->prefix("gemma4"), "gemma4") == 0);
    assert(gemma->attention_value_shares_key("gemma4"));
    assert(gemma->activation("gemma4") == 2);

    BnConfig c = {0};
    bn_model_arch_apply_config(&c, gemma);
    assert(c.policy_flags == gemma->policy_flags);
    assert(bn_model_backend_policy_requires_stable_per_layer_input_layout(&c));
    assert(!bn_model_arch_requires_float_kquant_fallback(&c));
    assert(bn_model_arch_requires_reference_attention(&c));
    assert(bn_transformer_prefill_uses_reference_dot_accumulation(&c));
    assert(bn_model_arch_attention_scale(&c, 128) == 1.0f);
    assert(bn_model_arch_rmsnorm_mode(&c) ==
           BN_MODEL_ARCH_RMSNORM_REFERENCE_ORDER);
    assert(bn_model_arch_attention_value_shares_key_config(&c));
    assert(bn_model_arch_uses_per_layer_embedding(&c));
    assert(bn_model_arch_uses_attention_post_norm(&c));
    assert(bn_model_arch_uses_ffn_post_norm(&c));
    assert(bn_model_arch_uses_layer_output_scale(&c));
    c.qk_norm_per_head = 1;
    c.head_size = 128;
    assert(bn_model_arch_attention_uses_per_head_qk_norm(&c));
    assert(bn_model_arch_attention_qk_norm_stride(&c, c.head_size) == 128);
    c.qk_norm_per_head = 0;
    assert(!bn_model_arch_attention_uses_per_head_qk_norm(&c));
    assert(bn_model_arch_attention_qk_norm_stride(&c, c.head_size) == 0);
    c.has_ffn_gate = 1;
    c.act_type = BN_MODEL_ACTIVATION_GELU;
    assert(bn_model_arch_has_ffn_gate(&c));
    assert(bn_model_arch_config_activation(&c) == BN_MODEL_ACTIVATION_GELU);
    assert(bn_model_arch_loads_extra_metadata(&c));
    c.per_layer_input_dim = 128;
    assert(bn_model_arch_loads_per_layer_input_weights(&c));
    assert(bn_model_arch_divides_rope_freqs(&c, 0));
    assert(!bn_model_arch_prefill_uses_decode_for_parity(&c));
    assert(bn_model_arch_prefill_uses_reference_activation(&c));
    assert(bn_model_arch_ffn_uses_reference_activation(&c));
    assert(!bn_model_arch_moe_requires_float_kquant_gateup_fallback(&c));
    assert(bn_model_arch_moe_uses_scaled_router_input(&c));
    assert(bn_model_arch_moe_uses_reference_router_accumulation(&c));
    assert(bn_model_arch_moe_uses_dense_residual_branch(&c));
    c.n_experts = 8;
    assert(!bn_model_arch_attention_uses_padded_weighted_v_reduction(&c));
    c.n_experts = 0;
    BnGPUBackend lowbit_backend = {0};
    BnMoEExpertMap lowbit_map = {0};
    lowbit_map.gate_type = BN_GGUF_TENSOR_Q4_0;
    lowbit_map.up_type = BN_GGUF_TENSOR_Q4_0;
    lowbit_map.down_type = BN_GGUF_TENSOR_Q4_0;
    assert(bn_transformer_gpu_dense_residual_moe_requires_cpu_ffn(
        &lowbit_backend, &c, &lowbit_map));
    lowbit_backend.caps = BN_GPU_CAP_MOE_ROUTED_LOWBIT_BLOCK32;
    assert(bn_transformer_gpu_dense_residual_moe_requires_cpu_ffn(
        &lowbit_backend, &c, &lowbit_map));
    lowbit_backend.caps = BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32;
    assert(!bn_transformer_gpu_dense_residual_moe_requires_cpu_ffn(
        &lowbit_backend, &c, &lowbit_map));
    c.dim = 2816;
    c.hidden_dim = 11264;
    assert(bn_model_backend_policy_ffn_sub_norm_elements(&c) == c.dim);
    assert(bn_model_arch_loads_extra_ffn_post_norms(&c));
    assert(bn_model_arch_loads_moe_aux_weights(&c));

    c.per_layer_input_dim = 0;
    c.n_experts = 0;
    c.n_layers = 60;
    for (int i = 0; i < c.n_layers; i++)
        c.sliding_window_pattern[i] = (i % 6) != 5;
    assert(!bn_model_arch_divides_rope_freqs(&c, 0));
    assert(bn_model_arch_divides_rope_freqs(&c, 5));

    c.n_experts = 4;
    c.n_layers = 30;
    c.kv_unique_layer_count = 20;
    for (int i = 0; i < c.n_layers; i++)
        c.sliding_window_pattern[i] = (i % 6) != 5;
    c.sliding_window_pattern[20] = 0;
    c.sliding_window_pattern[21] = 1;
    assert(!bn_model_arch_loads_per_layer_input_weights(&c));
    assert(!bn_model_arch_divides_rope_freqs(&c, 0));
    assert(bn_model_arch_divides_rope_freqs(&c, 5));
    assert(bn_model_arch_divides_rope_freqs(&c, 11));
    assert(bn_model_arch_divides_rope_freqs(&c, 17));
    assert(bn_model_arch_divides_rope_freqs(&c, 23));
    assert(bn_model_arch_divides_rope_freqs(&c, 29));
    assert(!bn_model_arch_layer_reuses_kv(&c, 19));
    assert(bn_model_arch_layer_reuses_kv(&c, 20));
    assert(bn_model_arch_kv_reuse_layer(&c, 20) == 19);
    assert(bn_model_arch_kv_reuse_layer(&c, 21) == 18);

    const BnModelArchOps *bitnet = bn_model_arch_ops_for("bitnet");
    assert(bitnet);
    assert(strcmp(bitnet->name, "bitnet") == 0);
    assert(strcmp(bitnet->prefix("bitnet"), "bitnet") == 0);
    assert(bitnet->activation("bitnet") == 1);
    assert(!bitnet->attention_value_shares_key("bitnet"));

    const BnModelArchOps *qwen3 = bn_model_arch_ops_for("qwen3");
    assert(qwen3);
    assert(bn_model_arch_ops_for("qwen3moe") == qwen3);
    assert(qwen3->moe_policy_flags &
           BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION);
    assert(qwen3->policy_flags &
           BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION);
    assert(!(qwen3->policy_flags &
             BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY));
    assert(!(qwen3->policy_flags &
             BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK));
    assert(qwen3->policy_flags &
           BN_MODEL_ARCH_POLICY_SEPARATE_ROPE_NORM);
    assert(qwen3->policy_flags &
           BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION_LAST_THIRD);
    memset(&c, 0, sizeof(c));
    c.policy_flags = qwen3->policy_flags;
    c.n_layers = 36;
    assert(!bn_model_arch_attention_uses_padded_weighted_v_reduction(&c));
    assert(bn_model_arch_reference_attention_from_layer(&c) == 24);
    assert(bn_model_transformer_policy_reference_attention_from_layer(&c) ==
           24);
    c.n_experts = 8;
    c.policy_flags |= qwen3->moe_policy_flags;
    assert(bn_model_arch_attention_uses_padded_weighted_v_reduction(&c));
    assert(bn_model_arch_reference_attention_from_layer(&c) == -1);
    c.n_experts = 0;
    assert(!bn_model_arch_requires_float_kquant_fallback(&c));

    const BnModelArchOps *qwen = bn_model_arch_ops_for("qwen35");
    assert(qwen);
    assert(strcmp(qwen->name, "qwen35") == 0);
    assert(qwen->policy_flags & BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION);
    assert(qwen->policy_flags & BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT);
    assert(!(qwen->policy_flags & BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY));
    assert(qwen->policy_flags & BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS);
    assert(strcmp(qwen->prefix("qwen35"), "qwen35") == 0);
    assert(qwen->activation("qwen35") == 0);
    assert(!qwen->attention_value_shares_key("qwen35"));
    assert(qwen->policy_flags &
           BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER);

    const BnModelArchOps *qwen4exp = bn_model_arch_ops_for("qwen4exp");
    assert(qwen4exp != NULL);
    assert(strcmp(qwen4exp->name, "qwen4exp") == 0);
    assert(qwen4exp->policy_flags & BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS);
    assert(qwen4exp->policy_flags &
           BN_MODEL_ARCH_POLICY_QUERY_SPARSE_ATTENTION);
    assert(qwen4exp->policy_flags &
           BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY);
    /* Routed SwiGLU uses the backend activation, not forced scalar SiLU. */
    assert(!(qwen4exp->moe_policy_flags &
             BN_MODEL_ARCH_POLICY_MOE_REFERENCE_SILU));
    assert(qwen4exp->moe_policy_flags &
           BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ATTENTION);
    BnConfig qwen4exp_config = {0};
    bn_model_arch_apply_config(&qwen4exp_config, qwen4exp);
    assert(bn_model_arch_uses_hyper_connections(&qwen4exp_config));
    assert(bn_model_arch_uses_query_sparse_attention(&qwen4exp_config));
    assert(bn_model_arch_moe_uses_separate_router_topk(&qwen4exp_config));
    assert(bn_transformer_gpu_moe_route_normalization_flags(
               &qwen4exp_config) & BN_GPU_OP_FLAG_MOE_ROUTE_SEPARATE_TOPK);
    assert(!(qwen4exp->policy_flags &
             BN_MODEL_ARCH_POLICY_MOE_UNNORMALIZED_TOPK));
    c.policy_flags = qwen->policy_flags;
    assert(bn_model_arch_requires_reference_attention(&c));
    assert(bn_model_transformer_policy_requires_reference_attention(&c));
    assert(bn_model_arch_requires_reference_recurrent(&c));
    assert(bn_model_transformer_policy_requires_reference_recurrent(&c));
    assert(!bn_model_transformer_policy_requires_host_reference_prefill(&c));
    c.policy_flags |= BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
    assert(bn_model_transformer_policy_requires_host_reference_prefill(&c));
    BnGPUBackend cuda_prefill_gpu = {.kind = BN_GPU_BACKEND_CUDA};
    assert(!bn_transformer_prefill_host_reference_enabled(
        &cuda_prefill_gpu, &c));
    assert(bn_transformer_prefill_host_reference_enabled(NULL, &c));
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT;
    assert(!bn_model_transformer_policy_requires_host_reference_prefill(&c));
    memset(&c, 0, sizeof(c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS;
    assert(bn_model_arch_config_uses_full_rope_text_dims(&c));
    c.policy_flags = 0;
    assert(!bn_model_arch_config_uses_full_rope_text_dims(&c));
    assert(bn_model_arch_tokenizer_uses_metaspace("gemma4"));
    assert(!bn_model_arch_tokenizer_uses_metaspace("llama"));

    memset(&c, 0, sizeof(c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK |
                     BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
                     BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK |
                     BN_MODEL_ARCH_POLICY_PREFILL_REFERENCE_ACTIVATION |
                     BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION;
    assert(bn_model_arch_requires_float_kquant_fallback(&c));
    assert(fabsf(bn_model_arch_attention_scale(&c, 128) -
                 (1.0f / sqrtf(128.0f))) < 1e-7f);
    assert(bn_model_arch_rmsnorm_mode(&c) ==
           BN_MODEL_ARCH_RMSNORM_BACKEND_ORDER);
    assert(!bn_model_arch_attention_value_shares_key_config(&c));
    assert(!bn_model_arch_uses_per_layer_embedding(&c));
    assert(!bn_model_arch_uses_attention_post_norm(&c));
    assert(!bn_model_arch_uses_ffn_post_norm(&c));
    assert(!bn_model_arch_uses_layer_output_scale(&c));
    assert(!bn_model_arch_uses_hybrid_layer_layout(&c));
    assert(!bn_model_arch_uses_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_large_dense_hybrid_ssm(&c));
    assert(bn_model_arch_uses_dense_attention_only(&c));
    assert(!bn_model_arch_uses_large_dense_shape(&c));
    assert(!bn_model_arch_moe_requires_float_kquant_gateup_fallback(&c));
    assert(!bn_model_arch_moe_requires_reference_attention(&c));
    assert(!bn_model_arch_moe_uses_scaled_router_input(&c));
    assert(!bn_model_arch_moe_uses_dense_residual_branch(&c));
    assert(!bn_model_arch_uses_moe(&c));
    assert(!bn_model_arch_uses_non_hybrid_moe(&c));
    assert(!bn_model_arch_uses_two_expert_all_active_moe(&c));
    assert(!bn_model_arch_uses_more_than_two_expert_moe(&c));
    assert(!bn_model_arch_moe_prefill_requires_matvec(&c));
    assert(!bn_model_arch_uses_all_active_two_expert_moe(&c, c.dim));
    assert(bn_model_arch_uses_small_dense_shape(&c));
    assert(!bn_transformer_uses_small_dense_native_quant_shape(&c));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 0));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    assert(bn_model_arch_allows_small_dense_prefill_decode_fallback(&c));
    assert(bn_model_arch_prefill_uses_decode_for_parity(&c));
    assert(bn_transformer_allows_small_dense_native_quant(&c));
    assert(bn_transformer_allows_small_dense_native_logit_refine(&c));
    assert(bn_transformer_small_dense_prefill_min_tokens(&c) == 7);
    assert(bn_model_arch_prefill_uses_reference_activation(&c));
    assert(bn_model_arch_ffn_uses_reference_activation(&c));

    c.full_attn_interval = 4;
    assert(bn_model_arch_uses_hybrid_layer_layout(&c));
    assert(!bn_model_arch_uses_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_dense_attention_only(&c));
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 0));
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    assert(!bn_model_arch_uses_small_dense_shape(&c));
    assert(!bn_transformer_uses_small_dense_native_quant_shape(&c));
    assert(!bn_model_arch_allows_small_dense_prefill_decode_fallback(&c));
    assert(!bn_transformer_allows_small_dense_native_quant(&c));
    assert(!bn_transformer_allows_small_dense_native_logit_refine(&c));
    assert(bn_transformer_small_dense_prefill_min_tokens(&c) == 0);
    c.ssm_inner_size = 128;
    c.dim = 4095;
    assert(bn_model_arch_uses_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_large_dense_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_large_dense_shape(&c));
    assert(!bn_transformer_uses_small_dense_native_quant_shape(&c));
    c.dim = 4096;
    assert(bn_model_arch_uses_large_dense_shape(&c));
    assert(bn_model_arch_uses_large_dense_hybrid_ssm(&c));
    c.n_experts = 1;
    assert(!bn_model_arch_uses_large_dense_shape(&c));
    assert(!bn_model_arch_uses_large_dense_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_small_dense_shape(&c));
    assert(!bn_transformer_uses_small_dense_native_quant_shape(&c));
    assert(bn_model_arch_uses_moe(&c));
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    assert(!bn_model_arch_uses_non_hybrid_moe(&c));
    assert(bn_model_arch_uses_hybrid_moe(&c));
    assert(!bn_model_arch_uses_dense_attention_only(&c));
    assert(!bn_model_arch_dense_logits_argmax_shape_allowed(&c, 300000));
    assert(bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(&c, 1536));
    assert(!bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(&c, 2048));

    memset(&c, 0, sizeof(c));
    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    c.dim = 2048;
    c.moe_norm_topk_prob = 1;
    c.moe_expert_weights_scale = 0.5f;
    c.moe_uses_reference_silu = 1;
    assert(!bn_model_arch_uses_dense_attention_only(&c));
    assert(bn_model_arch_uses_moe(&c));
    assert(bn_model_arch_uses_non_hybrid_moe(&c));
    assert(!bn_model_arch_uses_hybrid_moe(&c));
    assert(bn_model_arch_moe_total_experts(&c) == 2);
    assert(bn_model_arch_moe_active_experts(&c) == 2);
    assert(bn_model_arch_moe_expert_hidden_dim(&c) == 4096);
    assert(bn_model_arch_moe_route_shape_valid(&c));
    assert(bn_model_arch_moe_normalizes_topk_route_weights(&c));
    assert(fabsf(bn_model_arch_moe_expert_weights_scale(&c) - 0.5f) <
           1e-7f);
    assert(bn_model_arch_moe_uses_reference_silu(&c) == 1);
    assert(bn_model_arch_uses_two_expert_all_active_moe(&c));
    assert(!bn_model_arch_uses_more_than_two_expert_moe(&c));
    assert(!bn_model_arch_moe_prefill_requires_matvec(&c));
    assert(bn_model_arch_uses_all_active_two_expert_moe(&c, c.dim));
    c.has_shared_expert = 1;
    c.shared_expert_intermediate_size = 2048;
    assert(bn_model_arch_config_has_shared_expert(&c));
    assert(bn_model_arch_shared_expert_hidden_dim(&c) == 2048);
    assert(bn_model_arch_moe_prefill_requires_matvec(&c));
    c.has_shared_expert = 0;
    assert(!bn_model_arch_config_has_shared_expert(&c));
    assert(bn_model_arch_shared_expert_hidden_dim(&c) == 0);
    c.n_experts_active = 1;
    assert(!bn_model_arch_uses_two_expert_all_active_moe(&c));
    assert(!bn_model_arch_uses_all_active_two_expert_moe(&c, c.dim));
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4095;
    assert(bn_model_arch_uses_two_expert_all_active_moe(&c));
    assert(!bn_model_arch_uses_all_active_two_expert_moe(&c, c.dim));
    c.moe_intermediate_size = 4096;
    assert(!bn_model_arch_uses_all_active_two_expert_moe(&c, 2049));
    c.n_experts = 3;
    assert(bn_model_arch_uses_more_than_two_expert_moe(&c));

    memset(&c, 0, sizeof(c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER |
                     BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
                     BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION;
    assert(!bn_model_arch_requires_float_kquant_fallback(&c));
    assert(!bn_model_arch_moe_requires_float_kquant_gateup_fallback(&c));
    assert(!bn_model_arch_moe_requires_reference_attention(&c));
    assert(bn_model_arch_rmsnorm_mode(&c) ==
           BN_MODEL_ARCH_RMSNORM_REFERENCE_ORDER);
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT;
    assert(!bn_model_backend_policy_requires_stable_per_layer_input_layout(&c));
    assert(bn_model_arch_rmsnorm_uses_reference_order(&c));
    assert(bn_model_arch_uses_small_dense_shape(&c));
    assert(!bn_transformer_uses_small_dense_native_quant_shape(&c));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 0));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    assert(bn_transformer_small_dense_prefill_min_tokens(&c) == 2);
    assert(!bn_model_arch_prefill_uses_reference_activation(&c));
    assert(bn_model_arch_ffn_uses_reference_activation(&c));
    assert(bn_model_arch_dense_logits_argmax_shape_allowed(&c, 300000));
    assert(!bn_model_arch_dense_logits_argmax_shape_allowed(&c, 262144));
    assert(!bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(&c, 1536));
    c.dim = 1025;
    assert(bn_transformer_uses_small_dense_native_quant_shape(&c));
    c.dim = 2561;
    assert(!bn_model_arch_uses_small_dense_shape(&c));
    assert(!bn_transformer_uses_small_dense_native_quant_shape(&c));
    assert(!bn_transformer_small_dense_prefill_min_tokens(&c));
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 0));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    c.dim = 4096;
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 0));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    c.dim = 8193;
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    c.dim = 0;

    c.policy_flags |= BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP_FALLBACK |
                      BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ATTENTION;
    assert(bn_model_arch_moe_requires_float_kquant_gateup_fallback(&c));
    assert(bn_model_arch_moe_requires_reference_attention(&c));

    memset(&c, 0, sizeof(c));
    c.head_size = 128;
    c.rope_theta = 10000.0f;
    assert(bn_model_arch_rope_dims_for_head(&c, 128) == 128);
    assert(bn_model_arch_rope_theta_for_head(&c, 128) == 10000.0f);
    assert(bn_model_arch_rope_base_theta(&c) == 10000.0f);
    assert(bn_model_arch_rope_uses_base_frequency(&c, 128));
    c.rope_dim_count = 64;
    assert(bn_model_arch_rope_dims_for_head(&c, 128) == 64);
    c.rope_text_dims = 32;
    assert(bn_model_arch_rope_dims_for_head(&c, 128) == 32);
    c.rope_theta_swa = 500000.0f;
    c.rope_dim_count_swa = 16;
    assert(bn_model_arch_uses_swa_rope(&c, 64));
    assert(bn_model_arch_rope_dims_for_head(&c, 64) == 16);
    assert(bn_model_arch_rope_theta_for_head(&c, 64) == 500000.0f);
    assert(!bn_model_arch_rope_uses_base_frequency(&c, 64));
    assert(bn_model_arch_rope_dims_for_head(&c, 8) == 8);
    float freqs[4] = {0};
    c.rope_theta_swa = 0.0f;
    c.rope_dim_count = 8;
    c.rope_text_dims = 4;
    bn_model_arch_init_rope_frequencies(&c, freqs, 4);
    assert(freqs[0] > 0.0f);
    assert(freqs[1] > 0.0f);
    assert(freqs[2] == 0.0f);
    assert(freqs[3] == 0.0f);
    float angles[4] = {0};
    float expected_angle = 7.0f;
    float theta_scale = powf(10000.0f, -2.0f / 8.0f);
    bn_model_arch_init_rope_angles_for_theta(
        10000.0f, 8, 7, angles, 4);
    for (int i = 0; i < 4; i++) {
        assert(angles[i] == expected_angle);
        expected_angle *= theta_scale;
    }
    char name[128];
    char scale[128];
    assert(bn_model_arch_tensor_name_for(qwen, name, sizeof(name), 7,
                                         BN_MODEL_TENSOR_ATTN_Q) == 0);
    assert(strcmp(name, "blk.7.attn_q.weight") == 0);
    assert(bn_model_arch_tensor_scale_name_for(qwen, scale, sizeof(scale), 7,
                                               BN_MODEL_TENSOR_ATTN_Q) == 0);
    assert(strcmp(scale, "blk.7.attn_q.scale") == 0);
    assert(bn_model_arch_tensor_name_for(gemma, name, sizeof(name), 2,
                                         BN_MODEL_TENSOR_ATTN_K_BIAS) == 0);
    assert(strcmp(name, "blk.2.attn_k.bias") == 0);
    assert(bn_model_arch_tensor_name_for(bitnet, name, sizeof(name), 3,
                                         BN_MODEL_TENSOR_FFN_DOWN) == 0);
    assert(strcmp(name, "blk.3.ffn_down.weight") == 0);
    assert(bn_model_arch_tensor_name_for(qwen, name, sizeof(name), 4,
                                         BN_MODEL_TENSOR_SSM_ALPHA) == 0);
    assert(strcmp(name, "blk.4.ssm_alpha.weight") == 0);
    assert(bn_model_arch_tensor_scale_name_for(qwen, scale, sizeof(scale), 4,
                                               BN_MODEL_TENSOR_SSM_ALPHA) == 0);
    assert(strcmp(scale, "blk.4.ssm_alpha.scale") == 0);
    assert(bn_model_arch_tensor_name_for(qwen, name, sizeof(name), 5,
                                         BN_MODEL_TENSOR_MOE_GATE_UP_EXPS) == 0);
    assert(strcmp(name, "blk.5.ffn_gate_up_exps.weight") == 0);
    assert(bn_model_arch_tensor_name_for(qwen, name, sizeof(name), 6,
                                         BN_MODEL_TENSOR_SHARED_FFN_ROUTER) == 0);
    assert(strcmp(name, "blk.6.ffn_gate_inp_shexp.weight") == 0);
    assert(bn_model_arch_tensor_name_for(qwen, name, 8, 7,
                                         (BnModelTensorRole)12345) != 0);
    assert(bn_model_arch_tensor_scale_name_for(qwen, scale, sizeof(scale), 7,
                                               BN_MODEL_TENSOR_ATTN_Q_BIAS) != 0);
    assert(bn_model_arch_tensor_scale_name_for(qwen, scale, sizeof(scale), 7,
                                               BN_MODEL_TENSOR_SSM_A) != 0);

    const BnModelArchOps *fallback = bn_model_arch_ops_for(NULL);
    assert(fallback);
    assert(strcmp(fallback->name, "default") == 0);
    assert(strcmp(fallback->prefix(NULL), "llama") == 0);

    memset(&c, 0, sizeof(c));
    c.n_heads = 8;
    c.n_kv_heads = 2;
    gemma->apply_shapes(&c, 256, 512);
    assert(c.head_size == 256);
    assert(c.kv_dim == 512);
    assert(c.kv_mul == 4);

    memset(&c, 0, sizeof(c));
    c.n_heads = 8;
    c.n_kv_heads = 2;
    c.head_size = 128;
    c.kv_dim = 256;
    bitnet->apply_shapes(&c, 512, 1024);
    assert(c.head_size == 128);
    assert(c.kv_dim == 256);

    memset(&c, 0, sizeof(c));
    c.full_attn_interval = 4;
    assert(gemma->is_ssm_layer(&c, 0));
    assert(!gemma->is_ssm_layer(&c, 3));
    assert(!bn_model_arch_is_attention_layer(&c, 0));
    assert(bn_model_arch_is_attention_layer(&c, 3));
    assert(bn_model_arch_attention_layer_index(&c, 3) == 0);
    assert(bn_model_arch_attention_layer_index(&c, 7) == 1);
    assert(bn_model_arch_ssm_layer_index(&c, 0) == 0);
    assert(bn_model_arch_ssm_layer_index(&c, 4) == 3);
    c.n_layers = 12;
    assert(bn_model_arch_attention_layer_count(&c) == 3);
    assert(bn_model_arch_ssm_layer_count(&c) == 9);
    assert(bn_transformer_attention_layer_count(&c) == 3);
    assert(bn_transformer_ssm_layer_count(&c) == 9);
    assert(!bn_transformer_uses_hybrid_ssm(&c));
    c.ssm_inner_size = 128;
    assert(bn_transformer_uses_hybrid_ssm(&c));
    c.ssm_inner_size = 0;
    c.full_attn_interval = 0;
    assert(bn_model_arch_is_attention_layer(&c, 0));
    assert(bn_model_arch_attention_layer_index(&c, 2) == 2);
    assert(bn_model_arch_ssm_layer_index(&c, 2) == -1);
    assert(bn_model_arch_attention_layer_count(&c) == 12);
    assert(bn_model_arch_ssm_layer_count(&c) == 0);
    assert(bn_transformer_attention_layer_count(&c) == 12);
    assert(bn_transformer_ssm_layer_count(&c) == 0);
    assert(!bn_transformer_uses_hybrid_moe(&c));
    c.n_experts = 1;
    c.full_attn_interval = 4;
    assert(bn_transformer_uses_hybrid_moe(&c));

    printf("PASSED\n");
}

static void test_layer_shape_planning(void) {
    printf("test_layer_shape_planning... ");

    BnConfig c;
    BnLayerWeights lw;
    BnLayerShapePlan p;
    memset(&c, 0, sizeof(c));
    memset(&lw, 0, sizeof(lw));

    c.dim = 2048;
    c.n_heads = 16;
    c.n_kv_heads = 4;
    c.head_size = 128;
    c.kv_dim = 512;
    c.kv_mul = 4;
    c.qk_norm_per_head = 1;
    c.kv_f16 = 0;
    c.kv_tq_bits = 0;
    lw.block_kind = BN_LAYER_BLOCK_ATTENTION;
    lw.ffn_kind = BN_LAYER_FFN_DENSE;
    lw.attn.wq.data = (void *)1;
    lw.attn.wq.rows = 2048;
    lw.attn.head_size = 0;
    lw.attn.kv_dim = 0;
    lw.attn.n_kv_heads = 0;
    lw.attn.kv_mul = 0;
    lw.attn.q_norm = (float *)1;
    lw.attn.k_bias = (float *)1;

    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 0);
    assert(p.is_attn);
    assert(p.kind == BN_LAYER_ATTN_CLASSIC);
    assert(bn_transformer_layer_kind(1, 0, 0) == BN_LAYER_ATTN_CLASSIC);
    assert(p.attn_idx == 0);
    lw.attn.has_kv = 1;
    assert(bn_transformer_attention_kv_read_index(&c, &lw, 7) == 7);
    lw.attn.has_kv = 0;
    lw.attn.kv_reuse_layer = 3;
    assert(bn_transformer_attention_kv_read_index(&c, &lw, 7) == 3);
    lw.attn.has_kv = 1;
    lw.attn.kv_reuse_layer = -1;
    assert(p.ssm_idx == -1);
    assert(p.q_dim == 2048);
    assert(!p.q_gated);
    assert(!p.q_wide);
    assert(p.n_heads == 16);
    assert(p.head_size == 128);
    assert(p.kv_dim == 512);
    assert(p.n_kv_heads == 4);
    assert(p.kv_mul == 4);
    assert(p.qk_stride == 128);
    assert(p.qk_norm_per_head == 1);
    assert(!p.value_shares_key);
    assert(p.has_qk_norm);
    assert(p.has_bias);
    assert(p.kv_mode == BN_KV_FP32);
    assert(bn_transformer_attention_head_size(&c, &lw) == 128);
    assert(bn_transformer_attention_n_heads(&c, &lw) == 16);
    assert(bn_transformer_attention_kv_dim(&c, &lw) == 512);
    assert(bn_transformer_attention_n_kv_heads(&c, &lw) == 4);
    assert(bn_transformer_attention_kv_mul(&c, &lw) == 4);
    assert(bn_transformer_attention_qk_stride(&c, p.head_size) == 128);
    assert(bn_transformer_attention_has_qk_norm(&lw));
    assert(bn_transformer_attention_has_bias(&lw));
    assert(!bn_transformer_layer_has_attention_ssm_qkv(&lw));
    lw.ssm.wqkv.data = (void *)1;
    assert(bn_transformer_layer_has_attention_ssm_qkv(&lw));
    lw.ssm.wqkv.data = NULL;

    lw.attn.wq.rows = 4096;
    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 0);
    assert(p.kind == BN_LAYER_ATTN_GATED_Q);
    assert(bn_transformer_layer_kind(1, 1, 0) == BN_LAYER_ATTN_GATED_Q);
    assert(p.q_gated);
    assert(!p.q_wide);
    assert(bn_transformer_attention_q_projection_is_gated(
        &lw.attn.wq, p.q_dim));
    lw.attn.wq.rows = p.q_dim;
    assert(!bn_transformer_attention_q_projection_is_gated(
        &lw.attn.wq, p.q_dim));
    lw.attn.wq.rows = 4096;
    lw.attn.wq.data = NULL;
    assert(!bn_transformer_attention_q_projection_is_gated(
        &lw.attn.wq, p.q_dim));
    lw.attn.wq.data = (void *)1;

    lw.attn.wq.rows = 3072;
    lw.attn.head_size = 192;
    lw.attn.kv_dim = 768;
    lw.attn.n_kv_heads = 4;
    lw.attn.kv_mul = 4;
    c.kv_tq_bits = 3;
    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 1);
    assert(p.kind == BN_LAYER_ATTN_WIDE_Q);
    assert(bn_transformer_layer_kind(1, 0, 1) == BN_LAYER_ATTN_WIDE_Q);
    assert(bn_transformer_layer_kind(1, 1, 1) == BN_LAYER_ATTN_GATED_Q);
    assert(!p.q_gated);
    assert(p.q_wide);
    assert(p.n_heads == 16);
    assert(p.q_dim == 3072);
    assert(p.head_size == 192);
    assert(p.kv_dim == 768);
    assert(p.kv_mode == BN_KV_TQ);
    assert(bn_transformer_attention_head_size(&c, &lw) == 192);
    assert(bn_transformer_attention_kv_dim(&c, &lw) == 768);
    c.qk_norm_per_head = 0;
    assert(bn_transformer_attention_qk_stride(&c, p.head_size) == 0);
    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 1);
    assert(p.qk_stride == 0);
    assert(p.qk_norm_per_head == 0);
    c.policy_flags |= BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY;
    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 1);
    assert(p.value_shares_key);
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY;
    c.qk_norm_per_head = 1;
    lw.attn.q_norm = NULL;
    lw.attn.k_bias = NULL;
    assert(!bn_transformer_attention_has_qk_norm(&lw));
    assert(!bn_transformer_attention_has_bias(&lw));
    lw.attn.q_norm = (float *)1;
    lw.attn.k_bias = (float *)1;
    assert(bn_transformer_attention_q_projection_is_wide(
        &lw.attn.wq, c.dim, p.q_dim));
    lw.attn.wq.rows = 4096;
    assert(!bn_transformer_attention_q_projection_is_wide(
        &lw.attn.wq, c.dim, p.q_dim));
    lw.attn.wq.rows = c.dim;
    assert(!bn_transformer_attention_q_projection_is_wide(
        &lw.attn.wq, c.dim, p.q_dim));
    lw.attn.wq.data = NULL;
    lw.attn.wq.rows = 3072;
    assert(!bn_transformer_attention_q_projection_is_wide(
        &lw.attn.wq, c.dim, p.q_dim));
    lw.attn.wq.data = (void *)1;
    lw.attn.wq.rows = 3072;

    c.full_attn_interval = 4;
    c.kv_tq_bits = 0;
    c.kv_f16 = 1;
    assert(!bn_transformer_is_attn_layer(&c, 0));
    assert(bn_transformer_is_attn_layer(&c, 3));
    assert(bn_transformer_attn_index(&c, 7) == 1);
    assert(bn_transformer_ssm_index(&c, 4) == 3);
    lw.block_kind = BN_LAYER_BLOCK_SSM;
    lw.ssm.wqkv.data = (void *)1;
    assert(!bn_transformer_layer_has_attention_ssm_qkv(&lw));
    lw.ssm.wqkv.data = NULL;
    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 0);
    assert(!p.is_attn);
    assert(p.kind == BN_LAYER_SSM);
    assert(bn_transformer_layer_kind(0, 1, 1) == BN_LAYER_SSM);
    assert(p.attn_idx == -1);
    assert(p.ssm_idx == 0);
    assert(p.kv_mode == BN_KV_FP16);

    lw.block_kind = BN_LAYER_BLOCK_ATTENTION;
    bn_transformer_plan_layer_shape(&p, &c, &lw, 3, 0);
    assert(p.is_attn);
    assert(p.attn_idx == 0);
    assert(p.ssm_idx == -1);

    lw.block_kind = BN_LAYER_BLOCK_SSM;
    bn_transformer_plan_layer_shape(&p, &c, &lw, 4, 0);
    assert(!p.is_attn);
    assert(p.ssm_idx == 3);

    BnQWeight packed;
    memset(&packed, 0, sizeof(packed));
    packed.data = (void *)1;
    packed.cols = 64;
    packed.rows = 96;
    assert(bn_transformer_weight_is_packed_qkv(&packed, 64, 32, 32));
    packed.rows = 95;
    assert(!bn_transformer_weight_is_packed_qkv(&packed, 64, 32, 32));
    packed.rows = 96;
    packed.cols = 32;
    assert(!bn_transformer_weight_is_packed_qkv(&packed, 64, 32, 32));
    packed.cols = 64;
    packed.data = NULL;
    assert(!bn_transformer_weight_is_packed_qkv(&packed, 64, 32, 32));
    assert(!bn_transformer_weight_is_packed_qkv(NULL, 64, 32, 32));

    printf("PASSED\n");
}

static void test_block_planning(void) {
    printf("test_block_planning... ");

    BnConfig c;
    BnLayerWeights lw;
    BnWeights w;
    BnGPUBackend gpu;
    BnAttentionPlan attn;
    BnFFNPlan ffn;
    BnSSMPlan ssm;
    BnMoEPlan moe;
    BnLogitsPlan logits;

    memset(&c, 0, sizeof(c));
    memset(&lw, 0, sizeof(lw));
    memset(&w, 0, sizeof(w));
    memset(&gpu, 0, sizeof(gpu));

    c.dim = 2048;
    c.hidden_dim = 8192;
    c.n_heads = 16;
    c.n_kv_heads = 4;
    c.head_size = 128;
    c.kv_dim = 512;
    c.kv_mul = 4;
    c.vocab_size = 32000;
    c.has_ffn_gate = 1;
    c.flash_attn = 1;
    c.ssm_state_size = 128;
    c.ssm_conv_kernel = 4;
    c.ssm_inner_size = 4096;
    c.ssm_time_step_rank = 32;
    c.ssm_group_count = 16;
    c.n_experts = 128;
    c.n_experts_active = 8;
    c.moe_intermediate_size = 1024;
    c.has_shared_expert = 1;
    c.shared_expert_intermediate_size = 2048;

    gpu.caps = BN_GPU_CAP_LOWBIT_BLOCK32_MATVEC_SPLIT |
               BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT |
               BN_GPU_CAP_NATIVE_QUANT_MATVEC_SPLIT |
               BN_GPU_CAP_DEINTERLEAVED_KQUANT_MATVEC_SPLIT |
               BN_GPU_CAP_LOWBIT_BLOCK32_FUSED_GATEUP_SILU |
               BN_GPU_CAP_FLASH_ATTN;
    gpu.kind = BN_GPU_BACKEND_METAL;

    lw.block_kind = BN_LAYER_BLOCK_ATTENTION;
    lw.ffn_kind = BN_LAYER_FFN_DENSE;
    lw.attn.wq.data = (void *)1;
    lw.attn.wq.rows = 2048;
    lw.attn.wq.cols = 2048;
    lw.attn.wq.type = BN_GGUF_TENSOR_Q4_0;
    lw.attn.wk.type = BN_GGUF_TENSOR_Q4_0;
    lw.attn.wk.rows = 512;
    lw.attn.wk.cols = 2048;
    lw.attn.wv.type = BN_GGUF_TENSOR_Q4_0;
    lw.attn.wv.rows = 512;
    lw.attn.wv.cols = 2048;
    lw.attn.wo.type = BN_GGUF_TENSOR_Q8_0;
    lw.attn.wo.rows = 2048;
    lw.attn.wo.cols = 2048;
    lw.ssm.wqkv.type = BN_GGUF_TENSOR_Q5_K;
    lw.ssm.wqkv.rows = 3072;
    lw.ssm.wqkv.cols = 2048;

    BnBackendModel *backend = bn_backend_model_create();
    assert(backend != NULL);
    assert(bn_backend_model_register_handle(backend, 0,
                                            BN_BACKEND_HANDLE_QKV_STACKED,
                                            (void *)1) == 0);
    assert(bn_backend_model_register_handle(backend, 0,
                                            BN_BACKEND_HANDLE_Q_BIAS,
                                            (void *)2) == 0);
    assert(bn_backend_model_register_handle(backend, 0,
                                            BN_BACKEND_HANDLE_K_BIAS,
                                            (void *)3) == 0);
    assert(bn_backend_model_register_handle(backend, 0,
                                            BN_BACKEND_HANDLE_V_BIAS,
                                            (void *)4) == 0);
    BnTransformerGPUQKVResources qkv_res =
        bn_transformer_gpu_resolve_qkv_resources(&gpu, backend, &lw, 0);
    assert(qkv_res.qkv_stacked == (void *)1);
    assert(qkv_res.q_bias == (void *)2);
    assert(qkv_res.k_bias == (void *)3);
    assert(qkv_res.v_bias == (void *)4);

    BnLayerShapePlan attn_shape;
    bn_transformer_plan_layer_shape(&attn_shape, &c, &lw, 0, 0);
    assert(!bn_transformer_attention_requires_cpu_fallback(
        &attn_shape, BN_EXEC_GPU));
    assert(bn_transformer_attention_flash_requested(&c));
    assert(bn_transformer_attention_uses_flash(&c, &gpu));
    c.flash_attn = 0;
    assert(!bn_transformer_attention_flash_requested(&c));
    assert(!bn_transformer_attention_uses_flash(&c, &gpu));
    c.flash_attn = 1;
    assert(bn_transformer_attention_uses_packed_qkv(
        &gpu, &attn_shape, &lw, (void *)1, (void *)2, (void *)3, (void *)4));
    assert(bn_transformer_attention_uses_qkv_split(
        &gpu, &attn_shape, &lw, (void *)1));
    BnTransformerPlanAttentionProjectionTypes plan_attn_types = {0};
    assert(bn_transformer_plan_resolve_attention_projection_types(
        &plan_attn_types, &lw));
    assert(plan_attn_types.q_type == BN_GGUF_TENSOR_Q4_0);
    assert(plan_attn_types.k_type == BN_GGUF_TENSOR_Q4_0);
    assert(plan_attn_types.v_type == BN_GGUF_TENSOR_Q4_0);
    assert(!bn_transformer_plan_resolve_attention_projection_types(
        NULL, &lw));
    assert(!bn_transformer_plan_resolve_attention_projection_types(
        &plan_attn_types, NULL));
    assert(!bn_transformer_attention_uses_rope_qk_fusion(
        BN_EXEC_GPU, (void *)3));

    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, backend, 0, 0, 1);
    assert(attn.placement == BN_EXEC_GPU);
    assert(attn.backend == BN_BACKEND_METAL);
    assert(attn.shape.kind == BN_LAYER_ATTN_CLASSIC);
    assert(attn.use_flash);
    assert(attn.use_packed_qkv);
    assert(attn.use_qkv_split);
    assert(attn.fusion_flags & BN_FUSION_QKV_SPLIT);
    assert(attn.fusion_flags & BN_FUSION_FLASH_ATTN);
    assert(!(attn.fusion_flags & BN_FUSION_ROPE_QK));
    assert(!attn.needs_cpu_fallback);
    BnTransformerGPUQKVProjectionLayout qkv_layout = {0};
    assert(bn_transformer_gpu_resolve_qkv_projection_layout(
        &qkv_layout, &lw));
    assert(qkv_layout.packed_type == BN_GGUF_TENSOR_Q5_K);
    assert(qkv_layout.packed_rows == 3072);
    assert(qkv_layout.packed_cols == 2048);
    assert(qkv_layout.q_type == BN_GGUF_TENSOR_Q4_0);
    assert(qkv_layout.q_rows == 2048);
    assert(qkv_layout.q_cols == 2048);
    assert(qkv_layout.k_type == BN_GGUF_TENSOR_Q4_0);
    assert(qkv_layout.k_rows == 512);
    assert(qkv_layout.k_cols == 2048);
    assert(qkv_layout.v_type == BN_GGUF_TENSOR_Q4_0);
    assert(qkv_layout.v_rows == 512);
    assert(qkv_layout.v_cols == 2048);
    assert(!bn_transformer_gpu_resolve_qkv_projection_layout(
        NULL, &lw));
    assert(!bn_transformer_gpu_resolve_qkv_projection_layout(
        &qkv_layout, NULL));
    BnTransformerGPUAttentionOutputProjectionLayout out_layout = {0};
    assert(bn_transformer_gpu_resolve_attention_output_projection_layout(
        &out_layout, &lw));
    assert(out_layout.out_type == BN_GGUF_TENSOR_Q8_0);
    assert(out_layout.out_rows == 2048);
    assert(out_layout.out_cols == 2048);
    assert(!bn_transformer_gpu_resolve_attention_output_projection_layout(
        NULL, &lw));
    assert(!bn_transformer_gpu_resolve_attention_output_projection_layout(
        &out_layout, NULL));

    assert(bn_backend_model_register_handle(backend, 0,
                                            BN_BACKEND_HANDLE_K_BIAS,
                                            NULL) == 0);
    assert(bn_transformer_attention_uses_rope_qk_fusion(BN_EXEC_GPU, NULL));
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, backend, 0, 0, 1);
    assert(attn.fusion_flags & BN_FUSION_ROPE_QK);
    assert(bn_backend_model_register_handle(backend, 0,
                                            BN_BACKEND_HANDLE_K_BIAS,
                                            (void *)3) == 0);

    lw.attn.wq.type = BN_GGUF_TENSOR_Q8_0;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, backend, 0, 0, 1);
    assert(attn.use_qkv_split);
    assert(attn.fusion_flags & BN_FUSION_QKV_SPLIT);

    lw.attn.wq.type = BN_GGUF_TENSOR_Q5_K;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, backend, 0, 0, 1);
    assert(attn.use_qkv_split);
    assert(attn.fusion_flags & BN_FUSION_QKV_SPLIT);

    lw.attn.wq.type = BN_GGUF_TENSOR_Q4_0;
    lw.attn.wq.rows = 4096;
    bn_transformer_plan_layer_shape(&attn_shape, &c, &lw, 0, 0);
    assert(!bn_transformer_attention_uses_packed_qkv(
        &gpu, &attn_shape, &lw, (void *)1, (void *)2, (void *)3, (void *)4));
    assert(!bn_transformer_attention_uses_qkv_split(
        &gpu, &attn_shape, &lw, (void *)1));
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, backend, 0, 0, 1);
    assert(attn.shape.kind == BN_LAYER_ATTN_GATED_Q);
    assert(!attn.use_packed_qkv);
    assert(!attn.use_qkv_split);
    assert(!(attn.fusion_flags & BN_FUSION_QKV_SPLIT));
    assert(!attn.use_post_norm);
    c.policy_flags |= BN_MODEL_ARCH_POLICY_ATTENTION_POST_NORM;
    lw.norm.attn_post_norm = (float *)1;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, backend, 0, 0, 1);
    assert(attn.use_post_norm);
    assert(bn_transformer_attention_uses_post_norm_layer(&c, &lw));
    lw.norm.attn_post_norm = NULL;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, backend, 0, 0, 1);
    assert(!attn.use_post_norm);
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_ATTENTION_POST_NORM;

    c.full_attn_interval = 4;
    lw.block_kind = BN_LAYER_BLOCK_SSM;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, backend, 0, 0, 1);
    assert(attn.placement == BN_EXEC_CPU_FALLBACK);
    assert(attn.backend == BN_BACKEND_CPU);
    assert(attn.needs_cpu_fallback);
    assert(bn_transformer_attention_requires_cpu_fallback(
        &attn.shape, BN_EXEC_GPU));
    c.full_attn_interval = 0;
    lw.block_kind = BN_LAYER_BLOCK_ATTENTION;
    lw.attn.wq.rows = 2048;

    lw.ffn.ffn_gate.type = BN_GGUF_TENSOR_Q4_0;
    lw.ffn.ffn_up.type = BN_GGUF_TENSOR_Q4_0;
    lw.ffn.ffn_down.type = BN_GGUF_TENSOR_Q8_0;
    lw.ffn.ffn_gate.rows = 8192;
    lw.ffn.ffn_up.rows = 8192;
    lw.ffn.ffn_down.rows = 2048;
    lw.ffn.ffn_gate.cols = 2048;
    lw.ffn.ffn_up.cols = 2048;
    lw.ffn.ffn_down.cols = 8192;
    BnTransformerGPUDenseFFNProjectionLayout dense_ffn_layout = {0};
    assert(bn_transformer_gpu_resolve_dense_ffn_projection_layout(
        &dense_ffn_layout, &lw));
    assert(dense_ffn_layout.gate_type == BN_GGUF_TENSOR_Q4_0);
    assert(dense_ffn_layout.gate_rows == 8192);
    assert(dense_ffn_layout.gate_cols == 2048);
    assert(dense_ffn_layout.up_type == BN_GGUF_TENSOR_Q4_0);
    assert(dense_ffn_layout.up_rows == 8192);
    assert(dense_ffn_layout.up_cols == 2048);
    assert(dense_ffn_layout.down_type == BN_GGUF_TENSOR_Q8_0);
    assert(dense_ffn_layout.down_rows == 2048);
    assert(dense_ffn_layout.down_cols == 8192);
    assert(!bn_transformer_gpu_resolve_dense_ffn_projection_layout(
        NULL, &lw));
    assert(!bn_transformer_gpu_resolve_dense_ffn_projection_layout(
        &dense_ffn_layout, NULL));
    BnTransformerPlanFFNProjectionTypes plan_ffn_types = {0};
    assert(bn_transformer_plan_resolve_ffn_projection_types(
        &plan_ffn_types, &lw));
    assert(plan_ffn_types.gate_type == BN_GGUF_TENSOR_Q4_0);
    assert(plan_ffn_types.up_type == BN_GGUF_TENSOR_Q4_0);
    assert(plan_ffn_types.up_rows == 8192);
    assert(!bn_transformer_plan_resolve_ffn_projection_types(
        NULL, &lw));
    assert(!bn_transformer_plan_resolve_ffn_projection_types(
        &plan_ffn_types, NULL));
    assert(bn_backend_model_register_handle(backend, 0,
                                            BN_BACKEND_HANDLE_GATEUP_STACKED,
                                            (void *)5) == 0);
    BnTransformerGPUDenseFFNResources dense_ffn_res =
        bn_transformer_gpu_resolve_dense_ffn_resources(&gpu, backend, &lw, 0);
    assert(dense_ffn_res.gateup_stacked == (void *)5);
    BnModelBackendState backend_state = {.backend = backend};
    BnModel resource_model = {
        .config = c,
        .weights = {.layers = &lw},
        .backend_state = &backend_state,
    };
    resource_model.config.n_layers = 1;
    BnTransformerGPULayerResources layer_resources;
    assert(bn_transformer_gpu_resolve_model_layer_resources(
               &layer_resources, &resource_model, &lw, 0, (void *)9) == 0);
    assert(layer_resources.qkv.qkv_stacked == (void *)1);
    assert(layer_resources.qkv.q_bias == (void *)2);
    assert(layer_resources.dense_ffn.gateup_stacked == (void *)5);
    assert(layer_resources.ssm.ssm_qkvz_stacked == NULL);
    assert(layer_resources.moe_decode.router == NULL);
    assert(bn_transformer_gpu_resolve_model_layer_resources(
               NULL, &resource_model, &lw, 0, (void *)9) == -1);
    assert(bn_transformer_gpu_resolve_model_layer_resources(
               &layer_resources, NULL, &lw, 0, (void *)9) == -1);
    assert(bn_transformer_gpu_resolve_model_layer_resources(
               &layer_resources, &resource_model, &lw, 1, (void *)9) == -1);
    lw.norm.ffn_sub_norm = (float *)1;

    bn_transformer_plan_ffn_resources(
        &ffn, &c, &lw, &gpu, &dense_ffn_res, 0, 1);
    assert(ffn.kind == BN_FFN_DENSE_GATE_UP);
    assert(bn_transformer_ffn_kind(&c, &lw) == BN_FFN_DENSE_GATE_UP);
    assert(bn_transformer_ffn_has_gate(&c));
    assert(bn_transformer_ffn_has_sub_norm(&lw));
    assert(bn_transformer_ffn_uses_fused_gateup_silu(
        &gpu, &c, &lw, BN_EXEC_GPU));
    assert(bn_transformer_ffn_uses_gateup_split(
        &gpu, &c, &lw, BN_EXEC_GPU, (void *)5));
    assert(!bn_transformer_ffn_uses_gateup_split(
        &gpu, &c, &lw, BN_EXEC_GPU, NULL));
    assert(bn_transformer_ffn_uses_residual_rmsnorm_fusion(BN_EXEC_GPU));
    assert(!bn_transformer_ffn_uses_residual_rmsnorm_fusion(BN_EXEC_CPU));
    c.has_ffn_gate = 0;
    assert(bn_transformer_ffn_kind(&c, &lw) == BN_FFN_DENSE_UP);
    assert(!bn_transformer_ffn_has_gate(&c));
    assert(!bn_transformer_ffn_uses_fused_gateup_silu(
        &gpu, &c, &lw, BN_EXEC_GPU));
    c.has_ffn_gate = 1;
    assert(ffn.placement == BN_EXEC_GPU);
    assert(ffn.backend == BN_BACKEND_METAL);
    assert(ffn.hidden_dim == 8192);
    assert(bn_transformer_ffn_hidden_dim(&c, &lw) == 8192);
    assert(!ffn.use_post_norm);
    assert(!ffn.use_layer_output_scale);
    c.policy_flags |= BN_MODEL_ARCH_POLICY_FFN_POST_NORM;
    lw.norm.ffn_post_norm = (float *)1;
    bn_transformer_plan_ffn(&ffn, &c, &lw, &gpu, backend, 0, 1);
    assert(ffn.use_post_norm);
    assert(bn_transformer_ffn_uses_post_norm_layer(&c, &lw));
    lw.norm.ffn_post_norm = NULL;
    bn_transformer_plan_ffn(&ffn, &c, &lw, &gpu, backend, 0, 1);
    assert(!ffn.use_post_norm);
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_FFN_POST_NORM;
    lw.norm.ffn_post_norm = NULL;
    c.policy_flags |= BN_MODEL_ARCH_POLICY_LAYER_OUTPUT_SCALE;
    lw.norm.layer_output_scale = (float *)1;
    bn_transformer_plan_ffn(&ffn, &c, &lw, &gpu, backend, 0, 1);
    assert(ffn.use_layer_output_scale);
    assert(bn_transformer_uses_layer_output_scale_layer(&c, &lw));
    lw.norm.layer_output_scale = NULL;
    bn_transformer_plan_ffn(&ffn, &c, &lw, &gpu, backend, 0, 1);
    assert(!ffn.use_layer_output_scale);
    c.policy_flags &= ~BN_MODEL_ARCH_POLICY_LAYER_OUTPUT_SCALE;
    lw.ffn.ffn_up.rows = 0;
    assert(bn_transformer_ffn_hidden_dim(&c, &lw) == c.hidden_dim);
    lw.ffn.ffn_up.rows = 8192;
    assert(ffn.has_gate);
    assert(ffn.has_sub_norm);
    assert(ffn.use_fused_gateup_silu);
    assert(ffn.use_gateup_split);
    assert(ffn.fusion_flags & BN_FUSION_GATEUP_SILU);
    assert(ffn.fusion_flags & BN_FUSION_GATEUP_SPLIT);
    assert(ffn.fusion_flags & BN_FUSION_RESIDUAL_RMSNORM);

    lw.ffn_kind = BN_LAYER_FFN_MOE;
    lw.moe.router_weight = (float *)1;
    bn_transformer_plan_ffn(&ffn, &c, &lw, &gpu, backend, 0, 1);
    assert(ffn.kind == BN_FFN_MOE);
    assert(bn_transformer_ffn_kind(&c, &lw) == BN_FFN_MOE);
    assert(bn_transformer_ffn_requires_cpu_fallback(
        BN_FFN_MOE, BN_EXEC_GPU));
    assert(!bn_transformer_ffn_requires_cpu_fallback(
        BN_FFN_MOE, BN_EXEC_CPU));
    assert(ffn.placement == BN_EXEC_CPU_FALLBACK);
    assert(ffn.backend == BN_BACKEND_CPU);
    assert(ffn.needs_cpu_fallback);
    assert(ffn.fusion_flags == BN_FUSION_NONE);

    assert(bn_backend_model_register_handle(backend, 1,
                                            BN_BACKEND_HANDLE_SSM_QKVZ_STACKED,
                                            (void *)6) == 0);
    assert(bn_backend_model_register_handle(backend, 1,
                                            BN_BACKEND_HANDLE_SSM_AB_STACKED,
                                            (void *)7) == 0);
    BnTransformerGPUSSMResources ssm_res =
        bn_transformer_gpu_resolve_ssm_resources(&gpu, backend, &lw, 1);
    assert(ssm_res.ssm_qkvz_stacked == (void *)6);
    assert(ssm_res.ssm_ab_stacked == (void *)7);
    lw.ssm.wqkv.type = BN_GGUF_TENSOR_Q4_0;
    lw.ssm.wqkv.rows = 8192;
    lw.ssm.wqkv.cols = 2048;
    lw.ssm.wz.type = BN_GGUF_TENSOR_Q4_0;
    lw.ssm.wz.rows = 4096;
    lw.ssm.wz.cols = 2048;
    lw.ssm.ssm_alpha.type = BN_GGUF_TENSOR_Q8_0;
    lw.ssm.ssm_alpha.rows = 4096;
    lw.ssm.ssm_alpha.cols = 2048;
    lw.ssm.ssm_beta.type = BN_GGUF_TENSOR_Q8_0;
    lw.ssm.ssm_beta.rows = 4096;
    lw.ssm.ssm_beta.cols = 2048;
    lw.ssm.ssm_out.type = BN_GGUF_TENSOR_Q5_K;
    lw.ssm.ssm_out.rows = 2048;
    lw.ssm.ssm_out.cols = 4096;
    BnTransformerGPUSSMProjectionLayout ssm_layout = {0};
    assert(bn_transformer_gpu_resolve_ssm_projection_layout(
        &ssm_layout, &lw));
    assert(ssm_layout.qkv_type == BN_GGUF_TENSOR_Q4_0);
    assert(ssm_layout.qkv_rows == 8192);
    assert(ssm_layout.qkv_cols == 2048);
    assert(ssm_layout.z_type == BN_GGUF_TENSOR_Q4_0);
    assert(ssm_layout.z_rows == 4096);
    assert(ssm_layout.z_cols == 2048);
    assert(ssm_layout.alpha_type == BN_GGUF_TENSOR_Q8_0);
    assert(ssm_layout.alpha_rows == 4096);
    assert(ssm_layout.alpha_cols == 2048);
    assert(ssm_layout.beta_type == BN_GGUF_TENSOR_Q8_0);
    assert(ssm_layout.beta_rows == 4096);
    assert(ssm_layout.beta_cols == 2048);
    assert(ssm_layout.out_type == BN_GGUF_TENSOR_Q5_K);
    assert(ssm_layout.out_rows == 2048);
    assert(ssm_layout.out_cols == 4096);
    assert(!bn_transformer_gpu_resolve_ssm_projection_layout(NULL, &lw));
    assert(!bn_transformer_gpu_resolve_ssm_projection_layout(
        &ssm_layout, NULL));
    bn_transformer_plan_ssm(&ssm, &c, &lw, 1, 1, &gpu, backend);
    assert(ssm.placement == BN_EXEC_GPU);
    assert(ssm.backend == BN_BACKEND_METAL);
    assert(ssm.ssm_idx == -1);
    assert(ssm.state_size == 128);
    assert(ssm.conv_kernel == 4);
    assert(ssm.inner_size == 4096);
    assert(ssm.time_step_rank == 32);
    assert(ssm.group_count == 16);
    BnTransformerSSMShapePolicy ssm_shape;
    assert(bn_transformer_ssm_shape_policy(&ssm_shape, &c));
    assert(ssm_shape.num_k_heads == 16);
    assert(ssm_shape.head_k_dim == 128);
    assert(ssm_shape.num_v_heads == 32);
    assert(ssm_shape.head_v_dim == 128);
    assert(ssm_shape.key_dim == 2048);
    assert(ssm_shape.value_dim == 4096);
    assert(ssm_shape.qkv_dim == 8192);
    assert(ssm_shape.conv_kernel == 4);
    c.ssm_conv_kernel = 0;
    assert(bn_transformer_ssm_shape_policy(&ssm_shape, &c));
    assert(ssm_shape.conv_kernel == 4);
    c.ssm_conv_kernel = 4;
    c.ssm_time_step_rank = 0;
    assert(!bn_transformer_ssm_shape_policy(&ssm_shape, &c));
    c.ssm_time_step_rank = 32;
    assert(!bn_transformer_ssm_shape_policy(NULL, &c));
    assert(bn_transformer_ssm_uses_qkvz_stack(BN_EXEC_GPU, (void *)6));
    assert(!bn_transformer_ssm_uses_qkvz_stack(BN_EXEC_GPU, NULL));
    assert(!bn_transformer_ssm_uses_qkvz_stack(BN_EXEC_CPU, (void *)6));
    assert(bn_transformer_ssm_uses_alpha_beta_stack(BN_EXEC_GPU, (void *)7));
    assert(!bn_transformer_ssm_uses_alpha_beta_stack(BN_EXEC_GPU, NULL));
    assert(!bn_transformer_ssm_uses_alpha_beta_stack(
        BN_EXEC_CPU, (void *)7));
    assert(ssm.use_qkvz_stack);
    assert(ssm.use_alpha_beta_stack);

    lw.ffn_kind = BN_LAYER_FFN_DENSE;
    lw.moe.router_weight = NULL;
    lw.attn.wq.type = BN_GGUF_TENSOR_Q4_0;
    lw.attn.wq.rows = 2048;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, backend, 0, 0, 1);
    assert(attn.use_packed_qkv);
    assert(attn.use_qkv_split);
    assert(!(attn.fusion_flags & BN_FUSION_ROPE_QK));
    bn_transformer_plan_ffn(&ffn, &c, &lw, &gpu, backend, 0, 1);
    assert(ffn.use_gateup_split);
    bn_transformer_plan_ssm(&ssm, &c, &lw, 1, 1, &gpu, backend);
    assert(ssm.use_qkvz_stack);
    assert(ssm.use_alpha_beta_stack);
    bn_backend_model_free(backend);

    lw.ffn_kind = BN_LAYER_FFN_MOE;
    lw.moe.router_weight = (float *)1;
    assert(bn_transformer_moe_layer_has_router(&lw));
    assert(bn_transformer_moe_has_shared_expert(&c, &lw));
    c.has_shared_expert = 0;
    assert(!bn_transformer_moe_has_shared_expert(&c, &lw));
    lw.shared.shared_expert_gate = (float *)1;
    assert(bn_transformer_moe_has_shared_expert(&c, &lw));
    BnTransformerMoESharedExpertGatePolicy gate_policy =
        bn_transformer_moe_shared_expert_gate_policy(&lw);
    assert(gate_policy.has_gate_vector);
    assert(!bn_transformer_moe_has_loaded_shared_expert_path(&c, &lw));
    lw.shared.shared_gate.data = (void *)1;
    assert(bn_transformer_moe_has_loaded_shared_expert_path(&c, &lw));
    c.has_shared_expert = 0;
    assert(bn_transformer_moe_has_loaded_shared_expert_path(&c, &lw));
    lw.shared.shared_expert_gate = NULL;
    assert(!bn_transformer_moe_has_loaded_shared_expert_path(&c, &lw));
    c.has_shared_expert = 1;
    assert(bn_transformer_moe_has_loaded_shared_expert_path(&c, &lw));
    lw.shared.shared_gate.data = NULL;
    lw.shared.shared_expert_gate = NULL;
    gate_policy = bn_transformer_moe_shared_expert_gate_policy(&lw);
    assert(!gate_policy.has_gate_vector);
    c.has_shared_expert = 1;
    assert(bn_transformer_moe_requires_cpu_fallback(BN_EXEC_GPU, &lw));
    assert(!bn_transformer_moe_requires_cpu_fallback(BN_EXEC_CPU, &lw));
    bn_transformer_plan_moe(&moe, &c, &lw, &gpu, 0, 1);
    assert(moe.placement == BN_EXEC_CPU_FALLBACK);
    assert(moe.backend == BN_BACKEND_CPU);
    assert(moe.n_experts == 128);
    assert(moe.n_active == 8);
    assert(moe.hidden_dim == 1024);
    assert(moe.has_shared_expert);
    assert(bn_transformer_moe_shared_expert_hidden_dim(&c) == 2048);
    BnTransformerMoESharedExpertShapePolicy shared_policy =
        bn_transformer_moe_shared_expert_shape_policy(&c, &lw);
    assert(shared_policy.has_shared_expert);
    assert(shared_policy.hidden_dim == 2048);
    assert(moe.shared_hidden_dim == 2048);
    assert(moe.needs_cpu_fallback);
    c.has_shared_expert = 0;
    shared_policy = bn_transformer_moe_shared_expert_shape_policy(&c, &lw);
    assert(!shared_policy.has_shared_expert);
    assert(shared_policy.hidden_dim == 0);
    c.has_shared_expert = 1;
    lw.moe.router_weight = NULL;
    assert(!bn_transformer_moe_layer_has_router(&lw));
    assert(!bn_transformer_moe_layer_has_router(NULL));

    w.emb_out_i8 = (int8_t *)1;
    w.emb_type = BN_GGUF_TENSOR_F16;
    bn_transformer_plan_logits(&logits, &c, &w, &gpu, 1);
    assert(bn_transformer_logits_uses_i8_output(&w));
    assert(!bn_transformer_logits_has_untied_output(&w));
    assert(bn_transformer_logits_kind(&w) == BN_LOGITS_TIED_I8);
    assert(bn_transformer_logits_weight_type(&w) ==
           bn_transformer_logits_tied_i8_weight_type());
    assert(logits.kind == BN_LOGITS_TIED_I8);
    assert(logits.placement == BN_EXEC_GPU);
    assert(logits.backend == BN_BACKEND_METAL);
    assert(logits.vocab_size == 32000);
    assert(logits.dim == 2048);
    assert(logits.use_i8_output);

    w.emb_out_i8 = NULL;
    w.output_weight.data = (void *)1;
    w.output_weight.type = BN_GGUF_TENSOR_F16;
    bn_transformer_plan_logits(&logits, &c, &w, NULL, 1);
    assert(!bn_transformer_logits_uses_i8_output(&w));
    assert(bn_transformer_logits_has_untied_output(&w));
    assert(bn_transformer_logits_kind(&w) == BN_LOGITS_UNTIED_F16);
    assert(bn_transformer_logits_weight_type(&w) == BN_GGUF_TENSOR_F16);
    assert(logits.kind == BN_LOGITS_UNTIED_F16);
    assert(logits.weight_type == BN_GGUF_TENSOR_F16);

    w.output_weight.type = BN_GGUF_TENSOR_Q4_K;
    bn_transformer_plan_logits(&logits, &c, &w, NULL, 1);
    assert(bn_transformer_logits_kind(&w) == BN_LOGITS_UNTIED_QUANT);
    assert(logits.kind == BN_LOGITS_UNTIED_QUANT);
    assert(logits.placement == BN_EXEC_CPU);
    assert(logits.backend == BN_BACKEND_CPU);
    assert(logits.weight_type == BN_GGUF_TENSOR_Q4_K);

    w.output_weight.data = NULL;
    w.emb_type = BN_GGUF_TENSOR_Q6_K;
    bn_transformer_plan_logits(&logits, &c, &w, NULL, 0);
    assert(!bn_transformer_logits_has_untied_output(&w));
    assert(bn_transformer_logits_kind(&w) == BN_LOGITS_TIED_QUANT);
    assert(bn_transformer_logits_weight_type(&w) == BN_GGUF_TENSOR_Q6_K);
    assert(logits.kind == BN_LOGITS_TIED_QUANT);
    assert(logits.weight_type == BN_GGUF_TENSOR_Q6_K);

    w.emb_type = BN_GGUF_TENSOR_F32;
    bn_transformer_plan_logits(&logits, &c, &w, NULL, 0);
    assert(bn_transformer_logits_kind(&w) == BN_LOGITS_TIED_DENSE_FLOAT);
    assert(bn_transformer_logits_weight_type(&w) ==
           bn_transformer_logits_tied_dense_float_weight_type());
    assert(logits.kind == BN_LOGITS_TIED_DENSE_FLOAT);
    assert(logits.weight_type == BN_GGUF_TENSOR_F32);

    gpu.kind = BN_GPU_BACKEND_WEBGPU;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, NULL, 0, 0, 1);
    assert(attn.backend == BN_BACKEND_WEBGPU);

    gpu.kind = BN_GPU_BACKEND_CUDA;
    w.output_weight.data = (void *)1;
    w.output_weight.type = BN_GGUF_TENSOR_Q4_K;
    bn_transformer_plan_logits(&logits, &c, &w, &gpu, 1);
    assert(logits.backend == BN_BACKEND_CUDA);

    BnCPUBackendPlacement cpu_backend = bn_transformer_cpu_backend_placement();
    assert(cpu_backend == BN_CPU_BACKEND_SCALAR ||
           cpu_backend == BN_CPU_BACKEND_NEON ||
           cpu_backend == BN_CPU_BACKEND_AVX2 ||
           cpu_backend == BN_CPU_BACKEND_AVX512 ||
           cpu_backend == BN_CPU_BACKEND_WASM_SIMD);
#ifdef __ARM_NEON
    unsetenv("BN_CPU_REFERENCE_MATH");
    unsetenv("BN_CPU_SCALAR_TRANSFORMER_MATH");
    assert(bn_transformer_cpu_backend_ops(test_cpu_policy())->rmsnorm ==
           bn_transformer_rmsnorm_scalar);
#endif
    memset(&c, 0, sizeof(c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK |
                     BN_MODEL_ARCH_POLICY_PREFILL_REFERENCE_ACTIVATION;
    assert(bn_transformer_cpu_float_kquant_task_flags(0) == 0);
    assert(bn_transformer_cpu_float_kquant_task_flags(1) ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);
    assert(bn_transformer_cpu_float_kquant_fallback_task_flags(&c) ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);
    BnTransformerCPUMatvecResourcePolicy cpu_matvec_resource =
        bn_transformer_cpu_matvec_resource_policy(
            test_cpu_policy(), &c, NULL, NULL);
    assert(!cpu_matvec_resource.valid);
    BnQWeight cpu_matvec_weight = {0};
    cpu_matvec_weight.type = BN_GGUF_TENSOR_Q4_K;
    BnBackendModel *cpu_backend_model = bn_backend_model_create();
    assert(cpu_backend_model);
    BnPreparedWeight cpu_prepared = {0};
    cpu_prepared.kind = BN_PREPARED_WEIGHT_Q4_K_SCALES;
    int cpu_gpu_buf;
    assert(bn_backend_model_register_qweight(
               cpu_backend_model, &cpu_matvec_weight, &cpu_gpu_buf) == 0);
    assert(bn_backend_model_register_prepared_qweight(
               cpu_backend_model, &cpu_matvec_weight, &cpu_prepared) == 0);
    unsetenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS");
    cpu_matvec_resource =
        bn_transformer_cpu_matvec_resource_policy(
            test_cpu_policy(), &c, cpu_backend_model, &cpu_matvec_weight);
    assert(cpu_matvec_resource.valid);
    assert(cpu_matvec_resource.prepared != NULL);
    assert(cpu_matvec_resource.prepared->kind == cpu_prepared.kind);
    assert(cpu_matvec_resource.gpu_buffer == &cpu_gpu_buf);
    assert(cpu_matvec_resource.task_flags ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);
    setenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS", "1", 1);
    cpu_matvec_resource =
        bn_transformer_cpu_matvec_resource_policy(
            test_cpu_policy(), &c, cpu_backend_model, &cpu_matvec_weight);
    assert(cpu_matvec_resource.valid);
    assert(cpu_matvec_resource.prepared == NULL);
    assert(cpu_matvec_resource.gpu_buffer == &cpu_gpu_buf);
    assert(cpu_matvec_resource.task_flags ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);
    unsetenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS");
    bn_backend_model_free(cpu_backend_model);
    assert(fabsf(bn_transformer_attention_scale(&c, 128) -
                 (1.0f / sqrtf(128.0f))) < 1e-7f);
    assert(!bn_transformer_attention_value_shares_key(&c));
    assert(!bn_transformer_attention_uses_post_norm(&c));
    assert(!bn_transformer_ffn_uses_post_norm(&c));
    assert(!bn_transformer_uses_layer_output_scale(&c));
    assert(!bn_transformer_uses_per_layer_embedding(&c));
    assert(bn_transformer_per_layer_embedding_dim(&c) == 0);
    assert(!bn_transformer_divides_rope_freqs(&c, 0));
    c.policy_flags |= BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT;
    c.per_layer_input_dim = 128;
    assert(bn_transformer_uses_per_layer_embedding(&c));
    assert(bn_transformer_per_layer_embedding_dim(&c) == 128);
    assert(bn_transformer_divides_rope_freqs(&c, 0));
    c.head_size = 128;
    c.rope_theta = 10000.0f;
    assert(bn_transformer_rope_dims_for_head(&c, 128) == 128);
    assert(bn_transformer_rope_theta_for_head(&c, 128) == 10000.0f);
    assert(bn_transformer_rope_base_theta(&c) == 10000.0f);
    assert(bn_transformer_rope_uses_base_frequency(&c, 128));
    c.rope_dim_count = 64;
    assert(bn_transformer_rope_dims_for_head(&c, 128) == 64);
    c.rope_text_dims = 32;
    assert(bn_transformer_rope_dims_for_head(&c, 128) == 32);
    c.rope_theta_swa = 500000.0f;
    c.rope_dim_count_swa = 16;
    assert(bn_transformer_rope_dims_for_head(&c, 64) == 16);
    assert(bn_transformer_rope_theta_for_head(&c, 64) == 500000.0f);
    assert(!bn_transformer_rope_uses_base_frequency(&c, 64));
    assert(bn_transformer_rope_dims_for_head(&c, 8) == 8);
    c.rope_dim_count = 0;
    c.rope_text_dims = 0;
    c.rope_theta_swa = 0.0f;
    c.rope_dim_count_swa = 0;
    c.per_layer_input_dim = 0;
    c.n_experts = 4;
    c.n_layers = 30;
    for (int i = 0; i < c.n_layers; i++)
        c.sliding_window_pattern[i] = (i % 6) != 5;
    assert(!bn_transformer_per_layer_embedding_dim(&c));
    assert(!bn_transformer_divides_rope_freqs(&c, 0));
    assert(bn_transformer_divides_rope_freqs(&c, 5));
    assert(bn_transformer_divides_rope_freqs(&c, 11));
    assert(bn_transformer_divides_rope_freqs(&c, 17));
    assert(bn_transformer_prefill_uses_reference_activation(&c));
    assert(!bn_transformer_rmsnorm_uses_reference_order(&c));
    int uses_float_kquant_fallback =
        bn_transformer_cpu_prefill_uses_float_kquant_fallback(&c);
    int backend_supports_float_kquant_prefill =
        bn_transformer_cpu_backend_supports_float_kquant_prefill();
    assert(bn_transformer_cpu_backend_prefill_projection_replay() ==
           bn_transformer_cpu_backend_ops(
               test_cpu_policy())->prefill_projection_replay);
#if defined(__AVX2__)
    assert(bn_transformer_cpu_backend_prefill_projection_replay() ==
           BN_CPU_PREFILL_PROJECTION_REPLAY_NONE);
#endif
    assert(backend_supports_float_kquant_prefill ==
           bn_transformer_cpu_backend_ops(
               test_cpu_policy())->supports_float_kquant_prefill);
    assert(uses_float_kquant_fallback ==
           backend_supports_float_kquant_prefill);
    BnTransformerCPUPreparedKQuantInputDispatchPolicy
        prepared_kquant_input =
            bn_transformer_cpu_prepared_kquant_input_dispatch_policy(NULL, 0);
    assert(prepared_kquant_input.path ==
           BN_TRANSFORMER_CPU_PREPARED_KQUANT_INPUT_REUSE);
    prepared_kquant_input =
        bn_transformer_cpu_prepared_kquant_input_dispatch_policy(&gpu, 0);
    assert(prepared_kquant_input.path ==
           BN_TRANSFORMER_CPU_PREPARED_KQUANT_INPUT_FLOAT_FALLBACK);
    prepared_kquant_input =
        bn_transformer_cpu_prepared_kquant_input_dispatch_policy(NULL, 1);
    assert(prepared_kquant_input.path ==
           BN_TRANSFORMER_CPU_PREPARED_KQUANT_INPUT_FLOAT_FALLBACK);
    c.policy_flags = 0;
    assert(bn_transformer_cpu_float_kquant_fallback_task_flags(&c) == 0);
    assert(bn_transformer_cpu_prefill_uses_float_kquant_fallback(&c) ==
           0);
    assert(!bn_transformer_rmsnorm_uses_reference_order(&c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY;
    assert(bn_transformer_cpu_prefill_decode_for_parity_enabled(&c, 0));
    assert(!bn_transformer_cpu_prefill_decode_for_parity_enabled(&c, 1));
    c.policy_flags |= BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS;
    c.hyper_connection_count = 4;
    assert(bn_transformer_cpu_prefill_decode_for_parity_enabled(&c, 0) ==
           !bn_transformer_cpu_backend_supports_hyper_connection_batch_prefill());
    assert(!bn_transformer_cpu_prefill_decode_for_parity_enabled(&c, 1));
    c.hyper_connection_count = 0;
    c.policy_flags = 0;
    assert(!bn_transformer_cpu_prefill_decode_for_parity_enabled(&c, 0));
    c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER;
    assert(bn_transformer_rmsnorm_uses_reference_order(&c));

    unsetenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS");
    assert(bn_transformer_cpu_prepared_qweights_enabled(test_cpu_policy()));
    setenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS", "1", 1);
    assert(!bn_transformer_cpu_prepared_qweights_enabled(test_cpu_policy()));
    unsetenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS");

    unsetenv("BN_DUMP_LAYER_INP");
    unsetenv("BN_DUMP_LAYER_POS");
    unsetenv("BN_DUMP_BINARY_PATH");
    unsetenv("BN_DUMP_BINARY_TAG");
    unsetenv("BN_DUMP_BINARY_LAYER");
    unsetenv("BN_DUMP_ALL_HEADS");
    assert(bn_transformer_cpu_debug_dump_path(test_cpu_policy()) == NULL);
    assert(bn_transformer_cpu_debug_dump_pos_selected(test_cpu_policy(), 3));
    assert(!bn_transformer_cpu_debug_dump_heads_enabled(test_cpu_policy()));
    setenv("BN_DUMP_LAYER_INP", "/tmp/bitnet-dump.txt", 1);
    assert(bn_transformer_cpu_debug_dump_path(test_cpu_policy()) != NULL);
    setenv("BN_DUMP_LAYER_POS", "7", 1);
    setenv("BN_DUMP_BINARY_PATH", "/tmp/bitnet-dump.bin", 1);
    setenv("BN_DUMP_BINARY_TAG", "bitnet_lout", 1);
    setenv("BN_DUMP_BINARY_LAYER", "3", 1);
    assert(strcmp(bn_transformer_cpu_debug_binary_path(test_cpu_policy()),
                  "/tmp/bitnet-dump.bin") == 0);
    assert(bn_transformer_cpu_debug_binary_selected(
        test_cpu_policy(), "bitnet_lout", 3));
    assert(!bn_transformer_cpu_debug_binary_selected(
        test_cpu_policy(), "bitnet_lout", 2));

    unsetenv("BN_DUMP_LAYER_INP");
    assert(strcmp(bn_transformer_cpu_debug_binary_path(test_cpu_policy()),
                  "/tmp/bitnet-dump.bin") == 0);
    assert(bn_transformer_cpu_debug_binary_selected(
        test_cpu_policy(), "bitnet_lout", 3));
    assert(!bn_transformer_cpu_debug_binary_selected(
        test_cpu_policy(), "bitnet_inp", 3));
    assert(bn_transformer_cpu_debug_dump_pos_selected(test_cpu_policy(), 7));
    assert(!bn_transformer_cpu_debug_dump_pos_selected(test_cpu_policy(), 6));
    setenv("BN_DUMP_ALL_HEADS", "1", 1);
    assert(bn_transformer_cpu_debug_dump_heads_enabled(test_cpu_policy()));
    unsetenv("BN_DUMP_LAYER_INP");
    unsetenv("BN_DUMP_LAYER_POS");
    unsetenv("BN_DUMP_BINARY_PATH");
    unsetenv("BN_DUMP_BINARY_TAG");
    unsetenv("BN_DUMP_BINARY_LAYER");
    unsetenv("BN_DUMP_ALL_HEADS");

    unsetenv("BN_CPU_LLAMA_DOT");
    unsetenv("BN_CPU_LLAMA_Q4_DOT");
    unsetenv("BN_CPU_REFERENCE_DOT");
    unsetenv("BN_CPU_REFERENCE_BLOCK_QUANT_DOT");
    unsetenv("BN_CPU_REFERENCE_Q4_DOT");
    assert(bn_transformer_cpu_fused_kquant_gateup_silu_allowed(
        test_cpu_policy()));
    assert(bn_transformer_cpu_can_fused_kquant_gateup_silu(
        test_cpu_policy(), BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_transformer_cpu_can_fused_kquant_gateup_silu(
        test_cpu_policy(), BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0));
    setenv("BN_CPU_REFERENCE_DOT", "1", 1);
    assert(!bn_transformer_cpu_fused_kquant_gateup_silu_allowed(
        test_cpu_policy()));
    assert(!bn_transformer_cpu_can_fused_kquant_gateup_silu(
        test_cpu_policy(), BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    unsetenv("BN_CPU_REFERENCE_DOT");
    setenv("BN_CPU_REFERENCE_BLOCK_QUANT_DOT", "1", 1);
    assert(!bn_transformer_cpu_fused_kquant_gateup_silu_allowed(
        test_cpu_policy()));
    assert(!bn_transformer_cpu_can_fused_kquant_gateup_silu(
        test_cpu_policy(), BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    unsetenv("BN_CPU_REFERENCE_BLOCK_QUANT_DOT");
    setenv("BN_CPU_REFERENCE_Q4_DOT", "1", 1);
    assert(!bn_transformer_cpu_fused_kquant_gateup_silu_allowed(
        test_cpu_policy()));
    assert(!bn_transformer_cpu_can_fused_kquant_gateup_silu(
        test_cpu_policy(), BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    unsetenv("BN_CPU_REFERENCE_Q4_DOT");

    assert(!bn_transformer_cpu_can_prepared_kquant_pair(
        NULL, BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K));
    int supports_prepared_kquant = bn_transformer_cpu_backend_ops(
        test_cpu_policy())->supports_prepared_kquant;
    assert(bn_transformer_cpu_can_prepared_kquant_pair(
               bn_transformer_cpu_backend_ops(test_cpu_policy()),
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K) ==
           supports_prepared_kquant);
    assert(bn_transformer_cpu_can_prepared_kquant_triple(
               bn_transformer_cpu_backend_ops(test_cpu_policy()),
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K,
               BN_GGUF_TENSOR_Q6_K) == supports_prepared_kquant);
    assert(!bn_transformer_cpu_can_prepared_kquant_pair(
        bn_transformer_cpu_backend_ops(test_cpu_policy()), BN_GGUF_TENSOR_Q4_K,
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_transformer_cpu_can_prepared_kquant_triple(
        bn_transformer_cpu_backend_ops(test_cpu_policy()), BN_GGUF_TENSOR_Q4_K,
        BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0));
    BnLayerWeights cpu_lw;
    memset(&cpu_lw, 0, sizeof(cpu_lw));
    cpu_lw.attn.wq.type = BN_GGUF_TENSOR_Q4_K;
    cpu_lw.attn.wk.type = BN_GGUF_TENSOR_Q5_K;
    cpu_lw.attn.wv.type = BN_GGUF_TENSOR_Q6_K;
    cpu_lw.ffn.ffn_gate.type = BN_GGUF_TENSOR_Q4_K;
    cpu_lw.ffn.ffn_up.type = BN_GGUF_TENSOR_Q5_K;
    cpu_lw.ffn.ffn_down.type = BN_GGUF_TENSOR_Q6_K;
    cpu_lw.ssm.wqkv.type = BN_GGUF_TENSOR_Q4_K;
    cpu_lw.ssm.wz.type = BN_GGUF_TENSOR_Q5_K;
    cpu_lw.ssm.ssm_alpha.type = BN_GGUF_TENSOR_Q6_K;
    cpu_lw.ssm.ssm_beta.type = BN_GGUF_TENSOR_Q4_K;
    BnTransformerCPUAttentionProjectionTypes cpu_attn_types = {0};
    assert(bn_transformer_cpu_resolve_attention_projection_types(
        &cpu_attn_types, &cpu_lw));
    assert(cpu_attn_types.q_type == BN_GGUF_TENSOR_Q4_K);
    assert(cpu_attn_types.k_type == BN_GGUF_TENSOR_Q5_K);
    assert(cpu_attn_types.v_type == BN_GGUF_TENSOR_Q6_K);
    assert(!bn_transformer_cpu_resolve_attention_projection_types(
        NULL, &cpu_lw));
    assert(!bn_transformer_cpu_resolve_attention_projection_types(
        &cpu_attn_types, NULL));
    BnTransformerCPUFFNProjectionTypes cpu_ffn_types = {0};
    assert(bn_transformer_cpu_resolve_ffn_projection_types(
        &cpu_ffn_types, &cpu_lw));
    assert(cpu_ffn_types.gate_type == BN_GGUF_TENSOR_Q4_K);
    assert(cpu_ffn_types.up_type == BN_GGUF_TENSOR_Q5_K);
    assert(cpu_ffn_types.down_type == BN_GGUF_TENSOR_Q6_K);
    assert(!bn_transformer_cpu_resolve_ffn_projection_types(
        NULL, &cpu_lw));
    assert(!bn_transformer_cpu_resolve_ffn_projection_types(
        &cpu_ffn_types, NULL));
    BnTransformerCPUSSMProjectionTypes cpu_ssm_types = {0};
    assert(bn_transformer_cpu_resolve_ssm_projection_types(
        &cpu_ssm_types, &cpu_lw));
    assert(cpu_ssm_types.qkv_type == BN_GGUF_TENSOR_Q4_K);
    assert(cpu_ssm_types.z_type == BN_GGUF_TENSOR_Q5_K);
    assert(cpu_ssm_types.alpha_type == BN_GGUF_TENSOR_Q6_K);
    assert(cpu_ssm_types.beta_type == BN_GGUF_TENSOR_Q4_K);
    assert(!bn_transformer_cpu_resolve_ssm_projection_types(
        NULL, &cpu_lw));
    assert(!bn_transformer_cpu_resolve_ssm_projection_types(
        &cpu_ssm_types, NULL));
    BnGPUBackend route_gpu = {0};
    assert(bn_transformer_cpu_route_prepared_kquant_pair_enabled(
               bn_transformer_cpu_backend_ops(test_cpu_policy()), NULL, BN_QK_K,
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K) ==
           supports_prepared_kquant);
    assert(!bn_transformer_cpu_route_prepared_kquant_pair_enabled(
        bn_transformer_cpu_backend_ops(test_cpu_policy()), &route_gpu, BN_QK_K,
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_transformer_cpu_route_prepared_kquant_pair_enabled(
        bn_transformer_cpu_backend_ops(test_cpu_policy()), NULL, BN_QK_K - 1,
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K));
    assert(bn_transformer_cpu_route_prepared_kquant_triple_enabled(
               bn_transformer_cpu_backend_ops(test_cpu_policy()), NULL, BN_QK_K,
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K,
               BN_GGUF_TENSOR_Q6_K) == supports_prepared_kquant);
    assert(!bn_transformer_cpu_route_prepared_kquant_triple_enabled(
        bn_transformer_cpu_backend_ops(test_cpu_policy()), &route_gpu, BN_QK_K,
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K));
    assert(!bn_transformer_cpu_route_prepared_kquant_triple_enabled(
        bn_transformer_cpu_backend_ops(test_cpu_policy()), NULL, BN_QK_K,
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0));
    int cpu_prepared_group[3] = {
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K
    };
    BnTransformerCPUPreparedKQuantRoutePolicy cpu_prepared_route =
        bn_transformer_cpu_prepared_kquant_route_policy(
            bn_transformer_cpu_backend_ops(test_cpu_policy()), NULL, BN_QK_K,
            cpu_prepared_group, 3, 4);
    assert(cpu_prepared_route.enabled == supports_prepared_kquant);
    cpu_prepared_route =
        bn_transformer_cpu_prepared_kquant_route_policy(
            bn_transformer_cpu_backend_ops(test_cpu_policy()), &route_gpu, BN_QK_K,
            cpu_prepared_group, 3, 4);
    assert(!cpu_prepared_route.enabled);
    cpu_prepared_route =
        bn_transformer_cpu_prepared_kquant_route_policy(
            bn_transformer_cpu_backend_ops(test_cpu_policy()), NULL, BN_QK_K - 1,
            cpu_prepared_group, 3, 4);
    assert(!cpu_prepared_route.enabled);
    cpu_prepared_route =
        bn_transformer_cpu_prepared_kquant_route_policy(
            bn_transformer_cpu_backend_ops(test_cpu_policy()), NULL, BN_QK_K,
            cpu_prepared_group, 5, 4);
    assert(!cpu_prepared_route.enabled);
    cpu_prepared_group[2] = BN_GGUF_TENSOR_Q8_0;
    cpu_prepared_route =
        bn_transformer_cpu_prepared_kquant_route_policy(
            bn_transformer_cpu_backend_ops(test_cpu_policy()), NULL, BN_QK_K,
            cpu_prepared_group, 3, 4);
    assert(!cpu_prepared_route.enabled);

    BnFFNPlan ffn_plan = {0};
    ffn_plan.activation = 0;
    assert(bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        test_cpu_policy(), NULL, &ffn_plan, 32,
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        test_cpu_policy(), &route_gpu, &ffn_plan, 32, BN_GGUF_TENSOR_Q4_0,
        BN_GGUF_TENSOR_Q4_0));
    ffn_plan.reference_activation = 1;
    assert(!bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        test_cpu_policy(), NULL, &ffn_plan, 32,
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    ffn_plan.reference_activation = 0;
    ffn_plan.activation = 1;
    assert(!bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        test_cpu_policy(), NULL, &ffn_plan, 32,
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    ffn_plan.activation = 0;
    assert(!bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        test_cpu_policy(), NULL, &ffn_plan, 31,
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        test_cpu_policy(), NULL, &ffn_plan, 32,
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0));

    assert(!bn_transformer_cpu_gpu_dense_ffn_fast_path_available(
        NULL, &ffn_plan));
    assert(!bn_transformer_cpu_gpu_dense_ffn_fast_path_available(
        &route_gpu, &ffn_plan));
    route_gpu.dense_ffn = mock_dense_ffn;
    assert(!bn_transformer_cpu_gpu_dense_ffn_fast_path_available(
        &route_gpu, NULL));
    assert(!bn_transformer_cpu_gpu_dense_ffn_fast_path_available(
        &route_gpu, &ffn_plan));
    ffn_plan.has_gate = 1;
    assert(bn_transformer_cpu_gpu_dense_ffn_fast_path_available(
        &route_gpu, &ffn_plan));
    ffn_plan.has_sub_norm = 1;
    assert(!bn_transformer_cpu_gpu_dense_ffn_fast_path_available(
        &route_gpu, &ffn_plan));
    ffn_plan.has_sub_norm = 0;
    ffn_plan.activation = 1;
    assert(!bn_transformer_cpu_gpu_dense_ffn_fast_path_available(
        &route_gpu, &ffn_plan));
    ffn_plan.activation = 0;
    route_gpu.dense_ffn = NULL;

    assert(bn_transformer_cpu_activation_is_relu2(
        BN_MODEL_ACTIVATION_RELU2));
    assert(!bn_transformer_cpu_activation_is_relu2(
        BN_MODEL_ACTIVATION_SILU));
    assert(bn_transformer_cpu_activation_is_gelu(
        BN_MODEL_ACTIVATION_GELU));
    assert(!bn_transformer_cpu_activation_is_gelu(
        BN_MODEL_ACTIVATION_SILU));
    assert(bn_transformer_cpu_activation_uses_silu_path(
        BN_MODEL_ACTIVATION_SILU));
    assert(!bn_transformer_cpu_activation_uses_silu_path(
        BN_MODEL_ACTIVATION_RELU2));
    BnConfig cpu_norm_config = {0};
    cpu_norm_config.norm_eps = 1.0e-5f;
    assert(bn_transformer_cpu_norm_epsilon(&cpu_norm_config) == 1.0e-5f);
    assert(bn_transformer_cpu_norm_epsilon(NULL) == 0.0f);

    BnTransformerCPUPostNormPolicy cpu_post_norm =
        bn_transformer_cpu_attention_post_norm_policy(1, 1);
    assert(cpu_post_norm.apply);
    cpu_post_norm =
        bn_transformer_cpu_attention_post_norm_policy(0, 1);
    assert(!cpu_post_norm.apply);
    cpu_post_norm =
        bn_transformer_cpu_attention_post_norm_policy(1, 0);
    assert(!cpu_post_norm.apply);
    cpu_post_norm =
        bn_transformer_cpu_ffn_post_norm_policy(1, 1);
    assert(cpu_post_norm.apply);
    cpu_post_norm =
        bn_transformer_cpu_ffn_post_norm_policy(0, 1);
    assert(!cpu_post_norm.apply);
    cpu_post_norm =
        bn_transformer_cpu_ffn_post_norm_policy(1, 0);
    assert(!cpu_post_norm.apply);
    BnTransformerCPULayerOutputScalePolicy cpu_layer_scale =
        bn_transformer_cpu_layer_output_scale_policy(1, 1);
    assert(cpu_layer_scale.apply);
    cpu_layer_scale =
        bn_transformer_cpu_layer_output_scale_policy(0, 1);
    assert(!cpu_layer_scale.apply);
    cpu_layer_scale =
        bn_transformer_cpu_layer_output_scale_policy(1, 0);
    assert(!cpu_layer_scale.apply);

    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    BnConfig ssm_upload_config = c;
    ssm_upload_config.full_attn_interval = 2;
    ssm_upload_config.ssm_inner_size = 16;
    BnTransformerPrefillSSMStateUploadPolicy ssm_upload =
        bn_transformer_prefill_ssm_state_upload_policy(
            &ssm_upload_config, &gpu, 1, 0);
    assert(!ssm_upload.upload);
    ssm_upload = bn_transformer_prefill_ssm_state_upload_policy(
        &ssm_upload_config, &gpu, 1, 1);
    assert(ssm_upload.upload);
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    test_gpu_runtime_refresh(&gpu);
    ssm_upload = bn_transformer_prefill_ssm_state_upload_policy(
        &ssm_upload_config, &gpu, 1, 0);
    assert(ssm_upload.upload);
    ssm_upload = bn_transformer_prefill_ssm_state_upload_policy(
        &ssm_upload_config, &gpu, 0, 1);
    assert(!ssm_upload.upload);
    ssm_upload_config.full_attn_interval = 0;
    ssm_upload_config.ssm_inner_size = 0;
    ssm_upload = bn_transformer_prefill_ssm_state_upload_policy(
        &ssm_upload_config, &gpu, 1, 1);
    assert(!ssm_upload.upload);
    ssm_upload = bn_transformer_prefill_ssm_state_upload_policy(NULL, &gpu,
                                                                1, 1);
    assert(!ssm_upload.upload);
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    test_gpu_runtime_refresh(&gpu);

    assert(bn_transformer_prefill_activation_is_relu2(
        BN_MODEL_ACTIVATION_RELU2));
    assert(!bn_transformer_prefill_activation_is_relu2(
        BN_MODEL_ACTIVATION_SILU));
    assert(bn_transformer_prefill_activation_is_gelu(
        BN_MODEL_ACTIVATION_GELU));
    assert(!bn_transformer_prefill_activation_is_gelu(
        BN_MODEL_ACTIVATION_SILU));
    assert(bn_transformer_prefill_activation_uses_silu_path(
        BN_MODEL_ACTIVATION_SILU));
    assert(!bn_transformer_prefill_activation_uses_silu_path(
        BN_MODEL_ACTIVATION_RELU2));
    BnTransformerPrefillActivationPolicy prefill_activation =
        bn_transformer_prefill_activation_policy(NULL, BN_MODEL_ACTIVATION_GELU, 1);
    assert(prefill_activation.activation == BN_MODEL_ACTIVATION_GELU);
    assert(prefill_activation.uses_reference_activation);
    BnGPUBackend activation_gpu = {0};
    for (int fp32_gelu = 0; fp32_gelu < 2; fp32_gelu++) {
        activation_gpu.caps = fp32_gelu ? BN_GPU_CAP_FP32_GELU : 0;
        prefill_activation = bn_transformer_prefill_activation_policy(
            &activation_gpu, BN_MODEL_ACTIVATION_GELU, 1);
        assert(prefill_activation.uses_reference_activation == !fp32_gelu);
        prefill_activation = bn_transformer_prefill_activation_policy(
            &activation_gpu, BN_MODEL_ACTIVATION_SILU, 1);
        assert(prefill_activation.uses_reference_activation);
        prefill_activation = bn_transformer_prefill_activation_policy(
            &activation_gpu, BN_MODEL_ACTIVATION_GELU, 0);
        assert(!prefill_activation.uses_reference_activation);
    }
    BnConfig prefill_config = {0};
    prefill_config.act_type = BN_MODEL_ACTIVATION_GELU;
    prefill_config.has_ffn_gate = 1;
    prefill_config.norm_eps = 1.0e-5f;
    assert(bn_transformer_prefill_config_activation(&prefill_config) ==
           BN_MODEL_ACTIVATION_GELU);
    assert(bn_transformer_prefill_has_ffn_gate(&prefill_config));
    assert(bn_transformer_prefill_norm_epsilon(&prefill_config) == 1.0e-5f);

    BnTransformerPrefillEntryPolicy prefill_entry =
        bn_transformer_prefill_entry_policy(0, 0, 2, 0, 0);
    assert(prefill_entry.batch);
    prefill_entry = bn_transformer_prefill_entry_policy(0, 0, 2, 1, 1);
    assert(prefill_entry.batch);
    prefill_entry = bn_transformer_prefill_entry_policy(1, 0, 2, 0, 0);
    assert(!prefill_entry.batch);
    prefill_entry = bn_transformer_prefill_entry_policy(0, 1, 2, 0, 0);
    assert(!prefill_entry.batch);
    prefill_entry = bn_transformer_prefill_entry_policy(0, 0, 1, 0, 0);
    assert(!prefill_entry.batch);
    prefill_entry = bn_transformer_prefill_entry_policy(0, 0, 2, 1, 0);
    assert(!prefill_entry.batch);

    BnTransformerPrefillKVUploadPolicy kv_upload =
        bn_transformer_prefill_kv_upload_policy(1, 0);
    assert(kv_upload.upload);
    kv_upload = bn_transformer_prefill_kv_upload_policy(0, 0);
    assert(!kv_upload.upload);
    kv_upload = bn_transformer_prefill_kv_upload_policy(1, 1);
    assert(!kv_upload.upload);

    BnTransformerPrefillChainKVPolicy chain_kv =
        bn_transformer_prefill_chain_kv_policy(0);
    assert(chain_kv.write_host_kv);
    assert(!chain_kv.mark_direct_valid);
    chain_kv = bn_transformer_prefill_chain_kv_policy(1);
    assert(!chain_kv.write_host_kv);
    assert(chain_kv.mark_direct_valid);

    const BnCPUBackendOps *cpu_ops = bn_transformer_cpu_backend_ops(
        test_cpu_policy());
    assert(bn_transformer_cpu_ssm_conv_silu_op(cpu_ops) ==
           cpu_ops->ssm_conv_silu);
    assert(bn_transformer_cpu_ssm_l2norm_op(cpu_ops) ==
           cpu_ops->ssm_l2norm);
    assert(bn_transformer_cpu_ssm_delta_op(cpu_ops) ==
           cpu_ops->ssm_delta);
    assert(bn_transformer_cpu_ssm_gate_op(cpu_ops) ==
           cpu_ops->ssm_gate);

    assert(!bn_transformer_ffn_uses_reference_activation(&c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION;
    assert(bn_transformer_ffn_uses_reference_activation(&c));
    c.policy_flags = 0;

    BnLayerWeights prefill_dense_lw = {0};
    BnLayerWeights prefill_moe_lw = {0};
    prefill_moe_lw.moe.router_weight = (float *)1;

    BnTransformerPrefillLayerKindPolicy prefill_layer_kind =
        bn_transformer_prefill_layer_kind_policy(NULL);
    assert(!prefill_layer_kind.uses_moe);
    prefill_layer_kind =
        bn_transformer_prefill_layer_kind_policy(&prefill_moe_lw);
    assert(prefill_layer_kind.uses_moe);

    prefill_layer_kind =
        bn_transformer_prefill_layer_kind_policy(&prefill_dense_lw);

    BnQWeight prefill_q = {0};
    BnQWeight prefill_k = {0};
    BnQWeight prefill_v = {0};
    prefill_q.type = BN_GGUF_TENSOR_Q4_K;
    prefill_k.type = BN_GGUF_TENSOR_Q4_K;
    prefill_v.type = BN_GGUF_TENSOR_Q5_K;
    prefill_q.rows = 32;
    prefill_k.rows = 16;
    prefill_v.rows = 16;
    prefill_q.cols = prefill_k.cols = prefill_v.cols = 64;
    assert(bn_transformer_prefill_qk_stack_compatible(
        &prefill_q, &prefill_k, 48, 64));
    assert(!bn_transformer_prefill_qk_stack_compatible(
        &prefill_q, &prefill_k, 47, 64));
    prefill_k.cols = 32;
    assert(!bn_transformer_prefill_qk_stack_compatible(
        &prefill_q, &prefill_k, 48, 64));
    prefill_k.cols = 64;
    prefill_k.type = BN_GGUF_TENSOR_Q5_K;
    assert(!bn_transformer_prefill_qk_stack_compatible(
        &prefill_q, &prefill_k, 48, 64));
    prefill_k.type = BN_GGUF_TENSOR_Q4_K;
    assert(bn_transformer_prefill_qkv_stack_batch_compatible(
        &prefill_q, &prefill_k, &prefill_v, 48, 64));
    prefill_v.cols = 32;
    assert(!bn_transformer_prefill_qkv_stack_batch_compatible(
        &prefill_q, &prefill_k, &prefill_v, 48, 64));
    prefill_v.cols = 64;

    BnConfig shared_all_active_two = {0};
    shared_all_active_two.dim = 1024;
    shared_all_active_two.n_experts = 2;
    shared_all_active_two.n_experts_active = 2;
    shared_all_active_two.moe_intermediate_size = 4096;
    shared_all_active_two.has_shared_expert = 1;
    BnTransformerPrefillSharedAllActiveTwoDecodeFallbackPolicy
        shared_all_active_two_fallback =
            bn_transformer_prefill_shared_all_active_two_decode_fallback_policy(
                &shared_all_active_two, 0);
    assert(!shared_all_active_two_fallback.enabled);
    shared_all_active_two_fallback =
        bn_transformer_prefill_shared_all_active_two_decode_fallback_policy(
            &shared_all_active_two, 1);
    assert(!shared_all_active_two_fallback.enabled);
    shared_all_active_two.has_shared_expert = 0;
    shared_all_active_two_fallback =
        bn_transformer_prefill_shared_all_active_two_decode_fallback_policy(
            &shared_all_active_two, 0);
    assert(!shared_all_active_two_fallback.enabled);
    shared_all_active_two.n_experts_active = 1;
    shared_all_active_two.has_shared_expert = 1;
    shared_all_active_two_fallback =
        bn_transformer_prefill_shared_all_active_two_decode_fallback_policy(
            &shared_all_active_two, 0);
    assert(!shared_all_active_two_fallback.enabled);

    BnTransformerPrefillSequencePolicy sequence_policy =
        bn_transformer_prefill_sequence_policy(NULL);
    assert(!sequence_policy.uses_hybrid_layer_layout);
    assert(!sequence_policy.uses_hybrid_ssm);
    assert(!sequence_policy.uses_large_dense_hybrid_ssm);
    assert(!bn_transformer_prefill_uses_hybrid_layer_layout(NULL));
    assert(!bn_transformer_prefill_uses_hybrid_ssm(NULL));
    assert(!bn_transformer_prefill_uses_large_dense_hybrid_ssm(NULL));

    BnConfig sequence = {0};
    sequence_policy = bn_transformer_prefill_sequence_policy(&sequence);
    assert(!sequence_policy.uses_hybrid_layer_layout);
    assert(!sequence_policy.uses_hybrid_ssm);
    assert(!sequence_policy.uses_large_dense_hybrid_ssm);

    sequence.full_attn_interval = 4;
    sequence_policy = bn_transformer_prefill_sequence_policy(&sequence);
    assert(sequence_policy.uses_hybrid_layer_layout);
    assert(!sequence_policy.uses_hybrid_ssm);
    assert(!sequence_policy.uses_large_dense_hybrid_ssm);
    assert(bn_transformer_prefill_uses_hybrid_layer_layout(&sequence));
    assert(!bn_transformer_prefill_uses_large_dense_hybrid_ssm(&sequence));

    sequence.ssm_inner_size = 64;
    sequence.dim = 2048;
    sequence_policy = bn_transformer_prefill_sequence_policy(&sequence);
    assert(sequence_policy.uses_hybrid_layer_layout);
    assert(sequence_policy.uses_hybrid_ssm);
    assert(!sequence_policy.uses_large_dense_hybrid_ssm);
    assert(bn_transformer_prefill_uses_hybrid_ssm(&sequence));

    sequence.dim = 4096;
    sequence_policy = bn_transformer_prefill_sequence_policy(&sequence);
    assert(sequence_policy.uses_large_dense_hybrid_ssm);
    assert(bn_transformer_prefill_uses_large_dense_hybrid_ssm(&sequence));

    sequence.n_experts = 1;
    sequence_policy = bn_transformer_prefill_sequence_policy(&sequence);
    assert(sequence_policy.uses_hybrid_ssm);
    assert(!sequence_policy.uses_large_dense_hybrid_ssm);

    BnTransformerPrefillBufferShapePolicy buffer_shape;
    BnConfig buffer_c = {0};
    buffer_c.kv_dim = 8;
    buffer_c.hidden_dim = 48;
    BnTransformerPrefillSequencePolicy dense_buffer_sequence = {0};
    assert(bn_transformer_prefill_buffer_shape_policy(
        &buffer_shape, &buffer_c, dense_buffer_sequence, 3, 32, 64, 10));
    assert(buffer_shape.kv_dim == 8);
    assert(buffer_shape.hidden_dim == 48);
    assert(buffer_shape.q_buf_stride == 128);
    assert(buffer_shape.xb2_stride == 32);
    assert(buffer_shape.hb_stride == 48);
    assert(buffer_shape.hb2_stride == 48);
    assert(buffer_shape.half_rope == 5);
    assert(buffer_shape.batch_floats == 912);

    /* Partial rotary width controls CPU angle buffers, while GPU layer
     * frequencies retain the full head-width stride used during upload. */
    buffer_c.head_size = 256;
    assert(bn_transformer_prefill_buffer_shape_policy(
        &buffer_shape, &buffer_c, dense_buffer_sequence, 51, 5120, 6144, 64));
    assert(buffer_shape.half_rope == 32);
    assert(buffer_shape.gpu_rope_freq_stride == 128);
    assert(3 * buffer_shape.gpu_rope_freq_stride == 384);
    assert(bn_transformer_prefill_buffer_shape_policy(
        &buffer_shape, &buffer_c, dense_buffer_sequence, 51, 5120, 6144, 256));
    assert(buffer_shape.half_rope == buffer_shape.gpu_rope_freq_stride);

    BnTransformerPrefillSequencePolicy ssm_buffer_sequence = {0};
    ssm_buffer_sequence.uses_hybrid_ssm = 1;
    buffer_c.ssm_time_step_rank = 4;
    buffer_c.ssm_state_size = 8;
    buffer_c.ssm_inner_size = 96;
    buffer_c.ssm_group_count = 2;
    buffer_c.ssm_conv_kernel = 4;
    assert(bn_transformer_prefill_buffer_shape_policy(
        &buffer_shape, &buffer_c, ssm_buffer_sequence, 3, 32, 10, 10));
    assert(buffer_shape.q_buf_stride == 128);
    assert(buffer_shape.xb2_stride == 96);
    assert(buffer_shape.hb_stride == 96);
    assert(buffer_shape.hb2_stride == 96);
    assert(buffer_shape.batch_floats == 1392);
    buffer_c.hidden_dim = 0;
    buffer_c.n_experts = 256;
    assert(bn_transformer_prefill_buffer_shape_policy(
        &buffer_shape, &buffer_c, ssm_buffer_sequence, 3, 32, 10, 10));
    assert(buffer_shape.hidden_dim == 0);
    assert(buffer_shape.hb_stride == 96);
    assert(buffer_shape.hb2_stride == 96);
    buffer_c.hidden_dim = 48;
    buffer_c.n_experts = 0;
    buffer_c.ssm_time_step_rank = 0;
    assert(!bn_transformer_prefill_buffer_shape_policy(
        &buffer_shape, &buffer_c, ssm_buffer_sequence, 3, 32, 10, 10));
    assert(!bn_transformer_prefill_buffer_shape_policy(
        NULL, &buffer_c, dense_buffer_sequence, 3, 32, 10, 10));

    BnTransformerPrefillSequencePolicy decode_sequence = {0};
    BnTransformerPrefillDecodeFallbackPolicy decode_fallback =
        bn_transformer_prefill_decode_fallback_policy(
            decode_sequence, 1, 0, 16, 1, 0, 1, 0, 0, 1, 0);
    assert(decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 1, 1, 8, 16, 0, 1, 0, 0, 1, 0);
    assert(decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 0, 1, 8, 1, 1, 16, 0, 0, 1, 0);
    assert(decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_sequence.uses_large_dense_hybrid_ssm = 1;
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 0, 1, 16, 1, 0, 1, 1, 1, 1, 0);
    assert(decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_sequence.uses_hybrid_ssm = 1;
    decode_sequence.uses_large_dense_hybrid_ssm = 0;
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 0, 1, 16, 1, 0, 1, 0, 0, 0, 0);
    assert(decode_fallback.decode);
    assert(decode_fallback.require_logits_decode);
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 0, 1, 16, 1, 0, 1, 1, 0, 0, 0);
    assert(!decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_sequence.uses_hybrid_ssm = 0;
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 0, 1, 16, 1, 0, 1, 0, 0, 1, 0);
    assert(!decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 1, 0, 16, 1, 0, 1, 0, 0, 1, 1);
    assert(!decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);

    BnTransformerPrefillDenseModelChainPolicy dense_model_chain =
        bn_transformer_prefill_dense_model_chain_policy(1, 1, 0, 1);
    assert(dense_model_chain.enabled);
    dense_model_chain =
        bn_transformer_prefill_dense_model_chain_policy(0, 1, 0, 1);
    assert(!dense_model_chain.enabled);
    dense_model_chain =
        bn_transformer_prefill_dense_model_chain_policy(1, 0, 0, 1);
    assert(!dense_model_chain.enabled);
    dense_model_chain =
        bn_transformer_prefill_dense_model_chain_policy(1, 1, 1, 1);
    assert(!dense_model_chain.enabled);
    dense_model_chain =
        bn_transformer_prefill_dense_model_chain_policy(1, 1, 0, 0);
    assert(!dense_model_chain.enabled);

    BnTransformerPrefillHybridModelChainPolicy hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(1, 1, 0, 1, 0, 0);
    assert(hybrid_model_chain.enabled);
    hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(0, 1, 0, 1, 0, 0);
    assert(!hybrid_model_chain.enabled);
    hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(1, 0, 0, 1, 0, 0);
    assert(!hybrid_model_chain.enabled);
    hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(1, 1, 1, 1, 0, 0);
    assert(!hybrid_model_chain.enabled);
    hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(1, 1, 0, 0, 0, 0);
    assert(!hybrid_model_chain.enabled);
    hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(1, 1, 0, 1, 1, 0);
    assert(!hybrid_model_chain.enabled);

    BnTransformerPrefillAttentionModePolicy attention_mode =
        bn_transformer_prefill_attention_mode_policy(0, 0, 0);
    assert(attention_mode.use_batched_attention);
    attention_mode =
        bn_transformer_prefill_attention_mode_policy(1, 0, 0);
    assert(!attention_mode.use_batched_attention);
    attention_mode =
        bn_transformer_prefill_attention_mode_policy(0, 1, 0);
    assert(!attention_mode.use_batched_attention);
    attention_mode =
        bn_transformer_prefill_attention_mode_policy(0, 0, 1);
    assert(!attention_mode.use_batched_attention);
    attention_mode =
        bn_transformer_prefill_attention_mode_policy(1, 1, 1);
    assert(!attention_mode.use_batched_attention);

    BnTransformerPrefillDenseLayerBatchPolicy dense_layer_batch =
        bn_transformer_prefill_dense_layer_batch_policy(
            1, 0, 1, 16, 16, 0, 10000.0f, 10000.0f,
            prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    assert(dense_layer_batch.enabled);
    dense_layer_batch = bn_transformer_prefill_dense_layer_batch_policy(
        0, 0, 1, 16, 16, 0, 10000.0f, 10000.0f,
        prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    assert(!dense_layer_batch.enabled);
    dense_layer_batch = bn_transformer_prefill_dense_layer_batch_policy(
        1, 1, 1, 16, 16, 0, 10000.0f, 10000.0f,
        prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    assert(!dense_layer_batch.enabled);
    dense_layer_batch = bn_transformer_prefill_dense_layer_batch_policy(
        1, 0, 1, 15, 16, 0, 10000.0f, 10000.0f,
        prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    assert(!dense_layer_batch.enabled);
    dense_layer_batch = bn_transformer_prefill_dense_layer_batch_policy(
        1, 0, 1, 16, 16, 0, 10000.0f, 10000.0f,
        bn_transformer_prefill_layer_kind_policy(&prefill_moe_lw),
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    assert(!dense_layer_batch.enabled);
    dense_layer_batch = bn_transformer_prefill_dense_layer_batch_policy(
        1, 0, 1, 16, 16, 0, 10000.0f, 10000.0f,
        prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0);
    assert(dense_layer_batch.enabled);

    BnTransformerPrefillDenseLayerChainPolicy dense_layer_chain =
        bn_transformer_prefill_dense_layer_chain_policy(
            1, 1, 0, 16, 16, 10000.0f, 10000.0f, 1,
            prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 0);
    assert(dense_layer_chain.enabled);
    dense_layer_chain = bn_transformer_prefill_dense_layer_chain_policy(
        1, 0, 0, 16, 16, 10000.0f, 10000.0f, 1,
        prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 0);
    assert(!dense_layer_chain.enabled);
    dense_layer_chain = bn_transformer_prefill_dense_layer_chain_policy(
        1, 1, 0, 16, 16, 10000.0f, 10000.0f, 0,
        prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 0);
    assert(!dense_layer_chain.enabled);
    dense_layer_chain = bn_transformer_prefill_dense_layer_chain_policy(
        1, 1, 0, 16, 16, 10000.0f, 10000.0f, 1,
        bn_transformer_prefill_layer_kind_policy(&prefill_moe_lw),
        1, 1, 0, 0, 0, 0, 0, 0);
    assert(!dense_layer_chain.enabled);
    dense_layer_chain = bn_transformer_prefill_dense_layer_chain_policy(
        1, 1, 0, 16, 16, 10000.0f, 10000.0f, 1,
        prefill_layer_kind, 1, 1, 1, 0, 0, 0, 0, 0);
    assert(!dense_layer_chain.enabled);
    dense_layer_chain = bn_transformer_prefill_dense_layer_chain_policy(
        1, 1, 0, 16, 16, 10000.0f, 10000.0f, 1,
        prefill_layer_kind, 1, 1, 0, 0, 0, 1, 1, 0);
    assert(dense_layer_chain.enabled);

    BnGPUBackend dense_layer_gpu = {0};
    assert(!bn_transformer_prefill_dense_layer_gpu_available(
        NULL, 1, 1, 1, 1, 1, 1));
    assert(!bn_transformer_prefill_dense_layer_gpu_available(
        &dense_layer_gpu, 1, 1, 1, 1, 1, 1));
    dense_layer_gpu.prefill_dense_layer = mock_prefill_dense_layer;
    assert(bn_transformer_prefill_dense_layer_gpu_available(
        &dense_layer_gpu, 1, 1, 1, 1, 1, 1));
    assert(!bn_transformer_prefill_dense_layer_gpu_available(
        &dense_layer_gpu, 0, 1, 1, 1, 1, 1));
    assert(!bn_transformer_prefill_dense_layer_gpu_available(
        &dense_layer_gpu, 1, 0, 1, 1, 1, 1));
    assert(!bn_transformer_prefill_dense_layer_gpu_available(
        &dense_layer_gpu, 1, 1, 0, 1, 1, 1));
    assert(!bn_transformer_prefill_dense_layer_gpu_available(
        &dense_layer_gpu, 1, 1, 1, 0, 1, 1));
    assert(!bn_transformer_prefill_dense_layer_gpu_available(
        &dense_layer_gpu, 1, 1, 1, 1, 0, 1));
    assert(!bn_transformer_prefill_dense_layer_gpu_available(
        &dense_layer_gpu, 1, 1, 1, 1, 1, 0));

    BnGPUBackend matmul_gpu = {0};
    assert(!bn_transformer_prefill_quant_matmul_gpu_available(
        NULL, 1, 1, 1, 1));
    assert(!bn_transformer_prefill_quant_matmul_gpu_available(
        &matmul_gpu, 1, 1, 1, 1));
    matmul_gpu.matmul = mock_gpu_matmul;
    assert(bn_transformer_prefill_quant_matmul_gpu_available(
        &matmul_gpu, 1, 1, 1, 1));
    setenv("BN_GPU_DISABLE_PREFILL_MATMUL", "1", 1);
    test_gpu_runtime_refresh(&matmul_gpu);
    assert(!bn_transformer_prefill_quant_matmul_gpu_available(
        &matmul_gpu, 1, 1, 1, 1));
    unsetenv("BN_GPU_DISABLE_PREFILL_MATMUL");
    test_gpu_runtime_refresh(&matmul_gpu);
    assert(!bn_transformer_prefill_quant_matmul_gpu_available(
        &matmul_gpu, 0, 1, 1, 1));
    assert(!bn_transformer_prefill_quant_matmul_gpu_available(
        &matmul_gpu, 1, 0, 1, 1));
    assert(!bn_transformer_prefill_quant_matmul_gpu_available(
        &matmul_gpu, 1, 1, 0, 1));
    assert(!bn_transformer_prefill_quant_matmul_gpu_available(
        &matmul_gpu, 1, 1, 1, 0));

    BnGPUBackend matmul_batch_gpu = {0};
    assert(!bn_transformer_prefill_quant_matmul_batch_gpu_available(
        NULL, 2, 1, 1, 1, 1));
    assert(!bn_transformer_prefill_quant_matmul_batch_gpu_available(
        &matmul_batch_gpu, 2, 1, 1, 1, 1));
    matmul_batch_gpu.matmul_batch = mock_gpu_matmul_batch;
    assert(bn_transformer_prefill_quant_matmul_batch_gpu_available(
        &matmul_batch_gpu, 2, 1, 1, 1, 1));
    setenv("BN_GPU_DISABLE_PREFILL_MATMUL", "1", 1);
    test_gpu_runtime_refresh(&matmul_batch_gpu);
    assert(!bn_transformer_prefill_quant_matmul_batch_gpu_available(
        &matmul_batch_gpu, 2, 1, 1, 1, 1));
    unsetenv("BN_GPU_DISABLE_PREFILL_MATMUL");
    test_gpu_runtime_refresh(&matmul_batch_gpu);
    assert(!bn_transformer_prefill_quant_matmul_batch_gpu_available(
        &matmul_batch_gpu, 1, 1, 1, 1, 1));
    assert(!bn_transformer_prefill_quant_matmul_batch_gpu_available(
        &matmul_batch_gpu, 17, 1, 1, 1, 1));
    assert(!bn_transformer_prefill_quant_matmul_batch_gpu_available(
        &matmul_batch_gpu, 2, 0, 1, 1, 1));
    assert(!bn_transformer_prefill_quant_matmul_batch_gpu_available(
        &matmul_batch_gpu, 2, 1, 0, 1, 1));
    assert(!bn_transformer_prefill_quant_matmul_batch_gpu_available(
        &matmul_batch_gpu, 2, 1, 1, 0, 1));
    assert(!bn_transformer_prefill_quant_matmul_batch_gpu_available(
        &matmul_batch_gpu, 2, 1, 1, 1, 0));

    BnGPUBackend moe_prefill_gpu = {0};
    moe_prefill_gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&moe_prefill_gpu);
    assert(bn_transformer_gpu_moe_prefill_backend_available(
        &moe_prefill_gpu));
    setenv("BN_GPU_DISABLE_PREFILL_MATMUL", "1", 1);
    test_gpu_runtime_refresh(&moe_prefill_gpu);
    assert(!bn_transformer_gpu_moe_prefill_backend_available(
        &moe_prefill_gpu));
    BnConfig cpu_batch_fallback_config = {0};
    assert(bn_transformer_gpu_cpu_batch_prefill_fallback_enabled(
        &moe_prefill_gpu, &cpu_batch_fallback_config));
    cpu_batch_fallback_config.ssm_inner_size = 16;
    cpu_batch_fallback_config.full_attn_interval = 4;
    assert(!bn_transformer_gpu_cpu_batch_prefill_fallback_enabled(
        &moe_prefill_gpu, &cpu_batch_fallback_config));
    unsetenv("BN_GPU_DISABLE_PREFILL_MATMUL");
    test_gpu_runtime_refresh(&moe_prefill_gpu);
    setenv("BN_CUDA_DISABLE_MOE_PREFILL", "1", 1);
    test_gpu_runtime_refresh(&moe_prefill_gpu);
    assert(!bn_transformer_gpu_moe_prefill_backend_available(
        &moe_prefill_gpu));
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL");
    test_gpu_runtime_refresh(&moe_prefill_gpu);
    assert(bn_transformer_gpu_moe_prefill_backend_available(
        &moe_prefill_gpu));
    cpu_batch_fallback_config.ssm_inner_size = 0;
    cpu_batch_fallback_config.full_attn_interval = 0;
    assert(!bn_transformer_gpu_cpu_batch_prefill_fallback_enabled(
        &moe_prefill_gpu, &cpu_batch_fallback_config));
    cpu_batch_fallback_config.n_experts = 8;
    setenv("BN_CUDA_DISABLE_MOE_PREFILL", "1", 1);
    test_gpu_runtime_refresh(&moe_prefill_gpu);
    assert(bn_transformer_gpu_cpu_batch_prefill_fallback_enabled(
        &moe_prefill_gpu, &cpu_batch_fallback_config));
    cpu_batch_fallback_config.ssm_inner_size = 16;
    cpu_batch_fallback_config.full_attn_interval = 4;
    assert(!bn_transformer_gpu_cpu_batch_prefill_fallback_enabled(
        &moe_prefill_gpu, &cpu_batch_fallback_config));
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL");
    test_gpu_runtime_refresh(&moe_prefill_gpu);

    BnConfig prefill_dense_c = {0};
    BnGPUBackend prefill_dense_gpu = {0};
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, NULL) == 16);
    prefill_dense_gpu.kind = BN_GPU_BACKEND_CUDA;
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 16);
    prefill_dense_gpu.caps |= BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL;
    assert(bn_transformer_prefill_attention_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 16);
    prefill_dense_c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT;
    assert(bn_transformer_prefill_attention_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 1);
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 1);
    setenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "11", 1);
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    assert(bn_transformer_prefill_attention_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 11);
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 11);
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    prefill_dense_gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_prefill_attention_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 16);
    prefill_dense_gpu.kind = BN_GPU_BACKEND_CUDA;
    prefill_dense_gpu.caps &= ~BN_GPU_CAP_REFERENCE_RECURRENT_PREFILL;
    assert(bn_transformer_prefill_attention_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 16);
    prefill_dense_c.policy_flags = 0;
    prefill_dense_c.act_type = BN_MODEL_ACTIVATION_GELU;
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 16);
    prefill_dense_gpu.caps |= BN_GPU_CAP_FP32_GELU;
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 2);
    prefill_dense_c.act_type = BN_MODEL_ACTIVATION_SILU;
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 16);
    prefill_dense_c.act_type = BN_MODEL_ACTIVATION_GELU;
    setenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "9", 1);
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 9);
    prefill_dense_c.act_type = BN_MODEL_ACTIVATION_SILU;
    prefill_dense_gpu.caps &= ~BN_GPU_CAP_FP32_GELU;
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    assert(bn_transformer_prefill_dense_chain_enabled(&prefill_dense_gpu));
    setenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN", "1", 1);
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    assert(!bn_transformer_prefill_dense_chain_enabled(&prefill_dense_gpu));
    prefill_dense_gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_prefill_dense_chain_enabled(&prefill_dense_gpu));
    prefill_dense_gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    assert(!bn_transformer_prefill_dense_ffn_batch_tokens_allowed(
        &prefill_dense_gpu, &prefill_dense_c, 15));
    assert(bn_transformer_prefill_dense_ffn_batch_tokens_allowed(
        &prefill_dense_gpu, &prefill_dense_c, 16));
    prefill_dense_gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_prefill_dense_ffn_batch_tokens_allowed(
        &prefill_dense_gpu, &prefill_dense_c, 1));

    BnTransformerSSMShapePolicy prefill_ssm_shape = {0};
    BnConfig prefill_ssm_c = {0};
    prefill_ssm_c.ssm_time_step_rank = 16;
    prefill_ssm_c.ssm_state_size = 64;
    prefill_ssm_c.ssm_inner_size = 256;
    prefill_ssm_c.ssm_group_count = 4;
    prefill_ssm_c.ssm_conv_kernel = 4;
    assert(bn_transformer_ssm_shape_policy(&prefill_ssm_shape,
                                           &prefill_ssm_c));
    BnTransformerSSMShapePolicy invalid_prefill_ssm_shape =
        prefill_ssm_shape;
    invalid_prefill_ssm_shape.head_v_dim = 0;

    BnTransformerPrefillSSMChainPolicy ssm_chain =
        bn_transformer_prefill_ssm_chain_policy(
            1, prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0,
            &prefill_ssm_shape);
    assert(ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        0, prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0,
        &prefill_ssm_shape);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, bn_transformer_prefill_layer_kind_policy(&prefill_moe_lw),
        1, 1, 0, 0, 0, 0, 0, &prefill_ssm_shape);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, prefill_layer_kind, 0, 1, 0, 0, 0, 0, 0,
        &prefill_ssm_shape);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, prefill_layer_kind, 1, 1, 1, 0, 0, 0, 0,
        &prefill_ssm_shape);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, prefill_layer_kind, 1, 1, 0, 0, 1, 1, 0,
        &prefill_ssm_shape);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, NULL);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0,
        &invalid_prefill_ssm_shape);
    assert(!ssm_chain.enabled);
    assert(!bn_transformer_prefill_ssm_layer_backend_available(NULL));
    assert(!bn_transformer_prefill_ssm_layer_backend_available(
        &prefill_dense_gpu));
    prefill_dense_gpu.kind = BN_GPU_BACKEND_CUDA;
    prefill_dense_gpu.prefill_ssm_layer = mock_prefill_ssm_layer;
    assert(bn_transformer_prefill_ssm_layer_backend_available(
        &prefill_dense_gpu));
    prefill_dense_c.ssm_inner_size = 128;
    assert(!bn_transformer_prefill_ssm_dense_chain_available(
        &prefill_dense_gpu, &prefill_dense_c, 15));
    assert(bn_transformer_prefill_ssm_dense_chain_available(
        &prefill_dense_gpu, &prefill_dense_c, 16));
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN");
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    assert(bn_transformer_prefill_ssm_run_chain_enabled(&prefill_dense_gpu));
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN", "1", 1);
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    assert(!bn_transformer_prefill_ssm_run_chain_enabled(&prefill_dense_gpu));
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN");
    unsetenv("BN_CUDA_DISABLE_SSM_FFN_FUSE");
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    assert(bn_transformer_prefill_ssm_ffn_fuse_allowed(&prefill_dense_gpu));
    setenv("BN_CUDA_DISABLE_SSM_FFN_FUSE", "1", 1);
    test_gpu_runtime_refresh(&prefill_dense_gpu);
    assert(!bn_transformer_prefill_ssm_ffn_fuse_allowed(&prefill_dense_gpu));
    unsetenv("BN_CUDA_DISABLE_SSM_FFN_FUSE");
    memset(&prefill_dense_gpu, 0, sizeof(prefill_dense_gpu));

    BnTransformerPrefillSSMMoEChainPolicy ssm_moe_chain =
        bn_transformer_prefill_ssm_moe_chain_policy(
            1, bn_transformer_prefill_layer_kind_policy(&prefill_moe_lw),
            0, 0, 0, 0, 0, &prefill_ssm_shape);
    assert(ssm_moe_chain.enabled);
    ssm_moe_chain = bn_transformer_prefill_ssm_moe_chain_policy(
        1, prefill_layer_kind, 0, 0, 0, 0, 0, &prefill_ssm_shape);
    assert(!ssm_moe_chain.enabled);
    ssm_moe_chain = bn_transformer_prefill_ssm_moe_chain_policy(
        1, bn_transformer_prefill_layer_kind_policy(&prefill_moe_lw),
        0, 1, 0, 0, 0, &prefill_ssm_shape);
    assert(!ssm_moe_chain.enabled);
    ssm_moe_chain = bn_transformer_prefill_ssm_moe_chain_policy(
        1, bn_transformer_prefill_layer_kind_policy(&prefill_moe_lw),
        0, 0, 1, 0, 1, &prefill_ssm_shape);
    assert(!ssm_moe_chain.enabled);
    ssm_moe_chain = bn_transformer_prefill_ssm_moe_chain_policy(
        1, bn_transformer_prefill_layer_kind_policy(&prefill_moe_lw),
        0, 0, 0, 0, 0, NULL);
    assert(!ssm_moe_chain.enabled);
    ssm_moe_chain = bn_transformer_prefill_ssm_moe_chain_policy(
        1, bn_transformer_prefill_layer_kind_policy(&prefill_moe_lw),
        0, 0, 0, 0, 0, &invalid_prefill_ssm_shape);
    assert(!ssm_moe_chain.enabled);

    BnGPUBackend prefill_gpu = {0};
    prefill_gpu.kind = BN_GPU_BACKEND_CUDA;
    prefill_gpu.moe_route_routed_ffn_batch_norm_resid =
        mock_moe_route_routed_ffn_batch_norm_resid;
    BnConfig prefill_c = {0};
    prefill_c.n_experts = 2;
    prefill_c.n_experts_active = 2;
    prefill_c.moe_intermediate_size = 4096;
    prefill_c.dim = 2048;
    BnMoEExpertMap prefill_map = {0};
    prefill_map.gate_type = BN_GGUF_TENSOR_Q4_K;
    prefill_map.up_type = BN_GGUF_TENSOR_Q4_K;
    prefill_map.down_type = BN_GGUF_TENSOR_Q6_K;
    prefill_map.gate_rows = 4096;
    prefill_map.gate_cols = 2048;
    prefill_map.up_rows = 4096;
    prefill_map.up_cols = 2048;
    prefill_map.down_rows = 2048;
    prefill_map.down_cols = 4096;
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");
    assert(!bn_transformer_prefill_moe_ffn_batch_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0));
    setenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN", "1", 1);
    test_gpu_runtime_refresh(&prefill_gpu);
    assert(bn_transformer_prefill_moe_ffn_batch_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0));
    prefill_gpu.prefill_ssm_layer = mock_prefill_ssm_layer;
    assert(bn_transformer_prefill_ssm_moe_chain_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0, 1));
    prefill_gpu.prefill_moe_layer = mock_prefill_moe_layer;
    assert(bn_transformer_prefill_moe_layer_chain_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0, 1));
    prefill_gpu.prefill_moe_layer = NULL;
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    test_gpu_runtime_refresh(&prefill_gpu);
    assert(bn_transformer_prefill_moe_chain_min_tokens(
        &prefill_c, &prefill_gpu) == 1);
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "7", 1);
    test_gpu_runtime_refresh(&prefill_gpu);
    assert(bn_transformer_prefill_moe_chain_min_tokens(
        &prefill_c, &prefill_gpu) == 7);
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    unsetenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");
    test_gpu_runtime_refresh(&prefill_gpu);
    assert(!bn_transformer_prefill_moe_chain_debug_enabled(&prefill_gpu));
    setenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN", "1", 1);
    test_gpu_runtime_refresh(&prefill_gpu);
    assert(bn_transformer_prefill_moe_chain_debug_enabled(&prefill_gpu));
    unsetenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    test_gpu_runtime_refresh(&prefill_gpu);
    assert(!bn_transformer_prefill_ssm_moe_chain_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0, 16));
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    test_gpu_runtime_refresh(&prefill_gpu);
    prefill_gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_prefill_moe_ffn_batch_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0));
    assert(!bn_transformer_prefill_ssm_moe_chain_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0, 16));
    prefill_gpu.kind = BN_GPU_BACKEND_CUDA;
    prefill_gpu.moe_route_routed_ffn_batch_norm_resid = NULL;
    assert(!bn_transformer_prefill_moe_ffn_batch_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0));
    assert(!bn_transformer_prefill_ssm_moe_chain_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0, 16));
    unsetenv("BN_CUDA_ENABLE_ALL_ACTIVE_TWO_KQUANT_MOE_FAST_FFN");

    BnTransformerPrefillSSMFFNFusePolicy ssm_ffn_fuse =
        bn_transformer_prefill_ssm_ffn_fuse_policy(
            1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
    assert(ssm_ffn_fuse.enabled);
    ssm_ffn_fuse = bn_transformer_prefill_ssm_ffn_fuse_policy(
        0, 1, 1, 1, 1, 1, 0, 0, 0, 0);
    assert(!ssm_ffn_fuse.enabled);
    ssm_ffn_fuse = bn_transformer_prefill_ssm_ffn_fuse_policy(
        1, 0, 1, 1, 1, 1, 0, 0, 0, 0);
    assert(!ssm_ffn_fuse.enabled);
    ssm_ffn_fuse = bn_transformer_prefill_ssm_ffn_fuse_policy(
        1, 1, 0, 1, 1, 1, 0, 0, 0, 0);
    assert(!ssm_ffn_fuse.enabled);
    ssm_ffn_fuse = bn_transformer_prefill_ssm_ffn_fuse_policy(
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
    assert(!ssm_ffn_fuse.enabled);
    ssm_ffn_fuse = bn_transformer_prefill_ssm_ffn_fuse_policy(
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0);
    assert(!ssm_ffn_fuse.enabled);
    ssm_ffn_fuse = bn_transformer_prefill_ssm_ffn_fuse_policy(
        1, 1, 1, 1, 1, 1, 0, 0, 1, 1);
    assert(!ssm_ffn_fuse.enabled);

    BnTransformerPrefillRawAttentionPolicy raw_attention =
        bn_transformer_prefill_raw_attention_policy(
            1, 1, 1, 1, 0, 0, 0, 16, 16, 10000.0f, 10000.0f,
            0, 0, 0, 0, 0, 0);
    assert(raw_attention.eligible);
    assert(raw_attention.fuses_input_norm);
    raw_attention = bn_transformer_prefill_raw_attention_policy(
        1, 1, 1, 1, 0, 0, 0, 8, 16, 10000.0f, 10000.0f,
        0, 0, 0, 0, 0, 0);
    assert(raw_attention.eligible);
    assert(!raw_attention.fuses_input_norm);
    raw_attention = bn_transformer_prefill_raw_attention_policy(
        1, 0, 1, 1, 0, 0, 0, 16, 16, 10000.0f, 10000.0f,
        0, 0, 0, 0, 0, 0);
    assert(!raw_attention.eligible);
    assert(!raw_attention.fuses_input_norm);
    raw_attention = bn_transformer_prefill_raw_attention_policy(
        1, 1, 1, 1, 1, 0, 0, 16, 16, 10000.0f, 10000.0f,
        0, 0, 0, 0, 0, 0);
    assert(!raw_attention.eligible);
    raw_attention = bn_transformer_prefill_raw_attention_policy(
        1, 1, 1, 1, 0, 1, 0, 16, 16, 10000.0f, 10000.0f,
        0, 0, 0, 0, 0, 0);
    assert(!raw_attention.eligible);
    raw_attention = bn_transformer_prefill_raw_attention_policy(
        1, 1, 1, 1, 0, 0, 1, 16, 16, 10000.0f, 10000.0f,
        0, 0, 0, 0, 0, 0);
    assert(!raw_attention.eligible);
    raw_attention = bn_transformer_prefill_raw_attention_policy(
        1, 1, 1, 1, 0, 0, 0, 16, 16, 10000.0f, 10001.0f,
        0, 0, 0, 0, 0, 0);
    assert(!raw_attention.eligible);
    raw_attention = bn_transformer_prefill_raw_attention_policy(
        1, 1, 1, 1, 0, 0, 0, 16, 16, 10000.0f, 10000.0f,
        1, 0, 0, 0, 0, 0);
    assert(!raw_attention.eligible);
    raw_attention = bn_transformer_prefill_raw_attention_policy(
        1, 1, 1, 1, 0, 0, 0, 16, 16, 10000.0f, 10000.0f,
        0, 0, 0, 0, 1, 1);
    assert(!raw_attention.eligible);

    BnTransformerPrefillRawAttentionCallPolicy raw_attention_call =
        bn_transformer_prefill_raw_attention_call_policy(raw_attention);
    assert(raw_attention_call.preferred_kind ==
           BN_TRANSFORMER_PREFILL_RAW_ATTENTION_PLAIN);
    raw_attention = bn_transformer_prefill_raw_attention_policy(
        1, 1, 1, 1, 0, 0, 0, 16, 16, 10000.0f, 10000.0f,
        0, 0, 0, 0, 0, 0);
    raw_attention_call =
        bn_transformer_prefill_raw_attention_call_policy(raw_attention);
    assert(raw_attention_call.preferred_kind ==
           BN_TRANSFORMER_PREFILL_RAW_ATTENTION_NORM_RESID);

    BnGPUBackend raw_attention_gpu = {0};
    assert(!bn_transformer_prefill_raw_attention_gpu_available(NULL));
    assert(!bn_transformer_prefill_raw_attention_gpu_available(
        &raw_attention_gpu));
    assert(!bn_transformer_prefill_raw_attention_norm_resid_gpu_available(
        &raw_attention_gpu));
    raw_attention_gpu.prefill_qkv_attention_wo =
        mock_prefill_qkv_attention_wo;
    assert(bn_transformer_prefill_raw_attention_gpu_available(
        &raw_attention_gpu));
    assert(!bn_transformer_prefill_raw_attention_norm_resid_gpu_available(
        &raw_attention_gpu));
    raw_attention_gpu.prefill_qkv_attention_wo_norm_resid =
        mock_prefill_qkv_attention_wo_norm_resid;
    assert(bn_transformer_prefill_raw_attention_norm_resid_gpu_available(
        &raw_attention_gpu));

    BnGPUBackend attention_gpu = {0};
    assert(!bn_transformer_prefill_attention_gpu_available(NULL));
    assert(!bn_transformer_prefill_attention_gpu_available(&attention_gpu));
    assert(!bn_transformer_prefill_attention_wo_gpu_available(
        &attention_gpu));
    attention_gpu.prefill_attention = mock_prefill_attention;
    assert(bn_transformer_prefill_attention_gpu_available(&attention_gpu));
    assert(!bn_transformer_prefill_attention_wo_gpu_available(
        &attention_gpu));
    attention_gpu.prefill_attention_wo = mock_prefill_attention_wo;
    assert(bn_transformer_prefill_attention_wo_gpu_available(
        &attention_gpu));

    BnGPUAttentionPrefillPlan prepared_plan = {0};
    prepared_plan.n_tokens = 2; prepared_plan.q_gated = 1;
    prepared_plan.q_row_stride = 39; prepared_plan.rope_freq_offset = 17;
    float prepared_values[5] = {1, 2, 3, 4, 5};
    MockPreparedAttention prepared_expected = {
        &prepared_values[0], &prepared_values[1], &prepared_values[2],
        &prepared_values[3], &prepared_values[4], &prepared_values[2],
        &prepared_values[3], &prepared_plan, 0
    };
    assert(!bn_gpu_backend_can_prefill_attention_prepared(NULL));
    assert(!bn_gpu_backend_can_prefill_attention_prepared(&attention_gpu));
    assert(bn_transformer_gpu_prefill_attention_prepared_backend_run(
        &attention_gpu, prepared_expected.out, prepared_expected.k_out,
        prepared_expected.q, prepared_expected.k, prepared_expected.v,
        prepared_expected.q_norm, prepared_expected.k_norm, &prepared_plan) == -1);
    attention_gpu.ctx = &prepared_expected;
    attention_gpu.prefill_attention_prepared = mock_prefill_attention_prepared;
    assert(bn_gpu_backend_can_prefill_attention_prepared(&attention_gpu));
    assert(bn_transformer_gpu_prefill_attention_prepared_backend_run(
        &attention_gpu, prepared_expected.out, prepared_expected.k_out,
        prepared_expected.q, prepared_expected.k, prepared_expected.v,
        prepared_expected.q_norm, prepared_expected.k_norm, &prepared_plan) == -7);
    assert(prepared_expected.calls == 1);
    assert(prepared_values[0] == 1 && prepared_values[1] == 2);

    BnTransformerPrefillAttentionBatchPolicy attention_batch =
        bn_transformer_prefill_attention_batch_policy(
            0, 1, 1, 1, 1, 1, 16, 16, 0, 0, 0);
    assert(attention_batch.eligible);
    assert(attention_batch.fuses_output_projection);
    attention_batch = bn_transformer_prefill_attention_batch_policy(
        1, 1, 1, 1, 1, 1, 16, 16, 0, 0, 0);
    assert(!attention_batch.eligible);
    assert(!attention_batch.fuses_output_projection);
    attention_batch = bn_transformer_prefill_attention_batch_policy(
        0, 0, 1, 1, 1, 1, 16, 16, 0, 0, 0);
    assert(!attention_batch.eligible);
    attention_batch = bn_transformer_prefill_attention_batch_policy(
        0, 1, 0, 1, 1, 1, 16, 16, 0, 0, 0);
    assert(!attention_batch.eligible);
    attention_batch = bn_transformer_prefill_attention_batch_policy(
        0, 1, 1, 1, 0, 1, 16, 16, 0, 0, 0);
    assert(!attention_batch.eligible);
    attention_batch = bn_transformer_prefill_attention_batch_policy(
        0, 1, 1, 1, 1, 1, 15, 16, 0, 0, 0);
    assert(!attention_batch.eligible);
    attention_batch = bn_transformer_prefill_attention_batch_policy(
        0, 1, 1, 1, 1, 0, 16, 16, 0, 0, 0);
    assert(attention_batch.eligible);
    assert(!attention_batch.fuses_output_projection);
    attention_batch = bn_transformer_prefill_attention_batch_policy(
        0, 1, 1, 1, 1, 1, 16, 16, 1, 0, 0);
    assert(attention_batch.eligible);
    assert(!attention_batch.fuses_output_projection);
    attention_batch = bn_transformer_prefill_attention_batch_policy(
        0, 1, 1, 1, 1, 1, 16, 16, 0, 1, 1);
    assert(attention_batch.eligible);
    assert(!attention_batch.fuses_output_projection);

    BnTransformerPrefillAttentionBatchCallPolicy attention_call =
        bn_transformer_prefill_attention_batch_call_policy(attention_batch);
    assert(attention_call.preferred_kind ==
           BN_TRANSFORMER_PREFILL_ATTENTION_BATCH_PLAIN);
    attention_batch = bn_transformer_prefill_attention_batch_policy(
        0, 1, 1, 1, 1, 1, 16, 16, 0, 0, 0);
    attention_call =
        bn_transformer_prefill_attention_batch_call_policy(attention_batch);
    assert(attention_call.preferred_kind ==
           BN_TRANSFORMER_PREFILL_ATTENTION_BATCH_WO);

    BnTransformerPrefillFFNBatchPolicy ffn_batch =
        bn_transformer_prefill_ffn_batch_policy(
            1, 1, 1, 1, 16, 16, 0, 0, 0, 0);
    assert(ffn_batch.eligible);
    assert(ffn_batch.fuses_norm_residual);
    ffn_batch = bn_transformer_prefill_ffn_batch_policy(
        1, 1, 1, 1, 8, 16, 0, 0, 0, 0);
    assert(ffn_batch.eligible);
    assert(!ffn_batch.fuses_norm_residual);
    ffn_batch = bn_transformer_prefill_ffn_batch_policy(
        1, 1, 1, 1, 8, 16, 1, 0, 0, 0);
    assert(!ffn_batch.eligible);
    assert(!ffn_batch.fuses_norm_residual);
    ffn_batch = bn_transformer_prefill_ffn_batch_policy(
        0, 1, 1, 1, 16, 16, 0, 0, 0, 0);
    assert(!ffn_batch.eligible);
    ffn_batch = bn_transformer_prefill_ffn_batch_policy(
        1, 0, 1, 1, 16, 16, 0, 0, 0, 0);
    assert(!ffn_batch.eligible);
    ffn_batch = bn_transformer_prefill_ffn_batch_policy(
        1, 1, 1, 1, 16, 16, 0, 1, 0, 0);
    assert(!ffn_batch.eligible);
    ffn_batch = bn_transformer_prefill_ffn_batch_policy(
        1, 1, 1, 1, 16, 16, 0, 0, 1, 1);
    assert(!ffn_batch.eligible);
    ffn_batch = bn_transformer_prefill_ffn_batch_policy(
        1, 1, 0, 1, 16, 16, 0, 0, 0, 0);
    assert(ffn_batch.eligible);
    assert(!ffn_batch.fuses_norm_residual);
    ffn_batch = bn_transformer_prefill_ffn_batch_policy(
        1, 1, 1, 0, 16, 16, 0, 0, 0, 0);
    assert(ffn_batch.eligible);
    assert(!ffn_batch.fuses_norm_residual);

    BnTransformerPrefillFFNBatchCallPolicy ffn_call =
        bn_transformer_prefill_ffn_batch_call_policy(1, 1, 1, 1);
    assert(ffn_call.kind == BN_TRANSFORMER_PREFILL_FFN_BATCH_NORM_RESID);
    ffn_call = bn_transformer_prefill_ffn_batch_call_policy(1, 1, 1, 0);
    assert(ffn_call.kind == BN_TRANSFORMER_PREFILL_FFN_BATCH_NORM);
    ffn_call = bn_transformer_prefill_ffn_batch_call_policy(1, 0, 1, 1);
    assert(ffn_call.kind == BN_TRANSFORMER_PREFILL_FFN_BATCH_NORM);
    ffn_call = bn_transformer_prefill_ffn_batch_call_policy(1, 1, 0, 0);
    assert(ffn_call.kind == BN_TRANSFORMER_PREFILL_FFN_BATCH_PLAIN);
    ffn_call = bn_transformer_prefill_ffn_batch_call_policy(0, 1, 1, 1);
    assert(ffn_call.kind == BN_TRANSFORMER_PREFILL_FFN_BATCH_PLAIN);

    assert(!bn_transformer_prefill_can_prepared_kquant_type(
        NULL, BN_GGUF_TENSOR_Q4_K));
    int prefill_supports_prepared_kquant =
        bn_transformer_prefill_cpu_ops()->supports_prepared_kquant;
    assert(bn_transformer_prefill_can_prepared_kquant_type(
               bn_transformer_prefill_cpu_ops(), BN_GGUF_TENSOR_Q4_K) ==
           prefill_supports_prepared_kquant);
    assert(bn_transformer_prefill_can_prepared_kquant_pair(
               bn_transformer_prefill_cpu_ops(),
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K) ==
           prefill_supports_prepared_kquant);
    assert(bn_transformer_prefill_can_prepared_kquant_triple(
               bn_transformer_prefill_cpu_ops(),
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K,
               BN_GGUF_TENSOR_Q6_K) == prefill_supports_prepared_kquant);
    assert(!bn_transformer_prefill_can_prepared_kquant_pair(
        bn_transformer_prefill_cpu_ops(), BN_GGUF_TENSOR_Q4_K,
        BN_GGUF_TENSOR_Q8_0));
    assert(bn_transformer_prefill_route_prepared_kquant_type_enabled(
               bn_transformer_prefill_cpu_ops(), NULL, 0, BN_QK_K,
               BN_GGUF_TENSOR_Q4_K) == prefill_supports_prepared_kquant);
    assert(!bn_transformer_prefill_route_prepared_kquant_type_enabled(
        bn_transformer_prefill_cpu_ops(), &route_gpu, 0, BN_QK_K,
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_transformer_prefill_route_prepared_kquant_type_enabled(
        bn_transformer_prefill_cpu_ops(), NULL, 1, BN_QK_K,
        BN_GGUF_TENSOR_Q4_K));
    assert(!bn_transformer_prefill_route_prepared_kquant_type_enabled(
        bn_transformer_prefill_cpu_ops(), NULL, 0, BN_QK_K - 1,
        BN_GGUF_TENSOR_Q4_K));
    assert(bn_transformer_prefill_route_prepared_kquant_pair_enabled(
               bn_transformer_prefill_cpu_ops(), NULL, 0, BN_QK_K,
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K) ==
           prefill_supports_prepared_kquant);
    assert(!bn_transformer_prefill_route_prepared_kquant_pair_enabled(
        bn_transformer_prefill_cpu_ops(), NULL, 0, BN_QK_K,
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q8_0));
    assert(bn_transformer_prefill_route_prepared_kquant_triple_enabled(
               bn_transformer_prefill_cpu_ops(), NULL, 0, BN_QK_K,
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K,
               BN_GGUF_TENSOR_Q6_K) == prefill_supports_prepared_kquant);
    assert(!bn_transformer_prefill_route_prepared_kquant_triple_enabled(
        bn_transformer_prefill_cpu_ops(), NULL, 0, BN_QK_K,
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0));
    int prepared_group[3] = {
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K
    };
    BnTransformerPrefillPreparedKQuantDispatchPolicy prepared_dispatch =
        bn_transformer_prefill_prepared_kquant_dispatch_policy(
            bn_transformer_prefill_cpu_ops(), NULL, 0, BN_QK_K,
            prepared_group, 3, 4);
    assert(prepared_dispatch.enabled == prefill_supports_prepared_kquant);
    prepared_dispatch =
        bn_transformer_prefill_prepared_kquant_dispatch_policy(
            bn_transformer_prefill_cpu_ops(), &route_gpu, 0, BN_QK_K,
            prepared_group, 3, 4);
    assert(!prepared_dispatch.enabled);
    prepared_dispatch =
        bn_transformer_prefill_prepared_kquant_dispatch_policy(
            bn_transformer_prefill_cpu_ops(), NULL, 1, BN_QK_K,
            prepared_group, 3, 4);
    assert(!prepared_dispatch.enabled);
    prepared_dispatch =
        bn_transformer_prefill_prepared_kquant_dispatch_policy(
            bn_transformer_prefill_cpu_ops(), NULL, 0, BN_QK_K - 1,
            prepared_group, 3, 4);
    assert(!prepared_dispatch.enabled);
    prepared_dispatch =
        bn_transformer_prefill_prepared_kquant_dispatch_policy(
            bn_transformer_prefill_cpu_ops(), NULL, 0, BN_QK_K,
            prepared_group, 5, 4);
    assert(!prepared_dispatch.enabled);
    prepared_group[2] = BN_GGUF_TENSOR_Q8_0;
    prepared_dispatch =
        bn_transformer_prefill_prepared_kquant_dispatch_policy(
            bn_transformer_prefill_cpu_ops(), NULL, 0, BN_QK_K,
            prepared_group, 3, 4);
    assert(!prepared_dispatch.enabled);
    assert(bn_transformer_prefill_same_quant_format_pair_stackable(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_transformer_prefill_same_quant_format_pair_stackable(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K));
    BnLayerWeights prefill_lw;
    memset(&prefill_lw, 0, sizeof(prefill_lw));
    prefill_lw.attn.wq.type = BN_GGUF_TENSOR_Q4_K;
    prefill_lw.attn.wq.rows = 64;
    prefill_lw.attn.wq.cols = 32;
    prefill_lw.attn.wk.type = BN_GGUF_TENSOR_Q5_K;
    prefill_lw.attn.wk.rows = 16;
    prefill_lw.attn.wk.cols = 32;
    prefill_lw.attn.wv.type = BN_GGUF_TENSOR_Q6_K;
    prefill_lw.attn.wv.rows = 16;
    prefill_lw.attn.wv.cols = 32;
    prefill_lw.attn.wo.type = BN_GGUF_TENSOR_Q4_K;
    prefill_lw.attn.wo.rows = 32;
    prefill_lw.attn.wo.cols = 64;
    prefill_lw.ffn.ffn_gate.type = BN_GGUF_TENSOR_Q4_K;
    prefill_lw.ffn.ffn_gate.rows = 96;
    prefill_lw.ffn.ffn_gate.cols = 32;
    prefill_lw.ffn.ffn_up.type = BN_GGUF_TENSOR_Q5_K;
    prefill_lw.ffn.ffn_up.rows = 96;
    prefill_lw.ffn.ffn_up.cols = 32;
    prefill_lw.ffn.ffn_down.type = BN_GGUF_TENSOR_Q6_K;
    prefill_lw.ffn.ffn_down.rows = 32;
    prefill_lw.ffn.ffn_down.cols = 96;
    prefill_lw.ssm.wqkv.type = BN_GGUF_TENSOR_Q4_K;
    prefill_lw.ssm.wqkv.rows = 80;
    prefill_lw.ssm.wqkv.cols = 32;
    prefill_lw.ssm.wz.type = BN_GGUF_TENSOR_Q5_K;
    prefill_lw.ssm.wz.rows = 24;
    prefill_lw.ssm.wz.cols = 32;
    prefill_lw.ssm.ssm_alpha.type = BN_GGUF_TENSOR_Q6_K;
    prefill_lw.ssm.ssm_alpha.rows = 12;
    prefill_lw.ssm.ssm_alpha.cols = 32;
    prefill_lw.ssm.ssm_beta.type = BN_GGUF_TENSOR_Q4_K;
    prefill_lw.ssm.ssm_beta.rows = 12;
    prefill_lw.ssm.ssm_beta.cols = 32;
    prefill_lw.ssm.ssm_out.type = BN_GGUF_TENSOR_Q5_K;
    prefill_lw.ssm.ssm_out.rows = 32;
    prefill_lw.ssm.ssm_out.cols = 48;
    BnTransformerPrefillAttentionProjectionTypes prefill_attn_types = {0};
    assert(bn_transformer_prefill_resolve_attention_projection_types(
        &prefill_attn_types, &prefill_lw));
    assert(prefill_attn_types.q_type == BN_GGUF_TENSOR_Q4_K);
    assert(prefill_attn_types.q_rows == 64);
    assert(prefill_attn_types.q_cols == 32);
    assert(prefill_attn_types.k_type == BN_GGUF_TENSOR_Q5_K);
    assert(prefill_attn_types.k_rows == 16);
    assert(prefill_attn_types.k_cols == 32);
    assert(prefill_attn_types.v_type == BN_GGUF_TENSOR_Q6_K);
    assert(prefill_attn_types.v_rows == 16);
    assert(prefill_attn_types.v_cols == 32);
    assert(prefill_attn_types.out_type == BN_GGUF_TENSOR_Q4_K);
    assert(prefill_attn_types.out_rows == 32);
    assert(prefill_attn_types.out_cols == 64);
    assert(!bn_transformer_prefill_resolve_attention_projection_types(
        NULL, &prefill_lw));
    assert(!bn_transformer_prefill_resolve_attention_projection_types(
        &prefill_attn_types, NULL));
    BnTransformerPrefillFFNProjectionTypes prefill_ffn_types = {0};
    assert(bn_transformer_prefill_resolve_ffn_projection_types(
        &prefill_ffn_types, &prefill_lw));
    assert(prefill_ffn_types.gate_type == BN_GGUF_TENSOR_Q4_K);
    assert(prefill_ffn_types.gate_rows == 96);
    assert(prefill_ffn_types.gate_cols == 32);
    assert(prefill_ffn_types.up_type == BN_GGUF_TENSOR_Q5_K);
    assert(prefill_ffn_types.up_rows == 96);
    assert(prefill_ffn_types.up_cols == 32);
    assert(prefill_ffn_types.down_type == BN_GGUF_TENSOR_Q6_K);
    assert(prefill_ffn_types.down_rows == 32);
    assert(prefill_ffn_types.down_cols == 96);
    assert(!bn_transformer_prefill_resolve_ffn_projection_types(
        NULL, &prefill_lw));
    assert(!bn_transformer_prefill_resolve_ffn_projection_types(
        &prefill_ffn_types, NULL));
    BnTransformerPrefillSSMProjectionTypes prefill_ssm_types = {0};
    assert(bn_transformer_prefill_resolve_ssm_projection_types(
        &prefill_ssm_types, &prefill_lw));
    assert(prefill_ssm_types.qkv_type == BN_GGUF_TENSOR_Q4_K);
    assert(prefill_ssm_types.qkv_rows == 80);
    assert(prefill_ssm_types.qkv_cols == 32);
    assert(prefill_ssm_types.z_type == BN_GGUF_TENSOR_Q5_K);
    assert(prefill_ssm_types.z_rows == 24);
    assert(prefill_ssm_types.z_cols == 32);
    assert(prefill_ssm_types.alpha_type == BN_GGUF_TENSOR_Q6_K);
    assert(prefill_ssm_types.alpha_rows == 12);
    assert(prefill_ssm_types.alpha_cols == 32);
    assert(prefill_ssm_types.beta_type == BN_GGUF_TENSOR_Q4_K);
    assert(prefill_ssm_types.beta_rows == 12);
    assert(prefill_ssm_types.beta_cols == 32);
    assert(prefill_ssm_types.out_type == BN_GGUF_TENSOR_Q5_K);
    assert(prefill_ssm_types.out_rows == 32);
    assert(prefill_ssm_types.out_cols == 48);
    assert(!bn_transformer_prefill_resolve_ssm_projection_types(
        NULL, &prefill_lw));
    assert(!bn_transformer_prefill_resolve_ssm_projection_types(
        &prefill_ssm_types, NULL));

    const BnPrefillCPUOps *prefill_ops = bn_transformer_prefill_cpu_ops();
    if (strcmp(prefill_ops->name, "avx2") == 0 ||
        strcmp(prefill_ops->name, "avx512") == 0)
        assert(prefill_ops->dispatch_ssm_heads == bn_tp_dispatch_fine);
    else
        assert(prefill_ops->dispatch_ssm_heads == bn_tp_dispatch);
    assert(bn_transformer_prefill_ssm_conv_silu_op(prefill_ops) ==
           prefill_ops->ssm_conv_silu);
    assert(bn_transformer_prefill_ssm_l2norm_op(prefill_ops) ==
           prefill_ops->ssm_l2norm);
    assert(bn_transformer_prefill_ssm_delta_op(prefill_ops) ==
           prefill_ops->ssm_delta);
    assert(bn_transformer_prefill_ssm_gate_op(prefill_ops) ==
           prefill_ops->ssm_gate);

    unsetenv("BN_PREFILL_PROFILE");
    assert(!bn_transformer_prefill_profile_enabled(test_cpu_policy()));
    setenv("BN_PREFILL_PROFILE", "1", 1);
    assert(bn_transformer_prefill_profile_enabled(test_cpu_policy()));
    unsetenv("BN_PREFILL_PROFILE");

    unsetenv("BN_PREFILL_ALLOW_HYBRID_BATCH");
    assert(bn_transformer_prefill_hybrid_batch_allowed(test_cpu_policy()) ==
           bn_transformer_cpu_backend_supports_hybrid_batch_prefill());
    setenv("BN_PREFILL_ALLOW_HYBRID_BATCH", "1", 1);
    assert(bn_transformer_prefill_hybrid_batch_allowed(test_cpu_policy()));
    unsetenv("BN_PREFILL_ALLOW_HYBRID_BATCH");

    unsetenv("BN_PREFILL_FORCE_TOKEN_ATTN");
    assert(!bn_transformer_prefill_requires_token_attention(
        test_cpu_policy()));
    setenv("BN_PREFILL_FORCE_TOKEN_ATTN", "1", 1);
    assert(bn_transformer_prefill_requires_token_attention(
        test_cpu_policy()));
    unsetenv("BN_PREFILL_FORCE_TOKEN_ATTN");

    c.policy_flags = BN_MODEL_ARCH_POLICY_UNIT_ATTENTION_SCALE |
                     BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY |
                     BN_MODEL_ARCH_POLICY_ATTENTION_POST_NORM |
                     BN_MODEL_ARCH_POLICY_FFN_POST_NORM |
                     BN_MODEL_ARCH_POLICY_LAYER_OUTPUT_SCALE |
                     BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT;
    c.per_layer_input_dim = 128;
    assert(bn_transformer_attention_scale(&c, 128) == 1.0f);
    assert(bn_transformer_attention_value_shares_key(&c));
    assert(bn_transformer_attention_uses_post_norm(&c));
    assert(bn_transformer_ffn_uses_post_norm(&c));
    assert(bn_transformer_uses_layer_output_scale(&c));
    assert(bn_transformer_per_layer_embedding_dim(&c) == 128);
    assert(!bn_transformer_prefill_uses_reference_activation(&c));
#if defined(__AVX512F__) && !defined(BN_FORCE_SCALAR)
    assert(cpu_backend == BN_CPU_BACKEND_AVX512);
#endif

    printf("PASSED\n");
}

static void test_batched_attn_padded_alias(void) {
    printf("test_batched_attn_padded_alias... ");
    enum { hs = 128, nt = 32, stride = 2 * hs };
    float input[nt * stride], query[nt * stride], expected[nt * hs];
    float key[nt * hs], value[nt * hs], rope[1] = {0};
    for (int i = 0; i < nt * stride; i++) input[i] = sinf((float)(i * 3 + 1));
    for (int i = 0; i < nt * hs; i++) {
        key[i] = sinf((float)(i * 7 + 2));
        value[i] = cosf((float)(i * 11 + 1));
    }
    BnModel model = {0};
    BnRunState state = {0};
    state.key_cache = key; state.value_cache = value;
    BnThreadPool *pool = bn_tp_create(7);
    assert(pool);
    bn_model_set_thread_pool(&model, pool, 1);
    BnBatchedAttnCtx ctx = {
        .c = &model.config, .s = &state,
        .Q_buf = query, .out = expected,
        .n_tokens = nt, .n_heads = 1, .n_kv_heads = 1,
        .head_size = hs, .kv_dim = hs, .kv_mul = 1, .seq_len = nt,
        .rope_cos = rope, .rope_sin = rope, .attention_scale = 0.125f,
        .wq_rows = hs, .q_row_stride = stride, .wo_cols = hs,
    };
    for (int flash = 0; flash < 2; flash++) {
        model.config.flash_attn = flash;
        memcpy(query, input, sizeof(query));
        ctx.out = expected;
        assert(bn_transformer_batched_attn_dispatch(&model, &ctx) == 0);
        ctx.out = query;
        for (int repeat = 0; repeat < 64; repeat++) {
            memcpy(query, input, sizeof(query));
            assert(bn_transformer_batched_attn_dispatch(&model, &ctx) == 0);
            assert(memcmp(query, expected, sizeof(expected)) == 0);
        }
    }
    bn_tp_free(pool);
    bn_model_set_thread_pool(&model, NULL, 0);
    bn_model_backend_free(&model);
    free(model.runtime);
    printf("PASSED\n");
}

static void test_batched_attn_fp16_kv(void) {
    printf("test_batched_attn_fp16_kv... ");

    BnConfig c;
    BnRunState s;
    memset(&c, 0, sizeof(c));
    memset(&s, 0, sizeof(s));

    c.kv_f16 = 1;
    enum { head_size = 4, kv_dim = 4, seq_len = 4, n_tokens = 2 };
    uint16_t key_cache[seq_len * kv_dim];
    uint16_t value_cache[seq_len * kv_dim];
    float q_buf[n_tokens * head_size];
    float out_scalar[n_tokens * head_size];
#ifdef __ARM_NEON
    float out_neon[n_tokens * head_size];
#endif
    float rope_cos[1] = {0.0f};
    float rope_sin[1] = {0.0f};

    float keys[seq_len * kv_dim] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    float values[seq_len * kv_dim] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    for (int i = 0; i < seq_len * kv_dim; i++) {
        key_cache[i] = bn_fp32_to_fp16(keys[i]);
        value_cache[i] = bn_fp32_to_fp16(values[i]);
    }

    q_buf[0] = 1.0f; q_buf[1] = 0.0f; q_buf[2] = 0.0f; q_buf[3] = 0.0f;
    q_buf[4] = 0.0f; q_buf[5] = 1.0f; q_buf[6] = 0.0f; q_buf[7] = 0.0f;
    memset(out_scalar, 0, sizeof(out_scalar));

    s.key_cache = (float *)key_cache;
    s.value_cache = (float *)value_cache;

    BnBatchedAttnCtx ctx = {
        .c = &c, .s = &s,
        .Q_buf = q_buf, .K_new = NULL, .V_new = NULL, .out = out_scalar,
        .loff = 0, .pos0 = 0, .n_tokens = n_tokens,
        .n_heads = 1, .n_kv_heads = 1,
        .head_size = head_size, .kv_dim = kv_dim, .kv_mul = 1,
        .seq_len = seq_len, .rope_dims = 0,
        .rope_freq = NULL, .rope_cos = rope_cos, .rope_sin = rope_sin,
        .attention_scale = 1.0f / sqrtf((float)head_size),
        .kv_cache_uses_fp16_rows =
            bn_transformer_kv_host_cache_uses_fp16_rows(&c),
        .q_norm = NULL, .k_norm = NULL,
        .q_bias = NULL, .k_bias = NULL, .v_bias = NULL,
        .qk_norm_per_head = 0, .norm_eps = 1e-5f,
        .q_gated = 0, .wq_rows = head_size, .wo_cols = head_size,
    };

    bn_transformer_batched_attn_naive_scalar_range(&ctx, 0, 1);

    for (int d = 0; d < head_size; d++)
        assert(fabsf(out_scalar[d] - values[d]) < 1e-5f);

    float inv_sqrt = 1.0f / sqrtf((float)head_size);
    float w0 = expf(0.0f);
    float w1 = expf(inv_sqrt);
    float denom = w0 + w1;
    for (int d = 0; d < head_size; d++) {
        float expected = (w0 * values[d] + w1 * values[head_size + d]) / denom;
        assert(fabsf(out_scalar[head_size + d] - expected) < 1e-5f);
    }

#ifdef __ARM_NEON
    memset(out_neon, 0, sizeof(out_neon));
    ctx.out = out_neon;
    bn_transformer_batched_attn_naive_neon_range(&ctx, 0, 1);
    for (int i = 0; i < n_tokens * head_size; i++)
        assert(fabsf(out_neon[i] - out_scalar[i]) < 1e-5f);
#endif

    printf("PASSED\n");
}

static void test_gpu_attention_window_emission(void) {
    printf("test_gpu_attention_window_emission... ");
    BnConfig c = {0};
    c.n_layers = 2; c.seq_len = 16; c.sliding_window = 3;
    c.sliding_window_pattern[0] = 1; c.flash_attn = 1;
    BnLayerWeights lw = {0};
    BnGPUBackend gpu = {0}; gpu.kind = BN_GPU_BACKEND_CUDA;
    BnTransformerGPUAttentionResources resources = {0}; resources.gpu = &gpu;
    for (int flash = 0; flash < 2; flash++) for (int layer = 0; layer < 2; layer++) {
        gpu.caps = flash ? BN_GPU_CAP_FLASH_ATTN : 0;
        BnLayerShapePlan plan = {0};
        plan.layer = layer; plan.n_heads = 2; plan.n_kv_heads = 1;
        plan.head_size = 32; plan.kv_mul = 2;
        BnGPUOp ops[16] = {{0}};
        BnTransformerGPUEmitContext ctx;
        bn_transformer_gpu_emit_context_init(&ctx, ops, 16);
        bn_transformer_gpu_emit_context_attention_gqa(
            &ctx, &c, &lw, &resources, &plan, 5, 32, 6, 0, 5 * 32, 32, 0);
        assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
        int found = 0;
        for (int i = 0; i < ctx.n; i++) {
            if (ops[i].op_code == BN_GPU_CODE_FLASH_ATTN ||
                ops[i].op_code == BN_GPU_CODE_GQA_SCORES) {
                assert(ops[i].op_code == (flash ? BN_GPU_CODE_FLASH_ATTN : BN_GPU_CODE_GQA_SCORES));
                assert(ops[i].attention_window == (layer == 0 ? 3 : 0));
                assert(ops[i].p[2] == 6); /* Masking retains the original cache coordinates. */
                found++;
            }
        }
        assert(found == 1);
        bn_transformer_gpu_emit_context_free(&ctx);
    }
    printf("PASSED\n");
}

static void test_attention_sliding_window(void) {
    printf("test_attention_sliding_window... ");
    enum { hs = 32, nh = 2, seq = 8, nt = 5, width = hs * nh };
    BnModel m = {0};
    m.config.n_layers = 2; m.config.n_heads = nh; m.config.n_kv_heads = 1;
    m.config.seq_len = seq; m.config.sliding_window = 3;
    m.config.sliding_window_pattern[0] = 1;
    BnRunState state = {0};
    float keys[seq * hs] = {0}, values[seq * hs];
    uint16_t keys16[seq * hs] = {0}, values16[seq * hs];
    float queries[nt * width] = {0}, output[nt * width + 2], att[nh * seq];
    float rope[1] = {0};
    void (*batch[])(void *, int, int) = {
        bn_transformer_batched_attn_naive_scalar_range,
        bn_transformer_batched_attn_flash_scalar_range,
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
        bn_transformer_batched_attn_naive_avx2_range,
        bn_transformer_batched_attn_flash_avx2_range,
#endif
#ifdef __ARM_NEON
        bn_transformer_batched_attn_naive_neon_range,
        bn_transformer_batched_attn_flash_neon_range,
#endif
    };
    void (*decode[])(void *, int, int) = {
        bn_transformer_gqa_scalar_range, bn_transformer_flash_gqa_scalar_range,
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
        bn_transformer_gqa_avx2_range, bn_transformer_flash_gqa_avx2_range,
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && !defined(BN_FORCE_SCALAR)
        bn_transformer_gqa_avx512_range,
#endif
    };
    for (int fp16 = 0; fp16 < 2; fp16++) {
        state.key_cache = fp16 ? (float *)keys16 : keys;
        state.value_cache = fp16 ? (float *)values16 : values;
        state.q = queries; state.xb = output + 1; state.att = att;
        for (int flash = 0; flash < 2; flash++) {
            m.config.flash_attn = flash;
            for (int layer = 0; layer < 2; layer++) {
                for (int t = 0; t < seq; t++) for (int d = 0; d < hs; d++) {
                    values[t * hs + d] = 3.0f * (t + 1) + (float)(d % 4) / 8.0f;
                    values16[t * hs + d] = bn_fp32_to_fp16(values[t * hs + d]);
                }
                BnBatchedAttnCtx b = {
                    .c = &m.config, .s = &state, .Q_buf = queries,
                    .out = output + 1, .layer = layer, .n_tokens = nt,
                    .n_heads = nh, .n_kv_heads = 1, .head_size = hs,
                    .kv_dim = hs, .kv_mul = nh, .seq_len = seq,
                    .rope_cos = rope, .rope_sin = rope, .attention_scale = 1.0f,
                    .kv_cache_uses_fp16_rows = fp16, .wq_rows = width, .wo_cols = width,
                };
                for (size_t fn = 0; fn <= sizeof(batch) / sizeof(batch[0]); fn++) {
                    output[0] = output[nt * width + 1] = -12345.0f;
                    if (fn == 0) assert(bn_transformer_batched_attn_dispatch(&m, &b) == 0);
                    else {
                        b.attention_window = layer == 0 ? 3 : 0;
                        batch[fn - 1](&b, 0, nh);
                    }
                    for (int t = 0; t < nt; t++) for (int h = 0; h < nh; h++) for (int d = 0; d < hs; d++) {
                        int first = layer == 0 && t >= 3 ? t - 2 : 0;
                        float expected = 1.5f * (first + t + 2) + (float)(d % 4) / 8.0f;
                        assert(fabsf(output[1 + t * width + h * hs + d] - expected) < 2e-4f);
                    }
                    assert(output[0] == -12345.0f && output[nt * width + 1] == -12345.0f);
                }
                /* Position nine wraps the cache; the three-token window must
                 * select absolute positions seven, eight and nine. */
                for (int slot = 0; slot < seq; slot++) for (int d = 0; d < hs; d++) {
                    int pos = slot < 2 ? slot + seq : slot;
                    values[slot * hs + d] = 3.0f * (pos + 1) + (float)(d % 4) / 8.0f;
                    values16[slot * hs + d] = bn_fp32_to_fp16(values[slot * hs + d]);
                }
                BnGQACtx g = {
                    .c = &m.config, .s = &state, .pos = 9, .n_kv = seq,
                    .kv_mul = nh, .head_size = hs, .kv_dim = hs, .seq_len = seq,
                    .attention_scale = 1.0f, .kv_cache_uses_fp16_rows = fp16, .layer = layer,
                };
                bn_transformer_cpu_gqa_dispatch(&m, &g, nh, nh);
                for (size_t fn = 0; fn <= sizeof(decode) / sizeof(decode[0]); fn++) {
                    if (fn) decode[fn - 1](&g, 0, nh);
                    for (int h = 0; h < nh; h++) for (int d = 0; d < hs; d++) {
                        float expected = (layer == 0 ? 27.0f : 19.5f) + (float)(d % 4) / 8.0f;
                        assert(fabsf(output[1 + h * hs + d] - expected) < 2e-4f);
                    }
                }
            }
        }
    }
    printf("PASSED\n");
}

static void test_batched_attn_fp32_wrapped_kv(void) {
    printf("test_batched_attn_fp32_wrapped_kv... ");
#if defined(__AVX2__) && !defined(BN_FORCE_SCALAR)
    enum { head_size = 128, seq_len = 61, pos = 70 };
    BnConfig c = {0};
    BnRunState s = {0};
    float key_cache[seq_len * head_size];
    float value_cache[seq_len * head_size];
    float q_buf[head_size];
    float out_scalar[head_size];
    float out_avx[head_size];
    float rope_cos[1] = {0.0f};
    float rope_sin[1] = {0.0f};
    for (int i = 0; i < seq_len * head_size; i++) {
        key_cache[i] = (float)((i * 17) % 101 - 50) / 127.0f;
        value_cache[i] = (float)((i * 29) % 113 - 56) / 131.0f;
    }
    for (int i = 0; i < head_size; i++)
        q_buf[i] = (float)((i * 11) % 31 - 15) / 97.0f;
    s.key_cache = key_cache;
    s.value_cache = value_cache;
    BnBatchedAttnCtx ctx = {
        .c = &c, .s = &s, .Q_buf = q_buf, .out = out_scalar,
        .loff = 0, .pos0 = pos, .n_tokens = 1,
        .n_heads = 1, .n_kv_heads = 1, .head_size = head_size,
        .kv_dim = head_size, .kv_mul = 1, .seq_len = seq_len,
        .rope_dims = 0, .rope_cos = rope_cos, .rope_sin = rope_sin,
        .attention_scale = 1.0f / sqrtf((float)head_size),
        .kv_cache_uses_fp16_rows = 0, .norm_eps = 1e-5f,
        .wq_rows = head_size, .wo_cols = head_size,
    };
    bn_transformer_batched_attn_naive_scalar_range(&ctx, 0, 1);
    ctx.out = out_avx;
    bn_transformer_batched_attn_naive_avx2_range(&ctx, 0, 1);
    for (int i = 0; i < head_size; i++)
        assert(fabsf(out_avx[i] - out_scalar[i]) < 2e-6f);
#endif
    printf("PASSED\n");
}

static void test_batched_attn_avx512_value_order(void) {
#if defined(__AVX512F__) && !defined(BN_FORCE_SCALAR)
    printf("test_batched_attn_avx512_value_order... ");
    const int dimensions[] = {8, 24, 64, 72, 128, 256};
    const int lengths[] = {1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 257};
    enum { max_head = 256, heads = 2, max_seq = 264, prefix = 13 };
    size_t capacity = prefix + (size_t)max_seq * heads * max_head;
    float *keys = calloc(capacity, sizeof(float));
    float *values = malloc(capacity * sizeof(float));
    assert(keys && values);
    for (size_t i = 0; i < capacity; i++) {
        keys[i] = sinf((float)i * 0.019f);
        values[i] = sinf((float)i * 0.17f) * 11.0f;
    }
    for (size_t shape = 0; shape < sizeof(dimensions) / sizeof(dimensions[0]); shape++) {
        int head = dimensions[shape], kv_dim = heads * head;
        float q[heads * max_head] = {0}, actual[heads * max_head];
        float expected[heads * max_head], rope[1] = {0};
        if (head >= 64)
            for (int h = 0; h < heads; h++) q[h * head] = 1.0f;
        for (int length = 0; length < 11; length++) {
            int n = lengths[length];
            for (int wrapped = 0; wrapped < 2; wrapped++) {
                int seq = wrapped ? n : n + 7;
                int pos = wrapped ? n + 13 : n - 1;
                int start = pos - n + 1;
                BnConfig c = {0};
                BnRunState s = {.key_cache = keys, .value_cache = values};
                BnBatchedAttnCtx ctx = {
                    .c = &c, .s = &s, .Q_buf = q, .out = actual,
                    .loff = prefix, .pos0 = pos, .n_tokens = 1,
                    .n_heads = heads, .n_kv_heads = heads, .head_size = head,
                    .kv_dim = kv_dim, .kv_mul = 1, .seq_len = seq,
                    .rope_dims = 0, .rope_cos = rope, .rope_sin = rope,
                    .attention_scale = 0.375f, .wq_rows = heads * head,
                    .wo_cols = heads * head,
                };
                bn_transformer_batched_attn_naive_avx2_range(&ctx, 0, heads);
                for (int h = 0; h < heads; h++) {
                    float att[512];
                    int padded = (n + 255) & ~255;
                    for (int i = 0; i < n; i++)
                        att[i] = head >= 64
                            ? keys[prefix + ((start + i) % seq) * kv_dim + h * head] * 0.375f
                            : 0.0f;
                    for (int i = n; i < padded; i++) att[i] = -INFINITY;
                    bn_transformer_softmax(att, padded);
                    for (int d = 0; d < head; d++) {
                        // Independent token-strided reference for the native
                        // SGEMM order: one vector accumulator per output.
                        __m512 acc = _mm512_setzero_ps();
                        for (int i = 0; i < n; i += 16) {
                            int offsets[16] = {0};
                            int active = n - i < 16 ? n - i : 16;
                            for (int j = 0; j < active; j++)
                                offsets[j] = ((start + i + j) % seq) * kv_dim;
                            __mmask16 mask = (__mmask16)((1u << active) - 1u);
                            __m512 v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(),
                                mask, _mm512_loadu_si512((const void *)offsets),
                                values + prefix + h * head + d, 4);
                            acc = _mm512_mask3_fmadd_ps(
                                _mm512_maskz_loadu_ps(mask, att + i), v, acc, mask);
                        }
                        expected[h * head + d] = _mm512_reduce_add_ps(acc);
                    }
                }
                assert(memcmp(actual, expected, (size_t)heads * head * sizeof(float)) == 0);
            }
        }
    }
    free(keys);
    free(values);
    printf("PASSED\n");
#endif
}

static void test_gqa_neon_small_kv_matches_scalar(void) {
    printf("test_gqa_neon_small_kv_matches_scalar... ");
#ifdef __ARM_NEON
    enum { n_heads = 2, head_size = 8, seq_len = 16, kv_dim = 8 };
    float q[n_heads * head_size];
    float key_cache[seq_len * kv_dim];
    float value_cache[seq_len * kv_dim];
    float att[n_heads * seq_len];
    float scalar_out[n_heads * head_size];
    float neon_out[n_heads * head_size];
    for (int i = 0; i < n_heads * head_size; i++)
        q[i] = 0.03125f * (float)(((i * 11 + 3) % 29) - 14);
    for (int i = 0; i < seq_len * kv_dim; i++) {
        key_cache[i] = 0.015625f * (float)(((i * 7 + 5) % 31) - 15);
        value_cache[i] = 0.0234375f * (float)(((i * 13 + 1) % 37) - 18);
    }

    BnRunState s = {0};
    s.q = q;
    s.key_cache = key_cache;
    s.value_cache = value_cache;
    s.att = att;
    BnGQACtx ctx = {
        .s = &s,
        .loff = 0,
        .kv_mul = 2,
        .head_size = head_size,
        .kv_dim = kv_dim,
        .seq_len = seq_len,
        .attention_scale = 1.0f / sqrtf((float)head_size),
        .kv_cache_uses_fp16_rows = 0,
    };
    for (int n_kv = 1; n_kv <= seq_len; n_kv++) {
        ctx.pos = n_kv - 1;
        ctx.n_kv = n_kv;
        memset(att, 0, sizeof(att));
        memset(scalar_out, 0, sizeof(scalar_out));
        s.xb = scalar_out;
        bn_transformer_gqa_scalar_range(&ctx, 0, n_heads);
        memset(att, 0, sizeof(att));
        memset(neon_out, 0, sizeof(neon_out));
        s.xb = neon_out;
        bn_transformer_gqa_neon_range(&ctx, 0, n_heads);
        for (int i = 0; i < n_heads * head_size; i++)
            assert(fabsf(neon_out[i] - scalar_out[i]) < 1e-6f);
    }
#endif
    printf("PASSED\n");
}

static void test_attention_sigmoid_gate_reference(void) {
#if defined(__AVX2__)
    printf("test_attention_sigmoid_gate_reference... ");
    const BnCPUBackendOps *ops =
        bn_transformer_cpu_backend_ops(test_cpu_policy());
    float gate[259], actual[259], expected[259];
    const int sizes[] = {256, 0, 1, 7, 8, 9, 15, 16, 17, 257};
    for (size_t c = 0; c < sizeof(sizes) / sizeof(sizes[0]); c++) {
        int n = sizes[c];
        for (int i = 0; i < 259; i++) {
            gate[i] = sinf(i * 0.13f) * 12.0f;
            actual[i] = expected[i] = cosf(i * 0.17f);
        }
        for (int i = 1; i <= n; i++)
            expected[i] *= 1.0f / (1.0f + expf(-gate[i]));
        ops->apply_sigmoid_gate(actual + 1, gate + 1, n);
        assert(memcmp(actual, expected, sizeof(actual)) == 0);
    }
    printf("PASSED\n");
#endif
}

static void test_gqa_avx512_matches_scalar(void) {
    printf("test_gqa_avx512_matches_scalar... ");
#ifdef __AVX512F__
    enum { n_heads = 2, head_size = 128, seq_len = 16, kv_dim = 128 };
    float q[n_heads * head_size];
    float keys[seq_len * kv_dim];
    float values[seq_len * kv_dim];
    uint16_t keys_f16[seq_len * kv_dim];
    uint16_t values_f16[seq_len * kv_dim];
    float att[n_heads * seq_len];
    float scalar_out[n_heads * head_size];
    float avx512_out[n_heads * head_size];
    for (int i = 0; i < n_heads * head_size; i++)
        q[i] = 0.00390625f * (float)(((i * 11 + 3) % 61) - 30);
    for (int i = 0; i < seq_len * kv_dim; i++) {
        keys[i] = 0.0078125f * (float)(((i * 7 + 5) % 47) - 23);
        values[i] = 0.005859375f * (float)(((i * 13 + 1) % 53) - 26);
        keys_f16[i] = bn_fp32_to_fp16(keys[i]);
        values_f16[i] = bn_fp32_to_fp16(values[i]);
    }

    BnRunState s = {0};
    s.q = q;
    s.att = att;
    BnGQACtx ctx = {
        .s = &s,
        .loff = 0,
        .pos = 11,
        .n_kv = 12,
        .kv_mul = 2,
        .head_size = head_size,
        .kv_dim = kv_dim,
        .seq_len = seq_len,
        .attention_scale = 1.0f / sqrtf((float)head_size),
    };

    for (int fp16 = 0; fp16 <= 1; fp16++) {
        s.key_cache = fp16 ? (float *)keys_f16 : keys;
        s.value_cache = fp16 ? (float *)values_f16 : values;
        ctx.kv_cache_uses_fp16_rows = fp16;
        s.xb = scalar_out;
        bn_transformer_gqa_scalar_range(&ctx, 0, n_heads);
        s.xb = avx512_out;
        bn_transformer_gqa_avx512_range(&ctx, 0, n_heads);
        for (int i = 0; i < n_heads * head_size; i++)
            assert(fabsf(avx512_out[i] - scalar_out[i]) < 1e-6f);
    }
#endif
    printf("PASSED\n");
}

static void test_prefill_reference_silu_order(void) {
    printf("test_prefill_reference_silu_order... ");
#ifdef __AVX512F__
    float gate_storage[16] = {
        -3.25f, -1.5f, -0.375f, 0.125f,
        0.75f, 1.625f, 2.875f, 4.5f
    };
    float *gate = gate_storage;
    const float up[8] = {
        0.625f, -1.25f, 2.5f, -3.75f,
        4.125f, -0.875f, 1.75f, -2.25f
    };
    float expected_storage[16];
    __m512 g = _mm512_loadu_ps(gate_storage);
    __m512 denominator = _mm512_add_ps(
        _mm512_set1_ps(1.0f),
        bn_avx512_fast_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), g)));
    __m512 up_v = _mm512_maskz_loadu_ps((__mmask16)0xff, up);
    _mm512_storeu_ps(expected_storage, _mm512_mul_ps(
        _mm512_div_ps(g, denominator), up_v));
    float *expected = expected_storage;
#elif defined(__AVX2__)
    float gate[8] = {
        -3.25f, -1.5f, -0.375f, 0.125f,
        0.75f, 1.625f, 2.875f, 4.5f
    };
    const float up[8] = {
        0.625f, -1.25f, 2.5f, -3.75f,
        4.125f, -0.875f, 1.75f, -2.25f
    };
    float expected[8];
    __m256 g = _mm256_loadu_ps(gate);
    __m256 denominator = _mm256_add_ps(
        _mm256_set1_ps(1.0f),
        bn_avx2_fast_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), g)));
    _mm256_storeu_ps(expected, _mm256_mul_ps(
        _mm256_div_ps(g, denominator), _mm256_loadu_ps(up)));
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
    BnPrefillFFNActCtx ctx = {
        gate, up, 8, BN_MODEL_ACTIVATION_SILU, 1
    };
    bn_transformer_prefill_cpu_ops()->ffn_activation(&ctx, 0, 1);
    assert(memcmp(gate, expected, 8 * sizeof(*gate)) == 0);
#endif
    printf("PASSED\n");
}

/* Execute the emitted layer boundary on synthetic activations. Starting with
 * a stale XB mirrors the fused FFN residual/normalization path. Non-unit and
 * tiny scales expose both stale normalization and x + (scale - 1) * x. */
static void test_gpu_layer_output_scale_normalization(void) {
    const float scales[] = {0.08935546875f, -0.5f, 0.0f, 1.0f, 1.0e-8f};
    const float input[4] = {1.0f, -2.0f, 3.0f, 4.0f};
    float weight[4] = {1.0f, 0.5f, -1.0f, 2.0f};
    const float eps = 0.25f;
    uint32_t u_eps;
    memcpy(&u_eps, &eps, sizeof(u_eps));
    for (size_t c = 0; c < sizeof(scales) / sizeof(scales[0]); c++) {
        BnGPUOp ops[4];
        BnTransformerGPUEmitContext ctx;
        bn_transformer_gpu_emit_context_init(&ctx, ops, 4);
        assert(bn_transformer_gpu_emit_context_layer_output_scale(
                   &ctx, 4, scales[c], weight, u_eps) == 0);
        assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
        float x[4], xb[4];
        memcpy(x, input, sizeof(x));
        for (int j = 0; j < 4; j++)
            xb[j] = input[j] * weight[j] / sqrtf(7.5f + eps);
        for (int i = 0; i < ctx.n; i++) {
            const BnGPUOp *op = &ops[i];
            assert(op->buf_in == BN_GPU_VALUE_X);
            assert(op->p[0] == 4);
            if (op->op_code == BN_GPU_CODE_WEIGHTED_ADD) {
                assert(op->buf_aux == BN_GPU_VALUE_X);
                float alpha;
                memcpy(&alpha, &op->p[1], sizeof(alpha));
                for (int j = 0; j < 4; j++)
                    x[j] = op->p[2] ? alpha * x[j] : x[j] + alpha * x[j];
            } else {
                assert(op->op_code == BN_GPU_CODE_RMSNORM);
                assert(op->buf_out == BN_GPU_VALUE_XB);
                assert(op->W_buf == weight);
                float op_eps, sum = 0.0f;
                memcpy(&op_eps, &op->p[1], sizeof(op_eps));
                for (int j = 0; j < 4; j++) sum += x[j] * x[j];
                for (int j = 0; j < 4; j++)
                    xb[j] = x[j] * weight[j] / sqrtf(sum / 4.0f + op_eps);
            }
        }
        for (int j = 0; j < 4; j++) {
            float expected_x = input[j] * scales[c];
            float expected_xb = expected_x * weight[j] /
                sqrtf(7.5f * scales[c] * scales[c] + eps);
            assert(x[j] == expected_x);
            assert(fabsf(xb[j] - expected_xb) <=
                   1.0e-6f * fabsf(expected_xb));
        }
        bn_transformer_gpu_emit_context_free(&ctx);
    }
    printf("test_gpu_layer_output_scale_normalization... PASSED\n");
}

static void test_gpu_reference_rmsnorm_emission(void) {
    int norm_handle = 0;
    BnGPUOp ops[4] = {{0}};
    BnTransformerGPUEmitContext ctx;
    bn_transformer_gpu_emit_context_init(&ctx, ops, 4);
    ctx.reference_rmsnorm_order = 1;
    assert(bn_transformer_gpu_emit_context_rmsnorm(
               &ctx, &norm_handle, BN_GPU_VALUE_X, BN_GPU_VALUE_XB,
               5120, 0) == 0);
    assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
    assert(ctx.n == 1);
    assert(ops[0].op_code == BN_GPU_CODE_RMSNORM);
    assert(ops[0].flags & BN_GPU_OP_FLAG_RMSNORM_REFERENCE_ORDER);
    bn_transformer_gpu_emit_context_free(&ctx);

    memset(ops, 0, sizeof(ops));
    bn_transformer_gpu_emit_context_init(&ctx, ops, 4);
    ctx.reference_rmsnorm_order = 1;
    assert(bn_transformer_gpu_emit_context_residual_rmsnorm(
               &ctx, BN_GPU_VALUE_X, BN_GPU_VALUE_XB,
               BN_GPU_VALUE_HB, 5120, 0, &norm_handle) == 0);
    assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
    assert(ctx.n == 1);
    assert(ops[0].op_code == BN_GPU_CODE_RESIDUAL_RMSNORM);
    assert(ops[0].p[7] == 1);
    bn_transformer_gpu_emit_context_free(&ctx);
    printf("test_gpu_reference_rmsnorm_emission... PASSED\n");
}

static void test_gpu_recurrent_quant_contract(void) {
    const int types[] = {BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K,
                         BN_GGUF_TENSOR_Q6_K, BN_GGUF_TENSOR_Q8_0,
                         BN_GGUF_TENSOR_F32};
    for (int reference = 0; reference < 2; reference++) {
        for (int block32 = 0; block32 < 2; block32++) {
            for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
                BnGPUBackend gpu = {0};
                gpu.caps = BN_GPU_CAP_REFERENCE_RECURRENT |
                    (block32 ? BN_GPU_CAP_KQUANT_BLOCK32_RECURRENT : 0);
                BnConfig config = {0};
                config.dim = 256;
                config.ssm_group_count = config.ssm_time_step_rank = 1;
                config.ssm_state_size = config.ssm_inner_size = 128;
                config.ssm_conv_kernel = 4;
                config.policy_flags = reference ?
                    BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT : 0;
                BnLayerWeights layer = {0};
                BnQWeight *weights[] = {&layer.ssm.wqkv, &layer.ssm.wz,
                    &layer.ssm.ssm_alpha, &layer.ssm.ssm_beta, &layer.ssm.ssm_out};
                const int rows[] = {384, 128, 1, 1, 256};
                for (int j = 0; j < 5; j++) {
                    weights[j]->type = types[t]; weights[j]->rows = rows[j];
                    weights[j]->cols = j == 4 ? 128 : 256;
                }
                int handles[9] = {0};
                BnTransformerGPUSSMResources res = {0};
                res.gpu = &gpu;
                res.wqkv = &handles[0]; res.wz = &handles[1];
                res.ssm_alpha = &handles[2]; res.ssm_beta = &handles[3];
                res.ssm_out = &handles[4]; res.ssm_conv1d = &handles[5];
                res.ssm_dt_bias = &handles[6]; res.ssm_a_log = &handles[7];
                res.ssm_norm = &handles[8];
                BnLayerShapePlan plan = {0};
                BnGPUOp ops[32] = {{0}};
                BnTransformerGPUEmitContext ctx;
                bn_transformer_gpu_emit_context_init(&ctx, ops, 32);
                ctx.gpu = &gpu;
                bn_transformer_gpu_emit_context_ssm(&ctx, &config, &layer,
                    &plan, &res, 256, 0, 0, 0, 1);
                assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
                int projections = 0;
                for (int i = 0; i < ctx.n; i++) {
                    if (ops[i].op_code != BN_GPU_CODE_MATVEC) continue;
                    assert(projections < 5);
                    assert(ops[i].W_buf == &handles[projections]);
                    uint32_t expected = reference && !block32 && t < 3
                        ? BN_GPU_OP_FLAG_MATVEC_REFERENCE_KQUANT : 0;
                    assert(ops[i].flags == expected);
                    projections++;
                }
                assert(projections == 5);
                bn_transformer_gpu_emit_context_free(&ctx);
            }
        }
    }
    printf("GPU recurrent quant contract PASSED\n");
}

static void test_greedy_token_state_preparation(void) {
    BnModel model = {0};
    BnSession session = {0};
    float embeddings[] = {0, 0, 1, -1, 2, -2, 3, -3};
    float input[2] = {99, 99};
    int history[4] = {-1, -1, -1, -1};
    model.config.dim = 2;
    model.config.vocab_size = 4;
    model.config.seq_len = 4;
    model.weights.emb_type = BN_GGUF_TENSOR_F32;
    model.weights.token_embedding = embeddings;
    session.state.x = input;
    session.state.token_history = history;
    int next = -7;
    /* Even a declined acceleration attempt prepares the current token before
     * falling back. Consecutive calls retain the predecessors needed by PLE. */
    for (int token = 1; token <= 3; token++) {
        assert(bn_transformer_forward_argmax(&model, &session, token, token + 1,
            NULL, 0, 1.0f, &next) == -1);
        assert(next == -7);
        assert(input[0] == (float)token && input[1] == -(float)token);
        assert(history[(token + 1) % 4] == token);
    }
    const int expected[] = {3, -1, 1, 2};
    assert(memcmp(history, expected, sizeof(history)) == 0);
    assert(bn_transformer_forward_argmax(&model, &session, 4, 5,
        NULL, 0, 1.0f, &next) == -1);
    assert(bn_transformer_forward_argmax(&model, &session, 0, -1,
        NULL, 0, 1.0f, &next) == -1);
    assert(bn_transformer_forward_argmax(&model, &session, 0, 5,
        NULL, 0, 1.0f, NULL) == -1);
    assert(memcmp(history, expected, sizeof(history)) == 0);
    assert(input[0] == 3 && input[1] == -3);
    printf("Greedy token state preparation PASSED\n");
}

int main(void) {
    test_greedy_token_state_preparation();
    test_prefill_expert_batch_geometry();
    test_prefill_logical_projection_rows();
    test_prefill_scaled_norm_raw_weight_contract();
    printf("=== Transformer Tests ===\n");
    test_gpu_layer_output_scale_normalization();
    test_gpu_reference_rmsnorm_emission();
    test_gpu_token_staging_preserves_prepared_rope();
    test_rmsnorm();
    test_rmsnorm_reference_contract();
    test_rmsnorm_simd_matches_scalar_order();
    test_sum_products_rounding();
    test_scaled_silu_order();
    test_dilated_conv_and_branch_order();
    test_scaled_residual_rounding();
    test_softmax();
    test_runtime_softmax();
    test_rope();
    test_fp16_embed();
    test_fast_silu();
    test_cpu_execution_helpers();
    test_gpu_capability_routing();
    test_gpu_recurrent_quant_contract();
    test_prefill_microbatch_policy();
    test_gpu_policy_helpers();
    test_logits_policy_helpers();
    test_gpu_op_kind_mapping();
    test_gpu_staged_value_normalization();
    test_model_arch_registry();
    test_layer_shape_planning();
    test_block_planning();
    test_batched_attn_padded_alias();
    test_batched_attn_fp16_kv();
    test_gpu_attention_window_emission();
    test_attention_sliding_window();
    test_batched_attn_fp32_wrapped_kv();
    test_batched_attn_avx512_value_order();
    test_gqa_neon_small_kv_matches_scalar();
    test_gqa_avx512_matches_scalar();
    test_attention_sigmoid_gate_reference();
    test_prefill_reference_silu_order();
    printf("All transformer tests passed!\n");
    return 0;
}
