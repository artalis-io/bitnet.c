#include "transformer_cpu_internal.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_batched_attn_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_logits_internal.h"
#include "transformer_prefill_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "../src/transformer/gpu_internal.h"
#include "../src/gpu_shader.h"
#include "transformer_plan_internal.h"
#include "gpu_policy.h"
#include "model_arch.h"
#include "quant.h"
#include "simd_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

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

static int mock_prefill_qkv_attention_wo(
    void *ctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
    void *q_norm_buf, void *k_norm_buf, const float *X, float *K_out,
    float *V_out, int n_tokens, int dim, int n_heads, int n_kv_heads,
    int head_size, int kv_mul, int kv_dim, int qk_rows, int qk_type,
    int wv_rows, int wv_type, int wo_rows, int wo_cols, int wo_type,
    int qk_norm_per_head, float norm_eps, int pos0, int rope_dims,
    float attention_scale) {
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
    int pos0, int rope_dims, float attention_scale) {
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
                                  float attention_scale) {
    (void)ctx; (void)out; (void)Q; (void)K; (void)V;
    (void)n_tokens; (void)n_heads; (void)n_kv_heads;
    (void)head_size; (void)kv_mul; (void)kv_dim;
    (void)attention_scale;
    return 0;
}

static int mock_prefill_attention_wo(
    void *ctx, float *out, void *wo_buf, const float *Q, const float *K,
    const float *V, int n_tokens, int n_heads, int n_kv_heads,
    int head_size, int kv_mul, int kv_dim, int wo_rows, int wo_cols,
    int wo_type, float attention_scale) {
    (void)ctx; (void)out; (void)wo_buf; (void)Q; (void)K; (void)V;
    (void)n_tokens; (void)n_heads; (void)n_kv_heads; (void)head_size;
    (void)kv_mul; (void)kv_dim; (void)wo_rows; (void)wo_cols;
    (void)wo_type; (void)attention_scale;
    return 0;
}

static int mock_prefill_dense_layer(
    void *ctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
    void *gate_buf, void *up_buf, void *down_buf, void *attn_norm_buf,
    void *ffn_norm_buf, void *q_norm_buf, void *k_norm_buf,
    void *q_bias_buf, void *k_bias_buf, void *v_bias_buf,
    const float *X, float *K_out, float *V_out, int n_tokens, int dim,
    int hidden_dim, int n_heads, int n_kv_heads, int head_size,
    int kv_mul, int kv_dim, int qk_rows, int qk_type, int wv_rows,
    int wv_type, int wo_rows, int wo_cols, int wo_type, int gate_type,
    int up_type, int down_type, int act_type, int qk_norm_per_head,
    float norm_eps, int pos0, int rope_dims, uint32_t kv_cache_off,
    int kv_cache_stride, float attention_scale) {
    (void)ctx; (void)out; (void)qk_buf; (void)wv_buf; (void)wo_buf;
    (void)gate_buf; (void)up_buf; (void)down_buf; (void)attn_norm_buf;
    (void)ffn_norm_buf; (void)q_norm_buf; (void)k_norm_buf;
    (void)q_bias_buf; (void)k_bias_buf; (void)v_bias_buf; (void)X;
    (void)K_out; (void)V_out; (void)n_tokens; (void)dim;
    (void)hidden_dim; (void)n_heads; (void)n_kv_heads; (void)head_size;
    (void)kv_mul; (void)kv_dim; (void)qk_rows; (void)qk_type;
    (void)wv_rows; (void)wv_type; (void)wo_rows; (void)wo_cols;
    (void)wo_type; (void)gate_type; (void)up_type; (void)down_type;
    (void)act_type; (void)qk_norm_per_head; (void)norm_eps;
    (void)pos0; (void)rope_dims; (void)kv_cache_off;
    (void)kv_cache_stride; (void)attention_scale;
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
    const float *X, int n_tokens, int dim, int hidden_dim,
    int n_experts, int k, int gate_type, int up_type, int down_type,
    int act_type) {
    (void)ctx; (void)out; (void)gate_all_buf; (void)up_all_buf;
    (void)down_all_buf; (void)indices; (void)weights; (void)X;
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
    int norm_topk_prob, float expert_weights_scale) {
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
    int ffn_down_type, int act_type, float norm_eps, int *did_ffn) {
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

static void test_rmsnorm_scalar_matches_avx2_order(void) {
    printf("test_rmsnorm_scalar_matches_avx2_order... ");

#ifdef __AVX2__
    enum { N = 2560 };
    float x[N], w[N], out_scalar[N], out_avx2[N];
    for (int i = 0; i < N; i++) {
        x[i] = sinf((float)i * 0.017f) * 3.0f + cosf((float)i * 0.031f);
        w[i] = 0.75f + 0.25f * sinf((float)i * 0.013f);
    }

    bn_transformer_rmsnorm_scalar(out_scalar, x, w, N, 1e-6f);
    bn_transformer_rmsnorm_avx2(out_avx2, x, w, N, 1e-6f);

    for (int i = 0; i < N; i++)
        assert(fabsf(out_scalar[i] - out_avx2[i]) < 1e-6f);
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

    float x[8] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
    float r[8] = {0.5f, 0.5f, -1.0f, -1.0f, 2.0f, 2.0f, -3.0f, -3.0f};
    bn_transformer_cpu_residual_add(x, r, 8);
    float expected_residual[8] = {1.5f, -1.5f, 2.0f, -5.0f, 7.0f, -4.0f, 4.0f, -11.0f};
    for (int i = 0; i < 8; i++)
        assert(fabsf(x[i] - expected_residual[i]) < 1e-6f);

    float rope_buf[8] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
    float rc[2] = {0.0f, 1.0f};
    float rs[2] = {1.0f, 0.0f};
    bn_transformer_cpu_apply_rope_heads(rope_buf, 2, 4, 4, rc, rs);
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
    bn_transformer_cpu_apply_ffn_activation(&s, &ffn, 8, 0);
    for (int i = 0; i < 8; i++) {
        float g = (float[]){-2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f}[i];
        float u = (float[]){1.0f, 0.5f, 2.0f, 3.0f, -1.0f, 1.5f, -0.5f, 2.0f}[i];
        float expected = (g / (1.0f + expf(-g))) * u;
        assert(fabsf(hb[i] - expected) < 2e-3f);
    }

    float relu_hb[8] = {-2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f};
    s.hb = relu_hb;
    ffn.has_gate = 0;
    ffn.activation = 1;
    bn_transformer_cpu_apply_ffn_activation(&s, &ffn, 8, 0);
    float expected_relu2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0625f, 1.0f, 4.0f, 16.0f};
    for (int i = 0; i < 8; i++)
        assert(fabsf(relu_hb[i] - expected_relu2[i]) < 1e-6f);

    float unchanged[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    s.hb = unchanged;
    bn_transformer_cpu_apply_ffn_activation(&s, &ffn, 8, 1);
    for (int i = 0; i < 8; i++)
        assert(fabsf(unchanged[i] - (float)(i + 1)) < 1e-6f);

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

    gpu.caps = BN_GPU_CAP_Q4_MATVEC_SPLIT |
               BN_GPU_CAP_Q5_MATVEC_SPLIT |
               BN_GPU_CAP_Q4K_MATVEC_SPLIT |
               BN_GPU_CAP_Q8_MATVEC_SPLIT |
               BN_GPU_CAP_Q5K_MATVEC_SPLIT |
               BN_GPU_CAP_Q4_FUSED_GATEUP_SILU |
               BN_GPU_CAP_Q5_FUSED_GATEUP_SILU |
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
               BN_GGUF_TENSOR_Q4_K, 1) == BN_GPU_OP_FLAG_MATVEC_Q8K);
    assert(bn_transformer_gpu_matvec_kquant_dot_flags(
               BN_GGUF_TENSOR_Q4_K, 0) == 0);
    assert(bn_transformer_gpu_matvec_kquant_dot_flags(
               BN_GGUF_TENSOR_Q8_0, 1) == 0);
    assert(bn_transformer_gpu_moe_route_raw_compare_matvec_flags(
               BN_GGUF_TENSOR_Q4_K) == BN_GPU_OP_FLAG_MATVEC_Q8K);
    assert(bn_transformer_gpu_moe_route_raw_compare_matvec_flags(
               BN_GGUF_TENSOR_F32) == 0);
    assert(bn_transformer_gpu_matvec_exact_kquant_flags(
               BN_GGUF_TENSOR_Q6_K, 1) == BN_GPU_OP_FLAG_MATVEC_EXACT_Q6K);
    assert(bn_transformer_gpu_matvec_exact_kquant_flags(
               BN_GGUF_TENSOR_Q6_K, 0) == 0);
    assert(bn_transformer_gpu_matvec_exact_kquant_flags(
               BN_GGUF_TENSOR_Q4_K, 1) == 0);
    assert(bn_transformer_gpu_float_buffer_type() == BN_GGUF_TENSOR_F32);
    assert(bn_transformer_gpu_exact_silu_flags(
               BN_GGUF_TENSOR_Q8_0, 1) == BN_GPU_OP_FLAG_EXACT_SILU);
    assert(bn_transformer_gpu_exact_silu_flags(
               BN_GGUF_TENSOR_Q8_0, 0) == 0);
    assert(bn_transformer_gpu_exact_silu_flags(
               BN_GGUF_TENSOR_Q4_0, 1) == 0);
    assert(bn_transformer_gpu_exact_silu_active_flags(1) ==
           BN_GPU_OP_FLAG_EXACT_SILU);
    assert(bn_transformer_gpu_exact_silu_active_flags(0) == 0);
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

    setenv("BN_GPU_DISABLE_FUSED_GATEUP", "1", 1);
    assert(!bn_transformer_gpu_fused_gateup_silu_policy_allows(
        &gpu, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_transformer_gpu_can_fused_gateup_silu(
        &gpu, BN_GGUF_TENSOR_Q4_0, 0));
    unsetenv("BN_GPU_DISABLE_FUSED_GATEUP");
    gpu.kind = BN_GPU_BACKEND_CUDA;
    gpu.caps |= BN_GPU_CAP_Q5K_FUSED_GATEUP_SILU;
    unsetenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP");
    assert(!bn_transformer_gpu_fused_gateup_silu_policy_allows(
        &gpu, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_transformer_gpu_can_fused_gateup_silu(
        &gpu, BN_GGUF_TENSOR_Q5_K, 0));
    setenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP", "1", 1);
    assert(bn_transformer_gpu_fused_gateup_silu_policy_allows(
        &gpu, BN_GGUF_TENSOR_Q5_K));
    assert(bn_transformer_gpu_can_fused_gateup_silu(
        &gpu, BN_GGUF_TENSOR_Q5_K, 0));
    unsetenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP");

    unsetenv("BN_GPU_Q4_Q8_DISABLE_GATEUP");
    assert(!bn_transformer_gpu_small_dense_exact_native_fused_gateup_enabled(0));
    assert(bn_transformer_gpu_small_dense_exact_native_fused_gateup_enabled(1));
    setenv("BN_GPU_Q4_Q8_DISABLE_GATEUP", "1", 1);
    assert(!bn_transformer_gpu_small_dense_exact_native_fused_gateup_enabled(1));
    unsetenv("BN_GPU_Q4_Q8_DISABLE_GATEUP");

    unsetenv("BN_GPU_DISABLE_GATEUP_SPLIT");
    assert(bn_transformer_gpu_gateup_split_enabled());
    setenv("BN_GPU_DISABLE_GATEUP_SPLIT", "1", 1);
    assert(!bn_transformer_gpu_gateup_split_enabled());
    unsetenv("BN_GPU_DISABLE_GATEUP_SPLIT");

    unsetenv("BN_GPU_Q4_Q8_DISABLE_FFN_DOWN");
    assert(!bn_transformer_gpu_small_dense_exact_native_down_enabled(0));
    assert(bn_transformer_gpu_small_dense_exact_native_down_enabled(1));
    setenv("BN_GPU_Q4_Q8_DISABLE_FFN_DOWN", "1", 1);
    assert(!bn_transformer_gpu_small_dense_exact_native_down_enabled(1));
    unsetenv("BN_GPU_Q4_Q8_DISABLE_FFN_DOWN");

    unsetenv("BN_GPU_DISABLE_QKV_SPLIT");
    assert(bn_transformer_gpu_qkv_split_enabled(0));
    assert(!bn_transformer_gpu_qkv_split_enabled(1));
    assert(bn_transformer_gpu_qk_split_enabled());
    setenv("BN_GPU_DISABLE_QKV_SPLIT", "1", 1);
    assert(!bn_transformer_gpu_qkv_split_enabled(0));
    assert(!bn_transformer_gpu_qk_split_enabled());
    unsetenv("BN_GPU_DISABLE_QKV_SPLIT");

    unsetenv("BN_GPU_DEBUG_QKV_SPLIT");
    assert(!bn_transformer_gpu_qkv_split_debug_enabled());
    setenv("BN_GPU_DEBUG_QKV_SPLIT", "1", 1);
    assert(bn_transformer_gpu_qkv_split_debug_enabled());
    unsetenv("BN_GPU_DEBUG_QKV_SPLIT");

    unsetenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT");
    assert(bn_transformer_gpu_ssm_qkvz_split_enabled());
    setenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT", "1", 1);
    assert(!bn_transformer_gpu_ssm_qkvz_split_enabled());
    unsetenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT");

    unsetenv("BN_GPU_DISABLE_SSM_AB_STACK");
    assert(bn_transformer_gpu_ssm_ab_stack_enabled());
    setenv("BN_GPU_DISABLE_SSM_AB_STACK", "1", 1);
    assert(!bn_transformer_gpu_ssm_ab_stack_enabled());
    unsetenv("BN_GPU_DISABLE_SSM_AB_STACK");

    unsetenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM");
    assert(!bn_transformer_gpu_split_residual_rmsnorm_enabled());
    setenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM", "1", 1);
    assert(bn_transformer_gpu_split_residual_rmsnorm_enabled());
    unsetenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM");

    unsetenv("BN_GPU_DEBUG_FALLBACK");
    assert(!bn_transformer_gpu_debug_fallback_enabled());
    setenv("BN_GPU_DEBUG_FALLBACK", "1", 1);
    assert(bn_transformer_gpu_debug_fallback_enabled());
    unsetenv("BN_GPU_DEBUG_FALLBACK");

    unsetenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT");
    assert(!bn_transformer_gpu_shared_kquant_dot_enabled(0));
    assert(bn_transformer_gpu_shared_kquant_dot_enabled(1));
    setenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT", "1", 1);
    assert(!bn_transformer_gpu_shared_kquant_dot_enabled(1));
    unsetenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT");

    unsetenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE");
    assert(!bn_transformer_gpu_shared_expert_gate_enabled(0));
    assert(bn_transformer_gpu_shared_expert_gate_enabled(1));
    setenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE", "1", 1);
    assert(!bn_transformer_gpu_shared_expert_gate_enabled(1));
    unsetenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE");

    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_can_flash_attn(&gpu));
    assert(!bn_transformer_gpu_can_layerwise_rope(&gpu));
    gpu.caps |= BN_GPU_CAP_LAYERWISE_ROPE;
    assert(bn_transformer_gpu_can_layerwise_rope(&gpu));

    printf("PASSED\n");
}

static void test_gpu_policy_helpers(void) {
    printf("test_gpu_policy_helpers... ");

    BnConfig c;
    memset(&c, 0, sizeof(c));
    c.n_layers = 4;
    assert(bn_transformer_gpu_graph_op_capacity(&c) >
           80 * c.n_layers);
    assert(bn_transformer_gpu_uses_small_dense_shape(&c));
    assert(!bn_transformer_gpu_uses_large_dense_shape(&c));
    assert(!bn_transformer_gpu_uses_large_graph_fallback_shape(&c));
    assert(!bn_transformer_gpu_uses_per_layer_embedding(&c));
    assert(bn_transformer_gpu_uses_dense_attention_only(&c));
    assert(!bn_transformer_gpu_uses_hybrid_ssm(&c));
    assert(!bn_transformer_gpu_uses_moe(&c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT;
    assert(bn_transformer_gpu_uses_per_layer_embedding(&c));
    c.policy_flags = 0;
    c.dim = 4096;
    c.full_attn_interval = 4;
    assert(!bn_transformer_gpu_uses_small_dense_shape(&c));
    assert(bn_transformer_gpu_uses_large_dense_shape(&c));
    assert(bn_transformer_gpu_uses_large_graph_fallback_shape(&c));
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
               strcmp(reject_reason, "output norm not uploaded") == 0);
        memset(&gpu, 0, sizeof(gpu));
    }

    c.policy_flags = BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE |
                     BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE |
                     BN_MODEL_ARCH_POLICY_PREFILL_EXACT_ACTIVATION;
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
    assert(!bn_transformer_gpu_prefill_ssm_layer_disabled());
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    assert(bn_transformer_gpu_prefill_ssm_layer_disabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");

    unsetenv("BN_GPU_PROFILE");
    assert(bn_transformer_gpu_profile_level() == 0);
    setenv("BN_GPU_PROFILE", "3", 1);
    assert(bn_transformer_gpu_profile_level() == 3);
    unsetenv("BN_GPU_PROFILE");

    unsetenv("BN_GPU_MOE_ROUTE_PROFILE");
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY");
    assert(!bn_transformer_gpu_moe_route_profile_enabled());
    assert(bn_transformer_gpu_moe_route_profile_every() == 28);
    setenv("BN_GPU_MOE_ROUTE_PROFILE", "1", 1);
    setenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY", "5", 1);
    assert(bn_transformer_gpu_moe_route_profile_enabled());
    assert(bn_transformer_gpu_moe_route_profile_every() == 5);
    setenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY", "0", 1);
    assert(bn_transformer_gpu_moe_route_profile_every() == 28);
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE");
    unsetenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY");

    assert(!bn_gpu_policy_auto_caps_sequence(0, 0, 0, 0, 8193, 4096));
    assert(!bn_gpu_policy_auto_caps_sequence(1, 0, 0, 0, 4096, 4096));
    assert(bn_gpu_policy_auto_caps_sequence(1, 0, 0, 0, 4097, 4096));
    assert(bn_gpu_policy_auto_caps_sequence(0, 1, 0, 0, 4097, 4096));
    assert(!bn_gpu_policy_auto_caps_sequence(0, 0, 1, 0, 4097, 4096));
    assert(bn_gpu_policy_auto_caps_sequence(0, 0, 1, 1, 4097, 4096));

    unsetenv("BN_CUDA_DISABLE_MOE_FFN");
    assert(!bn_transformer_gpu_moe_ffn_disabled());
    setenv("BN_CUDA_DISABLE_MOE_FFN", "1", 1);
    assert(bn_transformer_gpu_moe_ffn_disabled());
    unsetenv("BN_CUDA_DISABLE_MOE_FFN");

    unsetenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL");
    assert(!bn_transformer_gpu_moe_cpu_actual_override_enabled(0));
    assert(bn_transformer_gpu_moe_cpu_actual_override_enabled(1));
    setenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL", "1", 1);
    assert(bn_transformer_gpu_moe_cpu_actual_override_enabled(0));
    unsetenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL");

    unsetenv("BN_GPU_COMPARE_MOE_LAYER");
    unsetenv("BN_GPU_COMPARE_MOE_POS");
    assert(!bn_transformer_gpu_moe_compare_layer_selected(3, 7));
    setenv("BN_GPU_COMPARE_MOE_LAYER", "3", 1);
    assert(bn_transformer_gpu_moe_compare_layer_selected(3, 7));
    assert(!bn_transformer_gpu_moe_compare_layer_selected(4, 7));
    setenv("BN_GPU_COMPARE_MOE_POS", "7", 1);
    assert(bn_transformer_gpu_moe_compare_layer_selected(3, 7));
    assert(!bn_transformer_gpu_moe_compare_layer_selected(3, 8));
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
    assert(!bn_transformer_gpu_moe_compare_input_norm_enabled());
    assert(!bn_transformer_gpu_moe_compare_actual_enabled());
    assert(!bn_transformer_gpu_moe_compare_route_enabled());
    assert(!bn_transformer_gpu_moe_compare_raw_enabled());
    assert(!bn_transformer_gpu_moe_compare_mid_enabled());
    assert(!bn_transformer_gpu_moe_compare_parts_enabled());
    assert(!bn_transformer_gpu_moe_compare_shared_mid_enabled());
    assert(!bn_transformer_gpu_moe_compare_shared_down_enabled());
    assert(!bn_transformer_gpu_moe_compare_norm_enabled());
    setenv("BN_GPU_COMPARE_MOE_INPUT_NORM", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_ACTUAL", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_ROUTE", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_RAW", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_MID", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_PARTS", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_SHARED_MID", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_SHARED_DOWN", "1", 1);
    setenv("BN_GPU_COMPARE_MOE_NORM", "1", 1);
    assert(bn_transformer_gpu_moe_compare_input_norm_enabled());
    assert(bn_transformer_gpu_moe_compare_actual_enabled());
    assert(bn_transformer_gpu_moe_compare_route_enabled());
    assert(bn_transformer_gpu_moe_compare_raw_enabled());
    assert(bn_transformer_gpu_moe_compare_mid_enabled());
    assert(bn_transformer_gpu_moe_compare_parts_enabled());
    assert(bn_transformer_gpu_moe_compare_shared_mid_enabled());
    assert(bn_transformer_gpu_moe_compare_shared_down_enabled());
    assert(bn_transformer_gpu_moe_compare_norm_enabled());
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
        bn_transformer_gpu_moe_debug_policy(0, 0);
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
    moe_debug = bn_transformer_gpu_moe_debug_policy(1, 0);
    assert(moe_debug.override_cpu_actual);
    assert(!moe_debug.compare_layer);
    moe_debug = bn_transformer_gpu_moe_debug_policy(0, 1);
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
    moe_debug = bn_transformer_gpu_moe_decode_debug_policy(&c, NULL, 2, 0);
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
    moe_debug = bn_transformer_gpu_moe_debug_policy(0, 0);
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
    moe_debug = bn_transformer_gpu_moe_debug_policy(0, 1);
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
    assert(!bn_transformer_gpu_cpu_logits_enabled(0));
    assert(bn_transformer_gpu_cpu_logits_enabled(1));
    assert(!bn_transformer_gpu_debug_argmax_compare_enabled());
    assert(!bn_transformer_gpu_compare_logits_enabled());
    setenv("BN_GPU_CPU_LOGITS", "1", 1);
    setenv("BN_GPU_DEBUG_ARGMAX_COMPARE", "1", 1);
    setenv("BN_GPU_COMPARE_LOGITS", "1", 1);
    assert(bn_transformer_gpu_cpu_logits_enabled(0));
    assert(bn_transformer_gpu_debug_argmax_compare_enabled());
    assert(bn_transformer_gpu_compare_logits_enabled());
    unsetenv("BN_GPU_CPU_LOGITS");
    unsetenv("BN_GPU_DEBUG_ARGMAX_COMPARE");
    unsetenv("BN_GPU_COMPARE_LOGITS");

    unsetenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK");
    unsetenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK");
    BnLayerWeights shared_layer;
    memset(&shared_layer, 0, sizeof(shared_layer));
    c.has_shared_expert = 1;
    BnTransformerGPUMoESharedCPUFallbackPolicy shared_fallback =
        bn_transformer_gpu_moe_shared_cpu_fallback_policy(
            &c, &shared_layer);
    assert(!shared_fallback.enabled);
    shared_layer.shared.shared_gate.data = (void *)1;
    shared_fallback = bn_transformer_gpu_moe_shared_cpu_fallback_policy(
        &c, &shared_layer);
    assert(!shared_fallback.enabled);
    assert(!bn_transformer_gpu_moe_shared_cpu_fallback_enabled(0));
    assert(!bn_transformer_gpu_moe_shared_cpu_fallback_enabled(1));
    setenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK", "1", 1);
    shared_fallback = bn_transformer_gpu_moe_shared_cpu_fallback_policy(
        &c, &shared_layer);
    assert(shared_fallback.enabled);
    c.has_shared_expert = 0;
    shared_fallback = bn_transformer_gpu_moe_shared_cpu_fallback_policy(
        &c, &shared_layer);
    assert(!shared_fallback.enabled);
    c.has_shared_expert = 1;
    assert(!bn_transformer_gpu_moe_shared_cpu_fallback_enabled(0));
    assert(bn_transformer_gpu_moe_shared_cpu_fallback_enabled(1));
    setenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK", "1", 1);
    shared_fallback = bn_transformer_gpu_moe_shared_cpu_fallback_policy(
        &c, &shared_layer);
    assert(!shared_fallback.enabled);
    assert(!bn_transformer_gpu_moe_shared_cpu_fallback_enabled(1));
    unsetenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK");
    unsetenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK");
    c.has_shared_expert = 0;

    setenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE", "1", 1);
    assert(!bn_transformer_gpu_moe_decode_cacheable(&c, NULL, NULL));
    unsetenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE");

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
    unsetenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE");
    unsetenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE");
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
    setenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE", "1", 1);
    assert(bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    matvec_fallback =
        bn_transformer_gpu_matvec_fallback_policy(&model, &gpu);
    assert(matvec_fallback.keep_backend_matvec);
    assert(!matvec_fallback.disable_backend_matvec);
    unsetenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE");
    model.weights.emb_type = BN_GGUF_TENSOR_Q8_0;
    layer.attn.wq.data = (void *)1;
    layer.attn.wq.type = BN_GGUF_TENSOR_Q4_K;
    assert(!bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    layer.attn.wq.type = BN_GGUF_TENSOR_Q8_0;
    assert(bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
    model.config.n_experts = 1;
    assert(!bn_transformer_gpu_backend_matvec_fallback_kept(&model, &gpu));
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
    c.dim = 2561;
    assert(!bn_transformer_gpu_dense_batch_prefill_shape_allowed(&gpu, &c));
    assert(!bn_transformer_gpu_batch_prefill_enabled(&gpu, &c));
    assert(!bn_transformer_gpu_dense_batch_prefill_shape_allowed(&gpu, NULL));
    c.dim = 9000;
    setenv("BN_GPU_PREFILL_MATMUL", "1", 1);
    assert(bn_transformer_gpu_batch_prefill_enabled(&gpu, &c));
    unsetenv("BN_GPU_PREFILL_MATMUL");

    assert(bn_transformer_gpu_moe_gateup_task_flags(&c) == 0);
    c.policy_flags |= BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP;
    assert(bn_transformer_gpu_moe_gateup_task_flags(&c) ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);
    c.policy_flags = 0;
    assert(bn_transformer_prefill_float_kquant_fallback_task_flags(0) == 0);
    assert(bn_transformer_prefill_float_kquant_fallback_task_flags(1) ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);

    BnMoEExpertMap expert_map;
    memset(&expert_map, 0, sizeof(expert_map));
    expert_map.gate_type = BN_GGUF_TENSOR_Q4_K;
    expert_map.up_type = BN_GGUF_TENSOR_Q4_K;
    expert_map.gate_rows = 32;
    expert_map.up_rows = 32;
    expert_map.gate_cols = 64;
    expert_map.up_cols = 64;
    gpu.caps = BN_GPU_CAP_Q4K_MATVEC_SPLIT;
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
    assert(!bn_transformer_gpu_moe_gateup_split_supported(
        &gpu, &expert_map, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    expert_map.up_rows = 32;
    expert_map.up_cols = 32;
    assert(!bn_transformer_gpu_moe_gateup_split_supported(
        &gpu, &expert_map, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    expert_map.up_cols = 64;
    gpu.caps = 0;
    assert(!bn_transformer_gpu_moe_gateup_split_supported(
        &gpu, &expert_map, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_moe_gateup_split_enabled(&gpu, 1));
    setenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT", "1", 1);
    assert(!bn_transformer_gpu_moe_gateup_split_enabled(&gpu, 1));
    unsetenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT");
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
    gpu.caps = BN_GPU_CAP_Q4K_MATVEC_SPLIT;
    assert(bn_transformer_gpu_dense_gateup_exact_split_supported(
        &gpu, &gate_w, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_exact_split_supported(
        NULL, &gate_w, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_exact_split_supported(
        &gpu, NULL, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_exact_split_supported(
        &gpu, &gate_w, NULL, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_exact_split_supported(
        &gpu, &gate_w, &up_w, 1, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    assert(!bn_transformer_gpu_dense_gateup_exact_split_supported(
        &gpu, &gate_w, &up_w, 0, BN_GPU_CODE_Q5K_MATVEC_SPLIT));
    up_w.rows = 16;
    assert(!bn_transformer_gpu_dense_gateup_exact_split_supported(
        &gpu, &gate_w, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    up_w.rows = 32;
    up_w.cols = 32;
    assert(!bn_transformer_gpu_dense_gateup_exact_split_supported(
        &gpu, &gate_w, &up_w, 0, BN_GPU_CODE_Q4K_MATVEC_SPLIT));
    up_w.cols = 64;
    gpu.caps = 0;
    assert(!bn_transformer_gpu_dense_gateup_exact_split_supported(
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
    gpu.caps = BN_GPU_CAP_Q4_MATVEC_SPLIT |
               BN_GPU_CAP_Q5K_MATVEC_SPLIT |
               BN_GPU_CAP_Q8_MATVEC_SPLIT;
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
    gpu.caps = BN_GPU_CAP_Q4_MATVEC_SPLIT;
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

    unsetenv("BN_GPU_Q8_REFINE_TOP");
    assert(bn_transformer_gpu_native_quant_logits_refine_top(1) == 16);
    setenv("BN_GPU_Q8_REFINE_TOP", "5", 1);
    assert(bn_transformer_gpu_native_quant_logits_refine_top(1) == 5);
    unsetenv("BN_GPU_Q8_REFINE_TOP");

    W.type = BN_GGUF_TENSOR_Q6_K;
    W.rows = 1024;
    W.cols = 2048;
    logits.type = BN_GGUF_TENSOR_Q6_K;
    logits.rows = W.rows;
    logits.cols = W.cols;
    logits.cpu_weight = &W;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE");
    assert(!bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 0));
    assert(bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 1));
    setenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE", "1", 1);
    assert(bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 0));
    unsetenv("BN_GPU_ENABLE_Q6_LOGITS_REFINE");
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 0));
    setenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE", "1", 1);
    assert(!bn_transformer_gpu_kquant_logits_refine_enabled(&gpu, 0));
    unsetenv("BN_GPU_DISABLE_Q6_LOGITS_REFINE");
    assert(bn_transformer_gpu_kquant_logits_refine_captures_xb(
        &logits, 1, 1));
    assert(!bn_transformer_gpu_kquant_logits_refine_captures_xb(
        &logits, 1, 0));
    logits.cpu_weight = NULL;
    assert(!bn_transformer_gpu_kquant_logits_refine_captures_xb(
        &logits, 1, 1));
    logits.cpu_weight = &W;

    unsetenv("BN_GPU_Q6_Q8K_REFINE_TOP");
    assert(bn_transformer_gpu_kquant_logits_refine_top(1) == 64);
    assert(bn_transformer_gpu_kquant_logits_refine_top(0) == 8);
    setenv("BN_GPU_Q6_Q8K_REFINE_TOP", "11", 1);
    assert(bn_transformer_gpu_kquant_logits_refine_top(1) == 11);
    unsetenv("BN_GPU_Q6_Q8K_REFINE_TOP");
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
    unsetenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE");
    unsetenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE");
    assert(!bn_transformer_gpu_native_quant_logits_refine_active(&gpu, 0));
    assert(bn_transformer_gpu_native_quant_logits_refine_active(&gpu, 1));
    setenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE", "1", 1);
    assert(bn_transformer_gpu_native_quant_logits_refine_active(&gpu, 0));
    unsetenv("BN_GPU_ENABLE_Q8_LOGITS_REFINE");
    gpu.kind = BN_GPU_BACKEND_WEBGPU;
    assert(bn_transformer_gpu_native_quant_logits_refine_active(&gpu, 0));
    setenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE", "1", 1);
    assert(!bn_transformer_gpu_native_quant_logits_refine_active(&gpu, 0));
    unsetenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE");
    assert(bn_transformer_gpu_native_quant_logits_refine_captures_xb(&logits, 1));
    logits.cpu_weight = NULL;
    assert(!bn_transformer_gpu_native_quant_logits_refine_captures_xb(&logits, 1));
    logits.cpu_weight = &W;

    memset(&c, 0, sizeof(c));
    c.dim = 2048;
    c.policy_flags = BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE |
                     BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE");
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE", "1", 1);
    BnTransformerGPULogitsRefinePolicy refine_policy =
        bn_transformer_gpu_logits_refine_policy(&gpu, &c, NULL, &logits, 1);
    assert(!refine_policy.kquant_default);
    assert(!refine_policy.kquant_enabled);
    assert(!refine_policy.kquant_captures_xb);
    assert(refine_policy.kquant_refine_top == 8);
    assert(refine_policy.native_quant_default);
    assert(refine_policy.native_quant_enabled);
    assert(refine_policy.native_quant_captures_xb);
    assert(refine_policy.native_quant_refine_top == 16);
    setenv("BN_GPU_Q6_Q8K_REFINE_TOP", "13", 1);
    setenv("BN_GPU_Q8_REFINE_TOP", "7", 1);
    refine_policy =
        bn_transformer_gpu_logits_refine_policy(&gpu, &c, NULL, &logits, 1);
    assert(refine_policy.kquant_refine_top == 13);
    assert(refine_policy.native_quant_refine_top == 7);
    unsetenv("BN_GPU_Q6_Q8K_REFINE_TOP");
    unsetenv("BN_GPU_Q8_REFINE_TOP");
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE");

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
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q6_LOGITS_REFINE");
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN", "1", 1);
    refine_policy = bn_transformer_gpu_logits_refine_policy(
        &gpu, &c, &refine_weights, &logits, 0);
    assert(refine_policy.kquant_default);
    assert(refine_policy.kquant_enabled);
    assert(refine_policy.kquant_captures_xb);
    assert(!refine_policy.native_quant_default);
    assert(!refine_policy.native_quant_captures_xb);
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");

    unsetenv("BN_CUDA_DISABLE_SSM_FFN_FUSE");
    assert(bn_transformer_gpu_prefill_ssm_ffn_fuse_allowed());
    setenv("BN_CUDA_DISABLE_SSM_FFN_FUSE", "1", 1);
    assert(!bn_transformer_gpu_prefill_ssm_ffn_fuse_allowed());
    unsetenv("BN_CUDA_DISABLE_SSM_FFN_FUSE");

    unsetenv("BN_CUDA_ENABLE_MOE_PREFILL");
    assert(!bn_transformer_prefill_moe_enabled());
    setenv("BN_CUDA_ENABLE_MOE_PREFILL", "1", 1);
    assert(bn_transformer_prefill_moe_enabled());
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
    assert(bn_transformer_gpu_moe_prefill_min_tokens() == 1);
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "0", 1);
    assert(bn_transformer_gpu_moe_prefill_min_tokens() == 1);
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "9", 1);
    assert(bn_transformer_gpu_moe_prefill_min_tokens() == 9);
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");

    unsetenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL");
    assert(bn_transformer_gpu_moe_cache_prefill_enabled());
    setenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL", "1", 1);
    assert(!bn_transformer_gpu_moe_cache_prefill_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL");

    memset(&c, 0, sizeof(c));
    c.n_experts = 2;
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ROUTE_RESIDENT");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ROUTE_RESIDENT");
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &c, 0, 0, 1));
    c.n_experts = 3;
    assert(bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &c, 0, 0, 1));
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &c, 0, 1, 1));
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &c, 0, 0, 0));
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        NULL, 0, 0, 1));
    setenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT", "1", 1);
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &c, 0, 0, 1));
    unsetenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT");
    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    c.dim = 2048;
    assert(bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &c, 1, 0, 0));
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ROUTE_RESIDENT", "1", 1);
    assert(!bn_transformer_gpu_moe_cpu_route_resident_ffn_enabled(
        &c, 1, 0, 0));
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ROUTE_RESIDENT");

    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
    assert(bn_transformer_gpu_moe_prefill_shared_fuse_enabled());
    setenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE", "1", 1);
    assert(!bn_transformer_gpu_moe_prefill_shared_fuse_enabled());
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");

    unsetenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH");
    assert(!bn_transformer_gpu_moe_route_batch_debug_enabled());
    setenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH", "1", 1);
    assert(bn_transformer_gpu_moe_route_batch_debug_enabled());
    unsetenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH");

    unsetenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE");
    assert(!bn_transformer_gpu_moe_lazy_aux_cache_enabled());
    setenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE", "1", 1);
    assert(bn_transformer_gpu_moe_lazy_aux_cache_enabled());
    unsetenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE");

    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL");
    assert(!bn_transformer_gpu_large_hybrid_prefill_disabled());
    assert(!bn_transformer_prefill_large_hybrid_disabled());
    setenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL", "1", 1);
    assert(bn_transformer_gpu_large_hybrid_prefill_disabled());
    assert(bn_transformer_prefill_large_hybrid_disabled());
    unsetenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL");

    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");
    assert(bn_transformer_gpu_prefill_dense_chain_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN", "1", 1);
    assert(!bn_transformer_gpu_prefill_dense_chain_enabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");

    memset(&c, 0, sizeof(c));
    c.dim = 2048;
    c.full_attn_interval = 1;
    c.ssm_inner_size = 128;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN");
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN");
    assert(bn_transformer_prefill_hybrid_chain_applicable(&gpu, &c));
    assert(!bn_transformer_gpu_large_hybrid_prefill_decode_fallback_default(
        &gpu, &c));
    assert(bn_transformer_prefill_hybrid_chain_enabled(&gpu, &c));
    assert(bn_transformer_gpu_prefill_hybrid_chain_enabled(&gpu, &c));
    setenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN", "1", 1);
    assert(!bn_transformer_prefill_hybrid_chain_enabled(&gpu, &c));
    assert(!bn_transformer_gpu_prefill_hybrid_chain_enabled(&gpu, &c));
    unsetenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN");
    c.ssm_inner_size = 0;
    assert(!bn_transformer_prefill_hybrid_chain_applicable(&gpu, &c));
    c.ssm_inner_size = 128;
    c.dim = 4096;
    assert(bn_transformer_gpu_large_hybrid_prefill_decode_fallback_default(
        &gpu, &c));
    assert(bn_transformer_gpu_large_hybrid_prefill_chain_disabled_default(
        &gpu, &c));
    assert(!bn_transformer_prefill_hybrid_chain_enabled(&gpu, &c));
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN", "1", 1);
    assert(!bn_transformer_gpu_large_hybrid_prefill_chain_disabled_default(
        &gpu, &c));
    assert(bn_transformer_prefill_hybrid_chain_enabled(&gpu, &c));
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN");

    unsetenv("BN_CUDA_DISABLE_PREFILL_ATTN");
    assert(bn_transformer_prefill_attention_enabled());
    assert(bn_transformer_gpu_prefill_attention_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_ATTN", "1", 1);
    assert(!bn_transformer_prefill_attention_enabled());
    assert(!bn_transformer_gpu_prefill_attention_enabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_ATTN");
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    assert(bn_transformer_gpu_prefill_attention_min_tokens() == 16);
    assert(bn_transformer_prefill_attention_min_tokens() == 16);
    setenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "11", 1);
    assert(bn_transformer_gpu_prefill_attention_min_tokens() == 11);
    assert(bn_transformer_prefill_attention_min_tokens() == 11);
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");

    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN");
    assert(bn_transformer_gpu_prefill_ssm_run_chain_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN", "1", 1);
    assert(!bn_transformer_gpu_prefill_ssm_run_chain_enabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN");

    unsetenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");
    assert(!bn_transformer_gpu_prefill_moe_chain_debug_enabled());
    setenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN", "1", 1);
    assert(bn_transformer_gpu_prefill_moe_chain_debug_enabled());
    unsetenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");

    unsetenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN");
    assert(!bn_transformer_prefill_hybrid_chain_debug_enabled());
    assert(!bn_transformer_gpu_prefill_hybrid_chain_debug_enabled());
    setenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN", "1", 1);
    assert(bn_transformer_prefill_hybrid_chain_debug_enabled());
    assert(bn_transformer_gpu_prefill_hybrid_chain_debug_enabled());
    unsetenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN");

    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    unsetenv("BN_GPU_CPU_FALLBACK_FROM_LAYER");
    unsetenv("BN_GPU_CPU_ATTN_LAYER");
    unsetenv("BN_GPU_CPU_ATTN_FROM_LAYER");
    unsetenv("BN_GPU_CPU_FFN_LAYER");
    unsetenv("BN_GPU_CPU_FFN_FROM_LAYER");
    unsetenv("BN_GPU_CPU_FFN_DOWN_FROM_LAYER");
    BnTransformerGPUCPUFallbackPolicy fallback_policy =
        bn_transformer_gpu_cpu_fallback_policy();
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
    fallback_policy = bn_transformer_gpu_cpu_fallback_policy();
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
    BnTransformerGPUComparePolicy compare_policy =
        bn_transformer_gpu_compare_policy();
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
    compare_policy = bn_transformer_gpu_compare_policy();
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

    memset(&c, 0, sizeof(c));
    c.n_layers = 40;
    unsetenv("BN_GPU_Q4_Q8");
    unsetenv("BN_GPU_Q4_Q8_FROM_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TO_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
    unsetenv("BN_GPU_Q4_Q8_ATTN_ONLY");
    unsetenv("BN_GPU_Q4_Q8_FFN_ONLY");
    unsetenv("BN_METAL_Q4_PREPARED");
    BnTransformerGPUSmallDenseExactNativeLayerPolicy small_dense_exact_native_policy =
        bn_transformer_gpu_small_dense_exact_native_layer_policy(&c);
    assert(small_dense_exact_native_policy.from_layer == -1);
    assert(small_dense_exact_native_policy.to_layer == -1);
    assert(!small_dense_exact_native_policy.attn_only);
    assert(!small_dense_exact_native_policy.ffn_only);
    setenv("BN_GPU_Q4_Q8", "1", 1);
    small_dense_exact_native_policy = bn_transformer_gpu_small_dense_exact_native_layer_policy(&c);
    assert(small_dense_exact_native_policy.from_layer == 39);
    assert(small_dense_exact_native_policy.to_layer == 6);
    setenv("BN_METAL_Q4_PREPARED", "1", 1);
    small_dense_exact_native_policy = bn_transformer_gpu_small_dense_exact_native_layer_policy(&c);
    assert(small_dense_exact_native_policy.from_layer == 39);
    assert(small_dense_exact_native_policy.to_layer == -1);
    unsetenv("BN_METAL_Q4_PREPARED");
    setenv("BN_GPU_Q4_Q8_FROM_LAYER", "10", 1);
    setenv("BN_GPU_Q4_Q8_TO_LAYER", "20", 1);
    setenv("BN_GPU_Q4_Q8_ATTN_ONLY", "1", 1);
    setenv("BN_GPU_Q4_Q8_FFN_ONLY", "1", 1);
    small_dense_exact_native_policy = bn_transformer_gpu_small_dense_exact_native_layer_policy(&c);
    assert(small_dense_exact_native_policy.from_layer == 10);
    assert(small_dense_exact_native_policy.to_layer == 20);
    assert(small_dense_exact_native_policy.attn_only);
    assert(small_dense_exact_native_policy.ffn_only);
    unsetenv("BN_GPU_Q4_Q8_TO_LAYER");
    setenv("BN_GPU_Q4_Q8_TAIL_NATIVE", "4", 1);
    small_dense_exact_native_policy = bn_transformer_gpu_small_dense_exact_native_layer_policy(&c);
    assert(small_dense_exact_native_policy.to_layer == 35);
    setenv("BN_GPU_Q4_Q8_TAIL_NATIVE", "100", 1);
    small_dense_exact_native_policy = bn_transformer_gpu_small_dense_exact_native_layer_policy(&c);
    assert(small_dense_exact_native_policy.to_layer == -1);
    unsetenv("BN_GPU_Q4_Q8");
    unsetenv("BN_GPU_Q4_Q8_FROM_LAYER");
    unsetenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
    unsetenv("BN_GPU_Q4_Q8_ATTN_ONLY");
    unsetenv("BN_GPU_Q4_Q8_FFN_ONLY");
    unsetenv("BN_METAL_Q4_PREPARED");

    BnTransformerGPUSmallDenseExactNativeLayerPolicy manual_small_dense_exact_native_policy = {
        .from_layer = 2,
        .to_layer = 4,
        .attn_only = 0,
        .ffn_only = 0,
    };
    c.policy_flags = 0;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    BnTransformerGPUSmallDenseExactNativeLayerUsePolicy small_dense_exact_native_use =
        bn_transformer_gpu_small_dense_exact_native_layer_use_policy(
            &gpu, &c, &manual_small_dense_exact_native_policy, 1, 0, -1);
    assert(!small_dense_exact_native_use.use_layer);
    assert(!small_dense_exact_native_use.use_attention);
    assert(!small_dense_exact_native_use.use_ffn);
    assert(!small_dense_exact_native_use.use_ffn_down);
    small_dense_exact_native_use = bn_transformer_gpu_small_dense_exact_native_layer_use_policy(
        &gpu, &c, &manual_small_dense_exact_native_policy, 2, 0, -1);
    assert(small_dense_exact_native_use.use_layer);
    assert(small_dense_exact_native_use.use_attention);
    assert(small_dense_exact_native_use.use_ffn);
    assert(small_dense_exact_native_use.use_ffn_down);
    manual_small_dense_exact_native_policy.attn_only = 1;
    small_dense_exact_native_use = bn_transformer_gpu_small_dense_exact_native_layer_use_policy(
        &gpu, &c, &manual_small_dense_exact_native_policy, 2, 0, -1);
    assert(small_dense_exact_native_use.use_attention);
    assert(!small_dense_exact_native_use.use_ffn);
    assert(!small_dense_exact_native_use.use_ffn_down);
    manual_small_dense_exact_native_policy.attn_only = 0;
    manual_small_dense_exact_native_policy.ffn_only = 1;
    small_dense_exact_native_use = bn_transformer_gpu_small_dense_exact_native_layer_use_policy(
        &gpu, &c, &manual_small_dense_exact_native_policy, 2, 0, -1);
    assert(!small_dense_exact_native_use.use_attention);
    assert(small_dense_exact_native_use.use_ffn);
    assert(small_dense_exact_native_use.use_ffn_down);
    manual_small_dense_exact_native_policy.from_layer = -1;
    manual_small_dense_exact_native_policy.to_layer = -1;
    manual_small_dense_exact_native_policy.ffn_only = 0;
    c.policy_flags = BN_MODEL_ARCH_POLICY_MOE_EXACT_GPU_ATTENTION;
    small_dense_exact_native_use = bn_transformer_gpu_small_dense_exact_native_layer_use_policy(
        &gpu, &c, &manual_small_dense_exact_native_policy, 0, 0, -1);
    assert(!small_dense_exact_native_use.use_layer);
    assert(small_dense_exact_native_use.use_attention);
    assert(!small_dense_exact_native_use.use_ffn);
    manual_small_dense_exact_native_policy.ffn_only = 1;
    small_dense_exact_native_use = bn_transformer_gpu_small_dense_exact_native_layer_use_policy(
        &gpu, &c, &manual_small_dense_exact_native_policy, 0, 0, -1);
    assert(!small_dense_exact_native_use.use_attention);
    manual_small_dense_exact_native_policy.ffn_only = 0;
    c.policy_flags = BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE;
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN");
    unsetenv("BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN");
    small_dense_exact_native_use = bn_transformer_gpu_small_dense_exact_native_layer_use_policy(
        &gpu, &c, &manual_small_dense_exact_native_policy, 4, 1, 3);
    assert(!small_dense_exact_native_use.use_layer);
    assert(!small_dense_exact_native_use.small_dense_exact_native_path);
    small_dense_exact_native_use = bn_transformer_gpu_small_dense_exact_native_layer_use_policy(
        &gpu, &c, &manual_small_dense_exact_native_policy, 3, 1, 3);
    assert(small_dense_exact_native_use.use_layer);
    assert(small_dense_exact_native_use.small_dense_exact_native_path);
    assert(small_dense_exact_native_use.use_attention);
    assert(small_dense_exact_native_use.use_ffn);
    assert(!small_dense_exact_native_use.use_ffn_down);
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN", "1", 1);
    small_dense_exact_native_use = bn_transformer_gpu_small_dense_exact_native_layer_use_policy(
        &gpu, &c, &manual_small_dense_exact_native_policy, 3, 1, 3);
    assert(small_dense_exact_native_use.use_ffn_down);
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN");
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

    setenv("BN_GPU_FLASH_MIN_KV", "0", 1);
    setenv("BN_GPU_FLASH_MAX_KV", "2048", 1);
    gpu.caps = BN_GPU_CAP_FLASH_ATTN;
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_flash_attention_enabled(&gpu, 0, 0, 128));
    assert(!bn_transformer_gpu_flash_attention_enabled(&gpu, 0, 0, 4096));
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_flash_attention_enabled(&gpu, 0, 0, 128));
    assert(bn_transformer_gpu_flash_attention_enabled(&gpu, 1, 0, 128));

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
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_all_active_two_kquant_moe_model(&c, &moe_w));
    assert(bn_transformer_gpu_all_active_two_kquant_moe_layer(
        &c, &moe_layers[0], c.dim));
    unsetenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE");
    BnBackendModel *resident_backend = bn_backend_model_create();
    assert(resident_backend);
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
    assert(bn_transformer_gpu_moe_decode_cacheable(
        &c, &moe_w, resident_backend));
    moe_layers[0].moe.expert_map.down_cols = 4095;
    assert(!bn_transformer_gpu_moe_decode_cacheable(
        &c, &moe_w, resident_backend));
    moe_layers[0].moe.expert_map.down_cols = 4096;
    bn_backend_model_free(resident_backend);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_safe_default(
        &c, &moe_w));
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
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU");
    BnTransformerGPUMoERouteLayerPolicy route_layers = {-1, -1};
    BnTransformerGPUMoEDecodeRoutePolicy route_policy =
        bn_transformer_gpu_moe_decode_route_policy(
            &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
            (void *)2, (void *)3, (void *)4, (void *)5, (void *)6);
    assert(route_policy.all_active_two_kquant_moe);
    assert(!route_policy.route_layer_selected);
    assert(!route_policy.exact_gpu_route);
    assert(route_policy.router == (void *)2);
    assert(!route_policy.gpu_route_topk);
    assert(route_policy.cpu_route_resident_ffn);
    assert(route_policy.gpu_routed_ffn);
    assert(!bn_transformer_gpu_moe_route_topk_enabled(
        (void *)2, 1, 0));
    assert(bn_transformer_gpu_moe_routed_ffn_enabled(
        0, 1, (void *)4, (void *)5, (void *)6, &map, 4096, 2048));
    assert(route_policy.route_flags == 0);
    assert(bn_transformer_gpu_moe_route_normalization_flags(&c) == 0);
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
    setenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU", "1", 1);
    route_policy = bn_transformer_gpu_moe_decode_route_policy(
        &gpu, &c, &moe_layers[0], &route_layers, 0, c.dim,
        (void *)2, (void *)3, (void *)4, (void *)5, (void *)6);
    assert(route_policy.route_layer_selected);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_route_layer_selected(
        0, -1, -1));
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_exact_gpu_route_enabled(
        1, 1));
    assert(route_policy.router == (void *)3);
    assert(route_policy.gpu_route_topk);
    assert(!route_policy.cpu_route_resident_ffn);
    assert(route_policy.gpu_routed_ffn);
    assert(bn_transformer_gpu_moe_route_topk_enabled(
        (void *)2, 1, 1));
    assert(bn_transformer_gpu_moe_routed_ffn_enabled(
        1, 0, (void *)4, (void *)5, (void *)6, &map, 4096, 2048));
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN");
    unsetenv("BN_CUDA_DISABLE_MOE_FFN");
    assert(bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 0, -1, -1));
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN", "1", 1);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_exact_gpu_route_enabled(
        1, 1));
    assert(!bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 0, -1, -1));
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    setenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN", "1", 1);
    assert(!bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 0, -1, -1));
    assert(bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 3, 3, -1));
    assert(bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 3, -1, 2));
    setenv("BN_CUDA_DISABLE_MOE_FFN", "1", 1);
    assert(bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
        &gpu, &c, &map, c.dim, 1, 0, -1, -1));
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN");
    unsetenv("BN_CUDA_DISABLE_MOE_FFN");
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ATTN_SAFE", "1", 1);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_safe_default(
        &c, &moe_w));
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_fallback_enabled(
        &gpu, &c, &moe_w));
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_ATTN_SAFE");
    setenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE", "1", 1);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_safe_default(
        &c, &moe_w));
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_fallback_enabled(
        &gpu, &c, &moe_w));
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_ATTN_SAFE");
    assert(bn_transformer_gpu_all_active_two_kquant_moe_cpu_moe_safe_default(
        &c, &moe_w));
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN", "1", 1);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_moe_safe_default(
        &c, &moe_w));
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_MOE_SAFE", "1", 1);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_moe_safe_default(
        &c, &moe_w));
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_CPU_MOE_SAFE");
    setenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_MOE_SAFE", "1", 1);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_cpu_moe_safe_default(
        &c, &moe_w));
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_CPU_MOE_SAFE");
    c.policy_flags = BN_MODEL_ARCH_POLICY_MOE_EXACT_GPU_ATTENTION;
    assert(bn_transformer_gpu_moe_exact_attention_enabled(&gpu, &c));
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_ATTN", "1", 1);
    assert(!bn_transformer_gpu_moe_exact_attention_enabled(&gpu, &c));
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_EXACT_ATTN");
    setenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN", "1", 1);
    assert(!bn_transformer_gpu_moe_exact_attention_enabled(&gpu, &c));
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_EXACT_ATTN");
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_moe_exact_attention_enabled(&gpu, &c));
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
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE");
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    assert(bn_transformer_gpu_moe_routed_ffn_batch_allowed(&c));
    assert(bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(
        &gpu, &c));
    assert(!bn_transformer_gpu_moe_prefill_route_batch_available(
        &gpu, &c, 1));
    assert(!bn_transformer_gpu_moe_prefill_routed_ffn_batch_available(
        &gpu, &c, &map, c.dim, 0));
    assert(!bn_transformer_gpu_moe_prefill_resident_expert_batch_available(
        &gpu, &c, &map, c.dim, 0, 0));
    assert(!bn_transformer_gpu_moe_prefill_split_expert_batch_available(
        &gpu, &c, &map, c.dim, 0, 0));
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN", "1", 1);
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
    assert(!bn_transformer_gpu_moe_prefill_prefers_cached_expert_batch(
        &gpu, &c, 1));
    unsetenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL");
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
    assert(!bn_transformer_gpu_moe_prefill_shared_batch_available(
        &gpu, 3, 1));
    assert(bn_transformer_gpu_moe_prefill_shared_batch_available(
        &gpu, 4, 1));
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    setenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE", "1", 1);
    assert(!bn_transformer_gpu_moe_prefill_shared_batch_available(
        &gpu, 1, 1));
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
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
    assert(!bn_transformer_gpu_moe_prefill_split_shared_fuse_available(
        &gpu, &c, &moe_layers[0], 1));
    unsetenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE");
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
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    c.n_experts = 3;
    assert(!bn_transformer_gpu_moe_routed_ffn_batch_allowed(&c));
    assert(bn_transformer_gpu_moe_prefill_route_batch_available(
        &gpu, &c, 1));
    assert(!bn_transformer_gpu_moe_prefill_route_batch_available(
        &gpu, &c, 0));
    assert(!bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(
        &gpu, &c));
    setenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE", "1", 1);
    assert(bn_transformer_gpu_moe_routed_ffn_batch_allowed(&c));
    assert(bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(
        &gpu, &c));
    assert(bn_transformer_gpu_moe_prefill_routed_ffn_batch_available(
        &gpu, &c, &map, c.dim, 0));
    setenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH", "1", 1);
    assert(!bn_transformer_gpu_moe_routed_ffn_batch_allowed(&c));
    assert(!bn_transformer_gpu_moe_prefill_routed_ffn_norm_resid_available(
        &gpu, &c));
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH");
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE");
    c.n_experts = 2;
    assert(bn_transformer_gpu_prefill_ssm_layer_backend_available(&gpu));
    assert(!bn_transformer_gpu_prefill_moe_ffn_batch_available(
        &gpu, &c, &map, c.dim, 0));
    assert(!bn_transformer_prefill_moe_layer_backend_available(
        &gpu, &c, &map, c.dim, 0));
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN", "1", 1);
    assert(bn_transformer_gpu_prefill_moe_ffn_batch_available(
        &gpu, &c, &map, c.dim, 0));
    assert(bn_transformer_prefill_moe_layer_backend_available(
        &gpu, &c, &map, c.dim, 0));
    assert(!bn_transformer_gpu_prefill_moe_layer_chain_available(
        &gpu, &c, &map, c.dim, 0, 15));
    assert(bn_transformer_gpu_prefill_moe_layer_chain_available(
        &gpu, &c, &map, c.dim, 0, 16));
    assert(!bn_transformer_gpu_prefill_ssm_moe_chain_available(
        &gpu, &c, &map, c.dim, 0, 15));
    assert(bn_transformer_gpu_prefill_ssm_moe_chain_available(
        &gpu, &c, &map, c.dim, 0, 16));
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    assert(!bn_transformer_gpu_prefill_ssm_moe_chain_available(
        &gpu, &c, &map, c.dim, 0, 16));
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
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN");

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
    c.policy_flags = BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE;
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
        &gpu, &c, &dense_w));
    assert(!bn_transformer_gpu_small_dense_exact_native_default(
        &gpu, &c, -1));
    assert(!bn_transformer_gpu_small_dense_exact_native_ffn_down_enabled(
        &gpu, &c));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
        &gpu, &c, &dense_w));
    fallback = (BnTransformerGPUCPUFallbackPolicy)
        {-1, -1, -1, -1, -1, -1, -1};
    fallback = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        fallback, &gpu, &c, &dense_w);
    assert(fallback.attn_from_layer == 0);
    assert(bn_transformer_gpu_small_dense_exact_native_default(
        &gpu, &c, -1));
    BnTransformerGPUSmallDenseExactNativeLayerPolicy small_dense_exact_native_layer =
        {.from_layer = -1, .to_layer = -1};
    BnTransformerGPUSmallDenseExactNativeDecodePolicy small_dense_exact_native_decode =
        bn_transformer_gpu_small_dense_exact_native_decode_policy(&gpu, &c, &small_dense_exact_native_layer);
    assert(small_dense_exact_native_decode.small_dense_exact_native_default);
    c.n_layers = 61;
    assert(bn_model_arch_small_dense_exact_native_to_layer(&c) == 27);
    small_dense_exact_native_decode =
        bn_transformer_gpu_small_dense_exact_native_decode_policy(&gpu, &c, &small_dense_exact_native_layer);
    assert(small_dense_exact_native_decode.small_dense_exact_native_default);
    assert(small_dense_exact_native_decode.small_dense_exact_native_to_layer == 27);
    assert(bn_transformer_gpu_small_dense_exact_native_to_layer(
               &c, 1, -1) == 27);
    small_dense_exact_native_layer.to_layer = 9;
    small_dense_exact_native_decode =
        bn_transformer_gpu_small_dense_exact_native_decode_policy(&gpu, &c, &small_dense_exact_native_layer);
    assert(small_dense_exact_native_decode.small_dense_exact_native_to_layer == 9);
    assert(bn_transformer_gpu_small_dense_exact_native_to_layer(
               &c, 1, 9) == 9);
    small_dense_exact_native_layer.to_layer = -1;
    c.n_layers = 33;
    assert(bn_model_arch_small_dense_exact_native_to_layer(&c) == -1);
    small_dense_exact_native_decode =
        bn_transformer_gpu_small_dense_exact_native_decode_policy(&gpu, &c, &small_dense_exact_native_layer);
    assert(small_dense_exact_native_decode.small_dense_exact_native_to_layer == -1);
    assert(bn_transformer_gpu_small_dense_exact_native_to_layer(
               &c, 1, -1) == -1);
    c.n_layers = 0;
    assert(bn_transformer_gpu_small_dense_exact_native_to_layer(
               &c, 0, -1) == -1);
    assert(!bn_transformer_gpu_small_dense_exact_native_default(
        &gpu, &c, 0));
    small_dense_exact_native_layer.from_layer = 0;
    small_dense_exact_native_decode =
        bn_transformer_gpu_small_dense_exact_native_decode_policy(&gpu, &c, &small_dense_exact_native_layer);
    assert(!small_dense_exact_native_decode.small_dense_exact_native_default);
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_CPU_ATTN_SAFE", "1", 1);
    assert(!bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
        &gpu, &c, &dense_w));
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_CPU_ATTN_SAFE");
    setenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE", "1", 1);
    assert(!bn_transformer_gpu_small_dense_native_quant_cpu_attn_fallback_enabled(
        &gpu, &c, &dense_w));
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_CPU_ATTN_SAFE");
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_EXACT_Q4_Q8", "1", 1);
    assert(!bn_transformer_gpu_small_dense_exact_native_default(
        &gpu, &c, -1));
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_EXACT_Q4_Q8");
    setenv("BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8", "1", 1);
    assert(!bn_transformer_gpu_small_dense_exact_native_default(
        &gpu, &c, -1));
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_EXACT_Q4_Q8");
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN", "1", 1);
    assert(bn_transformer_gpu_small_dense_exact_native_ffn_down_enabled(
        &gpu, &c));
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_EXACT_FFN_DOWN");
    setenv("BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN", "1", 1);
    assert(bn_transformer_gpu_small_dense_exact_native_ffn_down_enabled(
        &gpu, &c));
    unsetenv("BN_CUDA_ENABLE_SMALL_QWEN_EXACT_FFN_DOWN");
    c.policy_flags |= BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK;
    assert(!bn_transformer_gpu_small_dense_prefill_decode_fallback_requested(
        &gpu, &c));
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_PREFILL", "1", 1);
    assert(bn_transformer_gpu_small_dense_prefill_decode_fallback_requested(
        &gpu, &c));
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_PREFILL");
    setenv("BN_CUDA_DISABLE_SMALL_QWEN_PREFILL", "1", 1);
    assert(bn_transformer_gpu_small_dense_prefill_decode_fallback_requested(
        &gpu, &c));
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_PREFILL");
    c.policy_flags |= BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE;
    assert(!bn_transformer_gpu_native_quant_logits_refine_enabled(
        &gpu, &c, BN_GGUF_TENSOR_Q8_0));
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE", "1", 1);
    assert(bn_transformer_gpu_native_quant_logits_refine_enabled(
        &gpu, &c, BN_GGUF_TENSOR_Q8_0));
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE", "1", 1);
    assert(!bn_transformer_gpu_native_quant_logits_refine_enabled(
        &gpu, &c, BN_GGUF_TENSOR_Q8_0));
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE");
    setenv("BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE", "1", 1);
    assert(bn_transformer_gpu_native_quant_logits_refine_enabled(
        &gpu, &c, BN_GGUF_TENSOR_Q8_0));
    setenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE", "1", 1);
    assert(!bn_transformer_gpu_native_quant_logits_refine_enabled(
        &gpu, &c, BN_GGUF_TENSOR_Q8_0));
    unsetenv("BN_CUDA_ENABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_QWEN_Q8_LOGITS_REFINE");
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
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN", "1", 1);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_logits_refine_default(
        &gpu, &c, &logits_refine_weights));
    setenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q6_LOGITS_REFINE", "1", 1);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_logits_refine_default(
        &gpu, &c, &logits_refine_weights));
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    unsetenv("BN_CUDA_DISABLE_ALL2_Q4Q6_MOE_Q6_LOGITS_REFINE");
    setenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN", "1", 1);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_logits_refine_default(
        &gpu, &c, &logits_refine_weights));
    setenv("BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE", "1", 1);
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_logits_refine_default(
        &gpu, &c, &logits_refine_weights));
    unsetenv("BN_CUDA_ENABLE_QWEN2MOE_FAST_MOE_FFN");
    unsetenv("BN_CUDA_DISABLE_QWEN2MOE_Q6_LOGITS_REFINE");
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
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_large_hybrid_cpu_attn_safe_fallback_enabled(
        &gpu, &c, &hybrid_w));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_gpu_large_hybrid_cpu_attn_safe_default(
        &c, &hybrid_w));
    assert(bn_transformer_gpu_large_hybrid_cpu_attn_safe_fallback_enabled(
        &gpu, &c, &hybrid_w));
    BnTransformerGPUDecodeEntryPolicy decode_entry =
        bn_transformer_gpu_decode_entry_policy(&gpu, &c, &hybrid_w, 1);
    assert(decode_entry.block_argmax);
    decode_entry =
        bn_transformer_gpu_decode_entry_policy(&gpu, &c, &hybrid_w, 0);
    assert(!decode_entry.block_argmax);
    setenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX", "1", 1);
    decode_entry =
        bn_transformer_gpu_decode_entry_policy(&gpu, &c, &hybrid_w, 1);
    assert(!decode_entry.block_argmax);
    unsetenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX");
    fallback = (BnTransformerGPUCPUFallbackPolicy)
        {-1, -1, -1, -1, -1, -1, -1};
    fallback = bn_transformer_gpu_decode_cpu_attention_fallback_policy(
        fallback, &gpu, &c, &hybrid_w);
    assert(fallback.attn_from_layer == 0);
    unsetenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE");
    assert(!bn_transformer_gpu_ssm_cpu_fallback_required(&gpu));
    setenv("BN_CUDA_DISABLE_SSM_GRAPH", "1", 1);
    assert(bn_transformer_gpu_ssm_cpu_fallback_required(&gpu));
    unsetenv("BN_CUDA_DISABLE_SSM_GRAPH");
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_gpu_ssm_cpu_fallback_required(&gpu));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    c.full_attn_interval = 0;
    c.ssm_inner_size = 0;
    c.dim = 2048;

    c.seq_len = 32;
    c.kv_f16 = 0;
    unsetenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV");
    unsetenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK");
    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    assert(bn_transformer_gpu_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    assert(bn_transformer_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    setenv("BN_GPU_CPU_FALLBACK_LAYER", "0", 1);
    assert(!bn_transformer_gpu_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    assert(!bn_transformer_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    setenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK", "1", 1);
    assert(bn_transformer_gpu_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    assert(bn_transformer_prefill_direct_kv_allowed(
        &c, &dense_w, &gpu, 0, 16));
    unsetenv("BN_GPU_CPU_FALLBACK_LAYER");
    unsetenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK");
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
    BnTransformerGPUGenerateArgmaxPolicy generate_argmax =
        bn_transformer_gpu_generate_argmax_policy(&gpu, 0, 0.0f, 1.0f);
    assert(generate_argmax.enabled);
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
    assert(bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    W.rows = 1024;
    logits.rows = W.rows;
    assert(!bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    setenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX", "1", 1);
    assert(bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    unsetenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX");
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
    assert(!bn_transformer_gpu_matvec_argmax_enabled(
        &gpu, &c, &logits, 1, 0, 0));
    unsetenv("BN_GPU_CPU_LOGITS");
    setenv("BN_CUDA_DISABLE_LOGITS_ARGMAX", "1", 1);
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
    unsetenv("BN_METAL_ENABLE_Q6_Q8K");
    BnTransformerGPULogitsRefinePolicy decode_refine = {0};
    BnTransformerGPUCPUFallbackPolicy decode_fallback =
        {-1, -1, -1, -1, -1, -1, -1};
    BnTransformerGPUComparePolicy decode_compare =
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
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
    setenv("BN_METAL_ENABLE_Q6_Q8K", "1", 1);
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    unsetenv("BN_METAL_ENABLE_Q6_Q8K");
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 1, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    setenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE", "1", 1);
    assert(bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 1, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    unsetenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE");
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 1, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    assert(bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 1, 0, 0, 0, 1, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 0, 0, 0,
        0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    setenv("BN_CUDA_DISABLE_DECODE_CACHE", "1", 1);
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    unsetenv("BN_CUDA_DISABLE_DECODE_CACHE");
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_decode_cacheable(
        &gpu, 1, 0, 0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    unsetenv("BN_CUDA_ENABLE_LOGITS_CACHE");

    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
    gpu.kind = BN_GPU_BACKEND_METAL;
    assert(!bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 16));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(!bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 15));
    assert(bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 16));
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    assert(!bn_transformer_gpu_prefill_ssm_dense_chain_available(
        &gpu, &c, 16));
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
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

    int route_from = 0;
    int route_to = 0;
    unsetenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_TO_LAYER");
    unsetenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER");
    bn_transformer_gpu_all_active_two_kquant_moe_route_layer_range(
        &route_from, &route_to);
    assert(route_from == -1);
    assert(route_to == -1);

    setenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_FROM_LAYER", "2", 1);
    setenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_TO_LAYER", "6", 1);
    bn_transformer_gpu_all_active_two_kquant_moe_route_layer_range(
        &route_from, &route_to);
    assert(route_from == 2);
    assert(route_to == 6);
    unsetenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_ALL2_Q4Q6_MOE_GPU_ROUTE_TO_LAYER");

    setenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER", "3", 1);
    setenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER", "7", 1);
    bn_transformer_gpu_all_active_two_kquant_moe_route_layer_range(
        &route_from, &route_to);
    assert(route_from == 3);
    assert(route_to == 7);
    unsetenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER");
    unsetenv("BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER");
    BnTransformerGPUMoERouteLayerPolicy route_layer_policy =
        bn_transformer_gpu_moe_route_layer_policy();
    assert(route_layer_policy.from_layer == -1);
    assert(route_layer_policy.to_layer == -1);

    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    c.dim = 2048;
    c.moe_norm_topk_prob = 1;
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU");
    unsetenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2");
    setenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU", "1", 1);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
        &c, (void *)1, NULL));
    gpu.kind = BN_GPU_BACKEND_CUDA;
    BnTransformerGPUMoEDirectRoutePolicy direct_route =
        bn_transformer_gpu_moe_direct_route_policy(&gpu, &c, (void *)1, NULL);
    assert(direct_route.enabled);
    assert(direct_route.router_diff == (void *)1);
    gpu.kind = BN_GPU_BACKEND_METAL;
    direct_route =
        bn_transformer_gpu_moe_direct_route_policy(&gpu, &c, (void *)1, NULL);
    assert(!direct_route.enabled);
    gpu.kind = BN_GPU_BACKEND_CUDA;
    direct_route =
        bn_transformer_gpu_moe_direct_route_policy(&gpu, &c, NULL, NULL);
    assert(!direct_route.enabled);
    assert(bn_transformer_gpu_all_active_two_kquant_moe_router(
        &c, (void *)2, (void *)1, 1, 0) == (void *)1);
    c.n_experts_active = 1;
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
        &c, (void *)1, NULL));
    assert(bn_transformer_gpu_all_active_two_kquant_moe_router(
        &c, (void *)2, (void *)1, 1, 0) == (void *)2);
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4095;
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
        &c, (void *)1, NULL));
    c.moe_intermediate_size = 4096;
    assert(!bn_transformer_gpu_all_active_two_kquant_moe_direct_route_enabled(
        &c, (void *)1, (void *)3));
    BnTransformerGPUMoEAllActiveTwoResourcePolicy all_active_two_resources =
        bn_transformer_gpu_moe_all_active_two_resource_policy(&c);
    assert(all_active_two_resources.enabled);
    c.dim = 2049;
    all_active_two_resources =
        bn_transformer_gpu_moe_all_active_two_resource_policy(&c);
    assert(!all_active_two_resources.enabled);
    c.dim = 2048;
    c.n_experts_active = 1;
    all_active_two_resources =
        bn_transformer_gpu_moe_all_active_two_resource_policy(&c);
    assert(!all_active_two_resources.enabled);
    c.n_experts_active = 2;
    unsetenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU");

    printf("PASSED\n");
}

static void test_logits_policy_helpers(void) {
    printf("test_logits_policy_helpers... ");

    unsetenv("BN_CPU_TIED_Q6K_REFINE_TOP");
    assert(bn_transformer_logits_cpu_tied_kquant_refine_top() == 0);
    setenv("BN_CPU_TIED_Q6K_REFINE_TOP", "0", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_refine_top() == 0);
    setenv("BN_CPU_TIED_Q6K_REFINE_TOP", "7", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_refine_top() == 7);
    setenv("BN_CPU_TIED_Q6K_REFINE_TOP", "200", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_refine_top() == 128);
    unsetenv("BN_CPU_TIED_Q6K_REFINE_TOP");

    unsetenv("BN_CPU_TIED_Q6K_HYBRID_TOP");
    assert(bn_transformer_logits_cpu_tied_kquant_hybrid_top() == 0);
    setenv("BN_CPU_TIED_Q6K_HYBRID_TOP", "1", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_hybrid_top() == 0);
    setenv("BN_CPU_TIED_Q6K_HYBRID_TOP", "9", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_hybrid_top() == 9);
    setenv("BN_CPU_TIED_Q6K_HYBRID_TOP", "200", 1);
    assert(bn_transformer_logits_cpu_tied_kquant_hybrid_top() == 128);
    unsetenv("BN_CPU_TIED_Q6K_HYBRID_TOP");

    unsetenv("BN_CPU_NATIVE_TIED_LOGITS");
    assert(!bn_transformer_logits_cpu_native_tied_quant_enabled());
    setenv("BN_CPU_NATIVE_TIED_LOGITS", "1", 1);
    assert(bn_transformer_logits_cpu_native_tied_quant_enabled());
    unsetenv("BN_CPU_NATIVE_TIED_LOGITS");

    assert(bn_transformer_logits_untied_uses_f16_path(BN_GGUF_TENSOR_F16));
    assert(!bn_transformer_logits_untied_uses_f16_path(BN_GGUF_TENSOR_Q4_K));
    assert(bn_transformer_logits_tied_uses_quant_path(BN_GGUF_TENSOR_Q6_K));
    assert(!bn_transformer_logits_tied_uses_quant_path(BN_GGUF_TENSOR_F32));
    assert(bn_transformer_logits_tied_uses_f16_path(BN_GGUF_TENSOR_F16));
    assert(!bn_transformer_logits_tied_uses_f16_path(BN_GGUF_TENSOR_Q6_K));
    assert(bn_transformer_logits_tied_i8_weight_type() == BN_GGUF_TENSOR_Q8_0);
    assert(bn_transformer_logits_tied_f16_weight_type() == BN_GGUF_TENSOR_F16);
    assert(bn_transformer_logits_tied_f32_weight_type() == BN_GGUF_TENSOR_F32);
    assert(bn_transformer_logits_native_quant_task_flags(0) == 0);
    assert(bn_transformer_logits_native_quant_task_flags(1) ==
           BN_MATVEC_TASK_NATIVE_QUANT);

    unsetenv("BN_GPU_Q8_REFINE_TOP");
    assert(bn_transformer_logits_native_quant_refine_top() == 16);
    setenv("BN_GPU_Q8_REFINE_TOP", "6", 1);
    assert(bn_transformer_logits_native_quant_refine_top() == 6);
    unsetenv("BN_GPU_Q8_REFINE_TOP");

    BnGPUBackend gpu = {0};
    BnConfig c = {0};
    BnQWeight q8 = {0};
    q8.type = BN_GGUF_TENSOR_Q8_0;
    assert(!bn_transformer_logits_native_quant_refine_enabled(
        NULL, &c, &q8));
    gpu.kind = BN_GPU_BACKEND_METAL;
    c.dim = 2048;
    c.policy_flags = BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE |
                     BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE;
    setenv("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE", "1", 1);
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
    assert(!bn_transformer_logits_native_quant_refine_enabled(
        &gpu, &c, &q8));
    c.policy_flags = BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE |
                     BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE;
    setenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE", "1", 1);
    assert(!bn_transformer_logits_native_quant_refine_enabled(
        &gpu, &c, &q8));
    unsetenv("BN_CUDA_ENABLE_SMALL_DENSE_Q8_LOGITS_REFINE");
    unsetenv("BN_CUDA_DISABLE_SMALL_DENSE_Q8_LOGITS_REFINE");

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
    assert(bn_transformer_gpu_activation_is_relu2(
        BN_MODEL_ACTIVATION_RELU2));
    assert(!bn_transformer_gpu_activation_is_relu2(
        BN_MODEL_ACTIVATION_SILU));
    assert(bn_transformer_gpu_ffn_activation_kind(
               BN_MODEL_ACTIVATION_SILU) == BN_GPU_IR_ACTIVATION_SILU);
    assert(bn_transformer_gpu_ffn_activation_kind(
               BN_MODEL_ACTIVATION_RELU2) == BN_GPU_IR_ACTIVATION_RELU2);
    assert(bn_transformer_gpu_ffn_activation_kind(
               BN_MODEL_ACTIVATION_GELU) == BN_GPU_IR_ACTIVATION_SILU);

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
    assert(ctx_ops[0].W_buf == (void *)3);
    assert(ctx_ops[1].op_kind == BN_GPU_OP_LOGITS);
    assert(ctx_ops[1].op_code == BN_GPU_CODE_MATVEC);
    assert(ctx_ops[1].type == BN_GGUF_TENSOR_Q8_0);
    assert(ctx_ops[1].W_buf == (void *)4);
    bn_transformer_gpu_emit_context_free(&ctx);

    BnGPUOp ctx_ops2[5];
    bn_transformer_gpu_emit_context_init(&ctx, ctx_ops2, 5);
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
    assert(bn_transformer_gpu_emit_context_lower_pending(&ctx) == 0);
    assert(ctx.n == 5);
    assert(ctx_ops2[0].op_code == BN_GPU_CODE_COPY);
    assert(ctx_ops2[1].op_code == BN_GPU_CODE_RESIDUAL_ADD);
    assert(ctx_ops2[2].op_code == BN_GPU_CODE_SILU_GATE);
    assert(ctx_ops2[3].op_code == BN_GPU_CODE_MATVEC);
    assert(ctx_ops2[4].op_code == BN_GPU_CODE_FUSED_GATEUP_SILU);
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
    assert(gemma->policy_flags & BN_MODEL_ARCH_POLICY_LARGE_GPU_GRAPH_FALLBACK);
    assert(strcmp(gemma->prefix("gemma4"), "gemma4") == 0);
    assert(gemma->attention_value_shares_key("gemma4"));
    assert(gemma->activation("gemma4") == 2);

    BnConfig c = {0};
    bn_model_arch_apply_config(&c, gemma);
    assert(c.policy_flags == gemma->policy_flags);
    assert(bn_model_arch_requires_large_gpu_graph_fallback(&c));
    assert(!bn_model_arch_requires_float_kquant_fallback(&c));
    assert(bn_model_arch_attention_scale(&c, 128) == 1.0f);
    assert(bn_model_arch_rmsnorm_mode(&c) ==
           BN_MODEL_ARCH_RMSNORM_BACKEND_ORDER);
    assert(bn_model_arch_attention_value_shares_key_config(&c));
    assert(bn_model_arch_uses_per_layer_embedding(&c));
    assert(bn_model_arch_uses_attention_post_norm(&c));
    assert(bn_model_arch_uses_ffn_post_norm(&c));
    assert(bn_model_arch_uses_layer_output_scale(&c));
    assert(bn_model_arch_loads_extra_metadata(&c));
    c.per_layer_input_dim = 128;
    assert(bn_model_arch_loads_per_layer_input_weights(&c));
    assert(bn_model_arch_divides_rope_freqs(&c, 0));
    assert(bn_model_arch_prefill_uses_decode_for_parity(&c));
    assert(bn_model_arch_cpu_prefill_uses_decode_for_parity(&c));
    assert(!bn_model_arch_ffn_uses_exact_scalar_activation(&c));
    assert(!bn_model_arch_moe_forces_float_kquant_gateup(&c));
    assert(bn_model_arch_moe_uses_scaled_router_input(&c));
    assert(bn_model_arch_moe_uses_dense_residual_branch(&c));
    assert(bn_model_arch_loads_extra_ffn_post_norms(&c));
    assert(bn_model_arch_loads_moe_aux_weights(&c));

    c.per_layer_input_dim = 0;
    c.n_experts = 4;
    c.n_layers = 30;
    c.kv_unique_layer_count = 20;
    c.sliding_window_pattern[20] = 0;
    c.sliding_window_pattern[21] = 1;
    assert(!bn_model_arch_loads_per_layer_input_weights(&c));
    assert(!bn_model_arch_divides_rope_freqs(&c, 0));
    assert(bn_model_arch_divides_rope_freqs(&c, 5));
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

    const BnModelArchOps *qwen = bn_model_arch_ops_for("qwen35");
    assert(qwen);
    assert(strcmp(qwen->name, "qwen35") == 0);
    assert(qwen->policy_flags & BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM);
    assert(qwen->policy_flags & BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS);
    assert(strcmp(qwen->prefix("qwen35"), "qwen35") == 0);
    assert(qwen->activation("qwen35") == 0);
    assert(!qwen->attention_value_shares_key("qwen35"));
    memset(&c, 0, sizeof(c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS;
    assert(bn_model_arch_config_uses_full_rope_text_dims(&c));
    c.policy_flags = 0;
    assert(!bn_model_arch_config_uses_full_rope_text_dims(&c));
    assert(bn_model_arch_tokenizer_uses_metaspace("gemma4"));
    assert(!bn_model_arch_tokenizer_uses_metaspace("llama"));

    memset(&c, 0, sizeof(c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK |
                     BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM |
                     BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
                     BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK |
                     BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE |
                     BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE |
                     BN_MODEL_ARCH_POLICY_PREFILL_EXACT_ACTIVATION |
                     BN_MODEL_ARCH_POLICY_EXACT_SCALAR_FFN_ACTIVATION;
    assert(bn_model_arch_requires_float_kquant_fallback(&c));
    assert(bn_model_arch_cpu_force_float_kquant(&c));
    assert(fabsf(bn_model_arch_attention_scale(&c, 128) -
                 (1.0f / sqrtf(128.0f))) < 1e-7f);
    assert(bn_model_arch_rmsnorm_mode(&c) ==
           BN_MODEL_ARCH_RMSNORM_BACKEND_ORDER);
    assert(!bn_model_arch_attention_value_shares_key_config(&c));
    assert(!bn_model_arch_uses_per_layer_embedding(&c));
    assert(!bn_model_arch_uses_attention_post_norm(&c));
    assert(!bn_model_arch_uses_ffn_post_norm(&c));
    assert(!bn_model_arch_uses_layer_output_scale(&c));
    assert(!bn_model_arch_uses_reference_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_hybrid_layer_layout(&c));
    assert(!bn_model_arch_uses_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_large_dense_hybrid_ssm(&c));
    assert(bn_model_arch_uses_dense_attention_only(&c));
    assert(!bn_model_arch_uses_large_dense_shape(&c));
    assert(!bn_model_arch_uses_large_gpu_graph_fallback_shape(&c));
    assert(!bn_model_arch_moe_forces_float_kquant_gateup(&c));
    assert(!bn_model_arch_moe_prefers_exact_gpu_attention(&c));
    assert(!bn_model_arch_moe_uses_scaled_router_input(&c));
    assert(!bn_model_arch_moe_uses_dense_residual_branch(&c));
    assert(!bn_model_arch_uses_moe(&c));
    assert(!bn_model_arch_uses_non_hybrid_moe(&c));
    assert(!bn_model_arch_uses_two_expert_all_active_moe(&c));
    assert(!bn_model_arch_uses_more_than_two_expert_moe(&c));
    assert(!bn_model_arch_moe_prefill_forces_matvec(&c));
    assert(!bn_model_arch_uses_all_active_two_expert_moe(&c, c.dim));
    assert(bn_model_arch_uses_small_dense_shape(&c));
    assert(!bn_model_arch_uses_small_dense_native_quant_shape(&c));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 0));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    assert(bn_model_arch_allows_small_dense_prefill_decode_fallback(&c));
    assert(bn_model_arch_prefill_uses_decode_for_parity(&c));
    assert(bn_model_arch_allows_small_dense_exact_native(&c));
    assert(bn_model_arch_allows_small_dense_native_logit_refine(&c));
    assert(bn_model_arch_small_dense_prefill_min_tokens(&c) == 7);
    assert(bn_model_arch_prefill_uses_exact_activation(&c));
    assert(bn_model_arch_ffn_uses_exact_scalar_activation(&c));

    c.full_attn_interval = 4;
    assert(bn_model_arch_uses_reference_hybrid_ssm(&c));
    assert(bn_model_arch_uses_scalar_hybrid_ssm_cpu(&c));
    assert(bn_model_arch_uses_hybrid_layer_layout(&c));
    assert(!bn_model_arch_uses_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_dense_attention_only(&c));
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 0));
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    assert(!bn_model_arch_uses_small_dense_shape(&c));
    assert(!bn_model_arch_uses_small_dense_native_quant_shape(&c));
    assert(!bn_model_arch_allows_small_dense_prefill_decode_fallback(&c));
    assert(!bn_model_arch_allows_small_dense_exact_native(&c));
    assert(!bn_model_arch_allows_small_dense_native_logit_refine(&c));
    assert(bn_model_arch_small_dense_prefill_min_tokens(&c) == 0);
    c.ssm_inner_size = 128;
    c.dim = 4095;
    assert(bn_model_arch_uses_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_large_dense_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_large_dense_shape(&c));
    assert(!bn_model_arch_uses_large_gpu_graph_fallback_shape(&c));
    assert(!bn_model_arch_uses_small_dense_native_quant_shape(&c));
    c.dim = 4096;
    assert(bn_model_arch_uses_large_dense_shape(&c));
    assert(bn_model_arch_uses_large_dense_hybrid_ssm(&c));
    assert(bn_model_arch_uses_large_gpu_graph_fallback_shape(&c));
    c.n_experts = 1;
    assert(!bn_model_arch_uses_large_dense_shape(&c));
    assert(!bn_model_arch_uses_large_dense_hybrid_ssm(&c));
    assert(!bn_model_arch_uses_small_dense_shape(&c));
    assert(!bn_model_arch_uses_small_dense_native_quant_shape(&c));
    assert(bn_model_arch_uses_moe(&c));
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    assert(!bn_model_arch_uses_non_hybrid_moe(&c));
    assert(bn_model_arch_uses_hybrid_moe(&c));
    assert(bn_model_arch_uses_large_gpu_graph_fallback_shape(&c));
    assert(!bn_model_arch_uses_dense_attention_only(&c));
    assert(!bn_model_arch_dense_logits_argmax_shape_allowed(&c, 300000));
    assert(bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(&c, 1536));
    assert(!bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(&c, 2048));

    memset(&c, 0, sizeof(c));
    c.n_experts = 2;
    c.n_experts_active = 2;
    c.moe_intermediate_size = 4096;
    c.dim = 2048;
    assert(!bn_model_arch_uses_dense_attention_only(&c));
    assert(bn_model_arch_uses_moe(&c));
    assert(bn_model_arch_uses_non_hybrid_moe(&c));
    assert(!bn_model_arch_uses_hybrid_moe(&c));
    assert(bn_model_arch_uses_two_expert_all_active_moe(&c));
    assert(!bn_model_arch_uses_more_than_two_expert_moe(&c));
    assert(!bn_model_arch_moe_prefill_forces_matvec(&c));
    assert(bn_model_arch_uses_all_active_two_expert_moe(&c, c.dim));
    c.has_shared_expert = 1;
    assert(bn_model_arch_moe_prefill_forces_matvec(&c));
    c.has_shared_expert = 0;
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
                     BN_MODEL_ARCH_POLICY_LARGE_GPU_GRAPH_FALLBACK |
                     BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_NATIVE |
                     BN_MODEL_ARCH_POLICY_SMALL_DENSE_NATIVE_LOGIT_REFINE |
                     BN_MODEL_ARCH_POLICY_EXACT_SCALAR_FFN_ACTIVATION;
    assert(!bn_model_arch_requires_float_kquant_fallback(&c));
    assert(!bn_model_arch_moe_forces_float_kquant_gateup(&c));
    assert(!bn_model_arch_moe_prefers_exact_gpu_attention(&c));
    assert(bn_model_arch_rmsnorm_mode(&c) ==
           BN_MODEL_ARCH_RMSNORM_REFERENCE_SCALAR_ORDER);
    assert(bn_model_arch_rmsnorm_requires_reference_scalar_order(&c));
    assert(bn_model_arch_uses_small_dense_shape(&c));
    assert(!bn_model_arch_uses_small_dense_native_quant_shape(&c));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 0));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    assert(bn_model_arch_small_dense_prefill_min_tokens(&c) == 2);
    assert(!bn_model_arch_uses_large_gpu_graph_fallback_shape(&c));
    assert(!bn_model_arch_prefill_uses_exact_activation(&c));
    assert(bn_model_arch_ffn_uses_exact_scalar_activation(&c));
    assert(bn_model_arch_dense_logits_argmax_shape_allowed(&c, 300000));
    assert(!bn_model_arch_dense_logits_argmax_shape_allowed(&c, 262144));
    assert(!bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(&c, 1536));
    c.dim = 1025;
    assert(bn_model_arch_uses_small_dense_native_quant_shape(&c));
    c.dim = 2561;
    assert(!bn_model_arch_uses_small_dense_shape(&c));
    assert(!bn_model_arch_uses_small_dense_native_quant_shape(&c));
    assert(!bn_model_arch_small_dense_prefill_min_tokens(&c));
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 0));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    assert(!bn_model_arch_uses_large_gpu_graph_fallback_shape(&c));
    c.dim = 4096;
    assert(bn_model_arch_uses_large_gpu_graph_fallback_shape(&c));
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 0));
    assert(bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    c.dim = 8193;
    assert(!bn_model_arch_dense_batch_prefill_shape_allowed(&c, 1));
    c.dim = 0;

    c.policy_flags |= BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP |
                      BN_MODEL_ARCH_POLICY_MOE_EXACT_GPU_ATTENTION;
    assert(bn_model_arch_moe_forces_float_kquant_gateup(&c));
    assert(bn_model_arch_moe_prefers_exact_gpu_attention(&c));

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
    assert(p.ssm_idx == -1);
    assert(p.q_dim == 2048);
    assert(!p.q_gated);
    assert(!p.q_wide);
    assert(p.head_size == 128);
    assert(p.kv_dim == 512);
    assert(p.n_kv_heads == 4);
    assert(p.kv_mul == 4);
    assert(p.qk_stride == 128);
    assert(p.has_qk_norm);
    assert(p.has_bias);
    assert(p.kv_mode == BN_KV_FP32);
    assert(bn_transformer_attention_head_size(&c, &lw) == 128);
    assert(bn_transformer_attention_kv_dim(&c, &lw) == 512);
    assert(bn_transformer_attention_n_kv_heads(&c, &lw) == 4);
    assert(bn_transformer_attention_kv_mul(&c, &lw) == 4);
    assert(bn_transformer_attention_qk_stride(&c, p.head_size) == 128);
    assert(bn_transformer_attention_has_qk_norm(&lw));
    assert(bn_transformer_attention_has_bias(&lw));

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
    assert(p.q_dim == 3072);
    assert(p.head_size == 192);
    assert(p.kv_dim == 768);
    assert(p.kv_mode == BN_KV_TQ);
    assert(bn_transformer_attention_head_size(&c, &lw) == 192);
    assert(bn_transformer_attention_kv_dim(&c, &lw) == 768);
    c.qk_norm_per_head = 0;
    assert(bn_transformer_attention_qk_stride(&c, p.head_size) == 0);
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

    gpu.caps = BN_GPU_CAP_Q4_MATVEC_SPLIT |
               BN_GPU_CAP_Q4K_MATVEC_SPLIT |
               BN_GPU_CAP_Q8_MATVEC_SPLIT |
               BN_GPU_CAP_Q5K_MATVEC_SPLIT |
               BN_GPU_CAP_Q4_FUSED_GATEUP_SILU |
               BN_GPU_CAP_FLASH_ATTN;
    gpu.kind = BN_GPU_BACKEND_METAL;

    lw.block_kind = BN_LAYER_BLOCK_ATTENTION;
    lw.ffn_kind = BN_LAYER_FFN_DENSE;
    lw.attn.wq.data = (void *)1;
    lw.attn.wq.rows = 2048;
    lw.attn.wq.type = BN_GGUF_TENSOR_Q4_0;
    lw.attn.wk.type = BN_GGUF_TENSOR_Q4_0;
    lw.attn.wv.type = BN_GGUF_TENSOR_Q4_0;

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

    BnLayerShapePlan attn_shape;
    bn_transformer_plan_layer_shape(&attn_shape, &c, &lw, 0, 0);
    assert(!bn_transformer_attention_requires_cpu_fallback(
        &attn_shape, BN_EXEC_GPU));
    assert(bn_transformer_attention_uses_flash(&c, &gpu));
    assert(bn_transformer_attention_uses_packed_qkv(
        &gpu, &attn_shape, &lw, (void *)1, (void *)2, (void *)3, (void *)4));
    assert(bn_transformer_attention_uses_qkv_split(
        &gpu, &attn_shape, &lw, (void *)1));
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
    lw.ffn.ffn_gate.rows = 8192;
    lw.ffn.ffn_up.rows = 8192;
    lw.ffn.ffn_gate.cols = 2048;
    lw.ffn.ffn_up.cols = 2048;
    assert(bn_backend_model_register_handle(backend, 0,
                                            BN_BACKEND_HANDLE_GATEUP_STACKED,
                                            (void *)5) == 0);
    lw.norm.ffn_sub_norm = (float *)1;

    bn_transformer_plan_ffn(&ffn, &c, &lw, &gpu, backend, 0, 1);
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
    bn_transformer_plan_ssm(&ssm, &c, &lw, 1, 1, &gpu, backend);
    assert(ssm.placement == BN_EXEC_GPU);
    assert(ssm.backend == BN_BACKEND_METAL);
    assert(ssm.ssm_idx == -1);
    assert(ssm.state_size == 128);
    assert(ssm.conv_kernel == 4);
    assert(ssm.inner_size == 4096);
    assert(ssm.time_step_rank == 32);
    assert(ssm.group_count == 16);
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
    assert(bn_transformer_moe_has_shared_expert(&c, &lw));
    c.has_shared_expert = 0;
    assert(!bn_transformer_moe_has_shared_expert(&c, &lw));
    lw.shared.shared_expert_gate = (float *)1;
    assert(bn_transformer_moe_has_shared_expert(&c, &lw));
    lw.shared.shared_expert_gate = NULL;
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
    assert(moe.shared_hidden_dim == 2048);
    assert(moe.needs_cpu_fallback);

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
    assert(bn_transformer_logits_kind(&w) == BN_LOGITS_TIED_F32);
    assert(bn_transformer_logits_weight_type(&w) ==
           bn_transformer_logits_tied_f32_weight_type());
    assert(logits.kind == BN_LOGITS_TIED_F32);
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
    memset(&c, 0, sizeof(c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK |
                     BN_MODEL_ARCH_POLICY_PREFILL_EXACT_ACTIVATION;
    assert(bn_transformer_cpu_float_kquant_task_flags(0) == 0);
    assert(bn_transformer_cpu_float_kquant_task_flags(1) ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);
    assert(bn_transformer_cpu_force_float_kquant_task_flags(&c) ==
           BN_MATVEC_TASK_FORCE_FLOAT_KQUANT);
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
    c.per_layer_input_dim = 0;
    c.n_experts = 4;
    c.n_layers = 30;
    assert(!bn_transformer_per_layer_embedding_dim(&c));
    assert(!bn_transformer_divides_rope_freqs(&c, 0));
    assert(bn_transformer_divides_rope_freqs(&c, 5));
    assert(!bn_transformer_cpu_uses_scalar_hybrid_ssm(&c));
    assert(bn_transformer_prefill_uses_exact_activation(&c));
    assert(!bn_transformer_rmsnorm_requires_reference_scalar_order(&c));
    int force_float_kquant =
        bn_transformer_cpu_prefill_force_float_kquant_enabled(&c);
    int backend_supports_float_kquant_prefill =
        bn_transformer_cpu_backend_supports_float_kquant_prefill();
    assert(backend_supports_float_kquant_prefill ==
           bn_transformer_cpu_backend_ops()->supports_float_kquant_prefill);
    assert(force_float_kquant ==
           backend_supports_float_kquant_prefill);
    c.policy_flags = 0;
    assert(bn_transformer_cpu_force_float_kquant_task_flags(&c) == 0);
    assert(!bn_transformer_cpu_prefill_force_float_kquant_enabled(&c));
    assert(!bn_transformer_rmsnorm_requires_reference_scalar_order(&c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY;
    assert(bn_transformer_cpu_prefill_decode_for_parity_enabled(&c, 0));
    assert(!bn_transformer_cpu_prefill_decode_for_parity_enabled(&c, 1));
    c.policy_flags = 0;
    assert(!bn_transformer_cpu_prefill_decode_for_parity_enabled(&c, 0));
    c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER;
    assert(bn_transformer_rmsnorm_requires_reference_scalar_order(&c));

    unsetenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS");
    assert(bn_transformer_cpu_prepared_qweights_enabled());
    setenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS", "1", 1);
    assert(!bn_transformer_cpu_prepared_qweights_enabled());
    unsetenv("BN_CPU_DISABLE_PREPARED_QWEIGHTS");

    unsetenv("BN_DUMP_LAYER_INP");
    unsetenv("BN_DUMP_LAYER_POS");
    unsetenv("BN_DUMP_ALL_HEADS");
    assert(bn_transformer_cpu_debug_dump_path() == NULL);
    assert(bn_transformer_cpu_debug_dump_pos_selected(3));
    assert(!bn_transformer_cpu_debug_dump_heads_enabled());
    setenv("BN_DUMP_LAYER_INP", "/tmp/bitnet-dump.txt", 1);
    assert(bn_transformer_cpu_debug_dump_path() != NULL);
    setenv("BN_DUMP_LAYER_POS", "7", 1);
    assert(bn_transformer_cpu_debug_dump_pos_selected(7));
    assert(!bn_transformer_cpu_debug_dump_pos_selected(6));
    setenv("BN_DUMP_ALL_HEADS", "1", 1);
    assert(bn_transformer_cpu_debug_dump_heads_enabled());
    unsetenv("BN_DUMP_LAYER_INP");
    unsetenv("BN_DUMP_LAYER_POS");
    unsetenv("BN_DUMP_ALL_HEADS");

    unsetenv("BN_CPU_LLAMA_DOT");
    unsetenv("BN_CPU_LLAMA_Q4_DOT");
    unsetenv("BN_CPU_REFERENCE_DOT");
    unsetenv("BN_CPU_REFERENCE_Q4_DOT");
    assert(bn_transformer_cpu_fused_kquant_gateup_silu_allowed());
    assert(bn_transformer_cpu_can_fused_kquant_gateup_silu(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_transformer_cpu_can_fused_kquant_gateup_silu(
        BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0));
    setenv("BN_CPU_REFERENCE_DOT", "1", 1);
    assert(!bn_transformer_cpu_fused_kquant_gateup_silu_allowed());
    assert(!bn_transformer_cpu_can_fused_kquant_gateup_silu(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    unsetenv("BN_CPU_REFERENCE_DOT");
    setenv("BN_CPU_REFERENCE_Q4_DOT", "1", 1);
    assert(!bn_transformer_cpu_fused_kquant_gateup_silu_allowed());
    assert(!bn_transformer_cpu_can_fused_kquant_gateup_silu(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    unsetenv("BN_CPU_REFERENCE_Q4_DOT");
    setenv("BN_CPU_LLAMA_DOT", "1", 1);
    assert(!bn_transformer_cpu_fused_kquant_gateup_silu_allowed());
    assert(!bn_transformer_cpu_can_fused_kquant_gateup_silu(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    unsetenv("BN_CPU_LLAMA_DOT");
    setenv("BN_CPU_LLAMA_Q4_DOT", "1", 1);
    assert(!bn_transformer_cpu_fused_kquant_gateup_silu_allowed());
    assert(!bn_transformer_cpu_can_fused_kquant_gateup_silu(
        BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    unsetenv("BN_CPU_LLAMA_Q4_DOT");

    assert(!bn_transformer_cpu_can_prepared_kquant_pair(
        NULL, BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K));
    int supports_prepared_kquant = bn_transformer_cpu_backend_ops()->supports_prepared_kquant;
    assert(bn_transformer_cpu_can_prepared_kquant_pair(
               bn_transformer_cpu_backend_ops(),
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K) ==
           supports_prepared_kquant);
    assert(bn_transformer_cpu_can_prepared_kquant_triple(
               bn_transformer_cpu_backend_ops(),
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K,
               BN_GGUF_TENSOR_Q6_K) == supports_prepared_kquant);
    assert(!bn_transformer_cpu_can_prepared_kquant_pair(
        bn_transformer_cpu_backend_ops(), BN_GGUF_TENSOR_Q4_K,
        BN_GGUF_TENSOR_Q8_0));
    assert(!bn_transformer_cpu_can_prepared_kquant_triple(
        bn_transformer_cpu_backend_ops(), BN_GGUF_TENSOR_Q4_K,
        BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0));
    BnGPUBackend route_gpu = {0};
    assert(bn_transformer_cpu_route_prepared_kquant_pair_enabled(
               bn_transformer_cpu_backend_ops(), NULL, BN_QK_K,
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K) ==
           supports_prepared_kquant);
    assert(!bn_transformer_cpu_route_prepared_kquant_pair_enabled(
        bn_transformer_cpu_backend_ops(), &route_gpu, BN_QK_K,
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_transformer_cpu_route_prepared_kquant_pair_enabled(
        bn_transformer_cpu_backend_ops(), NULL, BN_QK_K - 1,
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K));
    assert(bn_transformer_cpu_route_prepared_kquant_triple_enabled(
               bn_transformer_cpu_backend_ops(), NULL, BN_QK_K,
               BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K,
               BN_GGUF_TENSOR_Q6_K) == supports_prepared_kquant);
    assert(!bn_transformer_cpu_route_prepared_kquant_triple_enabled(
        bn_transformer_cpu_backend_ops(), &route_gpu, BN_QK_K,
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K));
    assert(!bn_transformer_cpu_route_prepared_kquant_triple_enabled(
        bn_transformer_cpu_backend_ops(), NULL, BN_QK_K,
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q8_0));

    BnFFNPlan ffn_plan = {0};
    ffn_plan.activation = 0;
    assert(bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        NULL, &ffn_plan, 32, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        &route_gpu, &ffn_plan, 32, BN_GGUF_TENSOR_Q4_0,
        BN_GGUF_TENSOR_Q4_0));
    ffn_plan.scalar_exact_activation = 1;
    assert(!bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        NULL, &ffn_plan, 32, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    ffn_plan.scalar_exact_activation = 0;
    ffn_plan.activation = 1;
    assert(!bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        NULL, &ffn_plan, 32, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    ffn_plan.activation = 0;
    assert(!bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        NULL, &ffn_plan, 31, BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_transformer_cpu_route_fused_kquant_gateup_silu_enabled(
        NULL, &ffn_plan, 32, BN_GGUF_TENSOR_Q8_0, BN_GGUF_TENSOR_Q8_0));

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
            &ssm_upload_config, 1);
    assert(!ssm_upload.upload);
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    ssm_upload = bn_transformer_prefill_ssm_state_upload_policy(
        &ssm_upload_config, 1);
    assert(ssm_upload.upload);
    ssm_upload = bn_transformer_prefill_ssm_state_upload_policy(
        &ssm_upload_config, 0);
    assert(!ssm_upload.upload);
    ssm_upload_config.full_attn_interval = 0;
    ssm_upload_config.ssm_inner_size = 0;
    ssm_upload = bn_transformer_prefill_ssm_state_upload_policy(
        &ssm_upload_config, 1);
    assert(!ssm_upload.upload);
    ssm_upload = bn_transformer_prefill_ssm_state_upload_policy(NULL, 1);
    assert(!ssm_upload.upload);
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");

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

    const BnCPUBackendOps *cpu_ops = bn_transformer_cpu_backend_ops();
    c.policy_flags = 0;
    assert(bn_transformer_cpu_ssm_conv_silu_op(&c, cpu_ops) ==
           cpu_ops->ssm_conv_silu);
    assert(bn_transformer_cpu_ssm_l2norm_op(&c, cpu_ops) ==
           cpu_ops->ssm_l2norm);
    assert(bn_transformer_cpu_ssm_delta_op(&c, cpu_ops) ==
           cpu_ops->ssm_delta);
    assert(bn_transformer_cpu_ssm_gate_op(&c, cpu_ops) ==
           cpu_ops->ssm_gate);
    c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM;
    c.full_attn_interval = 4;
    assert(bn_transformer_cpu_ssm_conv_silu_op(&c, cpu_ops) ==
           bn_transformer_ssm_conv_silu_scalar_range);
    assert(bn_transformer_cpu_ssm_l2norm_op(&c, cpu_ops) ==
           bn_transformer_ssm_l2norm_scalar_range);
    assert(bn_transformer_cpu_ssm_delta_op(&c, cpu_ops) ==
           bn_transformer_ssm_delta_scalar_range);
    assert(bn_transformer_cpu_ssm_gate_op(&c, cpu_ops) ==
           bn_transformer_ssm_gate_scalar_range);
    c.policy_flags = 0;
    c.full_attn_interval = 0;

    assert(!bn_transformer_ffn_uses_exact_scalar_activation(&c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_EXACT_SCALAR_FFN_ACTIVATION;
    assert(bn_transformer_ffn_uses_exact_scalar_activation(&c));
    c.policy_flags = 0;

    BnTransformerPrefillLayerKindPolicy prefill_layer_kind =
        bn_transformer_prefill_layer_kind_policy(NULL);
    assert(!prefill_layer_kind.uses_moe);
    prefill_layer_kind =
        bn_transformer_prefill_layer_kind_policy((void *)1);
    assert(prefill_layer_kind.uses_moe);

    prefill_layer_kind =
        bn_transformer_prefill_layer_kind_policy(NULL);

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
    assert(shared_all_active_two_fallback.enabled);
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
    assert(!bn_transformer_prefill_uses_hybrid_ssm(NULL));

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

    sequence.n_experts = 1;
    sequence_policy = bn_transformer_prefill_sequence_policy(&sequence);
    assert(sequence_policy.uses_hybrid_ssm);
    assert(!sequence_policy.uses_large_dense_hybrid_ssm);

    BnTransformerPrefillSequencePolicy decode_sequence = {0};
    BnTransformerPrefillDecodeFallbackPolicy decode_fallback =
        bn_transformer_prefill_decode_fallback_policy(
            decode_sequence, 1, 0, 16, 1, 0, 1, 0, 0, 1);
    assert(decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 1, 1, 8, 16, 0, 1, 0, 0, 1);
    assert(decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 0, 1, 8, 1, 1, 16, 0, 0, 1);
    assert(decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_sequence.uses_large_dense_hybrid_ssm = 1;
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 0, 1, 16, 1, 0, 1, 1, 1, 1);
    assert(decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_sequence.uses_hybrid_ssm = 1;
    decode_sequence.uses_large_dense_hybrid_ssm = 0;
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 0, 1, 16, 1, 0, 1, 0, 0, 0);
    assert(decode_fallback.decode);
    assert(decode_fallback.require_logits_decode);
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 0, 1, 16, 1, 0, 1, 1, 0, 0);
    assert(!decode_fallback.decode);
    assert(!decode_fallback.require_logits_decode);
    decode_sequence.uses_hybrid_ssm = 0;
    decode_fallback = bn_transformer_prefill_decode_fallback_policy(
        decode_sequence, 0, 1, 16, 1, 0, 1, 0, 0, 1);
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
        bn_transformer_prefill_hybrid_model_chain_policy(1, 1, 0, 1, 0);
    assert(hybrid_model_chain.enabled);
    hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(0, 1, 0, 1, 0);
    assert(!hybrid_model_chain.enabled);
    hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(1, 0, 0, 1, 0);
    assert(!hybrid_model_chain.enabled);
    hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(1, 1, 1, 1, 0);
    assert(!hybrid_model_chain.enabled);
    hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(1, 1, 0, 0, 0);
    assert(!hybrid_model_chain.enabled);
    hybrid_model_chain =
        bn_transformer_prefill_hybrid_model_chain_policy(1, 1, 0, 1, 1);
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
        bn_transformer_prefill_layer_kind_policy((void *)1),
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    assert(!dense_layer_batch.enabled);
    dense_layer_batch = bn_transformer_prefill_dense_layer_batch_policy(
        1, 0, 1, 16, 16, 0, 10000.0f, 10000.0f,
        prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0);
    assert(!dense_layer_batch.enabled);

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
        bn_transformer_prefill_layer_kind_policy((void *)1),
        1, 1, 0, 0, 0, 0, 0, 0);
    assert(!dense_layer_chain.enabled);
    dense_layer_chain = bn_transformer_prefill_dense_layer_chain_policy(
        1, 1, 0, 16, 16, 10000.0f, 10000.0f, 1,
        prefill_layer_kind, 1, 1, 1, 0, 0, 0, 0, 0);
    assert(!dense_layer_chain.enabled);
    dense_layer_chain = bn_transformer_prefill_dense_layer_chain_policy(
        1, 1, 0, 16, 16, 10000.0f, 10000.0f, 1,
        prefill_layer_kind, 1, 1, 0, 0, 0, 1, 1, 0);
    assert(!dense_layer_chain.enabled);

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

    BnConfig prefill_dense_c = {0};
    BnGPUBackend prefill_dense_gpu = {0};
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, NULL) == 16);
    prefill_dense_gpu.kind = BN_GPU_BACKEND_CUDA;
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 16);
    setenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", "9", 1);
    assert(bn_transformer_prefill_dense_chain_min_tokens(
        &prefill_dense_c, &prefill_dense_gpu) == 9);
    unsetenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS");
    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");
    assert(bn_transformer_prefill_dense_chain_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN", "1", 1);
    assert(!bn_transformer_prefill_dense_chain_enabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN");
    assert(!bn_transformer_prefill_dense_ffn_batch_tokens_allowed(
        &prefill_dense_gpu, &prefill_dense_c, 15));
    assert(bn_transformer_prefill_dense_ffn_batch_tokens_allowed(
        &prefill_dense_gpu, &prefill_dense_c, 16));
    prefill_dense_gpu.kind = BN_GPU_BACKEND_METAL;
    assert(bn_transformer_prefill_dense_ffn_batch_tokens_allowed(
        &prefill_dense_gpu, &prefill_dense_c, 1));

    BnTransformerPrefillSSMChainPolicy ssm_chain =
        bn_transformer_prefill_ssm_chain_policy(
            1, prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 16, 64, 256, 4);
    assert(ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        0, prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 16, 64, 256, 4);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, bn_transformer_prefill_layer_kind_policy((void *)1),
        1, 1, 0, 0, 0, 0, 0, 16, 64, 256, 4);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, prefill_layer_kind, 0, 1, 0, 0, 0, 0, 0, 16, 64, 256, 4);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, prefill_layer_kind, 1, 1, 1, 0, 0, 0, 0, 16, 64, 256, 4);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, prefill_layer_kind, 1, 1, 0, 0, 1, 1, 0, 16, 64, 256, 4);
    assert(!ssm_chain.enabled);
    ssm_chain = bn_transformer_prefill_ssm_chain_policy(
        1, prefill_layer_kind, 1, 1, 0, 0, 0, 0, 0, 0, 64, 256, 4);
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
    assert(bn_transformer_prefill_ssm_run_chain_enabled());
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN", "1", 1);
    assert(!bn_transformer_prefill_ssm_run_chain_enabled());
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN");
    unsetenv("BN_CUDA_DISABLE_SSM_FFN_FUSE");
    assert(bn_transformer_prefill_ssm_ffn_fuse_allowed());
    setenv("BN_CUDA_DISABLE_SSM_FFN_FUSE", "1", 1);
    assert(!bn_transformer_prefill_ssm_ffn_fuse_allowed());
    unsetenv("BN_CUDA_DISABLE_SSM_FFN_FUSE");
    memset(&prefill_dense_gpu, 0, sizeof(prefill_dense_gpu));

    BnTransformerPrefillSSMMoEChainPolicy ssm_moe_chain =
        bn_transformer_prefill_ssm_moe_chain_policy(
            1, bn_transformer_prefill_layer_kind_policy((void *)1),
            0, 0, 0, 0, 0, 16, 64, 256, 4);
    assert(ssm_moe_chain.enabled);
    ssm_moe_chain = bn_transformer_prefill_ssm_moe_chain_policy(
        1, prefill_layer_kind, 0, 0, 0, 0, 0, 16, 64, 256, 4);
    assert(!ssm_moe_chain.enabled);
    ssm_moe_chain = bn_transformer_prefill_ssm_moe_chain_policy(
        1, bn_transformer_prefill_layer_kind_policy((void *)1),
        0, 1, 0, 0, 0, 16, 64, 256, 4);
    assert(!ssm_moe_chain.enabled);
    ssm_moe_chain = bn_transformer_prefill_ssm_moe_chain_policy(
        1, bn_transformer_prefill_layer_kind_policy((void *)1),
        0, 0, 1, 0, 1, 16, 64, 256, 4);
    assert(!ssm_moe_chain.enabled);
    ssm_moe_chain = bn_transformer_prefill_ssm_moe_chain_policy(
        1, bn_transformer_prefill_layer_kind_policy((void *)1),
        0, 0, 0, 0, 0, 16, 0, 256, 4);
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
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");
    assert(!bn_transformer_prefill_moe_ffn_batch_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0));
    setenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN", "1", 1);
    assert(bn_transformer_prefill_moe_ffn_batch_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0));
    prefill_gpu.prefill_ssm_layer = mock_prefill_ssm_layer;
    assert(!bn_transformer_prefill_ssm_moe_chain_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0, 15));
    assert(bn_transformer_prefill_ssm_moe_chain_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0, 16));
    prefill_gpu.prefill_moe_layer = mock_prefill_moe_layer;
    assert(!bn_transformer_prefill_moe_layer_chain_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0, 15));
    assert(bn_transformer_prefill_moe_layer_chain_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0, 16));
    prefill_gpu.prefill_moe_layer = NULL;
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    assert(bn_transformer_prefill_moe_chain_min_tokens(
        &prefill_c, &prefill_gpu) == 16);
    setenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS", "7", 1);
    assert(bn_transformer_prefill_moe_chain_min_tokens(
        &prefill_c, &prefill_gpu) == 7);
    unsetenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS");
    unsetenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");
    assert(!bn_transformer_prefill_moe_chain_debug_enabled());
    setenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN", "1", 1);
    assert(bn_transformer_prefill_moe_chain_debug_enabled());
    unsetenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN");
    setenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER", "1", 1);
    assert(!bn_transformer_prefill_ssm_moe_chain_available(
        &prefill_gpu, &prefill_c, &prefill_map, prefill_c.dim, 0, 16));
    unsetenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER");
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
    unsetenv("BN_CUDA_ENABLE_ALL2_Q4Q6_MOE_FAST_FFN");

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
    assert(bn_transformer_prefill_same_quant_format_pair_stackable(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_transformer_prefill_same_quant_format_pair_stackable(
        BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K));

    const BnPrefillCPUOps *prefill_ops = bn_transformer_prefill_cpu_ops();
    c.policy_flags = 0;
    assert(bn_transformer_prefill_ssm_conv_silu_op(&c, prefill_ops) ==
           prefill_ops->ssm_conv_silu);
    assert(bn_transformer_prefill_ssm_l2norm_op(&c, prefill_ops) ==
           prefill_ops->ssm_l2norm);
    assert(bn_transformer_prefill_ssm_delta_op(&c, prefill_ops) ==
           prefill_ops->ssm_delta);
    assert(bn_transformer_prefill_ssm_gate_op(&c, prefill_ops) ==
           prefill_ops->ssm_gate);
    c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM;
    c.full_attn_interval = 4;
    assert(bn_transformer_prefill_ssm_conv_silu_op(&c, prefill_ops) ==
           bn_transformer_ssm_conv_silu_scalar_range);
    assert(bn_transformer_prefill_ssm_l2norm_op(&c, prefill_ops) ==
           bn_transformer_ssm_l2norm_scalar_range);
    assert(bn_transformer_prefill_ssm_delta_op(&c, prefill_ops) ==
           bn_transformer_ssm_delta_scalar_range);
    assert(bn_transformer_prefill_ssm_gate_op(&c, prefill_ops) ==
           bn_transformer_ssm_gate_scalar_range);
    c.policy_flags = 0;
    c.full_attn_interval = 0;

    unsetenv("BN_PREFILL_PROFILE");
    assert(!bn_transformer_prefill_profile_enabled());
    setenv("BN_PREFILL_PROFILE", "1", 1);
    assert(bn_transformer_prefill_profile_enabled());
    unsetenv("BN_PREFILL_PROFILE");

    unsetenv("BN_PREFILL_ALLOW_HYBRID_BATCH");
    assert(!bn_transformer_prefill_hybrid_batch_allowed());
    setenv("BN_PREFILL_ALLOW_HYBRID_BATCH", "1", 1);
    assert(bn_transformer_prefill_hybrid_batch_allowed());
    unsetenv("BN_PREFILL_ALLOW_HYBRID_BATCH");

    unsetenv("BN_PREFILL_FORCE_TOKEN_ATTN");
    assert(!bn_transformer_prefill_force_token_attention_enabled());
    setenv("BN_PREFILL_FORCE_TOKEN_ATTN", "1", 1);
    assert(bn_transformer_prefill_force_token_attention_enabled());
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
    assert(!bn_transformer_prefill_uses_exact_activation(&c));
    c.policy_flags = BN_MODEL_ARCH_POLICY_REFERENCE_HYBRID_SSM;
    c.per_layer_input_dim = 0;
    c.full_attn_interval = 4;
    assert(bn_transformer_cpu_uses_scalar_hybrid_ssm(&c));
#if defined(__AVX512F__) && !defined(BN_FORCE_SCALAR)
    assert(cpu_backend == BN_CPU_BACKEND_AVX512);
#endif

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

int main(void) {
    printf("=== Transformer Tests ===\n");
    test_rmsnorm();
    test_rmsnorm_scalar_matches_avx2_order();
    test_softmax();
    test_rope();
    test_fp16_embed();
    test_fast_silu();
    test_cpu_execution_helpers();
    test_gpu_capability_routing();
    test_gpu_policy_helpers();
    test_logits_policy_helpers();
    test_gpu_op_kind_mapping();
    test_model_arch_registry();
    test_layer_shape_planning();
    test_block_planning();
    test_batched_attn_fp16_kv();
    printf("All transformer tests passed!\n");
    return 0;
}
