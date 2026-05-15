#include "transformer_cpu_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "../src/transformer/gpu_internal.h"
#include "../src/gpu_shader.h"
#include "transformer_plan_internal.h"
#include "../src/gpu_quant_lowering_internal.h"
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
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q4_K));

    gpu.caps = BN_GPU_CAP_Q4_MATVEC_SPLIT |
               BN_GPU_CAP_Q4K_MATVEC_SPLIT |
               BN_GPU_CAP_Q8_MATVEC_SPLIT |
               BN_GPU_CAP_Q5K_MATVEC_SPLIT |
               BN_GPU_CAP_Q4_FUSED_GATEUP_SILU |
               BN_GPU_CAP_FLASH_ATTN;
    gpu.kind = BN_GPU_BACKEND_METAL;

    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q4_0));
    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q8_0));
    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q5_K));
    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_F16));

    assert(bn_transformer_gpu_can_fused_gateup_silu(&gpu, BN_GGUF_TENSOR_Q4_0, 0));
    assert(!bn_transformer_gpu_can_fused_gateup_silu(&gpu, BN_GGUF_TENSOR_Q4_0, 1));
    assert(!bn_transformer_gpu_can_fused_gateup_silu(&gpu, BN_GGUF_TENSOR_Q8_0, 0));
    assert(bn_transformer_gpu_can_flash_attn(&gpu));

    printf("PASSED\n");
}

static void test_gpu_policy_helpers(void) {
    printf("test_gpu_policy_helpers... ");

    BnConfig c;
    memset(&c, 0, sizeof(c));
    c.n_layers = 4;
    assert(bn_transformer_gpu_graph_op_capacity(&c) >
           80 * c.n_layers);

    BnGPUBackend gpu;
    BnTransformerGPULogitResources logits;
    BnQWeight W;
    memset(&gpu, 0, sizeof(gpu));
    memset(&logits, 0, sizeof(logits));
    memset(&W, 0, sizeof(W));

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
    assert(bn_gpu_quant_split_op_code(BN_GGUF_TENSOR_Q4_0) ==
           BN_GPU_CODE_MATVEC_SPLIT);
    assert(bn_gpu_quant_split_op_code(BN_GGUF_TENSOR_Q8_0) ==
           BN_GPU_CODE_Q8_MATVEC_SPLIT);
    assert(bn_gpu_quant_split_op_code(BN_GGUF_TENSOR_Q5_K) ==
           BN_GPU_CODE_Q5K_MATVEC_SPLIT);
    assert(bn_gpu_quant_split_op_code(BN_GGUF_TENSOR_Q4_K) ==
           BN_GPU_CODE_Q4K_MATVEC_SPLIT);
    assert(bn_gpu_quant_split_op_code(BN_GGUF_TENSOR_I2_S) == 0);

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
               BN_GPU_VALUE_HB, 64, 64, 32, 0) == 0);
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
    assert(gemma->flags & BN_MODEL_ARCH_FLAG_GEMMA4);
    assert(gemma->flags & BN_MODEL_ARCH_FLAG_LARGE_GPU_GRAPH_FALLBACK);
    assert(strcmp(gemma->prefix("gemma4"), "gemma4") == 0);
    assert(gemma->attention_value_shares_key("gemma4"));
    assert(gemma->activation("gemma4") == 2);

    BnConfig c = {0};
    c.arch_flags = gemma->flags;
    assert(bn_model_arch_requires_large_gpu_graph_fallback(&c));

    const BnModelArchOps *bitnet = bn_model_arch_ops_for("bitnet");
    assert(bitnet);
    assert(strcmp(bitnet->name, "bitnet") == 0);
    assert(bitnet->flags & BN_MODEL_ARCH_FLAG_BITNET);
    assert(strcmp(bitnet->prefix("bitnet"), "bitnet") == 0);
    assert(bitnet->activation("bitnet") == 1);
    assert(!bitnet->attention_value_shares_key("bitnet"));

    const BnModelArchOps *qwen = bn_model_arch_ops_for("qwen35");
    assert(qwen);
    assert(strcmp(qwen->name, "qwen") == 0);
    assert(qwen->flags & BN_MODEL_ARCH_FLAG_QWEN);
    assert(strcmp(qwen->prefix("qwen35"), "qwen35") == 0);
    assert(qwen->activation("qwen35") == 0);
    assert(!qwen->attention_value_shares_key("qwen35"));

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

    lw.attn.wq.rows = 4096;
    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 0);
    assert(p.kind == BN_LAYER_ATTN_GATED_Q);
    assert(p.q_gated);
    assert(!p.q_wide);

    lw.attn.wq.rows = 3072;
    lw.attn.head_size = 192;
    lw.attn.kv_dim = 768;
    lw.attn.n_kv_heads = 4;
    lw.attn.kv_mul = 4;
    c.kv_tq_bits = 3;
    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 1);
    assert(p.kind == BN_LAYER_ATTN_WIDE_Q);
    assert(!p.q_gated);
    assert(p.q_wide);
    assert(p.q_dim == 3072);
    assert(p.head_size == 192);
    assert(p.kv_dim == 768);
    assert(p.kv_mode == BN_KV_TQ);

    c.full_attn_interval = 4;
    c.kv_tq_bits = 0;
    c.kv_f16 = 1;
    lw.block_kind = BN_LAYER_BLOCK_SSM;
    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 0);
    assert(!p.is_attn);
    assert(p.kind == BN_LAYER_SSM);
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
    assert(ffn.placement == BN_EXEC_GPU);
    assert(ffn.backend == BN_BACKEND_METAL);
    assert(ffn.hidden_dim == 8192);
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
    assert(logits.kind == BN_LOGITS_TIED_I8);
    assert(logits.placement == BN_EXEC_GPU);
    assert(logits.backend == BN_BACKEND_METAL);
    assert(logits.vocab_size == 32000);
    assert(logits.dim == 2048);
    assert(logits.use_i8_output);

    w.emb_out_i8 = NULL;
    w.output_weight.data = (void *)1;
    w.output_weight.type = BN_GGUF_TENSOR_Q4_K;
    bn_transformer_plan_logits(&logits, &c, &w, NULL, 1);
    assert(logits.kind == BN_LOGITS_UNTIED);
    assert(logits.placement == BN_EXEC_CPU);
    assert(logits.backend == BN_BACKEND_CPU);
    assert(logits.weight_type == BN_GGUF_TENSOR_Q4_K);

    gpu.kind = BN_GPU_BACKEND_WEBGPU;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, NULL, 0, 0, 1);
    assert(attn.backend == BN_BACKEND_WEBGPU);

    gpu.kind = BN_GPU_BACKEND_CUDA;
    bn_transformer_plan_logits(&logits, &c, &w, &gpu, 1);
    assert(logits.backend == BN_BACKEND_CUDA);

    BnCPUBackendPlacement cpu_backend = bn_transformer_cpu_backend_placement();
    assert(cpu_backend == BN_CPU_BACKEND_SCALAR ||
           cpu_backend == BN_CPU_BACKEND_NEON ||
           cpu_backend == BN_CPU_BACKEND_AVX2 ||
           cpu_backend == BN_CPU_BACKEND_AVX512 ||
           cpu_backend == BN_CPU_BACKEND_WASM_SIMD);
#if defined(__AVX512F__) && !defined(BN_FORCE_SCALAR)
    assert(cpu_backend == BN_CPU_BACKEND_AVX512);
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
    test_gpu_op_kind_mapping();
    test_model_arch_registry();
    test_layer_shape_planning();
    test_block_planning();
    printf("All transformer tests passed!\n");
    return 0;
}
