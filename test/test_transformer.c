#include "transformer_internal.h"
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
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(theta, (float)head_dim / (float)head_size);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * cos_a - v1 * sin_a;
        vec[i + 1] = v0 * sin_a + v1 * cos_a;
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

static void test_gpu_capability_routing(void) {
    printf("test_gpu_capability_routing... ");

    BnGPUBackend gpu;
    memset(&gpu, 0, sizeof(gpu));

    assert(!bn_transformer_gpu_has_cap(NULL, BN_GPU_CAP_FLASH_ATTN));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q4_0));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q8_0));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q5_K));

    gpu.caps = BN_GPU_CAP_Q4_MATVEC_SPLIT |
               BN_GPU_CAP_Q8_MATVEC_SPLIT |
               BN_GPU_CAP_Q5K_MATVEC_SPLIT |
               BN_GPU_CAP_Q4_FUSED_GATEUP_SILU |
               BN_GPU_CAP_FLASH_ATTN;
    gpu.kind = BN_GPU_BACKEND_METAL;

    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q4_0));
    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q8_0));
    assert(bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q5_K));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_Q4_K));
    assert(!bn_transformer_gpu_can_matvec_split(&gpu, BN_GGUF_TENSOR_F16));

    assert(bn_transformer_gpu_can_fused_gateup_silu(&gpu, BN_GGUF_TENSOR_Q4_0, 0));
    assert(!bn_transformer_gpu_can_fused_gateup_silu(&gpu, BN_GGUF_TENSOR_Q4_0, 1));
    assert(!bn_transformer_gpu_can_fused_gateup_silu(&gpu, BN_GGUF_TENSOR_Q8_0, 0));
    assert(bn_transformer_gpu_can_flash_attn(&gpu));

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
    lw.wq.data = (void *)1;
    lw.wq.rows = 2048;
    lw.head_size = 0;
    lw.kv_dim = 0;
    lw.n_kv_heads = 0;
    lw.kv_mul = 0;
    lw.q_norm = (float *)1;
    lw.k_bias = (float *)1;

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

    lw.wq.rows = 4096;
    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 0);
    assert(p.kind == BN_LAYER_ATTN_GATED_Q);
    assert(p.q_gated);
    assert(!p.q_wide);

    lw.wq.rows = 3072;
    lw.head_size = 192;
    lw.kv_dim = 768;
    lw.n_kv_heads = 4;
    lw.kv_mul = 4;
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
    bn_transformer_plan_layer_shape(&p, &c, &lw, 0, 0);
    assert(!p.is_attn);
    assert(p.kind == BN_LAYER_SSM);
    assert(p.attn_idx == -1);
    assert(p.ssm_idx == 0);
    assert(p.kv_mode == BN_KV_FP16);

    bn_transformer_plan_layer_shape(&p, &c, &lw, 3, 0);
    assert(p.is_attn);
    assert(p.attn_idx == 0);
    assert(p.ssm_idx == -1);

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
               BN_GPU_CAP_Q8_MATVEC_SPLIT |
               BN_GPU_CAP_Q5K_MATVEC_SPLIT |
               BN_GPU_CAP_Q4_FUSED_GATEUP_SILU |
               BN_GPU_CAP_FLASH_ATTN;
    gpu.kind = BN_GPU_BACKEND_METAL;

    lw.wq.data = (void *)1;
    lw.wq.rows = 2048;
    lw.wq.type = BN_GGUF_TENSOR_Q4_0;
    lw.wk.type = BN_GGUF_TENSOR_Q4_0;
    lw.wv.type = BN_GGUF_TENSOR_Q4_0;
    lw.qkv_stacked_gpu = (void *)1;
    lw.q_bias_gpu = (void *)1;
    lw.k_bias_gpu = (void *)1;
    lw.v_bias_gpu = (void *)1;

    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, 0, 0, 1);
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

    lw.k_bias_gpu = NULL;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, 0, 0, 1);
    assert(attn.fusion_flags & BN_FUSION_ROPE_QK);
    lw.k_bias_gpu = (void *)1;

    lw.wq.type = BN_GGUF_TENSOR_Q8_0;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, 0, 0, 1);
    assert(attn.use_q8_qkv_split);
    assert(attn.fusion_flags & BN_FUSION_Q8_QKV_SPLIT);

    lw.wq.type = BN_GGUF_TENSOR_Q5_K;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, 0, 0, 1);
    assert(attn.use_q5_qkv_split);
    assert(attn.fusion_flags & BN_FUSION_Q5_QKV_SPLIT);

    lw.wq.type = BN_GGUF_TENSOR_Q4_0;
    lw.wq.rows = 4096;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, 0, 0, 1);
    assert(attn.shape.kind == BN_LAYER_ATTN_GATED_Q);
    assert(!attn.use_packed_qkv);
    assert(!attn.use_qkv_split);
    assert(!(attn.fusion_flags & BN_FUSION_QKV_SPLIT));

    c.full_attn_interval = 4;
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, 0, 0, 1);
    assert(attn.placement == BN_EXEC_CPU_FALLBACK);
    assert(attn.backend == BN_BACKEND_CPU);
    assert(attn.needs_cpu_fallback);
    c.full_attn_interval = 0;
    lw.wq.rows = 2048;

    lw.ffn_gate.type = BN_GGUF_TENSOR_Q4_0;
    lw.ffn_up.type = BN_GGUF_TENSOR_Q4_0;
    lw.ffn_gate.rows = 8192;
    lw.ffn_up.rows = 8192;
    lw.ffn_gate.cols = 2048;
    lw.ffn_up.cols = 2048;
    lw.gateup_stacked_gpu = (void *)1;
    lw.ffn_sub_norm = (float *)1;

    bn_transformer_plan_ffn(&ffn, &c, &lw, &gpu, 0, 1);
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

    lw.router_weight = (float *)1;
    bn_transformer_plan_ffn(&ffn, &c, &lw, &gpu, 0, 1);
    assert(ffn.kind == BN_FFN_MOE);
    assert(ffn.placement == BN_EXEC_CPU_FALLBACK);
    assert(ffn.backend == BN_BACKEND_CPU);
    assert(ffn.needs_cpu_fallback);
    assert(ffn.fusion_flags == BN_FUSION_NONE);

    lw.ssm_qkvz_stacked_gpu = (void *)1;
    lw.ssm_ab_stacked_gpu = (void *)1;
    bn_transformer_plan_ssm(&ssm, &c, &lw, 1, 1, &gpu);
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
    bn_transformer_plan_attention(&attn, &c, &lw, &gpu, 0, 0, 1);
    assert(attn.backend == BN_BACKEND_WEBGPU);

    gpu.kind = BN_GPU_BACKEND_CUDA;
    bn_transformer_plan_logits(&logits, &c, &w, &gpu, 1);
    assert(logits.backend == BN_BACKEND_CUDA);

    printf("PASSED\n");
}

int main(void) {
    printf("=== Transformer Tests ===\n");
    test_rmsnorm();
    test_softmax();
    test_rope();
    test_fp16_embed();
    test_fast_silu();
    test_gpu_capability_routing();
    test_layer_shape_planning();
    test_block_planning();
    printf("All transformer tests passed!\n");
    return 0;
}
