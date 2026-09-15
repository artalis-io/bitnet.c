#include "moe_internal.h"

// Build a temporary BnQWeight from pread'd expert data
BnQWeight bn_moe_make_qweight(const void *data, int type, int rows, int cols) {
    BnQWeight w = {0};
    w.data = data;
    w.type = type;
    w.rows = rows;
    w.cols = cols;
    if (bn_moe_quant_uses_embedded_tensor_scale(type)) {
        const uint8_t *base = (const uint8_t *)data;
        memcpy(&w.scale,
               base + bn_moe_quant_embedded_tensor_scale_offset(type, rows,
                                                                cols),
               sizeof(float));
    } else {
        w.scale = 1.0f;
    }
    return w;
}

BnQWeight bn_moe_make_f32_weight(const float *data, int rows, int cols) {
    return bn_quant_f32_weight(data, rows, cols);
}

// --- Phase 3: SwiGLU range function for parallel dispatch ---

static float moe_gelu_tanh(float x) {
    if (x <= -10.0f)
        return 0.0f;
    if (x >= 10.0f)
        return x;
    float inner = 0.7978845608028654f * x *
                  (1.0f + 0.044715f * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static float moe_reference_gelu(float x) {
    // The reference uses its FP16 table only inside these FP32 limits.
    if (x <= -10.0f)
        return 0.0f;
    if (x >= 10.0f)
        return x;
    uint16_t rounded_bits = bn_fp32_to_fp16(x);
    // Preserve the reference table entry at this FP16 rounding boundary,
    // matching the dense CPU activation policy.
    if (rounded_bits == 0xbfffu)
        return bn_fp16_to_fp32(0xa9d3u);
    float rounded_x = bn_fp16_to_fp32(rounded_bits);
    float gelu = moe_gelu_tanh(rounded_x);
    return bn_fp16_to_fp32(bn_fp32_to_fp16(gelu));
}

void bn_moe_swiglu_range(void *ctx, int start, int end) {
    BnSwiGLUCtx *c = (BnSwiGLUCtx *)ctx;
    int i = start;
    if (c->uses_reference_silu < 0) {
        for (; i < end; i++) {
            float g = c->gate[i];
            c->hb[i] = (c->uses_reference_ffn_activation
                            ? moe_reference_gelu(g)
                            : moe_gelu_tanh(g)) * c->up[i];
        }
        return;
    }
    bn_moe_swiglu_silu(c->hb + i, c->gate + i, c->up + i, end - i,
                       c->uses_reference_silu);
}

// Vectorized SwiGLU for pread path (single expert, no dispatch overhead)
void bn_moe_swiglu(float *hb, const float *gate, const float *up, int n,
                   int uses_reference_silu,
                   int uses_reference_ffn_activation) {
    int i = 0;
    if (uses_reference_silu < 0) {
        for (; i < n; i++) {
            float g = gate[i];
            hb[i] = (uses_reference_ffn_activation
                         ? moe_reference_gelu(g)
                         : moe_gelu_tanh(g)) * up[i];
        }
        return;
    }
    bn_moe_swiglu_silu(hb + i, gate + i, up + i, n - i, uses_reference_silu);
}

// Compiler barrier to prevent reordering of timing calls around dispatches
double bn_moe_time_ms(void) {
    double t = bn_platform_time_ms();
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" ::: "memory");
#endif
    return t;
}

float bn_moe_shared_expert_gate_weight(const BnLayerWeights *lw,
                                       const float *x,
                                       int dim) {
    const float *gate_vector = bn_moe_shared_expert_gate_vector(lw);
    if (!gate_vector || !x || dim <= 0)
        return 1.0f;
    float gate_dot = bn_moe_dot_row(gate_vector, x, dim);
    return 1.0f / (1.0f + expf(-gate_dot));
}

void bn_moe_scaled_router_input(float *out, const float *x,
                                 const float *scale, int size, float eps) {
    double sum = 0.0;
    for (int i = 0; i < size; i++)
        sum += (double)(x[i] * x[i]);
    float mean = (float)(sum / size);
    float inv_rms = 1.0f / sqrtf(mean + eps);
    float inv_sqrt_dim = 1.0f / sqrtf((float)size);
    for (int i = 0; i < size; i++)
        out[i] = ((x[i] * inv_rms) * inv_sqrt_dim) *
                 (scale ? scale[i] : 1.0f);
}

void bn_moe_scale_expert_output(float *out, float scale, int size) {
    for (int i = 0; i < size; i++)
        out[i] *= scale;
}
