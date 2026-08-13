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
    float rounded_x = bn_fp16_to_fp32(bn_fp32_to_fp16(x));
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
    float gate_dot = 0.0f;
    for (int d = 0; d < dim; d++)
        gate_dot += x[d] * gate_vector[d];
    return 1.0f / (1.0f + expf(-gate_dot));
}
