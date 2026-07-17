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

void bn_moe_swiglu_range(void *ctx, int start, int end) {
    BnSwiGLUCtx *c = (BnSwiGLUCtx *)ctx;
    int i = start;
    if (c->exact_silu < 0) {
        for (; i < end; i++) {
            float g = c->gate[i];
            float gelu = 0.5f * g *
                         (1.0f + tanhf(0.7978845608028654f * g *
                                       (1.0f + 0.044715f * g * g)));
            c->hb[i] = gelu * c->up[i];
        }
        return;
    }
    bn_moe_swiglu_silu(c->hb + i, c->gate + i, c->up + i, end - i,
                       c->exact_silu);
}

// Vectorized SwiGLU for pread path (single expert, no dispatch overhead)
void bn_moe_swiglu(float *hb, const float *gate, const float *up, int n,
                   int exact_silu) {
    int i = 0;
    if (exact_silu < 0) {
        for (; i < n; i++) {
            float g = gate[i];
            float gelu = 0.5f * g *
                         (1.0f + tanhf(0.7978845608028654f * g *
                                       (1.0f + 0.044715f * g * g)));
            hb[i] = gelu * up[i];
        }
        return;
    }
    bn_moe_swiglu_silu(hb + i, gate + i, up + i, n - i, exact_silu);
}

// Compiler barrier to prevent reordering of timing calls around dispatches
double bn_moe_time_ms(void) {
    double t = bn_platform_time_ms();
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" ::: "memory");
#endif
    return t;
}
