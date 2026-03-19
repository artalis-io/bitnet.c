#include "moe.h"
#include "quant.h"
#include "sh_log.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef __EMSCRIPTEN__
#include <unistd.h>
#endif

// Backend-selected rmsnorm (same selection as transformer.c)
#ifdef __ARM_NEON
extern void bn_transformer_rmsnorm_neon(float *out, const float *x, const float *w, int size, float eps);
#define moe_rmsnorm bn_transformer_rmsnorm_neon
#elif defined(__AVX2__)
extern void bn_transformer_rmsnorm_avx2(float *out, const float *x, const float *w, int size, float eps);
#define moe_rmsnorm bn_transformer_rmsnorm_avx2
#elif defined(__wasm_simd128__)
extern void bn_transformer_rmsnorm_wasm(float *out, const float *x, const float *w, int size, float eps);
#define moe_rmsnorm bn_transformer_rmsnorm_wasm
#else
extern void bn_transformer_rmsnorm_scalar(float *out, const float *x, const float *w, int size, float eps);
#define moe_rmsnorm bn_transformer_rmsnorm_scalar
#endif

// Router: matvec -> softmax -> top-K selection
void bn_moe_route(BnMoEState *ms, const float *x, const float *router_w,
                  int dim, int n_experts, int k) {
    // Router matvec: router_logits = router_weight @ x
    // router_weight is [n_experts, dim] stored row-major
    for (int e = 0; e < n_experts; e++) {
        float sum = 0.0f;
        const float *row = router_w + (size_t)e * dim;
        for (int d = 0; d < dim; d++)
            sum += row[d] * x[d];
        ms->router_logits[e] = sum;
    }

    // Softmax over all experts
    float max_val = ms->router_logits[0];
    for (int e = 1; e < n_experts; e++)
        if (ms->router_logits[e] > max_val)
            max_val = ms->router_logits[e];

    float sum = 0.0f;
    for (int e = 0; e < n_experts; e++) {
        ms->router_logits[e] = expf(ms->router_logits[e] - max_val);
        sum += ms->router_logits[e];
    }
    for (int e = 0; e < n_experts; e++)
        ms->router_logits[e] /= sum;

    // Top-K selection (partial sort)
    for (int i = 0; i < k; i++) {
        int best = -1;
        float best_val = -1.0f;
        for (int e = 0; e < n_experts; e++) {
            // Skip already-selected experts
            int skip = 0;
            for (int j = 0; j < i; j++)
                if (ms->expert_indices[j] == e) { skip = 1; break; }
            if (skip) continue;
            if (ms->router_logits[e] > best_val) {
                best_val = ms->router_logits[e];
                best = e;
            }
        }
        ms->expert_indices[i] = best;
        ms->expert_weights[i] = best_val;
    }

    // Normalize selected weights to sum to 1.0
    float wsum = 0.0f;
    for (int i = 0; i < k; i++)
        wsum += ms->expert_weights[i];
    if (wsum > 0.0f) {
        for (int i = 0; i < k; i++)
            ms->expert_weights[i] /= wsum;
    }
}

// Load one expert projection from GGUF file via pread.
// proj: 0=gate, 1=up, 2=down
// Returns pointer to data in expert_buf, or NULL on error.
static const void *moe_load_expert_proj(BnMoEState *ms, const BnMoEExpertMap *map,
                                         int expert_idx, int proj) {
    size_t offset;
    switch (proj) {
        case 0:
            offset = map->gate_offset + (size_t)expert_idx * map->expert_gate_bytes;
            break;
        case 1:
            offset = map->up_offset + (size_t)expert_idx * map->expert_up_bytes;
            break;
        case 2:
            offset = map->down_offset + (size_t)expert_idx * map->expert_down_bytes;
            break;
        default:
            return NULL;
    }

    // Use mmap pointer if available (fast path, avoids pread copy)
    if (ms->mmap_base) {
        return ms->mmap_base + offset;
    }

#if !defined(__EMSCRIPTEN__)
    // Fallback: pread into scratch buffer
    if (ms->fd < 0) return NULL;
    size_t size;
    switch (proj) {
        case 0: size = map->expert_gate_bytes; break;
        case 1: size = map->expert_up_bytes; break;
        case 2: size = map->expert_down_bytes; break;
        default: return NULL;
    }
    if (size > ms->expert_buf_size) return NULL;
    ssize_t n = pread(ms->fd, ms->expert_buf, size, (off_t)offset);
    if (n != (ssize_t)size) return NULL;
    return ms->expert_buf;
#else
    return NULL;
#endif
}

// Build a temporary BnQWeight from pread'd expert data
static BnQWeight moe_make_qweight(const void *data, int type, int rows, int cols) {
    BnQWeight w = {0};
    w.data = data;
    w.type = type;
    w.rows = rows;
    w.cols = cols;
    // Per-block quants have embedded scales
    if (type == BN_GGUF_TENSOR_I2_S) {
        size_t nelements = (size_t)rows * cols;
        const uint8_t *base = (const uint8_t *)data;
        memcpy(&w.scale, base + nelements / 4, sizeof(float));
    } else {
        w.scale = 1.0f;
    }
    return w;
}

// SwiGLU activation: hb = SiLU(gate) * up
static void moe_swiglu(float *hb, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        hb[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// Full MoE FFN block
void bn_moe_forward(BnModel *m, BnLayerWeights *lw, int l) {
    (void)l;
    BnConfig *c = &m->config;
    BnRunState *s = &m->state;
    BnMoEState *ms = m->moe_state;
    int dim = c->dim;
    int moe_hidden = c->moe_intermediate_size;
    int K = c->n_experts_active;

    // 1. RMSNorm input
    moe_rmsnorm(s->xb, s->x, lw->ffn_norm, dim, c->norm_eps);

    // 2. Route: select top-K experts
    bn_moe_route(ms, s->xb, lw->router_weight, dim, c->n_experts, K);

    // 3. Zero output accumulator
    memset(ms->expert_out, 0, dim * sizeof(float));

    // 4. For each selected expert: load projections, compute SwiGLU FFN
    for (int k = 0; k < K; k++) {
        int eidx = ms->expert_indices[k];
        float weight = ms->expert_weights[k];
        if (eidx < 0) continue;

        // Load gate projection and compute gate_proj @ xb
        const void *gate_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 0);
        if (!gate_data) {
            SH_LOG_ERROR("Failed to load expert gate projection");
            continue;
        }
        BnQWeight wgate = moe_make_qweight(gate_data, lw->expert_map.gate_type,
                                            lw->expert_map.gate_rows, lw->expert_map.gate_cols);
        bn_quant_matvec(ms->expert_hb, &wgate, s->xb, s->x_q, NULL);

        // Load up projection and compute up_proj @ xb
        const void *up_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 1);
        if (!up_data) {
            SH_LOG_ERROR("Failed to load expert up projection");
            continue;
        }
        BnQWeight wup = moe_make_qweight(up_data, lw->expert_map.up_type,
                                          lw->expert_map.up_rows, lw->expert_map.up_cols);
        bn_quant_matvec(ms->expert_hb2, &wup, s->xb, s->x_q, NULL);

        // SwiGLU activation
        moe_swiglu(ms->expert_hb, ms->expert_hb, ms->expert_hb2, moe_hidden);

        // Load down projection and compute down_proj @ hb -> xb2
        const void *down_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 2);
        if (!down_data) {
            SH_LOG_ERROR("Failed to load expert down projection");
            continue;
        }
        BnQWeight wdown = moe_make_qweight(down_data, lw->expert_map.down_type,
                                            lw->expert_map.down_rows, lw->expert_map.down_cols);
        bn_quant_matvec(s->xb2, &wdown, ms->expert_hb, s->x_q, NULL);

        // Weighted accumulation
        for (int d = 0; d < dim; d++)
            ms->expert_out[d] += weight * s->xb2[d];
    }

    // 5. Shared expert (if present, always resident)
    if (c->has_shared_expert && lw->shared_gate.data) {
        int shared_hidden = c->shared_expert_intermediate_size;

        // gate_proj @ xb
        bn_quant_matvec(s->hb, &lw->shared_gate, s->xb, s->x_q, m->pool);
        // up_proj @ xb
        bn_quant_matvec(s->hb2, &lw->shared_up, s->xb, s->x_q, m->pool);

        // SwiGLU
        moe_swiglu(s->hb, s->hb, s->hb2, shared_hidden);

        // down_proj @ hb
        bn_quant_matvec(s->xb2, &lw->shared_down, s->hb, s->x_q, m->pool);

        // Add shared expert output to accumulated output
        for (int d = 0; d < dim; d++)
            ms->expert_out[d] += s->xb2[d];
    }

    // 6. Copy result to xb for residual add by caller
    memcpy(s->xb, ms->expert_out, dim * sizeof(float));

    // 7. Residual add
    for (int d = 0; d < dim; d++)
        s->x[d] += s->xb[d];
}
