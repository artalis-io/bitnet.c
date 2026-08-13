#include "moe_internal.h"

// --- Phase 4: Vectorized router ---

typedef struct {
    float *logits;
    const float *router_w;
    const float *x;
    int dim;
    int uses_reference_router_accumulation;
} BnRouterCtx;

static void moe_router_range(void *ctx, int start, int end) {
    BnRouterCtx *c = (BnRouterCtx *)ctx;
    for (; start + 3 < end; start += 4) {
        if (!bn_moe_dot4_rows(c->logits + start, c->router_w, c->x,
                              c->dim, start))
            break;
    }
    for (int e = start; e < end; e++) {
        const float *row = c->router_w + (size_t)e * c->dim;
        c->logits[e] = c->uses_reference_router_accumulation
            ? bn_moe_dot_row_reference(row, c->x, c->dim)
            : bn_moe_dot_row(row, c->x, c->dim);
    }
}

// Router: SIMD matvec -> softmax -> top-K selection
void bn_moe_route(BnMoEState *ms, const float *x, const float *router_w,
                  int dim, int n_experts, int k, int norm_topk_prob,
                  float expert_weights_scale,
                  int uses_reference_router_accumulation,
                  BnThreadPool *pool) {
    // Router matvec: vectorized + thread-dispatched
    BnRouterCtx rctx = {
        ms->router_logits, router_w, x, dim,
        uses_reference_router_accumulation
    };
    BnTPTask rtask = { moe_router_range, &rctx, n_experts };
    bn_tp_dispatch(pool, &rtask, 1);

    // Softmax denominator over all experts. Keep raw logits intact so routing
    // diagnostics and downstream observers can inspect the actual scores.
    float max_val = ms->router_logits[0];
    for (int e = 1; e < n_experts; e++)
        if (ms->router_logits[e] > max_val)
            max_val = ms->router_logits[e];

    float probs[n_experts];
    double sum = bn_moe_softmax_exp(probs, ms->router_logits,
                                    n_experts, max_val);

    // Top-K selection over raw logits. Softmax is monotonic, and retaining the
    // scores avoids rewriting the full router output just to mark selections.
    for (int i = 0; i < k; i++) {
        int best = -1;
        float best_val = -INFINITY;
        for (int e = 0; e < n_experts; e++) {
            int already_selected = 0;
            for (int j = 0; j < i; j++) {
                if (ms->expert_indices[j] == e) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected)
                continue;
            if (ms->router_logits[e] > best_val) {
                best_val = ms->router_logits[e];
                best = e;
            }
        }
        ms->expert_indices[i] = best;
        ms->expert_weights[i] = (float)((double)probs[best] / sum);
    }

    if (norm_topk_prob) {
        float wsum = 0.0f;
        for (int i = 0; i < k; i++)
            wsum += ms->expert_weights[i];
        if (wsum > 0.0f) {
            for (int i = 0; i < k; i++)
                ms->expert_weights[i] /= wsum;
        }
    }
    if (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) {
        for (int i = 0; i < k; i++)
            ms->expert_weights[i] *= expert_weights_scale;
    }
}
