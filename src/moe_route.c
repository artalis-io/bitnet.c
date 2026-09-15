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
    if (!c->uses_reference_router_accumulation)
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

void bn_moe_route_logits(const float *router_logits,
                          int *expert_indices,
                          float *expert_weights,
                          int n_experts,
                          int k,
                          int norm_topk_prob,
                          float expert_weights_scale) {
    // Softmax denominator over all experts. Keep raw logits intact so routing
    // diagnostics and downstream observers can inspect the actual scores.
    float max_val = router_logits[0];
    for (int e = 1; e < n_experts; e++)
        if (router_logits[e] > max_val)
            max_val = router_logits[e];

    float probs[n_experts];
    double sum = bn_moe_softmax_exp(probs, router_logits,
                                    n_experts, max_val);
    float inv_sum = (float)(1.0 / sum);
    for (int e = 0; e < n_experts; e++)
        probs[e] *= inv_sum;

    // Top-K selection over raw logits. Softmax is monotonic, and retaining the
    // scores avoids rewriting the full router output just to mark selections.
    for (int i = 0; i < k; i++) {
        int best = -1;
        float best_val = -INFINITY;
        for (int e = 0; e < n_experts; e++) {
            int already_selected = 0;
            for (int j = 0; j < i; j++) {
                if (expert_indices[j] == e) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected)
                continue;
            if (router_logits[e] > best_val) {
                best_val = router_logits[e];
                best = e;
            }
        }
        expert_indices[i] = best;
        expert_weights[i] = probs[best];
    }

    if (norm_topk_prob) {
        double wsum64 = 0.0;
        for (int i = 0; i < k; i++)
            wsum64 += (double)expert_weights[i];
        float wsum = (float)wsum64;
        if (wsum > 0.0f) {
            for (int i = 0; i < k; i++)
                expert_weights[i] /= wsum;
        }
    }
    if (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) {
        for (int i = 0; i < k; i++)
            expert_weights[i] *= expert_weights_scale;
    }
}

void bn_moe_route_buffers(float *router_logits, int *expert_indices,
                          float *expert_weights, const float *x,
                          const float *router_w, int dim, int n_experts,
                          int k, int norm_topk_prob,
                          float expert_weights_scale,
                          int uses_reference_router_accumulation,
                          BnThreadPool *pool) {
    BnRouterCtx rctx = {router_logits, router_w, x, dim,
                       uses_reference_router_accumulation};
    BnTPTask task = {moe_router_range, &rctx, n_experts};
    bn_tp_dispatch(pool, &task, 1);
    bn_moe_route_logits(router_logits, expert_indices, expert_weights,
                        n_experts, k, norm_topk_prob, expert_weights_scale);
}

// Router: SIMD matvec -> softmax -> top-K selection
void bn_moe_route(BnMoEState *ms, const float *x, const float *router_w,
                  int dim, int n_experts, int k, int norm_topk_prob,
                  float expert_weights_scale,
                  int uses_reference_router_accumulation,
                  BnThreadPool *pool) {
    bn_moe_route_buffers(ms->router_logits, ms->expert_indices,
                         ms->expert_weights, x, router_w, dim, n_experts, k,
                         norm_topk_prob, expert_weights_scale,
                         uses_reference_router_accumulation, pool);
}
