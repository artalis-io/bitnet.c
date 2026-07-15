#include "moe_internal.h"

// --- Phase 4: Vectorized router ---

typedef struct {
    float *logits;
    const float *router_w;
    const float *x;
    int dim;
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
        c->logits[e] = bn_moe_dot_row(row, c->x, c->dim);
    }
}

// Router: SIMD matvec -> softmax -> top-K selection
void bn_moe_route(BnMoEState *ms, const float *x, const float *router_w,
                  int dim, int n_experts, int k, int norm_topk_prob,
                  float expert_weights_scale, BnThreadPool *pool) {
    // Router matvec: vectorized + thread-dispatched
    BnRouterCtx rctx = { ms->router_logits, router_w, x, dim };
    BnTPTask rtask = { moe_router_range, &rctx, n_experts };
    bn_tp_dispatch(pool, &rtask, 1);

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
    // Top-K selection (partial sort). Select over exp(logit - max);
    // the all-expert softmax denominator is common and cancels after
    // the selected weights are normalized below.
    for (int i = 0; i < k; i++) {
        int best = -1;
        float best_val = -1.0f;
        for (int e = 0; e < n_experts; e++) {
            if (ms->router_logits[e] > best_val) {
                best_val = ms->router_logits[e];
                best = e;
            }
        }
        ms->expert_indices[i] = best;
        ms->expert_weights[i] = best_val / sum;
        ms->router_logits[best] = -1.0f;
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
