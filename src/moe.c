#include "moe.h"
#include "platform.h"
#include "quant.h"
#include "sh_log.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef __EMSCRIPTEN__
#include <unistd.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
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

// --- Phase 4: Vectorized router ---

typedef struct {
    float *logits;
    const float *router_w;
    const float *x;
    int dim;
} BnRouterCtx;

static void moe_router_range(void *ctx, int start, int end) {
    BnRouterCtx *c = (BnRouterCtx *)ctx;
    for (int e = start; e < end; e++) {
        const float *row = c->router_w + (size_t)e * c->dim;
        float sum = 0.0f;
#ifdef __ARM_NEON
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        int d = 0;
        for (; d + 15 < c->dim; d += 16) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(row + d),      vld1q_f32(c->x + d));
            acc1 = vfmaq_f32(acc1, vld1q_f32(row + d + 4),  vld1q_f32(c->x + d + 4));
            acc2 = vfmaq_f32(acc2, vld1q_f32(row + d + 8),  vld1q_f32(c->x + d + 8));
            acc3 = vfmaq_f32(acc3, vld1q_f32(row + d + 12), vld1q_f32(c->x + d + 12));
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        sum = vaddvq_f32(acc0);
        for (; d < c->dim; d++)
            sum += row[d] * c->x[d];
#elif defined(__AVX2__)
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        int d = 0;
        for (; d + 15 < c->dim; d += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + d),     _mm256_loadu_ps(c->x + d),     a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + d + 8), _mm256_loadu_ps(c->x + d + 8), a1);
        }
        a0 = _mm256_add_ps(a0, a1);
        __m128 hi = _mm256_extractf128_ps(a0, 1);
        __m128 lo = _mm256_castps256_ps128(a0);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum = _mm_cvtss_f32(lo);
        for (; d < c->dim; d++)
            sum += row[d] * c->x[d];
#else
        for (int d = 0; d < c->dim; d++)
            sum += row[d] * c->x[d];
#endif
        c->logits[e] = sum;
    }
}

// Router: SIMD matvec -> softmax -> top-K selection
void bn_moe_route(BnMoEState *ms, const float *x, const float *router_w,
                  int dim, int n_experts, int k, BnThreadPool *pool) {
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

    size_t proj_bytes;
    switch (proj) {
        case 0: proj_bytes = map->expert_gate_bytes; break;
        case 1: proj_bytes = map->expert_up_bytes; break;
        case 2: proj_bytes = map->expert_down_bytes; break;
        default: return NULL;
    }

    ms->io_bytes += proj_bytes;
    ms->io_count++;

    // Use mmap pointer if available (fast path, avoids pread copy)
    if (ms->mmap_base) {
        return ms->mmap_base + offset;
    }

#if !defined(__EMSCRIPTEN__)
    // Fallback: pread into scratch buffer
    if (ms->fd < 0) return NULL;
    if (proj_bytes > ms->expert_buf_size) return NULL;
    double t0 = bn_platform_time_ms();
    ssize_t n = pread(ms->fd, ms->expert_buf, proj_bytes, (off_t)offset);
    ms->io_time_ms += bn_platform_time_ms() - t0;
    if (n != (ssize_t)proj_bytes) return NULL;
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

// --- Phase 3: SwiGLU range function for parallel dispatch ---

typedef struct {
    float *hb;
    const float *gate;
    const float *up;
} BnSwiGLUCtx;

static void moe_swiglu_range(void *ctx, int start, int end) {
    BnSwiGLUCtx *c = (BnSwiGLUCtx *)ctx;
    for (int i = start; i < end; i++) {
        float g = c->gate[i];
        c->hb[i] = (g / (1.0f + expf(-g))) * c->up[i];
    }
}

// Scalar SwiGLU for pread path (single expert, no dispatch overhead)
static void moe_swiglu(float *hb, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        hb[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// --- Phase 2: Cross-expert batched dispatch context ---
// Layout-compatible with all float-x quant ctx types { out, W, x }
typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnFloatCtx;

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

    // 2. Route: select top-K experts (SIMD + threaded)
    double t_route = bn_platform_time_ms();
    bn_moe_route(ms, s->xb, lw->router_weight, dim, c->n_experts, K, m->pool);
    ms->route_time_ms += bn_platform_time_ms() - t_route;

    // 3. Zero output accumulator
    memset(ms->expert_out, 0, dim * sizeof(float));

    // 4. Expert FFN compute
    double t_compute = bn_platform_time_ms();

    if (ms->mmap_base && K <= BN_MAX_MOE_K) {
        // --- Cross-expert batched dispatch (mmap path) ---
        // Collect all K gate+up projections, dispatch as 2K batch
        int valid_k = 0;
        int valid_indices[BN_MAX_MOE_K];
        float valid_weights[BN_MAX_MOE_K];
        BnQWeight wgates[BN_MAX_MOE_K], wups[BN_MAX_MOE_K];

        for (int k = 0; k < K; k++) {
            int eidx = ms->expert_indices[k];
            if (eidx < 0) continue;
            const void *gate_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 0);
            const void *up_data   = moe_load_expert_proj(ms, &lw->expert_map, eidx, 1);
            if (!gate_data || !up_data) {
                SH_LOG_ERROR("Failed to load expert gate/up projection");
                continue;
            }
            wgates[valid_k] = moe_make_qweight(gate_data, lw->expert_map.gate_type,
                                                lw->expert_map.gate_rows, lw->expert_map.gate_cols);
            wups[valid_k]   = moe_make_qweight(up_data, lw->expert_map.up_type,
                                                lw->expert_map.up_rows, lw->expert_map.up_cols);
            valid_indices[valid_k] = eidx;
            valid_weights[valid_k] = ms->expert_weights[k];
            valid_k++;
        }

        if (valid_k > 0) {
            // Dispatch all 2K gate+up matvecs in one batch
            BnMatvecTask gu_tasks[2 * BN_MAX_MOE_K];
            for (int k = 0; k < valid_k; k++) {
                gu_tasks[2*k]     = (BnMatvecTask){ ms->expert_hb_batch[k],  &wgates[k] };
                gu_tasks[2*k + 1] = (BnMatvecTask){ ms->expert_hb2_batch[k], &wups[k]   };
            }
            bn_quant_matvec_batch(gu_tasks, 2 * valid_k, s->xb, s->x_q, m->pool);

            // Parallel SwiGLU across K experts
            BnSwiGLUCtx swiglu_ctxs[BN_MAX_MOE_K];
            BnTPTask swiglu_tasks[BN_MAX_MOE_K];
            for (int k = 0; k < valid_k; k++) {
                swiglu_ctxs[k] = (BnSwiGLUCtx){
                    ms->expert_hb_batch[k],
                    ms->expert_hb_batch[k],
                    ms->expert_hb2_batch[k]
                };
                swiglu_tasks[k] = (BnTPTask){ moe_swiglu_range, &swiglu_ctxs[k], moe_hidden };
            }
            bn_tp_dispatch(m->pool, swiglu_tasks, valid_k);

            // Down projections: each expert has different x, dispatch directly
            BnQWeight wdowns[BN_MAX_MOE_K];
            for (int k = 0; k < valid_k; k++) {
                const void *down_data = moe_load_expert_proj(ms, &lw->expert_map, valid_indices[k], 2);
                if (!down_data) {
                    SH_LOG_ERROR("Failed to load expert down projection");
                    valid_weights[k] = 0.0f;
                    continue;
                }
                wdowns[k] = moe_make_qweight(down_data, lw->expert_map.down_type,
                                              lw->expert_map.down_rows, lw->expert_map.down_cols);
            }

            // Try batched dispatch for down projections using per-expert ctx
            bn_tp_fn down_kernel = bn_quant_get_float_kernel(lw->expert_map.down_type);
            if (down_kernel) {
                BnFloatCtx down_ctxs[BN_MAX_MOE_K];
                BnTPTask down_tasks[BN_MAX_MOE_K];
                int n_down = 0;
                for (int k = 0; k < valid_k; k++) {
                    if (valid_weights[k] == 0.0f) continue;
                    down_ctxs[n_down] = (BnFloatCtx){
                        ms->expert_down_batch[k],
                        &wdowns[k],
                        ms->expert_hb_batch[k]
                    };
                    down_tasks[n_down] = (BnTPTask){ down_kernel, &down_ctxs[n_down], wdowns[k].rows };
                    n_down++;
                }
                if (n_down > 0)
                    bn_tp_dispatch(m->pool, down_tasks, n_down);
            } else {
                // Fallback for int8-quantized down projections
                for (int k = 0; k < valid_k; k++) {
                    if (valid_weights[k] == 0.0f) continue;
                    bn_quant_matvec(ms->expert_down_batch[k], &wdowns[k],
                                    ms->expert_hb_batch[k], s->x_q, m->pool);
                }
            }

            // Weighted accumulation
            for (int k = 0; k < valid_k; k++) {
                float w = valid_weights[k];
                if (w == 0.0f) continue;
                for (int d = 0; d < dim; d++)
                    ms->expert_out[d] += w * ms->expert_down_batch[k][d];
            }
        }
    } else {
        // --- Serial path (pread or K > BN_MAX_MOE_K fallback) ---
        for (int k = 0; k < K; k++) {
            int eidx = ms->expert_indices[k];
            float weight = ms->expert_weights[k];
            if (eidx < 0) continue;

            if (ms->mmap_base) {
                // mmap but K > BN_MAX_MOE_K: per-expert batch dispatch
                const void *gate_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 0);
                const void *up_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 1);
                if (!gate_data || !up_data) {
                    SH_LOG_ERROR("Failed to load expert gate/up projection");
                    continue;
                }
                BnQWeight wgate = moe_make_qweight(gate_data, lw->expert_map.gate_type,
                                                    lw->expert_map.gate_rows, lw->expert_map.gate_cols);
                BnQWeight wup = moe_make_qweight(up_data, lw->expert_map.up_type,
                                                  lw->expert_map.up_rows, lw->expert_map.up_cols);
                BnMatvecTask gu[2] = {
                    { ms->expert_hb,  &wgate },
                    { ms->expert_hb2, &wup   },
                };
                bn_quant_matvec_batch(gu, 2, s->xb, s->x_q, m->pool);
            } else {
                // pread: single buffer — load and compute sequentially
                const void *gate_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 0);
                if (!gate_data) { SH_LOG_ERROR("Failed to load expert gate"); continue; }
                BnQWeight wgate = moe_make_qweight(gate_data, lw->expert_map.gate_type,
                                                    lw->expert_map.gate_rows, lw->expert_map.gate_cols);
                bn_quant_matvec(ms->expert_hb, &wgate, s->xb, s->x_q, m->pool);

                const void *up_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 1);
                if (!up_data) { SH_LOG_ERROR("Failed to load expert up"); continue; }
                BnQWeight wup = moe_make_qweight(up_data, lw->expert_map.up_type,
                                                  lw->expert_map.up_rows, lw->expert_map.up_cols);
                bn_quant_matvec(ms->expert_hb2, &wup, s->xb, s->x_q, m->pool);
            }

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
            bn_quant_matvec(s->xb2, &wdown, ms->expert_hb, s->x_q, m->pool);

            // Weighted accumulation
            for (int d = 0; d < dim; d++)
                ms->expert_out[d] += weight * s->xb2[d];
        }
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

    ms->compute_time_ms += bn_platform_time_ms() - t_compute;

    // 6. Copy result to xb for residual add by caller
    memcpy(s->xb, ms->expert_out, dim * sizeof(float));

    // 7. Residual add
    for (int d = 0; d < dim; d++)
        s->x[d] += s->xb[d];
}

void bn_moe_print_stats(const BnMoEState *ms, int n_tokens) {
    if (!ms || n_tokens <= 0) return;

    double io_gb = (double)ms->io_bytes / (1024.0 * 1024.0 * 1024.0);
    double io_per_tok = (double)ms->io_bytes / (1024.0 * 1024.0) / n_tokens;

    char io_s[32], iot_s[32], bw_s[32], rt_s[32], ct_s[32], rss_s[32];
    snprintf(io_s, sizeof(io_s), "%.2f", io_gb);
    snprintf(iot_s, sizeof(iot_s), "%.1f", io_per_tok);

    // Streaming bandwidth: bytes loaded / io_time (pread only; mmap io_time is 0)
    if (ms->io_time_ms > 0.1)
        snprintf(bw_s, sizeof(bw_s), "%.0f",
                 (double)ms->io_bytes / (1024.0 * 1024.0) / (ms->io_time_ms / 1000.0));
    else
        snprintf(bw_s, sizeof(bw_s), "mmap");

    snprintf(rt_s, sizeof(rt_s), "%.1f", ms->route_time_ms);
    snprintf(ct_s, sizeof(ct_s), "%.1f", ms->compute_time_ms);

    size_t rss = bn_platform_rss_bytes();
    snprintf(rss_s, sizeof(rss_s), "%.2f", (double)rss / (1024.0 * 1024.0 * 1024.0));

    SH_LOG_INFO("MoE stats",
                "expert_io_GB", io_s,
                "MB/tok", iot_s,
                "stream_MB/s", bw_s,
                "route_ms", rt_s,
                "compute_ms", ct_s,
                "rss_GB", rss_s);
}

void bn_moe_reset_stats(BnMoEState *ms) {
    if (!ms) return;
    ms->io_bytes = 0;
    ms->io_time_ms = 0;
    ms->route_time_ms = 0;
    ms->compute_time_ms = 0;
    ms->io_count = 0;
}
