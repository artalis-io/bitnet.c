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
#include <pthread.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

// --- I/O Prefetch Thread (pread pipeline) ---
#if !defined(__EMSCRIPTEN__)

typedef struct {
    pthread_t thread;
    pthread_mutex_t mtx;
    pthread_cond_t req_cv;      // I/O thread waits for work
    pthread_cond_t done_cv;     // main thread waits for completion
    int fd;
    int active;                 // request in progress
    int shutdown;
    int success;                // result of last I/O
    double io_time_ms;          // accumulated pread time
    size_t io_bytes;            // accumulated bytes read
    // Request: up to 2 preads per submission
    struct { uint8_t *buf; size_t size; off_t offset; } reqs[2];
    int n_reqs;
} BnMoEPrefetch;

static void *moe_prefetch_worker(void *arg) {
    BnMoEPrefetch *pf = (BnMoEPrefetch *)arg;
    pthread_mutex_lock(&pf->mtx);
    while (1) {
        while (!pf->active && !pf->shutdown)
            pthread_cond_wait(&pf->req_cv, &pf->mtx);
        if (pf->shutdown) break;

        // Copy request params under lock
        int n = pf->n_reqs;
        uint8_t *bufs[2]; size_t sizes[2]; off_t offsets[2];
        for (int i = 0; i < n; i++) {
            bufs[i] = pf->reqs[i].buf;
            sizes[i] = pf->reqs[i].size;
            offsets[i] = pf->reqs[i].offset;
        }
        pthread_mutex_unlock(&pf->mtx);

        // Do I/O without holding lock
        double t0 = bn_platform_time_ms();
        int ok = 1;
        size_t bytes = 0;
        for (int i = 0; i < n; i++) {
            ssize_t r = pread(pf->fd, bufs[i], sizes[i], offsets[i]);
            if (r != (ssize_t)sizes[i]) { ok = 0; break; }
            bytes += sizes[i];
        }
        double elapsed = bn_platform_time_ms() - t0;

        pthread_mutex_lock(&pf->mtx);
        pf->success = ok;
        pf->io_time_ms += elapsed;
        pf->io_bytes += bytes;
        pf->active = 0;
        pthread_cond_signal(&pf->done_cv);
    }
    pthread_mutex_unlock(&pf->mtx);
    return NULL;
}

static BnMoEPrefetch *moe_prefetch_init(int fd) {
    BnMoEPrefetch *pf = (BnMoEPrefetch *)calloc(1, sizeof(BnMoEPrefetch));
    if (!pf) return NULL;
    pf->fd = fd;
    pthread_mutex_init(&pf->mtx, NULL);
    pthread_cond_init(&pf->req_cv, NULL);
    pthread_cond_init(&pf->done_cv, NULL);
    if (pthread_create(&pf->thread, NULL, moe_prefetch_worker, pf) != 0) {
        pthread_mutex_destroy(&pf->mtx);
        pthread_cond_destroy(&pf->req_cv);
        pthread_cond_destroy(&pf->done_cv);
        free(pf);
        return NULL;
    }
    return pf;
}

static void moe_prefetch_free(BnMoEPrefetch *pf) {
    if (!pf) return;
    pthread_mutex_lock(&pf->mtx);
    pf->shutdown = 1;
    pthread_cond_signal(&pf->req_cv);
    pthread_mutex_unlock(&pf->mtx);
    pthread_join(pf->thread, NULL);
    pthread_mutex_destroy(&pf->mtx);
    pthread_cond_destroy(&pf->req_cv);
    pthread_cond_destroy(&pf->done_cv);
    free(pf);
}

// Post a prefetch request (1 or 2 preads). Non-blocking.
static void moe_prefetch_start(BnMoEPrefetch *pf,
                               uint8_t *buf1, size_t size1, off_t off1,
                               uint8_t *buf2, size_t size2, off_t off2) {
    pthread_mutex_lock(&pf->mtx);
    pf->reqs[0].buf = buf1; pf->reqs[0].size = size1; pf->reqs[0].offset = off1;
    pf->reqs[1].buf = buf2; pf->reqs[1].size = size2; pf->reqs[1].offset = off2;
    pf->n_reqs = 2;
    pf->active = 1;
    pthread_cond_signal(&pf->req_cv);
    pthread_mutex_unlock(&pf->mtx);
}

// Wait for prefetch to complete. Returns success flag.
static int moe_prefetch_wait(BnMoEPrefetch *pf) {
    pthread_mutex_lock(&pf->mtx);
    while (pf->active)
        pthread_cond_wait(&pf->done_cv, &pf->mtx);
    int ok = pf->success;
    pthread_mutex_unlock(&pf->mtx);
    return ok;
}

#endif // !__EMSCRIPTEN__

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

// Get offset and size for an expert projection.
// proj: 0=gate, 1=up, 2=down
static int moe_proj_info(const BnMoEExpertMap *map, int expert_idx, int proj,
                          size_t *offset, size_t *proj_bytes) {
    switch (proj) {
        case 0:
            *offset = map->gate_offset + (size_t)expert_idx * map->expert_gate_bytes;
            *proj_bytes = map->expert_gate_bytes;
            return 0;
        case 1:
            *offset = map->up_offset + (size_t)expert_idx * map->expert_up_bytes;
            *proj_bytes = map->expert_up_bytes;
            return 0;
        case 2:
            *offset = map->down_offset + (size_t)expert_idx * map->expert_down_bytes;
            *proj_bytes = map->expert_down_bytes;
            return 0;
        default:
            return -1;
    }
}

// Load one expert projection into a specific buffer.
// Returns pointer to data (mmap pointer or buf), or NULL on error.
static const void *moe_load_expert_proj_into(BnMoEState *ms, const BnMoEExpertMap *map,
                                              int expert_idx, int proj,
                                              uint8_t *buf, size_t buf_size) {
    size_t offset, proj_bytes;
    if (moe_proj_info(map, expert_idx, proj, &offset, &proj_bytes) < 0)
        return NULL;

    ms->io_bytes += proj_bytes;
    ms->io_count++;

    if (ms->mmap_base)
        return ms->mmap_base + offset;

#if !defined(__EMSCRIPTEN__)
    if (ms->fd < 0 || proj_bytes > buf_size) return NULL;
    double t0 = bn_platform_time_ms();
    ssize_t n = pread(ms->fd, buf, proj_bytes, (off_t)offset);
    ms->io_time_ms += bn_platform_time_ms() - t0;
    if (n != (ssize_t)proj_bytes) return NULL;
    return buf;
#else
    return NULL;
#endif
}

// Load one expert projection into the default expert_buf.
static const void *moe_load_expert_proj(BnMoEState *ms, const BnMoEExpertMap *map,
                                         int expert_idx, int proj) {
    return moe_load_expert_proj_into(ms, map, expert_idx, proj,
                                      ms->expert_buf, ms->expert_buf_size);
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

// Compiler barrier to prevent reordering of timing calls around dispatches
static inline double moe_time_ms(void) {
    double t = bn_platform_time_ms();
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" ::: "memory");
#endif
    return t;
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
    double t0;

    // 1. RMSNorm input
    t0 = moe_time_ms();
    moe_rmsnorm(s->xb, s->x, lw->ffn_norm, dim, c->norm_eps);
    ms->norm_time_ms += moe_time_ms() - t0;

    // 2. Route: select top-K experts (SIMD + threaded)
    t0 = moe_time_ms();
    bn_moe_route(ms, s->xb, lw->router_weight, dim, c->n_experts, K, m->pool);
    ms->route_time_ms += moe_time_ms() - t0;

    // 3. Zero output accumulator
    memset(ms->expert_out, 0, dim * sizeof(float));

    // 4. Expert FFN compute
    double t_compute = moe_time_ms();

    if (ms->mmap_base && K <= BN_MAX_MOE_K) {
        // --- Cross-expert batched dispatch (mmap path) ---
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
            // Gate+up batch
            t0 = moe_time_ms();
            BnMatvecTask gu_tasks[2 * BN_MAX_MOE_K];
            for (int k = 0; k < valid_k; k++) {
                gu_tasks[2*k]     = (BnMatvecTask){ ms->expert_hb_batch[k],  &wgates[k] };
                gu_tasks[2*k + 1] = (BnMatvecTask){ ms->expert_hb2_batch[k], &wups[k]   };
            }
            bn_quant_matvec_batch(gu_tasks, 2 * valid_k, s->xb, s->x_q, m->pool);
            ms->gate_up_time_ms += moe_time_ms() - t0;

            // Parallel SwiGLU
            t0 = moe_time_ms();
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
            ms->swiglu_time_ms += moe_time_ms() - t0;

            // Down projections
            t0 = moe_time_ms();
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

            // Individual matvec for each expert's down projection.
            // Each expert has different input (hb after SwiGLU), so we can't
            // share x quantization. Individual dispatch uses SDOT automatically.
            for (int k = 0; k < valid_k; k++) {
                if (valid_weights[k] == 0.0f) continue;
                bn_quant_matvec(ms->expert_down_batch[k], &wdowns[k],
                                ms->expert_hb_batch[k], s->x_q, m->pool);
            }
            ms->down_time_ms += moe_time_ms() - t0;

            // Weighted accumulation
            t0 = moe_time_ms();
            for (int k = 0; k < valid_k; k++) {
                float w = valid_weights[k];
                if (w == 0.0f) continue;
                for (int d = 0; d < dim; d++)
                    ms->expert_out[d] += w * ms->expert_down_batch[k][d];
            }
            ms->accum_time_ms += moe_time_ms() - t0;
        }
    }
#if !defined(__EMSCRIPTEN__)
    else if (ms->fd >= 0 && !ms->mmap_base) {
        // --- Pipelined pread path: overlap I/O with compute ---
        BnMoEPrefetch *pf = (BnMoEPrefetch *)ms->prefetch;
        uint8_t *cur_gate = ms->expert_buf, *cur_up = ms->expert_buf2;
        uint8_t *nxt_gate = ms->expert_buf3, *nxt_up = ms->expert_buf4;
        const BnMoEExpertMap *map = &lw->expert_map;

        // Bootstrap: load first expert's gate+up synchronously
        int first_k = -1;
        for (int k = 0; k < K; k++) {
            if (ms->expert_indices[k] >= 0) { first_k = k; break; }
        }
        if (first_k >= 0) {
            int first_eidx = ms->expert_indices[first_k];
            const void *g = moe_load_expert_proj_into(ms, map, first_eidx, 0,
                                                       cur_gate, ms->expert_buf_size);
            const void *u = moe_load_expert_proj_into(ms, map, first_eidx, 1,
                                                       cur_up, ms->expert_buf2_size);
            if (!g || !u) {
                SH_LOG_ERROR("Failed to bootstrap expert gate/up");
                first_k = -1;
            }
        }

        for (int k = first_k; k < K && k >= 0; k++) {
            int eidx = ms->expert_indices[k];
            float weight = ms->expert_weights[k];
            if (eidx < 0) continue;

            // Start prefetch for next valid expert
            int next_k = -1;
            if (pf) {
                for (int j = k + 1; j < K; j++) {
                    if (ms->expert_indices[j] >= 0) { next_k = j; break; }
                }
                if (next_k >= 0) {
                    int next_eidx = ms->expert_indices[next_k];
                    size_t g_off, g_sz, u_off, u_sz;
                    moe_proj_info(map, next_eidx, 0, &g_off, &g_sz);
                    moe_proj_info(map, next_eidx, 1, &u_off, &u_sz);
                    moe_prefetch_start(pf, nxt_gate, g_sz, (off_t)g_off,
                                           nxt_up, u_sz, (off_t)u_off);
                }
            }

            // Gate+up matvec
            t0 = moe_time_ms();
            BnQWeight wgate = moe_make_qweight(cur_gate, map->gate_type,
                                                map->gate_rows, map->gate_cols);
            BnQWeight wup = moe_make_qweight(cur_up, map->up_type,
                                              map->up_rows, map->up_cols);
            BnMatvecTask gu[2] = {
                { ms->expert_hb,  &wgate },
                { ms->expert_hb2, &wup   },
            };
            bn_quant_matvec_batch(gu, 2, s->xb, s->x_q, m->pool);
            ms->gate_up_time_ms += moe_time_ms() - t0;

            // SwiGLU activation
            t0 = moe_time_ms();
            moe_swiglu(ms->expert_hb, ms->expert_hb, ms->expert_hb2, moe_hidden);
            ms->swiglu_time_ms += moe_time_ms() - t0;

            // Down projection: reuse cur_gate buffer (free after gate+up matvec)
            t0 = moe_time_ms();
            const void *down_data = moe_load_expert_proj_into(ms, map, eidx, 2,
                                                               cur_gate, ms->expert_buf_size);
            if (!down_data) {
                SH_LOG_ERROR("Failed to load expert down projection");
                // Still need to wait for prefetch before continuing
                if (pf && next_k >= 0) {
                    double tw = moe_time_ms();
                    moe_prefetch_wait(pf);
                    ms->prefetch_wait_ms += moe_time_ms() - tw;
                    uint8_t *tmp;
                    tmp = cur_gate; cur_gate = nxt_gate; nxt_gate = tmp;
                    tmp = cur_up; cur_up = nxt_up; nxt_up = tmp;
                }
                continue;
            }
            BnQWeight wdown = moe_make_qweight(down_data, map->down_type,
                                                map->down_rows, map->down_cols);
            bn_quant_matvec(s->xb2, &wdown, ms->expert_hb, s->x_q, m->pool);
            ms->down_time_ms += moe_time_ms() - t0;

            // Weighted accumulation
            t0 = moe_time_ms();
            for (int d = 0; d < dim; d++)
                ms->expert_out[d] += weight * s->xb2[d];
            ms->accum_time_ms += moe_time_ms() - t0;

            // Wait for prefetch and swap buffers
            if (pf && next_k >= 0) {
                double tw = moe_time_ms();
                int ok = moe_prefetch_wait(pf);
                ms->prefetch_wait_ms += moe_time_ms() - tw;
                if (ok) {
                    // Track prefetched I/O in stats
                    pthread_mutex_lock(&pf->mtx);
                    ms->io_time_ms += pf->io_time_ms;
                    ms->io_bytes += pf->io_bytes;
                    ms->io_count += pf->n_reqs;
                    pf->io_time_ms = 0;
                    pf->io_bytes = 0;
                    pthread_mutex_unlock(&pf->mtx);
                }
                uint8_t *tmp;
                tmp = cur_gate; cur_gate = nxt_gate; nxt_gate = tmp;
                tmp = cur_up; cur_up = nxt_up; nxt_up = tmp;
                if (!ok) {
                    SH_LOG_ERROR("Prefetch failed for next expert");
                }
            }
        }
    }
#endif
    else {
        // --- Serial fallback (mmap K > BN_MAX_MOE_K or EMSCRIPTEN) ---
        for (int k = 0; k < K; k++) {
            int eidx = ms->expert_indices[k];
            float weight = ms->expert_weights[k];
            if (eidx < 0) continue;

            t0 = moe_time_ms();
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
            ms->gate_up_time_ms += moe_time_ms() - t0;

            // SwiGLU activation
            t0 = moe_time_ms();
            moe_swiglu(ms->expert_hb, ms->expert_hb, ms->expert_hb2, moe_hidden);
            ms->swiglu_time_ms += moe_time_ms() - t0;

            // Down projection
            t0 = moe_time_ms();
            const void *down_data = moe_load_expert_proj(ms, &lw->expert_map, eidx, 2);
            if (!down_data) {
                SH_LOG_ERROR("Failed to load expert down projection");
                continue;
            }
            BnQWeight wdown = moe_make_qweight(down_data, lw->expert_map.down_type,
                                                lw->expert_map.down_rows, lw->expert_map.down_cols);
            bn_quant_matvec(s->xb2, &wdown, ms->expert_hb, s->x_q, m->pool);
            ms->down_time_ms += moe_time_ms() - t0;

            // Weighted accumulation
            t0 = moe_time_ms();
            for (int d = 0; d < dim; d++)
                ms->expert_out[d] += weight * s->xb2[d];
            ms->accum_time_ms += moe_time_ms() - t0;
        }
    }

    // 5. Shared expert (if present, always resident)
    t0 = moe_time_ms();
    if (c->has_shared_expert && lw->shared_gate.data) {
        int shared_hidden = c->shared_expert_intermediate_size;

        bn_quant_matvec(s->hb, &lw->shared_gate, s->xb, s->x_q, m->pool);
        bn_quant_matvec(s->hb2, &lw->shared_up, s->xb, s->x_q, m->pool);
        moe_swiglu(s->hb, s->hb, s->hb2, shared_hidden);
        bn_quant_matvec(s->xb2, &lw->shared_down, s->hb, s->x_q, m->pool);

        for (int d = 0; d < dim; d++)
            ms->expert_out[d] += s->xb2[d];
    }
    ms->shared_time_ms += moe_time_ms() - t0;

    ms->compute_time_ms += moe_time_ms() - t_compute;

    // 6. Copy result to xb for residual add by caller
    memcpy(s->xb, ms->expert_out, dim * sizeof(float));

    // 7. Residual add
    for (int d = 0; d < dim; d++)
        s->x[d] += s->xb[d];
}

void bn_moe_print_stats(const BnMoEState *ms, int n_tokens) {
    if (!ms || n_tokens <= 0) return;

    double io_per_tok = (double)ms->io_bytes / (1024.0 * 1024.0) / n_tokens;

    char iot_s[32], bw_s[32], rss_s[32];
    char norm_s[32], rt_s[32], gu_s[32], sw_s[32], dn_s[32], ac_s[32], sh_s[32], ct_s[32];

    snprintf(iot_s, sizeof(iot_s), "%.1f", io_per_tok);

    if (ms->io_time_ms > 0.1)
        snprintf(bw_s, sizeof(bw_s), "%.0f",
                 (double)ms->io_bytes / (1024.0 * 1024.0) / (ms->io_time_ms / 1000.0));
    else
        snprintf(bw_s, sizeof(bw_s), "mmap");

    snprintf(norm_s, sizeof(norm_s), "%.1f", ms->norm_time_ms);
    snprintf(rt_s, sizeof(rt_s), "%.1f", ms->route_time_ms);
    snprintf(gu_s, sizeof(gu_s), "%.1f", ms->gate_up_time_ms);
    snprintf(sw_s, sizeof(sw_s), "%.1f", ms->swiglu_time_ms);
    snprintf(dn_s, sizeof(dn_s), "%.1f", ms->down_time_ms);
    snprintf(ac_s, sizeof(ac_s), "%.1f", ms->accum_time_ms);
    snprintf(sh_s, sizeof(sh_s), "%.1f", ms->shared_time_ms);
    snprintf(ct_s, sizeof(ct_s), "%.1f", ms->compute_time_ms);

    char pw_s[32];
    snprintf(pw_s, sizeof(pw_s), "%.1f", ms->prefetch_wait_ms);

    size_t rss = bn_platform_rss_bytes();
    snprintf(rss_s, sizeof(rss_s), "%.2f", (double)rss / (1024.0 * 1024.0 * 1024.0));

    SH_LOG_INFO("MoE stats",
                "MB/tok", iot_s,
                "stream_MB/s", bw_s,
                "rss_GB", rss_s);
    SH_LOG_INFO("MoE breakdown (ms)",
                "norm", norm_s,
                "route", rt_s,
                "gate+up", gu_s,
                "swiglu", sw_s,
                "down", dn_s,
                "accum", ac_s,
                "shared", sh_s,
                "pf_wait", pw_s,
                "total", ct_s);
}

void bn_moe_reset_stats(BnMoEState *ms) {
    if (!ms) return;
    ms->io_bytes = 0;
    ms->io_time_ms = 0;
    ms->route_time_ms = 0;
    ms->compute_time_ms = 0;
    ms->gate_up_time_ms = 0;
    ms->swiglu_time_ms = 0;
    ms->down_time_ms = 0;
    ms->accum_time_ms = 0;
    ms->shared_time_ms = 0;
    ms->norm_time_ms = 0;
    ms->io_count = 0;
    ms->prefetch_wait_ms = 0;
}

void bn_moe_prefetch_create(BnMoEState *ms) {
    if (!ms || ms->prefetch) return;
#if !defined(__EMSCRIPTEN__)
    if (ms->fd >= 0 && !ms->mmap_base) {
        ms->prefetch = moe_prefetch_init(ms->fd);
        if (ms->prefetch)
            SH_LOG_INFO("MoE I/O prefetch thread", "status", "created");
    }
#endif
}

void bn_moe_prefetch_destroy(BnMoEState *ms) {
    if (!ms || !ms->prefetch) return;
#if !defined(__EMSCRIPTEN__)
    moe_prefetch_free((BnMoEPrefetch *)ms->prefetch);
    ms->prefetch = NULL;
#endif
}
