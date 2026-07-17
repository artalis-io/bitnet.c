#include "gpu_internal.h"
#include "backend_session.h"
#include "model_arch.h"
#include "platform.h"
#include "quant.h"
#include "quant_dispatch_internal.h"
#include "session.h"
#include "transformer_cpu_backend_internal.h"
#include "../gpu_shader_ir_internal.h"
#include "../moe_internal.h"
#include "moe.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BN_GPU_LOGITS_REFINE_MAX_SCALE_BLOCKS 8192

static void gpu_debug_compare_vec_local(const char *label,
                                        int layer,
                                        int pos,
                                        const float *cpu,
                                        const float *gpu,
                                        int n) {
    if (!label || !cpu || !gpu || n <= 0) return;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int max_i = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(cpu[i] - gpu[i]);
        sum_abs += (double)diff;
        sum_sq += (double)diff * (double)diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
    }
    fprintf(stderr,
            "[bn:gpu:debug] %s layer=%d pos=%d max_abs=%.9g max_i=%d "
            "cpu=%.9g gpu=%.9g mean_abs=%.9g rms=%.9g\n",
            label, layer, pos, max_abs, max_i, cpu[max_i], gpu[max_i],
            sum_abs / (double)n, sqrt(sum_sq / (double)n));
}

static void gpu_debug_rmsnorm_scalar_local(float *out,
                                           const float *x,
                                           const float *w,
                                           int n,
                                           float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * w[i];
}

static int gpu_debug_compute_moe_actual_cpu_from_state(
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    float *x_out) {
    if (!m || !sess || !lw || !x_out || !sess->moe_state)
        return -1;
    BnRunState *s = &sess->state;
    int K = m->config.n_experts_active;
    if (K > BN_MAX_MOE_K)
        K = BN_MAX_MOE_K;
    float *save_x = (float *)malloc((size_t)dim * sizeof(float));
    float *save_xb = (float *)malloc((size_t)dim * sizeof(float));
    int save_indices[BN_MAX_MOE_K];
    float save_weights[BN_MAX_MOE_K];
    if (!save_x || !save_xb) {
        free(save_x);
        free(save_xb);
        return -1;
    }
    memcpy(save_x, s->x, (size_t)dim * sizeof(float));
    memcpy(save_xb, s->xb, (size_t)dim * sizeof(float));
    for (int k = 0; k < K; k++) {
        save_indices[k] = sess->moe_state->expert_indices[k];
        save_weights[k] = sess->moe_state->expert_weights[k];
    }

    bn_model_set_gpu_disabled(m, 1);
    bn_moe_forward(m, sess, lw, layer);
    bn_model_set_gpu_disabled(m, 0);
    memcpy(x_out, s->x, (size_t)dim * sizeof(float));

    memcpy(s->x, save_x, (size_t)dim * sizeof(float));
    memcpy(s->xb, save_xb, (size_t)dim * sizeof(float));
    for (int k = 0; k < K; k++) {
        sess->moe_state->expert_indices[k] = save_indices[k];
        sess->moe_state->expert_weights[k] = save_weights[k];
    }
    free(save_x);
    free(save_xb);
    return 0;
}

static int gpu_debug_compute_moe_cpu_from_xb(
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int dim,
    const float *x_in,
    const float *xb_in,
    float *x_out) {
    if (!m || !sess || !lw || !x_in || !xb_in || !x_out || !sess->moe_state)
        return -1;
    BnMoEState *ms = sess->moe_state;
    BnConfig *c = &m->config;
    int K = c->n_experts_active;
    int moe_hidden = c->moe_intermediate_size;
    uint32_t gateup_flags = bn_transformer_gpu_moe_gateup_task_flags(c);
    int hidden_cap = moe_hidden;
    if (c->shared_expert_intermediate_size > hidden_cap)
        hidden_cap = c->shared_expert_intermediate_size;
    float *expert_out = (float *)calloc((size_t)dim, sizeof(float));
    float *hb = (float *)malloc((size_t)hidden_cap * sizeof(float));
    float *hb2 = (float *)malloc((size_t)hidden_cap * sizeof(float));
    float *down = (float *)malloc((size_t)dim * sizeof(float));
    if (!expert_out || !hb || !hb2 || !down) {
        free(expert_out);
        free(hb);
        free(hb2);
        free(down);
        return -1;
    }

    const BnMoEExpertMap *em = &lw->moe.expert_map;
    for (int k = 0; k < K; k++) {
        int eidx = ms->expert_indices[k];
        float weight = ms->expert_weights[k];
        if (eidx < 0) continue;
        const void *gate_data = bn_moe_get_expert_proj(
            bn_model_moe_io(m), ms, em, eidx, 0);
        const void *up_data = bn_moe_get_expert_proj(
            bn_model_moe_io(m), ms, em, eidx, 1);
        const void *down_data = bn_moe_get_expert_proj(
            bn_model_moe_io(m), ms, em, eidx, 2);
        if (!gate_data || !up_data || !down_data) {
            free(expert_out);
            free(hb);
            free(hb2);
            free(down);
            return -1;
        }
        BnQWeight wgate = bn_moe_make_qweight(
            gate_data, em->gate_type, em->gate_rows, em->gate_cols);
        BnQWeight wup = bn_moe_make_qweight(
            up_data, em->up_type, em->up_rows, em->up_cols);
        BnQWeight wdown = bn_moe_make_qweight(
            down_data, em->down_type, em->down_rows, em->down_cols);
        BnMatvecTask gu_tasks[2] = {
            { hb,  &wgate, NULL, gateup_flags },
            { hb2, &wup,   NULL, gateup_flags },
        };
        bn_quant_matvec_batch(gu_tasks, 2, xb_in, sess->state.x_q,
                              bn_model_pool(m));
        bn_moe_swiglu(hb, hb, hb2, moe_hidden, c->moe_exact_silu);
        bn_quant_matvec(down, &wdown, hb, sess->state.x_q, bn_model_pool(m));
        bn_moe_weighted_add(expert_out, down, weight, dim);
    }

    if (c->has_shared_expert && lw->shared.shared_gate.data) {
        int shared_hidden = c->shared_expert_intermediate_size;
        BnMatvecTask shared_gu_tasks[2] = {
            { hb,  &lw->shared.shared_gate, NULL, gateup_flags },
            { hb2, &lw->shared.shared_up,   NULL, gateup_flags },
        };
        bn_quant_matvec_batch(shared_gu_tasks, 2, xb_in, sess->state.x_q,
                              bn_model_pool(m));
        bn_moe_swiglu(hb, hb, hb2, shared_hidden, c->moe_exact_silu);
        bn_quant_matvec(down, &lw->shared.shared_down, hb,
                        sess->state.x_q, bn_model_pool(m));
        if (lw->shared.shared_expert_gate) {
            float gate_dot = 0.0f;
            for (int d = 0; d < dim; d++)
                gate_dot += xb_in[d] * lw->shared.shared_expert_gate[d];
            float gate = 1.0f / (1.0f + expf(-gate_dot));
            bn_moe_weighted_add(expert_out, down, gate, dim);
        } else {
            bn_moe_weighted_add(expert_out, down, 1.0f, dim);
        }
    }

    for (int i = 0; i < dim; i++)
        x_out[i] = x_in[i] + expert_out[i];

    free(expert_out);
    free(hb);
    free(hb2);
    free(down);
    return 0;
}

static int gpu_debug_compute_moe_parts_cpu_from_xb(
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int dim,
    const float *xb_in,
    float *routed_out,
    float *shared_out) {
    if (!m || !sess || !lw || !xb_in || !routed_out || !shared_out ||
        !sess->moe_state)
        return -1;
    BnMoEState *ms = sess->moe_state;
    BnConfig *c = &m->config;
    int K = c->n_experts_active;
    int moe_hidden = c->moe_intermediate_size;
    uint32_t gateup_flags = bn_transformer_gpu_moe_gateup_task_flags(c);
    int hidden_cap = moe_hidden;
    if (c->shared_expert_intermediate_size > hidden_cap)
        hidden_cap = c->shared_expert_intermediate_size;
    float *hb = (float *)malloc((size_t)hidden_cap * sizeof(float));
    float *hb2 = (float *)malloc((size_t)hidden_cap * sizeof(float));
    float *down = (float *)malloc((size_t)dim * sizeof(float));
    if (!hb || !hb2 || !down) {
        free(hb);
        free(hb2);
        free(down);
        return -1;
    }
    memset(routed_out, 0, (size_t)dim * sizeof(float));
    memset(shared_out, 0, (size_t)dim * sizeof(float));

    const BnMoEExpertMap *em = &lw->moe.expert_map;
    for (int k = 0; k < K; k++) {
        int eidx = ms->expert_indices[k];
        float weight = ms->expert_weights[k];
        if (eidx < 0) continue;
        const void *gate_data = bn_moe_get_expert_proj(
            bn_model_moe_io(m), ms, em, eidx, 0);
        const void *up_data = bn_moe_get_expert_proj(
            bn_model_moe_io(m), ms, em, eidx, 1);
        const void *down_data = bn_moe_get_expert_proj(
            bn_model_moe_io(m), ms, em, eidx, 2);
        if (!gate_data || !up_data || !down_data) {
            free(hb);
            free(hb2);
            free(down);
            return -1;
        }
        BnQWeight wgate = bn_moe_make_qweight(
            gate_data, em->gate_type, em->gate_rows, em->gate_cols);
        BnQWeight wup = bn_moe_make_qweight(
            up_data, em->up_type, em->up_rows, em->up_cols);
        BnQWeight wdown = bn_moe_make_qweight(
            down_data, em->down_type, em->down_rows, em->down_cols);
        BnMatvecTask gu_tasks[2] = {
            { hb,  &wgate, NULL, gateup_flags },
            { hb2, &wup,   NULL, gateup_flags },
        };
        bn_quant_matvec_batch(gu_tasks, 2, xb_in, sess->state.x_q,
                              bn_model_pool(m));
        bn_moe_swiglu(hb, hb, hb2, moe_hidden, c->moe_exact_silu);
        bn_quant_matvec(down, &wdown, hb, sess->state.x_q, bn_model_pool(m));
        bn_moe_weighted_add(routed_out, down, weight, dim);
    }

    if (c->has_shared_expert && lw->shared.shared_gate.data) {
        int shared_hidden = c->shared_expert_intermediate_size;
        BnMatvecTask shared_gu_tasks[2] = {
            { hb,  &lw->shared.shared_gate, NULL, gateup_flags },
            { hb2, &lw->shared.shared_up,   NULL, gateup_flags },
        };
        bn_quant_matvec_batch(shared_gu_tasks, 2, xb_in, sess->state.x_q,
                              bn_model_pool(m));
        bn_moe_swiglu(hb, hb, hb2, shared_hidden, c->moe_exact_silu);
        bn_quant_matvec(down, &lw->shared.shared_down, hb,
                        sess->state.x_q, bn_model_pool(m));
        if (lw->shared.shared_expert_gate) {
            float gate_dot = 0.0f;
            for (int d = 0; d < dim; d++)
                gate_dot += xb_in[d] * lw->shared.shared_expert_gate[d];
            float gate = 1.0f / (1.0f + expf(-gate_dot));
            bn_moe_weighted_add(shared_out, down, gate, dim);
        } else {
            bn_moe_weighted_add(shared_out, down, 1.0f, dim);
        }
    }

    free(hb);
    free(hb2);
    free(down);
    return 0;
}

static int gpu_compute_shared_expert_cpu_from_xb(
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int dim,
    const float *xb_in,
    float *shared_out) {
    if (!m || !sess || !lw || !xb_in || !shared_out ||
        !lw->shared.shared_gate.data)
        return -1;
    BnConfig *c = &m->config;
    int shared_hidden = c->shared_expert_intermediate_size;
    uint32_t gateup_flags = bn_transformer_gpu_moe_gateup_task_flags(c);
    if (shared_hidden <= 0)
        return -1;
    float *hb = (float *)malloc((size_t)shared_hidden * sizeof(float));
    float *hb2 = (float *)malloc((size_t)shared_hidden * sizeof(float));
    float *down = (float *)malloc((size_t)dim * sizeof(float));
    if (!hb || !hb2 || !down) {
        free(hb);
        free(hb2);
        free(down);
        return -1;
    }

    BnMatvecTask gu_tasks[2] = {
        { hb,  &lw->shared.shared_gate, NULL, gateup_flags },
        { hb2, &lw->shared.shared_up,   NULL, gateup_flags },
    };
    bn_quant_matvec_batch(gu_tasks, 2, xb_in, sess->state.x_q,
                          bn_model_pool(m));
    bn_moe_swiglu(hb, hb, hb2, shared_hidden, c->moe_exact_silu);
    bn_quant_matvec(down, &lw->shared.shared_down, hb,
                    sess->state.x_q, bn_model_pool(m));
    memset(shared_out, 0, (size_t)dim * sizeof(float));
    if (lw->shared.shared_expert_gate) {
        float gate_dot = 0.0f;
        for (int d = 0; d < dim; d++)
            gate_dot += xb_in[d] * lw->shared.shared_expert_gate[d];
        float gate = 1.0f / (1.0f + expf(-gate_dot));
        bn_moe_weighted_add(shared_out, down, gate, dim);
    } else {
        bn_moe_weighted_add(shared_out, down, 1.0f, dim);
    }

    free(hb);
    free(hb2);
    free(down);
    return 0;
}

static int gpu_debug_compute_moe_mid_cpu_from_xb(
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const float *xb_in,
    float *mid_out) {
    if (!m || !sess || !lw || !xb_in || !mid_out || !sess->moe_state)
        return -1;
    BnMoEState *ms = sess->moe_state;
    BnConfig *c = &m->config;
    int K = c->n_experts_active;
    int moe_hidden = c->moe_intermediate_size;
    uint32_t gateup_flags = bn_transformer_gpu_moe_gateup_task_flags(c);
    float *hb = (float *)malloc((size_t)moe_hidden * sizeof(float));
    float *hb2 = (float *)malloc((size_t)moe_hidden * sizeof(float));
    if (!hb || !hb2) {
        free(hb);
        free(hb2);
        return -1;
    }

    const BnMoEExpertMap *em = &lw->moe.expert_map;
    for (int k = 0; k < K; k++) {
        int eidx = ms->expert_indices[k];
        if (eidx < 0) {
            memset(mid_out + (size_t)k * (size_t)moe_hidden, 0,
                   (size_t)moe_hidden * sizeof(float));
            continue;
        }
        const void *gate_data = bn_moe_get_expert_proj(
            bn_model_moe_io(m), ms, em, eidx, 0);
        const void *up_data = bn_moe_get_expert_proj(
            bn_model_moe_io(m), ms, em, eidx, 1);
        if (!gate_data || !up_data) {
            free(hb);
            free(hb2);
            return -1;
        }
        BnQWeight wgate = bn_moe_make_qweight(
            gate_data, em->gate_type, em->gate_rows, em->gate_cols);
        BnQWeight wup = bn_moe_make_qweight(
            up_data, em->up_type, em->up_rows, em->up_cols);
        BnMatvecTask gu_tasks[2] = {
            { hb,  &wgate, NULL, gateup_flags },
            { hb2, &wup,   NULL, gateup_flags },
        };
        bn_quant_matvec_batch(gu_tasks, 2, xb_in, sess->state.x_q,
                              bn_model_pool(m));
        bn_moe_swiglu(hb, hb, hb2, moe_hidden, c->moe_exact_silu);
        memcpy(mid_out + (size_t)k * (size_t)moe_hidden, hb,
               (size_t)moe_hidden * sizeof(float));
    }

    free(hb);
    free(hb2);
    return 0;
}

static int gpu_debug_compute_moe_raw_all_cpu_from_xb(
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const float *xb_in,
    float *gate_out,
    float *up_out) {
    if (!m || !sess || !lw || !xb_in || !gate_out || !up_out ||
        !sess->moe_state)
        return -1;
    BnMoEState *ms = sess->moe_state;
    BnConfig *c = &m->config;
    int n_experts = c->n_experts;
    int moe_hidden = c->moe_intermediate_size;
    if (n_experts <= 0 || moe_hidden <= 0)
        return -1;
    uint32_t gateup_flags = bn_transformer_gpu_moe_gateup_task_flags(c);

    const BnMoEExpertMap *em = &lw->moe.expert_map;
    for (int eidx = 0; eidx < n_experts; eidx++) {
        const void *gate_data = bn_moe_get_expert_proj(
            bn_model_moe_io(m), ms, em, eidx, 0);
        const void *up_data = bn_moe_get_expert_proj(
            bn_model_moe_io(m), ms, em, eidx, 1);
        if (!gate_data || !up_data)
            return -1;
        BnQWeight wgate = bn_moe_make_qweight(
            gate_data, em->gate_type, em->gate_rows, em->gate_cols);
        BnQWeight wup = bn_moe_make_qweight(
            up_data, em->up_type, em->up_rows, em->up_cols);
        BnMatvecTask gu_tasks[2] = {
            {
                gate_out + (size_t)eidx * (size_t)moe_hidden,
                &wgate,
                NULL,
                gateup_flags
            },
            {
                up_out + (size_t)eidx * (size_t)moe_hidden,
                &wup,
                NULL,
                gateup_flags
            },
        };
        bn_quant_matvec_batch(gu_tasks, 2, xb_in, sess->state.x_q,
                              bn_model_pool(m));
    }
    return 0;
}

static int gpu_debug_compute_shared_mid_cpu_from_xb(
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const float *xb_in,
    float *mid_out) {
    if (!m || !sess || !lw || !xb_in || !mid_out ||
        !lw->shared.shared_gate.data)
        return -1;
    BnConfig *c = &m->config;
    int hidden = c->shared_expert_intermediate_size;
    uint32_t gateup_flags = bn_transformer_gpu_moe_gateup_task_flags(c);
    float *hb2 = (float *)malloc((size_t)hidden * sizeof(float));
    if (!hb2)
        return -1;
    BnMatvecTask gu_tasks[2] = {
        { mid_out, &lw->shared.shared_gate, NULL, gateup_flags },
        { hb2,     &lw->shared.shared_up,   NULL, gateup_flags },
    };
    bn_quant_matvec_batch(gu_tasks, 2, xb_in, sess->state.x_q,
                          bn_model_pool(m));
    bn_moe_swiglu(mid_out, mid_out, hb2, hidden, c->moe_exact_silu);
    free(hb2);
    return 0;
}

static int gpu_debug_compute_shared_down_cpu_from_xb(
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int dim,
    const float *xb_in,
    float *down_out) {
    if (!m || !sess || !lw || !xb_in || !down_out || dim <= 0 ||
        !lw->shared.shared_gate.data)
        return -1;
    BnConfig *c = &m->config;
    int hidden = c->shared_expert_intermediate_size;
    uint32_t gateup_flags = bn_transformer_gpu_moe_gateup_task_flags(c);
    if (hidden <= 0)
        return -1;
    float *hb = (float *)malloc((size_t)hidden * sizeof(float));
    float *hb2 = (float *)malloc((size_t)hidden * sizeof(float));
    if (!hb || !hb2) {
        free(hb);
        free(hb2);
        return -1;
    }
    BnMatvecTask gu_tasks[2] = {
        { hb,  &lw->shared.shared_gate, NULL, gateup_flags },
        { hb2, &lw->shared.shared_up,   NULL, gateup_flags },
    };
    bn_quant_matvec_batch(gu_tasks, 2, xb_in, sess->state.x_q,
                          bn_model_pool(m));
    bn_moe_swiglu(hb, hb, hb2, hidden, c->moe_exact_silu);
    bn_quant_matvec(down_out, &lw->shared.shared_down, hb,
                    sess->state.x_q, bn_model_pool(m));
    free(hb);
    free(hb2);
    return 0;
}

static void gpu_moe_route_profile_add(int dim,
                                      int n_experts,
                                      double flush_ms,
                                      double read_ms,
                                      double route_ms,
                                      double resolve_ms) {
    static unsigned long long calls = 0;
    static double total_flush = 0.0;
    static double total_read = 0.0;
    static double total_route = 0.0;
    static double total_resolve = 0.0;
    if (!bn_transformer_gpu_moe_route_profile_enabled())
        return;
    calls++;
    total_flush += flush_ms;
    total_read += read_ms;
    total_route += route_ms;
    total_resolve += resolve_ms;
    int every = bn_transformer_gpu_moe_route_profile_every();
    if ((calls % (unsigned long long)every) == 0) {
        fprintf(stderr,
                "[bn:gpu:moe_route_profile] calls=%llu dim=%d experts=%d "
                "flush=%.3f read=%.3f route=%.3f resolve=%.3f total=%.3f\n",
                calls, dim, n_experts, total_flush, total_read,
                total_route, total_resolve,
                total_flush + total_read + total_route + total_resolve);
        total_flush = 0.0;
        total_read = 0.0;
        total_route = 0.0;
        total_resolve = 0.0;
    }
}

static int gpu_resolve_moe_all2_resources(BnGPUMoEResources *out,
                                          BnGPUMoEResolvedExpert *storage,
                                          BnModel *m,
                                          BnSession *sess,
                                          const BnLayerWeights *lw,
                                          int layer,
                                          void *router_diff,
                                          BnGPUMoETemporaryBuffers *temps) {
    if (!out || !storage || !m || !sess || !lw || !router_diff || !temps)
        return -1;
    BnConfig *c = &m->config;
    if (!bn_model_arch_uses_all_active_two_expert_moe(c, c->dim))
        return -1;
    memset(out, 0, sizeof(*out));
    memset(temps, 0, sizeof(*temps));
    out->expert_map = &lw->moe.expert_map;
    out->experts = storage;
    out->n_experts = 2;
    out->moe_hidden = c->moe_intermediate_size;
    for (int e = 0; e < 2; e++) {
        memset(&storage[e], 0, sizeof(storage[e]));
        if (bn_gpu_moe_bridge_get_expert(m, sess, lw, layer, e, temps,
                                         &storage[e].buffers) != 0)
            return -1;
        storage[e].weight = 1.0f;
        storage[e].route_gate = router_diff;
        storage[e].route_complement = e == 1;
    }
    return 0;
}

static int gpu_refine_q6k_logits_top(float *logits, int n_logits,
                                     const BnQWeight *W, const float *x,
                                     int8_t *x_q_buf, int top_n) {
    if (!logits || !W || !W->data || !x || !x_q_buf)
        return 0;
    if (top_n <= 0) return 0;
    if (top_n > 4096) top_n = 4096;
    if (top_n > n_logits) top_n = n_logits;
    int n_blocks = W->cols / BN_QK_K;
    if (n_blocks < 1 || n_blocks > BN_GPU_LOGITS_REFINE_MAX_SCALE_BLOCKS / 16)
        return 0;
    float x_d[n_blocks];
    int16_t x_bsums[n_blocks * 16];
    bn_transformer_cpu_quantize_q8k_activation(x, x_q_buf, x_d, x_bsums,
                                               W->cols);

    int ids[4096];
    float vals[4096];
    int n_top = 0;
    for (int i = 0; i < n_logits; i++) {
        float v = logits[i];
        int j = n_top;
        if (j == top_n && v <= vals[j - 1]) continue;
        if (j < top_n) {
            ids[j] = i;
            vals[j] = v;
            n_top++;
        } else {
            j--;
        }
        while (j > 0 && v > vals[j - 1]) {
            ids[j] = ids[j - 1];
            vals[j] = vals[j - 1];
            j--;
        }
        ids[j] = i;
        vals[j] = v;
    }

    for (int i = 0; i < n_top; i++) {
        float row_sum;
        if (bn_quant_q6_logits_refine_q8k_row(W, x_q_buf, x_d, x_bsums,
                                              ids[i], &row_sum) == 0)
            logits[ids[i]] = row_sum;
    }
    return n_top;
}

static int gpu_refine_q8_logits_top(float *logits, int n_logits,
                                    const BnQWeight *W, const float *x,
                                    int8_t *x_q, int top_n) {
    if (!logits || !W || !W->data || !x || !x_q)
        return 0;
    if (!bn_transformer_cpu_has_native_q8x_quant())
        return 0;
    if (top_n <= 0) return 0;
    if (top_n > 128) top_n = 128;
    if (top_n > n_logits) top_n = n_logits;
    int n_blocks = W->cols / 32;
    if (n_blocks <= 0 || n_blocks > BN_GPU_LOGITS_REFINE_MAX_SCALE_BLOCKS)
        return 0;

    int ids[128];
    float vals[128];
    int n_top = 0;
    for (int i = 0; i < n_logits; i++) {
        float v = logits[i];
        int j = n_top;
        if (j == top_n && v <= vals[j - 1]) continue;
        if (j < top_n) {
            ids[j] = i;
            vals[j] = v;
            n_top++;
        } else {
            j--;
        }
        while (j > 0 && v > vals[j - 1]) {
            ids[j] = ids[j - 1];
            vals[j] = vals[j - 1];
            j--;
        }
        ids[j] = i;
        vals[j] = v;
    }

    float x_scales[BN_GPU_LOGITS_REFINE_MAX_SCALE_BLOCKS];
    if (bn_transformer_cpu_quantize_q8_blocks_native(x, x_q, x_scales,
                                                     W->cols) != 0)
        return 0;
    for (int i = 0; i < n_top; i++) {
        float row_sum;
        if (bn_quant_q8_logits_refine_row(W, x_q, x_scales, ids[i],
                                          &row_sum) == 0)
            logits[ids[i]] = row_sum;
    }
    return n_top;
}

static int gpu_patch_cached_decode_ops(BnGPUOp *ops, int n_ops,
                                       const BnConfig *c, int pos) {
    if (!ops || !c || n_ops <= 0 || c->seq_len <= 0 || c->kv_dim <= 0)
        return -1;
    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
    int cache_pos = pos % c->seq_len;
    uint32_t kv_dim = (uint32_t)c->kv_dim;
    uint32_t layer_span = (uint32_t)(c->seq_len * c->kv_dim);
    uint32_t cache_off = (uint32_t)(cache_pos * c->kv_dim);
    for (int i = 0; i < n_ops; i++) {
        BnGPUOp *op = &ops[i];
        switch (op->op_code) {
        case BN_GPU_CODE_MATVEC:
            if ((op->buf_out == BN_GPU_VALUE_KEY_CACHE ||
                 op->buf_out == BN_GPU_VALUE_VALUE_CACHE) &&
                op->rows == (int)kv_dim && layer_span > 0) {
                uint32_t base = (op->p[5] / layer_span) * layer_span;
                op->p[5] = base + cache_off;
            }
            break;
        case BN_GPU_CODE_MATVEC_SPLIT:
        case BN_GPU_CODE_Q4K_MATVEC_SPLIT:
        case BN_GPU_CODE_Q8_MATVEC_SPLIT:
        case BN_GPU_CODE_Q5K_MATVEC_SPLIT:
            if (op->buf_aux == BN_GPU_VALUE_KEY_CACHE &&
                op->p[0] - op->p[2] >= kv_dim && layer_span > 0) {
                uint32_t base = (op->p[6] / layer_span) * layer_span;
                op->p[6] = base + cache_off;
            }
            if (op->rows == BN_GPU_VALUE_VALUE_CACHE &&
                op->p[3] > op->p[2] && layer_span > 0) {
                uint32_t base = (op->p[7] / layer_span) * layer_span;
                op->p[7] = base + cache_off;
            }
            break;
        case BN_GPU_CODE_ROPE:
        case BN_GPU_CODE_ROPE_QK:
            op->p[2] = (uint32_t)pos;
            if (op->op_code == BN_GPU_CODE_ROPE_QK &&
                op->buf_aux == BN_GPU_VALUE_KEY_CACHE &&
                layer_span > 0) {
                uint32_t base = (op->p[5] / layer_span) * layer_span;
                op->p[5] = base + cache_off;
            }
            break;
        case BN_GPU_CODE_FLASH_ATTN:
            op->p[2] = (uint32_t)n_kv;
            break;
        case BN_GPU_CODE_PER_HEAD_RMSNORM:
            if (op->buf_in == BN_GPU_VALUE_KEY_CACHE &&
                op->p[0] == (uint32_t)c->head_size && layer_span > 0) {
                uint32_t base = (op->p[3] / layer_span) * layer_span;
                op->p[3] = base + cache_off;
            }
            break;
        case BN_GPU_CODE_COPY:
            if ((op->buf_out == BN_GPU_VALUE_KEY_CACHE ||
                 op->buf_out == BN_GPU_VALUE_VALUE_CACHE) &&
                op->p[2] == kv_dim && layer_span > 0) {
                uint32_t base = (op->p[1] / layer_span) * layer_span;
                op->p[1] = base + cache_off;
            }
            break;
        default:
            break;
        }
    }
    return 0;
}

static int gpu_qkv_resources_missing(
    const BnLayerWeights *lw,
    const BnLayerShapePlan *plan,
    const BnTransformerGPUQKVResources *res) {
    if (lw->ssm.wqkv.data)
        return !(res && res->packed_qkv);
    if (!lw->attn.wq.data)
        return 0;
    if (!(res && res->wq && res->wk && res->wv))
        return 1;

    int has_qkv = res && res->qkv_stacked && !plan->q_gated &&
                  !lw->attn.q_bias && !lw->attn.k_bias && !lw->attn.v_bias;
    int has_qk = res && res->qk_stacked && !plan->q_gated &&
                 lw->attn.wq.rows == plan->q_dim &&
                 lw->attn.wk.rows == plan->kv_dim &&
                 lw->attn.wq.cols == lw->attn.wk.cols &&
                 bn_transformer_gpu_can_use_stacked_qk(lw->attn.wq.type,
                                                       lw->attn.wk.type);
    if (has_qkv)
        return 0;
    if (has_qk)
        return !(res && res->wv);
    return !(res && res->wq && res->wk && res->wv);
}

static int gpu_attention_resources_missing(
    const BnLayerWeights *lw,
    const BnTransformerGPUAttentionResources *res) {
    return lw->attn.wo.data && !(res && res->wo);
}

static int gpu_dense_ffn_resources_missing(
    const BnLayerWeights *lw,
    const BnFFNPlan *plan,
    const BnTransformerGPUDenseFFNResources *res) {
    if (lw->moe.router_weight)
        return 0;
    if ((lw->ffn.ffn_gate.data && !(res && res->ffn_gate)) ||
        (lw->ffn.ffn_up.data && !(res && res->ffn_up)) ||
        (lw->ffn.ffn_down.data && !(res && res->ffn_down)))
        return 1;
    if (plan->has_gate && lw->ffn.ffn_gate.data) {
        int has_gateup = res && res->gateup_stacked &&
                         lw->ffn.ffn_gate.rows == lw->ffn.ffn_up.rows &&
                         lw->ffn.ffn_gate.cols == lw->ffn.ffn_up.cols;
        if (!has_gateup && !(res && res->ffn_gate && res->ffn_up))
            return 1;
    } else if (lw->ffn.ffn_up.data && !(res && res->ffn_up)) {
        return 1;
    }
    return lw->ffn.ffn_down.data && !(res && res->ffn_down);
}

// GPU-resident forward pass: one submit per token, reads back logits only.
// Supports classic transformer only (no MoE, no SSM, no gated-Q, no wide-Q,
// no Q/K norms, no sub-norms, no FP16 KV cache).
// Supports attention biases and tied embeddings.
// Returns s->logits on success, NULL to fall back to CPU.
static float *bn_transformer_gpu_forward_impl(BnModel *m, BnSession *sess,
                                              int token, int pos,
                                              int need_logits,
                                              int *argmax_token,
                                              const int *penalty_tokens,
                                              int n_penalty_tokens,
                                              float repeat_penalty) {
    /* no-op */
    BnConfig *c = &m->config;
    BnWeights *w = &m->weights;
    BnRunState *s = &sess->state;
    BnGPUBackend *gpu = bn_model_gpu(m);
    const BnBackendModel *backend = bn_model_backend(m);
    BnTransformerGPUEmitContext emit;
    bn_transformer_gpu_emit_context_init(&emit, NULL, 0);
    int emit_logits = need_logits || argmax_token != NULL;
    if (argmax_token && (!gpu || !gpu->argmax_activation))
        return NULL;

    BnTransformerGPUForwardPolicy policy;
    const char *reject_reason = NULL;
    if (bn_transformer_gpu_validate_forward(
            &policy, gpu, backend, c, w, token, pos, &reject_reason) != 0)
        return bn_transformer_gpu_reject_forward(&emit, reject_reason);

    int dim = c->dim;
    int kv_dim = c->kv_dim;
    int head_size = c->head_size;
    int n_heads = c->n_heads;
    int q_dim = n_heads * head_size;
    int rope_dims = c->rope_text_dims > 0
        ? c->rope_text_dims
        : (c->rope_dim_count > 0 ? c->rope_dim_count : head_size);
    int half_rope = rope_dims / 2;
    float rope_cos[half_rope], rope_sin[half_rope];
    for (int i = 0; i < half_rope; i++) {
        float angle = pos * s->rope_freq[i];
        rope_cos[i] = cosf(angle);
        rope_sin[i] = sinf(angle);
    }
    int cache_pos = pos % c->seq_len;
    int compare_attention_layer = -1;
    int compare_attention_pos = -1;
    int compare_gqa_layer = -1;
    int compare_gqa_pos = -1;
    int compare_qkv_layer = -1;
    int compare_qkv_pos = -1;
    int compare_ffn_down_layer = -1;
    int compare_ffn_down_pos = -1;
    int compare_ffn_state_layer = -1;
    int compare_ffn_state_pos = -1;
    BnTransformerGPUCPUFallbackPolicy cpu_fallback =
        bn_transformer_gpu_cpu_fallback_policy();
    BnTransformerGPUQ4Q8LayerPolicy q4_q8 =
        bn_transformer_gpu_q4_q8_layer_policy(c);
    BnTransformerGPUComparePolicy compare_policy =
        bn_transformer_gpu_compare_policy();
    int all2_q4q6_moe_gpu_route_from_layer = -1;
    int all2_q4q6_moe_gpu_route_to_layer = -1;
    {
        static int init = 0;
        static int s_all2_q4q6_moe_gpu_route_from_layer = -1;
        static int s_all2_q4q6_moe_gpu_route_to_layer = -1;
        if (!init) {
            bn_transformer_gpu_cuda_all2_q4q6_moe_route_layer_range(
                &s_all2_q4q6_moe_gpu_route_from_layer,
                &s_all2_q4q6_moe_gpu_route_to_layer);
            init = 1;
        }
        compare_attention_layer = compare_policy.attention_layer;
        compare_attention_pos = compare_policy.attention_pos;
        compare_gqa_layer = compare_policy.gqa_layer;
        compare_gqa_pos = compare_policy.gqa_pos;
        compare_qkv_layer = compare_policy.qkv_layer;
        compare_qkv_pos = compare_policy.qkv_pos;
        compare_ffn_down_layer = compare_policy.ffn_down_layer;
        compare_ffn_down_pos = compare_policy.ffn_down_pos;
        compare_ffn_state_layer = compare_policy.ffn_state_layer;
        compare_ffn_state_pos = compare_policy.ffn_state_pos;
        all2_q4q6_moe_gpu_route_from_layer =
            s_all2_q4q6_moe_gpu_route_from_layer;
        all2_q4q6_moe_gpu_route_to_layer =
            s_all2_q4q6_moe_gpu_route_to_layer;
    }
    int all2_q4q6_moe_safe_cpu_attn =
        bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_attn_fallback_enabled(
            gpu, c, w);
    int small_dense_q8_safe_cpu_attn =
        bn_transformer_gpu_cuda_small_dense_q8_cpu_attn_fallback_enabled(
            gpu, c, w);
    int large_hybrid_safe_cpu_attn =
        bn_transformer_gpu_cuda_large_hybrid_cpu_attn_safe_fallback_enabled(
            gpu, c, w);
    if (all2_q4q6_moe_safe_cpu_attn &&
        cpu_fallback.layer < 0 && cpu_fallback.from_layer < 0 &&
        cpu_fallback.attn_layer < 0 && cpu_fallback.attn_from_layer < 0)
        cpu_fallback.attn_from_layer = 0;
    if (small_dense_q8_safe_cpu_attn &&
        cpu_fallback.layer < 0 && cpu_fallback.from_layer < 0 &&
        cpu_fallback.attn_layer < 0 && cpu_fallback.attn_from_layer < 0)
        cpu_fallback.attn_from_layer = 0;
    if (large_hybrid_safe_cpu_attn &&
        cpu_fallback.layer < 0 && cpu_fallback.from_layer < 0 &&
        cpu_fallback.attn_layer < 0 && cpu_fallback.attn_from_layer < 0)
        cpu_fallback.attn_from_layer = 0;
    if (bn_transformer_gpu_cuda_large_hybrid_argmax_blocked(
            gpu, c, w, argmax_token != NULL))
        return NULL;

    // Embed token on CPU, upload to GPU x buffer.
    float emb[dim];
    bn_model_embed_token(m, emb, token);
    if (bn_transformer_gpu_write_x(gpu, emb,
                                   (size_t)dim * sizeof(float)) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "write token embedding failed");

    /* no-op */

    void *output_norm = policy.output_norm;
    BnTransformerGPULogitResources *logit_res = &policy.logits;
    int has_moe = policy.has_moe;

    // Precompute eps as uint32
    uint32_t u_eps;
    { float eps = c->norm_eps; memcpy(&u_eps, &eps, 4); }

    int max_ops = bn_transformer_gpu_graph_op_capacity(c);

    // Reuse the session-owned GPU IR/lowering storage to avoid per-token malloc.
    int command_cap = 0;
    void *command_buffer = bn_backend_session_ensure_gpu_command_buffer(
        sess->backend, max_ops, &command_cap);
    if (!command_buffer)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu graph allocation failed");
    int cacheable_resident_moe =
        policy.has_moe &&
        bn_transformer_gpu_cuda_moe_decode_cacheable(c, w, backend);
    int gpu_logits_need_cpu =
        bn_transformer_gpu_logits_needs_cpu_fallback(gpu, logit_res);
    int all2_q4q6_moe_cuda_q6_logits_refine_default =
        bn_transformer_gpu_cuda_all2_q4q6_moe_q6_logits_refine_default(
            gpu, c, w);
    int refine_q6_logits =
        bn_transformer_gpu_q6_logits_refine_enabled(
            gpu, all2_q4q6_moe_cuda_q6_logits_refine_default);
    int q6_logits_refine_captures_xb =
        bn_transformer_gpu_q6_logits_refine_captures_xb(
            logit_res, refine_q6_logits,
            all2_q4q6_moe_cuda_q6_logits_refine_default);
    int use_matvec_argmax =
        bn_transformer_gpu_matvec_argmax_enabled(
            gpu, c, logit_res, argmax_token != NULL, need_logits,
            gpu_logits_need_cpu);
    int small_dense_cuda_exact_q4_q8_default =
        bn_transformer_gpu_cuda_small_dense_exact_q4_q8_default(
            gpu, c, q4_q8.from_layer);
    int small_dense_cuda_q8_logits_refine_default =
        small_dense_cuda_exact_q4_q8_default &&
        bn_transformer_gpu_cuda_small_dense_q8_logits_refine_enabled(
            gpu, c, logit_res->type);
    int refine_q8_logits =
        bn_transformer_gpu_q8_logits_refine_enabled(
            gpu, small_dense_cuda_q8_logits_refine_default);
    int q8_logits_refine_captures_xb =
        bn_transformer_gpu_q8_logits_refine_captures_xb(
            logit_res, refine_q8_logits);
    int small_dense_cuda_exact_q4_q8_to_layer = q4_q8.to_layer;
    if (small_dense_cuda_exact_q4_q8_default &&
        small_dense_cuda_exact_q4_q8_to_layer < 0 &&
        c->n_layers > 33)
        small_dense_cuda_exact_q4_q8_to_layer = c->n_layers - 33 - 1;
    int cacheable_decode =
        bn_transformer_gpu_cuda_decode_cacheable(
            gpu, emit_logits, argmax_token != NULL, gpu_logits_need_cpu,
            policy.has_moe, cacheable_resident_moe,
            q6_logits_refine_captures_xb, q8_logits_refine_captures_xb,
            need_logits, cpu_fallback.layer, cpu_fallback.from_layer,
            cpu_fallback.attn_layer, cpu_fallback.attn_from_layer,
            cpu_fallback.ffn_layer, cpu_fallback.ffn_from_layer,
            cpu_fallback.ffn_down_from_layer, compare_attention_layer,
            compare_gqa_layer, compare_qkv_layer, compare_ffn_down_layer,
            compare_ffn_state_layer);
    int cached_n = cacheable_decode
        ? bn_backend_session_gpu_cached_op_count(sess->backend)
        : 0;
    if (argmax_token && cached_n > 0 &&
        !bn_backend_session_gpu_cached_has_logits(sess->backend) &&
        !use_matvec_argmax) {
        bn_backend_session_clear_gpu_cached_ops(sess->backend);
        cached_n = 0;
    }
    int cached_has_logits = cached_n > 0 &&
        bn_backend_session_gpu_cached_has_logits(sess->backend);
    if (cached_n > 0 && cached_n <= command_cap) {
        if (gpu_patch_cached_decode_ops((BnGPUOp *)command_buffer, cached_n,
                                        c, pos) == 0 &&
            bn_transformer_gpu_execute_ops(
                gpu, command_buffer, cached_n,
                need_logits ? BN_GPU_VALUE_LOGITS : -1,
                need_logits ? s->logits : NULL,
                need_logits ? c->vocab_size : 0) == 0) {
            if (argmax_token) {
                int argmax_rc = cached_has_logits
                    ? gpu->argmax_activation(
                          gpu->ctx, BN_GPU_VALUE_LOGITS, c->vocab_size,
                          penalty_tokens, n_penalty_tokens, repeat_penalty,
                          argmax_token)
                    : gpu->matvec_argmax_activation(
                          gpu->ctx, logit_res->gpu_buf, logit_res->type,
                          logit_res->rows, logit_res->cols, BN_GPU_VALUE_XB,
                          penalty_tokens, n_penalty_tokens, repeat_penalty,
                          argmax_token);
                if (argmax_rc != 0) {
                    bn_backend_session_clear_gpu_cached_ops(sess->backend);
                    bn_transformer_gpu_emit_context_free(&emit);
                    return NULL;
                }
            }
            bn_transformer_gpu_emit_context_free(&emit);
            return need_logits ? s->logits : s->x;
        }
        bn_backend_session_clear_gpu_cached_ops(sess->backend);
    }
    if (bn_transformer_gpu_emit_context_init_session(
            &emit, sess->backend, command_buffer, command_cap,
            max_ops * 4, max_ops) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu graph reserve failed");

    // ---- Initial RMSNorm: x -> xb (using layer 0 attn_norm) ----
    if (bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
            &emit, bn_transformer_gpu_resolve_initial_norm(backend),
            dim, u_eps) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu graph rmsnorm emit failed");

    /* no-op */

    for (int l = 0; l < c->n_layers; l++) {
        BnLayerWeights *lw = &w->layers[l];
        BnLayerShapePlan plan;
        bn_transformer_plan_layer_shape(&plan, c, lw, l, bn_model_tq_state(m) != NULL);
        int is_attn = plan.is_attn;
        BnFFNPlan layer_ffn_plan;
        int layer_ffn_plan_valid = 0;
        BnTransformerGPUDenseFFNResources layer_ffn_res = {0};
        if (!lw->moe.router_weight) {
            bn_transformer_plan_ffn(
                &layer_ffn_plan, c, lw, gpu, backend, l, 1);
            layer_ffn_plan_valid = 1;
            layer_ffn_res =
                bn_transformer_gpu_resolve_dense_ffn_resources(
                    gpu, backend, lw, l);
        }
        int use_q4_q8_layer = q4_q8.from_layer >= 0 &&
                               l >= q4_q8.from_layer &&
                               (q4_q8.to_layer < 0 ||
                                l <= q4_q8.to_layer);
        int small_dense_cuda_exact_q4_q8 =
            small_dense_cuda_exact_q4_q8_default &&
            (small_dense_cuda_exact_q4_q8_to_layer < 0 ||
             l <= small_dense_cuda_exact_q4_q8_to_layer);
        if (small_dense_cuda_exact_q4_q8)
            use_q4_q8_layer = 1;
        int all2_q4q6_moe_cuda_exact_attn =
            bn_transformer_gpu_cuda_moe_exact_attention_enabled(gpu, c);
        int use_q4_q8_attn =
            (use_q4_q8_layer || all2_q4q6_moe_cuda_exact_attn) &&
            !q4_q8.ffn_only;
        int use_q4_q8_ffn = use_q4_q8_layer && !q4_q8.attn_only;
        int use_q4_q8_ffn_down = use_q4_q8_ffn;
        if (small_dense_cuda_exact_q4_q8 &&
            !bn_transformer_gpu_cuda_small_dense_exact_q4_q8_ffn_down_enabled(
                gpu, c))
            use_q4_q8_ffn_down = 0;

        // ---- SSM layer ----
        if (!is_attn) {
            int use_cpu_ssm_fallback =
                bn_transformer_gpu_ssm_cpu_fallback_required(gpu);
            if (use_cpu_ssm_fallback) {
                void *nn = bn_transformer_gpu_resolve_next_norm(
                    backend, l, c->n_layers, output_norm);
                if (bn_transformer_gpu_fallback_ssm_layer(
                        &emit, gpu, m, sess, lw, l, dim, u_eps, nn) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu ssm cpu fallback failed");
                continue;
            }

            BnTransformerGPUSSMResources ssm_res =
                bn_transformer_gpu_resolve_ssm_resources(gpu, backend, lw, l);
            bn_transformer_gpu_emit_context_ssm(
                &emit, c, lw, &plan, &ssm_res, dim, u_eps);

            // SSM layer's FFN (dense or MoE) — same as attention layer below
            goto ffn_block;
        }

        // KV cache addressing
        int attn_idx = plan.attn_idx;
        size_t loff = (size_t)attn_idx * c->seq_len * kv_dim;
        int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
        if ((cpu_fallback.layer >= 0 && l == cpu_fallback.layer) ||
            (cpu_fallback.from_layer >= 0 && l >= cpu_fallback.from_layer)) {
            void *next_norm = bn_transformer_gpu_resolve_next_norm(
                backend, l, c->n_layers, output_norm);
            if (bn_transformer_gpu_fallback_cpu_layer(
                    &emit, gpu, m, sess, l, pos, cache_pos, rope_dims,
                    rope_cos, rope_sin, dim, u_eps, next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-layer fallback failed");
            continue;
        }

        uint32_t kv_cache_off = (uint32_t)(loff + (size_t)cache_pos * kv_dim);
        BnTransformerGPUQKVResources qkv_res =
            bn_transformer_gpu_resolve_qkv_resources(gpu, backend, lw, l);
        BnTransformerGPUAttentionResources attn_res =
            bn_transformer_gpu_resolve_attention_resources(gpu, backend, lw, l);
        if (gpu_qkv_resources_missing(lw, &plan, &qkv_res) ||
            gpu_attention_resources_missing(lw, &attn_res) ||
            (layer_ffn_plan_valid &&
             gpu_dense_ffn_resources_missing(
                 lw, &layer_ffn_plan, &layer_ffn_res))) {
            void *next_norm = bn_transformer_gpu_resolve_next_norm(
                backend, l, c->n_layers, output_norm);
            if (bn_transformer_gpu_fallback_cpu_layer(
                    &emit, gpu, m, sess, l, pos, cache_pos, rope_dims,
                    rope_cos, rope_sin, dim, u_eps, next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu missing-qweight cpu-layer fallback failed");
            continue;
        }
        if (cpu_fallback.attn_layer == l ||
            (cpu_fallback.attn_from_layer >= 0 &&
             l >= cpu_fallback.attn_from_layer)) {
            void *ffn_norm = attn_res.ffn_norm;
            if (bn_transformer_gpu_fallback_cpu_attention(
                    &emit, gpu, m, sess, lw, l, pos, cache_pos, rope_dims,
                    rope_cos, rope_sin, dim, u_eps, ffn_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-attention fallback failed");
        } else {
            int compare_attention = compare_attention_layer == l &&
                (compare_attention_pos < 0 || compare_attention_pos == pos);
            int compare_gqa = compare_gqa_layer == l &&
                (compare_gqa_pos < 0 || compare_gqa_pos == pos);
            if (compare_attention || compare_gqa) {
                if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                    bn_transformer_gpu_read_x(gpu, sess->state.x,
                                              (size_t)dim * sizeof(float)) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu attention pre-compare snapshot failed");
            }
            bn_transformer_gpu_emit_context_qkv(
                &emit, c, lw, &plan, &qkv_res, pos, q_dim,
                head_size, n_heads, kv_dim, rope_dims, kv_cache_off, u_eps,
                use_q4_q8_attn);
            if (!emit_logits && l + 1 == c->n_layers) {
                continue;
            }
            if (compare_qkv_layer == l &&
                (compare_qkv_pos < 0 || compare_qkv_pos == pos)) {
                if (bn_transformer_gpu_debug_compare_qkv(
                        &emit, gpu, m, sess, lw, l, pos, kv_cache_off,
                        dim, q_dim, kv_dim) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu qkv compare failed");
            }
            if (compare_gqa) {
                bn_transformer_gpu_emit_context_attention_gqa(
                    &emit, c, lw, &attn_res, pos, q_dim,
                    head_size, n_heads, kv_dim, rope_dims, n_kv, loff,
                    kv_cache_off, has_moe);
                if (bn_transformer_gpu_debug_compare_gqa(
                        &emit, gpu, m, sess, lw, l, pos, cache_pos,
                        rope_dims, rope_cos, rope_sin, dim) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu gqa compare failed");
                bn_transformer_gpu_emit_context_attention_finish(
                    &emit, c, lw, &attn_res, dim, q_dim, head_size, u_eps,
                    use_q4_q8_attn);
            } else {
                bn_transformer_gpu_emit_context_attention(
                    &emit, c, lw, &attn_res, pos, dim, q_dim,
                    head_size, n_heads, kv_dim, rope_dims, n_kv, loff,
                    kv_cache_off, has_moe, u_eps, use_q4_q8_attn);
            }
            if (compare_attention) {
                if (bn_transformer_gpu_debug_compare_attention(
                        &emit, gpu, m, sess, lw, l, pos, cache_pos,
                        rope_dims, rope_cos, rope_sin, dim) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu attention compare failed");
            }
        }

        // ---- FFN (MoE or dense) ----
        ffn_block:;
        if (lw->moe.router_weight) {
            int use_cpu_moe_fallback =
                bn_transformer_gpu_moe_ffn_cpu_fallback_enabled(
                    gpu, c, &lw->moe.expert_map, dim, 1, l,
                    cpu_fallback.ffn_layer, cpu_fallback.ffn_from_layer);
            if (use_cpu_moe_fallback) {
                void *moe_next_norm = bn_transformer_gpu_resolve_next_norm(
                    backend, l, c->n_layers, output_norm);
                if (bn_transformer_gpu_fallback_moe_layer(
                        &emit, gpu, m, sess, lw, l, dim, u_eps,
                        moe_next_norm) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe cpu fallback failed");
                continue;
            }

            BnGPUMoETemporaryBuffers moe_temporaries;
            void *next_norm = bn_transformer_gpu_resolve_next_norm(
                backend, l, c->n_layers, output_norm);
            BnGPUMoEResolvedExpert expert_emit[BN_MAX_MOE_K];
            BnGPUMoEResources moe_res;
            void *router_diff = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_ROUTER_DIFF);
            void *moe_gate_all = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_GATE_ALL);
            int gpu_route_all2 =
                bn_transformer_gpu_cuda_all2_moe_direct_route_enabled(
                    c, router_diff, moe_gate_all);
            if (gpu_route_all2) {
                if (gpu_resolve_moe_all2_resources(
                        &moe_res, expert_emit, m, sess, lw, l, router_diff,
                        &moe_temporaries) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe all2 resource resolution failed");
                BnTransformerGPUMoESharedResources moe_shared =
                    bn_transformer_gpu_resolve_moe_shared_resources(
                        gpu, backend, lw, l);
                bn_transformer_gpu_emit_context_moe(
                    &emit, &moe_res, &moe_shared, lw, dim, u_eps, next_norm,
                    c->moe_exact_silu);
                if (moe_temporaries.n_buffers > 0) {
                    if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu execute flush failed");
                    bn_gpu_moe_bridge_release_temporaries(m, &moe_temporaries);
                }
                continue;
            }
            int moe_route_profile =
                bn_transformer_gpu_moe_route_profile_enabled();
            double moe_prof_t0 = moe_route_profile
                ? bn_platform_time_ms() : 0.0;
            if (!sess->moe_state)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe session state missing");
            void *moe_router = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_ROUTER);
            int all2_q4q6_moe =
                bn_transformer_gpu_cuda_all2_q4q6_moe_layer_enabled(
                    gpu, c, lw, dim);
            int all2_q4q6_moe_gpu_route_layer_selected =
                bn_transformer_gpu_cuda_all2_q4q6_moe_route_layer_selected(
                    l, all2_q4q6_moe_gpu_route_from_layer,
                    all2_q4q6_moe_gpu_route_to_layer);
            int all2_q4q6_moe_exact_gpu_route =
                bn_transformer_gpu_cuda_all2_q4q6_moe_exact_gpu_route_enabled(
                    all2_q4q6_moe,
                    all2_q4q6_moe_gpu_route_layer_selected);
            moe_router = bn_transformer_gpu_cuda_all2_q4q6_moe_router(
                c, moe_router, router_diff,
                all2_q4q6_moe_gpu_route_layer_selected,
                all2_q4q6_moe_exact_gpu_route);
            void *moe_up_all = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_UP_ALL);
            void *moe_down_all = bn_backend_model_handle(
                backend, l, BN_BACKEND_HANDLE_MOE_DOWN_ALL);
            int moe_routed_q8 =
                bn_transformer_gpu_moe_routed_q8(&lw->moe.expert_map);
            uint32_t moe_route_flags = 0u;
            if (!c->moe_norm_topk_prob)
                moe_route_flags |= BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM;
            int gpu_route_topk =
                bn_transformer_gpu_cuda_moe_route_topk_enabled(
                    moe_router, all2_q4q6_moe,
                    all2_q4q6_moe_gpu_route_layer_selected);
            int cpu_route_resident_ffn =
                bn_transformer_gpu_cuda_moe_cpu_route_resident_ffn_enabled(
                    c, all2_q4q6_moe, gpu_route_topk, moe_routed_q8);
            int gpu_routed_ffn =
                bn_transformer_gpu_cuda_moe_routed_ffn_enabled(
                    gpu_route_topk, cpu_route_resident_ffn, moe_gate_all,
                    moe_up_all, moe_down_all, &lw->moe.expert_map,
                    c->moe_intermediate_size, dim);
            if (gpu_routed_ffn) {
                int all2_q4q6_moe_cpu_moe_safe =
                    bn_transformer_gpu_cuda_all2_q4q6_moe_cpu_moe_safe_default(
                        c, w);
                int override_moe_cpu_actual =
                    bn_transformer_gpu_cuda_moe_cpu_actual_override_enabled(
                        all2_q4q6_moe_cpu_moe_safe);
                int compare_moe =
                    bn_transformer_gpu_moe_compare_layer_selected(l, pos);
                float *moe_cpu_x = NULL;
                float *moe_gpu_x = NULL;
                float *moe_cpu_routed_part = NULL;
                float *moe_cpu_shared_part = NULL;
                float *moe_gpu_routed_part = NULL;
                float *moe_override_x = NULL;
                if (override_moe_cpu_actual) {
                    moe_override_x =
                        (float *)malloc((size_t)dim * sizeof(float));
                    if (!moe_override_x ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_x(
                            gpu, s->x, (size_t)dim * sizeof(float)) != 0 ||
                        bn_transformer_gpu_read_xb(
                            gpu, s->xb, (size_t)dim * sizeof(float)) != 0 ||
                        gpu_debug_compute_moe_actual_cpu_from_state(
                            m, sess, lw, l, dim, moe_override_x) != 0) {
                        free(moe_override_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe cpu override setup failed");
                    }
                }
                if (compare_moe) {
                    moe_cpu_x = (float *)malloc((size_t)dim * sizeof(float));
                    moe_gpu_x = (float *)malloc((size_t)dim * sizeof(float));
                    if (!moe_cpu_x || !moe_gpu_x ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_x(gpu, s->x,
                                                  (size_t)dim * sizeof(float)) != 0 ||
                        bn_transformer_gpu_read_xb(gpu, s->xb,
                                                   (size_t)dim * sizeof(float)) != 0) {
                        free(moe_cpu_x);
                        free(moe_gpu_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe compare setup failed");
                    }
                    if (bn_transformer_gpu_moe_compare_input_norm_enabled() &&
                        lw->norm.ffn_norm) {
                        float *moe_cpu_xb =
                            (float *)malloc((size_t)dim * sizeof(float));
                        if (!moe_cpu_xb) {
                            free(moe_cpu_x);
                            free(moe_gpu_x);
                            free(moe_override_x);
                            return bn_transformer_gpu_reject_forward(
                                &emit, "gpu routed moe compare setup failed");
                        }
                        gpu_debug_rmsnorm_scalar_local(
                            moe_cpu_xb, s->x, lw->norm.ffn_norm, dim,
                            c->norm_eps);
                        gpu_debug_compare_vec_local(
                            "moe_input_norm_compare", l, pos,
                            moe_cpu_xb, s->xb, dim);
                        free(moe_cpu_xb);
                    }
                    if (bn_transformer_gpu_moe_compare_actual_enabled()) {
                        if (gpu_debug_compute_moe_actual_cpu_from_state(
                                m, sess, lw, l, dim, moe_cpu_x) != 0) {
                            free(moe_cpu_x);
                            free(moe_gpu_x);
                            free(moe_override_x);
                            return bn_transformer_gpu_reject_forward(
                                &emit, "gpu routed moe compare setup failed");
                        }
                    } else {
                        bn_moe_route(sess->moe_state, s->xb,
                                     lw->moe.router_weight, dim,
                                     c->n_experts, c->n_experts_active,
                                     c->moe_norm_topk_prob,
                                     c->moe_expert_weights_scale,
                                     bn_model_pool(m));
                        if (gpu_debug_compute_moe_cpu_from_xb(
                                m, sess, lw, dim, s->x, s->xb,
                                moe_cpu_x) != 0) {
                            free(moe_cpu_x);
                            free(moe_gpu_x);
                            free(moe_override_x);
                            return bn_transformer_gpu_reject_forward(
                                &emit, "gpu routed moe compare setup failed");
                        }
                    }
                }
                if (cpu_route_resident_ffn) {
                    if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_xb(gpu, s->xb,
                                                   (size_t)dim * sizeof(float)) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe cpu route input readback failed");
                    bn_moe_route(sess->moe_state, s->xb,
                                 lw->moe.router_weight, dim,
                                 c->n_experts, c->n_experts_active,
                                 c->moe_norm_topk_prob,
                                 c->moe_expert_weights_scale,
                                 bn_model_pool(m));
                    float route_tmp[BN_MAX_MOE_K * 2];
                    int K = c->n_experts_active;
                    if (K > BN_MAX_MOE_K)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe route K too large");
                    for (int k = 0; k < K; k++) {
                        route_tmp[k] = sess->moe_state->expert_weights[k];
                        route_tmp[K + k] =
                            (float)sess->moe_state->expert_indices[k];
                    }
                    if (!gpu->write_activation ||
                        gpu->write_activation(
                            gpu->ctx, BN_GPU_VALUE_MOE_HB2, route_tmp,
                            (size_t)(2 * K) * sizeof(float), 0) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe cpu route upload failed");
                } else if (bn_transformer_gpu_emit_context_moe_route_topk(
                               &emit, moe_router, BN_GPU_VALUE_XB,
                               BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_HB2,
                               dim, c->n_experts, c->n_experts_active,
                               c->moe_expert_weights_scale,
                               moe_route_flags) != 0) {
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe route emit failed");
                }
                if (compare_moe &&
                    bn_transformer_gpu_moe_compare_route_enabled()) {
                    float route_tmp[BN_MAX_MOE_K * 2];
                    int K = c->n_experts_active;
                    if (K > BN_MAX_MOE_K ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB2, route_tmp,
                            (size_t)(2 * K) * sizeof(float)) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe route compare failed");
                    for (int rk = 0; rk < K; rk++) {
                        fprintf(stderr,
                                "[bn:gpu:debug] moe_route_compare layer=%d pos=%d slot=%d cpu_w=%.9g gpu_w=%.9g cpu_e=%d gpu_e=%d\n",
                                l, pos, rk,
                                sess->moe_state->expert_weights[rk],
                                route_tmp[rk],
                                sess->moe_state->expert_indices[rk],
                                (int)(route_tmp[K + rk] + 0.5f));
                    }
                }
                if (compare_moe &&
                    bn_transformer_gpu_moe_compare_raw_enabled() &&
                    moe_gate_all && moe_up_all &&
                    all2_q4q6_moe) {
                    int K = c->n_experts_active;
                    int n_experts = c->n_experts;
                    int moe_hidden = c->moe_intermediate_size;
                    size_t raw_bytes =
                        (size_t)n_experts * (size_t)moe_hidden *
                        sizeof(float);
                    float *cpu_gate = (float *)malloc(raw_bytes);
                    float *cpu_up = (float *)malloc(raw_bytes);
                    float *gpu_gate = (float *)malloc(raw_bytes);
                    float *gpu_up = (float *)malloc(raw_bytes);
                    float route_save[BN_MAX_MOE_K * 2];
                    if (!cpu_gate || !cpu_up || !gpu_gate || !gpu_up ||
                        K > BN_MAX_MOE_K ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB2, route_save,
                            (size_t)(2 * K) * sizeof(float)) != 0 ||
                        gpu_debug_compute_moe_raw_all_cpu_from_xb(
                            m, sess, lw, s->xb, cpu_gate, cpu_up) != 0 ||
                        bn_transformer_gpu_emit_context_matvec_flags(
                            &emit, lw->moe.expert_map.gate_type,
                            moe_gate_all, BN_GPU_VALUE_XB,
                            BN_GPU_VALUE_MOE_HB,
                            n_experts * moe_hidden, dim, 0,
                            BN_GPU_OP_FLAG_MATVEC_Q8K) != 0 ||
                        bn_transformer_gpu_emit_context_matvec_flags(
                            &emit, lw->moe.expert_map.up_type,
                            moe_up_all, BN_GPU_VALUE_XB,
                            BN_GPU_VALUE_MOE_HB2,
                            n_experts * moe_hidden, dim, 0,
                            BN_GPU_OP_FLAG_MATVEC_Q8K) != 0 ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB, gpu_gate,
                            raw_bytes) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB2, gpu_up,
                            raw_bytes) != 0 ||
                        !gpu->write_activation ||
                        gpu->write_activation(
                            gpu->ctx, BN_GPU_VALUE_MOE_HB2, route_save,
                            (size_t)(2 * K) * sizeof(float), 0) != 0) {
                        free(cpu_gate);
                        free(cpu_up);
                        free(gpu_gate);
                        free(gpu_up);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe raw compare failed");
                    }
                    for (int eidx = 0; eidx < n_experts; eidx++) {
                        char label[64];
                        snprintf(label, sizeof(label),
                                 "moe_raw_gate_compare[%d]", eidx);
                        gpu_debug_compare_vec_local(
                            label, l, pos,
                            cpu_gate + (size_t)eidx * (size_t)moe_hidden,
                            gpu_gate + (size_t)eidx * (size_t)moe_hidden,
                            moe_hidden);
                        snprintf(label, sizeof(label),
                                 "moe_raw_up_compare[%d]", eidx);
                        gpu_debug_compare_vec_local(
                            label, l, pos,
                            cpu_up + (size_t)eidx * (size_t)moe_hidden,
                            gpu_up + (size_t)eidx * (size_t)moe_hidden,
                            moe_hidden);
                    }
                    free(cpu_gate);
                    free(cpu_up);
                    free(gpu_gate);
                    free(gpu_up);
                }
                if (bn_transformer_gpu_emit_context_moe_routed_ffn(
                        &emit, moe_gate_all, moe_up_all, moe_down_all,
                        BN_GPU_VALUE_XB, BN_GPU_VALUE_MOE_HB2,
                        BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_OUT,
                        lw->moe.expert_map.gate_type,
                        lw->moe.expert_map.down_type, dim,
                        c->moe_intermediate_size, c->n_experts,
                        c->n_experts_active, c->moe_exact_silu, l) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe routed ffn emit failed");
                if (compare_moe &&
                    bn_transformer_gpu_moe_compare_mid_enabled()) {
                    int K = c->n_experts_active;
                    int moe_hidden = c->moe_intermediate_size;
                    size_t mid_bytes =
                        (size_t)K * (size_t)moe_hidden * sizeof(float);
                    float *moe_cpu_mid = (float *)malloc(mid_bytes);
                    float *moe_gpu_mid = (float *)malloc(mid_bytes);
                    if (!moe_cpu_mid || !moe_gpu_mid ||
                        gpu_debug_compute_moe_mid_cpu_from_xb(
                            m, sess, lw, s->xb, moe_cpu_mid) != 0 ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_HB, moe_gpu_mid,
                            mid_bytes) != 0) {
                        free(moe_cpu_mid);
                        free(moe_gpu_mid);
                        free(moe_cpu_x);
                        free(moe_gpu_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe mid compare failed");
                    }
                    for (int mk = 0; mk < K; mk++) {
                        char label[64];
                        snprintf(label, sizeof(label), "moe_mid_compare[%d]",
                                 mk);
                        gpu_debug_compare_vec_local(
                            label, l, pos,
                            moe_cpu_mid + (size_t)mk * (size_t)moe_hidden,
                            moe_gpu_mid + (size_t)mk * (size_t)moe_hidden,
                            moe_hidden);
                    }
                    free(moe_cpu_mid);
                    free(moe_gpu_mid);
                }
                if (compare_moe &&
                    bn_transformer_gpu_moe_compare_parts_enabled()) {
                    moe_cpu_routed_part =
                        (float *)malloc((size_t)dim * sizeof(float));
                    moe_cpu_shared_part =
                        (float *)malloc((size_t)dim * sizeof(float));
                    moe_gpu_routed_part =
                        (float *)malloc((size_t)dim * sizeof(float));
                    if (!moe_cpu_routed_part || !moe_cpu_shared_part ||
                        !moe_gpu_routed_part ||
                        gpu_debug_compute_moe_parts_cpu_from_xb(
                            m, sess, lw, dim, s->xb, moe_cpu_routed_part,
                            moe_cpu_shared_part) != 0 ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_OUT, moe_gpu_routed_part,
                            (size_t)dim * sizeof(float)) != 0) {
                        free(moe_cpu_routed_part);
                        free(moe_cpu_shared_part);
                        free(moe_gpu_routed_part);
                        free(moe_cpu_x);
                        free(moe_gpu_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe parts compare failed");
                    }
                    gpu_debug_compare_vec_local("moe_routed_part_compare",
                                                l, pos, moe_cpu_routed_part,
                                                moe_gpu_routed_part, dim);
                }
                int cpu_shared_fallback =
                    bn_transformer_gpu_cuda_moe_shared_cpu_fallback_enabled(
                        c->has_shared_expert &&
                        lw->shared.shared_gate.data != NULL);
                if (cpu_shared_fallback) {
                    size_t dim_bytes = (size_t)dim * sizeof(float);
                    float *shared_cpu_xb = (float *)malloc(dim_bytes);
                    float *shared_cpu_out = (float *)malloc(dim_bytes);
                    float *shared_gpu_out = (float *)malloc(dim_bytes);
                    if (!shared_cpu_xb || !shared_cpu_out ||
                        !shared_gpu_out ||
                        bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_xb(
                            gpu, shared_cpu_xb, dim_bytes) != 0 ||
                        bn_transformer_gpu_read_activation_buf(
                            gpu, BN_GPU_VALUE_MOE_OUT, shared_gpu_out,
                            dim_bytes) != 0 ||
                        gpu_compute_shared_expert_cpu_from_xb(
                            m, sess, lw, dim, shared_cpu_xb,
                            shared_cpu_out) != 0) {
                        free(shared_cpu_xb);
                        free(shared_cpu_out);
                        free(shared_gpu_out);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu shared moe cpu fallback failed");
                    }
                    for (int si = 0; si < dim; si++)
                        shared_gpu_out[si] += shared_cpu_out[si];
                    if (!gpu->write_activation ||
                        gpu->write_activation(
                            gpu->ctx, BN_GPU_VALUE_MOE_OUT,
                            shared_gpu_out, dim_bytes, 0) != 0) {
                        free(shared_cpu_xb);
                        free(shared_cpu_out);
                        free(shared_gpu_out);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu shared moe cpu fallback upload failed");
                    }
                    free(shared_cpu_xb);
                    free(shared_cpu_out);
                    free(shared_gpu_out);
                    bn_transformer_gpu_emit_context_residual_rmsnorm(
                        &emit, BN_GPU_VALUE_X, BN_GPU_VALUE_MOE_OUT,
                        BN_GPU_VALUE_XB, dim, u_eps, next_norm);
                } else if (c->has_shared_expert && lw->shared.shared_gate.data) {
                    BnTransformerGPUMoESharedResources moe_shared =
                        bn_transformer_gpu_resolve_moe_shared_resources(
                            gpu, backend, lw, l);
                    BnGPUMoEResources shared_only = {
                        &lw->moe.expert_map, NULL, 1,
                        c->moe_intermediate_size, 1
                    };
                    bn_transformer_gpu_emit_context_moe(
                        &emit, &shared_only, &moe_shared, lw, dim, u_eps,
                        next_norm, c->moe_exact_silu);
                } else {
                    bn_transformer_gpu_emit_context_residual_rmsnorm(
                        &emit, BN_GPU_VALUE_X, BN_GPU_VALUE_MOE_OUT,
                        BN_GPU_VALUE_XB, dim, u_eps, next_norm);
                }
                if (compare_moe) {
                    if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_read_x(gpu, moe_gpu_x,
                                                  (size_t)dim * sizeof(float)) != 0) {
                        free(moe_cpu_x);
                        free(moe_gpu_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe compare readback failed");
                    }
                    gpu_debug_compare_vec_local("moe_routed_state_compare",
                                                l, pos, moe_cpu_x, moe_gpu_x,
                                                dim);
                    if (moe_cpu_shared_part && moe_gpu_routed_part) {
                        float *moe_gpu_shared_part =
                            (float *)malloc((size_t)dim * sizeof(float));
                        if (moe_gpu_shared_part) {
                            for (int i = 0; i < dim; i++)
                                moe_gpu_shared_part[i] =
                                    moe_gpu_x[i] - s->x[i] -
                                    moe_gpu_routed_part[i];
                            gpu_debug_compare_vec_local(
                                "moe_shared_part_compare", l, pos,
                                moe_cpu_shared_part, moe_gpu_shared_part,
                                dim);
                        }
                        free(moe_gpu_shared_part);
                    }
                    if (bn_transformer_gpu_moe_compare_shared_mid_enabled() &&
                        c->has_shared_expert && lw->shared.shared_gate.data) {
                        int shared_hidden =
                            c->shared_expert_intermediate_size;
                        size_t shared_mid_bytes =
                            (size_t)shared_hidden * sizeof(float);
                        float *moe_cpu_shared_mid =
                            (float *)malloc(shared_mid_bytes);
                        float *moe_gpu_shared_mid =
                            (float *)malloc(shared_mid_bytes);
                        if (moe_cpu_shared_mid && moe_gpu_shared_mid &&
                            gpu_debug_compute_shared_mid_cpu_from_xb(
                                m, sess, lw, s->xb,
                                moe_cpu_shared_mid) == 0 &&
                            bn_transformer_gpu_read_activation_buf(
                                gpu, BN_GPU_VALUE_HB, moe_gpu_shared_mid,
                                shared_mid_bytes) == 0) {
                            gpu_debug_compare_vec_local(
                                "moe_shared_mid_compare", l, pos,
                                moe_cpu_shared_mid, moe_gpu_shared_mid,
                                shared_hidden);
                        }
                        free(moe_cpu_shared_mid);
                        free(moe_gpu_shared_mid);
                    }
                    if (bn_transformer_gpu_moe_compare_shared_down_enabled() &&
                        c->has_shared_expert && lw->shared.shared_gate.data) {
                        size_t shared_down_bytes =
                            (size_t)dim * sizeof(float);
                        float *moe_cpu_shared_down =
                            (float *)malloc(shared_down_bytes);
                        float *moe_gpu_shared_down =
                            (float *)malloc(shared_down_bytes);
                        if (moe_cpu_shared_down && moe_gpu_shared_down &&
                            gpu_debug_compute_shared_down_cpu_from_xb(
                                m, sess, lw, dim, s->xb,
                                moe_cpu_shared_down) == 0 &&
                            bn_transformer_gpu_read_activation_buf(
                                gpu, BN_GPU_VALUE_XB2, moe_gpu_shared_down,
                                shared_down_bytes) == 0) {
                            gpu_debug_compare_vec_local(
                                "moe_shared_down_compare", l, pos,
                                moe_cpu_shared_down, moe_gpu_shared_down,
                                dim);
                        }
                        free(moe_cpu_shared_down);
                        free(moe_gpu_shared_down);
                    }
                    if (bn_transformer_gpu_moe_compare_norm_enabled()) {
                        float *moe_cpu_norm =
                            (float *)malloc((size_t)dim * sizeof(float));
                        float *moe_gpu_norm =
                            (float *)malloc((size_t)dim * sizeof(float));
                        if (moe_cpu_norm && moe_gpu_norm &&
                            bn_transformer_gpu_read_xb(
                                gpu, moe_gpu_norm,
                                (size_t)dim * sizeof(float)) == 0) {
                            const float *nw = (l + 1 < c->n_layers)
                                ? w->layers[l + 1].norm.attn_norm
                                : w->output_norm;
                            if (nw) {
                                float ss = 0.0f;
                                for (int i = 0; i < dim; i++)
                                    ss += moe_cpu_x[i] * moe_cpu_x[i];
                                float scale = 1.0f /
                                    sqrtf(ss / (float)dim + c->norm_eps);
                                for (int i = 0; i < dim; i++)
                                    moe_cpu_norm[i] =
                                        moe_cpu_x[i] * scale * nw[i];
                                gpu_debug_compare_vec_local(
                                    "moe_routed_norm_compare", l, pos,
                                    moe_cpu_norm, moe_gpu_norm, dim);
                            }
                        }
                        free(moe_cpu_norm);
                        free(moe_gpu_norm);
                    }
                    free(moe_cpu_routed_part);
                    free(moe_cpu_shared_part);
                    free(moe_gpu_routed_part);
                    free(moe_cpu_x);
                    free(moe_gpu_x);
                }
                if (override_moe_cpu_actual) {
                    if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                        bn_transformer_gpu_write_x(
                            gpu, moe_override_x,
                            (size_t)dim * sizeof(float)) != 0 ||
                        bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
                            &emit, next_norm, dim, u_eps) != 0) {
                        free(moe_override_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu routed moe cpu override apply failed");
                    }
                }
                free(moe_override_x);
                continue;
            }
            int compare_moe =
                bn_transformer_gpu_moe_compare_layer_selected(l, pos);
            int did_gpu_route_topk = 0;
            if (gpu_route_topk) {
                if (bn_transformer_gpu_emit_context_moe_route_topk(
                        &emit, moe_router, BN_GPU_VALUE_XB,
                        BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_HB2,
                        dim, c->n_experts, c->n_experts_active,
                        c->moe_expert_weights_scale, moe_route_flags) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe route emit failed");
                if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe route topk failed");
                float route_tmp[BN_MAX_MOE_K * 2];
                int K = c->n_experts_active;
                if (K > BN_MAX_MOE_K)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe route K too large");
                if (bn_transformer_gpu_read_activation_buf(
                        gpu, BN_GPU_VALUE_MOE_HB2, route_tmp,
                        (size_t)(2 * K) * sizeof(float)) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe route readback failed");
                for (int k = 0; k < K; k++) {
                    sess->moe_state->expert_weights[k] = route_tmp[k];
                    int eidx = (int)(route_tmp[K + k] + 0.5f);
                    sess->moe_state->expert_indices[k] = eidx;
                }
                if (compare_moe &&
                    bn_transformer_gpu_moe_compare_route_enabled()) {
                    if (bn_transformer_gpu_read_xb(gpu, s->xb,
                                                   (size_t)dim *
                                                   sizeof(float)) != 0)
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe route compare input failed");
                    float cpu_weights[BN_MAX_MOE_K];
                    int cpu_indices[BN_MAX_MOE_K];
                    bn_moe_route(sess->moe_state, s->xb,
                                 lw->moe.router_weight, dim,
                                 c->n_experts, c->n_experts_active,
                                 c->moe_norm_topk_prob,
                                 c->moe_expert_weights_scale,
                                 bn_model_pool(m));
                    for (int k = 0; k < K; k++) {
                        cpu_weights[k] = sess->moe_state->expert_weights[k];
                        cpu_indices[k] = sess->moe_state->expert_indices[k];
                        sess->moe_state->expert_weights[k] = route_tmp[k];
                        sess->moe_state->expert_indices[k] =
                            (int)(route_tmp[K + k] + 0.5f);
                    }
                    for (int k = 0; k < K; k++) {
                        fprintf(stderr,
                                "[bn:gpu:debug] moe_route_compare layer=%d pos=%d slot=%d cpu_w=%.9g gpu_w=%.9g cpu_e=%d gpu_e=%d\n",
                                l, pos, k, cpu_weights[k], route_tmp[k],
                                cpu_indices[k],
                                (int)(route_tmp[K + k] + 0.5f));
                    }
                }
                did_gpu_route_topk = 1;
            } else if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0) {
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe route input readback failed");
            }
            double moe_prof_t1 = moe_route_profile
                ? bn_platform_time_ms() : 0.0;
            if (!did_gpu_route_topk &&
                bn_transformer_gpu_read_xb(gpu, s->xb,
                                           (size_t)dim * sizeof(float)) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe route input readback failed");
            double moe_prof_t2 = moe_route_profile
                ? bn_platform_time_ms() : 0.0;
            if (!did_gpu_route_topk) {
                bn_moe_route(sess->moe_state, s->xb, lw->moe.router_weight,
                             dim, c->n_experts, c->n_experts_active,
                             c->moe_norm_topk_prob,
                             c->moe_expert_weights_scale,
                             bn_model_pool(m));
            }
            double moe_prof_t3 = moe_route_profile
                ? bn_platform_time_ms() : 0.0;
            float *moe_cpu_x = NULL;
            float *moe_gpu_x = NULL;
            if (compare_moe) {
                moe_cpu_x = (float *)malloc((size_t)dim * sizeof(float));
                moe_gpu_x = (float *)malloc((size_t)dim * sizeof(float));
                if (!moe_cpu_x || !moe_gpu_x ||
                    bn_transformer_gpu_read_x(gpu, s->x,
                                              (size_t)dim * sizeof(float)) != 0 ||
                    gpu_debug_compute_moe_cpu_from_xb(
                        m, sess, lw, dim, s->x, s->xb, moe_cpu_x) != 0) {
                    free(moe_cpu_x);
                    free(moe_gpu_x);
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu moe compare setup failed");
                }
            }
            if (bn_gpu_moe_bridge_resolve_resources(
                    &moe_res, expert_emit, BN_MAX_MOE_K, m, sess, lw, l,
                    &moe_temporaries) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe resource resolution failed");
            double moe_prof_t4 = moe_route_profile
                ? bn_platform_time_ms() : 0.0;
            gpu_moe_route_profile_add(
                dim, c->n_experts, moe_prof_t1 - moe_prof_t0,
                moe_prof_t2 - moe_prof_t1, moe_prof_t3 - moe_prof_t2,
                moe_prof_t4 - moe_prof_t3);
            BnTransformerGPUMoESharedResources moe_shared =
                bn_transformer_gpu_resolve_moe_shared_resources(
                    gpu, backend, lw, l);
            bn_transformer_gpu_emit_context_moe(
                &emit, &moe_res, &moe_shared, lw, dim, u_eps, next_norm,
                c->moe_exact_silu);
            if (moe_temporaries.n_buffers > 0 || compare_moe) {
                if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0)
                    return bn_transformer_gpu_reject_forward(
                        &emit, "gpu execute flush failed");
                if (compare_moe) {
                    if (bn_transformer_gpu_read_x(gpu, moe_gpu_x,
                                                  (size_t)dim * sizeof(float)) != 0) {
                        free(moe_cpu_x);
                        free(moe_gpu_x);
                        return bn_transformer_gpu_reject_forward(
                            &emit, "gpu moe compare readback failed");
                    }
                    gpu_debug_compare_vec_local("moe_state_compare", l, pos,
                                                moe_cpu_x, moe_gpu_x, dim);
                    if (bn_transformer_gpu_moe_compare_norm_enabled()) {
                        float *moe_cpu_norm =
                            (float *)malloc((size_t)dim * sizeof(float));
                        float *moe_gpu_norm =
                            (float *)malloc((size_t)dim * sizeof(float));
                        if (moe_cpu_norm && moe_gpu_norm &&
                            bn_transformer_gpu_read_xb(
                                gpu, moe_gpu_norm,
                                (size_t)dim * sizeof(float)) == 0) {
                            const float *nw = (l + 1 < c->n_layers)
                                ? w->layers[l + 1].norm.attn_norm
                                : w->output_norm;
                            if (nw) {
                                float ss = 0.0f;
                                for (int i = 0; i < dim; i++)
                                    ss += moe_cpu_x[i] * moe_cpu_x[i];
                                float scale = 1.0f /
                                    sqrtf(ss / (float)dim + c->norm_eps);
                                for (int i = 0; i < dim; i++)
                                    moe_cpu_norm[i] =
                                        moe_cpu_x[i] * scale * nw[i];
                                gpu_debug_compare_vec_local(
                                    "moe_norm_compare", l, pos,
                                    moe_cpu_norm, moe_gpu_norm, dim);
                            }
                        }
                        free(moe_cpu_norm);
                        free(moe_gpu_norm);
                    }
                    free(moe_cpu_x);
                    free(moe_gpu_x);
                }
                bn_gpu_moe_bridge_release_temporaries(m, &moe_temporaries);
            }
            continue;  // skip dense FFN below
        }
        void *next_norm = bn_transformer_gpu_resolve_next_norm(
            backend, l, c->n_layers, output_norm);
        BnFFNPlan ffn_plan;
        if (layer_ffn_plan_valid)
            ffn_plan = layer_ffn_plan;
        else
            bn_transformer_plan_ffn(&ffn_plan, c, lw, gpu, backend, l, 1);
        if ((cpu_fallback.ffn_layer >= 0 && l == cpu_fallback.ffn_layer) ||
            (cpu_fallback.ffn_from_layer >= 0 &&
             l >= cpu_fallback.ffn_from_layer)) {
            if (bn_transformer_gpu_fallback_cpu_ffn(
                    &emit, gpu, m, sess, lw, &ffn_plan, dim, u_eps,
                    next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-ffn fallback failed");
            continue;
        }
        BnTransformerGPUDenseFFNResources ffn_res = layer_ffn_res;
        int ffn_down_input_buf = -1;
        int skip_ffn_down = cpu_fallback.ffn_down_from_layer >= 0 &&
                            l >= cpu_fallback.ffn_down_from_layer;
        int compare_ffn_state = compare_ffn_state_layer == l &&
            (compare_ffn_state_pos < 0 || compare_ffn_state_pos == pos);
        if (compare_ffn_state) {
            if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                bn_transformer_gpu_read_x(gpu, sess->state.x,
                                          (size_t)dim * sizeof(float)) != 0 ||
                bn_transformer_gpu_read_xb(gpu, sess->state.xb,
                                           (size_t)dim * sizeof(float)) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ffn-state pre-compare snapshot failed");
        }
        bn_transformer_gpu_emit_context_dense_ffn(
            &emit, c, lw, &ffn_plan, &ffn_res, dim, u_eps,
            next_norm, skip_ffn_down, &ffn_down_input_buf,
            use_q4_q8_ffn, use_q4_q8_ffn_down);
        if (!skip_ffn_down &&
            compare_ffn_down_layer == l &&
            (compare_ffn_down_pos < 0 || compare_ffn_down_pos == pos)) {
            if (bn_transformer_gpu_debug_compare_ffn_down(
                    &emit, gpu, m, sess, lw, l, pos, ffn_down_input_buf,
                    ffn_plan.hidden_dim, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ffn-down compare failed");
        }
        if (!skip_ffn_down && compare_ffn_state) {
            const float *next_norm_cpu = (l + 1 < c->n_layers)
                ? w->layers[l + 1].norm.attn_norm
                : w->output_norm;
            if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                bn_transformer_gpu_debug_compare_ffn_state(
                    &emit, gpu, m, sess, lw, &ffn_plan, next_norm_cpu,
                    l, pos, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu ffn-state compare failed");
        }
        if (skip_ffn_down) {
            if (bn_transformer_gpu_fallback_cpu_ffn_down(
                    &emit, gpu, m, sess, lw, ffn_down_input_buf,
                    ffn_plan.hidden_dim, dim, u_eps, next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-ffn-down fallback failed");
        }
    }

    // ---- Logits matvec: xb -> logits (xb is already normalized) ----
    int q6_logits_refine_has_xb_snapshot = 0;
    if (emit_logits && !use_matvec_argmax) {
        if (bn_transformer_gpu_cpu_logits_enabled(gpu_logits_need_cpu)) {
            if (argmax_token)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu argmax requires gpu logits");
            if (bn_transformer_gpu_fallback_logits(
                    &emit, gpu, m, sess, logit_res, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu logits cpu fallback failed");
            return s->logits;
        }
        if (need_logits && !argmax_token && q6_logits_refine_captures_xb) {
            if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                bn_transformer_gpu_read_xb(gpu, s->xb,
                                           (size_t)dim * sizeof(float)) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu logits pre-refine snapshot failed");
            q6_logits_refine_has_xb_snapshot = 1;
        }
        if (bn_transformer_gpu_emit_context_logits(
                &emit, logit_res->gpu_buf, logit_res->type,
                logit_res->rows, logit_res->cols) != 0)
            return bn_transformer_gpu_reject_forward(
                &emit, "gpu graph logits emit failed");
    }

    // Safety: verify we didn't overflow the ops array
    if (emit.n + emit.graph->n_ops > max_ops)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu op graph capacity exceeded");

    // Execute final batch (logits + any remaining layer ops).
    if (bn_transformer_gpu_emit_context_lower_pending(&emit) != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu final lower failed");
    int final_n = emit.n;
    int rc = bn_transformer_gpu_execute_ops(
        gpu, emit.lowered_ops, emit.n,
        need_logits ? BN_GPU_VALUE_LOGITS : -1,
        need_logits ? s->logits : NULL,
        need_logits ? c->vocab_size : 0);
    if (rc != 0)
        return bn_transformer_gpu_reject_forward(
            &emit, "gpu final execute failed");
    if (cacheable_decode && final_n > 0)
        bn_backend_session_set_gpu_cached_op_count(sess->backend, final_n,
                                                   emit_logits &&
                                                   !use_matvec_argmax);
    if (argmax_token) {
        if (!use_matvec_argmax &&
            q8_logits_refine_captures_xb) {
            int refine_top = bn_transformer_gpu_q8_logits_refine_top(
                small_dense_cuda_q8_logits_refine_default);
            if (refine_top > 0 &&
                gpu->read_activation &&
                gpu->read_activation(gpu->ctx, BN_GPU_VALUE_LOGITS,
                                     s->logits,
                                     (size_t)c->vocab_size * sizeof(float),
                                     0) == 0 &&
                bn_transformer_gpu_read_xb(gpu, s->xb,
                                           (size_t)dim * sizeof(float)) == 0) {
                gpu_refine_q8_logits_top(s->logits, c->vocab_size,
                                         logit_res->cpu_weight, s->xb,
                                         s->x_q, refine_top);
                int best = 0;
                float best_v = -INFINITY;
                for (int i = 0; i < c->vocab_size; i++) {
                    float v = s->logits[i];
                    if (repeat_penalty != 1.0f && penalty_tokens &&
                        n_penalty_tokens > 0) {
                        for (int j = 0; j < n_penalty_tokens; j++) {
                            if (penalty_tokens[j] == i) {
                                v = v > 0.0f ? v / repeat_penalty
                                             : v * repeat_penalty;
                                break;
                            }
                        }
                    }
                    if (v > best_v) {
                        best_v = v;
                        best = i;
                    }
                }
                *argmax_token = best;
                bn_transformer_gpu_emit_context_free(&emit);
                return s->x;
            }
        }
        int argmax_rc = use_matvec_argmax
            ? gpu->matvec_argmax_activation(
                  gpu->ctx, logit_res->gpu_buf, logit_res->type,
                  logit_res->rows, logit_res->cols, BN_GPU_VALUE_XB,
                  penalty_tokens, n_penalty_tokens, repeat_penalty,
                  argmax_token)
            : gpu->argmax_activation(
                  gpu->ctx, BN_GPU_VALUE_LOGITS, c->vocab_size,
                  penalty_tokens, n_penalty_tokens, repeat_penalty,
                  argmax_token);
        if (argmax_rc != 0)
            return bn_transformer_gpu_reject_forward(
                &emit, "gpu argmax failed");
        if (bn_transformer_gpu_debug_argmax_compare_enabled() &&
            gpu->read_activation && c->vocab_size > 0) {
            float *dbg_logits =
                (float *)malloc((size_t)c->vocab_size * sizeof(float));
            if (dbg_logits &&
                gpu->read_activation(gpu->ctx, BN_GPU_VALUE_LOGITS,
                                     dbg_logits,
                                     (size_t)c->vocab_size * sizeof(float),
                                     0) == 0) {
                int cpu_argmax = 0;
                float cpu_best = -INFINITY;
                for (int i = 0; i < c->vocab_size; i++) {
                    float v = dbg_logits[i];
                    if (repeat_penalty != 1.0f && penalty_tokens &&
                        n_penalty_tokens > 0) {
                        for (int j = 0; j < n_penalty_tokens; j++) {
                            if (penalty_tokens[j] == i) {
                                v = v > 0.0f ? v / repeat_penalty
                                             : v * repeat_penalty;
                                break;
                            }
                        }
                    }
                    if (v > cpu_best) {
                        cpu_best = v;
                        cpu_argmax = i;
                    }
                }
                fprintf(stderr,
                        "[bn:gpu:argmax:cmp] cuda=%d cpu=%d cpu_logit=%.6g\n",
                        *argmax_token, cpu_argmax, cpu_best);
            }
            free(dbg_logits);
        }
        bn_transformer_gpu_emit_context_free(&emit);
        return s->x;
    }
    if (!need_logits) {
        bn_transformer_gpu_emit_context_free(&emit);
        return s->x;
    }
    if (q6_logits_refine_captures_xb) {
        int refine_top = bn_transformer_gpu_q6_logits_refine_top(
            all2_q4q6_moe_cuda_q6_logits_refine_default);
        int has_xb = q6_logits_refine_has_xb_snapshot;
        if (!has_xb && refine_top > 0 &&
            bn_transformer_gpu_read_xb(gpu, s->xb,
                                       (size_t)dim * sizeof(float)) == 0)
            has_xb = 1;
        if (refine_top > 0 && has_xb) {
            gpu_refine_q6k_logits_top(s->logits, c->vocab_size,
                                      logit_res->cpu_weight, s->xb,
                                      s->x_q, refine_top);
        }
    }
    if (q8_logits_refine_captures_xb) {
        int refine_top = bn_transformer_gpu_q8_logits_refine_top(
            small_dense_cuda_q8_logits_refine_default);
        if (refine_top > 0 &&
            bn_transformer_gpu_read_xb(gpu, s->xb,
                                       (size_t)dim * sizeof(float)) == 0) {
            gpu_refine_q8_logits_top(s->logits, c->vocab_size,
                                     logit_res->cpu_weight, s->xb,
                                     s->x_q, refine_top);
        }
    }
    if (bn_transformer_gpu_compare_logits_enabled()) {
        float *cpu_logits = (float *)malloc((size_t)c->vocab_size *
                                            sizeof(float));
        if (cpu_logits &&
            bn_transformer_gpu_read_xb(gpu, s->xb,
                                       (size_t)dim * sizeof(float)) == 0) {
            bn_quant_matvec(cpu_logits, logit_res->cpu_weight, s->xb,
                            s->x_q, bn_model_pool(m));
            double sum_abs = 0.0;
            double sum_sq = 0.0;
            float max_abs = 0.0f;
            int max_i = 0;
            for (int i = 0; i < c->vocab_size; i++) {
                float diff = fabsf(s->logits[i] - cpu_logits[i]);
                sum_abs += (double)diff;
                sum_sq += (double)diff * (double)diff;
                if (diff > max_abs) {
                    max_abs = diff;
                    max_i = i;
                }
            }
            fprintf(stderr,
                    "[bn:gpu:debug] logits_compare pos=%d max_abs=%.9g "
                    "max_i=%d cpu=%.9g gpu=%.9g mean_abs=%.9g rms=%.9g\n",
                    pos, max_abs, max_i, cpu_logits[max_i],
                    s->logits[max_i], sum_abs / (double)c->vocab_size,
                    sqrt(sum_sq / (double)c->vocab_size));
        }
        free(cpu_logits);
    }
    bn_transformer_gpu_emit_context_free(&emit);
    #undef GPU_LEGACY_OPS
    return s->logits;
}

float *bn_transformer_gpu_forward(BnModel *m, BnSession *sess,
                                  int token, int pos) {
    return bn_transformer_gpu_forward_impl(m, sess, token, pos, 1,
                                           NULL, NULL, 0, 1.0f);
}

float *bn_transformer_gpu_forward_no_logits(BnModel *m, BnSession *sess,
                                            int token, int pos) {
    return bn_transformer_gpu_forward_impl(m, sess, token, pos, 0,
                                           NULL, NULL, 0, 1.0f);
}

int bn_transformer_gpu_forward_argmax(BnModel *m, BnSession *sess,
                                      int token, int pos,
                                      const int *penalty_tokens,
                                      int n_penalty_tokens,
                                      float repeat_penalty,
                                      int *out_token) {
    if (!out_token) return -1;
    float *state = bn_transformer_gpu_forward_impl(
        m, sess, token, pos, 0, out_token, penalty_tokens,
        n_penalty_tokens, repeat_penalty);
    return state ? 0 : -1;
}
