#include "gpu_internal.h"
#include "../gpu_shader.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_cpu_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_plan_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "model_internal.h"
#include "../moe_internal.h"
#include "moe.h"
#include "platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BN_GPU_LOGITS_REFINE_MAX_SCALE_BLOCKS 8192

static void fallback_rmsnorm(float *out,
                             const float *x,
                             const float *w,
                             int size,
                             float eps) {
    bn_transformer_rmsnorm_default(out, x, w, size, eps);
}

static void fallback_cpu_matvec_batch(const BnModel *m,
                                      const BnMatvecTask *tasks,
                                      int n_tasks,
                                      const float *x,
                                      int8_t *quantized_buf) {
    BnMatvecTask inline_tasks[8];
    BnMatvecTask *prepared = inline_tasks;
    if (n_tasks > 8) {
        prepared = (BnMatvecTask *)malloc((size_t)n_tasks * sizeof(*prepared));
        if (!prepared) {
            bn_transformer_cpu_quant_matvec_batch(tasks, n_tasks, x,
                                                  quantized_buf,
                                                  bn_model_pool(m));
            return;
        }
    }
    for (int i = 0; i < n_tasks; i++) {
        BnTransformerCPUMatvecResourcePolicy resource =
            bn_transformer_cpu_matvec_resource_policy(
                &m->config, bn_model_backend(m), tasks[i].W);
        prepared[i] = tasks[i];
        prepared[i].prepared = resource.prepared;
        prepared[i].flags |= resource.task_flags;
    }
    bn_transformer_cpu_quant_matvec_batch(prepared, n_tasks, x, quantized_buf,
                                          bn_model_pool(m));
    if (prepared != inline_tasks)
        free(prepared);
}

static void fallback_cpu_matvec(const BnModel *m,
                                float *out,
                                const BnQWeight *W,
                                const float *x,
                                int8_t *quantized_buf) {
    BnMatvecTask task = { out, W, NULL, 0 };
    fallback_cpu_matvec_batch(m, &task, 1, x, quantized_buf);
}

void bn_transformer_gpu_cpu_quant_matvec_batch_model(
    const BnModel *model,
    const BnMatvecTask *tasks,
    int n_tasks,
    const float *x,
    int8_t *quantized_buf) {
    bn_transformer_cpu_quant_matvec_batch(
        tasks, n_tasks, x, quantized_buf, bn_model_pool(model));
}

void bn_transformer_gpu_cpu_quant_matvec_model(
    const BnModel *model,
    float *out,
    const BnQWeight *weight,
    const float *x,
    int8_t *quantized_buf) {
    bn_transformer_cpu_quant_matvec(
        out, weight, x, quantized_buf, bn_model_pool(model));
}

int bn_transformer_gpu_refine_kquant_logits_top(
    float *logits,
    int n_logits,
    const BnQWeight *weight,
    const float *x,
    int8_t *quantized,
    int top_n) {
    if (!logits || !weight || !weight->data || !x || !quantized ||
        top_n <= 0)
        return 0;
    if (top_n > 4096) top_n = 4096;
    if (top_n > n_logits) top_n = n_logits;
    int n_blocks =
        bn_transformer_gpu_kquant_logits_refine_blocks_per_row(weight->cols);
    int n_block_sums =
        bn_transformer_gpu_kquant_logits_refine_block_sums_per_row(n_blocks);
    if (n_blocks < 1 ||
        n_block_sums > BN_GPU_LOGITS_REFINE_MAX_SCALE_BLOCKS)
        return 0;
    float scales[n_blocks];
    int16_t block_sums[n_block_sums];
    bn_transformer_cpu_prepare_kquant_activation(
        x, quantized, scales, block_sums, weight->cols);

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
        if (bn_transformer_cpu_refine_kquant_logits_prepared_activation_row(
                weight, quantized, scales, block_sums, ids[i],
                &row_sum) == 0)
            logits[ids[i]] = row_sum;
    }
    return n_top;
}

int bn_transformer_gpu_refine_native_quant_logits_top(
    float *logits,
    int n_logits,
    const BnQWeight *weight,
    const float *x,
    int8_t *quantized,
    int top_n) {
    if (!logits || !weight || !weight->data || !x || !quantized ||
        !bn_transformer_cpu_has_native_quant_activation() || top_n <= 0)
        return 0;
    if (top_n > 128) top_n = 128;
    if (top_n > n_logits) top_n = n_logits;
    int n_blocks = weight->cols / 32;
    if (n_blocks <= 0 ||
        n_blocks > BN_GPU_LOGITS_REFINE_MAX_SCALE_BLOCKS)
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

    float scales[BN_GPU_LOGITS_REFINE_MAX_SCALE_BLOCKS];
    if (bn_transformer_cpu_quantize_native_logits_refine_activation(
            x, quantized, scales, weight->cols) != 0)
        return 0;
    for (int i = 0; i < n_top; i++) {
        float row_sum;
        if (bn_transformer_cpu_refine_native_logits_row(
                weight, quantized, scales, ids[i], &row_sum) == 0)
            logits[ids[i]] = row_sum;
    }
    return n_top;
}

int bn_transformer_gpu_try_refined_argmax(
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    const BnTransformerGPULogitResources *logits,
    const BnTransformerGPULogitsRefinePolicy *refine,
    int dim,
    const int *penalty_tokens,
    int n_penalty_tokens,
    float repeat_penalty,
    int *out_token) {
    if (!gpu || !model || !session || !logits || !refine ||
        !out_token || dim <= 0 || !refine->native_quant_captures_xb ||
        refine->native_quant_refine_top <= 0 ||
        model->config.vocab_size <= 0)
        return 0;
    BnRunState *state = &session->state;
    int vocab_size = model->config.vocab_size;
    if (bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_LOGITS, state->logits,
            (size_t)vocab_size * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_xb(
            gpu, state->xb, (size_t)dim * sizeof(float)) != 0)
        return 0;
    bn_transformer_gpu_refine_native_quant_logits_top(
        state->logits, vocab_size, logits->cpu_weight,
        state->xb, state->x_q, refine->native_quant_refine_top);
    int best = 0;
    float best_v = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        float v = state->logits[i];
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
    *out_token = best;
    return 1;
}

void bn_transformer_gpu_refine_output_logits(
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    const BnTransformerGPULogitResources *logits,
    const BnTransformerGPULogitsRefinePolicy *refine,
    int dim,
    int kquant_has_xb_snapshot) {
    if (!gpu || !model || !session || !logits || !refine ||
        dim <= 0 || model->config.vocab_size <= 0)
        return;
    BnRunState *state = &session->state;
    int vocab_size = model->config.vocab_size;
    if (refine->kquant_captures_xb) {
        int refine_top = refine->kquant_refine_top;
        int has_xb = kquant_has_xb_snapshot;
        if (!has_xb && refine_top > 0 &&
            bn_transformer_gpu_read_xb(
                gpu, state->xb, (size_t)dim * sizeof(float)) == 0)
            has_xb = 1;
        if (refine_top > 0 && has_xb)
            bn_transformer_gpu_refine_kquant_logits_top(
                state->logits, vocab_size, logits->cpu_weight,
                state->xb, state->x_q, refine_top);
    }
    if (refine->native_quant_captures_xb) {
        int refine_top = refine->native_quant_refine_top;
        if (refine_top > 0 &&
            bn_transformer_gpu_read_xb(
                gpu, state->xb, (size_t)dim * sizeof(float)) == 0)
            bn_transformer_gpu_refine_native_quant_logits_top(
                state->logits, vocab_size, logits->cpu_weight,
                state->xb, state->x_q, refine_top);
    }
}

const void *bn_transformer_gpu_model_expert_projection(
    BnModel *model,
    BnMoEState *state,
    const BnMoEExpertMap *map,
    int expert,
    int projection) {
    return bn_moe_get_expert_proj(
        bn_model_moe_io(model), state, map, expert, projection);
}

void bn_transformer_gpu_route_model_moe(
    BnModel *model,
    BnMoEState *state,
    const float *input,
    const BnLayerWeights *layer,
    int total_experts,
    int active_experts,
    int normalize_topk,
    float expert_weights_scale) {
    if (!model || !state || !input || !layer)
        return;
    bn_moe_route(state, input, layer->moe.router_weight,
                 model->config.dim, total_experts, active_experts,
                 normalize_topk, expert_weights_scale, bn_model_pool(model));
}

int bn_transformer_gpu_resolve_moe_route(
    BnTransformerGPUMoERouteResolution *resolution,
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnTransformerGPUMoEExecutionPolicy *route_policy,
    const BnTransformerGPUMoEDecodeRoutePolicy *route,
    const BnTransformerGPUMoEDebugPolicy *debug,
    int layer_index,
    int pos,
    int dim,
    int profile_enabled,
    const char **reason) {
    if (reason)
        *reason = "gpu moe route resolution failed";
    if (!resolution || !emit || !gpu || !model || !session ||
        !session->moe_state || !layer || !route_policy || !route ||
        !debug || dim <= 0)
        return -1;
    memset(resolution, 0, sizeof(*resolution));
    double t0 = profile_enabled ? bn_platform_time_ms() : 0.0;
    int used_gpu_topk = 0;
    if (route->gpu_route_topk) {
        if (bn_transformer_gpu_emit_context_moe_route_topk(
                emit, route->router, BN_GPU_VALUE_XB,
                BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_HB2,
                dim, route_policy->total_experts,
                route_policy->active_experts,
                route_policy->expert_weights_scale,
                route->route_flags) != 0) {
            if (reason) *reason = "gpu moe route emit failed";
            return -1;
        }
        if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0) {
            if (reason) *reason = "gpu moe route topk failed";
            return -1;
        }
        float route_tmp[BN_MAX_MOE_K * 2];
        int K = route_policy->active_experts;
        if (K > BN_MAX_MOE_K) {
            if (reason) *reason = "gpu moe route K too large";
            return -1;
        }
        if (bn_transformer_gpu_read_activation_buf(
                gpu, BN_GPU_VALUE_MOE_HB2, route_tmp,
                (size_t)(2 * K) * sizeof(float)) != 0) {
            if (reason) *reason = "gpu moe route readback failed";
            return -1;
        }
        for (int k = 0; k < K; k++) {
            session->moe_state->expert_weights[k] = route_tmp[k];
            session->moe_state->expert_indices[k] =
                (int)(route_tmp[K + k] + 0.5f);
        }
        if (debug->compare_route) {
            BnRunState *state = &session->state;
            if (bn_transformer_gpu_read_xb(
                    gpu, state->xb, (size_t)dim * sizeof(float)) != 0) {
                if (reason) *reason = "gpu moe route compare input failed";
                return -1;
            }
            float cpu_weights[BN_MAX_MOE_K];
            int cpu_indices[BN_MAX_MOE_K];
            bn_transformer_gpu_route_model_moe(
                model, session->moe_state, state->xb, layer,
                route_policy->total_experts,
                route_policy->active_experts,
                route_policy->normalize_topk,
                route_policy->expert_weights_scale);
            for (int k = 0; k < K; k++) {
                cpu_weights[k] = session->moe_state->expert_weights[k];
                cpu_indices[k] = session->moe_state->expert_indices[k];
                session->moe_state->expert_weights[k] = route_tmp[k];
                session->moe_state->expert_indices[k] =
                    (int)(route_tmp[K + k] + 0.5f);
            }
            for (int k = 0; k < K; k++) {
                fprintf(stderr,
                        "[bn:gpu:debug] moe_route_compare layer=%d pos=%d "
                        "slot=%d cpu_w=%.9g gpu_w=%.9g cpu_e=%d gpu_e=%d\n",
                        layer_index, pos, k, cpu_weights[k], route_tmp[k],
                        cpu_indices[k],
                        (int)(route_tmp[K + k] + 0.5f));
            }
        }
        used_gpu_topk = 1;
    } else if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0) {
        if (reason) *reason = "gpu moe route input readback failed";
        return -1;
    }
    double t1 = profile_enabled ? bn_platform_time_ms() : 0.0;
    if (!used_gpu_topk &&
        bn_transformer_gpu_read_xb(
            gpu, session->state.xb, (size_t)dim * sizeof(float)) != 0) {
        if (reason) *reason = "gpu moe route input readback failed";
        return -1;
    }
    double t2 = profile_enabled ? bn_platform_time_ms() : 0.0;
    if (!used_gpu_topk)
        bn_transformer_gpu_route_model_moe(
            model, session->moe_state, session->state.xb, layer,
            route_policy->total_experts, route_policy->active_experts,
            route_policy->normalize_topk,
            route_policy->expert_weights_scale);
    double t3 = profile_enabled ? bn_platform_time_ms() : 0.0;
    resolution->flush_ms = t1 - t0;
    resolution->read_ms = t2 - t1;
    resolution->route_ms = t3 - t2;
    return 0;
}

int bn_transformer_gpu_prepare_routed_moe_route(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnTransformerGPUMoEExecutionPolicy *route_policy,
    const BnTransformerGPUMoEDecodeRoutePolicy *route,
    const BnTransformerGPUMoEDebugPolicy *debug,
    int layer_index,
    int pos,
    int dim,
    const char **reason) {
    if (reason)
        *reason = "gpu routed moe route preparation failed";
    if (!emit || !gpu || !model || !session || !session->moe_state ||
        !layer || !route_policy || !route || !debug || dim <= 0)
        return -1;
    if (route->cpu_route_resident_ffn) {
        if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
            bn_transformer_gpu_read_xb(
                gpu, session->state.xb,
                (size_t)dim * sizeof(float)) != 0) {
            if (reason) *reason = "gpu moe cpu route input readback failed";
            return -1;
        }
        bn_transformer_gpu_route_model_moe(
            model, session->moe_state, session->state.xb, layer,
            route_policy->total_experts, route_policy->active_experts,
            route_policy->normalize_topk,
            route_policy->expert_weights_scale);
        float route_values[BN_MAX_MOE_K * 2];
        int active_experts = route_policy->active_experts;
        if (active_experts < 0 || active_experts > BN_MAX_MOE_K) {
            if (reason) *reason = "gpu moe route K too large";
            return -1;
        }
        for (int k = 0; k < active_experts; k++) {
            route_values[k] = session->moe_state->expert_weights[k];
            route_values[active_experts + k] =
                (float)session->moe_state->expert_indices[k];
        }
        if (bn_transformer_gpu_write_activation_buf(
                gpu, BN_GPU_VALUE_MOE_HB2, route_values,
                (size_t)(2 * active_experts) * sizeof(float)) != 0) {
            if (reason) *reason = "gpu moe cpu route upload failed";
            return -1;
        }
    } else if (bn_transformer_gpu_emit_context_moe_route_topk(
                   emit, route->router, BN_GPU_VALUE_XB,
                   BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_HB2,
                   dim, route_policy->total_experts,
                   route_policy->active_experts,
                   route_policy->expert_weights_scale,
                   route->route_flags) != 0) {
        if (reason) *reason = "gpu moe route emit failed";
        return -1;
    }
    if (debug->compare_route) {
        float route_values[BN_MAX_MOE_K * 2];
        int active_experts = route_policy->active_experts;
        if (active_experts < 0 || active_experts > BN_MAX_MOE_K ||
            bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
            bn_transformer_gpu_read_activation_buf(
                gpu, BN_GPU_VALUE_MOE_HB2, route_values,
                (size_t)(2 * active_experts) * sizeof(float)) != 0) {
            if (reason) *reason = "gpu moe route compare failed";
            return -1;
        }
        for (int k = 0; k < active_experts; k++) {
            fprintf(stderr,
                    "[bn:gpu:debug] moe_route_compare layer=%d pos=%d "
                    "slot=%d cpu_w=%.9g gpu_w=%.9g cpu_e=%d gpu_e=%d\n",
                    layer_index, pos, k,
                    session->moe_state->expert_weights[k],
                    route_values[k],
                    session->moe_state->expert_indices[k],
                    (int)(route_values[active_experts + k] + 0.5f));
        }
    }
    return 0;
}

void bn_transformer_gpu_discard_moe_layer_comparison(
    BnTransformerGPUMoELayerComparison *comparison) {
    if (!comparison)
        return;
    free(comparison->cpu_state);
    free(comparison->gpu_state);
    memset(comparison, 0, sizeof(*comparison));
}

int bn_transformer_gpu_prepare_moe_layer_comparison(
    BnTransformerGPUMoELayerComparison *comparison,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnTransformerGPUMoEDebugPolicy *debug,
    int dim) {
    if (!comparison)
        return -1;
    memset(comparison, 0, sizeof(*comparison));
    if (!debug || !debug->compare_layer)
        return 0;
    if (!gpu || !model || !session || !layer || dim <= 0)
        return -1;
    comparison->cpu_state =
        (float *)malloc((size_t)dim * sizeof(float));
    comparison->gpu_state =
        (float *)malloc((size_t)dim * sizeof(float));
    comparison->enabled = 1;
    comparison->compare_norm = debug->compare_norm;
    BnRunState *state = &session->state;
    if (!comparison->cpu_state || !comparison->gpu_state ||
        bn_transformer_gpu_read_x(
            gpu, state->x, (size_t)dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_fallback_moe_output(
            model, session, layer, dim, state->x, state->xb,
            comparison->cpu_state) != 0) {
        bn_transformer_gpu_discard_moe_layer_comparison(comparison);
        return -1;
    }
    return 0;
}

int bn_transformer_gpu_complete_moe_layer_comparison(
    BnTransformerGPUMoELayerComparison *comparison,
    const BnGPUBackend *gpu,
    BnModel *model,
    int layer_index,
    int pos,
    int dim,
    float norm_eps) {
    if (!comparison || !comparison->enabled || !gpu || !model || dim <= 0)
        return -1;
    if (bn_transformer_gpu_read_x(
            gpu, comparison->gpu_state,
            (size_t)dim * sizeof(float)) != 0) {
        bn_transformer_gpu_discard_moe_layer_comparison(comparison);
        return -1;
    }
    bn_transformer_gpu_debug_compare_vec(
        "moe_state_compare", layer_index, pos,
        comparison->cpu_state, comparison->gpu_state, dim);
    if (comparison->compare_norm) {
        float *cpu_norm = (float *)malloc((size_t)dim * sizeof(float));
        float *gpu_norm = (float *)malloc((size_t)dim * sizeof(float));
        if (cpu_norm && gpu_norm &&
            bn_transformer_gpu_read_xb(
                gpu, gpu_norm, (size_t)dim * sizeof(float)) == 0) {
            const float *norm_weight =
                layer_index + 1 < model->config.n_layers
                    ? model->weights.layers[layer_index + 1].norm.attn_norm
                    : model->weights.output_norm;
            if (norm_weight) {
                float ss = 0.0f;
                for (int i = 0; i < dim; i++)
                    ss += comparison->cpu_state[i] *
                          comparison->cpu_state[i];
                float scale =
                    1.0f / sqrtf(ss / (float)dim + norm_eps);
                for (int i = 0; i < dim; i++)
                    cpu_norm[i] =
                        comparison->cpu_state[i] * scale * norm_weight[i];
                bn_transformer_gpu_debug_compare_vec(
                    "moe_norm_compare", layer_index, pos,
                    cpu_norm, gpu_norm, dim);
            }
        }
        free(cpu_norm);
        free(gpu_norm);
    }
    bn_transformer_gpu_discard_moe_layer_comparison(comparison);
    return 0;
}

void bn_transformer_gpu_run_model_moe_cpu(
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    int layer_index) {
    if (!model || !session || !layer)
        return;
    bn_model_set_gpu_disabled(model, 1);
    bn_moe_forward(model, session, layer, layer_index);
    bn_model_set_gpu_disabled(model, 0);
}

int bn_transformer_gpu_fallback_moe_output_from_state(
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    int layer_index,
    int dim,
    float *output) {
    if (!model || !session || !layer || !output || dim <= 0 ||
        !session->moe_state)
        return -1;
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&model->config);
    int active_experts = route_policy.active_experts;
    if (active_experts < 0)
        return -1;
    if (active_experts > BN_MAX_MOE_K)
        active_experts = BN_MAX_MOE_K;

    BnRunState *state = &session->state;
    float *saved_x = (float *)malloc((size_t)dim * sizeof(float));
    float *saved_xb = (float *)malloc((size_t)dim * sizeof(float));
    int saved_indices[BN_MAX_MOE_K];
    float saved_weights[BN_MAX_MOE_K];
    if (!saved_x || !saved_xb) {
        free(saved_x);
        free(saved_xb);
        return -1;
    }
    memcpy(saved_x, state->x, (size_t)dim * sizeof(float));
    memcpy(saved_xb, state->xb, (size_t)dim * sizeof(float));
    for (int k = 0; k < active_experts; k++) {
        saved_indices[k] = session->moe_state->expert_indices[k];
        saved_weights[k] = session->moe_state->expert_weights[k];
    }

    bn_transformer_gpu_run_model_moe_cpu(
        model, session, layer, layer_index);
    memcpy(output, state->x, (size_t)dim * sizeof(float));

    memcpy(state->x, saved_x, (size_t)dim * sizeof(float));
    memcpy(state->xb, saved_xb, (size_t)dim * sizeof(float));
    for (int k = 0; k < active_experts; k++) {
        session->moe_state->expert_indices[k] = saved_indices[k];
        session->moe_state->expert_weights[k] = saved_weights[k];
    }
    free(saved_x);
    free(saved_xb);
    return 0;
}

static int fallback_moe_expert_projection_weight(
    BnQWeight *weight,
    BnModel *model,
    BnMoEState *state,
    const BnMoEExpertMap *map,
    int expert,
    int projection) {
    const void *data = bn_transformer_gpu_model_expert_projection(
        model, state, map, expert, projection);
    return data &&
           bn_moe_expert_projection_weight(weight, data, map, projection);
}

int bn_transformer_gpu_fallback_moe_mid(
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const float *input,
    float *mid_out) {
    if (!model || !session || !layer || !input || !mid_out ||
        !session->moe_state)
        return -1;
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&model->config);
    BnTransformerGPUMoEActivationPolicy activation_policy =
        bn_transformer_gpu_moe_activation_policy(&model->config);
    int active_experts = route_policy.active_experts;
    int hidden = route_policy.expert_hidden_dim;
    if (active_experts < 0 || hidden <= 0)
        return -1;
    float *gate = (float *)malloc((size_t)hidden * sizeof(float));
    float *up = (float *)malloc((size_t)hidden * sizeof(float));
    if (!gate || !up) {
        free(gate);
        free(up);
        return -1;
    }

    BnMoEState *moe_state = session->moe_state;
    const BnMoEExpertMap *map = &layer->moe.expert_map;
    uint32_t task_flags =
        bn_transformer_gpu_moe_gateup_task_flags(&model->config);
    for (int k = 0; k < active_experts; k++) {
        float *expert_mid = mid_out + (size_t)k * (size_t)hidden;
        int expert = moe_state->expert_indices[k];
        if (expert < 0) {
            memset(expert_mid, 0, (size_t)hidden * sizeof(float));
            continue;
        }
        BnQWeight gate_weight;
        BnQWeight up_weight;
        if (!fallback_moe_expert_projection_weight(
                &gate_weight, model, moe_state, map, expert, 0) ||
            !fallback_moe_expert_projection_weight(
                &up_weight, model, moe_state, map, expert, 1)) {
            free(gate);
            free(up);
            return -1;
        }
        BnMatvecTask tasks[2] = {
            { gate, &gate_weight, NULL, task_flags },
            { up, &up_weight, NULL, task_flags },
        };
        bn_transformer_gpu_cpu_quant_matvec_batch_model(
            model, tasks, 2, input, session->state.x_q);
        bn_moe_swiglu(gate, gate, up, hidden,
                      activation_policy.uses_reference_silu);
        memcpy(expert_mid, gate, (size_t)hidden * sizeof(float));
    }

    free(gate);
    free(up);
    return 0;
}

int bn_transformer_gpu_fallback_moe_raw_gate_up(
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const float *input,
    float *gate_out,
    float *up_out) {
    if (!model || !session || !layer || !input || !gate_out || !up_out ||
        !session->moe_state)
        return -1;
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&model->config);
    int total_experts = route_policy.total_experts;
    int hidden = route_policy.expert_hidden_dim;
    if (total_experts <= 0 || hidden <= 0)
        return -1;
    BnMoEState *moe_state = session->moe_state;
    const BnMoEExpertMap *map = &layer->moe.expert_map;
    uint32_t task_flags =
        bn_transformer_gpu_moe_gateup_task_flags(&model->config);
    for (int expert = 0; expert < total_experts; expert++) {
        BnQWeight gate_weight;
        BnQWeight up_weight;
        if (!fallback_moe_expert_projection_weight(
                &gate_weight, model, moe_state, map, expert, 0) ||
            !fallback_moe_expert_projection_weight(
                &up_weight, model, moe_state, map, expert, 1))
            return -1;
        BnMatvecTask tasks[2] = {
            {
                gate_out + (size_t)expert * (size_t)hidden,
                &gate_weight,
                NULL,
                task_flags
            },
            {
                up_out + (size_t)expert * (size_t)hidden,
                &up_weight,
                NULL,
                task_flags
            },
        };
        bn_transformer_gpu_cpu_quant_matvec_batch_model(
            model, tasks, 2, input, session->state.x_q);
    }
    return 0;
}

static int fallback_moe_routed_output(
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    int dim,
    const float *input,
    float *output) {
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&model->config);
    BnTransformerGPUMoEActivationPolicy activation_policy =
        bn_transformer_gpu_moe_activation_policy(&model->config);
    int active_experts = route_policy.active_experts;
    int hidden = route_policy.expert_hidden_dim;
    if (active_experts < 0 || hidden <= 0)
        return -1;
    float *gate = (float *)malloc((size_t)hidden * sizeof(float));
    float *up = (float *)malloc((size_t)hidden * sizeof(float));
    float *down = (float *)malloc((size_t)dim * sizeof(float));
    if (!gate || !up || !down) {
        free(gate);
        free(up);
        free(down);
        return -1;
    }
    memset(output, 0, (size_t)dim * sizeof(float));

    BnMoEState *moe_state = session->moe_state;
    const BnMoEExpertMap *map = &layer->moe.expert_map;
    uint32_t task_flags =
        bn_transformer_gpu_moe_gateup_task_flags(&model->config);
    for (int k = 0; k < active_experts; k++) {
        int expert = moe_state->expert_indices[k];
        if (expert < 0)
            continue;
        BnQWeight gate_weight;
        BnQWeight up_weight;
        BnQWeight down_weight;
        if (!fallback_moe_expert_projection_weight(
                &gate_weight, model, moe_state, map, expert, 0) ||
            !fallback_moe_expert_projection_weight(
                &up_weight, model, moe_state, map, expert, 1) ||
            !fallback_moe_expert_projection_weight(
                &down_weight, model, moe_state, map, expert, 2)) {
            free(gate);
            free(up);
            free(down);
            return -1;
        }
        BnMatvecTask tasks[2] = {
            { gate, &gate_weight, NULL, task_flags },
            { up, &up_weight, NULL, task_flags },
        };
        bn_transformer_gpu_cpu_quant_matvec_batch_model(
            model, tasks, 2, input, session->state.x_q);
        bn_moe_swiglu(gate, gate, up, hidden,
                      activation_policy.uses_reference_silu);
        bn_transformer_gpu_cpu_quant_matvec_model(
            model, down, &down_weight, gate, session->state.x_q);
        bn_moe_weighted_add(
            output, down, moe_state->expert_weights[k], dim);
    }

    free(gate);
    free(up);
    free(down);
    return 0;
}

int bn_transformer_gpu_fallback_moe_output(
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    int dim,
    const float *residual,
    const float *input,
    float *output) {
    if (!model || !session || !layer || !residual || !input || !output ||
        dim <= 0 || !session->moe_state)
        return -1;
    float *routed = (float *)malloc((size_t)dim * sizeof(float));
    if (!routed)
        return -1;
    if (fallback_moe_routed_output(
            model, session, layer, dim, input, routed) != 0) {
        free(routed);
        return -1;
    }
    if (bn_transformer_gpu_moe_has_loaded_shared_expert(
            &model->config, layer)) {
        float *shared = (float *)malloc((size_t)dim * sizeof(float));
        if (!shared ||
            bn_transformer_gpu_fallback_shared_expert_output(
                model, session, layer, dim, input, shared) != 0) {
            free(shared);
            free(routed);
            return -1;
        }
        bn_moe_residual_add(routed, shared, dim);
        free(shared);
    }
    for (int i = 0; i < dim; i++)
        output[i] = residual[i] + routed[i];
    free(routed);
    return 0;
}

int bn_transformer_gpu_fallback_moe_parts(
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    int dim,
    const float *input,
    float *routed_out,
    float *shared_out) {
    if (!model || !session || !layer || !input || !routed_out ||
        !shared_out || dim <= 0 || !session->moe_state)
        return -1;
    if (fallback_moe_routed_output(
            model, session, layer, dim, input, routed_out) != 0)
        return -1;
    memset(shared_out, 0, (size_t)dim * sizeof(float));
    if (bn_transformer_gpu_moe_has_loaded_shared_expert(
            &model->config, layer) &&
        bn_transformer_gpu_fallback_shared_expert_output(
            model, session, layer, dim, input, shared_out) != 0)
        return -1;
    return 0;
}

int bn_transformer_gpu_fallback_shared_expert_mid(
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const float *input,
    float *mid_out) {
    if (!model || !session || !layer || !input || !mid_out ||
        !bn_transformer_gpu_moe_has_loaded_shared_expert(
            &model->config, layer))
        return -1;
    BnTransformerGPUMoESharedExpertShapePolicy shared_shape =
        bn_transformer_gpu_moe_shared_expert_shape_policy(&model->config);
    BnTransformerGPUMoEActivationPolicy activation_policy =
        bn_transformer_gpu_moe_activation_policy(&model->config);
    int hidden = shared_shape.hidden_dim;
    if (hidden <= 0)
        return -1;
    float *up = (float *)malloc((size_t)hidden * sizeof(float));
    if (!up)
        return -1;

    BnMatvecTask tasks[2];
    int n_tasks = bn_moe_shared_expert_gateup_tasks(
        tasks, mid_out, up, layer,
        bn_transformer_gpu_moe_gateup_task_flags(&model->config));
    if (n_tasks <= 0) {
        free(up);
        return -1;
    }
    bn_transformer_gpu_cpu_quant_matvec_batch_model(
        model, tasks, n_tasks, input, session->state.x_q);
    bn_moe_swiglu(mid_out, mid_out, up, hidden,
                  activation_policy.uses_reference_silu);
    free(up);
    return 0;
}

int bn_transformer_gpu_fallback_shared_expert_output(
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    int dim,
    const float *input,
    float *output) {
    if (!model || !session || !layer || !input || !output || dim <= 0 ||
        !bn_transformer_gpu_moe_has_loaded_shared_expert(
            &model->config, layer))
        return -1;
    BnTransformerGPUMoESharedExpertShapePolicy shared_shape =
        bn_transformer_gpu_moe_shared_expert_shape_policy(&model->config);
    int hidden = shared_shape.hidden_dim;
    if (hidden <= 0)
        return -1;
    float *mid = (float *)malloc((size_t)hidden * sizeof(float));
    float *down = (float *)malloc((size_t)dim * sizeof(float));
    if (!mid || !down) {
        free(mid);
        free(down);
        return -1;
    }
    if (bn_transformer_gpu_fallback_shared_expert_mid(
            model, session, layer, input, mid) != 0) {
        free(mid);
        free(down);
        return -1;
    }
    const BnQWeight *weight = bn_moe_shared_expert_down_weight(layer);
    if (!weight) {
        free(mid);
        free(down);
        return -1;
    }
    bn_transformer_gpu_cpu_quant_matvec_model(
        model, down, weight, mid, session->state.x_q);
    memset(output, 0, (size_t)dim * sizeof(float));
    bn_moe_weighted_add(
        output, down,
        bn_moe_shared_expert_gate_weight(layer, input, dim), dim);
    free(mid);
    free(down);
    return 0;
}

int bn_transformer_gpu_fallback_shared_expert_down(
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    int dim,
    const float *input,
    float *down_out) {
    if (!model || !session || !layer || !input || !down_out || dim <= 0 ||
        !bn_transformer_gpu_moe_has_loaded_shared_expert(
            &model->config, layer))
        return -1;
    BnTransformerGPUMoESharedExpertShapePolicy shared_shape =
        bn_transformer_gpu_moe_shared_expert_shape_policy(&model->config);
    int hidden = shared_shape.hidden_dim;
    if (hidden <= 0)
        return -1;
    float *mid = (float *)malloc((size_t)hidden * sizeof(float));
    if (!mid)
        return -1;
    if (bn_transformer_gpu_fallback_shared_expert_mid(
            model, session, layer, input, mid) != 0) {
        free(mid);
        return -1;
    }
    const BnQWeight *weight = bn_moe_shared_expert_down_weight(layer);
    if (!weight) {
        free(mid);
        return -1;
    }
    bn_transformer_gpu_cpu_quant_matvec_model(
        model, down_out, weight, mid, session->state.x_q);
    free(mid);
    return 0;
}

int bn_transformer_gpu_fallback_ssm_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    bn_transformer_cpu_forward_ssm_block(m, sess, lw, layer);
    bn_transformer_cpu_residual_add(s->x, s->xb, dim);
    BnTransformerGPULayerKindPolicy layer_kind =
        bn_transformer_gpu_layer_kind_policy(lw);
    if (layer_kind.uses_moe)
        bn_moe_forward(m, sess, lw, layer);
    else
        bn_transformer_cpu_forward_ffn_block(m, sess, lw, layer, sess->pos, NULL);
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

int bn_transformer_gpu_fallback_moe_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    bn_model_set_gpu_disabled(m, 1);
    bn_moe_forward(m, sess, lw, layer);
    bn_model_set_gpu_disabled(m, 0);
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

int bn_transformer_gpu_fallback_cpu_layer(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    bn_model_set_gpu_disabled(m, 1);
    int rc = bn_transformer_cpu_forward_layer(m, sess, layer, pos, cache_pos,
                                              rope_dims, rope_cos, rope_sin);
    bn_model_set_gpu_disabled(m, 0);
    if (rc != 0)
        return -1;
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

int bn_transformer_gpu_fallback_cpu_attention(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnLayerShapePlan shape;
    bn_transformer_plan_layer_shape(&shape, c, lw, layer,
                                    bn_model_tq_state(m) != NULL);
    if (!shape.is_attn || shape.q_gated ||
        !bn_transformer_kv_host_float_cache_rows_available(c) ||
        lw->norm.attn_sub_norm)
        return -1;

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;

    int head_size = shape.head_size;
    int n_heads = shape.n_heads;
    int kv_dim = shape.kv_dim;
    int n_kv_heads = shape.n_kv_heads;
    int kv_mul = shape.kv_mul;
    int layer_rope_dims = rope_dims > head_size ? head_size : rope_dims;
    size_t loff = (size_t)shape.attn_idx * c->seq_len * c->kv_dim;
    float *key_cache_row =
        s->key_cache + loff + (size_t)cache_pos * c->kv_dim;
    float *value_cache_row =
        s->value_cache + loff + (size_t)cache_pos * c->kv_dim;

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim,
                     bn_transformer_gpu_norm_epsilon(c));
    {
        BnMatvecTask qkv[3] = {
            { s->q, &lw->attn.wq, NULL, 0 },
            { key_cache_row, &lw->attn.wk, NULL, 0 },
            { value_cache_row, &lw->attn.wv, NULL, 0 },
        };
        fallback_cpu_matvec_batch(m, qkv, 3, s->xb, s->x_q);
    }

    if (lw->attn.q_bias) {
        for (int i = 0; i < shape.q_dim; i++) s->q[i] += lw->attn.q_bias[i];
    }
    if (lw->attn.k_bias) {
        for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->attn.k_bias[i];
    }
    if (lw->attn.v_bias) {
        for (int i = 0; i < kv_dim; i++)
            value_cache_row[i] += lw->attn.v_bias[i];
    }
    if (lw->attn.q_norm) {
        for (int h = 0; h < n_heads; h++)
            fallback_rmsnorm(s->q + (size_t)h * head_size,
                             s->q + (size_t)h * head_size,
                             lw->attn.q_norm + (size_t)h * shape.qk_stride,
                             head_size, bn_transformer_gpu_norm_epsilon(c));
    }
    if (lw->attn.k_norm) {
        for (int h = 0; h < n_kv_heads; h++)
            fallback_rmsnorm(key_cache_row + (size_t)h * head_size,
                             key_cache_row + (size_t)h * head_size,
                             lw->attn.k_norm + (size_t)h * shape.qk_stride,
                             head_size, bn_transformer_gpu_norm_epsilon(c));
    }

    bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);
    bn_transformer_cpu_apply_rope_heads(key_cache_row, n_kv_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);

    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
    BnGQACtx gctx = {
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len,
        bn_transformer_attention_scale(c, head_size),
        bn_transformer_kv_host_cache_uses_fp16_rows(c)
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);

    {
        BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL, 0 }};
        fallback_cpu_matvec_batch(m, wo, 1, s->xb, s->x_q);
    }
    bn_transformer_cpu_residual_add(s->x, s->xb2, dim);

    size_t kv_row_bytes = (size_t)c->kv_dim * sizeof(float);
    size_t kv_row_off = (loff + (size_t)cache_pos * (size_t)c->kv_dim) *
                        sizeof(float);
    if (bn_transformer_gpu_write_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, key_cache_row, kv_row_bytes,
            kv_row_off) != 0 ||
        bn_transformer_gpu_write_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, value_cache_row, kv_row_bytes,
            kv_row_off) != 0)
        return -1;

    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

static void fallback_cpu_forward_ffn_from_xb(BnModel *m,
                                             BnSession *sess,
                                             BnLayerWeights *lw,
                                             const BnFFNPlan *ffn_plan,
                                             int dim) {
    BnRunState *s = &sess->state;
    int hidden_dim = ffn_plan->hidden_dim;
    if (ffn_plan->has_gate) {
        BnMatvecTask ffn[2] = {
            { s->hb, &lw->ffn.ffn_gate, NULL, 0 },
            { s->hb2, &lw->ffn.ffn_up, NULL, 0 },
        };
        fallback_cpu_matvec_batch(m, ffn, 2, s->xb, s->x_q);
    } else {
        fallback_cpu_matvec(m, s->hb, &lw->ffn.ffn_up, s->xb, s->x_q);
    }
    bn_transformer_cpu_apply_ffn_activation(s, ffn_plan, hidden_dim, 0);
    if (ffn_plan->has_sub_norm)
        fallback_rmsnorm(s->hb, s->hb, lw->norm.ffn_sub_norm,
                         hidden_dim,
                         bn_transformer_gpu_norm_epsilon(&m->config));
    fallback_cpu_matvec(m, s->xb, &lw->ffn.ffn_down, s->hb, s->x_q);
    if (ffn_plan->use_post_norm)
        fallback_rmsnorm(s->xb, s->xb, lw->norm.ffn_post_norm,
                         dim, bn_transformer_gpu_norm_epsilon(&m->config));
    bn_transformer_cpu_residual_add(s->x, s->xb, dim);
}

int bn_transformer_gpu_fallback_cpu_ffn(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const BnFFNPlan *ffn_plan,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    if (bn_transformer_gpu_read_xb(gpu, s->xb,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    fallback_cpu_forward_ffn_from_xb(m, sess, lw, ffn_plan, dim);
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

int bn_transformer_gpu_fallback_cpu_ffn_down(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int down_input_buf,
    int hidden_dim,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    if (bn_transformer_gpu_read_activation_buf(
            gpu, down_input_buf, s->hb,
            (size_t)hidden_dim * sizeof(float)) != 0)
        return -1;
    fallback_cpu_matvec(m, s->xb, &lw->ffn.ffn_down, s->hb, s->x_q);
    bn_transformer_cpu_residual_add(s->x, s->xb, dim);
    if (bn_transformer_gpu_write_x(gpu, s->x,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    return bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
}

int bn_transformer_gpu_debug_compare_ffn_down(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int down_input_buf,
    int hidden_dim,
    int dim) {
    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_activation_buf(
            gpu, down_input_buf, s->hb,
            (size_t)hidden_dim * sizeof(float)) != 0)
        return -1;
    if (bn_transformer_gpu_read_xb2(gpu, s->xb2,
                                    (size_t)dim * sizeof(float)) != 0)
        return -1;

    BnTransformerGPUDenseFFNProjectionLayout ffn_layout;
    if (!bn_transformer_gpu_resolve_dense_ffn_projection_layout(
            &ffn_layout, lw))
        return -1;
    fallback_cpu_matvec(m, s->xb, &lw->ffn.ffn_down, s->hb, s->x_q);

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int max_i = 0;
    for (int i = 0; i < dim; i++) {
        float diff = fabsf(s->xb2[i] - s->xb[i]);
        sum_abs += (double)diff;
        sum_sq += (double)diff * (double)diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
    }
    fprintf(stderr,
            "[bn:gpu:debug] ffn_down_compare layer=%d pos=%d type=%d "
            "rows=%d cols=%d max_abs=%.9g max_i=%d cpu=%.9g gpu=%.9g "
            "mean_abs=%.9g rms=%.9g\n",
            layer, pos, ffn_layout.down_type, ffn_layout.down_rows,
            ffn_layout.down_cols, max_abs, max_i, s->xb[max_i],
            s->xb2[max_i], sum_abs / (double)dim,
            sqrt(sum_sq / (double)dim));
    return 0;
}

void bn_transformer_gpu_debug_compare_vec(const char *label,
                                          int layer,
                                          int pos,
                                          const float *cpu,
                                          const float *gpu,
                                          int n) {
    if (!label || !cpu || !gpu || n <= 0)
        return;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int max_i = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(gpu[i] - cpu[i]);
        sum_abs += (double)diff;
        sum_sq += (double)diff * (double)diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
    }
    fprintf(stderr,
            "[bn:gpu:debug] %s layer=%d pos=%d "
            "max_abs=%.9g max_i=%d cpu=%.9g gpu=%.9g "
            "mean_abs=%.9g rms=%.9g\n",
            label, layer, pos, max_abs, max_i, cpu[max_i], gpu[max_i],
            sum_abs / (double)n, sqrt(sum_sq / (double)n));
}

void bn_transformer_gpu_debug_rmsnorm(float *out,
                                      const float *x,
                                      const float *w,
                                      int n,
                                      float eps) {
    if (!out || !x || !w || n <= 0)
        return;
    bn_transformer_rmsnorm_scalar(out, x, w, n, eps);
}

void bn_transformer_gpu_moe_route_profile_add(int dim,
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
    if ((calls % (unsigned long long)every) != 0)
        return;
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

void bn_transformer_gpu_debug_compare_argmax(
    const BnGPUBackend *gpu,
    int vocab_size,
    const int *penalty_tokens,
    int n_penalty_tokens,
    float repeat_penalty,
    int gpu_argmax) {
    if (!bn_transformer_gpu_debug_argmax_compare_enabled() ||
        !gpu || vocab_size <= 0)
        return;
    float *logits = (float *)malloc((size_t)vocab_size * sizeof(float));
    if (!logits)
        return;
    if (bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_LOGITS, logits,
            (size_t)vocab_size * sizeof(float)) == 0) {
        int cpu_argmax = 0;
        float cpu_best = -INFINITY;
        for (int i = 0; i < vocab_size; i++) {
            float v = logits[i];
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
                "[bn:gpu:argmax:cmp] gpu=%d cpu=%d cpu_logit=%.6g\n",
                gpu_argmax, cpu_argmax, cpu_best);
    }
    free(logits);
}

void bn_transformer_gpu_debug_compare_logits(
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    const BnTransformerGPULogitResources *logits,
    int pos,
    int dim) {
    if (!bn_transformer_gpu_compare_logits_enabled() ||
        !gpu || !model || !session || !logits ||
        model->config.vocab_size <= 0 || dim <= 0)
        return;
    BnRunState *state = &session->state;
    int vocab_size = model->config.vocab_size;
    float *cpu_logits =
        (float *)malloc((size_t)vocab_size * sizeof(float));
    if (!cpu_logits)
        return;
    if (bn_transformer_gpu_read_xb(
            gpu, state->xb, (size_t)dim * sizeof(float)) == 0) {
        bn_transformer_gpu_cpu_quant_matvec_model(
            model, cpu_logits, logits->cpu_weight, state->xb, state->x_q);
        double sum_abs = 0.0;
        double sum_sq = 0.0;
        float max_abs = 0.0f;
        int max_i = 0;
        for (int i = 0; i < vocab_size; i++) {
            float diff = fabsf(state->logits[i] - cpu_logits[i]);
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
                state->logits[max_i], sum_abs / (double)vocab_size,
                sqrt(sum_sq / (double)vocab_size));
    }
    free(cpu_logits);
}

static const BnPreparedWeight *debug_prepared_qweight(BnModel *m,
                                                      const BnQWeight *w) {
    if (!m || !w)
        return NULL;
    BnTransformerCPUMatvecResourcePolicy resource =
        bn_transformer_cpu_matvec_resource_policy(
            &m->config, bn_model_backend(m), w);
    return resource.prepared;
}

static void debug_quant_matvec_prepared(BnModel *m,
                                        float *out,
                                        const BnQWeight *W,
                                        const float *x,
                                        int8_t *quantized_buf) {
    bn_transformer_cpu_quant_matvec_prepared_flags(
        out, W, debug_prepared_qweight(m, W), x, quantized_buf,
        bn_model_pool(m), 0);
}

static void debug_compare_native_quant_activation(const BnGPUBackend *gpu,
                                                 int layer,
                                                 int pos,
                                                 const float *x,
                                                 int cols) {
    if (!gpu || !x || cols <= 0 || (cols % 32) != 0 ||
        !bn_transformer_cpu_has_native_quant_activation())
        return;
    int n_blocks = cols / 32;
    int8_t *cpu_q = (int8_t *)malloc((size_t)cols);
    int8_t *gpu_q = (int8_t *)malloc((size_t)cols);
    float *cpu_scales = (float *)malloc((size_t)n_blocks * sizeof(float));
    float *gpu_scales = (float *)malloc((size_t)n_blocks * sizeof(float));
    if (!cpu_q || !gpu_q || !cpu_scales || !gpu_scales) {
        free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
        return;
    }

    if (bn_transformer_cpu_quantize_native_logits_refine_activation(
            x, cpu_q, cpu_scales, cols) != 0) {
        free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
        return;
    }
    if (bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_DEBUG_BUF_NATIVE_QUANT_ACT, gpu_q,
            (size_t)cols) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_DEBUG_BUF_NATIVE_QUANT_SCALE, gpu_scales,
            (size_t)n_blocks * sizeof(float)) != 0) {
        free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
        return;
    }

    int max_q_abs = 0;
    int max_q_i = 0;
    long long sum_q_abs = 0;
    int n_q_diff = 0;
    for (int i = 0; i < cols; i++) {
        int diff = (int)gpu_q[i] - (int)cpu_q[i];
        int ad = diff < 0 ? -diff : diff;
        if (ad > max_q_abs) {
            max_q_abs = ad;
            max_q_i = i;
        }
        if (ad) n_q_diff++;
        sum_q_abs += ad;
    }

    double sum_scale_abs = 0.0;
    double sum_scale_sq = 0.0;
    float max_scale_abs = 0.0f;
    int max_scale_i = 0;
    for (int i = 0; i < n_blocks; i++) {
        float diff = fabsf(gpu_scales[i] - cpu_scales[i]);
        sum_scale_abs += (double)diff;
        sum_scale_sq += (double)diff * (double)diff;
        if (diff > max_scale_abs) {
            max_scale_abs = diff;
            max_scale_i = i;
        }
    }

    fprintf(stderr,
            "[bn:gpu:debug] native_quant_act_compare layer=%d pos=%d "
            "q_max_abs=%d q_max_i=%d cpu_q=%d gpu_q=%d "
            "q_diff_count=%d q_mean_abs=%.9g "
            "scale_max_abs=%.9g scale_max_i=%d cpu_scale=%.9g "
            "gpu_scale=%.9g scale_mean_abs=%.9g scale_rms=%.9g\n",
            layer, pos, max_q_abs, max_q_i, (int)cpu_q[max_q_i],
            (int)gpu_q[max_q_i], n_q_diff,
            (double)sum_q_abs / (double)cols, max_scale_abs, max_scale_i,
            cpu_scales[max_scale_i], gpu_scales[max_scale_i],
            sum_scale_abs / (double)n_blocks,
            sqrt(sum_scale_sq / (double)n_blocks));

    free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
}

int bn_transformer_gpu_debug_compare_ffn_state(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const BnFFNPlan *ffn_plan,
    const float *next_norm,
    int layer,
    int pos,
    int dim) {
    BnRunState *s = &sess->state;
    float *cpu_x_in = (float *)malloc((size_t)dim * sizeof(float));
    float *cpu_xb_in = (float *)malloc((size_t)dim * sizeof(float));
    float *cpu_x = (float *)malloc((size_t)dim * sizeof(float));
    float *cpu_xb = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_x = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_xb = (float *)malloc((size_t)dim * sizeof(float));
    float *cpu_hb = NULL;
    float *cpu_hb2 = NULL;
    float *gpu_hb = NULL;
    int hidden_dim = ffn_plan ? ffn_plan->hidden_dim : 0;
    if (hidden_dim > 0) {
        cpu_hb = (float *)malloc((size_t)hidden_dim * sizeof(float));
        cpu_hb2 = (float *)malloc((size_t)hidden_dim * sizeof(float));
        gpu_hb = (float *)malloc((size_t)hidden_dim * sizeof(float));
    }
    if (!cpu_x_in || !cpu_xb_in || !cpu_x || !cpu_xb || !gpu_x || !gpu_xb ||
        (hidden_dim > 0 && (!cpu_hb || !cpu_hb2 || !gpu_hb))) {
        free(cpu_x_in);
        free(cpu_xb_in);
        free(cpu_x);
        free(cpu_xb);
        free(gpu_x);
        free(gpu_xb);
        free(cpu_hb);
        free(cpu_hb2);
        free(gpu_hb);
        return -1;
    }

    memcpy(cpu_x_in, s->x, (size_t)dim * sizeof(float));
    memcpy(cpu_xb_in, s->xb, (size_t)dim * sizeof(float));

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0) {
        free(cpu_x_in);
        free(cpu_xb_in);
        free(cpu_x);
        free(cpu_xb);
        free(gpu_x);
        free(gpu_xb);
        free(cpu_hb);
        free(cpu_hb2);
        free(gpu_hb);
        return -1;
    }

    memcpy(s->x, cpu_x_in, (size_t)dim * sizeof(float));
    memcpy(s->xb, cpu_xb_in, (size_t)dim * sizeof(float));
    if (hidden_dim > 0) {
        if (ffn_plan->has_gate) {
            debug_quant_matvec_prepared(m, cpu_hb, &lw->ffn.ffn_gate,
                                        s->xb, s->x_q);
            debug_quant_matvec_prepared(m, cpu_hb2, &lw->ffn.ffn_up,
                                        s->xb, s->x_q);
            for (int i = 0; i < hidden_dim; i++) {
                float g = cpu_hb[i];
                if (bn_transformer_gpu_ffn_activation_kind(
                        ffn_plan->activation) == BN_GPU_IR_ACTIVATION_RELU2) {
                    float r = g > 0.0f ? g : 0.0f;
                    cpu_hb[i] = r * r * cpu_hb2[i];
                } else {
                    cpu_hb[i] = (g / (1.0f + expf(-g))) * cpu_hb2[i];
                }
            }
        } else {
            debug_quant_matvec_prepared(m, cpu_hb, &lw->ffn.ffn_up,
                                        s->xb, s->x_q);
            for (int i = 0; i < hidden_dim; i++) {
                float v = cpu_hb[i];
                if (bn_transformer_gpu_ffn_activation_kind(
                        ffn_plan->activation) == BN_GPU_IR_ACTIVATION_RELU2) {
                    float r = v > 0.0f ? v : 0.0f;
                    cpu_hb[i] = r * r;
                } else {
                    cpu_hb[i] = v / (1.0f + expf(-v));
                }
            }
        }
    }
    fallback_cpu_forward_ffn_from_xb(m, sess, lw, ffn_plan, dim);
    memcpy(cpu_x, s->x, (size_t)dim * sizeof(float));
    if (next_norm)
        fallback_rmsnorm(cpu_xb, cpu_x, next_norm, dim,
                         bn_transformer_gpu_norm_epsilon(&m->config));

    if ((hidden_dim > 0 &&
         bn_transformer_gpu_read_activation_buf(
             gpu, BN_GPU_VALUE_HB, gpu_hb,
             (size_t)hidden_dim * sizeof(float)) != 0) ||
        bn_transformer_gpu_read_x(gpu, gpu_x,
                                  (size_t)dim * sizeof(float)) != 0 ||
        (next_norm && bn_transformer_gpu_read_xb(gpu, gpu_xb,
                                                 (size_t)dim * sizeof(float)) != 0)) {
        free(cpu_x_in);
        free(cpu_xb_in);
        free(cpu_x);
        free(cpu_xb);
        free(gpu_x);
        free(gpu_xb);
        free(cpu_hb);
        free(cpu_hb2);
        free(gpu_hb);
        return -1;
    }
    if (hidden_dim > 0)
        bn_transformer_gpu_debug_compare_vec(
            "ffn_hidden_compare", layer, pos, cpu_hb, gpu_hb, hidden_dim);
    bn_transformer_gpu_debug_compare_vec(
        "ffn_state_compare", layer, pos, cpu_x, gpu_x, dim);
    if (next_norm)
        bn_transformer_gpu_debug_compare_vec(
            "ffn_next_norm_compare", layer, pos, cpu_xb, gpu_xb, dim);

    memcpy(s->x, gpu_x, (size_t)dim * sizeof(float));
    free(cpu_x_in);
    free(cpu_xb_in);
    free(cpu_x);
    free(cpu_xb);
    free(gpu_x);
    free(gpu_xb);
    free(cpu_hb);
    free(cpu_hb2);
    free(gpu_hb);
    return 0;
}

int bn_transformer_gpu_debug_compare_attention(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnLayerShapePlan shape;
    bn_transformer_plan_layer_shape(&shape, c, lw, layer,
                                    bn_model_tq_state(m) != NULL);
    if (!shape.is_attn || shape.q_gated ||
        !bn_transformer_kv_host_float_cache_rows_available(c) ||
        lw->norm.attn_sub_norm)
        return -1;

    float *cpu_in = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_x = (float *)malloc((size_t)dim * sizeof(float));
    if (!cpu_in || !gpu_x) {
        free(cpu_in);
        free(gpu_x);
        return -1;
    }
    memcpy(cpu_in, s->x, (size_t)dim * sizeof(float));

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0) {
        free(cpu_in);
        free(gpu_x);
        return -1;
    }
    if (bn_transformer_gpu_read_x(gpu, gpu_x,
                                  (size_t)dim * sizeof(float)) != 0) {
        free(cpu_in);
        free(gpu_x);
        return -1;
    }

    int head_size = shape.head_size;
    int n_heads = shape.n_heads;
    int kv_dim = shape.kv_dim;
    int n_kv_heads = shape.n_kv_heads;
    int kv_mul = shape.kv_mul;
    int layer_rope_dims = rope_dims > head_size ? head_size : rope_dims;
    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
    size_t loff = (size_t)shape.attn_idx * c->seq_len * c->kv_dim;
    size_t kv_bytes = (size_t)n_kv * c->kv_dim * sizeof(float);
    size_t kv_off = loff * sizeof(float);

    if (bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, s->key_cache + loff, kv_bytes,
            kv_off) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, s->value_cache + loff, kv_bytes,
            kv_off) != 0) {
        free(cpu_in);
        free(gpu_x);
        return -1;
    }

    memcpy(s->x, cpu_in, (size_t)dim * sizeof(float));
    float *key_cache_row =
        s->key_cache + loff + (size_t)cache_pos * c->kv_dim;
    float *value_cache_row =
        s->value_cache + loff + (size_t)cache_pos * c->kv_dim;

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim,
                     bn_transformer_gpu_norm_epsilon(c));
    fallback_cpu_matvec(m, s->q, &lw->attn.wq, s->xb, s->x_q);
    fallback_cpu_matvec(m, key_cache_row, &lw->attn.wk, s->xb, s->x_q);
    fallback_cpu_matvec(m, value_cache_row, &lw->attn.wv, s->xb, s->x_q);

    if (lw->attn.q_bias) {
        for (int i = 0; i < shape.q_dim; i++) s->q[i] += lw->attn.q_bias[i];
    }
    if (lw->attn.k_bias) {
        for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->attn.k_bias[i];
    }
    if (lw->attn.v_bias) {
        for (int i = 0; i < kv_dim; i++)
            value_cache_row[i] += lw->attn.v_bias[i];
    }
    if (lw->attn.q_norm) {
        for (int h = 0; h < n_heads; h++)
            fallback_rmsnorm(s->q + (size_t)h * head_size,
                             s->q + (size_t)h * head_size,
                             lw->attn.q_norm + (size_t)h * shape.qk_stride,
                             head_size, bn_transformer_gpu_norm_epsilon(c));
    }
    if (lw->attn.k_norm) {
        for (int h = 0; h < n_kv_heads; h++)
            fallback_rmsnorm(key_cache_row + (size_t)h * head_size,
                             key_cache_row + (size_t)h * head_size,
                             lw->attn.k_norm + (size_t)h * shape.qk_stride,
                             head_size, bn_transformer_gpu_norm_epsilon(c));
    }

    bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);
    bn_transformer_cpu_apply_rope_heads(key_cache_row, n_kv_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);

    BnGQACtx gctx = {
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len,
        bn_transformer_attention_scale(c, head_size),
        bn_transformer_kv_host_cache_uses_fp16_rows(c)
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);

    fallback_cpu_matvec(m, s->xb2, &lw->attn.wo, s->xb, s->x_q);
    bn_transformer_cpu_residual_add(s->x, s->xb2, dim);

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int max_i = 0;
    for (int i = 0; i < dim; i++) {
        float diff = fabsf(gpu_x[i] - s->x[i]);
        sum_abs += (double)diff;
        sum_sq += (double)diff * (double)diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
    }
    fprintf(stderr,
            "[bn:gpu:debug] attention_compare layer=%d pos=%d "
            "max_abs=%.9g max_i=%d cpu=%.9g gpu=%.9g "
            "mean_abs=%.9g rms=%.9g\n",
            layer, pos, max_abs, max_i, s->x[max_i], gpu_x[max_i],
            sum_abs / (double)dim, sqrt(sum_sq / (double)dim));

    free(cpu_in);
    free(gpu_x);
    return 0;
}

int bn_transformer_gpu_debug_compare_gqa(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int cache_pos,
    int rope_dims,
    const float *rope_cos,
    const float *rope_sin,
    int dim) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnLayerShapePlan shape;
    bn_transformer_plan_layer_shape(&shape, c, lw, layer,
                                    bn_model_tq_state(m) != NULL);
    if (!shape.is_attn || shape.q_gated || shape.q_wide ||
        !bn_transformer_kv_host_float_cache_rows_available(c) ||
        lw->attn.q_norm || lw->attn.k_norm || lw->norm.attn_sub_norm)
        return -1;

    float *cpu_in = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_xb = (float *)malloc((size_t)dim * sizeof(float));
    if (!cpu_in || !gpu_xb) {
        free(cpu_in);
        free(gpu_xb);
        return -1;
    }
    memcpy(cpu_in, s->x, (size_t)dim * sizeof(float));

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_xb(gpu, gpu_xb,
                                   (size_t)dim * sizeof(float)) != 0) {
        free(cpu_in);
        free(gpu_xb);
        return -1;
    }

    int head_size = shape.head_size;
    int n_heads = shape.n_heads;
    int kv_dim = shape.kv_dim;
    int n_kv_heads = shape.n_kv_heads;
    int kv_mul = shape.kv_mul;
    int layer_rope_dims = rope_dims > head_size ? head_size : rope_dims;
    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
    size_t loff = (size_t)shape.attn_idx * c->seq_len * c->kv_dim;
    size_t kv_bytes = (size_t)n_kv * c->kv_dim * sizeof(float);
    size_t kv_off = loff * sizeof(float);

    if (bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, s->key_cache + loff, kv_bytes,
            kv_off) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, s->value_cache + loff, kv_bytes,
            kv_off) != 0) {
        free(cpu_in);
        free(gpu_xb);
        return -1;
    }

    memcpy(s->x, cpu_in, (size_t)dim * sizeof(float));
    float *key_cache_row =
        s->key_cache + loff + (size_t)cache_pos * c->kv_dim;
    float *value_cache_row =
        s->value_cache + loff + (size_t)cache_pos * c->kv_dim;

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim,
                     bn_transformer_gpu_norm_epsilon(c));
    fallback_cpu_matvec(m, s->q, &lw->attn.wq, s->xb, s->x_q);
    fallback_cpu_matvec(m, key_cache_row, &lw->attn.wk, s->xb, s->x_q);
    fallback_cpu_matvec(m, value_cache_row, &lw->attn.wv, s->xb, s->x_q);

    if (lw->attn.q_bias) {
        for (int i = 0; i < dim; i++) s->q[i] += lw->attn.q_bias[i];
    }
    if (lw->attn.k_bias) {
        for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->attn.k_bias[i];
    }
    if (lw->attn.v_bias) {
        for (int i = 0; i < kv_dim; i++)
            value_cache_row[i] += lw->attn.v_bias[i];
    }

    bn_transformer_cpu_apply_rope_heads(s->q, n_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);
    bn_transformer_cpu_apply_rope_heads(key_cache_row, n_kv_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);

    BnGQACtx gctx = {
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len,
        bn_transformer_attention_scale(c, head_size),
        bn_transformer_kv_host_cache_uses_fp16_rows(c)
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);

    bn_transformer_gpu_debug_compare_vec(
        "gqa_compare", layer, pos, s->xb, gpu_xb, dim);

    free(cpu_in);
    free(gpu_xb);
    return 0;
}

int bn_transformer_gpu_debug_compare_qkv(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    uint32_t kv_cache_off,
    int dim,
    int q_dim,
    int kv_dim) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnLayerShapePlan shape;
    bn_transformer_plan_layer_shape(&shape, c, lw, layer,
                                    bn_model_tq_state(m) != NULL);
    if (!shape.is_attn || shape.head_size <= 0 ||
        q_dim % shape.head_size != 0)
        return -1;
    int head_size = shape.head_size;
    int n_kv_heads = shape.n_kv_heads;
    int qk_stride = shape.qk_stride;
    float *cpu_q = (float *)malloc((size_t)q_dim * sizeof(float));
    float *cpu_k = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *cpu_v = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *gpu_xb = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_q = (float *)malloc((size_t)q_dim * sizeof(float));
    float *gpu_k = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *gpu_v = (float *)malloc((size_t)kv_dim * sizeof(float));
    if (!cpu_q || !cpu_k || !cpu_v || !gpu_xb || !gpu_q || !gpu_k || !gpu_v) {
        free(cpu_q); free(cpu_k); free(cpu_v);
        free(gpu_xb); free(gpu_q); free(gpu_k); free(gpu_v);
        return -1;
    }

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_xb(gpu, gpu_xb,
                                   (size_t)dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf(gpu, BN_GPU_VALUE_Q, gpu_q,
                                               (size_t)q_dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, gpu_k,
            (size_t)kv_dim * sizeof(float),
            (size_t)kv_cache_off * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, gpu_v,
            (size_t)kv_dim * sizeof(float),
            (size_t)kv_cache_off * sizeof(float)) != 0) {
        free(cpu_q); free(cpu_k); free(cpu_v);
        free(gpu_xb); free(gpu_q); free(gpu_k); free(gpu_v);
        return -1;
    }

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim,
                     bn_transformer_gpu_norm_epsilon(&m->config));
    bn_transformer_gpu_debug_compare_vec(
        "attn_norm_compare", layer, pos, s->xb, gpu_xb, dim);
    debug_compare_native_quant_activation(gpu, layer, pos, s->xb, dim);
    debug_quant_matvec_prepared(m, cpu_q, &lw->attn.wq, s->xb, s->x_q);
    debug_quant_matvec_prepared(m, cpu_k, &lw->attn.wk, s->xb, s->x_q);
    debug_quant_matvec_prepared(m, cpu_v, &lw->attn.wv, s->xb, s->x_q);
    if (lw->attn.q_bias) {
        for (int i = 0; i < q_dim; i++) cpu_q[i] += lw->attn.q_bias[i];
    }
    if (lw->attn.k_bias) {
        for (int i = 0; i < kv_dim; i++) cpu_k[i] += lw->attn.k_bias[i];
    }
    if (lw->attn.v_bias) {
        for (int i = 0; i < kv_dim; i++) cpu_v[i] += lw->attn.v_bias[i];
    }
    if (lw->attn.q_norm) {
        int n_heads = q_dim / head_size;
        for (int h = 0; h < n_heads; h++)
            fallback_rmsnorm(cpu_q + (size_t)h * head_size,
                             cpu_q + (size_t)h * head_size,
                             lw->attn.q_norm + (size_t)h * qk_stride,
                             head_size,
                             bn_transformer_gpu_norm_epsilon(c));
    }
    if (lw->attn.k_norm) {
        for (int h = 0; h < n_kv_heads; h++)
            fallback_rmsnorm(cpu_k + (size_t)h * head_size,
                             cpu_k + (size_t)h * head_size,
                             lw->attn.k_norm + (size_t)h * qk_stride,
                             head_size,
                             bn_transformer_gpu_norm_epsilon(c));
    }
    if (lw->attn.k_bias) {
        int rope_dims = bn_transformer_rope_dims_for_head(c, head_size);
        int half = rope_dims / 2;
        for (int h = 0; h < n_kv_heads; h++) {
            float *kh = cpu_k + (size_t)h * head_size;
            for (int i = 0; i < half; i++) {
                float angle = (float)pos * s->rope_freq[i];
                float cosv = cosf(angle);
                float sinv = sinf(angle);
                float x0 = kh[i];
                float x1 = kh[i + half];
                kh[i] = x0 * cosv - x1 * sinv;
                kh[i + half] = x0 * sinv + x1 * cosv;
            }
        }
    }

    bn_transformer_gpu_debug_compare_vec(
        "qkv_q_compare", layer, pos, cpu_q, gpu_q, q_dim);
    bn_transformer_gpu_debug_compare_vec(
        "qkv_k_compare", layer, pos, cpu_k, gpu_k, kv_dim);
    bn_transformer_gpu_debug_compare_vec(
        "qkv_v_compare", layer, pos, cpu_v, gpu_v, kv_dim);

    free(cpu_q); free(cpu_k); free(cpu_v);
    free(gpu_xb); free(gpu_q); free(gpu_k); free(gpu_v);
    return 0;
}

int bn_transformer_gpu_fallback_logits(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    const BnTransformerGPULogitResources *logits,
    int dim) {
    BnRunState *s = &sess->state;
    double t0 = bn_platform_time_ms();
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0) {
        bn_transformer_gpu_report_fallback("gpu logits cpu fallback flush failed");
        return -1;
    }
    double t_flush = bn_platform_time_ms();
    if (bn_transformer_gpu_read_xb(gpu, s->xb,
                                   (size_t)dim * sizeof(float)) != 0) {
        bn_transformer_gpu_report_fallback("gpu logits cpu fallback read_xb failed");
        return -1;
    }
    double t_read = bn_platform_time_ms();
    fallback_cpu_matvec(m, s->logits, logits->cpu_weight, s->xb, s->x_q);
    double t_logits = bn_platform_time_ms();
    if (bn_transformer_gpu_profile_level() >= 3) {
        fprintf(stderr,
                "[gpu:fallback:logits] flush=%.3fms read=%.3fms cpu=%.3fms total=%.3fms\n",
                t_flush - t0, t_read - t_flush, t_logits - t_read,
                t_logits - t0);
    }
    return 0;
}
