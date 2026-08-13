#include "gpu_internal.h"
#include "../gpu_shader.h"
#include "transformer_cpu_backend_internal.h"
#include "transformer_cpu_internal.h"
#include "transformer_gqa_internal.h"
#include "transformer_kv_internal.h"
#include "transformer_plan_internal.h"
#include "transformer_rmsnorm_internal.h"
#include "transformer_ssm_internal.h"
#include "model_internal.h"
#include "../moe_internal.h"
#include "moe.h"
#include "platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BN_GPU_LOGITS_REFINE_MAX_SCALE_BLOCKS 8192

static const BnCPURuntimePolicy *fallback_cpu_runtime(const BnModel *m) {
    return bn_tp_cpu_policy(bn_model_pool(m));
}

int bn_transformer_gpu_debug_dump_layer_input(
    const BnCPURuntimePolicy *runtime,
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    int layer,
    int pos,
    int dim) {
    const char *path = bn_transformer_cpu_debug_binary_path(runtime);
    if (!path || !bn_transformer_cpu_debug_dump_pos_selected(runtime, pos) ||
        !bn_transformer_cpu_debug_binary_selected(runtime, "gpu_inp", layer))
        return 0;
    if (!emit || !gpu || dim <= 0)
        return -1;
    size_t bytes = (size_t)dim * sizeof(float);
    float *state = (float *)malloc(bytes);
    if (!state)
        return -1;
    int rc = bn_transformer_gpu_emit_context_flush(emit, gpu);
    if (rc == 0)
        rc = bn_transformer_gpu_read_x(gpu, state, bytes);
    if (rc == 0) {
        FILE *binary = fopen(path, "wb");
        if (!binary || fwrite(state, sizeof(*state), (size_t)dim, binary) !=
                           (size_t)dim)
            rc = -1;
        if (binary)
            fclose(binary);
    }
    free(state);
    return rc;
}

static float fallback_reference_gelu(float x) {
    if (x <= -10.0f)
        return 0.0f;
    if (x >= 10.0f)
        return x;
    float rounded_x = bn_fp16_to_fp32(bn_fp32_to_fp16(x));
    float inner = 0.7978845608028654f * rounded_x *
                  (1.0f + 0.044715f * rounded_x * rounded_x);
    float gelu = 0.5f * rounded_x * (1.0f + tanhf(inner));
    return bn_fp16_to_fp32(bn_fp32_to_fp16(gelu));
}

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
                fallback_cpu_runtime(m), &m->config,
                bn_model_backend(m), tasks[i].W);
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
                 normalize_topk, expert_weights_scale,
                 bn_model_moe_policy_uses_reference_router_accumulation(
                     &model->config),
                 bn_model_pool(model));
}

static const float *gpu_moe_cpu_route_input(
    const BnGPUBackend *gpu,
    BnSession *session,
    const BnLayerWeights *layer,
    const BnMoEExecutionPolicy *execution,
    int dim) {
    BnRunState *state = &session->state;
    if (!execution->uses_scaled_router_input)
        return bn_transformer_gpu_read_xb(
                   gpu, state->xb, (size_t)dim * sizeof(float)) == 0
            ? state->xb : NULL;

    if (bn_transformer_gpu_read_x(
            gpu, state->x, (size_t)dim * sizeof(float)) != 0)
        return NULL;
    float ss = 0.0f;
    for (int d = 0; d < dim; d++)
        ss += state->x[d] * state->x[d];
    float scale = (1.0f / sqrtf(ss / (float)dim + execution->norm_eps)) /
                  sqrtf((float)dim);
    for (int d = 0; d < dim; d++)
        state->xb2[d] = state->x[d] * scale *
            (layer->moe.router_scale ? layer->moe.router_scale[d] : 1.0f);
    return state->xb2;
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
    BnMoEExecutionPolicy execution =
        bn_moe_execution_policy(&model->config);
    double t0 = profile_enabled ? bn_platform_time_ms() : 0.0;
    int used_gpu_topk = 0;
    if (route->gpu_route_topk && !execution.uses_scaled_router_input) {
        if (bn_transformer_gpu_emit_context_moe_route_topk(
                emit, route->router, route->expert_down_scale,
                BN_GPU_VALUE_XB,
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
    const float *cpu_route_input = NULL;
    if (!used_gpu_topk)
        cpu_route_input = gpu_moe_cpu_route_input(
            gpu, session, layer, &execution, dim);
    if (!used_gpu_topk && !cpu_route_input) {
        if (reason) *reason = "gpu moe route input readback failed";
        return -1;
    }
    double t2 = profile_enabled ? bn_platform_time_ms() : 0.0;
    if (!used_gpu_topk)
        bn_transformer_gpu_route_model_moe(
            model, session->moe_state, cpu_route_input, layer,
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
        BnMoEExecutionPolicy execution =
            bn_moe_execution_policy(&model->config);
        if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0) {
            if (reason) *reason = "gpu moe cpu route input readback failed";
            return -1;
        }
        const float *route_input = gpu_moe_cpu_route_input(
            gpu, session, layer, &execution, dim);
        if (!route_input) {
            if (reason) *reason = "gpu moe cpu route input readback failed";
            return -1;
        }
        bn_transformer_gpu_route_model_moe(
            model, session->moe_state, route_input, layer,
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
            int expert = session->moe_state->expert_indices[k];
            route_values[k] = session->moe_state->expert_weights[k] *
                bn_moe_expert_weight_scale(layer, expert);
            route_values[active_experts + k] =
                (float)expert;
        }
        if (bn_transformer_gpu_write_activation_buf(
                gpu, BN_GPU_VALUE_MOE_HB2, route_values,
                (size_t)(2 * active_experts) * sizeof(float)) != 0) {
            if (reason) *reason = "gpu moe cpu route upload failed";
            return -1;
        }
    } else {
        int route_input_buf = BN_GPU_VALUE_XB;
        if (route->uses_scaled_router_input) {
            uint32_t eps_bits;
            float eps = bn_moe_execution_policy(&model->config).norm_eps;
            memcpy(&eps_bits, &eps, sizeof(eps_bits));
            if (!route->router_scale ||
                bn_transformer_gpu_emit_context_rmsnorm(
                    emit, route->router_scale, BN_GPU_VALUE_X,
                    BN_GPU_VALUE_MOE_OUT, dim, eps_bits) != 0) {
                if (reason) *reason =
                    "gpu scaled moe route input emit failed";
                return -1;
            }
            route_input_buf = BN_GPU_VALUE_MOE_OUT;
        }
        if (bn_transformer_gpu_emit_context_moe_route_topk(
                   emit, route->router, route->expert_down_scale,
                   route_input_buf,
                   BN_GPU_VALUE_MOE_HB, BN_GPU_VALUE_MOE_HB2,
                   dim, route_policy->total_experts,
                   route_policy->active_experts,
                   route_policy->expert_weights_scale,
                   route->route_flags) != 0) {
            if (reason) *reason = "gpu moe route emit failed";
            return -1;
        }
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
        BnMoEExecutionPolicy execution =
            bn_moe_execution_policy(&model->config);
        const float *cpu_route_input = gpu_moe_cpu_route_input(
            gpu, session, layer, &execution, dim);
        if (!cpu_route_input) {
            if (reason) *reason = "gpu moe route compare input failed";
            return -1;
        }
        if (route->uses_scaled_router_input) {
            float *gpu_route_input =
                (float *)malloc((size_t)dim * sizeof(float));
            if (!gpu_route_input ||
                bn_transformer_gpu_read_activation_buf(
                    gpu, BN_GPU_VALUE_MOE_OUT, gpu_route_input,
                    (size_t)dim * sizeof(float)) != 0) {
                free(gpu_route_input);
                if (reason) *reason = "gpu moe scaled route compare failed";
                return -1;
            }
            float max_abs = 0.0f;
            double sum_abs = 0.0;
            for (int d = 0; d < dim; d++) {
                float diff = fabsf(cpu_route_input[d] - gpu_route_input[d]);
                if (diff > max_abs)
                    max_abs = diff;
                sum_abs += diff;
            }
            fprintf(stderr,
                    "[bn:gpu:debug] moe_route_input_compare layer=%d pos=%d "
                    "max_abs=%.9g mean_abs=%.9g\n",
                    layer_index, pos, max_abs, sum_abs / (double)dim);
            free(gpu_route_input);
        }
        bn_transformer_gpu_route_model_moe(
            model, session->moe_state, cpu_route_input, layer,
            route_policy->total_experts, route_policy->active_experts,
            route_policy->normalize_topk,
            route_policy->expert_weights_scale);
        for (int k = 0; k < active_experts; k++) {
            fprintf(stderr,
                    "[bn:gpu:debug] moe_route_compare layer=%d pos=%d "
                    "slot=%d cpu_w=%.9g gpu_w=%.9g cpu_e=%d gpu_e=%d\n",
                    layer_index, pos, k,
                    session->moe_state->expert_weights[k] *
                        bn_moe_expert_weight_scale(
                            layer, session->moe_state->expert_indices[k]),
                    route_values[k],
                    session->moe_state->expert_indices[k],
                    (int)(route_values[active_experts + k] + 0.5f));
        }
    }
    return 0;
}

int bn_transformer_gpu_debug_compare_routed_moe_raw(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnTransformerGPUMoEDecodeResources *resources,
    const BnTransformerGPUMoEExecutionPolicy *route_policy,
    const BnTransformerGPUMoEDecodeRoutePolicy *route,
    const BnTransformerGPUMoEProjectionPolicy *projection,
    const BnTransformerGPUMoEDebugPolicy *debug,
    int layer_index,
    int pos,
    int dim) {
    if (!debug || !debug->compare_raw ||
        !resources || !resources->gate_all || !resources->up_all ||
        !route || !route->all_active_two_kquant_moe)
        return 0;
    if (!emit || !gpu || !model || !session || !layer || !route_policy ||
        !projection || !projection->valid || dim <= 0)
        return -1;
    int active_experts = route_policy->active_experts;
    int total_experts = route_policy->total_experts;
    int hidden_dim = route_policy->expert_hidden_dim;
    if (active_experts < 0 || active_experts > BN_MAX_MOE_K ||
        total_experts <= 0 || hidden_dim <= 0)
        return -1;
    size_t raw_bytes =
        (size_t)total_experts * (size_t)hidden_dim * sizeof(float);
    float *cpu_gate = (float *)malloc(raw_bytes);
    float *cpu_up = (float *)malloc(raw_bytes);
    float *gpu_gate = (float *)malloc(raw_bytes);
    float *gpu_up = (float *)malloc(raw_bytes);
    float route_save[BN_MAX_MOE_K * 2];
    int route_saved = 0;
    int rc = -1;
    uint32_t gate_flags =
        bn_transformer_gpu_moe_expert_projection_matvec_flags(
            &layer->moe.expert_map, 0, 1);
    uint32_t up_flags =
        bn_transformer_gpu_moe_expert_projection_matvec_flags(
            &layer->moe.expert_map, 1, 1);
    if (!cpu_gate || !cpu_up || !gpu_gate || !gpu_up ||
        bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_MOE_HB2, route_save,
            (size_t)(2 * active_experts) * sizeof(float)) != 0)
        goto cleanup;
    route_saved = 1;
    if (bn_transformer_gpu_fallback_moe_raw_gate_up(
            model, session, layer, session->state.xb,
            cpu_gate, cpu_up) != 0 ||
        bn_transformer_gpu_emit_context_matvec_flags(
            emit, projection->gate_type, resources->gate_all,
            BN_GPU_VALUE_XB, BN_GPU_VALUE_MOE_HB,
            total_experts * hidden_dim, dim, 0, gate_flags) != 0 ||
        bn_transformer_gpu_emit_context_matvec_flags(
            emit, projection->up_type, resources->up_all,
            BN_GPU_VALUE_XB, BN_GPU_VALUE_MOE_HB2,
            total_experts * hidden_dim, dim, 0, up_flags) != 0 ||
        bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_MOE_HB, gpu_gate, raw_bytes) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_MOE_HB2, gpu_up, raw_bytes) != 0 ||
        bn_transformer_gpu_write_activation_buf(
            gpu, BN_GPU_VALUE_MOE_HB2, route_save,
            (size_t)(2 * active_experts) * sizeof(float)) != 0)
        goto cleanup;
    route_saved = 0;
    for (int expert = 0; expert < total_experts; expert++) {
        char label[64];
        snprintf(label, sizeof(label),
                 "moe_raw_gate_compare[%d]", expert);
        bn_transformer_gpu_debug_compare_vec(
            label, layer_index, pos,
            cpu_gate + (size_t)expert * (size_t)hidden_dim,
            gpu_gate + (size_t)expert * (size_t)hidden_dim,
            hidden_dim);
        snprintf(label, sizeof(label),
                 "moe_raw_up_compare[%d]", expert);
        bn_transformer_gpu_debug_compare_vec(
            label, layer_index, pos,
            cpu_up + (size_t)expert * (size_t)hidden_dim,
            gpu_up + (size_t)expert * (size_t)hidden_dim,
            hidden_dim);
    }
    rc = 0;

cleanup:
    if (route_saved &&
        bn_transformer_gpu_write_activation_buf(
            gpu, BN_GPU_VALUE_MOE_HB2, route_save,
            (size_t)(2 * active_experts) * sizeof(float)) != 0)
        rc = -1;
    free(cpu_gate);
    free(cpu_up);
    free(gpu_gate);
    free(gpu_up);
    return rc;
}

int bn_transformer_gpu_debug_compare_routed_moe_mid(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnTransformerGPUMoEExecutionPolicy *route_policy,
    const BnTransformerGPUMoEDebugPolicy *debug,
    int layer_index,
    int pos) {
    if (!debug || !debug->compare_mid)
        return 0;
    if (!emit || !gpu || !model || !session || !layer || !route_policy)
        return -1;
    int active_experts = route_policy->active_experts;
    int hidden_dim = route_policy->expert_hidden_dim;
    if (active_experts < 0 || active_experts > BN_MAX_MOE_K ||
        hidden_dim <= 0)
        return -1;
    size_t mid_bytes =
        (size_t)active_experts * (size_t)hidden_dim * sizeof(float);
    float *cpu_mid = (float *)malloc(mid_bytes);
    float *gpu_mid = (float *)malloc(mid_bytes);
    int rc = -1;
    if (!cpu_mid || !gpu_mid ||
        bn_transformer_gpu_fallback_moe_mid(
            model, session, layer, session->state.xb, cpu_mid) != 0 ||
        bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_MOE_HB, gpu_mid, mid_bytes) != 0)
        goto cleanup;
    for (int k = 0; k < active_experts; k++) {
        char label[64];
        snprintf(label, sizeof(label), "moe_mid_compare[%d]", k);
        bn_transformer_gpu_debug_compare_vec(
            label, layer_index, pos,
            cpu_mid + (size_t)k * (size_t)hidden_dim,
            gpu_mid + (size_t)k * (size_t)hidden_dim,
            hidden_dim);
    }
    rc = 0;

cleanup:
    free(cpu_mid);
    free(gpu_mid);
    return rc;
}

void bn_transformer_gpu_discard_routed_moe_debug_state(
    BnTransformerGPURoutedMoEDebugState *state) {
    if (!state)
        return;
    free(state->cpu_state);
    free(state->gpu_state);
    free(state->override_state);
    free(state->input_state);
    memset(state, 0, sizeof(*state));
}

int bn_transformer_gpu_prepare_routed_moe_debug_state(
    BnTransformerGPURoutedMoEDebugState *debug_state,
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnTransformerGPUMoEExecutionPolicy *route_policy,
    const BnTransformerGPUMoEDebugPolicy *debug,
    int layer_index,
    int pos,
    int dim,
    float norm_eps) {
    if (!debug_state)
        return -1;
    memset(debug_state, 0, sizeof(*debug_state));
    if (!debug || (!debug->override_cpu_actual && !debug->compare_layer))
        return 0;
    if (!emit || !gpu || !model || !session || !layer || !route_policy ||
        dim <= 0)
        return -1;
    size_t bytes = (size_t)dim * sizeof(float);
    BnRunState *state = &session->state;
    if (debug->override_cpu_actual) {
        debug_state->override_enabled = 1;
        debug_state->override_state = (float *)malloc(bytes);
        if (!debug_state->override_state ||
            bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
            bn_transformer_gpu_read_x(gpu, state->x, bytes) != 0 ||
            bn_transformer_gpu_read_xb(gpu, state->xb, bytes) != 0 ||
            bn_transformer_gpu_fallback_moe_output_from_state(
                model, session, layer, layer_index, dim,
                debug_state->override_state) != 0) {
            bn_transformer_gpu_discard_routed_moe_debug_state(debug_state);
            return -1;
        }
    }
    if (!debug->compare_layer)
        return 0;
    debug_state->compare_enabled = 1;
    debug_state->cpu_state = (float *)malloc(bytes);
    debug_state->gpu_state = (float *)malloc(bytes);
    debug_state->input_state = (float *)malloc(bytes);
    if (!debug_state->cpu_state || !debug_state->gpu_state ||
        !debug_state->input_state ||
        bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_x(gpu, state->x, bytes) != 0 ||
        bn_transformer_gpu_read_xb(gpu, state->xb, bytes) != 0)
        goto compare_error;
    memcpy(debug_state->input_state, state->x, bytes);
    if (debug->compare_input_norm && layer->norm.ffn_norm) {
        float *cpu_norm = (float *)malloc(bytes);
        if (!cpu_norm)
            goto compare_error;
        bn_transformer_gpu_debug_rmsnorm(
            cpu_norm, state->x, layer->norm.ffn_norm, dim, norm_eps);
        bn_transformer_gpu_debug_compare_vec(
            "moe_input_norm_compare", layer_index, pos,
            cpu_norm, state->xb, dim);
        free(cpu_norm);
    }
    if (debug->compare_actual) {
        if (bn_transformer_gpu_fallback_moe_output_from_state(
                model, session, layer, layer_index, dim,
                debug_state->cpu_state) != 0)
            goto compare_error;
    } else {
        bn_transformer_gpu_route_model_moe(
            model, session->moe_state, state->xb, layer,
            route_policy->total_experts, route_policy->active_experts,
            route_policy->normalize_topk,
            route_policy->expert_weights_scale);
        if (bn_transformer_gpu_fallback_moe_output(
                model, session, layer, dim, state->x, state->xb,
                debug_state->cpu_state) != 0)
            goto compare_error;
    }
    return 0;

compare_error:
    bn_transformer_gpu_discard_routed_moe_debug_state(debug_state);
    return -2;
}

void bn_transformer_gpu_discard_routed_moe_parts_comparison(
    BnTransformerGPUMoEPartsComparison *comparison) {
    if (!comparison)
        return;
    free(comparison->cpu_routed);
    free(comparison->cpu_shared);
    free(comparison->gpu_routed);
    memset(comparison, 0, sizeof(*comparison));
}

int bn_transformer_gpu_debug_compare_cached_moe_expert(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const float *input_state,
    int route_index,
    int layer_index,
    int pos,
    int dim) {
    if (emit && bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&model->config);
    const BnMoEExpertMap *map = &layer->moe.expert_map;
    int k = route_index;
    int hidden = route_policy.expert_hidden_dim;
    if (!session->moe_state || !input_state || k < 0 ||
        k >= route_policy.active_experts || hidden <= 0 ||
        map->gate_rows != hidden || map->up_rows != hidden ||
        map->down_rows != dim || map->down_cols != hidden)
        return -1;

    int expert = session->moe_state->expert_indices[k];
    const void *gate_data = bn_moe_get_expert_proj(
        bn_model_moe_io(model), session->moe_state, map, expert, 0);
    const void *up_data = bn_moe_get_expert_proj(
        bn_model_moe_io(model), session->moe_state, map, expert, 1);
    const void *down_data = bn_moe_get_expert_proj(
        bn_model_moe_io(model), session->moe_state, map, expert, 2);
    BnQWeight gate_weight;
    BnQWeight up_weight;
    BnQWeight down_weight;
    if (!gate_data || !up_data || !down_data ||
        !bn_moe_expert_projection_weight(&gate_weight, gate_data, map, 0) ||
        !bn_moe_expert_projection_weight(&up_weight, up_data, map, 1) ||
        !bn_moe_expert_projection_weight(&down_weight, down_data, map, 2))
        return -1;

    float *cpu_gate = (float *)malloc((size_t)hidden * sizeof(float));
    float *cpu_up = (float *)malloc((size_t)hidden * sizeof(float));
    float *gpu_mid = (float *)malloc((size_t)hidden * sizeof(float));
    float *cpu_down = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_down = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_input = (float *)malloc((size_t)dim * sizeof(float));
    if (!cpu_gate || !cpu_up || !gpu_mid || !cpu_down || !gpu_down ||
        !gpu_input) {
        free(cpu_gate);
        free(cpu_up);
        free(gpu_mid);
        free(cpu_down);
        free(gpu_down);
        free(gpu_input);
        return -1;
    }

    bn_transformer_gpu_cpu_quant_matvec_model(
        model, cpu_gate, &gate_weight, input_state,
        session->state.x_q);
    bn_transformer_gpu_cpu_quant_matvec_model(
        model, cpu_up, &up_weight, input_state,
        session->state.x_q);
    BnTransformerGPUMoEActivationPolicy activation_policy =
        bn_transformer_gpu_moe_activation_policy(&model->config);
    bn_moe_swiglu(cpu_gate, cpu_gate, cpu_up, hidden,
                  activation_policy.uses_reference_silu,
                  activation_policy.uses_reference_ffn_activation);
    bn_transformer_gpu_cpu_quant_matvec_model(
        model, cpu_down, &down_weight, cpu_gate, session->state.x_q);

    int rc = 0;
    if (bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_MOE_HB, gpu_mid,
            (size_t)hidden * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_XB2, gpu_down,
            (size_t)dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_XB, gpu_input,
            (size_t)dim * sizeof(float)) != 0) {
        rc = -1;
    } else {
        char input_label[80];
        char mid_label[80];
        char down_label[80];
        snprintf(input_label, sizeof(input_label),
                 "moe_cached_actual_input_compare[%d:%d]", k, expert);
        snprintf(mid_label, sizeof(mid_label),
                 "moe_cached_actual_mid_compare[%d:%d]", k, expert);
        snprintf(down_label, sizeof(down_label),
                 "moe_cached_actual_down_compare[%d:%d]", k, expert);
        bn_transformer_gpu_debug_compare_vec(
            input_label, layer_index, pos,
            input_state, gpu_input, dim);
        bn_transformer_gpu_cpu_quant_matvec_model(
            model, cpu_gate, &gate_weight, input_state,
            session->state.x_q);
        bn_transformer_gpu_cpu_quant_matvec_model(
            model, cpu_up, &up_weight, input_state,
            session->state.x_q);
        for (int i = 0; i < hidden; i++) {
            if (!isfinite(gpu_mid[i])) {
                fprintf(stderr,
                        "[bn:gpu:debug] moe_cached_nonfinite_mid "
                        "layer=%d pos=%d slot=%d expert=%d index=%d "
                        "gate=%.9g up=%.9g gpu=%.9g\n",
                        layer_index, pos, k, expert, i,
                        cpu_gate[i], cpu_up[i], gpu_mid[i]);
            }
        }
        bn_moe_swiglu(cpu_gate, cpu_gate, cpu_up, hidden,
                      activation_policy.uses_reference_silu,
                      activation_policy.uses_reference_ffn_activation);
        bn_transformer_gpu_cpu_quant_matvec_model(
            model, cpu_down, &down_weight, cpu_gate,
            session->state.x_q);
        bn_transformer_gpu_debug_compare_vec(
            mid_label, layer_index, pos,
            cpu_gate, gpu_mid, hidden);
        bn_transformer_gpu_debug_compare_vec(
            down_label, layer_index, pos,
            cpu_down, gpu_down, dim);
    }

    free(cpu_gate);
    free(cpu_up);
    free(gpu_mid);
    free(cpu_down);
    free(gpu_down);
    free(gpu_input);
    return rc;
}

int bn_transformer_gpu_prepare_routed_moe_parts_comparison(
    BnTransformerGPUMoEPartsComparison *comparison,
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnTransformerGPUMoEDebugPolicy *debug,
    const float *input_state,
    int layer_index,
    int pos,
    int dim) {
    if (!comparison)
        return -1;
    memset(comparison, 0, sizeof(*comparison));
    if (!debug || !debug->compare_parts)
        return 0;
    if (!emit || !gpu || !model || !session || !layer || !input_state ||
        dim <= 0)
        return -1;
    size_t dim_bytes = (size_t)dim * sizeof(float);
    comparison->cpu_routed = (float *)malloc(dim_bytes);
    comparison->cpu_shared = (float *)malloc(dim_bytes);
    comparison->gpu_routed = (float *)malloc(dim_bytes);
    float *gpu_input = (float *)malloc(dim_bytes);
    comparison->enabled = 1;
    if (!comparison->cpu_routed || !comparison->cpu_shared ||
        !comparison->gpu_routed || !gpu_input ||
        bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_XB, gpu_input, dim_bytes) != 0 ||
        bn_transformer_gpu_fallback_moe_parts(
            model, session, layer, dim, input_state,
            comparison->cpu_routed, comparison->cpu_shared) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_MOE_OUT,
            comparison->gpu_routed, dim_bytes) != 0) {
        free(gpu_input);
        bn_transformer_gpu_discard_routed_moe_parts_comparison(comparison);
        return -1;
    }
    free(gpu_input);
    if (bn_transformer_gpu_debug_compare_cached_moe_expert(
            NULL, gpu, model, session, layer, input_state,
            bn_moe_route_policy(&model->config).active_experts - 1,
            layer_index, pos, dim) != 0) {
        bn_transformer_gpu_discard_routed_moe_parts_comparison(comparison);
        return -1;
    }
    bn_transformer_gpu_debug_compare_vec(
        "moe_routed_part_compare", layer_index, pos,
        comparison->cpu_routed, comparison->gpu_routed, dim);
    return 0;
}

int bn_transformer_gpu_debug_compare_cached_moe_gate_up(
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnGPUMoEResources *resources,
    const BnTransformerGPUMoEDebugPolicy *debug,
    int layer_index,
    int pos,
    int dim) {
    if (!debug || !debug->compare_raw)
        return 0;
    if (!gpu || !model || !session || !session->moe_state || !layer ||
        !resources || !resources->expert_map || dim <= 0)
        return -1;
    const BnMoEExpertMap *map = resources->expert_map;
    int gate_rows = map->gate_rows;
    int up_rows = map->up_rows;
    int max_rows = gate_rows > up_rows ? gate_rows : up_rows;
    int stacked_rows = gate_rows + up_rows;
    float *cpu_gate = (float *)malloc((size_t)gate_rows * sizeof(float));
    float *cpu_up = (float *)malloc((size_t)up_rows * sizeof(float));
    float *gpu_out = (float *)malloc(
        (size_t)(stacked_rows > max_rows ? stacked_rows : max_rows) *
        sizeof(float));
    float *cpu_down = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_down = (float *)malloc((size_t)dim * sizeof(float));
    if (!cpu_gate || !cpu_up || !gpu_out || !cpu_down || !gpu_down) {
        free(cpu_gate);
        free(cpu_up);
        free(gpu_out);
        free(cpu_down);
        free(gpu_down);
        return -1;
    }
    BnTransformerGPUMoEActivationPolicy activation_policy =
        bn_transformer_gpu_moe_activation_policy(&model->config);
    int rc = 0;
    for (int k = 0; k < resources->n_experts; k++) {
        int expert_idx = session->moe_state->expert_indices[k];
        const void *gate_data = bn_moe_get_expert_proj(
            bn_model_moe_io(model), session->moe_state, map, expert_idx, 0);
        const void *up_data = bn_moe_get_expert_proj(
            bn_model_moe_io(model), session->moe_state, map, expert_idx, 1);
        const void *down_data = bn_moe_get_expert_proj(
            bn_model_moe_io(model), session->moe_state, map, expert_idx, 2);
        BnQWeight gate_weight;
        BnQWeight up_weight;
        BnQWeight down_weight;
        if (!gate_data || !up_data || !down_data ||
            !bn_moe_expert_projection_weight(
                &gate_weight, gate_data, map, 0) ||
            !bn_moe_expert_projection_weight(
                &up_weight, up_data, map, 1) ||
            !bn_moe_expert_projection_weight(
                &down_weight, down_data, map, 2)) {
            rc = -1;
            break;
        }
        bn_transformer_gpu_cpu_quant_matvec_model(
            model, cpu_gate, &gate_weight, session->state.xb,
            session->state.x_q);
        bn_transformer_gpu_cpu_quant_matvec_model(
            model, cpu_up, &up_weight, session->state.xb,
            session->state.x_q);
        const BnGPUMoEExpertBuffers *buffers =
            &resources->experts[k].buffers;
        if (buffers->use_gateup_split) {
            if (bn_gpu_backend_matvec(
                    gpu, gpu_out, buffers->gate, session->state.xb,
                    stacked_rows, dim, map->gate_type) != 0) {
                rc = -1;
                break;
            }
            char gate_label[80];
            char up_label[80];
            snprintf(gate_label, sizeof(gate_label),
                     "moe_cached_gate_compare[%d:%d]", k, expert_idx);
            snprintf(up_label, sizeof(up_label),
                     "moe_cached_up_compare[%d:%d]", k, expert_idx);
            bn_transformer_gpu_debug_compare_vec(
                gate_label, layer_index, pos, cpu_gate, gpu_out, gate_rows);
            bn_transformer_gpu_debug_compare_vec(
                up_label, layer_index, pos, cpu_up,
                gpu_out + gate_rows, up_rows);
        } else {
            char label[80];
            if (bn_gpu_backend_matvec(
                    gpu, gpu_out, buffers->gate, session->state.xb,
                    gate_rows, dim, map->gate_type) != 0) {
                rc = -1;
                break;
            }
            snprintf(label, sizeof(label),
                     "moe_cached_gate_compare[%d:%d]", k, expert_idx);
            bn_transformer_gpu_debug_compare_vec(
                label, layer_index, pos, cpu_gate, gpu_out, gate_rows);
            if (bn_gpu_backend_matvec(
                    gpu, gpu_out, buffers->up, session->state.xb,
                    up_rows, dim, map->up_type) != 0) {
                rc = -1;
                break;
            }
            snprintf(label, sizeof(label),
                     "moe_cached_up_compare[%d:%d]", k, expert_idx);
            bn_transformer_gpu_debug_compare_vec(
                label, layer_index, pos, cpu_up, gpu_out, up_rows);
        }
        bn_moe_swiglu(cpu_gate, cpu_gate, cpu_up, gate_rows,
                      activation_policy.uses_reference_silu,
                      activation_policy.uses_reference_ffn_activation);
        bn_transformer_gpu_cpu_quant_matvec_model(
            model, cpu_down, &down_weight, cpu_gate, session->state.x_q);
        if (bn_gpu_backend_matvec(
                gpu, gpu_down, buffers->down, cpu_gate,
                map->down_rows, map->down_cols, map->down_type) != 0) {
            rc = -1;
            break;
        }
        char down_label[80];
        snprintf(down_label, sizeof(down_label),
                 "moe_cached_down_compare[%d:%d]", k, expert_idx);
        bn_transformer_gpu_debug_compare_vec(
            down_label, layer_index, pos, cpu_down, gpu_down,
            map->down_rows);
    }
    free(cpu_gate);
    free(cpu_up);
    free(gpu_out);
    free(cpu_down);
    free(gpu_down);
    return rc;
}

void bn_transformer_gpu_compare_routed_moe_shared_part(
    const BnTransformerGPUMoEPartsComparison *comparison,
    const float *gpu_state,
    const float *input_state,
    int layer_index,
    int pos,
    int dim) {
    if (!comparison || !comparison->enabled ||
        !gpu_state || !input_state || dim <= 0)
        return;
    float *gpu_shared =
        (float *)malloc((size_t)dim * sizeof(float));
    if (!gpu_shared)
        return;
    for (int i = 0; i < dim; i++)
        gpu_shared[i] =
            gpu_state[i] - input_state[i] - comparison->gpu_routed[i];
    bn_transformer_gpu_debug_compare_vec(
        "moe_shared_part_compare", layer_index, pos,
        comparison->cpu_shared, gpu_shared, dim);
    free(gpu_shared);
}

void bn_transformer_gpu_debug_compare_routed_moe_post_layer(
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnTransformerGPUMoEDebugPolicy *debug,
    const float *cpu_state,
    int layer_index,
    int pos,
    int dim,
    float norm_eps) {
    if (!gpu || !model || !session || !layer || !debug || !cpu_state ||
        dim <= 0)
        return;
    BnRunState *state = &session->state;
    int has_shared = bn_transformer_gpu_moe_has_loaded_shared_expert(
        &model->config, layer);
    if (debug->compare_shared_mid && has_shared) {
        BnTransformerGPUMoESharedExpertShapePolicy shared_shape =
            bn_transformer_gpu_moe_shared_expert_shape_policy(
                &model->config);
        int hidden = shared_shape.hidden_dim;
        size_t bytes = (size_t)hidden * sizeof(float);
        float *cpu_mid = hidden > 0 ? (float *)malloc(bytes) : NULL;
        float *gpu_mid = hidden > 0 ? (float *)malloc(bytes) : NULL;
        if (cpu_mid && gpu_mid &&
            bn_transformer_gpu_fallback_shared_expert_mid(
                model, session, layer, state->xb, cpu_mid) == 0 &&
            bn_transformer_gpu_read_activation_buf(
                gpu, BN_GPU_VALUE_HB, gpu_mid, bytes) == 0)
            bn_transformer_gpu_debug_compare_vec(
                "moe_shared_mid_compare", layer_index, pos,
                cpu_mid, gpu_mid, hidden);
        free(cpu_mid);
        free(gpu_mid);
    }
    if (debug->compare_shared_down && has_shared) {
        size_t bytes = (size_t)dim * sizeof(float);
        float *cpu_down = (float *)malloc(bytes);
        float *gpu_down = (float *)malloc(bytes);
        if (cpu_down && gpu_down &&
            bn_transformer_gpu_fallback_shared_expert_down(
                model, session, layer, dim, state->xb, cpu_down) == 0 &&
            bn_transformer_gpu_read_activation_buf(
                gpu, BN_GPU_VALUE_XB2, gpu_down, bytes) == 0)
            bn_transformer_gpu_debug_compare_vec(
                "moe_shared_down_compare", layer_index, pos,
                cpu_down, gpu_down, dim);
        free(cpu_down);
        free(gpu_down);
    }
    if (debug->compare_norm) {
        size_t bytes = (size_t)dim * sizeof(float);
        float *cpu_norm = (float *)malloc(bytes);
        float *gpu_norm = (float *)malloc(bytes);
        const float *norm_weight =
            layer_index + 1 < model->config.n_layers
                ? model->weights.layers[layer_index + 1].norm.attn_norm
                : model->weights.output_norm;
        if (cpu_norm && gpu_norm &&
            bn_transformer_gpu_read_xb(gpu, gpu_norm, bytes) == 0) {
            if (norm_weight) {
                bn_transformer_gpu_debug_rmsnorm(
                    cpu_norm, cpu_state, norm_weight, dim, norm_eps);
                bn_transformer_gpu_debug_compare_vec(
                    "moe_routed_norm_compare", layer_index, pos,
                    cpu_norm, gpu_norm, dim);
            }
        }
        free(cpu_norm);
        free(gpu_norm);
    }
}

int bn_transformer_gpu_complete_routed_moe_debug_state(
    BnTransformerGPURoutedMoEDebugState *debug_state,
    BnTransformerGPUMoEPartsComparison *parts,
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnTransformerGPUMoEDebugPolicy *debug,
    void *next_norm,
    int layer_index,
    int pos,
    int dim,
    uint32_t u_eps,
    float norm_eps) {
    if (!debug_state || !parts)
        return -1;
    if (!emit || !gpu || !model || !session || !layer || !debug ||
        dim <= 0) {
        bn_transformer_gpu_discard_routed_moe_parts_comparison(parts);
        bn_transformer_gpu_discard_routed_moe_debug_state(debug_state);
        return -1;
    }
    size_t bytes = (size_t)dim * sizeof(float);
    if (debug_state->compare_enabled) {
        if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
            bn_transformer_gpu_read_x(
                gpu, debug_state->gpu_state, bytes) != 0) {
            bn_transformer_gpu_discard_routed_moe_parts_comparison(parts);
            bn_transformer_gpu_discard_routed_moe_debug_state(debug_state);
            return -1;
        }
        bn_transformer_gpu_debug_compare_vec(
            "moe_routed_state_compare", layer_index, pos,
            debug_state->cpu_state, debug_state->gpu_state, dim);
        if (debug->compare_actual && debug_state->input_state &&
            bn_moe_execution_policy(
                &model->config).uses_dense_residual_branch &&
            layer->norm.ffn_post_norm_1) {
            int hidden = model->config.hidden_dim;
            float *input = (float *)malloc(bytes);
            float *gate = hidden > 0
                ? (float *)malloc((size_t)hidden * sizeof(float)) : NULL;
            float *up = hidden > 0
                ? (float *)malloc((size_t)hidden * sizeof(float)) : NULL;
            float *gpu_activation = hidden > 0
                ? (float *)malloc((size_t)hidden * sizeof(float)) : NULL;
            float *cpu_dense = (float *)malloc(bytes);
            float *cpu_dense_norm = (float *)malloc(bytes);
            float *gpu_dense_raw = (float *)malloc(bytes);
            float *gpu_dense = (float *)malloc(bytes);
            if (input && gate && up && gpu_activation && cpu_dense &&
                cpu_dense_norm && gpu_dense_raw && gpu_dense) {
                BnMoEExecutionPolicy policy =
                    bn_moe_execution_policy(&model->config);
                fallback_rmsnorm(
                    input, debug_state->input_state,
                    layer->norm.ffn_norm, dim, policy.norm_eps);
                bn_transformer_gpu_cpu_quant_matvec_model(
                    model, gate, &layer->ffn.ffn_gate, input,
                    session->state.x_q);
                bn_transformer_gpu_cpu_quant_matvec_model(
                    model, up, &layer->ffn.ffn_up, input,
                    session->state.x_q);
                bn_moe_swiglu(
                    gate, gate, up, hidden, -1,
                    policy.uses_reference_ffn_activation);
                if (bn_transformer_gpu_read_activation_buf(
                        gpu, BN_GPU_VALUE_HB2, gpu_activation,
                        (size_t)hidden * sizeof(float)) == 0)
                    bn_transformer_gpu_debug_compare_vec(
                        "moe_dense_residual_activation_compare",
                        layer_index, pos, gate, gpu_activation, hidden);
                bn_transformer_gpu_cpu_quant_matvec_model(
                    model, cpu_dense, &layer->ffn.ffn_down, gate,
                    session->state.x_q);
                if (bn_transformer_gpu_read_activation_buf(
                        gpu, BN_GPU_VALUE_XB2, gpu_dense_raw, bytes) == 0)
                    bn_transformer_gpu_debug_compare_vec(
                        "moe_dense_residual_down_compare", layer_index, pos,
                        cpu_dense, gpu_dense_raw, dim);
                fallback_rmsnorm(cpu_dense_norm, cpu_dense,
                                 layer->norm.ffn_post_norm_1,
                                 dim, policy.norm_eps);
                if (bn_transformer_gpu_read_activation_buf(
                        gpu, BN_GPU_VALUE_HB, gpu_dense, bytes) == 0)
                    bn_transformer_gpu_debug_compare_vec(
                        "moe_dense_residual_compare", layer_index, pos,
                        cpu_dense_norm, gpu_dense, dim);
            }
            free(input);
            free(gate);
            free(up);
            free(gpu_activation);
            free(cpu_dense);
            free(cpu_dense_norm);
            free(gpu_dense_raw);
            free(gpu_dense);
        }
        bn_transformer_gpu_compare_routed_moe_shared_part(
            parts, debug_state->gpu_state, session->state.x,
            layer_index, pos, dim);
        bn_transformer_gpu_debug_compare_routed_moe_post_layer(
            gpu, model, session, layer, debug, debug_state->cpu_state,
            layer_index, pos, dim, norm_eps);
    }
    bn_transformer_gpu_discard_routed_moe_parts_comparison(parts);
    if (debug_state->override_enabled &&
        (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
         bn_transformer_gpu_write_x(
             gpu, debug_state->override_state, bytes) != 0 ||
         bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
             emit, next_norm, dim, u_eps) != 0)) {
        bn_transformer_gpu_discard_routed_moe_debug_state(debug_state);
        return -2;
    }
    bn_transformer_gpu_discard_routed_moe_debug_state(debug_state);
    return 0;
}

void bn_transformer_gpu_discard_moe_layer_comparison(
    BnTransformerGPUMoELayerComparison *comparison) {
    if (!comparison)
        return;
    free(comparison->cpu_state);
    free(comparison->gpu_state);
    free(comparison->input_state);
    memset(comparison, 0, sizeof(*comparison));
}

int bn_transformer_gpu_prepare_moe_layer_comparison(
    BnTransformerGPUMoELayerComparison *comparison,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const BnTransformerGPUMoEDebugPolicy *debug,
    int layer_index,
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
    comparison->input_state =
        (float *)malloc((size_t)dim * sizeof(float));
    comparison->enabled = 1;
    comparison->compare_norm = debug->compare_norm;
    BnRunState *state = &session->state;
    BnMoERoutePolicy route_policy = bn_moe_route_policy(&model->config);
    int active_experts = route_policy.active_experts;
    int saved_indices[BN_MAX_MOE_K];
    float saved_weights[BN_MAX_MOE_K];
    if (!session->moe_state || active_experts < 0 ||
        active_experts > BN_MAX_MOE_K) {
        bn_transformer_gpu_discard_moe_layer_comparison(comparison);
        return -1;
    }
    for (int k = 0; k < active_experts; k++) {
        saved_indices[k] = session->moe_state->expert_indices[k];
        saved_weights[k] = session->moe_state->expert_weights[k];
    }
    int reference_rc = -1;
    if (comparison->cpu_state && comparison->gpu_state &&
        comparison->input_state &&
        bn_transformer_gpu_read_x(
            gpu, state->x, (size_t)dim * sizeof(float)) == 0 &&
        bn_transformer_gpu_read_xb(
            gpu, comparison->input_state,
            (size_t)dim * sizeof(float)) == 0) {
        memcpy(state->xb, comparison->input_state,
               (size_t)dim * sizeof(float));
        reference_rc = debug->compare_actual
            ? bn_transformer_gpu_fallback_moe_output_from_state(
                  model, session, layer, layer_index, dim,
                  comparison->cpu_state)
            : bn_transformer_gpu_fallback_moe_output(
                  model, session, layer, dim, state->x,
                  comparison->input_state,
                  comparison->cpu_state);
    }
    if (reference_rc != 0) {
        for (int k = 0; k < active_experts; k++) {
            session->moe_state->expert_indices[k] = saved_indices[k];
            session->moe_state->expert_weights[k] = saved_weights[k];
        }
        bn_transformer_gpu_discard_moe_layer_comparison(comparison);
        return -1;
    }
    for (int k = 0; k < active_experts; k++) {
        session->moe_state->expert_indices[k] = saved_indices[k];
        session->moe_state->expert_weights[k] = saved_weights[k];
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
                      activation_policy.uses_reference_silu,
                      activation_policy.uses_reference_ffn_activation);
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
                      activation_policy.uses_reference_silu,
                      activation_policy.uses_reference_ffn_activation);
        bn_transformer_gpu_cpu_quant_matvec_model(
            model, down, &down_weight, gate, session->state.x_q);
        bn_moe_weighted_add(
            output, down,
            moe_state->expert_weights[k] *
                bn_moe_expert_weight_scale(layer, expert),
            dim);
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

int bn_transformer_gpu_fallback_moe_dense_residual_branch(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    int dim) {
    if (!emit || !gpu || !model || !session || !layer || dim <= 0)
        return -1;
    BnMoEExecutionPolicy policy =
        bn_moe_execution_policy(&model->config);
    if (!policy.uses_dense_residual_branch)
        return 0;
    int hidden = model->config.hidden_dim;
    if (hidden <= 0 || !layer->ffn.ffn_gate.data ||
        !layer->ffn.ffn_up.data || !layer->ffn.ffn_down.data)
        return -1;

    size_t dim_bytes = (size_t)dim * sizeof(float);
    float *output = (float *)malloc(dim_bytes);
    float *residual = (float *)malloc(dim_bytes);
    float *input = (float *)malloc(dim_bytes);
    float *gate = (float *)malloc((size_t)hidden * sizeof(float));
    float *up = (float *)malloc((size_t)hidden * sizeof(float));
    float *down = (float *)malloc(dim_bytes);
    if (!output || !residual || !input || !gate || !up || !down) {
        free(output);
        free(residual);
        free(input);
        free(gate);
        free(up);
        free(down);
        return -1;
    }

    int rc = -1;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_MOE_OUT, output, dim_bytes) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_X, residual, dim_bytes) != 0)
        goto cleanup;

    if (layer->norm.ffn_post_norm_2)
        fallback_rmsnorm(output, output, layer->norm.ffn_post_norm_2,
                         dim, policy.norm_eps);
    fallback_rmsnorm(input, residual, layer->norm.ffn_norm,
                     dim, policy.norm_eps);
    bn_transformer_gpu_cpu_quant_matvec_model(
        model, gate, &layer->ffn.ffn_gate, input,
        session->state.x_q);
    bn_transformer_gpu_cpu_quant_matvec_model(
        model, up, &layer->ffn.ffn_up, input,
        session->state.x_q);
    bn_moe_swiglu(gate, gate, up, hidden, -1,
                  policy.uses_reference_ffn_activation);
    bn_transformer_gpu_cpu_quant_matvec_model(
        model, down, &layer->ffn.ffn_down, gate,
        session->state.x_q);
    if (layer->norm.ffn_post_norm_1)
        fallback_rmsnorm(down, down, layer->norm.ffn_post_norm_1,
                         dim, policy.norm_eps);
    bn_moe_weighted_add(output, down, 1.0f, dim);
    if (layer->norm.ffn_post_norm)
        fallback_rmsnorm(output, output, layer->norm.ffn_post_norm,
                         dim, policy.norm_eps);
    if (bn_transformer_gpu_write_activation_buf(
            gpu, BN_GPU_VALUE_MOE_OUT, output, dim_bytes) != 0)
        goto cleanup;
    rc = 0;

cleanup:
    free(output);
    free(residual);
    free(input);
    free(gate);
    free(up);
    free(down);
    return rc;
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
                  activation_policy.uses_reference_silu,
                  activation_policy.uses_reference_ffn_activation);
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

static int fallback_shared_expert_residual_impl(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const float *resolved_input,
    int dim) {
    if (!emit || !gpu || !model || !session || !layer || dim <= 0)
        return -1;
    size_t dim_bytes = (size_t)dim * sizeof(float);
    float *input = (float *)malloc(dim_bytes);
    float *shared = (float *)malloc(dim_bytes);
    float *routed = (float *)malloc(dim_bytes);
    if (!input || !shared || !routed ||
        bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        (resolved_input
             ? (memcpy(input, resolved_input, dim_bytes), 0)
             : bn_transformer_gpu_read_xb(gpu, input, dim_bytes)) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_MOE_OUT, routed, dim_bytes) != 0 ||
        bn_transformer_gpu_fallback_shared_expert_output(
            model, session, layer, dim, input, shared) != 0) {
        free(input);
        free(shared);
        free(routed);
        return -1;
    }
    for (int i = 0; i < dim; i++)
        routed[i] += shared[i];
    int rc = bn_transformer_gpu_write_activation_buf(
        gpu, BN_GPU_VALUE_MOE_OUT, routed, dim_bytes);
    free(input);
    free(shared);
    free(routed);
    return rc == 0 ? 0 : -2;
}

int bn_transformer_gpu_fallback_shared_expert_residual(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    int dim) {
    return fallback_shared_expert_residual_impl(
        emit, gpu, model, session, layer, NULL, dim);
}

int bn_transformer_gpu_fallback_shared_expert_residual_from_input(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *model,
    BnSession *session,
    BnLayerWeights *layer,
    const float *input,
    int dim) {
    if (!input) return -1;
    return fallback_shared_expert_residual_impl(
        emit, gpu, model, session, layer, input, dim);
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

int bn_transformer_gpu_fallback_ssm_layers(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    int layer_start,
    int layer_end,
    int dim,
    uint32_t u_eps,
    void *next_norm) {
    if (!m || !sess || layer_start < 0 || layer_end <= layer_start ||
        layer_end > m->config.n_layers)
        return -1;
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;

    BnRunState *s = &sess->state;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;

    int cpu_only = 1;
    for (int layer = layer_start; layer < layer_end; layer++) {
        BnTransformerGPULayerKindPolicy layer_kind =
            bn_transformer_gpu_layer_kind_policy(&m->weights.layers[layer]);
        if (layer_kind.uses_moe) {
            cpu_only = 0;
            break;
        }
    }
    if (cpu_only)
        bn_model_set_gpu_disabled(m, 1);
    for (int layer = layer_start; layer < layer_end; layer++) {
        BnLayerWeights *lw = &m->weights.layers[layer];
        bn_transformer_cpu_forward_ssm_block(
            m, sess, lw, layer, sess->pos);
        bn_transformer_cpu_residual_add(
            fallback_cpu_runtime(m), s->x, s->xb, dim);
        BnTransformerGPULayerKindPolicy layer_kind =
            bn_transformer_gpu_layer_kind_policy(lw);
        if (layer_kind.uses_moe)
            bn_moe_forward(m, sess, lw, layer);
        else
            bn_transformer_cpu_forward_ffn_block(
                m, sess, lw, layer, sess->pos, NULL);
    }
    if (cpu_only)
        bn_model_set_gpu_disabled(m, 0);

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
    if (!shape.is_attn ||
        !bn_transformer_kv_host_float_cache_rows_available(c))
        return -1;

    int profile = bn_transformer_gpu_profile_level(gpu) >= 3;
    double t0 = profile ? bn_platform_time_ms() : 0.0;

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    if (bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0)
        return -1;
    double t_read = profile ? bn_platform_time_ms() : 0.0;

    int head_size = shape.head_size;
    int n_heads = shape.n_heads;
    int kv_dim = shape.kv_dim;
    int n_kv_heads = shape.n_kv_heads;
    int kv_mul = shape.kv_mul;
    BnAttentionPlan attn_plan;
    bn_transformer_plan_attention(
        &attn_plan, c, lw, NULL, bn_model_backend(m), layer,
        bn_model_tq_state(m) != NULL, 0);
    int layer_rope_dims = rope_dims > head_size ? head_size : rope_dims;
    size_t loff = (size_t)shape.attn_idx * c->seq_len * c->kv_dim;
    float *key_cache_row =
        s->key_cache + loff + (size_t)cache_pos * c->kv_dim;
    float *value_cache_row =
        s->value_cache + loff + (size_t)cache_pos * c->kv_dim;

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim,
                     bn_transformer_gpu_norm_epsilon(c));
    float *q_full = shape.q_gated ? s->hb : s->q;
    {
        BnMatvecTask qkv[3] = {
            { q_full, &lw->attn.wq, NULL, 0 },
            { key_cache_row, &lw->attn.wk, NULL, 0 },
            { value_cache_row, &lw->attn.wv, NULL, 0 },
        };
        fallback_cpu_matvec_batch(m, qkv, 3, s->xb, s->x_q);
    }
    double t_qkv = profile ? bn_platform_time_ms() : 0.0;

    if (shape.q_gated) {
        for (int h = 0; h < n_heads; h++)
            memcpy(s->q + (size_t)h * head_size,
                   q_full + (size_t)h * 2 * head_size,
                   (size_t)head_size * sizeof(float));
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
    if (shape.value_shares_key) {
        float eps = bn_transformer_gpu_norm_epsilon(c);
        for (int h = 0; h < n_kv_heads; h++) {
            float *vh = value_cache_row + (size_t)h * head_size;
            float ss = 0.0f;
            for (int i = 0; i < head_size; i++)
                ss += vh[i] * vh[i];
            float scale = 1.0f / sqrtf(ss / (float)head_size + eps);
            for (int i = 0; i < head_size; i++)
                vh[i] *= scale;
        }
    }

    bn_transformer_cpu_apply_rope_heads(fallback_cpu_runtime(m), s->q,
                                        n_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);
    bn_transformer_cpu_apply_rope_heads(fallback_cpu_runtime(m), key_cache_row,
                                        n_kv_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);

    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
    BnGQACtx gctx = {
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len,
        bn_transformer_attention_scale(c, head_size),
        bn_transformer_kv_host_cache_uses_fp16_rows(c)
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);
    double t_gqa = profile ? bn_platform_time_ms() : 0.0;

    if (shape.q_gated) {
        const BnCPUBackendOps *cpu_ops = bn_transformer_cpu_backend_ops(
            fallback_cpu_runtime(m));
        for (int h = 0; h < n_heads; h++)
            cpu_ops->apply_sigmoid_gate(
                s->xb + (size_t)h * head_size,
                q_full + (size_t)h * 2 * head_size + head_size,
                head_size);
    }
    if (lw->norm.attn_sub_norm)
        fallback_rmsnorm(s->xb, s->xb, lw->norm.attn_sub_norm,
                         dim, bn_transformer_gpu_norm_epsilon(c));

    {
        BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL, 0 }};
        fallback_cpu_matvec_batch(m, wo, 1, s->xb, s->x_q);
    }
    if (attn_plan.use_post_norm)
        fallback_rmsnorm(s->xb2, s->xb2, lw->norm.attn_post_norm,
                         dim, bn_transformer_gpu_norm_epsilon(c));
    bn_transformer_cpu_residual_add(
        fallback_cpu_runtime(m), s->x, s->xb2, dim);

    double t_out = profile ? bn_platform_time_ms() : 0.0;

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
    int rc = bn_transformer_gpu_emit_context_x_to_xb_rmsnorm(
        emit, next_norm, dim, u_eps);
    if (profile) {
        double t_done = bn_platform_time_ms();
        fprintf(stderr,
                "[gpu:fallback:attention] layer=%d pos=%d "
                "handoff=%.3fms qkv=%.3fms gqa=%.3fms out=%.3fms "
                "write=%.3fms total=%.3fms\n",
                layer, pos, t_read - t0, t_qkv - t_read,
                t_gqa - t_qkv, t_out - t_gqa, t_done - t_out,
                t_done - t0);
    }
    return rc;
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
    bn_transformer_cpu_apply_ffn_activation(
        fallback_cpu_runtime(m), s, ffn_plan, hidden_dim, 0);
    if (ffn_plan->has_sub_norm)
        fallback_rmsnorm(s->hb, s->hb, lw->norm.ffn_sub_norm,
                         hidden_dim,
                         bn_transformer_gpu_norm_epsilon(&m->config));
    fallback_cpu_matvec(m, s->xb, &lw->ffn.ffn_down, s->hb, s->x_q);
    if (ffn_plan->use_post_norm)
        fallback_rmsnorm(s->xb, s->xb, lw->norm.ffn_post_norm,
                         dim, bn_transformer_gpu_norm_epsilon(&m->config));
    bn_transformer_cpu_residual_add(
        fallback_cpu_runtime(m), s->x, s->xb, dim);
}

static void fallback_cpu_apply_per_layer_input(BnModel *m,
                                               BnSession *sess,
                                               BnLayerWeights *lw,
                                               int layer,
                                               int dim) {
    BnRunState *s = &sess->state;
    int per_dim = bn_transformer_per_layer_embedding_dim(&m->config);
    if (per_dim <= 0 || !s->per_layer_input ||
        !lw->per_layer.inp_gate.data || !lw->per_layer.proj.data ||
        !lw->per_layer.post_norm)
        return;

    memcpy(s->xb2, s->x, (size_t)dim * sizeof(float));
    fallback_cpu_matvec(m, s->hb, &lw->per_layer.inp_gate, s->x, s->x_q);
    for (int i = 0; i < per_dim; i++) {
        float g = s->hb[i];
        s->hb[i] = fallback_reference_gelu(g) *
            s->per_layer_input[(size_t)layer * per_dim + i];
    }
    fallback_cpu_matvec(m, s->x, &lw->per_layer.proj, s->hb, s->x_q);
    fallback_rmsnorm(s->x, s->x, lw->per_layer.post_norm, dim,
                     bn_transformer_gpu_norm_epsilon(&m->config));
    bn_transformer_cpu_residual_add(
        fallback_cpu_runtime(m), s->x, s->xb2, dim);
}

int bn_transformer_gpu_fallback_cpu_ffn(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const BnFFNPlan *ffn_plan,
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
    if (bn_transformer_gpu_read_xb(gpu, s->xb,
                                   (size_t)dim * sizeof(float)) != 0)
        return -1;
    fallback_cpu_forward_ffn_from_xb(m, sess, lw, ffn_plan, dim);
    fallback_cpu_apply_per_layer_input(m, sess, lw, layer, dim);
    if (ffn_plan->use_layer_output_scale) {
        float scale = lw->norm.layer_output_scale[0];
        for (int i = 0; i < dim; i++)
            s->x[i] *= scale;
    }
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
    const BnFFNPlan *ffn_plan,
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
    if (ffn_plan && ffn_plan->use_post_norm)
        fallback_rmsnorm(s->xb, s->xb, lw->norm.ffn_post_norm,
                         dim, bn_transformer_gpu_norm_epsilon(&m->config));
    bn_transformer_cpu_residual_add(
        fallback_cpu_runtime(m), s->x, s->xb, dim);
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
    const BnFFNPlan *ffn_plan,
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

    if (ffn_plan && ffn_plan->use_post_norm) {
        fallback_rmsnorm(s->xb, s->xb, lw->norm.ffn_post_norm, dim,
                         bn_transformer_gpu_norm_epsilon(&m->config));
        if (bn_transformer_gpu_read_activation_buf(
                gpu, BN_GPU_VALUE_SCRATCH, s->xb2,
                (size_t)dim * sizeof(float)) != 0)
            return -1;
        bn_transformer_gpu_debug_compare_vec(
            "ffn_post_norm_compare", layer, pos, s->xb, s->xb2, dim);
    }
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

void bn_transformer_gpu_moe_route_profile_add(const BnGPUBackend *gpu,
                                              int dim,
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
    if (!bn_transformer_gpu_moe_route_profile_enabled(gpu))
        return;
    calls++;
    total_flush += flush_ms;
    total_read += read_ms;
    total_route += route_ms;
    total_resolve += resolve_ms;
    int every = bn_transformer_gpu_moe_route_profile_every(gpu);
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
    if (!bn_transformer_gpu_debug_argmax_compare_enabled(gpu) ||
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
    if (!bn_transformer_gpu_compare_logits_enabled(gpu) ||
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
            fallback_cpu_runtime(m), &m->config, bn_model_backend(m), w);
    return resource.prepared;
}

static void debug_quant_matvec_prepared(BnModel *m,
                                        float *out,
                                        const BnQWeight *W,
                                        const float *x,
                                        int8_t *quantized_buf,
                                        int force_float_kquant) {
    bn_transformer_cpu_quant_matvec_prepared_flags(
        out, W, debug_prepared_qweight(m, W), x, quantized_buf,
        bn_model_pool(m), force_float_kquant
            ? BN_MATVEC_TASK_FORCE_FLOAT_KQUANT : 0);
}

static void debug_quant_matvec_batch_prepared(BnModel *m,
                                               const BnMatvecTask *tasks,
                                               int n_tasks,
                                               const float *x,
                                               int8_t *quantized_buf,
                                               int force_float_kquant) {
    if (!force_float_kquant) {
        fallback_cpu_matvec_batch(m, tasks, n_tasks, x, quantized_buf);
        return;
    }
    for (int i = 0; i < n_tasks; i++)
        debug_quant_matvec_prepared(m, tasks[i].out, tasks[i].W, x,
                                    quantized_buf, 1);
}

static void debug_compare_native_quant_activation(const BnGPUBackend *gpu,
                                                 int layer,
                                                 int pos,
                                                 const float *x,
                                                 int cols) {
    if (!gpu || !x || cols <= 0 || (cols % 32) != 0 ||
        !bn_transformer_cpu_has_native_quant_activation())
        return;
    int n_blocks = cols / BN_QK_K;
    int8_t *cpu_q = (int8_t *)malloc((size_t)cols);
    int8_t *gpu_q = (int8_t *)malloc((size_t)cols);
    float *cpu_scales = (float *)malloc((size_t)n_blocks * sizeof(float));
    float *gpu_scales = (float *)malloc((size_t)n_blocks * sizeof(float));
    int16_t *cpu_bsums =
        (int16_t *)malloc((size_t)n_blocks * 16 * sizeof(int16_t));
    int16_t *gpu_bsums =
        (int16_t *)malloc((size_t)n_blocks * 16 * sizeof(int16_t));
    if (!cpu_q || !gpu_q || !cpu_scales || !gpu_scales || !cpu_bsums ||
        !gpu_bsums) {
        free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
        free(cpu_bsums); free(gpu_bsums);
        return;
    }

    bn_transformer_cpu_prepare_kquant_activation(
        x, cpu_q, cpu_scales, cpu_bsums, cols);
    if (bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_DEBUG_BUF_NATIVE_QUANT_ACT, gpu_q,
            (size_t)cols) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_DEBUG_BUF_NATIVE_QUANT_SCALE, gpu_scales,
            (size_t)n_blocks * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_DEBUG_BUF_NATIVE_QUANT_BLOCK_SUM, gpu_bsums,
            (size_t)n_blocks * 16 * sizeof(int16_t)) != 0) {
        free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
        free(cpu_bsums); free(gpu_bsums);
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
    int bsum_diff_count = 0;
    int max_bsum_abs = 0;
    for (int i = 0; i < n_blocks * 16; i++) {
        int diff = (int)gpu_bsums[i] - (int)cpu_bsums[i];
        int ad = diff < 0 ? -diff : diff;
        if (ad) bsum_diff_count++;
        if (ad > max_bsum_abs) max_bsum_abs = ad;
    }
    fprintf(stderr,
            "[bn:gpu:debug] native_quant_bsum_compare layer=%d pos=%d "
            "diff_count=%d max_abs=%d\n",
            layer, pos, bsum_diff_count, max_bsum_abs);

    free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
    free(cpu_bsums); free(gpu_bsums);
}

static void debug_compare_native_block_activation(const BnGPUBackend *gpu,
                                                  int layer,
                                                  int pos,
                                                  const float *x,
                                                  int cols) {
    if (!gpu || !x || cols <= 0 || (cols % 32) != 0)
        return;
    int n_blocks = cols / 32;
    int8_t *cpu_q = malloc((size_t)cols);
    int8_t *gpu_q = malloc((size_t)cols);
    float *cpu_scales = malloc((size_t)n_blocks * sizeof(float));
    float *gpu_scales = malloc((size_t)n_blocks * sizeof(float));
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
    int q_diff = 0;
    int q_max = 0;
    int q_max_i = 0;
    float scale_max = 0.0f;
    int scale_max_i = 0;
    for (int i = 0; i < cols; i++) {
        int d = (int)gpu_q[i] - (int)cpu_q[i];
        int ad = d < 0 ? -d : d;
        if (ad) q_diff++;
        if (ad > q_max) {
            q_max = ad;
            q_max_i = i;
        }
    }
    for (int i = 0; i < n_blocks; i++) {
        float d = fabsf(gpu_scales[i] - cpu_scales[i]);
        if (d > scale_max) {
            scale_max = d;
            scale_max_i = i;
        }
    }
    fprintf(stderr,
            "[bn:gpu:debug] native_quant_block_compare layer=%d pos=%d "
            "q_diff_count=%d q_max_abs=%d q_max_i=%d cpu_q=%d gpu_q=%d "
            "scale_max_abs=%.9g scale_max_i=%d cpu_scale=%.9g "
            "gpu_scale=%.9g\n",
            layer, pos, q_diff, q_max, q_max_i, (int)cpu_q[q_max_i],
            (int)gpu_q[q_max_i], scale_max, scale_max_i,
            cpu_scales[scale_max_i], gpu_scales[scale_max_i]);
    free(cpu_q); free(gpu_q); free(cpu_scales); free(gpu_scales);
}

int bn_transformer_gpu_debug_snapshot_attention_state(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnSession *session,
    int dim) {
    if (!emit || !gpu || !session || dim <= 0)
        return -1;
    return bn_transformer_gpu_emit_context_flush(emit, gpu) == 0 &&
           bn_transformer_gpu_read_x(
               gpu, session->state.x,
               (size_t)dim * sizeof(float)) == 0
        ? 0 : -1;
}

int bn_transformer_gpu_debug_snapshot_ffn_state(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnSession *session,
    int dim) {
    if (!emit || !gpu || !session || dim <= 0)
        return -1;
    size_t bytes = (size_t)dim * sizeof(float);
    return bn_transformer_gpu_emit_context_flush(emit, gpu) == 0 &&
           bn_transformer_gpu_read_x(
               gpu, session->state.x, bytes) == 0 &&
           bn_transformer_gpu_read_xb(
               gpu, session->state.xb, bytes) == 0
        ? 0 : -1;
}

int bn_transformer_gpu_capture_logits_refine_state(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnSession *session,
    int dim) {
    if (!emit || !gpu || !session || dim <= 0)
        return -1;
    return bn_transformer_gpu_emit_context_flush(emit, gpu) == 0 &&
           bn_transformer_gpu_read_xb(
               gpu, session->state.xb,
               (size_t)dim * sizeof(float)) == 0
        ? 0 : -1;
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
    float *cpu_residual = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_x = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_xb = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_residual = (float *)malloc((size_t)dim * sizeof(float));
    float *cpu_hb = NULL;
    float *cpu_hb2 = NULL;
    float *gpu_hb = NULL;
    int hidden_dim = ffn_plan ? ffn_plan->hidden_dim : 0;
    fprintf(stderr,
            "[bn:gpu:debug] ffn_types layer=%d gate=%d up=%d down=%d\n",
            layer, lw->ffn.ffn_gate.type, lw->ffn.ffn_up.type,
            lw->ffn.ffn_down.type);
    if (hidden_dim > 0) {
        cpu_hb = (float *)malloc((size_t)hidden_dim * sizeof(float));
        cpu_hb2 = (float *)malloc((size_t)hidden_dim * sizeof(float));
        gpu_hb = (float *)malloc((size_t)hidden_dim * sizeof(float));
    }
    if (!cpu_x_in || !cpu_xb_in || !cpu_x || !cpu_xb || !cpu_residual ||
        !gpu_x || !gpu_xb ||
        !gpu_residual ||
        (hidden_dim > 0 && (!cpu_hb || !cpu_hb2 || !gpu_hb))) {
        free(cpu_x_in);
        free(cpu_xb_in);
        free(cpu_x);
        free(cpu_xb);
        free(cpu_residual);
        free(gpu_x);
        free(gpu_xb);
        free(gpu_residual);
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
        free(cpu_residual);
        free(gpu_x);
        free(gpu_xb);
        free(gpu_residual);
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
                                        s->xb, s->x_q, 0);
            debug_quant_matvec_prepared(m, cpu_hb2, &lw->ffn.ffn_up,
                                        s->xb, s->x_q, 0);
        } else {
            debug_quant_matvec_prepared(m, cpu_hb, &lw->ffn.ffn_up,
                                        s->xb, s->x_q, 0);
        }
        BnRunState activation_state = {0};
        activation_state.hb = cpu_hb;
        activation_state.hb2 = cpu_hb2;
        bn_transformer_cpu_apply_ffn_activation(
            fallback_cpu_runtime(m), &activation_state,
            ffn_plan, hidden_dim, 0);
    }
    fallback_cpu_forward_ffn_from_xb(m, sess, lw, ffn_plan, dim);
    memcpy(cpu_x, s->x, (size_t)dim * sizeof(float));
    for (int i = 0; i < dim; i++)
        cpu_residual[i] = cpu_x[i] - cpu_x_in[i];
    if (next_norm)
        fallback_rmsnorm(cpu_xb, cpu_x, next_norm, dim,
                         bn_transformer_gpu_norm_epsilon(&m->config));

    if ((hidden_dim > 0 &&
         bn_transformer_gpu_read_activation_buf(
             gpu, BN_GPU_VALUE_HB, gpu_hb,
             (size_t)hidden_dim * sizeof(float)) != 0) ||
        (hidden_dim > 0 && ffn_plan->has_gate && !ffn_plan->has_sub_norm &&
         bn_transformer_gpu_read_activation_buf(
             gpu, BN_GPU_VALUE_HB2, s->hb2,
             (size_t)hidden_dim * sizeof(float)) != 0) ||
        bn_transformer_gpu_read_x(gpu, gpu_x,
                                  (size_t)dim * sizeof(float)) != 0 ||
        (next_norm && bn_transformer_gpu_read_xb(gpu, gpu_xb,
                                                 (size_t)dim * sizeof(float)) != 0) ||
        (ffn_plan->use_post_norm &&
         bn_transformer_gpu_read_activation_buf(
             gpu, BN_GPU_VALUE_SCRATCH, gpu_residual,
             (size_t)dim * sizeof(float)) != 0)) {
        free(cpu_x_in);
        free(cpu_xb_in);
        free(cpu_x);
        free(cpu_xb);
        free(cpu_residual);
        free(gpu_x);
        free(gpu_xb);
        free(gpu_residual);
        free(cpu_hb);
        free(cpu_hb2);
        free(gpu_hb);
        return -1;
    }
    if (hidden_dim > 0) {
        if (ffn_plan->has_gate && !ffn_plan->has_sub_norm)
            bn_transformer_gpu_debug_compare_vec(
                "ffn_up_compare", layer, pos, cpu_hb2, s->hb2,
                hidden_dim);
        bn_transformer_gpu_debug_compare_vec(
            "ffn_hidden_compare", layer, pos, cpu_hb, gpu_hb, hidden_dim);
    }
    bn_transformer_gpu_debug_compare_vec(
        "ffn_state_compare", layer, pos, cpu_x, gpu_x, dim);
    if (ffn_plan->use_post_norm) {
        bn_transformer_gpu_debug_compare_vec(
            "ffn_post_norm_state_compare", layer, pos,
            cpu_residual, gpu_residual, dim);
        for (int i = 0; i < dim; i++)
            gpu_residual[i] += cpu_x_in[i];
        bn_transformer_gpu_debug_compare_vec(
            "ffn_residual_identity_compare", layer, pos,
            gpu_residual, gpu_x, dim);
    }
    if (next_norm)
        bn_transformer_gpu_debug_compare_vec(
            "ffn_next_norm_compare", layer, pos, cpu_xb, gpu_xb, dim);

    memcpy(s->x, gpu_x, (size_t)dim * sizeof(float));
    free(cpu_x_in);
    free(cpu_xb_in);
    free(cpu_x);
    free(cpu_xb);
    free(cpu_residual);
    free(gpu_x);
    free(gpu_xb);
    free(gpu_residual);
    free(cpu_hb);
    free(cpu_hb2);
    free(gpu_hb);
    return 0;
}

int bn_transformer_gpu_debug_snapshot_ssm_state(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    int dim) {
    if (!emit || !gpu || !m || !sess || dim <= 0)
        return -1;
    BnTransformerSSMShapePolicy shape;
    int n_ssm = bn_transformer_ssm_layer_count(&m->config);
    if (n_ssm <= 0 ||
        !bn_transformer_ssm_shape_policy(&shape, &m->config))
        return -1;
    size_t state_values = (size_t)n_ssm * shape.num_v_heads *
                          shape.head_k_dim * shape.head_v_dim;
    size_t conv_values = (size_t)n_ssm * (shape.conv_kernel - 1) *
                         shape.qkv_dim;
    BnRunState *s = &sess->state;
    if (!s->ssm_state || !s->ssm_conv_state ||
        bn_transformer_gpu_emit_context_flush(emit, gpu) != 0)
        return -1;
    return bn_transformer_gpu_read_x(
               gpu, s->x, (size_t)dim * sizeof(float)) == 0 &&
           bn_transformer_gpu_read_xb(
               gpu, s->xb, (size_t)dim * sizeof(float)) == 0 &&
           bn_transformer_gpu_read_activation_buf(
               gpu, BN_GPU_VALUE_SSM_STATE, s->ssm_state,
               state_values * sizeof(float)) == 0 &&
           bn_transformer_gpu_read_activation_buf(
               gpu, BN_GPU_VALUE_SSM_CONV_STATE, s->ssm_conv_state,
               conv_values * sizeof(float)) == 0 ? 0 : -1;
}

int bn_transformer_gpu_debug_compare_ssm(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    const BnTransformerGPUSSMResources *res,
    int layer,
    int pos,
    int dim) {
    BnTransformerSSMShapePolicy shape;
    int n_ssm = bn_transformer_ssm_layer_count(&m->config);
    if (!emit || !gpu || !m || !sess || !lw || n_ssm <= 0 ||
        !bn_transformer_ssm_shape_policy(&shape, &m->config))
        return -1;
    fprintf(stderr,
            "[bn:gpu:debug] ssm_types layer=%d qkv=%d z=%d alpha=%d beta=%d out=%d\n",
            layer, lw->ssm.wqkv.type, lw->ssm.wz.type,
            lw->ssm.ssm_alpha.type, lw->ssm.ssm_beta.type,
            lw->ssm.ssm_out.type);
    size_t state_values = (size_t)n_ssm * shape.num_v_heads *
                          shape.head_k_dim * shape.head_v_dim;
    size_t conv_values = (size_t)n_ssm * (shape.conv_kernel - 1) *
                         shape.qkv_dim;
    size_t state_bytes = state_values * sizeof(float);
    size_t conv_bytes = conv_values * sizeof(float);
    float *pre_x = malloc((size_t)dim * sizeof(float));
    float *pre_xb = malloc((size_t)dim * sizeof(float));
    float *cpu_norm = malloc((size_t)dim * sizeof(float));
    float *cpu_z_direct = malloc((size_t)shape.value_dim * sizeof(float));
    float *pre_state = malloc(state_bytes);
    float *pre_conv = malloc(conv_bytes);
    float *gpu_proj = malloc((size_t)dim * sizeof(float));
    float *gpu_gate = malloc((size_t)dim * sizeof(float));
    float *gpu_qkv = malloc((size_t)shape.qkv_dim * sizeof(float));
    float *gpu_projection_input = malloc((size_t)dim * sizeof(float));
    float *gpu_qkv_raw = malloc((size_t)shape.qkv_dim * sizeof(float));
    float *cpu_qkv_raw = malloc((size_t)shape.qkv_dim * sizeof(float));
    float *gpu_z = malloc((size_t)shape.value_dim * sizeof(float));
    float *cpu_alpha = malloc((size_t)shape.num_v_heads * sizeof(float));
    float *cpu_beta = malloc((size_t)shape.num_v_heads * sizeof(float));
    float *gpu_alpha = malloc((size_t)shape.num_v_heads * sizeof(float));
    float *gpu_beta = malloc((size_t)shape.num_v_heads * sizeof(float));
    float *gpu_state = malloc(state_bytes);
    float *gpu_conv = malloc(conv_bytes);
    size_t state_per = (size_t)shape.num_v_heads * shape.head_k_dim *
                       shape.head_v_dim;
    float *same_input_state = malloc(state_per * sizeof(float));
    float *same_input_out = malloc((size_t)shape.value_dim * sizeof(float));
    size_t conv_per = (size_t)(shape.conv_kernel - 1) * shape.qkv_dim;
    float *same_input_conv = malloc(conv_per * sizeof(float));
    BnRunState *s = &sess->state;
    if (!pre_x || !pre_xb || !cpu_norm || !cpu_z_direct ||
        !pre_state || !pre_conv ||
        !gpu_proj || !gpu_gate ||
        !gpu_qkv || !gpu_projection_input || !gpu_qkv_raw || !cpu_qkv_raw ||
        !gpu_z || !cpu_alpha || !cpu_beta ||
        !gpu_alpha || !gpu_beta || !gpu_state || !gpu_conv ||
        !same_input_state || !same_input_out || !same_input_conv) {
        free(pre_x); free(pre_xb); free(cpu_norm); free(cpu_z_direct);
        free(pre_state); free(pre_conv); free(gpu_proj);
        free(gpu_gate);
        free(gpu_qkv); free(gpu_projection_input); free(gpu_qkv_raw);
        free(cpu_qkv_raw); free(gpu_z);
        free(cpu_alpha); free(cpu_beta); free(gpu_alpha); free(gpu_beta);
        free(gpu_state); free(gpu_conv);
        free(same_input_state); free(same_input_out);
        free(same_input_conv);
        return -1;
    }
    memcpy(pre_x, s->x, (size_t)dim * sizeof(float));
    memcpy(pre_xb, s->xb, (size_t)dim * sizeof(float));
    memcpy(pre_state, s->ssm_state, state_bytes);
    memcpy(pre_conv, s->ssm_conv_state, conv_bytes);
    int rc = bn_transformer_gpu_emit_context_flush(emit, gpu);
    rc = rc == 0 ? bn_transformer_gpu_read_activation_buf(
        gpu, BN_GPU_VALUE_SCRATCH, gpu_proj,
        (size_t)dim * sizeof(float)) : rc;
    rc = rc == 0 ? bn_transformer_gpu_read_xb2(
        gpu, gpu_gate, (size_t)dim * sizeof(float)) : rc;
    rc = rc == 0 ? bn_transformer_gpu_read_activation_buf(
        gpu, BN_GPU_VALUE_SSM_QKV, gpu_qkv,
        (size_t)shape.qkv_dim * sizeof(float)) : rc;
    rc = rc == 0 ? bn_transformer_gpu_read_activation_buf(
        gpu, BN_GPU_VALUE_Q, gpu_projection_input,
        (size_t)dim * sizeof(float)) : rc;
    rc = rc == 0 ? bn_transformer_gpu_read_activation_buf(
        gpu, BN_GPU_VALUE_QKV, gpu_qkv_raw,
        (size_t)shape.qkv_dim * sizeof(float)) : rc;
    rc = rc == 0 ? bn_transformer_gpu_read_activation_buf(
        gpu, BN_GPU_VALUE_SSM_Z, gpu_z,
        (size_t)shape.value_dim * sizeof(float)) : rc;
    rc = rc == 0 ? bn_transformer_gpu_read_activation_buf(
        gpu, BN_GPU_VALUE_SSM_ALPHA, gpu_alpha,
        (size_t)shape.num_v_heads * sizeof(float)) : rc;
    rc = rc == 0 ? bn_transformer_gpu_read_activation_buf(
        gpu, BN_GPU_VALUE_SSM_BETA, gpu_beta,
        (size_t)shape.num_v_heads * sizeof(float)) : rc;
    rc = rc == 0 ? bn_transformer_gpu_read_activation_buf(
        gpu, BN_GPU_VALUE_SSM_STATE, gpu_state, state_bytes) : rc;
    rc = rc == 0 ? bn_transformer_gpu_read_activation_buf(
        gpu, BN_GPU_VALUE_SSM_CONV_STATE, gpu_conv, conv_bytes) : rc;
    if (rc == 0) {
        fallback_rmsnorm(cpu_norm, pre_x, lw->norm.attn_norm, dim,
                         bn_transformer_gpu_norm_epsilon(&m->config));
        bn_transformer_gpu_debug_compare_vec(
            "ssm_input_norm_compare", layer, pos,
            cpu_norm, pre_xb, dim);
        debug_quant_matvec_prepared(
            m, cpu_qkv_raw, &lw->ssm.wqkv, gpu_projection_input,
            s->x_q, 0);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_qkv_raw_same_input_compare", layer, pos,
            cpu_qkv_raw, gpu_qkv_raw, shape.qkv_dim);
        debug_quant_matvec_prepared(
            m, cpu_qkv_raw, &lw->ssm.wqkv, cpu_norm, s->x_q, 0);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_qkv_raw_cpu_norm_compare", layer, pos,
            cpu_qkv_raw, gpu_qkv_raw, shape.qkv_dim);
        BnMatvecTask same_input_qz_tasks[2] = {
            { cpu_qkv_raw, &lw->ssm.wqkv, NULL, 0 },
            { cpu_z_direct, &lw->ssm.wz, NULL, 0 },
        };
        debug_quant_matvec_batch_prepared(
            m, same_input_qz_tasks, 2, gpu_projection_input, s->x_q, 0);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_qkv_raw_same_input_batch_compare", layer, pos,
            cpu_qkv_raw, gpu_qkv_raw, shape.qkv_dim);
        int same_input_ssm_idx = bn_transformer_ssm_index(&m->config, layer);
        size_t gpu_conv_current =
            (size_t)same_input_ssm_idx * conv_per +
            (size_t)(shape.conv_kernel - 2) * shape.qkv_dim;
        bn_transformer_gpu_debug_compare_vec(
            "ssm_conv_state_raw_capture_compare", layer, pos,
            gpu_qkv_raw, gpu_conv + gpu_conv_current, shape.qkv_dim);
        memcpy(cpu_qkv_raw, gpu_qkv_raw,
               (size_t)shape.qkv_dim * sizeof(float));
        memcpy(same_input_conv,
               pre_conv + (size_t)same_input_ssm_idx * conv_per,
               conv_per * sizeof(float));
        BnSSMConvCtx same_input_conv_ctx = {
            cpu_qkv_raw, same_input_conv, lw->ssm.ssm_conv1d,
            shape.qkv_dim, shape.conv_kernel
        };
        bn_transformer_cpu_ssm_conv_silu_op(
            bn_transformer_cpu_backend_ops(fallback_cpu_runtime(m)))(
                &same_input_conv_ctx, 0, shape.qkv_dim);
        BnSSML2NormCtx same_input_l2_ctx = {
            cpu_qkv_raw, cpu_qkv_raw + shape.key_dim,
            bn_transformer_gpu_norm_epsilon(&m->config), shape.head_k_dim
        };
        bn_transformer_cpu_ssm_l2norm_op(
            bn_transformer_cpu_backend_ops(fallback_cpu_runtime(m)))(
                &same_input_l2_ctx, 0, shape.num_k_heads);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_qkv_same_input_conv_l2_compare", layer, pos,
            cpu_qkv_raw, gpu_qkv, shape.qkv_dim);
        if (res && res->wqkv && gpu->matvec &&
            gpu->matvec(gpu->ctx, cpu_qkv_raw, res->wqkv,
                        gpu_projection_input, lw->ssm.wqkv.rows,
                        lw->ssm.wqkv.cols, lw->ssm.wqkv.type) == 0) {
            bn_transformer_gpu_debug_compare_vec(
                "ssm_qkv_graph_standalone_compare", layer, pos,
                cpu_qkv_raw, gpu_qkv_raw, shape.qkv_dim);
            debug_compare_native_quant_activation(
                gpu, layer, pos, gpu_projection_input, dim);
        }
        debug_quant_matvec_prepared(
            m, cpu_z_direct, &lw->ssm.wz, pre_xb, s->x_q, 0);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_z_direct_compare", layer, pos,
            cpu_z_direct, gpu_z, lw->ssm.wz.rows);
        debug_quant_matvec_prepared(
            m, cpu_alpha, &lw->ssm.ssm_alpha, pre_xb, s->x_q, 0);
        debug_quant_matvec_prepared(
            m, cpu_beta, &lw->ssm.ssm_beta, pre_xb, s->x_q, 0);
        for (int h = 0; h < shape.num_v_heads; h++) {
            float dt = cpu_alpha[h] + lw->ssm.ssm_dt_bias[h];
            float dt_sp = dt > 20.0f ? dt : logf(1.0f + expf(dt));
            cpu_alpha[h] = expf(dt_sp * lw->ssm.ssm_a[h]);
            cpu_beta[h] = 1.0f / (1.0f + expf(-cpu_beta[h]));
        }
        bn_transformer_gpu_debug_compare_vec(
            "ssm_alpha_compare", layer, pos, cpu_alpha, gpu_alpha,
            shape.num_v_heads);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_beta_compare", layer, pos, cpu_beta, gpu_beta,
            shape.num_v_heads);
        debug_quant_matvec_prepared(
            m, cpu_qkv_raw, &lw->ssm.ssm_out, gpu_gate, s->x_q, 0);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_out_same_input_compare", layer, pos,
            cpu_qkv_raw, gpu_proj, dim);
        memcpy(s->x, pre_x, (size_t)dim * sizeof(float));
        memcpy(s->ssm_state, pre_state, state_bytes);
        memcpy(s->ssm_conv_state, pre_conv, conv_bytes);
        bn_transformer_cpu_forward_ssm_block(m, sess, lw, layer, pos);
        int ssm_idx = bn_transformer_ssm_index(&m->config, layer);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_projection_compare", layer, pos, s->xb, gpu_proj, dim);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_gate_compare", layer, pos, s->xb2, gpu_gate, dim);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_qkv_compare", layer, pos, s->hb, gpu_qkv, shape.qkv_dim);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_z_compare", layer, pos, s->hb2, gpu_z, dim);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_state_compare", layer, pos,
            s->ssm_state + (size_t)ssm_idx * state_per,
            gpu_state + (size_t)ssm_idx * state_per, (int)state_per);
        memcpy(same_input_state,
               pre_state + (size_t)ssm_idx * state_per,
               state_per * sizeof(float));
        BnSSMDeltaCtx same_input_delta = {
            same_input_state, same_input_out,
            gpu_qkv, gpu_qkv + shape.key_dim,
            gpu_qkv + 2 * shape.key_dim,
            gpu_alpha, gpu_beta,
            shape.num_k_heads, shape.head_k_dim,
            shape.head_v_dim,
            1.0f / sqrtf((float)shape.head_k_dim)
        };
        bn_tp_fn same_input_delta_op = bn_transformer_cpu_ssm_delta_op(
            bn_transformer_cpu_backend_ops(fallback_cpu_runtime(m)));
        same_input_delta_op(&same_input_delta, 0, shape.num_v_heads);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_state_same_input_compare", layer, pos,
            same_input_state,
            gpu_state + (size_t)ssm_idx * state_per, (int)state_per);
        BnSSMGateCtx same_input_gate = {
            same_input_out, gpu_z, lw->ssm.ssm_norm,
            bn_transformer_gpu_norm_epsilon(&m->config),
            shape.head_v_dim
        };
        bn_tp_fn same_input_gate_op = bn_transformer_cpu_ssm_gate_op(
            bn_transformer_cpu_backend_ops(fallback_cpu_runtime(m)));
        same_input_gate_op(&same_input_gate, 0, shape.num_v_heads);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_gate_same_input_compare", layer, pos,
            same_input_out, gpu_gate, shape.value_dim);
        bn_transformer_gpu_debug_compare_vec(
            "ssm_conv_state_compare", layer, pos,
            s->ssm_conv_state + (size_t)ssm_idx * conv_per,
            gpu_conv + (size_t)ssm_idx * conv_per, (int)conv_per);
        memcpy(s->ssm_state, gpu_state, state_bytes);
        memcpy(s->ssm_conv_state, gpu_conv, conv_bytes);
        (void)bn_transformer_gpu_read_x(
            gpu, s->x, (size_t)dim * sizeof(float));
        (void)bn_transformer_gpu_read_xb(
            gpu, s->xb, (size_t)dim * sizeof(float));
    }
    free(pre_x); free(pre_xb); free(cpu_norm); free(cpu_z_direct);
    free(pre_state); free(pre_conv); free(gpu_proj);
    free(gpu_gate);
    free(gpu_qkv); free(gpu_projection_input); free(gpu_qkv_raw);
    free(cpu_qkv_raw); free(gpu_z);
    free(cpu_alpha); free(cpu_beta); free(gpu_alpha); free(gpu_beta);
    free(gpu_state); free(gpu_conv);
    free(same_input_state); free(same_input_out);
    free(same_input_conv);
    return rc;
}

int bn_transformer_gpu_debug_compare_per_layer_state(
    BnTransformerGPUEmitContext *emit,
    const BnGPUBackend *gpu,
    BnModel *m,
    BnSession *sess,
    BnLayerWeights *lw,
    int layer,
    int pos,
    int dim) {
    BnRunState *s = &sess->state;
    int per_dim = bn_transformer_per_layer_embedding_dim(&m->config);
    if (per_dim <= 0 || !s->per_layer_input ||
        !lw->per_layer.inp_gate.data || !lw->per_layer.proj.data ||
        !lw->per_layer.post_norm)
        return -1;
    float *cpu_x = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_x = (float *)malloc((size_t)dim * sizeof(float));
    float *gate = (float *)malloc((size_t)per_dim * sizeof(float));
    if (!cpu_x || !gpu_x || !gate) {
        free(cpu_x); free(gpu_x); free(gate);
        return -1;
    }
    memcpy(cpu_x, s->x, (size_t)dim * sizeof(float));
    fallback_cpu_matvec(m, gate, &lw->per_layer.inp_gate, cpu_x, s->x_q);
    for (int i = 0; i < per_dim; i++) {
        float g = gate[i];
        gate[i] = fallback_reference_gelu(g) *
                  s->per_layer_input[(size_t)layer * per_dim + i];
    }
    fallback_cpu_matvec(m, s->xb2, &lw->per_layer.proj, gate, s->x_q);
    fallback_rmsnorm(s->xb2, s->xb2, lw->per_layer.post_norm, dim,
                     bn_transformer_gpu_norm_epsilon(&m->config));
    for (int i = 0; i < dim; i++)
        cpu_x[i] += s->xb2[i];
    if (lw->norm.layer_output_scale) {
        float scale = lw->norm.layer_output_scale[0];
        for (int i = 0; i < dim; i++)
            cpu_x[i] *= scale;
    }
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_x(gpu, gpu_x,
                                  (size_t)dim * sizeof(float)) != 0) {
        free(cpu_x); free(gpu_x); free(gate);
        return -1;
    }
    bn_transformer_gpu_debug_compare_vec(
        "per_layer_state_compare", layer, pos, cpu_x, gpu_x, dim);
    memcpy(s->x, gpu_x, (size_t)dim * sizeof(float));
    free(cpu_x); free(gpu_x); free(gate);
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
    int dim,
    int reference_uses_float_kquant) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnLayerShapePlan shape;
    bn_transformer_plan_layer_shape(&shape, c, lw, layer,
                                    bn_model_tq_state(m) != NULL);
    if (!shape.is_attn ||
        !bn_transformer_kv_host_float_cache_rows_available(c)) {
        fprintf(stderr,
                "[bn:gpu:debug] attention_compare_unavailable layer=%d "
                "is_attn=%d host_float_kv=%d\n",
                layer, shape.is_attn,
                bn_transformer_kv_host_float_cache_rows_available(c));
        return -1;
    }

    float *cpu_in = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_x = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_q = (float *)malloc((size_t)shape.q_dim * sizeof(float));
    float *gpu_k = (float *)malloc((size_t)shape.kv_dim * sizeof(float));
    if (!cpu_in || !gpu_x || !gpu_q || !gpu_k) {
        fprintf(stderr, "[bn:gpu:debug] attention_compare_alloc_failed\n");
        free(cpu_in);
        free(gpu_x);
        free(gpu_q);
        free(gpu_k);
        return -1;
    }
    memcpy(cpu_in, s->x, (size_t)dim * sizeof(float));
    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0) {
        fprintf(stderr, "[bn:gpu:debug] attention_compare_flush_failed\n");
        free(cpu_in);
        free(gpu_x);
        return -1;
    }
    if (bn_transformer_gpu_read_x(gpu, gpu_x,
                                  (size_t)dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf(
            gpu, BN_GPU_VALUE_Q, gpu_q,
            (size_t)shape.q_dim * sizeof(float)) != 0) {
        fprintf(stderr,
                "[bn:gpu:debug] attention_compare_state_read_failed\n");
        free(cpu_in);
        free(gpu_x);
        free(gpu_q);
        free(gpu_k);
        return -1;
    }

    int head_size = shape.head_size;
    int n_heads = shape.n_heads;
    int kv_dim = shape.kv_dim;
    int n_kv_heads = shape.n_kv_heads;
    int kv_mul = shape.kv_mul;
    int layer_rope_dims = rope_dims > head_size ? head_size : rope_dims;
    int n_kv = (pos + 1 < c->seq_len) ? pos + 1 : c->seq_len;
    int kv_read_idx = bn_transformer_attention_kv_read_index(c, lw, layer);
    size_t loff = (size_t)kv_read_idx * c->seq_len * c->kv_dim;
    size_t kv_bytes = (size_t)n_kv * c->kv_dim * sizeof(float);
    size_t kv_off = loff * sizeof(float);

    if (bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, s->key_cache + loff, kv_bytes,
            kv_off) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, s->value_cache + loff, kv_bytes,
            kv_off) != 0) {
        fprintf(stderr,
                "[bn:gpu:debug] attention_compare_cache_read_failed\n");
        free(cpu_in);
        free(gpu_x);
        free(gpu_q);
        free(gpu_k);
        return -1;
    }

    memcpy(s->x, cpu_in, (size_t)dim * sizeof(float));
    float *key_cache_row =
        s->key_cache + loff + (size_t)cache_pos * c->kv_dim;
    float *value_cache_row =
        s->value_cache + loff + (size_t)cache_pos * c->kv_dim;

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim,
                     bn_transformer_gpu_norm_epsilon(c));
    float *q_full = shape.q_gated ? s->hb : s->q;
    {
        BnMatvecTask qkv[3] = {
            { q_full, &lw->attn.wq, NULL, 0 },
            { key_cache_row, &lw->attn.wk, NULL, 0 },
            { value_cache_row, &lw->attn.wv, NULL, 0 },
        };
        debug_quant_matvec_batch_prepared(
            m, qkv, lw->attn.has_kv ? 3 : 1, s->xb, s->x_q,
            reference_uses_float_kquant);
    }

    if (shape.q_gated) {
        for (int h = 0; h < n_heads; h++)
            memcpy(s->q + (size_t)h * head_size,
                   q_full + (size_t)h * 2 * head_size,
                   (size_t)head_size * sizeof(float));
    }
    if (lw->attn.q_bias) {
        for (int i = 0; i < shape.q_dim; i++) s->q[i] += lw->attn.q_bias[i];
    }
    if (lw->attn.has_kv && lw->attn.k_bias) {
        for (int i = 0; i < kv_dim; i++) key_cache_row[i] += lw->attn.k_bias[i];
    }
    if (lw->attn.has_kv && lw->attn.v_bias) {
        for (int i = 0; i < kv_dim; i++)
            value_cache_row[i] += lw->attn.v_bias[i];
    }
    if (lw->attn.has_kv && shape.value_shares_key) {
        float eps = bn_transformer_gpu_norm_epsilon(c);
        for (int h = 0; h < n_kv_heads; h++) {
            float *vh = value_cache_row + (size_t)h * head_size;
            float ss = 0.0f;
            for (int i = 0; i < head_size; i++)
                ss += vh[i] * vh[i];
            float scale = 1.0f / sqrtf(ss / (float)head_size + eps);
            for (int i = 0; i < head_size; i++)
                vh[i] *= scale;
        }
    }
    if (lw->attn.q_norm) {
        for (int h = 0; h < n_heads; h++)
            fallback_rmsnorm(s->q + (size_t)h * head_size,
                             s->q + (size_t)h * head_size,
                             lw->attn.q_norm + (size_t)h * shape.qk_stride,
                             head_size, bn_transformer_gpu_norm_epsilon(c));
    }
    if (lw->attn.has_kv && lw->attn.k_norm) {
        for (int h = 0; h < n_kv_heads; h++)
            fallback_rmsnorm(key_cache_row + (size_t)h * head_size,
                             key_cache_row + (size_t)h * head_size,
                             lw->attn.k_norm + (size_t)h * shape.qk_stride,
                             head_size, bn_transformer_gpu_norm_epsilon(c));
    }

    bn_transformer_cpu_apply_rope_heads(fallback_cpu_runtime(m), s->q,
                                        n_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);
    if (lw->attn.has_kv)
        bn_transformer_cpu_apply_rope_heads(
            fallback_cpu_runtime(m), key_cache_row, n_kv_heads, head_size,
            layer_rope_dims, rope_cos, rope_sin);
    if (lw->attn.has_kv && bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, gpu_k,
            (size_t)kv_dim * sizeof(float),
            (loff + (size_t)cache_pos * c->kv_dim) * sizeof(float)) != 0) {
        fprintf(stderr,
                "[bn:gpu:debug] attention_compare_key_read_failed\n");
        free(cpu_in);
        free(gpu_x);
        free(gpu_q);
        free(gpu_k);
        return -1;
    }
    bn_transformer_gpu_debug_compare_vec(
        "attention_rope_q_compare", layer, pos, s->q, gpu_q, shape.q_dim);
    if (lw->attn.has_kv)
        bn_transformer_gpu_debug_compare_vec(
            "attention_rope_k_compare", layer, pos,
            key_cache_row, gpu_k, kv_dim);

    BnGQACtx gctx = {
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len,
        bn_transformer_attention_scale(c, head_size),
        bn_transformer_kv_host_cache_uses_fp16_rows(c)
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);

    if (shape.q_gated) {
        const BnCPUBackendOps *cpu_ops = bn_transformer_cpu_backend_ops(
            fallback_cpu_runtime(m));
        for (int h = 0; h < n_heads; h++)
            cpu_ops->apply_sigmoid_gate(
                s->xb + (size_t)h * head_size,
                q_full + (size_t)h * 2 * head_size + head_size,
                head_size);
    }
    if (lw->norm.attn_sub_norm)
        fallback_rmsnorm(s->xb, s->xb, lw->norm.attn_sub_norm,
                         shape.q_dim, bn_transformer_gpu_norm_epsilon(c));
    if (bn_transformer_cpu_weight_uses_native_quant_activation(
            &lw->attn.wo))
        debug_compare_native_block_activation(
            gpu, layer, pos, s->xb, shape.q_dim);
    {
        BnMatvecTask wo[1] = {{ s->xb2, &lw->attn.wo, NULL, 0 }};
        debug_quant_matvec_batch_prepared(
            m, wo, 1, s->xb, s->x_q, reference_uses_float_kquant);
    }
    if (lw->norm.attn_post_norm)
        fallback_rmsnorm(s->xb2, s->xb2, lw->norm.attn_post_norm,
                         dim, bn_transformer_gpu_norm_epsilon(c));
    bn_transformer_cpu_residual_add(
        fallback_cpu_runtime(m), s->x, s->xb2, dim);

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

    if (lw->norm.ffn_norm) {
        float *cpu_next = (float *)malloc((size_t)dim * sizeof(float));
        float *gpu_next = (float *)malloc((size_t)dim * sizeof(float));
        if (cpu_next && gpu_next &&
            bn_transformer_gpu_read_xb(
                gpu, gpu_next, (size_t)dim * sizeof(float)) == 0) {
            fallback_rmsnorm(cpu_next, s->x, lw->norm.ffn_norm, dim,
                             bn_transformer_gpu_norm_epsilon(c));
            bn_transformer_gpu_debug_compare_vec(
                "attention_next_norm_compare", layer, pos,
                cpu_next, gpu_next, dim);
        }
        free(cpu_next);
        free(gpu_next);
    }

    memcpy(s->x, gpu_x, (size_t)dim * sizeof(float));
    if (bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, key_cache_row,
            (size_t)kv_dim * sizeof(float),
            (loff + (size_t)cache_pos * c->kv_dim) * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, value_cache_row,
            (size_t)kv_dim * sizeof(float),
            (loff + (size_t)cache_pos * c->kv_dim) * sizeof(float)) != 0) {
        fprintf(stderr,
                "[bn:gpu:debug] attention_compare_restore_failed\n");
        free(cpu_in);
        free(gpu_x);
        free(gpu_q);
        free(gpu_k);
        return -1;
    }

    free(cpu_in);
    free(gpu_x);
    free(gpu_q);
    free(gpu_k);
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
    int dim,
    int reference_uses_float_kquant) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnLayerShapePlan shape;
    bn_transformer_plan_layer_shape(&shape, c, lw, layer,
                                    bn_model_tq_state(m) != NULL);
    if (!shape.is_attn ||
        !bn_transformer_kv_host_float_cache_rows_available(c))
        return -1;

    float *cpu_in = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_xb = (float *)malloc((size_t)shape.q_dim * sizeof(float));
    float *gpu_value_row =
        (float *)malloc((size_t)shape.kv_dim * sizeof(float));
    if (!cpu_in || !gpu_xb || !gpu_value_row) {
        free(cpu_in);
        free(gpu_xb);
        free(gpu_value_row);
        return -1;
    }
    memcpy(cpu_in, s->x, (size_t)dim * sizeof(float));

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_xb(gpu, gpu_xb,
                                   (size_t)shape.q_dim * sizeof(float)) != 0) {
        free(cpu_in);
        free(gpu_xb);
        free(gpu_value_row);
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
        free(gpu_value_row);
        return -1;
    }

    memcpy(s->x, cpu_in, (size_t)dim * sizeof(float));
    float *key_cache_row =
        s->key_cache + loff + (size_t)cache_pos * c->kv_dim;
    float *value_cache_row =
        s->value_cache + loff + (size_t)cache_pos * c->kv_dim;
    memcpy(gpu_value_row, value_cache_row,
           (size_t)kv_dim * sizeof(float));

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim,
                     bn_transformer_gpu_norm_epsilon(c));
    float *q_full = shape.q_gated ? s->hb : s->q;
    {
        BnMatvecTask qkv[3] = {
            { q_full, &lw->attn.wq, NULL, 0 },
            { key_cache_row, &lw->attn.wk, NULL, 0 },
            { value_cache_row, &lw->attn.wv, NULL, 0 },
        };
        debug_quant_matvec_batch_prepared(
            m, qkv, lw->attn.has_kv ? 3 : 1, s->xb, s->x_q,
            reference_uses_float_kquant);
    }

    if (shape.q_gated) {
        for (int h = 0; h < n_heads; h++)
            memcpy(s->q + (size_t)h * head_size,
                   q_full + (size_t)h * 2 * head_size,
                   (size_t)head_size * sizeof(float));
    }

    if (lw->attn.q_bias) {
        for (int i = 0; i < shape.q_dim; i++)
            s->q[i] += lw->attn.q_bias[i];
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
    if (shape.value_shares_key) {
        float eps = bn_transformer_gpu_norm_epsilon(c);
        for (int h = 0; h < n_kv_heads; h++) {
            float *vh = value_cache_row + (size_t)h * head_size;
            float ss = 0.0f;
            for (int i = 0; i < head_size; i++)
                ss += vh[i] * vh[i];
            float scale = 1.0f / sqrtf(ss / (float)head_size + eps);
            for (int i = 0; i < head_size; i++)
                vh[i] *= scale;
        }
    }

    bn_transformer_gpu_debug_compare_vec(
        "gqa_value_cache_compare", layer, pos,
        value_cache_row, gpu_value_row, kv_dim);
    if (n_kv == 1) {
        float *expanded_value =
            (float *)malloc((size_t)shape.q_dim * sizeof(float));
        if (expanded_value) {
            for (int h = 0; h < n_heads; h++) {
                int kv_h = h / kv_mul;
                memcpy(expanded_value + (size_t)h * head_size,
                       gpu_value_row + (size_t)kv_h * head_size,
                       (size_t)head_size * sizeof(float));
            }
            bn_transformer_gpu_debug_compare_vec(
                "gqa_single_value_compare", layer, pos,
                expanded_value, gpu_xb, shape.q_dim);
            free(expanded_value);
        }
    }

    bn_transformer_cpu_apply_rope_heads(fallback_cpu_runtime(m), s->q,
                                        n_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);
    bn_transformer_cpu_apply_rope_heads(fallback_cpu_runtime(m), key_cache_row,
                                        n_kv_heads, head_size,
                                        layer_rope_dims, rope_cos, rope_sin);

    BnGQACtx gctx = {
        c, s, loff, pos, n_kv, kv_mul, head_size, c->kv_dim, c->seq_len,
        bn_transformer_attention_scale(c, head_size),
        bn_transformer_kv_host_cache_uses_fp16_rows(c)
    };
    bn_transformer_cpu_gqa_dispatch(m, &gctx, n_heads, kv_mul);

    bn_transformer_gpu_debug_compare_vec(
        "gqa_compare", layer, pos, s->xb, gpu_xb, shape.q_dim);

    free(cpu_in);
    free(gpu_xb);
    free(gpu_value_row);
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
    int kv_dim,
    int gpu_k_rope_applied,
    int reference_uses_float_kquant) {
    BnConfig *c = &m->config;
    BnRunState *s = &sess->state;
    BnLayerShapePlan shape;
    bn_transformer_plan_layer_shape(&shape, c, lw, layer,
                                    bn_model_tq_state(m) != NULL);
    if (!shape.is_attn || shape.head_size <= 0 ||
        q_dim % shape.head_size != 0)
        return -1;
    fprintf(stderr,
            "[bn:gpu:debug] qkv_types layer=%d q=%d k=%d v=%d "
            "reference_float_kquant=%d\n",
            layer, lw->attn.wq.type, lw->attn.wk.type, lw->attn.wv.type,
            reference_uses_float_kquant);
    int head_size = shape.head_size;
    int n_kv_heads = shape.n_kv_heads;
    int qk_stride = shape.qk_stride;
    int cpu_q_rows = shape.q_gated ? 2 * q_dim : q_dim;
    float *cpu_q_storage =
        (float *)malloc((size_t)cpu_q_rows * sizeof(float));
    float *cpu_q = shape.q_gated
        ? (float *)malloc((size_t)q_dim * sizeof(float))
        : cpu_q_storage;
    float *cpu_k = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *cpu_v = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *gpu_xb = (float *)malloc((size_t)dim * sizeof(float));
    float *gpu_q = (float *)malloc((size_t)q_dim * sizeof(float));
    float *gpu_q_storage = shape.q_gated
        ? (float *)malloc((size_t)cpu_q_rows * sizeof(float)) : NULL;
    float *gpu_k = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *gpu_v = (float *)malloc((size_t)kv_dim * sizeof(float));
    if (!cpu_q_storage || !cpu_q || !cpu_k || !cpu_v || !gpu_xb ||
        !gpu_q || (shape.q_gated && !gpu_q_storage) || !gpu_k || !gpu_v) {
        if (cpu_q != cpu_q_storage) free(cpu_q);
        free(cpu_q_storage); free(cpu_k); free(cpu_v);
        free(gpu_xb); free(gpu_q); free(gpu_q_storage); free(gpu_k); free(gpu_v);
        return -1;
    }

    if (bn_transformer_gpu_emit_context_flush(emit, gpu) != 0 ||
        bn_transformer_gpu_read_x(gpu, s->x,
                                  (size_t)dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_xb(gpu, gpu_xb,
                                   (size_t)dim * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf(gpu, BN_GPU_VALUE_Q, gpu_q,
                                               (size_t)q_dim * sizeof(float)) != 0 ||
        (shape.q_gated &&
         bn_transformer_gpu_read_activation_buf(
             gpu, BN_GPU_VALUE_QKV, gpu_q_storage,
             (size_t)cpu_q_rows * sizeof(float)) != 0) ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_KEY_CACHE, gpu_k,
            (size_t)kv_dim * sizeof(float),
            (size_t)kv_cache_off * sizeof(float)) != 0 ||
        bn_transformer_gpu_read_activation_buf_offset(
            gpu, BN_GPU_VALUE_VALUE_CACHE, gpu_v,
            (size_t)kv_dim * sizeof(float),
            (size_t)kv_cache_off * sizeof(float)) != 0) {
        if (cpu_q != cpu_q_storage) free(cpu_q);
        free(cpu_q_storage); free(cpu_k); free(cpu_v);
        free(gpu_xb); free(gpu_q); free(gpu_q_storage); free(gpu_k); free(gpu_v);
        return -1;
    }

    fallback_rmsnorm(s->xb, s->x, lw->norm.attn_norm, dim,
                     bn_transformer_gpu_norm_epsilon(&m->config));
    bn_transformer_gpu_debug_compare_vec(
        "attn_norm_compare", layer, pos, s->xb, gpu_xb, dim);
    if (bn_transformer_cpu_weight_uses_native_quant_activation(
            &lw->attn.wq) && !reference_uses_float_kquant)
        debug_compare_native_block_activation(gpu, layer, pos, s->xb, dim);
    else if (!bn_transformer_cpu_weight_uses_native_quant_activation(
                 &lw->attn.wq))
        debug_compare_native_quant_activation(gpu, layer, pos, s->xb, dim);
    debug_quant_matvec_prepared(
        m, cpu_q_storage, &lw->attn.wq, gpu_xb, s->x_q,
        reference_uses_float_kquant);
    if (shape.q_gated) {
        bn_transformer_gpu_debug_compare_vec(
            "qkv_q_gate_raw_compare", layer, pos,
            cpu_q_storage, gpu_q_storage, cpu_q_rows);
        int n_heads = q_dim / head_size;
        for (int h = 0; h < n_heads; h++)
            memcpy(cpu_q + (size_t)h * head_size,
                   cpu_q_storage + (size_t)h * 2 * head_size,
                   (size_t)head_size * sizeof(float));
    }
    debug_quant_matvec_prepared(m, cpu_k, &lw->attn.wk, gpu_xb, s->x_q,
                                reference_uses_float_kquant);
    debug_quant_matvec_prepared(m, cpu_v, &lw->attn.wv, gpu_xb, s->x_q,
                                reference_uses_float_kquant);
    if (lw->attn.q_bias) {
        for (int i = 0; i < q_dim; i++) cpu_q[i] += lw->attn.q_bias[i];
    }
    if (gpu_k_rope_applied) {
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
    if (shape.value_shares_key) {
        float eps = bn_transformer_gpu_norm_epsilon(c);
        for (int h = 0; h < n_kv_heads; h++) {
            float *vh = cpu_v + (size_t)h * head_size;
            float ss = 0.0f;
            for (int i = 0; i < head_size; i++)
                ss += vh[i] * vh[i];
            float scale = 1.0f / sqrtf(ss / (float)head_size + eps);
            for (int i = 0; i < head_size; i++)
                vh[i] *= scale;
        }
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

    if (cpu_q != cpu_q_storage) free(cpu_q);
    free(cpu_q_storage); free(cpu_k); free(cpu_v);
    free(gpu_xb); free(gpu_q); free(gpu_q_storage); free(gpu_k); free(gpu_v);
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
        bn_transformer_gpu_report_fallback(
            gpu, "gpu logits cpu fallback flush failed");
        return -1;
    }
    double t_flush = bn_platform_time_ms();
    if (bn_transformer_gpu_read_xb(gpu, s->xb,
                                   (size_t)dim * sizeof(float)) != 0) {
        bn_transformer_gpu_report_fallback(
            gpu, "gpu logits cpu fallback read_xb failed");
        return -1;
    }
    double t_read = bn_platform_time_ms();
    fallback_cpu_matvec(m, s->logits, logits->cpu_weight, s->xb, s->x_q);
    double t_logits = bn_platform_time_ms();
    if (bn_transformer_gpu_profile_level(gpu) >= 3) {
        fprintf(stderr,
                "[gpu:fallback:logits] flush=%.3fms read=%.3fms cpu=%.3fms total=%.3fms\n",
                t_flush - t0, t_read - t_flush, t_logits - t_read,
                t_logits - t0);
    }
    return 0;
}
