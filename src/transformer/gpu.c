#include "gpu_internal.h"
#include "backend_session.h"
#include "session.h"
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
        bn_quant_matvec(hb, &wgate, xb_in, sess->state.x_q, bn_model_pool(m));
        bn_quant_matvec(hb2, &wup, xb_in, sess->state.x_q, bn_model_pool(m));
        bn_moe_swiglu(hb, hb, hb2, moe_hidden);
        bn_quant_matvec(down, &wdown, hb, sess->state.x_q, bn_model_pool(m));
        bn_moe_weighted_add(expert_out, down, weight, dim);
    }

    if (c->has_shared_expert && lw->shared.shared_gate.data) {
        int shared_hidden = c->shared_expert_intermediate_size;
        bn_quant_matvec(hb, &lw->shared.shared_gate, xb_in,
                        sess->state.x_q, bn_model_pool(m));
        bn_quant_matvec(hb2, &lw->shared.shared_up, xb_in,
                        sess->state.x_q, bn_model_pool(m));
        bn_moe_swiglu(hb, hb, hb2, shared_hidden);
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

static float gpu_exact_q6k_row_dot(const BnQWeight *W, int row,
                                   const float *x) {
    int cols = W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)W->data;
    float row_sum = 0.0f;

    for (int b = 0; b < n_blocks_per_row; b++) {
        const BnBlockQ6K *blk =
            &blocks[(size_t)row * n_blocks_per_row + b];
        float d = bn_fp16_to_fp32(blk->d);
        const uint8_t *ql = blk->ql;
        const uint8_t *qh = blk->qh;
        const int8_t *sc = blk->scales;
        const float *xb = x + (size_t)b * BN_QK_K;

        for (int n = 0; n < BN_QK_K; n += 128) {
            for (int is = 0; is < 2; is++) {
                float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
                int l0 = is * 16;
                for (int i = 0; i < 16; i++) {
                    int l = l0 + i;
                    int q1 = (int)((ql[l]      & 0xF) |
                                   (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0xF) |
                                   (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l]      >> 4) |
                                   (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4) |
                                   (((qh[l] >> 6) & 3) << 4)) - 32;
                    sum1 += (float)q1 * xb[l + 0];
                    sum2 += (float)q2 * xb[l + 32];
                    sum3 += (float)q3 * xb[l + 64];
                    sum4 += (float)q4 * xb[l + 96];
                }
                row_sum += d * ((float)sc[is + 0] * sum1 +
                                (float)sc[is + 2] * sum2 +
                                (float)sc[is + 4] * sum3 +
                                (float)sc[is + 6] * sum4);
            }
            xb += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return row_sum;
}

static int gpu_refine_q6k_logits_top(float *logits, int n_logits,
                                     const BnQWeight *W, const float *x,
                                     int top_n) {
    if (!logits || !W || !W->data || !x || W->type != BN_GGUF_TENSOR_Q6_K)
        return 0;
    if (top_n <= 0) return 0;
    if (top_n > 128) top_n = 128;
    if (top_n > n_logits) top_n = n_logits;

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

    for (int i = 0; i < n_top; i++)
        logits[ids[i]] = gpu_exact_q6k_row_dot(W, ids[i], x);
    return n_top;
}

static float gpu_exact_q8_row_dot_q8x(const BnQWeight *W, int row,
                                      const int8_t *x_q,
                                      const float *x_scales) {
    int n_blocks_per_row = W->cols / 32;
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)W->data;
    float row_sum = 0.0f;

    for (int b = 0; b < n_blocks_per_row; b++) {
        const BnBlockQ8_0 *blk =
            &blocks[(size_t)row * n_blocks_per_row + b];
        const int8_t *xb = x_q + b * 32;
        int32_t sumi = 0;
        for (int i = 0; i < 32; i++)
            sumi += (int32_t)blk->qs[i] * (int32_t)xb[i];
        row_sum += (float)sumi * bn_fp16_to_fp32(blk->d) * x_scales[b];
    }
    return row_sum;
}

static int gpu_refine_q8_logits_top(float *logits, int n_logits,
                                    const BnQWeight *W, const float *x,
                                    int8_t *x_q, int top_n) {
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || \
    defined(__AVX2__) || defined(__wasm_relaxed_simd__)
    if (!logits || !W || !W->data || !x || !x_q ||
        W->type != BN_GGUF_TENSOR_Q8_0)
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
    bn_quant_x_to_q8_blocks(x, x_q, x_scales, W->cols);
    for (int i = 0; i < n_top; i++)
        logits[ids[i]] = gpu_exact_q8_row_dot_q8x(
            W, ids[i], x_q, x_scales);
    return n_top;
#else
    (void)logits; (void)n_logits; (void)W; (void)x; (void)x_q; (void)top_n;
    return 0;
#endif
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
                 lw->attn.wq.type == lw->attn.wk.type;
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
// Supports attention biases (Qwen2.5) and tied embeddings (BitNet).
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
    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : head_size;
    int half_rope = rope_dims / 2;
    float rope_cos[half_rope], rope_sin[half_rope];
    for (int i = 0; i < half_rope; i++) {
        float angle = pos * s->rope_freq[i];
        rope_cos[i] = cosf(angle);
        rope_sin[i] = sinf(angle);
    }
    int cache_pos = pos % c->seq_len;
    int cpu_fallback_layer = -1;
    int cpu_fallback_from_layer = -1;
    int cpu_fallback_attn_layer = -1;
    int cpu_fallback_attn_from_layer = -1;
    int cpu_fallback_ffn_layer = -1;
    int cpu_fallback_ffn_from_layer = -1;
    int cpu_fallback_ffn_down_from_layer = -1;
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
    int q4_q8_from_layer = -1;
    int q4_q8_to_layer = -1;
    int q4_q8_attn_only = 0;
    int q4_q8_ffn_only = 0;
    {
        static int init = 0;
        static int s_cpu_fallback_layer = -1;
        static int s_cpu_fallback_from_layer = -1;
        static int s_cpu_fallback_attn_layer = -1;
        static int s_cpu_fallback_attn_from_layer = -1;
        static int s_cpu_fallback_ffn_layer = -1;
        static int s_cpu_fallback_ffn_from_layer = -1;
        static int s_cpu_fallback_ffn_down_from_layer = -1;
        static int s_compare_attention_layer = -1;
        static int s_compare_attention_pos = -1;
        static int s_compare_gqa_layer = -1;
        static int s_compare_gqa_pos = -1;
        static int s_compare_qkv_layer = -1;
        static int s_compare_qkv_pos = -1;
        static int s_compare_ffn_down_layer = -1;
        static int s_compare_ffn_down_pos = -1;
        static int s_compare_ffn_state_layer = -1;
        static int s_compare_ffn_state_pos = -1;
        static int s_q4_q8_from_layer = -1;
        static int s_q4_q8_to_layer = -1;
        static int s_q4_q8_attn_only = 0;
        static int s_q4_q8_ffn_only = 0;
        if (!init) {
            const char *env = getenv("BN_GPU_CPU_FALLBACK_FROM_LAYER");
            if (env) s_cpu_fallback_from_layer = atoi(env);
            env = getenv("BN_GPU_CPU_FALLBACK_LAYER");
            if (env) s_cpu_fallback_layer = atoi(env);
            env = getenv("BN_GPU_CPU_ATTN_LAYER");
            if (env) s_cpu_fallback_attn_layer = atoi(env);
            env = getenv("BN_GPU_CPU_ATTN_FROM_LAYER");
            if (env) s_cpu_fallback_attn_from_layer = atoi(env);
            env = getenv("BN_GPU_CPU_FFN_LAYER");
            if (env) s_cpu_fallback_ffn_layer = atoi(env);
            env = getenv("BN_GPU_CPU_FFN_FROM_LAYER");
            if (env) s_cpu_fallback_ffn_from_layer = atoi(env);
            env = getenv("BN_GPU_CPU_FFN_DOWN_FROM_LAYER");
            if (env) s_cpu_fallback_ffn_down_from_layer = atoi(env);
            env = getenv("BN_GPU_COMPARE_ATTENTION_LAYER");
            if (env) s_compare_attention_layer = atoi(env);
            env = getenv("BN_GPU_COMPARE_ATTENTION_POS");
            if (env) s_compare_attention_pos = atoi(env);
            env = getenv("BN_GPU_COMPARE_GQA_LAYER");
            if (env) s_compare_gqa_layer = atoi(env);
            env = getenv("BN_GPU_COMPARE_GQA_POS");
            if (env) s_compare_gqa_pos = atoi(env);
            env = getenv("BN_GPU_COMPARE_QKV_LAYER");
            if (env) s_compare_qkv_layer = atoi(env);
            env = getenv("BN_GPU_COMPARE_QKV_POS");
            if (env) s_compare_qkv_pos = atoi(env);
            env = getenv("BN_GPU_COMPARE_FFN_DOWN_LAYER");
            if (env) s_compare_ffn_down_layer = atoi(env);
            env = getenv("BN_GPU_COMPARE_FFN_DOWN_POS");
            if (env) s_compare_ffn_down_pos = atoi(env);
            env = getenv("BN_GPU_COMPARE_FFN_STATE_LAYER");
            if (env) s_compare_ffn_state_layer = atoi(env);
            env = getenv("BN_GPU_COMPARE_FFN_STATE_POS");
            if (env) s_compare_ffn_state_pos = atoi(env);
            env = getenv("BN_GPU_Q4_Q8_FROM_LAYER");
            if (env) {
                s_q4_q8_from_layer = atoi(env);
            } else if (getenv("BN_GPU_Q4_Q8")) {
                s_q4_q8_from_layer = c->n_layers - 1;
            }
            env = getenv("BN_GPU_Q4_Q8_TO_LAYER");
            if (env) {
                s_q4_q8_to_layer = atoi(env);
            } else {
                env = getenv("BN_GPU_Q4_Q8_TAIL_NATIVE");
                if (env) {
                    int tail_native = atoi(env);
                    if (tail_native > 0) {
                        s_q4_q8_to_layer = c->n_layers - tail_native - 1;
                        if (s_q4_q8_to_layer < -1)
                            s_q4_q8_to_layer = -1;
                    }
                } else if (getenv("BN_GPU_Q4_Q8") &&
                           !getenv("BN_METAL_Q4_PREPARED") &&
                           c->n_layers > 33) {
                    s_q4_q8_to_layer = c->n_layers - 33 - 1;
                }
            }
            s_q4_q8_attn_only = getenv("BN_GPU_Q4_Q8_ATTN_ONLY") != NULL;
            s_q4_q8_ffn_only = getenv("BN_GPU_Q4_Q8_FFN_ONLY") != NULL;
            init = 1;
        }
        cpu_fallback_layer = s_cpu_fallback_layer;
        cpu_fallback_from_layer = s_cpu_fallback_from_layer;
        cpu_fallback_attn_layer = s_cpu_fallback_attn_layer;
        cpu_fallback_attn_from_layer = s_cpu_fallback_attn_from_layer;
        cpu_fallback_ffn_layer = s_cpu_fallback_ffn_layer;
        cpu_fallback_ffn_from_layer = s_cpu_fallback_ffn_from_layer;
        cpu_fallback_ffn_down_from_layer =
            s_cpu_fallback_ffn_down_from_layer;
        compare_attention_layer = s_compare_attention_layer;
        compare_attention_pos = s_compare_attention_pos;
        compare_gqa_layer = s_compare_gqa_layer;
        compare_gqa_pos = s_compare_gqa_pos;
        compare_qkv_layer = s_compare_qkv_layer;
        compare_qkv_pos = s_compare_qkv_pos;
        compare_ffn_down_layer = s_compare_ffn_down_layer;
        compare_ffn_down_pos = s_compare_ffn_down_pos;
        compare_ffn_state_layer = s_compare_ffn_state_layer;
        compare_ffn_state_pos = s_compare_ffn_state_pos;
        q4_q8_from_layer = s_q4_q8_from_layer;
        q4_q8_to_layer = s_q4_q8_to_layer;
        q4_q8_attn_only = s_q4_q8_attn_only;
        q4_q8_ffn_only = s_q4_q8_ffn_only;
    }

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
    int cacheable_decode =
        (!emit_logits || argmax_token ||
         (getenv("BN_CUDA_ENABLE_LOGITS_CACHE") &&
          !bn_transformer_gpu_logits_needs_cpu_fallback(gpu, logit_res))) &&
        gpu->kind == BN_GPU_BACKEND_CUDA && !policy.has_moe &&
        !policy.has_ssm && !getenv("BN_CUDA_DISABLE_DECODE_CACHE") &&
        cpu_fallback_layer < 0 && cpu_fallback_from_layer < 0 &&
        cpu_fallback_attn_layer < 0 && cpu_fallback_attn_from_layer < 0 &&
        cpu_fallback_ffn_layer < 0 && cpu_fallback_ffn_from_layer < 0 &&
        cpu_fallback_ffn_down_from_layer < 0 &&
        compare_attention_layer < 0 && compare_gqa_layer < 0 &&
        compare_qkv_layer < 0 && compare_ffn_down_layer < 0 &&
        compare_ffn_state_layer < 0 && q4_q8_from_layer < 0 &&
        !getenv("BN_GPU_CPU_LOGITS") && !getenv("BN_GPU_COMPARE_LOGITS") &&
        !getenv("BN_METAL_ENABLE_Q6_Q8K");
    int cached_n = cacheable_decode
        ? bn_backend_session_gpu_cached_op_count(sess->backend)
        : 0;
    if (argmax_token && cached_n > 0 &&
        !bn_backend_session_gpu_cached_has_logits(sess->backend)) {
        bn_backend_session_clear_gpu_cached_ops(sess->backend);
        cached_n = 0;
    }
    if (cached_n > 0 && cached_n <= command_cap) {
        if (gpu_patch_cached_decode_ops((BnGPUOp *)command_buffer, cached_n,
                                        c, pos) == 0 &&
            bn_transformer_gpu_execute_ops(
                gpu, command_buffer, cached_n,
                need_logits ? BN_GPU_VALUE_LOGITS : -1,
                need_logits ? s->logits : NULL,
                need_logits ? c->vocab_size : 0) == 0) {
            if (argmax_token &&
                gpu->argmax_activation(
                    gpu->ctx, BN_GPU_VALUE_LOGITS, c->vocab_size,
                    penalty_tokens, n_penalty_tokens, repeat_penalty,
                    argmax_token) != 0) {
                bn_backend_session_clear_gpu_cached_ops(sess->backend);
                bn_transformer_gpu_emit_context_free(&emit);
                return NULL;
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
        int use_q4_q8_layer = q4_q8_from_layer >= 0 &&
                               l >= q4_q8_from_layer &&
                               (q4_q8_to_layer < 0 || l <= q4_q8_to_layer);
        int use_q4_q8_attn = use_q4_q8_layer && !q4_q8_ffn_only;
        int use_q4_q8_ffn = use_q4_q8_layer && !q4_q8_attn_only;

        // ---- SSM layer: CPU fallback until the GPU SSM path is token-coherent ----
        if (!is_attn) {
            int use_cpu_ssm_fallback = 1;
            if (gpu->kind == BN_GPU_BACKEND_CUDA &&
                getenv("BN_CUDA_ENABLE_SSM_GRAPH"))
                use_cpu_ssm_fallback = 0;
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
        if ((cpu_fallback_layer >= 0 && l == cpu_fallback_layer) ||
            (cpu_fallback_from_layer >= 0 && l >= cpu_fallback_from_layer)) {
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
        if (cpu_fallback_attn_layer == l ||
            (cpu_fallback_attn_from_layer >= 0 &&
             l >= cpu_fallback_attn_from_layer)) {
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
            int use_cpu_moe_fallback = 1;
            if (gpu->kind == BN_GPU_BACKEND_CUDA &&
                getenv("BN_CUDA_ENABLE_UNSAFE_MOE_FFN"))
                use_cpu_moe_fallback = 0;
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
            if (bn_transformer_gpu_emit_context_flush(&emit, gpu) != 0 ||
                bn_transformer_gpu_read_xb(gpu, s->xb,
                                           (size_t)dim * sizeof(float)) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe route input readback failed");
            if (!sess->moe_state)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe session state missing");
            bn_moe_route(sess->moe_state, s->xb, lw->moe.router_weight,
                         dim, c->n_experts, c->n_experts_active,
                         bn_model_pool(m));
            int compare_moe = 0;
            const char *compare_moe_env = getenv("BN_GPU_COMPARE_MOE_LAYER");
            if (compare_moe_env) {
                int compare_layer = atoi(compare_moe_env);
                const char *compare_pos_env = getenv("BN_GPU_COMPARE_MOE_POS");
                int compare_pos = compare_pos_env ? atoi(compare_pos_env) : -1;
                compare_moe = compare_layer == l &&
                              (compare_pos < 0 || compare_pos == pos);
            }
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
            BnGPUMoEResolvedExpert expert_emit[BN_MAX_MOE_K];
            BnGPUMoEResources moe_res;
            if (bn_gpu_moe_bridge_resolve_resources(
                    &moe_res, expert_emit, BN_MAX_MOE_K, m, sess, lw, l,
                    &moe_temporaries) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu moe resource resolution failed");
            BnTransformerGPUMoESharedResources moe_shared =
                bn_transformer_gpu_resolve_moe_shared_resources(
                    backend, lw, l);
            bn_transformer_gpu_emit_context_moe(
                &emit, &moe_res, &moe_shared, lw, dim, u_eps, next_norm);
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
        if ((cpu_fallback_ffn_layer >= 0 && l == cpu_fallback_ffn_layer) ||
            (cpu_fallback_ffn_from_layer >= 0 &&
             l >= cpu_fallback_ffn_from_layer)) {
            if (bn_transformer_gpu_fallback_cpu_ffn(
                    &emit, gpu, m, sess, lw, &ffn_plan, dim, u_eps,
                    next_norm) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu cpu-ffn fallback failed");
            continue;
        }
        BnTransformerGPUDenseFFNResources ffn_res = layer_ffn_res;
        int ffn_down_input_buf = -1;
        int skip_ffn_down = cpu_fallback_ffn_down_from_layer >= 0 &&
                            l >= cpu_fallback_ffn_down_from_layer;
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
            next_norm, skip_ffn_down, &ffn_down_input_buf, use_q4_q8_ffn);
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
    if (emit_logits) {
        if (getenv("BN_GPU_CPU_LOGITS") ||
            bn_transformer_gpu_logits_needs_cpu_fallback(gpu, logit_res)) {
            if (argmax_token)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu argmax requires gpu logits");
            if (bn_transformer_gpu_fallback_logits(
                    &emit, gpu, m, sess, logit_res, dim) != 0)
                return bn_transformer_gpu_reject_forward(
                    &emit, "gpu logits cpu fallback failed");
            return s->logits;
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
                                                   emit_logits);
    if (argmax_token) {
        if (gpu->argmax_activation(
                gpu->ctx, BN_GPU_VALUE_LOGITS, c->vocab_size,
                penalty_tokens, n_penalty_tokens, repeat_penalty,
                argmax_token) != 0)
            return bn_transformer_gpu_reject_forward(
                &emit, "gpu argmax failed");
        if (getenv("BN_GPU_DEBUG_ARGMAX_COMPARE") &&
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
    if (getenv("BN_METAL_ENABLE_Q6_Q8K") &&
        logit_res->type == BN_GGUF_TENSOR_Q6_K &&
        logit_res->cpu_weight) {
        int refine_top = 8;
        const char *env = getenv("BN_GPU_Q6_Q8K_REFINE_TOP");
        if (env) refine_top = atoi(env);
        if (refine_top > 0 &&
            bn_transformer_gpu_read_xb(gpu, s->xb,
                                       (size_t)dim * sizeof(float)) == 0) {
            gpu_refine_q6k_logits_top(s->logits, c->vocab_size,
                                      logit_res->cpu_weight, s->xb,
                                      refine_top);
        }
    }
    if (!getenv("BN_GPU_DISABLE_Q8_LOGITS_REFINE") &&
        logit_res->type == BN_GGUF_TENSOR_Q8_0 &&
        logit_res->cpu_weight) {
        int refine_top = 8;
        const char *env = getenv("BN_GPU_Q8_REFINE_TOP");
        if (env) refine_top = atoi(env);
        if (refine_top > 0 &&
            bn_transformer_gpu_read_xb(gpu, s->xb,
                                       (size_t)dim * sizeof(float)) == 0) {
            gpu_refine_q8_logits_top(s->logits, c->vocab_size,
                                     logit_res->cpu_weight, s->xb,
                                     s->x_q, refine_top);
        }
    }
    if (getenv("BN_GPU_COMPARE_LOGITS")) {
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
