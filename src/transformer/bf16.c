#include "bf16_internal.h"
#include "backend_layout.h"
#include "backend_quant.h"
#include "model_internal.h"
#include "model_anatomy.h"
#include "transformer_prefill_internal.h"
#include "session.h"
#include "session_internal.h"
#include "moe.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

#define BF16 bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_BF16)

typedef struct {
    BnGPUResidentView state[2], history[2], key, value;
} BF16LayerState;

typedef struct {
    const BnModel *model;
    BnGPUBackend *gpu;
    const BnBackendBF16Model *weights;
    void *request, *cos_sin;
    int planning;size_t planned_bytes;
    int max_tokens, capacity, position, parity, failed, rotary_capacity;
    BnTransformerBF16Arithmetic arithmetic;
    BnTransformerBF16FFNArithmetic ffn_arithmetic;
    BnGPUAttentionSplitPolicy attention_split_policy;
    BnTransformerBF16GDNPrefillSelection gdn_prefill_selection;
    BnGPURMSNormReduction post_norm_reductions[64];
    BnGPURMSNormReduction input_norm_reductions[65];
    uint16_t *host_logits;
    int32_t *host_tokens;
    BF16LayerState layers[64];
    BnGPUResidentView embedding, norm, residual_a, residual_b, ffn_output;
    BnGPUResidentView block_output, gate, up, qkv, z, ba, conv_output;
    BnGPUResidentView q_gate, k_input, v_input, q, k, v, g;
    BnGPUResidentView attention, core, gated, transposed, log_decay, beta;
    BnGPUResidentView ids, logits;
} BF16Request;

static const char bf16_request_owner = 0;

static int bf16_flash_runtime_query(const BnBackendSession *session, int query);
static int bf16_flash_runtime_restart(const BnModel *model, BnSession *session);
static int bf16_flash_runtime_run(BnModel *model, BnSession *session,
    const int *tokens, int nt, int position, int want_logits, int prefill);

static int bf16_glm_runtime_query(const BnBackendSession *session, int query);
static int bf16_glm_runtime_restart(const BnModel *model, BnSession *session);
static int bf16_glm_runtime_run(BnModel *model, BnSession *session,
    const int *tokens, int nt, int position, int want_logits, int prefill);

static void bf16_request_free(void *ptr) {
    BF16Request *r = ptr;
    free(r->host_logits);
    free(r->host_tokens);
    free(r);
}

static BF16Request *bf16_request(const BnBackendSession *session) {
    return bn_backend_session_execution_resources(session, &bf16_request_owner);
}

int bn_transformer_bf16_arithmetic(const BnBackendSession *session) {
    const BF16Request *r = bf16_request(session);
    return r && !r->failed ? (int)r->arithmetic : -1;
}

int bn_transformer_bf16_position(const BnBackendSession *session) {
    const BF16Request *r = bf16_request(session);
    if (r) return !r->failed ? r->position : -1;
    int result = bf16_flash_runtime_query(session, 0);
    return result >= 0 ? result : bf16_glm_runtime_query(session, 0);
}

int bn_transformer_bf16_max_tokens(const BnBackendSession *session) {
    const BF16Request *r = bf16_request(session);
    if (r) return !r->failed ? r->max_tokens : 0;
    int result = bf16_flash_runtime_query(session, 1);
    return result > 0 ? result : bf16_glm_runtime_query(session, 1);
}

int bn_transformer_bf16_capacity(const BnBackendSession *session) {
    const BF16Request *r = bf16_request(session);
    if (r) return !r->failed ? r->capacity : 0;
    int result = bf16_flash_runtime_query(session, 2);
    return result > 0 ? result : bf16_glm_runtime_query(session, 2);
}

static BnGPUResidentView bf16_slice(BnGPUResidentView view, size_t count) {
    return bn_gpu_resident_view_slice(view, 0, count);
}

static int bf16_alloc(BF16Request *r, BnGPUResidentView *view, int type,
    size_t count, int initialize_zero) {
    if (r->planning) {
        size_t width=bn_backend_quant_resident_scalar_bytes(type);
        if(!width||count>SIZE_MAX/width||count*width>SIZE_MAX-r->planned_bytes)return -1;
        r->planned_bytes+=count*width;return 0;
    }
    if (r->gpu->resident_alloc(r->request, type, count, view)) return -1;
    return initialize_zero ? r->gpu->resident_zero(r->request, *view) : 0;
}

static int bf16_alloc_activations(BF16Request *r,int max_tokens,int logit_rows) {
#define ALLOC(field, width) \
    if (bf16_alloc(r, &r->field, BF16, (size_t)max_tokens * (width), 0)) return -1
    ALLOC(embedding, 5120); ALLOC(norm, 5120);
    ALLOC(residual_a, 5120); ALLOC(residual_b, 5120);
    ALLOC(ffn_output, 5120); ALLOC(block_output, 5120);
    ALLOC(gate, r->ffn_arithmetic==BN_TRANSFORMER_BF16_FFN_COMPILED&&
        r->weights->stacked_ffn_layers?34816:17408); ALLOC(up, 17408);
    ALLOC(qkv, 10240); ALLOC(z, 6144); ALLOC(ba, 96); ALLOC(conv_output, 10240);
    ALLOC(q_gate, 12288); ALLOC(k_input, 1024); ALLOC(v_input, 1024);
    ALLOC(q, 6144); ALLOC(k, 6144); ALLOC(v, 6144); ALLOC(g, 6144);
    ALLOC(attention, 6144); ALLOC(core, 6144); ALLOC(gated, 6144); ALLOC(transposed, 6144);
#undef ALLOC
    if (bf16_alloc(r, &r->log_decay, bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32), (size_t)max_tokens * 48, 0) ||
        bf16_alloc(r, &r->beta, bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32), (size_t)max_tokens * 48, 0) ||
        bf16_alloc(r, &r->ids, bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32), (size_t)max_tokens, 0) ||
        bf16_alloc(r, &r->logits, BF16, (size_t)logit_rows*r->model->config.vocab_size, 0)) return -1;
    return 0;
}

static int bf16_capable(const BnGPUBackend *g) {
    return bn_gpu_backend_can_resident_request(g) && g->resident_zero &&
        g->linear_bf16_reserve && g->linear_bf16_resident &&
        g->embedding_bf16_reserve && g->embedding_bf16_resident &&
        g->gdn_prefill_bf16_reserve && g->gdn_prefill_bf16_resident &&
        g->gdn_prepare_bf16_resident && g->gdn_decode_bf16_resident &&
        g->causal_conv_bf16_resident && g->gated_rmsnorm_bf16_resident &&
        g->rmsnorm_bf16_resident && g->add_rmsnorm_bf16_resident &&
        g->qk_prep_bf16_resident && g->attention_bf16_reserve &&
        g->attention_bf16_resident && g->transpose_axes_bf16_resident &&
        g->join_columns_resident && g->dense_ffn_bf16_resident;
}

int bn_transformer_bf16_session_restart(const BnModel *model, BnSession *session) {
    if (!model || !session || !session->backend_only) return -1;
    BnBackendSession *backend = bn_session_backend(session);
    BF16Request *r = bf16_request(backend);
    if (!r) {
        if (!bf16_flash_runtime_restart(model, session)) return 0;
        return bf16_glm_runtime_restart(model, session);
    }
    if (!r || r->failed || r->model != model || r->gpu != bn_model_gpu(model) ||
        r->weights != bn_backend_layout_bf16(bn_model_backend(model))) return -1;
    /* Failure after even a partial clear destroys the previous transaction. */
    r->failed = 1;
    for (int l = 0; l < r->weights->n_layers; ++l) {
        if (model->weights.layers[l].block_kind != BN_LAYER_BLOCK_SSM) continue;
        if (r->gpu->resident_zero(r->request, r->layers[l].state[0]) ||
            r->gpu->resident_zero(r->request, r->layers[l].history[0])) return -1;
    }
    if (r->gpu->resident_synchronize(r->request)) return -1;
    r->position = 0;
    r->parity = 0;
    r->failed = 0;
    memset(session->state.token_history, 0,
        (size_t)model->config.seq_len * sizeof(*session->state.token_history));
    session->pos = 0;
    session->gpu_kv_direct_valid = 0;
    session->moe_cache_prefill = 0;
    session->state.batched_prompt_contract = 0;
    bn_backend_session_set_gpu_decode_ssm_pending(backend, 0);
    return 0;
}

int bn_transformer_bf16_request_init(const BnModel *model, BnGPUBackend *gpu,
    BnBackendSession *session, int max_tokens, int capacity,
    void *cos_sin, int rotary_capacity) {
    return bn_transformer_bf16_request_init_with_arithmetic(model, gpu, session,
        max_tokens, capacity, cos_sin, rotary_capacity, BN_TRANSFORMER_BF16_ARITHMETIC_EAGER);
}

static BnGPURMSNormReduction bf16_base_reduction(BnTransformerBF16Arithmetic arithmetic) {
    return arithmetic>=BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32
        ? BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA : BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32;
}

static void bf16_policy_copy(BF16Request *r,const BnTransformerBF16Policy *policy) {
    r->attention_split_policy=policy->attention_split_policy;
    r->gdn_prefill_selection=policy->gdn_prefill_selection;
    r->ffn_arithmetic=policy->ffn_arithmetic!=BN_TRANSFORMER_BF16_FFN_DEFAULT
        ?policy->ffn_arithmetic:policy->arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_EAGER
        ?BN_TRANSFORMER_BF16_FFN_DEFAULT:BN_TRANSFORMER_BF16_FFN_FUSED_RN;
    for (int l=0;l<64;l++) r->post_norm_reductions[l]=
        policy->post_norm_reductions[l]==BN_GPU_RMSNORM_REDUCTION_DEFAULT
            ? bf16_base_reduction(policy->arithmetic) : policy->post_norm_reductions[l];
    for (int l=0;l<65;l++) r->input_norm_reductions[l]=
        policy->input_norm_reductions[l]==BN_GPU_RMSNORM_REDUCTION_DEFAULT
            ? bf16_base_reduction(policy->arithmetic) : policy->input_norm_reductions[l];
}

static int bf16_policy_capable(const BnGPUBackend *g,
    const BnTransformerBF16Policy *policy) {
    if (!g || !policy) return 0;
    if (policy->gdn_prefill_selection!=BN_TRANSFORMER_BF16_GDN_PREFILL_DEFAULT &&
        (policy->gdn_prefill_selection!=BN_TRANSFORMER_BF16_GDN_PREFILL_SINGLE_CP ||
         policy->arithmetic!=BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32 ||
         (g->resident_gdn_prefill_recurrences & ((1u<<BN_GPU_GDN_PREFILL_CP)|(1u<<BN_GPU_GDN_PREFILL_NONCP)))
             != ((1u<<BN_GPU_GDN_PREFILL_CP)|(1u<<BN_GPU_GDN_PREFILL_NONCP)))) return 0;
    if(policy->attention_split_policy!=BN_GPU_ATTENTION_SPLIT_SINGLETON &&
       (policy->attention_split_policy!=BN_GPU_ATTENTION_SPLIT_BATCH ||
        !g->attention_decode_group_bf16_resident ||
        !(g->resident_attention_split_policies&(1u<<BN_GPU_ATTENTION_SPLIT_BATCH))))return 0;
    if (policy->ffn_arithmetic!=BN_TRANSFORMER_BF16_FFN_DEFAULT) {
        if (policy->arithmetic<BN_TRANSFORMER_BF16_ARITHMETIC_FP32_INTERMEDIATES ||
            (policy->ffn_arithmetic!=BN_TRANSFORMER_BF16_FFN_FUSED_RN &&
             policy->ffn_arithmetic!=BN_TRANSFORMER_BF16_FFN_COMPILED)) return 0;
        unsigned profile=policy->ffn_arithmetic==BN_TRANSFORMER_BF16_FFN_COMPILED
            ?BN_GPU_BF16_FFN_SILU_COMPILED:BN_GPU_BF16_FFN_SILU_FUSED;
        if (!(g->resident_ffn_activation_profiles&(1u<<profile))) return 0;
    }
    for (int l=0;l<129;l++) {
        BnGPURMSNormReduction r=l<64?policy->post_norm_reductions[l]:policy->input_norm_reductions[l-64];
        if (r==BN_GPU_RMSNORM_REDUCTION_DEFAULT || r==bf16_base_reduction(policy->arithmetic)) continue;
        if (policy->arithmetic<BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32 ||
            (r!=BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32 &&
             r!=BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA &&
             r!=BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA &&
             r!=BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT &&
             r!=BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED &&
             r!=BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES) ||
            !(g->resident_rmsnorm_reductions & (1u << r))) return 0;
    }
    return 1;
}

int bn_transformer_bf16_request_init_with_arithmetic(const BnModel *model,
    BnGPUBackend *gpu, BnBackendSession *session, int max_tokens, int capacity,
    void *cos_sin, int rotary_capacity, BnTransformerBF16Arithmetic arithmetic) {
    BnTransformerBF16Policy policy={0};policy.arithmetic=arithmetic;
    return bn_transformer_bf16_request_init_with_policy(model,gpu,session,
        max_tokens,capacity,cos_sin,rotary_capacity,&policy);
}

static int bf16_dense_storage(BF16Request *r,int capacity) {
    int status = 0;
    for (int l = 0; l < 64 && !status; ++l) {
        BF16LayerState *s = &r->layers[l];
        if (r->model->weights.layers[l].block_kind == BN_LAYER_BLOCK_SSM) {
            status = bf16_alloc(r, &s->state[0], bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32), 48u * 128 * 128, 1) ||
                bf16_alloc(r, &s->state[1], bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32), 48u * 128 * 128, 0) ||
                bf16_alloc(r, &s->history[0], BF16, 3u * 10240, 1) ||
                bf16_alloc(r, &s->history[1], BF16, 3u * 10240, 0);
        } else {
            status = bf16_alloc(r, &s->key, BF16, (size_t)capacity * 1024, 0) ||
                bf16_alloc(r, &s->value, BF16, (size_t)capacity * 1024, 0);
        }
    }
    if (status) return -1;
    if (bf16_alloc_activations(r,r->max_tokens,1)) return -1;
    return 0;
}

static const BnBackendBF16Model *bf16_dense_admit(const BnModel *model,
    BnGPUBackend *gpu,int max_tokens,int capacity,void *cos_sin,int rotary_capacity,
    const BnTransformerBF16Policy *policy) {
    BnTransformerBF16Arithmetic arithmetic=policy?policy->arithmetic:BN_TRANSFORMER_BF16_ARITHMETIC_EAGER;
    if (!model || !bf16_capable(gpu) || !bf16_policy_capable(gpu,policy) ||
        (arithmetic != BN_TRANSFORMER_BF16_ARITHMETIC_EAGER &&
         arithmetic != BN_TRANSFORMER_BF16_ARITHMETIC_FP32_INTERMEDIATES &&
         arithmetic != BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32 &&
         arithmetic != BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32 &&
         arithmetic != BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32_GATED_R2 &&
         arithmetic != BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32) ||
        (arithmetic == BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32 &&
         !(gpu->resident_gdn_prefill_recurrences & (1u << BN_GPU_GDN_PREFILL_NONCP))) ||
        (arithmetic != BN_TRANSFORMER_BF16_ARITHMETIC_EAGER &&
         !gpu->add3_rmsnorm_bf16_resident) ||
        (arithmetic >= BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32 &&
         !(gpu->resident_rmsnorm_reductions &
           (1u << BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA))) ||
        (arithmetic >= BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32 &&
         (!(gpu->resident_attention_gate_profiles &
            (1u << BN_GPU_ATTENTION_GATE_COMPILED_F32)) ||
          !gpu->gdn_cp_prefill_chunk_tokens || !gpu->gdn_cp_prefill_bf16_workspace_size ||
          !gpu->gdn_cp_prefill_bf16_reserve || !gpu->gdn_cp_prefill_bf16_resident ||
          !gpu->gdn_cp_prefill_chunk_tokens(gpu->ctx,max_tokens) ||
          !gpu->gdn_cp_prefill_bf16_workspace_size(gpu->ctx,max_tokens,512))) ||
        (arithmetic == BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32_GATED_R2 &&
         !(gpu->resident_gated_norm_profiles & (1u << BN_GPU_GATED_NORM_COMPILED_R2)))) return NULL;
    const BnConfig *c = &model->config;
    const BnBackendBF16Model *weights = bn_backend_layout_bf16(bn_model_backend(model));
    if (weights && !cos_sin) {
        cos_sin = weights->rotary_cache;
        rotary_capacity = weights->rotary_capacity;
    }
    if (!weights || !cos_sin || weights->n_layers != 64 || max_tokens <= 0 ||
        max_tokens > weights->max_tokens || max_tokens > INT_MAX / 17408 ||
        max_tokens > capacity || capacity <= 0 || capacity > INT_MAX / 1024 ||
        capacity > c->seq_len || capacity > rotary_capacity ||
        c->bf16_layout != BN_MODEL_BF16_LAYOUT_HYBRID_OFFSET_NORM ||
        bn_model_transformer_policy_rope_dims_for_head(c, 256) != 64 ||
        bn_model_transformer_policy_norm_epsilon(c) != 1e-6f ||
        c->ssm_qk_norm != BN_SSM_QK_NORM_RMS_SCALED ||
        bn_model_gpu(model) != gpu) return NULL;
    return weights;
}

size_t bn_transformer_bf16_dense_kv_bytes(const BnModel *model,int capacity) {
    if(!model||!model->weights.layers||capacity<1||capacity>model->config.seq_len||
       model->config.n_layers!=64||model->config.bf16_layout!=BN_MODEL_BF16_LAYOUT_HYBRID_OFFSET_NORM)return 0;
    size_t layers=0;
    for(int l=0;l<64;l++) {
        if(model->weights.layers[l].block_kind==BN_LAYER_BLOCK_ATTENTION)layers++;
        else if(model->weights.layers[l].block_kind!=BN_LAYER_BLOCK_SSM)return 0;
    }
    return !layers||(size_t)capacity>SIZE_MAX/4096/layers?0:(size_t)capacity*4096*layers;
}

int bn_transformer_bf16_dense_memory_plan(const BnModel *model,
    BnGPUBackend *gpu,int max_tokens,int capacity,
    const BnTransformerBF16Policy *policy,BnTransformerBF16RequestMemory *out) {
    const BnBackendBF16Model *weights=bf16_dense_admit(model,gpu,max_tokens,capacity,NULL,0,policy);
    if(!out||!weights||!model->weights.layers||!gpu->linear_bf16_workspace_size||
       !gpu->linear_bf16_input_workspace_size||!gpu->embedding_bf16_workspace_size||
       !gpu->attention_bf16_workspace_size||!gpu->gdn_prefill_bf16_workspace_size)return -1;
    BF16Request r={0};r.planning=1;r.model=model;r.gpu=gpu;r.weights=weights;r.max_tokens=max_tokens;r.arithmetic=policy->arithmetic;
    bf16_policy_copy(&r,policy);
    if(bf16_dense_storage(&r,capacity))return -1;
    size_t workspace[]={gpu->linear_bf16_workspace_size(gpu->ctx),
        gpu->linear_bf16_input_workspace_size(gpu->ctx,(size_t)max_tokens*17408),
        gpu->embedding_bf16_workspace_size(gpu->ctx),
        gpu->attention_bf16_workspace_size(gpu->ctx,max_tokens,24,
            policy->attention_split_policy==BN_GPU_ATTENTION_SPLIT_BATCH&&capacity<8192?8192:capacity),
        policy->arithmetic>=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32?
            gpu->gdn_cp_prefill_bf16_workspace_size(gpu->ctx,max_tokens,512):
            gpu->gdn_prefill_bf16_workspace_size(gpu->ctx,max_tokens)};
    for(size_t i=0;i<sizeof(workspace)/sizeof(*workspace);i++) {
        if(!workspace[i]||workspace[i]>SIZE_MAX-r.planned_bytes)return -1;
        r.planned_bytes+=workspace[i];
    }
    size_t host=sizeof(r)+(size_t)model->config.vocab_size*sizeof(uint16_t)+(size_t)max_tokens*sizeof(int32_t);
    *out=(BnTransformerBF16RequestMemory){r.planned_bytes,host,host};return 0;
}

int bn_transformer_bf16_request_init_with_policy(const BnModel *model,
    BnGPUBackend *gpu, BnBackendSession *session, int max_tokens, int capacity,
    void *cos_sin, int rotary_capacity, const BnTransformerBF16Policy *policy) {
    if(!session||bn_backend_session_has_gpu_resources(session)||
       bn_backend_session_has_execution_resources(session))return -1;
    const BnBackendBF16Model *weights=bf16_dense_admit(model,gpu,max_tokens,capacity,
        cos_sin,rotary_capacity,policy);
    if(!weights)return -1;
    const BnConfig *c=&model->config;
    BnTransformerBF16Arithmetic arithmetic=policy->arithmetic;
    if(!cos_sin){cos_sin=weights->rotary_cache;rotary_capacity=weights->rotary_capacity;}
    BF16Request *r = calloc(1, sizeof(*r));
    if (!r) return -1;
    r->host_logits = malloc((size_t)c->vocab_size * sizeof(*r->host_logits));
    r->host_tokens = malloc((size_t)max_tokens * sizeof(*r->host_tokens));
    if (!r->host_logits || !r->host_tokens) { bf16_request_free(r); return -1; }
    r->model = model;
    r->gpu = gpu;
    r->weights = weights;
    r->arithmetic = arithmetic;
    bf16_policy_copy(r,policy);
    r->cos_sin = cos_sin;
    r->max_tokens = max_tokens;
    r->capacity = capacity;
    r->rotary_capacity = rotary_capacity;
    r->request = bn_gpu_backend_resident_request(gpu, session);
    if (!r->request) { bf16_request_free(r); return -1; }
    /* Cohort selection can request more partitions than a short member's
     * singleton heuristic. Reserve all 128 FA2 split slots without changing
     * the request's actual KV capacity or accessible token range. */
    if (gpu->linear_bf16_reserve(gpu->ctx, r->request, (size_t)max_tokens * 17408) ||
        gpu->embedding_bf16_reserve(gpu->ctx, r->request) ||
        (arithmetic >= BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32
            ? gpu->gdn_cp_prefill_bf16_reserve(gpu->ctx,r->request,max_tokens,512)
            : gpu->gdn_prefill_bf16_reserve(gpu->ctx,r->request,max_tokens)) ||
        gpu->attention_bf16_reserve(gpu->ctx, r->request, max_tokens, 24,
            r->attention_split_policy==BN_GPU_ATTENTION_SPLIT_BATCH&&capacity<8192
                ?8192:capacity)) goto fail;
    if(bf16_dense_storage(r,capacity))goto fail;
    if (bn_backend_session_attach_execution_resources(session, &bf16_request_owner,
            r, bf16_request_free)) goto fail;
    return 0;
fail:
    bn_backend_session_release_gpu_resources(session);
    bf16_request_free(r);
    fprintf(stderr, "BF16 request preparation failed\n");
    return -1;
}

int bn_transformer_bf16_resize_idle_kv(const BnModel *model,BnSession *session,int capacity) {
    BF16Request *r=session?bf16_request(bn_session_backend(session)):NULL;
    if(!r||r->failed||r->model!=model||!r->gpu->resident_resize_discard||
       r->position||r->parity||capacity<r->max_tokens||capacity>INT_MAX/1024||
       capacity>model->config.seq_len||capacity>r->rotary_capacity)return -1;
    if(capacity==r->capacity)return 0;
    /* Only restart-cleared, idle requests enter this boundary. Free-before-
     * allocate keeps each intermediate KV footprint between old and new sizes.
     * Any partial failure poisons the request; the pool must discard it. */
    for(int l=0;l<64;l++)if(model->weights.layers[l].block_kind==BN_LAYER_BLOCK_ATTENTION) {
        BF16LayerState *s=&r->layers[l];
        if(r->gpu->resident_resize_discard(r->request,&s->key,(size_t)capacity*1024)||
           r->gpu->resident_resize_discard(r->request,&s->value,(size_t)capacity*1024))goto resize_fail;
    }
    if(r->gpu->attention_bf16_reserve(r->gpu->ctx,r->request,r->max_tokens,24,
        r->attention_split_policy==BN_GPU_ATTENTION_SPLIT_BATCH&&capacity<8192?8192:capacity))goto resize_fail;
    r->capacity=capacity;return 0;
resize_fail:
    r->failed=1;return -1;
}

static int bf16_linear(BF16Request *r, BnGPUBF16LinearWeight weight,
    int rows, int cols, int nt, BnGPUResidentView x, BnGPUResidentView out) {
    BnGPUBF16LinearResidentPlan p = {0};
    p.n_tokens = nt; p.rows = rows; p.cols = cols;
    p.weight = weight; p.x = x; p.out = out;
    return r->gpu->linear_bf16_resident(r->gpu->ctx, r->request, &p);
}

static int bf16_norm(BF16Request *r, void *weight, int nt,
    BnGPUResidentView x, BnGPUResidentView out) {
    BnGPURMSNormResidentPlan p = {0};
    p.n_tokens = nt; p.dim = 5120; p.eps = 1e-6f; p.weight_offset = 1;
    p.reduction = r->arithmetic >= BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32
        ? BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA
        : BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32;
    p.reduction=r->input_norm_reductions[0];
    p.weight = weight; p.x = x; p.out = out;
    return r->gpu->rmsnorm_bf16_resident(r->gpu->ctx, r->request, &p);
}

static int bf16_add_norm(BF16Request *r, void *weight, int nt, int layer,
    BnGPUResidentView x, BnGPUResidentView residual, BnGPUResidentView out,
    BnGPUResidentView next) {
    BnGPUAddRMSNormResidentPlan p = {0};
    p.n_tokens = nt; p.dim = 5120; p.eps = 1e-6f; p.weight_offset = 1;
    p.reduction = r->arithmetic >= BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32
        ? BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA
        : BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32;
    if (layer>=0 && r->post_norm_reductions[layer]!=BN_GPU_RMSNORM_REDUCTION_DEFAULT)
        p.reduction=r->post_norm_reductions[layer];
    p.weight = weight; p.x = x; p.residual = residual;
    p.out = out; p.residual_out = next;
    return r->gpu->add_rmsnorm_bf16_resident(r->gpu->ctx, r->request, &p);
}

static int bf16_add3_norm(BF16Request *r, void *weight, int nt, int layer,
    BnGPUResidentView block, BnGPUResidentView residual, BnGPUResidentView hidden,
    BnGPUResidentView out, BnGPUResidentView next) {
    BnGPUAdd3RMSNormResidentPlan p = {0};
    p.n_tokens = nt; p.dim = 5120; p.eps = 1e-6f; p.weight_offset = 1;
    p.reduction = r->arithmetic >= BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32
        ? BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA
        : BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32;
    p.reduction=r->input_norm_reductions[layer];
    p.weight = weight; p.x = block; p.residual = residual; p.addend = hidden;
    p.out = out; p.residual_out = next;
    return r->gpu->add3_rmsnorm_bf16_resident(r->gpu->ctx, r->request, &p);
}

static int bf16_ssm_projections(BF16Request *r,int layer,int nt,BnGPUResidentView input) {
    BnGPUBackend *g=r->gpu;const BnBackendBF16Layer *w=&r->weights->layers[layer];
    BnGPUResidentView qkv=bf16_slice(r->qkv,(size_t)nt*10240);
    BnGPUResidentView z=bf16_slice(r->z,(size_t)nt*6144),ba=bf16_slice(r->ba,(size_t)nt*96);
    if (w->split_qk_v) {
        BnGPUJoinColumnsResidentPlan p = {0};
        p.rows = nt; p.cols = 10240; p.left_cols = 4096;
        p.left = bf16_slice(r->q_gate, (size_t)nt * 4096);
        p.right = bf16_slice(r->g, (size_t)nt * 6144); p.x = qkv;
        BnGPUJoinColumnsResidentPlan qk = {0};
        qk.rows = nt; qk.cols = 4096; qk.left_cols = 2048;
        qk.left = bf16_slice(r->q, (size_t)nt * 2048);
        qk.right = bf16_slice(r->k, (size_t)nt * 2048); qk.x = p.left;
        if (bf16_linear(r, w->qkv, 2048, 5120, nt, input, qk.left) ||
            bf16_linear(r, w->k, 2048, 5120, nt, input, qk.right) ||
            g->join_columns_resident(g->ctx, r->request, &qk) ||
            bf16_linear(r, w->v, 6144, 5120, nt, input, p.right) ||
            g->join_columns_resident(g->ctx, r->request, &p)) return -1;
    } else if (bf16_linear(r, w->qkv, 10240, 5120, nt, input, qkv)) return -1;
    if (bf16_linear(r, w->z, 6144, 5120, nt, input, z) ||
        bf16_linear(r, w->ba, 96, 5120, nt, input, ba)) return -1;
    return 0;
}
static int bf16_ssm_finish(BF16Request *r,int layer,int nt,BnGPUResidentView output) {
    BnGPUBackend *g=r->gpu;const BnBackendBF16Layer *w=&r->weights->layers[layer];
    BnGPUResidentView core=bf16_slice(r->core,(size_t)nt*6144),z=bf16_slice(r->z,(size_t)nt*6144);
    BnGPUResidentView gated=bf16_slice(r->gated,(size_t)nt*6144),transposed=bf16_slice(r->transposed,(size_t)nt*6144);
    BnGPUGatedRMSNormResidentPlan gn = {0};
    gn.rows = nt * 48; gn.dim = 128; gn.eps = 1e-6f;
    gn.profile = r->arithmetic == BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32_GATED_R2
        ? BN_GPU_GATED_NORM_COMPILED_R2 : BN_GPU_GATED_NORM_FLA;
    gn.weight = w->gn; gn.x = core; gn.z = z; gn.out = gated;
    if (g->gated_rmsnorm_bf16_resident(g->ctx, r->request, &gn)) return -1;
    BnGPUTransposeAxesResidentPlan tp = {0};
    tp.n_tokens = nt; tp.outer = 16; tp.inner = 3; tp.width = 128;
    tp.x = gated; tp.out = transposed;
    return g->transpose_axes_bf16_resident(g->ctx, r->request, &tp) ||
        bf16_linear(r, w->o, 5120, 6144, nt, transposed, output) ? -1 : 0;
}

/* Count recurrent prefill roles, including one-token tails. Logical prompt
 * progress and the number of decode participants do not select recurrence. */
static BnGPUGDNPrefillRecurrence bf16_prefill_recurrence(const BF16Request *r,int count) {
    if (r->gdn_prefill_selection==BN_TRANSFORMER_BF16_GDN_PREFILL_SINGLE_CP && count==1)
        return BN_GPU_GDN_PREFILL_CP;
    return r->arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32
        ? BN_GPU_GDN_PREFILL_NONCP : BN_GPU_GDN_PREFILL_CP;
}

static int bf16_ssm(BF16Request *r, int layer, int nt,
    BnGPUResidentView input, BnGPUResidentView output, int prefill) {
    BnGPUBackend *g = r->gpu;
    const BnBackendBF16Layer *w = &r->weights->layers[layer];
    BF16LayerState *s = &r->layers[layer];
    BnGPUResidentView qkv = bf16_slice(r->qkv, (size_t)nt * 10240);
    BnGPUResidentView ba = bf16_slice(r->ba, (size_t)nt * 96);
    BnGPUResidentView post = bf16_slice(r->conv_output, (size_t)nt * 10240);
    BnGPUResidentView core = bf16_slice(r->core, (size_t)nt * 6144);
    if (bf16_ssm_projections(r,layer,nt,input)) return -1;
    BnGPUCausalConvResidentPlan cv = {0};
    cv.n_tokens = nt; cv.dim = 10240; cv.kernel = 4; cv.weight = w->conv;
    cv.x = qkv; cv.initial_history = s->history[r->parity];
    cv.next_history = s->history[1-r->parity]; cv.out = post;
    if (g->causal_conv_bf16_resident(g->ctx, r->request, &cv)) return -1;
    if (!prefill) {
        BnGPUGDNDecodeResidentPlan p = {0};
        p.hk = 16; p.hv = 48; p.dk = p.dv = 128; p.q_scale = 0x1.6a09e6p-4f;
        p.a_log = w->an; p.dt_bias = w->dt; p.x = post; p.ba = ba;
        p.initial_state = s->state[r->parity]; p.next_state = s->state[1-r->parity];
        p.out = core;
        if (g->gdn_decode_bf16_resident(g->ctx, r->request, &p)) return -1;
    } else {
        BnGPUGDNPrepareResidentPlan p = {0};
        p.n_tokens = nt; p.hk = 16; p.hv = 48; p.dk = p.dv = 128;
        p.norm_eps = 1e-6f; p.softplus_threshold = 20;
        p.a_log = w->an; p.dt_bias = w->dt; p.x = post; p.ba = ba;
        p.q = bf16_slice(r->q, (size_t)nt * 2048);
        p.k = bf16_slice(r->k, (size_t)nt * 2048);
        p.v = bf16_slice(r->v, (size_t)nt * 6144);
        p.log_decay = bf16_slice(r->log_decay, (size_t)nt * 48);
        p.beta = bf16_slice(r->beta, (size_t)nt * 48);
        if (g->gdn_prepare_bf16_resident(g->ctx, r->request, &p)) return -1;
        BnGPUGDNPrefillResidentPlan d = {0};
        d.n_tokens = nt; d.hk = 16; d.hv = 48; d.dk = d.dv = 128;
        d.q_scale = 0x1.6a09e6p-4f; d.q = p.q; d.k = p.k; d.v = p.v;
        d.log_decay = p.log_decay; d.beta = p.beta;
        d.initial_state = s->state[r->parity]; d.next_state = s->state[1-r->parity];
        d.out = core;
        if (r->arithmetic >= BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32) {
            BnGPUGDNCPPrefillResidentPlan cp = {0};
            cp.n_tokens = nt; cp.chunk_tokens = g->gdn_cp_prefill_chunk_tokens(g->ctx,nt);
            cp.hk = 16; cp.hv = 48; cp.dk = cp.dv = 128; cp.q_scale = d.q_scale;
            cp.decay_domain = BN_GPU_GDN_DECAY_LOG;
            cp.recurrence = bf16_prefill_recurrence(r,1);
            cp.q = d.q; cp.k = d.k; cp.v = d.v; cp.alpha = d.log_decay; cp.beta = d.beta;
            cp.initial_state = d.initial_state; cp.next_state = d.next_state; cp.out = d.out;
            if (g->gdn_cp_prefill_bf16_resident(g->ctx,r->request,&cp)) return -1;
        } else if (g->gdn_prefill_bf16_resident(g->ctx, r->request, &d)) return -1;
    }
    return bf16_ssm_finish(r,layer,nt,output);
}

static int bf16_attention(BF16Request *r, int layer, int nt,
    BnGPUResidentView input, BnGPUResidentView output) {
    BnGPUBackend *g = r->gpu;
    const BnBackendBF16Layer *w = &r->weights->layers[layer];
    BF16LayerState *s = &r->layers[layer];
    BnGPUQKPrepResidentPlan p = {0};
    p.n_tokens = nt; p.n_heads = 24; p.n_kv_heads = 4; p.head_size = 256;
    p.rotary_dim = 64; p.position = r->position; p.eps = 1e-6f; p.weight_offset = 1;
    p.q_weight = w->qn; p.k_weight = w->kn; p.cos_sin = r->cos_sin;
    p.q_gate = bf16_slice(r->q_gate, (size_t)nt * 12288);
    p.k_in = bf16_slice(r->k_input, (size_t)nt * 1024);
    p.v_in = bf16_slice(r->v_input, (size_t)nt * 1024);
    p.q = bf16_slice(r->q, (size_t)nt * 6144);
    p.k = bf16_slice(r->k, (size_t)nt * 1024);
    p.v = bf16_slice(r->v, (size_t)nt * 1024);
    p.gate = bf16_slice(r->g, (size_t)nt * 6144);
    if (bf16_linear(r, w->q, 12288, 5120, nt, input, p.q_gate) ||
        bf16_linear(r, w->k, 1024, 5120, nt, input, p.k_in) ||
        bf16_linear(r, w->v, 1024, 5120, nt, input, p.v_in) ||
        g->qk_prep_bf16_resident(g->ctx, r->request, &p)) return -1;
    BnGPUAttentionResidentPlan a = {0};
    a.n_tokens = nt; a.n_heads = 24; a.n_kv_heads = 4; a.head_size = 256;
    a.position = r->position; a.cache_capacity = r->capacity;
    a.gate_profile = r->arithmetic >= BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32
        ? BN_GPU_ATTENTION_GATE_COMPILED_F32 : BN_GPU_ATTENTION_GATE_BF16;
    a.gated = 1; a.scale = .0625f; a.q = p.q; a.k = p.k; a.v = p.v;
    a.gate = p.gate; a.key_cache = s->key; a.value_cache = s->value;
    a.attention_out = bf16_slice(r->attention, (size_t)nt * 6144);
    a.out = bf16_slice(r->core, (size_t)nt * 6144);
    return g->attention_bf16_resident(g->ctx, r->request, &a) ||
        bf16_linear(r, w->o, 5120, 6144, nt, a.out, output) ? -1 : 0;
}

static int bf16_observe(BF16Request *r, BnTransformerBF16Observe observe,
    void *context, BnTransformerBF16Stage stage, int layer, int nt,
    BnGPUResidentView view) {
    return observe ? observe(context, stage, layer, r->position, nt, view) : 0;
}

typedef struct {
    int count, n_outputs, n_decode;
    unsigned char decode[256];
    BF16Request *members[256];
    int offsets[256], lengths[256], chunks[256], last_rows[256];
} BF16PackedPrefill;

/* Fixed metadata slots can still be uploading from pinned host memory when a
 * later layer arrives. Retry once after an explicit coordinator fence. */
#define BF16_BATCH_GROUP(CALL) do { \
    int status=(CALL); \
    if(status==BN_GPU_GROUP_BUSY) { \
        ++*busy_fences; \
        if(g->resident_synchronize(r->request))return -1; \
        status=(CALL); \
    } \
    if(status)return -1; \
} while(0)

static int bf16_batch_ssm(BF16Request *r,BF16Request *const *members,int layer,
    int nt,BnGPUResidentView input,BnGPUResidentView output,uint64_t *busy_fences) {
    BnGPUBackend *g=r->gpu;const BnBackendBF16Layer *w=&r->weights->layers[layer];
    if(bf16_ssm_projections(r,layer,nt,input))return -1;
    BnGPUCausalConvSequence cv[256];BnGPUGDNDecodeSequence gd[256];
    for(int i=0;i<nt;i++) {
        BF16Request *m=members[i];BF16LayerState *state=&m->layers[layer];int parity=m->parity;
        cv[i].request=m->request;cv[i].x=bn_gpu_resident_view_slice(r->qkv,(size_t)i*10240,10240);
        cv[i].out=bn_gpu_resident_view_slice(r->conv_output,(size_t)i*10240,10240);
        cv[i].initial_history=state->history[parity];cv[i].next_history=state->history[1-parity];
        gd[i].request=m->request;gd[i].x=cv[i].out;gd[i].ba=bn_gpu_resident_view_slice(r->ba,(size_t)i*96,96);
        gd[i].initial_state=state->state[parity];gd[i].next_state=state->state[1-parity];
        gd[i].out=bn_gpu_resident_view_slice(r->core,(size_t)i*6144,6144);
    }
    BnGPUCausalConvGroupPlan cp={0};cp.n_sequences=nt;cp.dim=10240;cp.kernel=4;cp.weight=w->conv;cp.sequences=cv;
    BF16_BATCH_GROUP(g->causal_conv_bf16_group_resident(g->ctx,r->request,&cp));
    BnGPUGDNDecodeGroupPlan dp={0};dp.n_sequences=nt;dp.hk=16;dp.hv=48;dp.dk=dp.dv=128;
    dp.q_scale=0x1.6a09e6p-4f;dp.a_log=w->an;dp.dt_bias=w->dt;dp.sequences=gd;
    BF16_BATCH_GROUP(g->gdn_decode_bf16_group_resident(g->ctx,r->request,&dp));
    return bf16_ssm_finish(r,layer,nt,output);
}
static int bf16_packed_ssm(BF16Request *r,const BF16PackedPrefill *packed,int layer,
    int nt,BnGPUResidentView input,BnGPUResidentView output,uint64_t *busy_fences) {
    BnGPUBackend *g=r->gpu;const BnBackendBF16Layer *w=&r->weights->layers[layer];
    if(bf16_ssm_projections(r,layer,nt,input))return -1;
    BnGPUCausalConvSequence cv[256];BnGPUGDNCPPrefillSequence seq[256];
    for(int i=0;i<packed->count;i++) {
        BF16Request *m=packed->members[i];BF16LayerState *state=&m->layers[layer];
        size_t off=(size_t)packed->offsets[i],n=(size_t)packed->lengths[i];
        cv[i].request=m->request;cv[i].x=bn_gpu_resident_view_slice(r->qkv,off*10240,n*10240);
        cv[i].out=bn_gpu_resident_view_slice(r->conv_output,off*10240,n*10240);
        cv[i].initial_history=state->history[m->parity];cv[i].next_history=state->history[1-m->parity];
        seq[i].request=m->request;BnGPUGDNCPPrefillResidentPlan *p=&seq[i].plan;
        memset(p,0,sizeof(*p));p->n_tokens=(int)n;p->chunk_tokens=packed->chunks[i];
        p->hk=16;p->hv=48;p->dk=p->dv=128;p->q_scale=0x1.6a09e6p-4f;p->decay_domain=BN_GPU_GDN_DECAY_LOG;
        p->recurrence=bf16_prefill_recurrence(r,packed->count);
        p->q=bn_gpu_resident_view_slice(r->q,off*2048,n*2048);p->k=bn_gpu_resident_view_slice(r->k,off*2048,n*2048);
        p->v=bn_gpu_resident_view_slice(r->v,off*6144,n*6144);p->out=bn_gpu_resident_view_slice(r->core,off*6144,n*6144);
        p->alpha=bn_gpu_resident_view_slice(r->log_decay,off*48,n*48);p->beta=bn_gpu_resident_view_slice(r->beta,off*48,n*48);
        p->initial_state=state->state[m->parity];p->next_state=state->state[1-m->parity];
    }
    BnGPUCausalConvPrefillGroupPlan conv={{packed->count,10240,4,w->conv,cv},packed->lengths};
    BF16_BATCH_GROUP(g->causal_conv_bf16_prefill_group_resident(g->ctx,r->request,&conv));
    BnGPUGDNPrepareResidentPlan prep={0};prep.n_tokens=nt;prep.hk=16;prep.hv=48;prep.dk=prep.dv=128;
    prep.norm_eps=1e-6f;prep.softplus_threshold=20;prep.a_log=w->an;prep.dt_bias=w->dt;
    prep.x=bf16_slice(r->conv_output,(size_t)nt*10240);prep.ba=bf16_slice(r->ba,(size_t)nt*96);
    prep.q=bf16_slice(r->q,(size_t)nt*2048);prep.k=bf16_slice(r->k,(size_t)nt*2048);prep.v=bf16_slice(r->v,(size_t)nt*6144);
    prep.log_decay=bf16_slice(r->log_decay,(size_t)nt*48);prep.beta=bf16_slice(r->beta,(size_t)nt*48);
    if(g->gdn_prepare_bf16_resident(g->ctx,r->request,&prep))return -1;
    BnGPUGDNCPPrefillGroupPlan cp={packed->count,seq};
    if(g->gdn_cp_prefill_group_resident(g->ctx,r->request,&cp))return -1;
    return bf16_ssm_finish(r,layer,nt,output);
}
/* Body projections stay packed across both roles. Only recurrent operators
 * split by role; views retain caller offsets and request-local state owners. */
static int bf16_mixed_ssm(BF16Request *r,const BF16PackedPrefill *packed,int layer,
    int nt,BnGPUResidentView input,BnGPUResidentView output,uint64_t *busy_fences) {
    BnGPUBackend *g=r->gpu;const BnBackendBF16Layer *w=&r->weights->layers[layer];
    if(bf16_ssm_projections(r,layer,nt,input))return -1;
    BnGPUCausalConvSequence decode_conv[256],prefill_conv[256];
    BnGPUGDNDecodeSequence decode[256];BnGPUGDNCPPrefillSequence prefill[256];
    int lengths[256],nd=0,np=0;
    for(int i=0;i<packed->count;i++) {
        BF16Request *m=packed->members[i];BF16LayerState *state=&m->layers[layer];
        size_t off=(size_t)packed->offsets[i],n=(size_t)packed->lengths[i];
        BnGPUCausalConvSequence cv={0};cv.request=m->request;
        cv.x=bn_gpu_resident_view_slice(r->qkv,off*10240,n*10240);
        cv.out=bn_gpu_resident_view_slice(r->conv_output,off*10240,n*10240);
        cv.initial_history=state->history[m->parity];cv.next_history=state->history[1-m->parity];
        if(packed->decode[i]) {
            decode_conv[nd]=cv;BnGPUGDNDecodeSequence *d=&decode[nd++];memset(d,0,sizeof(*d));
            d->request=m->request;d->x=cv.out;d->ba=bn_gpu_resident_view_slice(r->ba,off*96,96);
            d->initial_state=state->state[m->parity];d->next_state=state->state[1-m->parity];
            d->out=bn_gpu_resident_view_slice(r->core,off*6144,6144);
        } else {
            prefill_conv[np]=cv;lengths[np]=(int)n;BnGPUGDNCPPrefillSequence *sequence=&prefill[np++];
            memset(sequence,0,sizeof(*sequence));sequence->request=m->request;
            BnGPUGDNCPPrefillResidentPlan *a=&sequence->plan;
            a->n_tokens=(int)n;a->chunk_tokens=packed->chunks[i];a->hk=16;a->hv=48;a->dk=a->dv=128;
            a->q_scale=0x1.6a09e6p-4f;a->decay_domain=BN_GPU_GDN_DECAY_LOG;
            a->q=bn_gpu_resident_view_slice(r->q,off*2048,n*2048);a->k=bn_gpu_resident_view_slice(r->k,off*2048,n*2048);
            a->v=bn_gpu_resident_view_slice(r->v,off*6144,n*6144);a->out=bn_gpu_resident_view_slice(r->core,off*6144,n*6144);
            a->alpha=bn_gpu_resident_view_slice(r->log_decay,off*48,n*48);a->beta=bn_gpu_resident_view_slice(r->beta,off*48,n*48);
            a->initial_state=state->state[m->parity];a->next_state=state->state[1-m->parity];
        }
    }
    if(nd) {
        BnGPUCausalConvGroupPlan conv={0};conv.n_sequences=nd;conv.dim=10240;conv.kernel=4;conv.weight=w->conv;conv.sequences=decode_conv;
        BF16_BATCH_GROUP(g->causal_conv_bf16_group_resident(g->ctx,r->request,&conv));
    }
    if(np) {
        for(int i=0;i<np;i++)prefill[i].plan.recurrence=bf16_prefill_recurrence(r,np);
        BnGPUCausalConvPrefillGroupPlan conv={{np,10240,4,w->conv,prefill_conv},lengths};
        BF16_BATCH_GROUP(g->causal_conv_bf16_prefill_group_resident(g->ctx,r->request,&conv));
        /* Preparing the packed span permits noncontiguous prefill rows without
         * repacking or changing projection arithmetic. Decode prep is unused. */
        BnGPUGDNPrepareResidentPlan prep={0};prep.n_tokens=nt;prep.hk=16;prep.hv=48;prep.dk=prep.dv=128;
        prep.norm_eps=1e-6f;prep.softplus_threshold=20;prep.a_log=w->an;prep.dt_bias=w->dt;
        prep.x=bf16_slice(r->conv_output,(size_t)nt*10240);prep.ba=bf16_slice(r->ba,(size_t)nt*96);
        prep.q=bf16_slice(r->q,(size_t)nt*2048);prep.k=bf16_slice(r->k,(size_t)nt*2048);prep.v=bf16_slice(r->v,(size_t)nt*6144);
        prep.log_decay=bf16_slice(r->log_decay,(size_t)nt*48);prep.beta=bf16_slice(r->beta,(size_t)nt*48);
        if(g->gdn_prepare_bf16_resident(g->ctx,r->request,&prep))return -1;
        BnGPUGDNCPPrefillGroupPlan group={np,prefill};
        if(g->gdn_cp_prefill_group_resident(g->ctx,r->request,&group))return -1;
    }
    if(nd) {
        BnGPUGDNDecodeGroupPlan group={0};group.n_sequences=nd;group.hk=16;group.hv=48;group.dk=group.dv=128;
        /* Explicit mixed recurrent roles use fused sigmoid update. Pure
         * decode groups retain their existing packed normalization contract. */
        group.profile=np?BN_GPU_GDN_DECODE_RSQRT:BN_GPU_GDN_DECODE_PACKED;
        group.q_scale=0x1.6a09e6p-4f;group.a_log=w->an;group.dt_bias=w->dt;group.sequences=decode;
        BF16_BATCH_GROUP(g->gdn_decode_bf16_group_resident(g->ctx,r->request,&group));
    }
    return bf16_ssm_finish(r,layer,nt,output);
}
static int bf16_batch_attention(BF16Request *r,BF16Request *const *members,int layer,
    int nt,BnGPUResidentView input,BnGPUResidentView output,uint64_t *busy_fences,
    const BF16PackedPrefill *packed) {
    BnGPUBackend *g=r->gpu;const BnBackendBF16Layer *w=&r->weights->layers[layer];
    BnGPUResidentView qg=bf16_slice(r->q_gate,(size_t)nt*12288),ki=bf16_slice(r->k_input,(size_t)nt*1024),vi=bf16_slice(r->v_input,(size_t)nt*1024);
    if(bf16_linear(r,w->q,12288,5120,nt,input,qg)||bf16_linear(r,w->k,1024,5120,nt,input,ki)||bf16_linear(r,w->v,1024,5120,nt,input,vi))return -1;
    BnGPUQKPrepSequence prep[256];BnGPUAttentionDecodeSequence attention[256];
    int count=packed?packed->count:nt;
    memset(prep,0,(size_t)count*sizeof(*prep));memset(attention,0,(size_t)count*sizeof(*attention));
    for(int i=0;i<count;i++) {
        size_t off=packed?(size_t)packed->offsets[i]:(size_t)i,n=packed?(size_t)packed->lengths[i]:1;
        BF16Request *m=packed?packed->members[i]:members[i];prep[i].request=m->request;BnGPUQKPrepResidentPlan *p=&prep[i].plan;
        p->n_tokens=(int)n;p->n_heads=24;p->n_kv_heads=4;p->head_size=256;p->rotary_dim=64;
        p->position=m->position;p->eps=1e-6f;p->weight_offset=1;p->q_weight=w->qn;p->k_weight=w->kn;p->cos_sin=m->cos_sin;
        p->q_gate=bn_gpu_resident_view_slice(qg,off*12288,n*12288);p->k_in=bn_gpu_resident_view_slice(ki,off*1024,n*1024);p->v_in=bn_gpu_resident_view_slice(vi,off*1024,n*1024);
        p->q=bn_gpu_resident_view_slice(r->q,off*6144,n*6144);p->k=bn_gpu_resident_view_slice(r->k,off*1024,n*1024);p->v=bn_gpu_resident_view_slice(r->v,off*1024,n*1024);p->gate=bn_gpu_resident_view_slice(r->g,off*6144,n*6144);
        attention[i].request=m->request;BnGPUAttentionResidentPlan *a=&attention[i].plan;
        a->n_tokens=(int)n;a->n_heads=24;a->n_kv_heads=4;a->head_size=256;a->position=m->position;a->cache_capacity=m->capacity;
        a->scale=.0625f;a->gated=1;a->gate_profile=r->arithmetic>=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32?BN_GPU_ATTENTION_GATE_COMPILED_F32:BN_GPU_ATTENTION_GATE_BF16;
        a->q=p->q;a->k=p->k;a->v=p->v;a->gate=p->gate;a->key_cache=m->layers[layer].key;a->value_cache=m->layers[layer].value;
        a->attention_out=bn_gpu_resident_view_slice(r->attention,off*6144,n*6144);a->out=bn_gpu_resident_view_slice(r->core,off*6144,n*6144);
    }
    BnGPUQKPrepGroupPlan qp={count,prep};BF16_BATCH_GROUP(g->qk_prep_group_bf16_resident(g->ctx,r->request,&qp));
    /* One-token prefill tails keep prefill recurrence, but their attention
     * has decode geometry and needs the same cohort partition selection. */
    int decode_shaped=!packed;
    if(packed && r->attention_split_policy==BN_GPU_ATTENTION_SPLIT_BATCH) {
        decode_shaped=1;
        for(int i=0;i<count;i++)if(packed->lengths[i]!=1){decode_shaped=0;break;}
    }
    if(!decode_shaped) {
        BnGPUAttentionPrefillGroupPlan ap={count,attention,r->attention_split_policy};BF16_BATCH_GROUP(g->attention_prefill_group_bf16_resident(g->ctx,r->request,&ap));
    } else {
        BnGPUAttentionDecodeGroupPlan ap={count,attention,r->attention_split_policy};BF16_BATCH_GROUP(g->attention_decode_group_bf16_resident(g->ctx,r->request,&ap));
    }
    return bf16_linear(r,w->o,5120,6144,nt,bf16_slice(r->core,(size_t)nt*6144),output);
}
#undef BF16_BATCH_GROUP

static int bf16_forward_graph(BF16Request *r,const int32_t *tokens,int nt,
    float *logits,size_t logits_count,BnTransformerBF16Observe observe,void *context,
    int want_logits,int prefill,BF16Request *const *members,uint64_t *busy_fences,
    const BF16PackedPrefill *packed) {
    const BnModel *model=r->model;BnGPUBackend *gpu=r->gpu;
    if (gpu->resident_write(r->request, bf16_slice(r->ids, (size_t)nt), tokens)) goto fail;
    BnGPUEmbeddingResidentPlan e = {0};
    e.n_tokens = nt; e.dim = 5120; e.vocab_size = model->config.vocab_size;
    e.weight = r->weights->embedding.raw; e.indices = bf16_slice(r->ids, (size_t)nt);
    e.out = bf16_slice(r->embedding, (size_t)nt * 5120);
    if (gpu->embedding_bf16_resident(gpu->ctx, r->request, &e)) goto fail;
    BnGPUResidentView residual = e.out;
    size_t nd = (size_t)nt * 5120;
    BnGPUResidentView norm = bf16_slice(r->norm, nd);
    BnGPUResidentView hidden = bf16_slice(r->ffn_output, nd);
    BnGPUResidentView block = bf16_slice(r->block_output, nd);
    int retained = r->arithmetic != BN_TRANSFORMER_BF16_ARITHMETIC_EAGER;
    for (int l = 0; l < 64; ++l) {
        const BnBackendBF16Layer *w = &r->weights->layers[l];
        if (!l) {
            if (bf16_norm(r, w->ni, nt, e.out, norm)) goto fail;
        } else {
            BnGPUResidentView next = bf16_slice(retained && !(l & 1)
                ? r->residual_a : r->residual_b, nd);
            if (retained) {
                if (bf16_add3_norm(r, w->ni, nt, l, block, residual, hidden, norm, next)) goto fail;
            } else if (bf16_add_norm(r, w->ni, nt, -1, hidden, residual, norm, next)) goto fail;
            residual = next;
        }
        if (bf16_observe(r, observe, context, BN_TRANSFORMER_BF16_INPUT_NORM, l, nt, norm)) goto fail;
        if (model->weights.layers[l].block_kind == BN_LAYER_BLOCK_SSM) {
            if (packed ? (packed->n_decode?bf16_mixed_ssm(r,packed,l,nt,norm,block,busy_fences):bf16_packed_ssm(r,packed,l,nt,norm,block,busy_fences)) : members ? bf16_batch_ssm(r,members,l,nt,norm,block,busy_fences) : bf16_ssm(r,l,nt,norm,block,prefill)) goto fail;
        } else if ((members || packed) ? bf16_batch_attention(r,members,l,nt,norm,block,busy_fences,packed) : bf16_attention(r,l,nt,norm,block)) goto fail;
        /* Embedding is dead after the first combined residual. Reuse it for
         * diagnostic rounded sums while the retained profile ping-pongs a/b. */
        BnGPUResidentView diagnostic = retained && l ? e.out : bf16_slice(r->residual_a, nd);
        if (bf16_add_norm(r, w->nf, nt, l, block, residual, norm, diagnostic)) goto fail;
        if (!retained) residual = diagnostic;
        if (bf16_observe(r, observe, context, BN_TRANSFORMER_BF16_FFN_NORM, l, nt, norm)) goto fail;
        BnGPUFFNResidentPlan f = {0};
        f.n_tokens = nt; f.dim = 5120; f.hidden_dim = 17408;
        f.weights = w->ffn; f.x = norm;
        f.activation_profile = r->ffn_arithmetic==BN_TRANSFORMER_BF16_FFN_COMPILED
            ?BN_GPU_BF16_FFN_SILU_COMPILED:retained
            ?BN_GPU_BF16_FFN_SILU_FUSED:BN_GPU_BF16_FFN_SILU_SEPARATE;
        f.gate = bf16_slice(r->gate, (size_t)nt * 17408);
        f.up = bf16_slice(r->up, (size_t)nt * 17408); f.out = hidden;
        if(r->ffn_arithmetic==BN_TRANSFORMER_BF16_FFN_COMPILED&&w->ffn_gate_up) {
            BnBackendBF16MatmulPlan plan;
            if(bn_backend_quant_bf16_matmul_plan(w->ffn.gate.type,17408,5120,nt,&plan))goto fail;
            if(plan.kind==BN_BACKEND_BF16_MATMUL_DECODED_GEMM) {
                f.layout=BN_GPU_BF16_FFN_STACKED_DECODED;f.gate_up=w->ffn_gate_up;
                f.gate_up_out=bf16_slice(r->gate,(size_t)nt*34816);
            }
        }
        if (gpu->dense_ffn_bf16_resident(gpu->ctx, r->request, &f) ||
            bf16_observe(r, observe, context, BN_TRANSFORMER_BF16_FFN_OUTPUT, l, nt, hidden) ||
            bf16_observe(r, observe, context, BN_TRANSFORMER_BF16_RESIDUAL, l, nt, diagnostic)) goto fail;
    }
    if (want_logits) {
        if (retained) {
            BnGPUResidentView next = bf16_slice(residual.slot == r->residual_a.slot
                ? r->residual_b : r->residual_a, nd);
            if (bf16_add3_norm(r, r->weights->final_norm, nt, 64, block, residual, hidden, norm, next)) goto fail;
        } else if (bf16_add_norm(r, r->weights->final_norm, nt, -1, hidden, residual,
                norm, bf16_slice(r->residual_b, nd))) goto fail;
        if (bf16_observe(r, observe, context, BN_TRANSFORMER_BF16_FINAL_NORM, -1, nt, norm)) goto fail;
        int output_rows=packed?packed->n_outputs:members?nt:1;
        BnGPUResidentView last = members ? norm : bn_gpu_resident_view_slice(norm, (size_t)(nt-1)*5120,5120);
        if(packed) {
            last=bf16_slice(r->embedding,(size_t)output_rows*5120);
            BnGPUGatherRowsResidentPlan gather={nt,5120,output_rows,packed->last_rows,norm,last};
            if(gpu->gather_rows_resident(gpu->ctx,r->request,&gather))goto fail;
        }
        BnGPUResidentView logits_view=bf16_slice(r->logits,(size_t)output_rows*model->config.vocab_size);
        if (bf16_linear(r, r->weights->output, model->config.vocab_size,5120,output_rows,last,logits_view) ||
            bf16_observe(r, observe, context, BN_TRANSFORMER_BF16_LOGITS, -1, packed?output_rows:nt, packed?logits_view:r->logits) ||
            gpu->resident_read(r->request,logits_view,r->host_logits)) goto fail;
        for (size_t i = 0; !members && !packed && i < logits_count; ++i) {
            uint32_t bits = (uint32_t)r->host_logits[i] << 16;
            memcpy(&logits[i], &bits, sizeof(bits));
        }
    } else if (gpu->resident_synchronize(r->request)) goto fail;
    return 0;
fail:
    r->failed = 1;
    fprintf(stderr, "BF16 forward failed; reset the session before reuse\n");
    return -1;
}

static int bf16_forward_impl(const BnModel *model, BnGPUBackend *gpu,
    BnBackendSession *session, const int32_t *tokens, int nt,
    float *logits, size_t logits_count, BnTransformerBF16Observe observe,
    void *context, int want_logits, int prefill) {
    BF16Request *r = bf16_request(session);
    if (!r || r->failed || r->model != model || r->gpu != gpu || !model ||
        r->weights != bn_backend_layout_bf16(bn_model_backend(model)) ||
        !tokens || (want_logits && (!logits ||
            logits_count != (size_t)model->config.vocab_size)) ||
        nt <= 0 || (!prefill && nt != 1) ||
        nt > r->max_tokens || nt > r->capacity - r->position) return -1;
    for (int i = 0; i < nt; ++i)
        if (tokens[i] < 0 || tokens[i] >= model->config.vocab_size) return -1;
    if(bf16_forward_graph(r,tokens,nt,logits,logits_count,observe,context,want_logits,prefill,NULL,NULL,NULL))return -1;
    r->position+=nt;r->parity=1-r->parity;return 0;
}

int bn_transformer_bf16_model_active(const BnModel *model) {
    return model && (bn_backend_layout_bf16(bn_model_backend(model)) != NULL ||
        bn_backend_layout_bf16_flash_model(bn_model_backend(model)) != NULL ||
        bn_backend_layout_bf16_glm_model(bn_model_backend(model)) != NULL);
}

int bn_transformer_bf16_forward(const BnModel *model, BnGPUBackend *gpu,
    BnBackendSession *session, const int32_t *tokens, int nt,
    float *logits, size_t logits_count, BnTransformerBF16Observe observe,
    void *context) {
    return bf16_forward_impl(model, gpu, session, tokens, nt, logits,
        logits_count, observe, context, 1, nt > 1);
}

static int bf16_session_run(BnModel *model, BnSession *session,
    const int *tokens, int nt, int position, int want_logits, int prefill) {
    if (!model || !session || !tokens) return -1;
    BnBackendSession *backend = bn_session_backend(session);
    BF16Request *r = bf16_request(backend);
    if (!r) {
        if (bn_backend_layout_bf16_glm_model(bn_model_backend(model)))
            return bf16_glm_runtime_run(model, session, tokens, nt,
                position, want_logits, prefill);
        return bf16_flash_runtime_run(model, session, tokens, nt,
            position, want_logits, prefill);
    }
    if (!r || r->failed || r->position != position || nt <= 0 || nt > r->max_tokens)
        return -1;
    for (int i = 0; i < nt; ++i) r->host_tokens[i] = tokens[i];
    return bf16_forward_impl(model, bn_model_gpu(model), backend,
        r->host_tokens, nt, want_logits ? session->state.logits : NULL,
        want_logits ? (size_t)model->config.vocab_size : 0, NULL, NULL,
        want_logits, prefill);
}

float *bn_transformer_bf16_forward_session(BnModel *model, BnSession *session,
    const int *tokens, int nt, int position, int prefill) {
    return bf16_session_run(model, session, tokens, nt, position, 1, prefill) == 0
        ? session->state.logits : NULL;
}

int bn_transformer_bf16_no_logits_session(BnModel *model, BnSession *session,
    const int *tokens, int nt, int position, int prefill) {
    return bf16_session_run(model, session, tokens, nt, position, 0, prefill);
}

typedef struct {
    BF16Request scratch;
    int max_requests,max_prefill_tokens,cp_chunk;
    uint64_t decode_calls,busy_fences,prefill_calls,mixed_calls;
} BF16Batch;
static const char bf16_batch_owner=0;
static BF16Batch *bf16_batch(const BnBackendSession *session) {
    return bn_backend_session_execution_resources(session,&bf16_batch_owner);
}
static void bf16_batch_free(void *ptr) {
    BF16Batch *b=ptr;free(b->scratch.host_logits);free(b->scratch.host_tokens);free(b);
}
int bn_transformer_bf16_policy(const BnBackendSession *session,
    BnTransformerBF16Policy *out) {
    if (!session || !out) return -1;
    const BF16Request *r=bf16_request(session);
    const BF16Batch *b=r?NULL:bf16_batch(session);
    if (b) r=&b->scratch;
    if (!r || r->failed) return -1;
    out->arithmetic=r->arithmetic;
    out->ffn_arithmetic=r->ffn_arithmetic;
    out->attention_split_policy=r->attention_split_policy;
    out->gdn_prefill_selection=r->gdn_prefill_selection;
    memcpy(out->post_norm_reductions,r->post_norm_reductions,sizeof(out->post_norm_reductions));
    memcpy(out->input_norm_reductions,r->input_norm_reductions,sizeof(out->input_norm_reductions));
    return 0;
}

int bn_transformer_bf16_batch_prepare(const BnModel *model,BnGPUBackend *g,
    BnBackendSession *coordinator,int max_requests,BnTransformerBF16Arithmetic arithmetic) {
    BnTransformerBF16Policy policy={0};policy.arithmetic=arithmetic;
    return bn_transformer_bf16_batch_prepare_with_policy(model,g,coordinator,max_requests,&policy);
}

typedef struct {const BnBackendBF16Model *weights;int max_rows,cp_chunk;} BF16BatchPlan;
static int bf16_batch_admit(const BnModel *model,BnGPUBackend *g,
    int max_requests,int max_prefill_tokens,const BnTransformerBF16Policy *policy,BF16BatchPlan *out) {
    BnTransformerBF16Arithmetic arithmetic=policy?policy->arithmetic:BN_TRANSFORMER_BF16_ARITHMETIC_EAGER;
    if(!model||!bf16_capable(g)||!bf16_policy_capable(g,policy)||max_requests<1||max_requests>256||
       arithmetic<BN_TRANSFORMER_BF16_ARITHMETIC_EAGER||arithmetic>BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32||
       (arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32&&!(g->resident_gdn_prefill_recurrences&(1u<<BN_GPU_GDN_PREFILL_NONCP)))||
       (arithmetic!=BN_TRANSFORMER_BF16_ARITHMETIC_EAGER&&!g->add3_rmsnorm_bf16_resident)||
       (arithmetic>=BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32&&!(g->resident_rmsnorm_reductions&(1u<<BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA)))||
       (arithmetic>=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32&&!(g->resident_attention_gate_profiles&(1u<<BN_GPU_ATTENTION_GATE_COMPILED_F32)))||
       (arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32_GATED_R2&&!(g->resident_gated_norm_profiles&(1u<<BN_GPU_GATED_NORM_COMPILED_R2)))||
       !g->causal_conv_bf16_group_workspace_size||!g->causal_conv_bf16_group_reserve||!g->causal_conv_bf16_group_resident||
       !g->gdn_decode_bf16_group_workspace_size||!g->gdn_decode_bf16_group_reserve||!g->gdn_decode_bf16_group_resident||
       !g->qk_prep_group_workspace_size||!g->qk_prep_group_reserve||!g->qk_prep_group_bf16_resident||
       !g->attention_decode_group_workspace_size||!g->attention_decode_group_reserve||!g->attention_decode_group_bf16_resident)return -1;
    if(max_prefill_tokens<0 || max_prefill_tokens>8192)return -1;
    int max_rows=max_requests>max_prefill_tokens?max_requests:max_prefill_tokens,cp_chunk=0;
    if(max_prefill_tokens) {
        if(arithmetic<BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32 || !g->gather_rows_resident ||
           !g->causal_conv_bf16_prefill_group_resident || !g->attention_prefill_group_bf16_resident ||
           !g->gdn_cp_prefill_group_workspace_size || !g->gdn_cp_prefill_group_reserve || !g->gdn_cp_prefill_group_resident ||
           !g->gdn_cp_prefill_chunk_tokens || !g->gdn_cp_prefill_bf16_workspace_size || !g->gdn_cp_prefill_bf16_reserve)return -1;
        cp_chunk=g->gdn_cp_prefill_chunk_tokens(g->ctx,1);
        if(!cp_chunk || !g->gdn_cp_prefill_bf16_workspace_size(g->ctx,max_prefill_tokens,cp_chunk))return -1;
    }
    const BnBackendBF16Model *weights=bn_backend_layout_bf16(bn_model_backend(model));
    if(!weights||weights->n_layers!=64||max_rows>weights->max_tokens||bn_model_gpu(model)!=g||
       model->config.bf16_layout!=BN_MODEL_BF16_LAYOUT_HYBRID_OFFSET_NORM||
       model->config.vocab_size<=0||(size_t)max_requests>SIZE_MAX/(size_t)model->config.vocab_size/sizeof(uint16_t))return -1;
    *out=(BF16BatchPlan){weights,max_rows,cp_chunk};return 0;
}

static int bf16_memory_add(size_t *total,size_t bytes) {
    if(!bytes||bytes>SIZE_MAX-*total)return -1;
    *total+=bytes;return 0;
}

int bn_transformer_bf16_batch_memory_plan(const BnModel *model,BnGPUBackend *g,
    int max_requests,int max_prefill_tokens,const BnTransformerBF16Policy *policy,
    BnTransformerBF16RequestMemory *out) {
    BF16BatchPlan plan;
    if(!out||bf16_batch_admit(model,g,max_requests,max_prefill_tokens,policy,&plan)||
       !g->linear_bf16_workspace_size||!g->linear_bf16_input_workspace_size||
       !g->embedding_bf16_workspace_size)return -1;
    BF16Request r={0};r.planning=1;r.model=model;r.weights=plan.weights;r.max_tokens=plan.max_rows;r.arithmetic=policy->arithmetic;
    bf16_policy_copy(&r,policy);
    if(bf16_alloc_activations(&r,plan.max_rows,max_requests))return -1;
    size_t host=sizeof(BF16Batch)+(size_t)plan.max_rows*sizeof(int32_t)+
        (size_t)max_requests*model->config.vocab_size*sizeof(uint16_t);
    size_t device=r.planned_bytes;
    if(bf16_memory_add(&device,g->linear_bf16_workspace_size(g->ctx))||
       bf16_memory_add(&device,g->linear_bf16_input_workspace_size(g->ctx,(size_t)plan.max_rows*17408))||
       bf16_memory_add(&device,g->embedding_bf16_workspace_size(g->ctx)))return -1;
    if(g->resident_group_queue_workspace_size&&g->resident_group_queue_reserve) {
        for(int kind=BN_GPU_RESIDENT_GROUP_GDN;kind<=BN_GPU_RESIDENT_GROUP_ATTENTION;kind++) {
            size_t bytes=g->resident_group_queue_workspace_size(g->ctx,(BnGPUResidentGroupKind)kind,max_requests,64);
            if(bf16_memory_add(&device,bytes)||bf16_memory_add(&host,bytes))return -1;
        }
    }else {
        size_t bytes[]={g->causal_conv_bf16_group_workspace_size(g->ctx,max_requests),
            g->gdn_decode_bf16_group_workspace_size(g->ctx,max_requests),
            g->qk_prep_group_workspace_size(g->ctx,max_requests),
            g->attention_decode_group_workspace_size(g->ctx,max_requests)};
        for(size_t i=0;i<sizeof(bytes)/sizeof(*bytes);i++)
            if(bf16_memory_add(&device,bytes[i])||bf16_memory_add(&host,bytes[i]))return -1;
    }
    if(max_prefill_tokens) {
        size_t bytes=g->gdn_cp_prefill_group_workspace_size(g->ctx,max_requests);
        if(bf16_memory_add(&device,bytes)||bf16_memory_add(&host,bytes)||
           bf16_memory_add(&device,g->gdn_cp_prefill_bf16_workspace_size(g->ctx,max_prefill_tokens,plan.cp_chunk)))return -1;
    }
    *out=(BnTransformerBF16RequestMemory){device,host,host};return 0;
}

static int bf16_batch_prepare(const BnModel *model,BnGPUBackend *g,
    BnBackendSession *coordinator,int max_requests,int max_prefill_tokens,const BnTransformerBF16Policy *policy) {
    if(!coordinator||bn_backend_session_has_gpu_resources(coordinator)||
       bn_backend_session_has_execution_resources(coordinator))return -1;
    BF16BatchPlan plan;
    if(bf16_batch_admit(model,g,max_requests,max_prefill_tokens,policy,&plan))return -1;
    BnTransformerBF16Arithmetic arithmetic=policy->arithmetic;
    const BnBackendBF16Model *weights=plan.weights;
    int max_rows=plan.max_rows,cp_chunk=plan.cp_chunk;
    BF16Batch *b=calloc(1,sizeof(*b));if(!b)return -1;BF16Request *r=&b->scratch;
    r->model=model;r->gpu=g;r->weights=weights;r->max_tokens=max_rows;r->arithmetic=arithmetic;
    b->max_requests=max_requests;b->max_prefill_tokens=max_prefill_tokens;b->cp_chunk=cp_chunk;
    bf16_policy_copy(r,policy);
    r->host_tokens=malloc((size_t)max_rows*sizeof(*r->host_tokens));
    r->host_logits=malloc((size_t)max_requests*model->config.vocab_size*sizeof(*r->host_logits));
    if(!r->host_tokens||!r->host_logits){bf16_batch_free(b);return -1;}
    r->request=bn_gpu_backend_resident_request(g,coordinator);if(!r->request){bf16_batch_free(b);return -1;}
    if(g->linear_bf16_reserve(g->ctx,r->request,(size_t)max_rows*17408)||g->embedding_bf16_reserve(g->ctx,r->request)||bf16_alloc_activations(r,max_rows,max_requests))goto fail;
#define RESERVE(NAME) do {size_t bytes=g->NAME##_workspace_size(g->ctx,max_requests); \
    if(!bytes||g->NAME##_reserve(g->ctx,r->request,max_requests,bytes))goto fail;}while(0)
    if(g->resident_group_queue_workspace_size&&g->resident_group_queue_reserve) {
        for(int kind=BN_GPU_RESIDENT_GROUP_GDN;kind<=BN_GPU_RESIDENT_GROUP_ATTENTION;kind++) {
            size_t bytes=g->resident_group_queue_workspace_size(g->ctx,(BnGPUResidentGroupKind)kind,max_requests,64);
            if(!bytes||g->resident_group_queue_reserve(g->ctx,r->request,(BnGPUResidentGroupKind)kind,max_requests,64,bytes))goto fail;
        }
    } else {
        RESERVE(causal_conv_bf16_group);RESERVE(gdn_decode_bf16_group);RESERVE(qk_prep_group);RESERVE(attention_decode_group);
    }
#undef RESERVE
    if(max_prefill_tokens) {
        size_t bytes=g->gdn_cp_prefill_group_workspace_size(g->ctx,max_requests);
        if(!bytes || g->gdn_cp_prefill_group_reserve(g->ctx,r->request,max_requests,bytes) ||
           g->gdn_cp_prefill_bf16_reserve(g->ctx,r->request,max_prefill_tokens,cp_chunk))goto fail;
    }
    if(bn_backend_session_attach_execution_resources(coordinator,&bf16_batch_owner,b,bf16_batch_free))goto fail;
    return 0;
fail:
    bn_backend_session_release_gpu_resources(coordinator);bf16_batch_free(b);return -1;
}
int bn_transformer_bf16_batch_prepare_with_policy(const BnModel *model,BnGPUBackend *g,
    BnBackendSession *coordinator,int max_requests,const BnTransformerBF16Policy *policy) {
    return bf16_batch_prepare(model,g,coordinator,max_requests,0,policy);
}
int bn_transformer_bf16_batch_prepare_packed_with_policy(const BnModel *model,BnGPUBackend *g,
    BnBackendSession *coordinator,int max_requests,int max_prefill_tokens,const BnTransformerBF16Policy *policy) {
    if(max_prefill_tokens<1)return -1;
    return bf16_batch_prepare(model,g,coordinator,max_requests,max_prefill_tokens,policy);
}
int bn_transformer_bf16_batch_info(const BnBackendSession *coordinator,BnTransformerBF16BatchInfo *out) {
    const BF16Batch *b=bf16_batch(coordinator);if(!b||!out)return -1;
    memset(out,0,sizeof(*out));out->mixed_calls=b->mixed_calls;
    out->capacity=b->max_requests;out->max_prefill_tokens=b->max_prefill_tokens;out->prefill_calls=b->prefill_calls;out->failed=b->scratch.failed;
    out->decode_calls=b->decode_calls;out->busy_fences=b->busy_fences;return 0;
}
int bn_transformer_bf16_batch_decode(const BnModel *model,BnGPUBackend *g,
    BnBackendSession *coordinator,const BnTransformerBF16DecodeRow *rows,int count) {
    BF16Batch *b=bf16_batch(coordinator);if(!b||!model||!g||!rows||count<1||count>b->max_requests)return -1;
    BF16Request *r=&b->scratch;if(r->failed||r->model!=model||r->gpu!=g||r->weights!=bn_backend_layout_bf16(bn_model_backend(model)))return -1;
    BF16Request *members[256];float *outputs[256];size_t vocab=(size_t)model->config.vocab_size,bytes=vocab*sizeof(float);
    /* Validate the entire cohort before queueing or publishing any output. */
    for(int i=0;i<count;i++) {
        const BnTransformerBF16DecodeRow *row=&rows[i];BF16Request *m=bf16_request(row->session);
        if(!m||m->failed||m->model!=model||m->gpu!=g||m->weights!=r->weights||m->arithmetic!=r->arithmetic||
           m->gdn_prefill_selection!=r->gdn_prefill_selection||m->ffn_arithmetic!=r->ffn_arithmetic||m->attention_split_policy!=r->attention_split_policy||
           memcmp(m->post_norm_reductions,r->post_norm_reductions,sizeof(r->post_norm_reductions))||
           memcmp(m->input_norm_reductions,r->input_norm_reductions,sizeof(r->input_norm_reductions))||
           m->position!=row->position||m->position>=m->capacity||row->token<0||row->token>=model->config.vocab_size||
           !row->logits||row->logits_count!=vocab||(uintptr_t)row->logits>UINTPTR_MAX-bytes)return -1;
        for(int j=0;j<i;j++)if(members[j]==m||
            ((uintptr_t)row->logits<(uintptr_t)outputs[j]+bytes&&(uintptr_t)outputs[j]<(uintptr_t)row->logits+bytes))return -1;
        members[i]=m;outputs[i]=row->logits;r->host_tokens[i]=row->token;
    }
    if(bf16_forward_graph(r,r->host_tokens,count,NULL,0,NULL,NULL,1,0,members,&b->busy_fences,NULL))goto fail;
    for(int i=0;i<count;i++)if(g->resident_synchronize(members[i]->request))goto fail;
    if(g->resident_synchronize(r->request))goto fail;
    for(int i=0;i<count;i++)for(size_t j=0;j<vocab;j++) {
        uint32_t bits=(uint32_t)r->host_logits[(size_t)i*vocab+j]<<16;memcpy(outputs[i]+j,&bits,sizeof(bits));
    }
    for(int i=0;i<count;i++){members[i]->position++;members[i]->parity=1-members[i]->parity;}
    b->decode_calls++;return 0;
fail:
    /* Device writes may already have occurred. Never expose reusable requests
     * with uncommitted positions or partially updated recurrent/KV storage. */
    (void)g->resident_synchronize(r->request);r->failed=1;
    for(int i=0;i<count;i++){(void)g->resident_synchronize(members[i]->request);members[i]->failed=1;}
    fprintf(stderr,"BF16 batch decode failed; reset coordinator and participants\n");return -1;
}

static int bf16_batch_packed_run(const BnModel *model,BnGPUBackend *g,
    BnBackendSession *coordinator,const BnTransformerBF16PrefillRow *rows,int count,
    const unsigned char *decode,BnTransformerBF16Observe observe,void *context) {
    BF16Batch *b=bf16_batch(coordinator);
    if(!b || !model || !g || !rows || count<1 || count>b->max_requests || !b->max_prefill_tokens)return -1;
    BF16Request *r=&b->scratch;
    if(r->failed || r->model!=model || r->gpu!=g || r->weights!=bn_backend_layout_bf16(bn_model_backend(model)))return -1;
    BF16PackedPrefill packed={0};float *outputs[256];int total=0;
    size_t vocab=(size_t)model->config.vocab_size,bytes=vocab*sizeof(float);
    packed.count=count;
    for(int i=0;i<count;i++) {
        const BnTransformerBF16PrefillRow *row=&rows[i];BF16Request *m=bf16_request(row->session);
        if(!m || m->failed || m->model!=model || m->gpu!=g || m->weights!=r->weights || m->arithmetic!=r->arithmetic ||
           m->gdn_prefill_selection!=r->gdn_prefill_selection||m->ffn_arithmetic!=r->ffn_arithmetic || m->attention_split_policy!=r->attention_split_policy || memcmp(m->post_norm_reductions,r->post_norm_reductions,sizeof(r->post_norm_reductions)) ||
           memcmp(m->input_norm_reductions,r->input_norm_reductions,sizeof(r->input_norm_reductions)) ||
           !row->tokens || row->n_tokens<1 || row->n_tokens>m->max_tokens || m->position!=row->position ||
           row->n_tokens>m->capacity-m->position || row->n_tokens>b->max_prefill_tokens-total ||
           (row->logits?(row->logits_count!=vocab || (uintptr_t)row->logits>UINTPTR_MAX-bytes):row->logits_count!=0))return -1;
        for(int j=0;j<i;j++)if(packed.members[j]==m)return -1;
        for(int j=0;row->logits && j<packed.n_outputs;j++)if((uintptr_t)row->logits<(uintptr_t)outputs[j]+bytes &&
            (uintptr_t)outputs[j]<(uintptr_t)row->logits+bytes)return -1;
        int chunk=g->gdn_cp_prefill_chunk_tokens(g->ctx,row->n_tokens);if(chunk<b->cp_chunk)return -1;
        packed.decode[i]=decode?decode[i]:0;packed.n_decode+=packed.decode[i]!=0;
        packed.members[i]=m;packed.offsets[i]=total;packed.lengths[i]=row->n_tokens;packed.chunks[i]=chunk;
        for(int j=0;j<row->n_tokens;j++) {
            int id=row->tokens[j];if(id<0 || id>=model->config.vocab_size)return -1;
            r->host_tokens[total+j]=id;
        }
        total+=row->n_tokens;
        if(row->logits){packed.last_rows[packed.n_outputs]=total-1;outputs[packed.n_outputs++]=row->logits;}
    }
    if(packed.n_decode&&packed.n_decode<count&&
       !(g->resident_gdn_decode_profiles&(1u<<BN_GPU_GDN_DECODE_RSQRT)))return -1;
    r->position=-1; /* Packed observers receive rows in caller sequence order. */
    if(bf16_forward_graph(r,r->host_tokens,total,NULL,0,observe,context,packed.n_outputs!=0,1,NULL,&b->busy_fences,&packed))goto fail;
    for(int i=0;i<count;i++)if(g->resident_synchronize(packed.members[i]->request))goto fail;
    if(g->resident_synchronize(r->request))goto fail;
    for(int i=0;i<packed.n_outputs;i++)for(size_t j=0;j<vocab;j++) {
        uint32_t bits=(uint32_t)r->host_logits[(size_t)i*vocab+j]<<16;memcpy(outputs[i]+j,&bits,sizeof(bits));
    }
    for(int i=0;i<count;i++){packed.members[i]->position+=packed.lengths[i];packed.members[i]->parity=1-packed.members[i]->parity;}
    r->position=0;
    if(!packed.n_decode)b->prefill_calls++;else if(packed.n_decode==count)b->decode_calls++;else b->mixed_calls++;
    return 0;
fail:
    (void)g->resident_synchronize(r->request);r->failed=1;
    for(int i=0;i<count;i++){(void)g->resident_synchronize(packed.members[i]->request);packed.members[i]->failed=1;}
    fprintf(stderr,"BF16 packed prefill failed; reset coordinator and participants\n");return -1;
}

int bn_transformer_bf16_batch_prefill(const BnModel *model,BnGPUBackend *gpu,
    BnBackendSession *coordinator,const BnTransformerBF16PrefillRow *rows,int count,
    BnTransformerBF16Observe observe,void *context) {
    return bf16_batch_packed_run(model,gpu,coordinator,rows,count,NULL,observe,context);
}
int bn_transformer_bf16_batch_step(const BnModel *model,BnGPUBackend *gpu,
    BnBackendSession *coordinator,const BnTransformerBF16StepRow *rows,int count,
    BnTransformerBF16Observe observe,void *context) {
    if(!rows||count<1||count>256)return -1;
    BnTransformerBF16PrefillRow packed[256];unsigned char decode[256];
    for(int i=0;i<count;i++) {
        if(rows[i].kind!=BN_TRANSFORMER_BF16_STEP_PREFILL&&rows[i].kind!=BN_TRANSFORMER_BF16_STEP_DECODE)return -1;
        packed[i]=rows[i].row;decode[i]=rows[i].kind==BN_TRANSFORMER_BF16_STEP_DECODE;
        if(decode[i]&&(packed[i].n_tokens!=1||!packed[i].logits))return -1;
    }
    return bf16_batch_packed_run(model,gpu,coordinator,packed,count,decode,observe,context);
}

/* PLE request state shares the generic resident-request ownership contract.
 * A partial Flash operator remains distinct from dense model admission. */
typedef struct {
    BnGPUResidentView key, value, compressed, history[2];
} BF16FlashQSALayer;

typedef struct {
    const BnBackendBF16FlashFFNModel *weights;
    BnTransformerBF16FlashFFNViews v;
    BnGPUResidentView metadata, cursors;
    int32_t offsets[520];
    int pending, layer, next_expert, ready;
    int cache_started;
    BnGPUExpertCacheLayer cache_layer;
    int positions[64];
    unsigned bank;
    uint64_t generation;
    size_t submitted_blocks, metadata_read_bytes;
    double metadata_ms, source_ms, bank_retry_ms, wait_start;
    BnMoEState source;
} BF16FlashFFNRequest;

typedef struct {
    const BnBackendBF16FlashGDNModel *weights;
    const BnBackendBF16HCModel *hc_weights;
    uint64_t hc_attention_done, expected_ffn, submitted_ffn;
    int hc_open, hc_layer, hc_branch, hc_required;
    BnGPUResidentView hc_residual, hc_combined;
    BF16FlashFFNRequest ffn;
    const BnBackendBF16FlashIOModel *io_weights;
    BnGPUResidentView token_ids, logits;
    int tokens_submitted, head_submitted;
    int32_t *model_tokens;
    uint16_t *host_logits;
    int model_running, model_layer, model_phase, model_logits;
    const BnBackendBF16FlashQSAModel *qsa_weights;
    uint64_t qsa_layers;
    int capacity, qsa_pages, qsa_groups;
    size_t qsa_workspace_bytes, host_peak;
    BF16FlashQSALayer qsa[64];
    BnGPUResidentView index_projected, index_q, scores, indices, slots, table, requests, qsa_workspace;
    uint64_t valid_layers, expected_layers, submitted_layers;
    int active, position, n_tokens, prefill, includes_ple, lease_valid;
    BnGPUHostUploadLease lease;
    int positions[64], parity[64];
    BF16LayerState layers[64];
    BnGPUResidentView input, projected, ba, qkv, z, post, q, k, v;
    BnGPUResidentView log_decay, beta, core, norm, out;
} BF16FlashRequest;

typedef struct {
    const BnModel *model;
    BnGPUBackend *gpu;
    const BnBackendBF16PLEWeights *weights;
    void *request;
    BnTransformerBF16PLEInfo info;
    int parity, next_slot, last_tokens;
    int32_t context[8], next_context[8];
    int32_t *ids;
    size_t stride, history_words;
    BnGPUResidentView packed, hidden, embedding, merged, key, value;
    BnGPUResidentView kn, qn, gated, normed, out, history[2];
    BF16FlashRequest *flash;
} BF16PLERequest;

static const char bf16_ple_request_owner = 0;

static BF16PLERequest *bf16_ple_request(const BnBackendSession *session) {
    return bn_backend_session_execution_resources(session, &bf16_ple_request_owner);
}

static void bf16_ple_free(void *ptr) {
    BF16PLERequest *r = ptr;
    free(r->ids);
    if(r->flash) {
        free(r->flash->model_tokens);
        free(r->flash->host_logits);
    }
    free(r->flash);
    free(r); /* GPU resources already finished work and released pinned slots. */
}

static int bf16_ple_current(const BF16PLERequest *r) {
    return r && !r->info.failed && r->weights ==
        bn_backend_layout_bf16_ple_model(bn_model_backend(r->model)) &&
        (!r->flash || (r->flash->weights ==
            bn_backend_layout_bf16_flash_gdn_model(bn_model_backend(r->model)) &&
            (!r->flash->qsa_weights || r->flash->qsa_weights ==
             bn_backend_layout_bf16_flash_qsa_model(bn_model_backend(r->model))) &&
            r->flash->hc_weights == bn_backend_layout_bf16_hc_model(bn_model_backend(r->model)) &&
            (!r->flash->ffn.weights || r->flash->ffn.weights ==
             bn_backend_layout_bf16_flash_ffn_model(bn_model_backend(r->model))) &&
            (!r->flash->io_weights || r->flash->io_weights ==
             bn_backend_layout_bf16_flash_io_model(bn_model_backend(r->model)))));
}

static int bf16_ple_alloc(BF16PLERequest *r, BnGPUResidentView *v, int type, size_t count) {
    return r->gpu->resident_alloc(r->request, type, count, v);
}

int bn_transformer_bf16_ple_info(const BnBackendSession *session,
    BnTransformerBF16PLEInfo *out) {
    const BF16PLERequest *r = bf16_ple_request(session);
    if (!r || !out) return -1;
    *out = r->info;
    return 0;
}

int bn_transformer_bf16_ple_views(const BnBackendSession *session, int nt,
    BnTransformerBF16PLEViews *out) {
    const BF16PLERequest *r = bf16_ple_request(session);
    if (!bf16_ple_current(r) || !out || nt <= 0 || nt > r->info.max_tokens) return -1;
    size_t n = (size_t)nt * r->weights->dim, wide = n * r->weights->streams;
    BnTransformerBF16PLEViews v = {0};
    v.hidden = bf16_slice(r->hidden, wide); v.embedding = bf16_slice(r->embedding, n);
    v.key = bf16_slice(r->key, wide); v.value = bf16_slice(r->value, n);
    v.gated = bf16_slice(r->gated, wide); v.normed = bf16_slice(r->normed, wide);
    v.out = bf16_slice(r->out, wide); v.history = r->history[r->parity];
    v.row_ids = r->last_tokens ? r->ids : NULL;
    v.row_id_count = (size_t)r->last_tokens * r->model->config.ple_head_count;
    *out = v;
    return 0;
}

typedef struct {
    const BnBackendBF16FlashQSAModel *weights;
    int capacity, pages, groups;
    uint64_t layers;
    size_t tensors, workspace, host_staging;
} BF16FlashQSAPlan;

static int bf16_flash_add_bytes(size_t *bytes,size_t count,size_t width) {
    if(width&&count>(SIZE_MAX-*bytes)/width)return -1;
    *bytes+=count*width;return 0;
}
static int bf16_flash_qsa_admit(const BnModel *model,BnGPUBackend *g,int nt,int capacity,
    const BnBackendBF16FlashGDNModel *gdn,BF16FlashQSAPlan *out) {
    const BnBackendBF16FlashQSAModel *w=bn_backend_layout_bf16_flash_qsa_model(bn_model_backend(model));
    BF16FlashQSAPlan p={0};
    if(!w){*out=p;return 0;}
    if(w->n_layers!=gdn->n_layers||!w->layers||w->dim!=2560||w->heads!=24||w->kv_heads!=2||
       w->head_size!=256||w->index_heads!=4||w->index_head_size!=128||w->compress_ratio!=4||
       w->rotary_dim!=64||w->norm_eps!=1e-6f||!w->rotary_cache||capacity>w->rotary_capacity||
       capacity<=0||capacity>262144||!g->qk_prep_bf16_resident||!g->qsa_indexer_bf16_resident||
       !g->qsa_score_bf16_resident||!g->qsa_select_resident||!g->paged_kv_append_bf16_resident||
       !g->qsa_attention_bf16_prepare||!g->qsa_attention_bf16_workspace_size||
       !g->qsa_attention_bf16_resident||!g->resident_write)return -1;
    int count=0;
    for(int l=0;l<w->n_layers;l++) {
        int expected=model->weights.layers[l].block_kind==BN_LAYER_BLOCK_ATTENTION;
        if(!!w->layers[l].valid!=expected||(expected&&gdn->layers[l].valid))return -1;
        if(expected){p.layers|=UINT64_C(1)<<l;count++;}
    }
    if(!count||count!=w->n_qsa_layers||g->qsa_attention_bf16_workspace_size(g->ctx,nt,&p.workspace))return -1;
    p.weights=w;p.capacity=capacity;p.pages=(capacity+15)/16;p.groups=(capacity+3)/4;
    size_t per_layer=(size_t)p.pages*32768+(size_t)p.groups*256+2048;
    size_t per_token=18708+(size_t)p.groups*4;
    if(bf16_flash_add_bytes(&p.tensors,(size_t)count,per_layer)||
       bf16_flash_add_bytes(&p.tensors,(size_t)nt,per_token)||
       bf16_flash_add_bytes(&p.tensors,(size_t)p.pages,4)||
       bf16_flash_add_bytes(&p.tensors,(size_t)capacity,4))return -1;
    p.host_staging=(size_t)capacity*sizeof(int32_t);
    *out=p;return 0;
}

static int bf16_flash_allocate(BF16PLERequest *r);

typedef struct {
    const BnBackendBF16FlashFFNModel *weights;
    size_t tensors, workspace, host;
} BF16FlashFFNPlan;

static int bf16_flash_ffn_admit(const BnModel *model, BnGPUBackend *g, int nt,
    BF16FlashFFNPlan *out) {
    BF16FlashFFNPlan p = {0};
    p.weights = bn_backend_layout_bf16_flash_ffn_model(bn_model_backend(model));
    if (!p.weights) { *out = p; return 0; }
    const BnBackendBF16FlashFFNModel *w = p.weights;
    const BnMoEIO *io = bn_model_moe_io_const(model);
    if (w->n_layers != model->config.n_layers || w->n_layers <= 0 || w->n_layers > 64 ||
        w->dim != 2560 || w->hidden_dim != 640 || w->shared_hidden_dim != 640 ||
        w->n_experts != 512 || w->k != 10 || !w->renormalize ||
        w->shared_activation != BN_GPU_BF16_FFN_SILU_FUSED || !w->layers ||
        nt > INT_MAX / 10 / 2560 || !io ||
        (!bn_moe_io_has_mmap(io) && !bn_moe_io_has_pread(io)) ||
        !g->moe_topk_bf16_resident || !g->moe_group_bf16_resident || !g->moe_reduce_bf16_resident ||
        !g->moe_shared_bf16_resident || !g->dense_ffn_bf16_resident ||
        !g->moe_bank_reserve || !g->moe_bank_acquire || !g->moe_bank_cancel ||
        !g->moe_bank_submit || !g->moe_bank_stats || !g->moe_bank_input_reserve ||
        !g->moe_bank_grouped_reserve || !g->moe_bank_prepare_input ||
        !g->linear_a8_bf16_reserve || !g->resident_read) return -1;
    /* Two cap-8 slots, worst-case Q5_K gate/up + Q8 down; descriptor payload
     * is two slots * eight items * 12 bytes. A8 blocks carry 32 values/36 bytes. */
    p.host = 63897600u + 192u + sizeof(((BF16FlashFFNRequest *)0)->offsets);
    p.workspace = 63897600u + 192u; p.tensors = 4128u;
    if (w->expert_cache) {
        if (!g->expert_cache_bind || !g->expert_cache_begin || !g->expert_cache_end ||
            !g->expert_cache_cancel || !g->expert_cache_stats) return -1;
        p.host += 16u*sizeof(void*) + sizeof(BnGPUExpertCacheLayer);
        p.workspace += 16u*sizeof(void*);
    }
    if (bf16_flash_add_bytes(&p.workspace,(size_t)nt,31680u+7200u+2880u) ||
        bf16_flash_add_bytes(&p.tensors,(size_t)nt,198348u)) return -1;
    *out = p; return 0;
}

typedef struct {
    const BnBackendBF16PLEWeights *weights;
    const BnBackendBF16FlashGDNModel *gdn;
    const BnBackendBF16FlashIOModel *io;
    BF16FlashQSAPlan qsa;BF16FlashFFNPlan ffn;
    size_t stride,history_words,tensors,workspace,id_bytes,staging,extra_host;
    uint64_t valid_layers;
} BF16PLEPlan;
static int bf16_ple_admit(const BnModel *model,BnGPUBackend *g,int max_tokens,
    size_t gpu_budget,size_t host_budget,int flash,int max_context,BF16PLEPlan *out) {
    if (!model || !g || !out ||
        !bn_gpu_backend_can_resident_request(g) || !g->resident_zero ||
        !g->resident_upload_reserve || !g->resident_upload_acquire ||
        !g->resident_upload_submit || !g->resident_upload_cancel ||
        !g->linear_bf16_workspace_size || !g->linear_bf16_reserve || !g->linear_bf16_resident ||
        !g->quant_rows_bf16_resident || !g->split_columns_resident ||
        !g->ple_gate_bf16_resident || !g->ple_conv_bf16_resident) return -1;
    const BnConfig *c = &model->config;
    const BnBackendBF16PLEWeights *w = bn_backend_layout_bf16_ple_model(bn_model_backend(model));
    const BnQWeight *table = &model->weights.ple_token_embd;
    if (!w || w->gpu != g || w->dim != c->dim || w->streams != c->hyper_connection_count ||
        w->layer != c->ple_layer || w->kernel != c->ple_conv_kernel || w->dilation != c->ple_ngram_size ||
        max_tokens <= 0 || max_tokens > c->seq_len || c->vocab_size <= 0 ||
        bn_model_transformer_policy_norm_epsilon(c) != 1e-6f ||
        c->ple_ngram_size < 2 || c->ple_ngram_size > 8 || c->ple_heads_per_ngram <= 0 ||
        c->ple_head_count <= 0 || c->ple_head_count > 64 ||
        c->ple_heads_per_ngram > 64 / (c->ple_ngram_size - 1) ||
        c->ple_head_count != (c->ple_ngram_size - 1) * c->ple_heads_per_ngram ||
        c->ple_head_dim <= 0 || c->ple_head_dim > c->dim ||
        c->ple_head_count * c->ple_head_dim != c->dim ||
        c->ple_eos_token_id < 0 || c->ple_eos_token_id >= c->vocab_size ||
        !bn_backend_quant_host_rows_bf16_available(table) || table->cols != c->ple_head_dim ||
        table->rows <= 0) return -1;
    for (int i = 0; i < c->ple_head_count; i++)
        if (!c->ple_head_vocab_sizes[i] ||
            (uint64_t)c->ple_head_offsets[i] + c->ple_head_vocab_sizes[i] > (uint64_t)table->rows)
            return -1;
    int wide = w->dim * w->streams;
    if (max_tokens > INT_MAX / (wide + w->dim)) return -1;
    size_t stride = bn_backend_quant_host_row_stride(table);
    size_t h = (size_t)(w->kernel - 1) * w->dilation * wide;
    size_t row_bytes = (size_t)c->ple_head_count * stride;
    size_t per_token = (8u * wide + 3u * w->dim) * 2u;
    if (!stride || row_bytes > SIZE_MAX - per_token) return -1;
    per_token += row_bytes;
    if (h > SIZE_MAX / 4 || (size_t)max_tokens > (SIZE_MAX - h * 4) / per_token) return -1;
    size_t tensors = (size_t)max_tokens * per_token + h * 4;
    size_t workspace = g->linear_bf16_workspace_size(g->ctx);
    const BnBackendBF16FlashGDNModel *fw = NULL;
    BF16FlashQSAPlan qsa = {0};
    BF16FlashFFNPlan ffn = {0};
    const BnBackendBF16FlashIOModel *io = NULL;
    uint64_t valid_layers = 0;
    if (flash) {
        if (max_context < max_tokens || max_context > c->seq_len) return -1;
        fw = bn_backend_layout_bf16_flash_gdn_model(bn_model_backend(model));
        if (!fw || fw->n_layers != c->n_layers || fw->n_layers <= 0 || fw->n_layers > 64 ||
            fw->dim != 2560 || fw->hk != 16 || fw->hv != 48 || fw->dk != 128 ||
            fw->dv != 128 || fw->kernel != 4 || fw->norm_eps != 1e-6f ||
            fw->gate_activation != BN_GPU_GATED_NORM_SIGMOID ||
            fw->math_profile != BN_BACKEND_BF16_MATH_MKL_VML_HA || !fw->layers ||
            !g->gdn_prefill_bf16_workspace_size || !g->gdn_prefill_bf16_reserve ||
            !g->gdn_prefill_bf16_resident || !g->gdn_prepare_bf16_resident ||
            !g->gdn_decode_bf16_resident || !g->causal_conv_bf16_resident ||
            !g->gated_rmsnorm_bf16_resident || max_tokens > INT_MAX / 16384) return -1;
        int count = 0;
        for (int l = 0; l < fw->n_layers; ++l) {
            if (!!fw->layers[l].valid !=
                (model->weights.layers[l].block_kind == BN_LAYER_BLOCK_SSM)) return -1;
            if (fw->layers[l].valid) { valid_layers |= UINT64_C(1) << l; ++count; }
        }
        if (!count || count != fw->n_gdn_layers) return -1;
        /* Double F32 recurrent state + double BF16 chronological history for
         * each layer. Fourteen scratch buffers are shared across all layers. */
        size_t banks = (size_t)count * (48u * 128 * 128 * 8 + 3u * 10240 * 4);
        if ((size_t)max_tokens > (SIZE_MAX - banks) / 141888) return -1;
        size_t extra = banks + (size_t)max_tokens * 141888;
        size_t gdn_workspace = g->gdn_prefill_bf16_workspace_size(g->ctx, max_tokens);
        if (!gdn_workspace || extra > SIZE_MAX - tensors ||
            gdn_workspace > SIZE_MAX - workspace) return -1;
        tensors += extra; workspace += gdn_workspace;
        if (bf16_flash_qsa_admit(model,g,max_tokens,max_context,fw,&qsa) ||
            qsa.tensors > SIZE_MAX-tensors || qsa.workspace > SIZE_MAX-workspace) return -1;
        tensors += qsa.tensors; workspace += qsa.workspace;
        if (bf16_flash_ffn_admit(model,g,max_tokens,&ffn) ||
            ffn.tensors > SIZE_MAX-tensors || ffn.workspace > SIZE_MAX-workspace) return -1;
        tensors += ffn.tensors; workspace += ffn.workspace;
        io=bn_backend_layout_bf16_flash_io_model(bn_model_backend(model));
        if(io) {
            if(!ffn.weights||io->dim!=w->dim||io->vocab_size!=c->vocab_size||
               !g->embedding_bf16_reserve||!g->embedding_bf16_resident||!g->join_columns_resident||
               !g->rmsnorm_bf16_resident||!g->scaled_silu_bf16_resident||
               !g->hyper_connection_mix_bf16_resident||
               bf16_flash_add_bytes(&tensors,(size_t)max_tokens,4)||
               bf16_flash_add_bytes(&tensors,(size_t)c->vocab_size,2)||
               bf16_flash_add_bytes(&workspace,1,4))return -1;
        }
        valid_layers |= qsa.layers;
    }
    size_t id_bytes = (size_t)max_tokens * c->ple_head_count * sizeof(int32_t);
    if ((size_t)max_tokens > (SIZE_MAX - id_bytes) / 2 / row_bytes ||
        workspace > SIZE_MAX - tensors || tensors + workspace > gpu_budget) return -1;
    size_t staging = (size_t)max_tokens * row_bytes * 2;
    size_t io_host = 0;
    if (io && (bf16_flash_add_bytes(&io_host, (size_t)max_tokens, sizeof(int32_t)) ||
        bf16_flash_add_bytes(&io_host, (size_t)c->vocab_size, sizeof(uint16_t)))) return -1;
    if (io_host>SIZE_MAX-ffn.host) return -1;
    size_t extra_host=ffn.host+io_host;
    if (staging + id_bytes > host_budget || extra_host > host_budget-staging-id_bytes ||
        qsa.host_staging > host_budget-staging-id_bytes-extra_host) return -1;
    *out=(BF16PLEPlan){.weights=w,.gdn=fw,.io=io,.qsa=qsa,.ffn=ffn,
        .stride=stride,.history_words=h,.tensors=tensors,.workspace=workspace,
        .id_bytes=id_bytes,.staging=staging,.extra_host=extra_host,.valid_layers=valid_layers};
    return 0;
}
int bn_transformer_bf16_flash_memory_plan(const BnModel *model,BnGPUBackend *gpu,
    int max_tokens,int capacity,BnTransformerBF16RequestMemory *out) {
    BF16PLEPlan p;
    if(!out||bf16_ple_admit(model,gpu,max_tokens,SIZE_MAX,SIZE_MAX,1,capacity,&p))return -1;
    *out=(BnTransformerBF16RequestMemory){p.tensors+p.workspace,
        p.staging+p.id_bytes+p.extra_host,p.staging+p.id_bytes+p.extra_host+p.qsa.host_staging};return 0;
}
static int bf16_ple_prepare(const BnModel *model,BnGPUBackend *g,
    BnBackendSession *session,int max_tokens,size_t gpu_budget,size_t host_budget,
    int flash,int max_context) {
    if(!session||bn_backend_session_has_gpu_resources(session)||bn_backend_session_has_execution_resources(session))return -1;
    BF16PLEPlan p;if(bf16_ple_admit(model,g,max_tokens,gpu_budget,host_budget,flash,max_context,&p))return -1;
    const BnConfig *c=&model->config;const BnBackendBF16PLEWeights *w=p.weights;
    const BnBackendBF16FlashGDNModel *fw=p.gdn;const BnBackendBF16FlashIOModel *io=p.io;
    BF16FlashQSAPlan qsa=p.qsa;BF16FlashFFNPlan ffn=p.ffn;
    size_t stride=p.stride,h=p.history_words,tensors=p.tensors,workspace=p.workspace;
    size_t id_bytes=p.id_bytes,staging=p.staging,extra_host=p.extra_host;uint64_t valid_layers=p.valid_layers;
    BF16PLERequest *r = calloc(1, sizeof(*r));
    if (!r) return -1;
    r->ids = malloc(id_bytes);
    if (!r->ids) { bf16_ple_free(r); return -1; }
    r->model = model; r->gpu = g; r->weights = w; r->stride = stride; r->history_words = h;
    if (flash) {
        r->flash = calloc(1, sizeof(*r->flash));
        if (!r->flash) { bf16_ple_free(r); return -1; }
        r->flash->weights = fw; r->flash->valid_layers = valid_layers;
        r->flash->hc_weights = bn_backend_layout_bf16_hc_model(bn_model_backend(model));
        r->flash->qsa_weights = qsa.weights; r->flash->qsa_layers = qsa.layers;
        r->flash->ffn.weights = ffn.weights;
        r->flash->io_weights = io;
        if(io) {
            r->flash->model_tokens=malloc((size_t)max_tokens*sizeof(int32_t));
            r->flash->host_logits=malloc((size_t)c->vocab_size*sizeof(uint16_t));
            if(!r->flash->model_tokens||!r->flash->host_logits){bf16_ple_free(r);return -1;}
        }
        r->flash->capacity = max_context; r->flash->qsa_pages = qsa.pages; r->flash->qsa_groups = qsa.groups;
        r->flash->qsa_workspace_bytes = qsa.workspace;
        r->flash->host_peak = staging + id_bytes + qsa.host_staging + extra_host;
    }
    r->info.max_tokens = max_tokens; r->info.tensor_bytes = tensors;
    r->info.workspace_bytes = workspace; r->info.gpu_bytes = tensors + workspace;
    r->info.host_bytes = staging + id_bytes + extra_host; r->info.staging_bytes = staging;
    r->request = bn_gpu_backend_resident_request(g, session);
    if (!r->request) { bf16_ple_free(r); return -1; }
    size_t n = (size_t)max_tokens * w->dim, nw = n * w->streams;
    if (g->linear_bf16_reserve(g->ctx, r->request, 0) ||
        g->resident_upload_reserve(r->request, 2, staging / 2, staging) ||
        bf16_ple_alloc(r, &r->packed, bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32), staging / 2 / 4) ||
        bf16_ple_alloc(r, &r->embedding, BF16, n) ||
        bf16_ple_alloc(r, &r->merged, BF16, nw + n) ||
        bf16_ple_alloc(r, &r->value, BF16, n) ||
        bf16_ple_alloc(r, &r->hidden, BF16, nw) ||
        bf16_ple_alloc(r, &r->key, BF16, nw) ||
        bf16_ple_alloc(r, &r->kn, BF16, nw) ||
        bf16_ple_alloc(r, &r->qn, BF16, nw) ||
        bf16_ple_alloc(r, &r->gated, BF16, nw) ||
        bf16_ple_alloc(r, &r->normed, BF16, nw) ||
        bf16_ple_alloc(r, &r->out, BF16, nw) ||
        bf16_ple_alloc(r, &r->history[0], BF16, h) ||
        bf16_ple_alloc(r, &r->history[1], BF16, h) ||
        g->resident_zero(r->request, r->history[0]) ||
        (flash && bf16_flash_allocate(r)) || g->resident_synchronize(r->request)) goto fail;
    for (int i = 0; i < c->ple_ngram_size - 1; i++) r->context[i] = c->ple_eos_token_id;
    if (bn_backend_session_attach_execution_resources(session, &bf16_ple_request_owner,
            r, bf16_ple_free)) goto fail;
    return 0;
fail:
    bn_backend_session_release_gpu_resources(session);
    bf16_ple_free(r);
    return -1;
}

int bn_transformer_bf16_ple_prepare(const BnModel *model, BnGPUBackend *g,
    BnBackendSession *session, int max_tokens, size_t gpu_budget, size_t host_budget) {
    return bf16_ple_prepare(model, g, session, max_tokens, gpu_budget, host_budget, 0, 0);
}

int bn_transformer_bf16_flash_prepare(const BnModel *model, BnGPUBackend *g,
    BnBackendSession *session, int max_tokens, size_t gpu_budget, size_t host_budget) {
    return bf16_ple_prepare(model, g, session, max_tokens, gpu_budget, host_budget, 1,
        model ? model->config.seq_len : 0);
}

int bn_transformer_bf16_flash_prepare_context(const BnModel *model, BnGPUBackend *g,
    BnBackendSession *session, int max_tokens, int max_context, size_t gpu_budget, size_t host_budget) {
    return bf16_ple_prepare(model,g,session,max_tokens,gpu_budget,host_budget,1,max_context);
}

int bn_transformer_bf16_ple_reset(BnBackendSession *session, int position,
    const int32_t *context, size_t context_count, const uint16_t *history, size_t history_words) {
    BF16PLERequest *r = bf16_ple_request(session);
    if (!bf16_ple_current(r) || r->info.pending_tokens || (r->flash && r->flash->active) || position < 0 ||
        position > (r->flash ? r->flash->capacity : r->model->config.seq_len)) return -1;
    int ctx = r->model->config.ple_ngram_size - 1;
    if ((context ? context_count != (size_t)ctx : context_count != 0) ||
        (history ? history_words != r->history_words : history_words != 0)) return -1;
    for (int i = 0; context && i < ctx; i++)
        if (context[i] < 0 || context[i] >= r->model->config.vocab_size) return -1;
    r->info.failed = 1;
    if ((history ? r->gpu->resident_write(r->request, r->history[0], history) :
            r->gpu->resident_zero(r->request, r->history[0])) ||
        r->gpu->resident_synchronize(r->request)) return -1;
    r->parity = r->next_slot = r->last_tokens = 0;
    r->info.position = position;
    for (int i = 0; i < ctx; i++) r->context[i] = context ? context[i] : r->model->config.ple_eos_token_id;
    r->info.failed = 0;
    return 0;
}

int bn_transformer_bf16_ple_submit(BnBackendSession *session,
    const int32_t *tokens, int nt, int position) {
    BF16PLERequest *r = bf16_ple_request(session);
    if (!bf16_ple_current(r) || !tokens || nt <= 0 || nt > r->info.max_tokens ||
        r->info.pending_tokens || position != r->info.position ||
        nt > r->model->config.seq_len - position) return -1;
    if (r->flash && (r->flash->hc_open || !r->flash->active || !r->flash->includes_ple ||
        !r->flash->lease_valid || nt != r->flash->n_tokens || position != r->flash->position)) return -1;
    const BnConfig *c = &r->model->config;
    for (int i = 0; i < nt; i++) if (tokens[i] < 0 || tokens[i] >= c->vocab_size) return -1;
    BnGPUHostUploadLease lease = {0};
    int status = 0;
    if (r->flash) lease = r->flash->lease;
    else status = r->gpu->resident_upload_acquire(r->request, (unsigned)r->next_slot, &lease);
    if (status) { if (status < 0) r->info.failed = 1; return status; }
    int32_t starts[] = {0, nt};
    if (bn_transformer_prefill_ple_ids(c, tokens, nt, starts, 1, r->context,
            (size_t)c->ple_ngram_size - 1, r->ids, (size_t)nt * c->ple_head_count) ||
        bn_backend_quant_gather_host_rows(&r->model->weights.ple_token_embd, r->ids,
            (size_t)nt * c->ple_head_count, lease.data, (size_t)nt * c->ple_head_count * r->stride)) {
        if (!r->flash && r->gpu->resident_upload_cancel(r->request, &lease)) r->info.failed = 1;
        return -1;
    }
    int dim = r->weights->dim, wide = dim * r->weights->streams;
    size_t n = (size_t)nt * dim, nw = n * r->weights->streams;
    BnGPUQuantRowsBF16ResidentPlan rows = {0};
    rows.rows = nt * c->ple_head_count; rows.cols = c->ple_head_dim;
    rows.type = r->model->weights.ple_token_embd.type; rows.row_stride_bytes = r->stride;
    rows.packed = bf16_slice(r->packed, (size_t)nt * c->ple_head_count * r->stride / 4);
    rows.out = bf16_slice(r->embedding, n);
    r->info.failed = 1;
    if (r->gpu->resident_upload_submit(r->request, &lease, rows.packed)) {
        r->gpu->resident_upload_cancel(r->request, &lease);
        return -1;
    }
    if (r->flash) r->flash->lease_valid = 0;
    if (r->gpu->quant_rows_bf16_resident(r->gpu->ctx, r->request, &rows)) return -1;
    BnGPUBF16LinearResidentPlan linear = {0};
    linear.n_tokens = nt; linear.rows = wide + dim; linear.cols = dim;
    linear.weight = r->weights->kv; linear.x = rows.out; linear.out = bf16_slice(r->merged, nw + n);
    if (r->gpu->linear_bf16_resident(r->gpu->ctx, r->request, &linear)) return -1;
    BnGPUSplitColumnsResidentPlan split = {0};
    split.rows = nt; split.cols = wide + dim; split.left_cols = wide; split.x = linear.out;
    split.left = bf16_slice(r->key, nw); split.right = bf16_slice(r->value, n);
    if (r->gpu->split_columns_resident(r->gpu->ctx, r->request, &split)) return -1;
    BnGPUPLEGateResidentPlan gate = {0};
    gate.n_tokens = nt; gate.dim = dim; gate.streams = r->weights->streams; gate.eps = 1e-6f;
    gate.norm_key = r->weights->norm_key; gate.norm_query = r->weights->norm_query;
    gate.norm_conv = r->weights->norm_conv; gate.key = split.left; gate.value = split.right;
    gate.hidden = bf16_slice(r->hidden, nw); gate.normalized_key = bf16_slice(r->kn, nw);
    gate.normalized_query = bf16_slice(r->qn, nw); gate.gated = bf16_slice(r->gated, nw);
    gate.normed = bf16_slice(r->normed, nw);
    if (r->gpu->ple_gate_bf16_resident(r->gpu->ctx, r->request, &gate)) return -1;
    BnGPUPLEConvResidentPlan conv = {0};
    conv.n_tokens = nt; conv.dim = wide; conv.kernel = r->weights->kernel;
    conv.dilation = r->weights->dilation; conv.weight = r->weights->conv;
    conv.x = gate.normed; conv.residual = gate.gated; conv.outer = gate.hidden;
    conv.initial_history = r->history[r->parity]; conv.next_history = r->history[1-r->parity];
    conv.out = bf16_slice(r->out, nw);
    if (r->gpu->ple_conv_bf16_resident(r->gpu->ctx, r->request, &conv)) return -1;
    int ctx = c->ple_ngram_size - 1;
    for (int i = 0; i < ctx; i++) {
        int j = nt - ctx + i;
        r->next_context[i] = j < 0 ? r->context[j + ctx] : tokens[j];
    }
    r->last_tokens = nt; r->next_slot = 1-r->next_slot;
    r->info.pending_tokens = nt;
    r->info.failed = 0;
    return 0;
}

int bn_transformer_bf16_ple_complete(BnBackendSession *session) {
    BF16PLERequest *r = bf16_ple_request(session);
    if (!bf16_ple_current(r) || r->flash || !r->info.pending_tokens) return -1;
    if (r->gpu->resident_synchronize(r->request)) { r->info.failed = 1; return -1; }
    r->parity = 1-r->parity;
    r->info.position += r->info.pending_tokens;
    r->info.pending_tokens = 0;
    memcpy(r->context, r->next_context,
        (size_t)(r->model->config.ple_ngram_size - 1) * sizeof(int32_t));
    return 0;
}

static int bf16_flash_qsa_allocate(BF16PLERequest *r) {
    BF16FlashRequest *f=r->flash;BnGPUBackend *g=r->gpu;
    if(!f->qsa_weights)return 0;
    if(g->qsa_attention_bf16_prepare(g->ctx))return -1;
    size_t nt=(size_t)r->info.max_tokens;
    int i32=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32);
    int f32=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32);
    if(bf16_ple_alloc(r,&f->index_projected,BF16,nt*640)||
       bf16_ple_alloc(r,&f->index_q,BF16,nt*512)||
       bf16_ple_alloc(r,&f->scores,f32,nt*f->qsa_groups)||
       bf16_ple_alloc(r,&f->indices,i32,nt*2052)||
       bf16_ple_alloc(r,&f->qsa_workspace,f32,f->qsa_workspace_bytes/sizeof(float))||
       bf16_ple_alloc(r,&f->slots,i32,(size_t)f->capacity)||
       bf16_ple_alloc(r,&f->table,i32,(size_t)f->qsa_pages)||
       bf16_ple_alloc(r,&f->requests,i32,nt)||g->resident_zero(r->request,f->requests))return -1;
    int32_t *map=malloc((size_t)f->capacity*sizeof(*map));if(!map)return -1;
    for(int i=0;i<f->capacity;i++)map[i]=i;
    int status=g->resident_write(r->request,f->slots,map)||g->resident_write(r->request,f->table,map);
    free(map);if(status)return -1;
    for(int l=0;l<f->qsa_weights->n_layers;l++)if(f->qsa_weights->layers[l].valid) {
        BF16FlashQSALayer *s=&f->qsa[l];
        if(bf16_ple_alloc(r,&s->key,BF16,(size_t)f->qsa_pages*8192)||
           bf16_ple_alloc(r,&s->value,BF16,(size_t)f->qsa_pages*8192)||
           bf16_ple_alloc(r,&s->compressed,BF16,(size_t)f->qsa_groups*128)||
           bf16_ple_alloc(r,&s->history[0],BF16,512)||bf16_ple_alloc(r,&s->history[1],BF16,512)||
           g->resident_zero(r->request,s->key)||g->resident_zero(r->request,s->value)||
           g->resident_zero(r->request,s->compressed)||g->resident_zero(r->request,s->history[0]))return -1;
    }
    return 0;
}

static int bf16_flash_ffn_allocate(BF16PLERequest *r) {
    BF16FlashFFNRequest *f = &r->flash->ffn;
    if (!f->weights) return 0;
    BnGPUBackend *g = r->gpu; size_t n = (size_t)r->info.max_tokens;
    BnTransformerBF16FlashFFNViews *v = &f->v;
    int i32 = bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32);
    int f32 = bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32);
#define FFN_ALLOC(field,type,count) if (bf16_ple_alloc(r,&v->field,type,count)) return -1
    FFN_ALLOC(logits,BF16,n*512); FFN_ALLOC(weights,f32,n*10);
    FFN_ALLOC(indices,i32,n*10); FFN_ALLOC(source_rows,i32,n*10);
    FFN_ALLOC(gathered,BF16,n*10*2560); FFN_ALLOC(slots,i32,n*10); FFN_ALLOC(inverse,i32,n*10);
    FFN_ALLOC(gate,BF16,n*10*640); FFN_ALLOC(up,BF16,n*10*640);
    FFN_ALLOC(down,BF16,n*10*2560); FFN_ALLOC(weighted,BF16,n*10*2560); FFN_ALLOC(routed,BF16,n*2560);
    FFN_ALLOC(shared_gate,BF16,n*640); FFN_ALLOC(shared_up,BF16,n*640); FFN_ALLOC(shared_down,BF16,n*2560);
    FFN_ALLOC(shared_logits,BF16,n); FFN_ALLOC(sigmoid,BF16,n); FFN_ALLOC(shared,BF16,n*2560);
#undef FFN_ALLOC
    if (bf16_ple_alloc(r,&f->metadata,i32,520) || bf16_ple_alloc(r,&f->cursors,i32,512) ||
        g->resident_zero(r->request,f->metadata)) return -1;
    BnGPUMoEBankConfig c = {2,8,2560,640,63897600u,63897600u,0,0};
    if (g->linear_a8_bf16_reserve(g->ctx,r->request,n*2560) ||
        g->moe_bank_reserve(g->ctx,r->request,&c) ||
        g->moe_bank_input_reserve(g->ctx,r->request,r->info.max_tokens,n*31680) ||
        g->moe_bank_grouped_reserve(g->ctx,r->request,r->info.max_tokens,192,n*7200+192)) return -1;
    return f->weights->expert_cache ? g->expert_cache_bind(r->request,f->weights->expert_cache,
        16u*sizeof(void*),16u*sizeof(void*)) : 0;
}

static int bf16_flash_allocate(BF16PLERequest *r) {
    BF16FlashRequest *f = r->flash;
    BnGPUBackend *g = r->gpu;
    size_t nt = (size_t)r->info.max_tokens;
    BnGPUResidentView *views[] = {&f->input, &f->projected, &f->ba, &f->qkv, &f->z,
        &f->post, &f->q, &f->k, &f->v, &f->core, &f->norm, &f->out};
    const size_t widths[] = {2560, 16384, 96, 10240, 6144, 10240, f->qsa_weights ? 6144u : 2048u, 2048,
        6144, 6144, 6144, 2560};
    if (g->gdn_prefill_bf16_reserve(g->ctx, r->request, r->info.max_tokens)) return -1;
    for (unsigned i = 0; i < sizeof(views) / sizeof(views[0]); ++i)
        if (bf16_ple_alloc(r, views[i], BF16, nt * widths[i])) return -1;
    int f32 = bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32);
    if (bf16_ple_alloc(r, &f->log_decay, f32, nt * 48) ||
        bf16_ple_alloc(r, &f->beta, f32, nt * 48)) return -1;
    for (int l = 0; l < f->weights->n_layers; ++l) {
        if (!f->weights->layers[l].valid) continue;
        BF16LayerState *s = &f->layers[l];
        for (int bank = 0; bank < 2; ++bank)
            if (bf16_ple_alloc(r, &s->state[bank], f32, 48u * 128 * 128) ||
                bf16_ple_alloc(r, &s->history[bank], BF16, 3u * 10240)) return -1;
        if (g->resident_zero(r->request, s->state[0]) ||
            g->resident_zero(r->request, s->history[0])) return -1;
    }
    if(bf16_flash_qsa_allocate(r) || bf16_flash_ffn_allocate(r))return -1;
    if(f->io_weights && (bf16_ple_alloc(r,&f->token_ids,bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32),nt)||
        bf16_ple_alloc(r,&f->logits,BF16,(size_t)f->io_weights->vocab_size)||
        g->embedding_bf16_reserve(g->ctx,r->request)))return -1;
    return 0;
}

int bn_transformer_bf16_flash_info(const BnBackendSession *session,
    BnTransformerBF16FlashInfo *out) {
    const BF16PLERequest *r = bf16_ple_request(session);
    if (!r || !r->flash || !out) return -1;
    const BF16FlashRequest *f = r->flash;
    BnTransformerBF16FlashInfo info = {0};
    info.max_tokens = r->info.max_tokens; info.failed = r->info.failed;
    info.active = f->active; info.position = f->position; info.n_tokens = f->n_tokens;
    info.prefill = f->prefill; info.includes_ple = f->includes_ple;
    info.valid_layers = f->valid_layers; info.expected_layers = f->expected_layers;
    info.submitted_layers = f->submitted_layers; info.tensor_bytes = r->info.tensor_bytes;
    info.workspace_bytes = r->info.workspace_bytes; info.gpu_bytes = r->info.gpu_bytes;
    info.host_bytes = r->info.host_bytes;
    info.max_context = f->capacity; info.preparation_host_peak_bytes = f->host_peak;
    info.valid_qsa_layers = f->qsa_layers; info.valid_gdn_layers = f->valid_layers & ~f->qsa_layers;
    info.expected_ffn = f->expected_ffn; info.submitted_ffn = f->submitted_ffn;
    info.hc_open = f->hc_open; info.hc_attention_done = f->hc_attention_done;
    *out = info;
    return 0;
}

int bn_transformer_bf16_flash_gdn_views(const BnBackendSession *session,
    int layer, int nt, BnTransformerBF16FlashGDNViews *out) {
    const BF16PLERequest *r = bf16_ple_request(session);
    if (!bf16_ple_current(r) || !r->flash || !out || layer < 0 ||
        layer >= r->flash->weights->n_layers || nt <= 0 || nt > r->info.max_tokens ||
        !r->flash->weights->layers[layer].valid) return -1;
    const BF16FlashRequest *f = r->flash;
    BnTransformerBF16FlashGDNViews v = {0};
    v.position = f->positions[layer];
    v.input = bf16_slice(f->input, (size_t)nt * 2560);
    v.qkvz = bf16_slice(f->projected, (size_t)nt * 16384);
    v.ba = bf16_slice(f->ba, (size_t)nt * 96);
    v.post = bf16_slice(f->post, (size_t)nt * 10240);
    v.core = bf16_slice(f->core, (size_t)nt * 6144);
    v.norm = bf16_slice(f->norm, (size_t)nt * 6144);
    v.out = bf16_slice(f->out, (size_t)nt * 2560);
    v.history = f->layers[layer].history[f->parity[layer]];
    v.state = f->layers[layer].state[f->parity[layer]];
    *out = v;
    return 0;
}

int bn_transformer_bf16_flash_qsa_views(const BnBackendSession *session,int layer,int nt,
    BnTransformerBF16FlashQSAViews *out) {
    const BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->qsa_weights||!out||layer<0||
       layer>=r->flash->qsa_weights->n_layers||nt<=0||nt>r->info.max_tokens||
       !(r->flash->qsa_layers&(UINT64_C(1)<<layer)))return -1;
    const BF16FlashRequest *f=r->flash;const BF16FlashQSALayer *s=&f->qsa[layer];
    size_t n=(size_t)nt;BnTransformerBF16FlashQSAViews v={0};
    v.position=f->positions[layer];v.max_context=f->capacity;v.pages=f->qsa_pages;v.groups=f->qsa_groups;
    v.input=bf16_slice(f->input,n*2560);v.projected=bf16_slice(f->projected,n*13312);
    v.q=bf16_slice(f->q,n*6144);v.k=bf16_slice(f->k,n*512);v.v=bf16_slice(f->v,n*512);
    v.gate=bf16_slice(f->norm,n*6144);v.index_projected=bf16_slice(f->index_projected,n*640);
    v.index_q=bf16_slice(f->index_q,n*512);v.scores=bf16_slice(f->scores,n*f->qsa_groups);
    v.indices=bf16_slice(f->indices,n*2052);v.attention=bf16_slice(f->core,n*6144);
    v.out=bf16_slice(f->out,n*2560);v.key_cache=s->key;v.value_cache=s->value;
    v.compressed=s->compressed;v.history=s->history[f->parity[layer]];*out=v;return 0;
}
int bn_transformer_bf16_flash_qsa_submit(BnBackendSession *session,int layer) {
    BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->active||!r->flash->qsa_weights||layer<0||
       layer>=r->flash->qsa_weights->n_layers)return -1;
    BF16FlashRequest *f=r->flash;uint64_t bit=UINT64_C(1)<<layer;
    if(!(f->expected_layers&bit)||(f->submitted_layers&bit)||!(f->qsa_layers&bit)||
       (f->hc_required&&!f->hc_open)||(f->hc_open&&(f->hc_layer!=layer||f->hc_branch)))return -1;
    BnGPUBackend *g=r->gpu;const BnBackendBF16FlashQSAModel *w=f->qsa_weights;
    const BnBackendBF16FlashQSAWeights *lw=&w->layers[layer];BF16FlashQSALayer *s=&f->qsa[layer];
    int nt=f->n_tokens,pos=f->position;size_t n=(size_t)nt;
    BnGPUResidentView merged=bf16_slice(f->projected,n*13312);
    BnGPUBF16LinearResidentPlan lp={0};lp.n_tokens=nt;lp.rows=13312;lp.cols=2560;
    lp.weight=lw->qkv;lp.x=bf16_slice(f->input,n*2560);lp.out=merged;
    r->info.failed=1;
    if(g->linear_bf16_resident(g->ctx,r->request,&lp))return -1;
    BnGPUQKPrepResidentPlan q={0};q.n_tokens=nt;q.n_heads=24;q.n_kv_heads=2;q.head_size=256;
    q.rotary_dim=64;q.position=pos;q.eps=w->norm_eps;q.weight_offset=1;
    q.q_weight=lw->q_norm;q.k_weight=lw->k_norm;q.cos_sin=w->rotary_cache;
    q.q_gate_row_stride=q.k_in_row_stride=q.v_in_row_stride=13312;
    q.q_gate=bn_gpu_resident_view_slice(merged,0,(n-1)*13312+12288);
    q.k_in=bn_gpu_resident_view_slice(merged,12288,(n-1)*13312+512);
    q.v_in=bn_gpu_resident_view_slice(merged,12800,(n-1)*13312+512);
    q.q=bf16_slice(f->q,n*6144);q.k=bf16_slice(f->k,n*512);q.v=bf16_slice(f->v,n*512);q.gate=bf16_slice(f->norm,n*6144);
    if(g->qk_prep_bf16_resident(g->ctx,r->request,&q))return -1;
    lp.rows=640;lp.weight=lw->index;lp.out=bf16_slice(f->index_projected,n*640);
    if(g->linear_bf16_resident(g->ctx,r->request,&lp))return -1;
    BnGPUQSAIndexerResidentPlan ix={0};ix.n_tokens=nt;ix.position=pos;ix.eps=w->norm_eps;
    ix.q_weight=lw->index_q_norm;ix.k_weight=lw->index_k_norm;ix.cos_sin=w->rotary_cache;
    ix.projected=lp.out;ix.initial_history=s->history[f->parity[layer]];ix.next_history=s->history[1-f->parity[layer]];
    ix.q=bf16_slice(f->index_q,n*512);int groups=(pos+nt)/4-pos/4;
    if(groups)ix.compressed=bn_gpu_resident_view_slice(s->compressed,(size_t)(pos/4)*128,(size_t)groups*128);
    if(g->qsa_indexer_bf16_resident(g->ctx,r->request,&ix))return -1;
    BnGPUQSAScoreResidentPlan sc={nt,pos,f->qsa_groups,ix.q,s->compressed,bf16_slice(f->scores,n*f->qsa_groups)};
    if(g->qsa_score_bf16_resident(g->ctx,r->request,&sc))return -1;
    BnGPUQSASelectResidentPlan se={nt,pos,f->qsa_groups,sc.scores,bf16_slice(f->indices,n*2052)};
    if(g->qsa_select_resident(g->ctx,r->request,&se))return -1;
    BnGPUPagedKVAppendResidentPlan ap={nt,f->qsa_pages,16,512,q.k,q.v,
        bn_gpu_resident_view_slice(f->slots,(size_t)pos,n),s->key,s->value};
    if(g->paged_kv_append_bf16_resident(g->ctx,r->request,&ap))return -1;
    BnGPUQSAAttentionResidentPlan a={0};a.n_tokens=nt;a.use_prefill_config=f->prefill;
    a.num_cache_blocks=f->qsa_pages;a.num_requests=1;a.table_width=f->qsa_pages;
    a.q=q.q;a.k_cache=s->key;a.v_cache=s->value;a.indices=se.out;a.block_table=f->table;
    a.token_to_req=bf16_slice(f->requests,n);a.gate=q.gate;a.out=bf16_slice(f->core,n*6144);a.workspace=f->qsa_workspace;
    if(g->qsa_attention_bf16_resident(g->ctx,r->request,&a))return -1;
    lp.rows=2560;lp.cols=6144;lp.weight=lw->out;lp.x=a.out;lp.out=bf16_slice(f->out,n*2560);
    if(g->linear_bf16_resident(g->ctx,r->request,&lp))return -1;
    f->submitted_layers|=bit;r->info.failed=0;return 0;
}

static int bf16_flash_begin(BnBackendSession *session, int nt,
    int position, int prefill, uint64_t layers, uint64_t ffn, int include_ple, int hc) {
    BF16PLERequest *r = bf16_ple_request(session);
    if (!bf16_ple_current(r) || !r->flash || r->flash->active || r->info.pending_tokens ||
        nt <= 0 || nt > r->info.max_tokens || position < 0 ||
        position > r->flash->capacity || nt > r->flash->capacity - position ||
        (prefill != 0 && prefill != 1) || (!prefill && nt != 1) ||
        (include_ple != 0 && include_ple != 1) || (!layers && !ffn && !include_ple) ||
        (layers & ~r->flash->valid_layers) || (include_ple && position != r->info.position)) return -1;
    BF16FlashRequest *f = r->flash;
    int count = f->weights->n_layers;
    uint64_t all = count == 64 ? UINT64_MAX : (UINT64_C(1) << count)-1;
    if ((ffn & ~all) || (hc && (!f->hc_weights || f->hc_weights->n_layers != count ||
        f->hc_weights->dim != r->weights->dim || f->hc_weights->streams != r->weights->streams ||
        f->hc_weights->rank <= 0 || f->hc_weights->rank > r->weights->dim ||
        !r->gpu->rmsnorm_bf16_resident || !r->gpu->scaled_silu_bf16_resident ||
        !r->gpu->hyper_connection_mix_bf16_resident || !r->gpu->hyper_connection_combine_bf16_resident))) return -1;
    for (int l = 0; l < f->weights->n_layers; ++l)
        if (((layers & (UINT64_C(1) << l)) && f->positions[l] != position) ||
            (f->ffn.weights && (ffn & (UINT64_C(1) << l)) && f->ffn.positions[l] != position)) return -1;
    BnGPUHostUploadLease lease = {0};
    if (include_ple) {
        int status = r->gpu->resident_upload_acquire(r->request, (unsigned)r->next_slot, &lease);
        if (status) { if (status < 0) r->info.failed = 1; return status; }
    }
    f->lease = lease; f->lease_valid = include_ple;
    f->active = 1; f->position = position; f->n_tokens = nt; f->prefill = prefill;
    f->includes_ple = include_ple; f->expected_layers = layers; f->submitted_layers = 0;
    f->expected_ffn = ffn; f->submitted_ffn = f->hc_attention_done = 0;
    f->hc_open = 0; f->hc_required = hc;
    f->ffn.ready = 0;
    f->tokens_submitted=f->head_submitted=0;
    f->hc_residual = f->hc_combined = (BnGPUResidentView){0};
    return 0;
}

int bn_transformer_bf16_flash_begin(BnBackendSession *session, int nt,
    int position, int prefill, uint64_t layers, int include_ple) {
    return bf16_flash_begin(session,nt,position,prefill,layers,0,include_ple,0);
}
int bn_transformer_bf16_flash_begin_blocks(BnBackendSession *session, int nt,
    int position, int prefill, uint64_t attention, uint64_t ffn, int include_ple) {
    return bf16_flash_begin(session,nt,position,prefill,attention,ffn,include_ple,1);
}

int bn_transformer_bf16_flash_gdn_submit(BnBackendSession *session, int layer) {
    BF16PLERequest *r = bf16_ple_request(session);
    if (!bf16_ple_current(r) || !r->flash || !r->flash->active || layer < 0 ||
        layer >= r->flash->weights->n_layers) return -1;
    BF16FlashRequest *f = r->flash;
    uint64_t bit = UINT64_C(1) << layer;
    if (!(f->expected_layers & bit) || (f->submitted_layers & bit) || !f->weights->layers[layer].valid ||
        (f->hc_required && !f->hc_open) || (f->hc_open && (f->hc_layer != layer || f->hc_branch))) return -1;
    BnGPUBackend *g = r->gpu;
    const BnBackendBF16FlashGDNModel *w = f->weights;
    const BnBackendBF16FlashGDNWeights *lw = &w->layers[layer];
    int nt = f->n_tokens, parity = f->parity[layer];
    size_t n = (size_t)nt;
    BnGPUBF16LinearResidentPlan linear = {0};
    linear.n_tokens = nt; linear.rows = 16384; linear.cols = w->dim;
    linear.weight = lw->qkvz; linear.x = bf16_slice(f->input, n * w->dim);
    linear.out = bf16_slice(f->projected, n * 16384);
    r->info.failed = 1;
    if (g->linear_bf16_resident(g->ctx, r->request, &linear)) return -1;
    linear.rows = w->hv * 2; linear.weight = lw->ba; linear.out = bf16_slice(f->ba, n * 96);
    if (g->linear_bf16_resident(g->ctx, r->request, &linear)) return -1;
    BnGPUSplitColumnsResidentPlan split = {0};
    split.rows = nt; split.cols = 16384; split.left_cols = 10240;
    split.x = bf16_slice(f->projected, n * 16384);
    split.left = bf16_slice(f->qkv, n * 10240); split.right = bf16_slice(f->z, n * 6144);
    if (g->split_columns_resident(g->ctx, r->request, &split)) return -1;
    BnGPUCausalConvResidentPlan cv = {0};
    cv.n_tokens = nt; cv.dim = 10240; cv.kernel = w->kernel; cv.weight = lw->conv;
    cv.x = split.left; cv.initial_history = f->layers[layer].history[parity];
    cv.next_history = f->layers[layer].history[1-parity]; cv.out = bf16_slice(f->post, n * 10240);
    if (g->causal_conv_bf16_resident(g->ctx, r->request, &cv)) return -1;
    BnGPUResidentView core = bf16_slice(f->core, n * 6144);
    if (f->prefill) {
        BnGPUGDNPrepareResidentPlan p = {0};
        p.n_tokens = nt; p.hk = w->hk; p.hv = w->hv; p.dk = w->dk; p.dv = w->dv;
        p.norm_eps = w->norm_eps; p.softplus_threshold = 20.0f;
        p.a_log = lw->a_log; p.dt_bias = lw->dt_bias; p.x = cv.out;
        p.ba = bf16_slice(f->ba, n * 96); p.q = bf16_slice(f->q, n * 2048);
        p.k = bf16_slice(f->k, n * 2048); p.v = bf16_slice(f->v, n * 6144);
        p.log_decay = bf16_slice(f->log_decay, n * 48); p.beta = bf16_slice(f->beta, n * 48);
        if (g->gdn_prepare_bf16_resident(g->ctx, r->request, &p)) return -1;
        BnGPUGDNPrefillResidentPlan d = {0};
        d.n_tokens = nt; d.hk = w->hk; d.hv = w->hv; d.dk = w->dk; d.dv = w->dv;
        d.output_profile = BN_GPU_GDN_OUTPUT_HISTORY_FMA;
        d.q_scale = 0x1.6a09e6p-4f; d.q = p.q; d.k = p.k; d.v = p.v;
        d.log_decay = p.log_decay; d.beta = p.beta;
        d.initial_state = f->layers[layer].state[parity];
        d.next_state = f->layers[layer].state[1-parity]; d.out = core;
        if (g->gdn_prefill_bf16_resident(g->ctx, r->request, &d)) return -1;
    } else {
        BnGPUGDNDecodeResidentPlan d = {0};
        d.hk = w->hk; d.hv = w->hv; d.dk = w->dk; d.dv = w->dv;
        d.q_scale = 0x1.6a09e6p-4f; d.a_log = lw->a_log; d.dt_bias = lw->dt_bias;
        d.x = cv.out; d.ba = bf16_slice(f->ba, 96);
        d.initial_state = f->layers[layer].state[parity];
        d.next_state = f->layers[layer].state[1-parity]; d.out = core;
        if (g->gdn_decode_bf16_resident(g->ctx, r->request, &d)) return -1;
    }
    BnGPUGatedRMSNormResidentPlan gn = {0};
    gn.rows = nt * w->hv; gn.dim = w->dv; gn.eps = w->norm_eps;
    gn.activation = w->gate_activation; gn.weight = lw->norm;
    gn.x = core; gn.z = split.right; gn.out = bf16_slice(f->norm, n * 6144);
    if (g->gated_rmsnorm_bf16_resident(g->ctx, r->request, &gn)) return -1;
    linear.rows = w->dim; linear.cols = w->hv * w->dv; linear.weight = lw->out;
    linear.x = gn.out; linear.out = bf16_slice(f->out, n * w->dim);
    if (g->linear_bf16_resident(g->ctx, r->request, &linear)) return -1;
    f->submitted_layers |= bit;
    r->info.failed = 0;
    return 0;
}

int bn_transformer_bf16_flash_hc_views(const BnBackendSession *session,int nt,
    BnTransformerBF16FlashHCViews *out) {
    const BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->hc_weights||!out||nt<=0||nt>r->info.max_tokens)return -1;
    const BF16FlashRequest *f=r->flash;const BnBackendBF16HCModel *w=f->hc_weights;
    if(w->dim!=r->weights->dim||w->streams!=r->weights->streams||w->rank<=0||w->rank>w->dim)return -1;
    size_t n=(size_t)nt,wide=n*w->dim*w->streams;BnTransformerBF16FlashHCViews v={0};
    v.open=f->hc_open;v.layer=f->hc_layer;v.ffn=f->hc_branch;
    v.hidden=bf16_slice(r->hidden,wide);v.normalized=bf16_slice(r->kn,wide);
    v.down=bf16_slice(r->value,n*w->rank);v.scaled=bf16_slice(r->embedding,n*w->rank);
    v.gate=bf16_slice(r->qn,wide);v.mixed=bf16_slice(f->input,n*w->dim);
    v.inject=bf16_slice(r->key,n*w->streams);v.block=bf16_slice(f->out,n*w->dim);
    v.residual=f->hc_residual.request?f->hc_residual:v.hidden;
    v.combined=f->hc_combined.request?f->hc_combined:bf16_slice(r->out,wide);*out=v;return 0;
}
static int bf16_flash_hc_linear(BF16PLERequest *r,BnGPUBF16LinearWeight weight,
    int rows,int cols,BnGPUResidentView x,BnGPUResidentView out) {
    BnGPUBF16LinearResidentPlan p={0};p.n_tokens=r->flash->n_tokens;p.rows=rows;p.cols=cols;
    p.weight=weight;p.x=x;p.out=out;return r->gpu->linear_bf16_resident(r->gpu->ctx,r->request,&p);
}
static int bf16_flash_hc_mix_weights(BF16PLERequest *r,const BnBackendBF16HCModel *model,
    const BnBackendBF16HCWeights *w,const BnTransformerBF16FlashHCViews *v) {
    BnGPUBackend *g=r->gpu;BF16FlashRequest *f=r->flash;
    BnGPURMSNormResidentPlan norm={0};norm.n_tokens=f->n_tokens*model->streams;norm.dim=model->dim;
    norm.weight_offset=1;norm.reduction=BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32;
    norm.eps=1e-6f;norm.weight=w->norm;norm.weight_groups=model->streams;norm.x=v->hidden;norm.out=v->normalized;
    if(g->rmsnorm_bf16_resident(g->ctx,r->request,&norm)||
       bf16_flash_hc_linear(r,w->down,model->rank,model->dim*model->streams,v->normalized,v->down))return -1;
    BnGPUScaledSiluResidentPlan silu={0};silu.rows=f->n_tokens;silu.cols=model->rank;
    silu.divisor=model->streams;silu.x=v->down;silu.out=v->scaled;
    if(g->scaled_silu_bf16_resident(g->ctx,r->request,&silu)||
       bf16_flash_hc_linear(r,w->up,model->dim*model->streams,model->rank,v->scaled,v->gate))return -1;
    BnGPUHyperMixResidentPlan mix={0};mix.n_tokens=f->n_tokens;mix.dim=model->dim;mix.streams=model->streams;
    mix.normalized=v->normalized;mix.gate=v->gate;mix.out=v->mixed;
    if(g->hyper_connection_mix_bf16_resident(g->ctx,r->request,&mix))return -1;
    return 0;
}
int bn_transformer_bf16_flash_hc_mix(BnBackendSession *session,int layer,int ffn) {
    BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->active||r->flash->hc_open||
       !r->flash->hc_weights||layer<0||layer>=r->flash->hc_weights->n_layers||(ffn!=0&&ffn!=1))return -1;
    BF16FlashRequest *f=r->flash;uint64_t bit=UINT64_C(1)<<layer;
    if(ffn?(!(f->expected_ffn&bit)||(f->submitted_ffn&bit)):
           (!(f->expected_layers&bit)||(f->submitted_layers&bit)))return -1;
    BnGPUBackend *g=r->gpu;
    if(!g->rmsnorm_bf16_resident||!g->scaled_silu_bf16_resident||
       !g->hyper_connection_mix_bf16_resident||!g->hyper_connection_combine_bf16_resident)return -1;
    const BnBackendBF16HCModel *model=f->hc_weights;
    const BnBackendBF16HCWeights *w=ffn?&model->ffn[layer]:&model->attention[layer];
    BnTransformerBF16FlashHCViews v;
    if(bn_transformer_bf16_flash_hc_views(session,f->n_tokens,&v)||!w->norm||!w->inject.decoded)return -1;
    r->info.failed=1;
    if(bf16_flash_hc_mix_weights(r,model,w,&v))return -1;
    f->hc_open=1;f->hc_layer=layer;f->hc_branch=ffn;f->hc_residual=v.hidden;
    f->hc_combined=(BnGPUResidentView){0};r->info.failed=0;return 0;
}
int bn_transformer_bf16_flash_hc_combine(BnBackendSession *session) {
    BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->active||!r->flash->hc_open)return -1;
    BF16FlashRequest *f=r->flash;uint64_t bit=UINT64_C(1)<<f->hc_layer;
    if(!f->hc_branch&&!(f->submitted_layers&bit))return -1;
    if(f->hc_branch&&f->ffn.weights&&!f->ffn.ready)return -1;
    const BnBackendBF16HCModel *m=f->hc_weights;
    const BnBackendBF16HCWeights *w=f->hc_branch?&m->ffn[f->hc_layer]:&m->attention[f->hc_layer];
    size_t n=(size_t)f->n_tokens;
    BnGPUResidentView norm=bf16_slice(r->kn,n*m->dim*m->streams),inject=bf16_slice(r->key,n*m->streams);
    r->info.failed=1;
    if(bf16_flash_hc_linear(r,w->inject,m->streams,m->dim*m->streams,norm,inject))return -1;
    BnGPUHyperCombineResidentPlan c={0};c.n_tokens=f->n_tokens;c.dim=m->dim;c.streams=m->streams;
    c.residual=f->hc_residual;c.block_out=bf16_slice(f->out,n*m->dim);c.inject=inject;
    c.out=bf16_slice(r->out,n*m->dim*m->streams);
    if(r->gpu->hyper_connection_combine_bf16_resident(r->gpu->ctx,r->request,&c))return -1;
    f->hc_combined=c.out;
    BnGPUResidentView previous=r->hidden;r->hidden=r->out;r->out=previous;
    if(f->hc_branch)f->submitted_ffn|=bit;else f->hc_attention_done|=bit;
    f->hc_open=0;r->info.failed=0;return 0;
}
int bn_transformer_bf16_flash_attention_submit(BnBackendSession *session,int layer) {
    if(bn_transformer_bf16_flash_hc_mix(session,layer,0))return -1;
    BF16PLERequest *r=bf16_ple_request(session);
    int status=(r->flash->qsa_layers&(UINT64_C(1)<<layer))?
        bn_transformer_bf16_flash_qsa_submit(session,layer):bn_transformer_bf16_flash_gdn_submit(session,layer);
    return status?status:bn_transformer_bf16_flash_hc_combine(session);
}

int bn_transformer_bf16_flash_ffn_views(const BnBackendSession *session, int nt,
    BnTransformerBF16FlashFFNViews *out) {
    const BF16PLERequest *r = bf16_ple_request(session);
    if (!bf16_ple_current(r) || !r->flash || !r->flash->ffn.weights ||
        !out || nt <= 0 || nt > r->info.max_tokens) return -1;
    const BnTransformerBF16FlashFFNViews *v = &r->flash->ffn.v;
    BnTransformerBF16FlashFFNViews s = {0}; size_t n = (size_t)nt;
#define FFN_SLICE(field,count) s.field=bf16_slice(v->field,count)
    FFN_SLICE(logits,n*512); FFN_SLICE(weights,n*10); FFN_SLICE(indices,n*10); FFN_SLICE(source_rows,n*10);
    FFN_SLICE(gathered,n*10*2560); FFN_SLICE(slots,n*10); FFN_SLICE(inverse,n*10);
    FFN_SLICE(gate,n*10*640); FFN_SLICE(up,n*10*640); FFN_SLICE(down,n*10*2560);
    FFN_SLICE(weighted,n*10*2560); FFN_SLICE(routed,n*2560);
    FFN_SLICE(shared_gate,n*640); FFN_SLICE(shared_up,n*640); FFN_SLICE(shared_down,n*2560);
    FFN_SLICE(shared_logits,n); FFN_SLICE(sigmoid,n); FFN_SLICE(shared,n*2560);
#undef FFN_SLICE
    s.block=bf16_slice(r->flash->out,n*2560); *out=s; return 0;
}

int bn_transformer_bf16_flash_ffn_info(const BnBackendSession *session,
    BnTransformerBF16FlashFFNInfo *out) {
    const BF16PLERequest *r=bf16_ple_request(session);
    if (!r || !r->flash || !r->flash->ffn.weights || !out) return -1;
    const BF16FlashFFNRequest *f=&r->flash->ffn;
    BnTransformerBF16FlashFFNInfo s={0};
    s.pending=f->pending;s.layer=f->layer;s.next_expert=f->next_expert;
    s.submitted_blocks=f->submitted_blocks;s.metadata_read_bytes=f->metadata_read_bytes;
    s.metadata_ms=f->metadata_ms;s.source_ms=f->source_ms;s.bank_retry_ms=f->bank_retry_ms;
    s.source=f->source.stats;
    if (r->gpu->expert_cache_stats && r->gpu->expert_cache_stats(r->request,&s.cache)) return -1;
    if(r->gpu->moe_bank_stats(r->request,&s.banks))return -1;
    *out=s;return 0;
}

typedef struct {
    const BnModel *model;
    const BnBackendBF16FlashFFNModel *weights;
    BnMoEState state;
} BF16FlashPreloadSource;

static int bf16_flash_preload_read(void *ptr,int layer,int expert,void *dst,size_t bytes) {
    BF16FlashPreloadSource *source=ptr;
    if(layer<0||layer>=source->weights->n_layers||expert<0||expert>=512)return -1;
    const BnMoEExpertMap *map=&source->weights->layers[layer].expert_map;
    size_t gate=map->expert_gate_bytes,down=map->expert_down_bytes;
    if(gate!=map->expert_up_bytes||gate>SIZE_MAX/2||down>SIZE_MAX-gate*2||bytes!=gate*2+down)return -1;
    unsigned char *raw=dst;
    source->state.buf=raw;source->state.buf_size=gate;
    source->state.buf2=raw+gate;source->state.buf2_size=gate;
    source->state.buf5=raw+gate*2;source->state.buf5_size=down;
    BnMoEHostExpert host={0};
    int status=bn_moe_host_expert_acquire(bn_model_moe_io_const(source->model),
        &source->state,map,layer,expert,0,&host);
    if(!status)for(int role=0;role<3;role++) {
        void *target=raw+(size_t)role*gate;
        if(target!=host.projection[role])memcpy(target,host.projection[role],role==2?down:gate);
    }
    bn_moe_host_expert_release(&host);
    source->state.buf=source->state.buf2=source->state.buf5=NULL;
    return status;
}

int bn_transformer_bf16_flash_preload_experts(const BnModel *model,size_t host_budget,
    BnTransformerBF16FlashPreloadInfo *out) {
    if(!model||!out)return -1;
    BnGPUBackend *g=bn_model_gpu(model);
    const BnBackendBF16FlashFFNModel *w=bn_backend_layout_bf16_flash_ffn_model(bn_model_backend(model));
    if(!g||!g->expert_cache_preload||!w||!w->expert_cache)return -1;
    BF16FlashPreloadSource source={0};source.model=model;source.weights=w;
    BnTransformerBF16FlashPreloadInfo info={0};
    if(g->expert_cache_preload(w->expert_cache,host_budget,bf16_flash_preload_read,&source,&info.cache))return -1;
    info.source=source.state.stats;*out=info;return 0;
}

static int bf16_flash_ffn_start(BF16PLERequest *r,int layer) {
    BF16FlashFFNRequest *s=&r->flash->ffn;
    const BnBackendBF16FlashFFNWeights *w=&s->weights->layers[layer];
    BnTransformerBF16FlashFFNViews v;
    int nt=r->flash->n_tokens;
    /* All views have fixed preparation capacity; slice only, never allocate. */
    v=s->v;
    size_t n=(size_t)nt;
    BnGPUResidentView input=bf16_slice(r->flash->input,n*2560);
    if(bf16_flash_hc_linear(r,w->router,512,2560,input,bf16_slice(v.logits,n*512)) ||
       bf16_flash_hc_linear(r,w->shared_router,1,2560,input,bf16_slice(v.shared_logits,n)))return -1;
    BnGPUMoETopKBF16ResidentPlan top={0};
    top.n_tokens=nt;top.n_experts=512;top.k=10;top.renormalize=1;
    top.logits=bf16_slice(v.logits,n*512);top.weights=bf16_slice(v.weights,n*10);
    top.indices=bf16_slice(v.indices,n*10);top.source_rows=bf16_slice(v.source_rows,n*10);
    BnGPUMoEGroupBF16ResidentPlan group={0};
    group.n_tokens=nt;group.n_experts=512;group.k=10;group.dim=2560;
    group.x=input;group.indices=top.indices;group.offsets=bn_gpu_resident_view_slice(s->metadata,0,513);
    group.status=bn_gpu_resident_view_slice(s->metadata,516,1);group.cursors=s->cursors;
    group.slots=bf16_slice(v.slots,n*10);group.inverse=bf16_slice(v.inverse,n*10);
    group.gathered=bf16_slice(v.gathered,n*10*2560);
    BnGPUBackend *g=r->gpu;
    if(g->moe_topk_bf16_resident(g->ctx,r->request,&top) ||
       g->moe_group_bf16_resident(g->ctx,r->request,&group))return -1;
    double start=bn_platform_time_ms();
    int status=g->resident_read(r->request,s->metadata,s->offsets);
    s->metadata_ms+=bn_platform_time_ms()-start;s->metadata_read_bytes+=sizeof(s->offsets);
    if(status || s->offsets[0] || s->offsets[512]!=nt*10 || s->offsets[516])return -1;
    for(int e=0;e<512;e++)if(s->offsets[e]<0 || s->offsets[e+1]<s->offsets[e] ||
        s->offsets[e+1]-s->offsets[e]>nt)return -1;
    BnGPUFFNResidentPlan shared={0};shared.n_tokens=nt;shared.dim=2560;shared.hidden_dim=640;
    shared.activation_profile=s->weights->shared_activation;shared.weights=w->shared;
    shared.x=input;shared.gate=bf16_slice(v.shared_gate,n*640);shared.up=bf16_slice(v.shared_up,n*640);
    shared.out=bf16_slice(v.shared_down,n*2560);
    BnGPUMoEBankInputPlan prep={nt,input,group.slots,group.status};
    if(g->dense_ffn_bf16_resident(g->ctx,r->request,&shared) ||
       g->moe_bank_prepare_input(g->ctx,r->request,&prep,&s->generation) || !s->generation)return -1;
    s->pending=1;s->layer=layer;s->next_expert=0;s->bank=0;s->ready=0;s->cache_started=0;return 0;
}

static int bf16_flash_ffn_prepare(BnBackendSession *session,int layer) {
    BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r) || !r->flash || !r->flash->active || !r->flash->hc_required ||
       !r->flash->ffn.weights || layer<0 || layer>=r->flash->ffn.weights->n_layers)return -1;
    BF16FlashRequest *f=r->flash;BF16FlashFFNRequest *s=&f->ffn;
    uint64_t bit=UINT64_C(1)<<layer;
    if(!(f->expected_ffn&bit) || (f->submitted_ffn&bit) ||
       (f->hc_open && (!f->hc_branch || f->hc_layer!=layer)) ||
       (s->pending && s->layer!=layer))return -1;
    if(!f->hc_open && bn_transformer_bf16_flash_hc_mix(session,layer,1))return -1;
    r->info.failed=1;
    if(!s->pending && bf16_flash_ffn_start(r,layer))return -1;
    r->info.failed=0;return 0;
}
int bn_transformer_bf16_flash_ffn_submit(BnBackendSession *session,int layer) {
    if(bf16_flash_ffn_prepare(session,layer))return -1;
    BF16PLERequest *r=bf16_ple_request(session);
    BF16FlashRequest *f=r->flash;BF16FlashFFNRequest *s=&f->ffn;
    r->info.failed=1;
    BnGPUBackend *g=r->gpu;int nt=f->n_tokens;size_t n=(size_t)nt;
    if (s->weights->expert_cache && !s->cache_started && s->next_expert<512) {
        int status=g->expert_cache_begin(r->request,layer,f->prefill,s->offsets,nt,&s->cache_layer);
        if (status==BN_GPU_MOE_BANK_BUSY) { r->info.failed=0;return status; }
        if (status) return -1;
        s->cache_started=1;
    }
    const BnMoEExpertMap *map=&s->weights->layers[layer].expert_map;
    while(s->next_expert<512) {
        while(s->next_expert<512 && s->offsets[s->next_expert]==s->offsets[s->next_expert+1])s->next_expert++;
        if(s->next_expert==512)break;
        BnGPUMoEBankLease lease={0};
        int status=g->moe_bank_acquire(r->request,s->bank%2,map->gate_type,map->down_type,&lease);
        if(status==BN_GPU_MOE_BANK_BUSY) {
            if(!s->wait_start)s->wait_start=bn_platform_time_ms();
            r->info.failed=0;return status;
        }
        if(status)return -1;
        if(s->wait_start){s->bank_retry_ms+=bn_platform_time_ms()-s->wait_start;s->wait_start=0;}
        if(lease.capacity!=8 || lease.gate_bytes!=map->expert_gate_bytes ||
           lease.gate_bytes!=map->expert_up_bytes || lease.down_bytes!=map->expert_down_bytes) {
            g->moe_bank_cancel(r->request,&lease);return -1;
        }
        BnGPUMoEBankItem items[8];int count=0,expert=s->next_expert;
        double start=bn_platform_time_ms();
        while(expert<512 && count<8) {
            int first=s->offsets[expert],rows=s->offsets[expert+1]-first;
            if(rows) {
                unsigned char *raw=(unsigned char *)lease.data+(size_t)count*lease.expert_bytes;
                s->source.buf=raw;s->source.buf_size=lease.gate_bytes;
                s->source.buf2=raw+lease.gate_bytes;s->source.buf2_size=lease.gate_bytes;
                s->source.buf5=raw+lease.gate_bytes*2;s->source.buf5_size=lease.down_bytes;
                int cached=s->cache_started &&
                    ((s->cache_layer.hits[expert/64]>>(expert%64))&1);
                BnMoEHostExpert host={0};
                if (!cached) {
                status=bn_moe_host_expert_acquire(bn_model_moe_io_const(r->model),&s->source,map,
                    layer,expert,!f->prefill,&host);
                if(!status)for(int role=0;role<3;role++) {
                    void *dst=raw+(size_t)role*lease.gate_bytes;
                    size_t bytes=role==2?lease.down_bytes:lease.gate_bytes;
                    if(dst!=host.projection[role])memcpy(dst,host.projection[role],bytes);
                }
                bn_moe_host_expert_release(&host);
                }
                s->source.buf=s->source.buf2=s->source.buf5=NULL;
                if(status){g->moe_bank_cancel(r->request,&lease);return -1;}
                items[count++]=(BnGPUMoEBankItem){expert,first,rows};
            }
            expert++;
        }
        s->source_ms+=bn_platform_time_ms()-start;
        BnGPUMoEBankResidentPlan bank={0};bank.lease=lease;bank.total_tokens=nt;
        bank.n_experts=count;bank.experts=items;bank.prepared_input=s->generation;bank.grouped=1;
        bank.x=bf16_slice(s->v.gathered,n*10*2560);bank.gate=bf16_slice(s->v.gate,n*10*640);
        bank.up=bf16_slice(s->v.up,n*10*640);bank.out=bf16_slice(s->v.down,n*10*2560);
        if(g->moe_bank_submit(g->ctx,r->request,&bank)) {
            g->moe_bank_cancel(r->request,&lease);return -1;
        }
        s->next_expert=expert;s->bank++;
    }
    BnGPUMoEReduceBF16ResidentPlan reduce={0};reduce.n_tokens=nt;reduce.k=10;reduce.dim=2560;
    reduce.grouped=bf16_slice(s->v.down,n*10*2560);reduce.inverse=bf16_slice(s->v.inverse,n*10);
    reduce.weights=bf16_slice(s->v.weights,n*10);reduce.status=bn_gpu_resident_view_slice(s->metadata,516,1);
    reduce.weighted=bf16_slice(s->v.weighted,n*10*2560);reduce.out=bf16_slice(s->v.routed,n*2560);
    BnGPUMoESharedBF16ResidentPlan combine={0};combine.n_tokens=nt;combine.dim=2560;
    combine.routed=reduce.out;combine.down=bf16_slice(s->v.shared_down,n*2560);
    combine.logits=bf16_slice(s->v.shared_logits,n);combine.sigmoid=bf16_slice(s->v.sigmoid,n);
    combine.shared=bf16_slice(s->v.shared,n*2560);combine.out=bf16_slice(f->out,n*2560);
    if(g->moe_reduce_bf16_resident(g->ctx,r->request,&reduce) ||
       g->moe_shared_bf16_resident(g->ctx,r->request,&combine))return -1;
    s->ready=1;r->info.failed=0;
    if(bn_transformer_bf16_flash_hc_combine(session))return -1;
    if(s->cache_started && g->expert_cache_end(r->request,&s->cache_layer))return -1;
    s->cache_started=0;
    s->pending=s->ready=0;s->submitted_blocks++;return 0;
}
int bn_transformer_bf16_flash_io_views(const BnBackendSession *session,int nt,
    BnTransformerBF16FlashIOViews *out) {
    const BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->io_weights||!out||nt<=0||nt>r->info.max_tokens)return -1;
    const BF16FlashRequest *f=r->flash;size_t n=(size_t)nt;
    BnTransformerBF16FlashIOViews v={0};v.ids=bf16_slice(f->token_ids,n);
    v.embedding=bf16_slice(r->embedding,n*2560);v.repeated=bf16_slice(r->key,n*5120);
    v.hidden=bf16_slice(r->hidden,n*10240);v.normalized=bf16_slice(r->kn,n*10240);
    v.down=bf16_slice(r->value,n*r->flash->hc_weights->rank);
    v.scaled=bf16_slice(r->embedding,n*r->flash->hc_weights->rank);
    v.gate=bf16_slice(r->qn,n*10240);v.mixed=bf16_slice(f->input,n*2560);v.logits=f->logits;
    *out=v;return 0;
}

int bn_transformer_bf16_flash_tokens_submit(BnBackendSession *session,const int32_t *tokens,int nt) {
    BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->active||!r->flash->io_weights||!tokens||
       r->flash->hc_open||r->flash->tokens_submitted||r->flash->submitted_layers||
       r->flash->submitted_ffn||r->info.pending_tokens||nt!=r->flash->n_tokens)return -1;
    BF16FlashRequest *f=r->flash;BnGPUBackend *g=r->gpu;
    for(int i=0;i<nt;i++)if(tokens[i]<0||tokens[i]>=f->io_weights->vocab_size)return -1;
    BnTransformerBF16FlashIOViews v;if(bn_transformer_bf16_flash_io_views(session,nt,&v))return -1;
    r->info.failed=1;
    BnGPUEmbeddingResidentPlan e={0};e.n_tokens=nt;e.dim=2560;e.vocab_size=f->io_weights->vocab_size;
    e.weight=f->io_weights->embedding;e.indices=v.ids;e.out=v.embedding;
    if(g->resident_write(r->request,v.ids,tokens)||g->embedding_bf16_resident(g->ctx,r->request,&e))return -1;
    BnGPUJoinColumnsResidentPlan join={0};join.rows=nt;join.cols=5120;join.left_cols=2560;
    join.left=join.right=e.out;join.x=v.repeated;
    if(g->join_columns_resident(g->ctx,r->request,&join))return -1;
    join.cols=10240;join.left_cols=5120;join.left=join.right=join.x;join.x=v.hidden;
    if(g->join_columns_resident(g->ctx,r->request,&join))return -1;
    f->tokens_submitted=1;r->info.failed=0;return 0;
}

int bn_transformer_bf16_flash_head_submit(BnBackendSession *session) {
    BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->active||!r->flash->io_weights||
       r->flash->hc_open||r->flash->head_submitted||
       r->flash->submitted_layers!=r->flash->expected_layers||
       r->flash->submitted_ffn!=r->flash->expected_ffn||
       (r->flash->hc_required&&r->flash->hc_attention_done!=r->flash->expected_layers)||
       (r->flash->includes_ple&&r->info.pending_tokens!=r->flash->n_tokens))return -1;
    BF16FlashRequest *f=r->flash;BnTransformerBF16FlashHCViews v;
    if(bn_transformer_bf16_flash_hc_views(session,f->n_tokens,&v))return -1;
    r->info.failed=1;
    if(bf16_flash_hc_mix_weights(r,f->hc_weights,f->hc_weights->output,&v))return -1;
    BnGPUBF16LinearResidentPlan linear={0};linear.n_tokens=1;linear.rows=f->io_weights->vocab_size;linear.cols=2560;
    linear.weight=f->io_weights->output;linear.x=bn_gpu_resident_view_slice(v.mixed,(size_t)(f->n_tokens-1)*2560,2560);
    linear.out=f->logits;
    if(r->gpu->linear_bf16_resident(r->gpu->ctx,r->request,&linear))return -1;
    f->head_submitted=1;r->info.failed=0;return 0;
}

int bn_transformer_bf16_flash_model_begin(BnBackendSession *session,const int32_t *tokens,
    int nt,int position,int prefill) {
    BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->io_weights||!r->flash->model_tokens||
       !tokens||nt<=0||nt>r->info.max_tokens||r->flash->model_running)return -1;
    BF16FlashRequest *f=r->flash;int count=f->weights->n_layers;
    uint64_t all=count==64?UINT64_MAX:(UINT64_C(1)<<count)-1;
    if(f->valid_layers!=all)return -1;
    for(int i=0;i<nt;i++)if(tokens[i]<0||tokens[i]>=f->io_weights->vocab_size)return -1;
    int status=bn_transformer_bf16_flash_begin_blocks(session,nt,position,prefill,all,all,1);
    if(status)return status;
    if (f->model_tokens != tokens) memcpy(f->model_tokens,tokens,(size_t)nt*sizeof(*tokens));
    f->model_logits=1;f->model_running=1;f->model_layer=f->model_phase=0;return 0;
}

int bn_transformer_bf16_flash_model_submit(BnBackendSession *session) {
    BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->active||!r->flash->model_running)return -1;
    BF16FlashRequest *f=r->flash;
    if(!f->model_phase) {
        if(bn_transformer_bf16_flash_tokens_submit(session,f->model_tokens,f->n_tokens))goto fail;
        f->model_phase=1;
    }
    while(f->model_layer<f->weights->n_layers) {
        if(f->model_phase==1) {
            if(f->model_layer==r->weights->layer &&
               bn_transformer_bf16_flash_ple_apply(session,f->model_tokens,f->n_tokens,f->position))goto fail;
            if(bn_transformer_bf16_flash_attention_submit(session,f->model_layer))goto fail;
            f->model_phase=2;
        }
        int status=bn_transformer_bf16_flash_ffn_submit(session,f->model_layer);
        if(status==BN_GPU_MOE_BANK_BUSY)return status;
        if(status)goto fail;
        f->model_layer++;f->model_phase=1;
    }
    if(f->model_phase!=4) {
        f->model_phase=3;
        if(f->model_logits&&bn_transformer_bf16_flash_head_submit(session))goto fail;
        f->model_phase=4;
    }
    return 0;
fail:
    if(r->gpu->expert_cache_cancel)(void)r->gpu->expert_cache_cancel(r->request);
    r->info.failed=1;return -1;
}
int bn_transformer_bf16_flash_ple_apply(BnBackendSession *session,const int32_t *tokens,int nt,int position) {
    BF16PLERequest *r=bf16_ple_request(session);
    if(!bf16_ple_current(r)||!r->flash||!r->flash->active||r->flash->hc_open)return -1;
    if(bn_transformer_bf16_ple_submit(session,tokens,nt,position))return -1;
    BnGPUResidentView previous=r->hidden;r->hidden=r->out;r->out=previous;return 0;
}

static int bf16_flash_can_complete(const BF16PLERequest *r) {
    if (!bf16_ple_current(r) || !r->flash || !r->flash->active) return -1;
    BF16FlashRequest *f = r->flash;
    if (f->hc_open || f->submitted_layers != f->expected_layers ||
        (f->model_running&&f->model_phase!=4) ||
        f->submitted_ffn != f->expected_ffn || (f->hc_required && f->hc_attention_done != f->expected_layers) ||
        (f->includes_ple && r->info.pending_tokens != f->n_tokens)) return -1;
    return 0;
}
/* No GPU calls or other fallible work: cohort publication uses this only
 * after every participant has passed validation, readback and fences. */
static void bf16_flash_commit(BF16PLERequest *r) {
    BF16FlashRequest *f=r->flash;
    for (int l = 0; l < f->weights->n_layers; ++l) {
        if (f->submitted_ffn & (UINT64_C(1) << l)) f->ffn.positions[l] += f->n_tokens;
        if (!(f->submitted_layers & (UINT64_C(1) << l))) continue;
        f->parity[l] = 1-f->parity[l]; f->positions[l] += f->n_tokens;
    }
    if (f->includes_ple) {
        r->parity = 1-r->parity; r->info.position += r->info.pending_tokens;
        r->info.pending_tokens = 0;
        memcpy(r->context, r->next_context,
            (size_t)(r->model->config.ple_ngram_size - 1) * sizeof(int32_t));
    }
    f->position += f->n_tokens; f->n_tokens = 0; f->active = 0;
    f->expected_layers = f->submitted_layers = 0; f->includes_ple = f->lease_valid = 0;
    f->expected_ffn = f->submitted_ffn = f->hc_attention_done = 0; f->hc_required = 0;
    f->model_running=0;
}
int bn_transformer_bf16_flash_complete(BnBackendSession *session) {
    BF16PLERequest *r=bf16_ple_request(session);
    if(bf16_flash_can_complete(r))return -1;
    if(r->gpu->resident_synchronize(r->request)){r->info.failed=1;return -1;}
    bf16_flash_commit(r);return 0;
}

int bn_transformer_bf16_flash_restart(BnBackendSession *session) {
    BF16PLERequest *r = bf16_ple_request(session);
    if (!bf16_ple_current(r) || !r->flash || r->flash->active || r->info.pending_tokens) return -1;
    BF16FlashRequest *f = r->flash;
    r->info.failed = 1;
    if (r->gpu->resident_zero(r->request, r->history[0])) return -1;
    for (int l = 0; l < f->weights->n_layers; ++l) {
        if (!(f->valid_layers & (UINT64_C(1) << l))) continue;
        if (f->qsa_layers & (UINT64_C(1) << l)) {
            BF16FlashQSALayer *q = &f->qsa[l];
            if (r->gpu->resident_zero(r->request,q->key) || r->gpu->resident_zero(r->request,q->value) ||
                r->gpu->resident_zero(r->request,q->compressed) || r->gpu->resident_zero(r->request,q->history[0])) return -1;
        } else if (r->gpu->resident_zero(r->request, f->layers[l].state[0]) ||
            r->gpu->resident_zero(r->request, f->layers[l].history[0])) return -1;
    }
    if (r->gpu->resident_synchronize(r->request)) return -1;
    memset(f->positions, 0, sizeof(f->positions)); memset(f->parity, 0, sizeof(f->parity));
    memset(f->ffn.positions, 0, sizeof(f->ffn.positions));
    f->position = f->n_tokens = f->prefill = 0;
    f->hc_residual = f->hc_combined = (BnGPUResidentView){0}; f->hc_open = f->hc_required = 0;
    r->parity = r->next_slot = r->last_tokens = 0; r->info.position = 0;
    for (int i = 0; i < r->model->config.ple_ngram_size - 1; ++i)
        r->context[i] = r->model->config.ple_eos_token_id;
    r->info.failed = 0;
    return 0;
}

static int bf16_flash_runtime_current(const BF16PLERequest *r) {
    return bf16_ple_current(r) && r->flash && r->flash->io_weights &&
        r->flash->host_logits && r->gpu == bn_model_gpu(r->model) &&
        r->flash->io_weights == bn_backend_layout_bf16_flash_model(bn_model_backend(r->model));
}

static int bf16_flash_runtime_query(const BnBackendSession *session, int query) {
    const BF16PLERequest *r = bf16_ple_request(session);
    if (!bf16_flash_runtime_current(r)) return query ? 0 : -1;
    return query == 0 ? r->flash->position :
        (query == 1 ? r->info.max_tokens : r->flash->capacity);
}

static int bf16_flash_runtime_restart(const BnModel *model, BnSession *session) {
    BnBackendSession *backend = bn_session_backend(session);
    const BF16PLERequest *r = bf16_ple_request(backend);
    if (!bf16_flash_runtime_current(r) || r->model != model ||
        bn_transformer_bf16_flash_restart(backend)) return -1;
    memset(session->state.token_history, 0,
        (size_t)model->config.seq_len * sizeof(*session->state.token_history));
    session->pos = 0;
    session->gpu_kv_direct_valid = 0;
    session->moe_cache_prefill = 0;
    session->state.batched_prompt_contract = 0;
    bn_backend_session_set_gpu_decode_ssm_pending(backend, 0);
    return 0;
}

static int bf16_flash_runtime_run(BnModel *model, BnSession *session,
    const int *tokens, int nt, int position, int want_logits, int prefill) {
    BnBackendSession *backend = bn_session_backend(session);
    BF16PLERequest *r = bf16_ple_request(backend);
    if (!session->backend_only || !bf16_flash_runtime_current(r) || r->model != model ||
        nt <= 0 || nt > r->info.max_tokens || r->flash->active ||
        position != r->flash->position ||
        (want_logits && !session->state.logits)) return -1;
    BF16FlashRequest *f = r->flash;
    for (int i = 0; i < nt; i++)
        if (tokens[i] < 0 || tokens[i] >= model->config.vocab_size) return -1;
    for (int i = 0; i < nt; i++) f->model_tokens[i] = tokens[i];
    int status;
    while ((status = bn_transformer_bf16_flash_model_begin(backend,
            f->model_tokens, nt, position, prefill)) == BN_GPU_MOE_BANK_BUSY) sched_yield();
    if (status) return -1;
    f->model_logits = want_logits;
    while ((status = bn_transformer_bf16_flash_model_submit(backend)) ==
            BN_GPU_MOE_BANK_BUSY) sched_yield();
    if (status || (want_logits &&
        r->gpu->resident_read(r->request, f->logits, f->host_logits)) ||
        bn_transformer_bf16_flash_complete(backend)) {
        r->info.failed = 1;
        return -1;
    }
    /* Publish only after readback and the whole state transaction succeed. */
    for(int i=0;i<nt;i++)session->state.token_history[position+i]=f->model_tokens[i];
    session->pos=position+nt;
    if (want_logits) for (int i = 0; i < model->config.vocab_size; i++) {
        uint32_t bits = (uint32_t)f->host_logits[i] << 16;
        memcpy(&session->state.logits[i], &bits, sizeof(bits));
    }
    return 0;
}

/* Public view metadata can be checked without queueing anything. Opaque slot
 * generations, device pointers and reserved scratch remain backend checks. */
int bn_transformer_bf16_kda_layer(const BnBackendBF16KDAWeights *w,
    void *request, int nt, const BnTransformerBF16KDAWorkspace *s,
    BnGPUResidentView x, BnGPUResidentView ih, BnGPUResidentView nh,
    BnGPUResidentView is, BnGPUResidentView ns, BnGPUResidentView out) {
    if (!w || !s || !request || nt <= 0 || w->dim <= 0 || w->dim%32 ||
        w->heads <= 0 || w->heads > (INT_MAX-256)/385 || w->head_dim!=128 ||
        w->rank!=128 || w->kernel!=4 || !w->in.raw || !w->f_b.raw ||
        !w->g_b.raw || !w->out.raw || !bn_backend_quant_uses_native_quant(w->in.type) ||
        !bn_backend_quant_uses_native_quant(w->f_b.type) || !bn_backend_quant_uses_native_quant(w->g_b.type) ||
        !bn_backend_quant_uses_native_quant(w->out.type) || !w->conv || !w->decay ||
        !w->bias || !w->norm || !isfinite(w->lower_bound) || !(w->lower_bound<0) ||
        !isfinite(w->output_eps) ||
        !(w->output_eps>0)) return -1;
    BnGPUBackend *gpu=w->gpu;
    if (!gpu || !gpu->linear_a8_bf16_resident || !gpu->split_columns_resident ||
        !gpu->causal_conv_f32_weight_bf16_resident ||
        !gpu->kda_recurrent_bf16_resident ||
        !gpu->gated_rmsnorm_f32_weight_bf16_resident) return -1;
    int p=w->heads*128, merged=3*p+w->heads+256;
    if (nt>INT_MAX/merged || nt>INT_MAX/w->dim) return -1;
    BnGPUResidentView views[]={x,ih,nh,is,ns,out,s->projected,s->qkv,s->tail,
        s->fa_ga,s->beta,s->fa,s->ga,s->g1,s->g2,s->conv,s->core,s->gated};
    size_t n=(size_t)nt;
    size_t counts[]={n*w->dim,(size_t)3*3*p,(size_t)3*3*p,
        (size_t)p*128,(size_t)p*128,n*w->dim,n*merged,n*3*p,
        n*(w->heads+256),n*256,n*w->heads,n*128,n*128,n*p,n*p,n*3*p,n*p,n*p};
    for (size_t i=0;i<sizeof(views)/sizeof(views[0]);++i) {
        int type=(i==3||i==4)?bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32):BF16;
        if (views[i].request!=request || views[i].type!=type ||
            views[i].count!=counts[i] || views[i].offset>SIZE_MAX-counts[i]) return -1;
        /* All extents include read-only generations so accidental aliasing is
         * rejected even if a later operator could tolerate it in isolation. */
        for (size_t j=0;j<i;++j) if (views[i].slot==views[j].slot &&
            views[i].offset<views[j].offset+views[j].count &&
            views[j].offset<views[i].offset+views[i].count) return -1;
    }
    BnGPUBF16LinearResidentPlan first={.n_tokens=nt,.rows=merged,.cols=w->dim,
        .weight=w->in,.x=x,.out=s->projected};
    BnGPUSplitColumnsResidentPlan splits[]={
        {.rows=nt,.cols=merged,.left_cols=3*p,.x=s->projected,.left=s->qkv,.right=s->tail},
        {.rows=nt,.cols=w->heads+256,.left_cols=w->heads,.x=s->tail,.left=s->beta,.right=s->fa_ga},
        {.rows=nt,.cols=256,.left_cols=128,.x=s->fa_ga,.left=s->fa,.right=s->ga}};
    BnGPUBF16LinearResidentPlan fb={.n_tokens=nt,.rows=p,.cols=128,
        .weight=w->f_b,.x=s->fa,.out=s->g1};
    BnGPUBF16LinearResidentPlan gb={.n_tokens=nt,.rows=p,.cols=128,
        .weight=w->g_b,.x=s->ga,.out=s->g2};
    BnGPUCausalConvResidentPlan conv={.n_tokens=nt,.dim=3*p,.kernel=4,
        .weight=w->conv,.x=s->qkv,.initial_history=ih,.next_history=nh,.out=s->conv};
    BnGPUKDARecurrentResidentPlan kda={.n_tokens=nt,.num_heads=w->heads,.head_dim=128,
        .q_scale=0.08838834764831845f,.lower_bound=w->lower_bound,
        .parameter_storage=BN_GPU_KDA_A_NEGATIVE_EXP,.decay_parameter=w->decay,
        .gate_bias=w->bias,.x=s->conv,.gate=s->g1,.beta_raw=s->beta,
        .initial_state=is,.next_state=ns,.out=s->core};
    BnGPUGatedRMSNormResidentPlan norm={.rows=nt*w->heads,.dim=128,
        .eps=w->output_eps,.activation=BN_GPU_GATED_NORM_SIGMOID,
        .profile=BN_GPU_GATED_NORM_GLM_R2,.weight=w->norm,
        .x=s->core,.z=s->g2,.out=s->gated};
    BnGPUBF16LinearResidentPlan last={.n_tokens=nt,.rows=w->dim,.cols=p,
        .weight=w->out,.x=s->gated,.out=out};
    if(gpu->linear_a8_bf16_resident(gpu->ctx,request,&first)) return -1;
    for(int i=0;i<3;++i)
        if(gpu->split_columns_resident(gpu->ctx,request,&splits[i])) return -1;
    if(gpu->linear_a8_bf16_resident(gpu->ctx,request,&fb) ||
       gpu->linear_a8_bf16_resident(gpu->ctx,request,&gb) ||
       gpu->causal_conv_f32_weight_bf16_resident(gpu->ctx,request,&conv) ||
       gpu->kda_recurrent_bf16_resident(gpu->ctx,request,&kda) ||
       gpu->gated_rmsnorm_f32_weight_bf16_resident(gpu->ctx,request,&norm) ||
       gpu->linear_a8_bf16_resident(gpu->ctx,request,&last)) return -1;
    return 0;
}

static int bf16_mhc_kda_validate(void *request,const BnTransformerBF16MHCKDAPlan *p) {
    if(!request||!p||!p->weights||(p->deferred!=0&&p->deferred!=1)||
       (p->has_previous!=0&&p->has_previous!=1))return -1;
    const BnBackendBF16KDAWeights*w=p->weights;BnGPUBackend*g=w->gpu;
    const BnGPUMHCPreResidentPlan*m=&p->mhc;
    if(!g||(!p->has_previous&&!g->mhc_pre_bf16_resident)||
       (p->has_previous&&!g->mhc_fused_post_pre_bf16_resident)||
       (!p->deferred&&!g->mhc_post_bf16_resident)||
       !g->linear_a8_bf16_resident||!g->split_columns_resident||
       !g->causal_conv_f32_weight_bf16_resident||!g->kda_recurrent_bf16_resident||
       !g->gated_rmsnorm_f32_weight_bf16_resident||!m->norm||
       m->streams!=4||m->dim!=w->dim||m->dim<=0||m->dim%1024||
       m->n_tokens<=0||m->dim>INT_MAX/4||m->n_tokens>INT_MAX/(m->dim*4)||
       m->n_splits<1||m->n_splits>m->dim*4/64||m->n_splits>65535||
       w->heads<=0||w->heads>(INT_MAX-256)/385||w->head_dim!=128||w->rank!=128||w->kernel!=4||
       !w->in.raw||!w->f_b.raw||!w->g_b.raw||!w->out.raw||!w->conv||!w->norm||!w->decay||!w->bias||
       !bn_backend_quant_uses_native_quant(w->in.type)||!bn_backend_quant_uses_native_quant(w->f_b.type)||
       !bn_backend_quant_uses_native_quant(w->g_b.type)||!bn_backend_quant_uses_native_quant(w->out.type)||
       !isfinite(w->lower_bound)||w->lower_bound>=0||!isfinite(w->output_eps)||w->output_eps<=0)return -1;
    int wide=w->heads*128,merged=wide*3+w->heads+256;
    if(m->n_tokens>INT_MAX/merged)return -1;
    const BnTransformerBF16KDAWorkspace*s=&p->kda;size_t n=m->n_tokens,ns=n*m->n_splits;
    BnGPUResidentView v[30]={m->residual,m->dot,m->sqrsum,m->pre,m->post,m->comb,m->collapsed,m->out,
        s->projected,s->qkv,s->tail,s->fa_ga,s->beta,s->fa,s->ga,s->g1,s->g2,s->conv,s->core,s->gated,
        p->initial_history,p->next_history,p->initial_state,p->next_state,p->branch_out,p->residual_out};
    size_t counts[30]={n*w->dim*4,ns*24,ns,n*4,n*4,n*16,n*w->dim,n*w->dim,
        n*merged,n*wide*3,n*(w->heads+256),n*256,n*w->heads,n*128,n*128,n*wide,n*wide,n*wide*3,n*wide,n*wide,
        (size_t)wide*9,(size_t)wide*9,(size_t)wide*128,(size_t)wide*128,n*w->dim,n*w->dim*4};
    size_t nv=p->deferred?25:26,prior_start=nv;
    if(p->has_previous) {
        BnGPUResidentView prior[]={p->previous_residual,p->previous_branch,p->previous_post,p->previous_comb};
        size_t sizes[]={n*w->dim*4,n*w->dim,n*4,n*16};
        for(int i=0;i<4;++i){v[nv]=prior[i];counts[nv++]=sizes[i];}
    }
    for(size_t i=0;i<nv;++i) {
        int type=((i>=1&&i<=5)||i==22||i==23||(p->has_previous&&i>=prior_start+2))?bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32):BF16;
        if(v[i].request!=request||v[i].type!=type||v[i].count!=counts[i]||v[i].offset>SIZE_MAX-counts[i])return -1;
        for(size_t j=0;j<i;++j)if(v[i].slot==v[j].slot&&v[i].offset<v[j].offset+v[j].count&&
            v[j].offset<v[i].offset+v[i].count)return -1;
    }
    return 0;
}

int bn_transformer_bf16_mhc_kda_attention(void *request,const BnTransformerBF16MHCKDAPlan *p) {
    if(bf16_mhc_kda_validate(request,p))return -1;
    const BnBackendBF16KDAWeights*w=p->weights;BnGPUBackend*g=w->gpu;
    const BnGPUMHCPreResidentPlan*m=&p->mhc;
    const BnTransformerBF16KDAWorkspace*s=&p->kda;
    if(p->has_previous) {
        BnGPUMHCFusedResidentPlan fused={.next=*m,.previous_residual=p->previous_residual,
            .previous_branch=p->previous_branch,.previous_post=p->previous_post,.previous_comb=p->previous_comb};
        if(g->mhc_fused_post_pre_bf16_resident(g->ctx,request,&fused))return -1;
    } else if(g->mhc_pre_bf16_resident(g->ctx,request,m))return -1;
    if(bn_transformer_bf16_kda_layer(w,request,m->n_tokens,s,m->out,
        p->initial_history,p->next_history,p->initial_state,p->next_state,p->branch_out))return -1;
    if(p->deferred)return 0;
    BnGPUMHCPostResidentPlan post={.n_tokens=m->n_tokens,.dim=m->dim,.streams=4,
        .residual=m->residual,.branch=p->branch_out,.post=m->post,.comb=m->comb,.out=p->residual_out};
    return g->mhc_post_bf16_resident(g->ctx,request,&post);
}

static int bf16_dense_ffn_validate(const BnBackendBF16DenseFFNWeights *w,
    void *request,int nt,BnGPUResidentView x,BnGPUResidentView gate_up,
    BnGPUResidentView activated,BnGPUResidentView out) {
    if(!w||!request||nt<=0||w->dim<=0||w->dim%32||w->hidden_dim<=0||
       w->hidden_dim%32||w->hidden_dim>INT_MAX/2||
       nt>INT_MAX/(2*w->hidden_dim)||nt>INT_MAX/w->dim||
       !isfinite(w->clamp_limit)||w->clamp_limit<0||!w->gate_up.raw||!w->down.raw||
       !bn_backend_quant_uses_native_quant(w->gate_up.type)||
       !bn_backend_quant_uses_native_quant(w->down.type))return -1;
    BnGPUBackend*g=w->gpu;
    if(!g||!g->linear_a8_bf16_resident||!g->clamped_swiglu_bf16_resident)return -1;
    BnGPUResidentView v[]={x,gate_up,activated,out};
    size_t n=(size_t)nt,counts[]={n*w->dim,n*2*w->hidden_dim,n*w->hidden_dim,n*w->dim};
    for(int i=0;i<4;++i) {
        if(v[i].request!=request||v[i].type!=BF16||v[i].count!=counts[i]||
           v[i].offset>SIZE_MAX-counts[i])return -1;
        for(int j=0;j<i;++j)if(v[i].slot==v[j].slot&&v[i].offset<v[j].offset+counts[j]&&
            v[j].offset<v[i].offset+counts[i])return -1;
    }
    return 0;
}

int bn_transformer_bf16_dense_ffn(const BnBackendBF16DenseFFNWeights *w,
    void *request,int nt,BnGPUResidentView x,BnGPUResidentView gate_up,
    BnGPUResidentView activated,BnGPUResidentView out) {
    if(bf16_dense_ffn_validate(w,request,nt,x,gate_up,activated,out))return -1;
    BnGPUBackend*g=w->gpu;
    BnGPUBF16LinearResidentPlan gu={.n_tokens=nt,.rows=2*w->hidden_dim,.cols=w->dim,
        .weight=w->gate_up,.x=x,.out=gate_up};
    BnGPUClampedSwiGLUResidentPlan act={.n_tokens=nt,.hidden_dim=w->hidden_dim,
        .limit=w->clamp_limit,.gate_up=gate_up,.out=activated};
    BnGPUBF16LinearResidentPlan down={.n_tokens=nt,.rows=w->dim,.cols=w->hidden_dim,
        .weight=w->down,.x=activated,.out=out};
    return g->linear_a8_bf16_resident(g->ctx,request,&gu)||
        g->clamped_swiglu_bf16_resident(g->ctx,request,&act)||
        g->linear_a8_bf16_resident(g->ctx,request,&down)?-1:0;
}

int bn_transformer_bf16_dense_kda_block(void *request,
    const BnTransformerBF16DenseKDABlockPlan*p) {
    if(!p||p->attention.deferred!=1||bf16_mhc_kda_validate(request,&p->attention))return -1;
    const BnTransformerBF16MHCKDAPlan*a=&p->attention;
    const BnGPUMHCPreResidentPlan*f=&p->ffn;
    const BnBackendBF16DenseFFNWeights*w=p->weights;BnGPUBackend*g=a->weights->gpu;
    int nt=a->mhc.n_tokens,dim=a->mhc.dim;
    if(!w||w->gpu!=g||w->dim!=dim||!g->mhc_fused_post_pre_bf16_resident||
       f->n_tokens!=nt||f->dim!=dim||f->streams!=4||f->n_splits<1||
       f->n_splits>dim*4/64||f->n_splits>65535||f->iterations<1||f->iterations>128||
       !f->fn||!f->scale||!f->base||!f->norm||
       !isfinite(f->rms_eps)||f->rms_eps<=0||!isfinite(f->pre_eps)||f->pre_eps<0||
       !isfinite(f->sinkhorn_eps)||f->sinkhorn_eps<=0||!isfinite(f->post_mult)||f->post_mult<0||
       !isfinite(f->norm_eps)||f->norm_eps<=0||
       bf16_dense_ffn_validate(w,request,nt,f->out,p->gate_up,p->activated,p->out))return -1;
    /* A single global alias preflight includes both sublayers and incoming
     * deferred state. Shared logical inputs appear once in this list. */
    const BnTransformerBF16KDAWorkspace*s=&a->kda;
    BnGPUResidentView v[40]={a->mhc.residual,a->mhc.dot,a->mhc.sqrsum,a->mhc.pre,
        a->mhc.post,a->mhc.comb,a->mhc.collapsed,a->mhc.out,
        s->projected,s->qkv,s->tail,s->fa_ga,s->beta,s->fa,s->ga,s->g1,s->g2,s->conv,s->core,s->gated,
        a->initial_history,a->next_history,a->initial_state,a->next_state,a->branch_out,
        f->residual,f->dot,f->sqrsum,f->pre,f->post,f->comb,f->collapsed,f->out,
        p->gate_up,p->activated,p->out};
    size_t n=(size_t)nt,ns=n*f->n_splits;
    size_t counts[]={n*dim*4,ns*24,ns,n*4,n*4,n*16,n*dim,n*dim};
    for(int i=0;i<8;++i){int type=(i>=1&&i<=5)?bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32):BF16;
        if(v[25+i].request!=request||v[25+i].type!=type||v[25+i].count!=counts[i]||
           v[25+i].offset>SIZE_MAX-counts[i])return -1;}
    int nv=36;
    if(a->has_previous){v[nv++]=a->previous_residual;v[nv++]=a->previous_branch;v[nv++]=a->previous_post;v[nv++]=a->previous_comb;}
    for(int i=0;i<nv;++i)for(int j=0;j<i;++j)
        if(v[i].slot==v[j].slot&&v[i].offset<v[j].offset+v[j].count&&
           v[j].offset<v[i].offset+v[i].count)return -1;
    BnGPUMHCFusedResidentPlan next={.next=*f,.previous_residual=a->mhc.residual,
        .previous_branch=a->branch_out,.previous_post=a->mhc.post,.previous_comb=a->mhc.comb};
    if(bn_transformer_bf16_mhc_kda_attention(request,a)||
       g->mhc_fused_post_pre_bf16_resident(g->ctx,request,&next))return -1;
    return bn_transformer_bf16_dense_ffn(w,request,nt,f->out,p->gate_up,p->activated,p->out);
}

static int bf16_moe_validate(void*request,const BnTransformerBF16MoEPlan*p) {
    if(!request||!p||!p->weights||!p->source||!p->host_state||p->layer<0||
       p->n_tokens<1||(p->prefill!=0&&p->prefill!=1)||p->bank_capacity<1||p->bank_capacity>64)return -1;
    const BnBackendBF16MoEWeights*w=p->weights;const BnMoEExpertMap*m=w->source_map;
    BnGPUBackend*g=w->gpu;size_t n=(size_t)p->n_tokens;
    if(!g||!m||!w->router||!w->bias||p->layer!=w->layer||!w->gate_bytes||!w->down_bytes||
       w->gate_bytes>(SIZE_MAX-w->down_bytes)/2||w->n_experts!=288||w->k!=8||w->dim<=0||w->dim%256||
       w->hidden_dim<=0||w->hidden_dim%256||w->shared.gpu!=g||w->shared.dim!=w->dim||
       n>INT_MAX/8/(size_t)w->dim||n>INT_MAX/8/(size_t)w->hidden_dim||
       !isfinite(w->route_scale)||w->route_scale<=0||!isfinite(w->clamp_limit)||w->clamp_limit<0||
       w->gate_bytes!=m->expert_gate_bytes||w->gate_bytes!=m->expert_up_bytes||w->down_bytes!=m->expert_down_bytes||
       w->gate_type!=m->gate_type||w->gate_type!=m->up_type||w->down_type!=m->down_type||
       !g->linear_bf16_f32_weight_resident||!g->moe_biased_sigmoid_f32_resident||
       !g->moe_group_bf16_resident||!g->moe_reduce_bf16_resident||!g->add_bf16_resident||
       !g->moe_bank_prepare_input||!g->moe_bank_acquire||!g->moe_bank_submit||!g->moe_bank_cancel||
       !g->resident_read||!g->resident_zero||(!bn_moe_io_has_mmap(p->source)&&!bn_moe_io_has_pread(p->source))||
       bf16_dense_ffn_validate(&w->shared,request,p->n_tokens,p->x,p->shared_gate_up,p->shared_activated,p->shared_out))return -1;
    BnGPUResidentView v[]={p->x,p->cast,p->logits,p->route_weights,p->indices,p->metadata,p->cursors,
        p->slots,p->inverse,p->gathered,p->gate,p->up,p->down,p->weighted,p->routed,
        p->shared_gate_up,p->shared_activated,p->shared_out,p->out};
    size_t counts[]={n*w->dim,n*w->dim,n*288,n*8,n*8,293,288,n*8,n*8,n*8*w->dim,
        n*8*w->hidden_dim,n*8*w->hidden_dim,n*8*w->dim,n*8*w->dim,n*w->dim,
        n*2*w->shared.hidden_dim,n*w->shared.hidden_dim,n*w->dim,n*w->dim};
    for(size_t i=0;i<sizeof(v)/sizeof(v[0]);++i){int type=i==1||i==2||i==3?bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32):
            i>=4&&i<=8?bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32):BF16;
        if(v[i].request!=request||v[i].type!=type||v[i].count!=counts[i]||v[i].offset>SIZE_MAX-counts[i])return -1;
        for(size_t j=0;j<i;++j)if(v[i].slot==v[j].slot&&v[i].offset<v[j].offset+v[j].count&&
            v[j].offset<v[i].offset+v[i].count)return -1;
    }
    return 0;
}
int bn_transformer_bf16_moe_begin(void*request,const BnTransformerBF16MoEPlan*p,BnTransformerBF16MoEWork*s) {
    if(!s||s->pending||s->failed||bf16_moe_validate(request,p))return -1;
    const BnBackendBF16MoEWeights*w=p->weights;BnGPUBackend*g=w->gpu;int nt=p->n_tokens;
    s->active=*p;s->pending=1;s->next_expert=0;s->bank=0;s->generation=0;
    BnGPUBF16F32LinearResidentPlan router={.n_tokens=nt,.rows=288,.cols=w->dim,
        .weight=w->router,.x=p->x,.cast=p->cast,.out=p->logits};
    BnGPUBiasedSigmoidF32ResidentPlan top={.n_tokens=nt,.n_experts=288,.k=8,.route_scale=w->route_scale,
        .bias=w->bias,.logits=p->logits,.weights=p->route_weights,.indices=p->indices};
    BnGPUMoEGroupBF16ResidentPlan group={.n_tokens=nt,.n_experts=288,.k=8,.dim=w->dim,
        .indices=p->indices,.x=p->x,.offsets=bn_gpu_resident_view_slice(p->metadata,0,289),
        .status=bn_gpu_resident_view_slice(p->metadata,292,1),.cursors=p->cursors,.slots=p->slots,
        .inverse=p->inverse,.gathered=p->gathered};
    if(g->resident_zero(request,p->metadata)||g->linear_bf16_f32_weight_resident(g->ctx,request,&router)||
       g->moe_biased_sigmoid_f32_resident(g->ctx,request,&top)||g->moe_group_bf16_resident(g->ctx,request,&group)||
       g->resident_read(request,p->metadata,s->offsets))goto fail;
    s->metadata_read_bytes+=293*sizeof(int32_t);
    if(s->offsets[0]||s->offsets[288]!=nt*8||s->offsets[292])goto fail;
    for(int e=0;e<288;++e)if(s->offsets[e]<0||s->offsets[e+1]<s->offsets[e]||s->offsets[e+1]-s->offsets[e]>nt)goto fail;
    BnGPUMoEBankInputPlan prep={nt,p->x,p->slots,group.status};
    if(bn_transformer_bf16_dense_ffn(&w->shared,request,nt,p->x,p->shared_gate_up,p->shared_activated,p->shared_out)||
       g->moe_bank_prepare_input(g->ctx,request,&prep,&s->generation)||!s->generation)goto fail;
    return 0;
fail:
    s->failed=1;return -1;
}
static int bf16_moe_source_copy(const BnTransformerBF16MoEPlan*p,int expert,void*target) {
    const BnBackendBF16MoEWeights*w=p->weights;BnMoEState*ms=p->host_state;
    uint8_t*old[]={ms->buf,ms->buf2,ms->buf5};size_t sizes[]={ms->buf_size,ms->buf2_size,ms->buf5_size};
    uint8_t*raw=target;ms->buf=raw;ms->buf_size=w->gate_bytes;ms->buf2=raw+w->gate_bytes;
    ms->buf2_size=w->gate_bytes;ms->buf5=raw+w->gate_bytes*2;ms->buf5_size=w->down_bytes;
    BnMoEHostExpert host={0};int status=bn_moe_host_expert_acquire(p->source,ms,w->source_map,p->layer,expert,!p->prefill,&host);
    if(!status)for(int role=0;role<3;++role){void*dst=raw+(size_t)role*w->gate_bytes;
        if(!host.projection[role]){status=-1;break;}
        if(dst!=host.projection[role])memcpy(dst,host.projection[role],role==2?w->down_bytes:w->gate_bytes);}
    bn_moe_host_expert_release(&host);ms->buf=old[0];ms->buf2=old[1];ms->buf5=old[2];
    ms->buf_size=sizes[0];ms->buf2_size=sizes[1];ms->buf5_size=sizes[2];return status;
}
int bn_transformer_bf16_moe_submit(void*request,BnTransformerBF16MoEWork*s) {
    if(!s||!s->pending||s->failed||s->active.x.request!=request)return -1;
    const BnTransformerBF16MoEPlan*p=&s->active;const BnBackendBF16MoEWeights*w=p->weights;BnGPUBackend*g=w->gpu;
    while(s->next_expert<288){int e=s->next_expert;while(e<288&&s->offsets[e]==s->offsets[e+1])++e;
        if(e==288){s->next_expert=e;break;}
        BnGPUMoEBankLease lease={0};int status=g->moe_bank_acquire(request,s->bank%2,w->gate_type,w->down_type,&lease);
        if(status==BN_GPU_MOE_BANK_BUSY)return status;
        if(status)goto fail;
        if(lease.capacity<(unsigned)p->bank_capacity||lease.gate_bytes!=w->gate_bytes||lease.down_bytes!=w->down_bytes||
           lease.expert_bytes!=2*w->gate_bytes+w->down_bytes){g->moe_bank_cancel(request,&lease);goto fail;}
        BnGPUMoEBankItem items[64];int count=0;
        while(e<288&&count<p->bank_capacity){int rows=s->offsets[e+1]-s->offsets[e];
            if(rows){items[count]=(BnGPUMoEBankItem){e,s->offsets[e],rows};
                if(bf16_moe_source_copy(p,e,(unsigned char*)lease.data+(size_t)count*lease.expert_bytes)){
                    g->moe_bank_cancel(request,&lease);goto fail;}++count;}++e;}
        BnGPUMoEBankResidentPlan bank={.lease=lease,.total_tokens=p->n_tokens,.n_experts=count,.experts=items,
            .x=p->gathered,.gate=p->gate,.up=p->up,.out=p->down,.prepared_input=s->generation,
            .grouped=1,.clamped=1,.clamp_limit=w->clamp_limit};
        if(g->moe_bank_submit(g->ctx,request,&bank)){g->moe_bank_cancel(request,&lease);goto fail;}
        s->next_expert=e;++s->bank;++s->submitted_banks;s->submitted_experts+=(size_t)count;
    }
    BnGPUMoEReduceBF16ResidentPlan red={.n_tokens=p->n_tokens,.k=8,.dim=w->dim,.grouped=p->down,.inverse=p->inverse,
        .weights=p->route_weights,.weighted=p->weighted,.out=p->routed,.status=bn_gpu_resident_view_slice(p->metadata,292,1)};
    BnGPUAddBF16ResidentPlan add={.count=(size_t)p->n_tokens*w->dim,.x=p->routed,.y=p->shared_out,.out=p->out};
    if(g->moe_reduce_bf16_resident(g->ctx,request,&red)||g->add_bf16_resident(g->ctx,request,&add))goto fail;
    s->pending=0;return 0;
fail:
    s->failed=1;return -1;
}

static int bf16_indexer_validate(void *request,const BnTransformerBF16IndexerPlan *p) {
    if(!request||!p||!p->weights||p->n_tokens<1||p->position<0||p->position>INT_MAX-p->n_tokens)return -1;
    const BnBackendBF16IndexerWeights *w=p->weights;BnGPUBackend *g=w->gpu;size_t n=(size_t)p->n_tokens;
    if(!g||p->layer!=w->layer||!w->query.raw||w->query.decoded||!w->key.raw||w->key.decoded||!w->gate.raw||w->gate.decoded||
       !bn_backend_quant_uses_native_quant(w->query.type)||!bn_backend_quant_uses_native_quant(w->key.type)||
       !bn_backend_quant_uses_native_quant(w->gate.type)||!w->head_weights||!w->norm||!w->bias||!w->ape||
       w->dim<1||w->dim%32||w->q_rank<1||w->q_rank%32||w->heads!=32||w->head_dim!=128||w->pool_size!=4||
       w->norm_eps!=1e-6f||w->weight_scale!=.015625f||n>INT_MAX/(size_t)w->dim||n>INT_MAX/(size_t)w->q_rank||n>INT_MAX/((size_t)w->heads*w->head_dim)||
       !g->linear_a8_bf16_resident||!g->linear_bf16_f32_weight_resident||!g->layernorm128_f32_bf16_resident||
       !g->fwht128_fp8_resident||!g->indexer_weight_scale_resident||!g->kpool_fp8_resident)return -1;
    int hd=w->head_dim,qw=w->heads*hd,pool_size=w->pool_size;
    size_t groups=(p->position+p->n_tokens)/pool_size-p->position/pool_size;
    BnGPUResidentView v[]={p->x,p->qr,p->initial_tail,p->next_tail,p->q,p->key_raw,p->key,p->gate,p->cast,p->head_raw,
        p->q_fp8,p->q_scales,p->head_weights,p->compressed,p->compressed_scales};
    size_t counts[]={n*w->dim,n*w->q_rank,(size_t)2*pool_size*hd,(size_t)2*pool_size*hd,n*qw,n*hd,n*hd,n*hd,n*w->dim,n*w->heads,n*qw,n*w->heads,n*w->heads,groups*hd,groups};
    for(int i=0;i<15;++i){int type=i==8||i==9||i==11||i==12||i==14?bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32):
            i==10||i==13?bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_BYTES):BF16;
        if(!counts[i]){if(v[i].request||v[i].count)return -1;continue;}
        if(v[i].request!=request||v[i].type!=type||v[i].count!=counts[i]||v[i].offset>SIZE_MAX-counts[i])return -1;
        for(int j=0;j<i;++j)if(counts[j]&&v[i].slot==v[j].slot&&v[i].offset<v[j].offset+v[j].count&&v[j].offset<v[i].offset+v[i].count)return -1;}
    return 0;
}
int bn_transformer_bf16_indexer(void *request,const BnTransformerBF16IndexerPlan *p) {
    if(bf16_indexer_validate(request,p))return -1;
    const BnBackendBF16IndexerWeights *w=p->weights;BnGPUBackend *g=w->gpu;
    int hd=w->head_dim,qw=w->heads*hd;
    BnGPUBF16LinearResidentPlan q={.n_tokens=p->n_tokens,.rows=qw,.cols=w->q_rank,.weight=w->query,.x=p->qr,.out=p->q};
    BnGPUFWHT128FP8ResidentPlan fp8={.rows=p->n_tokens*w->heads,.x=p->q,.out=p->q_fp8,.scales=p->q_scales};
    BnGPUBF16LinearResidentPlan k={.n_tokens=p->n_tokens,.rows=hd,.cols=w->dim,.weight=w->key,.x=p->x,.out=p->key_raw};
    BnGPULayerNorm128F32BF16ResidentPlan norm={.n_tokens=p->n_tokens,.eps=w->norm_eps,.weight=w->norm,.bias=w->bias,.x=p->key_raw,.out=p->key};
    BnGPUBF16F32LinearResidentPlan head={.n_tokens=p->n_tokens,.rows=w->heads,.cols=w->dim,.weight=w->head_weights,.x=p->x,.cast=p->cast,.out=p->head_raw};
    BnGPUIndexerWeightScaleResidentPlan scale={.rows=p->n_tokens*w->heads,.factor=w->weight_scale,.weights=p->head_raw,.query_scales=p->q_scales,.out=p->head_weights};
    BnGPUBF16LinearResidentPlan gate={.n_tokens=p->n_tokens,.rows=hd,.cols=w->dim,.weight=w->gate,.x=p->x,.out=p->gate};
    BnGPUKPoolFP8ResidentPlan pool={.n_tokens=p->n_tokens,.position=p->position,.ape=w->ape,.key=p->key,.gate=p->gate,
        .initial_tail=p->initial_tail,.next_tail=p->next_tail,.out=p->compressed,.scales=p->compressed_scales};
    if(g->linear_a8_bf16_resident(g->ctx,request,&q)||g->fwht128_fp8_resident(g->ctx,request,&fp8)||
       g->linear_a8_bf16_resident(g->ctx,request,&k)||g->layernorm128_f32_bf16_resident(g->ctx,request,&norm)||
       g->linear_bf16_f32_weight_resident(g->ctx,request,&head)||g->indexer_weight_scale_resident(g->ctx,request,&scale)||
       g->linear_a8_bf16_resident(g->ctx,request,&gate)||g->kpool_fp8_resident(g->ctx,request,&pool))return -1;
    return 0;
}

static int bf16_indexer_cache_validate(void *request,const BnTransformerBF16IndexerCachePlan *p) {
    if(!p||bf16_indexer_validate(request,&p->indexer)||p->num_cache_blocks<1||p->num_cache_blocks>INT_MAX/64||
       (size_t)p->num_cache_blocks>SIZE_MAX/8448||p->logical_blocks<1||p->cache.offset%4)return -1;
    const BnTransformerBF16IndexerPlan *i=&p->indexer;BnGPUBackend *g=i->weights->gpu;
    size_t end=(i->position+i->n_tokens)/i->weights->pool_size,groups=end-i->position/i->weights->pool_size;
    if(end>INT_MAX/128u||(groups&&i->compressed.offset%4)||(end&&p->gathered.offset%4)||
       (size_t)p->logical_blocks<(end+63)/64||!g->indexer_fp8_append_resident||!g->indexer_fp8_gather_resident)return -1;
    BnGPUResidentView v[]={i->x,i->qr,i->initial_tail,i->next_tail,i->q,i->key_raw,i->key,i->gate,i->cast,i->head_raw,
        i->q_fp8,i->q_scales,i->head_weights,i->compressed,i->compressed_scales,p->cache,p->block_table,p->slots,p->gathered,p->scales,p->status};
    size_t counts[]={(size_t)p->num_cache_blocks*8448,(size_t)p->logical_blocks,groups,end*128,end,1};
    int types[]={BN_BACKEND_SCALAR_BYTES,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_BYTES,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_I32};
    for(int j=15;j<21;++j){size_t count=counts[j-15];if(!count){if(v[j].request||v[j].count)return -1;continue;}
        if(v[j].request!=request||v[j].count!=count||v[j].type!=bn_backend_quant_scalar_type((BnBackendScalarKind)types[j-15])||v[j].offset>SIZE_MAX-count)return -1;
        for(int k=0;k<j;++k)if(v[k].count&&v[j].slot==v[k].slot&&v[j].offset<v[k].offset+v[k].count&&v[k].offset<v[j].offset+count)return -1;
    }
    return 0;
}
int bn_transformer_bf16_indexer_cache(void *request,const BnTransformerBF16IndexerCachePlan *p) {
    if(bf16_indexer_cache_validate(request,p))return -1;
    const BnTransformerBF16IndexerPlan *i=&p->indexer;BnGPUBackend *g=i->weights->gpu;
    size_t end=(i->position+i->n_tokens)/i->weights->pool_size,groups=end-i->position/i->weights->pool_size;
    BnGPUIndexerFP8AppendResidentPlan append={.n_rows=(int)groups,.num_cache_blocks=p->num_cache_blocks,
        .values=i->compressed,.scales=i->compressed_scales,.slots=p->slots,.cache=p->cache};
    BnGPUIndexerFP8GatherResidentPlan gather={.n_rows=(int)end,.num_cache_blocks=p->num_cache_blocks,.logical_blocks=p->logical_blocks,
        .cache=p->cache,.block_table=p->block_table,.out=p->gathered,.scales=p->scales,.status=p->status};
    if(bn_transformer_bf16_indexer(request,i)||g->indexer_fp8_append_resident(g->ctx,request,&append)||
       g->indexer_fp8_gather_resident(g->ctx,request,&gather))return -1;
    return 0;
}

static int bf16_indexer_scores_validate(void *request,const BnTransformerBF16IndexerScoresPlan *p) {
    if(!p||bf16_indexer_cache_validate(request,&p->cache))return -1;
    const BnTransformerBF16IndexerCachePlan *c=&p->cache;
    const BnTransformerBF16IndexerPlan *i=&c->indexer;BnGPUBackend *g=i->weights->gpu;
    size_t n=(size_t)i->n_tokens,end=(i->position+i->n_tokens)/i->weights->pool_size;
    if(!g->indexer_fp8_score_resident||i->q_fp8.offset%4||
       (end&&(n>SIZE_MAX/sizeof(float)/end||n>(size_t)INT_MAX/((end+63)/64))))return -1;
    size_t count=n*end;
    if(!count){if(p->scores.request||p->scores.count)return -1;}
    else {
        if(p->scores.request!=request||p->scores.count!=count||p->scores.offset>SIZE_MAX-count||
           p->scores.type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32))return -1;
        BnGPUResidentView v[]={i->x,i->qr,i->initial_tail,i->next_tail,i->q,i->key_raw,i->key,i->gate,i->cast,i->head_raw,
            i->q_fp8,i->q_scales,i->head_weights,i->compressed,i->compressed_scales,c->cache,c->block_table,c->slots,c->gathered,c->scales,c->status};
        for(int j=0;j<21;++j)if(v[j].count&&v[j].slot==p->scores.slot&&
            p->scores.offset<v[j].offset+v[j].count&&v[j].offset<p->scores.offset+count)return -1;
    }
    return 0;
}
int bn_transformer_bf16_indexer_scores(void *request,const BnTransformerBF16IndexerScoresPlan *p) {
    if(bf16_indexer_scores_validate(request,p))return -1;
    const BnTransformerBF16IndexerCachePlan *c=&p->cache;
    const BnTransformerBF16IndexerPlan *i=&c->indexer;BnGPUBackend *g=i->weights->gpu;
    size_t end=(i->position+i->n_tokens)/i->weights->pool_size;
    BnGPUIndexerFP8ScoreResidentPlan score={.n_tokens=i->n_tokens,.position=i->position,.n_keys=(int)end,
        .q=i->q_fp8,.keys=c->gathered,.scales=c->scales,.weights=i->head_weights,.scores=p->scores};
    if(bn_transformer_bf16_indexer_cache(request,c)||g->indexer_fp8_score_resident(g->ctx,request,&score))return -1;
    return 0;
}

int bn_transformer_bf16_indexer_select(void *request,const BnTransformerBF16IndexerSelectPlan *p) {
    if(!p||(p->short_prefill!=0&&p->short_prefill!=1))return -1;
    const BnTransformerBF16IndexerScoresPlan *s=&p->indexed;
    if(p->short_prefill){if(bf16_indexer_cache_validate(request,&s->cache)||s->scores.request||s->scores.count)return -1;}
    else if(bf16_indexer_scores_validate(request,s))return -1;
    const BnTransformerBF16IndexerCachePlan *c=&s->cache;const BnTransformerBF16IndexerPlan *i=&c->indexer;
    BnGPUBackend *g=i->weights->gpu;size_t n=(size_t)i->n_tokens;
    if(!g->indexer_select_resident||n>INT_MAX/2051u||(p->short_prefill&&i->position+i->n_tokens>2048))return -1;
    BnGPUResidentView v[]={i->x,i->qr,i->initial_tail,i->next_tail,i->q,i->key_raw,i->key,i->gate,i->cast,i->head_raw,
        i->q_fp8,i->q_scales,i->head_weights,i->compressed,i->compressed_scales,c->cache,c->block_table,c->slots,c->gathered,c->scales,c->status,s->scores,p->pools,p->out};
    size_t counts[]={n*512,n*2051};
    for(int j=22;j<24;++j){if(v[j].request!=request||v[j].count!=counts[j-22]||v[j].offset>SIZE_MAX-v[j].count||
        v[j].type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32))return -1;
        for(int k=0;k<j;++k)if(v[k].count&&v[j].slot==v[k].slot&&v[j].offset<v[k].offset+v[k].count&&v[k].offset<v[j].offset+v[j].count)return -1;
    }
    BnGPUIndexerSelectResidentPlan select={.n_tokens=i->n_tokens,.position=i->position,.n_keys=(i->position+i->n_tokens)/4,
        .short_prefill=p->short_prefill,.scores=s->scores,.pools=p->pools,.out=p->out,.status=c->status};
    if((p->short_prefill?bn_transformer_bf16_indexer_cache(request,c):bn_transformer_bf16_indexer_scores(request,s))||
        g->indexer_select_resident(g->ctx,request,&select))return -1;
    return 0;
}

static int bf16_indexer_decode_validate(void *request,const BnTransformerBF16IndexerDecodePlan *p) {
    if(!p||bf16_indexer_validate(request,&p->indexer)||(p->short_decode!=0&&p->short_decode!=1)||
       p->num_cache_blocks<1||p->num_cache_blocks>INT_MAX/64||(size_t)p->num_cache_blocks>SIZE_MAX/8448||
       p->logical_blocks<1)return -1;
    const BnTransformerBF16IndexerPlan *i=&p->indexer;BnGPUBackend *g=i->weights->gpu;
    size_t n=(size_t)i->n_tokens,end=(i->position+i->n_tokens)/4,groups=end-i->position/4;
    if(!g->indexer_fp8_append_resident||!g->indexer_select_resident||
       (!p->short_decode&&!g->indexer_fp8_paged_score_resident)||n>INT_MAX/2051u||end>INT_MAX/128u||
       (size_t)p->logical_blocks<(end+63)/64||p->cache.offset%4||i->q_fp8.offset%4||
       (groups&&i->compressed.offset%4)||(p->short_decode&&i->position+i->n_tokens>2048)||
       (end&&(n>SIZE_MAX/sizeof(float)/end||n>(size_t)INT_MAX/((end+63)/64))))return -1;
    BnGPUResidentView v[]={i->x,i->qr,i->initial_tail,i->next_tail,i->q,i->key_raw,i->key,i->gate,i->cast,i->head_raw,
        i->q_fp8,i->q_scales,i->head_weights,i->compressed,i->compressed_scales,p->cache,p->block_table,p->slots,p->scores,p->pools,p->out,p->status};
    size_t counts[]={(size_t)p->num_cache_blocks*8448,(size_t)p->logical_blocks,groups,p->short_decode?0:n*end,n*512,n*2051,1};
    BnBackendScalarKind types[]={BN_BACKEND_SCALAR_BYTES,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32};
    for(int j=15;j<22;++j){size_t count=counts[j-15];if(!count){if(v[j].request||v[j].count)return -1;continue;}
        if(v[j].request!=request||v[j].count!=count||v[j].type!=bn_backend_quant_scalar_type(types[j-15])||v[j].offset>SIZE_MAX-count)return -1;
        for(int k=0;k<j;++k)if(v[k].count&&v[j].slot==v[k].slot&&v[j].offset<v[k].offset+v[k].count&&v[k].offset<v[j].offset+count)return -1;
    }
    return 0;
}
int bn_transformer_bf16_indexer_decode(void *request,const BnTransformerBF16IndexerDecodePlan *p) {
    if(bf16_indexer_decode_validate(request,p))return -1;
    const BnTransformerBF16IndexerPlan *i=&p->indexer;BnGPUBackend *g=i->weights->gpu;
    size_t end=(i->position+i->n_tokens)/4,groups=end-i->position/4;
    BnGPUIndexerFP8AppendResidentPlan append={.n_rows=(int)groups,.num_cache_blocks=p->num_cache_blocks,
        .values=i->compressed,.scales=i->compressed_scales,.slots=p->slots,.cache=p->cache};
    BnGPUIndexerFP8PagedScoreResidentPlan score={.n_tokens=i->n_tokens,.position=i->position,.n_keys=(int)end,
        .num_cache_blocks=p->num_cache_blocks,.logical_blocks=p->logical_blocks,.q=i->q_fp8,.weights=i->head_weights,
        .cache=p->cache,.block_table=p->block_table,.scores=p->scores,.status=p->status};
    BnGPUIndexerSelectResidentPlan select={.n_tokens=i->n_tokens,.position=i->position,.n_keys=(int)end,
        .short_prefill=p->short_decode,.scores=p->scores,.pools=p->pools,.out=p->out,.status=p->status};
    if(bn_transformer_bf16_indexer(request,i)||g->indexer_fp8_append_resident(g->ctx,request,&append)||
       (!p->short_decode&&g->indexer_fp8_paged_score_resident(g->ctx,request,&score))||
       g->indexer_select_resident(g->ctx,request,&select))return -1;
    return 0;
}

static int bf16_mla_weights_valid(const BnBackendBF16MLAWeights *w,int layer,int nt) {
    if(!w||!w->gpu||nt<1||layer!=w->layer||w->dim<32||w->q_rank<32||w->kv_rank<32||
       w->dim%32||w->q_rank%32||w->kv_rank%32||w->heads<1||w->heads>128||w->key_dim<32||w->key_dim%32||
       w->value_dim<32||w->value_dim%32||w->key_dim>INT_MAX/w->heads||w->value_dim>INT_MAX/w->heads||w->kv_rank>INT_MAX/w->heads||
       nt>INT_MAX/w->dim||nt>INT_MAX/w->q_rank||nt>INT_MAX/(w->heads*w->kv_rank)||
       nt>INT_MAX/(w->heads*w->key_dim)||nt>INT_MAX/(w->heads*w->value_dim)||!isfinite(w->norm_eps)||w->norm_eps<=0||
       !w->q_a.raw||w->q_a.decoded||!w->q_b.raw||w->q_b.decoded||!w->kv_a.raw||w->kv_a.decoded||!w->output.raw||w->output.decoded||
       !bn_backend_quant_uses_native_quant(w->q_a.type)||!bn_backend_quant_uses_native_quant(w->q_b.type)||
       !bn_backend_quant_uses_native_quant(w->kv_a.type)||!bn_backend_quant_uses_native_quant(w->output.type)||
       !w->q_norm||!w->kv_norm||!w->key||!w->value)return 0;
    return 1;
}
static int bf16_mla_views_valid(void *request,const BnGPUResidentView *v,const size_t *counts,int count) {
    for(int j=0;j<count;++j){if(v[j].request!=request||v[j].type!=BF16||v[j].count!=counts[j]||v[j].offset>SIZE_MAX-counts[j]||v[j].offset%8)return 0;
        for(int k=0;k<j;++k)if(v[j].slot==v[k].slot&&v[j].offset<v[k].offset+v[k].count&&v[k].offset<v[j].offset+v[j].count)return 0;}
    return 1;
}
static int bf16_mla_prepare_validate(void *request,const BnTransformerBF16MLAPreparePlan *p) {
    if(!request||!p||!bf16_mla_weights_valid(p->weights,p->layer,p->n_tokens))return -1;
    const BnBackendBF16MLAWeights *w=p->weights;BnGPUBackend *g=w->gpu;size_t n=(size_t)p->n_tokens;
    if(w->q_rank!=1536||w->kv_rank!=512||w->norm_eps!=1e-5f||!g->linear_a8_bf16_resident||!g->rmsnorm_bf16_resident||!g->head_linear_bf16_resident)return -1;
    BnGPUResidentView v[]={p->x,p->q_raw,p->qr,p->kv_raw,p->kv,p->q,p->query};
    size_t counts[]={n*w->dim,n*w->q_rank,n*w->q_rank,n*w->kv_rank,n*w->kv_rank,n*w->heads*w->key_dim,n*w->heads*w->kv_rank};
    if(!bf16_mla_views_valid(request,v,counts,7))return -1;
    return 0;
}
int bn_transformer_bf16_mla_prepare(void *request,const BnTransformerBF16MLAPreparePlan *p) {
    if(bf16_mla_prepare_validate(request,p))return -1;
    const BnBackendBF16MLAWeights *w=p->weights;BnGPUBackend *g=w->gpu;
    BnGPUBF16LinearResidentPlan qa={.n_tokens=p->n_tokens,.rows=w->q_rank,.cols=w->dim,.weight=w->q_a,.x=p->x,.out=p->q_raw};
    BnGPUBF16LinearResidentPlan kv={.n_tokens=p->n_tokens,.rows=w->kv_rank,.cols=w->dim,.weight=w->kv_a,.x=p->x,.out=p->kv_raw};
    BnGPURMSNormResidentPlan qnorm={.reduction=BN_GPU_RMSNORM_REDUCTION_VECTOR8_F32,.n_tokens=p->n_tokens,.dim=w->q_rank,.eps=w->norm_eps,.weight=w->q_norm,.x=p->q_raw,.out=p->qr};
    BnGPURMSNormResidentPlan knorm={.reduction=BN_GPU_RMSNORM_REDUCTION_VECTOR8_F32,.n_tokens=p->n_tokens,.dim=w->kv_rank,.eps=w->norm_eps,.weight=w->kv_norm,.x=p->kv_raw,.out=p->kv};
    BnGPUBF16LinearResidentPlan qb={.n_tokens=p->n_tokens,.rows=w->heads*w->key_dim,.cols=w->q_rank,.weight=w->q_b,.x=p->qr,.out=p->q};
    BnGPUHeadLinearBF16ResidentPlan absorb={.n_tokens=p->n_tokens,.heads=w->heads,.rows=w->kv_rank,.cols=w->key_dim,.weight=w->key,.x=p->q,.out=p->query};
    if(g->linear_a8_bf16_resident(g->ctx,request,&qa)||g->linear_a8_bf16_resident(g->ctx,request,&kv)||
       g->rmsnorm_bf16_resident(g->ctx,request,&qnorm)||g->rmsnorm_bf16_resident(g->ctx,request,&knorm)||
       g->linear_a8_bf16_resident(g->ctx,request,&qb)||g->head_linear_bf16_resident(g->ctx,request,&absorb))return -1;
    return 0;
}
int bn_transformer_bf16_mla_output(void *request,const BnTransformerBF16MLAOutputPlan *p) {
    if(!request||!p||!bf16_mla_weights_valid(p->weights,p->layer,p->n_tokens))return -1;
    const BnBackendBF16MLAWeights *w=p->weights;BnGPUBackend *g=w->gpu;size_t n=(size_t)p->n_tokens;
    if(!g->head_linear_bf16_resident||!g->linear_a8_bf16_resident)return -1;
    BnGPUResidentView v[]={p->context,p->values,p->out};size_t counts[]={n*w->heads*w->kv_rank,n*w->heads*w->value_dim,n*w->dim};
    if(!bf16_mla_views_valid(request,v,counts,3))return -1;
    BnGPUHeadLinearBF16ResidentPlan value={.n_tokens=p->n_tokens,.heads=w->heads,.rows=w->value_dim,.cols=w->kv_rank,.weight=w->value,.x=p->context,.out=p->values};
    BnGPUBF16LinearResidentPlan output={.n_tokens=p->n_tokens,.rows=w->dim,.cols=w->heads*w->value_dim,.weight=w->output,.x=p->values,.out=p->out};
    if(g->head_linear_bf16_resident(g->ctx,request,&value)||g->linear_a8_bf16_resident(g->ctx,request,&output))return -1;
    return 0;
}

static int bf16_mla_cache_validate(void *request,const BnTransformerBF16MLACachePlan *p) {
    if(!p||bf16_mla_prepare_validate(request,&p->prepare)||p->num_cache_blocks<1||
       p->num_cache_blocks>INT_MAX/64||(size_t)p->num_cache_blocks>SIZE_MAX/(64u*656)||p->cache.offset%16)return -1;
    const BnTransformerBF16MLAPreparePlan *q=&p->prepare;BnGPUBackend *g=q->weights->gpu;
    if(!g->mla_fp8_cache_append_resident)return -1;
    BnGPUResidentView v[]={q->x,q->q_raw,q->qr,q->kv_raw,q->kv,q->q,q->query,p->slots,p->cache};
    size_t counts[]={(size_t)q->n_tokens,(size_t)p->num_cache_blocks*64*656};
    BnBackendScalarKind types[]={BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_BYTES};
    for(int j=7;j<9;++j){if(v[j].request!=request||v[j].count!=counts[j-7]||v[j].offset>SIZE_MAX-v[j].count||
        v[j].type!=bn_backend_quant_scalar_type(types[j-7]))return -1;
        for(int k=0;k<j;++k)if(v[j].slot==v[k].slot&&v[j].offset<v[k].offset+v[k].count&&v[k].offset<v[j].offset+v[j].count)return -1;}
    return 0;
}
int bn_transformer_bf16_mla_cache(void *request,const BnTransformerBF16MLACachePlan *p) {
    if(bf16_mla_cache_validate(request,p))return -1;
    const BnTransformerBF16MLAPreparePlan *q=&p->prepare;BnGPUBackend *g=q->weights->gpu;
    BnGPUMlaFP8CacheAppendResidentPlan append={.n_tokens=q->n_tokens,.num_cache_blocks=p->num_cache_blocks,.kv=q->kv,.slots=p->slots,.cache=p->cache};
    if(bn_transformer_bf16_mla_prepare(request,q)||g->mla_fp8_cache_append_resident(g->ctx,request,&append))return -1;
    return 0;
}

static int bf16_mla_attention_validate(void *request,const BnTransformerBF16MLAAttentionPlan *p) {
    if(!request||!p||!bf16_mla_weights_valid(p->weights,p->layer,p->attention.n_tokens))return -1;
    const BnBackendBF16MLAWeights *w=p->weights;const BnGPUMlaFP8AttentionResidentPlan *a=&p->attention;
    BnGPUBackend *g=w->gpu;
    if(w->heads!=64||w->kv_rank!=512||w->key_dim!=256||w->value_dim!=256||w->scale!=.0625f||
       !g->mla_fp8_attention_resident||!g->head_linear_bf16_resident||!g->linear_a8_bf16_resident||
       a->n_tokens>INT_MAX/32768||a->position<0||a->position>INT_MAX-a->n_tokens||
       a->num_cache_blocks<1||a->num_cache_blocks>INT_MAX/64||a->logical_blocks<1||
       a->logical_blocks>INT_MAX/64||a->position+a->n_tokens>a->logical_blocks*64||
       (size_t)a->num_cache_blocks>SIZE_MAX/(64u*656))return -1;
    size_t n=(size_t)a->n_tokens,split=n<=64?n*64*34:0;
    BnGPUResidentView v[]={a->query,a->cache,a->selection,a->block_table,a->physical,a->out,
        a->lse,a->partial,a->partial_lse,a->status,p->values,p->out};
    size_t counts[]={n*64*512,(size_t)a->num_cache_blocks*64*656,n*2051,(size_t)a->logical_blocks,
        n*2176,n*64*512,n*64,split*512,split,1,n*64*256,n*w->dim};
    BnBackendScalarKind kinds[]={BN_BACKEND_SCALAR_BF16,BN_BACKEND_SCALAR_BYTES,BN_BACKEND_SCALAR_I32,
        BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_BF16,BN_BACKEND_SCALAR_F32,
        BN_BACKEND_SCALAR_BF16,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_BF16,
        BN_BACKEND_SCALAR_BF16};
    for(int j=0;j<12;++j){
        if(!counts[j]){if(v[j].request||v[j].count)return -1;continue;}
        size_t width=bn_backend_quant_scalar_bytes(kinds[j]);
        if(v[j].request!=request||v[j].count!=counts[j]||v[j].type!=bn_backend_quant_scalar_type(kinds[j])||
           !width||v[j].offset>SIZE_MAX-counts[j]||
           (j!=2&&j!=3&&j!=9&&v[j].offset%(16/width)))return -1;
        for(int k=0;k<j;++k)if(v[k].count&&v[j].slot==v[k].slot&&v[j].offset<v[k].offset+v[k].count&&v[k].offset<v[j].offset+v[j].count)return -1;
    }
    return 0;
}
int bn_transformer_bf16_mla_attention(void *request,const BnTransformerBF16MLAAttentionPlan *p) {
    if(bf16_mla_attention_validate(request,p))return -1;
    const BnBackendBF16MLAWeights *w=p->weights;const BnGPUMlaFP8AttentionResidentPlan *a=&p->attention;
    BnGPUBackend *g=w->gpu;
    BnTransformerBF16MLAOutputPlan output={.weights=w,.layer=p->layer,.n_tokens=a->n_tokens,
        .context=a->out,.values=p->values,.out=p->out};
    if(g->mla_fp8_attention_resident(g->ctx,request,a)||bn_transformer_bf16_mla_output(request,&output))return -1;
    return 0;
}

static int bf16_mla_same_view(BnGPUResidentView a,BnGPUResidentView b) {
    return a.request==b.request&&a.slot==b.slot&&a.offset==b.offset&&a.count==b.count&&a.type==b.type;
}
static void bf16_indexed_mla_views(const BnTransformerBF16IndexedMLAPlan *p,BnGPUResidentView out[37]) {
    const BnTransformerBF16MLACachePlan *c=&p->prepare;
    const BnTransformerBF16MLAPreparePlan *q=&c->prepare;
    const BnTransformerBF16IndexerDecodePlan *d=&p->indexer;
    const BnTransformerBF16IndexerPlan *i=&d->indexer;
    const BnTransformerBF16MLAAttentionPlan *o=&p->output;
    const BnGPUMlaFP8AttentionResidentPlan *a=&o->attention;
    BnGPUResidentView v[]={q->x,q->q_raw,q->qr,q->kv_raw,q->kv,q->q,q->query,c->slots,c->cache,
        i->initial_tail,i->next_tail,i->q,i->key_raw,i->key,i->gate,i->cast,i->head_raw,
        i->q_fp8,i->q_scales,i->head_weights,i->compressed,i->compressed_scales,
        d->cache,d->block_table,d->slots,d->scores,d->pools,d->out,d->status,
        a->block_table,a->physical,a->out,a->lse,a->partial,a->partial_lse,o->values,o->out};
    memcpy(out,v,sizeof(v));
}
static int bf16_indexed_mla_validate(void *request,const BnTransformerBF16IndexedMLAPlan *p) {
    if(!p||bf16_mla_cache_validate(request,&p->prepare)||
       bf16_indexer_decode_validate(request,&p->indexer)||bf16_mla_attention_validate(request,&p->output))return -1;
    const BnTransformerBF16MLACachePlan *c=&p->prepare;
    const BnTransformerBF16MLAPreparePlan *q=&c->prepare;
    const BnTransformerBF16IndexerDecodePlan *d=&p->indexer;
    const BnTransformerBF16IndexerPlan *i=&d->indexer;
    const BnTransformerBF16MLAAttentionPlan *o=&p->output;
    const BnGPUMlaFP8AttentionResidentPlan *a=&o->attention;
    if(i->weights->gpu!=q->weights->gpu||o->weights!=q->weights||o->layer!=q->layer||i->layer!=q->layer||
       i->weights->dim!=q->weights->dim||i->weights->q_rank!=q->weights->q_rank||
       i->n_tokens!=q->n_tokens||a->n_tokens!=q->n_tokens||a->position!=i->position||
       a->num_cache_blocks!=c->num_cache_blocks||d->short_decode!=(i->position+i->n_tokens<=2048)||
       !bf16_mla_same_view(i->x,q->x)||!bf16_mla_same_view(i->qr,q->qr)||
       !bf16_mla_same_view(a->query,q->query)||!bf16_mla_same_view(a->cache,c->cache)||
       !bf16_mla_same_view(a->selection,d->out)||!bf16_mla_same_view(a->status,d->status))return -1;
    /* Only the six explicitly linked borrows above are shared. Every other
     * stage resource must be disjoint across the entire request operation. */
    BnGPUResidentView v[37];bf16_indexed_mla_views(p,v);
    for(size_t j=0;j<sizeof(v)/sizeof(v[0]);++j)if(v[j].count)
        for(size_t k=0;k<j;++k)if(v[k].count&&v[j].slot==v[k].slot&&
            v[j].offset<v[k].offset+v[k].count&&v[k].offset<v[j].offset+v[j].count)return -1;
    return 0;
}
int bn_transformer_bf16_indexed_mla(void *request,const BnTransformerBF16IndexedMLAPlan *p) {
    if(bf16_indexed_mla_validate(request,p))return -1;
    const BnTransformerBF16MLACachePlan *c=&p->prepare;
    const BnTransformerBF16IndexerDecodePlan *d=&p->indexer;
    const BnTransformerBF16MLAAttentionPlan *o=&p->output;
    if(bn_transformer_bf16_mla_cache(request,c)||bn_transformer_bf16_indexer_decode(request,d)||
       bn_transformer_bf16_mla_attention(request,o))return -1;
    return 0;
}

static int bf16_mhc_block_pre_validate(void *request,BnGPUBackend *g,const BnGPUMHCPreResidentPlan *m,
    int nt,int dim,int fused) {
    if(!m||m->n_tokens!=nt||m->dim!=dim||dim!=4096||m->streams!=4||m->n_splits<1||
       m->n_splits>65535||m->n_splits>dim*4/64||m->iterations<1||m->iterations>128||
       !m->fn||!m->scale||!m->base||!m->norm||!isfinite(m->rms_eps)||m->rms_eps<=0||
       !isfinite(m->pre_eps)||m->pre_eps<0||!isfinite(m->sinkhorn_eps)||m->sinkhorn_eps<=0||
       !isfinite(m->post_mult)||m->post_mult<0||!isfinite(m->norm_eps)||m->norm_eps<=0||
       (fused?(!g->mhc_fused_post_pre_bf16_resident||!g->mhc_fused_bf16_split_count):
               (!g->mhc_pre_bf16_resident||!g->mhc_pre_bf16_split_count)))return -1;
    int splits=fused?g->mhc_fused_bf16_split_count(g->ctx,nt,dim):g->mhc_pre_bf16_split_count(g->ctx,nt,dim);
    if(m->n_splits!=splits)return -1;
    size_t n=(size_t)nt,ns=n*m->n_splits;
    BnGPUResidentView v[]={m->residual,m->dot,m->sqrsum,m->pre,m->post,m->comb,m->collapsed,m->out};
    size_t counts[]={n*dim*4,ns*24,ns,n*4,n*4,n*16,n*dim,n*dim};
    for(int j=0;j<8;++j){int type=j>=1&&j<=5?bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32):BF16;
        if(v[j].request!=request||v[j].type!=type||v[j].count!=counts[j]||v[j].offset>SIZE_MAX-counts[j])return -1;}
    return 0;
}
static int bf16_mla_moe_block_validate(void *request,const BnTransformerBF16MLAMoEBlockPlan *p) {
    if(!p||(p->has_previous!=0&&p->has_previous!=1)||bf16_indexed_mla_validate(request,&p->attention)||
       bf16_moe_validate(request,&p->moe))return -1;
    const BnBackendBF16MLAWeights *w=p->attention.prepare.prepare.weights;BnGPUBackend *g=w->gpu;
    int nt=p->attention.prepare.prepare.n_tokens,dim=w->dim;
    if(p->moe.weights->gpu!=g||p->moe.layer!=w->layer||p->moe.n_tokens!=nt||p->moe.weights->dim!=dim||dim!=4096||
       nt>INT_MAX/(dim*4)||
       !bf16_mla_same_view(p->pre.out,p->attention.prepare.prepare.x)||
       !bf16_mla_same_view(p->ffn.out,p->moe.x)||
       bf16_mhc_block_pre_validate(request,g,&p->pre,nt,dim,p->has_previous)||
       bf16_mhc_block_pre_validate(request,g,&p->ffn,nt,dim,1))return -1;
    BnGPUResidentView v[74]={p->pre.residual,p->pre.dot,p->pre.sqrsum,p->pre.pre,p->pre.post,p->pre.comb,p->pre.collapsed,p->pre.out};
    size_t nv=8;BnGPUResidentView indexed[37];bf16_indexed_mla_views(&p->attention,indexed);
    for(int j=1;j<37;++j)v[nv++]=indexed[j];
    BnGPUResidentView f[]={p->ffn.residual,p->ffn.dot,p->ffn.sqrsum,p->ffn.pre,p->ffn.post,p->ffn.comb,p->ffn.collapsed,p->ffn.out};
    for(int j=0;j<8;++j)v[nv++]=f[j];
    const BnTransformerBF16MoEPlan *m=&p->moe;
    BnGPUResidentView moe[]={m->cast,m->logits,m->route_weights,m->indices,m->metadata,m->cursors,m->slots,m->inverse,m->gathered,m->gate,m->up,m->down,m->weighted,m->routed,m->shared_gate_up,m->shared_activated,m->shared_out,m->out};
    for(int j=0;j<18;++j)v[nv++]=moe[j];
    if(p->has_previous){
        BnGPUResidentView prior[]={p->previous_residual,p->previous_branch,p->previous_post,p->previous_comb};
        size_t counts[]={(size_t)nt*dim*4,(size_t)nt*dim,(size_t)nt*4,(size_t)nt*16};
        for(int j=0;j<4;++j){int type=j<2?BF16:bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32);
            if(prior[j].request!=request||prior[j].type!=type||prior[j].count!=counts[j]||prior[j].offset>SIZE_MAX-counts[j])return -1;
            v[nv++]=prior[j];}
    }
    for(size_t j=0;j<nv;++j)if(v[j].count)for(size_t k=0;k<j;++k)if(v[k].count&&
        v[j].slot==v[k].slot&&v[j].offset<v[k].offset+v[k].count&&v[k].offset<v[j].offset+v[j].count)return -1;
    return 0;
}
int bn_transformer_bf16_mla_moe_begin(void *request,const BnTransformerBF16MLAMoEBlockPlan *p,
    BnTransformerBF16MLAMoEBlockWork *s) {
    if(!s||s->pending||s->failed||bf16_mla_moe_block_validate(request,p))return -1;
    BnGPUBackend *g=p->attention.prepare.prepare.weights->gpu;
    *s=(BnTransformerBF16MLAMoEBlockWork){.request=request,.pending=1};
    if(p->has_previous){BnGPUMHCFusedResidentPlan pre={.next=p->pre,.previous_residual=p->previous_residual,
        .previous_branch=p->previous_branch,.previous_post=p->previous_post,.previous_comb=p->previous_comb};
        if(g->mhc_fused_post_pre_bf16_resident(g->ctx,request,&pre))goto fail;
    }else if(g->mhc_pre_bf16_resident(g->ctx,request,&p->pre))goto fail;
    if(bn_transformer_bf16_indexed_mla(request,&p->attention))goto fail;
    BnGPUMHCFusedResidentPlan ffn={.next=p->ffn,.previous_residual=p->pre.residual,
        .previous_branch=p->attention.output.out,.previous_post=p->pre.post,.previous_comb=p->pre.comb};
    if(g->mhc_fused_post_pre_bf16_resident(g->ctx,request,&ffn))goto fail;
    int32_t status=0;
    if(g->resident_read(request,p->attention.output.attention.status,&status))goto fail;
    s->status_read_bytes=sizeof(status);
    if(status||bn_transformer_bf16_moe_begin(request,&p->moe,&s->moe))goto fail;
    return 0;
fail:
    s->failed=1;return -1;
}
int bn_transformer_bf16_mla_moe_submit(void *request,BnTransformerBF16MLAMoEBlockWork *s) {
    if(!s||s->request!=request||!s->pending||s->failed)return -1;
    int status=bn_transformer_bf16_moe_submit(request,&s->moe);
    if(status==BN_GPU_MOE_BANK_BUSY)return status;
    if(status){s->failed=1;return -1;}
    s->pending=0;return 0;
}

static int bf16_kda_moe_block_validate(void *request,const BnTransformerBF16KDAMoEBlockPlan *p) {
    if(!p||p->attention.deferred!=1||bf16_mhc_kda_validate(request,&p->attention)||
       bf16_moe_validate(request,&p->moe))return -1;
    const BnTransformerBF16MHCKDAPlan *a=&p->attention;
    const BnBackendBF16KDAWeights *w=a->weights;BnGPUBackend *g=w->gpu;
    const BnGPUMHCPreResidentPlan *f=&p->ffn;
    const BnTransformerBF16MoEPlan *m=&p->moe;
    int nt=a->mhc.n_tokens,dim=w->dim;
    if(dim!=4096||w->layer!=m->layer||m->weights->gpu!=g||m->n_tokens!=nt||m->weights->dim!=dim||
       !bf16_mla_same_view(f->out,m->x)||
       bf16_mhc_block_pre_validate(request,g,&a->mhc,nt,dim,a->has_previous)||
       bf16_mhc_block_pre_validate(request,g,f,nt,dim,1))return -1;
    const BnTransformerBF16KDAWorkspace *k=&a->kda;
    BnGPUResidentView v[55]={a->mhc.residual,a->mhc.dot,a->mhc.sqrsum,a->mhc.pre,
        a->mhc.post,a->mhc.comb,a->mhc.collapsed,a->mhc.out,
        k->projected,k->qkv,k->tail,k->fa_ga,k->beta,k->fa,k->ga,k->g1,k->g2,k->conv,k->core,k->gated,
        a->initial_history,a->next_history,a->initial_state,a->next_state,a->branch_out,
        f->residual,f->dot,f->sqrsum,f->pre,f->post,f->comb,f->collapsed,f->out,
        m->cast,m->logits,m->route_weights,m->indices,m->metadata,m->cursors,m->slots,m->inverse,
        m->gathered,m->gate,m->up,m->down,m->weighted,m->routed,m->shared_gate_up,m->shared_activated,m->shared_out,m->out};
    size_t nv=51;
    if(a->has_previous){v[nv++]=a->previous_residual;v[nv++]=a->previous_branch;
        v[nv++]=a->previous_post;v[nv++]=a->previous_comb;}
    for(size_t i=0;i<nv;++i)for(size_t j=0;j<i;++j)
        if(v[i].slot==v[j].slot&&v[i].offset<v[j].offset+v[j].count&&
           v[j].offset<v[i].offset+v[i].count)return -1;
    return 0;
}
int bn_transformer_bf16_kda_moe_begin(void *request,const BnTransformerBF16KDAMoEBlockPlan *p,
    BnTransformerBF16KDAMoEBlockWork *s) {
    if(!s||s->pending||s->failed||bf16_kda_moe_block_validate(request,p))return -1;
    *s=(BnTransformerBF16KDAMoEBlockWork){.request=request,.pending=1};
    const BnTransformerBF16MHCKDAPlan *a=&p->attention;BnGPUBackend *g=a->weights->gpu;
    BnGPUMHCFusedResidentPlan next={.next=p->ffn,.previous_residual=a->mhc.residual,
        .previous_branch=a->branch_out,.previous_post=a->mhc.post,.previous_comb=a->mhc.comb};
    if(bn_transformer_bf16_mhc_kda_attention(request,a)||
       g->mhc_fused_post_pre_bf16_resident(g->ctx,request,&next)||
       bn_transformer_bf16_moe_begin(request,&p->moe,&s->moe)) {
        s->failed=1;return -1;
    }
    return 0;
}
int bn_transformer_bf16_kda_moe_submit(void *request,BnTransformerBF16KDAMoEBlockWork *s) {
    if(!s||s->request!=request||!s->pending||s->failed)return -1;
    int status=bn_transformer_bf16_moe_submit(request,&s->moe);
    if(status==BN_GPU_MOE_BANK_BUSY)return status;
    if(status){s->failed=1;return -1;}
    s->pending=0;return 0;
}

int bn_transformer_bf16_glm_output(void *request,const BnTransformerBF16GLMOutputPlan *p) {
    if(!request||!p||!p->weights||p->n_tokens<1||(p->last_only!=0&&p->last_only!=1))return -1;
    const BnBackendBF16GLMModel *w=p->weights;BnGPUBackend *g=w->gpu;
    int nt=p->n_tokens,dim=w->dim,head=p->last_only?1:nt;
    if(!g||dim!=4096||nt>INT_MAX/(dim*4)||w->vocab_size<1||head>INT_MAX/w->vocab_size||
       !w->anatomy||w->anatomy->dim!=dim||w->anatomy->norm_eps!=1e-5f||
       !w->output_norm||!w->output.raw||w->output.decoded||
       w->output.type!=BN_BACKEND_QUANT_NATIVE_QUANT_TENSOR_TYPE||
       !g->mhc_post_bf16_resident||!g->stream_mean_bf16_resident||!g->rmsnorm_bf16_resident||!g->linear_a8_bf16_resident||
       !(g->resident_plain_rmsnorm_reductions&(1u<<BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096))||
       p->collapsed.offset%8||p->normalized.offset%8||p->logits.offset%8)return -1;
    BnGPUResidentView v[]={p->residual,p->branch,p->post,p->comb,p->completed,p->collapsed,p->normalized,p->logits};
    size_t n=(size_t)nt,counts[]={n*dim*4,n*dim,n*4,n*16,n*dim*4,n*dim,n*dim,(size_t)head*w->vocab_size};
    for(int i=0;i<8;++i){int type=i==2||i==3?bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32):BF16;
        if(v[i].request!=request||v[i].type!=type||v[i].count!=counts[i]||v[i].offset>SIZE_MAX-counts[i])return -1;
        for(int j=0;j<i;++j)if(v[i].slot==v[j].slot&&(v[i].type!=v[j].type||
            (v[i].offset<v[j].offset+v[j].count&&v[j].offset<v[i].offset+v[i].count)))return -1;
    }
    BnGPUMHCPostResidentPlan post={.n_tokens=nt,.dim=dim,.streams=4,.residual=p->residual,
        .branch=p->branch,.post=p->post,.comb=p->comb,.out=p->completed};
    BnGPUStreamMeanResidentPlan mean={.n_tokens=nt,.dim=dim,.streams=4,.x=p->completed,.out=p->collapsed};
    BnGPURMSNormResidentPlan norm={.n_tokens=nt,.dim=dim,.eps=w->anatomy->norm_eps,
        .reduction=BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096,.weight=w->output_norm,.x=p->collapsed,.out=p->normalized};
    BnGPUBF16LinearResidentPlan output={.n_tokens=head,.rows=w->vocab_size,.cols=dim,.weight=w->output,
        .x=p->last_only?bn_gpu_resident_view_slice(p->normalized,(size_t)(nt-1)*dim,dim):p->normalized,.out=p->logits};
    return g->mhc_post_bf16_resident(g->ctx,request,&post)||g->stream_mean_bf16_resident(g->ctx,request,&mean)||
        g->rmsnorm_bf16_resident(g->ctx,request,&norm)||g->linear_a8_bf16_resident(g->ctx,request,&output)?-1:0;
}

/* Explicit GLM request. The resident container owns every device allocation;
 * this object owns host metadata only and is released after device teardown. */
typedef struct {
    BnGPUResidentView history[2],state[2],latent,index,tail[2];
} BF16GLMLayerState;
typedef struct {
    BnGPUResidentView mhc[2][8],kda[12],mla[37],moe[18];
    BnGPUResidentView branch,gate_up,activated,dense_out;
} BF16GLMWorkspace;
typedef struct {
    const BnBackendModel *model;
    const BnModel *context_model;
    BnSession *context_session;
    int32_t *context_tokens;
    uint16_t *host_logits;
    int want_logits;
    const BnBackendBF16GLMModel *weights;
    const BnMoEIO *source;
    BnGPUBackend *gpu;
    void *request;
    int max_tokens,capacity,position,parity,active,failed,nt,prefill,layer,started,head;
    int latent_blocks,index_blocks;
    size_t device_bytes,host_bytes;
    BnMoEState host;
    BF16GLMLayerState states[45];
    BF16GLMWorkspace banks[2];
    BnGPUResidentView ids,embedding,twice,token_slots,pool_slots,latent_table,index_table,status;
    BnGPUResidentView completed,collapsed,normalized,logits;
    int32_t *slots;
    BnGPUResidentView previous[4];
    BnTransformerBF16KDAMoEBlockWork kda_work;
    BnTransformerBF16MLAMoEBlockWork mla_work;
} BF16GLMRequest;
static const unsigned char bf16_glm_request_owner;
static BF16GLMRequest *bf16_glm_request(const BnBackendSession *s) {
    return bn_backend_session_execution_resources(s,&bf16_glm_request_owner);
}
static void bf16_glm_free(void *ptr) {
    BF16GLMRequest *r=ptr;
    if(!r)return;
    free(r->host.buf);free(r->host.buf2);free(r->host.buf5);free(r->slots);free(r->context_tokens);free(r->host_logits);free(r);
}
static int bf16_glm_current(const BF16GLMRequest *r) {
    return r&&!r->failed&&r->weights==bn_backend_layout_bf16_glm_model(r->model);
}
static int bf16_glm_add(size_t *total,size_t count,size_t width) {
    if(count>SIZE_MAX/width||*total>SIZE_MAX-count*width)return -1;
    *total+=count*width;return 0;
}
static int bf16_glm_alloc(BF16GLMRequest *r,BnGPUResidentView *v,int type,size_t count) {
    size_t width=type==BF16?2:type==bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_BYTES)?1:4;
    if(!count)count=1;
    if(bf16_glm_add(&r->device_bytes,count,width))return -1;
    if(!r->request){*v=(BnGPUResidentView){.type=type,.count=count};return 0;}
    return r->gpu->resident_alloc(r->request,type,count,v);
}
static BnGPUResidentView bf16_glm_take(BnGPUResidentView v,size_t n) {
    return bn_gpu_resident_view_slice(v,0,n);
}
/* Same allocation walk is used for admission and actual preparation. No GPU
 * storage is touched by the first pass; integer limits precede this walk. */
static int bf16_glm_storage(BF16GLMRequest *r) {
    size_t n=(size_t)r->max_tokens,mb=(size_t)r->latent_blocks,ib=(size_t)r->index_blocks;
    int f32=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32),i32=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32),bytes=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_BYTES);
#define GLM_ALLOC(v,t,c) do {if(bf16_glm_alloc(r,&(v),(t),(c)))return -1;} while(0)
    GLM_ALLOC(r->ids,i32,n);GLM_ALLOC(r->embedding,BF16,n*4096);GLM_ALLOC(r->twice,BF16,n*8192);
    GLM_ALLOC(r->token_slots,i32,n);GLM_ALLOC(r->pool_slots,i32,(n+3)/4+1);
    GLM_ALLOC(r->latent_table,i32,mb);GLM_ALLOC(r->index_table,i32,ib);GLM_ALLOC(r->status,i32,1);
    GLM_ALLOC(r->completed,BF16,n*16384);GLM_ALLOC(r->collapsed,BF16,n*4096);
    GLM_ALLOC(r->normalized,BF16,n*4096);GLM_ALLOC(r->logits,BF16,(size_t)r->weights->vocab_size);
    for(int l=0;l<45;++l){BF16GLMLayerState *s=&r->states[l];
        if(r->weights->layers[l].kda.gpu){for(int j=0;j<2;++j){GLM_ALLOC(s->history[j],BF16,73728);GLM_ALLOC(s->state[j],f32,1048576);}}
        else{GLM_ALLOC(s->latent,bytes,mb*64*656);GLM_ALLOC(s->index,bytes,ib*8448);for(int j=0;j<2;++j)GLM_ALLOC(s->tail[j],BF16,1024);}
    }
    size_t kc[]={24896,24576,320,256,64,128,128,8192,8192,24576,8192,8192};
    size_t mc[]={n*16384,n*64*24,n*64,n*4,n*4,n*16,n*4096,n*4096};
    size_t ec[]={n*4096,n*288,n*8,n*8,293,288,n*8,n*8,n*8*4096,n*8*2048,n*8*2048,n*8*4096,n*8*4096,n*4096,n*4096,n*2048,n*4096,n*4096};
    size_t splits=(n<64?n:64)*64*34,groups=(n+3)/4+1;
    size_t ac[]={0,n*1536,n*1536,n*512,n*512,n*64*256,n*64*512,0,0,0,0,n*4096,n*128,n*128,n*128,n*4096,n*32,n*4096,n*32,n*32,groups*128,groups,0,0,0,n*((size_t)r->capacity/4),n*512,n*2051,0,0,n*2176,n*64*512,n*64,splits*512,splits,n*64*256,n*4096};
    for(int b=0;b<2;++b){BF16GLMWorkspace *w=&r->banks[b];
        for(int k=0;k<2;++k)for(int j=0;j<8;++j)GLM_ALLOC(w->mhc[k][j],j>0&&j<6?f32:BF16,mc[j]);
        for(int j=0;j<12;++j)GLM_ALLOC(w->kda[j],BF16,n*kc[j]);
        for(int j=0;j<18;++j)GLM_ALLOC(w->moe[j],j<=2?f32:j<=7?i32:BF16,ec[j]);
        for(int j=1;j<37;++j)if(ac[j]||j==25){int type=j==15||j==16||j==18||j==19||j==21||j==25||j==32||j==34?f32:j==17||j==20?bytes:j==26||j==27||j==30?i32:BF16;GLM_ALLOC(w->mla[j],type,ac[j]);}
        GLM_ALLOC(w->branch,BF16,n*4096);GLM_ALLOC(w->gate_up,BF16,n*24576);
        GLM_ALLOC(w->activated,BF16,n*12288);GLM_ALLOC(w->dense_out,BF16,n*4096);
    }
#undef GLM_ALLOC
    return 0;
}
static int bf16_glm_clear(BF16GLMRequest *r) {
    BnGPUBackend *g=r->gpu;
    for(int l=0;l<45;++l){BF16GLMLayerState *s=&r->states[l];
        if(r->weights->layers[l].kda.gpu){for(int j=0;j<2;++j)if(g->resident_zero(r->request,s->history[j])||g->resident_zero(r->request,s->state[j]))return -1;}
        else if(g->resident_zero(r->request,s->latent)||g->resident_zero(r->request,s->index)||g->resident_zero(r->request,s->tail[0])||g->resident_zero(r->request,s->tail[1]))return -1;
    }
    return g->resident_zero(r->request,r->status)||g->resident_synchronize(r->request)?-1:0;
}
typedef struct {size_t gate,down,slot_count,bank_bytes,input_bytes,group_bytes,meta;} BF16GLMStoragePlan;
static int bf16_glm_admit(const BnBackendModel *model,const BnMoEIO *source,
    int max_tokens,int capacity,size_t device_budget,size_t host_budget,
    BF16GLMRequest *r,BF16GLMStoragePlan *out) {
    const BnBackendBF16GLMModel *w=bn_backend_layout_bf16_glm_model(model);
    if(!r||!out||!w||!source||w->n_layers!=45||w->dim!=4096||w->vocab_size!=154880||!w->layers||
       !w->anatomy||w->anatomy->norm_eps!=1e-5f||
       (!bn_moe_io_has_mmap(source)&&!bn_moe_io_has_pread(source))||
       max_tokens<1||capacity<max_tokens||capacity>INT_MAX/128*4||
       (size_t)((capacity+63)/64)>SIZE_MAX/(64u*656)||max_tokens>INT_MAX/(8*4096)||
       (size_t)max_tokens>SIZE_MAX/(size_t)capacity||(size_t)max_tokens>SIZE_MAX/(8u*4096*2))return -1;
    BnGPUBackend *g=w->gpu;
    if(!bn_gpu_backend_can_resident_request(g)||!g->resident_zero||!g->linear_bf16_reserve||!g->linear_bf16_workspace_size||
       !g->embedding_bf16_reserve||!g->embedding_bf16_resident||!g->join_columns_resident||
       !g->mhc_pre_bf16_split_count||!g->mhc_fused_bf16_split_count||!g->moe_bank_reserve||
       !g->moe_bank_input_reserve||!g->moe_bank_grouped_reserve)return -1;
    if(!g->linear_a8_bf16_resident
       ||!g->split_columns_resident
       ||!g->causal_conv_f32_weight_bf16_resident
       ||!g->kda_recurrent_bf16_resident
       ||!g->gated_rmsnorm_f32_weight_bf16_resident
       ||!g->clamped_swiglu_bf16_resident
       ||!g->mhc_pre_bf16_resident
       ||!g->mhc_fused_post_pre_bf16_resident
       ||!g->mhc_post_bf16_resident
       ||!g->stream_mean_bf16_resident
       ||!g->rmsnorm_bf16_resident
       ||!g->linear_bf16_f32_weight_resident
       ||!g->moe_biased_sigmoid_f32_resident
       ||!g->moe_group_bf16_resident
       ||!g->moe_bank_acquire
       ||!g->moe_bank_cancel
       ||!g->moe_bank_submit
       ||!g->moe_bank_prepare_input
       ||!g->moe_reduce_bf16_resident
       ||!g->add_bf16_resident
       ||!g->head_linear_bf16_resident
       ||!g->fwht128_fp8_resident
       ||!g->indexer_fp8_append_resident
       ||!g->indexer_fp8_paged_score_resident
       ||!g->indexer_select_resident
       ||!g->indexer_weight_scale_resident
       ||!g->kpool_fp8_resident
       ||!g->layernorm128_f32_bf16_resident
       ||!g->mla_fp8_attention_resident
       ||!g->mla_fp8_cache_append_resident
       ||!(g->resident_plain_rmsnorm_reductions&(1u<<BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096)))return -1;
    size_t gate=0,down=0;
    for(int l=0;l<45;++l){const BnBackendBF16GLMLayer *p=&w->layers[l];int mla=l%4==3;
        if(p->attention_mhc.gpu!=g||p->ffn_mhc.gpu!=g||(mla?(p->mla.gpu!=g||p->indexer.gpu!=g):p->kda.gpu!=g)||
           (l<3?p->dense.gpu!=g:p->moe.gpu!=g))return -1;
        if(p->attention_mhc.dim!=4096||p->ffn_mhc.dim!=4096||
           (mla?(p->mla.dim!=4096||p->mla.q_rank!=1536||p->mla.kv_rank!=512||p->mla.heads!=64||
                 p->mla.key_dim!=256||p->mla.value_dim!=256||p->indexer.dim!=4096||p->indexer.q_rank!=1536||
                 p->indexer.heads!=32||p->indexer.head_dim!=128||p->indexer.pool_size!=4):
                (p->kda.dim!=4096||p->kda.heads!=64||p->kda.head_dim!=128||p->kda.rank!=128||p->kda.kernel!=4))||
           (l<3?(p->dense.dim!=4096||p->dense.hidden_dim!=12288):
                (p->moe.dim!=4096||p->moe.hidden_dim!=2048||p->moe.n_experts!=288||p->moe.k!=8||
                 p->moe.shared.dim!=4096||p->moe.shared.hidden_dim!=2048)))return -1;
        if(p->moe.gate_bytes>gate)gate=p->moe.gate_bytes;
        if(p->moe.down_bytes>down)down=p->moe.down_bytes;
    }
    memset(r,0,sizeof(*r));
    r->model=model;r->weights=w;r->source=source;r->gpu=g;r->max_tokens=max_tokens;r->capacity=capacity;
    r->latent_blocks=(capacity+63)/64;r->index_blocks=((capacity+3)/4+63)/64;
    if(bf16_glm_storage(r))return -1;
    size_t n=(size_t)max_tokens,meta=6*sizeof(BnGPUMoEBankItem),bank_bytes=122683392u;
    size_t input_bytes=n*(4096/32)*36*9,group_bytes=n*8*(2048/32)*36+meta;
    size_t scratch=g->linear_bf16_workspace_size(g->ctx);
    if(!scratch||bf16_glm_add(&r->device_bytes,n*16384/32,36)||bf16_glm_add(&r->device_bytes,1,scratch)||
       bf16_glm_add(&r->device_bytes,1,bank_bytes+input_bytes+group_bytes+4))return -1;
    r->host_bytes=sizeof(*r);
    size_t slot_count=(size_t)(r->latent_blocks>max_tokens?r->latent_blocks:max_tokens);
    if(slot_count<(n+3)/4+1)slot_count=(n+3)/4+1;
    if(bf16_glm_add(&r->host_bytes,slot_count,4)||bf16_glm_add(&r->host_bytes,2,gate)||
       bf16_glm_add(&r->host_bytes,1,down)||bf16_glm_add(&r->host_bytes,1,bank_bytes+meta)||
       r->device_bytes>device_budget||r->host_bytes>host_budget)return -1;
    *out=(BF16GLMStoragePlan){gate,down,slot_count,bank_bytes,input_bytes,group_bytes,meta};return 0;
}
int bn_transformer_bf16_glm_memory_plan(const BnModel *model,int max_tokens,
    int capacity,BnTransformerBF16RequestMemory *out) {
    if(!model||!out||capacity>model->config.seq_len)return -1;
    BF16GLMRequest r;BF16GLMStoragePlan p;
    if(bf16_glm_admit(bn_model_backend(model),bn_model_moe_io_const(model),max_tokens,capacity,SIZE_MAX,SIZE_MAX,&r,&p))return -1;
    size_t host=r.host_bytes;
    if(bf16_glm_add(&host,(size_t)max_tokens,sizeof(int32_t))||
       bf16_glm_add(&host,(size_t)r.weights->vocab_size,sizeof(uint16_t)))return -1;
    *out=(BnTransformerBF16RequestMemory){r.device_bytes,host,host};return 0;
}
int bn_transformer_bf16_glm_prepare(BnBackendSession *session,const BnBackendModel *model,
    const BnMoEIO *source,int max_tokens,int capacity,size_t device_budget,size_t host_budget) {
    if(!session||bn_backend_session_has_gpu_resources(session)||bn_backend_session_has_execution_resources(session))return -1;
    BF16GLMRequest *r=calloc(1,sizeof(*r));if(!r)return -1;BF16GLMStoragePlan p;
    if(bf16_glm_admit(model,source,max_tokens,capacity,device_budget,host_budget,r,&p))goto fail_host;
    BnGPUBackend *g=r->gpu;size_t n=(size_t)max_tokens,planned=r->device_bytes;
    size_t gate=p.gate,down=p.down,slot_count=p.slot_count,bank_bytes=p.bank_bytes;
    size_t input_bytes=p.input_bytes,group_bytes=p.group_bytes,meta=p.meta;
    r->slots=malloc(slot_count*4);r->host.buf=malloc(gate);r->host.buf2=malloc(gate);r->host.buf5=malloc(down);
    if(!r->slots||!r->host.buf||!r->host.buf2||!r->host.buf5)goto fail_host;
    r->host.buf_size=gate;r->host.buf2_size=gate;r->host.buf5_size=down;
    r->request=bn_gpu_backend_resident_request(g,session);if(!r->request)goto fail_host;
    r->device_bytes=0;
    if(bf16_glm_storage(r))goto fail_device;
    r->device_bytes=planned;
    BnGPUMoEBankConfig bank={.slots=2,.experts_per_slot=3,.dim=4096,.hidden_dim=2048,
        .host_budget_bytes=bank_bytes,.device_budget_bytes=bank_bytes,.n_experts=288,.k=8};
    if(g->linear_bf16_reserve(g->ctx,r->request,n*16384)||g->embedding_bf16_reserve(g->ctx,r->request)||
       g->moe_bank_reserve(g->ctx,r->request,&bank)||g->moe_bank_input_reserve(g->ctx,r->request,max_tokens,input_bytes)||
       g->moe_bank_grouped_reserve(g->ctx,r->request,max_tokens,meta,group_bytes))goto fail_device;
    for(int j=0;j<r->latent_blocks;++j)r->slots[j]=j;
    if(g->resident_write(r->request,r->latent_table,r->slots))goto fail_device;
    if(g->resident_write(r->request,r->index_table,r->slots)||bf16_glm_clear(r))goto fail_device;
    if(bn_backend_session_attach_execution_resources(session,&bf16_glm_request_owner,r,bf16_glm_free))goto fail_device;
    return 0;
fail_device:
    bn_backend_session_release_gpu_resources(session);
fail_host:
    bf16_glm_free(r);return -1;
}
static BnGPUMHCPreResidentPlan bf16_glm_mhc(BF16GLMRequest *r,BF16GLMWorkspace *s,
    int which,const BnBackendBF16MHCWeights *w,int first) {
    size_t n=(size_t)r->nt;BnGPUBackend *g=r->gpu;
    int sp=first?g->mhc_pre_bf16_split_count(g->ctx,r->nt,4096):g->mhc_fused_bf16_split_count(g->ctx,r->nt,4096);
    BnGPUMHCPreResidentPlan p={.n_tokens=r->nt,.dim=4096,.streams=4,.n_splits=sp,.iterations=w->iterations,
        .rms_eps=w->rms_eps,.pre_eps=w->hc_eps,.sinkhorn_eps=w->hc_eps,.post_mult=2,.norm_eps=w->norm_eps,
        .fn=w->fn,.scale=w->scale,.base=w->base,.norm=w->norm};
    size_t sizes[]={n*16384,n*(sp>0?sp:0)*24,n*(sp>0?sp:0),n*4,n*4,n*16,n*4096,n*4096};
    BnGPUResidentView v[8];for(int j=0;j<8;++j)v[j]=bf16_glm_take(s->mhc[which][j],sizes[j]);
    p.residual=v[0];p.dot=v[1];p.sqrsum=v[2];p.pre=v[3];p.post=v[4];p.comb=v[5];p.collapsed=v[6];p.out=v[7];return p;
}
static BnTransformerBF16MHCKDAPlan bf16_glm_kda_plan(BF16GLMRequest *r,BnGPUMHCPreResidentPlan pre) {
    BF16GLMWorkspace *s=&r->banks[r->layer&1];BF16GLMLayerState *state=&r->states[r->layer];
    size_t n=(size_t)r->nt,sizes[]={24896,24576,320,256,64,128,128,8192,8192,24576,8192,8192};
    BnGPUResidentView v[12];for(int j=0;j<12;++j)v[j]=bf16_glm_take(s->kda[j],n*sizes[j]);
    BnTransformerBF16MHCKDAPlan p={.weights=&r->weights->layers[r->layer].kda,.mhc=pre,.deferred=1,.has_previous=r->layer>0,
        .initial_history=state->history[r->parity],.next_history=state->history[r->parity^1],
        .initial_state=state->state[r->parity],.next_state=state->state[r->parity^1],.branch_out=bf16_glm_take(s->branch,n*4096),
        .previous_residual=r->previous[0],.previous_branch=r->previous[1],.previous_post=r->previous[2],.previous_comb=r->previous[3]};
    p.kda=(BnTransformerBF16KDAWorkspace){v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11]};return p;
}
static BnTransformerBF16MoEPlan bf16_glm_moe_plan(BF16GLMRequest *r,BnGPUResidentView input) {
    size_t n=(size_t)r->nt,sizes[]={n*4096,n*288,n*8,n*8,293,288,n*8,n*8,n*8*4096,n*8*2048,n*8*2048,n*8*4096,n*8*4096,n*4096,n*4096,n*2048,n*4096,n*4096};
    BnGPUResidentView v[18];for(int j=0;j<18;++j)v[j]=bf16_glm_take(r->banks[r->layer&1].moe[j],sizes[j]);
    return (BnTransformerBF16MoEPlan){.weights=&r->weights->layers[r->layer].moe,.source=r->source,.host_state=&r->host,
        .layer=r->layer,.n_tokens=r->nt,.prefill=r->prefill,.bank_capacity=3,.x=input,.cast=v[0],.logits=v[1],.route_weights=v[2],
        .indices=v[3],.metadata=v[4],.cursors=v[5],.slots=v[6],.inverse=v[7],.gathered=v[8],.gate=v[9],.up=v[10],.down=v[11],
        .weighted=v[12],.routed=v[13],.shared_gate_up=v[14],.shared_activated=v[15],.shared_out=v[16],.out=v[17]};
}
static BnTransformerBF16IndexedMLAPlan bf16_glm_mla_plan(BF16GLMRequest *r,BnGPUResidentView input) {
    int l=r->layer,nt=r->nt,pos=r->position,mb=r->latent_blocks,ib=r->index_blocks,short_mode=pos+nt<=2048;
    size_t n=(size_t)nt,end=(size_t)(pos+nt)/4,groups=end-(size_t)pos/4,split=nt<=64?n*64*34:0;
    BF16GLMLayerState *s=&r->states[l];BF16GLMWorkspace *b=&r->banks[l&1];
    size_t sizes[]={0,n*1536,n*1536,n*512,n*512,n*64*256,n*64*512,0,0,0,0,n*4096,n*128,n*128,n*128,n*4096,n*32,n*4096,n*32,n*32,groups*128,groups,0,0,0,short_mode?0:n*end,n*512,n*2051,0,0,n*2176,n*64*512,n*64,split*512,split,n*64*256,n*4096};
    BnGPUResidentView v[37]={0};for(int j=1;j<37;++j)if(b->mla[j].count)v[j]=bf16_glm_take(b->mla[j],sizes[j]);
    v[0]=input;v[7]=bf16_glm_take(r->token_slots,n);v[8]=s->latent;v[9]=s->tail[r->parity];v[10]=s->tail[r->parity^1];
    v[22]=s->index;v[23]=r->index_table;v[24]=bf16_glm_take(r->pool_slots,groups);v[28]=r->status;v[29]=r->latent_table;
    const BnBackendBF16GLMLayer *w=&r->weights->layers[l];
    return (BnTransformerBF16IndexedMLAPlan){.prepare={.prepare={.weights=&w->mla,.layer=l,.n_tokens=nt,.x=v[0],.q_raw=v[1],.qr=v[2],.kv_raw=v[3],.kv=v[4],.q=v[5],.query=v[6]},.num_cache_blocks=mb,.slots=v[7],.cache=v[8]},
        .indexer={.indexer={.weights=&w->indexer,.layer=l,.n_tokens=nt,.position=pos,.x=v[0],.qr=v[2],.initial_tail=v[9],.next_tail=v[10],.q=v[11],.key_raw=v[12],.key=v[13],.gate=v[14],.cast=v[15],.head_raw=v[16],.q_fp8=v[17],.q_scales=v[18],.head_weights=v[19],.compressed=v[20],.compressed_scales=v[21]},.num_cache_blocks=ib,.logical_blocks=ib,.short_decode=short_mode,.cache=v[22],.block_table=v[23],.slots=v[24],.scores=v[25],.pools=v[26],.out=v[27],.status=v[28]},
        .output={.weights=&w->mla,.layer=l,.attention={.n_tokens=nt,.position=pos,.num_cache_blocks=mb,.logical_blocks=mb,.query=v[6],.cache=v[8],.selection=v[27],.status=v[28],.block_table=v[29],.physical=v[30],.out=v[31],.lse=v[32],.partial=v[33],.partial_lse=v[34]},.values=v[35],.out=v[36]}};
}
int bn_transformer_bf16_glm_begin(BnBackendSession *session,const int32_t *tokens,int nt,int position,int prefill) {
    BF16GLMRequest *r=bf16_glm_request(session);
    if(!bf16_glm_current(r)||r->active||!tokens||nt<1||nt>r->max_tokens||position!=r->position||
       nt>r->capacity-position||(prefill!=0&&prefill!=1)||(!prefill&&nt!=1))return -1;
    for(int j=0;j<nt;++j)if(tokens[j]<0||tokens[j]>=r->weights->vocab_size)return -1;
    int sp=r->gpu->mhc_pre_bf16_split_count(r->gpu->ctx,nt,4096),fsp=r->gpu->mhc_fused_bf16_split_count(r->gpu->ctx,nt,4096);
    if(sp<1||sp>64||fsp<1||fsp>64)return -1;
    r->nt=nt;r->prefill=prefill;r->layer=0;r->started=0;r->head=0;r->active=1;r->want_logits=1;
    r->kda_work=(BnTransformerBF16KDAMoEBlockWork){0};r->mla_work=(BnTransformerBF16MLAMoEBlockWork){0};
    BnGPUBackend *g=r->gpu;void *q=r->request;size_t n=(size_t)nt;
    if(g->resident_write(q,bf16_glm_take(r->ids,n),tokens))goto failed;
    for(int j=0;j<nt;++j)r->slots[j]=position+j;
    if(g->resident_write(q,bf16_glm_take(r->token_slots,n),r->slots))goto failed;
    size_t groups=(size_t)(position+nt)/4-(size_t)position/4;
    for(size_t j=0;j<groups;++j)r->slots[j]=position/4+(int)j;
    if((groups&&g->resident_write(q,bf16_glm_take(r->pool_slots,groups),r->slots))||g->resident_zero(q,r->status))goto failed;
    BnGPUResidentView e=bf16_glm_take(r->embedding,n*4096),two=bf16_glm_take(r->twice,n*8192),wide=bf16_glm_take(r->banks[0].mhc[0][0],n*16384);
    BnGPUEmbeddingResidentPlan ep={.n_tokens=nt,.dim=4096,.vocab_size=r->weights->vocab_size,.weight=r->weights->embedding.raw,.indices=bf16_glm_take(r->ids,n),.out=e};
    BnGPUJoinColumnsResidentPlan a={.rows=nt,.cols=8192,.left_cols=4096,.left=e,.right=e,.x=two},b={.rows=nt,.cols=16384,.left_cols=8192,.left=two,.right=two,.x=wide};
    if(g->embedding_bf16_resident(g->ctx,q,&ep)||g->join_columns_resident(g->ctx,q,&a)||g->join_columns_resident(g->ctx,q,&b))goto failed;
    return 0;
failed:r->failed=1;return -1;
}
static int bf16_glm_layer_submit(BF16GLMRequest *r,int prepare_only) {
    const BnBackendBF16GLMLayer *w=&r->weights->layers[r->layer];BF16GLMWorkspace *s=&r->banks[r->layer&1];
    BnGPUMHCPreResidentPlan pre=bf16_glm_mhc(r,s,0,&w->attention_mhc,r->layer==0),ffn=bf16_glm_mhc(r,s,1,&w->ffn_mhc,0);
    BnGPUResidentView out;
    if(r->layer<3){BnTransformerBF16DenseKDABlockPlan p={.attention=bf16_glm_kda_plan(r,pre),.ffn=ffn,.weights=&w->dense,
            .gate_up=bf16_glm_take(s->gate_up,(size_t)r->nt*24576),.activated=bf16_glm_take(s->activated,(size_t)r->nt*12288),.out=bf16_glm_take(s->dense_out,(size_t)r->nt*4096)};
        if(bn_transformer_bf16_dense_kda_block(r->request,&p))goto failed;
        out=p.out;
    }else{int status;
        if(w->kda.gpu){if(!r->started){BnTransformerBF16KDAMoEBlockPlan p={.attention=bf16_glm_kda_plan(r,pre),.ffn=ffn,.moe=bf16_glm_moe_plan(r,ffn.out)};
                r->kda_work=(BnTransformerBF16KDAMoEBlockWork){0};if(bn_transformer_bf16_kda_moe_begin(r->request,&p,&r->kda_work))goto failed;r->started=1;}
            if(prepare_only)return 1;
            status=bn_transformer_bf16_kda_moe_submit(r->request,&r->kda_work);out=r->kda_work.moe.active.out;
        }else{if(!r->started){BnTransformerBF16MLAMoEBlockPlan p={.pre=pre,.ffn=ffn,.has_previous=1,.previous_residual=r->previous[0],.previous_branch=r->previous[1],.previous_post=r->previous[2],.previous_comb=r->previous[3],.attention=bf16_glm_mla_plan(r,pre.out),.moe=bf16_glm_moe_plan(r,ffn.out)};
                r->mla_work=(BnTransformerBF16MLAMoEBlockWork){0};if(bn_transformer_bf16_mla_moe_begin(r->request,&p,&r->mla_work))goto failed;r->started=1;}
            if(prepare_only)return 1;
            status=bn_transformer_bf16_mla_moe_submit(r->request,&r->mla_work);out=r->mla_work.moe.active.out;
        }
        if(status==BN_GPU_MOE_BANK_BUSY)return status;
        if(status)goto failed;
    }
    r->previous[0]=ffn.residual;r->previous[1]=out;r->previous[2]=ffn.post;r->previous[3]=ffn.comb;r->started=0;r->layer++;
    return 0;
failed:r->failed=1;return -1;
}

static int bf16_glm_head_submit(BF16GLMRequest *r) {
    if (!r->want_logits) { r->head=1; return 0; }
    BnTransformerBF16GLMOutputPlan p={.weights=r->weights,.n_tokens=r->nt,.last_only=1,
        .residual=r->previous[0],.branch=r->previous[1],.post=r->previous[2],.comb=r->previous[3],
        .completed=bf16_glm_take(r->completed,(size_t)r->nt*16384),.collapsed=bf16_glm_take(r->collapsed,(size_t)r->nt*4096),
        .normalized=bf16_glm_take(r->normalized,(size_t)r->nt*4096),.logits=r->logits};
    if(bn_transformer_bf16_glm_output(r->request,&p))goto failed;
    r->head=1;return 0;
failed:r->failed=1;return -1;
}

int bn_transformer_bf16_glm_submit(BnBackendSession *session) {
    BF16GLMRequest *r=bf16_glm_request(session);
    if(!bf16_glm_current(r)||!r->active||r->head)return -1;
    while(r->layer<45){int status=bf16_glm_layer_submit(r,0);if(status)return status;}
    return bf16_glm_head_submit(r);
}
int bn_transformer_bf16_glm_complete(BnBackendSession *session) {
    BF16GLMRequest *r=bf16_glm_request(session);
    if(!bf16_glm_current(r)||!r->active||!r->head||r->layer!=45)return -1;
    if(r->gpu->resident_synchronize(r->request)){r->failed=1;return -1;}
    r->position+=r->nt;r->parity^=1;r->active=0;return 0;
}
int bn_transformer_bf16_glm_restart(BnBackendSession *session) {
    BF16GLMRequest *r=bf16_glm_request(session);
    if(!bf16_glm_current(r)||r->active)return -1;
    if(bf16_glm_clear(r)){r->failed=1;return -1;}
    r->position=0;r->parity=0;r->head=0;r->layer=0;r->started=0;return 0;
}
int bn_transformer_bf16_glm_info(const BnBackendSession *session,BnTransformerBF16GLMRequestInfo *out) {
    BF16GLMRequest *r=bf16_glm_request(session);if(!r||!out)return -1;
    *out=(BnTransformerBF16GLMRequestInfo){.max_tokens=r->max_tokens,.capacity=r->capacity,.position=r->position,.parity=r->parity,
        .active=r->active,.failed=r->failed,.layer=r->layer,.head_queued=r->head&&r->want_logits,.device_bytes=r->device_bytes,.host_bytes=r->host_bytes,
        .request=r->request,.logits=r->logits,.source=r->host.stats};return 0;
}

static int bf16_glm_context_current(const BF16GLMRequest *r) {
    const BnModel *m = r ? r->context_model : NULL;
    BnSession *s = r ? r->context_session : NULL;
    return bf16_glm_current(r) && m && s && s->backend_only &&
        bn_session_matches_model(s, m) && s->pos == r->position &&
        bf16_glm_request(bn_session_backend(s)) == r &&
        r->model == bn_model_backend(m) && r->gpu == bn_model_gpu(m) &&
        r->source == bn_model_moe_io_const(m) &&
        m->config.dim == r->weights->dim &&
        m->config.n_layers == r->weights->n_layers;
}

int bn_transformer_bf16_glm_prepare_context(const BnModel *model,
    BnSession *session, int max_tokens, int capacity,
    size_t device_budget, size_t host_budget) {
    if (!model || !session || !session->backend_only || session->pos ||
        !bn_session_matches_model(session, model) || !bn_model_anatomy(model) ||
        !session->state.logits || !session->state.token_history ||
        max_tokens < 1 || capacity < max_tokens || capacity > model->config.seq_len)
        return -1;
    const BnBackendBF16GLMModel *w =
        bn_backend_layout_bf16_glm_model(bn_model_backend(model));
    if (!w || w->gpu != bn_model_gpu(model) || w->dim != model->config.dim ||
        w->n_layers != model->config.n_layers ||
        w->vocab_size != model->config.vocab_size) return -1;
    size_t extra = 0;
    if (bf16_glm_add(&extra, (size_t)max_tokens, sizeof(int32_t)) ||
        bf16_glm_add(&extra, (size_t)w->vocab_size, sizeof(uint16_t)) ||
        extra > host_budget) return -1;
    int32_t *ids = malloc((size_t)max_tokens * sizeof(*ids));
    uint16_t *logits = malloc((size_t)w->vocab_size * sizeof(*logits));
    if (!ids || !logits || bn_transformer_bf16_glm_prepare(
            bn_session_backend(session), bn_model_backend(model),
            bn_model_moe_io_const(model), max_tokens, capacity,
            device_budget, host_budget - extra)) {
        free(ids); free(logits);
        return -1;
    }
    BF16GLMRequest *r = bf16_glm_request(bn_session_backend(session));
    r->context_model = model;
    r->context_session = session;
    r->context_tokens = ids;
    r->host_logits = logits;
    r->host_bytes += extra;
    return 0;
}

static int bf16_glm_runtime_query(const BnBackendSession *session, int query) {
    const BF16GLMRequest *r = bf16_glm_request(session);
    if (!bf16_glm_context_current(r)) return query ? 0 : -1;
    return query == 0 ? r->position : (query == 1 ? r->max_tokens : r->capacity);
}

static int bf16_glm_runtime_restart(const BnModel *model, BnSession *session) {
    BnBackendSession *backend = bn_session_backend(session);
    BF16GLMRequest *r = bf16_glm_request(backend);
    if (!bf16_glm_context_current(r) || r->context_model != model ||
        r->context_session != session || bn_transformer_bf16_glm_restart(backend))
        return -1;
    memset(session->state.token_history, 0,
        (size_t)model->config.seq_len * sizeof(*session->state.token_history));
    session->pos = 0;
    session->gpu_kv_direct_valid = 0;
    session->moe_cache_prefill = 0;
    session->state.batched_prompt_contract = 0;
    bn_backend_session_set_gpu_decode_ssm_pending(backend, 0);
    return 0;
}

static int bf16_glm_runtime_run(BnModel *model, BnSession *session,
    const int *tokens, int nt, int position, int want_logits, int prefill) {
    BnBackendSession *backend = bn_session_backend(session);
    BF16GLMRequest *r = bf16_glm_request(backend);
    if (!tokens || !bf16_glm_context_current(r) || r->context_model != model ||
        r->context_session != session || r->active || nt < 1 ||
        nt > r->max_tokens || position != r->position || nt > r->capacity-position)
        return -1;
    for (int i = 0; i < nt; ++i)
        if (tokens[i] < 0 || tokens[i] >= r->weights->vocab_size) return -1;
    for (int i = 0; i < nt; ++i) r->context_tokens[i] = tokens[i];
    if (bn_transformer_bf16_glm_begin(backend, r->context_tokens, nt, position, prefill))
        return -1;
    r->want_logits = want_logits;
    int status;
    while ((status = bn_transformer_bf16_glm_submit(backend)) == BN_GPU_MOE_BANK_BUSY)
        sched_yield();
    if (status || (want_logits && r->gpu->resident_read(
            r->request, r->logits, r->host_logits)) ||
        bn_transformer_bf16_glm_complete(backend)) {
        r->failed = 1;
        return -1;
    }
    if (want_logits) for (int i = 0; i < r->weights->vocab_size; ++i) {
        uint32_t bits = (uint32_t)r->host_logits[i] << 16;
        memcpy(&session->state.logits[i], &bits, sizeof(bits));
    }
    for (int i = 0; i < nt; ++i)
        session->state.token_history[position+i] = r->context_tokens[i];
    session->pos = r->position;
    return 0;
}

int bn_transformer_bf16_glm_layer_views(const BnBackendSession *session,int layer,
    BnTransformerBF16GLMLayerStateViews *out) {
    BF16GLMRequest *r=bf16_glm_request(session);
    if(!bf16_glm_current(r)||r->active||!out||layer<0||layer>=45)return -1;
    const BF16GLMLayerState *s=&r->states[layer];
    *out=(BnTransformerBF16GLMLayerStateViews){.layer=layer,.position=r->position};
    if(r->weights->layers[layer].kda.gpu){out->history=s->history[r->parity];out->state=s->state[r->parity];}
    else{out->latent=s->latent;out->index=s->index;out->tail=s->tail[r->parity];}
    return 0;
}

#include "bf16_shared.h"
