#include "model.h"
#include "session_internal.h"
#include "backend_session.h"
#include "gpu_backend.h"
#include "model_internal.h"
#include "turboquant.h"
#include "sh_log.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int checked_mul3_size(size_t a, size_t b, size_t c, size_t *out) {
    size_t tmp;
    if (checked_mul_size(a, b, &tmp) != 0) return -1;
    return checked_mul_size(tmp, c, out);
}

static int checked_mul4_size(size_t a, size_t b, size_t c, size_t d, size_t *out) {
    size_t tmp;
    if (checked_mul3_size(a, b, c, &tmp) != 0) return -1;
    return checked_mul_size(tmp, d, out);
}

BnBackendSession *bn_session_backend(const BnSession *session) {
    return (session && session->backend_state)
        ? session->backend_state->backend
        : NULL;
}

static int session_host_extents(const BnModel *model,int *vocab,int *capacity,
                                const void **native_anatomy) {
    if(!model)return -1;
    *native_anatomy=NULL;
    if(model->runtime&&(model->runtime->trellis_experts||model->runtime->native_config.state)) {
        const BnModelNativeAnatomy *a=bn_model_native_anatomy(model);
        if(!a||a->vocab<=0||a->context<=0)return -1;
        *vocab=a->vocab;*capacity=a->context;*native_anatomy=a;
    } else {
        *vocab=model->config.vocab_size;*capacity=model->config.seq_len;
    }
    return 0;
}

int bn_session_matches_model(const BnSession *session, const BnModel *model) {
    const BnSessionBackendState *s = session ? session->backend_state : NULL;
    int vocab,capacity;const void *anatomy;
    return s && model && s->backend && s->model == model &&
        !session_host_extents(model,&vocab,&capacity,&anatomy)&&
        s->host_vocabulary==vocab&&s->host_capacity==capacity&&
        s->native_anatomy==anatomy;
}

/* Native Engram history is host bookkeeping. These transactions do not execute
 * or roll back GPU/KV/recurrent work; the executor owns that completion gate. */
_Static_assert(sizeof(int)==sizeof(uint32_t), "Native token IDs require 32-bit int");
struct BnSessionNativeHistory {
    const uint32_t *map;
    size_t host_bytes;
    int capacity, max_tokens, prefix, committed, staged;
    uint32_t pad;
    uint64_t epoch;
    uint32_t *committed_ids, *window;
    int *raw;
};

static BnSessionNativeHistory *session_history(const BnSession *s,const BnModel *m) {
    BnSessionNativeHistory *h=s&&s->backend_state?s->backend_state->native_history:NULL;
    return h&&s->backend_only&&bn_session_matches_model(s,m)&&
        h->map==bn_model_native_token_map(m,NULL)?h:NULL;
}

int bn_session_native_history_info(const BnSession *s,const BnModel *m,BnSessionNativeHistoryInfo *out) {
    BnSessionNativeHistory *h=session_history(s,m);
    if(!h||!out)return -1;
    *out=(BnSessionNativeHistoryInfo){h->max_tokens,h->capacity,h->committed,h->staged,h->host_bytes};return 0;
}

int bn_session_native_history_plan(const BnModel *m,int max_tokens,size_t *bytes) {
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(m);
    BnModelNativeTokenMapInfo info;
    if(!bytes||!a||bn_model_native_token_map_info(m,&info)||max_tokens<1||
       max_tokens>a->context||a->engram_ngram<2||
       a->engram_ngram>BN_MODEL_NATIVE_ENGRAM_MAX_NGRAM)return -1;
    size_t total=sizeof(BnSessionNativeHistory),part;
    size_t counts[4]={(size_t)a->context,(size_t)max_tokens,(size_t)max_tokens,(size_t)a->engram_ngram-1};
    for(int i=0;i<4;i++) {
        if(checked_mul_size(counts[i],sizeof(uint32_t),&part)||part>SIZE_MAX-total)return -1;
        total+=part;
    }
    *bytes=total;return 0;
}

int bn_session_native_history_prepare(BnSession *s,const BnModel *m,int max_tokens,size_t budget) {
    size_t bytes;BnModelNativeTokenMapInfo info;
    if(!s||!s->backend_only||!bn_session_matches_model(s,m)||s->pos||
       s->backend_state->native_history||!s->state.token_history||
       bn_session_native_history_plan(m,max_tokens,&bytes)||bytes>budget||
       bn_model_native_token_map_info(m,&info))return -1;
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(m);
    BnSessionNativeHistory *h=calloc(1,bytes);if(!h)return -1;
    h->map=bn_model_native_token_map(m,NULL);h->host_bytes=bytes;
    h->capacity=a->context;h->max_tokens=max_tokens;h->prefix=a->engram_ngram-1;h->pad=info.mapped_pad;
    h->committed_ids=(uint32_t*)(h+1);h->window=h->committed_ids+h->capacity;
    h->raw=(int*)(void*)(h->window+(size_t)h->prefix+(size_t)max_tokens);
    s->backend_state->native_history=h;return 0;
}

int bn_session_native_history_stage(BnSession *s,const BnModel *m,const int *tokens,int count) {
    BnSessionNativeHistory *h=session_history(s,m);
    if(!h||!tokens||h->staged||h->epoch==UINT64_MAX||count<1||
       count>h->max_tokens||s->pos!=h->committed||
       count>h->capacity-h->committed)return -1;
    /* Borrowed history views cannot be recycled as raw input. */
    uintptr_t input=(uintptr_t)tokens,base=(uintptr_t)h;
    size_t n=(size_t)count*sizeof(*tokens);
    if(input>UINTPTR_MAX-n||base>UINTPTR_MAX-h->host_bytes||
       (input<base+h->host_bytes&&base<input+n))return -1;
    if(bn_model_native_map_tokens(m,tokens,(size_t)count,h->window+h->prefix))return -1;
    memcpy(h->raw,tokens,n);
    for(int i=0;i<h->prefix;i++) {
        int p=h->committed-h->prefix+i;
        h->window[i]=p<0?h->pad:h->committed_ids[p];
    }
    h->staged=count;h->epoch++;return 0;
}

int bn_session_native_history_view(const BnSession *s,const BnModel *m,BnSessionNativeHistoryView *out) {
    BnSessionNativeHistory *h=session_history(s,m);
    if(!h||!out||!h->staged||s->pos!=h->committed)return -1;
    *out=(BnSessionNativeHistoryView){h->window,(size_t)h->prefix+(size_t)h->staged,
        h->prefix,h->staged,h->committed,h->epoch,h->host_bytes,s,h->raw};return 0;
}

static int session_history_ticket(const BnSession *s,const BnSessionNativeHistory *h,
    const BnSessionNativeHistoryView *v) {
    return h&&v&&h->staged&&v->owner==s&&v->mapped==h->window&&v->raw==h->raw&&
        v->epoch==h->epoch&&v->tokens==h->staged&&v->position==h->committed&&
        v->prefix==h->prefix&&v->length==(size_t)h->prefix+(size_t)h->staged&&
        s->pos==h->committed;
}

int bn_session_native_history_abort(BnSession *s,const BnModel *m,const BnSessionNativeHistoryView *view) {
    BnSessionNativeHistory *h=session_history(s,m);
    if(!session_history_ticket(s,h,view))return -1;
    h->staged=0;return 0;
}

int bn_session_native_history_commit(BnSession *s,const BnModel *m,const BnSessionNativeHistoryView *view) {
    BnSessionNativeHistory *h=session_history(s,m);
    if(!session_history_ticket(s,h,view))return -1;
    memcpy(h->committed_ids+h->committed,h->window+h->prefix,(size_t)h->staged*sizeof(uint32_t));
    for(int i=0;i<h->staged;i++)s->state.token_history[h->committed+i]=h->raw[i];
    h->committed+=h->staged;h->staged=0;s->pos=h->committed;return 0;
}

int bn_session_native_history_commit_group(const BnModel *m,BnSession *const *sessions,
    const BnSessionNativeHistoryView *views,int count) {
    if(!sessions||!views||count<1||count>16)return -1;
    BnSession *owners[16];BnSessionNativeHistory *histories[16];BnSessionNativeHistoryView tickets[16];
    for(int i=0;i<count;i++) {
        owners[i]=sessions[i];tickets[i]=views[i];histories[i]=session_history(owners[i],m);
        if(!session_history_ticket(owners[i],histories[i],tickets+i))return -1;
        for(int j=0;j<i;j++)if(owners[i]==owners[j]||histories[i]==histories[j])return -1;
    }
    /* Validate and snapshot every descriptor before any publication. The
     * remaining copies cannot fail or call external code. */
    for(int i=0;i<count;i++) {
        BnSession *s=owners[i];BnSessionNativeHistory *h=histories[i];
        memcpy(h->committed_ids+h->committed,h->window+h->prefix,(size_t)h->staged*sizeof(uint32_t));
        for(int k=0;k<h->staged;k++)s->state.token_history[h->committed+k]=h->raw[k];
        h->committed+=h->staged;h->staged=0;s->pos=h->committed;
    }
    return 0;
}

int bn_session_native_history_rewind(BnSession *s,const BnModel *m,int position) {
    BnSessionNativeHistory *h=session_history(s,m);
    if(!h||h->staged||position<0||position>h->committed||s->pos!=position)return -1;
    memset(h->committed_ids+position,0,(size_t)(h->committed-position)*sizeof(uint32_t));
    memset(s->state.token_history+position,0,(size_t)(h->committed-position)*sizeof(int32_t));
    h->committed=position;return 0;
}

static void session_history_reset(BnSession *s) {
    BnSessionNativeHistory *h=s->backend_state?s->backend_state->native_history:NULL;
    if(!h)return;
    memset(h+1,0,h->host_bytes-sizeof(*h));h->committed=0;h->staged=0;
    /* Never wrap: a saturated history remains unable to stage until free. */
    if(h->epoch!=UINT64_MAX)h->epoch++;
}


int bn_session_native_history_restart(BnSession *s,const BnModel *m) {
    BnSessionNativeHistory *h=session_history(s,m);
    if(!h||h->staged||h->epoch==UINT64_MAX||s->pos!=h->committed||!s->state.token_history)return -1;
    session_history_reset(s);
    memset(s->state.token_history,0,(size_t)h->capacity*sizeof(*s->state.token_history));
    s->pos=0;return 0;
}

static BnSession *session_create(const BnModel *model, BnAllocator *alloc,
                                 int backend_only) {
    if (!model || (!backend_only && bn_model_anatomy(model))) return NULL;
    int vocab,capacity;const void *anatomy;
    if(session_host_extents(model,&vocab,&capacity,&anatomy)||
       (!backend_only&&anatomy))return NULL;
    const BnConfig *c = &model->config;
    size_t logits_bytes = 0, history_bytes = 0;
    if (backend_only && (vocab <= 0 || capacity <= 0 ||
        checked_mul_size((size_t)vocab, sizeof(float), &logits_bytes) ||
        checked_mul_size((size_t)capacity, sizeof(int32_t), &history_bytes) ||
        history_bytes > SIZE_MAX - 2 * SH_ARENA_ALIGN ||
        logits_bytes > SIZE_MAX - history_bytes - 2 * SH_ARENA_ALIGN)) return NULL;

    // Allocate session struct
    BnSession *s;
    if (alloc) {
        s = (BnSession *)bn_malloc(alloc, sizeof(BnSession));
    } else {
        s = (BnSession *)malloc(sizeof(BnSession));
    }
    if (!s) return NULL;
    memset(s, 0, sizeof(BnSession));
    s->backend_state =
        (BnSessionBackendState *)calloc(1, sizeof(BnSessionBackendState));
    if (!s->backend_state) {
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }
    s->backend_state->model = model;
    s->backend_state->host_vocabulary = vocab;
    s->backend_state->host_capacity = capacity;
    s->backend_state->native_anatomy = anatomy;
    s->backend_state->backend = bn_backend_session_create();
    if (!s->backend_state->backend) {
        free(s->backend_state);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    // Create session arena
    size_t arena_size = backend_only
        ? logits_bytes + history_bytes + 2 * SH_ARENA_ALIGN
        : bn_model_session_arena_size(c, &model->weights);
    if (arena_size > SIZE_MAX / 2) {
        bn_backend_session_free(bn_session_backend(s));
        free(s->backend_state);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    s->arena = sh_arena_create(arena_size);
    if (!s->arena) {
        SH_LOG_ERROR("Failed to allocate session arena");
        bn_backend_session_free(bn_session_backend(s));
        free(s->backend_state);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    // Allocate session buffers
    int status = 0;
    if (backend_only) {
        s->state.logits = sh_arena_calloc(s->arena, (size_t)vocab, sizeof(float));
        s->state.token_history = sh_arena_calloc(s->arena, (size_t)capacity, sizeof(int32_t));
        status = !s->state.logits || !s->state.token_history;
    } else {
        status = bn_model_alloc_session_buffers(c, &model->weights, s->arena,
                                                &s->state, &s->moe_state);
    }
    if (status) {
        SH_LOG_ERROR("Failed to allocate session buffers");
        sh_arena_free(s->arena);
        bn_backend_session_free(bn_session_backend(s));
        free(s->backend_state);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    s->pos = 0;
    s->gpu_kv_direct_valid = 0;
    s->moe_cache_prefill = 0;
    s->state.batched_prompt_contract = 0;
    s->backend_only = backend_only;
    return s;
}

BnSession *bn_session_create(const BnModel *model, BnAllocator *alloc) {
    return session_create(model, alloc, 0);
}

BnSession *bn_session_create_backend(const BnModel *model, BnAllocator *alloc) {
    return session_create(model, alloc, 1);
}

void bn_session_free(BnSession *s, BnAllocator *alloc) {
    if (!s) return;
    bn_backend_session_free(bn_session_backend(s));
    if(s->backend_state)free(s->backend_state->native_history);
    free(s->backend_state);
    sh_arena_free(s->arena);
    if (alloc) {
        bn_free(alloc, s, sizeof(BnSession));
    } else {
        free(s);
    }
}

void bn_session_reset(BnSession *s, const BnModel *model) {
    if (!s || !model || (s->backend_state && s->backend_state->model &&
        !bn_session_matches_model(s, model))) return;
    /* Finish request-owned GPU work before resetting its host mirrors. */
    bn_backend_session_release_gpu_resources(bn_session_backend(s));
    session_history_reset(s);
    const BnConfig *c = &model->config;
    BnRunState *rs = &s->state;

    if (!s->backend_only) {
        // KV cache
        int n_attn = bn_model_session_policy_attention_layer_count(c);
        size_t kv_size = 0;
        if (n_attn < 0 ||
            checked_mul3_size((size_t)n_attn, (size_t)c->seq_len,
                              (size_t)c->kv_dim, &kv_size) != 0)
            return;
        size_t kv_elem = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
        size_t kv_bytes = 0;
        if (checked_mul_size(kv_size, kv_elem, &kv_bytes) != 0) return;
        if (rs->key_cache) memset(rs->key_cache, 0, kv_bytes);
        if (rs->value_cache) memset(rs->value_cache, 0, kv_bytes);

        // TQ compressed KV cache
        if (rs->key_cache_tq && rs->value_cache_tq && c->kv_tq_bits > 0 && bn_model_tq_state(model)) {
            int kb = bn_tq_key_bytes(bn_model_tq_state(model));
            int vb = bn_tq_value_bytes(bn_model_tq_state(model));
            size_t tq_key_total = 0, tq_val_total = 0;
            if (checked_mul4_size((size_t)n_attn, (size_t)c->seq_len,
                                  (size_t)c->n_kv_heads, (size_t)kb, &tq_key_total) != 0 ||
                checked_mul4_size((size_t)n_attn, (size_t)c->seq_len,
                                  (size_t)c->n_kv_heads, (size_t)vb, &tq_val_total) != 0)
                return;
            memset(rs->key_cache_tq, 0, tq_key_total);
            memset(rs->value_cache_tq, 0, tq_val_total);
        }

        // SSM state
        if (rs->ssm_state && c->ssm_time_step_rank > 0) {
            int n_ssm = bn_model_session_policy_ssm_layer_count(c);
            int head_v_dim = c->ssm_inner_size / c->ssm_time_step_rank;
            size_t state_total = 0;
            if (n_ssm < 0 ||
                checked_mul4_size((size_t)n_ssm, (size_t)c->ssm_time_step_rank,
                                  (size_t)c->ssm_state_size, (size_t)head_v_dim,
                                  &state_total) != 0 ||
                checked_mul_size(state_total, sizeof(float), &state_total) != 0)
                return;
            memset(rs->ssm_state, 0, state_total);
        }
        if (rs->ssm_conv_state) {
            int n_ssm = bn_model_session_policy_ssm_layer_count(c);
            int conv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
            size_t conv_total = 0;
            if (n_ssm < 0 ||
                checked_mul4_size((size_t)n_ssm, (size_t)(c->ssm_conv_kernel - 1),
                                  (size_t)conv_dim, sizeof(float), &conv_total) != 0)
                return;
            memset(rs->ssm_conv_state, 0, conv_total);
        }

    }

    if (rs->token_history)
        memset(rs->token_history, 0,
               (size_t)(s->backend_only&&s->backend_state?
                   s->backend_state->host_capacity:c->seq_len) * sizeof(*rs->token_history));
    if (rs->ple_conv_state && c->ple_head_count > 0) {
        size_t hc_dim = (size_t)c->hyper_connection_count * c->dim;
        size_t ple_count = (size_t)(c->ple_conv_kernel - 1) *
                           c->ple_ngram_size * hc_dim;
        memset(rs->ple_conv_state, 0, ple_count * sizeof(float));
    }

    s->pos = 0;
    s->gpu_kv_direct_valid = 0;
    s->moe_cache_prefill = 0;
    s->state.batched_prompt_contract = 0;
    (void)bn_gpu_backend_reset_activations(bn_model_gpu(model));
    bn_backend_session_set_gpu_decode_ssm_pending(bn_session_backend(s), 0);
}
