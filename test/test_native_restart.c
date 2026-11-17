/* Include the transformer module to inspect its private storage with synthetic
 * views. Model identity/history adapters isolate restart from checkpoint IO;
 * the real history helper is tested in native_session_checks.h. */
#include "transformer_internal.h"
#include "model_internal.h"
#include "gpu_moe_bridge.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "backend_quant.h"
#include "backend_session.h"
#include "session_internal.h"
#include "moe.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {unsigned char data[256][16];int mutable[256],slots,zero,sync,destroyed,fail_zero,fail_sync;} ReuseGPU;
static BnModel reuse_model;
static BnModelNativeAnatomy reuse_anatomy;
static BnMoEExpertReader reuse_reader;
static BnGPUMoETrellisStore reuse_store;
static BnGPUBackend reuse_gpu;
static int reuse_history_failure,reuse_history_committed=3,reuse_history_staged;
static const BnModelNativeAnatomy *reuse_get_anatomy(const BnModel *m){return m==&reuse_model?&reuse_anatomy:NULL;}
static const BnMoEExpertReader *reuse_get_reader(const BnModel *m){return m==&reuse_model?&reuse_reader:NULL;}
static BnGPUBackend *reuse_get_gpu(const BnModel *m){return m==&reuse_model?&reuse_gpu:NULL;}
static BnGPUMoETrellisStore *reuse_get_store(const BnModel *m){return m==&reuse_model?&reuse_store:NULL;}
static int reuse_matches(const BnSession *s,const BnModel *m){return s&&s->backend_state&&s->backend_state->model==m&&m==&reuse_model;}
static int reuse_history_info(const BnSession *s,const BnModel *m,BnSessionNativeHistoryInfo *out) {
    if(!reuse_matches(s,m))return -1;
    *out=(BnSessionNativeHistoryInfo){2,8,reuse_history_committed,reuse_history_staged,128};return 0;
}
static int reuse_history_restart(BnSession *s,const BnModel *m) {
    assert(reuse_matches(s,m));if(reuse_history_failure)return -1;
    memset(s->state.token_history,0,8*sizeof(int32_t));reuse_history_committed=0;s->pos=0;return 0;
}
static int reuse_rows_info(const BnBackendRowStage *s,BnBackendRowStageInfo *out){(void)s;memset(out,0,sizeof(*out));return 0;}
#define bn_model_native_anatomy reuse_get_anatomy
#define bn_model_trellis_expert_reader reuse_get_reader
#define bn_model_gpu reuse_get_gpu
#define bn_gpu_moe_trellis_model_store reuse_get_store
#define bn_session_matches_model reuse_matches
#define bn_session_native_history_info reuse_history_info
#define bn_session_native_history_restart reuse_history_restart
#define bn_backend_session_row_stage_info reuse_rows_info
#include "../src/transformer/trellis.c"
#undef bn_model_native_anatomy
#undef bn_model_trellis_expert_reader
#undef bn_model_gpu
#undef bn_gpu_moe_trellis_model_store
#undef bn_session_matches_model
#undef bn_session_native_history_info
#undef bn_session_native_history_restart
#undef bn_backend_session_row_stage_info
static int reuse_zero(void *ptr,BnGPUResidentView view) {
    ReuseGPU *r=ptr;assert(view.request==r&&view.slot<(unsigned)r->slots&&r->mutable[view.slot]);
    if(++r->zero==r->fail_zero)return -1;
    memset(r->data[view.slot],0,view.count*(view.type==BN_GGUF_TENSOR_F16?2:4));return 0;
}
static int reuse_sync(void *ptr){ReuseGPU *r=ptr;return ++r->sync==r->fail_sync?-1:0;}
static void reuse_destroy(void *ptr){((ReuseGPU*)ptr)->destroyed++;}
static BnGPUResidentView reuse_view(ReuseGPU *g,int mutable) {
    assert(g->slots<256);int slot=g->slots++;g->mutable[slot]=mutable;memset(g->data[slot],0xa5,16);
    return (BnGPUResidentView){.request=g,.offset=0,.count=4,.slot=(unsigned)slot,.type=slot%2?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_F16};
}
static BnSession *reuse_fixture(ReuseGPU *g,TrellisRequest **request) {
    memset(g,0,sizeof(*g));reuse_history_failure=0;reuse_history_committed=3;reuse_history_staged=0;
    reuse_anatomy.experts.layers=2;reuse_anatomy.experts.experts=2;reuse_anatomy.vocab=4;reuse_anatomy.context=8;
    reuse_gpu=(BnGPUBackend){.resident_zero=reuse_zero,.resident_synchronize=reuse_sync};
    BnSession *s=calloc(1,sizeof(*s));assert(s);s->backend_only=1;s->pos=3;
    s->backend_state=calloc(1,sizeof(*s->backend_state));assert(s->backend_state);
    s->backend_state->model=&reuse_model;s->backend_state->backend=bn_backend_session_create();assert(s->backend_state->backend);
    s->backend_state->native_request_generation=7;
    s->state.logits=malloc(4*sizeof(float));s->state.token_history=malloc(8*sizeof(int32_t));assert(s->state.logits&&s->state.token_history);
    for(int j=0;j<4;j++)s->state.logits[j]=17;
    for(int j=0;j<8;j++)s->state.token_history[j]=11;
    TrellisRequest *r=calloc(1,sizeof(*r));assert(r);*request=r;
    r->model=&reuse_model;r->anatomy=&reuse_anatomy;r->reader=&reuse_reader;r->store=&reuse_store;r->gpu=&reuse_gpu;r->request=g;r->destroy=reuse_destroy;r->generation=7;
    r->options.max_tokens=2;r->metadata=calloc(14,sizeof(int));r->plan.groups=calloc(2,sizeof(*r->plan.groups));r->plan.group_capacity=2;
    r->kv=calloc(1,sizeof(*r->kv));r->swa=calloc(1,sizeof(*r->swa));r->index=calloc(1,sizeof(*r->index));
    r->attention=calloc(1,sizeof(*r->attention));r->producer=calloc(1,sizeof(*r->producer));r->engram=calloc(1,sizeof(*r->engram));r->hc=calloc(1,sizeof(*r->hc));
    r->forward=calloc(1,sizeof(*r->forward)+8);assert(r->kv&&r->swa&&r->index&&r->attention&&r->producer&&r->engram&&r->hc&&r->forward);
    r->forward->last=(uint16_t*)(r->forward+1);r->forward->workspace.vocab=4;r->forward->workspace.max_tokens=2;
    r->forward->epoch=r->forward->completed=3;r->forward->history=(BnSessionNativeHistoryView){.epoch=9,.position=2,.tokens=1};
    r->forward->logits=reuse_view(g,1);r->hc->phase=3;r->hc->site=4;r->hc->position=2;r->hc->tokens=1;r->hc->history_epoch=9;r->hc->epoch=5;
    r->index->generation=11;r->index->active=1;r->index->pos=2;r->index->tokens=1;r->index->workspace.max_tokens=2;r->block_index_generation=11;
    r->kv->maxseq=512;r->kv->block_table=reuse_view(g,0);
    for(int j=0;j<TR_COUNT;j++)r->buffers[j]=reuse_view(g,1);
    for(int j=0;j<TI_COUNT;j++)r->index->buffers[j]=reuse_view(g,1);
    for(int j=0;j<TA_COUNT;j++)r->attention->buffers[j]=reuse_view(g,1);
    for(int j=0;j<TP_COUNT;j++)r->producer->buffers[j]=reuse_view(g,j<TP_MAIN_FREQ);
    for(int j=0;j<TE_COUNT;j++)r->engram->buffers[j]=reuse_view(g,1);
    for(int j=0;j<TH_COUNT;j++)r->hc->buffers[j]=reuse_view(g,1);
    for(int l=0;l<2;l++) {
        TrellisKVSlot *k=r->kv->slots+l;k->cursor=3;k->generation=17;k->rate=1;
        k->c=reuse_view(g,1);k->r=reuse_view(g,1);k->idx=reuse_view(g,1);k->ring_kv=reuse_view(g,1);k->ring_gate=reuse_view(g,1);
        r->swa->slots[l].cursor=3;r->swa->slots[l].beg=1;r->swa->slots[l].ring=reuse_view(g,1);
        r->producer->norms[l]=reuse_view(g,0);r->attention->used[l]=11;
    }
    r->swa->scratch=reuse_view(g,1);
    assert(!bn_backend_session_attach_gpu_resources(bn_session_backend(s),&native_trellis_session_owner,r,trellis_request_free));return s;
}
static void reuse_free(BnSession *s){free(s->state.logits);free(s->state.token_history);bn_session_free(s,NULL);}
int main(void) {
    BnTransformerTrellisRequestConfig config={2,512,7,1,0};
    BnTransformerTrellisOptions options=trellis_full_options(&config);
    assert(options.max_tokens==2&&options.fused_window==7&&options.f16acc_enabled==1&&
        !options.device_budget&&!options.host_budget);
    ReuseGPU *g=malloc(sizeof(*g));assert(g);TrellisRequest *r;BnSession *s=reuse_fixture(g,&r);
    BnTransformerTrellisForwardResult old={g,s,7,3,2,1,1,4,r->forward->logits,s->state.logits};
    assert(bn_transformer_trellis_forward_current(&reuse_model,s,&old));
    int mutable=0;for(int j=0;j<g->slots;j++)mutable+=g->mutable[j];
    assert(!bn_transformer_trellis_session_restart(&reuse_model,s)&&g->zero==mutable&&g->sync==2&&!g->destroyed);
    assert(r->request==g&&r->generation==8&&s->backend_state->native_request_generation==8&&!s->pos);
    assert(!r->index->active&&r->index->generation==11&&!r->hc->phase&&!r->forward->completed&&r->forward->epoch==3);
    assert(!bn_transformer_trellis_forward_current(&reuse_model,s,&old));
    for(int j=0;j<g->slots;j++)for(int k=0;k<16;k++)assert(g->data[j][k]==(g->mutable[j]&&k<(j%2?16:8)?0:0xa5));
    for(int j=0;j<8;j++)assert(!s->state.token_history[j]);
    for(int j=0;j<4;j++)assert(!s->state.logits[j]);
    assert(!bn_transformer_trellis_session_restart(&reuse_model,s)&&r->generation==9);
    assert(!bn_transformer_trellis_index_begin(&reuse_model,s,0,1)&&r->index->generation>11);
    reuse_free(s);assert(g->destroyed==1);
    /* Every rejected boundary does no GPU work or destruction. */
    for(int failure=0;failure<7;failure++) {
        s=reuse_fixture(g,&r);
        if(failure==0)reuse_history_staged=1;
        if(failure==1)r->forward->phase=1;
        if(failure==2)r->hc->phase=1;
        if(failure==3)r->block_layer=1;
        if(failure==4)s->pos=4;
        if(failure==5)r->generation=UINT64_MAX;
        if(failure==6)r->index->generation=UINT64_MAX;
        assert(bn_transformer_trellis_session_restart(&reuse_model,s)==-1&&!g->zero&&!g->sync&&!g->destroyed);
        reuse_free(s);assert(g->destroyed==1);
    }
    for(int failure=0;failure<6;failure++) {
        s=reuse_fixture(g,&r);
        if(failure==0)g->fail_sync=1;
        if(failure==1)g->fail_sync=2;
        if(failure==2)g->fail_zero=1;
        if(failure==3)g->fail_zero=mutable/2;
        if(failure==4)g->fail_zero=mutable;
        if(failure==5)reuse_history_failure=1;
        assert(bn_transformer_trellis_session_restart(&reuse_model,s)==-1&&g->destroyed==1&&s->pos==3);
        assert(s->state.logits[0]==17&&s->state.token_history[0]==11&&!bn_backend_session_has_gpu_resources(bn_session_backend(s)));
        int zero=g->zero,sync=g->sync;
        assert(bn_transformer_trellis_session_restart(&reuse_model,s)==-1&&g->zero==zero&&g->sync==sync);
        reuse_free(s);assert(g->destroyed==1);
    }
    free(g);puts("Native restart: mutable/immutable storage, stale epochs, healthy/partial boundaries, zero/fence/history failure invalidation PASSED");return 0;
}
