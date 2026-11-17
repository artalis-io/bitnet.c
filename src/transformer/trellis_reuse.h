/* Whole-request planning and reuse, within the transformer module. */
typedef struct {
    BnTransformerTrellisOptions f;
    BnTransformerTrellisKVOptions kv;
    BnTransformerTrellisSWAOptions swa;
    BnTransformerTrellisIndexOptions index;
    BnTransformerTrellisAttentionOptions attention;
    BnTransformerTrellisProducerOptions producer;
    BnTransformerTrellisRowsOptions rows;
    BnTransformerTrellisEngramOptions engram;
    BnTransformerTrellisHCOptions hc;
    BnTransformerTrellisForwardOptions forward;
    BnTransformerTrellisRequestPlan plan;
} TrellisFullSetup;
static int trellis_reuse_add(size_t *total,size_t value) {
    if(value>SIZE_MAX-*total)return -1;
    *total+=value;return 0;
}
static BnTransformerTrellisOptions trellis_full_options(const BnTransformerTrellisRequestConfig *c) {
    return (BnTransformerTrellisOptions){.max_tokens=c->max_tokens,.fused_window=c->fused_window,
        .f16acc_enabled=c->f16acc_enabled};
}
static int trellis_full_plan(const BnModel *m,const BnTransformerTrellisRequestConfig *c,TrellisFullSetup *out) {
    if(!c||!out||c->max_tokens<1||c->max_tokens>c->maxseq)return -1;
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(m);
    if(!a||a->experts.layers<1||a->experts.layers>BN_MODEL_ANATOMY_MAX_LAYERS||a->vocab<1||a->context<1)return -1;
    int heads=0;
    for(int l=0;l<a->experts.layers;l++)if(a->layers[l].engram_rows) {
        BnModelNativeEngramMath math;
        if(bn_model_native_engram_math(m,l,&math)||math.heads<1)return -1;
        if(math.heads>heads)heads=math.heads;
    }
    if(!heads||c->max_tokens>INT_MAX/heads)return -1;
    TrellisFullSetup s={0};s.f=trellis_full_options(c);
    s.kv=(BnTransformerTrellisKVOptions){c->maxseq,0,0};
    s.rows=(BnTransformerTrellisRowsOptions){c->max_tokens*heads,2,0,0,0};
    s.forward=(BnTransformerTrellisForwardOptions){c->all_logits,0,0};
    BnTransformerTrellisWorkspace f;BnTransformerTrellisKVWorkspace k;BnTransformerTrellisSWAWorkspace w;
    BnTransformerTrellisIndexWorkspace i;BnTransformerTrellisAttentionWorkspace t;BnTransformerTrellisProducerWorkspace p;
    BnTransformerTrellisRowsWorkspace r;BnTransformerTrellisEngramWorkspace e;BnTransformerTrellisHCWorkspace h;
    BnTransformerTrellisForwardWorkspace v;
    if(bn_transformer_trellis_session_plan(m,&s.f,&f)||bn_transformer_trellis_kv_session_plan(m,&s.kv,&k)||
       bn_transformer_trellis_swa_session_plan(m,&w)||bn_transformer_trellis_index_session_plan(m,&s.f,&s.kv,&i)||
       bn_transformer_trellis_attention_session_plan(m,&s.f,&s.kv,&t)||
       bn_transformer_trellis_producer_session_plan(m,&s.f,&s.kv,&p)||
       bn_transformer_trellis_rows_session_plan(m,&s.rows,&r)||
       bn_transformer_trellis_engram_session_plan(m,&s.f,&s.rows,&e)||
       bn_transformer_trellis_hc_session_plan(m,&s.f,&h)||bn_transformer_trellis_forward_session_plan(m,&s.f,&s.forward,&v))return -1;
    s.f.device_budget=f.device_bytes;s.f.host_budget=f.host_bytes;s.kv.device_budget=k.device_bytes;s.kv.host_budget=k.host_bytes;
    s.swa=(BnTransformerTrellisSWAOptions){w.device_bytes,w.host_bytes};s.index=(BnTransformerTrellisIndexOptions){i.device_bytes,i.host_bytes};
    s.attention=(BnTransformerTrellisAttentionOptions){t.device_bytes,t.host_bytes};s.producer=(BnTransformerTrellisProducerOptions){p.device_bytes,p.host_bytes};
    s.rows.host_budget=r.host_bytes;s.rows.device_budget=r.device_bytes;s.rows.pinned_budget=r.pinned_bytes;
    s.engram=(BnTransformerTrellisEngramOptions){e.device_bytes,e.host_bytes};s.hc=(BnTransformerTrellisHCOptions){h.device_bytes,h.host_bytes};
    s.forward.device_budget=v.device_bytes;s.forward.host_budget=v.host_bytes;
    size_t device[]={f.device_bytes,k.device_bytes,w.device_bytes,i.device_bytes,t.device_bytes,p.device_bytes,r.device_bytes,e.device_bytes,h.device_bytes,v.device_bytes};
    size_t host[]={f.host_bytes,k.host_bytes,w.host_bytes,i.host_bytes,t.host_bytes,p.host_bytes,r.host_bytes,e.host_bytes,h.host_bytes,v.host_bytes,e.history_bytes};
    for(size_t n=0;n<sizeof(device)/sizeof(device[0]);n++)if(trellis_reuse_add(&s.plan.device_bytes,device[n]))return -1;
    for(size_t n=0;n<sizeof(host)/sizeof(host[0]);n++)if(trellis_reuse_add(&s.plan.host_bytes,host[n]))return -1;
    if(trellis_add_bytes(&s.plan.host_bytes,(size_t)a->vocab,sizeof(float))||
       trellis_add_bytes(&s.plan.host_bytes,(size_t)a->context,sizeof(int32_t))||
       trellis_reuse_add(&s.plan.host_bytes,2*SH_ARENA_ALIGN))return -1;
    s.plan.pinned_bytes=r.pinned_bytes;s.plan.kv_bytes=k.device_bytes;s.plan.history_bytes=e.history_bytes;
    *out=s;return 0;
}
int bn_transformer_trellis_request_plan(const BnModel *m,const BnTransformerTrellisRequestConfig *c,
    BnTransformerTrellisRequestPlan *out) {
    TrellisFullSetup setup;if(!out||trellis_full_plan(m,c,&setup))return -1;
    *out=setup.plan;return 0;
}
int bn_transformer_trellis_prepare_request(const BnModel *m,BnSession *s,const BnTransformerTrellisRequestConfig *c) {
    TrellisFullSetup p;
    if(!s||!s->backend_only||!bn_session_matches_model(s,m)||s->pos||
       bn_backend_session_has_gpu_resources(bn_session_backend(s))||
       bn_backend_session_has_execution_resources(bn_session_backend(s))||
       s->backend_state->native_history||trellis_full_plan(m,c,&p))return -1;
    if(bn_session_native_history_prepare(s,m,c->max_tokens,p.plan.history_bytes))return -1;
    return bn_transformer_trellis_prepare_session_forward(m,s,&p.f,&p.kv,&p.swa,&p.index,
        &p.attention,&p.producer,&p.rows,&p.engram,&p.hc,&p.forward);
}
static int trellis_reuse_zero(TrellisRequest *r,const BnGPUResidentView *views,int count) {
    for(int i=0;i<count;i++)if(views[i].count&&r->gpu->resident_zero(r->request,views[i]))return -1;
    return 0;
}
int bn_transformer_trellis_session_restart(const BnModel *m,BnSession *s) {
    TrellisRequest *r=trellis_request_get(m,s);BnSessionNativeHistoryInfo history;BnBackendRowStageInfo rows;
    if(!r||!r->kv||!r->swa||!r->index||!r->attention||!r->producer||!r->engram||!r->hc||!r->forward||
       !r->gpu->resident_zero||!r->gpu->resident_synchronize||!s->state.logits||!s->state.token_history||
       bn_session_native_history_info(s,m,&history)||history.staged||history.committed!=s->pos||
       bn_backend_session_row_stage_info(&r->rows,&rows)||rows.failed||r->block_layer||r->forward->phase||
       r->generation!=s->backend_state->native_request_generation||r->generation==UINT64_MAX||
       r->forward->epoch==UINT64_MAX||r->index->generation==UINT64_MAX||r->engram->epoch==UINT64_MAX||
       r->hc->epoch>UINT64_MAX-(uint64_t)(2*r->anatomy->experts.layers+1))return -1;
    TrellisForward *f=r->forward;TrellisHC *h=r->hc;TrellisIndex *i=r->index;
    if(h->phase==0) {
        if(s->pos||f->completed||f->history.tokens||i->active)return -1;
    }else {
        BnTransformerTrellisForwardResult last={r->request,s,r->generation,f->epoch,f->history.position,
            f->history.tokens,f->all_logits?f->history.tokens:1,f->workspace.vocab,trellis_forward_logits(f),s->state.logits};
        if(h->phase!=3||h->site!=2*r->anatomy->experts.layers||h->tokens!=f->history.tokens||
           h->position!=f->history.position||h->history_epoch!=f->history.epoch||
           r->block_index_generation!=i->generation||!bn_transformer_trellis_forward_current(m,s,&last))return -1;
    }
    for(int l=0;l<r->anatomy->experts.layers;l++)if(r->kv->slots[l].generation==UINT64_MAX)return -1;
    /* Quiesce uploads/consumers before clearing; immutable page tables,
     * generated rotary frequencies and converted norm weights remain intact. */
    if(r->gpu->resident_synchronize(r->request)||trellis_reuse_zero(r,r->buffers,TR_COUNT))goto fail;
    for(int l=0;l<r->anatomy->experts.layers;l++) {
        TrellisKVSlot *k=r->kv->slots+l;
        BnGPUResidentView views[]={k->c,k->r,k->idx,k->ring_kv,k->ring_gate,r->swa->slots[l].ring};
        if(trellis_reuse_zero(r,views,6))goto fail;
    }
    if(trellis_reuse_zero(r,&r->swa->scratch,1)||trellis_reuse_zero(r,i->buffers,TI_COUNT)||
       trellis_reuse_zero(r,r->attention->buffers,TA_COUNT)||
       trellis_reuse_zero(r,r->producer->buffers,TP_MAIN_FREQ)||
       trellis_reuse_zero(r,r->engram->buffers,TE_COUNT)||
       trellis_reuse_zero(r,h->buffers,TH_COUNT)||trellis_reuse_zero(r,&f->logits,1)||
       r->gpu->resident_synchronize(r->request)||bn_session_native_history_restart(s,m))goto fail;
    for(int l=0;l<r->anatomy->experts.layers;l++) {
        r->kv->slots[l].cursor=0;r->swa->slots[l].cursor=r->swa->slots[l].beg=0;
        i->seen[l]=0;r->attention->used[l]=0;
    }
    i->active=i->pos=i->tokens=i->selection_valid=i->selection_source=i->selection_owner=i->selection_entries=0;
    i->candidate_valid=i->candidate_owner=i->candidate_entries=i->candidate_rate=0;
    i->selection_version=i->candidate_version=i->selection_revision=0;i->mode=BN_TRELLIS_INDEX_SLIDING;
    r->engram->last_layer=-1;r->engram->last_tokens=0;r->engram->last_bank=0;
    h->phase=h->site=h->tokens=h->position=0;h->stream=TH_STREAM0;h->history_epoch=0;
    f->history=(BnSessionNativeHistoryView){0};f->completed=0;f->phase=f->admit=f->layer=0;
    memset(f->last,0,(size_t)f->workspace.vocab*sizeof(*f->last));
    memset(s->state.logits,0,(size_t)f->workspace.vocab*sizeof(*s->state.logits));
    memset(r->metadata,0,(6*(size_t)r->anatomy->experts.experts+2)*sizeof(int));
    if(r->plan.groups)memset(r->plan.groups,0,r->plan.group_capacity*sizeof(*r->plan.groups));
    r->plan.group_count=r->plan.slot_count=r->plan.fused_count=r->plan.batch_cap=0;
    memset(r->decode_ids,0,sizeof(r->decode_ids));memset(r->decode_weights,0,sizeof(r->decode_weights));
    for(int bank=0;bank<4;bank++)r->row_layers[bank]=-1;
    r->block_layer=r->block_phase=r->block_admit=0;r->block_epoch=r->block_index_generation=0;
    r->generation++;s->backend_state->native_request_generation=r->generation;
    s->gpu_kv_direct_valid=s->moe_cache_prefill=s->state.batched_prompt_contract=0;
    bn_backend_session_set_gpu_decode_ssm_pending(bn_session_backend(s),0);
    return 0;
fail:return trellis_hc_fail(s);
}
