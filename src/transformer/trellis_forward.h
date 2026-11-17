#ifndef BN_TRANSFORMER_TRELLIS_FORWARD_H
#define BN_TRANSFORMER_TRELLIS_FORWARD_H
struct TrellisForward {
    BnTransformerTrellisForwardWorkspace workspace;
    BnGPUResidentView logits;
    BnSessionNativeHistoryView history;
    void *weight;
    uint16_t *last;
    uint64_t epoch,completed;
    int phase,admit,layer,all_logits;
    BnSession *group_owner,*group_members[BN_TRANSFORMER_TRELLIS_GROUP_MAX];
    uint64_t group_generations[BN_TRANSFORMER_TRELLIS_GROUP_MAX];
    int group_count;
    BnTransformerTrellisGroupStats group_stats;
};
static int trellis_forward_shape(const BnModel*m,const BnTransformerTrellisOptions*ffn,
    const BnTransformerTrellisForwardOptions*o,BnTransformerTrellisForwardWorkspace*out,void**weight) {
    BnModelNativeHead head;BnTransformerTrellisHCWorkspace hw;BnGPUBackend*g=bn_model_gpu(m);
    if(!o||!out||(o->all_logits!=0&&o->all_logits!=1)||!g||!g->linear_f16_resident||!g->resident_read||
       bn_model_native_head(m,&head)||head.profile!=BN_NATIVE_HEAD_F16_LINEAR_F16||
       bn_transformer_trellis_hc_session_plan(m,ffn,&hw))return -1;
    const BnBackendNativeParameter*w=bn_backend_layout_native_parameter(bn_model_backend(m),-1,BN_NATIVE_GLOBAL_OUTPUT);
    if(!w||!w->handle||w->layout!=BN_NATIVE_PARAMETER_F16_LINEAR||
       w->type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16)||w->rows!=head.vocab||w->cols!=head.dim)return -1;
    int rows=o->all_logits?ffn->max_tokens:1;size_t device=0,host=sizeof(TrellisForward);
    if((size_t)rows>INT_MAX/(size_t)head.vocab||trellis_add_bytes(&device,(size_t)rows*head.vocab,2)||
       trellis_add_bytes(&host,(size_t)head.vocab,2))return -1;
    if(weight)*weight=w->handle;
    *out=(BnTransformerTrellisForwardWorkspace){device,host,ffn->max_tokens,head.vocab,rows};return 0;
}
int bn_transformer_trellis_forward_session_plan(const BnModel*m,const BnTransformerTrellisOptions*ffn,
    const BnTransformerTrellisForwardOptions*o,BnTransformerTrellisForwardWorkspace*out) {
    return trellis_forward_shape(m,ffn,o,out,NULL);
}
static int trellis_forward_setup(TrellisRequest*r,const BnTransformerTrellisForwardOptions*o,
    const BnTransformerTrellisForwardWorkspace*w) {
    BnTransformerTrellisForwardWorkspace check;void*weight;
    if(!r->hc||!r->producer||trellis_forward_shape(r->model,&r->options,o,&check,&weight))return -1;
    TrellisForward*f=calloc(1,w->host_bytes);if(!f)return -1;r->forward=f;
    f->workspace=*w;f->weight=weight;f->last=(uint16_t*)(f+1);f->all_logits=o->all_logits;
    return r->gpu->resident_alloc(r->request,bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16),
        (size_t)w->vocab*w->logit_rows,&f->logits);
}
static BnGPUResidentView trellis_forward_logits(TrellisForward*f) {
    return bn_gpu_resident_view_slice(f->logits,0,(size_t)(f->all_logits?f->history.tokens:1)*f->workspace.vocab);
}
int bn_transformer_trellis_forward_run(const BnModel*m,BnSession*s,int admit,BnTransformerTrellisForwardResult*out) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisForward*f=r?r->forward:NULL;BnSessionNativeHistoryView history;
    if(!f||f->group_owner||!out||!s->state.logits||(admit!=0&&admit!=1))return -1;
    if(f->phase) {
        if(admit!=f->admit)return -1;
        if(bn_session_native_history_view(s,m,&history)||history.epoch!=f->history.epoch||
           history.tokens!=f->history.tokens||history.position!=f->history.position||
           history.raw!=f->history.raw||history.mapped!=f->history.mapped)return trellis_hc_fail(s);
    }else {
        if(f->epoch==UINT64_MAX||bn_session_native_history_view(s,m,&history)||
           history.tokens<1||history.tokens>f->workspace.max_tokens||!r->index||!r->kv||
           history.tokens>r->index->workspace.max_tokens||history.position>r->kv->maxseq-history.tokens||
           r->index->generation==UINT64_MAX||(r->hc->phase!=0&&r->hc->phase!=3)||r->block_layer)return -1;
        f->history=history;f->epoch++;f->admit=admit;f->layer=0;f->phase=1;
    }
    if(f->phase==1) {
        int status=bn_transformer_trellis_hc_begin(m,s);
        if(status==BN_GPU_UPLOAD_BUSY)return trellis_request_get(m,s)?status:trellis_hc_fail(s);
        if(status)return trellis_hc_fail(s);
        if(bn_transformer_trellis_index_begin(m,s,f->history.position,f->history.tokens))return trellis_hc_fail(s);
        f->phase=2;
    }
    while(f->layer<r->hc->workspace.layers) {
        BnTransformerTrellisBlockResult block;
        int status=bn_transformer_trellis_block_step(m,s,f->layer,admit,&block);
        if(status==BN_GPU_UPLOAD_BUSY)return trellis_request_get(m,s)?status:trellis_hc_fail(s);
        if(status)return trellis_hc_fail(s);
        f->layer++;
    }
    BnGPUResidentView collapsed,normalized;
    if(bn_transformer_trellis_hc_finish(m,s,&collapsed,&normalized))return trellis_hc_fail(s);
    int D=r->anatomy->experts.dim,V=f->workspace.vocab,R=f->history.tokens,rows=f->all_logits?R:1;
    if(!f->all_logits)normalized=bn_gpu_resident_view_slice(normalized,(size_t)(R-1)*D,(size_t)D);
    BnGPUResidentView logits=trellis_forward_logits(f),last=bn_gpu_resident_view_slice(logits,(size_t)(rows-1)*V,(size_t)V);
    BnGPUF16LinearResidentPlan head={f->weight,D,V,rows,normalized,logits};
    if(r->gpu->linear_f16_resident(r->gpu->ctx,r->request,&head)||
       r->gpu->resident_read(r->request,last,f->last))return trellis_hc_fail(s);
    for(int j=0;j<V;j++)if((f->last[j]&UINT16_C(0x7c00))==UINT16_C(0x7c00))return trellis_hc_fail(s);
    if(bn_session_native_history_commit(s,m,&f->history))return trellis_hc_fail(s);
    for(int j=0;j<V;j++)s->state.logits[j]=bn_fp16_to_fp32(f->last[j]);
    f->phase=0;f->completed=f->epoch;
    *out=(BnTransformerTrellisForwardResult){r->request,s,r->generation,f->epoch,f->history.position,R,rows,V,logits,s->state.logits};return 0;
}
int bn_transformer_trellis_forward_current(const BnModel*m,BnSession*s,const BnTransformerTrellisForwardResult*v) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisForward*f=r?r->forward:NULL;
    return f&&v&&v->epoch&&!f->phase&&v->session==s&&v->request==r->request&&v->generation==r->generation&&
        v->epoch==f->epoch&&v->epoch==f->completed&&v->position==f->history.position&&v->tokens==f->history.tokens&&
        v->vocab==f->workspace.vocab&&v->rows==(f->all_logits?f->history.tokens:1)&&v->host_logits==s->state.logits&&
        s->pos==f->history.position+f->history.tokens&&trellis_engram_same(v->logits,trellis_forward_logits(f));
}
#endif
