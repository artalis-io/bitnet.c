#ifndef BN_TRANSFORMER_TRELLIS_SWA_H
#define BN_TRANSFORMER_TRELLIS_SWA_H
/* Per-layer mutable source ring state, same session-owned resident request. */
typedef struct {int cursor,beg;BnGPUResidentView ring;} TrellisSWASlot;
struct TrellisSWA {
    BnTransformerTrellisSWAWorkspace workspace;
    BnGPUResidentView scratch;
    TrellisSWASlot slots[BN_MODEL_ANATOMY_MAX_LAYERS];
};
int bn_transformer_trellis_swa_session_plan(const BnModel*model,BnTransformerTrellisSWAWorkspace*out) {
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(model);BnGPUBackend*g=bn_model_gpu(model);
    if(!a||!g||!out||!g->swa_ring_update_f16_resident||a->experts.layers<1||
       a->experts.layers>BN_MODEL_ANATOMY_MAX_LAYERS||a->head_dim<2||a->head_dim>2048||a->head_dim%2)return -1;
    int B=bn_gpu_swa_ring_rows(a->sliding_window);if(!B)return -1;
    size_t device=0;
    if(trellis_add_bytes(&device,(size_t)B*a->head_dim*((size_t)a->experts.layers+1),2))return -1;
    BnTransformerTrellisSWAWorkspace w={device,sizeof(TrellisSWA),B,a->experts.layers};*out=w;return 0;
}
static int trellis_swa_setup(TrellisRequest*r,const BnTransformerTrellisSWAWorkspace*w) {
    TrellisSWA*s=calloc(1,sizeof(*s));if(!s)return -1;r->swa=s;s->workspace=*w;
    size_t count=(size_t)w->ring_rows*r->anatomy->head_dim;
    for(int l=0;l<w->layers;l++)if(r->gpu->resident_alloc(r->request,
        bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16),count,&s->slots[l].ring)||
        r->gpu->resident_zero(r->request,s->slots[l].ring))return -1;
    if(r->gpu->resident_alloc(r->request,bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16),count,&s->scratch))return -1;
    return r->gpu->resident_synchronize(r->request);
}
int bn_transformer_trellis_swa_views(const BnModel*model,BnSession*session,
    int layer,int pos0,BnTransformerTrellisSWAViews*out) {
    TrellisRequest*r=trellis_request_get(model,session);
    if(!r||!r->swa||!out||layer<0||layer>=r->anatomy->experts.layers||pos0<0)return -1;
    TrellisSWASlot*s=&r->swa->slots[layer];if(pos0<s->beg||pos0>s->cursor)return -1;
    int previous=r->anatomy->sliding_window-1;
    if(previous>pos0-s->beg){previous=pos0-s->beg;}
    if(previous>pos0){previous=pos0;}
    BnTransformerTrellisSWAViews v={r->request,s->cursor,s->beg,r->anatomy->sliding_window,
        r->swa->workspace.ring_rows,previous,pos0-previous,s->ring};*out=v;return 0;
}
int bn_transformer_trellis_swa_commit(const BnModel*model,BnSession*session,
    int layer,int pos0,int tokens,BnGPUResidentView kv) {
    TrellisRequest*r=trellis_request_get(model,session);BnTransformerTrellisSWAViews v;BnGPUSWARingUpdateInfo info;
    if(!r||!r->swa||!r->kv||bn_transformer_trellis_swa_views(model,session,layer,pos0,&v)||
       tokens<1||tokens>r->options.max_tokens||tokens>r->kv->maxseq||pos0>r->kv->maxseq-tokens||
       kv.request!=r->request||kv.type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16)||
       kv.count!=(size_t)tokens*r->anatomy->head_dim||
       bn_gpu_swa_ring_update_info(tokens,pos0,v.window_beg,v.window,v.ring_rows,&info))return -1;
    BnGPUSWARingUpdateResidentPlan p={tokens,pos0,v.window_beg,v.window,v.ring_rows,r->anatomy->head_dim,kv,v.ring,r->swa->scratch};
    if(r->gpu->swa_ring_update_f16_resident(r->gpu->ctx,r->request,&p)||r->gpu->resident_synchronize(r->request)) {
        r->destroy(r->request);r->request=NULL;return -1;
    }
    r->swa->slots[layer].cursor=pos0+tokens;r->swa->slots[layer].beg=info.next_beg;return 0;
}
#endif
