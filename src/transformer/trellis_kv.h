#ifndef BN_TRANSFORMER_TRELLIS_KV_H
#define BN_TRANSFORMER_TRELLIS_KV_H
/* Private native request pool ownership; architecture supplies layer sharing. */
typedef struct {
    int rate,cursor;
    uint64_t generation;
    BnGPUResidentView c,r,idx,ring_kv,ring_gate;
} TrellisKVSlot;
struct TrellisKV {
    int maxseq;
    BnTransformerTrellisKVWorkspace workspace;
    BnGPUResidentView block_table;
    TrellisKVSlot slots[BN_MODEL_ANATOMY_MAX_LAYERS];
};
int bn_transformer_trellis_kv_session_plan(const BnModel*model,
    const BnTransformerTrellisKVOptions*o,BnTransformerTrellisKVWorkspace*out) {
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(model);BnGPUBackend*g=bn_model_gpu(model);
    if(!a||!g||!o||!out||!g->paged_latent_publish_f16_resident||!g->resident_zero||
       a->experts.layers<1||a->experts.layers>BN_MODEL_ANATOMY_MAX_LAYERS||
       a->head_dim<4||a->head_dim>2048||a->head_dim%2||a->rope_dim<2||a->rope_dim%2||a->rope_dim>=a->head_dim||
       a->index_dim<2||a->index_dim>2048||a->index_dim%2||
       o->maxseq<256||o->maxseq%256||o->maxseq>a->context)return -1;
    size_t device=0,host=sizeof(TrellisKV);int pages=o->maxseq/256,owners=0;
    if(trellis_add_bytes(&device,(size_t)pages,4)||trellis_add_bytes(&host,(size_t)pages,4))return -1;
    for(int l=0;l<a->experts.layers;l++) {
        int rate=a->layers[l].compress_ratio,owner=a->layers[l].kv_source;
        if(!rate){if(owner!=-1)return -1;continue;}
        if(rate<1||rate>256||256%rate||owner<0||owner>l||a->layers[owner].kv_source!=owner||
           a->layers[owner].compress_ratio!=rate)return -1;
        if(owner!=l)continue;
        if(a->layers[l].tensor[BN_NATIVE_INDEX_K]<0||a->layers[l].tensor[BN_NATIVE_INDEX_K_NORM]<0)return -1;
        size_t entries=(size_t)o->maxseq/rate,ring=(size_t)(256+rate)*a->head_dim;
        if(trellis_add_bytes(&device,entries*((size_t)a->head_dim+a->index_dim),2)||
           trellis_add_bytes(&device,ring,4))return -1;
        owners++;
    }
    if(!owners)return -1;
    BnTransformerTrellisKVWorkspace w={device,host,owners,pages,256};*out=w;return 0;
}
static int trellis_kv_setup(TrellisRequest*r,const BnTransformerTrellisKVOptions*o,
    const BnTransformerTrellisKVWorkspace*w) {
    TrellisKV*k=calloc(1,sizeof(*k));if(!k)return -1;r->kv=k;k->maxseq=o->maxseq;k->workspace=*w;
    int32_t*bt=malloc((size_t)w->pages*4);if(!bt)return -1;
    for(int i=0;i<w->pages;i++)bt[i]=i;
    int status=r->gpu->resident_alloc(r->request,bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32),(size_t)w->pages,&k->block_table);
    if(!status)status=r->gpu->resident_write(r->request,k->block_table,bt);
    free(bt);if(status)return -1;
    const BnModelNativeAnatomy*a=r->anatomy;
    for(int l=0;l<a->experts.layers;l++)if(a->layers[l].kv_source==l) {
        TrellisKVSlot*s=&k->slots[l];s->rate=a->layers[l].compress_ratio;
        size_t entries=(size_t)o->maxseq/s->rate,ring=(size_t)(256+s->rate)*a->head_dim;
        BnGPUResidentView*views[]={&s->c,&s->r,&s->idx,&s->ring_kv,&s->ring_gate};
        size_t counts[]={entries*(a->head_dim-a->rope_dim),entries*a->rope_dim,entries*a->index_dim,ring,ring};
        for(int i=0;i<5;i++)if(r->gpu->resident_alloc(r->request,bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16),counts[i],views[i])||
            r->gpu->resident_zero(r->request,*views[i]))return -1;
    }
    return r->gpu->resident_synchronize(r->request);
}
int bn_transformer_trellis_kv_views(const BnModel*model,BnSession*session,
    int layer,BnTransformerTrellisKVViews*out) {
    TrellisRequest*r=trellis_request_get(model,session);
    if(!r||!r->kv||!out||layer<0||layer>=r->anatomy->experts.layers)return -1;
    int owner=r->anatomy->layers[layer].kv_source;if(owner<0||owner>=r->anatomy->experts.layers)return -1;
    TrellisKVSlot*s=&r->kv->slots[owner];if(!s->rate)return -1;
    BnTransformerTrellisKVViews v={r->request,owner,s->rate,s->cursor,s->cursor/s->rate,
        256/s->rate,r->kv->workspace.pages,r->kv->block_table,s->c,s->r,s->idx,s->ring_kv,s->ring_gate};*out=v;return 0;
}
static int trellis_kv_empty(BnGPUResidentView v) {return !v.request&&!v.count&&!v.offset&&!v.slot&&!v.type;}
int bn_transformer_trellis_kv_publish(const BnModel*model,BnSession*session,
    int layer,int pos0,int tokens,BnGPUResidentView latents,BnGPUResidentView keys) {
    TrellisRequest*r=trellis_request_get(model,session);BnTransformerTrellisKVViews v;
    if(!r||!r->kv||bn_transformer_trellis_kv_views(model,session,layer,&v)||layer!=v.owner||
       tokens<1||tokens>r->options.max_tokens||tokens>r->kv->maxseq||pos0<0||
       pos0>r->kv->maxseq-tokens||pos0>v.cursor||v.cursor-pos0>256)return -1;
    if(r->kv->slots[layer].generation==UINT64_MAX)return -1;
    int rows=(pos0+tokens)/v.rate-pos0/v.rate;
    if(rows) {
        if(latents.request!=r->request||keys.request!=r->request||latents.type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16)||
           keys.type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16)||latents.count!=(size_t)rows*r->anatomy->head_dim||
           keys.count!=(size_t)rows*r->anatomy->index_dim)return -1;
        BnGPUPagedLatentPublishResidentPlan p={rows,pos0/v.rate,v.entries_per_page,v.pages,
            r->anatomy->head_dim,r->anatomy->rope_dim,r->anatomy->index_dim,
            latents,keys,v.block_table,v.pool_c,v.pool_r,v.pool_idx};
        if(r->gpu->paged_latent_publish_f16_resident(r->gpu->ctx,r->request,&p))goto failed;
    } else if(!trellis_kv_empty(latents)||!trellis_kv_empty(keys))return -1;
    if(r->gpu->resident_synchronize(r->request))goto failed;
    r->kv->slots[layer].cursor=pos0+tokens;r->kv->slots[layer].generation++;return 0;
failed:r->destroy(r->request);r->request=NULL;return -1;
}
#endif
