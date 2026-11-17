#ifndef BN_TRANSFORMER_TRELLIS_ATTENTION_H
#define BN_TRANSFORMER_TRELLIS_ATTENTION_H
/* Prepared execution: source selection and PRE-update ring must remain live
 * through projection. Publish only the layer ring cursor after completion. */
enum {TA_GROUP,TA_CONCAT,TA_DELTA,TA_ML,TA_ACC,TA_COUNT};
struct TrellisAttention {
    BnTransformerTrellisAttentionWorkspace workspace;
    BnGPUResidentView buffers[TA_COUNT];
    uint64_t used[BN_MODEL_ANATOMY_MAX_LAYERS];
};
static int trellis_attention_weights(const BnModel*m,int layer,
    const BnBackendNativeF16Weight**up,const BnBackendNativeF16Weight**down,
    const BnBackendNativeParameter**sink) {
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(m);BnGPUBackend*g=bn_model_gpu(m);
    *up=bn_backend_layout_native_core_weight(bn_model_backend(m),layer,BN_NATIVE_O_A);
    *down=bn_backend_layout_native_core_weight(bn_model_backend(m),layer,BN_NATIVE_O_B);
    *sink=bn_backend_layout_native_parameter(bn_model_backend(m),layer,BN_NATIVE_ATTN_SINK);
    return !a||!*up||!(*up)->weight||(*up)->gpu!=g||(*up)->rows!=a->o_groups*a->o_rank||
        (*up)->cols!=a->heads/a->o_groups*a->head_dim||!*down||!(*down)->weight||(*down)->gpu!=g||
        (*down)->rows!=a->experts.dim||(*down)->cols!=a->o_groups*a->o_rank||!*sink||!(*sink)->handle||
        (*sink)->layout!=BN_NATIVE_PARAMETER_RAW||(*sink)->type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32)||
        (*sink)->rows!=1||(*sink)->cols!=a->heads?-1:0;
}
static int trellis_attention_shape(const BnModel*m,const BnTransformerTrellisOptions*ffn,
    const BnTransformerTrellisKVOptions*kv,BnTransformerTrellisAttentionWorkspace*out,size_t*counts,int*types) {
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(m);BnGPUBackend*g=bn_model_gpu(m);
    BnTransformerTrellisIndexWorkspace iw;BnTransformerTrellisSWAWorkspace sw;
    if(!a||!g||!out||!g->latent_attention_f16_resident||!g->grouped_output_f16_resident||
       bn_transformer_trellis_index_session_plan(m,ffn,kv,&iw)||bn_transformer_trellis_swa_session_plan(m,&sw)||
       a->heads!=64||a->head_dim!=512||a->rope_dim!=64||a->o_groups!=8||a->o_rank<1)return -1;
    for(int l=0;l<a->experts.layers;l++) {
        const BnBackendNativeF16Weight*up,*down;const BnBackendNativeParameter*sink;
        if(trellis_attention_weights(m,l,&up,&down,&sink))return -1;
    }
    size_t n[TA_COUNT]={0};int R=iw.max_tokens;
    BnGPUF16GroupedOutputWorkspace gw;
    if(bn_gpu_f16_grouped_output_workspace(a->o_groups,4096,a->o_rank,R,&gw))return -1;
    n[TA_GROUP]=gw.input_count;n[TA_CONCAT]=gw.concat_count;n[TA_DELTA]=(size_t)R*a->experts.dim;
    /* Split choice depends on the live chunk length, position and dense versus
     * gathered pool, not just max_tokens. Reserve the maximum legal <=8 rows. */
    for(int rows=1;rows<=R&&rows<=8;rows++) {
        BnGPUF16LatentAttentionWorkspace w;
        if(bn_gpu_latent_attention_workspace(rows,iw.max_entries,1,1,a->sliding_window,kv->maxseq-rows,&w))return -1;
        if(w.ml_count>n[TA_ML])n[TA_ML]=w.ml_count;
        if(w.acc_count>n[TA_ACC])n[TA_ACC]=w.acc_count;
    }
    BnBackendScalarKind k[TA_COUNT]={BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32};size_t device=0;
    for(int j=0;j<TA_COUNT;j++)if(trellis_add_bytes(&device,n[j],bn_backend_quant_scalar_bytes(k[j])))return -1;
    if(counts)memcpy(counts,n,sizeof(n));
    if(types)for(int j=0;j<TA_COUNT;j++)types[j]=bn_backend_quant_scalar_type(k[j]);
    BnTransformerTrellisAttentionWorkspace w={device,sizeof(TrellisAttention),n[TA_ML],n[TA_ACC],R};*out=w;return 0;
}
int bn_transformer_trellis_attention_session_plan(const BnModel*m,
    const BnTransformerTrellisOptions*ffn,const BnTransformerTrellisKVOptions*kv,
    BnTransformerTrellisAttentionWorkspace*out) {return trellis_attention_shape(m,ffn,kv,out,NULL,NULL);}
static int trellis_attention_setup(TrellisRequest*r,const BnTransformerTrellisAttentionWorkspace*w) {
    TrellisAttention*a=calloc(1,sizeof(*a));if(!a)return -1;r->attention=a;a->workspace=*w;
    BnTransformerTrellisKVOptions kv={r->kv->maxseq,0,0};BnTransformerTrellisAttentionWorkspace check;size_t n[TA_COUNT];int t[TA_COUNT];
    if(trellis_attention_shape(r->model,&r->options,&kv,&check,n,t))return -1;
    for(int j=0;j<TA_COUNT;j++)if(n[j]&&r->gpu->resident_alloc(r->request,t[j],n[j],&a->buffers[j]))return -1;
    return r->gpu->resident_synchronize(r->request);
}
static int trellis_attention_same_view(BnGPUResidentView a,BnGPUResidentView b) {
    return a.request==b.request&&a.offset==b.offset&&a.count==b.count&&a.slot==b.slot&&a.type==b.type;
}
static BnGPUResidentView trellis_attention_view(TrellisRequest*r,int slot,size_t count) {
    return count?bn_gpu_resident_view_slice(r->attention->buffers[slot],0,count):(BnGPUResidentView){0};
}
static int trellis_attention_input(TrellisRequest*r,BnGPUResidentView q,BnGPUResidentView kv,BnGPUResidentView freq) {
    BnGPUResidentView v[]={q,kv,freq};size_t n[]={(size_t)r->index->tokens*64*512,(size_t)r->index->tokens*512,32};
    for(int j=0;j<3;j++) {
        if(v[j].request!=r->request||v[j].count!=n[j]||v[j].offset>SIZE_MAX-v[j].count||
           v[j].type!=bn_backend_quant_scalar_type(j==2?BN_BACKEND_SCALAR_F32:BN_BACKEND_SCALAR_F16))return -1;
        for(int k=0;k<j;k++)if(trellis_index_overlap(v[j],v[k]))return -1;
        for(int k=0;k<TA_COUNT;k++)if(trellis_index_overlap(v[j],r->attention->buffers[k]))return -1;
        for(int k=0;k<TI_COUNT;k++)if(trellis_index_overlap(v[j],r->index->buffers[k]))return -1;
        if(trellis_index_overlap(v[j],r->kv->block_table)||trellis_index_overlap(v[j],r->swa->scratch))return -1;
        for(int l=0;l<r->anatomy->experts.layers;l++) {
            const TrellisKVSlot*s=&r->kv->slots[l];BnGPUResidentView persistent[]={s->c,s->r,s->idx,s->ring_kv,s->ring_gate,r->swa->slots[l].ring};
            for(size_t k=0;k<sizeof(persistent)/sizeof(persistent[0]);k++)if(trellis_index_overlap(v[j],persistent[k]))return -1;
        }
    }return 0;
}
int bn_transformer_trellis_attention_project(const BnModel*m,BnSession*s,int layer,
    const BnTransformerTrellisIndexResult*selection,BnGPUResidentView q,BnGPUResidentView kv,
    BnGPUResidentView freq,BnTransformerTrellisAttentionResult*out) {
    TrellisRequest*r=trellis_request_get(m,s);
    if(!r||!r->attention||!r->index||!r->swa||!r->kv||!out||!selection||!r->index->active||
       layer<0||layer>=r->anatomy->experts.layers||selection->request!=r->request||
       selection->generation!=r->index->generation||r->attention->used[layer]==r->index->generation||
       trellis_attention_input(r,q,kv,freq))return -1;
    TrellisIndex*i=r->index;const BnModelNativeAnatomy*a=r->anatomy;const BnModelNativeLayerAnatomy*l=&a->layers[layer];
    BnTransformerTrellisSWAViews sw;BnGPUSWARingUpdateInfo next;
    if(bn_transformer_trellis_swa_views(m,s,layer,i->pos,&sw)||
       bn_gpu_swa_ring_update_info(i->tokens,i->pos,sw.window_beg,sw.window,sw.ring_rows,&next))return -1;
    BnTransformerTrellisKVViews pool={0};BnGPUResidentView ids={0};int entries=0,rate=1,dense=1;
    if(l->compress_ratio) {
        if(bn_transformer_trellis_kv_views(m,s,layer,&pool)||pool.cursor!=i->pos+i->tokens||
           r->kv->slots[pool.owner].generation<=i->seen[pool.owner]||selection->kv_owner!=pool.owner||
           selection->pool_generation!=r->kv->slots[pool.owner].generation||selection->entries!=pool.entries||selection->rate!=pool.rate||selection->source_layer!=l->index_source)return -1;
        entries=pool.entries;rate=pool.rate;
        if(entries<=512) {
            if(selection->mode!=BN_TRELLIS_INDEX_DENSE||selection->selection_revision||selection->borrowed||selection->k_len||!trellis_kv_empty(selection->indices))return -1;
        } else {
            ids=trellis_index_view(i,TI_IDS,(size_t)i->tokens*512);dense=0;
            if(!i->selection_valid||i->selection_source!=l->index_source||i->selection_owner!=pool.owner||
               i->selection_entries!=entries||i->selection_version!=r->kv->slots[pool.owner].generation||
               selection->selection_revision!=i->selection_revision||selection->mode!=i->mode||selection->borrowed!=(l->index_source!=layer)||selection->k_len!=512||
               !trellis_attention_same_view(selection->indices,ids))return -1;
        }
    } else if(selection->mode!=BN_TRELLIS_INDEX_SLIDING||selection->source_layer!=l->index_source||
        selection->kv_owner!=l->kv_source||selection->pool_generation||selection->selection_revision||selection->entries||selection->rate||selection->k_len||selection->borrowed||
        !trellis_kv_empty(selection->indices))return -1;
    const BnBackendNativeF16Weight*up,*down;const BnBackendNativeParameter*sink;BnGPUF16LatentAttentionWorkspace ws;
    if(trellis_attention_weights(m,layer,&up,&down,&sink)||
       bn_gpu_latent_attention_workspace(i->tokens,entries,rate,dense,sw.window,i->pos,&ws)||
       ws.ml_count>r->attention->workspace.ml_count||ws.acc_count>r->attention->workspace.acc_count)return -1;
    BnGPUResidentView group=trellis_attention_view(r,TA_GROUP,(size_t)i->tokens*64*512),
        concat=trellis_attention_view(r,TA_CONCAT,(size_t)i->tokens*a->o_groups*a->o_rank),
        delta=trellis_attention_view(r,TA_DELTA,(size_t)i->tokens*a->experts.dim);
    BnGPUF16LatentAttentionResidentPlan ap={i->tokens,entries,rate,dense,sw.window,i->pos,sw.window_beg,sw.win_floor,sw.ring_rows,
        pool.pages,pool.entries_per_page,sink->handle,q,sw.ring,kv,pool.pool_c,pool.pool_r,pool.block_table,ids,freq,group,
        trellis_attention_view(r,TA_ML,ws.ml_count),trellis_attention_view(r,TA_ACC,ws.acc_count)};
    BnGPUF16GroupedOutputResidentPlan op={up->weight,a->o_groups,4096,a->o_rank,i->tokens,group,
        {down->weight,a->o_groups*a->o_rank,a->experts.dim,i->tokens,concat,delta}};
    BnGPUSWARingUpdateResidentPlan rp={i->tokens,i->pos,sw.window_beg,sw.window,sw.ring_rows,a->head_dim,kv,sw.ring,r->swa->scratch};
    if(r->gpu->latent_attention_f16_resident(r->gpu->ctx,r->request,&ap)||
       r->gpu->grouped_output_f16_resident(r->gpu->ctx,r->request,&op)||
       r->gpu->swa_ring_update_f16_resident(r->gpu->ctx,r->request,&rp)||r->gpu->resident_synchronize(r->request)) {
        r->destroy(r->request);r->request=NULL;return -1;
    }
    r->swa->slots[layer].cursor=i->pos+i->tokens;r->swa->slots[layer].beg=next.next_beg;
    r->attention->used[layer]=i->generation;
    BnTransformerTrellisAttentionResult result={r->request,i->generation,layer,i->pos,i->tokens,delta};*out=result;return 0;
}
#endif
