#ifndef BN_TRANSFORMER_TRELLIS_INDEX_H
#define BN_TRANSFORMER_TRELLIS_INDEX_H
/* Mutable index orchestration belongs to the native session request. */
enum {TI_Q,TI_W,TI_SCORE,TI_IDS,TI_CAND,TI_BLOCK_SCORE,TI_BLOCK_IDS,
    TI_ENTRY_IDS,TI_ENTRY_COUNT,TI_GATHER,TI_BLOCK_COUNTS,TI_BLOCK_OFFSETS,
    TI_TILE_OFFSETS,TI_SELECT_IDS,TI_SELECT_SCORES,TI_SELECT_COUNTS,TI_COUNT};
struct TrellisIndex {
    BnTransformerTrellisIndexWorkspace workspace;
    BnGPUResidentView buffers[TI_COUNT];
    uint64_t generation,seen[BN_MODEL_ANATOMY_MAX_LAYERS],selection_version,candidate_version,selection_revision;
    int active,pos,tokens,selection_valid,selection_source,selection_owner,selection_entries;
    int candidate_valid,candidate_owner,candidate_entries,candidate_rate;
    BnTransformerTrellisIndexMode mode;
};
static int trellis_index_weights(const BnModel*m,int l,const BnBackendNativeF16Weight**q,
    const BnBackendNativeParameter**w) {
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(m);BnGPUBackend*g=bn_model_gpu(m);
    *q=bn_backend_layout_native_core_weight(bn_model_backend(m),l,BN_NATIVE_INDEX_Q);
    *w=bn_backend_layout_native_parameter(bn_model_backend(m),l,BN_NATIVE_INDEX_WEIGHTS);
    return !a||!*q||!(*q)->weight||(*q)->gpu!=g||(*q)->rows!=a->index_heads*a->index_dim||(*q)->cols!=a->q_rank||
        !*w||!(*w)->handle||(*w)->layout!=BN_NATIVE_PARAMETER_F16_LINEAR||
        (*w)->type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16)||
        (*w)->rows!=a->index_heads||(*w)->cols!=a->experts.dim?-1:0;
}
static void trellis_index_max_select(BnGPUF16IndexTopKWorkspace*max,const BnGPUF16IndexTopKWorkspace*w) {
    if(w->indices_count>max->indices_count)max->indices_count=w->indices_count;
    if(w->scores_count>max->scores_count)max->scores_count=w->scores_count;
    if(w->counts_count>max->counts_count)max->counts_count=w->counts_count;
}
static int trellis_index_shape(const BnModel*m,const BnTransformerTrellisOptions*ffn,
    const BnTransformerTrellisKVOptions*kv,BnTransformerTrellisIndexWorkspace*out,size_t*counts,int*types) {
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(m);BnGPUBackend*g=bn_model_gpu(m);
    BnTransformerTrellisWorkspace fw;BnTransformerTrellisKVWorkspace kw;
    if(!a||!g||!out||trellis_request_shape(m,ffn,&fw,NULL,NULL)||
       bn_transformer_trellis_kv_session_plan(m,kv,&kw)||a->index_heads!=32||a->index_dim!=128||
       a->index_topk!=512||a->q_rank<1||a->q_rank>32768||a->rope_dim<2||a->rope_dim>128||a->rope_dim%2)return -1;
    int R=ffn->max_tokens;if(R>kv->maxseq)R=kv->maxseq;int S=R>272?272:R,T=0,C=0,use=0;
    for(int l=0;l<a->experts.layers;l++) {
        const BnModelNativeLayerAnatomy*v=&a->layers[l];if(!v->compress_ratio)continue;
        int s=v->index_source;
        if(s<0||s>l||a->layers[s].index_source!=s||a->layers[s].kv_source!=v->kv_source)return -1;
        int entries=kv->maxseq/v->compress_ratio;if(entries>T)T=entries;
        if(s==l&&entries>512){const BnBackendNativeF16Weight*q;const BnBackendNativeParameter*w;if(trellis_index_weights(m,l,&q,&w))return -1;}
    }
    if(a->candidate_source>=0) {
        int c=a->candidate_source;if(c>=a->experts.layers||a->candidate_blocks!=2048||a->candidate_block_size!=8||
            a->layers[c].index_source!=c||a->layers[c].candidate_mode!=BN_NATIVE_CANDIDATE_SOURCE||a->layers[c].compress_ratio<1)return -1;
        C=kv->maxseq/a->layers[c].compress_ratio;
        for(int l=0;l<a->experts.layers;l++)if(a->layers[l].candidate_mode==BN_NATIVE_CANDIDATE_USE){
            if(a->layers[l].index_source!=l||a->layers[l].compress_ratio!=a->layers[c].compress_ratio)return -1;
            use=1;}
    }
    if(T<1||T>1048576)return -1;
    if(T>512&&(!g->index_query_f16_resident||!g->index_score_f16_resident||!g->index_topk_f16_resident))return -1;
    if(C>16384&&(!g->index_candidate_source_f16_resident||(use&&!g->index_candidate_use_f16_resident)))return -1;
    int stride=((T+127)/128)*128,B=C/8+1,bs=((B+127)/128)*128;
    size_t n[TI_COUNT]={0};BnGPUF16IndexTopKWorkspace peak={0};
    if(T>512){n[TI_Q]=(size_t)R*4096;n[TI_W]=(size_t)R*32;n[TI_SCORE]=(size_t)S*stride;n[TI_IDS]=(size_t)R*512;}
    if(C>16384){n[TI_CAND]=(size_t)R*B;n[TI_BLOCK_SCORE]=(size_t)S*bs;n[TI_BLOCK_IDS]=(size_t)S*2048;}
    if(C>16384&&use){int cap=S*16384;if(cap>C)cap=C;n[TI_ENTRY_IDS]=cap;n[TI_ENTRY_COUNT]=1;n[TI_GATHER]=(size_t)cap*128;n[TI_BLOCK_COUNTS]=B;n[TI_BLOCK_OFFSETS]=B;n[TI_TILE_OFFSETS]=(B+255)/256;}
    for(int rows=1;rows<=S&&T>512;rows++) {
        BnGPUF16IndexTopKWorkspace w;if(bn_gpu_index_topk_workspace(rows,T,stride,512,512,&w))return -1;trellis_index_max_select(&peak,&w);
        if(C>16384){if(bn_gpu_index_topk_workspace(rows,B,bs,2048,2048,&w))return -1;trellis_index_max_select(&peak,&w);}
        if(C>16384&&use){BnGPUIndexCandidateUseWorkspace wuse;int cap=rows*16384;if(cap>C)cap=C;
            if(bn_gpu_index_candidate_use_workspace(rows,C,((cap+127)/128)*128,0,1,&wuse))return -1;
            trellis_index_max_select(&peak,&wuse.select);}
    }
    n[TI_SELECT_IDS]=peak.indices_count;n[TI_SELECT_SCORES]=peak.scores_count;n[TI_SELECT_COUNTS]=peak.counts_count;
    BnBackendScalarKind k[TI_COUNT]={BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_I32,
        BN_BACKEND_SCALAR_BYTES,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,
        BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,
        BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_I32};size_t device=0;
    for(int i=0;i<TI_COUNT;i++)if(trellis_add_bytes(&device,n[i],bn_backend_quant_scalar_bytes(k[i])))return -1;
    if(counts){memcpy(counts,n,sizeof(n));}if(types)for(int i=0;i<TI_COUNT;i++)types[i]=bn_backend_quant_scalar_type(k[i]);
    BnTransformerTrellisIndexWorkspace w={device,sizeof(TrellisIndex),R,S,T,C,stride};*out=w;return 0;
}
int bn_transformer_trellis_index_session_plan(const BnModel*m,const BnTransformerTrellisOptions*ffn,
    const BnTransformerTrellisKVOptions*kv,BnTransformerTrellisIndexWorkspace*out) {
    return trellis_index_shape(m,ffn,kv,out,NULL,NULL);
}
static int trellis_index_setup(TrellisRequest*r,const BnTransformerTrellisIndexWorkspace*w) {
    TrellisIndex*i=calloc(1,sizeof(*i));if(!i)return -1;r->index=i;i->workspace=*w;
    BnTransformerTrellisKVOptions kv={r->kv->maxseq,0,0};BnTransformerTrellisIndexWorkspace check;size_t n[TI_COUNT];int t[TI_COUNT];
    if(trellis_index_shape(r->model,&r->options,&kv,&check,n,t))return -1;
    for(int j=0;j<TI_COUNT;j++)if(n[j]&&r->gpu->resident_alloc(r->request,t[j],n[j],&i->buffers[j]))return -1;
    return r->gpu->resident_synchronize(r->request);
}
static BnGPUResidentView trellis_index_view(TrellisIndex*i,int slot,size_t n) {
    return n?bn_gpu_resident_view_slice(i->buffers[slot],0,n):(BnGPUResidentView){0};
}
int bn_transformer_trellis_index_begin(const BnModel*m,BnSession*s,int pos,int tokens) {
    TrellisRequest*r=trellis_request_get(m,s);
    if(!r||!r->kv||!r->index||pos<0||tokens<1||tokens>r->index->workspace.max_tokens||
       tokens>r->kv->maxseq||pos>r->kv->maxseq-tokens||r->index->generation==UINT64_MAX)return -1;
    TrellisIndex*i=r->index;i->generation++;i->active=1;i->pos=pos;i->tokens=tokens;i->selection_valid=0;i->candidate_valid=0;i->selection_revision=0;
    for(int l=0;l<r->anatomy->experts.layers;l++)i->seen[l]=r->kv->slots[l].generation;
    return 0;
}
static int trellis_index_overlap(BnGPUResidentView a,BnGPUResidentView b) {
    return a.count&&b.count&&a.request==b.request&&a.slot==b.slot&&
        a.offset<b.offset+b.count&&b.offset<a.offset+a.count;
}
static int trellis_index_input(TrellisRequest*r,BnGPUResidentView x,BnGPUResidentView q,BnGPUResidentView freq) {
    BnGPUResidentView v[]={x,q,freq};size_t n[]={(size_t)r->index->tokens*r->anatomy->experts.dim,(size_t)r->index->tokens*r->anatomy->q_rank,(size_t)r->anatomy->rope_dim/2};
    for(int j=0;j<3;j++){
        if(v[j].request!=r->request||v[j].count!=n[j]||v[j].type!=bn_backend_quant_scalar_type(j==2?BN_BACKEND_SCALAR_F32:BN_BACKEND_SCALAR_F16)||v[j].offset>SIZE_MAX-v[j].count)return -1;
        for(int k=0;k<j;k++)if(trellis_index_overlap(v[j],v[k]))return -1;
        for(int k=0;k<TI_COUNT;k++)if(trellis_index_overlap(v[j],r->index->buffers[k]))return -1;
    }return 0;
}
int bn_transformer_trellis_index_select(const BnModel*m,BnSession*s,int layer,
    BnGPUResidentView x,BnGPUResidentView qres,BnGPUResidentView freq,BnTransformerTrellisIndexResult*out) {
    TrellisRequest*r=trellis_request_get(m,s);
    if(!r||!r->index||!r->kv||!out||!r->index->active||layer<0||layer>=r->anatomy->experts.layers)return -1;
    TrellisIndex*i=r->index;const BnModelNativeAnatomy*a=r->anatomy;const BnModelNativeLayerAnatomy*l=&a->layers[layer];
    BnTransformerTrellisIndexResult result={0};result.request=r->request;result.generation=i->generation;result.source_layer=l->index_source;result.kv_owner=l->kv_source;
    if(!l->compress_ratio){result.mode=BN_TRELLIS_INDEX_SLIDING;*out=result;return 0;}
    BnTransformerTrellisKVViews kv;
    if(bn_transformer_trellis_kv_views(m,s,layer,&kv)||kv.cursor!=i->pos+i->tokens||
       r->kv->slots[kv.owner].generation<=i->seen[kv.owner])return -1;
    uint64_t version=r->kv->slots[kv.owner].generation;result.pool_generation=version;result.entries=kv.entries;result.rate=kv.rate;
    if(kv.entries<=512){result.mode=BN_TRELLIS_INDEX_DENSE;*out=result;return 0;}
    result.k_len=512;result.indices=trellis_index_view(i,TI_IDS,(size_t)i->tokens*512);
    if(l->index_source!=layer) {
        if(!i->selection_valid||i->selection_source!=l->index_source||i->selection_owner!=kv.owner||
           i->selection_entries!=kv.entries||i->selection_version!=version)return -1;
        result.mode=i->mode;result.borrowed=1;result.selection_revision=i->selection_revision;*out=result;return 0;
    }
    BnTransformerTrellisIndexMode mode=BN_TRELLIS_INDEX_TOPK;
    if(kv.entries>16384&&l->candidate_mode==BN_NATIVE_CANDIDATE_SOURCE)mode=BN_TRELLIS_INDEX_CANDIDATE_SOURCE;
    if(kv.entries>16384&&l->candidate_mode==BN_NATIVE_CANDIDATE_USE) {
        if(!i->candidate_valid||i->candidate_entries!=kv.entries||i->candidate_rate!=kv.rate||
           r->kv->slots[i->candidate_owner].generation!=i->candidate_version)return -1;
        mode=BN_TRELLIS_INDEX_CANDIDATE_USE;
    }
    const BnBackendNativeF16Weight*qw;const BnBackendNativeParameter*ww;
    if(trellis_index_weights(m,layer,&qw,&ww)||trellis_index_input(r,x,qres,freq))return -1;
    BnGPUResidentView q=trellis_index_view(i,TI_Q,(size_t)i->tokens*4096),w=trellis_index_view(i,TI_W,(size_t)i->tokens*32);
    BnGPUF16IndexQueryResidentPlan qp={{qw->weight,a->q_rank,4096,i->tokens,qres,q},{ww->handle,a->experts.dim,32,i->tokens,x,w},
        {i->tokens,32,a->rope_dim,128,128-a->rope_dim,a->rope_dim,0,i->pos,bn_model_native_numerics(m)->rms_epsilon,NULL,q,freq,{0}}};
    if(i->selection_revision==UINT64_MAX)return -1;
    result.selection_revision=++i->selection_revision;
    i->selection_valid=0;if(mode==BN_TRELLIS_INDEX_CANDIDATE_SOURCE)i->candidate_valid=0;
    if(r->gpu->index_query_f16_resident(r->gpu->ctx,r->request,&qp))goto failed;
    for(int start=0;start<i->tokens;) {
        int rows=bn_gpu_index_slab_rows(i->tokens-start),T=kv.entries,stride=((T+127)/128)*128,B=T/8+1;
        BnGPUResidentView qs=bn_gpu_resident_view_slice(q,(size_t)start*4096,(size_t)rows*4096),ws=bn_gpu_resident_view_slice(w,(size_t)start*32,(size_t)rows*32),ids=bn_gpu_resident_view_slice(result.indices,(size_t)start*512,(size_t)rows*512);
        if(mode==BN_TRELLIS_INDEX_CANDIDATE_USE) {
            BnGPUIndexCandidateUseWorkspace uw;int cap=rows*16384;if(cap>T)cap=T;stride=((cap+127)/128)*128;
            if(bn_gpu_index_candidate_use_workspace(rows,T,stride,i->pos+start,kv.rate,&uw))goto failed;
            BnGPUIndexCandidateUseResidentPlan p={rows,T,stride,i->pos+start,kv.rate,kv.pages,kv.entries_per_page,
                qs,ws,kv.pool_idx,kv.block_table,bn_gpu_resident_view_slice(i->buffers[TI_CAND],(size_t)start*B,(size_t)rows*B),
                trellis_index_view(i,TI_ENTRY_IDS,uw.capacity),trellis_index_view(i,TI_ENTRY_COUNT,1),trellis_index_view(i,TI_GATHER,uw.gathered_count),trellis_index_view(i,TI_SCORE,uw.scores_count),ids,
                trellis_index_view(i,TI_BLOCK_COUNTS,uw.blocks),trellis_index_view(i,TI_BLOCK_OFFSETS,uw.blocks),trellis_index_view(i,TI_TILE_OFFSETS,uw.tiles),
                trellis_index_view(i,TI_SELECT_IDS,uw.select.indices_count),trellis_index_view(i,TI_SELECT_SCORES,uw.select.scores_count),trellis_index_view(i,TI_SELECT_COUNTS,uw.select.counts_count)};
            if(r->gpu->index_candidate_use_f16_resident(r->gpu->ctx,r->request,&p))goto failed;
        } else {
            BnGPUResidentView score=trellis_index_view(i,TI_SCORE,(size_t)rows*stride);
            BnGPUF16IndexScoreResidentPlan sp={rows,T,i->pos+start,kv.rate,stride,kv.pages,kv.entries_per_page,qs,ws,kv.pool_idx,kv.block_table,score};
            if(r->gpu->index_score_f16_resident(r->gpu->ctx,r->request,&sp))goto failed;
            if(mode==BN_TRELLIS_INDEX_CANDIDATE_SOURCE) {
                int bs=((B+127)/128)*128;BnGPUIndexCandidateWorkspace cw;
                if(bn_gpu_index_candidate_workspace(rows,T,stride,bs,i->pos+start,kv.rate,&cw))goto failed;
                BnGPUIndexCandidateSourceResidentPlan cp={rows,T,stride,bs,i->pos+start,kv.rate,score,
                    trellis_index_view(i,TI_BLOCK_SCORE,cw.block_scores_count),trellis_index_view(i,TI_BLOCK_IDS,cw.block_indices_count),bn_gpu_resident_view_slice(i->buffers[TI_CAND],(size_t)start*B,(size_t)rows*B),
                    trellis_index_view(i,TI_SELECT_IDS,cw.select.indices_count),trellis_index_view(i,TI_SELECT_SCORES,cw.select.scores_count),trellis_index_view(i,TI_SELECT_COUNTS,cw.select.counts_count)};
                if(r->gpu->index_candidate_source_f16_resident(r->gpu->ctx,r->request,&cp))goto failed;
            }
            BnGPUF16IndexTopKWorkspace tw;if(bn_gpu_index_topk_workspace(rows,T,stride,512,512,&tw))goto failed;
            BnGPUF16IndexTopKResidentPlan tp={rows,T,stride,512,512,score,ids,trellis_index_view(i,TI_SELECT_IDS,tw.indices_count),trellis_index_view(i,TI_SELECT_SCORES,tw.scores_count),trellis_index_view(i,TI_SELECT_COUNTS,tw.counts_count)};
            if(r->gpu->index_topk_f16_resident(r->gpu->ctx,r->request,&tp))goto failed;
        }
        start+=rows;
    }
    if(r->gpu->resident_synchronize(r->request))goto failed;
    i->selection_valid=1;i->selection_source=layer;i->selection_owner=kv.owner;i->selection_entries=kv.entries;i->selection_version=version;i->mode=mode;
    if(mode==BN_TRELLIS_INDEX_CANDIDATE_SOURCE){i->candidate_valid=1;i->candidate_owner=kv.owner;i->candidate_version=version;i->candidate_entries=kv.entries;i->candidate_rate=kv.rate;}
    result.mode=mode;*out=result;return 0;
failed:r->destroy(r->request);r->request=NULL;return -1;
}
#endif
