#ifndef BN_TRANSFORMER_TRELLIS_ENGRAM_H
#define BN_TRANSFORMER_TRELLIS_ENGRAM_H
enum {TE_STREAM,TE_EMBED,TE_KEY,TE_VALUE,TE_KN,TE_QN,TE_DOT,TE_DELTA,TE_OUT,TE_COUNT};
typedef struct {
    BnModelNativeEngramMath spec;
    BnModelNativeEngramLayout layout;
    void *weight,*norm_key,*norm_query;
} TrellisEngramLayer;
struct TrellisEngram {
    BnTransformerTrellisEngramWorkspace workspace;
    BnGPUResidentView buffers[TE_COUNT];
    TrellisEngramLayer *layers;
    uint64_t *ids;
    uint64_t epoch;
    int last_layer,last_tokens;
    unsigned last_bank;
};
static int trellis_engram_weights(const BnModel*m,int layer,BnModelNativeEngramMath*spec,
    void **weight,void **key,void **query) {
    if(bn_model_native_engram_math(m,layer,spec)||spec->profile!=BN_NATIVE_ENGRAM_MATH_BF16_LOOKUP_F16)return -1;
    const BnBackendNativeF16Weight*w=bn_backend_layout_native_core_weight(bn_model_backend(m),layer,BN_NATIVE_ENGRAM_WKV);
    const BnBackendNativeParameter*k=bn_backend_layout_native_parameter(bn_model_backend(m),layer,BN_NATIVE_ENGRAM_K);
    const BnBackendNativeParameter*q=bn_backend_layout_native_parameter(bn_model_backend(m),layer,BN_NATIVE_ENGRAM_Q);
    if(!w||!w->weight||w->gpu!=bn_model_gpu(m)||w->cols!=spec->embedding_dim||w->rows!=(spec->streams+1)*spec->dim||
       !k||!q||!k->handle||!q->handle||k->layout!=BN_NATIVE_PARAMETER_RAW||q->layout!=BN_NATIVE_PARAMETER_RAW||
       k->type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_BF16)||q->type!=k->type||
       k->rows!=spec->streams||q->rows!=k->rows||k->cols!=spec->dim||q->cols!=k->cols)return -1;
    if(weight)*weight=w->weight;
    if(key)*key=k->handle;
    if(query)*query=q->handle;
    return 0;
}
static int trellis_engram_shape(const BnModel*m,const BnTransformerTrellisOptions*ffn,
    const BnTransformerTrellisRowsOptions*rows,BnTransformerTrellisEngramWorkspace*out,size_t*counts,int*types) {
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(m);BnGPUBackend*g=bn_model_gpu(m);
    BnTransformerTrellisWorkspace base;BnTransformerTrellisRowsWorkspace rw;BnModelNativeTokenMapInfo map;
    if(!a||!g||!out||!g->ple_f16_from_bf16_resident||!bn_backend_quant_rows_support_f16(a->engram_format)||
       bn_model_native_token_map_info(m,&map)||bn_transformer_trellis_session_plan(m,ffn,&base)||
       bn_transformer_trellis_rows_session_plan(m,rows,&rw))return -1;
    int tables=0;BnModelNativeEngramMath common={0};
    for(int l=0;l<a->experts.layers;l++)if(a->layers[l].engram_rows) {
        BnModelNativeEngramMath spec;
        if(trellis_engram_weights(m,l,&spec,NULL,NULL,NULL))return -1;
        if(!tables)common=spec;
        else if(spec.profile!=common.profile||spec.dim!=common.dim||spec.streams!=common.streams||
                spec.heads!=common.heads||spec.head_dim!=common.head_dim||spec.embedding_dim!=common.embedding_dim)return -1;
        tables++;
    }
    if(!tables||tables!=rw.tables||ffn->max_tokens>rows->max_rows/common.heads)return -1;
    size_t R=(size_t)ffn->max_tokens,D=(size_t)common.dim,H=(size_t)common.streams,K=(size_t)common.embedding_dim;
    if(R>SIZE_MAX/H/D/4||R*H*D>INT_MAX)return -1;
    size_t n[TE_COUNT]={R*H*D,R*K,R*H*D,R*D,R*H*D,R*H*D,R*H,R*H*D,R*H*D};
    BnBackendScalarKind k[TE_COUNT]={BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,
        BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32,
        BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32};
    size_t device=0,host=sizeof(TrellisEngram),history;
    for(int j=0;j<TE_COUNT;j++)if(trellis_add_bytes(&device,n[j],bn_backend_quant_scalar_bytes(k[j])))return -1;
    if(trellis_add_bytes(&host,(size_t)tables,sizeof(TrellisEngramLayer))||
       trellis_add_bytes(&host,R*(size_t)common.heads,sizeof(uint64_t))||
       bn_session_native_history_plan(m,ffn->max_tokens,&history))return -1;
    if(counts)memcpy(counts,n,sizeof(n));
    if(types)for(int j=0;j<TE_COUNT;j++)types[j]=bn_backend_quant_scalar_type(k[j]);
    *out=(BnTransformerTrellisEngramWorkspace){device,host,history,ffn->max_tokens,tables,common.heads};return 0;
}
int bn_transformer_trellis_engram_session_plan(const BnModel*m,const BnTransformerTrellisOptions*ffn,
    const BnTransformerTrellisRowsOptions*rows,BnTransformerTrellisEngramWorkspace*out) {
    return trellis_engram_shape(m,ffn,rows,out,NULL,NULL);
}
static int trellis_engram_setup(TrellisRequest*r,BnSession*s,const BnTransformerTrellisEngramWorkspace*w) {
    BnSessionNativeHistoryInfo history;BnTransformerTrellisEngramWorkspace check;
    size_t n[TE_COUNT];int t[TE_COUNT];
    if(s->pos||bn_session_native_history_info(s,r->model,&history)||history.committed||history.staged||
       history.max_tokens!=w->max_tokens||history.host_bytes!=w->history_bytes||
       trellis_engram_shape(r->model,&r->options,&r->rows_options,&check,n,t))return -1;
    TrellisEngram*e=calloc(1,w->host_bytes);if(!e)return -1;r->engram=e;e->workspace=*w;
    e->layers=(TrellisEngramLayer*)(e+1);e->ids=(uint64_t*)(e->layers+w->tables);e->last_layer=-1;
    int table=0;
    for(int l=0;l<r->anatomy->experts.layers;l++)if(r->anatomy->layers[l].engram_rows) {
        TrellisEngramLayer*entry=e->layers+table++;
        const BnTrellisRowLayout*layout=bn_backend_model_row_cache_layout(bn_model_native_row_cache(r->model,l));
        if(trellis_engram_weights(r->model,l,&entry->spec,&entry->weight,&entry->norm_key,&entry->norm_query)||
           bn_model_native_engram_layout(r->model,l,&entry->layout)||!layout||
           layout->rows!=entry->layout.rows||entry->layout.heads!=entry->spec.heads)return -1;
    }
    for(int j=0;j<TE_COUNT;j++)if(r->gpu->resident_alloc(r->request,t[j],n[j],&e->buffers[j]))return -1;
    return 0;
}
static TrellisEngramLayer*trellis_engram_layer(TrellisEngram*e,int layer) {
    if(e)for(int i=0;i<e->workspace.tables;i++)if(e->layers[i].spec.layer==layer)return e->layers+i;
    return NULL;
}
static BnGPUResidentView trellis_engram_view(TrellisEngram*e,int slot,int tokens) {
    BnModelNativeEngramMath*p=&e->layers[0].spec;
    size_t count=(size_t)tokens*p->streams*p->dim;
    if(slot==TE_EMBED)count=(size_t)tokens*p->embedding_dim;
    else if(slot==TE_VALUE)count=(size_t)tokens*p->dim;
    else if(slot==TE_DOT)count=(size_t)tokens*p->streams;
    return bn_gpu_resident_view_slice(e->buffers[slot],0,count);
}
static int trellis_engram_same(BnGPUResidentView a,BnGPUResidentView b) {
    return a.request==b.request&&a.slot==b.slot&&a.offset==b.offset&&a.count==b.count&&a.type==b.type;
}
static int trellis_engram_overlap(BnGPUResidentView a,BnGPUResidentView b) {
    if(a.request!=b.request||a.slot!=b.slot)return 0;
    if(a.type!=b.type)return 1;
    return a.offset<=b.offset?b.offset-a.offset<a.count:a.offset-b.offset<b.count;
}
int bn_transformer_trellis_engram_input(const BnModel*m,BnSession*s,int tokens,void**request,BnGPUResidentView*stream) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisEngram*e=r?r->engram:NULL;
    if(!e||!request||!stream||tokens<1||tokens>e->workspace.max_tokens)return -1;
    *request=r->request;*stream=trellis_engram_view(e,TE_STREAM,tokens);return 0;
}
int bn_transformer_trellis_engram_submit(const BnModel*m,BnSession*s,int layer,unsigned bank,
    BnGPUResidentView stream,BnTransformerTrellisEngramResult*out) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisEngram*e=r?r->engram:NULL;
    TrellisEngramLayer*entry=trellis_engram_layer(e,layer);BnSessionNativeHistoryView history;BnModelNativeTokenMapInfo map;
    if(!entry||!out||bank>=(unsigned)r->rows_options.banks||e->epoch==UINT64_MAX||
       bn_session_native_history_view(s,m,&history)||bn_model_native_token_map_info(m,&map)||
       history.tokens<1||history.tokens>e->workspace.max_tokens||history.prefix!=entry->layout.ngram-1||
       history.length!=(size_t)(history.prefix+history.tokens)||stream.request!=r->request||
       stream.type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32)||
       stream.count!=(size_t)history.tokens*entry->spec.streams*entry->spec.dim||stream.offset>SIZE_MAX-stream.count)return -1;
    for(int j=TE_EMBED;j<TE_COUNT;j++)if(trellis_engram_overlap(stream,trellis_engram_view(e,j,history.tokens)))return -1;
    size_t row_count=(size_t)history.tokens*entry->spec.heads;
    if(bn_model_native_engram_hash_rows(&entry->layout,map.mapped_pad,history.mapped,1,history.length,
       (size_t)history.tokens,e->ids,row_count))return -1;
    BnGPUResidentView lookup;uint64_t row_epoch;
    int status=bn_backend_session_row_stage_submit_f16(&r->rows,bn_model_native_row_cache(m,layer),bank,
        e->ids,row_count,&lookup,&row_epoch);
    if(status) {
        BnBackendRowStageInfo info;
        if(bn_backend_session_row_stage_info(&r->rows,&info)||info.failed)
            bn_backend_session_release_gpu_resources(bn_session_backend(s));
        return status;
    }
    r->row_layers[bank]=layer;e->epoch++;e->last_layer=layer;e->last_tokens=history.tokens;e->last_bank=bank;
    BnGPUResidentView embedding=trellis_engram_view(e,TE_EMBED,history.tokens),key=trellis_engram_view(e,TE_KEY,history.tokens),
        value=trellis_engram_view(e,TE_VALUE,history.tokens),kn=trellis_engram_view(e,TE_KN,history.tokens),
        qn=trellis_engram_view(e,TE_QN,history.tokens),dot=trellis_engram_view(e,TE_DOT,history.tokens),
        delta=trellis_engram_view(e,TE_DELTA,history.tokens),result=trellis_engram_view(e,TE_OUT,history.tokens);
    BnGPUPLEF16FromBF16ResidentPlan p={history.tokens,entry->spec.embedding_dim,entry->spec.dim,entry->spec.streams,
        entry->spec.eps,entry->spec.gate_scale,entry->weight,entry->norm_key,entry->norm_query,
        lookup,embedding,stream,key,value,kn,qn,dot,delta,result};
    if(r->gpu->ple_f16_from_bf16_resident(r->gpu->ctx,r->request,&p)) {
        bn_backend_session_release_gpu_resources(bn_session_backend(s));return -1;
    }
    *out=(BnTransformerTrellisEngramResult){r->request,s,layer,history.tokens,bank,r->generation,e->epoch,history.epoch,row_epoch,
        lookup,embedding,key,value,kn,qn,dot,delta,result};return 0;
}
int bn_transformer_trellis_engram_current(const BnModel*m,BnSession*s,const BnTransformerTrellisEngramResult*v) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisEngram*e=r?r->engram:NULL;BnSessionNativeHistoryView history;
    if(!e||!v||!v->epoch||v->request!=r->request||v->session!=s||v->generation!=r->generation||
       v->epoch!=e->epoch||v->layer!=e->last_layer||v->tokens!=e->last_tokens||v->bank!=e->last_bank||
       bn_session_native_history_view(s,m,&history)||v->history_epoch!=history.epoch||
       !bn_backend_session_row_stage_current(&r->rows,v->bank,v->row_epoch,&v->lookup))return 0;
    BnGPUResidentView views[]={v->embedding,v->key,v->value,v->normalized_key,v->normalized_query,v->dot,v->delta,v->out};
    for(int j=0;j<8;j++)if(!trellis_engram_same(views[j],trellis_engram_view(e,j+TE_EMBED,v->tokens)))return 0;
    return 1;
}
#endif
