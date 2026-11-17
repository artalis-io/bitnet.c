#ifndef BN_TRANSFORMER_TRELLIS_HC_H
#define BN_TRANSFORMER_TRELLIS_HC_H
enum {TH_STREAM0,TH_STREAM1,TH_PARTIALS,TH_POST,TH_COMB,TH_PRE0,TH_PRE1,TH_COLLAPSED,TH_HEAD,TH_IDS,TH_COUNT};
typedef struct {void *fn,*base,*scale,*norm;} TrellisHCSite;
struct TrellisHC {
    BnTransformerTrellisHCWorkspace workspace;
    BnGPUResidentView buffers[TH_COUNT];
    TrellisHCSite *sites;
    void *embedding,*final_norm;
    uint64_t epoch,history_epoch;
    int phase,site,tokens,position,stream;
};
static void *trellis_hc_parameter(const BnModel*m,int layer,int role,int rows,int cols,int f32) {
    const BnBackendNativeParameter*p=bn_backend_layout_native_parameter(bn_model_backend(m),layer,role);
    return p&&p->handle&&p->layout==BN_NATIVE_PARAMETER_RAW&&p->rows==rows&&p->cols==cols&&
        p->type==bn_backend_quant_scalar_type(f32?BN_BACKEND_SCALAR_F32:BN_BACKEND_SCALAR_BF16)?p->handle:NULL;
}
static int trellis_hc_site(const BnModel*m,int site,TrellisHCSite*out) {
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(m);int f=site%2,l=site/2,D=a->experts.dim;
    TrellisHCSite v={
        trellis_hc_parameter(m,l,f?BN_NATIVE_HC_FFN_FN:BN_NATIVE_HC_ATTN_FN,24,4*D,1),
        trellis_hc_parameter(m,l,f?BN_NATIVE_HC_FFN_BASE:BN_NATIVE_HC_ATTN_BASE,1,24,1),
        trellis_hc_parameter(m,l,f?BN_NATIVE_HC_FFN_SCALE:BN_NATIVE_HC_ATTN_SCALE,1,3,1),
        trellis_hc_parameter(m,l,f?BN_NATIVE_FFN_NORM:BN_NATIVE_ATTN_NORM,1,D,0)};
    if(!v.fn||!v.base||!v.scale||!v.norm)return -1;
    if(out)*out=v;
    return 0;
}
static int trellis_hc_shape(const BnModel*m,const BnTransformerTrellisOptions*o,
    BnTransformerTrellisHCWorkspace*out,size_t*counts,int*types) {
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(m);const BnModelNativeNumerics*n=bn_model_native_numerics(m);
    BnGPUBackend*g=bn_model_gpu(m);BnTransformerTrellisWorkspace base;
    if(!a||!n||!g||!out||sizeof(int)!=4||!g->embedding_bf16_reserve||!g->embedding_hc4_f32_resident||
       !g->hc4_carry_resident||!g->hc4_apply_resident||!g->hc4_head_workspace_size||!g->hc4_head_reserve||!g->hc4_head_resident||
       !g->rmsnorm_vector4_f16_f32_resident||a->streams!=4||n->hc_streams!=4||!n->hc_carry_pre||
       a->experts.layers<1||a->experts.layers>BN_MODEL_ANATOMY_MAX_LAYERS||
       a->experts.dim<128||a->experts.dim>32768||a->experts.dim%4||
       n->hc_iterations<1||n->hc_iterations>256||!isfinite(n->rms_epsilon)||n->rms_epsilon<=0||
       !isfinite(n->hc_epsilon)||n->hc_epsilon<=0||bn_transformer_trellis_session_plan(m,o,&base))return -1;
    int D=a->experts.dim;
    if(!trellis_hc_parameter(m,-1,BN_NATIVE_GLOBAL_EMBEDDING,a->vocab,D,0)||
       !trellis_hc_parameter(m,-1,BN_NATIVE_GLOBAL_OUTPUT_NORM,1,D,0))return -1;
    for(int j=0;j<2*a->experts.layers;j++)if(trellis_hc_site(m,j,NULL))return -1;
    size_t R=(size_t)o->max_tokens,partial=0;
    if(R>INT_MAX/(size_t)D/4)return -1;
    /* Partials are not monotonic in R: e.g. R32 needs more than R33. */
    for(int r=1;r<=o->max_tokens;r++) {
        size_t p=bn_gpu_hc4_partials_count(r,D);if(!p)return -1;if(p>partial)partial=p;
    }
    size_t v[TH_COUNT]={R*4*D,R*4*D,partial,R*4,R*16,R*4,R*4,R*D,R*D,R};
    size_t device=sizeof(int),host=sizeof(TrellisHC); /* embedding error flag */
    for(int j=0;j<TH_COUNT;j++)if(trellis_add_bytes(&device,v[j],4))return -1;
    if(trellis_add_bytes(&host,(size_t)2*a->experts.layers,sizeof(TrellisHCSite)))return -1;
    /* Existing linear_f16_reserve already budgets/reserves shared head BLAS. */
    size_t head=g->hc4_head_workspace_size(g->ctx),linear=g->linear_f16_workspace_size(g->ctx);
    if(!head||head>linear)return -1;
    if(counts)memcpy(counts,v,sizeof(v));
    if(types)for(int j=0;j<TH_COUNT;j++)types[j]=bn_backend_quant_scalar_type(j==TH_IDS?BN_BACKEND_SCALAR_I32:BN_BACKEND_SCALAR_F32);
    *out=(BnTransformerTrellisHCWorkspace){device,host,o->max_tokens,a->experts.layers};return 0;
}
int bn_transformer_trellis_hc_session_plan(const BnModel*m,const BnTransformerTrellisOptions*o,
    BnTransformerTrellisHCWorkspace*out) {return trellis_hc_shape(m,o,out,NULL,NULL);}
static int trellis_hc_setup(TrellisRequest*r,const BnTransformerTrellisHCWorkspace*w) {
    size_t n[TH_COUNT];int t[TH_COUNT];BnTransformerTrellisHCWorkspace check;
    if(!r->engram||trellis_hc_shape(r->model,&r->options,&check,n,t))return -1;
    TrellisHC*h=calloc(1,w->host_bytes);if(!h)return -1;r->hc=h;h->workspace=*w;h->sites=(TrellisHCSite*)(h+1);
    for(int j=0;j<2*w->layers;j++)if(trellis_hc_site(r->model,j,h->sites+j))return -1;
    h->embedding=trellis_hc_parameter(r->model,-1,BN_NATIVE_GLOBAL_EMBEDDING,r->anatomy->vocab,r->anatomy->experts.dim,0);
    h->final_norm=trellis_hc_parameter(r->model,-1,BN_NATIVE_GLOBAL_OUTPUT_NORM,1,r->anatomy->experts.dim,0);
    if(r->gpu->embedding_bf16_reserve(r->gpu->ctx,r->request)||r->gpu->hc4_head_reserve(r->gpu->ctx,r->request))return -1;
    for(int j=0;j<TH_COUNT;j++)if(r->gpu->resident_alloc(r->request,t[j],n[j],h->buffers+j))return -1;
    return 0;
}
static BnGPUResidentView trellis_hc_view(TrellisRequest*r,int slot) {
    TrellisHC*h=r->hc;size_t R=(size_t)h->tokens,D=(size_t)r->anatomy->experts.dim,n=R*D;
    if(slot==TH_STREAM0||slot==TH_STREAM1)n*=4;
    else if(slot==TH_PARTIALS)n=bn_gpu_hc4_partials_count(h->tokens,(int)D);
    else if(slot==TH_POST||slot==TH_PRE0||slot==TH_PRE1)n=R*4;
    else if(slot==TH_COMB)n=R*16;
    else if(slot==TH_IDS)n=R;
    return bn_gpu_resident_view_slice(h->buffers[slot],0,n);
}
static int trellis_hc_history(const BnModel*m,BnSession*s,TrellisHC*h) {
    BnSessionNativeHistoryView v;
    return !bn_session_native_history_view(s,m,&v)&&v.epoch==h->history_epoch&&v.tokens==h->tokens&&v.position==h->position;
}
static int trellis_hc_fail(BnSession*s) {bn_backend_session_release_gpu_resources(bn_session_backend(s));return -1;}
int bn_transformer_trellis_hc_begin(const BnModel*m,BnSession*s) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisHC*h=r?r->hc:NULL;BnSessionNativeHistoryView v;
    if(!h||h->epoch>UINT64_MAX-(uint64_t)(2*h->workspace.layers+1)||(h->phase!=0&&h->phase!=3)||bn_session_native_history_view(s,m,&v)||
       !v.raw||v.tokens<1||v.tokens>h->workspace.max_tokens||
       (h->phase==3&&(v.epoch==h->history_epoch||v.position!=h->position+h->tokens)))return -1;
    BnGPUResidentView ids=bn_gpu_resident_view_slice(h->buffers[TH_IDS],0,(size_t)v.tokens);
    int status=bn_backend_session_row_stage_upload_i32(&r->rows,(const int32_t*)v.raw,(size_t)v.tokens,&ids);
    if(status==BN_GPU_UPLOAD_BUSY)return status;
    if(status)return trellis_hc_fail(s);
    h->tokens=v.tokens;h->position=v.position;h->history_epoch=v.epoch;h->site=0;h->stream=TH_STREAM0;h->epoch++;
    BnGPUBF16EmbeddingHC4ResidentPlan p={h->tokens,r->anatomy->experts.dim,r->anatomy->vocab,
        h->embedding,trellis_hc_view(r,TH_IDS),trellis_hc_view(r,h->stream)};
    if(r->gpu->embedding_hc4_f32_resident(r->gpu->ctx,r->request,&p))return trellis_hc_fail(s);
    r->block_layer=0;r->block_index_generation=0;h->phase=1;return 0;
}
static BnTransformerTrellisHCResult trellis_hc_result(TrellisRequest*r,BnSession*s) {
    TrellisHC*h=r->hc;
    BnTransformerTrellisHCResult v={r->request,s,r->generation,h->epoch,h->history_epoch,h->site,h->tokens,
        trellis_hc_view(r,h->stream),trellis_hc_view(r,TH_PARTIALS),trellis_hc_view(r,TH_POST),trellis_hc_view(r,TH_COMB),
        trellis_hc_view(r,TH_COLLAPSED),trellis_hc_view(r,h->site%2?TH_PRE1:TH_PRE0),
        bn_gpu_resident_view_slice(r->buffers[TR_X],0,(size_t)h->tokens*r->anatomy->experts.dim)};
    return v;
}
int bn_transformer_trellis_hc_mix(const BnModel*m,BnSession*s,BnTransformerTrellisHCResult*out) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisHC*h=r?r->hc:NULL;
    if(!h||!out||h->phase!=1||h->site>=2*h->workspace.layers||h->epoch==UINT64_MAX||!trellis_hc_history(m,s,h))return -1;
    int layer=h->site/2;
    if(!(h->site%2)&&trellis_engram_layer(r->engram,layer)) {
        BnTransformerTrellisEngramResult e;
        int status=bn_transformer_trellis_engram_submit(m,s,layer,(unsigned)layer%r->rows_options.banks,trellis_hc_view(r,h->stream),&e);
        if(status==BN_GPU_UPLOAD_BUSY)return status;
        if(status)return trellis_hc_fail(s);
        /* Keep persistent streams outside Engram's reusable output scratch.
         * This duplicates its final add, preserving arithmetic without a
         * copy/readback or an unsafe alias at the next Engram layer. */
        int next=h->stream==TH_STREAM0?TH_STREAM1:TH_STREAM0;
        BnGPUAddF32ResidentPlan add={e.delta.count,trellis_hc_view(r,h->stream),e.delta,trellis_hc_view(r,next)};
        if(r->gpu->add_f32_resident(r->gpu->ctx,r->request,&add))return trellis_hc_fail(s);
        h->stream=next;
    }
    const BnModelNativeNumerics*n=bn_model_native_numerics(m);TrellisHCSite*w=h->sites+h->site;
    BnTransformerTrellisHCResult v=trellis_hc_result(r,s);
    BnGPUResidentView prev=h->site?trellis_hc_view(r,h->site%2?TH_PRE0:TH_PRE1):(BnGPUResidentView){0};
    BnGPUHC4CarryResidentPlan p={h->tokens,r->anatomy->experts.dim,n->hc_iterations,n->rms_epsilon,n->hc_epsilon,
        w->fn,w->base,w->scale,v.streams,v.partials,v.post,v.comb,v.collapsed,prev,v.pre};
    BnGPUVector4RMSNormResidentPlan norm={h->tokens,p.dim,1,n->rms_epsilon,0.f,1.f,w->norm,v.collapsed,v.normalized};
    if(r->gpu->hc4_carry_resident(r->gpu->ctx,r->request,&p)||
       r->gpu->rmsnorm_vector4_f16_f32_resident(r->gpu->ctx,r->request,&norm))return trellis_hc_fail(s);
    h->epoch++;h->phase=2;v.epoch=h->epoch;*out=v;return 0;
}
int bn_transformer_trellis_hc_current(const BnModel*m,BnSession*s,const BnTransformerTrellisHCResult*v) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisHC*h=r?r->hc:NULL;
    if(!h||!v||h->phase!=2||!v->epoch||v->session!=s||v->request!=r->request||v->generation!=r->generation||
       v->epoch!=h->epoch||v->history_epoch!=h->history_epoch||v->site!=h->site||v->tokens!=h->tokens||!trellis_hc_history(m,s,h))return 0;
    BnTransformerTrellisHCResult want=trellis_hc_result(r,s);
    BnGPUResidentView a[]={v->streams,v->partials,v->post,v->comb,v->collapsed,v->pre,v->normalized};
    BnGPUResidentView b[]={want.streams,want.partials,want.post,want.comb,want.collapsed,want.pre,want.normalized};
    for(int j=0;j<7;j++)if(!trellis_engram_same(a[j],b[j]))return 0;
    return 1;
}
int bn_transformer_trellis_hc_apply(const BnModel*m,BnSession*s,const BnTransformerTrellisHCResult*v,BnGPUResidentView branch) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisHC*h=r?r->hc:NULL;
    if(!bn_transformer_trellis_hc_current(m,s,v)||!h||branch.request!=r->request||
       branch.count!=(size_t)h->tokens*r->anatomy->experts.dim||branch.offset>SIZE_MAX-branch.count||
       (branch.type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16)&&branch.type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F32)))return -1;
    for(int j=0;j<TH_COUNT;j++)if(trellis_engram_overlap(branch,h->buffers[j]))return -1;
    BnGPUHC4ApplyResidentPlan p={h->tokens,r->anatomy->experts.dim,v->streams,branch,v->post,v->comb};
    if(r->gpu->hc4_apply_resident(r->gpu->ctx,r->request,&p))return trellis_hc_fail(s);
    h->site++;h->phase=1;return 0;
}
int bn_transformer_trellis_hc_finish(const BnModel*m,BnSession*s,BnGPUResidentView*collapsed,BnGPUResidentView*head_input) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisHC*h=r?r->hc:NULL;
    if(!h||!collapsed||!head_input||collapsed==head_input||h->phase!=1||h->site!=2*h->workspace.layers||!trellis_hc_history(m,s,h))return -1;
    BnGPUResidentView out=trellis_hc_view(r,TH_HEAD),normed=bn_gpu_resident_view_slice(r->buffers[TR_X],0,out.count);
    BnGPUHC4HeadResidentPlan p={h->tokens,r->anatomy->experts.dim,trellis_hc_view(r,h->stream),trellis_hc_view(r,TH_PRE1),out};
    BnGPUVector4RMSNormResidentPlan norm={h->tokens,p.dim,1,bn_model_native_numerics(m)->rms_epsilon,0.f,1.f,h->final_norm,out,normed};
    if(r->gpu->hc4_head_resident(r->gpu->ctx,r->request,&p)||
       r->gpu->rmsnorm_vector4_f16_f32_resident(r->gpu->ctx,r->request,&norm))return trellis_hc_fail(s);
    h->phase=3;*collapsed=out;*head_input=normed;return 0;
}
#endif
