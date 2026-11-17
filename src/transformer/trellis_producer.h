#ifndef BN_TRANSFORMER_TRELLIS_PRODUCER_H
#define BN_TRANSFORMER_TRELLIS_PRODUCER_H
/* Source producers use the same request and ownership epochs as consumers. */
enum {TP_QRAW,TP_QRES,TP_Q,TP_KV,TP_COMP_KV,TP_COMP_GATE,TP_LATENT,
    TP_KEY_RAW,TP_KEYS,TP_POS,TP_ZERO_FREQ,TP_APE,TP_MAIN_FREQ,TP_MAIN_NEG,TP_COMP_FREQ,TP_COMP_NEG,TP_COUNT};
struct TrellisProducer {
    BnTransformerTrellisProducerWorkspace workspace;
    BnGPUResidentView buffers[TP_COUNT],norms[BN_MODEL_ANATOMY_MAX_LAYERS];
};
static const BnBackendNativeF16Weight*trellis_producer_core(const BnModel*m,int l,int role,int rows,int cols) {
    const BnBackendNativeF16Weight*w=bn_backend_layout_native_core_weight(bn_model_backend(m),l,role);
    return w&&w->weight&&w->gpu==bn_model_gpu(m)&&w->rows==rows&&w->cols==cols?w:NULL;
}
static const BnBackendNativeParameter*trellis_producer_parameter(const BnModel*m,int l,int role,int rows,int cols,int linear) {
    const BnBackendNativeParameter*w=bn_backend_layout_native_parameter(bn_model_backend(m),l,role);
    if(!w||!w->handle||w->rows!=rows||w->cols!=cols||
       w->layout!=(linear?BN_NATIVE_PARAMETER_F16_LINEAR:BN_NATIVE_PARAMETER_RAW))return NULL;
    if(linear)return w->type==bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16)?w:NULL;
    return w->type==bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16)||
        w->type==bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_BF16)?w:NULL;
}
static int trellis_producer_weights(const BnModel*m,int l) {
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(m);int D=a->experts.dim;
    if(!trellis_producer_core(m,l,BN_NATIVE_Q_A,a->q_rank,D)||
       !trellis_producer_core(m,l,BN_NATIVE_Q_B,a->heads*a->head_dim,a->q_rank)||
       !trellis_producer_core(m,l,BN_NATIVE_KV,a->head_dim,D)||
       !trellis_producer_parameter(m,l,BN_NATIVE_Q_NORM,1,a->q_rank,0)||
       !trellis_producer_parameter(m,l,BN_NATIVE_KV_NORM,1,a->head_dim,0))return -1;
    if(a->layers[l].kv_source==l&&(!trellis_producer_parameter(m,l,BN_NATIVE_COMP_KV,a->head_dim,D,1)||
       (a->layers[l].compress_ratio>1&&!trellis_producer_parameter(m,l,BN_NATIVE_COMP_GATE,a->head_dim,D,1))||
       !trellis_producer_parameter(m,l,BN_NATIVE_COMP_NORM,1,a->head_dim,0)||
       !trellis_producer_parameter(m,l,BN_NATIVE_INDEX_K,a->index_dim,a->head_dim,1)||
       !trellis_producer_parameter(m,l,BN_NATIVE_INDEX_K_NORM,1,a->index_dim,0)))return -1;
    return 0;
}
static int trellis_producer_shape(const BnModel*m,const BnTransformerTrellisOptions*ffn,
    const BnTransformerTrellisKVOptions*kv,BnTransformerTrellisProducerWorkspace*out,size_t*counts,int*types) {
    BnTransformerTrellisAttentionWorkspace aw;const BnModelNativeAnatomy*a=bn_model_native_anatomy(m);BnGPUBackend*g=bn_model_gpu(m);
    if(!a||!g||!out||!g->rope_frequency_f32_resident||!g->latent_qkv_f16_resident||!g->compressor_f16_resident||!g->latent_keys_f16_resident||
       bn_transformer_trellis_attention_session_plan(m,ffn,kv,&aw)||a->q_rank%4)return -1;
    BnModelNativeRopeFrequency freq;
    for(int family=0;family<2;family++)if(bn_model_native_rope_frequency(m,family,&freq))return -1;
    const BnSafetensorsIndex*index=bn_model_native_index(m);int maxrate=1,owners=0;
    for(int l=0;l<a->experts.layers;l++) {
        if(trellis_producer_weights(m,l))return -1;
        if(a->layers[l].kv_source==l) {
            int rate=a->layers[l].compress_ratio;if(rate>maxrate)maxrate=rate;owners++;
            int id=a->layers[l].tensor[BN_NATIVE_COMP_NORM];const BnSafetensorsTensor*t=id<0?NULL:bn_safetensors_index_tensor(index,(size_t)id);
            if(!t||t->dtype!=BN_SAFETENSORS_BF16||t->n_dims!=1||t->shape[0]!=(uint64_t)a->head_dim||t->bytes!=(uint64_t)a->head_dim*2)return -1;
        }
    }
    size_t R=(size_t)aw.max_tokens,D=(size_t)a->head_dim;
    size_t n[TP_COUNT]={R*a->q_rank,R*a->q_rank,R*a->heads*D,R*D,R*D,R*D,R*D,
        R*a->index_dim,R*a->index_dim,R,(size_t)a->rope_dim/2,(size_t)maxrate*D,(size_t)a->rope_dim/2,(size_t)a->rope_dim/2,(size_t)a->rope_dim/2,(size_t)a->rope_dim/2};
    BnBackendScalarKind k[TP_COUNT]={BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,
        BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,
        BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32,
        BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32};
    size_t device=0,host=sizeof(TrellisProducer);for(int j=0;j<TP_COUNT;j++)if(trellis_add_bytes(&device,n[j],bn_backend_quant_scalar_bytes(k[j])))return -1;
    if(trellis_add_bytes(&device,(size_t)owners*D,2)||trellis_add_bytes(&host,D,2))return -1;
    if(counts){memcpy(counts,n,sizeof(n));}if(types)for(int j=0;j<TP_COUNT;j++)types[j]=bn_backend_quant_scalar_type(k[j]);
    BnTransformerTrellisProducerWorkspace w={device,host,aw.max_tokens,maxrate,owners};*out=w;return 0;
}
int bn_transformer_trellis_producer_session_plan(const BnModel*m,const BnTransformerTrellisOptions*ffn,
    const BnTransformerTrellisKVOptions*kv,BnTransformerTrellisProducerWorkspace*out) {
    return trellis_producer_shape(m,ffn,kv,out,NULL,NULL);
}
static int trellis_producer_setup(TrellisRequest*r,const BnTransformerTrellisProducerWorkspace*w) {
    TrellisProducer*p=calloc(1,sizeof(*p));if(!p)return -1;r->producer=p;p->workspace=*w;
    BnTransformerTrellisKVOptions kv={r->kv->maxseq,0,0};BnTransformerTrellisProducerWorkspace check;size_t n[TP_COUNT];int t[TP_COUNT];
    if(trellis_producer_shape(r->model,&r->options,&kv,&check,n,t))return -1;
    for(int j=0;j<TP_COUNT;j++)if(r->gpu->resident_alloc(r->request,t[j],n[j],&p->buffers[j]))return -1;
    if(r->gpu->resident_zero(r->request,p->buffers[TP_ZERO_FREQ])||r->gpu->resident_zero(r->request,p->buffers[TP_APE]))return -1;
    const BnModelNativeAnatomy*a=r->anatomy;const BnSafetensorsIndex*index=bn_model_native_index(r->model);
    uint16_t*norm=malloc((size_t)a->head_dim*2);if(!norm)return -1;int status=-1;
    for(int l=0;l<a->experts.layers;l++)if(a->layers[l].kv_source==l) {
        if(bn_safetensors_index_read(index,(size_t)a->layers[l].tensor[BN_NATIVE_COMP_NORM],0,norm,(size_t)a->head_dim*2))goto done;
        /* Original setup casts BF16 compressor gamma to half once. */
        for(int d=0;d<a->head_dim;d++){uint32_t bits=(uint32_t)norm[d]<<16;float x;memcpy(&x,&bits,4);norm[d]=bn_fp32_to_fp16(x);}
        if(r->gpu->resident_alloc(r->request,bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16),(size_t)a->head_dim,&p->norms[l])||
           r->gpu->resident_write(r->request,p->norms[l],norm))goto done;
    }
    for(int family=0;family<2;family++) {
        BnModelNativeRopeFrequency f;if(bn_model_native_rope_frequency(r->model,family,&f))goto done;
        int slot=family?TP_COMP_FREQ:TP_MAIN_FREQ;
        BnGPUF32RopeFrequencyResidentPlan fp={f.rotary_dim,f.yarn,f.base,f.factor,f.ramp_start,f.ramp_scale,p->buffers[slot],p->buffers[slot+1]};
        if(r->gpu->rope_frequency_f32_resident(r->gpu->ctx,r->request,&fp))goto done;
    }
    status=r->gpu->resident_synchronize(r->request);
done:free(norm);return status;
}
static BnGPUResidentView trellis_producer_view(TrellisRequest*r,int slot,size_t n) {
    return n?bn_gpu_resident_view_slice(r->producer->buffers[slot],0,n):(BnGPUResidentView){0};
}
static int trellis_producer_inputs(TrellisRequest*r,int layer,BnGPUResidentView x,BnGPUResidentView qf,
    BnGPUResidentView cf,BnGPUResidentView nf) {
    if(!r->producer||!r->attention||!r->index||!r->index->active||layer<0||layer>=r->anatomy->experts.layers||
       r->attention->used[layer]==r->index->generation)return -1;
    BnGPUResidentView v[]={x,qf,cf,nf};size_t n[]={(size_t)r->index->tokens*r->anatomy->experts.dim,32,32,32};
    for(int j=0;j<4;j++) {
        if(v[j].request!=r->request||v[j].count!=n[j]||v[j].offset>SIZE_MAX-v[j].count||
           v[j].type!=bn_backend_quant_scalar_type(j?BN_BACKEND_SCALAR_F32:BN_BACKEND_SCALAR_F16))return -1;
        for(int k=0;k<j;k++)if(trellis_index_overlap(v[j],v[k])&&!(j&&k&&trellis_attention_same_view(v[j],v[k])))return -1;
        for(int k=0;k<TP_COUNT;k++)if(trellis_index_overlap(v[j],r->producer->buffers[k])&&
            !(j&&k>=TP_MAIN_FREQ&&trellis_attention_same_view(v[j],r->producer->buffers[k])))return -1;
        for(int k=0;k<TA_COUNT;k++)if(trellis_index_overlap(v[j],r->attention->buffers[k]))return -1;
        for(int k=0;k<TI_COUNT;k++)if(trellis_index_overlap(v[j],r->index->buffers[k]))return -1;
        if(trellis_index_overlap(v[j],r->swa->scratch)||trellis_index_overlap(v[j],r->kv->block_table))return -1;
        for(int l=0;l<r->anatomy->experts.layers;l++) {
            const TrellisKVSlot*s=&r->kv->slots[l];BnGPUResidentView b[]={s->c,s->r,s->idx,s->ring_kv,s->ring_gate,r->swa->slots[l].ring,r->producer->norms[l]};
            for(size_t k=0;k<sizeof(b)/sizeof(*b);k++)if(trellis_index_overlap(v[j],b[k]))return -1;
        }
    }
    /* Inspect the same retained-history bounds without a second session lookup. */
    const TrellisSWASlot*s=&r->swa->slots[layer];if(r->index->pos<s->beg||r->index->pos>s->cursor)return -1;
    const BnModelNativeLayerAnatomy*l=&r->anatomy->layers[layer];if(!l->compress_ratio)return 0;
    const TrellisKVSlot*pool=&r->kv->slots[l->kv_source];int end=r->index->pos+r->index->tokens;
    if(l->index_source==layer&&end/l->compress_ratio>512&&r->index->selection_revision==UINT64_MAX)return -1;
    if(l->kv_source==layer)return r->index->pos>pool->cursor||pool->cursor-r->index->pos>256||pool->generation==UINT64_MAX?-1:0;
    if(pool->cursor!=end||pool->generation<=r->index->seen[l->kv_source])return -1;
    int entries=end/l->compress_ratio;TrellisIndex*i=r->index;
    if(entries>512&&l->index_source!=layer&&(!i->selection_valid||i->selection_source!=l->index_source||
       i->selection_owner!=l->kv_source||i->selection_entries!=entries||i->selection_version!=pool->generation))return -1;
    if(entries>16384&&l->candidate_mode==BN_NATIVE_CANDIDATE_USE&&(!i->candidate_valid||i->candidate_entries!=entries||
       i->candidate_rate!=l->compress_ratio||r->kv->slots[i->candidate_owner].generation!=i->candidate_version))return -1;
    return 0;
}
int bn_transformer_trellis_attention_forward(const BnModel*m,BnSession*s,int layer,BnGPUResidentView x,
    BnGPUResidentView qfreq,BnGPUResidentView cfreq,BnGPUResidentView nfreq,BnTransformerTrellisAttentionResult*out) {
    TrellisRequest*r=trellis_request_get(m,s);
    if(!r||!out||trellis_producer_inputs(r,layer,x,qfreq,cfreq,nfreq)||trellis_producer_weights(m,layer))return -1;
    const BnModelNativeAnatomy*a=r->anatomy;const BnModelNativeLayerAnatomy*l=&a->layers[layer];int R=r->index->tokens,pos=r->index->pos,D=a->experts.dim;
    float eps=bn_model_native_numerics(m)->rms_epsilon;BnGPUBackend*g=r->gpu;
    BnGPUResidentView raw=trellis_producer_view(r,TP_QRAW,(size_t)R*a->q_rank),qres=trellis_producer_view(r,TP_QRES,(size_t)R*a->q_rank),
        q=trellis_producer_view(r,TP_Q,(size_t)R*a->heads*a->head_dim),kv=trellis_producer_view(r,TP_KV,(size_t)R*a->head_dim);
    const BnBackendNativeF16Weight*qa=trellis_producer_core(m,layer,BN_NATIVE_Q_A,a->q_rank,D),*qb=trellis_producer_core(m,layer,BN_NATIVE_Q_B,a->heads*a->head_dim,a->q_rank),*kw=trellis_producer_core(m,layer,BN_NATIVE_KV,a->head_dim,D);
    const BnBackendNativeParameter*qn=trellis_producer_parameter(m,layer,BN_NATIVE_Q_NORM,1,a->q_rank,0),*kn=trellis_producer_parameter(m,layer,BN_NATIVE_KV_NORM,1,a->head_dim,0);
    BnGPUF16LatentQKVResidentPlan qp={{qa->weight,D,a->q_rank,R,x,raw},{qb->weight,a->q_rank,a->heads*a->head_dim,R,qres,q},{kw->weight,D,a->head_dim,R,x,kv},
        {R,a->q_rank,1,eps,0,1,qn->handle,raw,qres},
        {R,a->heads,a->rope_dim,a->head_dim,a->head_dim-a->rope_dim,a->rope_dim,0,pos,eps,NULL,q,qfreq,{0}},
        {R,1,a->head_dim,a->head_dim,0,a->rope_dim,a->head_dim-a->rope_dim,pos,eps,kn->handle,kv,qfreq,{0}}};
    if(g->latent_qkv_f16_resident(g->ctx,r->request,&qp))goto failed;
    if(l->kv_source==layer) {
        int rate=l->compress_ratio,rows=(pos+R)/rate-pos/rate,cap=rows?rows:1;
        BnGPUResidentView ck=trellis_producer_view(r,TP_COMP_KV,(size_t)R*a->head_dim),cg=rate>1?trellis_producer_view(r,TP_COMP_GATE,(size_t)R*a->head_dim):ck,
            lat=trellis_producer_view(r,TP_LATENT,(size_t)cap*a->head_dim);TrellisKVSlot*pool=&r->kv->slots[layer];
        const BnBackendNativeParameter*w=trellis_producer_parameter(m,layer,BN_NATIVE_COMP_KV,a->head_dim,D,1),*gate=rate>1?trellis_producer_parameter(m,layer,BN_NATIVE_COMP_GATE,a->head_dim,D,1):NULL;
        BnGPUCompressF16ResidentPlan cp={R,pos,rate,a->head_dim,a->rope_dim,256+rate,cap,a->head_dim,1,eps,ck,cg,pool->ring_kv,pool->ring_gate,
            trellis_producer_view(r,TP_APE,(size_t)rate*a->head_dim),r->producer->norms[layer],r->producer->buffers[TP_ZERO_FREQ],lat,{0}};
        BnGPUF16CompressorResidentPlan project={{w->handle,D,a->head_dim,R,x,ck},{0},cp};if(rate>1)project.gate=(BnGPUF16LinearResidentPlan){gate->handle,D,a->head_dim,R,x,cg};
        if(g->compressor_f16_resident(g->ctx,r->request,&project))goto failed;
        BnGPUResidentView keys={0};if(rows) {
            BnGPUResidentView kr=trellis_producer_view(r,TP_KEY_RAW,(size_t)rows*a->index_dim);keys=trellis_producer_view(r,TP_KEYS,(size_t)rows*a->index_dim);
            BnGPUResidentView positions=trellis_producer_view(r,TP_POS,(size_t)rows);
            const BnBackendNativeParameter*ik=trellis_producer_parameter(m,layer,BN_NATIVE_INDEX_K,a->index_dim,a->head_dim,1),*in=trellis_producer_parameter(m,layer,BN_NATIVE_INDEX_K_NORM,1,a->index_dim,0);
            BnGPUF16LatentKeysResidentPlan kp={pos/rate,rate,{ik->handle,a->head_dim,a->index_dim,rows,lat,kr},
                {rows,a->index_dim,1,eps,0,1,in->handle,kr,keys},
                {rows,1,a->rope_dim,a->index_dim,a->index_dim-a->rope_dim,a->rope_dim,0,0,eps,NULL,keys,cfreq,positions},
                {rows,1,a->rope_dim,a->head_dim,a->head_dim-a->rope_dim,a->rope_dim,0,0,eps,NULL,lat,cfreq,positions}};
            if(g->latent_keys_f16_resident(g->ctx,r->request,&kp))goto failed;
        } else lat=(BnGPUResidentView){0};
        if(bn_transformer_trellis_kv_publish(m,s,layer,pos,R,lat,keys))goto failed;
    }
    BnTransformerTrellisIndexResult selection;
    if(bn_transformer_trellis_index_select(m,s,layer,x,qres,cfreq,&selection)||
       bn_transformer_trellis_attention_project(m,s,layer,&selection,q,kv,nfreq,out))goto failed;
    return 0;
failed:if(r->request){r->destroy(r->request);r->request=NULL;}return -1;
}
int bn_transformer_trellis_attention_forward_native(const BnModel*m,BnSession*s,int layer,
    BnGPUResidentView x,BnTransformerTrellisAttentionResult*out) {
    TrellisRequest*r=trellis_request_get(m,s);
    if(!r||!r->producer||layer<0||layer>=r->anatomy->experts.layers)return -1;
    int slot=r->anatomy->layers[layer].compress_ratio?TP_COMP_FREQ:TP_MAIN_FREQ;
    return bn_transformer_trellis_attention_forward(m,s,layer,x,r->producer->buffers[slot],
        r->producer->buffers[TP_COMP_FREQ],r->producer->buffers[slot+1],out);
}
#endif
