#ifndef BN_TRANSFORMER_TRELLIS_BLOCK_H
#define BN_TRANSFORMER_TRELLIS_BLOCK_H
static int trellis_block_prepare(const BnModel*m,BnSession*s,int layer,int admit) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisHC*h=r?r->hc:NULL;
    if(!h||!r->producer||!r->attention||!r->index||!r->index->active||
       layer<0||layer>=h->workspace.layers||(admit!=0&&admit!=1)||!trellis_hc_history(m,s,h)||
       r->index->tokens!=h->tokens||r->index->pos!=h->position)return -1;
    if(r->block_layer) {
        if(r->block_layer!=layer+1||r->block_admit!=admit||r->block_epoch!=h->epoch||
           r->block_index_generation!=r->index->generation||
           h->site!=2*layer+r->block_phase||h->phase!=(r->block_phase?2:1))return -1;
    }else {
        if(h->site!=2*layer||h->phase!=1||
           (layer&&r->block_index_generation!=r->index->generation))return -1;
        r->block_layer=layer+1;r->block_phase=0;r->block_admit=admit;
        r->block_epoch=h->epoch;r->block_index_generation=r->index->generation;
    }
    if(!r->block_phase) {
        BnTransformerTrellisHCResult v;BnTransformerTrellisAttentionResult a;
        int status=bn_transformer_trellis_hc_mix(m,s,&v);
        if(status==BN_GPU_UPLOAD_BUSY)return trellis_request_get(m,s)?status:trellis_hc_fail(s);
        if(status)return trellis_hc_fail(s);
        if(bn_transformer_trellis_attention_forward_native(m,s,layer,v.normalized,&a)||
           bn_transformer_trellis_hc_apply(m,s,&v,a.delta))return trellis_hc_fail(s);
        if(bn_transformer_trellis_hc_mix(m,s,&v))return trellis_hc_fail(s);
        r->block_phase=1;r->block_epoch=h->epoch;
    }
    return 0;
}
int bn_transformer_trellis_block_step(const BnModel*m,BnSession*s,int layer,int admit,
    BnTransformerTrellisBlockResult*out) {
    if(!out)return -1;
    int prepared=trellis_block_prepare(m,s,layer,admit);if(prepared)return prepared;
    TrellisRequest *r=trellis_request_get(m,s);TrellisHC *h=r->hc;
    BnTransformerTrellisResult f;
    int status=bn_transformer_trellis_run_learned(m,s,layer,h->tokens,admit,&f);
    if(status==BN_GPU_UPLOAD_BUSY)return trellis_request_get(m,s)?status:trellis_hc_fail(s);
    if(status)return trellis_hc_fail(s);
    BnTransformerTrellisHCResult v=trellis_hc_result(r,s);
    if(bn_transformer_trellis_hc_apply(m,s,&v,f.out))return trellis_hc_fail(s);
    r->block_layer=0;
    *out=(BnTransformerTrellisBlockResult){r->request,s,r->generation,h->epoch,h->history_epoch,
        r->index->generation,layer,h->tokens,h->position,trellis_hc_view(r,h->stream)};
    return 0;
}
int bn_transformer_trellis_block_current(const BnModel*m,BnSession*s,const BnTransformerTrellisBlockResult*v) {
    TrellisRequest*r=trellis_request_get(m,s);TrellisHC*h=r?r->hc:NULL;
    return h&&r->index&&v&&v->epoch&&!r->block_layer&&h->phase==1&&v->layer>=0&&v->layer<h->workspace.layers&&
        v->request==r->request&&v->session==s&&v->generation==r->generation&&v->epoch==h->epoch&&
        v->history_epoch==h->history_epoch&&v->index_generation==r->index->generation&&
        v->tokens==h->tokens&&v->position==h->position&&h->site==2*(v->layer+1)&&trellis_hc_history(m,s,h)&&
        trellis_engram_same(v->streams,trellis_hc_view(r,h->stream));
}
#endif
