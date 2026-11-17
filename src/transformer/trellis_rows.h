static int trellis_rows_options(const BnModel *m,const BnTransformerTrellisRowsOptions *o,
    BnBackendRowStageOptions *out,BnModelNativeRowsInfo *info) {
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(m);BnGPUBackend *g=bn_model_gpu(m);
    if(!a||!o||!g||!g->resident_alloc||!g->resident_upload_reserve||!g->resident_upload_acquire||
        !g->resident_upload_submit||!g->resident_upload_cancel||!g->quant_rows_bf16_resident||
        bn_model_native_row_cache_info(m,info)||info->max_rows<o->max_rows)return -1;
    for(int layer=0;layer<a->experts.layers;layer++)if(a->layers[layer].engram_rows) {
        BnBackendRowCache *cache=bn_model_native_row_cache(m,layer);
        const BnTrellisRowLayout *layout=bn_backend_model_row_cache_layout(cache);
        if(!layout||layout->format!=a->engram_format||layout->cols!=a->engram_dim)return -1;
    }
    *out=(BnBackendRowStageOptions){a->engram_format,a->engram_dim,o->max_rows,o->banks,
        o->host_budget,o->pinned_budget,o->device_budget};return 0;
}
int bn_transformer_trellis_rows_session_plan(const BnModel *m,const BnTransformerTrellisRowsOptions *o,
    BnTransformerTrellisRowsWorkspace *out) {
    BnBackendRowStageOptions options;BnModelNativeRowsInfo info;BnBackendRowStageWorkspace w;
    if(!out||trellis_rows_options(m,o,&options,&info)||bn_backend_session_row_stage_plan(&options,&w))return -1;
    *out=(BnTransformerTrellisRowsWorkspace){w.host_bytes,w.pinned_bytes,w.device_bytes,info.tables,o->max_rows,o->banks};return 0;
}
static int trellis_rows_setup(TrellisRequest *r,const BnTransformerTrellisRowsOptions *o) {
    BnBackendRowStageOptions options;BnModelNativeRowsInfo info;
    if(trellis_rows_options(r->model,o,&options,&info)||bn_backend_session_row_stage_init(&r->rows,r->gpu,r->request,&options))return -1;
    r->rows_options=*o;for(int i=0;i<4;i++)r->row_layers[i]=-1;return 0;
}
int bn_transformer_trellis_rows_submit(const BnModel *m,BnSession *session,int layer,unsigned bank,
    const uint64_t *ids,size_t count,BnTransformerTrellisRowsResult *out) {
    TrellisRequest *r=trellis_request_get(m,session);BnBackendRowCache *cache=bn_model_native_row_cache(m,layer);
    if(!r||!r->rows.state||!cache||!out||bank>=(unsigned)r->rows_options.banks)return -1;
    if(!count)return 0;
    BnGPUResidentView view;uint64_t epoch;
    int result=bn_backend_session_row_stage_submit(&r->rows,cache,bank,ids,count,&view,&epoch);
    if(result) {
        BnBackendRowStageInfo info;
        if(bn_backend_session_row_stage_info(&r->rows,&info)||info.failed)
            bn_backend_session_release_gpu_resources(bn_session_backend(session));
        return result;
    }
    r->row_layers[bank]=layer;
    *out=(BnTransformerTrellisRowsResult){r->request,session,layer,bank,r->generation,epoch,view};return 0;
}
int bn_transformer_trellis_rows_current(const BnModel *m,BnSession *session,const BnTransformerTrellisRowsResult *result) {
    TrellisRequest *r=trellis_request_get(m,session);
    return r&&result&&r->rows.state&&result->session==session&&result->generation==r->generation&&result->request==r->request&&result->bank<(unsigned)r->rows_options.banks&&
        result->layer==r->row_layers[result->bank]&&
        bn_backend_session_row_stage_current(&r->rows,result->bank,result->epoch,&result->values);
}
