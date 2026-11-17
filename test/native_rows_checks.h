#ifndef __EMSCRIPTEN__
/* Control-flow mock only: numerical decode is qualified against source on CUDA. */
typedef struct {NativeExecFake base;int busy,fail_decode,uploads,decodes;} NativeRowsFake;
typedef struct {NativeExecRequest base;unsigned char *pins[8];size_t capacity;unsigned pin_count;} NativeRowsRequest;
static void *native_rows_create(void *ctx) {
    NativeRowsRequest *r=calloc(1,sizeof(*r));assert(r);r->base.owner=ctx;
    ((NativeRowsFake*)ctx)->base.created++;return r;
}
static void native_rows_destroy(void *ptr) {
    NativeRowsRequest *r=ptr;for(unsigned i=0;i<8;i++)free(r->pins[i]);native_exec_destroy(ptr);
}
static int native_rows_reserve(void *ptr,unsigned slots,size_t bytes,size_t budget) {
    NativeRowsRequest *r=ptr;assert(slots<=8&&slots*bytes<=budget&&!r->pin_count);
    r->pin_count=slots;r->capacity=bytes;for(unsigned i=0;i<slots;i++){r->pins[i]=malloc(bytes);assert(r->pins[i]);}return 0;
}
static int native_rows_acquire(void *ptr,unsigned slot,BnGPUHostUploadLease *out) {
    NativeRowsRequest *r=ptr;NativeRowsFake *t=(NativeRowsFake*)r->base.owner;
    if(t->busy)return BN_GPU_UPLOAD_BUSY;
    assert(slot<r->pin_count);*out=(BnGPUHostUploadLease){ptr,r->pins[slot],r->capacity,1,slot};return 0;
}
static int native_rows_cancel(void *ptr,const BnGPUHostUploadLease *lease) {assert(lease->request==ptr);return 0;}
static int native_rows_upload(void *ptr,const BnGPUHostUploadLease *lease,BnGPUResidentView v) {
    NativeRowsRequest *r=ptr;NativeRowsFake *t=(NativeRowsFake*)r->base.owner;
    assert(lease->request==ptr&&v.request==ptr);t->uploads++;return 0;
}
static int native_rows_decode(void *ctx,void *ptr,const BnGPUQuantRowsBF16ResidentPlan *p) {
    NativeRowsFake *t=ctx;assert(p->out.request==ptr&&p->scales.request==ptr&&p->packed.request==ptr);
    t->decodes++;return t->fail_decode?-1:0;
}
static void native_rows_foreign_release(void *ptr) {(*(int*)ptr)++;}
static void test_native_rows(void) {
    char dir[]="/tmp/bn-native-rows-XXXXXX",path[256];assert(mkdtemp(dir));native_anatomy_fixture(dir,0);
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    char *config=native_numerics_replace(native_anatomy_config,"\"engram_compressed_vocab_size\":99092","\"engram_compressed_vocab_size\":256");
    BnModel m={0};assert(!bn_model_load_trellis_config(&m,&index,config,strlen(config)));free(config);
    assert(bn_model_prepare_native_rows(&m)==-1);
    assert(!bn_model_prepare_native_anatomy(&m)&&!bn_model_prepare_native_numerics(&m));
    BnTokenizerMapSpec ms;assert(!bn_model_native_engram_map_spec(&m,&ms)&&ms.profile==BN_TOKENIZER_MAP_ENGRAM_V41&&ms.vocab_size==256&&ms.compressed_vocab==256&&ms.pad_token==2);
    uint32_t map_ids[256];for(unsigned i=0;i<256;i++)map_ids[i]=i;
    int raw_tokens[3]={1,2,255};uint32_t mapped[3]={99,98,97},unchanged[3];memcpy(unchanged,mapped,sizeof(mapped));size_t map_count=777;
    assert(!bn_model_native_token_map(&m,&map_count)&&map_count==777&&bn_model_native_map_tokens(&m,raw_tokens,3,mapped)==-1&&!memcmp(mapped,unchanged,sizeof(mapped)));
    assert(bn_model_prepare_native_token_map(&m,map_ids,256,sizeof(map_ids)-1)==-1);
    map_ids[0]=1;assert(bn_model_prepare_native_token_map(&m,map_ids,256,sizeof(map_ids))==-1);map_ids[0]=0;
    map_ids[255]=254;assert(bn_model_prepare_native_token_map(&m,map_ids,256,sizeof(map_ids))==-1);map_ids[255]=255;
    assert(!bn_model_prepare_native_token_map(&m,map_ids,256,sizeof(map_ids)));const uint32_t *owned_map=bn_model_native_token_map(&m,&map_count);assert(owned_map&&map_count==256&&!memcmp(owned_map,map_ids,sizeof(map_ids)));
    map_ids[0]=17;assert(owned_map[0]==0&&bn_model_prepare_native_token_map(&m,map_ids,256,sizeof(map_ids))==-1);
    BnModelNativeTokenMapInfo mi;assert(!bn_model_native_token_map_info(&m,&mi)&&mi.mapped_pad==2&&mi.host_bytes==1024);
    assert(!bn_model_native_map_tokens(&m,raw_tokens,3,mapped)&&mapped[0]==1&&mapped[1]==2&&mapped[2]==255);memcpy(unchanged,mapped,sizeof(mapped));
    raw_tokens[2]=-1;assert(bn_model_native_map_tokens(&m,raw_tokens,3,mapped)==-1&&!memcmp(mapped,unchanged,sizeof(mapped)));raw_tokens[2]=255;
    assert(bn_model_native_map_tokens(&m,raw_tokens,3,(uint32_t*)(void*)raw_tokens)==-1&&raw_tokens[0]==1);
    assert(bn_model_native_map_tokens(&m,raw_tokens,3,(uint32_t*)(void*)owned_map)==-1&&owned_map[0]==0);
    assert(!bn_model_native_map_tokens(&m,NULL,0,NULL));
    BnModelNativeAnatomy *a=(BnModelNativeAnatomy*)bn_model_native_anatomy(&m);assert(a->engram_format==BN_QUANT_FORMAT_INT4_FP16S32);
    int saved=a->layers[14].tensor[BN_NATIVE_ENGRAM_SCALE];a->layers[14].tensor[BN_NATIVE_ENGRAM_SCALE]=-1;
    assert(bn_model_prepare_native_rows(&m)==-1&&!bn_model_native_row_source(&m,1));a->layers[14].tensor[BN_NATIVE_ENGRAM_SCALE]=saved;
    assert(!bn_model_prepare_native_rows(&m)&&bn_model_prepare_native_rows(&m)==-1);
    assert(!bn_model_native_row_source(&m,-1)&&!bn_model_native_row_source(&m,0)&&!bn_model_native_row_source(&m,40));
    const BnTrellisRowLayout *layout=bn_trellis_row_source_layout(bn_model_native_row_source(&m,14));assert(layout&&layout->rows==96&&layout->cols==32);
    assert(bn_model_prepare_native_row_caches(&m,16384,8,65536)==-1);
    NativeRowsFake t={0};t.base.owner.model=&m;
    BnGPUBackend g={.ctx=&t,.buffer_create_f16_rows=native_owner_core_create,.buffer_destroy=native_owner_destroy,
        .trellis_buffer_create=native_owner_expert_create,.resident_request_create=native_rows_create,.resident_request_destroy=native_rows_destroy,
        .resident_alloc=native_exec_alloc,.resident_write=native_exec_write,.resident_read=native_exec_read,
        .resident_zero=native_exec_zero,.resident_synchronize=native_exec_sync,.router_sqrtsp_resident=native_exec_router,
        .trellis_decode_resident=native_exec_decode,.linear_f16_workspace_size=native_exec_size,.linear_f16_reserve=native_exec_reserve,
        .ffn_f16_resident=native_exec_shared,.add_f32_resident=native_exec_add,.resident_upload_reserve=native_rows_reserve,
        .resident_upload_acquire=native_rows_acquire,.resident_upload_submit=native_rows_upload,.resident_upload_cancel=native_rows_cancel,
        .quant_rows_bf16_resident=native_rows_decode};
    BnBackendNativeCoreStats cs;assert(!bn_backend_layout_native_core_plan(&index,a,&cs));assert(!bn_model_prepare_native_core(&m,&g,cs.device_peak_bytes,cs.host_staging_bytes));
    assert(bn_model_prepare_native_row_caches(&m,16384,8,1)==-1&&!bn_model_native_row_cache(&m,1));
    assert(!bn_model_prepare_native_row_caches(&m,16384,8,65536));BnModelNativeRowsInfo info;
    assert(!bn_model_native_row_cache_info(&m,&info)&&info.tables==2&&info.host_allocated<=info.host_bound&&info.host_bound<=65536);
    assert(!bn_model_prepare_native_row_caches(&m,16384,8,info.host_bound));
    assert(bn_model_prepare_native_row_caches(&m,16384,8,info.host_bound-1)==-1);
    assert(bn_model_prepare_native_row_caches(&m,16384,9,65536)==-1);
    BnBackendRowCache *cache=bn_model_native_row_cache(&m,1);assert(cache&&!bn_model_native_row_cache(&m,0));
    const BnMoEExpertReader *reader=bn_model_trellis_expert_reader(&m);size_t entry=reader->bytes[0]+reader->bytes[1]+reader->bytes[2];
    assert(!bn_gpu_moe_trellis_prepare_model(&m,entry,entry,entry,entry));
    BnTransformerTrellisOptions f={1,1,0,0,0};BnTransformerTrellisWorkspace fw;assert(!bn_transformer_trellis_session_plan(&m,&f,&fw));f.device_budget=fw.device_bytes;f.host_budget=fw.host_bytes;
    BnTransformerTrellisRowsOptions o={8,2,0,0,0};BnTransformerTrellisRowsWorkspace w;assert(!bn_transformer_trellis_rows_session_plan(&m,&o,&w)&&w.tables==2);
    o.host_budget=w.host_bytes;o.pinned_budget=w.pinned_bytes;o.device_budget=w.device_bytes;
    BnSession *s=bn_session_create_backend(&m,NULL),*other=bn_session_create_backend(&m,NULL);assert(s&&other);
    BnTransformerTrellisRowsOptions bad=o;bad.pinned_budget--;assert(bn_transformer_trellis_prepare_session_rows(&m,s,&f,&bad)==-1&&!t.base.created);
    assert(!bn_transformer_trellis_prepare_session_rows(&m,s,&f,&o));int created=t.base.created;
    assert(!bn_transformer_trellis_prepare_session_rows(&m,s,&f,&o)&&created==t.base.created);
    uint64_t ids[]={0,63,0};BnTransformerTrellisRowsResult result,old,bank1;memset(&result,0xa5,sizeof(result));old=result;
    t.busy=1;assert(bn_transformer_trellis_rows_submit(&m,s,1,0,ids,3,&result)==BN_GPU_UPLOAD_BUSY&&!memcmp(&result,&old,sizeof(result))&&!t.uploads);t.busy=0;
    assert(!bn_transformer_trellis_rows_submit(&m,s,1,0,ids,3,&result)&&t.uploads==2&&t.decodes==1);old=result;
    void *request=NULL;BnGPUResidentView input;assert(!bn_transformer_trellis_session_input(&m,s,1,&request,&input)&&result.request==request);
    assert(bn_transformer_trellis_rows_current(&m,s,&result)&&!bn_transformer_trellis_rows_current(&m,other,&result));
    assert(!bn_transformer_trellis_rows_submit(&m,s,14,1,ids,3,&bank1)&&bn_transformer_trellis_rows_current(&m,s,&old));
    assert(!bn_transformer_trellis_rows_submit(&m,s,14,0,ids,3,&result)&&!bn_transformer_trellis_rows_current(&m,s,&old)&&bn_transformer_trellis_rows_current(&m,s,&bank1));
    ids[1]=96;old=result;assert(bn_transformer_trellis_rows_submit(&m,s,14,0,ids,3,&result)==-1&&!memcmp(&old,&result,sizeof(old))&&bn_transformer_trellis_rows_current(&m,s,&old));ids[1]=63;
    bn_session_reset(s,&m);assert(!bn_transformer_trellis_rows_current(&m,s,&old)&&bn_model_native_row_cache(&m,1)==cache);
    assert(!bn_transformer_trellis_prepare_session_rows(&m,s,&f,&o));assert(!bn_transformer_trellis_rows_submit(&m,s,14,0,ids,3,&result));
    /* Force the old request/view addresses to match: generation must still reject it. */
    old.request=result.request;old.values=result.values;old.epoch=result.epoch;assert(!bn_transformer_trellis_rows_current(&m,s,&old));
    t.fail_decode=1;old=result;assert(bn_transformer_trellis_rows_submit(&m,s,1,1,ids,3,&result)==-1&&!memcmp(&old,&result,sizeof(old)));
    assert(!bn_backend_session_has_gpu_resources(bn_session_backend(s))&&!bn_transformer_trellis_rows_current(&m,s,&old));
    bn_session_free(s,NULL);bn_session_free(other,NULL);assert(t.base.created==t.base.destroyed);bn_model_free(&m);
    char *small_config=native_numerics_replace(native_anatomy_config,"\"engram_compressed_vocab_size\":99092","\"engram_compressed_vocab_size\":128");
    for(unsigned i=0;i<256;i++)map_ids[i]=i/2;
    for(int late=0;late<2;late++) {
        BnModel mapped_model={0};assert(!bn_model_load_trellis_config(&mapped_model,&index,small_config,strlen(small_config)));
        assert(!bn_model_prepare_native_anatomy(&mapped_model)&&!bn_model_prepare_native_numerics(&mapped_model));
        if(late){assert(!bn_model_ensure_backend(&mapped_model));assert(bn_model_prepare_native_token_map(&mapped_model,map_ids,256,sizeof(map_ids))==-1&&!bn_model_native_token_map(&mapped_model,NULL));}
        else {assert(!bn_model_prepare_native_token_map(&mapped_model,map_ids,256,sizeof(map_ids))&&!bn_model_native_token_map_info(&mapped_model,&mi)&&mi.mapped_pad==1);
            assert(!bn_model_native_map_tokens(&mapped_model,raw_tokens,3,mapped)&&mapped[0]==0&&mapped[1]==1&&mapped[2]==127);}
        bn_model_free(&mapped_model);
    }
    free(small_config);
    BnBackendModel *b=bn_backend_model_create();int released=0;assert(b);
    assert(!bn_backend_model_attach_cpu_resources(b,&released,&released,native_rows_foreign_release));
    assert(bn_backend_model_attach_cpu_resources(b,&released,&released,native_rows_foreign_release)==-1);
    assert(bn_backend_model_cpu_resources(b,&released)==&released&&!bn_backend_model_cpu_resources(b,&m));bn_backend_model_free(b);assert(released==1);
    bn_safetensors_index_free(&index);assert(!remove(path));snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!remove(path)&&!remove(dir));
    puts("Native model row binding, backend cache ownership and request staging lifecycle PASSED");
}
#else
static void test_native_rows(void) {}
#endif
