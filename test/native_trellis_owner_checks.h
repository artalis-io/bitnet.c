#ifndef __EMSCRIPTEN__
typedef struct {
    NativeF16Fake core;
    void *core_handles[BN_MODEL_NATIVE_MAX_CORE_PROJECTIONS];
    int core_count, uploads, expert_frees, foreign_releases;
    size_t expert_live;
    BnModel *model;
} NativeTrellisOwnerGPU;
typedef struct {size_t bytes;unsigned char data[];} NativeTrellisOwnerWeight;
static void *native_owner_core_create(void *context,const BnGPUF16RowsPlan *p) {
    NativeTrellisOwnerGPU *t=context;
    void *w=native_f16_fake_create(&t->core,p);assert(w);
    assert(t->core_count<BN_MODEL_NATIVE_MAX_CORE_PROJECTIONS);
    t->core_handles[t->core_count++]=w;return w;
}
static void *native_owner_expert_create(void *context,const void *data,
    size_t bytes,int format,int k,int n,uint32_t marker) {
    NativeTrellisOwnerGPU *t=context;
    assert(data&&format==BN_QUANT_FORMAT_EXL3_MUL1_3&&marker==UINT32_C(0x83dcd12d));
    assert(bytes==(size_t)k*n*3/8+(size_t)(k+n)*2&&t->core.active);
    BnGPUMoETrellisStore *store=bn_gpu_moe_trellis_model_store(t->model);
    BnGPUMoETrellisStats stats;assert(store&&!bn_gpu_moe_trellis_stats(store,&stats));
    assert(stats.decode.pending+stats.temporary.pending==1);
    NativeTrellisOwnerWeight *w=malloc(sizeof(*w)+bytes);assert(w);
    w->bytes=bytes;memcpy(w->data,data,bytes);t->uploads++;t->expert_live+=bytes;return w;
}
static void native_owner_destroy(void *context,void *handle) {
    NativeTrellisOwnerGPU *t=context;
    for(int i=0;i<t->core_count;i++)if(t->core_handles[i]==handle) {
        assert(!t->expert_live);
        t->core_handles[i]=NULL;native_f16_fake_destroy(&t->core,handle);return;
    }
    NativeTrellisOwnerWeight *w=handle;
    assert(t->core.active&&bn_model_trellis_expert_reader(t->model));
    assert(t->expert_live>=w->bytes);t->expert_live-=w->bytes;t->expert_frees++;free(w);
}
static void native_owner_foreign_release(void *context) {
    NativeTrellisOwnerGPU *t=context;
    assert(t->core.active&&bn_model_trellis_expert_reader(t->model));
    t->foreign_releases++;
}
static void test_native_trellis_model_owner(void) {
    char dir[]="/tmp/bn-trellis-owner-XXXXXX",path[256];assert(mkdtemp(dir));
    native_anatomy_fixture(dir,0);snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);
    BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    char *config=native_numerics_replace(native_anatomy_config,
        "\"engram_compressed_vocab_size\":99092","\"engram_compressed_vocab_size\":256");
    assert(bn_gpu_moe_trellis_prepare_model(NULL,0,0,0,0)==-1&&!bn_gpu_moe_trellis_model_store(NULL));
    for(int mode=0;mode<3;mode++) {
        int foreign=mode==1;
        BnModel model={0};assert(!bn_model_load_trellis_config(&model,&index,config,strlen(config)));
        assert(!bn_model_prepare_native_anatomy(&model)&&!bn_model_prepare_native_numerics(&model));
        NativeTrellisOwnerGPU t={0};t.model=&model;
        BnGPUBackend gpu={.ctx=&t,.buffer_create_f16_rows=native_owner_core_create,
            .buffer_destroy=native_owner_destroy,.trellis_buffer_create=native_owner_expert_create};
        const BnMoEExpertReader *reader=bn_model_trellis_expert_reader(&model);assert(reader);
        size_t entry=reader->bytes[0]+reader->bytes[1]+reader->bytes[2];
        assert(bn_gpu_moe_trellis_prepare_model(&model,entry,entry,entry,entry)==-1&&!model.backend_state);
        BnBackendNativeCoreStats plan;assert(!bn_backend_layout_native_core_plan(&index,bn_model_native_anatomy(&model),&plan));
        assert(!bn_model_prepare_native_core(&model,&gpu,plan.device_peak_bytes,plan.host_staging_bytes));
        BnBackendModel *backend=bn_model_backend(&model);int owner=0,other=0;
        const BnBackendNativeCoreStats *core=bn_backend_layout_native_core_stats(backend);
        assert(core&&t.core_count==330&&!bn_gpu_moe_trellis_model_store(&model));
        assert(bn_backend_layout_attach_native_execution(NULL,&owner,&t,native_owner_foreign_release)==-1);
        assert(bn_backend_layout_attach_native_execution(backend,NULL,&t,native_owner_foreign_release)==-1);
        assert(bn_backend_layout_attach_native_execution(backend,&owner,NULL,native_owner_foreign_release)==-1);
        assert(bn_backend_layout_attach_native_execution(backend,&owner,&t,NULL)==-1);
        assert(bn_backend_layout_attach_native_execution(backend,&owner,(void*)core,native_owner_foreign_release)==-1);
        assert(!bn_backend_layout_native_execution(NULL,&owner)&&!bn_backend_layout_native_execution(backend,NULL));
        bn_backend_model_set_gpu_disabled(backend,1);
        assert(bn_backend_layout_attach_native_execution(backend,&owner,&t,native_owner_foreign_release)==-1);
        assert(bn_gpu_moe_trellis_prepare_model(&model,entry,entry,entry,entry)==-1);
        bn_backend_model_set_gpu_disabled(backend,0);
        assert(bn_gpu_moe_trellis_prepare_model(&model,entry,entry,entry,entry-1)==-1);
        gpu.trellis_buffer_create=NULL;
        assert(bn_gpu_moe_trellis_prepare_model(&model,entry,entry,entry,entry)==-1);
        gpu.trellis_buffer_create=native_owner_expert_create;
        assert(!bn_gpu_moe_trellis_model_store(&model)&&!t.uploads&&t.core_count==330);
        assert(bn_backend_layout_native_core_stats(backend)==core);
        if(foreign) {
            assert(!bn_backend_layout_attach_native_execution(backend,&owner,&t,native_owner_foreign_release));
            assert(bn_backend_layout_native_execution(backend,&owner)==&t&&!bn_backend_layout_native_execution(backend,&other));
            assert(bn_backend_layout_attach_native_execution(backend,&other,&t,native_owner_foreign_release)==-1);
            assert(bn_gpu_moe_trellis_prepare_model(&model,entry,entry,entry,entry)==-1);
            assert(!bn_gpu_moe_trellis_model_store(&model)&&!t.uploads&&!t.foreign_releases);
            BnBackendModel *target=bn_backend_model_create();assert(target);
            assert(!bn_backend_model_move_gpu_resources(target,backend));
            assert(!bn_backend_layout_native_execution(backend,&owner));
            assert(bn_backend_layout_native_execution(target,&owner)==&t);
            assert(bn_backend_layout_native_core_stats(target)==core);
            bn_backend_model_free(target);assert(t.foreign_releases==1&&!t.core.active);
        } else {
            BnGPUMoETrellisPolicy policy={.ram_admit_misses=1,.ram_pages=BN_MOE_CACHE_PAGES_SYSTEM};
            if(mode==2)assert(!bn_gpu_moe_trellis_prepare_model_policy(&model,entry,entry,entry,entry,0,&policy));
            else assert(!bn_gpu_moe_trellis_prepare_model(&model,entry,entry,entry,entry));
            BnGPUMoETrellisStore *store=bn_gpu_moe_trellis_model_store(&model);assert(store);
            assert(bn_gpu_moe_trellis_prepare_model(&model,entry,entry,entry,entry)==-1);
            assert(store==bn_gpu_moe_trellis_model_store(&model)&&!t.uploads&&t.core_count==330);
            BnModel other_model={0};
            assert(!bn_model_load_trellis_config(&other_model,&index,config,strlen(config)));
            assert(!bn_model_ensure_backend(&other_model));
            BnBackendModel *other_backend=bn_model_backend(&other_model);
            assert(!bn_backend_model_move_gpu_resources(other_backend,backend));
            assert(!bn_gpu_moe_trellis_model_store(&model));
            assert(!bn_gpu_moe_trellis_model_store(&other_model));
            assert(bn_gpu_moe_trellis_prepare_model(&other_model,entry,entry,entry,entry)==-1);
            assert(!bn_backend_model_move_gpu_resources(backend,other_backend));
            assert(bn_gpu_moe_trellis_model_store(&model)==store&&!t.uploads);
            bn_model_free(&other_model);
            BnGPUMoETrellisLease a={0},b={0},scan={0};
            assert(!bn_gpu_moe_trellis_acquire(store,0,0,1,&a));
            assert(!bn_gpu_moe_trellis_acquire(store,0,0,0,&b));
            assert(a.buffers.gate==b.buffers.gate&&t.uploads==3);
            assert(!bn_gpu_moe_trellis_release(store,&a));
            assert(bn_gpu_moe_trellis_acquire(store,1,1,1,&scan)==BN_GPU_MOE_CACHE_BUSY&&!scan.owner&&t.uploads==3);
            assert(!bn_gpu_moe_trellis_acquire(store,1,1,0,&scan)&&t.uploads==6&&t.expert_live==entry*2);
            assert(!bn_gpu_moe_trellis_release(store,&scan)&&t.expert_live==entry);
            assert(!bn_gpu_moe_trellis_release(store,&b));
            BnGPUMoETrellisStats stats;assert(!bn_gpu_moe_trellis_stats(store,&stats));
            assert(!stats.live_leases&&stats.decode.published==1&&!stats.temporary.pending);
            assert(stats.ram_pages==(mode==2?BN_MOE_CACHE_PAGES_SYSTEM:BN_MOE_CACHE_PAGES_HUGE_HINT));
            assert(stats.host.io_bytes==entry*2&&stats.upload_bytes==entry*2);
        }
        bn_model_free(&model);
        assert(!t.core.active&&!t.expert_live&&t.expert_frees==t.uploads);
        assert(!bn_gpu_moe_trellis_model_store(&model));
        for(int i=0;i<t.core_count;i++)assert(!t.core_handles[i]);
    }
    bn_safetensors_index_free(&index);free(config);
    snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!remove(path));
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!remove(path)&&!rmdir(dir));
    puts("Native backend-owned expert store atomic setup, moves and cleanup order PASSED");
}
#else
static void test_native_trellis_model_owner(void) {}
#endif
