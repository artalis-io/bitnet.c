#ifndef __EMSCRIPTEN__
static void native_core_foreign_release(void *p){(*(int*)p)++;}
static void test_native_core(void) {
    char dir[]="/tmp/bn-native-core-XXXXXX",path[256];assert(mkdtemp(dir));native_anatomy_fixture(dir,0);snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    char *config=native_numerics_replace(native_anatomy_config,"\"engram_compressed_vocab_size\":99092","\"engram_compressed_vocab_size\":256");BnModel model={0};assert(!bn_model_load_trellis_config(&model,&index,config,strlen(config)));assert(!bn_model_prepare_native_anatomy(&model));assert(!bn_model_prepare_native_numerics(&model));const BnModelNativeAnatomy *a=bn_model_native_anatomy(&model);assert(a&&a->core_projection_count==330);
    BnBackendNativeCoreStats stats,saved;memset(&stats,0xa5,sizeof(stats));saved=stats;assert(bn_backend_layout_native_core_plan(NULL,a,&stats)==-1&&!memcmp(&stats,&saved,sizeof(stats)));assert(!bn_backend_layout_native_core_plan(&index,a,&stats)&&stats.count==330);
    size_t bytes=0,peak=0,host=0;int seen[40][BN_NATIVE_ROLE_COUNT]={{0}};
    for(int i=0;i<330;i++) {
        const BnModelNativeCoreProjection *p=&a->core_projections[i];assert(!seen[p->layer][p->weight_role]++);const BnSafetensorsTensor *w=bn_safetensors_index_tensor(&index,(size_t)p->weight_tensor),*s=bn_safetensors_index_tensor(&index,(size_t)p->scale_tensor);assert(w&&s&&w->dtype==BN_SAFETENSORS_F8_E4M3&&s->dtype==BN_SAFETENSORS_F8_E8M0&&w->shape[0]==(uint64_t)p->rows&&w->shape[1]==(uint64_t)p->cols);
        size_t n=(size_t)p->rows*p->cols*2,tile=(size_t)(p->rows<32?p->rows:32)*p->cols*2,h=tile/2*3+((size_t)p->cols+31)/32;if(bytes+n+tile>peak)peak=bytes+n+tile;bytes+=n;if(h>host)host=h;
    }
    assert(bytes==stats.resident_bytes&&peak==stats.device_peak_bytes&&host==stats.host_staging_bytes);
    NativeF16Fake state={0};BnGPUBackend gpu={0};gpu.ctx=&state;gpu.buffer_create_f16_rows=native_f16_fake_create;gpu.buffer_destroy=native_f16_fake_destroy;
    assert(bn_model_prepare_native_core(&model,&gpu,peak-1,host)==-1&&!state.creates&&!model.backend_state);assert(bn_model_prepare_native_core(&model,&gpu,peak,host-1)==-1&&!state.creates&&!model.backend_state);
    BnModelNativeAnatomy *bad=malloc(sizeof(*bad));assert(bad);
    for(int variant=0;variant<7;variant++) {
        *bad=*a;
        switch(variant){case 0:bad->core_projection_count--;break;case 1:bad->core_projections[329]=bad->core_projections[0];break;case 2:bad->core_projections[329].rows++;break;case 3:bad->core_projections[329].scale_tensor=bad->core_projections[329].weight_tensor;break;case 4:bad->core_projections[329].layer=128;break;case 5:bad->core_format=BN_QUANT_FORMAT_EXL3_MUL1_3;break;default:bad->core_projections[329].weight_role=BN_NATIVE_ROLE_COUNT;break;}
        BnBackendModel *backend=bn_backend_model_create();assert(backend);assert(bn_backend_layout_prepare_native_core(backend,&gpu,&index,bad,peak,host)==-1&&!state.creates&&!bn_backend_model_gpu(backend)&&!bn_backend_layout_native_core_stats(backend));bn_backend_model_free(backend);
    }free(bad);
    const int failures[]={1,100,330};for(int i=0;i<3;i++) {
        state.creates=0;state.fail_at=failures[i];assert(bn_model_prepare_native_core(&model,&gpu,peak,host)==-1&&!model.backend_state&&!state.active&&state.creates==failures[i]&&bn_model_native_anatomy(&model)==a&&bn_model_native_numerics(&model));
    }
    state.fail_at=0;state.creates=0;state.peak=0;
    BnBackendModel *blocked=bn_backend_model_create();assert(blocked);int released=0;assert(!bn_backend_model_attach_gpu_resources(blocked,&released,&released,native_core_foreign_release));assert(bn_backend_layout_prepare_native_core(blocked,&gpu,&index,a,peak,host)==-1&&!state.active&&!released&&bn_backend_model_gpu_resources(blocked,&released)==&released);bn_backend_model_free(blocked);assert(released==1);
    state.creates=0;state.peak=0;assert(!bn_model_prepare_native_core(&model,&gpu,peak,host));BnBackendModel *backend=bn_model_backend(&model);const BnBackendNativeCoreStats *owned=bn_backend_layout_native_core_stats(backend);assert(owned&&owned->count==330&&owned->resident_bytes==bytes&&state.active==bytes&&state.peak==peak&&state.creates==330&&bn_backend_model_gpu(backend)==&gpu);
    int creates=state.creates;assert(bn_model_prepare_native_core(&model,&gpu,peak,host)==-1&&state.creates==creates&&owned==bn_backend_layout_native_core_stats(backend));assert(!bn_backend_layout_native_core_weight(backend,-1,0)&&!bn_backend_layout_native_core_weight(backend,40,0)&&!bn_backend_layout_native_core_weight(backend,0,BN_NATIVE_ROLE_COUNT));
    for(int l=0;l<40;l++)for(int role=0;role<BN_NATIVE_ROLE_COUNT;role++) {
        const BnBackendNativeF16Weight *w=bn_backend_layout_native_core_weight(backend,l,role);assert((w!=NULL)==(seen[l][role]!=0));if(w){NativeF16FakeWeight *fake=w->weight;assert(w->gpu==&gpu&&fake&&fake->bytes==w->bytes);for(size_t j=0;j<fake->bytes/2;j++)assert(fake->packed[j]==0);}
    }
    bn_model_free(&model);assert(!state.active);bn_safetensors_index_free(&index);free(config);snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!remove(path));snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!remove(path)&&!rmdir(dir));puts("Native core global preflight, exact budget, staged rollback and model ownership PASSED");
}
#else
static void test_native_core(void) {}
#endif
