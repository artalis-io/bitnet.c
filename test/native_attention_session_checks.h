#ifndef __EMSCRIPTEN__
typedef struct {
    NativeIndexFake index;
    int phase,attn,project,ring,fail_stage,warm,rows,pos,entries,dense;
    BnGPUResidentView group,kv;
} NativeAttentionFake;
static int native_attn_math(void*c,void*r,const BnGPUF16LatentAttentionResidentPlan*p) {
    NativeAttentionFake*t=c;assert(!t->phase&&p->queries.request==r&&p->kv.request==r&&p->out.request==r);
    assert(p->n_tokens==t->rows&&p->pos0==t->pos&&p->pool_entries==t->entries&&p->dense_pool==t->dense);
    assert(p->window==128&&p->win_floor<=p->pos0&&p->sinks&&p->inv_freq.count==32);
    assert(p->ring_beg<=p->pos0&&p->ring_rows==768);
    if(p->pool_entries){assert(p->pages>0&&p->entries_per_page>0&&p->pool_c.request==r);}
    else if(!p->pages)assert(!p->pool_c.request&&!p->block_table.request);
    assert((p->dense_pool&&!p->indices.request)||(!p->dense_pool&&p->indices.count==(size_t)p->n_tokens*512));
    BnGPUF16LatentAttentionWorkspace w;assert(!bn_gpu_latent_attention_workspace(p->n_tokens,p->pool_entries,p->rate,p->dense_pool,p->window,p->pos0,&w));assert(p->scratch_ml.count==w.ml_count&&p->scratch_acc.count==w.acc_count);
    t->group=p->out;t->kv=p->kv;t->phase=1;t->attn++;return t->fail_stage==1?-1:0;
}
static int native_attn_project(void*c,void*r,const BnGPUF16GroupedOutputResidentPlan*p) {
    NativeAttentionFake*t=c;assert(t->phase==1&&native_idx_same(p->x,t->group)&&p->groups==8&&p->k==4096&&p->n==128&&p->n_tokens==t->rows);
    assert(p->down.x.request==r&&p->down.k==1024&&p->down.n==128&&p->down.out.type==BN_GGUF_TENSOR_F32);
    t->phase=2;t->project++;return t->fail_stage==2?-1:0;
}
static int native_attn_ring(void*c,void*r,const BnGPUSWARingUpdateResidentPlan*p) {
    NativeAttentionFake*t=c;if(t->warm){assert(p->kv.request==r);return 0;}assert(t->phase==2&&native_idx_same(p->kv,t->kv)&&p->ring.request==r&&p->pos0==t->pos&&p->n_tokens==t->rows);
    t->phase=3;t->ring++;return t->fail_stage==3?-1:0;
}
static int native_attn_sync(void*r) {
    NativeAttentionFake*t=(void*)((NativeExecRequest*)r)->owner;
    if(t->phase){assert(t->phase==3||t->fail_stage);t->phase=0;}
    return native_idx_sync(r);
}
static void native_attn_input(BnGPUBackend*g,BnSession*s,BnModel*m,int rows,BnGPUResidentView*v) {
    void*r;BnGPUResidentView unused;assert(!bn_transformer_trellis_session_input(m,s,rows,&r,&unused));
    size_t n[]={(size_t)rows*64*512,(size_t)rows*512,32};for(int j=0;j<3;j++)assert(!g->resident_alloc(r,j==2?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_F16,n[j],v+j));
}
static void native_attn_publish(BnModel*m,BnSession*s,BnGPUBackend*g,int layer,int pos,int rows) {
    BnTransformerTrellisKVViews p;assert(!bn_transformer_trellis_kv_views(m,s,layer,&p));int n=(pos+rows)/p.rate-pos/p.rate;BnGPUResidentView lat={0},key={0};
    if(n)assert(!g->resident_alloc(p.request,BN_GGUF_TENSOR_F16,(size_t)n*512,&lat)&&!g->resident_alloc(p.request,BN_GGUF_TENSOR_F16,(size_t)n*128,&key));
    assert(!bn_transformer_trellis_kv_publish(m,s,layer,pos,rows,lat,key));
}
static void test_native_attention_session(void) {
    char dir[]="/tmp/bn-native-attention-session-XXXXXX",path[256];assert(mkdtemp(dir));native_anatomy_fixture_geometry(dir,0,32,128,64,512,8);
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    const char*from[]={"\"engram_compressed_vocab_size\":99092","\"index_n_heads\":2","\"index_head_dim\":32","\"num_attention_heads\":8","\"head_dim\":128","\"o_groups\":2","\"qk_rope_head_dim\":32"};
    const char*to[]={"\"engram_compressed_vocab_size\":256","\"index_n_heads\":32","\"index_head_dim\":128","\"num_attention_heads\":64","\"head_dim\":512","\"o_groups\":8","\"qk_rope_head_dim\":64"};
    char*config=strdup(native_anatomy_config);assert(config);for(int j=0;j<7;j++){char*z=native_numerics_replace(config,from[j],to[j]);free(config);config=z;}
    BnModel m={0};assert(!bn_model_load_trellis_config(&m,&index,config,strlen(config)));assert(!bn_model_prepare_native_anatomy(&m));assert(!bn_model_prepare_native_numerics(&m));free(config);
    NativeAttentionFake t={0};t.index.base.owner.model=&m;BnGPUBackend g=native_idx_gpu(&t.index);g.latent_attention_f16_resident=native_attn_math;g.grouped_output_f16_resident=native_attn_project;g.swa_ring_update_f16_resident=native_attn_ring;g.resident_synchronize=native_attn_sync;
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(&m);BnBackendNativeCoreStats cs;BnBackendNativeParameterStats ps;
    assert(!bn_backend_layout_native_core_plan(&index,a,&cs)&&!bn_backend_layout_native_parameter_plan(&index,a,&ps)&&!bn_model_prepare_native_core(&m,&g,cs.device_peak_bytes,cs.host_staging_bytes)&&!bn_model_prepare_native_parameters(&m,ps.device_peak_bytes,ps.host_staging_bytes));
    const BnMoEExpertReader*reader=bn_model_trellis_expert_reader(&m);size_t entry=reader->bytes[0]+reader->bytes[1]+reader->bytes[2];assert(!bn_gpu_moe_trellis_prepare_model(&m,entry,entry,entry,entry));
    BnTransformerTrellisOptions ffn={273,1,0,0,0};BnTransformerTrellisKVOptions kv={32768,0,0};BnTransformerTrellisWorkspace fw;BnTransformerTrellisKVWorkspace kw;BnTransformerTrellisSWAWorkspace sw;BnTransformerTrellisIndexWorkspace iw;BnTransformerTrellisAttentionWorkspace aw,keep;
    memset(&keep,0xa5,sizeof(keep));aw=keep;g.grouped_output_f16_resident=NULL;assert(bn_transformer_trellis_attention_session_plan(&m,&ffn,&kv,&aw)==-1&&!memcmp(&aw,&keep,sizeof(aw)));g.grouped_output_f16_resident=native_attn_project;
    assert(!bn_transformer_trellis_session_plan(&m,&ffn,&fw)&&!bn_transformer_trellis_kv_session_plan(&m,&kv,&kw)&&!bn_transformer_trellis_swa_session_plan(&m,&sw)&&!bn_transformer_trellis_index_session_plan(&m,&ffn,&kv,&iw)&&!bn_transformer_trellis_attention_session_plan(&m,&ffn,&kv,&aw));
    assert(aw.max_tokens==273&&aw.ml_count==8*64*16*2&&aw.acc_count==8*64*16*512);
    assert(aw.device_bytes==(size_t)273*(32768*2+1024*2+128*4)+(aw.ml_count+aw.acc_count)*4);
    ffn.device_budget=fw.device_bytes;ffn.host_budget=fw.host_bytes;kv.device_budget=kw.device_bytes;kv.host_budget=kw.host_bytes;
    BnTransformerTrellisSWAOptions so={sw.device_bytes,sw.host_bytes};BnTransformerTrellisIndexOptions io={iw.device_bytes,iw.host_bytes};BnTransformerTrellisAttentionOptions ao={aw.device_bytes,aw.host_bytes};
    BnSession*base=bn_session_create_backend(&m,NULL);assert(base);int before=t.index.base.allocations;
    assert(!bn_transformer_trellis_prepare_session_index(&m,base,&ffn,&kv,&so,&io));int base_alloc=t.index.base.allocations-before;bn_session_free(base,NULL);
    for(int j=1;j<=5;j++) {
        BnSession*z=bn_session_create_backend(&m,NULL);assert(z);int destroyed=t.index.base.destroyed;t.index.base.fail_at=t.index.base.allocations+base_alloc+j;
        assert(bn_transformer_trellis_prepare_session_projected_attention(&m,z,&ffn,&kv,&so,&io,&ao)==-1&&t.index.base.destroyed==destroyed+1&&!bn_backend_session_has_gpu_resources(bn_session_backend(z)));
        t.index.base.fail_at=0;bn_session_free(z,NULL);
    }
    {BnSession*z=bn_session_create_backend(&m,NULL);assert(z);t.index.fail_sync=4;int destroyed=t.index.base.destroyed;
        assert(bn_transformer_trellis_prepare_session_projected_attention(&m,z,&ffn,&kv,&so,&io,&ao)==-1&&t.index.base.destroyed==destroyed+1);bn_session_free(z,NULL);}
    BnSession*s[2];for(int j=0;j<2;j++) {
        s[j]=bn_session_create_backend(&m,NULL);assert(s[j]);int created=t.index.base.created;BnTransformerTrellisAttentionOptions bad=ao;bad.device_budget--;assert(bn_transformer_trellis_prepare_session_projected_attention(&m,s[j],&ffn,&kv,&so,&io,&bad)==-1&&t.index.base.created==created);bad=ao;bad.host_budget--;assert(bn_transformer_trellis_prepare_session_projected_attention(&m,s[j],&ffn,&kv,&so,&io,&bad)==-1&&t.index.base.created==created);
        assert(!bn_transformer_trellis_prepare_session_projected_attention(&m,s[j],&ffn,&kv,&so,&io,&ao));int alloc=t.index.base.allocations,sync=t.index.syncs;assert(!bn_transformer_trellis_prepare_session_projected_attention(&m,s[j],&ffn,&kv,&so,&io,&ao)&&t.index.base.allocations==alloc&&t.index.syncs==sync);
    }
    BnGPUResidentView v[2][3],empty={0};for(int j=0;j<2;j++)native_attn_input(&g,s[j],&m,273,v[j]);
    BnTransformerTrellisAttentionResult out,saved;memset(&saved,0xa5,sizeof(saved));out=saved;BnTransformerTrellisIndexResult selected;
    assert(!bn_transformer_trellis_index_begin(&m,s[0],0,273)&&!bn_transformer_trellis_index_select(&m,s[0],0,empty,empty,empty,&selected));
    BnGPUResidentView bad=v[0][0];bad.count--;assert(bn_transformer_trellis_attention_project(&m,s[0],0,&selected,bad,v[0][1],v[0][2],&out)==-1&&!memcmp(&out,&saved,sizeof(out))&&!t.attn);
    assert(bn_transformer_trellis_attention_project(&m,s[0],0,&selected,v[1][0],v[0][1],v[0][2],&out)==-1&&!t.attn);
    t.rows=273;t.pos=0;t.entries=0;t.dense=1;int alloc=t.index.base.allocations,sync=t.index.syncs;
    assert(!bn_transformer_trellis_attention_project(&m,s[0],0,&selected,v[0][0],v[0][1],v[0][2],&out));assert(t.attn==1&&t.project==1&&t.ring==1&&t.index.syncs==sync+1&&t.index.base.allocations==alloc&&out.delta.count==273*128&&out.tokens==273);
    BnTransformerTrellisSWAViews rv;assert(!bn_transformer_trellis_swa_views(&m,s[0],0,273,&rv)&&rv.cursor==273);
    assert(bn_transformer_trellis_attention_project(&m,s[0],0,&selected,v[0][0],v[0][1],v[0][2],&out)==-1&&t.attn==1);
    /* New epoch expires old selections, even identical-position replay. */
    assert(!bn_transformer_trellis_index_begin(&m,s[0],0,273));assert(bn_transformer_trellis_attention_project(&m,s[0],0,&selected,v[0][0],v[0][1],v[0][2],&out)==-1);
    assert(!bn_transformer_trellis_index_select(&m,s[0],0,empty,empty,empty,&selected)&&!bn_transformer_trellis_attention_project(&m,s[0],0,&selected,v[0][0],v[0][1],v[0][2],&out));
    /* Pool freshness required; borrowed dense owner metadata and PRE-update ring. */
    assert(bn_transformer_trellis_index_select(&m,s[0],2,empty,empty,empty,&selected)==-1);native_attn_publish(&m,s[0],&g,2,0,273);
    assert(!bn_transformer_trellis_index_select(&m,s[0],2,empty,empty,empty,&selected));t.entries=136;
    BnTransformerTrellisIndexResult wrong=selected;wrong.entries++;assert(bn_transformer_trellis_attention_project(&m,s[0],2,&wrong,v[0][0],v[0][1],v[0][2],&out)==-1);
    assert(!bn_transformer_trellis_attention_project(&m,s[0],2,&selected,v[0][0],v[0][1],v[0][2],&out));
    BnTransformerTrellisIndexResult old_dense=selected;native_attn_publish(&m,s[0],&g,2,17,256);
    assert(bn_transformer_trellis_attention_project(&m,s[0],3,&old_dense,v[0][0],v[0][1],v[0][2],&out)==-1);
    assert(!bn_transformer_trellis_index_select(&m,s[0],3,empty,empty,empty,&selected)&&!bn_transformer_trellis_attention_project(&m,s[0],3,&selected,v[0][0],v[0][1],v[0][2],&out));
    /* Decode uses split workspace and retains only completed ring state. */
    BnGPUResidentView d[3];native_attn_input(&g,s[0],&m,1,d);assert(!bn_transformer_trellis_index_begin(&m,s[0],273,1)&&!bn_transformer_trellis_index_select(&m,s[0],0,empty,empty,empty,&selected));t.rows=1;t.pos=273;t.entries=0;
    assert(!bn_transformer_trellis_attention_project(&m,s[0],0,&selected,d[0],d[1],d[2],&out)&&!bn_transformer_trellis_swa_views(&m,s[0],0,274,&rv)&&rv.cursor==274);
    /* Long-context owner20 candidate source, reader21 borrowing, then source24
     * candidate use. Each consumes the SAME owner20 pool after fresh publish. */
    const int layers[]={20,21,22,24};t.warm=1;
    for(int j=0;j<4;j++)for(int pos=0;pos<16727;) {
        int rows=16727-pos;if(rows>256)rows=256;
        BnGPUResidentView warm=bn_gpu_resident_view_slice(v[0][1],0,(size_t)rows*512);
        assert(!bn_transformer_trellis_swa_commit(&m,s[0],layers[j],pos,rows,warm));pos+=rows;
    }
    t.warm=0;
    for(int pos=0;pos<16727;){int rows=16727-pos;if(rows>256)rows=256;native_attn_publish(&m,s[0],&g,20,pos,rows);pos+=rows;}
    assert(!bn_transformer_trellis_index_begin(&m,s[0],16727,273));native_attn_publish(&m,s[0],&g,20,16727,273);
    void*request;BnGPUResidentView xi,qi,fi;assert(!bn_transformer_trellis_session_input(&m,s[0],273,&request,&xi));
    assert(!g.resident_alloc(request,BN_GGUF_TENSOR_F16,273*128,&qi)&&!g.resident_alloc(request,BN_GGUF_TENSOR_F32,32,&fi));
    t.rows=273;t.pos=16727;t.entries=17000;t.dense=0;t.index.nr=0;
    assert(!bn_transformer_trellis_index_select(&m,s[0],20,xi,qi,fi,&selected)&&selected.mode==BN_TRELLIS_INDEX_CANDIDATE_SOURCE);
    assert(!bn_transformer_trellis_attention_project(&m,s[0],20,&selected,v[0][0],v[0][1],v[0][2],&out));
    assert(!bn_transformer_trellis_index_select(&m,s[0],21,empty,empty,empty,&selected)&&selected.borrowed&&
        !bn_transformer_trellis_attention_project(&m,s[0],21,&selected,v[0][0],v[0][1],v[0][2],&out));BnTransformerTrellisIndexResult expired=selected;
    t.index.nr=0;assert(!bn_transformer_trellis_index_select(&m,s[0],24,xi,qi,fi,&selected)&&selected.mode==BN_TRELLIS_INDEX_CANDIDATE_USE);
    assert(bn_transformer_trellis_attention_project(&m,s[0],22,&expired,v[0][0],v[0][1],v[0][2],&out)==-1);
    expired=selected;t.index.nr=0;assert(!bn_transformer_trellis_index_select(&m,s[0],24,xi,qi,fi,&selected)&&selected.selection_revision>expired.selection_revision);
    assert(bn_transformer_trellis_attention_project(&m,s[0],24,&expired,v[0][0],v[0][1],v[0][2],&out)==-1);
    alloc=t.index.base.allocations;sync=t.index.syncs;assert(!bn_transformer_trellis_attention_project(&m,s[0],24,&selected,v[0][0],v[0][1],v[0][2],&out)&&t.index.base.allocations==alloc&&t.index.syncs==sync+1);
    assert(!bn_transformer_trellis_swa_views(&m,s[0],24,17000,&rv)&&rv.cursor==17000);
    /* Every backend failure invalidates request, leaves result untouched. */
    for(int stage=1;stage<=4;stage++) {
        BnSession*z=s[1];assert(!bn_transformer_trellis_index_begin(&m,z,0,273)&&!bn_transformer_trellis_index_select(&m,z,0,empty,empty,empty,&selected));t.rows=273;t.pos=0;t.entries=0;t.dense=1;t.phase=0;t.fail_stage=stage;out=saved;int destroyed=t.index.base.destroyed;if(stage==4)t.index.fail_sync=1;
        assert(bn_transformer_trellis_attention_project(&m,z,0,&selected,v[1][0],v[1][1],v[1][2],&out)==-1&&!memcmp(&out,&saved,sizeof(out))&&t.index.base.destroyed==destroyed+1);assert(bn_transformer_trellis_index_begin(&m,z,0,1)==-1);
        t.fail_stage=0;t.phase=0;bn_session_reset(z,&m);assert(!bn_transformer_trellis_prepare_session_projected_attention(&m,z,&ffn,&kv,&so,&io,&ao));native_attn_input(&g,z,&m,273,v[1]);
    }
    for(int j=0;j<2;j++){bn_session_free(s[j],NULL);}
    assert(t.index.base.created==t.index.base.destroyed);bn_model_free(&m);bn_safetensors_index_free(&index);snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!remove(path));snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!remove(path)&&!rmdir(dir));puts("Native attention session quotas/epochs/pre-update ordering/commit/failure PASSED");
}
#else
static void test_native_attention_session(void){}
#endif
