#ifndef __EMSCRIPTEN__
typedef struct {NativeAttentionFake attention;int qkv,compress,keys,publish,frequency,fail,owner,rate;BnGPUResidentView q,kv,x,lat,key;} NativeProducerFake;
static int native_prod_frequency(void*c,void*r,const BnGPUF32RopeFrequencyResidentPlan*p) {
    NativeProducerFake*t=c;t->frequency++;
    assert(p->rotary_dim==64&&p->out.request==r&&p->negative.request==r&&p->out.count==32&&p->negative.count==32&&p->out.type==BN_GGUF_TENSOR_F32);
    assert((!p->yarn&&p->base==10000&&p->factor==1&&p->ramp_start==0&&p->ramp_scale==1)||
        (p->yarn&&p->base==160000&&p->factor==16&&p->ramp_start==15&&p->ramp_scale==.1f));
    return t->fail==7?-1:0;
}
static int native_prod_qkv(void*c,void*r,const BnGPUF16LatentQKVResidentPlan*p) {
    NativeProducerFake*t=c;assert(p->q_a.x.request==r&&p->q_a.n==128&&p->q_b.n==32768&&p->kv.n==512&&p->q_norm.constant_bias==0&&p->q_norm.constant_scale==1&&p->q_norm.weight);
    assert(p->q_rope.pos0==t->attention.pos&&p->kv_rope.pos0==t->attention.pos&&p->q_rope.slice_offset==448&&p->kv_rope.rotate_offset==448&&p->kv_rope.norm&&!p->q_rope.norm);
    t->q=p->q_b.out;t->kv=p->kv.out;t->x=p->q_a.x;t->qkv++;return t->fail==1?-1:0;
}
static int native_prod_compress(void*c,void*r,const BnGPUF16CompressorResidentPlan*p) {
    NativeProducerFake*t=c;assert(t->owner&&p->kv.x.request==r&&native_idx_same(p->kv.x,t->x)&&p->compress.pos0==t->attention.pos&&p->compress.rate==t->rate&&p->compress.relative==1&&p->compress.split_dim==512&&p->compress.ring_rows==256+t->rate&&p->compress.norm.count==512&&p->compress.ape.count==(size_t)t->rate*512&&p->compress.inv_freq.count==32);
    if(t->rate==1)assert(!p->gate.weight&&native_idx_same(p->compress.kv,p->compress.gate));else assert(p->gate.weight&&native_idx_same(p->gate.x,t->x));
    t->lat=p->compress.dest_a;t->compress++;return t->fail==2?-1:0;
}
static int native_prod_keys(void*c,void*r,const BnGPUF16LatentKeysResidentPlan*p) {
    NativeProducerFake*t=c;assert(t->owner&&p->project.x.request==r&&native_idx_same(p->project.x,t->lat)&&p->entry0==t->attention.pos/t->rate&&p->rate==t->rate&&p->project.n==128&&p->project.k==512&&p->norm.weight&&p->norm.constant_bias==0&&p->latents_rope.slice_offset==448&&p->keys_rope.slice_offset==64&&p->keys_rope.positions.count==(size_t)p->project.n_tokens);
    t->key=p->norm.out;t->keys++;return t->fail==3?-1:0;
}
static int native_prod_publish(void*c,void*r,const BnGPUPagedLatentPublishResidentPlan*p) {
    NativeProducerFake*t=c;assert(native_idx_same(p->latents,t->lat)&&native_idx_same(p->keys,t->key));t->publish++;return t->fail==4?-1:native_idx_publish(c,r,p);
}
static int native_prod_attn(void*c,void*r,const BnGPUF16LatentAttentionResidentPlan*p) {
    NativeProducerFake*t=c;assert(native_idx_same(p->queries,t->q)&&native_idx_same(p->kv,t->kv));return native_attn_math(c,r,p);
}
static void test_native_producer(void) {
    char dir[]="/tmp/bn-native-producer-XXXXXX",path[256];assert(mkdtemp(dir));native_anatomy_fixture_geometry(dir,0,32,128,64,512,8);
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    const char*from[]={"\"engram_compressed_vocab_size\":99092","\"index_n_heads\":2","\"index_head_dim\":32","\"num_attention_heads\":8","\"head_dim\":128","\"o_groups\":2","\"qk_rope_head_dim\":32"};
    const char*to[]={"\"engram_compressed_vocab_size\":256","\"index_n_heads\":32","\"index_head_dim\":128","\"num_attention_heads\":64","\"head_dim\":512","\"o_groups\":8","\"qk_rope_head_dim\":64"};char*config=strdup(native_anatomy_config);assert(config);for(int j=0;j<7;j++){char*z=native_numerics_replace(config,from[j],to[j]);free(config);config=z;}
    BnModel m={0};assert(!bn_model_load_trellis_config(&m,&index,config,strlen(config))&&!bn_model_prepare_native_anatomy(&m)&&!bn_model_prepare_native_numerics(&m));free(config);
    NativeProducerFake t={0};t.attention.index.base.owner.model=&m;BnGPUBackend g=native_idx_gpu(&t.attention.index);g.rope_frequency_f32_resident=native_prod_frequency;g.latent_qkv_f16_resident=native_prod_qkv;g.compressor_f16_resident=native_prod_compress;g.latent_keys_f16_resident=native_prod_keys;g.latent_attention_f16_resident=native_prod_attn;g.grouped_output_f16_resident=native_attn_project;g.swa_ring_update_f16_resident=native_attn_ring;g.resident_synchronize=native_attn_sync;g.paged_latent_publish_f16_resident=native_prod_publish;
    const BnModelNativeAnatomy*a=bn_model_native_anatomy(&m);BnBackendNativeCoreStats cs;BnBackendNativeParameterStats ps;assert(!bn_backend_layout_native_core_plan(&index,a,&cs)&&!bn_backend_layout_native_parameter_plan(&index,a,&ps)&&!bn_model_prepare_native_core(&m,&g,cs.device_peak_bytes,cs.host_staging_bytes)&&!bn_model_prepare_native_parameters(&m,ps.device_peak_bytes,ps.host_staging_bytes));
    const BnMoEExpertReader*reader=bn_model_trellis_expert_reader(&m);size_t entry=reader->bytes[0]+reader->bytes[1]+reader->bytes[2];assert(!bn_gpu_moe_trellis_prepare_model(&m,entry,entry,entry,entry));
    BnTransformerTrellisOptions ffn={273,1,0,0,0};BnTransformerTrellisKVOptions kv={1024,0,0};BnTransformerTrellisWorkspace fw;BnTransformerTrellisKVWorkspace kw;BnTransformerTrellisSWAWorkspace sw;BnTransformerTrellisIndexWorkspace iw;BnTransformerTrellisAttentionWorkspace aw;BnTransformerTrellisProducerWorkspace pw,saved;memset(&saved,0xa5,sizeof(saved));pw=saved;g.latent_keys_f16_resident=NULL;assert(bn_transformer_trellis_producer_session_plan(&m,&ffn,&kv,&pw)==-1&&!memcmp(&pw,&saved,sizeof(pw)));g.latent_keys_f16_resident=native_prod_keys;
    assert(!bn_transformer_trellis_session_plan(&m,&ffn,&fw)&&!bn_transformer_trellis_kv_session_plan(&m,&kv,&kw)&&!bn_transformer_trellis_swa_session_plan(&m,&sw)&&!bn_transformer_trellis_index_session_plan(&m,&ffn,&kv,&iw)&&!bn_transformer_trellis_attention_session_plan(&m,&ffn,&kv,&aw)&&!bn_transformer_trellis_producer_session_plan(&m,&ffn,&kv,&pw)&&pw.max_tokens==273&&pw.max_rate==2&&pw.owners==4);
    ffn.device_budget=fw.device_bytes;ffn.host_budget=fw.host_bytes;kv.device_budget=kw.device_bytes;kv.host_budget=kw.host_bytes;BnTransformerTrellisSWAOptions so={sw.device_bytes,sw.host_bytes};BnTransformerTrellisIndexOptions io={iw.device_bytes,iw.host_bytes};BnTransformerTrellisAttentionOptions ao={aw.device_bytes,aw.host_bytes};BnTransformerTrellisProducerOptions po={pw.device_bytes,pw.host_bytes};
    BnSession*base=bn_session_create_backend(&m,NULL);assert(base);int count=t.attention.index.base.allocations;size_t bytes=t.attention.index.bytes;assert(!bn_transformer_trellis_prepare_session_projected_attention(&m,base,&ffn,&kv,&so,&io,&ao));int base_alloc=t.attention.index.base.allocations-count;size_t base_bytes=t.attention.index.bytes-bytes;
    BnTransformerTrellisBlockResult block,saved_block;memset(&block,0x5a,sizeof(block));saved_block=block;
    int before_block=t.attention.index.base.allocations;
    assert(bn_transformer_trellis_block_step(&m,base,0,0,&block)==-1&&!memcmp(&block,&saved_block,sizeof(block)));
    assert(!bn_transformer_trellis_block_current(&m,base,&block)&&t.attention.index.base.allocations==before_block);
    BnTransformerTrellisForwardResult forward,saved_forward;memset(&forward,0x5a,sizeof(forward));saved_forward=forward;
    assert(bn_transformer_trellis_forward_run(&m,base,0,&forward)==-1&&!memcmp(&forward,&saved_forward,sizeof(forward)));
    assert(!bn_transformer_trellis_forward_current(&m,base,&forward)&&t.attention.index.base.allocations==before_block);
    assert(bn_backend_session_has_gpu_resources(bn_session_backend(base)));bn_session_free(base,NULL);
    for(int j=1;j<=20;j++){BnSession*z=bn_session_create_backend(&m,NULL);assert(z);int destroyed=t.attention.index.base.destroyed;t.attention.index.base.fail_at=t.attention.index.base.allocations+base_alloc+j;assert(bn_transformer_trellis_prepare_session_producer(&m,z,&ffn,&kv,&so,&io,&ao,&po)==-1&&t.attention.index.base.destroyed==destroyed+1&&!bn_backend_session_has_gpu_resources(bn_session_backend(z)));t.attention.index.base.fail_at=0;bn_session_free(z,NULL);}
    {BnSession*z=bn_session_create_backend(&m,NULL);assert(z);t.fail=7;int destroyed=t.attention.index.base.destroyed;
        assert(bn_transformer_trellis_prepare_session_producer(&m,z,&ffn,&kv,&so,&io,&ao,&po)==-1&&t.attention.index.base.destroyed==destroyed+1);t.fail=0;bn_session_free(z,NULL);}
    BnSession*s[2];BnGPUResidentView input[2],freq[2];void*r[2];for(int j=0;j<2;j++) {
        s[j]=bn_session_create_backend(&m,NULL);assert(s[j]);BnTransformerTrellisProducerOptions bad=po;bad.device_budget--;int created=t.attention.index.base.created;assert(bn_transformer_trellis_prepare_session_producer(&m,s[j],&ffn,&kv,&so,&io,&ao,&bad)==-1&&t.attention.index.base.created==created);bad=po;bad.host_budget--;assert(bn_transformer_trellis_prepare_session_producer(&m,s[j],&ffn,&kv,&so,&io,&ao,&bad)==-1&&t.attention.index.base.created==created);
        count=t.attention.index.base.allocations;bytes=t.attention.index.bytes;assert(!bn_transformer_trellis_prepare_session_producer(&m,s[j],&ffn,&kv,&so,&io,&ao,&po)&&t.attention.index.base.allocations-count==base_alloc+20&&t.attention.index.bytes-bytes==base_bytes+pw.device_bytes);count=t.attention.index.base.allocations;assert(!bn_transformer_trellis_prepare_session_producer(&m,s[j],&ffn,&kv,&so,&io,&ao,&po)&&t.attention.index.base.allocations==count);
        assert(!bn_transformer_trellis_session_input(&m,s[j],273,r+j,input+j)&&!g.resident_alloc(r[j],BN_GGUF_TENSOR_F32,32,freq+j));
    }
    BnTransformerTrellisAttentionResult out,keep;memset(&keep,0xa5,sizeof(keep));out=keep;assert(!bn_transformer_trellis_index_begin(&m,s[0],0,273));t.attention.rows=273;t.attention.pos=0;t.attention.dense=1;t.attention.entries=0;
    BnGPUResidentView bad=input[0];bad.count--;assert(bn_transformer_trellis_attention_forward(&m,s[0],0,bad,freq[0],freq[0],freq[0],&out)==-1&&!t.qkv&&!memcmp(&out,&keep,sizeof(out)));
    assert(bn_transformer_trellis_attention_forward(&m,s[0],0,input[1],freq[0],freq[0],freq[0],&out)==-1&&!t.qkv);
    assert(!bn_transformer_trellis_attention_forward_native(&m,s[0],0,input[0],&out)&&t.qkv==1&&!t.compress&&!t.keys);
    int q=t.qkv;assert(bn_transformer_trellis_attention_forward(&m,s[0],3,input[0],freq[0],freq[0],freq[0],&out)==-1&&t.qkv==q);
    t.owner=1;t.rate=2;t.attention.entries=136;count=t.attention.index.base.allocations;int sync=t.attention.index.syncs;
    assert(!bn_transformer_trellis_attention_forward(&m,s[0],2,input[0],freq[0],freq[0],freq[0],&out)&&t.compress==1&&t.keys==1&&t.publish==1&&t.attention.index.base.allocations==count&&t.attention.index.syncs==sync+2);
    t.owner=0;assert(!bn_transformer_trellis_attention_forward(&m,s[0],3,input[0],freq[0],freq[0],freq[0],&out)&&t.compress==1);
    t.owner=1;t.rate=1;t.attention.entries=273;assert(!bn_transformer_trellis_attention_forward(&m,s[0],20,input[0],freq[0],freq[0],freq[0],&out)&&t.compress==2&&t.keys==2);
    t.owner=0;assert(!bn_transformer_trellis_attention_forward(&m,s[0],21,input[0],freq[0],freq[0],freq[0],&out));
    assert(!bn_transformer_trellis_index_begin(&m,s[0],273,1));BnGPUResidentView one=bn_gpu_resident_view_slice(input[0],0,128);t.attention.pos=273;t.attention.rows=1;t.attention.entries=137;t.owner=1;t.rate=2;
    assert(!bn_transformer_trellis_attention_forward(&m,s[0],2,one,freq[0],freq[0],freq[0],&out)&&t.keys==3);
    assert(!bn_transformer_trellis_index_begin(&m,s[0],274,1));t.attention.pos=274;
    assert(!bn_transformer_trellis_attention_forward(&m,s[0],2,one,freq[0],freq[0],freq[0],&out)&&t.keys==3);
    assert(!bn_transformer_trellis_index_begin(&m,s[0],273,273));t.attention.pos=273;t.attention.rows=273;t.attention.entries=546;t.attention.dense=0;t.owner=1;t.rate=1;t.attention.index.nr=0;
    int queries=t.attention.index.query;assert(!bn_transformer_trellis_attention_forward(&m,s[0],20,input[0],freq[0],freq[0],freq[0],&out)&&t.attention.index.query==queries+1&&t.attention.index.nr==2&&t.attention.index.rows[0]==256&&t.attention.index.rows[1]==17);
    t.owner=0;sync=t.attention.index.syncs;assert(!bn_transformer_trellis_attention_forward(&m,s[0],21,input[0],freq[0],freq[0],freq[0],&out)&&t.attention.index.syncs==sync+1);
    for(int fail=1;fail<=6;fail++) {
        BnSession*z=s[1];assert(!bn_transformer_trellis_index_begin(&m,z,0,273));t.attention.pos=0;t.attention.rows=273;t.attention.entries=136;t.attention.dense=1;t.owner=1;t.rate=2;t.fail=fail;
        if(fail==5){t.attention.fail_stage=1;}if(fail==6){t.attention.index.fail_sync=2;}out=keep;int destroyed=t.attention.index.base.destroyed;
        assert(bn_transformer_trellis_attention_forward(&m,z,2,input[1],freq[1],freq[1],freq[1],&out)==-1&&t.attention.index.base.destroyed==destroyed+1&&!memcmp(&out,&keep,sizeof(out)));t.fail=0;t.attention.fail_stage=0;t.attention.phase=0;bn_session_reset(z,&m);assert(!bn_transformer_trellis_prepare_session_producer(&m,z,&ffn,&kv,&so,&io,&ao,&po));assert(!bn_transformer_trellis_session_input(&m,z,273,r+1,input+1)&&!g.resident_alloc(r[1],BN_GGUF_TENSOR_F32,32,freq+1));
    }
    for(int j=0;j<2;j++){bn_session_free(s[j],NULL);}assert(t.attention.index.base.created==t.attention.index.base.destroyed);bn_model_free(&m);bn_safetensors_index_free(&index);snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!remove(path));snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!remove(path)&&!rmdir(dir));puts("Native QKV/compressor/key/attention producer quotas/epochs/zero-emission/failure PASSED");
}
#else
static void test_native_producer(void){}
#endif
