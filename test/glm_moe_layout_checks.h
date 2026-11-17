/* Budgeted preparation of the complete GPU routing/shared MoE resource. */
typedef struct {int calls,fail,freed;} MoELayoutProbe;
static void*moe_layout_upload(void*ctx,const void*data,size_t bytes,int type,int rows,int cols) {
    MoELayoutProbe*p=ctx;assert(data&&bytes&&rows>0&&cols>0);
    assert(type==BN_GGUF_TENSOR_F32||type==BN_GGUF_TENSOR_Q8_0);
    if(++p->calls==p->fail)return NULL;
    void*out=malloc(bytes);assert(out);memcpy(out,data,bytes);return out;
}
static void moe_layout_destroy(void*ctx,void*ptr){MoELayoutProbe*p=ctx;assert(ptr);p->freed++;free(ptr);}
static void test_moe_checkpoint_preparation(void) {
    for(int profile=0;profile<4;++profile){
    int gate_type=profile&1?BN_GGUF_TENSOR_Q5_K:BN_GGUF_TENSOR_Q4_K;
    int down_type=profile&2?BN_GGUF_TENSOR_Q6_K:BN_GGUF_TENSOR_Q5_K;
    BnModelAnatomy*a=calloc(1,sizeof(*a));assert(a);a->n_layers=1;a->dim=a->expert_hidden=a->shared_hidden=256;
    a->experts=288;a->active_experts=8;a->shared_experts=1;a->normalize_expert_weights=1;a->expert_weights_scale=2.5f;
    a->layers[0].kind=BN_MODEL_ANATOMY_INDEXED_MLA;a->layers[0].uses_moe=1;
    a->layers[0].swiglu_clamp_exp=10;a->layers[0].swiglu_clamp_shexp=1.5f;
    for(int j=0;j<BN_MODEL_TENSOR_ROLE_COUNT;++j)a->layers[0].tensor[j]=-1;
    BnModelTensorRole roles[]={BN_MODEL_TENSOR_MOE_ROUTER,BN_MODEL_TENSOR_MOE_ROUTER_BIAS,
        BN_MODEL_TENSOR_SHARED_FFN_GATE,BN_MODEL_TENSOR_SHARED_FFN_UP,BN_MODEL_TENSOR_SHARED_FFN_DOWN,
        BN_MODEL_TENSOR_MOE_GATE_EXPS,BN_MODEL_TENSOR_MOE_UP_EXPS,BN_MODEL_TENSOR_MOE_DOWN_EXPS};
    BnGGUFTensorInfo ts[8]={0};size_t sizes[8],total=0,gpu_bytes=0;
    for(int j=0;j<8;++j){a->layers[0].tensor[roles[j]]=j;ts[j].type=j<2?BN_GGUF_TENSOR_F32:j<5?BN_GGUF_TENSOR_Q8_0:j<7?gate_type:down_type;
        ts[j].n_dims=j==1?1:j<5?2:3;ts[j].dims[0]=j==1?288:256;ts[j].dims[1]=j==0?288:256;ts[j].dims[2]=288;
        uint64_t count=ts[j].dims[0]*(j==1?1:ts[j].dims[1])*(j>=5?288:1);assert(bn_gguf_tensor_size(ts[j].type,count,&sizes[j]));ts[j].offset=total;total+=sizes[j];if(j<5)gpu_bytes+=sizes[j];}
    unsigned char*raw=calloc(1,total);assert(raw);float first=1.00390625f;memcpy(raw,&first,4);
    BnGGUFFile f={.raw=raw,.raw_size=total,.tensors=ts,.n_tensors=8};
    size_t gb=sizes[5]/288,db=sizes[7]/288,host=sizes[2]+sizes[3];
    BnMoEExpertMap map={.gate_type=gate_type,.up_type=gate_type,.down_type=down_type,
        .gate_rows=256,.up_rows=256,.down_rows=256,.gate_cols=256,.up_cols=256,.down_cols=256,
        .expert_gate_bytes=gb,.expert_up_bytes=gb,.expert_down_bytes=db,.gate_stride=gb,.up_stride=gb,.down_stride=db,
        .gate_offset=ts[5].offset,.up_offset=ts[6].offset,.down_offset=ts[7].offset};
    MoELayoutProbe probe={0};BnGPUBackend g={.ctx=&probe,.buffer_create_quant_only=moe_layout_upload,.buffer_destroy=moe_layout_destroy};BnBackendBF16MoEWeights w={0};
    assert(!bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&map,gpu_bytes,host));
    assert(w.bytes==gpu_bytes&&w.host_staging_bytes==host&&w.source_map==&map&&w.layer==0);
    assert(w.clamp_limit==10&&w.shared.clamp_limit==1.5f&&probe.calls==4&&!memcmp(w.router,raw,sizes[0]));
    assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&map,gpu_bytes,host)==-1);
    bn_backend_layout_bf16_moe_free(&w);bn_backend_layout_bf16_moe_free(&w);assert(probe.freed==4);
    for(int fail=1;fail<=4;++fail){probe=(MoELayoutProbe){.fail=fail};unsigned char before[sizeof(w)];memcpy(before,&w,sizeof(w));
        assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&map,gpu_bytes,host)==-1);assert(probe.calls==fail&&probe.freed==fail-1&&!memcmp(before,&w,sizeof(w)));}
    probe=(MoELayoutProbe){0};
    assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&map,gpu_bytes-1,host)==-1);
    assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&map,gpu_bytes,host-1)==-1);
    a->normalize_expert_weights=0;assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&map,gpu_bytes,host)==-1);a->normalize_expert_weights=1;
    BnMoEExpertMap bad=map;bad.gate_offset++;assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&bad,gpu_bytes,host)==-1);
    bad=map;bad.up_type=BN_GGUF_TENSOR_Q8_0;assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&bad,gpu_bytes,host)==-1);
    bad=map;bad.down_type=BN_GGUF_TENSOR_Q8_0;assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&bad,gpu_bytes,host)==-1);
    bad=map;bad.expert_gate_bytes--;assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&bad,gpu_bytes,host)==-1);
    int saved_type=ts[6].type;ts[6].type=BN_GGUF_TENSOR_Q8_0;
    assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&map,gpu_bytes,host)==-1);ts[6].type=saved_type;
    bad=map;bad.down_stride--;assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&bad,gpu_bytes,host)==-1);
    bad=map;bad.up_shard_idx=1;assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&bad,gpu_bytes,host)==-1);
    float invalid=NAN;memcpy(raw,&invalid,4);assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&map,gpu_bytes,host)==-1);memcpy(raw,&first,4);
    invalid=INFINITY;memcpy(raw+ts[1].offset,&invalid,4);assert(bn_backend_layout_bf16_moe_init(&w,&g,&f,a,0,&map,gpu_bytes,host)==-1);memset(raw+ts[1].offset,0,4);
    assert(!probe.calls&&!w.gpu);free(raw);free(a);}
    puts("  MoE F32 router/shared preparation, metadata, budgets and upload rollback PASSED");
}
