/* Synthetic ownership, budgets, rollback and submission validation. */
typedef struct { int calls, freed, fail; } DenseFFNProbe;
static void *dense_ffn_upload(void *ctx,const void *data,size_t bytes,int type,int rows,int cols) {
    DenseFFNProbe*p=ctx;assert(data&&bytes&&rows>0&&cols>0&&type==BN_GGUF_TENSOR_Q8_0);
    if(++p->calls==p->fail)return NULL;
    void*b=malloc(bytes);assert(b);memcpy(b,data,bytes);return b;
}
static void dense_ffn_destroy(void*ctx,void*b){DenseFFNProbe*p=ctx;assert(b);p->freed++;free(b);}
static void test_dense_ffn_preparation(void) {
    BnModelAnatomy*a=calloc(1,sizeof(*a));assert(a);a->n_layers=1;a->dim=32;a->dense_hidden=64;
    a->layers[0].kind=BN_MODEL_ANATOMY_KDA;a->layers[0].swiglu_clamp_exp=1.234567f;
    for(int i=0;i<BN_MODEL_TENSOR_ROLE_COUNT;++i)a->layers[0].tensor[i]=-1;
    BnModelTensorRole roles[]={BN_MODEL_TENSOR_FFN_GATE,BN_MODEL_TENSOR_FFN_UP,BN_MODEL_TENSOR_FFN_DOWN};
    size_t matrix=32u*64/32*34,bytes=3*matrix,host=2*matrix;
    unsigned char*raw=malloc(bytes),*saved=malloc(bytes);assert(raw&&saved);
    for(size_t i=0;i<bytes;++i)raw[i]=(unsigned char)(i*37+3);
    memcpy(saved,raw,bytes);BnGGUFTensorInfo ts[3]={0};
    for(int i=0;i<3;++i){a->layers[0].tensor[roles[i]]=i;ts[i].type=BN_GGUF_TENSOR_Q8_0;ts[i].n_dims=2;ts[i].dims[0]=i==2?64:32;ts[i].dims[1]=i==2?32:64;ts[i].offset=i*matrix;}
    BnGGUFFile f={.tensors=ts,.n_tensors=3,.raw=raw,.raw_size=bytes};
    DenseFFNProbe probe={0};BnGPUBackend g={.ctx=&probe,.buffer_create_quant_only=dense_ffn_upload,.buffer_destroy=dense_ffn_destroy};
    BnBackendBF16DenseFFNWeights w={0};
    assert(!bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes,host));
    assert(probe.calls==2&&w.bytes==bytes&&w.host_staging_bytes==host&&w.dim==32&&w.hidden_dim==64&&w.clamp_limit==1.234567f);
    assert(!memcmp(w.gate_up.raw,raw,host)&&!memcmp(w.down.raw,raw+host,matrix));
    assert(bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);
    bn_backend_layout_bf16_dense_ffn_free(&w);bn_backend_layout_bf16_dense_ffn_free(&w);assert(probe.freed==2);
    for(int fail=1;fail<=2;++fail){probe=(DenseFFNProbe){.fail=fail};BnBackendBF16DenseFFNWeights before=w;
        assert(bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);
        assert(probe.calls==fail&&probe.freed==fail-1&&!memcmp(&w,&before,sizeof(w)));}
    probe=(DenseFFNProbe){0};
    assert(bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes-1,host)==-1);
    assert(bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes,host-1)==-1);
    a->layers[0].uses_moe=1;assert(bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);a->layers[0].uses_moe=0;
    float limits[]={-1,INFINITY,NAN};for(int i=0;i<3;++i){a->layers[0].swiglu_clamp_exp=limits[i];assert(bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);}a->layers[0].swiglu_clamp_exp=0;
    for(int i=0;i<3;++i){uint64_t d=ts[i].dims[0];ts[i].dims[0]++;assert(bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);ts[i].dims[0]=d;
        uint64_t off=ts[i].offset;ts[i].offset=bytes;assert(bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);ts[i].offset=off;
        ts[i].type=BN_GGUF_TENSOR_Q4_0;assert(bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);ts[i].type=BN_GGUF_TENSOR_Q8_0;}
    assert(!probe.calls&&!memcmp(raw,saved,bytes));
    assert(!bn_backend_layout_bf16_dense_ffn_init(&w,&g,&f,a,0,bytes,host));bn_backend_layout_bf16_dense_ffn_free(&w);
    a->layers[0].uses_moe=1;a->shared_hidden=64;a->shared_experts=1;a->layers[0].swiglu_clamp_shexp=20.1f;
    BnModelTensorRole shared[]={BN_MODEL_TENSOR_SHARED_FFN_GATE,BN_MODEL_TENSOR_SHARED_FFN_UP,BN_MODEL_TENSOR_SHARED_FFN_DOWN};
    for(int j=0;j<3;++j){a->layers[0].tensor[roles[j]]=-1;a->layers[0].tensor[shared[j]]=j;}
    probe=(DenseFFNProbe){0};
    assert(!bn_backend_layout_bf16_shared_ffn_init(&w,&g,&f,a,0,bytes,host));
    assert(w.hidden_dim==64&&w.clamp_limit==20.1f&&w.bytes==bytes&&!memcmp(w.gate_up.raw,raw,host));bn_backend_layout_bf16_dense_ffn_free(&w);
    for(int fail=1;fail<=2;++fail){probe=(DenseFFNProbe){.fail=fail};assert(bn_backend_layout_bf16_shared_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);assert(probe.freed==fail-1&&!w.gpu);}
    probe=(DenseFFNProbe){0};
    assert(bn_backend_layout_bf16_shared_ffn_init(&w,&g,&f,a,0,bytes-1,host)==-1);
    assert(bn_backend_layout_bf16_shared_ffn_init(&w,&g,&f,a,0,bytes,host-1)==-1);
    a->shared_experts=2;assert(bn_backend_layout_bf16_shared_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);a->shared_experts=1;
    a->layers[0].swiglu_clamp_shexp=NAN;assert(bn_backend_layout_bf16_shared_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);a->layers[0].swiglu_clamp_shexp=0;
    a->layers[0].uses_moe=0;assert(bn_backend_layout_bf16_shared_ffn_init(&w,&g,&f,a,0,bytes,host)==-1);
    assert(!probe.calls&&!memcmp(raw,saved,bytes));
    free(raw);free(saved);free(a);puts("  dense FFN checkpoint order, budgets and preparation failures PASSED");
}
static int dense_ffn_linear_probe(void*ctx,void*r,const BnGPUBF16LinearResidentPlan*p){DenseFFNProbe*q=ctx;assert(r==ctx&&p);assert(q->calls==0||q->calls==2);return ++q->calls==q->fail?-1:0;}
static int dense_ffn_activation_probe(void*ctx,void*r,const BnGPUClampedSwiGLUResidentPlan*p){DenseFFNProbe*q=ctx;assert(r==ctx&&p&&q->calls==1);return ++q->calls==q->fail?-1:0;}
static void test_dense_ffn_orchestration(void) {
    DenseFFNProbe probe={0};BnGPUBackend g={.ctx=&probe,.linear_a8_bf16_resident=dense_ffn_linear_probe,.clamped_swiglu_bf16_resident=dense_ffn_activation_probe};
    BnBackendBF16DenseFFNWeights w={.gpu=&g,.dim=32,.hidden_dim=64,.clamp_limit=10};w.gate_up=w.down=(BnGPUBF16LinearWeight){.raw=&g,.type=BN_GGUF_TENSOR_Q8_0};
    BnGPUResidentView v[4];size_t counts[]={96,384,192,96};for(int i=0;i<4;++i)v[i]=(BnGPUResidentView){.request=&probe,.slot=(uint32_t)i,.type=BN_GGUF_TENSOR_BF16,.count=counts[i]};
    for(int fail=0;fail<=3;++fail){probe=(DenseFFNProbe){.fail=fail};assert(bn_transformer_bf16_dense_ffn(&w,&probe,3,v[0],v[1],v[2],v[3])==(fail?-1:0));assert(probe.calls==(fail?fail:3));}
    probe=(DenseFFNProbe){0};
    for(int i=0;i<4;++i){BnGPUResidentView save=v[i];v[i].count--;assert(bn_transformer_bf16_dense_ffn(&w,&probe,3,v[0],v[1],v[2],v[3])==-1);v[i]=save;
        v[i].request=&g;assert(bn_transformer_bf16_dense_ffn(&w,&probe,3,v[0],v[1],v[2],v[3])==-1);v[i]=save;
        v[i].type=BN_GGUF_TENSOR_F32;assert(bn_transformer_bf16_dense_ffn(&w,&probe,3,v[0],v[1],v[2],v[3])==-1);v[i]=save;
        v[i].offset=SIZE_MAX;assert(bn_transformer_bf16_dense_ffn(&w,&probe,3,v[0],v[1],v[2],v[3])==-1);v[i]=save;}
    v[3]=v[0];assert(bn_transformer_bf16_dense_ffn(&w,&probe,3,v[0],v[1],v[2],v[3])==-1);v[3].slot=3;
    w.clamp_limit=NAN;assert(bn_transformer_bf16_dense_ffn(&w,&probe,3,v[0],v[1],v[2],v[3])==-1);assert(!probe.calls);
    puts("  dense FFN preflight and execution failure propagation PASSED");
}
