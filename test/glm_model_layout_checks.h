/* Synthetic model ownership/rollback checks. No model file or GPU required. */
typedef struct {int calls,fail,live,freed;} GLMModelProbe;
typedef struct {int type,rows,cols;size_t bytes;} GLMModelHandle;
static void *glm_model_handle(GLMModelProbe*p,int type,int rows,int cols,size_t bytes) {
    if(++p->calls==p->fail)return NULL;
    GLMModelHandle*h=malloc(sizeof(*h));assert(h);*h=(GLMModelHandle){type,rows,cols,bytes};p->live++;return h;
}
static void *glm_model_upload(void*ctx,const void*data,size_t bytes,int type,int rows,int cols) {
    assert(data&&bytes&&rows>0&&cols>0);return glm_model_handle(ctx,type,rows,cols,bytes);
}
static void *glm_model_decode(void*ctx,void*source) {
    GLMModelHandle*h=source;assert(h&&h->type==BN_GGUF_TENSOR_F32);
    return glm_model_handle(ctx,BN_GGUF_TENSOR_BF16,h->rows,h->cols,(size_t)h->rows*h->cols*2);
}
static void glm_model_destroy(void*ctx,void*ptr){GLMModelProbe*p=ctx;assert(ptr&&p->live>0);p->live--;p->freed++;free(ptr);}
static int glm_model_tensor(BnGGUFTensorInfo*t,int*n,int type,int nd,int d0,int d1,int d2,size_t*maximum) {
    int i=(*n)++;assert(i<64);t[i]=(BnGGUFTensorInfo){.type=(uint32_t)type,.n_dims=(uint32_t)nd,.dims={(uint64_t)d0,(uint64_t)d1,(uint64_t)d2}};
    size_t bytes;assert(bn_gguf_tensor_size(type,(uint64_t)d0*(nd>1?d1:1)*(nd>2?d2:1),&bytes));
    if(bytes>*maximum)*maximum=bytes;
    return i;
}
static void test_glm_model_preparation(void) {
    BnModelAnatomy*a=calloc(1,sizeof(*a));assert(a);a->n_layers=a->stored_layers=2;a->dim=4096;a->vocab_size=32;
    a->dense_hidden=a->expert_hidden=a->shared_hidden=256;a->experts=288;a->active_experts=8;a->shared_experts=1;
    a->normalize_expert_weights=1;a->expert_weights_scale=2.5f;a->kda_heads=1;a->kda_head_dim=a->kda_gate_rank=128;a->conv_kernel=4;
    a->kda_parameter_is_negative_exp=1;a->kda_qk_eps=1e-6f;a->kda_output_eps=1e-5f;a->kda_lower_bound=-5;
    a->mhc_streams=4;a->mhc_iterations=20;a->mhc_eps=1e-6f;a->norm_eps=1e-5f;
    for(int l=0;l<2;++l){a->layers[l].kind=BN_MODEL_ANATOMY_KDA;a->layers[l].uses_moe=l;
        a->layers[l].swiglu_clamp_exp=a->layers[l].swiglu_clamp_shexp=10;
        for(int j=0;j<BN_MODEL_TENSOR_ROLE_COUNT;++j)a->layers[l].tensor[j]=-1;}
    BnGGUFTensorInfo ts[64]={0};int nt=0;size_t maximum=0;
    BnModelTensorRole kr[]={BN_MODEL_TENSOR_ATTN_Q,BN_MODEL_TENSOR_ATTN_K,BN_MODEL_TENSOR_ATTN_V,
        BN_MODEL_TENSOR_SSM_BETA,BN_MODEL_TENSOR_SSM_F_A,BN_MODEL_TENSOR_SSM_G_A,BN_MODEL_TENSOR_SSM_F_B,
        BN_MODEL_TENSOR_SSM_G_B,BN_MODEL_TENSOR_SSM_OUT,BN_MODEL_TENSOR_SSM_CONV1D_Q,BN_MODEL_TENSOR_SSM_CONV1D_K,
        BN_MODEL_TENSOR_SSM_CONV1D_V,BN_MODEL_TENSOR_SSM_A,BN_MODEL_TENSOR_SSM_DT_BIAS,BN_MODEL_TENSOR_SSM_NORM};
    int rows[]={128,128,128,1,128,128,128,128,4096},cols[]={4096,4096,4096,4096,4096,4096,128,128,128};
    for(int i=0;i<15;++i){int type=i<9?BN_GGUF_TENSOR_Q8_0:BN_GGUF_TENSOR_F32;
        int index=i<9?glm_model_tensor(ts,&nt,type,2,cols[i],rows[i],1,&maximum):
            i<12?glm_model_tensor(ts,&nt,type,3,4,1,128,&maximum):glm_model_tensor(ts,&nt,type,1,i==12?1:128,1,1,&maximum);
        for(int l=0;l<2;++l)a->layers[l].tensor[kr[i]]=index;}
    BnModelTensorRole mr[]={BN_MODEL_TENSOR_MHC_ATTN_FN,BN_MODEL_TENSOR_MHC_ATTN_SCALE,BN_MODEL_TENSOR_MHC_ATTN_BASE,BN_MODEL_TENSOR_ATTN_NORM,
        BN_MODEL_TENSOR_MHC_FFN_FN,BN_MODEL_TENSOR_MHC_FFN_SCALE,BN_MODEL_TENSOR_MHC_FFN_BASE,BN_MODEL_TENSOR_FFN_NORM};
    for(int i=0;i<8;++i){int j=i%4,index=glm_model_tensor(ts,&nt,j?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_Q8_0,j?1:2,j==0?16384:j==1?3:j==2?24:4096,j?1:24,1,&maximum);
        for(int l=0;l<2;++l)a->layers[l].tensor[mr[i]]=index;}
    BnModelTensorRole dr[]={BN_MODEL_TENSOR_FFN_GATE,BN_MODEL_TENSOR_FFN_UP,BN_MODEL_TENSOR_FFN_DOWN};
    BnModelTensorRole sr[]={BN_MODEL_TENSOR_SHARED_FFN_GATE,BN_MODEL_TENSOR_SHARED_FFN_UP,BN_MODEL_TENSOR_SHARED_FFN_DOWN};
    for(int i=0;i<3;++i){int index=glm_model_tensor(ts,&nt,BN_GGUF_TENSOR_Q8_0,2,i==2?256:4096,i==2?4096:256,1,&maximum);
        a->layers[0].tensor[dr[i]]=a->layers[1].tensor[sr[i]]=index;}
    a->layers[1].tensor[BN_MODEL_TENSOR_MOE_ROUTER]=glm_model_tensor(ts,&nt,BN_GGUF_TENSOR_F32,2,4096,288,1,&maximum);
    a->layers[1].tensor[BN_MODEL_TENSOR_MOE_ROUTER_BIAS]=glm_model_tensor(ts,&nt,BN_GGUF_TENSOR_F32,1,288,1,1,&maximum);
    BnModelTensorRole er[]={BN_MODEL_TENSOR_MOE_GATE_EXPS,BN_MODEL_TENSOR_MOE_UP_EXPS,BN_MODEL_TENSOR_MOE_DOWN_EXPS};
    for(int i=0;i<3;++i)a->layers[1].tensor[er[i]]=glm_model_tensor(ts,&nt,i==2?BN_GGUF_TENSOR_Q5_K:BN_GGUF_TENSOR_Q4_K,3,i==2?256:4096,i==2?4096:256,288,&maximum);
    a->embedding_tensor=glm_model_tensor(ts,&nt,BN_GGUF_TENSOR_Q8_0,2,4096,32,1,&maximum);
    a->output_tensor=glm_model_tensor(ts,&nt,BN_GGUF_TENSOR_Q8_0,2,4096,32,1,&maximum);
    a->output_norm_tensor=glm_model_tensor(ts,&nt,BN_GGUF_TENSOR_F32,1,4096,1,1,&maximum);
    /* Tensor sources intentionally share valid zero-filled immutable storage. */
    unsigned char*raw=calloc(1,maximum);assert(raw);BnGGUFFile file={.raw=raw,.raw_size=maximum,.tensors=ts,.n_tensors=(uint64_t)nt};
    size_t gb,db;assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_Q4_K,4096u*256,&gb));assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_Q5_K,4096u*256,&db));
    BnMoEExpertMap maps[2]={{0},{.gate_type=BN_GGUF_TENSOR_Q4_K,.up_type=BN_GGUF_TENSOR_Q4_K,.down_type=BN_GGUF_TENSOR_Q5_K,
        .gate_rows=256,.up_rows=256,.down_rows=4096,.gate_cols=4096,.up_cols=4096,.down_cols=256,
        .expert_gate_bytes=gb,.expert_up_bytes=gb,.expert_down_bytes=db,.gate_stride=gb,.up_stride=gb,.down_stride=db}};
    GLMModelProbe q={0};BnGPUBackend gpu={.ctx=&q,.buffer_create_quant_only=glm_model_upload,.buffer_destroy=glm_model_destroy,.buffer_create_bf16_decoded=glm_model_decode};
    BnBackendModel*b=bn_backend_model_create();assert(b);
    assert(!bn_backend_layout_prepare_bf16_glm_model(b,&gpu,&file,a,maps,2,SIZE_MAX,SIZE_MAX));
    const BnBackendBF16GLMModel*m=bn_backend_layout_bf16_glm_model(b);assert(m&&m->n_layers==2&&m->anatomy!=a&&m->anatomy->dim==4096&&m->layers[1].kda.layer==1);
    assert(m->layers[1].moe.source_map==&m->layers[1].expert_map&&m->layers[1].moe.source_map!=&maps[1]);
    int calls=q.calls;size_t budget=m->peak_bytes,host=m->host_staging_bytes;assert(budget>=m->bytes&&host);
    a->dim=0;maps[1].gate_offset=123;assert(m->anatomy->dim==4096&&!m->layers[1].expert_map.gate_offset);a->dim=4096;maps[1].gate_offset=0;
    assert(bn_backend_layout_prepare_bf16_glm_model(b,&gpu,&file,a,maps,2,SIZE_MAX,SIZE_MAX)==-1&&q.calls==calls&&bn_backend_layout_bf16_glm_model(b)==m);
    bn_backend_model_release_gpu(b);assert(!q.live&&q.freed==calls&&!bn_backend_layout_bf16_glm_model(b));
    for(int fail=1;fail<=calls;++fail){q=(GLMModelProbe){.fail=fail};
        assert(bn_backend_layout_prepare_bf16_glm_model(b,&gpu,&file,a,maps,2,SIZE_MAX,SIZE_MAX)==-1);
        assert(q.calls==fail&&!q.live&&q.freed==fail-1&&!bn_backend_layout_bf16_glm_model(b)&&!bn_backend_model_gpu(b));}
    q=(GLMModelProbe){0};assert(bn_backend_layout_prepare_bf16_glm_model(b,&gpu,&file,a,maps,1,budget,host)==-1&&!q.calls);
    assert(bn_backend_layout_prepare_bf16_glm_model(b,&gpu,&file,a,maps,2,0,host)==-1&&!q.calls);
    assert(bn_backend_layout_prepare_bf16_glm_model(b,&gpu,&file,a,maps,2,budget-1,host)==-1&&!q.live);
    q=(GLMModelProbe){0};assert(bn_backend_layout_prepare_bf16_glm_model(b,&gpu,&file,a,maps,2,budget,host-1)==-1&&!q.live);
    q=(GLMModelProbe){0};float bad=NAN;memcpy(raw,&bad,4);
    assert(bn_backend_layout_prepare_bf16_glm_model(b,&gpu,&file,a,maps,2,budget,host)==-1&&!q.calls);memset(raw,0,4);
    int saved=a->layers[1].tensor[BN_MODEL_TENSOR_MOE_DOWN_EXPS];a->layers[1].tensor[BN_MODEL_TENSOR_MOE_DOWN_EXPS]=-1;
    assert(bn_backend_layout_prepare_bf16_glm_model(b,&gpu,&file,a,maps,2,budget,host)==-1&&!q.live&&!bn_backend_model_gpu(b));a->layers[1].tensor[BN_MODEL_TENSOR_MOE_DOWN_EXPS]=saved;
    q=(GLMModelProbe){0};const char foreign=0;void*owned=malloc(1);assert(owned);assert(!bn_backend_model_attach_gpu_resources(b,&foreign,owned,free));
    assert(bn_backend_layout_prepare_bf16_glm_model(b,&gpu,&file,a,maps,2,budget,host)==-1&&!q.live&&bn_backend_model_gpu_resources(b,&foreign)==owned);
    bn_backend_model_free(b);free(raw);free(a);
    puts("  GLM model ownership, aggregate budgets and every upload rollback PASSED");
}
