typedef struct {int calls,fail,freed;} MHCPrepareProbe;
static void *mhc_upload_probe(void *ctx,const void *data,size_t bytes,int type,int rows,int cols) {
    MHCPrepareProbe*p=ctx;assert(data&&bytes&&rows>0&&cols>0);
    assert(type==BN_GGUF_TENSOR_F32||type==BN_GGUF_TENSOR_BF16);
    if(++p->calls==p->fail)return NULL;
    void *copy=malloc(bytes);assert(copy);memcpy(copy,data,bytes);return copy;
}
static void mhc_destroy_probe(void *ctx,void *buffer) {MHCPrepareProbe*p=ctx;assert(buffer);p->freed++;free(buffer);}
static void test_mhc_checkpoint_preparation(void) {
    BnModelAnatomy*a=calloc(1,sizeof(*a));assert(a);a->n_layers=1;a->dim=1024;
    a->mhc_streams=4;a->mhc_iterations=20;a->mhc_eps=1e-6f;a->norm_eps=1e-5f;
    a->layers[0].kind=BN_MODEL_ANATOMY_KDA;
    for(int i=0;i<BN_MODEL_TENSOR_ROLE_COUNT;++i)a->layers[0].tensor[i]=-1;
    BnModelTensorRole roles[]={BN_MODEL_TENSOR_MHC_ATTN_FN,BN_MODEL_TENSOR_MHC_ATTN_SCALE,BN_MODEL_TENSOR_MHC_ATTN_BASE,BN_MODEL_TENSOR_ATTN_NORM};
    size_t fn_bytes=24u*4096/32*34,raw_bytes=fn_bytes+12+96+4096;
    unsigned char*raw=calloc(1,raw_bytes),*saved=malloc(raw_bytes);assert(raw&&saved);
    for(size_t i=0;i<fn_bytes/34;++i) {
        uint16_t d=0x3c01;memcpy(raw+i*34,&d,2);
        for(int j=0;j<32;++j)raw[i*34+2+j]=(unsigned char)(j-16);
    }
    BnGGUFTensorInfo tensors[4]={0};size_t offsets[]={0,fn_bytes,fn_bytes+12,fn_bytes+108};
    int widths[]={4096,3,24,1024};
    for(int i=0;i<4;++i) {
        a->layers[0].tensor[roles[i]]=i;
        tensors[i].type=i?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_Q8_0;
        tensors[i].n_dims=i?1:2;tensors[i].dims[0]=widths[i];tensors[i].dims[1]=i?1:24;tensors[i].offset=offsets[i];
        if(i)for(int j=0;j<widths[i];++j){float value=1.00390625f;memcpy(raw+offsets[i]+j*4,&value,4);}
    }
    memcpy(saved,raw,raw_bytes);BnGGUFFile f={0};f.tensors=tensors;f.n_tensors=4;f.raw=raw;f.raw_size=raw_bytes;
    MHCPrepareProbe probe={0};BnGPUBackend g={0};g.ctx=&probe;g.buffer_create_quant_only=mhc_upload_probe;g.buffer_destroy=mhc_destroy_probe;
    BnBackendBF16MHCWeights w={0};size_t bytes=24u*4096*4+2048+108,host=24u*4096*4;
    assert(!bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes,host));
    assert(w.bytes==bytes&&w.host_staging_bytes==host&&probe.calls==4);
    for(size_t j=0;j<24u*4096;++j)assert(((float*)w.fn)[j]==bn_fp16_to_fp32(0x3c01)*(float)((int)(j%32)-16));
    for(int j=0;j<1024;++j)assert(((uint16_t*)w.norm)[j]==0x3f80);
    assert(!memcmp(w.scale,raw+offsets[1],12)&&!memcmp(w.base,raw+offsets[2],96));
    assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes,host)==-1);
    bn_backend_layout_bf16_mhc_free(&w);bn_backend_layout_bf16_mhc_free(&w);assert(probe.freed==4);
    for(int fail=1;fail<=4;++fail) {
        probe=(MHCPrepareProbe){.fail=fail};BnBackendBF16MHCWeights before=w;
        assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes,host)==-1);
        assert(probe.calls==fail&&probe.freed==fail-1&&!memcmp(&w,&before,sizeof(w)));
    }
    probe=(MHCPrepareProbe){0};
    assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes-1,host)==-1);
    assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes,host-1)==-1);
    assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,1,bytes,host)==-1);
    assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,1,0,bytes,host)==-1);
    for(int i=0;i<4;++i) {
        uint64_t d=tensors[i].dims[0];tensors[i].dims[0]++;
        assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes,host)==-1);tensors[i].dims[0]=d;
        uint64_t off=tensors[i].offset;tensors[i].offset=raw_bytes;
        assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes,host)==-1);tensors[i].offset=off;
        uint32_t type=tensors[i].type;tensors[i].type=BN_GGUF_TENSOR_F16;
        assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes,host)==-1);tensors[i].type=type;
    }
    for(int i=1;i<4;++i) {
        float bad=NAN;memcpy(raw+offsets[i],&bad,4);
        assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes,host)==-1);memcpy(raw+offsets[i],saved+offsets[i],4);
    }
    uint16_t inf=0x7c00;memcpy(raw,&inf,2);
    assert(bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes,host)==-1);memcpy(raw,saved,2);
    assert(!probe.calls&&!memcmp(raw,saved,raw_bytes));
    /* The same derived layout is shared by indexed-MLA and KDA blocks. */
    a->layers[0].kind=BN_MODEL_ANATOMY_INDEXED_MLA;
    assert(!bn_backend_layout_bf16_mhc_init(&w,&g,&f,a,0,0,bytes,host));
    bn_backend_layout_bf16_mhc_free(&w);assert(probe.calls==4&&probe.freed==4);
    free(raw);free(saved);free(a);puts("  mHC F32 projection, BF16 norm, budgets and preparation failures PASSED");
}

typedef struct {int calls,fail,fused,block;} MHCChainProbe;
static int mhc_chain_step(void*ctx,int kind) {
    MHCChainProbe*p=ctx;int kinds[]={1,2,3,3,3,2,2,4,5,6,2,7};
    int block_kinds[]={1,2,3,3,3,2,2,4,5,6,2,8,2,9,2};
    int bound=p->block?15:12;assert(p->calls<bound);
    int expected=p->block?block_kinds[p->calls]:kinds[p->calls];
    assert(kind==(p->calls?expected:(p->fused?8:1)));return ++p->calls==p->fail?-1:0;
}
#define MHC_CHAIN_CALLBACK(NAME,PLAN,KIND) static int NAME(void*ctx,void*r,const PLAN*p) {assert(ctx==r&&p);return mhc_chain_step(ctx,KIND);}
MHC_CHAIN_CALLBACK(mhc_chain_pre,BnGPUMHCPreResidentPlan,1)
MHC_CHAIN_CALLBACK(mhc_chain_fused,BnGPUMHCFusedResidentPlan,8)
MHC_CHAIN_CALLBACK(mhc_chain_linear,BnGPUBF16LinearResidentPlan,2)
MHC_CHAIN_CALLBACK(mhc_chain_split,BnGPUSplitColumnsResidentPlan,3)
MHC_CHAIN_CALLBACK(mhc_chain_conv,BnGPUCausalConvResidentPlan,4)
MHC_CHAIN_CALLBACK(mhc_chain_kda,BnGPUKDARecurrentResidentPlan,5)
MHC_CHAIN_CALLBACK(mhc_chain_norm,BnGPUGatedRMSNormResidentPlan,6)
MHC_CHAIN_CALLBACK(mhc_chain_post,BnGPUMHCPostResidentPlan,7)
MHC_CHAIN_CALLBACK(mhc_chain_clamp,BnGPUClampedSwiGLUResidentPlan,9)
#undef MHC_CHAIN_CALLBACK
static void test_mhc_kda_orchestration(void) {
    MHCChainProbe probe={0};BnGPUBackend g={0};g.ctx=&probe;
    g.mhc_fused_post_pre_bf16_resident=mhc_chain_fused;g.mhc_pre_bf16_resident=mhc_chain_pre;g.mhc_post_bf16_resident=mhc_chain_post;
    g.linear_a8_bf16_resident=mhc_chain_linear;g.split_columns_resident=mhc_chain_split;
    g.causal_conv_f32_weight_bf16_resident=mhc_chain_conv;g.kda_recurrent_bf16_resident=mhc_chain_kda;
    g.gated_rmsnorm_f32_weight_bf16_resident=mhc_chain_norm;
    BnBackendBF16KDAWeights w={.gpu=&g,.dim=1024,.heads=1,.head_dim=128,.rank=128,.kernel=4,
        .lower_bound=-5,.output_eps=1e-5f,.conv=&g,.decay=&g,.bias=&g,.norm=&g};
    w.in=w.f_b=w.g_b=w.out=(BnGPUBF16LinearWeight){.raw=&g,.type=BN_GGUF_TENSOR_Q8_0};
    size_t counts[]={12288,72,3,12,12,48,3072,3072,1923,1152,771,768,3,384,384,384,384,1152,384,384,1152,1152,16384,16384,3072,12288};
    BnGPUResidentView v[26];for(int i=0;i<26;++i)v[i]=(BnGPUResidentView){.request=&probe,
        .slot=(uint32_t)i,.count=counts[i],.type=((i>=1&&i<=5)||i==22||i==23)?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_BF16};
    BnTransformerBF16MHCKDAPlan p={0};p.weights=&w;
    p.mhc=(BnGPUMHCPreResidentPlan){.n_tokens=3,.dim=1024,.streams=4,.n_splits=1,.iterations=20,
        .norm=&g,.residual=v[0],.dot=v[1],.sqrsum=v[2],.pre=v[3],.post=v[4],.comb=v[5],.collapsed=v[6],.out=v[7]};
    p.kda=(BnTransformerBF16KDAWorkspace){v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18],v[19]};
    p.initial_history=v[20];p.next_history=v[21];p.initial_state=v[22];p.next_state=v[23];p.branch_out=v[24];p.residual_out=v[25];
    assert(!bn_transformer_bf16_mhc_kda_attention(&probe,&p)&&probe.calls==12);
    for(int fail=1;fail<=12;++fail){probe=(MHCChainProbe){.fail=fail};assert(bn_transformer_bf16_mhc_kda_attention(&probe,&p)==-1&&probe.calls==fail);}
    probe=(MHCChainProbe){0};BnTransformerBF16MHCKDAPlan bad=p;bad.residual_out.count--;
    assert(bn_transformer_bf16_mhc_kda_attention(&probe,&bad)==-1);
    bad=p;bad.kda.g1=bad.kda.g2;assert(bn_transformer_bf16_mhc_kda_attention(&probe,&bad)==-1);
    bad=p;bad.mhc.comb.request=&g;assert(bn_transformer_bf16_mhc_kda_attention(&probe,&bad)==-1);
    bad=p;bad.mhc.norm=NULL;assert(bn_transformer_bf16_mhc_kda_attention(&probe,&bad)==-1);
    assert(!probe.calls);
    p.deferred=1;p.residual_out=(BnGPUResidentView){0};
    assert(!bn_transformer_bf16_mhc_kda_attention(&probe,&p)&&probe.calls==11);
    p.has_previous=1;p.previous_residual=v[0];p.previous_residual.slot=26;
    p.previous_branch=v[24];p.previous_branch.slot=27;
    p.previous_post=v[4];p.previous_post.slot=28;
    p.previous_comb=v[5];p.previous_comb.slot=29;
    for(int fail=0;fail<=11;++fail) {
        probe=(MHCChainProbe){.fail=fail,.fused=1};int rc=bn_transformer_bf16_mhc_kda_attention(&probe,&p);
        assert(rc==(fail?-1:0)&&probe.calls==(fail?fail:11));
    }
    probe=(MHCChainProbe){.fused=1};bad=p;bad.previous_residual=p.mhc.residual;
    assert(bn_transformer_bf16_mhc_kda_attention(&probe,&bad)==-1);
    bad=p;bad.previous_post.type=BN_GGUF_TENSOR_BF16;
    assert(bn_transformer_bf16_mhc_kda_attention(&probe,&bad)==-1);
    bad=p;bad.deferred=2;assert(bn_transformer_bf16_mhc_kda_attention(&probe,&bad)==-1);
    assert(!probe.calls);puts("  mHC/KDA materialized/deferred composition preflight and all submission failures PASSED");
    g.clamped_swiglu_bf16_resident=mhc_chain_clamp;
    BnBackendBF16DenseFFNWeights dw={.gpu=&g,.dim=1024,.hidden_dim=2048,.clamp_limit=10};
    dw.gate_up=dw.down=w.in;
    BnTransformerBF16DenseKDABlockPlan block={.attention=p,.weights=&dw};
    block.ffn=p.mhc;block.ffn.fn=block.ffn.scale=block.ffn.base=block.ffn.norm=&g;
    block.ffn.rms_eps=block.ffn.norm_eps=1e-5f;block.ffn.pre_eps=block.ffn.sinkhorn_eps=1e-6f;
    block.ffn.post_mult=2;
    BnGPUResidentView *fv[]={&block.ffn.residual,&block.ffn.dot,&block.ffn.sqrsum,&block.ffn.pre,&block.ffn.post,&block.ffn.comb,&block.ffn.collapsed,&block.ffn.out};
    for(int i=0;i<8;++i)fv[i]->slot=30+i;
    block.gate_up=(BnGPUResidentView){.request=&probe,.slot=38,.type=BN_GGUF_TENSOR_BF16,.count=12288};
    block.activated=block.gate_up;block.activated.slot=39;block.activated.count=6144;
    block.out=block.gate_up;block.out.slot=40;block.out.count=3072;
    for(int prior=0;prior<2;++prior){block.attention.has_previous=prior;
        for(int fail=0;fail<=15;++fail){probe=(MHCChainProbe){.fail=fail,.fused=prior,.block=1};
            assert(bn_transformer_bf16_dense_kda_block(&probe,&block)==(fail?-1:0));assert(probe.calls==(fail?fail:15));}
    }
    probe=(MHCChainProbe){.block=1};
    BnTransformerBF16DenseKDABlockPlan bb=block;bb.out=block.attention.branch_out;
    assert(bn_transformer_bf16_dense_kda_block(&probe,&bb)==-1);
    bb=block;bb.ffn.out=block.attention.mhc.out;assert(bn_transformer_bf16_dense_kda_block(&probe,&bb)==-1);
    bb=block;bb.ffn.comb.type=BN_GGUF_TENSOR_BF16;assert(bn_transformer_bf16_dense_kda_block(&probe,&bb)==-1);
    bb=block;bb.ffn.sqrsum.count--;assert(bn_transformer_bf16_dense_kda_block(&probe,&bb)==-1);
    bb=block;bb.ffn.norm_eps=NAN;assert(bn_transformer_bf16_dense_kda_block(&probe,&bb)==-1);
    bb=block;bb.attention.deferred=0;assert(bn_transformer_bf16_dense_kda_block(&probe,&bb)==-1);
    assert(!probe.calls);puts("  complete dense KDA block alias preflight and all submission failures PASSED");

}
