/* CPU-only failure/continuation probes for the streaming orchestrator. */
typedef struct {
    int calls,fail,busy,cancel;
    size_t gate_bytes,down_bytes;
    unsigned char *bank;
} MoEWorkProbe;
static int moe_work_call(MoEWorkProbe *q){return ++q->calls==q->fail?-1:0;}
#define MOE_WORK_CALLBACK(name,type) \
static int name(void *ctx,void *r,const type *p){assert(ctx==r&&p);return moe_work_call(ctx);}
MOE_WORK_CALLBACK(moe_work_router,BnGPUBF16F32LinearResidentPlan)
MOE_WORK_CALLBACK(moe_work_top,BnGPUBiasedSigmoidF32ResidentPlan)
MOE_WORK_CALLBACK(moe_work_group,BnGPUMoEGroupBF16ResidentPlan)
MOE_WORK_CALLBACK(moe_work_linear,BnGPUBF16LinearResidentPlan)
MOE_WORK_CALLBACK(moe_work_clamp,BnGPUClampedSwiGLUResidentPlan)
MOE_WORK_CALLBACK(moe_work_bank_submit,BnGPUMoEBankResidentPlan)
MOE_WORK_CALLBACK(moe_work_reduce,BnGPUMoEReduceBF16ResidentPlan)
MOE_WORK_CALLBACK(moe_work_add,BnGPUAddBF16ResidentPlan)
#undef MOE_WORK_CALLBACK
static int moe_work_zero(void *r,BnGPUResidentView v){assert(v.count==293);return moe_work_call(r);}
static int moe_work_read(void *r,BnGPUResidentView v,void *target){
    assert(v.count==293);if(moe_work_call(r))return -1;
    int32_t *o=target;memset(o,0,293*sizeof(*o));
    for(int e=0;e<=288;++e)o[e]=(e<8?e:8)*2;
    return 0;
}
static int moe_work_prepare(void *ctx,void *r,const BnGPUMoEBankInputPlan *p,uint64_t *generation){
    assert(ctx==r&&p);if(moe_work_call(ctx))return -1;*generation=1;return 0;
}
static int moe_work_acquire(void *r,unsigned slot,int gt,int dt,BnGPUMoEBankLease *out){
    MoEWorkProbe *q=r;assert(slot<2&&gt==BN_GGUF_TENSOR_Q4_K&&dt==BN_GGUF_TENSOR_Q5_K);
    if(moe_work_call(q))return -1;
    if(q->busy){q->busy=0;return BN_GPU_MOE_BANK_BUSY;}
    *out=(BnGPUMoEBankLease){.capacity=2,.data=q->bank,.gate_bytes=q->gate_bytes,.down_bytes=q->down_bytes,.expert_bytes=2*q->gate_bytes+q->down_bytes};return 0;
}
static int moe_work_cancel(void *r,const BnGPUMoEBankLease *lease){MoEWorkProbe *q=r;assert(lease->data==q->bank);q->cancel++;return 0;}
static void test_kda_moe_work_continuation(void);
static void test_moe_work_continuation(void){
    size_t gb,db;assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_Q4_K,256u*256,&gb));assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_Q5_K,256u*256,&db));
    size_t stride=2*gb+db;unsigned char *raw=calloc(288,stride),*bank=malloc(2*stride);assert(raw&&bank);
    BnMoEExpertMap map={.gate_type=BN_GGUF_TENSOR_Q4_K,.up_type=BN_GGUF_TENSOR_Q4_K,.down_type=BN_GGUF_TENSOR_Q5_K,
        .gate_rows=256,.up_rows=256,.down_rows=256,.gate_cols=256,.up_cols=256,.down_cols=256,
        .expert_gate_bytes=gb,.expert_up_bytes=gb,.expert_down_bytes=db,.gate_stride=gb,.up_stride=gb,.down_stride=db,
        .up_offset=288*gb,.down_offset=576*gb};
    MoEWorkProbe q={.gate_bytes=gb,.down_bytes=db,.bank=bank};
    BnGPUBackend g={.ctx=&q,.resident_zero=moe_work_zero,.resident_read=moe_work_read,
        .linear_bf16_f32_weight_resident=moe_work_router,.moe_biased_sigmoid_f32_resident=moe_work_top,
        .moe_group_bf16_resident=moe_work_group,.linear_a8_bf16_resident=moe_work_linear,
        .clamped_swiglu_bf16_resident=moe_work_clamp,.moe_bank_prepare_input=moe_work_prepare,
        .moe_bank_acquire=moe_work_acquire,.moe_bank_submit=moe_work_bank_submit,.moe_bank_cancel=moe_work_cancel,
        .moe_reduce_bf16_resident=moe_work_reduce,.add_bf16_resident=moe_work_add};
    BnBackendBF16MoEWeights w={.gpu=&g,.router=&g,.bias=&g,.source_map=&map,.dim=256,.hidden_dim=256,
        .n_experts=288,.k=8,.gate_type=BN_GGUF_TENSOR_Q4_K,.down_type=BN_GGUF_TENSOR_Q5_K,
        .gate_bytes=gb,.down_bytes=db,.route_scale=2.5f,.clamp_limit=10};
    w.shared=(BnBackendBF16DenseFFNWeights){.gpu=&g,.dim=256,.hidden_dim=256,.clamp_limit=10};
    w.shared.gate_up=w.shared.down=(BnGPUBF16LinearWeight){.raw=&g,.type=BN_GGUF_TENSOR_Q8_0};
    BnMoEIO io={.fd=-1,.mmap_base=raw};BnMoEState host={0};
    size_t counts[]={512,512,576,16,16,293,288,16,16,4096,4096,4096,4096,4096,512,1024,512,512,512};
    BnGPUResidentView v[19];for(int i=0;i<19;++i)v[i]=(BnGPUResidentView){.request=&q,.slot=(uint32_t)i,
        .type=i==1||i==2||i==3?BN_GGUF_TENSOR_F32:i>=4&&i<=8?BN_GGUF_TENSOR_I32:BN_GGUF_TENSOR_BF16,.count=counts[i]};
    BnTransformerBF16MoEPlan p={.weights=&w,.source=&io,.host_state=&host,.n_tokens=2,.prefill=1,.bank_capacity=2,
        .x=v[0],.cast=v[1],.logits=v[2],.route_weights=v[3],.indices=v[4],.metadata=v[5],.cursors=v[6],.slots=v[7],
        .inverse=v[8],.gathered=v[9],.gate=v[10],.up=v[11],.down=v[12],.weighted=v[13],.routed=v[14],
        .shared_gate_up=v[15],.shared_activated=v[16],.shared_out=v[17],.out=v[18]};
    for(int fail=1;fail<=9;++fail){q.calls=0;q.fail=fail;BnTransformerBF16MoEWork work={0};
        assert(bn_transformer_bf16_moe_begin(&q,&p,&work)==-1&&work.failed&&q.calls==fail);
        assert(bn_transformer_bf16_moe_submit(&q,&work)==-1&&q.calls==fail);}
    q.calls=q.fail=0;BnTransformerBF16MoEWork work={0};
    assert(!bn_transformer_bf16_moe_begin(&q,&p,&work)&&q.calls==9);q.busy=1;
    assert(bn_transformer_bf16_moe_submit(&q,&work)==BN_GPU_MOE_BANK_BUSY);
    assert(work.pending&&!work.failed&&!work.next_expert&&!work.submitted_banks&&!host.stats.io_bytes);
    assert(!bn_transformer_bf16_moe_submit(&q,&work));assert(!work.pending&&!work.failed&&work.submitted_banks==4&&work.submitted_experts==8);
    assert(work.metadata_read_bytes==1172&&!host.buf&&!host.buf2&&!host.buf5&&!q.cancel);
    /* A host read error cancels the lease and restores borrowed state. */
    work=(BnTransformerBF16MoEWork){0};assert(!bn_transformer_bf16_moe_begin(&q,&p,&work));io.mmap_base=NULL;
    assert(bn_transformer_bf16_moe_submit(&q,&work)==-1&&work.failed&&q.cancel==1);
    assert(!host.buf&&!host.buf2&&!host.buf5&&!host.buf_size&&!host.buf2_size&&!host.buf5_size);
    assert(bn_transformer_bf16_moe_submit(&q,&work)==-1&&q.cancel==1);io.mmap_base=raw;
    /* A later enqueue failure retains already-submitted progress. */
    q.calls=0;q.fail=13;q.cancel=0;work=(BnTransformerBF16MoEWork){0};
    assert(!bn_transformer_bf16_moe_begin(&q,&p,&work));
    assert(bn_transformer_bf16_moe_submit(&q,&work)==-1&&work.failed&&q.cancel==1);
    assert(work.submitted_banks==1&&work.submitted_experts==2&&work.next_expert==2);
    assert(!host.buf&&!host.buf2&&!host.buf5);
    test_kda_moe_work_continuation();
    free(raw);free(bank);puts("  MoE BUSY continuation, poisoned work, source restoration and partial failure PASSED");
}
#define KDA_MOE_CALLBACK(name,type) \
static int name(void *ctx,void *r,const type *p){assert(ctx==r&&p);return moe_work_call(ctx);}
KDA_MOE_CALLBACK(kda_moe_pre,BnGPUMHCPreResidentPlan)
KDA_MOE_CALLBACK(kda_moe_fused,BnGPUMHCFusedResidentPlan)
KDA_MOE_CALLBACK(kda_moe_split,BnGPUSplitColumnsResidentPlan)
KDA_MOE_CALLBACK(kda_moe_conv,BnGPUCausalConvResidentPlan)
KDA_MOE_CALLBACK(kda_moe_recurrent,BnGPUKDARecurrentResidentPlan)
KDA_MOE_CALLBACK(kda_moe_norm,BnGPUGatedRMSNormResidentPlan)
#undef KDA_MOE_CALLBACK
static int kda_moe_splits(void *ctx,int nt,int dim){assert(ctx&&nt==2&&dim==4096);return 1;}
static void test_kda_moe_work_continuation(void) {
    /* Opaque synthetic source payloads test orchestration, not kernel math. */
    unsigned char raw[288*48]={0},bank[96];MoEWorkProbe q={.gate_bytes=16,.down_bytes=16,.bank=bank};
    BnGPUBackend g={.ctx=&q,.resident_zero=moe_work_zero,.resident_read=moe_work_read,
        .linear_bf16_f32_weight_resident=moe_work_router,.moe_biased_sigmoid_f32_resident=moe_work_top,
        .moe_group_bf16_resident=moe_work_group,.linear_a8_bf16_resident=moe_work_linear,
        .clamped_swiglu_bf16_resident=moe_work_clamp,.moe_bank_prepare_input=moe_work_prepare,
        .moe_bank_acquire=moe_work_acquire,.moe_bank_submit=moe_work_bank_submit,.moe_bank_cancel=moe_work_cancel,
        .moe_reduce_bf16_resident=moe_work_reduce,.add_bf16_resident=moe_work_add,
        .mhc_pre_bf16_resident=kda_moe_pre,.mhc_fused_post_pre_bf16_resident=kda_moe_fused,
        .mhc_pre_bf16_split_count=kda_moe_splits,.mhc_fused_bf16_split_count=kda_moe_splits,
        .split_columns_resident=kda_moe_split,.causal_conv_f32_weight_bf16_resident=kda_moe_conv,
        .kda_recurrent_bf16_resident=kda_moe_recurrent,.gated_rmsnorm_f32_weight_bf16_resident=kda_moe_norm};
    BnBackendBF16KDAWeights kw={.gpu=&g,.dim=4096,.heads=1,.head_dim=128,.rank=128,.kernel=4,.layer=4,
        .lower_bound=-5,.output_eps=1e-5f,.conv=&g,.decay=&g,.bias=&g,.norm=&g};
    kw.in=kw.f_b=kw.g_b=kw.out=(BnGPUBF16LinearWeight){.raw=&g,.type=BN_GGUF_TENSOR_Q8_0};
    BnMoEExpertMap map={.gate_type=BN_GGUF_TENSOR_Q4_K,.up_type=BN_GGUF_TENSOR_Q4_K,.down_type=BN_GGUF_TENSOR_Q5_K,
        .gate_rows=256,.up_rows=256,.down_rows=4096,.gate_cols=4096,.up_cols=4096,.down_cols=256,
        .expert_gate_bytes=16,.expert_up_bytes=16,.expert_down_bytes=16,.gate_stride=16,.up_stride=16,.down_stride=16,
        .up_offset=288*16,.down_offset=576*16};
    BnBackendBF16MoEWeights mw={.gpu=&g,.router=&g,.bias=&g,.source_map=&map,.dim=4096,.hidden_dim=256,
        .n_experts=288,.k=8,.layer=4,.gate_type=map.gate_type,.down_type=map.down_type,
        .gate_bytes=16,.down_bytes=16,.route_scale=2.5f,.clamp_limit=10};
    mw.shared=(BnBackendBF16DenseFFNWeights){.gpu=&g,.dim=4096,.hidden_dim=256,.clamp_limit=10};
    mw.shared.gate_up=mw.shared.down=kw.in;
    BnMoEIO io={.fd=-1,.mmap_base=raw};BnMoEState host={0};
    size_t counts[]={32768,48,2,8,8,32,8192,8192,1282,768,514,512,2,256,256,256,256,768,256,256,
        1152,1152,16384,16384,8192,32768,48,2,8,8,32,8192,8192,
        8192,576,16,16,293,288,16,16,65536,4096,4096,65536,65536,8192,1024,512,8192,8192,
        32768,8192,8,32};
    BnGPUResidentView v[55];for(int i=0;i<55;++i){int type=BN_GGUF_TENSOR_BF16;
        if((i>=1&&i<=5)||i==22||i==23||(i>=26&&i<=30)||(i>=33&&i<=35)||i>=53)type=BN_GGUF_TENSOR_F32;
        if(i>=36&&i<=40)type=BN_GGUF_TENSOR_I32;
        v[i]=(BnGPUResidentView){.request=&q,.slot=(uint32_t)i,.count=counts[i],.type=type};}
    BnGPUMHCPreResidentPlan pre={.n_tokens=2,.dim=4096,.streams=4,.n_splits=1,.iterations=20,
        .fn=&g,.scale=&g,.base=&g,.norm=&g,.rms_eps=1e-5f,.pre_eps=1e-6f,.sinkhorn_eps=1e-6f,.post_mult=2,.norm_eps=1e-5f,
        .residual=v[0],.dot=v[1],.sqrsum=v[2],.pre=v[3],.post=v[4],.comb=v[5],.collapsed=v[6],.out=v[7]};
    BnTransformerBF16KDAMoEBlockPlan p={.attention={.weights=&kw,.mhc=pre,.deferred=1,
        .kda={v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18],v[19]},
        .initial_history=v[20],.next_history=v[21],.initial_state=v[22],.next_state=v[23],.branch_out=v[24],
        .previous_residual=v[51],.previous_branch=v[52],.previous_post=v[53],.previous_comb=v[54]},.ffn=pre,
        .moe={.weights=&mw,.source=&io,.host_state=&host,.layer=4,.n_tokens=2,.prefill=1,.bank_capacity=2,
        .x=v[32],.cast=v[33],.logits=v[34],.route_weights=v[35],.indices=v[36],.metadata=v[37],.cursors=v[38],.slots=v[39],.inverse=v[40],
        .gathered=v[41],.gate=v[42],.up=v[43],.down=v[44],.weighted=v[45],.routed=v[46],.shared_gate_up=v[47],.shared_activated=v[48],.shared_out=v[49],.out=v[50]}};
    p.ffn.residual=v[25];p.ffn.dot=v[26];p.ffn.sqrsum=v[27];p.ffn.pre=v[28];p.ffn.post=v[29];p.ffn.comb=v[30];p.ffn.collapsed=v[31];p.ffn.out=v[32];
    for(int prior=0;prior<2;++prior){p.attention.has_previous=prior;
        for(int fail=0;fail<=21;++fail){q.calls=0;q.fail=fail;BnTransformerBF16KDAMoEBlockWork work={0};
            int rc=bn_transformer_bf16_kda_moe_begin(&q,&p,&work);assert(rc==(fail?-1:0)&&q.calls==(fail?fail:21));
            assert(work.pending&&work.failed==(fail!=0));
            if(fail)assert(bn_transformer_bf16_kda_moe_submit(&q,&work)==-1&&q.calls==fail);}
    }
    q.calls=q.fail=0;BnTransformerBF16KDAMoEBlockWork work={0},before=work;
    BnTransformerBF16KDAMoEBlockPlan bad=p;bad.ffn.residual=p.attention.mhc.residual;
    assert(bn_transformer_bf16_kda_moe_begin(&q,&bad,&work)==-1&&!memcmp(&work,&before,sizeof(work)));
    bad=p;bad.moe.x=p.attention.mhc.out;assert(bn_transformer_bf16_kda_moe_begin(&q,&bad,&work)==-1);
    bad=p;bad.ffn.n_splits=2;assert(bn_transformer_bf16_kda_moe_begin(&q,&bad,&work)==-1);
    bad=p;bad.attention.deferred=0;assert(bn_transformer_bf16_kda_moe_begin(&q,&bad,&work)==-1);
    bad=p;bad.ffn.norm_eps=NAN;assert(bn_transformer_bf16_kda_moe_begin(&q,&bad,&work)==-1);
    kw.layer=5;assert(bn_transformer_bf16_kda_moe_begin(&q,&p,&work)==-1);kw.layer=4;assert(!q.calls);
    assert(!bn_transformer_bf16_kda_moe_begin(&q,&p,&work)&&q.calls==21);
    assert(bn_transformer_bf16_kda_moe_begin(&q,&p,&work)==-1&&q.calls==21);
    assert(bn_transformer_bf16_kda_moe_submit(&g,&work)==-1&&q.calls==21);q.busy=1;
    assert(bn_transformer_bf16_kda_moe_submit(&q,&work)==BN_GPU_MOE_BANK_BUSY&&work.pending&&!work.failed&&q.calls==22);
    assert(!bn_transformer_bf16_kda_moe_submit(&q,&work)&&!work.pending&&!work.failed&&q.calls==32);
    assert(work.moe.submitted_experts==8&&work.moe.metadata_read_bytes==1172);
    assert(bn_transformer_bf16_kda_moe_submit(&q,&work)==-1&&q.calls==32);
    q.calls=0;q.fail=22;work=(BnTransformerBF16KDAMoEBlockWork){0};
    assert(!bn_transformer_bf16_kda_moe_begin(&q,&p,&work));
    assert(bn_transformer_bf16_kda_moe_submit(&q,&work)==-1&&work.failed);
    assert(bn_transformer_bf16_kda_moe_submit(&q,&work)==-1&&q.calls==22);
    assert(!host.buf&&!host.buf2&&!host.buf5);
    puts("  complete KDA/MoE validation, all begin failures and BUSY/poisoned continuation PASSED");
}
