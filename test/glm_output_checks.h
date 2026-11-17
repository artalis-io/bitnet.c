typedef struct {int calls,fail,nt,last;} GLMOutputProbe;
static int glm_output_call(GLMOutputProbe*q,int expected){assert(q->calls+1==expected);return ++q->calls==q->fail?-1:0;}
static int glm_output_post(void*ctx,void*r,const BnGPUMHCPostResidentPlan*p){assert(ctx==r&&p->n_tokens==((GLMOutputProbe*)ctx)->nt);return glm_output_call(ctx,1);}
static int glm_output_mean(void*ctx,void*r,const BnGPUStreamMeanResidentPlan*p){assert(ctx==r&&p->streams==4);return glm_output_call(ctx,2);}
static int glm_output_norm(void*ctx,void*r,const BnGPURMSNormResidentPlan*p){GLMOutputProbe*q=ctx;assert(ctx==r&&p->n_tokens==q->nt&&p->reduction==BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096);return glm_output_call(q,3);}
static int glm_output_head(void*ctx,void*r,const BnGPUBF16LinearResidentPlan*p){GLMOutputProbe*q=ctx;assert(ctx==r&&p->n_tokens==(q->last?1:q->nt)&&p->x.offset==(size_t)(q->last?q->nt-1:0)*4096);return glm_output_call(q,4);}
static void test_glm_output_orchestration(void){
    GLMOutputProbe q={0};BnGPUBackend g={.ctx=&q,.mhc_post_bf16_resident=glm_output_post,.stream_mean_bf16_resident=glm_output_mean,
        .rmsnorm_bf16_resident=glm_output_norm,.linear_a8_bf16_resident=glm_output_head,
        .resident_plain_rmsnorm_reductions=1u<<BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096};
    BnModelAnatomy*a=calloc(1,sizeof(*a));assert(a);a->dim=4096;a->norm_eps=1e-5f;
    BnBackendBF16GLMModel w={.gpu=&g,.dim=4096,.vocab_size=32,.anatomy=a,.output_norm=&g,.output={.raw=&g,.type=BN_GGUF_TENSOR_Q8_0}};
    int tokens[]={1,9,257};for(int ti=0;ti<3;++ti)for(int last=0;last<2;++last){int nt=tokens[ti],head=last?1:nt;size_t n=nt;
        size_t sizes[]={n*16384,n*4096,n*4,n*16,n*16384,n*4096,n*4096,(size_t)head*32};BnGPUResidentView v[8];
        for(int i=0;i<8;++i)v[i]=(BnGPUResidentView){.request=&q,.slot=(uint32_t)i,.count=sizes[i],.type=i==2||i==3?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_BF16};
        BnTransformerBF16GLMOutputPlan p={&w,nt,last,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]};
        for(int fail=0;fail<=4;++fail){q=(GLMOutputProbe){.nt=nt,.last=last,.fail=fail};assert(bn_transformer_bf16_glm_output(&q,&p)==(fail?-1:0)&&q.calls==(fail?fail:4));}
        q=(GLMOutputProbe){.nt=nt,.last=last};BnTransformerBF16GLMOutputPlan b=p;b.completed=p.residual;assert(bn_transformer_bf16_glm_output(&q,&b)==-1);
        b=p;b.logits.count--;assert(bn_transformer_bf16_glm_output(&q,&b)==-1);
        b=p;b.normalized.offset=1;assert(bn_transformer_bf16_glm_output(&q,&b)==-1);
        b=p;b.last_only=2;assert(bn_transformer_bf16_glm_output(&q,&b)==-1);
        g.resident_plain_rmsnorm_reductions=0;assert(bn_transformer_bf16_glm_output(&q,&p)==-1);g.resident_plain_rmsnorm_reductions=1u<<BN_GPU_RMSNORM_REDUCTION_VECTOR8_4096;
        a->norm_eps=1e-6f;assert(bn_transformer_bf16_glm_output(&q,&p)==-1);a->norm_eps=1e-5f;assert(!q.calls);
    }
    free(a);puts("  GLM final output global preflight, all failures and sampled head geometry PASSED");
}
