/* Synthetic checkpoint preparation tests: no model/GPU required. */
typedef struct { int calls, fail, freed; } KDAPrepareProbe;
static void *kda_upload_probe(void *ctx,const void *data,size_t bytes,
    int type,int rows,int cols) {
    KDAPrepareProbe *p=ctx; assert(data&&bytes&&rows>0&&cols>0);
    assert(type==BN_GGUF_TENSOR_Q8_0||type==BN_GGUF_TENSOR_F32);
    if(++p->calls==p->fail) return NULL;
    void *copy=malloc(bytes); assert(copy); memcpy(copy,data,bytes); return copy;
}
static void kda_destroy_probe(void *ctx,void *buffer) {
    KDAPrepareProbe *p=ctx; assert(buffer); p->freed++; free(buffer);
}
static void test_kda_checkpoint_preparation(void) {
    BnModelAnatomy *a=calloc(1,sizeof(*a)); assert(a);
    a->n_layers=1; a->dim=128; a->kda_heads=1; a->kda_head_dim=128;
    a->kda_gate_rank=128; a->conv_kernel=4; a->kda_parameter_is_negative_exp=1;
    a->kda_qk_eps=1e-6f; a->kda_output_eps=1e-5f; a->kda_lower_bound=-5;
    a->layers[0].kind=BN_MODEL_ANATOMY_KDA;
    for(int i=0;i<BN_MODEL_TENSOR_ROLE_COUNT;++i) a->layers[0].tensor[i]=-1;
    const BnModelTensorRole roles[]={BN_MODEL_TENSOR_ATTN_Q,BN_MODEL_TENSOR_ATTN_K,
        BN_MODEL_TENSOR_ATTN_V,BN_MODEL_TENSOR_SSM_BETA,BN_MODEL_TENSOR_SSM_F_A,
        BN_MODEL_TENSOR_SSM_G_A,BN_MODEL_TENSOR_SSM_F_B,BN_MODEL_TENSOR_SSM_G_B,
        BN_MODEL_TENSOR_SSM_OUT,BN_MODEL_TENSOR_SSM_CONV1D_Q,BN_MODEL_TENSOR_SSM_CONV1D_K,
        BN_MODEL_TENSOR_SSM_CONV1D_V,BN_MODEL_TENSOR_SSM_A,BN_MODEL_TENSOR_SSM_DT_BIAS,
        BN_MODEL_TENSOR_SSM_NORM};
    BnGGUFTensorInfo tensors[15]={0};size_t sizes[15], offsets[15], total=0;
    for(int i=0;i<15;++i) {
        a->layers[0].tensor[roles[i]]=i;
        BnGGUFTensorInfo *t=&tensors[i];t->type=i<9?BN_GGUF_TENSOR_Q8_0:BN_GGUF_TENSOR_F32;
        t->n_dims=i<9?2:(i<12?3:1);t->dims[0]=i<9?128:(i<12?4:(i==12?1:128));
        t->dims[1]=i<9?(i==3?1:128):1;t->dims[2]=i<12&&i>=9?128:1;
        uint64_t count=1;for(uint32_t d=0;d<t->n_dims;++d)count*=t->dims[d];
        assert(bn_gguf_tensor_size(t->type,count,&sizes[i]));
        offsets[i]=total;t->offset=total;total+=sizes[i];
    }
    assert(total==146572);
    unsigned char *raw=malloc(total),*saved=malloc(total);assert(raw&&saved);
    for(int i=0;i<15;++i) {
        if(i<9) memset(raw+offsets[i],i+1,sizes[i]);
        else for(size_t j=0;j<sizes[i]/4;++j) {
            float value=i==12?-0.25f:(float)(i+1)+(float)j/8192;
            memcpy(raw+offsets[i]+j*4,&value,4);
        }
    }
    memcpy(saved,raw,total);
    BnGGUFFile file={0};file.n_tensors=15;file.tensors=tensors;file.raw=raw;file.raw_size=total;
    KDAPrepareProbe probe={0};BnGPUBackend gpu={0};gpu.ctx=&probe;
    gpu.buffer_create_quant_only=kda_upload_probe;gpu.buffer_destroy=kda_destroy_probe;
    BnBackendBF16KDAWeights w={0};
    assert(!bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total,87176));
    assert(w.layer==0);assert(w.bytes==total&&w.host_staging_bytes==87176&&probe.calls==8);
    size_t offset=0;for(int i=0;i<6;++i) {
        assert(!memcmp((unsigned char*)w.in.raw+offset,raw+offsets[i],sizes[i]));offset+=sizes[i];
    }
    assert(!memcmp(w.f_b.raw,raw+offsets[6],sizes[6]));
    assert(!memcmp(w.g_b.raw,raw+offsets[7],sizes[7]));
    assert(!memcmp(w.out.raw,raw+offsets[8],sizes[8]));
    offset=0;for(int i=9;i<12;++i) {
        assert(!memcmp((unsigned char*)w.conv+offset,raw+offsets[i],sizes[i]));offset+=sizes[i];
    }
    assert(!memcmp(w.decay,raw+offsets[12],sizes[12]));
    assert(!memcmp(w.bias,raw+offsets[13],sizes[13]));
    assert(!memcmp(w.norm,raw+offsets[14],sizes[14]));
    assert(bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total,87176)==-1);
    bn_backend_layout_bf16_kda_free(&w);assert(probe.freed==8);
    bn_backend_layout_bf16_kda_free(&w);assert(probe.freed==8);
    for(int fail=1;fail<=8;++fail) {
        probe=(KDAPrepareProbe){.fail=fail};BnBackendBF16KDAWeights before=w;
        assert(bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total,87176)==-1);
        assert(probe.calls==fail&&probe.freed==fail-1&&!memcmp(&before,&w,sizeof(w)));
    }
    probe=(KDAPrepareProbe){0};
    assert(bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total-1,87176)==-1);
    assert(bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total,87175)==-1);
    assert(bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,1,total,87176)==-1);
    for(int i=0;i<15;++i) {
        uint64_t d=tensors[i].dims[0];tensors[i].dims[0]++;
        assert(bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total,87176)==-1);
        tensors[i].dims[0]=d;uint32_t type=tensors[i].type;tensors[i].type=BN_GGUF_TENSOR_F16;
        assert(bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total,87176)==-1);
        tensors[i].type=type;
        uint64_t saved_offset=tensors[i].offset;tensors[i].offset=total;
        assert(bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total,87176)==-1);
        tensors[i].offset=saved_offset;
    }
    for(int i=9;i<15;++i) {
        float bad=NAN;memcpy(raw+offsets[i],&bad,4);
        assert(bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total,87176)==-1);
        memcpy(raw+offsets[i],saved+offsets[i],4);
    }
    float positive=0.25f;memcpy(raw+offsets[12],&positive,4);
    assert(bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total,87176)==-1);
    memcpy(raw+offsets[12],saved+offsets[12],4);
    assert(probe.calls==0&&!memcmp(raw,saved,total));
    /* The directory's shard index, rather than the primary file, owns payload. */
    unsigned char *shards[]={NULL,raw};size_t lengths[]={0,total},data_offsets[]={0,0};
    file.n_shards=2;file.shard_raws=shards;file.shard_raw_sizes=lengths;
    file.shard_data_offsets=data_offsets;file.raw=NULL;file.raw_size=0;
    for(int i=0;i<15;++i)tensors[i].shard_idx=1;
    assert(!bn_backend_layout_bf16_kda_init(&w,&gpu,&file,a,0,total,87176));
    bn_backend_layout_bf16_kda_free(&w);assert(probe.calls==8&&probe.freed==8);
    assert(!memcmp(raw,saved,total));free(raw);free(saved);free(a);
    puts("  KDA checkpoint layouts, budgets, shard bounds and upload failure cleanup PASSED");
}

#include "../src/transformer/bf16_internal.h"
typedef struct {int calls,fail;} KDAExecuteProbe;
static int kda_step(KDAExecuteProbe *p,int expected) {
    assert(++p->calls==expected);return p->fail==expected?-1:0;
}
static int kda_linear_probe(void *ctx,void *request,const BnGPUBF16LinearResidentPlan *p) {
    KDAExecuteProbe *q=ctx;assert(request==ctx&&p->n_tokens==3);
    int step=q->calls+1;
    if(step==1)assert(p->rows==641&&p->cols==128);
    else if(step==5||step==6)assert(p->rows==128&&p->cols==128);
    else assert(step==10&&p->rows==128&&p->cols==128);
    return kda_step(q,step);
}
static int kda_split_probe(void *ctx,void *request,const BnGPUSplitColumnsResidentPlan *p) {
    KDAExecuteProbe *q=ctx;assert(request==ctx&&p->rows==3);
    int step=q->calls+1;assert(step>=2&&step<=4);
    assert(p->cols==(step==2?641:(step==3?257:256)));
    assert(p->left_cols==(step==2?384:(step==3?1:128)));
    return kda_step(q,step);
}
static int kda_conv_probe(void *ctx,void *request,const BnGPUCausalConvResidentPlan *p) {
    assert(request==ctx&&p->dim==384&&p->kernel==4&&p->n_tokens==3);
    return kda_step(ctx,7);
}
static int kda_recurrent_probe(void *ctx,void *request,const BnGPUKDARecurrentResidentPlan *p) {
    assert(request==ctx&&p->num_heads==1&&p->head_dim==128&&p->n_tokens==3);
    assert(p->lower_bound==-5&&p->parameter_storage==BN_GPU_KDA_A_NEGATIVE_EXP);
    assert(p->q_scale==0x1.6a09e6p-4f);return kda_step(ctx,8);
}
static int kda_norm_probe(void *ctx,void *request,const BnGPUGatedRMSNormResidentPlan *p) {
    assert(request==ctx&&p->rows==3&&p->dim==128&&p->eps==1e-5f);
    assert(p->activation==BN_GPU_GATED_NORM_SIGMOID&&p->profile==BN_GPU_GATED_NORM_GLM_R2);
    return kda_step(ctx,9);
}
static void test_kda_resident_orchestration(void) {
    KDAExecuteProbe probe={0};BnGPUBackend gpu={0};gpu.ctx=&probe;
    gpu.linear_a8_bf16_resident=kda_linear_probe;gpu.split_columns_resident=kda_split_probe;
    gpu.causal_conv_f32_weight_bf16_resident=kda_conv_probe;
    gpu.kda_recurrent_bf16_resident=kda_recurrent_probe;
    gpu.gated_rmsnorm_f32_weight_bf16_resident=kda_norm_probe;
    BnBackendBF16KDAWeights w={.gpu=&gpu,.dim=128,.heads=1,.head_dim=128,.rank=128,
        .kernel=4,.lower_bound=-5,.output_eps=1e-5f,.conv=&gpu,.decay=&gpu,.bias=&gpu,.norm=&gpu};
    w.in=w.f_b=w.g_b=w.out=(BnGPUBF16LinearWeight){.raw=&gpu,.type=BN_GGUF_TENSOR_Q8_0};
    size_t counts[]={384,1152,1152,16384,16384,384,1923,1152,771,768,3,384,384,384,384,1152,384,384};
    BnGPUResidentView v[18];for(int i=0;i<18;++i)
        v[i]=(BnGPUResidentView){.request=&probe,.slot=(uint32_t)i,.count=counts[i],
            .type=i==3||i==4?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_BF16};
    BnTransformerBF16KDAWorkspace s={v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17]};
    assert(!bn_transformer_bf16_kda_layer(&w,&probe,3,&s,v[0],v[1],v[2],v[3],v[4],v[5]));
    assert(probe.calls==10);
    for(int fail=1;fail<=10;++fail) {
        probe=(KDAExecuteProbe){.fail=fail};
        assert(bn_transformer_bf16_kda_layer(&w,&probe,3,&s,v[0],v[1],v[2],v[3],v[4],v[5])==-1);
        assert(probe.calls==fail);
    }
    probe=(KDAExecuteProbe){0};BnTransformerBF16KDAWorkspace saved=s;
    s.gated.count--;assert(bn_transformer_bf16_kda_layer(&w,&probe,3,&s,v[0],v[1],v[2],v[3],v[4],v[5])==-1);
    s=saved;s.g1=s.g2;assert(bn_transformer_bf16_kda_layer(&w,&probe,3,&s,v[0],v[1],v[2],v[3],v[4],v[5])==-1);
    s=saved;s.gated.request=&gpu;assert(bn_transformer_bf16_kda_layer(&w,&probe,3,&s,v[0],v[1],v[2],v[3],v[4],v[5])==-1);
    s=saved;assert(bn_transformer_bf16_kda_layer(&w,&probe,3,&s,v[0],v[1],v[2],v[3],v[3],v[5])==-1);
    assert(!probe.calls);puts("  KDA resident ordering, preflight and submission failures PASSED");
}
