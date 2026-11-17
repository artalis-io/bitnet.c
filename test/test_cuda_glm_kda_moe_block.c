/* Complete KDA/MoE from the independent original deferred boundary. Layer44
 * is replayed through its FFN tuple; final contraction/logits are outside scope. */
#define main glm_deferred_attention_reference_main
#include "test_cuda_glm_deferred_kda.c"
#undef main
#include "moe.h"
#include <threads.h>
extern int cudaProfilerStart(void);
extern int cudaProfilerStop(void);
static size_t check_routed(BnGPUBackend*g,void*r,const BnTransformerBF16MoEPlan*p,
    const char*dir,int c,int phase) {
    size_t n=(size_t)p->n_tokens*8,h=p->weights->hidden_dim,d=p->weights->dim;
    int32_t*inv=malloc(n*4);uint16_t*gate=malloc(n*h*2),*up=malloc(n*h*2),*down=malloc(n*d*2);assert(inv&&gate&&up&&down);
    assert(!g->resident_read(r,p->inverse,inv)&&!g->resident_read(r,p->gate,gate)&&!g->resident_read(r,p->up,up)&&!g->resident_read(r,p->down,down));
    char name[128];snprintf(name,sizeof(name),"%d-%d-moe_routed_gate_up",c,phase);uint16_t*gu=load(dir,name,n*h*4);
    snprintf(name,sizeof(name),"%d-%d-moe_routed_activated",c,phase);uint16_t*act=load(dir,name,n*h*2);
    snprintf(name,sizeof(name),"%d-%d-moe_routed_down",c,phase);uint16_t*rd=load(dir,name,n*d*2);
    for(size_t i=0;i<n;++i){assert(inv[i]>=0&&(size_t)inv[i]<n);size_t row=(size_t)inv[i];
        assert(!memcmp(gate+row*h,act+i*h,h*2));assert(!memcmp(up+row*h,gu+i*h*2+h,h*2));assert(!memcmp(down+row*d,rd+i*d,d*2));}
    free(inv);free(gate);free(up);free(down);free(gu);free(act);free(rd);return n*(2*h+d);
}
int main(int argc,char**argv) {
    assert(argc==3);BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModelAnatomy*a=malloc(sizeof(*a));assert(a);assert(!bn_model_arch_plan_anatomy(f,a));BnGPUBackend*g=bn_gpu_cuda_create();assert(g);
    int layer=44;char dir[1024];snprintf(dir,sizeof(dir),"%s/layer%d",argv[2],layer);
    BnBackendBF16KDAWeights kw={0};BnBackendBF16MHCWeights aw={0},fw={0};BnBackendBF16MoEWeights mw={0};
    assert(!bn_backend_layout_bf16_kda_init(&kw,g,f,a,layer,512u*1024*1024,256u*1024*1024)&&kw.layer==layer);
    assert(!bn_backend_layout_bf16_mhc_init(&aw,g,f,a,layer,0,4u*1024*1024,2u*1024*1024));assert(!bn_backend_layout_bf16_mhc_init(&fw,g,f,a,layer,1,4u*1024*1024,2u*1024*1024));
    BnModelLayerAnatomy*l=&a->layers[layer];BnMoEExpertTensorNames names={.gate=f->tensors[l->tensor[BN_MODEL_TENSOR_MOE_GATE_EXPS]].name,.up=f->tensors[l->tensor[BN_MODEL_TENSOR_MOE_UP_EXPS]].name,.down=f->tensors[l->tensor[BN_MODEL_TENSOR_MOE_DOWN_EXPS]].name};
    BnMoEExpertMap map={0};assert(!bn_moe_load_expert_map(f,&names,288,2048,&map));assert(!bn_backend_layout_bf16_moe_init(&mw,g,f,a,layer,&map,31458432,17825792));
    assert(f->n_shards&&f->owned_maps);int*fds=calloc(f->n_shards,sizeof(*fds));assert(fds);for(size_t i=0;i<f->n_shards;++i){fds[i]=f->owned_maps[i].fd;assert(fds[i]>=0);}
    BnMoEIO io={.fd=-1,.shard_fds=fds,.n_shard_fds=f->n_shards};io.cache=bn_moe_cache_create(128u*1024*1024,mw.gate_bytes,mw.gate_bytes,mw.down_bytes);assert(io.cache);
    int dim=kw.dim,p=kw.heads*128,merged=3*p+kw.heads+256;size_t history=(size_t)9*p,state=(size_t)p*128;
    void *oldres=load(dir,"old_residual",9u*4*dim*2),*oldbranch=load(dir,"old_branch",9u*dim*2),*oldpost=load(dir,"old_post",9*4*4),*oldcomb=load(dir,"old_comb",9*16*4),*h0=load(dir,"history0",history*2),*s0=load(dir,"initial",state*4);
    int schedules[6][9]={{9},{2,7},{3,3,3},{7,1,1},{6,3},{1,1,1,1,1,1,1,1,1}},lengths[]={1,2,3,3,2,9};size_t words=0;
    for(int c=0;c<6;++c) {
        BnBackendSession*session=bn_backend_session_create();assert(session);void*r=bn_gpu_backend_resident_request(g,session);assert(r);
        assert(!g->linear_a8_bf16_reserve(g->ctx,r,9u*p));assert(!g->linear_bf16_reserve(g->ctx,r,9u*dim));
        size_t maxgate,maxdown;assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_Q5_K,(uint64_t)dim*mw.hidden_dim,&maxgate));assert(bn_gguf_tensor_size(BN_GGUF_TENSOR_Q8_0,(uint64_t)dim*mw.hidden_dim,&maxdown));
        BnGPUMoEBankConfig cfg={.slots=2,.experts_per_slot=3,.dim=dim,.hidden_dim=mw.hidden_dim,.host_budget_bytes=6*(2*maxgate+maxdown),.device_budget_bytes=6*(2*maxgate+maxdown),.n_experts=288,.k=8};assert(!g->moe_bank_reserve(g->ctx,r,&cfg));
        assert(!g->moe_bank_input_reserve(g->ctx,r,9,9u*(dim/32)*36*9));size_t meta=6*sizeof(BnGPUMoEBankItem);assert(!g->moe_bank_grouped_reserve(g->ctx,r,9,meta,9u*8*(mw.hidden_dim/32)*36+meta));
        BnGPUResidentView pr=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*4*dim,oldres),pb=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*dim,oldbranch),pp=allocv(g,r,BN_GGUF_TENSOR_F32,9*4,oldpost),pc=allocv(g,r,BN_GGUF_TENSOR_F32,9*16,oldcomb);
        BnGPUResidentView ih=allocv(g,r,BN_GGUF_TENSOR_BF16,history,h0),nh=allocv(g,r,BN_GGUF_TENSOR_BF16,history,NULL),is=allocv(g,r,BN_GGUF_TENSOR_F32,state,s0),ns=allocv(g,r,BN_GGUF_TENSOR_F32,state,NULL),branch=allocv(g,r,BN_GGUF_TENSOR_BF16,9u*dim,NULL);
        BnGPUMHCPreResidentPlan attn=mhc_workspace(g,r,&aw),ffn=mhc_workspace(g,r,&fw);
        size_t sizes[]={9u*merged,9u*3*p,9u*(kw.heads+256),9u*256,9u*kw.heads,9u*128,9u*128,9u*p,9u*p,9u*3*p,9u*p,9u*p};BnGPUResidentView workspace[12];for(int i=0;i<12;++i)workspace[i]=allocv(g,r,BN_GGUF_TENSOR_BF16,sizes[i],NULL);
        size_t ms[]={9u*dim,9u*288,9u*8,9u*8,293,288,9u*8,9u*8,9u*8*dim,9u*8*2048,9u*8*2048,9u*8*dim,9u*8*dim,9u*dim,9u*4096,9u*2048,9u*dim,9u*dim};
        BnGPUResidentView mo[18];for(int i=0;i<18;++i){int type=i<=2?BN_GGUF_TENSOR_F32:i<=7?BN_GGUF_TENSOR_I32:BN_GGUF_TENSOR_BF16;mo[i]=allocv(g,r,type,ms[i],NULL);}
        int off=0;for(int phase=0;phase<lengths[c];++phase) {
            int t=schedules[c][phase],sp=g->mhc_fused_bf16_split_count(g->ctx,t,dim);BnGPUResidentView v[12],mv[18];for(int i=0;i<12;++i)v[i]=slice(workspace[i],0,sizes[i]/9*t);
            for(int i=0;i<18;++i)mv[i]=slice(mo[i],0,i==4||i==5?ms[i]:ms[i]/9*t);
            BnTransformerBF16KDAWorkspace s={v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11]};BnMoEState host={0};
            BnTransformerBF16KDAMoEBlockPlan block={.attention={.weights=&kw,.mhc=mhc_take(attn,t,sp),.kda=s,.initial_history=ih,.next_history=nh,.initial_state=is,.next_state=ns,.branch_out=slice(branch,0,(size_t)t*dim),.deferred=1,.has_previous=1,
                .previous_residual=slice(pr,(size_t)off*4*dim,(size_t)t*4*dim),.previous_branch=slice(pb,(size_t)off*dim,(size_t)t*dim),.previous_post=slice(pp,(size_t)off*4,(size_t)t*4),.previous_comb=slice(pc,(size_t)off*16,(size_t)t*16)},.ffn=mhc_take(ffn,t,sp)};
            block.moe=(BnTransformerBF16MoEPlan){.weights=&mw,.source=&io,.host_state=&host,.layer=layer,.n_tokens=t,.prefill=t>1,.bank_capacity=3,.x=block.ffn.out,
                .cast=mv[0],.logits=mv[1],.route_weights=mv[2],.indices=mv[3],.metadata=mv[4],.cursors=mv[5],.slots=mv[6],.inverse=mv[7],.gathered=mv[8],.gate=mv[9],.up=mv[10],.down=mv[11],.weighted=mv[12],.routed=mv[13],.shared_gate_up=mv[14],.shared_activated=mv[15],.shared_out=mv[16],.out=mv[17]};
            BnTransformerBF16KDAMoEBlockWork work={0};int trace=c==5&&phase==1&&getenv("BN_KDA_MOE_TRACE");
            if(trace){assert(!bn_transformer_bf16_kda_moe_begin(r,&block,&work));int warm;
                while((warm=bn_transformer_bf16_kda_moe_submit(r,&work))==BN_GPU_MOE_BANK_BUSY)thrd_yield();
                assert(!warm&&!g->resident_synchronize(r));work=(BnTransformerBF16KDAMoEBlockWork){0};host.stats=(BnMoEStats){0};assert(!cudaProfilerStart());}
            assert(!bn_transformer_bf16_kda_moe_begin(r,&block,&work));int status;while((status=bn_transformer_bf16_kda_moe_submit(r,&work))==BN_GPU_MOE_BANK_BUSY)thrd_yield();assert(!status&&!work.pending&&!work.failed&&work.moe.metadata_read_bytes==1172);assert(!g->resident_synchronize(r));
            if(trace){assert(!cudaProfilerStop());assert(host.stats.gpu_host_cache_hits==8&&!host.stats.gpu_host_cache_misses&&!host.stats.io_bytes);}
            BnGPUResidentView outputs[]={block.attention.mhc.out,block.attention.mhc.residual,block.attention.mhc.post,block.attention.mhc.comb,s.projected,s.g1,s.g2,s.conv,s.core,s.gated,block.attention.branch_out,ns,nh,block.ffn.residual,block.ffn.post,block.ffn.comb,block.ffn.out};
            const char*names2[]={"input","entry_residual","entry_post","entry_comb","projected","g1","g2","conv","core","gated","out","state","history","ffn_residual","ffn_post","ffn_comb","ffn_input"};for(int i=0;i<17;++i)words+=check(g,r,outputs[i],dir,c,phase,names2[i]);
            BnGPUResidentView mout[]={mv[1],mv[2],mv[3],mv[14],mv[15],mv[16],mv[12],mv[13],mv[17]};const char*mn[]={"moe_logits","moe_weights","moe_ids","moe_shared_gate_up","moe_shared_activated","moe_shared_out","moe_weighted","moe_routed_out","moe_out"};for(int i=0;i<9;++i)words+=check(g,r,mout[i],dir,c,phase,mn[i]);words+=check_routed(g,r,&block.moe,dir,c,phase);
            BnGPUMoEBankStats stats={0};assert(!g->moe_bank_stats(r,&stats));assert(stats.upload_bytes>=work.moe.submitted_experts*(2*mw.gate_bytes+mw.down_bytes));assert(!host.buf&&!host.buf2&&!host.buf5);
            /* Commit both generations only after the complete block fence and checks. */
            BnGPUResidentView tmp=ih;ih=nh;nh=tmp;tmp=is;is=ns;ns=tmp;off+=t;
            printf("KDA/MoE layer%d case%d phase%d tokens%d PASSED experts%zu reads%zu hits%zu\n",layer,c,phase,t,work.moe.submitted_experts,host.stats.io_bytes,host.stats.gpu_host_cache_hits);fflush(stdout);
        }
        bn_backend_session_free(session);
    }
    printf("Complete KDA/MoE PASSED: %zu exact reference elements\n",words);
    free(oldres);free(oldbranch);free(oldpost);free(oldcomb);free(h0);free(s0);bn_moe_cache_free(io.cache);free(fds);
    bn_backend_layout_bf16_moe_free(&mw);bn_backend_layout_bf16_mhc_free(&aw);bn_backend_layout_bf16_mhc_free(&fw);bn_backend_layout_bf16_kda_free(&kw);bn_gpu_cuda_destroy(g);bn_gguf_free(f);free(a);return 0;
}
