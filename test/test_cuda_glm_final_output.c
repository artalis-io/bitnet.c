#define main glm_final_deferred_helpers_main
#include "test_cuda_glm_deferred_kda.c"
#undef main
#include "backend_model.h"
#include "moe.h"
extern int cudaProfilerStart(void);
extern int cudaProfilerStop(void);
static size_t final_check(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*d,const char*n){
    size_t word=v.type==BN_GGUF_TENSOR_BF16?2:4;void*expected=load(d,n,v.count*word),*actual=malloc(v.count*word);assert(actual);
    assert(!g->resident_read(r,v,actual));size_t bad=0;
    for(size_t i=0;i<v.count;++i)if(memcmp((char*)actual+i*word,(char*)expected+i*word,word)){if(bad<3)fprintf(stderr,"%s word%zu mismatch\n",n,i);bad++;}
    if(bad){fprintf(stderr,"%s mismatches %zu/%zu\n",n,bad,v.count);exit(3);}free(expected);free(actual);return v.count;
}
int main(int argc,char**argv){
    assert(argc==3);BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModelAnatomy*a=malloc(sizeof(*a));assert(a);assert(!bn_model_arch_plan_anatomy(f,a));
    BnMoEExpertMap*maps=calloc((size_t)a->n_layers,sizeof(*maps));assert(maps);
    for(int i=0;i<a->n_layers;++i)if(a->layers[i].uses_moe){const BnModelLayerAnatomy*l=&a->layers[i];BnMoEExpertTensorNames names={.gate=f->tensors[l->tensor[BN_MODEL_TENSOR_MOE_GATE_EXPS]].name,.up=f->tensors[l->tensor[BN_MODEL_TENSOR_MOE_UP_EXPS]].name,.down=f->tensors[l->tensor[BN_MODEL_TENSOR_MOE_DOWN_EXPS]].name};assert(!bn_moe_load_expert_map(f,&names,288,2048,&maps[i]));}
    BnGPUBackend*g=bn_gpu_cuda_create();assert(g);BnBackendModel*b=bn_backend_model_create();assert(b);
    assert(!bn_backend_layout_prepare_bf16_glm_model(b,g,f,a,maps,(size_t)a->n_layers,(size_t)12*1024*1024*1024,256u*1024*1024));
    const BnBackendBF16GLMModel*w=bn_backend_layout_bf16_glm_model(b);assert(w);size_t words=0;int tokens[]={1,9,255,256,257};
    for(int ci=0;ci<5;++ci){int nt=tokens[ci];size_t n=nt;char d[1024];snprintf(d,sizeof(d),"%s/%d",argv[2],ci);
        BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);assert(!g->linear_a8_bf16_reserve(g->ctx,r,n*4096));
        const char*names[]={"residual","branch","post","comb"};size_t counts[]={n*16384,n*4096,n*4,n*16};BnGPUResidentView v[8];
        for(int i=0;i<4;++i){int type=i>=2?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_BF16;void*data=load(d,names[i],counts[i]*(i>=2?4:2));v[i]=allocv(g,r,type,counts[i],data);free(data);}
        v[4]=allocv(g,r,BN_GGUF_TENSOR_BF16,n*16384,NULL);v[5]=allocv(g,r,BN_GGUF_TENSOR_BF16,n*4096,NULL);v[6]=allocv(g,r,BN_GGUF_TENSOR_BF16,n*4096,NULL);
        for(int last=0;last<2;++last){int head=last?1:nt;v[7]=allocv(g,r,BN_GGUF_TENSOR_BF16,(size_t)head*w->vocab_size,NULL);
            BnTransformerBF16GLMOutputPlan p={w,nt,last,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]};int trace=ci==0&&last&&getenv("BN_GLM_HEAD_TRACE");
            if(trace){assert(!bn_transformer_bf16_glm_output(r,&p)&&!g->resident_synchronize(r));assert(!cudaProfilerStart());}
            assert(!bn_transformer_bf16_glm_output(r,&p)&&!g->resident_synchronize(r));if(trace)assert(!cudaProfilerStop());
            words+=final_check(g,r,v[4],d,"completed");words+=final_check(g,r,v[5],d,"collapsed");words+=final_check(g,r,v[6],d,"normalized");words+=final_check(g,r,v[7],d,last?"logits_last":"logits_all");
            for(int i=0;i<4;++i)words+=final_check(g,r,v[i],d,names[i]);
            printf("GLM final output T%d last%d PASSED\n",nt,last);fflush(stdout);
        }
        bn_backend_session_free(s);
    }
    printf("GLM final output PASSED: %zu exact reference elements\n",words);
    bn_backend_model_free(b);bn_gpu_cuda_destroy(g);free(maps);free(a);bn_gguf_free(f);return 0;
}
