#include "gpu_cuda.h"
#include "backend_session.h"
#include "backend_layout.h"
#include "model_arch.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void*load(const char*d,const char*n,size_t bytes){char p[1024];snprintf(p,sizeof(p),"%s/%s.bin",d,n);FILE*f=fopen(p,"rb");assert(f);void*b=malloc(bytes);assert(b);assert(fread(b,1,bytes,f)==bytes&&fgetc(f)==EOF);fclose(f);return b;}
static BnGPUResidentView allocv(BnGPUBackend*g,void*r,int type,size_t n,const void*b){BnGPUResidentView v={0};assert(!g->resident_alloc(r,type,n,&v));if(b)assert(!g->resident_write(r,v,b));return v;}
static size_t check(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*d,const char*n){size_t word=v.type==BN_GGUF_TENSOR_BF16?2:4;void*expected=load(d,n,v.count*word),*actual=malloc(v.count*word);assert(actual);assert(!g->resident_read(r,v,actual));for(size_t i=0;i<v.count;++i)if(memcmp((char*)actual+i*word,(char*)expected+i*word,word)){uint32_t a=0,b=0;memcpy(&a,(char*)actual+i*word,word);memcpy(&b,(char*)expected+i*word,word);fprintf(stderr,"%s word%zu got%08x expected%08x\n",n,i,a,b);exit(3);}free(actual);free(expected);return v.count;}
#include "glm_router_edge_checks.h"
int main(int argc,char**argv){assert(argc==3);BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModelAnatomy*a=malloc(sizeof(*a));assert(a);assert(!bn_model_arch_plan_anatomy(f,a));BnGPUBackend*g=bn_gpu_cuda_create();assert(g);glm_router_edge_checks(g);BnBackendBF16DenseFFNWeights sw={0};assert(!bn_backend_layout_bf16_shared_ffn_init(&sw,g,f,a,3,26738688,17825792));assert(sw.dim==4096&&sw.hidden_dim==2048&&sw.clamp_limit==10);
 int ri=a->layers[3].tensor[BN_MODEL_TENSOR_MOE_ROUTER],bi=a->layers[3].tensor[BN_MODEL_TENSOR_MOE_ROUTER_BIAS];assert(ri>=0&&bi>=0);void*router=bn_gpu_backend_create_quant_only_buffer(g,bn_gguf_tensor_data(f,ri),4718592,BN_GGUF_TENSOR_F32,288,4096),*bias=bn_gpu_backend_create_quant_only_buffer(g,bn_gguf_tensor_data(f,bi),1152,BN_GGUF_TENSOR_F32,1,288);assert(router&&bias);size_t words=0;int tokens[]={1,2,6,7,9,64,65,129,1024,1025,2048};
 for(int ci=0;ci<11;++ci){int nt=tokens[ci];size_t n=nt;char dir[1024];snprintf(dir,sizeof(dir),"%s/%d",argv[2],ci);void*x=load(dir,"x",n*4096*2);BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);assert(!g->linear_bf16_reserve(g->ctx,r,n*4096));
 BnGPUResidentView xv=allocv(g,r,BN_GGUF_TENSOR_BF16,n*4096,x),cast=allocv(g,r,BN_GGUF_TENSOR_F32,n*4096,NULL),logits=allocv(g,r,BN_GGUF_TENSOR_F32,n*288,NULL),weights=allocv(g,r,BN_GGUF_TENSOR_F32,n*8,NULL),ids=allocv(g,r,BN_GGUF_TENSOR_I32,n*8,NULL),gu=allocv(g,r,BN_GGUF_TENSOR_BF16,n*4096,NULL),act=allocv(g,r,BN_GGUF_TENSOR_BF16,n*2048,NULL),out=allocv(g,r,BN_GGUF_TENSOR_BF16,n*4096,NULL);
 BnGPUBF16F32LinearResidentPlan linear={.n_tokens=nt,.rows=288,.cols=4096,.weight=router,.x=xv,.cast=cast,.out=logits};assert(!g->linear_bf16_f32_weight_resident(g->ctx,r,&linear));words+=check(g,r,logits,dir,"logits");
 BnGPUBiasedSigmoidF32ResidentPlan top={.n_tokens=nt,.n_experts=288,.k=8,.route_scale=2.5f,.bias=bias,.logits=logits,.weights=weights,.indices=ids};assert(!g->moe_biased_sigmoid_f32_resident(g->ctx,r,&top));words+=check(g,r,weights,dir,"weights");words+=check(g,r,ids,dir,"ids");
 assert(!bn_transformer_bf16_dense_ffn(&sw,r,nt,xv,gu,act,out));words+=check(g,r,gu,dir,"shared_gate_up");words+=check(g,r,act,dir,"shared_activated");words+=check(g,r,out,dir,"shared_out");printf("T%d raw-F32 router/top8/shared PASSED\n",nt);fflush(stdout);free(x);bn_backend_session_free(s);
 }printf("PASSED %zu exact router/shared words\n",words);bn_gpu_backend_destroy_buffer(g,router);bn_gpu_backend_destroy_buffer(g,bias);bn_backend_layout_bf16_dense_ffn_free(&sw);bn_gpu_cuda_destroy(g);bn_gguf_free(f);free(a);return 0;}
