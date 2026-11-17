#define main mla_preindexer_existing_main
#include "test_cuda_glm_preindexer.c"
#undef main
#include <limits.h>
extern int cudaProfilerStart(void);
extern int cudaProfilerStop(void);
static void mla_attention_guards(BnGPUBackend*g,void*r,const BnGPUMlaFP8AttentionResidentPlan*p){
 BnGPUMlaFP8AttentionResidentPlan b;
#define BAD(change) do{b=*p;change;assert(g->mla_fp8_attention_resident(g->ctx,r,&b)==-1);}while(0)
 BAD(b.n_tokens=0);BAD(b.n_tokens=INT_MAX);BAD(b.position=-1);BAD(b.position=INT_MAX);BAD(b.logical_blocks=0);BAD(b.num_cache_blocks=INT_MAX);BAD(b.query.count--);BAD(b.physical.count--);BAD(b.partial.count--);BAD(b.partial_lse.count--);BAD(b.out=b.query);BAD(b.lse.type=BN_GGUF_TENSOR_I32);BAD(b.cache.offset=SIZE_MAX);BAD(b.query.request=&b);
#undef BAD
 int32_t status=-1;assert(!g->resident_read(r,p->status,&status)&&!status);
}
static void mla_attention_bad_indices(BnGPUBackend*g,void*r,const BnGPUMlaFP8AttentionResidentPlan*p){
 assert(p->n_tokens==1);int32_t selection[2051],table[3],physical[2176],status=0;assert(p->logical_blocks==3);assert(!g->resident_read(r,p->selection,selection));assert(!g->resident_read(r,p->block_table,table));
 int32_t original=selection[0],page=table[0];
 for(int kind=0;kind<5;++kind){selection[0]=kind==0?-2:kind==1?INT_MAX:kind==2?1:original;table[0]=kind==3?-1:kind==4?p->num_cache_blocks:page;status=0;assert(!g->resident_write(r,p->status,&status));assert(!g->resident_write(r,p->selection,selection));assert(!g->resident_write(r,p->block_table,table));assert(!g->mla_fp8_attention_resident(g->ctx,r,p));assert(!g->resident_read(r,p->status,&status)&&status==1);assert(!g->resident_read(r,p->physical,physical));for(int col=0;col<2176;++col)assert(physical[col]==-1);}
 selection[0]=original;table[0]=page;assert(!g->resident_write(r,p->selection,selection));assert(!g->resident_write(r,p->block_table,table));assert(!g->mla_fp8_attention_resident(g->ctx,r,p));assert(!g->resident_read(r,p->status,&status)&&status==1); /* sticky */
}
int main(int argc,char**argv){
 assert(argc==4);BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModelAnatomy*a=malloc(sizeof(*a));assert(a&&!bn_model_arch_plan_anatomy(f,a));BnGPUBackend*g=bn_gpu_cuda_create();assert(g&&g->mla_fp8_attention_resident&&g->indexer_select_resident);BnBackendBF16MLAWeights w={0};assert(!bn_backend_layout_bf16_mla_init(&w,g,f,a,3,150000000));int sizes[]={1,2,3,7,17,65,129,1025};size_t words=0,bytes=0;
 for(int ci=0;ci<8;++ci){int nt=sizes[ci],blocks=(nt+63)/64+2;size_t n=(size_t)nt,cache_bytes=(size_t)blocks*64*656,split=n<=64?n*64*34:0;char input[1024],ref[1024];snprintf(input,sizeof(input),"%s/%d",argv[2],ci);snprintf(ref,sizeof(ref),"%s/%d",argv[3],ci);void*x=load(input,"x",n*w.dim*2),*slots=load(ref,"slots",n*4),*table=load(ref,"table",(size_t)blocks*4);unsigned char*initial=malloc(cache_bytes);assert(initial);memset(initial,0xa5,cache_bytes);
  BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r&&!g->linear_bf16_reserve(g->ctx,r,n*w.heads*w.key_dim));size_t counts[]={n*w.dim,n*w.q_rank,n*w.q_rank,n*w.kv_rank,n*w.kv_rank,n*w.heads*w.key_dim,n*w.heads*w.kv_rank};BnGPUResidentView v[7];for(int j=0;j<7;++j)v[j]=alloc(g,r,BN_GGUF_TENSOR_BF16,counts[j],j?NULL:x);
  BnTransformerBF16MLACachePlan prepare={.prepare={.weights=&w,.layer=3,.n_tokens=nt,.x=v[0],.q_raw=v[1],.qr=v[2],.kv_raw=v[3],.kv=v[4],.q=v[5],.query=v[6]},.num_cache_blocks=blocks,.slots=alloc(g,r,BN_GGUF_TENSOR_I32,n,slots),.cache=alloc(g,r,BN_GGUF_TENSOR_I8,cache_bytes,initial)};
  int32_t status=0;BnGPUResidentView sv=alloc(g,r,BN_GGUF_TENSOR_I32,1,&status);
  BnGPUIndexerSelectResidentPlan select={.n_tokens=nt,.n_keys=nt/4,.short_prefill=1,.pools=alloc(g,r,BN_GGUF_TENSOR_I32,n*512,NULL),.out=alloc(g,r,BN_GGUF_TENSOR_I32,n*2051,NULL),.status=sv};
  BnTransformerBF16MLAAttentionPlan attention={.weights=&w,.layer=3,.attention={.n_tokens=nt,.num_cache_blocks=blocks,.logical_blocks=blocks,.query=v[6],.cache=prepare.cache,.selection=select.out,.block_table=alloc(g,r,BN_GGUF_TENSOR_I32,blocks,table),.physical=alloc(g,r,BN_GGUF_TENSOR_I32,n*2176,NULL),.out=alloc(g,r,BN_GGUF_TENSOR_BF16,n*64*512,NULL),.lse=alloc(g,r,BN_GGUF_TENSOR_F32,n*64,NULL),.partial=alloc(g,r,BN_GGUF_TENSOR_BF16,split*512,NULL),.partial_lse=alloc(g,r,BN_GGUF_TENSOR_F32,split,NULL),.status=sv},.values=alloc(g,r,BN_GGUF_TENSOR_BF16,n*64*256,NULL),.out=alloc(g,r,BN_GGUF_TENSOR_BF16,n*w.dim,NULL)};
  if(!ci)mla_attention_guards(g,r,&attention.attention);
  assert(!bn_transformer_bf16_mla_cache(r,&prepare));assert(!g->indexer_select_resident(g->ctx,r,&select));
  int trace=ci==6&&getenv("BN_MLA_TRACE");if(trace){assert(!bn_transformer_bf16_mla_attention(r,&attention));assert(!g->resident_synchronize(r));assert(!cudaProfilerStart());}
  assert(!bn_transformer_bf16_mla_attention(r,&attention));if(trace){assert(!g->resident_synchronize(r));assert(!cudaProfilerStop());}
  assert(!g->resident_read(r,sv,&status)&&!status);bytes+=check(g,r,prepare.cache,ref,"cache");words+=check(g,r,select.out,ref,"selection");words+=check(g,r,attention.attention.physical,ref,"physical");words+=check(g,r,attention.attention.out,ref,"context");words+=check(g,r,attention.attention.lse,ref,"lse");words+=check(g,r,attention.values,ref,"values");words+=check(g,r,attention.out,ref,"out");words+=check(g,r,v[6],input,"query");
  if(!ci)mla_attention_bad_indices(g,r,&attention.attention);
  bn_backend_session_free(s);free(x);free(slots);free(table);free(initial);printf("MLA attention chain case%d tokens%d PASSED\n",ci,nt);fflush(stdout);
 }printf("MLA attention chain PASSED: %zu exact reference words and %zu cache bytes\n",words,bytes);bn_backend_layout_bf16_mla_free(&w);bn_gpu_cuda_destroy(g);bn_gguf_free(f);free(a);return 0;
}
