#define main glm_indexer_existing_tests_main
#include "test_cuda_glm_indexer.c"
#undef main
#include "cuda_glm_mla_attention_reference.h"
static size_t mla_kernel_cases(BnGPUBackend*g,const char*directory){
 size_t total=0;
 for(size_t ci=0;ci<sizeof(glm_mla_attention_reference)/sizeof(glm_mla_attention_reference[0]);++ci){
  BnGLMMLAAttentionReference c=glm_mla_attention_reference[ci];int nt=c.tokens,end=c.position+nt;size_t n=(size_t)nt,bytes=(size_t)c.blocks*64*656,split=nt<=64?n*64*34:0;
  uint16_t*kv=malloc((size_t)end*512*2),*query=malloc(n*64*512*2);int32_t*slots=malloc((size_t)end*4),*table=malloc((size_t)c.blocks*4),*selection=malloc(n*2051*4);unsigned char*initial=malloc(bytes);assert(kv&&query&&slots&&table&&selection&&initial);memset(initial,0xa5,bytes);
  for(size_t i=0;i<(size_t)end*512;++i)kv[i]=bf16((float)((int)((i*37+i/13)%509)-254)/128);
  for(size_t i=0;i<n*64*512;++i)query[i]=bf16((float)((int)((i*17+i/11)%251)-125)/64);
  for(int page=0;page<c.blocks;++page)table[page]=c.blocks-1-page;
  for(int t=0;t<end;++t)slots[t]=table[t/64]*64+t%64;
  for(int row=0;row<nt;++row){int visible=c.position+row+1;for(int col=0;col<2051;++col)selection[(size_t)row*2051+col]=visible<=2051?(col<visible?col:-1):(col<2048?(col*37)%2048:visible-3+col-2048);}
  BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);
  BnGPUMlaFP8CacheAppendResidentPlan append={.n_tokens=end,.num_cache_blocks=c.blocks,.kv=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)end*512,kv),.slots=alloc(g,r,BN_GGUF_TENSOR_I32,end,slots),.cache=alloc(g,r,BN_GGUF_TENSOR_I8,bytes,initial)};int32_t status=0;
  BnGPUMlaFP8AttentionResidentPlan p={.n_tokens=nt,.position=c.position,.num_cache_blocks=c.blocks,.logical_blocks=c.blocks,.query=alloc(g,r,BN_GGUF_TENSOR_BF16,n*64*512,query),.cache=append.cache,.selection=alloc(g,r,BN_GGUF_TENSOR_I32,n*2051,selection),.block_table=alloc(g,r,BN_GGUF_TENSOR_I32,c.blocks,table),.physical=alloc(g,r,BN_GGUF_TENSOR_I32,n*2176,NULL),.out=alloc(g,r,BN_GGUF_TENSOR_BF16,n*64*512,NULL),.lse=alloc(g,r,BN_GGUF_TENSOR_F32,n*64,NULL),.partial=alloc(g,r,BN_GGUF_TENSOR_BF16,split*512,NULL),.partial_lse=alloc(g,r,BN_GGUF_TENSOR_F32,split,NULL),.status=alloc(g,r,BN_GGUF_TENSOR_I32,1,&status)},bad;
#define REJECT(change) do{bad=p;change;assert(g->mla_fp8_attention_resident(g->ctx,r,&bad)==-1);}while(0)
  REJECT(bad.n_tokens=0);REJECT(bad.position=-1);REJECT(bad.position=INT_MAX);REJECT(bad.physical.count--);REJECT(bad.out=bad.query);REJECT(bad.status.type=BN_GGUF_TENSOR_F32);if(!split){REJECT(bad.partial.request=r);REJECT(bad.partial_lse.count=1);}
#undef REJECT
  assert(!g->mla_fp8_cache_append_resident(g->ctx,r,&append));assert(!g->mla_fp8_attention_resident(g->ctx,r,&p));assert(!g->resident_read(r,p.status,&status)&&!status);
  total+=check(g,r,p.cache,1,c.cache);total+=check(g,r,p.out,2,c.out);total+=check(g,r,p.lse,4,c.lse);total+=check(g,r,p.physical,4,c.physical);
  if(directory){char dir[1024];snprintf(dir,sizeof(dir),"%s/%d-%d",directory,c.position,nt);compare_file(g,r,p.cache,1,dir,"cache","");compare_file(g,r,p.out,2,dir,"out","");compare_file(g,r,p.lse,4,dir,"lse","");compare_file(g,r,p.physical,4,dir,"physical","");}
  if(!ci){selection[0]=1;assert(!g->resident_write(r,p.selection,selection));assert(!g->mla_fp8_attention_resident(g->ctx,r,&p));assert(!g->resident_read(r,p.status,&status)&&status==1);selection[0]=0;table[0]=c.blocks;assert(!g->resident_write(r,p.selection,selection));assert(!g->resident_write(r,p.block_table,table));assert(!g->mla_fp8_attention_resident(g->ctx,r,&p));assert(!g->resident_read(r,p.status,&status)&&status==1);int32_t*out=malloc(n*2176*4);assert(out&&!g->resident_read(r,p.physical,out));for(size_t col=0;col<n*2176;++col)assert(out[col]==-1);free(out);}
  bn_backend_session_free(s);free(kv);free(query);free(slots);free(table);free(selection);free(initial);
 }return total;
}
int main(int argc,char**argv){assert(argc==1||argc==2);BnGPUBackend*g=bn_gpu_cuda_create();assert(g);
#ifndef BN_CUDA_MXFP4_SM120
if(!g->mla_fp8_attention_resident){puts("CUDA GLM sparse MLA skipped: requires SM120 reference build");bn_gpu_cuda_destroy(g);return 0;}
#endif
assert(g->mla_fp8_cache_append_resident&&g->mla_fp8_attention_resident);size_t n=mla_kernel_cases(g,argc==2?argv[1]:NULL);bn_gpu_cuda_destroy(g);printf("CUDA model-free GLM sparse MLA PASSED: %zu reference elements\n",n);return 0;}
