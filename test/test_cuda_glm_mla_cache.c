#define main glm_indexer_existing_tests_main
#include "test_cuda_glm_indexer.c"
#undef main
#include "cuda_glm_mla_cache_reference.h"
static size_t mla_cases(BnGPUBackend *g,const char *directory){
 size_t bytes_checked=0;int exponents[]={-30,-16,-4,0,4,12,24,0};
 for(size_t ci=0;ci<sizeof(glm_mla_cache_reference)/sizeof(glm_mla_cache_reference[0]);++ci){
  BnGLMMLACacheReference c=glm_mla_cache_reference[ci];size_t n=(size_t)c.tokens,bytes=(size_t)c.blocks*64*656;
  uint16_t*x=malloc((n?n:1)*1024);int32_t*slots=malloc((n?n:1)*4);unsigned char*initial=malloc(bytes);assert(x&&slots&&initial);memset(initial,0xa5,bytes);
  for(size_t row=0;row<n;++row){slots[row]=(int)((row*37+c.seed*11)%(c.blocks*64));if(row%11==3||row%17==5)slots[row]=-1;
   for(int d=0;d<512;++d){size_t i=row*512+d;float v=ldexpf((float)((int)((i*37+i/13+c.seed*11)%509)-254)/128,exponents[(row+c.seed)%8]);if(row==0)v=0;if(row==1)v=-0.f;if(row==2)v=1;x[i]=bf16(v);}}
  for(int mode=0;mode<3;++mode){BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);
   BnGPUMlaFP8CacheAppendResidentPlan p={.n_tokens=c.tokens,.num_cache_blocks=c.blocks,.kv=alloc(g,r,BN_GGUF_TENSOR_BF16,n*512,x),.slots=alloc(g,r,BN_GGUF_TENSOR_I32,n,slots),.cache=alloc(g,r,BN_GGUF_TENSOR_I8,bytes,initial)};
   int chunk=mode==0?INT_MAX:mode==1?1:17;
   if(!n)assert(!g->mla_fp8_cache_append_resident(g->ctx,r,&p));
   for(int pos=0;pos<c.tokens;){int take=c.tokens-pos;if(take>chunk)take=chunk;BnGPUMlaFP8CacheAppendResidentPlan part=p;part.n_tokens=take;part.kv=bn_gpu_resident_view_slice(p.kv,(size_t)pos*512,(size_t)take*512);part.slots=bn_gpu_resident_view_slice(p.slots,(size_t)pos,(size_t)take);assert(!g->mla_fp8_cache_append_resident(g->ctx,r,&part));pos+=take;if(mode==2)assert(!g->resident_synchronize(r));}
   bytes_checked+=check(g,r,p.cache,1,c.digest);char prefix[80];snprintf(prefix,sizeof(prefix),"mla-cache-%d-%d",c.tokens,c.seed);compare_file(g,r,p.cache,1,directory,prefix,"");bn_backend_session_free(s);
  }free(x);free(slots);free(initial);
 }return bytes_checked;
}
static void mla_guards(BnGPUBackend*g){
 BnBackendSession*s=bn_backend_session_create(),*other=bn_backend_session_create();assert(s&&other);void*r=bn_gpu_backend_resident_request(g,s),*peer=bn_gpu_backend_resident_request(g,other);uint16_t x[512]={0};int32_t slot=0;unsigned char initial[64*656];memset(initial,0xa5,sizeof(initial));
 BnGPUMlaFP8CacheAppendResidentPlan p={.n_tokens=1,.num_cache_blocks=1,.kv=alloc(g,r,BN_GGUF_TENSOR_BF16,512,x),.slots=alloc(g,r,BN_GGUF_TENSOR_I32,1,&slot),.cache=alloc(g,r,BN_GGUF_TENSOR_I8,sizeof(initial),initial)},b;
 BnGPUResidentView raw=alloc(g,r,BN_GGUF_TENSOR_I8,sizeof(initial)+1,NULL);
#define REJECT(change) do{b=p;change;assert(g->mla_fp8_cache_append_resident(g->ctx,r,&b)==-1);}while(0)
 REJECT(b.n_tokens=-1);REJECT(b.n_tokens=INT_MAX);REJECT(b.num_cache_blocks=0);REJECT(b.num_cache_blocks=INT_MAX);REJECT(b.kv.count--);REJECT(b.slots.count++);REJECT(b.cache.count--);REJECT(b.cache=bn_gpu_resident_view_slice(raw,1,sizeof(initial)));REJECT(b.cache.request=peer);REJECT(b.kv.request=peer);REJECT(b.slots=p.cache);REJECT(b.slots.type=BN_GGUF_TENSOR_F32);
#undef REJECT
 check(g,r,p.cache,1,hash(initial,sizeof(initial)));for(int i=0;i<2;++i){slot=i?64:-1;assert(!g->resident_write(r,p.slots,&slot));assert(!g->mla_fp8_cache_append_resident(g->ctx,r,&p));check(g,r,p.cache,1,hash(initial,sizeof(initial)));}
 b=p;b.n_tokens=0;b.kv=(BnGPUResidentView){0};b.slots=(BnGPUResidentView){0};assert(!g->mla_fp8_cache_append_resident(g->ctx,r,&b));bn_backend_session_free(other);bn_backend_session_free(s);
}
int main(int argc,char**argv){assert(argc==1||argc==2);BnGPUBackend*g=bn_gpu_cuda_create();assert(g&&g->mla_fp8_cache_append_resident);mla_guards(g);size_t n=mla_cases(g,argc==2?argv[1]:NULL);bn_gpu_cuda_destroy(g);printf("CUDA GLM MLA packed cache PASSED: %zu reference bytes\n",n);return 0;}
