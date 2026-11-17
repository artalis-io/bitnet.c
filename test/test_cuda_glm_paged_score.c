#define main glm_indexer_parent_main
#include "test_cuda_glm_indexer.c"
#undef main
#include "cuda_glm_paged_score_reference.h"
static unsigned char paged_byte(size_t i,int seed){return (unsigned char)((i*37+i/13+(size_t)seed*11)%127+((i*17+(size_t)seed)%3==0?128:0));}
static size_t paged_cases(BnGPUBackend*g,const char*directory){size_t total=0;
 for(size_t ci=0;ci<sizeof(glm_paged_score_reference)/sizeof(glm_paged_score_reference[0]);++ci){BnGLMPagedScoreReference c=glm_paged_score_reference[ci];assert(c.tokens>0&&c.tokens<=4096&&c.position>=0&&c.position<=INT_MAX-c.tokens);size_t n=(size_t)c.tokens,k=(size_t)(c.tokens+c.position)/4;int blocks=(int)((k+63)/64)+2;
  unsigned char*q=malloc(n*4096),*cache=malloc((size_t)blocks*8448);float*w=malloc(n*32*4),*out=malloc((n*k+1)*4);int32_t*table=malloc((size_t)blocks*4);assert(q&&cache&&w&&out&&table);memset(cache,0xa5,(size_t)blocks*8448);
  for(size_t i=0;i<n*4096;++i)q[i]=paged_byte(i,c.seed);
  for(size_t i=0;i<n*32;++i)w[i]=(float)((int)((i*17+(size_t)c.seed*13)%251)-125)/128+0x1p-17f;
  for(int j=0;j<blocks;++j)table[j]=blocks-1-j;
  for(size_t row=0;row<k;++row){size_t base=(size_t)table[row/64]*8448;for(int d=0;d<128;++d)cache[base+(row%64)*128+d]=paged_byte(row*128+d,c.seed+17);float scale=ldexpf(1.f,(int)(row%13)-6);memcpy(cache+base+8192+(row%64)*4,&scale,4);}
  BnBackendSession*s=bn_backend_session_create(),*other=bn_backend_session_create();assert(s&&other);void*r=bn_gpu_backend_resident_request(g,s),*peer=bn_gpu_backend_resident_request(g,other);int32_t status=0;
  BnGPUIndexerFP8PagedScoreResidentPlan p={.n_tokens=c.tokens,.position=c.position,.n_keys=(int)k,.num_cache_blocks=blocks,.logical_blocks=blocks,
   .q=alloc(g,r,BN_GGUF_TENSOR_I8,n*4096,q),.weights=alloc(g,r,BN_GGUF_TENSOR_F32,n*32,w),.cache=alloc(g,r,BN_GGUF_TENSOR_I8,(size_t)blocks*8448,cache),.block_table=alloc(g,r,BN_GGUF_TENSOR_I32,blocks,table),.scores=alloc(g,r,BN_GGUF_TENSOR_F32,n*k,NULL),.status=alloc(g,r,BN_GGUF_TENSOR_I32,1,&status)};
  assert(!g->indexer_fp8_paged_score_resident(g->ctx,r,&p));total+=check(g,r,p.scores,4,c.hash);char prefix[32];snprintf(prefix,sizeof(prefix),"%zu",ci);compare_file(g,r,p.scores,4,directory,prefix,"");assert(!g->resident_read(r,p.status,&status)&&!status);
  BnGPUIndexerFP8PagedScoreResidentPlan bad;
#define BAD(change) do{bad=p;change;assert(g->indexer_fp8_paged_score_resident(g->ctx,r,&bad)==-1);}while(0)
  BAD(bad.n_tokens=0);BAD(bad.n_tokens=INT_MAX);BAD(bad.position=INT_MAX);BAD(bad.position=-1);BAD(bad.n_keys++);BAD(bad.num_cache_blocks=0);BAD(bad.logical_blocks=0);BAD(bad.cache.count--);BAD(bad.weights.request=peer);BAD(bad.weights=p.scores);BAD(bad.status=p.block_table);BAD(bad.q.offset=SIZE_MAX);
  BnGPUResidentView unaligned=alloc(g,r,BN_GGUF_TENSOR_I8,n*4096+1,NULL);BAD(bad.q=unaligned;bad.q.offset++;bad.q.count--);
  unaligned=alloc(g,r,BN_GGUF_TENSOR_I8,(size_t)blocks*8448+1,NULL);BAD(bad.cache=unaligned;bad.cache.offset++;bad.cache.count--);
  assert(check(g,r,p.scores,4,c.hash)==n*k);
#undef BAD
  if(k){int first=table[0];for(int invalid=0;invalid<2;++invalid){table[0]=invalid?blocks:-1;status=0;assert(!g->resident_write(r,p.block_table,table)&&!g->resident_write(r,p.status,&status));assert(!g->indexer_fp8_paged_score_resident(g->ctx,r,&p));assert(!g->resident_read(r,p.status,&status)&&status==1);assert(!g->resident_read(r,p.scores,out));for(int row=0;row<c.tokens;++row)for(size_t j=0;j<k&&j<64;++j)assert(isinf(out[(size_t)row*k+j])&&out[(size_t)row*k+j]<0);}
   table[0]=first;assert(!g->resident_write(r,p.block_table,table));assert(!g->indexer_fp8_paged_score_resident(g->ctx,r,&p));assert(!g->resident_read(r,p.status,&status)&&status==1);assert(check(g,r,p.scores,4,c.hash)==n*k);
  }
  bn_backend_session_free(other);bn_backend_session_free(s);free(q);free(cache);free(w);free(out);free(table);
 }return total;
}
int main(int argc,char**argv){assert(argc==1||argc==2);BnGPUBackend*g=bn_gpu_cuda_create();assert(g&&g->indexer_fp8_paged_score_resident);size_t n=paged_cases(g,argc==2?argv[1]:NULL);bn_gpu_cuda_destroy(g);printf("GLM direct paged FP8 score PASSED: %zu exact F32 elements\n",n);return 0;}
