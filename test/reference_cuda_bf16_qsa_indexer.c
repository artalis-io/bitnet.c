/* Actual Flash layer3 index QK projection + QSA pre-indexer, BF16 text profile.
 * Link like test_cuda_backend; run FIRST_SHARD CAPTURE_DIRECTORY. */
#include "gpu_cuda.h"
#include "backend_session.h"
#include "model_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
static uint16_t bf16(float x){uint32_t u;memcpy(&u,&x,4);return(uint16_t)((u+0x7fff+((u>>16)&1))>>16);}
static BnGPUResidentView alloc(BnGPUBackend*g,void*r,size_t n,int zero){BnGPUResidentView v;assert(!g->resident_alloc(r,BN_GGUF_TENSOR_BF16,n,&v));if(zero)assert(!g->resident_zero(r,v));return v;}
static BnGPUResidentView slice(BnGPUResidentView v,size_t off,size_t n){return bn_gpu_resident_view_slice(v,off,n);}
static void*norm(BnGPUBackend*g,const float*x){uint16_t y[128];for(int i=0;i<128;i++)y[i]=bf16(x[i]-1.0f);void*w=g->buffer_create(g->ctx,y,sizeof(y),BN_GGUF_TENSOR_BF16,1,128);assert(w);return w;}
static size_t compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*prefix,const char*stage){
 char path[2048];snprintf(path,sizeof(path),"%s/%s-%s.bin",dir,prefix,stage);FILE*f=fopen(path,"rb");assert(f);
 uint16_t*a=malloc(v.count*2),*b=malloc(v.count*2);assert(a&&b&&fread(b,2,v.count,f)==v.count&&fgetc(f)==EOF);fclose(f);assert(!g->resident_read(r,v,a));
 size_t d=0;for(size_t i=0;i<v.count;i++)if(a[i]!=b[i]){if(d<3)fprintf(stderr,"%s %s word%zu %04x expected%04x\n",prefix,stage,i,a[i],b[i]);d++;}
 printf("%s %s words%zu differences%zu\n",prefix,stage,v.count,d);fflush(stdout);assert(!d);free(a);free(b);return v.count;
}
int main(int argc,char**argv){
 assert(argc==3);BnGGUFFile*file=bn_gguf_open_file(argv[1]);assert(file);BnModel m={0};assert(!bn_model_load_with_cpu_preparation(&m,file,8192,0,0,0));
 assert(m.config.dim==2560&&m.config.indexer_head_count==4&&m.config.indexer_head_size==128&&m.config.attention_compress_ratios[3]==4);
 BnAttentionIndexerWeights*w=&m.weights.layers[3].indexer;
 assert(w->q.type==BN_GGUF_TENSOR_BF16&&w->k.type==BN_GGUF_TENSOR_BF16&&w->q.rows==512&&w->k.rows==128&&w->q.cols==2560&&w->k.cols==2560);
 BnGPUBackend*g=bn_gpu_cuda_create();assert(g&&g->qsa_indexer_bf16_resident);
 size_t bytes=640u*2560*2;unsigned char*stack=malloc(bytes);assert(stack);memcpy(stack,w->q.data,512u*2560*2);memcpy(stack+512u*2560*2,w->k.data,128u*2560*2);
 void*projection=g->buffer_create(g->ctx,stack,bytes,BN_GGUF_TENSOR_BF16,640,2560);assert(projection);free(stack);
 void*qw=norm(g,w->q_norm),*kw=norm(g,w->k_norm);BnGPURotaryCachePlan rp={8192,64,10000000.f};void*cs=g->buffer_create_rotary_bf16(g->ctx,&rp);assert(cs);
 const int nts[]={1,3,4,5,17,129,129,129,129,2053,2053,4097},chunks[]={1,3,4,5,7,129,64,7,1,2053,257,4097};size_t words=0;int phases=0;
 for(int ci=0;ci<12;ci++){
  int nt=nts[ci],cap=chunks[ci];BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);assert(!g->linear_bf16_reserve(g->ctx,r,0));
  BnGPUResidentView input=alloc(g,r,(size_t)cap*2560,0),merged=alloc(g,r,(size_t)cap*640,0),q=alloc(g,r,(size_t)cap*512,0);
  BnGPUResidentView history[2]={alloc(g,r,512,1),alloc(g,r,512,0)},compressed=alloc(g,r,(size_t)((nt+8)/4)*128,1);
  int pos=0,phase=0,parity=0;
  while(pos<nt+8){
   int take=pos<nt?(nt-pos<cap?nt-pos:cap):1;size_t n=(size_t)take;
   uint16_t*x=malloc(n*2560*2);assert(x);for(size_t i=0;i<n*2560;i++){size_t j=(size_t)pos*2560+i;x[i]=bf16((float)((int)((j*17+j/37)%509)-254)/128);}
   assert(!g->resident_write(r,slice(input,0,n*2560),x));free(x);
   BnGPUBF16LinearResidentPlan lp={0};lp.n_tokens=take;lp.rows=640;lp.cols=2560;lp.weight.type=BN_GGUF_TENSOR_BF16;lp.weight.decoded=projection;lp.x=slice(input,0,n*2560);lp.out=slice(merged,0,n*640);assert(!g->linear_bf16_resident(g->ctx,r,&lp));
   BnGPUQSAIndexerResidentPlan p={0};p.n_tokens=take;p.position=pos;p.eps=1e-6f;p.q_weight=qw;p.k_weight=kw;p.cos_sin=cs;p.projected=lp.out;p.initial_history=history[parity];p.next_history=history[1-parity];p.q=slice(q,0,n*512);
   int groups=(pos+take)/4-pos/4;if(groups)p.compressed=slice(compressed,(size_t)(pos/4)*128,(size_t)groups*128);
   BnGPUQSAIndexerResidentPlan bad=p;bad.next_history=p.initial_history;assert(g->qsa_indexer_bf16_resident(g->ctx,r,&bad)==-1);
   bad=p;bad.position=INT_MAX;assert(g->qsa_indexer_bf16_resident(g->ctx,r,&bad)==-1);
   bad=p;bad.eps=NAN;assert(g->qsa_indexer_bf16_resident(g->ctx,r,&bad)==-1);
   bad=p;bad.q_weight=projection;assert(g->qsa_indexer_bf16_resident(g->ctx,r,&bad)==-1);
   bad=p;if(groups)bad.compressed=(BnGPUResidentView){0};else bad.compressed=slice(compressed,0,128);assert(g->qsa_indexer_bf16_resident(g->ctx,r,&bad)==-1);
   assert(!g->qsa_indexer_bf16_resident(g->ctx,r,&p));assert(!g->resident_synchronize(r));parity=1-parity;
   char prefix[64];snprintf(prefix,sizeof(prefix),"%d-%d-%d",nt,cap,phase);
   words+=compare(g,r,lp.x,argv[2],prefix,"input");words+=compare(g,r,lp.out,argv[2],prefix,"projected");words+=compare(g,r,p.q,argv[2],prefix,"q");words+=compare(g,r,history[parity],argv[2],prefix,"history");
   int visible=(pos+take)/4;if(visible)words+=compare(g,r,slice(compressed,0,(size_t)visible*128),argv[2],prefix,"compressed");
   pos+=take;phase++;phases++;
  }
  bn_backend_session_free(s);
 }
 g->buffer_destroy(g->ctx,cs);g->buffer_destroy(g->ctx,kw);g->buffer_destroy(g->ctx,qw);g->buffer_destroy(g->ctx,projection);bn_model_free(&m);bn_gpu_cuda_destroy(g);bn_gguf_free(file);
 printf("QSA_INDEXER_PASSED words%zu phases%d cases12\n",words,phases);return 0;
}
