/* Production Flash QSA request API: FIRST_SHARD CASE_INDEX CAPTURE_DIR
 * [PROFILE_PHASE | --guards-only GDN_CAPTURES]. Exact layer3 captures, shared-state guards,
 * bounded prepare rollback, and one outer commit; not full-model admission. */
#include "gpu_cuda.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "backend_session.h"
#include "model_internal.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif
static int sync_calls,alloc_calls,fail_at;
static int (*real_sync)(void*);
static int (*real_alloc)(void*,int,size_t,BnGPUResidentView*);
static int counted_sync(void *r){sync_calls++;return real_sync(r);}
static int failed_sync(void *r){(void)r;return -1;}
static int fail_alloc(void *r,int type,size_t n,BnGPUResidentView *v){alloc_calls++;return alloc_calls==fail_at?-1:real_alloc(r,type,n,v);}
static int unexpected_alloc(void *r,int type,size_t n,BnGPUResidentView *v){(void)r;(void)type;(void)n;(void)v;assert(!"submit must not allocate");return -1;}
static int unexpected_write(void *r,BnGPUResidentView v,const void *data){(void)r;(void)v;(void)data;assert(!"submit must not upload");return -1;}
static int failed_score(void *ctx,void *r,const BnGPUQSAScoreResidentPlan *p){(void)ctx;(void)r;(void)p;return -1;}
static void *load(const char *path,size_t *bytes){FILE*f=fopen(path,"rb");assert(f&&!fseek(f,0,SEEK_END));long n=ftell(f);assert(n>0&&!fseek(f,0,SEEK_SET));*bytes=(size_t)n;void*v=malloc(*bytes);assert(v&&fread(v,1,*bytes,f)==*bytes&&fgetc(f)==EOF);fclose(f);return v;}
static void path(char *out,size_t n,const char *dir,const char *prefix,const char *stage){int len=snprintf(out,n,"%s/%s-%s.bin",dir,prefix,stage);assert(len>0&&(size_t)len<n);}
static size_t compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*prefix,const char*stage,int width){
 char file[4096];path(file,sizeof(file),dir,prefix,stage);size_t bytes;void*gold=load(file,&bytes);assert(bytes==v.count*(size_t)width);void*actual=malloc(bytes);assert(actual&&!g->resident_read(r,v,actual));size_t d=0;
 for(size_t i=0;i<v.count;i++)if(memcmp((char*)gold+i*width,(char*)actual+i*width,(size_t)width)){if(d<2)fprintf(stderr,"%s %s word%zu differs\n",prefix,stage,i);d++;}
 if(d)fprintf(stderr,"%s %s differences%zu\n",prefix,stage,d);
 assert(!d);free(gold);free(actual);return v.count;
}
static void input(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*prefix){char file[4096];path(file,sizeof(file),dir,prefix,"input");size_t bytes;void*raw=load(file,&bytes);assert(bytes==v.count*2&&!g->resident_write(r,v,raw));free(raw);}
static void empty_failed(BnBackendSession*s){assert(!bn_backend_session_has_gpu_resources(s)&&!bn_backend_session_has_execution_resources(s));}
static void guards(BnModel*m,BnGPUBackend*g,const char*dir,const char*gdn_dir){
 BnBackendSession*s=bn_backend_session_create();assert(s);
 real_alloc=g->resident_alloc;g->resident_alloc=fail_alloc;fail_at=0;alloc_calls=0;
 assert(!bn_transformer_bf16_flash_prepare_context(m,g,s,1,8,SIZE_MAX,SIZE_MAX));int allocations=alloc_calls;
 assert(allocations==239);BnTransformerBF16FlashInfo info;assert(!bn_transformer_bf16_flash_info(s,&info));
 assert(info.max_context==8&&info.valid_qsa_layers&&info.valid_gdn_layers&&info.valid_layers==(info.valid_qsa_layers|info.valid_gdn_layers));
 size_t gpu=info.gpu_bytes,host=info.preparation_host_peak_bytes;
 assert(host==info.host_bytes+8*sizeof(int32_t));bn_backend_session_free(s);
 s=bn_backend_session_create();assert(s);alloc_calls=0;
 assert(bn_transformer_bf16_flash_prepare_context(m,g,s,1,8,gpu-1,host)==-1);empty_failed(s);assert(!alloc_calls);
 assert(bn_transformer_bf16_flash_prepare_context(m,g,s,1,8,gpu,host-1)==-1);empty_failed(s);assert(!alloc_calls);
 assert(bn_transformer_bf16_flash_prepare_context(m,g,s,9,8,SIZE_MAX,SIZE_MAX)==-1);empty_failed(s);assert(!alloc_calls);
 for(fail_at=1;fail_at<=allocations;fail_at++){
  alloc_calls=0;assert(bn_transformer_bf16_flash_prepare_context(m,g,s,1,8,gpu,host)==-1);
  assert(alloc_calls==fail_at);empty_failed(s);
 }
 fail_at=0;alloc_calls=0;assert(!bn_transformer_bf16_flash_prepare_context(m,g,s,1,8,gpu,host)&&alloc_calls==allocations);
 g->resident_alloc=real_alloc;
 assert(bn_transformer_bf16_flash_prepare_context(m,g,s,1,8,SIZE_MAX,SIZE_MAX)==-1);
 bn_backend_session_free(s);printf("QSA_REQUEST_ROLLBACK_PASSED allocations%d gpu_bytes%zu host_peak%zu\n",allocations,gpu,host);fflush(stdout);
 s=bn_backend_session_create();assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,5,8,SIZE_MAX,SIZE_MAX));
 void *mixed_r=bn_gpu_backend_resident_request(g,s);BnTransformerBF16FlashQSAViews qbefore,qafter;
 BnTransformerBF16FlashGDNViews gbefore,gafter,gpeer;
 assert(!bn_transformer_bf16_flash_qsa_views(s,3,5,&qbefore));assert(!bn_transformer_bf16_flash_gdn_views(s,0,5,&gbefore));
 input(g,mixed_r,qbefore.input,dir,"5-5-0");
 real_sync=g->resident_synchronize;g->resident_synchronize=counted_sync;sync_calls=0;
 assert(!bn_transformer_bf16_flash_begin(s,5,0,1,(UINT64_C(1)<<3)|1,0));
 assert(!bn_transformer_bf16_flash_qsa_submit(s,3));assert(bn_transformer_bf16_flash_complete(s)==-1&&!sync_calls);
 assert(!bn_transformer_bf16_flash_qsa_views(s,3,5,&qafter)&&!qafter.position&&qafter.history.slot==qbefore.history.slot);
 assert(!bn_transformer_bf16_flash_gdn_submit(s,0));assert(!bn_transformer_bf16_flash_complete(s)&&sync_calls==1);
 g->resident_synchronize=real_sync;
 assert(!bn_transformer_bf16_flash_qsa_views(s,3,5,&qafter)&&qafter.position==5&&qafter.history.slot!=qbefore.history.slot);
 assert(!bn_transformer_bf16_flash_gdn_views(s,0,5,&gafter)&&gafter.position==5&&gafter.state.slot!=gbefore.state.slot);
 assert(!bn_transformer_bf16_flash_gdn_views(s,1,1,&gpeer)&&!gpeer.position);
 compare(g,mixed_r,gafter.out,gdn_dir,"5-5-0","out",2);compare(g,mixed_r,gafter.state,gdn_dir,"5-5-0","state",4);
 compare(g,mixed_r,bn_gpu_resident_view_slice(qafter.key_cache,0,5u*512),dir,"5-5-0","key_cache",2);
 bn_backend_session_free(s);puts("QSA_GDN_ATOMIC_COMMIT_PASSED");fflush(stdout);
 BnBackendSession *sa=bn_backend_session_create(),*sb=bn_backend_session_create();assert(sa&&sb);
 assert(!bn_transformer_bf16_flash_prepare_context(m,g,sa,1,8,SIZE_MAX,SIZE_MAX));
 assert(!bn_transformer_bf16_flash_prepare_context(m,g,sb,5,9,SIZE_MAX,SIZE_MAX));
 BnTransformerBF16FlashQSAViews va,vb;
 assert(!bn_transformer_bf16_flash_qsa_views(sa,3,1,&va)&&!bn_transformer_bf16_flash_qsa_views(sb,3,5,&vb));
 void *ra=bn_gpu_backend_resident_request(g,sa),*rb=bn_gpu_backend_resident_request(g,sb);assert(ra!=rb);
 input(g,ra,va.input,dir,"1-1-0");input(g,rb,vb.input,dir,"5-5-0");
 assert(!bn_transformer_bf16_flash_begin(sa,1,0,1,UINT64_C(1)<<3,0));
 assert(!bn_transformer_bf16_flash_qsa_submit(sa,3));
 assert(!bn_transformer_bf16_flash_begin(sb,5,0,1,UINT64_C(1)<<3,0));
 assert(!bn_transformer_bf16_flash_qsa_submit(sb,3));
 assert(!bn_transformer_bf16_flash_qsa_views(sa,3,1,&va)&&!va.position);
 assert(!bn_transformer_bf16_flash_qsa_views(sb,3,5,&vb)&&!vb.position);
 assert(!bn_transformer_bf16_flash_complete(sb));assert(!bn_transformer_bf16_flash_complete(sa));
 assert(!bn_transformer_bf16_flash_qsa_views(sa,3,1,&va)&&va.position==1);
 assert(!bn_transformer_bf16_flash_qsa_views(sb,3,5,&vb)&&vb.position==5);
 compare(g,ra,va.out,dir,"1-1-0","out",2);compare(g,rb,vb.out,dir,"5-5-0","out",2);
 compare(g,ra,bn_gpu_resident_view_slice(va.key_cache,0,512),dir,"1-1-0","key_cache",2);
 compare(g,rb,bn_gpu_resident_view_slice(vb.key_cache,0,5u*512),dir,"5-5-0","key_cache",2);
 bn_backend_session_free(sa);bn_backend_session_free(sb);puts("QSA_REQUEST_ISOLATION_PASSED reversed completion, distinct capacities and histories");fflush(stdout);
 for(int failure=0;failure<2;failure++){
  s=bn_backend_session_create();assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,1,8,SIZE_MAX,SIZE_MAX));
  void*r=bn_gpu_backend_resident_request(g,s);BnTransformerBF16FlashQSAViews v;assert(!bn_transformer_bf16_flash_qsa_views(s,3,1,&v));
  input(g,r,v.input,dir,"1-1-0");assert(!bn_transformer_bf16_flash_begin(s,1,0,1,UINT64_C(1)<<3,0));
  int(*score)(void*,void*,const BnGPUQSAScoreResidentPlan*)=g->qsa_score_bf16_resident;
  if(failure==0){g->qsa_score_bf16_resident=failed_score;assert(bn_transformer_bf16_flash_qsa_submit(s,3)==-1);g->qsa_score_bf16_resident=score;}
  else {assert(!bn_transformer_bf16_flash_qsa_submit(s,3));real_sync=g->resident_synchronize;g->resident_synchronize=failed_sync;assert(bn_transformer_bf16_flash_complete(s)==-1);g->resident_synchronize=real_sync;}
  assert(!bn_transformer_bf16_flash_info(s,&info)&&info.failed&&info.position==0&&info.active);
  assert(bn_transformer_bf16_flash_begin(s,1,0,1,UINT64_C(1)<<3,0)==-1);
  assert(bn_transformer_bf16_flash_restart(s)==-1);
  uint16_t old[512];assert(!g->resident_read(r,v.history,old));for(int i=0;i<512;i++)assert(!old[i]);
  bn_backend_session_free(s);
 }
 s=bn_backend_session_create();assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,1,3,SIZE_MAX,SIZE_MAX));
 void*r=bn_gpu_backend_resident_request(g,s);BnTransformerBF16FlashQSAViews v;
 for(int pos=0;pos<3;pos++){
  assert(!bn_transformer_bf16_flash_qsa_views(s,3,1,&v)&&v.position==pos);input(g,r,v.input,dir,"1-1-0");
  assert(!bn_transformer_bf16_flash_begin(s,1,pos,1,UINT64_C(1)<<3,0));assert(!bn_transformer_bf16_flash_qsa_submit(s,3));assert(!bn_transformer_bf16_flash_complete(s));
 }
 assert(bn_transformer_bf16_flash_begin(s,1,3,1,UINT64_C(1)<<3,0)==-1);
 assert(!bn_transformer_bf16_flash_info(s,&info)&&!info.failed&&!info.active&&info.position==3);
 int32_t eos[2]={m->config.ple_eos_token_id,m->config.ple_eos_token_id};
 assert(bn_transformer_bf16_ple_reset(s,4,eos,2,NULL,0)==-1);
 assert(!bn_transformer_bf16_flash_restart(s));assert(!bn_transformer_bf16_flash_qsa_views(s,3,1,&v)&&!v.position);
 assert(!bn_transformer_bf16_flash_info(s,&info)&&!info.position);bn_backend_session_free(s);
 puts("QSA_REQUEST_FAILURE_AND_CONTEXT_GUARDS_PASSED");fflush(stdout);
}
typedef struct{int ci,total,cap,pos,nt,prefill;char prefix[256];} Phase;
int main(int argc,char**argv){
 assert(argc>=4&&argc<=6);int only=argc>=5&&!strcmp(argv[4],"--guards-only");assert(only?argc==6:argc<=5);int profile=argc==5&&!only?atoi(argv[4]):-1,profiled=0;
 BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModel m={0};assert(!bn_model_load_with_cpu_preparation(&m,f,8192,0,0,0));
 BnGPUBackend*g=bn_gpu_cuda_create();assert(g);BnBackendModel*owner=bn_model_backend(&m);
 size_t budget=(size_t)12*1024*1024*1024,host=64u*1024*1024;
 assert(!bn_backend_layout_prepare_bf16_hc_model(owner,g,&m.config,&m.weights,budget));
 assert(!bn_backend_layout_prepare_bf16_ple_model(owner,g,&m.config,&m.weights,budget,host));
 assert(!bn_backend_layout_prepare_bf16_flash_gdn_model(owner,g,&m.config,&m.weights,budget,host));
 assert(!bn_backend_layout_prepare_bf16_flash_qsa_model(owner,g,&m.config,&m.weights,budget,host));
 if(only){guards(&m,g,argv[3],argv[5]);bn_model_free(&m);bn_gpu_cuda_destroy(g);bn_gguf_free(f);return 0;}
 FILE*index=fopen(argv[2],"r");assert(index);Phase phases[512];int np=0;
 while(np<512&&fscanf(index,"%d %d %d %d %d %d %255s",&phases[np].ci,&phases[np].total,&phases[np].cap,&phases[np].pos,&phases[np].nt,&phases[np].prefill,phases[np].prefix)==7)np++;
 assert(feof(index)&&np==265);fclose(index);size_t words=0,indices=0;int cases=0;
 for(int first=0;first<np;){int end=first+1;while(end<np&&phases[end].ci==phases[first].ci)end++;
  Phase*c=&phases[first];BnBackendSession*s=bn_backend_session_create();assert(s);
  assert(!bn_transformer_bf16_flash_prepare_context(&m,g,s,c->cap,c->total+8,(size_t)4*1024*1024*1024,host));
  void*r=bn_gpu_backend_resident_request(g,s);int repeats=profile>=first&&profile<end?2:1;
  for(int repeat=0;repeat<repeats;repeat++){
   if(repeat)assert(!bn_transformer_bf16_flash_restart(s));
   for(int pi=first;pi<end;pi++){
    Phase*p=&phases[pi];BnTransformerBF16FlashQSAViews before,after,peer;BnTransformerBF16FlashInfo info;
    assert(!bn_transformer_bf16_flash_qsa_views(s,3,p->nt,&before)&&before.position==p->pos);
    assert(!bn_transformer_bf16_flash_qsa_views(s,7,1,&peer)&&!peer.position);
    input(g,r,before.input,argv[3],p->prefix);
    int capture=repeat==repeats-1&&pi==profile;
#ifdef BN_CAPTURE_CUDA_PROFILER
    if(capture){assert(cudaProfilerStart()==cudaSuccess);profiled++;}
#else
    assert(!capture);
#endif
    real_sync=g->resident_synchronize;real_alloc=g->resident_alloc;
    int(*write)(void*,BnGPUResidentView,const void*)=g->resident_write;
    g->resident_synchronize=counted_sync;sync_calls=0;g->resident_alloc=unexpected_alloc;g->resident_write=unexpected_write;
    assert(!bn_transformer_bf16_flash_begin(s,p->nt,p->pos,p->prefill,UINT64_C(1)<<3,0));
    assert(bn_transformer_bf16_flash_gdn_submit(s,3)==-1);
    assert(bn_transformer_bf16_flash_qsa_submit(s,7)==-1);
    assert(bn_transformer_bf16_flash_complete(s)==-1&&!sync_calls);
    assert(!bn_transformer_bf16_flash_qsa_submit(s,3));
    assert(bn_transformer_bf16_flash_qsa_submit(s,3)==-1&&!sync_calls);
    assert(!bn_transformer_bf16_flash_qsa_views(s,3,p->nt,&after)&&after.position==before.position&&after.history.slot==before.history.slot);
    assert(!bn_transformer_bf16_flash_info(s,&info)&&info.active&&info.submitted_layers==(UINT64_C(1)<<3));
    assert(!bn_transformer_bf16_flash_complete(s)&&sync_calls==1);
    g->resident_synchronize=real_sync;g->resident_alloc=real_alloc;g->resident_write=write;
#ifdef BN_CAPTURE_CUDA_PROFILER
    if(capture)assert(cudaProfilerStop()==cudaSuccess);
#endif
    assert(!bn_transformer_bf16_flash_qsa_views(s,3,p->nt,&after)&&after.position==p->pos+p->nt&&after.history.slot!=before.history.slot);
    assert(!bn_transformer_bf16_flash_qsa_views(s,7,1,&peer)&&!peer.position);
    uint16_t untouched[16];assert(!g->resident_read(r,bn_gpu_resident_view_slice(peer.key_cache,0,16),untouched));for(int i=0;i<16;i++)assert(!untouched[i]);
    if(repeat<repeats-1)continue;
    BnGPUResidentView views[]={after.input,after.projected,after.q,after.k,after.v,after.gate,after.index_q,after.attention,after.out};
    const char*stages[]={"input","projected","q","k","v","gate","index_q","attention","out"};
    for(int i=0;i<9;i++)words+=compare(g,r,views[i],argv[3],p->prefix,stages[i],2);
    indices+=compare(g,r,after.indices,argv[3],p->prefix,"indices",4);
    words+=compare(g,r,bn_gpu_resident_view_slice(after.key_cache,0,(size_t)after.position*512),argv[3],p->prefix,"key_cache",2);
    words+=compare(g,r,bn_gpu_resident_view_slice(after.value_cache,0,(size_t)after.position*512),argv[3],p->prefix,"value_cache",2);
    printf("phase%d %s production request exact committed%d\n",pi,p->prefix,after.position);fflush(stdout);
   }
  }
  bn_backend_session_free(s);first=end;cases++;
 }
 assert(profile<0||profiled==1);bn_model_free(&m);bn_gpu_cuda_destroy(g);bn_gguf_free(f);
 printf("QSA_PRODUCTION_REQUEST_PASSED cases%d phases%d bf16_words%zu index_words%zu\n",cases,np,words,indices);return 0;
}
