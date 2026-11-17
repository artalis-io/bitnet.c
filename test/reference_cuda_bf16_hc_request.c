/* Production HC/QSA request API: FIRST_SHARD CASE_INDEX CAPTURE_DIR
 * [PROFILE_PHASE | --guards-only HC_CAPTURES PLE_CAPTURES]. Exact layer3 captures, shared-state guards,
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
static int sync_calls;
static int (*real_sync)(void*);
static int (*real_alloc)(void*,int,size_t,BnGPUResidentView*);
static int counted_sync(void *r){sync_calls++;return real_sync(r);}
static int unexpected_alloc(void *r,int type,size_t n,BnGPUResidentView *v){(void)r;(void)type;(void)n;(void)v;assert(!"submit must not allocate");return -1;}
static int unexpected_write(void *r,BnGPUResidentView v,const void *data){(void)r;(void)v;(void)data;assert(!"submit must not upload");return -1;}
static void *load(const char *path,size_t *bytes){FILE*f=fopen(path,"rb");assert(f&&!fseek(f,0,SEEK_END));long n=ftell(f);assert(n>0&&!fseek(f,0,SEEK_SET));*bytes=(size_t)n;void*v=malloc(*bytes);assert(v&&fread(v,1,*bytes,f)==*bytes&&fgetc(f)==EOF);fclose(f);return v;}
static void path(char *out,size_t n,const char *dir,const char *prefix,const char *stage){int len=snprintf(out,n,"%s/%s-%s.bin",dir,prefix,stage);assert(len>0&&(size_t)len<n);}
static size_t compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*prefix,const char*stage,int width){
 char file[4096];path(file,sizeof(file),dir,prefix,stage);size_t bytes;void*gold=load(file,&bytes);assert(bytes==v.count*(size_t)width);void*actual=malloc(bytes);assert(actual&&!g->resident_read(r,v,actual));size_t d=0;
 for(size_t i=0;i<v.count;i++)if(memcmp((char*)gold+i*width,(char*)actual+i*width,(size_t)width)){if(d<2)fprintf(stderr,"%s %s word%zu differs\n",prefix,stage,i);d++;}
 if(d)fprintf(stderr,"%s %s differences%zu\n",prefix,stage,d);
 assert(!d);free(gold);free(actual);return v.count;
}
static void input(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*prefix){char file[4096];path(file,sizeof(file),dir,prefix,"input");size_t bytes;void*raw=load(file,&bytes);assert(bytes==v.count*2&&!g->resident_write(r,v,raw));free(raw);}
static int reject_silu(void*ctx,void*r,const BnGPUScaledSiluResidentPlan*p){(void)ctx;(void)r;(void)p;return -1;}
static void hc_guards(BnModel*m,BnGPUBackend*g,const char*hcdir,const char*pledir){
 BnBackendSession*s=bn_backend_session_create();assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,129,137,SIZE_MAX,SIZE_MAX));
 void*r=bn_gpu_backend_resident_request(g,s);int nts[]={1,5,17,129};size_t words=0;
 int(*saved_silu)(void*,void*,const BnGPUScaledSiluResidentPlan*)=g->scaled_silu_bf16_resident;
 g->scaled_silu_bf16_resident=NULL;
 assert(bn_transformer_bf16_flash_begin_blocks(s,1,0,1,0,UINT64_C(1)<<2,0)==-1);
 g->scaled_silu_bf16_resident=saved_silu;BnTransformerBF16FlashInfo initial;
 assert(!bn_transformer_bf16_flash_info(s,&initial)&&!initial.active&&!initial.failed);
 for(int ci=0;ci<4;ci++){
  int nt=nts[ci];char prefix[64];snprintf(prefix,sizeof(prefix),"2-ffn-%d",nt);
  BnTransformerBF16FlashHCViews v;assert(!bn_transformer_bf16_flash_hc_views(s,nt,&v));
  input(g,r,v.hidden,hcdir,prefix);char file[4096];path(file,sizeof(file),hcdir,prefix,"block");size_t bytes;void*raw=load(file,&bytes);assert(bytes==v.block.count*2&&!g->resident_write(r,v.block,raw));free(raw);
  real_sync=g->resident_synchronize;g->resident_synchronize=counted_sync;sync_calls=0;
  assert(!bn_transformer_bf16_flash_begin_blocks(s,nt,0,1,0,UINT64_C(1)<<2,0));
  assert(bn_transformer_bf16_flash_hc_mix(s,2,2)==-1);assert(!bn_transformer_bf16_flash_hc_mix(s,2,1));
  assert(bn_transformer_bf16_flash_hc_mix(s,2,1)==-1);assert(bn_transformer_bf16_flash_complete(s)==-1&&!sync_calls);
  assert(bn_transformer_bf16_flash_gdn_submit(s,2)==-1);assert(!bn_transformer_bf16_flash_hc_combine(s));
  assert(bn_transformer_bf16_flash_hc_combine(s)==-1);assert(!bn_transformer_bf16_flash_complete(s)&&sync_calls==1);
  g->resident_synchronize=real_sync;assert(!bn_transformer_bf16_flash_hc_views(s,nt,&v));
  BnGPUResidentView views[]={v.residual,v.normalized,v.down,v.scaled,v.gate,v.mixed,v.inject,v.combined,v.block};
  const char*names[]={"input","norm","down","scaled","up","mixed","inject","combined","block"};
  for(int j=0;j<9;j++)words+=compare(g,r,views[j],hcdir,prefix,names[j],2);
  assert(v.hidden.slot==v.combined.slot);BnTransformerBF16FlashGDNViews untouched;
  assert(!bn_transformer_bf16_flash_gdn_views(s,2,1,&untouched)&&!untouched.position);
 }
 bn_backend_session_free(s);printf("HC_FFN_RESIDENT_PASSED cases4 bf16_words%zu\n",words);fflush(stdout);
 s=bn_backend_session_create();assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,5,8,SIZE_MAX,SIZE_MAX));r=bn_gpu_backend_resident_request(g,s);
 BnTransformerBF16FlashHCViews before,after;assert(!bn_transformer_bf16_flash_hc_views(s,5,&before));
 char file[4096];path(file,sizeof(file),pledir,"5-5","hidden");size_t bytes;void*raw=load(file,&bytes);assert(bytes==before.hidden.count*2&&!g->resident_write(r,before.hidden,raw));free(raw);
 path(file,sizeof(file),pledir,"5-5","initial");raw=load(file,&bytes);assert(bytes==9u*10240*2);
 int32_t ctx[]={248044,116},tokens[]={97,134,171,208,245};assert(!bn_transformer_bf16_ple_reset(s,0,ctx,2,raw,9u*10240));free(raw);
 real_sync=g->resident_synchronize;g->resident_synchronize=counted_sync;sync_calls=0;
 assert(!bn_transformer_bf16_flash_begin_blocks(s,5,0,1,0,0,1));
 assert(!bn_transformer_bf16_flash_ple_apply(s,tokens,5,0)&&!sync_calls);
 assert(!bn_transformer_bf16_flash_hc_views(s,5,&after)&&after.hidden.slot!=before.hidden.slot);
 assert(!bn_transformer_bf16_flash_complete(s)&&sync_calls==1);g->resident_synchronize=real_sync;
 compare(g,r,after.hidden,pledir,"5-5","out",2);BnTransformerBF16PLEViews pv;
 assert(!bn_transformer_bf16_ple_views(s,5,&pv));compare(g,r,pv.history,pledir,"5-5","history",2);
 bn_backend_session_free(s);puts("HC_PLE_RESIDUAL_ADVANCE_PASSED");fflush(stdout);
 s=bn_backend_session_create();assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,1,8,SIZE_MAX,SIZE_MAX));
 int(*silu)(void*,void*,const BnGPUScaledSiluResidentPlan*)=g->scaled_silu_bf16_resident;
 assert(!bn_transformer_bf16_flash_begin_blocks(s,1,0,1,0,UINT64_C(1)<<2,0));g->scaled_silu_bf16_resident=reject_silu;
 assert(bn_transformer_bf16_flash_hc_mix(s,2,1)==-1);g->scaled_silu_bf16_resident=silu;
 BnTransformerBF16FlashInfo info;assert(!bn_transformer_bf16_flash_info(s,&info)&&info.failed&&!info.submitted_ffn&&!info.position);
 assert(bn_transformer_bf16_flash_complete(s)==-1);bn_backend_session_free(s);puts("HC_REQUEST_FAILURE_GUARDS_PASSED");
}

typedef struct{int ci,total,cap,pos,nt,prefill;char prefix[256];} Phase;
int main(int argc,char**argv){
 assert(argc>=4&&argc<=7);int only=argc>=5&&!strcmp(argv[4],"--guards-only");assert(only?argc==7:argc<=5);int profile=argc==5&&!only?atoi(argv[4]):-1,profiled=0;
 BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModel m={0};assert(!bn_model_load_with_cpu_preparation(&m,f,8192,0,0,0));
 BnGPUBackend*g=bn_gpu_cuda_create();assert(g);BnBackendModel*owner=bn_model_backend(&m);
 size_t budget=(size_t)12*1024*1024*1024,host=64u*1024*1024;
 assert(!bn_backend_layout_prepare_bf16_hc_model(owner,g,&m.config,&m.weights,budget));
 assert(!bn_backend_layout_prepare_bf16_ple_model(owner,g,&m.config,&m.weights,budget,host));
 assert(!bn_backend_layout_prepare_bf16_flash_gdn_model(owner,g,&m.config,&m.weights,budget,host));
 assert(!bn_backend_layout_prepare_bf16_flash_qsa_model(owner,g,&m.config,&m.weights,budget,host));
 if(only){hc_guards(&m,g,argv[5],argv[6]);bn_model_free(&m);bn_gpu_cuda_destroy(g);bn_gguf_free(f);return 0;}
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
    BnTransformerBF16FlashHCViews hb,ha;
    assert(!bn_transformer_bf16_flash_hc_views(s,p->nt,&hb));
    char hidden_file[4096];path(hidden_file,sizeof(hidden_file),argv[3],p->prefix,"hc-hidden");size_t hidden_bytes;
    void *hidden=load(hidden_file,&hidden_bytes);assert(hidden_bytes==hb.hidden.count*2);
    assert(!g->resident_write(r,hb.hidden,hidden));free(hidden);
    int capture=repeat==repeats-1&&pi==profile;
#ifdef BN_CAPTURE_CUDA_PROFILER
    if(capture){assert(cudaProfilerStart()==cudaSuccess);profiled++;}
#else
    assert(!capture);
#endif
    real_sync=g->resident_synchronize;real_alloc=g->resident_alloc;
    int(*write)(void*,BnGPUResidentView,const void*)=g->resident_write;
    g->resident_synchronize=counted_sync;sync_calls=0;g->resident_alloc=unexpected_alloc;g->resident_write=unexpected_write;
    assert(!bn_transformer_bf16_flash_begin_blocks(s,p->nt,p->pos,p->prefill,UINT64_C(1)<<3,0,0));
    assert(bn_transformer_bf16_flash_gdn_submit(s,3)==-1);
    assert(bn_transformer_bf16_flash_qsa_submit(s,7)==-1);
    assert(bn_transformer_bf16_flash_complete(s)==-1&&!sync_calls);
    assert(bn_transformer_bf16_flash_qsa_submit(s,3)==-1);
    assert(!bn_transformer_bf16_flash_attention_submit(s,3));
    assert(!bn_transformer_bf16_flash_hc_views(s,p->nt,&ha)&&ha.hidden.slot!=hb.hidden.slot&&ha.residual.slot==hb.hidden.slot);
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
    BnGPUResidentView hv[]={ha.residual,ha.normalized,ha.down,ha.scaled,ha.gate,ha.inject,ha.combined};
    const char *hn[]={"hc-hidden","hc-norm","hc-down","hc-scaled","hc-up","hc-inject","hc-combined"};
    for(int j=0;j<7;j++)words+=compare(g,r,hv[j],argv[3],p->prefix,hn[j],2);
    indices+=compare(g,r,after.indices,argv[3],p->prefix,"indices",4);
    words+=compare(g,r,bn_gpu_resident_view_slice(after.key_cache,0,(size_t)after.position*512),argv[3],p->prefix,"key_cache",2);
    words+=compare(g,r,bn_gpu_resident_view_slice(after.value_cache,0,(size_t)after.position*512),argv[3],p->prefix,"value_cache",2);
    printf("phase%d %s production request exact committed%d\n",pi,p->prefix,after.position);fflush(stdout);
   }
  }
  bn_backend_session_free(s);first=end;cases++;
 }
 assert(profile<0||profiled==1);bn_model_free(&m);bn_gpu_cuda_destroy(g);bn_gguf_free(f);
 printf("HC_QSA_PRODUCTION_REQUEST_PASSED cases%d phases%d bf16_words%zu index_words%zu\n",cases,np,words,indices);return 0;
}
