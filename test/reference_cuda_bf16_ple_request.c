/* Production PLE request orchestration against immutable independent captures.
 * Link like test_cuda_backend; run FIRST_SHARD CAPTURE_DIRECTORY. */
#include "gpu_cuda.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "model_internal.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif
static uint16_t bf16(float x){uint32_t u;memcpy(&u,&x,4);return(uint16_t)((u+0x7fff+((u>>16)&1))>>16);}
static void pattern(uint16_t*x,size_t n,int role){for(size_t i=0;i<n;i++)x[i]=bf16(ldexpf((float)((int)((i*(13+role*6)+role*19)%509)-254)/128,(int)(i%7)-3));}
static size_t compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*name,const char*stage,size_t offset,size_t total){
 char path[2048];snprintf(path,sizeof(path),"%s/%s-%s.bin",dir,name,stage);FILE*f=fopen(path,"rb");assert(f);
 assert(!fseek(f,0,SEEK_END)&&(size_t)ftell(f)==total*2);assert(!fseek(f,(long)(offset*2),SEEK_SET));
 uint16_t*gold=malloc(v.count*2),*out=malloc(v.count*2);assert(gold&&out);assert(fread(gold,2,v.count,f)==v.count);fclose(f);
 assert(!g->resident_read(r,v,out));size_t differences=0;
 for(size_t i=0;i<v.count;i++)if(out[i]!=gold[i]){if(differences<3)fprintf(stderr,"%s %s word%zu %04x expected%04x\n",name,stage,offset+i,out[i],gold[i]);differences++;}
 assert(!differences);free(gold);free(out);return v.count;
}
static int busy(void*r,unsigned slot,BnGPUHostUploadLease*out){(void)r;(void)slot;(void)out;return BN_GPU_UPLOAD_BUSY;}
static int failed_sync(void*r){(void)r;return -1;}
static int unexpected_sync(void*r){(void)r;assert(0 && "BUSY must not synchronize");return -1;}
static int fail_alloc_at,alloc_calls;
static int(*real_alloc)(void*,int,size_t,BnGPUResidentView*);
static int fail_alloc(void*r,int type,size_t n,BnGPUResidentView*v){return ++alloc_calls==fail_alloc_at?-1:real_alloc(r,type,n,v);}
int main(int argc,char**argv){
 assert(argc==3);BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModel m={0};assert(!bn_model_load_with_cpu_preparation(&m,f,256,0,0,0));
 BnGPUBackend*g=bn_gpu_cuda_create();assert(g);BnBackendModel*owner=bn_model_backend(&m);assert(owner);
 assert(!bn_backend_layout_prepare_bf16_hc_model(owner,g,&m.config,&m.weights,SIZE_MAX));
 assert(!bn_backend_layout_prepare_bf16_ple_model(owner,g,&m.config,&m.weights,SIZE_MAX,SIZE_MAX));
 const BnBackendBF16PLEWeights*w=bn_backend_layout_bf16_ple_model(owner);assert(w&&w->dim==2560&&w->streams==4&&w->kernel==4&&w->dilation==3);
 assert(!bn_backend_layout_bf16(owner));
 BnBackendSession*failure=bn_backend_session_create();assert(failure);
 real_alloc=g->resident_alloc;g->resident_alloc=fail_alloc;
 for(fail_alloc_at=1;fail_alloc_at<=13;fail_alloc_at++){
  alloc_calls=0;assert(bn_transformer_bf16_ple_prepare(&m,g,failure,5,SIZE_MAX,SIZE_MAX)==-1);
  assert(!bn_backend_session_has_gpu_resources(failure)&&!bn_backend_session_has_execution_resources(failure));
  assert(bn_backend_layout_bf16_ple_model(owner)==w);
 }
 g->resident_alloc=real_alloc;bn_backend_session_free(failure);puts("PLE_REQUEST_ALLOCATION_ROLLBACK 13 points PASSED");
 const int nts[]={1,5,17,129,129,129},chunks[]={1,5,17,129,64,1};size_t words=0,ids_checked=0;
 uint16_t*initial=malloc(9u*10240*2);assert(initial);pattern(initial,9u*10240,4);int32_t ctx[]={248044,116};
 for(int ci=0;ci<6;ci++){
  int nt=nts[ci],chunk=chunks[ci];char name[64];snprintf(name,sizeof(name),"%d-%d",nt,chunk);
  BnBackendSession*s=bn_backend_session_create();assert(s);
  size_t tensor_bytes=(size_t)chunk*180736+368640,workspace=g->linear_bf16_workspace_size(g->ctx),host_bytes=(size_t)chunk*3136;
  assert(workspace==32u*1024*1024||workspace==4u*1024*1024);
  assert(bn_transformer_bf16_ple_prepare(&m,g,s,chunk,tensor_bytes+workspace-1,host_bytes)==-1);
  assert(bn_transformer_bf16_ple_prepare(&m,g,s,chunk,tensor_bytes+workspace,host_bytes-1)==-1);
  assert(!bn_backend_session_has_gpu_resources(s)&&!bn_backend_session_has_execution_resources(s));
  assert(!bn_transformer_bf16_ple_prepare(&m,g,s,chunk,tensor_bytes+workspace,host_bytes));
  assert(bn_transformer_bf16_ple_prepare(&m,g,s,chunk,SIZE_MAX,SIZE_MAX)==-1);
  void*r=bn_gpu_backend_resident_request(g,s);assert(r);
  BnTransformerBF16PLEInfo info;assert(!bn_transformer_bf16_ple_info(s,&info));
  assert(info.tensor_bytes==tensor_bytes&&info.workspace_bytes==workspace&&info.gpu_bytes==tensor_bytes+workspace&&info.host_bytes==host_bytes);
  printf("PLE_REQUEST_BUDGET %s gpu%zu tensor%zu workspace%zu host%zu\n",name,info.gpu_bytes,info.tensor_bytes,info.workspace_bytes,info.host_bytes);
  int32_t tokens[129],goldids[129*16];for(int i=0;i<nt;i++)tokens[i]=i%11==5?248044:(i*37+97)%248000;
  char path[2048];snprintf(path,sizeof(path),"%s/%s-ids.bin",argv[2],name);FILE*idf=fopen(path,"rb");assert(idf&&fread(goldids,4,(size_t)nt*16,idf)==(size_t)nt*16&&fgetc(idf)==EOF);fclose(idf);
  uint16_t*hidden=malloc((size_t)nt*10240*2);assert(hidden);pattern(hidden,(size_t)nt*10240,2);
  for(int repeat=0;repeat<2;repeat++){
   assert(!bn_transformer_bf16_ple_reset(s,0,NULL,0,NULL,0));
   BnTransformerBF16PLEViews v;assert(!bn_transformer_bf16_ple_views(s,1,&v));
   uint16_t*zero=malloc(v.history.count*2);assert(zero);assert(!g->resident_read(r,v.history,zero));for(size_t i=0;i<v.history.count;i++)assert(!zero[i]);free(zero);
   assert(bn_transformer_bf16_ple_reset(s,0,ctx,1,initial,9u*10240)==-1);
   assert(!bn_transformer_bf16_ple_reset(s,0,ctx,2,initial,9u*10240));
   for(int start=0;start<nt;start+=chunk){
    int take=nt-start<chunk?nt-start:chunk;assert(!bn_transformer_bf16_ple_views(s,take,&v));
    assert(!g->resident_write(r,v.hidden,hidden+(size_t)start*10240));
    if(start==0){
     int(*acquire)(void*,unsigned,BnGPUHostUploadLease*)=g->resident_upload_acquire;g->resident_upload_acquire=busy;
     int(*saved_sync)(void*)=g->resident_synchronize;g->resident_synchronize=unexpected_sync;
     assert(bn_transformer_bf16_ple_submit(s,tokens,take,0)==BN_GPU_UPLOAD_BUSY);g->resident_upload_acquire=acquire;g->resident_synchronize=saved_sync;
     assert(!bn_transformer_bf16_ple_info(s,&info)&&!info.position&&!info.pending_tokens&&!info.failed);
     int32_t invalid=-1;assert(bn_transformer_bf16_ple_submit(s,&invalid,1,0)==-1);
    }
    assert(!bn_transformer_bf16_ple_submit(s,tokens+start,take,start));
    assert(!bn_transformer_bf16_ple_info(s,&info)&&info.position==start&&info.pending_tokens==take&&!info.failed);
    assert(bn_transformer_bf16_ple_submit(s,tokens+start,take,start)==-1);
    assert(bn_transformer_bf16_ple_reset(s,0,NULL,0,NULL,0)==-1);
    assert(!bn_transformer_bf16_ple_complete(s));
    assert(!bn_transformer_bf16_ple_info(s,&info)&&info.position==start+take&&!info.pending_tokens&&!info.failed);
    assert(bn_transformer_bf16_ple_complete(s)==-1);
    assert(!bn_transformer_bf16_ple_views(s,take,&v));
    assert(v.row_id_count==(size_t)take*16&&!memcmp(v.row_ids,goldids+(size_t)start*16,v.row_id_count*4));ids_checked+=v.row_id_count;
    const char*stages[]={"embedding","key","value","gated","normed","out","hidden"};
    BnGPUResidentView views[]={v.embedding,v.key,v.value,v.gated,v.normed,v.out,v.hidden};
    for(int j=0;j<7;j++){size_t width=j==0||j==2?2560:10240;words+=compare(g,r,views[j],argv[2],name,stages[j],(size_t)start*width,(size_t)nt*width);}
   }
   assert(!bn_transformer_bf16_ple_views(s,1,&v));words+=compare(g,r,v.history,argv[2],name,"history",0,9u*10240);
   printf("PLE_REQUEST_EXACT %s repeat%d position%d\n",name,repeat,info.position);fflush(stdout);
  }
  free(hidden);bn_backend_session_free(s);
 }
 /* Interleave distinct requests sharing immutable weights. B has different
  * context, hidden and history; compare it with its isolated baseline. A keeps
  * its independent checkpoint oracle. Completion of B must not commit A. */
 {
  BnBackendSession*a=bn_backend_session_create(),*b=bn_backend_session_create();assert(a&&b);
  assert(!bn_transformer_bf16_ple_prepare(&m,g,a,5,SIZE_MAX,SIZE_MAX));
  assert(!bn_transformer_bf16_ple_prepare(&m,g,b,1,SIZE_MAX,SIZE_MAX));
  BnTransformerBF16PLEViews av,bv;assert(!bn_transformer_bf16_ple_views(a,5,&av));assert(!bn_transformer_bf16_ple_views(b,1,&bv));assert(av.hidden.request!=bv.hidden.request);
  uint16_t*ah=malloc(5u*10240*2),*bh=malloc(10240u*2),*history=malloc(9u*10240*2),*expected_out=malloc(10240u*2),*expected_history=malloc(9u*10240*2);assert(ah&&bh&&history&&expected_out&&expected_history);
  pattern(ah,5u*10240,2);pattern(bh,10240,0);pattern(history,9u*10240,3);int32_t bc[]={248044,117},bt=97,at[]={97,134,171,208,245};
  assert(!bn_transformer_bf16_ple_reset(b,0,bc,2,history,9u*10240));assert(!g->resident_write(bv.hidden.request,bv.hidden,bh));
  assert(!bn_transformer_bf16_ple_submit(b,&bt,1,0));assert(!bn_transformer_bf16_ple_complete(b));assert(!bn_transformer_bf16_ple_views(b,1,&bv));
  assert(!g->resident_read(bv.out.request,bv.out,expected_out));assert(!g->resident_read(bv.history.request,bv.history,expected_history));
  assert(!bn_transformer_bf16_ple_reset(a,0,ctx,2,initial,9u*10240));assert(!bn_transformer_bf16_ple_reset(b,0,bc,2,history,9u*10240));
  assert(!g->resident_write(av.hidden.request,av.hidden,ah));assert(!g->resident_write(bv.hidden.request,bv.hidden,bh));
  assert(!bn_transformer_bf16_ple_submit(a,at,5,0));assert(!bn_transformer_bf16_ple_submit(b,&bt,1,0));
  assert(!bn_transformer_bf16_ple_complete(b));BnTransformerBF16PLEInfo ai,bi;
  assert(!bn_transformer_bf16_ple_info(a,&ai)&&!ai.position&&ai.pending_tokens==5);
  assert(!bn_transformer_bf16_ple_info(b,&bi)&&bi.position==1&&!bi.pending_tokens);
  assert(!bn_transformer_bf16_ple_views(b,1,&bv));uint16_t*observed=malloc(9u*10240*2);assert(observed);
  assert(!g->resident_read(bv.out.request,bv.out,observed)&&!memcmp(observed,expected_out,10240u*2));
  assert(!g->resident_read(bv.history.request,bv.history,observed)&&!memcmp(observed,expected_history,9u*10240*2));
  assert(!bn_transformer_bf16_ple_complete(a));assert(!bn_transformer_bf16_ple_views(a,5,&av));
  compare(g,av.out.request,av.out,argv[2],"5-5","out",0,5u*10240);compare(g,av.history.request,av.history,argv[2],"5-5","history",0,9u*10240);
  free(observed);free(expected_history);free(expected_out);free(history);free(bh);free(ah);bn_backend_session_free(a);bn_backend_session_free(b);
  puts("PLE_REQUEST_ISOLATION_PASSED distinct contexts and histories, reversed completion order");
 }
 /* A reported completion failure must not commit position or permit reuse. */
 BnBackendSession*s=bn_backend_session_create();assert(s);assert(!bn_transformer_bf16_ple_prepare(&m,g,s,1,SIZE_MAX,SIZE_MAX));
 void*r=bn_gpu_backend_resident_request(g,s);BnTransformerBF16PLEViews v;assert(!bn_transformer_bf16_ple_views(s,1,&v));assert(!g->resident_zero(r,v.hidden));int32_t token=97;
 assert(!bn_transformer_bf16_ple_submit(s,&token,1,0));int(*synchronize)(void*)=g->resident_synchronize;g->resident_synchronize=failed_sync;
 assert(bn_transformer_bf16_ple_complete(s)==-1);g->resident_synchronize=synchronize;
 BnTransformerBF16PLEInfo info;assert(!bn_transformer_bf16_ple_info(s,&info)&&!info.position&&info.failed);
 assert(bn_transformer_bf16_ple_submit(s,&token,1,0)==-1);assert(bn_transformer_bf16_ple_reset(s,0,NULL,0,NULL,0)==-1);
 bn_backend_session_release_gpu_resources(s);assert(!bn_backend_session_has_execution_resources(s));
 assert(!bn_transformer_bf16_ple_prepare(&m,g,s,1,SIZE_MAX,SIZE_MAX));bn_backend_session_free(s);
 free(initial);bn_model_free(&m);bn_gpu_cuda_destroy(g);bn_gguf_free(f);
 printf("PLE_REQUEST_PASSED exact_words%zu ids%zu 12 reset/reuse cases\n",words,ids_checked);return 0;
}
