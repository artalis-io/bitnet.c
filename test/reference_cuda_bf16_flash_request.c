/* Production Flash PLE/GDN request comparison, decoded-BF16 profile.
 * Link like test_cuda_backend; run FIRST_SHARD CAPTURE_DIRECTORY COEFFICIENT_DIRECTORY [PLE_CAPTURE_DIRECTORY | --coeff-only]. */
#include "gpu_cuda.h"
#include "backend_quant.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "backend_session.h"
#include "model_internal.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef BN_ENABLE_MKL_VML
#include <mkl.h>
#endif
static uint16_t bf16(float x){uint32_t u;memcpy(&u,&x,4);return(uint16_t)((u+0x7fff+((u>>16)&1))>>16);}
static size_t compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*prefix,const char*stage,int bytes){
 char path[2048];snprintf(path,sizeof(path),"%s/%s-%s.bin",dir,prefix,stage);FILE*f=fopen(path,"rb");assert(f);
 void*expected=malloc(v.count*(size_t)bytes),*out=malloc(v.count*(size_t)bytes);assert(expected&&out);assert(fread(expected,bytes,v.count,f)==v.count&&fgetc(f)==EOF);fclose(f);assert(!g->resident_read(r,v,out));
 size_t d=0;for(size_t i=0;i<v.count;i++){uint32_t a=bytes==2?((uint16_t*)out)[i]:((uint32_t*)out)[i],b=bytes==2?((uint16_t*)expected)[i]:((uint32_t*)expected)[i];if(a!=b){if(d<3)fprintf(stderr,"%s %s %zu %08x expected%08x\n",prefix,stage,i,a,b);d++;}}
 printf("%s %s words%zu differences%zu\n",prefix,stage,v.count,d);fflush(stdout);assert(!d);free(out);free(expected);return v.count;
}
static void pattern(uint16_t*x,size_t n,int role){for(size_t i=0;i<n;i++)x[i]=bf16(ldexpf((float)((int)((i*(13+role*6)+role*19)%509)-254)/128,(int)(i%7)-3));}
static int busy(void*r,unsigned slot,BnGPUHostUploadLease*out){(void)r;(void)slot;(void)out;return BN_GPU_UPLOAD_BUSY;}
static int failed_sync(void*r){(void)r;return -1;}
static int unexpected_sync(void*r){(void)r;assert(0&&"submission/rejection must not fence");return -1;}
static int sync_calls;
static int(*real_sync)(void*);
static int counted_sync(void*r){sync_calls++;return real_sync(r);}
static int fail_alloc_at,alloc_calls;
static int(*real_alloc)(void*,int,size_t,BnGPUResidentView*);
static int fail_alloc(void*r,int type,size_t n,BnGPUResidentView*v){return ++alloc_calls==fail_alloc_at?-1:real_alloc(r,type,n,v);}
static void rollback(BnModel*m,BnGPUBackend*g) {
 BnBackendSession*s=bn_backend_session_create();assert(s);real_alloc=g->resident_alloc;g->resident_alloc=fail_alloc;
 for(fail_alloc_at=1;fail_alloc_at<=171;fail_alloc_at++){
  alloc_calls=0;assert(bn_transformer_bf16_flash_prepare(m,g,s,1,SIZE_MAX,SIZE_MAX)==-1);
  assert(alloc_calls==fail_alloc_at&&!bn_backend_session_has_gpu_resources(s)&&!bn_backend_session_has_execution_resources(s));
 }
 alloc_calls=0;assert(!bn_transformer_bf16_flash_prepare(m,g,s,1,SIZE_MAX,SIZE_MAX)&&alloc_calls==171);
 g->resident_alloc=real_alloc;bn_backend_session_free(s);
 s=bn_backend_session_create();assert(s);assert(!bn_transformer_bf16_ple_prepare(m,g,s,1,SIZE_MAX,SIZE_MAX));
 void*existing=bn_gpu_backend_resident_request(g,s);assert(existing);
 assert(bn_transformer_bf16_flash_prepare(m,g,s,1,SIZE_MAX,SIZE_MAX)==-1);
 assert(bn_gpu_backend_resident_request(g,s)==existing&&bn_backend_session_has_execution_resources(s));
 bn_backend_session_free(s);puts("FLASH_REQUEST_ALLOCATION_ROLLBACK 171 points PASSED");
}
static void combined(BnModel*m,BnGPUBackend*g,const char*dir,const char*pledir) {
 BnBackendSession*s=bn_backend_session_create();assert(s);assert(!bn_transformer_bf16_flash_prepare(m,g,s,5,SIZE_MAX,SIZE_MAX));
 void*r=bn_gpu_backend_resident_request(g,s);assert(r);
 BnTransformerBF16PLEViews pv;BnTransformerBF16FlashGDNViews gv;
 assert(!bn_transformer_bf16_ple_views(s,5,&pv));assert(!bn_transformer_bf16_flash_gdn_views(s,0,5,&gv));
 uint16_t*initial=malloc(9u*10240*2),*hidden=malloc(5u*10240*2),*input=malloc(5u*2560*2);assert(initial&&hidden&&input);
 pattern(initial,9u*10240,4);pattern(hidden,5u*10240,2);
 for(size_t i=0;i<5u*2560;i++)input[i]=bf16((float)((int)((i*17+i/37)%509)-254)/128);
 int32_t ctx[]={248044,116},tokens[]={97,134,171,208,245};
 assert(!bn_transformer_bf16_ple_reset(s,0,ctx,2,initial,9u*10240));
 assert(!g->resident_write(r,pv.hidden,hidden));assert(!g->resident_write(r,gv.input,input));
 BnTransformerBF16FlashInfo before,after;assert(!bn_transformer_bf16_flash_info(s,&before));
 int(*acquire)(void*,unsigned,BnGPUHostUploadLease*)=g->resident_upload_acquire;
 real_sync=g->resident_synchronize;g->resident_synchronize=unexpected_sync;g->resident_upload_acquire=busy;
 assert(bn_transformer_bf16_flash_begin(s,5,0,1,1,1)==BN_GPU_UPLOAD_BUSY);
 g->resident_upload_acquire=acquire;
 assert(!bn_transformer_bf16_flash_info(s,&after)&&after.active==before.active&&after.position==before.position&&after.n_tokens==before.n_tokens&&after.failed==before.failed&&after.expected_layers==before.expected_layers&&after.submitted_layers==before.submitted_layers);
 assert(bn_transformer_bf16_flash_begin(s,5,0,0,1,1)==-1);
 assert(bn_transformer_bf16_flash_begin(s,5,0,1,UINT64_MAX,1)==-1);
 assert(!bn_transformer_bf16_flash_begin(s,5,0,1,1,1));
 assert(bn_transformer_bf16_flash_begin(s,5,0,1,1,1)==-1);
 assert(bn_transformer_bf16_flash_complete(s)==-1);
 assert(!bn_transformer_bf16_flash_gdn_submit(s,0));
 assert(bn_transformer_bf16_flash_complete(s)==-1);
 assert(bn_transformer_bf16_ple_complete(s)==-1);
 int32_t invalid=-1;assert(bn_transformer_bf16_ple_submit(s,&invalid,1,0)==-1);
 assert(!bn_transformer_bf16_ple_submit(s,tokens,5,0));
 assert(bn_transformer_bf16_ple_submit(s,tokens,5,0)==-1);
 assert(bn_transformer_bf16_ple_reset(s,0,NULL,0,NULL,0)==-1);
 assert(bn_transformer_bf16_flash_restart(s)==-1);
 assert(!bn_transformer_bf16_flash_gdn_views(s,0,5,&gv)&&!gv.position);
 BnTransformerBF16PLEInfo pi;assert(!bn_transformer_bf16_ple_info(s,&pi)&&!pi.position&&pi.pending_tokens==5);
 g->resident_synchronize=counted_sync;sync_calls=0;
 assert(!bn_transformer_bf16_flash_complete(s)&&sync_calls==1);g->resident_synchronize=real_sync;
 assert(!bn_transformer_bf16_flash_gdn_views(s,0,5,&gv)&&gv.position==5);
 assert(!bn_transformer_bf16_ple_info(s,&pi)&&pi.position==5&&!pi.pending_tokens);
 assert(!bn_transformer_bf16_ple_views(s,5,&pv));
 compare(g,r,gv.out,dir,"5-5-0","out",2);compare(g,r,gv.state,dir,"5-5-0","state",4);
 const char*names[]={"embedding","key","value","gated","normed","out","hidden","history"};
 BnGPUResidentView views[]={pv.embedding,pv.key,pv.value,pv.gated,pv.normed,pv.out,pv.hidden,pv.history};
 for(int j=0;j<8;j++)compare(g,r,views[j],pledir,"5-5",names[j],2);
 char path[2048];snprintf(path,sizeof(path),"%s/5-5-ids.bin",pledir);FILE*idf=fopen(path,"rb");assert(idf);int32_t ids[80];assert(fread(ids,4,80,idf)==80&&fgetc(idf)==EOF);fclose(idf);
 assert(pv.row_id_count==80&&!memcmp(ids,pv.row_ids,sizeof(ids)));
 assert(bn_transformer_bf16_flash_begin(s,1,0,0,1,0)==-1);
 assert(!bn_transformer_bf16_flash_begin(s,1,5,0,1,0));assert(!bn_transformer_bf16_flash_gdn_submit(s,0));
 g->resident_synchronize=failed_sync;assert(bn_transformer_bf16_flash_complete(s)==-1);g->resident_synchronize=real_sync;
 assert(!bn_transformer_bf16_flash_info(s,&after)&&after.failed&&after.position==5&&after.active);
 assert(!bn_transformer_bf16_ple_info(s,&pi)&&pi.position==5);
 compare(g,r,gv.state,dir,"5-5-0","state",4);
 assert(bn_transformer_bf16_flash_restart(s)==-1);assert(bn_transformer_bf16_flash_begin(s,1,5,0,1,0)==-1);
 bn_backend_session_release_gpu_resources(s);assert(!bn_backend_session_has_execution_resources(s));
 assert(!bn_transformer_bf16_flash_prepare(m,g,s,1,SIZE_MAX,SIZE_MAX));
 bn_backend_session_free(s);free(input);free(hidden);free(initial);
 puts("FLASH_REQUEST_COMBINED_PASSED independent PLE/GDN, prequeue BUSY, single fence, atomic failure");
}
static void isolation(BnModel*m,BnGPUBackend*g,const char*dir) {
 BnBackendSession*a=bn_backend_session_create(),*b=bn_backend_session_create();assert(a&&b);
 assert(!bn_transformer_bf16_flash_prepare(m,g,a,1,SIZE_MAX,SIZE_MAX));
 assert(!bn_transformer_bf16_flash_prepare(m,g,b,5,SIZE_MAX,SIZE_MAX));
 BnTransformerBF16FlashGDNViews av,bv;
 assert(!bn_transformer_bf16_flash_gdn_views(a,0,1,&av));assert(!bn_transformer_bf16_flash_gdn_views(b,0,5,&bv));assert(av.input.request!=bv.input.request);
 uint16_t*x=malloc(5u*2560*2);assert(x);for(size_t i=0;i<5u*2560;i++)x[i]=bf16((float)((int)((i*17+i/37)%509)-254)/128);
 assert(!g->resident_write(av.input.request,av.input,x));assert(!g->resident_write(bv.input.request,bv.input,x));free(x);
 assert(!bn_transformer_bf16_flash_begin(a,1,0,1,1,0));assert(!bn_transformer_bf16_flash_gdn_submit(a,0));
 assert(!bn_transformer_bf16_flash_begin(b,5,0,1,1,0));assert(!bn_transformer_bf16_flash_gdn_submit(b,0));
 assert(!bn_transformer_bf16_flash_complete(b));
 assert(!bn_transformer_bf16_flash_gdn_views(a,0,1,&av)&&av.position==0);
 assert(!bn_transformer_bf16_flash_gdn_views(b,0,5,&bv)&&bv.position==5);
 compare(g,bv.out.request,bv.out,dir,"5-5-0","out",2);compare(g,bv.state.request,bv.state,dir,"5-5-0","state",4);
 assert(!bn_transformer_bf16_flash_complete(a));assert(!bn_transformer_bf16_flash_gdn_views(a,0,1,&av)&&av.position==1);
 compare(g,av.out.request,av.out,dir,"1-1-0","out",2);compare(g,av.state.request,av.state,dir,"1-1-0","state",4);
 bn_backend_session_free(a);bn_backend_session_free(b);puts("FLASH_REQUEST_ISOLATION_PASSED reversed completion, distinct inputs/capacities");
}
static void all_layers(BnModel*m,BnGPUBackend*g,const char*dir) {
 BnBackendSession*s=bn_backend_session_create();assert(s);assert(!bn_transformer_bf16_flash_prepare(m,g,s,1,SIZE_MAX,SIZE_MAX));
 BnTransformerBF16FlashInfo info;assert(!bn_transformer_bf16_flash_info(s,&info));
 BnTransformerBF16FlashGDNViews views[64];
 for(int l=0;l<m->config.n_layers;l++)if(info.valid_layers&(UINT64_C(1)<<l)){
  assert(!bn_transformer_bf16_flash_gdn_views(s,l,1,&views[l])&&!views[l].position);
  for(int j=0;j<l;j++)if(info.valid_layers&(UINT64_C(1)<<j)){
   assert(views[l].state.slot!=views[j].state.slot&&views[l].history.slot!=views[j].history.slot);
  }
 }
 uint16_t input[2560];for(size_t i=0;i<2560;i++)input[i]=bf16((float)((int)((i*17+i/37)%509)-254)/128);
 assert(!g->resident_write(views[0].input.request,views[0].input,input));
 assert(!bn_transformer_bf16_flash_begin(s,1,0,1,info.valid_layers,0));
 real_sync=g->resident_synchronize;g->resident_synchronize=unexpected_sync;
 for(int l=0;l<m->config.n_layers;l++)if(info.valid_layers&(UINT64_C(1)<<l)){
  assert(!bn_transformer_bf16_flash_gdn_submit(s,l));
  assert(!bn_transformer_bf16_flash_gdn_views(s,l,1,&views[l])&&!views[l].position);
  BnTransformerBF16FlashInfo pending;assert(!bn_transformer_bf16_flash_info(s,&pending));
  if(pending.submitted_layers!=pending.expected_layers)assert(bn_transformer_bf16_flash_complete(s)==-1);
 }
 g->resident_synchronize=counted_sync;sync_calls=0;assert(!bn_transformer_bf16_flash_complete(s)&&sync_calls==1);g->resident_synchronize=real_sync;
 for(int l=0;l<m->config.n_layers;l++)if(info.valid_layers&(UINT64_C(1)<<l))assert(!bn_transformer_bf16_flash_gdn_views(s,l,1,&views[l])&&views[l].position==1);
 compare(g,views[0].state.request,views[0].state,dir,"1-1-0","state",4);
 assert(!bn_transformer_bf16_flash_restart(s));
 for(int l=0;l<m->config.n_layers;l++)if(info.valid_layers&(UINT64_C(1)<<l))assert(!bn_transformer_bf16_flash_gdn_views(s,l,1,&views[l])&&!views[l].position);
 bn_backend_session_free(s);puts("FLASH_REQUEST_ALL_LAYER_TRANSACTION_PASSED 36 disjoint banks, one fence; layer0 independent oracle only");
}
int main(int argc,char**argv){
 assert(argc==4||argc==5);BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModel m={0};assert(!bn_model_load_with_cpu_preparation(&m,f,2048,0,0,0));
 assert(m.config.dim==2560&&bn_model_transformer_policy_ssm_uses_sigmoid_gate(&m.config));BnSSMWeights*w=&m.weights.layers[0].ssm;
 size_t coefficient_words=0;char coeffile[2048];
 for(int layer=0;layer<m.config.n_layers;layer++) {
  if(m.weights.layers[layer].block_kind!=BN_LAYER_BLOCK_SSM)continue;
  float values[48],expected[48];for(int h=0;h<48;h++){assert(m.weights.layers[layer].ssm.ssm_a[h]<0);values[h]=-m.weights.layers[layer].ssm.ssm_a[h];}
  vmsLn(48,values,values,VML_HA|VML_FTZDAZ_OFF|VML_ERRMODE_IGNORE);
  snprintf(coeffile,sizeof(coeffile),"%s/a-log-%d.bin",argv[3],layer);FILE*cf=fopen(coeffile,"rb");assert(cf);assert(fread(expected,4,48,cf)==48&&fgetc(cf)==EOF);fclose(cf);
  if(memcmp(values,expected,sizeof(values)))fprintf(stderr,"Flash coefficient layer%d differs\n",layer);
  assert(!memcmp(values,expected,sizeof(values)));coefficient_words+=48;
 }
 assert(coefficient_words==1728);printf("FLASH_COEFFICIENTS_PASSED words%zu layers36\n",coefficient_words);
 if(argc==5&&!strcmp(argv[4],"--coeff-only")){bn_model_free(&m);bn_gguf_free(f);return 0;}
 BnGPUBackend*g=bn_gpu_cuda_create();assert(g);BnBackendModel*owner=bn_model_backend(&m);assert(owner);
 assert(!bn_backend_layout_prepare_bf16_hc_model(owner,g,&m.config,&m.weights,SIZE_MAX));
 assert(!bn_backend_layout_prepare_bf16_ple_model(owner,g,&m.config,&m.weights,SIZE_MAX,SIZE_MAX));
 assert(!bn_backend_layout_prepare_bf16_flash_gdn_model(owner,g,&m.config,&m.weights,SIZE_MAX,SIZE_MAX));
 const BnBackendBF16FlashGDNModel*owned=bn_backend_layout_bf16_flash_gdn_model(owner);assert(owned&&owned->n_gdn_layers==36&&!bn_backend_layout_bf16(owner));
 assert(owned->bytes==4173024384u&&owned->peak_bytes==4189162080u&&owned->host_staging_bytes==44564480u);
 printf("FLASH_GDN_OWNED bytes%zu peak%zu host%zu model_bytes%zu model_peak%zu\n",owned->bytes,owned->peak_bytes,owned->host_staging_bytes,owned->model_bytes,owned->model_peak_bytes);
 for(int l=0;l<m.config.n_layers;l++)assert(owned->layers[l].valid==(m.weights.layers[l].block_kind==BN_LAYER_BLOCK_SSM));
 assert(owned->gate_activation==BN_GPU_GATED_NORM_SIGMOID&&owned->math_profile==BN_BACKEND_BF16_MATH_MKL_VML_HA);

 float al[48];for(int h=0;h<48;h++){assert(w->ssm_a[h]<0);al[h]=-w->ssm_a[h];}
#ifdef BN_ENABLE_MKL_VML
 vmsLn(48,al,al,VML_HA|VML_FTZDAZ_OFF|VML_ERRMODE_IGNORE);
#else
#error Independent Flash coefficient profile requires MKL_VML_HA
#endif
 char path[2048];snprintf(path,sizeof(path),"%s/a-log.bin",argv[2]);FILE*af=fopen(path,"rb");assert(af);float gold[48];assert(fread(gold,4,48,af)==48&&fgetc(af)==EOF);fclose(af);assert(!memcmp(al,gold,sizeof(al)));

 rollback(&m,g);combined(&m,g,argv[2],argc==5?argv[4]:"/tmp/bitnet-goal-ple-chain/captures");isolation(&m,g,argv[2]);all_layers(&m,g,argv[2]);
 const int nts[]={1,5,129,129,1025,1025},chunks[]={1,5,129,64,1025,512};size_t bf16_words=0,f32_words=0;
 for(int ci=0;ci<6;ci++){
  int nt=nts[ci],cap=chunks[ci];BnBackendSession*s=bn_backend_session_create();assert(s);
  size_t tensors=(size_t)cap*(180736+141888)+368640+230916096;
  size_t workspace=g->linear_bf16_workspace_size(g->ctx)+g->gdn_prefill_bf16_workspace_size(g->ctx,cap);
  size_t host=(size_t)cap*3136;
  assert(bn_transformer_bf16_flash_prepare(&m,g,s,cap,tensors+workspace-1,host)==-1);
  assert(bn_transformer_bf16_flash_prepare(&m,g,s,cap,tensors+workspace,host-1)==-1);
  assert(!bn_backend_session_has_gpu_resources(s)&&!bn_backend_session_has_execution_resources(s));
  assert(!bn_transformer_bf16_flash_prepare(&m,g,s,cap,tensors+workspace,host));
  void*r=bn_gpu_backend_resident_request(g,s);assert(r);
  BnTransformerBF16FlashInfo info;assert(!bn_transformer_bf16_flash_info(s,&info));
  assert(info.tensor_bytes==tensors&&info.workspace_bytes==workspace&&info.gpu_bytes==tensors+workspace&&info.host_bytes==host);
  assert(bn_transformer_bf16_flash_prepare(&m,g,s,cap,SIZE_MAX,SIZE_MAX)==-1);
  for(int repeat=0;repeat<2;repeat++) {
  assert(!bn_transformer_bf16_flash_restart(s));
  int phase=0,pos=0;
  while(pos<nt+8){
   int prefill=pos<nt,take=prefill?(nt-pos<cap?nt-pos:cap):1;char prefix[64];snprintf(prefix,sizeof(prefix),"%d-%d-%d",nt,cap,phase);
   BnTransformerBF16FlashGDNViews v;assert(!bn_transformer_bf16_flash_gdn_views(s,0,take,&v)&&v.position==pos);
   size_t ni=(size_t)take*2560;uint16_t*x=malloc(ni*2);assert(x);for(size_t i=0;i<ni;i++)x[i]=bf16((float)((int)((i*17+i/37+phase*11)%509)-254)/128);assert(!g->resident_write(r,v.input,x));free(x);
   assert(!bn_transformer_bf16_flash_begin(s,take,pos,prefill,1,0));
   assert(bn_transformer_bf16_flash_complete(s)==-1);
   assert(!bn_transformer_bf16_flash_gdn_submit(s,0));
   assert(bn_transformer_bf16_flash_gdn_submit(s,0)==-1);
   assert(!bn_transformer_bf16_flash_gdn_views(s,0,take,&v)&&v.position==pos);
   assert(!bn_transformer_bf16_flash_complete(s));
   assert(!bn_transformer_bf16_flash_gdn_views(s,0,take,&v)&&v.position==pos+take);
   const char*names[]={"input","qkvz","ba","post","core","norm","out","history"};BnGPUResidentView views[]={v.input,v.qkvz,v.ba,v.post,v.core,v.norm,v.out,v.history};
   for(int j=0;j<8;j++)bf16_words+=compare(g,r,views[j],argv[2],prefix,names[j],2);
   f32_words+=compare(g,r,v.state,argv[2],prefix,"state",4);phase++;pos+=take;
  }
  int other=1;while(other<owned->n_layers&&!owned->layers[other].valid)other++;assert(other<owned->n_layers);
  BnTransformerBF16FlashGDNViews untouched;assert(!bn_transformer_bf16_flash_gdn_views(s,other,1,&untouched)&&untouched.position==0);
  }
  bn_backend_session_free(s);
 }
 bn_model_free(&m);bn_gpu_cuda_destroy(g);bn_gguf_free(f);
 printf("FLASH_REQUEST_PASSED bf16_words%zu f32_words%zu\n",bf16_words,f32_words);return 0;
}
