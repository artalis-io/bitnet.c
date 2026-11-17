/* Actual Flash natural-head GDN block comparison, decoded-BF16 profile.
 * Link like test_cuda_backend; run FIRST_SHARD CAPTURE_DIRECTORY COEFFICIENT_DIRECTORY [--coeff-only]. */
#include "gpu_cuda.h"
#include "backend_quant.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "backend_session.h"
#include "model_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef BN_ENABLE_MKL_VML
#include <mkl.h>
#endif
static uint16_t bf16(float x){uint32_t u;memcpy(&u,&x,4);return(uint16_t)((u+0x7fff+((u>>16)&1))>>16);}
static BnGPUResidentView alloc(BnGPUBackend*g,void*r,int type,size_t n,int zero){BnGPUResidentView v;assert(!g->resident_alloc(r,type,n,&v));if(zero)assert(!g->resident_zero(r,v));return v;}
static BnGPUResidentView slice(BnGPUResidentView v,size_t n){return bn_gpu_resident_view_slice(v,0,n);}
static void linear(BnGPUBackend*g,void*r,void*w,int nt,int rows,int cols,BnGPUResidentView x,BnGPUResidentView y){BnGPUBF16LinearResidentPlan p={0};p.n_tokens=nt;p.rows=rows;p.cols=cols;p.weight.type=BN_GGUF_TENSOR_BF16;p.weight.decoded=w;p.x=x;p.out=y;assert(!g->linear_bf16_resident(g->ctx,r,&p));}
static size_t compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*prefix,const char*stage,int bytes){
 char path[2048];snprintf(path,sizeof(path),"%s/%s-%s.bin",dir,prefix,stage);FILE*f=fopen(path,"rb");assert(f);
 void*expected=malloc(v.count*(size_t)bytes),*out=malloc(v.count*(size_t)bytes);assert(expected&&out);assert(fread(expected,bytes,v.count,f)==v.count&&fgetc(f)==EOF);fclose(f);assert(!g->resident_read(r,v,out));
 size_t d=0;for(size_t i=0;i<v.count;i++){uint32_t a=bytes==2?((uint16_t*)out)[i]:((uint32_t*)out)[i],b=bytes==2?((uint16_t*)expected)[i]:((uint32_t*)expected)[i];if(a!=b){if(d<3)fprintf(stderr,"%s %s %zu %08x expected%08x\n",prefix,stage,i,a,b);d++;}}
 printf("%s %s words%zu differences%zu\n",prefix,stage,v.count,d);fflush(stdout);assert(!d);free(out);free(expected);return v.count;
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
 if(argc==5){assert(!strcmp(argv[4],"--coeff-only"));bn_model_free(&m);bn_gguf_free(f);return 0;}
 BnGPUBackend*g=bn_gpu_cuda_create();assert(g);BnBackendModel*owner=bn_model_backend(&m);assert(owner);
 assert(!bn_backend_layout_prepare_bf16_hc_model(owner,g,&m.config,&m.weights,SIZE_MAX));
 assert(!bn_backend_layout_prepare_bf16_ple_model(owner,g,&m.config,&m.weights,SIZE_MAX,SIZE_MAX));
 assert(!bn_backend_layout_prepare_bf16_flash_gdn_model(owner,g,&m.config,&m.weights,SIZE_MAX,SIZE_MAX));
 const BnBackendBF16FlashGDNModel*owned=bn_backend_layout_bf16_flash_gdn_model(owner);assert(owned&&owned->n_gdn_layers==36&&!bn_backend_layout_bf16(owner));
 assert(owned->bytes==4173024384u&&owned->peak_bytes==4189162080u&&owned->host_staging_bytes==44564480u);
 printf("FLASH_GDN_OWNED bytes%zu peak%zu host%zu model_bytes%zu model_peak%zu\n",owned->bytes,owned->peak_bytes,owned->host_staging_bytes,owned->model_bytes,owned->model_peak_bytes);
 for(int l=0;l<m.config.n_layers;l++)assert(owned->layers[l].valid==(m.weights.layers[l].block_kind==BN_LAYER_BLOCK_SSM));
 assert(owned->gate_activation==BN_GPU_GATED_NORM_SIGMOID&&owned->math_profile==BN_BACKEND_BF16_MATH_MKL_VML_HA);
 const BnBackendBF16FlashGDNWeights*cached=&owned->layers[0];assert(cached->valid);
 void*qkvz=cached->qkvz.decoded,*ba_w=cached->ba.decoded,*out_w=cached->out.decoded,*wc=cached->conv,*wn=cached->norm,*dt=cached->dt_bias;
 float al[48];for(int h=0;h<48;h++){assert(w->ssm_a[h]<0);al[h]=-w->ssm_a[h];}
#ifdef BN_ENABLE_MKL_VML
 vmsLn(48,al,al,VML_HA|VML_FTZDAZ_OFF|VML_ERRMODE_IGNORE);
#else
#error Independent Flash coefficient profile requires MKL_VML_HA
#endif
 char path[2048];snprintf(path,sizeof(path),"%s/a-log.bin",argv[2]);FILE*af=fopen(path,"rb");assert(af);float gold[48];assert(fread(gold,4,48,af)==48&&fgetc(af)==EOF);fclose(af);assert(!memcmp(al,gold,sizeof(al)));
 void*a=cached->a_log;assert(a);
 const int nts[]={1,5,129,129,1025,1025},chunks[]={1,5,129,64,1025,512};size_t bf16_words=0,f32_words=0;
 for(int ci=0;ci<6;ci++){
  int nt=nts[ci],cap=chunks[ci];BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);
  assert(!g->linear_bf16_reserve(g->ctx,r,0));assert(!g->gdn_prefill_bf16_reserve(g->ctx,r,cap));
  BnGPUResidentView input=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*2560,0),proj=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*16384,0),ba=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*96,0);
  BnGPUResidentView qkv=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*10240,0),z=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*6144,0),post=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*10240,0);
  BnGPUResidentView q=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*2048,0),k=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*2048,0),v=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*6144,0),ld=alloc(g,r,BN_GGUF_TENSOR_F32,(size_t)cap*48,0),beta=alloc(g,r,BN_GGUF_TENSOR_F32,(size_t)cap*48,0);
  BnGPUResidentView core=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*6144,0),norm=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*6144,0),out=alloc(g,r,BN_GGUF_TENSOR_BF16,(size_t)cap*2560,0);
  BnGPUResidentView history[]={alloc(g,r,BN_GGUF_TENSOR_BF16,3u*10240,1),alloc(g,r,BN_GGUF_TENSOR_BF16,3u*10240,0)},state[]={alloc(g,r,BN_GGUF_TENSOR_F32,48u*128*128,1),alloc(g,r,BN_GGUF_TENSOR_F32,48u*128*128,0)};
  int parity=0,phase=0,pos=0;
  while(pos<nt+8){
   int prefill=pos<nt,take=prefill?(nt-pos<cap?nt-pos:cap):1;char prefix[64];snprintf(prefix,sizeof(prefix),"%d-%d-%d",nt,cap,phase);
   size_t ni=(size_t)take*2560;uint16_t*x=malloc(ni*2);assert(x);for(size_t i=0;i<ni;i++)x[i]=bf16((float)((int)((i*17+i/37+phase*11)%509)-254)/128);assert(!g->resident_write(r,slice(input,ni),x));free(x);
   linear(g,r,qkvz,take,16384,2560,slice(input,ni),slice(proj,(size_t)take*16384));linear(g,r,ba_w,take,96,2560,slice(input,ni),slice(ba,(size_t)take*96));
   BnGPUSplitColumnsResidentPlan split={0};split.rows=take;split.cols=16384;split.left_cols=10240;split.x=slice(proj,(size_t)take*16384);split.left=slice(qkv,(size_t)take*10240);split.right=slice(z,(size_t)take*6144);assert(!g->split_columns_resident(g->ctx,r,&split));
   BnGPUCausalConvResidentPlan cv={0};cv.n_tokens=take;cv.dim=10240;cv.kernel=4;cv.weight=wc;cv.x=split.left;cv.initial_history=history[parity];cv.next_history=history[1-parity];cv.out=slice(post,(size_t)take*10240);assert(!g->causal_conv_bf16_resident(g->ctx,r,&cv));
   BnGPUResidentView co=slice(core,(size_t)take*6144);
   if(prefill){
    BnGPUGDNPrepareResidentPlan p={0};p.n_tokens=take;p.hk=16;p.dk=p.dv=128;p.hv=48;p.norm_eps=1e-6f;p.softplus_threshold=20;p.a_log=a;p.dt_bias=dt;p.x=cv.out;p.ba=slice(ba,(size_t)take*96);p.q=slice(q,(size_t)take*2048);p.k=slice(k,(size_t)take*2048);p.v=slice(v,(size_t)take*6144);p.log_decay=slice(ld,(size_t)take*48);p.beta=slice(beta,(size_t)take*48);assert(!g->gdn_prepare_bf16_resident(g->ctx,r,&p));
    BnGPUGDNPrefillResidentPlan d={0};d.n_tokens=take;d.hk=16;d.hv=48;d.dk=d.dv=128;d.q_scale=0x1.6a09e6p-4f;d.q=p.q;d.k=p.k;d.v=p.v;d.log_decay=p.log_decay;d.beta=p.beta;d.initial_state=state[parity];d.next_state=state[1-parity];d.out=co;assert(!g->gdn_prefill_bf16_resident(g->ctx,r,&d));
   }else{
    BnGPUGDNDecodeResidentPlan d={0};d.hk=16;d.hv=48;d.dk=d.dv=128;d.q_scale=0x1.6a09e6p-4f;d.a_log=a;d.dt_bias=dt;d.x=cv.out;d.ba=slice(ba,96);d.initial_state=state[parity];d.next_state=state[1-parity];d.out=co;assert(!g->gdn_decode_bf16_resident(g->ctx,r,&d));
   }
   BnGPUGatedRMSNormResidentPlan gn={0};gn.rows=take*48;gn.dim=128;gn.eps=owned->norm_eps;gn.activation=owned->gate_activation;gn.weight=wn;gn.x=co;gn.z=split.right;gn.out=slice(norm,(size_t)take*6144);assert(!g->gated_rmsnorm_bf16_resident(g->ctx,r,&gn));
   linear(g,r,out_w,take,2560,6144,gn.out,slice(out,(size_t)take*2560));assert(!g->resident_synchronize(r));parity=1-parity;
   const char*names[]={"input","qkvz","ba","post","core","norm","out","history"};BnGPUResidentView views[]={slice(input,ni),slice(proj,(size_t)take*16384),slice(ba,(size_t)take*96),cv.out,co,gn.out,slice(out,(size_t)take*2560),history[parity]};
   for(int j=0;j<8;j++)bf16_words+=compare(g,r,views[j],argv[2],prefix,names[j],2);
   f32_words+=compare(g,r,state[parity],argv[2],prefix,"state",4);phase++;pos+=take;
  }
  bn_backend_session_free(s);
 }
 bn_model_free(&m);bn_gpu_cuda_destroy(g);bn_gguf_free(f);
 printf("FLASH_GDN_PASSED bf16_words%zu f32_words%zu\n",bf16_words,f32_words);return 0;
}
