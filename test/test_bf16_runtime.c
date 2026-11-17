#include "model.h"
#include "model_internal.h"
#include "session.h"
#include "session_internal.h"
#include "transformer.h"
#include "generate.h"
#include "gpu_cuda.h"
#include "backend_layout.h"
#include "backend_quant.h"
#include "backend_model.h"
#include "backend_session.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Integration fixture: pinned Qwen27 Q4_K_XL. --native selects model-owned
 * rotary generation; an independent cache file remains a diagnostic override.
 * Requires real model data; not in make test. */
static BnGPUBackend *gpu;
static const char *dump_directory;
static void *reject_rotary(void *ctx, const BnGPURotaryCachePlan *plan) {
 (void)ctx; (void)plan; return NULL;
}
static size_t pool_kv_elements,pool_kv_allocated;
static int resize_calls,fail_resize;
static size_t resized_kv_bytes;
static int (*original_resize)(void*,BnGPUResidentView*,size_t);
static int count_resize(void *r,BnGPUResidentView *v,size_t n) {
 if(++resize_calls==fail_resize)return -1;
 assert(v->type==bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_BF16));
 int status=original_resize(r,v,n);if(!status)resized_kv_bytes+=n*2;return status;
}
static int alloc_calls, fail_alloc, write_calls, fail_write, read_calls, fail_read;
static int linear_calls, fail_linear, cpu_fallbacks, fail_reserve, foreign_releases;
static int sync_calls, fail_sync, zero_calls, fail_zero;
static int (*original_zero)(void*,BnGPUResidentView);
static int count_zero(void*r,BnGPUResidentView v){if(++zero_calls==fail_zero)return -1;return original_zero(r,v);}
static int (*original_sync)(void*);
static int count_sync(void*r){if(++sync_calls==fail_sync)return -1;return original_sync(r);}
typedef struct {const int *expected; int count;} GeneratedProbe;
static int collect_generated(const char*piece,int id,void*context){
 (void)piece;GeneratedProbe*p=context;assert(p->count<8&&id==p->expected[p->count]);p->count++;return 0;
}
static int (*original_reserve)(void*,void*,int,int,int);
static int count_reserve(void*c,void*r,int n,int h,int cap){if(fail_reserve)return -1;return original_reserve(c,r,n,h,cap);}
static void release_foreign(void*p){foreign_releases++;free(p);}
static int (*original_alloc)(void *,int,size_t,BnGPUResidentView *);
static int (*original_write)(void *,BnGPUResidentView,const void *);
static int (*original_read)(void *,BnGPUResidentView,void *);
static int (*original_linear)(void *,void *,const BnGPUBF16LinearResidentPlan *);
static size_t alloc_payload;
static int count_alloc(void*r,int t,size_t n,BnGPUResidentView*out){if(++alloc_calls==fail_alloc)return -1;int status=original_alloc(r,t,n,out);if(!status)alloc_payload+=n*bn_backend_quant_resident_scalar_bytes(t);if(!status&&pool_kv_elements&&n==pool_kv_elements&&t==bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_BF16))pool_kv_allocated+=n*2;return status;}
static int count_write(void*r,BnGPUResidentView v,const void*p){if(++write_calls==fail_write)return -1;return original_write(r,v,p);}
static int count_read(void*r,BnGPUResidentView v,void*p){if(++read_calls==fail_read)return -1;return original_read(r,v,p);}
static int count_linear(void*c,void*r,const BnGPUBF16LinearResidentPlan*p){if(++linear_calls==fail_linear)return -1;return original_linear(c,r,p);}
static int reject_cpu(void*ctx){(void)ctx;cpu_fallbacks++;return -1;}
static void reset_counts(void){alloc_payload=0;alloc_calls=write_calls=read_calls=linear_calls=0;fail_alloc=fail_write=fail_read=fail_linear=0;sync_calls=fail_sync=zero_calls=fail_zero=0;}
static int token(const float*x,int n){float best=-INFINITY;int id=0;for(int i=0;i<n;i++)if(x[i]>best){best=x[i];id=i;}return id;}
static void output(const char*name,BnGPUResidentView view,void*request){char path[512];snprintf(path,sizeof(path),"%s/%s.bin",dump_directory,name);void*data=malloc(view.count*2);assert(data&&gpu->resident_read(request,view,data)==0);FILE*f=fopen(path,"wb");assert(f&&fwrite(data,2,view.count,f)==view.count);fclose(f);free(data);}
static int observe(void*context,BnTransformerBF16Stage stage,int layer,int position,int nt,BnGPUResidentView view){
 (void)nt;if(position!=0&&position!=5)return 0;int phase=position==0?0:1;char name[128];
 switch(stage){
 case BN_TRANSFORMER_BF16_INPUT_NORM:if(layer>=4)return 0;snprintf(name,sizeof(name),layer?"norm-in-%d-%d-0":"norm-in-%d-%d",layer,phase);break;
 case BN_TRANSFORMER_BF16_FFN_NORM:if(layer>=4)return 0;snprintf(name,sizeof(name),"norm-ffn-%d-%d-0",layer,phase);break;
 case BN_TRANSFORMER_BF16_FFN_OUTPUT:snprintf(name,sizeof(name),"layer-%d-%d-0",layer,phase);break;
 case BN_TRANSFORMER_BF16_RESIDUAL:snprintf(name,sizeof(name),"layer-%d-%d-1",layer,phase);break;
 case BN_TRANSFORMER_BF16_FINAL_NORM:snprintf(name,sizeof(name),"final-norm-%d-0",phase);break;
 case BN_TRANSFORMER_BF16_LOGITS:snprintf(name,sizeof(name),"logits-%d",phase);break;
 default:return -1;
 }
 output(name,view,context);return 0;
}
typedef struct {void *request;int calls;} ProfileObserver;
static int profile_observer(void *context,BnTransformerBF16Stage stage,int layer,int position,int nt,BnGPUResidentView view) {
 (void)layer;(void)position;ProfileObserver *probe=context;probe->calls++;
 if(stage==BN_TRANSFORMER_BF16_RESIDUAL) {
  assert(view.count==(size_t)nt*5120);uint16_t *data=malloc(view.count*2);assert(data);
  assert(gpu->resident_read(probe->request,view,data)==0);free(data);
 }
 return 0;
}
static int reject_observer(void*c,BnTransformerBF16Stage s,int l,int p,int nt,BnGPUResidentView v){(void)c;(void)s;(void)l;(void)p;(void)nt;(void)v;return -1;}
static void init(BnModel*m,BnSession*s,void*cos){assert(bn_transformer_bf16_request_init(m,gpu,bn_session_backend(s),5,512,cos,262144)==0);assert(bn_transformer_bf16_position(bn_session_backend(s))==0);}
static void empty(BnSession*s){assert(!bn_backend_session_has_gpu_resources(bn_session_backend(s)));assert(!bn_backend_session_has_execution_resources(bn_session_backend(s)));assert(bn_transformer_bf16_position(bn_session_backend(s))==-1);}
static void test_arithmetic_policy(BnModel *m,BnSession *s,void *cos,const int *prompt) {
 BnBackendSession *b=bn_session_backend(s);bn_session_reset(s,m);reset_counts();
 BnTransformerBF16Policy policy={0},actual={0};policy.arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32_GATED_R2;
 assert(bn_transformer_bf16_policy(b,&actual)==-1);
 assert(bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,NULL)==-1&&alloc_calls==0);
 policy.attention_split_policy=(BnGPUAttentionSplitPolicy)99;
 assert(bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&policy)==-1&&alloc_calls==0);empty(s);
 policy.attention_split_policy=BN_GPU_ATTENTION_SPLIT_BATCH;
 uint32_t split_caps=gpu->resident_attention_split_policies;gpu->resident_attention_split_policies=0;
 assert(bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&policy)==-1&&alloc_calls==0);empty(s);
 BnBackendSession *split_missing=bn_backend_session_create();assert(split_missing);
 assert(bn_transformer_bf16_batch_prepare_with_policy(m,gpu,split_missing,1,&policy)==-1);
 assert(!bn_backend_session_has_gpu_resources(split_missing)&&!bn_backend_session_has_execution_resources(split_missing));
 bn_backend_session_free(split_missing);gpu->resident_attention_split_policies=split_caps;
 policy.attention_split_policy=BN_GPU_ATTENTION_SPLIT_SINGLETON;
 policy.post_norm_reductions[63]=(BnGPURMSNormReduction)99;
 assert(bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&policy)==-1&&alloc_calls==0);empty(s);
 policy.post_norm_reductions[63]=BN_GPU_RMSNORM_REDUCTION_DEFAULT;
 policy.input_norm_reductions[64]=(BnGPURMSNormReduction)99;
 assert(bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&policy)==-1&&alloc_calls==0);empty(s);
 policy.input_norm_reductions[64]=BN_GPU_RMSNORM_REDUCTION_DEFAULT;
 policy.ffn_arithmetic=(BnTransformerBF16FFNArithmetic)99;
 assert(bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&policy)==-1&&alloc_calls==0);empty(s);
 policy.ffn_arithmetic=BN_TRANSFORMER_BF16_FFN_COMPILED;
 uint32_t ffn_caps=gpu->resident_ffn_activation_profiles;
 gpu->resident_ffn_activation_profiles&=~(1u<<BN_GPU_BF16_FFN_SILU_COMPILED);
 assert(bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&policy)==-1&&alloc_calls==0);empty(s);
 BnBackendSession *missing=bn_backend_session_create();assert(missing);
 assert(bn_transformer_bf16_batch_prepare_with_policy(m,gpu,missing,1,&policy)==-1);
 assert(!bn_backend_session_has_gpu_resources(missing)&&!bn_backend_session_has_execution_resources(missing));
 bn_backend_session_free(missing);gpu->resident_ffn_activation_profiles=ffn_caps;
 BnTransformerBF16Policy eager={0};eager.ffn_arithmetic=BN_TRANSFORMER_BF16_FFN_COMPILED;
 assert(bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&eager)==-1&&alloc_calls==0);empty(s);
 policy.input_norm_reductions[0]=BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT;
 policy.input_norm_reductions[64]=BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES;
 policy.post_norm_reductions[62]=BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED;
 policy.post_norm_reductions[63]=BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA;
 policy.post_norm_reductions[0]=BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA;
 uint32_t caps=gpu->resident_rmsnorm_reductions;
 gpu->resident_rmsnorm_reductions&=~(1u<<BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA);
 assert(bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&policy)==-1&&alloc_calls==0);empty(s);
 BnBackendSession *co=bn_backend_session_create();assert(co);
 assert(bn_transformer_bf16_batch_prepare_with_policy(m,gpu,co,1,&policy)==-1);
 assert(!bn_backend_session_has_gpu_resources(co)&&!bn_backend_session_has_execution_resources(co));
 gpu->resident_rmsnorm_reductions=caps;bn_backend_session_free(co);
 for(int reduction=BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT;reduction<=BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES;reduction++){
  gpu->resident_rmsnorm_reductions=caps&~(1u<<reduction);
  assert(bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&policy)==-1&&alloc_calls==0);empty(s);
 }
 gpu->resident_rmsnorm_reductions=caps;
 policy.attention_split_policy=BN_GPU_ATTENTION_SPLIT_BATCH;
 assert(!bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&policy));
 assert(!bn_transformer_bf16_policy(b,&actual));
 assert(actual.ffn_arithmetic==BN_TRANSFORMER_BF16_FFN_COMPILED);
 assert(actual.attention_split_policy==BN_GPU_ATTENTION_SPLIT_BATCH);
 for(int l=0;l<64;l++)assert(actual.post_norm_reductions[l]==(l==0||l==63
     ?BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA:l==62?BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED:BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA));
 assert(actual.input_norm_reductions[0]==BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT);
 assert(actual.input_norm_reductions[64]==BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES);
 for(int l=1;l<64;l++)assert(actual.input_norm_reductions[l]==BN_GPU_RMSNORM_REDUCTION_TILED_5120_FMA);
 policy.ffn_arithmetic=BN_TRANSFORMER_BF16_FFN_DEFAULT;
 policy.input_norm_reductions[0]=policy.input_norm_reductions[64]=BN_GPU_RMSNORM_REDUCTION_DEFAULT;
 policy.post_norm_reductions[0]=BN_GPU_RMSNORM_REDUCTION_DEFAULT;
 policy.post_norm_reductions[63]=BN_GPU_RMSNORM_REDUCTION_DEFAULT;
 policy.post_norm_reductions[62]=BN_GPU_RMSNORM_REDUCTION_DEFAULT;
 size_t bytes=(size_t)m->config.vocab_size*sizeof(float);float *saved=malloc(bytes),*expected=malloc(bytes);assert(saved&&expected);
 float *x=bn_prefill(m,s,prompt,5,0,0);assert(x);memcpy(saved,x,bytes);int next=token(x,m->config.vocab_size);
 assert(!bn_transformer_bf16_session_restart(m,s));BnTransformerBF16Policy restarted;
 assert(!bn_transformer_bf16_policy(b,&restarted));
 assert(restarted.ffn_arithmetic==actual.ffn_arithmetic);
 assert(restarted.attention_split_policy==actual.attention_split_policy);
 assert(!memcmp(actual.post_norm_reductions,restarted.post_norm_reductions,sizeof(actual.post_norm_reductions)));
 assert(!memcmp(actual.input_norm_reductions,restarted.input_norm_reductions,sizeof(actual.input_norm_reductions)));
 x=bn_prefill(m,s,prompt,5,0,0);assert(x&&!memcmp(saved,x,bytes));
 co=bn_backend_session_create();assert(co);
 BnTransformerBF16Policy incompatible=actual;
 incompatible.input_norm_reductions[64]=BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT;
 assert(!bn_transformer_bf16_batch_prepare_with_policy(m,gpu,co,1,&incompatible));
 BnTransformerBF16DecodeRow row={b,next,5,s->state.logits,(size_t)m->config.vocab_size};
 reset_counts();assert(bn_transformer_bf16_batch_decode(m,gpu,co,&row,1)==-1);
 assert(!read_calls&&!write_calls&&!sync_calls&&!alloc_calls&&bn_transformer_bf16_position(b)==5);
 assert(!memcmp(saved,s->state.logits,bytes));BnTransformerBF16BatchInfo info;
 assert(!bn_transformer_bf16_batch_info(co,&info)&&!info.failed&&!info.decode_calls);
 bn_backend_session_free(co);
 co=bn_backend_session_create();assert(co);incompatible=actual;
 incompatible.ffn_arithmetic=BN_TRANSFORMER_BF16_FFN_FUSED_RN;
 assert(!bn_transformer_bf16_batch_prepare_with_policy(m,gpu,co,1,&incompatible));
 reset_counts();assert(bn_transformer_bf16_batch_decode(m,gpu,co,&row,1)==-1);
 assert(!read_calls&&!write_calls&&!sync_calls&&!alloc_calls&&bn_transformer_bf16_position(b)==5);
 assert(!memcmp(saved,s->state.logits,bytes));
 assert(!bn_transformer_bf16_batch_info(co,&info)&&!info.failed&&!info.decode_calls);
 bn_backend_session_free(co);
 co=bn_backend_session_create();assert(co);incompatible=actual;
 incompatible.attention_split_policy=BN_GPU_ATTENTION_SPLIT_SINGLETON;
 assert(!bn_transformer_bf16_batch_prepare_with_policy(m,gpu,co,1,&incompatible));
 reset_counts();assert(bn_transformer_bf16_batch_decode(m,gpu,co,&row,1)==-1);
 assert(!read_calls&&!write_calls&&!sync_calls&&!alloc_calls&&bn_transformer_bf16_position(b)==5);
 assert(!memcmp(saved,s->state.logits,bytes));
 assert(!bn_transformer_bf16_batch_info(co,&info)&&!info.failed&&!info.decode_calls);
 bn_backend_session_free(co);
 BnSession *clone=bn_session_create_backend(m,NULL);assert(clone);
 assert(!bn_transformer_bf16_request_init_with_policy(m,gpu,bn_session_backend(clone),5,512,cos,262144,&actual));
 assert(bn_prefill(m,clone,prompt,5,0,0));x=bn_transformer_forward(m,clone,next,5);assert(x);memcpy(expected,x,bytes);
 co=bn_backend_session_create();assert(co);
 assert(!bn_transformer_bf16_batch_prepare_with_policy(m,gpu,co,1,&actual));
 assert(!bn_transformer_bf16_policy(co,&restarted));
 assert(restarted.ffn_arithmetic==actual.ffn_arithmetic);
 assert(restarted.attention_split_policy==actual.attention_split_policy);
 assert(!memcmp(restarted.post_norm_reductions,actual.post_norm_reductions,sizeof(actual.post_norm_reductions)));
 reset_counts();assert(!bn_transformer_bf16_batch_decode(m,gpu,co,&row,1));
 assert(!alloc_calls&&!cpu_fallbacks&&bn_transformer_bf16_position(b)==6&&!memcmp(expected,s->state.logits,bytes));
 bn_backend_session_free(co);bn_session_free(clone,NULL);free(saved);free(expected);bn_session_reset(s,m);reset_counts();
 BnTransformerBF16Policy base={0};base.arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32;
 assert(!bn_transformer_bf16_request_init_with_policy(m,gpu,b,5,512,cos,262144,&base));
 assert(!bn_transformer_bf16_policy(b,&restarted)&&restarted.ffn_arithmetic==BN_TRANSFORMER_BF16_FFN_FUSED_RN);
 assert(restarted.attention_split_policy==BN_GPU_ATTENTION_SPLIT_SINGLETON);
 bn_session_reset(s,m);reset_counts();
 printf("REQUEST_ARITHMETIC_POLICY_ADMISSION_COPY_RESTART_BATCH PASSED\n");
}

typedef struct {int tokens,outputs,calls;} PackedObserver;
static int observe_packed(void *context,BnTransformerBF16Stage stage,int layer,int position,int nt,BnGPUResidentView view) {
 (void)layer;PackedObserver *p=context;assert(position==-1);p->calls++;
 if(stage==BN_TRANSFORMER_BF16_LOGITS){assert(nt==p->outputs);}
 else {assert(nt==p->tokens&&view.count==(size_t)nt*5120);}
 return 0;
}
static int reject_packed_ffn(void *ctx,void *request,const BnGPUFFNResidentPlan *p) {
 (void)ctx;(void)request;(void)p;return -1;
}
static void test_packed_prefill(BnModel *m,void *cos,const int *prompt,BnTransformerBF16Arithmetic arithmetic) {
 BnTransformerBF16Policy policy={0};policy.arithmetic=arithmetic;
 policy.ffn_arithmetic=BN_TRANSFORMER_BF16_FFN_COMPILED;
 BnBackendSession *co=bn_backend_session_create();assert(co);
 reset_counts();
 assert(bn_transformer_bf16_batch_prepare_packed_with_policy(m,gpu,co,2,0,&policy)==-1);
 assert(bn_transformer_bf16_batch_prepare_packed_with_policy(m,gpu,co,2,6,&policy)==-1);
 int (*gather)(void*,void*,const BnGPUGatherRowsResidentPlan*)=gpu->gather_rows_resident;
 gpu->gather_rows_resident=NULL;
 assert(bn_transformer_bf16_batch_prepare_packed_with_policy(m,gpu,co,2,5,&policy)==-1);
 gpu->gather_rows_resident=gather;
 assert(!alloc_calls&&!bn_backend_session_has_gpu_resources(co)&&!bn_backend_session_has_execution_resources(co));
 if(arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32){
  uint32_t profiles=gpu->resident_gdn_prefill_recurrences;gpu->resident_gdn_prefill_recurrences=0;
  assert(bn_transformer_bf16_batch_prepare_packed_with_policy(m,gpu,co,2,5,&policy)==-1&&!alloc_calls);
  gpu->resident_gdn_prefill_recurrences=profiles;
 }
 assert(!bn_transformer_bf16_batch_prepare_packed_with_policy(m,gpu,co,2,5,&policy));
 BnTransformerBF16BatchInfo info,before;
 assert(!bn_transformer_bf16_batch_info(co,&info)&&info.capacity==2&&info.max_prefill_tokens==5&&!info.prefill_calls);
 BnSession *session[2],*isolated=bn_session_create_backend(m,NULL);assert(isolated);
 assert(!bn_transformer_bf16_request_init_with_policy(m,gpu,bn_session_backend(isolated),5,512,cos,262144,&policy));
 for(int i=0;i<2;i++) {
  session[i]=bn_session_create_backend(m,NULL);assert(session[i]);
  assert(!bn_transformer_bf16_request_init_with_policy(m,gpu,bn_session_backend(session[i]),5,512,cos,262144,&policy));
 }
 size_t vocab=(size_t)m->config.vocab_size,bytes=vocab*sizeof(float);
 float *expected=malloc(bytes),*snapshots=malloc(2*bytes);assert(expected&&snapshots);
 BnTransformerBF16PrefillRow rows[2]={{bn_session_backend(session[0]),prompt,3,0,session[0]->state.logits,vocab},
  {bn_session_backend(session[1]),prompt,2,0,session[1]->state.logits,vocab}};
 for(int i=0;i<2;i++)for(size_t j=0;j<vocab;j++)session[i]->state.logits[j]=7.f+i;
 for(int i=0;i<2;i++)memcpy((char*)snapshots+(size_t)i*bytes,session[i]->state.logits,bytes);
 for(int reject=0;reject<9;reject++) {
  BnTransformerBF16PrefillRow bad[2]={rows[0],rows[1]};int32_t invalid[]={-1,0};
  switch(reject) {
   case 0:bad[1].n_tokens=0;break;case 1:bad[1].position=1;break;
   case 2:bad[1].session=bad[0].session;break;case 3:bad[1].logits=bad[0].logits;break;
   case 4:bad[1].tokens=invalid;break;case 5:bad[1].logits_count--;break;
   case 6:bad[1].logits=NULL;break;case 7:bad[1].n_tokens=3;break;
   default:bad[1].session=co;break;
  }
  reset_counts();assert(bn_transformer_bf16_batch_prefill(m,gpu,co,bad,2,NULL,NULL)==-1);
  assert(!alloc_calls&&!write_calls&&!read_calls&&!sync_calls);
  for(int i=0;i<2;i++){assert(bn_transformer_bf16_position(rows[i].session)==0);assert(!memcmp((char*)snapshots+(size_t)i*bytes,session[i]->state.logits,bytes));}
 }
 BnTransformerBF16DecodeRow too_many[3]={{0}};reset_counts();
 assert(bn_transformer_bf16_batch_decode(m,gpu,co,too_many,3)==-1&&!write_calls&&!read_calls&&!sync_calls);
 float *out=bn_transformer_bf16_forward_session(m,isolated,prompt,5,0,1);assert(out);memcpy(expected,out,bytes);
 rows[0].n_tokens=5;PackedObserver probe={5,1,0};
 assert(!bn_transformer_bf16_batch_prefill(m,gpu,co,rows,1,observe_packed,&probe)&&probe.calls);
 assert(!memcmp(expected,rows[0].logits,bytes)&&bn_transformer_bf16_position(rows[0].session)==5);
 int id=token(expected,m->config.vocab_size);out=bn_transformer_forward(m,isolated,id,5);assert(out);memcpy(expected,out,bytes);
 BnTransformerBF16DecodeRow decode={rows[0].session,id,5,rows[0].logits,vocab};
 assert(!bn_transformer_bf16_batch_decode(m,gpu,co,&decode,1)&&!memcmp(expected,rows[0].logits,bytes));
 assert(!bn_transformer_bf16_session_restart(m,session[0])&&!bn_transformer_bf16_session_restart(m,isolated));
 rows[0].n_tokens=3;rows[0].logits=NULL;rows[0].logits_count=0;
 memcpy(snapshots,session[0]->state.logits,bytes);
 assert(!bn_transformer_bf16_no_logits_session(m,isolated,prompt,3,0,1));
 assert(!bn_transformer_bf16_batch_prefill(m,gpu,co,rows,1,NULL,NULL));
 assert(!memcmp(snapshots,session[0]->state.logits,bytes)&&bn_transformer_bf16_position(rows[0].session)==3);
 rows[0].tokens=prompt+3;rows[0].n_tokens=2;rows[0].position=3;rows[0].logits=session[0]->state.logits;rows[0].logits_count=vocab;
 out=bn_transformer_bf16_forward_session(m,isolated,prompt+3,2,3,1);assert(out);memcpy(expected,out,bytes);
 assert(!bn_transformer_bf16_batch_prefill(m,gpu,co,rows,1,NULL,NULL)&&!memcmp(expected,rows[0].logits,bytes));
 id=token(expected,m->config.vocab_size);out=bn_transformer_forward(m,isolated,id,5);assert(out);memcpy(expected,out,bytes);
 decode.token=id;assert(!bn_transformer_bf16_batch_decode(m,gpu,co,&decode,1)&&!memcmp(expected,rows[0].logits,bytes));
 assert(!bn_transformer_bf16_session_restart(m,session[0]));
 rows[0].tokens=prompt;rows[0].n_tokens=3;rows[0].position=0;rows[0].logits=NULL;rows[0].logits_count=0;
 memcpy(snapshots,session[0]->state.logits,bytes);
 assert(!bn_transformer_bf16_batch_prefill(m,gpu,co,rows,2,NULL,NULL));
 assert(!memcmp(snapshots,session[0]->state.logits,bytes));
 assert(bn_transformer_bf16_position(rows[0].session)==3&&bn_transformer_bf16_position(rows[1].session)==2);
 for(size_t j=0;j<vocab;j++)assert(isfinite(rows[1].logits[j]));
 /* Same packed geometry under sequence permutation must preserve isolation;
  * this is a metamorphic check, not an independent model oracle. */
 for(int i=0;i<2;i++) {
  assert(!bn_transformer_bf16_session_restart(m,session[i]));
  rows[i].logits=session[i]->state.logits;rows[i].logits_count=vocab;
 }
 assert(!bn_transformer_bf16_batch_prefill(m,gpu,co,rows,2,NULL,NULL));
 for(int i=0;i<2;i++) {
  memcpy((char*)snapshots+(size_t)i*bytes,rows[i].logits,bytes);
  assert(!bn_transformer_bf16_session_restart(m,session[i]));
 }
 BnTransformerBF16PrefillRow reversed[2]={rows[1],rows[0]};
 assert(!bn_transformer_bf16_batch_prefill(m,gpu,co,reversed,2,NULL,NULL));
 for(int i=0;i<2;i++)assert(!memcmp((char*)snapshots+(size_t)i*bytes,rows[i].logits,bytes));
 for(int i=0;i<2;i++) {
  rows[i].tokens=prompt+i;rows[i].n_tokens=1;rows[i].position=i?2:3;rows[i].logits=session[i]->state.logits;rows[i].logits_count=vocab;
  memcpy((char*)snapshots+(size_t)i*bytes,rows[i].logits,bytes);
 }
 assert(!bn_transformer_bf16_batch_info(co,&before));
 int (*ffn)(void*,void*,const BnGPUFFNResidentPlan*)=gpu->dense_ffn_bf16_resident;
 gpu->dense_ffn_bf16_resident=reject_packed_ffn;
 int status=bn_transformer_bf16_batch_prefill(m,gpu,co,rows,2,NULL,NULL);
 gpu->dense_ffn_bf16_resident=ffn;assert(status==-1);
 assert(!bn_transformer_bf16_batch_info(co,&info)&&info.failed&&info.prefill_calls==before.prefill_calls);
 for(int i=0;i<2;i++){assert(bn_transformer_bf16_position(rows[i].session)==-1);assert(!memcmp((char*)snapshots+(size_t)i*bytes,rows[i].logits,bytes));}
 reset_counts();assert(bn_transformer_bf16_batch_prefill(m,gpu,co,rows,2,NULL,NULL)==-1&&!write_calls&&!read_calls&&!sync_calls);
 for(int i=0;i<2;i++)bn_session_free(session[i],NULL);
 bn_session_free(isolated,NULL);bn_backend_session_free(co);free(expected);free(snapshots);reset_counts();
 puts("PACKED_PREFILL_ADMISSION_SINGLE_CHUNK_RESUME_DECODE_MIXED_FAILURE PASSED");
}

static int tail_decode_calls,tail_prefill_calls;
static int (*tail_original_decode)(void*,void*,const BnGPUAttentionDecodeGroupPlan*);
static int (*tail_original_prefill)(void*,void*,const BnGPUAttentionPrefillGroupPlan*);
static int count_tail_decode(void *ctx,void *request,const BnGPUAttentionDecodeGroupPlan *p) {
 tail_decode_calls++;assert(p->split_policy==BN_GPU_ATTENTION_SPLIT_BATCH);
 for(int i=0;i<p->n_sequences;i++)assert(p->sequences[i].plan.n_tokens==1);
 return tail_original_decode(ctx,request,p);
}
static int count_tail_prefill(void *ctx,void *request,const BnGPUAttentionPrefillGroupPlan *p) {
 tail_prefill_calls++;return tail_original_prefill(ctx,request,p);
}
static void test_batch_policy_prefill_tails(BnModel *m,void *cos,const int *prompt) {
 BnTransformerBF16Policy policy={0};policy.arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32;
 policy.ffn_arithmetic=BN_TRANSFORMER_BF16_FFN_COMPILED;policy.attention_split_policy=BN_GPU_ATTENTION_SPLIT_BATCH;
 BnBackendSession *co=bn_backend_session_create();assert(co);
 assert(!bn_transformer_bf16_batch_prepare_packed_with_policy(m,gpu,co,2,4,&policy));
 BnSession *sessions[2];BnTransformerBF16PrefillRow rows[2];
 size_t vocab=(size_t)m->config.vocab_size,bytes=vocab*sizeof(float);float *saved=malloc(2*bytes);assert(saved);
 for(int i=0;i<2;i++){
  sessions[i]=bn_session_create_backend(m,NULL);assert(sessions[i]);
  assert(!bn_transformer_bf16_request_init_with_policy(m,gpu,bn_session_backend(sessions[i]),2,512,cos,262144,&policy));
  rows[i]=(BnTransformerBF16PrefillRow){bn_session_backend(sessions[i]),prompt,2,0,sessions[i]->state.logits,vocab};
 }
 tail_original_decode=gpu->attention_decode_group_bf16_resident;tail_original_prefill=gpu->attention_prefill_group_bf16_resident;
 gpu->attention_decode_group_bf16_resident=count_tail_decode;gpu->attention_prefill_group_bf16_resident=count_tail_prefill;
 tail_decode_calls=tail_prefill_calls=0;reset_counts();
 assert(!bn_transformer_bf16_batch_prefill(m,gpu,co,rows,2,NULL,NULL));
 assert(tail_prefill_calls==16&&!tail_decode_calls&&!alloc_calls&&!cpu_fallbacks);
 for(int i=0;i<2;i++){
  memcpy(saved+(size_t)i*vocab,sessions[i]->state.logits,bytes);
  rows[i].tokens=prompt+2;rows[i].n_tokens=1;rows[i].position=2;rows[i].logits=NULL;rows[i].logits_count=0;
 }
 tail_decode_calls=tail_prefill_calls=0;reset_counts();
 assert(!bn_transformer_bf16_batch_prefill(m,gpu,co,rows,2,NULL,NULL));
 assert(tail_decode_calls==16&&!tail_prefill_calls&&!alloc_calls&&!read_calls&&!cpu_fallbacks);
 for(int i=0;i<2;i++)assert(bn_transformer_bf16_position(rows[i].session)==3&&!memcmp(saved+(size_t)i*vocab,sessions[i]->state.logits,bytes));
 BnTransformerBF16BatchInfo info;assert(!bn_transformer_bf16_batch_info(co,&info)&&info.prefill_calls==2&&!info.decode_calls&&!info.failed);
 /* A mixed one-/two-token cohort must retain the prefill attention path. */
 for(int i=0;i<2;i++){rows[i].position=3;rows[i].tokens=prompt;rows[i].n_tokens=1+i;rows[i].logits=sessions[i]->state.logits;rows[i].logits_count=vocab;}
 tail_decode_calls=tail_prefill_calls=0;reset_counts();
 assert(!bn_transformer_bf16_batch_prefill(m,gpu,co,rows,2,NULL,NULL));
 assert(tail_prefill_calls==16&&!tail_decode_calls&&!alloc_calls&&!cpu_fallbacks);
 for(int i=0;i<2;i++){assert(bn_transformer_bf16_position(rows[i].session)==4+i);bn_session_free(sessions[i],NULL);}
 gpu->attention_decode_group_bf16_resident=tail_original_decode;gpu->attention_prefill_group_bf16_resident=tail_original_prefill;
 bn_backend_session_free(co);free(saved);reset_counts();puts("BATCH_POLICY_PREFILL_TAILS PASSED");
}

static void test_mixed_steps(BnModel *m,void *cos,const int *prompt) {
 BnTransformerBF16Policy policy={0};policy.arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32;
 policy.ffn_arithmetic=BN_TRANSFORMER_BF16_FFN_COMPILED;policy.attention_split_policy=BN_GPU_ATTENTION_SPLIT_BATCH;
 BnBackendSession *co=bn_backend_session_create();assert(co);
 assert(!bn_transformer_bf16_batch_prepare_packed_with_policy(m,gpu,co,2,4,&policy));
 BnSession *session[2];BnTransformerBF16StepRow rows[2];size_t vocab=(size_t)m->config.vocab_size,bytes=vocab*sizeof(float);
 float *saved=malloc(2*bytes);assert(saved);
 for(int i=0;i<2;i++){
  session[i]=bn_session_create_backend(m,NULL);assert(session[i]);
  assert(!bn_transformer_bf16_request_init_with_policy(m,gpu,bn_session_backend(session[i]),2,512,cos,262144,&policy));
  rows[i]=(BnTransformerBF16StepRow){{bn_session_backend(session[i]),prompt,2,0,session[i]->state.logits,vocab},BN_TRANSFORMER_BF16_STEP_PREFILL};
 }
 reset_counts();assert(!bn_transformer_bf16_batch_step(m,gpu,co,rows,2,NULL,NULL)&&!alloc_calls&&!cpu_fallbacks);
 BnTransformerBF16BatchInfo info;assert(!bn_transformer_bf16_batch_info(co,&info)&&info.prefill_calls==1&&!info.decode_calls&&!info.mixed_calls);
 for(int i=0;i<2;i++){memcpy(saved+(size_t)i*vocab,session[i]->state.logits,bytes);rows[i].row.position=2;rows[i].row.tokens=prompt+2;}
 rows[0].kind=BN_TRANSFORMER_BF16_STEP_DECODE;rows[0].row.n_tokens=1;
 for(int scenario=0;scenario<6;scenario++){
  BnTransformerBF16StepRow bad[2];memcpy(bad,rows,sizeof(bad));
  if(scenario==0)bad[1].kind=(BnTransformerBF16StepKind)99;
  if(scenario==1)bad[0].row.n_tokens=2;
  if(scenario==2){bad[0].row.logits=NULL;bad[0].row.logits_count=0;}
  if(scenario==3)bad[1].row.session=bad[0].row.session;
  if(scenario==4)bad[1].row.position=3;
  uint32_t profiles=gpu->resident_gdn_decode_profiles;
  if(scenario==5)gpu->resident_gdn_decode_profiles&=~(1u<<BN_GPU_GDN_DECODE_RSQRT);
  reset_counts();assert(bn_transformer_bf16_batch_step(m,gpu,co,bad,2,NULL,NULL)==-1);
  gpu->resident_gdn_decode_profiles=profiles;
  assert(!alloc_calls&&!write_calls&&!read_calls&&!sync_calls);
  for(int i=0;i<2;i++)assert(bn_transformer_bf16_position(rows[i].row.session)==2&&!memcmp(saved+(size_t)i*vocab,session[i]->state.logits,bytes));
  assert(!bn_transformer_bf16_batch_info(co,&info)&&!info.failed&&!info.mixed_calls&&info.prefill_calls==1);
 }
 PackedObserver probe={3,2,0};reset_counts();
 assert(!bn_transformer_bf16_batch_step(m,gpu,co,rows,2,observe_packed,&probe)&&probe.calls&&!alloc_calls&&!cpu_fallbacks);
 assert(bn_transformer_bf16_position(rows[0].row.session)==3&&bn_transformer_bf16_position(rows[1].row.session)==4);
 assert(!bn_transformer_bf16_batch_info(co,&info)&&info.mixed_calls==1&&info.prefill_calls==1&&!info.decode_calls&&!info.failed);
 for(int i=0;i<2;i++){rows[i].row.position=3+i;rows[i].row.n_tokens=1;rows[i].kind=BN_TRANSFORMER_BF16_STEP_DECODE;}
 reset_counts();assert(!bn_transformer_bf16_batch_step(m,gpu,co,rows,2,NULL,NULL)&&!alloc_calls&&!cpu_fallbacks);
 assert(!bn_transformer_bf16_batch_info(co,&info)&&info.decode_calls==1&&info.mixed_calls==1&&info.prefill_calls==1);
 for(int i=0;i<2;i++){memcpy(saved+(size_t)i*vocab,session[i]->state.logits,bytes);rows[i].row.position=4+i;}
 rows[1].kind=BN_TRANSFORMER_BF16_STEP_PREFILL;rows[1].row.n_tokens=2;rows[1].row.tokens=prompt;
 int (*ffn)(void*,void*,const BnGPUFFNResidentPlan*)=gpu->dense_ffn_bf16_resident;
 gpu->dense_ffn_bf16_resident=reject_packed_ffn;reset_counts();
 assert(bn_transformer_bf16_batch_step(m,gpu,co,rows,2,NULL,NULL)==-1);gpu->dense_ffn_bf16_resident=ffn;
 assert(!bn_transformer_bf16_batch_info(co,&info)&&info.failed&&info.mixed_calls==1);
 for(int i=0;i<2;i++)assert(bn_transformer_bf16_position(rows[i].row.session)==-1&&!memcmp(saved+(size_t)i*vocab,session[i]->state.logits,bytes));
 reset_counts();assert(bn_transformer_bf16_batch_step(m,gpu,co,rows,2,NULL,NULL)==-1&&!alloc_calls&&!write_calls&&!read_calls&&!sync_calls);
 for(int i=0;i<2;i++)bn_session_free(session[i],NULL);
 bn_backend_session_free(co);free(saved);reset_counts();puts("MIXED_STEP_VALIDATION_COMMIT_POISON PASSED");
}

static int selected_prefill_count,selected_prefill_calls;
static BnGPUGDNPrefillRecurrence selected_recurrence;
static int (*original_selected_group)(void*,void*,const BnGPUGDNCPPrefillGroupPlan*);
static int check_selected_group(void *ctx,void *request,const BnGPUGDNCPPrefillGroupPlan *p) {
 assert(p->n_sequences==selected_prefill_count);
 for(int i=0;i<p->n_sequences;i++)assert(p->sequences[i].plan.recurrence==selected_recurrence);
 selected_prefill_calls++;return original_selected_group(ctx,request,p);
}
static void test_cohort_recurrence_policy(BnModel *m,void *cos,const int *prompt) {
 BnTransformerBF16Policy policy={0},actual={0};
 policy.arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32;
 policy.gdn_prefill_selection=BN_TRANSFORMER_BF16_GDN_PREFILL_SINGLE_CP;
 BnSession *sessions[2];BnBackendSession *co=bn_backend_session_create();assert(co);
 for(int i=0;i<2;i++){sessions[i]=bn_session_create_backend(m,NULL);assert(sessions[i]);}
 BnBackendSession *a=bn_session_backend(sessions[0]);
 uint32_t caps=gpu->resident_gdn_prefill_recurrences;
 for(int scenario=0;scenario<4;scenario++) {
  BnTransformerBF16Policy bad=policy;
  if(scenario==0)bad.gdn_prefill_selection=(BnTransformerBF16GDNPrefillSelection)99;
  if(scenario==1)bad.arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32;
  if(scenario>=2)gpu->resident_gdn_prefill_recurrences=caps&~(1u<<(scenario-2));
  reset_counts();assert(bn_transformer_bf16_request_init_with_policy(m,gpu,a,2,512,cos,262144,&bad)==-1);
  assert(bn_transformer_bf16_batch_prepare_with_policy(m,gpu,co,2,&bad)==-1);
  assert(!alloc_calls&&!write_calls&&!read_calls&&!sync_calls);empty(sessions[0]);
  assert(!bn_backend_session_has_gpu_resources(co)&&!bn_backend_session_has_execution_resources(co));
  gpu->resident_gdn_prefill_recurrences=caps;
 }
 assert(!bn_transformer_bf16_batch_prepare_packed_with_policy(m,gpu,co,2,4,&policy));
 assert(!bn_transformer_bf16_policy(co,&actual)&&actual.gdn_prefill_selection==policy.gdn_prefill_selection);
 size_t vocab=(size_t)m->config.vocab_size;
 BnTransformerBF16StepRow rows[2];
 for(int i=0;i<2;i++) {
  assert(!bn_transformer_bf16_request_init_with_policy(m,gpu,bn_session_backend(sessions[i]),2,512,cos,262144,&policy));
  rows[i]=(BnTransformerBF16StepRow){{bn_session_backend(sessions[i]),prompt,2,0,sessions[i]->state.logits,vocab},BN_TRANSFORMER_BF16_STEP_PREFILL};
 }
 original_selected_group=gpu->gdn_cp_prefill_group_resident;gpu->gdn_cp_prefill_group_resident=check_selected_group;
 selected_prefill_count=2;selected_recurrence=BN_GPU_GDN_PREFILL_NONCP;selected_prefill_calls=0;reset_counts();
 assert(!bn_transformer_bf16_batch_step(m,gpu,co,rows,2,NULL,NULL));
 assert(selected_prefill_calls==48&&!alloc_calls&&!cpu_fallbacks);
 for(int i=0;i<2;i++){rows[i].row.position=2;rows[i].row.tokens=prompt+2;rows[i].row.n_tokens=1;}
 /* Two one-token prefill tails still select NONCP. */
 selected_prefill_calls=0;reset_counts();assert(!bn_transformer_bf16_batch_step(m,gpu,co,rows,2,NULL,NULL));
 assert(selected_prefill_calls==48&&!alloc_calls&&!cpu_fallbacks);
 for(int i=0;i<2;i++){rows[i].row.position=3;rows[i].row.tokens=prompt+3;}
 rows[0].kind=BN_TRANSFORMER_BF16_STEP_DECODE;
 selected_prefill_count=1;selected_recurrence=BN_GPU_GDN_PREFILL_CP;selected_prefill_calls=0;reset_counts();
 assert(!bn_transformer_bf16_batch_step(m,gpu,co,rows,2,NULL,NULL));
 assert(selected_prefill_calls==48&&!alloc_calls&&!cpu_fallbacks);
 gpu->gdn_cp_prefill_group_resident=original_selected_group;
 /* An incompatible coordinator must reject before changing either session. */
 BnBackendSession *other=bn_backend_session_create();assert(other);
 policy.gdn_prefill_selection=BN_TRANSFORMER_BF16_GDN_PREFILL_DEFAULT;
 assert(!bn_transformer_bf16_batch_prepare_packed_with_policy(m,gpu,other,2,4,&policy));
 for(int i=0;i<2;i++){rows[i].row.position=4;rows[i].kind=BN_TRANSFORMER_BF16_STEP_DECODE;}
 reset_counts();assert(bn_transformer_bf16_batch_step(m,gpu,other,rows,2,NULL,NULL)==-1);
 BnTransformerBF16DecodeRow decode={a,prompt[4],4,sessions[0]->state.logits,vocab};
 assert(bn_transformer_bf16_batch_decode(m,gpu,other,&decode,1)==-1);
 assert(!alloc_calls&&!write_calls&&!read_calls&&!sync_calls);
 for(int i=0;i<2;i++)assert(bn_transformer_bf16_position(rows[i].row.session)==4);
 assert(!bn_transformer_bf16_session_restart(m,sessions[0]));
 assert(!bn_transformer_bf16_policy(a,&actual)&&actual.gdn_prefill_selection==BN_TRANSFORMER_BF16_GDN_PREFILL_SINGLE_CP);
 bn_backend_session_free(other);bn_backend_session_free(co);
 for(int i=0;i<2;i++)bn_session_free(sessions[i],NULL);
 reset_counts();puts("COHORT_RECURRENCE_CAPABILITY_COPY_TAIL_ROLE_RESTART_VALIDATION PASSED");
}

static void test_sized_request_pool(BnModel *m,const int *prompt) {
 BnTransformerBF16Policy policy={0};size_t full=512u*65536,small=64u*65536;
 BnTransformerBF16Pool fixed={0},sized={0};BnSession *reference,*a,*b;
 assert(!bn_transformer_bf16_pool_init(&fixed,m,gpu,1,5,512,full,&policy));
 assert(!bn_transformer_bf16_pool_init(&sized,m,gpu,2,5,512,full,&policy));
 assert(!bn_transformer_bf16_pool_acquire(&fixed,&reference));
 reset_counts();pool_kv_elements=64u*1024;pool_kv_allocated=0;
 assert(!bn_transformer_bf16_pool_acquire_capacity(&sized,33,&a));
 assert(!bn_transformer_bf16_pool_acquire_capacity(&sized,33,&b));
 assert(bn_transformer_bf16_capacity(bn_session_backend(a))==64&&bn_transformer_bf16_capacity(bn_session_backend(b))==64);
 assert(pool_kv_allocated==2*small);BnTransformerBF16PoolInfo info;
 assert(!bn_transformer_bf16_pool_info(&sized,&info)&&info.reserved_kv_bytes==2*small&&info.leased_slots==2&&info.kv_bytes_per_slot==full);
 int tokens[64];for(int i=0;i<64;i++)tokens[i]=prompt[i%5];
 size_t bytes=(size_t)m->config.vocab_size*sizeof(float);float *initial=malloc(bytes);assert(initial);
 for(int pos=0;pos<64;) {
  int n=64-pos;if(n>5)n=5;
  assert(!bn_transformer_bf16_forward(m,gpu,bn_session_backend(reference),tokens+pos,n,reference->state.logits,(size_t)m->config.vocab_size,NULL,NULL));
  assert(!bn_transformer_bf16_forward(m,gpu,bn_session_backend(a),tokens+pos,n,a->state.logits,(size_t)m->config.vocab_size,NULL,NULL));
  assert(!memcmp(reference->state.logits,a->state.logits,bytes));
  if(!pos)memcpy(initial,a->state.logits,bytes);
  pos+=n;
 }
 reset_counts();assert(bn_transformer_bf16_forward(m,gpu,bn_session_backend(a),tokens,1,a->state.logits,(size_t)m->config.vocab_size,NULL,NULL)==-1);
 assert(!alloc_calls&&!write_calls&&!read_calls&&!linear_calls&&!sync_calls);
 assert(bn_transformer_bf16_position(bn_session_backend(a))==64);
 assert(!bn_transformer_bf16_pool_release(&sized,a));BnSession *reused;reset_counts();
 assert(!bn_transformer_bf16_pool_acquire_capacity(&sized,33,&reused)&&reused==a&&!alloc_calls);
 assert(!bn_transformer_bf16_forward(m,gpu,bn_session_backend(reused),tokens,5,reused->state.logits,(size_t)m->config.vocab_size,NULL,NULL));
 assert(!memcmp(initial,reused->state.logits,bytes));
 assert(!bn_transformer_bf16_pool_info(&sized,&info)&&info.created==2&&info.reused==1&&info.peak_kv_bytes==2*small&&!info.evicted);
 assert(!bn_transformer_bf16_pool_release(&sized,reused)&&!bn_transformer_bf16_pool_release(&sized,b));
 free(initial);bn_transformer_bf16_pool_free(&sized);bn_transformer_bf16_pool_free(&fixed);
 pool_kv_elements=0;reset_counts();
 puts("Sized KV pool: two64-capacity leases, actual bytes, all prefix logits and boundary rejection PASSED");
}

static void test_growing_request_pool(BnModel *m,const int *prompt) {
 BnTransformerBF16Policy policy={0};BnTransformerBF16Pool pool={0};size_t unit=65536;
 assert(!bn_transformer_bf16_pool_init(&pool,m,gpu,1,5,512,512*unit,&policy));
 BnSession *a,*b;assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,33,&a));
 assert(!bn_transformer_bf16_forward(m,gpu,bn_session_backend(a),prompt,5,a->state.logits,(size_t)m->config.vocab_size,NULL,NULL));
 reset_counts();resize_calls=0;
 assert(bn_transformer_bf16_resize_idle_kv(m,a,256)==-1&&!resize_calls&&!alloc_calls&&!sync_calls);
 assert(bn_transformer_bf16_position(bn_session_backend(a))==5);
 assert(!bn_transformer_bf16_pool_release(&pool,a));
 reset_counts();resize_calls=0;resized_kv_bytes=0;
 assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,129,&b)&&b==a);
 assert(resize_calls==32&&!alloc_calls&&resized_kv_bytes==256*unit);
 assert(bn_transformer_bf16_capacity(bn_session_backend(b))==256);
 BnTransformerBF16PoolInfo info;assert(!bn_transformer_bf16_pool_info(&pool,&info));
 assert(info.created==1&&info.reused==1&&info.resized==1&&!info.evicted&&info.reserved_kv_bytes==256*unit);
 BnTransformerBF16Pool fixed={0};BnSession *reference;
 assert(!bn_transformer_bf16_pool_init(&fixed,m,gpu,1,5,512,512*unit,&policy));
 assert(!bn_transformer_bf16_pool_acquire(&fixed,&reference));
 int tokens[129];for(int i=0;i<129;i++)tokens[i]=prompt[i%5];
 for(int pos=0;pos<129;) {
  int n=129-pos;if(n>5)n=5;
  assert(!bn_transformer_bf16_forward(m,gpu,bn_session_backend(b),tokens+pos,n,b->state.logits,(size_t)m->config.vocab_size,NULL,NULL));
  assert(!bn_transformer_bf16_forward(m,gpu,bn_session_backend(reference),tokens+pos,n,reference->state.logits,(size_t)m->config.vocab_size,NULL,NULL));
  assert(!memcmp(b->state.logits,reference->state.logits,(size_t)m->config.vocab_size*sizeof(float)));pos+=n;
 }
 bn_transformer_bf16_pool_free(&fixed);bn_transformer_bf16_pool_free(&pool);
 assert(!bn_transformer_bf16_pool_init(&pool,m,gpu,1,5,512,512*unit,&policy));
 assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,33,&a));
 assert(!bn_transformer_bf16_pool_release(&pool,a));resize_calls=0;fail_resize=2;
 assert(bn_transformer_bf16_pool_acquire_capacity(&pool,129,&b)==-1&&!b);fail_resize=0;
 assert(!bn_transformer_bf16_pool_info(&pool,&info)&&info.discarded==1&&!info.allocated_slots&&!info.reserved_kv_bytes);
 bn_transformer_bf16_pool_free(&pool);resize_calls=0;resized_kv_bytes=0;reset_counts();
 puts("Idle KV growth:32 replacement buffers, non-KV reuse,26 exact prefix heads and partial-failure disposal PASSED");
}

static void test_request_pool(BnModel *m,const int *prompt) {
 BnTransformerBF16Policy policy={0};size_t kv=512u*1024*2*2*16;
 BnTransformerBF16Pool storage={0},*pool=&storage;assert(!bn_transformer_bf16_pool_init(pool,m,gpu,2,5,512,kv,&policy));
 BnTransformerBF16PoolInfo info;BnSession *first=NULL,*again=NULL;
 reset_counts();pool_kv_elements=512u*1024;pool_kv_allocated=0;
 assert(!bn_transformer_bf16_pool_acquire(pool,&first)&&pool_kv_allocated==kv);
 assert(!bn_transformer_bf16_pool_info(pool,&info)&&info.reserved_kv_bytes==kv&&info.kv_bytes_per_slot==kv);
 int allocations=alloc_calls,writes=write_calls,syncs=sync_calls;
 assert(bn_transformer_bf16_pool_acquire(pool,&again)==1&&!again&&alloc_calls==allocations&&write_calls==writes&&sync_calls==syncs);
 size_t bytes=(size_t)m->config.vocab_size*sizeof(float);float *saved=malloc(bytes);assert(saved);
 assert(!bn_transformer_bf16_forward(m,gpu,bn_session_backend(first),prompt,5,first->state.logits,(size_t)m->config.vocab_size,NULL,NULL));
 memcpy(saved,first->state.logits,bytes);
 assert(!bn_transformer_bf16_pool_release(pool,first));reset_counts();
 assert(!bn_transformer_bf16_pool_acquire(pool,&again)&&again==first&&!alloc_calls&&pool_kv_allocated==kv);
 assert(bn_transformer_bf16_position(bn_session_backend(again))==0);
 assert(!bn_transformer_bf16_forward(m,gpu,bn_session_backend(again),prompt,5,again->state.logits,(size_t)m->config.vocab_size,NULL,NULL));
 assert(!memcmp(saved,again->state.logits,bytes));
 assert(!bn_transformer_bf16_pool_info(pool,&info)&&info.created==1&&info.reused==1&&info.leased_slots==1&&info.reserved_kv_bytes==kv);
 reset_counts();fail_linear=1;
 assert(bn_transformer_bf16_forward(m,gpu,bn_session_backend(again),prompt,1,again->state.logits,(size_t)m->config.vocab_size,NULL,NULL)==-1);
 reset_counts();assert(!bn_transformer_bf16_pool_release(pool,again));
 assert(!bn_transformer_bf16_pool_info(pool,&info)&&!info.reserved_kv_bytes&&!info.allocated_slots&&info.discarded==1);
 assert(!bn_transformer_bf16_pool_acquire(pool,&again)&&pool_kv_allocated==kv*2);
 bn_transformer_bf16_pool_free(pool);free(saved);pool_kv_elements=0;reset_counts();
 puts("Dense KV pool actual byte accounting, full-logit reuse and poison disposal PASSED");
 test_sized_request_pool(m,prompt);
 test_growing_request_pool(m,prompt);
}

static void test_memory_plans(BnModel *m) {
 BnTransformerBF16RequestMemory quote,sentinel;
 BnTransformerBF16Policy policy={0};
 for(int profile=0;profile<3;profile++) {
  policy.arithmetic=profile==0?BN_TRANSFORMER_BF16_ARITHMETIC_EAGER:
      profile==1?BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32:BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32;
  policy.attention_split_policy=profile?BN_GPU_ATTENTION_SPLIT_BATCH:BN_GPU_ATTENTION_SPLIT_SINGLETON;
  for(int cap=64;cap<=128;cap*=2) {
   reset_counts();memset(&quote,0x5a,sizeof(quote));sentinel=quote;
   assert(bn_transformer_bf16_dense_memory_plan(m,gpu,5,513,&policy,&quote)==-1&&!memcmp(&quote,&sentinel,sizeof(quote)));
   assert(!bn_transformer_bf16_dense_memory_plan(m,gpu,5,cap,&policy,&quote));
   assert(!alloc_calls&&!write_calls&&!read_calls&&!sync_calls&&!zero_calls&&!linear_calls);
   BnSession *s=bn_session_create_backend(m,NULL);assert(s);
   assert(!bn_transformer_bf16_request_init_with_policy(m,gpu,bn_session_backend(s),5,cap,NULL,0,&policy));
   size_t scratch=gpu->linear_bf16_workspace_size(gpu->ctx)+gpu->linear_bf16_input_workspace_size(gpu->ctx,5u*17408)+
      gpu->embedding_bf16_workspace_size(gpu->ctx)+gpu->attention_bf16_workspace_size(gpu->ctx,5,24,profile?8192:cap)+
      (profile?gpu->gdn_cp_prefill_bf16_workspace_size(gpu->ctx,5,512):gpu->gdn_prefill_bf16_workspace_size(gpu->ctx,5));
   assert(quote.device_bytes==alloc_payload+scratch&&quote.host_bytes==quote.host_peak_bytes);
   assert(bn_transformer_bf16_dense_kv_bytes(m,cap)==(size_t)cap*65536);
   assert(quote.host_bytes>(size_t)m->config.vocab_size*2+5*4);
   bn_session_free(s,NULL);
  }
 }
 policy.arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32;reset_counts();
 assert(!bn_transformer_bf16_batch_memory_plan(m,gpu,4,5,&policy,&quote));
 assert(!alloc_calls&&!write_calls&&!read_calls&&!sync_calls&&!zero_calls&&!linear_calls);
 BnBackendSession *coordinator=bn_backend_session_create();assert(coordinator);
 assert(!bn_transformer_bf16_batch_prepare_packed_with_policy(m,gpu,coordinator,4,5,&policy));
 assert(quote.device_bytes>alloc_payload&&quote.host_bytes==quote.host_peak_bytes);
 bn_backend_session_free(coordinator);reset_counts();
 puts("DENSE_MEMORY_PLANS allocation-free queries, exact request tensors/workspaces, six profiles/capacities and packed coordinator PASSED");
}

int main(int argc,char**argv){
 if(argc!=4&&argc!=5&&argc!=6){fprintf(stderr,"usage: %s MODEL.gguf COS_SIN.bin|--native DUMP_DIR|--plain [--mkl-vml [--pool-only]]\n",argv[0]);return 2;}
 if(argc==6&&(strcmp(argv[5],"--pool-only")||strcmp(argv[2],"--native")||strcmp(argv[3],"--plain")))return 2;
 BnBackendBF16Math math_profile=BN_BACKEND_BF16_MATH_LIBC;
 if(argc>=5){if(strcmp(argv[4],"--mkl-vml")){fprintf(stderr,"unknown math profile\n");return 2;}math_profile=BN_BACKEND_BF16_MATH_MKL_VML_HA;}
 if(!bn_backend_layout_bf16_math_available(math_profile)){fprintf(stderr,"preparation math profile unavailable\n");return 2;}
 int observed=strcmp(argv[3],"--plain")!=0;dump_directory=argv[3];
 BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModel m={0};assert(bn_model_load_with_cpu_preparation(&m,f,512,0,0,0)==0);
 gpu=bn_gpu_cuda_create();assert(gpu);
 void *(*build_rotary)(void*,const BnGPURotaryCachePlan*)=gpu->buffer_create_rotary_bf16;
 gpu->buffer_create_rotary_bf16=reject_rotary;
 assert(bn_backend_layout_prepare_bf16_with_math(bn_model_backend(&m),gpu,&m.config,&m.weights,5,math_profile)==-1);
 assert(!bn_backend_layout_bf16(bn_model_backend(&m))&&!bn_model_gpu(&m));
 gpu->buffer_create_rotary_bf16=build_rotary;
 assert(bn_backend_layout_prepare_bf16_with_math(bn_model_backend(&m),gpu,&m.config,&m.weights,5,math_profile)==0);
 void*cos=NULL;
 if(strcmp(argv[2],"--native")!=0){
  size_t bytes=262144u*64*2;void*data=malloc(bytes);FILE*cache=fopen(argv[2],"rb");assert(cache&&data&&fread(data,1,bytes,cache)==bytes);fclose(cache);cos=bn_gpu_backend_create_quant_only_buffer(gpu,data,bytes,BN_GGUF_TENSOR_BF16,262144,64);free(data);assert(cos);
 }
 const BnBackendBF16Model *layout=bn_backend_layout_bf16(bn_model_backend(&m));
 assert(layout&&layout->rotary_cache&&layout->rotary_capacity==512);
 assert(layout->math_profile==math_profile&&layout->math_library[0]);
 printf("PREPARATION_MATH %d %s\n",math_profile,layout->math_library);

 original_zero=gpu->resident_zero;gpu->resident_zero=count_zero;
 original_sync=gpu->resident_synchronize;gpu->resident_synchronize=count_sync;
 original_reserve=gpu->attention_bf16_reserve;gpu->attention_bf16_reserve=count_reserve;
 original_alloc=gpu->resident_alloc;original_write=gpu->resident_write;original_read=gpu->resident_read;original_linear=gpu->linear_bf16_resident;
 gpu->resident_alloc=count_alloc;gpu->resident_write=count_write;gpu->resident_read=count_read;gpu->linear_bf16_resident=count_linear;gpu->prepare_cpu_operations=reject_cpu;
 original_resize=gpu->resident_resize_discard;assert(original_resize);gpu->resident_resize_discard=count_resize;
 test_memory_plans(&m);
 if(argc==6) {
  const int pool_prompt[]={760,6511,314,9338,369};test_request_pool(&m,pool_prompt);assert(!cpu_fallbacks);
  bn_model_free(&m);bn_gguf_free(f);bn_gpu_cuda_destroy(gpu);puts("POOL_RUNTIME PASSED");return 0;
 }
 BnSession*s=bn_session_create_backend(&m,NULL),*other=bn_session_create_backend(&m,NULL);assert(s&&other);BnBackendSession*b=bn_session_backend(s),*b2=bn_session_backend(other);
 const int prompt[]={760,6511,314,9338,369};const int expected[]={11751,13,198,760,6511,314,9564,369};float*logits=malloc((size_t)m.config.vocab_size*sizeof(float));assert(logits);
 assert(bn_transformer_forward(&m,s,760,0)==NULL&&cpu_fallbacks==0);
 assert(bn_transformer_bf16_request_init(&m,gpu,b,6,512,cos,262144)==-1);
 assert(bn_transformer_bf16_request_init(&m,gpu,b,5,512,layout->rotary_cache,128)==-1);assert(alloc_calls==0);
 int foreign_owner=0;void*foreign=malloc(1);assert(foreign);
 assert(bn_backend_session_attach_gpu_resources(b,&foreign_owner,foreign,release_foreign)==0);
 assert(bn_transformer_bf16_request_init(&m,gpu,b,5,512,cos,262144)==-1);
 assert(bn_backend_session_gpu_resources(b,&foreign_owner)==foreign&&foreign_releases==0);
 bn_backend_session_release_gpu_resources(b);assert(foreign_releases==1);
 foreign=malloc(1);assert(foreign);
 assert(bn_backend_session_attach_execution_resources(b,&foreign_owner,foreign,release_foreign)==0);
 assert(bn_transformer_bf16_request_init(&m,gpu,b,5,512,cos,262144)==-1);
 assert(bn_backend_session_execution_resources(b,&foreign_owner)==foreign&&foreign_releases==1);
 bn_backend_session_release_gpu_resources(b);assert(foreign_releases==2);
 init(&m,s,cos);int allocation_count=alloc_calls;assert(allocation_count>224);assert(zero_calls==96&&write_calls==0);
 assert(bn_transformer_bf16_request_init(&m,gpu,b,5,512,cos,262144)==-1);assert(bn_transformer_bf16_position(b)==0);
 void*request=bn_gpu_backend_resident_request(gpu,b);int id=0;
 for(int phase=0;phase<8;phase++){
  int pos=phase?phase+4:0,nt=phase?1:5;
  if(observed){int32_t ids[5];if(phase)ids[0]=id;else for(int i=0;i<5;i++)ids[i]=prompt[i];assert(bn_transformer_bf16_forward(&m,gpu,b,ids,nt,logits,(size_t)m.config.vocab_size,observe,request)==0);}
  else{float*x=phase?bn_transformer_forward(&m,s,id,pos):bn_prefill(&m,s,prompt,5,0,0);assert(x);memcpy(logits,x,(size_t)m.config.vocab_size*4);}
  id=token(logits,m.config.vocab_size);assert(id==expected[phase]);assert(bn_transformer_bf16_position(b)==pos+nt);printf("TOKEN phase%d pos%d id%d\n",phase,pos,id);fflush(stdout);
 }
 /* Session isolation and public forward/prefill dispatch with ordinary logits. */
 reset_counts();init(&m,other,cos);float*x=bn_transformer_prefill(&m,other,prompt,5,0);assert(x&&token(x,m.config.vocab_size)==11751);assert(bn_transformer_bf16_position(b2)==5&&bn_transformer_bf16_position(b)==12);
 int before=linear_calls;assert(bn_transformer_forward(&m,other,13,7)==NULL);assert(linear_calls==before&&bn_transformer_bf16_position(b2)==5);
 int invalid=-1;assert(bn_transformer_prefill(&m,other,&invalid,1,5)==NULL);assert(linear_calls==before&&bn_transformer_bf16_position(b2)==5);
 int out=-1;assert(bn_transformer_forward_argmax(&m,other,13,5,NULL,0,1,&out)==-1);assert(bn_transformer_prefill_all(&m,other,prompt,5,5,logits)==-1);assert(linear_calls==before&&bn_transformer_bf16_position(b2)==5);
 x=bn_transformer_forward(&m,other,11751,5);assert(x&&token(x,m.config.vocab_size)==13);
 bn_session_reset(s,&m);empty(s);assert(bn_transformer_bf16_position(b2)==6);reset_counts();init(&m,s,cos);x=bn_transformer_prefill(&m,s,prompt,5,0);assert(x&&token(x,m.config.vocab_size)==11751);
 x=bn_transformer_forward(&m,other,13,6);assert(x&&token(x,m.config.vocab_size)==198&&bn_transformer_bf16_position(b2)==7);
 /* Restart keeps storage but discards both recurrent parity and old KV prefix. */
 size_t logit_bytes=(size_t)m.config.vocab_size*sizeof(float);
 float*reference=malloc(logit_bytes*8);assert(reference);
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);
 request=bn_gpu_backend_resident_request(gpu,b);int replay_id=0;
 for(int phase=0;phase<8;phase++){
  x=phase?bn_transformer_forward(&m,s,replay_id,phase+4):bn_prefill(&m,s,prompt,5,0,0);
  assert(x);memcpy(reference+(size_t)phase*m.config.vocab_size,x,logit_bytes);
  replay_id=token(x,m.config.vocab_size);assert(replay_id==expected[phase]);
 }
 BnModel wrong_model=m;BnSession wrong_session=*s;wrong_session.backend_only=0;
 reset_counts();assert(bn_transformer_bf16_session_restart(NULL,s)==-1);
 assert(bn_transformer_bf16_session_restart(&m,NULL)==-1);
 assert(bn_transformer_bf16_session_restart(&wrong_model,s)==-1);
 assert(bn_transformer_bf16_session_restart(&m,&wrong_session)==-1);
 assert(zero_calls==0&&sync_calls==0&&bn_transformer_bf16_position(b)==12);
 for(int replay=0;replay<3;replay++){
  if(replay==1)assert(bn_transformer_forward(&m,s,replay_id,12));
  s->pos=123;s->state.token_history[0]=760;s->state.token_history[511]=369;
  reset_counts();assert(bn_transformer_bf16_session_restart(&m,s)==0);
  assert(alloc_calls==0&&write_calls==0&&zero_calls==96&&sync_calls==1);
  assert(bn_gpu_backend_resident_request(gpu,b)==request);
  assert(s->pos==0&&bn_transformer_bf16_position(b)==0&&bn_transformer_bf16_position(b2)==7);
  for(int i=0;i<m.config.seq_len;i++)assert(s->state.token_history[i]==0);
  replay_id=0;
  for(int phase=0;phase<8;phase++){
   x=phase?bn_transformer_forward(&m,s,replay_id,phase+4):bn_prefill(&m,s,prompt,5,0,0);
   assert(x&&memcmp(x,reference+(size_t)phase*m.config.vocab_size,logit_bytes)==0);
   replay_id=token(x,m.config.vocab_size);assert(replay_id==expected[phase]);
  }
 }
 free(reference);puts("REQUEST_RESTART_EXACT PASSED");
 /* Generation wrappers must not upload legacy CPU mirrors into typed state. */
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);
 int invalid_prompt[6]={760,6511,314,9338,369,-1};
 assert(bn_prefill(&m,s,invalid_prompt,6,0,0)==NULL&&linear_calls==0&&bn_transformer_bf16_position(b)==0);
 int over_capacity[513]={0};assert(bn_prefill(&m,s,over_capacity,513,0,0)==NULL&&linear_calls==0);
 assert(bn_prefill_no_logits(&m,s,NULL,0,0,0)==0&&bn_transformer_bf16_position(b)==0);
 BnTokenizer tokenizer;BnSampler sampler;assert(bn_tokenizer_init(&tokenizer,f)==0);
 assert(bn_sampler_init(&sampler,m.config.vocab_size,0.0f,.9f,1)==0);
 x=bn_prefill(&m,s,prompt,5,0,0);assert(x&&token(x,m.config.vocab_size)==11751);
 GeneratedProbe generated={expected,0};int position=5;
 assert(bn_generate(&m,s,&tokenizer,&sampler,8,&position,collect_generated,&generated,NULL,NULL)==8);
 assert(generated.count==8&&position==12&&bn_transformer_bf16_position(b)==12&&cpu_fallbacks==0);
 /* No-logits prefill preserves caller logits and yields the same next state. */
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);
 for(int i=0;i<m.config.vocab_size;i++)s->state.logits[i]=-123.0f;
 assert(bn_prefill_no_logits(&m,s,prompt,5,0,0)==0&&bn_transformer_bf16_position(b)==5&&sync_calls==1);
 for(int i=0;i<m.config.vocab_size;i++)assert(s->state.logits[i]==-123.0f);
 x=bn_transformer_forward(&m,s,11751,5);assert(x&&token(x,m.config.vocab_size)==13);
 /* Same chunk schedule: generation wrapper versus direct transformer calls. */
 int six[6]={760,6511,314,9338,369,11751};
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);
 x=bn_prefill(&m,s,six,6,0,0);assert(x&&bn_transformer_bf16_position(b)==6);
 memcpy(logits,x,(size_t)m.config.vocab_size*4);
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);
 assert(bn_transformer_prefill_no_logits(&m,s,prompt,5,0)==0);
 x=bn_transformer_prefill(&m,s,six+5,1,5);assert(x&&memcmp(x,logits,(size_t)m.config.vocab_size*4)==0);
 /* An explicit resumed prompt has the same three call boundaries. */
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);
 assert(bn_prefill(&m,s,prompt,5,0,0));
 x=bn_prefill(&m,s,six+5,1,5,0);assert(x&&memcmp(x,logits,(size_t)m.config.vocab_size*4)==0);
 /* Explicit tokenwise policy is equivalent to direct tokenwise execution. */
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);x=bn_prefill(&m,s,prompt,5,0,1);assert(x);memcpy(logits,x,(size_t)m.config.vocab_size*4);
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);for(int i=0;i<5;i++)x=bn_transformer_forward(&m,s,prompt[i],i);assert(x&&memcmp(x,logits,(size_t)m.config.vocab_size*4)==0);
 /* Prefix success followed by an upload failure must not permit reuse. */
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);reset_counts();fail_write=2;
 assert(bn_prefill(&m,s,six,6,0,0)==NULL&&bn_transformer_bf16_position(b)==-1&&cpu_fallbacks==0);
 /* Failing the no-logits fence invalidates the request without publication. */
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);reset_counts();fail_sync=1;
 assert(bn_prefill_no_logits(&m,s,prompt,5,0,0)==-1&&bn_transformer_bf16_position(b)==-1);
 /* External generation position commits only when a forward step succeeds. */
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);x=bn_prefill(&m,s,prompt,5,0,0);assert(x);
 reset_counts();fail_linear=1;position=5;
 assert(bn_generate(&m,s,&tokenizer,&sampler,2,&position,NULL,NULL,NULL,NULL)==-2&&position==5&&bn_transformer_bf16_position(b)==-1);
 bn_sampler_free(&sampler);bn_tokenizer_free(&tokenizer);
 puts("GENERATION_RUNTIME PASSED");
 /* Partial allocation and queued device-zero initialization failures. */
 for(int i=0;i<4;i++){int points[]={1,3,200,allocation_count};bn_session_reset(s,&m);reset_counts();fail_alloc=points[i];assert(bn_transformer_bf16_request_init(&m,gpu,b,5,512,cos,262144)==-1);assert(alloc_calls==points[i]);empty(s);printf("ALLOC_ROLLBACK %d\n",points[i]);}
 bn_session_reset(s,&m);reset_counts();fail_reserve=1;
 assert(bn_transformer_bf16_request_init(&m,gpu,b,5,512,cos,262144)==-1);empty(s);assert(alloc_calls==0);fail_reserve=0;
 for(int i=0;i<3;i++){int points[]={1,2,96};bn_session_reset(s,&m);reset_counts();fail_zero=points[i];assert(bn_transformer_bf16_request_init(&m,gpu,b,5,512,cos,262144)==-1);assert(zero_calls==points[i]);empty(s);printf("ZERO_ROLLBACK %d\n",points[i]);}
 /* A partial restart cannot expose the previous request or recover by retry. */
 for(int failure=0;failure<4;failure++){
  bn_session_reset(s,&m);reset_counts();init(&m,s,cos);
  assert(bn_prefill(&m,s,prompt,5,0,0));s->pos=5;s->state.token_history[0]=760;
  memcpy(logits,s->state.logits,(size_t)m.config.vocab_size*4);
  reset_counts();if(failure<3){int points[]={1,2,96};fail_zero=points[failure];}else fail_sync=1;
  assert(bn_transformer_bf16_session_restart(&m,s)==-1);
  assert(alloc_calls==0&&write_calls==0&&bn_transformer_bf16_position(b)==-1);
  assert(s->pos==5&&s->state.token_history[0]==760);
  assert(memcmp(logits,s->state.logits,(size_t)m.config.vocab_size*4)==0);
  int clears=zero_calls;assert(bn_transformer_bf16_session_restart(&m,s)==-1&&zero_calls==clears);
  assert(bn_transformer_forward(&m,s,11751,5)==NULL);
  printf("RESTART_INVALIDATED %d\n",failure);
 }
 /* A failure after queued device work invalidates state until reset. */
 for(int failure=0;failure<4;failure++){
  bn_session_reset(s,&m);reset_counts();init(&m,s,cos);reset_counts();
  for(int i=0;i<m.config.vocab_size;i++)logits[i]=-123.0f;
  if(failure==0)fail_write=1;
  if(failure==1)fail_linear=5;
  if(failure==2)fail_read=1;
  int32_t ids[5];for(int i=0;i<5;i++)ids[i]=prompt[i];
  assert(bn_transformer_bf16_forward(&m,gpu,b,ids,5,logits,(size_t)m.config.vocab_size,failure==3?reject_observer:NULL,NULL)==-1);
  assert(bn_transformer_bf16_position(b)==-1);for(int i=0;i<m.config.vocab_size;i++)assert(logits[i]==-123.0f);
  before=linear_calls;assert(bn_transformer_forward(&m,s,760,0)==NULL);assert(bn_transformer_prefill(&m,s,prompt,5,0)==NULL);assert(linear_calls==before&&cpu_fallbacks==0);
  printf("EXECUTION_INVALIDATED %d\n",failure);
 }
 bn_session_reset(s,&m);reset_counts();init(&m,s,cos);x=bn_transformer_prefill(&m,s,prompt,5,0);assert(x&&token(x,m.config.vocab_size)==11751);assert(cpu_fallbacks==0);
 /* Arithmetic is request-local, immutable across restart, and explicitly admitted. */
 bn_session_reset(s,&m);reset_counts();assert(bn_transformer_bf16_arithmetic(b)==-1);
 assert(bn_transformer_bf16_request_init_with_arithmetic(&m,gpu,b,5,512,cos,262144,(BnTransformerBF16Arithmetic)6)==-1&&alloc_calls==0);
 int (*add3)(void*,void*,const BnGPUAdd3RMSNormResidentPlan*)=gpu->add3_rmsnorm_bf16_resident;
 gpu->add3_rmsnorm_bf16_resident=NULL;
 assert(bn_transformer_bf16_request_init_with_arithmetic(&m,gpu,b,5,512,cos,262144,BN_TRANSFORMER_BF16_ARITHMETIC_FP32_INTERMEDIATES)==-1&&alloc_calls==0);
 gpu->add3_rmsnorm_bf16_resident=add3;empty(s);
 uint32_t reductions=gpu->resident_rmsnorm_reductions;
 gpu->resident_rmsnorm_reductions=0;
 assert(bn_transformer_bf16_request_init_with_arithmetic(&m,gpu,b,5,512,cos,262144,BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32)==-1&&alloc_calls==0);
 gpu->resident_rmsnorm_reductions=reductions;empty(s);
 int (*cp)(void*,void*,const BnGPUGDNCPPrefillResidentPlan*)=gpu->gdn_cp_prefill_bf16_resident;
 gpu->gdn_cp_prefill_bf16_resident=NULL;
 assert(bn_transformer_bf16_request_init_with_arithmetic(&m,gpu,b,5,512,cos,262144,BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32)==-1&&alloc_calls==0);
 gpu->gdn_cp_prefill_bf16_resident=cp;empty(s);
 uint32_t gate_profiles=gpu->resident_attention_gate_profiles;
 gpu->resident_attention_gate_profiles=0;
 assert(bn_transformer_bf16_request_init_with_arithmetic(&m,gpu,b,5,512,cos,262144,BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32)==-1&&alloc_calls==0);
 gpu->resident_attention_gate_profiles=gate_profiles;empty(s);
 uint32_t norm_profiles=gpu->resident_gated_norm_profiles;
 gpu->resident_gated_norm_profiles=0;
 assert(bn_transformer_bf16_request_init_with_arithmetic(&m,gpu,b,5,512,cos,262144,BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32_GATED_R2)==-1&&alloc_calls==0);
 gpu->resident_gated_norm_profiles=norm_profiles;empty(s);
 uint32_t recurrence_profiles=gpu->resident_gdn_prefill_recurrences;
 gpu->resident_gdn_prefill_recurrences=0;
 assert(bn_transformer_bf16_request_init_with_arithmetic(&m,gpu,b,5,512,cos,262144,BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32)==-1&&alloc_calls==0);
 gpu->resident_gdn_prefill_recurrences=recurrence_profiles;empty(s);
 test_arithmetic_policy(&m,s,cos,prompt);
 test_batch_policy_prefill_tails(&m,cos,prompt);
 test_mixed_steps(&m,cos,prompt);
 test_cohort_recurrence_policy(&m,cos,prompt);
 test_request_pool(&m,prompt);
 test_packed_prefill(&m,cos,prompt,BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32);
 test_packed_prefill(&m,cos,prompt,BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32);
 for(int profile=1;profile<=5;profile++) {
 BnTransformerBF16Arithmetic arithmetic=(BnTransformerBF16Arithmetic)profile;
 bn_session_reset(s,&m);reset_counts();
 assert(bn_transformer_bf16_request_init_with_arithmetic(&m,gpu,b,5,512,cos,262144,arithmetic)==0);
 assert(alloc_calls==allocation_count&&bn_transformer_bf16_arithmetic(b)==(int)arithmetic);
 float *profile_reference=malloc(logit_bytes*8);assert(profile_reference);int profile_ids[8];
 for(int run=0;run<3;run++) {
  reset_counts();
  if(run)assert(bn_transformer_bf16_session_restart(&m,s)==0&&alloc_calls==0);
  assert(bn_transformer_bf16_arithmetic(b)==(int)arithmetic);
  ProfileObserver probe={bn_gpu_backend_resident_request(gpu,b),0};int previous=0;
  for(int phase=0;phase<8;phase++) {
   if(run==1) {
    int32_t ids[5];int nt=phase?1:5;if(phase)ids[0]=previous;else for(int i=0;i<5;i++)ids[i]=prompt[i];
    assert(bn_transformer_bf16_forward(&m,gpu,b,ids,nt,logits,(size_t)m.config.vocab_size,profile_observer,&probe)==0);x=logits;
   } else x=phase?bn_transformer_forward(&m,s,previous,phase+4):bn_prefill(&m,s,prompt,5,0,0);
   assert(x);previous=token(x,m.config.vocab_size);
   if(!run){memcpy(profile_reference+(size_t)phase*m.config.vocab_size,x,logit_bytes);profile_ids[phase]=previous;}
   else assert(memcmp(profile_reference+(size_t)phase*m.config.vocab_size,x,logit_bytes)==0&&previous==profile_ids[phase]);
   if(!phase) {
    bn_session_reset(other,&m);init(&m,other,cos);
    assert(bn_transformer_bf16_arithmetic(b2)==BN_TRANSFORMER_BF16_ARITHMETIC_EAGER);
    float *eager=bn_prefill(&m,other,prompt,5,0,0);assert(eager&&token(eager,m.config.vocab_size)==11751);
    assert(bn_transformer_bf16_arithmetic(b)==(int)arithmetic);
   }
  }
  assert(bn_transformer_bf16_position(b)==12);if(run==1)assert(probe.calls==8*258);
 }
 /* Same profile and chunk boundaries across wrapper, explicit resume and decode. */
 assert(bn_transformer_bf16_session_restart(&m,s)==0);reset_counts();
 for(int i=0;i<m.config.vocab_size;i++)s->state.logits[i]=-123.0f;
 assert(bn_prefill_no_logits(&m,s,prompt,5,0,0)==0&&sync_calls==1);
 for(int i=0;i<m.config.vocab_size;i++)assert(s->state.logits[i]==-123.0f);
 x=bn_transformer_forward(&m,s,profile_ids[0],5);
 assert(x&&memcmp(x,profile_reference+m.config.vocab_size,logit_bytes)==0);
 assert(bn_transformer_bf16_session_restart(&m,s)==0);
 x=bn_prefill(&m,s,six,6,0,0);assert(x);memcpy(logits,x,logit_bytes);
 int resumed_token=token(x,m.config.vocab_size);
 x=bn_transformer_forward(&m,s,resumed_token,6);assert(x);
 memcpy(profile_reference,x,logit_bytes);
 for(int schedule=0;schedule<2;schedule++) {
  assert(bn_transformer_bf16_session_restart(&m,s)==0);
  if(schedule)assert(bn_prefill(&m,s,prompt,5,0,0));
  else assert(bn_transformer_prefill_no_logits(&m,s,prompt,5,0)==0);
  x=bn_prefill(&m,s,six+5,1,5,0);assert(x&&memcmp(x,logits,logit_bytes)==0);
  x=bn_transformer_forward(&m,s,resumed_token,6);
  assert(x&&memcmp(x,profile_reference,logit_bytes)==0);
 }
 /* Failed publication poisons this arithmetic profile until reset as well. */
 assert(bn_transformer_bf16_session_restart(&m,s)==0);reset_counts();fail_sync=1;
 assert(bn_prefill_no_logits(&m,s,prompt,5,0,0)==-1);
 assert(bn_transformer_bf16_position(b)==-1&&bn_transformer_bf16_arithmetic(b)==-1);
 assert(bn_transformer_bf16_session_restart(&m,s)==-1&&cpu_fallbacks==0);
 puts("REQUEST_FP32_INTERMEDIATES_CHUNK_RESUME_FAILURE PASSED");
 free(profile_reference);bn_session_reset(s,&m);reset_counts();init(&m,s,cos);
 assert(bn_transformer_bf16_arithmetic(b)==BN_TRANSFORMER_BF16_ARITHMETIC_EAGER);
 x=bn_prefill(&m,s,prompt,5,0,0);assert(x&&token(x,m.config.vocab_size)==11751&&cpu_fallbacks==0);
 printf("REQUEST_ARITHMETIC_%d_ISOLATION_RESTART PASSED\n",profile);
 }
 bn_session_free(s,NULL);bn_session_free(other,NULL);free(logits);if(cos)bn_gpu_backend_destroy_buffer(gpu,cos);bn_model_free(&m);bn_gguf_free(f);bn_gpu_cuda_destroy(gpu);puts("REQUEST_RUNTIME PASSED");return 0;
}
