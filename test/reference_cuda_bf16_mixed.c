/* Untimed exact-input diagnostic; schedule is converted from original traces. */
#include "model.h"
#include "model_internal.h"
#include "session.h"
#include "session_internal.h"
#include "gpu_cuda.h"
#include "backend_layout.h"
#include "transformer.h"
#include "bf16_policy_io.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
typedef struct {int id,pos,kind,n;int32_t *tokens;} Row;
typedef struct {int count;Row rows[16];} Call;
static int integer(FILE *f){unsigned char b[4];assert(fread(b,1,4,f)==4);uint32_t x=(uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;assert(x<=INT32_MAX);return (int)x;}
static int reject_cpu(void*p){(void)p;return -1;}
int main(int argc,char **argv){
 if(argc!=5){fprintf(stderr,"usage: %s MODEL.gguf SCHEDULE.bin POLICY.json OUTPUT_DIR\n",argv[0]);return 2;}
 BnTransformerBF16Policy policy;assert(!bench_policy_read(argv[3],&policy));FILE *f=fopen(argv[2],"rb");assert(f);char magic[4];assert(fread(magic,1,4,f)==4&&!memcmp(magic,"BMS1",4));
 int nr=integer(f),budget=integer(f),capacity=integer(f),nc=integer(f);assert(nr>=1&&nr<=16&&budget>=1&&budget<=8192&&capacity>=1&&capacity<=8192&&nc>=1&&nc<=256);
 int positions[16]={0},maximum[16]={0};Call *calls=calloc((size_t)nc,sizeof(*calls));assert(calls);
 for(int c=0;c<nc;c++){
  Call *call=&calls[c];call->count=integer(f);assert(call->count>=1&&call->count<=nr);int total=0,seen[16]={0};
  for(int i=0;i<call->count;i++){
   Row *r=&call->rows[i];r->id=integer(f);r->pos=integer(f);r->kind=integer(f);r->n=integer(f);
   assert(r->id<nr&&!seen[r->id]++&&r->pos==positions[r->id]&&r->kind<=1&&r->n>=1&&r->n<=budget-total&&r->n<=capacity-r->pos&&(!r->kind||r->n==1));
   r->tokens=malloc((size_t)r->n*sizeof(*r->tokens));assert(r->tokens);for(int j=0;j<r->n;j++)r->tokens[j]=integer(f);
   total+=r->n;positions[r->id]+=r->n;if(r->n>maximum[r->id])maximum[r->id]=r->n;
  }
 }
 assert(fgetc(f)==EOF&&!ferror(f));fclose(f);for(int i=0;i<nr;i++)assert(maximum[i]>0);
 BnGGUFFile *file=bn_gguf_open_file(argv[1]);assert(file);BnModel model={0};assert(!bn_model_load_with_cpu_preparation(&model,file,capacity,0,0,0));
 for(int c=0;c<nc;c++)for(int i=0;i<calls[c].count;i++)for(int j=0;j<calls[c].rows[i].n;j++)assert(calls[c].rows[i].tokens[j]<model.config.vocab_size);
 BnGPUBackend *gpu=bn_gpu_cuda_create();assert(gpu);assert(!bn_backend_layout_prepare_bf16_with_math(bn_model_backend(&model),gpu,&model.config,&model.weights,budget,BN_BACKEND_BF16_MATH_MKL_VML_HA));gpu->prepare_cpu_operations=reject_cpu;
 BnSession *sessions[16];for(int i=0;i<nr;i++){sessions[i]=bn_session_create_backend(&model,NULL);assert(sessions[i]);assert(!bn_transformer_bf16_request_init_with_policy(&model,gpu,bn_session_backend(sessions[i]),maximum[i],capacity,NULL,0,&policy));}
 BnBackendSession *co=bn_backend_session_create();assert(co);assert(!bn_transformer_bf16_batch_prepare_packed_with_policy(&model,gpu,co,nr,budget,&policy));
 size_t vocab=(size_t)model.config.vocab_size;unsigned char *raw=malloc(vocab*2);assert(raw);
 for(int c=0;c<nc;c++){
  Call *call=&calls[c];BnTransformerBF16StepRow rows[16];
  for(int i=0;i<call->count;i++){Row*r=&call->rows[i];rows[i]=(BnTransformerBF16StepRow){{bn_session_backend(sessions[r->id]),r->tokens,r->n,r->pos,sessions[r->id]->state.logits,vocab},(BnTransformerBF16StepKind)r->kind};}
  assert(!bn_transformer_bf16_batch_step(&model,gpu,co,rows,call->count,NULL,NULL));
  char path[4096];int n=snprintf(path,sizeof(path),"%s/logits-%d.bin",argv[4],c);assert(n>0&&(size_t)n<sizeof(path));f=fopen(path,"wb");assert(f);
  for(int i=0;i<call->count;i++){
   assert(bn_transformer_bf16_position(rows[i].row.session)==rows[i].row.position+rows[i].row.n_tokens);
   for(size_t j=0;j<vocab;j++){uint32_t word;memcpy(&word,rows[i].row.logits+j,4);assert(!(word&65535u));raw[2*j]=(unsigned char)(word>>16);raw[2*j+1]=(unsigned char)(word>>24);}
   assert(fwrite(raw,2,vocab,f)==vocab);
  }
  assert(!fclose(f));printf("{\"call\":%d,\"rows\":%d}\n",c,call->count);fflush(stdout);
 }
 BnTransformerBF16BatchInfo info;assert(!bn_transformer_bf16_batch_info(co,&info)&&!info.failed&&info.prefill_calls+info.mixed_calls+info.decode_calls==(uint64_t)nc);
 printf("{\"prefill\":%llu,\"mixed\":%llu,\"decode\":%llu}\n",(unsigned long long)info.prefill_calls,(unsigned long long)info.mixed_calls,(unsigned long long)info.decode_calls);
 for(int i=0;i<nr;i++){assert(bn_transformer_bf16_position(bn_session_backend(sessions[i]))==positions[i]);bn_session_free(sessions[i],NULL);}free(raw);bn_backend_session_free(co);bn_model_free(&model);bn_gpu_cuda_destroy(gpu);bn_gguf_free(file);
 for(int c=0;c<nc;c++){for(int i=0;i<calls[c].count;i++)free(calls[c].rows[i].tokens);}
 free(calls);return 0;
}
