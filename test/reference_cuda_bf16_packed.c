/* Uniform prompts (1..16 requests, at most 8192 tokens per packed chunk) plus 63 decode steps for independent
 * reference qualification. Output directory must exist. Not a timing benchmark. */
#include "model.h"
#include "model_internal.h"
#include "session.h"
#include "session_internal.h"
#include "gpu_cuda.h"
#include "backend_layout.h"
#include "transformer.h"
#include "bf16_policy_io.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int greedy(const float *x,int n){int id=0;for(int i=1;i<n;i++)if(x[i]>x[id])id=i;assert(isfinite(x[id]));return id;}
static int reject_cpu(void *p){(void)p;return -1;}
int main(int argc,char **argv){
 if(argc<5||argc>7){fprintf(stderr,"usage: %s MODEL.gguf PROMPT.i32 POLICY.json OUTPUT_DIR [REQUESTS [PREFILL_CHUNK]]\n",argv[0]);return 2;}
 int count=4;if(argc>=6){char *end;long value=strtol(argv[5],&end,10);if(!argv[5][0]||*end||value<1||value>16)return 2;count=(int)value;}
 int chunk=0;if(argc==7){char *end;long value=strtol(argv[6],&end,10);if(!argv[6][0]||*end||value<1||value>8192/count)return 2;chunk=(int)value;}
 BnTransformerBF16Policy policy;assert(!bench_policy_read(argv[3],&policy));
 int32_t (*tokens)[8192]=calloc((size_t)count,sizeof(*tokens));assert(tokens);
 int max_prompt=chunk?8192-64:8192/count;if(max_prompt>8192-64)max_prompt=8192-64;
 FILE *f=fopen(argv[2],"rb");assert(f);
 size_t bytes=fread(tokens[0],1,(size_t)max_prompt*sizeof(tokens[0][0]),f);
 assert(!ferror(f)&&bytes>=2*sizeof(tokens[0][0])&&bytes%sizeof(tokens[0][0])==0&&fgetc(f)==EOF);
 fclose(f);int np=(int)(bytes/sizeof(tokens[0][0]));if(!chunk)chunk=np;
 BnGGUFFile *file=bn_gguf_open_file(argv[1]);assert(file);BnModel model={0};
 assert(!bn_model_load_with_cpu_preparation(&model,file,8192,0,0,0));BnGPUBackend *gpu=bn_gpu_cuda_create();assert(gpu);
 assert(!bn_backend_layout_prepare_bf16_with_math(bn_model_backend(&model),gpu,&model.config,&model.weights,count*chunk,BN_BACKEND_BF16_MATH_MKL_VML_HA));
 gpu->prepare_cpu_operations=reject_cpu;
 BnSession *sessions[16];BnTransformerBF16PrefillRow rows[16];
 for(int i=0;i<count;i++){
  if(i)memcpy(tokens[i],tokens[0],(size_t)np*sizeof(tokens[0][0]));
  int j=1+i%(np-1),t=tokens[i][1];tokens[i][1]=tokens[i][j];tokens[i][j]=t;
  sessions[i]=bn_session_create_backend(&model,NULL);assert(sessions[i]);
  assert(!bn_transformer_bf16_request_init_with_policy(&model,gpu,bn_session_backend(sessions[i]),chunk,8192,NULL,0,&policy));
  rows[i]=(BnTransformerBF16PrefillRow){bn_session_backend(sessions[i]),tokens[i],np,0,sessions[i]->state.logits,(size_t)model.config.vocab_size};
 }
 BnBackendSession *co=bn_backend_session_create();assert(co);
 assert(!bn_transformer_bf16_batch_prepare_packed_with_policy(&model,gpu,co,count,count*chunk,&policy));
 BnTransformerBF16Policy admitted;assert(!bn_transformer_bf16_policy(co,&admitted));fputs("{\"kind\":\"policy\",\"admitted\":",stdout);assert(!bench_policy_write(stdout,&admitted));puts("}");
 int ids[16]={0};size_t vocab=(size_t)model.config.vocab_size;uint16_t *bits=malloc(vocab*2);assert(bits);
 for(int pos=0;pos<np;pos+=chunk){
  int n=np-pos;if(n>chunk)n=chunk;BnTransformerBF16PrefillRow prefill[16];
  for(int i=0;i<count;i++){
   prefill[i]=rows[i];prefill[i].tokens=tokens[i]+pos;prefill[i].n_tokens=n;prefill[i].position=pos;
   if(pos+n<np){prefill[i].logits=NULL;prefill[i].logits_count=0;}
  }
  assert(!bn_transformer_bf16_batch_prefill(&model,gpu,co,prefill,count,NULL,NULL));
 }
 for(int step=0;step<64;step++){
  if(step){BnTransformerBF16DecodeRow decode[16];for(int i=0;i<count;i++)decode[i]=(BnTransformerBF16DecodeRow){rows[i].session,ids[i],np-1+step,rows[i].logits,vocab};assert(!bn_transformer_bf16_batch_decode(&model,gpu,co,decode,count));}
  char path[4096];int n=snprintf(path,sizeof(path),"%s/logits-%d.bin",argv[4],step);assert(n>0&&(size_t)n<sizeof(path));f=fopen(path,"wb");assert(f);
  printf("{\"kind\":\"step\",\"step\":%d,\"tokens\":[",step);
  for(int i=0;i<count;i++){
   ids[i]=greedy(rows[i].logits,model.config.vocab_size);printf("%s%d",i?",":"",ids[i]);
   for(size_t j=0;j<vocab;j++){uint32_t word;memcpy(&word,rows[i].logits+j,4);assert((word&65535)==0);bits[j]=(uint16_t)(word>>16);}
   assert(fwrite(bits,2,vocab,f)==vocab);
  }
  puts("]}");fclose(f);fflush(stdout);
 }
 for(int i=0;i<count;i++){assert(bn_transformer_bf16_position(rows[i].session)==np+63);bn_session_free(sessions[i],NULL);}
 free(tokens);free(bits);bn_backend_session_free(co);bn_model_free(&model);bn_gpu_cuda_destroy(gpu);bn_gguf_free(file);return 0;
}
