/* Resident paged QSA API qualification using independent stage captures.
 * Usage: probe CASE_INDEX WORKSPACE_ORACLE [--paged | --profile].
 * Captured Q/K/V/indices/gates are explicit inputs, not full-model proof. */
#include "gpu_cuda.h"
#include "gguf.h"
#include "gpu_backend.h"
#include "backend_session.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif
static BnGPUResidentView slice(BnGPUResidentView v,size_t offset,size_t count){return bn_gpu_resident_view_slice(v,offset,count);}
static void *load(const char*path,size_t *bytes){FILE*f=fopen(path,"rb");assert(f);assert(!fseek(f,0,SEEK_END));long end=ftell(f);assert(end>0&&!fseek(f,0,SEEK_SET));*bytes=(size_t)end;void*p=malloc(*bytes);assert(p&&fread(p,1,*bytes,f)==*bytes&&fgetc(f)==EOF);fclose(f);return p;}
static BnGPUResidentView alloc(BnGPUBackend*g,void*r,int type,size_t count){BnGPUResidentView v;assert(!g->resident_alloc(r,type,count,&v));return v;}
static BnGPUResidentView upload(BnGPUBackend*g,void*r,int type,size_t count,const void*data){BnGPUResidentView v=alloc(g,r,type,count);assert(!g->resident_write(r,v,data));return v;}
static BnGPUResidentView file_view(BnGPUBackend*g,void*r,int type,int width,const char*path,size_t expected){size_t n;void*data=load(path,&n);assert(n==expected*(size_t)width);BnGPUResidentView v;if(type==BN_GGUF_TENSOR_I32){v=alloc(g,r,type,expected+1);v=slice(v,1,expected);assert(!g->resident_write(r,v,data));}else v=upload(g,r,type,expected,data);free(data);return v;}
static size_t compare(BnGPUBackend*g,void*r,const char*path,BnGPUResidentView v,int width,int ci,const char*stage){
 size_t bytes;void*gold=load(path,&bytes);assert(bytes==v.count*(size_t)width);void*out=malloc(bytes);assert(out&&!g->resident_read(r,v,out));size_t errors=0;
 for(size_t i=0;i<v.count;i++)if(memcmp((char*)out+i*width,(char*)gold+i*width,(size_t)width)){if(errors<2)fprintf(stderr,"case%d %s word%zu differs\n",ci,stage,i);errors++;}
 printf("case%d %s words%zu differences%zu\n",ci,stage,v.count,errors);fflush(stdout);assert(!errors);free(out);free(gold);return v.count;
}
static int submit(BnGPUBackend*g,void*r,const BnGPUQSAAttentionResidentPlan*p){return g->qsa_attention_bf16_resident(g->ctx,r,p);}
static void guards(BnGPUBackend*g,void*r,void*foreign,const BnGPUQSAAttentionResidentPlan*p,size_t needed){
 uint16_t sentinel[16],actual[16];for(int i=0;i<16;i++)sentinel[i]=(uint16_t)(0x3e00+i);
 assert(!g->resident_write(r,slice(p->out,0,16),sentinel));
 BnGPUQSAAttentionResidentPlan bad=*p;
 assert(g->qsa_attention_bf16_resident(NULL,r,p)==-1);assert(submit(g,NULL,p)==-1);assert(submit(g,r,NULL)==-1);assert(submit(g,foreign,p)==-1);
 bad.n_tokens=0;assert(submit(g,r,&bad)==-1);bad=*p;bad.n_tokens=131071;assert(submit(g,r,&bad)==-1);
 bad=*p;bad.use_prefill_config=2;assert(submit(g,r,&bad)==-1);
 bad=*p;bad.num_cache_blocks=0;assert(submit(g,r,&bad)==-1);bad=*p;bad.num_requests=0;assert(submit(g,r,&bad)==-1);bad=*p;bad.table_width=0;assert(submit(g,r,&bad)==-1);
 bad=*p;bad.num_cache_blocks=INT_MAX;assert(submit(g,r,&bad)==-1);bad=*p;bad.num_requests=INT_MAX;bad.table_width=INT_MAX;assert(submit(g,r,&bad)==-1);
 for(int i=0;i<8;i++){
  bad=*p;BnGPUResidentView*views[]={&bad.q,&bad.k_cache,&bad.v_cache,&bad.indices,&bad.block_table,&bad.token_to_req,&bad.gate,&bad.out};
  views[i]->request=foreign;assert(submit(g,r,&bad)==-1);
  bad=*p;views[i]->type=BN_GGUF_TENSOR_F32;assert(submit(g,r,&bad)==-1);
  bad=*p;views[i]->count--;assert(submit(g,r,&bad)==-1);
  bad=*p;views[i]->count++;assert(submit(g,r,&bad)==-1);
  bad=*p;views[i]->slot=UINT32_MAX;assert(submit(g,r,&bad)==-1);
 }
 bad=*p;bad.out=p->q;assert(submit(g,r,&bad)==-1);
 bad=*p;bad.q.offset++;assert(submit(g,r,&bad)==-1);
 bad=*p;bad.workspace.request=foreign;assert(submit(g,r,&bad)==-1);
 bad=*p;bad.workspace.type=BN_GGUF_TENSOR_I32;assert(submit(g,r,&bad)==-1);
 bad=*p;bad.workspace.offset++;bad.workspace.count--;assert(submit(g,r,&bad)==-1);
 bad=*p;bad.workspace=p->out;assert(submit(g,r,&bad)==-1);
 bad=*p;bad.workspace=(BnGPUResidentView){0};bad.workspace.slot=1;assert(submit(g,r,&bad)==-1);
 if(needed){bad=*p;bad.workspace.count=needed-1;assert(submit(g,r,&bad)==-1);bad=*p;bad.workspace=(BnGPUResidentView){0};assert(submit(g,r,&bad)==-1);}
 assert(!g->resident_read(r,slice(p->out,0,16),actual));assert(!memcmp(actual,sentinel,sizeof(actual)));
}
int main(int argc,char**argv){
 assert(argc>=3&&argc<=4);int paged=0,profile=0;if(argc==4){paged=!strcmp(argv[3],"--paged");profile=!strcmp(argv[3],"--profile");assert(paged||profile);}
 BnGPUBackend*g=bn_gpu_cuda_create(),*other=bn_gpu_cuda_create();assert(g&&other&&g->qsa_attention_bf16_resident&&g->qsa_attention_bf16_workspace_size&&g->qsa_attention_bf16_prepare);
 BnBackendSession*foreign_session=bn_backend_session_create();assert(foreign_session);void*foreign=bn_gpu_backend_resident_request(other,foreign_session);assert(foreign);
 size_t unchanged=77;assert(g->qsa_attention_bf16_workspace_size(NULL,1,&unchanged)==-1&&unchanged==77);
 int invalid[]={0,-1,131071,INT_MAX};for(size_t j=0;j<sizeof(invalid)/sizeof(*invalid);j++)assert(g->qsa_attention_bf16_workspace_size(g->ctx,invalid[j],&unchanged)==-1&&unchanged==77);
 assert(g->qsa_attention_bf16_workspace_size(g->ctx,1,NULL)==-1);
 FILE*oracle=fopen(argv[2],"r");assert(oracle);int cap,queries=0;size_t expected;
 while(fscanf(oracle,"%d %zu",&cap,&expected)==2){size_t bytes=0;assert(!g->qsa_attention_bf16_workspace_size(g->ctx,cap,&bytes)&&bytes==expected);queries++;}assert(feof(oracle)&&queries==303);fclose(oracle);printf("QSA_WORKSPACE_QUERIES_PASSED %d\n",queries);
 FILE*f=fopen(argv[1],"r");assert(f);char qp[4096],kp[4096],vp[4096],ip[4096],gp[4096],pp[4096],lp[4096],op[4096];int rows,bn,warps,splits,cases=0;size_t partial_words=0,lse_words=0,out_words=0;
 while(fscanf(f,"%4095s %4095s %4095s %4095s %4095s %4095s %4095s %4095s %d %d %d %d",qp,kp,vp,ip,gp,pp,lp,op,&rows,&bn,&warps,&splits)==12){
  assert(rows>0);int prefill=1;if(bn==64&&splits==1&&rows>1024)prefill=0;
  BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);
  BnGPUQSAAttentionResidentPlan p={0};p.n_tokens=rows;p.use_prefill_config=rows<=1024?cases%2:prefill;
  size_t n=(size_t)rows*24*256,raw_bytes;void*raw=load(kp,&raw_bytes);
  if(paged){
   char tp[4096],rp[4096];assert(fscanf(f,"%4095s %4095s %d %d",tp,rp,&p.table_width,&p.num_requests)==4);assert(raw_bytes%(16*1024)==0);p.num_cache_blocks=(int)(raw_bytes/(16*1024));
   p.k_cache=upload(g,r,BN_GGUF_TENSOR_BF16,raw_bytes/2,raw);free(raw);p.v_cache=file_view(g,r,BN_GGUF_TENSOR_BF16,2,vp,raw_bytes/2);
   p.block_table=file_view(g,r,BN_GGUF_TENSOR_I32,4,tp,(size_t)p.table_width*p.num_requests);p.token_to_req=file_view(g,r,BN_GGUF_TENSOR_I32,4,rp,(size_t)rows);
  }else{
   assert(raw_bytes%1024==0);int visible=(int)(raw_bytes/1024);p.table_width=(visible+15)/16;p.num_cache_blocks=p.table_width+1;p.num_requests=1;
   size_t bytes=(size_t)p.num_cache_blocks*16*1024;void*cache=calloc(1,bytes);assert(cache);memcpy((char*)cache+16*1024,raw,raw_bytes);free(raw);p.k_cache=upload(g,r,BN_GGUF_TENSOR_BF16,bytes/2,cache);
   size_t value_bytes;raw=load(vp,&value_bytes);assert(value_bytes==raw_bytes);memcpy((char*)cache+16*1024,raw,value_bytes);free(raw);p.v_cache=upload(g,r,BN_GGUF_TENSOR_BF16,bytes/2,cache);free(cache);
   int*table=malloc((size_t)p.table_width*4);assert(table);for(int j=0;j<p.table_width;j++)table[j]=j+1;p.block_table=upload(g,r,BN_GGUF_TENSOR_I32,(size_t)p.table_width,table);free(table);
   p.token_to_req=alloc(g,r,BN_GGUF_TENSOR_I32,(size_t)rows);assert(!g->resident_zero(r,p.token_to_req));
  }
  p.q=file_view(g,r,BN_GGUF_TENSOR_BF16,2,qp,n);p.gate=file_view(g,r,BN_GGUF_TENSOR_BF16,2,gp,n);p.indices=file_view(g,r,BN_GGUF_TENSOR_I32,4,ip,(size_t)rows*2052);p.out=alloc(g,r,BN_GGUF_TENSOR_BF16,n);
  size_t scratch_bytes;assert(!g->qsa_attention_bf16_workspace_size(g->ctx,rows,&scratch_bytes));assert(scratch_bytes&&scratch_bytes%4==0);size_t scratch_count=scratch_bytes/4;
  BnGPUResidentView scratch=alloc(g,r,BN_GGUF_TENSOR_F32,scratch_count+8);p.workspace=slice(scratch,4,scratch_count);
  uint32_t canary[]={0x3eaaaaab,0x3f7fffff,0xbf234567,0x01234567},got[4];
  assert(!g->resident_write(r,slice(scratch,0,4),canary));assert(!g->resident_write(r,slice(scratch,scratch_count+4,4),canary));
  size_t partial_count=splits>1?n*(size_t)splits:0,lse_count=splits>1?(size_t)rows*splits*24:0;
  assert(partial_count+lse_count<=scratch_count);
  if(!cases){assert(submit(g,r,&p)==-1);assert(g->qsa_attention_bf16_prepare(NULL)==-1);assert(!g->qsa_attention_bf16_prepare(g->ctx));assert(!g->qsa_attention_bf16_prepare(g->ctx));}
  guards(g,r,foreign,&p,partial_count+lse_count);
  BnGPUQSAAttentionResidentPlan alias=p;alias.gate=p.q;assert(!submit(g,r,&alias));assert(!submit(g,r,&p));
  if(splits==1){BnGPUQSAAttentionResidentPlan no_scratch=p;no_scratch.workspace=(BnGPUResidentView){0};assert(!submit(g,r,&no_scratch));}
  assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
  if(profile){assert(cudaProfilerStart()==cudaSuccess);assert(!submit(g,r,&p));assert(!g->resident_synchronize(r));assert(cudaProfilerStop()==cudaSuccess);}
#else
  assert(!profile);
#endif
  if(splits>1){partial_words+=compare(g,r,pp,slice(p.workspace,0,partial_count),4,cases,"partial");lse_words+=compare(g,r,lp,slice(p.workspace,partial_count,lse_count),4,cases,"lse");}
  out_words+=compare(g,r,op,p.out,2,cases,"attention");
  assert(!g->resident_read(r,slice(scratch,0,4),got)&&!memcmp(got,canary,sizeof(got)));assert(!g->resident_read(r,slice(scratch,scratch_count+4,4),got)&&!memcmp(got,canary,sizeof(got)));
  printf("case%d API_GUARDS_PASSED workspace_bytes%zu\n",cases,scratch_bytes);cases++;bn_backend_session_free(s);
 }
 assert(cases>0&&feof(f));fclose(f);bn_backend_session_free(foreign_session);bn_gpu_cuda_destroy(other);bn_gpu_cuda_destroy(g);
 printf("QSA_RESIDENT_PASSED cases%d partial_words%zu lse_words%zu out_words%zu\n",cases,partial_words,lse_words,out_words);return 0;
}
