/* Actual checkpoint PLE resident chain. Link like test_cuda_backend.
 * Usage: probe FIRST_SHARD CAPTURE_DIRECTORY. Captures are comparisons only. */
#include "gpu_cuda.h"
#include "gpu_backend.h"
#include "model.h"
#include "model_internal.h"
#include "backend_quant.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "backend_session.h"
#include "transformer_prefill_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif
static uint16_t bf16(float x){uint32_t u;memcpy(&u,&x,4);return(uint16_t)((u+0x7fff+((u>>16)&1))>>16);}
static void pattern(uint16_t *x,size_t n,int role){for(size_t i=0;i<n;i++)x[i]=bf16(ldexpf((float)((int)((i*(13+role*6)+role*19)%509)-254)/128,(int)(i%7)-3));}
static BnGPUResidentView alloc(BnGPUBackend*g,void*r,size_t n){BnGPUResidentView v;assert(!g->resident_alloc(r,BN_GGUF_TENSOR_BF16,n,&v));return v;}
static size_t compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*name,const char*stage){
 char path[2048];int len=snprintf(path,sizeof(path),"%s/%s-%s.bin",dir,name,stage);assert(len>0&&(size_t)len<sizeof(path));
 FILE*f=fopen(path,"rb");assert(f);uint16_t *gold=malloc(v.count*2),*out=malloc(v.count*2);assert(gold&&out);
 assert(fread(gold,2,v.count,f)==v.count&&fgetc(f)==EOF);fclose(f);assert(!g->resident_read(r,v,out));
 size_t diffs=0;for(size_t i=0;i<v.count;i++)if(gold[i]!=out[i]){if(diffs<3)fprintf(stderr,"%s %s word%zu %04x expected%04x\n",name,stage,i,out[i],gold[i]);diffs++;}
 printf("%s %s words%zu differences%zu\n",name,stage,v.count,diffs);fflush(stdout);assert(!diffs);free(gold);free(out);return v.count;
}
int main(int argc,char**argv){
 assert(argc==3);BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModel m={0};assert(!bn_model_load_with_cpu_preparation(&m,f,256,0,0,0));
 assert(m.config.ple_layer==1&&m.config.dim==2560&&m.config.hyper_connection_count==4);
 BnPositionalLayerEmbeddingWeights*p=&m.weights.layers[1].ple;
 assert(p->key.rows==10240&&p->key.cols==2560&&p->value.rows==2560&&p->value.cols==2560&&p->key.type==p->value.type);
 BnGPUBackend*g=bn_gpu_cuda_create();assert(g);assert(g->buffer_create_stacked2);
 BnBackendModel *owner=bn_model_backend(&m);assert(owner);
 assert(!bn_backend_layout_prepare_bf16_hc_model(owner,g,&m.config,&m.weights,SIZE_MAX));
 assert(!bn_backend_layout_prepare_bf16_ple_model(owner,g,&m.config,&m.weights,SIZE_MAX,SIZE_MAX));
 const BnBackendBF16PLEWeights *owned=bn_backend_layout_bf16_ple_model(owner);assert(owned);
 assert(!bn_backend_layout_bf16(owner));
 assert(owned->bytes==65679360u&&owned->peak_bytes==100352000u&&owned->host_staging_bytes==34816000u);
 printf("PLE_OWNED bytes%zu peak%zu host%zu model_bytes%zu model_peak%zu\n",owned->bytes,owned->peak_bytes,owned->host_staging_bytes,owned->model_bytes,owned->model_peak_bytes);
 void *kv=owned->kv.decoded,*conv=owned->conv;void *norms[]={owned->norm_key,owned->norm_query,owned->norm_conv};
 const int nts[]={1,5,17,129,129,129},chunks[]={1,5,17,129,64,1};size_t words=0;
 for(int ci=0;ci<6;ci++){
  int nt=nts[ci],chunk=chunks[ci];char name[64];snprintf(name,sizeof(name),"%d-%d",nt,chunk);
  BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);
  assert(!g->linear_bf16_reserve(g->ctx,r,0));assert(!g->resident_upload_reserve(r,2,129u*16*96,2u*129*16*96));
  int32_t tokens[129],ids[129*16],starts[]={0,nt},ctx[]={248044,116};
  for(int i=0;i<nt;i++)tokens[i]=i%11==5?248044:(i*37+97)%248000;
  assert(!bn_transformer_prefill_ple_ids(&m.config,tokens,nt,starts,1,ctx,2,ids,(size_t)nt*16));
  char path[2048];snprintf(path,sizeof(path),"%s/%s-ids.bin",argv[2],name);FILE*idf=fopen(path,"rb");assert(idf);int32_t goldids[129*16];assert(fread(goldids,4,(size_t)nt*16,idf)==(size_t)nt*16&&fgetc(idf)==EOF);fclose(idf);assert(!memcmp(ids,goldids,(size_t)nt*16*4));
  BnGPUResidentView packed;assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,129u*16*96/4,&packed));
  BnGPUResidentView embedding=alloc(g,r,(size_t)nt*2560),merged=alloc(g,r,(size_t)nt*12800),key=alloc(g,r,(size_t)nt*10240),value=alloc(g,r,(size_t)nt*2560);
  BnGPUResidentView hidden=alloc(g,r,(size_t)nt*10240),kn=alloc(g,r,(size_t)nt*10240),qn=alloc(g,r,(size_t)nt*10240),gated=alloc(g,r,(size_t)nt*10240),normed=alloc(g,r,(size_t)nt*10240),out=alloc(g,r,(size_t)nt*10240);
  BnGPUResidentView hist[]={alloc(g,r,9u*10240),alloc(g,r,9u*10240)};
  uint16_t*host=malloc(((size_t)nt*10240>9u*10240?(size_t)nt*10240:9u*10240)*2);assert(host);
  pattern(host,(size_t)nt*10240,2);assert(!g->resident_write(r,hidden,host));pattern(host,9u*10240,4);assert(!g->resident_write(r,hist[0],host));free(host);
  int busy_fences=0;
  int step=0;
  for(int start=0;start<nt;start+=chunk,step++){
   int take=nt-start<chunk?nt-start:chunk;
   BnGPUHostUploadLease lease={0};int status=g->resident_upload_acquire(r,(unsigned)step%2,&lease);
   if(status==BN_GPU_UPLOAD_BUSY){busy_fences++;assert(!g->resident_synchronize(r));status=g->resident_upload_acquire(r,(unsigned)step%2,&lease);}assert(!status);
   assert(!bn_backend_quant_gather_host_rows(&m.weights.ple_token_embd,ids+(size_t)start*16,(size_t)take*16,lease.data,(size_t)take*16*96));
   BnGPUQuantRowsBF16ResidentPlan rows={0};rows.rows=take*16;rows.cols=160;rows.type=m.weights.ple_token_embd.type;rows.row_stride_bytes=96;rows.packed=bn_gpu_resident_view_slice(packed,0,(size_t)take*16*96/4);rows.out=bn_gpu_resident_view_slice(embedding,(size_t)start*2560,(size_t)take*2560);
   assert(!g->resident_upload_submit(r,&lease,rows.packed));assert(!g->quant_rows_bf16_resident(g->ctx,r,&rows));
   BnGPUBF16LinearResidentPlan linear={0};linear.n_tokens=take;linear.rows=12800;linear.cols=2560;linear.weight.type=BN_GGUF_TENSOR_BF16;linear.weight.decoded=kv;linear.x=rows.out;linear.out=bn_gpu_resident_view_slice(merged,(size_t)start*12800,(size_t)take*12800);assert(!g->linear_bf16_resident(g->ctx,r,&linear));
   BnGPUSplitColumnsResidentPlan split={0};split.rows=take;split.cols=12800;split.left_cols=10240;split.x=linear.out;split.left=bn_gpu_resident_view_slice(key,(size_t)start*10240,(size_t)take*10240);split.right=bn_gpu_resident_view_slice(value,(size_t)start*2560,(size_t)take*2560);assert(!g->split_columns_resident(g->ctx,r,&split));
   BnGPUPLEGateResidentPlan gate={0};gate.n_tokens=take;gate.dim=2560;gate.streams=4;gate.eps=1e-6f;gate.norm_key=norms[0];gate.norm_query=norms[1];gate.norm_conv=norms[2];gate.key=split.left;gate.value=split.right;
   gate.hidden=bn_gpu_resident_view_slice(hidden,(size_t)start*10240,(size_t)take*10240);gate.normalized_key=bn_gpu_resident_view_slice(kn,(size_t)start*10240,(size_t)take*10240);gate.normalized_query=bn_gpu_resident_view_slice(qn,(size_t)start*10240,(size_t)take*10240);gate.gated=bn_gpu_resident_view_slice(gated,(size_t)start*10240,(size_t)take*10240);gate.normed=bn_gpu_resident_view_slice(normed,(size_t)start*10240,(size_t)take*10240);assert(!g->ple_gate_bf16_resident(g->ctx,r,&gate));
   BnGPUPLEConvResidentPlan cp={0};cp.n_tokens=take;cp.dim=10240;cp.kernel=4;cp.dilation=3;cp.weight=conv;cp.x=gate.normed;cp.residual=gate.gated;cp.outer=gate.hidden;cp.initial_history=hist[step%2];cp.next_history=hist[(step+1)%2];cp.out=bn_gpu_resident_view_slice(out,(size_t)start*10240,(size_t)take*10240);assert(!g->ple_conv_bf16_resident(g->ctx,r,&cp));
  }
  /* All comparison reads follow the complete queued chain. */
  const char*stages[]={"embedding","key","value","gated","normed","out","history","hidden"};BnGPUResidentView views[]={embedding,key,value,gated,normed,out,hist[step%2],hidden};
  for(int j=0;j<8;j++){words+=compare(g,r,views[j],argv[2],name,stages[j]);}
  printf("PLE_CHAIN_SYNC %s staging_busy_fences%d\n",name,busy_fences);
  bn_backend_session_free(s);
 }
 bn_model_free(&m);bn_gpu_cuda_destroy(g);bn_gguf_free(f);
 printf("PLE_CHAIN_PASSED 6 cases exact_words%zu\n",words);return 0;
}
