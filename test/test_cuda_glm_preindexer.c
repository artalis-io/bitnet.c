#include "gpu_cuda.h"
#include "backend_session.h"
#include "backend_layout.h"
#include "model_arch.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void *load(const char *dir,const char *name,size_t bytes){size_t nd=strlen(dir),nn=strlen(name);assert(nn<=SIZE_MAX-6&&nd<=SIZE_MAX-nn-6);char *path=malloc(nd+nn+6);assert(path);memcpy(path,dir,nd);path[nd]='/';memcpy(path+nd+1,name,nn);memcpy(path+nd+nn+1,".bin",5);FILE *f=fopen(path,"rb");free(path);assert(f);void *data=malloc(bytes?bytes:1);assert(data);assert(fread(data,1,bytes,f)==bytes&&fgetc(f)==EOF);fclose(f);return data;}
static BnGPUResidentView alloc(BnGPUBackend *g,void *r,int type,size_t n,const void *data){BnGPUResidentView v={0};if(n){assert(!g->resident_alloc(r,type,n,&v));if(data)assert(!g->resident_write(r,v,data));}return v;}
static size_t check(BnGPUBackend *g,void *r,BnGPUResidentView v,const char *dir,const char *name){size_t width=bn_quant_format_resident_scalar_bytes(v.type),bytes=v.count*width;if(!v.count)bytes=0;void *expected=load(dir,name,bytes),*actual=malloc(bytes?bytes:1);assert(actual);if(bytes)assert(!g->resident_read(r,v,actual));if(memcmp(expected,actual,bytes)){for(size_t i=0;i<v.count;++i)if(memcmp((char*)expected+i*width,(char*)actual+i*width,width)){fprintf(stderr,"%s %s mismatch element%zu\n",dir,name,i);break;}exit(3);}free(expected);free(actual);return v.count;}
int main(int argc,char **argv){assert(argc==3);BnGGUFFile *file=bn_gguf_open_file(argv[1]);assert(file);BnModelAnatomy *a=malloc(sizeof(*a));assert(a);assert(!bn_model_arch_plan_anatomy(file,a));BnGPUBackend *g=bn_gpu_cuda_create();assert(g);BnBackendBF16IndexerWeights w={0};assert(!bn_backend_layout_bf16_indexer_init(&w,g,file,a,3,8326144)&&w.bytes==8326144);
 int seq[][10]={{1},{2},{6},{7},{9},{65},{129},{1025},{2,7},{3,3,3},{7,1,1},{1,1,1,1,1,1,1,1,1}};int lengths[]={1,1,1,1,1,1,1,1,2,3,3,9};size_t words=0;uint16_t zero[1024]={0};
 for(int warm=0;warm<2;++warm)for(int ci=0;ci<12;++ci){int steps[12],ns=0,maxnt=1;if(warm)steps[ns++]=3;for(int j=0;j<lengths[ci];++j){steps[ns++]=seq[ci][j];if(seq[ci][j]>maxnt)maxnt=seq[ci][j];}if(warm&&maxnt<3)maxnt=3;
  BnBackendSession *s=bn_backend_session_create();assert(s);void *r=bn_gpu_backend_resident_request(g,s);assert(r);assert(!g->linear_bf16_reserve(g->ctx,r,(size_t)maxnt*w.dim));BnGPUResidentView tail=alloc(g,r,BN_GGUF_TENSOR_BF16,1024,zero),next=alloc(g,r,BN_GGUF_TENSOR_BF16,1024,NULL);int pos=0;
  for(int phase=0;phase<ns;++phase){int nt=steps[phase];size_t n=nt,groups=(pos+nt)/4-pos/4;char dir[1024];snprintf(dir,sizeof(dir),"%s/%d/%d",argv[2],warm*12+ci,phase);void *x=load(dir,"x",n*w.dim*2),*qr=load(dir,"qr",n*w.q_rank*2);size_t counts[]={n*w.dim,n*w.q_rank,1024,1024,n*4096,n*128,n*128,n*128,n*w.dim,n*32,n*4096,n*32,n*32,groups*128,groups};BnGPUResidentView v[15];
   for(int j=0;j<15;++j){int type=j==8||j==9||j==11||j==12||j==14?BN_GGUF_TENSOR_F32:j==10||j==13?BN_GGUF_TENSOR_I8:BN_GGUF_TENSOR_BF16;v[j]=j==2?tail:j==3?next:alloc(g,r,type,counts[j],j==0?x:j==1?qr:NULL);}
   BnTransformerBF16IndexerPlan p={.weights=&w,.layer=3,.n_tokens=nt,.position=pos,.x=v[0],.qr=v[1],.initial_tail=v[2],.next_tail=v[3],.q=v[4],.key_raw=v[5],.key=v[6],.gate=v[7],.cast=v[8],.head_raw=v[9],.q_fp8=v[10],.q_scales=v[11],.head_weights=v[12],.compressed=v[13],.compressed_scales=v[14]};BnTransformerBF16IndexerPlan bad=p;bad.head_weights=bad.head_raw;assert(bn_transformer_bf16_indexer(r,&bad)==-1);bad=p;bad.layer++;assert(bn_transformer_bf16_indexer(r,&bad)==-1);assert(!bn_transformer_bf16_indexer(r,&p));
   int slots[]={4,5,6,7,9,10,11,12,13,14,3};const char *names[]={"q","key_raw","key","gate","head_raw","q_fp8","q_scales","head_weights","compressed","compressed_scales","next_tail"};for(int j=0;j<11;++j)words+=check(g,r,v[slots[j]],dir,names[j]);check(g,r,p.x,dir,"x");check(g,r,p.qr,dir,"qr");check(g,r,p.initial_tail,dir,"initial_tail");BnGPUResidentView swap=tail;tail=next;next=swap;pos+=nt;free(x);free(qr);
  }bn_backend_session_free(s);printf("case%d pre-indexer PASSED tokens%d phases%d\n",warm*12+ci,pos,ns);fflush(stdout);
 }printf("PASSED %zu exact pre-indexer output elements\n",words);bn_backend_layout_bf16_indexer_free(&w);bn_gpu_cuda_destroy(g);bn_gguf_free(file);free(a);return 0;}
