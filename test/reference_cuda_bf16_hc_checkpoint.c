/* Compare native BF16 GEMM against the independent checkpoint HC captures.
 * Link with the same objects/libraries as test_cuda_backend, then run:
 *   probe CAPTURE_DIRECTORY CAPTURE_DIRECTORY/plan.txt
 * This checks projections with captured inputs; no resident composition claim.
 */

#include "gpu_cuda.h"
#include "gpu_backend.h"
#include "gguf.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static uint16_t *read_words(const char *base,const char *name,size_t n){
 char path[2048];int len=snprintf(path,sizeof(path),"%s/%s",base,name);assert(len>0&&(size_t)len<sizeof(path));
 FILE*f=fopen(path,"rb");assert(f);uint16_t*p=malloc(n*2);assert(p);
 assert(fread(p,2,n,f)==n);assert(fgetc(f)==EOF);fclose(f);return p;
}
int main(int argc,char**argv){
 if(argc!=3){fprintf(stderr,"usage: %s captures plan.txt\n",argv[0]);return 2;}FILE*f=fopen(argv[2],"r");assert(f);BnGPUBackend*g=bn_gpu_cuda_create();assert(g);
 char w[256],x[256],y[256];int rows,cols,nt,cases=0;size_t total=0,diffs=0;
 while(fscanf(f,"%255s %255s %255s %d %d %d",w,x,y,&rows,&cols,&nt)==6){
  assert(rows>0&&cols>0&&nt>0);
  uint16_t*W=read_words(argv[1],w,(size_t)rows*cols),*X=read_words(argv[1],x,(size_t)nt*cols);
  size_t n=(size_t)nt*rows;uint16_t*Y=read_words(argv[1],y,n),*O=malloc(n*2);assert(O);
  void*h=bn_gpu_backend_create_quant_only_buffer(g,W,(size_t)rows*cols*2,BN_GGUF_TENSOR_BF16,rows,cols);assert(h);
  assert(!bn_gpu_backend_matmul_bf16(g,O,h,X,rows,cols,nt));size_t d=0;
  for(size_t i=0;i<n;i++)if(O[i]!=Y[i])d++;
  printf("%s %d x %d tokens%d differences%zu/%zu\n",y,rows,cols,nt,d,n);fflush(stdout);
  diffs+=d;total+=n;cases++;g->buffer_destroy(g->ctx,h);free(W);free(X);free(Y);free(O);
 }
 assert(feof(f)&&cases==72);fclose(f);bn_gpu_cuda_destroy(g);printf("PROJECTION cases%d words%zu differences%zu\n",cases,total,diffs);
 return diffs?1:0;
}
