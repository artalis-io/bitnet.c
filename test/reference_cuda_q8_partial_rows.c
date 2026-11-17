/* Independent GGML df18bee11 CUDA SM120 partial-row MMQ oracle. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t hash(const float*x,size_t n){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++){uint32_t u;memcpy(&u,x+i,4);h=(h^u)*UINT64_C(1099511628211);}return h;}
int main(void){
 const int shapes[][2]={{64,1024},{96,256},{320,10240},{128,1024},{640,2560}},tokens[]={65,129,193,257};
 ggml_backend_t b=ggml_backend_cuda_init(0);assert(b);
 for(int shape=0;shape<5;shape++)for(int ti=0;ti<4;ti++)for(int seed=0;seed<=23;seed+=23){
  int rows=shapes[shape][0],cols=shapes[shape][1],nt=tokens[ti];size_t blocks=(size_t)rows*cols/32,nx=(size_t)nt*cols,ny=(size_t)nt*rows;
  uint8_t*w=malloc(blocks*34);float*x=malloc(nx*4),*out=malloc(ny*4);assert(w&&x&&out);
  for(size_t j=0;j<blocks;j++){uint16_t d=(uint16_t)(0x2c00+((j+seed)%17)*37);memcpy(w+j*34,&d,2);for(int i=0;i<32;i++)w[j*34+2+i]=(uint8_t)((int)((j*17+i*13+seed)%255)-127);}
  for(size_t i=0;i<nx;i++)x[i]=(float)((int)((i*37+seed+11)%257)-128)/259.f;
  struct ggml_init_params ip={1024*1024,NULL,true};struct ggml_context*c=ggml_init(ip);assert(c);struct ggml_tensor*gx=ggml_new_tensor_2d(c,GGML_TYPE_F32,cols,nt),*gw=ggml_new_tensor_2d(c,GGML_TYPE_Q8_0,cols,rows),*gy=ggml_mul_mat(c,gw,gx);struct ggml_cgraph*g=ggml_new_graph_custom(c,16,false);ggml_build_forward_expand(g,gy);ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(c,b);assert(buf);ggml_backend_tensor_set(gx,x,0,nx*4);ggml_backend_tensor_set(gw,w,0,blocks*34);assert(ggml_backend_graph_compute(b,g)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(gy,out,0,ny*4);printf("{%d,%d,%d,%d,UINT64_C(0x%016llx)},\n",rows,cols,nt,seed,(unsigned long long)hash(out,ny));ggml_backend_buffer_free(buf);ggml_free(c);free(w);free(x);free(out);
 }ggml_backend_free(b);return 0;
}
