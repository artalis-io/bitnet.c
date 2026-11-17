/* Independent pinned GGML df18bee11 CUDA, two logical Q8 projections. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void fill(uint8_t*w,size_t bytes,int seed){for(size_t b=0;b<bytes/34;b++){uint16_t d=(uint16_t)(0x1800+((b+seed)%17)*37);memcpy(w+b*34,&d,2);for(int i=0;i<32;i++)w[b*34+2+i]=(uint8_t)((int)((b*19+i*23+seed)%255)-127);}}
static uint64_t hash(const float*x,size_t n){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++){uint32_t u;memcpy(&u,x+i,4);h=(h^u)*UINT64_C(1099511628211);}return h;}
int main(void){
 const int shapes[][2]={{256,96},{2560,640}},tokens[]={1,2,3,4,8,9,16,17,33,129,257,513,1047};ggml_backend_t b=ggml_backend_cuda_init(0);assert(b);
 for(int shape=0;shape<2;shape++)for(size_t t=0;t<sizeof(tokens)/sizeof(tokens[0]);t++)for(int seed=0;seed<=23;seed+=23){int dim=shapes[shape][0],hidden=shapes[shape][1],nt=tokens[t];size_t bytes=(size_t)dim*hidden/32*34,nx=(size_t)nt*dim;
 uint8_t*wg=malloc(bytes),*wu=malloc(bytes),*wd=malloc(bytes);float*x=malloc(nx*4),*out=malloc(nx*4);assert(wg&&wu&&wd&&x&&out);fill(wg,bytes,seed);fill(wu,bytes,seed+13);fill(wd,bytes,seed+29);for(size_t i=0;i<nx;i++)x[i]=(float)((int)((i*37+seed+11)%251)-125)/259.0f;
 struct ggml_init_params p={1024*1024,NULL,true};struct ggml_context*c=ggml_init(p);assert(c);struct ggml_tensor*gx=ggml_new_tensor_2d(c,GGML_TYPE_F32,dim,nt),*gg=ggml_new_tensor_2d(c,GGML_TYPE_Q8_0,dim,hidden),*gu=ggml_new_tensor_2d(c,GGML_TYPE_Q8_0,dim,hidden),*gd=ggml_new_tensor_2d(c,GGML_TYPE_Q8_0,hidden,dim);
 struct ggml_tensor*gate=ggml_mul_mat(c,gg,gx),*up=ggml_mul_mat(c,gu,gx),*mid=ggml_swiglu_split(c,gate,up),*result=ggml_mul_mat(c,gd,mid),*half=ggml_scale(c,result,0.5f);struct ggml_cgraph*graph=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(graph,half);ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(c,b);assert(buf);ggml_backend_tensor_set(gx,x,0,nx*4);ggml_backend_tensor_set(gg,wg,0,bytes);ggml_backend_tensor_set(gu,wu,0,bytes);ggml_backend_tensor_set(gd,wd,0,bytes);assert(ggml_backend_graph_compute(b,graph)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(result,out,0,nx*4);uint64_t full=hash(out,nx);ggml_backend_tensor_get(half,out,0,nx*4);printf("{%d,%d,%d,%d,UINT64_C(0x%016llx),UINT64_C(0x%016llx)},\n",dim,hidden,nt,seed,(unsigned long long)full,(unsigned long long)hash(out,nx));ggml_backend_buffer_free(buf);ggml_free(c);free(wg);free(wu);free(wd);free(x);free(out);}
 ggml_backend_free(b);return 0;}
