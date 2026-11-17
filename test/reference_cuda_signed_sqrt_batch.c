/* Independent GGML df18bee11 CUDA row-geometry fixtures, 188 SM device. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t hash(const float*x,size_t n){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++){uint32_t u;memcpy(&u,x+i,4);h=(h^u)*UINT64_C(1099511628211);}return h;}
int main(void){ggml_backend_t b=ggml_backend_cuda_init(0);assert(b);const int ts[]={1,8,93,94,95,512,1047},ds[]={256,2560};for(int di=0;di<2;di++)for(int ti=0;ti<7;ti++)for(int seed=0;seed<=23;seed+=23){int dim=ds[di],nt=ts[ti];size_t wide=(size_t)dim*4,n=nt*wide;float*k=malloc(n*4),*q=malloc(n*4),*v=malloc((size_t)nt*dim*4),*out=malloc(3*(wide+4)*4);assert(k&&q&&v&&out);for(size_t i=0;i<n;i++){k[i]=(float)((int)((i*37+seed)%251)-125)/193.f;q[i]=(float)((int)((i*71+seed+19)%257)-128)/239.f;}for(size_t i=0;i<(size_t)nt*dim;i++)v[i]=(float)((int)((i*29+seed+11)%263)-131)/127.f;struct ggml_init_params p={1024*1024,NULL,true};struct ggml_context*c=ggml_init(p);struct ggml_tensor*tk=ggml_new_tensor_3d(c,GGML_TYPE_F32,dim,4,nt),*tq=ggml_new_tensor_3d(c,GGML_TYPE_F32,dim,4,nt),*tv=ggml_new_tensor_3d(c,GGML_TYPE_F32,dim,1,nt);struct ggml_tensor*s=ggml_scale(c,ggml_sum_rows(c,ggml_mul(c,tk,tq)),1.f/sqrtf((float)dim));struct ggml_tensor*mag=ggml_sqrt(c,ggml_clamp(c,ggml_abs(c,s),1e-6f,INFINITY));struct ggml_tensor*g=ggml_sigmoid(c,ggml_mul(c,ggml_sgn(c,s),mag)),*gated=ggml_mul(c,ggml_repeat_4d(c,tv,dim,4,nt,1),g);struct ggml_cgraph*graph=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(graph,gated);ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(c,b);assert(buf);ggml_backend_tensor_set(tk,k,0,n*4);ggml_backend_tensor_set(tq,q,0,n*4);ggml_backend_tensor_set(tv,v,0,(size_t)nt*dim*4);assert(ggml_backend_graph_compute(b,graph)==GGML_STATUS_SUCCESS);int rows[]={0,nt/2,nt-1};for(int j=0;j<3;j++){float*o=out+j*(wide+4);ggml_backend_tensor_get(g,o,(size_t)rows[j]*16,16);ggml_backend_tensor_get(gated,o+4,(size_t)rows[j]*wide*4,wide*4);}printf("{%d,%d,%d,UINT64_C(0x%016llx)},\n",dim,nt,seed,(unsigned long long)hash(out,3*(wide+4)));ggml_backend_buffer_free(buf);ggml_free(c);free(k);free(q);free(v);free(out);}ggml_backend_free(b);return 0;}
