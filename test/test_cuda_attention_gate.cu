/* Frozen original SM120 compiled cubin oracle; see reference_cuda_attention_gate.cu. */
#include "gpu_cuda_attention_gate.cuh"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
int main(){cudaDeviceProp prop;assert(cudaGetDeviceProperties(&prop,0)==cudaSuccess);if(prop.major!=12||prop.minor!=0){puts("CUDA compiled attention gate skipped: SM120 reference contract");return 0;}const int n=2097152;unsigned short *x=(unsigned short*)malloc(n*2),*g=(unsigned short*)malloc(n*2),*out=(unsigned short*)malloc((n+2)*2);assert(x&&g&&out);__nv_bfloat16 *dx,*dg,*dy;assert(cudaMalloc(&dx,n*2)==0&&cudaMalloc(&dg,n*2)==0&&cudaMalloc(&dy,(n+2)*2)==0);
const uint64_t expected[]={UINT64_C(4305463184359221387),UINT64_C(17169930474415092949)};
for(int seed=1;seed<=2;seed++){for(int i=0;i<n;i++){x[i]=(unsigned short)((unsigned)i*37+(unsigned)(i/17)*seed);g[i]=(unsigned short)((unsigned)i*131+(unsigned)(i/31)*seed);if((x[i]&0x7f80)==0x7f80)x[i]&=0xff7f;if((g[i]&0x7f80)==0x7f80)g[i]&=0xff7f;}for(int i=0;i<n+2;i++)out[i]=0x5a5a;
assert(cudaMemcpy(dx,x,n*2,cudaMemcpyHostToDevice)==0&&cudaMemcpy(dg,g,n*2,cudaMemcpyHostToDevice)==0&&cudaMemcpy(dy,out,(n+2)*2,cudaMemcpyHostToDevice)==0);
bn_cuda_attention_gate::apply<true><<<(n+255)/256,256>>>(dy+1,dx,dg,n);assert(cudaMemcpy(out,dy,(n+2)*2,cudaMemcpyDeviceToHost)==0);assert(out[0]==0x5a5a&&out[n+1]==0x5a5a);uint64_t h=UINT64_C(14695981039346656037);for(int i=0;i<n;i++){h=(h^(out[i+1]&255))*UINT64_C(1099511628211);h=(h^(out[i+1]>>8))*UINT64_C(1099511628211);}assert(h==expected[seed-1]);}
cudaFree(dx);cudaFree(dg);cudaFree(dy);free(x);free(g);free(out);puts("CUDA compiled attention gate: 4194304 original-reference finite BF16 values... PASSED");}
