"""Freeze compiled F32 SiLU from the original independently inspected cubin.
Usage: extract_compiled_silu_reference.py CUBIN OUTPUT.h
Pin CUDA_VISIBLE_DEVICES. No native arithmetic supplies expected outputs.
The sweep covers every BF16 gate encoding and eight finite up values; NaN
payloads are canonicalized when hashing. Two full-model edge pairs follow.
"""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile
cubin=Path(sys.argv[1]);output=Path(sys.argv[2])
assert hashlib.sha256(cubin.read_bytes()).hexdigest()=='86c146381f1974d89ec5359a687e01add3c850bd851452bd855488071a647c62'
source=r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int main(){assert(cudaSetDevice(0)==cudaSuccess&&cuInit(0)==CUDA_SUCCESS);
const int sweep=65536*8,n=sweep+16;const size_t bytes=(size_t)n*2;
uint16_t *g=(uint16_t*)malloc(bytes),*u=(uint16_t*)malloc(bytes),*out=(uint16_t*)malloc(bytes);assert(g&&u&&out);
const uint16_t ups[]={0xc000,0xbf80,0xbf00,0,0x3f00,0x3f80,0x4000,0x4049};
for(int i=0;i<sweep;i++){g[i]=(uint16_t)(i/8);u[i]=ups[i%8];}
for(int i=sweep;i<n;i++){g[i]=14825;u[i]=(i&1)?48763:48379;}
void*d[3];for(int i=0;i<3;i++)assert(cudaMalloc(&d[i],bytes)==cudaSuccess);
assert(cudaMemcpy(d[0],g,bytes,cudaMemcpyHostToDevice)==cudaSuccess&&cudaMemcpy(d[1],u,bytes,cudaMemcpyHostToDevice)==cudaSuccess);
CUmodule mod;CUfunction fn;assert(cuModuleLoad(&mod,FILE)==CUDA_SUCCESS&&cuModuleGetFunction(&fn,mod,"triton_poi_fused_cat_mul_silu_slice_2")==CUDA_SUCCESS);
int count=n;void*scratch=NULL;void*args[]={&d[0],&d[1],&d[2],&count,&scratch,&scratch};
assert(cuLaunchKernel(fn,(n+511)/512,1,1,256,1,1,0,0,args,NULL)==CUDA_SUCCESS&&cudaMemcpy(out,d[2],bytes,cudaMemcpyDeviceToHost)==cudaSuccess);
uint64_t hash=UINT64_C(14695981039346656037);for(int i=0;i<sweep;i++){uint16_t v=out[i];if((v&0x7f80)==0x7f80&&(v&127))v=0x7fc0;hash=(hash^v)*UINT64_C(1099511628211);}
printf("/* Original compiled F32 SiLU; all BF16 gates x eight finite up values. */\n#define BN_SILU_COMPILED_FNV UINT64_C(%llu)\n",(unsigned long long)hash);
printf("static const uint16_t cuda_compiled_silu_edges[2][3]={{14825,48379,%u},{14825,48763,%u}};\n",out[sweep],out[sweep+1]);
for(int i=0;i<3;i++)cudaFree(d[i]);free(g);free(u);free(out);return 0;}
'''
source=source.replace('FILE',json.dumps(str(cubin)))
with tempfile.TemporaryDirectory()as tmp:
 cpp=Path(tmp)/'reference.cu';exe=Path(tmp)/'reference';cpp.write_text(source)
 subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++11','-gencode=arch=compute_120a,code=sm_120a',str(cpp),'-lcuda','-o',str(exe)],check=True)
 result=subprocess.check_output([str(exe)],text=True)
output.write_text('/* Original cubin SHA-256: '+hashlib.sha256(cubin.read_bytes()).hexdigest()+' */\n'+result)
print(output,hashlib.sha256(output.read_bytes()).hexdigest())
