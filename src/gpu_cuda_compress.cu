/* Source stateful compressor, exllamav3 bd1f5191 MIT. No ATen or hidden state. */
#include "gpu_cuda_compress.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
namespace bn_cuda_compress_detail {
#include "gpu_cuda_compress_kernels.cuh"
}
int bn_cuda_compress(cudaStream_t s,void**a,int R,int pos,int m,int D,int rd,int ring_rows,int split,int relative,float eps) {
    using namespace bn_cuda_compress_detail;
    int nw=(pos+R)/m-pos/m,threads=(D+31)/32*32;
    if(nw) {
        dsv4_compress_windows_kernel<<<nw,threads,(D+threads/32)*sizeof(float),s>>>(
            (const half*)a[0],(const half*)a[1],(const half*)a[2],(const half*)a[3],nullptr,
            (const float*)a[4],(const half*)a[5],eps,(const float*)a[6],(half*)a[7],(half*)a[8],
            nullptr,pos,R,m,ring_rows,1,D,D,rd,split,false,nullptr,0,0,0,0,nullptr,0,0,relative);
        if(cudaGetLastError()!=cudaSuccess)return -1;
    }
    int j0=R>ring_rows?R-ring_rows:0;size_t total=(size_t)(R-j0)*D;
    dsv4_compress_store_kernel<<<(total+255)/256,256,0,s>>>((const half*)a[0],(const half*)a[1],
        (half*)a[2],(half*)a[3],nullptr,pos,R,ring_rows,D,j0,nullptr,0);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
