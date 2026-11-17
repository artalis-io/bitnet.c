#include "gpu_cuda_native_rope.h"
#include <cmath>

namespace bn_cuda_frequency_detail {
static __global__ void frequency(float*out,float*negative,int dim,int yarn,float base,float factor,float start,float scale) {
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=dim/2)return;
    float exponent=__fmul_rn((float)(2*i),__fdiv_rn(1.f,(float)dim));
    float power=powf(base,exponent),extra=__fdiv_rn(1.f,power),value=extra;
    if(yarn) {
        float ramp=__fmul_rn(__fsub_rn((float)i,start),scale);
        float extrap=__fsub_rn(1.f,fminf(1.f,fmaxf(0.f,ramp)));
        float interp=__fdiv_rn(1.f,__fmul_rn(factor,power));
        value=__fadd_rn(__fmul_rn(interp,__fsub_rn(1.f,extrap)),__fmul_rn(extra,extrap));
    }
    out[i]=value;negative[i]=-value;
}
}
int bn_cuda_native_frequency(cudaStream_t stream,float*out,float*negative,int dim,int yarn,float base,float factor,float start,float scale) {
    bn_cuda_frequency_detail::frequency<<<(dim/2+127)/128,128,0,stream>>>(out,negative,dim,yarn,base,factor,start,scale);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
