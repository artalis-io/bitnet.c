/* Source-shaped small-batch EXL3 arithmetic, exllamav3 bd1f5191, MIT.
 * Request storage and borrowed weights belong to the native backend. */
#include "gpu_cuda_exl3_decode.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <mutex>
#include <tuple>
#include <cublas_v2.h>
#include <cooperative_groups.h>
namespace bn_cuda_decode_detail {
#define EXL3_MOE_COOP_DEFINE_ROT
#include "cuda_exl3_fused/quant/exl3_moe_coop_kernel.cuh"
using namespace exl3_moe_coop_ns;
static void *kernel_a(bool wide) {return wide?(void*)exl3_moe_coop_a_kernel<3,2,true>:(void*)exl3_moe_coop_a_kernel<3,2,false>;}
static void *kernel_b(bool wide) {return wide?(void*)exl3_moe_coop_b_kernel<3,2,true>:(void*)exl3_moe_coop_b_kernel<3,2,false>;}
/* Metadata only; no activation arithmetic or host transfer. Unlisted experts
 * are masked, so a caller must bind every selected nonzero-weight expert for
 * a complete routed FFN. No cross-window reduction is supported here. */
__global__ void bind(BnCudaDecodeEXL3Pointers p,int64_t*tables,int64_t*sel,
                    const int*ids,int A,int E) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    for(int j=i;j<9*E;j+=gridDim.x*blockDim.x) {
        int e=j%E,q=j/E,pr=q/3,part=q%3,b=0;
        while(b<p.count&&p.experts[b]!=e)++b;
        const void *v=b==p.count?nullptr:part==0?p.packed[pr][b]:part==1?p.pre[pr][b]:p.post[pr][b];
        tables[j]=(int64_t)v;
    }
    if(i<A) {int e=ids[i],b=0;while(b<p.count&&p.experts[b]!=e)++b;sel[i]=b<p.count?e:E;}
}
}
int bn_cuda_exl3_decode_info(int R,int D,int H,int K,int *out) {
    if(!out||R<1||R>8||K<1||K>32||D<128||H<128||D>32768||H>65536||D%128||H%128||D>INT32_MAX/H)return -1;
    int device;cudaDeviceProp dev={};if(cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&dev,device)!=cudaSuccess||dev.major<8)return -1;
    int S=R*K,sm=dev.multiProcessorCount;
    bool wa=dev.major<10||D/16>=256||(D/16>=128&&S>=32);
    bool wb=dev.major<10||H/16>=256||(H/16>=128&&S>=32);
    int ba=S*2*(H/(wa?128:32)),bb=S*(D/(wb?128:32));
    int sa=std::max(1,std::min(std::min((sm+ba-1)/ba,std::max(D/16/64,1)),8/R));
    int sb=std::max(1,std::min(std::min((sm+bb-1)/bb,std::max(H/16/64,1)),8/R));
    int ma=bn_cuda_decode_detail::exl3_moe_coop_ns::smem_a_bytes<3>(D);
    int mb=bn_cuda_decode_detail::exl3_moe_coop_ns::smem_b_bytes<3>();
    if(ma>dev.sharedMemPerBlockOptin||mb>dev.sharedMemPerBlockOptin)return -1;
    void *ka=bn_cuda_decode_detail::kernel_a(wa),*kb=bn_cuda_decode_detail::kernel_b(wb);
    /* Different request widths must never lower a shared kernel's limit. */
    int ceiling=(int)dev.sharedMemPerBlockOptin;
    if(ma>49152&&cudaFuncSetAttribute(ka,cudaFuncAttributeMaxDynamicSharedMemorySize,ceiling)!=cudaSuccess)return -1;
    if(mb>49152&&cudaFuncSetAttribute(kb,cudaFuncAttributeMaxDynamicSharedMemorySize,ceiling)!=cudaSuccess)return -1;
    int info[]={wa,wb,sa,sb,ma,mb,sm};std::copy(info,info+7,out);return 0;
}
int bn_cuda_exl3_decode_launch(cudaStream_t stream,const BnCudaDecodeEXL3Pointers*w,
    const BnCudaDecodeEXL3IO*io,int R,int D,int H,int E,int K,float limit) {
    using namespace bn_cuda_decode_detail;
    int info[7];if(!w||!io||bn_cuda_exl3_decode_info(R,D,H,K,info))return -1;
    bind<<<(9*E+255)/256,256,0,stream>>>(*w,(int64_t*)io->tables,(int64_t*)io->indices64,(const int*)io->indices,R*K,E);
    if(cudaGetLastError()!=cudaSuccess)return -1;
    MoeCoopParams p={};p.x=(const half*)io->x;p.x_stride=D;p.sel=(const int64_t*)io->indices64;p.rw=(const half*)io->weights;
    p.bsz=R;p.topk=K;p.H=p.Hi=p.Ho=p.H_out=D;p.I=H;p.min_expert=0;p.max_expert=E;p.n_local=E;
    const int64_t*t=(const int64_t*)io->tables;
    p.g_trellis=t;p.g_suh=t+E;p.g_svh=t+2*E;p.u_trellis=t+3*E;p.u_suh=t+4*E;p.u_svh=t+5*E;p.d_trellis=t+6*E;p.d_suh=t+7*E;p.d_svh=t+8*E;
    p.act=MOE_COOP_ACT_SILU;p.act_limit=limit;p.gated=true;p.gu_f32=false;
    p.had_g=(half*)io->had_g;p.had_u=(half*)io->had_u;p.gu_g=io->gate;p.gu_u=io->up;p.act_out=(half*)io->act;p.d_out=(float*)io->down;
    p.slots_max=8*K;p.rows_max=8;p.ctr_a_len=p.slots_max*(H/128);p.ctr_b_len=8*(D/128);
    p.ctr_a=(int*)io->counters;p.ctr_b=p.ctr_a+p.ctr_a_len;p.runs=p.ctr_b+p.ctr_b_len;
    p.out=(float*)io->out;p.out_stride=D;p.a_global=R>1;p.ksplit_a=info[2];p.ksplit_b=info[3];
    if(p.a_global){int items=R*K*(D/128)*2;exl3_moe_coop_rot_kernel<<<(items+15)/16,512,0,stream>>>(p);if(cudaGetLastError()!=cudaSuccess)return -1;}
    void *args[]={&p};int ga=R*K*2*(H/(info[0]?128:32))*info[2],gb=R*K*(D/(info[1]?128:32))*info[3];
    if(cudaLaunchKernel(kernel_a(info[0]),dim3(ga),dim3(512),args,info[4],stream)!=cudaSuccess)return -1;
    if(cudaLaunchKernel(kernel_b(info[1]),dim3(gb),dim3(512),args,info[5],stream)!=cudaSuccess)return -1;
    return cudaGetLastError()==cudaSuccess?0:-1;
}
