#ifndef BN_GPU_CUDA_EXL3_FUSED_H
#define BN_GPU_CUDA_EXL3_FUSED_H
#include <cuda_runtime.h>
struct BnCudaFusedEXL3Pointers {
    const void *packed[3][32],*pre[3][32],*post[3][32];
    int experts[32],count;
};
/* Backend-private launch parameters, copied by value into a bounded kernel
 * parameter block. These are device addresses, never model-owned handles. */
struct BnCudaFusedEXL3Item {
    BnCudaFusedEXL3Pointers pointers;
    const void *x,*counts,*order,*weights,*base;
    void *state_g,*state_u,*intermediate_g,*intermediate_u,*slots,*locks;
    int tokens,dim,hidden,experts,topk,capacity,lo,hi;
    int tile_m,tile_n,width,groups,slot_rows;
    float limit;
};
enum { BN_CUDA_FUSED_EXL3_GROUP_MAX = 8 };
struct BnCudaFusedEXL3Batch {
    BnCudaFusedEXL3Item items[BN_CUDA_FUSED_EXL3_GROUP_MAX];
    int count;
};
static_assert(sizeof(BnCudaFusedEXL3Batch)<32764,"CUDA parameter block bound");
int bn_cuda_exl3_fused_prepare(int*smem,int*blocks,int*joint_blocks);
size_t bn_cuda_exl3_fused_lock_bytes();
int bn_cuda_exl3_fused_launch(cudaStream_t,const BnCudaFusedEXL3Item*,int);
int bn_cuda_exl3_fused_group_launch(cudaStream_t,const BnCudaFusedEXL3Batch*,int);
#endif
