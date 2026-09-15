#ifndef BN_CUDA_Q451_ROUTED_REFERENCE_H
#define BN_CUDA_Q451_ROUTED_REFERENCE_H
#include "cuda_routed_reference.h"
/* Original llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e, SM120.
 * Q4_K gate/up, SiLU, Q5_1 down, ordered route-weighted reduction.
 * All cases use 16 experts and K=8; 84,480 baseline output words. */
static const BnCudaRoutedReference cuda_q451_routed_reference[] = {
    {256,512,16,1,0,UINT64_C(0x7022a9dcd1bb9155)},
    {256,512,16,2,0,UINT64_C(0x3c74895cfc7d5b34)},
    {256,512,16,4,0,UINT64_C(0xf865daf72c6d3e4f)},
    {256,512,16,8,0,UINT64_C(0xcb33f8559b9a7537)},
    {2560,640,16,1,0,UINT64_C(0x416736cd856438b7)},
    {2560,640,16,2,0,UINT64_C(0x3898868be5ca8f1d)},
    {2560,640,16,4,0,UINT64_C(0x77cea3611932ebbe)},
    {2560,640,16,8,0,UINT64_C(0x8aaf24470511e88f)},
    {256,512,16,1,23,UINT64_C(0x934c49f709bf9a4e)},
    {256,512,16,2,23,UINT64_C(0xe56086db6ab4cd77)},
    {256,512,16,4,23,UINT64_C(0x70348eec35c3f442)},
    {256,512,16,8,23,UINT64_C(0x2e7ac64550759e5b)},
    {2560,640,16,1,23,UINT64_C(0x62bdb2bc598691c6)},
    {2560,640,16,2,23,UINT64_C(0x2e55676604237278)},
    {2560,640,16,4,23,UINT64_C(0x3c505b9496349e15)},
    {2560,640,16,8,23,UINT64_C(0x42e702076aae4442)},
};
#endif
