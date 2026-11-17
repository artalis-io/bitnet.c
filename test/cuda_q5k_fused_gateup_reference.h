#ifndef BN_CUDA_Q5K_FUSED_GATEUP_REFERENCE_H
#define BN_CUDA_Q5K_FUSED_GATEUP_REFERENCE_H
/* Independent pinned GGML CUDA df18bee11; regenerate with
 * reference_cuda_q5k_fused_gateup.c. Original fixtures are unchanged. */
typedef struct { int rows, cols, seed; uint64_t hash; } BnCudaQ5KFusedReference;
static const BnCudaQ5KFusedReference cuda_q5k_fused_reference[] = {
{37,256,0,UINT64_C(0x86956625518369eb)},
{37,256,23,UINT64_C(0xca625f03268af80f)},
{640,256,0,UINT64_C(0x49522e235d7e4801)},
{640,256,23,UINT64_C(0x871a73fe4fd002a0)},
{37,512,0,UINT64_C(0x99c6c30bb37b1fd9)},
{37,512,23,UINT64_C(0xe7912112eebaa59a)},
{640,512,0,UINT64_C(0x98f766971250c654)},
{640,512,23,UINT64_C(0xbfb911ffee2d3298)},
{37,2560,0,UINT64_C(0xb8e6fd97c774bc16)},
{37,2560,23,UINT64_C(0xf8446c8d86f3d643)},
{640,2560,0,UINT64_C(0x39dd2dbb123b4c17)},
{640,2560,23,UINT64_C(0x345b3a315a07f8e1)},
};
#endif
