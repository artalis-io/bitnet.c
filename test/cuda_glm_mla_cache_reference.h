/* Installed vLLM concat_and_cache_mla, NoPE fp8_ds_mla, captured twice. */
#ifndef BN_CUDA_GLM_MLA_CACHE_REFERENCE_H
#define BN_CUDA_GLM_MLA_CACHE_REFERENCE_H
typedef struct {int tokens,seed,blocks;uint64_t digest;} BnGLMMLACacheReference;
static const BnGLMMLACacheReference glm_mla_cache_reference[] = {
 {0,0,2,UINT64_C(0x01f1d91bc1b98b25)},
 {0,1,2,UINT64_C(0x01f1d91bc1b98b25)},
 {1,0,3,UINT64_C(0x498d2647eeb7bd35)},
 {1,1,3,UINT64_C(0x04816b31e8ba05b5)},
 {2,0,3,UINT64_C(0xbb7cc137b3f02a05)},
 {2,1,3,UINT64_C(0x9b4abe0c68f46505)},
 {63,0,3,UINT64_C(0x610270ef323a730e)},
 {63,1,3,UINT64_C(0x162fb1be7db9f12a)},
 {64,0,3,UINT64_C(0x1e4ed94abe061155)},
 {64,1,3,UINT64_C(0x74b8a92e450be21a)},
 {65,0,4,UINT64_C(0xc344acc09fcd5c21)},
 {65,1,4,UINT64_C(0x6971005a8ebefaf5)},
 {129,0,5,UINT64_C(0xf998e4c5c3ae3d8a)},
 {129,1,5,UINT64_C(0x3e9a75585fc8f6e6)},
 {1025,0,19,UINT64_C(0x003ef300c114c7cc)},
 {1025,1,19,UINT64_C(0x70c4aa529d0591f3)},
 {4105,0,67,UINT64_C(0xcb0f1bc9daab9e1f)},
 {4105,1,67,UINT64_C(0xa4a9fb70622fd1a2)},
};
#endif
