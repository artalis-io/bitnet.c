/* Original SM120 paged/nonpaged score captures, raw future mask. */
#ifndef BN_CUDA_GLM_PAGED_SCORE_REFERENCE_H
#define BN_CUDA_GLM_PAGED_SCORE_REFERENCE_H
typedef struct {int tokens,position,seed;uint64_t hash;} BnGLMPagedScoreReference;
static const BnGLMPagedScoreReference glm_paged_score_reference[]={
{1,0,1,UINT64_C(0xcbf29ce484222325)},
{2,1,2,UINT64_C(0xcbf29ce484222325)},
{1,3,3,UINT64_C(0x643009f9b67915a7)},
{2,254,4,UINT64_C(0xa17fbcc7ec75e577)},
{3,2047,5,UINT64_C(0xa57a32cf4f5b1efa)},
{4,4098,6,UINT64_C(0xb83efb4b223a414b)},
{7,8191,7,UINT64_C(0xe281f058fee0ee85)},
{17,262143,8,UINT64_C(0x0d2b670b4c827217)},
};
#endif
