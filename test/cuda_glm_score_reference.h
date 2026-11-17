/* Original DeepGEMM SM120, masked causal scores. */
#ifndef BN_CUDA_GLM_SCORE_REFERENCE_H
#define BN_CUDA_GLM_SCORE_REFERENCE_H
typedef struct {int tokens,position,seed;uint64_t hash;} BnGLMScoreReference;
static const BnGLMScoreReference glm_score_reference[]={
{1,0,1,UINT64_C(0xcbf29ce484222325)},
{2,1,2,UINT64_C(0xcbf29ce484222325)},
{1,3,3,UINT64_C(0x643009f9b67915a7)},
{2,62,4,UINT64_C(0x398c849f74440d2e)},
{17,0,5,UINT64_C(0xb1cf23d3866ee64e)},
{65,0,6,UINT64_C(0x630f97369fa69d3f)},
{129,3,7,UINT64_C(0x75e4db82b19a7bbb)},
{17,2047,8,UINT64_C(0x9fc68c9d954e43eb)},
{65,4098,9,UINT64_C(0x9a5c8deb87be1fde)},
{129,8191,10,UINT64_C(0xc4ff3731e2b799cf)},
};
#endif
