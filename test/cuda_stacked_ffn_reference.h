#ifndef BN_CUDA_STACKED_FFN_REFERENCE_H
#define BN_CUDA_STACKED_FFN_REFERENCE_H
#include <stdint.h>
/* Independent pinned Torch merged BF16 GEMM + compiled SiLU/product oracle. */
static const struct {int tokens;uint64_t hash;} cuda_stacked_ffn_cases[] = {
    {3,UINT64_C(5083505245648340361)},
    {24,UINT64_C(13906957959448477205)},
    {32,UINT64_C(3750649654227034788)},
};
#endif
