#ifndef BN_CUDA_BF16_SILU_PROFILES_REFERENCE_H
#define BN_CUDA_BF16_SILU_PROFILES_REFERENCE_H
#include <stdint.h>
/* Original pinned vLLM fused op and eager Torch BF16 op; canonical NaN hashes. */
#define BN_SILU_FUSED_FNV UINT64_C(0x4cb738d360ae9406)
#define BN_SILU_TWO_FNV UINT64_C(0x98e3c5268aa898ba)
#define BN_SILU_SCALED_FNV UINT64_C(0x99fd218d892336c5)
#endif
