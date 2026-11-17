#ifndef BN_QUANT_FP16_CUDA_CUH
#define BN_QUANT_FP16_CUDA_CUH
#include <cuda_fp16.h>
#include <stdint.h>
static __device__ float cuda_fp16_to_fp32(uint16_t h) {
    /* CUDA conversion preserves all finite values, signed zeros and infinities.
     * Preserve the old NaN sign/payload rather than canonicalizing it. */
    if ((h & 0x7fffu) > 0x7c00u) {
        uint32_t bits =
            (((uint32_t)h & 0x8000u) << 16) | 0x7f800000u | (((uint32_t)h & 0x03ffu) << 13);
        return __uint_as_float(bits);
    }
    return __half2float(__ushort_as_half(h));
}

#endif
