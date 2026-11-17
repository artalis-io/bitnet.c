#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "quant/fp16_cuda.cuh"

/* Integer IEEE-754 oracle, including the original NaN sign/payload contract. */
static uint32_t expected_bits(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 31u, fraction = h & 1023u;
    if (exponent == 31)
        return sign | 0x7f800000u | (fraction << 13);
    if (exponent)
        return sign | ((exponent + 112u) << 23) | (fraction << 13);
    if (!fraction) return sign;
    int shift = 0;
    while (fraction < 1024u) {
        fraction <<= 1;
        shift++;
    }
    return sign | ((uint32_t)(113 - shift) << 23) | ((fraction & 1023u) << 13);
}

static __global__ void conversion_bits(uint32_t *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536)
        out[i] = __float_as_uint(cuda_fp16_to_fp32((uint16_t)i));
}

int main(void) {
    uint32_t *out = (uint32_t *)malloc(65536 * sizeof(*out)), *device = NULL;
    assert(out);
    assert(cudaMalloc(&device, 65536 * sizeof(*device)) == cudaSuccess);
    conversion_bits<<<256, 256>>>(device);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaMemcpy(out, device, 65536 * sizeof(*out), cudaMemcpyDeviceToHost) == cudaSuccess);
    unsigned different = 0;
    for (unsigned i = 0; i < 65536; i++) {
        uint32_t expected = expected_bits((uint16_t)i);
        if (out[i] != expected) {
            if (different < 8)
                fprintf(stderr, "FP16 conversion %04x: %08x != %08x\n", i, out[i], expected);
            different++;
        }
    }
    assert(cudaFree(device) == cudaSuccess);
    free(out);
    assert(!different);
    puts("CUDA FP16 conversion PASSED: 65536 exact bit patterns");
    return 0;
}
