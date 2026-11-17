#ifndef BN_QUANT_INT8_MMA_CUDA_CUH
#define BN_QUANT_INT8_MMA_CUDA_CUH
#include <stdint.h>
/* Shared integer-only dot32 helper. Original lane/layout mapping is retained. */
static __device__ void cuda_bf16_mma_dot32(const int8_t a[16][272], const int8_t b[64][272],
                                           int warp, int lane, int g, int c[4]) {
#if __CUDA_ARCH__ >= 800
    int av[4], bv[2];
    unsigned ap = (unsigned)__cvta_generic_to_shared(&a[lane % 16][g * 32 + (lane / 16) * 16]);
    unsigned bp =
        (unsigned)__cvta_generic_to_shared(&b[warp * 8 + lane % 8][g * 32 + ((lane / 8) & 1) * 16]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(av[0]), "=r"(av[1]), "=r"(av[2]), "=r"(av[3])
                 : "r"(ap));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
                 : "=r"(bv[0]), "=r"(bv[1])
                 : "r"(bp));
    c[0] = c[1] = c[2] = c[3] = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                 "{%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bv[0]), "r"(bv[1]));
#else
    /* This branch keeps lower-architecture compilation valid. The host routes
     * pre-SM80 devices through the existing DP4A kernels. */
    for (int l = 0; l < 4; ++l) {
        int row = lane / 4 + (l / 2) * 8, token = warp * 8 + (lane % 4) * 2 + (l & 1);
        c[l] = 0;
        for (int k = 0; k < 32; ++k)
            c[l] += (int)a[row][g * 32 + k] * (int)b[token][g * 32 + k];
    }
#endif
}
#endif
