#ifndef BN_QUANT_IQ4NL_CUDA_CUH
#define BN_QUANT_IQ4NL_CUDA_CUH

/* Eight prompt tokens share each decoded IQ4_NL codebook block. Integer
 * dot products stay exact; each token retains the ordered Stream-K FP32
 * accumulation used by the one-token kernel. */
static __global__ void iq4nl_mmq_ordered_t8_kernel(
        float *out, const BnBlockIQ4NL *weights,
        const BnCudaBlockQ8MmqF32 *input, int rows, int cols,
        int n_tokens, int width, int grid) {
    int lane = threadIdx.x & 7;
    int row = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    int token0 = blockIdx.y * 8;
    if (row >= rows) return;
    unsigned mask = __activemask();
    int groups = cols / 32, blocks = cols / 256;
    int token_tiles = (n_tokens - 1) / width + 1;
    int64_t tile = (int64_t)(row / 128) * token_tiles + token0 / width;
    int64_t tile_begin = tile * blocks;
    int64_t total = (int64_t)((rows - 1) / 128 + 1) * token_tiles * blocks;
    int first = (int)(((tile_begin + 1) * grid + total - 1) / total) - 1;
    int last = (int)(((tile_begin + blocks) * grid + total - 1) / total) - 1;
    float tail[8] = {0}, prefix[8] = {0};
    int have_tail = 0;
    for (int bid = last; bid >= first; bid--) {
        int64_t raw_begin = (int64_t)bid * total / grid - tile_begin;
        int64_t raw_end = (int64_t)(bid + 1) * total / grid - tile_begin;
        int begin = raw_begin < 0 ? 0 : (int)raw_begin;
        int end = raw_end > blocks ? blocks : (int)raw_end;
        if (begin >= end) continue;
        float acc[8] = {0};
        for (int b = begin; b < end; b++) {
            int group = b * 8 + lane;
            const BnBlockIQ4NL *w = weights + (size_t)row * groups + group;
            float wd = cuda_fp16_to_fp32(w->d);
            uint32_t packed[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t qv = 0;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    int at = j * 4 + k;
                    int q = (w->qs[at & 15] >> (4 * (at >> 4))) & 15;
                    qv |= (uint32_t)(uint8_t)bn_kvalues_iq4nl[q] << (8 * k);
                }
                packed[j] = qv;
            }
#pragma unroll
            for (int t = 0; t < 8; t++) {
                int token = token0 + t;
                if (token >= n_tokens) continue;
                const BnCudaBlockQ8MmqF32 *x =
                    input + (size_t)token * groups + group;
                int dot = 0;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    uint32_t xv;
                    memcpy(&xv, x->qs + 4 * j, sizeof(xv));
                    dot = cuda_dp4a_i32((int)packed[j], (int)xv, dot);
                }
                float xd = x->d;
#pragma unroll
                for (int l = 0; l < 8; l++) {
                    int d = __shfl_sync(mask, dot, l, 8);
                    float dw = __shfl_sync(mask, wd, l, 8);
                    float dx = __shfl_sync(mask, xd, l, 8);
                    acc[t] = fmaf((float)d * dw, dx, acc[t]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < 8; t++) {
            if (!have_tail) tail[t] = acc[t];
            else prefix[t] += acc[t];
        }
        have_tail = 1;
    }
    if (lane == 0)
#pragma unroll
        for (int t = 0; t < 8; t++)
            if (token0 + t < n_tokens)
                out[(size_t)(token0 + t) * rows + row] = tail[t] + prefix[t];
}

#endif
