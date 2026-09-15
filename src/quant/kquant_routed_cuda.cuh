#ifndef BN_QUANT_KQUANT_ROUTED_CUDA_CUH
#define BN_QUANT_KQUANT_ROUTED_CUDA_CUH
/* Format-local routed K-quant arithmetic. Input repeat is the route width
 * for shared gate/up inputs and one for per-slot down inputs. Keep MMVQ
 * type specialization: contraction changes its numerical behavior. */
static __global__ void kquant_routed_mmq_kernel(
    float *out, const void *weights, const BnCudaBlockQ8Mmq *input,
    int rows, int cols, int type, int nt, int jwidth, int grid, const int *ids, int experts, int k, int repeat) {
    int lane = threadIdx.x & 7;
    int row = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    int token = blockIdx.y;
    int expert = ids[token], rank = ids[nt*k+token];
    if (row >= rows) return;
    unsigned mask = __activemask();
    int groups = cols / 32, nb = cols / 256;
    int token_tiles = (nt - 1) / jwidth + 1;
    int64_t tile = ((int64_t)(row / 128) * experts + expert) * token_tiles + rank / jwidth;
    int64_t tile_begin = tile * nb;
    int64_t total = (int64_t)((rows - 1) / 128 + 1) * experts * token_tiles * nb;
    int first = (int)(((tile_begin + 1) * grid + total - 1) / total) - 1;
    int last = (int)(((tile_begin + nb) * grid + total - 1) / total) - 1;
    float tail = 0.0f, prefix = 0.0f;
    int have_tail = 0;
    for (int bid = last; bid >= first; bid--) {
        int64_t raw_begin = (int64_t)bid * total / grid - tile_begin;
        int64_t raw_end = (int64_t)(bid + 1) * total / grid - tile_begin;
        int begin = raw_begin < 0 ? 0 : (int)raw_begin;
        int end = raw_end > nb ? nb : (int)raw_end;
        if (begin >= end) continue;
        float acc = 0.0f;
        for (int base = begin * 8; base < end * 8; base += 8) {
            int g = base + lane, group = g & 7;
            size_t block = ((size_t)expert * rows + row) * nb + g / 8;
            int dot = 0;
            float wd = 0.0f, wm = 0.0f, xd = 0.0f, xs = 0.0f;
            if (type == BN_GGUF_TENSOR_Q6_K) {
                const BnBlockQ6K *w = (const BnBlockQ6K *)weights + block;
                const BnCudaBlockQ8MmqF32 *x = (const BnCudaBlockQ8MmqF32 *)input + (size_t)(token/repeat) * groups + g;
                int chunk = group / 4, segment = group & 3;
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    int part = 0;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int j = half * 16 + k * 4;
                        uint32_t lo, hi, xv;
                        memcpy(&lo, w->ql + chunk * 64 + (segment & 1) * 32 + j, 4);
                        memcpy(&hi, w->qh + chunk * 32 + j, 4);
                        memcpy(&xv, x->qs + j, 4);
                        uint32_t q = ((lo >> ((segment / 2) * 4)) & 0x0f0f0f0fu) |
                            (((hi >> (segment * 2)) & 0x03030303u) << 4);
                        /* Bytewise sign conversion: values 0..63 to -32..31. */
                        q = (q | 0x80808080u) - 0x20202020u;
                        q ^= 0x80808080u;
                        part = cuda_dp4a_i32((int)q, (int)xv, part);
                    }
                    dot += part * w->scales[group * 2 + half];
                }
                wd = cuda_fp16_to_fp32(w->d); xd = x->d;
            } else {
                const BnCudaBlockQ8Mmq *x = input + (size_t)(token/repeat) * groups + g;
                int sc, mn;
                if (type == BN_GGUF_TENSOR_Q4_K) {
                    const BnBlockQ4K *w = (const BnBlockQ4K *)weights + block;
                    cuda_kquant_group_scale_min(w->scales, group, &sc, &mn);
                    wd = cuda_fp16_to_fp32(w->d) * sc;
                    wm = cuda_fp16_to_fp32(w->dmin) * mn;
                    dot = cuda_q4k_dot_32(w->qs + (group / 2) * 32, x->qs, (group & 1) * 4);
                } else {
                    const BnBlockQ5K *w = (const BnBlockQ5K *)weights + block;
                    cuda_kquant_group_scale_min(w->scales, group, &sc, &mn);
                    wd = cuda_fp16_to_fp32(w->d) * sc;
                    wm = cuda_fp16_to_fp32(w->dmin) * mn;
#pragma unroll
                    for (int j = 0; j < 32; j += 4) {
                        uint32_t lo, hi, xv;
                        memcpy(&lo, w->qs + (group / 2) * 32 + j, 4);
                        memcpy(&hi, w->qh + j, 4);
                        memcpy(&xv, x->qs + j, 4);
                        uint32_t q = ((lo >> ((group & 1) * 4)) & 0x0f0f0f0fu) |
                            (((hi >> group) & 0x01010101u) << 4);
                        dot = cuda_dp4a_i32((int)q, (int)xv, dot);
                    }
                }
                wd = cuda_fp16_to_fp32(cuda_fp32_to_fp16_bits(wd));
                wm = cuda_fp16_to_fp32(cuda_fp32_to_fp16_bits(wm));
                xd = cuda_fp16_to_fp32(x->d); xs = cuda_fp16_to_fp32(x->original_sum);
            }
            if (type == BN_GGUF_TENSOR_Q6_K) {
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    float part = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int d = __shfl_sync(mask, dot, half * 4 + i, 8);
                        float dx = __shfl_sync(mask, xd, half * 4 + i, 8);
                        part = fmaf((float)d, dx, part);
                    }
                    acc = fmaf(part, wd, acc);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    int d = __shfl_sync(mask, dot, i, 8);
                    float dw = __shfl_sync(mask, wd, i, 8);
                    float dm = __shfl_sync(mask, wm, i, 8);
                    float dx = __shfl_sync(mask, xd, i, 8);
                    float sx = __shfl_sync(mask, xs, i, 8);
                    acc = fmaf(dw * dx, (float)d, acc);
                    acc = fmaf(-dm, sx, acc);
                }
            }
        }
        if (!have_tail) { tail = acc; have_tail = 1; }
        else prefix += acc;
    }
    if (lane == 0) out[(size_t)token * rows + row] = tail + prefix;
}


template<int type> static __global__ void kquant_routed_mmvq_kernel(float*out,const void*weights,const BnCudaBlockQ8_1*x,int rows,int cols,const int*ids,int repeat){
 __shared__ float partial[3][32];int row=blockIdx.x,item=blockIdx.y,tid=threadIdx.x,lane=tid&31,warp=tid/32,nb=cols/256;float sum=0;
 int group=type==BN_GGUF_TENSOR_Q6_K?32:16;
 for(int b=tid/group;b<nb;b+=blockDim.x/group){
  size_t wi=((size_t)ids[item]*rows+row)*nb+b;
  const BnCudaBlockQ8_1 *xb=x+(size_t)(item/repeat)*(cols/32)+b*8;
  sum+=type==BN_GGUF_TENSOR_Q4_K?cuda_vec_dot_q4k_q8_1((const BnBlockQ4K*)weights+wi,xb,2*(tid&15)):type==BN_GGUF_TENSOR_Q5_K?cuda_vec_dot_q5k_q8_1((const BnBlockQ5K*)weights+wi,xb,2*(tid&15)):cuda_vec_dot_q6k_q8_1_mmvq((const BnBlockQ6K*)weights+wi,xb,tid&31);
 }
 if(warp)partial[warp-1][lane]=sum;__syncthreads();
 if(!warp){for(int i=0;i<(int)blockDim.x/32-1;i++)sum+=partial[i][lane];for(int i=16;i;i/=2)sum+=__shfl_xor_sync(0xffffffffu,sum,i);if(!lane)out[(size_t)item*rows+row]=sum;}
}

#endif
