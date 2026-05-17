#include "gpu_cuda.h"

#ifdef BN_ENABLE_CUDA

#include "gguf.h"
#include "model_config.h"
#include "quant.h"
#include "gpu_shader.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    void *data;
    size_t size;
    int type;
    int rows;
    int cols;
} BnCudaBuffer;

typedef struct {
    int device;
    cudaStream_t stream;
    cudaStream_t exec_stream;
    cudaGraph_t exec_graph_def;
    cudaGraphExec_t exec_graph;
    cudaGraphNode_t *exec_nodes;
    int exec_nodes_cap;
    int exec_node_count;
    int exec_node_cursor;
    int exec_graph_ops;
    int kv_f16;
    float *d_x;
    size_t d_x_bytes;
    float *d_out;
    size_t d_out_bytes;
    void *d_ops;
    size_t d_ops_bytes;
    float *h_out;
    size_t h_out_bytes;
    void *d_runtime;
    float *act_bufs[BN_GPU_VALUE_COUNT];
    size_t act_sizes[BN_GPU_VALUE_COUNT];
} BnCudaCtx;

typedef struct {
    int pos;
    int n_kv;
    int cache_pos;
    int seq_len;
} BnCudaRuntimeParams;

typedef struct {
    const void *wdata;
    size_t out_offset;
    int rows;
    int cols;
    int type;
} BnCudaDeviceOp;

static int cuda_ensure_graph_nodes(BnCudaCtx *ctx, int need) {
    if (!ctx || need <= ctx->exec_nodes_cap) return ctx ? 0 : -1;
    int cap = ctx->exec_nodes_cap ? ctx->exec_nodes_cap * 2 : 256;
    while (cap < need) cap *= 2;
    cudaGraphNode_t *nodes = (cudaGraphNode_t *)realloc(
        ctx->exec_nodes, (size_t)cap * sizeof(cudaGraphNode_t));
    if (!nodes) return -1;
    ctx->exec_nodes = nodes;
    ctx->exec_nodes_cap = cap;
    return 0;
}

template <typename Kernel, typename... Args>
static int cuda_dispatch_kernel(BnCudaCtx *ctx, int graph_exec,
                                int graph_building, int graph_update_node,
                                Kernel kernel,
                                dim3 grid, dim3 block, size_t shared,
                                Args... args) {
    if (!graph_exec) {
        kernel<<<grid, block, shared, ctx->exec_stream>>>(args...);
        return 0;
    }

    if (!graph_building && !graph_update_node) {
        if (ctx->exec_node_cursor >= ctx->exec_node_count) {
            fprintf(stderr,
                    "[bn:gpu:cuda] graph replay node count mismatch\n");
            return -1;
        }
        ctx->exec_node_cursor++;
        return 0;
    }

    void *arg_ptrs[] = { (void *)&args... };
    cudaKernelNodeParams params;
    memset(&params, 0, sizeof(params));
    params.func = (void *)kernel;
    params.gridDim = grid;
    params.blockDim = block;
    params.sharedMemBytes = shared;
    params.kernelParams = arg_ptrs;

    cudaError_t err;
    if (graph_building) {
        if (cuda_ensure_graph_nodes(ctx, ctx->exec_node_count + 1) != 0)
            return -1;
        cudaGraphNode_t dep = NULL;
        const cudaGraphNode_t *deps = NULL;
        size_t n_deps = 0;
        if (ctx->exec_node_count > 0) {
            dep = ctx->exec_nodes[ctx->exec_node_count - 1];
            deps = &dep;
            n_deps = 1;
        }
        err = cudaGraphAddKernelNode(
            &ctx->exec_nodes[ctx->exec_node_count], ctx->exec_graph_def,
            deps, n_deps, &params);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] graph add kernel failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        ctx->exec_node_count++;
    } else {
        if (ctx->exec_node_cursor >= ctx->exec_node_count) {
            fprintf(stderr,
                    "[bn:gpu:cuda] graph update node count mismatch\n");
            return -1;
        }
        if (graph_update_node) {
            err = cudaGraphExecKernelNodeSetParams(
                ctx->exec_graph, ctx->exec_nodes[ctx->exec_node_cursor],
                &params);
            if (err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] graph set kernel failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
        }
        ctx->exec_node_cursor++;
    }
    return 0;
}

#define BN_CUDA_LAUNCH_EX(ctx, dynamic, kernel, grid, block, shared, ...) do { \
    if (cuda_dispatch_kernel((ctx), graph_exec, graph_building, (dynamic), \
                             (kernel), dim3(grid), dim3(block), (shared), \
                             __VA_ARGS__) != 0) return -1; \
} while (0)

#define BN_CUDA_LAUNCH(ctx, kernel, grid, block, shared, ...) \
    BN_CUDA_LAUNCH_EX(ctx, 1, kernel, grid, block, shared, __VA_ARGS__)

#define BN_CUDA_LAUNCH_STATIC(ctx, kernel, grid, block, shared, ...) \
    BN_CUDA_LAUNCH_EX(ctx, 0, kernel, grid, block, shared, __VA_ARGS__)

static int cuda_ctx_set_device(BnCudaCtx *ctx) {
    if (!ctx) return -1;
    cudaError_t err = cudaSetDevice(ctx->device);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] failed to select device %d: %s\n",
                ctx->device, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static double cuda_wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int cuda_env_int(const char *name, int fallback) {
    const char *env = getenv(name);
    if (!env || !*env) return fallback;
    return atoi(env);
}

static __device__ float cuda_fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static __device__ uint16_t cuda_fp32_to_fp16_bits(float f) {
    return __half_as_ushort(__float2half_rn(f));
}

static __device__ void cuda_kv_store(void *cache, size_t idx, float v,
                                     int kv_f16) {
    if (kv_f16) {
        ((uint16_t *)cache)[idx] = cuda_fp32_to_fp16_bits(v);
    } else {
        ((float *)cache)[idx] = v;
    }
}

static __device__ float cuda_kv_load(const void *cache, size_t idx,
                                     int kv_f16) {
    return kv_f16
        ? cuda_fp16_to_fp32(((const uint16_t *)cache)[idx])
        : ((const float *)cache)[idx];
}

static __device__ float cuda_q4k_value(const BnBlockQ4K *blk, int i) {
    const float d = cuda_fp16_to_fp32(blk->d);
    const float dmin = cuda_fp16_to_fp32(blk->dmin);
    const int group = i / 32;
    const int pair = i / 64;
    const int half = i & 31;
    const uint8_t packed = blk->qs[pair * 32 + half];
    const int q = (i & 32) ? (packed >> 4) : (packed & 15);

    uint32_t utmp[3];
    memcpy(utmp, blk->scales, 12);
    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t kmask3 = 0x03030303u;
    uint32_t mins_lo = utmp[1] & kmask1;
    uint32_t mins_hi = ((utmp[2] >> 4) & kmask2) |
                       (((utmp[1] >> 6) & kmask3) << 4);
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[0] &= kmask1;

    uint8_t scales[8];
    uint8_t mins[8];
    memcpy(scales, utmp, 8);
    memcpy(mins, &mins_lo, 4);
    memcpy(mins + 4, &mins_hi, 4);
    return d * (float)scales[group] * (float)q -
           dmin * (float)mins[group];
}

static __device__ void cuda_q4k_group_scale_min(const BnBlockQ4K *blk,
                                                int group,
                                                int *scale,
                                                int *minv) {
    uint32_t utmp[3];
    memcpy(utmp, blk->scales, 12);
    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t kmask3 = 0x03030303u;
    uint32_t mins_lo = utmp[1] & kmask1;
    uint32_t mins_hi = ((utmp[2] >> 4) & kmask2) |
                       (((utmp[1] >> 6) & kmask3) << 4);
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[0] &= kmask1;

    uint8_t scales[8];
    uint8_t mins[8];
    memcpy(scales, utmp, 8);
    memcpy(mins, &mins_lo, 4);
    memcpy(mins + 4, &mins_hi, 4);
    *scale = scales[group];
    *minv = mins[group];
}

static __device__ float cuda_q5k_value(const BnBlockQ5K *blk, int i) {
    const float d = cuda_fp16_to_fp32(blk->d);
    const float dmin = cuda_fp16_to_fp32(blk->dmin);
    const int group = i / 32;
    const int pair = i / 64;
    const int half = i & 31;
    const uint8_t packed = blk->qs[pair * 32 + half];
    const int bit = pair * 2 + ((i & 32) ? 1 : 0);
    const int q = ((i & 32) ? (packed >> 4) : (packed & 15)) |
                  (((blk->qh[half] >> bit) & 1) << 4);

    uint32_t utmp[3];
    memcpy(utmp, blk->scales, 12);
    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t kmask3 = 0x03030303u;
    uint32_t mins_lo = utmp[1] & kmask1;
    uint32_t mins_hi = ((utmp[2] >> 4) & kmask2) |
                       (((utmp[1] >> 6) & kmask3) << 4);
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[0] &= kmask1;

    uint8_t scales[8];
    uint8_t mins[8];
    memcpy(scales, utmp, 8);
    memcpy(mins, &mins_lo, 4);
    memcpy(mins + 4, &mins_hi, 4);
    return d * (float)scales[group] * (float)q -
           dmin * (float)mins[group];
}

static __device__ float cuda_q6k_value(const BnBlockQ6K *blk, int i) {
    const float d = cuda_fp16_to_fp32(blk->d);
    const int chunk = i / 128;
    const int in_chunk = i & 127;
    const int l = in_chunk & 31;
    const uint8_t *ql = blk->ql + chunk * 64;
    const uint8_t *qh = blk->qh + chunk * 32;
    const int8_t *sc = blk->scales + chunk * 8;

    int q;
    int scale_idx;
    if (in_chunk < 32) {
        q = (int)((ql[l] & 0x0f) | ((qh[l] & 0x03) << 4)) - 32;
        scale_idx = l / 16;
    } else if (in_chunk < 64) {
        q = (int)((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) - 32;
        scale_idx = l / 16 + 2;
    } else if (in_chunk < 96) {
        q = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) - 32;
        scale_idx = l / 16 + 4;
    } else {
        q = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) - 32;
        scale_idx = l / 16 + 6;
    }
    return d * (float)sc[scale_idx] * (float)q;
}

static __device__ float cuda_dot_row(const void *wdata, const float *x,
                                     int row, int cols, int type) {
    int tid = threadIdx.x;
    float sum = 0.0f;

    if (type == BN_GGUF_TENSOR_F32) {
        const float *w = (const float *)wdata + (size_t)row * cols;
        for (int c = tid; c < cols; c += blockDim.x)
            sum += w[c] * x[c];
    } else if (type == BN_GGUF_TENSOR_F16) {
        const uint16_t *w = (const uint16_t *)wdata + (size_t)row * cols;
        for (int c = tid; c < cols; c += blockDim.x)
            sum += cuda_fp16_to_fp32(w[c]) * x[c];
    } else if (type == BN_GGUF_TENSOR_Q8_0) {
        const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)wdata;
        int n_bpr = cols / 32;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / 32;
            int i = c & 31;
            const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_bpr + b];
            float d = cuda_fp16_to_fp32(blk->d);
            sum += d * (float)blk->qs[i] * x[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q4_0) {
        const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)wdata;
        int n_bpr = cols / 32;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / 32;
            int i = c & 31;
            const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_bpr + b];
            float d = cuda_fp16_to_fp32(blk->d);
            uint8_t qs = blk->qs[i & 15];
            int q = i < 16 ? (int)(qs & 15) - 8 : (int)(qs >> 4) - 8;
            sum += d * (float)q * x[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q5_0) {
        const BnBlockQ5_0 *blocks = (const BnBlockQ5_0 *)wdata;
        int n_bpr = cols / 32;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / 32;
            int i = c & 31;
            const BnBlockQ5_0 *blk = &blocks[(size_t)row * n_bpr + b];
            float d = cuda_fp16_to_fp32(blk->d);
            uint32_t qh = (uint32_t)blk->qh[0] |
                          ((uint32_t)blk->qh[1] << 8) |
                          ((uint32_t)blk->qh[2] << 16) |
                          ((uint32_t)blk->qh[3] << 24);
            uint8_t qs = blk->qs[i & 15];
            int q = i < 16
                ? (int)((qs & 15) | (((qh >> i) & 1u) << 4)) - 16
                : (int)((qs >> 4) | (((qh >> i) & 1u) << 4)) - 16;
            sum += d * (float)q * x[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q4_K) {
        const BnBlockQ4K *blocks = (const BnBlockQ4K *)wdata;
        int n_bpr = cols / BN_QK_K;
        if (blockDim.x == BN_QK_K && tid < BN_QK_K) {
            int lane = tid & 31;
            int group = tid >> 5;
            int pair = tid >> 6;
            int high = tid & 32;
            const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ4K *blk = &row_blocks[b];
                int sc = 0;
                int mn = 0;
                float d = 0.0f;
                float dmin = 0.0f;
                if (lane == 0) {
                    cuda_q4k_group_scale_min(blk, group, &sc, &mn);
                    d = cuda_fp16_to_fp32(blk->d);
                    dmin = cuda_fp16_to_fp32(blk->dmin);
                }
                sc = __shfl_sync(0xffffffffu, sc, 0);
                mn = __shfl_sync(0xffffffffu, mn, 0);
                d = __shfl_sync(0xffffffffu, d, 0);
                dmin = __shfl_sync(0xffffffffu, dmin, 0);
                uint8_t packed = blk->qs[pair * 32 + lane];
                int q = high ? (packed >> 4) : (packed & 15);
                sum += (d * (float)sc * (float)q - dmin * (float)mn) *
                       x[(size_t)b * BN_QK_K + tid];
            }
        } else {
            for (int c = tid; c < cols; c += blockDim.x) {
                int b = c / BN_QK_K;
                int i = c & (BN_QK_K - 1);
                const BnBlockQ4K *blk = &blocks[(size_t)row * n_bpr + b];
                sum += cuda_q4k_value(blk, i) * x[c];
            }
        }
    } else if (type == BN_GGUF_TENSOR_Q5_K) {
        const BnBlockQ5K *blocks = (const BnBlockQ5K *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            const BnBlockQ5K *blk = &blocks[(size_t)row * n_bpr + b];
            sum += cuda_q5k_value(blk, i) * x[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q6_K) {
        const BnBlockQ6K *blocks = (const BnBlockQ6K *)wdata;
        int n_bpr = cols / BN_QK_K;
        if (blockDim.x == BN_QK_K && tid < BN_QK_K) {
            int chunk = tid / 128;
            int in_chunk = tid & 127;
            int l = in_chunk & 31;
            int ql_off = chunk * 64;
            int qh_off = chunk * 32;
            int scale_idx = 0;
            const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
            if (in_chunk < 32) {
                scale_idx = l / 16;
                for (int b = 0; b < n_bpr; b++) {
                    const BnBlockQ6K *blk = &row_blocks[b];
                    int q = (int)((blk->ql[ql_off + l] & 0x0f) |
                                  ((blk->qh[qh_off + l] & 0x03) << 4)) - 32;
                    sum += cuda_fp16_to_fp32(blk->d) *
                           (float)blk->scales[chunk * 8 + scale_idx] *
                           (float)q * x[(size_t)b * BN_QK_K + tid];
                }
            } else if (in_chunk < 64) {
                scale_idx = l / 16 + 2;
                for (int b = 0; b < n_bpr; b++) {
                    const BnBlockQ6K *blk = &row_blocks[b];
                    int q = (int)((blk->ql[ql_off + l + 32] & 0x0f) |
                                  (((blk->qh[qh_off + l] >> 2) & 0x03) << 4)) - 32;
                    sum += cuda_fp16_to_fp32(blk->d) *
                           (float)blk->scales[chunk * 8 + scale_idx] *
                           (float)q * x[(size_t)b * BN_QK_K + tid];
                }
            } else if (in_chunk < 96) {
                scale_idx = l / 16 + 4;
                for (int b = 0; b < n_bpr; b++) {
                    const BnBlockQ6K *blk = &row_blocks[b];
                    int q = (int)((blk->ql[ql_off + l] >> 4) |
                                  (((blk->qh[qh_off + l] >> 4) & 0x03) << 4)) - 32;
                    sum += cuda_fp16_to_fp32(blk->d) *
                           (float)blk->scales[chunk * 8 + scale_idx] *
                           (float)q * x[(size_t)b * BN_QK_K + tid];
                }
            } else {
                scale_idx = l / 16 + 6;
                for (int b = 0; b < n_bpr; b++) {
                    const BnBlockQ6K *blk = &row_blocks[b];
                    int q = (int)((blk->ql[ql_off + l + 32] >> 4) |
                                  (((blk->qh[qh_off + l] >> 6) & 0x03) << 4)) - 32;
                    sum += cuda_fp16_to_fp32(blk->d) *
                           (float)blk->scales[chunk * 8 + scale_idx] *
                           (float)q * x[(size_t)b * BN_QK_K + tid];
                }
            }
        } else {
            for (int c = tid; c < cols; c += blockDim.x) {
                int b = c / BN_QK_K;
                int i = c & (BN_QK_K - 1);
                const BnBlockQ6K *blk = &blocks[(size_t)row * n_bpr + b];
                sum += cuda_q6k_value(blk, i) * x[c];
            }
        }
    } else if (type == BN_GGUF_TENSOR_Q8_K) {
        const BnBlockQ8K *blocks = (const BnBlockQ8K *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            const BnBlockQ8K *blk = &blocks[(size_t)row * n_bpr + b];
            sum += blk->d * (float)blk->qs[i] * x[c];
        }
    }
    return sum;
}

static __device__ float cuda_block_reduce_sum(float v, float *scratch) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    if (lane == 0)
        scratch[warp] = v;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    v = tid < n_warps ? scratch[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

static __device__ float cuda_block_reduce_sum_all(float v, float *scratch) {
    int tid = threadIdx.x;
    v = cuda_block_reduce_sum(v, scratch);
    if (tid == 0)
        scratch[0] = v;
    __syncthreads();
    return scratch[0];
}

static __device__ float cuda_block_reduce_max_all(float v, float *scratch) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    if (lane == 0)
        scratch[warp] = v;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    v = tid < n_warps ? scratch[lane] : -INFINITY;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    if (tid == 0)
        scratch[0] = v;
    __syncthreads();
    return scratch[0];
}

static __global__ void matvec_kernel(float *out, const void *wdata,
                                     const float *x, const float *bias,
                                     int rows, int cols, int type,
                                     size_t out_offset) {
    int row = blockIdx.x;
    int token = blockIdx.y;
    int tid = threadIdx.x;
    const float *x_token = x + (size_t)token * cols;
    float sum = 0.0f;

    if (type == BN_GGUF_TENSOR_F32) {
        const float *w = (const float *)wdata + (size_t)row * cols;
        for (int c = tid; c < cols; c += blockDim.x)
            sum += w[c] * x_token[c];
    } else if (type == BN_GGUF_TENSOR_F16) {
        const uint16_t *w = (const uint16_t *)wdata + (size_t)row * cols;
        for (int c = tid; c < cols; c += blockDim.x)
            sum += cuda_fp16_to_fp32(w[c]) * x_token[c];
    } else if (type == BN_GGUF_TENSOR_Q8_0) {
        const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)wdata;
        int n_bpr = cols / 32;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / 32;
            int i = c & 31;
            const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_bpr + b];
            float d = cuda_fp16_to_fp32(blk->d);
            sum += d * (float)blk->qs[i] * x_token[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q4_0) {
        const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)wdata;
        int n_bpr = cols / 32;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / 32;
            int i = c & 31;
            const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_bpr + b];
            float d = cuda_fp16_to_fp32(blk->d);
            uint8_t qs = blk->qs[i & 15];
            int q = i < 16 ? (int)(qs & 15) - 8 : (int)(qs >> 4) - 8;
            sum += d * (float)q * x_token[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q5_0) {
        const BnBlockQ5_0 *blocks = (const BnBlockQ5_0 *)wdata;
        int n_bpr = cols / 32;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / 32;
            int i = c & 31;
            const BnBlockQ5_0 *blk = &blocks[(size_t)row * n_bpr + b];
            float d = cuda_fp16_to_fp32(blk->d);
            uint32_t qh = (uint32_t)blk->qh[0] |
                          ((uint32_t)blk->qh[1] << 8) |
                          ((uint32_t)blk->qh[2] << 16) |
                          ((uint32_t)blk->qh[3] << 24);
            uint8_t qs = blk->qs[i & 15];
            int q = i < 16
                ? (int)((qs & 15) | (((qh >> i) & 1u) << 4)) - 16
                : (int)((qs >> 4) | (((qh >> i) & 1u) << 4)) - 16;
            sum += d * (float)q * x_token[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q4_K) {
        const BnBlockQ4K *blocks = (const BnBlockQ4K *)wdata;
        int n_bpr = cols / BN_QK_K;
        if (blockDim.x == BN_QK_K && tid < BN_QK_K) {
            int lane = tid & 31;
            int group = tid >> 5;
            int pair = tid >> 6;
            int high = tid & 32;
            const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ4K *blk = &row_blocks[b];
                int sc = 0;
                int mn = 0;
                float d = 0.0f;
                float dmin = 0.0f;
                if (lane == 0) {
                    cuda_q4k_group_scale_min(blk, group, &sc, &mn);
                    d = cuda_fp16_to_fp32(blk->d);
                    dmin = cuda_fp16_to_fp32(blk->dmin);
                }
                sc = __shfl_sync(0xffffffffu, sc, 0);
                mn = __shfl_sync(0xffffffffu, mn, 0);
                d = __shfl_sync(0xffffffffu, d, 0);
                dmin = __shfl_sync(0xffffffffu, dmin, 0);
                uint8_t packed = blk->qs[pair * 32 + lane];
                int q = high ? (packed >> 4) : (packed & 15);
                sum += (d * (float)sc * (float)q - dmin * (float)mn) *
                       x_token[(size_t)b * BN_QK_K + tid];
            }
        } else {
            for (int c = tid; c < cols; c += blockDim.x) {
                int b = c / BN_QK_K;
                int i = c & (BN_QK_K - 1);
                const BnBlockQ4K *blk = &blocks[(size_t)row * n_bpr + b];
                sum += cuda_q4k_value(blk, i) * x_token[c];
            }
        }
    } else if (type == BN_GGUF_TENSOR_Q5_K) {
        const BnBlockQ5K *blocks = (const BnBlockQ5K *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            const BnBlockQ5K *blk = &blocks[(size_t)row * n_bpr + b];
            sum += cuda_q5k_value(blk, i) * x_token[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q6_K) {
        const BnBlockQ6K *blocks = (const BnBlockQ6K *)wdata;
        int n_bpr = cols / BN_QK_K;
        if (blockDim.x == BN_QK_K && tid < BN_QK_K) {
            int chunk = tid / 128;
            int in_chunk = tid & 127;
            int l = in_chunk & 31;
            int ql_off = chunk * 64;
            int qh_off = chunk * 32;
            int scale_idx = 0;
            const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
            if (in_chunk < 32) {
                scale_idx = l / 16;
                for (int b = 0; b < n_bpr; b++) {
                    const BnBlockQ6K *blk = &row_blocks[b];
                    int q = (int)((blk->ql[ql_off + l] & 0x0f) |
                                  ((blk->qh[qh_off + l] & 0x03) << 4)) - 32;
                    sum += cuda_fp16_to_fp32(blk->d) *
                           (float)blk->scales[chunk * 8 + scale_idx] *
                           (float)q * x_token[(size_t)b * BN_QK_K + tid];
                }
            } else if (in_chunk < 64) {
                scale_idx = l / 16 + 2;
                for (int b = 0; b < n_bpr; b++) {
                    const BnBlockQ6K *blk = &row_blocks[b];
                    int q = (int)((blk->ql[ql_off + l + 32] & 0x0f) |
                                  (((blk->qh[qh_off + l] >> 2) & 0x03) << 4)) - 32;
                    sum += cuda_fp16_to_fp32(blk->d) *
                           (float)blk->scales[chunk * 8 + scale_idx] *
                           (float)q * x_token[(size_t)b * BN_QK_K + tid];
                }
            } else if (in_chunk < 96) {
                scale_idx = l / 16 + 4;
                for (int b = 0; b < n_bpr; b++) {
                    const BnBlockQ6K *blk = &row_blocks[b];
                    int q = (int)((blk->ql[ql_off + l] >> 4) |
                                  (((blk->qh[qh_off + l] >> 4) & 0x03) << 4)) - 32;
                    sum += cuda_fp16_to_fp32(blk->d) *
                           (float)blk->scales[chunk * 8 + scale_idx] *
                           (float)q * x_token[(size_t)b * BN_QK_K + tid];
                }
            } else {
                scale_idx = l / 16 + 6;
                for (int b = 0; b < n_bpr; b++) {
                    const BnBlockQ6K *blk = &row_blocks[b];
                    int q = (int)((blk->ql[ql_off + l + 32] >> 4) |
                                  (((blk->qh[qh_off + l] >> 6) & 0x03) << 4)) - 32;
                    sum += cuda_fp16_to_fp32(blk->d) *
                           (float)blk->scales[chunk * 8 + scale_idx] *
                           (float)q * x_token[(size_t)b * BN_QK_K + tid];
                }
            }
        } else {
            for (int c = tid; c < cols; c += blockDim.x) {
                int b = c / BN_QK_K;
                int i = c & (BN_QK_K - 1);
                const BnBlockQ6K *blk = &blocks[(size_t)row * n_bpr + b];
                sum += cuda_q6k_value(blk, i) * x_token[c];
            }
        }
    } else if (type == BN_GGUF_TENSOR_Q8_K) {
        const BnBlockQ8K *blocks = (const BnBlockQ8K *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            const BnBlockQ8K *blk = &blocks[(size_t)row * n_bpr + b];
            sum += blk->d * (float)blk->qs[i] * x_token[c];
        }
    }

    extern __shared__ float scratch[];
    sum = cuda_block_reduce_sum(sum, scratch);
    if (tid == 0) {
        float v = sum;
        if (bias) v += bias[row];
        out[out_offset + (size_t)token * rows + row] = v;
    }
}

static __global__ void q8_0_matvec4_warp_kernel(float *out,
                                                const BnBlockQ8_0 *blocks,
                                                const float *x, int rows,
                                                int cols,
                                                size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row0 = (blockIdx.x * warps_per_block + warp) * 4;
    if (row0 >= rows) return;

    int n_bpr = cols / 32;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    const BnBlockQ8_0 *row_blocks0 = blocks + (size_t)row0 * n_bpr;
    const BnBlockQ8_0 *row_blocks1 = row_blocks0 + n_bpr;
    const BnBlockQ8_0 *row_blocks2 = row_blocks1 + n_bpr;
    const BnBlockQ8_0 *row_blocks3 = row_blocks2 + n_bpr;

    for (int b = 0; b < n_bpr; b++) {
        float xv = x[(size_t)b * 32 + lane];
        const BnBlockQ8_0 *blk0 = &row_blocks0[b];
        sum0 += cuda_fp16_to_fp32(blk0->d) * (float)blk0->qs[lane] * xv;
        if (row0 + 1 < rows) {
            const BnBlockQ8_0 *blk1 = &row_blocks1[b];
            sum1 += cuda_fp16_to_fp32(blk1->d) * (float)blk1->qs[lane] * xv;
        }
        if (row0 + 2 < rows) {
            const BnBlockQ8_0 *blk2 = &row_blocks2[b];
            sum2 += cuda_fp16_to_fp32(blk2->d) * (float)blk2->qs[lane] * xv;
        }
        if (row0 + 3 < rows) {
            const BnBlockQ8_0 *blk3 = &row_blocks3[b];
            sum3 += cuda_fp16_to_fp32(blk3->d) * (float)blk3->qs[lane] * xv;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
    }
    if (lane == 0) {
        out[out_offset + (size_t)row0] = sum0;
        if (row0 + 1 < rows) out[out_offset + (size_t)row0 + 1] = sum1;
        if (row0 + 2 < rows) out[out_offset + (size_t)row0 + 2] = sum2;
        if (row0 + 3 < rows) out[out_offset + (size_t)row0 + 3] = sum3;
    }
}

static __global__ void q5_0_matvec_warp_kernel(float *out,
                                               const BnBlockQ5_0 *blocks,
                                               const float *x,
                                               const float *bias,
                                               int rows, int cols,
                                               size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;

    int n_bpr = cols / 32;
    float sum = 0.0f;
    for (int b = 0; b < n_bpr; b++) {
        const BnBlockQ5_0 *blk = &blocks[(size_t)row * n_bpr + b];
        uint32_t qh = (uint32_t)blk->qh[0] |
                      ((uint32_t)blk->qh[1] << 8) |
                      ((uint32_t)blk->qh[2] << 16) |
                      ((uint32_t)blk->qh[3] << 24);
        uint8_t qs = blk->qs[lane & 15];
        int q = lane < 16
            ? (int)((qs & 15) | (((qh >> lane) & 1u) << 4)) - 16
            : (int)((qs >> 4) | (((qh >> lane) & 1u) << 4)) - 16;
        sum += cuda_fp16_to_fp32(blk->d) * (float)q *
               x[(size_t)b * 32 + lane];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) {
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __device__ float cuda_q5_0_value(const BnBlockQ5_0 *blocks, int row,
                                        int n_bpr, int c) {
    int b = c / 32;
    int i = c & 31;
    const BnBlockQ5_0 *blk = &blocks[(size_t)row * n_bpr + b];
    uint32_t qh = (uint32_t)blk->qh[0] |
                  ((uint32_t)blk->qh[1] << 8) |
                  ((uint32_t)blk->qh[2] << 16) |
                  ((uint32_t)blk->qh[3] << 24);
    uint8_t qs = blk->qs[i & 15];
    int q = i < 16
        ? (int)((qs & 15) | (((qh >> i) & 1u) << 4)) - 16
        : (int)((qs >> 4) | (((qh >> i) & 1u) << 4)) - 16;
    return cuda_fp16_to_fp32(blk->d) * (float)q;
}

static __global__ void q5_0_matvec4_kernel(float *out,
                                           const BnBlockQ5_0 *blocks,
                                           const float *x,
                                           const float *bias,
                                           int rows, int cols,
                                           size_t out_offset) {
    int row0 = blockIdx.x * 4;
    int tid = threadIdx.x;
    int n_bpr = cols / 32;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (int c = tid; c < cols; c += blockDim.x) {
        float xv = x[c];
        if (row0 + 0 < rows)
            sum0 += cuda_q5_0_value(blocks, row0 + 0, n_bpr, c) * xv;
        if (row0 + 1 < rows)
            sum1 += cuda_q5_0_value(blocks, row0 + 1, n_bpr, c) * xv;
        if (row0 + 2 < rows)
            sum2 += cuda_q5_0_value(blocks, row0 + 2, n_bpr, c) * xv;
        if (row0 + 3 < rows)
            sum3 += cuda_q5_0_value(blocks, row0 + 3, n_bpr, c) * xv;
    }

    extern __shared__ float scratch[];
    float *s0 = scratch;
    float *s1 = s0 + blockDim.x;
    float *s2 = s1 + blockDim.x;
    float *s3 = s2 + blockDim.x;
    s0[tid] = sum0;
    s1[tid] = sum1;
    s2[tid] = sum2;
    s3[tid] = sum3;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] += s1[tid + stride];
            s2[tid] += s2[tid + stride];
            s3[tid] += s3[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (row0 + 0 < rows)
            out[out_offset + (size_t)row0 + 0] =
                s0[0] + (bias ? bias[row0 + 0] : 0.0f);
        if (row0 + 1 < rows)
            out[out_offset + (size_t)row0 + 1] =
                s1[0] + (bias ? bias[row0 + 1] : 0.0f);
        if (row0 + 2 < rows)
            out[out_offset + (size_t)row0 + 2] =
                s2[0] + (bias ? bias[row0 + 2] : 0.0f);
        if (row0 + 3 < rows)
            out[out_offset + (size_t)row0 + 3] =
                s3[0] + (bias ? bias[row0 + 3] : 0.0f);
    }
}

static __global__ void q6k_matvec_warp_kernel(float *out,
                                              const BnBlockQ6K *blocks,
                                              const float *x,
                                              const float *bias,
                                              int rows, int cols,
                                              size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;

    int n_bpr = cols / BN_QK_K;
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = 0; b < n_bpr; b++) {
        const BnBlockQ6K *blk = &row_blocks[b];
        for (int i = lane; i < BN_QK_K; i += 32)
            sum += cuda_q6k_value(blk, i) * x[(size_t)b * BN_QK_K + i];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) {
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void matvec_split_kernel(float *out0, float *out1,
                                           float *out2, const void *wdata,
                                           const float *x,
                                           const float *bias0,
                                           int total_rows, int cols, int type,
                                           int split0, int split1,
                                           size_t out1_offset,
                                           size_t out2_offset) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= total_rows) return;

    float sum = cuda_dot_row(wdata, x, row, cols, type);
    extern __shared__ float scratch[];
    sum = cuda_block_reduce_sum(sum, scratch);
    if (tid != 0) return;

    if (row < split0) {
        float v = sum;
        if (bias0) v += bias0[row];
        out0[row] = v;
    } else if (split1 > split0 && row >= split1) {
        if (out2)
            out2[out2_offset + (size_t)(row - split1)] = sum;
    } else {
            out1[out1_offset + (size_t)(row - split0)] = sum;
    }
}

static __global__ void qkv_mixed_matvec_kernel(
    float *q_out, float *k_out, void *value_cache,
    const void *qk_wdata, const void *v_wdata, const float *x,
    const float *q_bias, const float *k_bias, const float *v_bias,
    const float *freq, void *key_cache, int q_rows, int k_rows,
    int v_rows, int cols, int qk_type, int v_type, size_t key_offset,
    size_t value_offset, int k_heads, int head_size, int pos,
    int rope_dims, int k_pair_write, int kv_f16) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int k_pair_rows = k_pair_write ? (k_rows + 1) / 2 : k_rows;
    int total = q_rows + k_pair_rows + v_rows;
    if (row >= total) return;

    int local = row;
    const void *wdata = qk_wdata;
    int type = qk_type;
    int out_kind = 0;
    if (local >= q_rows + k_pair_rows) {
        local -= q_rows + k_pair_rows;
        wdata = v_wdata;
        type = v_type;
        out_kind = 2;
    } else if (local >= q_rows) {
        local -= q_rows;
        if (k_pair_write)
            local *= 2;
        out_kind = 1;
    }

    float pair_sum = 0.0f;
    int k_rotate = 0;
    int k_head = 0;
    int k_dim = 0;
    if (out_kind == 1 && freq && key_cache && k_heads > 0 &&
        head_size > 0 && k_heads * head_size == k_rows) {
        k_head = local / head_size;
        k_dim = local - k_head * head_size;
        k_rotate = k_dim < rope_dims;
        if (k_pair_write && k_rotate && (k_dim & 1))
            return;
    }

    float sum = cuda_dot_row(wdata, x, out_kind == 2 ? local : row,
                             cols, type);
    if (out_kind == 1 && k_rotate) {
        int pair_dim = k_dim ^ 1;
        int pair_row = q_rows + k_head * head_size + pair_dim;
        pair_sum = cuda_dot_row(qk_wdata, x, pair_row, cols, qk_type);
    }
    extern __shared__ float scratch[];
    float *scratch_pair = scratch + blockDim.x;
    sum = cuda_block_reduce_sum(sum, scratch);
    pair_sum = cuda_block_reduce_sum(pair_sum, scratch_pair);
    if (tid != 0) return;

    float v = sum;
    if (out_kind == 0) {
        q_out[local] = v + (q_bias ? q_bias[local] : 0.0f);
    } else if (out_kind == 1) {
        v += k_bias ? k_bias[local] : 0.0f;
        if (key_cache && k_heads > 0 && head_size > 0 &&
            k_heads * head_size == k_rows) {
        if (k_rotate) {
            int pair_dim = k_dim ^ 1;
            float pair = pair_sum +
                (k_bias ? k_bias[k_head * head_size + pair_dim] : 0.0f);
            float angle = (float)pos * freq[k_dim / 2];
            float s, c;
            __sincosf(angle, &s, &c);
            if (k_pair_write) {
                cuda_kv_store(key_cache, key_offset + (size_t)local,
                              v * c - pair * s, kv_f16);
                cuda_kv_store(key_cache, key_offset + (size_t)(local + 1),
                              v * s + pair * c, kv_f16);
            } else {
                float x0 = (k_dim & 1) ? pair : v;
                float x1 = (k_dim & 1) ? v : pair;
                cuda_kv_store(key_cache, key_offset + (size_t)local,
                              (k_dim & 1) ? (x0 * s + x1 * c)
                                          : (x0 * c - x1 * s),
                              kv_f16);
            }
        } else {
            cuda_kv_store(key_cache, key_offset + (size_t)local, v, kv_f16);
        }
        } else {
            k_out[local] = v;
        }
    } else {
        cuda_kv_store(value_cache, value_offset + (size_t)local,
                      v + (v_bias ? v_bias[local] : 0.0f), kv_f16);
    }
}

static __global__ void qkv_mixed_matvec_runtime_kernel(
    float *q_out, float *k_out, void *value_cache,
    const void *qk_wdata, const void *v_wdata, const float *x,
    const float *q_bias, const float *k_bias, const float *v_bias,
    const float *freq, void *key_cache, int q_rows, int k_rows,
    int v_rows, int cols, int qk_type, int v_type, size_t key_base_offset,
    size_t value_base_offset, int k_heads, int head_size,
    int rope_dims, int k_pair_write, int kv_f16,
    const BnCudaRuntimeParams *runtime) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int k_pair_rows = k_pair_write ? (k_rows + 1) / 2 : k_rows;
    int total = q_rows + k_pair_rows + v_rows;
    if (row >= total) return;

    int local = row;
    const void *wdata = qk_wdata;
    int type = qk_type;
    int out_kind = 0;
    if (local >= q_rows + k_pair_rows) {
        local -= q_rows + k_pair_rows;
        wdata = v_wdata;
        type = v_type;
        out_kind = 2;
    } else if (local >= q_rows) {
        local -= q_rows;
        if (k_pair_write)
            local *= 2;
        out_kind = 1;
    }

    int pos = runtime ? runtime->pos : 0;
    int cache_pos = runtime ? runtime->cache_pos : 0;
    size_t dyn_offset = (size_t)cache_pos * (size_t)k_rows;
    size_t key_offset = key_base_offset + dyn_offset;
    size_t value_offset = value_base_offset + dyn_offset;

    float pair_sum = 0.0f;
    int k_rotate = 0;
    int k_head = 0;
    int k_dim = 0;
    if (out_kind == 1 && freq && key_cache && k_heads > 0 &&
        head_size > 0 && k_heads * head_size == k_rows) {
        k_head = local / head_size;
        k_dim = local - k_head * head_size;
        k_rotate = k_dim < rope_dims;
        if (k_pair_write && k_rotate && (k_dim & 1))
            return;
    }

    float sum = cuda_dot_row(wdata, x, out_kind == 2 ? local : row,
                             cols, type);
    if (out_kind == 1 && k_rotate) {
        int pair_dim = k_dim ^ 1;
        int pair_row = q_rows + k_head * head_size + pair_dim;
        pair_sum = cuda_dot_row(qk_wdata, x, pair_row, cols, qk_type);
    }
    extern __shared__ float scratch[];
    float *scratch_pair = scratch + blockDim.x;
    sum = cuda_block_reduce_sum(sum, scratch);
    pair_sum = cuda_block_reduce_sum(pair_sum, scratch_pair);
    if (tid != 0) return;

    float v = sum;
    if (out_kind == 0) {
        q_out[local] = v + (q_bias ? q_bias[local] : 0.0f);
    } else if (out_kind == 1) {
        v += k_bias ? k_bias[local] : 0.0f;
        if (key_cache && k_heads > 0 && head_size > 0 &&
            k_heads * head_size == k_rows) {
            if (k_rotate) {
                int pair_dim = k_dim ^ 1;
                float pair = pair_sum +
                    (k_bias ? k_bias[k_head * head_size + pair_dim] : 0.0f);
                float angle = (float)pos * freq[k_dim / 2];
                float s, c;
                __sincosf(angle, &s, &c);
                if (k_pair_write) {
                    cuda_kv_store(key_cache, key_offset + (size_t)local,
                                  v * c - pair * s, kv_f16);
                    cuda_kv_store(key_cache, key_offset + (size_t)(local + 1),
                                  v * s + pair * c, kv_f16);
                } else {
                    float x0 = (k_dim & 1) ? pair : v;
                    float x1 = (k_dim & 1) ? v : pair;
                    cuda_kv_store(key_cache, key_offset + (size_t)local,
                                  (k_dim & 1) ? (x0 * s + x1 * c)
                                              : (x0 * c - x1 * s),
                                  kv_f16);
                }
            } else {
                cuda_kv_store(key_cache, key_offset + (size_t)local, v,
                              kv_f16);
            }
        } else {
            k_out[local] = v;
        }
    } else {
        cuda_kv_store(value_cache, value_offset + (size_t)local,
                      v + (v_bias ? v_bias[local] : 0.0f), kv_f16);
    }
}

static __global__ void kv_mixed_matvec_kernel(
    void *key_cache, void *value_cache,
    const void *qk_wdata, const void *v_wdata, const float *x,
    const float *k_bias, const float *v_bias, const float *freq,
    int q_offset, int k_rows, int v_rows, int cols, int qk_type, int v_type,
    size_t key_offset, size_t value_offset, int k_heads, int head_size,
    int pos, int rope_dims, int k_pair_write, int kv_f16) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int k_pair_rows = k_pair_write ? (k_rows + 1) / 2 : k_rows;
    int total = k_pair_rows + v_rows;
    if (row >= total) return;

    int local = row;
    const void *wdata = qk_wdata;
    int type = qk_type;
    int out_kind = 1;
    if (local >= k_pair_rows) {
        local -= k_pair_rows;
        wdata = v_wdata;
        type = v_type;
        out_kind = 2;
    } else if (k_pair_write) {
        local *= 2;
    }

    float pair_sum = 0.0f;
    int k_rotate = 0;
    int k_head = 0;
    int k_dim = 0;
    if (out_kind == 1 && freq && key_cache && k_heads > 0 &&
        head_size > 0 && k_heads * head_size == k_rows) {
        k_head = local / head_size;
        k_dim = local - k_head * head_size;
        k_rotate = k_dim < rope_dims;
        if (k_pair_write && k_rotate && (k_dim & 1))
            return;
    }

    int dot_row = out_kind == 1 ? q_offset + local : local;
    float sum = cuda_dot_row(wdata, x, dot_row, cols, type);
    if (out_kind == 1 && k_rotate) {
        int pair_dim = k_dim ^ 1;
        int pair_row = q_offset + k_head * head_size + pair_dim;
        pair_sum = cuda_dot_row(qk_wdata, x, pair_row, cols, qk_type);
    }

    extern __shared__ float scratch[];
    float *scratch_pair = scratch + blockDim.x;
    sum = cuda_block_reduce_sum(sum, scratch);
    pair_sum = cuda_block_reduce_sum(pair_sum, scratch_pair);
    if (tid != 0) return;

    float v = sum;
    if (out_kind == 1) {
        v += k_bias ? k_bias[local] : 0.0f;
        if (k_rotate) {
            int pair_dim = k_dim ^ 1;
            float pair = pair_sum +
                (k_bias ? k_bias[k_head * head_size + pair_dim] : 0.0f);
            float angle = (float)pos * freq[k_dim / 2];
            float s, c;
            __sincosf(angle, &s, &c);
            if (k_pair_write) {
                cuda_kv_store(key_cache, key_offset + (size_t)local,
                              v * c - pair * s, kv_f16);
                cuda_kv_store(key_cache, key_offset + (size_t)(local + 1),
                              v * s + pair * c, kv_f16);
            } else {
                float x0 = (k_dim & 1) ? pair : v;
                float x1 = (k_dim & 1) ? v : pair;
                cuda_kv_store(key_cache, key_offset + (size_t)local,
                              (k_dim & 1) ? (x0 * s + x1 * c)
                                          : (x0 * c - x1 * s),
                              kv_f16);
            }
        } else {
            cuda_kv_store(key_cache, key_offset + (size_t)local, v, kv_f16);
        }
    } else {
        cuda_kv_store(value_cache, value_offset + (size_t)local,
                      v + (v_bias ? v_bias[local] : 0.0f), kv_f16);
    }
}

static __global__ void fused_gateup_silu_kernel(float *out, const void *wdata,
                                                const float *x, int gate_rows,
                                                int up_rows, int cols,
                                                int type) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= gate_rows || row >= up_rows) return;

    float gate = cuda_dot_row(wdata, x, row, cols, type);
    float up = cuda_dot_row(wdata, x, gate_rows + row, cols, type);
    extern __shared__ float scratch[];
    float *scratch_gate = scratch;
    float *scratch_up = scratch + blockDim.x;
    gate = cuda_block_reduce_sum(gate, scratch_gate);
    up = cuda_block_reduce_sum(up, scratch_up);
    if (tid == 0) {
        out[row] = (gate / (1.0f + __expf(-gate))) * up;
    }
}

static __global__ void q5_0_fused_gateup_silu_warp_kernel(
    float *out, const BnBlockQ5_0 *blocks, const float *x, int gate_rows,
    int up_rows, int cols) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= gate_rows || row >= up_rows) return;

    int n_bpr = cols / 32;
    float gate = 0.0f;
    float up = 0.0f;
    const BnBlockQ5_0 *gate_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ5_0 *up_blocks =
        blocks + (size_t)(gate_rows + row) * n_bpr;
    for (int b = 0; b < n_bpr; b++) {
        const BnBlockQ5_0 *gb = &gate_blocks[b];
        const BnBlockQ5_0 *ub = &up_blocks[b];
        uint32_t gqh = (uint32_t)gb->qh[0] |
                       ((uint32_t)gb->qh[1] << 8) |
                       ((uint32_t)gb->qh[2] << 16) |
                       ((uint32_t)gb->qh[3] << 24);
        uint32_t uqh = (uint32_t)ub->qh[0] |
                       ((uint32_t)ub->qh[1] << 8) |
                       ((uint32_t)ub->qh[2] << 16) |
                       ((uint32_t)ub->qh[3] << 24);
        uint8_t gqs = gb->qs[lane & 15];
        uint8_t uqs = ub->qs[lane & 15];
        int gq = lane < 16
            ? (int)((gqs & 15) | (((gqh >> lane) & 1u) << 4)) - 16
            : (int)((gqs >> 4) | (((gqh >> lane) & 1u) << 4)) - 16;
        int uq = lane < 16
            ? (int)((uqs & 15) | (((uqh >> lane) & 1u) << 4)) - 16
            : (int)((uqs >> 4) | (((uqh >> lane) & 1u) << 4)) - 16;
        float xv = x[(size_t)b * 32 + lane];
        gate += cuda_fp16_to_fp32(gb->d) * (float)gq * xv;
        up += cuda_fp16_to_fp32(ub->d) * (float)uq * xv;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if (lane == 0)
        out[row] = (gate / (1.0f + __expf(-gate))) * up;
}

static __global__ void q8_0_fused_gateup_silu_warp_kernel(
    float *out, const BnBlockQ8_0 *blocks, const float *x, int gate_rows,
    int up_rows, int cols) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= gate_rows || row >= up_rows) return;

    int n_bpr = cols / 32;
    float gate = 0.0f;
    float up = 0.0f;
    const BnBlockQ8_0 *gate_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ8_0 *up_blocks =
        blocks + (size_t)(gate_rows + row) * n_bpr;
    for (int b = 0; b < n_bpr; b++) {
        const BnBlockQ8_0 *gb = &gate_blocks[b];
        const BnBlockQ8_0 *ub = &up_blocks[b];
        float xv = x[(size_t)b * 32 + lane];
        float gd = lane == 0 ? cuda_fp16_to_fp32(gb->d) : 0.0f;
        float ud = lane == 0 ? cuda_fp16_to_fp32(ub->d) : 0.0f;
        gd = __shfl_sync(0xffffffffu, gd, 0);
        ud = __shfl_sync(0xffffffffu, ud, 0);
        gate += gd * (float)gb->qs[lane] * xv;
        up += ud * (float)ub->qs[lane] * xv;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if (lane == 0)
        out[row] = (gate / (1.0f + __expf(-gate))) * up;
}

static __global__ void rmsnorm_kernel(float *out, const float *x,
                                      const float *weight, int n, float eps) {
    int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        ss += x[i] * x[i];
    extern __shared__ float scratch[];
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, offset);
    if (lane == 0) scratch[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        float total = lane < n_warps ? scratch[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffffu, total, offset);
        if (lane == 0) scratch[0] = total;
    }
    __syncthreads();
    float scale = rsqrtf(scratch[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = x[i] * scale * weight[i];
}

static __global__ void residual_rmsnorm_kernel(float *x, const float *r,
                                               float *out,
                                               const float *weight,
                                               int n, float eps) {
    int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[i] + r[i];
        x[i] = v;
        ss += v * v;
    }
    extern __shared__ float scratch[];
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, offset);
    if (lane == 0) scratch[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        float total = lane < n_warps ? scratch[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffffu, total, offset);
        if (lane == 0) scratch[0] = total;
    }
    __syncthreads();
    float scale = rsqrtf(scratch[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = x[i] * scale * weight[i];
}

static __global__ void copy_kernel(const float *src, float *dst,
                                   int src_off, int dst_off, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[dst_off + i] = src[src_off + i];
}

static __global__ void bias_add_kernel(float *x, const float *bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += bias[i];
}

static __global__ void bias_add_copy_kernel(float *x, float *dst,
                                            const float *bias, int src_off,
                                            int dst_off, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[src_off + i] + bias[i];
    x[src_off + i] = v;
    dst[dst_off + i] = v;
}

static __global__ void bias_rope_copy_kernel(float *x, float *dst,
                                             const float *bias,
                                             const float *freq,
                                             int dst_off, int n_heads,
                                             int head_size, int pos,
                                             int rope_dims) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= n_heads) return;
    int base = h * head_size;

    for (int i = tid; i < head_size; i += blockDim.x) {
        if (i >= rope_dims) {
            float v = x[base + i] + bias[base + i];
            x[base + i] = v;
            dst[dst_off + base + i] = v;
        }
    }
    for (int i = tid * 2; i + 1 < rope_dims; i += blockDim.x * 2) {
        float angle = (float)pos * freq[i / 2];
        float s, c;
        __sincosf(angle, &s, &c);
        float x0 = x[base + i] + bias[base + i];
        float x1 = x[base + i + 1] + bias[base + i + 1];
        float y0 = x0 * c - x1 * s;
        float y1 = x0 * s + x1 * c;
        x[base + i] = y0;
        x[base + i + 1] = y1;
        dst[dst_off + base + i] = y0;
        dst[dst_off + base + i + 1] = y1;
    }
}

static __global__ void residual_add_kernel(float *x, const float *r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += r[i];
}

static __global__ void activation_gate_kernel(float *x, const float *aux,
                                              int n, int kind) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (kind == BN_GPU_CODE_RELU2_GATE) {
        float r = v > 0.0f ? v : 0.0f;
        x[i] = r * r * aux[i];
    } else if (kind == BN_GPU_CODE_SIGMOID_GATE) {
        x[i] = aux[i] / (1.0f + __expf(-v));
    } else {
        x[i] = (v / (1.0f + __expf(-v))) * aux[i];
    }
}

static __global__ void activation_kernel(float *x, int n, int kind) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (kind == BN_GPU_CODE_RELU2_ACT) {
        float r = v > 0.0f ? v : 0.0f;
        x[i] = r * r;
    } else {
        x[i] = v / (1.0f + __expf(-v));
    }
}

static __global__ void rope_kernel(float *q, float *k, const float *freq,
                                   int n_heads, int head_size, int pos,
                                   int rope_dims, int n_kv_heads,
                                   uint32_t kv_cache_off) {
    int h = blockIdx.x;
    int i = threadIdx.x * 2;
    if (h >= n_heads || i + 1 >= rope_dims) return;
    float angle = (float)pos * freq[i / 2];
    float s, c;
    __sincosf(angle, &s, &c);
    float *qh = q + (size_t)h * head_size;
    float x0 = qh[i], x1 = qh[i + 1];
    qh[i] = x0 * c - x1 * s;
    qh[i + 1] = x0 * s + x1 * c;
    if (k && h < n_kv_heads) {
        float *kh = k + kv_cache_off + (size_t)h * head_size;
        x0 = kh[i]; x1 = kh[i + 1];
        kh[i] = x0 * c - x1 * s;
        kh[i + 1] = x0 * s + x1 * c;
    }
}

static __global__ void gqa_scores_kernel(float *att, const float *q,
                                         const void *key_cache,
                                         int n_heads, int head_size,
                                         int n_kv, int kv_mul, int kv_dim,
                                         int seq_len, uint32_t loff,
                                         float scale, int kv_f16) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;
    if (h >= n_heads || t >= n_kv) return;
    int kh = h / kv_mul;
    const float *qh = q + (size_t)h * head_size;
    size_t koff = (size_t)loff + (size_t)t * kv_dim +
                  (size_t)kh * head_size;
    float sum = 0.0f;
    for (int i = tid; i < head_size; i += blockDim.x)
        sum += qh[i] * cuda_kv_load(key_cache, koff + (size_t)i, kv_f16);
    extern __shared__ float scratch[];
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        att[(size_t)h * seq_len + t] = scratch[0] * scale;
}

static __global__ void softmax_kernel(float *att, int n_heads, int n_kv,
                                      int seq_len) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= n_heads) return;
    float *row = att + (size_t)h * seq_len;
    float maxv = -INFINITY;
    for (int i = tid; i < n_kv; i += blockDim.x)
        maxv = fmaxf(maxv, row[i]);
    extern __shared__ float scratch[];
    scratch[tid] = maxv;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid],
                                               scratch[tid + stride]);
        __syncthreads();
    }
    maxv = scratch[0];
    float sum = 0.0f;
    for (int i = tid; i < n_kv; i += blockDim.x) {
        float e = __expf(row[i] - maxv);
        row[i] = e;
        sum += e;
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float inv = 1.0f / scratch[0];
    for (int i = tid; i < n_kv; i += blockDim.x)
        row[i] *= inv;
}

static __global__ void gqa_combine_kernel(float *out, const float *att,
                                          const void *value_cache,
                                          int n_heads, int head_size,
                                          int n_kv, int kv_mul, int kv_dim,
                                          int seq_len, uint32_t loff,
                                          int kv_f16) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    if (h >= n_heads || i >= head_size) return;
    int vh = h / kv_mul;
    const float *row = att + (size_t)h * seq_len;
    float sum = 0.0f;
    for (int t = 0; t < n_kv; t++) {
        size_t voff = (size_t)loff + (size_t)t * kv_dim +
                      (size_t)vh * head_size;
        sum += row[t] * cuda_kv_load(value_cache, voff + (size_t)i, kv_f16);
    }
    out[(size_t)h * head_size + i] = sum;
}

static __global__ void flash_attention_kernel(float *out, const float *q,
                                              const void *key_cache,
                                              const void *value_cache,
                                              int n_heads, int head_size,
                                              int n_kv, int kv_mul,
                                              int kv_dim, int seq_len,
                                              uint32_t loff,
                                              float inv_sqrt_hs,
                                              int kv_f16) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= n_heads) return;
    int kvh = h / kv_mul;
    const float *qh = q + (size_t)h * head_size;
    extern __shared__ float shared[];
    float *scores = shared;
    float *scratch = shared + n_kv;

    for (int t = tid; t < n_kv; t += blockDim.x) {
        size_t koff = (size_t)loff + (size_t)t * kv_dim +
                      (size_t)kvh * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
            score += qh[i] * cuda_kv_load(key_cache, koff + (size_t)i,
                                          kv_f16);
        scores[t] = score * inv_sqrt_hs;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < n_kv; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    float max_score = cuda_block_reduce_max_all(local_max, scratch);

    float local_sum = 0.0f;
    for (int t = tid; t < n_kv; t += blockDim.x) {
        float p = __expf(scores[t] - max_score);
        scores[t] = p;
        local_sum += p;
    }
    float inv_sum = 1.0f / cuda_block_reduce_sum_all(local_sum, scratch);

    for (int i = tid; i < head_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < n_kv; t++) {
            size_t voff = (size_t)loff + (size_t)t * kv_dim +
                          (size_t)kvh * head_size;
            sum += scores[t] * inv_sum *
                   cuda_kv_load(value_cache, voff + (size_t)i, kv_f16);
        }
        out[(size_t)h * head_size + i] = sum;
    }
}

static __global__ void flash_attention_rope_q_kernel(
    float *out, const float *q, const void *key_cache,
    const void *value_cache, const float *freq, const float *bias,
    int n_heads, int head_size, int n_kv, int kv_mul, int kv_dim,
    int seq_len, uint32_t loff, float inv_sqrt_hs, int pos,
    int rope_dims, int kv_f16) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= n_heads) return;
    int kvh = h / kv_mul;
    const float *qh = q + (size_t)h * head_size;
    extern __shared__ float shared[];
    float *scores = shared;
    float *scratch = shared + n_kv;
    float *qrot = scratch + blockDim.x;

    for (int i = tid; i < head_size; i += blockDim.x) {
        if (i < rope_dims) {
            if ((i & 1) == 0 && i + 1 < rope_dims) {
                float angle = (float)pos * freq[i / 2];
                float s, c;
                __sincosf(angle, &s, &c);
                float x0 = qh[i];
                float x1 = qh[i + 1];
                if (bias) {
                    const float *bh = bias + (size_t)h * head_size;
                    x0 += bh[i];
                    x1 += bh[i + 1];
                }
                qrot[i] = x0 * c - x1 * s;
                qrot[i + 1] = x0 * s + x1 * c;
            }
        } else {
            qrot[i] = qh[i] + (bias ? bias[(size_t)h * head_size + i]
                                    : 0.0f);
        }
    }
    __syncthreads();

    for (int t = tid; t < n_kv; t += blockDim.x) {
        size_t koff = (size_t)loff + (size_t)t * kv_dim +
                      (size_t)kvh * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
            score += qrot[i] * cuda_kv_load(key_cache, koff + (size_t)i,
                                            kv_f16);
        scores[t] = score * inv_sqrt_hs;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < n_kv; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    float max_score = cuda_block_reduce_max_all(local_max, scratch);

    float local_sum = 0.0f;
    for (int t = tid; t < n_kv; t += blockDim.x) {
        float p = __expf(scores[t] - max_score);
        scores[t] = p;
        local_sum += p;
    }
    float inv_sum = 1.0f / cuda_block_reduce_sum_all(local_sum, scratch);

    for (int i = tid; i < head_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < n_kv; t++) {
            size_t voff = (size_t)loff + (size_t)t * kv_dim +
                          (size_t)kvh * head_size;
            sum += scores[t] * inv_sum *
                   cuda_kv_load(value_cache, voff + (size_t)i, kv_f16);
        }
        out[(size_t)h * head_size + i] = sum;
    }
}

static __global__ void flash_attention_rope_q_runtime_kernel(
    float *out, const float *q, const void *key_cache,
    const void *value_cache, const float *freq, const float *bias,
    int n_heads, int head_size, int kv_mul, int kv_dim,
    int seq_len, uint32_t loff, float inv_sqrt_hs,
    int rope_dims, int kv_f16, const BnCudaRuntimeParams *runtime) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= n_heads) return;
    int n_kv = runtime ? runtime->n_kv : 1;
    int pos = runtime ? runtime->pos : 0;
    int kvh = h / kv_mul;
    const float *qh = q + (size_t)h * head_size;
    extern __shared__ float shared[];
    float *scores = shared;
    float *scratch = shared + n_kv;
    float *qrot = scratch + blockDim.x;

    for (int i = tid; i < head_size; i += blockDim.x) {
        if (i < rope_dims) {
            if ((i & 1) == 0 && i + 1 < rope_dims) {
                float angle = (float)pos * freq[i / 2];
                float s, c;
                __sincosf(angle, &s, &c);
                float x0 = qh[i];
                float x1 = qh[i + 1];
                if (bias) {
                    const float *bh = bias + (size_t)h * head_size;
                    x0 += bh[i];
                    x1 += bh[i + 1];
                }
                qrot[i] = x0 * c - x1 * s;
                qrot[i + 1] = x0 * s + x1 * c;
            }
        } else {
            qrot[i] = qh[i] + (bias ? bias[(size_t)h * head_size + i]
                                    : 0.0f);
        }
    }
    __syncthreads();

    for (int t = tid; t < n_kv; t += blockDim.x) {
        size_t koff = (size_t)loff + (size_t)t * kv_dim +
                      (size_t)kvh * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
            score += qrot[i] * cuda_kv_load(key_cache, koff + (size_t)i,
                                            kv_f16);
        scores[t] = score * inv_sqrt_hs;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < n_kv; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    float max_score = cuda_block_reduce_max_all(local_max, scratch);

    float local_sum = 0.0f;
    for (int t = tid; t < n_kv; t += blockDim.x) {
        float p = __expf(scores[t] - max_score);
        scores[t] = p;
        local_sum += p;
    }
    float inv_sum = 1.0f / cuda_block_reduce_sum_all(local_sum, scratch);

    for (int i = tid; i < head_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < n_kv; t++) {
            size_t voff = (size_t)loff + (size_t)t * kv_dim +
                          (size_t)kvh * head_size;
            sum += scores[t] * inv_sum *
                   cuda_kv_load(value_cache, voff + (size_t)i, kv_f16);
        }
        out[(size_t)h * head_size + i] = sum;
    }
}

static __global__ void matvec_batch_kernel(float *out,
                                           const BnCudaDeviceOp *ops,
                                           int n_ops,
                                           const float *x,
                                           int x_cols) {
    int global_row = blockIdx.x;
    int op_idx = 0;
    int row = global_row;
    while (op_idx < n_ops && row >= ops[op_idx].rows) {
        row -= ops[op_idx].rows;
        op_idx++;
    }
    if (op_idx >= n_ops) return;

    const BnCudaDeviceOp op = ops[op_idx];
    int tid = threadIdx.x;
    const void *wdata = op.wdata;
    int cols = x_cols;
    int type = op.type;
    float sum = 0.0f;

    if (type == BN_GGUF_TENSOR_F32) {
        const float *w = (const float *)wdata + (size_t)row * cols;
        for (int c = tid; c < cols; c += blockDim.x)
            sum += w[c] * x[c];
    } else if (type == BN_GGUF_TENSOR_F16) {
        const uint16_t *w = (const uint16_t *)wdata + (size_t)row * cols;
        for (int c = tid; c < cols; c += blockDim.x)
            sum += cuda_fp16_to_fp32(w[c]) * x[c];
    } else if (type == BN_GGUF_TENSOR_Q8_0) {
        const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)wdata;
        int n_bpr = cols / 32;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / 32;
            int i = c & 31;
            const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_bpr + b];
            float d = cuda_fp16_to_fp32(blk->d);
            sum += d * (float)blk->qs[i] * x[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q4_0) {
        const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)wdata;
        int n_bpr = cols / 32;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / 32;
            int i = c & 31;
            const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_bpr + b];
            float d = cuda_fp16_to_fp32(blk->d);
            uint8_t qs = blk->qs[i & 15];
            int q = i < 16 ? (int)(qs & 15) - 8 : (int)(qs >> 4) - 8;
            sum += d * (float)q * x[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q5_0) {
        const BnBlockQ5_0 *blocks = (const BnBlockQ5_0 *)wdata;
        int n_bpr = cols / 32;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / 32;
            int i = c & 31;
            const BnBlockQ5_0 *blk = &blocks[(size_t)row * n_bpr + b];
            float d = cuda_fp16_to_fp32(blk->d);
            uint32_t qh = (uint32_t)blk->qh[0] |
                          ((uint32_t)blk->qh[1] << 8) |
                          ((uint32_t)blk->qh[2] << 16) |
                          ((uint32_t)blk->qh[3] << 24);
            uint8_t qs = blk->qs[i & 15];
            int q = i < 16
                ? (int)((qs & 15) | (((qh >> i) & 1u) << 4)) - 16
                : (int)((qs >> 4) | (((qh >> i) & 1u) << 4)) - 16;
            sum += d * (float)q * x[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q4_K) {
        const BnBlockQ4K *blocks = (const BnBlockQ4K *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            const BnBlockQ4K *blk = &blocks[(size_t)row * n_bpr + b];
            sum += cuda_q4k_value(blk, i) * x[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q5_K) {
        const BnBlockQ5K *blocks = (const BnBlockQ5K *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            const BnBlockQ5K *blk = &blocks[(size_t)row * n_bpr + b];
            sum += cuda_q5k_value(blk, i) * x[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q6_K) {
        const BnBlockQ6K *blocks = (const BnBlockQ6K *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            const BnBlockQ6K *blk = &blocks[(size_t)row * n_bpr + b];
            sum += cuda_q6k_value(blk, i) * x[c];
        }
    } else if (type == BN_GGUF_TENSOR_Q8_K) {
        const BnBlockQ8K *blocks = (const BnBlockQ8K *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            const BnBlockQ8K *blk = &blocks[(size_t)row * n_bpr + b];
            sum += blk->d * (float)blk->qs[i] * x[c];
        }
    }

    extern __shared__ float scratch[];
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        out[op.out_offset + (size_t)row] = scratch[0];
}

static __global__ void ffn_activation_kernel(float *out,
                                             const float *gate_up,
                                             int hidden_dim,
                                             int act_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_dim) return;
    float gate = gate_up[i];
    float up = gate_up[(size_t)hidden_dim + i];
    if (act_type == 0) {
        float silu = gate / (1.0f + __expf(-gate));
        out[i] = silu * up;
    } else {
        out[i] = gate * up;
    }
}

static int cuda_type_supported(int type) {
    static int init = 0;
    static int disable_matvec = 0;
    static int disable_q8_0 = 0;
    static int disable_q5_0 = 0;
    static int disable_q4_k = 0;
    static int disable_q5_k = 0;
    static int disable_q6_k = 0;
    static int disable_q8_k = 0;
    if (!init) {
        disable_matvec = getenv("BN_CUDA_DISABLE_MATVEC") != NULL;
        disable_q8_0 = getenv("BN_CUDA_DISABLE_Q8_0") != NULL;
        disable_q5_0 = getenv("BN_CUDA_DISABLE_Q5_0") != NULL;
        disable_q4_k = getenv("BN_CUDA_DISABLE_Q4_K") != NULL;
        disable_q5_k = getenv("BN_CUDA_DISABLE_Q5_K") != NULL;
        disable_q6_k = getenv("BN_CUDA_DISABLE_Q6_K") != NULL;
        disable_q8_k = getenv("BN_CUDA_DISABLE_Q8_K") != NULL;
        init = 1;
    }
    if (disable_matvec)
        return 0;
    if (type == BN_GGUF_TENSOR_Q8_0 && disable_q8_0)
        return 0;
    if (type == BN_GGUF_TENSOR_Q5_0 && disable_q5_0)
        return 0;
    if (type == BN_GGUF_TENSOR_Q4_K && disable_q4_k)
        return 0;
    if (type == BN_GGUF_TENSOR_Q5_K && disable_q5_k)
        return 0;
    if (type == BN_GGUF_TENSOR_Q6_K && disable_q6_k)
        return 0;
    if (type == BN_GGUF_TENSOR_Q8_K && disable_q8_k)
        return 0;
    return type == BN_GGUF_TENSOR_F32 || type == BN_GGUF_TENSOR_F16 ||
           type == BN_GGUF_TENSOR_Q8_0 || type == BN_GGUF_TENSOR_Q4_0 ||
           type == BN_GGUF_TENSOR_Q5_0 || type == BN_GGUF_TENSOR_Q4_K ||
           type == BN_GGUF_TENSOR_Q5_K ||
           type == BN_GGUF_TENSOR_Q6_K || type == BN_GGUF_TENSOR_Q8_K;
}

static int cuda_ensure_scratch(BnCudaCtx *ctx, size_t x_bytes,
                               size_t out_bytes) {
    if (!ctx) return -1;
    if (cuda_ctx_set_device(ctx) != 0) return -1;
    if (x_bytes > ctx->d_x_bytes) {
        if (ctx->d_x) cudaFree(ctx->d_x);
        ctx->d_x = NULL;
        ctx->d_x_bytes = 0;
        cudaError_t err = cudaMalloc(&ctx->d_x, x_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] x scratch alloc failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        ctx->d_x_bytes = x_bytes;
    }
    if (out_bytes > ctx->d_out_bytes) {
        if (ctx->d_out) cudaFree(ctx->d_out);
        ctx->d_out = NULL;
        ctx->d_out_bytes = 0;
        cudaError_t err = cudaMalloc(&ctx->d_out, out_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] output scratch alloc failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        ctx->d_out_bytes = out_bytes;
    }
    return 0;
}

static int cuda_ensure_ops(BnCudaCtx *ctx, size_t ops_bytes) {
    if (!ctx) return -1;
    if (cuda_ctx_set_device(ctx) != 0) return -1;
    if (ops_bytes > ctx->d_ops_bytes) {
        if (ctx->d_ops) cudaFree(ctx->d_ops);
        ctx->d_ops = NULL;
        ctx->d_ops_bytes = 0;
        cudaError_t err = cudaMalloc(&ctx->d_ops, ops_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] op scratch alloc failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        ctx->d_ops_bytes = ops_bytes;
    }
    return 0;
}

static int cuda_ensure_host_out(BnCudaCtx *ctx, size_t bytes) {
    if (!ctx) return -1;
    if (bytes <= ctx->h_out_bytes) return 0;
    float *next = (float *)realloc(ctx->h_out, bytes);
    if (!next) return -1;
    ctx->h_out = next;
    ctx->h_out_bytes = bytes;
    return 0;
}

static void cuda_free_activations(void *vctx) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    if (!ctx) return;
    if (cuda_ctx_set_device(ctx) != 0) return;
    for (int i = 0; i < BN_GPU_VALUE_COUNT; i++) {
        if (ctx->act_bufs[i]) {
            cudaFree(ctx->act_bufs[i]);
            ctx->act_bufs[i] = NULL;
        }
        ctx->act_sizes[i] = 0;
    }
}

static int cuda_alloc_activation(BnCudaCtx *ctx, int idx, size_t bytes) {
    if (!ctx || idx < 0 || idx >= BN_GPU_VALUE_COUNT) return -1;
    if (bytes == 0) return 0;
    if (cuda_ctx_set_device(ctx) != 0) return -1;
    size_t aligned = (bytes + 255u) & ~(size_t)255u;
    cudaError_t err = cudaMalloc(&ctx->act_bufs[idx], aligned);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] activation alloc failed idx=%d bytes=%zu: %s\n",
                idx, aligned, cudaGetErrorString(err));
        return -1;
    }
    ctx->act_sizes[idx] = aligned;
    return 0;
}

static int cuda_init_activations(void *vctx, const void *config_ptr) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    const BnConfig *c = (const BnConfig *)config_ptr;
    if (!ctx || !c) return -1;
    if (cuda_ctx_set_device(ctx) != 0) return -1;

    cuda_free_activations(ctx);
    ctx->kv_f16 = c->kv_f16;

    int n_attn = (c->full_attn_interval > 0)
        ? c->n_layers / c->full_attn_interval
        : c->n_layers;
    int q_dim = c->n_heads * c->head_size;
    int xb_size = q_dim > c->dim ? q_dim : c->dim;
    int hb_dim = c->hidden_dim;
    if (c->moe_intermediate_size > hb_dim)
        hb_dim = c->moe_intermediate_size;

    size_t sizes[BN_GPU_VALUE_COUNT] = {0};
    sizes[BN_GPU_VALUE_X] = (size_t)c->dim * sizeof(float);
    sizes[BN_GPU_VALUE_XB] = (size_t)xb_size * sizeof(float);
    sizes[BN_GPU_VALUE_XB2] = (size_t)c->dim * sizeof(float);
    sizes[BN_GPU_VALUE_Q] = (size_t)q_dim * sizeof(float);
    sizes[BN_GPU_VALUE_HB] = (size_t)hb_dim * sizeof(float);
    sizes[BN_GPU_VALUE_HB2] = (size_t)hb_dim * sizeof(float);
    size_t kv_elem_size = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
    sizes[BN_GPU_VALUE_KEY_CACHE] =
        (size_t)n_attn * c->seq_len * c->kv_dim * kv_elem_size;
    sizes[BN_GPU_VALUE_VALUE_CACHE] =
        (size_t)n_attn * c->seq_len * c->kv_dim * kv_elem_size;
    sizes[BN_GPU_VALUE_ATT] =
        (size_t)c->n_heads * c->seq_len * sizeof(float);
    sizes[BN_GPU_VALUE_LOGITS] =
        (size_t)c->vocab_size * sizeof(float);
    sizes[BN_GPU_VALUE_ROPE_FREQ] =
        (size_t)(c->head_size / 2) * sizeof(float);
    sizes[BN_GPU_VALUE_SCRATCH] = (size_t)xb_size * sizeof(float);
    {
        size_t qkv_size = (size_t)(q_dim + 2 * c->kv_dim) * sizeof(float);
        size_t gated_q_size = (size_t)(2 * q_dim) * sizeof(float);
        sizes[BN_GPU_VALUE_QKV] = qkv_size > gated_q_size
            ? qkv_size
            : gated_q_size;
    }
    if (c->moe_intermediate_size > 0) {
        sizes[BN_GPU_VALUE_MOE_HB] =
            (size_t)c->moe_intermediate_size * sizeof(float);
        sizes[BN_GPU_VALUE_MOE_HB2] =
            (size_t)c->moe_intermediate_size * sizeof(float);
        sizes[BN_GPU_VALUE_MOE_OUT] =
            (size_t)c->dim * sizeof(float);
    }
    if (c->full_attn_interval > 0 && c->ssm_inner_size > 0) {
        int n_ssm = c->n_layers - n_attn;
        int num_v_heads = c->ssm_time_step_rank;
        int head_k_dim = c->ssm_state_size;
        int head_v_dim = c->ssm_inner_size /
            (num_v_heads > 0 ? num_v_heads : 1);
        int key_dim = c->ssm_group_count * head_k_dim;
        int value_dim = c->ssm_inner_size;
        int qkv_dim = key_dim * 2 + value_dim;
        int kern = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;
        sizes[BN_GPU_VALUE_SSM_STATE] =
            (size_t)n_ssm * num_v_heads * head_k_dim * head_v_dim *
            sizeof(float);
        sizes[BN_GPU_VALUE_SSM_CONV_STATE] =
            (size_t)n_ssm * (kern - 1) * qkv_dim * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_QKV] = (size_t)qkv_dim * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_Z] = (size_t)value_dim * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_ALPHA] =
            (size_t)num_v_heads * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_BETA] =
            (size_t)num_v_heads * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_V] = (size_t)value_dim * sizeof(float);
    }

    for (int i = 0; i < BN_GPU_VALUE_COUNT; i++) {
        if (cuda_alloc_activation(ctx, i, sizes[i]) != 0) {
            cuda_free_activations(ctx);
            return -1;
        }
    }

    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : c->head_size;
    int half = rope_dims / 2;
    if (half > 0 && ctx->act_bufs[BN_GPU_VALUE_ROPE_FREQ]) {
        float *freq = (float *)malloc((size_t)half * sizeof(float));
        if (!freq) {
            cuda_free_activations(ctx);
            return -1;
        }
        for (int i = 0; i < half; i++)
            freq[i] = 1.0f / powf(c->rope_theta,
                                  (float)(2 * i) / (float)rope_dims);
        cudaError_t err = cudaMemcpy(ctx->act_bufs[BN_GPU_VALUE_ROPE_FREQ],
                                     freq, (size_t)half * sizeof(float),
                                     cudaMemcpyHostToDevice);
        free(freq);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] rope frequency upload failed: %s\n",
                    cudaGetErrorString(err));
            cuda_free_activations(ctx);
            return -1;
        }
    }

    return 0;
}

static int cuda_write_activation(void *vctx, int buf_idx, const void *data,
                                 size_t size, size_t offset) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    if (!ctx || !data || buf_idx < 0 || buf_idx >= BN_GPU_VALUE_COUNT)
        return -1;
    if (cuda_ctx_set_device(ctx) != 0) return -1;
    if (!ctx->act_bufs[buf_idx] || offset + size > ctx->act_sizes[buf_idx])
        return -1;
    cudaError_t err = cudaMemcpy((char *)ctx->act_bufs[buf_idx] + offset,
                                 data, size, cudaMemcpyHostToDevice);
    return err == cudaSuccess ? 0 : -1;
}

static int cuda_read_activation(void *vctx, int buf_idx, void *out,
                                size_t size, size_t offset) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    if (!ctx || !out || buf_idx < 0 || buf_idx >= BN_GPU_VALUE_COUNT)
        return -1;
    if (cuda_ctx_set_device(ctx) != 0) return -1;
    if (!ctx->act_bufs[buf_idx] || offset + size > ctx->act_sizes[buf_idx])
        return -1;
    cudaError_t err = cudaMemcpy(out,
                                 (char *)ctx->act_bufs[buf_idx] + offset,
                                 size, cudaMemcpyDeviceToHost);
    return err == cudaSuccess ? 0 : -1;
}

static void *cuda_buffer_create(void *vctx, const void *data, size_t size,
                                int type, int rows, int cols) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    if (cuda_ctx_set_device(ctx) != 0) return NULL;
    if (!data || size == 0) return NULL;

    BnCudaBuffer *buf = (BnCudaBuffer *)calloc(1, sizeof(BnCudaBuffer));
    if (!buf) return NULL;
    buf->size = size;
    buf->type = type;
    buf->rows = rows;
    buf->cols = cols;

    cudaError_t err = cudaMalloc(&buf->data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] weight alloc failed: %s\n",
                cudaGetErrorString(err));
        free(buf);
        return NULL;
    }
    err = cudaMemcpy(buf->data, data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] weight upload failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(buf->data);
        free(buf);
        return NULL;
    }
    return buf;
}

static void cuda_buffer_destroy(void *vctx, void *buffer) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    if (cuda_ctx_set_device(ctx) != 0) return;
    BnCudaBuffer *buf = (BnCudaBuffer *)buffer;
    if (!buf) return;
    if (buf->data) cudaFree(buf->data);
    free(buf);
}

static int cuda_matvec(void *vctx, float *out, void *W_buf, const float *x,
                       int rows, int cols, int type) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *w = (BnCudaBuffer *)W_buf;
    if (!out || !w || !w->data || !x || rows <= 0 || cols <= 0)
        return -1;
    if (!cuda_type_supported(type))
        return -1;

    size_t x_bytes = (size_t)cols * sizeof(float);
    size_t out_bytes = (size_t)rows * sizeof(float);
    if (cuda_ensure_scratch(ctx, x_bytes, out_bytes) != 0) return -1;
    cudaError_t err = cudaMemcpy(ctx->d_x, x, x_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] matvec input upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int threads = 256;
    if (type == BN_GGUF_TENSOR_Q5_0 && (cols & 31) == 0 &&
        getenv("BN_CUDA_ENABLE_Q5_MATVEC4")) {
        q5_0_matvec4_kernel<<<(rows + 3) / 4, threads,
            (size_t)threads * sizeof(float) * 4>>>(
            ctx->d_out, (const BnBlockQ5_0 *)w->data, ctx->d_x, NULL,
            rows, cols, 0);
    } else if (type == BN_GGUF_TENSOR_Q5_0 && (cols & 31) == 0 &&
               getenv("BN_CUDA_ENABLE_Q5_WARP")) {
        int warps = threads / 32;
        int blocks = (rows + warps - 1) / warps;
        q5_0_matvec_warp_kernel<<<blocks, threads>>>(
            ctx->d_out, (const BnBlockQ5_0 *)w->data, ctx->d_x, NULL,
            rows, cols, 0);
    } else if (type == BN_GGUF_TENSOR_Q6_K && (cols % BN_QK_K) == 0 &&
               getenv("BN_CUDA_ENABLE_Q6K_WARP")) {
        int warps = threads / 32;
        int blocks = (rows + warps - 1) / warps;
        q6k_matvec_warp_kernel<<<blocks, threads>>>(
            ctx->d_out, (const BnBlockQ6K *)w->data, ctx->d_x, NULL,
            rows, cols, 0);
    } else if (type == BN_GGUF_TENSOR_Q8_0 && rows >= 16384 &&
               (cols & 31) == 0) {
        int warps = threads / 32;
        int blocks = ((rows + 3) / 4 + warps - 1) / warps;
        q8_0_matvec4_warp_kernel<<<blocks, threads>>>(
            ctx->d_out, (const BnBlockQ8_0 *)w->data, ctx->d_x, rows,
            cols, 0);
    } else {
        dim3 grid(rows, 1, 1);
        matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
            ctx->d_out, w->data, ctx->d_x, NULL, rows, cols, type, 0);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] matvec launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(out, ctx->d_out, out_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] matvec output readback failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }

    return 0;
}

static int cuda_matmul(void *vctx, float *out, void *W_buf, const float *X,
                       int rows, int cols, int n_tokens, int type) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *w = (BnCudaBuffer *)W_buf;
    if (!out || !w || !w->data || !X || rows <= 0 || cols <= 0 ||
        n_tokens <= 0)
        return -1;
    if (!cuda_type_supported(type))
        return -1;

    size_t x_bytes = (size_t)n_tokens * cols * sizeof(float);
    size_t out_bytes = (size_t)n_tokens * rows * sizeof(float);
    if (cuda_ensure_scratch(ctx, x_bytes, out_bytes) != 0) return -1;
    cudaError_t err = cudaMemcpy(ctx->d_x, X, x_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] matmul input upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int threads = 256;
    dim3 grid(rows, n_tokens, 1);
    matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
        ctx->d_out, w->data, ctx->d_x, NULL, rows, cols, type, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] matmul launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(out, ctx->d_out, out_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] matmul output readback failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    return 0;
}

static int cuda_matvec_batch(void *vctx, const BnGPUMatvecOp *ops, int n_ops,
                             const float *x, int x_cols) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    if (getenv("BN_CUDA_DISABLE_MATVEC_BATCH")) return -1;
    if (!ctx || !ops || !x || n_ops <= 0) return -1;
    size_t total_rows = 0;
    for (int i = 0; i < n_ops; i++) {
        BnCudaBuffer *w = (BnCudaBuffer *)ops[i].W_buf;
        if (!ops[i].out || !w || !w->data || ops[i].cols != x_cols ||
            ops[i].rows <= 0 || !cuda_type_supported(ops[i].type))
            return -1;
        total_rows += (size_t)ops[i].rows;
    }

    size_t x_bytes = (size_t)x_cols * sizeof(float);
    size_t out_bytes = total_rows * sizeof(float);
    if (cuda_ensure_scratch(ctx, x_bytes, out_bytes) != 0) return -1;
    size_t ops_bytes = (size_t)n_ops * sizeof(BnCudaDeviceOp);
    if (cuda_ensure_ops(ctx, ops_bytes) != 0) return -1;

    BnCudaDeviceOp host_ops_inline[16];
    BnCudaDeviceOp *host_ops = host_ops_inline;
    BnCudaDeviceOp *heap_ops = NULL;
    if (n_ops > (int)(sizeof(host_ops_inline) / sizeof(host_ops_inline[0]))) {
        heap_ops = (BnCudaDeviceOp *)malloc(ops_bytes);
        if (!heap_ops) return -1;
        host_ops = heap_ops;
    }
    size_t row_offset = 0;
    for (int i = 0; i < n_ops; i++) {
        BnCudaBuffer *w = (BnCudaBuffer *)ops[i].W_buf;
        host_ops[i] = (BnCudaDeviceOp){
            w->data,
            row_offset,
            ops[i].rows,
            ops[i].cols,
            ops[i].type,
        };
        row_offset += (size_t)ops[i].rows;
    }

    cudaError_t err = cudaMemcpy(ctx->d_x, x, x_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] batch input upload failed: %s\n",
                cudaGetErrorString(err));
        free(heap_ops);
        return -1;
    }
    err = cudaMemcpy(ctx->d_ops, host_ops, ops_bytes, cudaMemcpyHostToDevice);
    free(heap_ops);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] batch op upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int threads = 256;
    matvec_batch_kernel<<<(unsigned)total_rows, threads,
        (size_t)threads * sizeof(float)>>>(
        ctx->d_out, (const BnCudaDeviceOp *)ctx->d_ops, n_ops,
        ctx->d_x, x_cols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] batch matvec launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    if (n_ops == 1) {
        err = cudaMemcpy(ops[0].out, ctx->d_out, out_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] batch output readback failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    } else {
        if (cuda_ensure_host_out(ctx, out_bytes) != 0) return -1;
        err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] batch output readback failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        row_offset = 0;
        for (int i = 0; i < n_ops; i++) {
            size_t bytes = (size_t)ops[i].rows * sizeof(float);
            memcpy(ops[i].out, ctx->h_out + row_offset, bytes);
            row_offset += (size_t)ops[i].rows;
        }
    }
    return 0;
}

static int cuda_dense_ffn(void *vctx, float *out,
                          void *gate_buf, void *up_buf, void *down_buf,
                          const float *x, int dim, int hidden_dim,
                          int gate_type, int up_type, int down_type,
                          int act_type) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *gate = (BnCudaBuffer *)gate_buf;
    BnCudaBuffer *up = (BnCudaBuffer *)up_buf;
    BnCudaBuffer *down = (BnCudaBuffer *)down_buf;
    if (!getenv("BN_CUDA_ENABLE_DENSE_FFN"))
        return -1;
    if (!ctx || !out || !gate || !up || !down || !x ||
        !gate->data || !up->data || !down->data ||
        dim <= 0 || hidden_dim <= 0 || act_type != 0)
        return -1;
    if (gate->rows != hidden_dim || up->rows != hidden_dim ||
        gate->cols != dim || up->cols != dim ||
        down->rows != dim || down->cols != hidden_dim)
        return -1;
    if (!cuda_type_supported(gate_type) || !cuda_type_supported(up_type) ||
        !cuda_type_supported(down_type))
        return -1;

    size_t x_bytes = (size_t)dim * sizeof(float);
    size_t hidden_bytes = (size_t)hidden_dim * sizeof(float);
    size_t gateup_bytes = (size_t)hidden_dim * 2 * sizeof(float);
    size_t out_bytes = (size_t)dim * sizeof(float);
    size_t scratch_x = x_bytes > hidden_bytes ? x_bytes : hidden_bytes;
    size_t scratch_out = gateup_bytes > out_bytes ? gateup_bytes : out_bytes;
    if (cuda_ensure_scratch(ctx, scratch_x, scratch_out) != 0) return -1;
    if (cuda_ensure_ops(ctx, 2 * sizeof(BnCudaDeviceOp)) != 0) return -1;

    BnCudaDeviceOp ops[2] = {
        { gate->data, 0, hidden_dim, dim, gate_type },
        { up->data, (size_t)hidden_dim, hidden_dim, dim, up_type },
    };

    cudaError_t err = cudaMemcpy(ctx->d_x, x, x_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn input upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaMemcpy(ctx->d_ops, ops, 2 * sizeof(BnCudaDeviceOp),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn op upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int threads = 256;
    matvec_batch_kernel<<<(unsigned)(hidden_dim * 2), threads,
        (size_t)threads * sizeof(float)>>>(
        ctx->d_out, (const BnCudaDeviceOp *)ctx->d_ops, 2,
        ctx->d_x, dim);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn gate/up launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int act_blocks = (hidden_dim + threads - 1) / threads;
    ffn_activation_kernel<<<act_blocks, threads>>>(
        ctx->d_x, ctx->d_out, hidden_dim, act_type);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn activation launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    matvec_kernel<<<dim, threads, (size_t)threads * sizeof(float)>>>(
        ctx->d_out, down->data, ctx->d_x, NULL, dim, hidden_dim, down_type,
        0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn down launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn sync failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaMemcpy(out, ctx->d_out, out_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn output readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static float cuda_u32_to_f32(uint32_t bits) {
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float *cuda_act(BnCudaCtx *ctx, int idx) {
    if (!ctx || idx < 0 || idx >= BN_GPU_VALUE_COUNT)
        return NULL;
    return ctx->act_bufs[idx];
}

static const char *cuda_op_name(int code) {
    switch (code) {
    case BN_GPU_CODE_MATVEC: return "matvec";
    case BN_GPU_CODE_MATVEC_SPLIT: return "matvec_split";
    case BN_GPU_CODE_Q4K_MATVEC_SPLIT: return "q4k_split";
    case BN_GPU_CODE_Q8_MATVEC_SPLIT: return "q8_split";
    case BN_GPU_CODE_Q5K_MATVEC_SPLIT: return "q5k_split";
    case BN_GPU_CODE_FUSED_GATEUP_SILU: return "fused_gateup";
    case BN_GPU_CODE_RMSNORM: return "rmsnorm";
    case BN_GPU_CODE_RESIDUAL_RMSNORM: return "residual_rmsnorm";
    case BN_GPU_CODE_COPY: return "copy";
    case BN_GPU_CODE_BIAS_ADD: return "bias_add";
    case BN_GPU_CODE_RESIDUAL_ADD: return "residual_add";
    case BN_GPU_CODE_SILU_GATE: return "silu_gate";
    case BN_GPU_CODE_RELU2_GATE: return "relu2_gate";
    case BN_GPU_CODE_SIGMOID_GATE: return "sigmoid_gate";
    case BN_GPU_CODE_SILU_ACT: return "silu_act";
    case BN_GPU_CODE_RELU2_ACT: return "relu2_act";
    case BN_GPU_CODE_ROPE: return "rope";
    case BN_GPU_CODE_ROPE_QK: return "rope_qk";
    case BN_GPU_CODE_GQA_SCORES: return "gqa_scores";
    case BN_GPU_CODE_SOFTMAX: return "softmax";
    case BN_GPU_CODE_GQA_COMBINE: return "gqa_combine";
    case BN_GPU_CODE_FLASH_ATTN: return "flash_attn";
    default: return "unknown";
    }
}

enum {
    BN_CUDA_PROFILE_QKV_MIXED = 64,
    BN_CUDA_PROFILE_READBACK = 65,
    BN_CUDA_PROFILE_LOGITS = 66,
    BN_CUDA_PROFILE_MAX = 67
};

static const char *cuda_profile_name(int code) {
    if (code == BN_CUDA_PROFILE_QKV_MIXED) return "qkv_mixed";
    if (code == BN_CUDA_PROFILE_READBACK) return "readback";
    if (code == BN_CUDA_PROFILE_LOGITS) return "logits";
    return cuda_op_name(code);
}

static int cuda_op_mentions_buf(const BnGPUOp *op, int buf) {
    if (!op || buf < 0) return 0;
    return op->buf_in == buf || op->buf_out == buf || op->buf_aux == buf;
}

static int cuda_op_writes_buf(const BnGPUOp *op, int buf) {
    if (!op || buf < 0) return 0;
    if (op->buf_out == buf) return 1;
    switch (op->op_code) {
    case BN_GPU_CODE_MATVEC_SPLIT:
    case BN_GPU_CODE_Q4K_MATVEC_SPLIT:
    case BN_GPU_CODE_Q8_MATVEC_SPLIT:
    case BN_GPU_CODE_Q5K_MATVEC_SPLIT:
        return op->buf_aux == buf || (int)op->rows == buf;
    default:
        return 0;
    }
}

static int cuda_op_reads_buf(const BnGPUOp *op, int buf) {
    if (!op || buf < 0) return 0;
    switch (op->op_code) {
    case BN_GPU_CODE_MATVEC:
    case BN_GPU_CODE_MATVEC_SPLIT:
    case BN_GPU_CODE_Q4K_MATVEC_SPLIT:
    case BN_GPU_CODE_Q8_MATVEC_SPLIT:
    case BN_GPU_CODE_Q5K_MATVEC_SPLIT:
    case BN_GPU_CODE_FUSED_GATEUP_SILU:
    case BN_GPU_CODE_RMSNORM:
    case BN_GPU_CODE_COPY:
    case BN_GPU_CODE_BIAS_ADD:
    case BN_GPU_CODE_SILU_ACT:
    case BN_GPU_CODE_RELU2_ACT:
    case BN_GPU_CODE_ROPE:
    case BN_GPU_CODE_GQA_SCORES:
    case BN_GPU_CODE_FLASH_ATTN:
        return op->buf_in == buf;
    case BN_GPU_CODE_RESIDUAL_RMSNORM:
    case BN_GPU_CODE_RESIDUAL_ADD:
    case BN_GPU_CODE_SILU_GATE:
    case BN_GPU_CODE_RELU2_GATE:
    case BN_GPU_CODE_SIGMOID_GATE:
    case BN_GPU_CODE_ROPE_QK:
        return op->buf_in == buf || op->buf_aux == buf;
    default:
        return 0;
    }
}

static int cuda_buf_unused_until_write(const BnGPUOp *ops, int n_ops,
                                       int start, int buf) {
    if (!ops || buf < 0) return 0;
    for (int i = start; i < n_ops; i++) {
        if (cuda_op_reads_buf(&ops[i], buf)) return 0;
        if (cuda_op_writes_buf(&ops[i], buf)) return 1;
    }
    return 1;
}

static int cuda_find_fusable_bias(const BnGPUOp *ops, int n_ops, int start,
                                  int buf, int n) {
    if (!ops || buf < 0 || n <= 0) return -1;
    for (int i = start + 1; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];
        if (op->op_code == BN_GPU_CODE_BIAS_ADD && op->buf_in == buf &&
            (int)op->p[0] == n) {
            BnCudaBuffer *bw = (BnCudaBuffer *)op->W_buf;
            return (bw && bw->data) ? i : -1;
        }
        if (cuda_op_mentions_buf(op, buf))
            return -1;
    }
    return -1;
}

static int cuda_bias_followed_by_copy(const BnGPUOp *ops, int n_ops,
                                      int bias_idx) {
    if (!ops || bias_idx < 0 || bias_idx + 1 >= n_ops) return 0;
    const BnGPUOp *bias = &ops[bias_idx];
    const BnGPUOp *copy = &ops[bias_idx + 1];
    return bias->op_code == BN_GPU_CODE_BIAS_ADD &&
           copy->op_code == BN_GPU_CODE_COPY &&
           copy->buf_in == bias->buf_in &&
           (int)copy->p[0] == 0 &&
           copy->p[2] == bias->p[0];
}

static int cuda_execute(void *vctx, const void *ops_raw, int n_ops,
                        int readback_buf, float *out_host, int out_len) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    const BnGPUOp *ops = (const BnGPUOp *)ops_raw;
    if (!ctx || !ops || n_ops <= 0) return -1;
    if (cuda_ctx_set_device(ctx) != 0) return -1;
    ctx->exec_stream = getenv("BN_CUDA_DISABLE_STREAM_EXEC")
        ? (cudaStream_t)0
        : ctx->stream;

    const int profile = getenv("BN_CUDA_PROFILE") != NULL;
    const int profile_wall = getenv("BN_CUDA_PROFILE_WALL") != NULL;
    static unsigned long long profile_calls = 0;
    static unsigned long long profile_ops[BN_CUDA_PROFILE_MAX] = {0};
    static double profile_ms[BN_CUDA_PROFILE_MAX] = {0.0};
    static unsigned long long wall_calls = 0;
    static unsigned long long wall_ops = 0;
    static unsigned long long wall_launches = 0;
    static unsigned long long wall_launch_by_code[BN_CUDA_PROFILE_MAX] = {0};
    static double wall_ms = 0.0;
    double wall_start = profile_wall ? cuda_wall_ms() : 0.0;
    unsigned long long exec_launches = 0;
    unsigned int exec_launch_by_code[BN_CUDA_PROFILE_MAX] = {0};
    cudaEvent_t ev_start = NULL;
    cudaEvent_t ev_stop = NULL;
    if (profile) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
    }

    if (getenv("BN_CUDA_DUMP_OPS") && n_ops > 10) {
        static int dumped = 0;
        if (!dumped) {
            int limit = n_ops < 256 ? n_ops : 256;
            fprintf(stderr, "[bn:gpu:cuda:ops] n_ops=%d showing=%d\n",
                    n_ops, limit);
            for (int i = 0; i < limit; i++) {
                const BnGPUOp *op = &ops[i];
                fprintf(stderr,
                        "  %03d %-18s type=%d in=%d out=%d aux=%d "
                        "rows=%d cols=%d p0=%u p1=%u p2=%u p3=%u p5=%u\n",
                        i, cuda_op_name(op->op_code), op->type, op->buf_in,
                        op->buf_out, op->buf_aux, op->rows,
                        op->cols, op->p[0], op->p[1], op->p[2], op->p[3],
                        op->p[5]);
            }
            dumped = 1;
        }
    }

    const int threads = 256;
    static int flags_init = 0;
    static int fuse_bias_enabled_flag = 1;
    static int fuse_rope_flash_enabled_flag = 1;
    static int enable_q5_matvec4_flag = 0;
    static int enable_q5_warp_flag = 0;
    static int enable_q6k_warp_flag = 0;
    static int disable_q8_warp_flag = 0;
    static int disable_qkv_mixed_fuse_flag = 0;
    static int qkv_fuse_key_cache_flag = 1;
    static int enable_qkv_kpair_opt_flag = 1;
    static int disable_q5_gateup_warp_flag = 0;
    static int disable_q8_gateup_warp_flag = 0;
    static int enable_bias_rope_flash_fuse_flag = 0;
    static int enable_graph_exec_flag = 0;
    if (!flags_init) {
        fuse_bias_enabled_flag =
            getenv("BN_CUDA_DISABLE_FUSE_BIAS") == NULL;
        fuse_rope_flash_enabled_flag =
            getenv("BN_CUDA_DISABLE_ROPE_FLASH_FUSE") == NULL;
        enable_q5_matvec4_flag =
            getenv("BN_CUDA_ENABLE_Q5_MATVEC4") != NULL;
        enable_q5_warp_flag = getenv("BN_CUDA_ENABLE_Q5_WARP") != NULL;
        enable_q6k_warp_flag = getenv("BN_CUDA_ENABLE_Q6K_WARP") != NULL;
        disable_q8_warp_flag = getenv("BN_CUDA_DISABLE_Q8_WARP") != NULL;
        disable_qkv_mixed_fuse_flag =
            getenv("BN_CUDA_DISABLE_QKV_MIXED_FUSE") != NULL;
        qkv_fuse_key_cache_flag =
            getenv("BN_CUDA_DISABLE_QKV_KCACHE_FUSE") == NULL;
        enable_qkv_kpair_opt_flag =
            getenv("BN_CUDA_DISABLE_QKV_KPAIR_OPT") == NULL;
        disable_q5_gateup_warp_flag =
            getenv("BN_CUDA_DISABLE_Q5_GATEUP_WARP") != NULL;
        disable_q8_gateup_warp_flag =
            getenv("BN_CUDA_DISABLE_Q8_GATEUP_WARP") != NULL;
        enable_bias_rope_flash_fuse_flag =
            getenv("BN_CUDA_ENABLE_BIAS_ROPE_FLASH_FUSE") != NULL;
        enable_graph_exec_flag =
            getenv("BN_CUDA_ENABLE_GRAPH_EXEC") != NULL;
        flags_init = 1;
    }
    const int fuse_bias_enabled = fuse_bias_enabled_flag;
    const int fuse_rope_flash_enabled = fuse_rope_flash_enabled_flag;
    const int enable_q5_matvec4 = enable_q5_matvec4_flag;
    const int enable_q5_warp = enable_q5_warp_flag;
    const int enable_q6k_warp = enable_q6k_warp_flag;
    const int disable_q8_warp = disable_q8_warp_flag;
    const int disable_qkv_mixed_fuse = disable_qkv_mixed_fuse_flag;
    const int qkv_fuse_key_cache = qkv_fuse_key_cache_flag;
    const int enable_qkv_kpair_opt = enable_qkv_kpair_opt_flag;
    const int disable_q5_gateup_warp = disable_q5_gateup_warp_flag;
    const int disable_q8_gateup_warp = disable_q8_gateup_warp_flag;
    const int enable_bias_rope_flash_fuse =
        enable_bias_rope_flash_fuse_flag;
    unsigned char skip_ops[8192];
    if (n_ops > (int)sizeof(skip_ops)) {
        fprintf(stderr, "[bn:gpu:cuda] execute graph too large: %d ops\n",
                n_ops);
        if (profile) {
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
        }
        return -1;
    }
    memset(skip_ops, 0, (size_t)n_ops);
    int graph_exec = (enable_graph_exec_flag ||
                      (readback_buf < 0 && !out_host && out_len <= 0)) &&
                     n_ops > 10 && !profile;
    int graph_building = 0;
    cudaGraphExec_t graph_instance = NULL;
    if (graph_exec) {
        if (ctx->exec_graph && ctx->exec_graph_ops != n_ops) {
            cudaGraphExecDestroy(ctx->exec_graph);
            ctx->exec_graph = NULL;
            if (ctx->exec_graph_def) {
                cudaGraphDestroy(ctx->exec_graph_def);
                ctx->exec_graph_def = NULL;
            }
            ctx->exec_node_count = 0;
            ctx->exec_node_cursor = 0;
            ctx->exec_graph_ops = 0;
        }
        if (!ctx->exec_graph) {
            cudaError_t graph_err = cudaGraphCreate(&ctx->exec_graph_def, 0);
            if (graph_err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] graph create failed: %s\n",
                        cudaGetErrorString(graph_err));
                graph_exec = 0;
            } else {
                graph_building = 1;
                ctx->exec_node_count = 0;
                ctx->exec_node_cursor = 0;
            }
        } else {
            ctx->exec_node_cursor = 0;
        }
        if (ctx->exec_stream == (cudaStream_t)0) {
            fprintf(stderr,
                    "[bn:gpu:cuda] graph execution requires stream mode\n");
            graph_exec = 0;
        }
        if (graph_exec) {
            if (!ctx->d_runtime) {
                cudaError_t alloc_err =
                    cudaMalloc(&ctx->d_runtime,
                               sizeof(BnCudaRuntimeParams));
                if (alloc_err != cudaSuccess) {
                    fprintf(stderr,
                            "[bn:gpu:cuda] runtime params alloc failed: %s\n",
                            cudaGetErrorString(alloc_err));
                    return -1;
                }
            }
            BnCudaRuntimeParams rt;
            memset(&rt, 0, sizeof(rt));
            rt.pos = -1;
            rt.n_kv = -1;
            rt.seq_len = -1;
            for (int ri = 0; ri < n_ops; ri++) {
                if (rt.pos < 0 &&
                    (ops[ri].op_code == BN_GPU_CODE_ROPE ||
                     ops[ri].op_code == BN_GPU_CODE_ROPE_QK)) {
                    rt.pos = (int)ops[ri].p[2];
                }
                if (rt.n_kv < 0 &&
                    ops[ri].op_code == BN_GPU_CODE_FLASH_ATTN) {
                    rt.n_kv = (int)ops[ri].p[2];
                    rt.seq_len = (int)ops[ri].p[5];
                }
                if (rt.pos >= 0 && rt.n_kv >= 0 && rt.seq_len > 0)
                    break;
            }
            if (rt.pos < 0 || rt.n_kv <= 0 || rt.seq_len <= 0) {
                graph_exec = 0;
            } else {
                rt.cache_pos = rt.pos % rt.seq_len;
                cudaError_t copy_err =
                    cudaMemcpyAsync(ctx->d_runtime, &rt, sizeof(rt),
                                    cudaMemcpyHostToDevice,
                                    ctx->exec_stream);
                if (copy_err != cudaSuccess) {
                    fprintf(stderr,
                            "[bn:gpu:cuda] runtime params copy failed: %s\n",
                            cudaGetErrorString(copy_err));
                    return -1;
                }
            }
        }
    }
    for (int i = 0; i < n_ops; i++) {
        if (skip_ops[i])
            continue;
        const BnGPUOp *op = &ops[i];
        const BnGPUOp *next = (i + 1 < n_ops) ? &ops[i + 1] : NULL;
        cudaError_t err = cudaSuccess;
        int profile_code = (op->op_kind == BN_GPU_OP_LOGITS ||
                            (op->op_code == BN_GPU_CODE_MATVEC &&
                             op->buf_out == BN_GPU_VALUE_LOGITS))
            ? BN_CUDA_PROFILE_LOGITS
            : op->op_code;
        if (profile)
        cudaEventRecord(ev_start, ctx->exec_stream);
        switch (op->op_code) {
        case BN_GPU_CODE_MATVEC: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *in = cuda_act(ctx, op->buf_in);
            float *out = cuda_act(ctx, op->buf_out);
            const float *bias = NULL;
            int bias_idx = -1;
            int fused_copy_idx = -1;
            size_t out_offset = (size_t)op->p[5];
            if (!w || !w->data || !in || !out ||
                !cuda_type_supported(op->type) ||
                op->rows <= 0 || op->cols <= 0)
                return -1;
            if (next && i + 2 < n_ops &&
                next->op_code == BN_GPU_CODE_BIAS_ADD &&
                next->buf_in == op->buf_out &&
                (int)next->p[0] == op->rows) {
                const BnGPUOp *copy = &ops[i + 2];
                if (copy->op_code == BN_GPU_CODE_COPY &&
                    copy->buf_in == op->buf_out &&
                    (int)copy->p[0] == (int)op->p[5] &&
                    (int)copy->p[2] == op->rows) {
                    BnCudaBuffer *bw = (BnCudaBuffer *)next->W_buf;
                    float *copy_out = cuda_act(ctx, copy->buf_out);
                    if (bw && bw->data && copy_out) {
                        bias = (const float *)bw->data;
                        out = copy_out;
                        out_offset = (size_t)copy->p[1];
                        fused_copy_idx = i + 2;
                    }
                }
            }
            bias_idx = (fused_copy_idx < 0 && fuse_bias_enabled &&
                        op->type != BN_GGUF_TENSOR_Q8_0)
                ? cuda_find_fusable_bias(ops, n_ops, i, op->buf_out,
                                          op->rows)
                : -1;
            if (bias_idx >= 0) {
                BnCudaBuffer *bw = (BnCudaBuffer *)ops[bias_idx].W_buf;
                if (bw && bw->data &&
                    !cuda_bias_followed_by_copy(ops, n_ops, bias_idx)) {
                    bias = (const float *)bw->data;
                } else {
                    bias_idx = -1;
                }
            }
            if (op->type == BN_GGUF_TENSOR_Q5_0 &&
                (op->cols & 31) == 0 && enable_q5_matvec4) {
                BN_CUDA_LAUNCH(ctx, q5_0_matvec4_kernel,
                    (op->rows + 3) / 4, threads,
                    (size_t)threads * sizeof(float) * 4,
                    out, (const BnBlockQ5_0 *)w->data, in, bias,
                    op->rows, op->cols, out_offset);
            } else if (op->type == BN_GGUF_TENSOR_Q5_0 &&
                       (op->cols & 31) == 0 && enable_q5_warp) {
                int q5_threads = 256;
                int warps = q5_threads / 32;
                int blocks = (op->rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q5_0_matvec_warp_kernel, blocks,
                    q5_threads, 0,
                    out, (const BnBlockQ5_0 *)w->data, in, bias,
                    op->rows, op->cols, out_offset);
            } else if (op->type == BN_GGUF_TENSOR_Q6_K &&
                       (op->cols % BN_QK_K) == 0 && enable_q6k_warp) {
                int q6_threads = 256;
                int warps = q6_threads / 32;
                int blocks = (op->rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q6k_matvec_warp_kernel, blocks,
                    q6_threads, 0,
                    out, (const BnBlockQ6K *)w->data, in, bias,
                    op->rows, op->cols, out_offset);
            } else if (!disable_q8_warp && op->type == BN_GGUF_TENSOR_Q8_0 &&
                       op->rows >= 16384 && (op->cols & 31) == 0 && !bias) {
                int q8_threads = 256;
                int warps = q8_threads / 32;
                int blocks = ((op->rows + 3) / 4 + warps - 1) / warps;
                if (out_offset == 0) {
                    BN_CUDA_LAUNCH_STATIC(ctx, q8_0_matvec4_warp_kernel,
                        blocks, q8_threads, 0,
                        out, (const BnBlockQ8_0 *)w->data, in, op->rows,
                        op->cols, out_offset);
                } else {
                    BN_CUDA_LAUNCH(ctx, q8_0_matvec4_warp_kernel, blocks,
                        q8_threads, 0,
                        out, (const BnBlockQ8_0 *)w->data, in, op->rows,
                        op->cols, out_offset);
                }
            } else {
                if (out_offset == 0 && !bias) {
                    BN_CUDA_LAUNCH_STATIC(ctx, matvec_kernel,
                        dim3(op->rows, 1, 1), threads,
                        (size_t)threads * sizeof(float),
                        out, w->data, in, bias, op->rows, op->cols, op->type,
                        out_offset);
                } else {
                    BN_CUDA_LAUNCH(ctx, matvec_kernel,
                        dim3(op->rows, 1, 1), threads,
                        (size_t)threads * sizeof(float),
                        out, w->data, in, bias, op->rows, op->cols, op->type,
                        out_offset);
                }
            }
            if (fused_copy_idx >= 0) {
                i += 2;
            } else if (bias_idx >= 0) {
                skip_ops[bias_idx] = 1;
            }
            break;
        }
        case BN_GPU_CODE_MATVEC_SPLIT:
        case BN_GPU_CODE_Q4K_MATVEC_SPLIT:
        case BN_GPU_CODE_Q8_MATVEC_SPLIT:
        case BN_GPU_CODE_Q5K_MATVEC_SPLIT: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *in = cuda_act(ctx, op->buf_in);
            float *out0 = cuda_act(ctx, op->buf_out);
            float *out1 = cuda_act(ctx, op->buf_aux);
            float *out2 = (op->p[3] > op->p[2])
                ? cuda_act(ctx, op->rows)
                : NULL;
            const float *bias0 = NULL;
            int bias0_idx = -1;
            int total_rows = (int)op->p[0];
            int cols = (int)op->p[1];
            int split0 = (int)op->p[2];
            int split1 = (int)op->p[3];
            if (!w || !w->data || !in || !out0 || !out1 ||
                !cuda_type_supported(op->type) || total_rows <= 0 ||
                cols <= 0 || split0 <= 0 || split0 > total_rows)
                return -1;
            if (split1 > split0 && (!out2 || split1 > total_rows))
                return -1;
            if (!disable_qkv_mixed_fuse &&
                next && i + 7 < n_ops &&
                next->op_code == BN_GPU_CODE_BIAS_ADD &&
                next->buf_in == op->buf_out &&
                (int)next->p[0] == split0 &&
                ops[i + 2].op_code == BN_GPU_CODE_BIAS_ADD &&
                ops[i + 2].buf_in == op->buf_aux &&
                (int)ops[i + 2].p[0] == total_rows - split0 &&
                ops[i + 3].op_code == BN_GPU_CODE_ROPE &&
                ops[i + 3].buf_in == op->buf_aux &&
                (int)ops[i + 3].p[0] * (int)ops[i + 3].p[1] ==
                    total_rows - split0 &&
                ops[i + 4].op_code == BN_GPU_CODE_COPY &&
                ops[i + 4].buf_in == op->buf_aux &&
                (int)ops[i + 4].p[0] == 0 &&
                (int)ops[i + 4].p[2] == total_rows - split0 &&
                ops[i + 5].op_code == BN_GPU_CODE_MATVEC &&
                ops[i + 5].buf_in == op->buf_in &&
                ops[i + 5].rows == total_rows - split0 &&
                ops[i + 5].cols == cols &&
                ops[i + 6].op_code == BN_GPU_CODE_BIAS_ADD &&
                ops[i + 6].buf_in == ops[i + 5].buf_out &&
                (int)ops[i + 6].p[0] == ops[i + 5].rows &&
                ops[i + 7].op_code == BN_GPU_CODE_COPY &&
                ops[i + 7].buf_in == ops[i + 5].buf_out &&
                (int)ops[i + 7].p[0] == 0 &&
                (int)ops[i + 7].p[2] == ops[i + 5].rows) {
                BnCudaBuffer *qbw = (BnCudaBuffer *)next->W_buf;
                BnCudaBuffer *vw = (BnCudaBuffer *)ops[i + 5].W_buf;
                BnCudaBuffer *vbw = (BnCudaBuffer *)ops[i + 6].W_buf;
                BnCudaBuffer *kbw = qkv_fuse_key_cache
                    ? (BnCudaBuffer *)ops[i + 2].W_buf
                    : NULL;
                void *key_cache = qkv_fuse_key_cache
                    ? cuda_act(ctx, ops[i + 4].buf_out)
                    : NULL;
                void *value_cache = cuda_act(ctx, ops[i + 7].buf_out);
                if (qbw && qbw->data && vw && vw->data && vbw &&
                    vbw->data && value_cache &&
                    (!qkv_fuse_key_cache ||
                     (kbw && kbw->data && key_cache)) &&
                    cuda_type_supported(ops[i + 5].type)) {
                    float *freq = qkv_fuse_key_cache
                        ? cuda_act(ctx, BN_GPU_VALUE_ROPE_FREQ)
                        : NULL;
                    if (qkv_fuse_key_cache && !freq) return -1;
                    int k_pair_opt = enable_qkv_kpair_opt;
                    int k_grid_rows = k_pair_opt
                        ? (total_rows - split0 + 1) / 2
                        : total_rows - split0;
                    int q_unused = qkv_fuse_key_cache &&
                        cuda_buf_unused_until_write(ops, n_ops, i + 8,
                                                    op->buf_out);
                    if (q_unused) {
                        BN_CUDA_LAUNCH(ctx, kv_mixed_matvec_kernel,
                            k_grid_rows + ops[i + 5].rows, threads,
                            (size_t)threads * sizeof(float) * 2,
                            key_cache, value_cache, w->data, vw->data, in,
                            kbw ? (const float *)kbw->data : NULL,
                            (const float *)vbw->data, freq, split0,
                            total_rows - split0, ops[i + 5].rows, cols,
                            op->type, ops[i + 5].type,
                            (size_t)ops[i + 4].p[1],
                            (size_t)ops[i + 7].p[1], (int)ops[i + 3].p[0],
                            (int)ops[i + 3].p[1], (int)ops[i + 3].p[2],
                            (int)ops[i + 3].p[3], k_pair_opt, ctx->kv_f16);
                    } else if (graph_exec) {
                        int k_dim = total_rows - split0;
                        int pos = (int)ops[i + 3].p[2];
                        size_t dyn = pos > 0
                            ? (size_t)pos * (size_t)k_dim
                            : 0;
                        size_t key_base = (size_t)ops[i + 4].p[1];
                        size_t value_base = (size_t)ops[i + 7].p[1];
                        if (key_base >= dyn) key_base -= dyn;
                        if (value_base >= dyn) value_base -= dyn;
                        BN_CUDA_LAUNCH_STATIC(
                            ctx, qkv_mixed_matvec_runtime_kernel,
                            split0 + k_grid_rows + ops[i + 5].rows, threads,
                            (size_t)threads * sizeof(float) * 2,
                            out0, out1, value_cache, w->data, vw->data, in,
                            (const float *)qbw->data,
                            kbw ? (const float *)kbw->data : NULL,
                            (const float *)vbw->data, freq, key_cache,
                            split0, k_dim, ops[i + 5].rows, cols, op->type,
                            ops[i + 5].type, key_base, value_base,
                            (int)ops[i + 3].p[0], (int)ops[i + 3].p[1],
                            (int)ops[i + 3].p[3], k_pair_opt, ctx->kv_f16,
                            (const BnCudaRuntimeParams *)ctx->d_runtime);
                    } else {
                        BN_CUDA_LAUNCH(ctx, qkv_mixed_matvec_kernel,
                            split0 + k_grid_rows + ops[i + 5].rows, threads,
                            (size_t)threads * sizeof(float) * 2,
                            out0, out1, value_cache, w->data, vw->data, in,
                            (const float *)qbw->data,
                            kbw ? (const float *)kbw->data : NULL,
                            (const float *)vbw->data, freq, key_cache,
                            split0, total_rows - split0, ops[i + 5].rows,
                            cols, op->type, ops[i + 5].type,
                            (size_t)ops[i + 4].p[1],
                            (size_t)ops[i + 7].p[1], (int)ops[i + 3].p[0],
                            (int)ops[i + 3].p[1], (int)ops[i + 3].p[2],
                            (int)ops[i + 3].p[3], k_pair_opt, ctx->kv_f16);
                    }
                    skip_ops[i + 1] = 1;
                    if (qkv_fuse_key_cache) {
                        skip_ops[i + 2] = 1;
                        skip_ops[i + 3] = 1;
                        skip_ops[i + 4] = 1;
                    }
                    skip_ops[i + 5] = 1;
                    skip_ops[i + 6] = 1;
                    skip_ops[i + 7] = 1;
                    profile_code = BN_CUDA_PROFILE_QKV_MIXED;
                    break;
                }
            }
            bias0_idx = (fuse_bias_enabled && op->type != BN_GGUF_TENSOR_Q8_0)
                ? cuda_find_fusable_bias(ops, n_ops, i, op->buf_out, split0)
                : -1;
            if (bias0_idx >= 0) {
                BnCudaBuffer *bw = (BnCudaBuffer *)ops[bias0_idx].W_buf;
                if (bw && bw->data) {
                    bias0 = (const float *)bw->data;
                }
            }
            BN_CUDA_LAUNCH(ctx, matvec_split_kernel, total_rows, threads,
                (size_t)threads * sizeof(float),
                out0, out1, out2, w->data, in, bias0, total_rows, cols,
                op->type, split0, split1, (size_t)op->p[6],
                (size_t)op->p[7]);
            if (bias0_idx >= 0)
                skip_ops[bias0_idx] = 1;
            break;
        }
        case BN_GPU_CODE_FUSED_GATEUP_SILU: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *in = cuda_act(ctx, op->buf_in);
            float *out = cuda_act(ctx, op->buf_out);
            int total_rows = (int)op->p[0];
            int cols = (int)op->p[1];
            int gate_rows = (int)op->p[2];
            int up_rows = total_rows - gate_rows;
            if (!w || !w->data || !in || !out ||
                !cuda_type_supported(op->type) || total_rows <= 0 ||
                cols <= 0 || gate_rows <= 0 || up_rows <= 0)
                return -1;
            if (op->type == BN_GGUF_TENSOR_Q5_0 && (cols & 31) == 0 &&
                !disable_q5_gateup_warp) {
                int q5_gateup_threads = 64;
                int warps = q5_gateup_threads / 32;
                int blocks = (gate_rows + warps - 1) / warps;
                BN_CUDA_LAUNCH_STATIC(ctx, q5_0_fused_gateup_silu_warp_kernel,
                    blocks, q5_gateup_threads, 0,
                    out, (const BnBlockQ5_0 *)w->data, in, gate_rows,
                    up_rows, cols);
            } else if (op->type == BN_GGUF_TENSOR_Q8_0 &&
                       (cols & 31) == 0 &&
                       !disable_q8_gateup_warp) {
                int q8_gateup_threads = 64;
                int warps = q8_gateup_threads / 32;
                int blocks = (gate_rows + warps - 1) / warps;
                BN_CUDA_LAUNCH_STATIC(ctx, q8_0_fused_gateup_silu_warp_kernel,
                    blocks, q8_gateup_threads, 0,
                    out, (const BnBlockQ8_0 *)w->data, in, gate_rows,
                    up_rows, cols);
            } else {
                BN_CUDA_LAUNCH_STATIC(ctx, fused_gateup_silu_kernel, gate_rows,
                    threads, (size_t)threads * sizeof(float) * 2,
                    out, w->data, in, gate_rows, up_rows, cols, op->type);
            }
            break;
        }
        case BN_GPU_CODE_RMSNORM: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *in = cuda_act(ctx, op->buf_in);
            float *out = cuda_act(ctx, op->buf_out);
            int n = (int)op->p[0];
            if (!w || !w->data || !in || !out || n <= 0) return -1;
            BN_CUDA_LAUNCH_STATIC(ctx, rmsnorm_kernel, 1, threads,
                (size_t)threads * sizeof(float),
                out, in, (const float *)w->data, n,
                cuda_u32_to_f32(op->p[1]));
            break;
        }
        case BN_GPU_CODE_RESIDUAL_RMSNORM: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *in = cuda_act(ctx, op->buf_in);
            float *aux = cuda_act(ctx, op->buf_aux);
            float *out = cuda_act(ctx, op->buf_out);
            int n = (int)op->p[0];
            if (!w || !w->data || !in || !aux || !out || n <= 0)
                return -1;
            BN_CUDA_LAUNCH_STATIC(ctx, residual_rmsnorm_kernel, 1,
                threads, (size_t)threads * sizeof(float),
                in, aux, out, (const float *)w->data, n,
                cuda_u32_to_f32(op->p[1]));
            break;
        }
        case BN_GPU_CODE_COPY: {
            float *in = cuda_act(ctx, op->buf_in);
            float *out = cuda_act(ctx, op->buf_out);
            int n = (int)op->p[2];
            if (!in || !out || n < 0) return -1;
            BN_CUDA_LAUNCH(ctx, copy_kernel, (n + threads - 1) / threads,
                threads, 0,
                in, out, (int)op->p[0], (int)op->p[1], n);
            break;
        }
        case BN_GPU_CODE_BIAS_ADD: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *in = cuda_act(ctx, op->buf_in);
            int n = (int)op->p[0];
            if (!w || !w->data || !in || n <= 0) return -1;
            const BnGPUOp *after_next = (i + 2 < n_ops) ? &ops[i + 2] : NULL;
            if (enable_bias_rope_flash_fuse && fuse_rope_flash_enabled &&
                next && after_next &&
                next->op_code == BN_GPU_CODE_ROPE &&
                next->buf_in == op->buf_in &&
                after_next->op_code == BN_GPU_CODE_FLASH_ATTN &&
                after_next->buf_in == op->buf_in &&
                (int)next->p[0] == (int)after_next->p[0] &&
                (int)next->p[1] == (int)after_next->p[1] &&
                (int)next->p[0] * (int)next->p[1] == n &&
                (int)next->p[3] > 0 &&
                (int)after_next->p[2] > 0 &&
                (int)after_next->p[2] <= 2048 &&
                (int)after_next->p[3] > 0 &&
                (int)after_next->p[4] > 0 &&
                (int)after_next->p[5] > 0 &&
                cuda_buf_unused_until_write(ops, n_ops, i + 3,
                                            op->buf_in)) {
                float *out = cuda_act(ctx, after_next->buf_out);
                void *key = cuda_act(ctx, BN_GPU_VALUE_KEY_CACHE);
                void *value = cuda_act(ctx, BN_GPU_VALUE_VALUE_CACHE);
                float *freq = cuda_act(ctx, BN_GPU_VALUE_ROPE_FREQ);
                if (!out || !key || !value || !freq) return -1;
                int n_heads = (int)next->p[0];
                int head_size = (int)next->p[1];
                int flash_scratch = graph_exec ? (int)after_next->p[5]
                                                : (int)after_next->p[2];
                size_t shared = (size_t)(flash_scratch + threads + head_size) *
                                sizeof(float);
                if (graph_exec) {
                    BN_CUDA_LAUNCH_STATIC(
                        ctx, flash_attention_rope_q_runtime_kernel,
                        n_heads, threads, shared,
                        out, in, key, value, freq, (const float *)w->data,
                        n_heads, head_size, (int)after_next->p[3],
                        (int)after_next->p[4], (int)after_next->p[5],
                        after_next->p[6],
                        cuda_u32_to_f32(after_next->p[7]),
                        (int)next->p[3], ctx->kv_f16,
                        (const BnCudaRuntimeParams *)ctx->d_runtime);
                } else {
                    BN_CUDA_LAUNCH(ctx, flash_attention_rope_q_kernel,
                        n_heads, threads, shared,
                        out, in, key, value, freq, (const float *)w->data,
                        n_heads, head_size, (int)after_next->p[2],
                        (int)after_next->p[3], (int)after_next->p[4],
                        (int)after_next->p[5], after_next->p[6],
                        cuda_u32_to_f32(after_next->p[7]), (int)next->p[2],
                        (int)next->p[3], ctx->kv_f16);
                }
                i += 2;
            } else if (next && after_next &&
                next->op_code == BN_GPU_CODE_ROPE &&
                next->buf_in == op->buf_in &&
                after_next->op_code == BN_GPU_CODE_COPY &&
                after_next->buf_in == op->buf_in &&
                (int)after_next->p[0] == 0 &&
                (int)after_next->p[2] == n &&
                (int)next->p[0] > 0 && (int)next->p[1] > 0 &&
                (int)next->p[0] * (int)next->p[1] == n) {
                float *out = cuda_act(ctx, after_next->buf_out);
                float *freq = cuda_act(ctx, BN_GPU_VALUE_ROPE_FREQ);
                if (!out || !freq) return -1;
                BN_CUDA_LAUNCH(ctx, bias_rope_copy_kernel,
                    (int)next->p[0], threads, 0,
                    in, out, (const float *)w->data, freq,
                    (int)after_next->p[1], (int)next->p[0],
                    (int)next->p[1], (int)next->p[2],
                    (int)next->p[3]);
                i += 2;
            } else if (next && next->op_code == BN_GPU_CODE_COPY &&
                next->buf_in == op->buf_in && (int)next->p[0] == 0 &&
                (int)next->p[2] == n) {
                float *out = cuda_act(ctx, next->buf_out);
                if (!out) return -1;
                BN_CUDA_LAUNCH(ctx, bias_add_copy_kernel,
                    (n + threads - 1) / threads, threads, 0,
                    in, out, (const float *)w->data, (int)next->p[0],
                    (int)next->p[1], n);
                i++;
            } else {
                BN_CUDA_LAUNCH(ctx, bias_add_kernel,
                    (n + threads - 1) / threads, threads, 0,
                    in, (const float *)w->data, n);
            }
            break;
        }
        case BN_GPU_CODE_RESIDUAL_ADD: {
            float *in = cuda_act(ctx, op->buf_in);
            float *aux = cuda_act(ctx, op->buf_aux);
            int n = (int)op->p[0];
            if (!in || !aux || n <= 0) return -1;
            BN_CUDA_LAUNCH(ctx, residual_add_kernel,
                (n + threads - 1) / threads, threads, 0,
                in, aux, n);
            break;
        }
        case BN_GPU_CODE_SILU_GATE:
        case BN_GPU_CODE_RELU2_GATE:
        case BN_GPU_CODE_SIGMOID_GATE: {
            float *in = cuda_act(ctx, op->buf_in);
            float *aux = cuda_act(ctx, op->buf_aux);
            int n = (int)op->p[0];
            if (!in || !aux || n <= 0) return -1;
            BN_CUDA_LAUNCH(ctx, activation_gate_kernel,
                (n + threads - 1) / threads, threads, 0,
                in, aux, n, op->op_code);
            break;
        }
        case BN_GPU_CODE_SILU_ACT:
        case BN_GPU_CODE_RELU2_ACT: {
            float *in = cuda_act(ctx, op->buf_in);
            int n = (int)op->p[0];
            if (!in || n <= 0) return -1;
            BN_CUDA_LAUNCH(ctx, activation_kernel,
                (n + threads - 1) / threads, threads, 0,
                in, n, op->op_code);
            break;
        }
        case BN_GPU_CODE_ROPE:
        case BN_GPU_CODE_ROPE_QK: {
            float *q = cuda_act(ctx, op->buf_in);
            float *k = op->op_code == BN_GPU_CODE_ROPE_QK
                ? cuda_act(ctx, op->buf_aux)
                : NULL;
            float *freq = cuda_act(ctx, BN_GPU_VALUE_ROPE_FREQ);
            int n_heads = (int)op->p[0];
            int head_size = (int)op->p[1];
            int rope_dims = (int)op->p[3];
            if (!q || !freq || n_heads <= 0 || head_size <= 0 ||
                rope_dims <= 0)
                return -1;
            if (fuse_rope_flash_enabled &&
                op->op_code == BN_GPU_CODE_ROPE && next &&
                next->op_code == BN_GPU_CODE_FLASH_ATTN &&
                next->buf_in == op->buf_in &&
                (int)next->p[0] == n_heads &&
                (int)next->p[1] == head_size &&
                (int)next->p[2] > 0 && (int)next->p[2] <= 2048 &&
                (int)next->p[3] > 0 && (int)next->p[4] > 0 &&
                (int)next->p[5] > 0 &&
                cuda_buf_unused_until_write(ops, n_ops, i + 2,
                                            op->buf_in)) {
                float *out = cuda_act(ctx, next->buf_out);
                void *key = cuda_act(ctx, BN_GPU_VALUE_KEY_CACHE);
                void *value = cuda_act(ctx, BN_GPU_VALUE_VALUE_CACHE);
                if (!out || !key || !value) return -1;
                int flash_scratch = graph_exec ? (int)next->p[5]
                                                : (int)next->p[2];
                size_t shared = (size_t)(flash_scratch + threads + head_size) *
                                sizeof(float);
                if (graph_exec) {
                    BN_CUDA_LAUNCH_STATIC(
                        ctx, flash_attention_rope_q_runtime_kernel,
                        n_heads, threads, shared,
                        out, q, key, value, freq, (const float *)NULL,
                        n_heads, head_size, (int)next->p[3],
                        (int)next->p[4], (int)next->p[5], next->p[6],
                        cuda_u32_to_f32(next->p[7]), rope_dims, ctx->kv_f16,
                        (const BnCudaRuntimeParams *)ctx->d_runtime);
                } else {
                    BN_CUDA_LAUNCH(ctx, flash_attention_rope_q_kernel,
                        n_heads, threads, shared,
                        out, q, key, value, freq, (const float *)NULL,
                        n_heads, head_size,
                        (int)next->p[2], (int)next->p[3], (int)next->p[4],
                        (int)next->p[5], next->p[6],
                        cuda_u32_to_f32(next->p[7]), (int)op->p[2],
                        rope_dims, ctx->kv_f16);
                }
                i++;
                break;
            }
            BN_CUDA_LAUNCH(ctx, rope_kernel, n_heads,
                (rope_dims + 1) / 2, 0,
                q, k, freq, n_heads, head_size, (int)op->p[2],
                rope_dims, (int)op->p[4], op->p[5]);
            break;
        }
        case BN_GPU_CODE_GQA_SCORES: {
            float *q = cuda_act(ctx, op->buf_in);
            void *key = cuda_act(ctx, BN_GPU_VALUE_KEY_CACHE);
            float *att = cuda_act(ctx, BN_GPU_VALUE_ATT);
            int n_heads = (int)op->p[0];
            int head_size = (int)op->p[1];
            int n_kv = (int)op->p[2];
            int kv_mul = (int)op->p[3];
            int kv_dim = (int)op->p[4];
            int seq_len = (int)op->p[5];
            if (!q || !key || !att || n_heads <= 0 || head_size <= 0 ||
                n_kv <= 0 || kv_mul <= 0 || kv_dim <= 0 || seq_len <= 0)
                return -1;
            BN_CUDA_LAUNCH(ctx, gqa_scores_kernel,
                dim3(n_heads, n_kv, 1), threads,
                (size_t)threads * sizeof(float),
                att, q, key, n_heads, head_size, n_kv, kv_mul, kv_dim,
                seq_len, op->p[6], cuda_u32_to_f32(op->p[7]),
                ctx->kv_f16);
            break;
        }
        case BN_GPU_CODE_SOFTMAX: {
            float *att = cuda_act(ctx, BN_GPU_VALUE_ATT);
            int n_heads = (int)op->p[0];
            int n_kv = (int)op->p[1];
            int seq_len = (int)op->p[2];
            if (!att || n_heads <= 0 || n_kv <= 0 || seq_len <= 0)
                return -1;
            BN_CUDA_LAUNCH(ctx, softmax_kernel, n_heads, threads,
                (size_t)threads * sizeof(float),
                att, n_heads, n_kv, seq_len);
            break;
        }
        case BN_GPU_CODE_GQA_COMBINE: {
            float *out = cuda_act(ctx, op->buf_out);
            float *att = cuda_act(ctx, BN_GPU_VALUE_ATT);
            void *value = cuda_act(ctx, BN_GPU_VALUE_VALUE_CACHE);
            int n_heads = (int)op->p[0];
            int head_size = (int)op->p[1];
            int n_kv = (int)op->p[2];
            int kv_mul = (int)op->p[3];
            int kv_dim = (int)op->p[4];
            int seq_len = (int)op->p[5];
            if (!out || !att || !value || n_heads <= 0 || head_size <= 0 ||
                n_kv <= 0 || kv_mul <= 0 || kv_dim <= 0 || seq_len <= 0)
                return -1;
            BN_CUDA_LAUNCH(ctx, gqa_combine_kernel, n_heads, head_size, 0,
                out, att, value, n_heads, head_size, n_kv, kv_mul, kv_dim,
                seq_len, op->p[6], ctx->kv_f16);
            break;
        }
        case BN_GPU_CODE_FLASH_ATTN: {
            float *q = cuda_act(ctx, op->buf_in);
            float *out = cuda_act(ctx, op->buf_out);
            void *key = cuda_act(ctx, BN_GPU_VALUE_KEY_CACHE);
            void *value = cuda_act(ctx, BN_GPU_VALUE_VALUE_CACHE);
            int n_heads = (int)op->p[0];
            int head_size = (int)op->p[1];
            int n_kv = (int)op->p[2];
            int kv_mul = (int)op->p[3];
            int kv_dim = (int)op->p[4];
            int seq_len = (int)op->p[5];
            if (!q || !out || !key || !value || n_heads <= 0 ||
                head_size <= 0 || n_kv <= 0 || kv_mul <= 0 ||
                kv_dim <= 0 || seq_len <= 0 || n_kv > 2048)
                return -1;
            int flash_scratch = graph_exec ? seq_len : n_kv;
            size_t shared = (size_t)(flash_scratch + threads) * sizeof(float);
            BN_CUDA_LAUNCH(ctx, flash_attention_kernel, n_heads, threads,
                shared,
                out, q, key, value, n_heads, head_size, n_kv, kv_mul,
                kv_dim, seq_len, op->p[6], cuda_u32_to_f32(op->p[7]),
                ctx->kv_f16);
            break;
        }
        default:
            return -1;
        }
        exec_launches++;
        if (profile_code >= 0 && profile_code < BN_CUDA_PROFILE_MAX)
            exec_launch_by_code[profile_code]++;
        if (!graph_exec) {
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] execute op %d launch failed: %s\n",
                        op->op_code, cudaGetErrorString(err));
                if (profile) {
                    cudaEventDestroy(ev_start);
                    cudaEventDestroy(ev_stop);
                }
                return -1;
            }
        }
        if (profile) {
            cudaEventRecord(ev_stop, ctx->exec_stream);
            err = cudaEventSynchronize(ev_stop);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] profile sync failed: %s\n",
                        cudaGetErrorString(err));
                cudaEventDestroy(ev_start);
                cudaEventDestroy(ev_stop);
                return -1;
            }
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            if (profile_code >= 0 && profile_code < BN_CUDA_PROFILE_MAX) {
                profile_ops[profile_code]++;
                profile_ms[profile_code] += (double)ms;
            }
        }
    }

    if (graph_exec) {
        cudaError_t graph_err = cudaSuccess;
        if (graph_building) {
            graph_err = cudaGraphInstantiate(&ctx->exec_graph,
                                             ctx->exec_graph_def, NULL,
                                             NULL, 0);
            if (graph_err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] graph instantiate failed: %s\n",
                        cudaGetErrorString(graph_err));
                cudaGraphDestroy(ctx->exec_graph_def);
                ctx->exec_graph_def = NULL;
                ctx->exec_node_count = 0;
                return -1;
            }
            ctx->exec_graph_ops = n_ops;
        } else if (ctx->exec_node_cursor != ctx->exec_node_count) {
            fprintf(stderr,
                    "[bn:gpu:cuda] graph replay node count mismatch: "
                    "%d != %d\n",
                    ctx->exec_node_cursor, ctx->exec_node_count);
            return -1;
        }
        graph_instance = ctx->exec_graph;
        graph_err = cudaGraphLaunch(graph_instance, ctx->exec_stream);
        if (graph_err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] graph launch failed: %s\n",
                    cudaGetErrorString(graph_err));
            return -1;
        }
    }

    if (readback_buf >= 0 && out_host && out_len > 0) {
        float *src = cuda_act(ctx, readback_buf);
        if (!src) return -1;
        if (profile)
            cudaEventRecord(ev_start, ctx->exec_stream);
        cudaError_t err = cudaSuccess;
        if (ctx->exec_stream) {
            err = cudaMemcpyAsync(out_host, src,
                                  (size_t)out_len * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  ctx->exec_stream);
        } else {
            err = cudaMemcpy(out_host, src, (size_t)out_len * sizeof(float),
                             cudaMemcpyDeviceToHost);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] execute readback failed: %s\n",
                    cudaGetErrorString(err));
            if (profile) {
                cudaEventDestroy(ev_start);
                cudaEventDestroy(ev_stop);
            }
            return -1;
        }
        if (profile) {
            cudaEventRecord(ev_stop, ctx->exec_stream);
            err = cudaEventSynchronize(ev_stop);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] profile readback sync failed: %s\n",
                        cudaGetErrorString(err));
                cudaEventDestroy(ev_start);
                cudaEventDestroy(ev_stop);
                return -1;
            }
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            profile_ms[BN_CUDA_PROFILE_READBACK] += (double)ms;
            profile_ops[BN_CUDA_PROFILE_READBACK]++;
        } else {
            err = ctx->exec_stream
                ? cudaStreamSynchronize(ctx->exec_stream)
                : cudaSuccess;
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] execute readback sync failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
        }
    } else if (graph_instance) {
        cudaError_t err = cudaStreamSynchronize(ctx->exec_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] graph sync failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (profile) {
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        profile_calls++;
        int every = cuda_env_int("BN_CUDA_PROFILE_EVERY", 1);
        if (every <= 0) every = 1;
        if ((profile_calls % (unsigned long long)every) == 0) {
            fprintf(stderr, "[bn:gpu:cuda:profile] calls=%llu\n",
                    profile_calls);
            for (int code = 0; code < BN_CUDA_PROFILE_MAX; code++) {
                if (!profile_ops[code]) continue;
                fprintf(stderr,
                        "  %-18s ops=%llu total_ms=%.3f avg_us=%.2f\n",
                        cuda_profile_name(code), profile_ops[code],
                        profile_ms[code],
                        profile_ms[code] * 1000.0 /
                            (double)profile_ops[code]);
            }
        }
    }
    if (profile_wall) {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] wall profile sync failed: %s\n",
                    cudaGetErrorString(sync_err));
            return -1;
        }
        double exec_ms = cuda_wall_ms() - wall_start;
        unsigned long long next_wall_call = wall_calls + 1;
        int detail_limit = cuda_env_int("BN_CUDA_PROFILE_WALL_DETAIL", 0);
        if (detail_limit < 0 ||
            (detail_limit > 0 &&
             next_wall_call <= (unsigned long long)detail_limit)) {
            fprintf(stderr,
                    "[bn:gpu:cuda:wall:detail] call=%llu ops=%d "
                    "launches=%llu ms=%.3f avg_launch_us=%.2f\n",
                    next_wall_call, n_ops, exec_launches, exec_ms,
                    exec_launches
                        ? exec_ms * 1000.0 / (double)exec_launches
                        : 0.0);
            for (int code = 0; code < BN_CUDA_PROFILE_MAX; code++) {
                if (!exec_launch_by_code[code]) continue;
                fprintf(stderr, "  %-18s launches=%u\n",
                        cuda_profile_name(code),
                        exec_launch_by_code[code]);
            }
        }
        wall_ms += exec_ms;
        wall_ops += (unsigned long long)n_ops;
        wall_launches += exec_launches;
        for (int code = 0; code < BN_CUDA_PROFILE_MAX; code++)
            wall_launch_by_code[code] +=
                (unsigned long long)exec_launch_by_code[code];
        wall_calls++;
        int every = cuda_env_int("BN_CUDA_PROFILE_WALL_EVERY", 16);
        if (every <= 0) every = 16;
        if ((wall_calls % (unsigned long long)every) == 0) {
            fprintf(stderr,
                    "[bn:gpu:cuda:wall] calls=%llu ops=%llu launches=%llu "
                    "total_ms=%.3f avg_call_ms=%.3f "
                    "avg_op_us=%.2f avg_launch_us=%.2f\n",
                    wall_calls, wall_ops, wall_launches, wall_ms,
                    wall_ms / (double)wall_calls,
                    wall_ms * 1000.0 / (double)wall_ops,
                    wall_ms * 1000.0 / (double)wall_launches);
            for (int code = 0; code < BN_CUDA_PROFILE_MAX; code++) {
                if (!wall_launch_by_code[code]) continue;
                fprintf(stderr, "  %-18s launches=%llu\n",
                        cuda_profile_name(code),
                        wall_launch_by_code[code]);
            }
        }
    }
    return 0;
}

BnGPUBackend *bn_gpu_cuda_create(void) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
        fprintf(stderr, "[bn:gpu:cuda] no CUDA device available\n");
        return NULL;
    }
    int device = 0;
    const char *device_env = getenv("BN_CUDA_DEVICE");
    if (device_env && device_env[0]) {
        if (strcmp(device_env, "auto") == 0) {
            size_t best_free = 0;
            int best_device = 0;
            for (int d = 0; d < ndev; d++) {
                cudaError_t set_err = cudaSetDevice(d);
                if (set_err != cudaSuccess) continue;
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                cudaError_t mem_err = cudaMemGetInfo(&free_bytes,
                                                     &total_bytes);
                if (mem_err == cudaSuccess && free_bytes > best_free) {
                    best_free = free_bytes;
                    best_device = d;
                }
            }
            device = best_device;
        } else {
            char *end = NULL;
            long parsed = strtol(device_env, &end, 10);
            if (!end || *end != '\0' || parsed < 0 || parsed >= ndev) {
                fprintf(stderr,
                        "[bn:gpu:cuda] invalid BN_CUDA_DEVICE=%s "
                        "(use 0..%d or auto)\n",
                        device_env, ndev - 1);
                return NULL;
            }
            device = (int)parsed;
        }
    }
    cudaError_t set_err = cudaSetDevice(device);
    if (set_err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] failed to select device %d: %s\n",
                device, cudaGetErrorString(set_err));
        return NULL;
    }

    BnCudaCtx *ctx = (BnCudaCtx *)calloc(1, sizeof(BnCudaCtx));
    BnGPUBackend *gpu = (BnGPUBackend *)calloc(1, sizeof(BnGPUBackend));
    if (!ctx || !gpu) {
        free(ctx);
        free(gpu);
        return NULL;
    }
    ctx->device = device;
    cudaError_t stream_err = cudaStreamCreateWithFlags(&ctx->stream,
                                                       cudaStreamNonBlocking);
    if (stream_err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] stream create failed: %s\n",
                cudaGetErrorString(stream_err));
        free(ctx);
        free(gpu);
        return NULL;
    }
    gpu->buffer_create = cuda_buffer_create;
    gpu->buffer_destroy = cuda_buffer_destroy;
    gpu->matvec = cuda_matvec;
    gpu->matmul = cuda_matmul;
    gpu->matvec_batch = cuda_matvec_batch;
    gpu->dense_ffn = cuda_dense_ffn;
    gpu->init_activations = cuda_init_activations;
    gpu->free_activations = cuda_free_activations;
    gpu->write_activation = cuda_write_activation;
    gpu->read_activation = cuda_read_activation;
    if (!getenv("BN_CUDA_DISABLE_GRAPH"))
        gpu->execute = cuda_execute;
    gpu->ctx = ctx;
    gpu->kind = BN_GPU_BACKEND_CUDA;
    gpu->max_storage_binding_size = 0;
    gpu->caps = BN_GPU_CAP_FLASH_ATTN |
                BN_GPU_CAP_Q4_MATVEC_SPLIT |
                BN_GPU_CAP_Q5_MATVEC_SPLIT |
                BN_GPU_CAP_Q4K_MATVEC_SPLIT |
                BN_GPU_CAP_Q4_FUSED_GATEUP_SILU |
                BN_GPU_CAP_Q5_FUSED_GATEUP_SILU |
                BN_GPU_CAP_Q8_FUSED_GATEUP_SILU |
                BN_GPU_CAP_Q8_MATVEC_SPLIT |
                BN_GPU_CAP_Q5K_MATVEC_SPLIT;
    return gpu;
}

void bn_gpu_cuda_destroy(BnGPUBackend *gpu) {
    if (!gpu) return;
    BnCudaCtx *ctx = (BnCudaCtx *)gpu->ctx;
    if (ctx) {
        if (cuda_ctx_set_device(ctx) != 0) {
            free(ctx);
            free(gpu);
            return;
        }
        if (ctx->d_x) cudaFree(ctx->d_x);
        if (ctx->d_out) cudaFree(ctx->d_out);
        if (ctx->d_ops) cudaFree(ctx->d_ops);
        if (ctx->d_runtime) cudaFree(ctx->d_runtime);
        if (ctx->exec_graph) cudaGraphExecDestroy(ctx->exec_graph);
        if (ctx->exec_graph_def) cudaGraphDestroy(ctx->exec_graph_def);
        free(ctx->exec_nodes);
        free(ctx->h_out);
        cuda_free_activations(ctx);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
    }
    free(ctx);
    free(gpu);
}

#endif // BN_ENABLE_CUDA
