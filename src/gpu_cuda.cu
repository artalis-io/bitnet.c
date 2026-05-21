#include "gpu_cuda.h"

#ifdef BN_ENABLE_CUDA

#include "gguf.h"
#include "model_config.h"
#include "quant.h"
#include "gpu_shader.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    void *data;
    void *f16_data;
    size_t f16_size;
    void *f32_data;
    size_t f32_size;
    size_t size;
    int type;
    int rows;
    int cols;
} BnCudaBuffer;

typedef struct {
    int device;
    cudaStream_t stream;
    cudaStream_t exec_stream;
    cublasHandle_t cublas;
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
    void *d_q8_1;
    size_t d_q8_1_bytes;
    void *d_q8_k;
    size_t d_q8_k_bytes;
    void *d_x_f16;
    size_t d_x_f16_bytes;
    void *d_argmax;
    size_t d_argmax_bytes;
    int *d_penalty_tokens;
    size_t d_penalty_tokens_bytes;
    void **d_gemm_ptrs;
    size_t d_gemm_ptrs_bytes;
    void **h_gemm_ptrs;
    size_t h_gemm_ptrs_bytes;
    float *d_prefill;
    size_t d_prefill_bytes;
    float *act_bufs[BN_GPU_VALUE_COUNT];
    size_t act_sizes[BN_GPU_VALUE_COUNT];
} BnCudaCtx;

static int cuda_ensure_gemm_ptrs(BnCudaCtx *ctx, int n_ptrs);

typedef struct {
    uint16_t d;
    uint16_t s;
    int8_t qs[32];
} BnCudaBlockQ8_1;

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

typedef struct {
    float value;
    int index;
} BnCudaArgmaxPair;

static float *cuda_act(BnCudaCtx *ctx, int idx);

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

enum {
    BN_CUDA_DENSE_PROF_INPUT_NORM,
    BN_CUDA_DENSE_PROF_QK,
    BN_CUDA_DENSE_PROF_WV,
    BN_CUDA_DENSE_PROF_QK_ROPE,
    BN_CUDA_DENSE_PROF_ATTN,
    BN_CUDA_DENSE_PROF_WO_RESID,
    BN_CUDA_DENSE_PROF_FFN_NORM,
    BN_CUDA_DENSE_PROF_GATEUP,
    BN_CUDA_DENSE_PROF_ACT,
    BN_CUDA_DENSE_PROF_DOWN_RESID,
    BN_CUDA_DENSE_PROF_KV_READBACK,
    BN_CUDA_DENSE_PROF_OUT_READBACK,
    BN_CUDA_DENSE_PROF_MAX
};

static const char *cuda_dense_profile_name(int code) {
    switch (code) {
    case BN_CUDA_DENSE_PROF_INPUT_NORM: return "input_norm";
    case BN_CUDA_DENSE_PROF_QK: return "qk";
    case BN_CUDA_DENSE_PROF_WV: return "wv";
    case BN_CUDA_DENSE_PROF_QK_ROPE: return "qk_rope";
    case BN_CUDA_DENSE_PROF_ATTN: return "attn";
    case BN_CUDA_DENSE_PROF_WO_RESID: return "wo_resid";
    case BN_CUDA_DENSE_PROF_FFN_NORM: return "ffn_norm";
    case BN_CUDA_DENSE_PROF_GATEUP: return "gateup";
    case BN_CUDA_DENSE_PROF_ACT: return "act";
    case BN_CUDA_DENSE_PROF_DOWN_RESID: return "down_resid";
    case BN_CUDA_DENSE_PROF_KV_READBACK: return "kv_store";
    case BN_CUDA_DENSE_PROF_OUT_READBACK: return "out_readback";
    default: return "unknown";
    }
}

static void cuda_dense_profile_add(double *totals, int code, double ms) {
    if (code >= 0 && code < BN_CUDA_DENSE_PROF_MAX)
        totals[code] += ms;
}

static void cuda_dense_profile_maybe_print(double *totals,
                                           unsigned long long *layers,
                                           int n_tokens, int dim) {
    (*layers)++;
    int every = cuda_env_int("BN_CUDA_PREFILL_DENSE_PROFILE_EVERY", 0);
    if (every <= 0) every = 36;
    if ((*layers % (unsigned long long)every) != 0)
        return;
    double sum = 0.0;
    for (int i = 0; i < BN_CUDA_DENSE_PROF_MAX; i++)
        sum += totals[i];
    fprintf(stderr,
            "[bn:gpu:cuda:dense-prefill] layers=%llu window=%d tokens=%d dim=%d total=%.3fms\n",
            *layers, every, n_tokens, dim, sum);
    for (int i = 0; i < BN_CUDA_DENSE_PROF_MAX; i++) {
        if (totals[i] <= 0.0) continue;
        fprintf(stderr,
                "[bn:gpu:cuda:dense-prefill]   %-12s %.3fms %.1f%%\n",
                cuda_dense_profile_name(i), totals[i],
                sum > 0.0 ? (totals[i] * 100.0 / sum) : 0.0);
        totals[i] = 0.0;
    }
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

static __device__ float cuda_q4k_value(const BnBlockQ4K *blk, int i);
static __device__ float cuda_q6k_value(const BnBlockQ6K *blk, int i);

static __global__ void f32_to_f16_kernel(__half *out, const float *in,
                                         size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

static __global__ void dequant_q8_0_to_f16_kernel(
    __half *out, const BnBlockQ8_0 *blocks, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * (size_t)cols;
    if (i >= n) return;
    int row = (int)(i / (size_t)cols);
    int col = (int)(i - (size_t)row * (size_t)cols);
    int n_bpr = cols / 32;
    const BnBlockQ8_0 *blk = &blocks[(size_t)row * n_bpr + col / 32];
    float v = cuda_fp16_to_fp32(blk->d) * (float)blk->qs[col & 31];
    out[i] = __float2half_rn(v);
}

static __global__ void dequant_q5_0_to_f16_kernel(
    __half *out, const BnBlockQ5_0 *blocks, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * (size_t)cols;
    if (i >= n) return;
    int row = (int)(i / (size_t)cols);
    int col = (int)(i - (size_t)row * (size_t)cols);
    int n_bpr = cols / 32;
    const BnBlockQ5_0 *blk = &blocks[(size_t)row * n_bpr + col / 32];
    uint32_t qh = (uint32_t)blk->qh[0] |
                  ((uint32_t)blk->qh[1] << 8) |
                  ((uint32_t)blk->qh[2] << 16) |
                  ((uint32_t)blk->qh[3] << 24);
    int j = col & 31;
    uint8_t qs = blk->qs[j & 15];
    int q = j < 16
        ? (int)((qs & 15) | (((qh >> j) & 1u) << 4)) - 16
        : (int)((qs >> 4) | (((qh >> j) & 1u) << 4)) - 16;
    float v = cuda_fp16_to_fp32(blk->d) * (float)q;
    out[i] = __float2half_rn(v);
}

static __global__ void dequant_q4k_to_f16_kernel(
    __half *out, const BnBlockQ4K *blocks, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * (size_t)cols;
    if (i >= n) return;
    int row = (int)(i / (size_t)cols);
    int col = (int)(i - (size_t)row * (size_t)cols);
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ4K *blk = &blocks[(size_t)row * n_bpr +
                                    col / BN_QK_K];
    float v = cuda_q4k_value(blk, col & (BN_QK_K - 1));
    out[i] = __float2half_rn(v);
}

static __global__ void dequant_q6k_to_f32_kernel(
    float *out, const BnBlockQ6K *blocks, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * (size_t)cols;
    if (i >= n) return;
    int row = (int)(i / (size_t)cols);
    int col = (int)(i - (size_t)row * (size_t)cols);
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ6K *blk = &blocks[(size_t)row * n_bpr +
                                    col / BN_QK_K];
    out[i] = cuda_q6k_value(blk, col & (BN_QK_K - 1));
}

static __global__ void dequant_q6k_to_f16_kernel(
    __half *out, const BnBlockQ6K *blocks, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * (size_t)cols;
    if (i >= n) return;
    int row = (int)(i / (size_t)cols);
    int col = (int)(i - (size_t)row * (size_t)cols);
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ6K *blk = &blocks[(size_t)row * n_bpr +
                                    col / BN_QK_K];
    out[i] = __float2half_rn(cuda_q6k_value(blk, col & (BN_QK_K - 1)));
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

static __device__ __forceinline__ int cuda_dp4a_i32(int a, int b, int c) {
    return __dp4a(a, b, c);
}

static __global__ void quantize_q8_1_kernel(BnCudaBlockQ8_1 *out,
                                            const float *x, int cols) {
    int block = blockIdx.x;
    int lane = threadIdx.x;
    int c = block * 32 + lane;
    float v = c < cols ? x[c] : 0.0f;
    float amax = fabsf(v);
    float sum = v;

    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    float d = amax / 127.0f;
    int q = d == 0.0f ? 0 : (int)rintf(v / d);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);

    out[block].qs[lane] = (int8_t)q;
    if (lane == 0) {
        out[block].d = cuda_fp32_to_fp16_bits(d);
        out[block].s = cuda_fp32_to_fp16_bits(sum);
    }
}

static __global__ void quantize_q8_1_batch_kernel(BnCudaBlockQ8_1 *out,
                                                  const float *x, int cols,
                                                  int n_tokens) {
    int block = blockIdx.x;
    int token = blockIdx.y;
    int lane = threadIdx.x;
    if (token >= n_tokens) return;
    int n_blocks = (cols + 31) / 32;
    int c = block * 32 + lane;
    const float *x_token = x + (size_t)token * cols;
    float v = c < cols ? x_token[c] : 0.0f;
    float amax = fabsf(v);
    float sum = v;

    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    float d = amax / 127.0f;
    int q = d == 0.0f ? 0 : (int)rintf(v / d);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);

    BnCudaBlockQ8_1 *dst = out + (size_t)token * n_blocks + block;
    dst->qs[lane] = (int8_t)q;
    if (lane == 0) {
        dst->d = cuda_fp32_to_fp16_bits(d);
        dst->s = cuda_fp32_to_fp16_bits(sum);
    }
}

static __global__ void quantize_q8k_batch_kernel(BnBlockQ8K *out,
                                                 const float *x, int cols,
                                                 int n_tokens) {
    int block = blockIdx.x;
    int token = blockIdx.y;
    int tid = threadIdx.x;
    if (token >= n_tokens || tid >= BN_QK_K) return;
    int n_blocks = (cols + BN_QK_K - 1) / BN_QK_K;
    int c = block * BN_QK_K + tid;
    const float *x_token = x + (size_t)token * cols;
    float v = c < cols ? x_token[c] : 0.0f;

    __shared__ float amax_s[BN_QK_K];
    __shared__ int q_s[BN_QK_K];
    amax_s[tid] = fabsf(v);
    __syncthreads();
    for (int stride = BN_QK_K / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            amax_s[tid] = fmaxf(amax_s[tid], amax_s[tid + stride]);
        __syncthreads();
    }

    float d = amax_s[0] / 127.0f;
    int q = d == 0.0f ? 0 : (int)rintf(v / d);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    q_s[tid] = q;
    BnBlockQ8K *dst = out + (size_t)token * n_blocks + block;
    dst->qs[tid] = (int8_t)q;
    __syncthreads();
    if (tid < BN_QK_K / 16) {
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 16; i++)
            sum += q_s[tid * 16 + i];
        dst->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0)
        dst->d = d;
}

static __device__ __forceinline__ int cuda_dot_i8x32_dp4a(const int8_t *a,
                                                          const int8_t *b) {
    int acc = 0;
    for (int i = 0; i < 32; i += 4) {
        int av;
        int bv;
        memcpy(&av, a + i, sizeof(av));
        memcpy(&bv, b + i, sizeof(bv));
        acc = cuda_dp4a_i32(av, bv, acc);
    }
    return acc;
}

static __global__ void q8_0_matvec_preq_warp8_kernel(
    float *out,
    const BnBlockQ8_0 *blocks,
    const BnCudaBlockQ8_1 *xq,
    const float *bias,
    int rows,
    int cols,
    size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;

    int n_bpr = cols / 32;
    const BnBlockQ8_0 *row_blocks = blocks + (size_t)row * n_bpr;
    float sum = 0.0f;
    for (int b = lane; b < n_bpr; b += 32) {
        const BnBlockQ8_0 *blk = &row_blocks[b];
        int dot = cuda_dot_i8x32_dp4a(blk->qs, xq[b].qs);
        sum += cuda_fp16_to_fp32(blk->d) *
               cuda_fp16_to_fp32(xq[b].d) * (float)dot;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) {
        if (bias) sum += bias[row];
        out[out_offset + (size_t)row] = sum;
    }
}

static __device__ float cuda_dot_row_q8_0_preq(
    const void *wdata, const BnCudaBlockQ8_1 *xq, int row, int cols) {
    const BnBlockQ8_0 *blocks = (const BnBlockQ8_0 *)wdata;
    int n_bpr = cols / 32;
    const BnBlockQ8_0 *row_blocks = blocks + (size_t)row * n_bpr;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int b = tid; b < n_bpr; b += blockDim.x) {
        const BnBlockQ8_0 *blk = &row_blocks[b];
        int dot = cuda_dot_i8x32_dp4a(blk->qs, xq[b].qs);
        sum += cuda_fp16_to_fp32(blk->d) *
               cuda_fp16_to_fp32(xq[b].d) * (float)dot;
    }
    return sum;
}

static __device__ __forceinline__ float cuda_vec_dot_q4k_q8_1(
    const BnBlockQ4K *blk, const BnCudaBlockQ8_1 *xq, int iqs) {
    int v[2];
    int u[4];
    float d8[2];

    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int q4_offset = 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    const int *q4 = (const int *)(blk->qs + q4_offset);
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t *scales = (const uint16_t *)blk->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) |
                 ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) |
                 ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t *sc = (const uint8_t *)aux;
    const uint8_t *mn = sc + 2;

#pragma unroll
    for (int i = 0; i < 2; i++) {
        const BnCudaBlockQ8_1 *xb = xq + bq8_offset + i;
        d8[i] = cuda_fp16_to_fp32(xb->d);
        const int *q8 = (const int *)xb->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    float sum_d = 0.0f;
    float sum_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot = cuda_dp4a_i32(v1i, u[2 * i + 1],
                                      cuda_dp4a_i32(v0i, u[2 * i + 0], 0));
        const int usum = cuda_dp4a_i32(0x01010101, u[2 * i + 1],
                                       cuda_dp4a_i32(0x01010101,
                                                     u[2 * i + 0], 0));
        sum_d += d8[i] * (float)(dot * sc[i]);
        sum_m += d8[i] * (float)(usum * mn[i]);
    }

    return cuda_fp16_to_fp32(blk->d) * sum_d -
           cuda_fp16_to_fp32(blk->dmin) * sum_m;
}

static __device__ __forceinline__ int cuda_q4k_dot_32(const uint8_t *qs,
                                                       const int8_t *q8,
                                                       int shift) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        int qv;
        int xv;
        memcpy(&qv, qs + i, sizeof(qv));
        memcpy(&xv, q8 + i, sizeof(xv));
        qv = (qv >> shift) & 0x0f0f0f0f;
        sum = cuda_dp4a_i32(qv, xv, sum);
    }
    return sum;
}

static __device__ __forceinline__ float cuda_vec_dot_q4k_q8k(
    const BnBlockQ4K *blk, const BnBlockQ8K *xq) {
    float xd = cuda_fp16_to_fp32(blk->d);
    float xmin = cuda_fp16_to_fp32(blk->dmin);
    int isum = 0;
    int summs = 0;
#pragma unroll
    for (int group = 0; group < 8; group++) {
        int sc = 0;
        int mn = 0;
        cuda_q4k_group_scale_min(blk, group, &sc, &mn);
        summs += mn * (int)(xq->bsums[2 * group] +
                            xq->bsums[2 * group + 1]);
        int byte_off = (group >> 1) * 32;
        int shift = (group & 1) ? 4 : 0;
        isum += sc * cuda_q4k_dot_32(blk->qs + byte_off,
                                     xq->qs + group * 32, shift);
    }
    return xq->d * xd * (float)isum - xq->d * xmin * (float)summs;
}

static __global__ void q4k_q8k_dot_matvec_kernel(float *out,
                                                 const BnBlockQ4K *blocks,
                                                 const BnBlockQ8K *xq,
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
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = lane; b < n_bpr; b += 32)
        sum += cuda_vec_dot_q4k_q8k(&row_blocks[b], xq + b);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) {
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __device__ __forceinline__ int cuda_q5k_hbits4(const uint8_t *qh,
                                                       int offset,
                                                       int bit) {
    int out = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int hb = ((qh[offset + i] >> bit) & 1) << 4;
        out |= hb << (8 * i);
    }
    return out;
}

static __device__ __forceinline__ float cuda_vec_dot_q5k_q8_1(
    const BnBlockQ5K *blk, const BnCudaBlockQ8_1 *xq, int iqs) {
    int qlow[2];
    int u[4];
    float d8[2];

    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int q5_offset = 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    const int h_offset = 4 * ((iqs / 2) % 4);
    const int *q5 = (const int *)(blk->qs + q5_offset);
    qlow[0] = q5[0];
    qlow[1] = q5[4];

    const uint16_t *scales = (const uint16_t *)blk->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) |
                 ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) |
                 ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t *sc = (const uint8_t *)aux;
    const uint8_t *mn = sc + 2;

#pragma unroll
    for (int i = 0; i < 2; i++) {
        const BnCudaBlockQ8_1 *xb = xq + bq8_offset + i;
        d8[i] = cuda_fp16_to_fp32(xb->d);
        const int *q8 = (const int *)xb->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    float sum_d = 0.0f;
    float sum_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int bit = bq8_offset + i;
        int v0 = cuda_q5k_hbits4(blk->qh, h_offset, bit) |
                 ((qlow[0] >> (4 * i)) & 0x0f0f0f0f);
        int v1 = cuda_q5k_hbits4(blk->qh, h_offset + 16, bit) |
                 ((qlow[1] >> (4 * i)) & 0x0f0f0f0f);
        const int dot = cuda_dp4a_i32(v1, u[2 * i + 1],
                                      cuda_dp4a_i32(v0, u[2 * i + 0], 0));
        const int usum = cuda_dp4a_i32(0x01010101, u[2 * i + 1],
                                       cuda_dp4a_i32(0x01010101,
                                                     u[2 * i + 0], 0));
        sum_d += d8[i] * (float)(dot * sc[i]);
        sum_m += d8[i] * (float)(usum * mn[i]);
    }

    return cuda_fp16_to_fp32(blk->d) * sum_d -
           cuda_fp16_to_fp32(blk->dmin) * sum_m;
}

static __global__ void q4k_dot_matvec_kernel(float *out,
                                             const BnBlockQ4K *blocks,
                                             const BnCudaBlockQ8_1 *xq,
                                             const float *bias,
                                             int rows, int cols,
                                             size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = kbx; b < n_bpr; b += 2)
        sum += cuda_vec_dot_q4k_q8_1(&row_blocks[b], xq + (size_t)b * 8,
                                     iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) {
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q4k_dot_matvec_4warp_kernel(float *out,
                                                   const BnBlockQ4K *blocks,
                                                   const BnCudaBlockQ8_1 *xq,
                                                   const float *bias,
                                                   int rows, int cols,
                                                   size_t out_offset) {
    __shared__ float partial[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= rows || warp >= 4) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = warp * 2 + kbx; b < n_bpr; b += 8)
        sum += cuda_vec_dot_q4k_q8_1(&row_blocks[b], xq + (size_t)b * 8,
                                     iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();

    if (warp == 0 && lane == 0) {
        sum = partial[0] + partial[1] + partial[2] + partial[3];
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q5k_dot_matvec_kernel(float *out,
                                             const BnBlockQ5K *blocks,
                                             const BnCudaBlockQ8_1 *xq,
                                             const float *bias,
                                             int rows, int cols,
                                             size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum = 0.0f;
    const BnBlockQ5K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = kbx; b < n_bpr; b += 2)
        sum += cuda_vec_dot_q5k_q8_1(&row_blocks[b], xq + (size_t)b * 8,
                                     iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) {
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q4k_dot_matmul_kernel(float *out,
                                             const BnBlockQ4K *blocks,
                                             const BnCudaBlockQ8_1 *xq,
                                             int rows, int cols,
                                             int n_tokens,
                                             size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token = blockIdx.y;
    if (row >= rows || token >= n_tokens) return;

    int n_bpr = cols / BN_QK_K;
    int x_blocks = (cols + 31) / 32;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnCudaBlockQ8_1 *xq_token = xq + (size_t)token * x_blocks;
    for (int b = kbx; b < n_bpr; b += 2)
        sum += cuda_vec_dot_q4k_q8_1(&row_blocks[b],
                                     xq_token + (size_t)b * 8, iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[out_offset + (size_t)token * rows + row] = sum;
}

static __global__ void q4k_dot_matvec_split_kernel(
    float *out0, float *out1, float *out2, const BnBlockQ4K *blocks,
    const BnCudaBlockQ8_1 *xq, const float *bias0, int total_rows, int cols,
    int split0, int split1, size_t out1_offset, size_t out2_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= total_rows) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = kbx; b < n_bpr; b += 2)
        sum += cuda_vec_dot_q4k_q8_1(&row_blocks[b], xq + (size_t)b * 8,
                                     iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane != 0) return;

    if (row < split0) {
        if (bias0) sum += bias0[row];
        out0[row] = sum;
    } else if (split1 > split0 && row >= split1) {
        if (out2)
            out2[out2_offset + (size_t)(row - split1)] = sum;
    } else {
        out1[out1_offset + (size_t)(row - split0)] = sum;
    }
}

static __global__ void q8_0_matvec_split_preq_warp8_kernel(
    float *out0, float *out1, float *out2, const BnBlockQ8_0 *blocks,
    const BnCudaBlockQ8_1 *xq, const float *bias0, int total_rows, int cols,
    int split0, int split1, size_t out1_offset, size_t out2_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= total_rows) return;

    int n_bpr = cols / 32;
    const BnBlockQ8_0 *row_blocks = blocks + (size_t)row * n_bpr;
    float sum = 0.0f;
    for (int b = lane; b < n_bpr; b += 32) {
        const BnBlockQ8_0 *blk = &row_blocks[b];
        int dot = cuda_dot_i8x32_dp4a(blk->qs, xq[b].qs);
        sum += cuda_fp16_to_fp32(blk->d) *
               cuda_fp16_to_fp32(xq[b].d) * (float)dot;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane != 0) return;

    if (row < split0) {
        if (bias0) sum += bias0[row];
        out0[row] = sum;
    } else if (split1 > split0 && row >= split1) {
        if (out2)
            out2[out2_offset + (size_t)(row - split1)] = sum;
    } else {
        out1[out1_offset + (size_t)(row - split0)] = sum;
    }
}

static __global__ void q5k_dot_matvec_split_kernel(
    float *out0, float *out1, float *out2, const BnBlockQ5K *blocks,
    const BnCudaBlockQ8_1 *xq, const float *bias0, int total_rows, int cols,
    int split0, int split1, size_t out1_offset, size_t out2_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= total_rows) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum = 0.0f;
    const BnBlockQ5K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = kbx; b < n_bpr; b += 2)
        sum += cuda_vec_dot_q5k_q8_1(&row_blocks[b], xq + (size_t)b * 8,
                                     iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane != 0) return;

    if (row < split0) {
        if (bias0) sum += bias0[row];
        out0[row] = sum;
    } else if (split1 > split0 && row >= split1) {
        if (out2)
            out2[out2_offset + (size_t)(row - split1)] = sum;
    } else {
        out1[out1_offset + (size_t)(row - split0)] = sum;
    }
}

static __global__ void q4k_q8k_dot_fused_gateup_silu_kernel(
    float *out, const BnBlockQ4K *blocks, const BnBlockQ8K *xq,
    int gate_rows, int up_rows, int cols) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= gate_rows || row >= up_rows) return;

    int n_bpr = cols / BN_QK_K;
    float gate = 0.0f;
    float up = 0.0f;
    const BnBlockQ4K *gate_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ4K *up_blocks =
        blocks + (size_t)(gate_rows + row) * n_bpr;
    for (int b = lane; b < n_bpr; b += 32) {
        gate += cuda_vec_dot_q4k_q8k(&gate_blocks[b], xq + b);
        up += cuda_vec_dot_q4k_q8k(&up_blocks[b], xq + b);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if (lane == 0)
        out[row] = (gate / (1.0f + __expf(-gate))) * up;
}

static __global__ void q4k_dot_fused_gateup_silu_kernel(
    float *out, const BnBlockQ4K *blocks, const BnCudaBlockQ8_1 *xq,
    int gate_rows, int up_rows, int cols) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= gate_rows || row >= up_rows) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float gate = 0.0f;
    float up = 0.0f;
    const BnBlockQ4K *gate_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ4K *up_blocks =
        blocks + (size_t)(gate_rows + row) * n_bpr;
    for (int b = kbx; b < n_bpr; b += 2) {
        const BnCudaBlockQ8_1 *xqb = xq + (size_t)b * 8;
        gate += cuda_vec_dot_q4k_q8_1(&gate_blocks[b], xqb, iqs);
        up += cuda_vec_dot_q4k_q8_1(&up_blocks[b], xqb, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if (lane == 0)
        out[row] = (gate / (1.0f + __expf(-gate))) * up;
}

static __global__ void q4k_dot_fused_gateup_silu_4warp_kernel(
    float *out, const BnBlockQ4K *blocks, const BnCudaBlockQ8_1 *xq,
    int gate_rows, int up_rows, int cols) {
    __shared__ float gate_partial[4];
    __shared__ float up_partial[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= gate_rows || row >= up_rows || warp >= 4) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float gate = 0.0f;
    float up = 0.0f;
    const BnBlockQ4K *gate_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ4K *up_blocks =
        blocks + (size_t)(gate_rows + row) * n_bpr;
    for (int b = warp * 2 + kbx; b < n_bpr; b += 8) {
        const BnCudaBlockQ8_1 *xqb = xq + (size_t)b * 8;
        gate += cuda_vec_dot_q4k_q8_1(&gate_blocks[b], xqb, iqs);
        up += cuda_vec_dot_q4k_q8_1(&up_blocks[b], xqb, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if (lane == 0) {
        gate_partial[warp] = gate;
        up_partial[warp] = up;
    }
    __syncthreads();

    if (warp == 0 && lane == 0) {
        gate = gate_partial[0] + gate_partial[1] +
               gate_partial[2] + gate_partial[3];
        up = up_partial[0] + up_partial[1] +
             up_partial[2] + up_partial[3];
        out[row] = (gate / (1.0f + __expf(-gate))) * up;
    }
}

static __global__ void q5k_dot_fused_gateup_silu_kernel(
    float *out, const BnBlockQ5K *blocks, const BnCudaBlockQ8_1 *xq,
    int gate_rows, int up_rows, int cols) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= gate_rows || row >= up_rows) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float gate = 0.0f;
    float up = 0.0f;
    const BnBlockQ5K *gate_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ5K *up_blocks =
        blocks + (size_t)(gate_rows + row) * n_bpr;
    for (int b = kbx; b < n_bpr; b += 2) {
        const BnCudaBlockQ8_1 *xqb = xq + (size_t)b * 8;
        gate += cuda_vec_dot_q5k_q8_1(&gate_blocks[b], xqb, iqs);
        up += cuda_vec_dot_q5k_q8_1(&up_blocks[b], xqb, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if (lane == 0)
        out[row] = (gate / (1.0f + __expf(-gate))) * up;
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

static __global__ void q8_0_matmul4_warp_kernel(float *out,
                                                const BnBlockQ8_0 *blocks,
                                                const float *x, int rows,
                                                int cols, int n_tokens,
                                                size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row0 = (blockIdx.x * warps_per_block + warp) * 4;
    int token = blockIdx.y;
    if (row0 >= rows || token >= n_tokens) return;

    int n_bpr = cols / 32;
    const float *x_token = x + (size_t)token * cols;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    const BnBlockQ8_0 *row_blocks0 = blocks + (size_t)row0 * n_bpr;
    const BnBlockQ8_0 *row_blocks1 = row_blocks0 + n_bpr;
    const BnBlockQ8_0 *row_blocks2 = row_blocks1 + n_bpr;
    const BnBlockQ8_0 *row_blocks3 = row_blocks2 + n_bpr;

    for (int b = 0; b < n_bpr; b++) {
        float xv = x_token[(size_t)b * 32 + lane];
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
        size_t base = out_offset + (size_t)token * rows + (size_t)row0;
        out[base] = sum0;
        if (row0 + 1 < rows) out[base + 1] = sum1;
        if (row0 + 2 < rows) out[base + 2] = sum2;
        if (row0 + 3 < rows) out[base + 3] = sum3;
    }
}

static __global__ void q5_0_matmul4_warp_kernel(float *out,
                                                const BnBlockQ5_0 *blocks,
                                                const float *x, int rows,
                                                int cols, int n_tokens,
                                                size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row0 = (blockIdx.x * warps_per_block + warp) * 4;
    int token = blockIdx.y;
    if (row0 >= rows || token >= n_tokens) return;

    int n_bpr = cols / 32;
    const float *x_token = x + (size_t)token * cols;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    const BnBlockQ5_0 *row_blocks0 = blocks + (size_t)row0 * n_bpr;
    const BnBlockQ5_0 *row_blocks1 = row_blocks0 + n_bpr;
    const BnBlockQ5_0 *row_blocks2 = row_blocks1 + n_bpr;
    const BnBlockQ5_0 *row_blocks3 = row_blocks2 + n_bpr;

    for (int b = 0; b < n_bpr; b++) {
        float xv = x_token[(size_t)b * 32 + lane];

        const BnBlockQ5_0 *blk0 = &row_blocks0[b];
        uint32_t qh0 = (uint32_t)blk0->qh[0] |
                       ((uint32_t)blk0->qh[1] << 8) |
                       ((uint32_t)blk0->qh[2] << 16) |
                       ((uint32_t)blk0->qh[3] << 24);
        uint8_t qs0 = blk0->qs[lane & 15];
        int q0 = lane < 16
            ? (int)((qs0 & 15) | (((qh0 >> lane) & 1u) << 4)) - 16
            : (int)((qs0 >> 4) | (((qh0 >> lane) & 1u) << 4)) - 16;
        sum0 += cuda_fp16_to_fp32(blk0->d) * (float)q0 * xv;

        if (row0 + 1 < rows) {
            const BnBlockQ5_0 *blk1 = &row_blocks1[b];
            uint32_t qh1 = (uint32_t)blk1->qh[0] |
                           ((uint32_t)blk1->qh[1] << 8) |
                           ((uint32_t)blk1->qh[2] << 16) |
                           ((uint32_t)blk1->qh[3] << 24);
            uint8_t qs1 = blk1->qs[lane & 15];
            int q1 = lane < 16
                ? (int)((qs1 & 15) | (((qh1 >> lane) & 1u) << 4)) - 16
                : (int)((qs1 >> 4) | (((qh1 >> lane) & 1u) << 4)) - 16;
            sum1 += cuda_fp16_to_fp32(blk1->d) * (float)q1 * xv;
        }
        if (row0 + 2 < rows) {
            const BnBlockQ5_0 *blk2 = &row_blocks2[b];
            uint32_t qh2 = (uint32_t)blk2->qh[0] |
                           ((uint32_t)blk2->qh[1] << 8) |
                           ((uint32_t)blk2->qh[2] << 16) |
                           ((uint32_t)blk2->qh[3] << 24);
            uint8_t qs2 = blk2->qs[lane & 15];
            int q2 = lane < 16
                ? (int)((qs2 & 15) | (((qh2 >> lane) & 1u) << 4)) - 16
                : (int)((qs2 >> 4) | (((qh2 >> lane) & 1u) << 4)) - 16;
            sum2 += cuda_fp16_to_fp32(blk2->d) * (float)q2 * xv;
        }
        if (row0 + 3 < rows) {
            const BnBlockQ5_0 *blk3 = &row_blocks3[b];
            uint32_t qh3 = (uint32_t)blk3->qh[0] |
                           ((uint32_t)blk3->qh[1] << 8) |
                           ((uint32_t)blk3->qh[2] << 16) |
                           ((uint32_t)blk3->qh[3] << 24);
            uint8_t qs3 = blk3->qs[lane & 15];
            int q3 = lane < 16
                ? (int)((qs3 & 15) | (((qh3 >> lane) & 1u) << 4)) - 16
                : (int)((qs3 >> 4) | (((qh3 >> lane) & 1u) << 4)) - 16;
            sum3 += cuda_fp16_to_fp32(blk3->d) * (float)q3 * xv;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
    }
    if (lane == 0) {
        size_t base = out_offset + (size_t)token * rows + (size_t)row0;
        out[base] = sum0;
        if (row0 + 1 < rows) out[base + 1] = sum1;
        if (row0 + 2 < rows) out[base + 2] = sum2;
        if (row0 + 3 < rows) out[base + 3] = sum3;
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

static __device__ __forceinline__ int cuda_q6k_dot_16(
    const uint8_t *ql, const uint8_t *qh, const int8_t *q8, int chunk,
    int group) {
    int acc = 0;
    int sub = group >> 1;
    int half = group & 1;
    int base = chunk * 128 + sub * 32 + half * 16;
    int ql_off = chunk * 64 + (sub & 1) * 32;
    int qh_off = chunk * 32;
    int lo_high = sub >= 2;
    int qh_shift = (sub & 1) * 2 + lo_high * 4;
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        int qv = 0;
        int xv = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int idx = half * 16 + i + j;
            int lo = ql[ql_off + idx];
            int q = lo_high ? (lo >> 4) : (lo & 15);
            q |= ((qh[qh_off + idx] >> qh_shift) & 3) << 4;
            qv |= (q & 0xff) << (8 * j);
            xv |= ((int)(uint8_t)q8[base + i + j]) << (8 * j);
        }
        acc = cuda_dp4a_i32(qv, xv, acc);
    }
    return acc;
}

static __device__ __forceinline__ float cuda_vec_dot_q6k_q8k(
    const BnBlockQ6K *blk, const BnBlockQ8K *xq) {
    int isum = 0;
    int corr = 0;
#pragma unroll
    for (int chunk = 0; chunk < 2; chunk++) {
#pragma unroll
        for (int group = 0; group < 8; group++) {
            int scale_idx = chunk * 8 + group;
            int s = (int)blk->scales[scale_idx];
            int dot = cuda_q6k_dot_16(blk->ql, blk->qh, xq->qs, chunk,
                                      group);
            isum += s * dot;
            corr += s * (int)xq->bsums[scale_idx];
        }
    }
    return cuda_fp16_to_fp32(blk->d) * xq->d *
           (float)(isum - (corr << 5));
}

static __global__ void q6k_dot_matvec_kernel(float *out,
                                             const BnBlockQ6K *blocks,
                                             const BnBlockQ8K *xq,
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
    for (int b = lane; b < n_bpr; b += 32)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], xq + b);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) {
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q6k_dot_matmul_kernel(float *out,
                                             const BnBlockQ6K *blocks,
                                             const BnBlockQ8K *xq,
                                             int rows, int cols,
                                             int n_tokens,
                                             size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token = blockIdx.y;
    if (row >= rows || token >= n_tokens) return;

    int n_bpr = cols / BN_QK_K;
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ8K *xq_token = xq + (size_t)token * n_bpr;
    for (int b = lane; b < n_bpr; b += 32)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], xq_token + b);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[out_offset + (size_t)token * rows + row] = sum;
}

static __global__ void q6k_matmul_warp_kernel(float *out,
                                              const BnBlockQ6K *blocks,
                                              const float *x,
                                              int rows, int cols,
                                              int n_tokens,
                                              size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token = blockIdx.y;
    if (row >= rows || token >= n_tokens) return;

    int n_bpr = cols / BN_QK_K;
    const float *x_token = x + (size_t)token * cols;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    float sum = 0.0f;
    for (int b = 0; b < n_bpr; b++) {
        const BnBlockQ6K *blk = &row_blocks[b];
        for (int i = lane; i < BN_QK_K; i += 32)
            sum += cuda_q6k_value(blk, i) *
                   x_token[(size_t)b * BN_QK_K + i];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[out_offset + (size_t)token * rows + row] = sum;
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
    float sum2 = 0.0f;
    if (split1 == 1 && row < split0)
        sum2 = cuda_dot_row(wdata, x, row + split0, cols, type);
    extern __shared__ float scratch[];
    float *scratch2 = scratch + blockDim.x;
    sum = cuda_block_reduce_sum(sum, scratch);
    if (split1 == 1)
        sum2 = cuda_block_reduce_sum(sum2, scratch2);
    if (tid != 0) return;

    if (split1 == 1) {
        if (row < split0) {
            float gate = sum;
            out0[row] = (gate / (1.0f + expf(-gate))) * sum2;
        }
    } else if (row < split0) {
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
    const BnCudaBlockQ8_1 *xq,
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
        int half_rope = rope_dims / 2;
        if (k_pair_write && k_rotate && k_dim >= half_rope)
            return;
    }

    int dot_row = out_kind == 2 ? local : row;
    float sum = (type == BN_GGUF_TENSOR_Q8_0 && xq && (cols & 31) == 0)
        ? cuda_dot_row_q8_0_preq(wdata, xq, dot_row, cols)
        : cuda_dot_row(wdata, x, dot_row, cols, type);
    if (out_kind == 1 && k_rotate) {
        int half_rope = rope_dims / 2;
        int pair_dim = k_dim < half_rope
            ? k_dim + half_rope
            : k_dim - half_rope;
        int pair_row = q_rows + k_head * head_size + pair_dim;
        pair_sum =
            (qk_type == BN_GGUF_TENSOR_Q8_0 && xq && (cols & 31) == 0)
            ? cuda_dot_row_q8_0_preq(qk_wdata, xq, pair_row, cols)
            : cuda_dot_row(qk_wdata, x, pair_row, cols, qk_type);
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
                int half_rope = rope_dims / 2;
                int pair_dim = k_dim < half_rope
                    ? k_dim + half_rope
                    : k_dim - half_rope;
                float pair = pair_sum +
                    (k_bias ? k_bias[k_head * head_size + pair_dim] : 0.0f);
                int freq_idx = k_dim < half_rope ? k_dim : pair_dim;
                float angle = (float)pos * freq[freq_idx];
                float s, c;
                __sincosf(angle, &s, &c);
                if (k_pair_write) {
                    cuda_kv_store(key_cache, key_offset + (size_t)local,
                                  v * c - pair * s, kv_f16);
                    cuda_kv_store(key_cache,
                                  key_offset + (size_t)(k_head * head_size +
                                                        pair_dim),
                                  v * s + pair * c, kv_f16);
                } else {
                    float x0 = k_dim < half_rope ? v : pair;
                    float x1 = k_dim < half_rope ? pair : v;
                    cuda_kv_store(key_cache, key_offset + (size_t)local,
                                  k_dim < half_rope ? (x0 * c - x1 * s)
                                                    : (x0 * s + x1 * c),
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
    const BnCudaBlockQ8_1 *xq,
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
        int half_rope = rope_dims / 2;
        if (k_pair_write && k_rotate && k_dim >= half_rope)
            return;
    }

    int dot_row = out_kind == 2 ? local : row;
    float sum = (type == BN_GGUF_TENSOR_Q8_0 && xq && (cols & 31) == 0)
        ? cuda_dot_row_q8_0_preq(wdata, xq, dot_row, cols)
        : cuda_dot_row(wdata, x, dot_row, cols, type);
    if (out_kind == 1 && k_rotate) {
        int half_rope = rope_dims / 2;
        int pair_dim = k_dim < half_rope
            ? k_dim + half_rope
            : k_dim - half_rope;
        int pair_row = q_rows + k_head * head_size + pair_dim;
        pair_sum =
            (qk_type == BN_GGUF_TENSOR_Q8_0 && xq && (cols & 31) == 0)
            ? cuda_dot_row_q8_0_preq(qk_wdata, xq, pair_row, cols)
            : cuda_dot_row(qk_wdata, x, pair_row, cols, qk_type);
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
                int half_rope = rope_dims / 2;
                int pair_dim = k_dim < half_rope
                    ? k_dim + half_rope
                    : k_dim - half_rope;
                float pair = pair_sum +
                    (k_bias ? k_bias[k_head * head_size + pair_dim] : 0.0f);
                int freq_idx = k_dim < half_rope ? k_dim : pair_dim;
                float angle = (float)pos * freq[freq_idx];
                float s, c;
                __sincosf(angle, &s, &c);
                if (k_pair_write) {
                    cuda_kv_store(key_cache, key_offset + (size_t)local,
                                  v * c - pair * s, kv_f16);
                    cuda_kv_store(key_cache,
                                  key_offset + (size_t)(k_head * head_size +
                                                        pair_dim),
                                  v * s + pair * c, kv_f16);
                } else {
                    float x0 = k_dim < half_rope ? v : pair;
                    float x1 = k_dim < half_rope ? pair : v;
                    cuda_kv_store(key_cache, key_offset + (size_t)local,
                                  k_dim < half_rope ? (x0 * c - x1 * s)
                                                    : (x0 * s + x1 * c),
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
    const BnCudaBlockQ8_1 *xq,
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
        int half_rope = rope_dims / 2;
        if (k_pair_write && k_rotate && k_dim >= half_rope)
            return;
    }

    int dot_row = out_kind == 1 ? q_offset + local : local;
    float sum = (type == BN_GGUF_TENSOR_Q8_0 && xq && (cols & 31) == 0)
        ? cuda_dot_row_q8_0_preq(wdata, xq, dot_row, cols)
        : cuda_dot_row(wdata, x, dot_row, cols, type);
    if (out_kind == 1 && k_rotate) {
        int half_rope = rope_dims / 2;
        int pair_dim = k_dim < half_rope
            ? k_dim + half_rope
            : k_dim - half_rope;
        int pair_row = q_offset + k_head * head_size + pair_dim;
        pair_sum =
            (qk_type == BN_GGUF_TENSOR_Q8_0 && xq && (cols & 31) == 0)
            ? cuda_dot_row_q8_0_preq(qk_wdata, xq, pair_row, cols)
            : cuda_dot_row(qk_wdata, x, pair_row, cols, qk_type);
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
            int half_rope = rope_dims / 2;
            int pair_dim = k_dim < half_rope
                ? k_dim + half_rope
                : k_dim - half_rope;
            float pair = pair_sum +
                (k_bias ? k_bias[k_head * head_size + pair_dim] : 0.0f);
            int freq_idx = k_dim < half_rope ? k_dim : pair_dim;
            float angle = (float)pos * freq[freq_idx];
            float s, c;
            __sincosf(angle, &s, &c);
            if (k_pair_write) {
                cuda_kv_store(key_cache, key_offset + (size_t)local,
                              v * c - pair * s, kv_f16);
                cuda_kv_store(key_cache,
                              key_offset + (size_t)(k_head * head_size +
                                                    pair_dim),
                              v * s + pair * c, kv_f16);
            } else {
                float x0 = k_dim < half_rope ? v : pair;
                float x1 = k_dim < half_rope ? pair : v;
                cuda_kv_store(key_cache, key_offset + (size_t)local,
                              k_dim < half_rope ? (x0 * c - x1 * s)
                                                : (x0 * s + x1 * c),
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

static __global__ void rmsnorm_batch_kernel(float *out, const float *x,
                                            const float *weight, int n,
                                            int n_tokens, float eps) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= n_tokens || n <= 0) return;
    const float *xt = x + (size_t)t * n;
    float *ot = out + (size_t)t * n;

    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        ss += xt[i] * xt[i];

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
        ot[i] = xt[i] * scale * weight[i];
}

static __global__ void per_head_rmsnorm_kernel(float *x,
                                               const float *weight,
                                               int n_heads,
                                               int head_size,
                                               float eps,
                                               int per_head_weight,
                                               size_t x_offset) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= n_heads || head_size <= 0) return;

    float *xh = x + x_offset + (size_t)h * head_size;
    const float *wh = weight + (per_head_weight ? (size_t)h * head_size : 0);
    float ss = 0.0f;
    for (int i = tid; i < head_size; i += blockDim.x)
        ss += xh[i] * xh[i];

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

    float scale = rsqrtf(scratch[0] / (float)head_size + eps);
    for (int i = tid; i < head_size; i += blockDim.x)
        xh[i] = xh[i] * scale * wh[i];
}

static __global__ void qk_rmsnorm_rope_kernel(
    float *q, float *k, const float *q_weight, const float *k_weight,
    const float *freq, int n_heads, int n_kv_heads, int head_size,
    float eps, int per_head_weight, size_t k_offset, int pos,
    int rope_dims) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int is_q = h < n_heads;
    int idx = is_q ? h : h - n_heads;
    int n = is_q ? n_heads : n_kv_heads;
    if (idx >= n || head_size <= 0) return;

    float *xh = is_q
        ? q + (size_t)idx * head_size
        : k + k_offset + (size_t)idx * head_size;
    const float *weight = is_q ? q_weight : k_weight;
    const float *wh = weight + (per_head_weight ? (size_t)idx * head_size : 0);

    float ss = 0.0f;
    for (int i = tid; i < head_size; i += blockDim.x)
        ss += xh[i] * xh[i];

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

    float scale = rsqrtf(scratch[0] / (float)head_size + eps);
    for (int i = tid; i < head_size; i += blockDim.x)
        xh[i] = xh[i] * scale * wh[i];
    __syncthreads();

    int half_rope = rope_dims / 2;
    for (int i = tid; i < half_rope; i += blockDim.x) {
        int j = i + half_rope;
        float angle = (float)pos * freq[i];
        float s, c;
        __sincosf(angle, &s, &c);
        float x0 = xh[i];
        float x1 = xh[j];
        xh[i] = x0 * c - x1 * s;
        xh[j] = x0 * s + x1 * c;
    }
}

static __global__ void split_qk_prefill_kernel(const float *qk,
                                               float *q,
                                               float *k,
                                               int n_tokens,
                                               int q_dim,
                                               int kv_dim,
                                               int qk_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * (q_dim + kv_dim);
    if (i >= total) return;
    int row = i / (q_dim + kv_dim);
    int col = i - row * (q_dim + kv_dim);
    const float *src = qk + (size_t)row * qk_rows;
    if (col < q_dim)
        q[(size_t)row * q_dim + col] = src[col];
    else
        k[(size_t)row * kv_dim + (col - q_dim)] = src[q_dim + (col - q_dim)];
}

static __global__ void qk_prefill_rmsnorm_rope_kernel(
    float *q, float *k, const float *q_weight, const float *k_weight,
    const float *freq, int n_tokens, int pos0, int n_heads,
    int n_kv_heads, int head_size, float eps, int per_head_weight,
    int rope_dims) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;
    int is_q = h < n_heads;
    int idx = is_q ? h : h - n_heads;
    int n = is_q ? n_heads : n_kv_heads;
    if (t >= n_tokens || idx >= n || head_size <= 0) return;

    int q_dim = n_heads * head_size;
    int kv_dim = n_kv_heads * head_size;
    float *xh = is_q
        ? q + (size_t)t * q_dim + (size_t)idx * head_size
        : k + (size_t)t * kv_dim + (size_t)idx * head_size;
    const float *weight = is_q ? q_weight : k_weight;

    if (weight) {
        const float *wh = weight +
            (per_head_weight ? (size_t)idx * head_size : 0);
        float ss = 0.0f;
        for (int i = tid; i < head_size; i += blockDim.x)
            ss += xh[i] * xh[i];

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

        float scale = rsqrtf(scratch[0] / (float)head_size + eps);
        for (int i = tid; i < head_size; i += blockDim.x)
            xh[i] = xh[i] * scale * wh[i];
        __syncthreads();
    }

    int half_rope = rope_dims / 2;
    for (int i = tid; i < half_rope; i += blockDim.x) {
        int j = i + half_rope;
        float angle = (float)(pos0 + t) * freq[i];
        float s, c;
        __sincosf(angle, &s, &c);
        float x0 = xh[i];
        float x1 = xh[j];
        xh[i] = x0 * c - x1 * s;
        xh[j] = x0 * s + x1 * c;
    }
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

static __global__ void residual_rmsnorm_batch_copy_kernel(
        float *x, const float *r, float *residual_out, float *norm_out,
        const float *weight, int n, int n_tokens, float eps) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= n_tokens || n <= 0) return;
    float *xt = x + (size_t)t * n;
    const float *rt = r + (size_t)t * n;
    float *res_t = residual_out + (size_t)t * n;
    float *norm_t = norm_out + (size_t)t * n;

    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = xt[i] + rt[i];
        xt[i] = v;
        res_t[i] = v;
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
        norm_t[i] = xt[i] * scale * weight[i];
}

static __global__ void copy_kernel(const float *src, float *dst,
                                   int src_off, int dst_off, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[dst_off + i] = src[src_off + i];
}

static __global__ void deinterleave_q_kernel(const float *src, float *dst,
                                             int q_dim, int head_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= q_dim) return;
    int h = i / head_size;
    int d = i - h * head_size;
    dst[i] = src[(size_t)h * 2u * (size_t)head_size + (size_t)d];
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
    int half_rope = rope_dims / 2;
    for (int i = tid; i < half_rope; i += blockDim.x) {
        int j = i + half_rope;
        float angle = (float)pos * freq[i];
        float s, c;
        __sincosf(angle, &s, &c);
        float x0 = x[base + i] + bias[base + i];
        float x1 = x[base + j] + bias[base + j];
        float y0 = x0 * c - x1 * s;
        float y1 = x0 * s + x1 * c;
        x[base + i] = y0;
        x[base + j] = y1;
        dst[dst_off + base + i] = y0;
        dst[dst_off + base + j] = y1;
    }
}

static __global__ void residual_add_kernel(float *x, const float *r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += r[i];
}

static __global__ void weighted_add_kernel(float *x, const float *r,
                                           float weight, int n, int reset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = weight * r[i];
        x[i] = reset ? v : x[i] + v;
    }
}

static __global__ void activation_gate_kernel(float *x, const float *aux,
                                              int n, int kind, int param1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (kind == BN_GPU_CODE_RELU2_GATE) {
        float r = v > 0.0f ? v : 0.0f;
        x[i] = r * r * aux[i];
    } else if (kind == BN_GPU_CODE_SIGMOID_GATE) {
        if (param1 > 0) {
            int h = i / param1;
            int d = i - h * param1;
            float gate = aux[(size_t)h * 2u * (size_t)param1 +
                             (size_t)param1 + (size_t)d];
            x[i] = v / (1.0f + __expf(-gate));
        } else {
            x[i] = aux[i] / (1.0f + __expf(-v));
        }
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

static __device__ BnCudaArgmaxPair argmax_better(BnCudaArgmaxPair a,
                                                 BnCudaArgmaxPair b) {
    if (b.value > a.value || (b.value == a.value && b.index < a.index))
        return b;
    return a;
}

static __global__ void argmax_penalty_stage1_kernel(
        BnCudaArgmaxPair *partials, const float *x, int n,
        const int *penalty_tokens, int n_penalty_tokens,
        float repeat_penalty) {
    extern __shared__ BnCudaArgmaxPair sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    BnCudaArgmaxPair best;
    best.value = -INFINITY;
    best.index = 0;
    if (idx < n) {
        float v = x[idx];
        if (repeat_penalty != 1.0f && penalty_tokens &&
            n_penalty_tokens > 0) {
            for (int i = 0; i < n_penalty_tokens; i++) {
                if (penalty_tokens[i] == idx) {
                    v = v > 0.0f ? v / repeat_penalty : v * repeat_penalty;
                    break;
                }
            }
        }
        best.value = v;
        best.index = idx;
    }
    sdata[tid] = best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] = argmax_better(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0)
        partials[blockIdx.x] = sdata[0];
}

static __global__ void argmax_stage2_kernel(int *out,
                                            const BnCudaArgmaxPair *partials,
                                            int n_partials) {
    extern __shared__ BnCudaArgmaxPair sdata[];
    int tid = threadIdx.x;
    BnCudaArgmaxPair best;
    best.value = -INFINITY;
    best.index = 0;
    for (int i = tid; i < n_partials; i += blockDim.x)
        best = argmax_better(best, partials[i]);
    sdata[tid] = best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] = argmax_better(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0)
        *out = sdata[0].index;
}

static __global__ void ssm_conv_silu_kernel(
    float *qkv, float *conv_state, const float *conv1d_w,
    int qkv_dim, int kern, size_t conv_off) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= qkv_dim || kern < 2) return;
    float sum = 0.0f;
    for (int k = 0; k < kern - 1; k++)
        sum += conv_state[conv_off + (size_t)k * qkv_dim + ch] *
               conv1d_w[(size_t)ch * kern + k];
    float cur = qkv[ch];
    sum += cur * conv1d_w[(size_t)ch * kern + (kern - 1)];
    for (int k = 0; k < kern - 2; k++)
        conv_state[conv_off + (size_t)k * qkv_dim + ch] =
            conv_state[conv_off + (size_t)(k + 1) * qkv_dim + ch];
    conv_state[conv_off + (size_t)(kern - 2) * qkv_dim + ch] = cur;
    qkv[ch] = sum / (1.0f + expf(-sum));
}

static __global__ void ssm_l2norm_kernel(
    float *q, float *k, int head_dim, int q_off, int k_off) {
    int head = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float scratch[];
    float qn = 0.0f;
    float kn = 0.0f;
    size_t qb = (size_t)q_off + (size_t)head * head_dim;
    size_t kb = (size_t)k_off + (size_t)head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float qv = q[qb + d];
        float kv = k[kb + d];
        qn += qv * qv;
        kn += kv * kv;
    }
    float *scratch_q = scratch;
    float *scratch_k = scratch + 8;
    qn = cuda_block_reduce_sum_all(qn, scratch_q);
    kn = cuda_block_reduce_sum_all(kn, scratch_k);
    float inv_q = 1.0f / (sqrtf(qn) + 1e-6f);
    float inv_k = 1.0f / (sqrtf(kn) + 1e-6f);
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q[qb + d] *= inv_q;
        k[kb + d] *= inv_k;
    }
}

static __global__ void ssm_alpha_beta_kernel(
    float *alpha, float *beta, const float *dt_bias, const float *a_log,
    int n) {
    int h = threadIdx.x;
    if (h >= n) return;
    float dt = alpha[h] + dt_bias[h];
    float dt_sp = dt > 20.0f ? dt : logf(1.0f + expf(dt));
    alpha[h] = expf(dt_sp * a_log[h]);
    beta[h] = 1.0f / (1.0f + expf(-beta[h]));
}

static __global__ void ssm_alpha_beta_split_kernel(
    const float *src, float *alpha, float *beta, const float *dt_bias,
    const float *a_log, int n, int beta_off) {
    int h = threadIdx.x;
    if (h >= n) return;
    float dt = src[h] + dt_bias[h];
    float dt_sp = dt > 20.0f ? dt : logf(1.0f + expf(dt));
    alpha[h] = expf(dt_sp * a_log[h]);
    float b = src[beta_off + h];
    beta[h] = 1.0f / (1.0f + expf(-b));
}

static __global__ void ssm_delta_kernel(
    float *state, float *out, const float *q, const float *k,
    const float *v, const float *alpha, const float *beta,
    int head_k_dim, int head_v_dim, int num_k_heads, float q_scale,
    size_t state_off, int q_off, int k_off, int v_off) {
    int hv_idx = blockIdx.x;
    int tid = threadIdx.x;
    int hk_idx = hv_idx % num_k_heads;
    extern __shared__ float sk[];
    size_t state_base = state_off +
        (size_t)hv_idx * (size_t)head_k_dim * (size_t)head_v_dim;
    float decay = alpha[hv_idx];
    float b = beta[hv_idx];
    int total = head_k_dim * head_v_dim;

    for (int i = tid; i < total; i += blockDim.x)
        state[state_base + i] *= decay;
    __syncthreads();

    for (int vi = tid; vi < head_v_dim; vi += blockDim.x) {
        float sum = 0.0f;
        float comp = 0.0f;
        for (int ki = 0; ki < head_k_dim; ki++) {
            float y = state[state_base + (size_t)ki * head_v_dim + vi] *
                      k[(size_t)k_off + (size_t)hk_idx * head_k_dim + ki] -
                      comp;
            float t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        sk[vi] = sum;
    }
    __syncthreads();

    for (int i = tid; i < total; i += blockDim.x) {
        int ki = i / head_v_dim;
        int vi = i - ki * head_v_dim;
        float kk = k[(size_t)k_off + (size_t)hk_idx * head_k_dim + ki];
        state[state_base + i] +=
            kk * b * (v[(size_t)v_off + (size_t)hv_idx * head_v_dim + vi] -
                      sk[vi]);
    }
    __syncthreads();

    for (int vi = tid; vi < head_v_dim; vi += blockDim.x) {
        float sum = 0.0f;
        float comp = 0.0f;
        for (int ki = 0; ki < head_k_dim; ki++) {
            float y = state[state_base + (size_t)ki * head_v_dim + vi] *
                      q[(size_t)q_off + (size_t)hk_idx * head_k_dim + ki] -
                      comp;
            float t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        out[(size_t)hv_idx * head_v_dim + vi] = sum * q_scale;
    }
}

static __global__ void ssm_delta_128_warp_kernel(
    float *state, float *out, const float *q, const float *k,
    const float *v, const float *alpha, const float *beta,
    int num_k_heads, float q_scale, size_t state_off,
    int q_off, int k_off, int v_off) {
    const int hv_idx = blockIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= 128) return;

    const int hk_idx = hv_idx % num_k_heads;
    const size_t state_base = state_off + (size_t)hv_idx * 128u * 128u;
    const float decay = alpha[hv_idx];
    const float b = beta[hv_idx];
    const float *qh = q + (size_t)q_off + (size_t)hk_idx * 128u;
    const float *kh = k + (size_t)k_off + (size_t)hk_idx * 128u;
    const float *vh = v + (size_t)v_off + (size_t)hv_idx * 128u;

    float s_shard[4];
    float k_reg[4];
    float q_reg[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int row = r * 32 + lane;
        size_t idx = state_base + (size_t)row * 128u + (size_t)col;
        s_shard[r] = state[idx] * decay;
        k_reg[r] = kh[row];
        q_reg[r] = qh[row];
    }

    float kv_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; r++)
        kv_partial += s_shard[r] * k_reg[r];
    for (int offset = 16; offset > 0; offset >>= 1)
        kv_partial += __shfl_down_sync(0xffffffffu, kv_partial, offset);
    float kv_col = __shfl_sync(0xffffffffu, kv_partial, 0);
    float delta = (vh[col] - kv_col) * b;

    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        s_shard[r] += k_reg[r] * delta;
        attn_partial += s_shard[r] * q_reg[r];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        attn_partial += __shfl_down_sync(0xffffffffu, attn_partial, offset);
    if (lane == 0)
        out[(size_t)hv_idx * 128u + (size_t)col] =
            attn_partial * q_scale;

#pragma unroll
    for (int r = 0; r < 4; r++) {
        int row = r * 32 + lane;
        state[state_base + (size_t)row * 128u + (size_t)col] = s_shard[r];
    }
}

static __global__ void ssm_gate_kernel(
    float *out, const float *z, const float *norm_w, int head_v_dim,
    float eps) {
    int hv_idx = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float scratch[];
    size_t base = (size_t)hv_idx * head_v_dim;
    float ss = 0.0f;
    for (int d = tid; d < head_v_dim; d += blockDim.x) {
        float v = out[base + d];
        ss += v * v;
    }
    ss = cuda_block_reduce_sum_all(ss, scratch);
    float inv = 1.0f / sqrtf(ss / (float)head_v_dim + eps);
    for (int d = tid; d < head_v_dim; d += blockDim.x) {
        float normed = out[base + d] * inv * norm_w[d];
        float g = z[base + d];
        out[base + d] = normed * (g / (1.0f + expf(-g)));
    }
}

static __global__ void rope_kernel(float *q, float *k, const float *freq,
                                   int n_heads, int head_size, int pos,
                                   int rope_dims, int n_kv_heads,
                                   uint32_t kv_cache_off) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    int half_rope = rope_dims / 2;
    if (h >= n_heads || i >= half_rope) return;
    int j = i + half_rope;
    float angle = (float)pos * freq[i];
    float s, c;
    __sincosf(angle, &s, &c);
    float *qh = q + (size_t)h * head_size;
    float x0 = qh[i], x1 = qh[j];
    qh[i] = x0 * c - x1 * s;
    qh[j] = x0 * s + x1 * c;
    if (k && h < n_kv_heads) {
        float *kh = k + kv_cache_off + (size_t)h * head_size;
        x0 = kh[i]; x1 = kh[j];
        kh[i] = x0 * c - x1 * s;
        kh[j] = x0 * s + x1 * c;
    }
}

static __global__ void prefill_write_kv_cache_kernel(
        float *key_cache, float *value_cache, const float *k, const float *v,
        int n_tokens, int kv_dim, int kv_cache_stride,
        uint32_t kv_cache_off) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * kv_dim;
    if (idx >= total) return;
    int t = idx / kv_dim;
    int d = idx - t * kv_dim;
    size_t dst = (size_t)kv_cache_off + (size_t)t * kv_cache_stride +
                 (size_t)d;
    key_cache[dst] = k[idx];
    value_cache[dst] = v[idx];
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

static __global__ void prefill_attention_kernel(
    float *out, const float *q, const float *k, const float *v,
    int n_tokens, int n_heads, int head_size, int kv_mul, int kv_dim,
    float attention_scale) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;
    if (h >= n_heads || t >= n_tokens) return;
    int kv_h = h / kv_mul;
    const float *qh = q + ((size_t)t * n_heads + (size_t)h) * head_size;

    extern __shared__ float shared[];
    float *scores = shared;
    float *scratch = shared + n_tokens;

    for (int j = tid; j <= t; j += blockDim.x) {
        const float *kh = k + (size_t)j * kv_dim + (size_t)kv_h * head_size;
        float score = 0.0f;
        for (int d = 0; d < head_size; d++)
            score += qh[d] * kh[d];
        scores[j] = score * attention_scale;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int j = tid; j <= t; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);
    float max_score = cuda_block_reduce_max_all(local_max, scratch);

    float local_sum = 0.0f;
    for (int j = tid; j <= t; j += blockDim.x) {
        float p = __expf(scores[j] - max_score);
        scores[j] = p;
        local_sum += p;
    }
    float inv_sum = 1.0f / cuda_block_reduce_sum_all(local_sum, scratch);

    for (int d = tid; d < head_size; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j <= t; j++) {
            const float *vh =
                v + (size_t)j * kv_dim + (size_t)kv_h * head_size;
            acc += scores[j] * inv_sum * vh[d];
        }
        out[((size_t)t * n_heads + (size_t)h) * head_size + d] = acc;
    }
}

static __global__ void prefill_causal_softmax_kernel(float *scores,
                                                     int n_tokens) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;
    if (t >= n_tokens) return;
    float *col = scores + (size_t)h * n_tokens * n_tokens +
                 (size_t)t * n_tokens;

    extern __shared__ float scratch[];
    float local_max = -INFINITY;
    for (int j = tid; j <= t; j += blockDim.x)
        local_max = fmaxf(local_max, col[j]);
    float max_score = cuda_block_reduce_max_all(local_max, scratch);

    float local_sum = 0.0f;
    for (int j = tid; j < n_tokens; j += blockDim.x) {
        float p = 0.0f;
        if (j <= t) {
            p = __expf(col[j] - max_score);
            local_sum += p;
        }
        col[j] = p;
    }
    float inv_sum = 1.0f / cuda_block_reduce_sum_all(local_sum, scratch);
    for (int j = tid; j <= t; j += blockDim.x)
        col[j] *= inv_sum;
}

static int cuda_prefill_attention_gemm(BnCudaCtx *ctx, float *d_out,
                                       const float *d_q, const float *d_k,
                                       const float *d_v, float *d_scores,
                                       int n_tokens, int n_heads,
                                       int n_kv_heads, int head_size,
                                       int kv_mul, int kv_dim,
                                       float attention_scale) {
    if (!ctx || !ctx->cublas || !d_out || !d_q || !d_k || !d_v ||
        !d_scores || n_tokens <= 1 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_size <= 0 || kv_mul <= 0 || kv_dim <= 0 ||
        n_heads / kv_mul != n_kv_heads)
        return -1;

    const float alpha = attention_scale;
    const float one = 1.0f;
    const float zero = 0.0f;
    int q_ld = n_heads * head_size;

    if (!getenv("BN_CUDA_DISABLE_PREFILL_BATCHED_GEMM") &&
        cuda_ensure_gemm_ptrs(ctx, n_heads * 3) == 0) {
        const float **d_a = (const float **)ctx->d_gemm_ptrs;
        const float **d_b = d_a + n_heads;
        float **d_c = (float **)(d_b + n_heads);
        void **h_a = ctx->h_gemm_ptrs;
        void **h_b = h_a + n_heads;
        void **h_c = h_b + n_heads;
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / kv_mul;
            h_a[h] = (void *)(d_k + (size_t)kv_h * head_size);
            h_b[h] = (void *)(d_q + (size_t)h * head_size);
            h_c[h] = (void *)(d_scores + (size_t)h * n_tokens * n_tokens);
        }
        size_t ptr_bytes = (size_t)n_heads * 3u * sizeof(void *);
        cudaError_t copy_err = cudaMemcpy(ctx->d_gemm_ptrs, ctx->h_gemm_ptrs,
                                          ptr_bytes, cudaMemcpyHostToDevice);
        if (copy_err != cudaSuccess) {
            fprintf(stderr,
                    "[bn:gpu:cuda] prefill score ptr upload failed: %s\n",
                    cudaGetErrorString(copy_err));
            return -1;
        }
        cublasStatus_t st = cublasSgemmBatched(
            ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            n_tokens, n_tokens, head_size,
            &alpha, d_a, kv_dim, d_b, q_ld,
            &zero, d_c, n_tokens, n_heads);
        if (st == CUBLAS_STATUS_SUCCESS) {
            int threads = 256;
            prefill_causal_softmax_kernel<<<dim3(n_heads, n_tokens, 1), threads,
                                            (size_t)threads * sizeof(float)>>>(
                d_scores, n_tokens);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] prefill softmax launch failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }

            for (int h = 0; h < n_heads; h++) {
                int kv_h = h / kv_mul;
                h_a[h] = (void *)(d_v + (size_t)kv_h * head_size);
                h_b[h] = (void *)(d_scores + (size_t)h * n_tokens * n_tokens);
                h_c[h] = (void *)(d_out + (size_t)h * head_size);
            }
            copy_err = cudaMemcpy(ctx->d_gemm_ptrs, ctx->h_gemm_ptrs,
                                  ptr_bytes, cudaMemcpyHostToDevice);
            if (copy_err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] prefill value ptr upload failed: %s\n",
                        cudaGetErrorString(copy_err));
                return -1;
            }
            st = cublasSgemmBatched(
                ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                head_size, n_tokens, n_tokens,
                &one, d_a, kv_dim, d_b, n_tokens,
                &zero, d_c, q_ld, n_heads);
            if (st == CUBLAS_STATUS_SUCCESS)
                return 0;
            fprintf(stderr,
                    "[bn:gpu:cuda] prefill batched value gemm failed: status %d\n",
                    (int)st);
            return -1;
        }
        if (getenv("BN_CUDA_DEBUG_PREFILL_GEMM"))
            fprintf(stderr,
                    "[bn:gpu:cuda] prefill batched score gemm unavailable: status %d\n",
                    (int)st);
    }

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / kv_mul;
        const float *q_h = d_q + (size_t)h * head_size;
        const float *k_h = d_k + (size_t)kv_h * head_size;
        float *scores_h = d_scores + (size_t)h * n_tokens * n_tokens;
        cublasStatus_t st = cublasSgemm(
            ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            n_tokens, n_tokens, head_size,
            &alpha, k_h, kv_dim, q_h, q_ld,
            &zero, scores_h, n_tokens);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[bn:gpu:cuda] prefill attention score gemm failed: status %d\n",
                    (int)st);
            return -1;
        }
    }

    int threads = 256;
    prefill_causal_softmax_kernel<<<dim3(n_heads, n_tokens, 1), threads,
                                    (size_t)threads * sizeof(float)>>>(
        d_scores, n_tokens);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill softmax launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / kv_mul;
        const float *v_h = d_v + (size_t)kv_h * head_size;
        const float *scores_h = d_scores + (size_t)h * n_tokens * n_tokens;
        float *out_h = d_out + (size_t)h * head_size;
        cublasStatus_t st = cublasSgemm(
            ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            head_size, n_tokens, n_tokens,
            &one, v_h, kv_dim, scores_h, n_tokens,
            &zero, out_h, q_ld);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[bn:gpu:cuda] prefill attention value gemm failed: status %d\n",
                    (int)st);
            return -1;
        }
    }

    return 0;
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

    int half_rope = rope_dims / 2;
    for (int i = tid; i < head_size; i += blockDim.x) {
        if (i < half_rope) {
            int j = i + half_rope;
            float angle = (float)pos * freq[i];
            float s, c;
            __sincosf(angle, &s, &c);
            float x0 = qh[i];
            float x1 = qh[j];
            if (bias) {
                const float *bh = bias + (size_t)h * head_size;
                x0 += bh[i];
                x1 += bh[j];
            }
            qrot[i] = x0 * c - x1 * s;
            qrot[j] = x0 * s + x1 * c;
        } else if (i >= rope_dims) {
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

    int half_rope = rope_dims / 2;
    for (int i = tid; i < head_size; i += blockDim.x) {
        if (i < half_rope) {
            int j = i + half_rope;
            float angle = (float)pos * freq[i];
            float s, c;
            __sincosf(angle, &s, &c);
            float x0 = qh[i];
            float x1 = qh[j];
            if (bias) {
                const float *bh = bias + (size_t)h * head_size;
                x0 += bh[i];
                x1 += bh[j];
            }
            qrot[i] = x0 * c - x1 * s;
            qrot[j] = x0 * s + x1 * c;
        } else if (i >= rope_dims) {
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

static __global__ void qk_norm_rope_flash_runtime_kernel(
    float *out, const float *q, void *key_cache, const void *value_cache,
    const float *freq, const float *q_weight, const float *k_weight,
    int n_heads, int n_kv_heads, int head_size, int kv_mul, int kv_dim,
    int seq_len, uint32_t loff, float inv_sqrt_hs, float eps,
    int per_head_weight, int rope_dims, int kv_f16,
    const BnCudaRuntimeParams *runtime) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= n_heads || head_size <= 0) return;
    int n_kv = runtime ? runtime->n_kv : 1;
    int pos = runtime ? runtime->pos : 0;
    int cache_pos = runtime ? runtime->cache_pos : 0;
    int kvh = h / kv_mul;
    if (kvh >= n_kv_heads) return;

    const float *qh = q + (size_t)h * head_size;
    size_t k_base = (size_t)loff + (size_t)cache_pos * kv_dim +
                    (size_t)kvh * head_size;

    extern __shared__ float shared[];
    float *scores = shared;
    float *scratch = scores + seq_len;
    float *qrot = scratch + blockDim.x;
    float *kcur = qrot + head_size;

    float ssq = 0.0f;
    float ssk = 0.0f;
    for (int i = tid; i < head_size; i += blockDim.x) {
        float qv = qh[i];
        float kv = cuda_kv_load(key_cache, k_base + (size_t)i, kv_f16);
        ssq += qv * qv;
        ssk += kv * kv;
    }
    float q_scale = rsqrtf(cuda_block_reduce_sum_all(ssq, scratch) /
                           (float)head_size + eps);
    float k_scale = rsqrtf(cuda_block_reduce_sum_all(ssk, scratch) /
                           (float)head_size + eps);

    const float *qw = q_weight +
        (per_head_weight ? (size_t)h * head_size : 0);
    const float *kw = k_weight +
        (per_head_weight ? (size_t)kvh * head_size : 0);
    int half_rope = rope_dims / 2;
    for (int i = tid; i < head_size; i += blockDim.x) {
        float qv = qh[i] * q_scale * qw[i];
        float kv = cuda_kv_load(key_cache, k_base + (size_t)i, kv_f16) *
                   k_scale * kw[i];
        qrot[i] = qv;
        kcur[i] = kv;
    }
    __syncthreads();
    for (int i = tid; i < half_rope; i += blockDim.x) {
        int j = i + half_rope;
        float angle = (float)pos * freq[i];
        float s, c;
        __sincosf(angle, &s, &c);
        float q0 = qrot[i], q1 = qrot[j];
        float k0 = kcur[i], k1 = kcur[j];
        qrot[i] = q0 * c - q1 * s;
        qrot[j] = q0 * s + q1 * c;
        kcur[i] = k0 * c - k1 * s;
        kcur[j] = k0 * s + k1 * c;
    }
    __syncthreads();

    for (int i = tid; i < head_size; i += blockDim.x)
        cuda_kv_store(key_cache, k_base + (size_t)i, kcur[i], kv_f16);
    __syncthreads();

    for (int t = tid; t < n_kv; t += blockDim.x) {
        size_t koff = (size_t)loff + (size_t)t * kv_dim +
                      (size_t)kvh * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            float kval = (t == cache_pos)
                ? kcur[i]
                : cuda_kv_load(key_cache, koff + (size_t)i, kv_f16);
            score += qrot[i] * kval;
        }
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

static __global__ void ffn_activation_batch_kernel(float *out,
                                                   const float *gate_up,
                                                   int hidden_dim,
                                                   int n_tokens,
                                                   int act_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_dim * n_tokens;
    if (i >= total) return;
    int token = i / hidden_dim;
    int h = i - token * hidden_dim;
    size_t base = (size_t)token * hidden_dim + h;
    float gate = gate_up[base];
    float up = gate_up[(size_t)n_tokens * hidden_dim + base];
    if (act_type == 0) {
        float silu = gate / (1.0f + __expf(-gate));
        out[base] = silu * up;
    } else {
        out[base] = gate * up;
    }
}

static __global__ void ffn_activation_batch_stacked_kernel(
    float *out, const float *gate_up, int hidden_dim, int n_tokens,
    int act_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_dim * n_tokens;
    if (i >= total) return;
    int token = i / hidden_dim;
    int h = i - token * hidden_dim;
    size_t src = (size_t)token * (size_t)hidden_dim * 2u + (size_t)h;
    float gate = gate_up[src];
    float up = gate_up[src + (size_t)hidden_dim];
    if (act_type == 0) {
        float silu = gate / (1.0f + __expf(-gate));
        out[(size_t)token * hidden_dim + h] = silu * up;
    } else {
        out[(size_t)token * hidden_dim + h] = gate * up;
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

static int cuda_ensure_q8_1(BnCudaCtx *ctx, int cols) {
    if (!ctx || cols <= 0) return -1;
    size_t n_blocks = ((size_t)cols + 31u) / 32u;
    size_t bytes = n_blocks * sizeof(BnCudaBlockQ8_1);
    if (bytes <= ctx->d_q8_1_bytes) return 0;
    if (ctx->d_q8_1) cudaFree(ctx->d_q8_1);
    ctx->d_q8_1 = NULL;
    ctx->d_q8_1_bytes = 0;
    cudaError_t err = cudaMalloc(&ctx->d_q8_1, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] q8_1 scratch alloc failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    ctx->d_q8_1_bytes = bytes;
    return 0;
}

static int cuda_ensure_q8_k(BnCudaCtx *ctx, int cols, int n_tokens) {
    if (!ctx || cols <= 0 || n_tokens <= 0) return -1;
    size_t n_blocks = (((size_t)cols + BN_QK_K - 1u) / BN_QK_K) *
                      (size_t)n_tokens;
    size_t bytes = n_blocks * sizeof(BnBlockQ8K);
    if (bytes <= ctx->d_q8_k_bytes) return 0;
    if (ctx->d_q8_k) cudaFree(ctx->d_q8_k);
    ctx->d_q8_k = NULL;
    ctx->d_q8_k_bytes = 0;
    cudaError_t err = cudaMalloc(&ctx->d_q8_k, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] q8_k scratch alloc failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    ctx->d_q8_k_bytes = bytes;
    return 0;
}

static int cuda_ensure_x_f16(BnCudaCtx *ctx, size_t values) {
    if (!ctx || values == 0) return -1;
    size_t bytes = values * sizeof(__half);
    if (bytes <= ctx->d_x_f16_bytes) return 0;
    if (ctx->d_x_f16) cudaFree(ctx->d_x_f16);
    ctx->d_x_f16 = NULL;
    ctx->d_x_f16_bytes = 0;
    cudaError_t err = cudaMalloc(&ctx->d_x_f16, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] f16 input scratch alloc failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    ctx->d_x_f16_bytes = bytes;
    return 0;
}

static int cuda_ensure_prefill(BnCudaCtx *ctx, size_t values) {
    if (!ctx || values == 0) return -1;
    size_t bytes = values * sizeof(float);
    if (bytes <= ctx->d_prefill_bytes) return 0;
    if (ctx->d_prefill) cudaFree(ctx->d_prefill);
    ctx->d_prefill = NULL;
    ctx->d_prefill_bytes = 0;
    cudaError_t err = cudaMalloc(&ctx->d_prefill, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill scratch alloc failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    ctx->d_prefill_bytes = bytes;
    return 0;
}

static int cuda_ensure_gemm_ptrs(BnCudaCtx *ctx, int n_ptrs) {
    if (!ctx || n_ptrs <= 0) return -1;
    size_t bytes = (size_t)n_ptrs * sizeof(void *);
    if (bytes > ctx->h_gemm_ptrs_bytes) {
        void **p = (void **)realloc(ctx->h_gemm_ptrs, bytes);
        if (!p) {
            fprintf(stderr, "[bn:gpu:cuda] host gemm ptr alloc failed\n");
            return -1;
        }
        ctx->h_gemm_ptrs = p;
        ctx->h_gemm_ptrs_bytes = bytes;
    }
    if (bytes > ctx->d_gemm_ptrs_bytes) {
        if (ctx->d_gemm_ptrs) cudaFree(ctx->d_gemm_ptrs);
        ctx->d_gemm_ptrs = NULL;
        ctx->d_gemm_ptrs_bytes = 0;
        cudaError_t err = cudaMalloc(&ctx->d_gemm_ptrs, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] device gemm ptr alloc failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        ctx->d_gemm_ptrs_bytes = bytes;
    }
    return 0;
}

static int cuda_ensure_argmax(BnCudaCtx *ctx, int n, int n_penalty_tokens) {
    if (!ctx || n <= 0) return -1;
    if (cuda_ctx_set_device(ctx) != 0) return -1;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t bytes = (size_t)blocks * sizeof(BnCudaArgmaxPair) + sizeof(int);
    if (bytes > ctx->d_argmax_bytes) {
        if (ctx->d_argmax) cudaFree(ctx->d_argmax);
        ctx->d_argmax = NULL;
        ctx->d_argmax_bytes = 0;
        cudaError_t err = cudaMalloc(&ctx->d_argmax, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] argmax scratch alloc failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        ctx->d_argmax_bytes = bytes;
    }
    if (n_penalty_tokens > 0) {
        size_t penalty_bytes = (size_t)n_penalty_tokens * sizeof(int);
        if (penalty_bytes > ctx->d_penalty_tokens_bytes) {
            if (ctx->d_penalty_tokens) cudaFree(ctx->d_penalty_tokens);
            ctx->d_penalty_tokens = NULL;
            ctx->d_penalty_tokens_bytes = 0;
            cudaError_t err = cudaMalloc(&ctx->d_penalty_tokens,
                                         penalty_bytes);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] penalty token scratch alloc failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
            ctx->d_penalty_tokens_bytes = penalty_bytes;
        }
    }
    return 0;
}

static int cuda_ensure_host_out(BnCudaCtx *ctx, size_t bytes) {
    if (!ctx) return -1;
    if (bytes <= ctx->h_out_bytes) return 0;
    float *next = NULL;
    cudaError_t err = cudaMallocHost((void **)&next, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] pinned host output alloc failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    if (ctx->h_out) cudaFreeHost(ctx->h_out);
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
    cudaError_t err = cudaSuccess;
    if (ctx->stream) {
        err = cudaMemcpyAsync((char *)ctx->act_bufs[buf_idx] + offset,
                              data, size, cudaMemcpyHostToDevice,
                              ctx->stream);
    } else {
        err = cudaMemcpy((char *)ctx->act_bufs[buf_idx] + offset,
                         data, size, cudaMemcpyHostToDevice);
    }
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
    cudaError_t err = cudaSuccess;
    if (ctx->stream) {
        err = cudaMemcpyAsync(out, (char *)ctx->act_bufs[buf_idx] + offset,
                              size, cudaMemcpyDeviceToHost, ctx->stream);
        if (err == cudaSuccess)
            err = cudaStreamSynchronize(ctx->stream);
    } else {
        err = cudaMemcpy(out, (char *)ctx->act_bufs[buf_idx] + offset,
                         size, cudaMemcpyDeviceToHost);
    }
    return err == cudaSuccess ? 0 : -1;
}

static int cuda_buffer_create_f16_cache(BnCudaBuffer *buf) {
    if (!buf || !buf->data || buf->rows <= 0 || buf->cols <= 0 ||
        (buf->cols & 31) != 0)
        return 0;
    if (buf->type != BN_GGUF_TENSOR_Q8_0 &&
        buf->type != BN_GGUF_TENSOR_Q5_0 &&
        buf->type != BN_GGUF_TENSOR_Q4_K &&
        buf->type != BN_GGUF_TENSOR_Q6_K)
        return 0;
    if (getenv("BN_CUDA_DISABLE_CUBLAS_MATMUL"))
        return 0;

    int q6_as_f16 = buf->type == BN_GGUF_TENSOR_Q6_K &&
                    getenv("BN_CUDA_DISABLE_Q6K_CUBLAS_F16") == NULL;
    size_t n = (size_t)buf->rows * (size_t)buf->cols;
    size_t bytes = n * (buf->type == BN_GGUF_TENSOR_Q6_K && !q6_as_f16
                        ? sizeof(float)
                        : sizeof(__half));
    int max_mb = cuda_env_int("BN_CUDA_CUBLAS_CACHE_MAX_MB", 128);
    if (max_mb > 0 && bytes > (size_t)max_mb * 1024u * 1024u)
        return 0;
    void **cache_ptr = buf->type == BN_GGUF_TENSOR_Q6_K && !q6_as_f16
        ? &buf->f32_data
        : &buf->f16_data;
    cudaError_t err = cudaMalloc(cache_ptr, bytes);
    if (err != cudaSuccess) {
        if (getenv("BN_CUDA_DEBUG_CUBLAS_CACHE"))
            fprintf(stderr,
                    "[bn:gpu:cuda] cublas cache alloc skipped type=%d rows=%d cols=%d bytes=%zu: %s\n",
                    buf->type, buf->rows, buf->cols, bytes,
                    cudaGetErrorString(err));
        *cache_ptr = NULL;
        return 0;
    }
    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1u) / (size_t)threads);
    if (buf->type == BN_GGUF_TENSOR_Q8_0) {
        dequant_q8_0_to_f16_kernel<<<blocks, threads>>>(
            (__half *)buf->f16_data, (const BnBlockQ8_0 *)buf->data,
            buf->rows, buf->cols);
    } else if (buf->type == BN_GGUF_TENSOR_Q5_0) {
        dequant_q5_0_to_f16_kernel<<<blocks, threads>>>(
            (__half *)buf->f16_data, (const BnBlockQ5_0 *)buf->data,
            buf->rows, buf->cols);
    } else if (buf->type == BN_GGUF_TENSOR_Q4_K) {
        dequant_q4k_to_f16_kernel<<<blocks, threads>>>(
            (__half *)buf->f16_data, (const BnBlockQ4K *)buf->data,
            buf->rows, buf->cols);
    } else if (buf->type == BN_GGUF_TENSOR_Q6_K && q6_as_f16) {
        dequant_q6k_to_f16_kernel<<<blocks, threads>>>(
            (__half *)buf->f16_data, (const BnBlockQ6K *)buf->data,
            buf->rows, buf->cols);
    } else {
        dequant_q6k_to_f32_kernel<<<blocks, threads>>>(
            (float *)buf->f32_data, (const BnBlockQ6K *)buf->data,
            buf->rows, buf->cols);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (getenv("BN_CUDA_DEBUG_CUBLAS_CACHE"))
            fprintf(stderr,
                    "[bn:gpu:cuda] cublas cache dequant skipped type=%d rows=%d cols=%d: %s\n",
                    buf->type, buf->rows, buf->cols,
                    cudaGetErrorString(err));
        cudaFree(*cache_ptr);
        *cache_ptr = NULL;
        return 0;
    }
    if (getenv("BN_CUDA_DEBUG_CUBLAS_CACHE"))
        fprintf(stderr,
                "[bn:gpu:cuda] cublas cache ready type=%d rows=%d cols=%d bytes=%zu precision=%s\n",
                buf->type, buf->rows, buf->cols, bytes,
                buf->type == BN_GGUF_TENSOR_Q6_K && !q6_as_f16 ? "f32" : "f16");
    if (buf->type == BN_GGUF_TENSOR_Q6_K && !q6_as_f16)
        buf->f32_size = bytes;
    else
        buf->f16_size = bytes;
    return 0;
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
    cuda_buffer_create_f16_cache(buf);
    return buf;
}

static void cuda_buffer_destroy(void *vctx, void *buffer) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    if (cuda_ctx_set_device(ctx) != 0) return;
    BnCudaBuffer *buf = (BnCudaBuffer *)buffer;
    if (!buf) return;
    if (buf->f16_data) cudaFree(buf->f16_data);
    if (buf->f32_data) cudaFree(buf->f32_data);
    if (buf->data) cudaFree(buf->data);
    free(buf);
}

static int cuda_cublas_matmul_f16_preconverted(BnCudaCtx *ctx, float *d_out,
                                               const BnCudaBuffer *w,
                                               const void *d_x_f16,
                                               int rows, int cols,
                                               int n_tokens);
static int cuda_matmul_device_out(BnCudaCtx *ctx, float *d_dst,
                                  const BnCudaBuffer *w, const float *d_x,
                                  int rows, int cols, int n_tokens,
                                  int type);

static int cuda_cublas_matmul_f16(BnCudaCtx *ctx, float *d_out,
                                  const BnCudaBuffer *w,
                                  const float *d_x,
                                  int rows, int cols, int n_tokens) {
    if (!ctx || !ctx->cublas || !d_out || !w || !d_x ||
        rows <= 0 || cols <= 0 || n_tokens <= 1)
        return -1;
    if (w->f32_data) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(
            ctx->cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rows, n_tokens, cols,
            &alpha,
            (const float *)w->f32_data, cols,
            d_x, cols,
            &beta,
            d_out, rows);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[bn:gpu:cuda] cublas sgemm failed: status %d\n",
                    (int)st);
            return -1;
        }
        return 0;
    }
    if (!w->f16_data)
        return -1;
    if (cuda_ensure_x_f16(ctx, (size_t)n_tokens * (size_t)cols) != 0)
        return -1;

    size_t x_values = (size_t)n_tokens * (size_t)cols;
    int threads = 256;
    int blocks = (int)((x_values + (size_t)threads - 1u) / (size_t)threads);
    f32_to_f16_kernel<<<blocks, threads>>>(
        (__half *)ctx->d_x_f16, d_x, x_values);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] cublas input convert failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return cuda_cublas_matmul_f16_preconverted(
        ctx, d_out, w, ctx->d_x_f16, rows, cols, n_tokens);
}

static int cuda_convert_f32_to_f16(BnCudaCtx *ctx, const float *d_x,
                                   size_t n_values) {
    if (!ctx || !d_x || n_values == 0)
        return -1;
    if (cuda_ensure_x_f16(ctx, n_values) != 0)
        return -1;
    int threads = 256;
    int blocks = (int)((n_values + (size_t)threads - 1u) /
                       (size_t)threads);
    f32_to_f16_kernel<<<blocks, threads>>>(
        (__half *)ctx->d_x_f16, d_x, n_values);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] f16 input convert failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static int cuda_cublas_matmul_f16_preconverted(BnCudaCtx *ctx, float *d_out,
                                               const BnCudaBuffer *w,
                                               const void *d_x_f16,
                                               int rows, int cols,
                                               int n_tokens) {
    if (!ctx || !ctx->cublas || !d_out || !w || !w->f16_data || !d_x_f16 ||
        rows <= 0 || cols <= 0 || n_tokens <= 1)
        return -1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        ctx->cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        rows, n_tokens, cols,
        &alpha,
        w->f16_data, CUDA_R_16F, cols,
        d_x_f16, CUDA_R_16F, cols,
        &beta,
        d_out, CUDA_R_32F, rows,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[bn:gpu:cuda] cublas matmul failed: status %d\n",
                (int)st);
        return -1;
    }
    return 0;
}

static int cuda_matmul_device_out_preconverted_f16(
        BnCudaCtx *ctx, float *d_dst, const BnCudaBuffer *w,
        const float *d_x, const void *d_x_f16,
        int rows, int cols, int n_tokens, int type) {
    if (w && w->f16_data && d_x_f16 &&
        cuda_cublas_matmul_f16_preconverted(ctx, d_dst, w, d_x_f16,
                                            rows, cols, n_tokens) == 0)
        return 0;
    return cuda_matmul_device_out(ctx, d_dst, w, d_x, rows, cols,
                                  n_tokens, type);
}

static int cuda_matmul_device_out(BnCudaCtx *ctx, float *d_dst,
                                  const BnCudaBuffer *w, const float *d_x,
                                  int rows, int cols, int n_tokens,
                                  int type) {
    if (!ctx || !d_dst || !w || !w->data || !d_x || rows <= 0 ||
        cols <= 0 || n_tokens <= 0 || !cuda_type_supported(type))
        return -1;

    int threads = 256;
    int warps = threads / 32;
    cudaError_t err = cudaSuccess;
    if ((w->f16_data || w->f32_data) && n_tokens > 1 &&
        cuda_cublas_matmul_f16(ctx, d_dst, w, d_x, rows, cols,
                               n_tokens) == 0) {
        err = cudaSuccess;
    } else if (type == BN_GGUF_TENSOR_Q4_K && (cols % BN_QK_K) == 0) {
        int x_blocks = (cols + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, d_x, cols, n_tokens);
        dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
        q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
            d_dst, (const BnBlockQ4K *)w->data, xq, rows, cols,
            n_tokens, 0);
    } else if (type == BN_GGUF_TENSOR_Q6_K && (cols % BN_QK_K) == 0 &&
               !getenv("BN_CUDA_DISABLE_Q6K_DOT")) {
        int x_blocks = cols / BN_QK_K;
        if (cuda_ensure_q8_k(ctx, cols, n_tokens) != 0) return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(x_blocks, n_tokens, 1),
                                    BN_QK_K>>>(xq, d_x, cols, n_tokens);
        dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
        q6k_dot_matmul_kernel<<<grid, threads, 0>>>(
            d_dst, (const BnBlockQ6K *)w->data, xq, rows, cols,
            n_tokens, 0);
    } else if (type == BN_GGUF_TENSOR_Q8_0 && (cols & 31) == 0) {
        dim3 grid(((rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q8_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            d_dst, (const BnBlockQ8_0 *)w->data, d_x, rows, cols,
            n_tokens, 0);
    } else if (type == BN_GGUF_TENSOR_Q5_0 && (cols & 31) == 0) {
        dim3 grid(((rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            d_dst, (const BnBlockQ5_0 *)w->data, d_x, rows, cols,
            n_tokens, 0);
    } else {
        dim3 grid(rows, n_tokens, 1);
        matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
            d_dst, w->data, d_x, NULL, rows, cols, type, 0);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] device matmul failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
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
    } else if (type == BN_GGUF_TENSOR_Q4_K && (cols % BN_QK_K) == 0 &&
               getenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT")) {
        if (cuda_ensure_q8_k(ctx, cols, 1) != 0) return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(cols / BN_QK_K, 1, 1),
                                    BN_QK_K>>>(xq, ctx->d_x, cols, 1);
        int warps = threads / 32;
        int blocks = (rows + warps - 1) / warps;
        q4k_q8k_dot_matvec_kernel<<<blocks, threads>>>(
            ctx->d_out, (const BnBlockQ4K *)w->data, xq, NULL,
            rows, cols, 0);
    } else if (type == BN_GGUF_TENSOR_Q6_K && (cols % BN_QK_K) == 0 &&
               getenv("BN_CUDA_ENABLE_Q6K_DOT")) {
        if (cuda_ensure_q8_k(ctx, cols, 1) != 0) return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(cols / BN_QK_K, 1, 1),
                                    BN_QK_K>>>(xq, ctx->d_x, cols, 1);
        int warps = threads / 32;
        int blocks = (rows + warps - 1) / warps;
        q6k_dot_matvec_kernel<<<blocks, threads>>>(
            ctx->d_out, (const BnBlockQ6K *)w->data, xq, NULL,
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
        if (cuda_ensure_host_out(ctx, out_bytes) != 0)
            return -1;
        err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] matvec output readback failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        memcpy(out, ctx->h_out, out_bytes);
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
    int warps = threads / 32;
    if ((w->f16_data || w->f32_data) && n_tokens > 1 &&
        cuda_cublas_matmul_f16(ctx, ctx->d_out, w, ctx->d_x, rows, cols,
                               n_tokens) == 0) {
        err = cudaSuccess;
    } else if (type == BN_GGUF_TENSOR_Q4_K && (cols % BN_QK_K) == 0) {
        int x_blocks = (cols + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, ctx->d_x, cols, n_tokens);
        dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
        q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ4K *)w->data, xq, rows, cols,
            n_tokens, 0);
    } else if (type == BN_GGUF_TENSOR_Q6_K && (cols % BN_QK_K) == 0 &&
               getenv("BN_CUDA_ENABLE_Q6K_DOT")) {
        int x_blocks = cols / BN_QK_K;
        if (cuda_ensure_q8_k(ctx, cols, n_tokens) != 0) return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(x_blocks, n_tokens, 1),
                                    BN_QK_K>>>(xq, ctx->d_x, cols,
                                               n_tokens);
        dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
        q6k_dot_matmul_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ6K *)w->data, xq, rows, cols,
            n_tokens, 0);
    } else if (type == BN_GGUF_TENSOR_Q8_0 && (cols & 31) == 0) {
        dim3 grid(((rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q8_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ8_0 *)w->data, ctx->d_x, rows, cols,
            n_tokens, 0);
    } else if (type == BN_GGUF_TENSOR_Q5_0 && (cols & 31) == 0) {
        dim3 grid(((rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ5_0 *)w->data, ctx->d_x, rows, cols,
            n_tokens, 0);
    } else {
        dim3 grid(rows, n_tokens, 1);
        matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
            ctx->d_out, w->data, ctx->d_x, NULL, rows, cols, type, 0);
    }
    if (err == cudaSuccess)
        err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] matmul launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    if (err == cudaSuccess) {
        if (cuda_ensure_host_out(ctx, out_bytes) != 0)
            return -1;
        err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] matmul output readback failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        memcpy(out, ctx->h_out, out_bytes);
    }
    return 0;
}

static int cuda_matmul_batch(void *vctx, const BnGPUMatvecOp *ops, int n_ops,
                             const float *X, int n_tokens, int x_cols) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    if (getenv("BN_CUDA_DISABLE_MATMUL_BATCH")) return -1;
    if (!ctx || !ops || !X || n_ops <= 1 || n_ops > 16 ||
        n_tokens <= 1 || x_cols <= 0)
        return -1;

    size_t total_values = 0;
    for (int i = 0; i < n_ops; i++) {
        BnCudaBuffer *w = (BnCudaBuffer *)ops[i].W_buf;
        if (!ops[i].out || !w || !w->data || ops[i].cols != x_cols ||
            ops[i].rows <= 0 || !cuda_type_supported(ops[i].type))
            return -1;
        total_values += (size_t)n_tokens * ops[i].rows;
    }

    size_t x_bytes = (size_t)n_tokens * x_cols * sizeof(float);
    size_t out_bytes = total_values * sizeof(float);
    if (cuda_ensure_scratch(ctx, x_bytes, out_bytes) != 0) return -1;

    cudaError_t err = cudaMemcpy(ctx->d_x, X, x_bytes,
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] matmul batch input upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int threads = 256;
    int warps = threads / 32;
    int x_f16_ready = 0;
    for (int i = 0; i < n_ops; i++) {
        BnCudaBuffer *w = (BnCudaBuffer *)ops[i].W_buf;
        if (w && w->f16_data) {
            if (cuda_ensure_x_f16(ctx,
                    (size_t)n_tokens * (size_t)x_cols) != 0)
                return -1;
            size_t x_values = (size_t)n_tokens * (size_t)x_cols;
            int cvt_threads = 256;
            int cvt_blocks = (int)((x_values + (size_t)cvt_threads - 1u) /
                                   (size_t)cvt_threads);
            f32_to_f16_kernel<<<cvt_blocks, cvt_threads>>>(
                (__half *)ctx->d_x_f16, ctx->d_x, x_values);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] matmul batch input convert failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
            x_f16_ready = 1;
            break;
        }
    }
    int q8_1_ready = 0;
    int q8_k_ready = 0;
    size_t out_offset = 0;
    for (int i = 0; i < n_ops; i++) {
        BnCudaBuffer *w = (BnCudaBuffer *)ops[i].W_buf;
        int rows = ops[i].rows;
        int type = ops[i].type;
        if (w->f16_data && x_f16_ready &&
            cuda_cublas_matmul_f16_preconverted(
                ctx, ctx->d_out + out_offset, w, ctx->d_x_f16, rows,
                x_cols, n_tokens) == 0) {
            err = cudaSuccess;
        } else if ((w->f16_data || w->f32_data) &&
            cuda_cublas_matmul_f16(ctx, ctx->d_out + out_offset, w,
                                   ctx->d_x, rows, x_cols,
                                   n_tokens) == 0) {
            err = cudaSuccess;
        } else if (type == BN_GGUF_TENSOR_Q8_0 && (x_cols & 31) == 0) {
            dim3 grid(((rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
            q8_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ8_0 *)w->data, ctx->d_x,
                rows, x_cols, n_tokens, out_offset);
        } else if (type == BN_GGUF_TENSOR_Q5_0 && (x_cols & 31) == 0) {
            dim3 grid(((rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
            q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5_0 *)w->data, ctx->d_x,
                rows, x_cols, n_tokens, out_offset);
        } else if (type == BN_GGUF_TENSOR_Q4_K &&
                   (x_cols % BN_QK_K) == 0) {
            int x_blocks = (x_cols + 31) / 32;
            if (!q8_1_ready) {
                if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
                    return -1;
                BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                dim3 qgrid(x_blocks, n_tokens, 1);
                quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
                    xq, ctx->d_x, x_cols, n_tokens);
                q8_1_ready = 1;
            }
            dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
            q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ4K *)w->data,
                (const BnCudaBlockQ8_1 *)ctx->d_q8_1, rows, x_cols,
                n_tokens, out_offset);
        } else if (type == BN_GGUF_TENSOR_Q6_K &&
                   (x_cols % BN_QK_K) == 0 &&
                   !getenv("BN_CUDA_DISABLE_Q6K_DOT")) {
            int x_blocks = x_cols / BN_QK_K;
            if (!q8_k_ready) {
                if (cuda_ensure_q8_k(ctx, x_cols, n_tokens) != 0)
                    return -1;
                BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                quantize_q8k_batch_kernel<<<dim3(x_blocks, n_tokens, 1),
                                            BN_QK_K>>>(
                    xq, ctx->d_x, x_cols, n_tokens);
                q8_k_ready = 1;
            }
            dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
            q6k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ6K *)w->data,
                (const BnBlockQ8K *)ctx->d_q8_k, rows, x_cols,
                n_tokens, out_offset);
        } else {
            dim3 grid(rows, n_tokens, 1);
            matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
                ctx->d_out, w->data, ctx->d_x, NULL, rows, x_cols, type,
                out_offset);
        }
        if (err == cudaSuccess)
            err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] matmul batch launch failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        out_offset += (size_t)n_tokens * rows;
    }

    if (cuda_ensure_host_out(ctx, out_bytes) != 0) return -1;
    err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] matmul batch output readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    out_offset = 0;
    for (int i = 0; i < n_ops; i++) {
        size_t values = (size_t)n_tokens * ops[i].rows;
        memcpy(ops[i].out, ctx->h_out + out_offset,
               values * sizeof(float));
        out_offset += values;
    }
    return 0;
}

static int cuda_argmax_activation(void *vctx, int buf_idx, int n,
                                  const int *penalty_tokens,
                                  int n_penalty_tokens,
                                  float repeat_penalty, int *out_token) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    if (!ctx || !out_token || buf_idx < 0 || buf_idx >= BN_GPU_VALUE_COUNT ||
        n <= 0)
        return -1;
    float *x = cuda_act(ctx, buf_idx);
    if (!x) return -1;
    if (n_penalty_tokens < 0) n_penalty_tokens = 0;
    if (repeat_penalty == 1.0f) n_penalty_tokens = 0;
    if (n_penalty_tokens > 0 && !penalty_tokens) return -1;
    if (cuda_ensure_argmax(ctx, n, n_penalty_tokens) != 0) return -1;
    cudaStream_t stream = ctx->exec_stream ? ctx->exec_stream : ctx->stream;
    if (n_penalty_tokens > 0) {
        cudaError_t copy_err = cudaMemcpyAsync(
            ctx->d_penalty_tokens, penalty_tokens,
            (size_t)n_penalty_tokens * sizeof(int),
            cudaMemcpyHostToDevice, stream);
        if (copy_err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] penalty token copy failed: %s\n",
                    cudaGetErrorString(copy_err));
            return -1;
        }
    }
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    BnCudaArgmaxPair *partials = (BnCudaArgmaxPair *)ctx->d_argmax;
    int *d_result = (int *)(partials + blocks);
    argmax_penalty_stage1_kernel<<<blocks, threads,
        (size_t)threads * sizeof(BnCudaArgmaxPair), stream>>>(
        partials, x, n,
        n_penalty_tokens > 0 ? ctx->d_penalty_tokens : NULL,
        n_penalty_tokens, repeat_penalty);
    argmax_stage2_kernel<<<1, threads,
        (size_t)threads * sizeof(BnCudaArgmaxPair), stream>>>(
        d_result, partials, blocks);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] argmax launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaMemcpyAsync(out_token, d_result, sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess)
        err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] argmax readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
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
    if (cuda_ensure_host_out(ctx, out_bytes) != 0)
        return -1;
    err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn output readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    memcpy(out, ctx->h_out, out_bytes);
    return 0;
}

static int cuda_dense_ffn_batch_impl(void *vctx, float *out,
                                     void *gate_buf, void *up_buf,
                                     void *down_buf, void *norm_buf,
                                     const float *X,
                                     int n_tokens, int dim, int hidden_dim,
                                     int gate_type, int up_type,
                                     int down_type, int act_type,
                                     float norm_eps,
                                     int add_residual) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *gate = (BnCudaBuffer *)gate_buf;
    BnCudaBuffer *up = (BnCudaBuffer *)up_buf;
    BnCudaBuffer *down = (BnCudaBuffer *)down_buf;
    BnCudaBuffer *norm = (BnCudaBuffer *)norm_buf;
    int stacked_gateup = up == NULL;
    if (getenv("BN_CUDA_DISABLE_DENSE_FFN_BATCH"))
        return -1;
    if (!ctx || !out || !gate || !down || !X ||
        !gate->data || !down->data ||
        n_tokens <= 1 || dim <= 0 || hidden_dim <= 0 || act_type != 0)
        return -1;
    if ((!stacked_gateup &&
         (up == NULL || !up->data || up->rows != hidden_dim ||
          up->cols != dim || gate->rows != hidden_dim)) ||
        (stacked_gateup &&
         (gate->rows != hidden_dim * 2 || gate->cols != dim ||
          gate_type != up_type)) ||
        gate->cols != dim ||
        down->rows != dim || down->cols != hidden_dim)
        return -1;
    if (norm && (!norm->data || norm->rows * norm->cols < dim))
        return -1;
    if (add_residual && !norm)
        return -1;
    if (!cuda_type_supported(gate_type) ||
        (!stacked_gateup && !cuda_type_supported(up_type)) ||
        !cuda_type_supported(down_type))
        return -1;

    size_t input_bytes = (size_t)n_tokens * dim * sizeof(float);
    size_t hidden_bytes = (size_t)n_tokens * hidden_dim * sizeof(float);
    size_t gateup_bytes = hidden_bytes * 2;
    size_t out_bytes = (size_t)n_tokens * dim * sizeof(float);
    size_t scratch_x = input_bytes > hidden_bytes ? input_bytes : hidden_bytes;
    size_t scratch_out = gateup_bytes > out_bytes ? gateup_bytes : out_bytes;
    if (cuda_ensure_scratch(ctx, scratch_x, scratch_out) != 0) return -1;

    cudaError_t err = cudaMemcpy(ctx->d_x, X, input_bytes,
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn batch input upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int threads = 256;
    int warps = threads / 32;
    const float *gate_input = ctx->d_x;
    float *residual_input = ctx->d_x;
    if (norm) {
        size_t residual_values = add_residual
            ? (size_t)n_tokens * (size_t)dim : 0u;
        size_t norm_values = (size_t)n_tokens * (size_t)dim;
        if (cuda_ensure_prefill(ctx, residual_values + norm_values) != 0)
            return -1;
        float *d_residual = ctx->d_prefill;
        float *d_norm = d_residual + residual_values;
        if (add_residual) {
            err = cudaMemcpy(d_residual, ctx->d_x, input_bytes,
                             cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] dense ffn residual copy failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
            residual_input = d_residual;
        }
        rmsnorm_batch_kernel<<<n_tokens, threads,
                               (size_t)warps * sizeof(float)>>>(
            d_norm, ctx->d_x, (const float *)norm->data, dim,
            n_tokens, norm_eps);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] dense ffn batch norm failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        gate_input = d_norm;
    }
    if (stacked_gateup && (gate->f16_data || gate->f32_data) &&
        cuda_cublas_matmul_f16(ctx, ctx->d_out, gate, gate_input,
                               hidden_dim * 2, dim, n_tokens) == 0) {
        err = cudaSuccess;
    } else if (stacked_gateup && gate_type == BN_GGUF_TENSOR_Q4_K &&
               (dim % BN_QK_K) == 0) {
        int x_blocks = (dim + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, gate_input, dim, n_tokens);
        dim3 grid((hidden_dim * 2 + warps - 1) / warps,
                  n_tokens, 1);
        q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ4K *)gate->data, xq,
            hidden_dim * 2, dim, n_tokens, 0);
    } else if (stacked_gateup && gate_type == BN_GGUF_TENSOR_Q5_0 &&
               (dim & 31) == 0) {
        dim3 grid((((hidden_dim * 2) + 3) / 4 + warps - 1) / warps,
                  n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ5_0 *)gate->data, gate_input,
            hidden_dim * 2, dim, n_tokens, 0);
    } else if (stacked_gateup && gate_type == BN_GGUF_TENSOR_Q8_0 &&
               (dim & 31) == 0) {
        dim3 grid((((hidden_dim * 2) + 3) / 4 + warps - 1) / warps,
                  n_tokens, 1);
        q8_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ8_0 *)gate->data, gate_input,
            hidden_dim * 2, dim, n_tokens, 0);
    } else if (!stacked_gateup &&
        (gate->f16_data || gate->f32_data) &&
        (up->f16_data || up->f32_data) &&
        cuda_cublas_matmul_f16(ctx, ctx->d_out, gate, gate_input,
                               hidden_dim, dim, n_tokens) == 0 &&
        cuda_cublas_matmul_f16(ctx, ctx->d_out + (size_t)n_tokens * hidden_dim,
                               up, gate_input, hidden_dim, dim,
                               n_tokens) == 0) {
        err = cudaSuccess;
    } else if (!stacked_gateup && gate_type == BN_GGUF_TENSOR_Q5_0 &&
        up_type == BN_GGUF_TENSOR_Q5_0 &&
        (dim & 31) == 0) {
        dim3 grid(((hidden_dim + 3) / 4 + warps - 1) / warps,
                  n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ5_0 *)gate->data, gate_input,
            hidden_dim, dim, n_tokens, 0);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ5_0 *)up->data, gate_input,
            hidden_dim, dim, n_tokens, (size_t)n_tokens * hidden_dim);
    } else if (!stacked_gateup && gate_type == BN_GGUF_TENSOR_Q8_0 &&
               up_type == BN_GGUF_TENSOR_Q8_0 && (dim & 31) == 0) {
        dim3 grid(((hidden_dim + 3) / 4 + warps - 1) / warps,
                  n_tokens, 1);
        q8_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ8_0 *)gate->data, gate_input,
            hidden_dim, dim, n_tokens, 0);
        q8_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ8_0 *)up->data, gate_input,
            hidden_dim, dim, n_tokens, (size_t)n_tokens * hidden_dim);
    } else if (!stacked_gateup && gate_type == BN_GGUF_TENSOR_Q4_K &&
               up_type == BN_GGUF_TENSOR_Q4_K && (dim % BN_QK_K) == 0) {
        int x_blocks = (dim + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, gate_input, dim, n_tokens);
        dim3 grid((hidden_dim + warps - 1) / warps, n_tokens, 1);
        q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ4K *)gate->data, xq, hidden_dim,
            dim, n_tokens, 0);
        q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ4K *)up->data, xq, hidden_dim,
            dim, n_tokens, (size_t)n_tokens * hidden_dim);
    } else {
        if (stacked_gateup)
            return -1;
        dim3 grid(hidden_dim, n_tokens, 1);
        matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
            ctx->d_out, gate->data, gate_input, NULL, hidden_dim, dim,
            gate_type, 0);
        matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
            ctx->d_out, up->data, gate_input, NULL, hidden_dim, dim,
            up_type, (size_t)n_tokens * hidden_dim);
    }
    if (err == cudaSuccess)
        err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn batch gate/up failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int act_total = n_tokens * hidden_dim;
    int act_blocks = (act_total + threads - 1) / threads;
    if (stacked_gateup) {
        ffn_activation_batch_stacked_kernel<<<act_blocks, threads>>>(
            ctx->d_x, ctx->d_out, hidden_dim, n_tokens, act_type);
    } else {
        ffn_activation_batch_kernel<<<act_blocks, threads>>>(
            ctx->d_x, ctx->d_out, hidden_dim, n_tokens, act_type);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn batch activation failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    if ((down->f16_data || down->f32_data) &&
        cuda_cublas_matmul_f16(ctx, ctx->d_out, down, ctx->d_x,
                               dim, hidden_dim, n_tokens) == 0) {
        err = cudaSuccess;
    } else if (down_type == BN_GGUF_TENSOR_Q8_0 && (hidden_dim & 31) == 0) {
        dim3 grid(((dim + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q8_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ8_0 *)down->data, ctx->d_x,
            dim, hidden_dim, n_tokens, 0);
    } else if (down_type == BN_GGUF_TENSOR_Q5_0 && (hidden_dim & 31) == 0) {
        dim3 grid(((dim + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ5_0 *)down->data, ctx->d_x,
            dim, hidden_dim, n_tokens, 0);
    } else if (down_type == BN_GGUF_TENSOR_Q6_K &&
               (hidden_dim % BN_QK_K) == 0 &&
               !getenv("BN_CUDA_DISABLE_Q6K_DOT")) {
        int x_blocks = hidden_dim / BN_QK_K;
        if (cuda_ensure_q8_k(ctx, hidden_dim, n_tokens) != 0)
            return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(x_blocks, n_tokens, 1),
                                    BN_QK_K>>>(
            xq, ctx->d_x, hidden_dim, n_tokens);
        dim3 grid((dim + warps - 1) / warps, n_tokens, 1);
        q6k_dot_matmul_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ6K *)down->data, xq,
            dim, hidden_dim, n_tokens, 0);
    } else if (down_type == BN_GGUF_TENSOR_Q6_K &&
               (hidden_dim % BN_QK_K) == 0 &&
               getenv("BN_CUDA_ENABLE_Q6K_BATCH_WARP")) {
        dim3 grid((dim + warps - 1) / warps, n_tokens, 1);
        q6k_matmul_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ6K *)down->data, ctx->d_x,
            dim, hidden_dim, n_tokens, 0);
    } else {
        dim3 grid(dim, n_tokens, 1);
        matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
            ctx->d_out, down->data, ctx->d_x, NULL, dim, hidden_dim,
            down_type, 0);
    }
    if (err == cudaSuccess)
        err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn batch down failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    if (add_residual) {
        int total = n_tokens * dim;
        int blocks = (total + threads - 1) / threads;
        residual_add_kernel<<<blocks, threads>>>(ctx->d_out, residual_input,
                                                 total);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] dense ffn residual failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (cuda_ensure_host_out(ctx, out_bytes) != 0)
        return -1;
    err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn batch output readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    memcpy(out, ctx->h_out, out_bytes);
    return 0;
}

static int cuda_dense_ffn_batch(void *vctx, float *out,
                                void *gate_buf, void *up_buf,
                                void *down_buf, const float *X,
                                int n_tokens, int dim, int hidden_dim,
                                int gate_type, int up_type, int down_type,
                                int act_type) {
    return cuda_dense_ffn_batch_impl(
        vctx, out, gate_buf, up_buf, down_buf, NULL, X, n_tokens, dim,
        hidden_dim, gate_type, up_type, down_type, act_type, 0.0f, 0);
}

static int cuda_dense_ffn_batch_norm(void *vctx, float *out,
                                     void *gate_buf, void *up_buf,
                                     void *down_buf, void *norm_buf,
                                     const float *X,
                                     int n_tokens, int dim, int hidden_dim,
                                     int gate_type, int up_type,
                                     int down_type, int act_type,
                                     float norm_eps) {
    return cuda_dense_ffn_batch_impl(
        vctx, out, gate_buf, up_buf, down_buf, norm_buf, X, n_tokens, dim,
        hidden_dim, gate_type, up_type, down_type, act_type, norm_eps, 0);
}

static int cuda_dense_ffn_batch_norm_resid(void *vctx, float *out,
                                           void *gate_buf, void *up_buf,
                                           void *down_buf, void *norm_buf,
                                           const float *X,
                                           int n_tokens, int dim,
                                           int hidden_dim,
                                           int gate_type, int up_type,
                                           int down_type, int act_type,
                                           float norm_eps) {
    return cuda_dense_ffn_batch_impl(
        vctx, out, gate_buf, up_buf, down_buf, norm_buf, X, n_tokens, dim,
        hidden_dim, gate_type, up_type, down_type, act_type, norm_eps, 1);
}

static int cuda_prefill_attention(void *vctx, float *out,
                                  const float *Q, const float *K,
                                  const float *V, int n_tokens,
                                  int n_heads, int n_kv_heads,
                                  int head_size, int kv_mul, int kv_dim,
                                  float attention_scale) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    if (getenv("BN_CUDA_DISABLE_PREFILL_ATTN"))
        return -1;
    if (!ctx || !out || !Q || !K || !V || n_tokens <= 1 ||
        n_heads <= 0 || n_kv_heads <= 0 || head_size <= 0 ||
        kv_mul <= 0 || kv_dim <= 0 || n_heads / kv_mul != n_kv_heads)
        return -1;
    int min_tokens = cuda_env_int("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", 64);
    if (n_tokens < min_tokens)
        return -1;
    if (n_tokens > 2048)
        return -1;

    size_t q_values = (size_t)n_tokens * (size_t)n_heads * (size_t)head_size;
    size_t kv_values = (size_t)n_tokens * (size_t)kv_dim;
    size_t total_values = q_values + 2u * kv_values;
    if (cuda_ensure_prefill(ctx, total_values) != 0)
        return -1;
    if (cuda_ensure_scratch(ctx, sizeof(float), q_values * sizeof(float)) != 0)
        return -1;

    float *d_q = ctx->d_prefill;
    float *d_k = d_q + q_values;
    float *d_v = d_k + kv_values;
    cudaError_t err = cudaMemcpy(d_q, Q, q_values * sizeof(float),
                                 cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(d_k, K, kv_values * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(d_v, V, kv_values * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill attention upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int threads = 256;
    size_t shared = (size_t)(n_tokens + threads) * sizeof(float);
    prefill_attention_kernel<<<dim3(n_heads, n_tokens, 1), threads,
                               shared>>>(
        ctx->d_out, d_q, d_k, d_v, n_tokens, n_heads, head_size,
        kv_mul, kv_dim, attention_scale);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill attention launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    size_t out_bytes = q_values * sizeof(float);
    if (cuda_ensure_host_out(ctx, out_bytes) != 0)
        return -1;
    err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill attention readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    memcpy(out, ctx->h_out, out_bytes);
    return 0;
}

static int cuda_prefill_attention_wo(void *vctx, float *out, void *wo_buf,
                                     const float *Q, const float *K,
                                     const float *V, int n_tokens,
                                     int n_heads, int n_kv_heads,
                                     int head_size, int kv_mul, int kv_dim,
                                     int wo_rows, int wo_cols, int wo_type,
                                     float attention_scale) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *wo = (BnCudaBuffer *)wo_buf;
    if (getenv("BN_CUDA_DISABLE_PREFILL_ATTN_WO"))
        return -1;
    if (!ctx || !out || !wo || !wo->data || !Q || !K || !V ||
        n_tokens <= 1 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_size <= 0 || kv_mul <= 0 || kv_dim <= 0 ||
        n_heads / kv_mul != n_kv_heads || wo_rows <= 0 ||
        wo_cols != n_heads * head_size || wo->rows != wo_rows ||
        wo->cols != wo_cols || !cuda_type_supported(wo_type))
        return -1;
    int min_tokens = cuda_env_int("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", 64);
    if (n_tokens < min_tokens || n_tokens > 2048)
        return -1;

    size_t q_values = (size_t)n_tokens * (size_t)n_heads *
                      (size_t)head_size;
    size_t kv_values = (size_t)n_tokens * (size_t)kv_dim;
    int gemm_min_tokens =
        cuda_env_int("BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS", 256);
    int use_gemm_attention =
        !getenv("BN_CUDA_DISABLE_PREFILL_GEMM_ATTN") &&
        (getenv("BN_CUDA_ENABLE_PREFILL_GEMM_ATTN") ||
         n_tokens >= gemm_min_tokens) &&
        n_tokens <= 512;
    size_t score_values = use_gemm_attention
        ? (size_t)n_heads * (size_t)n_tokens * (size_t)n_tokens
        : 0;
    size_t total_values = q_values + 2u * kv_values + score_values;
    size_t out_values = (size_t)n_tokens * (size_t)wo_rows;
    if (cuda_ensure_prefill(ctx, total_values) != 0)
        return -1;
    if (cuda_ensure_scratch(ctx, q_values * sizeof(float),
                            out_values * sizeof(float)) != 0)
        return -1;

    float *d_q = ctx->d_prefill;
    float *d_k = d_q + q_values;
    float *d_v = d_k + kv_values;
    float *d_scores = d_v + kv_values;
    cudaError_t err = cudaMemcpy(d_q, Q, q_values * sizeof(float),
                                 cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(d_k, K, kv_values * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(d_v, V, kv_values * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill attention+wo upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int threads = 256;
    int warps = threads / 32;
    if (use_gemm_attention) {
        if (cuda_prefill_attention_gemm(
                ctx, ctx->d_x, d_q, d_k, d_v, d_scores, n_tokens,
                n_heads, n_kv_heads, head_size, kv_mul, kv_dim,
                attention_scale) != 0)
            return -1;
    } else {
        size_t shared = (size_t)(n_tokens + threads) * sizeof(float);
        prefill_attention_kernel<<<dim3(n_heads, n_tokens, 1), threads,
                                   shared>>>(
            ctx->d_x, d_q, d_k, d_v, n_tokens, n_heads, head_size,
            kv_mul, kv_dim, attention_scale);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill attention+wo attention failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }

    if ((wo->f16_data || wo->f32_data) &&
        cuda_cublas_matmul_f16(ctx, ctx->d_out, wo, ctx->d_x, wo_rows,
                               wo_cols, n_tokens) == 0) {
        err = cudaSuccess;
    } else if (wo_type == BN_GGUF_TENSOR_Q4_K && (wo_cols % BN_QK_K) == 0) {
        int x_blocks = (wo_cols + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        quantize_q8_1_batch_kernel<<<dim3(x_blocks, n_tokens, 1), 32, 0>>>(
            xq, ctx->d_x, wo_cols, n_tokens);
        dim3 grid((wo_rows + warps - 1) / warps, n_tokens, 1);
        q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ4K *)wo->data, xq, wo_rows,
            wo_cols, n_tokens, 0);
    } else if (wo_type == BN_GGUF_TENSOR_Q6_K && (wo_cols % BN_QK_K) == 0 &&
               !getenv("BN_CUDA_DISABLE_Q6K_DOT")) {
        int x_blocks = wo_cols / BN_QK_K;
        if (cuda_ensure_q8_k(ctx, wo_cols, n_tokens) != 0)
            return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(x_blocks, n_tokens, 1), BN_QK_K>>>(
            xq, ctx->d_x, wo_cols, n_tokens);
        dim3 grid((wo_rows + warps - 1) / warps, n_tokens, 1);
        q6k_dot_matmul_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ6K *)wo->data, xq, wo_rows,
            wo_cols, n_tokens, 0);
    } else if (wo_type == BN_GGUF_TENSOR_Q5_0 && (wo_cols & 31) == 0) {
        dim3 grid(((wo_rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ5_0 *)wo->data, ctx->d_x, wo_rows,
            wo_cols, n_tokens, 0);
    } else if (wo_type == BN_GGUF_TENSOR_Q8_0 && (wo_cols & 31) == 0) {
        dim3 grid(((wo_rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q8_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ8_0 *)wo->data, ctx->d_x, wo_rows,
            wo_cols, n_tokens, 0);
    } else {
        return -1;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill attention+wo matmul failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    size_t out_bytes = out_values * sizeof(float);
    if (cuda_ensure_host_out(ctx, out_bytes) != 0)
        return -1;
    err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill attention+wo readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    memcpy(out, ctx->h_out, out_bytes);
    return 0;
}

static int cuda_prefill_qkv_attention_wo_impl(
        void *vctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
        void *attn_norm_buf, void *q_norm_buf, void *k_norm_buf,
        const float *X, float *K_out, float *V_out, int n_tokens, int dim,
        int n_heads, int n_kv_heads, int head_size, int kv_mul, int kv_dim,
        int qk_rows, int qk_type, int wv_rows, int wv_type, int wo_rows,
        int wo_cols, int wo_type, int qk_norm_per_head, float norm_eps,
        int pos0, int rope_dims,
        float attention_scale,
        int add_residual) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *qk = (BnCudaBuffer *)qk_buf;
    BnCudaBuffer *wv = (BnCudaBuffer *)wv_buf;
    BnCudaBuffer *wo = (BnCudaBuffer *)wo_buf;
    BnCudaBuffer *attn_norm = (BnCudaBuffer *)attn_norm_buf;
    BnCudaBuffer *q_norm = (BnCudaBuffer *)q_norm_buf;
    BnCudaBuffer *k_norm = (BnCudaBuffer *)k_norm_buf;
    if (getenv("BN_CUDA_DISABLE_PREFILL_QKV_ATTN_WO"))
        return -1;
    if (!ctx || !out || !qk || !qk->data || !wv || !wv->data ||
        !wo || !wo->data || !X || !K_out || !V_out || n_tokens <= 1 ||
        dim <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_size <= 0 ||
        kv_mul <= 0 || kv_dim <= 0 || n_heads / kv_mul != n_kv_heads ||
        qk_rows != n_heads * head_size + kv_dim || wv_rows != kv_dim ||
        wo_cols != n_heads * head_size || wo->rows != wo_rows ||
        wo->cols != wo_cols || !cuda_type_supported(qk_type) ||
        !cuda_type_supported(wv_type) || !cuda_type_supported(wo_type) ||
        pos0 != 0 || rope_dims <= 0 ||
        !ctx->act_bufs[BN_GPU_VALUE_ROPE_FREQ])
        return -1;
    if (attn_norm && (!attn_norm->data ||
                      attn_norm->rows * attn_norm->cols < dim))
        return -1;
    int min_tokens = getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS")
        ? cuda_env_int("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", 64)
        : (dim >= 2048 ? 16 : 64);
    if (n_tokens < min_tokens || n_tokens > 2048)
        return -1;

    int q_dim = n_heads * head_size;
    size_t q_values = (size_t)n_tokens * (size_t)q_dim;
    size_t kv_values = (size_t)n_tokens * (size_t)kv_dim;
    size_t qk_values = (size_t)n_tokens * (size_t)qk_rows;
    int use_gemm_attention =
        !getenv("BN_CUDA_DISABLE_PREFILL_GEMM_ATTN") &&
        (getenv("BN_CUDA_ENABLE_PREFILL_GEMM_ATTN") ||
         n_tokens >= cuda_env_int("BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS", 256)) &&
        n_tokens <= 512;
    if (!use_gemm_attention)
        return -1;
    size_t score_values =
        (size_t)n_heads * (size_t)n_tokens * (size_t)n_tokens;
    if (add_residual && !attn_norm)
        return -1;
    size_t residual_values = add_residual
        ? (size_t)n_tokens * (size_t)dim : 0u;
    size_t norm_values = attn_norm ? (size_t)n_tokens * (size_t)dim : 0u;
    size_t total_values = residual_values + norm_values + q_values +
                          2u * kv_values + qk_values + score_values;
    size_t out_values = (size_t)n_tokens * (size_t)wo_rows;
    if (cuda_ensure_scratch(ctx, (size_t)n_tokens * (size_t)dim * sizeof(float),
                            out_values * sizeof(float)) != 0)
        return -1;
    if (cuda_ensure_prefill(ctx, total_values) != 0)
        return -1;

    cudaError_t err = cudaMemcpy(ctx->d_x, X,
                                 (size_t)n_tokens * (size_t)dim * sizeof(float),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill qkv input upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    float *d_residual = ctx->d_prefill;
    float *d_norm = d_residual + residual_values;
    float *d_q = d_norm + norm_values;
    float *d_k = d_q + q_values;
    float *d_v = d_k + kv_values;
    float *d_qk = d_v + kv_values;
    float *d_scores = d_qk + qk_values;
    const float *matmul_x = ctx->d_x;
    if (attn_norm) {
        int threads = 256;
        int warps = threads / 32;
        if (add_residual) {
            cudaError_t copy_err = cudaMemcpy(
                d_residual, ctx->d_x,
                (size_t)n_tokens * (size_t)dim * sizeof(float),
                cudaMemcpyDeviceToDevice);
            if (copy_err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] prefill residual copy failed: %s\n",
                        cudaGetErrorString(copy_err));
                return -1;
            }
        }
        rmsnorm_batch_kernel<<<n_tokens, threads,
                               (size_t)warps * sizeof(float)>>>(
            d_norm, ctx->d_x, (const float *)attn_norm->data, dim,
            n_tokens, norm_eps);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill qkv input norm failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        matmul_x = d_norm;
    }

    if (cuda_matmul_device_out(ctx, d_qk, qk, matmul_x, qk_rows, dim,
                               n_tokens, qk_type) != 0 ||
        cuda_matmul_device_out(ctx, d_v, wv, matmul_x, wv_rows, dim,
                               n_tokens, wv_type) != 0)
        return -1;

    int threads = 256;
    int total_qk = n_tokens * (q_dim + kv_dim);
    int blocks = (total_qk + threads - 1) / threads;
    split_qk_prefill_kernel<<<blocks, threads>>>(
        d_qk, d_q, d_k, n_tokens, q_dim, kv_dim, qk_rows);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill qk split failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    const float *q_norm_data = q_norm ? (const float *)q_norm->data : NULL;
    const float *k_norm_data = k_norm ? (const float *)k_norm->data : NULL;
    qk_prefill_rmsnorm_rope_kernel<<<dim3(n_heads + n_kv_heads, n_tokens, 1),
                                     threads,
                                     (size_t)threads * sizeof(float)>>>(
        d_q, d_k, q_norm_data, k_norm_data,
        cuda_act(ctx, BN_GPU_VALUE_ROPE_FREQ), n_tokens, pos0,
        n_heads, n_kv_heads, head_size, norm_eps, qk_norm_per_head,
        rope_dims);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill qk norm/rope failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    if (cuda_prefill_attention_gemm(ctx, ctx->d_x, d_q, d_k, d_v, d_scores,
                                    n_tokens, n_heads, n_kv_heads,
                                    head_size, kv_mul, kv_dim,
                                    attention_scale) != 0)
        return -1;
    if (cuda_matmul_device_out(ctx, ctx->d_out, wo, ctx->d_x, wo_rows,
                               wo_cols, n_tokens, wo_type) != 0)
        return -1;
    if (add_residual) {
        int threads = 256;
        int total = n_tokens * dim;
        int blocks = (total + threads - 1) / threads;
        residual_add_kernel<<<blocks, threads>>>(ctx->d_out, d_residual,
                                                 total);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill residual failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }

    size_t kv_bytes = kv_values * sizeof(float);
    err = cudaMemcpy(K_out, d_k, kv_bytes, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess)
        err = cudaMemcpy(V_out, d_v, kv_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill kv readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    size_t out_bytes = out_values * sizeof(float);
    if (cuda_ensure_host_out(ctx, out_bytes) != 0)
        return -1;
    err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill qkv attention+wo readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    memcpy(out, ctx->h_out, out_bytes);
    return 0;
}

static int cuda_prefill_qkv_attention_wo(
        void *vctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
        void *q_norm_buf, void *k_norm_buf, const float *X, float *K_out,
        float *V_out, int n_tokens, int dim, int n_heads, int n_kv_heads,
        int head_size, int kv_mul, int kv_dim, int qk_rows, int qk_type,
        int wv_rows, int wv_type, int wo_rows, int wo_cols, int wo_type,
        int qk_norm_per_head, float norm_eps, int pos0, int rope_dims,
        float attention_scale) {
    return cuda_prefill_qkv_attention_wo_impl(
        vctx, out, qk_buf, wv_buf, wo_buf, NULL, q_norm_buf, k_norm_buf,
        X, K_out, V_out, n_tokens, dim, n_heads, n_kv_heads, head_size,
        kv_mul, kv_dim, qk_rows, qk_type, wv_rows, wv_type, wo_rows,
        wo_cols, wo_type, qk_norm_per_head, norm_eps, pos0, rope_dims,
        attention_scale, 0);
}

static int cuda_prefill_qkv_attention_wo_norm(
        void *vctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
        void *attn_norm_buf, void *q_norm_buf, void *k_norm_buf,
        const float *X, float *K_out, float *V_out, int n_tokens, int dim,
        int n_heads, int n_kv_heads, int head_size, int kv_mul, int kv_dim,
        int qk_rows, int qk_type, int wv_rows, int wv_type, int wo_rows,
        int wo_cols, int wo_type, int qk_norm_per_head, float norm_eps,
        int pos0, int rope_dims, float attention_scale) {
    return cuda_prefill_qkv_attention_wo_impl(
        vctx, out, qk_buf, wv_buf, wo_buf, attn_norm_buf, q_norm_buf,
        k_norm_buf, X, K_out, V_out, n_tokens, dim, n_heads, n_kv_heads,
        head_size, kv_mul, kv_dim, qk_rows, qk_type, wv_rows, wv_type,
        wo_rows, wo_cols, wo_type, qk_norm_per_head, norm_eps, pos0,
        rope_dims, attention_scale, 0);
}

static int cuda_prefill_qkv_attention_wo_norm_resid(
        void *vctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
        void *attn_norm_buf, void *q_norm_buf, void *k_norm_buf,
        const float *X, float *K_out, float *V_out, int n_tokens, int dim,
        int n_heads, int n_kv_heads, int head_size, int kv_mul, int kv_dim,
        int qk_rows, int qk_type, int wv_rows, int wv_type, int wo_rows,
        int wo_cols, int wo_type, int qk_norm_per_head, float norm_eps,
        int pos0, int rope_dims, float attention_scale) {
    return cuda_prefill_qkv_attention_wo_impl(
        vctx, out, qk_buf, wv_buf, wo_buf, attn_norm_buf, q_norm_buf,
        k_norm_buf, X, K_out, V_out, n_tokens, dim, n_heads, n_kv_heads,
        head_size, kv_mul, kv_dim, qk_rows, qk_type, wv_rows, wv_type,
        wo_rows, wo_cols, wo_type, qk_norm_per_head, norm_eps, pos0,
        rope_dims, attention_scale, 1);
}

static int cuda_prefill_dense_layer(
        void *vctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
        void *gate_buf, void *up_buf, void *down_buf, void *attn_norm_buf,
        void *ffn_norm_buf, void *q_norm_buf, void *k_norm_buf,
        const float *X, float *K_out, float *V_out, int n_tokens, int dim,
        int hidden_dim, int n_heads, int n_kv_heads, int head_size,
        int kv_mul, int kv_dim, int qk_rows, int qk_type, int wv_rows,
        int wv_type, int wo_rows, int wo_cols, int wo_type, int gate_type,
        int up_type, int down_type, int act_type, int qk_norm_per_head,
        float norm_eps, int pos0, int rope_dims, uint32_t kv_cache_off,
        int kv_cache_stride, float attention_scale) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *qk = (BnCudaBuffer *)qk_buf;
    BnCudaBuffer *wv = (BnCudaBuffer *)wv_buf;
    BnCudaBuffer *wo = (BnCudaBuffer *)wo_buf;
    BnCudaBuffer *gate = (BnCudaBuffer *)gate_buf;
    BnCudaBuffer *up = (BnCudaBuffer *)up_buf;
    BnCudaBuffer *down = (BnCudaBuffer *)down_buf;
    BnCudaBuffer *attn_norm = (BnCudaBuffer *)attn_norm_buf;
    BnCudaBuffer *ffn_norm = (BnCudaBuffer *)ffn_norm_buf;
    BnCudaBuffer *q_norm = (BnCudaBuffer *)q_norm_buf;
    BnCudaBuffer *k_norm = (BnCudaBuffer *)k_norm_buf;
    int stacked_gateup = up == NULL;
    if (getenv("BN_CUDA_DISABLE_PREFILL_DENSE_LAYER"))
        return -1;
    if (!ctx || !qk || !qk->data || !wv || !wv->data ||
        !wo || !wo->data || !gate || !gate->data || !down || !down->data ||
        !attn_norm || !attn_norm->data || !ffn_norm || !ffn_norm->data ||
        n_tokens <= 1 || dim <= 0 ||
        hidden_dim <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_size <= 0 || kv_mul <= 0 || kv_dim <= 0 ||
        kv_cache_stride < kv_dim ||
        n_heads / kv_mul != n_kv_heads || act_type != 0 ||
        qk_rows != n_heads * head_size + kv_dim || wv_rows != kv_dim ||
        wo_cols != n_heads * head_size || wo->rows != wo_rows ||
        wo->cols != wo_cols || down->rows != dim ||
        down->cols != hidden_dim || pos0 != 0 || rope_dims <= 0 ||
        !ctx->act_bufs[BN_GPU_VALUE_ROPE_FREQ])
        return -1;
    if ((!K_out || !V_out) &&
        (K_out || V_out ||
         !ctx->act_bufs[BN_GPU_VALUE_KEY_CACHE] ||
         !ctx->act_bufs[BN_GPU_VALUE_VALUE_CACHE]))
        return -1;
    if ((!stacked_gateup &&
         (!up || !up->data || gate->rows != hidden_dim ||
          up->rows != hidden_dim || gate->cols != dim ||
          up->cols != dim)) ||
        (stacked_gateup &&
         (gate->rows != hidden_dim * 2 || gate->cols != dim ||
          gate_type != up_type)))
        return -1;
    if (attn_norm->rows * attn_norm->cols < dim ||
        ffn_norm->rows * ffn_norm->cols < dim)
        return -1;
    if (!cuda_type_supported(qk_type) || !cuda_type_supported(wv_type) ||
        !cuda_type_supported(wo_type) || !cuda_type_supported(gate_type) ||
        (!stacked_gateup && !cuda_type_supported(up_type)) ||
        !cuda_type_supported(down_type))
        return -1;
    int min_tokens = getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS")
        ? cuda_env_int("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", 64)
        : (dim >= 2048 ? 16 : 64);
    if (n_tokens < min_tokens || n_tokens > 512)
        return -1;
    if (getenv("BN_CUDA_DISABLE_PREFILL_GEMM_ATTN"))
        return -1;

    const int dense_profile = getenv("BN_CUDA_PREFILL_DENSE_PROFILE") != NULL;
    static double dense_profile_totals[BN_CUDA_DENSE_PROF_MAX] = {0.0};
    static unsigned long long dense_profile_layers = 0;
    double dense_profile_t0 = dense_profile ? cuda_wall_ms() : 0.0;
#define BN_CUDA_DENSE_PROFILE_STEP(code_) do {                         \
        if (dense_profile) {                                           \
            cudaError_t profile_err__ = cudaDeviceSynchronize();       \
            double profile_now__ = cuda_wall_ms();                     \
            if (profile_err__ != cudaSuccess) {                        \
                fprintf(stderr,                                        \
                        "[bn:gpu:cuda] dense prefill profile sync failed: %s\n", \
                        cudaGetErrorString(profile_err__));            \
                return -1;                                             \
            }                                                          \
            cuda_dense_profile_add(dense_profile_totals, (code_),      \
                                   profile_now__ - dense_profile_t0);  \
            dense_profile_t0 = profile_now__;                          \
        }                                                              \
    } while (0)

    int q_dim = n_heads * head_size;
    size_t dim_values = (size_t)n_tokens * (size_t)dim;
    size_t q_values = (size_t)n_tokens * (size_t)q_dim;
    size_t kv_values = (size_t)n_tokens * (size_t)kv_dim;
    size_t qk_values = (size_t)n_tokens * (size_t)qk_rows;
    size_t score_values =
        (size_t)n_heads * (size_t)n_tokens * (size_t)n_tokens;
    size_t hidden_values = (size_t)n_tokens * (size_t)hidden_dim;
    size_t gateup_values = hidden_values * 2u;
    size_t total_values = dim_values + dim_values + q_values +
                          2u * kv_values + qk_values + score_values +
                          dim_values + dim_values + hidden_values +
                          gateup_values;
    size_t scratch_x_bytes = gateup_values * sizeof(float);
    size_t scratch_out_bytes = dim_values * sizeof(float);
    if (cuda_ensure_scratch(ctx, scratch_x_bytes, scratch_out_bytes) != 0)
        return -1;
    if (cuda_ensure_prefill(ctx, total_values) != 0)
        return -1;

    cudaError_t err = cudaSuccess;
    if (X) {
        err = cudaMemcpy(ctx->d_out, X, dim_values * sizeof(float),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill dense layer input upload failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }

    float *d_orig = ctx->d_prefill;
    float *d_attn_norm = d_orig + dim_values;
    float *d_q = d_attn_norm + dim_values;
    float *d_k = d_q + q_values;
    float *d_v = d_k + kv_values;
    float *d_qk = d_v + kv_values;
    float *d_scores = d_qk + qk_values;
    float *d_ffn_residual = d_scores + score_values;
    float *d_ffn_norm = d_ffn_residual + dim_values;
    float *d_ffn_act = d_ffn_norm + dim_values;
    float *d_gateup = d_ffn_act + hidden_values;

    err = cudaMemcpy(d_orig, ctx->d_out, dim_values * sizeof(float),
                     cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill dense layer residual copy failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int threads = 256;
    int warps = threads / 32;
    rmsnorm_batch_kernel<<<n_tokens, threads,
                           (size_t)warps * sizeof(float)>>>(
        d_attn_norm, ctx->d_out, (const float *)attn_norm->data, dim,
        n_tokens, norm_eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill dense layer attn norm failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_INPUT_NORM);
    const void *attn_norm_f16 = NULL;
    if (qk->f16_data && wv->f16_data && n_tokens > 1 &&
        cuda_convert_f32_to_f16(ctx, d_attn_norm, dim_values) == 0)
        attn_norm_f16 = ctx->d_x_f16;
    if (cuda_matmul_device_out_preconverted_f16(
            ctx, d_qk, qk, d_attn_norm, attn_norm_f16, qk_rows, dim,
            n_tokens, qk_type) != 0)
        return -1;
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_QK);
    if (cuda_matmul_device_out_preconverted_f16(
            ctx, d_v, wv, d_attn_norm, attn_norm_f16, wv_rows, dim,
            n_tokens, wv_type) != 0)
        return -1;
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_WV);

    int total_qk = n_tokens * (q_dim + kv_dim);
    int blocks = (total_qk + threads - 1) / threads;
    split_qk_prefill_kernel<<<blocks, threads>>>(
        d_qk, d_q, d_k, n_tokens, q_dim, kv_dim, qk_rows);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill dense layer qk split failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    const float *q_norm_data = q_norm ? (const float *)q_norm->data : NULL;
    const float *k_norm_data = k_norm ? (const float *)k_norm->data : NULL;
    qk_prefill_rmsnorm_rope_kernel<<<dim3(n_heads + n_kv_heads, n_tokens, 1),
                                     threads,
                                     (size_t)threads * sizeof(float)>>>(
        d_q, d_k, q_norm_data, k_norm_data,
        cuda_act(ctx, BN_GPU_VALUE_ROPE_FREQ), n_tokens, pos0,
        n_heads, n_kv_heads, head_size, norm_eps, qk_norm_per_head,
        rope_dims);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill dense layer qk norm/rope failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_QK_ROPE);

    if (cuda_prefill_attention_gemm(ctx, ctx->d_x, d_q, d_k, d_v, d_scores,
                                    n_tokens, n_heads, n_kv_heads,
                                    head_size, kv_mul, kv_dim,
                                    attention_scale) != 0)
        return -1;
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_ATTN);
    if (cuda_matmul_device_out(ctx, ctx->d_out, wo, ctx->d_x, wo_rows,
                               wo_cols, n_tokens, wo_type) != 0)
        return -1;
    residual_rmsnorm_batch_copy_kernel<<<n_tokens, threads,
                                         (size_t)warps * sizeof(float)>>>(
        ctx->d_out, d_orig, d_ffn_residual, d_ffn_norm,
        (const float *)ffn_norm->data, dim, n_tokens, norm_eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill dense layer attn residual/norm failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_WO_RESID);
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_FFN_NORM);
    if (stacked_gateup) {
        if (cuda_matmul_device_out(ctx, d_gateup, gate, d_ffn_norm,
                                   hidden_dim * 2, dim, n_tokens,
                                   gate_type) != 0)
            return -1;
    } else {
        if (cuda_matmul_device_out(ctx, d_gateup, gate, d_ffn_norm,
                                   hidden_dim, dim, n_tokens,
                                   gate_type) != 0 ||
            cuda_matmul_device_out(ctx, d_gateup + hidden_values, up,
                                   d_ffn_norm, hidden_dim, dim, n_tokens,
                                   up_type) != 0)
            return -1;
    }
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_GATEUP);

    int act_total = n_tokens * hidden_dim;
    blocks = (act_total + threads - 1) / threads;
    if (stacked_gateup) {
        ffn_activation_batch_stacked_kernel<<<blocks, threads>>>(
            d_ffn_act, d_gateup, hidden_dim, n_tokens, act_type);
    } else {
        ffn_activation_batch_kernel<<<blocks, threads>>>(
            d_ffn_act, d_gateup, hidden_dim, n_tokens, act_type);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill dense layer ffn activation failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_ACT);
    if (cuda_matmul_device_out(ctx, ctx->d_out, down, d_ffn_act, dim,
                               hidden_dim, n_tokens, down_type) != 0)
        return -1;
    blocks = ((int)dim_values + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads>>>(ctx->d_out, d_ffn_residual,
                                             (int)dim_values);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill dense layer ffn residual failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_DOWN_RESID);

    size_t kv_bytes = kv_values * sizeof(float);
    if (K_out && V_out) {
        err = cudaMemcpy(K_out, d_k, kv_bytes, cudaMemcpyDeviceToHost);
        if (err == cudaSuccess)
            err = cudaMemcpy(V_out, d_v, kv_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill dense layer kv readback failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    } else {
        int kv_total = n_tokens * kv_dim;
        int kv_blocks = (kv_total + threads - 1) / threads;
        prefill_write_kv_cache_kernel<<<kv_blocks, threads>>>(
            cuda_act(ctx, BN_GPU_VALUE_KEY_CACHE),
            cuda_act(ctx, BN_GPU_VALUE_VALUE_CACHE), d_k, d_v, n_tokens,
            kv_dim, kv_cache_stride, kv_cache_off);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill dense layer kv write failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_KV_READBACK);
    if (out) {
        size_t out_bytes = dim_values * sizeof(float);
        if (cuda_ensure_host_out(ctx, out_bytes) != 0)
            return -1;
        err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill dense layer output readback failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        memcpy(out, ctx->h_out, out_bytes);
        BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_OUT_READBACK);
    }
    if (dense_profile)
        cuda_dense_profile_maybe_print(dense_profile_totals,
                                       &dense_profile_layers, n_tokens, dim);
#undef BN_CUDA_DENSE_PROFILE_STEP
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
    case BN_GPU_CODE_PER_HEAD_RMSNORM: return "per_head_rmsnorm";
    case BN_GPU_CODE_COPY: return "copy";
    case BN_GPU_CODE_DEINTERLEAVE_Q: return "deinterleave_q";
    case BN_GPU_CODE_BIAS_ADD: return "bias_add";
    case BN_GPU_CODE_RESIDUAL_ADD: return "residual_add";
    case BN_GPU_CODE_WEIGHTED_ADD: return "weighted_add";
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
    case BN_GPU_CODE_SSM_CONV_SILU: return "ssm_conv_silu";
    case BN_GPU_CODE_SSM_L2NORM: return "ssm_l2norm";
    case BN_GPU_CODE_SSM_ALPHA_BETA: return "ssm_alpha_beta";
    case BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT: return "ssm_alpha_beta_split";
    case BN_GPU_CODE_SSM_DELTA: return "ssm_delta";
    case BN_GPU_CODE_SSM_GATE: return "ssm_gate";
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
    case BN_GPU_CODE_RESIDUAL_ADD:
    case BN_GPU_CODE_WEIGHTED_ADD:
        return op->buf_in == buf;
    case BN_GPU_CODE_MATVEC_SPLIT:
    case BN_GPU_CODE_Q4K_MATVEC_SPLIT:
    case BN_GPU_CODE_Q8_MATVEC_SPLIT:
    case BN_GPU_CODE_Q5K_MATVEC_SPLIT:
        return op->buf_aux == buf || (int)op->rows == buf;
    case BN_GPU_CODE_PER_HEAD_RMSNORM:
        return op->buf_in == buf;
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
    case BN_GPU_CODE_PER_HEAD_RMSNORM:
    case BN_GPU_CODE_COPY:
    case BN_GPU_CODE_DEINTERLEAVE_Q:
    case BN_GPU_CODE_BIAS_ADD:
    case BN_GPU_CODE_SILU_ACT:
    case BN_GPU_CODE_RELU2_ACT:
    case BN_GPU_CODE_ROPE:
    case BN_GPU_CODE_GQA_SCORES:
    case BN_GPU_CODE_FLASH_ATTN:
        return op->buf_in == buf;
    case BN_GPU_CODE_RESIDUAL_RMSNORM:
    case BN_GPU_CODE_RESIDUAL_ADD:
    case BN_GPU_CODE_WEIGHTED_ADD:
    case BN_GPU_CODE_SILU_GATE:
    case BN_GPU_CODE_RELU2_GATE:
    case BN_GPU_CODE_SIGMOID_GATE:
    case BN_GPU_CODE_ROPE_QK:
        return op->buf_in == buf || op->buf_aux == buf;
    default:
        return 0;
    }
}

static int cuda_ops_look_like_decode_graph(const BnGPUOp *ops, int n_ops,
                                           int readback_buf,
                                           const float *out_host,
                                           int out_len) {
    if (!ops || n_ops < 32)
        return 0;

    int has_rope = 0;
    int has_attention = 0;
    int has_logits = 0;
    for (int i = 0; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];
        if (op->op_code == BN_GPU_CODE_ROPE ||
            op->op_code == BN_GPU_CODE_ROPE_QK)
            has_rope = 1;
        if (op->op_code == BN_GPU_CODE_FLASH_ATTN ||
            op->op_code == BN_GPU_CODE_GQA_SCORES ||
            op->op_code == BN_GPU_CODE_GQA_COMBINE)
            has_attention = 1;
        if (op->op_kind == BN_GPU_OP_LOGITS ||
            (op->op_code == BN_GPU_CODE_MATVEC &&
             op->buf_out == BN_GPU_VALUE_LOGITS))
            has_logits = 1;
    }

    int wants_logits_readback =
        readback_buf == BN_GPU_VALUE_LOGITS && out_host && out_len > 0;
    int wants_gpu_resident =
        readback_buf < 0 && !out_host && out_len <= 0;
    return has_rope && has_attention &&
           ((wants_logits_readback && has_logits) || wants_gpu_resident);
}

static int cuda_ops_have_mixed_quant_matvec(const BnGPUOp *ops, int n_ops) {
    if (!ops) return 0;
    for (int i = 0; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];
        if (op->op_kind == BN_GPU_OP_LOGITS ||
            (op->op_code == BN_GPU_CODE_MATVEC &&
             op->buf_out == BN_GPU_VALUE_LOGITS))
            continue;
        switch (op->op_code) {
        case BN_GPU_CODE_MATVEC:
        case BN_GPU_CODE_MATVEC_SPLIT:
        case BN_GPU_CODE_Q4K_MATVEC_SPLIT:
        case BN_GPU_CODE_Q8_MATVEC_SPLIT:
        case BN_GPU_CODE_Q5K_MATVEC_SPLIT:
        case BN_GPU_CODE_FUSED_GATEUP_SILU:
            if (op->type != BN_GGUF_TENSOR_Q8_0)
                return 1;
            break;
        default:
            break;
        }
    }
    return 0;
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

    if (getenv("BN_CUDA_DUMP_OPS") && n_ops > 0) {
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
    static int enable_q4k_dot_flag = 1;
    static int enable_q5k_dot_flag = 1;
    static int enable_q6k_dot_flag = 1;
    static int force_q6k_dot_flag = 0;
    static int enable_q6k_warp_flag = 0;
    static int enable_q4k_4warp_flag = 1;
    static int disable_q8_warp_flag = 0;
    static int disable_qkv_mixed_fuse_flag = 0;
    static int qkv_fuse_key_cache_flag = 1;
    static int enable_qkv_kpair_opt_flag = 1;
    static int disable_q5_gateup_warp_flag = 0;
    static int disable_q8_gateup_warp_flag = 0;
    static int enable_bias_rope_flash_fuse_flag = 0;
    static int enable_graph_exec_flag = 0;
    static int enable_q8_preq_all_flag = 0;
    static int disable_q8_preq_logits_flag = 0;
    if (!flags_init) {
        fuse_bias_enabled_flag =
            getenv("BN_CUDA_DISABLE_FUSE_BIAS") == NULL;
        fuse_rope_flash_enabled_flag =
            getenv("BN_CUDA_DISABLE_ROPE_FLASH_FUSE") == NULL;
        enable_q5_matvec4_flag =
            getenv("BN_CUDA_ENABLE_Q5_MATVEC4") != NULL;
        enable_q5_warp_flag = getenv("BN_CUDA_ENABLE_Q5_WARP") != NULL;
        enable_q4k_dot_flag = getenv("BN_CUDA_DISABLE_Q4K_DOT") == NULL;
        enable_q5k_dot_flag = getenv("BN_CUDA_DISABLE_Q5K_DOT") == NULL;
        enable_q6k_dot_flag = getenv("BN_CUDA_DISABLE_Q6K_DOT") == NULL;
        force_q6k_dot_flag = getenv("BN_CUDA_ENABLE_Q6K_DOT") != NULL;
        enable_q6k_warp_flag = getenv("BN_CUDA_ENABLE_Q6K_WARP") != NULL;
        enable_q4k_4warp_flag =
            getenv("BN_CUDA_DISABLE_Q4K_4WARP") == NULL;
        disable_q8_warp_flag = getenv("BN_CUDA_DISABLE_Q8_WARP") != NULL;
        disable_qkv_mixed_fuse_flag =
            getenv("BN_CUDA_DISABLE_QKV_MIXED_FUSE") != NULL;
        qkv_fuse_key_cache_flag =
            getenv("BN_CUDA_DISABLE_QKV_KCACHE_FUSE") == NULL;
        enable_qkv_kpair_opt_flag =
            getenv("BN_CUDA_ENABLE_QKV_KPAIR_OPT") != NULL;
        disable_q5_gateup_warp_flag =
            getenv("BN_CUDA_DISABLE_Q5_GATEUP_WARP") != NULL;
        disable_q8_gateup_warp_flag =
            getenv("BN_CUDA_DISABLE_Q8_GATEUP_WARP") != NULL;
        enable_bias_rope_flash_fuse_flag =
            getenv("BN_CUDA_ENABLE_BIAS_ROPE_FLASH_FUSE") != NULL;
        enable_graph_exec_flag =
            getenv("BN_CUDA_ENABLE_GRAPH_EXEC") != NULL;
        enable_q8_preq_all_flag =
            getenv("BN_CUDA_ENABLE_Q8_PREQ") != NULL;
        disable_q8_preq_logits_flag =
            getenv("BN_CUDA_DISABLE_Q8_PREQ_LOGITS") != NULL;
        flags_init = 1;
    }
    const int fuse_bias_enabled = fuse_bias_enabled_flag;
    const int fuse_rope_flash_enabled = fuse_rope_flash_enabled_flag;
    const int enable_q5_matvec4 = enable_q5_matvec4_flag;
    const int enable_q5_warp = enable_q5_warp_flag;
    const int enable_q4k_dot = enable_q4k_dot_flag;
    const int enable_q5k_dot = enable_q5k_dot_flag;
    const int enable_q6k_dot = enable_q6k_dot_flag;
    const int force_q6k_dot = force_q6k_dot_flag;
    const int enable_q6k_warp = enable_q6k_warp_flag;
    const int enable_q4k_4warp = enable_q4k_4warp_flag;
    const int disable_q8_warp = disable_q8_warp_flag;
    const int disable_qkv_mixed_fuse = disable_qkv_mixed_fuse_flag;
    const int qkv_fuse_key_cache = qkv_fuse_key_cache_flag;
    const int enable_qkv_kpair_opt = enable_qkv_kpair_opt_flag;
    const int disable_q5_gateup_warp = disable_q5_gateup_warp_flag;
    const int disable_q8_gateup_warp = disable_q8_gateup_warp_flag;
    const int enable_bias_rope_flash_fuse =
        enable_bias_rope_flash_fuse_flag;
    const int enable_q8_preq_all = enable_q8_preq_all_flag;
    const int disable_q8_preq_logits = disable_q8_preq_logits_flag;
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
    int default_graph_exec =
        getenv("BN_CUDA_DISABLE_GRAPH_EXEC") == NULL &&
        getenv("BN_CUDA_ENABLE_MOE_FFN") == NULL &&
        cuda_ops_look_like_decode_graph(ops, n_ops, readback_buf,
                                        out_host, out_len);
    int graph_exec = (enable_graph_exec_flag || default_graph_exec) &&
                     n_ops > 10 && !profile;
    int q8_preq_logits_default =
        !disable_q8_preq_logits && cuda_ops_have_mixed_quant_matvec(ops, n_ops);
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
        int is_logits_op = op->op_kind == BN_GPU_OP_LOGITS ||
                           (op->op_code == BN_GPU_CODE_MATVEC &&
                            op->buf_out == BN_GPU_VALUE_LOGITS);
        int profile_code = is_logits_op
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
                       (op->cols % BN_QK_K) == 0 &&
                       (force_q6k_dot ||
                        (enable_q6k_dot && op->cols >= 2048))) {
                if (cuda_ensure_q8_k(ctx, op->cols, 1) != 0) return -1;
                BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                BN_CUDA_LAUNCH(ctx, quantize_q8k_batch_kernel,
                    dim3(op->cols / BN_QK_K, 1, 1), BN_QK_K, 0,
                    xq, in, op->cols, 1);
                int q6_threads = 256;
                int warps = q6_threads / 32;
                int blocks = (op->rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q6k_dot_matvec_kernel, blocks,
                    q6_threads, 0,
                    out, (const BnBlockQ6K *)w->data, xq, bias,
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
            } else if (op->type == BN_GGUF_TENSOR_Q4_K &&
                       (op->cols % BN_QK_K) == 0 && enable_q4k_dot &&
                       getenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT")) {
                if (cuda_ensure_q8_k(ctx, op->cols, 1) != 0) return -1;
                BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                BN_CUDA_LAUNCH(ctx, quantize_q8k_batch_kernel,
                    dim3(op->cols / BN_QK_K, 1, 1), BN_QK_K, 0,
                    xq, in, op->cols, 1);
                int q4_threads = 256;
                int warps = q4_threads / 32;
                int blocks = (op->rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q4k_q8k_dot_matvec_kernel, blocks,
                    q4_threads, 0,
                    out, (const BnBlockQ4K *)w->data, xq, bias,
                    op->rows, op->cols, out_offset);
            } else if (op->type == BN_GGUF_TENSOR_Q4_K &&
                       (op->cols % BN_QK_K) == 0 && enable_q4k_dot) {
                if (cuda_ensure_q8_1(ctx, op->cols) != 0) return -1;
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH(ctx, quantize_q8_1_kernel,
                    (op->cols + 31) / 32, 32, 0, xq, in, op->cols);
                if (enable_q4k_4warp && op->cols <= 4096) {
                    BN_CUDA_LAUNCH(ctx, q4k_dot_matvec_4warp_kernel,
                        op->rows, 128, 0,
                        out, (const BnBlockQ4K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                } else {
                    int q4_threads = 256;
                    int warps = q4_threads / 32;
                    int blocks = (op->rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH(ctx, q4k_dot_matvec_kernel, blocks,
                        q4_threads, 0,
                        out, (const BnBlockQ4K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                }
            } else if (op->type == BN_GGUF_TENSOR_Q5_K &&
                       (op->cols % BN_QK_K) == 0 && enable_q5k_dot) {
                if (cuda_ensure_q8_1(ctx, op->cols) != 0) return -1;
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH(ctx, quantize_q8_1_kernel,
                    (op->cols + 31) / 32, 32, 0, xq, in, op->cols);
                int q5_threads = 256;
                int warps = q5_threads / 32;
                int blocks = (op->rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q5k_dot_matvec_kernel, blocks,
                    q5_threads, 0,
                    out, (const BnBlockQ5K *)w->data, xq, bias,
                    op->rows, op->cols, out_offset);
            } else if ((enable_q8_preq_all ||
                        (is_logits_op && q8_preq_logits_default)) &&
                       op->type == BN_GGUF_TENSOR_Q8_0 &&
                       (op->cols & 31) == 0) {
                if (cuda_ensure_q8_1(ctx, op->cols) != 0) return -1;
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH(ctx, quantize_q8_1_kernel,
                    (op->cols + 31) / 32, 32, 0, xq, in, op->cols);
                int q8_threads = 256;
                int blocks = (op->rows + (q8_threads / 32) - 1) /
                             (q8_threads / 32);
                BN_CUDA_LAUNCH(ctx, q8_0_matvec_preq_warp8_kernel,
                    blocks, q8_threads, 0,
                    out, (const BnBlockQ8_0 *)w->data, xq, bias, op->rows,
                    op->cols, out_offset);
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
                    const BnCudaBlockQ8_1 *xq_mixed = NULL;
                    if (getenv("BN_CUDA_ENABLE_Q8_MIXED_PREQ") &&
                        (op->type == BN_GGUF_TENSOR_Q8_0 ||
                         ops[i + 5].type == BN_GGUF_TENSOR_Q8_0) &&
                        (cols & 31) == 0) {
                        if (cuda_ensure_q8_1(ctx, cols) != 0) return -1;
                        BnCudaBlockQ8_1 *xq =
                            (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                        BN_CUDA_LAUNCH(ctx, quantize_q8_1_kernel,
                            (cols + 31) / 32, 32, 0, xq, in, cols);
                        xq_mixed = xq;
                    }
                    int q_unused = qkv_fuse_key_cache &&
                        cuda_buf_unused_until_write(ops, n_ops, i + 8,
                                                    op->buf_out);
                    if (q_unused) {
                        BN_CUDA_LAUNCH(ctx, kv_mixed_matvec_kernel,
                            k_grid_rows + ops[i + 5].rows, threads,
                            (size_t)threads * sizeof(float) * 2,
                            key_cache, value_cache, w->data, vw->data, in,
                            xq_mixed,
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
                            xq_mixed,
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
                            xq_mixed,
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
            if (op->type == BN_GGUF_TENSOR_Q4_K &&
                (cols % BN_QK_K) == 0 && split1 != 1 && enable_q4k_dot) {
                if (cuda_ensure_q8_1(ctx, cols) != 0) return -1;
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH(ctx, quantize_q8_1_kernel,
                    (cols + 31) / 32, 32, 0, xq, in, cols);
                int q4_threads = 256;
                int warps = q4_threads / 32;
                int blocks = (total_rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q4k_dot_matvec_split_kernel, blocks,
                    q4_threads, 0,
                    out0, out1, out2, (const BnBlockQ4K *)w->data, xq,
                    bias0, total_rows, cols, split0, split1,
                    (size_t)op->p[6], (size_t)op->p[7]);
            } else if (op->type == BN_GGUF_TENSOR_Q5_K &&
                (cols % BN_QK_K) == 0 && split1 != 1 && enable_q5k_dot) {
                if (cuda_ensure_q8_1(ctx, cols) != 0) return -1;
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH(ctx, quantize_q8_1_kernel,
                    (cols + 31) / 32, 32, 0, xq, in, cols);
                int q5_threads = 256;
                int warps = q5_threads / 32;
                int blocks = (total_rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q5k_dot_matvec_split_kernel, blocks,
                    q5_threads, 0,
                    out0, out1, out2, (const BnBlockQ5K *)w->data, xq,
                    bias0, total_rows, cols, split0, split1,
                    (size_t)op->p[6], (size_t)op->p[7]);
            } else if (op->type == BN_GGUF_TENSOR_Q8_0 &&
                       (cols & 31) == 0 && split1 != 1) {
                if (cuda_ensure_q8_1(ctx, cols) != 0) return -1;
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH(ctx, quantize_q8_1_kernel,
                    (cols + 31) / 32, 32, 0, xq, in, cols);
                int q8_threads = 256;
                int warps = q8_threads / 32;
                int blocks = (total_rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q8_0_matvec_split_preq_warp8_kernel,
                    blocks, q8_threads, 0,
                    out0, out1, out2, (const BnBlockQ8_0 *)w->data, xq,
                    bias0, total_rows, cols, split0, split1,
                    (size_t)op->p[6], (size_t)op->p[7]);
            } else {
                BN_CUDA_LAUNCH(ctx, matvec_split_kernel, total_rows, threads,
                    (size_t)threads * sizeof(float) *
                        (split1 == 1 ? 2u : 1u),
                    out0, out1, out2, w->data, in, bias0, total_rows, cols,
                    op->type, split0, split1, (size_t)op->p[6],
                    (size_t)op->p[7]);
            }
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
            } else if (op->type == BN_GGUF_TENSOR_Q4_K &&
                       (cols % BN_QK_K) == 0 && enable_q4k_dot &&
                       getenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT")) {
                if (cuda_ensure_q8_k(ctx, cols, 1) != 0) return -1;
                BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                BN_CUDA_LAUNCH(ctx, quantize_q8k_batch_kernel,
                    dim3(cols / BN_QK_K, 1, 1), BN_QK_K, 0,
                    xq, in, cols, 1);
                int q4_gateup_threads = 256;
                int warps = q4_gateup_threads / 32;
                int blocks = (gate_rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q4k_q8k_dot_fused_gateup_silu_kernel,
                    blocks, q4_gateup_threads, 0,
                    out, (const BnBlockQ4K *)w->data, xq, gate_rows,
                    up_rows, cols);
            } else if (op->type == BN_GGUF_TENSOR_Q4_K &&
                       (cols % BN_QK_K) == 0 && enable_q4k_dot) {
                if (cuda_ensure_q8_1(ctx, cols) != 0) return -1;
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH(ctx, quantize_q8_1_kernel,
                    (cols + 31) / 32, 32, 0, xq, in, cols);
                if (enable_q4k_4warp && cols <= 4096) {
                    BN_CUDA_LAUNCH(ctx,
                        q4k_dot_fused_gateup_silu_4warp_kernel,
                        gate_rows, 128, 0,
                        out, (const BnBlockQ4K *)w->data, xq, gate_rows,
                        up_rows, cols);
                } else {
                    int q4_gateup_threads = 256;
                    int warps = q4_gateup_threads / 32;
                    int blocks = (gate_rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH(ctx, q4k_dot_fused_gateup_silu_kernel,
                        blocks, q4_gateup_threads, 0,
                        out, (const BnBlockQ4K *)w->data, xq, gate_rows,
                        up_rows, cols);
                }
            } else if (op->type == BN_GGUF_TENSOR_Q5_K &&
                       (cols % BN_QK_K) == 0 && enable_q5k_dot) {
                if (cuda_ensure_q8_1(ctx, cols) != 0) return -1;
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH(ctx, quantize_q8_1_kernel,
                    (cols + 31) / 32, 32, 0, xq, in, cols);
                int q5_gateup_threads = 256;
                int warps = q5_gateup_threads / 32;
                int blocks = (gate_rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q5k_dot_fused_gateup_silu_kernel,
                    blocks, q5_gateup_threads, 0,
                    out, (const BnBlockQ5K *)w->data, xq, gate_rows,
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
        case BN_GPU_CODE_PER_HEAD_RMSNORM: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *in = cuda_act(ctx, op->buf_in);
            int head_size = (int)op->p[0];
            int n_heads = op->rows;
            size_t x_offset = (size_t)op->p[3];
            if (!w || !w->data || !in || n_heads <= 0 || head_size <= 0)
                return -1;
            const BnGPUOp *after_next = (i + 2 < n_ops) ? &ops[i + 2] : NULL;
            const BnGPUOp *flash = (i + 3 < n_ops) ? &ops[i + 3] : NULL;
            if (graph_exec && next && after_next && flash &&
                getenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FLASH_FUSE") == NULL &&
                op->buf_in == BN_GPU_VALUE_Q &&
                next->op_code == BN_GPU_CODE_PER_HEAD_RMSNORM &&
                next->buf_in == BN_GPU_VALUE_KEY_CACHE &&
                after_next->op_code == BN_GPU_CODE_ROPE_QK &&
                after_next->buf_in == BN_GPU_VALUE_Q &&
                after_next->buf_aux == BN_GPU_VALUE_KEY_CACHE &&
                flash->op_code == BN_GPU_CODE_FLASH_ATTN &&
                flash->buf_in == BN_GPU_VALUE_Q &&
                (int)next->p[0] == head_size &&
                (int)next->p[1] == (int)op->p[1] &&
                (int)next->p[2] == (int)op->p[2] &&
                (int)after_next->p[0] == n_heads &&
                (int)after_next->p[1] == head_size &&
                (int)after_next->p[3] > 0 &&
                (int)after_next->p[4] == next->rows &&
                (size_t)after_next->p[5] == (size_t)next->p[3] &&
                (int)flash->p[0] == n_heads &&
                (int)flash->p[1] == head_size &&
                (int)flash->p[3] > 0 &&
                (int)flash->p[4] > 0 &&
                (int)flash->p[5] > 0) {
                BnCudaBuffer *kw = (BnCudaBuffer *)next->W_buf;
                float *out = cuda_act(ctx, flash->buf_out);
                float *key = cuda_act(ctx, BN_GPU_VALUE_KEY_CACHE);
                void *value = cuda_act(ctx, BN_GPU_VALUE_VALUE_CACHE);
                float *freq = cuda_act(ctx, BN_GPU_VALUE_ROPE_FREQ);
                if (!kw || !kw->data || !out || !key || !value || !freq ||
                    !ctx->d_runtime)
                    return -1;
                size_t shared = (size_t)((int)flash->p[5] + threads +
                                         2 * head_size) * sizeof(float);
                BN_CUDA_LAUNCH_STATIC(
                    ctx, qk_norm_rope_flash_runtime_kernel,
                    n_heads, threads, shared,
                    out, in, key, value, freq,
                    (const float *)w->data, (const float *)kw->data,
                    n_heads, next->rows, head_size, (int)flash->p[3],
                    (int)flash->p[4], (int)flash->p[5], flash->p[6],
                    cuda_u32_to_f32(flash->p[7]),
                    cuda_u32_to_f32(op->p[1]), (int)op->p[2],
                    (int)after_next->p[3], ctx->kv_f16,
                    (const BnCudaRuntimeParams *)ctx->d_runtime);
                i += 3;
            } else if (next && after_next &&
                getenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FUSE") == NULL &&
                op->buf_in == BN_GPU_VALUE_Q &&
                next->op_code == BN_GPU_CODE_PER_HEAD_RMSNORM &&
                next->buf_in == BN_GPU_VALUE_KEY_CACHE &&
                after_next->op_code == BN_GPU_CODE_ROPE_QK &&
                after_next->buf_in == BN_GPU_VALUE_Q &&
                after_next->buf_aux == BN_GPU_VALUE_KEY_CACHE &&
                (int)next->p[0] == head_size &&
                (int)next->p[1] == (int)op->p[1] &&
                (int)next->p[2] == (int)op->p[2] &&
                (int)after_next->p[0] == n_heads &&
                (int)after_next->p[1] == head_size &&
                (int)after_next->p[3] > 0 &&
                (int)after_next->p[4] == next->rows &&
                (size_t)after_next->p[5] == (size_t)next->p[3]) {
                BnCudaBuffer *kw = (BnCudaBuffer *)next->W_buf;
                float *key = cuda_act(ctx, BN_GPU_VALUE_KEY_CACHE);
                float *freq = cuda_act(ctx, BN_GPU_VALUE_ROPE_FREQ);
                if (!kw || !kw->data || !key || !freq)
                    return -1;
                BN_CUDA_LAUNCH(ctx, qk_rmsnorm_rope_kernel,
                    n_heads + next->rows, threads,
                    (size_t)threads * sizeof(float),
                    in, key, (const float *)w->data,
                    (const float *)kw->data, freq, n_heads, next->rows,
                    head_size, cuda_u32_to_f32(op->p[1]), (int)op->p[2],
                    (size_t)next->p[3], (int)after_next->p[2],
                    (int)after_next->p[3]);
                i += 2;
            } else {
                BN_CUDA_LAUNCH(ctx, per_head_rmsnorm_kernel, n_heads, threads,
                (size_t)threads * sizeof(float),
                in, (const float *)w->data, n_heads, head_size,
                cuda_u32_to_f32(op->p[1]), (int)op->p[2], x_offset);
            }
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
        case BN_GPU_CODE_DEINTERLEAVE_Q: {
            float *in = cuda_act(ctx, op->buf_in);
            float *out = cuda_act(ctx, op->buf_out);
            int q_dim = (int)op->p[0];
            int head_size = (int)op->p[1];
            if (!in || !out || q_dim <= 0 || head_size <= 0)
                return -1;
            BN_CUDA_LAUNCH(ctx, deinterleave_q_kernel,
                (q_dim + threads - 1) / threads, threads, 0,
                in, out, q_dim, head_size);
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
        case BN_GPU_CODE_WEIGHTED_ADD: {
            float *in = cuda_act(ctx, op->buf_in);
            float *aux = cuda_act(ctx, op->buf_aux);
            int n = (int)op->p[0];
            if (!in || !aux || n <= 0) return -1;
            BN_CUDA_LAUNCH(ctx, weighted_add_kernel,
                (n + threads - 1) / threads, threads, 0,
                in, aux, cuda_u32_to_f32(op->p[1]), n, (int)op->p[2]);
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
                in, aux, n, op->op_code, (int)op->p[1]);
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
        case BN_GPU_CODE_SSM_CONV_SILU: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *qkv = cuda_act(ctx, op->buf_in);
            float *conv_state = cuda_act(ctx, BN_GPU_VALUE_SSM_CONV_STATE);
            int qkv_dim = (int)op->p[0];
            int kern = (int)op->p[1];
            size_t conv_off = (size_t)op->p[2];
            if (!w || !w->data || !qkv || !conv_state ||
                qkv_dim <= 0 || kern <= 1)
                return -1;
            BN_CUDA_LAUNCH(ctx, ssm_conv_silu_kernel,
                (qkv_dim + threads - 1) / threads, threads, 0,
                qkv, conv_state, (const float *)w->data, qkv_dim, kern,
                conv_off);
            break;
        }
        case BN_GPU_CODE_SSM_L2NORM: {
            float *q = cuda_act(ctx, op->buf_in);
            float *k = cuda_act(ctx, op->buf_out);
            int head_dim = (int)op->p[0];
            int q_off = (int)op->p[1];
            int k_off = (int)op->p[2];
            int n_heads = op->rows;
            if (!q || !k || head_dim <= 0 || n_heads <= 0)
                return -1;
            BN_CUDA_LAUNCH(ctx, ssm_l2norm_kernel, n_heads, threads,
                16 * sizeof(float), q, k, head_dim, q_off, k_off);
            break;
        }
        case BN_GPU_CODE_SSM_ALPHA_BETA: {
            BnCudaBuffer *dt = (BnCudaBuffer *)op->W_buf;
            uintptr_t a_raw = (uintptr_t)op->p[6] |
                              ((uintptr_t)op->p[7] << 32);
            BnCudaBuffer *a = (BnCudaBuffer *)a_raw;
            float *alpha = cuda_act(ctx, BN_GPU_VALUE_SSM_ALPHA);
            float *beta = cuda_act(ctx, BN_GPU_VALUE_SSM_BETA);
            int n = (int)op->p[0];
            if (!dt || !dt->data || !a || !a->data ||
                !alpha || !beta || n <= 0)
                return -1;
            BN_CUDA_LAUNCH(ctx, ssm_alpha_beta_kernel, 1, threads, 0,
                alpha, beta, (const float *)dt->data, (const float *)a->data,
                n);
            break;
        }
        case BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT: {
            BnCudaBuffer *dt = (BnCudaBuffer *)op->W_buf;
            uintptr_t a_raw = (uintptr_t)op->p[6] |
                              ((uintptr_t)op->p[7] << 32);
            BnCudaBuffer *a = (BnCudaBuffer *)a_raw;
            const float *src = cuda_act(ctx, op->buf_in);
            float *alpha = cuda_act(ctx, BN_GPU_VALUE_SSM_ALPHA);
            float *beta = cuda_act(ctx, BN_GPU_VALUE_SSM_BETA);
            int n = (int)op->p[0];
            int beta_off = (int)op->p[1];
            if (!dt || !dt->data || !a || !a->data ||
                !src || !alpha || !beta || n <= 0 || beta_off < n)
                return -1;
            BN_CUDA_LAUNCH(ctx, ssm_alpha_beta_split_kernel, 1, threads, 0,
                src, alpha, beta, (const float *)dt->data,
                (const float *)a->data, n, beta_off);
            break;
        }
        case BN_GPU_CODE_SSM_DELTA: {
            float *state = cuda_act(ctx, BN_GPU_VALUE_SSM_STATE);
            float *out = cuda_act(ctx, op->buf_out);
            const float *q = cuda_act(ctx, op->buf_in);
            const float *k = cuda_act(ctx, op->buf_aux);
            int v_buf = op->p[7] ? op->buf_in : BN_GPU_VALUE_SSM_V;
            const float *v = cuda_act(ctx, v_buf);
            const float *alpha = cuda_act(ctx, BN_GPU_VALUE_SSM_ALPHA);
            const float *beta = cuda_act(ctx, BN_GPU_VALUE_SSM_BETA);
            int head_k_dim = (int)op->p[0];
            int head_v_dim = (int)op->p[1];
            int num_k_heads = (int)op->p[2];
            float q_scale = cuda_u32_to_f32(op->p[3]);
            size_t state_off = (size_t)op->p[4] / sizeof(float);
            int q_off = (int)op->p[6];
            int k_off = (int)op->p[7];
            int v_off = k_off ? 2 * num_k_heads * head_k_dim : 0;
            int num_v_heads = op->rows;
            if (!state || !out || !q || !k || !v || !alpha || !beta ||
                head_k_dim <= 0 || head_v_dim <= 0 || num_k_heads <= 0 ||
                num_v_heads <= 0)
                return -1;
            if (head_k_dim == 128 && head_v_dim == 128) {
                BN_CUDA_LAUNCH(ctx, ssm_delta_128_warp_kernel,
                    dim3(num_v_heads, 32, 1), dim3(32, 4, 1), 0,
                    state, out, q, k, v, alpha, beta, num_k_heads, q_scale,
                    state_off, q_off, k_off, v_off);
            } else {
                BN_CUDA_LAUNCH(ctx, ssm_delta_kernel, num_v_heads, threads,
                    (size_t)head_v_dim * sizeof(float), state, out, q, k, v,
                    alpha, beta, head_k_dim, head_v_dim, num_k_heads,
                    q_scale, state_off, q_off, k_off, v_off);
            }
            break;
        }
        case BN_GPU_CODE_SSM_GATE: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *out = cuda_act(ctx, op->buf_in);
            const float *z = cuda_act(ctx, op->buf_aux);
            int head_v_dim = (int)op->p[0];
            float eps = cuda_u32_to_f32(op->p[1]);
            int num_v_heads = op->rows;
            if (!w || !w->data || !out || !z ||
                head_v_dim <= 0 || num_v_heads <= 0)
                return -1;
            BN_CUDA_LAUNCH(ctx, ssm_gate_kernel, num_v_heads, threads,
                8 * sizeof(float), out, z, (const float *)w->data,
                head_v_dim, eps);
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
        size_t readback_bytes = (size_t)out_len * sizeof(float);
        if (cuda_ensure_host_out(ctx, readback_bytes) != 0) {
            if (profile) {
                cudaEventDestroy(ev_start);
                cudaEventDestroy(ev_stop);
            }
            return -1;
        }
        if (ctx->exec_stream) {
            err = cudaMemcpyAsync(ctx->h_out, src, readback_bytes,
                                  cudaMemcpyDeviceToHost,
                                  ctx->exec_stream);
        } else {
            err = cudaMemcpy(ctx->h_out, src, readback_bytes,
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
        memcpy(out_host, ctx->h_out, readback_bytes);
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
    cublasStatus_t blas_err = cublasCreate(&ctx->cublas);
    if (blas_err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[bn:gpu:cuda] cuBLAS create failed: status %d\n",
                (int)blas_err);
        cudaStreamDestroy(ctx->stream);
        free(ctx);
        free(gpu);
        return NULL;
    }
    (void)cublasSetMathMode(ctx->cublas, CUBLAS_TENSOR_OP_MATH);
    gpu->buffer_create = cuda_buffer_create;
    gpu->buffer_destroy = cuda_buffer_destroy;
    gpu->matvec = cuda_matvec;
    gpu->matmul = cuda_matmul;
    gpu->matmul_batch = cuda_matmul_batch;
    gpu->matvec_batch = cuda_matvec_batch;
    gpu->dense_ffn = cuda_dense_ffn;
    gpu->dense_ffn_batch = cuda_dense_ffn_batch;
    gpu->dense_ffn_batch_norm = cuda_dense_ffn_batch_norm;
    gpu->dense_ffn_batch_norm_resid = cuda_dense_ffn_batch_norm_resid;
    gpu->prefill_attention = cuda_prefill_attention;
    gpu->prefill_attention_wo = cuda_prefill_attention_wo;
    gpu->prefill_qkv_attention_wo = cuda_prefill_qkv_attention_wo;
    gpu->prefill_qkv_attention_wo_norm = cuda_prefill_qkv_attention_wo_norm;
    gpu->prefill_qkv_attention_wo_norm_resid =
        cuda_prefill_qkv_attention_wo_norm_resid;
    gpu->prefill_dense_layer = cuda_prefill_dense_layer;
    gpu->init_activations = cuda_init_activations;
    gpu->free_activations = cuda_free_activations;
    gpu->write_activation = cuda_write_activation;
    gpu->read_activation = cuda_read_activation;
    gpu->argmax_activation = cuda_argmax_activation;
    gpu->execute = cuda_execute;
    gpu->ctx = ctx;
    gpu->kind = BN_GPU_BACKEND_CUDA;
    gpu->max_storage_binding_size = (size_t)-1;
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
        if (ctx->d_q8_1) cudaFree(ctx->d_q8_1);
        if (ctx->d_q8_k) cudaFree(ctx->d_q8_k);
        if (ctx->d_x_f16) cudaFree(ctx->d_x_f16);
        if (ctx->d_argmax) cudaFree(ctx->d_argmax);
        if (ctx->d_penalty_tokens) cudaFree(ctx->d_penalty_tokens);
        if (ctx->d_gemm_ptrs) cudaFree(ctx->d_gemm_ptrs);
        if (ctx->d_prefill) cudaFree(ctx->d_prefill);
        if (ctx->exec_graph) cudaGraphExecDestroy(ctx->exec_graph);
        if (ctx->exec_graph_def) cudaGraphDestroy(ctx->exec_graph_def);
        free(ctx->exec_nodes);
        free(ctx->h_gemm_ptrs);
        if (ctx->h_out) cudaFreeHost(ctx->h_out);
        cuda_free_activations(ctx);
        if (ctx->cublas) cublasDestroy(ctx->cublas);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
    }
    free(ctx);
    free(gpu);
}

#endif // BN_ENABLE_CUDA
