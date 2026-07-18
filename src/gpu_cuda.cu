#include "gpu_cuda.h"

#ifdef BN_ENABLE_CUDA

#include "gguf.h"
#include "model_config.h"
#include "moe_types.h"
#include "quant.h"
#include "gpu_policy.h"
#include "gpu_shader.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

#define static static __device__ __constant__
#include "iq_tables.h"
#undef static

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
    cudaEvent_t ssm_stream_entry_event;
    cudaEvent_t ssm_stream_exit_event;
    cublasHandle_t cublas;
    void *cublas_workspace;
    size_t cublas_workspace_bytes;
    cudaGraph_t exec_graph_def;
    cudaGraphExec_t exec_graph;
    cudaGraphNode_t *exec_nodes;
    int exec_nodes_cap;
    int exec_node_count;
    int exec_node_cursor;
    int exec_graph_ops;
    int kv_f16;
    int has_moe_model;
    float *d_x;
    size_t d_x_bytes;
    float *d_out;
    size_t d_out_bytes;
    float *d_resident_in;
    size_t d_resident_in_bytes;
    void *d_ops;
    size_t d_ops_bytes;
    float *h_out;
    size_t h_out_bytes;
    void *d_runtime;
    void *d_q8_1;
    size_t d_q8_1_bytes;
    void *d_q8_0_f32;
    size_t d_q8_0_f32_bytes;
    void *d_q8_k;
    size_t d_q8_k_bytes;
    void *d_x_f16;
    size_t d_x_f16_bytes;
    void *d_argmax;
    size_t d_argmax_bytes;
    int *h_argmax;
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

#define BN_CUDA_ARGMAX_PENALTY_SORT_MAX 256

static int cuda_prepare_penalty_tokens(const int *src, int n, int *dst,
                                       int cap) {
    if (!src || !dst || n <= 0 || cap <= 0) return 0;
    if (n > cap) return -1;
    int out = 0;
    for (int i = 0; i < n; i++) {
        int tok = src[i];
        int lo = 0;
        int hi = out;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (dst[mid] < tok)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < out && dst[lo] == tok)
            continue;
        for (int j = out; j > lo; j--)
            dst[j] = dst[j - 1];
        dst[lo] = tok;
        out++;
    }
    return out;
}

static int cuda_penalty_tokens_contain_sorted(const int *tokens, int n,
                                              int token) {
    int lo = 0;
    int hi = n;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (tokens[mid] < token)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < n && tokens[lo] == token;
}

struct BnCudaExecStreamScope {
    BnCudaCtx *ctx;
    cudaStream_t prev;
    int active;

    BnCudaExecStreamScope(BnCudaCtx *c, cudaStream_t stream, int enable)
        : ctx(c), prev(c ? c->exec_stream : (cudaStream_t)0),
          active(c && enable) {
        if (active)
            ctx->exec_stream = stream;
    }

    ~BnCudaExecStreamScope() {
        if (active) {
            ctx->exec_stream = prev;
            if (ctx->cublas)
                cublasSetStream(ctx->cublas, prev);
        }
    }
};

static int cuda_ensure_gemm_ptrs(BnCudaCtx *ctx, int n_ptrs);

typedef struct {
    uint16_t d;
    uint16_t s;
    int8_t qs[32];
} BnCudaBlockQ8_1;

typedef struct {
    float d;
    int8_t qs[32];
} BnCudaBlockQ8_0F32;

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
    BN_CUDA_LAUNCH_EX(ctx, graph_static_params ? 0 : 1, kernel, grid, block, shared, __VA_ARGS__)

#define BN_CUDA_LAUNCH_STABLE(ctx, stable, kernel, grid, block, shared, ...) do { \
    if ((stable) && graph_static_params) { \
        BN_CUDA_LAUNCH_STATIC(ctx, kernel, grid, block, shared, __VA_ARGS__); \
    } else { \
        BN_CUDA_LAUNCH(ctx, kernel, grid, block, shared, __VA_ARGS__); \
    } \
} while (0)

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

static __device__ float cuda_bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static __device__ uint16_t cuda_fp32_to_fp16_bits(float f) {
    return __half_as_ushort(__float2half_rn(f));
}

static __device__ float cuda_q4k_value(const BnBlockQ4K *blk, int i);
static __device__ float cuda_q5k_value(const BnBlockQ5K *blk, int i);
static __device__ __forceinline__ float cuda_q6k_value(const BnBlockQ6K *blk,
                                                       int i);

static __device__ __forceinline__ float cuda_silu_select(float x,
                                                         int exact_silu) {
    return exact_silu
        ? x / (1.0f + expf(-x))
        : x / (1.0f + __expf(-x));
}

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

static __global__ void dequant_bf16_to_f16_kernel(
    __half *out, const uint16_t *src, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * (size_t)cols;
    if (i >= n) return;
    out[i] = __float2half_rn(cuda_bf16_to_fp32(src[i]));
}

static __device__ __forceinline__ float cuda_q3k_value(const BnBlockQ3K *blk,
                                                       int i) {
    uint32_t aux0 = (uint32_t)blk->scales[0] |
                    ((uint32_t)blk->scales[1] << 8) |
                    ((uint32_t)blk->scales[2] << 16) |
                    ((uint32_t)blk->scales[3] << 24);
    uint32_t aux1 = (uint32_t)blk->scales[4] |
                    ((uint32_t)blk->scales[5] << 8) |
                    ((uint32_t)blk->scales[6] << 16) |
                    ((uint32_t)blk->scales[7] << 24);
    uint32_t tmp = (uint32_t)blk->scales[8] |
                   ((uint32_t)blk->scales[9] << 8) |
                   ((uint32_t)blk->scales[10] << 16) |
                   ((uint32_t)blk->scales[11] << 24);
    uint32_t aux2 = ((aux0 >> 4) & 0x0f0f0f0fu) |
                    (((tmp >> 4) & 0x03030303u) << 4);
    uint32_t aux3 = ((aux1 >> 4) & 0x0f0f0f0fu) |
                    (((tmp >> 6) & 0x03030303u) << 4);
    aux0 = (aux0 & 0x0f0f0f0fu) |
           (((tmp >> 0) & 0x03030303u) << 4);
    aux1 = (aux1 & 0x0f0f0f0fu) |
           (((tmp >> 2) & 0x03030303u) << 4);
    uint32_t aux[4] = { aux0, aux1, aux2, aux3 };

    int group = i >> 4;
    int scale = (int)((aux[group >> 2] >> ((group & 3) * 8)) & 0xffu) - 32;
    int half = i >> 7;
    int in_half = i & 127;
    int j = in_half >> 5;
    int l = in_half & 15;
    int shift = j * 2;
    const uint8_t *q = blk->qs + half * 32;
    const uint8_t *hm = blk->hmask + half * 16;
    uint8_t m = (uint8_t)(1u << (group >> 1));
    int qoff = (group & 1) ? l + 16 : l;
    int q3 = ((q[qoff] >> shift) & 3) - ((hm[qoff] & m) ? 0 : 4);
    return cuda_fp16_to_fp32(blk->d) * (float)scale * (float)q3;
}

static __global__ void dequant_q3k_to_f16_kernel(
    __half *out, const BnBlockQ3K *blocks, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * (size_t)cols;
    if (i >= n) return;
    int row = (int)(i / (size_t)cols);
    int col = (int)(i - (size_t)row * (size_t)cols);
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ3K *blk = &blocks[(size_t)row * n_bpr +
                                    col / BN_QK_K];
    out[i] = __float2half_rn(cuda_q3k_value(blk, col & (BN_QK_K - 1)));
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

static __global__ void dequant_q4k_to_f32_kernel(
    float *out, const BnBlockQ4K *blocks, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * (size_t)cols;
    if (i >= n) return;
    int row = (int)(i / (size_t)cols);
    int col = (int)(i - (size_t)row * (size_t)cols);
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ4K *blk = &blocks[(size_t)row * n_bpr +
                                    col / BN_QK_K];
    out[i] = cuda_q4k_value(blk, col & (BN_QK_K - 1));
}

static __global__ void dequant_q5k_to_f16_kernel(
    __half *out, const BnBlockQ5K *blocks, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * (size_t)cols;
    if (i >= n) return;
    int row = (int)(i / (size_t)cols);
    int col = (int)(i - (size_t)row * (size_t)cols);
    int n_bpr = cols / BN_QK_K;
    const BnBlockQ5K *blk = &blocks[(size_t)row * n_bpr +
                                    col / BN_QK_K];
    out[i] = __float2half_rn(cuda_q5k_value(blk, col & (BN_QK_K - 1)));
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

static __device__ __forceinline__ float cuda_q6k_value(const BnBlockQ6K *blk,
                                                       int i) {
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
    int q = d == 0.0f ? 0 : (int)roundf(v / d);
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
    int q = d == 0.0f ? 0 : (int)roundf(v / d);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);

    BnCudaBlockQ8_1 *dst = out + (size_t)token * n_blocks + block;
    dst->qs[lane] = (int8_t)q;
    if (lane == 0) {
        dst->d = cuda_fp32_to_fp16_bits(d);
        dst->s = cuda_fp32_to_fp16_bits(sum);
    }
}

static __global__ void quantize_q8_0_f32_batch_kernel(
    BnCudaBlockQ8_0F32 *out, const float *x, int cols, int n_tokens) {
    int block = blockIdx.x;
    int token = blockIdx.y;
    int lane = threadIdx.x;
    if (token >= n_tokens) return;
    int n_blocks = (cols + 31) / 32;
    int c = block * 32 + lane;
    const float *x_token = x + (size_t)token * cols;
    float v = c < cols ? x_token[c] : 0.0f;
    float amax = fabsf(v);

    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
    amax = __shfl_sync(0xffffffffu, amax, 0);
    float d = amax / 127.0f;
    int q = d == 0.0f ? 0 : __float2int_rn(v / d);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);

    BnCudaBlockQ8_0F32 *dst = out + (size_t)token * n_blocks + block;
    dst->qs[lane] = (int8_t)q;
    if (lane == 0)
        dst->d = d;
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
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    float scaled = v * id;
    int q = (int)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
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

static __device__ __forceinline__ float cuda_vec_dot_q4k_q8k_half(
    const BnBlockQ4K *blk, const BnBlockQ8K *xq, int half) {
    float xd = cuda_fp16_to_fp32(blk->d);
    float xmin = cuda_fp16_to_fp32(blk->dmin);
    int isum = 0;
    int summs = 0;
    int group0 = half ? 4 : 0;
#pragma unroll
    for (int g = 0; g < 4; g++) {
        int group = group0 + g;
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

static __global__ void q4k_q8k_dot_matvec4_kernel(float *out,
                                                  const BnBlockQ4K *blocks,
                                                  const BnBlockQ8K *xq,
                                                  const float *bias,
                                                  int rows, int cols,
                                                  size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    if (row >= rows) return;

    int n_bpr = cols / BN_QK_K;
    float sum = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = sublane; b < n_bpr; b += 8)
        sum += cuda_vec_dot_q4k_q8k(&row_blocks[b], xq + b);

    unsigned mask = 0xffu << (lane_group * 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0) {
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q4k_q8k_dot_matvec_pair_kernel(
        float *out0, float *out1,
        const BnBlockQ4K *blocks0, const BnBlockQ4K *blocks1,
        const BnBlockQ8K *xq, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;

    int n_bpr = cols / BN_QK_K;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const BnBlockQ4K *row_blocks0 = blocks0 + (size_t)row * n_bpr;
    const BnBlockQ4K *row_blocks1 = blocks1 + (size_t)row * n_bpr;
    for (int b = lane; b < n_bpr; b += 32) {
        sum0 += cuda_vec_dot_q4k_q8k(&row_blocks0[b], xq + b);
        sum1 += cuda_vec_dot_q4k_q8k(&row_blocks1[b], xq + b);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }
    if (lane == 0) {
        out0[row] = sum0;
        out1[row] = sum1;
    }
}

static __global__ void q4k_q8k_dot_matvec_split_kernel(
    float *out0, float *out1, float *out2, const BnBlockQ4K *blocks,
    const BnBlockQ8K *xq, const float *bias0, int total_rows, int cols,
    int split0, int split1, size_t out1_offset, size_t out2_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= total_rows) return;

    int n_bpr = cols / BN_QK_K;
    float sum = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = lane; b < n_bpr; b += 32)
        sum += cuda_vec_dot_q4k_q8k(&row_blocks[b], xq + b);

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

static __global__ void q4k_q8k_dot_matmul_kernel(
        float *out, const BnBlockQ4K *blocks, const BnBlockQ8K *xq,
        int rows, int cols, int n_tokens, size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token = blockIdx.y;
    if (row >= rows || token >= n_tokens) return;

    int n_bpr = cols / BN_QK_K;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ8K *xq_token = xq + (size_t)token * n_bpr;
    float sum = 0.0f;
    for (int b = lane; b < n_bpr; b += 32)
        sum += cuda_vec_dot_q4k_q8k(&row_blocks[b], xq_token + b);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[out_offset + (size_t)token * rows + row] = sum;
}

static __global__ void q4k_q8k_dot_matmul4_token_kernel(
        float *out, const BnBlockQ4K *blocks, const BnBlockQ8K *xq,
        int rows, int cols, int n_tokens, size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token0 = blockIdx.y * 4;
    if (row >= rows || token0 >= n_tokens) return;

    int n_bpr = cols / BN_QK_K;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ8K *xq0 = xq + (size_t)token0 * n_bpr;
    const BnBlockQ8K *xq1 = xq0 + n_bpr;
    const BnBlockQ8K *xq2 = xq1 + n_bpr;
    const BnBlockQ8K *xq3 = xq2 + n_bpr;
    int have1 = token0 + 1 < n_tokens;
    int have2 = token0 + 2 < n_tokens;
    int have3 = token0 + 3 < n_tokens;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (int b = lane; b < n_bpr; b += 32) {
        const BnBlockQ4K *blk = &row_blocks[b];
        sum0 += cuda_vec_dot_q4k_q8k(blk, xq0 + b);
        if (have1) sum1 += cuda_vec_dot_q4k_q8k(blk, xq1 + b);
        if (have2) sum2 += cuda_vec_dot_q4k_q8k(blk, xq2 + b);
        if (have3) sum3 += cuda_vec_dot_q4k_q8k(blk, xq3 + b);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
    }
    if (lane == 0) {
        out[out_offset + (size_t)token0 * rows + row] = sum0;
        if (have1) out[out_offset + (size_t)(token0 + 1) * rows + row] = sum1;
        if (have2) out[out_offset + (size_t)(token0 + 2) * rows + row] = sum2;
        if (have3) out[out_offset + (size_t)(token0 + 3) * rows + row] = sum3;
    }
}

static __device__ __forceinline__ float cuda_vec_dot_q5k_q8_1(
    const BnBlockQ5K *blk, const BnCudaBlockQ8_1 *xq, int iqs) {
    int qlow[2];
    int qhigh[2];
    int u[4];
    float d8[2];

    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int q5_offset = 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    const int h_offset = 4 * ((iqs / 2) % 4);
    const int *q5 = (const int *)(blk->qs + q5_offset);
    const int *qh = (const int *)(blk->qh + h_offset);
    qlow[0] = q5[0];
    qlow[1] = q5[4];
    qhigh[0] = qh[0] >> bq8_offset;
    qhigh[1] = qh[4] >> bq8_offset;

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
        int v0 = (((qhigh[0] >> i) << 4) & 0x10101010) |
                 ((qlow[0] >> (4 * i)) & 0x0f0f0f0f);
        int v1 = (((qhigh[1] >> i) << 4) & 0x10101010) |
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

static __global__ void q4k_dot_matvec_4warp_residual_kernel(
                                                   float *x,
                                                   const BnBlockQ4K *blocks,
                                                   const BnCudaBlockQ8_1 *xq,
                                                   const float *residual,
                                                   int rows, int cols) {
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
    for (int b = warp * 2 + kbx; b < n_bpr; b += 8) {
        const BnCudaBlockQ8_1 *xqb = xq + (size_t)b * 8;
        sum += cuda_vec_dot_q4k_q8_1(&row_blocks[b], xqb, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();

    if (warp == 0 && lane == 0)
        x[row] = residual[row] + partial[0] + partial[1] +
                 partial[2] + partial[3];
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

static __global__ void q5k_dot_matvec_4warp_kernel(float *out,
                                                   const BnBlockQ5K *blocks,
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
    const BnBlockQ5K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = warp * 2 + kbx; b < n_bpr; b += 8)
        sum += cuda_vec_dot_q5k_q8_1(&row_blocks[b], xq + (size_t)b * 8,
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

static __global__ void q4k_dot_matmul4_token_kernel(
        float *out, const BnBlockQ4K *blocks, const BnCudaBlockQ8_1 *xq,
        int rows, int cols, int n_tokens, size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token0 = blockIdx.y * 4;
    if (row >= rows || token0 >= n_tokens) return;

    int n_bpr = cols / BN_QK_K;
    int x_blocks = (cols + 31) / 32;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnCudaBlockQ8_1 *xq0 = xq + (size_t)token0 * x_blocks;
    const BnCudaBlockQ8_1 *xq1 = xq0 + x_blocks;
    const BnCudaBlockQ8_1 *xq2 = xq1 + x_blocks;
    const BnCudaBlockQ8_1 *xq3 = xq2 + x_blocks;
    int have1 = token0 + 1 < n_tokens;
    int have2 = token0 + 2 < n_tokens;
    int have3 = token0 + 3 < n_tokens;
    for (int b = kbx; b < n_bpr; b += 2) {
        const BnBlockQ4K *blk = &row_blocks[b];
        sum0 += cuda_vec_dot_q4k_q8_1(blk, xq0 + (size_t)b * 8, iqs);
        if (have1)
            sum1 += cuda_vec_dot_q4k_q8_1(blk, xq1 + (size_t)b * 8, iqs);
        if (have2)
            sum2 += cuda_vec_dot_q4k_q8_1(blk, xq2 + (size_t)b * 8, iqs);
        if (have3)
            sum3 += cuda_vec_dot_q4k_q8_1(blk, xq3 + (size_t)b * 8, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
    }
    if (lane == 0) {
        out[out_offset + (size_t)token0 * rows + row] = sum0;
        if (have1) out[out_offset + (size_t)(token0 + 1) * rows + row] = sum1;
        if (have2) out[out_offset + (size_t)(token0 + 2) * rows + row] = sum2;
        if (have3) out[out_offset + (size_t)(token0 + 3) * rows + row] = sum3;
    }
}

static __global__ void q4k_dot_matmul4_token_sharedx_kernel(
        float *out, const BnBlockQ4K *blocks, const BnCudaBlockQ8_1 *xq,
        int rows, int cols, int n_tokens, size_t out_offset) {
    extern __shared__ BnCudaBlockQ8_1 sx[];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token0 = blockIdx.y * 4;
    if (token0 >= n_tokens) return;

    int x_blocks = (cols + 31) / 32;
    int nt = n_tokens - token0;
    if (nt > 4) nt = 4;
    int ncopy = nt * x_blocks;
    const BnCudaBlockQ8_1 *xq_base = xq + (size_t)token0 * x_blocks;
    for (int i = threadIdx.x; i < ncopy; i += blockDim.x)
        sx[i] = xq_base[i];
    __syncthreads();

    if (row >= rows) return;
    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    int have1 = nt > 1;
    int have2 = nt > 2;
    int have3 = nt > 3;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnCudaBlockQ8_1 *xq0 = sx;
    const BnCudaBlockQ8_1 *xq1 = xq0 + x_blocks;
    const BnCudaBlockQ8_1 *xq2 = xq1 + x_blocks;
    const BnCudaBlockQ8_1 *xq3 = xq2 + x_blocks;
    for (int b = kbx; b < n_bpr; b += 2) {
        const BnBlockQ4K *blk = &row_blocks[b];
        sum0 += cuda_vec_dot_q4k_q8_1(blk, xq0 + (size_t)b * 8, iqs);
        if (have1)
            sum1 += cuda_vec_dot_q4k_q8_1(blk, xq1 + (size_t)b * 8, iqs);
        if (have2)
            sum2 += cuda_vec_dot_q4k_q8_1(blk, xq2 + (size_t)b * 8, iqs);
        if (have3)
            sum3 += cuda_vec_dot_q4k_q8_1(blk, xq3 + (size_t)b * 8, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
    }
    if (lane == 0) {
        out[out_offset + (size_t)token0 * rows + row] = sum0;
        if (have1) out[out_offset + (size_t)(token0 + 1) * rows + row] = sum1;
        if (have2) out[out_offset + (size_t)(token0 + 2) * rows + row] = sum2;
        if (have3) out[out_offset + (size_t)(token0 + 3) * rows + row] = sum3;
    }
}

static __global__ void q4k_dot_matmul8_token_sharedx_kernel(
        float *out, const BnBlockQ4K *blocks, const BnCudaBlockQ8_1 *xq,
        int rows, int cols, int n_tokens, size_t out_offset) {
    extern __shared__ BnCudaBlockQ8_1 sx[];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token0 = blockIdx.y * 8;
    if (token0 >= n_tokens) return;

    int x_blocks = (cols + 31) / 32;
    int nt = n_tokens - token0;
    if (nt > 8) nt = 8;
    int ncopy = nt * x_blocks;
    const BnCudaBlockQ8_1 *xq_base = xq + (size_t)token0 * x_blocks;
    for (int i = threadIdx.x; i < ncopy; i += blockDim.x)
        sx[i] = xq_base[i];
    __syncthreads();

    if (row >= rows) return;
    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnCudaBlockQ8_1 *xq0 = sx;
    const BnCudaBlockQ8_1 *xq1 = xq0 + x_blocks;
    const BnCudaBlockQ8_1 *xq2 = xq1 + x_blocks;
    const BnCudaBlockQ8_1 *xq3 = xq2 + x_blocks;
    const BnCudaBlockQ8_1 *xq4 = xq3 + x_blocks;
    const BnCudaBlockQ8_1 *xq5 = xq4 + x_blocks;
    const BnCudaBlockQ8_1 *xq6 = xq5 + x_blocks;
    const BnCudaBlockQ8_1 *xq7 = xq6 + x_blocks;
    for (int b = kbx; b < n_bpr; b += 2) {
        const BnBlockQ4K *blk = &row_blocks[b];
        size_t xb = (size_t)b * 8;
        sum0 += cuda_vec_dot_q4k_q8_1(blk, xq0 + xb, iqs);
        if (nt > 1) sum1 += cuda_vec_dot_q4k_q8_1(blk, xq1 + xb, iqs);
        if (nt > 2) sum2 += cuda_vec_dot_q4k_q8_1(blk, xq2 + xb, iqs);
        if (nt > 3) sum3 += cuda_vec_dot_q4k_q8_1(blk, xq3 + xb, iqs);
        if (nt > 4) sum4 += cuda_vec_dot_q4k_q8_1(blk, xq4 + xb, iqs);
        if (nt > 5) sum5 += cuda_vec_dot_q4k_q8_1(blk, xq5 + xb, iqs);
        if (nt > 6) sum6 += cuda_vec_dot_q4k_q8_1(blk, xq6 + xb, iqs);
        if (nt > 7) sum7 += cuda_vec_dot_q4k_q8_1(blk, xq7 + xb, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
        sum4 += __shfl_down_sync(0xffffffffu, sum4, offset);
        sum5 += __shfl_down_sync(0xffffffffu, sum5, offset);
        sum6 += __shfl_down_sync(0xffffffffu, sum6, offset);
        sum7 += __shfl_down_sync(0xffffffffu, sum7, offset);
    }
    if (lane == 0) {
        out[out_offset + (size_t)token0 * rows + row] = sum0;
        if (nt > 1) out[out_offset + (size_t)(token0 + 1) * rows + row] = sum1;
        if (nt > 2) out[out_offset + (size_t)(token0 + 2) * rows + row] = sum2;
        if (nt > 3) out[out_offset + (size_t)(token0 + 3) * rows + row] = sum3;
        if (nt > 4) out[out_offset + (size_t)(token0 + 4) * rows + row] = sum4;
        if (nt > 5) out[out_offset + (size_t)(token0 + 5) * rows + row] = sum5;
        if (nt > 6) out[out_offset + (size_t)(token0 + 6) * rows + row] = sum6;
        if (nt > 7) out[out_offset + (size_t)(token0 + 7) * rows + row] = sum7;
    }
}

static __global__ void q5k_dot_matmul_kernel(float *out,
                                             const BnBlockQ5K *blocks,
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
    const BnBlockQ5K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnCudaBlockQ8_1 *xq_token = xq + (size_t)token * x_blocks;
    for (int b = kbx; b < n_bpr; b += 2)
        sum += cuda_vec_dot_q5k_q8_1(&row_blocks[b],
                                     xq_token + (size_t)b * 8, iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[out_offset + (size_t)token * rows + row] = sum;
}

static __global__ void q5k_dot_matmul4_token_kernel(
        float *out, const BnBlockQ5K *blocks, const BnCudaBlockQ8_1 *xq,
        int rows, int cols, int n_tokens, size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token0 = blockIdx.y * 4;
    if (row >= rows || token0 >= n_tokens) return;

    int n_bpr = cols / BN_QK_K;
    int x_blocks = (cols + 31) / 32;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    const BnBlockQ5K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnCudaBlockQ8_1 *xq0 = xq + (size_t)token0 * x_blocks;
    const BnCudaBlockQ8_1 *xq1 = xq0 + x_blocks;
    const BnCudaBlockQ8_1 *xq2 = xq1 + x_blocks;
    const BnCudaBlockQ8_1 *xq3 = xq2 + x_blocks;
    int have1 = token0 + 1 < n_tokens;
    int have2 = token0 + 2 < n_tokens;
    int have3 = token0 + 3 < n_tokens;
    for (int b = kbx; b < n_bpr; b += 2) {
        const BnBlockQ5K *blk = &row_blocks[b];
        sum0 += cuda_vec_dot_q5k_q8_1(blk, xq0 + (size_t)b * 8, iqs);
        if (have1)
            sum1 += cuda_vec_dot_q5k_q8_1(blk, xq1 + (size_t)b * 8, iqs);
        if (have2)
            sum2 += cuda_vec_dot_q5k_q8_1(blk, xq2 + (size_t)b * 8, iqs);
        if (have3)
            sum3 += cuda_vec_dot_q5k_q8_1(blk, xq3 + (size_t)b * 8, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
    }
    if (lane == 0) {
        out[out_offset + (size_t)token0 * rows + row] = sum0;
        if (have1) out[out_offset + (size_t)(token0 + 1) * rows + row] = sum1;
        if (have2) out[out_offset + (size_t)(token0 + 2) * rows + row] = sum2;
        if (have3) out[out_offset + (size_t)(token0 + 3) * rows + row] = sum3;
    }
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

static __global__ void q4k_dot_matvec_split_5warp_kernel(
    float *out0, float *out1, float *out2, const BnBlockQ4K *blocks,
    const BnCudaBlockQ8_1 *xq, const float *bias0, int total_rows, int cols,
    int split0, int split1, size_t out1_offset, size_t out2_offset) {
    __shared__ float partial[5];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= total_rows || warp >= 5) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = warp * 2 + kbx; b < n_bpr; b += 10)
        sum += cuda_vec_dot_q4k_q8_1(&row_blocks[b],
                                     xq + (size_t)b * 8, iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();
    if (warp != 0 || lane != 0) return;

    sum = partial[0] + partial[1] + partial[2] + partial[3] + partial[4];
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

static __global__ void q4k_dot_matvec_split_4warp_kernel(
    float *out0, float *out1, float *out2, const BnBlockQ4K *blocks,
    const BnCudaBlockQ8_1 *xq, const float *bias0, int total_rows, int cols,
    int split0, int split1, size_t out1_offset, size_t out2_offset) {
    __shared__ float partial[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= total_rows || warp >= 4) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum = 0.0f;
    const BnBlockQ4K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = warp * 2 + kbx; b < n_bpr; b += 8)
        sum += cuda_vec_dot_q4k_q8_1(&row_blocks[b],
                                     xq + (size_t)b * 8, iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();
    if (warp != 0 || lane != 0) return;

    sum = partial[0] + partial[1] + partial[2] + partial[3];
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

static __global__ void q4k_dot_matvec_split_k_rope_cache_kernel(
    float *out0, void *key_cache, const BnBlockQ4K *blocks,
    const BnCudaBlockQ8_1 *xq, const float *bias0, const float *bias1,
    const float *freq, int total_rows, int cols, int split0,
    int n_kv_heads, int head_size, int pos, int rope_dims,
    size_t key_offset, int kv_f16) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int unit = blockIdx.x * warps_per_block + warp;
    int half_rope = rope_dims / 2;
    int k_pair_units = n_kv_heads * half_rope;
    if (unit >= split0 + k_pair_units) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);

    if (unit < split0) {
        float sum = 0.0f;
        const BnBlockQ4K *row_blocks = blocks + (size_t)unit * n_bpr;
        for (int b = kbx; b < n_bpr; b += 2)
            sum += cuda_vec_dot_q4k_q8_1(&row_blocks[b],
                                         xq + (size_t)b * 8, iqs);
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) {
            if (bias0) sum += bias0[unit];
            out0[unit] = sum;
        }
        return;
    }

    int pair = unit - split0;
    int head = pair / half_rope;
    int dim0 = pair - head * half_rope;
    int dim1 = dim0 + half_rope;
    int local0 = head * head_size + dim0;
    int local1 = head * head_size + dim1;
    int row0 = split0 + local0;
    int row1 = split0 + local1;
    if (row1 >= total_rows) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const BnBlockQ4K *row_blocks0 = blocks + (size_t)row0 * n_bpr;
    const BnBlockQ4K *row_blocks1 = blocks + (size_t)row1 * n_bpr;
    for (int b = kbx; b < n_bpr; b += 2) {
        sum0 += cuda_vec_dot_q4k_q8_1(&row_blocks0[b],
                                      xq + (size_t)b * 8, iqs);
        sum1 += cuda_vec_dot_q4k_q8_1(&row_blocks1[b],
                                      xq + (size_t)b * 8, iqs);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }
    if (lane == 0) {
        if (bias1) {
            sum0 += bias1[local0];
            sum1 += bias1[local1];
        }
        float s, c;
        __sincosf((float)pos * freq[dim0], &s, &c);
        float y0 = sum0 * c - sum1 * s;
        float y1 = sum0 * s + sum1 * c;
        cuda_kv_store(key_cache, key_offset + (size_t)local0, y0, kv_f16);
        cuda_kv_store(key_cache, key_offset + (size_t)local1, y1, kv_f16);
    }
}

static __global__ void q4k_dot_matvec_split_qk_rope_cache_kernel(
    float *out0, void *key_cache, const BnBlockQ4K *blocks,
    const BnCudaBlockQ8_1 *xq, const float *bias0, const float *bias1,
    const float *freq, int cols, int split0, int n_heads, int n_kv_heads,
    int head_size, int pos, int rope_dims, size_t key_offset, int kv_f16) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int unit = blockIdx.x * warps_per_block + warp;
    int half_rope = rope_dims / 2;
    int q_pair_units = n_heads * half_rope;
    int k_pair_units = n_kv_heads * half_rope;
    if (unit >= q_pair_units + k_pair_units) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    int is_q = unit < q_pair_units;
    int pair = is_q ? unit : unit - q_pair_units;
    int head = pair / half_rope;
    int dim0 = pair - head * half_rope;
    int dim1 = dim0 + half_rope;
    int local0 = head * head_size + dim0;
    int local1 = head * head_size + dim1;
    int row0 = is_q ? local0 : split0 + local0;
    int row1 = is_q ? local1 : split0 + local1;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const BnBlockQ4K *row_blocks0 = blocks + (size_t)row0 * n_bpr;
    const BnBlockQ4K *row_blocks1 = blocks + (size_t)row1 * n_bpr;
    for (int b = kbx; b < n_bpr; b += 2) {
        sum0 += cuda_vec_dot_q4k_q8_1(&row_blocks0[b],
                                      xq + (size_t)b * 8, iqs);
        sum1 += cuda_vec_dot_q4k_q8_1(&row_blocks1[b],
                                      xq + (size_t)b * 8, iqs);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }
    if (lane == 0) {
        if (is_q) {
            if (bias0) {
                sum0 += bias0[local0];
                sum1 += bias0[local1];
            }
        } else if (bias1) {
            sum0 += bias1[local0];
            sum1 += bias1[local1];
        }
        float s, c;
        __sincosf((float)pos * freq[dim0], &s, &c);
        float y0 = sum0 * c - sum1 * s;
        float y1 = sum0 * s + sum1 * c;
        if (is_q) {
            out0[local0] = y0;
            out0[local1] = y1;
        } else {
            cuda_kv_store(key_cache, key_offset + (size_t)local0, y0,
                         kv_f16);
            cuda_kv_store(key_cache, key_offset + (size_t)local1, y1,
                         kv_f16);
        }
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

static __global__ void q8_0_matvec_split_warp8_kernel(
    float *out0, float *out1, float *out2, const BnBlockQ8_0 *blocks,
    const float *x, const float *bias0, int total_rows, int cols,
    int split0, int split1, size_t out1_offset, size_t out2_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= total_rows) return;

    int n_bpr = cols / 32;
    const BnBlockQ8_0 *row_blocks = blocks + (size_t)row * n_bpr;
    float sum = 0.0f;
    for (int b = 0; b < n_bpr; b++) {
        const BnBlockQ8_0 *blk = &row_blocks[b];
        float xv = x[(size_t)b * 32 + lane];
        sum += cuda_fp16_to_fp32(blk->d) * (float)blk->qs[lane] * xv;
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

static __global__ void q5k_dot_matvec_split_4warp_kernel(
    float *out0, float *out1, float *out2, const BnBlockQ5K *blocks,
    const BnCudaBlockQ8_1 *xq, const float *bias0, int total_rows, int cols,
    int split0, int split1, size_t out1_offset, size_t out2_offset) {
    __shared__ float partial[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= total_rows || warp >= 4) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float sum = 0.0f;
    const BnBlockQ5K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = warp * 2 + kbx; b < n_bpr; b += 8)
        sum += cuda_vec_dot_q5k_q8_1(&row_blocks[b], xq + (size_t)b * 8,
                                     iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();
    if (warp != 0 || lane != 0) return;

    sum = partial[0] + partial[1] + partial[2] + partial[3];
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
    int gate_rows, int up_rows, int cols, int exact_silu) {
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
        out[row] = cuda_silu_select(gate, exact_silu) * up;
}

static __global__ void q4k_q8k_dot_fused_gateup_silu_qwarp4_kernel(
    float *out, const BnBlockQ4K *blocks, const BnBlockQ8K *xq,
    int gate_rows, int up_rows, int cols, int exact_silu) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    if (row >= gate_rows || row >= up_rows) return;

    int n_bpr = cols / BN_QK_K;
    float gate = 0.0f;
    float up = 0.0f;
    const BnBlockQ4K *gate_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ4K *up_blocks =
        blocks + (size_t)(gate_rows + row) * n_bpr;
    for (int b = sublane; b < n_bpr; b += 8) {
        gate += cuda_vec_dot_q4k_q8k(&gate_blocks[b], xq + b);
        up += cuda_vec_dot_q4k_q8k(&up_blocks[b], xq + b);
    }

    unsigned mask = 0xffu << (lane_group * 8);
    gate += __shfl_down_sync(mask, gate, 4);
    up += __shfl_down_sync(mask, up, 4);
    gate += __shfl_down_sync(mask, gate, 2);
    up += __shfl_down_sync(mask, up, 2);
    gate += __shfl_down_sync(mask, gate, 1);
    up += __shfl_down_sync(mask, up, 1);
    if (sublane == 0)
        out[row] = cuda_silu_select(gate, exact_silu) * up;
}

static __global__ void q4k_dot_fused_gateup_silu_kernel(
    float *out, const BnBlockQ4K *blocks, const BnCudaBlockQ8_1 *xq,
    int gate_rows, int up_rows, int cols, int exact_silu) {
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
        out[row] = cuda_silu_select(gate, exact_silu) * up;
}

static __global__ void q4k_dot_fused_gateup_silu_4warp_kernel(
    float *out, const BnBlockQ4K *blocks, const BnCudaBlockQ8_1 *xq,
    int gate_rows, int up_rows, int cols, int exact_silu) {
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
        out[row] = cuda_silu_select(gate, exact_silu) * up;
    }
}

static __global__ void q4k_dot_fused_gateup_silu_2warp_kernel(
    float *out, const BnBlockQ4K *blocks, const BnCudaBlockQ8_1 *xq,
    int gate_rows, int up_rows, int cols, int exact_silu) {
    __shared__ float gate_partial[2];
    __shared__ float up_partial[2];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= gate_rows || row >= up_rows || warp >= 2) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float gate = 0.0f;
    float up = 0.0f;
    const BnBlockQ4K *gate_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ4K *up_blocks =
        blocks + (size_t)(gate_rows + row) * n_bpr;
    for (int b = warp * 2 + kbx; b < n_bpr; b += 4) {
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
        gate = gate_partial[0] + gate_partial[1];
        up = up_partial[0] + up_partial[1];
        out[row] = cuda_silu_select(gate, exact_silu) * up;
    }
}

static __global__ void q4k_dot_fused_gateup_silu_5warp_kernel(
    float *out, const BnBlockQ4K *blocks, const BnCudaBlockQ8_1 *xq,
    int gate_rows, int up_rows, int cols, int exact_silu) {
    __shared__ float gate_partial[5];
    __shared__ float up_partial[5];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= gate_rows || row >= up_rows || warp >= 5) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    float gate = 0.0f;
    float up = 0.0f;
    const BnBlockQ4K *gate_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ4K *up_blocks =
        blocks + (size_t)(gate_rows + row) * n_bpr;
    for (int b = warp * 2 + kbx; b < n_bpr; b += 10) {
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
        gate = gate_partial[0] + gate_partial[1] + gate_partial[2] +
               gate_partial[3] + gate_partial[4];
        up = up_partial[0] + up_partial[1] + up_partial[2] +
             up_partial[3] + up_partial[4];
        out[row] = cuda_silu_select(gate, exact_silu) * up;
    }
}

static __global__ void q4k_dot_fused_gateup_silu_batch4_token_kernel(
        float *out, const BnBlockQ4K *blocks, const BnCudaBlockQ8_1 *xq,
        int gate_rows, int up_rows, int cols, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token0 = blockIdx.y * 4;
    if (row >= gate_rows || row >= up_rows || token0 >= n_tokens) return;

    int n_bpr = cols / BN_QK_K;
    int x_blocks = (cols + 31) / 32;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    int have1 = token0 + 1 < n_tokens;
    int have2 = token0 + 2 < n_tokens;
    int have3 = token0 + 3 < n_tokens;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    const BnBlockQ4K *gate_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ4K *up_blocks =
        blocks + (size_t)(gate_rows + row) * n_bpr;
    const BnCudaBlockQ8_1 *xq0 = xq + (size_t)token0 * x_blocks;
    const BnCudaBlockQ8_1 *xq1 = xq0 + x_blocks;
    const BnCudaBlockQ8_1 *xq2 = xq1 + x_blocks;
    const BnCudaBlockQ8_1 *xq3 = xq2 + x_blocks;
    for (int b = kbx; b < n_bpr; b += 2) {
        const BnBlockQ4K *gb = &gate_blocks[b];
        const BnBlockQ4K *ub = &up_blocks[b];
        g0 += cuda_vec_dot_q4k_q8_1(gb, xq0 + (size_t)b * 8, iqs);
        u0 += cuda_vec_dot_q4k_q8_1(ub, xq0 + (size_t)b * 8, iqs);
        if (have1) {
            g1 += cuda_vec_dot_q4k_q8_1(gb, xq1 + (size_t)b * 8, iqs);
            u1 += cuda_vec_dot_q4k_q8_1(ub, xq1 + (size_t)b * 8, iqs);
        }
        if (have2) {
            g2 += cuda_vec_dot_q4k_q8_1(gb, xq2 + (size_t)b * 8, iqs);
            u2 += cuda_vec_dot_q4k_q8_1(ub, xq2 + (size_t)b * 8, iqs);
        }
        if (have3) {
            g3 += cuda_vec_dot_q4k_q8_1(gb, xq3 + (size_t)b * 8, iqs);
            u3 += cuda_vec_dot_q4k_q8_1(ub, xq3 + (size_t)b * 8, iqs);
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        g0 += __shfl_down_sync(0xffffffffu, g0, offset);
        u0 += __shfl_down_sync(0xffffffffu, u0, offset);
        g1 += __shfl_down_sync(0xffffffffu, g1, offset);
        u1 += __shfl_down_sync(0xffffffffu, u1, offset);
        g2 += __shfl_down_sync(0xffffffffu, g2, offset);
        u2 += __shfl_down_sync(0xffffffffu, u2, offset);
        g3 += __shfl_down_sync(0xffffffffu, g3, offset);
        u3 += __shfl_down_sync(0xffffffffu, u3, offset);
    }
    if (lane == 0) {
        out[(size_t)token0 * gate_rows + row] =
            (g0 / (1.0f + __expf(-g0))) * u0;
        if (have1)
            out[(size_t)(token0 + 1) * gate_rows + row] =
                (g1 / (1.0f + __expf(-g1))) * u1;
        if (have2)
            out[(size_t)(token0 + 2) * gate_rows + row] =
                (g2 / (1.0f + __expf(-g2))) * u2;
        if (have3)
            out[(size_t)(token0 + 3) * gate_rows + row] =
                (g3 / (1.0f + __expf(-g3))) * u3;
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

static __global__ void q5k_dot_fused_gateup_silu_2warp_kernel(
    float *out, const BnBlockQ5K *blocks, const BnCudaBlockQ8_1 *xq,
    int gate_rows, int up_rows, int cols) {
    __shared__ float partial[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = warp >> 1;
    int half = warp & 1;
    int groups_per_block = blockDim.x >> 6;
    int row = blockIdx.x * groups_per_block + group;
    if (row >= gate_rows || row >= up_rows) return;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    const BnBlockQ5K *row_blocks =
        blocks + (size_t)(half ? gate_rows + row : row) * n_bpr;
    float sum = 0.0f;
    for (int b = kbx; b < n_bpr; b += 2)
        sum += cuda_vec_dot_q5k_q8_1(&row_blocks[b],
                                     xq + (size_t)b * 8, iqs);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();

    if (half == 0 && lane == 0) {
        float gate = partial[warp];
        float up = partial[warp + 1];
        out[row] = (gate / (1.0f + __expf(-gate))) * up;
    }
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

static __global__ void matvec_kernel(void *out, const void *wdata,
                                     const float *x, const float *bias,
                                     int rows, int cols, int type,
                                     size_t out_offset, int out_f16) {
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
    } else if (type == BN_GGUF_TENSOR_BF16) {
        const uint16_t *w = (const uint16_t *)wdata + (size_t)row * cols;
        for (int c = tid; c < cols; c += blockDim.x)
            sum += cuda_bf16_to_fp32(w[c]) * x_token[c];
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
    } else if (type == BN_GGUF_TENSOR_Q3_K) {
        const BnBlockQ3K *blocks = (const BnBlockQ3K *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            const BnBlockQ3K *blk = &blocks[(size_t)row * n_bpr + b];
            sum += cuda_q3k_value(blk, i) * x_token[c];
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
    } else if (type == BN_GGUF_TENSOR_IQ4_XS) {
        const BnBlockIQ4XS *blocks = (const BnBlockIQ4XS *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            int group = i >> 5;
            int in_group = i & 31;
            const BnBlockIQ4XS *blk =
                &blocks[(size_t)row * n_bpr + b];
            int lo = (blk->scales_l[group >> 1] >>
                      ((group & 1) * 4)) & 0xF;
            int hi = (blk->scales_h >> (group * 2)) & 3;
            float dl = cuda_fp16_to_fp32(blk->d) *
                       (float)((lo | (hi << 4)) - 32);
            uint8_t packed = blk->qs[group * 16 + (in_group & 15)];
            int q = in_group < 16 ? (packed & 0xF) : (packed >> 4);
            sum += dl * (float)bn_kvalues_iq4nl[q] * x_token[c];
        }
    } else if (type == BN_GGUF_TENSOR_IQ3_XXS) {
        const BnBlockIQ3XXS *blocks = (const BnBlockIQ3XXS *)wdata;
        int n_bpr = cols / BN_QK_K;
        for (int c = tid; c < cols; c += blockDim.x) {
            int b = c / BN_QK_K;
            int i = c & (BN_QK_K - 1);
            int ib32 = i >> 5;
            int in32 = i & 31;
            int l = in32 >> 3;
            int j = in32 & 7;
            const BnBlockIQ3XXS *blk =
                &blocks[(size_t)row * n_bpr + b];
            const uint8_t *qs = blk->qs + ib32 * 8;
            const uint8_t *scales_and_signs = blk->qs + BN_QK_K / 4;
            const uint8_t *auxp = scales_and_signs + 4 * ib32;
            uint32_t aux32 = (uint32_t)auxp[0] |
                             ((uint32_t)auxp[1] << 8) |
                             ((uint32_t)auxp[2] << 16) |
                             ((uint32_t)auxp[3] << 24);
            float db = cuda_fp16_to_fp32(blk->d) *
                       (0.5f + (float)(aux32 >> 28)) * 0.5f;
            uint8_t signs =
                bn_ksigns_iq2xs[(aux32 >> (7 * l)) & 0x7Fu];
            uint32_t gridv = bn_iq3xxs_grid[qs[2 * l + (j >= 4)]];
            uint8_t gv = (uint8_t)(gridv >> ((j & 3) * 8));
            float w = (float)gv;
            if (signs & bn_kmask_iq2xs[j])
                w = -w;
            sum += db * w * x_token[c];
        }
    }

    extern __shared__ float scratch[];
    sum = cuda_block_reduce_sum(sum, scratch);
    if (tid == 0) {
        float v = sum;
        if (bias) v += bias[row];
        cuda_kv_store(out, out_offset + (size_t)token * rows + row, v,
                      out_f16);
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

static __global__ void q8_0_q8f32_matmul4_warp_kernel(
        float *out, const BnBlockQ8_0 *blocks,
        const BnCudaBlockQ8_0F32 *xq, int rows, int cols, int n_tokens,
        size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 4;
    int sublane = lane & 15;
    int row = (blockIdx.x * warps_per_block + warp) * 2 + lane_group;
    int token = blockIdx.y;
    if (row >= rows || token >= n_tokens) return;

    int n_bpr = cols / 32;
    const BnBlockQ8_0 *row_blocks = blocks + (size_t)row * n_bpr;
    const BnCudaBlockQ8_0F32 *xq_token = xq + (size_t)token * n_bpr;
    float sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 16) {
        const BnBlockQ8_0 *blk = &row_blocks[b];
        int dot = cuda_dot_i8x32_dp4a(blk->qs, xq_token[b].qs);
        sum += cuda_fp16_to_fp32(blk->d) * xq_token[b].d * (float)dot;
    }

    unsigned mask = 0xffffu << (lane_group * 16);
    for (int offset = 8; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(mask, sum, offset);
    if (sublane == 0)
        out[out_offset + (size_t)token * rows + (size_t)row] = sum;
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

static __device__ __forceinline__ int cuda_q6k_dot_16_centered(
    const uint8_t *ql, const uint8_t *qh, const int8_t *q8, int chunk,
    int group) {
    int acc = 0;
    int sub = group >> 1;
    int half = group & 1;
    int base = half * 16;
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
            qv |= ((q - 32) & 0xff) << (8 * j);
            xv |= ((int)(uint8_t)q8[base + i + j]) << (8 * j);
        }
        acc = cuda_dp4a_i32(qv, xv, acc);
    }
    return acc;
}

static __device__ __forceinline__ int cuda_get_i32_b2(const void *p,
                                                       int i32) {
    const uint16_t *x = (const uint16_t *)p;
    int v = (int)x[2 * i32 + 0];
    v |= (int)x[2 * i32 + 1] << 16;
    return v;
}

static __device__ __forceinline__ int cuda_get_i32_b4(const void *p,
                                                       int i32) {
    return ((const int *)p)[i32];
}

static __device__ __forceinline__ float cuda_vec_dot_q6k_q8_1_mmvq(
    const BnBlockQ6K *blk, const BnCudaBlockQ8_1 *xq, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);

    const int vl = cuda_get_i32_b2(blk->ql, iqs);
    const int vh = cuda_get_i32_b2(blk->qh, 8 * (iqs / 16) + iqs % 8) >>
                   vh_shift;
    const int8_t *scales = blk->scales + scale_offset;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int sc = (int)scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        int vi = vil | vih;
        vi = __vsubss4(vi, 0x20202020);

        const BnCudaBlockQ8_1 *xb = xq + bq8_offset + 2 * i;
        const int u = cuda_get_i32_b4(xb->qs, iqs & 7);
        const float d8 = cuda_fp16_to_fp32(xb->d);
        sum += d8 * (float)(cuda_dp4a_i32(vi, u, 0) * sc);
    }

    return cuda_fp16_to_fp32(blk->d) * sum;
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

static __device__ __forceinline__ float cuda_q6k_q8k_part(
    const BnBlockQ6K *blk, const BnBlockQ8K *xq, int part) {
    int chunk = part >> 3;
    int group = part & 7;
    int scale_idx = chunk * 8 + group;
    int s = (int)blk->scales[scale_idx];
    int dot = cuda_q6k_dot_16(blk->ql, blk->qh, xq->qs, chunk, group);
    int corr = s * (int)xq->bsums[scale_idx];
    return cuda_fp16_to_fp32(blk->d) * xq->d *
           (float)(s * dot - (corr << 5));
}

static __device__ __forceinline__ float cuda_vec_dot_q6k_q8_1(
    const BnBlockQ6K *blk, const BnCudaBlockQ8_1 *xq) {
    float sum = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < 2; chunk++) {
#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int group0 = sub * 2 + 0;
            int group1 = sub * 2 + 1;
            int scale_idx0 = chunk * 8 + group0;
            int scale_idx1 = chunk * 8 + group1;
            int q8_idx = chunk * 4 + sub;
            const float xd = cuda_fp16_to_fp32(xq[q8_idx].d);
            int dot0 = cuda_q6k_dot_16_centered(blk->ql, blk->qh,
                                                xq[q8_idx].qs, chunk,
                                                group0);
            int dot1 = cuda_q6k_dot_16_centered(blk->ql, blk->qh,
                                                xq[q8_idx].qs, chunk,
                                                group1);
            sum += xd * ((float)blk->scales[scale_idx0] * (float)dot0 +
                         (float)blk->scales[scale_idx1] * (float)dot1);
        }
    }
    return cuda_fp16_to_fp32(blk->d) * sum;
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

static __global__ void q6k_dot_matvec_q8_1_kernel(float *out,
                                                  const BnBlockQ6K *blocks,
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
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = lane; b < n_bpr; b += 32)
        sum += cuda_vec_dot_q6k_q8_1(&row_blocks[b], xq + (size_t)b * 8);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) {
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q6k_dot_matvec_mmvq_kernel(float *out,
                                                  const BnBlockQ6K *blocks,
                                                  const BnCudaBlockQ8_1 *xq,
                                                  const float *bias,
                                                  int rows, int cols,
                                                  size_t out_offset) {
    __shared__ float partial[3][32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows || tid >= 128) return;

    int n_bpr = cols / BN_QK_K;
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = tid / 32; b < n_bpr; b += 4) {
        int iqs = tid & 31;
        sum += cuda_vec_dot_q6k_q8_1_mmvq(&row_blocks[b],
                                          xq + (size_t)b * 8, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (warp > 0 && lane == 0)
        partial[warp - 1][0] = sum;
    __syncthreads();

    if (warp == 0 && lane == 0) {
        sum += partial[0][0] + partial[1][0] + partial[2][0];
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q6k_dot_matvec_mmvq_2warp_kernel(float *out,
                                                        const BnBlockQ6K *blocks,
                                                        const BnCudaBlockQ8_1 *xq,
                                                        const float *bias,
                                                        int rows, int cols,
                                                        size_t out_offset) {
    __shared__ float partial[1];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows || tid >= 64) return;

    int n_bpr = cols / BN_QK_K;
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = tid / 32; b < n_bpr; b += 2) {
        int iqs = tid & 31;
        sum += cuda_vec_dot_q6k_q8_1_mmvq(&row_blocks[b],
                                          xq + (size_t)b * 8, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (warp == 1 && lane == 0)
        partial[0] = sum;
    __syncthreads();

    if (warp == 0 && lane == 0) {
        sum += partial[0];
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q6k_dot_matvec4_kernel(float *out,
                                              const BnBlockQ6K *blocks,
                                              const BnBlockQ8K *xq,
                                              const float *bias,
                                              int rows, int cols,
                                              size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    if (row >= rows) return;

    int n_bpr = cols / BN_QK_K;
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = sublane; b < n_bpr; b += 8)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], xq + b);

    unsigned mask = 0xffu << (lane_group * 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0) {
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q6k_dot_matvec_4warp_kernel(float *out,
                                                   const BnBlockQ6K *blocks,
                                                   const BnBlockQ8K *xq,
                                                   const float *bias,
                                                   int rows, int cols,
                                                   size_t out_offset) {
    __shared__ float partial[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= rows || warp >= 4) return;

    int n_bpr = cols / BN_QK_K;
    int tid = warp * 32 + lane;
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = tid; b < n_bpr; b += 128)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], xq + b);

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

static __global__ void q6k_dot_matvec_3warp_kernel(float *out,
                                                   const BnBlockQ6K *blocks,
                                                   const BnBlockQ8K *xq,
                                                   const float *bias,
                                                   int rows, int cols,
                                                   size_t out_offset) {
    __shared__ float partial[3];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= rows || warp >= 3) return;

    int n_bpr = cols / BN_QK_K;
    int tid = warp * 32 + lane;
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = tid; b < n_bpr; b += 96)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], xq + b);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();

    if (warp == 0 && lane == 0) {
        sum = partial[0] + partial[1] + partial[2];
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q6k_dot_matvec_5warp_kernel(float *out,
                                                   const BnBlockQ6K *blocks,
                                                   const BnBlockQ8K *xq,
                                                   const float *bias,
                                                   int rows, int cols,
                                                   size_t out_offset) {
    __shared__ float partial[5];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= rows || warp >= 5) return;

    int n_bpr = cols / BN_QK_K;
    int tid = warp * 32 + lane;
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = tid; b < n_bpr; b += 160)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], xq + b);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();

    if (warp == 0 && lane == 0) {
        sum = partial[0] + partial[1] + partial[2] + partial[3] +
              partial[4];
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __global__ void q6k_dot_matvec_4warp_residual_kernel(
                                                   float *x,
                                                   const BnBlockQ6K *blocks,
                                                   const BnBlockQ8K *xq,
                                                   const float *residual,
                                                   int rows, int cols) {
    __shared__ float partial[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= rows || warp >= 4) return;

    int n_bpr = cols / BN_QK_K;
    int tid = warp * 32 + lane;
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = tid; b < n_bpr; b += 128)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], xq + b);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();

    if (warp == 0 && lane == 0)
        x[row] = residual[row] + partial[0] + partial[1] +
                 partial[2] + partial[3];
}

static __global__ void q6k_dot_matvec_2warp_kernel(float *out,
                                                   const BnBlockQ6K *blocks,
                                                   const BnBlockQ8K *xq,
                                                   const float *bias,
                                                   int rows, int cols,
                                                   size_t out_offset) {
    __shared__ float partial[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = warp >> 1;
    int warp_in_group = warp & 1;
    int row = blockIdx.x * 4 + group;
    if (row >= rows || warp >= 8) return;

    int n_bpr = cols / BN_QK_K;
    int tid = warp_in_group * 32 + lane;
    float sum = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    for (int b = tid; b < n_bpr; b += 64)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], xq + b);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[group * 2 + warp_in_group] = sum;
    __syncthreads();

    if (warp_in_group == 0 && lane == 0) {
        sum = partial[group * 2] + partial[group * 2 + 1];
        if (bias) sum += bias[row];
        out[out_offset + row] = sum;
    }
}

static __device__ int argmax_penalty_contains(const int *penalty_tokens,
                                              int n_penalty_tokens,
                                              int token);

static __global__ void q6k_dot_argmax32_kernel(
    BnCudaArgmaxPair *partials, const BnBlockQ6K *blocks,
    const BnBlockQ8K *xq, const int *penalty_tokens, int n_penalty_tokens,
    float repeat_penalty, int rows, int cols) {
    __shared__ BnCudaArgmaxPair row_best[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = blockIdx.x * 32 + warp * 4 + lane_group;

    BnCudaArgmaxPair best;
    best.value = -INFINITY;
    best.index = row < rows ? row : 0;
    if (row < rows) {
        int n_bpr = cols / BN_QK_K;
        float sum = 0.0f;
        const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
        for (int b = sublane; b < n_bpr; b += 8)
            sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], xq + b);

        unsigned mask = 0xffu << (lane_group * 8);
        sum += __shfl_down_sync(mask, sum, 4);
        sum += __shfl_down_sync(mask, sum, 2);
        sum += __shfl_down_sync(mask, sum, 1);
        if (sublane == 0) {
            if (repeat_penalty != 1.0f && penalty_tokens &&
                n_penalty_tokens > 0) {
                if (argmax_penalty_contains(penalty_tokens,
                                            n_penalty_tokens, row))
                    sum = sum > 0.0f ? sum / repeat_penalty
                                     : sum * repeat_penalty;
            }
            best.value = sum;
            best.index = row;
        }
    }
    if (sublane == 0)
        row_best[warp * 4 + lane_group] = best;
    __syncthreads();

    if (threadIdx.x < 32) {
        BnCudaArgmaxPair b;
        b.value = threadIdx.x < 32 ? row_best[threadIdx.x].value : -INFINITY;
        b.index = threadIdx.x < 32 ? row_best[threadIdx.x].index : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            BnCudaArgmaxPair other;
            other.value = __shfl_down_sync(0xffffffffu, b.value, offset);
            other.index = __shfl_down_sync(0xffffffffu, b.index, offset);
            if (other.value > b.value ||
                (other.value == b.value && other.index < b.index))
                b = other;
        }
        if (threadIdx.x == 0)
            partials[blockIdx.x] = b;
    }
}

static __global__ void q6k_dot_argmax_mmvq_2warp4_kernel(
    BnCudaArgmaxPair *partials, const BnBlockQ6K *blocks,
    const BnCudaBlockQ8_1 *xq, const int *penalty_tokens,
    int n_penalty_tokens, float repeat_penalty, int rows, int cols) {
    __shared__ float row_partial[4][2];
    __shared__ BnCudaArgmaxPair row_best[4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = warp >> 1;
    int warp_in_group = warp & 1;
    int row = blockIdx.x * 4 + group;

    if (warp < 8 && group < 4) {
        float sum = 0.0f;
        if (row < rows) {
            int n_bpr = cols / BN_QK_K;
            const BnBlockQ6K *row_blocks =
                blocks + (size_t)row * (size_t)n_bpr;
            for (int b = warp_in_group; b < n_bpr; b += 2)
                sum += cuda_vec_dot_q6k_q8_1_mmvq(
                    &row_blocks[b], xq + (size_t)b * 8, lane);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0)
            row_partial[group][warp_in_group] = sum;
    }
    __syncthreads();

    if (threadIdx.x < 4) {
        int r = blockIdx.x * 4 + threadIdx.x;
        BnCudaArgmaxPair b;
        b.value = -INFINITY;
        b.index = r < rows ? r : 0;
        if (r < rows) {
            float v = row_partial[threadIdx.x][0] +
                      row_partial[threadIdx.x][1];
            if (repeat_penalty != 1.0f && penalty_tokens &&
                n_penalty_tokens > 0) {
                if (argmax_penalty_contains(penalty_tokens,
                                            n_penalty_tokens, r))
                    v = v > 0.0f ? v / repeat_penalty
                                  : v * repeat_penalty;
            }
            b.value = v;
            b.index = r;
        }
        row_best[threadIdx.x] = b;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        BnCudaArgmaxPair b = row_best[0];
        for (int i = 1; i < 4; i++) {
            BnCudaArgmaxPair other = row_best[i];
            if (other.value > b.value ||
                (other.value == b.value && other.index < b.index))
                b = other;
        }
        partials[blockIdx.x] = b;
    }
}

static __global__ void q6k_dot_argmax_mmvq_1warp8_kernel(
    BnCudaArgmaxPair *partials, const BnBlockQ6K *blocks,
    const BnCudaBlockQ8_1 *xq, const int *penalty_tokens,
    int n_penalty_tokens, float repeat_penalty, int rows, int cols) {
    __shared__ BnCudaArgmaxPair row_best[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * 8 + warp;

    BnCudaArgmaxPair b;
    b.value = -INFINITY;
    b.index = row < rows ? row : 0;
    if (warp < 8 && row < rows) {
        int n_bpr = cols / BN_QK_K;
        const BnBlockQ6K *row_blocks =
            blocks + (size_t)row * (size_t)n_bpr;
        float sum = 0.0f;
        for (int blk = 0; blk < n_bpr; blk++)
            sum += cuda_vec_dot_q6k_q8_1_mmvq(
                &row_blocks[blk], xq + (size_t)blk * 8, lane);
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) {
            if (repeat_penalty != 1.0f && penalty_tokens &&
                n_penalty_tokens > 0) {
                if (argmax_penalty_contains(penalty_tokens,
                                            n_penalty_tokens, row))
                    sum = sum > 0.0f ? sum / repeat_penalty
                                     : sum * repeat_penalty;
            }
            b.value = sum;
            b.index = row;
        }
    }
    if (lane == 0)
        row_best[warp] = b;
    __syncthreads();

    if (threadIdx.x == 0) {
        BnCudaArgmaxPair best = row_best[0];
        for (int i = 1; i < 8; i++) {
            BnCudaArgmaxPair other = row_best[i];
            if (other.value > best.value ||
                (other.value == best.value && other.index < best.index))
                best = other;
        }
        partials[blockIdx.x] = best;
    }
}

static __global__ void q6k_dot_argmax_mmvq_1warp8_1536_kernel(
    BnCudaArgmaxPair *partials, const BnBlockQ6K *blocks,
    const BnCudaBlockQ8_1 *xq, const int *penalty_tokens,
    int n_penalty_tokens, float repeat_penalty, int rows) {
    __shared__ BnCudaArgmaxPair row_best[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * 8 + warp;

    BnCudaArgmaxPair b;
    b.value = -INFINITY;
    b.index = row < rows ? row : 0;
    if (warp < 8 && row < rows) {
        const BnBlockQ6K *row_blocks = blocks + (size_t)row * 6;
        float sum = 0.0f;
#pragma unroll
        for (int blk = 0; blk < 6; blk++)
            sum += cuda_vec_dot_q6k_q8_1_mmvq(
                &row_blocks[blk], xq + (size_t)blk * 8, lane);
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) {
            if (repeat_penalty != 1.0f && penalty_tokens &&
                n_penalty_tokens > 0) {
                if (argmax_penalty_contains(penalty_tokens,
                                            n_penalty_tokens, row))
                    sum = sum > 0.0f ? sum / repeat_penalty
                                     : sum * repeat_penalty;
            }
            b.value = sum;
            b.index = row;
        }
    }
    if (lane == 0)
        row_best[warp] = b;
    __syncthreads();

    if (threadIdx.x == 0) {
        BnCudaArgmaxPair best = row_best[0];
        for (int i = 1; i < 8; i++) {
            BnCudaArgmaxPair other = row_best[i];
            if (other.value > best.value ||
                (other.value == best.value && other.index < best.index))
                best = other;
        }
        partials[blockIdx.x] = best;
    }
}

static __global__ void q6k_dot_argmax_mmvq_1warp16_kernel(
    BnCudaArgmaxPair *partials, const BnBlockQ6K *blocks,
    const BnCudaBlockQ8_1 *xq, const int *penalty_tokens,
    int n_penalty_tokens, float repeat_penalty, int rows, int cols) {
    __shared__ BnCudaArgmaxPair row_best[16];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * 16 + warp;

    BnCudaArgmaxPair b;
    b.value = -INFINITY;
    b.index = row < rows ? row : 0;
    if (warp < 16 && row < rows) {
        int n_bpr = cols / BN_QK_K;
        const BnBlockQ6K *row_blocks =
            blocks + (size_t)row * (size_t)n_bpr;
        float sum = 0.0f;
        for (int blk = 0; blk < n_bpr; blk++)
            sum += cuda_vec_dot_q6k_q8_1_mmvq(
                &row_blocks[blk], xq + (size_t)blk * 8, lane);
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) {
            if (repeat_penalty != 1.0f && penalty_tokens &&
                n_penalty_tokens > 0) {
                if (argmax_penalty_contains(penalty_tokens,
                                            n_penalty_tokens, row))
                    sum = sum > 0.0f ? sum / repeat_penalty
                                     : sum * repeat_penalty;
            }
            b.value = sum;
            b.index = row;
        }
    }
    if (lane == 0)
        row_best[warp] = b;
    __syncthreads();

    if (threadIdx.x == 0) {
        BnCudaArgmaxPair best = row_best[0];
        for (int i = 1; i < 16; i++) {
            BnCudaArgmaxPair other = row_best[i];
            if (other.value > best.value ||
                (other.value == best.value && other.index < best.index))
                best = other;
        }
        partials[blockIdx.x] = best;
    }
}

static __global__ void f32_matvec_warp_kernel(
    float *out, const float *w, const float *x, const float *bias,
    int rows, int cols, size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;

    float sum = 0.0f;
    const float *wr = w + (size_t)row * (size_t)cols;
    for (int c = lane; c < cols; c += 32)
        sum += wr[c] * x[c];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[out_offset + (size_t)row] = sum + (bias ? bias[row] : 0.0f);
}

static __global__ void f16_matvec_warp_kernel(
    float *out, const __half *w, const float *x, const float *bias,
    int rows, int cols, size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;

    float sum = 0.0f;
    const __half *wr = w + (size_t)row * (size_t)cols;
    for (int c = lane; c < cols; c += 32)
        sum += __half2float(wr[c]) * x[c];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[out_offset + (size_t)row] = sum + (bias ? bias[row] : 0.0f);
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

static __global__ void q6k_dot_matmul4_token_kernel(
        float *out, const BnBlockQ6K *blocks, const BnBlockQ8K *xq,
        int rows, int cols, int n_tokens, size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token0 = blockIdx.y * 4;
    if (row >= rows || token0 >= n_tokens) return;

    int n_bpr = cols / BN_QK_K;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ8K *xq0 = xq + (size_t)token0 * n_bpr;
    const BnBlockQ8K *xq1 = xq0 + n_bpr;
    const BnBlockQ8K *xq2 = xq1 + n_bpr;
    const BnBlockQ8K *xq3 = xq2 + n_bpr;
    int have1 = token0 + 1 < n_tokens;
    int have2 = token0 + 2 < n_tokens;
    int have3 = token0 + 3 < n_tokens;
    for (int b = lane; b < n_bpr; b += 32) {
        const BnBlockQ6K *blk = &row_blocks[b];
        sum0 += cuda_vec_dot_q6k_q8k(blk, xq0 + b);
        if (have1) sum1 += cuda_vec_dot_q6k_q8k(blk, xq1 + b);
        if (have2) sum2 += cuda_vec_dot_q6k_q8k(blk, xq2 + b);
        if (have3) sum3 += cuda_vec_dot_q6k_q8k(blk, xq3 + b);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
    }
    if (lane == 0) {
        out[out_offset + (size_t)token0 * rows + row] = sum0;
        if (have1) out[out_offset + (size_t)(token0 + 1) * rows + row] = sum1;
        if (have2) out[out_offset + (size_t)(token0 + 2) * rows + row] = sum2;
        if (have3) out[out_offset + (size_t)(token0 + 3) * rows + row] = sum3;
    }
}

static __global__ void q6k_dot_matmul8_token_kernel(
        float *out, const BnBlockQ6K *blocks, const BnBlockQ8K *xq,
        int rows, int cols, int n_tokens, size_t out_offset) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    int token0 = blockIdx.y * 8;
    if (row >= rows || token0 >= n_tokens) return;

    int n_bpr = cols / BN_QK_K;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
    const BnBlockQ6K *row_blocks = blocks + (size_t)row * n_bpr;
    const BnBlockQ8K *xq0 = xq + (size_t)token0 * n_bpr;
    const BnBlockQ8K *xq1 = xq0 + n_bpr;
    const BnBlockQ8K *xq2 = xq1 + n_bpr;
    const BnBlockQ8K *xq3 = xq2 + n_bpr;
    const BnBlockQ8K *xq4 = xq3 + n_bpr;
    const BnBlockQ8K *xq5 = xq4 + n_bpr;
    const BnBlockQ8K *xq6 = xq5 + n_bpr;
    const BnBlockQ8K *xq7 = xq6 + n_bpr;
    int have1 = token0 + 1 < n_tokens;
    int have2 = token0 + 2 < n_tokens;
    int have3 = token0 + 3 < n_tokens;
    int have4 = token0 + 4 < n_tokens;
    int have5 = token0 + 5 < n_tokens;
    int have6 = token0 + 6 < n_tokens;
    int have7 = token0 + 7 < n_tokens;
    for (int b = lane; b < n_bpr; b += 32) {
        const BnBlockQ6K *blk = &row_blocks[b];
        sum0 += cuda_vec_dot_q6k_q8k(blk, xq0 + b);
        if (have1) sum1 += cuda_vec_dot_q6k_q8k(blk, xq1 + b);
        if (have2) sum2 += cuda_vec_dot_q6k_q8k(blk, xq2 + b);
        if (have3) sum3 += cuda_vec_dot_q6k_q8k(blk, xq3 + b);
        if (have4) sum4 += cuda_vec_dot_q6k_q8k(blk, xq4 + b);
        if (have5) sum5 += cuda_vec_dot_q6k_q8k(blk, xq5 + b);
        if (have6) sum6 += cuda_vec_dot_q6k_q8k(blk, xq6 + b);
        if (have7) sum7 += cuda_vec_dot_q6k_q8k(blk, xq7 + b);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
        sum4 += __shfl_down_sync(0xffffffffu, sum4, offset);
        sum5 += __shfl_down_sync(0xffffffffu, sum5, offset);
        sum6 += __shfl_down_sync(0xffffffffu, sum6, offset);
        sum7 += __shfl_down_sync(0xffffffffu, sum7, offset);
    }
    if (lane == 0) {
        out[out_offset + (size_t)token0 * rows + row] = sum0;
        if (have1) out[out_offset + (size_t)(token0 + 1) * rows + row] = sum1;
        if (have2) out[out_offset + (size_t)(token0 + 2) * rows + row] = sum2;
        if (have3) out[out_offset + (size_t)(token0 + 3) * rows + row] = sum3;
        if (have4) out[out_offset + (size_t)(token0 + 4) * rows + row] = sum4;
        if (have5) out[out_offset + (size_t)(token0 + 5) * rows + row] = sum5;
        if (have6) out[out_offset + (size_t)(token0 + 6) * rows + row] = sum6;
        if (have7) out[out_offset + (size_t)(token0 + 7) * rows + row] = sum7;
    }
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
    int up_rows, int cols, int exact_silu) {
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
        out[row] = cuda_silu_select(gate, exact_silu) * up;
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

static __global__ void rmsnorm_batch_to_f16_kernel(
        float *out, __half *out_f16, const float *x, const float *weight,
        int n, int n_tokens, float eps) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= n_tokens || n <= 0) return;
    const float *xt = x + (size_t)t * n;
    float *ot = out + (size_t)t * n;
    __half *ht = out_f16 + (size_t)t * n;

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
    for (int i = tid; i < n; i += blockDim.x) {
        float v = xt[i] * scale * weight[i];
        ot[i] = v;
        ht[i] = __float2half_rn(v);
    }
}

static __global__ void rmsnorm_batch_copy_kernel(float *out, float *copy_out,
                                                 const float *x,
                                                 const float *weight, int n,
                                                 int n_tokens, float eps) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= n_tokens || n <= 0) return;
    const float *xt = x + (size_t)t * n;
    float *ot = out + (size_t)t * n;
    float *ct = copy_out + (size_t)t * n;

    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = xt[i];
        ct[i] = v;
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
        ot[i] = xt[i] * scale * weight[i];
}

static __global__ void rmsnorm_batch_copy_to_f16_kernel(
        float *out, __half *out_f16, float *copy_out, const float *x,
        const float *weight, int n, int n_tokens, float eps) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= n_tokens || n <= 0) return;
    const float *xt = x + (size_t)t * n;
    float *ot = out + (size_t)t * n;
    __half *ht = out_f16 + (size_t)t * n;
    float *ct = copy_out + (size_t)t * n;

    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = xt[i];
        ct[i] = v;
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
    for (int i = tid; i < n; i += blockDim.x) {
        float v = xt[i] * scale * weight[i];
        ot[i] = v;
        ht[i] = __float2half_rn(v);
    }
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
    float *q, void *k, const float *q_weight, const float *k_weight,
    const float *freq, int n_heads, int n_kv_heads, int head_size,
    float eps, int per_head_weight, size_t k_offset, int pos,
    int rope_dims, int kv_f16) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int is_q = h < n_heads;
    int idx = is_q ? h : h - n_heads;
    int n = is_q ? n_heads : n_kv_heads;
    if (idx >= n || head_size <= 0) return;

    float *qh = q + (size_t)idx * head_size;
    size_t k_base = k_offset + (size_t)idx * head_size;
    const float *weight = is_q ? q_weight : k_weight;
    const float *wh = weight + (per_head_weight ? (size_t)idx * head_size : 0);

    float ss = 0.0f;
    for (int i = tid; i < head_size; i += blockDim.x) {
        float xv = is_q ? qh[i] : cuda_kv_load(k, k_base + (size_t)i, kv_f16);
        ss += xv * xv;
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

    float scale = rsqrtf(scratch[0] / (float)head_size + eps);
    for (int i = tid; i < head_size; i += blockDim.x) {
        float xv = is_q ? qh[i] : cuda_kv_load(k, k_base + (size_t)i, kv_f16);
        xv = xv * scale * wh[i];
        if (is_q)
            qh[i] = xv;
        else
            cuda_kv_store(k, k_base + (size_t)i, xv, kv_f16);
    }
    __syncthreads();

    int half_rope = rope_dims / 2;
    for (int i = tid; i < half_rope; i += blockDim.x) {
        int j = i + half_rope;
        float angle = (float)pos * freq[i];
        float s, c;
        __sincosf(angle, &s, &c);
        float x0 = is_q ? qh[i] : cuda_kv_load(k, k_base + (size_t)i, kv_f16);
        float x1 = is_q ? qh[j] : cuda_kv_load(k, k_base + (size_t)j, kv_f16);
        float y0 = x0 * c - x1 * s;
        float y1 = x0 * s + x1 * c;
        if (is_q) {
            qh[i] = y0;
            qh[j] = y1;
        } else {
            cuda_kv_store(k, k_base + (size_t)i, y0, kv_f16);
            cuda_kv_store(k, k_base + (size_t)j, y1, kv_f16);
        }
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

static __global__ void split_qgk_prefill_kernel(const float *qgk,
                                                float *q,
                                                float *q_gate,
                                                float *k,
                                                int n_tokens,
                                                int q_dim,
                                                int kv_dim,
                                                int qgk_rows,
                                                int head_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int row_width = 2 * q_dim + kv_dim;
    int total = n_tokens * row_width;
    if (i >= total) return;
    int row = i / row_width;
    int col = i - row * row_width;
    const float *src = qgk + (size_t)row * qgk_rows;
    if (col < 2 * q_dim) {
        int h = col / (2 * head_size);
        int hc = col - h * 2 * head_size;
        if (hc < head_size)
            q[(size_t)row * q_dim + (size_t)h * head_size + hc] = src[col];
        else
            q_gate[(size_t)row * q_dim + (size_t)h * head_size +
                   (hc - head_size)] = src[col];
    } else {
        k[(size_t)row * kv_dim + (col - 2 * q_dim)] = src[col];
    }
}

static __global__ void split_qkv_prefill_kernel(const float *qkv,
                                                float *q,
                                                float *k,
                                                float *v,
                                                int n_tokens,
                                                int q_dim,
                                                int kv_dim,
                                                int qkv_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * (q_dim + 2 * kv_dim);
    if (i >= total) return;
    int row_width = q_dim + 2 * kv_dim;
    int row = i / row_width;
    int col = i - row * row_width;
    const float *src = qkv + (size_t)row * qkv_rows;
    if (col < q_dim) {
        q[(size_t)row * q_dim + col] = src[col];
    } else if (col < q_dim + kv_dim) {
        k[(size_t)row * kv_dim + (col - q_dim)] = src[col];
    } else {
        v[(size_t)row * kv_dim + (col - q_dim - kv_dim)] = src[col];
    }
}

static __global__ void apply_q_gate_prefill_kernel(float *attn,
                                                   const float *q_gate,
                                                   int n_values) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_values) return;
    float gate = q_gate[i];
    attn[i] *= 1.0f / (1.0f + __expf(-gate));
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

static __global__ void bias_add_batch_kernel(float *x, const float *bias,
                                             int rows, int n_tokens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * n_tokens;
    if (i < total) x[i] += bias[i % rows];
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

static __global__ void weighted_add_sigmoid_kernel(
    float *x, const float *r, const float *gate, const float *gate_in,
    int n, int dim, int reset, int complement) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float dot = 0.0f;
        for (int d = 0; d < dim; d++)
            dot += gate_in[d] * gate[d];
        float weight = 1.0f / (1.0f + __expf(-dot));
        if (complement)
            weight = 1.0f - weight;
        float v = weight * r[i];
        x[i] = reset ? v : x[i] + v;
    }
}

static __global__ void weighted_add_sigmoid_reduce_kernel(
    float *x, const float *r, const float *gate, const float *gate_in,
    int n, int dim, int reset, int complement) {
    __shared__ float partial[512];
    int tid = threadIdx.x;
    float dot = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x)
        dot += gate_in[d] * gate[d];
    partial[tid] = dot;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    float weight = 1.0f / (1.0f + __expf(-partial[0]));
    if (complement)
        weight = 1.0f - weight;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = weight * r[i];
        x[i] = reset ? v : x[i] + v;
    }
}

static __global__ void shared_expert_add_sigmoid_batch_kernel(
    float *out, const float *shared, const float *gate, const float *x,
    int n_tokens, int dim) {
    __shared__ float partial[512];
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= n_tokens)
        return;
    const float *x_t = x + (size_t)t * dim;
    float dot = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x)
        dot += x_t[d] * gate[d];
    partial[tid] = dot;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    float weight = 1.0f / (1.0f + __expf(-partial[0]));
    float *out_t = out + (size_t)t * dim;
    const float *shared_t = shared + (size_t)t * dim;
    for (int d = tid; d < dim; d += blockDim.x)
        out_t[d] += weight * shared_t[d];
}

static __global__ void weighted_add_sigmoid_residual_reduce_kernel(
    float *resid, const float *x, const float *r, const float *gate,
    const float *gate_in, int n, int dim, int reset, int complement) {
    __shared__ float partial[512];
    int tid = threadIdx.x;
    float dot = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x)
        dot += gate_in[d] * gate[d];
    partial[tid] = dot;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    float weight = 1.0f / (1.0f + __expf(-partial[0]));
    if (complement)
        weight = 1.0f - weight;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = weight * r[i];
        float combined = reset ? v : x[i] + v;
        resid[i] += combined;
    }
}

static __global__ void weighted_add_sigmoid_residual_rmsnorm_kernel(
    float *resid, const float *x, const float *r, const float *gate,
    const float *gate_in, float *out, const float *norm_weight,
    int n, int dim, int reset, int complement, float eps) {
    __shared__ float partial_dot[512];
    __shared__ float partial_ss[512];
    int tid = threadIdx.x;
    float dot = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x)
        dot += gate_in[d] * gate[d];
    partial_dot[tid] = dot;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial_dot[tid] += partial_dot[tid + stride];
        __syncthreads();
    }

    float weight = 1.0f / (1.0f + __expf(-partial_dot[0]));
    if (complement)
        weight = 1.0f - weight;

    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = weight * r[i];
        float combined = reset ? v : x[i] + v;
        float rv = resid[i] + combined;
        resid[i] = rv;
        ss += rv * rv;
    }
    partial_ss[tid] = ss;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial_ss[tid] += partial_ss[tid + stride];
        __syncthreads();
    }

    float scale = rsqrtf(partial_ss[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = resid[i] * scale * norm_weight[i];
}

static __global__ void moe_router_logits_kernel(float *logits,
                                                const float *router,
                                                const float *x,
                                                int n_experts,
                                                int dim) {
    int e = blockIdx.x;
    int tid = threadIdx.x;
    if (e >= n_experts) return;
    const float *row = router + (size_t)e * (size_t)dim;
    float sum = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x)
        sum += row[d] * x[d];
    extern __shared__ float scratch[];
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        logits[e] = scratch[0];
}

static __global__ void moe_router_logits_warp_kernel(float *logits,
                                                     const float *router,
                                                     const float *x,
                                                     int n_experts,
                                                     int dim) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int e = blockIdx.x * warps_per_block + warp;
    if (e >= n_experts) return;
    const float *row = router + (size_t)e * (size_t)dim;
    float sum = 0.0f;
    for (int d = lane; d < dim; d += 32)
        sum += row[d] * x[d];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        logits[e] = sum;
}

static __global__ void moe_router_logits_2warp_kernel(float *logits,
                                                      const float *router,
                                                      const float *x,
                                                      int n_experts,
                                                      int dim) {
    __shared__ float partial[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = warp >> 1;
    int warp_in_group = warp & 1;
    int groups_per_block = blockDim.x >> 6;
    int e = blockIdx.x * groups_per_block + group;
    if (e >= n_experts || groups_per_block <= 0) return;

    const float *row = router + (size_t)e * (size_t)dim;
    int tid = warp_in_group * 32 + lane;
    float sum = 0.0f;
    for (int d = tid; d < dim; d += 64)
        sum += row[d] * x[d];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[group * 2 + warp_in_group] = sum;
    __syncthreads();

    if (warp_in_group == 0 && lane == 0)
        logits[e] = partial[group * 2] + partial[group * 2 + 1];
}

static __global__ void moe_router_logits_4warp_kernel(float *logits,
                                                      const float *router,
                                                      const float *x,
                                                      int n_experts,
                                                      int dim) {
    __shared__ float partial[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = warp >> 2;
    int warp_in_group = warp & 3;
    int groups_per_block = blockDim.x >> 7;
    int e = blockIdx.x * groups_per_block + group;
    if (e >= n_experts || groups_per_block <= 0) return;

    const float *row = router + (size_t)e * (size_t)dim;
    int tid = warp_in_group * 32 + lane;
    float sum = 0.0f;
    for (int d = tid; d < dim; d += 128)
        sum += row[d] * x[d];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[group * 4 + warp_in_group] = sum;
    __syncthreads();

    if (warp_in_group == 0 && lane == 0)
        logits[e] = partial[group * 4] + partial[group * 4 + 1] +
                    partial[group * 4 + 2] + partial[group * 4 + 3];
}

static __global__ void moe_router_logits_batch_warp_kernel(
    float *logits, const float *router, const float *x,
    int n_tokens, int n_experts, int dim) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    int total = n_tokens * n_experts;
    if (task >= total) return;
    int token = task / n_experts;
    int e = task - token * n_experts;
    const float *row = router + (size_t)e * (size_t)dim;
    const float *xt = x + (size_t)token * (size_t)dim;
    float sum = 0.0f;
    for (int d = lane; d < dim; d += 32)
        sum += row[d] * xt[d];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        logits[(size_t)token * (size_t)n_experts + (size_t)e] = sum;
}

static __global__ void moe_route_topk_kernel(float *route,
                                             const float *logits,
                                             int n_experts,
                                             int k,
                                             int norm_topk,
                                             float expert_weights_scale) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (n_experts <= 0 || k <= 0) return;
    if (k > BN_MAX_MOE_K) k = BN_MAX_MOE_K;

    int selected[BN_MAX_MOE_K];
    float selected_logits[BN_MAX_MOE_K];
    for (int i = 0; i < k; i++) {
        int best = -1;
        float best_val = -INFINITY;
        for (int e = 0; e < n_experts; e++) {
            int used = 0;
            for (int j = 0; j < i; j++) {
                if (selected[j] == e) {
                    used = 1;
                    break;
                }
            }
            if (used) continue;
            float v = logits[e];
            if (v > best_val) {
                best_val = v;
                best = e;
            }
        }
        selected[i] = best < 0 ? 0 : best;
        selected_logits[i] = best_val;
    }

    float max_selected = selected_logits[0];
    for (int i = 1; i < k; i++)
        if (selected_logits[i] > max_selected)
            max_selected = selected_logits[i];
    float max_all = max_selected;
    if (!norm_topk) {
        max_all = logits[0];
        for (int e = 1; e < n_experts; e++)
            if (logits[e] > max_all)
                max_all = logits[e];
    }
    float sum = 0.0f;
    if (norm_topk) {
        for (int i = 0; i < k; i++) {
            float w = expf(selected_logits[i] - max_selected);
            route[i] = w;
            sum += w;
        }
    } else {
        for (int e = 0; e < n_experts; e++)
            sum += expf(logits[e] - max_all);
        for (int i = 0; i < k; i++)
            route[i] = expf(selected_logits[i] - max_all);
    }
    if (sum > 0.0f) {
        for (int i = 0; i < k; i++)
            route[i] /= sum;
    }
    if (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) {
        for (int i = 0; i < k; i++)
            route[i] *= expert_weights_scale;
    }
    for (int i = 0; i < k; i++)
        route[k + i] = (float)selected[i];
}

static __global__ void moe_route_topk_batch_warp_kernel(
    int *indices, float *weights, const float *logits,
    int n_tokens, int n_experts, int k, int norm_topk,
    float expert_weights_scale) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int token = blockIdx.x * warps_per_block + warp;
    if (token >= n_tokens) return;
    if (n_experts <= 0 || k <= 0) return;
    if (k > BN_MAX_MOE_K) k = BN_MAX_MOE_K;

    const float *row = logits + (size_t)token * (size_t)n_experts;
    int selected[BN_MAX_MOE_K];
    float selected_logits[BN_MAX_MOE_K];
    for (int i = 0; i < k; i++) {
        int local_best = -1;
        float local_best_val = -INFINITY;
        for (int e = lane; e < n_experts; e += 32) {
            int used = 0;
            for (int j = 0; j < i; j++) {
                if (selected[j] == e) {
                    used = 1;
                    break;
                }
            }
            if (used) continue;
            float v = row[e];
            if (v > local_best_val) {
                local_best_val = v;
                local_best = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val =
                __shfl_down_sync(0xffffffffu, local_best_val, offset);
            int other =
                __shfl_down_sync(0xffffffffu, local_best, offset);
            if (other_val > local_best_val ||
                (other_val == local_best_val &&
                 other >= 0 && (local_best < 0 || other < local_best))) {
                local_best_val = other_val;
                local_best = other;
            }
        }
        int best = __shfl_sync(0xffffffffu, local_best, 0);
        float best_val = __shfl_sync(0xffffffffu, local_best_val, 0);
        selected[i] = best < 0 ? 0 : best;
        selected_logits[i] = best_val;
    }

    if (lane != 0) return;
    float max_selected = selected_logits[0];
    for (int i = 1; i < k; i++)
        if (selected_logits[i] > max_selected)
            max_selected = selected_logits[i];
    float max_all = max_selected;
    if (!norm_topk) {
        max_all = row[0];
        for (int e = 1; e < n_experts; e++)
            if (row[e] > max_all)
                max_all = row[e];
    }
    float sum = 0.0f;
    if (norm_topk) {
        for (int i = 0; i < k; i++) {
            float w = expf(selected_logits[i] - max_selected);
            weights[(size_t)token * (size_t)k + (size_t)i] = w;
            sum += w;
        }
    } else {
        for (int e = 0; e < n_experts; e++)
            sum += expf(row[e] - max_all);
        for (int i = 0; i < k; i++)
            weights[(size_t)token * (size_t)k + (size_t)i] =
                expf(selected_logits[i] - max_all);
    }
    if (sum > 0.0f) {
        for (int i = 0; i < k; i++)
            weights[(size_t)token * (size_t)k + (size_t)i] /= sum;
    }
    if (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) {
        for (int i = 0; i < k; i++)
            weights[(size_t)token * (size_t)k + (size_t)i] *=
                expert_weights_scale;
    }
    for (int i = 0; i < k; i++)
        indices[(size_t)token * (size_t)k + (size_t)i] = selected[i];
}

static __global__ void moe_route_fused_batch_warp_topk_kernel(
    int *indices, float *weights, const float *router, const float *x,
    int n_tokens, int n_experts, int dim, int k, int norm_topk,
    float expert_weights_scale) {
    __shared__ float logits[256];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int token = blockIdx.x;
    if (token >= n_tokens || n_experts <= 0 || n_experts > 256 ||
        k <= 0 || dim <= 0)
        return;
    if (k > BN_MAX_MOE_K) k = BN_MAX_MOE_K;

    const float *xt = x + (size_t)token * (size_t)dim;
    for (int e = warp; e < n_experts; e += warps_per_block) {
        const float *row = router + (size_t)e * (size_t)dim;
        float sum = 0.0f;
        for (int d = lane; d < dim; d += 32)
            sum += row[d] * xt[d];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0)
            logits[e] = sum;
    }
    __syncthreads();
    if (warp != 0) return;

    int selected[BN_MAX_MOE_K];
    float selected_logits[BN_MAX_MOE_K];
    for (int i = 0; i < k; i++) {
        int local_best = -1;
        float local_best_val = -INFINITY;
        for (int e = lane; e < n_experts; e += 32) {
            int used = 0;
            for (int j = 0; j < i; j++) {
                if (selected[j] == e) {
                    used = 1;
                    break;
                }
            }
            if (used) continue;
            float v = logits[e];
            if (v > local_best_val) {
                local_best_val = v;
                local_best = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val =
                __shfl_down_sync(0xffffffffu, local_best_val, offset);
            int other = __shfl_down_sync(0xffffffffu, local_best, offset);
            if (other_val > local_best_val ||
                (other_val == local_best_val &&
                 other >= 0 && (local_best < 0 || other < local_best))) {
                local_best_val = other_val;
                local_best = other;
            }
        }
        int best = __shfl_sync(0xffffffffu, local_best, 0);
        float best_val = __shfl_sync(0xffffffffu, local_best_val, 0);
        selected[i] = best < 0 ? 0 : best;
        selected_logits[i] = best_val;
    }

    if (lane != 0) return;
    float max_selected = selected_logits[0];
    for (int i = 1; i < k; i++)
        if (selected_logits[i] > max_selected)
            max_selected = selected_logits[i];
    float max_all = max_selected;
    if (!norm_topk) {
        max_all = logits[0];
        for (int e = 1; e < n_experts; e++)
            if (logits[e] > max_all)
                max_all = logits[e];
    }
    float sum = 0.0f;
    if (norm_topk) {
        for (int i = 0; i < k; i++) {
            float w = expf(selected_logits[i] - max_selected);
            weights[(size_t)token * (size_t)k + (size_t)i] = w;
            sum += w;
        }
    } else {
        for (int e = 0; e < n_experts; e++)
            sum += expf(logits[e] - max_all);
        for (int i = 0; i < k; i++)
            weights[(size_t)token * (size_t)k + (size_t)i] =
                expf(selected_logits[i] - max_all);
    }
    if (sum > 0.0f) {
        for (int i = 0; i < k; i++)
            weights[(size_t)token * (size_t)k + (size_t)i] /= sum;
    }
    if (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) {
        for (int i = 0; i < k; i++)
            weights[(size_t)token * (size_t)k + (size_t)i] *=
                expert_weights_scale;
    }
    for (int i = 0; i < k; i++)
        indices[(size_t)token * (size_t)k + (size_t)i] = selected[i];
}

static __global__ void moe_route_count_experts_kernel(
    int *counts, const int *indices, int route_items, int n_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= route_items) return;
    int e = indices[idx];
    if (e >= 0 && e < n_experts)
        atomicAdd(counts + e, 1);
}

static __global__ void moe_route_offsets_kernel(
    int *offsets, int *fill, const int *counts, int n_experts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int sum = 0;
    for (int e = 0; e < n_experts; e++) {
        offsets[e] = sum;
        fill[e] = sum;
        sum += counts[e];
    }
}

static __global__ void moe_route_fill_slots_kernel(
    int *slot_order, int *fill, const int *indices, int route_items,
    int n_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= route_items) return;
    int e = indices[idx];
    if (e < 0 || e >= n_experts) return;
    int pos = atomicAdd(fill + e, 1);
    slot_order[pos] = idx;
}

static __global__ void moe_route_topk_warp_kernel(float *route,
                                                  const float *logits,
                                                  int n_experts,
                                                  int k,
                                                  int norm_topk,
                                                  float expert_weights_scale) {
    int lane = threadIdx.x & 31;
    if (blockIdx.x != 0 || threadIdx.x >= 32) return;
    if (n_experts <= 0 || k <= 0) return;
    if (k > BN_MAX_MOE_K) k = BN_MAX_MOE_K;

    int selected[BN_MAX_MOE_K];
    float selected_logits[BN_MAX_MOE_K];
    for (int i = 0; i < k; i++) {
        int local_best = -1;
        float local_best_val = -INFINITY;
        for (int e = lane; e < n_experts; e += 32) {
            int used = 0;
            for (int j = 0; j < i; j++) {
                if (selected[j] == e) {
                    used = 1;
                    break;
                }
            }
            if (used) continue;
            float v = logits[e];
            if (v > local_best_val) {
                local_best_val = v;
                local_best = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val =
                __shfl_down_sync(0xffffffffu, local_best_val, offset);
            int other =
                __shfl_down_sync(0xffffffffu, local_best, offset);
            if (other_val > local_best_val ||
                (other_val == local_best_val &&
                 other >= 0 && (local_best < 0 || other < local_best))) {
                local_best_val = other_val;
                local_best = other;
            }
        }
        int best = __shfl_sync(0xffffffffu, local_best, 0);
        float best_val = __shfl_sync(0xffffffffu, local_best_val, 0);
        selected[i] = best < 0 ? 0 : best;
        selected_logits[i] = best_val;
    }

    float max_selected = selected_logits[0];
    for (int i = 1; i < k; i++)
        if (selected_logits[i] > max_selected)
            max_selected = selected_logits[i];
    float max_all = max_selected;
    if (!norm_topk) {
        float local_max = -INFINITY;
        for (int e = lane; e < n_experts; e += 32)
            if (logits[e] > local_max)
                local_max = logits[e];
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffffu, local_max, offset);
            if (other > local_max)
                local_max = other;
        }
        max_all = __shfl_sync(0xffffffffu, local_max, 0);
    }
    float sum = 0.0f;
    if (norm_topk) {
        if (lane != 0) return;
        for (int i = 0; i < k; i++) {
            float w = expf(selected_logits[i] - max_selected);
            route[i] = w;
            sum += w;
        }
    } else {
        float local_sum = 0.0f;
        for (int e = lane; e < n_experts; e += 32)
            local_sum += expf(logits[e] - max_all);
        for (int offset = 16; offset > 0; offset >>= 1)
            local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
        sum = __shfl_sync(0xffffffffu, local_sum, 0);
        if (lane != 0) return;
        for (int i = 0; i < k; i++)
            route[i] = expf(selected_logits[i] - max_all);
    }
    if (sum > 0.0f) {
        for (int i = 0; i < k; i++)
            route[i] /= sum;
    }
    if (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) {
        for (int i = 0; i < k; i++)
            route[i] *= expert_weights_scale;
    }
    for (int i = 0; i < k; i++)
        route[k + i] = (float)selected[i];
}

static __global__ void moe_route_fused_warp_topk_kernel(
    float *route, const float *router, const float *x, int n_experts,
    int dim, int k, int norm_topk, float expert_weights_scale) {
    __shared__ float logits[256];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    if (blockIdx.x != 0 || n_experts <= 0 || n_experts > 256 ||
        k <= 0 || dim <= 0)
        return;
    if (k > BN_MAX_MOE_K) k = BN_MAX_MOE_K;

    for (int e = warp; e < n_experts; e += warps_per_block) {
        const float *row = router + (size_t)e * (size_t)dim;
        float sum = 0.0f;
        for (int d = lane; d < dim; d += 32)
            sum += row[d] * x[d];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0)
            logits[e] = sum;
    }
    __syncthreads();
    if (warp != 0) return;

    int selected[BN_MAX_MOE_K];
    float selected_logits[BN_MAX_MOE_K];
    for (int i = 0; i < k; i++) {
        int local_best = -1;
        float local_best_val = -INFINITY;
        for (int e = lane; e < n_experts; e += 32) {
            int used = 0;
            for (int j = 0; j < i; j++) {
                if (selected[j] == e) {
                    used = 1;
                    break;
                }
            }
            if (used) continue;
            float v = logits[e];
            if (v > local_best_val) {
                local_best_val = v;
                local_best = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val =
                __shfl_down_sync(0xffffffffu, local_best_val, offset);
            int other =
                __shfl_down_sync(0xffffffffu, local_best, offset);
            if (other_val > local_best_val ||
                (other_val == local_best_val &&
                 other >= 0 && (local_best < 0 || other < local_best))) {
                local_best_val = other_val;
                local_best = other;
            }
        }
        int best = __shfl_sync(0xffffffffu, local_best, 0);
        float best_val = __shfl_sync(0xffffffffu, local_best_val, 0);
        selected[i] = best < 0 ? 0 : best;
        selected_logits[i] = best_val;
    }

    float max_selected = selected_logits[0];
    for (int i = 1; i < k; i++)
        if (selected_logits[i] > max_selected)
            max_selected = selected_logits[i];
    float max_all = max_selected;
    if (!norm_topk) {
        float local_max = -INFINITY;
        for (int e = lane; e < n_experts; e += 32)
            if (logits[e] > local_max)
                local_max = logits[e];
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffffu, local_max, offset);
            if (other > local_max)
                local_max = other;
        }
        max_all = __shfl_sync(0xffffffffu, local_max, 0);
    }
    float sum = 0.0f;
    if (norm_topk) {
        if (lane != 0) return;
        for (int i = 0; i < k; i++) {
            float w = expf(selected_logits[i] - max_selected);
            route[i] = w;
            sum += w;
        }
    } else {
        float local_sum = 0.0f;
        for (int e = lane; e < n_experts; e += 32)
            local_sum += expf(logits[e] - max_all);
        for (int offset = 16; offset > 0; offset >>= 1)
            local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
        sum = __shfl_sync(0xffffffffu, local_sum, 0);
        if (lane != 0) return;
        for (int i = 0; i < k; i++)
            route[i] = expf(selected_logits[i] - max_all);
    }
    if (sum > 0.0f) {
        for (int i = 0; i < k; i++)
            route[i] /= sum;
    }
    if (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) {
        for (int i = 0; i < k; i++)
            route[i] *= expert_weights_scale;
    }
    for (int i = 0; i < k; i++)
        route[k + i] = (float)selected[i];
}

static __global__ void moe_route_diff2_warp_kernel(float *route,
                                                   const float *router_diff,
                                                   const float *x,
                                                   int dim,
                                                   float expert_weights_scale) {
    int lane = threadIdx.x & 31;
    float sum = 0.0f;
    for (int d = lane; d < dim; d += 32)
        sum += router_diff[d] * x[d];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) {
        float w0 = 1.0f / (1.0f + expf(-sum));
        float w1 = 1.0f - w0;
        if (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) {
            w0 *= expert_weights_scale;
            w1 *= expert_weights_scale;
        }
        if (w1 > w0) {
            route[0] = w1;
            route[1] = w0;
            route[2] = 1.0f;
            route[3] = 0.0f;
        } else {
            route[0] = w0;
            route[1] = w1;
            route[2] = 0.0f;
            route[3] = 1.0f;
        }
    }
}

static __device__ __forceinline__ float cuda_router_dot_avx2_lane(
    const float *row, const float *x, int dim, int lane16) {
    int lane8 = lane16 & 7;
    int base = (lane16 >> 3) * 8 + lane8;
    float sum = 0.0f;
    for (int d = base; d < dim; d += 16)
        sum = __fmaf_rn(row[d], x[d], sum);
    return sum;
}

static __device__ __forceinline__ float cuda_router_avx2_hsum16(
    const float *partials) {
    float v0 = partials[0] + partials[8];
    float v1 = partials[1] + partials[9];
    float v2 = partials[2] + partials[10];
    float v3 = partials[3] + partials[11];
    float v4 = partials[4] + partials[12];
    float v5 = partials[5] + partials[13];
    float v6 = partials[6] + partials[14];
    float v7 = partials[7] + partials[15];
    float h0 = v0 + v4;
    float h1 = v1 + v5;
    float h2 = v2 + v6;
    float h3 = v3 + v7;
    return (h0 + h1) + (h2 + h3);
}

static __global__ void moe_route_all2_avx2_quantize_q8k_kernel(
    float *route,
    BnBlockQ8K *xq,
    const float *router,
    const float *x,
    int dim,
    float expert_weights_scale,
    int norm_topk_prob) {
    int tid = threadIdx.x;
    if (blockIdx.x == 0) {
        __shared__ float partial[2][16];
        if (tid < 16) {
            partial[0][tid] =
                cuda_router_dot_avx2_lane(router, x, dim, tid);
            partial[1][tid] =
                cuda_router_dot_avx2_lane(router + (size_t)dim, x, dim, tid);
        }
        __syncthreads();
        if (tid == 0) {
            float l0 = cuda_router_avx2_hsum16(partial[0]);
            float l1 = cuda_router_avx2_hsum16(partial[1]);
            float max_val = l0 > l1 ? l0 : l1;
            float w0 = expf(l0 - max_val);
            float w1 = expf(l1 - max_val);
            float denom = w0 + w1;
            if (denom > 0.0f) {
                w0 /= denom;
                w1 /= denom;
            }
            if (norm_topk_prob) {
                float wsum = w0 + w1;
                if (wsum > 0.0f) {
                    w0 /= wsum;
                    w1 /= wsum;
                }
            }
            if (expert_weights_scale != 0.0f &&
                expert_weights_scale != 1.0f) {
                w0 *= expert_weights_scale;
                w1 *= expert_weights_scale;
            }
            if (w1 > w0) {
                route[0] = w1;
                route[1] = w0;
                route[2] = 1.0f;
                route[3] = 0.0f;
            } else {
                route[0] = w0;
                route[1] = w1;
                route[2] = 0.0f;
                route[3] = 1.0f;
            }
        }
        return;
    }

    int block = blockIdx.x - 1;
    if (tid >= BN_QK_K) return;
    int n_blocks = (dim + BN_QK_K - 1) / BN_QK_K;
    if (block >= n_blocks) return;
    int c = block * BN_QK_K + tid;
    float v = c < dim ? x[c] : 0.0f;

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
    int q = d == 0.0f ? 0 : (int)roundf(v / d);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    q_s[tid] = q;
    BnBlockQ8K *dst = xq + block;
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

static __global__ void moe_route_diff2_quantize_q8k_kernel(
    float *route,
    BnBlockQ8K *xq,
    const float *router_diff,
    const float *x,
    int dim,
    float expert_weights_scale) {
    int tid = threadIdx.x;
    if (blockIdx.x == 0) {
        if (tid >= 32) return;
        float sum = 0.0f;
        for (int d = tid; d < dim; d += 32)
            sum += router_diff[d] * x[d];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (tid == 0) {
            float w0 = 1.0f / (1.0f + expf(-sum));
            float w1 = 1.0f - w0;
            if (expert_weights_scale != 0.0f &&
                expert_weights_scale != 1.0f) {
                w0 *= expert_weights_scale;
                w1 *= expert_weights_scale;
            }
            if (w1 > w0) {
                route[0] = w1;
                route[1] = w0;
                route[2] = 1.0f;
                route[3] = 0.0f;
            } else {
                route[0] = w0;
                route[1] = w1;
                route[2] = 0.0f;
                route[3] = 1.0f;
            }
        }
        return;
    }

    int block = blockIdx.x - 1;
    if (tid >= BN_QK_K) return;
    int n_blocks = (dim + BN_QK_K - 1) / BN_QK_K;
    if (block >= n_blocks) return;
    int c = block * BN_QK_K + tid;
    float v = c < dim ? x[c] : 0.0f;

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
    int q = d == 0.0f ? 0 : (int)roundf(v / d);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    q_s[tid] = q;
    BnBlockQ8K *dst = xq + block;
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

static __global__ void moe_route_diff2_quantize_q8_1_kernel(
    float *route,
    BnCudaBlockQ8_1 *xq,
    const float *router_diff,
    const float *x,
    int dim,
    float expert_weights_scale) {
    int tid = threadIdx.x;
    if (blockIdx.x == 0) {
        if (tid >= 32) return;
        float sum = 0.0f;
        for (int d = tid; d < dim; d += 32)
            sum += router_diff[d] * x[d];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (tid == 0) {
            float w0 = 1.0f / (1.0f + expf(-sum));
            float w1 = 1.0f - w0;
            if (expert_weights_scale != 0.0f &&
                expert_weights_scale != 1.0f) {
                w0 *= expert_weights_scale;
                w1 *= expert_weights_scale;
            }
            if (w1 > w0) {
                route[0] = w1;
                route[1] = w0;
                route[2] = 1.0f;
                route[3] = 0.0f;
            } else {
                route[0] = w0;
                route[1] = w1;
                route[2] = 0.0f;
                route[3] = 1.0f;
            }
        }
        return;
    }

    int block = blockIdx.x - 1;
    int c = block * 32 + tid;
    float v = c < dim ? x[c] : 0.0f;
    float amax = fabsf(v);
    float sum = v;
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    float d = amax / 127.0f;
    int q = d == 0.0f ? 0 : (int)roundf(v / d);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);

    BnCudaBlockQ8_1 *dst = xq + block;
    dst->qs[tid] = (int8_t)q;
    if (tid == 0) {
        dst->d = cuda_fp32_to_fp16_bits(d);
        dst->s = cuda_fp32_to_fp16_bits(sum);
    }
}

static __global__ void moe_q4k_gateup_routed_mid_kernel(
    float *mid,
    const BnBlockQ4K *gate,
    const BnBlockQ4K *up,
    const BnCudaBlockQ8_1 *xq,
    const float *route,
    int hidden,
    int cols,
    int n_experts,
    int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    if (task >= hidden * k) return;

    int slot = task / hidden;
    int row = task - slot * hidden;
    int expert = (int)(route[k + slot] + 0.5f);
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = kbx; b < n_bpr; b += 2) {
        const BnCudaBlockQ8_1 *xqb = xq + (size_t)b * 8;
        gate_sum += cuda_vec_dot_q4k_q8_1(&gate_blocks[b], xqb, iqs);
        up_sum += cuda_vec_dot_q4k_q8_1(&up_blocks[b], xqb, iqs);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
        up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
    }
    if (lane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[(size_t)slot * (size_t)hidden + (size_t)row] =
            silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_all2_mid_kernel(
    float *mid,
    const BnBlockQ4K *gate,
    const BnBlockQ4K *up,
    const BnCudaBlockQ8_1 *xq,
    const float *route,
    int hidden,
    int cols) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    if (task >= hidden * 2) return;

    int slot = task / hidden;
    int row = task - slot * hidden;
    if (route && fabsf(route[slot]) <= 1.0e-7f) {
        mid[task] = 0.0f;
        return;
    }
    int expert = slot;
    if (route) {
        expert = (int)(route[2 + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert > 1) expert = 1;
    }
    int n_bpr = cols / BN_QK_K;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = kbx; b < n_bpr; b += 2) {
        const BnCudaBlockQ8_1 *xqb = xq + (size_t)b * 8;
        gate_sum += cuda_vec_dot_q4k_q8_1(&gate_blocks[b], xqb, iqs);
        up_sum += cuda_vec_dot_q4k_q8_1(&up_blocks[b], xqb, iqs);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
        up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
    }
    if (lane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[(size_t)slot * (size_t)hidden + (size_t)row] =
            silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_routed_mid_batch_kernel(
    float *mid,
    const BnBlockQ4K *gate,
    const BnBlockQ4K *up,
    const BnCudaBlockQ8_1 *xq,
    const int *indices,
    const float *weights,
    int hidden,
    int cols,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int slot_task = task / hidden;
    int slot = slot_task % k;
    int token = slot_task / k;
    int route_off = token * k + slot;
    if (weights && fabsf(weights[route_off]) <= 1.0e-7f) {
        mid[(size_t)route_off * (size_t)hidden + (size_t)row] = 0.0f;
        return;
    }
    int expert = indices[route_off];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / BN_QK_K;
    int x_blocks = (cols + 31) / 32;
    int kbx = lane / 16;
    int iqs = 2 * (lane & 15);
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    const BnCudaBlockQ8_1 *token_xq =
        xq + (size_t)token * (size_t)x_blocks;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = kbx; b < n_bpr; b += 2) {
        const BnCudaBlockQ8_1 *xqb = token_xq + (size_t)b * 8u;
        gate_sum += cuda_vec_dot_q4k_q8_1(&gate_blocks[b], xqb, iqs);
        up_sum += cuda_vec_dot_q4k_q8_1(&up_blocks[b], xqb, iqs);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
        up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
    }
    if (lane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)hidden + (size_t)row] =
            silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_routed_mid_q8k_kernel(
    float *mid, const BnBlockQ4K *gate, const BnBlockQ4K *up,
    const BnBlockQ8K *xq, const float *route, int hidden, int cols,
    int n_experts, int k, int exact_silu) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    if (task >= hidden * k) return;

    int slot = task / hidden;
    int row = task - slot * hidden;
    int expert = (int)(route[k + slot] + 0.5f);
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / BN_QK_K;
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = lane; b < n_bpr; b += 32) {
        gate_sum += cuda_vec_dot_q4k_q8k(&gate_blocks[b], xq + b);
        up_sum += cuda_vec_dot_q4k_q8k(&up_blocks[b], xq + b);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
        up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
    }
    if (lane == 0) {
        float silu = cuda_silu_select(gate_sum, exact_silu);
        mid[(size_t)slot * (size_t)hidden + (size_t)row] = silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_routed_mid_q8k_4row_kernel(
    float *mid, const BnBlockQ4K *gate, const BnBlockQ4K *up,
    const BnBlockQ8K *xq, const float *route, int hidden, int cols,
    int n_experts, int k, int exact_silu) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int task = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    int total_tasks = hidden * k;
    if (task >= total_tasks) return;

    int slot = task / hidden;
    int row = task - slot * hidden;
    int expert = (int)(route[k + slot] + 0.5f);
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / BN_QK_K;
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 8) {
        gate_sum += cuda_vec_dot_q4k_q8k(&gate_blocks[b], xq + b);
        up_sum += cuda_vec_dot_q4k_q8k(&up_blocks[b], xq + b);
    }
    unsigned mask = 0xffu << (lane_group * 8);
    gate_sum += __shfl_down_sync(mask, gate_sum, 4);
    gate_sum += __shfl_down_sync(mask, gate_sum, 2);
    gate_sum += __shfl_down_sync(mask, gate_sum, 1);
    up_sum += __shfl_down_sync(mask, up_sum, 4);
    up_sum += __shfl_down_sync(mask, up_sum, 2);
    up_sum += __shfl_down_sync(mask, up_sum, 1);
    if (sublane == 0) {
        float silu = cuda_silu_select(gate_sum, exact_silu);
        mid[(size_t)slot * (size_t)hidden + (size_t)row] = silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_all2_mid_q8k_4row_kernel(
    float *mid,
    const BnBlockQ4K *gate,
    const BnBlockQ4K *up,
    const BnBlockQ8K *xq,
    const float *route,
    int hidden,
    int cols,
    int exact_silu) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int task = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    int total_tasks = hidden * 2;
    if (task >= total_tasks) return;

    int slot = task / hidden;
    int row = task - slot * hidden;
    if (route && route[slot] == 0.0f) {
        mid[task] = 0.0f;
        return;
    }
    int expert = slot;
    if (route) {
        expert = (int)(route[2 + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert > 1) expert = 1;
    }
    int n_bpr = cols / BN_QK_K;
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 8) {
        gate_sum += cuda_vec_dot_q4k_q8k(&gate_blocks[b], xq + b);
        up_sum += cuda_vec_dot_q4k_q8k(&up_blocks[b], xq + b);
    }
    unsigned mask = 0xffu << (lane_group * 8);
    gate_sum += __shfl_down_sync(mask, gate_sum, 4);
    gate_sum += __shfl_down_sync(mask, gate_sum, 2);
    gate_sum += __shfl_down_sync(mask, gate_sum, 1);
    up_sum += __shfl_down_sync(mask, up_sum, 4);
    up_sum += __shfl_down_sync(mask, up_sum, 2);
    up_sum += __shfl_down_sync(mask, up_sum, 1);
    if (sublane == 0) {
        float silu = cuda_silu_select(gate_sum, exact_silu);
        mid[(size_t)slot * (size_t)hidden + (size_t)row] = silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_routed_mid_q8k_4row_batch_kernel(
    float *mid, const BnBlockQ4K *gate, const BnBlockQ4K *up,
    const BnBlockQ8K *xq, const int *indices, int hidden, int cols,
    int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int task = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int slot_task = task / hidden;
    int slot = slot_task % k;
    int token = slot_task / k;
    int expert = indices[token * k + slot];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / BN_QK_K;
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    const BnBlockQ8K *token_xq = xq + (size_t)token * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 8) {
        gate_sum += cuda_vec_dot_q4k_q8k(&gate_blocks[b], token_xq + b);
        up_sum += cuda_vec_dot_q4k_q8k(&up_blocks[b], token_xq + b);
    }
    unsigned mask = 0xffu << (lane_group * 8);
    gate_sum += __shfl_down_sync(mask, gate_sum, 4);
    gate_sum += __shfl_down_sync(mask, gate_sum, 2);
    gate_sum += __shfl_down_sync(mask, gate_sum, 1);
    up_sum += __shfl_down_sync(mask, up_sum, 4);
    up_sum += __shfl_down_sync(mask, up_sum, 2);
    up_sum += __shfl_down_sync(mask, up_sum, 1);
    if (sublane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
            (size_t)hidden + (size_t)row] = silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_routed_mid_q8k_4row_sorted_batch_kernel(
    float *mid, const BnBlockQ4K *gate, const BnBlockQ4K *up,
    const BnBlockQ8K *xq, const int *indices, const int *slot_order,
    int hidden, int cols, int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int task = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int sorted_slot = task / hidden;
    int route_pos = slot_order[sorted_slot];
    int slot = route_pos % k;
    int token = route_pos / k;
    int expert = indices[route_pos];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / BN_QK_K;
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    const BnBlockQ8K *token_xq = xq + (size_t)token * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 8) {
        gate_sum += cuda_vec_dot_q4k_q8k(&gate_blocks[b], token_xq + b);
        up_sum += cuda_vec_dot_q4k_q8k(&up_blocks[b], token_xq + b);
    }
    unsigned mask = 0xffu << (lane_group * 8);
    gate_sum += __shfl_down_sync(mask, gate_sum, 4);
    gate_sum += __shfl_down_sync(mask, gate_sum, 2);
    gate_sum += __shfl_down_sync(mask, gate_sum, 1);
    up_sum += __shfl_down_sync(mask, up_sum, 4);
    up_sum += __shfl_down_sync(mask, up_sum, 2);
    up_sum += __shfl_down_sync(mask, up_sum, 1);
    if (sublane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
            (size_t)hidden + (size_t)row] = silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_routed_mid_q8k_2row_split_batch_kernel(
    float *mid, const BnBlockQ4K *gate, const BnBlockQ4K *up,
    const BnBlockQ8K *xq, const int *indices, int hidden, int cols,
    int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 4;
    int sublane = lane & 15;
    int task = (blockIdx.x * warps_per_block + warp) * 2 + lane_group;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int slot_task = task / hidden;
    int slot = slot_task % k;
    int token = slot_task / k;
    int expert = indices[token * k + slot];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / BN_QK_K;
    int block_lane = sublane >> 1;
    int half = sublane & 1;
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    const BnBlockQ8K *token_xq = xq + (size_t)token * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = block_lane; b < n_bpr; b += 8) {
        gate_sum += cuda_vec_dot_q4k_q8k_half(&gate_blocks[b],
                                              token_xq + b, half);
        up_sum += cuda_vec_dot_q4k_q8k_half(&up_blocks[b],
                                            token_xq + b, half);
    }
    unsigned mask = 0xffffu << (lane_group * 16);
    gate_sum += __shfl_down_sync(mask, gate_sum, 8);
    gate_sum += __shfl_down_sync(mask, gate_sum, 4);
    gate_sum += __shfl_down_sync(mask, gate_sum, 2);
    gate_sum += __shfl_down_sync(mask, gate_sum, 1);
    up_sum += __shfl_down_sync(mask, up_sum, 8);
    up_sum += __shfl_down_sync(mask, up_sum, 4);
    up_sum += __shfl_down_sync(mask, up_sum, 2);
    up_sum += __shfl_down_sync(mask, up_sum, 1);
    if (sublane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
            (size_t)hidden + (size_t)row] = silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_routed_mid_q8k_2row_split_sorted_batch_kernel(
    float *mid, const BnBlockQ4K *gate, const BnBlockQ4K *up,
    const BnBlockQ8K *xq, const int *indices, const int *slot_order,
    int hidden, int cols, int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 4;
    int sublane = lane & 15;
    int task = (blockIdx.x * warps_per_block + warp) * 2 + lane_group;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int sorted_slot = task / hidden;
    int route_pos = slot_order[sorted_slot];
    int slot = route_pos % k;
    int token = route_pos / k;
    int expert = indices[route_pos];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / BN_QK_K;
    int block_lane = sublane >> 1;
    int half = sublane & 1;
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    const BnBlockQ8K *token_xq = xq + (size_t)token * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = block_lane; b < n_bpr; b += 8) {
        gate_sum += cuda_vec_dot_q4k_q8k_half(&gate_blocks[b],
                                              token_xq + b, half);
        up_sum += cuda_vec_dot_q4k_q8k_half(&up_blocks[b],
                                            token_xq + b, half);
    }
    unsigned mask = 0xffffu << (lane_group * 16);
    gate_sum += __shfl_down_sync(mask, gate_sum, 8);
    gate_sum += __shfl_down_sync(mask, gate_sum, 4);
    gate_sum += __shfl_down_sync(mask, gate_sum, 2);
    gate_sum += __shfl_down_sync(mask, gate_sum, 1);
    up_sum += __shfl_down_sync(mask, up_sum, 8);
    up_sum += __shfl_down_sync(mask, up_sum, 4);
    up_sum += __shfl_down_sync(mask, up_sum, 2);
    up_sum += __shfl_down_sync(mask, up_sum, 1);
    if (sublane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
            (size_t)hidden + (size_t)row] = silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_routed_mid_q8k_8row_batch_kernel(
    float *mid, const BnBlockQ4K *gate, const BnBlockQ4K *up,
    const BnBlockQ8K *xq, const int *indices, int hidden, int cols,
    int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 2;
    int sublane = lane & 3;
    int task = (blockIdx.x * warps_per_block + warp) * 8 + lane_group;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int slot_task = task / hidden;
    int slot = slot_task % k;
    int token = slot_task / k;
    int expert = indices[token * k + slot];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / BN_QK_K;
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    const BnBlockQ8K *token_xq = xq + (size_t)token * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 4) {
        gate_sum += cuda_vec_dot_q4k_q8k(&gate_blocks[b], token_xq + b);
        up_sum += cuda_vec_dot_q4k_q8k(&up_blocks[b], token_xq + b);
    }
    unsigned mask = 0x0fu << (lane_group * 4);
    gate_sum += __shfl_down_sync(mask, gate_sum, 2);
    gate_sum += __shfl_down_sync(mask, gate_sum, 1);
    up_sum += __shfl_down_sync(mask, up_sum, 2);
    up_sum += __shfl_down_sync(mask, up_sum, 1);
    if (sublane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
            (size_t)hidden + (size_t)row] = silu * up_sum;
    }
}

static __global__ void moe_q4k_gateup_routed_mid_q8k_8row_sorted_batch_kernel(
    float *mid, const BnBlockQ4K *gate, const BnBlockQ4K *up,
    const BnBlockQ8K *xq, const int *indices, const int *slot_order,
    int hidden, int cols, int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 2;
    int sublane = lane & 3;
    int task = (blockIdx.x * warps_per_block + warp) * 8 + lane_group;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int sorted_slot = task / hidden;
    int route_pos = slot_order[sorted_slot];
    int slot = route_pos % k;
    int token = route_pos / k;
    int expert = indices[route_pos];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / BN_QK_K;
    size_t expert_row = ((size_t)expert * (size_t)hidden + (size_t)row);
    const BnBlockQ4K *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ4K *up_blocks = up + expert_row * (size_t)n_bpr;
    const BnBlockQ8K *token_xq = xq + (size_t)token * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 4) {
        gate_sum += cuda_vec_dot_q4k_q8k(&gate_blocks[b], token_xq + b);
        up_sum += cuda_vec_dot_q4k_q8k(&up_blocks[b], token_xq + b);
    }
    unsigned mask = 0x0fu << (lane_group * 4);
    gate_sum += __shfl_down_sync(mask, gate_sum, 2);
    gate_sum += __shfl_down_sync(mask, gate_sum, 1);
    up_sum += __shfl_down_sync(mask, up_sum, 2);
    up_sum += __shfl_down_sync(mask, up_sum, 1);
    if (sublane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
            (size_t)hidden + (size_t)row] = silu * up_sum;
    }
}

static __global__ void moe_q8_0_gateup_routed_mid_kernel(
    float *mid, const BnBlockQ8_0 *gate, const BnBlockQ8_0 *up,
    const float *x, const float *route, int hidden, int cols,
    int n_experts, int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    if (task >= hidden * k) return;

    int slot = task / hidden;
    int row = task - slot * hidden;
    int expert = (int)(route[k + slot] + 0.5f);
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / 32;
    size_t expert_row = (size_t)expert * (size_t)hidden + (size_t)row;
    const BnBlockQ8_0 *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ8_0 *up_blocks = up + expert_row * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = 0; b < n_bpr; b++) {
        float xv = x[(size_t)b * 32u + (size_t)lane];
        const BnBlockQ8_0 *gb = &gate_blocks[b];
        const BnBlockQ8_0 *ub = &up_blocks[b];
        float gd = lane == 0 ? cuda_fp16_to_fp32(gb->d) : 0.0f;
        float ud = lane == 0 ? cuda_fp16_to_fp32(ub->d) : 0.0f;
        gd = __shfl_sync(0xffffffffu, gd, 0);
        ud = __shfl_sync(0xffffffffu, ud, 0);
        gate_sum += gd * (float)gb->qs[lane] * xv;
        up_sum += ud * (float)ub->qs[lane] * xv;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
        up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
    }
    if (lane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[(size_t)slot * (size_t)hidden + (size_t)row] =
            silu * up_sum;
    }
}

static __global__ void moe_q8_0_gateup_routed_mid_q8_1_kernel(
    float *mid, const BnBlockQ8_0 *gate, const BnBlockQ8_0 *up,
    const BnCudaBlockQ8_1 *xq, const float *route, int hidden, int cols,
    int n_experts, int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    if (task >= hidden * k) return;

    int slot = task / hidden;
    int row = task - slot * hidden;
    int expert = (int)(route[k + slot] + 0.5f);
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / 32;
    size_t expert_row = (size_t)expert * (size_t)hidden + (size_t)row;
    const BnBlockQ8_0 *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ8_0 *up_blocks = up + expert_row * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = lane; b < n_bpr; b += 32) {
        const BnBlockQ8_0 *gb = &gate_blocks[b];
        const BnBlockQ8_0 *ub = &up_blocks[b];
        float xd = cuda_fp16_to_fp32(xq[b].d);
        int gate_dot = cuda_dot_i8x32_dp4a(gb->qs, xq[b].qs);
        int up_dot = cuda_dot_i8x32_dp4a(ub->qs, xq[b].qs);
        gate_sum += cuda_fp16_to_fp32(gb->d) * xd * (float)gate_dot;
        up_sum += cuda_fp16_to_fp32(ub->d) * xd * (float)up_dot;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
        up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
    }
    if (lane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[(size_t)slot * (size_t)hidden + (size_t)row] =
            silu * up_sum;
    }
}

static __global__ void moe_q8_0_gateup_routed_mid_batch_kernel(
    float *mid, const BnBlockQ8_0 *gate, const BnBlockQ8_0 *up,
    const float *x, const int *indices, const float *weights,
    int hidden, int cols, int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int slot_task = task / hidden;
    int slot = slot_task % k;
    int token = slot_task / k;
    int route_off = token * k + slot;
    int expert = indices[route_off];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / 32;
    size_t expert_row = (size_t)expert * (size_t)hidden + (size_t)row;
    const BnBlockQ8_0 *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ8_0 *up_blocks = up + expert_row * (size_t)n_bpr;
    const float *x_token = x + (size_t)token * (size_t)cols;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = 0; b < n_bpr; b++) {
        float xv = x_token[(size_t)b * 32u + (size_t)lane];
        const BnBlockQ8_0 *gb = &gate_blocks[b];
        const BnBlockQ8_0 *ub = &up_blocks[b];
        float gd = lane == 0 ? cuda_fp16_to_fp32(gb->d) : 0.0f;
        float ud = lane == 0 ? cuda_fp16_to_fp32(ub->d) : 0.0f;
        gd = __shfl_sync(0xffffffffu, gd, 0);
        ud = __shfl_sync(0xffffffffu, ud, 0);
        gate_sum += gd * (float)gb->qs[lane] * xv;
        up_sum += ud * (float)ub->qs[lane] * xv;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
        up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
    }
    if (lane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)hidden + (size_t)row] =
            silu * up_sum;
    }
}

static __global__ void moe_q8_0_gateup_routed_mid_q8_1_batch_kernel(
    float *mid, const BnBlockQ8_0 *gate, const BnBlockQ8_0 *up,
    const BnCudaBlockQ8_1 *xq, const int *indices,
    int hidden, int cols, int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int slot_task = task / hidden;
    int slot = slot_task % k;
    int token = slot_task / k;
    int expert = indices[token * k + slot];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / 32;
    size_t expert_row = (size_t)expert * (size_t)hidden + (size_t)row;
    const BnBlockQ8_0 *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ8_0 *up_blocks = up + expert_row * (size_t)n_bpr;
    const BnCudaBlockQ8_1 *xq_token =
        xq + (size_t)token * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = lane; b < n_bpr; b += 32) {
        const BnBlockQ8_0 *gb = &gate_blocks[b];
        const BnBlockQ8_0 *ub = &up_blocks[b];
        const BnCudaBlockQ8_1 *xb = &xq_token[b];
        int gate_dot = cuda_dot_i8x32_dp4a(gb->qs, xb->qs);
        int up_dot = cuda_dot_i8x32_dp4a(ub->qs, xb->qs);
        float xd = cuda_fp16_to_fp32(xb->d);
        gate_sum += cuda_fp16_to_fp32(gb->d) * xd * (float)gate_dot;
        up_sum += cuda_fp16_to_fp32(ub->d) * xd * (float)up_dot;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
        up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
    }
    if (lane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)hidden + (size_t)row] =
            silu * up_sum;
    }
}

static __global__ void moe_q8_0_gateup_routed_mid_q8_1_2row_batch_kernel(
    float *mid, const BnBlockQ8_0 *gate, const BnBlockQ8_0 *up,
    const BnCudaBlockQ8_1 *xq, const int *indices,
    int hidden, int cols, int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 4;
    int sublane = lane & 15;
    int task = (blockIdx.x * warps_per_block + warp) * 2 + lane_group;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int slot_task = task / hidden;
    int slot = slot_task % k;
    int token = slot_task / k;
    int expert = indices[token * k + slot];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / 32;
    size_t expert_row = (size_t)expert * (size_t)hidden + (size_t)row;
    const BnBlockQ8_0 *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ8_0 *up_blocks = up + expert_row * (size_t)n_bpr;
    const BnCudaBlockQ8_1 *xq_token =
        xq + (size_t)token * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 16) {
        const BnBlockQ8_0 *gb = &gate_blocks[b];
        const BnBlockQ8_0 *ub = &up_blocks[b];
        const BnCudaBlockQ8_1 *xb = &xq_token[b];
        int gate_dot = cuda_dot_i8x32_dp4a(gb->qs, xb->qs);
        int up_dot = cuda_dot_i8x32_dp4a(ub->qs, xb->qs);
        float xd = cuda_fp16_to_fp32(xb->d);
        gate_sum += cuda_fp16_to_fp32(gb->d) * xd * (float)gate_dot;
        up_sum += cuda_fp16_to_fp32(ub->d) * xd * (float)up_dot;
    }
    unsigned mask = 0xffffu << (lane_group * 16);
    gate_sum += __shfl_down_sync(mask, gate_sum, 8);
    gate_sum += __shfl_down_sync(mask, gate_sum, 4);
    gate_sum += __shfl_down_sync(mask, gate_sum, 2);
    gate_sum += __shfl_down_sync(mask, gate_sum, 1);
    up_sum += __shfl_down_sync(mask, up_sum, 8);
    up_sum += __shfl_down_sync(mask, up_sum, 4);
    up_sum += __shfl_down_sync(mask, up_sum, 2);
    up_sum += __shfl_down_sync(mask, up_sum, 1);
    if (sublane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)hidden + (size_t)row] =
            silu * up_sum;
    }
}

static __global__ void moe_q8_0_gateup_routed_mid_q8_1_2row_sorted_batch_kernel(
    float *mid, const BnBlockQ8_0 *gate, const BnBlockQ8_0 *up,
    const BnCudaBlockQ8_1 *xq, const int *indices, const int *slot_order,
    int hidden, int cols, int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 4;
    int sublane = lane & 15;
    int task = (blockIdx.x * warps_per_block + warp) * 2 + lane_group;
    int total_tasks = n_tokens * k * hidden;
    if (task >= total_tasks) return;

    int row = task % hidden;
    int sorted_slot = task / hidden;
    int route_pos = slot_order[sorted_slot];
    int token = route_pos / k;
    int slot = route_pos - token * k;
    int expert = indices[route_pos];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = cols / 32;
    size_t expert_row = (size_t)expert * (size_t)hidden + (size_t)row;
    const BnBlockQ8_0 *gate_blocks = gate + expert_row * (size_t)n_bpr;
    const BnBlockQ8_0 *up_blocks = up + expert_row * (size_t)n_bpr;
    const BnCudaBlockQ8_1 *xq_token =
        xq + (size_t)token * (size_t)n_bpr;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 16) {
        const BnBlockQ8_0 *gb = &gate_blocks[b];
        const BnBlockQ8_0 *ub = &up_blocks[b];
        const BnCudaBlockQ8_1 *xb = &xq_token[b];
        int gate_dot = cuda_dot_i8x32_dp4a(gb->qs, xb->qs);
        int up_dot = cuda_dot_i8x32_dp4a(ub->qs, xb->qs);
        float xd = cuda_fp16_to_fp32(xb->d);
        gate_sum += cuda_fp16_to_fp32(gb->d) * xd * (float)gate_dot;
        up_sum += cuda_fp16_to_fp32(ub->d) * xd * (float)up_dot;
    }
    unsigned mask = 0xffffu << (lane_group * 16);
    gate_sum += __shfl_down_sync(mask, gate_sum, 8);
    gate_sum += __shfl_down_sync(mask, gate_sum, 4);
    gate_sum += __shfl_down_sync(mask, gate_sum, 2);
    gate_sum += __shfl_down_sync(mask, gate_sum, 1);
    up_sum += __shfl_down_sync(mask, up_sum, 8);
    up_sum += __shfl_down_sync(mask, up_sum, 4);
    up_sum += __shfl_down_sync(mask, up_sum, 2);
    up_sum += __shfl_down_sync(mask, up_sum, 1);
    if (sublane == 0) {
        float silu = gate_sum / (1.0f + __expf(-gate_sum));
        mid[((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)hidden + (size_t)row] =
            silu * up_sum;
    }
}

static __global__ void moe_q6k_down_routed_q8k_accum_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= dim) return;

    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = (int)(route[k + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q = mid_q + (size_t)slot * (size_t)n_bpr;
        float route_weight = route[slot];
        float slot_sum = 0.0f;
        for (int b = lane; b < n_bpr; b += 32)
            slot_sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
        sum += route_weight * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[row] = sum;
}

static __global__ void moe_q6k_down_routed_q8k_ordered_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;

    int n_bpr = hidden / BN_QK_K;
    float total = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        float route_weight = route[slot];
        int expert = (int)(route[k + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q = mid_q + (size_t)slot * (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = 0; b < n_bpr; b++)
            slot_sum += cuda_vec_dot_q6k_q8k(&row_blocks[b],
                                             slot_mid_q + b);
        total += route_weight * slot_sum;
    }
    out[row] = total;
}

static __global__ void moe_q6k_down_routed_q8k_pair_8row_kernel(
    float *pair_out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 2;
    int sublane = lane & 3;
    int row = (blockIdx.x * warps_per_block + warp) * 8 + lane_group;
    int slot = blockIdx.y;
    if (row >= dim || slot >= k) return;

    int expert = (int)(route[k + slot] + 0.5f);
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;
    int n_bpr = hidden / BN_QK_K;
    const BnBlockQ6K *row_blocks =
        down + (((size_t)expert * (size_t)dim + (size_t)row) *
                (size_t)n_bpr);
    const BnBlockQ8K *slot_mid_q = mid_q + (size_t)slot * (size_t)n_bpr;
    float sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 4)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
    unsigned mask = 0x0fu << (lane_group * 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0)
        pair_out[(size_t)slot * (size_t)dim + (size_t)row] =
            route[slot] * sum;
}

static __global__ void moe_q6k_down_routed_q8k_pair_4row_kernel(
    float *pair_out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    int slot = blockIdx.y;
    if (row >= dim || slot >= k) return;

    int expert = (int)(route[k + slot] + 0.5f);
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;
    int n_bpr = hidden / BN_QK_K;
    const BnBlockQ6K *row_blocks =
        down + (((size_t)expert * (size_t)dim + (size_t)row) *
                (size_t)n_bpr);
    const BnBlockQ8K *slot_mid_q = mid_q + (size_t)slot * (size_t)n_bpr;
    float sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 8)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
    unsigned mask = 0xffu << (lane_group * 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0)
        pair_out[(size_t)slot * (size_t)dim + (size_t)row] =
            route[slot] * sum;
}

static __global__ void moe_q6k_down_routed_q8k_k8_4row_sum_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    __shared__ float partial[8][4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = blockIdx.x * 4 + lane_group;
    int slot = warp;

    float routed = 0.0f;
    if (row < dim && slot < k) {
        float route_weight = route[slot];
        if (route_weight != 0.0f) {
            int expert = (int)(route[k + slot] + 0.5f);
            if (expert < 0) expert = 0;
            if (expert >= n_experts) expert = n_experts - 1;
            int n_bpr = hidden / BN_QK_K;
            const BnBlockQ6K *row_blocks =
                down + (((size_t)expert * (size_t)dim + (size_t)row) *
                        (size_t)n_bpr);
            const BnBlockQ8K *slot_mid_q =
                mid_q + (size_t)slot * (size_t)n_bpr;
            float sum = 0.0f;
            for (int b = sublane; b < n_bpr; b += 8)
                sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
            unsigned mask = 0xffu << (lane_group * 8);
            sum += __shfl_down_sync(mask, sum, 4);
            sum += __shfl_down_sync(mask, sum, 2);
            sum += __shfl_down_sync(mask, sum, 1);
            routed = route_weight * sum;
        }
    }
    if (sublane == 0)
        partial[slot][lane_group] = routed;
    __syncthreads();

    if (warp == 0 && lane < 4) {
        int out_row = blockIdx.x * 4 + lane;
        if (out_row < dim) {
            float sum = 0.0f;
            for (int s = 0; s < k; s++)
                sum += partial[s][lane];
            out[out_row] = sum;
        }
    }
}

static __global__ void moe_q6k_down_routed_q8k_k8_4row_residual_sum_kernel(
    float *residual,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    __shared__ float partial[8][4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = blockIdx.x * 4 + lane_group;
    int slot = warp;

    float routed = 0.0f;
    if (row < dim && slot < k) {
        float route_weight = route[slot];
        if (route_weight != 0.0f) {
            int expert = (int)(route[k + slot] + 0.5f);
            if (expert < 0) expert = 0;
            if (expert >= n_experts) expert = n_experts - 1;
            int n_bpr = hidden / BN_QK_K;
            const BnBlockQ6K *row_blocks =
                down + (((size_t)expert * (size_t)dim + (size_t)row) *
                        (size_t)n_bpr);
            const BnBlockQ8K *slot_mid_q =
                mid_q + (size_t)slot * (size_t)n_bpr;
            float sum = 0.0f;
            for (int b = sublane; b < n_bpr; b += 8)
                sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
            unsigned mask = 0xffu << (lane_group * 8);
            sum += __shfl_down_sync(mask, sum, 4);
            sum += __shfl_down_sync(mask, sum, 2);
            sum += __shfl_down_sync(mask, sum, 1);
            routed = route_weight * sum;
        }
    }
    if (sublane == 0)
        partial[slot][lane_group] = routed;
    __syncthreads();

    if (warp == 0 && lane < 4) {
        int out_row = blockIdx.x * 4 + lane;
        if (out_row < dim) {
            float sum = 0.0f;
            for (int s = 0; s < k; s++)
                sum += partial[s][lane];
            residual[out_row] += sum;
        }
    }
}

static __global__ void moe_q6k_down_routed_q8k_k8_4row_residual_sum_2048_768_kernel(
    float *residual,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int n_experts) {
    __shared__ float partial[8][4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = blockIdx.x * 4 + lane_group;
    int slot = warp;

    float route_weight = route[slot];
    int expert = (int)(route[8 + slot] + 0.5f);
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;
    const int dim = 2048;
    const int hidden = 768;
    const int n_bpr = hidden / BN_QK_K;
    const BnBlockQ6K *row_blocks =
        down + (((size_t)expert * (size_t)dim + (size_t)row) *
                (size_t)n_bpr);
    const BnBlockQ8K *slot_mid_q = mid_q + (size_t)slot * (size_t)n_bpr;
    float sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 8)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
    unsigned mask = 0xffu << (lane_group * 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0)
        partial[slot][lane_group] = route_weight * sum;
    __syncthreads();

    if (warp == 0 && lane < 4) {
        int out_row = blockIdx.x * 4 + lane;
        float sum_out =
            partial[0][lane] + partial[1][lane] +
            partial[2][lane] + partial[3][lane] +
            partial[4][lane] + partial[5][lane] +
            partial[6][lane] + partial[7][lane];
        residual[out_row] += sum_out;
    }
}

static __global__ void moe_q6k_down_routed_q8k_k8_8row_sum_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    __shared__ float partial[8][8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int lane_group = lane >> 2;
    int sublane = lane & 3;
    int row = blockIdx.x * 8 + lane_group;
    int slot = warp;

    float routed = 0.0f;
    if (row < dim && slot < k) {
        float route_weight = route[slot];
        if (route_weight != 0.0f) {
            int expert = (int)(route[k + slot] + 0.5f);
            if (expert < 0) expert = 0;
            if (expert >= n_experts) expert = n_experts - 1;
            int n_bpr = hidden / BN_QK_K;
            const BnBlockQ6K *row_blocks =
                down + (((size_t)expert * (size_t)dim + (size_t)row) *
                        (size_t)n_bpr);
            const BnBlockQ8K *slot_mid_q =
                mid_q + (size_t)slot * (size_t)n_bpr;
            float sum = 0.0f;
            for (int b = sublane; b < n_bpr; b += 4)
                sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
            unsigned mask = 0x0fu << (lane_group * 4);
            sum += __shfl_down_sync(mask, sum, 2);
            sum += __shfl_down_sync(mask, sum, 1);
            routed = route_weight * sum;
        }
    }
    if (sublane == 0)
        partial[slot][lane_group] = routed;
    __syncthreads();

    if (warp == 0 && lane < 8) {
        int out_row = blockIdx.x * 8 + lane;
        if (out_row < dim) {
            float sum = 0.0f;
            for (int s = 0; s < k; s++)
                sum += partial[s][lane];
            out[out_row] = sum;
        }
    }
}

static __global__ void moe_q6k_down_all2_q8k_accum_2row_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 4;
    int sublane = lane & 15;
    int row = (blockIdx.x * warps_per_block + warp) * 2 + lane_group;
    if (row >= dim) return;

    int n_bpr = hidden / BN_QK_K;
    float pair0 = 0.0f;
    float pair1 = 0.0f;
#pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        float route_weight = route[slot];
        if (fabsf(route_weight) <= 1.0e-7f) continue;
        int expert = (int)(route[2 + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert > 1) expert = 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q = mid_q + (size_t)slot * (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = sublane; b < n_bpr; b += 16)
            slot_sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
        unsigned mask = 0xffffu << (lane_group * 16);
        slot_sum += __shfl_down_sync(mask, slot_sum, 8);
        slot_sum += __shfl_down_sync(mask, slot_sum, 4);
        slot_sum += __shfl_down_sync(mask, slot_sum, 2);
        slot_sum += __shfl_down_sync(mask, slot_sum, 1);
        if (sublane == 0) {
            float routed = route_weight * slot_sum;
            if (slot == 0)
                pair0 = routed;
            else
                pair1 = routed;
        }
    }
    if (sublane == 0)
        out[row] = pair0 + pair1;
}

static __global__ void moe_q6k_down_all2_q8k_accum_4row_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    if (row >= dim) return;

    int n_bpr = hidden / BN_QK_K;
    float pair0 = 0.0f;
    float pair1 = 0.0f;
#pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        float route_weight = route[slot];
        if (fabsf(route_weight) <= 1.0e-7f) continue;
        int expert = (int)(route[2 + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert > 1) expert = 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q = mid_q + (size_t)slot * (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = sublane; b < n_bpr; b += 8)
            slot_sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
        unsigned mask = 0xffu << (lane_group * 8);
        slot_sum += __shfl_down_sync(mask, slot_sum, 4);
        slot_sum += __shfl_down_sync(mask, slot_sum, 2);
        slot_sum += __shfl_down_sync(mask, slot_sum, 1);
        if (sublane == 0) {
            float routed = route_weight * slot_sum;
            if (slot == 0)
                pair0 = routed;
            else
                pair1 = routed;
        }
    }
    if (sublane == 0)
        out[row] = pair0 + pair1;
}

static __global__ void moe_q6k_down_all2_q8k_pair4_sum_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden) {
    __shared__ float partial[8][4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int slot = warp >> 2;
    int warp_in_slot = warp & 3;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = (blockIdx.x * 4 + warp_in_slot) * 4 + lane_group;

    float routed = 0.0f;
    if (row < dim && slot < 2) {
        int expert = (int)(route[2 + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert > 1) expert = 1;
        int n_bpr = hidden / BN_QK_K;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q =
            mid_q + (size_t)slot * (size_t)n_bpr;
        float sum = 0.0f;
        for (int b = sublane; b < n_bpr; b += 8)
            sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
        unsigned mask = 0xffu << (lane_group * 8);
        sum += __shfl_down_sync(mask, sum, 4);
        sum += __shfl_down_sync(mask, sum, 2);
        sum += __shfl_down_sync(mask, sum, 1);
        routed = route[slot] * sum;
    }
    if (sublane == 0)
        partial[warp][lane_group] = routed;
    __syncthreads();

    if (slot == 0 && sublane == 0 && row < dim)
        out[row] = partial[warp][lane_group] +
                   partial[warp + 4][lane_group];
}

static __global__ void moe_q6k_down_all2_q8k_pair4_sum_fixed_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden) {
    __shared__ float partial[8][4];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int slot = warp >> 2;
    int warp_in_slot = warp & 3;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int row = (blockIdx.x * 4 + warp_in_slot) * 4 + lane_group;

    float routed = 0.0f;
    if (row < dim && slot < 2) {
        float route_weight = route[slot];
        if (route_weight != 0.0f) {
            int expert = (int)(route[2 + slot] + 0.5f);
            if (expert < 0) expert = 0;
            if (expert > 1) expert = 1;
            int n_bpr = hidden / BN_QK_K;
            const BnBlockQ6K *row_blocks =
                down + (((size_t)expert * (size_t)dim + (size_t)row) *
                        (size_t)n_bpr);
            const BnBlockQ8K *slot_mid_q =
                mid_q + (size_t)slot * (size_t)n_bpr;
            float sum = 0.0f;
            for (int b = sublane; b < n_bpr; b += 8)
                sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
            unsigned mask = 0xffu << (lane_group * 8);
            sum += __shfl_down_sync(mask, sum, 4);
            sum += __shfl_down_sync(mask, sum, 2);
            sum += __shfl_down_sync(mask, sum, 1);
            routed = route_weight * sum;
        }
    }
    if (sublane == 0)
        partial[warp][lane_group] = routed;
    __syncthreads();

    if (slot == 0 && sublane == 0 && row < dim)
        out[row] = partial[warp][lane_group] +
                   partial[warp + 4][lane_group];
}

static __global__ void moe_q4k_down_routed_q8k_pair_8row_kernel(
    float *pair_out,
    const BnBlockQ4K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 2;
    int sublane = lane & 3;
    int row = (blockIdx.x * warps_per_block + warp) * 8 + lane_group;
    int slot = blockIdx.y;
    if (row >= dim || slot >= k) return;

    int expert = (int)(route[k + slot] + 0.5f);
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;
    int n_bpr = hidden / BN_QK_K;
    const BnBlockQ4K *row_blocks =
        down + (((size_t)expert * (size_t)dim + (size_t)row) *
                (size_t)n_bpr);
    const BnBlockQ8K *slot_mid_q = mid_q + (size_t)slot * (size_t)n_bpr;
    float sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 4)
        sum += cuda_vec_dot_q4k_q8k(&row_blocks[b], slot_mid_q + b);
    unsigned mask = 0x0fu << (lane_group * 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0)
        pair_out[(size_t)slot * (size_t)dim + (size_t)row] =
            route[slot] * sum;
}

static __global__ void moe_sum_pairs_kernel(float *out,
                                            const float *pair_out,
                                            int dim,
                                            int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++)
        sum += pair_out[(size_t)slot * (size_t)dim + (size_t)idx];
    out[idx] = sum;
}

static __global__ void moe_q6k_down_routed_float_accum_row_kernel(
    float *out,
    const BnBlockQ6K *down,
    const float *mid,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= dim) return;

    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = (int)(route[k + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const float *slot_mid = mid + (size_t)slot * (size_t)hidden;
        float slot_sum = 0.0f;
        for (int b = 0; b < n_bpr; b++)
            slot_sum += cuda_q6k_value(&row_blocks[b], tid) *
                        slot_mid[(size_t)b * BN_QK_K + tid];
        sum += route[slot] * slot_sum;
    }

    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        out[row] = partial[0];
}

static __global__ void moe_q6k_down_all2_float_accum_4row_kernel(
    float *out,
    const BnBlockQ6K *down,
    const float *mid,
    const float *route,
    int dim,
    int hidden) {
    int tid = threadIdx.x;
    int row_in_group = threadIdx.y;
    int row = blockIdx.x * 4 + row_in_group;
    if (row >= dim) return;

    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        float route_weight = route[slot];
        if (fabsf(route_weight) <= 1.0e-7f) continue;
        int expert = (int)(route[2 + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert > 1) expert = 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const float *slot_mid = mid + (size_t)slot * (size_t)hidden;
        float slot_sum = 0.0f;
        for (int b = 0; b < n_bpr; b++)
            slot_sum += cuda_q6k_value(&row_blocks[b], tid) *
                        slot_mid[(size_t)b * BN_QK_K + tid];
        sum += route_weight * slot_sum;
    }

    __shared__ float partial[4][256];
    partial[row_in_group][tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial[row_in_group][tid] +=
                partial[row_in_group][tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        out[row] = partial[row_in_group][0];
}

static __global__ void moe_q6k_down_routed_f32_cache_warp_kernel(
    float *out,
    const float *down,
    const float *mid,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= dim) return;

    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = (int)(route[k + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const float *row_w =
            down + ((size_t)expert * (size_t)dim + (size_t)row) *
                       (size_t)hidden;
        const float *slot_mid = mid + (size_t)slot * (size_t)hidden;
        float slot_sum = 0.0f;
        for (int c = lane; c < hidden; c += 32)
            slot_sum += row_w[c] * slot_mid[c];
        sum += route[slot] * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[row] = sum;
}

static __global__ void moe_q6k_down_all2_f32_cache_warp_kernel(
    float *out,
    const float *down,
    const float *mid,
    const float *route,
    int dim,
    int hidden) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= dim) return;

    float sum = 0.0f;
    for (int slot = 0; slot < 2; slot++) {
        float route_weight = route[slot];
        if (fabsf(route_weight) <= 1.0e-7f) continue;
        int expert = (int)(route[2 + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert > 1) expert = 1;
        const float *row_w =
            down + ((size_t)expert * (size_t)dim + (size_t)row) *
                       (size_t)hidden;
        const float *slot_mid = mid + (size_t)slot * (size_t)hidden;
        float slot_sum = 0.0f;
        for (int c = lane; c < hidden; c += 32)
            slot_sum += row_w[c] * slot_mid[c];
        sum += route_weight * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[row] = sum;
}

static __global__ void moe_q6k_down_all2_f32_cache_pair2_sum_kernel(
    float *out,
    const float *down,
    const float *mid,
    const float *route,
    int dim,
    int hidden) {
    int lane = threadIdx.x;
    int lane_group = threadIdx.y;
    int row_in_pair = lane_group & 1;
    int slot = lane_group >> 1;
    int row = blockIdx.x * 2 + row_in_pair;
    __shared__ float partial[4];

    float sum = 0.0f;
    float route_weight = slot < 2 ? route[slot] : 0.0f;
    if (row < dim && slot < 2 && fabsf(route_weight) > 1.0e-7f) {
        int expert = (int)(route[2 + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert > 1) expert = 1;
        const float *row_w =
            down + ((size_t)expert * (size_t)dim + (size_t)row) *
                       (size_t)hidden;
        const float *slot_mid = mid + (size_t)slot * (size_t)hidden;
        for (int c = lane; c < hidden; c += 32)
            sum += row_w[c] * slot_mid[c];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0)
        partial[lane_group] =
            (row < dim && slot < 2) ? route_weight * sum : 0.0f;
    __syncthreads();
    if (lane == 0 && slot == 0 && row < dim)
        out[row] = partial[row_in_pair] + partial[2 + row_in_pair];
}

static __global__ void moe_q6k_down_all2_f32_cache_4row_sum_kernel(
    float *out,
    const float *down,
    const float *mid,
    const float *route,
    int dim,
    int hidden) {
    int lane = threadIdx.x;
    int lane_group = threadIdx.y;
    int row_in_group = lane_group & 3;
    int slot = lane_group >> 2;
    int row = blockIdx.x * 4 + row_in_group;
    __shared__ float partial[8];

    float sum = 0.0f;
    float route_weight = slot < 2 ? route[slot] : 0.0f;
    if (row < dim && slot < 2 && fabsf(route_weight) > 1.0e-7f) {
        int expert = (int)(route[2 + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert > 1) expert = 1;
        const float *row_w =
            down + ((size_t)expert * (size_t)dim + (size_t)row) *
                       (size_t)hidden;
        const float *slot_mid = mid + (size_t)slot * (size_t)hidden;
        for (int c = lane; c < hidden; c += 32)
            sum += row_w[c] * slot_mid[c];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0)
        partial[lane_group] =
            (row < dim && slot < 2) ? route_weight * sum : 0.0f;
    __syncthreads();
    if (lane == 0 && slot == 0 && row < dim)
        out[row] = partial[row_in_group] + partial[4 + row_in_group];
}

static __global__ void moe_q6k_down_all2_f32_cache_exact_4row_kernel(
    float *out,
    const float *down,
    const float *mid,
    const float *route,
    int dim,
    int hidden,
    float skip_eps) {
    int tid = threadIdx.x;
    int row_in_group = threadIdx.y;
    int row = blockIdx.x * 4 + row_in_group;
    if (row >= dim) return;

    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        float route_weight = route[slot];
        if (fabsf(route_weight) <= skip_eps) continue;
        int expert = (int)(route[2 + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert > 1) expert = 1;
        const float *row_w =
            down + ((size_t)expert * (size_t)dim + (size_t)row) *
                       (size_t)hidden;
        const float *slot_mid = mid + (size_t)slot * (size_t)hidden;
        float slot_sum = 0.0f;
        for (int b = 0; b < n_bpr; b++)
            slot_sum += row_w[(size_t)b * BN_QK_K + tid] *
                        slot_mid[(size_t)b * BN_QK_K + tid];
        sum += route_weight * slot_sum;
    }

    __shared__ float partial[4][256];
    partial[row_in_group][tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial[row_in_group][tid] +=
                partial[row_in_group][tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        out[row] = partial[row_in_group][0];
}

static __global__ void moe_q6k_down_routed_f16_cache_warp_kernel(
    float *out,
    const __half *down,
    const float *mid,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= dim) return;

    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = (int)(route[k + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const uint16_t *row_w =
            (const uint16_t *)down +
            ((size_t)expert * (size_t)dim + (size_t)row) * (size_t)hidden;
        const float *slot_mid = mid + (size_t)slot * (size_t)hidden;
        float slot_sum = 0.0f;
        for (int c = lane; c < hidden; c += 32)
            slot_sum += cuda_fp16_to_fp32(row_w[c]) * slot_mid[c];
        sum += route[slot] * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[row] = sum;
}

static __global__ void moe_q6k_down_routed_f32_cache_batch_kernel(
    float *out,
    const float *down,
    const float *mid,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int row = task % dim;
    int token = task / dim;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int route_pos = token * k + slot;
        float route_weight = weights[route_pos];
        if (fabsf(route_weight) <= 1.0e-7f)
            continue;
        int expert = indices[route_pos];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const float *row_w =
            down + ((size_t)expert * (size_t)dim + (size_t)row) *
                       (size_t)hidden;
        const float *slot_mid =
            mid + (size_t)route_pos * (size_t)hidden;
        float slot_sum = 0.0f;
        for (int c = lane; c < hidden; c += 32)
            slot_sum += row_w[c] * slot_mid[c];
        sum += route_weight * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] += sum;
}

static __global__ void moe_q6k_down_routed_q8k_accum_batch_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = indices[token * k + slot];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q =
            mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)n_bpr;
        float route_weight = weights[token * k + slot];
        float slot_sum = 0.0f;
        for (int b = lane; b < n_bpr; b += 32)
            slot_sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
        sum += route_weight * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] += sum;
}

static __global__ void moe_q6k_down_routed_q8k_accum_halfwarp_batch_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 4;
    int sublane = lane & 15;
    int task = (blockIdx.x * warps_per_block + warp) * 2 + lane_group;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / BN_QK_K;
    float total = 0.0f;
    unsigned mask = 0xffffu << (lane_group * 16);
    for (int slot = 0; slot < k; slot++) {
        int expert = indices[token * k + slot];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q =
            mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = 0; b < n_bpr; b++)
            slot_sum += cuda_q6k_q8k_part(&row_blocks[b], slot_mid_q + b,
                                          sublane);
        slot_sum += __shfl_down_sync(mask, slot_sum, 8);
        slot_sum += __shfl_down_sync(mask, slot_sum, 4);
        slot_sum += __shfl_down_sync(mask, slot_sum, 2);
        slot_sum += __shfl_down_sync(mask, slot_sum, 1);
        if (sublane == 0)
            total += weights[token * k + slot] * slot_sum;
    }
    if (sublane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] = total;
}

static __global__ void moe_q6k_down_routed_q8k_accum_4row_split_batch_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int task = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / BN_QK_K;
    float total = 0.0f;
    unsigned mask = 0xffu << (lane_group * 8);
    for (int slot = 0; slot < k; slot++) {
        int expert = indices[token * k + slot];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q =
            mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = 0; b < n_bpr; b++) {
            slot_sum += cuda_q6k_q8k_part(&row_blocks[b],
                                          slot_mid_q + b, sublane);
            slot_sum += cuda_q6k_q8k_part(&row_blocks[b],
                                          slot_mid_q + b, sublane + 8);
        }
        slot_sum += __shfl_down_sync(mask, slot_sum, 4);
        slot_sum += __shfl_down_sync(mask, slot_sum, 2);
        slot_sum += __shfl_down_sync(mask, slot_sum, 1);
        if (sublane == 0)
            total += weights[token * k + slot] * slot_sum;
    }
    if (sublane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] = total;
}

static __global__ void moe_q6k_down_routed_q8k_accum_4row_batch_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int task = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = indices[token * k + slot];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q =
            mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = sublane; b < n_bpr; b += 8)
            slot_sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
        sum += weights[token * k + slot] * slot_sum;
    }
    unsigned mask = 0xffu << (lane_group * 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] += sum;
}

static __global__ void moe_q6k_down_routed_q8k_accum_8row_batch_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 2;
    int sublane = lane & 3;
    int task = (blockIdx.x * warps_per_block + warp) * 8 + lane_group;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = indices[token * k + slot];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ6K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q =
            mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = sublane; b < n_bpr; b += 4)
            slot_sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
        sum += weights[token * k + slot] * slot_sum;
    }
    unsigned mask = 0x0fu << (lane_group * 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] = sum;
}

static __global__ void moe_q6k_down_routed_q8k_scatter_8row_batch_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 2;
    int sublane = lane & 3;
    int task = (blockIdx.x * warps_per_block + warp) * 8 + lane_group;
    int total_tasks = n_tokens * k * dim;
    if (task >= total_tasks) return;

    int row = task % dim;
    int slot_task = task / dim;
    int slot = slot_task % k;
    int token = slot_task / k;
    int expert = indices[token * k + slot];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = hidden / BN_QK_K;
    const BnBlockQ6K *row_blocks =
        down + (((size_t)expert * (size_t)dim + (size_t)row) *
                (size_t)n_bpr);
    const BnBlockQ8K *slot_mid_q =
        mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
            (size_t)n_bpr;
    float sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 4)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
    unsigned mask = 0x0fu << (lane_group * 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0) {
        float w = weights[token * k + slot];
        atomicAdd(out + (size_t)token * (size_t)dim + (size_t)row, w * sum);
    }
}

static __global__ void moe_q6k_down_routed_q8k_scatter_8row_sorted_batch_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    const int *slot_order,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 2;
    int sublane = lane & 3;
    int task = (blockIdx.x * warps_per_block + warp) * 8 + lane_group;
    int total_tasks = n_tokens * k * dim;
    if (task >= total_tasks) return;

    int row = task % dim;
    int sorted_slot = task / dim;
    int route_pos = slot_order[sorted_slot];
    int slot = route_pos % k;
    int token = route_pos / k;
    int expert = indices[route_pos];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = hidden / BN_QK_K;
    const BnBlockQ6K *row_blocks =
        down + (((size_t)expert * (size_t)dim + (size_t)row) *
                (size_t)n_bpr);
    const BnBlockQ8K *slot_mid_q =
        mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
            (size_t)n_bpr;
    float sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 4)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
    unsigned mask = 0x0fu << (lane_group * 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0) {
        float w = weights[route_pos];
        atomicAdd(out + (size_t)token * (size_t)dim + (size_t)row, w * sum);
    }
}

static __global__ void moe_q6k_down_routed_q8k_scatter_16row_batch_kernel(
    float *out,
    const BnBlockQ6K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 1;
    int sublane = lane & 1;
    int task = (blockIdx.x * warps_per_block + warp) * 16 + lane_group;
    int total_tasks = n_tokens * k * dim;
    if (task >= total_tasks) return;

    int row = task % dim;
    int slot_task = task / dim;
    int slot = slot_task % k;
    int token = slot_task / k;
    int expert = indices[token * k + slot];
    if (expert < 0) expert = 0;
    if (expert >= n_experts) expert = n_experts - 1;

    int n_bpr = hidden / BN_QK_K;
    const BnBlockQ6K *row_blocks =
        down + (((size_t)expert * (size_t)dim + (size_t)row) *
                (size_t)n_bpr);
    const BnBlockQ8K *slot_mid_q =
        mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
            (size_t)n_bpr;
    float sum = 0.0f;
    for (int b = sublane; b < n_bpr; b += 2)
        sum += cuda_vec_dot_q6k_q8k(&row_blocks[b], slot_mid_q + b);
    unsigned mask = 0x03u << (lane_group * 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0) {
        float w = weights[token * k + slot];
        atomicAdd(out + (size_t)token * (size_t)dim + (size_t)row, w * sum);
    }
}

static __global__ void moe_q4k_down_routed_q8k_accum_kernel(
    float *out,
    const BnBlockQ4K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= dim) return;

    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = (int)(route[k + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ4K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q = mid_q + (size_t)slot * (size_t)n_bpr;
        float route_weight = route[slot];
        float slot_sum = 0.0f;
        for (int b = lane; b < n_bpr; b += 32)
            slot_sum += cuda_vec_dot_q4k_q8k(&row_blocks[b], slot_mid_q + b);
        sum += route_weight * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[row] = sum;
}

static __global__ void moe_q4k_down_routed_q8k_accum_8row_kernel(
    float *out,
    const BnBlockQ4K *down,
    const BnBlockQ8K *mid_q,
    const float *route,
    int dim,
    int hidden,
    int n_experts,
    int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 2;
    int sublane = lane & 3;
    int row = (blockIdx.x * warps_per_block + warp) * 8 + lane_group;
    if (row >= dim) return;

    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        float route_weight = route[slot];
        if (fabsf(route_weight) <= 1.0e-7f) continue;
        int expert = (int)(route[k + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ4K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q =
            mid_q + (size_t)slot * (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = sublane; b < n_bpr; b += 4)
            slot_sum += cuda_vec_dot_q4k_q8k(&row_blocks[b], slot_mid_q + b);
        unsigned mask = 0x0fu << (lane_group * 4);
        slot_sum += __shfl_down_sync(mask, slot_sum, 2);
        slot_sum += __shfl_down_sync(mask, slot_sum, 1);
        sum += route_weight * slot_sum;
    }
    if (sublane == 0)
        out[row] = sum;
}

static __global__ void moe_q4k_down_routed_q8k_accum_batch_kernel(
    float *out,
    const BnBlockQ4K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = indices[token * k + slot];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ4K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q =
            mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)n_bpr;
        float route_weight = weights[token * k + slot];
        float slot_sum = 0.0f;
        for (int b = lane; b < n_bpr; b += 32)
            slot_sum += cuda_vec_dot_q4k_q8k(&row_blocks[b], slot_mid_q + b);
        sum += route_weight * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] += sum;
}

static __global__ void moe_q4k_down_routed_q8k_accum_4row_batch_kernel(
    float *out,
    const BnBlockQ4K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int task = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = indices[token * k + slot];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ4K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q =
            mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = sublane; b < n_bpr; b += 8)
            slot_sum += cuda_vec_dot_q4k_q8k(&row_blocks[b], slot_mid_q + b);
        sum += weights[token * k + slot] * slot_sum;
    }
    unsigned mask = 0xffu << (lane_group * 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] = sum;
}

static __global__ void moe_q4k_down_routed_q8k_accum_8row_batch_kernel(
    float *out,
    const BnBlockQ4K *down,
    const BnBlockQ8K *mid_q,
    const int *indices,
    const float *weights,
    int dim,
    int hidden,
    int n_experts,
    int k,
    int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 2;
    int sublane = lane & 3;
    int task = (blockIdx.x * warps_per_block + warp) * 8 + lane_group;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / BN_QK_K;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = indices[token * k + slot];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ4K *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnBlockQ8K *slot_mid_q =
            mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = sublane; b < n_bpr; b += 4)
            slot_sum += cuda_vec_dot_q4k_q8k(&row_blocks[b], slot_mid_q + b);
        sum += weights[token * k + slot] * slot_sum;
    }
    unsigned mask = 0x0fu << (lane_group * 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (sublane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] = sum;
}

static __global__ void moe_q8_0_down_routed_accum_kernel(
    float *out, const BnBlockQ8_0 *down, const float *mid,
    const float *route, int dim, int hidden, int n_experts, int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= dim) return;

    int n_bpr = hidden / 32;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = (int)(route[k + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ8_0 *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const float *slot_mid = mid + (size_t)slot * (size_t)hidden;
        float route_weight = route[slot];
        float slot_sum = 0.0f;
        for (int b = 0; b < n_bpr; b++) {
            const BnBlockQ8_0 *blk = &row_blocks[b];
            float d = lane == 0 ? cuda_fp16_to_fp32(blk->d) : 0.0f;
            d = __shfl_sync(0xffffffffu, d, 0);
            float xv = slot_mid[(size_t)b * 32u + (size_t)lane];
            slot_sum += d * (float)blk->qs[lane] * xv;
        }
        sum += route_weight * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[row] = sum;
}

static __global__ void moe_q8_0_down_routed_q8_1_accum_kernel(
    float *out, const BnBlockQ8_0 *down, const BnCudaBlockQ8_1 *mid_q,
    const float *route, int dim, int hidden, int n_experts, int k) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= dim) return;

    int n_bpr = hidden / 32;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = (int)(route[k + slot] + 0.5f);
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ8_0 *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnCudaBlockQ8_1 *slot_mid_q =
            mid_q + (size_t)slot * (size_t)n_bpr;
        float route_weight = route[slot];
        float slot_sum = 0.0f;
        for (int b = lane; b < n_bpr; b += 32) {
            const BnBlockQ8_0 *blk = &row_blocks[b];
            int dot = cuda_dot_i8x32_dp4a(blk->qs, slot_mid_q[b].qs);
            slot_sum += cuda_fp16_to_fp32(blk->d) *
                        cuda_fp16_to_fp32(slot_mid_q[b].d) * (float)dot;
        }
        sum += route_weight * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[row] = sum;
}

static __global__ void moe_q8_0_down_routed_accum_batch_kernel(
    float *out, const BnBlockQ8_0 *down, const float *mid,
    const int *indices, const float *weights, int dim, int hidden,
    int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / 32;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = indices[token * k + slot];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ8_0 *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const float *slot_mid =
            mid + ((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)hidden;
        float route_weight = weights[token * k + slot];
        float slot_sum = 0.0f;
        for (int b = 0; b < n_bpr; b++) {
            const BnBlockQ8_0 *blk = &row_blocks[b];
            float d = lane == 0 ? cuda_fp16_to_fp32(blk->d) : 0.0f;
            d = __shfl_sync(0xffffffffu, d, 0);
            float xv = slot_mid[(size_t)b * 32u + (size_t)lane];
            slot_sum += d * (float)blk->qs[lane] * xv;
        }
        sum += route_weight * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] += sum;
}

static __global__ void moe_q8_0_down_routed_q8_1_accum_batch_kernel(
    float *out, const BnBlockQ8_0 *down, const BnCudaBlockQ8_1 *mid_q,
    const int *indices, const float *weights, int dim, int hidden,
    int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int task = blockIdx.x * warps_per_block + warp;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / 32;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int expert = indices[token * k + slot];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ8_0 *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnCudaBlockQ8_1 *slot_mid_q =
            mid_q + ((size_t)token * (size_t)k + (size_t)slot) *
                (size_t)n_bpr;
        float route_weight = weights[token * k + slot];
        float slot_sum = 0.0f;
        for (int b = lane; b < n_bpr; b += 32) {
            const BnBlockQ8_0 *blk = &row_blocks[b];
            int dot = cuda_dot_i8x32_dp4a(blk->qs, slot_mid_q[b].qs);
            slot_sum += cuda_fp16_to_fp32(blk->d) *
                        cuda_fp16_to_fp32(slot_mid_q[b].d) * (float)dot;
        }
        sum += route_weight * slot_sum;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] += sum;
}

static __global__ void moe_q8_0_down_routed_q8_1_accum_2row_batch_kernel(
    float *out, const BnBlockQ8_0 *down, const BnCudaBlockQ8_1 *mid_q,
    const int *indices, const float *weights, int dim, int hidden,
    int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 4;
    int sublane = lane & 15;
    int task = (blockIdx.x * warps_per_block + warp) * 2 + lane_group;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / 32;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int route_pos = token * k + slot;
        float route_weight = weights[route_pos];
        if (fabsf(route_weight) <= 1.0e-7f)
            continue;
        int expert = indices[route_pos];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ8_0 *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnCudaBlockQ8_1 *slot_mid_q =
            mid_q + (size_t)route_pos * (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = sublane; b < n_bpr; b += 16) {
            const BnBlockQ8_0 *blk = &row_blocks[b];
            int dot = cuda_dot_i8x32_dp4a(blk->qs, slot_mid_q[b].qs);
            slot_sum += cuda_fp16_to_fp32(blk->d) *
                        cuda_fp16_to_fp32(slot_mid_q[b].d) * (float)dot;
        }
        unsigned mask = 0xffffu << (lane_group * 16);
        slot_sum += __shfl_down_sync(mask, slot_sum, 8);
        slot_sum += __shfl_down_sync(mask, slot_sum, 4);
        slot_sum += __shfl_down_sync(mask, slot_sum, 2);
        slot_sum += __shfl_down_sync(mask, slot_sum, 1);
        sum += route_weight * slot_sum;
    }
    if (sublane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] += sum;
}

static __global__ void moe_q8_0_down_routed_q8_1_accum_4row_batch_kernel(
    float *out, const BnBlockQ8_0 *down, const BnCudaBlockQ8_1 *mid_q,
    const int *indices, const float *weights, int dim, int hidden,
    int n_experts, int k, int n_tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int lane_group = lane >> 3;
    int sublane = lane & 7;
    int task = (blockIdx.x * warps_per_block + warp) * 4 + lane_group;
    int total_tasks = n_tokens * dim;
    if (task >= total_tasks) return;

    int token = task / dim;
    int row = task - token * dim;
    int n_bpr = hidden / 32;
    float sum = 0.0f;
    for (int slot = 0; slot < k; slot++) {
        int route_pos = token * k + slot;
        float route_weight = weights[route_pos];
        if (fabsf(route_weight) <= 1.0e-7f)
            continue;
        int expert = indices[route_pos];
        if (expert < 0) expert = 0;
        if (expert >= n_experts) expert = n_experts - 1;
        const BnBlockQ8_0 *row_blocks =
            down + (((size_t)expert * (size_t)dim + (size_t)row) *
                    (size_t)n_bpr);
        const BnCudaBlockQ8_1 *slot_mid_q =
            mid_q + (size_t)route_pos * (size_t)n_bpr;
        float slot_sum = 0.0f;
        for (int b = sublane; b < n_bpr; b += 8) {
            const BnBlockQ8_0 *blk = &row_blocks[b];
            int dot = cuda_dot_i8x32_dp4a(blk->qs, slot_mid_q[b].qs);
            slot_sum += cuda_fp16_to_fp32(blk->d) *
                        cuda_fp16_to_fp32(slot_mid_q[b].d) * (float)dot;
        }
        unsigned mask = 0xffu << (lane_group * 8);
        slot_sum += __shfl_down_sync(mask, slot_sum, 4);
        slot_sum += __shfl_down_sync(mask, slot_sum, 2);
        slot_sum += __shfl_down_sync(mask, slot_sum, 1);
        sum += route_weight * slot_sum;
    }
    if (sublane == 0)
        out[(size_t)token * (size_t)dim + (size_t)row] += sum;
}

static __device__ __forceinline__ float cuda_fast_exp(float x) {
    x = fminf(88.7f, fmaxf(-87.3f, x));
    float n_f = floorf(x * 1.4426950409f + 0.5f);
    int n = (int)n_f;
    float r = x - n_f * 0.6931471806f;
    float poly = fmaf(0.04166664f, r, 0.16666667f);
    poly = fmaf(poly, r, 0.49999994f);
    poly = fmaf(poly, r, 1.0f);
    poly = fmaf(poly, r, 1.0f);
    return poly * __int_as_float((n + 127) << 23);
}

static __device__ __forceinline__ float cuda_fast_sigmoid(float x) {
    return 1.0f / (1.0f + cuda_fast_exp(-x));
}

static __global__ void activation_gate_kernel(float *x, const float *aux,
                                              int n, int kind, int param1,
                                              uint32_t flags) {
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
            x[i] = v * cuda_fast_sigmoid(gate);
        } else {
            x[i] = aux[i] * cuda_fast_sigmoid(v);
        }
    } else {
        float silu = (flags & BN_GPU_OP_FLAG_EXACT_SILU)
            ? cuda_silu_select(v, 1)
            : v * cuda_fast_sigmoid(v);
        x[i] = silu * aux[i];
    }
}

static __global__ void activation_kernel(float *x, int n, int kind,
                                         uint32_t flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (kind == BN_GPU_CODE_RELU2_ACT) {
        float r = v > 0.0f ? v : 0.0f;
        x[i] = r * r;
    } else {
        x[i] = (flags & BN_GPU_OP_FLAG_EXACT_SILU)
            ? cuda_silu_select(v, 1)
            : v * cuda_fast_sigmoid(v);
    }
}

static __device__ BnCudaArgmaxPair argmax_better(BnCudaArgmaxPair a,
                                                 BnCudaArgmaxPair b) {
    if (b.value > a.value || (b.value == a.value && b.index < a.index))
        return b;
    return a;
}

static __device__ int argmax_penalty_contains(const int *penalty_tokens,
                                              int n_penalty_tokens,
                                              int token) {
    int lo = 0;
    int hi = n_penalty_tokens;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        int v = penalty_tokens[mid];
        if (v < token)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < n_penalty_tokens && penalty_tokens[lo] == token;
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
            if (argmax_penalty_contains(penalty_tokens,
                                        n_penalty_tokens, idx))
                v = v > 0.0f ? v / repeat_penalty : v * repeat_penalty;
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

static __global__ void argmax_stage1_fast_kernel(
        BnCudaArgmaxPair *partials, const float *x, int n) {
    __shared__ BnCudaArgmaxPair warp_best[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int idx = blockIdx.x * blockDim.x + tid;
    BnCudaArgmaxPair best;
    best.value = idx < n ? x[idx] : -INFINITY;
    best.index = idx < n ? idx : 0;

    for (int offset = 16; offset > 0; offset >>= 1) {
        BnCudaArgmaxPair other;
        other.value = __shfl_down_sync(0xffffffffu, best.value, offset);
        other.index = __shfl_down_sync(0xffffffffu, best.index, offset);
        best = argmax_better(best, other);
    }
    if (lane == 0)
        warp_best[warp] = best;
    __syncthreads();

    if (warp == 0) {
        int n_warps = blockDim.x >> 5;
        best.value = lane < n_warps ? warp_best[lane].value : -INFINITY;
        best.index = lane < n_warps ? warp_best[lane].index : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            BnCudaArgmaxPair other;
            other.value = __shfl_down_sync(0xffffffffu, best.value, offset);
            other.index = __shfl_down_sync(0xffffffffu, best.index, offset);
            best = argmax_better(best, other);
        }
        if (lane == 0)
            partials[blockIdx.x] = best;
    }
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

static __global__ void argmax_stage2_fast_kernel(
        int *out, const BnCudaArgmaxPair *partials, int n_partials) {
    __shared__ BnCudaArgmaxPair warp_best[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    BnCudaArgmaxPair best;
    best.value = -INFINITY;
    best.index = 0;
    for (int i = tid; i < n_partials; i += blockDim.x)
        best = argmax_better(best, partials[i]);

    for (int offset = 16; offset > 0; offset >>= 1) {
        BnCudaArgmaxPair other;
        other.value = __shfl_down_sync(0xffffffffu, best.value, offset);
        other.index = __shfl_down_sync(0xffffffffu, best.index, offset);
        best = argmax_better(best, other);
    }
    if (lane == 0)
        warp_best[warp] = best;
    __syncthreads();

    if (warp == 0) {
        int n_warps = blockDim.x >> 5;
        best.value = lane < n_warps ? warp_best[lane].value : -INFINITY;
        best.index = lane < n_warps ? warp_best[lane].index : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            BnCudaArgmaxPair other;
            other.value = __shfl_down_sync(0xffffffffu, best.value, offset);
            other.index = __shfl_down_sync(0xffffffffu, best.index, offset);
            best = argmax_better(best, other);
        }
        if (lane == 0)
            *out = best.index;
    }
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
            float y = state[state_base + (size_t)vi * head_k_dim + ki] *
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
        state[state_base + (size_t)vi * head_k_dim + ki] +=
            kk * b * (v[(size_t)v_off + (size_t)hv_idx * head_v_dim + vi] -
                      sk[vi]);
    }
    __syncthreads();

    for (int vi = tid; vi < head_v_dim; vi += blockDim.x) {
        float sum = 0.0f;
        float comp = 0.0f;
        for (int ki = 0; ki < head_k_dim; ki++) {
            float y = state[state_base + (size_t)vi * head_k_dim + ki] *
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
        size_t idx = state_base + (size_t)col * 128u + (size_t)row;
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
        state[state_base + (size_t)col * 128u + (size_t)row] = s_shard[r];
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

static __global__ void ssm_prefill_conv_silu_kernel(
    float *qkv, float *conv_state, const float *conv1d_w,
    int qkv_dim, int kern, int n_tokens, size_t conv_off) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= qkv_dim || kern < 2) return;
    float hist[7];
    int hist_n = kern - 1;
    if (hist_n > 7) return;
    for (int k = 0; k < hist_n; k++)
        hist[k] = conv_state[conv_off + (size_t)k * qkv_dim + ch];
    for (int t = 0; t < n_tokens; t++) {
        float cur = qkv[(size_t)t * qkv_dim + ch];
        float sum = cur * conv1d_w[(size_t)ch * kern + (kern - 1)];
        for (int k = 0; k < hist_n; k++)
            sum += hist[k] * conv1d_w[(size_t)ch * kern + k];
        for (int k = 0; k < hist_n - 1; k++)
            hist[k] = hist[k + 1];
        hist[hist_n - 1] = cur;
        qkv[(size_t)t * qkv_dim + ch] = sum / (1.0f + expf(-sum));
    }
    for (int k = 0; k < hist_n; k++)
        conv_state[conv_off + (size_t)k * qkv_dim + ch] = hist[k];
}

static __global__ void ssm_prefill_l2norm_kernel(
    float *qkv, int n_tokens, int head_dim, int q_off, int k_off,
    int num_k_heads, int qkv_dim) {
    int head = blockIdx.x;
    int tok = blockIdx.y;
    int tid = threadIdx.x;
    if (head >= num_k_heads || tok >= n_tokens) return;
    extern __shared__ float scratch[];
    float qn = 0.0f;
    float kn = 0.0f;
    size_t tok_off = (size_t)tok * qkv_dim;
    size_t qb = tok_off + (size_t)q_off + (size_t)head * head_dim;
    size_t kb = tok_off + (size_t)k_off + (size_t)head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float qv = qkv[qb + d];
        float kv = qkv[kb + d];
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
        qkv[qb + d] *= inv_q;
        qkv[kb + d] *= inv_k;
    }
}

static __global__ void ssm_prefill_alpha_beta_kernel(
    float *alpha, float *beta, const float *dt_bias, const float *a_log,
    int num_v_heads, int n_tokens) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_v_heads * n_tokens;
    if (h >= total) return;
    int hv = h % num_v_heads;
    float dt = alpha[h] + dt_bias[hv];
    float dt_sp = dt > 20.0f ? dt : logf(1.0f + expf(dt));
    alpha[h] = expf(dt_sp * a_log[hv]);
    beta[h] = 1.0f / (1.0f + expf(-beta[h]));
}

static __global__ void ssm_prefill_alpha_beta_f32_kernel(
    float *alpha, float *beta, const float *alpha_w, const float *beta_w,
    const float *x, const float *dt_bias, const float *a_log,
    int num_v_heads, int dim, int n_tokens) {
    int h = blockIdx.x;
    int tok = blockIdx.y;
    int tid = threadIdx.x;
    if (h >= num_v_heads || tok >= n_tokens) return;
    extern __shared__ float scratch[];
    const float *x_t = x + (size_t)tok * (size_t)dim;
    const float *aw = alpha_w + (size_t)h * (size_t)dim;
    const float *bw = beta_w + (size_t)h * (size_t)dim;
    float as = 0.0f;
    float bs = 0.0f;
    for (int c = tid; c < dim; c += blockDim.x) {
        float xv = x_t[c];
        as += aw[c] * xv;
        bs += bw[c] * xv;
    }
    float *scratch_a = scratch;
    float *scratch_b = scratch + 8;
    as = cuda_block_reduce_sum_all(as, scratch_a);
    bs = cuda_block_reduce_sum_all(bs, scratch_b);
    if (tid == 0) {
        float dt = as + dt_bias[h];
        float dt_sp = dt > 20.0f ? dt : logf(1.0f + expf(dt));
        size_t idx = (size_t)tok * (size_t)num_v_heads + (size_t)h;
        alpha[idx] = expf(dt_sp * a_log[h]);
        beta[idx] = 1.0f / (1.0f + expf(-bs));
    }
}

static __global__ void ssm_prefill_split_ab_kernel(
    const float *ab, float *alpha, float *beta, int num_v_heads,
    int n_tokens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_v_heads * n_tokens;
    if (i >= total) return;
    int tok = i / num_v_heads;
    int h = i - tok * num_v_heads;
    const float *ab_t = ab + (size_t)tok * (size_t)(2 * num_v_heads);
    alpha[i] = ab_t[h];
    beta[i] = ab_t[num_v_heads + h];
}

static __global__ void ssm_prefill_split_qkvz_kernel(
    const float *qkvz, float *qkv, float *z, int qkv_dim, int inner_dim,
    int n_tokens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * (qkv_dim + inner_dim);
    if (i >= total) return;
    int stride = qkv_dim + inner_dim;
    int tok = i / stride;
    int col = i - tok * stride;
    if (col < qkv_dim) {
        qkv[(size_t)tok * (size_t)qkv_dim + col] = qkvz[i];
    } else {
        z[(size_t)tok * (size_t)inner_dim + (col - qkv_dim)] = qkvz[i];
    }
}

static __global__ void ssm_prefill_delta_128_warp_kernel(
    float *state, float *out, const float *qkv,
    const float *alpha, const float *beta, int n_tokens, int qkv_dim,
    int num_k_heads, int num_v_heads, float q_scale, size_t state_off,
    int q_off, int k_off, int v_off) {
    const int hv_idx = blockIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= 128 || hv_idx >= num_v_heads) return;

    const int hk_idx = hv_idx % num_k_heads;
    const size_t state_base = state_off + (size_t)hv_idx * 128u * 128u;
    float s_shard[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int row = r * 32 + lane;
        s_shard[r] = state[state_base + (size_t)col * 128u + (size_t)row];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float *q_t = qkv + (size_t)t * qkv_dim + q_off +
                           (size_t)hk_idx * 128u;
        const float *k_t = qkv + (size_t)t * qkv_dim + k_off +
                           (size_t)hk_idx * 128u;
        const float *v_t = qkv + (size_t)t * qkv_dim + v_off +
                           (size_t)hv_idx * 128u;
        const float decay = alpha[(size_t)t * num_v_heads + hv_idx];
        const float b = beta[(size_t)t * num_v_heads + hv_idx];
        float k_reg[4];
        float q_reg[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = r * 32 + lane;
            s_shard[r] *= decay;
            k_reg[r] = k_t[row];
            q_reg[r] = q_t[row];
        }

        float kv_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; r++)
            kv_partial += s_shard[r] * k_reg[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            kv_partial += __shfl_down_sync(0xffffffffu, kv_partial, offset);
        float kv_col = __shfl_sync(0xffffffffu, kv_partial, 0);
        float delta = (v_t[col] - kv_col) * b;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            s_shard[r] += k_reg[r] * delta;
            attn_partial += s_shard[r] * q_reg[r];
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            attn_partial += __shfl_down_sync(0xffffffffu, attn_partial, offset);
        if (lane == 0)
            out[(size_t)t * (size_t)num_v_heads * 128u +
                (size_t)hv_idx * 128u + (size_t)col] =
                attn_partial * q_scale;
    }

#pragma unroll
    for (int r = 0; r < 4; r++) {
        int row = r * 32 + lane;
        state[state_base + (size_t)col * 128u + (size_t)row] = s_shard[r];
    }
}

static __global__ void ssm_prefill_gate_kernel(
    float *out, const float *z, const float *norm_w, int head_v_dim,
    int num_v_heads, int n_tokens, float eps) {
    int hv_idx = blockIdx.x;
    int tok = blockIdx.y;
    int tid = threadIdx.x;
    if (hv_idx >= num_v_heads || tok >= n_tokens) return;
    extern __shared__ float scratch[];
    size_t base = ((size_t)tok * num_v_heads + hv_idx) * head_v_dim;
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
        void *key_cache, void *value_cache, const float *k, const float *v,
        int n_tokens, int kv_dim, int kv_cache_stride,
        uint32_t kv_cache_off, int kv_f16) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * kv_dim;
    if (idx >= total) return;
    int t = idx / kv_dim;
    int d = idx - t * kv_dim;
    size_t dst = (size_t)kv_cache_off + (size_t)t * kv_cache_stride +
                 (size_t)d;
    cuda_kv_store(key_cache, dst, k[idx], kv_f16);
    cuda_kv_store(value_cache, dst, v[idx], kv_f16);
}

static __global__ void kv_store_vector_kernel(void *cache, const float *src,
                                              int n, size_t offset,
                                              int kv_f16) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cuda_kv_store(cache, offset + (size_t)i, src[i], kv_f16);
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

static __global__ void ffn_activation_batch_to_f16_kernel(
    __half *out, const float *gate_up, int hidden_dim, int n_tokens,
    int act_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_dim * n_tokens;
    if (i >= total) return;
    int token = i / hidden_dim;
    int h = i - token * hidden_dim;
    size_t base = (size_t)token * hidden_dim + h;
    float gate = gate_up[base];
    float up = gate_up[(size_t)n_tokens * hidden_dim + base];
    float v = act_type == 0
        ? (gate / (1.0f + __expf(-gate))) * up
        : gate * up;
    out[base] = __float2half_rn(v);
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

static __global__ void ffn_activation_batch_stacked_to_f16_kernel(
    __half *out, const float *gate_up, int hidden_dim, int n_tokens,
    int act_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_dim * n_tokens;
    if (i >= total) return;
    int token = i / hidden_dim;
    int h = i - token * hidden_dim;
    size_t src = (size_t)token * (size_t)hidden_dim * 2u + (size_t)h;
    float gate = gate_up[src];
    float up = gate_up[src + (size_t)hidden_dim];
    float v = act_type == 0
        ? (gate / (1.0f + __expf(-gate))) * up
        : gate * up;
    out[(size_t)token * hidden_dim + h] = __float2half_rn(v);
}

static __global__ void ffn_activation_batch_stacked_f16_to_f16_kernel(
    __half *out, const __half *gate_up, int hidden_dim, int n_tokens,
    int act_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_dim * n_tokens;
    if (i >= total) return;
    int token = i / hidden_dim;
    int h = i - token * hidden_dim;
    size_t src = (size_t)token * (size_t)hidden_dim * 2u + (size_t)h;
    float gate = __half2float(gate_up[src]);
    float up = __half2float(gate_up[src + (size_t)hidden_dim]);
    float v = act_type == 0
        ? (gate / (1.0f + __expf(-gate))) * up
        : gate * up;
    out[(size_t)token * hidden_dim + h] = __float2half_rn(v);
}

static __global__ void moe_gather_kernel(float *dst, const float *src,
                                         const int *token_ids,
                                         int offset, int n_assignments,
                                         int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_assignments * dim;
    if (idx >= total) return;
    int row = idx / dim;
    int col = idx - row * dim;
    int token = token_ids[offset + row];
    dst[(size_t)row * dim + col] = src[(size_t)token * dim + col];
}

static __global__ void moe_scatter_add_kernel(float *dst,
                                              const float *src,
                                              const int *token_ids,
                                              const float *weights,
                                              int offset,
                                              int n_assignments,
                                              int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_assignments * dim;
    if (idx >= total) return;
    int row = idx / dim;
    int col = idx - row * dim;
    int flat = offset + row;
    int token = token_ids[flat];
    float w = weights[flat];
    atomicAdd(dst + (size_t)token * dim + col,
              w * src[(size_t)row * dim + col]);
}

static __global__ void moe_identity_weighted_add_kernel(float *dst,
                                                        const float *src,
                                                        const float *weights,
                                                        int offset,
                                                        int total,
                                                        int dim,
                                                        int set_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row = idx / dim;
    float v = weights[offset + row] * src[idx];
    if (set_output)
        dst[idx] = v;
    else
        dst[idx] += v;
}

static __global__ void moe_gather_sorted_f16_kernel(
    __half *dst, const float *src, const int *slot_order,
    const int *expert_offsets, const int *expert_counts,
    const int *active_experts, int n_active, int max_count, int dim, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_active * max_count * dim;
    if (idx >= total) return;
    int col = idx % dim;
    int j = (idx / dim) % max_count;
    int a = idx / (dim * max_count);
    int expert = active_experts[a];
    int count = expert_counts[expert];
    if (j >= count) return;
    int route_pos = slot_order[expert_offsets[expert] + j];
    int token = route_pos / k;
    dst[(size_t)a * (size_t)max_count * (size_t)dim +
        (size_t)j * (size_t)dim + (size_t)col] =
        __float2half_rn(src[(size_t)token * (size_t)dim + (size_t)col]);
}

static __global__ void moe_gateup_grouped_to_f16_kernel(
    __half *mid, const float *gate, const float *up,
    const int *active_experts, const int *expert_counts,
    int n_active, int max_count, int hidden, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_active * max_count * hidden;
    if (idx >= total) return;
    int h = idx % hidden;
    int j = (idx / hidden) % max_count;
    int a = idx / (hidden * max_count);
    int expert = active_experts[a];
    if (j >= expert_counts[expert]) return;
    size_t off = (size_t)a * (size_t)max_count * (size_t)hidden +
                 (size_t)j * (size_t)hidden + (size_t)h;
    float gv = gate[off];
    float uv = up[off];
    float v = act_type == 0 ? (gv / (1.0f + __expf(-gv))) * uv : gv * uv;
    mid[off] = __float2half_rn(v);
}

static __global__ void moe_gateup_grouped_to_route_float_kernel(
    float *mid, const float *gate, const float *up,
    const int *slot_order, const int *expert_offsets,
    const int *expert_counts, const int *active_experts,
    int n_active, int max_count, int hidden, int k, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_active * max_count * hidden;
    if (idx >= total) return;
    int h = idx % hidden;
    int j = (idx / hidden) % max_count;
    int a = idx / (hidden * max_count);
    int expert = active_experts[a];
    if (j >= expert_counts[expert]) return;
    int route_pos = slot_order[expert_offsets[expert] + j];
    size_t off = (size_t)a * (size_t)max_count * (size_t)hidden +
                 (size_t)j * (size_t)hidden + (size_t)h;
    float gv = gate[off];
    float uv = up[off];
    mid[(size_t)route_pos * (size_t)hidden + (size_t)h] =
        act_type == 0 ? (gv / (1.0f + __expf(-gv))) * uv : gv * uv;
    (void)k;
}

static __global__ void moe_gateup_grouped_f16_to_f16_kernel(
    __half *mid, const __half *gate, const __half *up,
    const int *active_experts, const int *expert_counts,
    int n_active, int max_count, int hidden, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_active * max_count * hidden;
    if (idx >= total) return;
    int h = idx % hidden;
    int j = (idx / hidden) % max_count;
    int a = idx / (hidden * max_count);
    int expert = active_experts[a];
    if (j >= expert_counts[expert]) return;
    size_t off = (size_t)a * (size_t)max_count * (size_t)hidden +
                 (size_t)j * (size_t)hidden + (size_t)h;
    float gv = __half2float(gate[off]);
    float uv = __half2float(up[off]);
    float v = act_type == 0 ? (gv / (1.0f + __expf(-gv))) * uv : gv * uv;
    mid[off] = __float2half_rn(v);
}

static __global__ void moe_all2_gather_f16_kernel(
    __half *dst, const float *src, int n_tokens, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 2 * n_tokens * dim;
    if (idx >= total) return;
    int col = idx % dim;
    int token = (idx / dim) % n_tokens;
    dst[idx] = __float2half_rn(src[(size_t)token * (size_t)dim +
                                   (size_t)col]);
}

static __global__ void moe_all2_gateup_f16_to_f16_kernel(
    __half *mid, const __half *gate, const __half *up,
    int n_tokens, int hidden, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 2 * n_tokens * hidden;
    if (idx >= total) return;
    float gv = __half2float(gate[idx]);
    float uv = __half2float(up[idx]);
    float v = act_type == 0 ? (gv / (1.0f + __expf(-gv))) * uv : gv * uv;
    mid[idx] = __float2half_rn(v);
}

static __global__ void moe_all2_gateup_f16_to_f16_decode_kernel(
    __half *mid, const __half *gate, const __half *up,
    int hidden, int exact_silu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 2 * hidden;
    if (idx >= total) return;
    float gv = __half2float(gate[idx]);
    float uv = __half2float(up[idx]);
    mid[idx] = __float2half_rn(cuda_silu_select(gv, exact_silu) * uv);
}

static __global__ void moe_all2_scatter_route_kernel(
    float *dst, const float *src, const float *route, int dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= dim) return;
    float w0 = 0.0f;
    float w1 = 0.0f;
#pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        int expert = (int)(route[2 + slot] + 0.5f);
        if (expert == 0)
            w0 = route[slot];
        else
            w1 = route[slot];
    }
    dst[col] = w0 * src[col] + w1 * src[(size_t)dim + (size_t)col];
}

static __global__ void moe_pack_all2_route_kernel(
    float *route, const int *indices, const float *weights) {
    route[0] = weights[0];
    route[1] = weights[1];
    route[2] = (float)indices[0];
    route[3] = (float)indices[1];
}

static __global__ void moe_all2_scatter_kernel(
    float *dst, const float *src, const int *indices, const float *weights,
    int n_tokens, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * dim;
    if (idx >= total) return;
    int col = idx % dim;
    int token = idx / dim;
    int i0 = indices[token * 2 + 0];
    float w0 = weights[token * 2 + 0];
    float w1 = weights[token * 2 + 1];
    if (i0 != 0) {
        float tmp = w0;
        w0 = w1;
        w1 = tmp;
    }
    size_t off = (size_t)token * (size_t)dim + (size_t)col;
    dst[off] += w0 * src[off] +
                w1 * src[(size_t)n_tokens * (size_t)dim + off];
}

static __global__ void moe_scatter_sorted_grouped_kernel(
    float *dst, const float *src, const int *slot_order,
    const int *expert_offsets, const int *expert_counts,
    const int *active_experts, const float *weights,
    int n_active, int max_count, int dim, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_active * max_count * dim;
    if (idx >= total) return;
    int col = idx % dim;
    int j = (idx / dim) % max_count;
    int a = idx / (dim * max_count);
    int expert = active_experts[a];
    int count = expert_counts[expert];
    if (j >= count) return;
    int route_pos = slot_order[expert_offsets[expert] + j];
    int token = route_pos / k;
    float w = weights[route_pos];
    size_t src_off = (size_t)a * (size_t)max_count * (size_t)dim +
                     (size_t)j * (size_t)dim + (size_t)col;
    atomicAdd(dst + (size_t)token * (size_t)dim + (size_t)col,
              w * src[src_off]);
}

static int cuda_type_supported(int type) {
    return !bn_gpu_policy_cuda_matvec_disabled() &&
           bn_gpu_policy_cuda_matvec_type_supported(type);
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

static int cuda_ensure_resident_input(BnCudaCtx *ctx, size_t bytes) {
    if (!ctx || bytes == 0) return -1;
    if (cuda_ctx_set_device(ctx) != 0) return -1;
    if (bytes <= ctx->d_resident_in_bytes) return 0;
    if (ctx->d_resident_in) cudaFree(ctx->d_resident_in);
    ctx->d_resident_in = NULL;
    ctx->d_resident_in_bytes = 0;
    cudaError_t err = cudaMalloc(&ctx->d_resident_in, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] resident input scratch alloc failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    ctx->d_resident_in_bytes = bytes;
    return 0;
}

static int cuda_prepare_resident_input(BnCudaCtx *ctx, size_t input_bytes,
                                       size_t next_out_bytes,
                                       const float **d_input) {
    if (!ctx || !ctx->d_out || !d_input || input_bytes == 0)
        return -1;
    if (next_out_bytes <= ctx->d_out_bytes) {
        *d_input = ctx->d_out;
        return 0;
    }
    if (cuda_ensure_resident_input(ctx, input_bytes) != 0)
        return -1;
    cudaError_t err = cudaMemcpy(ctx->d_resident_in, ctx->d_out,
                                 input_bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] resident input preserve failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    *d_input = ctx->d_resident_in;
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

static int cuda_ensure_q8_0_f32(BnCudaCtx *ctx, int cols, int n_tokens) {
    if (!ctx || cols <= 0 || n_tokens <= 0) return -1;
    size_t n_blocks = (((size_t)cols + 31u) / 32u) * (size_t)n_tokens;
    size_t bytes = n_blocks * sizeof(BnCudaBlockQ8_0F32);
    if (bytes <= ctx->d_q8_0_f32_bytes) return 0;
    if (ctx->d_q8_0_f32) cudaFree(ctx->d_q8_0_f32);
    ctx->d_q8_0_f32 = NULL;
    ctx->d_q8_0_f32_bytes = 0;
    cudaError_t err = cudaMalloc(&ctx->d_q8_0_f32, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] q8_0 f32 scratch alloc failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    ctx->d_q8_0_f32_bytes = bytes;
    return 0;
}

static int cuda_use_q8_0_quant_matmul(void) {
    return bn_gpu_policy_cuda_q8_0_quant_matmul_enabled();
}

static int cuda_use_f16_q8_0_matmul(void) {
    return bn_gpu_policy_cuda_f16_q8_0_matmul_enabled();
}

static int cuda_use_q8_0_preq_split(void) {
    return bn_gpu_policy_cuda_q8_0_preq_split_enabled();
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

static int cuda_launch_q8_0_matmul(BnCudaCtx *ctx, float *d_dst,
                                   const BnBlockQ8_0 *blocks,
                                   const float *d_x, int rows, int cols,
                                   int n_tokens, size_t out_offset) {
    if (!ctx || !d_dst || !blocks || !d_x || rows <= 0 || cols <= 0 ||
        n_tokens <= 0 || (cols & 31) != 0)
        return -1;
    int threads = 256;
    int warps = threads / 32;
    if (cuda_use_q8_0_quant_matmul()) {
        int x_blocks = cols / 32;
        if (cuda_ensure_q8_0_f32(ctx, cols, n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_0F32 *xq = (BnCudaBlockQ8_0F32 *)ctx->d_q8_0_f32;
        quantize_q8_0_f32_batch_kernel<<<dim3(x_blocks, n_tokens, 1), 32, 0>>>(
            xq, d_x, cols, n_tokens);
        dim3 grid(((rows + 1) / 2 + warps - 1) / warps, n_tokens, 1);
        q8_0_q8f32_matmul4_warp_kernel<<<grid, threads, 0>>>(
            d_dst, blocks, xq, rows, cols, n_tokens, out_offset);
        return 0;
    }

    dim3 grid(((rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
    q8_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
        d_dst, blocks, d_x, rows, cols, n_tokens, out_offset);
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
    if (!ctx->h_argmax) {
        cudaError_t host_err =
            cudaMallocHost((void **)&ctx->h_argmax, sizeof(int));
        if (host_err != cudaSuccess) {
            fprintf(stderr,
                    "[bn:gpu:cuda] pinned argmax host alloc failed: %s\n",
                    cudaGetErrorString(host_err));
            return -1;
        }
    }
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

static int cuda_ensure_argmax_partials(BnCudaCtx *ctx, int n_partials,
                                        int n_penalty_tokens) {
    if (!ctx || n_partials <= 0) return -1;
    if (cuda_ctx_set_device(ctx) != 0) return -1;
    if (!ctx->h_argmax) {
        cudaError_t host_err =
            cudaMallocHost((void **)&ctx->h_argmax, sizeof(int));
        if (host_err != cudaSuccess) {
            fprintf(stderr,
                    "[bn:gpu:cuda] pinned argmax host alloc failed: %s\n",
                    cudaGetErrorString(host_err));
            return -1;
        }
    }
    size_t bytes = (size_t)n_partials * sizeof(BnCudaArgmaxPair) +
                   sizeof(int);
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
    err = cudaMemset(ctx->act_bufs[idx], 0, aligned);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] activation zero failed idx=%d bytes=%zu: %s\n",
                idx, aligned, cudaGetErrorString(err));
        cudaFree(ctx->act_bufs[idx]);
        ctx->act_bufs[idx] = NULL;
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
    ctx->has_moe_model = c->n_experts > 0;

    int n_attn = bn_gpu_policy_attention_layer_count(c);
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
        if (bn_gpu_policy_uses_hybrid_moe(c)) {
            size_t hybrid_q_size = (size_t)(4 * q_dim) * sizeof(float);
            if (hybrid_q_size > gated_q_size)
                gated_q_size = hybrid_q_size;
        }
        sizes[BN_GPU_VALUE_QKV] = qkv_size > gated_q_size
            ? qkv_size
            : gated_q_size;
    }
    if (c->moe_intermediate_size > 0) {
        int moe_scratch = c->moe_intermediate_size;
        if (c->n_experts > moe_scratch)
            moe_scratch = c->n_experts;
        if (2 * c->n_experts_active > moe_scratch)
            moe_scratch = 2 * c->n_experts_active;
        if (c->n_experts_active > 0 &&
            c->moe_intermediate_size <= INT_MAX / c->n_experts_active &&
            c->moe_intermediate_size * c->n_experts_active > moe_scratch)
            moe_scratch = c->moe_intermediate_size * c->n_experts_active;
        sizes[BN_GPU_VALUE_MOE_HB] =
            (size_t)moe_scratch * sizeof(float);
        sizes[BN_GPU_VALUE_MOE_HB2] =
            (size_t)moe_scratch * sizeof(float);
        sizes[BN_GPU_VALUE_MOE_OUT] =
            (size_t)c->dim * sizeof(float);
    }
    if (bn_gpu_policy_uses_hybrid_ssm(c)) {
        int n_ssm = bn_gpu_policy_ssm_layer_count(c);
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
        if (value_dim > c->dim)
            sizes[BN_GPU_VALUE_XB2] = (size_t)value_dim * sizeof(float);
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
    if (!ctx || !out || buf_idx < 0 || buf_idx >= BN_GPU_VALUE_COUNT) {
        if (getenv("BN_CUDA_DEBUG_READBACK")) {
            fprintf(stderr,
                    "[bn:gpu:cuda] read_activation invalid args: ctx=%p "
                    "out=%p buf=%d size=%zu offset=%zu\n",
                    (void *)ctx, out, buf_idx, size, offset);
        }
        return -1;
    }
    if (cuda_ctx_set_device(ctx) != 0) return -1;
    if (!ctx->act_bufs[buf_idx] || offset + size > ctx->act_sizes[buf_idx]) {
        if (getenv("BN_CUDA_DEBUG_READBACK")) {
            fprintf(stderr,
                    "[bn:gpu:cuda] read_activation invalid buffer: buf=%d "
                    "ptr=%p size=%zu offset=%zu alloc=%zu\n",
                    buf_idx, ctx->act_bufs[buf_idx], size, offset,
                    ctx->act_sizes[buf_idx]);
        }
        return -1;
    }
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
    if (err != cudaSuccess && getenv("BN_CUDA_DEBUG_READBACK")) {
        fprintf(stderr,
                "[bn:gpu:cuda] read_activation failed: buf=%d size=%zu "
                "offset=%zu err=%s\n",
                buf_idx, size, offset, cudaGetErrorString(err));
    }
    return err == cudaSuccess ? 0 : -1;
}

static int cuda_buffer_create_f16_cache(BnCudaBuffer *buf,
                                        int aux_cache_mode) {
    if (!buf || !buf->data || buf->rows <= 0 || buf->cols <= 0 ||
        (buf->cols & 31) != 0)
        return 0;
    if (!bn_gpu_policy_cuda_cublas_aux_cache_supported(buf->type, buf->cols))
        return 0;

    int force_q6_f32 = aux_cache_mode == 2;
    int force_f16 = aux_cache_mode == 3;
    int force_q4_f32 =
        aux_cache_mode == 2 && buf->type == BN_GGUF_TENSOR_Q4_K;
    int q6_as_f16 = buf->type == BN_GGUF_TENSOR_Q6_K &&
                    (force_f16 || !force_q6_f32) &&
                    bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled();
    int add_q6_f32_cache =
        force_f16 && buf->type == BN_GGUF_TENSOR_Q6_K &&
        bn_gpu_policy_cuda_q6k_f16_cache_adds_f32_down_cache();
    size_t n = (size_t)buf->rows * (size_t)buf->cols;
    int f32_cache =
        force_q4_f32 || (buf->type == BN_GGUF_TENSOR_Q6_K && !q6_as_f16);
    size_t bytes = n * (f32_cache
                        ? sizeof(float)
                        : sizeof(__half));
    int max_mb = bn_gpu_policy_cuda_cublas_aux_cache_max_mb(
        buf->type, force_q6_f32, force_f16);
    if (max_mb > 0 && bytes > (size_t)max_mb * 1024u * 1024u)
        return 0;
    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        int reserve_mb = cuda_env_int("BN_CUDA_CUBLAS_CACHE_RESERVE_MB", 4096);
        size_t reserve = reserve_mb > 0
            ? (size_t)reserve_mb * 1024u * 1024u : 0u;
        if (free_mem <= bytes + reserve)
            return 0;
    }
    void **cache_ptr = f32_cache
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
    if (buf->type == BN_GGUF_TENSOR_BF16) {
        dequant_bf16_to_f16_kernel<<<blocks, threads>>>(
            (__half *)buf->f16_data, (const uint16_t *)buf->data,
            buf->rows, buf->cols);
    } else if (buf->type == BN_GGUF_TENSOR_Q8_0) {
        dequant_q8_0_to_f16_kernel<<<blocks, threads>>>(
            (__half *)buf->f16_data, (const BnBlockQ8_0 *)buf->data,
            buf->rows, buf->cols);
    } else if (buf->type == BN_GGUF_TENSOR_Q5_0) {
        dequant_q5_0_to_f16_kernel<<<blocks, threads>>>(
            (__half *)buf->f16_data, (const BnBlockQ5_0 *)buf->data,
            buf->rows, buf->cols);
    } else if (buf->type == BN_GGUF_TENSOR_Q3_K) {
        dequant_q3k_to_f16_kernel<<<blocks, threads>>>(
            (__half *)buf->f16_data, (const BnBlockQ3K *)buf->data,
            buf->rows, buf->cols);
    } else if (buf->type == BN_GGUF_TENSOR_Q4_K && force_q4_f32) {
        dequant_q4k_to_f32_kernel<<<blocks, threads>>>(
            (float *)buf->f32_data, (const BnBlockQ4K *)buf->data,
            buf->rows, buf->cols);
    } else if (buf->type == BN_GGUF_TENSOR_Q4_K) {
        dequant_q4k_to_f16_kernel<<<blocks, threads>>>(
            (__half *)buf->f16_data, (const BnBlockQ4K *)buf->data,
            buf->rows, buf->cols);
    } else if (buf->type == BN_GGUF_TENSOR_Q5_K) {
        dequant_q5k_to_f16_kernel<<<blocks, threads>>>(
            (__half *)buf->f16_data, (const BnBlockQ5K *)buf->data,
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
                f32_cache ? "f32" : "f16");
    if (f32_cache)
        buf->f32_size = bytes;
    else
        buf->f16_size = bytes;
    if (add_q6_f32_cache && !buf->f32_data) {
        size_t f32_bytes = n * sizeof(float);
        size_t free_mem2 = 0;
        size_t total_mem2 = 0;
        if (cudaMemGetInfo(&free_mem2, &total_mem2) == cudaSuccess) {
            int reserve_mb =
                cuda_env_int("BN_CUDA_CUBLAS_CACHE_RESERVE_MB", 4096);
            size_t reserve = reserve_mb > 0
                ? (size_t)reserve_mb * 1024u * 1024u : 0u;
            if (free_mem2 <= f32_bytes + reserve)
                return 0;
        }
        cudaError_t f32_err = cudaMalloc(&buf->f32_data, f32_bytes);
        if (f32_err != cudaSuccess) {
            if (getenv("BN_CUDA_DEBUG_CUBLAS_CACHE"))
                fprintf(stderr,
                        "[bn:gpu:cuda] q6 f32 companion cache skipped "
                        "rows=%d cols=%d bytes=%zu: %s\n",
                        buf->rows, buf->cols, f32_bytes,
                        cudaGetErrorString(f32_err));
            buf->f32_data = NULL;
            return 0;
        }
        dequant_q6k_to_f32_kernel<<<blocks, threads>>>(
            (float *)buf->f32_data, (const BnBlockQ6K *)buf->data,
            buf->rows, buf->cols);
        f32_err = cudaGetLastError();
        if (f32_err != cudaSuccess) {
            if (getenv("BN_CUDA_DEBUG_CUBLAS_CACHE"))
                fprintf(stderr,
                        "[bn:gpu:cuda] q6 f32 companion dequant skipped "
                        "rows=%d cols=%d: %s\n",
                        buf->rows, buf->cols,
                        cudaGetErrorString(f32_err));
            cudaFree(buf->f32_data);
            buf->f32_data = NULL;
            return 0;
        }
        buf->f32_size = f32_bytes;
        if (getenv("BN_CUDA_DEBUG_CUBLAS_CACHE"))
            fprintf(stderr,
                    "[bn:gpu:cuda] q6 f32 companion cache ready rows=%d "
                    "cols=%d bytes=%zu\n",
                    buf->rows, buf->cols, f32_bytes);
    }
    return 0;
}

static int cuda_buffer_create_iq_f16_cache(BnCudaBuffer *buf,
                                           const void *host_data) {
    if (!buf || !host_data || !buf->data || buf->rows <= 0 ||
        buf->cols <= 0 || (buf->cols % BN_QK_K) != 0)
        return 0;
    if (buf->type != BN_GGUF_TENSOR_IQ3_XXS &&
        buf->type != BN_GGUF_TENSOR_IQ4_XS)
        return 0;
    if (!bn_gpu_policy_cuda_cublas_matmul_enabled())
        return 0;

    size_t n = (size_t)buf->rows * (size_t)buf->cols;
    size_t bytes = n * sizeof(uint16_t);
    int max_mb = bn_gpu_policy_cuda_cublas_cache_max_mb(128, 0);
    if (max_mb > 0 && bytes > (size_t)max_mb * 1024u * 1024u)
        return 0;
    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        int reserve_mb = cuda_env_int("BN_CUDA_CUBLAS_CACHE_RESERVE_MB", 4096);
        size_t reserve = reserve_mb > 0
            ? (size_t)reserve_mb * 1024u * 1024u : 0u;
        if (free_mem <= bytes + reserve)
            return 0;
    }

    uint16_t *h_f16 = (uint16_t *)malloc(bytes);
    if (!h_f16)
        return 0;
    float tmp[BN_QK_K];
    int blocks_per_row = buf->cols / BN_QK_K;
    for (int r = 0; r < buf->rows; r++) {
        for (int b = 0; b < blocks_per_row; b++) {
            size_t block_idx = (size_t)r * (size_t)blocks_per_row + (size_t)b;
            if (buf->type == BN_GGUF_TENSOR_IQ3_XXS) {
                const BnBlockIQ3XXS *blocks =
                    (const BnBlockIQ3XXS *)host_data;
                bn_quant_dequant_iq3xxs(&blocks[block_idx], tmp);
            } else {
                const BnBlockIQ4XS *blocks =
                    (const BnBlockIQ4XS *)host_data;
                bn_quant_dequant_iq4xs(&blocks[block_idx], tmp);
            }
            size_t out = ((size_t)r * (size_t)buf->cols +
                          (size_t)b * BN_QK_K);
            for (int i = 0; i < BN_QK_K; i++)
                h_f16[out + (size_t)i] = bn_fp32_to_fp16(tmp[i]);
        }
    }

    cudaError_t err = cudaMalloc(&buf->f16_data, bytes);
    if (err == cudaSuccess) {
        err = cudaMemcpy(buf->f16_data, h_f16, bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            buf->f16_size = bytes;
    }
    free(h_f16);
    if (err != cudaSuccess) {
        if (buf->f16_data)
            cudaFree(buf->f16_data);
        buf->f16_data = NULL;
        buf->f16_size = 0;
        if (getenv("BN_CUDA_DEBUG_CUBLAS_CACHE"))
            fprintf(stderr,
                    "[bn:gpu:cuda] iq f16 cache skipped type=%d rows=%d "
                    "cols=%d bytes=%zu: %s\n",
                    buf->type, buf->rows, buf->cols, bytes,
                    cudaGetErrorString(err));
    } else if (getenv("BN_CUDA_DEBUG_CUBLAS_CACHE")) {
        fprintf(stderr,
                "[bn:gpu:cuda] iq f16 cache ready type=%d rows=%d cols=%d "
                "bytes=%zu\n",
                buf->type, buf->rows, buf->cols, bytes);
    }
    return 0;
}

static void *cuda_buffer_create_impl(void *vctx, const void *data, size_t size,
                                     int type, int rows, int cols,
                                     int create_aux_cache) {
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
    if (create_aux_cache) {
        if (type == BN_GGUF_TENSOR_IQ3_XXS ||
            type == BN_GGUF_TENSOR_IQ4_XS)
            cuda_buffer_create_iq_f16_cache(buf, data);
        else
            cuda_buffer_create_f16_cache(buf, create_aux_cache);
    }
    return buf;
}

static void *cuda_buffer_create(void *vctx, const void *data, size_t size,
                                int type, int rows, int cols) {
    return cuda_buffer_create_impl(vctx, data, size, type, rows, cols, 1);
}

static void *cuda_buffer_create_quant_only(void *vctx, const void *data,
                                           size_t size, int type, int rows,
                                           int cols) {
    return cuda_buffer_create_impl(vctx, data, size, type, rows, cols, 0);
}

static void *cuda_buffer_create_q6_f32_cache(void *vctx, const void *data,
                                             size_t size, int type, int rows,
                                             int cols) {
    return cuda_buffer_create_impl(vctx, data, size, type, rows, cols, 2);
}

static void *cuda_buffer_create_force_f16_cache(void *vctx, const void *data,
                                                size_t size, int type,
                                                int rows, int cols) {
    return cuda_buffer_create_impl(vctx, data, size, type, rows, cols, 3);
}

static void *cuda_buffer_create_stacked2(void *vctx,
                                         const void *data0, size_t size0,
                                         const void *data1, size_t size1,
                                         int type, int rows, int cols) {
    if (!data0 || !data1 || size0 == 0 || size1 == 0)
        return NULL;
    size_t size = size0 + size1;
    uint8_t *combined = (uint8_t *)malloc(size);
    if (!combined)
        return NULL;
    memcpy(combined, data0, size0);
    memcpy(combined + size0, data1, size1);
    void *buf = cuda_buffer_create_impl(vctx, combined, size, type, rows,
                                        cols, 1);
    free(combined);
    return buf;
}

static void *cuda_buffer_create_stacked3(void *vctx,
                                         const void *data0, size_t size0,
                                         const void *data1, size_t size1,
                                         const void *data2, size_t size2,
                                         int type, int rows, int cols) {
    if (!data0 || !data1 || !data2 || size0 == 0 || size1 == 0 ||
        size2 == 0)
        return NULL;
    size_t size = size0 + size1 + size2;
    uint8_t *combined = (uint8_t *)malloc(size);
    if (!combined)
        return NULL;
    memcpy(combined, data0, size0);
    memcpy(combined + size0, data1, size1);
    memcpy(combined + size0 + size1, data2, size2);
    void *buf = cuda_buffer_create_impl(vctx, combined, size, type, rows,
                                        cols, 1);
    free(combined);
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

static int cuda_force_quant_matmul_for_type(int type) {
    return (type == BN_GGUF_TENSOR_Q8_0 && !cuda_use_f16_q8_0_matmul()) ||
           (type == BN_GGUF_TENSOR_Q4_K &&
            getenv("BN_CUDA_FORCE_Q4K_QUANT_MATMUL") != NULL) ||
           (type == BN_GGUF_TENSOR_Q6_K &&
            getenv("BN_CUDA_FORCE_Q6K_QUANT_MATMUL") != NULL);
}

static int cuda_use_q6k_4warp_long(int rows, int cols) {
    if (getenv("BN_CUDA_DISABLE_Q6K_4WARP_LONG") != NULL)
        return 0;
    if (rows == 1536 && cols == 8960 &&
        getenv("BN_CUDA_DISABLE_Q6K_4WARP_1536_8960") == NULL)
        return 1;
    if (rows >= 2560 && cols >= 8192 && cols <= 16384)
        return 1;
    return rows >= 2560 && cols >= 5120 && cols < 8192 &&
           getenv("BN_CUDA_ENABLE_Q6K_4WARP_5120") != NULL;
}

static int cuda_use_q6k_5warp_exact(int rows, int cols) {
    if (rows == 1536 && cols == 8960 &&
        getenv("BN_CUDA_DISABLE_Q6K_5WARP_1536_8960") == NULL)
        return 1;
    if (rows == 2560 && cols == 9728 &&
        getenv("BN_CUDA_DISABLE_Q6K_5WARP_2560_9728") == NULL)
        return 1;
    return 0;
}

static int cuda_use_q6k_3warp_exact(int rows, int cols) {
    if (rows == 1536 && cols == 8960 &&
        getenv("BN_CUDA_DISABLE_Q6K_3WARP_1536_8960") == NULL)
        return 1;
    if (rows == 2560 && cols == 9728 &&
        getenv("BN_CUDA_DISABLE_Q6K_3WARP_2560_9728") == NULL)
        return 1;
    return 0;
}

static int cuda_use_q6k_2warp_long(int rows, int cols) {
    return rows >= 2560 && cols >= 8192 && cols <= 12288 &&
           getenv("BN_CUDA_ENABLE_Q6K_2WARP_LONG") != NULL &&
           getenv("BN_CUDA_DISABLE_Q6K_2WARP_LONG") == NULL;
}

static int cuda_disable_q6k_matvec4_shape(int rows, int cols) {
    if (rows == 1024 && cols == 2560 &&
        getenv("BN_CUDA_ENABLE_Q6K_MATVEC4_1024_2560") == NULL)
        return 1;
    if (rows == 512 && cols == 2048 &&
        getenv("BN_CUDA_ENABLE_Q6K_MATVEC4_512_2048") == NULL)
        return 1;
    return 0;
}

static int cuda_prefer_q6k_moe_quant_down(int routed_q4, int down_type,
                                          int hidden_dim, int n_experts,
                                          int k) {
    return routed_q4 && down_type == BN_GGUF_TENSOR_Q6_K &&
           hidden_dim <= 1024 && (n_experts > 2 || k > 2) &&
           getenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL;
}

static cublasGemmAlgo_t cuda_cublas_gemm_algo(void) {
    const char *env = getenv("BN_CUDA_CUBLAS_GEMM_ALGO");
    if (!env || !env[0])
        return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env || *end != '\0')
        return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    if (v < 0)
        return CUBLAS_GEMM_DEFAULT;
    if (v >= 0 && v <= 23)
        return (cublasGemmAlgo_t)((int)CUBLAS_GEMM_ALGO0_TENSOR_OP + (int)v);
    return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
}

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
        if (ctx->exec_stream)
            cublasSetStream(ctx->cublas, ctx->exec_stream);
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
    f32_to_f16_kernel<<<blocks, threads, 0, ctx->exec_stream>>>(
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
    f32_to_f16_kernel<<<blocks, threads, 0, ctx->exec_stream>>>(
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
        rows <= 0 || cols <= 0 || n_tokens <= 0)
        return -1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (ctx->exec_stream)
        cublasSetStream(ctx->cublas, ctx->exec_stream);
    cublasStatus_t st = cublasGemmEx(
        ctx->cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        rows, n_tokens, cols,
        &alpha,
        w->f16_data, CUDA_R_16F, cols,
        d_x_f16, CUDA_R_16F, cols,
        &beta,
        d_out, CUDA_R_32F, rows,
        CUBLAS_COMPUTE_32F_FAST_16F,
        cuda_cublas_gemm_algo());
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[bn:gpu:cuda] cublas matmul failed: status %d\n",
                (int)st);
        return -1;
    }
    return 0;
}

static int cuda_cublas_matmul_f16_preconverted_out_f16(
        BnCudaCtx *ctx, void *d_out, const BnCudaBuffer *w,
        const void *d_x_f16, int rows, int cols, int n_tokens) {
    if (!ctx || !ctx->cublas || !d_out || !w || !w->f16_data || !d_x_f16 ||
        rows <= 0 || cols <= 0 || n_tokens <= 0)
        return -1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (ctx->exec_stream)
        cublasSetStream(ctx->cublas, ctx->exec_stream);
    cublasStatus_t st = cublasGemmEx(
        ctx->cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        rows, n_tokens, cols,
        &alpha,
        w->f16_data, CUDA_R_16F, cols,
        d_x_f16, CUDA_R_16F, cols,
        &beta,
        d_out, CUDA_R_16F, rows,
        CUBLAS_COMPUTE_32F_FAST_16F,
        cuda_cublas_gemm_algo());
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[bn:gpu:cuda] cublas matmul f16 out failed: status %d\n",
                (int)st);
        return -1;
    }
    return 0;
}

static int cuda_moe_cublas_all2_prefill(
    BnCudaCtx *ctx, float *d_full_out,
    const BnCudaBuffer *gate, const BnCudaBuffer *up,
    const BnCudaBuffer *down, const float *d_full_x,
    const float *d_weights, const int *d_indices,
    int n_tokens, int dim, int hidden_dim, int act_type) {
    if (!ctx || !ctx->cublas || !d_full_out || !gate || !up || !down ||
        !gate->f16_data || !up->f16_data || !down->f16_data ||
        !d_full_x || !d_weights || !d_indices ||
        n_tokens <= 1 || dim <= 0 || hidden_dim <= 0)
        return -1;

    size_t gather_values = (size_t)2 * (size_t)n_tokens * (size_t)dim;
    size_t hidden_values = (size_t)2 * (size_t)n_tokens *
                           (size_t)hidden_dim;
    size_t out_values = (size_t)2 * (size_t)n_tokens * (size_t)dim;
    size_t x_bytes = gather_values * sizeof(__half) +
                     hidden_values * 3u * sizeof(__half);
    size_t out_bytes = out_values * sizeof(float);
    if (cuda_ensure_scratch(ctx, x_bytes, out_bytes) != 0)
        return -1;

    __half *d_gather = (__half *)ctx->d_x;
    __half *d_mid = (__half *)((uint8_t *)ctx->d_x +
                               gather_values * sizeof(__half));
    __half *d_gate_h = d_mid + hidden_values;
    __half *d_up_h = d_gate_h + hidden_values;
    float *d_down = ctx->d_out;

    int threads = 256;
    int gather_total = (int)gather_values;
    moe_all2_gather_f16_kernel<<<(gather_total + threads - 1) / threads,
                                 threads>>>(
        d_gather, d_full_x, n_tokens, dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return -1;

    int n_active = 2;
    int n_ptrs = 9 * n_active;
    if (cuda_ensure_gemm_ptrs(ctx, n_ptrs) != 0)
        return -1;
    void **gate_a = ctx->h_gemm_ptrs;
    void **gate_b = gate_a + n_active;
    void **gate_c = gate_b + n_active;
    void **up_a = gate_c + n_active;
    void **up_b = up_a + n_active;
    void **up_c = up_b + n_active;
    void **down_a = up_c + n_active;
    void **down_b = down_a + n_active;
    void **down_c = down_b + n_active;
    for (int e = 0; e < 2; e++) {
        size_t in_off = (size_t)e * (size_t)n_tokens * (size_t)dim;
        size_t hidden_off =
            (size_t)e * (size_t)n_tokens * (size_t)hidden_dim;
        gate_a[e] = (uint8_t *)gate->f16_data +
                    (size_t)e * (size_t)hidden_dim * (size_t)dim *
                    sizeof(__half);
        gate_b[e] = d_gather + in_off;
        gate_c[e] = d_gate_h + hidden_off;
        up_a[e] = (uint8_t *)up->f16_data +
                  (size_t)e * (size_t)hidden_dim * (size_t)dim *
                  sizeof(__half);
        up_b[e] = d_gather + in_off;
        up_c[e] = d_up_h + hidden_off;
        down_a[e] = (uint8_t *)down->f16_data +
                    (size_t)e * (size_t)dim * (size_t)hidden_dim *
                    sizeof(__half);
        down_b[e] = d_mid + hidden_off;
        down_c[e] = d_down + (size_t)e * (size_t)n_tokens * (size_t)dim;
    }
    err = cudaMemcpy(ctx->d_gemm_ptrs, ctx->h_gemm_ptrs,
                     (size_t)n_ptrs * sizeof(void *),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return -1;
    void **d_gate_a = ctx->d_gemm_ptrs;
    void **d_gate_b = d_gate_a + n_active;
    void **d_gate_c = d_gate_b + n_active;
    void **d_up_a = d_gate_c + n_active;
    void **d_up_b = d_up_a + n_active;
    void **d_up_c = d_up_b + n_active;
    void **d_down_a = d_up_c + n_active;
    void **d_down_b = d_down_a + n_active;
    void **d_down_c = d_down_b + n_active;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (ctx->exec_stream)
        cublasSetStream(ctx->cublas, ctx->exec_stream);
    cublasStatus_t st = cublasGemmBatchedEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_dim, n_tokens, dim, &alpha,
        (const void *const *)d_gate_a, CUDA_R_16F, dim,
        (const void *const *)d_gate_b, CUDA_R_16F, dim,
        &beta, (void *const *)d_gate_c, CUDA_R_16F, hidden_dim,
        n_active, CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasGemmBatchedEx(
            ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_dim, n_tokens, dim, &alpha,
            (const void *const *)d_up_a, CUDA_R_16F, dim,
            (const void *const *)d_up_b, CUDA_R_16F, dim,
            &beta, (void *const *)d_up_c, CUDA_R_16F, hidden_dim,
            n_active, CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if (st != CUBLAS_STATUS_SUCCESS)
        return -1;

    int act_total = (int)hidden_values;
    moe_all2_gateup_f16_to_f16_kernel<<<
        (act_total + threads - 1) / threads, threads>>>(
        d_mid, d_gate_h, d_up_h, n_tokens, hidden_dim, act_type);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return -1;

    st = cublasGemmBatchedEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        dim, n_tokens, hidden_dim, &alpha,
        (const void *const *)d_down_a, CUDA_R_16F, hidden_dim,
        (const void *const *)d_down_b, CUDA_R_16F, hidden_dim,
        &beta, (void *const *)d_down_c, CUDA_R_32F, dim,
        n_active, CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS)
        return -1;

    int scatter_total = n_tokens * dim;
    moe_all2_scatter_kernel<<<(scatter_total + threads - 1) / threads,
                              threads>>>(
        d_full_out, d_down, d_indices, d_weights, n_tokens, dim);
    err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

static int cuda_moe_cublas_all2_decode(
    BnCudaCtx *ctx, float *out,
    const BnCudaBuffer *gate, const BnCudaBuffer *up,
    const BnCudaBuffer *down, const float *in, const float *route,
    int dim, int hidden_dim, int exact_silu) {
    if (!ctx || !ctx->cublas || !out || !gate || !up || !down ||
        !gate->f16_data || !up->f16_data || !down->f16_data ||
        !in || !route || dim <= 0 || hidden_dim <= 0)
        return -1;

    if (cuda_convert_f32_to_f16(ctx, in, (size_t)dim) != 0)
        return -1;

    size_t hidden_values = 2u * (size_t)hidden_dim;
    size_t x_bytes = hidden_values * 3u * sizeof(__half);
    size_t out_bytes = 2u * (size_t)dim * sizeof(float);
    if (cuda_ensure_scratch(ctx, x_bytes, out_bytes) != 0)
        return -1;

    __half *d_mid = (__half *)ctx->d_x;
    __half *d_gate_h = d_mid + hidden_values;
    __half *d_up_h = d_gate_h + hidden_values;
    float *d_down = ctx->d_out;

    int n_active = 2;
    int n_ptrs = 9 * n_active;
    if (cuda_ensure_gemm_ptrs(ctx, n_ptrs) != 0)
        return -1;
    void **gate_a = ctx->h_gemm_ptrs;
    void **gate_b = gate_a + n_active;
    void **gate_c = gate_b + n_active;
    void **up_a = gate_c + n_active;
    void **up_b = up_a + n_active;
    void **up_c = up_b + n_active;
    void **down_a = up_c + n_active;
    void **down_b = down_a + n_active;
    void **down_c = down_b + n_active;
    for (int e = 0; e < 2; e++) {
        size_t hidden_off = (size_t)e * (size_t)hidden_dim;
        gate_a[e] = (uint8_t *)gate->f16_data +
                    (size_t)e * (size_t)hidden_dim * (size_t)dim *
                    sizeof(__half);
        gate_b[e] = ctx->d_x_f16;
        gate_c[e] = d_gate_h + hidden_off;
        up_a[e] = (uint8_t *)up->f16_data +
                  (size_t)e * (size_t)hidden_dim * (size_t)dim *
                  sizeof(__half);
        up_b[e] = ctx->d_x_f16;
        up_c[e] = d_up_h + hidden_off;
        down_a[e] = (uint8_t *)down->f16_data +
                    (size_t)e * (size_t)dim * (size_t)hidden_dim *
                    sizeof(__half);
        down_b[e] = d_mid + hidden_off;
        down_c[e] = d_down + (size_t)e * (size_t)dim;
    }
    cudaError_t err = cudaMemcpyAsync(
        ctx->d_gemm_ptrs, ctx->h_gemm_ptrs, (size_t)n_ptrs * sizeof(void *),
        cudaMemcpyHostToDevice, ctx->exec_stream);
    if (err != cudaSuccess)
        return -1;
    void **d_gate_a = ctx->d_gemm_ptrs;
    void **d_gate_b = d_gate_a + n_active;
    void **d_gate_c = d_gate_b + n_active;
    void **d_up_a = d_gate_c + n_active;
    void **d_up_b = d_up_a + n_active;
    void **d_up_c = d_up_b + n_active;
    void **d_down_a = d_up_c + n_active;
    void **d_down_b = d_down_a + n_active;
    void **d_down_c = d_down_b + n_active;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (ctx->exec_stream)
        cublasSetStream(ctx->cublas, ctx->exec_stream);
    cublasStatus_t st = cublasGemmBatchedEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_dim, 1, dim, &alpha,
        (const void *const *)d_gate_a, CUDA_R_16F, dim,
        (const void *const *)d_gate_b, CUDA_R_16F, dim,
        &beta, (void *const *)d_gate_c, CUDA_R_16F, hidden_dim,
        n_active, CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasGemmBatchedEx(
            ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_dim, 1, dim, &alpha,
            (const void *const *)d_up_a, CUDA_R_16F, dim,
            (const void *const *)d_up_b, CUDA_R_16F, dim,
            &beta, (void *const *)d_up_c, CUDA_R_16F, hidden_dim,
            n_active, CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if (st != CUBLAS_STATUS_SUCCESS)
        return -1;

    int threads = 256;
    int act_total = (int)hidden_values;
    moe_all2_gateup_f16_to_f16_decode_kernel<<<
        (act_total + threads - 1) / threads, threads, 0, ctx->exec_stream>>>(
        d_mid, d_gate_h, d_up_h, hidden_dim, exact_silu);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return -1;

    st = cublasGemmBatchedEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        dim, 1, hidden_dim, &alpha,
        (const void *const *)d_down_a, CUDA_R_16F, hidden_dim,
        (const void *const *)d_down_b, CUDA_R_16F, hidden_dim,
        &beta, (void *const *)d_down_c, CUDA_R_32F, dim,
        n_active, CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS)
        return -1;

    moe_all2_scatter_route_kernel<<<
        (dim + threads - 1) / threads, threads, 0, ctx->exec_stream>>>(
        out, d_down, route, dim);
    err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

static int cuda_moe_cublas_grouped_prefill(
    BnCudaCtx *ctx, float *d_full_out,
    const BnCudaBuffer *gate, const BnCudaBuffer *up,
    const BnCudaBuffer *down, const float *d_full_x,
    const float *d_weights, const int *d_slot_order,
    const int *d_expert_offsets, const int *d_expert_counts,
    int *d_active_experts, int n_tokens, int dim, int hidden_dim,
    int n_experts, int k, int act_type) {
    if (!ctx || !ctx->cublas || !d_full_out || !gate || !up || !down ||
        !gate->f16_data || !up->f16_data || !down->f16_data ||
        !d_full_x || !d_weights || !d_slot_order || !d_expert_offsets ||
        !d_expert_counts || !d_active_experts || n_tokens <= 1 ||
        dim <= 0 || hidden_dim <= 0 || n_experts <= 0 || k <= 0)
        return -1;

    int *h_counts = (int *)malloc((size_t)n_experts * sizeof(int));
    int *h_offsets = (int *)malloc((size_t)n_experts * sizeof(int));
    int *h_active = (int *)malloc((size_t)n_experts * sizeof(int));
    if (!h_counts || !h_offsets || !h_active) {
        if (h_counts) free(h_counts);
        if (h_offsets) free(h_offsets);
        if (h_active) free(h_active);
        return -1;
    }
    cudaError_t err = cudaMemcpy(h_counts, d_expert_counts,
                                 (size_t)n_experts * sizeof(int),
                                 cudaMemcpyDeviceToHost);
    if (err == cudaSuccess)
        err = cudaMemcpy(h_offsets, d_expert_offsets,
                         (size_t)n_experts * sizeof(int),
                         cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }
    int n_active = 0;
    int max_count = 0;
    for (int e = 0; e < n_experts; e++) {
        int count = h_counts[e];
        if (count > 0) {
            h_active[n_active++] = e;
            if (count > max_count) max_count = count;
        }
    }
    if (n_active <= 0 || max_count <= 0) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }
    err = cudaMemcpy(d_active_experts, h_active,
                     (size_t)n_active * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }

    size_t gather_values = (size_t)n_active * (size_t)max_count *
                           (size_t)dim;
    size_t hidden_values = (size_t)n_active * (size_t)max_count *
                           (size_t)hidden_dim;
    size_t out_values = gather_values;
    int gateup_f16_out =
        getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP_F16_OUT") == NULL;
    size_t x_bytes = gather_values * sizeof(__half) +
                     hidden_values * sizeof(__half);
    if (gateup_f16_out)
        x_bytes += hidden_values * 2u * sizeof(__half);
    size_t out_bytes = gateup_f16_out
        ? out_values * sizeof(float)
        : hidden_values * 2u * sizeof(float) + out_values * sizeof(float);
    if (cuda_ensure_scratch(ctx, x_bytes, out_bytes) != 0) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }
    __half *d_gather = (__half *)ctx->d_x;
    __half *d_mid = (__half *)((uint8_t *)ctx->d_x +
                               gather_values * sizeof(__half));
    __half *d_gate_h = gateup_f16_out
        ? (__half *)((uint8_t *)d_mid + hidden_values * sizeof(__half))
        : NULL;
    __half *d_up_h = gateup_f16_out ? d_gate_h + hidden_values : NULL;
    float *d_gate = gateup_f16_out ? NULL : ctx->d_out;
    float *d_up = gateup_f16_out ? NULL : d_gate + hidden_values;
    float *d_down = gateup_f16_out ? ctx->d_out : d_up + hidden_values;

    int threads = 256;
    int gather_total = (int)gather_values;
    moe_gather_sorted_f16_kernel<<<(gather_total + threads - 1) / threads,
                                   threads>>>(
        d_gather, d_full_x, d_slot_order, d_expert_offsets,
        d_expert_counts, d_active_experts, n_active, max_count, dim, k);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }

    int ptrs_per = 9;
    int n_ptrs = ptrs_per * n_active;
    if (cuda_ensure_gemm_ptrs(ctx, n_ptrs) != 0) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }
    void **gate_a = ctx->h_gemm_ptrs;
    void **gate_b = gate_a + n_active;
    void **gate_c = gate_b + n_active;
    void **up_a = gate_c + n_active;
    void **up_b = up_a + n_active;
    void **up_c = up_b + n_active;
    void **down_a = up_c + n_active;
    void **down_b = down_a + n_active;
    void **down_c = down_b + n_active;
    for (int i = 0; i < n_active; i++) {
        int e = h_active[i];
        size_t in_off = (size_t)i * (size_t)max_count * (size_t)dim;
        size_t hidden_off =
            (size_t)i * (size_t)max_count * (size_t)hidden_dim;
        gate_a[i] = (uint8_t *)gate->f16_data +
                    (size_t)e * (size_t)hidden_dim * (size_t)dim *
                    sizeof(__half);
        gate_b[i] = d_gather + in_off;
        gate_c[i] = gateup_f16_out ? (void *)(d_gate_h + hidden_off)
                                   : (void *)(d_gate + hidden_off);
        up_a[i] = (uint8_t *)up->f16_data +
                  (size_t)e * (size_t)hidden_dim * (size_t)dim *
                  sizeof(__half);
        up_b[i] = d_gather + in_off;
        up_c[i] = gateup_f16_out ? (void *)(d_up_h + hidden_off)
                                 : (void *)(d_up + hidden_off);
        down_a[i] = (uint8_t *)down->f16_data +
                    (size_t)e * (size_t)dim * (size_t)hidden_dim *
                    sizeof(__half);
        down_b[i] = d_mid + hidden_off;
        down_c[i] = d_down + (size_t)i * (size_t)max_count * (size_t)dim;
    }
    err = cudaMemcpy(ctx->d_gemm_ptrs, ctx->h_gemm_ptrs,
                     (size_t)n_ptrs * sizeof(void *),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }
    void **d_gate_a = ctx->d_gemm_ptrs;
    void **d_gate_b = d_gate_a + n_active;
    void **d_gate_c = d_gate_b + n_active;
    void **d_up_a = d_gate_c + n_active;
    void **d_up_b = d_up_a + n_active;
    void **d_up_c = d_up_b + n_active;
    void **d_down_a = d_up_c + n_active;
    void **d_down_b = d_down_a + n_active;
    void **d_down_c = d_down_b + n_active;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaDataType_t gateup_c_type =
        gateup_f16_out ? CUDA_R_16F : CUDA_R_32F;
    if (ctx->exec_stream)
        cublasSetStream(ctx->cublas, ctx->exec_stream);
    cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
    int use_grouped_var =
        getenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_VARIABLE") != NULL &&
        getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED_VARIABLE") == NULL;
    cublasOperation_t *h_ta = NULL;
    cublasOperation_t *h_tb = NULL;
    int *h_m = NULL;
    int *h_n = NULL;
    int *h_k = NULL;
    int *h_lda = NULL;
    int *h_ldb = NULL;
    int *h_ldc = NULL;
    int *h_group_size = NULL;
    float *h_alpha = NULL;
    float *h_beta = NULL;
    if (use_grouped_var) {
        size_t ops_bytes = (size_t)n_active * sizeof(cublasOperation_t);
        size_t int_bytes = (size_t)n_active * sizeof(int);
        size_t float_bytes = (size_t)n_active * sizeof(float);
        h_ta = (cublasOperation_t *)malloc(ops_bytes);
        h_tb = (cublasOperation_t *)malloc(ops_bytes);
        h_m = (int *)malloc(int_bytes);
        h_n = (int *)malloc(int_bytes);
        h_k = (int *)malloc(int_bytes);
        h_lda = (int *)malloc(int_bytes);
        h_ldb = (int *)malloc(int_bytes);
        h_ldc = (int *)malloc(int_bytes);
        h_group_size = (int *)malloc(int_bytes);
        h_alpha = (float *)malloc(float_bytes);
        h_beta = (float *)malloc(float_bytes);
        if (!h_ta || !h_tb || !h_m || !h_n || !h_k || !h_lda ||
            !h_ldb || !h_ldc || !h_group_size || !h_alpha || !h_beta) {
            use_grouped_var = 0;
        } else {
            for (int i = 0; i < n_active; i++) {
                int e = h_active[i];
                h_ta[i] = CUBLAS_OP_T;
                h_tb[i] = CUBLAS_OP_N;
                h_m[i] = hidden_dim;
                h_n[i] = h_counts[e];
                h_k[i] = dim;
                h_lda[i] = dim;
                h_ldb[i] = dim;
                h_ldc[i] = hidden_dim;
                h_group_size[i] = 1;
                h_alpha[i] = alpha;
                h_beta[i] = beta;
            }
            st = cublasGemmGroupedBatchedEx(
                ctx->cublas, h_ta, h_tb, h_m, h_n, h_k, h_alpha,
                (const void *const *)d_gate_a, CUDA_R_16F, h_lda,
                (const void *const *)d_gate_b, CUDA_R_16F, h_ldb,
                h_beta, (void *const *)d_gate_c, gateup_c_type, h_ldc,
                n_active, h_group_size, CUBLAS_COMPUTE_32F_FAST_16F);
            if (st == CUBLAS_STATUS_SUCCESS) {
                st = cublasGemmGroupedBatchedEx(
                    ctx->cublas, h_ta, h_tb, h_m, h_n, h_k, h_alpha,
                    (const void *const *)d_up_a, CUDA_R_16F, h_lda,
                    (const void *const *)d_up_b, CUDA_R_16F, h_ldb,
                    h_beta, (void *const *)d_up_c, gateup_c_type, h_ldc,
                    n_active, h_group_size, CUBLAS_COMPUTE_32F_FAST_16F);
            }
            if (st != CUBLAS_STATUS_SUCCESS)
                use_grouped_var = 0;
        }
    }
    if (!use_grouped_var) {
        st = cublasGemmBatchedEx(
            ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_dim, max_count, dim, &alpha,
            (const void *const *)d_gate_a, CUDA_R_16F, dim,
            (const void *const *)d_gate_b, CUDA_R_16F, dim,
            &beta, (void *const *)d_gate_c, gateup_c_type, hidden_dim,
            n_active, CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        if (!use_grouped_var) {
            st = cublasGemmBatchedEx(
                ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                hidden_dim, max_count, dim, &alpha,
                (const void *const *)d_up_a, CUDA_R_16F, dim,
                (const void *const *)d_up_b, CUDA_R_16F, dim,
                &beta, (void *const *)d_up_c, gateup_c_type, hidden_dim,
                n_active, CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
    if (st != CUBLAS_STATUS_SUCCESS) {
        if (h_ta) free(h_ta);
        if (h_tb) free(h_tb);
        if (h_m) free(h_m);
        if (h_n) free(h_n);
        if (h_k) free(h_k);
        if (h_lda) free(h_lda);
        if (h_ldb) free(h_ldb);
        if (h_ldc) free(h_ldc);
        if (h_group_size) free(h_group_size);
        if (h_alpha) free(h_alpha);
        if (h_beta) free(h_beta);
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }

    int act_total = (int)hidden_values;
    if (gateup_f16_out) {
        moe_gateup_grouped_f16_to_f16_kernel<<<
            (act_total + threads - 1) / threads, threads>>>(
            d_mid, d_gate_h, d_up_h, d_active_experts, d_expert_counts,
            n_active, max_count, hidden_dim, act_type);
    } else {
        moe_gateup_grouped_to_f16_kernel<<<
            (act_total + threads - 1) / threads, threads>>>(
            d_mid, d_gate, d_up, d_active_experts, d_expert_counts,
            n_active, max_count, hidden_dim, act_type);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (h_ta) free(h_ta);
        if (h_tb) free(h_tb);
        if (h_m) free(h_m);
        if (h_n) free(h_n);
        if (h_k) free(h_k);
        if (h_lda) free(h_lda);
        if (h_ldb) free(h_ldb);
        if (h_ldc) free(h_ldc);
        if (h_group_size) free(h_group_size);
        if (h_alpha) free(h_alpha);
        if (h_beta) free(h_beta);
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }

    if (use_grouped_var) {
        for (int i = 0; i < n_active; i++) {
            int e = h_active[i];
            h_m[i] = dim;
            h_n[i] = h_counts[e];
            h_k[i] = hidden_dim;
            h_lda[i] = hidden_dim;
            h_ldb[i] = hidden_dim;
            h_ldc[i] = dim;
        }
        st = cublasGemmGroupedBatchedEx(
            ctx->cublas, h_ta, h_tb, h_m, h_n, h_k, h_alpha,
            (const void *const *)d_down_a, CUDA_R_16F, h_lda,
            (const void *const *)d_down_b, CUDA_R_16F, h_ldb,
            h_beta, (void *const *)d_down_c, CUDA_R_32F, h_ldc,
            n_active, h_group_size, CUBLAS_COMPUTE_32F_FAST_16F);
    } else {
        st = cublasGemmBatchedEx(
            ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, max_count, hidden_dim, &alpha,
            (const void *const *)d_down_a, CUDA_R_16F, hidden_dim,
            (const void *const *)d_down_b, CUDA_R_16F, hidden_dim,
            &beta, (void *const *)d_down_c, CUDA_R_32F, dim,
            n_active, CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if (h_ta) free(h_ta);
    if (h_tb) free(h_tb);
    if (h_m) free(h_m);
    if (h_n) free(h_n);
    if (h_k) free(h_k);
    if (h_lda) free(h_lda);
    if (h_ldb) free(h_ldb);
    if (h_ldc) free(h_ldc);
    if (h_group_size) free(h_group_size);
    if (h_alpha) free(h_alpha);
    if (h_beta) free(h_beta);
    if (st != CUBLAS_STATUS_SUCCESS) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }

    int scatter_total = (int)out_values;
    moe_scatter_sorted_grouped_kernel<<<(scatter_total + threads - 1) / threads,
                                        threads>>>(
        d_full_out, d_down, d_slot_order, d_expert_offsets, d_expert_counts,
        d_active_experts, d_weights, n_active, max_count, dim, k);
    err = cudaGetLastError();
    free(h_counts);
    free(h_offsets);
    free(h_active);
    return err == cudaSuccess ? 0 : -1;
}

static int cuda_moe_cublas_grouped_gateup_prefill(
    BnCudaCtx *ctx, float **d_mid_out,
    const BnCudaBuffer *gate, const BnCudaBuffer *up,
    const float *d_full_x, const int *d_slot_order,
    const int *d_expert_offsets, const int *d_expert_counts,
    int *d_active_experts, int n_tokens, int dim, int hidden_dim,
    int n_experts, int k, int act_type) {
    if (!ctx || !ctx->cublas || !d_mid_out || !gate || !up ||
        !gate->f16_data || !up->f16_data || !d_full_x || !d_slot_order ||
        !d_expert_offsets || !d_expert_counts || !d_active_experts ||
        n_tokens <= 1 || dim <= 0 || hidden_dim <= 0 ||
        n_experts <= 0 || k <= 0)
        return -1;

    int *h_counts = (int *)malloc((size_t)n_experts * sizeof(int));
    int *h_offsets = (int *)malloc((size_t)n_experts * sizeof(int));
    int *h_active = (int *)malloc((size_t)n_experts * sizeof(int));
    if (!h_counts || !h_offsets || !h_active) {
        if (h_counts) free(h_counts);
        if (h_offsets) free(h_offsets);
        if (h_active) free(h_active);
        return -1;
    }
    cudaError_t err = cudaMemcpy(h_counts, d_expert_counts,
                                 (size_t)n_experts * sizeof(int),
                                 cudaMemcpyDeviceToHost);
    if (err == cudaSuccess)
        err = cudaMemcpy(h_offsets, d_expert_offsets,
                         (size_t)n_experts * sizeof(int),
                         cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }
    int n_active = 0;
    int max_count = 0;
    for (int e = 0; e < n_experts; e++) {
        int count = h_counts[e];
        if (count > 0) {
            h_active[n_active++] = e;
            if (count > max_count) max_count = count;
        }
    }
    if (n_active <= 0 || max_count <= 0) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }
    err = cudaMemcpy(d_active_experts, h_active,
                     (size_t)n_active * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }

    size_t gather_values = (size_t)n_active * (size_t)max_count *
                           (size_t)dim;
    size_t hidden_values = (size_t)n_active * (size_t)max_count *
                           (size_t)hidden_dim;
    size_t mid_values = (size_t)n_tokens * (size_t)k *
                        (size_t)hidden_dim;
    size_t x_bytes = gather_values * sizeof(__half);
    size_t out_bytes = (mid_values + hidden_values * 2u) * sizeof(float);
    if (cuda_ensure_scratch(ctx, x_bytes, out_bytes) != 0) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }
    __half *d_gather = (__half *)ctx->d_x;
    float *d_mid = ctx->d_out;
    float *d_gate = d_mid + mid_values;
    float *d_up = d_gate + hidden_values;
    *d_mid_out = d_mid;

    int threads = 256;
    int gather_total = (int)gather_values;
    moe_gather_sorted_f16_kernel<<<(gather_total + threads - 1) / threads,
                                   threads>>>(
        d_gather, d_full_x, d_slot_order, d_expert_offsets,
        d_expert_counts, d_active_experts, n_active, max_count, dim, k);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }

    int ptrs_per = 6;
    int n_ptrs = ptrs_per * n_active;
    if (cuda_ensure_gemm_ptrs(ctx, n_ptrs) != 0) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }
    void **gate_a = ctx->h_gemm_ptrs;
    void **gate_b = gate_a + n_active;
    void **gate_c = gate_b + n_active;
    void **up_a = gate_c + n_active;
    void **up_b = up_a + n_active;
    void **up_c = up_b + n_active;
    for (int i = 0; i < n_active; i++) {
        int e = h_active[i];
        size_t in_off = (size_t)i * (size_t)max_count * (size_t)dim;
        size_t hidden_off =
            (size_t)i * (size_t)max_count * (size_t)hidden_dim;
        gate_a[i] = (uint8_t *)gate->f16_data +
                    (size_t)e * (size_t)hidden_dim * (size_t)dim *
                    sizeof(__half);
        gate_b[i] = d_gather + in_off;
        gate_c[i] = d_gate + hidden_off;
        up_a[i] = (uint8_t *)up->f16_data +
                  (size_t)e * (size_t)hidden_dim * (size_t)dim *
                  sizeof(__half);
        up_b[i] = d_gather + in_off;
        up_c[i] = d_up + hidden_off;
    }
    err = cudaMemcpy(ctx->d_gemm_ptrs, ctx->h_gemm_ptrs,
                     (size_t)n_ptrs * sizeof(void *),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }
    void **d_gate_a = ctx->d_gemm_ptrs;
    void **d_gate_b = d_gate_a + n_active;
    void **d_gate_c = d_gate_b + n_active;
    void **d_up_a = d_gate_c + n_active;
    void **d_up_b = d_up_a + n_active;
    void **d_up_c = d_up_b + n_active;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (ctx->exec_stream)
        cublasSetStream(ctx->cublas, ctx->exec_stream);
    cublasStatus_t st = cublasGemmBatchedEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_dim, max_count, dim, &alpha,
        (const void *const *)d_gate_a, CUDA_R_16F, dim,
        (const void *const *)d_gate_b, CUDA_R_16F, dim,
        &beta, (void *const *)d_gate_c, CUDA_R_32F, hidden_dim,
        n_active, CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasGemmBatchedEx(
            ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_dim, max_count, dim, &alpha,
            (const void *const *)d_up_a, CUDA_R_16F, dim,
            (const void *const *)d_up_b, CUDA_R_16F, dim,
            &beta, (void *const *)d_up_c, CUDA_R_32F, hidden_dim,
            n_active, CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if (st != CUBLAS_STATUS_SUCCESS) {
        free(h_counts);
        free(h_offsets);
        free(h_active);
        return -1;
    }

    int act_total = (int)hidden_values;
    moe_gateup_grouped_to_route_float_kernel<<<
        (act_total + threads - 1) / threads, threads>>>(
        d_mid, d_gate, d_up, d_slot_order, d_expert_offsets,
        d_expert_counts, d_active_experts, n_active, max_count, hidden_dim,
        k, act_type);
    err = cudaGetLastError();
    free(h_counts);
    free(h_offsets);
    free(h_active);
    return err == cudaSuccess ? 0 : -1;
}

static int cuda_matmul_device_out_preconverted_f16(
        BnCudaCtx *ctx, float *d_dst, const BnCudaBuffer *w,
        const float *d_x, const void *d_x_f16,
        int rows, int cols, int n_tokens, int type) {
    if (!cuda_force_quant_matmul_for_type(type) &&
        w && w->f16_data && d_x_f16 &&
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
    if (!cuda_force_quant_matmul_for_type(type) &&
        (w->f16_data || w->f32_data) && n_tokens > 1 &&
        cuda_cublas_matmul_f16(ctx, d_dst, w, d_x, rows, cols,
                               n_tokens) == 0) {
        err = cudaSuccess;
    } else if (type == BN_GGUF_TENSOR_Q4_K && (cols % BN_QK_K) == 0 &&
               n_tokens > 1 &&
               getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT") == NULL) {
        int x_blocks = cols / BN_QK_K;
        if (cuda_ensure_q8_k(ctx, cols, n_tokens) != 0)
            return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(x_blocks, n_tokens, 1),
                                    BN_QK_K>>>(xq, d_x, cols, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q4k_q8k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                d_dst, (const BnBlockQ4K *)w->data, xq, rows, cols,
                n_tokens, 0);
        } else {
            dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
            q4k_q8k_dot_matmul_kernel<<<grid, threads, 0>>>(
                d_dst, (const BnBlockQ4K *)w->data, xq, rows, cols,
                n_tokens, 0);
        }
    } else if (type == BN_GGUF_TENSOR_Q4_K && (cols % BN_QK_K) == 0) {
        int x_blocks = (cols + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, d_x, cols, n_tokens);
        if (n_tokens >= 8 &&
            getenv("BN_CUDA_ENABLE_Q4K_MATMUL8") != NULL &&
            getenv("BN_CUDA_DISABLE_Q4K_SHAREDX_BATCH") == NULL) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 7) / 8, 1);
            size_t shared =
                (size_t)x_blocks * 8u * sizeof(BnCudaBlockQ8_1);
            q4k_dot_matmul8_token_sharedx_kernel<<<grid, threads,
                                                    shared>>>(
                d_dst, (const BnBlockQ4K *)w->data, xq, rows, cols,
                n_tokens, 0);
        } else if (n_tokens >= 4) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            if (getenv("BN_CUDA_DISABLE_Q4K_SHAREDX_BATCH") == NULL) {
                size_t shared =
                    (size_t)x_blocks * 4u * sizeof(BnCudaBlockQ8_1);
                q4k_dot_matmul4_token_sharedx_kernel<<<grid, threads,
                                                        shared>>>(
                    d_dst, (const BnBlockQ4K *)w->data, xq, rows, cols,
                    n_tokens, 0);
            } else {
                q4k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                    d_dst, (const BnBlockQ4K *)w->data, xq, rows, cols,
                    n_tokens, 0);
            }
        } else {
            dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
            q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
                d_dst, (const BnBlockQ4K *)w->data, xq, rows, cols,
                n_tokens, 0);
        }
    } else if (type == BN_GGUF_TENSOR_Q5_K && (cols % BN_QK_K) == 0) {
        int x_blocks = (cols + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, d_x, cols, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q5k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                d_dst, (const BnBlockQ5K *)w->data, xq, rows, cols,
                n_tokens, 0);
        } else {
            dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
            q5k_dot_matmul_kernel<<<grid, threads, 0>>>(
                d_dst, (const BnBlockQ5K *)w->data, xq, rows, cols,
                n_tokens, 0);
        }
    } else if (type == BN_GGUF_TENSOR_Q6_K && (cols % BN_QK_K) == 0 &&
               !getenv("BN_CUDA_DISABLE_Q6K_DOT")) {
        int x_blocks = cols / BN_QK_K;
        if (cuda_ensure_q8_k(ctx, cols, n_tokens) != 0) return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(x_blocks, n_tokens, 1),
                                    BN_QK_K>>>(xq, d_x, cols, n_tokens);
        if (n_tokens >= 8 && getenv("BN_CUDA_ENABLE_Q6K_MATMUL8") != NULL) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 7) / 8, 1);
            q6k_dot_matmul8_token_kernel<<<grid, threads, 0>>>(
                d_dst, (const BnBlockQ6K *)w->data, xq, rows, cols,
                n_tokens, 0);
        } else if (n_tokens >= 4 && getenv("BN_CUDA_DISABLE_Q6K_MATMUL4") == NULL) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q6k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                d_dst, (const BnBlockQ6K *)w->data, xq, rows, cols,
                n_tokens, 0);
        } else {
            dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
            q6k_dot_matmul_kernel<<<grid, threads, 0>>>(
                d_dst, (const BnBlockQ6K *)w->data, xq, rows, cols,
                n_tokens, 0);
        }
    } else if (type == BN_GGUF_TENSOR_Q8_0 && (cols & 31) == 0) {
        if (cuda_launch_q8_0_matmul(ctx, d_dst,
                (const BnBlockQ8_0 *)w->data, d_x, rows, cols,
                n_tokens, 0) != 0)
            return -1;
    } else if (type == BN_GGUF_TENSOR_Q5_0 && (cols & 31) == 0) {
        dim3 grid(((rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            d_dst, (const BnBlockQ5_0 *)w->data, d_x, rows, cols,
            n_tokens, 0);
    } else {
        dim3 grid(rows, n_tokens, 1);
        matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
            d_dst, w->data, d_x, NULL, rows, cols, type, 0, 0);
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
    if ((type == BN_GGUF_TENSOR_Q3_K ||
         type == BN_GGUF_TENSOR_IQ3_XXS ||
         type == BN_GGUF_TENSOR_IQ4_XS) && w->f16_data) {
        int warps = threads / 32;
        int blocks = (rows + warps - 1) / warps;
        f16_matvec_warp_kernel<<<blocks, threads>>>(
            ctx->d_out, (const __half *)w->f16_data, ctx->d_x, NULL,
            rows, cols, 0);
    } else if (type == BN_GGUF_TENSOR_Q5_0 && (cols & 31) == 0 &&
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
               getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT") == NULL) {
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
            ctx->d_out, w->data, ctx->d_x, NULL, rows, cols, type, 0, 0);
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
    if (!cuda_force_quant_matmul_for_type(type) &&
        (w->f16_data || w->f32_data) && n_tokens > 1 &&
        cuda_cublas_matmul_f16(ctx, ctx->d_out, w, ctx->d_x, rows, cols,
                               n_tokens) == 0) {
        err = cudaSuccess;
    } else if (type == BN_GGUF_TENSOR_Q4_K && (cols % BN_QK_K) == 0 &&
               getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT") == NULL) {
        int x_blocks = cols / BN_QK_K;
        if (cuda_ensure_q8_k(ctx, cols, n_tokens) != 0)
            return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8k_batch_kernel<<<qgrid, BN_QK_K>>>(
            xq, ctx->d_x, cols, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q4k_q8k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ4K *)w->data, xq, rows, cols,
                n_tokens, 0);
        } else {
            dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
            q4k_q8k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ4K *)w->data, xq, rows, cols,
                n_tokens, 0);
        }
    } else if (type == BN_GGUF_TENSOR_Q4_K && (cols % BN_QK_K) == 0) {
        int x_blocks = (cols + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, ctx->d_x, cols, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            if (getenv("BN_CUDA_DISABLE_Q4K_SHAREDX_BATCH") == NULL) {
                size_t shared =
                    (size_t)x_blocks * 4u * sizeof(BnCudaBlockQ8_1);
                q4k_dot_matmul4_token_sharedx_kernel<<<grid, threads,
                                                        shared>>>(
                    ctx->d_out, (const BnBlockQ4K *)w->data, xq, rows, cols,
                    n_tokens, 0);
            } else {
                q4k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ4K *)w->data, xq, rows, cols,
                    n_tokens, 0);
            }
        } else {
            dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
            q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ4K *)w->data, xq, rows, cols,
                n_tokens, 0);
        }
    } else if (type == BN_GGUF_TENSOR_Q5_K && (cols % BN_QK_K) == 0) {
        int x_blocks = (cols + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, ctx->d_x, cols, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q5k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)w->data, xq, rows, cols,
                n_tokens, 0);
        } else {
            dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
            q5k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)w->data, xq, rows, cols,
                n_tokens, 0);
        }
    } else if (type == BN_GGUF_TENSOR_Q6_K && (cols % BN_QK_K) == 0 &&
               getenv("BN_CUDA_ENABLE_Q6K_DOT")) {
        int x_blocks = cols / BN_QK_K;
        if (cuda_ensure_q8_k(ctx, cols, n_tokens) != 0) return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(x_blocks, n_tokens, 1),
                                    BN_QK_K>>>(xq, ctx->d_x, cols,
                                               n_tokens);
        if (n_tokens >= 8 && getenv("BN_CUDA_ENABLE_Q6K_MATMUL8") != NULL) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 7) / 8, 1);
            q6k_dot_matmul8_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ6K *)w->data, xq, rows,
                cols, n_tokens, 0);
        } else if (n_tokens >= 4) {
            dim3 grid((rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q6k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ6K *)w->data, xq, rows,
                cols, n_tokens, 0);
        } else {
            dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
            q6k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ6K *)w->data, xq, rows,
                cols, n_tokens, 0);
        }
    } else if (type == BN_GGUF_TENSOR_Q8_0 && (cols & 31) == 0) {
        if (cuda_launch_q8_0_matmul(ctx, ctx->d_out,
                (const BnBlockQ8_0 *)w->data, ctx->d_x, rows, cols,
                n_tokens, 0) != 0)
            return -1;
    } else if (type == BN_GGUF_TENSOR_Q5_0 && (cols & 31) == 0) {
        dim3 grid(((rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ5_0 *)w->data, ctx->d_x, rows, cols,
            n_tokens, 0);
    } else {
        dim3 grid(rows, n_tokens, 1);
        matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
            ctx->d_out, w->data, ctx->d_x, NULL, rows, cols, type, 0, 0);
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
        if (!cuda_force_quant_matmul_for_type(type) &&
            w->f16_data && x_f16_ready &&
            cuda_cublas_matmul_f16_preconverted(
                ctx, ctx->d_out + out_offset, w, ctx->d_x_f16, rows,
                x_cols, n_tokens) == 0) {
            err = cudaSuccess;
        } else if (!cuda_force_quant_matmul_for_type(type) &&
            (w->f16_data || w->f32_data) &&
            cuda_cublas_matmul_f16(ctx, ctx->d_out + out_offset, w,
                                   ctx->d_x, rows, x_cols,
                                   n_tokens) == 0) {
            err = cudaSuccess;
        } else if (type == BN_GGUF_TENSOR_Q8_0 && (x_cols & 31) == 0) {
            if (cuda_launch_q8_0_matmul(ctx, ctx->d_out,
                    (const BnBlockQ8_0 *)w->data, ctx->d_x, rows, x_cols,
                    n_tokens, out_offset) != 0)
                return -1;
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
            if (n_tokens >= 4) {
                dim3 grid((rows + warps - 1) / warps,
                          (n_tokens + 3) / 4, 1);
                if (getenv("BN_CUDA_ENABLE_Q4K_SHAREDX_BATCH")) {
                    size_t shared =
                        (size_t)x_blocks * 4u * sizeof(BnCudaBlockQ8_1);
                    q4k_dot_matmul4_token_sharedx_kernel<<<grid, threads,
                                                            shared>>>(
                        ctx->d_out, (const BnBlockQ4K *)w->data,
                        (const BnCudaBlockQ8_1 *)ctx->d_q8_1, rows, x_cols,
                        n_tokens, out_offset);
                } else {
                    q4k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                        ctx->d_out, (const BnBlockQ4K *)w->data,
                        (const BnCudaBlockQ8_1 *)ctx->d_q8_1, rows, x_cols,
                        n_tokens, out_offset);
                }
            } else {
                dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
                q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ4K *)w->data,
                    (const BnCudaBlockQ8_1 *)ctx->d_q8_1, rows, x_cols,
                    n_tokens, out_offset);
            }
        } else if (type == BN_GGUF_TENSOR_Q5_K &&
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
            if (n_tokens >= 4) {
                dim3 grid((rows + warps - 1) / warps,
                          (n_tokens + 3) / 4, 1);
                q5k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ5K *)w->data,
                    (const BnCudaBlockQ8_1 *)ctx->d_q8_1, rows, x_cols,
                    n_tokens, out_offset);
            } else {
                dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
                q5k_dot_matmul_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ5K *)w->data,
                    (const BnCudaBlockQ8_1 *)ctx->d_q8_1, rows, x_cols,
                    n_tokens, out_offset);
            }
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
            if (n_tokens >= 8 &&
                getenv("BN_CUDA_ENABLE_Q6K_MATMUL8") != NULL) {
                dim3 grid((rows + warps - 1) / warps,
                          (n_tokens + 7) / 8, 1);
                q6k_dot_matmul8_token_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ6K *)w->data,
                    (const BnBlockQ8K *)ctx->d_q8_k, rows, x_cols,
                    n_tokens, out_offset);
            } else if (n_tokens >= 4 &&
                       getenv("BN_CUDA_DISABLE_Q6K_MATMUL4") == NULL) {
                dim3 grid((rows + warps - 1) / warps,
                          (n_tokens + 3) / 4, 1);
                q6k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ6K *)w->data,
                    (const BnBlockQ8K *)ctx->d_q8_k, rows, x_cols,
                    n_tokens, out_offset);
            } else {
                dim3 grid((rows + warps - 1) / warps, n_tokens, 1);
                q6k_dot_matmul_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ6K *)w->data,
                    (const BnBlockQ8K *)ctx->d_q8_k, rows, x_cols,
                    n_tokens, out_offset);
            }
        } else {
            dim3 grid(rows, n_tokens, 1);
            matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
                ctx->d_out, w->data, ctx->d_x, NULL, rows, x_cols, type,
                out_offset, 0);
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
    int penalty_sorted[BN_CUDA_ARGMAX_PENALTY_SORT_MAX];
    if (n_penalty_tokens > 0) {
        n_penalty_tokens = cuda_prepare_penalty_tokens(
            penalty_tokens, n_penalty_tokens, penalty_sorted,
            BN_CUDA_ARGMAX_PENALTY_SORT_MAX);
        if (n_penalty_tokens < 0) return -1;
    }
    if (cuda_ensure_argmax(ctx, n, n_penalty_tokens) != 0) return -1;
    cudaStream_t stream = ctx->exec_stream ? ctx->exec_stream : ctx->stream;
    if (n_penalty_tokens > 0) {
        cudaError_t copy_err = cudaMemcpyAsync(
            ctx->d_penalty_tokens, penalty_sorted,
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
    int no_penalty_fast =
        n_penalty_tokens == 0 &&
        getenv("BN_CUDA_DISABLE_ARGMAX_FAST") == NULL;
    int optimistic_penalty =
        n_penalty_tokens > 0 &&
        getenv("BN_CUDA_ENABLE_OPTIMISTIC_ARGMAX_PENALTY") != NULL;
    if (no_penalty_fast) {
        argmax_stage1_fast_kernel<<<blocks, threads, 0, stream>>>(
            partials, x, n);
        argmax_stage2_fast_kernel<<<1, threads, 0, stream>>>(
            d_result, partials, blocks);
        cudaError_t fast_err = cudaGetLastError();
        if (fast_err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] argmax launch failed: %s\n",
                    cudaGetErrorString(fast_err));
            return -1;
        }
        fast_err = cudaMemcpyAsync(ctx->h_argmax, d_result, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream);
        if (fast_err == cudaSuccess)
            fast_err = cudaStreamSynchronize(stream);
        if (fast_err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] argmax readback failed: %s\n",
                    cudaGetErrorString(fast_err));
            return -1;
        }
        *out_token = *ctx->h_argmax;
        return 0;
    }
    if (optimistic_penalty) {
        argmax_penalty_stage1_kernel<<<blocks, threads,
            (size_t)threads * sizeof(BnCudaArgmaxPair), stream>>>(
            partials, x, n, NULL, 0, repeat_penalty);
        argmax_stage2_kernel<<<1, threads,
            (size_t)threads * sizeof(BnCudaArgmaxPair), stream>>>(
            d_result, partials, blocks);
        cudaError_t opt_err = cudaGetLastError();
        if (opt_err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] argmax launch failed: %s\n",
                    cudaGetErrorString(opt_err));
            return -1;
        }
        opt_err = cudaMemcpyAsync(ctx->h_argmax, d_result, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream);
        if (opt_err == cudaSuccess)
            opt_err = cudaStreamSynchronize(stream);
        if (opt_err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] argmax readback failed: %s\n",
                    cudaGetErrorString(opt_err));
            return -1;
        }
        *out_token = *ctx->h_argmax;
        if (!cuda_penalty_tokens_contain_sorted(
                penalty_sorted, n_penalty_tokens, *out_token))
            return 0;
    }
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
    err = cudaMemcpyAsync(ctx->h_argmax, d_result, sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess)
        err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] argmax readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    *out_token = *ctx->h_argmax;
    return 0;
}

static int cuda_matvec_argmax_activation(void *vctx, void *W_buf, int type,
                                         int rows, int cols, int buf_idx,
                                         const int *penalty_tokens,
                                         int n_penalty_tokens,
                                         float repeat_penalty,
                                         int *out_token) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *w = (BnCudaBuffer *)W_buf;
    if (!ctx || !w || !w->data || !out_token || rows <= 0 || cols <= 0 ||
        buf_idx < 0 || buf_idx >= BN_GPU_VALUE_COUNT)
        return -1;
    if (type != BN_GGUF_TENSOR_Q6_K || (cols % BN_QK_K) != 0)
        return -1;
    float *x = cuda_act(ctx, buf_idx);
    if (!x) return -1;
    if (n_penalty_tokens < 0) n_penalty_tokens = 0;
    if (repeat_penalty == 1.0f) n_penalty_tokens = 0;
    if (n_penalty_tokens > 0 && !penalty_tokens) return -1;
    int penalty_sorted[BN_CUDA_ARGMAX_PENALTY_SORT_MAX];
    if (n_penalty_tokens > 0) {
        n_penalty_tokens = cuda_prepare_penalty_tokens(
            penalty_tokens, n_penalty_tokens, penalty_sorted,
            BN_CUDA_ARGMAX_PENALTY_SORT_MAX);
        if (n_penalty_tokens < 0) return -1;
    }
    int threads = 256;
    int use_mmvq =
        getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX") == NULL &&
        rows >= 50000 &&
        (cols == 1536 ||
         getenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX") != NULL);
    int use_mmvq_1warp8 =
        use_mmvq && rows == 151936 && cols == 1536 &&
        getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536") == NULL;
    int use_mmvq_1warp16 =
        use_mmvq_1warp8 &&
        getenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_1WARP16_1536") != NULL;
    int use_mmvq_1warp8_1536 =
        use_mmvq_1warp8 && !use_mmvq_1warp16 &&
        getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536_UNROLL") == NULL;
    int blocks = use_mmvq
        ? (use_mmvq_1warp16 ? (rows + 15) / 16 :
           (use_mmvq_1warp8 ? (rows + 7) / 8 : (rows + 3) / 4))
        : (rows + 31) / 32;
    if (cuda_ensure_argmax_partials(ctx, blocks, n_penalty_tokens) != 0)
        return -1;
    cudaStream_t stream = ctx->exec_stream ? ctx->exec_stream : ctx->stream;
    if (n_penalty_tokens > 0) {
        cudaError_t copy_err = cudaMemcpyAsync(
            ctx->d_penalty_tokens, penalty_sorted,
            (size_t)n_penalty_tokens * sizeof(int),
            cudaMemcpyHostToDevice, stream);
        if (copy_err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] penalty token copy failed: %s\n",
                    cudaGetErrorString(copy_err));
            return -1;
        }
    }

    BnCudaArgmaxPair *partials = (BnCudaArgmaxPair *)ctx->d_argmax;
    int *d_result = (int *)(partials + blocks);
    int optimistic_penalty =
        n_penalty_tokens > 0 &&
        getenv("BN_CUDA_ENABLE_OPTIMISTIC_ARGMAX_PENALTY") != NULL;
    if (use_mmvq) {
        if (cuda_ensure_q8_1(ctx, cols) != 0) return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        quantize_q8_1_kernel<<<(cols + 31) / 32, 32, 0, stream>>>(
            xq, x, cols);
        if (optimistic_penalty) {
            if (use_mmvq_1warp16) {
                q6k_dot_argmax_mmvq_1warp16_kernel<<<blocks, 512, 0, stream>>>(
                    partials, (const BnBlockQ6K *)w->data, xq, NULL, 0,
                    repeat_penalty, rows, cols);
            } else if (use_mmvq_1warp8_1536) {
                q6k_dot_argmax_mmvq_1warp8_1536_kernel<<<blocks, threads, 0, stream>>>(
                    partials, (const BnBlockQ6K *)w->data, xq, NULL, 0,
                    repeat_penalty, rows);
            } else if (use_mmvq_1warp8) {
                q6k_dot_argmax_mmvq_1warp8_kernel<<<blocks, threads, 0, stream>>>(
                    partials, (const BnBlockQ6K *)w->data, xq, NULL, 0,
                    repeat_penalty, rows, cols);
            } else {
                q6k_dot_argmax_mmvq_2warp4_kernel<<<blocks, threads, 0, stream>>>(
                    partials, (const BnBlockQ6K *)w->data, xq, NULL, 0,
                    repeat_penalty, rows, cols);
            }
            argmax_stage2_kernel<<<1, threads,
                (size_t)threads * sizeof(BnCudaArgmaxPair), stream>>>(
                d_result, partials, blocks);
            cudaError_t opt_err = cudaGetLastError();
            if (opt_err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] matvec argmax launch failed: %s\n",
                        cudaGetErrorString(opt_err));
                return -1;
            }
            opt_err = cudaMemcpyAsync(ctx->h_argmax, d_result, sizeof(int),
                                      cudaMemcpyDeviceToHost, stream);
            if (opt_err == cudaSuccess)
                opt_err = cudaStreamSynchronize(stream);
            if (opt_err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] matvec argmax readback failed: %s\n",
                        cudaGetErrorString(opt_err));
                return -1;
            }
            *out_token = *ctx->h_argmax;
            if (!cuda_penalty_tokens_contain_sorted(
                    penalty_sorted, n_penalty_tokens, *out_token))
                return 0;
        }
        if (use_mmvq_1warp16) {
            q6k_dot_argmax_mmvq_1warp16_kernel<<<blocks, 512, 0, stream>>>(
                partials, (const BnBlockQ6K *)w->data, xq,
                n_penalty_tokens > 0 ? ctx->d_penalty_tokens : NULL,
                n_penalty_tokens, repeat_penalty, rows, cols);
        } else if (use_mmvq_1warp8_1536) {
            q6k_dot_argmax_mmvq_1warp8_1536_kernel<<<blocks, threads, 0, stream>>>(
                partials, (const BnBlockQ6K *)w->data, xq,
                n_penalty_tokens > 0 ? ctx->d_penalty_tokens : NULL,
                n_penalty_tokens, repeat_penalty, rows);
        } else if (use_mmvq_1warp8) {
            q6k_dot_argmax_mmvq_1warp8_kernel<<<blocks, threads, 0, stream>>>(
                partials, (const BnBlockQ6K *)w->data, xq,
                n_penalty_tokens > 0 ? ctx->d_penalty_tokens : NULL,
                n_penalty_tokens, repeat_penalty, rows, cols);
        } else {
            q6k_dot_argmax_mmvq_2warp4_kernel<<<blocks, threads, 0, stream>>>(
                partials, (const BnBlockQ6K *)w->data, xq,
                n_penalty_tokens > 0 ? ctx->d_penalty_tokens : NULL,
                n_penalty_tokens, repeat_penalty, rows, cols);
        }
    } else {
        if (cuda_ensure_q8_k(ctx, cols, 1) != 0) return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(cols / BN_QK_K, 1, 1), BN_QK_K,
                                    0, stream>>>(xq, x, cols, 1);
        if (optimistic_penalty) {
            q6k_dot_argmax32_kernel<<<blocks, threads, 0, stream>>>(
                partials, (const BnBlockQ6K *)w->data, xq, NULL, 0,
                repeat_penalty, rows, cols);
            argmax_stage2_kernel<<<1, threads,
                (size_t)threads * sizeof(BnCudaArgmaxPair), stream>>>(
                d_result, partials, blocks);
            cudaError_t opt_err = cudaGetLastError();
            if (opt_err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] matvec argmax launch failed: %s\n",
                        cudaGetErrorString(opt_err));
                return -1;
            }
            opt_err = cudaMemcpyAsync(ctx->h_argmax, d_result, sizeof(int),
                                      cudaMemcpyDeviceToHost, stream);
            if (opt_err == cudaSuccess)
                opt_err = cudaStreamSynchronize(stream);
            if (opt_err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] matvec argmax readback failed: %s\n",
                        cudaGetErrorString(opt_err));
                return -1;
            }
            *out_token = *ctx->h_argmax;
            if (!cuda_penalty_tokens_contain_sorted(
                    penalty_sorted, n_penalty_tokens, *out_token))
                return 0;
        }
        q6k_dot_argmax32_kernel<<<blocks, threads, 0, stream>>>(
            partials, (const BnBlockQ6K *)w->data, xq,
            n_penalty_tokens > 0 ? ctx->d_penalty_tokens : NULL,
            n_penalty_tokens, repeat_penalty, rows, cols);
    }
    argmax_stage2_kernel<<<1, threads,
        (size_t)threads * sizeof(BnCudaArgmaxPair), stream>>>(
        d_result, partials, blocks);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] matvec argmax launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaMemcpyAsync(ctx->h_argmax, d_result, sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess)
        err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] matvec argmax readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    *out_token = *ctx->h_argmax;
    return 0;
}

static int cuda_memory_info(void *vctx,
                            size_t *free_bytes,
                            size_t *total_bytes) {
    (void)vctx;
    if (!free_bytes || !total_bytes)
        return -1;
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    return err == cudaSuccess ? 0 : -1;
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
        0, 0);
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

static int cuda_dense_ffn_batch_device(BnCudaCtx *ctx, float *d_out,
                                       BnCudaBuffer *gate, BnCudaBuffer *up,
                                       BnCudaBuffer *down, const float *d_X,
                                       int n_tokens, int dim, int hidden_dim,
                                       int gate_type, int up_type,
                                       int down_type, int act_type) {
    int stacked_gateup = up == NULL;
    if (!ctx || !d_out || !gate || !down || !d_X ||
        !gate->data || !down->data ||
        n_tokens <= 0 || dim <= 0 || hidden_dim <= 0 || act_type != 0)
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
    if (!cuda_type_supported(gate_type) ||
        (!stacked_gateup && !cuda_type_supported(up_type)) ||
        !cuda_type_supported(down_type))
        return -1;

    if (stacked_gateup) {
        if (cuda_matmul_device_out(ctx, ctx->d_out, gate, d_X,
                                   hidden_dim * 2, dim, n_tokens,
                                   gate_type) != 0)
            return -1;
    } else {
        if (cuda_matmul_device_out(ctx, ctx->d_out, gate, d_X,
                                   hidden_dim, dim, n_tokens,
                                   gate_type) != 0)
            return -1;
        if (cuda_matmul_device_out(
                ctx, ctx->d_out + (size_t)n_tokens * hidden_dim,
                up, d_X, hidden_dim, dim, n_tokens, up_type) != 0)
            return -1;
    }

    int threads = 256;
    int act_total = n_tokens * hidden_dim;
    int act_blocks = (act_total + threads - 1) / threads;
    if (stacked_gateup) {
        ffn_activation_batch_stacked_kernel<<<act_blocks, threads>>>(
            ctx->d_x, ctx->d_out, hidden_dim, n_tokens, act_type);
    } else {
        ffn_activation_batch_kernel<<<act_blocks, threads>>>(
            ctx->d_x, ctx->d_out, hidden_dim, n_tokens, act_type);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] dense ffn device activation failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return cuda_matmul_device_out(ctx, d_out, down, ctx->d_x,
                                  dim, hidden_dim, n_tokens, down_type);
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
        n_tokens <= 0 || dim <= 0 || hidden_dim <= 0 || act_type != 0)
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
        if (n_tokens >= 4) {
            dim3 grid((hidden_dim * 2 + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            if (getenv("BN_CUDA_ENABLE_Q4K_SHAREDX_BATCH")) {
                size_t shared =
                    (size_t)x_blocks * 4u * sizeof(BnCudaBlockQ8_1);
                q4k_dot_matmul4_token_sharedx_kernel<<<grid, threads,
                                                        shared>>>(
                    ctx->d_out, (const BnBlockQ4K *)gate->data, xq,
                    hidden_dim * 2, dim, n_tokens, 0);
            } else {
                q4k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ4K *)gate->data, xq,
                    hidden_dim * 2, dim, n_tokens, 0);
            }
        } else {
            dim3 grid((hidden_dim * 2 + warps - 1) / warps,
                      n_tokens, 1);
            q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ4K *)gate->data, xq,
                hidden_dim * 2, dim, n_tokens, 0);
        }
    } else if (stacked_gateup && gate_type == BN_GGUF_TENSOR_Q5_K &&
               (dim % BN_QK_K) == 0) {
        int x_blocks = (dim + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, gate_input, dim, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((hidden_dim * 2 + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q5k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)gate->data, xq,
                hidden_dim * 2, dim, n_tokens, 0);
        } else {
            dim3 grid((hidden_dim * 2 + warps - 1) / warps,
                      n_tokens, 1);
            q5k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)gate->data, xq,
                hidden_dim * 2, dim, n_tokens, 0);
        }
    } else if (stacked_gateup && gate_type == BN_GGUF_TENSOR_Q5_0 &&
               (dim & 31) == 0) {
        dim3 grid((((hidden_dim * 2) + 3) / 4 + warps - 1) / warps,
                  n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ5_0 *)gate->data, gate_input,
            hidden_dim * 2, dim, n_tokens, 0);
    } else if (stacked_gateup && gate_type == BN_GGUF_TENSOR_Q8_0 &&
               (dim & 31) == 0) {
        if (cuda_launch_q8_0_matmul(ctx, ctx->d_out,
                (const BnBlockQ8_0 *)gate->data, gate_input,
                hidden_dim * 2, dim, n_tokens, 0) != 0)
            return -1;
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
        if (cuda_launch_q8_0_matmul(ctx, ctx->d_out,
                (const BnBlockQ8_0 *)gate->data, gate_input,
                hidden_dim, dim, n_tokens, 0) != 0)
            return -1;
        if (cuda_launch_q8_0_matmul(ctx, ctx->d_out,
                (const BnBlockQ8_0 *)up->data, gate_input,
                hidden_dim, dim, n_tokens, (size_t)n_tokens * hidden_dim) != 0)
            return -1;
    } else if (!stacked_gateup && gate_type == BN_GGUF_TENSOR_Q4_K &&
               up_type == BN_GGUF_TENSOR_Q4_K && (dim % BN_QK_K) == 0) {
        int x_blocks = (dim + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, gate_input, dim, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((hidden_dim + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            if (getenv("BN_CUDA_ENABLE_Q4K_SHAREDX_BATCH")) {
                size_t shared =
                    (size_t)x_blocks * 4u * sizeof(BnCudaBlockQ8_1);
                q4k_dot_matmul4_token_sharedx_kernel<<<grid, threads,
                                                        shared>>>(
                    ctx->d_out, (const BnBlockQ4K *)gate->data, xq,
                    hidden_dim, dim, n_tokens, 0);
                q4k_dot_matmul4_token_sharedx_kernel<<<grid, threads,
                                                        shared>>>(
                    ctx->d_out, (const BnBlockQ4K *)up->data, xq,
                    hidden_dim, dim, n_tokens,
                    (size_t)n_tokens * hidden_dim);
            } else {
                q4k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ4K *)gate->data, xq,
                    hidden_dim, dim, n_tokens, 0);
                q4k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ4K *)up->data, xq,
                    hidden_dim, dim, n_tokens,
                    (size_t)n_tokens * hidden_dim);
            }
        } else {
            dim3 grid((hidden_dim + warps - 1) / warps, n_tokens, 1);
            q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ4K *)gate->data, xq, hidden_dim,
                dim, n_tokens, 0);
            q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ4K *)up->data, xq, hidden_dim,
                dim, n_tokens, (size_t)n_tokens * hidden_dim);
        }
    } else if (!stacked_gateup && gate_type == BN_GGUF_TENSOR_Q5_K &&
               up_type == BN_GGUF_TENSOR_Q5_K && (dim % BN_QK_K) == 0) {
        int x_blocks = (dim + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        dim3 qgrid(x_blocks, n_tokens, 1);
        quantize_q8_1_batch_kernel<<<qgrid, 32, 0>>>(
            xq, gate_input, dim, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((hidden_dim + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q5k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)gate->data, xq, hidden_dim,
                dim, n_tokens, 0);
            q5k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)up->data, xq, hidden_dim,
                dim, n_tokens, (size_t)n_tokens * hidden_dim);
        } else {
            dim3 grid((hidden_dim + warps - 1) / warps, n_tokens, 1);
            q5k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)gate->data, xq, hidden_dim,
                dim, n_tokens, 0);
            q5k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)up->data, xq, hidden_dim,
                dim, n_tokens, (size_t)n_tokens * hidden_dim);
        }
    } else {
        if (stacked_gateup)
            return -1;
        dim3 grid(hidden_dim, n_tokens, 1);
        matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
            ctx->d_out, gate->data, gate_input, NULL, hidden_dim, dim,
            gate_type, 0, 0);
        matvec_kernel<<<grid, threads, (size_t)threads * sizeof(float)>>>(
            ctx->d_out, up->data, gate_input, NULL, hidden_dim, dim,
            up_type, (size_t)n_tokens * hidden_dim, 0);
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
        if (cuda_launch_q8_0_matmul(ctx, ctx->d_out,
                (const BnBlockQ8_0 *)down->data, ctx->d_x,
                dim, hidden_dim, n_tokens, 0) != 0)
            return -1;
    } else if (down_type == BN_GGUF_TENSOR_Q5_0 && (hidden_dim & 31) == 0) {
        dim3 grid(((dim + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ5_0 *)down->data, ctx->d_x,
            dim, hidden_dim, n_tokens, 0);
    } else if (down_type == BN_GGUF_TENSOR_Q5_K &&
               (hidden_dim % BN_QK_K) == 0) {
        int x_blocks = (hidden_dim + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        quantize_q8_1_batch_kernel<<<dim3(x_blocks, n_tokens, 1), 32, 0>>>(
            xq, ctx->d_x, hidden_dim, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((dim + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q5k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)down->data, xq,
                dim, hidden_dim, n_tokens, 0);
        } else {
            dim3 grid((dim + warps - 1) / warps, n_tokens, 1);
            q5k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)down->data, xq,
                dim, hidden_dim, n_tokens, 0);
        }
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
        if (n_tokens >= 4 && getenv("BN_CUDA_DISABLE_Q6K_MATMUL4") == NULL) {
            dim3 grid((dim + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q6k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ6K *)down->data, xq,
                dim, hidden_dim, n_tokens, 0);
        } else {
            dim3 grid((dim + warps - 1) / warps, n_tokens, 1);
            q6k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ6K *)down->data, xq,
                dim, hidden_dim, n_tokens, 0);
        }
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
            down_type, 0, 0);
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

static int cuda_moe_ffn_batch(void *vctx, float *out,
                              const BnGPUMoEPrefillExpert *experts,
                              int n_experts,
                              const int *expert_offsets,
                              const int *expert_counts,
                              const int *token_ids,
                              const float *weights,
                              const float *X,
                              int n_tokens, int dim, int hidden_dim,
                              int gate_type, int up_type, int down_type,
                              int act_type,
                              void *shared_gate_buf, void *shared_up_buf,
                              void *shared_down_buf,
                              void *shared_gate_weight_buf,
                              int shared_hidden_dim,
                              int shared_gate_type, int shared_up_type,
                              int shared_down_type) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *shared_gate = (BnCudaBuffer *)shared_gate_buf;
    BnCudaBuffer *shared_up = (BnCudaBuffer *)shared_up_buf;
    BnCudaBuffer *shared_down = (BnCudaBuffer *)shared_down_buf;
    BnCudaBuffer *shared_gate_weight =
        (BnCudaBuffer *)shared_gate_weight_buf;
    if (getenv("BN_CUDA_DISABLE_MOE_FFN_BATCH"))
        return -1;
    if (!ctx || !out || !experts || !expert_offsets || !expert_counts ||
        !token_ids || !weights || !X || n_experts <= 0 || n_tokens <= 0 ||
        dim <= 0 || hidden_dim <= 0 || act_type != 0)
        return -1;
    int has_shared = shared_gate || shared_up || shared_down ||
                     shared_gate_weight || shared_hidden_dim > 0;
    int shared_gateup_stacked = has_shared && shared_gate && !shared_up;
    if (has_shared) {
        if (!shared_gate || !shared_down ||
            !shared_gate->data || !shared_down->data ||
            (!shared_gateup_stacked && (!shared_up || !shared_up->data)) ||
            shared_hidden_dim <= 0 ||
            shared_gate->rows != (shared_gateup_stacked
                                      ? shared_hidden_dim * 2
                                      : shared_hidden_dim) ||
            shared_gate->cols != dim ||
            (!shared_gateup_stacked &&
             (shared_up->rows != shared_hidden_dim ||
              shared_up->cols != dim)) ||
            shared_down->rows != dim ||
            shared_down->cols != shared_hidden_dim ||
            !cuda_type_supported(shared_gate_type) ||
            (!shared_gateup_stacked && !cuda_type_supported(shared_up_type)) ||
            !cuda_type_supported(shared_down_type))
            return -1;
        if (shared_gateup_stacked && shared_gate_type != shared_up_type)
            return -1;
        if (shared_gate_weight &&
            (!shared_gate_weight->data ||
             shared_gate_weight->type != BN_GGUF_TENSOR_F32 ||
             shared_gate_weight->rows * shared_gate_weight->cols < dim))
            return -1;
    }

    int total_assignments = 0;
    int max_count = 0;
    int all_identity_tokens = 1;
    for (int e = 0; e < n_experts; e++) {
        if (expert_counts[e] < 0 || expert_offsets[e] < 0)
            return -1;
        total_assignments += expert_counts[e];
        if (expert_counts[e] > max_count)
            max_count = expert_counts[e];
        if (expert_counts[e] != n_tokens) {
            all_identity_tokens = 0;
        } else {
            for (int i = 0; i < expert_counts[e]; i++) {
                if (token_ids[expert_offsets[e] + i] != i) {
                    all_identity_tokens = 0;
                    break;
                }
            }
        }
    }
    if (total_assignments <= 0 || max_count <= 0)
        return -1;

    int profile =
        getenv("BN_CUDA_PROFILE_MOE_FFN_BATCH_INTERNAL") != NULL;
    cudaEvent_t ev_start = NULL;
    cudaEvent_t ev_stop = NULL;
    double prof_upload_ms = 0.0;
    double prof_assign_ms = 0.0;
    double prof_expert_ms = 0.0;
    double prof_shared_ms = 0.0;
    double prof_readback_ms = 0.0;
    if (profile) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
    }
#define BN_CUDA_MOE_FFN_PROFILE_START() do { \
        if (profile) cudaEventRecord(ev_start, ctx->exec_stream); \
    } while (0)
#define BN_CUDA_MOE_FFN_PROFILE_STOP(accum) do { \
        if (profile) { \
            cudaEventRecord(ev_stop, ctx->exec_stream); \
            cudaEventSynchronize(ev_stop); \
            float _ms = 0.0f; \
            cudaEventElapsedTime(&_ms, ev_start, ev_stop); \
            (accum) += (double)_ms; \
        } \
    } while (0)

    size_t full_values = (size_t)n_tokens * dim;
    size_t full_bytes = full_values * sizeof(float);
    size_t gather_bytes = (size_t)max_count * dim * sizeof(float);
    int scratch_tokens = max_count;
    int scratch_hidden = hidden_dim;
    if (has_shared && n_tokens > scratch_tokens)
        scratch_tokens = n_tokens;
    if (has_shared && shared_hidden_dim > scratch_hidden)
        scratch_hidden = shared_hidden_dim;
    size_t hidden_bytes =
        (size_t)scratch_tokens * (size_t)scratch_hidden * sizeof(float);
    size_t gateup_bytes = hidden_bytes * 2u;
    size_t scratch_x = gather_bytes > hidden_bytes ? gather_bytes : hidden_bytes;
    size_t scratch_out = gateup_bytes > gather_bytes ? gateup_bytes : gather_bytes;
    if (cuda_ensure_scratch(ctx, scratch_x, scratch_out) != 0)
        return -1;
    if (cuda_ensure_prefill(ctx, full_values * 2u) != 0)
        return -1;

    float *d_full_x = ctx->d_prefill;
    float *d_full_out = d_full_x + full_values;
    BN_CUDA_MOE_FFN_PROFILE_START();
    cudaError_t err = cudaMemcpy(d_full_x, X, full_bytes,
                                 cudaMemcpyHostToDevice);
    if (err == cudaSuccess && !all_identity_tokens)
        err = cudaMemset(d_full_out, 0, full_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] moe ffn input upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_MOE_FFN_PROFILE_STOP(prof_upload_ms);

    size_t assignment_int_bytes =
        all_identity_tokens ? 0u : (size_t)total_assignments * sizeof(int);
    size_t assignment_float_bytes =
        (size_t)total_assignments * sizeof(float);
    if (cuda_ensure_ops(ctx, assignment_int_bytes + assignment_float_bytes) != 0)
        return -1;
    int *d_token_ids = all_identity_tokens ? NULL : (int *)ctx->d_ops;
    float *d_weights = (float *)((uint8_t *)ctx->d_ops +
                                 assignment_int_bytes);
    BN_CUDA_MOE_FFN_PROFILE_START();
    if (!all_identity_tokens) {
        err = cudaMemcpy(d_token_ids, token_ids, assignment_int_bytes,
                         cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess)
        err = cudaMemcpy(d_weights, weights, assignment_float_bytes,
                         cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] moe ffn assignment upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_MOE_FFN_PROFILE_STOP(prof_assign_ms);

    int threads = 256;
    for (int e = 0; e < n_experts; e++) {
        int count = expert_counts[e];
        if (count <= 0)
            continue;
        BnCudaBuffer *gate = (BnCudaBuffer *)experts[e].gate_buf;
        BnCudaBuffer *up = experts[e].use_gateup_split
            ? NULL : (BnCudaBuffer *)experts[e].up_buf;
        BnCudaBuffer *down = (BnCudaBuffer *)experts[e].down_buf;
        if (!gate || !down || (!experts[e].use_gateup_split && !up)) {
            return -1;
        }

        int identity_tokens = count == n_tokens;
        if (identity_tokens) {
            for (int i = 0; i < count; i++) {
                if (token_ids[expert_offsets[e] + i] != i) {
                    identity_tokens = 0;
                    break;
                }
            }
        }

        int gather_total = count * dim;
        const float *expert_x = d_full_x;
        if (!identity_tokens) {
            moe_gather_kernel<<<(gather_total + threads - 1) / threads,
                                threads>>>(
                ctx->d_x, d_full_x, d_token_ids, expert_offsets[e], count, dim);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] moe ffn gather failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
            expert_x = ctx->d_x;
        }
        BN_CUDA_MOE_FFN_PROFILE_START();
        if (cuda_dense_ffn_batch_device(ctx, ctx->d_out, gate, up, down,
                                        expert_x, count, dim, hidden_dim,
                                        gate_type, up_type, down_type,
                                        act_type) != 0) {
            return -1;
        }
        if (identity_tokens) {
            moe_identity_weighted_add_kernel<<<
                (gather_total + threads - 1) / threads, threads>>>(
                d_full_out, ctx->d_out, d_weights, expert_offsets[e],
                gather_total, dim, all_identity_tokens && e == 0);
        } else {
            moe_scatter_add_kernel<<<(gather_total + threads - 1) / threads,
                                     threads>>>(
                d_full_out, ctx->d_out, d_token_ids, d_weights,
                expert_offsets[e], count, dim);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] moe ffn scatter failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        BN_CUDA_MOE_FFN_PROFILE_STOP(prof_expert_ms);
    }

    if (has_shared) {
        BN_CUDA_MOE_FFN_PROFILE_START();
        if (cuda_dense_ffn_batch_device(
                ctx, ctx->d_out, shared_gate, shared_up, shared_down,
                d_full_x, n_tokens, dim, shared_hidden_dim,
                shared_gate_type, shared_up_type, shared_down_type,
                act_type) != 0) {
            return -1;
        }
        int total = n_tokens * dim;
        if (shared_gate_weight) {
            shared_expert_add_sigmoid_batch_kernel<<<n_tokens, threads>>>(
                d_full_out, ctx->d_out,
                (const float *)shared_gate_weight->data, d_full_x,
                n_tokens, dim);
        } else {
            residual_add_kernel<<<(total + threads - 1) / threads, threads>>>(
                d_full_out, ctx->d_out, total);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "[bn:gpu:cuda] moe shared expert add failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        BN_CUDA_MOE_FFN_PROFILE_STOP(prof_shared_ms);
    }

    if (cuda_ensure_host_out(ctx, full_bytes) != 0)
        return -1;
    BN_CUDA_MOE_FFN_PROFILE_START();
    err = cudaMemcpy(ctx->h_out, d_full_out, full_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] moe ffn output readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_MOE_FFN_PROFILE_STOP(prof_readback_ms);
    memcpy(out, ctx->h_out, full_bytes);
    if (profile) {
        static unsigned long long calls = 0;
        static double upload_ms = 0.0;
        static double assign_ms = 0.0;
        static double expert_ms = 0.0;
        static double shared_ms = 0.0;
        static double readback_ms = 0.0;
        calls++;
        upload_ms += prof_upload_ms;
        assign_ms += prof_assign_ms;
        expert_ms += prof_expert_ms;
        shared_ms += prof_shared_ms;
        readback_ms += prof_readback_ms;
        int every = cuda_env_int("BN_CUDA_PROFILE_MOE_FFN_BATCH_EVERY", 24);
        if (every <= 0) every = 24;
        if ((calls % (unsigned long long)every) == 0) {
            fprintf(stderr,
                    "[bn:gpu:cuda:moe-ffn-batch] calls=%llu upload=%.3fms assign=%.3f expert=%.3f shared=%.3f readback=%.3f\n",
                    calls, upload_ms, assign_ms, expert_ms, shared_ms,
                    readback_ms);
        }
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
    }
#undef BN_CUDA_MOE_FFN_PROFILE_START
#undef BN_CUDA_MOE_FFN_PROFILE_STOP
    return 0;
}

static int cuda_moe_route_batch(void *vctx, int *indices, float *weights,
                                void *router_buf, const float *X,
                                int n_tokens, int dim, int n_experts,
                                int k, int norm_topk_prob,
                                float expert_weights_scale) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *router = (BnCudaBuffer *)router_buf;
    if (getenv("BN_CUDA_DISABLE_MOE_ROUTE_BATCH"))
        return -1;
    if (!ctx || !indices || !weights || !router || !router->data || !X ||
        n_tokens <= 0 || dim <= 0 || n_experts <= 0 || k <= 0 ||
        k > BN_MAX_MOE_K || router->type != BN_GGUF_TENSOR_F32 ||
        router->rows < n_experts || router->cols < dim)
        return -1;

    size_t x_values = (size_t)n_tokens * (size_t)dim;
    size_t logits_values = (size_t)n_tokens * (size_t)n_experts;
    if (cuda_ensure_prefill(ctx, x_values + logits_values) != 0)
        return -1;
    size_t route_items = (size_t)n_tokens * (size_t)k;
    size_t idx_bytes = route_items * sizeof(int);
    size_t weight_bytes = route_items * sizeof(float);
    if (cuda_ensure_ops(ctx, idx_bytes + weight_bytes) != 0)
        return -1;

    float *d_x = ctx->d_prefill;
    float *d_logits = d_x + x_values;
    int *d_indices = (int *)ctx->d_ops;
    float *d_weights = (float *)((uint8_t *)ctx->d_ops + idx_bytes);

    cudaError_t err = cudaMemcpy(d_x, X, x_values * sizeof(float),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] moe route batch upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int threads = 256;
    int warps = threads / 32;
    int route_tasks = n_tokens * n_experts;
    int route_blocks = (route_tasks + warps - 1) / warps;
    moe_router_logits_batch_warp_kernel<<<route_blocks, threads, 0>>>(
        d_logits, (const float *)router->data, d_x, n_tokens, n_experts, dim);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        int topk_threads = 128;
        int topk_warps = topk_threads / 32;
        int topk_blocks = (n_tokens + topk_warps - 1) / topk_warps;
        moe_route_topk_batch_warp_kernel<<<topk_blocks, topk_threads, 0>>>(
            d_indices, d_weights, d_logits, n_tokens, n_experts, k,
            norm_topk_prob, expert_weights_scale);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess)
        err = cudaMemcpy(indices, d_indices, idx_bytes, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess)
        err = cudaMemcpy(weights, d_weights, weight_bytes,
                         cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] moe route batch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static int cuda_moe_routed_ffn_batch(void *vctx, float *out,
                                     void *gate_all_buf, void *up_all_buf,
                                     void *down_all_buf,
                                     const int *indices,
                                     const float *weights,
                                     const float *X,
                                     int n_tokens, int dim, int hidden_dim,
                                     int n_experts, int k,
                                     int gate_type, int up_type,
                                     int down_type, int act_type) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *gate = (BnCudaBuffer *)gate_all_buf;
    BnCudaBuffer *up = (BnCudaBuffer *)up_all_buf;
    BnCudaBuffer *down = (BnCudaBuffer *)down_all_buf;
    if (getenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN_BATCH"))
        return -1;
    int routed_q4 = gate_type == BN_GGUF_TENSOR_Q4_K &&
                    up_type == BN_GGUF_TENSOR_Q4_K &&
                    (down_type == BN_GGUF_TENSOR_Q4_K ||
                     down_type == BN_GGUF_TENSOR_Q6_K);
    int routed_q8 = gate_type == BN_GGUF_TENSOR_Q8_0 &&
                    up_type == BN_GGUF_TENSOR_Q8_0 &&
                    down_type == BN_GGUF_TENSOR_Q8_0;
    if (!ctx || !out || !gate || !up || !down || !indices || !weights ||
        !X || !gate->data || !up->data || !down->data ||
        n_tokens <= 0 || dim <= 0 || hidden_dim <= 0 ||
        n_experts <= 0 || k <= 0 || act_type != 0 ||
        (!routed_q4 && !routed_q8) ||
        (dim % 32) != 0 || (hidden_dim % 32) != 0)
        return -1;
    if (gate->type != gate_type || up->type != up_type ||
        down->type != down_type ||
        gate->rows < hidden_dim * n_experts || gate->cols < dim ||
        up->rows < hidden_dim * n_experts || up->cols < dim ||
        down->rows < dim * n_experts || down->cols < hidden_dim)
        return -1;

    size_t full_values = (size_t)n_tokens * (size_t)dim;
    size_t full_bytes = full_values * sizeof(float);
    size_t mid_values = (size_t)n_tokens * (size_t)k *
                        (size_t)hidden_dim;
    size_t mid_bytes = mid_values * sizeof(float);
    if (cuda_ensure_scratch(ctx, sizeof(float), mid_bytes) != 0)
        return -1;
    if (cuda_ensure_prefill(ctx, full_values * 2u) != 0)
        return -1;

    size_t route_items = (size_t)n_tokens * (size_t)k;
    size_t idx_bytes = route_items * sizeof(int);
    size_t weight_bytes = route_items * sizeof(float);
    int use_cublas_all2_decode =
        n_tokens == 1 && routed_q4 && n_experts == 2 && k == 2 &&
        down_type == BN_GGUF_TENSOR_Q6_K && hidden_dim >= 4096 &&
        gate->f16_data && up->f16_data && down->f16_data &&
        getenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE") == NULL;
    size_t decode_route_bytes =
        use_cublas_all2_decode ? 4u * sizeof(float) : 0u;
    if (cuda_ensure_ops(ctx, idx_bytes + weight_bytes + decode_route_bytes) != 0)
        return -1;

    float *d_full_x = ctx->d_prefill;
    float *d_full_out = d_full_x + full_values;
    float *d_mid = ctx->d_out;
    int *d_indices = (int *)ctx->d_ops;
    float *d_weights = (float *)((uint8_t *)ctx->d_ops + idx_bytes);
    float *d_decode_route = use_cublas_all2_decode
        ? (float *)((uint8_t *)ctx->d_ops + idx_bytes + weight_bytes)
        : NULL;

    cudaError_t err = cudaMemcpy(d_full_x, X, full_bytes,
                                 cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemset(d_full_out, 0, full_bytes);
    if (err == cudaSuccess)
        err = cudaMemcpy(d_indices, indices, idx_bytes,
                         cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(d_weights, weights, weight_bytes,
                         cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] routed moe ffn upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    if (use_cublas_all2_decode) {
        float h_route[4] = {
            weights[0], weights[1], (float)indices[0], (float)indices[1]
        };
        err = cudaMemcpy(d_decode_route, h_route, sizeof(h_route),
                         cudaMemcpyHostToDevice);
        if (err == cudaSuccess &&
            cuda_moe_cublas_all2_decode(ctx, d_full_out, gate, up, down,
                                        d_full_x, d_decode_route, dim,
                                        hidden_dim, 1) == 0) {
            if (cuda_ensure_host_out(ctx, full_bytes) != 0)
                return -1;
            err = cudaMemcpy(ctx->h_out, d_full_out, full_bytes,
                             cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                memcpy(out, ctx->h_out, full_bytes);
                return 0;
            }
        }
        d_mid = ctx->d_out;
        if (getenv("BN_CUDA_DEBUG_MOE_CUBLAS_DECODE"))
            fprintf(stderr,
                    "[bn:gpu:cuda] all2 cublas moe decode failed; falling back\n");
    }

    int threads = 256;
    int warps = threads / 32;
    int gateup_tasks = n_tokens * k * hidden_dim;
    int gateup_blocks = (gateup_tasks + warps - 1) / warps;
    int use_q4k_q8k_gateup =
        getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT") == NULL &&
        (n_tokens <= 1 || dim <= 2048 ||
         getenv("BN_CUDA_ENABLE_Q4K_Q8K_MOE_GATEUP") != NULL);

    int use_q8_q8_1_batch =
        routed_q8 && getenv("BN_CUDA_DISABLE_Q8_MOE_BATCH_Q8_1") == NULL;
    int prefer_q6k_quant_down = cuda_prefer_q6k_moe_quant_down(
        routed_q4, down_type, hidden_dim, n_experts, k);
    if (use_q8_q8_1_batch) {
        int x_blocks = dim / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        quantize_q8_1_batch_kernel<<<dim3(x_blocks, n_tokens, 1), 32, 0>>>(
            xq, d_full_x, dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) return -1;
        if (hidden_dim <= 1024 &&
            getenv("BN_CUDA_DISABLE_Q8_MOE_GATEUP_2ROW") == NULL) {
            int gateup2_blocks =
                (gateup_tasks + warps * 2 - 1) / (warps * 2);
            moe_q8_0_gateup_routed_mid_q8_1_2row_batch_kernel<<<gateup2_blocks, threads, 0>>>(
                d_mid, (const BnBlockQ8_0 *)gate->data,
                (const BnBlockQ8_0 *)up->data, xq, d_indices,
                hidden_dim, dim, n_experts, k, n_tokens);
        } else {
            moe_q8_0_gateup_routed_mid_q8_1_batch_kernel<<<gateup_blocks, threads, 0>>>(
                d_mid, (const BnBlockQ8_0 *)gate->data,
                (const BnBlockQ8_0 *)up->data, xq, d_indices,
                hidden_dim, dim, n_experts, k, n_tokens);
        }
    } else if (routed_q8) {
        moe_q8_0_gateup_routed_mid_batch_kernel<<<gateup_blocks, threads, 0>>>(
            d_mid, (const BnBlockQ8_0 *)gate->data,
            (const BnBlockQ8_0 *)up->data, d_full_x, d_indices,
            d_weights, hidden_dim, dim, n_experts, k, n_tokens);
    } else {
        if (use_q4k_q8k_gateup) {
            if (cuda_ensure_q8_k(ctx, dim, n_tokens) != 0)
                return -1;
            BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
            quantize_q8k_batch_kernel<<<dim3(dim / BN_QK_K, n_tokens, 1),
                                        BN_QK_K, 0>>>(
                xq, d_full_x, dim, n_tokens);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] routed moe x q8k quantize failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
            int use_gateup_8row =
                dim <= 2048 &&
                getenv("BN_CUDA_DISABLE_MOE_GATEUP_8ROW") == NULL;
            int use_gateup_split =
                dim <= 2048 && n_experts > 2 &&
                getenv("BN_CUDA_ENABLE_MOE_GATEUP_SPLIT") != NULL &&
                getenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT") == NULL;
            if (use_gateup_split) {
                int gateup2_blocks =
                    (gateup_tasks + warps * 2 - 1) / (warps * 2);
                moe_q4k_gateup_routed_mid_q8k_2row_split_batch_kernel<<<gateup2_blocks, threads, 0>>>(
                    d_mid, (const BnBlockQ4K *)gate->data,
                    (const BnBlockQ4K *)up->data, xq, d_indices,
                    hidden_dim, dim, n_experts, k, n_tokens);
            } else if (use_gateup_8row) {
                int gateup8_blocks =
                    (gateup_tasks + warps * 8 - 1) / (warps * 8);
                moe_q4k_gateup_routed_mid_q8k_8row_batch_kernel<<<gateup8_blocks, threads, 0>>>(
                    d_mid, (const BnBlockQ4K *)gate->data,
                    (const BnBlockQ4K *)up->data, xq, d_indices,
                    hidden_dim, dim, n_experts, k, n_tokens);
            } else {
                int gateup4_tasks = (gateup_tasks + 3) / 4;
                int gateup4_blocks = (gateup4_tasks + warps - 1) / warps;
                moe_q4k_gateup_routed_mid_q8k_4row_batch_kernel<<<gateup4_blocks, threads, 0>>>(
                    d_mid, (const BnBlockQ4K *)gate->data,
                    (const BnBlockQ4K *)up->data, xq, d_indices,
                    hidden_dim, dim, n_experts, k, n_tokens);
            }
        } else {
            int x_blocks = (dim + 31) / 32;
            if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
                return -1;
            BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
            quantize_q8_1_batch_kernel<<<dim3(x_blocks, n_tokens, 1), 32, 0>>>(
                xq, d_full_x, dim, n_tokens);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] routed moe x quantize failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
            moe_q4k_gateup_routed_mid_batch_kernel<<<gateup_blocks, threads, 0>>>(
                d_mid, (const BnBlockQ4K *)gate->data,
                (const BnBlockQ4K *)up->data, xq, d_indices, d_weights,
                hidden_dim, dim, n_experts, k, n_tokens);
        }
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] routed moe gate/up failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int down_tasks = n_tokens * dim;
    int down_blocks = (down_tasks + warps - 1) / warps;
    if (use_q8_q8_1_batch) {
        int n_mid = n_tokens * k;
        int mid_blocks = hidden_dim / 32;
        if (cuda_ensure_q8_1(ctx, mid_blocks * 32 * n_mid) != 0)
            return -1;
        BnCudaBlockQ8_1 *mid_q = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        quantize_q8_1_batch_kernel<<<dim3(mid_blocks, n_mid, 1), 32, 0>>>(
            mid_q, d_mid, hidden_dim, n_mid);
        err = cudaGetLastError();
        if (err != cudaSuccess) return -1;
        if (hidden_dim <= 1024 &&
            getenv("BN_CUDA_ENABLE_Q8_MOE_DOWN_4ROW") != NULL) {
            int down4_blocks = (down_tasks + warps * 4 - 1) / (warps * 4);
            moe_q8_0_down_routed_q8_1_accum_4row_batch_kernel<<<down4_blocks, threads, 0>>>(
                d_full_out, (const BnBlockQ8_0 *)down->data, mid_q,
                d_indices, d_weights, dim, hidden_dim, n_experts, k,
                n_tokens);
        } else if (hidden_dim <= 1024 &&
            getenv("BN_CUDA_DISABLE_Q8_MOE_DOWN_2ROW") == NULL) {
            int down2_blocks = (down_tasks + warps * 2 - 1) / (warps * 2);
            moe_q8_0_down_routed_q8_1_accum_2row_batch_kernel<<<down2_blocks, threads, 0>>>(
                d_full_out, (const BnBlockQ8_0 *)down->data, mid_q,
                d_indices, d_weights, dim, hidden_dim, n_experts, k,
                n_tokens);
        } else {
            moe_q8_0_down_routed_q8_1_accum_batch_kernel<<<down_blocks, threads, 0>>>(
                d_full_out, (const BnBlockQ8_0 *)down->data, mid_q,
                d_indices, d_weights, dim, hidden_dim, n_experts, k,
                n_tokens);
        }
    } else if (routed_q8) {
        moe_q8_0_down_routed_accum_batch_kernel<<<down_blocks, threads, 0>>>(
            d_full_out, (const BnBlockQ8_0 *)down->data, d_mid, d_indices,
            d_weights, dim, hidden_dim, n_experts, k, n_tokens);
    } else if (down_type == BN_GGUF_TENSOR_Q6_K && down->f32_data &&
               !prefer_q6k_quant_down &&
               !(routed_q4 && n_experts == 2 && k == 2 &&
                 hidden_dim >= 4096 && dim <= 2048) &&
               getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL) {
        moe_q6k_down_routed_f32_cache_batch_kernel<<<down_blocks, threads, 0>>>(
            d_full_out, (const float *)down->f32_data, d_mid, d_indices,
            d_weights, dim, hidden_dim, n_experts, k, n_tokens);
    } else {
        int n_mid = n_tokens * k;
        if (cuda_ensure_q8_k(ctx, hidden_dim, n_mid) != 0)
            return -1;
        BnBlockQ8K *mid_q = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(hidden_dim / BN_QK_K, n_mid, 1),
                                    BN_QK_K>>>(
            mid_q, d_mid, hidden_dim, n_mid);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "[bn:gpu:cuda] routed moe mid quantize failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        int use_down_4row =
            hidden_dim <= 1024 &&
            getenv("BN_CUDA_DISABLE_MOE_DOWN_4ROW") == NULL;
        if (use_down_4row) {
            int down4_tasks = n_tokens * dim;
            int down4_blocks = (down4_tasks + warps * 4 - 1) / (warps * 4);
            int use_down_8row =
                hidden_dim <= 1024 &&
                getenv("BN_CUDA_DISABLE_MOE_DOWN_8ROW") == NULL;
            if (use_down_8row) {
                int use_halfwarp =
                    down_type == BN_GGUF_TENSOR_Q6_K &&
                    (n_experts > 2 || k > 2) &&
                    (prefer_q6k_quant_down ||
                     getenv("BN_CUDA_ENABLE_MOE_DOWN_HALFWARP") != NULL) &&
                    getenv("BN_CUDA_DISABLE_MOE_DOWN_HALFWARP") == NULL;
                int use_split4 =
                    !use_halfwarp &&
                    down_type == BN_GGUF_TENSOR_Q6_K &&
                    (n_experts > 2 || k > 2) &&
                    getenv("BN_CUDA_ENABLE_MOE_DOWN_SPLIT4") != NULL &&
                    getenv("BN_CUDA_DISABLE_MOE_DOWN_SPLIT4") == NULL;
                int use_scatter =
                    !use_halfwarp && !use_split4 &&
                    down_type == BN_GGUF_TENSOR_Q6_K &&
                    getenv("BN_CUDA_DISABLE_MOE_DOWN_SCATTER") == NULL;
                int use_scatter_16row =
                    use_scatter && hidden_dim <= 768 &&
                    getenv("BN_CUDA_ENABLE_MOE_DOWN_SCATTER_16ROW") != NULL;
                int down8_tasks = use_scatter ? n_tokens * k * dim : down4_tasks;
                int down8_blocks = (down8_tasks + warps * 8 - 1) / (warps * 8);
                if (use_halfwarp) {
                    int down2_blocks = (down4_tasks + warps * 2 - 1) / (warps * 2);
                    moe_q6k_down_routed_q8k_accum_halfwarp_batch_kernel<<<down2_blocks, threads, 0>>>(
                        d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                        d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                } else if (use_split4) {
                    int down_split4_blocks =
                        (down4_tasks + warps * 4 - 1) / (warps * 4);
                    moe_q6k_down_routed_q8k_accum_4row_split_batch_kernel<<<down_split4_blocks, threads, 0>>>(
                        d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                        d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                } else if (use_scatter) {
                    if (use_scatter_16row) {
                        int down16_blocks =
                            (down8_tasks + warps * 16 - 1) / (warps * 16);
                        moe_q6k_down_routed_q8k_scatter_16row_batch_kernel<<<down16_blocks, threads, 0>>>(
                            d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                            d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                    } else {
                        moe_q6k_down_routed_q8k_scatter_8row_batch_kernel<<<down8_blocks, threads, 0>>>(
                            d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                            d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                    }
                } else if (down_type == BN_GGUF_TENSOR_Q6_K) {
                    moe_q6k_down_routed_q8k_accum_8row_batch_kernel<<<down8_blocks, threads, 0>>>(
                        d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                        d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                } else {
                    moe_q4k_down_routed_q8k_accum_8row_batch_kernel<<<down8_blocks, threads, 0>>>(
                        d_full_out, (const BnBlockQ4K *)down->data, mid_q,
                        d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                }
            } else if (down_type == BN_GGUF_TENSOR_Q6_K) {
                moe_q6k_down_routed_q8k_accum_4row_batch_kernel<<<down4_blocks, threads, 0>>>(
                    d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                    d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
            } else {
                moe_q4k_down_routed_q8k_accum_4row_batch_kernel<<<down4_blocks, threads, 0>>>(
                    d_full_out, (const BnBlockQ4K *)down->data, mid_q,
                    d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
            }
        } else if (down_type == BN_GGUF_TENSOR_Q6_K) {
            moe_q6k_down_routed_q8k_accum_batch_kernel<<<down_blocks, threads, 0>>>(
                d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
        } else {
            moe_q4k_down_routed_q8k_accum_batch_kernel<<<down_blocks, threads, 0>>>(
                d_full_out, (const BnBlockQ4K *)down->data, mid_q,
                d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
        }
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] routed moe down failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    if (cuda_ensure_host_out(ctx, full_bytes) != 0)
        return -1;
    err = cudaMemcpy(ctx->h_out, d_full_out, full_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] routed moe output readback failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    memcpy(out, ctx->h_out, full_bytes);
    return 0;
}

static int cuda_moe_route_routed_ffn_batch_impl(
    void *vctx, float *out, void *router_buf, void *gate_all_buf,
    void *up_all_buf, void *down_all_buf, void *shared_gate_buf,
    void *shared_up_buf, void *shared_down_buf,
    void *shared_gate_weight_buf, void *norm_buf, const float *X,
    int n_tokens, int dim, int hidden_dim, int n_experts, int k,
    int gate_type, int up_type, int down_type, int act_type,
    int shared_hidden_dim, int shared_gate_type, int shared_up_type,
    int shared_down_type,
    float norm_eps, int add_norm_resid, int norm_topk_prob,
    float expert_weights_scale) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *router = (BnCudaBuffer *)router_buf;
    BnCudaBuffer *gate = (BnCudaBuffer *)gate_all_buf;
    BnCudaBuffer *up = (BnCudaBuffer *)up_all_buf;
    BnCudaBuffer *down = (BnCudaBuffer *)down_all_buf;
    BnCudaBuffer *shared_gate = (BnCudaBuffer *)shared_gate_buf;
    BnCudaBuffer *shared_up = (BnCudaBuffer *)shared_up_buf;
    BnCudaBuffer *shared_down = (BnCudaBuffer *)shared_down_buf;
    BnCudaBuffer *shared_gate_weight =
        (BnCudaBuffer *)shared_gate_weight_buf;
    BnCudaBuffer *norm = (BnCudaBuffer *)norm_buf;
    if (getenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH"))
        return -1;
    if (n_experts > 2 &&
        getenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE") == NULL)
        return -1;
    int routed_q4 = gate_type == BN_GGUF_TENSOR_Q4_K &&
                    up_type == BN_GGUF_TENSOR_Q4_K &&
                    (down_type == BN_GGUF_TENSOR_Q4_K ||
                     down_type == BN_GGUF_TENSOR_Q6_K);
    int routed_q8 = gate_type == BN_GGUF_TENSOR_Q8_0 &&
                    up_type == BN_GGUF_TENSOR_Q8_0 &&
                    down_type == BN_GGUF_TENSOR_Q8_0;
    if (!ctx || !router || !gate || !up || !down || (!X && !ctx->d_out) ||
        !router->data || !gate->data || !up->data || !down->data ||
        n_tokens <= 0 || dim <= 0 || hidden_dim <= 0 ||
        n_experts <= 0 || k <= 0 || k > BN_MAX_MOE_K || act_type != 0 ||
        router->type != BN_GGUF_TENSOR_F32 ||
        router->rows < n_experts || router->cols < dim ||
        (!routed_q4 && !routed_q8) ||
        (dim % 32) != 0 || (hidden_dim % 32) != 0)
        return -1;
    if (add_norm_resid && (!norm || !norm->data ||
                           norm->rows * norm->cols < dim))
        return -1;
    int has_shared = shared_gate || shared_up || shared_down ||
                     shared_gate_weight || shared_hidden_dim > 0;
    if (has_shared) {
        if (!shared_gate || !shared_up || !shared_down ||
            !shared_gate->data || !shared_up->data || !shared_down->data ||
            shared_hidden_dim <= 0 ||
            shared_gate->type != shared_gate_type ||
            shared_up->type != shared_up_type ||
            shared_down->type != shared_down_type ||
            shared_gate->rows != shared_hidden_dim ||
            shared_gate->cols != dim ||
            shared_up->rows != shared_hidden_dim ||
            shared_up->cols != dim ||
            shared_down->rows != dim ||
            shared_down->cols != shared_hidden_dim ||
            !cuda_type_supported(shared_gate_type) ||
            !cuda_type_supported(shared_up_type) ||
            !cuda_type_supported(shared_down_type))
            return -1;
        if (shared_gate_weight &&
            (!shared_gate_weight->data ||
             shared_gate_weight->type != BN_GGUF_TENSOR_F32 ||
             shared_gate_weight->rows * shared_gate_weight->cols < dim))
            return -1;
    }
    if (gate->type != gate_type || up->type != up_type ||
        down->type != down_type ||
        gate->rows < hidden_dim * n_experts || gate->cols < dim ||
        up->rows < hidden_dim * n_experts || up->cols < dim ||
        down->rows < dim * n_experts || down->cols < hidden_dim)
        return -1;

    size_t full_values = (size_t)n_tokens * (size_t)dim;
    size_t full_bytes = full_values * sizeof(float);
    size_t logits_values = (size_t)n_tokens * (size_t)n_experts;
    size_t mid_values = (size_t)n_tokens * (size_t)k *
                        (size_t)hidden_dim;
    size_t moe_scratch_out_bytes = mid_values * sizeof(float);
    const float *d_resident_input = NULL;
    if (!X &&
        cuda_prepare_resident_input(ctx, full_bytes, moe_scratch_out_bytes,
                                    &d_resident_input) != 0)
        return -1;
    if (cuda_ensure_scratch(ctx, sizeof(float),
                            moe_scratch_out_bytes) != 0)
        return -1;
    size_t prefill_values =
        add_norm_resid ? full_values * 3u + logits_values
                       : full_values * 2u + logits_values;
    size_t route_items = (size_t)n_tokens * (size_t)k;
    size_t idx_bytes = route_items * sizeof(int);
    size_t weight_bytes = route_items * sizeof(float);
    int use_cublas_grouped =
        gate->f16_data && up->f16_data && down->f16_data &&
        ((routed_q8 &&
          getenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GROUPED") == NULL) ||
         (routed_q4 &&
          getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED") == NULL));
    if (use_cublas_grouped && routed_q4 && !(n_experts == 2 && k == 2) &&
        (n_tokens * k) <= 256 &&
        getenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_SMALL") == NULL) {
        use_cublas_grouped = 0;
    }
    int use_cublas_gateup_only =
        !use_cublas_grouped && gate->f16_data && up->f16_data &&
        !down->f16_data && n_tokens > 1 &&
        ((routed_q8 &&
          getenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GATEUP") == NULL) ||
         (routed_q4 &&
          getenv("BN_CUDA_ENABLE_MOE_CUBLAS_GATEUP") != NULL &&
          getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP") == NULL));
    int use_cublas_all2_fixed =
        use_cublas_grouped && n_experts == 2 && k == 2 &&
        getenv("BN_CUDA_DISABLE_MOE_CUBLAS_ALL2_FIXED") == NULL;
    int use_cublas_all2_decode =
        n_tokens == 1 && routed_q4 && n_experts == 2 && k == 2 &&
        down_type == BN_GGUF_TENSOR_Q6_K && hidden_dim >= 4096 &&
        gate->f16_data && up->f16_data && down->f16_data &&
        getenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE") == NULL;
    int use_sorted_slots =
        (routed_q4 || routed_q8) && n_tokens > 1 &&
        !use_cublas_all2_fixed &&
        (use_cublas_grouped || use_cublas_gateup_only ||
         getenv("BN_CUDA_ENABLE_MOE_ROUTE_SORT") != NULL);
    size_t sorted_route_aux_bytes = use_sorted_slots
        ? (route_items * sizeof(int) + (size_t)n_experts * 4u * sizeof(int))
        : 0u;
    size_t decode_route_bytes =
        use_cublas_all2_decode ? 4u * sizeof(float) : 0u;
    size_t route_aux_bytes = sorted_route_aux_bytes + decode_route_bytes;

    int profile_prefill_moe =
        getenv("BN_CUDA_PROFILE_MOE_PREFILL_INTERNAL") != NULL;
    int prefer_q6k_quant_down = cuda_prefer_q6k_moe_quant_down(
        routed_q4, down_type, hidden_dim, n_experts, k);
    int init_out_with_residual =
        add_norm_resid &&
        ((routed_q8 && hidden_dim <= 1024) ||
         (down_type == BN_GGUF_TENSOR_Q6_K &&
          hidden_dim <= 1024 &&
          getenv("BN_CUDA_DISABLE_MOE_DOWN_4ROW") == NULL &&
          getenv("BN_CUDA_DISABLE_MOE_DOWN_8ROW") == NULL &&
          getenv("BN_CUDA_DISABLE_MOE_DOWN_SCATTER") == NULL &&
          !(down->f32_data && !prefer_q6k_quant_down &&
            getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL)));
    int direct_device_resid_out =
        add_norm_resid && !out && !has_shared && init_out_with_residual &&
        getenv("BN_CUDA_DISABLE_MOE_PREFILL_DIRECT_RESID_OUT") == NULL;
    if (direct_device_resid_out)
        prefill_values = full_values + logits_values + mid_values;
    if (cuda_ensure_prefill(ctx, prefill_values) != 0)
        return -1;
    if (cuda_ensure_ops(ctx, idx_bytes + weight_bytes + route_aux_bytes) != 0)
        return -1;

    float *d_residual = add_norm_resid ? ctx->d_prefill : NULL;
    float *d_full_x = add_norm_resid ? d_residual + full_values
                                     : ctx->d_prefill;
    float *d_full_out = d_full_x + full_values;
    float *d_logits = d_full_out + full_values;
    float *d_mid = ctx->d_out;
    if (direct_device_resid_out) {
        d_residual = ctx->d_out;
        d_full_x = ctx->d_prefill;
        d_logits = d_full_x + full_values;
        d_mid = d_logits + logits_values;
        d_full_out = ctx->d_out;
    }
    int *d_indices = (int *)ctx->d_ops;
    float *d_weights = (float *)((uint8_t *)ctx->d_ops + idx_bytes);
    uint8_t *d_route_aux = (uint8_t *)ctx->d_ops + idx_bytes + weight_bytes;
    int *d_slot_order = use_sorted_slots ? (int *)d_route_aux : NULL;
    int *d_expert_counts = use_sorted_slots
        ? (int *)(d_route_aux + route_items * sizeof(int)) : NULL;
    int *d_expert_offsets = use_sorted_slots
        ? d_expert_counts + n_experts : NULL;
    int *d_expert_fill = use_sorted_slots
        ? d_expert_offsets + n_experts : NULL;
    int *d_active_experts = use_sorted_slots
        ? d_expert_fill + n_experts : NULL;
    float *d_decode_route = use_cublas_all2_decode
        ? (float *)(d_route_aux + sorted_route_aux_bytes)
        : NULL;
    static double profile_totals[7] = {0.0};
    static unsigned long long profile_io_counts[4] = {0};
    static unsigned long long profile_path_counts[4] = {0};
    enum {
        BN_CUDA_MOE_PROFILE_X_HOST = 0,
        BN_CUDA_MOE_PROFILE_X_DEVICE = 1,
        BN_CUDA_MOE_PROFILE_OUT_HOST = 2,
        BN_CUDA_MOE_PROFILE_OUT_DEVICE = 3
    };
    enum {
        BN_CUDA_MOE_PROFILE_PATH_CUBLAS_GROUPED = 0,
        BN_CUDA_MOE_PROFILE_PATH_CUBLAS_GATEUP = 1,
        BN_CUDA_MOE_PROFILE_PATH_Q8_NATIVE = 2,
        BN_CUDA_MOE_PROFILE_PATH_QK_NATIVE = 3
    };
    static unsigned long long profile_calls = 0;
    double profile_t0 = profile_prefill_moe ? cuda_wall_ms() : 0.0;
#define BN_CUDA_MOE_PREFILL_PROFILE_STEP(code_) do {                 \
    if (profile_prefill_moe) {                                       \
        cudaError_t profile_err_ = cudaDeviceSynchronize();           \
        if (profile_err_ != cudaSuccess) return -1;                  \
        double profile_t1_ = cuda_wall_ms();                         \
        profile_totals[(code_)] += profile_t1_ - profile_t0;         \
        profile_t0 = profile_t1_;                                    \
    }                                                                \
} while (0)

    cudaError_t err = cudaSuccess;
    if (profile_prefill_moe) {
        if (X)
            profile_io_counts[BN_CUDA_MOE_PROFILE_X_HOST]++;
        else
            profile_io_counts[BN_CUDA_MOE_PROFILE_X_DEVICE]++;
    }
    if (X) {
        err = cudaMemcpy(add_norm_resid ? d_residual : d_full_x, X,
                         full_bytes, cudaMemcpyHostToDevice);
    } else if (add_norm_resid) {
        err = d_residual == d_resident_input
            ? cudaSuccess
            : cudaMemcpy(d_residual, d_resident_input, full_bytes,
                         cudaMemcpyDeviceToDevice);
    } else {
        err = cudaMemcpy(d_full_x, d_resident_input, full_bytes,
                         cudaMemcpyDeviceToDevice);
    }
    if (err == cudaSuccess && add_norm_resid) {
        int norm_threads = 256;
        int norm_warps = norm_threads / 32;
        rmsnorm_batch_kernel<<<n_tokens, norm_threads,
                               (size_t)norm_warps * sizeof(float)>>>(
            d_full_x, d_residual, (const float *)norm->data, dim,
            n_tokens, norm_eps);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        if (init_out_with_residual) {
            if (!direct_device_resid_out) {
                err = cudaMemcpy(d_full_out, d_residual, full_bytes,
                                 cudaMemcpyDeviceToDevice);
            }
        } else {
            err = cudaMemset(d_full_out, 0, full_bytes);
        }
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] routed moe combined upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_MOE_PREFILL_PROFILE_STEP(0);

    int threads = 256;
    int warps = threads / 32;
    int use_fused_batch_route =
        n_experts <= 256 &&
        getenv("BN_CUDA_ENABLE_MOE_BATCH_FUSED_ROUTE_TOPK") != NULL &&
        getenv("BN_CUDA_DISABLE_MOE_BATCH_FUSED_ROUTE_TOPK") == NULL;
    if (use_fused_batch_route) {
        moe_route_fused_batch_warp_topk_kernel<<<n_tokens, threads, 0>>>(
            d_indices, d_weights, (const float *)router->data, d_full_x,
            n_tokens, n_experts, dim, k, norm_topk_prob,
            expert_weights_scale);
        err = cudaGetLastError();
    } else {
        int route_tasks = n_tokens * n_experts;
        int route_blocks = (route_tasks + warps - 1) / warps;
        moe_router_logits_batch_warp_kernel<<<route_blocks, threads, 0>>>(
            d_logits, (const float *)router->data, d_full_x, n_tokens,
            n_experts, dim);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess && !use_fused_batch_route) {
        int topk_threads = 128;
        int topk_warps = topk_threads / 32;
        int topk_blocks = (n_tokens + topk_warps - 1) / topk_warps;
        moe_route_topk_batch_warp_kernel<<<topk_blocks, topk_threads, 0>>>(
            d_indices, d_weights, d_logits, n_tokens, n_experts, k,
            norm_topk_prob, expert_weights_scale);
        err = cudaGetLastError();
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] routed moe combined route failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    if (use_sorted_slots) {
        err = cudaMemset(d_expert_counts, 0,
                         (size_t)n_experts * sizeof(int));
        if (err == cudaSuccess) {
            int sort_threads = 128;
            int sort_blocks =
                ((int)route_items + sort_threads - 1) / sort_threads;
            moe_route_count_experts_kernel<<<sort_blocks, sort_threads>>>(
                d_expert_counts, d_indices, (int)route_items, n_experts);
            err = cudaGetLastError();
            if (err == cudaSuccess) {
                moe_route_offsets_kernel<<<1, 1>>>(
                    d_expert_offsets, d_expert_fill, d_expert_counts,
                    n_experts);
                err = cudaGetLastError();
            }
            if (err == cudaSuccess) {
                moe_route_fill_slots_kernel<<<sort_blocks, sort_threads>>>(
                    d_slot_order, d_expert_fill, d_indices, (int)route_items,
                    n_experts);
                err = cudaGetLastError();
            }
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "[bn:gpu:cuda] routed moe route sort failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (getenv("BN_CUDA_PROFILE_MOE_ROUTE_DIST")) {
        int *h_indices = (int *)malloc(route_items * sizeof(int));
        int *h_counts = (int *)calloc((size_t)n_experts, sizeof(int));
        if (h_indices && h_counts &&
            cudaMemcpy(h_indices, d_indices, idx_bytes,
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            int active = 0;
            int max_count = 0;
            int singleton = 0;
            for (size_t i = 0; i < route_items; i++) {
                int e = h_indices[i];
                if (e >= 0 && e < n_experts)
                    h_counts[e]++;
            }
            for (int e = 0; e < n_experts; e++) {
                if (h_counts[e] > 0) {
                    active++;
                    if (h_counts[e] == 1) singleton++;
                    if (h_counts[e] > max_count) max_count = h_counts[e];
                }
            }
            static unsigned long long dist_calls = 0;
            static unsigned long long dist_active = 0;
            static unsigned long long dist_singleton = 0;
            static unsigned long long dist_max_sum = 0;
            dist_calls++;
            dist_active += (unsigned long long)active;
            dist_singleton += (unsigned long long)singleton;
            dist_max_sum += (unsigned long long)max_count;
            int every = cuda_env_int("BN_CUDA_PROFILE_MOE_ROUTE_DIST_EVERY", 48);
            if (every <= 0) every = 48;
            if ((dist_calls % (unsigned long long)every) == 0) {
                fprintf(stderr,
                        "[bn:gpu:cuda:moe-route-dist] calls=%llu assign=%zu avg_active=%.2f avg_singleton=%.2f avg_max_per_expert=%.2f experts=%d\n",
                        dist_calls, route_items,
                        (double)dist_active / (double)dist_calls,
                        (double)dist_singleton / (double)dist_calls,
                        (double)dist_max_sum / (double)dist_calls,
                        n_experts);
            }
        }
        if (h_counts) free(h_counts);
        if (h_indices) free(h_indices);
    }
    BN_CUDA_MOE_PREFILL_PROFILE_STEP(1);

    int gateup_tasks = n_tokens * k * hidden_dim;
    int gateup_blocks = (gateup_tasks + warps - 1) / warps;
    int down_tasks = n_tokens * dim;
    int down_blocks = (down_tasks + warps - 1) / warps;
    int use_q4k_q8k_gateup =
        getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT") == NULL &&
        (n_tokens <= 1 ||
         getenv("BN_CUDA_ENABLE_Q4K_Q8K_MOE_GATEUP") != NULL);
    int use_q8_q8_1_batch =
        routed_q8 && getenv("BN_CUDA_DISABLE_Q8_MOE_BATCH_Q8_1") == NULL;

    if (use_cublas_all2_decode) {
        moe_pack_all2_route_kernel<<<1, 1>>>(d_decode_route, d_indices,
                                             d_weights);
        err = cudaGetLastError();
        if (err == cudaSuccess &&
            cuda_moe_cublas_all2_decode(ctx, d_full_out, gate, up, down,
                                        d_full_x, d_decode_route, dim,
                                        hidden_dim, 1) == 0) {
            err = cudaSuccess;
            if (profile_prefill_moe)
                profile_path_counts[
                    BN_CUDA_MOE_PROFILE_PATH_CUBLAS_GROUPED]++;
            BN_CUDA_MOE_PREFILL_PROFILE_STEP(3);
            goto moe_route_routed_readback;
        }
        d_mid = ctx->d_out;
        if (getenv("BN_CUDA_DEBUG_MOE_CUBLAS_DECODE"))
            fprintf(stderr,
                    "[bn:gpu:cuda] all2 cublas moe decode failed; falling back\n");
    } else if (use_cublas_all2_fixed) {
        if (cuda_moe_cublas_all2_prefill(
                ctx, d_full_out, gate, up, down, d_full_x, d_weights,
                d_indices, n_tokens, dim, hidden_dim, act_type) == 0) {
            err = cudaSuccess;
            if (profile_prefill_moe)
                profile_path_counts[
                    BN_CUDA_MOE_PROFILE_PATH_CUBLAS_GROUPED]++;
            BN_CUDA_MOE_PREFILL_PROFILE_STEP(3);
            goto moe_route_routed_readback;
        }
        if (getenv("BN_CUDA_DEBUG_MOE_CUBLAS_GROUPED"))
            fprintf(stderr,
                    "[bn:gpu:cuda] all2 cublas moe prefill failed; falling back\n");
    } else if (use_cublas_grouped) {
        if (cuda_moe_cublas_grouped_prefill(
                ctx, d_full_out, gate, up, down, d_full_x, d_weights,
                d_slot_order, d_expert_offsets, d_expert_counts,
                d_active_experts, n_tokens, dim, hidden_dim, n_experts, k,
                act_type) == 0) {
            err = cudaSuccess;
            if (profile_prefill_moe)
                profile_path_counts[
                    BN_CUDA_MOE_PROFILE_PATH_CUBLAS_GROUPED]++;
            BN_CUDA_MOE_PREFILL_PROFILE_STEP(3);
            goto moe_route_routed_readback;
        }
        if (getenv("BN_CUDA_DEBUG_MOE_CUBLAS_GROUPED"))
            fprintf(stderr,
                    "[bn:gpu:cuda] grouped cublas moe prefill failed; falling back\n");
    }

    if (use_cublas_gateup_only) {
        float *d_mid_gateup = d_mid;
        if (cuda_moe_cublas_grouped_gateup_prefill(
                ctx, &d_mid_gateup, gate, up, d_full_x, d_slot_order,
                d_expert_offsets, d_expert_counts, d_active_experts,
                n_tokens, dim, hidden_dim, n_experts, k, act_type) == 0) {
            d_mid = d_mid_gateup;
            err = cudaSuccess;
            if (profile_prefill_moe)
                profile_path_counts[
                    BN_CUDA_MOE_PROFILE_PATH_CUBLAS_GATEUP]++;
            BN_CUDA_MOE_PREFILL_PROFILE_STEP(3);
            goto moe_route_routed_down;
        }
        if (getenv("BN_CUDA_DEBUG_MOE_CUBLAS_GATEUP"))
            fprintf(stderr,
                    "[bn:gpu:cuda] grouped cublas moe gate/up failed; falling back\n");
    }

    if (use_q8_q8_1_batch) {
        if (profile_prefill_moe)
            profile_path_counts[BN_CUDA_MOE_PROFILE_PATH_Q8_NATIVE]++;
        int x_blocks = dim / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        quantize_q8_1_batch_kernel<<<dim3(x_blocks, n_tokens, 1), 32, 0>>>(
            xq, d_full_x, dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) return -1;
        BN_CUDA_MOE_PREFILL_PROFILE_STEP(2);
        if (hidden_dim <= 1024 &&
            getenv("BN_CUDA_DISABLE_Q8_MOE_GATEUP_2ROW") == NULL) {
            int gateup2_blocks =
                (gateup_tasks + warps * 2 - 1) / (warps * 2);
            if (use_sorted_slots) {
                moe_q8_0_gateup_routed_mid_q8_1_2row_sorted_batch_kernel<<<gateup2_blocks, threads, 0>>>(
                    d_mid, (const BnBlockQ8_0 *)gate->data,
                    (const BnBlockQ8_0 *)up->data, xq, d_indices,
                    d_slot_order, hidden_dim, dim, n_experts, k, n_tokens);
            } else {
                moe_q8_0_gateup_routed_mid_q8_1_2row_batch_kernel<<<gateup2_blocks, threads, 0>>>(
                    d_mid, (const BnBlockQ8_0 *)gate->data,
                    (const BnBlockQ8_0 *)up->data, xq, d_indices,
                    hidden_dim, dim, n_experts, k, n_tokens);
            }
        } else {
            moe_q8_0_gateup_routed_mid_q8_1_batch_kernel<<<gateup_blocks, threads, 0>>>(
                d_mid, (const BnBlockQ8_0 *)gate->data,
                (const BnBlockQ8_0 *)up->data, xq, d_indices,
                hidden_dim, dim, n_experts, k, n_tokens);
        }
    } else if (routed_q8) {
        if (profile_prefill_moe)
            profile_path_counts[BN_CUDA_MOE_PROFILE_PATH_Q8_NATIVE]++;
        moe_q8_0_gateup_routed_mid_batch_kernel<<<gateup_blocks, threads, 0>>>(
            d_mid, (const BnBlockQ8_0 *)gate->data,
            (const BnBlockQ8_0 *)up->data, d_full_x, d_indices,
            d_weights, hidden_dim, dim, n_experts, k, n_tokens);
    } else if (use_q4k_q8k_gateup) {
        if (profile_prefill_moe)
            profile_path_counts[BN_CUDA_MOE_PROFILE_PATH_QK_NATIVE]++;
        if (cuda_ensure_q8_k(ctx, dim, n_tokens) != 0)
            return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(dim / BN_QK_K, n_tokens, 1),
                                    BN_QK_K, 0>>>(xq, d_full_x, dim,
                                                  n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) return -1;
        BN_CUDA_MOE_PREFILL_PROFILE_STEP(2);
            int use_gateup_8row =
                dim <= 2048 &&
                getenv("BN_CUDA_DISABLE_MOE_GATEUP_8ROW") == NULL;
            int use_gateup_split =
                dim <= 2048 && n_experts > 2 &&
                getenv("BN_CUDA_ENABLE_MOE_GATEUP_SPLIT") != NULL &&
                getenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT") == NULL;
            if (use_gateup_split) {
                int gateup2_blocks =
                    (gateup_tasks + warps * 2 - 1) / (warps * 2);
                if (use_sorted_slots) {
                    moe_q4k_gateup_routed_mid_q8k_2row_split_sorted_batch_kernel<<<gateup2_blocks, threads, 0>>>(
                        d_mid, (const BnBlockQ4K *)gate->data,
                        (const BnBlockQ4K *)up->data, xq, d_indices,
                        d_slot_order, hidden_dim, dim, n_experts, k,
                        n_tokens);
                } else {
                    moe_q4k_gateup_routed_mid_q8k_2row_split_batch_kernel<<<gateup2_blocks, threads, 0>>>(
                        d_mid, (const BnBlockQ4K *)gate->data,
                        (const BnBlockQ4K *)up->data, xq, d_indices,
                        hidden_dim, dim, n_experts, k, n_tokens);
                }
            } else if (use_gateup_8row) {
                int gateup8_blocks =
                    (gateup_tasks + warps * 8 - 1) / (warps * 8);
                if (use_sorted_slots) {
                    moe_q4k_gateup_routed_mid_q8k_8row_sorted_batch_kernel<<<gateup8_blocks, threads, 0>>>(
                        d_mid, (const BnBlockQ4K *)gate->data,
                        (const BnBlockQ4K *)up->data, xq, d_indices,
                        d_slot_order, hidden_dim, dim, n_experts, k,
                        n_tokens);
                } else {
                    moe_q4k_gateup_routed_mid_q8k_8row_batch_kernel<<<gateup8_blocks, threads, 0>>>(
                        d_mid, (const BnBlockQ4K *)gate->data,
                        (const BnBlockQ4K *)up->data, xq, d_indices,
                        hidden_dim, dim, n_experts, k, n_tokens);
                }
            } else {
                int gateup4_tasks = (gateup_tasks + 3) / 4;
                int gateup4_blocks = (gateup4_tasks + warps - 1) / warps;
                if (use_sorted_slots) {
                    moe_q4k_gateup_routed_mid_q8k_4row_sorted_batch_kernel<<<gateup4_blocks, threads, 0>>>(
                        d_mid, (const BnBlockQ4K *)gate->data,
                        (const BnBlockQ4K *)up->data, xq, d_indices,
                        d_slot_order, hidden_dim, dim, n_experts, k, n_tokens);
                } else {
                    moe_q4k_gateup_routed_mid_q8k_4row_batch_kernel<<<gateup4_blocks, threads, 0>>>(
                        d_mid, (const BnBlockQ4K *)gate->data,
                        (const BnBlockQ4K *)up->data, xq, d_indices,
                        hidden_dim, dim, n_experts, k, n_tokens);
                }
            }
    } else {
        if (profile_prefill_moe)
            profile_path_counts[BN_CUDA_MOE_PROFILE_PATH_QK_NATIVE]++;
        int x_blocks = (dim + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        quantize_q8_1_batch_kernel<<<dim3(x_blocks, n_tokens, 1), 32, 0>>>(
            xq, d_full_x, dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) return -1;
        BN_CUDA_MOE_PREFILL_PROFILE_STEP(2);
        moe_q4k_gateup_routed_mid_batch_kernel<<<gateup_blocks, threads, 0>>>(
            d_mid, (const BnBlockQ4K *)gate->data,
            (const BnBlockQ4K *)up->data, xq, d_indices, d_weights,
            hidden_dim, dim, n_experts, k, n_tokens);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] routed moe combined gate/up failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_MOE_PREFILL_PROFILE_STEP(3);

moe_route_routed_down:
    if (use_q8_q8_1_batch) {
        int n_mid = n_tokens * k;
        int mid_blocks = hidden_dim / 32;
        if (cuda_ensure_q8_1(ctx, mid_blocks * 32 * n_mid) != 0)
            return -1;
        BnCudaBlockQ8_1 *mid_q = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        quantize_q8_1_batch_kernel<<<dim3(mid_blocks, n_mid, 1), 32, 0>>>(
            mid_q, d_mid, hidden_dim, n_mid);
        err = cudaGetLastError();
        if (err != cudaSuccess) return -1;
        BN_CUDA_MOE_PREFILL_PROFILE_STEP(4);
        if (hidden_dim <= 1024 &&
            getenv("BN_CUDA_ENABLE_Q8_MOE_DOWN_4ROW") != NULL) {
            int down4_blocks = (down_tasks + warps * 4 - 1) / (warps * 4);
            moe_q8_0_down_routed_q8_1_accum_4row_batch_kernel<<<down4_blocks, threads, 0>>>(
                d_full_out, (const BnBlockQ8_0 *)down->data, mid_q,
                d_indices, d_weights, dim, hidden_dim, n_experts, k,
                n_tokens);
        } else if (hidden_dim <= 1024 &&
            getenv("BN_CUDA_DISABLE_Q8_MOE_DOWN_2ROW") == NULL) {
            int down2_blocks = (down_tasks + warps * 2 - 1) / (warps * 2);
            moe_q8_0_down_routed_q8_1_accum_2row_batch_kernel<<<down2_blocks, threads, 0>>>(
                d_full_out, (const BnBlockQ8_0 *)down->data, mid_q,
                d_indices, d_weights, dim, hidden_dim, n_experts, k,
                n_tokens);
        } else {
            moe_q8_0_down_routed_q8_1_accum_batch_kernel<<<down_blocks, threads, 0>>>(
                d_full_out, (const BnBlockQ8_0 *)down->data, mid_q,
                d_indices, d_weights, dim, hidden_dim, n_experts, k,
                n_tokens);
        }
    } else if (routed_q8) {
        moe_q8_0_down_routed_accum_batch_kernel<<<down_blocks, threads, 0>>>(
            d_full_out, (const BnBlockQ8_0 *)down->data, d_mid, d_indices,
            d_weights, dim, hidden_dim, n_experts, k, n_tokens);
    } else if (down_type == BN_GGUF_TENSOR_Q6_K && down->f32_data &&
               !prefer_q6k_quant_down &&
               !(routed_q4 && n_experts == 2 && k == 2 &&
                 hidden_dim >= 4096 && dim <= 2048) &&
               getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL) {
        moe_q6k_down_routed_f32_cache_batch_kernel<<<down_blocks, threads, 0>>>(
            d_full_out, (const float *)down->f32_data, d_mid, d_indices,
            d_weights, dim, hidden_dim, n_experts, k, n_tokens);
    } else {
        int n_mid = n_tokens * k;
        if (cuda_ensure_q8_k(ctx, hidden_dim, n_mid) != 0)
            return -1;
        BnBlockQ8K *mid_q = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(hidden_dim / BN_QK_K, n_mid, 1),
                                    BN_QK_K>>>(mid_q, d_mid, hidden_dim,
                                               n_mid);
        err = cudaGetLastError();
        if (err != cudaSuccess) return -1;
        BN_CUDA_MOE_PREFILL_PROFILE_STEP(4);
        int use_down_4row =
            hidden_dim <= 1024 &&
            getenv("BN_CUDA_DISABLE_MOE_DOWN_4ROW") == NULL;
        if (use_down_4row) {
            int down4_tasks = n_tokens * dim;
            int down4_blocks = (down4_tasks + warps * 4 - 1) / (warps * 4);
            int use_down_8row =
                hidden_dim <= 1024 &&
                getenv("BN_CUDA_DISABLE_MOE_DOWN_8ROW") == NULL;
            if (use_down_8row) {
                int use_halfwarp =
                    down_type == BN_GGUF_TENSOR_Q6_K &&
                    (n_experts > 2 || k > 2) &&
                    (prefer_q6k_quant_down ||
                     getenv("BN_CUDA_ENABLE_MOE_DOWN_HALFWARP") != NULL) &&
                    getenv("BN_CUDA_DISABLE_MOE_DOWN_HALFWARP") == NULL;
                int use_split4 =
                    !use_halfwarp &&
                    down_type == BN_GGUF_TENSOR_Q6_K &&
                    (n_experts > 2 || k > 2) &&
                    getenv("BN_CUDA_ENABLE_MOE_DOWN_SPLIT4") != NULL &&
                    getenv("BN_CUDA_DISABLE_MOE_DOWN_SPLIT4") == NULL;
                int use_scatter =
                    !use_halfwarp && !use_split4 &&
                    down_type == BN_GGUF_TENSOR_Q6_K &&
                    getenv("BN_CUDA_DISABLE_MOE_DOWN_SCATTER") == NULL;
                int use_scatter_16row =
                    use_scatter && hidden_dim <= 768 &&
                    getenv("BN_CUDA_ENABLE_MOE_DOWN_SCATTER_16ROW") != NULL;
                int down8_tasks = use_scatter ? n_tokens * k * dim : down4_tasks;
                int down8_blocks = (down8_tasks + warps * 8 - 1) / (warps * 8);
                if (use_halfwarp) {
                    int down2_blocks = (down4_tasks + warps * 2 - 1) / (warps * 2);
                    moe_q6k_down_routed_q8k_accum_halfwarp_batch_kernel<<<down2_blocks, threads, 0>>>(
                        d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                        d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                } else if (use_split4) {
                    int down_split4_blocks =
                        (down4_tasks + warps * 4 - 1) / (warps * 4);
                    moe_q6k_down_routed_q8k_accum_4row_split_batch_kernel<<<down_split4_blocks, threads, 0>>>(
                        d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                        d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                } else if (use_scatter) {
                    if (use_scatter_16row) {
                        int down16_blocks =
                            (down8_tasks + warps * 16 - 1) / (warps * 16);
                        moe_q6k_down_routed_q8k_scatter_16row_batch_kernel<<<down16_blocks, threads, 0>>>(
                            d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                            d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                    } else {
                        if (use_sorted_slots) {
                            moe_q6k_down_routed_q8k_scatter_8row_sorted_batch_kernel<<<down8_blocks, threads, 0>>>(
                                d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                                d_indices, d_weights, d_slot_order, dim,
                                hidden_dim, n_experts, k, n_tokens);
                        } else {
                            moe_q6k_down_routed_q8k_scatter_8row_batch_kernel<<<down8_blocks, threads, 0>>>(
                                d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                                d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                        }
                    }
                } else if (down_type == BN_GGUF_TENSOR_Q6_K) {
                    moe_q6k_down_routed_q8k_accum_8row_batch_kernel<<<down8_blocks, threads, 0>>>(
                        d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                        d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                } else {
                    moe_q4k_down_routed_q8k_accum_8row_batch_kernel<<<down8_blocks, threads, 0>>>(
                        d_full_out, (const BnBlockQ4K *)down->data, mid_q,
                        d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
                }
            } else if (down_type == BN_GGUF_TENSOR_Q6_K) {
                moe_q6k_down_routed_q8k_accum_4row_batch_kernel<<<down4_blocks, threads, 0>>>(
                    d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                    d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
            } else {
                moe_q4k_down_routed_q8k_accum_4row_batch_kernel<<<down4_blocks, threads, 0>>>(
                    d_full_out, (const BnBlockQ4K *)down->data, mid_q,
                    d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
            }
        } else if (down_type == BN_GGUF_TENSOR_Q6_K) {
            moe_q6k_down_routed_q8k_accum_batch_kernel<<<down_blocks, threads, 0>>>(
                d_full_out, (const BnBlockQ6K *)down->data, mid_q,
                d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
        } else {
            moe_q4k_down_routed_q8k_accum_batch_kernel<<<down_blocks, threads, 0>>>(
                d_full_out, (const BnBlockQ4K *)down->data, mid_q,
                d_indices, d_weights, dim, hidden_dim, n_experts, k, n_tokens);
        }
    }
    err = cudaGetLastError();
    BN_CUDA_MOE_PREFILL_PROFILE_STEP(5);
moe_route_routed_readback:
    if (err == cudaSuccess && has_shared) {
        if (cuda_dense_ffn_batch_device(
                ctx, ctx->d_out, shared_gate, shared_up, shared_down,
                d_full_x, n_tokens, dim, shared_hidden_dim,
                shared_gate_type, shared_up_type, shared_down_type,
                act_type) != 0)
            return -1;
        int total = n_tokens * dim;
        if (shared_gate_weight) {
            shared_expert_add_sigmoid_batch_kernel<<<n_tokens, threads>>>(
                d_full_out, ctx->d_out,
                (const float *)shared_gate_weight->data, d_full_x,
                n_tokens, dim);
        } else {
            residual_add_kernel<<<(total + threads - 1) / threads, threads>>>(
                d_full_out, ctx->d_out, total);
        }
        err = cudaGetLastError();
    }
    if (err == cudaSuccess && add_norm_resid && !init_out_with_residual) {
        int total = (int)full_values;
        residual_add_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_full_out, d_residual, total);
        err = cudaGetLastError();
    }
    if (profile_prefill_moe) {
        if (out)
            profile_io_counts[BN_CUDA_MOE_PROFILE_OUT_HOST]++;
        else
            profile_io_counts[BN_CUDA_MOE_PROFILE_OUT_DEVICE]++;
    }
    if (err == cudaSuccess && out) {
        if (cuda_ensure_host_out(ctx, full_bytes) != 0)
            return -1;
        err = cudaMemcpy(ctx->h_out, d_full_out, full_bytes,
                         cudaMemcpyDeviceToHost);
    } else if (err == cudaSuccess && !direct_device_resid_out) {
        err = cudaMemcpy(ctx->d_out, d_full_out, full_bytes,
                         cudaMemcpyDeviceToDevice);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] routed moe combined down failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    if (out)
        memcpy(out, ctx->h_out, full_bytes);
    BN_CUDA_MOE_PREFILL_PROFILE_STEP(6);
    if (profile_prefill_moe) {
        profile_calls++;
        int every = cuda_env_int("BN_CUDA_PROFILE_MOE_PREFILL_EVERY", 48);
        if (every <= 0) every = 48;
        if ((profile_calls % (unsigned long long)every) == 0) {
            double total = 0.0;
            for (int i = 0; i < 7; i++) total += profile_totals[i];
            fprintf(stderr,
                    "[bn:gpu:cuda:moe-prefill] calls=%llu total=%.3fms upload=%.3f route=%.3f x_quant=%.3f gateup=%.3f mid_quant=%.3f down=%.3f readback=%.3f\n",
                    profile_calls, total, profile_totals[0],
                    profile_totals[1], profile_totals[2],
                    profile_totals[3], profile_totals[4],
                    profile_totals[5], profile_totals[6]);
            fprintf(stderr,
                    "[bn:gpu:cuda:moe-prefill-io] x_host=%llu x_device=%llu out_host=%llu out_device=%llu cublas_grouped=%llu cublas_gateup=%llu q8_native=%llu qk_native=%llu\n",
                    profile_io_counts[BN_CUDA_MOE_PROFILE_X_HOST],
                    profile_io_counts[BN_CUDA_MOE_PROFILE_X_DEVICE],
                    profile_io_counts[BN_CUDA_MOE_PROFILE_OUT_HOST],
                    profile_io_counts[BN_CUDA_MOE_PROFILE_OUT_DEVICE],
                    profile_path_counts[
                        BN_CUDA_MOE_PROFILE_PATH_CUBLAS_GROUPED],
                    profile_path_counts[
                        BN_CUDA_MOE_PROFILE_PATH_CUBLAS_GATEUP],
                    profile_path_counts[BN_CUDA_MOE_PROFILE_PATH_Q8_NATIVE],
                    profile_path_counts[BN_CUDA_MOE_PROFILE_PATH_QK_NATIVE]);
            for (int i = 0; i < 7; i++) profile_totals[i] = 0.0;
            for (int i = 0; i < 4; i++) {
                profile_io_counts[i] = 0;
                profile_path_counts[i] = 0;
            }
        }
    }
#undef BN_CUDA_MOE_PREFILL_PROFILE_STEP
    return 0;
}

static int cuda_moe_route_routed_ffn_batch(
    void *vctx, float *out, void *router_buf, void *gate_all_buf,
    void *up_all_buf, void *down_all_buf, const float *X,
    int n_tokens, int dim, int hidden_dim, int n_experts, int k,
    int gate_type, int up_type, int down_type, int act_type,
    int norm_topk_prob, float expert_weights_scale) {
    return cuda_moe_route_routed_ffn_batch_impl(
        vctx, out, router_buf, gate_all_buf, up_all_buf, down_all_buf,
        NULL, NULL, NULL, NULL, NULL, X, n_tokens, dim, hidden_dim,
        n_experts, k, gate_type, up_type, down_type, act_type, 0, 0, 0, 0,
        0.0f, 0, norm_topk_prob, expert_weights_scale);
}

static int cuda_moe_route_routed_ffn_batch_norm_resid(
    void *vctx, float *out, void *router_buf, void *gate_all_buf,
    void *up_all_buf, void *down_all_buf, void *shared_gate_buf,
    void *shared_up_buf, void *shared_down_buf,
    void *shared_gate_weight_buf, void *norm_buf, const float *X,
    int n_tokens, int dim, int hidden_dim, int n_experts, int k,
    int gate_type, int up_type, int down_type, int act_type,
    int shared_hidden_dim, int shared_gate_type, int shared_up_type,
    int shared_down_type, float norm_eps, int norm_topk_prob,
    float expert_weights_scale) {
    return cuda_moe_route_routed_ffn_batch_impl(
        vctx, out, router_buf, gate_all_buf, up_all_buf, down_all_buf,
        shared_gate_buf, shared_up_buf, shared_down_buf,
        shared_gate_weight_buf, norm_buf, X, n_tokens, dim, hidden_dim,
        n_experts, k, gate_type, up_type, down_type, act_type,
        shared_hidden_dim, shared_gate_type, shared_up_type,
        shared_down_type, norm_eps, 1, norm_topk_prob,
        expert_weights_scale);
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
    int min_tokens = cuda_env_int("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", 16);
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
    int min_tokens = cuda_env_int("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", 16);
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
        if (n_tokens >= 4) {
            dim3 grid((wo_rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            if (getenv("BN_CUDA_ENABLE_Q4K_SHAREDX_BATCH")) {
                size_t shared =
                    (size_t)x_blocks * 4u * sizeof(BnCudaBlockQ8_1);
                q4k_dot_matmul4_token_sharedx_kernel<<<grid, threads,
                                                        shared>>>(
                    ctx->d_out, (const BnBlockQ4K *)wo->data, xq, wo_rows,
                    wo_cols, n_tokens, 0);
            } else {
                q4k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                    ctx->d_out, (const BnBlockQ4K *)wo->data, xq, wo_rows,
                    wo_cols, n_tokens, 0);
            }
        } else {
            dim3 grid((wo_rows + warps - 1) / warps, n_tokens, 1);
            q4k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ4K *)wo->data, xq, wo_rows,
                wo_cols, n_tokens, 0);
        }
    } else if (wo_type == BN_GGUF_TENSOR_Q5_K &&
               (wo_cols % BN_QK_K) == 0) {
        int x_blocks = (wo_cols + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        quantize_q8_1_batch_kernel<<<dim3(x_blocks, n_tokens, 1), 32, 0>>>(
            xq, ctx->d_x, wo_cols, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((wo_rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q5k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)wo->data, xq, wo_rows,
                wo_cols, n_tokens, 0);
        } else {
            dim3 grid((wo_rows + warps - 1) / warps, n_tokens, 1);
            q5k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ5K *)wo->data, xq, wo_rows,
                wo_cols, n_tokens, 0);
        }
    } else if (wo_type == BN_GGUF_TENSOR_Q6_K && (wo_cols % BN_QK_K) == 0 &&
               !getenv("BN_CUDA_DISABLE_Q6K_DOT")) {
        int x_blocks = wo_cols / BN_QK_K;
        if (cuda_ensure_q8_k(ctx, wo_cols, n_tokens) != 0)
            return -1;
        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
        quantize_q8k_batch_kernel<<<dim3(x_blocks, n_tokens, 1), BN_QK_K>>>(
            xq, ctx->d_x, wo_cols, n_tokens);
        if (n_tokens >= 4) {
            dim3 grid((wo_rows + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q6k_dot_matmul4_token_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ6K *)wo->data, xq, wo_rows,
                wo_cols, n_tokens, 0);
        } else {
            dim3 grid((wo_rows + warps - 1) / warps, n_tokens, 1);
            q6k_dot_matmul_kernel<<<grid, threads, 0>>>(
                ctx->d_out, (const BnBlockQ6K *)wo->data, xq, wo_rows,
                wo_cols, n_tokens, 0);
        }
    } else if (wo_type == BN_GGUF_TENSOR_Q5_0 && (wo_cols & 31) == 0) {
        dim3 grid(((wo_rows + 3) / 4 + warps - 1) / warps, n_tokens, 1);
        q5_0_matmul4_warp_kernel<<<grid, threads, 0>>>(
            ctx->d_out, (const BnBlockQ5_0 *)wo->data, ctx->d_x, wo_rows,
            wo_cols, n_tokens, 0);
    } else if (wo_type == BN_GGUF_TENSOR_Q8_0 && (wo_cols & 31) == 0) {
        if (cuda_launch_q8_0_matmul(ctx, ctx->d_out,
                (const BnBlockQ8_0 *)wo->data, ctx->d_x, wo_rows, wo_cols,
                n_tokens, 0) != 0)
            return -1;
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
        void *q_bias_buf, void *k_bias_buf, void *v_bias_buf,
        const float *X, float *K_out, float *V_out, int n_tokens, int dim,
        int n_heads, int n_kv_heads, int head_size, int kv_mul, int kv_dim,
        int qk_rows, int qk_type, int wv_rows, int wv_type, int wo_rows,
        int wo_cols, int wo_type, int qk_norm_per_head, float norm_eps,
        int pos0, int rope_dims,
        float attention_scale,
        int add_residual,
        uint32_t kv_cache_off,
        int kv_cache_stride,
        int min_tokens_override) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *qk = (BnCudaBuffer *)qk_buf;
    BnCudaBuffer *wv = (BnCudaBuffer *)wv_buf;
    BnCudaBuffer *wo = (BnCudaBuffer *)wo_buf;
    BnCudaBuffer *attn_norm = (BnCudaBuffer *)attn_norm_buf;
    BnCudaBuffer *q_norm = (BnCudaBuffer *)q_norm_buf;
    BnCudaBuffer *k_norm = (BnCudaBuffer *)k_norm_buf;
    BnCudaBuffer *q_bias = (BnCudaBuffer *)q_bias_buf;
    BnCudaBuffer *k_bias = (BnCudaBuffer *)k_bias_buf;
    BnCudaBuffer *v_bias = (BnCudaBuffer *)v_bias_buf;
    if (getenv("BN_CUDA_DISABLE_PREFILL_QKV_ATTN_WO"))
        return -1;
    int q_dim = n_heads * head_size;
    int q_gated = qk && qk->data && qk_rows == 2 * q_dim + kv_dim;
    if (!ctx || !qk || !qk->data || !wv || !wv->data ||
        !wo || !wo->data || (!X && !ctx->d_out) ||
        ((!K_out || !V_out) &&
         (!ctx->act_bufs[BN_GPU_VALUE_KEY_CACHE] ||
          !ctx->act_bufs[BN_GPU_VALUE_VALUE_CACHE])) ||
        n_tokens <= 1 ||
        dim <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_size <= 0 ||
        kv_mul <= 0 || kv_dim <= 0 || n_heads / kv_mul != n_kv_heads ||
        (!q_gated && qk_rows != q_dim + kv_dim) || wv_rows != kv_dim ||
        wo_cols != n_heads * head_size || wo->rows != wo_rows ||
        wo->cols != wo_cols || !cuda_type_supported(qk_type) ||
        !cuda_type_supported(wv_type) || !cuda_type_supported(wo_type) ||
        pos0 != 0 || rope_dims <= 0 ||
        !ctx->act_bufs[BN_GPU_VALUE_ROPE_FREQ])
        return -1;
    if (attn_norm && (!attn_norm->data ||
                      attn_norm->rows * attn_norm->cols < dim))
        return -1;
    if ((q_bias && (!q_bias->data || q_bias->rows * q_bias->cols < q_dim)) ||
        (k_bias && (!k_bias->data || k_bias->rows * k_bias->cols < kv_dim)) ||
        (v_bias && (!v_bias->data || v_bias->rows * v_bias->cols < kv_dim)))
        return -1;
    int min_tokens = getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS")
        ? cuda_env_int("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", 64)
        : 16;
    if (min_tokens_override > 0)
        min_tokens = min_tokens_override;
    if (n_tokens < min_tokens || n_tokens > 2048)
        return -1;

    size_t q_values = (size_t)n_tokens * (size_t)q_dim;
    size_t q_gate_values = q_gated ? q_values : 0u;
    size_t kv_values = (size_t)n_tokens * (size_t)kv_dim;
    size_t qk_values = (size_t)n_tokens * (size_t)qk_rows;
    int use_gemm_attention =
        !getenv("BN_CUDA_DISABLE_PREFILL_GEMM_ATTN") &&
        (getenv("BN_CUDA_ENABLE_PREFILL_GEMM_ATTN") ||
         n_tokens >= cuda_env_int("BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS", 256)) &&
        n_tokens <= 512;
    size_t score_values =
        use_gemm_attention
            ? (size_t)n_heads * (size_t)n_tokens * (size_t)n_tokens
            : 0u;
    if (add_residual && !attn_norm)
        return -1;
    size_t residual_values = add_residual
        ? (size_t)n_tokens * (size_t)dim : 0u;
    size_t norm_values = attn_norm ? (size_t)n_tokens * (size_t)dim : 0u;
    size_t total_values = residual_values + norm_values + q_values +
                          q_gate_values + 2u * kv_values + qk_values +
                          score_values;
    size_t out_values = (size_t)n_tokens * (size_t)wo_rows;
    size_t x_values = (size_t)n_tokens * (size_t)dim;
    if (q_values > x_values)
        x_values = q_values;
    size_t x_bytes = (size_t)n_tokens * (size_t)dim * sizeof(float);
    size_t scratch_out_bytes = out_values * sizeof(float);
    const float *d_resident_input = NULL;
    if (!X &&
        cuda_prepare_resident_input(ctx, x_bytes, scratch_out_bytes,
                                    &d_resident_input) != 0)
        return -1;
    if (cuda_ensure_scratch(ctx, x_values * sizeof(float),
                            scratch_out_bytes) != 0)
        return -1;
    if (cuda_ensure_prefill(ctx, total_values) != 0)
        return -1;

    cudaError_t err = X
        ? cudaMemcpy(ctx->d_x, X, x_bytes, cudaMemcpyHostToDevice)
        : cudaMemcpy(ctx->d_x, d_resident_input, x_bytes,
                     cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill qkv input upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    float *d_residual = ctx->d_prefill;
    float *d_norm = d_residual + residual_values;
    float *d_q = d_norm + norm_values;
    float *d_q_gate = d_q + q_values;
    float *d_k = d_q_gate + q_gate_values;
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
    int total_qk = n_tokens * (q_dim + (q_gated ? q_dim : 0) + kv_dim);
    int blocks = (total_qk + threads - 1) / threads;
    if (q_gated) {
        split_qgk_prefill_kernel<<<blocks, threads>>>(
            d_qk, d_q, d_q_gate, d_k, n_tokens, q_dim, kv_dim, qk_rows,
            head_size);
    } else {
        split_qk_prefill_kernel<<<blocks, threads>>>(
            d_qk, d_q, d_k, n_tokens, q_dim, kv_dim, qk_rows);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill qk split failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    if (q_bias) {
        int total = n_tokens * q_dim;
        bias_add_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_q, (const float *)q_bias->data, q_dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill q bias failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (k_bias) {
        int total = n_tokens * kv_dim;
        bias_add_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_k, (const float *)k_bias->data, kv_dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill k bias failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (v_bias) {
        int total = n_tokens * kv_dim;
        bias_add_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_v, (const float *)v_bias->data, kv_dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill v bias failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
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

    if (use_gemm_attention) {
        if (cuda_prefill_attention_gemm(ctx, ctx->d_x, d_q, d_k, d_v,
                                        d_scores, n_tokens, n_heads,
                                        n_kv_heads, head_size, kv_mul,
                                        kv_dim, attention_scale) != 0)
            return -1;
    } else {
        int threads = 256;
        prefill_attention_kernel<<<dim3(n_heads, n_tokens, 1), threads,
                                   ((size_t)n_tokens + (size_t)threads) *
                                       sizeof(float)>>>(
            ctx->d_x, d_q, d_k, d_v, n_tokens, n_heads, head_size, kv_mul,
            kv_dim, attention_scale);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "[bn:gpu:cuda] prefill qkv attention failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (q_gated) {
        int n_q = n_tokens * q_dim;
        int gate_blocks = (n_q + threads - 1) / threads;
        apply_q_gate_prefill_kernel<<<gate_blocks, threads>>>(
            ctx->d_x, d_q_gate, n_q);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill qkv q gate failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
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
    if (K_out && V_out) {
        err = cudaMemcpy(K_out, d_k, kv_bytes, cudaMemcpyDeviceToHost);
        if (err == cudaSuccess)
            err = cudaMemcpy(V_out, d_v, kv_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill kv readback failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    } else {
        int kv_total = n_tokens * kv_dim;
        int kv_blocks = (kv_total + 255) / 256;
        prefill_write_kv_cache_kernel<<<kv_blocks, 256>>>(
            cuda_act(ctx, BN_GPU_VALUE_KEY_CACHE),
            cuda_act(ctx, BN_GPU_VALUE_VALUE_CACHE), d_k, d_v, n_tokens,
            kv_dim, kv_cache_stride, kv_cache_off, ctx->kv_f16);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill kv write failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }

    if (out) {
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
    }
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
        NULL, NULL, NULL, X, K_out, V_out, n_tokens, dim, n_heads, n_kv_heads, head_size,
        kv_mul, kv_dim, qk_rows, qk_type, wv_rows, wv_type, wo_rows,
        wo_cols, wo_type, qk_norm_per_head, norm_eps, pos0, rope_dims,
        attention_scale, 0, 0u, kv_dim, 0);
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
        k_norm_buf, NULL, NULL, NULL, X, K_out, V_out, n_tokens, dim, n_heads, n_kv_heads,
        head_size, kv_mul, kv_dim, qk_rows, qk_type, wv_rows, wv_type,
        wo_rows, wo_cols, wo_type, qk_norm_per_head, norm_eps, pos0,
        rope_dims, attention_scale, 0, 0u, kv_dim, 0);
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
        k_norm_buf, NULL, NULL, NULL, X, K_out, V_out, n_tokens, dim, n_heads, n_kv_heads,
        head_size, kv_mul, kv_dim, qk_rows, qk_type, wv_rows, wv_type,
        wo_rows, wo_cols, wo_type, qk_norm_per_head, norm_eps, pos0,
        rope_dims, attention_scale, 1, 0u, kv_dim, 0);
}

static int cuda_prefill_moe_layer(
        void *vctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
        void *router_buf, void *gate_all_buf, void *up_all_buf,
        void *down_all_buf, void *shared_gate_buf, void *shared_up_buf,
        void *shared_down_buf, void *shared_gate_weight_buf,
        void *attn_norm_buf, void *ffn_norm_buf,
        void *q_norm_buf, void *k_norm_buf,
        void *q_bias_buf, void *k_bias_buf, void *v_bias_buf,
        const float *X, float *K_out, float *V_out,
        int n_tokens, int dim, int moe_hidden_dim,
        int n_experts, int experts_active, int n_heads, int n_kv_heads,
        int head_size, int kv_mul, int kv_dim, int qk_rows, int qk_type,
        int wv_rows, int wv_type, int wo_rows, int wo_cols, int wo_type,
        int gate_type, int up_type, int down_type, int act_type,
        int shared_hidden_dim, int shared_gate_type, int shared_up_type,
        int shared_down_type,
        int qk_norm_per_head, float norm_eps, int pos0, int rope_dims,
        uint32_t kv_cache_off, int kv_cache_stride,
        float attention_scale, int norm_topk_prob,
        float expert_weights_scale) {
    if (getenv("BN_CUDA_DISABLE_PREFILL_MOE_LAYER"))
        return -1;
    if (act_type != 0)
        return -1;
    if (cuda_prefill_qkv_attention_wo_impl(
            vctx, NULL, qk_buf, wv_buf, wo_buf, attn_norm_buf, q_norm_buf,
            k_norm_buf, q_bias_buf, k_bias_buf, v_bias_buf, X, K_out, V_out, n_tokens, dim, n_heads,
            n_kv_heads, head_size, kv_mul, kv_dim, qk_rows, qk_type,
            wv_rows, wv_type, wo_rows, wo_cols, wo_type, qk_norm_per_head,
            norm_eps, pos0, rope_dims, attention_scale, 1, kv_cache_off,
            kv_cache_stride, cuda_env_int("BN_CUDA_MOE_PREFILL_MIN_TOKENS", 1)) != 0)
        return -1;
    return cuda_moe_route_routed_ffn_batch_impl(
        vctx, out, router_buf, gate_all_buf, up_all_buf, down_all_buf,
        shared_gate_buf, shared_up_buf, shared_down_buf,
        shared_gate_weight_buf, ffn_norm_buf, NULL, n_tokens, dim,
        moe_hidden_dim, n_experts, experts_active, gate_type, up_type,
        down_type, act_type, shared_hidden_dim, shared_gate_type,
        shared_up_type, shared_down_type, norm_eps, 1, norm_topk_prob,
        expert_weights_scale);
}

static int cuda_prefill_dense_layer(
        void *vctx, float *out, void *qk_buf, void *wv_buf, void *wo_buf,
        void *gate_buf, void *up_buf, void *down_buf, void *attn_norm_buf,
        void *ffn_norm_buf, void *q_norm_buf, void *k_norm_buf,
        void *q_bias_buf, void *k_bias_buf, void *v_bias_buf,
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
    BnCudaBuffer *q_bias = (BnCudaBuffer *)q_bias_buf;
    BnCudaBuffer *k_bias = (BnCudaBuffer *)k_bias_buf;
    BnCudaBuffer *v_bias = (BnCudaBuffer *)v_bias_buf;
    int stacked_gateup = up == NULL;
    int q_dim = n_heads * head_size;
    int packed_qkv = qk && qk->data && !wv && qk_rows == q_dim + 2 * kv_dim;
    int q_gated = !packed_qkv && qk_rows == 2 * q_dim + kv_dim;
    int debug_dense_prefill =
        getenv("BN_CUDA_DEBUG_PREFILL_DENSE_LAYER") != NULL;
    if (getenv("BN_CUDA_DISABLE_PREFILL_DENSE_LAYER"))
        return -1;
    if (!ctx || !qk || !qk->data || (!packed_qkv && (!wv || !wv->data)) ||
        !wo || !wo->data || !gate || !gate->data || !down || !down->data ||
        !attn_norm || !attn_norm->data || !ffn_norm || !ffn_norm->data ||
        n_tokens <= 1 || dim <= 0 ||
        hidden_dim <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_size <= 0 || kv_mul <= 0 || kv_dim <= 0 ||
        kv_cache_stride < kv_dim ||
        n_heads / kv_mul != n_kv_heads || act_type != 0 ||
        (!packed_qkv && !q_gated && qk_rows != q_dim + kv_dim) ||
        (!packed_qkv && wv_rows != kv_dim) ||
        wo_cols != n_heads * head_size || wo->rows != wo_rows ||
        wo->cols != wo_cols || down->rows != dim ||
        down->cols != hidden_dim || pos0 != 0 || rope_dims <= 0 ||
        !ctx->act_bufs[BN_GPU_VALUE_ROPE_FREQ]) {
        if (debug_dense_prefill) {
            fprintf(stderr,
                    "[bn:gpu:cuda:dense_prefill] reject args qk_rows=%d q_dim=%d kv_dim=%d packed=%d q_gated=%d wv_rows=%d wo_cols=%d head_size=%d n_heads=%d pos0=%d rope_dims=%d has_rope=%d\n",
                    qk_rows, q_dim, kv_dim, packed_qkv, q_gated, wv_rows,
                    wo_cols, head_size, n_heads, pos0, rope_dims,
                    ctx && ctx->act_bufs[BN_GPU_VALUE_ROPE_FREQ] ? 1 : 0);
        }
        return -1;
    }
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
    if ((q_bias && (!q_bias->data || q_bias->rows * q_bias->cols < q_dim)) ||
        (k_bias && (!k_bias->data || k_bias->rows * k_bias->cols < kv_dim)) ||
        (v_bias && (!v_bias->data || v_bias->rows * v_bias->cols < kv_dim)))
        return -1;
    if (!cuda_type_supported(qk_type) ||
        (!packed_qkv && !cuda_type_supported(wv_type)) ||
        !cuda_type_supported(wo_type) || !cuda_type_supported(gate_type) ||
        (!stacked_gateup && !cuda_type_supported(up_type)) ||
        !cuda_type_supported(down_type))
        return -1;
    int min_tokens = getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS")
        ? cuda_env_int("BN_CUDA_PREFILL_ATTN_MIN_TOKENS", 64)
        : (dim <= 2560 ? 2 : 16);
    if (n_tokens < min_tokens || n_tokens > 512)
        return -1;
    int use_gemm_attention =
        !getenv("BN_CUDA_DISABLE_PREFILL_GEMM_ATTN") &&
        (getenv("BN_CUDA_ENABLE_PREFILL_GEMM_ATTN") ||
         n_tokens >= cuda_env_int("BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS", 256));

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

    size_t dim_values = (size_t)n_tokens * (size_t)dim;
    size_t q_values = (size_t)n_tokens * (size_t)q_dim;
    size_t kv_values = (size_t)n_tokens * (size_t)kv_dim;
    size_t qk_values = (size_t)n_tokens * (size_t)qk_rows;
    size_t score_values = use_gemm_attention
        ? (size_t)n_heads * (size_t)n_tokens * (size_t)n_tokens
        : 0u;
    size_t hidden_values = (size_t)n_tokens * (size_t)hidden_dim;
    size_t gateup_values = hidden_values * 2u;
    size_t q_gate_values = q_gated ? q_values : 0u;
    size_t total_values = dim_values + dim_values + q_values + q_gate_values +
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
    float *d_q_gate = d_q + q_values;
    float *d_k = d_q_gate + q_gate_values;
    float *d_v = d_k + kv_values;
    float *d_qk = d_v + kv_values;
    float *d_scores = d_qk + qk_values;
    float *d_ffn_residual = d_scores + score_values;
    float *d_ffn_norm = d_ffn_residual + dim_values;
    float *d_ffn_act = d_ffn_norm + dim_values;
    float *d_gateup = d_ffn_act + hidden_values;

    int threads = 256;
    int warps = threads / 32;
    rmsnorm_batch_copy_kernel<<<n_tokens, threads,
                                (size_t)warps * sizeof(float)>>>(
        d_attn_norm, d_orig, ctx->d_out, (const float *)attn_norm->data,
        dim, n_tokens, norm_eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill dense layer attn norm failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_INPUT_NORM);
    const void *attn_norm_f16 = NULL;
    if (qk->f16_data && (packed_qkv || wv->f16_data) && n_tokens > 1 &&
        cuda_convert_f32_to_f16(ctx, d_attn_norm, dim_values) == 0)
        attn_norm_f16 = ctx->d_x_f16;
    if (cuda_matmul_device_out_preconverted_f16(
            ctx, d_qk, qk, d_attn_norm, attn_norm_f16, qk_rows, dim,
            n_tokens, qk_type) != 0)
        return -1;
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_QK);
    if (!packed_qkv) {
        if (cuda_matmul_device_out_preconverted_f16(
                ctx, d_v, wv, d_attn_norm, attn_norm_f16, wv_rows, dim,
                n_tokens, wv_type) != 0)
            return -1;
    }
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_WV);

    int total_qkv = n_tokens *
                    (q_dim + (q_gated ? q_dim : 0) +
                     (packed_qkv ? 2 * kv_dim : kv_dim));
    int blocks = (total_qkv + threads - 1) / threads;
    if (packed_qkv) {
        split_qkv_prefill_kernel<<<blocks, threads>>>(
            d_qk, d_q, d_k, d_v, n_tokens, q_dim, kv_dim, qk_rows);
    } else if (q_gated) {
        split_qgk_prefill_kernel<<<blocks, threads>>>(
            d_qk, d_q, d_q_gate, d_k, n_tokens, q_dim, kv_dim, qk_rows,
            head_size);
    } else {
        split_qk_prefill_kernel<<<blocks, threads>>>(
            d_qk, d_q, d_k, n_tokens, q_dim, kv_dim, qk_rows);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill dense layer qk split failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    if (q_bias) {
        int total = n_tokens * q_dim;
        bias_add_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_q, (const float *)q_bias->data, q_dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill dense layer q bias failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (k_bias) {
        int total = n_tokens * kv_dim;
        bias_add_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_k, (const float *)k_bias->data, kv_dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill dense layer k bias failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (v_bias) {
        int total = n_tokens * kv_dim;
        bias_add_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_v, (const float *)v_bias->data, kv_dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill dense layer v bias failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
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

    if (use_gemm_attention) {
        if (cuda_prefill_attention_gemm(ctx, ctx->d_x, d_q, d_k, d_v,
                                        d_scores, n_tokens, n_heads,
                                        n_kv_heads, head_size, kv_mul,
                                        kv_dim, attention_scale) != 0)
            return -1;
    } else {
        size_t shared = (size_t)(n_tokens + threads) * sizeof(float);
        prefill_attention_kernel<<<dim3(n_heads, n_tokens, 1), threads,
                                   shared>>>(
            ctx->d_x, d_q, d_k, d_v, n_tokens, n_heads, head_size, kv_mul,
            kv_dim, attention_scale);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill dense layer attention failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (q_gated) {
        int n_q = n_tokens * q_dim;
        int gate_blocks = (n_q + threads - 1) / threads;
        apply_q_gate_prefill_kernel<<<gate_blocks, threads>>>(
            ctx->d_x, d_q_gate, n_q);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill dense layer q gate failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
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
    int ffn_act_ready = 0;
    if (stacked_gateup && gate_type == BN_GGUF_TENSOR_Q4_K &&
        !gate->f16_data && (dim % BN_QK_K) == 0 &&
        getenv("BN_CUDA_DISABLE_PREFILL_FUSED_Q4K_GATEUP_BATCH") == NULL) {
        int x_blocks = (dim + 31) / 32;
        if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
            return -1;
        BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
        quantize_q8_1_batch_kernel<<<dim3(x_blocks, n_tokens, 1), 32, 0>>>(
            xq, d_ffn_norm, dim, n_tokens);
        dim3 grid((hidden_dim + warps - 1) / warps,
                  (n_tokens + 3) / 4, 1);
        q4k_dot_fused_gateup_silu_batch4_token_kernel<<<grid, threads, 0>>>(
            d_ffn_act, (const BnBlockQ4K *)gate->data, xq, hidden_dim,
            hidden_dim, dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill dense layer fused gate/up failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        ffn_act_ready = 1;
    } else if (stacked_gateup) {
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

    int ffn_act_f16_ready = 0;
    if (!ffn_act_ready) {
        int act_total = n_tokens * hidden_dim;
        blocks = (act_total + threads - 1) / threads;
        ffn_act_f16_ready = down->f16_data &&
                            cuda_ensure_x_f16(ctx, hidden_values) == 0;
        if (ffn_act_f16_ready && stacked_gateup) {
            ffn_activation_batch_stacked_to_f16_kernel<<<blocks, threads>>>(
                (__half *)ctx->d_x_f16, d_gateup, hidden_dim, n_tokens,
                act_type);
        } else if (ffn_act_f16_ready) {
            ffn_activation_batch_to_f16_kernel<<<blocks, threads>>>(
                (__half *)ctx->d_x_f16, d_gateup, hidden_dim, n_tokens,
                act_type);
        } else if (stacked_gateup) {
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
    }
    BN_CUDA_DENSE_PROFILE_STEP(BN_CUDA_DENSE_PROF_ACT);
    if (ffn_act_f16_ready) {
        if (cuda_cublas_matmul_f16_preconverted(
                ctx, ctx->d_out, down, ctx->d_x_f16, dim, hidden_dim,
                n_tokens) != 0)
            return -1;
    } else {
        if (cuda_matmul_device_out(ctx, ctx->d_out, down, d_ffn_act, dim,
                                   hidden_dim, n_tokens, down_type) != 0)
            return -1;
    }
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
            kv_dim, kv_cache_stride, kv_cache_off, ctx->kv_f16);
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

static int cuda_prefill_ssm_layer(
        void *vctx, float *out, void *wqkv_buf, void *wz_buf,
        void *alpha_buf, void *beta_buf, void *qkvz_stacked_buf,
        void *ab_stacked_buf, void *ssm_out_buf, void *attn_norm_buf,
        void *conv1d_buf, void *dt_bias_buf, void *a_log_buf,
        void *ssm_norm_buf, void *ffn_gate_buf, void *ffn_up_buf,
        void *ffn_down_buf, void *ffn_norm_buf, const float *X, int n_tokens, int dim,
        int qkv_dim, int inner_dim, int num_k_heads,
        int head_k_dim, int num_v_heads, int head_v_dim, int conv_kernel,
        int ssm_idx, int wqkv_type, int wz_type, int alpha_type,
        int beta_type, int out_type, int hidden_dim, int ffn_gate_type,
        int ffn_up_type, int ffn_down_type, int act_type, float norm_eps,
        int *did_ffn) {
    BnCudaCtx *ctx = (BnCudaCtx *)vctx;
    BnCudaBuffer *wqkv = (BnCudaBuffer *)wqkv_buf;
    BnCudaBuffer *wz = (BnCudaBuffer *)wz_buf;
    BnCudaBuffer *alpha = (BnCudaBuffer *)alpha_buf;
    BnCudaBuffer *beta = (BnCudaBuffer *)beta_buf;
    BnCudaBuffer *qkvz = (BnCudaBuffer *)qkvz_stacked_buf;
    BnCudaBuffer *ab = (BnCudaBuffer *)ab_stacked_buf;
    BnCudaBuffer *ssm_out = (BnCudaBuffer *)ssm_out_buf;
    BnCudaBuffer *attn_norm = (BnCudaBuffer *)attn_norm_buf;
    BnCudaBuffer *conv1d = (BnCudaBuffer *)conv1d_buf;
    BnCudaBuffer *dt_bias = (BnCudaBuffer *)dt_bias_buf;
    BnCudaBuffer *a_log = (BnCudaBuffer *)a_log_buf;
    BnCudaBuffer *ssm_norm = (BnCudaBuffer *)ssm_norm_buf;
    BnCudaBuffer *ffn_gate = (BnCudaBuffer *)ffn_gate_buf;
    BnCudaBuffer *ffn_up = (BnCudaBuffer *)ffn_up_buf;
    BnCudaBuffer *ffn_down = (BnCudaBuffer *)ffn_down_buf;
    BnCudaBuffer *ffn_norm = (BnCudaBuffer *)ffn_norm_buf;
    int stacked_ffn_gateup = ffn_gate && ffn_gate->data && ffn_up == NULL;
    int fuse_ffn = ffn_gate && ffn_gate->data && ffn_down && ffn_down->data &&
                   ffn_norm && ffn_norm->data && hidden_dim > 0 &&
                   act_type == 0;
    if (did_ffn)
        *did_ffn = 0;
    if (getenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER"))
        return -1;
    if (!ctx || (!X && !ctx->d_out) ||
        !wqkv || !wqkv->data || !wz || !wz->data ||
        !alpha || !alpha->data || !beta || !beta->data ||
        !ssm_out || !ssm_out->data || !attn_norm || !attn_norm->data ||
        !conv1d || !conv1d->data || !dt_bias || !dt_bias->data ||
        !a_log || !a_log->data || !ssm_norm || !ssm_norm->data ||
        n_tokens <= 1 || dim <= 0 || qkv_dim <= 0 || inner_dim <= 0 ||
        num_k_heads <= 0 || head_k_dim <= 0 || num_v_heads <= 0 ||
        head_v_dim <= 0 || inner_dim != num_v_heads * head_v_dim ||
        qkv_dim != 2 * num_k_heads * head_k_dim + inner_dim ||
        conv_kernel <= 1 || ssm_idx < 0 || wqkv->rows != qkv_dim ||
        wqkv->cols != dim || wz->rows != inner_dim || wz->cols != dim ||
        alpha->rows != num_v_heads || alpha->cols != dim ||
        beta->rows != num_v_heads || beta->cols != dim ||
        ssm_out->rows != dim || ssm_out->cols != inner_dim ||
        attn_norm->rows * attn_norm->cols < dim ||
        ssm_norm->rows * ssm_norm->cols < head_v_dim ||
        !ctx->act_bufs[BN_GPU_VALUE_SSM_STATE] ||
        !ctx->act_bufs[BN_GPU_VALUE_SSM_CONV_STATE])
        return -1;
    if (!cuda_type_supported(wqkv_type) || !cuda_type_supported(wz_type) ||
        !cuda_type_supported(alpha_type) || !cuda_type_supported(beta_type) ||
        !cuda_type_supported(out_type))
        return -1;
    if (fuse_ffn) {
        if ((!stacked_ffn_gateup &&
             (!ffn_up || !ffn_up->data ||
              ffn_gate->rows != hidden_dim || ffn_up->rows != hidden_dim ||
              ffn_gate->cols != dim || ffn_up->cols != dim)) ||
            (stacked_ffn_gateup &&
             (ffn_gate->rows != hidden_dim * 2 || ffn_gate->cols != dim ||
              ffn_gate_type != ffn_up_type)) ||
            ffn_down->rows != dim || ffn_down->cols != hidden_dim ||
            ffn_norm->rows * ffn_norm->cols < dim ||
            !cuda_type_supported(ffn_gate_type) ||
            (!stacked_ffn_gateup && !cuda_type_supported(ffn_up_type)) ||
            !cuda_type_supported(ffn_down_type)) {
            fuse_ffn = 0;
        }
    }

    const int ssm_profile = getenv("BN_CUDA_SSM_PROFILE") != NULL;
    enum {
        BN_CUDA_SSM_PROF_UPLOAD = 0,
        BN_CUDA_SSM_PROF_NORM,
        BN_CUDA_SSM_PROF_QKVZ,
        BN_CUDA_SSM_PROF_AB,
        BN_CUDA_SSM_PROF_SCAN,
        BN_CUDA_SSM_PROF_OUT,
        BN_CUDA_SSM_PROF_FFN,
        BN_CUDA_SSM_PROF_READBACK,
        BN_CUDA_SSM_PROF_MAX
    };
    static double ssm_profile_totals[BN_CUDA_SSM_PROF_MAX] = {0.0};
    static unsigned long long ssm_profile_layers = 0;
    double ssm_profile_t0 = ssm_profile ? cuda_wall_ms() : 0.0;
#define BN_CUDA_SSM_PROFILE_STEP(code_) do {                          \
        if (ssm_profile) {                                            \
            cudaError_t profile_err__ = cudaDeviceSynchronize();      \
            double profile_now__ = cuda_wall_ms();                    \
            if (profile_err__ != cudaSuccess) {                       \
                fprintf(stderr,                                       \
                        "[bn:gpu:cuda] ssm profile sync failed: %s\n", \
                        cudaGetErrorString(profile_err__));           \
                return -1;                                            \
            }                                                         \
            ssm_profile_totals[(code_)] += profile_now__ - ssm_profile_t0; \
            ssm_profile_t0 = profile_now__;                           \
        }                                                             \
    } while (0)

    int use_stacked_prefill =
        getenv("BN_CUDA_DISABLE_SSM_STACKED_PREFILL") == NULL;
    int use_qkvz = use_stacked_prefill &&
                   qkvz && qkvz->data && wqkv_type == wz_type &&
                   qkvz->rows == qkv_dim + inner_dim &&
                   qkvz->cols == dim;
    int use_ab = use_stacked_prefill &&
                 ab && ab->data && alpha_type == beta_type &&
                 ab->rows == 2 * num_v_heads && ab->cols == dim;
    int ssm_stream_safe =
        getenv("BN_CUDA_DISABLE_SSM_STREAM_PREFILL") == NULL &&
        ctx->stream &&
        ctx->ssm_stream_entry_event &&
        ctx->ssm_stream_exit_event &&
        !cuda_force_quant_matmul_for_type(wqkv_type) &&
        !cuda_force_quant_matmul_for_type(wz_type) &&
        !cuda_force_quant_matmul_for_type(alpha_type) &&
        !cuda_force_quant_matmul_for_type(beta_type) &&
        !cuda_force_quant_matmul_for_type(out_type) &&
        ((use_qkvz && qkvz->f16_data) ||
         (!use_qkvz && wqkv->f16_data && wz->f16_data)) &&
        ((alpha_type == BN_GGUF_TENSOR_F32 &&
          beta_type == BN_GGUF_TENSOR_F32) ||
         (use_ab && ab->f16_data) ||
         (!use_ab && alpha->f16_data && beta->f16_data)) &&
        ssm_out->f16_data &&
        (!fuse_ffn ||
         (!cuda_force_quant_matmul_for_type(ffn_gate_type) &&
          !cuda_force_quant_matmul_for_type(ffn_down_type) &&
          ffn_gate->f16_data && ffn_down->f16_data &&
          (stacked_ffn_gateup ||
           (!cuda_force_quant_matmul_for_type(ffn_up_type) &&
            ffn_up->f16_data))));
    BnCudaExecStreamScope ssm_stream_scope(
        ctx, ctx->stream, ssm_stream_safe);
    cudaStream_t ssm_stream =
        ssm_stream_safe ? ctx->stream : (cudaStream_t)0;
    cudaStream_t parent_stream = ssm_stream_safe
        ? (ssm_stream_scope.prev ? ssm_stream_scope.prev : (cudaStream_t)0)
        : (cudaStream_t)0;
    if (ssm_stream_safe) {
        cudaError_t dep_err =
            cudaEventRecord(ctx->ssm_stream_entry_event, parent_stream);
        if (dep_err != cudaSuccess)
            return -1;
        dep_err = cudaStreamWaitEvent(ssm_stream,
                                      ctx->ssm_stream_entry_event, 0);
        if (dep_err != cudaSuccess)
            return -1;
    }
    size_t dim_values = (size_t)n_tokens * (size_t)dim;
    size_t qkv_values = (size_t)n_tokens * (size_t)qkv_dim;
    size_t z_values = (size_t)n_tokens * (size_t)inner_dim;
    size_t ab_values = (size_t)n_tokens * (size_t)num_v_heads;
    size_t qkvz_values = use_qkvz
        ? (size_t)n_tokens * (size_t)(qkv_dim + inner_dim) : 0u;
    size_t ab_stacked_values = use_ab
        ? (size_t)n_tokens * (size_t)(2 * num_v_heads) : 0u;
    size_t hidden_values = fuse_ffn
        ? (size_t)n_tokens * (size_t)hidden_dim : 0u;
    size_t gateup_values = hidden_values * 2u;
    size_t total_values = dim_values + qkv_values + z_values +
                          z_values + 2u * ab_values + qkvz_values +
                          ab_stacked_values +
                          (fuse_ffn ? (2u * dim_values + hidden_values +
                                       gateup_values) : 0u);
    size_t scratch_x_bytes = fuse_ffn && gateup_values > dim_values
        ? gateup_values * sizeof(float) : dim_values * sizeof(float);
    if (cuda_ensure_scratch(ctx, scratch_x_bytes,
                            dim_values * sizeof(float)) != 0)
        return -1;
    if (cuda_ensure_prefill(ctx, total_values) != 0)
        return -1;

    cudaError_t err = cudaSuccess;
    int alias_prev_output = !X && fuse_ffn &&
                            getenv("BN_CUDA_DISABLE_SSM_PREFILL_INPUT_ALIAS") == NULL;
    const float *d_input = alias_prev_output ? ctx->d_out : ctx->d_x;
    if (X) {
        err = cudaMemcpy(ctx->d_x, X, dim_values * sizeof(float),
                         cudaMemcpyHostToDevice);
    } else if (!alias_prev_output) {
        /*
         * Keep chained SSM prefill GPU-resident.  A synchronous D2D memcpy
         * here forces the host to wait once per SSM layer, which is visible on
         * small prompt batches even though the copied buffer is tiny.
         */
        err = cudaMemcpyAsync(ctx->d_x, ctx->d_out,
                              dim_values * sizeof(float),
                              cudaMemcpyDeviceToDevice,
                              ctx->exec_stream ? ctx->exec_stream
                                               : (cudaStream_t)0);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill ssm input upload failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_SSM_PROFILE_STEP(BN_CUDA_SSM_PROF_UPLOAD);

    float *d_norm = ctx->d_prefill;
    float *d_qkv = d_norm + dim_values;
    float *d_z = d_qkv + qkv_values;
    float *d_ssm = d_z + z_values;
    float *d_alpha = d_ssm + z_values;
    float *d_beta = d_alpha + ab_values;
    float *d_qkvz = d_beta + ab_values;
    float *d_ab = d_qkvz + qkvz_values;
    float *d_ffn_residual = d_ab + ab_stacked_values;
    float *d_ffn_norm = d_ffn_residual + (fuse_ffn ? dim_values : 0u);
    float *d_ffn_act = d_ffn_norm + (fuse_ffn ? dim_values : 0u);
    float *d_gateup = d_ffn_act + hidden_values;

    int threads = 256;
    int warps = threads / 32;
    const void *norm_f16 = NULL;
    int need_norm_f16 =
        ((use_qkvz && qkvz->f16_data) ||
         (!use_qkvz && (wqkv->f16_data || wz->f16_data)) ||
         (use_ab && ab->f16_data) ||
         (!use_ab && (alpha->f16_data || beta->f16_data)));
    if (need_norm_f16 && cuda_ensure_x_f16(ctx, dim_values) == 0) {
        rmsnorm_batch_to_f16_kernel<<<n_tokens, threads,
                                      (size_t)warps * sizeof(float),
                                      ssm_stream>>>(
            d_norm, (__half *)ctx->d_x_f16, d_input,
            (const float *)attn_norm->data, dim, n_tokens, norm_eps);
        norm_f16 = ctx->d_x_f16;
    } else {
        rmsnorm_batch_kernel<<<n_tokens, threads,
                               (size_t)warps * sizeof(float),
                               ssm_stream>>>(
            d_norm, d_input, (const float *)attn_norm->data, dim,
            n_tokens, norm_eps);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill ssm norm failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_SSM_PROFILE_STEP(BN_CUDA_SSM_PROF_NORM);

    if (use_qkvz) {
        if (cuda_matmul_device_out_preconverted_f16(
                ctx, d_qkvz, qkvz, d_norm, norm_f16,
                qkv_dim + inner_dim, dim, n_tokens, wqkv_type) != 0)
            return -1;
        int split_total = n_tokens * (qkv_dim + inner_dim);
        ssm_prefill_split_qkvz_kernel<<<(split_total + threads - 1) / threads,
                                        threads, 0, ssm_stream>>>(
            d_qkvz, d_qkv, d_z, qkv_dim, inner_dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm qkvz split failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    } else if (cuda_matmul_device_out_preconverted_f16(
                   ctx, d_qkv, wqkv, d_norm, norm_f16, qkv_dim, dim,
                   n_tokens, wqkv_type) != 0 ||
               cuda_matmul_device_out_preconverted_f16(
                   ctx, d_z, wz, d_norm, norm_f16, inner_dim, dim,
                   n_tokens, wz_type) != 0) {
        return -1;
    }
    BN_CUDA_SSM_PROFILE_STEP(BN_CUDA_SSM_PROF_QKVZ);

    int ab_preactivated = 0;
    if (getenv("BN_CUDA_DISABLE_SSM_F32_AB_PREFILL") == NULL &&
        alpha_type == BN_GGUF_TENSOR_F32 && beta_type == BN_GGUF_TENSOR_F32) {
        ssm_prefill_alpha_beta_f32_kernel<<<dim3(num_v_heads, n_tokens, 1),
                                            threads, 16 * sizeof(float),
                                            ssm_stream>>>(
            d_alpha, d_beta, (const float *)alpha->data,
            (const float *)beta->data, d_norm, (const float *)dt_bias->data,
            (const float *)a_log->data, num_v_heads, dim, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm f32 alpha/beta failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        ab_preactivated = 1;
    } else if (use_ab) {
        if (cuda_matmul_device_out_preconverted_f16(
                ctx, d_ab, ab, d_norm, norm_f16, 2 * num_v_heads,
                dim, n_tokens, alpha_type) != 0)
            return -1;
        int split_total = n_tokens * num_v_heads;
        ssm_prefill_split_ab_kernel<<<(split_total + threads - 1) / threads,
                                      threads, 0, ssm_stream>>>(
            d_ab, d_alpha, d_beta, num_v_heads, n_tokens);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm alpha/beta split failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    } else if (cuda_matmul_device_out_preconverted_f16(
                   ctx, d_alpha, alpha, d_norm, norm_f16,
                   num_v_heads, dim, n_tokens, alpha_type) != 0 ||
               cuda_matmul_device_out_preconverted_f16(
                   ctx, d_beta, beta, d_norm, norm_f16,
                   num_v_heads, dim, n_tokens, beta_type) != 0) {
        return -1;
    }
    BN_CUDA_SSM_PROFILE_STEP(BN_CUDA_SSM_PROF_AB);

    float q_scale = 1.0f / sqrtf((float)head_k_dim);
    int key_dim = num_k_heads * head_k_dim;
    size_t state_off = (size_t)ssm_idx * (size_t)num_v_heads *
                       (size_t)head_k_dim * (size_t)head_v_dim;
    size_t conv_off = (size_t)ssm_idx * (size_t)(conv_kernel - 1) *
                      (size_t)qkv_dim;
    int fast_prefill = head_k_dim == 128 && head_v_dim == 128 &&
                       !getenv("BN_CUDA_DISABLE_SSM_PREFILL_SCAN");
    if (fast_prefill) {
        ssm_prefill_conv_silu_kernel<<<(qkv_dim + threads - 1) / threads,
                                       threads, 0, ssm_stream>>>(
            d_qkv, cuda_act(ctx, BN_GPU_VALUE_SSM_CONV_STATE),
            (const float *)conv1d->data, qkv_dim, conv_kernel, n_tokens,
            conv_off);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm batched conv failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }

        ssm_prefill_l2norm_kernel<<<dim3(num_k_heads, n_tokens, 1),
                                    threads, 16 * sizeof(float),
                                    ssm_stream>>>(
            d_qkv, n_tokens, head_k_dim, 0, key_dim, num_k_heads, qkv_dim);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm batched l2norm failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }

        if (!ab_preactivated) {
            ssm_prefill_alpha_beta_kernel<<<
                ((int)ab_values + threads - 1) / threads, threads, 0,
                ssm_stream>>>(
                d_alpha, d_beta, (const float *)dt_bias->data,
                (const float *)a_log->data, num_v_heads, n_tokens);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] prefill ssm batched alpha/beta failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
        }

        ssm_prefill_delta_128_warp_kernel<<<dim3(num_v_heads, 32, 1),
                                             dim3(32, 4, 1), 0,
                                             ssm_stream>>>(
            cuda_act(ctx, BN_GPU_VALUE_SSM_STATE), d_ssm, d_qkv,
            d_alpha, d_beta, n_tokens, qkv_dim, num_k_heads, num_v_heads,
            q_scale, state_off, 0, key_dim, 2 * key_dim);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm batched delta failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }

        ssm_prefill_gate_kernel<<<dim3(num_v_heads, n_tokens, 1),
                                  threads, 8 * sizeof(float),
                                  ssm_stream>>>(
            d_ssm, d_z, (const float *)ssm_norm->data, head_v_dim,
            num_v_heads, n_tokens, norm_eps);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm batched gate failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    } else {
        for (int t = 0; t < n_tokens; t++) {
            float *qkv_t = d_qkv + (size_t)t * (size_t)qkv_dim;
            float *z_t = d_z + (size_t)t * (size_t)inner_dim;
            float *out_t = d_ssm + (size_t)t * (size_t)inner_dim;
            float *alpha_t = d_alpha + (size_t)t * (size_t)num_v_heads;
            float *beta_t = d_beta + (size_t)t * (size_t)num_v_heads;

            ssm_conv_silu_kernel<<<(qkv_dim + threads - 1) / threads,
                                   threads, 0, ssm_stream>>>(
                qkv_t, cuda_act(ctx, BN_GPU_VALUE_SSM_CONV_STATE),
                (const float *)conv1d->data, qkv_dim, conv_kernel,
                conv_off);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] prefill ssm conv failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }

            ssm_l2norm_kernel<<<num_k_heads, threads, 16 * sizeof(float),
                                ssm_stream>>>(
                qkv_t, qkv_t, head_k_dim, 0, key_dim);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] prefill ssm l2norm failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }

            if (!ab_preactivated) {
                ssm_alpha_beta_kernel<<<1, threads, 0, ssm_stream>>>(
                    alpha_t, beta_t, (const float *)dt_bias->data,
                    (const float *)a_log->data, num_v_heads);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    fprintf(stderr,
                            "[bn:gpu:cuda] prefill ssm alpha/beta failed: %s\n",
                            cudaGetErrorString(err));
                    return -1;
                }
            }

            if (head_k_dim == 128 && head_v_dim == 128 &&
                getenv("BN_CUDA_DISABLE_SSM_DELTA_128_WARP") == NULL) {
                ssm_delta_128_warp_kernel<<<dim3(num_v_heads, 32, 1),
                                            dim3(32, 4, 1), 0,
                                            ssm_stream>>>(
                    cuda_act(ctx, BN_GPU_VALUE_SSM_STATE), out_t, qkv_t,
                    qkv_t, qkv_t, alpha_t, beta_t, num_k_heads, q_scale,
                    state_off, 0, key_dim, 2 * key_dim);
            } else {
                ssm_delta_kernel<<<num_v_heads, threads,
                                   (size_t)head_v_dim * sizeof(float),
                                   ssm_stream>>>(
                    cuda_act(ctx, BN_GPU_VALUE_SSM_STATE), out_t, qkv_t,
                    qkv_t, qkv_t, alpha_t, beta_t, head_k_dim, head_v_dim,
                    num_k_heads, q_scale, state_off, 0, key_dim,
                    2 * key_dim);
            }
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] prefill ssm delta failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }

            ssm_gate_kernel<<<num_v_heads, threads, 8 * sizeof(float),
                              ssm_stream>>>(
                out_t, z_t, (const float *)ssm_norm->data, head_v_dim,
                norm_eps);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] prefill ssm gate failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
        }
    }
    BN_CUDA_SSM_PROFILE_STEP(BN_CUDA_SSM_PROF_SCAN);

    float *d_ssm_residual = alias_prev_output ? d_ffn_residual : ctx->d_out;
    if (cuda_matmul_device_out(ctx, d_ssm_residual, ssm_out, d_ssm, dim,
                               inner_dim, n_tokens, out_type) != 0)
        return -1;
    residual_add_kernel<<<((int)dim_values + threads - 1) / threads,
                           threads, 0, ssm_stream>>>(
        d_ssm_residual, d_input, (int)dim_values);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[bn:gpu:cuda] prefill ssm residual failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    BN_CUDA_SSM_PROFILE_STEP(BN_CUDA_SSM_PROF_OUT);

    if (fuse_ffn) {
        const int ssm_ffn_profile = getenv("BN_CUDA_SSM_FFN_PROFILE") != NULL;
        static double ssm_ffn_profile_norm = 0.0;
        static double ssm_ffn_profile_gateup = 0.0;
        static double ssm_ffn_profile_act = 0.0;
        static double ssm_ffn_profile_down = 0.0;
        static double ssm_ffn_profile_resid = 0.0;
        static unsigned long long ssm_ffn_profile_layers = 0;
        double ssm_ffn_t0 = ssm_ffn_profile ? cuda_wall_ms() : 0.0;
#define BN_CUDA_SSM_FFN_PROFILE_STEP(dst_) do {                       \
            if (ssm_ffn_profile) {                                    \
                cudaError_t ffn_profile_err__ = cudaDeviceSynchronize(); \
                double ffn_profile_now__ = cuda_wall_ms();            \
                if (ffn_profile_err__ != cudaSuccess) {               \
                    fprintf(stderr,                                   \
                            "[bn:gpu:cuda] ssm ffn profile sync failed: %s\n", \
                            cudaGetErrorString(ffn_profile_err__));   \
                    return -1;                                        \
                }                                                     \
                (dst_) += ffn_profile_now__ - ssm_ffn_t0;             \
                ssm_ffn_t0 = ffn_profile_now__;                       \
            }                                                         \
        } while (0)
        const void *ffn_norm_f16 = NULL;
        int need_ffn_norm_f16 =
            (stacked_ffn_gateup && ffn_gate->f16_data) ||
            (!stacked_ffn_gateup &&
             ((ffn_gate && ffn_gate->f16_data) ||
              (ffn_up && ffn_up->f16_data)));
        int have_ffn_norm_f16 =
            need_ffn_norm_f16 && cuda_ensure_x_f16(ctx, dim_values) == 0;
        if (alias_prev_output) {
            if (have_ffn_norm_f16) {
                rmsnorm_batch_to_f16_kernel<<<n_tokens, threads,
                                              (size_t)warps * sizeof(float),
                                              ssm_stream>>>(
                    d_ffn_norm, (__half *)ctx->d_x_f16, d_ffn_residual,
                    (const float *)ffn_norm->data, dim, n_tokens, norm_eps);
                ffn_norm_f16 = ctx->d_x_f16;
            } else {
                rmsnorm_batch_kernel<<<n_tokens, threads,
                                       (size_t)warps * sizeof(float),
                                       ssm_stream>>>(
                    d_ffn_norm, d_ffn_residual,
                    (const float *)ffn_norm->data, dim, n_tokens, norm_eps);
            }
        } else {
            if (have_ffn_norm_f16) {
                rmsnorm_batch_copy_to_f16_kernel<<<
                    n_tokens, threads, (size_t)warps * sizeof(float),
                    ssm_stream>>>(
                    d_ffn_norm, (__half *)ctx->d_x_f16, d_ffn_residual,
                    ctx->d_out, (const float *)ffn_norm->data, dim,
                    n_tokens, norm_eps);
                ffn_norm_f16 = ctx->d_x_f16;
            } else {
                rmsnorm_batch_copy_kernel<<<n_tokens, threads,
                                            (size_t)warps * sizeof(float),
                                            ssm_stream>>>(
                    d_ffn_norm, d_ffn_residual, ctx->d_out,
                    (const float *)ffn_norm->data, dim, n_tokens, norm_eps);
            }
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm ffn norm failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        BN_CUDA_SSM_FFN_PROFILE_STEP(ssm_ffn_profile_norm);
        int ffn_act_ready = 0;
        int gateup_f16_ready = 0;
        if (stacked_ffn_gateup && ffn_norm_f16 && ffn_gate->f16_data &&
            ffn_down->f16_data &&
            getenv("BN_CUDA_ENABLE_SSM_FFN_GATEUP_F16_OUT") != NULL &&
            getenv("BN_CUDA_DISABLE_SSM_FFN_GATEUP_F16_OUT") == NULL) {
            if (cuda_cublas_matmul_f16_preconverted_out_f16(
                    ctx, ctx->d_x, ffn_gate, ffn_norm_f16,
                    hidden_dim * 2, dim, n_tokens) != 0)
                return -1;
            gateup_f16_ready = 1;
        } else if (stacked_ffn_gateup &&
            ffn_gate_type == BN_GGUF_TENSOR_Q4_K &&
            (dim % BN_QK_K) == 0 &&
            getenv("BN_CUDA_ENABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH") != NULL &&
            getenv("BN_CUDA_DISABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH") == NULL) {
            int x_blocks = (dim + 31) / 32;
            if (cuda_ensure_q8_1(ctx, x_blocks * 32 * n_tokens) != 0)
                return -1;
            BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
            quantize_q8_1_batch_kernel<<<dim3(x_blocks, n_tokens, 1),
                                          32, 0, ssm_stream>>>(
                xq, d_ffn_norm, dim, n_tokens);
            dim3 grid((hidden_dim + warps - 1) / warps,
                      (n_tokens + 3) / 4, 1);
            q4k_dot_fused_gateup_silu_batch4_token_kernel<<<grid, threads,
                                                             0, ssm_stream>>>(
                d_ffn_act, (const BnBlockQ4K *)ffn_gate->data, xq,
                hidden_dim, hidden_dim, dim, n_tokens);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[bn:gpu:cuda] prefill ssm fused gate/up failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
            ffn_act_ready = 1;
        } else if (stacked_ffn_gateup) {
            if (cuda_matmul_device_out_preconverted_f16(
                    ctx, d_gateup, ffn_gate, d_ffn_norm, ffn_norm_f16,
                    hidden_dim * 2, dim, n_tokens, ffn_gate_type) != 0)
                return -1;
        } else {
            if (cuda_matmul_device_out_preconverted_f16(
                    ctx, d_gateup, ffn_gate, d_ffn_norm, ffn_norm_f16,
                    hidden_dim, dim, n_tokens, ffn_gate_type) != 0 ||
                cuda_matmul_device_out_preconverted_f16(
                    ctx, d_gateup + hidden_values, ffn_up, d_ffn_norm,
                    ffn_norm_f16, hidden_dim, dim, n_tokens,
                    ffn_up_type) != 0)
                return -1;
        }
        BN_CUDA_SSM_FFN_PROFILE_STEP(ssm_ffn_profile_gateup);
        int act_total = n_tokens * hidden_dim;
        int act_to_f16 = !ffn_act_ready && ffn_down->f16_data &&
                         cuda_ensure_x_f16(ctx, hidden_values) == 0;
        if (!ffn_act_ready && gateup_f16_ready && act_to_f16) {
            ffn_activation_batch_stacked_f16_to_f16_kernel<<<
                (act_total + threads - 1) / threads, threads, 0,
                ssm_stream>>>(
                (__half *)ctx->d_x_f16, (const __half *)ctx->d_x,
                hidden_dim, n_tokens, act_type);
        } else if (!ffn_act_ready && act_to_f16 && stacked_ffn_gateup) {
            ffn_activation_batch_stacked_to_f16_kernel<<<
                (act_total + threads - 1) / threads, threads, 0,
                ssm_stream>>>(
                (__half *)ctx->d_x_f16, d_gateup, hidden_dim, n_tokens,
                act_type);
        } else if (!ffn_act_ready && act_to_f16) {
            ffn_activation_batch_to_f16_kernel<<<
                (act_total + threads - 1) / threads, threads, 0,
                ssm_stream>>>(
                (__half *)ctx->d_x_f16, d_gateup, hidden_dim, n_tokens,
                act_type);
        } else if (!ffn_act_ready && stacked_ffn_gateup) {
            ffn_activation_batch_stacked_kernel<<<
                (act_total + threads - 1) / threads, threads, 0,
                ssm_stream>>>(
                d_ffn_act, d_gateup, hidden_dim, n_tokens, act_type);
        } else if (!ffn_act_ready) {
            ffn_activation_batch_kernel<<<
                (act_total + threads - 1) / threads, threads, 0,
                ssm_stream>>>(
                d_ffn_act, d_gateup, hidden_dim, n_tokens, act_type);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm ffn activation failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        BN_CUDA_SSM_FFN_PROFILE_STEP(ssm_ffn_profile_act);
        if (act_to_f16) {
            if (cuda_cublas_matmul_f16_preconverted(
                    ctx, ctx->d_out, ffn_down, ctx->d_x_f16, dim,
                    hidden_dim, n_tokens) != 0)
                return -1;
        } else {
            if (cuda_matmul_device_out(ctx, ctx->d_out, ffn_down, d_ffn_act,
                                       dim, hidden_dim, n_tokens,
                                       ffn_down_type) != 0)
                return -1;
        }
        BN_CUDA_SSM_FFN_PROFILE_STEP(ssm_ffn_profile_down);
        residual_add_kernel<<<((int)dim_values + threads - 1) / threads,
                               threads, 0, ssm_stream>>>(
            ctx->d_out, d_ffn_residual, (int)dim_values);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm ffn residual failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        BN_CUDA_SSM_FFN_PROFILE_STEP(ssm_ffn_profile_resid);
        if (ssm_ffn_profile) {
            ssm_ffn_profile_layers++;
            if ((ssm_ffn_profile_layers % 64u) == 0u) {
                double sum = ssm_ffn_profile_norm + ssm_ffn_profile_gateup +
                             ssm_ffn_profile_act + ssm_ffn_profile_down +
                             ssm_ffn_profile_resid;
                fprintf(stderr,
                        "[bn:gpu:cuda:ssm_ffn_profile] layers=%llu tokens=%d total=%.3f norm=%.3f gateup=%.3f act=%.3f down=%.3f resid=%.3f\n",
                        ssm_ffn_profile_layers, n_tokens, sum,
                        ssm_ffn_profile_norm, ssm_ffn_profile_gateup,
                        ssm_ffn_profile_act, ssm_ffn_profile_down,
                        ssm_ffn_profile_resid);
                ssm_ffn_profile_norm = 0.0;
                ssm_ffn_profile_gateup = 0.0;
                ssm_ffn_profile_act = 0.0;
                ssm_ffn_profile_down = 0.0;
                ssm_ffn_profile_resid = 0.0;
            }
        }
#undef BN_CUDA_SSM_FFN_PROFILE_STEP
        if (did_ffn)
            *did_ffn = 1;
    }
    BN_CUDA_SSM_PROFILE_STEP(BN_CUDA_SSM_PROF_FFN);

    if (ssm_stream_safe) {
        err = cudaEventRecord(ctx->ssm_stream_exit_event, ssm_stream);
        if (err != cudaSuccess)
            return -1;
        err = cudaStreamWaitEvent(parent_stream, ctx->ssm_stream_exit_event, 0);
        if (err != cudaSuccess)
            return -1;
    }

    if (out) {
        size_t out_bytes = dim_values * sizeof(float);
        if (cuda_ensure_host_out(ctx, out_bytes) != 0)
            return -1;
        err = cudaMemcpy(ctx->h_out, ctx->d_out, out_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[bn:gpu:cuda] prefill ssm readback failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
        memcpy(out, ctx->h_out, out_bytes);
    }
    BN_CUDA_SSM_PROFILE_STEP(BN_CUDA_SSM_PROF_READBACK);
    if (ssm_profile) {
        ssm_profile_layers++;
        if ((ssm_profile_layers % 64u) == 0u) {
            fprintf(stderr,
                    "[bn:gpu:cuda:ssm_profile] layers=%llu tokens=%d upload=%.3f norm=%.3f qkvz=%.3f ab=%.3f scan=%.3f out=%.3f ffn=%.3f readback=%.3f\n",
                    ssm_profile_layers, n_tokens,
                    ssm_profile_totals[BN_CUDA_SSM_PROF_UPLOAD],
                    ssm_profile_totals[BN_CUDA_SSM_PROF_NORM],
                    ssm_profile_totals[BN_CUDA_SSM_PROF_QKVZ],
                    ssm_profile_totals[BN_CUDA_SSM_PROF_AB],
                    ssm_profile_totals[BN_CUDA_SSM_PROF_SCAN],
                    ssm_profile_totals[BN_CUDA_SSM_PROF_OUT],
                    ssm_profile_totals[BN_CUDA_SSM_PROF_FFN],
                    ssm_profile_totals[BN_CUDA_SSM_PROF_READBACK]);
            for (int i = 0; i < BN_CUDA_SSM_PROF_MAX; i++)
                ssm_profile_totals[i] = 0.0;
        }
    }
#undef BN_CUDA_SSM_PROFILE_STEP
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
    case BN_GPU_CODE_WEIGHTED_ADD_SIGMOID: return "weighted_add_sigmoid";
    case BN_GPU_CODE_SILU_GATE: return "silu_gate";
    case BN_GPU_CODE_RELU2_GATE: return "relu2_gate";
    case BN_GPU_CODE_SIGMOID_GATE: return "sigmoid_gate";
    case BN_GPU_CODE_SILU_ACT: return "silu_act";
    case BN_GPU_CODE_RELU2_ACT: return "relu2_act";
    case BN_GPU_CODE_MOE_ROUTE_TOPK: return "moe_route_topk";
    case BN_GPU_CODE_MOE_ROUTED_FFN: return "moe_routed_ffn";
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
    BN_CUDA_PROFILE_MOE_GATEUP = 67,
    BN_CUDA_PROFILE_MOE_DOWN = 68,
    BN_CUDA_PROFILE_MOE_MID_QUANT = 69,
    BN_CUDA_PROFILE_MAX = 70
};

static const char *cuda_profile_name(int code) {
    if (code == BN_CUDA_PROFILE_QKV_MIXED) return "qkv_mixed";
    if (code == BN_CUDA_PROFILE_READBACK) return "readback";
    if (code == BN_CUDA_PROFILE_LOGITS) return "logits";
    if (code == BN_CUDA_PROFILE_MOE_GATEUP) return "moe_gateup";
    if (code == BN_CUDA_PROFILE_MOE_DOWN) return "moe_down";
    if (code == BN_CUDA_PROFILE_MOE_MID_QUANT) return "moe_mid_quant";
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
    case BN_GPU_CODE_WEIGHTED_ADD_SIGMOID:
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
    case BN_GPU_CODE_MOE_ROUTE_TOPK:
    case BN_GPU_CODE_MOE_ROUTED_FFN:
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
    case BN_GPU_CODE_WEIGHTED_ADD_SIGMOID:
    case BN_GPU_CODE_SILU_GATE:
    case BN_GPU_CODE_RELU2_GATE:
    case BN_GPU_CODE_SIGMOID_GATE:
    case BN_GPU_CODE_ROPE_QK:
        return op->buf_in == buf || op->buf_aux == buf;
    default:
        return 0;
    }
}

static void cuda_debug_scan_activation(BnCudaCtx *ctx, int buf,
                                       const char *label, int op_index) {
    if (!ctx || buf < 0 || buf >= BN_GPU_VALUE_COUNT || !ctx->act_bufs[buf])
        return;
    size_t n = ctx->act_sizes[buf] / sizeof(float);
    if (n == 0) return;
    if (cudaStreamSynchronize(ctx->exec_stream) != cudaSuccess)
        return;
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return;
    cudaError_t err = cudaMemcpy(tmp, ctx->act_bufs[buf], n * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(tmp);
        return;
    }
    int bad = 0;
    size_t first_bad = 0;
    float min_v = INFINITY;
    float max_v = -INFINITY;
    for (size_t i = 0; i < n; i++) {
        float v = tmp[i];
        if (!isfinite(v)) {
            if (bad == 0) first_bad = i;
            bad++;
            continue;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    if (bad || getenv("BN_CUDA_DEBUG_NAN_VERBOSE")) {
        fprintf(stderr,
                "[bn:gpu:cuda:nan] op=%d %s buf=%d n=%zu bad=%d first=%zu "
                "min=%.9g max=%.9g first_v=%.9g\n",
                op_index, label ? label : "?", buf, n, bad, first_bad,
                min_v, max_v, bad ? tmp[first_bad] : 0.0f);
    }
    free(tmp);
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

static int cuda_ops_have_moe(const BnGPUOp *ops, int n_ops) {
    if (!ops) return 0;
    for (int i = 0; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];
        if (cuda_op_mentions_buf(op, BN_GPU_VALUE_MOE_HB) ||
            cuda_op_mentions_buf(op, BN_GPU_VALUE_MOE_HB2) ||
            cuda_op_mentions_buf(op, BN_GPU_VALUE_MOE_OUT))
            return 1;
    }
    return 0;
}

static int cuda_ops_have_q8_moe_routed_ffn(const BnGPUOp *ops, int n_ops) {
    if (!ops) return 0;
    for (int i = 0; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];
        if (op->op_code == BN_GPU_CODE_MOE_ROUTED_FFN &&
            op->type == BN_GGUF_TENSOR_Q8_0)
            return 1;
    }
    return 0;
}

static int cuda_ops_have_moe_all2_q4q6_routed_ffn(
    const BnGPUOp *ops, int n_ops) {
    if (!ops) return 0;
    for (int i = 0; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];
        if (op->op_code == BN_GPU_CODE_MOE_ROUTED_FFN &&
            op->type == BN_GGUF_TENSOR_Q4_K &&
            op->cols > 0 && op->cols <= 2048 &&
            (int)op->p[0] >= 4096 &&
            (int)op->p[1] == 2 &&
            (int)op->p[2] == 2 &&
            (int)op->p[3] == BN_GGUF_TENSOR_Q6_K)
            return 1;
    }
    return 0;
}

static int cuda_ops_have_logits(const BnGPUOp *ops, int n_ops) {
    if (!ops) return 0;
    for (int i = 0; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];
        if (op->op_kind == BN_GPU_OP_LOGITS ||
            (op->op_code == BN_GPU_CODE_MATVEC &&
             op->buf_out == BN_GPU_VALUE_LOGITS))
            return 1;
    }
    return 0;
}

static int cuda_ops_have_moe_cublas_decode(const BnGPUOp *ops, int n_ops) {
    if (!ops || getenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE"))
        return 0;
    if (!bn_gpu_policy_all2_q4q6_moe_cublas_decode_enabled())
        return 0;
    for (int i = 0; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];
        if (op->op_code == BN_GPU_CODE_MOE_ROUTED_FFN &&
            op->type == BN_GGUF_TENSOR_Q4_K &&
            (int)op->p[1] == 2 &&
            (int)op->p[2] == 2 &&
            (int)op->p[3] == BN_GGUF_TENSOR_Q6_K &&
            op->cols > 0 &&
            (int)op->p[0] >= 4096)
            return 1;
    }
    return 0;
}

static int cuda_ops_moe_max_experts(const BnGPUOp *ops, int n_ops) {
    if (!ops) return 0;
    int max_experts = 0;
    for (int i = 0; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];
        int n_experts = 0;
        if (op->op_code == BN_GPU_CODE_MOE_ROUTE_TOPK)
            n_experts = (int)op->p[0];
        else if (op->op_code == BN_GPU_CODE_MOE_ROUTED_FFN)
            n_experts = (int)op->p[1];
        if (n_experts > max_experts)
            max_experts = n_experts;
    }
    return max_experts;
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
    const int debug_exec_fail = getenv("BN_CUDA_DEBUG_EXEC_FAIL") != NULL;
    const int debug_sync_each_op =
        getenv("BN_CUDA_DEBUG_SYNC_EACH_OP") != NULL;
    const int debug_nan = getenv("BN_CUDA_DEBUG_NAN") != NULL;
    static unsigned long long profile_calls = 0;
    static unsigned long long profile_ops[BN_CUDA_PROFILE_MAX] = {0};
    static double profile_ms[BN_CUDA_PROFILE_MAX] = {0.0};
    typedef struct {
        int code;
        int type;
        int rows;
        int cols;
        unsigned long long ops;
        double ms;
    } BnCudaShapeProfile;
    static BnCudaShapeProfile shape_profile[128];
    static int shape_profile_count = 0;
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
        if (!dumped || getenv("BN_CUDA_DUMP_OPS_EVERY")) {
            int dump_limit = cuda_env_int("BN_CUDA_DUMP_OPS_LIMIT", 256);
            if (dump_limit <= 0 || dump_limit > n_ops) dump_limit = n_ops;
            int limit = n_ops < dump_limit ? n_ops : dump_limit;
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
            getenv("BN_CUDA_ENABLE_GRAPH_EXEC") != NULL ||
            getenv("BN_CUDA_ENABLE_UNSAFE_MOE_FFN") != NULL;
        enable_q8_preq_all_flag =
            getenv("BN_CUDA_ENABLE_Q8_PREQ") != NULL &&
            getenv("BN_CUDA_DISABLE_Q8_PREQ") == NULL;
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
    int q8k_input_cache_valid = 0;
    int q8k_input_cache_buf = -1;
    int q8k_input_cache_cols = 0;
    int q8k_input_cache_tokens = 0;
    int q8k_input_cache_mark_op = -1;
    const int enable_q8k_input_cache =
        getenv("BN_CUDA_DISABLE_Q8K_INPUT_CACHE") == NULL;
#define BN_CUDA_Q8K_INPUT_CACHE_MARK(buf_, cols_, tokens_) do { \
        q8k_input_cache_valid = enable_q8k_input_cache; \
        q8k_input_cache_buf = (buf_); \
        q8k_input_cache_cols = (cols_); \
        q8k_input_cache_tokens = (tokens_); \
        q8k_input_cache_mark_op = i; \
    } while (0)
#define BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE() do { \
        q8k_input_cache_valid = 0; \
        q8k_input_cache_buf = -1; \
        q8k_input_cache_cols = 0; \
        q8k_input_cache_tokens = 0; \
        q8k_input_cache_mark_op = -1; \
    } while (0)
#define BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE_BUF(buf_) do { \
        if (q8k_input_cache_valid && q8k_input_cache_buf == (buf_)) \
            BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE(); \
    } while (0)
#define BN_CUDA_Q8K_INPUT_CACHE_MATCH(buf_, cols_, tokens_) \
    (enable_q8k_input_cache && q8k_input_cache_valid && \
     q8k_input_cache_buf == (buf_) && q8k_input_cache_cols == (cols_) && \
     q8k_input_cache_tokens == (tokens_))
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
    int moe_graph = cuda_ops_have_moe(ops, n_ops);
    int q8_moe_graph = cuda_ops_have_q8_moe_routed_ffn(ops, n_ops);
    int moe_max_experts = moe_graph
        ? cuda_ops_moe_max_experts(ops, n_ops)
        : 0;
    int default_moe_graph =
        moe_graph && !q8_moe_graph && moe_max_experts > 0 &&
        moe_max_experts <= cuda_env_int("BN_CUDA_MOE_GRAPH_MAX_EXPERTS", 128);
    int default_graph_exec =
        getenv("BN_CUDA_DISABLE_GRAPH_EXEC") == NULL &&
        getenv("BN_CUDA_ENABLE_MOE_FFN") == NULL &&
        (!moe_graph || default_moe_graph) &&
        cuda_ops_look_like_decode_graph(ops, n_ops, readback_buf,
                                        out_host, out_len);
    int q8_preq_logits_default =
        !disable_q8_preq_logits &&
        (getenv("BN_CUDA_ENABLE_Q8_PREQ_LOGITS") != NULL ||
         getenv("BN_CUDA_DISABLE_Q8_PREQ_LOGITS") == NULL);
    int graph_exec = (enable_graph_exec_flag || default_graph_exec) &&
                     n_ops > 10 && !profile;
    int graph_static_params = graph_exec && cuda_ops_have_logits(ops, n_ops);
    if (graph_exec && cuda_ops_have_moe_cublas_decode(ops, n_ops))
        graph_exec = 0;
    if (graph_exec &&
        bn_gpu_policy_all2_q4q6_moe_fast_ffn_enabled() &&
        bn_gpu_policy_all2_q4q6_moe_fast_graph_disabled() &&
        cuda_ops_have_moe_all2_q4q6_routed_ffn(ops, n_ops))
        graph_exec = 0;
    graph_static_params = graph_exec && graph_static_params;
    int graph_building = 0;
    int runtime_params_ready = 0;
    cudaGraphExec_t graph_instance = NULL;
    if (graph_exec) {
        int reserve_q8_1_cols = 0;
        int reserve_q8_k_cols = 0;
        int reserve_q8_k_batch = 1;
        for (int ri = 0; ri < n_ops; ri++) {
            const BnGPUOp *rop = &ops[ri];
            if (rop->op_code == BN_GPU_CODE_Q4K_MATVEC_SPLIT ||
                rop->op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT ||
                rop->op_code == BN_GPU_CODE_Q8_MATVEC_SPLIT) {
                int cols = (int)rop->p[1];
                if (rop->op_code == BN_GPU_CODE_Q4K_MATVEC_SPLIT &&
                    (rop->flags & BN_GPU_OP_FLAG_MATVEC_Q8K)) {
                    if (cols > reserve_q8_k_cols) reserve_q8_k_cols = cols;
                } else {
                    if (cols > reserve_q8_1_cols) reserve_q8_1_cols = cols;
                }
            } else if (rop->op_code == BN_GPU_CODE_FUSED_GATEUP_SILU) {
                int cols = (int)rop->p[1];
                if (rop->type == BN_GGUF_TENSOR_Q4_K ||
                    rop->type == BN_GGUF_TENSOR_Q5_K ||
                    rop->type == BN_GGUF_TENSOR_Q8_0) {
                    if (cols > reserve_q8_1_cols) reserve_q8_1_cols = cols;
                }
            } else if (rop->op_code == BN_GPU_CODE_MATVEC) {
                if (rop->type == BN_GGUF_TENSOR_Q6_K &&
                    (rop->cols % BN_QK_K) == 0 &&
                    (force_q6k_dot ||
                     (rop->flags & BN_GPU_OP_FLAG_MATVEC_Q8K) ||
                     (enable_q6k_dot && rop->cols >= 2048))) {
                    if (rop->cols > reserve_q8_k_cols)
                        reserve_q8_k_cols = rop->cols;
                } else if (rop->type == BN_GGUF_TENSOR_Q4_K &&
                           (rop->cols % BN_QK_K) == 0 && enable_q4k_dot &&
                           ((rop->flags & BN_GPU_OP_FLAG_MATVEC_Q8K) ||
                            getenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT") != NULL) &&
                           getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT") == NULL) {
                    if (rop->cols > reserve_q8_k_cols)
                        reserve_q8_k_cols = rop->cols;
                } else if ((rop->type == BN_GGUF_TENSOR_Q4_K ||
                            rop->type == BN_GGUF_TENSOR_Q5_K ||
                            rop->type == BN_GGUF_TENSOR_Q8_0) &&
                           (rop->cols % BN_QK_K) == 0) {
                    if (rop->cols > reserve_q8_1_cols)
                        reserve_q8_1_cols = rop->cols;
                }
            } else if (rop->op_code == BN_GPU_CODE_MOE_ROUTED_FFN) {
                int hidden = (int)rop->p[0];
                int k = (int)rop->p[2];
                int down_type = (int)rop->p[3];
                if (hidden > 0 && k > 0) {
                    int mid_elems = hidden * k;
                    if (rop->type == BN_GGUF_TENSOR_Q8_0) {
                        int elems = mid_elems > rop->cols ? mid_elems : rop->cols;
                        if (elems > reserve_q8_1_cols)
                            reserve_q8_1_cols = elems;
                    } else if (rop->type == BN_GGUF_TENSOR_Q4_K) {
                        if (rop->cols > reserve_q8_1_cols)
                            reserve_q8_1_cols = rop->cols;
                        if (down_type == BN_GGUF_TENSOR_Q6_K ||
                            down_type == BN_GGUF_TENSOR_Q4_K) {
                            if (hidden > reserve_q8_k_cols)
                                reserve_q8_k_cols = hidden;
                            if (k > reserve_q8_k_batch)
                                reserve_q8_k_batch = k;
                        }
                    }
                }
            }
        }
        if (reserve_q8_1_cols > 0 &&
            cuda_ensure_q8_1(ctx, reserve_q8_1_cols) != 0)
            return -1;
        if (reserve_q8_k_cols > 0 &&
            cuda_ensure_q8_k(ctx, reserve_q8_k_cols,
                             reserve_q8_k_batch) != 0)
            return -1;
    }
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
        runtime_params_ready = graph_exec;
    }
    if (!graph_exec && !profile &&
        getenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FLASH_FUSE") == NULL) {
        int wants_qk_norm_flash_runtime = 0;
        for (int ri = 0; ri + 3 < n_ops; ri++) {
            const BnGPUOp *qnorm = &ops[ri];
            const BnGPUOp *knorm = &ops[ri + 1];
            const BnGPUOp *rope = &ops[ri + 2];
            const BnGPUOp *flash = &ops[ri + 3];
            if (qnorm->op_code == BN_GPU_CODE_PER_HEAD_RMSNORM &&
                qnorm->buf_in == BN_GPU_VALUE_Q &&
                knorm->op_code == BN_GPU_CODE_PER_HEAD_RMSNORM &&
                knorm->buf_in == BN_GPU_VALUE_KEY_CACHE &&
                rope->op_code == BN_GPU_CODE_ROPE_QK &&
                rope->buf_in == BN_GPU_VALUE_Q &&
                rope->buf_aux == BN_GPU_VALUE_KEY_CACHE &&
                flash->op_code == BN_GPU_CODE_FLASH_ATTN &&
                flash->buf_in == BN_GPU_VALUE_Q) {
                wants_qk_norm_flash_runtime = 1;
                break;
            }
        }
        if (wants_qk_norm_flash_runtime) {
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
            if (rt.pos >= 0 && rt.n_kv > 0 && rt.seq_len > 0) {
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
                runtime_params_ready = 1;
            }
        }
    }
    for (int i = 0; i < n_ops; i++) {
        if (skip_ops[i])
            continue;
        const BnGPUOp *op = &ops[i];
        const BnGPUOp *next = (i + 1 < n_ops) ? &ops[i + 1] : NULL;
#define BN_CUDA_EXEC_FAIL(reason) do { \
            if (debug_exec_fail) { \
                fprintf(stderr, \
                        "[bn:gpu:cuda:exec-fail] op=%d/%d code=%s(%d) " \
                        "reason=%s in=%d out=%d aux=%d rows=%d cols=%d " \
                        "p0=%u p1=%u p2=%u p3=%u p5=%u p6=%u p7=%u\n", \
                        i, n_ops, cuda_op_name(op->op_code), op->op_code, \
                        (reason), op->buf_in, op->buf_out, op->buf_aux, \
                        op->rows, op->cols, op->p[0], op->p[1], op->p[2], \
                        op->p[3], op->p[5], op->p[6], op->p[7]); \
            } \
            return -1; \
        } while (0)
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
                op->rows <= 0 || op->cols <= 0) {
                if (debug_exec_fail) {
                    fprintf(stderr,
                            "[bn:gpu:cuda:matvec-invalid] w=%p wdata=%p "
                            "in=%p out=%p type=%d supported=%d\n",
                            (void *)w, w ? w->data : NULL, (void *)in,
                            (void *)out, op->type,
                            cuda_type_supported(op->type));
                }
                BN_CUDA_EXEC_FAIL("matvec invalid args");
            }
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
                    int copy_to_kv_f16 = ctx->kv_f16 &&
                        (copy->buf_out == BN_GPU_VALUE_KEY_CACHE ||
                         copy->buf_out == BN_GPU_VALUE_VALUE_CACHE);
                    if (bw && bw->data && copy_out && !copy_to_kv_f16) {
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
            int direct_kv_f16 = ctx->kv_f16 &&
                (op->buf_out == BN_GPU_VALUE_KEY_CACHE ||
                 op->buf_out == BN_GPU_VALUE_VALUE_CACHE) &&
                fused_copy_idx < 0 && bias == NULL && bias_idx < 0;
            void *direct_kv_dst = out;
            size_t direct_kv_offset = out_offset;
            if (direct_kv_f16) {
                float *tmp = cuda_act(ctx, BN_GPU_VALUE_SCRATCH);
                if (!tmp || op->rows > (int)(ctx->act_sizes[BN_GPU_VALUE_SCRATCH] /
                                             sizeof(float)))
                    BN_CUDA_EXEC_FAIL("direct kv f16 scratch unavailable");
                out = tmp;
                out_offset = 0;
            }
            int stable_decode_matvec =
                graph_exec && out_offset == 0 && !direct_kv_f16 &&
                fused_copy_idx < 0;
            if (!direct_kv_f16 && next && i + 2 < n_ops &&
                op->type == BN_GGUF_TENSOR_Q5_K &&
                (op->cols % BN_QK_K) == 0 && enable_q5k_dot &&
                out_offset == 0 && bias == NULL && bias_idx < 0 &&
                fused_copy_idx < 0 &&
                next->op_code == BN_GPU_CODE_DEINTERLEAVE_Q &&
                next->buf_in == op->buf_out &&
                ops[i + 2].op_code == BN_GPU_CODE_MATVEC &&
                ops[i + 2].type == BN_GGUF_TENSOR_Q5_K &&
                ops[i + 2].buf_in == op->buf_in &&
                ops[i + 2].cols == op->cols &&
                ops[i + 2].rows > 0 &&
                cuda_find_fusable_bias(ops, n_ops, i + 2,
                                       ops[i + 2].buf_out,
                                       ops[i + 2].rows) < 0 &&
                getenv("BN_CUDA_ENABLE_Q5K_DEINT_PAIR_MATVEC") != NULL) {
                BnCudaBuffer *w1 = (BnCudaBuffer *)ops[i + 2].W_buf;
                float *out1 = cuda_act(ctx, ops[i + 2].buf_out);
                if (w1 && w1->data && out1) {
                    if (cuda_ensure_q8_1(ctx, op->cols) != 0)
                        BN_CUDA_EXEC_FAIL("q5k deint pair q8_1 scratch alloc failed");
                    BnCudaBlockQ8_1 *xq =
                        (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        quantize_q8_1_kernel, (op->cols + 31) / 32, 32,
                        0, xq, in, op->cols);
                    int q5_threads = 256;
                    int warps = q5_threads / 32;
                    int blocks0 = (op->rows + warps - 1) / warps;
                    int blocks1 = (ops[i + 2].rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH(ctx, q5k_dot_matvec_kernel, blocks0,
                        q5_threads, 0,
                        out, (const BnBlockQ5K *)w->data, xq,
                        (const float *)NULL,
                        op->rows, op->cols, 0);
                    BN_CUDA_LAUNCH(ctx, q5k_dot_matvec_kernel, blocks1,
                        q5_threads, 0,
                        out1, (const BnBlockQ5K *)w1->data, xq,
                        (const float *)NULL,
                        ops[i + 2].rows, ops[i + 2].cols,
                        (size_t)ops[i + 2].p[5]);
                    skip_ops[i + 2] = 1;
                    break;
                }
            }
            if (!direct_kv_f16 && next &&
                op->type == BN_GGUF_TENSOR_Q6_K &&
                next->op_code == BN_GPU_CODE_MATVEC &&
                next->type == BN_GGUF_TENSOR_Q4_K &&
                next->buf_in == op->buf_in &&
                next->cols == op->cols &&
                op->cols % BN_QK_K == 0 &&
                op->rows > 0 && next->rows > 0 &&
                bias == NULL && bias_idx < 0 &&
                (op->cols < 5120 ||
                 getenv("BN_CUDA_ENABLE_Q6K_Q4K_PAIR_MATVEC") != NULL) &&
                getenv("BN_CUDA_DISABLE_Q6K_Q4K_PAIR_MATVEC") == NULL) {
                BnCudaBuffer *w1 = (BnCudaBuffer *)next->W_buf;
                float *out1 = cuda_act(ctx, next->buf_out);
                if (w1 && w1->data && out1) {
                    if (cuda_ensure_q8_k(ctx, op->cols, 1) != 0)
                        BN_CUDA_EXEC_FAIL("q6k/q4k pair q8k scratch alloc failed");
                    BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8k_batch_kernel,
                        dim3(op->cols / BN_QK_K, 1, 1), BN_QK_K, 0,
                        xq, in, op->cols, 1);
                    int pair_threads = 256;
                    int warps = pair_threads / 32;
                    if (cuda_use_q6k_3warp_exact(op->rows, op->cols)) {
                        BN_CUDA_LAUNCH(ctx, q6k_dot_matvec_3warp_kernel,
                            op->rows, 96, 0,
                            out, (const BnBlockQ6K *)w->data, xq,
                            (const float *)NULL,
                            op->rows, op->cols, out_offset);
                    } else if (cuda_use_q6k_5warp_exact(op->rows, op->cols)) {
                        BN_CUDA_LAUNCH(ctx, q6k_dot_matvec_5warp_kernel,
                            op->rows, 160, 0,
                            out, (const BnBlockQ6K *)w->data, xq,
                            (const float *)NULL,
                            op->rows, op->cols, out_offset);
                    } else if (cuda_use_q6k_2warp_long(op->rows, op->cols)) {
                        int q6_blocks = (op->rows + 3) / 4;
                        BN_CUDA_LAUNCH(ctx, q6k_dot_matvec_2warp_kernel,
                            q6_blocks, pair_threads, 0,
                            out, (const BnBlockQ6K *)w->data, xq,
                            (const float *)NULL,
                            op->rows, op->cols, out_offset);
                    } else if (cuda_use_q6k_4warp_long(op->rows, op->cols)) {
                        int q6_blocks = op->rows;
                        BN_CUDA_LAUNCH(ctx, q6k_dot_matvec_4warp_kernel,
                            q6_blocks, pair_threads, 0,
                            out, (const BnBlockQ6K *)w->data, xq,
                            (const float *)NULL,
                            op->rows, op->cols, out_offset);
                    } else if ((op->cols <= 4096 ||
                         (op->cols >= 5120 && op->cols <= 8192) ||
                         op->cols >= 16384) &&
                        !cuda_disable_q6k_matvec4_shape(op->rows, op->cols) &&
                        getenv("BN_CUDA_DISABLE_Q6K_MATVEC4") == NULL) {
                        int q6_blocks =
                            (op->rows + warps * 4 - 1) / (warps * 4);
                        BN_CUDA_LAUNCH(ctx, q6k_dot_matvec4_kernel,
                            q6_blocks, pair_threads, 0,
                            out, (const BnBlockQ6K *)w->data, xq,
                            (const float *)NULL,
                            op->rows, op->cols, out_offset);
                    } else {
                        int q6_blocks = (op->rows + warps - 1) / warps;
                        BN_CUDA_LAUNCH(ctx, q6k_dot_matvec_kernel,
                            q6_blocks, pair_threads, 0,
                            out, (const BnBlockQ6K *)w->data, xq,
                            (const float *)NULL,
                            op->rows, op->cols, out_offset);
                    }
                    int q4_blocks = (next->rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH(ctx, q4k_q8k_dot_matvec_kernel,
                        q4_blocks, pair_threads, 0,
                        out1, (const BnBlockQ4K *)w1->data, xq,
                        (const float *)NULL,
                        next->rows, next->cols, (size_t)next->p[5]);
                    i++;
                    break;
                }
            }
            int use_f16_q6k_matvec =
                (op->flags & BN_GPU_OP_FLAG_MATVEC_EXACT_Q6K) == 0 &&
                (getenv("BN_CUDA_ENABLE_F16_Q6K_MATVEC") != NULL ||
                (getenv("BN_CUDA_DISABLE_F16_Q6K_MATVEC") == NULL &&
                 op->rows <= 2048 && op->cols >= 8192));
            int q8_small_ssm_matvec =
                op->type == BN_GGUF_TENSOR_Q8_0 &&
                op->rows <= 256 && op->cols >= 4096 &&
                (op->buf_out == BN_GPU_VALUE_SSM_V ||
                 op->buf_out == BN_GPU_VALUE_SSM_ALPHA ||
                 op->buf_out == BN_GPU_VALUE_SSM_BETA) &&
                getenv("BN_CUDA_DISABLE_Q8_0_SSM_MATVEC") == NULL;
            if (!is_logits_op && w->f16_data && out_offset == 0 &&
                bias == NULL && bias_idx < 0 &&
                ((op->type == BN_GGUF_TENSOR_Q8_0 &&
                  ((q8_small_ssm_matvec &&
                    getenv("BN_CUDA_ENABLE_F16_Q8_0_SSM_MATVEC") != NULL &&
                    getenv("BN_CUDA_DISABLE_F16_Q8_0_SSM_MATVEC") == NULL) ||
                   (op->cols <= 2048 &&
                    getenv("BN_CUDA_ENABLE_F16_Q8_0_MATVEC") != NULL)) &&
                  getenv("BN_CUDA_DISABLE_F16_Q8_0_MATVEC") == NULL) ||
                 op->type == BN_GGUF_TENSOR_Q3_K ||
                 op->type == BN_GGUF_TENSOR_IQ3_XXS ||
                 op->type == BN_GGUF_TENSOR_IQ4_XS ||
                 (op->type == BN_GGUF_TENSOR_Q5_K &&
                  getenv("BN_CUDA_ENABLE_F16_Q5K_MATVEC") != NULL))) {
                int q_threads = 256;
                int q_warps = q_threads / 32;
                int q_blocks = (op->rows + q_warps - 1) / q_warps;
                BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                    f16_matvec_warp_kernel, q_blocks,
                    q_threads, 0, out, (const __half *)w->f16_data, in,
                    (const float *)NULL, op->rows, op->cols, out_offset);
                break;
            }
            if (!is_logits_op && w->f16_data && out_offset == 0 &&
                bias == NULL && bias_idx < 0 &&
                op->type == BN_GGUF_TENSOR_Q6_K && use_f16_q6k_matvec) {
                int q_threads = 256;
                int q_warps = q_threads / 32;
                int q_blocks = (op->rows + q_warps - 1) / q_warps;
                BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                    f16_matvec_warp_kernel, q_blocks,
                    q_threads, 0, out, (const __half *)w->f16_data, in,
                    (const float *)NULL, op->rows, op->cols, out_offset);
                break;
            }
            if (is_logits_op && w->f16_data && out_offset == 0 &&
                bias == NULL && bias_idx < 0 &&
                getenv("BN_CUDA_ENABLE_CUBLAS_LOGITS") != NULL) {
                if (cuda_convert_f32_to_f16(ctx, in, (size_t)op->cols) == 0 &&
                    cuda_cublas_matmul_f16_preconverted(
                        ctx, out, w, ctx->d_x_f16, op->rows, op->cols, 1) == 0) {
                    break;
                }
            }
            if (is_logits_op && w->f32_data && out_offset == 0 &&
                bias == NULL && bias_idx < 0 &&
                op->type == BN_GGUF_TENSOR_Q6_K && op->rows >= 65536 &&
                getenv("BN_CUDA_ENABLE_F32_LOGITS_MATVEC") != NULL &&
                getenv("BN_CUDA_DISABLE_F32_LOGITS_MATVEC") == NULL) {
                int q_threads = 256;
                int q_warps = q_threads / 32;
                int q_blocks = (op->rows + q_warps - 1) / q_warps;
                BN_CUDA_LAUNCH(ctx, f32_matvec_warp_kernel, q_blocks,
                    q_threads, 0, out, (const float *)w->f32_data, in,
                    (const float *)NULL, op->rows, op->cols, out_offset);
                break;
            }
            if (is_logits_op && w->f16_data && out_offset == 0 &&
                bias == NULL && bias_idx < 0 &&
                op->type == BN_GGUF_TENSOR_Q6_K && op->rows >= 65536 &&
                getenv("BN_CUDA_ENABLE_F16_LOGITS_MATVEC") != NULL) {
                int q_threads = 256;
                int q_warps = q_threads / 32;
                int q_blocks = (op->rows + q_warps - 1) / q_warps;
                BN_CUDA_LAUNCH(ctx, f16_matvec_warp_kernel, q_blocks,
                    q_threads, 0, out, (const __half *)w->f16_data, in,
                    (const float *)NULL, op->rows, op->cols, out_offset);
                break;
            }
            if (!direct_kv_f16 && next && op->type == BN_GGUF_TENSOR_Q4_K &&
                next->op_code == BN_GPU_CODE_MATVEC &&
                next->type == BN_GGUF_TENSOR_Q4_K &&
                next->buf_in == op->buf_in &&
                next->rows == op->rows &&
                next->cols == op->cols &&
                op->p[5] == 0 && next->p[5] == 0 &&
                op->p[6] && next->p[6] &&
                bias == NULL && bias_idx < 0 &&
                (op->cols % BN_QK_K) == 0 &&
                enable_q4k_dot &&
                getenv("BN_CUDA_DISABLE_Q4K_PAIR_MATVEC") == NULL) {
                BnCudaBuffer *w1 = (BnCudaBuffer *)next->W_buf;
                float *out1 = cuda_act(ctx, next->buf_out);
                if (w1 && w1->data && out1) {
                    if (cuda_ensure_q8_k(ctx, op->cols, 1) != 0)
                        BN_CUDA_EXEC_FAIL("q4k pair q8k scratch alloc failed");
                    BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8k_batch_kernel,
                        dim3(op->cols / BN_QK_K, 1, 1), BN_QK_K, 0,
                        xq, in, op->cols, 1);
                    int q4_threads = 256;
                    int warps = q4_threads / 32;
                    int blocks = (op->rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH(ctx, q4k_q8k_dot_matvec_pair_kernel,
                        blocks, q4_threads, 0,
                        out, out1, (const BnBlockQ4K *)w->data,
                        (const BnBlockQ4K *)w1->data, xq, op->rows,
                        op->cols);
                    i++;
                    break;
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
                        (op->flags & BN_GPU_OP_FLAG_MATVEC_Q8K) ||
                        (enable_q6k_dot && op->cols >= 2048))) {
                int use_q6k_q8_1 =
                    getenv("BN_CUDA_ENABLE_Q6K_Q8_1_DOT") != NULL &&
                    getenv("BN_CUDA_DISABLE_Q6K_Q8_1_DOT") == NULL &&
                    (is_logits_op ||
                     getenv("BN_CUDA_ENABLE_Q6K_Q8_1_ALL") != NULL);
                int use_q6k_mmvq =
                    (op->flags & BN_GPU_OP_FLAG_MATVEC_EXACT_Q6K) == 0 &&
                    getenv("BN_CUDA_DISABLE_Q6K_MMVQ") == NULL &&
                    ((op->cols >= 4096 && op->rows >= 5120) ||
                     (op->cols >= 2048 && op->rows >= 50000) ||
                     (op->rows == 512 && op->cols == 2048 &&
                      getenv("BN_CUDA_DISABLE_Q6K_MMVQ_512_2048") == NULL) ||
                     (op->rows == 1536 && op->cols == 8960 &&
                      getenv("BN_CUDA_DISABLE_Q6K_MMVQ_1536_8960") == NULL) ||
                     ((op->rows == 2560 && op->cols == 9728) &&
                      getenv("BN_CUDA_DISABLE_Q6K_MMVQ_2560_9728") == NULL) ||
                     (is_logits_op && op->cols == 1536 &&
                      op->rows >= 50000 &&
                      getenv("BN_CUDA_DISABLE_Q6K_MMVQ_LOGITS_1536") == NULL)) &&
                    bias == NULL && bias_idx < 0;
                if (use_q6k_mmvq) {
                    if (cuda_ensure_q8_1(ctx, op->cols) != 0)
                        BN_CUDA_EXEC_FAIL("q6k mmvq q8_1 scratch alloc failed");
                    BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        quantize_q8_1_kernel, (op->cols + 31) / 32, 32, 0,
                        xq, in, op->cols);
                    if (is_logits_op && op->cols <= 2560 &&
                        op->rows >= 50000 &&
                        ((op->cols == 1536 &&
                          getenv("BN_CUDA_DISABLE_Q6K_MMVQ_LOGITS_1536") == NULL) ||
                         (op->cols > 1536 &&
                          getenv("BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_LOGITS_SMALL") == NULL)) &&
                        getenv("BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_1536") == NULL) {
                        BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                            q6k_dot_matvec_mmvq_2warp_kernel, op->rows,
                            64, 0, out, (const BnBlockQ6K *)w->data, xq,
                            (const float *)NULL, op->rows, op->cols,
                            out_offset);
                    } else {
                        BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                            q6k_dot_matvec_mmvq_kernel, op->rows, 128, 0,
                            out, (const BnBlockQ6K *)w->data, xq,
                            (const float *)NULL, op->rows, op->cols,
                            out_offset);
                    }
                    break;
                }
                if (use_q6k_q8_1) {
                    if (cuda_ensure_q8_1(ctx, op->cols) != 0)
                        BN_CUDA_EXEC_FAIL("q6k q8_1 scratch alloc failed");
                    BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        quantize_q8_1_kernel, (op->cols + 31) / 32, 32, 0,
                        xq, in, op->cols);
                    int q6_threads = 256;
                    int warps = q6_threads / 32;
                    int blocks = (op->rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q6k_dot_matvec_q8_1_kernel, blocks,
                        q6_threads, 0,
                        out, (const BnBlockQ6K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                    break;
                }
                int reuse_q8k_input =
                    BN_CUDA_Q8K_INPUT_CACHE_MATCH(op->buf_in, op->cols, 1) &&
                    q8k_input_cache_mark_op == i - 1;
                BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                if (!reuse_q8k_input) {
                    if (cuda_ensure_q8_k(ctx, op->cols, 1) != 0)
                        BN_CUDA_EXEC_FAIL("q6k q8k scratch alloc failed");
                    xq = (BnBlockQ8K *)ctx->d_q8_k;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        quantize_q8k_batch_kernel,
                        dim3(op->cols / BN_QK_K, 1, 1), BN_QK_K, 0,
                        xq, in, op->cols, 1);
                }
                BN_CUDA_Q8K_INPUT_CACHE_MARK(op->buf_in, op->cols, 1);
                int q6_threads = 256;
                int warps = q6_threads / 32;
                if (cuda_use_q6k_3warp_exact(op->rows, op->cols)) {
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q6k_dot_matvec_3warp_kernel, op->rows, 96, 0,
                        out, (const BnBlockQ6K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                } else if (cuda_use_q6k_5warp_exact(op->rows, op->cols)) {
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q6k_dot_matvec_5warp_kernel, op->rows, 160, 0,
                        out, (const BnBlockQ6K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                } else if (cuda_use_q6k_2warp_long(op->rows, op->cols)) {
                    int blocks = (op->rows + 3) / 4;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q6k_dot_matvec_2warp_kernel, blocks,
                        q6_threads, 0,
                        out, (const BnBlockQ6K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                } else if (cuda_use_q6k_4warp_long(op->rows, op->cols)) {
                    int blocks = op->rows;
                    int fuse_resid_norm =
                        getenv("BN_CUDA_ENABLE_Q6K_DOWN_RESID_RMSNORM_FUSE") != NULL &&
                        next && next->op_code == BN_GPU_CODE_RESIDUAL_RMSNORM &&
                        next->buf_in == BN_GPU_VALUE_X &&
                        next->buf_aux == op->buf_out &&
                        out_offset == 0 && bias == NULL &&
                        bias_idx < 0 && fused_copy_idx < 0 &&
                        op->rows == (int)next->p[0];
                    BnCudaBuffer *nw = fuse_resid_norm
                        ? (BnCudaBuffer *)next->W_buf : NULL;
                    float *resid = fuse_resid_norm
                        ? cuda_act(ctx, next->buf_in) : NULL;
                    float *norm_out = fuse_resid_norm
                        ? cuda_act(ctx, next->buf_out) : NULL;
                    if (fuse_resid_norm && nw && nw->data && resid &&
                        norm_out) {
                        BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                            q6k_dot_matvec_4warp_residual_kernel, blocks,
                            q6_threads, 0,
                            resid, (const BnBlockQ6K *)w->data, xq,
                            resid, op->rows, op->cols);
                        int norm_threads = op->rows >= 2048 ? 512 : threads;
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                            rmsnorm_kernel, 1, norm_threads,
                            (size_t)norm_threads * sizeof(float),
                            norm_out, resid, (const float *)nw->data,
                            op->rows, cuda_u32_to_f32(next->p[1]));
                        i++;
                    } else {
                        BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                            q6k_dot_matvec_4warp_kernel, blocks,
                            q6_threads, 0,
                            out, (const BnBlockQ6K *)w->data, xq, bias,
                            op->rows, op->cols, out_offset);
                    }
                } else if ((op->cols <= 4096 ||
                     (op->cols >= 5120 && op->cols <= 8192) ||
                     op->cols >= 16384) &&
                    !cuda_disable_q6k_matvec4_shape(op->rows, op->cols) &&
                    getenv("BN_CUDA_DISABLE_Q6K_MATVEC4") == NULL) {
                    int blocks = (op->rows + warps * 4 - 1) / (warps * 4);
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q6k_dot_matvec4_kernel, blocks,
                        q6_threads, 0,
                        out, (const BnBlockQ6K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                } else {
                    int blocks = (op->rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q6k_dot_matvec_kernel, blocks,
                        q6_threads, 0,
                        out, (const BnBlockQ6K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                }
            } else if (op->type == BN_GGUF_TENSOR_Q6_K &&
                       (op->cols % BN_QK_K) == 0 && enable_q6k_warp) {
                int q6_threads = 256;
                int warps = q6_threads / 32;
                int blocks = (op->rows + warps - 1) / warps;
                BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                    q6k_matvec_warp_kernel, blocks,
                    q6_threads, 0,
                    out, (const BnBlockQ6K *)w->data, in, bias,
                    op->rows, op->cols, out_offset);
            } else if (op->type == BN_GGUF_TENSOR_Q4_K &&
                       (op->cols % BN_QK_K) == 0 && enable_q4k_dot &&
                       ((op->flags & BN_GPU_OP_FLAG_MATVEC_Q8K) ||
                        getenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT") != NULL) &&
                       getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT") == NULL) {
                int reuse_q8k_input =
                    BN_CUDA_Q8K_INPUT_CACHE_MATCH(op->buf_in, op->cols, 1) &&
                    q8k_input_cache_mark_op == i - 1;
                BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                if (!reuse_q8k_input) {
                    if (cuda_ensure_q8_k(ctx, op->cols, 1) != 0)
                        BN_CUDA_EXEC_FAIL("q4k q8k scratch alloc failed");
                    xq = (BnBlockQ8K *)ctx->d_q8_k;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        quantize_q8k_batch_kernel,
                        dim3(op->cols / BN_QK_K, 1, 1), BN_QK_K, 0,
                        xq, in, op->cols, 1);
                }
                BN_CUDA_Q8K_INPUT_CACHE_MARK(op->buf_in, op->cols, 1);
                int q4_threads = 256;
                int warps = q4_threads / 32;
                if (op->cols >= 16384 &&
                    getenv("BN_CUDA_DISABLE_Q4K_Q8K_MATVEC4") == NULL) {
                    int blocks = (op->rows + warps * 4 - 1) / (warps * 4);
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q4k_q8k_dot_matvec4_kernel, blocks,
                        q4_threads, 0,
                        out, (const BnBlockQ4K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                } else {
                    int blocks = (op->rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q4k_q8k_dot_matvec_kernel, blocks,
                        q4_threads, 0,
                        out, (const BnBlockQ4K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                }
            } else if (op->type == BN_GGUF_TENSOR_Q4_K &&
                       (op->cols % BN_QK_K) == 0 && enable_q4k_dot) {
                if (cuda_ensure_q8_1(ctx, op->cols) != 0)
                    BN_CUDA_EXEC_FAIL("q4k q8_1 scratch alloc failed");
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_kernel,
                    (op->cols + 31) / 32, 32, 0, xq, in, op->cols);
                if (enable_q4k_4warp &&
                    (op->cols <= 8192 ||
                     (op->rows == 1536 && op->cols == 8960 &&
                      getenv("BN_CUDA_DISABLE_Q4K_4WARP_1536_8960") == NULL) ||
                     (op->rows == 2560 && op->cols == 9728 &&
                      getenv("BN_CUDA_DISABLE_Q4K_4WARP_2560_9728") == NULL))) {
                    int fuse_resid_norm =
                        getenv("BN_CUDA_ENABLE_Q4K_OUT_RESID_RMSNORM_FUSE") != NULL &&
                        next && next->op_code == BN_GPU_CODE_RESIDUAL_RMSNORM &&
                        next->buf_in == BN_GPU_VALUE_X &&
                        next->buf_aux == op->buf_out &&
                        out_offset == 0 && bias == NULL &&
                        bias_idx < 0 && fused_copy_idx < 0 &&
                        op->rows == (int)next->p[0];
                    BnCudaBuffer *nw = fuse_resid_norm
                        ? (BnCudaBuffer *)next->W_buf : NULL;
                    float *resid = fuse_resid_norm
                        ? cuda_act(ctx, next->buf_in) : NULL;
                    float *norm_out = fuse_resid_norm
                        ? cuda_act(ctx, next->buf_out) : NULL;
                    if (fuse_resid_norm && nw && nw->data && resid &&
                        norm_out) {
                        BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                            q4k_dot_matvec_4warp_residual_kernel,
                            op->rows, 128, 0,
                            resid, (const BnBlockQ4K *)w->data, xq,
                            resid, op->rows, op->cols);
                        int norm_threads = op->rows >= 2048 ? 512 : threads;
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                            rmsnorm_kernel, 1, norm_threads,
                            (size_t)norm_threads * sizeof(float),
                            norm_out, resid, (const float *)nw->data,
                            op->rows, cuda_u32_to_f32(next->p[1]));
                        i++;
                    } else {
                        BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                            q4k_dot_matvec_4warp_kernel,
                            op->rows, 128, 0,
                            out, (const BnBlockQ4K *)w->data, xq, bias,
                            op->rows, op->cols, out_offset);
                    }
                } else {
                    int q4_threads = 256;
                    int warps = q4_threads / 32;
                    int blocks = (op->rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q4k_dot_matvec_kernel, blocks,
                        q4_threads, 0,
                        out, (const BnBlockQ4K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                }
            } else if (op->type == BN_GGUF_TENSOR_Q5_K &&
                       (op->cols % BN_QK_K) == 0 && enable_q5k_dot) {
                if (cuda_ensure_q8_1(ctx, op->cols) != 0)
                    BN_CUDA_EXEC_FAIL("q5k q8_1 scratch alloc failed");
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_kernel,
                    (op->cols + 31) / 32, 32, 0, xq, in, op->cols);
                if (op->cols <= 8192 &&
                    getenv("BN_CUDA_DISABLE_Q5K_4WARP") == NULL) {
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q5k_dot_matvec_4warp_kernel, op->rows, 128, 0,
                        out, (const BnBlockQ5K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                } else {
                    int q5_threads = 256;
                    int warps = q5_threads / 32;
                    int blocks = (op->rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_matvec,
                        q5k_dot_matvec_kernel, blocks, q5_threads, 0,
                        out, (const BnBlockQ5K *)w->data, xq, bias,
                        op->rows, op->cols, out_offset);
                }
            } else if ((enable_q8_preq_all ||
                        (q8_small_ssm_matvec &&
                         getenv("BN_CUDA_DISABLE_Q8_0_SSM_PREQ") == NULL) ||
                        (is_logits_op && q8_preq_logits_default)) &&
                       op->type == BN_GGUF_TENSOR_Q8_0 &&
                       (op->cols & 31) == 0) {
                if (cuda_ensure_q8_1(ctx, op->cols) != 0)
                    BN_CUDA_EXEC_FAIL("q8 preq scratch alloc failed");
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_kernel,
                    (op->cols + 31) / 32, 32, 0, xq, in, op->cols);
                int q8_threads = 256;
                int blocks = (op->rows + (q8_threads / 32) - 1) /
                             (q8_threads / 32);
                BN_CUDA_LAUNCH(ctx, q8_0_matvec_preq_warp8_kernel,
                    blocks, q8_threads, 0,
                    out, (const BnBlockQ8_0 *)w->data, xq, bias, op->rows,
                    op->cols, out_offset);
            } else if (!disable_q8_warp && op->type == BN_GGUF_TENSOR_Q8_0 &&
                       (op->cols & 31) == 0 && !bias) {
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
                        out_offset, 0);
                } else {
                    BN_CUDA_LAUNCH(ctx, matvec_kernel,
                        dim3(op->rows, 1, 1), threads,
                        (size_t)threads * sizeof(float),
                        out, w->data, in, bias, op->rows, op->cols, op->type,
                        out_offset, 0);
                }
            }
            if (direct_kv_f16) {
                BN_CUDA_LAUNCH(ctx, kv_store_vector_kernel,
                    (op->rows + threads - 1) / threads, threads, 0,
                    direct_kv_dst, out, op->rows, direct_kv_offset, 1);
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
                BN_CUDA_EXEC_FAIL("matvec split invalid args");
            if (split1 > split0 && (!out2 || split1 > total_rows))
                BN_CUDA_EXEC_FAIL("matvec split invalid third output");
            int q4k_qkv_mixed_slow =
                op->type == BN_GGUF_TENSOR_Q4_K &&
                getenv("BN_CUDA_ENABLE_Q4K_QKV_MIXED_FUSE") == NULL;
            if (!disable_qkv_mixed_fuse && !q4k_qkv_mixed_slow &&
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
                    if (qkv_fuse_key_cache && !freq)
                        BN_CUDA_EXEC_FAIL("qkv mixed missing rope freq");
                    int k_pair_opt = enable_qkv_kpair_opt;
                    int k_grid_rows = k_pair_opt
                        ? (total_rows - split0 + 1) / 2
                        : total_rows - split0;
                    const BnCudaBlockQ8_1 *xq_mixed = NULL;
                    if (getenv("BN_CUDA_ENABLE_Q8_MIXED_PREQ") &&
                        (op->type == BN_GGUF_TENSOR_Q8_0 ||
                         ops[i + 5].type == BN_GGUF_TENSOR_Q8_0) &&
                        (cols & 31) == 0) {
                        if (cuda_ensure_q8_1(ctx, cols) != 0)
                            BN_CUDA_EXEC_FAIL("qkv mixed q8 scratch alloc failed");
                        BnCudaBlockQ8_1 *xq =
                            (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_kernel,
                            (cols + 31) / 32, 32, 0, xq, in, cols);
                        xq_mixed = xq;
                    }
                    int q_unused = 0;
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
                (cols % BN_QK_K) == 0 && split1 != 1 && enable_q4k_dot &&
                (op->flags & BN_GPU_OP_FLAG_MATVEC_Q8K) &&
                getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT") == NULL) {
                int reuse_q8k_input =
                    BN_CUDA_Q8K_INPUT_CACHE_MATCH(op->buf_in, cols, 1) &&
                    q8k_input_cache_mark_op == i - 1;
                BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                if (!reuse_q8k_input) {
                    if (cuda_ensure_q8_k(ctx, cols, 1) != 0)
                        BN_CUDA_EXEC_FAIL("q4k q8k split scratch alloc failed");
                    xq = (BnBlockQ8K *)ctx->d_q8_k;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        quantize_q8k_batch_kernel,
                        dim3(cols / BN_QK_K, 1, 1), BN_QK_K, 0,
                        xq, in, cols, 1);
                }
                BN_CUDA_Q8K_INPUT_CACHE_MARK(op->buf_in, cols, 1);
                int q4_threads = 256;
                int warps = q4_threads / 32;
                int blocks = (total_rows + warps - 1) / warps;
                BN_CUDA_LAUNCH(ctx, q4k_q8k_dot_matvec_split_kernel,
                    blocks, q4_threads, 0,
                    out0, out1, out2, (const BnBlockQ4K *)w->data, xq,
                    bias0, total_rows, cols, split0, split1,
                    (size_t)op->p[6], (size_t)op->p[7]);
            } else if (op->type == BN_GGUF_TENSOR_Q4_K &&
                (cols % BN_QK_K) == 0 && split1 != 1 && enable_q4k_dot) {
                if (cuda_ensure_q8_1(ctx, cols) != 0)
                    BN_CUDA_EXEC_FAIL("q4k split q8 scratch alloc failed");
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_kernel,
                    (cols + 31) / 32, 32, 0, xq, in, cols);
                int q4_threads = 256;
                int warps = q4_threads / 32;
                int fused_k_rope_cache = 0;
                int fused_q_rope_idx = -1;
                int k_bias_idx = -1;
                if (getenv("BN_CUDA_ENABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE") != NULL &&
                    next && i + 4 < n_ops &&
                    getenv("BN_CUDA_DISABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE") == NULL &&
                    split1 <= split0 &&
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
                    (int)ops[i + 3].p[1] == (int)ops[i + 3].p[3] &&
                    ops[i + 4].op_code == BN_GPU_CODE_COPY &&
                    ops[i + 4].buf_in == op->buf_aux &&
                    (int)ops[i + 4].p[0] == 0 &&
                    (int)ops[i + 4].p[2] == total_rows - split0) {
                    BnCudaBuffer *kbw = (BnCudaBuffer *)ops[i + 2].W_buf;
                    float *freq = cuda_act(ctx, BN_GPU_VALUE_ROPE_FREQ);
                    void *key_cache = cuda_act(ctx, ops[i + 4].buf_out);
                    if (kbw && kbw->data && freq && key_cache) {
                        k_bias_idx = i + 2;
                        int n_kv_heads = (int)ops[i + 3].p[0];
                        int head_size = (int)ops[i + 3].p[1];
                        int rope_dims = (int)ops[i + 3].p[3];
                        int q_rope_idx = i + 8;
                        if (q_rope_idx < n_ops &&
                            getenv("BN_CUDA_DISABLE_Q4K_SPLIT_QK_ROPE_CACHE_FUSE") == NULL &&
                            ops[q_rope_idx].op_code == BN_GPU_CODE_ROPE &&
                            ops[q_rope_idx].buf_in == op->buf_out &&
                            (int)ops[q_rope_idx].p[1] == head_size &&
                            (int)ops[q_rope_idx].p[2] == (int)ops[i + 3].p[2] &&
                            (int)ops[q_rope_idx].p[3] == rope_dims &&
                            (int)ops[q_rope_idx].p[1] ==
                                (int)ops[q_rope_idx].p[3] &&
                            (int)ops[q_rope_idx].p[0] *
                                (int)ops[q_rope_idx].p[1] == split0) {
                            int n_heads = (int)ops[q_rope_idx].p[0];
                            int units = (n_heads + n_kv_heads) *
                                        (rope_dims / 2);
                            int blocks = (units + warps - 1) / warps;
                            BN_CUDA_LAUNCH(ctx,
                                q4k_dot_matvec_split_qk_rope_cache_kernel,
                                blocks, q4_threads, 0,
                                out0, key_cache,
                                (const BnBlockQ4K *)w->data, xq, bias0,
                                (const float *)kbw->data, freq, cols,
                                split0, n_heads, n_kv_heads, head_size,
                                (int)ops[i + 3].p[2], rope_dims,
                                (size_t)ops[i + 4].p[1], ctx->kv_f16);
                            fused_q_rope_idx = q_rope_idx;
                        } else {
                            int units = split0 +
                                        n_kv_heads * (rope_dims / 2);
                            int blocks = (units + warps - 1) / warps;
                            BN_CUDA_LAUNCH(ctx,
                                q4k_dot_matvec_split_k_rope_cache_kernel,
                                blocks, q4_threads, 0,
                                out0, key_cache,
                                (const BnBlockQ4K *)w->data, xq, bias0,
                                (const float *)kbw->data, freq, total_rows,
                                cols, split0, n_kv_heads, head_size,
                                (int)ops[i + 3].p[2], rope_dims,
                                (size_t)ops[i + 4].p[1], ctx->kv_f16);
                        }
                        fused_k_rope_cache = 1;
                    }
                }
                if (fused_k_rope_cache) {
                    skip_ops[k_bias_idx] = 1;
                    skip_ops[i + 3] = 1;
                    skip_ops[i + 4] = 1;
                    if (fused_q_rope_idx >= 0)
                        skip_ops[fused_q_rope_idx] = 1;
                } else if (cols == 2048 &&
                           getenv("BN_CUDA_DISABLE_Q4K_SPLIT_4WARP_2048") == NULL) {
                    BN_CUDA_LAUNCH(ctx, q4k_dot_matvec_split_4warp_kernel,
                        total_rows, 128, 0,
                        out0, out1, out2, (const BnBlockQ4K *)w->data, xq,
                        bias0, total_rows, cols, split0, split1,
                        (size_t)op->p[6], (size_t)op->p[7]);
                } else if (cols == 2560 &&
                           getenv("BN_CUDA_DISABLE_Q4K_SPLIT_5WARP_2560") == NULL) {
                    BN_CUDA_LAUNCH(ctx, q4k_dot_matvec_split_5warp_kernel,
                        total_rows, 160, 0,
                        out0, out1, out2, (const BnBlockQ4K *)w->data, xq,
                        bias0, total_rows, cols, split0, split1,
                        (size_t)op->p[6], (size_t)op->p[7]);
                } else {
                    int blocks = (total_rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH(ctx, q4k_dot_matvec_split_kernel, blocks,
                        q4_threads, 0,
                        out0, out1, out2, (const BnBlockQ4K *)w->data, xq,
                        bias0, total_rows, cols, split0, split1,
                        (size_t)op->p[6], (size_t)op->p[7]);
                }
                int value_rows = 0;
                if (total_rows == 4608 && cols == 2048)
                    value_rows = 512;
                else if (total_rows == 2304 && cols == 2048)
                    value_rows = 256;
                else if (total_rows == 1792 && cols == 1536 &&
                         getenv("BN_CUDA_ENABLE_Q4K_SPLIT_VALUE_FUSE_1792") != NULL)
                    value_rows = 256;
                if (getenv("BN_CUDA_DISABLE_Q4K_SPLIT_VALUE_FUSE") == NULL &&
                    value_rows > 0) {
                    int v_idx = -1;
                    for (int si = i + 1; si < n_ops && si <= i + 6; si++) {
                        const BnGPUOp *scan = &ops[si];
                        if (scan->op_code == BN_GPU_CODE_MATVEC &&
                            scan->buf_in == op->buf_in &&
                            scan->cols == cols &&
                            scan->rows == value_rows &&
                            (scan->type == BN_GGUF_TENSOR_Q4_K ||
                             scan->type == BN_GGUF_TENSOR_Q6_K) &&
                            scan->W_buf &&
                            cuda_act(ctx, scan->buf_out)) {
                            v_idx = si;
                            break;
                        }
                        if (cuda_op_mentions_buf(scan, op->buf_in))
                            break;
                    }
                    if (v_idx >= 0) {
                        const BnGPUOp *vop = &ops[v_idx];
                        BnCudaBuffer *vw = (BnCudaBuffer *)vop->W_buf;
                        float *vout = cuda_act(ctx, vop->buf_out);
                        const float *vbias = NULL;
                        size_t voffset = (size_t)vop->p[5];
                        int v_bias_idx = -1;
                        int v_copy_idx = -1;
                        if (v_idx + 2 < n_ops &&
                            ops[v_idx + 1].op_code == BN_GPU_CODE_BIAS_ADD &&
                            ops[v_idx + 1].buf_in == vop->buf_out &&
                            (int)ops[v_idx + 1].p[0] == vop->rows &&
                            ops[v_idx + 2].op_code == BN_GPU_CODE_COPY &&
                            ops[v_idx + 2].buf_in == vop->buf_out &&
                            (int)ops[v_idx + 2].p[0] == (int)vop->p[5] &&
                            (int)ops[v_idx + 2].p[2] == vop->rows) {
                            BnCudaBuffer *bw =
                                (BnCudaBuffer *)ops[v_idx + 1].W_buf;
                            float *copy_out =
                                cuda_act(ctx, ops[v_idx + 2].buf_out);
                            if (bw && bw->data && copy_out) {
                                vbias = (const float *)bw->data;
                                vout = copy_out;
                                voffset = (size_t)ops[v_idx + 2].p[1];
                                v_bias_idx = v_idx + 1;
                                v_copy_idx = v_idx + 2;
                            }
                        }
                        if (vw && vw->data && vw->cols == cols &&
                            vw->rows >= vop->rows) {
                            if (vop->type == BN_GGUF_TENSOR_Q4_K) {
                                BN_CUDA_LAUNCH(ctx,
                                    q4k_dot_matvec_4warp_kernel, vop->rows,
                                    128, 0, vout,
                                    (const BnBlockQ4K *)vw->data, xq,
                                    vbias, vop->rows, vop->cols, voffset);
                            } else {
                                BN_CUDA_LAUNCH(ctx,
                                    q6k_dot_matvec_mmvq_kernel, vop->rows,
                                    128, 0, vout,
                                    (const BnBlockQ6K *)vw->data, xq,
                                    vbias, vop->rows, vop->cols, voffset);
                            }
                            skip_ops[v_idx] = 1;
                            if (v_bias_idx >= 0)
                                skip_ops[v_bias_idx] = 1;
                            if (v_copy_idx >= 0)
                                skip_ops[v_copy_idx] = 1;
                        }
                    }
                }
            } else if (op->type == BN_GGUF_TENSOR_Q5_K &&
                (cols % BN_QK_K) == 0 && split1 != 1 && enable_q5k_dot) {
                if (cuda_ensure_q8_1(ctx, cols) != 0)
                    BN_CUDA_EXEC_FAIL("q5k split q8 scratch alloc failed");
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_kernel,
                    (cols + 31) / 32, 32, 0, xq, in, cols);
                int q5_threads = 256;
                int warps = q5_threads / 32;
                if (cols <= 8192 &&
                    getenv("BN_CUDA_ENABLE_Q5K_SPLIT_4WARP") != NULL &&
                    getenv("BN_CUDA_DISABLE_Q5K_4WARP") == NULL) {
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        q5k_dot_matvec_split_4warp_kernel, total_rows, 128,
                        0, out0, out1, out2,
                        (const BnBlockQ5K *)w->data, xq, bias0,
                        total_rows, cols, split0, split1,
                        (size_t)op->p[6], (size_t)op->p[7]);
                } else {
                    int blocks = (total_rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        q5k_dot_matvec_split_kernel, blocks, q5_threads, 0,
                        out0, out1, out2, (const BnBlockQ5K *)w->data, xq,
                        bias0, total_rows, cols, split0, split1,
                        (size_t)op->p[6], (size_t)op->p[7]);
                }
            } else if (op->type == BN_GGUF_TENSOR_Q8_0 &&
                       (cols & 31) == 0 && split1 != 1) {
                int q8_threads = 256;
                int warps = q8_threads / 32;
                int blocks = (total_rows + warps - 1) / warps;
                if (cuda_use_q8_0_preq_split()) {
                    if (cuda_ensure_q8_1(ctx, cols) != 0)
                        BN_CUDA_EXEC_FAIL("q8 split q8 scratch alloc failed");
                    BnCudaBlockQ8_1 *xq =
                        (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_kernel,
                        (cols + 31) / 32, 32, 0, xq, in, cols);
                    BN_CUDA_LAUNCH(ctx, q8_0_matvec_split_preq_warp8_kernel,
                        blocks, q8_threads, 0,
                        out0, out1, out2, (const BnBlockQ8_0 *)w->data, xq,
                        bias0, total_rows, cols, split0, split1,
                        (size_t)op->p[6], (size_t)op->p[7]);
                } else {
                    BN_CUDA_LAUNCH(ctx, q8_0_matvec_split_warp8_kernel,
                        blocks, q8_threads, 0,
                        out0, out1, out2, (const BnBlockQ8_0 *)w->data, in,
                        bias0, total_rows, cols, split0, split1,
                        (size_t)op->p[6], (size_t)op->p[7]);
                }
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
            int stable_decode_gateup = graph_exec;
            int exact_silu =
                (op->flags & BN_GPU_OP_FLAG_EXACT_SILU) != 0;
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
                    up_rows, cols, exact_silu);
            } else if (op->type == BN_GGUF_TENSOR_Q4_K &&
                       (cols % BN_QK_K) == 0 && enable_q4k_dot &&
                       getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT") == NULL &&
                       ((op->flags & BN_GPU_OP_FLAG_MATVEC_Q8K) ||
                        getenv("BN_CUDA_DISABLE_Q4K_GATEUP_Q8_1_FAST") != NULL)) {
                const BnGPUOp *prev_q8k = (i > 0) ? &ops[i - 1] : NULL;
                int prev_routed_same_input =
                    prev_q8k &&
                    prev_q8k->op_code == BN_GPU_CODE_MOE_ROUTED_FFN &&
                    prev_q8k->buf_in == op->buf_in &&
                    prev_q8k->cols == cols;
                int reuse_q8k_input =
                    BN_CUDA_Q8K_INPUT_CACHE_MATCH(op->buf_in, cols, 1) &&
                    prev_routed_same_input &&
                    (q8k_input_cache_mark_op == i - 1 ||
                     (q8k_input_cache_mark_op == i - 2 && i >= 2 &&
                      ops[i - 2].op_code == BN_GPU_CODE_MOE_ROUTE_TOPK));
                BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                if (!reuse_q8k_input) {
                    if (cuda_ensure_q8_k(ctx, cols, 1) != 0) return -1;
                    xq = (BnBlockQ8K *)ctx->d_q8_k;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8k_batch_kernel,
                        dim3(cols / BN_QK_K, 1, 1), BN_QK_K, 0,
                        xq, in, cols, 1);
                    BN_CUDA_Q8K_INPUT_CACHE_MARK(op->buf_in, cols, 1);
                }
                int q4_gateup_threads = 256;
                int warps = q4_gateup_threads / 32;
                if (cols <= 4096 &&
                    getenv("BN_CUDA_DISABLE_Q4K_GATEUP_QWARP4") == NULL) {
                    int blocks = ((gate_rows + 3) / 4 + warps - 1) / warps;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_gateup,
                        q4k_q8k_dot_fused_gateup_silu_qwarp4_kernel,
                        blocks, q4_gateup_threads, 0,
                        out, (const BnBlockQ4K *)w->data, xq, gate_rows,
                        up_rows, cols, exact_silu);
                } else {
                    int blocks = (gate_rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_gateup,
                        q4k_q8k_dot_fused_gateup_silu_kernel,
                        blocks, q4_gateup_threads, 0,
                        out, (const BnBlockQ4K *)w->data, xq, gate_rows,
                        up_rows, cols, exact_silu);
                }
            } else if (op->type == BN_GGUF_TENSOR_Q4_K &&
                       (cols % BN_QK_K) == 0 && enable_q4k_dot) {
                if (cuda_ensure_q8_1(ctx, cols) != 0) return -1;
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_kernel,
                    (cols + 31) / 32, 32, 0, xq, in, cols);
                if (enable_q4k_4warp && cols == 2560 &&
                    getenv("BN_CUDA_DISABLE_Q4K_GATEUP_5WARP_2560") == NULL) {
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_gateup,
                        q4k_dot_fused_gateup_silu_5warp_kernel,
                        gate_rows, 160, 0,
                        out, (const BnBlockQ4K *)w->data, xq, gate_rows,
                        up_rows, cols, exact_silu);
                } else if (enable_q4k_4warp && cols <= 5120 &&
                    getenv("BN_CUDA_DISABLE_Q4K_GATEUP_2WARP") == NULL) {
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_gateup,
                        q4k_dot_fused_gateup_silu_2warp_kernel,
                        gate_rows, 64, 0,
                        out, (const BnBlockQ4K *)w->data, xq, gate_rows,
                        up_rows, cols, exact_silu);
                } else if (enable_q4k_4warp && cols <= 8192) {
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_gateup,
                        q4k_dot_fused_gateup_silu_4warp_kernel,
                        gate_rows, 128, 0,
                        out, (const BnBlockQ4K *)w->data, xq, gate_rows,
                        up_rows, cols, exact_silu);
                } else {
                    int q4_gateup_threads = 256;
                    int warps = q4_gateup_threads / 32;
                    int blocks = (gate_rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_gateup,
                        q4k_dot_fused_gateup_silu_kernel,
                        blocks, q4_gateup_threads, 0,
                        out, (const BnBlockQ4K *)w->data, xq, gate_rows,
                        up_rows, cols, exact_silu);
                }
            } else if (op->type == BN_GGUF_TENSOR_Q5_K &&
                       (cols % BN_QK_K) == 0 && enable_q5k_dot) {
                if (cuda_ensure_q8_1(ctx, cols) != 0) return -1;
                BnCudaBlockQ8_1 *xq =
                    (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_kernel,
                    (cols + 31) / 32, 32, 0, xq, in, cols);
                if (getenv("BN_CUDA_DISABLE_Q5K_GATEUP_2WARP") == NULL) {
                    int q5_gateup_threads = 256;
                    int groups = q5_gateup_threads / 64;
                    int blocks = (gate_rows + groups - 1) / groups;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_gateup,
                        q5k_dot_fused_gateup_silu_2warp_kernel, blocks,
                        q5_gateup_threads, 0,
                        out, (const BnBlockQ5K *)w->data, xq, gate_rows,
                        up_rows, cols);
                } else {
                    int q5_gateup_threads = 256;
                    int warps = q5_gateup_threads / 32;
                    int blocks = (gate_rows + warps - 1) / warps;
                    BN_CUDA_LAUNCH_STABLE(ctx, stable_decode_gateup,
                        q5k_dot_fused_gateup_silu_kernel, blocks,
                        q5_gateup_threads, 0,
                        out, (const BnBlockQ5K *)w->data, xq, gate_rows,
                        up_rows, cols);
                }
            } else {
                BN_CUDA_LAUNCH_STATIC(ctx, fused_gateup_silu_kernel, gate_rows,
                    threads, (size_t)threads * sizeof(float) * 2,
                    out, w->data, in, gate_rows, up_rows, cols, op->type);
            }
            break;
        }
        case BN_GPU_CODE_MOE_ROUTE_TOPK: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *in = cuda_act(ctx, op->buf_in);
            float *route = cuda_act(ctx, op->buf_out);
            float *logits = cuda_act(ctx, op->buf_aux);
            int n_experts = (int)op->p[0];
            int k = (int)op->p[1];
            float expert_weights_scale = cuda_u32_to_f32(op->p[2]);
            int dim = op->cols;
            int route_diff2 = n_experts == 2 && k == 2 && w && w->rows == 1;
            if (!w || !w->data || !in || !route || !logits ||
                n_experts <= 0 || k <= 0 || dim <= 0 ||
                w->type != BN_GGUF_TENSOR_F32 ||
                (!route_diff2 && w->rows < n_experts) || w->cols < dim ||
                ctx->act_sizes[op->buf_aux] <
                    (size_t)n_experts * sizeof(float) ||
                ctx->act_sizes[op->buf_out] <
                    (size_t)(2 * k) * sizeof(float))
                return -1;
            int next_moe_all2_q4q6 =
                next &&
                next->op_code == BN_GPU_CODE_MOE_ROUTED_FFN &&
                next->type == BN_GGUF_TENSOR_Q4_K &&
                next->cols == dim &&
                (int)next->p[1] == 2 &&
                (int)next->p[2] == 2 &&
                (int)next->p[3] == BN_GGUF_TENSOR_Q6_K;
            int moe_all2_route_q8k_default_enabled =
                (dim % BN_QK_K) == 0 &&
                next_moe_all2_q4q6 &&
                !bn_gpu_policy_all2_q4q6_moe_q8k_default_disabled() &&
                !bn_gpu_policy_all2_q4q6_route_q8k_default_disabled() &&
                getenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREQUANT") == NULL;
            if (route_diff2) {
                int route_prequant_q8_1 =
                    (dim % 32) == 0 &&
                    next_moe_all2_q4q6 &&
                    (next->flags & BN_GPU_OP_FLAG_EXACT_SILU) == 0 &&
                    (bn_gpu_policy_all2_q4q6_route_q8_1_prequant_enabled() ||
                     getenv("BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREQUANT") != NULL) &&
                    getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT") == NULL &&
                    getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT_ALL2") == NULL &&
                    getenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREQUANT") == NULL;
                int route_prequant_q8k =
                    !route_prequant_q8_1 &&
                    moe_all2_route_q8k_default_enabled;
                if (route_prequant_q8_1) {
                    if (cuda_ensure_q8_1(ctx, dim) != 0)
                        BN_CUDA_EXEC_FAIL("moe route q8_1 prequant scratch alloc failed");
                    BnCudaBlockQ8_1 *xq = (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        moe_route_diff2_quantize_q8_1_kernel,
                        dim / 32 + 1, 32, 0, route, xq,
                        (const float *)w->data, in, dim,
                        expert_weights_scale);
                } else if (route_prequant_q8k) {
                    if (cuda_ensure_q8_k(ctx, dim, 1) != 0)
                        BN_CUDA_EXEC_FAIL("moe route q8k prequant scratch alloc failed");
                    BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        moe_route_diff2_quantize_q8k_kernel,
                        dim / BN_QK_K + 1, BN_QK_K, 0, route, xq,
                        (const float *)w->data, in, dim,
                        expert_weights_scale);
                    BN_CUDA_Q8K_INPUT_CACHE_MARK(op->buf_in, dim, 1);
                } else {
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        moe_route_diff2_warp_kernel, 1, 32, 0, route,
                        (const float *)w->data, in, dim,
                        expert_weights_scale);
                }
            } else if (n_experts == 2 && k == 2 && w->rows >= 2 &&
                       (dim % 16) == 0 &&
                       moe_all2_route_q8k_default_enabled &&
                       !bn_gpu_policy_all2_q4q6_moe_exact_gpu_route_disabled()) {
                if (cuda_ensure_q8_k(ctx, dim, 1) != 0)
                    BN_CUDA_EXEC_FAIL("moe exact route q8k prequant scratch alloc failed");
                BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                    moe_route_all2_avx2_quantize_q8k_kernel,
                    dim / BN_QK_K + 1, BN_QK_K, 0, route, xq,
                    (const float *)w->data, in, dim,
                    expert_weights_scale,
                    (op->flags & BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM) == 0);
                BN_CUDA_Q8K_INPUT_CACHE_MARK(op->buf_in, dim, 1);
            } else if (n_experts <= 256 &&
                       !(op->flags & BN_GPU_OP_FLAG_MOE_ROUTE_BLOCK) &&
                       getenv("BN_CUDA_ENABLE_MOE_ROUTER_FUSED_TOPK") != NULL &&
                       !getenv("BN_CUDA_DISABLE_MOE_ROUTER_FUSED_TOPK")) {
                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                    moe_route_fused_warp_topk_kernel, 1, 256, 0, route,
                    (const float *)w->data, in, n_experts, dim, k,
                    (op->flags & BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM) == 0,
                    expert_weights_scale);
            } else if ((op->flags & BN_GPU_OP_FLAG_MOE_ROUTE_BLOCK) ||
                       getenv("BN_CUDA_DISABLE_MOE_ROUTER_WARP")) {
                BN_CUDA_LAUNCH(ctx, moe_router_logits_kernel, n_experts,
                    threads, (size_t)threads * sizeof(float),
                    logits, (const float *)w->data, in, n_experts, dim);
                BN_CUDA_LAUNCH(ctx, moe_route_topk_kernel, 1, 1, 0,
                    route, logits, n_experts, k,
                    (op->flags & BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM) == 0,
                    expert_weights_scale);
            } else {
                int warps = threads / 32;
                if (dim >= 2048 &&
                    getenv("BN_CUDA_DISABLE_MOE_ROUTER_4WARP") == NULL) {
                    int groups = threads / 128;
                    int blocks = (n_experts + groups - 1) / groups;
                    BN_CUDA_LAUNCH(ctx, moe_router_logits_4warp_kernel,
                        blocks, threads, 0, logits, (const float *)w->data,
                        in, n_experts, dim);
                } else if (dim >= 2048 &&
                    getenv("BN_CUDA_DISABLE_MOE_ROUTER_2WARP") == NULL) {
                    int groups = threads / 64;
                    int blocks = (n_experts + groups - 1) / groups;
                    BN_CUDA_LAUNCH(ctx, moe_router_logits_2warp_kernel,
                        blocks, threads, 0, logits, (const float *)w->data,
                        in, n_experts, dim);
                } else {
                    int blocks = (n_experts + warps - 1) / warps;
                    BN_CUDA_LAUNCH(ctx, moe_router_logits_warp_kernel, blocks,
                        threads, 0, logits, (const float *)w->data, in,
                        n_experts, dim);
                }
                if (n_experts <= 256 &&
                    !getenv("BN_CUDA_DISABLE_MOE_ROUTER_WARP_TOPK")) {
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        moe_route_topk_warp_kernel, 1, 32, 0, route, logits,
                        n_experts, k,
                        (op->flags & BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM) == 0,
                        expert_weights_scale);
                } else {
                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                        moe_route_topk_kernel, 1, 1, 0, route, logits,
                        n_experts, k,
                        (op->flags & BN_GPU_OP_FLAG_MOE_ROUTE_NO_NORM) == 0,
                        expert_weights_scale);
                }
            }
            break;
        }
        case BN_GPU_CODE_MOE_ROUTED_FFN: {
            BnCudaBuffer *gate = (BnCudaBuffer *)op->W_buf;
            BnCudaBuffer *up = (BnCudaBuffer *)op->W_buf2;
            BnCudaBuffer *down = (BnCudaBuffer *)op->W_buf3;
            float *in = cuda_act(ctx, op->buf_in);
            float *route = cuda_act(ctx, op->buf_aux);
            int mid_buf = (int)op->p[4];
            float *mid = cuda_act(ctx, mid_buf);
            float *out = cuda_act(ctx, op->buf_out);
            int hidden = (int)op->p[0];
            int n_experts = (int)op->p[1];
            int k = (int)op->p[2];
            int down_type = (int)op->p[3];
            int moe_layer = (int)op->p[5];
            int dim = op->cols;
            int routed_q4 = op->type == BN_GGUF_TENSOR_Q4_K &&
                            gate && gate->type == BN_GGUF_TENSOR_Q4_K &&
                            up && up->type == BN_GGUF_TENSOR_Q4_K &&
                            down &&
                            (down_type == BN_GGUF_TENSOR_Q6_K ||
                             down_type == BN_GGUF_TENSOR_Q4_K) &&
                            down->type == down_type;
            int routed_q8 = op->type == BN_GGUF_TENSOR_Q8_0 &&
                            gate && gate->type == BN_GGUF_TENSOR_Q8_0 &&
                            up && up->type == BN_GGUF_TENSOR_Q8_0 &&
                            down && down_type == BN_GGUF_TENSOR_Q8_0 &&
                            down->type == BN_GGUF_TENSOR_Q8_0;
            if (!gate || !gate->data || !up || !up->data ||
                !down || !down->data || !in || !route || !mid || !out ||
                (!routed_q4 && !routed_q8) ||
                dim <= 0 || hidden <= 0 || n_experts <= 0 || k <= 0 ||
                (dim % 32) != 0 || (hidden % 32) != 0 ||
                gate->rows < hidden * n_experts || gate->cols < dim ||
                up->rows < hidden * n_experts || up->cols < dim ||
                down->rows < dim * n_experts || down->cols < hidden ||
                ctx->act_sizes[op->buf_aux] <
                    (size_t)(2 * k) * sizeof(float) ||
                ctx->act_sizes[mid_buf] <
                    (size_t)k * (size_t)hidden * sizeof(float) ||
                ctx->act_sizes[op->buf_out] < (size_t)dim * sizeof(float))
                return -1;
            {
                int route_threads = 256;
                int warps = route_threads / 32;
                int gateup_tasks = hidden * k;
                int gateup_blocks = (gateup_tasks + warps - 1) / warps;
                int down_blocks = (dim + warps - 1) / warps;
                if (routed_q8) {
                    if (!getenv("BN_CUDA_DISABLE_Q8_MOE_Q8X")) {
                        int q8_scratch_elems =
                            dim > hidden * k ? dim : hidden * k;
                        if (cuda_ensure_q8_1(ctx, q8_scratch_elems) != 0)
                            return -1;
                        BnCudaBlockQ8_1 *xq =
                            (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_kernel,
                            (dim + 31) / 32, 32, 0, xq, in, dim);
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                            moe_q8_0_gateup_routed_mid_q8_1_kernel,
                            gateup_blocks, route_threads, 0,
                            mid, (const BnBlockQ8_0 *)gate->data,
                            (const BnBlockQ8_0 *)up->data, xq, route,
                            hidden, dim, n_experts, k);
                        BnCudaBlockQ8_1 *mid_q =
                            (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8_1_batch_kernel,
                            dim3(hidden / 32, k, 1), 32, 0, mid_q, mid,
                            hidden, k);
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                            moe_q8_0_down_routed_q8_1_accum_kernel,
                            down_blocks, route_threads, 0,
                            out, (const BnBlockQ8_0 *)down->data, mid_q,
                            route, dim, hidden, n_experts, k);
                    } else {
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                            moe_q8_0_gateup_routed_mid_kernel, gateup_blocks,
                            route_threads, 0,
                            mid, (const BnBlockQ8_0 *)gate->data,
                            (const BnBlockQ8_0 *)up->data, in, route,
                            hidden, dim, n_experts, k);
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                            moe_q8_0_down_routed_accum_kernel, down_blocks,
                            route_threads, 0,
                            out, (const BnBlockQ8_0 *)down->data, mid,
                            route, dim, hidden, n_experts, k);
                    }
                } else {
                    /* Q8_1 is faster for small routed inputs; large hidden
                       alone does not amortize Q8_K quantization overhead. */
                    int moe_all2_q4q6 =
                        n_experts == 2 && k == 2 &&
                        down_type == BN_GGUF_TENSOR_Q6_K &&
                        hidden >= 4096 && dim <= 2048;
                    int moe_all2_q4_or_q6 =
                        n_experts == 2 && k == 2 &&
                        (down_type == BN_GGUF_TENSOR_Q6_K ||
                         down_type == BN_GGUF_TENSOR_Q4_K) &&
                        hidden >= 4096 && dim <= 2048;
                    int moe_all2_fast_moe_ffn =
                        bn_gpu_policy_all2_q4q6_moe_fast_ffn_enabled();
                    int moe_all2_fast_enabled =
                        getenv("BN_CUDA_DISABLE_MOE_ALL2_FAST") == NULL &&
                        (!moe_all2_q4_or_q6 ||
                         bn_gpu_policy_all2_q4q6_moe_all2_fast_enabled());
                    int exact_silu =
                        (op->flags & BN_GPU_OP_FLAG_EXACT_SILU) != 0;
                    int use_cublas_all2_decode =
                        moe_all2_q4q6 &&
                        gate->f16_data && up->f16_data && down->f16_data &&
                        bn_gpu_policy_all2_q4q6_moe_cublas_decode_enabled() &&
                        getenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE") == NULL;
                    if (use_cublas_all2_decode) {
                        if (cuda_moe_cublas_all2_decode(
                                ctx, out, gate, up, down, in, route, dim,
                                hidden, exact_silu) == 0) {
                            break;
                        }
                        if (getenv("BN_CUDA_DEBUG_MOE_CUBLAS_DECODE")) {
                            fprintf(stderr,
                                    "[bn:gpu:cuda] all2 cublas moe decode "
                                    "failed; falling back\n");
                        }
                    }
                    const BnGPUOp *prev = (i > 0) ? &ops[i - 1] : NULL;
                    int moe_all2_route_q8_1_prequant_enabled =
                        !exact_silu &&
                        (bn_gpu_policy_all2_q4q6_route_q8_1_prequant_enabled() ||
                         getenv("BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREQUANT") != NULL) &&
                        getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT") == NULL &&
                        getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT_ALL2") == NULL &&
                        getenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREQUANT") == NULL;
                    int moe_all2_route_q8k_default_enabled =
                        !bn_gpu_policy_all2_q4q6_moe_q8k_default_disabled() &&
                        !bn_gpu_policy_all2_q4q6_route_q8k_default_disabled() &&
                        getenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREQUANT") == NULL;
                    int moe_all2_x_q8k_prequantized =
                        moe_all2_q4q6 &&
                        prev &&
                        prev->op_code == BN_GPU_CODE_MOE_ROUTE_TOPK &&
                        prev->cols == dim &&
                        prev->W_buf &&
                        (((BnCudaBuffer *)prev->W_buf)->rows == 1 ||
                         (((BnCudaBuffer *)prev->W_buf)->rows >= 2 &&
                          (dim % 16) == 0 &&
                          !bn_gpu_policy_all2_q4q6_moe_exact_gpu_route_disabled())) &&
                        !moe_all2_route_q8_1_prequant_enabled &&
                        moe_all2_route_q8k_default_enabled;
                    int moe_all2_x_q8_1_prequantized =
                        moe_all2_q4q6 &&
                        prev &&
                        prev->op_code == BN_GPU_CODE_MOE_ROUTE_TOPK &&
                        prev->cols == dim &&
                        prev->W_buf &&
                        ((BnCudaBuffer *)prev->W_buf)->rows == 1 &&
                        moe_all2_route_q8_1_prequant_enabled;
                    int use_moe_all2_q8k_default =
                        moe_all2_q4q6 &&
                        !bn_gpu_policy_all2_q4q6_moe_q8k_default_disabled();
                    int moe_all2_fast_q8k_gateup =
                        moe_all2_q4_or_q6 &&
                        bn_gpu_policy_all2_q4q6_fast_q8k_gateup_enabled() &&
                        !bn_gpu_policy_all2_q4q6_fast_q8k_gateup_disabled();
                    int use_q4k_q8k_dot =
                        ((use_moe_all2_q8k_default ||
                          moe_all2_fast_q8k_gateup ||
                          (moe_all2_q4q6 &&
                           getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT_ALL2") != NULL) ||
                          (hidden > 2048 && dim > 2048) ||
                          getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT") != NULL) &&
                         getenv("BN_CUDA_DISABLE_MOE_Q4K_Q8K_DOT") == NULL);
                    int profile_moe_internal =
                        profile && getenv("BN_CUDA_PROFILE_MOE_INTERNAL");
                    cudaEvent_t moe_ev_start = NULL;
                    cudaEvent_t moe_ev_stop = NULL;
                    if (profile_moe_internal) {
                        cudaEventCreate(&moe_ev_start);
                        cudaEventCreate(&moe_ev_stop);
                        cudaEventRecord(moe_ev_start, ctx->exec_stream);
                    }
                    if (use_q4k_q8k_dot) {
                        if (cuda_ensure_q8_k(ctx, dim, 1) != 0) return -1;
                        BnBlockQ8K *xq = (BnBlockQ8K *)ctx->d_q8_k;
                        if (!moe_all2_x_q8k_prequantized) {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8k_batch_kernel,
                                dim3(dim / BN_QK_K, 1, 1), BN_QK_K, 0,
                                xq, in, dim, 1);
                            BN_CUDA_Q8K_INPUT_CACHE_MARK(op->buf_in, dim, 1);
                        }
                        if (moe_all2_x_q8k_prequantized &&
                            moe_all2_fast_enabled &&
                            getenv("BN_CUDA_DISABLE_MOE_Q4K_ALL2_FIXED") == NULL &&
                            getenv("BN_CUDA_DISABLE_MOE_Q4K_GATEUP_4ROW") == NULL) {
                            int gateup4_tasks = (gateup_tasks + 3) / 4;
                            int gateup4_blocks =
                                (gateup4_tasks + warps - 1) / warps;
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q4k_gateup_all2_mid_q8k_4row_kernel,
                                gateup4_blocks, route_threads, 0,
                                mid, (const BnBlockQ4K *)gate->data,
                                (const BnBlockQ4K *)up->data, xq, route,
                                hidden, dim, exact_silu);
                        } else if (getenv("BN_CUDA_DISABLE_MOE_Q4K_GATEUP_4ROW")) {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q4k_gateup_routed_mid_q8k_kernel,
                                gateup_blocks, route_threads, 0,
                                mid, (const BnBlockQ4K *)gate->data,
                                (const BnBlockQ4K *)up->data, xq, route, hidden,
                                dim, n_experts, k, exact_silu);
                        } else {
                            int gateup4_tasks = (gateup_tasks + 3) / 4;
                            int gateup4_blocks =
                                (gateup4_tasks + warps - 1) / warps;
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q4k_gateup_routed_mid_q8k_4row_kernel,
                                gateup4_blocks, route_threads, 0,
                                mid, (const BnBlockQ4K *)gate->data,
                                (const BnBlockQ4K *)up->data, xq, route, hidden,
                                dim, n_experts, k, exact_silu);
                        }
                    } else {
                        if (cuda_ensure_q8_1(ctx, dim) != 0) return -1;
                        BnCudaBlockQ8_1 *xq =
                            (BnCudaBlockQ8_1 *)ctx->d_q8_1;
                        if (!moe_all2_x_q8_1_prequantized) {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                quantize_q8_1_kernel, (dim + 31) / 32, 32,
                                0, xq, in, dim);
                        }
                        if (n_experts == 2 && k == 2 &&
                            moe_all2_fast_enabled) {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q4k_gateup_all2_mid_kernel,
                                gateup_blocks, route_threads, 0,
                                mid, (const BnBlockQ4K *)gate->data,
                                (const BnBlockQ4K *)up->data, xq, route,
                                hidden, dim);
                        } else {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q4k_gateup_routed_mid_kernel,
                                gateup_blocks, route_threads, 0,
                                mid, (const BnBlockQ4K *)gate->data,
                                (const BnBlockQ4K *)up->data, xq, route,
                                hidden, dim, n_experts, k);
                        }
                    }
                    if (profile_moe_internal) {
                        cudaEventRecord(moe_ev_stop, ctx->exec_stream);
                        cudaEventSynchronize(moe_ev_stop);
                        float ms = 0.0f;
                        cudaEventElapsedTime(&ms, moe_ev_start, moe_ev_stop);
                        profile_ops[BN_CUDA_PROFILE_MOE_GATEUP]++;
                        profile_ms[BN_CUDA_PROFILE_MOE_GATEUP] += (double)ms;
                        cudaEventRecord(moe_ev_start, ctx->exec_stream);
                    }
                    if (down_type == BN_GGUF_TENSOR_Q6_K) {
                        int use_q6_float_down =
                            getenv("BN_CUDA_DISABLE_Q6K_FLOAT_MOE_DOWN") == NULL;
                        int moe_all2_pair_down_enabled =
                            bn_gpu_policy_all2_q4q6_q6k_pair_down_enabled();
                        int moe_all2_pair_down_f32_layer =
                            moe_all2_q4q6 &&
                            down->f32_data &&
                            moe_all2_pair_down_enabled &&
                            !bn_gpu_policy_all2_q4q6_q6k_pair_down_f32_layers_disabled() &&
                            bn_gpu_policy_all2_q4q6_q6k_pair_down_f32_layer_selected(moe_layer);
                        int moe_all2_q6k_ordered_down =
                            moe_all2_q4q6 &&
                            bn_gpu_policy_all2_q4q6_q6k_ordered_down_enabled() &&
                            !bn_gpu_policy_all2_q4q6_q6k_ordered_down_disabled();
                        int moe_all2_allow_f32_down_default =
                            !moe_all2_fast_moe_ffn ||
                            bn_gpu_policy_all2_q4q6_q6k_f32_down_default_enabled();
                        int moe_all2_f32_down_default =
                            moe_all2_q4q6 &&
                            down->f32_data &&
                            !moe_all2_q6k_ordered_down &&
                            !moe_all2_pair_down_enabled &&
                            moe_all2_allow_f32_down_default &&
                            !bn_gpu_policy_all2_q4q6_q6k_f32_down_default_disabled();
                        int moe_all2_f32_down =
                            (moe_all2_f32_down_default ||
                             moe_all2_pair_down_f32_layer) &&
                            !bn_gpu_policy_all2_q4q6_q6k_f32_all2_down_disabled();
                        int moe_all2_disable_pair_down =
                            moe_all2_q4q6 &&
                            moe_all2_fast_moe_ffn &&
                            !moe_all2_pair_down_enabled;
                        int moe_all2_disable_f32_cache =
                            moe_all2_q4q6 &&
                            moe_all2_fast_moe_ffn &&
                            !bn_gpu_policy_all2_q4q6_q6k_f32_cache_enabled();
                        int use_q6_pair_down =
                            !moe_all2_f32_down_default &&
                            !moe_all2_pair_down_f32_layer &&
                            !moe_all2_disable_pair_down &&
                            getenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN") == NULL;
                        int prefer_q6_f32_down =
                            down->f32_data && hidden >= 4096 &&
                            (!moe_all2_q4q6 || moe_all2_f32_down) &&
                            getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL;
                        int moe_all2_float_4row_down =
                            moe_all2_q4q6 &&
                            moe_all2_fast_moe_ffn &&
                            !bn_gpu_policy_all2_q4q6_q6k_float_4row_down_default_disabled() &&
                            !bn_gpu_policy_all2_q4q6_q6k_float_4row_down_disabled();
                        int moe_all2_exact_down_layer_selected =
                            bn_gpu_policy_all2_q4q6_q6k_f32_exact_4row_down_layer_selected(
                                moe_layer);
                        int moe_all2_f32_exact_4row_down =
                            moe_all2_q4q6 &&
                            moe_all2_fast_moe_ffn &&
                            down->f32_data &&
                            moe_all2_exact_down_layer_selected &&
                            !bn_gpu_policy_all2_q4q6_q6k_f32_exact_4row_down_default_disabled() &&
                            !bn_gpu_policy_all2_q4q6_q6k_f32_exact_4row_down_disabled();
                        if (moe_all2_f32_exact_4row_down) {
                            int row4_blocks = (dim + 3) / 4;
                            float skip_eps =
                                bn_gpu_policy_all2_q4q6_down_skip_eps_or_default(
                                    1.0e-7f);
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q6k_down_all2_f32_cache_exact_4row_kernel,
                                row4_blocks, dim3(256, 4, 1), 0,
                                out, (const float *)down->f32_data, mid,
                                route, dim, hidden, skip_eps);
                        } else if (prefer_q6_f32_down &&
                            n_experts == 2 && k == 2 &&
                            getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_PAIR2") == NULL) {
                            if (getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_4ROW")) {
                                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                    moe_q6k_down_all2_f32_cache_pair2_sum_kernel,
                                    (dim + 1) / 2, dim3(32, 4, 1), 0,
                                    out, (const float *)down->f32_data, mid,
                                    route, dim, hidden);
                            } else {
                                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                    moe_q6k_down_all2_f32_cache_4row_sum_kernel,
                                    (dim + 3) / 4, dim3(32, 8, 1), 0,
                                    out, (const float *)down->f32_data, mid,
                                    route, dim, hidden);
                            }
                        } else if (prefer_q6_f32_down &&
                            n_experts == 2 && k == 2 &&
                            moe_all2_fast_enabled) {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q6k_down_all2_f32_cache_warp_kernel,
                                down_blocks, route_threads, 0,
                                out, (const float *)down->f32_data, mid,
                                route, dim, hidden);
                        } else if (prefer_q6_f32_down) {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q6k_down_routed_f32_cache_warp_kernel,
                                down_blocks, route_threads, 0,
                                out, (const float *)down->f32_data, mid,
                                route, dim, hidden, n_experts, k);
                        } else if (use_q6_pair_down) {
                            int use_q6k_all2_accum =
                                moe_all2_q4q6 &&
                                getenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM") != NULL &&
                                getenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_ACCUM") == NULL;
                            int use_q6k_pair4_sum =
                                moe_all2_q4q6 &&
                                getenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR4_SUM") == NULL;
                            int use_q6k_k8_4row_sum =
                                !moe_all2_q4q6 &&
                                k <= 8 && hidden <= 1024 &&
                                getenv("BN_CUDA_DISABLE_MOE_Q6K_K8_4ROW_SUM") == NULL;
                            int use_q6k_k8_8row_sum =
                                use_q6k_k8_4row_sum &&
                                hidden <= 1024 &&
                                getenv("BN_CUDA_ENABLE_MOE_Q6K_K8_8ROW_SUM") != NULL;
                            if (cuda_ensure_q8_k(ctx, hidden, k) != 0 ||
                                cuda_ensure_prefill(ctx,
                                    (size_t)k * (size_t)dim) != 0)
                                return -1;
                            BnBlockQ8K *mid_q = (BnBlockQ8K *)ctx->d_q8_k;
                            float *pair_out = ctx->d_prefill;
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                quantize_q8k_batch_kernel,
                                dim3(hidden / BN_QK_K, k, 1), BN_QK_K, 0,
                                mid_q, mid, hidden, k);
                            BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE();
                            if (profile_moe_internal) {
                                cudaEventRecord(moe_ev_stop, ctx->exec_stream);
                                cudaEventSynchronize(moe_ev_stop);
                                float ms = 0.0f;
                                cudaEventElapsedTime(&ms, moe_ev_start, moe_ev_stop);
                                profile_ops[BN_CUDA_PROFILE_MOE_MID_QUANT]++;
                                profile_ms[BN_CUDA_PROFILE_MOE_MID_QUANT] += (double)ms;
                                cudaEventRecord(moe_ev_start, ctx->exec_stream);
                            }
                            if (moe_all2_q6k_ordered_down) {
                                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                    moe_q6k_down_routed_q8k_ordered_kernel,
                                    (dim + 127) / 128, 128, 0,
                                    out, (const BnBlockQ6K *)down->data,
                                    mid_q, route, dim, hidden, n_experts, k);
                            } else if (use_q6k_pair4_sum) {
                                if (moe_all2_q4q6 &&
                                    getenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_FIXED") == NULL) {
                                    int pair4_sum_blocks =
                                        ((dim + 15) / 16);
                                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                        moe_q6k_down_all2_q8k_pair4_sum_fixed_kernel,
                                        pair4_sum_blocks, route_threads, 0,
                                        out, (const BnBlockQ6K *)down->data,
                                        mid_q, route, dim, hidden);
                                } else {
                                    int pair4_sum_blocks =
                                        ((dim + 15) / 16);
                                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                        moe_q6k_down_all2_q8k_pair4_sum_kernel,
                                        pair4_sum_blocks, route_threads, 0,
                                        out, (const BnBlockQ6K *)down->data,
                                        mid_q, route, dim, hidden);
                                }
                            } else if (use_q6k_k8_8row_sum) {
                                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                    moe_q6k_down_routed_q8k_k8_8row_sum_kernel,
                                    (dim + 7) / 8, route_threads, 0,
                                    out, (const BnBlockQ6K *)down->data,
                                    mid_q, route, dim, hidden,
                                    n_experts, k);
                            } else if (use_q6k_k8_4row_sum) {
                                int fuse_resid_norm =
                                    getenv("BN_CUDA_DISABLE_MOE_DOWN_RESID_RMSNORM_FUSE") == NULL &&
                                    next &&
                                    next->op_code == BN_GPU_CODE_RESIDUAL_RMSNORM &&
                                    next->buf_in == BN_GPU_VALUE_X &&
                                    next->buf_aux == op->buf_out &&
                                    next->buf_out != op->buf_out &&
                                    dim == (int)next->p[0];
                                BnCudaBuffer *nw = fuse_resid_norm
                                    ? (BnCudaBuffer *)next->W_buf : NULL;
                                float *resid = fuse_resid_norm
                                    ? cuda_act(ctx, next->buf_in) : NULL;
                                float *norm_out = fuse_resid_norm
                                    ? cuda_act(ctx, next->buf_out) : NULL;
                                if (fuse_resid_norm && nw && nw->data &&
                                    resid && norm_out) {
                                    if (dim == 2048 && hidden == 768 &&
                                        k == 8 &&
                                        getenv("BN_CUDA_DISABLE_MOE_Q6K_K8_EXACT_2048_768") == NULL) {
                                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                            moe_q6k_down_routed_q8k_k8_4row_residual_sum_2048_768_kernel,
                                            dim / 4, route_threads, 0,
                                            resid,
                                            (const BnBlockQ6K *)down->data,
                                            mid_q, route, n_experts);
                                    } else {
                                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                            moe_q6k_down_routed_q8k_k8_4row_residual_sum_kernel,
                                            (dim + 3) / 4, route_threads, 0,
                                            resid,
                                            (const BnBlockQ6K *)down->data,
                                            mid_q, route, dim, hidden,
                                            n_experts, k);
                                    }
                                    int norm_threads =
                                        dim >= 2048 ? 512 : threads;
                                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                        rmsnorm_kernel, 1, norm_threads,
                                        (size_t)norm_threads * sizeof(float),
                                        norm_out, resid,
                                        (const float *)nw->data, dim,
                                        cuda_u32_to_f32(next->p[1]));
                                    i++;
                                } else {
                                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                        moe_q6k_down_routed_q8k_k8_4row_sum_kernel,
                                        (dim + 3) / 4, route_threads, 0,
                                        out, (const BnBlockQ6K *)down->data,
                                        mid_q, route, dim, hidden,
                                        n_experts, k);
                                }
                            } else if (use_q6k_all2_accum) {
                                if (getenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM_4ROW") == NULL) {
                                    int all2_blocks =
                                        ((dim + 1) / 2 + warps - 1) / warps;
                                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                        moe_q6k_down_all2_q8k_accum_2row_kernel,
                                        all2_blocks, route_threads, 0,
                                        out, (const BnBlockQ6K *)down->data,
                                        mid_q, route, dim, hidden);
                                } else {
                                    int all2_blocks =
                                        ((dim + 3) / 4 + warps - 1) / warps;
                                    BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                        moe_q6k_down_all2_q8k_accum_4row_kernel,
                                        all2_blocks, route_threads, 0,
                                        out, (const BnBlockQ6K *)down->data,
                                        mid_q, route, dim, hidden);
                                }
                            } else if (getenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN_4ROW") == NULL) {
                                int pair4_blocks =
                                    ((dim + 3) / 4 + warps - 1) / warps;
                                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                    moe_q6k_down_routed_q8k_pair_4row_kernel,
                                    dim3(pair4_blocks, k, 1),
                                    route_threads, 0,
                                    pair_out,
                                    (const BnBlockQ6K *)down->data,
                                    mid_q, route, dim, hidden,
                                    n_experts, k);
                            } else {
                                int pair_blocks =
                                    ((dim + 7) / 8 + warps - 1) / warps;
                                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                    moe_q6k_down_routed_q8k_pair_8row_kernel,
                                    dim3(pair_blocks, k, 1),
                                    route_threads, 0,
                                    pair_out,
                                    (const BnBlockQ6K *)down->data,
                                    mid_q, route, dim, hidden,
                                    n_experts, k);
                            }
                            if (!use_q6k_pair4_sum &&
                                !use_q6k_k8_4row_sum &&
                                !use_q6k_all2_accum &&
                                !moe_all2_q6k_ordered_down) {
                                BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                    moe_sum_pairs_kernel,
                                    (dim + threads - 1) / threads, threads, 0,
                                    out, pair_out, dim, k);
                            }
                        } else if (down->f32_data &&
                            !moe_all2_disable_f32_cache &&
                            getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE") == NULL) {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q6k_down_routed_f32_cache_warp_kernel,
                                down_blocks, route_threads, 0,
                                out, (const float *)down->f32_data, mid,
                                route, dim, hidden, n_experts, k);
                        } else if (down->f16_data &&
                                   getenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F16_CACHE") != NULL &&
                                   getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F16_CACHE") == NULL) {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q6k_down_routed_f16_cache_warp_kernel,
                                down_blocks, route_threads, 0,
                                out, (const __half *)down->f16_data, mid,
                                route, dim, hidden, n_experts, k);
                        } else if (moe_all2_float_4row_down &&
                                   use_q6_float_down) {
                            int row4_blocks = (dim + 3) / 4;
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q6k_down_all2_float_accum_4row_kernel,
                                row4_blocks, dim3(256, 4, 1), 0,
                                out, (const BnBlockQ6K *)down->data, mid,
                                route, dim, hidden);
                        } else if (use_q6_float_down) {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q6k_down_routed_float_accum_row_kernel,
                                dim, route_threads, 0,
                                out, (const BnBlockQ6K *)down->data, mid,
                                route, dim, hidden, n_experts, k);
                        } else {
                            if (cuda_ensure_q8_k(ctx, hidden, k) != 0)
                                return -1;
                            BnBlockQ8K *mid_q = (BnBlockQ8K *)ctx->d_q8_k;
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8k_batch_kernel,
                                dim3(hidden / BN_QK_K, k, 1), BN_QK_K, 0,
                                mid_q, mid, hidden, k);
                            BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE();
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q6k_down_routed_q8k_accum_kernel,
                                down_blocks, route_threads, 0,
                                out, (const BnBlockQ6K *)down->data, mid_q,
                                route, dim, hidden, n_experts, k);
                        }
                    } else if (down->f32_data &&
                               getenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE") == NULL) {
                        if (n_experts == 2 && k == 2 &&
                            moe_all2_fast_enabled) {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q6k_down_all2_f32_cache_4row_sum_kernel,
                                (dim + 3) / 4, dim3(32, 8, 1), 0,
                                out, (const float *)down->f32_data, mid,
                                route, dim, hidden);
                        } else {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q6k_down_routed_f32_cache_warp_kernel,
                                down_blocks, route_threads, 0,
                                out, (const float *)down->f32_data, mid,
                                route, dim, hidden, n_experts, k);
                        }
                    } else if (n_experts == 2 && k == 2 &&
                               hidden >= 4096 &&
                               getenv("BN_CUDA_ENABLE_MOE_Q4K_PAIR_DOWN") != NULL &&
                               getenv("BN_CUDA_DISABLE_MOE_Q4K_PAIR_DOWN") == NULL) {
                        if (cuda_ensure_q8_k(ctx, hidden, k) != 0 ||
                            cuda_ensure_prefill(ctx,
                                (size_t)k * (size_t)dim) != 0)
                            return -1;
                        BnBlockQ8K *mid_q = (BnBlockQ8K *)ctx->d_q8_k;
                        float *pair_out = ctx->d_prefill;
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                            quantize_q8k_batch_kernel,
                            dim3(hidden / BN_QK_K, k, 1), BN_QK_K, 0,
                            mid_q, mid, hidden, k);
                        BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE();
                        int pair_blocks =
                            ((dim + 7) / 8 + warps - 1) / warps;
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                            moe_q4k_down_routed_q8k_pair_8row_kernel,
                            dim3(pair_blocks, k, 1), route_threads, 0,
                            pair_out, (const BnBlockQ4K *)down->data,
                            mid_q, route, dim, hidden, n_experts, k);
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                            moe_sum_pairs_kernel,
                            (dim + threads - 1) / threads, threads, 0,
                            out, pair_out, dim, k);
                    } else {
                        if (cuda_ensure_q8_k(ctx, hidden, k) != 0) return -1;
                        BnBlockQ8K *mid_q = (BnBlockQ8K *)ctx->d_q8_k;
                        BN_CUDA_LAUNCH_STABLE(ctx, graph_exec, quantize_q8k_batch_kernel,
                            dim3(hidden / BN_QK_K, k, 1), BN_QK_K, 0,
                            mid_q, mid, hidden, k);
                        BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE();
                        if (hidden <= 1024 &&
                            getenv("BN_CUDA_ENABLE_MOE_Q4K_DOWN_8ROW") != NULL &&
                            getenv("BN_CUDA_DISABLE_MOE_Q4K_DOWN_8ROW") == NULL) {
                            int down8_blocks =
                                (dim + warps * 8 - 1) / (warps * 8);
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q4k_down_routed_q8k_accum_8row_kernel,
                                down8_blocks, route_threads, 0,
                                out, (const BnBlockQ4K *)down->data, mid_q,
                                route, dim, hidden, n_experts, k);
                        } else {
                            BN_CUDA_LAUNCH_STABLE(ctx, graph_exec,
                                moe_q4k_down_routed_q8k_accum_kernel,
                                down_blocks, route_threads, 0,
                                out, (const BnBlockQ4K *)down->data, mid_q,
                                route, dim, hidden, n_experts, k);
                        }
                    }
                    if (profile_moe_internal) {
                        cudaEventRecord(moe_ev_stop, ctx->exec_stream);
                        cudaEventSynchronize(moe_ev_stop);
                        float ms = 0.0f;
                        cudaEventElapsedTime(&ms, moe_ev_start, moe_ev_stop);
                        profile_ops[BN_CUDA_PROFILE_MOE_DOWN]++;
                        profile_ms[BN_CUDA_PROFILE_MOE_DOWN] += (double)ms;
                        cudaEventDestroy(moe_ev_start);
                        cudaEventDestroy(moe_ev_stop);
                    }
                }
            }
            break;
        }
        case BN_GPU_CODE_RMSNORM: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *in = cuda_act(ctx, op->buf_in);
            float *out = cuda_act(ctx, op->buf_out);
            int n = (int)op->p[0];
            int norm_threads = n >= 2048 ? 512 : threads;
            if (!w || !w->data || !in || !out || n <= 0) return -1;
            BN_CUDA_LAUNCH_STATIC(ctx, rmsnorm_kernel, 1, norm_threads,
                (size_t)norm_threads * sizeof(float),
                out, in, (const float *)w->data, n,
                cuda_u32_to_f32(op->p[1]));
            BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE_BUF(op->buf_out);
            break;
        }
        case BN_GPU_CODE_RESIDUAL_RMSNORM: {
            BnCudaBuffer *w = (BnCudaBuffer *)op->W_buf;
            float *in = cuda_act(ctx, op->buf_in);
            float *aux = cuda_act(ctx, op->buf_aux);
            float *out = cuda_act(ctx, op->buf_out);
            int n = (int)op->p[0];
            int norm_threads = n >= 2048 ? 512 : threads;
            if (!w || !w->data || !in || !aux || !out || n <= 0)
                return -1;
            BN_CUDA_LAUNCH_STATIC(ctx, residual_rmsnorm_kernel, 1,
                norm_threads, (size_t)norm_threads * sizeof(float),
                in, aux, out, (const float *)w->data, n,
                cuda_u32_to_f32(op->p[1]));
            BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE_BUF(op->buf_in);
            BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE_BUF(op->buf_out);
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
            if (runtime_params_ready && next && after_next && flash &&
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
                    (int)after_next->p[3], ctx->kv_f16);
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
            void *out = cuda_act(ctx, op->buf_out);
            int n = (int)op->p[2];
            if (!in || !out || n < 0) return -1;
            if (ctx->kv_f16 &&
                (op->buf_out == BN_GPU_VALUE_KEY_CACHE ||
                 op->buf_out == BN_GPU_VALUE_VALUE_CACHE)) {
                BN_CUDA_LAUNCH(ctx, kv_store_vector_kernel,
                    (n + threads - 1) / threads, threads, 0,
                    out, in + (int)op->p[0], n, (size_t)op->p[1], 1);
            } else {
                BN_CUDA_LAUNCH(ctx, copy_kernel, (n + threads - 1) / threads,
                    threads, 0,
                    in, (float *)out, (int)op->p[0], (int)op->p[1], n);
            }
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
                (int)next->p[0] * (int)next->p[1] == n &&
                !(ctx->kv_f16 &&
                  (after_next->buf_out == BN_GPU_VALUE_KEY_CACHE ||
                   after_next->buf_out == BN_GPU_VALUE_VALUE_CACHE))) {
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
                (int)next->p[2] == n &&
                !(ctx->kv_f16 &&
                  (next->buf_out == BN_GPU_VALUE_KEY_CACHE ||
                   next->buf_out == BN_GPU_VALUE_VALUE_CACHE))) {
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
        case BN_GPU_CODE_WEIGHTED_ADD_SIGMOID: {
            float *in = cuda_act(ctx, op->buf_in);
            float *aux = cuda_act(ctx, op->buf_aux);
            float *gate_in = cuda_act(ctx, BN_GPU_VALUE_XB);
            BnCudaBuffer *gate = (BnCudaBuffer *)op->W_buf;
            int n = (int)op->p[0];
            int dim = (int)op->p[3];
            if (!in || !aux || !gate_in || !gate || !gate->data ||
                n <= 0 || dim <= 0)
                return -1;
            int sigmoid_threads = dim >= 1536 ? 512 : 256;
            if (dim <= 8192 && next &&
                getenv("BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_RMSNORM_FUSE") == NULL &&
                next->op_code == BN_GPU_CODE_RESIDUAL_RMSNORM &&
                next->buf_aux == op->buf_in &&
                (int)next->p[0] == n) {
                float *resid = cuda_act(ctx, next->buf_in);
                float *norm_out = cuda_act(ctx, next->buf_out);
                BnCudaBuffer *norm = (BnCudaBuffer *)next->W_buf;
                if (!resid || !norm_out || !norm || !norm->data)
                    return -1;
                BN_CUDA_LAUNCH(ctx,
                    weighted_add_sigmoid_residual_rmsnorm_kernel,
                    1, sigmoid_threads, 0, resid, in, aux,
                    (const float *)gate->data, gate_in, norm_out,
                    (const float *)norm->data, n, dim, (int)op->p[2],
                    (int)op->p[4], cuda_u32_to_f32(next->p[1]));
                i++;
            } else if (dim <= 8192 && next &&
                getenv("BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_FUSE") == NULL &&
                next->op_code == BN_GPU_CODE_RESIDUAL_ADD &&
                next->buf_aux == op->buf_in &&
                (int)next->p[0] == n) {
                float *resid = cuda_act(ctx, next->buf_in);
                if (!resid) return -1;
                BN_CUDA_LAUNCH(ctx,
                    weighted_add_sigmoid_residual_reduce_kernel,
                    1, sigmoid_threads, 0, resid, in, aux, (const float *)gate->data,
                    gate_in, n, dim, (int)op->p[2], (int)op->p[4]);
                i++;
            } else if (dim <= 8192) {
                BN_CUDA_LAUNCH(ctx, weighted_add_sigmoid_reduce_kernel,
                    1, sigmoid_threads, 0, in, aux, (const float *)gate->data,
                    gate_in, n, dim, (int)op->p[2], (int)op->p[4]);
            } else {
                BN_CUDA_LAUNCH(ctx, weighted_add_sigmoid_kernel,
                    (n + threads - 1) / threads, threads, 0,
                    in, aux, (const float *)gate->data, gate_in, n, dim,
                    (int)op->p[2], (int)op->p[4]);
            }
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
                in, aux, n, op->op_code, (int)op->p[1], op->flags);
            break;
        }
        case BN_GPU_CODE_SILU_ACT:
        case BN_GPU_CODE_RELU2_ACT: {
            float *in = cuda_act(ctx, op->buf_in);
            int n = (int)op->p[0];
            if (!in || n <= 0) return -1;
            BN_CUDA_LAUNCH(ctx, activation_kernel,
                (n + threads - 1) / threads, threads, 0,
                in, n, op->op_code, op->flags);
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
                BN_CUDA_EXEC_FAIL("rope invalid args");
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
                if (!out || !key || !value)
                    BN_CUDA_EXEC_FAIL("rope flash missing buffers");
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
                BN_CUDA_EXEC_FAIL("gqa scores invalid args");
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
                BN_CUDA_EXEC_FAIL("softmax invalid args");
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
                BN_CUDA_EXEC_FAIL("gqa combine invalid args");
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
                BN_CUDA_EXEC_FAIL("flash attention invalid args");
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
                BN_CUDA_EXEC_FAIL("ssm conv invalid args");
            BN_CUDA_LAUNCH(ctx, ssm_conv_silu_kernel,
                (qkv_dim + threads - 1) / threads, threads, 0,
                qkv, conv_state, (const float *)w->data, qkv_dim, kern,
                conv_off);
            break;
        }
        case BN_GPU_CODE_SSM_L2NORM: {
            float *q = cuda_act(ctx, op->buf_in);
            float *k = cuda_act(ctx, op->buf_aux >= 0 ? op->buf_aux
                                                       : op->buf_out);
            int head_dim = (int)op->p[0];
            int q_off = (int)op->p[1];
            int k_off = (int)op->p[2];
            int n_heads = op->rows;
            if (!q || !k || head_dim <= 0 || n_heads <= 0)
                BN_CUDA_EXEC_FAIL("ssm l2norm invalid args");
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
                BN_CUDA_EXEC_FAIL("ssm alpha beta invalid args");
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
                BN_CUDA_EXEC_FAIL("ssm alpha beta split invalid args");
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
                BN_CUDA_EXEC_FAIL("ssm delta invalid args");
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
                BN_CUDA_EXEC_FAIL("ssm gate invalid args");
            BN_CUDA_LAUNCH(ctx, ssm_gate_kernel, num_v_heads, threads,
                8 * sizeof(float), out, z, (const float *)w->data,
                head_v_dim, eps);
            break;
        }
        default:
            BN_CUDA_EXEC_FAIL("unsupported op");
        }
#undef BN_CUDA_EXEC_FAIL
        if (debug_nan) {
            switch (op->op_code) {
            case BN_GPU_CODE_MATVEC:
            case BN_GPU_CODE_MATVEC_SPLIT:
            case BN_GPU_CODE_Q4K_MATVEC_SPLIT:
            case BN_GPU_CODE_Q8_MATVEC_SPLIT:
            case BN_GPU_CODE_Q5K_MATVEC_SPLIT:
            case BN_GPU_CODE_SILU_GATE:
            case BN_GPU_CODE_RELU2_GATE:
            case BN_GPU_CODE_SIGMOID_GATE:
            case BN_GPU_CODE_WEIGHTED_ADD:
            case BN_GPU_CODE_WEIGHTED_ADD_SIGMOID:
            case BN_GPU_CODE_SSM_CONV_SILU:
            case BN_GPU_CODE_SSM_L2NORM:
            case BN_GPU_CODE_SSM_ALPHA_BETA:
            case BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT:
            case BN_GPU_CODE_SSM_DELTA:
            case BN_GPU_CODE_SSM_GATE:
                if (op->buf_out >= 0)
                    cuda_debug_scan_activation(ctx, op->buf_out,
                                               cuda_op_name(op->op_code), i);
                if (op->buf_aux >= 0 &&
                    op->op_code != BN_GPU_CODE_SSM_L2NORM &&
                    op->op_code != BN_GPU_CODE_SSM_GATE)
                    cuda_debug_scan_activation(ctx, op->buf_aux,
                                               cuda_op_name(op->op_code), i);
                if (op->op_code == BN_GPU_CODE_SSM_CONV_SILU ||
                    op->op_code == BN_GPU_CODE_SSM_L2NORM ||
                    op->op_code == BN_GPU_CODE_SILU_GATE ||
                    op->op_code == BN_GPU_CODE_RELU2_GATE ||
                    op->op_code == BN_GPU_CODE_SIGMOID_GATE ||
                    op->op_code == BN_GPU_CODE_WEIGHTED_ADD ||
                    op->op_code == BN_GPU_CODE_WEIGHTED_ADD_SIGMOID)
                    cuda_debug_scan_activation(ctx, op->buf_in,
                                               cuda_op_name(op->op_code), i);
                if (op->op_code == BN_GPU_CODE_SSM_ALPHA_BETA ||
                    op->op_code == BN_GPU_CODE_SSM_ALPHA_BETA_SPLIT) {
                    cuda_debug_scan_activation(ctx, BN_GPU_VALUE_SSM_ALPHA,
                                               cuda_op_name(op->op_code), i);
                    cuda_debug_scan_activation(ctx, BN_GPU_VALUE_SSM_BETA,
                                               cuda_op_name(op->op_code), i);
                }
                break;
            default:
                break;
            }
        }
        exec_launches++;
        if (profile_code >= 0 && profile_code < BN_CUDA_PROFILE_MAX)
            exec_launch_by_code[profile_code]++;
        if (debug_sync_each_op) {
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] execute op[%d] %s sync failed: %s\n",
                        i, cuda_op_name(op->op_code),
                        cudaGetErrorString(err));
                if (profile) {
                    cudaEventDestroy(ev_start);
                    cudaEventDestroy(ev_stop);
                }
                return -1;
            }
        }
        if (!graph_exec) {
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[bn:gpu:cuda] execute op[%d] %s launch failed: %s\n",
                        i, cuda_op_name(op->op_code),
                        cudaGetErrorString(err));
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
            if (getenv("BN_CUDA_PROFILE_SHAPES") &&
                (op->op_code == BN_GPU_CODE_MATVEC ||
                 op->op_code == BN_GPU_CODE_FUSED_GATEUP_SILU ||
                 op->op_code == BN_GPU_CODE_Q4K_MATVEC_SPLIT ||
                 op->op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT ||
                 op->op_code == BN_GPU_CODE_Q8_MATVEC_SPLIT ||
                 op->op_code == BN_GPU_CODE_MATVEC_SPLIT)) {
                int shape_rows = op->rows;
                int shape_cols = op->cols;
                if (op->op_code == BN_GPU_CODE_MATVEC_SPLIT ||
                    op->op_code == BN_GPU_CODE_Q4K_MATVEC_SPLIT ||
                    op->op_code == BN_GPU_CODE_Q5K_MATVEC_SPLIT ||
                    op->op_code == BN_GPU_CODE_Q8_MATVEC_SPLIT) {
                    shape_rows = (int)op->p[0];
                    shape_cols = (int)op->p[1];
                }
                int found = -1;
                for (int si = 0; si < shape_profile_count; si++) {
                    if (shape_profile[si].code == op->op_code &&
                        shape_profile[si].type == op->type &&
                        shape_profile[si].rows == shape_rows &&
                        shape_profile[si].cols == shape_cols) {
                        found = si;
                        break;
                    }
                }
                if (found < 0 &&
                    shape_profile_count <
                        (int)(sizeof(shape_profile) / sizeof(shape_profile[0]))) {
                    found = shape_profile_count++;
                    shape_profile[found].code = op->op_code;
                    shape_profile[found].type = op->type;
                    shape_profile[found].rows = shape_rows;
                    shape_profile[found].cols = shape_cols;
                    shape_profile[found].ops = 0;
                    shape_profile[found].ms = 0.0;
                }
                if (found >= 0) {
                    shape_profile[found].ops++;
                    shape_profile[found].ms += (double)ms;
                }
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
            if (getenv("BN_CUDA_PROFILE_SHAPES")) {
                fprintf(stderr, "[bn:gpu:cuda:profile-shapes]\n");
                for (int si = 0; si < shape_profile_count; si++) {
                    BnCudaShapeProfile *sp = &shape_profile[si];
                    if (!sp->ops) continue;
                    fprintf(stderr,
                            "  %-18s type=%d rows=%d cols=%d ops=%llu "
                            "total_ms=%.3f avg_us=%.2f\n",
                            cuda_op_name(sp->code), sp->type, sp->rows,
                            sp->cols, sp->ops, sp->ms,
                            sp->ms * 1000.0 / (double)sp->ops);
                }
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
#undef BN_CUDA_Q8K_INPUT_CACHE_MATCH
#undef BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE_BUF
#undef BN_CUDA_Q8K_INPUT_CACHE_INVALIDATE
#undef BN_CUDA_Q8K_INPUT_CACHE_MARK
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
    if (cudaEventCreateWithFlags(&ctx->ssm_stream_entry_event,
                                 cudaEventDisableTiming) != cudaSuccess)
        ctx->ssm_stream_entry_event = NULL;
    if (cudaEventCreateWithFlags(&ctx->ssm_stream_exit_event,
                                 cudaEventDisableTiming) != cudaSuccess)
        ctx->ssm_stream_exit_event = NULL;
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
    int ws_mb = cuda_env_int("BN_CUDA_CUBLAS_WORKSPACE_MB", 32);
    if (ws_mb > 0) {
        size_t ws_bytes = (size_t)ws_mb * 1024u * 1024u;
        cudaError_t ws_err = cudaMalloc(&ctx->cublas_workspace, ws_bytes);
        if (ws_err == cudaSuccess) {
            cublasStatus_t ws_st = cublasSetWorkspace(
                ctx->cublas, ctx->cublas_workspace, ws_bytes);
            if (ws_st == CUBLAS_STATUS_SUCCESS) {
                ctx->cublas_workspace_bytes = ws_bytes;
            } else {
                cudaFree(ctx->cublas_workspace);
                ctx->cublas_workspace = NULL;
                ctx->cublas_workspace_bytes = 0;
            }
        } else {
            (void)cudaGetLastError();
            ctx->cublas_workspace = NULL;
            ctx->cublas_workspace_bytes = 0;
        }
    }
    gpu->buffer_create = cuda_buffer_create;
    gpu->buffer_create_quant_only = cuda_buffer_create_quant_only;
    gpu->buffer_create_q6_f32_cache = cuda_buffer_create_q6_f32_cache;
    gpu->buffer_create_f16_cache = cuda_buffer_create_force_f16_cache;
    gpu->buffer_create_stacked2 = cuda_buffer_create_stacked2;
    gpu->buffer_create_stacked3 = cuda_buffer_create_stacked3;
    gpu->buffer_destroy = cuda_buffer_destroy;
    gpu->matvec = cuda_matvec;
    gpu->matmul = cuda_matmul;
    gpu->matmul_batch = cuda_matmul_batch;
    gpu->matvec_batch = cuda_matvec_batch;
    gpu->dense_ffn = cuda_dense_ffn;
    gpu->dense_ffn_batch = cuda_dense_ffn_batch;
    gpu->dense_ffn_batch_norm = cuda_dense_ffn_batch_norm;
    gpu->dense_ffn_batch_norm_resid = cuda_dense_ffn_batch_norm_resid;
    gpu->moe_ffn_batch = cuda_moe_ffn_batch;
    gpu->moe_routed_ffn_batch = cuda_moe_routed_ffn_batch;
    gpu->moe_route_batch = cuda_moe_route_batch;
    gpu->moe_route_routed_ffn_batch = cuda_moe_route_routed_ffn_batch;
    gpu->moe_route_routed_ffn_batch_norm_resid =
        cuda_moe_route_routed_ffn_batch_norm_resid;
    gpu->prefill_attention = cuda_prefill_attention;
    gpu->prefill_attention_wo = cuda_prefill_attention_wo;
    gpu->prefill_qkv_attention_wo = cuda_prefill_qkv_attention_wo;
    gpu->prefill_qkv_attention_wo_norm = cuda_prefill_qkv_attention_wo_norm;
    gpu->prefill_qkv_attention_wo_norm_resid =
        cuda_prefill_qkv_attention_wo_norm_resid;
    gpu->prefill_dense_layer = cuda_prefill_dense_layer;
    gpu->prefill_moe_layer = cuda_prefill_moe_layer;
    gpu->prefill_ssm_layer = cuda_prefill_ssm_layer;
    gpu->init_activations = cuda_init_activations;
    gpu->free_activations = cuda_free_activations;
    gpu->write_activation = cuda_write_activation;
    gpu->read_activation = cuda_read_activation;
    gpu->argmax_activation = cuda_argmax_activation;
    gpu->matvec_argmax_activation = cuda_matvec_argmax_activation;
    gpu->memory_info = cuda_memory_info;
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
                BN_GPU_CAP_Q5K_FUSED_GATEUP_SILU |
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
        if (ctx->d_resident_in) cudaFree(ctx->d_resident_in);
        if (ctx->d_ops) cudaFree(ctx->d_ops);
        if (ctx->d_runtime) cudaFree(ctx->d_runtime);
        if (ctx->d_q8_1) cudaFree(ctx->d_q8_1);
        if (ctx->d_q8_0_f32) cudaFree(ctx->d_q8_0_f32);
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
        if (ctx->h_argmax) cudaFreeHost(ctx->h_argmax);
        cuda_free_activations(ctx);
        if (ctx->cublas) cublasDestroy(ctx->cublas);
        if (ctx->cublas_workspace) cudaFree(ctx->cublas_workspace);
        if (ctx->ssm_stream_entry_event)
            cudaEventDestroy(ctx->ssm_stream_entry_event);
        if (ctx->ssm_stream_exit_event)
            cudaEventDestroy(ctx->ssm_stream_exit_event);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
    }
    free(ctx);
    free(gpu);
}

#endif // BN_ENABLE_CUDA
