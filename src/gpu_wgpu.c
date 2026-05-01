/*
 * gpu_wgpu.c — wgpu-native backend for BnGPUBackend
 *
 * Implements the BnGPUBackend vtable using wgpu-native v27.
 * Compiles WGSL compute shaders (one per quant type), dispatches
 * matvec/matmul on GPU with staging buffer readback.
 */

#ifdef BN_ENABLE_GPU

#include "gpu_wgpu.h"
#include "gpu_backend.h"
#include "model.h"
#include "quant.h"
#include "gguf.h"
#include "webgpu.h"
#include "wgpu.h"

#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* Max tensor type enum value we index into (I2_S = 36, plus margin) */
#define BN_WGPU_MAX_TYPES 40

/* ── Helper: C string -> WGPUStringView ────────────────────────────── */

static WGPUStringView sv(const char *s)
{
    return (WGPUStringView){ .data = s, .length = s ? strlen(s) : 0 };
}

/* ── Internal context ──────────────────────────────────────────────── */

typedef struct {
    WGPUInstance        instance;
    WGPUAdapter         adapter;
    WGPUDevice          device;
    WGPUQueue           queue;
    WGPUComputePipeline pipelines[BN_WGPU_MAX_TYPES];
    WGPUBindGroupLayout layouts[BN_WGPU_MAX_TYPES];

    /* Persistent scratch buffers (reused across dispatches) */
    WGPUBuffer x_buf;           /* input vector/matrix */
    size_t     x_buf_size;      /* current allocation size */
    WGPUBuffer out_buf;         /* output vector/matrix */
    size_t     out_buf_size;
    WGPUBuffer uniform_buf;     /* 16-byte uniforms (always the same size) */
    WGPUBuffer staging_buf;     /* readback staging (MAP_READ | COPY_DST) */
    size_t     staging_buf_size;

    /* Forward-pass shader pipelines (indexed by BN_GPU_SHADER_*) */
    WGPUComputePipeline fwd_pipelines[BN_GPU_SHADER_COUNT];
    WGPUBindGroupLayout fwd_layouts[BN_GPU_SHADER_COUNT];

    /* GPU-resident activation buffers (indexed by BN_GPU_BUF_*) */
    WGPUBuffer act_bufs[BN_GPU_BUF_COUNT];
    size_t     act_sizes[BN_GPU_BUF_COUNT];

    /* Forward-pass staging buffer for logits readback */
    WGPUBuffer fwd_staging;
    size_t     fwd_staging_size;

    /* Uniform ring buffer for per-dispatch parameters */
    WGPUBuffer uniform_ring;
    size_t     uniform_ring_size;

    /* Shader directory path (stored from create) */
    char shader_dir[256];

    /* Device limits (stored at creation for runtime validation) */
    uint64_t max_buffer_size;

    /* Profiling state */
    int gpu_frame;
    int gpu_profile;  /* -1 = uninitialized, 0 = off, 1 = on */

    /* Slab allocator for weight buffers (eliminates per-buffer driver alloc) */
    WGPUBuffer slab_buf;
    size_t     slab_size;
    struct { size_t offset, size; } *slab_free;
    int        slab_free_count;
    int        slab_free_cap;
} BnWgpuCtx;

/* ── GPU buffer handle ─────────────────────────────────────────────── */

typedef struct {
    WGPUBuffer buf;
    size_t     size;
    size_t     offset;       /* byte offset into slab (0 for standalone buffers) */
    int        type;
    int        rows;
    int        cols;
    uint32_t   bias_offset;  /* u32 offset into buffer for fused bias, 0 = none */
    int        is_slab;      /* 1 = suballocated from slab, 0 = standalone */
} BnWgpuBuf;

/* ── Uniform block for compute shaders ─────────────────────────────── */

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t n_tokens;
    uint32_t extra;        /* row-tiling param for large-vocab matvec */
    uint32_t bias_offset;  /* fused bias u32 offset, 0 = none */
    uint32_t _pad[3];
} BnWgpuUniforms;  /* 32 bytes, matches p[8] in BnGPUOp */

/* ── Persistent buffer helper ──────────────────────────────────────── */

static int ensure_scratch(BnWgpuCtx *ctx, size_t x_need, size_t out_need,
                           size_t staging_need)
{
    /* Align all sizes to 4 bytes (WebGPU requirement) */
    x_need = (x_need + 3) & ~(size_t)3;
    out_need = (out_need + 3) & ~(size_t)3;
    staging_need = (staging_need + 3) & ~(size_t)3;

    /* x_buf: Storage | CopyDst */
    if (!ctx->x_buf || ctx->x_buf_size < x_need) {
        if (ctx->x_buf) {
            wgpuBufferDestroy(ctx->x_buf);
            wgpuBufferRelease(ctx->x_buf);
        }
        WGPUBufferDescriptor desc = {
            .label = sv("bn_x_persist"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size  = x_need,
        };
        ctx->x_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!ctx->x_buf) return -1;
        ctx->x_buf_size = x_need;
    }

    /* out_buf: Storage | CopySrc | CopyDst */
    if (!ctx->out_buf || ctx->out_buf_size < out_need) {
        if (ctx->out_buf) {
            wgpuBufferDestroy(ctx->out_buf);
            wgpuBufferRelease(ctx->out_buf);
        }
        WGPUBufferDescriptor desc = {
            .label = sv("bn_out_persist"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc
                     | WGPUBufferUsage_CopyDst,
            .size  = out_need,
        };
        ctx->out_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!ctx->out_buf) return -1;
        ctx->out_buf_size = out_need;
    }

    /* uniform_buf: 256-byte aligned per batch op (for minUniformBufferOffsetAlignment).
     * 16 batch ops × 256 bytes = 4096 bytes. */
    if (!ctx->uniform_buf) {
        size_t uni_size = 16 * 256;  /* BN_WGPU_MAX_BATCH_OPS * 256 */
        WGPUBufferDescriptor desc = {
            .label = sv("bn_uni_persist"),
            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
            .size  = uni_size,
        };
        ctx->uniform_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!ctx->uniform_buf) return -1;
    }

    /* staging_buf: MapRead | CopyDst */
    if (!ctx->staging_buf || ctx->staging_buf_size < staging_need) {
        if (ctx->staging_buf) {
            wgpuBufferDestroy(ctx->staging_buf);
            wgpuBufferRelease(ctx->staging_buf);
        }
        WGPUBufferDescriptor desc = {
            .label = sv("bn_staging_persist"),
            .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
            .size  = staging_need,
        };
        ctx->staging_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!ctx->staging_buf) return -1;
        ctx->staging_buf_size = staging_need;
    }

    return 0;
}

/* ── Error tracking ────────────────────────────────────────────────── */

#include <stdatomic.h>
static atomic_int wgpu_last_error;

static void on_uncaptured_error(WGPUDevice const *device,
                                 WGPUErrorType type,
                                 WGPUStringView message,
                                 void *userdata1, void *userdata2)
{
    (void)device; (void)type; (void)userdata1; (void)userdata2;
    wgpu_last_error = 1;
    if (message.data && message.length > 0)
        fprintf(stderr, "[bn:gpu:wgpu] uncaptured error: %.*s\n",
                (int)message.length, message.data);
}

static void on_device_lost(WGPUDevice const *device,
                            WGPUDeviceLostReason reason,
                            WGPUStringView message,
                            void *userdata1, void *userdata2)
{
    (void)device; (void)reason; (void)userdata1; (void)userdata2;
    if (message.data && message.length > 0)
        fprintf(stderr, "[bn:gpu:wgpu] device lost: %.*s\n",
                (int)message.length, message.data);
}

/* ── Sync callback helpers (v27: two userdata pointers) ────────────── */

typedef struct { WGPUAdapter adapter; int ok; } AdapterReq;

static void on_adapter_request(WGPURequestAdapterStatus status,
                                WGPUAdapter adapter,
                                WGPUStringView message,
                                void *userdata1, void *userdata2)
{
    (void)userdata2;
    AdapterReq *r = (AdapterReq *)userdata1;
    r->adapter = adapter;
    r->ok = (status == WGPURequestAdapterStatus_Success);
    if (!r->ok && message.data && message.length > 0)
        fprintf(stderr, "[bn:gpu:wgpu] adapter request failed: %.*s\n",
                (int)message.length, message.data);
}

typedef struct { WGPUDevice device; int ok; } DeviceReq;

static void on_device_request(WGPURequestDeviceStatus status,
                               WGPUDevice device,
                               WGPUStringView message,
                               void *userdata1, void *userdata2)
{
    (void)userdata2;
    DeviceReq *r = (DeviceReq *)userdata1;
    r->device = device;
    r->ok = (status == WGPURequestDeviceStatus_Success);
    if (!r->ok && message.data && message.length > 0)
        fprintf(stderr, "[bn:gpu:wgpu] device request failed: %.*s\n",
                (int)message.length, message.data);
}

typedef struct { int done; WGPUMapAsyncStatus status; } MapReq;

static void on_buffer_map(WGPUMapAsyncStatus status,
                            WGPUStringView message,
                            void *userdata1, void *userdata2)
{
    (void)message; (void)userdata2;
    MapReq *r = (MapReq *)userdata1;
    r->status = status;
    r->done = 1;
}

/* ── Shader type name mapping ──────────────────────────────────────── */

static const char *shader_name_for_type(int type)
{
    switch (type) {
        case BN_GGUF_TENSOR_F32:     return "f32";
        case BN_GGUF_TENSOR_F16:     return "f16";
        case BN_GGUF_TENSOR_I2_S:    return "i2s";
        case BN_GGUF_TENSOR_TQ1_0:   return "tq1";
        case BN_GGUF_TENSOR_TQ2_0:   return "tq2";
        case BN_GGUF_TENSOR_Q4_0:    return "q4";
        case BN_GGUF_TENSOR_Q4_1:    return "q4_1";
        case BN_GGUF_TENSOR_Q8_0:    return "q8";
        case BN_GGUF_TENSOR_BF16:    return "bf16";
        case BN_GGUF_TENSOR_Q2_K:    return "q2k";
        case BN_GGUF_TENSOR_Q3_K:    return "q3k";
        case BN_GGUF_TENSOR_Q4_K:    return "q4k";
        case BN_GGUF_TENSOR_Q5_K:    return "q5k";
        case BN_GGUF_TENSOR_Q6_K:    return "q6k";
        case BN_GGUF_TENSOR_Q8_K:    return "q8k";
        case BN_GGUF_TENSOR_IQ4_NL:  return "iq4nl";
        case BN_GGUF_TENSOR_IQ4_XS:  return "iq4xs";
        case BN_GGUF_TENSOR_IQ3_XXS: return "iq3xxs";
        case BN_GGUF_TENSOR_IQ3_S:   return "iq3s";
        case BN_GGUF_TENSOR_IQ2_XXS: return "iq2xxs";
        case BN_GGUF_TENSOR_IQ2_XS:  return "iq2xs";
        case BN_GGUF_TENSOR_IQ2_S:   return "iq2s";
        default:                      return NULL;
    }
}

/* All supported quant types for pipeline compilation */
static const int supported_types[] = {
    BN_GGUF_TENSOR_I2_S, BN_GGUF_TENSOR_TQ1_0, BN_GGUF_TENSOR_TQ2_0,
    BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_1, BN_GGUF_TENSOR_Q8_0,
    BN_GGUF_TENSOR_F16, BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_Q2_K, BN_GGUF_TENSOR_Q3_K,
    BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K,
    BN_GGUF_TENSOR_Q8_K, BN_GGUF_TENSOR_IQ4_NL, BN_GGUF_TENSOR_IQ4_XS,
    BN_GGUF_TENSOR_IQ3_XXS, BN_GGUF_TENSOR_IQ3_S, BN_GGUF_TENSOR_IQ2_XXS,
    BN_GGUF_TENSOR_IQ2_XS, BN_GGUF_TENSOR_IQ2_S,
};
#define N_SUPPORTED_TYPES ((int)(sizeof(supported_types) / sizeof(supported_types[0])))

/* ── Shader loading ────────────────────────────────────────────────── */

static char *load_shader_file(const char *shader_dir, const char *type_name)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_matvec.wgsl", shader_dir, type_name);

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);

    char *buf = malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t read = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if ((long)read != len) { free(buf); return NULL; }
    buf[len] = '\0';
    return buf;
}

static char *load_shader_generic(const char *shader_dir, const char *name)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.wgsl", shader_dir, name);

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);

    char *buf = malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t read = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if ((long)read != len) { free(buf); return NULL; }
    buf[len] = '\0';
    return buf;
}

/* ── Pipeline compilation ──────────────────────────────────────────── */

static int compile_pipeline(BnWgpuCtx *ctx, int type, const char *shader_dir)
{
    if (type < 0 || type >= BN_WGPU_MAX_TYPES) return -1;

    const char *name = shader_name_for_type(type);
    if (!name) return -1;

    /* Load shader source from file */
    char *wgsl = NULL;
    if (shader_dir) {
        wgsl = load_shader_file(shader_dir, name);
    }
    if (!wgsl) return -1;  /* No embedded shaders yet */

    size_t wgsl_len = strlen(wgsl);

    /* Clear error flag */
    wgpu_last_error = 0;

    /* Create shader module */
    WGPUShaderSourceWGSL wgsl_desc = {
        .chain = { .sType = WGPUSType_ShaderSourceWGSL },
        .code = { .data = wgsl, .length = wgsl_len },
    };
    WGPUShaderModuleDescriptor sm_desc = {
        .nextInChain = &wgsl_desc.chain,
        .label = sv("bn_shader"),
    };

    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(ctx->device, &sm_desc);
    free(wgsl);

    if (!shader || wgpu_last_error) {
        if (shader) wgpuShaderModuleRelease(shader);
        fprintf(stderr, "[bn:gpu:wgpu] shader compilation failed for %s\n", name);
        return -1;
    }

    /* Create compute pipeline with auto-layout */
    WGPUComputePipelineDescriptor pipe_desc = {
        .label = sv("bn_pipeline"),
        .layout = NULL,  /* auto-layout */
        .compute = {
            .module = shader,
            .entryPoint = sv("main"),
        },
    };

    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(
        ctx->device, &pipe_desc);
    wgpuShaderModuleRelease(shader);

    if (!pipeline || wgpu_last_error) {
        if (pipeline) wgpuComputePipelineRelease(pipeline);
        fprintf(stderr, "[bn:gpu:wgpu] pipeline creation failed for %s\n", name);
        return -1;
    }

    /* Get bind group layout from auto-layout pipeline */
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    if (!layout) {
        wgpuComputePipelineRelease(pipeline);
        return -1;
    }

    ctx->pipelines[type] = pipeline;
    ctx->layouts[type] = layout;
    return 0;
}

/* ── Forward-pass pipeline compilation ─────────────────────────────── */

static int compile_fwd_pipeline(BnWgpuCtx *ctx, int shader_id, const char *name)
{
    if (shader_id < 0 || shader_id >= BN_GPU_SHADER_COUNT) return -1;
    if (!ctx->shader_dir[0]) return -1;

    char *wgsl = load_shader_generic(ctx->shader_dir, name);
    if (!wgsl) {
        fprintf(stderr, "[bn:gpu:wgpu] failed to load shader: %s\n", name);
        return -1;
    }

    wgpu_last_error = 0;

    WGPUShaderSourceWGSL wgsl_desc = {
        .chain = { .sType = WGPUSType_ShaderSourceWGSL },
        .code = { .data = wgsl, .length = strlen(wgsl) },
    };
    WGPUShaderModuleDescriptor sm_desc = {
        .nextInChain = &wgsl_desc.chain,
        .label = sv("bn_fwd_shader"),
    };

    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(ctx->device, &sm_desc);
    free(wgsl);

    if (!shader || wgpu_last_error) {
        if (shader) wgpuShaderModuleRelease(shader);
        fprintf(stderr, "[bn:gpu:wgpu] fwd shader compilation failed: %s\n", name);
        return -1;
    }

    WGPUComputePipelineDescriptor pipe_desc = {
        .label = sv("bn_fwd_pipeline"),
        .layout = NULL,
        .compute = {
            .module = shader,
            .entryPoint = sv("main"),
        },
    };

    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(
        ctx->device, &pipe_desc);
    wgpuShaderModuleRelease(shader);

    if (!pipeline || wgpu_last_error) {
        if (pipeline) wgpuComputePipelineRelease(pipeline);
        fprintf(stderr, "[bn:gpu:wgpu] fwd pipeline creation failed: %s\n", name);
        return -1;
    }

    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    if (!layout) {
        wgpuComputePipelineRelease(pipeline);
        return -1;
    }

    ctx->fwd_pipelines[shader_id] = pipeline;
    ctx->fwd_layouts[shader_id] = layout;
    return 0;
}

/* ── Slab allocator (eliminates per-buffer wgpuDeviceCreateBuffer) ── */

static int slab_init(BnWgpuCtx *ctx, size_t size_bytes) {
    if (ctx->slab_buf) return 0;  /* already initialized */
    /* Clamp to device limit */
    if (size_bytes > ctx->max_buffer_size)
        size_bytes = (size_t)ctx->max_buffer_size;
    /* Align to 256 */
    size_bytes &= ~(size_t)255;
    if (size_bytes == 0) return -1;

    WGPUBufferDescriptor desc = {
        .label = sv("bn_slab"),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = size_bytes,
    };
    ctx->slab_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
    if (!ctx->slab_buf) return -1;
    ctx->slab_size = size_bytes;

    /* Initialize free list with one block spanning the whole slab */
    ctx->slab_free_cap = 256;
    ctx->slab_free = calloc((size_t)ctx->slab_free_cap, sizeof(ctx->slab_free[0]));
    if (!ctx->slab_free) { wgpuBufferDestroy(ctx->slab_buf); ctx->slab_buf = NULL; return -1; }
    ctx->slab_free[0].offset = 0;
    ctx->slab_free[0].size = size_bytes;
    ctx->slab_free_count = 1;

    fprintf(stderr, "[bn:gpu:wgpu] slab allocator: %zu MB\n", size_bytes / (1024 * 1024));
    return 0;
}

/* First-fit allocation from slab, 256-byte aligned. Returns offset or (size_t)-1. */
static size_t slab_alloc(BnWgpuCtx *ctx, size_t size) {
    size = (size + 255) & ~(size_t)255;
    for (int i = 0; i < ctx->slab_free_count; i++) {
        if (ctx->slab_free[i].size >= size) {
            size_t off = ctx->slab_free[i].offset;
            if (ctx->slab_free[i].size == size) {
                /* Remove block entirely */
                memmove(&ctx->slab_free[i], &ctx->slab_free[i + 1],
                        (size_t)(ctx->slab_free_count - i - 1) * sizeof(ctx->slab_free[0]));
                ctx->slab_free_count--;
            } else {
                /* Shrink block */
                ctx->slab_free[i].offset += size;
                ctx->slab_free[i].size -= size;
            }
            return off;
        }
    }
    return (size_t)-1;
}

/* Free a slab region, coalesce with neighbors. */
static void slab_free_region(BnWgpuCtx *ctx, size_t offset, size_t size) {
    /* Find insertion point (sorted by offset) */
    int pos = 0;
    while (pos < ctx->slab_free_count && ctx->slab_free[pos].offset < offset) pos++;

    /* Check coalescing with left neighbor */
    int merged = 0;
    if (pos > 0 && ctx->slab_free[pos - 1].offset + ctx->slab_free[pos - 1].size == offset) {
        ctx->slab_free[pos - 1].size += size;
        merged = 1;
        /* Check coalescing with right neighbor too */
        if (pos < ctx->slab_free_count &&
            ctx->slab_free[pos - 1].offset + ctx->slab_free[pos - 1].size == ctx->slab_free[pos].offset) {
            ctx->slab_free[pos - 1].size += ctx->slab_free[pos].size;
            memmove(&ctx->slab_free[pos], &ctx->slab_free[pos + 1],
                    (size_t)(ctx->slab_free_count - pos - 1) * sizeof(ctx->slab_free[0]));
            ctx->slab_free_count--;
        }
    }
    /* Check coalescing with right neighbor only */
    if (!merged && pos < ctx->slab_free_count &&
        offset + size == ctx->slab_free[pos].offset) {
        ctx->slab_free[pos].offset = offset;
        ctx->slab_free[pos].size += size;
        merged = 1;
    }
    /* No coalescing — insert new block */
    if (!merged) {
        if (ctx->slab_free_count >= ctx->slab_free_cap) {
            int new_cap = ctx->slab_free_cap * 2;
            void *tmp = realloc(ctx->slab_free, (size_t)new_cap * sizeof(ctx->slab_free[0]));
            if (!tmp) return;  /* drop the free — leaks slab region but doesn't crash */
            ctx->slab_free = tmp;
            ctx->slab_free_cap = new_cap;
        }
        memmove(&ctx->slab_free[pos + 1], &ctx->slab_free[pos],
                (size_t)(ctx->slab_free_count - pos) * sizeof(ctx->slab_free[0]));
        ctx->slab_free[pos].offset = offset;
        ctx->slab_free[pos].size = size;
        ctx->slab_free_count++;
    }
}

/* ── Vtable: buffer_create ─────────────────────────────────────────── */

/* ── Q4_0 weight repacking for GPU ─────────────────────────────────
 *
 * GGUF Q4_0: 18 bytes/block = [f16 scale][16 nibble bytes]
 *   nibble byte[i]: lo nibble = elem i, hi nibble = elem i+16
 *
 * Repacked GPU layout (all blocks for entire weight matrix):
 *   [f32 scales: n_blocks × 4 bytes][nibbles: n_blocks × 4 u32s]
 *   nibble u32[j]: 8 elements packed, elem e = (word[e/8] >> (e%8)*4) & 0xF
 *   Elements stored in sequential order (0..31) for clean GPU access.
 */
static void *repack_q4_0_for_gpu(BnWgpuCtx *ctx, const void *data, size_t size,
                                   int rows, int cols, size_t *out_size,
                                   const float *bias, int bias_len)
{
    (void)size;
    int n_blocks = (int)((size_t)rows * ((unsigned)cols / 32));
    size_t base_size = (size_t)n_blocks * 4  /* f32 scales */
                     + (size_t)n_blocks * 16; /* nibble data (4 u32s per block) */
    size_t bias_bytes = (bias && bias_len > 0) ? (size_t)bias_len * sizeof(float) : 0;
    size_t repacked_size = base_size + bias_bytes;
    repacked_size = (repacked_size + 3) & ~(size_t)3;

    uint8_t *repacked = calloc(1, repacked_size);
    if (!repacked) return NULL;

    float *scales = (float *)repacked;
    uint8_t *nibbles = repacked + (size_t)n_blocks * 4;
    const uint8_t *src = (const uint8_t *)data;

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * 18;

        /* Extract FP16 scale → f32 */
        uint16_t d_bits = (uint16_t)(block[0] | (block[1] << 8));
        scales[b] = bn_fp16_to_fp32(d_bits);

        /* Reorder nibbles: GGUF has lo=elem[0..15], hi=elem[16..31] in same byte.
         * Repack to sequential element order in u32 words. */
        uint8_t *dst_nib = nibbles + (size_t)b * 16;  /* 16 bytes = 4 u32s for 32 elements */
        const uint8_t *qs = block + 2;

        /* Elements 0-7 → dst_nib[0..3] (u32 word 0) */
        dst_nib[0] = (qs[0] & 0x0F) | ((qs[1] & 0x0F) << 4);
        dst_nib[1] = (qs[2] & 0x0F) | ((qs[3] & 0x0F) << 4);
        dst_nib[2] = (qs[4] & 0x0F) | ((qs[5] & 0x0F) << 4);
        dst_nib[3] = (qs[6] & 0x0F) | ((qs[7] & 0x0F) << 4);
        /* Elements 8-15 → dst_nib[4..7] (u32 word 1) */
        dst_nib[4] = (qs[8] & 0x0F) | ((qs[9] & 0x0F) << 4);
        dst_nib[5] = (qs[10] & 0x0F) | ((qs[11] & 0x0F) << 4);
        dst_nib[6] = (qs[12] & 0x0F) | ((qs[13] & 0x0F) << 4);
        dst_nib[7] = (qs[14] & 0x0F) | ((qs[15] & 0x0F) << 4);
        /* Elements 16-23 → dst_nib[8..11] (u32 word 2) */
        dst_nib[8]  = (qs[0] >> 4) | ((qs[1] >> 4) << 4);
        dst_nib[9]  = (qs[2] >> 4) | ((qs[3] >> 4) << 4);
        dst_nib[10] = (qs[4] >> 4) | ((qs[5] >> 4) << 4);
        dst_nib[11] = (qs[6] >> 4) | ((qs[7] >> 4) << 4);
        /* Elements 24-31 → dst_nib[12..15] (u32 word 3) */
        dst_nib[12] = (qs[8] >> 4) | ((qs[9] >> 4) << 4);
        dst_nib[13] = (qs[10] >> 4) | ((qs[11] >> 4) << 4);
        dst_nib[14] = (qs[12] >> 4) | ((qs[13] >> 4) << 4);
        dst_nib[15] = (qs[14] >> 4) | ((qs[15] >> 4) << 4);
    }

    /* Append fused bias data after nibbles */
    uint32_t fused_bias_offset = 0;
    if (bias && bias_len > 0) {
        fused_bias_offset = (uint32_t)(base_size / sizeof(uint32_t));
        memcpy(repacked + base_size, bias, (size_t)bias_len * sizeof(float));
    }

    /* Create GPU buffer and upload */
    WGPUBufferDescriptor desc = {
        .label = sv("bn_weight_q4_repacked"),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size  = repacked_size,
    };
    WGPUBuffer buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
    if (!buf) { free(repacked); return NULL; }

    wgpuQueueWriteBuffer(ctx->queue, buf, 0, repacked, repacked_size);
    free(repacked);

    BnWgpuBuf *handle = malloc(sizeof(BnWgpuBuf));
    if (!handle) {
        wgpuBufferDestroy(buf);
        wgpuBufferRelease(buf);
        return NULL;
    }
    handle->buf = buf;
    handle->size = repacked_size;
    handle->type = BN_GGUF_TENSOR_Q4_0;
    handle->rows = rows;
    handle->cols = cols;
    handle->bias_offset = fused_bias_offset;
    *out_size = repacked_size;
    return handle;
}

static void *wgpu_buffer_create(void *vctx, const void *data, size_t size,
                                 int type, int rows, int cols)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    if (!ctx || !ctx->device || !data || size == 0)
        return NULL;

    /* Q4_0: repack weights for optimized GPU access */
    if (type == BN_GGUF_TENSOR_Q4_0) {
        size_t repacked_size;
        return repack_q4_0_for_gpu(ctx, data, size, rows, cols, &repacked_size,
                                    NULL, 0);
    }

    /* Align size to 256 bytes (slab alignment) or 4 bytes (standalone) */
    size_t aligned = (size + 255) & ~(size_t)255;

    /* Try slab allocation first (avoids wgpuDeviceCreateBuffer overhead) */
    if (ctx->slab_buf) {
        size_t off = slab_alloc(ctx, aligned);
        if (off != (size_t)-1) {
            wgpuQueueWriteBuffer(ctx->queue, ctx->slab_buf, off, data, size);
            BnWgpuBuf *handle = malloc(sizeof(BnWgpuBuf));
            if (!handle) { slab_free_region(ctx, off, aligned); return NULL; }
            handle->buf = ctx->slab_buf;
            handle->size = aligned;
            handle->offset = off;
            handle->type = type;
            handle->rows = rows;
            handle->cols = cols;
            handle->bias_offset = 0;
            handle->is_slab = 1;
            return handle;
        }
    }

    /* Fallback: standalone buffer with mappedAtCreation (zero-copy on unified memory).
     * Creates buffer already mapped to CPU → memcpy → unmap.
     * On Apple Silicon (Metal): mapped pointer IS GPU shared memory, no staging copy.
     * On discrete GPUs: single DMA at unmap instead of command-queue staging. */
    size_t standalone_aligned = (size + 3) & ~(size_t)3;
    WGPUBufferDescriptor desc = {
        .label = sv("bn_weight"),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size  = standalone_aligned,
        .mappedAtCreation = 1,
    };
    WGPUBuffer buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
    if (!buf) return NULL;
    void *mapped = wgpuBufferGetMappedRange(buf, 0, standalone_aligned);
    if (mapped) {
        memcpy(mapped, data, size);
    }
    wgpuBufferUnmap(buf);

    BnWgpuBuf *handle = malloc(sizeof(BnWgpuBuf));
    if (!handle) { wgpuBufferDestroy(buf); wgpuBufferRelease(buf); return NULL; }
    handle->buf = buf;
    handle->size = standalone_aligned;
    handle->offset = 0;
    handle->type = type;
    handle->rows = rows;
    handle->cols = cols;
    handle->bias_offset = 0;
    handle->is_slab = 0;
    return handle;
}

/* ── Vtable: buffer_create_biased ──────────────────────────────────── */

static void *wgpu_buffer_create_biased(void *vctx, const void *data, size_t size,
                                         int type, int rows, int cols,
                                         const void *bias, size_t bias_size)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    if (!ctx || !ctx->device || !data || size == 0 || !bias || bias_size == 0)
        return NULL;
    /* Only Q4_0 supports fused bias (repacked layout with appended bias) */
    if (type != BN_GGUF_TENSOR_Q4_0) return NULL;

    int bias_len = (int)(bias_size / sizeof(float));
    size_t repacked_size;
    return repack_q4_0_for_gpu(ctx, data, size, rows, cols, &repacked_size,
                                (const float *)bias, bias_len);
}

/* ── Vtable: buffer_destroy ────────────────────────────────────────── */

static void wgpu_buffer_destroy(void *vctx, void *buffer)
{
    if (!buffer) return;
    BnWgpuBuf *h = (BnWgpuBuf *)buffer;
    if (h->is_slab) {
        /* Return region to slab free list (no GPU driver call) */
        BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
        if (ctx) slab_free_region(ctx, h->offset, h->size);
    } else if (h->buf) {
        wgpuBufferDestroy(h->buf);
        wgpuBufferRelease(h->buf);
    }
    free(h);
}

/* ── Vtable: matvec ────────────────────────────────────────────────── */

static int wgpu_matvec(void *vctx, float *out, void *W_buf, const float *x,
                        int rows, int cols, int type)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    BnWgpuBuf *wbuf = (BnWgpuBuf *)W_buf;
    if (!ctx || !wbuf || !x || !out) return -1;
    if (type < 0 || type >= BN_WGPU_MAX_TYPES) return -1;
    if (!ctx->pipelines[type]) return -1;  /* no pipeline -> CPU fallback */

    int rc = -1;
    WGPUBindGroup bind_group = NULL;
    WGPUCommandEncoder encoder = NULL;
    WGPUComputePassEncoder pass = NULL;
    WGPUCommandBuffer cmd = NULL;

    size_t x_size = (size_t)cols * sizeof(float);
    size_t x_aligned = (x_size + 3) & ~(size_t)3;
    size_t out_size = (size_t)rows * sizeof(float);
    size_t out_aligned = (out_size + 3) & ~(size_t)3;

    /* Ensure persistent scratch buffers are large enough */
    if (ensure_scratch(ctx, x_aligned, out_aligned, out_aligned) != 0)
        return -1;

    /* Upload x and uniforms to persistent buffers */
    wgpuQueueWriteBuffer(ctx->queue, ctx->x_buf, 0, x, x_size);
    {
        BnWgpuUniforms uniforms = {
            .rows = (uint32_t)rows,
            .cols = (uint32_t)cols,
            .n_tokens = 1,
            .extra = 0,
        };
        wgpuQueueWriteBuffer(ctx->queue, ctx->uniform_buf, 0,
                              &uniforms, sizeof(uniforms));
    }

    /* Create bind group: 0=W, 1=x, 2=out, 3=uniforms */
    {
        WGPUBindGroupEntry entries[4] = {
            { .binding = 0, .buffer = wbuf->buf, .offset = wbuf->offset, .size = wbuf->size },
            { .binding = 1, .buffer = ctx->x_buf,      .offset = 0, .size = x_aligned },
            { .binding = 2, .buffer = ctx->out_buf,    .offset = 0, .size = out_aligned },
            { .binding = 3, .buffer = ctx->uniform_buf, .offset = 0,
              .size = (sizeof(BnWgpuUniforms) + 15) & ~(size_t)15 },
        };
        WGPUBindGroupDescriptor bg_desc = {
            .label = sv("bn_matvec_bg"),
            .layout = ctx->layouts[type],
            .entryCount = 4,
            .entries = entries,
        };
        bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);
        if (!bind_group) goto cleanup;
    }

    /* Encode compute pass + staging copy in ONE command buffer */
    {
        WGPUCommandEncoderDescriptor enc_desc = { .label = sv("bn_matvec_enc") };
        encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);
        if (!encoder) goto cleanup;

        WGPUComputePassDescriptor pass_desc = { .label = sv("bn_matvec_pass") };
        pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
        if (!pass) goto cleanup;

        wgpuComputePassEncoderSetPipeline(pass, ctx->pipelines[type]);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
        {
            uint32_t tile = (type == BN_GGUF_TENSOR_Q4_0) ? 8u : 32u;
            wgpuComputePassEncoderDispatchWorkgroups(
                pass, ((uint32_t)rows + tile - 1u) / tile, 1, 1);
        }
        wgpuComputePassEncoderEnd(pass);

        /* Copy out_buf → staging_buf in the same command buffer */
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder, ctx->out_buf, 0, ctx->staging_buf, 0, out_size);

        WGPUCommandBufferDescriptor cmd_desc = { .label = sv("bn_matvec_cmd") };
        cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
        if (!cmd) goto cleanup;
    }

    /* Single submit + poll (no separate readback submit) */
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuDevicePoll(ctx->device, 1, NULL);

    /* Map staging buffer and read back */
    {
        MapReq map_req = {0};
        WGPUBufferMapCallbackInfo map_cb = {
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback = on_buffer_map,
            .userdata1 = &map_req,
        };
        wgpuBufferMapAsync(ctx->staging_buf, WGPUMapMode_Read, 0, out_size, map_cb);
        wgpuDevicePoll(ctx->device, 1, NULL);

        if (!map_req.done || map_req.status != WGPUMapAsyncStatus_Success)
            goto cleanup;

        const void *mapped = wgpuBufferGetConstMappedRange(ctx->staging_buf, 0, out_size);
        if (!mapped) {
            wgpuBufferUnmap(ctx->staging_buf);
            goto cleanup;
        }
        memcpy(out, mapped, out_size);
        wgpuBufferUnmap(ctx->staging_buf);
    }

    rc = 0;

cleanup:
    if (cmd) wgpuCommandBufferRelease(cmd);
    if (pass) wgpuComputePassEncoderRelease(pass);
    if (encoder) wgpuCommandEncoderRelease(encoder);
    if (bind_group) wgpuBindGroupRelease(bind_group);
    return rc;
}

/* ── Vtable: matmul ────────────────────────────────────────────────── */

static int wgpu_matmul(void *vctx, float *out, void *W_buf, const float *X,
                        int rows, int cols, int n_tokens, int type)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    BnWgpuBuf *wbuf = (BnWgpuBuf *)W_buf;
    if (!ctx || !wbuf || !X || !out) return -1;
    if (type < 0 || type >= BN_WGPU_MAX_TYPES) return -1;
    if (!ctx->pipelines[type]) return -1;

    int rc = -1;
    WGPUBindGroup bind_group = NULL;
    WGPUCommandEncoder encoder = NULL;
    WGPUComputePassEncoder pass = NULL;
    WGPUCommandBuffer cmd = NULL;

    size_t x_size = (size_t)n_tokens * (size_t)cols * sizeof(float);
    size_t x_aligned = (x_size + 3) & ~(size_t)3;
    size_t out_size = (size_t)n_tokens * (size_t)rows * sizeof(float);
    size_t out_aligned = (out_size + 3) & ~(size_t)3;

    /* Ensure persistent scratch buffers are large enough */
    if (ensure_scratch(ctx, x_aligned, out_aligned, out_aligned) != 0)
        return -1;

    /* Upload x and uniforms to persistent buffers */
    wgpuQueueWriteBuffer(ctx->queue, ctx->x_buf, 0, X, x_size);
    {
        BnWgpuUniforms uniforms = {
            .rows = (uint32_t)rows,
            .cols = (uint32_t)cols,
            .n_tokens = (uint32_t)n_tokens,
            .extra = 0,
        };
        wgpuQueueWriteBuffer(ctx->queue, ctx->uniform_buf, 0,
                              &uniforms, sizeof(uniforms));
    }

    /* Create bind group */
    {
        WGPUBindGroupEntry entries[4] = {
            { .binding = 0, .buffer = wbuf->buf, .offset = wbuf->offset, .size = wbuf->size },
            { .binding = 1, .buffer = ctx->x_buf,      .offset = 0, .size = x_aligned },
            { .binding = 2, .buffer = ctx->out_buf,    .offset = 0, .size = out_aligned },
            { .binding = 3, .buffer = ctx->uniform_buf, .offset = 0,
              .size = (sizeof(BnWgpuUniforms) + 15) & ~(size_t)15 },
        };
        WGPUBindGroupDescriptor bg_desc = {
            .label = sv("bn_matmul_bg"),
            .layout = ctx->layouts[type],
            .entryCount = 4,
            .entries = entries,
        };
        bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);
        if (!bind_group) goto cleanup;
    }

    /* Encode compute pass + staging copy in ONE command buffer */
    {
        WGPUCommandEncoderDescriptor enc_desc = { .label = sv("bn_matmul_enc") };
        encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);
        if (!encoder) goto cleanup;

        WGPUComputePassDescriptor pass_desc = { .label = sv("bn_matmul_pass") };
        pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
        if (!pass) goto cleanup;

        wgpuComputePassEncoderSetPipeline(pass, ctx->pipelines[type]);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
        {
            uint32_t tile = (type == BN_GGUF_TENSOR_Q4_0) ? 8u : 32u;
            wgpuComputePassEncoderDispatchWorkgroups(
                pass, ((uint32_t)rows + tile - 1u) / tile, (uint32_t)n_tokens, 1);
        }
        wgpuComputePassEncoderEnd(pass);

        /* Copy out_buf → staging_buf in the same command buffer */
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder, ctx->out_buf, 0, ctx->staging_buf, 0, out_size);

        WGPUCommandBufferDescriptor cmd_desc = { .label = sv("bn_matmul_cmd") };
        cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
        if (!cmd) goto cleanup;
    }

    /* Single submit + poll */
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuDevicePoll(ctx->device, 1, NULL);

    /* Map staging buffer and read back */
    {
        MapReq map_req = {0};
        WGPUBufferMapCallbackInfo map_cb = {
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback = on_buffer_map,
            .userdata1 = &map_req,
        };
        wgpuBufferMapAsync(ctx->staging_buf, WGPUMapMode_Read, 0, out_size, map_cb);
        wgpuDevicePoll(ctx->device, 1, NULL);

        if (!map_req.done || map_req.status != WGPUMapAsyncStatus_Success)
            goto cleanup;

        const void *mapped = wgpuBufferGetConstMappedRange(ctx->staging_buf, 0, out_size);
        if (!mapped) {
            wgpuBufferUnmap(ctx->staging_buf);
            goto cleanup;
        }
        memcpy(out, mapped, out_size);
        wgpuBufferUnmap(ctx->staging_buf);
    }

    rc = 0;

cleanup:
    if (cmd) wgpuCommandBufferRelease(cmd);
    if (pass) wgpuComputePassEncoderRelease(pass);
    if (encoder) wgpuCommandEncoderRelease(encoder);
    if (bind_group) wgpuBindGroupRelease(bind_group);
    return rc;
}

/* ── Vtable: matvec_batch ──────────────────────────────────────────── */

/* Max ops per batch (stack-allocated bind groups array) */
#define BN_WGPU_MAX_BATCH_OPS 16

static int wgpu_matvec_batch(void *vctx, const BnGPUMatvecOp *ops, int n_ops,
                              const float *x, int x_cols)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    if (!ctx || !ops || n_ops <= 0 || !x) return -1;
    if (n_ops > BN_WGPU_MAX_BATCH_OPS) return -1;

    /* Validate all ops have pipelines and compute total staging size */
    size_t total_staging = 0;
    size_t max_out = 0;
    for (int i = 0; i < n_ops; i++) {
        int t = ops[i].type;
        if (t < 0 || t >= BN_WGPU_MAX_TYPES || !ctx->pipelines[t])
            return -1;
        if (!ops[i].W_buf || !ops[i].out) return -1;
        size_t op_out = (size_t)ops[i].rows * sizeof(float);
        /* Align each op's output region to 4 bytes for copyBufferToBuffer */
        op_out = (op_out + 3) & ~(size_t)3;
        total_staging += op_out;
        if (op_out > max_out) max_out = op_out;
    }

    size_t x_size = (size_t)x_cols * sizeof(float);
    size_t x_aligned = (x_size + 3) & ~(size_t)3;

    /* Ensure persistent buffers: out_buf needs max single-op size,
       staging needs total accumulated size */
    if (ensure_scratch(ctx, x_aligned, max_out, total_staging) != 0)
        return -1;

    /* Upload x once */
    wgpuQueueWriteBuffer(ctx->queue, ctx->x_buf, 0, x, x_size);

    int rc = -1;
    WGPUCommandEncoder encoder = NULL;
    WGPUCommandBuffer cmd = NULL;
    WGPUBindGroup bind_groups[BN_WGPU_MAX_BATCH_OPS] = {0};
    WGPUComputePassEncoder passes[BN_WGPU_MAX_BATCH_OPS] = {0};
    int n_bind_groups = 0;
    int n_passes = 0;

    /* Create command encoder */
    {
        WGPUCommandEncoderDescriptor enc_desc = { .label = sv("bn_batch_enc") };
        encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);
        if (!encoder) goto cleanup;
    }

    /* Encode each op: uniforms → bind group → compute pass → copy to staging */
    size_t staging_offset = 0;
    /* Track staging offsets and sizes for readback */
    size_t op_offsets[BN_WGPU_MAX_BATCH_OPS];
    size_t op_sizes[BN_WGPU_MAX_BATCH_OPS];

    for (int i = 0; i < n_ops; i++) {
        const BnGPUMatvecOp *op = &ops[i];
        BnWgpuBuf *wbuf = (BnWgpuBuf *)op->W_buf;
        size_t out_size = (size_t)op->rows * sizeof(float);
        size_t out_aligned = (out_size + 3) & ~(size_t)3;

        op_offsets[i] = staging_offset;
        op_sizes[i] = out_size;

        /* Write uniforms for this op at its own 256-byte aligned slot */
        size_t uni_offset = (size_t)i * 256;
        BnWgpuUniforms uniforms = {
            .rows = (uint32_t)op->rows,
            .cols = (uint32_t)op->cols,
            .n_tokens = 1,
            .extra = 0,
        };
        wgpuQueueWriteBuffer(ctx->queue, ctx->uniform_buf, uni_offset,
                              &uniforms, sizeof(uniforms));

        /* Create bind group: W varies per op, x/out/uniform are persistent.
         * Each op reads from its own uniform slot at uni_offset. */
        WGPUBindGroupEntry entries[4] = {
            { .binding = 0, .buffer = wbuf->buf, .offset = wbuf->offset, .size = wbuf->size },
            { .binding = 1, .buffer = ctx->x_buf,      .offset = 0, .size = x_aligned },
            { .binding = 2, .buffer = ctx->out_buf,    .offset = 0, .size = out_aligned },
            { .binding = 3, .buffer = ctx->uniform_buf, .offset = uni_offset,
              .size = (sizeof(BnWgpuUniforms) + 15) & ~(size_t)15 },
        };
        WGPUBindGroupDescriptor bg_desc = {
            .label = sv("bn_batch_bg"),
            .layout = ctx->layouts[op->type],
            .entryCount = 4,
            .entries = entries,
        };
        bind_groups[i] = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);
        if (!bind_groups[i]) goto cleanup;
        n_bind_groups = i + 1;

        /* Compute pass: dispatch → end */
        WGPUComputePassDescriptor pass_desc = { .label = sv("bn_batch_pass") };
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(
            encoder, &pass_desc);
        if (!pass) goto cleanup;
        passes[i] = pass;
        n_passes = i + 1;

        wgpuComputePassEncoderSetPipeline(pass, ctx->pipelines[op->type]);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bind_groups[i], 0, NULL);
        {
            uint32_t tile = (op->type == BN_GGUF_TENSOR_Q4_0) ? 8u : 32u;
            wgpuComputePassEncoderDispatchWorkgroups(
                pass, ((uint32_t)op->rows + tile - 1u) / tile, 1, 1);
        }
        wgpuComputePassEncoderEnd(pass);

        /* Copy out_buf[0..out_size] → staging[staging_offset..] */
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder, ctx->out_buf, 0, ctx->staging_buf, staging_offset, out_aligned);

        staging_offset += out_aligned;
    }

    /* Finish and submit ONE command buffer */
    {
        WGPUCommandBufferDescriptor cmd_desc = { .label = sv("bn_batch_cmd") };
        cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
        if (!cmd) goto cleanup;
    }

    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuDevicePoll(ctx->device, 1, NULL);

    /* Map staging and copy each op's output to its host pointer */
    {
        MapReq map_req = {0};
        WGPUBufferMapCallbackInfo map_cb = {
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback = on_buffer_map,
            .userdata1 = &map_req,
        };
        wgpuBufferMapAsync(ctx->staging_buf, WGPUMapMode_Read,
                            0, total_staging, map_cb);
        wgpuDevicePoll(ctx->device, 1, NULL);

        if (!map_req.done || map_req.status != WGPUMapAsyncStatus_Success)
            goto cleanup;

        const char *mapped = (const char *)wgpuBufferGetConstMappedRange(
            ctx->staging_buf, 0, total_staging);
        if (!mapped) {
            wgpuBufferUnmap(ctx->staging_buf);
            goto cleanup;
        }

        for (int i = 0; i < n_ops; i++) {
            memcpy(ops[i].out, mapped + op_offsets[i], op_sizes[i]);
        }

        wgpuBufferUnmap(ctx->staging_buf);
    }

    rc = 0;

cleanup:
    if (cmd) wgpuCommandBufferRelease(cmd);
    for (int i = 0; i < n_passes; i++) {
        if (passes[i]) wgpuComputePassEncoderRelease(passes[i]);
    }
    if (encoder) wgpuCommandEncoderRelease(encoder);
    for (int i = 0; i < n_bind_groups; i++) {
        if (bind_groups[i]) wgpuBindGroupRelease(bind_groups[i]);
    }
    return rc;
}

static void wgpu_free_activations(void *vctx);  /* forward decl for cleanup */

/* ── Vtable: init_activations ──────────────────────────────────────── */

static int wgpu_init_activations(void *vctx, const void *config_ptr)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    const BnConfig *c = (const BnConfig *)config_ptr;
    if (!ctx || !c) return -1;

    /* Compute buffer sizes */
    int n_attn = (c->full_attn_interval > 0)
                     ? c->n_layers / c->full_attn_interval
                     : c->n_layers;
    int q_dim = c->n_heads * c->head_size;
    int xb_size = q_dim > c->dim ? q_dim : c->dim;

    size_t sizes[BN_GPU_BUF_COUNT] = {0};
    sizes[BN_GPU_BUF_X]           = (size_t)c->dim * sizeof(float);
    sizes[BN_GPU_BUF_XB]          = (size_t)xb_size * sizeof(float);
    sizes[BN_GPU_BUF_XB2]         = (size_t)c->dim * sizeof(float);
    sizes[BN_GPU_BUF_Q]           = (size_t)q_dim * sizeof(float);
    {
        // HB/HB2 must accommodate both dense FFN (hidden_dim) and shared MoE expert (moe_intermediate_size)
        int hb_dim = c->hidden_dim;
        if (c->moe_intermediate_size > hb_dim) hb_dim = c->moe_intermediate_size;
        sizes[BN_GPU_BUF_HB]  = (size_t)hb_dim * sizeof(float);
        sizes[BN_GPU_BUF_HB2] = (size_t)hb_dim * sizeof(float);
    }
    sizes[BN_GPU_BUF_KEY_CACHE]   = (size_t)n_attn * c->seq_len * c->kv_dim * sizeof(float);
    sizes[BN_GPU_BUF_VALUE_CACHE] = (size_t)n_attn * c->seq_len * c->kv_dim * sizeof(float);
    sizes[BN_GPU_BUF_ATT]         = (size_t)c->n_heads * c->seq_len * sizeof(float);
    sizes[BN_GPU_BUF_LOGITS]      = (size_t)c->vocab_size * sizeof(float);
    sizes[BN_GPU_BUF_ROPE_FREQ]   = (size_t)(c->head_size / 2) * sizeof(float);
    sizes[BN_GPU_BUF_SCRATCH]     = (size_t)xb_size * sizeof(float);
    {
        size_t qkv_size = (size_t)(q_dim + 2 * c->kv_dim) * sizeof(float);
        size_t gated_q_size = (size_t)(2 * q_dim) * sizeof(float);
        sizes[BN_GPU_BUF_QKV] = qkv_size > gated_q_size ? qkv_size : gated_q_size;
    }

    /* MoE activation buffers (if model has MoE layers) */
    if (c->moe_intermediate_size > 0) {
        sizes[BN_GPU_BUF_MOE_HB]  = (size_t)c->moe_intermediate_size * sizeof(float);
        sizes[BN_GPU_BUF_MOE_HB2] = (size_t)c->moe_intermediate_size * sizeof(float);
        sizes[BN_GPU_BUF_MOE_OUT] = (size_t)c->dim * sizeof(float);
    }

    /* SSM activation buffers (if model has SSM layers) */
    if (c->full_attn_interval > 0 && c->ssm_inner_size > 0) {
        int n_ssm = c->n_layers - n_attn;
        int num_v_heads = c->ssm_time_step_rank;
        int head_k_dim  = c->ssm_state_size;
        int head_v_dim  = c->ssm_inner_size / (num_v_heads > 0 ? num_v_heads : 1);
        int key_dim     = c->ssm_group_count * head_k_dim;
        int value_dim   = c->ssm_inner_size;
        int qkv_dim     = key_dim * 2 + value_dim;
        int kern        = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;

        sizes[BN_GPU_BUF_SSM_STATE]      = (size_t)n_ssm * num_v_heads * head_k_dim * head_v_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_CONV_STATE] = (size_t)n_ssm * (kern - 1) * qkv_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_QKV]        = (size_t)qkv_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_Z]          = (size_t)value_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_ALPHA]      = (size_t)num_v_heads * sizeof(float);
        sizes[BN_GPU_BUF_SSM_BETA]       = (size_t)num_v_heads * sizeof(float);
        sizes[BN_GPU_BUF_SSM_V]          = (size_t)value_dim * sizeof(float);
    }

    /* Validate buffer sizes against device limits before allocating */
    for (int i = 0; i < BN_GPU_BUF_COUNT; i++) {
        if (sizes[i] > 0 && (uint64_t)sizes[i] > ctx->max_buffer_size) {
            fprintf(stderr, "[bn:gpu:wgpu] activation buffer %d size %zu exceeds "
                    "maxBufferSize %llu (try --maxseq to reduce context length)\n",
                    i, sizes[i], (unsigned long long)ctx->max_buffer_size);
            return -1;
        }
    }

    /* Create each activation buffer (Storage | CopySrc | CopyDst) */
    for (int i = 0; i < BN_GPU_BUF_COUNT; i++) {
        if (sizes[i] == 0) continue;
        size_t aligned = (sizes[i] + 3) & ~(size_t)3;
        WGPUBufferDescriptor desc = {
            .label = sv("bn_act"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc
                     | WGPUBufferUsage_CopyDst,
            .size = aligned,
        };
        ctx->act_bufs[i] = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!ctx->act_bufs[i]) {
            wgpu_free_activations(ctx);
            return -1;
        }
        ctx->act_sizes[i] = aligned;
    }

    /* Upload precomputed RoPE frequencies */
    {
        int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : c->head_size;
        int half = rope_dims / 2;
        float *freq = malloc((size_t)half * sizeof(float));
        if (!freq) return -1;
        for (int i = 0; i < half; i++)
            freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)rope_dims);
        wgpuQueueWriteBuffer(ctx->queue, ctx->act_bufs[BN_GPU_BUF_ROPE_FREQ],
                              0, freq, (size_t)half * sizeof(float));
        free(freq);
    }

    /* Create staging buffer for logits readback */
    {
        size_t logits_size = (size_t)c->vocab_size * sizeof(float);
        size_t aligned = (logits_size + 3) & ~(size_t)3;
        WGPUBufferDescriptor desc = {
            .label = sv("bn_fwd_staging"),
            .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
            .size = aligned,
        };
        ctx->fwd_staging = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!ctx->fwd_staging) return -1;
        ctx->fwd_staging_size = aligned;
    }

    /* Create uniform ring buffer (~500 dispatches x 32 bytes) */
    {
        size_t ring_size = 1024 * 256;  /* ~1024 dispatches, 256B-aligned slots */
        WGPUBufferDescriptor desc = {
            .label = sv("bn_uniform_ring"),
            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
            .size = ring_size,
        };
        ctx->uniform_ring = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!ctx->uniform_ring) return -1;
        ctx->uniform_ring_size = ring_size;
    }

    /* Compile forward-pass shaders */
    static const struct { int id; const char *name; } fwd_shaders[] = {
        { BN_GPU_SHADER_RMSNORM,      "rmsnorm"      },
        { BN_GPU_SHADER_ROPE,         "rope"         },
        { BN_GPU_SHADER_GQA_SCORES,   "gqa_scores"   },
        { BN_GPU_SHADER_SOFTMAX,      "softmax"      },
        { BN_GPU_SHADER_GQA_COMBINE,  "gqa_combine"  },
        { BN_GPU_SHADER_SILU_GATE,    "silu_gate"    },
        { BN_GPU_SHADER_RELU2_GATE,   "relu2_gate"   },
        { BN_GPU_SHADER_RESIDUAL_ADD, "residual_add" },
        { BN_GPU_SHADER_BIAS_ADD,     "bias_add"     },
        { BN_GPU_SHADER_RESIDUAL_RMSNORM, "residual_rmsnorm" },
        { BN_GPU_SHADER_WEIGHTED_ADD,     "weighted_add"     },
        { BN_GPU_SHADER_SSM_CONV_SILU,    "ssm_conv_silu"    },
        { BN_GPU_SHADER_SSM_L2NORM,       "ssm_l2norm"       },
        { BN_GPU_SHADER_SSM_ALPHA_BETA,   "ssm_alpha_beta"   },
        { BN_GPU_SHADER_SSM_DELTA,        "ssm_delta"        },
        { BN_GPU_SHADER_SSM_GATE,         "ssm_gate"         },
        { BN_GPU_SHADER_PER_HEAD_RMSNORM, "per_head_rmsnorm" },
        { BN_GPU_SHADER_DEINTERLEAVE_Q,   "deinterleave_q"   },
        { BN_GPU_SHADER_SIGMOID_GATE,     "sigmoid_gate"     },
        { BN_GPU_SHADER_COPY,             "buf_copy"         },
    };
    int n_fwd = (int)(sizeof(fwd_shaders) / sizeof(fwd_shaders[0]));
    int compiled = 0;
    for (int i = 0; i < n_fwd; i++) {
        if (compile_fwd_pipeline(ctx, fwd_shaders[i].id, fwd_shaders[i].name) == 0)
            compiled++;
    }
    fprintf(stderr, "[bn:gpu:wgpu] compiled %d/%d forward-pass shaders\n",
            compiled, n_fwd);

    return 0;
}

/* ── Vtable: free_activations ──────────────────────────────────────── */

static void wgpu_free_activations(void *vctx)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    if (!ctx) return;

    for (int i = 0; i < BN_GPU_BUF_COUNT; i++) {
        if (ctx->act_bufs[i]) {
            wgpuBufferDestroy(ctx->act_bufs[i]);
            wgpuBufferRelease(ctx->act_bufs[i]);
            ctx->act_bufs[i] = NULL;
            ctx->act_sizes[i] = 0;
        }
    }
    if (ctx->fwd_staging) {
        wgpuBufferDestroy(ctx->fwd_staging);
        wgpuBufferRelease(ctx->fwd_staging);
        ctx->fwd_staging = NULL;
        ctx->fwd_staging_size = 0;
    }
    if (ctx->uniform_ring) {
        wgpuBufferDestroy(ctx->uniform_ring);
        wgpuBufferRelease(ctx->uniform_ring);
        ctx->uniform_ring = NULL;
        ctx->uniform_ring_size = 0;
    }
    for (int i = 0; i < BN_GPU_SHADER_COUNT; i++) {
        if (ctx->fwd_layouts[i]) {
            wgpuBindGroupLayoutRelease(ctx->fwd_layouts[i]);
            ctx->fwd_layouts[i] = NULL;
        }
        if (ctx->fwd_pipelines[i]) {
            wgpuComputePipelineRelease(ctx->fwd_pipelines[i]);
            ctx->fwd_pipelines[i] = NULL;
        }
    }
}

/* ── Vtable: write_activation ──────────────────────────────────────── */

static int wgpu_write_activation(void *vctx, int buf_idx, const void *data,
                                  size_t size, size_t offset)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    if (!ctx || !data || buf_idx < 0 || buf_idx >= BN_GPU_BUF_COUNT) return -1;
    if (!ctx->act_bufs[buf_idx]) return -1;
    if (offset + size > ctx->act_sizes[buf_idx]) return -1;
    wgpuQueueWriteBuffer(ctx->queue, ctx->act_bufs[buf_idx], offset, data, size);
    return 0;
}

/* ── Vtable: read_activation ───────────────────────────────────────── */

static int wgpu_read_activation(void *vctx, int buf_idx, void *out,
                                 size_t size, size_t offset)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    if (!ctx || !out || buf_idx < 0 || buf_idx >= BN_GPU_BUF_COUNT) return -1;
    if (!ctx->act_bufs[buf_idx]) return -1;
    if (offset + size > ctx->act_sizes[buf_idx]) return -1;

    /* Ensure staging buffer is large enough */
    size_t aligned = (size + 3) & ~(size_t)3;
    if (!ctx->fwd_staging || ctx->fwd_staging_size < aligned) return -1;

    /* Copy from activation buffer to staging */
    WGPUCommandEncoderDescriptor enc_desc = { .label = sv("bn_read_enc") };
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);
    if (!encoder) return -1;
    wgpuCommandEncoderCopyBufferToBuffer(encoder,
        ctx->act_bufs[buf_idx], offset,
        ctx->fwd_staging, 0, aligned);
    WGPUCommandBufferDescriptor cmd_desc = { .label = sv("bn_read_cmd") };
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuCommandEncoderRelease(encoder);
    if (!cmd) return -1;
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuDevicePoll(ctx->device, 1, NULL);
    wgpuCommandBufferRelease(cmd);

    /* Map and read */
    MapReq map_req = {0};
    WGPUBufferMapCallbackInfo map_cb = {
        .mode = WGPUCallbackMode_AllowSpontaneous,
        .callback = on_buffer_map,
        .userdata1 = &map_req,
    };
    wgpuBufferMapAsync(ctx->fwd_staging, WGPUMapMode_Read, 0, aligned, map_cb);
    wgpuDevicePoll(ctx->device, 1, NULL);
    if (!map_req.done || map_req.status != WGPUMapAsyncStatus_Success) return -1;
    const void *mapped = wgpuBufferGetConstMappedRange(ctx->fwd_staging, 0, aligned);
    if (!mapped) { wgpuBufferUnmap(ctx->fwd_staging); return -1; }
    memcpy(out, mapped, size);
    wgpuBufferUnmap(ctx->fwd_staging);
    return 0;
}

/* ── Vtable: execute ───────────────────────────────────────────────── */

static int wgpu_execute(void *vctx, const BnGPUOp *ops, int n_ops,
                         int readback_buf, float *out_host, int out_len)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    if (!ctx || !ops || n_ops <= 0) return -1;

    double t0_all = bn_platform_time_ms();

    /* 1. Upload all per-dispatch uniforms to the ring buffer.
     * Each slot is 256-byte aligned (min_uniform_buffer_offset_alignment).
     * Ring holds 1024 slots — sufficient for current models (max ~650 ops at 36 layers).
     * AUDIT(M7): if models exceed 1024 ops/token, this needs dynamic sizing. */
    size_t uni_stride = 256;
    size_t needed = (size_t)n_ops * uni_stride;
    if (needed > ctx->uniform_ring_size) return -1;

    uint8_t *uni_data = calloc(1, needed);
    if (!uni_data) return -1;
    for (int i = 0; i < n_ops; i++) {
        memcpy(uni_data + (size_t)i * uni_stride, ops[i].p,
               sizeof(uint32_t) * BN_GPU_OP_PARAMS);
        /* Fused bias: inject bias_offset from weight buffer metadata */
        if (ops[i].shader == BN_GPU_SHADER_MATVEC && ops[i].W_buf) {
            BnWgpuBuf *wbuf = (BnWgpuBuf *)ops[i].W_buf;
            if (wbuf->bias_offset > 0) {
                uint32_t *p = (uint32_t *)(uni_data + (size_t)i * uni_stride);
                p[4] = wbuf->bias_offset;
            }
        }
    }
    wgpuQueueWriteBuffer(ctx->queue, ctx->uniform_ring, 0, uni_data, needed);
    free(uni_data);
    double t1_uniforms = bn_platform_time_ms();

    /* 2. Create command encoder */
    WGPUCommandEncoderDescriptor enc_desc = { .label = sv("bn_fwd_enc") };
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);
    if (!encoder) return -1;

    /* 3. Encode compute passes with dependency-based merging.
     * Track which activation buffers are read/written in the current pass.
     * Only start a new pass when a dispatch has a RAW, WAR, or WAW conflict
     * with an earlier dispatch in the current pass. */

    /* Helper: get read/write buffer masks for an op.
     * Bit i set = buffer BN_GPU_BUF_i is accessed. */
    #define BUF_BIT(idx) (1u << (idx))

    WGPUComputePassEncoder cur_pass = NULL;
    uint32_t pass_reads = 0, pass_writes = 0;
    int n_passes = 0;

    for (int i = 0; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];

        /* COPY handled as compute shader — stays in same compute pass */

        /* Determine pipeline and layout */
        WGPUComputePipeline pipeline = NULL;
        WGPUBindGroupLayout layout = NULL;
        if (op->shader == BN_GPU_SHADER_MATVEC) {
            if (op->type >= 0 && op->type < BN_WGPU_MAX_TYPES) {
                pipeline = ctx->pipelines[op->type];
                layout = ctx->layouts[op->type];
            }
        } else if (op->shader > 0 && op->shader < BN_GPU_SHADER_COUNT) {
            pipeline = ctx->fwd_pipelines[op->shader];
            layout = ctx->fwd_layouts[op->shader];
        }
        if (!pipeline || !layout) continue;

        /* Compute this op's read/write buffer masks */
        uint32_t op_reads = 0, op_writes = 0;
        switch (op->shader) {
        case BN_GPU_SHADER_MATVEC:
            op_reads = BUF_BIT(op->buf_in);
            op_writes = BUF_BIT(op->buf_out);
            break;
        case BN_GPU_SHADER_RMSNORM:
            op_reads = BUF_BIT(op->buf_in);
            op_writes = BUF_BIT(op->buf_out);
            break;
        case BN_GPU_SHADER_ROPE:
            op_reads = BUF_BIT(op->buf_in) | BUF_BIT(BN_GPU_BUF_ROPE_FREQ);
            op_writes = BUF_BIT(op->buf_in);  /* in-place */
            break;
        case BN_GPU_SHADER_GQA_SCORES:
            op_reads = BUF_BIT(op->buf_in) | BUF_BIT(BN_GPU_BUF_KEY_CACHE);
            op_writes = BUF_BIT(BN_GPU_BUF_ATT);
            break;
        case BN_GPU_SHADER_SOFTMAX:
            op_reads = BUF_BIT(BN_GPU_BUF_ATT);
            op_writes = BUF_BIT(BN_GPU_BUF_ATT);  /* in-place */
            break;
        case BN_GPU_SHADER_GQA_COMBINE:
            op_reads = BUF_BIT(BN_GPU_BUF_ATT) | BUF_BIT(BN_GPU_BUF_VALUE_CACHE);
            op_writes = BUF_BIT(op->buf_out);
            break;
        case BN_GPU_SHADER_SILU_GATE:
        case BN_GPU_SHADER_RELU2_GATE:
            op_reads = BUF_BIT(op->buf_in) | BUF_BIT(op->buf_aux);
            op_writes = BUF_BIT(op->buf_in);  /* in-place */
            break;
        case BN_GPU_SHADER_RESIDUAL_ADD:
            op_reads = BUF_BIT(op->buf_in) | BUF_BIT(op->buf_aux);
            op_writes = BUF_BIT(op->buf_in);  /* in-place */
            break;
        case BN_GPU_SHADER_BIAS_ADD:
            op_reads = BUF_BIT(op->buf_in);
            op_writes = BUF_BIT(op->buf_in);  /* in-place */
            break;
        case BN_GPU_SHADER_RESIDUAL_RMSNORM:
            op_reads = BUF_BIT(op->buf_in) | BUF_BIT(op->buf_aux);
            op_writes = BUF_BIT(op->buf_in) | BUF_BIT(op->buf_out);
            break;
        case BN_GPU_SHADER_WEIGHTED_ADD:
            op_reads = BUF_BIT(op->buf_in) | BUF_BIT(op->buf_aux);
            op_writes = BUF_BIT(op->buf_in);  /* in-place: x += w*r */
            break;
        case BN_GPU_SHADER_SSM_CONV_SILU:
            op_reads = BUF_BIT(op->buf_in) | BUF_BIT(BN_GPU_BUF_SSM_CONV_STATE);
            op_writes = BUF_BIT(op->buf_in) | BUF_BIT(BN_GPU_BUF_SSM_CONV_STATE);
            break;
        case BN_GPU_SHADER_SSM_L2NORM:
            op_reads = BUF_BIT(op->buf_in) | BUF_BIT(op->buf_aux);
            op_writes = BUF_BIT(op->buf_in) | BUF_BIT(op->buf_aux);
            break;
        case BN_GPU_SHADER_SSM_ALPHA_BETA:
            op_reads = BUF_BIT(BN_GPU_BUF_SSM_ALPHA) | BUF_BIT(BN_GPU_BUF_SSM_BETA);
            op_writes = BUF_BIT(BN_GPU_BUF_SSM_ALPHA) | BUF_BIT(BN_GPU_BUF_SSM_BETA);
            break;
        case BN_GPU_SHADER_SSM_DELTA:
            op_reads = BUF_BIT(BN_GPU_BUF_SSM_STATE) | BUF_BIT(op->buf_in) | BUF_BIT(op->buf_aux)
                     | BUF_BIT(BN_GPU_BUF_SSM_V) | BUF_BIT(BN_GPU_BUF_SSM_ALPHA) | BUF_BIT(BN_GPU_BUF_SSM_BETA);
            op_writes = BUF_BIT(BN_GPU_BUF_SSM_STATE) | BUF_BIT(op->buf_out);
            break;
        case BN_GPU_SHADER_SSM_GATE:
            op_reads = BUF_BIT(op->buf_in) | BUF_BIT(op->buf_aux);
            op_writes = BUF_BIT(op->buf_in);  /* in-place */
            break;
        case BN_GPU_SHADER_PER_HEAD_RMSNORM:
            op_reads = BUF_BIT(op->buf_in);
            op_writes = BUF_BIT(op->buf_in);  /* in-place */
            break;
        case BN_GPU_SHADER_DEINTERLEAVE_Q:
            op_reads = BUF_BIT(op->buf_in);
            op_writes = BUF_BIT(op->buf_out);
            break;
        case BN_GPU_SHADER_SIGMOID_GATE:
            op_reads = BUF_BIT(op->buf_in) | BUF_BIT(op->buf_aux);
            op_writes = BUF_BIT(op->buf_in);  /* in-place */
            break;
        case BN_GPU_SHADER_COPY:
            op_reads = BUF_BIT(op->buf_in);
            op_writes = BUF_BIT(op->buf_out);
            break;
        default: continue;
        }

        /* Check for conflicts with current pass */
        int conflict = (op_reads & pass_writes) || (op_writes & pass_reads)
                     || (op_writes & pass_writes);

        if (conflict && cur_pass) {
            wgpuComputePassEncoderEnd(cur_pass);
            wgpuComputePassEncoderRelease(cur_pass);
            cur_pass = NULL;
            pass_reads = pass_writes = 0;
            n_passes++;
        }

        /* Start new pass if needed */
        if (!cur_pass) {
            WGPUComputePassDescriptor pass_desc = { .label = sv("bn_fwd_pass") };
            cur_pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
        }

        pass_reads |= op_reads;
        pass_writes |= op_writes;

        /* Build bind group entries per shader type */
        WGPUBindGroupEntry entries[8];
        int n_entries = 0;
        size_t uni_offset = (size_t)i * uni_stride;

        switch (op->shader) {
        case BN_GPU_SHADER_MATVEC: {
            BnWgpuBuf *wbuf = (BnWgpuBuf *)op->W_buf;
            if (!wbuf) continue;
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = wbuf->buf,
                .offset = wbuf->offset, .size = wbuf->size};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->act_bufs[op->buf_out],
                .offset = 0, .size = ctx->act_sizes[op->buf_out]};
            entries[3] = (WGPUBindGroupEntry){
                .binding = 3, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 4;
            break;
        }
        case BN_GPU_SHADER_RMSNORM: {
            BnWgpuBuf *wbuf = (BnWgpuBuf *)op->W_buf;
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1,
                .buffer = wbuf ? wbuf->buf : ctx->act_bufs[op->buf_in],
                .offset = wbuf ? wbuf->offset : 0,
                .size = wbuf ? wbuf->size : ctx->act_sizes[op->buf_in]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->act_bufs[op->buf_out],
                .offset = 0, .size = ctx->act_sizes[op->buf_out]};
            entries[3] = (WGPUBindGroupEntry){
                .binding = 3, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 4;
            break;
        }
        case BN_GPU_SHADER_ROPE: {
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[BN_GPU_BUF_ROPE_FREQ],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_ROPE_FREQ]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 3;
            break;
        }
        case BN_GPU_SHADER_GQA_SCORES: {
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[BN_GPU_BUF_KEY_CACHE],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_KEY_CACHE]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->act_bufs[BN_GPU_BUF_ATT],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_ATT]};
            entries[3] = (WGPUBindGroupEntry){
                .binding = 3, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 4;
            break;
        }
        case BN_GPU_SHADER_SOFTMAX: {
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[BN_GPU_BUF_ATT],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_ATT]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 2;
            break;
        }
        case BN_GPU_SHADER_GQA_COMBINE: {
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[BN_GPU_BUF_ATT],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_ATT]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[BN_GPU_BUF_VALUE_CACHE],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_VALUE_CACHE]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->act_bufs[op->buf_out],
                .offset = 0, .size = ctx->act_sizes[op->buf_out]};
            entries[3] = (WGPUBindGroupEntry){
                .binding = 3, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 4;
            break;
        }
        case BN_GPU_SHADER_SILU_GATE:
        case BN_GPU_SHADER_RELU2_GATE: {
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_aux],
                .offset = 0, .size = ctx->act_sizes[op->buf_aux]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 3;
            break;
        }
        case BN_GPU_SHADER_RESIDUAL_ADD: {
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_aux],
                .offset = 0, .size = ctx->act_sizes[op->buf_aux]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 3;
            break;
        }
        case BN_GPU_SHADER_BIAS_ADD: {
            BnWgpuBuf *wbuf = (BnWgpuBuf *)op->W_buf;
            if (!wbuf) continue;
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = wbuf->buf,
                .offset = wbuf->offset, .size = wbuf->size};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 3;
            break;
        }
        case BN_GPU_SHADER_RESIDUAL_RMSNORM: {
            /* 5 bindings: x(rw), r(ro), weight(ro), out(rw), uniforms */
            BnWgpuBuf *wbuf = (BnWgpuBuf *)op->W_buf;
            if (!wbuf) continue;
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_aux],
                .offset = 0, .size = ctx->act_sizes[op->buf_aux]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = wbuf->buf,
                .offset = wbuf->offset, .size = wbuf->size};
            entries[3] = (WGPUBindGroupEntry){
                .binding = 3, .buffer = ctx->act_bufs[op->buf_out],
                .offset = 0, .size = ctx->act_sizes[op->buf_out]};
            entries[4] = (WGPUBindGroupEntry){
                .binding = 4, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 5;
            break;
        }
        case BN_GPU_SHADER_WEIGHTED_ADD: {
            /* Same layout as residual_add: x(rw), r(ro), uniforms */
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_aux],
                .offset = 0, .size = ctx->act_sizes[op->buf_aux]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 3;
            break;
        }
        case BN_GPU_SHADER_SSM_CONV_SILU: {
            /* qkv(rw), conv_state(rw), conv1d_w(ro), uniforms */
            BnWgpuBuf *wbuf = (BnWgpuBuf *)op->W_buf;
            if (!wbuf) continue;
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[BN_GPU_BUF_SSM_CONV_STATE],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_SSM_CONV_STATE]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = wbuf->buf,
                .offset = wbuf->offset, .size = wbuf->size};
            entries[3] = (WGPUBindGroupEntry){
                .binding = 3, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 4;
            break;
        }
        case BN_GPU_SHADER_SSM_L2NORM: {
            /* q(rw), k(rw), uniforms */
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_aux],
                .offset = 0, .size = ctx->act_sizes[op->buf_aux]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 3;
            break;
        }
        case BN_GPU_SHADER_SSM_ALPHA_BETA: {
            /* alpha(rw)=SSM_ALPHA, beta(rw)=SSM_BETA, dt_bias(ro)=W_buf, a_log(ro)=p[6:7], uniforms */
            BnWgpuBuf *dt_buf = (BnWgpuBuf *)op->W_buf;
            if (!dt_buf) continue;
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[BN_GPU_BUF_SSM_ALPHA],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_SSM_ALPHA]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[BN_GPU_BUF_SSM_BETA],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_SSM_BETA]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = dt_buf->buf,
                .offset = dt_buf->offset, .size = dt_buf->size};
            /* a_log handle stored in buf_aux as a void* cast — resolve it */
            {
                void *a_ptr = (void *)(uintptr_t)((uint64_t)op->p[6] | ((uint64_t)op->p[7] << 32));
                BnWgpuBuf *a_wbuf = (BnWgpuBuf *)a_ptr;
                if (!a_wbuf) continue;
                entries[3] = (WGPUBindGroupEntry){
                    .binding = 3, .buffer = a_wbuf->buf,
                    .offset = a_wbuf->offset, .size = a_wbuf->size};
            }
            entries[4] = (WGPUBindGroupEntry){
                .binding = 4, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 5;
            break;
        }
        case BN_GPU_SHADER_SSM_DELTA: {
            /* state(rw), out(rw), q(ro), k(ro), v(rw), alpha(ro), beta(ro), uniforms */
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[BN_GPU_BUF_SSM_STATE],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_SSM_STATE]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_out],
                .offset = 0, .size = ctx->act_sizes[op->buf_out]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[3] = (WGPUBindGroupEntry){
                .binding = 3, .buffer = ctx->act_bufs[op->buf_aux],
                .offset = 0, .size = ctx->act_sizes[op->buf_aux]};
            entries[4] = (WGPUBindGroupEntry){
                .binding = 4, .buffer = ctx->act_bufs[BN_GPU_BUF_SSM_V],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_SSM_V]};
            entries[5] = (WGPUBindGroupEntry){
                .binding = 5, .buffer = ctx->act_bufs[BN_GPU_BUF_SSM_ALPHA],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_SSM_ALPHA]};
            entries[6] = (WGPUBindGroupEntry){
                .binding = 6, .buffer = ctx->act_bufs[BN_GPU_BUF_SSM_BETA],
                .offset = 0, .size = ctx->act_sizes[BN_GPU_BUF_SSM_BETA]};
            entries[7] = (WGPUBindGroupEntry){
                .binding = 7, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 8;
            break;
        }
        case BN_GPU_SHADER_SSM_GATE: {
            /* out(rw), z(ro), norm_w(ro), uniforms */
            BnWgpuBuf *wbuf = (BnWgpuBuf *)op->W_buf;
            if (!wbuf) continue;
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_aux],
                .offset = 0, .size = ctx->act_sizes[op->buf_aux]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = wbuf->buf,
                .offset = wbuf->offset, .size = wbuf->size};
            entries[3] = (WGPUBindGroupEntry){
                .binding = 3, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 4;
            break;
        }
        case BN_GPU_SHADER_PER_HEAD_RMSNORM: {
            /* x(rw), weight(ro), uniforms */
            BnWgpuBuf *wbuf = (BnWgpuBuf *)op->W_buf;
            if (!wbuf) continue;
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = wbuf->buf,
                .offset = wbuf->offset, .size = wbuf->size};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 3;
            break;
        }
        case BN_GPU_SHADER_DEINTERLEAVE_Q: {
            /* src(ro), dst(rw), uniforms */
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_out],
                .offset = 0, .size = ctx->act_sizes[op->buf_out]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 3;
            break;
        }
        case BN_GPU_SHADER_SIGMOID_GATE: {
            /* out(rw), gate(ro), uniforms */
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_aux],
                .offset = 0, .size = ctx->act_sizes[op->buf_aux]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 3;
            break;
        }
        case BN_GPU_SHADER_COPY: {
            entries[0] = (WGPUBindGroupEntry){
                .binding = 0, .buffer = ctx->act_bufs[op->buf_in],
                .offset = 0, .size = ctx->act_sizes[op->buf_in]};
            entries[1] = (WGPUBindGroupEntry){
                .binding = 1, .buffer = ctx->act_bufs[op->buf_out],
                .offset = 0, .size = ctx->act_sizes[op->buf_out]};
            entries[2] = (WGPUBindGroupEntry){
                .binding = 2, .buffer = ctx->uniform_ring,
                .offset = uni_offset, .size = 32};
            n_entries = 3;
            break;
        }
        default: continue;
        }

        /* Create bind group */
        WGPUBindGroupDescriptor bg_desc = {
            .label = sv("bn_fwd_bg"),
            .layout = layout,
            .entryCount = (size_t)n_entries,
            .entries = entries,
        };
        /* Skip ops with NULL activation buffers (buffer not allocated for this model) */
        {
            int valid = 1;
            for (int e = 0; e < n_entries; e++)
                if (!entries[e].buffer) { valid = 0; break; }
            if (!valid) continue;
        }

        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);
        if (!bg) continue;

        /* Compute workgroup count */
        uint32_t wg_x = 1, wg_y = 1;
        switch (op->shader) {
        case BN_GPU_SHADER_MATVEC: {
            /* Tiled dispatch: all types use TILE_ROWS=32 */
            if (op->p[3] > 0) {
                /* Large-vocab tiling: extra = wg_x per slice, rows split across Y */
                uint32_t tiled_rows = ((uint32_t)op->rows + 31) / 32;
                wg_x = op->p[3];
                wg_y = (tiled_rows + op->p[3] - 1) / op->p[3];
            } else {
                wg_x = ((uint32_t)op->rows + 31) / 32;
                wg_y = op->p[2];  /* n_tokens */
                if (wg_y == 0) wg_y = 1;
            }
            break;
        }
        case BN_GPU_SHADER_RMSNORM:
            wg_x = 1;  /* single workgroup */
            break;
        case BN_GPU_SHADER_ROPE:
        case BN_GPU_SHADER_GQA_SCORES:
        case BN_GPU_SHADER_SOFTMAX:
        case BN_GPU_SHADER_GQA_COMBINE:
            wg_x = op->p[0];  /* n_heads */
            break;
        case BN_GPU_SHADER_SILU_GATE:
        case BN_GPU_SHADER_RELU2_GATE:
        case BN_GPU_SHADER_RESIDUAL_ADD:
        case BN_GPU_SHADER_BIAS_ADD:
            wg_x = (op->p[0] + 255) / 256;  /* ceil(dim / 256) */
            break;
        case BN_GPU_SHADER_RESIDUAL_RMSNORM:
            wg_x = 1;  /* single workgroup (like rmsnorm) */
            break;
        case BN_GPU_SHADER_WEIGHTED_ADD:
            wg_x = (op->p[0] + 255) / 256;  /* ceil(dim / 256) */
            break;
        case BN_GPU_SHADER_SSM_CONV_SILU:
            wg_x = (op->p[0] + 255) / 256;  /* ceil(qkv_dim / 256) */
            break;
        case BN_GPU_SHADER_SSM_L2NORM:
            wg_x = op->p[0];  /* num_k_heads (p0 = head_dim but dispatch = num_heads) */
            wg_x = op->rows;  /* use rows field for num_k_heads */
            break;
        case BN_GPU_SHADER_SSM_ALPHA_BETA:
            wg_x = 1;  /* single workgroup, <= 64 v-heads */
            break;
        case BN_GPU_SHADER_SSM_DELTA:
            wg_x = op->rows;  /* num_v_heads */
            break;
        case BN_GPU_SHADER_SSM_GATE:
            wg_x = op->rows;  /* num_v_heads */
            break;
        case BN_GPU_SHADER_PER_HEAD_RMSNORM:
            wg_x = (uint32_t)op->rows;  /* n_heads */
            break;
        case BN_GPU_SHADER_DEINTERLEAVE_Q:
        case BN_GPU_SHADER_SIGMOID_GATE:
            wg_x = (op->p[0] + 255) / 256;  /* ceil(q_dim / 256) */
            break;
        case BN_GPU_SHADER_COPY:
            wg_x = (op->p[2] + 255) / 256;
            break;
        }

        /* Dispatch within current pass */
        wgpuComputePassEncoderSetPipeline(cur_pass, pipeline);
        wgpuComputePassEncoderSetBindGroup(cur_pass, 0, bg, 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(cur_pass, wg_x, wg_y, 1);
        wgpuBindGroupRelease(bg);
    }

    /* Close final pass */
    if (cur_pass) {
        wgpuComputePassEncoderEnd(cur_pass);
        wgpuComputePassEncoderRelease(cur_pass);
        n_passes++;
    }

    #undef BUF_BIT

    /* 4. Copy readback buffer to staging */
    size_t readback_size = (size_t)out_len * sizeof(float);
    if (readback_buf >= 0 && readback_buf < BN_GPU_BUF_COUNT
        && ctx->act_bufs[readback_buf] && out_host && out_len > 0) {
        wgpuCommandEncoderCopyBufferToBuffer(encoder,
            ctx->act_bufs[readback_buf], 0,
            ctx->fwd_staging, 0,
            readback_size);
    }

    double t2_encode = bn_platform_time_ms();

    /* 5. Finish, submit, poll */
    WGPUCommandBufferDescriptor cmd_desc = { .label = sv("bn_fwd_cmd") };
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuCommandEncoderRelease(encoder);
    if (!cmd) return -1;

    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuDevicePoll(ctx->device, 1, NULL);
    wgpuCommandBufferRelease(cmd);
    double t3_gpu = bn_platform_time_ms();

    /* 6. Map staging, read back results */
    if (out_host && out_len > 0 && readback_buf >= 0) {
        MapReq map_req = {0};
        WGPUBufferMapCallbackInfo map_cb = {
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback = on_buffer_map,
            .userdata1 = &map_req,
        };
        wgpuBufferMapAsync(ctx->fwd_staging, WGPUMapMode_Read,
                            0, readback_size, map_cb);
        wgpuDevicePoll(ctx->device, 1, NULL);

        if (!map_req.done || map_req.status != WGPUMapAsyncStatus_Success)
            return -1;

        const void *mapped = wgpuBufferGetConstMappedRange(
            ctx->fwd_staging, 0, readback_size);
        if (!mapped) {
            wgpuBufferUnmap(ctx->fwd_staging);
            return -1;
        }
        memcpy(out_host, mapped, readback_size);
        wgpuBufferUnmap(ctx->fwd_staging);
    }

    (void)n_passes;
    double t4_readback = bn_platform_time_ms();

    /* GPU profiling: set BN_GPU_PROFILE=1 to see per-frame timing */
    if (ctx->gpu_profile < 0) {
        const char *env = getenv("BN_GPU_PROFILE");
        ctx->gpu_profile = (env && env[0] == '1') ? 1 : 0;
    }
    if (ctx->gpu_profile && (ctx->gpu_frame < 5 || (ctx->gpu_frame % 50 == 0))) {
        fprintf(stderr, "[gpu:profile] frame=%d ops=%d passes=%d | "
                "uniforms=%.1fms encode=%.1fms gpu=%.1fms readback=%.1fms total=%.1fms\n",
                ctx->gpu_frame, n_ops, n_passes,
                t1_uniforms - t0_all,
                t2_encode - t1_uniforms,
                t3_gpu - t2_encode,
                t4_readback - t3_gpu,
                t4_readback - t0_all);
    }
    ctx->gpu_frame++;

    return 0;
}

/* ── Public API: create ────────────────────────────────────────────── */

BnGPUBackend *bn_gpu_wgpu_create(const char *shader_dir)
{
    BnWgpuCtx *ctx = calloc(1, sizeof(BnWgpuCtx));
    if (!ctx) return NULL;
    ctx->gpu_profile = -1;  /* uninitialized, checked on first execute */

    /* Create instance with primary backends.
     * Bit 3 = AllowUnderlyingNonCompliantAdapter (not in wgpu.h yet),
     * needed for Mesa dzn (Vulkan-over-D3D12) on WSL2. */
    WGPUInstanceExtras extras = {
        .chain = { .sType = (WGPUSType)WGPUSType_InstanceExtras },
        .backends = WGPUInstanceBackend_Primary,
        .flags = WGPUInstanceFlag_Default | (1 << 3),
    };
    WGPUInstanceDescriptor inst_desc = {
        .nextInChain = &extras.chain,
    };
    ctx->instance = wgpuCreateInstance(&inst_desc);
    if (!ctx->instance) {
        fprintf(stderr, "[bn:gpu:wgpu] failed to create instance\n");
        free(ctx);
        return NULL;
    }

    /* Request high-performance adapter */
    AdapterReq areq = {0};
    WGPURequestAdapterOptions adapter_opts = {
        .featureLevel = WGPUFeatureLevel_Compatibility,
        .powerPreference = WGPUPowerPreference_HighPerformance,
    };
    WGPURequestAdapterCallbackInfo adapter_cb = {
        .mode = WGPUCallbackMode_AllowSpontaneous,
        .callback = on_adapter_request,
        .userdata1 = &areq,
    };
    wgpuInstanceRequestAdapter(ctx->instance, &adapter_opts, adapter_cb);
    if (!areq.ok || !areq.adapter) {
        fprintf(stderr, "[bn:gpu:wgpu] no GPU adapter found\n");
        wgpuInstanceRelease(ctx->instance);
        free(ctx);
        return NULL;
    }
    ctx->adapter = areq.adapter;

    /* Request device with adapter's limits + increased buffer sizes */
    DeviceReq dreq = {0};
    WGPULimits limits = {0};
    WGPUStatus lim_status = wgpuAdapterGetLimits(ctx->adapter, &limits);
    if (lim_status != WGPUStatus_Success) {
        fprintf(stderr, "[bn:gpu:wgpu] failed to get adapter limits\n");
        wgpuAdapterRelease(ctx->adapter);
        wgpuInstanceRelease(ctx->instance);
        free(ctx);
        return NULL;
    }
    /* Use adapter limits as-is — they represent the hardware maximum.
     * Clamp to wgpu's INT32_MAX ceiling to avoid off-by-one validation. */
    const uint64_t wgpu_ceil = (uint64_t)INT32_MAX;
    if (limits.maxStorageBufferBindingSize > wgpu_ceil)
        limits.maxStorageBufferBindingSize = wgpu_ceil;
    if (limits.maxBufferSize > wgpu_ceil)
        limits.maxBufferSize = wgpu_ceil;
    ctx->max_buffer_size = limits.maxBufferSize;
    /* Request ShaderF16 feature for f16 WGSL shaders */
    WGPUFeatureName required_features[] = { WGPUFeatureName_ShaderF16 };
    WGPUDeviceDescriptor dev_desc = {
        .label = sv("bn_device"),
        .requiredFeatures = required_features,
        .requiredFeatureCount = 1,
        .requiredLimits = &limits,
        .uncapturedErrorCallbackInfo = { .callback = on_uncaptured_error },
        .deviceLostCallbackInfo = { .callback = on_device_lost },
    };
    WGPURequestDeviceCallbackInfo device_cb = {
        .mode = WGPUCallbackMode_AllowSpontaneous,
        .callback = on_device_request,
        .userdata1 = &dreq,
    };
    wgpuAdapterRequestDevice(ctx->adapter, &dev_desc, device_cb);
    if (!dreq.ok || !dreq.device) {
        fprintf(stderr, "[bn:gpu:wgpu] device request failed\n");
        wgpuAdapterRelease(ctx->adapter);
        wgpuInstanceRelease(ctx->instance);
        free(ctx);
        return NULL;
    }
    ctx->device = dreq.device;
    ctx->queue = wgpuDeviceGetQueue(ctx->device);
    if (!ctx->queue) {
        fprintf(stderr, "[bn:gpu:wgpu] failed to get device queue\n");
        wgpuDeviceRelease(ctx->device);
        wgpuAdapterRelease(ctx->adapter);
        wgpuInstanceRelease(ctx->instance);
        free(ctx);
        return NULL;
    }

    /* Store shader directory for forward-pass shader compilation */
    if (shader_dir) {
        snprintf(ctx->shader_dir, sizeof(ctx->shader_dir), "%s", shader_dir);
    }

    /* Compile pipelines for all supported quant types */
    int compiled = 0;
    for (int i = 0; i < N_SUPPORTED_TYPES; i++) {
        int type = supported_types[i];
        if (compile_pipeline(ctx, type, shader_dir) == 0)
            compiled++;
        /* Non-fatal: types without shaders just fall back to CPU */
    }

    fprintf(stderr, "[bn:gpu:wgpu] compiled %d/%d shader pipelines\n",
            compiled, N_SUPPORTED_TYPES);

    /* Build vtable */
    BnGPUBackend *gpu = calloc(1, sizeof(BnGPUBackend));
    if (!gpu) {
        /* Clean up ctx directly since bn_gpu_wgpu_destroy expects a vtable */
        wgpu_free_activations(ctx);
        if (ctx->queue) wgpuQueueRelease(ctx->queue);
        if (ctx->device) wgpuDeviceRelease(ctx->device);
        if (ctx->adapter) wgpuAdapterRelease(ctx->adapter);
        if (ctx->instance) wgpuInstanceRelease(ctx->instance);
        free(ctx);
        return NULL;
    }
    gpu->buffer_create         = wgpu_buffer_create;
    gpu->buffer_create_biased  = wgpu_buffer_create_biased;
    gpu->buffer_destroy        = wgpu_buffer_destroy;
    gpu->matvec            = wgpu_matvec;
    gpu->matmul            = wgpu_matmul;
    gpu->matvec_batch      = wgpu_matvec_batch;
    gpu->execute           = wgpu_execute;
    gpu->init_activations  = wgpu_init_activations;
    gpu->free_activations  = wgpu_free_activations;
    gpu->write_activation  = wgpu_write_activation;
    gpu->read_activation   = wgpu_read_activation;
    gpu->ctx               = ctx;

    return gpu;
}

/* ── Public API: destroy ───────────────────────────────────────────── */

int bn_gpu_wgpu_init_slab(BnGPUBackend *gpu, size_t size_mb)
{
    if (!gpu || !gpu->ctx || size_mb == 0) return -1;
    return slab_init((BnWgpuCtx *)gpu->ctx, size_mb * 1024 * 1024);
}

void bn_gpu_wgpu_destroy(BnGPUBackend *gpu)
{
    if (!gpu) return;

    BnWgpuCtx *ctx = (BnWgpuCtx *)gpu->ctx;
    if (ctx) {
        /* Release forward-pass activation buffers, staging, and shaders */
        wgpu_free_activations(ctx);

        /* Release persistent scratch buffers */
        if (ctx->x_buf) {
            wgpuBufferDestroy(ctx->x_buf);
            wgpuBufferRelease(ctx->x_buf);
        }
        if (ctx->out_buf) {
            wgpuBufferDestroy(ctx->out_buf);
            wgpuBufferRelease(ctx->out_buf);
        }
        if (ctx->uniform_buf) {
            wgpuBufferDestroy(ctx->uniform_buf);
            wgpuBufferRelease(ctx->uniform_buf);
        }
        if (ctx->staging_buf) {
            wgpuBufferDestroy(ctx->staging_buf);
            wgpuBufferRelease(ctx->staging_buf);
        }

        /* Release slab allocator */
        if (ctx->slab_buf) {
            wgpuBufferDestroy(ctx->slab_buf);
            wgpuBufferRelease(ctx->slab_buf);
        }
        free(ctx->slab_free);

        /* Release pipelines and layouts */
        for (int i = 0; i < BN_WGPU_MAX_TYPES; i++) {
            if (ctx->layouts[i])
                wgpuBindGroupLayoutRelease(ctx->layouts[i]);
            if (ctx->pipelines[i])
                wgpuComputePipelineRelease(ctx->pipelines[i]);
        }
        if (ctx->queue) wgpuQueueRelease(ctx->queue);
        if (ctx->device) wgpuDeviceRelease(ctx->device);
        if (ctx->adapter) wgpuAdapterRelease(ctx->adapter);
        if (ctx->instance) wgpuInstanceRelease(ctx->instance);
        free(ctx);
    }
    free(gpu);
}

#endif /* BN_ENABLE_GPU */
