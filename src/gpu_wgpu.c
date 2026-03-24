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
#include "quant.h"
#include "gguf.h"
#include "webgpu.h"
#include "wgpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
} BnWgpuCtx;

/* ── GPU buffer handle ─────────────────────────────────────────────── */

typedef struct {
    WGPUBuffer buf;
    size_t     size;
    int        type;
    int        rows;
    int        cols;
} BnWgpuBuf;

/* ── Uniform block for compute shaders ─────────────────────────────── */

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t n_tokens;
    uint32_t extra;  /* reserved / padding */
} BnWgpuUniforms;

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

    /* uniform_buf: 16 bytes, create once */
    if (!ctx->uniform_buf) {
        size_t uni_size = (sizeof(BnWgpuUniforms) + 15) & ~(size_t)15;
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

static _Thread_local int wgpu_last_error;

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
    BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_Q2_K, BN_GGUF_TENSOR_Q3_K,
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

/* ── Vtable: buffer_create ─────────────────────────────────────────── */

static void *wgpu_buffer_create(void *vctx, const void *data, size_t size,
                                 int type, int rows, int cols)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    if (!ctx || !ctx->device || !data || size == 0)
        return NULL;

    /* Align size to 4 bytes (WebGPU requirement) */
    size_t aligned = (size + 3) & ~(size_t)3;

    WGPUBufferDescriptor desc = {
        .label = sv("bn_weight"),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size  = aligned,
    };
    WGPUBuffer buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
    if (!buf) return NULL;

    /* Upload weight data */
    wgpuQueueWriteBuffer(ctx->queue, buf, 0, data, size);

    /* Wrap in handle struct */
    BnWgpuBuf *handle = malloc(sizeof(BnWgpuBuf));
    if (!handle) {
        wgpuBufferDestroy(buf);
        wgpuBufferRelease(buf);
        return NULL;
    }
    handle->buf = buf;
    handle->size = aligned;
    handle->type = type;
    handle->rows = rows;
    handle->cols = cols;
    return handle;
}

/* ── Vtable: buffer_destroy ────────────────────────────────────────── */

static void wgpu_buffer_destroy(void *vctx, void *buffer)
{
    (void)vctx;
    if (!buffer) return;
    BnWgpuBuf *h = (BnWgpuBuf *)buffer;
    if (h->buf) {
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
            { .binding = 0, .buffer = wbuf->buf,       .offset = 0, .size = wbuf->size },
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
        wgpuComputePassEncoderDispatchWorkgroups(pass, (uint32_t)rows, 1, 1);
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
            { .binding = 0, .buffer = wbuf->buf,       .offset = 0, .size = wbuf->size },
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
        wgpuComputePassEncoderDispatchWorkgroups(
            pass, (uint32_t)rows, (uint32_t)n_tokens, 1);
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

        /* Write uniforms for this op */
        BnWgpuUniforms uniforms = {
            .rows = (uint32_t)op->rows,
            .cols = (uint32_t)op->cols,
            .n_tokens = 1,
            .extra = 0,
        };
        wgpuQueueWriteBuffer(ctx->queue, ctx->uniform_buf, 0,
                              &uniforms, sizeof(uniforms));

        /* Create bind group: W varies per op, x/out/uniform are persistent */
        WGPUBindGroupEntry entries[4] = {
            { .binding = 0, .buffer = wbuf->buf,       .offset = 0, .size = wbuf->size },
            { .binding = 1, .buffer = ctx->x_buf,      .offset = 0, .size = x_aligned },
            { .binding = 2, .buffer = ctx->out_buf,    .offset = 0, .size = out_aligned },
            { .binding = 3, .buffer = ctx->uniform_buf, .offset = 0,
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
        wgpuComputePassEncoderDispatchWorkgroups(pass, (uint32_t)op->rows, 1, 1);
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

/* ── Public API: create ────────────────────────────────────────────── */

BnGPUBackend *bn_gpu_wgpu_create(const char *shader_dir)
{
    BnWgpuCtx *ctx = calloc(1, sizeof(BnWgpuCtx));
    if (!ctx) return NULL;

    /* Create instance with primary backends */
    WGPUInstanceExtras extras = {
        .chain = { .sType = (WGPUSType)WGPUSType_InstanceExtras },
        .backends = WGPUInstanceBackend_Primary,
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
    }
    /* Override buffer sizes to allow large weight tensors (up to 2GB) */
    if (limits.maxStorageBufferBindingSize < (uint64_t)2u * 1024 * 1024 * 1024)
        limits.maxStorageBufferBindingSize = (uint64_t)2u * 1024 * 1024 * 1024;
    if (limits.maxBufferSize < (uint64_t)2u * 1024 * 1024 * 1024)
        limits.maxBufferSize = (uint64_t)2u * 1024 * 1024 * 1024;
    WGPUDeviceDescriptor dev_desc = {
        .label = sv("bn_device"),
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
        bn_gpu_wgpu_destroy(NULL);  /* ctx leaked — but OOM anyway */
        return NULL;
    }
    gpu->buffer_create  = wgpu_buffer_create;
    gpu->buffer_destroy = wgpu_buffer_destroy;
    gpu->matvec         = wgpu_matvec;
    gpu->matmul         = wgpu_matmul;
    gpu->matvec_batch   = wgpu_matvec_batch;
    gpu->ctx            = ctx;

    return gpu;
}

/* ── Public API: destroy ───────────────────────────────────────────── */

void bn_gpu_wgpu_destroy(BnGPUBackend *gpu)
{
    if (!gpu) return;

    BnWgpuCtx *ctx = (BnWgpuCtx *)gpu->ctx;
    if (ctx) {
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
