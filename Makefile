CC      ?= cc
CXX     ?= c++
# Prefer the selected toolkit over an older distribution nvcc on PATH.
NVCC    ?= $(if $(wildcard /usr/local/cuda/bin/nvcc),/usr/local/cuda/bin/nvcc,nvcc)
CUDA_ARCH ?= sm_120
LDFLAGS = -lm

LLAMA_PROBE_CFLAGS ?= $(shell pkg-config --cflags llama ggml 2>/dev/null)
LLAMA_CPP_SRC ?= /home/mark/artalis.io/tools/llama.cpp/src
LLAMA_PROBE_LIBS ?= $(shell pkg-config --libs-only-L llama ggml 2>/dev/null) \
                    $(shell pkg-config --libs-only-l llama 2>/dev/null)

# Platform-specific arch flags:
# -mcpu=apple-m1 on Darwin enables FP16 vector arithmetic + dotprod.
# -march=native on Apple clang misses __ARM_FEATURE_FP16_VECTOR_ARITHMETIC.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
CFLAGS  = -O3 -mcpu=apple-m1 -Wall -Wextra -Wshadow -std=c11 -Iinclude
else
CFLAGS  = -O3 -march=native -Wall -Wextra -Wshadow -std=c11 -Iinclude
endif

# On Linux, enable GNU extensions for strdup, qsort_r, clock_gettime, etc.
ifeq ($(UNAME_S),Linux)
CFLAGS += -D_GNU_SOURCE
LDFLAGS += -lpthread
endif

# Optional host-only preparation math. Sequential linkage avoids changing
# process-wide MKL threading settings; inference kernels remain unchanged.
ifeq ($(BN_ENABLE_MKL_VML),1)
ifneq ($(UNAME_S),Linux)
$(error BN_ENABLE_MKL_VML requires Linux x86-64)
endif
ifneq ($(shell uname -m),x86_64)
$(error BN_ENABLE_MKL_VML requires Linux x86-64)
endif
MKL_VML_ROOT ?= $(CURDIR)/.deps/oneMKL-2021.1.1
CFLAGS += -DBN_ENABLE_MKL_VML -I$(MKL_VML_ROOT)/include
LDFLAGS += -L$(MKL_VML_ROOT)/lib \
    -Wl,--disable-new-dtags,-rpath,$(MKL_VML_ROOT)/lib \
    -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
    -Wl,--as-needed -lpthread -lm -ldl
endif

QUANT_COMMON = src/quant/int4_rows.c src/quant/fp8.c src/quant/exl3.c src/quant/fp16.c src/quant/dequant.c src/quant/registry.c src/quant/prepared.c src/quant/kernel_select.c src/quant/policy.c src/quant/dispatch.c src/quant/matvec_batch.c src/quant/matmul.c src/quant/fused_gateup.c src/quant/batch_prepared_kquant_input.c src/quant/matvec_multi.c src/quant/mxfp4_scalar.c

UNAME_M := $(shell uname -m)
ifneq ($(filter arm% aarch%,$(UNAME_M)),)
  # ARM: NEON + NEON SDOT + scalar
  QUANT_BACKEND = src/quant/x_quant_neon.c \
    src/quant/i2s_neon_sdot.c src/quant/i2s_neon.c src/quant/i2s_scalar.c \
    src/quant/tq2_neon_sdot.c src/quant/tq2_neon.c src/quant/tq2_scalar.c \
    src/quant/tq1_neon_sdot.c src/quant/tq1_neon.c src/quant/tq1_scalar.c \
    src/quant/q8_neon_sdot.c src/quant/q8_neon.c src/quant/q8_scalar.c \
    src/quant/q4_neon_sdot.c src/quant/q4_neon.c src/quant/q4_scalar.c \
    src/quant/mxfp4_neon.c \
    src/quant/q4_1_neon.c src/quant/q4_1_scalar.c \
    src/quant/f32_neon.c src/quant/f32_scalar.c \
    src/quant/f16_neon.c src/quant/f16_scalar.c \
    src/quant/bf16_neon.c src/quant/bf16_scalar.c \
    src/quant/q6k_neon_sdot.c src/quant/q6k_neon.c src/quant/q6k_scalar.c \
    src/quant/q8k_neon.c src/quant/q8k_scalar.c \
    src/quant/q4k_neon_sdot.c src/quant/q4k_neon.c src/quant/q4k_scalar.c \
    src/quant/q5k_neon.c src/quant/q5k_scalar.c \
    src/quant/q3k_neon.c src/quant/q3k_scalar.c \
    src/quant/q2k_neon.c src/quant/q2k_scalar.c \
    src/quant/iq4nl_neon.c src/quant/iq4nl_scalar.c \
    src/quant/iq4xs_neon.c src/quant/iq4xs_scalar.c \
    src/quant/iq3xxs_neon.c src/quant/iq3xxs_scalar.c \
    src/quant/iq3s_neon.c src/quant/iq3s_scalar.c \
    src/quant/iq2xxs_neon.c src/quant/iq2xxs_scalar.c \
    src/quant/iq2xs_neon.c src/quant/iq2xs_scalar.c \
    src/quant/iq2s_neon.c src/quant/iq2s_scalar.c

  TRANSFORMER_BACKEND = src/transformer/math_neon.c src/transformer/math_scalar.c \
    src/transformer/math_backend.c \
    src/transformer/rmsnorm_neon.c src/transformer/rmsnorm_scalar.c \
    src/transformer/rmsnorm_backend.c \
    src/transformer/gqa_neon.c src/transformer/gqa_scalar.c \
    src/transformer/gqa_tq_scalar.c src/transformer/gqa_tq_neon.c \
    src/transformer/batched_attn_avx2.c src/transformer/batched_attn_neon.c \
    src/transformer/batched_attn_scalar.c \
    src/transformer/logits_neon.c src/transformer/logits_scalar.c \
    src/transformer/logits_policy.c \
    src/transformer/logits_backend.c src/transformer/logits.c \
    src/transformer/cpu_policy.c \
    src/transformer/cpu_backend.c src/transformer/cpu.c \
    src/transformer/plan.c src/transformer/gpu_fallback.c \
    src/transformer/gpu_policy.c \
    src/transformer/gpu_resources.c \
    src/transformer/gpu_emit.c src/transformer/gpu.c src/transformer/bf16.c src/transformer/bf16_pool.c src/transformer/trellis.c \
    src/transformer/kv_backend.c src/transformer/kv.c \
    src/transformer/prefill_policy.c \
    src/transformer/prefill_backend.c src/transformer/prefill.c \
    src/transformer/ssm_neon.c src/transformer/ssm_scalar.c
else
  # x86: AVX512 VNNI where available, plus AVX2 + scalar fallback
  CFLAGS += -mprefer-vector-width=256
  QUANT_BACKEND = src/quant/x_quant_avx2.c src/quant/rmsnorm_q8k_avx2.c \
    src/quant/mxfp4_avx2.c \
    src/quant/i2s_avx2.c src/quant/i2s_avx2_4row.c src/quant/i2s_scalar.c \
    src/quant/tq2_avx2.c src/quant/tq2_scalar.c \
    src/quant/tq1_avx2.c src/quant/tq1_scalar.c \
    src/quant/q8_avx512_vnni.c src/quant/q8_avx2.c src/quant/q8_scalar.c \
    src/quant/q4_avx512_vnni.c src/quant/q4_avx2.c src/quant/q4_avx2_4row.c src/quant/q4_avx2_matmul.c src/quant/q4_scalar.c \
    src/quant/q4_1_avx2.c src/quant/q4_1_scalar.c \
    src/quant/f32_avx2.c src/quant/f32_scalar.c \
    src/quant/f16_avx2.c src/quant/f16_scalar.c \
    src/quant/bf16_avx2.c src/quant/bf16_scalar.c \
    src/quant/q6k_avx512_vnni.c src/quant/q6k_avx2.c src/quant/q6k_avx2_sdot.c src/quant/q6k_avx2_4row.c src/quant/q6k_scalar.c \
    src/quant/q8k_avx2.c src/quant/q8k_scalar.c \
    src/quant/q4k_avx512_vnni.c src/quant/q4k_avx2.c src/quant/q4k_avx2_sdot.c src/quant/q4k_avx2_4row.c src/quant/q4k_scalar.c \
    src/quant/q5k_avx512.c src/quant/q5k_avx2.c src/quant/q5k_scalar.c \
    src/quant/q3k_avx2.c src/quant/q3k_scalar.c \
    src/quant/q2k_avx2.c src/quant/q2k_scalar.c \
    src/quant/iq4nl_avx2.c src/quant/iq4nl_scalar.c \
    src/quant/iq4xs_avx2.c src/quant/iq4xs_scalar.c \
    src/quant/iq3xxs_avx2.c src/quant/iq3xxs_scalar.c \
    src/quant/iq3s_avx2.c src/quant/iq3s_scalar.c \
    src/quant/iq2xxs_avx2.c src/quant/iq2xxs_scalar.c \
    src/quant/iq2xs_avx2.c src/quant/iq2xs_scalar.c \
    src/quant/iq2s_avx2.c src/quant/iq2s_scalar.c

  TRANSFORMER_BACKEND = src/transformer/math_scalar.c src/transformer/math_backend.c \
    src/transformer/rmsnorm_avx2.c src/transformer/rmsnorm_scalar.c \
    src/transformer/rmsnorm_backend.c \
    src/transformer/gqa_avx2.c src/transformer/gqa_scalar.c \
    src/transformer/gqa_tq_scalar.c \
    src/transformer/batched_attn_avx2.c src/transformer/batched_attn_scalar.c \
    src/transformer/logits_avx2.c src/transformer/logits_scalar.c \
    src/transformer/logits_policy.c \
    src/transformer/logits_backend.c src/transformer/logits.c \
    src/transformer/cpu_policy.c \
    src/transformer/cpu_backend.c src/transformer/cpu.c \
    src/transformer/plan.c src/transformer/gpu_fallback.c \
    src/transformer/gpu_policy.c \
    src/transformer/gpu_resources.c \
    src/transformer/gpu_emit.c src/transformer/gpu.c src/transformer/bf16.c src/transformer/bf16_pool.c src/transformer/trellis.c \
    src/transformer/kv_backend.c src/transformer/kv.c \
    src/transformer/prefill_policy.c \
    src/transformer/prefill_backend.c src/transformer/prefill.c \
    src/transformer/ssm_avx2.c src/transformer/ssm_scalar.c
endif

QUANT_SRCS = $(QUANT_COMMON) $(QUANT_BACKEND)
MODEL_SRCS = src/json.c src/safetensors.c src/trellis.c src/model.c src/model_arch.c src/model_policy.c src/model_session.c src/model_gpu.c src/model_embed.c src/model_backend.c src/model_moe.c src/backend_layout.c src/backend_model.c src/backend_model_rows.c src/backend_session.c src/backend_session_rows.c src/backend_quant.c src/runtime_policy.c src/gpu_backend_runtime.c src/gpu_policy.c
MOE_SRCS = src/moe.c src/moe_cache.c src/moe_io.c src/moe_policy.c src/moe_cpu_kernels.c src/moe_route.c src/moe_math.c src/moe_execute.c src/moe_prefill.c src/moe_stats.c
TRANSFORMER_SRCS = src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND)
SAMPLER_SRCS = src/sampler.c src/sampler_backend.c

# --- WebGPU (optional: BN_ENABLE_WEBGPU=1; BN_ENABLE_GPU=1 is a compatibility alias) ---
ifdef BN_ENABLE_GPU
  BN_ENABLE_WEBGPU := 1
endif
ifdef BN_ENABLE_WEBGPU
  ifndef WGPU_LIB_DIR
    WGPU_LIB_DIR := vendor/wgpu
  endif
  WGPU_LIB := $(WGPU_LIB_DIR)/libwgpu_native.a
  WEBGPU_CFLAGS := -DBN_ENABLE_WEBGPU -I$(WGPU_LIB_DIR)
  ifeq ($(UNAME_S),Darwin)
    WGPU_FRAMEWORKS := -framework Metal -framework QuartzCore -framework CoreGraphics -framework Foundation
  else
    WGPU_FRAMEWORKS := -lvulkan
  endif
  WEBGPU_SRCS := src/gpu_wgpu.c
  WEBGPU_OBJS := src/gpu_wgpu.o
else
  WGPU_LIB :=
  WEBGPU_CFLAGS :=
  WGPU_FRAMEWORKS :=
  WEBGPU_SRCS :=
  WEBGPU_OBJS :=
endif

# --- Metal (optional: BN_ENABLE_METAL=1, macOS only) ---
.PHONY: test_metal_f32
ifdef BN_ENABLE_METAL
  METAL_CFLAGS := -DBN_ENABLE_METAL
  METAL_FRAMEWORKS := -framework Metal -framework Foundation
  METAL_SRCS := src/gpu_metal.m
  METAL_OBJS := src/gpu_metal.o
else
  METAL_CFLAGS :=
  METAL_FRAMEWORKS :=
  METAL_SRCS :=
  METAL_OBJS :=
endif

# --- CUDA (optional: BN_ENABLE_CUDA=1) ---
FA2_ROOT ?= .deps/flash-attention
MLA_ROOT ?= .deps/flashinfer
ifdef BN_ENABLE_CUDA
  CUDA_CFLAGS := -DBN_ENABLE_CUDA
  CUDA_NVCCFLAGS := -arch=$(CUDA_ARCH)
  ifneq ($(filter sm_120 sm_120a,$(CUDA_ARCH)),)
    # Block-scaled FP4 requires the architecture-specific SM120 instruction set.
    CUDA_NVCCFLAGS := -gencode=arch=compute_120a,code=sm_120a
    CUDA_CFLAGS += -DBN_CUDA_MXFP4_SM120
  endif
  CUDA_SRCS := src/gpu_cuda.cu src/gpu_cuda_native_frequency.cu src/gpu_cuda_native_index.cu src/gpu_cuda_native_topk.cu src/gpu_cuda_native_candidates.cu src/gpu_cuda_native_candidate_use.cu src/gpu_cuda_compress.cu src/gpu_cuda_native_rope.cu src/gpu_cuda_exl3_decode.cu src/gpu_cuda_exl3_fused.cu src/gpu_cuda_native_moe.cu src/gpu_cuda_exl3_fold.cu src/gpu_cuda_router.cu src/gpu_cuda_hc4.cu src/gpu_cuda_fa2.cu src/gpu_cuda_mla_attention.cu
  CUDA_OBJS := src/gpu_cuda.o src/gpu_cuda_native_frequency.o src/gpu_cuda_native_attention.o src/gpu_cuda_native_index.o src/gpu_cuda_native_topk.o src/gpu_cuda_native_candidates.o src/gpu_cuda_native_candidate_use.o src/gpu_cuda_compress.o src/gpu_cuda_native_rope.o src/gpu_cuda_exl3_decode.o src/gpu_cuda_exl3_fused.o src/gpu_cuda_native_moe.o src/gpu_cuda_exl3_fold.o src/gpu_cuda_router.o src/gpu_cuda_hc4.o src/gpu_cuda_fa2.o src/gpu_cuda_mla_attention.o
  CUDA_LDFLAGS := -L$(dir $(NVCC))../lib64 -lcudart -lcublas -lstdc++
  LINK := $(CC)
else
  CUDA_CFLAGS :=
  CUDA_NVCCFLAGS :=
  CUDA_SRCS :=
  CUDA_OBJS :=
  CUDA_LDFLAGS :=
  LINK := $(CC)
endif

SRCS = src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
       $(TRANSFORMER_SRCS) src/tokenizer.c $(SAMPLER_SRCS) \
       src/threadpool.c src/sh_arena.c src/sh_log.c src/bn_alloc.c src/session.c src/prompt_cache.c src/generate_batch_plan.c src/generate_batch.c src/generate.c $(WEBGPU_SRCS) src/main.c
CFLAGS += $(WEBGPU_CFLAGS) $(METAL_CFLAGS) $(CUDA_CFLAGS)
LDFLAGS += $(WGPU_LIB) $(WGPU_FRAMEWORKS) $(METAL_FRAMEWORKS) $(CUDA_LDFLAGS)
OBJS = $(SRCS:.c=.o) $(METAL_OBJS) $(CUDA_OBJS)
HEADERS = $(wildcard include/*.h src/*.h src/transformer/*.h)
BUILD_CONFIG := webgpu=$(if $(BN_ENABLE_WEBGPU),1,0) metal=$(if $(BN_ENABLE_METAL),1,0) cuda=$(if $(BN_ENABLE_CUDA),1,0) cc=$(CC) nvcc=$(NVCC) cuda_arch=$(CUDA_ARCH) cflags=$(CFLAGS)
BUILD_CONFIG_STAMP := .build-config
BUILD_CONFIG_CHANGED := $(shell old=$$(cat $(BUILD_CONFIG_STAMP) 2>/dev/null || true); \
	if [ "$$old" != "$(BUILD_CONFIG)" ]; then \
		printf '%s\n' "$(BUILD_CONFIG)" > $(BUILD_CONFIG_STAMP); \
		printf '1'; \
	fi)
ifneq ($(BUILD_CONFIG_CHANGED),)
  $(info Build config changed; rebuilding objects)
endif

# Default target
bitnet: $(BUILD_CONFIG_STAMP) $(OBJS)
	$(LINK) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

.PHONY: test_cli_bf16
test_cli_bf16: bitnet
	python3 test/test_cli_bf16.py --binary ./bitnet --model "$(MODEL)"

.PHONY: test_cli_bf16_cp
test_cli_bf16_cp: bitnet
	python3 test/test_cli_bf16_cp.py --binary ./bitnet --model "$(MODEL)"

# Debug build
debug: CFLAGS += -DDEBUG -g -O0
debug: bitnet

# Sanitizer build (ASan + UBSan)
asan: CFLAGS += -DDEBUG -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
asan: LDFLAGS += -fsanitize=address,undefined
asan: bitnet

# Pattern rules for object files
src/%.o: src/%.c $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(CC) $(CFLAGS) -c -o $@ $<

src/quant/%.o: src/quant/%.c $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(CC) $(CFLAGS) -c -o $@ $<

src/transformer/%.o: src/transformer/%.c $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(CC) $(CFLAGS) -c -o $@ $<

src/%.o: src/%.cu $(HEADERS) $(wildcard src/*.cuh src/quant/*.cuh) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++11 -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda.o: src/gpu_cuda_expert_cache_group.cuh src/gpu_cuda_expert_cache_bank.cuh src/gpu_cuda_native_ple.cuh src/gpu_cuda_moe_shared.cuh src/gpu_cuda_moe_joint.cuh
src/transformer/bf16.o: src/transformer/bf16_shared.h
src/gpu_cuda.o: src/gpu_cuda_native_attention.h src/gpu_cuda_native_candidate_use.h src/gpu_cuda_native_candidates.h src/gpu_cuda_native_topk.h src/gpu_cuda_native_index.h src/gpu_cuda_compress.h src/gpu_cuda_native_rope.h src/gpu_cuda_exl3_decode.h src/gpu_cuda_exl3_fused.h src/gpu_cuda_native_moe.h src/gpu_cuda_exl3_fold.h src/gpu_cuda_hc4.h src/gpu_cuda_router.h

src/gpu_cuda_native_attention.o: src/gpu_cuda_native_attention.cu src/gpu_cuda_native_attention.h $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_native_index.o: src/gpu_cuda_native_index.cu src/gpu_cuda_native_index.h $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_native_topk.o: src/gpu_cuda_native_topk.cu src/gpu_cuda_native_topk.h $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_native_candidates.o: src/gpu_cuda_native_candidates.cu src/gpu_cuda_native_candidates.h $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_native_candidate_use.o: src/gpu_cuda_native_candidate_use.cu src/gpu_cuda_native_candidate_use.h $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_compress.o: src/gpu_cuda_compress.cu src/gpu_cuda_compress.h src/gpu_cuda_compress_kernels.cuh $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 --use_fast_math -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

# Frequency construction preserves Torch pow/div/F32 operation boundaries.
# Keep it separate from the existing fast-math GPT-J rotation translation unit.
src/gpu_cuda_native_frequency.o: src/gpu_cuda_native_frequency.cu src/gpu_cuda_native_rope.h $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_native_rope.o: src/gpu_cuda_native_rope.cu src/gpu_cuda_native_rope.h src/gpu_cuda_native_rope_kernel.cuh $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 --use_fast_math --expt-extended-lambda -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_exl3_decode.o: src/gpu_cuda_exl3_decode.cu src/gpu_cuda_exl3_decode.h $(wildcard src/cuda_exl3_fused/*.cuh src/cuda_exl3_fused/*.h src/cuda_exl3_fused/quant/*.cuh) $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 --expt-relaxed-constexpr --use_fast_math -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_exl3_fused.o: src/gpu_cuda_exl3_fused.cu src/gpu_cuda_exl3_fused.h $(wildcard src/cuda_exl3_fused/*.cuh src/cuda_exl3_fused/*.h src/cuda_exl3_fused/quant/*.cuh) $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 --use_fast_math -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_native_moe.o: src/gpu_cuda_native_moe.cu src/gpu_cuda_native_moe.h $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 --use_fast_math -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_exl3_fold.o: src/gpu_cuda_exl3_fold.cu src/gpu_cuda_exl3_fold.h $(wildcard src/gpu_cuda_exl3_fold_*.cuh) $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 --use_fast_math -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_router.o: src/gpu_cuda_router.cu src/gpu_cuda_router.h src/gpu_cuda_router_det.cuh $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 --use_fast_math -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

src/gpu_cuda_hc4.o: src/gpu_cuda_hc4.cu src/gpu_cuda_hc4.h src/gpu_cuda_hc4_lanes.cuh $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 --use_fast_math -Iinclude $(CUDA_CFLAGS) $(CUDA_NVCCFLAGS) -c -o $@ $<

ifdef BN_ENABLE_CUDA
.PHONY: fetch-cuda-attention
fetch-cuda-attention: $(FA2_ROOT)/.bn-fa2-ready

$(FA2_ROOT)/.bn-fa2-ready: patches/fetch-cuda-attention.sh
	sh patches/fetch-cuda-attention.sh "$(FA2_ROOT)"

src/gpu_cuda_fa2.o: src/gpu_cuda_fa2.cu src/gpu_cuda_fa2.h $(FA2_ROOT)/.bn-fa2-ready $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 $(CUDA_NVCCFLAGS) --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
		-Xcompiler=-Wno-deprecated-declarations -DFLASH_NAMESPACE=bn_fa2 \
		-DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI \
		-DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_LOCAL -DFLASHATTENTION_DISABLE_UNEVEN_K \
		-I$(FA2_ROOT)/csrc/flash_attn/src -I$(FA2_ROOT)/csrc/cutlass/include \
		-MMD -MP -MF src/gpu_cuda_fa2.d -c -o $@ $<

$(MLA_ROOT)/.bn-mla-ready: patches/fetch-cuda-mla.sh
	sh patches/fetch-cuda-mla.sh "$(MLA_ROOT)"

src/gpu_cuda_mla_attention.o: src/gpu_cuda_mla_attention.cu src/gpu_cuda_mla_attention.h $(MLA_ROOT)/.bn-mla-ready $(BUILD_CONFIG_STAMP)
	$(NVCC) -O3 -std=c++17 $(CUDA_NVCCFLAGS) $(CUDA_CFLAGS) --expt-relaxed-constexpr --use_fast_math \
		-I$(MLA_ROOT)/include -I$(MLA_ROOT)/3rdparty/cutlass/include \
		-I$(MLA_ROOT)/3rdparty/cccl/libcudacxx/include -I$(MLA_ROOT)/3rdparty/cccl/cub -I$(MLA_ROOT)/3rdparty/cccl/thrust \
		-MMD -MP -MF src/gpu_cuda_mla_attention.d -c -o $@ $<

-include src/gpu_cuda_mla_attention.d
-include src/gpu_cuda_fa2.d
endif


# Objective-C pattern rule for Metal backend
src/%.o: src/%.m $(HEADERS) $(BUILD_CONFIG_STAMP)
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

# --- Tests ---
# --- Benchmark ---
BENCH_SRCS = bench/bench_kernels.c $(filter-out src/main.c, $(SRCS))
BENCH_OBJS = $(METAL_OBJS) $(CUDA_OBJS)

bench_kernels: $(BENCH_SRCS) $(BENCH_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

BENCH_PREFILL_SRCS = bench/bench_prefill.c $(filter-out src/main.c, $(SRCS))

bench_bf16_latency: bench/bench_bf16_latency.c bench/bf16_policy_io.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

bench_flash_offload: bench/bench_flash_offload.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ifeq ($(BN_ENABLE_CUDA),1)
bench_native_forward: bench/bench_native_forward.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)
endif

bench_bf16_concurrency: bench/bench_bf16_concurrency.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_bf16_arrivals: bench/bench_bf16_arrivals.c bench/bf16_policy_io.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

bench_bf16_batch: bench/bench_bf16_batch.c bench/bf16_policy_io.h bench/bf16_batch_schedule.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

bench_checkpoint_gdn: bench/bench_checkpoint_gdn.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_checkpoint_attention: bench/bench_checkpoint_attention.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_prefill: $(BENCH_PREFILL_SRCS) $(METAL_OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ifeq ($(BN_ENABLE_CUDA),1)
bench_prefill: CFLAGS += -I$(dir $(NVCC))../include
endif

# Scalar backend benchmark (no explicit SIMD kernels)
SCALAR_CFLAGS = -O3 -flto -Wall -Wextra -Wshadow -std=c11 -Iinclude -DBN_FORCE_SCALAR
ifeq ($(UNAME_S),Darwin)
ifneq ($(filter arm% aarch%,$(UNAME_M)),)
SCALAR_CFLAGS += -mcpu=apple-m1
endif
endif
ifeq ($(UNAME_S),Linux)
SCALAR_CFLAGS += -D_GNU_SOURCE
endif
SCALAR_QUANT_BACKEND = src/quant/x_quant_scalar.c src/quant/i2s_scalar.c \
    src/quant/tq2_scalar.c src/quant/tq1_scalar.c \
    src/quant/q8_scalar.c src/quant/q4_scalar.c src/quant/q4_1_scalar.c \
    src/quant/f32_scalar.c src/quant/f16_scalar.c src/quant/bf16_scalar.c \
    src/quant/q6k_scalar.c src/quant/q8k_scalar.c src/quant/q4k_scalar.c \
    src/quant/q5k_scalar.c src/quant/q3k_scalar.c src/quant/q2k_scalar.c \
    src/quant/iq4nl_scalar.c src/quant/iq4xs_scalar.c \
    src/quant/iq3xxs_scalar.c src/quant/iq3s_scalar.c \
    src/quant/iq2xxs_scalar.c src/quant/iq2xs_scalar.c src/quant/iq2s_scalar.c

SCALAR_TRANSFORMER_BACKEND = src/transformer/math_scalar.c src/transformer/math_backend.c \
    src/transformer/rmsnorm_scalar.c src/transformer/rmsnorm_backend.c \
    src/transformer/gqa_scalar.c src/transformer/gqa_tq_scalar.c \
    src/transformer/batched_attn_avx2.c src/transformer/batched_attn_scalar.c \
    src/transformer/logits_scalar.c src/transformer/logits_policy.c \
    src/transformer/logits_backend.c \
    src/transformer/logits.c src/transformer/cpu_policy.c \
    src/transformer/cpu_backend.c src/transformer/cpu.c \
    src/transformer/plan.c src/transformer/gpu_fallback.c \
    src/transformer/gpu_policy.c \
    src/transformer/gpu_resources.c \
    src/transformer/gpu_emit.c src/transformer/gpu.c src/transformer/bf16.c src/transformer/bf16_pool.c src/transformer/trellis.c \
    src/transformer/kv_backend.c src/transformer/kv.c \
    src/transformer/prefill_policy.c \
    src/transformer/prefill_backend.c src/transformer/prefill.c \
    src/transformer/ssm_scalar.c

SCALAR_SRCS = src/platform.c src/gguf.c $(QUANT_COMMON) $(SCALAR_QUANT_BACKEND) \
       src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c \
       $(SCALAR_TRANSFORMER_BACKEND) src/tokenizer.c $(SAMPLER_SRCS) \
       src/threadpool.c src/sh_arena.c src/sh_log.c src/bn_alloc.c src/session.c \
       src/prompt_cache.c src/generate_batch_plan.c src/generate_batch.c src/generate.c src/main.c

SCALAR_BENCH_SRCS = bench/bench_kernels.c $(filter-out src/main.c, $(SCALAR_SRCS))

bench_scalar: $(SCALAR_BENCH_SRCS)
	$(CC) $(SCALAR_CFLAGS) -o $@ $^ $(LDFLAGS)

bench_scalar_layers: SCALAR_CFLAGS += -DBN_BENCH_LAYERS
bench_scalar_layers: $(SCALAR_BENCH_SRCS)
	$(CC) $(SCALAR_CFLAGS) -o $@ $^ $(LDFLAGS)

bitnet_scalar: $(SCALAR_SRCS)
	$(CC) $(SCALAR_CFLAGS) -o $@ $^ $(LDFLAGS)

ifeq ($(BN_ENABLE_WEBGPU),1)
bench_webgpu: $(BENCH_SRCS) $(BENCH_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
else
bench_webgpu:
	@echo "bench_webgpu skipped: set BN_ENABLE_WEBGPU=1"
endif

# Per-layer timing build (BN_BENCH_LAYERS)
bench_layers: CFLAGS += -DBN_BENCH_LAYERS
bench_layers: $(BENCH_SRCS) $(BENCH_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: test_mhc debug asan bench bench_suite bench_llama_compare bench_llama_topk bench_llama_topk_server bench_cuda_compare bench_qwen_cuda_matrix bench_gemma4_cuda_matrix bench_kernels_run bitnet_scalar bitnet_avx2 bitnet_avx512 build_cpu_parity_backends bench_scalar bench_scalar_layers bench_avx2 bench_avx512 bench_webgpu bench_layers test test_architecture test_backend_matrix test_model_matrix test_cpu_parity test_cpu_parity_required test_qwen_cpu_parity test_gemma4_cpu_parity test_gguf test_quant test_quant_avx2 test_quant_scalar test_avx512_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_prefill test_kv_f16 test_q2k test_ssm test_gguf_fuzz test_moe test_qwen36 test_qwen36_cuda test_gemma4 test_gemma4_avx2 test_gemma4_webgpu test_gemma4_cuda test_gemma4_backend_matrix test_generate test_session test_prompt_cache test_turboquant test_gpu_graph_ir test_gpu_backend test_cuda_backend test_gpu_wgpu test_gpu_validate test_coherence pgo avx2-check avx512-check check-cpu-parity-tools check-cpu-parity-fixtures check-cpu-parity-remote-fixtures fetch-cpu-parity-fixtures fetch-qwen36-sparse-fixture fetch-gemma4-fixtures fetch-wgpu clean

bench: $(MAIN_TARGET)
	./bench/bench_suite.sh

bench_suite: $(MAIN_TARGET)
	./bench/bench_suite.sh

bench_llama_compare: bench_avx2
	./bench/compare_llama.sh

bench_cuda_compare: bench_kernels
	./bench/cuda_compare.sh

bench_qwen_cuda_matrix: bitnet test_coherence bench_kernels
	./bench/qwen_cuda_matrix.sh

bench_gemma4_cuda_matrix: bitnet test_coherence bench_kernels
	./bench/gemma4_cuda_matrix.sh

LLAMA_TOPK_MODEL ?= models/qwen2.5-3b-instruct-q4_0.gguf
LLAMA_TOPK_ARGS ?= --metal --llama-metal --flash --maxseq 512 --gpu-max-storage-binding-mb 4096 --top-k 10 --min-overlap 3 --benchmark --bench-runs 3
LLAMA_TOPK_PORT ?= 0
bench_llama_topk: bitnet
	python3 test/compare_llama_topk.py $(LLAMA_TOPK_MODEL) $(LLAMA_TOPK_ARGS)

bench_llama_topk_server: bitnet
	python3 test/run_llama_topk_gate.py $(LLAMA_TOPK_MODEL) --port $(LLAMA_TOPK_PORT) -- $(LLAMA_TOPK_ARGS)

bench_kernels_run: bench_kernels

GEMMA4_FETCH_CASE ?= all
fetch-gemma4-fixtures:
	./scripts/fetch_gemma4_fixtures.sh $(GEMMA4_FETCH_CASE)

fetch-qwen36-sparse-fixture:
	./scripts/fetch_qwen36_sparse_fixture.sh

fetch-cpu-parity-fixtures: check-cpu-parity-remote-fixtures fetch-qwen36-sparse-fixture fetch-gemma4-fixtures

check-cpu-parity-fixtures:
	./scripts/check_cpu_parity_fixtures.sh

check-cpu-parity-tools:
	./scripts/check_cpu_parity_tools.sh

check-cpu-parity-remote-fixtures:
	./scripts/check_cpu_parity_remote_fixtures.sh

test: test_platform test_backend_session_rows test_backend_model_rows test_int4_rows test_fp8 test_json test_gpu_moe_cache test_trellis test_exl3 test_safetensors test_mhc test_bf16_capacity_pool test_bf16_wall_clock_verifier test_bf16_arrival_workload test_generate_batch test_generate_batch_plan test_bf16_pool test_bf16_mixed_reference test_bf16_batch_schedule test_bench_bf16_policy test_bf16_serving_verifiers test_architecture test_cpu_parity_harness test_bf16_resume_reference_harness test_prefill_reference_harness test_gguf test_quant test_quant_scalar test_tokenizer test_transformer test_threadpool test_safety test_arena test_ssm test_gguf_fuzz test_moe test_qwen36 test_prefill_profile test_gemma4 test_generate test_session test_prompt_cache test_turboquant test_gpu_graph_ir test_gpu_backend

.PHONY: test_platform
test_platform: test/test_platform.c src/platform.c include/platform.h
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS) && ./$@

test_bf16_pool: test/test_bf16_pool.c src/transformer/bf16_pool.c src/transformer/bf16_internal.h
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS) && ./$@

.PHONY: test_bf16_mixed_reference
test_bf16_mixed_reference:
	python3 test/test_bf16_mixed_reference.py

.PHONY: test_bench_bf16_policy
test_bench_bf16_policy:
	python3 test/test_bench_bf16_policy.py

.PHONY: test_bf16_serving_verifiers
test_bf16_serving_verifiers:
	python3 test/test_bf16_serving_verifiers.py

.PHONY: test_prefill_profile
test_glm_cli_profile: bitnet
	python3 test/test_glm_cli_profile.py ./bitnet

test_prefill_profile: test_qwen36
	python3 test/test_prefill_profile.py ./test_qwen36

.PHONY: test_cpu_parity_harness
test_cpu_parity_harness:
	bash test/test_cpu_parity_harness.sh

.PHONY: test_bf16_resume_reference_harness
.PHONY: test_bf16_batch_schedule
test_bf16_batch_schedule: test/test_bf16_batch_schedule.c bench/bf16_batch_schedule.h
	$(CC) $(CFLAGS) -o $@ test/test_bf16_batch_schedule.c $(LDFLAGS) && ./$@

test_bf16_resume_reference_harness:
	python3 test/test_bf16_resume_reference.py

.PHONY: test_prefill_reference_harness
test_prefill_reference_harness:
	python3 test/test_prefill_reference_harness.py

# Optional real-model independent-reference replay; build only, needs CLI inputs.
test_prefill_reference: test/test_prefill_reference.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_architecture: test_backend_matrix test_model_matrix

test_backend_matrix:
	./test/backend_matrix.sh

test_model_matrix: test_coherence
	./test/model_matrix.sh

test_cpu_parity: bitnet bitnet_scalar build_cpu_parity_backends
	./test/cpu_parity.sh

test_cpu_parity_required:
	$(MAKE) check-cpu-parity-tools
	$(MAKE) check-cpu-parity-remote-fixtures
	$(MAKE) REQUIRE_MODELS=1 check-cpu-parity-fixtures
	$(MAKE) bitnet bitnet_scalar
	$(MAKE) build_cpu_parity_backends
	REQUIRE_MODELS=1 QWEN_CPU_PARITY_LEVEL=$${QWEN_CPU_PARITY_LEVEL:-full} GEMMA4_CPU_PARITY_LEVEL=$${GEMMA4_CPU_PARITY_LEVEL:-full} ./test/cpu_parity.sh

test_qwen_cpu_parity: bitnet bitnet_scalar build_cpu_parity_backends
	./test/qwen_cpu_parity.sh

test_gemma4_cpu_parity: bitnet bitnet_scalar build_cpu_parity_backends
	./test/gemma4_cpu_parity.sh

.PHONY: test_gpu_moe_cache
test_gpu_moe_cache: test/test_gpu_moe_cache.c src/gpu_moe_cache.c src/sh_log.c src/platform.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

.PHONY: test_trellis
test_trellis: test/test_trellis.c src/trellis.c src/safetensors.c src/platform.c $(QUANT_SRCS) src/threadpool.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

.PHONY: test_exl3
test_exl3: test/test_exl3.c src/quant/exl3.c src/quant/fp16.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

.PHONY: test_safetensors
test_safetensors: test/test_safetensors.c src/safetensors.c src/platform.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_gguf: test/test_gguf.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_quant: test/test_quant.c $(QUANT_SRCS) src/threadpool.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_quant_scalar: test/test_quant_scalar.c src/quant/fp16.c src/quant/prepared.c src/quant/q4_scalar.c src/sh_arena.c
	$(CC) $(SCALAR_CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_avx512_quant: test/test_avx512_quant.c src/quant/q8_avx512_vnni.c src/quant/q4_avx512_vnni.c \
                   src/quant/q4k_avx512_vnni.c src/quant/q5k_avx512.c src/quant/q6k_avx512_vnni.c \
                   src/quant/q8_avx2.c src/quant/q4_avx2_4row.c \
                   src/quant/q4k_avx2_4row.c src/quant/q5k_avx2.c src/quant/q6k_avx2_4row.c \
                   src/quant/q8_scalar.c src/quant/q4k_scalar.c src/quant/q5k_scalar.c src/quant/q6k_scalar.c src/quant/fp16.c
ifeq ($(UNAME_M),x86_64)
	$(CC) $(CFLAGS) -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mavx2 -mfma -mf16c -o $@ $^ $(LDFLAGS) && ./$@
else
	@echo "test_avx512_quant skipped: x86_64 only"
endif

test_tokenizer: test/test_tokenizer.c src/safetensors.c src/json.c src/tokenizer.c src/model_arch.c src/gguf.c src/platform.c src/sh_log.c src/sh_arena.c src/threadpool.c $(QUANT_SRCS)
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS) && ./$@

test_transformer: test/test_transformer.c $(TRANSFORMER_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
                  src/gguf.c $(QUANT_SRCS) src/platform.c src/tokenizer.c src/threadpool.c \
                  src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS) && ./$@

test_transformer: test/glm_anatomy_reference.h test/glm_anatomy_checks.h include/model_anatomy.h

test_threadpool: test/test_threadpool.c src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_safety: test/test_safety.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c $(SAMPLER_SRCS) src/threadpool.c \
             src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_arena: test/test_arena.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

SSM_BACKEND = $(filter src/transformer/ssm_%, $(TRANSFORMER_BACKEND))
test_ssm: test/test_ssm.c src/transformer/ssm_scalar.c \
          src/transformer/math_scalar.c $(SSM_BACKEND)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_mhc: test/test_mhc.c src/transformer/math_scalar.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_gguf_fuzz: test/test_gguf_fuzz.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_moe: test/native_trellis_owner_checks.h test/native_trellis_store_checks.h test/native_moe_plan_checks.h test/native_moe_plan_reference.h test/native_router_checks.h test/native_parameter_checks.h test/native_core_checks.h test/native_f16_layout_checks.h test/native_numerics_checks.h test/test_moe.c $(MOE_SRCS) src/turboquant.c $(MODEL_SRCS) src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) \
          src/gguf.c $(QUANT_SRCS) src/platform.c src/tokenizer.c src/threadpool.c \
          src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS) && ./$@

test_qwen36: test/ple_ids_reference.h

test_qwen36: test/test_qwen36.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS) && ./$@

test_gemma4: test/test_gemma4.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_gemma4_avx2: test/test_gemma4.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
ifeq ($(UNAME_M),x86_64)
	$(CC) $(CFLAGS) -mavx2 -mfma -mf16c -o $@ $^ $(LDFLAGS) && ./$@
else
	@echo "test_gemma4_avx2 skipped: requires x86_64 host"
endif

test_gemma4_webgpu: test/test_gemma4.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c $(WEBGPU_SRCS)
ifdef BN_ENABLE_WEBGPU
	$(CC) $(CFLAGS) -DBN_GEMMA4_TEST_WEBGPU -o $@ $^ $(LDFLAGS) && ./$@
else
	@echo "test_gemma4_webgpu skipped: set BN_ENABLE_WEBGPU=1"
endif

ifdef BN_ENABLE_CUDA
test_qwen36_cuda: test/test_qwen36.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c $(CUDA_OBJS)
	$(CC) $(CFLAGS) -DBN_QWEN36_TEST_CUDA -o $@ $^ $(LDFLAGS) && ./$@

test_gemma4_cuda: test/test_gemma4.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c $(CUDA_OBJS)
	$(CC) $(CFLAGS) -DBN_GEMMA4_TEST_CUDA -o $@ $^ $(LDFLAGS) && ./$@
else
test_qwen36_cuda:
	@echo "test_qwen36_cuda skipped: set BN_ENABLE_CUDA=1"

test_gemma4_cuda:
	@echo "test_gemma4_cuda skipped: set BN_ENABLE_CUDA=1"
endif

test_gemma4_backend_matrix: test_gemma4 test_gemma4_avx2 test_gemma4_webgpu test_gemma4_cuda

test_generate: test/test_generate.c src/generate.c src/bn_alloc.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
               src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c $(SAMPLER_SRCS) src/threadpool.c \
               src/sh_arena.c src/sh_log.c src/session.c
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS) && ./$@

test_q2k: test/test_q2k.c $(QUANT_SRCS) src/threadpool.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_moe: test/native_index_session_checks.h test/native_session_checks.h test/native_executor_checks.h
test_session: test/test_session.c src/session.c src/bn_alloc.c src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
              src/gguf.c $(QUANT_SRCS) src/platform.c src/tokenizer.c src/threadpool.c \
              src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_prompt_cache: test/test_prompt_cache.c src/prompt_cache.c src/session.c src/bn_alloc.c src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
                   src/gguf.c $(QUANT_SRCS) src/platform.c src/tokenizer.c src/threadpool.c \
                   src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_turboquant: test/test_turboquant.c src/turboquant.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_gpu_graph_ir: test/test_gpu_graph_ir.c $(QUANT_SRCS) src/threadpool.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_gpu_backend: test/glm_mla_layout_checks.h test/glm_indexer_layout_checks.h test/glm_moe_work_checks.h test/glm_moe_layout_checks.h test/glm_dense_ffn_checks.h test/glm_mhc_layout_checks.h test/glm_kda_layout_checks.h include/model_anatomy.h
test_gpu_backend: test/test_gpu_backend.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
                  src/gguf.c src/platform.c src/tokenizer.c src/threadpool.c \
                  src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/sh_arena.c src/sh_log.c \
                  src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS) && ./$@

ifdef BN_ENABLE_CUDA
# Materialized mHC -> normalized KDA -> mHC attention transition replay.
test_cuda_glm_context: test/test_cuda_glm_context.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

test_cuda_flash_group: test/test_cuda_flash_group.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -Wl,--wrap=bn_moe_host_expert_acquire

test_cuda_glm_group: test/test_cuda_glm_group.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -Wl,--wrap=bn_moe_host_expert_acquire

test_cuda_glm_request: test/test_cuda_glm_request.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

test_cuda_glm_mhc_kda: test/test_cuda_glm_mhc_kda.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

# Actual pre-indexer -> paged index cache append/gather composition.
test_cuda_glm_mla_block: test/test_cuda_glm_mla_block.c test/test_cuda_glm_preindexer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_mla_block.c $(filter %.o,$^) $(LDFLAGS)

test_cuda_glm_indexed_mla: test/test_cuda_glm_indexed_mla.c test/test_cuda_glm_preindexer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_indexed_mla.c $(filter %.o,$^) $(LDFLAGS)

test_cuda_glm_mla_attention_kernel: test/test_cuda_glm_mla_attention_kernel.c test/test_cuda_glm_indexer.c test/cuda_glm_mla_attention_reference.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_mla_attention_kernel.c $(filter %.o,$^) $(LDFLAGS) && ./$@

test_cuda_backend: test/cuda_f16_grouped_output_checks.h test/cuda_native_attention_checks.h test/cuda_native_attention_reference.h test/cuda_native_candidate_use_checks.h test/cuda_native_candidate_use_reference.h test/cuda_native_candidates_checks.h test/cuda_native_candidates_reference.h test/cuda_native_topk_checks.h test/cuda_native_topk_reference.h test/cuda_native_index_checks.h test/cuda_native_index_reference.h test/cuda_native_swa_ring_checks.h test/cuda_native_swa_ring_reference.h test/cuda_native_paged_latents_checks.h test/cuda_native_paged_latents_reference.h test/cuda_native_latent_keys_checks.h test/cuda_native_latent_keys_reference.h test/cuda_native_compress_checks.h test/cuda_native_compress_reference.h test/cuda_native_qkv_checks.h test/cuda_native_qkv_reference.h test/cuda_decode_exl3_checks.h test/cuda_decode_exl3_reference.h test_cuda_glm_mla_attention_kernel

test_cuda_glm_mla_attention: test/test_cuda_glm_mla_attention.c test/test_cuda_glm_preindexer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_mla_attention.c $(filter %.o,$^) $(LDFLAGS)

test_cuda_glm_mla_cache_chain: test/test_cuda_glm_mla_cache_chain.c test/test_cuda_glm_preindexer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_mla_cache_chain.c $(filter %.o,$^) $(LDFLAGS)

test_cuda_glm_mla_cache: test/test_cuda_glm_mla_cache.c test/cuda_glm_mla_cache_reference.h test/test_cuda_glm_indexer.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_mla_cache.c $(filter %.o,$^) $(LDFLAGS)

test_cuda_glm_mla_projection: test/test_cuda_glm_mla_projection.c test/test_cuda_glm_preindexer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_mla_projection.c $(filter %.o,$^) $(LDFLAGS)

test_cuda_glm_indexer_decode: test/test_cuda_glm_indexer_decode.c test/test_cuda_glm_preindexer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_indexer_decode.c $(filter %.o,$^) $(LDFLAGS)

test_cuda_glm_indexer_select: test/test_cuda_glm_indexer_select.c test/test_cuda_glm_preindexer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_indexer_select.c $(filter %.o,$^) $(LDFLAGS)

test_cuda_glm_indexer_scores: test/test_cuda_glm_indexer_scores.c test/test_cuda_glm_preindexer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_indexer_scores.c $(filter %.o,$^) $(LDFLAGS)

test_cuda_glm_indexer_cache: test/test_cuda_glm_indexer_cache.c test/test_cuda_glm_preindexer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_indexer_cache.c $(filter %.o,$^) $(LDFLAGS)

# Actual GLM pre-indexer projections, normalization, FP8 and resumed tail.
test_cuda_glm_preindexer: test/test_cuda_glm_preindexer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

# Actual GLM raw-F32 router/shared and complete streamed MoE replays.
test_cuda_glm_router: test/test_cuda_glm_router.c test/glm_router_edge_checks.h src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

test_cuda_glm_moe_layer: test/test_cuda_glm_moe_layer.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

# Final deferred mHC/mean/RMS/head from independent checkpoint tuples.
test_cuda_glm_final_output: test/test_cuda_glm_final_output.c test/test_cuda_glm_deferred_kda.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $< $(filter-out src/main.o,$(OBJS)) $(LDFLAGS)

# All main-model immutable GLM resources, with aggregate budget rollback.
test_cuda_glm_model_resources: test/test_cuda_glm_model_resources.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Complete KDA/MoE from the independent deferred boundary.
test_cuda_glm_kda_moe_block: test/test_cuda_glm_kda_moe_block.c test/test_cuda_glm_deferred_kda.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $< $(filter-out src/main.o,$(OBJS)) $(LDFLAGS)

# Actual GLM routed expert execution with bounded banks and supplied routes.
test_cuda_glm_routed_ffn: test/test_cuda_glm_routed_ffn.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Token embedding through the first three dense decoder blocks.
test_cuda_glm_dense_prefix: test/test_cuda_glm_dense_prefix.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

# Actual checkpoint dense FFN replay.
test_cuda_glm_dense_ffn: test/test_cuda_glm_dense_ffn.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

# Complete previous-FFN post -> attention -> following FFN-pre replay.
test_cuda_glm_deferred_kda: test/test_cuda_glm_deferred_kda.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

# Default deferred/fused mHC replay across decode and prefill dispatch regimes.
test_cuda_glm_mhc_fused: test/test_cuda_glm_mhc_fused.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Optional checkpoint mHC replay; independent reference files are required.
test_cuda_glm_mhc: test/test_cuda_glm_mhc.c include/model_anatomy.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)

# Optional actual-checkpoint KDA replay; requires independently captured files.
test_cuda_glm_kda_layer: test/test_cuda_glm_kda_layer.c src/transformer/bf16_internal.h include/model_anatomy.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter-out %.h,$^) $(LDFLAGS)
endif

ifdef BN_ENABLE_CUDA
# Build only: requires the pinned model and independent rotary-cache fixture.
test_bf16_runtime: test/test_bf16_runtime.c src/transformer/bf16_internal.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS)

# Build only: fixed packed geometry, explicit original-reference norm policy.
.PHONY: reference_cuda_bf16_packed
reference_cuda_bf16_packed: test/reference_cuda_bf16_packed.c bench/bf16_policy_io.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -Ibench -o $@ $(filter %.c %.o,$^) $(LDFLAGS)

.PHONY: reference_cuda_bf16_mixed
reference_cuda_bf16_mixed: test/reference_cuda_bf16_mixed.c bench/bf16_policy_io.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -Ibench -o $@ $(filter %.c %.o,$^) $(LDFLAGS)

.PHONY: test_cuda_fp16
test_cuda_fp16: test/test_cuda_fp16.cu src/quant/fp16_cuda.cuh
	$(NVCC) -O3 -std=c++11 -Isrc $(CUDA_NVCCFLAGS) -o $@ $< && ./$@

.PHONY: test_cuda_moe_group
test_cuda_moe_group: test_cuda_q8_mma test/reference_cuda_bf16_moe_group.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS) && ./$@ --synthetic

.PHONY: test_cuda_q8_mma
test_cuda_q8_mma: test_cuda_fp16 test/test_cuda_q8_mma.cu src/quant/q8_bf16_mma_cuda.cuh src/quant/int8_mma_cuda.cuh
	$(NVCC) -O3 -std=c++11 -Isrc -Iinclude $(CUDA_NVCCFLAGS) -o $@ test/test_cuda_q8_mma.cu && ./$@

.PHONY: test_cuda_moe_bank
test_cuda_moe_bank: test_cuda_moe_group test/test_cuda_moe_bank.c test/cuda_moe_shared_bank_checks.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS) && ./$@

# Chain GPU tests so make -j cannot run these probes concurrently.
.PHONY: test_cuda_expert_cache
test_cuda_expert_cache: test_cuda_moe_bank test/test_cuda_expert_cache.c test/cuda_expert_cache_group_checks.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS) && ./$@

# The resident GLM indexer probes continue the serial CUDA test chain.
.PHONY: test_cuda_glm_indexer
test_cuda_glm_indexer: test_cuda_expert_cache test/test_cuda_glm_indexer.c test/cuda_glm_indexer_reference.h test/cuda_glm_layernorm_reference.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS) && ./$@

.PHONY: test_cuda_glm_fp8_cache
test_cuda_glm_fp8_cache: test_cuda_glm_indexer test/test_cuda_glm_fp8_cache.c test/test_cuda_glm_indexer.c test/cuda_glm_fp8_cache_reference.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_fp8_cache.c $(filter %.o,$^) $(LDFLAGS) && ./$@

.PHONY: test_cuda_glm_score
test_cuda_glm_score: test_cuda_glm_fp8_cache test/test_cuda_glm_score.c test/test_cuda_glm_indexer.c test/cuda_glm_score_reference.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_score.c $(filter %.o,$^) $(LDFLAGS) && ./$@

.PHONY: test_cuda_glm_select
test_cuda_glm_select: test_cuda_glm_score test/test_cuda_glm_select.c test/test_cuda_glm_indexer.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_select.c $(filter %.o,$^) $(LDFLAGS) && ./$@

.PHONY: test_cuda_glm_paged_score
test_cuda_glm_paged_score: test_cuda_glm_select test/test_cuda_glm_paged_score.c test/test_cuda_glm_indexer.c test/cuda_glm_paged_score_reference.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ test/test_cuda_glm_paged_score.c $(filter %.o,$^) $(LDFLAGS) && ./$@

.PHONY: test_cuda_shared_math
test_cuda_shared_math: test_cuda_glm_paged_score test/test_cuda_shared_math.c test/cuda_bf16_shared_math_reference.h test/cuda_bf16_silu_profiles_reference.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS) && ./$@

.PHONY: test_cuda_silu_profiles
test_cuda_silu_profiles: test/cuda_compiled_silu_reference.h
test_cuda_silu_profiles: test/reference_cuda_silu_profiles.cu src/gpu_cuda_activation.cuh test/cuda_bf16_silu_profiles_reference.h test_cuda_shared_math
	$(NVCC) -O3 -std=c++11 -Isrc -Itest $(CUDA_NVCCFLAGS) -o $@ $< && ./$@

.PHONY: test_cuda_attention_gate
test_cuda_attention_gate: test/test_cuda_attention_gate.cu src/gpu_cuda_attention_gate.cuh test_cuda_silu_profiles
	$(NVCC) -O3 -std=c++11 -Isrc $(CUDA_NVCCFLAGS) -o $@ $< && ./$@

test_cuda_backend: test_cuda_attention_gate

test_cuda_backend: test/cuda_native_ple_checks.h test/cuda_native_ple_reference.h
test_cuda_backend: test/cuda_f16_ffn_checks.h test/cuda_f16_ffn_reference.h test/cuda_fused_exl3_checks.h test/cuda_fused_exl3_reference.h test/cuda_native_moe_checks.h test/cuda_native_moe_reference.h test/native_moe_plan_reference.h test/cuda_folded_exl3_checks.h test/cuda_folded_exl3_reference.h test/cuda_native_router_checks.h test/cuda_native_router_reference.h test/cuda_native_embedding_head_checks.h test/cuda_native_embedding_head_reference.h test/cuda_native_hc4_checks.h test/cuda_native_hc4_reference.h test/cuda_native_norm_checks.h test/cuda_native_norm_reference.h test/cuda_raw_rows_checks.h test/cuda_f16_linear_checks.h test/cuda_f16_rows_checks.h test/cuda_exl3_checks.h test/cuda_exl3_ffn_checks.h
test_cuda_backend: test/cuda_f32_add_checks.h test/cuda_f32_add_reference.h test/cuda_fused_exl3_group_checks.h
test_cuda_backend: test/cuda_stacked_ffn_reference.h test/cuda_bf16_mixed_attention_reference.h
test_cuda_backend: test/cuda_compiled_silu_reference.h
test_cuda_backend: test/cuda_clamped_swiglu_checks.h test/cuda_clamped_swiglu_reference.h
test_cuda_backend: test/cuda_mhc_fused_checks.h test/cuda_mhc_fused_reference.h test/cuda_mhc_checks.h test/cuda_mhc_reference.h
test_cuda_backend: test/cuda_f32_weight_conv_reference.h
test_cuda_backend: test/cuda_kda_reference.h
test_cuda_backend: test/cuda_glm_output_norm_reference.h
test_cuda_backend: test/cuda_fused_persistent_norm_reference.h
test_cuda_backend: test/cuda_compiled_gated_norm_reference.h test/cuda_looped_add_norm_reference.h test/cuda_general_tiled_norm_reference.h
test_cuda_backend: test/cuda_bf16_projected_gdn_reference.h

test_cuda_backend: test/cuda_bf16_gdn_noncp_reference.h
test_cuda_backend: test/cuda_bf16_attention_batch_reference.h test/cuda_bf16_gdn_decode_profiles_reference.h
test_cuda_backend: test/cuda_bf16_gdn_cp_boundary_reference.h test/cuda_bf16_attention_compiled_gate_reference.h test/cuda_bf16_gdn_cp_log_reference.h test/cuda_bf16_gdn_cp_reference.h test/cuda_compiled_norm_reference.h test/cuda_bf16_add3_norm_reference.h
test_cuda_backend: test/cuda_bf16_gdn_cancellation_reference.h test/cuda_paged_kv_append_reference.h test/cuda_bf16_ple_gate_reference.h test/cuda_bf16_ple_conv_reference.h test/cuda_bf16_rope_reference.h test/cuda_bf16_f32_mean_reference.h test/cuda_bf16_wide_mean_reference.h test/cuda_bf16_grouped_mean_reference.h test/cuda_bf16_hc_reference.h test/cuda_bf16_qk_prep_reference.h test/cuda_bf16_attention_reference.h test/cuda_bf16_iq3_reference.h test/cuda_bf16_iq3_ffn_reference.h test/cuda_bf16_q4q6_reference.h test/cuda_bf16_q4q6_ffn_reference.h test/cuda_bf16_gdn_gated_norm_reference.h test/cuda_bf16_gdn_conv_chain_reference.h test/cuda_bf16_gdn_handoff_reference.h test/cuda_bf16_gdn_composed_reference.h test/cuda_bf16_gdn_prefill_reference.h test/cuda_bf16_gdn_prep_reference.h test/cuda_bf16_conv_reference.h test/cuda_bf16_residual_reference.h test/cuda_bf16_ffn_reference.h test/cuda_bf16_q5_reference.h test/cuda_bf16_q5_ffn_reference.h test/cuda_bf16_q8_reference.h test/cuda_bf16_q3_reference.h test/cuda_bf16_dispatch_reference.h test/cuda_bf16_matrix_reference.h test/cuda_bf16_norm_reference.h test/cuda_q8_partial_rows_reference.h test/cuda_f32_prepared_attention_reference.h test/cuda_shared_q8_matvec_reference.h test/cuda_gated_mmf_reference.h test/cuda_gated_mmf_fixture.h test/cuda_q5k_fused_gateup_reference.h test/cuda_q51_matvec_reference.h test/cuda_qk8_prefill_reference.h test/cuda_qk8_fixture.h test/cuda_q451_prefill_reference.h test/cuda_hc_fused_post_reference.h test/cuda_hc_fused_mean_reference.h test/cuda_gqa6_prefix_gemm_reference.h test/cuda_gqa6_prefix_attention_reference.h test/cuda_decode_attention_f32_reference.h test/cuda_router512_reference.h test/cuda_q451_routed_reference.h test/cuda_hc_mixer_reference.h test/cuda_hc_rmsnorm_reference.h test/cuda_signed_sqrt_gate_reference.h test/cuda_hc_combine_reference.h test/cuda_prefix_attention_reference.h test/cuda_window_attention_reference.h test/cuda_routed_reference.h test/cuda_mxfp4_routed_reference.h test/cuda_q5q6_routed_reference.h test/cuda_merged_routed_q4_reference.h test/cuda_norm_residual_reference.h test/cuda_prepared_v_reference.h test/cuda_routed_q4_reference.h test/cuda_gelu_prefill_reference.h test/cuda_expert_reduction_reference.h test/cuda_scaled_rmsnorm_reference.h test/cuda_mxfp4_reference.h test/cuda_mxfp4_fixture.h test/cuda_q3k_reference.h test/cuda_q8_reference.h test/cuda_iq4nl_reference.h test/cuda_iq4xs_reference.h test/cuda_iq3s_reference.h test/test_cuda_backend.c test/test_cuda_diagnostics.sh test/cuda_signed_mmq_reference.h test/cuda_standalone_ffn_fixture.h test/cuda_standalone_ffn_reference.h test/cuda_rmsnorm_batch_reference.h test/cuda_prepared_rope_reference.h test/cuda_signed_mmq_fixture.h test/cuda_q6_reference.h test/cuda_kquant_reference.h test/cuda_dense_projection_reference.h test/cuda_dense_prefill_reference.h test/cuda_decode_attention_reference.h test/cuda_gelu_reference.h test/cuda_ssm_reference.h $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS) && ./$@
	bash test/test_cuda_diagnostics.sh ./$@
else
test_cuda_glm_mla_attention_kernel:
	@echo "test_cuda_glm_mla_attention_kernel skipped: set BN_ENABLE_CUDA=1"

test_cuda_glm_paged_score:
	@echo "test_cuda_glm_paged_score skipped: set BN_ENABLE_CUDA=1"

test_cuda_glm_select:
	@echo "test_cuda_glm_select skipped: set BN_ENABLE_CUDA=1"

test_cuda_glm_score:
	@echo "test_cuda_glm_score skipped: set BN_ENABLE_CUDA=1"

test_cuda_glm_fp8_cache:
	@echo "test_cuda_glm_fp8_cache skipped: set BN_ENABLE_CUDA=1"

test_cuda_glm_indexer:
	@echo "test_cuda_glm_indexer skipped: set BN_ENABLE_CUDA=1"

test_cuda_moe_group:
	@echo "test_cuda_moe_group skipped: set BN_ENABLE_CUDA=1"

test_cuda_moe_bank:
	@echo "test_cuda_moe_bank skipped: set BN_ENABLE_CUDA=1"

test_cuda_shared_math:
	@echo "test_cuda_shared_math skipped: set BN_ENABLE_CUDA=1"

test_cuda_silu_profiles:
	@echo "test_cuda_silu_profiles skipped: set BN_ENABLE_CUDA=1"

test_cuda_attention_gate:
	@echo "test_cuda_attention_gate skipped: set BN_ENABLE_CUDA=1"

test_cuda_q8_mma:
	@echo "test_cuda_q8_mma skipped: set BN_ENABLE_CUDA=1"

test_bf16_runtime:
	@echo "test_bf16_runtime requires BN_ENABLE_CUDA=1"
	@false

test_cuda_backend:
	@echo "test_cuda_backend skipped: set BN_ENABLE_CUDA=1"
endif

test_e2e: test/test_e2e.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
          src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c $(SAMPLER_SRCS) src/threadpool.c \
          src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_prefill: test/test_prefill.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
              src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
              src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_prefill_output_rows: test/test_prefill_output_rows.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_prefill_continuation: test/test_prefill_continuation.c $(filter-out src/main.o,$(OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_f16: test/test_kv_f16.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c $(SAMPLER_SRCS) src/threadpool.c \
             src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

PGO_MODEL ?= models/bitnet-b1.58-2B-4T.gguf

# Auto-detect compiler for PGO: clang uses -fprofile-instr-*, gcc uses -fprofile-*
IS_GCC := $(shell $(CC) --version 2>&1 | grep -q 'gcc\|GCC' && echo 1 || echo 0)

pgo:
ifeq ($(IS_GCC),1)
	@echo "=== PGO (GCC) Step 1: Instrumented build ==="
	$(MAKE) clean
	$(MAKE) bitnet CFLAGS="$(CFLAGS) -fprofile-generate" LDFLAGS="$(LDFLAGS) -fprofile-generate"
	@echo "=== PGO (GCC) Step 2: Training run ==="
	./bitnet $(PGO_MODEL) -p "The meaning of life is" -n 128
	@echo "=== PGO (GCC) Step 3: Optimized rebuild ==="
	rm -f bitnet src/*.o src/quant/*.o src/transformer/*.o
	$(MAKE) bitnet CFLAGS="$(CFLAGS) -fprofile-use -fprofile-correction" LDFLAGS="$(LDFLAGS) -fprofile-use"
	@rm -f src/*.gcda src/quant/*.gcda src/transformer/*.gcda
	@echo "=== PGO build complete ==="
else
	@echo "=== PGO (Clang) Step 1: Instrumented build ==="
	$(MAKE) clean
	$(MAKE) bitnet CFLAGS="$(CFLAGS) -fprofile-instr-generate" LDFLAGS="$(LDFLAGS) -fprofile-instr-generate"
	@echo "=== PGO (Clang) Step 2: Training run ==="
	LLVM_PROFILE_FILE=default.profraw ./bitnet $(PGO_MODEL) -p "The meaning of life is" -n 128
	@echo "=== PGO (Clang) Step 3: Merge profile ==="
	xcrun llvm-profdata merge -output=default.profdata default.profraw
	@echo "=== PGO (Clang) Step 4: Optimized rebuild ==="
	rm -f bitnet src/*.o src/quant/*.o src/transformer/*.o
	$(MAKE) bitnet CFLAGS="$(CFLAGS) -fprofile-instr-use=default.profdata"
	@rm -f default.profraw default.profdata
	@echo "=== PGO build complete ==="
endif

AVX2_QUANT_SRCS = $(QUANT_COMMON) \
    src/quant/mxfp4_avx2.c \
    src/quant/x_quant_avx2.c \
    src/quant/rmsnorm_q8k_avx2.c \
    src/quant/i2s_avx2.c src/quant/i2s_avx2_4row.c src/quant/i2s_scalar.c \
    src/quant/tq2_avx2.c src/quant/tq2_scalar.c \
    src/quant/tq1_avx2.c src/quant/tq1_scalar.c \
    src/quant/q8_avx2.c src/quant/q8_scalar.c \
    src/quant/q4_avx2.c src/quant/q4_avx2_4row.c src/quant/q4_avx2_matmul.c src/quant/q4_scalar.c \
    src/quant/q4_1_avx2.c src/quant/q4_1_scalar.c \
    src/quant/f32_avx2.c src/quant/f16_avx2.c \
    src/quant/bf16_avx2.c src/quant/bf16_scalar.c \
    src/quant/q6k_avx2.c src/quant/q6k_avx2_sdot.c src/quant/q6k_avx2_4row.c src/quant/q6k_scalar.c \
    src/quant/q8k_avx2.c src/quant/q8k_scalar.c \
    src/quant/q4k_avx2.c src/quant/q4k_avx2_sdot.c src/quant/q4k_avx2_4row.c src/quant/q4k_scalar.c \
    src/quant/q5k_avx2.c src/quant/q5k_scalar.c \
    src/quant/q3k_avx2.c src/quant/q3k_scalar.c \
    src/quant/q2k_avx2.c src/quant/q2k_scalar.c \
    src/quant/iq4nl_avx2.c src/quant/iq4nl_scalar.c \
    src/quant/iq4xs_avx2.c src/quant/iq4xs_scalar.c \
    src/quant/iq3xxs_avx2.c src/quant/iq3xxs_scalar.c \
    src/quant/iq3s_avx2.c src/quant/iq3s_scalar.c \
    src/quant/iq2xxs_avx2.c src/quant/iq2xxs_scalar.c \
    src/quant/iq2xs_avx2.c src/quant/iq2xs_scalar.c \
    src/quant/iq2s_avx2.c src/quant/iq2s_scalar.c

AVX512_QUANT_SRCS = $(AVX2_QUANT_SRCS) \
    src/quant/q8_avx512_vnni.c src/quant/q4_avx512_vnni.c \
    src/quant/q4k_avx512_vnni.c src/quant/q5k_avx512.c \
    src/quant/q6k_avx512_vnni.c

AVX2_TRANSFORMER_BACKEND = src/transformer/math_scalar.c src/transformer/math_backend.c \
    src/transformer/rmsnorm_avx2.c src/transformer/rmsnorm_scalar.c \
    src/transformer/rmsnorm_backend.c \
    src/transformer/gqa_avx2.c src/transformer/gqa_scalar.c \
    src/transformer/gqa_tq_scalar.c \
    src/transformer/batched_attn_avx2.c src/transformer/batched_attn_scalar.c \
    src/transformer/logits_avx2.c src/transformer/logits_scalar.c \
    src/transformer/logits_policy.c \
    src/transformer/logits_backend.c src/transformer/logits.c \
    src/transformer/cpu_policy.c \
    src/transformer/cpu_backend.c src/transformer/cpu.c \
    src/transformer/plan.c src/transformer/gpu_fallback.c \
    src/transformer/gpu_policy.c \
    src/transformer/gpu_resources.c \
    src/transformer/gpu_emit.c src/transformer/gpu.c src/transformer/bf16.c src/transformer/bf16_pool.c src/transformer/trellis.c \
    src/transformer/kv_backend.c src/transformer/kv.c \
    src/transformer/prefill_policy.c \
    src/transformer/prefill_backend.c src/transformer/prefill.c \
    src/transformer/ssm_avx2.c src/transformer/ssm_scalar.c

AVX2_SRCS = src/platform.c src/gguf.c $(AVX2_QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
            src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(AVX2_TRANSFORMER_BACKEND) src/tokenizer.c $(SAMPLER_SRCS) \
            src/threadpool.c src/sh_arena.c src/sh_log.c src/bn_alloc.c src/session.c src/generate.c

AVX2_BIN_SRCS = $(AVX2_SRCS) src/prompt_cache.c src/main.c
AVX2_BIN_FLAGS = -mavx2 -mfma -mf16c -O3 -Wall -Wextra -Wshadow -std=c11 -Iinclude
ifeq ($(UNAME_S),Linux)
AVX2_BIN_FLAGS += -D_GNU_SOURCE
endif

AVX2_CHECK_FLAGS = -mavx2 -mfma -mf16c -O3 -Wall -Wextra -Wshadow -std=c11 -Iinclude -fsyntax-only
ifeq ($(UNAME_S),Linux)
AVX2_CHECK_FLAGS += -D_GNU_SOURCE
endif

avx2-check:
ifeq ($(UNAME_M),x86_64)
	$(CC) $(AVX2_CHECK_FLAGS) $(AVX2_SRCS)
else
	$(CC) -target x86_64-apple-darwin $(AVX2_CHECK_FLAGS) $(AVX2_SRCS)
endif

bitnet_avx2: $(AVX2_BIN_SRCS)
ifeq ($(UNAME_M),x86_64)
	$(CC) $(AVX2_BIN_FLAGS) -o $@ $^ $(LDFLAGS)
else
	@echo "bitnet_avx2 skipped: requires x86_64 host"
endif

# Optional real-model replay with AVX2-only CPU execution; build only.
test_prefill_reference_avx2: test/test_prefill_reference.c $(filter-out src/main.c,$(AVX2_BIN_SRCS))
ifeq ($(UNAME_M),x86_64)
	$(CC) $(AVX2_BIN_FLAGS) -o $@ $^ $(LDFLAGS)
else
	@echo "test_prefill_reference_avx2 skipped: requires x86_64 host"
endif

test_quant_avx2: test/test_quant.c $(AVX2_QUANT_SRCS) src/threadpool.c src/sh_arena.c
ifeq ($(UNAME_M),x86_64)
	$(CC) $(AVX2_BIN_FLAGS) -o $@ $^ $(LDFLAGS) && ./$@
else
	@echo "test_quant_avx2 skipped: requires x86_64 host"
endif

AVX512_CHECK_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi \
    -mavx2 -mfma -mf16c -O3 -Wall -Wextra -Wshadow -std=c11 -Iinclude -fsyntax-only
ifeq ($(UNAME_S),Linux)
AVX512_CHECK_FLAGS += -D_GNU_SOURCE
endif

AVX512_SRCS = src/platform.c src/gguf.c $(AVX512_QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
            src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(AVX2_TRANSFORMER_BACKEND) src/tokenizer.c $(SAMPLER_SRCS) \
            src/threadpool.c src/sh_arena.c src/sh_log.c src/bn_alloc.c src/session.c src/generate.c

AVX512_BIN_SRCS = $(AVX512_SRCS) src/prompt_cache.c src/main.c
AVX512_BIN_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi \
    -mavx2 -mfma -mf16c -O3 -Wall -Wextra -Wshadow -std=c11 -Iinclude
ifeq ($(UNAME_S),Linux)
AVX512_BIN_FLAGS += -D_GNU_SOURCE
endif

avx512-check:
ifeq ($(UNAME_M),x86_64)
	$(CC) $(AVX512_CHECK_FLAGS) $(AVX512_SRCS)
else
	@echo "avx512-check skipped: x86_64 only"
endif

bitnet_avx512: $(AVX512_BIN_SRCS)
ifeq ($(UNAME_M),x86_64)
	$(CC) $(AVX512_BIN_FLAGS) -o $@ $^ $(LDFLAGS)
else
	@echo "bitnet_avx512 skipped: requires x86_64 host"
endif

AVX2_BENCH_SRCS = bench/bench_kernels.c $(AVX2_SRCS)
AVX512_BENCH_SRCS = bench/bench_kernels.c $(AVX512_SRCS)

bench_avx2: $(AVX2_BENCH_SRCS)
ifeq ($(UNAME_M),x86_64)
	$(CC) $(AVX2_BIN_FLAGS) -o $@ $^ $(LDFLAGS)
else
	@echo "bench_avx2 skipped: requires x86_64 host"
endif

bench_avx512: $(AVX512_BENCH_SRCS)
ifeq ($(UNAME_M),x86_64)
	$(CC) $(AVX512_BIN_FLAGS) -o $@ $^ $(LDFLAGS)
else
	@echo "bench_avx512 skipped: requires x86_64 host"
endif

build_cpu_parity_backends:
	@backends="$${CPU_PARITY_BACKENDS:-}"; \
	qwen_backends="$${QWEN_CPU_PARITY_BACKENDS:-$$backends}"; \
	gemma_backends="$${GEMMA4_CPU_PARITY_BACKENDS:-$$backends}"; \
	selected=",$$qwen_backends,$$gemma_backends,"; \
	if [ "$$selected" = ",,," ]; then \
		selected=",neon,scalar,avx2,avx512,"; \
	fi; \
	case "$$selected" in *,avx2,* ) $(MAKE) bitnet_avx2 ;; esac; \
	case "$$selected" in *,avx512,* ) $(MAKE) bitnet_avx512 ;; esac

# --- wgpu-native vendoring ---
WGPU_VERSION := v27.0.4.0
WGPU_SHA256_macos_aarch64 := 15367c26fdbe6892db35007d39f3883593384e777360b70e6bd704cb5dedde53
WGPU_SHA256_macos_x86_64  := 660fe9be59b555ec1d7c839e5cf8b6c71762938af61ab444a7a58dd87970dba2
WGPU_SHA256_linux_x86_64  := 271481ef76fbf3ea09631a6079e9493636ecf813cd9c92306c44a1a452991ba1
WGPU_SHA256_linux_aarch64  := a2f22248200997b69373273b10d50a58164f6ed840877289f3e46bff317b134e

WGPU_OS := $(shell uname -s | tr A-Z a-z | sed 's/darwin/macos/')
WGPU_ARCH := $(shell uname -m | sed 's/arm64/aarch64/')
WGPU_PLATFORM := $(WGPU_OS)-$(WGPU_ARCH)
WGPU_ZIP := wgpu-$(WGPU_PLATFORM)-release.zip
WGPU_URL := https://github.com/gfx-rs/wgpu-native/releases/download/$(WGPU_VERSION)/$(WGPU_ZIP)
WGPU_EXPECTED_SHA := $(WGPU_SHA256_$(subst -,_,$(WGPU_PLATFORM)))

ifndef WGPU_LIB_DIR
  WGPU_LIB_DIR := vendor/wgpu
endif

.PHONY: fetch-wgpu

fetch-wgpu:
	@if [ -f $(WGPU_LIB_DIR)/libwgpu_native.a ]; then \
		echo "wgpu-native already present"; \
	else \
		echo "=== Fetching wgpu-native $(WGPU_VERSION) ==="; \
		curl -sL -o /tmp/$(WGPU_ZIP) "$(WGPU_URL)"; \
		ACTUAL=$$(shasum -a 256 /tmp/$(WGPU_ZIP) | cut -d' ' -f1); \
		if [ "$$ACTUAL" != "$(WGPU_EXPECTED_SHA)" ]; then \
			echo "SHA-256 mismatch: expected $(WGPU_EXPECTED_SHA), got $$ACTUAL"; \
			rm -f /tmp/$(WGPU_ZIP); exit 1; \
		fi; \
		mkdir -p $(WGPU_LIB_DIR); \
		unzip -o -j /tmp/$(WGPU_ZIP) "lib/libwgpu_native.a" -d $(WGPU_LIB_DIR)/; \
		unzip -o -j /tmp/$(WGPU_ZIP) "include/webgpu/webgpu.h" -d $(WGPU_LIB_DIR)/; \
		unzip -o -j /tmp/$(WGPU_ZIP) "include/webgpu/wgpu.h" -d $(WGPU_LIB_DIR)/; \
		rm -f /tmp/$(WGPU_ZIP); \
		echo "=== wgpu-native installed to $(WGPU_LIB_DIR) ==="; \
	fi

# WebGPU test (requires BN_ENABLE_WEBGPU=1 and fetch-wgpu)
WEBGPU_TEST_SRCS = test/test_gpu_wgpu.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
                src/gguf.c src/platform.c src/tokenizer.c src/threadpool.c \
                src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/sh_arena.c src/sh_log.c \
                src/session.c src/bn_alloc.c
ifdef BN_ENABLE_WEBGPU
WEBGPU_TEST_SRCS += src/gpu_wgpu.c
endif

test_gpu_wgpu: $(WEBGPU_TEST_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

# WebGPU validation benchmark (all 22 quant types, requires BN_ENABLE_WEBGPU=1)
WEBGPU_VALIDATE_SRCS = test/test_gpu_validate.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
                    src/gguf.c src/platform.c src/tokenizer.c src/threadpool.c \
                    src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/sh_arena.c src/sh_log.c \
                    src/session.c src/bn_alloc.c
ifdef BN_ENABLE_WEBGPU
WEBGPU_VALIDATE_SRCS += src/gpu_wgpu.c
endif

test_gpu_validate: $(WEBGPU_VALIDATE_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test/llama_layer_probe: test/llama_layer_probe.cpp
	$(CXX) -O2 -std=c++17 $(LLAMA_PROBE_CFLAGS) -I$(LLAMA_CPP_SRC) -o $@ $< $(LLAMA_PROBE_LIBS)

test/libllama_token_trace.so: test/llama_token_trace.cpp
	$(CXX) -O2 -std=c++17 -fPIC -shared -o $@ $< -ldl

test_llama_layer_probe: test/llama_layer_probe test/libllama_token_trace.so

ifdef BN_ENABLE_METAL
test_metal_f32: test/test_metal_f32.c $(filter-out src/main.c, $(SRCS)) src/gpu_metal.m
	$(CC) $(CFLAGS) -c -o /tmp/bn_test_metal_f32.o src/gpu_metal.m -fobjc-arc
	$(CC) $(CFLAGS) -o $@ test/test_metal_f32.c \
		$(filter-out src/main.c, $(SRCS)) /tmp/bn_test_metal_f32.o $(LDFLAGS)
	./$@
else
test_metal_f32:
	@echo "test_metal_f32 requires BN_ENABLE_METAL=1"
	@exit 1
endif

# Coherence test (WebGPU/Metal vs CPU forward pass, SIMD vs scalar matvec, requires model file)
COHERENCE_SRCS = test/test_coherence.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
                 src/gguf.c src/platform.c src/tokenizer.c src/threadpool.c \
                 src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/sh_arena.c src/sh_log.c \
                 src/session.c src/bn_alloc.c src/prompt_cache.c src/generate_batch_plan.c src/generate_batch.c src/generate.c $(SAMPLER_SRCS)
ifdef BN_ENABLE_WEBGPU
COHERENCE_SRCS += src/gpu_wgpu.c
endif
COHERENCE_EXTRA_OBJS :=
ifdef BN_ENABLE_CUDA
COHERENCE_EXTRA_OBJS += $(CUDA_OBJS)
endif
ifdef BN_ENABLE_METAL
COHERENCE_OBJS = $(COHERENCE_SRCS:.c=.o) src/gpu_metal.o
else
COHERENCE_OBJS =
endif

ifdef BN_ENABLE_METAL
test_coherence: $(COHERENCE_SRCS) $(COHERENCE_EXTRA_OBJS) src/gpu_metal.m
	$(CC) $(CFLAGS) -c -o /tmp/bn_coherence_metal.o src/gpu_metal.m -fobjc-arc
	$(CC) $(CFLAGS) -o $@ $(COHERENCE_SRCS) $(COHERENCE_EXTRA_OBJS) /tmp/bn_coherence_metal.o $(LDFLAGS)
else
test_coherence: $(COHERENCE_SRCS) $(COHERENCE_EXTRA_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
endif

clean:
	rm -f test_platform
	rm -f test_cuda_flash_group
	rm -f test_cuda_glm_group
	rm -f test_gguf_batch_options
	rm -f bench_native_forward
	rm -f test_cuda_attention_gate
	rm -f test_cuda_q8_mma
	rm -f bench_bf16_latency
	rm -f bench_flash_offload
	rm -f bench_bf16_concurrency
	rm -f bench_bf16_batch bench_bf16_arrivals
	rm -f bench_checkpoint_gdn
	rm -f bench_checkpoint_attention
	rm -f test_cuda_silu_profiles
	rm -f test_cuda_shared_math
	rm -f test_cuda_glm_final_output
	rm -f test_cuda_glm_context
	rm -f test_cuda_glm_request
	rm -f test_cuda_glm_model_resources
	rm -f test_cuda_glm_kda_moe_block
	rm -f test_cuda_glm_paged_score test_cuda_glm_select test_cuda_glm_score test_cuda_glm_fp8_cache test_cuda_glm_indexer test_cuda_moe_bank
	rm -f test_cuda_moe_group
	rm -f test_cuda_expert_cache
	rm -f test_bf16_runtime
	rm -f test_backend_session_rows test_backend_model_rows test_int4_rows test_fp8 test_json test_bf16_batch_schedule test_bf16_pool test_generate_batch_plan test_generate_batch test_safetensors test_exl3 test_trellis test_gpu_moe_cache
	rm -f reference_cuda_bf16_packed reference_cuda_bf16_mixed
	rm -f test_cuda_glm_mla_block test_cuda_glm_indexed_mla test_cuda_glm_mla_attention_kernel test_cuda_glm_mla_attention test_cuda_glm_mla_cache_chain test_cuda_glm_mla_cache test_cuda_glm_mla_projection test_cuda_glm_indexer_decode test_cuda_glm_indexer_select test_cuda_glm_indexer_scores test_cuda_glm_indexer_cache test_cuda_glm_preindexer test_cuda_glm_router test_cuda_glm_moe_layer test_cuda_glm_routed_ffn test_cuda_glm_dense_prefix test_cuda_glm_dense_ffn test_cuda_glm_deferred_kda test_cuda_glm_mhc_fused test_cuda_glm_mhc_kda test_cuda_glm_mhc test_cuda_glm_kda_layer bitnet bitnet_scalar bitnet_avx2 bitnet_avx512 bench_kernels bench_prefill bench_scalar bench_scalar_layers bench_avx2 bench_avx512 bench_webgpu bench_layers src/*.o src/quant/*.o src/transformer/*.o test_gguf test_quant test_quant_avx2 test_quant_scalar test_tokenizer test_transformer test_threadpool test_safety test_arena test_q2k test_mhc test_ssm test_gguf_fuzz test_moe test_qwen36 test_generate test_generate_native_text test_native_cli test_native_text_timing test_session test_prompt_cache test_turboquant test_gpu_graph_ir test_gpu_backend test_cuda_backend test_cuda_fp16 test_gpu_wgpu test_gpu_validate test_metal_f32 test_coherence test_e2e test_prefill test_prefill_output_rows test_prefill_continuation test_prefill_reference test_prefill_reference_avx2 test_kv_f16 test/llama_layer_probe test/libllama_token_trace.so default.profraw default.profdata src/*.gcda src/quant/*.gcda src/transformer/*.gcda src/gpu_metal.o $(BUILD_CONFIG_STAMP)

.PHONY: test_generate_batch_plan
test_generate_batch_plan: test/test_generate_batch_plan.c test/generate_batch_plan_reference.h include/generate_batch_plan.h src/generate_batch_plan.c
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS) && ./$@

.PHONY: test_generate_batch
test_generate_batch: test/test_generate_batch.c test/generate_batch_plan_reference.h include/generate_batch.h include/generate_batch_plan.h src/generate_batch.c src/generate_batch_plan.c
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS) && ./$@

.PHONY: test_bf16_arrival_workload
test_bf16_arrival_workload:
	python3 test/test_bf16_arrival_workload.py

.PHONY: test_bf16_wall_clock_verifier
test_bf16_wall_clock_verifier:
	python3 test/test_bf16_wall_clock_verifier.py

.PHONY: test_bf16_capacity_pool
test_bf16_capacity_pool:
	python3 test/test_bf16_capacity_pool.py

.PHONY: test_json
test_json: test/test_json.c src/json.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

.PHONY: test_fp8
test_fp8: test/test_fp8.c src/quant/fp8.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

# Private native request orchestration headers.
test_moe: test/native_attention_session_checks.h test/native_anatomy_checks.h src/transformer/trellis_attention.h src/transformer/trellis_kv.h src/transformer/trellis_swa.h src/transformer/trellis_index.h

test_moe: test/native_producer_checks.h src/transformer/trellis_producer.h

test_cuda_backend: test/cuda_native_frequency_checks.h test/cuda_native_frequency_reference.h
test_cuda_backend: test/cuda_scaled_rows_checks.h

.PHONY: test_int4_rows
test_int4_rows: test/test_int4_rows.c src/quant/int4_rows.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

.PHONY: test_backend_model_rows
test_backend_model_rows: test/test_backend_model_rows.c src/backend_model_rows.c src/trellis.c src/safetensors.c src/json.c src/platform.c $(QUANT_SRCS) src/threadpool.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

.PHONY: test_backend_session_rows
test_backend_session_rows: test/test_backend_session_rows.c test/test_backend_model_rows.c src/backend_session_rows.c src/backend_model_rows.c src/trellis.c src/safetensors.c src/json.c src/platform.c $(QUANT_SRCS) src/threadpool.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $(filter-out test/test_backend_model_rows.c,$^) $(LDFLAGS) && ./$@

test_moe: test/native_rows_checks.h src/transformer/trellis_rows.h src/transformer/trellis_engram.h

src/model_arch.o: src/model_arch_engram.h
test_moe: src/model_arch_engram.h test/native_engram_layout_checks.h test/native_engram_layout_reference.h

test_moe: test/native_ngram_rows_checks.h test/native_ngram_rows_reference.h

src/tokenizer.o: src/tokenizer_vocab.h
src/tokenizer.o test_tokenizer: src/tokenizer_encode.h src/tokenizer_encode_unicode.h
test_tokenizer: src/tokenizer_vocab.h

test_tokenizer: test/native_vocab_checks.h test/native_vocab_reference.h

src/tokenizer.o: src/tokenizer_map.h src/tokenizer_map_data.h
test_tokenizer: src/tokenizer_map.h src/tokenizer_map_data.h

test_tokenizer: test/native_tokenmap_checks.h test/native_tokenmap_reference.h

src/transformer/trellis.o: src/transformer/trellis_engram.h src/transformer/trellis_hc.h src/transformer/trellis_block.h src/transformer/trellis_forward.h

test_moe: test/native_engram_stage_checks.h
test_moe: src/transformer/trellis_hc.h
test_moe: src/transformer/trellis_block.h
test_moe: src/transformer/trellis_forward.h

test_tokenizer: test/native_encode_checks.h test/native_encode_reference.h test/native_encode_edges.h

src/tokenizer.o test_tokenizer: src/tokenizer_decode_stream.h
test_tokenizer: test/native_stream_checks.h test/native_stream_reference.h test/native_stream_error_reference.h test/native_stream_fallback_reference.h

# Native text orchestration and its reference control.
src/generate.o test_generate: src/generate_native_text.h src/generate_native_schedule.h
bench_native_forward: bench/native_text_profile.h

.PHONY: test_generate_native_text
test: test_generate_native_text
test_generate_native_text: test/test_generate_native_text.c src/tokenizer.c src/json.c src/model_arch.c src/safetensors.c src/gguf.c src/platform.c src/sh_log.c src/sh_arena.c src/threadpool.c $(QUANT_SRCS) src/generate_native_text.h src/generate_native_schedule.h test/native_schedule_checks.h test/native_encode_reference.h test/native_text_control_reference.h
	$(CC) $(CFLAGS) -DBN_TEST_NATIVE_TEXT_HEADER -o $@ $(filter-out %.h,$^) $(LDFLAGS) && ./$@

src/main.o: src/main_gguf_batch.h src/main_gguf_batch_options.h src/main_bf16_policy.h src/main_native_text.h src/main_native_options.h src/main_native_io.h src/main_native_batch.h src/main_native_batch_input.h

.PHONY: test_native_cli
test: test_native_cli
test_native_cli: test/test_native_cli.c src/json.c src/main_native_options.h src/main_native_io.h src/main_native_batch_input.h include/generate_batch_plan.h
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS) && ./$@

.PHONY: test_native_text_timing
test: test_native_text_timing
test_native_text_timing: test/test_native_text_timing.c bench/native_text_timing.h
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS) && ./$@

bench_native_forward: bench/native_text_latency.h bench/native_text_timing.h bench/native_text_workload.h

.PHONY: test_native_text_workload
test: test_native_text_workload
test_native_text_workload: test/test_native_text_workload.c bench/native_text_workload.h
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS) && ./$@

clean: clean_native_text_workload
.PHONY: clean_native_text_workload
clean_native_text_workload:
	rm -f test_native_text_workload

# Full native storage reuse/payload retention stays in the transformer module.
src/transformer/trellis.o: src/transformer/trellis_reuse.h src/transformer/trellis_pool.h
.PHONY: test_native_pool test_native_restart clean-native-reuse
test: test_native_pool test_native_restart
test_native_pool: test/test_native_pool.c src/transformer/trellis_pool.h
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS) && ./$@
NATIVE_RESTART_DEPS = $(MOE_SRCS) src/turboquant.c $(MODEL_SRCS) src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(filter-out src/transformer/trellis.c,$(TRANSFORMER_BACKEND)) src/gguf.c $(QUANT_SRCS) src/platform.c src/tokenizer.c src/threadpool.c src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
test_native_restart: test/test_native_restart.c src/transformer/trellis.c src/transformer/trellis_reuse.h src/transformer/trellis_pool.h $(NATIVE_RESTART_DEPS)
	$(CC) $(CFLAGS) -o $@ $(filter-out src/transformer/trellis.c,$(filter %.c %.o,$^)) $(LDFLAGS) && ./$@
clean-native-reuse:
	rm -f test_native_pool test_native_restart
clean: clean-native-reuse

bench_native_forward: bench/native_text_schedule.h bench/native_schedule_metrics.h
test_native_text_timing: bench/native_schedule_metrics.h

src/transformer/trellis.o test_native_restart: src/transformer/trellis_group.h
.PHONY: test_native_group clean-native-group
test: test_native_group
test_native_group: test/test_native_group.c test/test_native_restart.c src/transformer/trellis.c src/transformer/trellis_group.h $(NATIVE_RESTART_DEPS)
	$(CC) $(CFLAGS) -o $@ $(filter-out test/test_native_restart.c src/transformer/trellis.c,$(filter %.c %.o,$^)) $(LDFLAGS) && ./$@
clean-native-group:
	rm -f test_native_group
clean: clean-native-group

bench_native_forward: bench/native_text_group.h

.PHONY: test_gguf_batch_options
test: test_gguf_batch_options
test_gguf_batch_options: test/test_gguf_batch_options.c src/json.c src/main_native_options.h src/main_gguf_batch_options.h src/main_bf16_policy.h include/generate_batch_plan.h
	$(CC) $(CFLAGS) -Isrc -o $@ $(filter %.c,$^) $(LDFLAGS) && ./$@

src/main.o: src/main_gguf_memory.h

.PHONY: test_gguf_memory clean_gguf_memory
test: test_gguf_memory
test_gguf_memory: test/test_gguf_memory.c src/main_gguf_memory.h src/main_gguf_batch_options.h src/main_native_options.h src/generate_batch_plan.c include/generate_batch_plan.h
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -Wl,--gc-sections -o $@ test/test_gguf_memory.c src/generate_batch_plan.c $(LDFLAGS) && ./$@
clean: clean_gguf_memory
clean_gguf_memory:
	rm -f test_gguf_memory

.PHONY: test_native_memory clean_native_memory
test: test_native_memory
test_native_memory: test/test_native_memory.c src/main_native_memory.h src/main_native_options.h src/generate_batch_plan.c include/generate_batch_plan.h
	$(CC) $(CFLAGS) -o $@ test/test_native_memory.c src/generate_batch_plan.c $(LDFLAGS) && ./$@
clean: clean_native_memory
clean_native_memory:
	rm -f test_native_memory
src/main.o: src/main_native_memory.h

.PHONY: test_serve_io clean_serve_io
test: test_serve_io
test_serve_io: test/test_serve_io.c src/main_serve_io.h src/json.c include/json.h
	$(CC) $(CFLAGS) -o $@ test/test_serve_io.c src/json.c $(LDFLAGS) && ./$@
clean: clean_serve_io
clean_serve_io:
	rm -f test_serve_io

.PHONY: test_serve clean_serve
test: test_serve
test_serve: test/test_serve.c src/main_serve.h src/main_serve_io.h src/generate_batch.c src/generate_batch_plan.c src/json.c include/generate_batch.h include/generate_batch_plan.h
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS) && ./$@
clean: clean_serve
clean_serve:
	rm -f test_serve
src/main.o: src/main_serve.h src/main_serve_io.h
