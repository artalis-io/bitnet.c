CC      ?= cc
LDFLAGS = -lm

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

QUANT_COMMON = src/quant/fp16.c src/quant/dequant.c src/quant/registry.c src/quant/kernel_select.c src/quant/dispatch.c src/quant/matvec_batch.c src/quant/matmul.c src/quant/fused_gateup.c src/quant/batch_preq8k.c src/quant/matvec_multi.c

UNAME_M := $(shell uname -m)
ifneq ($(filter arm% aarch%,$(UNAME_M)),)
  # ARM: NEON + NEON SDOT + scalar
  QUANT_BACKEND = src/quant/x_quant_neon.c \
    src/quant/i2s_neon_sdot.c src/quant/i2s_neon.c src/quant/i2s_scalar.c \
    src/quant/tq2_neon_sdot.c src/quant/tq2_neon.c src/quant/tq2_scalar.c \
    src/quant/tq1_neon_sdot.c src/quant/tq1_neon.c src/quant/tq1_scalar.c \
    src/quant/q8_neon_sdot.c src/quant/q8_neon.c src/quant/q8_scalar.c \
    src/quant/q4_neon_sdot.c src/quant/q4_neon.c src/quant/q4_scalar.c \
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

  TRANSFORMER_BACKEND = src/transformer/rmsnorm_neon.c src/transformer/rmsnorm_scalar.c \
    src/transformer/gqa_neon.c src/transformer/gqa_scalar.c \
    src/transformer/gqa_tq_scalar.c src/transformer/gqa_tq_neon.c \
    src/transformer/logits_neon.c src/transformer/logits_scalar.c src/transformer/logits.c \
    src/transformer/cpu.c \
    src/transformer/plan.c src/transformer/gpu_emit.c src/transformer/gpu.c \
    src/transformer/kv.c src/transformer/prefill.c \
    src/transformer/ssm_neon.c src/transformer/ssm_scalar.c
else
  # x86: AVX2 + scalar
  CFLAGS += -mprefer-vector-width=256
  QUANT_BACKEND = src/quant/x_quant_avx2.c src/quant/rmsnorm_q8k_avx2.c \
    src/quant/i2s_avx2.c src/quant/i2s_avx2_4row.c src/quant/i2s_scalar.c \
    src/quant/tq2_avx2.c src/quant/tq2_scalar.c \
    src/quant/tq1_avx2.c src/quant/tq1_scalar.c \
    src/quant/q8_avx2.c src/quant/q8_scalar.c \
    src/quant/q4_avx2.c src/quant/q4_avx2_4row.c src/quant/q4_avx2_matmul.c src/quant/q4_scalar.c \
    src/quant/q4_1_avx2.c src/quant/q4_1_scalar.c \
    src/quant/f32_avx2.c src/quant/f32_scalar.c \
    src/quant/f16_avx2.c src/quant/f16_scalar.c \
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

  TRANSFORMER_BACKEND = src/transformer/rmsnorm_avx2.c src/transformer/rmsnorm_scalar.c \
    src/transformer/gqa_avx2.c src/transformer/gqa_scalar.c \
    src/transformer/gqa_tq_scalar.c \
    src/transformer/logits_avx2.c src/transformer/logits_scalar.c src/transformer/logits.c \
    src/transformer/cpu.c \
    src/transformer/plan.c src/transformer/gpu_emit.c src/transformer/gpu.c \
    src/transformer/kv.c src/transformer/prefill.c \
    src/transformer/ssm_avx2.c src/transformer/ssm_scalar.c
endif

QUANT_SRCS = $(QUANT_COMMON) $(QUANT_BACKEND)
MODEL_SRCS = src/model.c src/model_arch.c src/model_session.c src/model_gpu.c src/model_embed.c src/backend_layout.c src/backend_model.c src/backend_session.c src/backend_quant.c
MOE_SRCS = src/moe.c src/moe_cache.c src/moe_io.c src/moe_route.c src/moe_math.c src/moe_execute.c src/moe_prefill.c src/moe_stats.c
TRANSFORMER_SRCS = src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND)

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

SRCS = src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
       $(TRANSFORMER_SRCS) src/tokenizer.c src/sampler.c \
       src/threadpool.c src/sh_arena.c src/sh_log.c src/bn_alloc.c src/session.c src/prompt_cache.c src/generate.c $(WEBGPU_SRCS) src/main.c
CFLAGS += $(WEBGPU_CFLAGS) $(METAL_CFLAGS)
LDFLAGS += $(WGPU_LIB) $(WGPU_FRAMEWORKS) $(METAL_FRAMEWORKS)
OBJS = $(SRCS:.c=.o) $(METAL_OBJS)
BUILD_CONFIG := webgpu=$(if $(BN_ENABLE_WEBGPU),1,0) metal=$(if $(BN_ENABLE_METAL),1,0) cc=$(CC) cflags=$(CFLAGS)
BUILD_CONFIG_STAMP := .build-config

.PHONY: config-check
config-check:
	@old=$$(cat $(BUILD_CONFIG_STAMP) 2>/dev/null || true); \
	if [ "$$old" != "$(BUILD_CONFIG)" ]; then \
		echo "Build config changed; rebuilding objects"; \
		rm -f src/*.o src/quant/*.o src/transformer/*.o src/gpu_metal.o bitnet; \
		printf '%s\n' "$(BUILD_CONFIG)" > $(BUILD_CONFIG_STAMP); \
	fi

# Default target
bitnet: config-check $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

# Debug build
debug: CFLAGS += -DDEBUG -g -O0
debug: bitnet

# Sanitizer build (ASan + UBSan)
asan: CFLAGS += -DDEBUG -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
asan: LDFLAGS += -fsanitize=address,undefined
asan: bitnet

# Pattern rules for object files
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

src/quant/%.o: src/quant/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

src/transformer/%.o: src/transformer/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Objective-C pattern rule for Metal backend
src/%.o: src/%.m
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

# --- Tests ---
# --- Benchmark ---
BENCH_SRCS = bench/bench_kernels.c $(filter-out src/main.c, $(SRCS))

bench_kernels: $(BENCH_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Scalar benchmark (no -march=native, no SIMD)
SCALAR_CFLAGS = -O3 -Wall -Wextra -Wshadow -std=c11 -Iinclude -DBN_FORCE_SCALAR
ifeq ($(UNAME_S),Linux)
SCALAR_CFLAGS += -D_GNU_SOURCE
endif
SCALAR_QUANT_BACKEND = src/quant/i2s_scalar.c \
    src/quant/tq2_scalar.c src/quant/tq1_scalar.c \
    src/quant/q8_scalar.c src/quant/q4_scalar.c src/quant/q4_1_scalar.c \
    src/quant/f32_scalar.c src/quant/f16_scalar.c src/quant/bf16_scalar.c \
    src/quant/q6k_scalar.c src/quant/q8k_scalar.c src/quant/q4k_scalar.c \
    src/quant/q5k_scalar.c src/quant/q3k_scalar.c src/quant/q2k_scalar.c \
    src/quant/iq4nl_scalar.c src/quant/iq4xs_scalar.c \
    src/quant/iq3xxs_scalar.c src/quant/iq3s_scalar.c \
    src/quant/iq2xxs_scalar.c src/quant/iq2xs_scalar.c src/quant/iq2s_scalar.c

SCALAR_TRANSFORMER_BACKEND = src/transformer/rmsnorm_scalar.c \
    src/transformer/gqa_scalar.c src/transformer/gqa_tq_scalar.c \
    src/transformer/logits_scalar.c src/transformer/logits.c src/transformer/cpu.c \
    src/transformer/plan.c src/transformer/gpu_emit.c src/transformer/gpu.c \
    src/transformer/kv.c src/transformer/prefill.c \
    src/transformer/ssm_scalar.c

SCALAR_SRCS = src/platform.c src/gguf.c $(QUANT_COMMON) $(SCALAR_QUANT_BACKEND) \
       src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c \
       $(SCALAR_TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c \
       src/threadpool.c src/sh_arena.c src/sh_log.c src/bn_alloc.c src/session.c \
       src/prompt_cache.c src/generate.c src/main.c

SCALAR_BENCH_SRCS = bench/bench_kernels.c $(filter-out src/main.c, $(SCALAR_SRCS))

bench_scalar: $(SCALAR_BENCH_SRCS)
	$(CC) $(SCALAR_CFLAGS) -o $@ $^ $(LDFLAGS)

bench_scalar_layers: SCALAR_CFLAGS += -DBN_BENCH_LAYERS
bench_scalar_layers: $(SCALAR_BENCH_SRCS)
	$(CC) $(SCALAR_CFLAGS) -o $@ $^ $(LDFLAGS)

bitnet_scalar: $(SCALAR_SRCS)
	$(CC) $(SCALAR_CFLAGS) -o $@ $^ $(LDFLAGS)

bench_avx2: $(BENCH_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ifeq ($(BN_ENABLE_WEBGPU),1)
bench_webgpu: $(BENCH_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
else
bench_webgpu:
	@echo "bench_webgpu skipped: set BN_ENABLE_WEBGPU=1"
endif

# Per-layer timing build (BN_BENCH_LAYERS)
bench_layers: CFLAGS += -DBN_BENCH_LAYERS
bench_layers: $(BENCH_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: debug asan bench bench_suite bench_llama_compare bench_kernels_run bitnet_scalar bench_scalar bench_scalar_layers bench_avx2 bench_webgpu bench_layers test test_architecture test_backend_matrix test_model_matrix test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_prefill test_kv_f16 test_q2k test_ssm test_gguf_fuzz test_moe test_qwen36 test_gemma4 test_gemma4_avx2 test_gemma4_webgpu test_gemma4_backend_matrix test_generate test_session test_prompt_cache test_turboquant test_gpu_backend test_gpu_wgpu test_gpu_validate test_coherence pgo avx2-check fetch-wgpu clean

bench: $(MAIN_TARGET)
	./bench/bench_suite.sh

bench_suite: $(MAIN_TARGET)
	./bench/bench_suite.sh

bench_llama_compare: bench_avx2
	./bench/compare_llama.sh

bench_kernels_run: bench_kernels

test: test_architecture test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_ssm test_gguf_fuzz test_moe test_qwen36 test_gemma4 test_generate test_session test_prompt_cache test_turboquant test_gpu_backend

test_architecture: test_backend_matrix test_model_matrix

test_backend_matrix:
	./test/backend_matrix.sh

test_model_matrix: test_coherence
	./test/model_matrix.sh

test_gguf: test/test_gguf.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_quant: test/test_quant.c $(QUANT_SRCS) src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_tokenizer: test/test_tokenizer.c src/tokenizer.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_transformer: test/test_transformer.c $(TRANSFORMER_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
                  src/gguf.c $(QUANT_SRCS) src/platform.c src/tokenizer.c src/threadpool.c \
                  src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_threadpool: test/test_threadpool.c src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_safety: test/test_safety.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_arena: test/test_arena.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

SSM_BACKEND = $(filter src/transformer/ssm_%, $(TRANSFORMER_BACKEND))
test_ssm: test/test_ssm.c src/transformer/ssm_scalar.c $(SSM_BACKEND)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_gguf_fuzz: test/test_gguf_fuzz.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_moe: test/test_moe.c $(MOE_SRCS) src/turboquant.c $(MODEL_SRCS) src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) \
          src/gguf.c $(QUANT_SRCS) src/platform.c src/tokenizer.c src/threadpool.c \
          src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_qwen36: test/test_qwen36.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

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

test_gemma4_backend_matrix: test_gemma4 test_gemma4_avx2 test_gemma4_webgpu

test_generate: test/test_generate.c src/generate.c src/bn_alloc.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
               src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c src/threadpool.c \
               src/sh_arena.c src/sh_log.c src/session.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_q2k: test/test_q2k.c $(QUANT_SRCS) src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

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

test_gpu_backend: test/test_gpu_backend.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
                  src/gguf.c src/platform.c src/tokenizer.c src/threadpool.c \
                  src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/sh_arena.c src/sh_log.c \
                  src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_e2e: test/test_e2e.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
          src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c src/threadpool.c \
          src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_prefill: test/test_prefill.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
              src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
              src/sh_arena.c src/sh_log.c src/session.c src/bn_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_f16: test/test_kv_f16.c src/platform.c src/gguf.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
             src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c src/threadpool.c \
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
    src/quant/x_quant_avx2.c \
    src/quant/i2s_avx2.c src/quant/i2s_avx2_4row.c src/quant/i2s_scalar.c \
    src/quant/tq2_avx2.c src/quant/tq2_scalar.c \
    src/quant/tq1_avx2.c src/quant/tq1_scalar.c \
    src/quant/q8_avx2.c src/quant/q8_scalar.c \
    src/quant/q4_avx2.c src/quant/q4_avx2_4row.c src/quant/q4_avx2_matmul.c src/quant/q4_scalar.c \
    src/quant/q4_1_avx2.c src/quant/q4_1_scalar.c \
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

AVX2_TRANSFORMER_BACKEND = src/transformer/rmsnorm_avx2.c src/transformer/rmsnorm_scalar.c \
    src/transformer/gqa_avx2.c src/transformer/gqa_scalar.c \
    src/transformer/gqa_tq_scalar.c \
    src/transformer/logits_avx2.c src/transformer/logits_scalar.c src/transformer/logits.c \
    src/transformer/cpu.c \
    src/transformer/plan.c src/transformer/gpu_emit.c src/transformer/gpu.c \
    src/transformer/kv.c src/transformer/prefill.c \
    src/transformer/ssm_avx2.c src/transformer/ssm_scalar.c

AVX2_SRCS = src/platform.c src/gguf.c $(AVX2_QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
            src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(AVX2_TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c \
            src/threadpool.c src/sh_arena.c src/sh_log.c src/bn_alloc.c src/session.c src/generate.c

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

# Coherence test (WebGPU/Metal vs CPU forward pass, SIMD vs scalar matvec, requires model file)
COHERENCE_SRCS = test/test_coherence.c $(QUANT_SRCS) src/turboquant.c $(MODEL_SRCS) $(MOE_SRCS) \
                 src/gguf.c src/platform.c src/tokenizer.c src/threadpool.c \
                 src/transformer.c src/gpu_moe_cache.c src/gpu_moe_bridge.c $(TRANSFORMER_BACKEND) src/sh_arena.c src/sh_log.c \
                 src/session.c src/bn_alloc.c src/prompt_cache.c src/generate.c src/sampler.c
ifdef BN_ENABLE_WEBGPU
COHERENCE_SRCS += src/gpu_wgpu.c
endif
ifdef BN_ENABLE_METAL
COHERENCE_OBJS = $(COHERENCE_SRCS:.c=.o) src/gpu_metal.o
else
COHERENCE_OBJS =
endif

ifdef BN_ENABLE_METAL
test_coherence: $(COHERENCE_SRCS) src/gpu_metal.m
	$(CC) $(CFLAGS) -c -o /tmp/bn_coherence_metal.o src/gpu_metal.m -fobjc-arc
	$(CC) $(CFLAGS) -o $@ $(COHERENCE_SRCS) /tmp/bn_coherence_metal.o $(LDFLAGS)
else
test_coherence: $(COHERENCE_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
endif

clean:
	rm -f bitnet bitnet_scalar bench_kernels bench_scalar bench_scalar_layers bench_avx2 bench_webgpu bench_layers src/*.o src/quant/*.o src/transformer/*.o test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_q2k test_ssm test_gguf_fuzz test_moe test_qwen36 test_generate test_session test_prompt_cache test_turboquant test_gpu_backend test_gpu_wgpu test_gpu_validate test_coherence test_e2e test_prefill test_kv_f16 default.profraw default.profdata src/*.gcda src/quant/*.gcda src/transformer/*.gcda src/gpu_metal.o $(BUILD_CONFIG_STAMP)
