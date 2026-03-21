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

QUANT_COMMON = src/quant/fp16.c src/quant/dequant.c src/quant/dispatch.c

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
    src/transformer/logits_neon.c src/transformer/logits_scalar.c \
    src/transformer/ssm_neon.c src/transformer/ssm_scalar.c
else
  # x86: AVX2 + scalar
  QUANT_BACKEND = src/quant/x_quant_avx2.c \
    src/quant/i2s_avx2.c src/quant/i2s_avx2_4row.c src/quant/i2s_scalar.c \
    src/quant/tq2_avx2.c src/quant/tq2_scalar.c \
    src/quant/tq1_avx2.c src/quant/tq1_scalar.c \
    src/quant/q8_avx2.c src/quant/q8_scalar.c \
    src/quant/q4_avx2.c src/quant/q4_avx2_4row.c src/quant/q4_scalar.c \
    src/quant/q4_1_avx2.c src/quant/q4_1_scalar.c \
    src/quant/bf16_avx2.c src/quant/bf16_scalar.c \
    src/quant/q6k_avx2.c src/quant/q6k_avx2_sdot.c src/quant/q6k_scalar.c \
    src/quant/q8k_avx2.c src/quant/q8k_scalar.c \
    src/quant/q4k_avx2.c src/quant/q4k_avx2_sdot.c src/quant/q4k_scalar.c \
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
    src/transformer/logits_avx2.c src/transformer/logits_scalar.c \
    src/transformer/ssm_avx2.c src/transformer/ssm_scalar.c
endif

QUANT_SRCS = $(QUANT_COMMON) $(QUANT_BACKEND)
TRANSFORMER_SRCS = src/transformer.c $(TRANSFORMER_BACKEND)

SRCS = src/platform.c src/gguf.c $(QUANT_SRCS) src/model.c src/moe.c \
       $(TRANSFORMER_SRCS) src/tokenizer.c src/sampler.c \
       src/threadpool.c src/sh_arena.c src/sh_log.c src/main.c
OBJS = $(SRCS:.c=.o)

# Default target
bitnet: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

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

# --- Tests ---
# --- Benchmark ---
BENCH_SRCS = bench/bench_kernels.c $(filter-out src/main.c, $(SRCS))

bench_kernels: $(BENCH_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Scalar benchmark (no -march=native, no SIMD)
SCALAR_CFLAGS = -O3 -Wall -Wextra -Wshadow -std=c11 -Iinclude
ifeq ($(UNAME_S),Linux)
SCALAR_CFLAGS += -D_GNU_SOURCE
endif
SCALAR_BENCH_SRCS = bench/bench_kernels.c $(filter-out src/main.c, $(SRCS))

bench_scalar: $(SCALAR_BENCH_SRCS)
	$(CC) $(SCALAR_CFLAGS) -o $@ $^ $(LDFLAGS)

bench_avx2: $(BENCH_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Per-layer timing build (BN_BENCH_LAYERS)
bench_layers: CFLAGS += -DBN_BENCH_LAYERS
bench_layers: $(BENCH_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: debug asan bench bench_scalar bench_avx2 bench_layers test test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_prefill test_kv_f16 test_q2k test_ssm test_gguf_fuzz test_moe pgo avx2-check clean

bench: bench_kernels

test: test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_ssm test_gguf_fuzz test_moe

test_gguf: test/test_gguf.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_quant: test/test_quant.c $(QUANT_SRCS) src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_tokenizer: test/test_tokenizer.c src/tokenizer.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_transformer: test/test_transformer.c src/transformer.c $(TRANSFORMER_BACKEND) src/model.c src/moe.c \
                  src/gguf.c $(QUANT_SRCS) src/platform.c src/tokenizer.c src/threadpool.c \
                  src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_threadpool: test/test_threadpool.c src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_safety: test/test_safety.c src/platform.c src/gguf.c $(QUANT_SRCS) src/model.c src/moe.c \
             src/transformer.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_arena: test/test_arena.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

SSM_BACKEND = $(filter src/transformer/ssm_%, $(TRANSFORMER_BACKEND))
test_ssm: test/test_ssm.c src/transformer/ssm_scalar.c $(SSM_BACKEND)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_gguf_fuzz: test/test_gguf_fuzz.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_moe: test/test_moe.c src/moe.c src/model.c src/transformer.c $(TRANSFORMER_BACKEND) \
          src/gguf.c $(QUANT_SRCS) src/platform.c src/tokenizer.c src/threadpool.c \
          src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_q2k: test/test_q2k.c $(QUANT_SRCS) src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_e2e: test/test_e2e.c src/platform.c src/gguf.c $(QUANT_SRCS) src/model.c src/moe.c \
          src/transformer.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c src/threadpool.c \
          src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_prefill: test/test_prefill.c src/platform.c src/gguf.c $(QUANT_SRCS) src/model.c src/moe.c \
              src/transformer.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
              src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_f16: test/test_kv_f16.c src/platform.c src/gguf.c $(QUANT_SRCS) src/model.c src/moe.c \
             src/transformer.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c
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
    src/quant/q4_avx2.c src/quant/q4_avx2_4row.c src/quant/q4_scalar.c \
    src/quant/q4_1_avx2.c src/quant/q4_1_scalar.c \
    src/quant/bf16_avx2.c src/quant/bf16_scalar.c \
    src/quant/q6k_avx2.c src/quant/q6k_avx2_sdot.c src/quant/q6k_scalar.c \
    src/quant/q8k_avx2.c src/quant/q8k_scalar.c \
    src/quant/q4k_avx2.c src/quant/q4k_avx2_sdot.c src/quant/q4k_scalar.c \
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
    src/transformer/logits_avx2.c src/transformer/logits_scalar.c \
    src/transformer/ssm_avx2.c src/transformer/ssm_scalar.c

AVX2_SRCS = src/platform.c src/gguf.c $(AVX2_QUANT_SRCS) src/model.c src/moe.c \
            src/transformer.c $(AVX2_TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c \
            src/threadpool.c src/sh_arena.c src/sh_log.c

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

clean:
	rm -f bitnet bench_kernels bench_scalar bench_avx2 bench_layers src/*.o src/quant/*.o src/transformer/*.o test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_q2k test_ssm test_gguf_fuzz test_moe test_e2e test_prefill test_kv_f16 default.profraw default.profdata src/*.gcda src/quant/*.gcda src/transformer/*.gcda
