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
    src/quant/tq2_neon.c src/quant/tq2_scalar.c \
    src/quant/tq1_neon_sdot.c src/quant/tq1_neon.c src/quant/tq1_scalar.c \
    src/quant/q8_neon.c src/quant/q8_scalar.c \
    src/quant/q4_neon_sdot.c src/quant/q4_neon.c src/quant/q4_scalar.c \
    src/quant/q6k_neon.c src/quant/q6k_scalar.c \
    src/quant/q8k_neon.c src/quant/q8k_scalar.c \
    src/quant/q4k_neon.c src/quant/q4k_scalar.c \
    src/quant/q5k_neon.c src/quant/q5k_scalar.c \
    src/quant/q3k_neon.c src/quant/q3k_scalar.c \
    src/quant/q2k_neon.c src/quant/q2k_scalar.c

  TRANSFORMER_BACKEND = src/transformer/rmsnorm_neon.c src/transformer/rmsnorm_scalar.c \
    src/transformer/gqa_neon.c src/transformer/gqa_scalar.c \
    src/transformer/logits_neon.c src/transformer/logits_scalar.c
else
  # x86: AVX2 + scalar
  QUANT_BACKEND = src/quant/x_quant_avx2.c \
    src/quant/i2s_avx2.c src/quant/i2s_scalar.c \
    src/quant/tq2_scalar.c src/quant/tq1_scalar.c \
    src/quant/q8_avx2.c src/quant/q8_scalar.c \
    src/quant/q4_avx2.c src/quant/q4_scalar.c \
    src/quant/q6k_avx2.c src/quant/q6k_scalar.c \
    src/quant/q8k_avx2.c src/quant/q8k_scalar.c \
    src/quant/q4k_avx2.c src/quant/q4k_scalar.c \
    src/quant/q5k_avx2.c src/quant/q5k_scalar.c \
    src/quant/q3k_avx2.c src/quant/q3k_scalar.c \
    src/quant/q2k_avx2.c src/quant/q2k_scalar.c

  TRANSFORMER_BACKEND = src/transformer/rmsnorm_avx2.c src/transformer/rmsnorm_scalar.c \
    src/transformer/gqa_avx2.c src/transformer/gqa_scalar.c \
    src/transformer/logits_avx2.c src/transformer/logits_scalar.c
endif

QUANT_SRCS = $(QUANT_COMMON) $(QUANT_BACKEND)
TRANSFORMER_SRCS = src/transformer.c $(TRANSFORMER_BACKEND)

SRCS = src/platform.c src/gguf.c $(QUANT_SRCS) src/model.c \
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
.PHONY: debug asan test test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_prefill test_kv_f16 test_q2k pgo avx2-check clean

test: test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena

test_gguf: test/test_gguf.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_quant: test/test_quant.c $(QUANT_SRCS) src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_tokenizer: test/test_tokenizer.c src/tokenizer.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_transformer: test/test_transformer.c src/transformer.c $(TRANSFORMER_BACKEND) src/model.c \
                  src/gguf.c $(QUANT_SRCS) src/platform.c src/tokenizer.c src/threadpool.c \
                  src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_threadpool: test/test_threadpool.c src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_safety: test/test_safety.c src/platform.c src/gguf.c $(QUANT_SRCS) src/model.c \
             src/transformer.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_arena: test/test_arena.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_q2k: test/test_q2k.c $(QUANT_SRCS) src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_e2e: test/test_e2e.c src/platform.c src/gguf.c $(QUANT_SRCS) src/model.c \
          src/transformer.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c src/threadpool.c \
          src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_prefill: test/test_prefill.c src/platform.c src/gguf.c $(QUANT_SRCS) src/model.c \
              src/transformer.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/threadpool.c \
              src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_f16: test/test_kv_f16.c src/platform.c src/gguf.c $(QUANT_SRCS) src/model.c \
             src/transformer.c $(TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

PGO_MODEL ?= models/bitnet-b1.58-2B-4T.gguf

pgo:
	@echo "=== PGO Step 1: Instrumented build ==="
	$(MAKE) clean
	$(MAKE) bitnet CFLAGS="$(CFLAGS) -fprofile-instr-generate" LDFLAGS="$(LDFLAGS) -fprofile-instr-generate"
	@echo "=== PGO Step 2: Training run ==="
	LLVM_PROFILE_FILE=default.profraw ./bitnet $(PGO_MODEL) -p "The meaning of life is" -n 128
	@echo "=== PGO Step 3: Merge profile ==="
	xcrun llvm-profdata merge -output=default.profdata default.profraw
	@echo "=== PGO Step 4: Optimized rebuild ==="
	$(MAKE) clean
	$(MAKE) bitnet CFLAGS="$(CFLAGS) -fprofile-instr-use=default.profdata"
	@rm -f default.profraw default.profdata
	@echo "=== PGO build complete ==="

AVX2_QUANT_SRCS = $(QUANT_COMMON) \
    src/quant/x_quant_avx2.c \
    src/quant/i2s_avx2.c src/quant/i2s_scalar.c \
    src/quant/tq2_scalar.c src/quant/tq1_scalar.c \
    src/quant/q8_avx2.c src/quant/q8_scalar.c \
    src/quant/q4_avx2.c src/quant/q4_scalar.c \
    src/quant/q6k_avx2.c src/quant/q6k_scalar.c \
    src/quant/q8k_avx2.c src/quant/q8k_scalar.c \
    src/quant/q4k_avx2.c src/quant/q4k_scalar.c \
    src/quant/q5k_avx2.c src/quant/q5k_scalar.c \
    src/quant/q3k_avx2.c src/quant/q3k_scalar.c \
    src/quant/q2k_avx2.c src/quant/q2k_scalar.c

AVX2_TRANSFORMER_BACKEND = src/transformer/rmsnorm_avx2.c src/transformer/rmsnorm_scalar.c \
    src/transformer/gqa_avx2.c src/transformer/gqa_scalar.c \
    src/transformer/logits_avx2.c src/transformer/logits_scalar.c

AVX2_SRCS = src/platform.c src/gguf.c $(AVX2_QUANT_SRCS) src/model.c \
            src/transformer.c $(AVX2_TRANSFORMER_BACKEND) src/tokenizer.c src/sampler.c \
            src/threadpool.c src/sh_arena.c src/sh_log.c

avx2-check:
	$(CC) -target x86_64-apple-darwin -mavx2 -mfma -mf16c -O3 -Wall -Wextra -Wshadow \
		-std=c11 -Iinclude -fsyntax-only $(AVX2_SRCS)

clean:
	rm -f bitnet src/*.o src/quant/*.o src/transformer/*.o test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_q2k test_e2e test_prefill test_kv_f16 default.profraw default.profdata
