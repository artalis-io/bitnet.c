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

SRCS = src/platform.c src/gguf.c src/quant.c src/model.c \
       src/transformer.c src/tokenizer.c src/sampler.c \
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

# Pattern rule for object files
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# --- Tests ---
.PHONY: debug asan test test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_prefill test_kv_f16 pgo clean

test: test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena

test_gguf: test/test_gguf.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_quant: test/test_quant.c src/quant.c src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_tokenizer: test/test_tokenizer.c src/tokenizer.c src/gguf.c src/platform.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_transformer: test/test_transformer.c src/transformer.c src/model.c \
                  src/gguf.c src/quant.c src/platform.c src/tokenizer.c src/threadpool.c \
                  src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_threadpool: test/test_threadpool.c src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_safety: test/test_safety.c src/platform.c src/gguf.c src/quant.c src/model.c \
             src/transformer.c src/tokenizer.c src/sampler.c src/threadpool.c \
             src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_arena: test/test_arena.c src/sh_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_e2e: test/test_e2e.c src/platform.c src/gguf.c src/quant.c src/model.c \
          src/transformer.c src/tokenizer.c src/sampler.c src/threadpool.c \
          src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_prefill: test/test_prefill.c src/platform.c src/gguf.c src/quant.c src/model.c \
              src/transformer.c src/tokenizer.c src/threadpool.c \
              src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_f16: test/test_kv_f16.c src/platform.c src/gguf.c src/quant.c src/model.c \
             src/transformer.c src/tokenizer.c src/sampler.c src/threadpool.c \
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

clean:
	rm -f bitnet src/*.o test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_arena test_e2e test_prefill test_kv_f16 default.profraw default.profdata
