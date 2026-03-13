CC      ?= cc
LDFLAGS = -lm

# Platform-specific arch flags:
# -mcpu=apple-m1 on Darwin enables FP16 vector arithmetic + dotprod.
# -march=native on Apple clang misses __ARM_FEATURE_FP16_VECTOR_ARITHMETIC.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
CFLAGS  = -O3 -mcpu=apple-m1 -Wall -Wextra -std=c11 -Iinclude
else
CFLAGS  = -O3 -march=native -Wall -Wextra -std=c11 -Iinclude
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

# Pattern rule for object files
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# --- Tests ---
.PHONY: debug test test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety clean

test: test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety

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

test_e2e: test/test_e2e.c src/platform.c src/gguf.c src/quant.c src/model.c \
          src/transformer.c src/tokenizer.c src/sampler.c src/threadpool.c \
          src/sh_arena.c src/sh_log.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

clean:
	rm -f bitnet src/*.o test_gguf test_quant test_tokenizer test_transformer test_threadpool test_safety test_e2e
