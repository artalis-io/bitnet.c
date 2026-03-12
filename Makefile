CC      ?= cc
CFLAGS  = -O3 -march=native -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -lm

# On Linux, enable GNU extensions for strdup, qsort_r, clock_gettime, etc.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
CFLAGS += -D_GNU_SOURCE
endif

# OpenMP support (auto-detected, disabled for WASM builds)
ifndef EMSCRIPTEN
OMP_TEST := $(shell echo 'int main(){return 0;}' | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo yes)
ifeq ($(OMP_TEST),yes)
CFLAGS  += -fopenmp
LDFLAGS += -fopenmp
else
# macOS with Homebrew libomp: needs explicit include/lib paths
LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(LIBOMP_PREFIX),)
OMP_TEST_MAC := $(shell echo 'int main(){return 0;}' | $(CC) -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include -L$(LIBOMP_PREFIX)/lib -lomp -x c - -o /dev/null 2>/dev/null && echo yes)
ifeq ($(OMP_TEST_MAC),yes)
CFLAGS  += -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
LDFLAGS += -L$(LIBOMP_PREFIX)/lib -lomp
endif
endif
endif
endif

SRCS = src/platform.c src/gguf.c src/quant.c src/model.c \
       src/transformer.c src/tokenizer.c src/sampler.c src/main.c
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
.PHONY: test test_gguf test_quant test_tokenizer test_transformer test_safety clean

test: test_gguf test_quant test_tokenizer test_transformer test_safety

test_gguf: test/test_gguf.c src/gguf.c src/platform.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_quant: test/test_quant.c src/quant.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_tokenizer: test/test_tokenizer.c src/tokenizer.c src/gguf.c src/platform.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_transformer: test/test_transformer.c src/transformer.c src/model.c \
                  src/gguf.c src/quant.c src/platform.c src/tokenizer.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_safety: test/test_safety.c src/platform.c src/gguf.c src/quant.c src/model.c \
             src/transformer.c src/tokenizer.c src/sampler.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_e2e: test/test_e2e.c src/platform.c src/gguf.c src/quant.c src/model.c \
          src/transformer.c src/tokenizer.c src/sampler.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

clean:
	rm -f bitnet src/*.o test_gguf test_quant test_tokenizer test_transformer test_safety test_e2e
