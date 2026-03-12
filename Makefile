CC      ?= cc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -lm

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
.PHONY: test test_gguf test_quant test_tokenizer test_transformer clean

test: test_gguf test_quant test_tokenizer test_transformer

test_gguf: test/test_gguf.c src/gguf.c src/platform.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_quant: test/test_quant.c src/quant.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_tokenizer: test/test_tokenizer.c src/tokenizer.c src/gguf.c src/platform.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_transformer: test/test_transformer.c src/transformer.c src/model.c \
                  src/gguf.c src/quant.c src/platform.c src/tokenizer.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

test_e2e: test/test_e2e.c src/platform.c src/gguf.c src/quant.c src/model.c \
          src/transformer.c src/tokenizer.c src/sampler.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) && ./$@

clean:
	rm -f bitnet src/*.o test_gguf test_quant test_tokenizer test_transformer test_e2e
