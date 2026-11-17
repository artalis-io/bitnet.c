/* Synthetic raw weights for reference_cuda_bf16_embedding.py.
 * cc -std=c11 -Wall -Wextra -Iinclude -Itest test/reference_cuda_bf16_embedding.c -o /tmp/embedding-weights
 */
#include "cuda_signed_mmq_fixture.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    assert(argc == 2);
    (void)signed_mmq_fixture_input;
    int rows[] = {1, 7, 33};
    for (int r = 0; r < 3; r++) for (int cols = 256; cols <= 512; cols *= 2)
        for (int seed = 0; seed < 2; seed++) {
            size_t n = (size_t)rows[r] * cols;
            size_t bytes = signed_mmq_fixture_bytes(12, n);
            void *raw = malloc(bytes); assert(raw);
            signed_mmq_fixture_weights(raw, 12, n, seed);
            char path[512];
            snprintf(path, sizeof(path), "%s/w-%d-%d-%d.bin", argv[1], rows[r], cols, seed);
            FILE *f = fopen(path, "wb"); assert(f);
            assert(fwrite(raw, 1, bytes, f) == bytes); fclose(f); free(raw);
        }
    return 0;
}
