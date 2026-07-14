#include "gguf.h"
#include "tokenizer.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf text\n", argv[0]);
        return 2;
    }
    BnGGUFFile *f = bn_gguf_open_file(argv[1]);
    if (!f) return 1;
    BnTokenizer tok;
    if (bn_tokenizer_init(&tok, f) != 0) {
        bn_gguf_free(f);
        return 1;
    }
    int tokens[4096];
    int n = bn_tokenizer_encode(&tok, argv[2], tok.add_bos, tokens, 4096);
    printf("n=%d add_bos=%d\n", n, tok.add_bos);
    for (int i = 0; i < n; i++)
        printf("%s%d", i ? " " : "", tokens[i]);
    printf("\n");
    for (int i = 0; i < n; i++)
        printf("[%d] %s\n", tokens[i], bn_tokenizer_decode(&tok, tokens[i]));
    bn_tokenizer_free(&tok);
    bn_gguf_free(f);
    return 0;
}
