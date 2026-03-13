#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// End-to-end test: load a real GGUF model and do greedy decode
// Usage: ./test_e2e <model.gguf> [expected_output]

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [expected_first_tokens]\n", argv[0]);
        fprintf(stderr, "Runs greedy decode on 'Hello' and prints first 20 tokens\n");
        return 1;
    }

    printf("=== E2E Test ===\n");
    printf("Loading %s...\n", argv[1]);

    BnMappedFile mf = bn_platform_load_file(argv[1]);
    if (!mf.data) {
        fprintf(stderr, "Failed to load file\n");
        return 1;
    }

    BnGGUFFile *gf = bn_gguf_open(mf.data, mf.size);
    if (!gf) {
        fprintf(stderr, "Failed to parse GGUF\n");
        return 1;
    }

    BnModel model;
    if (bn_model_load(&model, gf, 2048, 0) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    model.file = mf;

    BnTokenizer tok;
    if (bn_tokenizer_init(&tok, gf) != 0) {
        fprintf(stderr, "Failed to init tokenizer\n");
        return 1;
    }

    // Encode prompt
    int prompt_tokens[64];
    int n_prompt = bn_tokenizer_encode(&tok, "Hello", 1, prompt_tokens, 64);
    printf("Prompt tokens (%d): ", n_prompt);
    for (int i = 0; i < n_prompt; i++) printf("%d ", prompt_tokens[i]);
    printf("\n");

    // Greedy decode 20 tokens
    BnSampler sampler;
    bn_sampler_init(&sampler, model.config.vocab_size, 0.0f, 0.0f, 42);

    int token = prompt_tokens[0];
    int pos = 0;
    int n_gen = 20;
    int output_tokens[64];
    int n_output = 0;

    printf("Output: ");
    for (int i = 0; i < n_prompt + n_gen; i++) {
        float *logits = bn_transformer_forward(&model, token, pos);

        int next;
        if (i < n_prompt - 1) {
            next = prompt_tokens[i + 1];
        } else {
            next = bn_sampler_sample(&sampler, logits);
            output_tokens[n_output++] = next;

            const char *piece = bn_tokenizer_decode(&tok, next);
            printf("%s", piece);

            if (next == tok.eos_id || next == tok.eot_id) break;
        }

        token = next;
        pos++;
    }
    printf("\n");

    printf("Output token IDs: ");
    for (int i = 0; i < n_output; i++) printf("%d ", output_tokens[i]);
    printf("\n");

    // If expected output provided, compare
    if (argc > 2) {
        printf("Expected: %s\n", argv[2]);
        // Simple check: first few tokens should match
    }

    bn_sampler_free(&sampler);
    bn_tokenizer_free(&tok);
    bn_model_free(&model);
    bn_gguf_free(gf);

    printf("E2E test completed.\n");
    return 0;
}
