#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "transformer.h"
#include "tokenizer.h"
#include "session.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Prefill correctness test: verifies that bn_transformer_prefill() produces
// identical logits to sequential bn_transformer_forward() calls.
// Requires a real GGUF model file.
// Usage: ./test_prefill <model.gguf>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        fprintf(stderr, "Verifies prefill produces identical logits to sequential forward.\n");
        return 1;
    }

    printf("=== Prefill Correctness Test ===\n");
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
    if (bn_model_load(&model, gf, 2048, 0, 0) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    bn_model_set_file(&model, mf);

    BnSession *sess = bn_session_create(&model, NULL);
    if (!sess) {
        fprintf(stderr, "Failed to create session\n");
        return 1;
    }

    BnTokenizer tok;
    if (bn_tokenizer_init(&tok, gf) != 0) {
        fprintf(stderr, "Failed to init tokenizer\n");
        return 1;
    }

    // Encode test prompt
    const char *prompt = "The capital of France";
    int prompt_tokens[64];
    int n_prompt = bn_tokenizer_encode(&tok, prompt, 1, prompt_tokens, 64);
    printf("Prompt: \"%s\" (%d tokens)\n", prompt, n_prompt);
    assert(n_prompt >= 2);

    int vocab_size = model.config.vocab_size;
    float *logits_prefill = (float *)malloc(vocab_size * sizeof(float));
    float *logits_sequential = (float *)malloc(vocab_size * sizeof(float));
    assert(logits_prefill && logits_sequential);

    // --- Run 1: Prefill ---
    printf("Running prefill...\n");
    float *lp = bn_transformer_prefill(&model, sess, prompt_tokens, n_prompt, 0);
    assert(lp != NULL);
    memcpy(logits_prefill, lp, vocab_size * sizeof(float));

    // --- Reset ALL state (KV cache + SSM state) ---
    bn_session_reset(sess, &model);

    // --- Run 2: Sequential forward ---
    printf("Running sequential forward...\n");
    float *ls = NULL;
    for (int i = 0; i < n_prompt; i++) {
        ls = bn_transformer_forward(&model, sess, prompt_tokens[i], i);
        assert(ls != NULL);
    }
    memcpy(logits_sequential, ls, vocab_size * sizeof(float));

    // --- Compare ---
    printf("Comparing logits (%d values)...\n", vocab_size);
    float max_diff = 0.0f;
    int max_idx = -1;
    int top_prefill = 0;
    int top_sequential = 0;
    for (int i = 0; i < vocab_size; i++) {
        float diff = logits_prefill[i] - logits_sequential[i];
        if (diff < 0.0f) diff = -diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
        if (logits_prefill[i] > logits_prefill[top_prefill])
            top_prefill = i;
        if (logits_sequential[i] > logits_sequential[top_sequential])
            top_sequential = i;
    }
    printf("max_diff=%.6f at %d top_prefill=%d top_sequential=%d\n",
           max_diff, max_idx, top_prefill, top_sequential);
    assert(top_prefill == top_sequential);
    assert(max_diff < 1.0f);
    printf("PASS: Prefill logits are numerically coherent with sequential forward.\n");

    free(logits_prefill);
    free(logits_sequential);
    bn_session_free(sess, NULL);
    bn_tokenizer_free(&tok);
    bn_model_free(&model);
    bn_gguf_free(gf);
    bn_platform_unload_file(&mf);

    printf("Prefill test completed.\n");
    return 0;
}
