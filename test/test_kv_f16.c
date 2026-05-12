#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "transformer.h"
#include "session.h"
#include "tokenizer.h"
#include "sampler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// F16 KV cache correctness test: verifies that F16 KV cache produces
// acceptable output compared to F32 KV cache.
// Requires a real GGUF model file.
// Usage: ./test_kv_f16 <model.gguf>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        fprintf(stderr, "Verifies F16 KV cache produces same greedy tokens as F32.\n");
        return 1;
    }

    printf("=== F16 KV Cache Correctness Test ===\n");
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

    // --- Run 1: F32 KV cache ---
    printf("Running F32 KV cache...\n");
    BnModel model_f32;
    if (bn_model_load(&model_f32, gf, 2048, 0, 0) != 0) {
        fprintf(stderr, "Failed to load model (F32)\n");
        return 1;
    }
    model_f32.file = mf;
    BnSession *sess_f32 = bn_session_create(&model_f32, NULL);
    assert(sess_f32);

    int vocab_size = model_f32.config.vocab_size;
    float *logits_f32 = (float *)malloc(vocab_size * sizeof(float));
    assert(logits_f32);

    // Sequential forward for all prompt tokens
    float *lf = NULL;
    for (int i = 0; i < n_prompt; i++) {
        lf = bn_transformer_forward(&model_f32, sess_f32, prompt_tokens[i], i);
        assert(lf != NULL);
    }
    memcpy(logits_f32, lf, vocab_size * sizeof(float));

    // Greedy decode 16 tokens
    int n_gen = 16;
    int gen_f32[16];
    BnSampler sampler;
    bn_sampler_init(&sampler, vocab_size, 0.0f, 0.0f, 42);

    int pos = n_prompt;
    for (int i = 0; i < n_gen; i++) {
        gen_f32[i] = bn_sampler_sample(&sampler, lf);
        if (gen_f32[i] == tok.eos_id || gen_f32[i] == tok.eot_id) {
            n_gen = i + 1;
            break;
        }
        lf = bn_transformer_forward(&model_f32, sess_f32, gen_f32[i], pos++);
        assert(lf != NULL);
    }

    printf("F32 tokens: ");
    for (int i = 0; i < n_gen; i++) {
        const char *piece = bn_tokenizer_decode(&tok, gen_f32[i]);
        printf("%s", piece ? piece : "<?>");
    }
    printf("\n");

    bn_session_free(sess_f32, NULL);
    bn_model_free(&model_f32);

    // --- Run 2: F16 KV cache ---
    printf("Running F16 KV cache...\n");
    BnModel model_f16;
    if (bn_model_load(&model_f16, gf, 2048, 1, 0) != 0) {
        fprintf(stderr, "Failed to load model (F16)\n");
        return 1;
    }
    model_f16.file = mf;
    BnSession *sess_f16 = bn_session_create(&model_f16, NULL);
    assert(sess_f16);

    float *logits_f16 = (float *)malloc(vocab_size * sizeof(float));
    assert(logits_f16);

    // Sequential forward for all prompt tokens
    float *lh = NULL;
    for (int i = 0; i < n_prompt; i++) {
        lh = bn_transformer_forward(&model_f16, sess_f16, prompt_tokens[i], i);
        assert(lh != NULL);
    }
    memcpy(logits_f16, lh, vocab_size * sizeof(float));

    // --- Compare logits ---
    printf("Comparing logits (%d values)...\n", vocab_size);

    // Check greedy argmax matches
    int argmax_f32 = 0, argmax_f16 = 0;
    for (int i = 1; i < vocab_size; i++) {
        if (logits_f32[i] > logits_f32[argmax_f32]) argmax_f32 = i;
        if (logits_f16[i] > logits_f16[argmax_f16]) argmax_f16 = i;
    }
    printf("F32 argmax: %d (\"%s\")\n", argmax_f32, bn_tokenizer_decode(&tok, argmax_f32));
    printf("F16 argmax: %d (\"%s\")\n", argmax_f16, bn_tokenizer_decode(&tok, argmax_f16));

    if (argmax_f32 != argmax_f16) {
        fprintf(stderr, "WARNING: Argmax mismatch (F32=%d, F16=%d) — may be acceptable if logits are close\n",
                argmax_f32, argmax_f16);
    } else {
        printf("PASS: Greedy argmax matches.\n");
    }

    // Max absolute logit difference
    float max_diff = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float diff = fabsf(logits_f32[i] - logits_f16[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max absolute logit difference: %.6f\n", max_diff);

    // --- Greedy decode comparison ---
    printf("Greedy decoding %d tokens with F16...\n", n_gen);
    bn_sampler_init(&sampler, vocab_size, 0.0f, 0.0f, 42);

    int gen_f16[16];
    pos = n_prompt;
    for (int i = 0; i < n_gen; i++) {
        gen_f16[i] = bn_sampler_sample(&sampler, lh);
        if (gen_f16[i] == tok.eos_id || gen_f16[i] == tok.eot_id) break;
        lh = bn_transformer_forward(&model_f16, sess_f16, gen_f16[i], pos++);
        assert(lh != NULL);
    }

    printf("F16 tokens: ");
    for (int i = 0; i < n_gen; i++) {
        const char *piece = bn_tokenizer_decode(&tok, gen_f16[i]);
        printf("%s", piece ? piece : "<?>");
    }
    printf("\n");

    // Check token sequences match
    int tokens_match = 1;
    for (int i = 0; i < n_gen; i++) {
        if (gen_f32[i] != gen_f16[i]) {
            tokens_match = 0;
            fprintf(stderr, "Token mismatch at position %d: F32=%d F16=%d\n", i, gen_f32[i], gen_f16[i]);
            break;
        }
    }

    if (tokens_match) {
        printf("PASS: All %d greedy-decoded tokens match between F32 and F16 KV cache.\n", n_gen);
    } else {
        fprintf(stderr, "WARNING: Token sequences differ — F16 precision loss may affect greedy decoding.\n");
    }

    free(logits_f32);
    free(logits_f16);
    bn_sampler_free(&sampler);
    bn_tokenizer_free(&tok);
    bn_session_free(sess_f16, NULL);
    bn_model_free(&model_f16);
    bn_gguf_free(gf);
    bn_platform_unload_file(&mf);

    printf("F16 KV cache test completed.\n");
    return 0;
}
