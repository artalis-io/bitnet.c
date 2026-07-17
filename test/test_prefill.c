#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "model_arch.h"
#include "transformer.h"
#include "tokenizer.h"
#include "session.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

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
    BnSession *seq_sess = bn_session_create(&model, NULL);
    if (!sess || !seq_sess) {
        fprintf(stderr, "Failed to create session\n");
        return 1;
    }

    BnTokenizer tok;
    if (bn_tokenizer_init(&tok, gf) != 0) {
        fprintf(stderr, "Failed to init tokenizer\n");
        return 1;
    }

    // Encode test prompt
    const char *prompt = argc >= 3 ? argv[2] : "The capital of France";
    int prompt_tokens[256];
    int n_prompt = bn_tokenizer_encode(&tok, prompt, tok.add_bos,
                                       prompt_tokens, 256);
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

    // --- Run 2: Sequential forward ---
    printf("Running sequential forward...\n");
    float *ls = NULL;
    for (int i = 0; i < n_prompt; i++) {
        ls = bn_transformer_forward(&model, seq_sess, prompt_tokens[i], i);
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

    size_t kv_layers =
        (size_t)bn_model_arch_attention_layer_count(&model.config);
    size_t kv_count = kv_layers * (size_t)model.config.seq_len *
                      (size_t)model.config.kv_dim;
    float max_k_diff = 0.0f, max_v_diff = 0.0f;
    for (size_t i = 0; i < kv_count; i++) {
        float dk = fabsf(sess->state.key_cache[i] - seq_sess->state.key_cache[i]);
        float dv = fabsf(sess->state.value_cache[i] - seq_sess->state.value_cache[i]);
        if (dk > max_k_diff) max_k_diff = dk;
        if (dv > max_v_diff) max_v_diff = dv;
    }
    printf("state kv max_key_diff=%.6f max_value_diff=%.6f\n",
           max_k_diff, max_v_diff);

    if (bn_model_arch_uses_hybrid_ssm(&model.config)) {
        int n_ssm = bn_model_arch_ssm_layer_count(&model.config);
        size_t ssm_count = (size_t)n_ssm *
            (size_t)model.config.ssm_time_step_rank *
            (size_t)model.config.ssm_state_size *
            (size_t)(model.config.ssm_inner_size / model.config.ssm_time_step_rank);
        size_t conv_count = (size_t)n_ssm *
            (size_t)(model.config.ssm_conv_kernel - 1) *
            (size_t)(model.config.ssm_group_count * model.config.ssm_state_size * 2 +
                     model.config.ssm_inner_size);
        float max_ssm_diff = 0.0f, max_conv_diff = 0.0f;
        for (size_t i = 0; i < ssm_count; i++) {
            float d = fabsf(sess->state.ssm_state[i] - seq_sess->state.ssm_state[i]);
            if (d > max_ssm_diff) max_ssm_diff = d;
        }
        for (size_t i = 0; i < conv_count; i++) {
            float d = fabsf(sess->state.ssm_conv_state[i] - seq_sess->state.ssm_conv_state[i]);
            if (d > max_conv_diff) max_conv_diff = d;
        }
        printf("state ssm max_state_diff=%.6f max_conv_diff=%.6f\n",
               max_ssm_diff, max_conv_diff);
    }

    int next_token = top_sequential;
    float *prefill_next = bn_transformer_forward(&model, sess, next_token, n_prompt);
    float *seq_next = bn_transformer_forward(&model, seq_sess, next_token, n_prompt);
    assert(prefill_next && seq_next);
    float max_next_diff = 0.0f;
    int top_prefill_next = 0, top_seq_next = 0;
    for (int i = 0; i < vocab_size; i++) {
        float diff = fabsf(prefill_next[i] - seq_next[i]);
        if (diff > max_next_diff) max_next_diff = diff;
        if (prefill_next[i] > prefill_next[top_prefill_next])
            top_prefill_next = i;
        if (seq_next[i] > seq_next[top_seq_next])
            top_seq_next = i;
    }
    printf("next max_diff=%.6f top_prefill=%d top_sequential=%d\n",
           max_next_diff, top_prefill_next, top_seq_next);
    assert(top_prefill == top_sequential);
    assert(max_diff < 1.0f);
    assert(top_prefill_next == top_seq_next);
    assert(max_next_diff < 1.0f);
    printf("PASS: Prefill logits are numerically coherent with sequential forward.\n");

    free(logits_prefill);
    free(logits_sequential);
    bn_session_free(sess, NULL);
    bn_session_free(seq_sess, NULL);
    bn_tokenizer_free(&tok);
    bn_model_free(&model);
    bn_gguf_free(gf);
    bn_platform_unload_file(&mf);

    printf("Prefill test completed.\n");
    return 0;
}
