/* Real-model replay against supplied independent IDs, on matched schedules.
 * Obtain IDs with the separately linked llama_layer_probe --resume-token-ids.
 * --print-input-ids emits common inputs only; it never emits reference results.
 */
#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "model_internal.h"
#include "session.h"
#include "generate.h"
#include "transformer.h"
#include "tokenizer.h"
#include "threadpool.h"
#include "gpu_backend.h"
#ifdef BN_ENABLE_CUDA
#include "gpu_cuda.h"
#endif
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { decode_steps = 8, resume_tokens = 64, phases = 2 };

static int positive_int(const char *text) {
    char *end;
    errno = 0;
    long n = strtol(text, &end, 10);
    return errno || end == text || *end || n < 1 || n > 8192 ? -1 : (int)n;
}

static int *corpus_tokens(BnTokenizer *tok, const char *path, int needed) {
    BnMappedFile file = bn_platform_load_file(path);
    if (!file.data) return NULL;
    int *tokens = NULL;
    char *text = NULL;
    if (file.size > 64u * 1024u * 1024u ||
        memchr(file.data, 0, file.size)) goto done;
    text = malloc(file.size + 1);
    tokens = malloc((file.size + 16) * sizeof(int));
    if (!text || !tokens) goto fail;
    memcpy(text, file.data, file.size);
    text[file.size] = 0;
    int count = bn_tokenizer_encode(tok, text, tok->add_bos, tokens,
                                     (int)file.size + 16);
    if (count < needed) {
        fprintf(stderr, "Corpus has %d tokens; need %d\n", count, needed);
        goto fail;
    }
    goto done;
fail:
    free(tokens);
    tokens = NULL;
done:
    free(text);
    bn_platform_unload_file(&file);
    return tokens;
}

static float *prefill_chunks(BnModel *model, BnSession *session,
                             const int *tokens, int n, int pos, int chunk) {
    for (int done = 0; done < n;) {
        int count = n - done < chunk ? n - done : chunk;
        if (done + count == n)
            return bn_prefill(model, session, tokens + done, count, pos + done, 0);
        if (bn_prefill_no_logits(model, session, tokens + done,
                                 count, pos + done, 0) != 0) return NULL;
        done += count;
    }
    return NULL;
}

static int exercise(BnModel *model, BnSession *session, const int *tokens,
                    int n, int chunk, const int *expected) {
    bn_session_reset(session, model);
    int vocab = model->config.vocab_size, pos = 0, failures = 0;
    for (int phase = 0; phase < phases; phase++) {
        int count = phase ? resume_tokens : n;
        float *logits = prefill_chunks(model, session, tokens + (phase ? n : 0),
                                       count, pos, chunk);
        if (!logits) return -1;
        pos += count;
        for (int step = 0; step < decode_steps; step++) {
            int index = phase*decode_steps + step, top = 0;
            for (int i = 0; i < vocab; i++) {
                if (!isfinite(logits[i])) {
                    fprintf(stderr, "Nonfinite logits: phase=%d step=%d pos=%d index=%d\n",
                            phase, step, pos, i);
                    return -1;
                }
                if (logits[i] > logits[top]) top = i;
            }
            printf("phase=%s chunk=%d step=%d pos=%d token=%d reference=%d\n",
                   phase ? "resumed" : "initial", chunk, step, pos, top, expected[index]);
            if (top != expected[index]) {
                failures++;
                fprintf(stderr, "Independent token mismatch: phase=%d step=%d candidate_gap=%.9g\n",
                        phase, step, logits[top] - logits[expected[index]]);
            }
            // Teacher-force independent IDs after a mismatch so later scores
            // still compare the same history, including each phase's last ID.
            logits = bn_transformer_forward(model, session, expected[index], pos++);
            if (!logits) return -1;
        }
    }
    return failures;
}

static int read_reference_ids(const char *path, int *ids, int vocab) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open reference IDs: %s\n", path); return -1; }
    char word[64];
    for (int i=0;i<phases*decode_steps;i++) {
        if (fscanf(f, "%63s", word) != 1) { fclose(f); return -1; }
        char *end; errno=0; long value=strtol(word,&end,10);
        if (errno || end==word || *end || value<0 || value>=vocab) { fclose(f); return -1; }
        ids[i]=(int)value;
    }
    int extra=fscanf(f,"%63s",word);
    int rc=extra==EOF && !ferror(f) ? 0 : -1;
    fclose(f); return rc;
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 7 || (argc == 7 && strcmp(argv[6], "--cuda"))) {
        fprintf(stderr, "Usage: %s model.gguf corpus.txt prompt_tokens chunk (reference_ids.txt | --print-input-ids) [--cuda]\n", argv[0]);
        return 1;
    }
    int n = positive_int(argv[3]), chunk = positive_int(argv[4]);
    if (n < 1 || chunk < 1) return 1;
    int cuda = argc == 7;
    int print_inputs = strcmp(argv[5], "--print-input-ids") == 0;
#ifndef BN_ENABLE_CUDA
    if (cuda) {
        fprintf(stderr, "CUDA build required\n");
        return 1;
    }
#endif
    int expected[phases * decode_steps];
    if (!print_inputs && read_reference_ids(argv[5], expected, INT_MAX) != 0) {
        fprintf(stderr, "Reference file must contain exactly %d valid token IDs\n", phases*decode_steps);
        return 1;
    }
    int rc = 1, loaded = 0, tokenizer_ready = 0;
    BnModel model;
    BnTokenizer tok;
    BnSession *session = NULL;
    BnGPUBackend *gpu = NULL;
    int *tokens = NULL;
    BnGGUFFile *file = bn_gguf_open_file(argv[1]);
    if (!file) goto cleanup;
    if (bn_tokenizer_init(&tok, file) != 0) goto cleanup;
    tokenizer_ready = 1;
    tokens = corpus_tokens(&tok, argv[2], n + resume_tokens);
    if (!tokens) goto cleanup;
    if (print_inputs) {
        printf("prompt_token_ids=");
        for (int i=0;i<n;i++) printf("%s%d",i?",":"",tokens[i]);
        printf("\nresume_token_ids=");
        for (int i=0;i<resume_tokens;i++) printf("%s%d",i?",":"",tokens[n+i]);
        printf("\n"); rc=0; goto cleanup;
    }
    if (bn_model_load(&model, file, n + resume_tokens + phases * decode_steps,
                       0, 0) != 0) goto cleanup;
    loaded = 1;
    model.config.flash_attn = 0;
    model.config.kv_f16 = cuda;
    const BnMappedFile *mapping = bn_gguf_primary_file(file);
    if (mapping) bn_model_set_file(&model, *mapping);
    BnThreadPool *pool = bn_tp_create(7);
    if (!pool) goto cleanup;
    bn_model_set_thread_pool(&model, pool, 1);
#ifdef BN_ENABLE_CUDA
    if (cuda) {
        gpu = bn_gpu_cuda_create();
        if (!gpu || bn_model_upload_weights(&model, gpu) != 0 ||
            bn_model_init_gpu_activations(&model, gpu) != 0) goto cleanup;
    }
#endif
    session = bn_session_create(&model, NULL);
    if (!session) goto cleanup;
    for (int i=0;i<phases*decode_steps;i++) {
        if (expected[i] >= model.config.vocab_size) {
            fprintf(stderr, "Reference token ID outside model vocabulary\n");
            goto cleanup;
        }
    }
    printf("backend=%s tokens=%d chunk=%d resume_tokens=%d decode_steps=%d "
           "threads=8 kv=%s flash=off independent=1\n", cuda ? "cuda" : "cpu",
           n, chunk, resume_tokens, decode_steps, cuda ? "f16" : "f32");
    int failures = exercise(&model, session, tokens, n, chunk, expected);
    if (failures < 0) goto cleanup;
    printf("Independent prefill reference %s: %d/%d token IDs match\n",
           failures ? "FAILED" : "PASSED", phases*decode_steps-failures, phases*decode_steps);
    rc = failures ? 1 : 0;

cleanup:
    free(tokens);
    if (session) bn_session_free(session, NULL);
    if (tokenizer_ready) bn_tokenizer_free(&tok);
    if (loaded) bn_model_free(&model);
#ifdef BN_ENABLE_CUDA
    if (gpu) bn_gpu_cuda_destroy(gpu);
#else
    (void)gpu;
#endif
    if (file) bn_gguf_free(file);
    return rc;
}
