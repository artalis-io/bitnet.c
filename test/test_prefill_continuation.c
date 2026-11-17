/* Real-model regression: whole versus chunked prefill, decode, resume, decode.
 * This is an execution-invariance check, not an independent model reference.
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
                    int n, int chunk, int *expected, float *reference,
                    int record) {
    bn_session_reset(session, model);
    int vocab = model->config.vocab_size, pos = 0, failures = 0;
    float *logits = NULL;
    for (int phase = 0; phase < phases; phase++) {
        int count = phase ? resume_tokens : n;
        logits = prefill_chunks(model, session, tokens + (phase ? n : 0),
                                 count, pos, record ? count : chunk);
        if (!logits) {
            fprintf(stderr, "Prefill failed: phase=%d pos=%d count=%d chunk=%d\n",
                    phase, pos, count, chunk);
            return -1;
        }
        pos += count;
        for (int step = 0; step < decode_steps; step++) {
            int index = phase * decode_steps + step, top = 0;
            float *ref = reference + (size_t)index * vocab;
            double squared = 0.0;
            float max_diff = 0.0f;
            for (int i = 0; i < vocab; i++) {
                if (!isfinite(logits[i])) {
                    fprintf(stderr, "Nonfinite logits: phase=%d step=%d pos=%d "
                            "index=%d value=%g\n", phase, step, pos, i, logits[i]);
                    return -1;
                }
                if (logits[i] > logits[top]) top = i;
                if (!record) {
                    float diff = fabsf(logits[i] - ref[i]);
                    if (diff > max_diff) max_diff = diff;
                    squared += (double)diff * diff;
                }
            }
            if (record) {
                expected[index] = top;
                memcpy(ref, logits, (size_t)vocab * sizeof(float));
            } else {
                printf("phase=%s chunk=%d step=%d pos=%d token=%d reference=%d "
                       "max_diff=%.9g rms_diff=%.9g\n",
                       phase ? "resumed" : "initial", chunk, step, pos, top,
                       expected[index], max_diff, sqrt(squared / vocab));
                if (top != expected[index]) {
                    failures++;
                    fprintf(stderr, "Token mismatch: phase=%d step=%d "
                            "reference_gap=%.9g candidate_gap=%.9g\n",
                            phase, step, ref[expected[index]] - ref[top],
                            logits[top] - logits[expected[index]]);
                }
            }
            /* Replay identical histories even after a mismatch, to distinguish
             * state differences from autoregressive divergence. */
            logits = bn_transformer_forward(model, session, expected[index], pos++);
            if (!logits) return -1;
        }
    }
    return failures;
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 6 || (argc == 6 && strcmp(argv[5], "--cuda"))) {
        fprintf(stderr, "Usage: %s model.gguf corpus.txt prompt_tokens chunk [--cuda]\n", argv[0]);
        return 1;
    }
    int n = positive_int(argv[3]), chunk = positive_int(argv[4]);
    if (n < 1 || chunk < 1) return 1;
    int cuda = argc == 6;
#ifndef BN_ENABLE_CUDA
    if (cuda) {
        fprintf(stderr, "CUDA build required\n");
        return 1;
    }
#endif
    int rc = 1, loaded = 0, tokenizer_ready = 0;
    BnModel model;
    BnTokenizer tok;
    BnSession *session = NULL;
    BnGPUBackend *gpu = NULL;
    int *tokens = NULL;
    float *reference = NULL;
    BnGGUFFile *file = bn_gguf_open_file(argv[1]);
    if (!file) goto cleanup;
    if (bn_model_load(&model, file, n + resume_tokens + phases * decode_steps,
                       0, 0) != 0) goto cleanup;
    loaded = 1;
    model.config.flash_attn = 0;
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
    if (bn_tokenizer_init(&tok, file) != 0) goto cleanup;
    tokenizer_ready = 1;
    tokens = corpus_tokens(&tok, argv[2], n + resume_tokens);
    session = bn_session_create(&model, NULL);
    reference = malloc((size_t)phases * decode_steps *
                         model.config.vocab_size * sizeof(float));
    if (!tokens || !session || !reference) goto cleanup;
    int expected[phases * decode_steps];
    printf("backend=%s tokens=%d chunk=%d resume_tokens=%d decode_steps=%d "
           "threads=8 kv=f32 flash=off\n", cuda ? "cuda" : "cpu",
           n, chunk, resume_tokens, decode_steps);
    if (exercise(&model, session, tokens, n, n, expected, reference, 1) != 0)
        goto cleanup;
    int failures = exercise(&model, session, tokens, n, chunk,
                              expected, reference, 0);
    if (failures < 0) goto cleanup;
    printf("Prefill continuation %s: %d/%d token IDs match\n",
           failures ? "FAILED" : "PASSED",
           phases * decode_steps - failures, phases * decode_steps);
    rc = failures ? 1 : 0;
cleanup:
    free(reference);
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
