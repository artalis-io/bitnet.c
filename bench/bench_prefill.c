#include "platform.h"
#include "gguf.h"
#include "model.h"
#include "tokenizer.h"
#include "session.h"
#include "transformer.h"
#include "threadpool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s <model.gguf> [prompt_tokens] [runs] [threads]\n",
            prog);
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    int prompt_tokens = argc > 2 ? atoi(argv[2]) : 512;
    int runs = argc > 3 ? atoi(argv[3]) : 5;
    int threads = argc > 4 ? atoi(argv[4]) : 8;
    if (prompt_tokens <= 0 || runs <= 0 || threads <= 0) {
        usage(argv[0]);
        return 1;
    }

    BnMappedFile mf = bn_platform_load_file(argv[1]);
    if (!mf.data) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    BnGGUFFile *gf = bn_gguf_open(mf.data, mf.size);
    if (!gf) {
        fprintf(stderr, "failed to parse GGUF\n");
        bn_platform_unload_file(&mf);
        return 1;
    }

    BnModel model;
    if (bn_model_load(&model, gf, prompt_tokens * 2, 1, 0) != 0) {
        fprintf(stderr, "failed to load model\n");
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }
    bn_model_set_file(&model, mf);

    BnThreadPool *pool = threads > 1 ? bn_tp_create(threads - 1) : NULL;
    if (threads > 1 && !pool) {
        fprintf(stderr, "failed to create thread pool\n");
        bn_model_free(&model);
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }
    bn_model_set_thread_pool(&model, pool, 1);

    BnTokenizer tok;
    if (bn_tokenizer_init(&tok, gf) != 0) {
        fprintf(stderr, "failed to initialize tokenizer\n");
        bn_model_free(&model);
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }

    size_t prompt_len = (size_t)prompt_tokens * 6 + 1;
    char *prompt = (char *)malloc(prompt_len);
    int *tokens = (int *)malloc((size_t)(prompt_tokens + 16) * sizeof(int));
    double *times = (double *)malloc((size_t)runs * sizeof(double));
    if (!prompt || !tokens || !times) {
        fprintf(stderr, "allocation failed\n");
        free(prompt);
        free(tokens);
        free(times);
        bn_tokenizer_free(&tok);
        bn_model_free(&model);
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }

    int n_prompt = -1;
    for (int words = prompt_tokens; words > 0; words--) {
        prompt[0] = '\0';
        for (int i = 0; i < words; i++)
            strcat(prompt, "hello ");
        n_prompt = bn_tokenizer_encode(&tok, prompt, tok.add_bos, tokens,
                                       prompt_tokens + 16);
        if (n_prompt == prompt_tokens)
            break;
    }
    if (n_prompt != prompt_tokens) {
        fprintf(stderr, "could not construct %d-token prompt, got %d tokens\n",
                prompt_tokens, n_prompt);
        free(prompt);
        free(tokens);
        free(times);
        bn_tokenizer_free(&tok);
        bn_model_free(&model);
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }

    BnSession *sess = bn_session_create(&model, NULL);
    if (!sess) {
        fprintf(stderr, "failed to create session\n");
        free(prompt);
        free(tokens);
        free(times);
        bn_tokenizer_free(&tok);
        bn_model_free(&model);
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }

    if (bn_transformer_prefill_no_logits(&model, sess, tokens, n_prompt, 0) != 0) {
        fprintf(stderr, "warmup prefill failed\n");
        bn_session_free(sess, NULL);
        free(prompt);
        free(tokens);
        free(times);
        bn_tokenizer_free(&tok);
        bn_model_free(&model);
        bn_gguf_free(gf);
        bn_platform_unload_file(&mf);
        return 1;
    }

    for (int r = 0; r < runs; r++) {
        bn_session_reset(sess, &model);
        double t0 = bn_platform_time_ms();
        if (bn_transformer_prefill_no_logits(&model, sess, tokens,
                                             n_prompt, 0) != 0) {
            fprintf(stderr, "prefill failed at run %d\n", r);
            bn_session_free(sess, NULL);
            free(prompt);
            free(tokens);
            free(times);
            bn_tokenizer_free(&tok);
            bn_model_free(&model);
            bn_gguf_free(gf);
            bn_platform_unload_file(&mf);
            return 1;
        }
        times[r] = bn_platform_time_ms() - t0;
    }

    qsort(times, (size_t)runs, sizeof(double), cmp_double);
    double sum = 0.0;
    for (int r = 0; r < runs; r++) sum += times[r];
    printf("prompt_tokens=%d runs=%d threads=%d median_ms=%.3f mean_ms=%.3f tok_s=%.3f\n",
           n_prompt, runs, threads, times[runs / 2], sum / runs,
           1000.0 * (double)n_prompt / times[runs / 2]);

    bn_session_free(sess, NULL);
    free(prompt);
    free(tokens);
    free(times);
    bn_tokenizer_free(&tok);
    bn_model_free(&model);
    bn_gguf_free(gf);
    bn_platform_unload_file(&mf);
    return 0;
}
