#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct common_sampler;
struct llama_context;
struct llama_model;
struct llama_vocab;

using CommonSamplerSample = int (*)(common_sampler *, llama_context *, int,
                                    bool);
using GetLogits = float *(*)(llama_context *, int32_t);
using GetModel = const llama_model *(*)(const llama_context *);
using GetVocab = const llama_vocab *(*)(const llama_model *);
using VocabSize = int32_t (*)(const llama_vocab *);

int common_sampler_sample(common_sampler *sampler, llama_context *ctx,
                          int idx, bool grammar_first) {
    static CommonSamplerSample next = nullptr;
    if (!next) {
        next = reinterpret_cast<CommonSamplerSample>(
            dlsym(RTLD_NEXT,
                  "_Z21common_sampler_sampleP14common_samplerP13llama_contextib"));
        if (!next) {
            std::fprintf(stderr, "llama token trace: %s\n", dlerror());
            std::abort();
        }
    }
    const char *top_env = std::getenv("LLAMA_TOKEN_TRACE_TOP");
    static int step = 0;
    if (top_env && std::atoi(top_env) > 0) {
        static GetLogits get_logits = reinterpret_cast<GetLogits>(
            dlsym(RTLD_NEXT, "llama_get_logits_ith"));
        static GetModel get_model = reinterpret_cast<GetModel>(
            dlsym(RTLD_NEXT, "llama_get_model"));
        static GetVocab get_vocab = reinterpret_cast<GetVocab>(
            dlsym(RTLD_NEXT, "llama_model_get_vocab"));
        static VocabSize vocab_size = reinterpret_cast<VocabSize>(
            dlsym(RTLD_NEXT, "llama_vocab_n_tokens"));
        float *logits = get_logits ? get_logits(ctx, idx) : nullptr;
        int32_t n_vocab = get_model && get_vocab && vocab_size
            ? vocab_size(get_vocab(get_model(ctx))) : 0;
        int top_k = std::atoi(top_env);
        if (top_k > 32) top_k = 32;
        int ids[32];
        int n_top = 0;
        for (int id = 0; logits && id < n_vocab; id++) {
            int j = n_top;
            if (j == top_k && logits[id] <= logits[ids[j - 1]])
                continue;
            if (j < top_k) {
                ids[j] = id;
                n_top++;
            } else {
                j--;
            }
            while (j > 0 && logits[id] > logits[ids[j - 1]]) {
                ids[j] = ids[j - 1];
                j--;
            }
            ids[j] = id;
        }
        for (int rank = 0; rank < n_top; rank++)
            std::fprintf(stderr,
                         "llama_top_logit step=%d rank=%d token=%d logit=%.9g\n",
                         step, rank + 1, ids[rank], logits[ids[rank]]);
    }
    int token = next(sampler, ctx, idx, grammar_first);
    std::fprintf(stderr, "llama_token_id=%d\n", token);
    step++;
    return token;
}
