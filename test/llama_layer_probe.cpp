#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

static void usage(const char *argv0) {
    std::fprintf(stderr, "usage: %s -m model.gguf -p prompt [--pos N] [--sequential] [--flash] [--top-logits N]\n", argv0);
}

struct ProbeState {
    int dim;
    int n_tokens;
    int wanted_pos;
    int current_pos;
    int row_pos;
    bool all_heads;
};

static void print_stats(const char *tag, const char *name,
                        const struct ggml_tensor *t,
                        const float *x, int dim, int pos, int layer) {
    float sum = 0.0f;
    float ss = 0.0f;
    float minv = x[0];
    float maxv = x[0];
    for (int i = 0; i < dim; i++) {
        float v = x[i];
        sum += v;
        ss += v * v;
        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
    }

    std::printf("%s name=%s pos=%d layer=%d dim=%d ne=%lld,%lld,%lld,%lld sum=%.9g ss=%.9g min=%.9g max=%.9g first=",
                tag, name, pos, layer, dim,
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3],
                sum, ss, minv, maxv);
    int n = std::min(dim, 16);
    for (int i = 0; i < n; i++)
        std::printf("%s%.9g", i ? "," : "", x[i]);
    std::printf("\n");
}

static bool capture_layer_boundary(struct ggml_tensor *t, bool ask, void *user_data) {
    bool is_lout = std::strncmp(t->name, "l_out-", 6) == 0;
    bool is_attn_norm = std::strncmp(t->name, "attn_norm-", 10) == 0;
    bool is_qcur = std::strncmp(t->name, "Qcur-", 5) == 0;
    bool is_kcur = std::strncmp(t->name, "Kcur-", 5) == 0;
    bool is_vcur = std::strncmp(t->name, "Vcur-", 5) == 0;
    bool is_kq = std::strncmp(t->name, "kq-", 3) == 0;
    bool is_kq_soft_max = std::strncmp(t->name, "kq_soft_max-", 12) == 0;
    bool is_kqv = std::strncmp(t->name, "kqv-", 4) == 0;
    bool is_kqv_out = std::strncmp(t->name, "kqv_out-", 8) == 0;
    bool is_ffn_inp = std::strncmp(t->name, "ffn_inp-", 8) == 0;
    bool is_ffn_norm = std::strncmp(t->name, "ffn_norm-", 9) == 0;
    bool is_ffn_out = std::strncmp(t->name, "ffn_out-", 8) == 0;
    bool is_ffn_up = std::strncmp(t->name, "ffn_up-", 7) == 0;
    bool is_ffn_gate = std::strncmp(t->name, "ffn_gate-", 9) == 0;
    bool is_ffn_swiglu = std::strncmp(t->name, "ffn_swiglu-", 11) == 0;
    if (!is_lout && !is_attn_norm && !is_qcur && !is_kcur && !is_vcur &&
        !is_kq && !is_kq_soft_max && !is_kqv && !is_kqv_out &&
        !is_ffn_inp && !is_ffn_norm && !is_ffn_out && !is_ffn_up &&
        !is_ffn_gate && !is_ffn_swiglu)
        return ask ? false : true;
    if (ask)
        return true;

    int out_layer = -1;
    const char *fmt = is_lout ? "l_out-%d"
                    : is_attn_norm ? "attn_norm-%d"
                    : is_qcur ? "Qcur-%d"
                    : is_kcur ? "Kcur-%d"
                    : is_vcur ? "Vcur-%d"
                    : is_kq_soft_max ? "kq_soft_max-%d"
                    : is_kq ? "kq-%d"
                    : is_kqv ? "kqv-%d"
                    : is_kqv_out ? "kqv_out-%d"
                    : is_ffn_inp ? "ffn_inp-%d"
                    : is_ffn_norm ? "ffn_norm-%d"
                    : is_ffn_out ? "ffn_out-%d"
                    : is_ffn_up ? "ffn_up-%d"
                    : is_ffn_gate ? "ffn_gate-%d"
                    : "ffn_swiglu-%d";
    if (std::sscanf(t->name, fmt,
                    &out_layer) != 1)
        return true;

    ProbeState *st = (ProbeState *)user_data;
    if (st->current_pos != st->wanted_pos)
        return true;

    size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> buf(nbytes);
    const uint8_t *data = nullptr;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *)t->data;
    } else {
        ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
        data = buf.data();
    }

    const float *base = (const float *)data;
    int row_dim = (int)t->ne[0];
    int row_pos = 0;
    if (t->ne[1] >= st->n_tokens && st->n_tokens > 1)
        row_pos = st->row_pos;
    const char *tag = is_lout ? "llama_lout"
                    : is_attn_norm ? "llama_attn_norm"
                    : is_qcur ? "llama_attn_q"
                    : is_kcur ? "llama_attn_k"
                    : is_vcur ? "llama_attn_v"
                    : is_kq_soft_max ? "llama_attn_softmax"
                    : is_kq ? "llama_attn_scores"
                    : is_kqv ? "llama_attn_kqv"
                    : is_kqv_out ? "llama_attn_out"
                    : is_ffn_inp ? "llama_ffn_inp"
                    : is_ffn_norm ? "llama_ffn_norm"
                    : is_ffn_out ? "llama_ffn_out"
                    : is_ffn_up ? "llama_ffn_up"
                    : is_ffn_gate ? "llama_ffn_gate"
                    : "llama_ffn_swiglu";
    print_stats(tag, t->name, t, base + (size_t)row_pos * row_dim,
                row_dim, st->current_pos, out_layer);
    if (st->all_heads && t->ne[0] > 0 && t->ne[1] == st->n_tokens &&
        t->ne[2] > 1) {
        char head_tag[128];
        for (int h = 0; h < (int)t->ne[2]; h++) {
            const float *row = base + ((size_t)h * (size_t)t->ne[1] +
                                       (size_t)row_pos) * (size_t)row_dim;
            std::snprintf(head_tag, sizeof(head_tag), "%s_h%d", tag, h);
            print_stats(head_tag, t->name, t, row, row_dim, st->current_pos,
                        out_layer);
        }
    }
    return true;
}

int main(int argc, char **argv) {
    std::string model_path;
    std::string prompt;
    int wanted_pos = -1;
    bool sequential = false;
    bool flash = false;
    bool all_heads = false;
    int top_logits = 0;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--pos") == 0 && i + 1 < argc) {
            wanted_pos = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--sequential") == 0) {
            sequential = true;
        } else if (std::strcmp(argv[i], "--flash") == 0) {
            flash = true;
        } else if (std::strcmp(argv[i], "--all-heads") == 0) {
            all_heads = true;
        } else if (std::strcmp(argv[i], "--top-logits") == 0 && i + 1 < argc) {
            top_logits = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || prompt.empty()) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    ggml_backend_dev_t cpu_devices[] = {
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
        nullptr,
    };
    mparams.devices = cpu_devices;
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = true;

    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                   nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "failed to size tokenization\n");
        llama_model_free(model);
        return 1;
    }

    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                       tokens.data(), (int)tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }

    int pos = wanted_pos >= 0 ? wanted_pos : n_prompt - 1;
    if (pos < 0 || pos >= n_prompt) {
        std::fprintf(stderr, "position %d outside prompt token range [0,%d)\n",
                     pos, n_prompt);
        llama_model_free(model);
        return 1;
    }

    int dim = llama_model_n_embd(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_prompt + 1;
    cparams.n_batch = n_prompt;
    cparams.n_ubatch = n_prompt;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    cparams.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                    : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.no_perf = true;
    ProbeState cb_state = { dim, sequential ? 1 : n_prompt, pos, pos,
                            sequential ? 0 : pos, all_heads };
    cparams.cb_eval = capture_layer_boundary;
    cparams.cb_eval_user_data = &cb_state;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_set_n_threads(ctx, 1, 1);

    if (sequential) {
        for (int i = 0; i < n_prompt; i++) {
            cb_state.current_pos = i;
            llama_batch batch = llama_batch_get_one(&tokens[i], 1);
            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "llama_decode failed at position %d\n", i);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
        }
    } else {
        llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "llama_decode failed\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    if (top_logits > 0) {
        const float *logits = llama_get_logits_ith(ctx, sequential ? 0 : n_prompt - 1);
        if (!logits) {
            std::fprintf(stderr, "failed to get logits\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        int vocab_size = llama_vocab_n_tokens(vocab);
        std::vector<int> ids;
        ids.reserve((size_t)std::min(top_logits, vocab_size));
        for (int id = 0; id < vocab_size; id++) {
            int j = (int)ids.size();
            if (j == top_logits && logits[id] <= logits[ids[j - 1]])
                continue;
            if (j < top_logits) {
                ids.push_back(id);
            } else {
                j--;
            }
            while (j > 0 && logits[id] > logits[ids[j - 1]]) {
                ids[j] = ids[j - 1];
                j--;
            }
            ids[j] = id;
        }
        for (int i = 0; i < (int)ids.size(); i++) {
            int id = ids[i];
            std::printf("llama_top_logit rank=%d token=%d logit=%.9g\n",
                        i + 1, id, logits[id]);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
