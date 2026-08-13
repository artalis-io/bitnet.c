#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s -m model.gguf (-p prompt | --prompt-token-ids ids) "
                 "[--pos N] [--ctx N] [--sequential] [--flash] [--top-logits N] "
                 "[--generate N] [--gpu-layers N] [--kv-f16] [--no-observer]\n",
                 argv0);
}

static void quiet_llama_log(enum ggml_log_level level, const char *text,
                            void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

static bool parse_token_ids(const char *text, std::vector<llama_token> *out) {
    const char *p = text;
    while (*p) {
        char *end = nullptr;
        long value = std::strtol(p, &end, 10);
        if (end == p || value < 0 || value > INT32_MAX)
            return false;
        out->push_back((llama_token)value);
        if (*end == '\0')
            break;
        if (*end != ',')
            return false;
        p = end + 1;
    }
    return !out->empty();
}

struct ProbeState {
    int dim;
    int n_tokens;
    int wanted_pos;
    int current_pos;
    int row_pos;
    bool all_heads;
    bool token_ids_only;
    float *captured_logits;
    int captured_vocab;
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

    std::printf("%s name=%s type=%s pos=%d layer=%d dim=%d ne=%lld,%lld,%lld,%lld sum=%.9g ss=%.9g min=%.9g max=%.9g first=",
                tag, name, ggml_type_name(t->type), pos, layer, dim,
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3],
                sum, ss, minv, maxv);
    int n = std::min(dim, 16);
    for (int i = 0; i < n; i++)
        std::printf("%s%.9g", i ? "," : "", x[i]);
    std::printf("\n");
}

static void dump_selected_row(const char *tag, const char *name,
                              int layer, int pos,
                              const float *x, int dim) {
    static int last_dump_pos = INT32_MIN;
    const char *path = std::getenv("LLAMA_PROBE_DUMP_PATH");
    const char *wanted_tag = std::getenv("LLAMA_PROBE_DUMP_TAG");
    const char *wanted_name = std::getenv("LLAMA_PROBE_DUMP_NAME");
    const char *wanted_layer = std::getenv("LLAMA_PROBE_DUMP_LAYER");
    if (!path || !wanted_tag || std::strcmp(tag, wanted_tag) != 0)
        return;
    if (wanted_name && std::strcmp(name, wanted_name) != 0)
        return;
    if (wanted_layer && layer != std::atoi(wanted_layer))
        return;
    if (std::getenv("LLAMA_PROBE_DUMP_FIRST") && last_dump_pos == pos)
        return;

    FILE *f = std::fopen(path, "wb");
    if (!f || std::fwrite(x, sizeof(*x), (size_t)dim, f) != (size_t)dim) {
        std::fprintf(stderr,
                     "failed to dump %s layer %d position %d to %s\n",
                     tag, layer, pos, path);
    }
    if (f)
        std::fclose(f);
    last_dump_pos = pos;
}

static bool capture_layer_boundary(struct ggml_tensor *t, bool ask, void *user_data) {
    ProbeState *st = (ProbeState *)user_data;
    if (ask && st->current_pos == st->wanted_pos &&
        std::getenv("LLAMA_PROBE_LIST_NAMES"))
        std::printf("llama_tensor_name name=%s op=%s type=%s ne=%lld,%lld,%lld,%lld\n",
                    t->name, ggml_op_name(t->op), ggml_type_name(t->type),
                    (long long)t->ne[0], (long long)t->ne[1],
                    (long long)t->ne[2], (long long)t->ne[3]);
    bool is_lout = std::strncmp(t->name, "l_out-", 6) == 0;
    bool is_attn_norm = std::strncmp(t->name, "attn_norm-", 10) == 0;
    bool is_layer_norm = std::strncmp(t->name, "norm-", 5) == 0 &&
                         t->op == GGML_OP_RMS_NORM;
    bool is_qcur = std::strncmp(t->name, "Qcur-", 5) == 0;
    bool is_kcur = std::strncmp(t->name, "Kcur-", 5) == 0;
    bool is_vcur = std::strncmp(t->name, "Vcur-", 5) == 0;
    bool is_kq = std::strncmp(t->name, "kq-", 3) == 0;
    bool is_kq_soft_max = std::strncmp(t->name, "kq_soft_max-", 12) == 0;
    bool is_kqv = std::strncmp(t->name, "kqv-", 4) == 0;
    bool is_kqv_out = std::strncmp(t->name, "kqv_out-", 8) == 0;
    bool is_ffn_inp = std::strncmp(t->name, "ffn_inp-", 8) == 0;
    bool is_ffn_norm = std::strncmp(t->name, "ffn_norm-", 9) == 0;
    bool is_ffn_norm_1 = std::strncmp(t->name, "ffn_norm_1-", 11) == 0;
    bool is_ffn_norm_2 = std::strncmp(t->name, "ffn_norm_2-", 11) == 0;
    bool is_ffn_out = std::strncmp(t->name, "ffn_out-", 8) == 0;
    bool is_ffn_up = std::strncmp(t->name, "ffn_up-", 7) == 0;
    bool is_ffn_gate = std::strncmp(t->name, "ffn_gate-", 9) == 0;
    bool is_ffn_swiglu = std::strncmp(t->name, "ffn_swiglu-", 11) == 0;
    bool is_ffn_geglu = std::strncmp(t->name, "ffn_geglu-", 10) == 0;
    bool is_moe_logits = std::strncmp(t->name, "ffn_moe_logits-", 15) == 0;
    bool is_moe_geglu = std::strncmp(t->name, "ffn_moe_geglu-", 14) == 0;
    bool is_moe_out = std::strncmp(t->name, "ffn_moe_out-", 12) == 0;
    bool is_ffn_mlp = std::strncmp(t->name, "ffn_mlp-", 8) == 0;
    bool is_ffn_moe = std::strncmp(t->name, "ffn_moe-", 8) == 0;
    bool is_ffn_moe_combined =
        std::strncmp(t->name, "ffn_moe_combined-", 17) == 0;
    bool is_attn_out = std::strncmp(t->name, "attn_out-", 9) == 0;
    bool is_linear_attn_out =
        std::strncmp(t->name, "linear_attn_out-", 16) == 0;
    bool is_linear_qkv =
        std::strncmp(t->name, "linear_attn_qkv_mixed-", 22) == 0;
    bool is_linear_z = std::strncmp(t->name, "z-", 2) == 0 &&
                       std::strstr(t->name, "reshaped") == nullptr;
    bool is_linear_conv =
        std::strncmp(t->name, "conv_output_silu-", 17) == 0;
    bool is_linear_final =
        std::strncmp(t->name, "final_output-", 13) == 0;
    bool is_ssm_alpha = std::strncmp(t->name, "alpha-", 6) == 0;
    bool is_ssm_beta = std::strncmp(t->name, "beta-", 5) == 0;
    bool is_ssm_softplus =
        std::strncmp(t->name, "a_softplus-", 11) == 0;
    bool is_ssm_beta_sigmoid =
        std::strncmp(t->name, "beta_sigmoid-", 13) == 0;
    bool is_attn_residual =
        std::strncmp(t->name, "attn_residual-", 14) == 0;
    bool is_attn_post_norm =
        std::strncmp(t->name, "attn_post_norm-", 15) == 0;
    bool is_moe_topk = std::strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    bool is_moe_weights =
        std::strncmp(t->name, "ffn_moe_weights_norm-", 21) == 0 &&
        std::strstr(t->name, "reshaped") == nullptr;
    bool is_result_output = std::strcmp(t->name, "result_output") == 0;
    bool is_result_norm = std::strcmp(t->name, "result_norm") == 0;
    bool is_model_input = std::strcmp(t->name, "inp_scaled") == 0;
    if (st->token_ids_only && !is_result_output)
        return ask ? false : true;
    if (!is_lout && !is_attn_norm && !is_layer_norm &&
        !is_qcur && !is_kcur && !is_vcur &&
        !is_kq && !is_kq_soft_max && !is_kqv && !is_kqv_out &&
        !is_ffn_inp && !is_ffn_norm && !is_ffn_norm_1 && !is_ffn_norm_2 &&
        !is_ffn_out && !is_ffn_up && !is_ffn_gate && !is_ffn_swiglu &&
        !is_ffn_geglu && !is_moe_logits && !is_moe_geglu && !is_moe_out &&
        !is_ffn_mlp && !is_ffn_moe &&
        !is_ffn_moe_combined && !is_attn_out && !is_moe_topk &&
        !is_moe_weights && !is_linear_attn_out && !is_linear_qkv &&
        !is_linear_z && !is_linear_conv && !is_linear_final &&
        !is_ssm_alpha && !is_ssm_beta && !is_ssm_softplus &&
        !is_ssm_beta_sigmoid &&
        !is_attn_residual &&
        !is_attn_post_norm && !is_result_norm && !is_result_output &&
        !is_model_input)
        return ask ? false : true;
    if (ask)
        return true;

    int out_layer = -1;
    const char *fmt = (is_result_norm || is_result_output || is_model_input)
                    ? nullptr
                    : is_lout ? "l_out-%d"
                    : is_layer_norm ? "norm-%d"
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
                    : is_ffn_norm_1 ? "ffn_norm_1-%d"
                    : is_ffn_norm_2 ? "ffn_norm_2-%d"
                    : is_ffn_out ? "ffn_out-%d"
                    : is_ffn_up ? "ffn_up-%d"
                    : is_ffn_gate ? "ffn_gate-%d"
                    : is_ffn_swiglu ? "ffn_swiglu-%d"
                    : is_ffn_geglu ? "ffn_geglu-%d"
                    : is_moe_logits ? "ffn_moe_logits-%d"
                    : is_moe_geglu ? "ffn_moe_geglu-%d"
                    : is_moe_out ? "ffn_moe_out-%d"
                    : is_ffn_mlp ? "ffn_mlp-%d"
                    : is_ffn_moe ? "ffn_moe-%d"
                    : is_ffn_moe_combined ? "ffn_moe_combined-%d"
                    : is_moe_topk ? "ffn_moe_topk-%d"
                    : is_moe_weights ? "ffn_moe_weights_norm-%d"
                    : is_linear_qkv ? "linear_attn_qkv_mixed-%d"
                    : is_linear_z ? "z-%d"
                    : is_linear_conv ? "conv_output_silu-%d"
                    : is_linear_final ? "final_output-%d"
                    : is_ssm_alpha ? "alpha-%d"
                    : is_ssm_beta ? "beta-%d"
                    : is_ssm_softplus ? "a_softplus-%d"
                    : is_ssm_beta_sigmoid ? "beta_sigmoid-%d"
                    : is_linear_attn_out ? "linear_attn_out-%d"
                    : is_attn_residual ? "attn_residual-%d"
                    : is_attn_post_norm ? "attn_post_norm-%d"
                    : "attn_out-%d";
    if (fmt && std::sscanf(t->name, fmt, &out_layer) != 1)
        return true;

    if (st->current_pos != st->wanted_pos)
        return true;
    const char *print_layer = std::getenv("LLAMA_PROBE_PRINT_LAYER");
    if (print_layer && out_layer >= 0 && out_layer != std::atoi(print_layer))
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
    if (is_model_input) {
        int row_pos = t->ne[1] > 1 ? st->row_pos : 0;
        int dim = (int)t->ne[0];
        const float *row = base + (size_t)row_pos * (size_t)dim;
        print_stats("llama_model_input", t->name, t, row, dim,
                    st->current_pos, -1);
        dump_selected_row("llama_model_input", t->name, -1,
                          st->current_pos, row, dim);
        return true;
    }
    if (is_result_output) {
        int row_pos = t->ne[1] > 1 ? st->row_pos : 0;
        int vocab = (int)t->ne[0];
        const float *row = base + (size_t)row_pos * (size_t)vocab;
        if (st->captured_logits && vocab == st->captured_vocab)
            std::memcpy(st->captured_logits, row,
                        (size_t)vocab * sizeof(float));
        dump_selected_row("llama_result_output", t->name, -1,
                          st->current_pos, row, vocab);
        return true;
    }
    if (is_moe_topk || is_moe_weights) {
        int token_axis = -1;
        for (int axis = 0; axis < 4; axis++) {
            if (t->ne[axis] == st->n_tokens && st->n_tokens > 1) {
                token_axis = axis;
                break;
            }
        }
        if (token_axis < 0)
            token_axis = 1;
        int expert_axis = -1;
        for (int axis = 3; axis >= 0; axis--) {
            if (axis != token_axis && t->ne[axis] > 1) {
                expert_axis = axis;
                break;
            }
        }
        if (expert_axis < 0)
            expert_axis = 0;
        int count = (int)std::min<int64_t>(t->ne[expert_axis], 16);
        std::printf("%s name=%s pos=%d layer=%d values=",
                    is_moe_topk ? "llama_moe_topk" : "llama_moe_weights",
                    t->name, st->current_pos, out_layer);
        for (int i = 0; i < count; i++) {
            if (i) std::putchar(',');
            size_t offset = (size_t)st->row_pos * t->nb[token_axis] +
                            (size_t)i * t->nb[expert_axis];
            if (is_moe_topk) {
                int32_t value;
                std::memcpy(&value, data + offset, sizeof(value));
                std::printf("%d", value);
            } else {
                float value;
                std::memcpy(&value, data + offset, sizeof(value));
                std::printf("%.9g", value);
            }
        }
        std::putchar('\n');
        return true;
    }
    int row_dim = (int)t->ne[0];
    int row_pos = 0;
    const float *row_base = base;
    if ((is_ssm_beta || is_ssm_beta_sigmoid) &&
        t->ne[2] >= st->n_tokens && st->n_tokens > 1) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
        row_base = (const float *)(data +
            (size_t)st->row_pos * t->nb[2]);
    } else if (t->ne[1] >= st->n_tokens && st->n_tokens > 1) {
        row_pos = st->row_pos;
        row_base = base + (size_t)row_pos * row_dim;
    }
    const char *tag = is_result_norm ? "llama_result_norm"
                    : is_lout ? "llama_lout"
                    : is_layer_norm ? "llama_layer_norm"
                    : is_attn_norm ? "llama_attn_norm"
                    : is_qcur && t->op == GGML_OP_MUL_MAT
                        ? "llama_attn_q_matmul"
                    : is_qcur && t->op == GGML_OP_ADD
                        ? "llama_attn_q_raw"
                    : is_qcur ? "llama_attn_q"
                    : is_kcur && t->op == GGML_OP_MUL_MAT
                        ? "llama_attn_k_matmul"
                    : is_kcur && t->op == GGML_OP_ADD
                        ? "llama_attn_k_raw"
                    : is_kcur ? "llama_attn_k"
                    : is_vcur ? "llama_attn_v"
                    : is_kq_soft_max ? "llama_attn_softmax"
                    : is_kq ? "llama_attn_scores"
                    : is_kqv ? "llama_attn_kqv"
                    : is_kqv_out ? "llama_attn_out"
                    : is_ffn_inp ? "llama_ffn_inp"
                    : is_ffn_norm ? "llama_ffn_norm"
                    : is_ffn_norm_1 ? "llama_ffn_norm_1"
                    : is_ffn_norm_2 ? "llama_ffn_norm_2"
                    : is_ffn_out ? "llama_ffn_out"
                    : is_ffn_up ? "llama_ffn_up"
                    : is_ffn_gate ? "llama_ffn_gate"
                    : is_ffn_swiglu ? "llama_ffn_swiglu"
                    : is_ffn_geglu ? "llama_ffn_geglu"
                    : is_moe_logits ? "llama_moe_logits"
                    : is_moe_geglu ? "llama_moe_geglu"
                    : is_moe_out ? "llama_moe_out"
                    : is_ffn_mlp ? "llama_ffn_mlp"
                    : is_ffn_moe ? "llama_ffn_moe"
                    : is_ffn_moe_combined ? "llama_ffn_moe_combined"
                    : is_linear_qkv ? "llama_ssm_qkv"
                    : is_linear_z ? "llama_ssm_z"
                    : is_linear_conv ? "llama_ssm_conv"
                    : is_linear_final ? "llama_ssm_gate"
                    : is_ssm_alpha ? "llama_ssm_alpha"
                    : is_ssm_beta ? "llama_ssm_beta"
                    : is_ssm_softplus ? "llama_ssm_softplus"
                    : is_ssm_beta_sigmoid ? "llama_ssm_beta_sigmoid"
                    : is_linear_attn_out ? "llama_linear_attn_out"
                    : is_attn_residual ? "llama_attn_residual"
                    : is_attn_post_norm ? "llama_attn_post_norm"
                    : "llama_attn_out";
    print_stats(tag, t->name, t, row_base,
                row_dim, st->current_pos, out_layer);
    dump_selected_row(tag, t->name, out_layer, st->current_pos,
                      row_base, row_dim);
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
    std::vector<llama_token> tokens;
    int wanted_pos = -1;
    int context_size = 0;
    bool sequential = false;
    bool flash = false;
    bool all_heads = false;
    int top_logits = 0;
    int generate = 0;
    int gpu_layers = 0;
    bool kv_f16 = false;
    bool observer = true;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt-token-ids") == 0 &&
                   i + 1 < argc) {
            if (!parse_token_ids(argv[++i], &tokens)) {
                std::fprintf(stderr, "invalid prompt token IDs\n");
                return 1;
            }
        } else if (std::strcmp(argv[i], "--pos") == 0 && i + 1 < argc) {
            wanted_pos = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            context_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--sequential") == 0) {
            sequential = true;
        } else if (std::strcmp(argv[i], "--flash") == 0) {
            flash = true;
        } else if (std::strcmp(argv[i], "--all-heads") == 0) {
            all_heads = true;
        } else if (std::strcmp(argv[i], "--top-logits") == 0 && i + 1 < argc) {
            top_logits = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--gpu-layers") == 0 &&
                   i + 1 < argc) {
            gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--kv-f16") == 0) {
            kv_f16 = true;
        } else if (std::strcmp(argv[i], "--no-observer") == 0) {
            observer = false;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || (prompt.empty() && tokens.empty()) ||
        (!prompt.empty() && !tokens.empty())) {
        usage(argv[0]);
        return 1;
    }

    if (std::getenv("LLAMA_PROBE_QUIET_LOG"))
        llama_log_set(quiet_llama_log, nullptr);
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    ggml_backend_dev_t cpu_devices[] = {
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
        nullptr,
    };
    if (gpu_layers <= 0)
        mparams.devices = cpu_devices;
    mparams.n_gpu_layers = gpu_layers;
    mparams.use_extra_bufts = std::getenv("LLAMA_PROBE_EXTRA_BUFTS") != nullptr;

    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (tokens.empty()) {
        bool add_bos = llama_vocab_get_add_bos(vocab);
        int n_prompt = -llama_tokenize(vocab, prompt.c_str(),
                                       (int)prompt.size(), nullptr, 0,
                                       add_bos, true);
        if (n_prompt <= 0) {
            std::fprintf(stderr, "failed to size tokenization\n");
            llama_model_free(model);
            return 1;
        }
        tokens.resize((size_t)n_prompt);
        if (llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                           tokens.data(), (int)tokens.size(), add_bos, true) < 0) {
            std::fprintf(stderr, "failed to tokenize prompt\n");
            llama_model_free(model);
            return 1;
        }
    }
    int n_prompt = (int)tokens.size();

    int pos = wanted_pos >= 0 ? wanted_pos : n_prompt - 1;
    if (pos < 0 || pos >= n_prompt) {
        std::fprintf(stderr, "position %d outside prompt token range [0,%d)\n",
                     pos, n_prompt);
        llama_model_free(model);
        return 1;
    }

    int dim = llama_model_n_embd(model);

    llama_context_params cparams = llama_context_default_params();
    int native_context = (int)llama_model_n_ctx_train(model);
    int requested_context = context_size > 0 ? context_size : native_context;
    cparams.n_ctx = std::max(requested_context,
                             n_prompt + std::max(generate, 1));
    cparams.n_batch = n_prompt;
    cparams.n_ubatch = n_prompt;
    cparams.type_k = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    cparams.type_v = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    cparams.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                    : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.no_perf = true;
    int vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<float> captured_logits((size_t)vocab_size);
    bool generated_tokens_only = generate > 0 &&
        std::getenv("LLAMA_PROBE_DUMP_PATH") == nullptr &&
        std::getenv("LLAMA_PROBE_LIST_NAMES") == nullptr;
    ProbeState cb_state = { dim, sequential ? 1 : n_prompt, pos, pos,
                            sequential ? 0 : pos, all_heads, generated_tokens_only,
                            captured_logits.data(), vocab_size };
    if (observer) {
        cparams.cb_eval = capture_layer_boundary;
        cparams.cb_eval_user_data = &cb_state;
    }

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

    const float *latest_logits = llama_get_logits_ith(ctx, -1);
    if (!latest_logits) {
        std::fprintf(stderr, "failed to get last-token logits\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    std::memcpy(captured_logits.data(), latest_logits,
                (size_t)vocab_size * sizeof(float));

    if (top_logits > 0) {
        const float *logits = captured_logits.data();
        if (!logits) {
            std::fprintf(stderr, "failed to get logits\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
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
            std::printf("llama_top_logit rank=%d token=%d attr=%u text=%s logit=%.9g\n",
                        i + 1, id,
                        (unsigned)llama_vocab_get_attr(vocab, id),
                        llama_vocab_get_text(vocab, id), logits[id]);
        }
    }

    for (int generated = 0; generated < generate; generated++) {
        int next = -1;
        for (int id = 0; id < vocab_size; id++) {
            if (llama_vocab_is_control(vocab, id) &&
                !llama_vocab_is_eog(vocab, id))
                continue;
            if (next < 0 || captured_logits[(size_t)id] >
                                captured_logits[(size_t)next])
                next = id;
        }
        if (next < 0) {
            std::fprintf(stderr, "no eligible token in greedy sample\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (llama_vocab_is_eog(vocab, next))
            break;
        std::printf("llama_token_id=%d\n", next);
        if (generated + 1 == generate)
            break;
        cb_state.current_pos = n_prompt + generated;
        cb_state.wanted_pos = cb_state.current_pos;
        cb_state.row_pos = 0;
        cb_state.n_tokens = 1;
        llama_batch batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "llama_decode failed at generated position %d\n",
                         cb_state.current_pos);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        latest_logits = llama_get_logits_ith(ctx, -1);
        if (!latest_logits) {
            std::fprintf(stderr,
                         "failed to get generated last-token logits\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        std::memcpy(captured_logits.data(), latest_logits,
                    (size_t)vocab_size * sizeof(float));
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
