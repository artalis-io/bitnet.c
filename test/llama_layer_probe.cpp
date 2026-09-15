#include "llama.h"
#include "ggml-backend.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-ext.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s -m model.gguf (-p prompt | --prompt-token-ids ids) "
                 "[--pos N] [--ctx N] [--sequential] [--flash|--no-flash] [--no-mmap] [--no-extra-bufts] [--top-logits N] "
                 "[--generate N | --decode-token-ids ids] [--gpu-layers N] [--kv-f16] [--no-observer]\n",
                 argv0);
}

static void quiet_llama_log(enum ggml_log_level level, const char *text,
                            void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

static int dump_completed_kv_cache(llama_context *ctx) {
    const char *path = std::getenv("LLAMA_PROBE_KV_DUMP_PATH");
    const char *layer_text = std::getenv("LLAMA_PROBE_KV_DUMP_LAYER");
    if (!path || !layer_text)
        return 0;
    int layer = std::atoi(layer_text);
    llama_memory_i *memory = llama_get_memory(ctx);
    llama_kv_cache *cache = dynamic_cast<llama_kv_cache *>(memory);
    if (!cache) {
        if (auto *iswa = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
            for (llama_kv_cache *candidate : {iswa->get_base(), iswa->get_swa()}) {
                if (!candidate) continue;
                const auto ids = candidate->get_layer_ids();
                if (std::find(ids.begin(), ids.end(), (uint32_t)layer) != ids.end()) {
                    cache = candidate;
                    break;
                }
            }
        }
    }
    if (!cache) {
        std::fprintf(stderr, "unsupported KV memory implementation\n");
        return -1;
    }
    const char *kind = std::getenv("LLAMA_PROBE_KV_DUMP_KIND");
    ggml_context *view_ctx = nullptr;
    ggml_tensor *tensor = nullptr;
    if (kind && std::strcmp(kind, "v") == 0) {
        ggml_init_params params = {
            4 * ggml_tensor_overhead(), nullptr, true
        };
        view_ctx = ggml_init(params);
        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = 0;
        sinfo.s1 = 0;
        tensor = view_ctx
            ? cache->get_v(view_ctx, layer, cache->get_size(), sinfo)
            : nullptr;
    } else {
        tensor = cache->get_k_storage(layer);
    }
    if (!tensor || (tensor->type != GGML_TYPE_F32 &&
                    tensor->type != GGML_TYPE_F16)) {
        std::fprintf(stderr,
                     "layer %d KV storage is unavailable or not F32/F16\n",
                     layer);
        if (view_ctx) ggml_free(view_ctx);
        return -1;
    }
    std::vector<uint8_t> data(ggml_nbytes(tensor));
    llama_synchronize(ctx);
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
    const size_t count = ggml_nelements(tensor);
    std::vector<float> converted;
    const void *write_data = data.data();
    size_t write_bytes = data.size();
    if (tensor->type == GGML_TYPE_F16) {
        converted.resize(count);
        const ggml_fp16_t *packed =
            reinterpret_cast<const ggml_fp16_t *>(data.data());
        for (size_t i = 0; i < count; i++)
            converted[i] = ggml_fp16_to_fp32(packed[i]);
        write_data = converted.data();
        write_bytes = converted.size() * sizeof(converted[0]);
    }
    FILE *file = std::fopen(path, "wb");
    if (!file || std::fwrite(write_data, 1, write_bytes, file) != write_bytes) {
        if (file) std::fclose(file);
        if (view_ctx) ggml_free(view_ctx);
        std::fprintf(stderr, "failed to dump completed K storage to %s\n", path);
        return -1;
    }
    std::fclose(file);
    std::printf("llama_kv_dump layer=%d type=%s ne=%lld,%lld,%lld,%lld bytes=%zu\n",
                layer, ggml_type_name(tensor->type),
                (long long)tensor->ne[0], (long long)tensor->ne[1],
                (long long)tensor->ne[2], (long long)tensor->ne[3], write_bytes);
    if (view_ctx) ggml_free(view_ctx);
    return 0;
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
        if (*p == '\0')
            return false;
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

// Raw greedy completion does not mask control tokens. In particular, a
// generated chat delimiter must remain eligible when tracing raw prompts.
static int greedy_token(const float *logits, int count) {
    int next = -1;
    for (int id = 0; id < count; id++)
        if (next < 0 || logits[id] > logits[next]) next = id;
    return next;
}

static int probe_self_test() {
    std::vector<float> logits(248046, -1.0f);
    logits[248045] = 2.0f; // Control-token-shaped ID; no filtering.
    assert(greedy_token(logits.data(), (int)logits.size()) == 248045);
    logits[13] = 2.0f;
    assert(greedy_token(logits.data(), (int)logits.size()) == 13);
    assert(greedy_token(nullptr, 0) == -1);
    std::vector<llama_token> ids;
    assert(parse_token_ids("5388,13,271,248045", &ids));
    assert(ids.size() == 4 && ids.back() == 248045);
    for (const char *bad : {"", "-1", "2147483648", "1x", "1,,2", "1,"}) {
        ids.clear();
        assert(!parse_token_ids(bad, &ids));
    }
    std::puts("Layer probe self-tests PASSED");
    return 0;
}

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

    std::printf("%s name=%s type=%s pos=%d layer=%d dim=%d "
                "ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu "
                "sum=%.9g ss=%.9g min=%.9g max=%.9g first=",
                tag, name, ggml_type_name(t->type), pos, layer, dim,
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3],
                t->nb[0], t->nb[1], t->nb[2], t->nb[3],
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
    static int dump_match_pos = INT32_MIN;
    static int dump_match = 0;
    const char *path = std::getenv("LLAMA_PROBE_DUMP_PATH");
    const char *wanted_tag = std::getenv("LLAMA_PROBE_DUMP_TAG");
    const char *wanted_name = std::getenv("LLAMA_PROBE_DUMP_NAME");
    const char *wanted_layer = std::getenv("LLAMA_PROBE_DUMP_LAYER");
    const char *wanted_occurrence =
        std::getenv("LLAMA_PROBE_DUMP_OCCURRENCE");
    if (!path || !wanted_tag || std::strcmp(tag, wanted_tag) != 0)
        return;
    if (wanted_name && std::strcmp(name, wanted_name) != 0)
        return;
    if (wanted_layer && layer != std::atoi(wanted_layer))
        return;
    if (dump_match_pos != pos) {
        dump_match_pos = pos;
        dump_match = 0;
    }
    dump_match++;
    if (wanted_occurrence && dump_match != std::atoi(wanted_occurrence))
        return;
    if (std::getenv("LLAMA_PROBE_DUMP_FIRST") && last_dump_pos == pos)
        return;

    FILE *f = std::fopen(path,
                         std::getenv("LLAMA_PROBE_DUMP_APPEND") ? "ab" : "wb");
    if (!f || std::fwrite(x, sizeof(*x), (size_t)dim, f) != (size_t)dim) {
        std::fprintf(stderr,
                     "failed to dump %s layer %d position %d to %s\n",
                     tag, layer, pos, path);
    }
    if (f)
        std::fclose(f);
    last_dump_pos = pos;
}

static void dump_selected_raw(const char *tag, const char *name,
                              int layer, const void *data, size_t size) {
    static int dump_match = 0;
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
    dump_match++;
    const char *wanted_occurrence =
        std::getenv("LLAMA_PROBE_DUMP_OCCURRENCE");
    if (wanted_occurrence && dump_match != std::atoi(wanted_occurrence))
        return;
    FILE *f = std::fopen(path, "wb");
    if (!f || std::fwrite(data, 1, size, f) != size)
        std::fprintf(stderr, "failed to dump %s layer %d to %s\n",
                     tag, layer, path);
    if (f)
        std::fclose(f);
}

static bool capture_layer_boundary(struct ggml_tensor *t, bool ask, void *user_data) {
    ProbeState *st = (ProbeState *)user_data;
    bool selected_pos = st->current_pos == st->wanted_pos ||
                        std::getenv("LLAMA_PROBE_ALL_POS");
    if (ask && selected_pos &&
        std::getenv("LLAMA_PROBE_LIST_NAMES"))
        std::printf("llama_tensor_name name=%s op=%s type=%s ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu\n",
                    t->name, ggml_op_name(t->op), ggml_type_name(t->type),
                    (long long)t->ne[0], (long long)t->ne[1],
                    (long long)t->ne[2], (long long)t->ne[3],
                    t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    bool is_lout = std::strncmp(t->name, "l_out-", 6) == 0;
    bool is_attn_norm = std::strncmp(t->name, "attn_norm-", 10) == 0;
    bool is_layer_norm = std::strncmp(t->name, "norm-", 5) == 0 &&
                         t->op == GGML_OP_RMS_NORM;
    bool is_qcur = std::strncmp(t->name, "Qcur-", 5) == 0;
    bool is_kcur = std::strncmp(t->name, "Kcur-", 5) == 0;
    bool is_vcur = std::strncmp(t->name, "Vcur-", 5) == 0;
    bool is_qcur_normed = std::strncmp(t->name, "Qcur_normed-", 12) == 0;
    bool is_kcur_normed = std::strncmp(t->name, "Kcur_normed-", 12) == 0;
    bool is_vcur_normed = std::strncmp(t->name, "Vcur_normed-", 12) == 0;
    bool is_qcur_pos = std::strncmp(t->name, "Qcur_pos-", 9) == 0 &&
                       std::strstr(t->name, "view") == nullptr;
    bool is_kcur_pos = std::strncmp(t->name, "Kcur_pos-", 9) == 0 &&
                       std::strstr(t->name, "view") == nullptr;
    bool is_kq = std::strncmp(t->name, "kq-", 3) == 0;
    bool is_kq_soft_max = std::strncmp(t->name, "kq_soft_max-", 12) == 0;
    bool is_kqv = std::strncmp(t->name, "kqv-", 4) == 0;
    bool is_cache_k_operand = std::strncmp(t->name, "cache_k_l", 9) == 0 &&
        (t->op == GGML_OP_SET_ROWS ||
         std::strstr(t->name, "(copy)") != nullptr ||
         (std::strstr(t->name, "(permuted)") != nullptr &&
          std::strstr(t->name, "(copy)") == nullptr));
    bool is_cache_v_operand = std::strncmp(t->name, "cache_v_l", 9) == 0 &&
        (t->op == GGML_OP_SET_ROWS ||
         std::strstr(t->name, "(copy)") != nullptr ||
         (std::strstr(t->name, "(permuted)") != nullptr &&
          std::strstr(t->name, "(copy)") == nullptr));
    bool is_kqv_out = std::strncmp(t->name, "kqv_out-", 8) == 0;
    bool is_ffn_inp = std::strncmp(t->name, "ffn_inp-", 8) == 0;
    bool is_kqv_wo = std::strncmp(t->name, "kqv_wo-", 7) == 0;
    bool is_ffn_norm = std::strncmp(t->name, "ffn_norm-", 9) == 0;
    bool is_ffn_norm_1 = std::strncmp(t->name, "ffn_norm_1-", 11) == 0;
    bool is_ffn_norm_2 = std::strncmp(t->name, "ffn_norm_2-", 11) == 0;
    bool is_ffn_out = std::strncmp(t->name, "ffn_out-", 8) == 0;
    bool is_ffn_up = std::strncmp(t->name, "ffn_up-", 7) == 0;
    bool is_ffn_gate = std::strncmp(t->name, "ffn_gate-", 9) == 0;
    bool is_ffn_swiglu = std::strncmp(t->name, "ffn_swiglu-", 11) == 0;
    bool is_ffn_geglu = std::strncmp(t->name, "ffn_geglu-", 10) == 0;
    bool is_moe_logits = std::strncmp(t->name, "ffn_moe_logits-", 15) == 0;
    bool is_moe_probs = std::strncmp(t->name, "ffn_moe_probs-", 14) == 0;
    bool is_moe_gate = std::strncmp(t->name, "ffn_moe_gate-", 13) == 0;
    bool is_moe_up = std::strncmp(t->name, "ffn_moe_up-", 11) == 0;
    bool is_moe_swiglu = std::strncmp(t->name, "ffn_moe_swiglu-", 15) == 0;
    bool is_moe_down = std::strncmp(t->name, "ffn_moe_down-", 13) == 0;
    bool is_moe_weighted = std::strncmp(t->name, "ffn_moe_weighted-", 17) == 0 &&
                           std::strstr(t->name, "view") == nullptr;
    bool is_moe_geglu = std::strncmp(t->name, "ffn_moe_geglu-", 14) == 0;
    bool is_moe_out = std::strncmp(t->name, "ffn_moe_out-", 12) == 0;
    bool is_shared_expert = std::strncmp(t->name, "ffn_shexp-", 10) == 0;
    bool is_shared_gate =
        std::strncmp(t->name, "shared_expert_gate-", 19) == 0;
    bool is_shared_gate_sigmoid =
        std::strncmp(t->name, "shared_expert_gate_sigmoid-", 27) == 0;
    bool is_shared_expert_gated =
        std::strncmp(t->name, "ffn_shexp_gated-", 16) == 0;
    bool is_ffn_mlp = std::strncmp(t->name, "ffn_mlp-", 8) == 0;
    bool is_ffn_moe = std::strncmp(t->name, "ffn_moe-", 8) == 0;
    bool is_ffn_moe_combined =
        std::strncmp(t->name, "ffn_moe_combined-", 17) == 0;
    bool is_attn_pregate = std::strncmp(t->name, "attn_pregate-", 13) == 0;
    bool is_attn_gated = std::strncmp(t->name, "attn_gated-", 11) == 0;
    bool is_gate_sigmoid = std::strncmp(t->name, "gate_sigmoid-", 13) == 0;
    bool is_attn_output = std::strncmp(t->name, "attn_output-", 12) == 0;
    bool is_attn_out = std::strncmp(t->name, "attn_out-", 9) == 0 ||
                       is_attn_output ||
                       is_attn_pregate || is_attn_gated || is_gate_sigmoid;
    bool is_linear_attn_out =
        std::strncmp(t->name, "linear_attn_out-", 16) == 0;
    bool is_linear_qkv =
        std::strncmp(t->name, "linear_attn_qkv_mixed-", 22) == 0;
    bool is_linear_z = std::strncmp(t->name, "z-", 2) == 0 &&
                       std::strstr(t->name, "reshaped") == nullptr;
    bool is_linear_conv =
        std::strncmp(t->name, "conv_output_silu-", 17) == 0;
    bool is_linear_conv_raw =
        std::strncmp(t->name, "conv_output_raw-", 16) == 0;
    bool is_linear_final =
        std::strncmp(t->name, "final_output-", 13) == 0;
    bool is_ssm_alpha = std::strncmp(t->name, "alpha-", 6) == 0;
    bool is_ssm_beta = std::strncmp(t->name, "beta-", 5) == 0;
    bool is_ssm_softplus =
        std::strncmp(t->name, "a_softplus-", 11) == 0;
    bool is_ssm_decay_log = std::strncmp(t->name, "gate-", 5) == 0 &&
                             t->op == GGML_OP_MUL;
    bool is_ssm_beta_sigmoid =
        std::strncmp(t->name, "beta_sigmoid-", 13) == 0;
    bool is_ssm_q_norm =
        std::strncmp(t->name, "q_conv_predelta-", 16) == 0;
    bool is_ssm_k_norm =
        std::strncmp(t->name, "k_conv_predelta-", 16) == 0;
    bool is_ssm_gdn = t->op == GGML_OP_GATED_DELTA_NET;
    bool is_attn_residual =
        std::strncmp(t->name, "attn_residual-", 14) == 0;
    bool is_attn_post_norm =
        std::strncmp(t->name, "attn_post_norm-", 15) == 0;
    bool is_moe_topk = std::strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    bool is_moe_weights =
        std::strncmp(t->name, "ffn_moe_weights_norm-", 21) == 0;
    bool is_result_output = std::strcmp(t->name, "result_output") == 0;
    bool is_result_norm = std::strcmp(t->name, "result_norm") == 0;
    bool is_model_input = std::strcmp(t->name, "inp_scaled") == 0 ||
                          std::strcmp(t->name, "embd") == 0;
    bool is_ple_embd = std::strcmp(t->name, "ple_embd") == 0;
    bool is_ple_gate = std::strncmp(t->name, "ple_gate-", 9) == 0;
    bool is_ple_gated = std::strncmp(t->name, "ple_gated_value-", 16) == 0 &&
                        std::strstr(t->name, "reshaped") == nullptr;
    bool is_ple_gated_norm =
        std::strncmp(t->name, "ple_gated_norm-", 15) == 0;
    bool is_ple_conv = std::strncmp(t->name, "ple_conv_out-", 13) == 0;
    bool is_ple_conv_raw =
        std::strncmp(t->name, "ple_conv_raw-", 13) == 0;
    bool is_hc_after_ple = std::strncmp(t->name, "hc_after_ple-", 13) == 0;
    bool is_hc_combine = std::strncmp(t->name, "hc_combine-", 11) == 0;
    bool is_hc_attn_residual =
        std::strncmp(t->name, "hc_attn_residual-", 17) == 0;
    bool is_hc_layer_output = std::strncmp(t->name, "l_last-", 7) == 0;
    bool is_hc_norm = std::strncmp(t->name, "hc_norm-", 8) == 0;
    bool is_hc_gate = std::strncmp(t->name, "hc_gate-", 8) == 0;
    bool is_hc_ffn_low_raw =
        std::strncmp(t->name, "hc_ffn_low_raw-", 15) == 0;
    bool is_hc_ffn_low =
        std::strncmp(t->name, "hc_ffn_low-", 11) == 0;
    bool is_hc_ffn_gate =
        std::strncmp(t->name, "hc_ffn_gate-", 12) == 0;
    bool is_hc_mixed = std::strncmp(t->name, "hc_mixed-", 9) == 0 ||
                        std::strncmp(t->name, "hc_attn_mixed-", 14) == 0 ||
                        std::strncmp(t->name, "hc_ffn_mixed-", 13) == 0;
    bool is_hc_inject = std::strncmp(t->name, "hc_inject-", 10) == 0;
    if (st->token_ids_only && !is_result_output)
        return ask ? false : true;
    if (!is_lout && !is_attn_norm && !is_layer_norm &&
        !is_qcur && !is_kcur && !is_vcur && !is_qcur_normed &&
        !is_kcur_normed && !is_vcur_normed && !is_qcur_pos && !is_kcur_pos &&
        !is_kq && !is_kq_soft_max && !is_kqv && !is_cache_k_operand &&
        !is_cache_v_operand &&
        !is_kqv_out && !is_kqv_wo &&
        !is_ffn_inp && !is_ffn_norm && !is_ffn_norm_1 && !is_ffn_norm_2 &&
        !is_ffn_out && !is_ffn_up && !is_ffn_gate && !is_ffn_swiglu &&
        !is_ffn_geglu && !is_moe_logits && !is_moe_probs &&
        !is_moe_gate && !is_moe_up &&
        !is_moe_swiglu && !is_moe_down && !is_moe_weighted &&
        !is_moe_geglu && !is_moe_out && !is_shared_expert &&
        !is_shared_gate && !is_shared_gate_sigmoid &&
        !is_shared_expert_gated &&
        !is_ffn_mlp && !is_ffn_moe &&
        !is_ffn_moe_combined && !is_attn_out && !is_moe_topk &&
        !is_moe_weights && !is_linear_attn_out && !is_linear_qkv &&
        !is_linear_z && !is_linear_conv && !is_linear_conv_raw &&
        !is_linear_final &&
        !is_ssm_alpha && !is_ssm_beta && !is_ssm_softplus &&
        !is_ssm_decay_log &&
        !is_ssm_beta_sigmoid && !is_ssm_q_norm && !is_ssm_k_norm &&
        !is_ssm_gdn &&
        !is_attn_residual &&
        !is_attn_post_norm && !is_result_norm && !is_result_output &&
        !is_model_input && !is_ple_embd && !is_ple_gate &&
        !is_ple_gated && !is_ple_gated_norm && !is_ple_conv &&
        !is_ple_conv_raw &&
        !is_hc_after_ple && !is_hc_combine &&
        !is_hc_attn_residual && !is_hc_layer_output &&
        !is_hc_norm && !is_hc_gate && !is_hc_ffn_low_raw &&
        !is_hc_ffn_low && !is_hc_ffn_gate && !is_hc_mixed && !is_hc_inject)
        return ask ? false : true;
    if (ask) {
        // Do not retain a matching tensor in an earlier evaluation. Retaining
        // decode targets while processing the prompt can disable CUDA graph
        // fusions and change the KV state that the requested decode observes.
        if (!selected_pos)
            return false;
        const char *selective_name =
            std::getenv("LLAMA_PROBE_SELECTIVE_NAME");
        const char *selective_name2 =
            std::getenv("LLAMA_PROBE_SELECTIVE_NAME2");
        const char *selective_op =
            std::getenv("LLAMA_PROBE_SELECTIVE_OP");
        if (selective_name || selective_name2 || selective_op)
            return ((!selective_name && !selective_name2) ||
                    (selective_name &&
                     std::strcmp(t->name, selective_name) == 0) ||
                    (selective_name2 &&
                     std::strcmp(t->name, selective_name2) == 0)) &&
                   (!selective_op ||
                    std::strcmp(ggml_op_name(t->op), selective_op) == 0);
        return true;
    }

    int out_layer = -1;
    const char *fmt = (is_result_norm || is_result_output || is_model_input ||
                       is_ple_embd || is_ssm_gdn)
                    ? nullptr
                    : is_lout ? "l_out-%d"
                    : is_layer_norm ? "norm-%d"
                    : is_attn_norm ? "attn_norm-%d"
                    : is_qcur ? "Qcur-%d"
                    : is_kcur ? "Kcur-%d"
                    : is_vcur ? "Vcur-%d"
                    : is_qcur_normed ? "Qcur_normed-%d"
                    : is_kcur_normed ? "Kcur_normed-%d"
                    : is_vcur_normed ? "Vcur_normed-%d"
                    : is_qcur_pos ? "Qcur_pos-%d"
                    : is_kcur_pos ? "Kcur_pos-%d"
                    : is_kq_soft_max ? "kq_soft_max-%d"
                    : is_kq ? "kq-%d"
                    : is_kqv ? "kqv-%d"
                    : is_cache_k_operand ? "cache_k_l%d"
                    : is_cache_v_operand ? "cache_v_l%d"
                    : is_attn_pregate ? "attn_pregate-%d"
                    : is_attn_gated ? "attn_gated-%d"
                    : is_gate_sigmoid ? "gate_sigmoid-%d"
                    : is_kqv_out ? "kqv_out-%d"
                    : is_kqv_wo ? "kqv_wo-%d"
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
                    : is_moe_probs ? "ffn_moe_probs-%d"
                    : is_moe_gate ? "ffn_moe_gate-%d"
                    : is_moe_up ? "ffn_moe_up-%d"
                    : is_moe_swiglu ? "ffn_moe_swiglu-%d"
                    : is_moe_down ? "ffn_moe_down-%d"
                    : is_moe_weighted ? "ffn_moe_weighted-%d"
                    : is_moe_geglu ? "ffn_moe_geglu-%d"
                    : is_moe_out ? "ffn_moe_out-%d"
                    : is_shared_expert ? "ffn_shexp-%d"
                    : is_shared_gate_sigmoid
                        ? "shared_expert_gate_sigmoid-%d"
                    : is_shared_gate ? "shared_expert_gate-%d"
                    : is_shared_expert_gated ? "ffn_shexp_gated-%d"
                    : is_ffn_mlp ? "ffn_mlp-%d"
                    : is_ffn_moe ? "ffn_moe-%d"
                    : is_ffn_moe_combined ? "ffn_moe_combined-%d"
                    : is_moe_topk ? "ffn_moe_topk-%d"
                    : is_moe_weights ? "ffn_moe_weights_norm-%d"
                    : is_linear_qkv ? "linear_attn_qkv_mixed-%d"
                    : is_linear_z ? "z-%d"
                    : is_linear_conv ? "conv_output_silu-%d"
                    : is_linear_conv_raw ? "conv_output_raw-%d"
                    : is_linear_final ? "final_output-%d"
                    : is_ssm_alpha ? "alpha-%d"
                    : is_ssm_beta ? "beta-%d"
                    : is_ssm_softplus ? "a_softplus-%d"
                    : is_ssm_decay_log ? "gate-%d"
                    : is_ssm_beta_sigmoid ? "beta_sigmoid-%d"
                    : is_ssm_q_norm ? "q_conv_predelta-%d"
                    : is_ssm_k_norm ? "k_conv_predelta-%d"
                    : is_linear_attn_out ? "linear_attn_out-%d"
                    : is_attn_residual ? "attn_residual-%d"
                    : is_attn_post_norm ? "attn_post_norm-%d"
                    : is_ple_gate ? "ple_gate-%d"
                    : is_ple_gated ? "ple_gated_value-%d"
                    : is_ple_gated_norm ? "ple_gated_norm-%d"
                    : is_ple_conv ? "ple_conv_out-%d"
                    : is_ple_conv_raw ? "ple_conv_raw-%d"
                    : is_hc_after_ple ? "hc_after_ple-%d"
                    : is_hc_combine ? "hc_combine-%d"
                    : is_hc_attn_residual ? "hc_attn_residual-%d"
                    : is_hc_layer_output ? "l_last-%d"
                    : is_hc_norm ? "hc_norm-%d"
                    : is_hc_gate ? "hc_gate-%d"
                    : is_hc_ffn_low_raw ? "hc_ffn_low_raw-%d"
                    : is_hc_ffn_low ? "hc_ffn_low-%d"
                    : is_hc_ffn_gate ? "hc_ffn_gate-%d"
                    : std::strncmp(t->name, "hc_attn_mixed-", 14) == 0
                        ? "hc_attn_mixed-%d"
                    : std::strncmp(t->name, "hc_ffn_mixed-", 13) == 0
                        ? "hc_ffn_mixed-%d"
                    : is_hc_mixed ? "hc_mixed-%d"
                    : is_hc_inject ? "hc_inject-%d"
                    : is_attn_output ? "attn_output-%d"
                    : "attn_out-%d";
    if (fmt && std::sscanf(t->name, fmt, &out_layer) != 1)
        return true;

    if (!selected_pos)
        return true;
    if (std::getenv("LLAMA_PROBE_PRINT_ROPE_PARAMS") &&
        t->op == GGML_OP_ROPE) {
        float freq_base = 0.0f;
        std::memcpy(&freq_base,
                    (const int32_t *)t->op_params + 5,
                    sizeof(freq_base));
        int n_dims = ((const int32_t *)t->op_params)[1];
        float theta_scale = std::pow(freq_base, -2.0f / (float)n_dims);
        uint32_t base_bits = 0, scale_bits = 0;
        std::memcpy(&base_bits, &freq_base, sizeof(base_bits));
        std::memcpy(&scale_bits, &theta_scale, sizeof(scale_bits));
        std::printf("llama_rope_params name=%s n_dims=%d "
                    "freq_base=%.9g base_bits=%u theta_scale=%.9g "
                    "scale_bits=%u\n",
                    t->name, n_dims, freq_base, base_bits,
                    theta_scale, scale_bits);
    }
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
    if (is_lout && std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_lout_raw", t->name, out_layer,
                          data, nbytes);
    }
    if (is_kq_soft_max && std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_attn_softmax_raw", t->name, out_layer,
                          data, nbytes);
    }
    if (is_kq && std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_attn_scores_raw", t->name, out_layer,
                          data, nbytes);
    }
    if (is_kqv && std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_attn_kqv_raw", t->name, out_layer,
                          data, nbytes);
    }
    if ((is_qcur_pos || (is_qcur && t->op == GGML_OP_ROPE)) &&
        std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_attn_q_pos_raw", t->name, out_layer,
                          data, nbytes);
    }
    if (is_qcur_normed && std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_attn_q_normed_raw", t->name, out_layer,
                          data, nbytes);
    }
    if ((is_kcur_pos || (is_kcur && t->op == GGML_OP_ROPE)) &&
        std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_attn_k_pos_raw", t->name, out_layer,
                          data, nbytes);
    }
    if (is_vcur && std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_attn_v_raw", t->name, out_layer,
                          data, nbytes);
    }
    if (is_moe_swiglu && std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_moe_mid_raw", t->name, out_layer,
                          data, nbytes);
    }
    if (is_moe_down && std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_moe_down_raw", t->name, out_layer,
                          data, nbytes);
    }
    if (is_hc_combine && std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_hc_combine_raw", t->name, out_layer,
                          data, nbytes);
    }
    if (is_ssm_gdn && std::getenv("LLAMA_PROBE_DUMP_ALL_ROWS")) {
        dump_selected_raw("llama_ssm_gdn_raw", t->name, out_layer,
                          data, nbytes);
    }
    if (is_cache_k_operand || is_cache_v_operand) {
        if (std::getenv("LLAMA_PROBE_PRINT_CACHE_LAYOUT")) {
            std::printf("llama_cache_layout name=%s op=%s type=%s layer=%d "
                        "ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu "
                        "index_name=%s index_type=%s index_ne=%lld,%lld\n",
                        t->name, ggml_op_name(t->op),
                        ggml_type_name(t->type), out_layer,
                        (long long)t->ne[0], (long long)t->ne[1],
                        (long long)t->ne[2], (long long)t->ne[3],
                        t->nb[0], t->nb[1], t->nb[2], t->nb[3],
                        t->src[1] ? t->src[1]->name : "",
                        t->src[1] ? ggml_type_name(t->src[1]->type) : "none",
                        t->src[1] ? (long long)t->src[1]->ne[0] : 0LL,
                        t->src[1] ? (long long)t->src[1]->ne[1] : 0LL);
        }
        bool value_permuted = is_cache_v_operand &&
            std::strstr(t->name, "(permuted)") != nullptr;
        int64_t cache_positions = value_permuted ? t->ne[0] : t->ne[1];
        int64_t cache_dim = value_permuted ? t->ne[1] : t->ne[0];
        size_t count = (size_t)cache_positions * (size_t)cache_dim *
                       (size_t)t->ne[2];
        std::vector<float> logical(count);
        for (int64_t pos = 0; pos < cache_positions; pos++) {
            for (int64_t head = 0; head < t->ne[2]; head++) {
                for (int64_t d = 0; d < cache_dim; d++) {
                    size_t offset = value_permuted
                        ? (size_t)pos * t->nb[0] +
                          (size_t)head * t->nb[2] +
                          (size_t)d * t->nb[1]
                        : (size_t)pos * t->nb[1] +
                                    (size_t)head * t->nb[2] +
                                    (size_t)d * t->nb[0];
                    size_t logical_index =
                        ((size_t)pos * (size_t)t->ne[2] + (size_t)head) *
                        (size_t)cache_dim + (size_t)d;
                    if (t->type == GGML_TYPE_F16) {
                        ggml_fp16_t value;
                        std::memcpy(&value, data + offset, sizeof(value));
                        logical[logical_index] = ggml_fp16_to_fp32(value);
                    } else {
                        std::memcpy(&logical[logical_index], data + offset,
                                    sizeof(float));
                    }
                }
            }
        }
        dump_selected_raw(is_cache_k_operand ? "llama_attn_keys_raw"
                                             : "llama_attn_values_raw",
                          t->name, out_layer,
                          logical.data(), logical.size() * sizeof(float));
        return true;
    }
    if (is_attn_norm && t->type == GGML_TYPE_F32 && t->ne[1] > 1) {
        dump_selected_raw("llama_attn_norm_raw", t->name, out_layer,
                          data, ggml_nbytes(t));
    }
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
        // Both observed callbacks run while their tensors are [K, T]. The
        // normalized weights are reshaped to [1, K, T] only after the
        // ffn_moe_weights_norm callback. Inferring axes from extents is
        // ambiguous whenever the prompt length equals K.
        int expert_axis = 0;
        int token_axis = 1;
        int count = (int)std::min<int64_t>(t->ne[expert_axis], 16);
        std::vector<float> selected_weights;
        if (is_moe_weights)
            selected_weights.resize((size_t)count);
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
                selected_weights[(size_t)i] = value;
                std::printf("%.9g", value);
            }
        }
        std::putchar('\n');
        if (is_moe_weights)
            dump_selected_row("llama_moe_weights", t->name, out_layer,
                              st->current_pos, selected_weights.data(), count);
        return true;
    }
    int row_dim = (int)t->ne[0];
    int row_pos = 0;
    const float *row_base = base;
    if (is_hc_combine && t->ne[2] == 1 && t->ne[1] > 1) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
    } else if (is_kqv && t->ne[2] == 1 && t->ne[1] > 1) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
    } else if (is_layer_norm && t->ne[2] >= st->n_tokens && st->n_tokens > 1) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
        row_base = (const float *)(data +
            (size_t)st->row_pos * t->nb[2]);
    } else if ((is_lout || is_kqv || is_moe_gate || is_moe_up || is_moe_swiglu ||
         is_moe_down || is_moe_weighted ||
         is_ple_gate || is_ple_gated || is_ple_conv ||
         is_hc_combine) &&
        t->ne[2] >= st->n_tokens && st->n_tokens > 1) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
        row_base = (const float *)(data +
            (size_t)st->row_pos * t->nb[2]);
    } else if (is_hc_after_ple || is_hc_attn_residual ||
               is_hc_layer_output) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
        int token_row = t->ne[2] > st->row_pos ? st->row_pos : 0;
        row_base = (const float *)(data + (size_t)token_row * t->nb[2]);
    } else if ((is_ple_gated || is_ple_conv || is_ple_conv_raw) &&
               t->ne[1] > 1) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
    } else if ((is_ssm_beta || is_ssm_beta_sigmoid) &&
               t->ne[2] == 1 && t->ne[1] > 1) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
    } else if ((is_ssm_beta || is_ssm_beta_sigmoid) &&
        t->ne[2] >= st->n_tokens && st->n_tokens > 1) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
        row_base = (const float *)(data +
            (size_t)st->row_pos * t->nb[2]);
    } else if ((is_ssm_q_norm || is_ssm_k_norm) &&
               t->ne[2] == 1 && t->ne[1] > 1) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
    } else if ((is_ssm_q_norm || is_ssm_k_norm) &&
               t->ne[2] >= st->n_tokens && st->n_tokens > 1) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
        row_base = (const float *)(data +
            (size_t)st->row_pos * t->nb[2]);
    } else if ((is_qcur || is_kcur || is_vcur || is_qcur_normed ||
                is_kcur_normed || is_vcur_normed || is_qcur_pos ||
                is_kcur_pos) &&
               t->ne[2] >= st->n_tokens) {
        row_dim = (int)(t->ne[0] * t->ne[1]);
        row_base = (const float *)(data +
            (size_t)st->row_pos * t->nb[2]);
    } else if (t->ne[1] >= st->n_tokens && st->n_tokens > 1) {
        row_pos = st->row_pos;
        row_base = (const float *)(data + (size_t)row_pos * t->nb[1]);
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
                    : is_qcur_normed ? "llama_attn_q_normed"
                    : is_kcur_normed ? "llama_attn_k_normed"
                    : is_vcur_normed ? "llama_attn_v_normed"
                    : is_qcur_pos ? "llama_attn_q_pos"
                    : is_kcur_pos ? "llama_attn_k_pos"
                    : is_kq_soft_max ? "llama_attn_softmax"
                    : is_kq ? "llama_attn_scores"
                    : is_kqv ? "llama_attn_kqv"
                    : is_attn_pregate ? "llama_attn_pregate"
                    : is_attn_gated ? "llama_attn_gated"
                    : is_gate_sigmoid ? "llama_gate_sigmoid"
                    : is_kqv_out ? "llama_attn_out"
                    : is_kqv_wo ? "llama_attn_wo"
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
                    : is_moe_probs ? "llama_moe_probs"
                    : is_moe_gate ? "llama_moe_gate"
                    : is_moe_up ? "llama_moe_up"
                    : is_moe_swiglu ? "llama_moe_swiglu"
                    : is_moe_down ? "llama_moe_down"
                    : is_moe_weighted ? "llama_moe_weighted"
                    : is_moe_geglu ? "llama_moe_geglu"
                    : is_moe_out ? "llama_moe_out"
                    : is_moe_topk ? "llama_moe_topk"
                    : is_moe_weights ? "llama_moe_weights"
                    : is_shared_expert ? "llama_shared_expert"
                    : is_shared_gate_sigmoid ? "llama_shared_gate_sigmoid"
                    : is_shared_gate ? "llama_shared_gate_logit"
                    : is_shared_expert_gated ? "llama_shared_expert_gated"
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
                    : is_ssm_decay_log ? "llama_ssm_decay_log"
                    : is_ssm_beta_sigmoid ? "llama_ssm_beta_sigmoid"
                    : is_ssm_q_norm ? "llama_ssm_q_norm"
                    : is_ssm_k_norm ? "llama_ssm_k_norm"
                    : is_ssm_gdn ? "llama_ssm_gdn"
                    : is_linear_attn_out ? "llama_linear_attn_out"
                    : is_attn_residual ? "llama_attn_residual"
                    : is_attn_post_norm ? "llama_attn_post_norm"
                    : is_ple_embd ? "llama_ple_embd"
                    : is_ple_gate ? "llama_ple_gate"
                    : is_ple_gated ? "llama_ple_gated_value"
                    : is_ple_gated_norm ? "llama_ple_gated_norm"
                    : is_ple_conv ? "llama_ple_conv_out"
                    : is_ple_conv_raw ? "llama_ple_conv_raw"
                    : is_hc_after_ple ? "llama_hc_after_ple"
                    : is_hc_combine ? "llama_hc_combine"
                    : is_hc_attn_residual ? "llama_hc_attn_residual"
                    : is_hc_layer_output ? "llama_hc_layer_output"
                    : is_hc_norm ? "llama_hc_norm"
                    : is_hc_gate ? "llama_hc_gate"
                    : is_hc_ffn_low_raw ? "llama_hc_ffn_low_raw"
                    : is_hc_ffn_low ? "llama_hc_ffn_low"
                    : is_hc_ffn_gate ? "llama_hc_ffn_gate"
                    : is_hc_mixed ? "llama_hc_mixed"
                    : is_hc_inject ? "llama_hc_inject"
                    : "llama_attn_out";
    print_stats(tag, t->name, t, row_base,
                row_dim, st->current_pos, out_layer);
    if ((is_moe_gate || is_moe_up || is_moe_swiglu || is_moe_down ||
         is_moe_weighted) && std::getenv("LLAMA_PROBE_ALL_EXPERTS")) {
        for (int expert = 0; expert < (int)t->ne[1]; expert++) {
            char expert_tag[128];
            std::snprintf(expert_tag, sizeof(expert_tag), "%s_%d", tag, expert);
            const float *expert_row = (const float *)(
                data + (size_t)st->row_pos * t->nb[2] +
                (size_t)expert * t->nb[1]);
            print_stats(expert_tag, t->name, t, expert_row, (int)t->ne[0],
                        st->current_pos, out_layer);
            dump_selected_row(expert_tag, t->name, out_layer,
                              st->current_pos, expert_row, (int)t->ne[0]);
        }
    }
    if (is_hc_combine && t->ne[1] > 1) {
        char stream_tag[128];
        for (int stream = 0; stream < (int)t->ne[1]; stream++) {
            std::snprintf(stream_tag, sizeof(stream_tag),
                          "llama_hc_combine_s%d", stream);
            print_stats(stream_tag, t->name, t,
                        row_base + (size_t)stream * t->ne[0],
                        (int)t->ne[0], st->current_pos, out_layer);
        }
    }
    int dump_dim = row_dim;
    const float *dump_base = row_base;
    if (st->all_heads && t->ne[0] > 0 && t->ne[1] == st->n_tokens &&
        t->ne[2] > 1 && st->n_tokens == 1) {
        dump_base = base;
        dump_dim = row_dim * (int)t->ne[2];
    }
    dump_selected_row(tag, t->name, out_layer, st->current_pos,
                      dump_base, dump_dim);
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
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0)
        return probe_self_test();
    std::string model_path;
    std::string prompt;
    std::vector<llama_token> tokens;
    std::vector<llama_token> decode_tokens;
    int wanted_pos = -1;
    int context_size = 0;
    bool sequential = false;
    int flash_mode = 0;
    bool all_heads = false;
    int top_logits = 0;
    int generate = 0;
    int gpu_layers = 0;
    bool kv_f16 = false;
    bool observer = true;
    bool use_mmap = true;
    bool use_extra_bufts = true;

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
            flash_mode = 1;
        } else if (std::strcmp(argv[i], "--no-flash") == 0) {
            flash_mode = -1;
        } else if (std::strcmp(argv[i], "--all-heads") == 0) {
            all_heads = true;
        } else if (std::strcmp(argv[i], "--top-logits") == 0 && i + 1 < argc) {
            top_logits = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--decode-token-ids") == 0 && i + 1 < argc) {
            if (!parse_token_ids(argv[++i], &decode_tokens)) {
                std::fprintf(stderr, "invalid decode token IDs\n");
                return 1;
            }
        } else if (std::strcmp(argv[i], "--gpu-layers") == 0 &&
                   i + 1 < argc) {
            gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--kv-f16") == 0) {
            kv_f16 = true;
        } else if (std::strcmp(argv[i], "--no-observer") == 0) {
            observer = false;
        } else if (std::strcmp(argv[i], "--no-mmap") == 0) {
            use_mmap = false;
        } else if (std::strcmp(argv[i], "--no-extra-bufts") == 0) {
            use_extra_bufts = false;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || (prompt.empty() && tokens.empty()) ||
        (!prompt.empty() && !tokens.empty()) ||
        (!decode_tokens.empty() && generate != 0)) {
        usage(argv[0]);
        return 1;
    }
    if (!decode_tokens.empty()) generate = (int)decode_tokens.size() + 1;

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
    if (!use_mmap)
        mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    mparams.use_extra_bufts = use_extra_bufts;

    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    for (llama_token token : decode_tokens) {
        if (token >= llama_vocab_n_tokens(vocab)) {
            std::fprintf(stderr, "decode token ID outside model vocabulary\n");
            llama_model_free(model);
            return 1;
        }
    }
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
    int max_observed_pos = n_prompt + generate;
    if (pos < 0 || pos >= max_observed_pos) {
        std::fprintf(stderr, "position %d outside observed token range [0,%d)\n",
                     pos, max_observed_pos);
        llama_model_free(model);
        return 1;
    }

    int dim = llama_model_n_embd(model);

    llama_context_params cparams = llama_context_default_params();
    int native_context = (int)llama_model_n_ctx_train(model);
    int requested_context = context_size > 0 ? context_size : native_context;
    cparams.n_ctx = std::max(requested_context,
                             n_prompt + std::max(generate, 1));
    if (sequential) {
        /* Mirror compare_llama.sh's decode contract (-b 1 -ub 1).  Merely
         * submitting one-token batches with the large defaults can select a
         * different CUDA graph and produce different close logits. */
        cparams.n_batch = 1;
        cparams.n_ubatch = 1;
    } else {
        cparams.n_batch = std::max(cparams.n_batch, (uint32_t)n_prompt);
        cparams.n_ubatch = std::max(cparams.n_ubatch, (uint32_t)n_prompt);
    }
    cparams.type_k = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    cparams.type_v = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    if (flash_mode > 0)
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    else if (flash_mode < 0)
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.no_perf = true;
    int vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<float> captured_logits((size_t)vocab_size);
    bool generated_tokens_only = generate > 0 &&
        std::getenv("LLAMA_PROBE_DUMP_PATH") == nullptr &&
        std::getenv("LLAMA_PROBE_LIST_NAMES") == nullptr;
    int initial_pos = sequential ? 0
        : (pos < n_prompt ? pos : n_prompt - 1);
    int initial_row = sequential ? 0
        : (pos < n_prompt ? pos : n_prompt - 1);
    ProbeState cb_state = { dim, sequential ? 1 : n_prompt, pos, initial_pos,
                            initial_row, all_heads,
                            generated_tokens_only, captured_logits.data(),
                            vocab_size };
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

    const char *layer_input_path =
        std::getenv("LLAMA_PROBE_LAYER_INPUT_DUMP_PATH");
    const char *layer_input_text =
        std::getenv("LLAMA_PROBE_LAYER_INPUT_DUMP_LAYER");
    int layer_input = layer_input_text ? std::atoi(layer_input_text) : -1;
    if (layer_input_path && layer_input >= 0)
        llama_set_embeddings_layer_inp(ctx, (uint32_t)layer_input, true);

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

    bool defer_kv_dump =
        std::getenv("LLAMA_PROBE_KV_DUMP_AFTER_GENERATE") != nullptr;
    if (!defer_kv_dump && dump_completed_kv_cache(ctx) != 0) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    bool defer_layer_input_dump =
        std::getenv("LLAMA_PROBE_LAYER_INPUT_DUMP_AFTER_GENERATE") != nullptr;
    if (layer_input_path && layer_input >= 0 && !defer_layer_input_dump) {
        float *values = llama_get_embeddings_layer_inp(
            ctx, (uint32_t)layer_input);
        /* Sequential decode retains only the latest captured row. Batched
         * decode retains one row per prompt token. */
        size_t count = (sequential ? 1u : (size_t)n_prompt) * (size_t)dim;
        FILE *file = std::fopen(layer_input_path, "wb");
        if (!values || !file ||
            std::fwrite(values, sizeof(float), count, file) != count) {
            if (file) std::fclose(file);
            std::fprintf(stderr, "failed to dump layer input %d\n", layer_input);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        std::fclose(file);
        std::printf("llama_layer_input_dump layer=%d tokens=%d dim=%d\n",
                    layer_input, n_prompt, dim);
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
        int next = greedy_token(captured_logits.data(), vocab_size);
        if (next < 0) {
            std::fprintf(stderr, "no eligible token in greedy sample\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        std::printf("llama_token_id=%d\n", next);
        if ((size_t)generated < decode_tokens.size()) {
            next = decode_tokens[(size_t)generated];
            std::printf("llama_forced_token_id=%d\n", next);
        } else if (llama_vocab_is_eog(vocab, next))
            break;
        if (generated + 1 == generate)
            break;
        cb_state.current_pos = n_prompt + generated;
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
        if (top_logits > 0) {
            std::vector<int> ids;
            ids.reserve((size_t)std::min(top_logits, vocab_size));
            for (int id = 0; id < vocab_size; id++) {
                int j = (int)ids.size();
                if (j == top_logits &&
                    captured_logits[(size_t)id] <=
                        captured_logits[(size_t)ids[(size_t)j - 1]])
                    continue;
                if (j < top_logits)
                    ids.push_back(id);
                else
                    j--;
                while (j > 0 &&
                       captured_logits[(size_t)id] >
                           captured_logits[(size_t)ids[(size_t)j - 1]]) {
                    ids[(size_t)j] = ids[(size_t)j - 1];
                    j--;
                }
                ids[(size_t)j] = id;
            }
            for (int rank = 0; rank < (int)ids.size(); rank++) {
                int id = ids[(size_t)rank];
                std::printf("llama_top_logit step=%d rank=%d token=%d "
                            "attr=%u text=%s logit=%.9g\n",
                            generated + 1, rank + 1, id,
                            (unsigned)llama_vocab_get_attr(vocab, id),
                            llama_vocab_get_text(vocab, id),
                            captured_logits[(size_t)id]);
            }
        }
    }

    if (defer_kv_dump && dump_completed_kv_cache(ctx) != 0) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (layer_input_path && layer_input >= 0 && defer_layer_input_dump) {
        float *values = llama_get_embeddings_layer_inp(
            ctx, (uint32_t)layer_input);
        size_t count = (size_t)dim;
        FILE *file = std::fopen(layer_input_path, "wb");
        if (!values || !file ||
            std::fwrite(values, sizeof(float), count, file) != count) {
            if (file) std::fclose(file);
            std::fprintf(stderr, "failed to dump generated layer input %d\n",
                         layer_input);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        std::fclose(file);
        std::printf("llama_layer_input_dump layer=%d tokens=1 dim=%d generated=1\n",
                    layer_input, dim);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
