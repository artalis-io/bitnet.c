#include "generate.h"
#include "model.h"
#include "session.h"
#include "transformer.h"
#include "transformer_internal.h"
#include "transformer/gpu_internal.h"
#include "gpu_backend.h"
#include "model_arch.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

// Resolve allocator: if NULL, use stdlib default.
// The static is initialized on first call. Concurrent init is benign:
// bn_allocator_default() is pure (returns same struct every time).
static BnAllocator *resolve_alloc(BnAllocator *a) {
    static BnAllocator def;
    static int init;
    if (a) return a;
    if (!init) { def = bn_allocator_default(); init = 1; }
    return &def;
}

static int cuda_prefill_needs_decode_fallback(const BnModel *m,
                                              const BnGPUBackend *gpu) {
    const BnConfig *c = m ? &m->config : NULL;
    if (!c || !gpu || gpu->kind != BN_GPU_BACKEND_CUDA)
        return 0;
    if (bn_transformer_gpu_cuda_small_dense_prefill_decode_fallback_requested(
            gpu, c))
        return 1;
    if (c->n_experts <= 0 &&
        c->full_attn_interval > 0 &&
        c->ssm_inner_size > 0 &&
        c->dim >= 4096 &&
        getenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL") == NULL)
        return 1;
    return 0;
}

static int use_gpu_batch_prefill(const BnModel *model) {
    if (!model) return 0;
    if (getenv("BN_GPU_DISABLE_PREFILL_MATMUL")) return 0;
    if (getenv("BN_GPU_PREFILL_MATMUL")) return 1;
    const BnConfig *c = &model->config;
    if (c->kv_tq_bits != 0)
        return 0;
    BnGPUBackend *gpu = bn_model_gpu((BnModel *)model);
    if (cuda_prefill_needs_decode_fallback(model, gpu))
        return 0;
    if (c->full_attn_interval > 0) {
        if (gpu && gpu->kind == BN_GPU_BACKEND_CUDA &&
            gpu->prefill_ssm_layer &&
            !getenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN") &&
            !getenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER"))
            return 1;
        return 0;
    }
    if (gpu && gpu->kind == BN_GPU_BACKEND_CUDA && c->n_experts > 0)
        return getenv("BN_CUDA_ENABLE_MOE_PREFILL") != NULL;
    if (c->n_experts > 0)
        return 0;
    if (gpu && gpu->kind == BN_GPU_BACKEND_CUDA)
        return c->dim <= 8192;
    return c->dim <= 2560;
}

// --- Stop string matching ---
// We maintain a ring buffer of recent output text. After each token, we check
// if any stop string appears as a suffix of the accumulated text.

#define STOP_BUF_SIZE 256  // max stop string length we can detect

// Check if any stop string appears as a suffix of buf[0..buf_len-1].
// Returns the index of the matched stop string, or -1 if none.
static int check_stop_strings(const char *buf, int buf_len,
                               const BnStopStrings *stop) {
    for (int i = 0; i < stop->n; i++) {
        int slen = (int)strlen(stop->strings[i]);
        if (slen > 0 && slen <= buf_len) {
            if (memcmp(buf + buf_len - slen, stop->strings[i], slen) == 0)
                return i;
        }
    }
    return -1;
}

// --- Loop detection ---
#define LOOP_BUF_SIZE 32
#define LOOP_BUF_MASK (LOOP_BUF_SIZE - 1)
#define LOOP_NGRAM    4

#define BN_MAX_TOP_LOGITS 32

static void dump_top_logits(const float *logits, int vocab_size, int top_k,
                            const char *label, int step) {
    if (top_k <= 0) return;
    if (top_k > BN_MAX_TOP_LOGITS) top_k = BN_MAX_TOP_LOGITS;
    int top[BN_MAX_TOP_LOGITS];
    int n_top = 0;
    for (int v = 0; v < vocab_size; v++) {
        int k = n_top;
        if (k == top_k && logits[v] <= logits[top[k - 1]]) continue;
        if (k < top_k) n_top++;
        else k--;
        while (k > 0 && logits[v] > logits[top[k - 1]]) {
            top[k] = top[k - 1];
            k--;
        }
        top[k] = v;
    }
    fprintf(stderr, "top_logits %s step=%d k=%d\n", label, step, n_top);
    for (int k = 0; k < n_top; k++)
        fprintf(stderr, "top_logit step=%d rank=%d token=%d logit=%.9g\n",
                step, k + 1, top[k], logits[top[k]]);
}

static void dump_selected_logits(const float *logits, int vocab_size,
                                 const char *ids, int step) {
    if (!ids || !*ids) return;
    const char *p = ids;
    while (*p) {
        char *end = NULL;
        long id = strtol(p, &end, 10);
        if (end == p) {
            p++;
            continue;
        }
        if (id >= 0 && id < vocab_size) {
            fprintf(stderr, "selected_logit step=%d token=%ld logit=%.9g\n",
                    step, id, logits[id]);
        }
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\t')
            p++;
    }
}

int bn_generate(BnModel *model, BnSession *s, BnTokenizer *tok, BnSampler *sampler,
                int max_tokens, int *pos,
                bn_token_callback cb, void *user_data,
                const BnStopStrings *stop,
                BnAllocator *alloc) {
    if (!pos || !s) return -2;
    (void)alloc;

    int loop_buf[LOOP_BUF_SIZE];
    int loop_idx = 0, gen_count = 0;
    memset(loop_buf, -1, sizeof(loop_buf));

    // Stop string state: ring buffer of recent output text
    char stop_buf[STOP_BUF_SIZE];
    int stop_buf_len = 0;
    int max_stop_len = 0;
    if (stop && stop->n > 0) {
        for (int i = 0; i < stop->n; i++) {
            int slen = (int)strlen(stop->strings[i]);
            if (slen > max_stop_len) max_stop_len = slen;
        }
        if (max_stop_len > STOP_BUF_SIZE) max_stop_len = STOP_BUF_SIZE;
    }

    float *logits = s->state.logits;
    if (!logits) return -2;
    const char *top_env = getenv("BN_TOP_LOGITS");
    const char *logit_ids_env = getenv("BN_LOGIT_IDS");
    int top_logits = top_env ? atoi(top_env) : 0;
    int disable_loop_abort = getenv("BN_DISABLE_LOOP_ABORT") != NULL;
    int have_gpu_next = 0;
    int gpu_next = -1;

    for (int i = 0; i < max_tokens; i++) {
        if (!have_gpu_next && top_logits > 0)
            dump_top_logits(logits, model->config.vocab_size, top_logits,
                            "generate", i);
        if (!have_gpu_next && logit_ids_env)
            dump_selected_logits(logits, model->config.vocab_size,
                                 logit_ids_env, i);
        int next;
        if (have_gpu_next) {
            next = gpu_next;
            have_gpu_next = 0;
            gpu_next = -1;
        } else {
            next = bn_sampler_sample(sampler, logits);
        }

        if (next == tok->eot_id || next == tok->eos_id ||
            next == tok->im_end_id)
            break;

        // Ring buffer loop detection
        loop_buf[loop_idx] = next;
        loop_idx = (loop_idx + 1) & LOOP_BUF_MASK;
        gen_count++;

        if (!disable_loop_abort && gen_count >= 2 * LOOP_NGRAM) {
            int looping = 1;
            for (int k = 0; k < LOOP_NGRAM; k++) {
                int a = loop_buf[(loop_idx - 1 - k) & LOOP_BUF_MASK];
                int b = loop_buf[(loop_idx - 1 - k - LOOP_NGRAM) &
                                 LOOP_BUF_MASK];
                if (a != b) { looping = 0; break; }
            }
            if (looping) return -1;
        }

        if (sampler->repeat_penalty != 1.0f)
            bn_sampler_accept(sampler, next);

        const char *piece = (cb || max_stop_len > 0)
                                ? bn_tokenizer_decode(tok, next)
                                : NULL;
        if (piece && cb) {
            if (cb(piece, next, user_data))
                break;
        }

        // Stop string check: append piece to buffer and check for match
        if (max_stop_len > 0 && piece) {
            int plen = (int)strlen(piece);
            if (plen > 0) {
                // Shift buffer left if adding piece would overflow
                if (stop_buf_len + plen > STOP_BUF_SIZE) {
                    int keep = STOP_BUF_SIZE - plen;
                    if (keep < 0) keep = 0;
                    if (keep > 0 && keep < stop_buf_len)
                        memmove(stop_buf, stop_buf + stop_buf_len - keep, keep);
                    stop_buf_len = keep;
                }
                int copy = plen;
                if (copy > STOP_BUF_SIZE - stop_buf_len)
                    copy = STOP_BUF_SIZE - stop_buf_len;
                memcpy(stop_buf + stop_buf_len, piece + plen - copy, copy);
                stop_buf_len += copy;

                if (check_stop_strings(stop_buf, stop_buf_len, stop) >= 0)
                    return -3;
            }
        }

        if (i + 1 == max_tokens)
            break;

        BnGPUBackend *gpu = bn_model_gpu(model);
        int can_gpu_greedy =
            gpu && gpu->argmax_activation &&
            top_logits <= 0 && sampler->temperature == 0.0f &&
            sampler->repeat_penalty >= 1.0f;
        if (can_gpu_greedy &&
            bn_transformer_gpu_forward_argmax(
                model, s, next, *pos,
                sampler->recent_tokens, sampler->recent_len,
                sampler->repeat_penalty, &gpu_next) == 0) {
            if (getenv("BN_GPU_DEBUG_ARGMAX"))
                fprintf(stderr, "[bn:gpu:argmax] step=%d token=%d next=%d\n",
                        i, next, gpu_next);
            have_gpu_next = 1;
            (*pos)++;
        } else {
            logits = bn_transformer_forward(model, s, next, *pos);
            (*pos)++;
            if (!logits) return -2;
        }
    }

    return gen_count;
}

#define MAX_DRAFT_K 20

int bn_generate_speculative(
    BnModel *target, BnSession *ts,
    BnModel *draft, BnSession *ds, int draft_k,
    BnTokenizer *tok, BnSampler *sampler,
    int max_tokens, int *pos,
    bn_token_callback cb, void *user_data,
    BnAllocator *alloc)
{
    alloc = resolve_alloc(alloc);

    int gen_count = 0;
    int draft_tokens[MAX_DRAFT_K];
    if (draft_k > MAX_DRAFT_K) draft_k = MAX_DRAFT_K;
    int n_accepted_total = 0, n_drafted_total = 0;

    float *target_logits = ts->state.logits;
    float *draft_logits = ds->state.logits;
    if (!target_logits || !draft_logits) return -2;

    while (gen_count < max_tokens) {
        // --- Draft phase ---
        int k_actual = 0;
        for (int i = 0; i < draft_k && gen_count + i < max_tokens; i++) {
            int d = bn_sampler_sample(sampler, draft_logits);
            if (d == tok->eot_id || d == tok->eos_id || d == tok->im_end_id)
                break;
            draft_tokens[k_actual++] = d;
            draft_logits = bn_transformer_forward(draft, ds, d, *pos + k_actual - 1);
            if (!draft_logits) return -2;
        }

        if (k_actual == 0) {
            int t = bn_sampler_sample(sampler, target_logits);
            if (t == tok->eot_id || t == tok->eos_id || t == tok->im_end_id)
                break;
            bn_sampler_accept(sampler, t);
            const char *piece = bn_tokenizer_decode(tok, t);
            if (piece && cb && cb(piece, t, user_data)) break;
            target_logits = bn_transformer_forward(target, ts, t, *pos);
            draft_logits = bn_transformer_forward(draft, ds, t, *pos);
            (*pos)++;
            gen_count++;
            if (!target_logits || !draft_logits) return -2;
            continue;
        }

        n_drafted_total += k_actual;

        // --- Verify phase ---
        int vocab_size = target->config.vocab_size;
        size_t verify_size = (size_t)(k_actual + 1) * (size_t)vocab_size * sizeof(float);
        float *verify_logits = (float *)bn_malloc(alloc, verify_size);
        if (!verify_logits) return -2;

        memcpy(verify_logits, target_logits, (size_t)vocab_size * sizeof(float));

        if (bn_transformer_prefill_all(target, ts, draft_tokens, k_actual, *pos,
                                        verify_logits + vocab_size) != 0) {
            bn_free(alloc, verify_logits, verify_size);
            return -2;
        }

        int n_accepted = 0;
        int corrected = -1;
        for (int i = 0; i < k_actual; i++) {
            float *lg = verify_logits + (size_t)i * vocab_size;
            int t_i = bn_sampler_sample(sampler, lg);
            if (t_i == draft_tokens[i]) {
                n_accepted++;
            } else {
                corrected = t_i;
                break;
            }
        }
        target_logits = ts->state.logits;
        bn_free(alloc, verify_logits, verify_size);

        // Stream accepted draft tokens
        for (int i = 0; i < n_accepted; i++) {
            bn_sampler_accept(sampler, draft_tokens[i]);
            const char *piece = bn_tokenizer_decode(tok, draft_tokens[i]);
            if (piece && cb && cb(piece, draft_tokens[i], user_data)) goto done;
            gen_count++;
        }

        if (corrected >= 0) {
            if (corrected != tok->eot_id && corrected != tok->eos_id &&
                corrected != tok->im_end_id) {
                bn_sampler_accept(sampler, corrected);
                const char *piece = bn_tokenizer_decode(tok, corrected);
                if (piece && cb && cb(piece, corrected, user_data)) goto done;
                gen_count++;
            } else {
                *pos += n_accepted;
                break;
            }
            *pos += n_accepted + 1;

            target_logits = bn_transformer_forward(target, ts, corrected, *pos - 1);
            if (!target_logits) return -2;
            draft_logits = bn_transformer_forward(draft, ds, corrected, *pos - 1);
            if (!draft_logits) return -2;
        } else {
            *pos += n_accepted;
            int bonus = bn_sampler_sample(sampler, target_logits);
            if (bonus != tok->eot_id && bonus != tok->eos_id &&
                bonus != tok->im_end_id) {
                bn_sampler_accept(sampler, bonus);
                const char *piece = bn_tokenizer_decode(tok, bonus);
                if (piece && cb && cb(piece, bonus, user_data)) goto done;
                gen_count++;
                target_logits = bn_transformer_forward(target, ts, bonus, *pos);
                draft_logits = bn_transformer_forward(draft, ds, bonus, *pos);
                (*pos)++;
                if (!target_logits || !draft_logits) return -2;
            } else {
                break;
            }
        }

        n_accepted_total += n_accepted + 1;
    }

done:
    if (n_drafted_total > 0) {
        fprintf(stderr, "[spec] accepted %d/%d (%.1f%%)\n",
                n_accepted_total, n_drafted_total,
                100.0 * n_accepted_total / n_drafted_total);
    }
    return gen_count;
}

float *bn_prefill(BnModel *model, BnSession *s, const int *tokens, int n_tokens,
                  int pos0, int no_prefill) {
    float *logits = NULL;
    int gpu_attached = bn_model_gpu(model) != NULL;
    int parity_cpu =
        bn_model_arch_cpu_prefill_uses_decode_for_parity(&model->config) &&
                     !gpu_attached;
    /* GPU decode reads backend-resident KV buffers. For conservative small
     * dense models, batch prefill is followed by a CPU->GPU KV upload.
     */
    if (!no_prefill && !parity_cpu && n_tokens > 1 &&
        (!gpu_attached || use_gpu_batch_prefill(model))) {
        logits = bn_transformer_prefill(model, s, tokens, n_tokens, pos0);
        if (logits && gpu_attached &&
            !s->gpu_kv_direct_valid &&
            bn_transformer_gpu_upload_kv_cache(model, s, pos0,
                                               n_tokens) != 0)
            return NULL;
        if (logits && gpu_attached &&
            model->config.full_attn_interval > 0 &&
            getenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER") &&
            bn_transformer_gpu_upload_ssm_state(model, s) != 0)
            return NULL;
    } else {
        for (int i = 0; i < n_tokens; i++) {
            if (i + 1 == n_tokens) {
                logits = bn_transformer_forward(model, s, tokens[i], pos0 + i);
            } else if (bn_transformer_forward_no_logits(
                           model, s, tokens[i], pos0 + i) != 0) {
                return NULL;
            }
        }
    }
    return logits;
}

int bn_prefill_no_logits(BnModel *model, BnSession *s, const int *tokens,
                         int n_tokens, int pos0, int no_prefill) {
    int gpu_attached = bn_model_gpu(model) != NULL;
    int parity_cpu =
        bn_model_arch_cpu_prefill_uses_decode_for_parity(&model->config) &&
                     !gpu_attached;
    if (!no_prefill && !parity_cpu && n_tokens > 1 &&
        (!gpu_attached || use_gpu_batch_prefill(model))) {
        int rc = bn_transformer_prefill_no_logits(model, s, tokens,
                                                  n_tokens, pos0);
        if (rc == 0 && gpu_attached && !s->gpu_kv_direct_valid)
            rc = bn_transformer_gpu_upload_kv_cache(model, s, pos0,
                                                    n_tokens);
        if (rc == 0 && gpu_attached &&
            model->config.full_attn_interval > 0 &&
            getenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER"))
            rc = bn_transformer_gpu_upload_ssm_state(model, s);
        return rc;
    }
    for (int i = 0; i < n_tokens; i++) {
        if (bn_transformer_forward_no_logits(model, s, tokens[i],
                                             pos0 + i) != 0)
            return -1;
    }
    return 0;
}

int bn_count_tokens(const BnTokenizer *tok, const char *text,
                    BnAllocator *alloc) {
    alloc = resolve_alloc(alloc);
    size_t len = strlen(text);
    if (len > (size_t)INT_MAX - 3) return -1;
    int max = (int)len + 3;
    size_t buf_size = (size_t)max * sizeof(int);
    int *buf = (int *)bn_malloc(alloc, buf_size);
    if (!buf) return -1;
    int n = bn_tokenizer_encode(tok, text, 0, buf, max);
    bn_free(alloc, buf, buf_size);
    return n;
}

// --- Chat formatting ---

// Resolve BN_CHAT_AUTO to a concrete format based on tokenizer state.
static BnChatFormat resolve_format(const BnTokenizer *tok, BnChatFormat fmt) {
    if (fmt != BN_CHAT_AUTO) return fmt;
    return tok->chatml ? BN_CHAT_CHATML : BN_CHAT_LLAMA;
}

// Role names for each format
static const char *chatml_role_name(BnChatRole role) {
    switch (role) {
    case BN_ROLE_SYSTEM:    return "system";
    case BN_ROLE_USER:      return "user";
    case BN_ROLE_ASSISTANT: return "assistant";
    default:                return "user";
    }
}

static const char *llama_role_name(BnChatRole role) {
    switch (role) {
    case BN_ROLE_SYSTEM:    return "System";
    case BN_ROLE_USER:      return "User";
    case BN_ROLE_ASSISTANT: return "Assistant";
    default:                return "User";
    }
}

// Encode a single ChatML turn: <|im_start|>role\ncontent<|im_end|>\n
static int encode_chatml_turn(const BnTokenizer *tok, BnChatRole role,
                               const char *content, int *out, int max,
                               BnAllocator *alloc) {
    int n = 0;
    if (n < max) out[n++] = tok->im_start_id;

    const char *rname = chatml_role_name(role);
    size_t rlen = strlen(rname);
    size_t clen = strlen(content);
    size_t buf_len = rlen + 1 + clen + 1;  // "role\ncontent\0"
    char *buf = (char *)bn_malloc(alloc, buf_len);
    if (!buf) return n;
    snprintf(buf, buf_len, "%s\n%s", rname, content);
    n += bn_tokenizer_encode(tok, buf, 0, out + n, max - n);
    bn_free(alloc, buf, buf_len);

    if (n < max) out[n++] = tok->im_end_id;
    int n2 = bn_tokenizer_encode(tok, "\n", 0, out + n, max - n);
    n += n2;
    return n;
}

// Encode a single LLaMA turn: Role: content<|eot_id|>
static int encode_llama_turn(const BnTokenizer *tok, BnChatRole role,
                              const char *content, int *out, int max,
                              BnAllocator *alloc) {
    const char *rname = llama_role_name(role);
    size_t rlen = strlen(rname);
    size_t clen = strlen(content);
    size_t buf_len = rlen + 2 + clen + 1;  // "Role: content\0"
    char *buf = (char *)bn_malloc(alloc, buf_len);
    if (!buf) return 0;
    snprintf(buf, buf_len, "%s: %s", rname, content);
    int n = bn_tokenizer_encode(tok, buf, 0, out, max);
    bn_free(alloc, buf, buf_len);

    if (n < max && tok->eot_id >= 0)
        out[n++] = tok->eot_id;
    return n;
}

// Legacy single-turn formatters (delegate to multi-turn with one user message)
static int format_chatml(const BnTokenizer *tok, const char *user_msg,
                         int *out, int max, BnAllocator *alloc) {
    BnChatMessage msg = { BN_ROLE_USER, user_msg };
    return bn_chat_format_messages(tok, BN_CHAT_CHATML, &msg, 1, out, max, alloc);
}

static int format_llama(const BnTokenizer *tok, const char *user_msg,
                        int *out, int max, BnAllocator *alloc) {
    BnChatMessage msg = { BN_ROLE_USER, user_msg };
    return bn_chat_format_messages(tok, BN_CHAT_LLAMA, &msg, 1, out, max, alloc);
}

int bn_chat_format_turn(const BnTokenizer *tok, BnChatFormat fmt,
                        const char *user_msg,
                        int *out_tokens, int max_tokens,
                        BnAllocator *alloc) {
    alloc = resolve_alloc(alloc);
    fmt = resolve_format(tok, fmt);
    switch (fmt) {
    case BN_CHAT_CHATML:
        return format_chatml(tok, user_msg, out_tokens, max_tokens, alloc);
    case BN_CHAT_LLAMA:
        return format_llama(tok, user_msg, out_tokens, max_tokens, alloc);
    case BN_CHAT_RAW:
        return bn_tokenizer_encode(tok, user_msg, 0, out_tokens, max_tokens);
    default:
        return format_llama(tok, user_msg, out_tokens, max_tokens, alloc);
    }
}

int bn_chat_format_messages(const BnTokenizer *tok, BnChatFormat fmt,
                            const BnChatMessage *messages, int n_messages,
                            int *out_tokens, int max_tokens,
                            BnAllocator *alloc) {
    alloc = resolve_alloc(alloc);
    fmt = resolve_format(tok, fmt);
    int n = 0;

    if (fmt == BN_CHAT_RAW) {
        // RAW: concatenate all message contents
        for (int i = 0; i < n_messages; i++) {
            n += bn_tokenizer_encode(tok, messages[i].content, 0,
                                     out_tokens + n, max_tokens - n);
        }
        return n;
    }

    // Encode each message as a complete turn
    for (int i = 0; i < n_messages; i++) {
        if (fmt == BN_CHAT_CHATML) {
            n += encode_chatml_turn(tok, messages[i].role, messages[i].content,
                                    out_tokens + n, max_tokens - n, alloc);
        } else {
            n += encode_llama_turn(tok, messages[i].role, messages[i].content,
                                   out_tokens + n, max_tokens - n, alloc);
        }
    }

    // Append assistant prompt (open turn for the model to complete)
    if (fmt == BN_CHAT_CHATML) {
        if (n < max_tokens) out_tokens[n++] = tok->im_start_id;
        int n2 = bn_tokenizer_encode(tok, "assistant\n", 0,
                                     out_tokens + n, max_tokens - n);
        n += n2;
    } else {
        int n2 = bn_tokenizer_encode(tok, "Assistant: ", 0,
                                     out_tokens + n, max_tokens - n);
        n += n2;
    }

    return n;
}

int bn_chat_turn_end_id(const BnTokenizer *tok, BnChatFormat fmt) {
    fmt = resolve_format(tok, fmt);
    switch (fmt) {
    case BN_CHAT_CHATML: return tok->im_end_id;
    case BN_CHAT_LLAMA:  return tok->eot_id;
    case BN_CHAT_RAW:    return -1;
    default:             return -1;
    }
}

// --- SSE streaming format ---

static int json_escape(char *dst, int dst_size, const char *src) {
    int w = 0;
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        int need;
        if (*p == '"' || *p == '\\') need = 2;
        else if (*p == '\n' || *p == '\r' || *p == '\t') need = 2;
        else if (*p < 0x20) need = 6;  // \u00XX
        else need = 1;

        if (w + need >= dst_size) return -1;

        if (*p == '"')       { dst[w++] = '\\'; dst[w++] = '"'; }
        else if (*p == '\\') { dst[w++] = '\\'; dst[w++] = '\\'; }
        else if (*p == '\n') { dst[w++] = '\\'; dst[w++] = 'n'; }
        else if (*p == '\r') { dst[w++] = '\\'; dst[w++] = 'r'; }
        else if (*p == '\t') { dst[w++] = '\\'; dst[w++] = 't'; }
        else if (*p < 0x20)  { w += snprintf(dst + w, dst_size - w, "\\u%04x", *p); }
        else                 { dst[w++] = *p; }
    }
    if (w < dst_size) dst[w] = '\0';
    return w;
}

int bn_format_sse_chunk(char *buf, int buf_size,
                        const char *piece, const char *id,
                        const char *model, const char *finish_reason,
                        long long created) {
    if (!id) id = "chatcmpl-0";
    if (!model) model = "bitnet";

    char escaped[1024];
    if (piece) {
        if (json_escape(escaped, (int)sizeof(escaped), piece) < 0)
            return -1;
    }

    // Build the created field
    char created_field[64];
    if (created != 0)
        snprintf(created_field, sizeof(created_field), "\"created\":%lld,", created);
    else
        created_field[0] = '\0';

    int n;
    if (finish_reason) {
        // Finish chunk: empty delta, finish_reason set
        n = snprintf(buf, buf_size,
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",%s"
            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},"
            "\"finish_reason\":\"%s\"}]}\n\n",
            id, created_field, model, finish_reason);
    } else {
        // Normal chunk: delta with content
        n = snprintf(buf, buf_size,
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",%s"
            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":"
            "{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
            id, created_field, model, escaped);
    }

    if (n < 0 || n >= buf_size) return -1;
    return n;
}

int bn_format_sse_done(char *buf, int buf_size) {
    int n = snprintf(buf, buf_size, "data: [DONE]\n\n");
    if (n < 0 || n >= buf_size) return -1;
    return n;
}

// --- Logprobs ---

void bn_logprobs_compute(const float *logits, int vocab_size,
                         int chosen_token, int top_k,
                         const BnTokenizer *tok,
                         BnLogprobs *result) {
    if (top_k > BN_LOGPROBS_MAX_TOP_K) top_k = BN_LOGPROBS_MAX_TOP_K;
    if (top_k < 0) top_k = 0;
    result->top_k = top_k;

    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Compute log-sum-exp for log-softmax: log(sum(exp(logits - max)))
    // sum_exp >= 1.0 since exp(max_logit - max_logit) = 1, so logf is safe.
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }
    if (sum_exp < 1e-30f) sum_exp = 1e-30f;  // guard for cppcheck (unreachable: sum_exp >= 1.0)
    float log_sum = logf(sum_exp) + max_logit;  // log(sum(exp(logits)))

    // Chosen token logprob
    float chosen_logprob = logits[chosen_token] - log_sum;
    result->chosen.token_id = chosen_token;
    result->chosen.logprob = chosen_logprob;
    result->chosen.text = tok ? bn_tokenizer_decode(tok, chosen_token) : NULL;

    // Find top-K by logit value (equivalent to top-K by logprob since
    // log-softmax is monotonic with logits)
    if (top_k > 0) {
        // Partial selection: maintain a sorted top-K array via insertion
        for (int k = 0; k < top_k; k++) {
            result->top[k].token_id = -1;
            result->top[k].logprob = -INFINITY;
            result->top[k].text = NULL;
        }

        for (int i = 0; i < vocab_size; i++) {
            float lp = logits[i] - log_sum;
            // Check if this logprob is larger than the smallest in top-K
            if (top_k > 0 && lp > result->top[top_k - 1].logprob) {
                // Insert in sorted position (descending by logprob)
                int pos = top_k - 1;
                while (pos > 0 && lp > result->top[pos - 1].logprob) {
                    result->top[pos] = result->top[pos - 1];
                    pos--;
                }
                result->top[pos].token_id = i;
                result->top[pos].logprob = lp;
                result->top[pos].text = tok ? bn_tokenizer_decode(tok, i) : NULL;
            }
        }
    }
}
