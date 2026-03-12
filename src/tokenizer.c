#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// --- Internal: sorted vocab for binary search ---

typedef struct {
    char **vocab;
    int   *indices;
} SortContext;

static SortContext g_sort_ctx;

static int cmp_vocab_indirect(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return strcmp(g_sort_ctx.vocab[ia], g_sort_ctx.vocab[ib]);
}

// Binary search for a token string in sorted vocab
static int vocab_lookup(const Tokenizer *t, const char *str) {
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(str, t->vocab[t->sorted_indices[mid]]);
        if (cmp == 0) return t->sorted_indices[mid];
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

int tokenizer_init(Tokenizer *t, GGUFFile *f) {
    memset(t, 0, sizeof(Tokenizer));

    t->vocab_size = (int)gguf_get_arr_n(f, "tokenizer.ggml.tokens");
    if (t->vocab_size == 0) {
        fprintf(stderr, "tokenizer: no tokens found in GGUF\n");
        return -1;
    }

    // Allocate vocab
    t->vocab = (char **)malloc(t->vocab_size * sizeof(char *));
    t->max_token_length = 0;

    for (int i = 0; i < t->vocab_size; i++) {
        const char *tok = gguf_get_arr_str(f, "tokenizer.ggml.tokens", i);
        t->vocab[i] = tok ? strdup(tok) : strdup("");
        int len = (int)strlen(t->vocab[i]);
        if (len > t->max_token_length) t->max_token_length = len;
    }

    // Load scores (optional — some tokenizers don't have scores)
    t->scores = (float *)calloc(t->vocab_size, sizeof(float));
    const void *scores_data = gguf_get_arr_data(f, "tokenizer.ggml.scores");
    if (scores_data) {
        memcpy(t->scores, scores_data, t->vocab_size * sizeof(float));
    }

    // Special token IDs
    int idx;
    idx = gguf_find_key(f, "tokenizer.ggml.bos_token_id");
    t->bos_id = (idx >= 0) ? (int)gguf_get_u32(f, "tokenizer.ggml.bos_token_id") : 1;

    idx = gguf_find_key(f, "tokenizer.ggml.eos_token_id");
    t->eos_id = (idx >= 0) ? (int)gguf_get_u32(f, "tokenizer.ggml.eos_token_id") : 2;

    idx = gguf_find_key(f, "tokenizer.ggml.eot_token_id");
    t->eot_id = (idx >= 0) ? (int)gguf_get_u32(f, "tokenizer.ggml.eot_token_id") : -1;

    // Build sorted index for binary search
    t->sorted_indices = (int *)malloc(t->vocab_size * sizeof(int));
    for (int i = 0; i < t->vocab_size; i++) t->sorted_indices[i] = i;
    g_sort_ctx.vocab = t->vocab;
    g_sort_ctx.indices = t->sorted_indices;
    qsort(t->sorted_indices, t->vocab_size, sizeof(int), cmp_vocab_indirect);

    return 0;
}

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    free(t->scores);
    free(t->sorted_indices);
}

// Encode text using BPE merge algorithm
int tokenizer_encode(const Tokenizer *t, const char *text, int add_bos,
                     int *tokens, int max_tokens) {
    if (!text || !tokens || max_tokens <= 0) return 0;

    int n_tokens = 0;

    // Add BOS if requested
    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = t->bos_id;
    }

    // Initial tokenization: encode each byte/char as individual token
    // For UTF-8 text, first try to find each character as a token
    int text_len = (int)strlen(text);
    if (text_len == 0) return n_tokens;

    // Step 1: Initialize with individual character tokens
    // Try to find single-character tokens first
    int *work = (int *)malloc((text_len + 1) * sizeof(int));
    int n_work = 0;

    for (int i = 0; i < text_len; ) {
        // Try to find the longest matching token starting at position i
        int best_len = 0;
        int best_tok = -1;

        // Try single byte first — look up as token
        char single[2] = { text[i], '\0' };
        int tok = vocab_lookup(t, single);
        if (tok >= 0) {
            best_len = 1;
            best_tok = tok;
        }

        if (best_tok >= 0) {
            work[n_work++] = best_tok;
            i += best_len;
        } else {
            // Try byte fallback tokens like <0xNN>
            char byte_tok[8];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)text[i]);
            tok = vocab_lookup(t, byte_tok);
            if (tok >= 0) {
                work[n_work++] = tok;
            }
            // If not found, skip the character
            i++;
        }
    }

    // Step 2: BPE merge loop — greedily merge the pair with highest score
    char *merge_buf = (char *)malloc(t->max_token_length * 2 + 4);

    while (n_work >= 2) {
        float best_score = -FLT_MAX;
        int best_idx = -1;
        int best_tok = -1;

        // Find the best merge pair
        for (int i = 0; i < n_work - 1; i++) {
            snprintf(merge_buf, t->max_token_length * 2 + 4, "%s%s",
                     t->vocab[work[i]], t->vocab[work[i + 1]]);
            int tok = vocab_lookup(t, merge_buf);
            if (tok >= 0 && t->scores[tok] > best_score) {
                best_score = t->scores[tok];
                best_idx = i;
                best_tok = tok;
            }
        }

        if (best_idx < 0) break;  // No more merges possible

        // Apply the merge
        work[best_idx] = best_tok;
        // Shift remaining tokens
        for (int i = best_idx + 1; i < n_work - 1; i++) {
            work[i] = work[i + 1];
        }
        n_work--;
    }

    free(merge_buf);

    // Copy results
    for (int i = 0; i < n_work && n_tokens < max_tokens; i++) {
        tokens[n_tokens++] = work[i];
    }

    free(work);
    return n_tokens;
}

const char *tokenizer_decode(const Tokenizer *t, int token) {
    if (token < 0 || token >= t->vocab_size) return "";
    return t->vocab[token];
}
