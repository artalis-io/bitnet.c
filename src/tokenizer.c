#include "tokenizer.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// --- Internal: sorted vocab for binary search ---

// #6: Use qsort_r (or compatible) to avoid global mutable state.
// macOS/BSD and glibc have qsort_r but with different signatures.
// We use a portable approach: store context pointer in BnTokenizer struct
// and use a platform-appropriate qsort_r.

#if defined(__APPLE__) || defined(__FreeBSD__)
// BSD qsort_r: comparator takes context as last argument
static int cmp_vocab_indirect_bsd(void *ctx, const void *a, const void *b) {
    char **vocab = (char **)ctx;
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return strcmp(vocab[ia], vocab[ib]);
}
#define SORT_VOCAB(indices, n, vocab) \
    qsort_r(indices, n, sizeof(int), (void *)(vocab), cmp_vocab_indirect_bsd)
#elif defined(__linux__) && defined(_GNU_SOURCE)
// GNU qsort_r: comparator takes context as first argument
static int cmp_vocab_indirect_gnu(const void *a, const void *b, void *ctx) {
    char **vocab = (char **)ctx;
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return strcmp(vocab[ia], vocab[ib]);
}
#define SORT_VOCAB(indices, n, vocab) \
    qsort_r(indices, n, sizeof(int), cmp_vocab_indirect_gnu, (void *)(vocab))
#else
// Fallback: use global context (not thread-safe, but functional)
static char **g_sort_vocab;
static int cmp_vocab_indirect_fallback(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return strcmp(g_sort_vocab[ia], g_sort_vocab[ib]);
}
#define SORT_VOCAB(indices, n, vocab) do { \
    g_sort_vocab = (vocab); \
    qsort(indices, n, sizeof(int), cmp_vocab_indirect_fallback); \
} while (0)
#endif

// Binary search for a token string in sorted vocab
static int vocab_lookup(const BnTokenizer *t, const char *str) {
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

int bn_tokenizer_init(BnTokenizer *t, BnGGUFFile *f) {
    memset(t, 0, sizeof(BnTokenizer));

    t->vocab_size = (int)bn_gguf_get_arr_n(f, "tokenizer.ggml.tokens");
    if (t->vocab_size == 0) {
        SH_LOG_ERROR("No tokens found in GGUF");
        return -1;
    }

    // Overflow check before allocation
    if ((size_t)t->vocab_size > SIZE_MAX / sizeof(char *)) {
        SH_LOG_ERROR("Vocab size overflow");
        return -1;
    }

    // #16: Check all allocations
    t->vocab = (char **)malloc(t->vocab_size * sizeof(char *));
    if (!t->vocab) {
        SH_LOG_ERROR("Failed to allocate vocab");
        return -1;
    }
    t->max_token_length = 0;

    for (int i = 0; i < t->vocab_size; i++) {
        const char *tok = bn_gguf_get_arr_str(f, "tokenizer.ggml.tokens", i);
        // #33: Check strdup return
        t->vocab[i] = tok ? strdup(tok) : strdup("");
        if (!t->vocab[i]) {
            // Clean up partial allocations
            for (int j = 0; j < i; j++) free(t->vocab[j]);
            free(t->vocab);
            t->vocab = NULL;
            return -1;
        }
        int len = (int)strlen(t->vocab[i]);
        if (len > t->max_token_length) t->max_token_length = len;
    }

    // Load scores (optional — some tokenizers don't have scores)
    // #16: Check allocation
    t->scores = (float *)calloc(t->vocab_size, sizeof(float));
    if (!t->scores) {
        bn_tokenizer_free(t);
        return -1;
    }
    const void *scores_data = bn_gguf_get_arr_data(f, "tokenizer.ggml.scores");
    if (scores_data) {
        memcpy(t->scores, scores_data, t->vocab_size * sizeof(float));
    }

    // Special token IDs
    int idx;
    idx = bn_gguf_find_key(f, "tokenizer.ggml.bos_token_id");
    t->bos_id = (idx >= 0) ? (int)bn_gguf_get_u32(f, "tokenizer.ggml.bos_token_id") : 1;

    idx = bn_gguf_find_key(f, "tokenizer.ggml.eos_token_id");
    t->eos_id = (idx >= 0) ? (int)bn_gguf_get_u32(f, "tokenizer.ggml.eos_token_id") : 2;

    idx = bn_gguf_find_key(f, "tokenizer.ggml.eot_token_id");
    t->eot_id = (idx >= 0) ? (int)bn_gguf_get_u32(f, "tokenizer.ggml.eot_token_id") : -1;

    // #16: Build sorted index for binary search
    if ((size_t)t->vocab_size > SIZE_MAX / sizeof(int)) {
        bn_tokenizer_free(t);
        return -1;
    }
    t->sorted_indices = (int *)malloc(t->vocab_size * sizeof(int));
    if (!t->sorted_indices) {
        bn_tokenizer_free(t);
        return -1;
    }
    for (int i = 0; i < t->vocab_size; i++) t->sorted_indices[i] = i;

    // #6: Use platform qsort_r to avoid global mutable state
    SORT_VOCAB(t->sorted_indices, t->vocab_size, t->vocab);

    // Fallback: resolve eot_id from vocab if GGUF metadata didn't provide it
    if (t->eot_id < 0)
        t->eot_id = vocab_lookup(t, "<|eot_id|>");

    return 0;
}

void bn_tokenizer_free(BnTokenizer *t) {
    if (!t) return;
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    free(t->scores);
    free(t->sorted_indices);
}

// Encode text using BPE merge algorithm
int bn_tokenizer_encode(const BnTokenizer *t, const char *text, int add_bos,
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
    // #17: Check allocations
    int *work = (int *)malloc((text_len + 1) * sizeof(int));
    if (!work) return n_tokens;
    int n_work = 0;

    for (int i = 0; i < text_len; ) {
        unsigned char byte = (unsigned char)text[i];
        char bpe_char[4];
        int bpe_len;

        // Convert raw byte to its BPE Unicode representation
        if (byte >= 0x21 && byte <= 0x7E) {
            // Printable ASCII: maps to itself
            bpe_char[0] = (char)byte;
            bpe_char[1] = '\0';
            bpe_len = 1;
        } else if (byte >= 0xA1 && byte <= 0xAC) {
            // Latin-1 printable: maps to itself (2-byte UTF-8)
            bpe_char[0] = (char)(0xC0 | (byte >> 6));
            bpe_char[1] = (char)(0x80 | (byte & 0x3F));
            bpe_char[2] = '\0';
            bpe_len = 2;
        } else if (byte >= 0xAE && byte <= 0xFF) {
            // Latin-1 printable: maps to itself (2-byte UTF-8)
            bpe_char[0] = (char)(0xC0 | (byte >> 6));
            bpe_char[1] = (char)(0x80 | (byte & 0x3F));
            bpe_char[2] = '\0';
            bpe_len = 2;
        } else {
            // Non-printable byte: maps to U+0100..U+0143
            int cp;
            if (byte <= 0x20) {
                cp = BN_BPE_UNICODE_OFFSET + byte;  // 0x00-0x20 → U+0100-U+0120
            } else if (byte == 0x7F) {
                cp = 0x121;                          // DEL → U+0121
            } else if (byte >= 0x80 && byte <= 0xA0) {
                cp = 0x122 + (byte - 0x80);          // 0x80-0xA0 → U+0122-U+0142
            } else {
                cp = BN_BPE_UNICODE_END;              // 0xAD → U+0143
            }
            bpe_char[0] = (char)(0xC0 | (cp >> 6));
            bpe_char[1] = (char)(0x80 | (cp & 0x3F));
            bpe_char[2] = '\0';
            bpe_len = 2;
        }
        (void)bpe_len;

        int tok = vocab_lookup(t, bpe_char);
        if (tok >= 0) {
            work[n_work++] = tok;
        } else {
            // Try byte fallback tokens like <0xNN>
            char byte_tok[8];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", byte);
            tok = vocab_lookup(t, byte_tok);
            if (tok >= 0) {
                work[n_work++] = tok;
            }
        }
        i++;
    }

    // Step 2: BPE merge loop — greedily merge the pair with highest score
    // #17: Check allocation
    char *merge_buf = (char *)malloc(t->max_token_length * 2 + 4);
    if (!merge_buf) {
        // Copy what we have and return
        for (int i = 0; i < n_work && n_tokens < max_tokens; i++) {
            tokens[n_tokens++] = work[i];
        }
        free(work);
        return n_tokens;
    }

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

int bn_tokenizer_lookup(const BnTokenizer *t, const char *str) {
    return vocab_lookup(t, str);
}

// GPT-2/GPT-4 byte-level BPE reverse mapping.
// The BPE vocab uses Unicode codepoints to represent all 256 byte values:
// - Printable ASCII 0x21-0x7E map to themselves (U+0021-U+007E)
// - Latin-1 0xA1-0xAC map to themselves (U+00A1-U+00AC)
// - Latin-1 0xAE-0xFF map to themselves (U+00AE-U+00FF)
// - All other bytes (0x00-0x20, 0x7F-0xA0, 0xAD) map to U+0100..U+0143

// Reverse table: codepoint (U+0100 + index) → raw byte
static const uint8_t bpe_n2b[68] = {
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,  // U+0100-0107
    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,  // U+0108-010F  (0x0A = newline = Ċ)
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,  // U+0110-0117
    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,  // U+0118-011F
    0x20,                                       // U+0120 = space = Ġ
    0x7F,                                       // U+0121 = DEL
    0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,  // U+0122-0129
    0x88,0x89,0x8A,0x8B,0x8C,0x8D,0x8E,0x8F,  // U+012A-0131
    0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,  // U+0132-0139
    0x98,0x99,0x9A,0x9B,0x9C,0x9D,0x9E,0x9F,  // U+013A-0141
    0xA0,                                       // U+0142 = NBSP
    0xAD,                                       // U+0143 = soft hyphen
};

// Decode one UTF-8 codepoint from the BPE token string, return raw byte value
// #31: Check for string end before reading continuation byte
static int decode_bpe_cp(const unsigned char **p) {
    unsigned char c0 = **p;
    if (c0 < 0x80) {
        // ASCII: maps to itself in BPE
        (*p)++;
        return c0;
    }
    if ((c0 & 0xE0) == 0xC0) {
        // 2-byte UTF-8: check that continuation byte exists
        if ((*p)[1] == '\0') { (*p)++; return c0; }
        unsigned char c1 = (*p)[1];
        if ((c1 & 0xC0) != 0x80) { (*p)++; return c0; }
        int cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        *p += 2;
        // Identity-mapped Latin-1 ranges
        if (cp >= 0xA1 && cp <= 0xAC) return cp;
        if (cp >= 0xAE && cp <= 0xFF) return cp;
        // Non-printable byte mapping (U+0100..U+0143)
        if (cp >= BN_BPE_UNICODE_OFFSET && cp <= BN_BPE_UNICODE_END)
            return bpe_n2b[cp - BN_BPE_UNICODE_OFFSET];
        return cp;  // other codepoints: return as-is
    }
    // 3+ byte UTF-8: skip and return first byte
    (*p)++;
    return c0;
}

// #7: Decode buffer is now per-BnTokenizer to avoid global mutable state.
// For backwards compatibility, we use a thread-local static buffer.
// The caller must consume or copy the result before the next call.
static _Thread_local char tl_decode_buf[1024];

const char *bn_tokenizer_decode(const BnTokenizer *t, int token) {
    if (token < 0 || token >= t->vocab_size) return "";
    const char *raw = t->vocab[token];

    // Check if this token contains any high bytes (BPE encoded)
    int has_high = 0;
    for (const char *p = raw; *p; p++) {
        if ((unsigned char)*p >= 0x80) { has_high = 1; break; }
    }
    if (!has_high) return raw;  // pure ASCII, return as-is

    // Decode BPE character mappings
    const unsigned char *p = (const unsigned char *)raw;
    int out_pos = 0;
    while (*p && out_pos < (int)sizeof(tl_decode_buf) - 4) {
        int byte = decode_bpe_cp(&p);
        if (byte >= 0 && byte <= 0xFF) {
            tl_decode_buf[out_pos++] = (char)(unsigned char)byte;
        }
    }
    tl_decode_buf[out_pos] = '\0';
    return tl_decode_buf;
}
