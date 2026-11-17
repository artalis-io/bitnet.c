#include "tokenizer.h"
#include "model_tokenizer_policy.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

static const char *BN_TOKENIZER_METASPACE = "\xE2\x96\x81";

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

/* Ranked byte BPE uses pair ranks, not the score of the resulting token. */
struct BnTokenizerMerge { int left, right, token, rank; };

static int tokenizer_merge_compare(const void *a, const void *b) {
    const struct BnTokenizerMerge *x = a, *y = b;
    if (x->left != y->left) return x->left < y->left ? -1 : 1;
    if (x->right != y->right) return x->right < y->right ? -1 : 1;
    return (x->rank > y->rank) - (x->rank < y->rank);
}

static const struct BnTokenizerMerge *tokenizer_find_merge(
        const BnTokenizer *t, int left, int right) {
    int lo = 0, hi = t->n_merges;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        const struct BnTokenizerMerge *m = &t->merges[mid];
        if (m->left < left || (m->left == left && m->right < right)) lo = mid + 1;
        else hi = mid;
    }
    if (lo == t->n_merges || t->merges[lo].left != left ||
        t->merges[lo].right != right) return NULL;
    return &t->merges[lo];
}

static int tokenizer_load_merges(BnTokenizer *t, BnGGUFFile *f) {
    int idx = bn_gguf_find_key(f, "tokenizer.ggml.merges");
    if (idx < 0) return 0;
    if (f->kvs[idx].type != BN_GGUF_TYPE_ARRAY ||
        f->kvs[idx].value.arr.elem_type != BN_GGUF_TYPE_STRING ||
        f->kvs[idx].value.arr.n > INT_MAX ||
        f->kvs[idx].value.arr.n > SIZE_MAX / sizeof(*t->merges)) return -1;
    t->ranked_bpe = 1;
    t->n_merges = (int)f->kvs[idx].value.arr.n;
    if (!t->n_merges) return 0;
    t->merges = malloc((size_t)t->n_merges * sizeof(*t->merges));
    if (!t->merges) return -1;
    for (int i = 0; i < t->n_merges; ++i) {
        const char *entry = bn_gguf_get_arr_str(f, "tokenizer.ggml.merges", i);
        if (!entry || !entry[0]) return -1;
        char *pair = strdup(entry);
        if (!pair) return -1;
        char *sep = strchr(pair + 1, ' ');
        if (!sep || !sep[1]) { free(pair); return -1; }
        *sep = 0;
        int left = vocab_lookup(t, pair), right = vocab_lookup(t, sep + 1);
        memmove(sep, sep + 1, strlen(sep + 1) + 1);
        int token = vocab_lookup(t, pair);
        free(pair);
        if (left < 0 || right < 0 || token < 0) return -1;
        t->merges[i] = (struct BnTokenizerMerge){left, right, token, i};
    }
    qsort(t->merges, (size_t)t->n_merges, sizeof(*t->merges), tokenizer_merge_compare);
    return 0;
}

#include "tokenizer_unicode.h"

enum { TOKENIZER_NUMBER = 2, TOKENIZER_LETTER = 4,
       TOKENIZER_MARK = 16, TOKENIZER_SPACE = 256 };

/* Invalid UTF-8 advances one byte; byte BPE can still preserve the input. */
static int tokenizer_codepoint(const char *s, int pos, int len, uint32_t *cp) {
    if (pos >= len) { *cp = UINT32_MAX; return len; }
    const unsigned char *p = (const unsigned char *)s;
    unsigned c = p[pos];
    int n = c < 0x80 ? 1 : c >= 0xc2 && c < 0xe0 ? 2 :
            c < 0xf0 && c >= 0xe0 ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 1;
    uint32_t value = n == 1 ? c : c & ((1u << (7 - n)) - 1);
    if (n > len - pos) n = 1;
    for (int i = 1; i < n; i++) {
        if ((p[pos+i] & 0xc0) != 0x80) { n = 1; break; }
        value = (value << 6) | (p[pos+i] & 0x3f);
    }
    if ((n == 2 && value < 0x80) || (n == 3 && value < 0x800) ||
        (n == 4 && value < 0x10000) || value > 0x10ffff ||
        (value >= 0xd800 && value <= 0xdfff)) n = 1;
    *cp = n == 1 ? c : value;
    return pos + n;
}

static unsigned tokenizer_char_flags(uint32_t cp) {
    if (cp > 0x10ffff) return 0;
    size_t lo = 0, hi = sizeof(tokenizer_unicode_ranges) / sizeof(tokenizer_unicode_ranges[0]);
    while (lo + 1 < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (tokenizer_unicode_ranges[mid].start <= cp) lo = mid;
        else hi = mid;
    }
    return tokenizer_unicode_ranges[lo].flags;
}

/* Ordered alternatives of letter-based BPE pretokenizers. Byte offsets keep
 * UTF-8 codepoints intact before the byte-to-Unicode BPE mapping. */
static int tokenizer_piece_end(const char *s, int start, int len, int marks, int number_group) {
    uint32_t cp, next_cp;
    int next = tokenizer_codepoint(s, start, len, &cp);
    tokenizer_codepoint(s, next, len, &next_cp);
    unsigned flags = tokenizer_char_flags(cp);
    unsigned word = TOKENIZER_LETTER | (marks ? TOKENIZER_MARK : 0);
    if (cp == '\'' && next < len) {
        unsigned a = (unsigned char)s[next] | 32u;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return next + 1;
        if (next + 1 < len) {
            unsigned b = (unsigned char)s[next+1] | 32u;
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
                (a == 'l' && b == 'l')) return next + 2;
        }
    }
    if (cp != '\r' && cp != '\n' && !(flags & TOKENIZER_NUMBER) &&
        ((flags & word) || (tokenizer_char_flags(next_cp) & word))) {
        int end = next;
        while (end < len) {
            int after = tokenizer_codepoint(s, end, len, &next_cp);
            if (!(tokenizer_char_flags(next_cp) & word)) break;
            end = after;
        }
        return end;
    }
    if (flags & TOKENIZER_NUMBER) {
        for (int n = 1; n < number_group && next < len; n++) {
            int after = tokenizer_codepoint(s, next, len, &next_cp);
            if (!(tokenizer_char_flags(next_cp) & TOKENIZER_NUMBER)) break;
            next = after;
        }
        return next;
    }
    unsigned excluded = word | TOKENIZER_NUMBER | TOKENIZER_SPACE;
    int pos = cp == ' ' ? next : start;
    int end = pos;
    while (end < len) {
        int after = tokenizer_codepoint(s, end, len, &next_cp);
        if (tokenizer_char_flags(next_cp) & excluded) break;
        end = after;
    }
    if (end > pos) {
        while (end < len && (s[end] == '\r' || s[end] == '\n')) end++;
        return end;
    }
    int last_newline = start, previous = start;
    end = start;
    while (end < len) {
        int after = tokenizer_codepoint(s, end, len, &next_cp);
        if (!(tokenizer_char_flags(next_cp) & TOKENIZER_SPACE)) break;
        if (next_cp == '\r' || next_cp == '\n') last_newline = after;
        previous = end;
        end = after;
    }
    if (last_newline > start) return last_newline;
    if (end < len && previous > start) return previous;
    return end > start ? end : next;
}

static void tokenizer_add_eog(BnTokenizer *t, int token) {
    if (!t || token < 0)
        return;
    for (int i = 0; i < t->n_eog; i++)
        if (t->eog_ids[i] == token)
            return;
    if (t->n_eog < BN_TOKENIZER_MAX_EOG)
        t->eog_ids[t->n_eog++] = token;
}

#include "tokenizer_nfc_data.h"

static unsigned tokenizer_nfc_ccc(uint32_t cp) {
    size_t lo = 0, hi = sizeof(tokenizer_nfc_classes) / sizeof(tokenizer_nfc_classes[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (tokenizer_nfc_classes[mid].cp < cp) lo = mid + 1;
        else hi = mid;
    }
    return lo < sizeof(tokenizer_nfc_classes) / sizeof(tokenizer_nfc_classes[0]) &&
        tokenizer_nfc_classes[lo].cp == cp ? tokenizer_nfc_classes[lo].ccc : 0;
}

static uint32_t tokenizer_nfc_compose(uint32_t a, uint32_t b) {
    /* Hangul composition is algorithmic, including LV + trailing Jamo. */
    if (a >= 0x1100 && a < 0x1113 && b >= 0x1161 && b < 0x1176)
        return 0xac00 + ((a - 0x1100) * 21 + b - 0x1161) * 28;
    if (a >= 0xac00 && a < 0xd7a4 && (a - 0xac00) % 28 == 0 &&
        b > 0x11a7 && b < 0x11c3) return a + b - 0x11a7;
    size_t lo = 0, hi = sizeof(tokenizer_nfc_compositions) / sizeof(tokenizer_nfc_compositions[0]);
    uint64_t key = ((uint64_t)a << 32) | b;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        uint64_t entry = ((uint64_t)tokenizer_nfc_compositions[mid].a << 32) |
            tokenizer_nfc_compositions[mid].b;
        if (entry < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo < sizeof(tokenizer_nfc_compositions) / sizeof(tokenizer_nfc_compositions[0]) &&
        tokenizer_nfc_compositions[lo].a == a && tokenizer_nfc_compositions[lo].b == b)
        return tokenizer_nfc_compositions[lo].composed;
    return 0;
}

/* Invalid UTF-8 bytes remain individual opaque barriers. They cannot acquire
 * combining classes or compose across a malformed sequence. */
static uint32_t tokenizer_nfc_read(const unsigned char *s, size_t len, size_t *pos) {
    size_t start = *pos;
    unsigned a = s[start], width;
    uint32_t cp, minimum;
    if (a < 0x80) { (*pos)++; return a; }
    if (a >= 0xc2 && a <= 0xdf) { width = 2; cp = a & 31; minimum = 0x80; }
    else if (a >= 0xe0 && a <= 0xef) { width = 3; cp = a & 15; minimum = 0x800; }
    else if (a >= 0xf0 && a <= 0xf4) { width = 4; cp = a & 7; minimum = 0x10000; }
    else goto invalid;
    if (width > len - start) goto invalid;
    for (unsigned i = 1; i < width; i++) {
        if ((s[start + i] & 0xc0) != 0x80) goto invalid;
        cp = (cp << 6) | (s[start + i] & 63);
    }
    if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) goto invalid;
    *pos += width;
    return cp;
invalid:
    (*pos)++;
    return 0x110000 + a;
}

static size_t tokenizer_nfc_decompose(uint32_t cp, uint32_t *out) {
    if (cp >= 0xac00 && cp < 0xd7a4) {
        uint32_t index = cp - 0xac00;
        out[0] = 0x1100 + index / 588;
        out[1] = 0x1161 + index % 588 / 28;
        if (index % 28) { out[2] = 0x11a7 + index % 28; return 3; }
        return 2;
    }
    size_t lo = 0, hi = sizeof(tokenizer_nfc_decompositions) / sizeof(tokenizer_nfc_decompositions[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (tokenizer_nfc_decompositions[mid].cp < cp) lo = mid + 1;
        else hi = mid;
    }
    if (lo < sizeof(tokenizer_nfc_decompositions) / sizeof(tokenizer_nfc_decompositions[0]) &&
        tokenizer_nfc_decompositions[lo].cp == cp) {
        size_t n = tokenizer_nfc_decompositions[lo].n;
        memcpy(out, tokenizer_nfc_decompositions[lo].values, n * sizeof(uint32_t));
        return n;
    }
    out[0] = cp;
    return 1;
}

/* Stable merge ordering avoids quadratic work for long runs of combining marks. */
static void tokenizer_nfc_order(uint32_t *cp, uint32_t *scratch, size_t n) {
    if (n < 2) return;
    size_t middle = n / 2;
    tokenizer_nfc_order(cp, scratch, middle);
    tokenizer_nfc_order(cp + middle, scratch + middle, n - middle);
    size_t i = 0, j = middle, out = 0;
    while (i < middle && j < n)
        scratch[out++] = tokenizer_nfc_ccc(cp[i]) <= tokenizer_nfc_ccc(cp[j]) ? cp[i++] : cp[j++];
    while (i < middle) scratch[out++] = cp[i++];
    while (j < n) scratch[out++] = cp[j++];
    memcpy(cp, scratch, n * sizeof(uint32_t));
}

static size_t tokenizer_nfc_prepare(const char *text,size_t len,uint32_t *cp,uint32_t *provided) {
    size_t n = 0, pos = 0;
    while (pos < len) {
        uint32_t value = tokenizer_nfc_read((const unsigned char *)text, len, &pos);
        n += tokenizer_nfc_decompose(value, cp + n);
    }
    uint32_t *scratch = provided;
    for (size_t start = 0; start < n; ) {
        if (!tokenizer_nfc_ccc(cp[start])) { start++; continue; }
        size_t end = start + 1;
        unsigned previous = tokenizer_nfc_ccc(cp[start]);
        int ordered = 1;
        while (end < n) {
            unsigned current = tokenizer_nfc_ccc(cp[end]);
            if (!current) break;
            if (current < previous) ordered = 0;
            previous = current;
            end++;
        }
        if (!ordered) {
            if (!scratch) scratch = malloc(n * sizeof(uint32_t));
            if (!scratch)return SIZE_MAX;
            tokenizer_nfc_order(cp + start, scratch + start, end - start);
        }
        start = end;
    }
    if(!provided)free(scratch);
    size_t out = 0, starter = SIZE_MAX;
    unsigned previous = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned current = tokenizer_nfc_ccc(cp[i]);
        uint32_t combined = starter == SIZE_MAX ? 0 : tokenizer_nfc_compose(cp[starter], cp[i]);
        if (combined && (previous == 0 || previous < current)) cp[starter] = combined;
        else {
            if (!current) starter = out;
            cp[out++] = cp[i];
            previous = current;
        }
    }
    return out;
}
static size_t tokenizer_nfc_write(const uint32_t *cp,size_t out,char *result) {
    size_t written = 0;
    for (size_t i = 0; i < out; i++) {
        uint32_t c = cp[i];
        if (c > 0x10ffff) result[written++] = (char)(c - 0x110000);
        else if (c < 0x80) result[written++] = (char)c;
        else if (c < 0x800) {
            result[written++] = (char)(0xc0 | (c >> 6));
            result[written++] = (char)(0x80 | (c & 63));
        } else if (c < 0x10000) {
            result[written++] = (char)(0xe0 | (c >> 12));
            result[written++] = (char)(0x80 | ((c >> 6) & 63));
            result[written++] = (char)(0x80 | (c & 63));
        } else {
            result[written++] = (char)(0xf0 | (c >> 18));
            result[written++] = (char)(0x80 | ((c >> 12) & 63));
            result[written++] = (char)(0x80 | ((c >> 6) & 63));
            result[written++] = (char)(0x80 | (c & 63));
        }
    }
    result[written] = 0;
    return written;
}
static char *tokenizer_normalize_nfc(const char *text, size_t len) {
    if(len>SIZE_MAX/(4*sizeof(uint32_t)))return NULL;
    uint32_t *cp=malloc((len?len:1)*4*sizeof(uint32_t));if(!cp)return NULL;
    size_t out=tokenizer_nfc_prepare(text,len,cp,NULL);
    if(out==SIZE_MAX||out>(SIZE_MAX-1)/4){free(cp);return NULL;}
    char *result=malloc(out*4+1);if(!result){free(cp);return NULL;}
    (void)tokenizer_nfc_write(cp,out,result);free(cp);
    return result;
}

static int tokenizer_added_at(const BnTokenizer *t, const char *text, size_t remaining, size_t *length) {
    int added = -1;
    *length = 0;
    if (!remaining || !t->added_prefix[(unsigned char)text[0]]) return -1;
    for (int j = 0; j < t->n_added; j++) {
        int id = t->added_ids[j];
        size_t len = strlen(t->vocab[id]);
        if (len > *length && len <= remaining && memcmp(text, t->vocab[id], len) == 0) {
            added = id;
            *length = len;
        }
    }
    return added;
}

static int tokenizer_encode_nfc(const BnTokenizer *t, const char *text, int add_bos,
                                int *tokens, int max_tokens) {
    BnTokenizer plain = *t;
    plain.normalize_nfc = 0;
    plain.n_added = 0;
    memset(plain.added_prefix, 0, sizeof(plain.added_prefix));
    size_t len = strlen(text), start = 0;
    if (len > INT_MAX - 1) return 0;
    int n = 0;
    if (add_bos) tokens[n++] = t->bos_id;
    while (start < len && n < max_tokens) {
        size_t end = start, added_len = 0;
        int added = -1;
        while (end < len) {
            added = tokenizer_added_at(t, text + end, len - end, &added_len);
            if (added >= 0) break;
            end++;
        }
        if (end > start) {
            char *normalized = tokenizer_normalize_nfc(text + start, end - start);
            if (!normalized) return 0;
            int encoded = bn_tokenizer_encode(&plain, normalized, 0, tokens + n, max_tokens - n);
            int missing = normalized[0] && encoded == 0;
            free(normalized);
            if (missing) return 0;
            n += encoded;
        }
        if (added >= 0 && n < max_tokens) tokens[n++] = added;
        start = end + added_len;
    }
    return n;
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

    /* Added tokens are isolated before byte-level pretokenization. */
    int types_idx = bn_gguf_find_key(f, "tokenizer.ggml.token_type");
    if (!bn_model_tokenizer_uses_metaspace(bn_gguf_get_str(f, "tokenizer.ggml.model")) && types_idx >= 0) {
        const BnGGUFKeyValue *kv = &f->kvs[types_idx];
        if (kv->type != BN_GGUF_TYPE_ARRAY ||
            kv->value.arr.elem_type != BN_GGUF_TYPE_INT32 ||
            kv->value.arr.n < (uint64_t)t->vocab_size || !kv->value.arr.data) {
            bn_tokenizer_free(t);
            return -1;
        }
        t->added_ids = malloc((size_t)t->vocab_size * sizeof(int));
        if (!t->added_ids) { bn_tokenizer_free(t); return -1; }
        for (int i = 0; i < t->vocab_size; i++) {
            int32_t type;
            memcpy(&type, (const unsigned char *)kv->value.arr.data + (size_t)i * 4, 4);
            if ((type == 3 || type == 4) && t->vocab[i][0]) {
                t->added_ids[t->n_added++] = i;
                t->added_prefix[(unsigned char)t->vocab[i][0]] = 1;
            }
        }
    }

    // Load scores (optional — some tokenizers don't have scores)
    // #16: Check allocation
    t->scores = (float *)calloc(t->vocab_size, sizeof(float));
    if (!t->scores) {
        bn_tokenizer_free(t);
        return -1;
    }
    const void *scores_data = NULL;
    int scores_idx = bn_gguf_find_key(f, "tokenizer.ggml.scores");
    if (scores_idx >= 0 &&
        f->kvs[scores_idx].type == BN_GGUF_TYPE_ARRAY &&
        f->kvs[scores_idx].value.arr.elem_type == BN_GGUF_TYPE_FLOAT32 &&
        f->kvs[scores_idx].value.arr.n >= (uint64_t)t->vocab_size) {
        scores_data = f->kvs[scores_idx].value.arr.data;
    }
    if (scores_data) {
        memcpy(t->scores, scores_data, t->vocab_size * sizeof(float));
    }

    const char *model_name = bn_gguf_get_str(f, "tokenizer.ggml.model");
    t->metaspace = bn_model_tokenizer_uses_metaspace(model_name);
    t->normalize_nfc = bn_model_tokenizer_normalizes_nfc(
        bn_gguf_get_str(f, "tokenizer.ggml.pre"));

    // Special token IDs
    int idx;
    int has_bos = (bn_gguf_find_key(f, "tokenizer.ggml.bos_token_id") >= 0);
    t->bos_id = has_bos ? (int)bn_gguf_get_u32(f, "tokenizer.ggml.bos_token_id") : 1;

    idx = bn_gguf_find_key(f, "tokenizer.ggml.eos_token_id");
    t->eos_id = (idx >= 0) ? (int)bn_gguf_get_u32(f, "tokenizer.ggml.eos_token_id") : 2;

    idx = bn_gguf_find_key(f, "tokenizer.ggml.eot_token_id");
    t->eot_id = (idx >= 0) ? (int)bn_gguf_get_u32(f, "tokenizer.ggml.eot_token_id") : -1;

    // Respect explicit GGUF metadata. For omitted values, let model-owned
    // tokenizer policy supply the family default.
    idx = bn_gguf_find_key(f, "tokenizer.ggml.add_bos_token");
    const char *tokenizer_pre =
        bn_gguf_get_str(f, "tokenizer.ggml.pre");
    t->add_bos = (idx >= 0)
        ? (int)f->kvs[idx].value.b
        : bn_model_tokenizer_default_add_bos(tokenizer_pre, has_bos);
    if (t->metaspace && has_bos)
        t->add_bos = 1;

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

    if (tokenizer_load_merges(t, f) != 0) {
        SH_LOG_ERROR("Invalid tokenizer BPE merges");
        bn_tokenizer_free(t);
        return -1;
    }
    t->pretokenizer = bn_model_tokenizer_pretokenizer(
        tokenizer_pre ? tokenizer_pre : model_name);

    // Fallback: resolve eot_id from vocab if GGUF metadata didn't provide it
    if (t->eot_id < 0)
        t->eot_id = vocab_lookup(t, "<|eot_id|>");

    // Detect ChatML-style templates.
    t->im_start_id = vocab_lookup(t, "<|im_start|>");
    t->im_end_id   = vocab_lookup(t, "<|im_end|>");
    t->chatml = (t->im_start_id >= 0 && t->im_end_id >= 0) ? 1 : 0;

    tokenizer_add_eog(t, t->eos_id);
    tokenizer_add_eog(t, t->eot_id);
    tokenizer_add_eog(t, t->im_end_id);
    static const char *const eog_spellings[] = {
        "<|endoftext|>", "<|eot_id|>", "<|im_end|>", "<turn|>",
        "<|tool_response|>", "<|fim_pad|>", "<|repo_name|>",
        "<|file_sep|>",
    };
    for (size_t i = 0; i < sizeof(eog_spellings) / sizeof(eog_spellings[0]); i++)
        tokenizer_add_eog(t, vocab_lookup(t, eog_spellings[i]));

    for (int i = 0; ; i++) {
        const char *extra = bn_model_tokenizer_extra_eog(tokenizer_pre, i);
        if (!extra) break;
        tokenizer_add_eog(t, vocab_lookup(t, extra));
    }
    return 0;
}

int bn_tokenizer_is_eog(const BnTokenizer *t, int token) {
    if (!t)
        return 0;
    for (int i = 0; i < t->n_eog; i++)
        if (t->eog_ids[i] == token)
            return 1;
    return 0;
}

void bn_tokenizer_free(BnTokenizer *t) {
    if (!t) return;
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    free(t->added_ids);
    free(t->merges);
    free(t->scores);
    free(t->sorted_indices);
}

static int tokenizer_init_metaspace_work(const BnTokenizer *t, const char *text,
                                         int *work, int *groups, int max_work) {
    int n_work = 0, text_len = (int)strlen(text), piece_end = 0;
    for (int i = 0; i < text_len && n_work < max_work; ) {
        if (t->pretokenizer == BN_TOKENIZER_PRE_NEWLINES && i >= piece_end) {
            piece_end = i + 1;
            while (piece_end < text_len &&
                   (text[piece_end] == '\n') == (text[i] == '\n')) piece_end++;
            /* A whole newline run can be a vocabulary token even though
             * newline-containing pairs are absent from the merge table. */
            if (text[i] == '\n' && piece_end - i <= t->max_token_length) {
                size_t length = (size_t)(piece_end - i);
                char *run = (char *)malloc(length + 1);
                if (!run) return n_work;
                memcpy(run, text + i, length); run[length] = 0;
                int tok = vocab_lookup(t, run);
                free(run);
                if (tok >= 0) {
                    groups[n_work] = piece_end; work[n_work++] = tok;
                    i = piece_end;
                    continue;
                }
            }
        }
        char piece[8];
        uint32_t cp;
        int next = tokenizer_codepoint(text, i, text_len, &cp);
        int piece_len = next - i;
        if (cp == ' ') {
            memcpy(piece, BN_TOKENIZER_METASPACE, 4);
        } else {
            memcpy(piece, text + i, (size_t)piece_len); piece[piece_len] = 0;
        }
        int tok = vocab_lookup(t, piece);
        if (tok >= 0) {
            groups[n_work] = piece_end; work[n_work++] = tok;
        } else {
            /* Raw UTF-8 BPE falls back after normalization: an unknown
             * metaspace itself therefore yields its three UTF-8 bytes. */
            for (int j = 0; piece[j] && n_work < max_work; j++) {
                char byte_token[8];
                snprintf(byte_token, sizeof(byte_token), "<0x%02X>",
                         (unsigned char)piece[j]);
                tok = vocab_lookup(t, byte_token);
                if (tok >= 0) {
                    groups[n_work] = piece_end; work[n_work++] = tok;
                }
            }
        }
        i = next;
    }
    return n_work;
}

typedef struct {
    const BnTokenizerGGUFScratch *owner;const BnTokenizer *tokenizer;
    size_t input_capacity,work_capacity,merge_bytes;int *work;char *merge;
    int normalize_nfc;uint32_t *codepoints,*order;char *normalized;int *normalized_tokens;
} GGUFScratch;
typedef struct {size_t capacity,work,merge,codepoints,normalized,tokens,total;} GGUFScratchPlan;
static int tokenizer_gguf_scratch_shape(const BnTokenizer *t,size_t input_capacity,GGUFScratchPlan *out) {
    if(!t||!out||!t->vocab||!t->sorted_indices||t->vocab_size<1||t->metaspace||
       (t->normalize_nfc!=0&&t->normalize_nfc!=1)||
       t->max_token_length<1||t->max_token_length>(INT_MAX-4)/2||!input_capacity||input_capacity>INT_MAX-1||
       input_capacity+1>SIZE_MAX/(2*sizeof(int)))return -1;
    GGUFScratchPlan p={0};p.capacity=input_capacity;
    if(t->normalize_nfc) {
        /* The pinned normalizer bounds canonical codepoints at4 per input
         * byte; UTF8 encoding then needs at most16 bytes per input byte. */
        if(input_capacity>(INT_MAX-1)/16||input_capacity>SIZE_MAX/(4*sizeof(uint32_t)))return -1;
        p.capacity=input_capacity*16;p.codepoints=input_capacity*4*sizeof(uint32_t);
        p.normalized=p.capacity+1;
    }
    if(p.capacity+1>SIZE_MAX/(2*sizeof(int)))return -1;
    p.work=(p.capacity+1)*2*sizeof(int);p.merge=(size_t)t->max_token_length*2+4;
    if(t->normalize_nfc)p.tokens=(p.capacity+1)*sizeof(int);
    const size_t parts[]={sizeof(GGUFScratch),p.work,p.merge,p.codepoints,p.codepoints,p.normalized,p.tokens};
    for(size_t i=0;i<sizeof(parts)/sizeof(parts[0]);i++){if(parts[i]>SIZE_MAX-p.total)return -1;p.total+=parts[i];}
    *out=p;return 0;
}
int bn_tokenizer_gguf_scratch_plan(const BnTokenizer *t,size_t input_capacity,size_t *host_bytes) {
    GGUFScratchPlan p;if(!host_bytes||tokenizer_gguf_scratch_shape(t,input_capacity,&p))return -1;
    *host_bytes=p.total;return 0;
}
int bn_tokenizer_gguf_scratch_init(BnTokenizerGGUFScratch *scratch,const BnTokenizer *t,
    size_t input_capacity,size_t host_budget) {
    GGUFScratchPlan p;
    if(!scratch||scratch->state||tokenizer_gguf_scratch_shape(t,input_capacity,&p)||p.total>host_budget)return -1;
    GGUFScratch *s=calloc(1,sizeof(*s));if(!s)return -1;
    s->work=malloc(p.work);s->merge_bytes=p.merge;
    s->merge=malloc(s->merge_bytes);
    if(t->normalize_nfc) {
        s->codepoints=malloc(p.codepoints);s->order=malloc(p.codepoints);
        s->normalized=malloc(p.normalized);s->normalized_tokens=malloc(p.tokens);
    }
    if(!s->work||!s->merge||(t->normalize_nfc&&(!s->codepoints||!s->order||!s->normalized||!s->normalized_tokens))) {
        free(s->work);free(s->merge);free(s->codepoints);free(s->order);free(s->normalized);free(s->normalized_tokens);free(s);return -1;
    }
    s->owner=scratch;s->tokenizer=t;s->input_capacity=input_capacity;s->work_capacity=p.capacity+1;s->normalize_nfc=t->normalize_nfc;
    scratch->state=s;return 0;
}
void bn_tokenizer_gguf_scratch_free(BnTokenizerGGUFScratch *scratch) {
    GGUFScratch *s=scratch?scratch->state:NULL;if(!s||s->owner!=scratch)return;
    free(s->work);free(s->merge);free(s->codepoints);free(s->order);free(s->normalized);free(s->normalized_tokens);
    free(s);scratch->state=NULL;
}
// Original ranked/score BPE algorithm, shared by allocating and retained paths.
static int tokenizer_encode_work(const BnTokenizer *t,const char *text,int add_bos,
    int *tokens,int max_tokens,GGUFScratch *scratch,size_t input_len) {
    if (!text || !tokens || max_tokens <= 0) return 0;
    if (t->normalize_nfc) {
        const unsigned char *p = (const unsigned char *)text;
        while (*p && *p < 0x80) p++;
        if (*p) return tokenizer_encode_nfc(t, text, add_bos, tokens, max_tokens);
    }

    int n_tokens = 0;

    // Add BOS if requested
    if (add_bos && n_tokens < max_tokens) {
        if(!scratch)tokens[n_tokens]=t->bos_id;
        n_tokens++;
    }

    // Initial tokenization: encode each byte/char as individual token
    // For UTF-8 text, first try to find each character as a token
    size_t expansion = t->metaspace ? 3u : 1u;
    if (input_len > (INT_MAX - 1) / expansion) return n_tokens;
    size_t capacity = input_len * expansion + 1;
    if (capacity > SIZE_MAX / (2 * sizeof(int))) return n_tokens;
    int text_len = (int)input_len;
    if (text_len == 0) {if(scratch&&n_tokens)tokens[0]=t->bos_id;return n_tokens;}

    // Step 1: Initialize with individual character tokens
    // #17: Check allocations
    int *work = scratch?scratch->work:(int *)calloc(capacity, 2 * sizeof(int));
    if (!work) return n_tokens;
    int n_work = 0;
    int *groups = work + (scratch?scratch->work_capacity:capacity);
    int piece_end = 0;

    if (t->metaspace) {
        n_work = tokenizer_init_metaspace_work(t, text, work, groups, (int)capacity);
    } else for (int i = 0; i < text_len; ) {
        size_t added_len;
        int added = tokenizer_added_at(t, text + i, (size_t)(text_len - i), &added_len);
        if (added >= 0) {
            groups[n_work] = -(i + 1);
            work[n_work++] = added;
            i += (int)added_len;
            piece_end = i;
            continue;
        }
        if (t->pretokenizer != BN_TOKENIZER_PRE_NONE && i >= piece_end)
            piece_end = tokenizer_piece_end(text, i, text_len, t->pretokenizer == BN_TOKENIZER_PRE_LETTERS_MARKS,
                t->pretokenizer == BN_TOKENIZER_PRE_GLM4 ? 3 : 1);
        groups[n_work] = piece_end;
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
        } else if (byte >= 0xAE) {
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

    // Step 2: BPE merges within each pretokenizer piece. Ranked BPE uses
    // the lowest pair rank; legacy score-based tokenizers use the highest score.
    // #17: Check allocation
    char *merge_buf = scratch?scratch->merge:(char *)malloc(t->max_token_length * 2 + 4);
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
        int best_rank = INT_MAX;
        int best_idx = -1;
        int best_tok = -1;

        // Find the best merge pair
        for (int i = 0; i < n_work - 1; i++) {
            if (groups[i] != groups[i+1]) continue;
            if (t->ranked_bpe) {
                const struct BnTokenizerMerge *m = tokenizer_find_merge(t, work[i], work[i+1]);
                if (m && m->rank < best_rank) {
                    best_rank = m->rank;
                    best_idx = i;
                    best_tok = m->token;
                }
                continue;
            }
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
            groups[i] = groups[i + 1];
        }
        n_work--;
    }

    if(!scratch)free(merge_buf);

    if(scratch) {
        if(n_work>max_tokens-n_tokens)return -1;
        if(n_tokens)tokens[0]=t->bos_id;
    }

    // Copy results
    for (int i = 0; i < n_work && n_tokens < max_tokens; i++) {
        tokens[n_tokens++] = work[i];
    }

    if(!scratch)free(work);
    return n_tokens;
}

int bn_tokenizer_encode(const BnTokenizer *t,const char *text,int add_bos,int *tokens,int max_tokens) {
    if(!text||!tokens||max_tokens<=0)return 0;
    return tokenizer_encode_work(t,text,add_bos,tokens,max_tokens,NULL,strlen(text));
}
static int tokenizer_gguf_encode_nfc(const BnTokenizer *t,GGUFScratch *s,const char *text,size_t len,
    int add_bos,int *tokens,int max_tokens) {
    BnTokenizer plain=*t;plain.normalize_nfc=0;plain.n_added=0;memset(plain.added_prefix,0,sizeof(plain.added_prefix));
    int count=0;if(add_bos)s->normalized_tokens[count++]=t->bos_id;
    size_t start=0;
    while(start<len) {
        size_t end=start,added_len=0;int added=-1;
        while(end<len){added=tokenizer_added_at(t,text+end,len-end,&added_len);if(added>=0)break;end++;}
        if(end>start) {
            size_t n=tokenizer_nfc_prepare(text+start,end-start,s->codepoints,s->order);
            if(n==SIZE_MAX||n>(s->work_capacity-1)/4)return -1;
            size_t written=tokenizer_nfc_write(s->codepoints,n,s->normalized);
            int encoded=tokenizer_encode_work(&plain,s->normalized,0,s->normalized_tokens+count,
                (int)(s->work_capacity-(size_t)count),s,written);
            if(encoded<0||(written&&!encoded))return -1;
            count+=encoded;
        }
        if(added>=0) {
            if((size_t)count==s->work_capacity)return -1;
            s->normalized_tokens[count++]=added;end+=added_len;
        }
        start=end;
    }
    if(count>max_tokens)return -1;
    memcpy(tokens,s->normalized_tokens,(size_t)count*sizeof(int));return count;
}
int bn_tokenizer_gguf_encode(const BnTokenizer *t,BnTokenizerGGUFScratch *scratch,
    const char *text,int add_bos,int *tokens,int max_tokens) {
    GGUFScratch *s=scratch?scratch->state:NULL;
    if(!s||s->owner!=scratch||s->tokenizer!=t||!text||!tokens||max_tokens<1||
       (add_bos!=0&&add_bos!=1)||t->metaspace||t->normalize_nfc!=s->normalize_nfc||t->max_token_length<1||
       (size_t)t->max_token_length>(s->merge_bytes-4)/2)return -1;
    size_t bytes=0;while(bytes<=s->input_capacity&&text[bytes])bytes++;
    if(bytes>s->input_capacity)return -1;
    if(t->normalize_nfc) {
        size_t i=0;while(i<bytes&&(unsigned char)text[i]<128)i++;
        if(i<bytes)return tokenizer_gguf_encode_nfc(t,s,text,bytes,add_bos,tokens,max_tokens);
    }
    return tokenizer_encode_work(t,text,add_bos,tokens,max_tokens,s,bytes);
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

static int tokenizer_hex_digit(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

const char *bn_tokenizer_decode(const BnTokenizer *t, int token) {
    if (token < 0 || token >= t->vocab_size) return "";
    const char *raw = t->vocab[token];

    if (t->metaspace) {
        if (strlen(raw) == 6 && strncmp(raw, "<0x", 3) == 0 && raw[5] == '>') {
            int hi = tokenizer_hex_digit((unsigned char)raw[3]);
            int lo = tokenizer_hex_digit((unsigned char)raw[4]);
            if (hi >= 0 && lo >= 0) {
                tl_decode_buf[0] = (char)((hi << 4) | lo);
                tl_decode_buf[1] = 0;
                return tl_decode_buf;
            }
        }
        int out_pos = 0;
        for (const unsigned char *p = (const unsigned char *)raw;
             *p && out_pos < (int)sizeof(tl_decode_buf) - 4; ) {
            if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) {
                tl_decode_buf[out_pos++] = ' ';
                p += 3;
            } else {
                tl_decode_buf[out_pos++] = (char)*p++;
            }
        }
        tl_decode_buf[out_pos] = '\0';
        return tl_decode_buf;
    }

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

#include "tokenizer_vocab.h"

#include "tokenizer_map.h"

#include "tokenizer_encode.h"

#include "tokenizer_decode_stream.h"
