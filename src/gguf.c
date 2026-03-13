#include "gguf.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Low-level read helpers (little-endian assumed) ---

typedef struct {
    const uint8_t *buf;
    size_t pos;
    size_t size;
    int error;  // set on OOB read
} Reader;

static int reader_ok(Reader *r, size_t need) {
    if (r->error) return 0;
    return r->pos + need <= r->size;
}

// #12: All read helpers now check bounds and set error flag on OOB
static uint8_t read_u8(Reader *r) {
    if (!reader_ok(r, 1)) { r->error = 1; return 0; }
    return r->buf[r->pos++];
}

static uint16_t read_u16(Reader *r) {
    if (!reader_ok(r, 2)) { r->error = 1; return 0; }
    uint16_t v;
    memcpy(&v, r->buf + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(Reader *r) {
    if (!reader_ok(r, 4)) { r->error = 1; return 0; }
    uint32_t v;
    memcpy(&v, r->buf + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(Reader *r) {
    if (!reader_ok(r, 8)) { r->error = 1; return 0; }
    uint64_t v;
    memcpy(&v, r->buf + r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(Reader *r) {
    if (!reader_ok(r, 4)) { r->error = 1; return 0.0f; }
    float v;
    memcpy(&v, r->buf + r->pos, 4);
    r->pos += 4;
    return v;
}

static double read_f64(Reader *r) {
    if (!reader_ok(r, 8)) { r->error = 1; return 0.0; }
    double v;
    memcpy(&v, r->buf + r->pos, 8);
    r->pos += 8;
    return v;
}

// #3: NULL-check malloc return
static char *read_string(Reader *r) {
    uint64_t len = read_u64(r);
    if (r->error || !reader_ok(r, len)) { r->error = 1; return NULL; }
    // #22: Guard against huge len causing malloc issues
    if (len > BN_GGUF_MAX_STRING_LEN) { r->error = 1; return NULL; }
    char *s = (char *)malloc((size_t)len + 1);
    if (!s) { r->error = 1; return NULL; }
    memcpy(s, r->buf + r->pos, (size_t)len);
    s[len] = '\0';
    r->pos += (size_t)len;
    return s;
}

// #3: NULL-check malloc return
static BnGGUFString read_gguf_string(Reader *r) {
    BnGGUFString s = {0};
    s.len = read_u64(r);
    if (r->error || !reader_ok(r, s.len)) { r->error = 1; return s; }
    if (s.len > BN_GGUF_MAX_STRING_LEN) { r->error = 1; return s; }
    s.str = (char *)malloc((size_t)s.len + 1);
    if (!s.str) { r->error = 1; s.len = 0; return s; }
    memcpy(s.str, r->buf + r->pos, (size_t)s.len);
    s.str[s.len] = '\0';
    r->pos += (size_t)s.len;
    return s;
}

// Size of a scalar GGUF value type
static size_t gguf_type_size(uint32_t type) {
    switch (type) {
        case BN_GGUF_TYPE_UINT8:   return 1;
        case BN_GGUF_TYPE_INT8:    return 1;
        case BN_GGUF_TYPE_UINT16:  return 2;
        case BN_GGUF_TYPE_INT16:   return 2;
        case BN_GGUF_TYPE_UINT32:  return 4;
        case BN_GGUF_TYPE_INT32:   return 4;
        case BN_GGUF_TYPE_FLOAT32: return 4;
        case BN_GGUF_TYPE_BOOL:    return 1;
        case BN_GGUF_TYPE_UINT64:  return 8;
        case BN_GGUF_TYPE_INT64:   return 8;
        case BN_GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static void read_kv_value(Reader *r, BnGGUFKeyValue *kv) {
    if (r->error) return;
    switch (kv->type) {
        case BN_GGUF_TYPE_UINT8:   kv->value.u8 = read_u8(r);   break;
        case BN_GGUF_TYPE_INT8:    kv->value.i8 = (int8_t)read_u8(r); break;
        case BN_GGUF_TYPE_UINT16:  kv->value.u16 = read_u16(r);  break;
        case BN_GGUF_TYPE_INT16:   kv->value.i16 = (int16_t)read_u16(r); break;
        case BN_GGUF_TYPE_UINT32:  kv->value.u32 = read_u32(r);  break;
        case BN_GGUF_TYPE_INT32:   kv->value.i32 = (int32_t)read_u32(r); break;
        case BN_GGUF_TYPE_FLOAT32: kv->value.f32 = read_f32(r);  break;
        case BN_GGUF_TYPE_BOOL:    kv->value.b = read_u8(r);     break;
        case BN_GGUF_TYPE_STRING:  kv->value.str = read_gguf_string(r); break;
        case BN_GGUF_TYPE_UINT64:  kv->value.u64 = read_u64(r);  break;
        case BN_GGUF_TYPE_INT64:   kv->value.i64 = (int64_t)read_u64(r); break;
        case BN_GGUF_TYPE_FLOAT64: kv->value.f64 = read_f64(r);  break;
        case BN_GGUF_TYPE_ARRAY: {
            BnGGUFArray *a = &kv->value.arr;
            a->elem_type = read_u32(r);
            a->n = read_u64(r);
            a->data = NULL;
            a->strings = NULL;
            if (r->error) break;

            // #4: Overflow check on array size
            if (a->elem_type == BN_GGUF_TYPE_STRING) {
                if (a->n > SIZE_MAX / sizeof(BnGGUFString)) { r->error = 1; break; }
                // #18: NULL-check malloc
                a->strings = (BnGGUFString *)malloc((size_t)a->n * sizeof(BnGGUFString));
                if (!a->strings) { r->error = 1; break; }
                for (uint64_t i = 0; i < a->n; i++) {
                    a->strings[i] = read_gguf_string(r);
                    if (r->error) break;
                }
            } else {
                size_t elem_sz = gguf_type_size(a->elem_type);
                // #4: Check multiplication overflow
                if (elem_sz > 0 && a->n <= SIZE_MAX / elem_sz && reader_ok(r, (size_t)a->n * elem_sz)) {
                    a->data = (void *)(r->buf + r->pos);  // point into buffer
                    r->pos += (size_t)a->n * elem_sz;
                }
            }
            break;
        }
        default:
            r->error = 1;  // unknown type
            break;
    }
}

static size_t align_up(size_t offset, size_t alignment) {
    return offset + (alignment - (offset % alignment)) % alignment;
}

BnGGUFFile *bn_gguf_open(const uint8_t *buf, size_t size) {
    Reader r = { buf, 0, size, 0 };

    // Check minimum size and magic
    if (!reader_ok(&r, 4 + 4 + 8 + 8)) return NULL;

    uint32_t magic = read_u32(&r);
    if (magic != BN_GGUF_MAGIC) {
        char buf[16]; snprintf(buf, sizeof(buf), "0x%08x", magic);
        SH_LOG_ERROR("Bad GGUF magic", "got", buf);
        return NULL;
    }

    BnGGUFFile *f = (BnGGUFFile *)calloc(1, sizeof(BnGGUFFile));
    if (!f) return NULL;
    f->raw = (uint8_t *)buf;
    f->raw_size = size;  // #13: store buffer size for bounds checking
    f->version = read_u32(&r);
    f->n_tensors = read_u64(&r);
    f->n_kv = read_u64(&r);
    f->alignment = BN_GGUF_DEFAULT_ALIGNMENT;

    if (f->version < 2 || f->version > 3) {
        char buf[16]; snprintf(buf, sizeof(buf), "%u", f->version);
        SH_LOG_ERROR("Unsupported GGUF version", "version", buf);
        free(f);
        return NULL;
    }

    // #22: Sanity check counts to avoid huge allocations from malicious files
    if (f->n_kv > BN_GGUF_MAX_COUNT || f->n_tensors > BN_GGUF_MAX_COUNT) {
        SH_LOG_ERROR("Unreasonable GGUF counts");
        free(f);
        return NULL;
    }

    // Read KV pairs
    f->kvs = (BnGGUFKeyValue *)calloc((size_t)f->n_kv, sizeof(BnGGUFKeyValue));
    if (f->n_kv > 0 && !f->kvs) { free(f); return NULL; }

    for (uint64_t i = 0; i < f->n_kv; i++) {
        f->kvs[i].key = read_string(&r);
        if (r.error) goto fail;
        f->kvs[i].type = read_u32(&r);
        if (r.error) goto fail;
        read_kv_value(&r, &f->kvs[i]);
        if (r.error) goto fail;

        // Check for alignment override
        if (f->kvs[i].key && strcmp(f->kvs[i].key, "general.alignment") == 0
            && f->kvs[i].type == BN_GGUF_TYPE_UINT32) {
            f->alignment = f->kvs[i].value.u32;
        }
    }

    // Read tensor infos
    f->tensors = (BnGGUFTensorInfo *)calloc((size_t)f->n_tensors, sizeof(BnGGUFTensorInfo));
    if (f->n_tensors > 0 && !f->tensors) goto fail;

    for (uint64_t i = 0; i < f->n_tensors; i++) {
        f->tensors[i].name = read_string(&r);
        if (r.error) goto fail;
        f->tensors[i].n_dims = read_u32(&r);
        if (r.error) goto fail;

        // #11: Validate n_dims to prevent dims[] buffer overflow
        if (f->tensors[i].n_dims > BN_GGUF_MAX_DIMS) {
            SH_LOG_ERROR("Tensor has too many dims",
                         "tensor", f->tensors[i].name ? f->tensors[i].name : "?");
            goto fail;
        }

        for (uint32_t d = 0; d < f->tensors[i].n_dims; d++) {
            f->tensors[i].dims[d] = read_u64(&r);
            if (r.error) goto fail;
        }
        f->tensors[i].type = read_u32(&r);
        f->tensors[i].offset = read_u64(&r);
        if (r.error) goto fail;
    }

    // Compute data offset (aligned after header)
    f->data_offset = align_up(r.pos, f->alignment);

    return f;

fail:
    bn_gguf_free(f);
    return NULL;
}

void bn_gguf_free(BnGGUFFile *f) {
    if (!f) return;
    if (f->kvs) {
        for (uint64_t i = 0; i < f->n_kv; i++) {
            free(f->kvs[i].key);
            if (f->kvs[i].type == BN_GGUF_TYPE_STRING) {
                free(f->kvs[i].value.str.str);
            } else if (f->kvs[i].type == BN_GGUF_TYPE_ARRAY) {
                BnGGUFArray *a = &f->kvs[i].value.arr;
                if (a->strings) {
                    for (uint64_t j = 0; j < a->n; j++) {
                        free(a->strings[j].str);
                    }
                    free(a->strings);
                }
            }
        }
        free(f->kvs);
    }
    if (f->tensors) {
        for (uint64_t i = 0; i < f->n_tensors; i++) {
            free(f->tensors[i].name);
        }
        free(f->tensors);
    }
    free(f);
}

int bn_gguf_find_key(BnGGUFFile *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (f->kvs[i].key && strcmp(f->kvs[i].key, key) == 0) {
            // #22: Guard against truncation of huge index
            if (i > (uint64_t)INT32_MAX) return -1;
            return (int)i;
        }
    }
    return -1;
}

// #21: Type-validated getters
uint32_t bn_gguf_get_u32(BnGGUFFile *f, const char *key) {
    int i = bn_gguf_find_key(f, key);
    if (i < 0) return 0;
    if (f->kvs[i].type != BN_GGUF_TYPE_UINT32) return 0;
    return f->kvs[i].value.u32;
}

float bn_gguf_get_f32(BnGGUFFile *f, const char *key) {
    int i = bn_gguf_find_key(f, key);
    if (i < 0) return 0.0f;
    if (f->kvs[i].type != BN_GGUF_TYPE_FLOAT32) return 0.0f;
    return f->kvs[i].value.f32;
}

const char *bn_gguf_get_str(BnGGUFFile *f, const char *key) {
    int i = bn_gguf_find_key(f, key);
    if (i < 0) return NULL;
    if (f->kvs[i].type != BN_GGUF_TYPE_STRING) return NULL;
    return f->kvs[i].value.str.str;
}

uint64_t bn_gguf_get_arr_n(BnGGUFFile *f, const char *key) {
    int i = bn_gguf_find_key(f, key);
    if (i < 0) return 0;
    if (f->kvs[i].type != BN_GGUF_TYPE_ARRAY) return 0;
    return f->kvs[i].value.arr.n;
}

// #34: Explicit negative idx check
const char *bn_gguf_get_arr_str(BnGGUFFile *f, const char *key, int idx) {
    int i = bn_gguf_find_key(f, key);
    if (i < 0) return NULL;
    if (f->kvs[i].type != BN_GGUF_TYPE_ARRAY) return NULL;
    BnGGUFArray *a = &f->kvs[i].value.arr;
    if (!a->strings || idx < 0 || (uint64_t)idx >= a->n) return NULL;
    return a->strings[idx].str;
}

const void *bn_gguf_get_arr_data(BnGGUFFile *f, const char *key) {
    int i = bn_gguf_find_key(f, key);
    if (i < 0) return NULL;
    if (f->kvs[i].type != BN_GGUF_TYPE_ARRAY) return NULL;
    return f->kvs[i].value.arr.data;
}

int bn_gguf_find_tensor(BnGGUFFile *f, const char *name) {
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        if (f->tensors[i].name && strcmp(f->tensors[i].name, name) == 0) {
            if (i > (uint64_t)INT32_MAX) return -1;
            return (int)i;
        }
    }
    return -1;
}

// #13: Validate that tensor data pointer falls within the mapped buffer
void *bn_gguf_tensor_data(BnGGUFFile *f, int idx) {
    if (idx < 0 || (uint64_t)idx >= f->n_tensors) return NULL;
    size_t offset = f->data_offset + f->tensors[idx].offset;
    if (offset >= f->raw_size) return NULL;
    return f->raw + offset;
}
