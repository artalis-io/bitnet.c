#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Low-level read helpers (little-endian assumed) ---

typedef struct {
    const uint8_t *buf;
    size_t pos;
    size_t size;
} Reader;

static int reader_ok(Reader *r, size_t need) {
    return r->pos + need <= r->size;
}

static uint8_t read_u8(Reader *r) {
    return r->buf[r->pos++];
}

static uint16_t read_u16(Reader *r) {
    uint16_t v;
    memcpy(&v, r->buf + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(Reader *r) {
    uint32_t v;
    memcpy(&v, r->buf + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(Reader *r) {
    uint64_t v;
    memcpy(&v, r->buf + r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(Reader *r) {
    float v;
    memcpy(&v, r->buf + r->pos, 4);
    r->pos += 4;
    return v;
}

static double read_f64(Reader *r) {
    double v;
    memcpy(&v, r->buf + r->pos, 8);
    r->pos += 8;
    return v;
}

static char *read_string(Reader *r) {
    uint64_t len = read_u64(r);
    if (!reader_ok(r, len)) return NULL;
    char *s = (char *)malloc(len + 1);
    memcpy(s, r->buf + r->pos, len);
    s[len] = '\0';
    r->pos += len;
    return s;
}

static GGUFString read_gguf_string(Reader *r) {
    GGUFString s = {0};
    s.len = read_u64(r);
    if (!reader_ok(r, s.len)) return s;
    s.str = (char *)malloc(s.len + 1);
    memcpy(s.str, r->buf + r->pos, s.len);
    s.str[s.len] = '\0';
    r->pos += s.len;
    return s;
}

// Size of a scalar GGUF value type
static size_t gguf_type_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static void read_kv_value(Reader *r, GGUFKeyValue *kv) {
    switch (kv->type) {
        case GGUF_TYPE_UINT8:   kv->value.u8 = read_u8(r);   break;
        case GGUF_TYPE_INT8:    kv->value.i8 = (int8_t)read_u8(r); break;
        case GGUF_TYPE_UINT16:  kv->value.u16 = read_u16(r);  break;
        case GGUF_TYPE_INT16:   kv->value.i16 = (int16_t)read_u16(r); break;
        case GGUF_TYPE_UINT32:  kv->value.u32 = read_u32(r);  break;
        case GGUF_TYPE_INT32:   kv->value.i32 = (int32_t)read_u32(r); break;
        case GGUF_TYPE_FLOAT32: kv->value.f32 = read_f32(r);  break;
        case GGUF_TYPE_BOOL:    kv->value.b = read_u8(r);     break;
        case GGUF_TYPE_STRING:  kv->value.str = read_gguf_string(r); break;
        case GGUF_TYPE_UINT64:  kv->value.u64 = read_u64(r);  break;
        case GGUF_TYPE_INT64:   kv->value.i64 = (int64_t)read_u64(r); break;
        case GGUF_TYPE_FLOAT64: kv->value.f64 = read_f64(r);  break;
        case GGUF_TYPE_ARRAY: {
            GGUFArray *a = &kv->value.arr;
            a->elem_type = read_u32(r);
            a->n = read_u64(r);
            a->data = NULL;
            a->strings = NULL;
            if (a->elem_type == GGUF_TYPE_STRING) {
                a->strings = (GGUFString *)malloc(a->n * sizeof(GGUFString));
                for (uint64_t i = 0; i < a->n; i++) {
                    a->strings[i] = read_gguf_string(r);
                }
            } else {
                size_t elem_sz = gguf_type_size(a->elem_type);
                if (elem_sz > 0 && reader_ok(r, a->n * elem_sz)) {
                    a->data = (void *)(r->buf + r->pos);  // point into buffer
                    r->pos += a->n * elem_sz;
                }
            }
            break;
        }
    }
}

static size_t align_up(size_t offset, size_t alignment) {
    return offset + (alignment - (offset % alignment)) % alignment;
}

GGUFFile *gguf_open(const uint8_t *buf, size_t size) {
    Reader r = { buf, 0, size };

    // Check minimum size and magic
    if (!reader_ok(&r, 4 + 4 + 8 + 8)) return NULL;

    uint32_t magic = read_u32(&r);
    if (magic != 0x46554747) {  // "GGUF" in little-endian
        fprintf(stderr, "gguf: bad magic 0x%08x\n", magic);
        return NULL;
    }

    GGUFFile *f = (GGUFFile *)calloc(1, sizeof(GGUFFile));
    f->raw = (uint8_t *)buf;
    f->version = read_u32(&r);
    f->n_tensors = read_u64(&r);
    f->n_kv = read_u64(&r);
    f->alignment = 32;  // default

    if (f->version < 2 || f->version > 3) {
        fprintf(stderr, "gguf: unsupported version %u\n", f->version);
        free(f);
        return NULL;
    }

    // Read KV pairs
    f->kvs = (GGUFKeyValue *)calloc(f->n_kv, sizeof(GGUFKeyValue));
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (!reader_ok(&r, 8)) goto fail;
        f->kvs[i].key = read_string(&r);
        if (!reader_ok(&r, 4)) goto fail;
        f->kvs[i].type = read_u32(&r);
        read_kv_value(&r, &f->kvs[i]);

        // Check for alignment override
        if (f->kvs[i].key && strcmp(f->kvs[i].key, "general.alignment") == 0) {
            f->alignment = f->kvs[i].value.u32;
        }
    }

    // Read tensor infos
    f->tensors = (GGUFTensorInfo *)calloc(f->n_tensors, sizeof(GGUFTensorInfo));
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        if (!reader_ok(&r, 8)) goto fail;
        f->tensors[i].name = read_string(&r);
        if (!reader_ok(&r, 4)) goto fail;
        f->tensors[i].n_dims = read_u32(&r);
        for (uint32_t d = 0; d < f->tensors[i].n_dims; d++) {
            if (!reader_ok(&r, 8)) goto fail;
            f->tensors[i].dims[d] = read_u64(&r);
        }
        if (!reader_ok(&r, 4 + 8)) goto fail;
        f->tensors[i].type = read_u32(&r);
        f->tensors[i].offset = read_u64(&r);
    }

    // Compute data offset (aligned after header)
    f->data_offset = align_up(r.pos, f->alignment);

    return f;

fail:
    gguf_free(f);
    return NULL;
}

void gguf_free(GGUFFile *f) {
    if (!f) return;
    if (f->kvs) {
        for (uint64_t i = 0; i < f->n_kv; i++) {
            free(f->kvs[i].key);
            if (f->kvs[i].type == GGUF_TYPE_STRING) {
                free(f->kvs[i].value.str.str);
            } else if (f->kvs[i].type == GGUF_TYPE_ARRAY) {
                GGUFArray *a = &f->kvs[i].value.arr;
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

int gguf_find_key(GGUFFile *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (f->kvs[i].key && strcmp(f->kvs[i].key, key) == 0) return (int)i;
    }
    return -1;
}

uint32_t gguf_get_u32(GGUFFile *f, const char *key) {
    int i = gguf_find_key(f, key);
    if (i < 0) return 0;
    return f->kvs[i].value.u32;
}

float gguf_get_f32(GGUFFile *f, const char *key) {
    int i = gguf_find_key(f, key);
    if (i < 0) return 0.0f;
    return f->kvs[i].value.f32;
}

const char *gguf_get_str(GGUFFile *f, const char *key) {
    int i = gguf_find_key(f, key);
    if (i < 0) return NULL;
    return f->kvs[i].value.str.str;
}

uint64_t gguf_get_arr_n(GGUFFile *f, const char *key) {
    int i = gguf_find_key(f, key);
    if (i < 0) return 0;
    return f->kvs[i].value.arr.n;
}

const char *gguf_get_arr_str(GGUFFile *f, const char *key, int idx) {
    int i = gguf_find_key(f, key);
    if (i < 0) return NULL;
    GGUFArray *a = &f->kvs[i].value.arr;
    if (!a->strings || (uint64_t)idx >= a->n) return NULL;
    return a->strings[idx].str;
}

const void *gguf_get_arr_data(GGUFFile *f, const char *key) {
    int i = gguf_find_key(f, key);
    if (i < 0) return NULL;
    return f->kvs[i].value.arr.data;
}

int gguf_find_tensor(GGUFFile *f, const char *name) {
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        if (f->tensors[i].name && strcmp(f->tensors[i].name, name) == 0)
            return (int)i;
    }
    return -1;
}

void *gguf_tensor_data(GGUFFile *f, int idx) {
    if (idx < 0 || (uint64_t)idx >= f->n_tensors) return NULL;
    return f->raw + f->data_offset + f->tensors[idx].offset;
}
