#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>

// GGUF value types
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// GGUF tensor types we care about
enum {
    GGUF_TENSOR_F32   = 0,
    GGUF_TENSOR_F16   = 1,
    GGUF_TENSOR_Q6_K  = 14,
    GGUF_TENSOR_TQ1_0 = 34,
    GGUF_TENSOR_TQ2_0 = 35,
};

typedef struct {
    uint64_t len;
    char    *str;
} GGUFString;

typedef struct {
    uint32_t elem_type;
    uint64_t n;
    void    *data;       // raw array data (for non-string arrays)
    GGUFString *strings; // for string arrays
} GGUFArray;

typedef struct {
    char    *key;
    uint32_t type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        float     f32;
        uint8_t   b;     // bool
        GGUFString str;
        GGUFArray  arr;
        uint64_t  u64;
        int64_t   i64;
        double    f64;
    } value;
} GGUFKeyValue;

typedef struct {
    char    *name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
} GGUFTensorInfo;

typedef struct {
    uint32_t      version;
    uint64_t      n_tensors;
    uint64_t      n_kv;
    GGUFKeyValue *kvs;
    GGUFTensorInfo *tensors;
    size_t        alignment;
    size_t        data_offset;
    uint8_t      *raw;       // pointer to start of buffer (for tensor data access)
} GGUFFile;

GGUFFile   *gguf_open(const uint8_t *buf, size_t size);
void        gguf_free(GGUFFile *f);
int         gguf_find_key(GGUFFile *f, const char *key);
uint32_t    gguf_get_u32(GGUFFile *f, const char *key);
float       gguf_get_f32(GGUFFile *f, const char *key);
const char *gguf_get_str(GGUFFile *f, const char *key);
uint64_t    gguf_get_arr_n(GGUFFile *f, const char *key);
const char *gguf_get_arr_str(GGUFFile *f, const char *key, int i);
const void *gguf_get_arr_data(GGUFFile *f, const char *key);
int         gguf_find_tensor(GGUFFile *f, const char *name);
void       *gguf_tensor_data(GGUFFile *f, int idx);

#endif // GGUF_H
