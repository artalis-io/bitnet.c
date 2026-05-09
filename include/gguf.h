#ifndef BN_GGUF_H
#define BN_GGUF_H

#include <stdint.h>
#include <stddef.h>

#define BN_GGUF_MAGIC              0x46554747   // "GGUF" little-endian
#define BN_GGUF_MAX_DIMS           4
#define BN_GGUF_DEFAULT_ALIGNMENT  32
#define BN_GGUF_MAX_STRING_LEN     ((uint64_t)1 << 30)
#define BN_GGUF_MAX_COUNT          ((uint64_t)1 << 20)

// GGUF value types
enum {
    BN_GGUF_TYPE_UINT8   = 0,
    BN_GGUF_TYPE_INT8    = 1,
    BN_GGUF_TYPE_UINT16  = 2,
    BN_GGUF_TYPE_INT16   = 3,
    BN_GGUF_TYPE_UINT32  = 4,
    BN_GGUF_TYPE_INT32   = 5,
    BN_GGUF_TYPE_FLOAT32 = 6,
    BN_GGUF_TYPE_BOOL    = 7,
    BN_GGUF_TYPE_STRING  = 8,
    BN_GGUF_TYPE_ARRAY   = 9,
    BN_GGUF_TYPE_UINT64  = 10,
    BN_GGUF_TYPE_INT64   = 11,
    BN_GGUF_TYPE_FLOAT64 = 12,
};

// GGUF tensor types we care about
enum {
    BN_GGUF_TENSOR_F32      = 0,
    BN_GGUF_TENSOR_F16      = 1,
    BN_GGUF_TENSOR_Q4_0     = 2,
    BN_GGUF_TENSOR_Q4_1     = 3,
    BN_GGUF_TENSOR_Q5_0     = 6,
    BN_GGUF_TENSOR_Q5_1     = 7,
    BN_GGUF_TENSOR_Q8_0     = 8,
    BN_GGUF_TENSOR_Q2_K     = 10,
    BN_GGUF_TENSOR_Q3_K     = 11,
    BN_GGUF_TENSOR_Q4_K     = 12,
    BN_GGUF_TENSOR_Q5_K     = 13,
    BN_GGUF_TENSOR_Q6_K     = 14,
    BN_GGUF_TENSOR_Q8_K     = 15,
    BN_GGUF_TENSOR_IQ2_XXS  = 16,
    BN_GGUF_TENSOR_IQ2_XS   = 17,
    BN_GGUF_TENSOR_IQ3_XXS  = 18,
    BN_GGUF_TENSOR_IQ4_NL   = 20,
    BN_GGUF_TENSOR_IQ3_S    = 21,
    BN_GGUF_TENSOR_IQ2_S    = 22,
    BN_GGUF_TENSOR_IQ4_XS   = 23,
    BN_GGUF_TENSOR_BF16     = 30,
    BN_GGUF_TENSOR_TQ1_0    = 34,
    BN_GGUF_TENSOR_TQ2_0    = 35,
    BN_GGUF_TENSOR_I2_S     = 36,
};

typedef struct {
    uint64_t len;
    char    *str;
} BnGGUFString;

typedef struct {
    uint32_t elem_type;
    uint64_t n;
    void    *data;       // raw array data (for non-string arrays)
    BnGGUFString *strings; // for string arrays
} BnGGUFArray;

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
        BnGGUFString str;
        BnGGUFArray  arr;
        uint64_t  u64;
        int64_t   i64;
        double    f64;
    } value;
} BnGGUFKeyValue;

typedef struct {
    char    *name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
} BnGGUFTensorInfo;

typedef struct {
    uint32_t      version;
    uint64_t      n_tensors;
    uint64_t      n_kv;
    BnGGUFKeyValue *kvs;
    BnGGUFTensorInfo *tensors;
    size_t        alignment;
    size_t        data_offset;
    uint8_t      *raw;       // pointer to start of buffer (for tensor data access)
    size_t        raw_size;  // total buffer size (for bounds checking)
} BnGGUFFile;

BnGGUFFile *bn_gguf_open(const uint8_t *buf, size_t size);
void        bn_gguf_free(BnGGUFFile *f);
int         bn_gguf_find_key(BnGGUFFile *f, const char *key);
uint32_t    bn_gguf_get_u32(BnGGUFFile *f, const char *key);
float       bn_gguf_get_f32(BnGGUFFile *f, const char *key);
const char *bn_gguf_get_str(BnGGUFFile *f, const char *key);
uint64_t    bn_gguf_get_arr_n(BnGGUFFile *f, const char *key);
const char *bn_gguf_get_arr_str(BnGGUFFile *f, const char *key, int i);
const void *bn_gguf_get_arr_data(BnGGUFFile *f, const char *key);
int         bn_gguf_find_tensor(BnGGUFFile *f, const char *name);
void       *bn_gguf_tensor_data(BnGGUFFile *f, int idx);
int         bn_gguf_tensor_size(uint32_t type, uint64_t nelements, size_t *out);

#endif // BN_GGUF_H
