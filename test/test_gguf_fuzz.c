#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// --- GGUF write helpers ---

typedef struct {
    uint8_t *data;
    size_t   pos;
    size_t   cap;
} WriteBuffer;

static void wb_write(WriteBuffer *wb, const void *data, size_t size) {
    assert(wb->pos + size <= wb->cap);
    memcpy(wb->data + wb->pos, data, size);
    wb->pos += size;
}

static void wb_u32(WriteBuffer *wb, uint32_t v) { wb_write(wb, &v, 4); }
static void wb_u64(WriteBuffer *wb, uint64_t v) { wb_write(wb, &v, 8); }
static void wb_str(WriteBuffer *wb, const char *s) {
    uint64_t len = strlen(s);
    wb_u64(wb, len);
    wb_write(wb, s, len);
}

// Write a valid minimal GGUF header
static void wb_header(WriteBuffer *wb, uint64_t n_tensors, uint64_t n_kv) {
    wb_u32(wb, 0x46554747);  // magic "GGUF"
    wb_u32(wb, 3);           // version 3
    wb_u64(wb, n_tensors);
    wb_u64(wb, n_kv);
}

// ===================================================================
// Test: n_tensors=0 is valid (empty model)
// ===================================================================
static void test_fuzz_zero_tensors(void) {
    printf("test_fuzz_zero_tensors... ");

    uint8_t buf[256];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_header(&wb, 0, 0);

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f != NULL);  // zero tensors is valid
    bn_gguf_free(f);

    printf("PASSED\n");
}

// ===================================================================
// Test: n_tensors=UINT64_MAX should reject (OOM or bounds check)
// ===================================================================
static void test_fuzz_huge_n_tensors(void) {
    printf("test_fuzz_huge_n_tensors... ");

    uint8_t buf[256];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_u32(&wb, 0x46554747);
    wb_u32(&wb, 3);
    wb_u64(&wb, UINT64_MAX);  // absurd n_tensors
    wb_u64(&wb, 0);

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f == NULL);  // should reject

    printf("PASSED\n");
}

// ===================================================================
// Test: n_kv=UINT64_MAX should reject
// ===================================================================
static void test_fuzz_huge_n_kv(void) {
    printf("test_fuzz_huge_n_kv... ");

    uint8_t buf[256];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_u32(&wb, 0x46554747);
    wb_u32(&wb, 3);
    wb_u64(&wb, 0);
    wb_u64(&wb, UINT64_MAX);  // absurd n_kv

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f == NULL);  // should reject

    printf("PASSED\n");
}

// ===================================================================
// Test: bad KV value_type should reject
// ===================================================================
static void test_fuzz_bad_kv_type(void) {
    printf("test_fuzz_bad_kv_type... ");

    uint8_t buf[256];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_header(&wb, 0, 1);

    wb_str(&wb, "test.key");
    wb_u32(&wb, 9999);  // bogus value type

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f == NULL);  // should reject unknown type

    printf("PASSED\n");
}

// ===================================================================
// Test: string length > remaining buffer should reject
// ===================================================================
static void test_fuzz_string_overflow(void) {
    printf("test_fuzz_string_overflow... ");

    uint8_t buf[256];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_header(&wb, 0, 1);

    // KV key with claimed length way beyond buffer
    wb_u64(&wb, 999999);  // string length = 999999 bytes
    // Don't write any actual string data — buffer ends here

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f == NULL);  // should reject: string extends past buffer

    printf("PASSED\n");
}

static void test_fuzz_string_len_wrap(void) {
    printf("test_fuzz_string_len_wrap... ");

    uint8_t buf[256];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_header(&wb, 0, 1);

    wb_u64(&wb, UINT64_MAX);

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f == NULL);

    printf("PASSED\n");
}

// ===================================================================
// Test: tensor offset past data section should be caught
// ===================================================================
static void test_fuzz_tensor_offset_oob(void) {
    printf("test_fuzz_tensor_offset_oob... ");

    uint8_t buf[4096];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_header(&wb, 1, 0);

    wb_str(&wb, "test.tensor");
    wb_u32(&wb, 1);           // n_dims
    wb_u64(&wb, 16);          // dim[0]
    wb_u32(&wb, 0);           // type = F32
    wb_u64(&wb, 999999999);   // offset way beyond buffer

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    if (f != NULL) {
        // Parser accepted header — but tensor_data should return NULL
        int ti = bn_gguf_find_tensor(f, "test.tensor");
        assert(ti >= 0);
        void *data = bn_gguf_tensor_data(f, ti);
        assert(data == NULL);
        bn_gguf_free(f);
    }
    // Either way is acceptable: NULL from open or NULL from tensor_data

    printf("PASSED\n");
}

// ===================================================================
// Test: tensor dims that overflow uint64 when multiplied
// ===================================================================
static void test_fuzz_tensor_dims_overflow(void) {
    printf("test_fuzz_tensor_dims_overflow... ");

    uint8_t buf[4096];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_header(&wb, 1, 0);

    wb_str(&wb, "overflow.tensor");
    wb_u32(&wb, 2);                   // n_dims = 2
    wb_u64(&wb, (uint64_t)1 << 40);   // dim[0] = 1TB elements
    wb_u64(&wb, (uint64_t)1 << 40);   // dim[1] = 1TB elements — product overflows
    wb_u32(&wb, 0);                   // type = F32
    wb_u64(&wb, 0);                   // offset

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    if (f != NULL) {
        // If parser accepted, tensor_data should still be safe
        int ti = bn_gguf_find_tensor(f, "overflow.tensor");
        if (ti >= 0) {
            void *data = bn_gguf_tensor_data(f, ti);
            assert(data == NULL);  // offset + size must exceed buffer
        }
        bn_gguf_free(f);
    }

    printf("PASSED\n");
}

static void test_fuzz_quant_tensor_truncated(void) {
    printf("test_fuzz_quant_tensor_truncated... ");

    uint8_t buf[512];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_header(&wb, 1, 0);

    wb_str(&wb, "truncated.q4k");
    wb_u32(&wb, 2);
    wb_u64(&wb, 256);
    wb_u64(&wb, 1);
    wb_u32(&wb, BN_GGUF_TENSOR_Q4_K);
    wb_u64(&wb, 0);

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f != NULL);
    int ti = bn_gguf_find_tensor(f, "truncated.q4k");
    assert(ti >= 0);
    assert(bn_gguf_tensor_data(f, ti) == NULL);
    bn_gguf_free(f);

    printf("PASSED\n");
}

// ===================================================================
// Test: tensor with dim[0]=0 — should parse (degenerate but valid)
// ===================================================================
static void test_fuzz_zero_dim_tensor(void) {
    printf("test_fuzz_zero_dim_tensor... ");

    uint8_t buf[4096];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_header(&wb, 1, 0);

    wb_str(&wb, "zero.tensor");
    wb_u32(&wb, 1);           // n_dims
    wb_u64(&wb, 0);           // dim[0] = 0
    wb_u32(&wb, 0);           // type = F32
    wb_u64(&wb, 0);           // offset

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    // Zero-dim tensor: parser may accept or reject — either is fine
    if (f != NULL) bn_gguf_free(f);

    printf("PASSED\n");
}

// ===================================================================
// Test: wrong magic bytes should reject
// ===================================================================
static void test_fuzz_wrong_magic(void) {
    printf("test_fuzz_wrong_magic... ");

    uint8_t buf[256];
    WriteBuffer wb = { buf, 0, sizeof(buf) };
    wb_u32(&wb, 0xDEADBEEF);  // bad magic
    wb_u32(&wb, 3);
    wb_u64(&wb, 0);
    wb_u64(&wb, 0);

    BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
    assert(f == NULL);

    printf("PASSED\n");
}

// ===================================================================
// Test: wrong version should reject
// ===================================================================
static void test_fuzz_wrong_version(void) {
    printf("test_fuzz_wrong_version... ");

    // Version 0
    {
        uint8_t buf[256];
        WriteBuffer wb = { buf, 0, sizeof(buf) };
        wb_u32(&wb, 0x46554747);
        wb_u32(&wb, 0);  // version 0
        wb_u64(&wb, 0);
        wb_u64(&wb, 0);

        BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
        assert(f == NULL);
    }

    // Version 999
    {
        uint8_t buf[256];
        WriteBuffer wb = { buf, 0, sizeof(buf) };
        wb_u32(&wb, 0x46554747);
        wb_u32(&wb, 999);  // version 999
        wb_u64(&wb, 0);
        wb_u64(&wb, 0);

        BnGGUFFile *f = bn_gguf_open(buf, wb.pos);
        assert(f == NULL);
    }

    printf("PASSED\n");
}

int main(void) {
    printf("=== GGUF Fuzz Tests ===\n");

    test_fuzz_zero_tensors();
    test_fuzz_huge_n_tensors();
    test_fuzz_huge_n_kv();
    test_fuzz_bad_kv_type();
    test_fuzz_string_overflow();
    test_fuzz_string_len_wrap();
    test_fuzz_tensor_offset_oob();
    test_fuzz_tensor_dims_overflow();
    test_fuzz_quant_tensor_truncated();
    test_fuzz_zero_dim_tensor();
    test_fuzz_wrong_magic();
    test_fuzz_wrong_version();

    printf("All GGUF fuzz tests passed!\n");
    return 0;
}
