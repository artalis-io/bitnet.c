#include "platform.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

// --- Test: Build a minimal GGUF buffer in memory and parse it ---

// Helper to write GGUF bytes into a buffer
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

// Build a test GGUF with 2 KV pairs and 1 tensor
static size_t build_test_gguf(uint8_t *buf, size_t buf_size) {
    WriteBuffer wb;
    wb.data = buf;
    wb.pos = 0;
    wb.cap = buf_size;

    // Header
    wb_u32(&wb, 0x46554747);  // magic "GGUF"
    wb_u32(&wb, 3);           // version
    wb_u64(&wb, 1);           // n_tensors
    wb_u64(&wb, 2);           // n_kv

    // KV 1: string key → u32 value
    wb_str(&wb, "test.dim");
    wb_u32(&wb, 4);           // type = BN_GGUF_TYPE_UINT32
    wb_u32(&wb, 512);         // value

    // KV 2: string key → string value
    wb_str(&wb, "general.architecture");
    wb_u32(&wb, 8);           // type = BN_GGUF_TYPE_STRING
    wb_str(&wb, "bitnet");

    // Tensor info
    wb_str(&wb, "test.weight");
    wb_u32(&wb, 2);           // n_dims
    wb_u64(&wb, 2);           // dim[0]
    wb_u64(&wb, 2);           // dim[1]
    wb_u32(&wb, 0);           // type = F32
    wb_u64(&wb, 0);           // offset (relative to data section)

    // Pad to alignment (32 bytes)
    size_t header_end = wb.pos;
    size_t data_start = header_end + (32 - (header_end % 32)) % 32;
    while (wb.pos < data_start) {
        uint8_t zero = 0;
        wb_write(&wb, &zero, 1);
    }

    // Tensor data: write a few F32 values
    float test_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    wb_write(&wb, test_vals, sizeof(test_vals));

    return wb.pos;
}

static size_t build_one_tensor_gguf(uint8_t *buf, size_t buf_size,
                                    const char *tensor_name,
                                    const float vals[4]) {
    WriteBuffer wb;
    wb.data = buf;
    wb.pos = 0;
    wb.cap = buf_size;

    wb_u32(&wb, 0x46554747);
    wb_u32(&wb, 3);
    wb_u64(&wb, 1);
    wb_u64(&wb, 1);

    wb_str(&wb, "general.architecture");
    wb_u32(&wb, BN_GGUF_TYPE_STRING);
    wb_str(&wb, "bitnet");

    wb_str(&wb, tensor_name);
    wb_u32(&wb, 2);
    wb_u64(&wb, 2);
    wb_u64(&wb, 2);
    wb_u32(&wb, BN_GGUF_TENSOR_F32);
    wb_u64(&wb, 0);

    size_t data_start = wb.pos + (32 - (wb.pos % 32)) % 32;
    while (wb.pos < data_start) {
        uint8_t zero = 0;
        wb_write(&wb, &zero, 1);
    }
    wb_write(&wb, vals, 4 * sizeof(float));
    return wb.pos;
}

static void test_parse_synthetic(void) {
    printf("test_parse_synthetic... ");

    uint8_t buf[4096];
    size_t size = build_test_gguf(buf, sizeof(buf));

    BnGGUFFile *f = bn_gguf_open(buf, size);
    assert(f != NULL);
    assert(f->version == 3);
    assert(f->n_tensors == 1);
    assert(f->n_kv == 2);

    // Check KV values
    assert(bn_gguf_get_u32(f, "test.dim") == 512);

    const char *arch = bn_gguf_get_str(f, "general.architecture");
    assert(arch != NULL);
    assert(strcmp(arch, "bitnet") == 0);

    // Check tensor info
    int ti = bn_gguf_find_tensor(f, "test.weight");
    assert(ti >= 0);
    assert(f->tensors[ti].n_dims == 2);
    assert(f->tensors[ti].dims[0] == 2);
    assert(f->tensors[ti].dims[1] == 2);
    assert(f->tensors[ti].type == 0);

    // Check tensor data
    float *data = (float *)bn_gguf_tensor_data(f, ti);
    assert(data != NULL);
    assert(data[0] == 1.0f);
    assert(data[1] == 2.0f);
    assert(data[2] == 3.0f);
    assert(data[3] == 4.0f);

    // Check missing key
    assert(bn_gguf_find_key(f, "nonexistent") == -1);
    assert(bn_gguf_find_tensor(f, "nonexistent") == -1);

    bn_gguf_free(f);
    printf("PASSED\n");
}

static void test_bad_magic(void) {
    printf("test_bad_magic... ");

    uint8_t buf[32] = {0};
    buf[0] = 'B'; buf[1] = 'A'; buf[2] = 'D'; buf[3] = '!';

    BnGGUFFile *f = bn_gguf_open(buf, sizeof(buf));
    assert(f == NULL);

    printf("PASSED\n");
}

static void test_find_key(void) {
    printf("test_find_key... ");

    uint8_t buf[4096];
    size_t size = build_test_gguf(buf, sizeof(buf));
    BnGGUFFile *f = bn_gguf_open(buf, size);
    assert(f != NULL);

    assert(bn_gguf_find_key(f, "test.dim") == 0);
    assert(bn_gguf_find_key(f, "general.architecture") == 1);
    assert(bn_gguf_find_key(f, "missing") == -1);

    bn_gguf_free(f);
    printf("PASSED\n");
}

static void write_file(const char *path, const uint8_t *buf, size_t size) {
    FILE *fp = fopen(path, "wb");
    assert(fp != NULL);
    assert(fwrite(buf, 1, size, fp) == size);
    assert(fclose(fp) == 0);
}

static void test_open_shards(void) {
    printf("test_open_shards... ");

    char path1[] = "/tmp/bitnet-gguf-test-00001-of-00002.gguf";
    char path2[] = "/tmp/bitnet-gguf-test-00002-of-00002.gguf";
    uint8_t buf1[4096], buf2[4096];
    float vals1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vals2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    size_t size1 = build_one_tensor_gguf(buf1, sizeof(buf1), "shard0.weight", vals1);
    size_t size2 = build_one_tensor_gguf(buf2, sizeof(buf2), "shard1.weight", vals2);
    write_file(path1, buf1, size1);
    write_file(path2, buf2, size2);

    BnGGUFFile *f = bn_gguf_open_file(path1);
    assert(f != NULL);
    assert(f->n_shards == 2);
    assert(f->n_tensors == 2);
    assert(strcmp(bn_gguf_get_str(f, "general.architecture"), "bitnet") == 0);

    int t0 = bn_gguf_find_tensor(f, "shard0.weight");
    int t1 = bn_gguf_find_tensor(f, "shard1.weight");
    assert(t0 >= 0);
    assert(t1 >= 0);
    float *data0 = (float *)bn_gguf_tensor_data(f, t0);
    float *data1 = (float *)bn_gguf_tensor_data(f, t1);
    assert(data0 != NULL);
    assert(data1 != NULL);
    assert(data0[0] == 1.0f && data0[3] == 4.0f);
    assert(data1[0] == 5.0f && data1[3] == 8.0f);

    bn_gguf_free(f);

    f = bn_gguf_open_file_resident(path1);
    assert(f != NULL);
    assert(f->n_shards == 2);
    assert(f->owned_maps != NULL);
    assert(f->owned_maps[0].is_mmap == 0);
    assert(f->owned_maps[1].is_mmap == 0);
    long page_value = sysconf(_SC_PAGESIZE);
    size_t page = page_value > 0 ? (size_t)page_value : 4096u;
    assert((uintptr_t)f->owned_maps[0].data % page == 0);
    assert((uintptr_t)f->owned_maps[1].data % page == 0);
    t0 = bn_gguf_find_tensor(f, "shard0.weight");
    t1 = bn_gguf_find_tensor(f, "shard1.weight");
    assert(t0 >= 0 && t1 >= 0);
    data0 = (float *)bn_gguf_tensor_data(f, t0);
    data1 = (float *)bn_gguf_tensor_data(f, t1);
    assert(data0[0] == 1.0f && data0[3] == 4.0f);
    assert(data1[0] == 5.0f && data1[3] == 8.0f);
    bn_gguf_free(f);

    unlink(path1);
    unlink(path2);
    printf("PASSED\n");
}

int main(void) {
    printf("=== GGUF Tests ===\n");
    test_parse_synthetic();
    test_bad_magic();
    test_find_key();
    test_open_shards();
    printf("All GGUF tests passed!\n");
    return 0;
}
