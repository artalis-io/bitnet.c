#ifndef BN_SAFETENSORS_H
#define BN_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#define BN_SAFETENSORS_MAX_HEADER 100000000u
#define BN_SAFETENSORS_MAX_DIMS 32
#define BN_SAFETENSORS_MAX_TENSORS 1048576u
#define BN_SAFETENSORS_MAX_SHARDS 4096u

/* Storage types only, not quant execution capabilities. Packed types require
 * byte-aligned tensor extents; no numerical conversion is performed. */
typedef enum {
    BN_SAFETENSORS_BOOL, BN_SAFETENSORS_U8, BN_SAFETENSORS_I8,
    BN_SAFETENSORS_F8_E5M2, BN_SAFETENSORS_F8_E4M3, BN_SAFETENSORS_F8_E8M0,
    BN_SAFETENSORS_F8_E4M3FNUZ, BN_SAFETENSORS_F8_E5M2FNUZ,
    BN_SAFETENSORS_I16, BN_SAFETENSORS_U16, BN_SAFETENSORS_F16,
    BN_SAFETENSORS_BF16, BN_SAFETENSORS_I32, BN_SAFETENSORS_U32,
    BN_SAFETENSORS_F32, BN_SAFETENSORS_C64, BN_SAFETENSORS_F64,
    BN_SAFETENSORS_I64, BN_SAFETENSORS_U64, BN_SAFETENSORS_F4,
    BN_SAFETENSORS_F6_E2M3, BN_SAFETENSORS_F6_E3M2,
    BN_SAFETENSORS_DTYPE_COUNT
} BnSafetensorsDtype;

typedef struct {
    const char *name;
    BnSafetensorsDtype dtype;
    int n_dims;
    uint64_t shape[BN_SAFETENSORS_MAX_DIMS]; /* File order, not GGUF order. */
    uint64_t offset, bytes; /* Relative to the payload after the JSON header. */
} BnSafetensorsTensor;

/* Caller owns a zero-initialized handle. Do not copy an initialized handle.
 * Metadata is owned and immutable; file/buffer contents must remain unchanged
 * for its lifetime. File init maps address space but reads metadata only;
 * native tensor reads use pread, avoiding payload mmap page faults. Buffer init
 * borrows storage. Rejection leaves the handle unchanged. UTF-8 is checked;
 * embedded NUL strings, ranks >32 and unknown tensor fields/types reject. */
typedef struct {void *state;} BnSafetensors;
int bn_safetensors_init_file(BnSafetensors *file, const char *path);
int bn_safetensors_init_buffer(BnSafetensors *file, const void *data, size_t size);
void bn_safetensors_free(BnSafetensors *file);
size_t bn_safetensors_count(const BnSafetensors *file);
/* Tensor indices are sorted by decoded UTF-8 name; returned metadata is borrowed. */
const BnSafetensorsTensor *bn_safetensors_tensor(const BnSafetensors *file, size_t index);
int bn_safetensors_find(const BnSafetensors *file, const char *name);
const char *bn_safetensors_metadata(const BnSafetensors *file, const char *key);
const char *bn_safetensors_dtype_name(BnSafetensorsDtype dtype);
unsigned bn_safetensors_dtype_bits(BnSafetensorsDtype dtype);
/* No allocation or conversion. Independent reads may run concurrently.
 * Invalid ranges/destinations reject before writing. I/O failure may leave a
 * partially filled destination. A zero-byte read permits a NULL destination. */
int bn_safetensors_read(const BnSafetensors *file, size_t index,
    uint64_t offset, void *destination, size_t bytes);

/* Owned multi-shard binding for an HF index JSON with metadata.total_size and
 * weight_map. Shards must be .safetensors basenames next to the index. All tensors
 * in referenced shards must occur exactly once in the map, in the declared shard;
 * total_size must match their raw payload. Unknown index/metadata fields reject.
 * No payload copy or numerical conversion. Same immutable-file, init/free,
 * read-range and threading contracts as the single-file reader above. */
typedef struct {void *state;} BnSafetensorsIndex;
int bn_safetensors_index_init(BnSafetensorsIndex *index, const char *path);
void bn_safetensors_index_free(BnSafetensorsIndex *index);
size_t bn_safetensors_index_count(const BnSafetensorsIndex *index);
size_t bn_safetensors_index_shards(const BnSafetensorsIndex *index);
uint64_t bn_safetensors_index_bytes(const BnSafetensorsIndex *index);
const BnSafetensorsTensor *bn_safetensors_index_tensor(const BnSafetensorsIndex *index,
    size_t tensor);
const char *bn_safetensors_index_shard_name(const BnSafetensorsIndex *index,
    size_t tensor);
int bn_safetensors_index_find(const BnSafetensorsIndex *index, const char *name);
int bn_safetensors_index_read(const BnSafetensorsIndex *index, size_t tensor,
    uint64_t offset, void *destination, size_t bytes);

#endif
