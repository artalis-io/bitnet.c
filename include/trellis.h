#ifndef BN_TRELLIS_H
#define BN_TRELLIS_H
#include "safetensors.h"
#include <stddef.h>
#include <stdint.h>

/* Immutable CPU checkpoint binding, suitable for model anatomy. The initialized
 * index and its files must outlive this handle and remain unchanged. Caller
 * owns a zeroed handle; init rejection leaves it unchanged. Do not copy it.
 * Tensor IDs are supplied by model-family naming logic, in packed/pre/post/
 * marker order. This module has no model names, GPU handles, or tensor math. */
typedef struct { void *state; } BnTrellisSource;
typedef struct {
    int format, k, n;
    uint32_t marker;
    size_t packed_bytes, pre_offset, post_offset, bundle_bytes;
} BnTrellisLayout;
typedef struct {
    const void *packed, *pre, *post;
} BnTrellisBundle;
int bn_trellis_source_init(BnTrellisSource *source,
    const BnSafetensorsIndex *index, int format, const size_t tensor_ids[4]);
void bn_trellis_source_free(BnTrellisSource *source);
const BnTrellisLayout *bn_trellis_source_layout(const BnTrellisSource *source);
/* Raw little-endian bytes: packed | pre F16[K] | post F16[N]. No conversion or
 * allocation. Exact bundle size required. Invalid arguments reject before
 * writes; underlying I/O failure may partially fill the destination. */
int bn_trellis_source_read(const BnTrellisSource *source, void *destination,
    size_t bytes);
/* Borrowed views into a caller-owned full bundle; output unchanged on error.
 * Views expire with that allocation/lease, independently of the source. */
int bn_trellis_bundle(const BnTrellisSource *source, const void *data,
    size_t bytes, BnTrellisBundle *out);

/* Immutable row-scaled tensor binding. Same borrowed-index lifetime as the
 * projection source; independent reads may run concurrently. Explicit IDs
 * name packed U8 rows and separate F16 scales, never model-family names. */
typedef struct BnTrellisRowSource { void *state; } BnTrellisRowSource;
typedef struct BnTrellisRowLayout {
    int format, cols;
    uint64_t rows;
    size_t weight_bytes, scale_bytes, bundle_bytes; /* Per row. */
} BnTrellisRowLayout;
int bn_trellis_row_source_init(BnTrellisRowSource *source,
    const BnSafetensorsIndex *index, int format, const size_t tensor_ids[2]);
void bn_trellis_row_source_free(BnTrellisRowSource *source);
const BnTrellisRowLayout *bn_trellis_row_source_layout(const BnTrellisRowSource *source);
/* One row: packed bytes followed by raw little-endian scales. Exact bundle
 * size required. No allocation/conversion. Invalid arguments reject before
 * writes; I/O failure may partially fill the destination. */
int bn_trellis_row_source_read(const BnTrellisRowSource *source,
    uint64_t row, void *destination, size_t bytes);
#endif
