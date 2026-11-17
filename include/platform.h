#ifndef BN_PLATFORM_H
#define BN_PLATFORM_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint8_t *data;
    size_t   size;
    int      is_mmap;   // 1 if mmap'd, 0 if malloc'd, 2 if externally owned
    int      fd;        // file descriptor (kept open for pread on mmap platforms, -1 otherwise)
} BnMappedFile;

BnMappedFile bn_platform_load_file(const char *path);
BnMappedFile bn_platform_load_file_resident(const char *path);
BnMappedFile bn_platform_load_buffer(const uint8_t *buf, size_t size);
void         bn_platform_unload_file(BnMappedFile *f);
double       bn_platform_time_ms(void);
size_t       bn_platform_rss_bytes(void);  // current resident set size, 0 if unavailable
void        *bn_platform_aligned_alloc(size_t alignment, size_t size);
void         bn_platform_aligned_free(void *ptr);
/* Best-effort huge-page hint for full OS pages inside a caller-owned span.
 * Does not allocate, touch or discard data, change ownership or require a
 * different free function. Unsupported platforms/kernel refusals are no-ops.
 * The allocation's entire size must already be included in its memory budget.
 */
void         bn_platform_advise_huge_pages(void *ptr, size_t size);
char *const *bn_platform_environment(void);

#endif // BN_PLATFORM_H
