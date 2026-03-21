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
BnMappedFile bn_platform_load_buffer(const uint8_t *buf, size_t size);
void         bn_platform_unload_file(BnMappedFile *f);
double       bn_platform_time_ms(void);
size_t       bn_platform_rss_bytes(void);  // current resident set size, 0 if unavailable

#endif // BN_PLATFORM_H
