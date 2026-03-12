#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint8_t *data;
    size_t   size;
    int      is_mmap;   // 1 if mmap'd, 0 if malloc'd
} MappedFile;

MappedFile platform_load_file(const char *path);
MappedFile platform_load_buffer(const uint8_t *buf, size_t size);
void       platform_unload_file(MappedFile *f);
double     platform_time_ms(void);

#endif // PLATFORM_H
