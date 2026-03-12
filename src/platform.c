#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#endif

MappedFile platform_load_file(const char *path) {
    MappedFile f = {0};
#ifdef __EMSCRIPTEN__
    FILE *fp = fopen(path, "rb");
    if (!fp) return f;
    fseek(fp, 0, SEEK_END);
    f.size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    f.data = (uint8_t *)malloc(f.size);
    if (!f.data) { fclose(fp); f.size = 0; return f; }
    if (fread(f.data, 1, f.size, fp) != f.size) {
        free(f.data);
        f.data = NULL;
        f.size = 0;
    }
    fclose(fp);
    f.is_mmap = 0;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return f;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return f; }
    f.size = (size_t)st.st_size;
    f.data = (uint8_t *)mmap(NULL, f.size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (f.data == MAP_FAILED) { f.data = NULL; f.size = 0; return f; }
    f.is_mmap = 1;
#endif
    return f;
}

MappedFile platform_load_buffer(const uint8_t *buf, size_t size) {
    MappedFile f = {0};
    f.data = (uint8_t *)buf;  // wrap external buffer, no copy
    f.size = size;
    f.is_mmap = 0;
    return f;
}

void platform_unload_file(MappedFile *f) {
    if (!f || !f->data) return;
#ifdef __EMSCRIPTEN__
    free(f->data);
#else
    if (f->is_mmap) {
        munmap(f->data, f->size);
    } else {
        // External buffer — don't free
    }
#endif
    f->data = NULL;
    f->size = 0;
}

double platform_time_ms(void) {
#ifdef __EMSCRIPTEN__
    return emscripten_get_now();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}
