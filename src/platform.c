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
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#endif

BnMappedFile bn_platform_load_file(const char *path) {
    BnMappedFile f = {0};
    f.fd = -1;
#ifdef __EMSCRIPTEN__
    FILE *fp = fopen(path, "rb");
    if (!fp) return f;
    // #26: Check ftell return value for errors
    fseek(fp, 0, SEEK_END);
    long tell_result = ftell(fp);
    if (tell_result < 0) { fclose(fp); return f; }
    f.size = (size_t)tell_result;
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
    // #27: Guard against mmap with size 0 (POSIX says behavior is unspecified)
    if (f.size == 0) { close(fd); return f; }
    f.data = (uint8_t *)mmap(NULL, f.size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (f.data == MAP_FAILED) { f.data = NULL; f.size = 0; close(fd); return f; }
    f.is_mmap = 1;
    f.fd = fd;  // keep fd open for pread (MoE expert loading)
#endif
    return f;
}

// #19, #20: bn_platform_load_buffer wraps an external buffer without taking ownership.
// The is_mmap=2 flag distinguishes it from both mmap'd and malloc'd buffers,
// preventing bn_platform_unload_file from freeing memory it doesn't own.
BnMappedFile bn_platform_load_buffer(const uint8_t *buf, size_t size) {
    BnMappedFile f = {0};
    f.data = (uint8_t *)buf;  // intentional const-discard for zero-copy interface
    f.size = size;
    f.is_mmap = 2;  // 2 = externally owned, do not free
    f.fd = -1;
    return f;
}

void bn_platform_unload_file(BnMappedFile *f) {
    if (!f || !f->data) return;
#ifdef __EMSCRIPTEN__
    if (f->is_mmap != 2) {  // Don't free externally-owned buffers
        free(f->data);
    }
#else
    if (f->is_mmap == 1) {
        munmap(f->data, f->size);
    } else if (f->is_mmap == 0) {
        free(f->data);
    }
    // is_mmap == 2: externally owned, don't free
    if (f->fd >= 0) {
        close(f->fd);
        f->fd = -1;
    }
#endif
    f->data = NULL;
    f->size = 0;
}

double bn_platform_time_ms(void) {
#ifdef __EMSCRIPTEN__
    return emscripten_get_now();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

size_t bn_platform_rss_bytes(void) {
#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)
    struct mach_task_basic_info info = {0};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return info.resident_size;
    return 0;
#elif defined(__linux__) && !defined(__EMSCRIPTEN__)
    FILE *fp = fopen("/proc/self/statm", "r");
    if (!fp) return 0;
    long pages = 0;
    if (fscanf(fp, "%*ld %ld", &pages) != 1) pages = 0;
    fclose(fp);
    return (size_t)pages * 4096;
#else
    return 0;
#endif
}
