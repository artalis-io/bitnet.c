#include "moe_internal.h"

// --- I/O Prefetch Thread (pread pipeline) ---
#if !defined(__EMSCRIPTEN__)

struct BnMoEPrefetch {
    pthread_t thread;
    pthread_mutex_t mtx;
    pthread_cond_t req_cv;      // I/O thread waits for work
    pthread_cond_t done_cv;     // main thread waits for completion
    int fd;
    int active;                 // request in progress
    int shutdown;
    int success;                // result of last I/O
    double io_time_ms;          // accumulated pread time
    size_t io_bytes;            // accumulated bytes read
    // Request: up to 3 preads per submission (gate, up, down)
    struct { uint8_t *buf; size_t size; off_t offset; } reqs[3];
    int n_reqs;
};

static void *moe_prefetch_worker(void *arg) {
    BnMoEPrefetch *pf = (BnMoEPrefetch *)arg;
    pthread_mutex_lock(&pf->mtx);
    while (1) {
        while (!pf->active && !pf->shutdown)
            pthread_cond_wait(&pf->req_cv, &pf->mtx);
        if (pf->shutdown) break;

        // Copy request params under lock
        int n = pf->n_reqs;
        uint8_t *bufs[3]; size_t sizes[3]; off_t offsets[3];
        for (int i = 0; i < n; i++) {
            bufs[i] = pf->reqs[i].buf;
            sizes[i] = pf->reqs[i].size;
            offsets[i] = pf->reqs[i].offset;
        }
        pthread_mutex_unlock(&pf->mtx);

        // Do I/O without holding lock
        double t0 = bn_platform_time_ms();
        int ok = 1;
        size_t bytes = 0;
        for (int i = 0; i < n; i++) {
            ssize_t r = pread(pf->fd, bufs[i], sizes[i], offsets[i]);
            if (r != (ssize_t)sizes[i]) { ok = 0; break; }
            bytes += sizes[i];
        }
        double elapsed = bn_platform_time_ms() - t0;

        pthread_mutex_lock(&pf->mtx);
        pf->success = ok;
        pf->io_time_ms += elapsed;
        pf->io_bytes += bytes;
        pf->active = 0;
        pthread_cond_signal(&pf->done_cv);
    }
    pthread_mutex_unlock(&pf->mtx);
    return NULL;
}

BnMoEPrefetch *bn_moe_prefetch_init_internal(int fd) {
    BnMoEPrefetch *pf = (BnMoEPrefetch *)calloc(1, sizeof(BnMoEPrefetch));
    if (!pf) return NULL;
    pf->fd = fd;
    pthread_mutex_init(&pf->mtx, NULL);
    pthread_cond_init(&pf->req_cv, NULL);
    pthread_cond_init(&pf->done_cv, NULL);
    if (pthread_create(&pf->thread, NULL, moe_prefetch_worker, pf) != 0) {
        pthread_mutex_destroy(&pf->mtx);
        pthread_cond_destroy(&pf->req_cv);
        pthread_cond_destroy(&pf->done_cv);
        free(pf);
        return NULL;
    }
    return pf;
}

void bn_moe_prefetch_free_internal(BnMoEPrefetch *pf) {
    if (!pf) return;
    pthread_mutex_lock(&pf->mtx);
    pf->shutdown = 1;
    pthread_cond_signal(&pf->req_cv);
    pthread_mutex_unlock(&pf->mtx);
    pthread_join(pf->thread, NULL);
    pthread_mutex_destroy(&pf->mtx);
    pthread_cond_destroy(&pf->req_cv);
    pthread_cond_destroy(&pf->done_cv);
    free(pf);
}

// Post a 2-read prefetch request (gate+up). Non-blocking.
void bn_moe_prefetch_start2_internal(BnMoEPrefetch *pf,
                                uint8_t *buf1, size_t size1, off_t off1,
                                uint8_t *buf2, size_t size2, off_t off2) {
    pthread_mutex_lock(&pf->mtx);
    pf->reqs[0].buf = buf1; pf->reqs[0].size = size1; pf->reqs[0].offset = off1;
    pf->reqs[1].buf = buf2; pf->reqs[1].size = size2; pf->reqs[1].offset = off2;
    pf->n_reqs = 2;
    pf->active = 1;
    pthread_cond_signal(&pf->req_cv);
    pthread_mutex_unlock(&pf->mtx);
}

// Post a 1-read prefetch request (down). Non-blocking.
void bn_moe_prefetch_start1_internal(BnMoEPrefetch *pf,
                                uint8_t *buf1, size_t size1, off_t off1) {
    pthread_mutex_lock(&pf->mtx);
    pf->reqs[0].buf = buf1; pf->reqs[0].size = size1; pf->reqs[0].offset = off1;
    pf->n_reqs = 1;
    pf->active = 1;
    pthread_cond_signal(&pf->req_cv);
    pthread_mutex_unlock(&pf->mtx);
}

// Wait for prefetch to complete. Returns success flag.
int bn_moe_prefetch_wait_internal(BnMoEPrefetch *pf) {
    pthread_mutex_lock(&pf->mtx);
    while (pf->active)
        pthread_cond_wait(&pf->done_cv, &pf->mtx);
    int ok = pf->success;
    pthread_mutex_unlock(&pf->mtx);
    return ok;
}

void bn_moe_prefetch_collect_stats(BnMoEPrefetch *pf, BnMoEStats *stats) {
    if (!pf || !stats) return;
    pthread_mutex_lock(&pf->mtx);
    stats->io_time_ms += pf->io_time_ms;
    stats->io_bytes += pf->io_bytes;
    stats->io_count += pf->n_reqs;
    pf->io_time_ms = 0;
    pf->io_bytes = 0;
    pthread_mutex_unlock(&pf->mtx);
}

#endif // !__EMSCRIPTEN__

// Get offset and size for an expert projection.
// proj: 0=gate, 1=up, 2=down
int bn_moe_proj_info(const BnMoEExpertMap *map, int expert_idx, int proj,
                          size_t *offset, size_t *proj_bytes) {
    switch (proj) {
        case 0:
            *offset = map->gate_offset + (size_t)expert_idx *
                      (map->gate_stride ? map->gate_stride : map->expert_gate_bytes);
            *proj_bytes = map->expert_gate_bytes;
            return 0;
        case 1:
            *offset = map->up_offset + (size_t)expert_idx *
                      (map->up_stride ? map->up_stride : map->expert_up_bytes);
            *proj_bytes = map->expert_up_bytes;
            return 0;
        case 2:
            *offset = map->down_offset + (size_t)expert_idx *
                      (map->down_stride ? map->down_stride : map->expert_down_bytes);
            *proj_bytes = map->expert_down_bytes;
            return 0;
        default:
            return -1;
    }
}

// madvise helper: issue WILLNEED or DONTNEED for expert projections.
// proj_mask: bitmask (1=gate, 2=up, 4=down), advice: MADV_WILLNEED or MADV_DONTNEED.
#if !defined(__EMSCRIPTEN__)
void bn_moe_madvise_experts(const BnMoEIO *io, const BnMoEExpertMap *map,
                                 const int *indices, int n, int advice, int proj_mask) {
    if (!io->mmap_base) return;
    long page_size = sysconf(_SC_PAGESIZE);
    for (int k = 0; k < n; k++) {
        int eidx = indices[k];
        if (eidx < 0) continue;
        for (int proj = 0; proj < 3; proj++) {
            if (!((proj_mask >> proj) & 1)) continue;
            size_t offset, proj_bytes;
            bn_moe_proj_info(map, eidx, proj, &offset, &proj_bytes);
            // Page-align: round down start, round up end
            uintptr_t addr = (uintptr_t)io->mmap_base + offset;
            uintptr_t aligned_start = addr & ~((uintptr_t)page_size - 1);
            size_t aligned_len = (addr + proj_bytes - aligned_start + page_size - 1) & ~((size_t)page_size - 1);
            madvise((void *)aligned_start, aligned_len, advice);
        }
    }
}
#endif

// Load one expert projection into a specific buffer.
// Returns pointer to data (mmap pointer or buf), or NULL on error.
const void *bn_moe_load_expert_proj_into(const BnMoEIO *io, BnMoEStats *stats,
                                              const BnMoEExpertMap *map,
                                              int expert_idx, int proj,
                                              uint8_t *buf, size_t buf_size) {
    size_t offset, proj_bytes;
    if (bn_moe_proj_info(map, expert_idx, proj, &offset, &proj_bytes) < 0)
        return NULL;

    stats->io_bytes += proj_bytes;
    stats->io_count++;

    if (io->mmap_base)
        return io->mmap_base + offset;

#if !defined(__EMSCRIPTEN__)
    if (io->fd < 0 || proj_bytes > buf_size) return NULL;
    double t0 = bn_platform_time_ms();
    ssize_t n = pread(io->fd, buf, proj_bytes, (off_t)offset);
    stats->io_time_ms += bn_platform_time_ms() - t0;
    if (n != (ssize_t)proj_bytes) return NULL;
    return buf;
#else
    (void)buf;
    (void)buf_size;
    return NULL;
#endif
}

// Load one expert projection into the default expert_buf.
const void *bn_moe_load_expert_proj(const BnMoEIO *io, BnMoEState *ms,
                                         const BnMoEExpertMap *map,
                                         int expert_idx, int proj) {
    return bn_moe_load_expert_proj_into(io, &ms->stats, map, expert_idx, proj,
                                      ms->buf, ms->buf_size);
}

// Public accessor for GPU path — wraps static bn_moe_load_expert_proj.
const void *bn_moe_get_expert_proj(BnMoEIO *io, BnMoEState *ms,
                                    const BnMoEExpertMap *em,
                                    int expert_idx, int proj) {
    return bn_moe_load_expert_proj(io, ms, em, expert_idx, proj);
}

void bn_moe_prefetch_create(BnMoEIO *io) {
    if (!io || io->prefetch) return;
#if !defined(__EMSCRIPTEN__)
    if (io->fd >= 0 && !io->mmap_base) {
        io->prefetch = bn_moe_prefetch_init_internal(io->fd);
        io->prefetch_down = bn_moe_prefetch_init_internal(io->fd);
        if (io->prefetch && io->prefetch_down) {
            SH_LOG_INFO("MoE I/O prefetch threads", "status", "2 created (gate+up, down)");
        } else {
            // Clean up partial init — free whichever succeeded
            if (io->prefetch) { bn_moe_prefetch_free_internal((BnMoEPrefetch *)io->prefetch); io->prefetch = NULL; }
            if (io->prefetch_down) { bn_moe_prefetch_free_internal((BnMoEPrefetch *)io->prefetch_down); io->prefetch_down = NULL; }
            SH_LOG_WARN("MoE I/O prefetch threads failed to create");
        }
    }
#endif
}

void bn_moe_prefetch_destroy(BnMoEIO *io) {
    if (!io) return;
#if !defined(__EMSCRIPTEN__)
    if (io->prefetch) {
        bn_moe_prefetch_free_internal((BnMoEPrefetch *)io->prefetch);
        io->prefetch = NULL;
    }
    if (io->prefetch_down) {
        bn_moe_prefetch_free_internal((BnMoEPrefetch *)io->prefetch_down);
        io->prefetch_down = NULL;
    }
#endif
}
