#ifndef BN_MOE_TYPES_H
#define BN_MOE_TYPES_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t gate_offset, up_offset, down_offset;
    size_t expert_gate_bytes, expert_up_bytes, expert_down_bytes;
    size_t gate_stride, up_stride, down_stride;
    int gate_type, up_type, down_type;
    int gate_rows, gate_cols;
    int up_rows, up_cols;
    int down_rows, down_cols;
    // Repacked: contiguous [gate|up|down] per expert for cache locality.
    // NULL if not repacked (pread mode or insufficient memory).
    uint8_t *repacked;          // [n_experts * expert_total_bytes]
    size_t expert_total_bytes;  // gate_bytes + up_bytes + down_bytes
} BnMoEExpertMap;

#define BN_MAX_MOE_K 16

// Shared MoE I/O control plane (lives on BnModel, shared across sessions)
typedef struct {
    int fd;
    const uint8_t *mmap_base; // mmap'd file base pointer (NULL if using pread)
    int madvise_mode;         // 1 = madvise-guided mmap (WILLNEED/DONTNEED)
    void *prefetch;           // BnMoEPrefetch* for gate+up (opaque, pread only)
    void *prefetch_down;      // BnMoEPrefetch* for down proj (opaque, pread only)
    void *cache;              // BnMoECache* for expert LRU cache (opaque, pread only)
    void *gpu_moe_cache;      // BnGPUMoECache* for GPU expert buffer cache (opaque)
} BnMoEIO;

// Accumulated MoE timing and I/O stats
typedef struct {
    size_t io_bytes;          // total bytes loaded from disk (pread) or touched (mmap)
    double io_time_ms;        // total time spent in expert loading (pread only)
    double route_time_ms;     // total time in routing (router matvec + top-K)
    double compute_time_ms;   // total time in expert FFN compute (all phases)
    double gate_up_time_ms;   // gate+up matvec time
    double swiglu_time_ms;    // SwiGLU activation time
    double down_time_ms;      // down projection matvec time
    double accum_time_ms;     // weighted accumulation time
    double shared_time_ms;    // shared expert time
    double norm_time_ms;      // RMSNorm time
    double prefetch_wait_ms;  // time main thread waited for I/O prefetch
    double madvise_time_ms;   // time spent in madvise calls
    int    io_count;          // number of expert projections loaded
    size_t cache_hits;        // expert cache hits (pread only)
    size_t cache_misses;      // expert cache misses (pread only)
} BnMoEStats;

// MoE per-session state (compute buffers + pread staging + stats)
typedef struct {
    BnMoEStats stats;         // accumulated timing stats
    // Compute buffers (arena-allocated)
    float *router_logits;
    float *expert_out;
    float *expert_weights;
    int   *expert_indices;
    float *expert_hb;
    float *expert_hb2;
    // Batch buffers for cross-expert dispatch (mmap path)
    float *expert_hb_batch[BN_MAX_MOE_K];   // K gate outputs [moe_hidden]
    float *expert_hb2_batch[BN_MAX_MOE_K];  // K up outputs [moe_hidden]
    float *expert_down_batch[BN_MAX_MOE_K]; // K down outputs [dim]
    int8_t *down_x_q_bufs;                  // [K * moe_hidden] int8 scratch for multi-dispatch down
    // Pread staging buffers (arena-allocated, per-session)
    uint8_t *buf;             // gate buffer
    size_t buf_size;
    uint8_t *buf2;            // up buffer / double-buffer
    size_t buf2_size;
    uint8_t *buf3;            // prefetch gate buffer
    size_t buf3_size;
    uint8_t *buf4;            // prefetch up buffer
    size_t buf4_size;
    uint8_t *buf5;            // down buffer
    size_t buf5_size;
} BnMoEState;

#endif // BN_MOE_TYPES_H
