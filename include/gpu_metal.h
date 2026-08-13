#ifndef BN_GPU_METAL_H
#define BN_GPU_METAL_H

#ifdef BN_ENABLE_METAL

#include "gpu_backend.h"

// Create a Metal compute backend for bitnet.c inference.
// Enumerates the default Metal device, compiles MSL compute shaders for all
// supported quant types, and returns a BnGPUBackend ready for bn_model_upload_weights.
//
// shader_dir: path to directory containing *.metal files (NULL = "shaders/metal/").
// Returns NULL if no Metal device is available or initialization fails.
BnGPUBackend *bn_gpu_metal_create(const char *shader_dir);
BnGPUBackend *bn_gpu_metal_create_with_policy(
    const char *shader_dir, const BnBackendRuntimePolicy *policy);

// Destroy the Metal GPU backend and release all device resources.
// Safe to call with NULL.
void bn_gpu_metal_destroy(BnGPUBackend *gpu);

// Initialize Metal slab allocator for MoE weight suballocation.
// Eliminates per-expert buffer allocation overhead.
// size_mb: slab size in MB (0 = disabled). Returns 0 on success.
int bn_gpu_metal_init_slab(BnGPUBackend *gpu, size_t size_mb);
size_t bn_gpu_metal_recommended_slab_mb(const BnGPUBackend *gpu);

// Set mmap range for zero-copy weight upload (Phase 5).
// Pointers within [base, base+size) will use newBufferWithBytesNoCopy
// instead of copying data, halving peak RSS for non-repacked types.
// Must be called before buffer_create. base must be page-aligned.
void bn_gpu_metal_set_mmap_range(BnGPUBackend *gpu, const void *base, size_t size);

#endif // BN_ENABLE_METAL

#endif // BN_GPU_METAL_H
