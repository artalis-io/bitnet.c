#ifndef BN_GPU_WGPU_H
#define BN_GPU_WGPU_H

#ifdef BN_ENABLE_GPU

#include "gpu_backend.h"

// Create a wgpu-native GPU backend for bitnet.c inference.
// Enumerates GPU devices, compiles WGSL compute shaders for all supported
// quant types, and returns a BnGPUBackend ready for bn_model_upload_weights.
//
// shader_dir: path to directory containing *.wgsl files (NULL = use embedded shaders).
// Returns NULL if no GPU is available or initialization fails.
BnGPUBackend *bn_gpu_wgpu_create(const char *shader_dir);

// Destroy the wgpu-native GPU backend and release all device resources.
// Safe to call with NULL.
void bn_gpu_wgpu_destroy(BnGPUBackend *gpu);

// Initialize GPU slab allocator for MoE weight suballocation.
// Eliminates per-expert wgpuDeviceCreateBuffer overhead.
// size_mb: slab size in MB (0 = disabled). Returns 0 on success.
int bn_gpu_wgpu_init_slab(BnGPUBackend *gpu, size_t size_mb);

#endif // BN_ENABLE_GPU

#endif // BN_GPU_WGPU_H
