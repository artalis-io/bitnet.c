#ifndef BN_CUDA_ROUTED_REFERENCE_H
#define BN_CUDA_ROUTED_REFERENCE_H
#include <stdint.h>
typedef struct { int dim, hidden, experts, nt, seed; uint64_t hash; } BnCudaRoutedReference;
#endif
