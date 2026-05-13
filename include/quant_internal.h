#ifndef BN_QUANT_INTERNAL_H
#define BN_QUANT_INTERNAL_H

// Compatibility aggregate for older internal users. New code should include the
// narrower quant_ctx/quant_kernels/kquant helper headers directly.

#include "quant_ctx.h"
#include "quant_dispatch_internal.h"
#include "quant_kernels_scalar.h"
#include "quant_kernels_neon.h"
#include "quant_kernels_avx2.h"
#include "quant_kernels_wasm.h"
#include "kquant_helpers.h"

#endif // BN_QUANT_INTERNAL_H
