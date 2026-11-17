#ifndef BN_CUDA_Q51_MATVEC_REFERENCE_H
#define BN_CUDA_Q51_MATVEC_REFERENCE_H
#include <stdint.h>
/* Independent GGML df18bee11 CUDA, reference_cuda_q51_matvec.c.
 * Quantized-activation single-column projection contract. Unflagged legacy
 * floating-input graph matvec is intentionally a different contract. */
typedef struct {int rows,cols,seed;uint64_t hash,biased_hash;} BnCudaQ51MatvecReference;
static const BnCudaQ51MatvecReference cuda_q51_matvec_reference[]={
{37,64,0,UINT64_C(0x473bcb22aabbd6a7),UINT64_C(0x1acbffb9ef34010f)},
{37,64,23,UINT64_C(0xc9a73577e1fcfb47),UINT64_C(0x8c49e020754bf1a7)},
{256,512,0,UINT64_C(0xcb782543fd0eef35),UINT64_C(0x02f0a784755a9de9)},
{256,512,23,UINT64_C(0x8d3516008358773f),UINT64_C(0xc8587c5598e015cf)},
{2560,640,0,UINT64_C(0x548e8f5abd66384b),UINT64_C(0xfbb1c5bac96e93f7)},
{2560,640,23,UINT64_C(0x1b356a08fb2d3069),UINT64_C(0xba44b21a287e301b)},
{2560,2560,0,UINT64_C(0x871605601500cac9),UINT64_C(0x2d9b0cdbefda86c9)},
{2560,2560,23,UINT64_C(0x49980168328a4249),UINT64_C(0xb889f13468fc8761)},
};
#endif
