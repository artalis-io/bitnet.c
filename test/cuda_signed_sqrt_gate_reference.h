#ifndef BN_CUDA_SIGNED_SQRT_GATE_REFERENCE_H
#define BN_CUDA_SIGNED_SQRT_GATE_REFERENCE_H
#include <stdint.h>
/* Original llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e CUDA, SM120.
 * Qwen3.8 signed-root gate, dim=2560, streams=4. */
typedef struct {
    int tokens, pattern, reduction_threads;
    uint64_t gate_hash, gated_hash;
} BnCudaSignedSqrtGateReference;
static const BnCudaSignedSqrtGateReference cuda_signed_sqrt_gate_reference[] = {
    {1,0,512,UINT64_C(0x169d8f851dce13f5),UINT64_C(0xefdfd45046751c85)},
    {1,1,512,UINT64_C(0xc1688836dfa29225),UINT64_C(0xe8a44a78102a1f15)},
    {1,2,512,UINT64_C(0x8b4b0b7e17f0009e),UINT64_C(0x2a829f3dee16cd3b)},
    {1,3,512,UINT64_C(0xf7be6b8616d58761),UINT64_C(0x51c6531cab588cb0)},
    {8,0,512,UINT64_C(0x45a7d6ce1d8af5a5),UINT64_C(0x536b23f22687730d)},
    {8,1,512,UINT64_C(0x7d563e6072359525),UINT64_C(0xeadb7806966a59c9)},
    {8,2,512,UINT64_C(0x21c9e1cbab3e97d3),UINT64_C(0xec53ca647f2f88ff)},
    {8,3,512,UINT64_C(0x0d5efef798f93301),UINT64_C(0x1207e4293253d800)},
    {317,0,128,UINT64_C(0x3d674295523f92b5),UINT64_C(0x045d85de4e228d3d)},
    {317,1,128,UINT64_C(0xc418bbcd9df1dcfb),UINT64_C(0x9ece972a78a51fc2)},
    {317,2,128,UINT64_C(0x926e2438cc7dae66),UINT64_C(0x1ec8b27d43e5219e)},
    {317,3,128,UINT64_C(0xb7fe7ab7a58dc161),UINT64_C(0x15888409da02d5c7)},
};
#endif
