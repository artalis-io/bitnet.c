#ifndef BN_TEST_CUDA_DECODE_ATTENTION_F32_REFERENCE_H
#define BN_TEST_CUDA_DECODE_ATTENTION_F32_REFERENCE_H
#include <stdint.h>
/* Independent ggml_flash_attn_ext CUDA graphs, llama.cpp commit
 * 3d3d7c81813067fc8c185da017e0af03b4269b1e, SM120. F32 Q/K/V,
 * head256, GQA6, four KV heads, F32 accumulation and padded causal mask. */
typedef struct { int valid, seed; uint64_t hash; }
    BnCudaDecodeAttentionF32ReferenceCase;
static const BnCudaDecodeAttentionF32ReferenceCase
cuda_decode_attention_f32_reference[] = {
    {1,0,UINT64_C(0x3332e91582e57ecd)}, {1,23,UINT64_C(0xc1944cc545c3b9f6)},
    {7,0,UINT64_C(0xaa9183771ed6e07c)}, {7,23,UINT64_C(0x27381e0fc655bd2e)},
    {11,0,UINT64_C(0xf72846b6a9c76464)}, {11,23,UINT64_C(0xf0dc38b4618190c2)},
    {32,0,UINT64_C(0xf3743656b1f4ebba)}, {32,23,UINT64_C(0x07d06260eb507abb)},
    {33,0,UINT64_C(0x2985e1c5c3740913)}, {33,23,UINT64_C(0x3fb82e1870187d41)},
};
#endif
