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
/* Separate non-flash oracle: GGML CUDA mul_mat(K,Q), masked softmax,
 * mul_mat(V,probabilities), F32 accumulation and 256 padded cache positions.
 * Generated independently by reference_decode_attention_split.c against
 * llama.cpp df18bee11e96557b07aa7b12db0976f3c7ac9165. The original flash
 * reference values above are retained verbatim; the two operator pipelines
 * have distinct reduction/normalization contracts. */
static const BnCudaDecodeAttentionF32ReferenceCase
cuda_decode_attention_f32_split_reference[] = {
    {1, 0, UINT64_C(0xab67ad72a2d3ac1d)},
    {1, 23, UINT64_C(0xe50cd7e75ebbb1e9)},
    {7, 0, UINT64_C(0x3c62b8e000e9f89f)},
    {7, 23, UINT64_C(0x906623a70350e888)},
    {11, 0, UINT64_C(0xa11534533269e990)},
    {11, 23, UINT64_C(0x8ab21341a784c702)},
    {32, 0, UINT64_C(0x2ea7876c04a1680c)},
    {32, 23, UINT64_C(0x0e290919aa04089e)},
    {33, 0, UINT64_C(0xf9a70d57472262ac)},
    {33, 23, UINT64_C(0x93d8b8c9a926f8a7)},
};
#endif
