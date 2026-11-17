/* Original pinned GGUF CUDA Q8 embedding; NaNs canonicalized for hashing. */
#ifndef BN_CUDA_BF16_Q8_EMBEDDING_REFERENCE_H
#define BN_CUDA_BF16_Q8_EMBEDDING_REFERENCE_H
typedef struct { int tokens; uint64_t hash; } BnQ8EmbeddingReference;
static const BnQ8EmbeddingReference cuda_q8_embedding_cases[] = {
 {1,UINT64_C(0x1d5ce5b38ba54b25)},
 {5,UINT64_C(0xa62359e7a35b5cac)},
 {17,UINT64_C(0x947090b2d6c34a6f)},
 {129,UINT64_C(0x165f8e93d8a7a52d)},
 {65537,UINT64_C(0x98f3bb647f131825)},
};
#endif
