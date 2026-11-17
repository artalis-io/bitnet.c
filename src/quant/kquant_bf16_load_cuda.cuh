#ifndef BN_QUANT_KQUANT_BF16_LOAD_CUDA_CUH
#define BN_QUANT_KQUANT_BF16_LOAD_CUDA_CUH
/* Backend raw Q4_K/Q5_K blocks and prepared Q8 scratch have four-byte aligned
 * bases/strides. Read packed words without temporary eight-group scale arrays.
 * Weight data and prepared input are read-only for each consuming kernel. */
static_assert(sizeof(BnBlockQ4K) % 4 == 0 && offsetof(BnBlockQ4K, qs) % 4 == 0 &&
                  offsetof(BnBlockQ4K, scales) % 4 == 0,
              "Q4_K word alignment");
static_assert(sizeof(BnBlockQ5K) % 4 == 0 && offsetof(BnBlockQ5K, qs) % 4 == 0 &&
                  offsetof(BnBlockQ5K, qh) % 4 == 0 && offsetof(BnBlockQ5K, scales) % 4 == 0,
              "Q5_K word alignment");
static_assert(sizeof(BnCudaBlockQ8_1) % 4 == 0 && offsetof(BnCudaBlockQ8_1, qs) % 4 == 0,
              "Q8 scratch word alignment");
static __device__ void cuda_bf16_kquant_scale_min(const uint8_t *packed, int group, int *scale,
                                                  int *minv) {
    uint32_t lo = __ldg((const uint32_t *)packed), mid = __ldg((const uint32_t *)(packed + 4)),
             hi = __ldg((const uint32_t *)(packed + 8));
    int shift = (group & 3) * 8;
    lo >>= shift;
    mid >>= shift;
    hi >>= shift;
    *scale = group < 4 ? (lo & 63) : ((hi & 15) | (((lo >> 6) & 3) << 4));
    *minv = group < 4 ? (mid & 63) : (((hi >> 4) & 15) | (((mid >> 6) & 3) << 4));
}
static __device__ uint32_t cuda_bf16_affine_quant4(const BnBlockQ4K *w, int group, int offset) {
    uint32_t q = __ldg((const uint32_t *)(w->qs + (group / 2) * 32 + offset));
    return (q >> ((group & 1) * 4)) & 0x0f0f0f0fu;
}
static __device__ uint32_t cuda_bf16_affine_quant4(const BnBlockQ5K *w, int group, int offset) {
    uint32_t q = __ldg((const uint32_t *)(w->qs + (group / 2) * 32 + offset));
    uint32_t h = __ldg((const uint32_t *)(w->qh + offset));
    return ((q >> ((group & 1) * 4)) & 0x0f0f0f0fu) | (((h >> group) & 0x01010101u) << 4);
}
/* One packed word supplies both adjacent 32-value affine groups. Reuse
 * its low/high nibbles and the same high-bit plane rather than loading and
 * unpacking each group separately. p is a four-byte offset in packed qs. */
static __device__ uint2 cuda_bf16_affine_quant8(const BnBlockQ4K *w, int p) {
    uint32_t q = __ldg((const uint32_t *)(w->qs + p));
    return make_uint2(q & 0x0f0f0f0fu, (q >> 4) & 0x0f0f0f0fu);
}
static __device__ uint2 cuda_bf16_affine_quant8(const BnBlockQ5K *w, int p) {
    int g = 2 * (p / 32), off = p % 32;
    uint32_t q = __ldg((const uint32_t *)(w->qs + p));
    uint32_t h = __ldg((const uint32_t *)(w->qh + off));
    return make_uint2((q & 0x0f0f0f0fu) | (((h >> g) & 0x01010101u) << 4),
        ((q >> 4) & 0x0f0f0f0fu) | (((h >> (g + 1)) & 0x01010101u) << 4));
}
/* Q6_K rows have a two-byte stride alignment. Read two halfwords per
 * plane and reuse them for both nibble halves, producing eight values.
 * Each byte is in [0,63]; adding 96 cannot carry between bytes. The final
 * XOR converts every lane to its signed value in [-32,31]. */
static_assert(sizeof(BnBlockQ6K) % 2 == 0 && offsetof(BnBlockQ6K, ql) % 2 == 0 &&
                  offsetof(BnBlockQ6K, qh) % 2 == 0,
              "Q6_K halfword alignment");
static __device__ uint2 cuda_bf16_q6_quant8(const BnBlockQ6K *w, int p) {
    int half = p / 64, g = (p % 64) / 32, off = p % 32;
    const uint16_t *lp = (const uint16_t *)(w->ql + p);
    const uint16_t *hp = (const uint16_t *)(w->qh + half * 32 + off);
    uint32_t lo = (uint32_t)__ldg(lp) | ((uint32_t)__ldg(lp + 1) << 16);
    uint32_t hi = (uint32_t)__ldg(hp) | ((uint32_t)__ldg(hp + 1) << 16);
    uint32_t q0 = (lo & 0x0f0f0f0fu) | (((hi >> (2 * g)) & 0x03030303u) << 4);
    uint32_t q1 = ((lo >> 4) & 0x0f0f0f0fu) |
        (((hi >> (2 * (g + 2))) & 0x03030303u) << 4);
    return make_uint2((q0 + 0x60606060u) ^ 0x80808080u,
                     (q1 + 0x60606060u) ^ 0x80808080u);
}
#endif
