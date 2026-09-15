#include "quant_ctx.h"
#include "quant_kernels_avx2.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <string.h>

static inline __m256i q6k_scale_pair(int8_t lo, int8_t hi) {
    return _mm256_set_epi16(hi, hi, hi, hi, hi, hi, hi, hi,
                            lo, lo, lo, lo, lo, lo, lo, lo);
}

void bn_quant_q6k_avx2_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const __m256i mask_lo4 = _mm256_set1_epi8(0xF);
    const __m256i mask_03  = _mm256_set1_epi8(0x03);
    const __m256i mask_0c  = _mm256_set1_epi8(0x0C);
    const __m256i mask_30  = _mm256_set1_epi8(0x30);
    const __m256i mask_c0  = _mm256_set1_epi8((char)0xC0);

    for (int row = row_start; row < row_end; row++) {
        float row_sum = 0.0f;
        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ6K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d  = bn_fp16_to_fp32(blk->d);
            float dx = x_d[b];
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)bsums);
            const __m128i sc128 = _mm_loadu_si128((const __m128i *)sc);
            const __m256i sc16 = _mm256_cvtepi8_epi16(sc128);
            const __m256i offset = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, sc16), 5);

            __m256i sumi = _mm256_setzero_si256();

            for (int chunk = 0; chunk < 2; chunk++) {
                __m256i ql0 = _mm256_loadu_si256((const __m256i *)ql);
                __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
                __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh);

                __m256i w0 = _mm256_or_si256(
                    _mm256_and_si256(ql0, mask_lo4),
                    _mm256_slli_epi16(_mm256_and_si256(qh0, mask_03), 4));
                __m256i w1 = _mm256_or_si256(
                    _mm256_and_si256(ql1, mask_lo4),
                    _mm256_slli_epi16(_mm256_and_si256(qh0, mask_0c), 2));
                __m256i w2 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_lo4),
                    _mm256_and_si256(qh0, mask_30));
                __m256i w3 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_lo4),
                    _mm256_srli_epi16(_mm256_and_si256(qh0, mask_c0), 2));

                int base = chunk * 4;
                __m256i xv0 = _mm256_loadu_si256((const __m256i *)(xb + base * 32));
                __m256i xv1 = _mm256_loadu_si256((const __m256i *)(xb + base * 32 + 32));
                __m256i xv2 = _mm256_loadu_si256((const __m256i *)(xb + base * 32 + 64));
                __m256i xv3 = _mm256_loadu_si256((const __m256i *)(xb + base * 32 + 96));

                __m256i p0 = _mm256_maddubs_epi16(w0, xv0);
                __m256i p1 = _mm256_maddubs_epi16(w1, xv1);
                __m256i p2 = _mm256_maddubs_epi16(w2, xv2);
                __m256i p3 = _mm256_maddubs_epi16(w3, xv3);

                const int8_t *sc_chunk = sc + chunk * 8;
                p0 = _mm256_madd_epi16(q6k_scale_pair(sc_chunk[0], sc_chunk[1]), p0);
                p1 = _mm256_madd_epi16(q6k_scale_pair(sc_chunk[2], sc_chunk[3]), p1);
                p2 = _mm256_madd_epi16(q6k_scale_pair(sc_chunk[4], sc_chunk[5]), p2);
                p3 = _mm256_madd_epi16(q6k_scale_pair(sc_chunk[6], sc_chunk[7]), p3);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p2, p3));

                ql += 64; qh += 32;
            }

            sumi = _mm256_sub_epi32(sumi, offset);
            row_sum += d * dx * (float)bn_avx2_hsum_epi32(sumi);
        }
        c->out[row] = row_sum;
    }
}

#define Q6K_TILE_T 8

void bn_quant_q6k_avx2_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ6K *blocks = (const BnBlockQ6K *)c->W->data;

    const __m256i mask_lo4 = _mm256_set1_epi8(0xF);
    const __m256i mask_03  = _mm256_set1_epi8(0x03);
    const __m256i mask_0c  = _mm256_set1_epi8(0x0C);
    const __m256i mask_30  = _mm256_set1_epi8(0x30);
    const __m256i mask_c0  = _mm256_set1_epi8((char)0xC0);

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < n_tokens; t0 += Q6K_TILE_T) {
            int tile_n = t0 + Q6K_TILE_T <= n_tokens ? Q6K_TILE_T : n_tokens - t0;
            float acc[Q6K_TILE_T] = {0};

            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ6K *blk = &blocks[(size_t)row * n_bpr + b];
                float d = bn_fp16_to_fp32(blk->d);

                __m256i W_all[8];
                {
                    const uint8_t *ql = blk->ql;
                    const uint8_t *qh = blk->qh;
                    for (int chunk = 0; chunk < 2; chunk++) {
                        __m256i ql0 = _mm256_loadu_si256((const __m256i *)ql);
                        __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
                        __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh);
                        int base = chunk * 4;

                        W_all[base+0] = _mm256_or_si256(
                            _mm256_and_si256(ql0, mask_lo4),
                            _mm256_slli_epi16(_mm256_and_si256(qh0, mask_03), 4));
                        W_all[base+1] = _mm256_or_si256(
                            _mm256_and_si256(ql1, mask_lo4),
                            _mm256_slli_epi16(_mm256_and_si256(qh0, mask_0c), 2));
                        W_all[base+2] = _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_lo4),
                            _mm256_and_si256(qh0, mask_30));
                        W_all[base+3] = _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_lo4),
                            _mm256_srli_epi16(_mm256_and_si256(qh0, mask_c0), 2));

                        ql += 64; qh += 32;
                    }
                }

                const int8_t *sc_base = blk->scales;
                __m256i sp[8];
                for (int chunk = 0; chunk < 2; chunk++) {
                    const int8_t *sc = sc_base + chunk * 8;
                    int base = chunk * 4;
                    sp[base+0] = q6k_scale_pair(sc[0], sc[1]);
                    sp[base+1] = q6k_scale_pair(sc[2], sc[3]);
                    sp[base+2] = q6k_scale_pair(sc[4], sc[5]);
                    sp[base+3] = q6k_scale_pair(sc[6], sc[7]);
                }

                const __m128i sc128 = _mm_loadu_si128((const __m128i *)sc_base);
                const __m256i sc16 = _mm256_cvtepi8_epi16(sc128);

                for (int ti = 0; ti < tile_n; ti++) {
                    int t = t0 + ti;
                    const int8_t *xb = c->x_q + (size_t)t * cols + b * BN_QK_K;
                    float dx = c->x_d[(size_t)t * n_bpr + b];
                    const int16_t *bsums = c->x_bsums + ((size_t)t * n_bpr + b) * 16;

                    const __m256i q8sums = _mm256_loadu_si256((const __m256i *)bsums);
                    const __m256i offset = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, sc16), 5);

                    __m256i sumi = _mm256_setzero_si256();
                    for (int chunk = 0; chunk < 2; chunk++) {
                        int base = chunk * 4;
                        const int8_t *xbc = xb + chunk * 128;
                        __m256i p0 = _mm256_maddubs_epi16(W_all[base+0], _mm256_loadu_si256((const __m256i *)xbc));
                        __m256i p1 = _mm256_maddubs_epi16(W_all[base+1], _mm256_loadu_si256((const __m256i *)(xbc + 32)));
                        __m256i p2 = _mm256_maddubs_epi16(W_all[base+2], _mm256_loadu_si256((const __m256i *)(xbc + 64)));
                        __m256i p3 = _mm256_maddubs_epi16(W_all[base+3], _mm256_loadu_si256((const __m256i *)(xbc + 96)));
                        p0 = _mm256_madd_epi16(sp[base+0], p0);
                        p1 = _mm256_madd_epi16(sp[base+1], p1);
                        p2 = _mm256_madd_epi16(sp[base+2], p2);
                        p3 = _mm256_madd_epi16(sp[base+3], p3);
                        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
                        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p2, p3));
                    }
                    sumi = _mm256_sub_epi32(sumi, offset);
                    acc[ti] += d * dx * (float)bn_avx2_hsum_epi32(sumi);
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] = acc[ti];
        }
    }
}

void bn_quant_q6k_x8_gemm_range(void *ctx, int group_start, int group_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    if (!c->prepared || c->prepared->kind != BN_PREPARED_WEIGHT_Q6_K_X8 ||
        !c->prepared->aux || !c->x_q8k_x4 || (c->n_tokens % 4) != 0) {
        bn_quant_q6k_avx2_sdot_matmul_4row_range(
            ctx, group_start * 2, group_end * 2);
        return;
    }

    const int nb = c->cols / BN_QK_K;
    const int rows = c->W->rows;
    const BnBlockQ6Kx8 *weights =
        (const BnBlockQ6Kx8 *)c->prepared->aux;
    const BnBlockQ8Kx4 *inputs = c->x_q8k_x4;
    enum { block_len = 8, blocks_per_half = 64 / block_len };

    for (int panel = 0; panel < c->n_tokens / 4; panel++) {
        const BnBlockQ8Kx4 *a = inputs + (size_t)panel * nb;
        for (int group = group_start; group < group_end; group++) {
            const BnBlockQ6Kx8 *w = weights + (size_t)group * nb;
            float sum[4][8] = {{0}};
            for (int b = 0; b < nb; b++) {
                for (int k = 0; k < BN_QK_K / (2 * block_len); k++) {
                    int base_l = (k / blocks_per_half) * 128 +
                                 (k % blocks_per_half) * block_len;
                    int base_h = base_l + 64;
                    int scale_l = base_l / 16;
                    int scale_h = base_h / 16;
                    int shift_l = ((base_l % 128) / 32) * 2;
                    int shift_h = ((base_h % 128) / 32) * 2;
                    int half_l = (base_l / 128) * 32;
                    int half_h = (base_h / 128) * 32;
                    int q8_base = (k / blocks_per_half) * 512 +
                                  (k % blocks_per_half) * (block_len * 4);

                    for (int t = 0; t < 4; t++) {
                        for (int r = 0; r < 8; r++) {
                            int dot_l = 0;
                            int dot_h = 0;
                            for (int i = 0; i < block_len; i++) {
                                int ql_pos = k * 8 * block_len +
                                             r * block_len + i;
                                int qh_idx_l = half_l + ((base_l + i) % 32);
                                int qh_idx_h = half_h + ((base_h + i) % 32);
                                int qh_off_l = (qh_idx_l / block_len) *
                                    (block_len * 8) + r * block_len +
                                    qh_idx_l % block_len;
                                int qh_off_h = (qh_idx_h / block_len) *
                                    (block_len * 8) + r * block_len +
                                    qh_idx_h % block_len;
                                int q_l = (((w[b].qh[qh_off_l] >> shift_l) & 3) << 4) |
                                          (w[b].ql[ql_pos] & 15);
                                int q_h = (((w[b].qh[qh_off_h] >> shift_h) & 3) << 4) |
                                          ((w[b].ql[ql_pos] >> 4) & 15);
                                int q8_l = a[b].qs[q8_base + t * block_len + i];
                                int q8_h = a[b].qs[q8_base + t * block_len + i + 256];
                                dot_l += (q_l - 32) * q8_l;
                                dot_h += (q_h - 32) * q8_h;
                            }
                            int scaled = dot_l * w[b].scales[scale_l * 8 + r] +
                                         dot_h * w[b].scales[scale_h * 8 + r];
                            sum[t][r] += (float)scaled *
                                bn_fp16_to_fp32(w[b].d[r]) * a[b].d[t];
                        }
                    }
                }
            }
            for (int t = 0; t < 4; t++)
                for (int r = 0; r < 8; r++)
                    c->out[(size_t)(panel * 4 + t) * rows + group * 8 + r] =
                        sum[t][r];
        }
    }
}
