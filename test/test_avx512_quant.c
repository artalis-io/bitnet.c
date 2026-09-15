#include "quant.h"
#include "quant_ctx.h"
#include "quant_kernels_avx2.h"
#include "quant_kernels_avx512.h"
#include "quant_kernels_scalar.h"
#include "gguf.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_state = 0x12345678u;

int bn_quant_format_supports_native_quant_logits_refine(int type) {
    (void)type;
    return 1;
}

int bn_quant_format_supports_kquant_logits_refine(int type) {
    (void)type;
    return 1;
}

static uint32_t next_u32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static int next_i(int lo, int hi) {
    return lo + (int)(next_u32() % (uint32_t)(hi - lo + 1));
}

static void fill_x(int8_t *x_q, float *x_d, int16_t *x_bsums, int n_bpr) {
    for (int b = 0; b < n_bpr; b++) {
        x_d[b] = 0.003f * (float)next_i(1, 120);
        for (int g = 0; g < 16; g++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) {
                int v = next_i(-96, 96);
                x_q[b * BN_QK_K + g * 16 + i] = (int8_t)v;
                sum += v;
            }
            x_bsums[b * 16 + g] = (int16_t)sum;
        }
    }
}

static void fill_x32(int8_t *x_q, float *x_scales, int n_bpr) {
    for (int b = 0; b < n_bpr; b++) {
        x_scales[b] = 0.004f * (float)next_i(1, 100);
        for (int i = 0; i < 32; i++)
            x_q[b * 32 + i] = (int8_t)next_i(-96, 96);
    }
}

static void fill_x_float(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = 0.01f * (float)next_i(-100, 100);
}

static void fill_q8(BnBlockQ8_0 *blocks, int n) {
    for (int i = 0; i < n; i++) {
        blocks[i].d = (uint16_t)(0x3400 + next_i(0, 0x0400));
        for (int j = 0; j < 32; j++)
            blocks[i].qs[j] = (int8_t)next_i(-64, 63);
    }
}

static void fill_q4(BnBlockQ4_0 *blocks, int n) {
    for (int i = 0; i < n; i++) {
        blocks[i].d = (uint16_t)(0x3400 + next_i(0, 0x0400));
        for (int j = 0; j < 16; j++)
            blocks[i].qs[j] = (uint8_t)next_i(0, 255);
    }
}

static void fill_q4k(BnBlockQ4K *blocks, int n) {
    for (int i = 0; i < n; i++) {
        blocks[i].d = (uint16_t)(0x3400 + next_i(0, 0x0400));
        blocks[i].dmin = (uint16_t)(0x3000 + next_i(0, 0x0400));
        for (int j = 0; j < 12; j++)
            blocks[i].scales[j] = (uint8_t)next_i(0, 255);
        for (int j = 0; j < BN_QK_K / 2; j++)
            blocks[i].qs[j] = (uint8_t)next_i(0, 255);
    }
}

static void fill_q5k(BnBlockQ5K *blocks, int n) {
    for (int i = 0; i < n; i++) {
        blocks[i].d = (uint16_t)(0x3400 + next_i(0, 0x0400));
        blocks[i].dmin = (uint16_t)(0x3000 + next_i(0, 0x0400));
        for (int j = 0; j < 12; j++)
            blocks[i].scales[j] = (uint8_t)next_i(0, 255);
        for (int j = 0; j < BN_QK_K / 8; j++)
            blocks[i].qh[j] = (uint8_t)next_i(0, 255);
        for (int j = 0; j < BN_QK_K / 2; j++)
            blocks[i].qs[j] = (uint8_t)next_i(0, 255);
    }
}

static void fill_q6k(BnBlockQ6K *blocks, int n) {
    for (int i = 0; i < n; i++) {
        blocks[i].d = (uint16_t)(0x3400 + next_i(0, 0x0400));
        for (int j = 0; j < BN_QK_K / 2; j++)
            blocks[i].ql[j] = (uint8_t)next_i(0, 255);
        for (int j = 0; j < BN_QK_K / 4; j++)
            blocks[i].qh[j] = (uint8_t)next_i(0, 255);
        for (int j = 0; j < BN_QK_K / 16; j++)
            blocks[i].scales[j] = (int8_t)next_i(-32, 31);
    }
}

static void assert_close(const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        float scale = fmaxf(fmaxf(fabsf(a[i]), fabsf(b[i])), 1.0f);
        assert(diff / scale < 2e-5f);
    }
}

static void q4_sdot_reference(float *out, const BnQWeight *W,
                              const int8_t *x_q, const float *x_scales) {
    const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)W->data;
    int n_bpr = W->cols / 32;
    for (int row = 0; row < W->rows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < n_bpr; b++) {
            const BnBlockQ4_0 *blk = &blocks[(size_t)row * n_bpr + b];
            int dot = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t v = blk->qs[i];
                dot += ((int)(v & 0x0F) - 8) * (int)x_q[b * 32 + i];
                dot += ((int)(v >> 4) - 8) * (int)x_q[b * 32 + 16 + i];
            }
            sum += bn_fp16_to_fp32(blk->d) * x_scales[b] * (float)dot;
        }
        out[row] = sum;
    }
}

static void test_q8(void) {
    enum { rows = 9, n_bpr = 5, cols = n_bpr * 32 };
    BnBlockQ8_0 *blocks = (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(*blocks));
    int8_t *x_q = (int8_t *)calloc(cols, sizeof(*x_q));
    float x_scales[n_bpr];
    float scalar[rows], avx2[rows], avx512[rows];

    fill_q8(blocks, rows * n_bpr);
    fill_x32(x_q, x_scales, n_bpr);

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };
    BnQ8SdotCtx cs = { scalar, &W, x_q, x_scales, NULL };
    BnQ8SdotCtx c2 = { avx2, &W, x_q, x_scales, NULL };
    BnQ8SdotCtx c512 = { avx512, &W, x_q, x_scales, NULL };
    bn_quant_q8_scalar_sdot_range(&cs, 0, rows);
    bn_quant_q8_avx2_4row_range(&c2, 0, (rows + 3) / 4);
    bn_quant_q8_avx512_vnni_4row_range(&c512, 0, (rows + 3) / 4);
    assert_close(scalar, avx512, rows);
    assert(memcmp(avx2, avx512, sizeof(avx2)) == 0);

    free(x_q);
    free(blocks);
}

static void test_q8_matmul_matches_matvec(void) {
    enum { rows = 9, n_bpr = 5, cols = n_bpr * 32, n_tokens = 11 };
    BnBlockQ8_0 *blocks =
        (BnBlockQ8_0 *)calloc((size_t)rows * n_bpr, sizeof(*blocks));
    int8_t *x_q =
        (int8_t *)calloc((size_t)n_tokens * cols, sizeof(*x_q));
    float *x_scales = (float *)calloc(
        (size_t)n_tokens * n_bpr, sizeof(*x_scales));
    float *matvec =
        (float *)calloc((size_t)n_tokens * rows, sizeof(*matvec));
    float *matmul =
        (float *)calloc((size_t)n_tokens * rows, sizeof(*matmul));
    assert(blocks && x_q && x_scales && matvec && matmul);

    fill_q8(blocks, rows * n_bpr);
    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q8_0, rows, cols, 1.0f };
    for (int t = 0; t < n_tokens; t++) {
        fill_x32(x_q + (size_t)t * cols,
                 x_scales + (size_t)t * n_bpr, n_bpr);
        BnQ8SdotCtx matvec_ctx = {
            matvec + (size_t)t * rows, &W,
            x_q + (size_t)t * cols,
            x_scales + (size_t)t * n_bpr, NULL
        };
        bn_quant_q8_avx512_vnni_4row_range(
            &matvec_ctx, 0, (rows + 3) / 4);
    }

    BnQ4MatmulCtx matmul_ctx = {
        matmul, &W, x_q, x_scales, NULL, n_tokens, cols,
        NULL, NULL, 0
    };
    bn_quant_q8_avx512_vnni_matmul_4row_range(
        &matmul_ctx, 0, (rows + 3) / 4);
    assert(memcmp(matmul, matvec,
                  (size_t)n_tokens * rows * sizeof(*matmul)) == 0);

    free(matmul);
    free(matvec);
    free(x_scales);
    free(x_q);
    free(blocks);
}

static void test_q4(void) {
    enum { rows = 9, n_bpr = 5, cols = n_bpr * 32 };
    BnBlockQ4_0 *blocks = (BnBlockQ4_0 *)calloc((size_t)rows * n_bpr, sizeof(*blocks));
    int8_t *x_q = (int8_t *)calloc(cols, sizeof(*x_q));
    float x_scales[n_bpr];
    float ref[rows], avx2[rows], avx512[rows];

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q4_0, rows, cols, 1.0f };
    BnQ4SdotCtx c2 = { avx2, &W, x_q, x_scales, NULL };
    BnQ4SdotCtx c512 = { avx512, &W, x_q, x_scales, NULL };
    /* Exercise odd block counts and partial row groups under both GCC and
     * Clang: the AVX512 dot must not reduce undefined upper-half lanes. */
    for (int count = 1; count <= n_bpr; count++) {
        W.cols = count * 32;
        for (int nr = 1; nr <= rows; nr++) {
            W.rows = nr;
            fill_q4(blocks, nr * count);
            fill_x32(x_q, x_scales, count);
            for (int i = 0; i < rows; i++)
                ref[i] = avx2[i] = avx512[i] = NAN;
            q4_sdot_reference(ref, &W, x_q, x_scales);
            bn_quant_q4_avx2_4row_range(&c2, 0, (nr + 3) / 4);
            bn_quant_q4_avx512_vnni_4row_range(&c512, 0, (nr + 3) / 4);
            assert_close(ref, avx512, nr);
            assert_close(avx2, avx512, nr);
            for (int i = nr; i < rows; i++)
                assert(isnan(avx512[i]));
        }
    }

    free(x_q);
    free(blocks);
}

static void test_q4k(void) {
    enum { rows = 9, n_bpr = 3, cols = n_bpr * BN_QK_K };
    BnBlockQ4K *blocks = (BnBlockQ4K *)calloc((size_t)rows * n_bpr, sizeof(*blocks));
    int8_t *x_q = (int8_t *)calloc(cols, sizeof(*x_q));
    float x_d[n_bpr];
    int16_t x_bsums[n_bpr * 16];
    float scalar[rows], avx2[rows], avx512[rows];

    fill_q4k(blocks, rows * n_bpr);
    fill_x(x_q, x_d, x_bsums, n_bpr);

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q4_K, rows, cols, 1.0f };
    BnKQuantSdotCtx cs = { scalar, &W, x_q, x_d, x_bsums, NULL };
    BnKQuantSdotCtx c2 = { avx2, &W, x_q, x_d, x_bsums, NULL };
    BnKQuantSdotCtx c512 = { avx512, &W, x_q, x_d, x_bsums, NULL };
    bn_quant_q4k_scalar_sdot_range(&cs, 0, rows);
    bn_quant_q4k_avx2_4row_range(&c2, 0, (rows + 3) / 4);
    bn_quant_q4k_avx512_vnni_4row_range(&c512, 0, (rows + 3) / 4);
    assert_close(scalar, avx512, rows);
    assert_close(avx2, avx512, rows);

    free(x_q);
    free(blocks);
}

static void test_q4k_matmul_matches_matvec(void) {
    enum {
        rows = 9,
        n_bpr = 3,
        cols = n_bpr * BN_QK_K,
        n_tokens = 5
    };
    BnBlockQ4K *blocks =
        (BnBlockQ4K *)calloc((size_t)rows * n_bpr, sizeof(*blocks));
    int8_t *x_q = (int8_t *)calloc((size_t)n_tokens * cols, sizeof(*x_q));
    float *x_d =
        (float *)calloc((size_t)n_tokens * n_bpr, sizeof(*x_d));
    int16_t *x_bsums = (int16_t *)calloc(
        (size_t)n_tokens * n_bpr * 16, sizeof(*x_bsums));
    float *matvec =
        (float *)calloc((size_t)n_tokens * rows, sizeof(*matvec));
    float *matmul =
        (float *)calloc((size_t)n_tokens * rows, sizeof(*matmul));
    assert(blocks && x_q && x_d && x_bsums && matvec && matmul);

    fill_q4k(blocks, rows * n_bpr);
    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q4_K, rows, cols, 1.0f };
    for (int t = 0; t < n_tokens; t++) {
        fill_x(x_q + (size_t)t * cols,
               x_d + (size_t)t * n_bpr,
               x_bsums + (size_t)t * n_bpr * 16, n_bpr);
        BnKQuantSdotCtx matvec_ctx = {
            matvec + (size_t)t * rows, &W,
            x_q + (size_t)t * cols,
            x_d + (size_t)t * n_bpr,
            x_bsums + (size_t)t * n_bpr * 16, NULL
        };
        bn_quant_q4k_avx512_vnni_4row_range(
            &matvec_ctx, 0, (rows + 3) / 4);
    }

    BnKQuantMatmulCtx matmul_ctx = {
        .out = matmul,
        .W = &W,
        .x_q = x_q,
        .x_d = x_d,
        .x_bsums = x_bsums,
        .n_tokens = n_tokens,
        .cols = cols,
        .prepared = NULL,
        .x_q8k_x4 = NULL,
    };
    bn_quant_q4k_avx512_vnni_matmul_4row_range(
        &matmul_ctx, 0, (rows + 3) / 4);
    assert(memcmp(matmul, matvec,
                  (size_t)n_tokens * rows * sizeof(*matmul)) == 0);

    free(matmul);
    free(matvec);
    free(x_bsums);
    free(x_d);
    free(x_q);
    free(blocks);
}

static void test_q5k(void) {
    enum { rows = 9, n_bpr = 3, cols = n_bpr * BN_QK_K };
    BnBlockQ5K *blocks = (BnBlockQ5K *)calloc((size_t)rows * n_bpr, sizeof(*blocks));
    float *x = (float *)calloc(cols, sizeof(*x));
    int8_t *x_q = (int8_t *)calloc(cols, sizeof(*x_q));
    float x_d[n_bpr];
    int16_t x_bsums[n_bpr * 16];
    float *x_deq = (float *)calloc(cols, sizeof(*x_deq));
    float scalar[rows], avx2[rows], avx512[rows], vnni_ref[rows], avx512_vnni[rows];

    fill_q5k(blocks, rows * n_bpr);
    fill_x_float(x, cols);
    fill_x(x_q, x_d, x_bsums, n_bpr);
    for (int b = 0; b < n_bpr; b++)
        for (int i = 0; i < BN_QK_K; i++)
            x_deq[b * BN_QK_K + i] = x_d[b] * (float)x_q[b * BN_QK_K + i];

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q5_K, rows, cols, 1.0f };
    BnQ5KCtx cs = { scalar, &W, x };
    BnQ5KCtx c2 = { avx2, &W, x };
    BnQ5KCtx c512 = { avx512, &W, x };
    BnQ5KCtx cref = { vnni_ref, &W, x_deq };
    BnKQuantSdotCtx cvnni = { avx512_vnni, &W, x_q, x_d, x_bsums, NULL };
    bn_quant_q5k_scalar_range(&cs, 0, rows);
    bn_quant_q5k_avx2_4row_range(&c2, 0, (rows + 3) / 4);
    bn_quant_q5k_avx512_4row_range(&c512, 0, (rows + 3) / 4);
    bn_quant_q5k_scalar_range(&cref, 0, rows);
    bn_quant_q5k_avx512_vnni_4row_range(&cvnni, 0, (rows + 3) / 4);
    assert_close(scalar, avx512, rows);
    assert_close(avx2, avx512, rows);
    assert_close(vnni_ref, avx512_vnni, rows);

    free(x_deq);
    free(x_q);
    free(x);
    free(blocks);
}

static void test_q6k(void) {
    enum { rows = 9, n_bpr = 3, cols = n_bpr * BN_QK_K };
    BnBlockQ6K *blocks = (BnBlockQ6K *)calloc((size_t)rows * n_bpr, sizeof(*blocks));
    int8_t *x_q = (int8_t *)calloc(cols, sizeof(*x_q));
    float x_d[n_bpr];
    int16_t x_bsums[n_bpr * 16];
    float scalar[rows], avx2[rows], avx512[rows];

    fill_q6k(blocks, rows * n_bpr);
    fill_x(x_q, x_d, x_bsums, n_bpr);

    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q6_K, rows, cols, 1.0f };
    BnKQuantSdotCtx cs = { scalar, &W, x_q, x_d, x_bsums, NULL };
    BnKQuantSdotCtx c2 = { avx2, &W, x_q, x_d, x_bsums, NULL };
    BnKQuantSdotCtx c512 = { avx512, &W, x_q, x_d, x_bsums, NULL };
    bn_quant_q6k_scalar_sdot_range(&cs, 0, rows);
    bn_quant_q6k_avx2_4row_range(&c2, 0, (rows + 3) / 4);
    bn_quant_q6k_avx512_vnni_4row_range(&c512, 0, (rows + 3) / 4);
    assert_close(scalar, avx512, rows);
    assert_close(avx2, avx512, rows);

    free(x_q);
    free(blocks);
}

static void test_q6k_matmul_matches_matvec(void) {
    enum {
        rows = 9,
        n_bpr = 3,
        cols = n_bpr * BN_QK_K,
        n_tokens = 5
    };
    BnBlockQ6K *blocks =
        (BnBlockQ6K *)calloc((size_t)rows * n_bpr, sizeof(*blocks));
    int8_t *x_q = (int8_t *)calloc((size_t)n_tokens * cols, sizeof(*x_q));
    float *x_d =
        (float *)calloc((size_t)n_tokens * n_bpr, sizeof(*x_d));
    int16_t *x_bsums = (int16_t *)calloc(
        (size_t)n_tokens * n_bpr * 16, sizeof(*x_bsums));
    float *matvec =
        (float *)calloc((size_t)n_tokens * rows, sizeof(*matvec));
    float *matmul =
        (float *)calloc((size_t)n_tokens * rows, sizeof(*matmul));
    assert(blocks && x_q && x_d && x_bsums && matvec && matmul);

    fill_q6k(blocks, rows * n_bpr);
    BnQWeight W = { blocks, BN_GGUF_TENSOR_Q6_K, rows, cols, 1.0f };
    for (int t = 0; t < n_tokens; t++) {
        fill_x(x_q + (size_t)t * cols,
               x_d + (size_t)t * n_bpr,
               x_bsums + (size_t)t * n_bpr * 16, n_bpr);
        BnKQuantSdotCtx matvec_ctx = {
            matvec + (size_t)t * rows, &W,
            x_q + (size_t)t * cols,
            x_d + (size_t)t * n_bpr,
            x_bsums + (size_t)t * n_bpr * 16, NULL
        };
        bn_quant_q6k_avx512_vnni_4row_range(
            &matvec_ctx, 0, (rows + 3) / 4);
    }

    BnKQuantMatmulCtx matmul_ctx = {
        .out = matmul,
        .W = &W,
        .x_q = x_q,
        .x_d = x_d,
        .x_bsums = x_bsums,
        .n_tokens = n_tokens,
        .cols = cols,
        .prepared = NULL,
        .x_q8k_x4 = NULL,
    };
    bn_quant_q6k_avx512_vnni_matmul_4row_range(
        &matmul_ctx, 0, (rows + 3) / 4);
    assert(memcmp(matmul, matvec,
                  (size_t)n_tokens * rows * sizeof(*matmul)) == 0);

    free(matmul);
    free(matvec);
    free(x_bsums);
    free(x_d);
    free(x_q);
    free(blocks);
}

int main(void) {
    test_q8();
    test_q8_matmul_matches_matvec();
    test_q4();
    test_q4k();
    test_q4k_matmul_matches_matvec();
    test_q5k();
    test_q6k();
    test_q6k_matmul_matches_matvec();
    printf("PASSED\n");
    return 0;
}
