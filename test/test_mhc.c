#include "transformer_math_internal.h"
#include "mhc_reference.h"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <string.h>

static void close_values(const float *a, const float *b, int count) {
    for (int i = 0; i < count; i++) {
        assert(isfinite(a[i]));
        assert(fabsf(a[i] - b[i]) < 5e-6f);
    }
}

static void test_reference(void) {
    for (int t = 0; t < 3; t++) {
        float pre[6], post[6], comb[18], input[5], output[14];
        for (int i = 0; i < 6; i++) pre[i] = post[i] = 77;
        for (int i = 0; i < 18; i++) comb[i] = 77;
        for (int i = 0; i < 5; i++) input[i] = 77;
        for (int i = 0; i < 14; i++) output[i] = 77;
        BnMHCCoeffCtx c = {pre + 1, post + 1, comb + 1,
            mhc_ref_mixes + t * 24, mhc_ref_scale, mhc_ref_base,
            4, 20, 1e-6f, 1e-6f, 2};
        assert(bn_transformer_mhc_coefficients(&c) == 0);
        close_values(pre + 1, mhc_ref_pre + t * 4, 4);
        close_values(post + 1, mhc_ref_post + t * 4, 4);
        close_values(comb + 1, mhc_ref_comb + t * 16, 16);
        assert(pre[0] == 77 && pre[5] == 77 && post[0] == 77 && post[5] == 77);
        assert(comb[0] == 77 && comb[17] == 77);
        assert(bn_transformer_mhc_mix_input(input + 1,
            mhc_ref_residual + t * 12, pre + 1, 4, 3) == 0);
        assert(bn_transformer_mhc_mix_output(output + 1,
            mhc_ref_residual + t * 12, mhc_ref_branch + t * 3,
            post + 1, comb + 1, 4, 3) == 0);
        close_values(input + 1, mhc_ref_collapsed + t * 3, 3);
        close_values(output + 1, mhc_ref_out + t * 12, 12);
        assert(input[0] == 77 && input[4] == 77);
        assert(output[0] == 77 && output[13] == 77);
    }
}

static void test_orientation(void) {
    const float residual[] = {1, 2, 3, 4}, branch[] = {10, 20};
    const float pre[] = {0.25f, 0.75f}, post[] = {0.5f, 2};
    const float comb[] = {0.25f, 0.5f, 0.75f, 0.5f};
    const float expected[] = {7.5f, 13.5f, 22, 43}, collapsed[] = {2.5f, 3.5f};
    float input[2], output[4];
    assert(bn_transformer_mhc_mix_input(input, residual, pre, 2, 2) == 0);
    assert(bn_transformer_mhc_mix_output(output, residual, branch, post, comb, 2, 2) == 0);
    assert(memcmp(input, collapsed, sizeof(input)) == 0);
    assert(memcmp(output, expected, sizeof(output)) == 0);
}

static void test_failures(void) {
    for (int bad = 0; bad < 14; bad++) {
        float mixes[24] = {0}, base[24] = {0}, scale[] = {1, 1, 1};
        float pre[4], post[4], comb[16];
        for (int i = 0; i < 4; i++) pre[i] = post[i] = 99;
        for (int i = 0; i < 16; i++) comb[i] = 99;
        BnMHCCoeffCtx c = {pre, post, comb, mixes, scale, base, 4, 20, 1e-6f, 1e-6f, 2};
        if (bad == 0) c.streams = 0;
        if (bad == 1) c.streams = 17;
        if (bad == 2) c.iterations = 0;
        if (bad == 3) c.iterations = 129;
        if (bad == 4) c.pre_eps = -1;
        if (bad == 5) c.sinkhorn_eps = 0;
        if (bad == 6) c.post_mult = NAN;
        if (bad == 7) mixes[23] = NAN;
        if (bad == 8) base[23] = INFINITY;
        if (bad == 9) scale[2] = NAN;
        if (bad == 10) {mixes[8] = FLT_MAX; scale[2] = FLT_MAX;}
        if (bad == 11) c.sinkhorn_eps = FLT_MAX;
        if (bad == 12) c.pre = NULL;
        if (bad == 13) c.mixes = NULL;
        assert(bn_transformer_mhc_coefficients(&c) == -1);
        for (int i = 0; i < 4; i++) assert(pre[i] == 99 && post[i] == 99);
        for (int i = 0; i < 16; i++) assert(comb[i] == 99);
    }
    float residual[] = {1, 2, 3, NAN}, branch[] = {1, 2};
    float pre[] = {1, 1}, post[] = {1, 1}, comb[] = {1, 0, 0, 1}, out[] = {77, 77, 77, 77};
    assert(bn_transformer_mhc_mix_input(out, residual, pre, 2, 2) == -1);
    assert(bn_transformer_mhc_mix_output(out, residual, branch, post, comb, 2, 2) == -1);
    for (int i = 0; i < 4; i++) assert(out[i] == 77);
    assert(bn_transformer_mhc_mix_input(NULL, NULL, NULL, 4, 0) == 0);
    assert(bn_transformer_mhc_mix_output(NULL, NULL, NULL, NULL, NULL, 4, 0) == 0);
    assert(bn_transformer_mhc_mix_input(out, residual, pre, 0, 2) == -1);
    assert(bn_transformer_mhc_mix_output(out, residual, branch, post, comb, 2, -1) == -1);
    assert(bn_transformer_mhc_coefficients(NULL) == -1);
}

int main(void) {
    test_reference(); test_orientation(); test_failures();
    puts("mHC independent coefficients, mixes, orientation and rejection PASSED");
    return 0;
}
