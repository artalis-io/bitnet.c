#include "sampler.h"
#include <math.h>
#include <stdlib.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static uint32_t rng_next(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * BN_RNG_MULTIPLIER) >> 32);
}

static float rng_float(uint64_t *state) {
    return (rng_next(state) >> 8) / BN_RNG_FLOAT_DIVISOR;
}

int bn_sampler_init(BnSampler *s, int vocab_size, float temp, float topp, uint64_t seed) {
    s->vocab_size = vocab_size;
    s->temperature = temp;
    s->topp = topp;
    s->repeat_penalty = 1.0f;
    s->rng_state = seed ? seed : 42;
    s->candidates = (BnProbIndex *)malloc(vocab_size * sizeof(BnProbIndex));
    s->candidates_cap = s->candidates ? vocab_size : 0;
    s->recent_tokens = NULL;
    s->recent_cap = 0;
    s->recent_len = 0;
    s->recent_pos = 0;
    return s->candidates ? 0 : -1;
}

void bn_sampler_free(BnSampler *s) {
    if (!s) return;
    free(s->candidates);
    free(s->recent_tokens);
    s->candidates = NULL;
    s->recent_tokens = NULL;
    s->candidates_cap = 0;
}

void bn_sampler_set_repeat_penalty(BnSampler *s, float penalty, int window) {
    s->repeat_penalty = penalty;
    free(s->recent_tokens);
    s->recent_tokens = (int *)calloc(window, sizeof(int));
    s->recent_cap = s->recent_tokens ? window : 0;
    s->recent_len = 0;
    s->recent_pos = 0;
}

void bn_sampler_reset_recent(BnSampler *s) {
    s->recent_len = 0;
    s->recent_pos = 0;
}

void bn_sampler_accept(BnSampler *s, int token) {
    if (!s->recent_tokens || s->recent_cap <= 0) return;
    s->recent_tokens[s->recent_pos] = token;
    s->recent_pos = (s->recent_pos + 1) % s->recent_cap;
    if (s->recent_len < s->recent_cap) s->recent_len++;
}

// #28: Handle n <= 0
static int argmax(float *v, int n) {
    if (n <= 0) return 0;
#if defined(__ARM_NEON)
    int i = 0;
    int best = 0;
    float best_val = v[0];

    if (n >= 4) {
        float32x4_t maxv = vld1q_f32(v);
        int32x4_t maxi = {0, 1, 2, 3};
        int32x4_t idx = {4, 5, 6, 7};
        int32x4_t four = vdupq_n_s32(4);

        for (i = 4; i + 3 < n; i += 4) {
            float32x4_t x = vld1q_f32(v + i);
            uint32x4_t gt = vcgtq_f32(x, maxv);
            maxv = vbslq_f32(gt, x, maxv);
            maxi = vbslq_s32(gt, idx, maxi);
            idx = vaddq_s32(idx, four);
        }

        float vals[4];
        int ids[4];
        vst1q_f32(vals, maxv);
        vst1q_s32(ids, maxi);
        best_val = vals[0];
        best = ids[0];
        for (int k = 1; k < 4; k++) {
            if (vals[k] > best_val) {
                best_val = vals[k];
                best = ids[k];
            }
        }
    }

    for (; i < n; i++) {
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
#else
    int best = 0;
    float best_val = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
#endif
}

// #28: Handle n <= 0
static void softmax(float *x, int n) {
    if (n <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static int sample_multinomial(float *probs, int n, uint64_t *rng) {
    float r = rng_float(rng);
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (r < cdf) return i;
    }
    return n - 1;
}

static int cmp_prob_desc(const void *a, const void *b) {
    float pa = ((const BnProbIndex *)a)->prob;
    float pb = ((const BnProbIndex *)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

static int sample_topp(BnSampler *s, float *probs, int n, float topp) {
    // #29: Handle n <= 1
    if (n <= 0) return 0;
    if (n == 1) return 0;

    // Use preallocated candidates buffer
    BnProbIndex *candidates = s->candidates;
    if (!candidates) return argmax(probs, n);  // fallback to argmax

    // Cutoff: skip tokens with very low probability
    float cutoff = (1.0f - topp) / (float)(n - 1);

    int n_candidates = 0;
    for (int i = 0; i < n; i++) {
        if (probs[i] >= cutoff) {
            candidates[n_candidates].prob = probs[i];
            candidates[n_candidates].index = i;
            n_candidates++;
        }
    }
    qsort(candidates, n_candidates, sizeof(BnProbIndex), cmp_prob_desc);

    // Truncate to top-p nucleus
    float cumulative = 0.0f;
    int last = n_candidates - 1;
    for (int i = 0; i < n_candidates; i++) {
        cumulative += candidates[i].prob;
        if (cumulative > topp) { last = i; break; }
    }

    // Renormalize and sample
    float r = rng_float(&s->rng_state) * cumulative;
    float cdf = 0.0f;
    int result = candidates[last].index;
    for (int i = 0; i <= last; i++) {
        cdf += candidates[i].prob;
        if (r < cdf) { result = candidates[i].index; break; }
    }

    return result;
}

int bn_sampler_sample(BnSampler *s, float *logits) {
    // Apply repetition penalty before temperature/argmax
    if (s->repeat_penalty != 1.0f && s->recent_len > 0) {
        for (int i = 0; i < s->recent_len; i++) {
            int tok = s->recent_tokens[i];
            if (tok >= 0 && tok < s->vocab_size) {
                int seen = 0;
                for (int j = 0; j < i; j++) {
                    if (s->recent_tokens[j] == tok) {
                        seen = 1;
                        break;
                    }
                }
                if (seen) continue;
                if (logits[tok] > 0)
                    logits[tok] /= s->repeat_penalty;
                else
                    logits[tok] *= s->repeat_penalty;
            }
        }
    }

    if (s->temperature == 0.0f) {
        return argmax(logits, s->vocab_size);
    }

    // Apply temperature
    for (int i = 0; i < s->vocab_size; i++) {
        logits[i] /= s->temperature;
    }
    softmax(logits, s->vocab_size);

    if (s->topp <= 0.0f || s->topp >= 1.0f) {
        return sample_multinomial(logits, s->vocab_size, &s->rng_state);
    }
    return sample_topp(s, logits, s->vocab_size, s->topp);
}
