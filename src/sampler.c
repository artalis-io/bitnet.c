#include "sampler.h"
#include <math.h>
#include <stdlib.h>

static uint32_t rng_next(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * 0x2545F4914F6CDD1DULL) >> 32);
}

static float rng_float(uint64_t *state) {
    return (rng_next(state) >> 8) / 16777216.0f;
}

void sampler_init(Sampler *s, int vocab_size, float temp, float topp, uint64_t seed) {
    s->vocab_size = vocab_size;
    s->temperature = temp;
    s->topp = topp;
    s->rng_state = seed ? seed : 42;
}

static int argmax(float *v, int n) {
    int best = 0;
    float best_val = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
}

static void softmax(float *x, int n) {
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

typedef struct { float prob; int index; } ProbIndex;

static int cmp_prob_desc(const void *a, const void *b) {
    float pa = ((const ProbIndex *)a)->prob;
    float pb = ((const ProbIndex *)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

static int sample_topp(float *probs, int n, float topp, uint64_t *rng) {
    // Cutoff: skip tokens with very low probability
    float cutoff = (1.0f - topp) / (float)(n - 1);

    ProbIndex *candidates = (ProbIndex *)malloc(n * sizeof(ProbIndex));
    int n_candidates = 0;
    for (int i = 0; i < n; i++) {
        if (probs[i] >= cutoff) {
            candidates[n_candidates].prob = probs[i];
            candidates[n_candidates].index = i;
            n_candidates++;
        }
    }
    qsort(candidates, n_candidates, sizeof(ProbIndex), cmp_prob_desc);

    // Truncate to top-p nucleus
    float cumulative = 0.0f;
    int last = n_candidates - 1;
    for (int i = 0; i < n_candidates; i++) {
        cumulative += candidates[i].prob;
        if (cumulative > topp) { last = i; break; }
    }

    // Renormalize and sample
    float r = rng_float(rng) * cumulative;
    float cdf = 0.0f;
    int result = candidates[last].index;
    for (int i = 0; i <= last; i++) {
        cdf += candidates[i].prob;
        if (r < cdf) { result = candidates[i].index; break; }
    }

    free(candidates);
    return result;
}

int sampler_sample(Sampler *s, float *logits) {
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
    return sample_topp(logits, s->vocab_size, s->topp, &s->rng_state);
}
