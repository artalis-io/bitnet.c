#ifndef BN_SAMPLER_H
#define BN_SAMPLER_H

#include <stdint.h>

#define BN_RNG_MULTIPLIER     0x2545F4914F6CDD1DULL  // xorshift64* constant
#define BN_RNG_FLOAT_DIVISOR  16777216.0f             // 2^24, for [0,1) float

typedef struct { float prob; int index; } BnProbIndex;

typedef struct {
    int      vocab_size;
    float    temperature;
    float    topp;
    float    repeat_penalty;
    uint64_t rng_state;
    BnProbIndex *candidates;  // preallocated for top-p sampling
    int      candidates_cap;
    // Recent token ring buffer for repetition penalty
    int     *recent_tokens;
    int      recent_cap;
    int      recent_len;
    int      recent_pos;
} BnSampler;

int  bn_sampler_init(BnSampler *s, int vocab_size, float temp, float topp, uint64_t seed);
void bn_sampler_free(BnSampler *s);
void bn_sampler_set_repeat_penalty(BnSampler *s, float penalty, int window);
void bn_sampler_accept(BnSampler *s, int token);
void bn_sampler_reset_recent(BnSampler *s);
int  bn_sampler_sample(BnSampler *s, float *logits);

#endif // BN_SAMPLER_H
