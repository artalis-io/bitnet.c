#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>

typedef struct {
    int      vocab_size;
    float    temperature;
    float    topp;
    uint64_t rng_state;
} Sampler;

void sampler_init(Sampler *s, int vocab_size, float temp, float topp, uint64_t seed);
int  sampler_sample(Sampler *s, float *logits);

#endif // SAMPLER_H
