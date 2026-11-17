#ifndef BN_SAMPLER_BACKEND_INTERNAL_H
#define BN_SAMPLER_BACKEND_INTERNAL_H

/* Equal maxima select the lowest token index on every SIMD backend. */
int bn_sampler_argmax(const float *v, int n);

#endif // BN_SAMPLER_BACKEND_INTERNAL_H
