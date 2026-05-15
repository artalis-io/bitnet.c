#ifndef BN_QUANT_CTX_H
#define BN_QUANT_CTX_H

#include "quant.h"

typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    float combined_scale;
} BnI2SCtx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnI2SFloatCtx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnTQ2Ctx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    float combined_scale;
} BnTQ2SdotCtx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnTQ1Ctx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    float combined_scale;
} BnTQ1SdotCtx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnQ8Ctx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_scales;
    const BnPreparedWeight *prepared;
} BnQ8SdotCtx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnQ4Ctx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_scales;
    const BnPreparedWeight *prepared;
} BnQ4SdotCtx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_scales;
    const BnPreparedWeight *prepared;
    int n_tokens;
    int cols;
} BnQ4MatmulCtx;

typedef struct {
    float *out;
    const BnQWeight *gate;
    const BnQWeight *up;
    const int8_t *x_q;
    const float *x_scales;
    const BnPreparedWeight *gate_prepared;
    const BnPreparedWeight *up_prepared;
} BnQ4GateUpCtx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnQ6KCtx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_d;
    const int16_t *x_bsums;
} BnKQuantSdotCtx;

typedef BnKQuantSdotCtx BnQ6KSdotCtx;
typedef BnKQuantSdotCtx BnQ4KSdotCtx;
typedef BnKQuantSdotCtx BnQ5KSdotCtx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const int8_t *x_q;
    const float *x_d;
    const int16_t *x_bsums;
    int n_tokens;
    int cols;
} BnKQuantMatmulCtx;

typedef struct {
    float *out;
    const BnQWeight *W;
    const float *x;
} BnFloatXCtx;

typedef BnFloatXCtx BnQ8KCtx;
typedef BnFloatXCtx BnQ4KCtx;
typedef BnFloatXCtx BnQ5KCtx;
typedef BnFloatXCtx BnQ4_1Ctx;
typedef BnFloatXCtx BnQ5_1Ctx;
typedef BnFloatXCtx BnF32Ctx;
typedef BnFloatXCtx BnF16Ctx;
typedef BnFloatXCtx BnBF16Ctx;
typedef BnFloatXCtx BnIQ4NLCtx;
typedef BnFloatXCtx BnIQ4XSCtx;
typedef BnFloatXCtx BnIQ3XXSCtx;
typedef BnFloatXCtx BnIQ3SCtx;
typedef BnFloatXCtx BnIQ2XXSCtx;
typedef BnFloatXCtx BnIQ2XSCtx;
typedef BnFloatXCtx BnIQ2SCtx;
typedef BnFloatXCtx BnQ2KCtx;
typedef BnFloatXCtx BnQ3KCtx;

#endif // BN_QUANT_CTX_H
