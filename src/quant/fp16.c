#include "quant.h"
#include <string.h>

// --- FP16 <-> FP32 conversion ---

float bn_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & BN_FP16_SIGN_MASK) << 16;
    uint32_t exp  = (h >> 10) & BN_FP16_EXP_MASK;
    uint32_t mant = h & BN_FP16_MANT_MASK;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // +/-0
        } else {
            // Subnormal: normalize by shifting mantissa left until hidden bit appears.
            exp = 1;
            while (!(mant & BN_FP16_HIDDEN_BIT)) { mant <<= 1; exp--; }
            mant &= BN_FP16_MANT_MASK;
            f = sign | ((uint32_t)(exp + BN_FP16_EXP_REBIAS) << 23) | ((uint32_t)mant << 13);
        }
    } else if (exp == 31) {
        f = sign | BN_FP32_EXP_INF | ((uint32_t)mant << 13);  // Inf/NaN
    } else {
        f = sign | ((uint32_t)(exp + BN_FP16_EXP_REBIAS) << 23) | ((uint32_t)mant << 13);
    }

    float result;
    memcpy(&result, &f, 4);
    return result;
}

uint16_t bn_fp32_to_fp16(float val) {
    uint32_t f;
    memcpy(&f, &val, 4);

    uint32_t sign = (f >> 16) & BN_FP16_SIGN_MASK;
    int32_t  exp  = ((f >> 23) & 0xFF) - 127;
    uint32_t mant = f & BN_FP32_MANT_MASK;

    if (exp > 15) {
        return (uint16_t)(sign | BN_FP16_INF);  // Inf
    } else if (exp < -14) {
        return (uint16_t)sign;  // Zero (flush subnormals)
    } else {
        return (uint16_t)(sign | ((uint32_t)(exp + 15) << 10) | (mant >> 13));
    }
}
