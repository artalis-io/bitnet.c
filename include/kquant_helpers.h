#ifndef BN_KQUANT_HELPERS_H
#define BN_KQUANT_HELPERS_H

#include <stdint.h>
#include <string.h>

static inline void bn_q4k_get_scale_min(int j, const uint8_t *q,
                                        uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

static inline void bn_q3k_unpack_scales(const uint8_t *scales, uint8_t *out) {
    uint32_t aux[4];
    memcpy(aux, scales, 3 * sizeof(uint32_t));
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0fu)         | (((tmp >> 0) & 0x03030303u) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0fu)         | (((tmp >> 2) & 0x03030303u) << 4);
    memcpy(out, aux, sizeof(aux));
}

#endif // BN_KQUANT_HELPERS_H
