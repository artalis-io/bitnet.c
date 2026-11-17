#ifndef BN_CUDA_QK8_FIXTURE_H
#define BN_CUDA_QK8_FIXTURE_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
/* Raw synthetic GGUF blocks, shared only with the independent oracle. */
static inline void qk8_fixture_gate(void *data, size_t elements, int five, int seed) {
    size_t stride = five ? 176 : 144;
    for (size_t b=0; b<elements/256; b++) {
        uint8_t *w=(uint8_t *)data+b*stride;
        uint16_t d=(uint16_t)(0x1800+((b+seed)%17)*37);
        uint16_t m=(uint16_t)(0x1400+((b*3+seed)%19)*29);
        memcpy(w,&d,2);memcpy(w+2,&m,2);
        for(int j=0;j<12;j++)w[4+j]=(uint8_t)(b*7+j*13+seed);
        if(five)for(int j=0;j<32;j++)w[16+j]=(uint8_t)(b*11+j*17+seed);
        for(int j=0;j<128;j++)w[(five?48:16)+j]=(uint8_t)(b*19+j*23+seed);
    }
}
static inline void qk8_fixture_down(void *data, size_t elements, int seed) {
    for(size_t b=0;b<elements/32;b++) {
        uint8_t *w=(uint8_t *)data+b*34;
        uint16_t d=(uint16_t)(0x1800+((b+seed)%17)*37);
        memcpy(w,&d,2);
        for(int j=0;j<32;j++)w[2+j]=(uint8_t)((int)((b*19+j*23+seed)%255)-127);
    }
}
#endif
