#ifndef BN_SIGNED_MMQ_FIXTURE_H
#define BN_SIGNED_MMQ_FIXTURE_H
#include "quant.h"
#include <stdint.h>
#include <stddef.h>
static size_t signed_mmq_fixture_bytes(int type, size_t elements) {
 return type==8 ? elements/32*sizeof(BnBlockQ8_0) : type==21 ? elements/256*sizeof(BnBlockIQ3S) : type==11 ? elements/256*sizeof(BnBlockQ3K) : type==23 ? elements/256*sizeof(BnBlockIQ4XS) : elements/32*sizeof(BnBlockIQ4NL);
}
static void signed_mmq_fixture_weights(void *data,int type,size_t elements,int seed) {
 size_t blocks=elements/((type==20||type==8)?32:256);
 for(size_t b=0;b<blocks;b++) {
  uint16_t d=(uint16_t)(0x2400+(b%19)*37);if(seed && b%3==0)d|=0x8000;if(seed && b%17==0)d=0;
  if(type==8){BnBlockQ8_0*w=(BnBlockQ8_0*)data+b;w->d=d;
   for(int i=0;i<32;i++)w->qs[i]=(int8_t)((int)((b*29+i*13+seed*3)%255)-127);
  }else if(type==21){BnBlockIQ3S*w=(BnBlockIQ3S*)data+b;w->d=d;
   for(int i=0;i<64;i++)w->qs[i]=(uint8_t)(b*29+i*13+seed*3);
   for(int i=0;i<8;i++)w->qh[i]=(uint8_t)(b*19+i*37+seed);
   for(int i=0;i<32;i++)w->signs[i]=(uint8_t)(b*31+i*17+seed*5);
   for(int i=0;i<4;i++)w->scales[i]=(uint8_t)(b*23+i*19+seed);
  }else if(type==11){BnBlockQ3K*w=(BnBlockQ3K*)data+b;w->d=d;
   for(int i=0;i<32;i++)w->hmask[i]=(uint8_t)(b*19+i*37+seed);
   for(int i=0;i<64;i++)w->qs[i]=(uint8_t)(b*29+i*13+seed*3);
   for(int i=0;i<12;i++)w->scales[i]=(uint8_t)(b*31+i*17+seed*5);
  }else if(type==23){BnBlockIQ4XS*w=(BnBlockIQ4XS*)data+b;w->d=d;
   w->scales_h=(uint16_t)(b*157+seed*71);
   for(int i=0;i<4;i++)w->scales_l[i]=(uint8_t)(b*23+i*19+seed);
   for(int i=0;i<128;i++)w->qs[i]=(uint8_t)(b*29+i*13+seed*3);
  }else{BnBlockIQ4NL*w=(BnBlockIQ4NL*)data+b;w->d=d;
   for(int i=0;i<16;i++)w->qs[i]=(uint8_t)(b*29+i*13+seed*3);
  }
 }
}
static void signed_mmq_fixture_input(float *x,size_t n,int seed) {
 for(size_t i=0;i<n;i++)x[i]=(float)((int)((i*17+i/37+seed*11)%509)-254)/128.0f;
}
#endif
