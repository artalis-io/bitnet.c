#ifndef BN_CUDA_STANDALONE_FFN_FIXTURE_H
#define BN_CUDA_STANDALONE_FFN_FIXTURE_H
#include "quant.h"
#include "cuda_signed_mmq_fixture.h"
#include <string.h>
static size_t ffn_bytes(int type,size_t n) {
 if(type==12)return n/256*sizeof(BnBlockQ4K);
 if(type==13)return n/256*sizeof(BnBlockQ5K);
 if(type==14)return n/256*sizeof(BnBlockQ6K);
 return signed_mmq_fixture_bytes(type,n);
}
static void ffn_weights(void*data,int type,size_t n,int seed) {
 if(type!=12&&type!=13&&type!=14){signed_mmq_fixture_weights(data,type,n,seed);return;}
 memset(data,0,ffn_bytes(type,n));
 for(size_t b=0;b<n/256;b++) {
  uint16_t d=(uint16_t)(0x1400+(b%19)*37);if(seed && b%3==0)d|=0x8000;if(seed && b%17==0)d=0;
  if(type==12){BnBlockQ4K*w=(BnBlockQ4K*)data+b;w->d=d;w->dmin=0x1000+(b%7)*23;for(int i=0;i<12;i++)w->scales[i]=(uint8_t)(b*31+i*17+seed*5);for(int i=0;i<128;i++)w->qs[i]=(uint8_t)(b*29+i*13+seed*3);}
  if(type==13){BnBlockQ5K*w=(BnBlockQ5K*)data+b;w->d=d;w->dmin=0x1000+(b%7)*23;for(int i=0;i<12;i++)w->scales[i]=(uint8_t)(b*31+i*17+seed*5);for(int i=0;i<128;i++)w->qs[i]=(uint8_t)(b*29+i*13+seed*3);for(int i=0;i<32;i++)w->qh[i]=(uint8_t)(b*19+i*37+seed);}
  if(type==14){BnBlockQ6K*w=(BnBlockQ6K*)data+b;w->d=d;for(int i=0;i<128;i++)w->ql[i]=(uint8_t)(b*29+i*13+seed*3);for(int i=0;i<64;i++)w->qh[i]=(uint8_t)(b*19+i*37+seed);for(int i=0;i<16;i++)w->scales[i]=(int8_t)((b*31+i*17+seed*5)%127-63);}
 }
}

#endif
