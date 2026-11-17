/* Frozen numerical hashes captured from unchanged upstream CUDA load routine
 * bd1f519172406a9560bd5c90e61000691bb69a07 before native implementation.
 * Partial matrices: original on zero-padded complete block32 grids, then crop. */
#include "quant.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <fenv.h>
static uint64_t hash(uint64_t h,uint16_t w){h=(h^(w&255))*UINT64_C(1099511628211);return(h^(w>>8))*UINT64_C(1099511628211);}
static void pairs(void) {
    uint64_t h=UINT64_C(14695981039346656037);size_t count=0;
    for(unsigned w=0;w<256;w++)for(unsigned s=0;s<256;s++) {
        uint16_t out=0xa55a;int result=bn_quant_e4m3_ue8m0_to_fp16((uint8_t)w,(uint8_t)s,&out);
        if((w&127)==127||s==255)assert(result==-1&&out==0xa55a);
        else {assert(!result);h=hash(h,out);count++;}
    }
    assert(count==64770&&h==UINT64_C(0x463cac3f204833bd));
}
static void special(void) {
    const unsigned values[][3]={{0,127,0},{128,127,0x8000},{0x38,127,0x3c00},{0xb8,127,0xbc00},{0x7e,127,0x5f00},{0xfe,127,0xdf00},{1,127,0x1800},{0x38,128,0x4000},{0x38,103,1},{0x38,102,0},{0x39,102,1},{0x3c,103,2},{0x3a,104,2}};
    for(size_t i=0;i<sizeof(values)/sizeof(values[0]);i++){uint16_t v;assert(!bn_quant_e4m3_ue8m0_to_fp16((uint8_t)values[i][0],(uint8_t)values[i][1],&v)&&v==values[i][2]);}
    assert(bn_quant_e4m3_ue8m0_to_fp16(0x38,127,NULL)==-1);
}
static void geometry(int rows,int cols,uint64_t expected) {
    size_t n=(size_t)rows*cols,ns=(size_t)((rows+31)/32)*((cols+31)/32);uint8_t *w=malloc(n),*s=malloc(ns);uint16_t *out=malloc((n+2)*2);assert(w&&s&&out);
    for(size_t i=0;i<n;i++)w[i]=(uint8_t)((i*37+11)%127);
    for(size_t i=0;i<ns;i++)s[i]=(uint8_t)(100+(i*13)%45);
    for(size_t i=0;i<n+2;i++)out[i]=0xa55a;
    assert(!bn_quant_e4m3_ue8m0_decode_region_fp16(w,n,s,ns,rows,cols,0,0,rows,cols,out+1,n));assert(out[0]==0xa55a&&out[n+1]==0xa55a);
    uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++)h=hash(h,out[i+1]);assert(h==expected);
    int rr[]={0,rows>31?31:rows-1,rows>32?32:rows-1,rows-1},cc[]={0,cols>31?31:cols-1,cols>32?32:cols-1,cols-1};
    for(int a=0;a<4;a++)for(int b=0;b<4;b++) {
        int nr=rows-rr[a]<3?rows-rr[a]:3,nc=cols-cc[b]<7?cols-cc[b]:7;uint16_t region[23];for(int i=0;i<23;i++)region[i]=0xa55a;
        assert(!bn_quant_e4m3_ue8m0_decode_region_fp16(w,n,s,ns,rows,cols,rr[a],cc[b],nr,nc,region+1,21));assert(region[0]==0xa55a&&region[nr*nc+1]==0xa55a);
        for(int i=0;i<nr;i++)for(int j=0;j<nc;j++)assert(region[1+i*nc+j]==out[1+(size_t)(rr[a]+i)*cols+cc[b]+j]);
    }
    free(out);free(s);free(w);
}
static void rejects(void) {
    uint8_t w[4096],s[4];uint16_t out[4098];memset(w,0x38,sizeof(w));memset(s,127,sizeof(s));for(int i=0;i<4098;i++)out[i]=0xa55a;
#define CALL(W,WB,S,SB,R,C,A,B,NR,NC,O,N) bn_quant_e4m3_ue8m0_decode_region_fp16(W,WB,S,SB,R,C,A,B,NR,NC,O,N)
    assert(CALL(w,4095,s,4,64,64,0,0,64,64,out+1,4096)==-1);assert(CALL(w,4096,s,3,64,64,0,0,64,64,out+1,4096)==-1);
    assert(CALL(w,4096,s,4,64,64,0,0,64,64,out+1,4095)==-1);assert(CALL(w,4096,s,4,64,64,-1,0,1,1,out+1,4096)==-1);
    assert(CALL(w,4096,s,4,64,64,64,0,1,1,out+1,4096)==-1);assert(CALL(w,4096,s,4,64,64,0,0,0,1,out+1,4096)==-1);
    assert(CALL(w,4096,s,4,INT_MAX,INT_MAX,0,0,INT_MAX,INT_MAX,out+1,4096)==-1);assert(CALL(NULL,4096,s,4,64,64,0,0,1,1,out+1,4096)==-1);
    w[4095]=127;assert(CALL(w,4096,s,4,64,64,0,0,64,64,out+1,4096)==-1);w[4095]=255;assert(CALL(w,4096,s,4,64,64,0,0,64,64,out+1,4096)==-1);w[4095]=0x38;s[3]=255;assert(CALL(w,4096,s,4,64,64,0,0,64,64,out+1,4096)==-1);
    for(int i=0;i<4098;i++)assert(out[i]==0xa55a);
    /* Invalid values outside the requested rectangle do not reject it. */
    assert(!CALL(w,4096,s,4,64,64,0,0,1,1,out+1,4096)&&out[1]==0x3c00);s[3]=127;
    uint16_t alias[4096];for(int i=0;i<4096;i++)alias[i]=0xa55a;
    assert(CALL(alias,4096,s,4,64,64,0,0,64,64,alias,4096)==-1);assert(CALL(w,4096,alias,4,64,64,0,0,64,64,alias,4096)==-1);for(int i=0;i<4096;i++)assert(alias[i]==0xa55a);
#undef CALL
}
int main(void) {
    pairs();special();geometry(35,67,UINT64_C(0x9c9739232c4d4aae));geometry(1,1,UINT64_C(0x08328807b4eb6fed));geometry(64,96,UINT64_C(0x9400df6ff3467df9));geometry(65,33,UINT64_C(0xbee657b37d8e81af));rejects();
    int previous=fegetround();const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};for(size_t i=0;i<sizeof(modes)/sizeof(modes[0]);i++)if(!fesetround(modes[i]))pairs();if(previous>=0)assert(!fesetround(previous));
    puts("E4M3/UE8M0 block32 exact FP16 decoding, bounded regions and atomic rejection PASSED");return 0;
}
