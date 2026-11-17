#include "quant.h"
#include <assert.h>
#include <fenv.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Unchanged Diffbot Triton lookup, captured BEFORE native implementation.
 * sm120-fixes.diff SHA256 3402f734d039b06b0eeb757dc271a88c2d3ffee94fbc92b41e2a303e3fd9e800.
 * All31744 finite nonnegative FP16 scales, nibbles0..15 repeated twice.
 * BF16 output, FNV1a over little-endian bytes. */
static void exhaustive(void) {
    uint64_t hash=UINT64_C(14695981039346656037);
    for(unsigned scale=0;scale<0x7c00;scale++)for(unsigned j=0;j<32;j++) {
        uint16_t out=0xa55a;
        assert(!bn_quant_int4_fp16s32_to_bf16((uint8_t)(j%16),(uint16_t)scale,&out));
        hash=(hash^(out&255))*UINT64_C(1099511628211);
        hash=(hash^(out>>8))*UINT64_C(1099511628211);
    }
    assert(hash==UINT64_C(0xba104b93f11abf25));
    for(unsigned scale=0x7c00;scale<=0xffff;scale++) {
        uint16_t out=0xa55a;
        assert(bn_quant_int4_fp16s32_to_bf16(0,(uint16_t)scale,&out)==-1&&out==0xa55a);
    }
    uint16_t out=0xa55a;
    assert(bn_quant_int4_fp16s32_to_bf16(16,0x3c00,&out)==-1&&out==0xa55a);
    assert(bn_quant_int4_fp16s32_to_bf16(1,0x3c00,NULL)==-1);
}

static void regions(void) {
    enum {R=3,C=96,N=R*C};
    uint8_t w[N/2],storage[R*C/16+1],*s=storage+1;
    uint16_t out[N+2];
    const uint16_t scales[]={0,1,0x3ff,0x400,0x3555,0x3c00,0x7bff,0x4321,0x2222};
    for(int i=0;i<N/2;i++)w[i]=(uint8_t)(i*37+11);
    for(int i=0;i<R*C/32;i++){s[2*i]=(uint8_t)scales[i];s[2*i+1]=(uint8_t)(scales[i]>>8);}
    for(int row=0;row<R;row++)for(int col=0;col<C;col++) {
        int nr=R-row,nc=C-col;
        for(int i=0;i<N+2;i++)out[i]=0xa55a;
        assert(!bn_quant_int4_fp16s32_decode_region_bf16(w,sizeof(w),s,R*C/16,R,C,row,col,nr,nc,out+1,N));
        assert(out[0]==0xa55a&&out[nr*nc+1]==0xa55a);
        for(int i=0;i<nr;i++)for(int j=0;j<nc;j++) {
            int rr=row+i,cc=col+j;
            uint16_t expected;
            uint8_t nibble=(uint8_t)((w[rr*(C/2)+cc/2]>>((cc%2)*4))&15);
            assert(!bn_quant_int4_fp16s32_to_bf16(nibble,scales[rr*(C/32)+cc/32],&expected));
            assert(out[1+i*nc+j]==expected);
        }
    }
}

static void rejects(void) {
    uint16_t storage[128],out[258];uint8_t *w=(uint8_t *)storage;
    uint8_t scales[16];memset(w,0x17,256);memset(scales,0,16);
    for(int i=0;i<258;i++)out[i]=0xa55a;
#define CALL(W,WB,S,SB,R,C,A,B,NR,NC,O,N) bn_quant_int4_fp16s32_decode_region_bf16(W,WB,S,SB,R,C,A,B,NR,NC,O,N)
    assert(CALL(w,127,scales,16,4,64,0,0,4,64,out+1,256)==-1);
    assert(CALL(w,128,scales,15,4,64,0,0,4,64,out+1,256)==-1);
    assert(CALL(w,128,scales,16,4,63,0,0,4,63,out+1,256)==-1);
    assert(CALL(w,128,scales,16,4,64,-1,0,1,1,out+1,256)==-1);
    assert(CALL(w,128,scales,16,4,64,0,64,1,1,out+1,256)==-1);
    assert(CALL(w,128,scales,16,4,64,0,0,4,64,out+1,255)==-1);
    assert(CALL(w,128,scales,16,4,64,0,0,0,64,out+1,256)==-1);
    assert(CALL(w,128,scales,16,INT_MAX,INT_MAX,0,0,INT_MAX,INT_MAX,out+1,256)==-1);
    assert(CALL(NULL,128,scales,16,4,64,0,0,4,64,out+1,256)==-1);
    assert(CALL(w,128,NULL,16,4,64,0,0,4,64,out+1,256)==-1);
    assert(CALL(w,128,scales,16,4,64,0,0,4,64,NULL,256)==-1);
    assert(CALL(w,128,scales,16,4,64,0,0,4,64,storage,256)==-1);
    assert(CALL(w,128,out,16,4,64,0,0,4,64,out+1,256)==-1);
    for(unsigned bad=0x7c00;bad<=0xffff;bad+=0x400) {
        scales[14]=(uint8_t)bad;scales[15]=(uint8_t)(bad>>8);
        assert(CALL(w,128,scales,16,4,64,0,0,4,64,out+1,256)==-1);
    }
    for(int i=0;i<258;i++)assert(out[i]==0xa55a);
    assert(!CALL(w,128,scales,16,4,64,0,0,1,1,out+1,256));
    for(int i=0;i<128;i++)assert(storage[i]==0x1717);
#undef CALL
}

static void finite_f16_domain(void) {
    uint8_t weights[16],scales[2];uint64_t h=UINT64_C(14695981039346656037);size_t accepted=0;
    for(unsigned scale=0;scale<0x7c00;scale++)for(unsigned nib=0;nib<16;nib++) {
        memset(weights,(int)(nib|(nib<<4)),sizeof(weights));scales[0]=(uint8_t)scale;scales[1]=(uint8_t)(scale>>8);
        int valid=!bn_quant_int4_fp16s32_validate_f16(weights,16,scales,2,1,32);accepted+=(size_t)valid;h=(h^(unsigned)valid)*UINT64_C(1099511628211);
    }
    assert(accepted==479000&&h==UINT64_C(0xf6e795da027822a9));
    memset(weights,0x88,16);scales[0]=0xff;scales[1]=0x7b;assert(!bn_quant_int4_fp16s32_validate_f16(weights,16,scales,2,1,32));
    weights[15]=0x08;assert(bn_quant_int4_fp16s32_validate_f16(weights,16,scales,2,1,32)==-1);
    assert(bn_quant_int4_fp16s32_validate_f16(weights,15,scales,2,1,32)==-1&&bn_quant_int4_fp16s32_validate_f16(weights,16,scales,1,1,32)==-1);
    assert(bn_quant_int4_fp16s32_validate_f16(NULL,16,scales,2,1,32)==-1&&bn_quant_int4_fp16s32_validate_f16(weights,16,NULL,2,1,32)==-1);
    assert(bn_quant_int4_fp16s32_validate_f16(weights,16,scales,2,0,32)==-1&&bn_quant_int4_fp16s32_validate_f16(weights,16,scales,2,1,31)==-1);
}

int main(void) {
    finite_f16_domain();
    assert(!bn_quant_int4_fp16s32_validate_scales(NULL,0));
    assert(bn_quant_int4_fp16s32_validate_scales(NULL,2)==-1);
    unsigned char scale_bytes[3];
    for(unsigned scale=0;scale<=0xffff;scale++) {
        scale_bytes[1]=(unsigned char)scale;scale_bytes[2]=(unsigned char)(scale>>8);
        assert(bn_quant_int4_fp16s32_validate_scales(scale_bytes+1,2)==(scale<0x7c00?0:-1));
        assert(bn_quant_int4_fp16s32_validate_scales(scale_bytes+1,1)==-1);
    }
    exhaustive();regions();rejects();
    int old=fegetround();
    const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    for(size_t i=0;i<sizeof(modes)/sizeof(modes[0]);i++)if(!fesetround(modes[i]))exhaustive();
    if(old>=0)assert(!fesetround(old));
    puts("INT4/FP16S32 source-exact BF16 decoding, bounded regions and atomic rejection PASSED");
    return 0;
}
