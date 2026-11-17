#include "quant.h"

/* Load-time FP16 profile of exllamav3/modules/linear.py dequant_e8m0_blocks_,
 * pinned at bd1f519172406a9560bd5c90e61000691bb69a07. E4M3FN value times
 * 2^(scale-127), F32 intermediate then FP16 round-to-nearest-even.
 * Integer implementation: the significand has at most four bits. Values in
 * FP16's nonzero finite range are exact in FP32, so a direct FP16 rounding is
 * equivalent. FP32 overflow/underflow is already outside that FP16 range. */
static uint16_t decode_finite(uint8_t weight,uint8_t scale) {
    unsigned exponent=(weight>>3)&15u,coefficient=weight&7u;
    uint16_t sign=(uint16_t)((weight&128u)<<8);
    if(exponent)coefficient+=8;
    if(!coefficient)return sign;
    int power=(exponent?(int)exponent-10:-9)+(int)scale-127;
    int leading=coefficient>=8?3:coefficient>=4?2:coefficient>=2?1:0;
    int top=power+leading;
    if(top>15)return (uint16_t)(sign|0x7c00u);
    if(top>=-14)return (uint16_t)(sign|((unsigned)(top+15)<<10)|
        ((coefficient<<(10-leading))-1024u));
    int shift=power+24;
    unsigned mantissa;
    if(shift>=0)mantissa=coefficient<<shift;
    else {
        unsigned right=(unsigned)-shift;
        if(right>4)mantissa=0;
        else {
            mantissa=coefficient>>right;
            unsigned remainder=coefficient&((1u<<right)-1),half=1u<<(right-1);
            if(remainder>half||(remainder==half&&(mantissa&1)))mantissa++;
        }
    }
    return (uint16_t)(sign|mantissa);
}
int bn_quant_e4m3_ue8m0_to_fp16(uint8_t weight,uint8_t scale,uint16_t *out) {
    if(!out||(weight&127u)==127u||scale==255)return -1;
    *out=decode_finite(weight,scale);return 0;
}
static int disjoint(const void *a,size_t n,const void *b,size_t m) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(n>UINTPTR_MAX-x||m>UINTPTR_MAX-y)return 0;
    return x+n<=y||y+m<=x;
}
int bn_quant_e4m3_ue8m0_decode_region_fp16(const void *weights,size_t weight_bytes,
    const void *scales,size_t scale_bytes,int rows,int cols,
    int row,int col,int count_rows,int count_cols,uint16_t *out,size_t out_elems) {
    if(!weights||!scales||!out||rows<=0||cols<=0||row<0||col<0||
        row>=rows||col>=cols||count_rows<=0||count_cols<=0||
        count_rows>rows-row||count_cols>cols-col)return -1;
    size_t r=(size_t)rows,c=(size_t)cols,nr=(size_t)count_rows,nc=(size_t)count_cols;
    size_t sr=(r+31)/32,sc=(c+31)/32;
    if(r>SIZE_MAX/c||weight_bytes!=r*c||sr>SIZE_MAX/sc||scale_bytes!=sr*sc||
        nr>SIZE_MAX/nc||nr*nc>SIZE_MAX/sizeof(*out)||out_elems<nr*nc||
        !disjoint(weights,weight_bytes,out,nr*nc*sizeof(*out))||
        !disjoint(scales,scale_bytes,out,nr*nc*sizeof(*out)))return -1;
    const uint8_t *w=weights,*s=scales;
    for(size_t i=0;i<nr;i++)for(size_t j=0;j<nc;j++) {
        size_t rr=(size_t)row+i,cc=(size_t)col+j;
        if((w[rr*c+cc]&127u)==127u||s[(rr/32)*sc+cc/32]==255)return -1;
    }
    for(size_t i=0;i<nr;i++)for(size_t j=0;j<nc;j++) {
        size_t rr=(size_t)row+i,cc=(size_t)col+j;
        out[i*nc+j]=decode_finite(w[rr*c+cc],s[(rr/32)*sc+cc/32]);
    }
    return 0;
}
