#include "quant.h"

/* Exact integer evaluation of the Diffbot int4_fp16s32 lookup profile.
 * FP16 scale times integer [-8,7] has at most14 significant bits and is
 * exactly representable in F32. Nonzero results are normal in BF16, so
 * rounding its integer significand once gives the source F32->BF16 result. */
static uint16_t decode(uint8_t nibble,uint16_t scale) {
    int q=(int)nibble-8;
    uint16_t sign=q<0?0x8000u:0;
    unsigned exponent=scale>>10;
    unsigned coefficient=((scale&1023u)+(exponent?1024u:0u))*(unsigned)(q<0?-q:q);
    if(!coefficient)return sign;
    int power=exponent?(int)exponent-25:-24;
    unsigned leading=0;
    for(unsigned v=coefficient;v>1;v>>=1)leading++;
    int top=power+(int)leading;
    unsigned significand;
    if(leading<=7)significand=coefficient<<(7-leading);
    else {
        unsigned shift=leading-7;
        significand=coefficient>>shift;
        unsigned remainder=coefficient&((1u<<shift)-1),half=1u<<(shift-1);
        if(remainder>half||(remainder==half&&(significand&1)))significand++;
        if(significand==256){significand=128;top++;}
    }
    return (uint16_t)(sign|((unsigned)(top+127)<<7)|(significand-128));
}

int bn_quant_int4_fp16s32_to_bf16(uint8_t nibble,uint16_t scale,uint16_t *out) {
    if(!out||nibble>15||scale>=0x7c00u)return -1;
    *out=decode(nibble,scale);return 0;
}

static int disjoint(const void *a,size_t n,const void *b,size_t m) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(n>UINTPTR_MAX-x||m>UINTPTR_MAX-y)return 0;
    return x+n<=y||y+m<=x;
}

static uint16_t scale_at(const uint8_t *s,size_t i) {
    return (uint16_t)((unsigned)s[2*i]|((unsigned)s[2*i+1]<<8));
}
int bn_quant_int4_fp16s32_validate_scales(const void *scales,size_t bytes) {
    if(bytes%2||(!scales&&bytes))return -1;
    const uint8_t *s=scales;
    for(size_t i=0;i<bytes/2;i++)if(scale_at(s,i)>=0x7c00u)return -1;
    return 0;
}

int bn_quant_int4_fp16s32_decode_region_bf16(const void *weights,size_t weight_bytes,
    const void *scales,size_t scale_bytes,int rows,int cols,
    int row,int col,int count_rows,int count_cols,uint16_t *out,size_t out_elems) {
    if(!weights||!scales||!out||rows<=0||cols<=0||cols%32||row<0||col<0||
        row>=rows||col>=cols||count_rows<=0||count_cols<=0||
        count_rows>rows-row||count_cols>cols-col)return -1;
    size_t r=(size_t)rows,c=(size_t)cols,nr=(size_t)count_rows,nc=(size_t)count_cols;
    if(r>SIZE_MAX/(c/2)||weight_bytes!=r*(c/2)||r>SIZE_MAX/(c/16)||
        scale_bytes!=r*(c/16)||nr>SIZE_MAX/nc||nr*nc>SIZE_MAX/sizeof(*out)||
        out_elems<nr*nc||!disjoint(weights,weight_bytes,out,nr*nc*sizeof(*out))||
        !disjoint(scales,scale_bytes,out,nr*nc*sizeof(*out)))return -1;
    const uint8_t *w=weights,*s=scales;
    size_t first=(size_t)col/32,last=((size_t)col+nc-1)/32;
    for(size_t i=0;i<nr;i++)for(size_t j=first;j<=last;j++)
        if(scale_at(s,((size_t)row+i)*(c/32)+j)>=0x7c00u)return -1;
    for(size_t i=0;i<nr;i++)for(size_t j=0;j<nc;j++) {
        size_t rr=(size_t)row+i,cc=(size_t)col+j;
        uint8_t packed=w[rr*(c/2)+cc/2];
        uint8_t nibble=(uint8_t)((packed>>((cc%2)*4))&15u);
        out[i*nc+j]=decode(nibble,scale_at(s,rr*(c/32)+cc/32));
    }
    return 0;
}

int bn_quant_int4_fp16s32_validate_f16(const void *weights,size_t weight_bytes,
    const void *scales,size_t scale_bytes,int rows,int cols) {
    if(!weights||!scales||rows<1||cols<32||cols%32||
       (size_t)rows>SIZE_MAX/((size_t)cols/2)||weight_bytes!=(size_t)rows*((size_t)cols/2)||
       (size_t)rows>SIZE_MAX/((size_t)cols/16)||scale_bytes!=(size_t)rows*((size_t)cols/16))return -1;
    const uint8_t *w=weights,*s=scales;
    for(size_t block=0;block<scale_bytes/2;block++) {
        uint16_t scale=scale_at(s,block);
        if(scale>=0x7c00u)return -1;
        /* Common small scales prove the entire block safe without decoding.
         * Larger scales require actual nibbles, including valid all-zero rows. */
        if(scale<=0x6c00u)continue;
        for(size_t col=0;col<32;col++) {
            uint8_t nibble=(uint8_t)((w[block*16+col/2]>>((col%2)*4))&15u);
            if((decode(nibble,scale)&0x7fffu)>=0x4780u)return -1;
        }
    }
    return 0;
}
