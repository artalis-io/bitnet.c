#ifndef BN_BENCH_BF16_POLICY_IO_H
#define BN_BENCH_BF16_POLICY_IO_H

/* Benchmark wire format only. Runtime admission owns capability validation. */
#include "../src/transformer/bf16_internal.h"
#include <stdio.h>
#include <string.h>

static inline void bench_policy_space(const char **p) {
    while (**p==' ' || **p=='\n' || **p=='\r' || **p=='\t') ++*p;
}
static inline int bench_policy_char(const char **p, char c) {
    bench_policy_space(p);
    if (**p!=c) return -1;
    ++*p; return 0;
}
static inline int bench_policy_value(const char **p, int max, int *out) {
    bench_policy_space(p);
    if (**p<'0' || **p>'0'+max) return -1;
    *out=*(*p)++-'0'; return 0;
}
/* Atomic on failure, including when input aliases storage owned by caller. */
static inline int bench_policy_parse(const char *text, BnTransformerBF16Policy *out) {
    const char *p=text;
    unsigned seen=0;
    BnTransformerBF16Policy result={0};
    if (!text || !out || bench_policy_char(&p,'{')) return -1;
    for (int field=0;field<6;field++) {
        if (field && bench_policy_char(&p,',')) return -1;
        if (bench_policy_char(&p,'"')) return -1;
        const char *start=p;
        while (*p && *p!='"') ++p;
        size_t len=(size_t)(p-start);
        int key=-1;
        const char *names[]={"arithmetic","ffn_arithmetic","input_norm_reductions","post_norm_reductions","attention_split_policy","gdn_prefill_selection"};
        for (int i=0;i<6;i++)
            if (len==strlen(names[i]) && !memcmp(start,names[i],len)) key=i;
        if (key<0 || (seen&(1u<<key)) || bench_policy_char(&p,'"') || bench_policy_char(&p,':')) return -1;
        seen|=1u<<key;
        int value=0;
        if (key<2 || key>=4) {
            int max=key==0?BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32
                :key==1?BN_TRANSFORMER_BF16_FFN_COMPILED:key==4?BN_GPU_ATTENTION_SPLIT_BATCH:BN_TRANSFORMER_BF16_GDN_PREFILL_SINGLE_CP;
            if (bench_policy_value(&p,max,&value)) return -1;
            if (key==0) result.arithmetic=(BnTransformerBF16Arithmetic)value;
            else if (key==1) result.ffn_arithmetic=(BnTransformerBF16FFNArithmetic)value;
            else if (key==4) result.attention_split_policy=(BnGPUAttentionSplitPolicy)value;
            else result.gdn_prefill_selection=(BnTransformerBF16GDNPrefillSelection)value;
        } else {
            int count=key==2?65:64;
            BnGPURMSNormReduction *values=key==2?result.input_norm_reductions:result.post_norm_reductions;
            if (bench_policy_char(&p,'[')) return -1;
            for (int i=0;i<count;i++) {
                if ((i && bench_policy_char(&p,',')) ||
                    bench_policy_value(&p,BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES,&value)) return -1;
                values[i]=(BnGPURMSNormReduction)value;
            }
            if (bench_policy_char(&p,']')) return -1;
        }
        bench_policy_space(&p);
        if (*p=='}') break;
    }
    if ((seen&15)!=15 || bench_policy_char(&p,'}')) return -1;
    bench_policy_space(&p);
    if (*p) return -1;
    *out=result; return 0;
}
static inline int bench_policy_read(const char *path, BnTransformerBF16Policy *out) {
    /* Bounds even adversarial whitespace without allocating unbounded input. */
    char text[16385];
    FILE *f=fopen(path,"rb");
    if (!f) return -1;
    size_t n=fread(text,1,sizeof(text)-1,f);
    int failed=ferror(f) || n==sizeof(text)-1 || memchr(text,0,n)!=NULL;
    if (fclose(f)) failed=1;
    if (failed) return -1;
    text[n]=0; return bench_policy_parse(text,out);
}
static inline int bench_policy_write(FILE *f,const BnTransformerBF16Policy *p) {
    fprintf(f,"{\"arithmetic\":%d,\"ffn_arithmetic\":%d,\"input_norm_reductions\":[",p->arithmetic,p->ffn_arithmetic);
    for (int i=0;i<65;i++) fprintf(f,"%s%d",i?",":"",p->input_norm_reductions[i]);
    fputs("],\"post_norm_reductions\":[",f);
    for (int i=0;i<64;i++) fprintf(f,"%s%d",i?",":"",p->post_norm_reductions[i]);
    fputs("]",f);
    /* Preserve the existing four-field wire form for the legacy default. */
    if (p->attention_split_policy!=BN_GPU_ATTENTION_SPLIT_SINGLETON)
        fprintf(f,",\"attention_split_policy\":%d",p->attention_split_policy);
    if (p->gdn_prefill_selection!=BN_TRANSFORMER_BF16_GDN_PREFILL_DEFAULT)
        fprintf(f,",\"gdn_prefill_selection\":%d",p->gdn_prefill_selection);
    fputs("}",f); return ferror(f)?-1:0;
}
#endif
