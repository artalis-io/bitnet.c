#ifndef __EMSCRIPTEN__
#include "native_ngram_rows_reference.h"
static BnModelNativeEngramLayout native_ngram_test_layout(int layer) {
    int li=layer==14;BnModelNativeEngramLayout p={0};p.layer=layer;p.ngram=4;p.heads_per_ngram=8;p.heads=24;p.compressed_vocab=99092;p.pad_token=2;
    memcpy(p.sizes,native_hash_ref_sizes[li],24*8);memcpy(p.offsets,native_hash_ref_offsets[li],24*8);memcpy(p.multipliers,native_hash_ref_multipliers[li],4*8);
    p.rows=p.offsets[23]+p.sizes[23];return p;
}
static void test_native_ngram_rows(void) {
    for(size_t ci=0;ci<sizeof(native_ngram_reference)/sizeof(native_ngram_reference[0]);ci++) {
        const NativeNGramReference *c=native_ngram_reference+ci;BnModelNativeEngramLayout p=native_ngram_test_layout(c->layer);
        size_t count=c->batch*c->length*24,hc=c->batch*(c->length+3);uint64_t *out=malloc((count+2)*8),*saved=malloc(count*8);
        uint32_t *history=malloc(hc*4);assert(out&&saved&&history);memcpy(history,c->history,hc*4);
        for(size_t i=0;i<count+2;i++)out[i]=UINT64_C(0xa5a5a5a5a5a5a5a5);
        assert(!bn_model_native_engram_hash_rows(&p,2,history,c->batch,c->length+3,c->length,out+1,count));
        assert(out[0]==UINT64_C(0xa5a5a5a5a5a5a5a5)&&out[count+1]==out[0]&&!memcmp(out+1,c->rows,count*8));memcpy(saved,out+1,count*8);
        /* Replay each token with carried source context, including pad positions. */
        for(size_t b=0;b<c->batch;b++)for(size_t t=0;t<c->length;t++) {
            uint64_t one[24];assert(!bn_model_native_engram_hash_rows(&p,2,history+b*(c->length+3)+t,1,4,1,one,24));
            assert(!memcmp(one,c->rows+(b*c->length+t)*24,sizeof(one)));
        }
        uint32_t old=history[hc-1];history[hc-1]=(uint32_t)p.compressed_vocab;
        assert(bn_model_native_engram_hash_rows(&p,2,history,c->batch,c->length+3,c->length,out+1,count)==-1&&!memcmp(out+1,saved,count*8));history[hc-1]=old;
        assert(bn_model_native_engram_hash_rows(&p,2,history,c->batch,c->length+3,c->length,out+1,count-1)==-1&&!memcmp(out+1,saved,count*8));
        assert(bn_model_native_engram_hash_rows(&p,99092,history,c->batch,c->length+3,c->length,out+1,count)==-1&&!memcmp(out+1,saved,count*8));
        assert(bn_model_native_engram_hash_rows(&p,2,history,c->batch,c->length+3,c->length,(uint64_t*)history,count)==-1&&!memcmp(history,c->history,hc*4));
        BnModelNativeEngramLayout bad=p;bad.sizes[23]=0;
        assert(bn_model_native_engram_hash_rows(&bad,2,history,c->batch,c->length+3,c->length,out+1,count)==-1&&!memcmp(out+1,saved,count*8));
        bad=p;bad.multipliers[3]=UINT64_MAX;
        assert(bn_model_native_engram_hash_rows(&bad,2,history,c->batch,c->length+3,c->length,out+1,count)==-1&&!memcmp(out+1,saved,count*8));
        bad=p;bad.offsets[23]++;
        assert(bn_model_native_engram_hash_rows(&bad,2,history,c->batch,c->length+3,c->length,out+1,count)==-1&&!memcmp(out+1,saved,count*8));
        assert(bn_model_native_engram_hash_rows(&p,2,history,c->batch,c->length+3,c->length,(uint64_t*)&p,count)==-1);
        assert(bn_model_native_engram_hash_rows(&p,2,history,SIZE_MAX,4,1,out+1,count)==-1&&!memcmp(out+1,saved,count*8));
        assert(!bn_model_native_engram_hash_rows(&p,2,NULL,1,0,0,NULL,0));
        assert(!bn_model_native_engram_hash_rows(&p,2,history,c->batch,c->length+3,c->length,out+1,count)&&!memcmp(out+1,saved,count*8));
        free(history);free(saved);free(out);
    }
    puts("Native mapped n-gram rows: source pads, batching, carried context and atomic preflight PASSED");
}
#else
static void test_native_ngram_rows(void) {}
#endif
