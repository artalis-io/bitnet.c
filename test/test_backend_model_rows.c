#include "backend_model.h"
#include "trellis.h"
#include "quant.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __EMSCRIPTEN__
#include <unistd.h>

#ifdef BN_TEST_ROW_ALLOC
static int deny_allocations;
static size_t denied_allocations;
void *__real_malloc(size_t);
void *__real_calloc(size_t,size_t);
void *__real_realloc(void *,size_t);
void *__wrap_malloc(size_t n){if(deny_allocations){denied_allocations++;return NULL;}return __real_malloc(n);}
void *__wrap_calloc(size_t n,size_t m){if(deny_allocations){denied_allocations++;return NULL;}return __real_calloc(n,m);}
void *__wrap_realloc(void *p,size_t n){if(deny_allocations){denied_allocations++;return NULL;}return __real_realloc(p,n);}
#endif

enum { ROWS=257,COLS=64,WB=32,SB=4 };
static void tensor_file(const char *dir,const char *file,const char *name,
    const char *dtype,int rows,int cols,int scale) {
    char path[512],header[512];snprintf(path,sizeof(path),"%s/%s",dir,file);
    size_t bytes=(size_t)rows*cols*(scale?2:1);
    int n=snprintf(header,sizeof(header),"{\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],\"data_offsets\":[0,%zu]}}",name,dtype,rows,cols,bytes);
    FILE *f=fopen(path,"wb");assert(f);
    for(int i=0;i<8;i++)assert(fputc((int)(((uint64_t)n>>(8*i))&255),f)!=EOF);
    assert(fwrite(header,1,(size_t)n,f)==(size_t)n);
    for(int row=0;row<rows;row++)for(int col=0;col<cols;col++) {
        if(scale){unsigned v=0x3000u+(unsigned)(row*7+col*19)%1024;assert(fputc((int)(v&255),f)!=EOF&&fputc((int)(v>>8),f)!=EOF);}
        else assert(fputc((row*37+col*19+11)&255,f)!=EOF);
    }
    assert(!fclose(f));
}
static void fixture(const char *dir,int valid) {
    tensor_file(dir,"w.safetensors","packed","U8",ROWS,WB,0);
    tensor_file(dir,"s.safetensors","scales",valid?"F16":"U16",ROWS,COLS/32,1);
    char path[512];snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);
    FILE *f=fopen(path,"w");assert(f);
    assert(fprintf(f,"{\"metadata\":{\"total_size\":%d},\"weight_map\":{\"packed\":\"w.safetensors\",\"scales\":\"s.safetensors\"}}",ROWS*(WB+SB))>0);assert(!fclose(f));
}
static void expected(uint64_t row,unsigned char *w,unsigned char *s) {
    for(int j=0;j<WB;j++)w[j]=(unsigned char)(row*37+(unsigned)j*19+11);
    for(int j=0;j<COLS/32;j++) {
        unsigned v=0x3000u+(unsigned)(row*7+(unsigned)j*19)%1024;
        s[2*j]=(unsigned char)v;s[2*j+1]=(unsigned char)(v>>8);
    }
}
static void check(const uint64_t *ids,size_t n,const unsigned char *w,const unsigned char *s) {
    unsigned char ew[WB],es[SB];
    for(size_t i=0;i<n;i++){expected(ids[i],ew,es);assert(!memcmp(w+i*WB,ew,WB)&&!memcmp(s+i*SB,es,SB));}
}
static size_t minimal(BnBackendRowCache *cache,const BnTrellisRowSource *source) {
    for(size_t budget=1;budget<4096;budget++)if(!bn_backend_model_row_cache_init(cache,source,budget,8)) {
        BnBackendRowCacheStats stats;assert(!bn_backend_model_row_cache_stats(cache,&stats));assert(stats.capacity==1&&stats.allocated_bytes==budget);return budget;
    }
    assert(0);return 0;
}
static void churn(const BnTrellisRowSource *source) {
    BnBackendRowCache cache={0};assert(!bn_backend_model_row_cache_init(&cache,source,4096,8));
    BnBackendRowCacheStats stats;assert(!bn_backend_model_row_cache_stats(&cache,&stats));
    assert(stats.allocated_bytes<=4096&&stats.capacity>1&&stats.capacity<ROWS);
    uint64_t history[ROWS],hits=0,misses=0,evictions=0;size_t used=0;
    unsigned char w[8*WB+2],s[8*SB+2];
    for(size_t call=0;call<1500;call++) {
        size_t n=call%8+1;uint64_t ids[8];
        for(size_t i=0;i<n;i++) {
            ids[i]=(call*79+i*131+(call%3==0?call:0))%ROWS;
            if(i&&call%4==0)ids[i]=ids[i-1];
            size_t found=0;while(found<used&&history[found]!=ids[i])found++;
            if(found<used)hits++;
            else {misses++;if(used==stats.capacity){evictions++;found=used-1;}else found=used++;}
            for(size_t j=found;j>0;j--)history[j]=history[j-1];
            history[0]=ids[i];
        }
        memset(w,0xa5,sizeof(w));memset(s,0xa5,sizeof(s));
        assert(!bn_backend_model_row_cache_gather(&cache,ids,n,w+1,n*WB,s+1,n*SB));check(ids,n,w+1,s+1);
        assert(w[0]==0xa5&&w[n*WB+1]==0xa5&&s[0]==0xa5&&s[n*SB+1]==0xa5);
        assert(!bn_backend_model_row_cache_stats(&cache,&stats));
        assert(stats.hits==hits&&stats.misses==misses&&stats.evictions==evictions&&stats.resident_rows==used);
        assert(stats.source_bytes_requested==misses*(WB+SB)&&!stats.io_failures&&stats.source_read_ms>=0);
    }
    bn_backend_model_row_cache_free(&cache);bn_backend_model_row_cache_free(&cache);
}
static void batch_reuse(const BnTrellisRowSource *source) {
    BnBackendRowCache cache={0};minimal(&cache,source);
    const uint64_t ids[]={0,1,2,0,1,2,1,0};unsigned char w[8*WB],s[8*SB];
#ifdef BN_TEST_ROW_ALLOC
    deny_allocations=1;denied_allocations=0;
#endif
    assert(!bn_backend_model_row_cache_gather(&cache,ids,8,w,sizeof(w),s,sizeof(s)));
#ifdef BN_TEST_ROW_ALLOC
    deny_allocations=0;assert(!denied_allocations);
#endif
    check(ids,8,w,s);BnBackendRowCacheStats stats;
    assert(!bn_backend_model_row_cache_stats(&cache,&stats));
    assert(stats.capacity==1&&!stats.hits&&stats.misses==8&&stats.staging_hits==5);
    assert(stats.evictions==7&&stats.source_bytes_requested==3*(WB+SB));
    assert(!bn_backend_model_row_cache_gather(&cache,ids,1,w,WB,s,SB));
    assert(!bn_backend_model_row_cache_stats(&cache,&stats)&&stats.hits==1);
    bn_backend_model_row_cache_free(&cache);
#ifdef BN_TEST_ROW_ALLOC
    deny_allocations=1;denied_allocations=0;
    assert(bn_backend_model_row_cache_init(&cache,source,4096,8)==-1&&!cache.state);
    deny_allocations=0;assert(denied_allocations==1);
#endif
}
int main(void) {
    char dir[]="/tmp/bn-row-cache-XXXXXX";assert(mkdtemp(dir));fixture(dir,1);
    char path[512];snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);
    BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    int w_id=bn_safetensors_index_find(&index,"packed"),s_id=bn_safetensors_index_find(&index,"scales");assert(w_id>=0&&s_id>=0);
    size_t ids[2]={(size_t)w_id,(size_t)s_id};BnTrellisRowSource source={0};
    assert(!bn_trellis_row_source_init(&source,&index,BN_QUANT_FORMAT_INT4_FP16S32,ids));
    const BnTrellisRowLayout *l=bn_trellis_row_source_layout(&source);assert(l&&l->cols==COLS&&l->rows==ROWS&&l->weight_bytes==WB&&l->scale_bytes==SB&&l->bundle_bytes==WB+SB);
    BnTrellisRowSource bad={0};assert(bn_trellis_row_source_init(&bad,&index,BN_GGUF_TENSOR_Q4_0,ids)==-1&&!bad.state);
    size_t duplicate[2]={ids[0],ids[0]};assert(bn_trellis_row_source_init(&bad,&index,BN_QUANT_FORMAT_INT4_FP16S32,duplicate)==-1);
    duplicate[1]=SIZE_MAX;assert(bn_trellis_row_source_init(&bad,&index,BN_QUANT_FORMAT_INT4_FP16S32,duplicate)==-1);
    assert(bn_trellis_row_source_init(&source,&index,BN_QUANT_FORMAT_INT4_FP16S32,ids)==-1);
    unsigned char bundle[WB+SB+2],ew[WB],es[SB];memset(bundle,0xa5,sizeof(bundle));
    assert(!bn_trellis_row_source_read(&source,ROWS-1,bundle+1,WB+SB));expected(ROWS-1,ew,es);
    assert(!memcmp(bundle+1,ew,WB)&&!memcmp(bundle+1+WB,es,SB)&&bundle[0]==0xa5&&bundle[WB+SB+1]==0xa5);
    memset(bundle,0xa5,sizeof(bundle));assert(bn_trellis_row_source_read(&source,ROWS,bundle+1,WB+SB)==-1);
    assert(bn_trellis_row_source_read(&source,0,bundle+1,WB+SB-1)==-1);
    for(size_t i=0;i<sizeof(bundle);i++)assert(bundle[i]==0xa5);
    churn(&source);batch_reuse(&source);
#ifdef BN_TEST_ROW_ALLOC
    deny_allocations=1;denied_allocations=0;
    assert(bn_trellis_row_source_init(&bad,&index,BN_QUANT_FORMAT_INT4_FP16S32,ids)==-1&&!bad.state);
    deny_allocations=0;assert(denied_allocations==1);
#endif
    BnBackendRowCache cache={0};assert(bn_backend_model_row_cache_init(&cache,&source,0,8)==-1&&!cache.state);
    assert(bn_backend_model_row_cache_init(&cache,&source,SIZE_MAX,SIZE_MAX)==-1&&!cache.state);
    size_t budget=minimal(&cache,&source);assert(bn_backend_model_row_cache_init(&cache,&source,budget,8)==-1);
    BnBackendRowCacheStats before,after;assert(!bn_backend_model_row_cache_stats(&cache,&before));
    unsigned char w[2*WB],s[2*SB];memset(w,0xa5,sizeof(w));memset(s,0xa5,sizeof(s));
    uint64_t requested[2]={0,ROWS};
    assert(bn_backend_model_row_cache_gather(&cache,requested,2,w,sizeof(w),s,sizeof(s))==-1);
    assert(bn_backend_model_row_cache_gather(&cache,requested,9,w,sizeof(w),s,sizeof(s))==-1);
    assert(bn_backend_model_row_cache_gather(&cache,requested,1,w,WB-1,s,SB)==-1);
    assert(bn_backend_model_row_cache_gather(&cache,requested,1,w,WB,w,SB)==-1);
    assert(bn_backend_model_row_cache_gather(&cache,requested,1,requested,WB,s,SB)==-1);
    assert(bn_backend_model_row_cache_gather(&cache,requested,1,cache.state,WB,s,SB)==-1);
    assert(!bn_backend_model_row_cache_gather(&cache,NULL,0,NULL,0,NULL,0));
    assert(!bn_backend_model_row_cache_stats(&cache,&after)&&!after.misses&&!after.hits&&!after.source_bytes_requested);
    for(size_t i=0;i<sizeof(w);i++)assert(w[i]==0xa5);
    for(size_t i=0;i<sizeof(s);i++)assert(s[i]==0xa5);
    assert(!bn_backend_model_row_cache_gather(&cache,requested,1,w,WB,s,SB));check(requested,1,w,s);
    /* A short read in the second plane must preserve the cache victim and all
     * caller output, even when an earlier row in this gather was a hit. */
    snprintf(path,sizeof(path),"%s/s.safetensors",dir);FILE *f=fopen(path,"wb");assert(f&&!fclose(f));
    requested[1]=ROWS-1;memset(w,0xa5,sizeof(w));memset(s,0xa5,sizeof(s));
    assert(bn_backend_model_row_cache_gather(&cache,requested,2,w,sizeof(w),s,sizeof(s))==-1);
    for(size_t i=0;i<sizeof(w);i++)assert(w[i]==0xa5);
    for(size_t i=0;i<sizeof(s);i++)assert(s[i]==0xa5);
    assert(!bn_backend_model_row_cache_stats(&cache,&after)&&after.io_failures==1&&!after.evictions&&after.resident_rows==1);
    assert(!bn_backend_model_row_cache_gather(&cache,requested,1,w,WB,s,SB));check(requested,1,w,s);
    bn_trellis_row_source_free(&source);assert(bn_backend_model_row_cache_gather(&cache,requested,1,w,WB,s,SB)==-1);
    bn_backend_model_row_cache_free(&cache);bn_safetensors_index_free(&index);
    fixture(dir,0);snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!bn_safetensors_index_init(&index,path));
    assert(bn_trellis_row_source_init(&bad,&index,BN_QUANT_FORMAT_INT4_FP16S32,ids)==-1&&!bad.state);bn_safetensors_index_free(&index);
    assert(!unlink(path));snprintf(path,sizeof(path),"%s/w.safetensors",dir);assert(!unlink(path));snprintf(path,sizeof(path),"%s/s.safetensors",dir);assert(!unlink(path)&&!rmdir(dir));
    puts("Bounded backend row cache: cross-shard reads, LRU collisions, budgets, atomic I/O failure PASSED");return 0;
}
#else
int main(void){puts("Backend row-cache file fixtures skipped on WASM");return 0;}
#endif
