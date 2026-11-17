#ifndef __EMSCRIPTEN__
#include "native_engram_layout_reference.h"
static char *native_hash_replace(char *source,const char *from,const char *to) {
    char *result=native_numerics_replace(source,from,to);free(source);return result;
}
static char *native_hash_layer_bits(char *config) {
    const char *start=strstr(config,"\"layer_bits\":{");assert(start);const char *end=strchr(start,'}');assert(end);
    size_t n=(size_t)(end-start+1);char *old=malloc(n+1);assert(old);memcpy(old,start,n);old[n]=0;
    char replacement[2048];size_t used=(size_t)snprintf(replacement,sizeof(replacement),"\"layer_bits\":{");
    for(int i=0;i<128;i++){int written=snprintf(replacement+used,sizeof(replacement)-used,"%s\"%d\":3",i?",":"",i);assert(written>0&&(size_t)written<sizeof(replacement)-used);used+=(size_t)written;}
    assert(used+2<=sizeof(replacement));replacement[used++]='}';replacement[used]=0;
    char *result=native_hash_replace(config,old,replacement);free(old);return result;
}
static void native_hash_check(const char *config,int layer,int accepted,BnModelNativeEngramLayout *out) {
    BnJSON doc={0};assert(!bn_json_init(&doc,config,strlen(config)));BnModelNativeEngramLayout saved;
    memset(&saved,0xa5,sizeof(saved));*out=saved;int rc=bn_model_arch_native_engram_layout(&doc,layer,out);
    if(accepted?rc:rc!=-1||memcmp(out,&saved,sizeof(saved)))fprintf(stderr,"Engram layout unexpected layer%d accepted%d rc%d config%s\n",layer,accepted,rc,config);
    assert(accepted?!rc:rc==-1&&!memcmp(out,&saved,sizeof(saved)));bn_json_free(&doc);
}
static void test_native_engram_layout(void) {
    char *config=native_numerics_replace(native_anatomy_config,"\"vocab_size\":256","\"vocab_size\":129280");
    config=native_hash_replace(config,"\"engram_n_heads\":2","\"engram_n_heads\":8");
    config=native_hash_replace(config,"\"engram_num_embeddings\":[64,96]","\"engram_num_embeddings\":[384006168,384016682]");
    BnModelNativeEngramLayout p;
    for(int li=0;li<2;li++) {
        native_hash_check(config,li?14:1,1,&p);assert(p.ngram==4&&p.heads==24&&p.heads_per_ngram==8&&p.compressed_vocab==99092&&p.pad_token==2);
        assert(!memcmp(p.sizes,native_hash_ref_sizes[li],24*8)&&!memcmp(p.offsets,native_hash_ref_offsets[li],24*8)&&!memcmp(p.multipliers,native_hash_ref_multipliers[li],4*8));
    }
    const char *changes[][2]={{"384016682","384016683"},{"[1,14]","[1,1]"},{"[1,14]","[1,40]"},{"\"engram_max_ngram_size\":4","\"engram_max_ngram_size\":17"},{"\"engram_n_heads\":8","\"engram_n_heads\":128"}};
    for(size_t i=0;i<sizeof(changes)/sizeof(changes[0]);i++){char *bad=native_numerics_replace(config,changes[i][0],changes[i][1]);native_hash_check(bad,1,0,&p);free(bad);}
    native_hash_check(config,-1,0,&p);native_hash_check(config,0,0,&p);native_hash_check(config,40,0,&p);
    char *reversed=native_numerics_replace(config,"[1,14]","[14,1]");native_hash_check(reversed,1,1,&p);
    assert(!memcmp(p.sizes,native_hash_ref_sizes[1],24*8)&&!memcmp(p.multipliers,native_hash_ref_multipliers[0],4*8));free(reversed);
    free(config);
    /* Exercise both 64-bit bounded draws and NumPy's cached 32-bit path. */
    for(int v=0;v<2;v++)for(int layer=0;layer<128;layer++) {
        char rows[128],ids[64],vocab[128];
        snprintf(rows,sizeof(rows),"\"engram_num_embeddings\":[%llu]",(unsigned long long)native_hash_ref_first15_rows);
        snprintf(ids,sizeof(ids),"\"engram_layer_ids\":[%d]",layer);
        snprintf(vocab,sizeof(vocab),"\"engram_compressed_vocab_size\":%d",native_hash_ref_vocab[v]);
        config=native_numerics_replace(native_anatomy_config,"\"vocab_size\":256","\"vocab_size\":2147483647");
        config=native_hash_replace(config,"\"num_hidden_layers\":40","\"num_hidden_layers\":128");
        config=native_hash_layer_bits(config);
        config=native_hash_replace(config,"\"engram_n_heads\":2","\"engram_n_heads\":1");
        config=native_hash_replace(config,"\"engram_max_ngram_size\":4","\"engram_max_ngram_size\":16");
        config=native_hash_replace(config,"\"engram_layer_ids\":[1,14]",ids);
        config=native_hash_replace(config,"\"engram_num_embeddings\":[64,96]",rows);
        config=native_hash_replace(config,"\"engram_compressed_vocab_size\":99092",vocab);
        native_hash_check(config,layer,1,&p);assert(p.rows==native_hash_ref_first15_rows&&p.heads==15&&p.ngram==16);
        assert(!memcmp(p.multipliers,native_hash_ref_draws[v][layer],16*8));free(config);
    }
    assert(bn_model_native_engram_layout(NULL,1,&p)==-1);
    puts("Native Engram layout: source buckets/order and4096 seeded multiplier words PASSED");
}
#else
static void test_native_engram_layout(void) {}
#endif
