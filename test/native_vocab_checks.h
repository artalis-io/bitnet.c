#include "native_vocab_reference.h"
static char *native_vocab_replace(const char *s,const char *from,const char *to) {
    const char *at=strstr(s,from);assert(at);size_t head=(size_t)(at-s),tail=strlen(at+strlen(from));
    char *out=malloc(head+strlen(to)+tail+1);assert(out);memcpy(out,s,head);memcpy(out+head,to,strlen(to));memcpy(out+head+strlen(to),at+strlen(from),tail+1);return out;
}
static void test_native_vocab(void) {
    enum {N=12};BnJSON doc={0};assert(!bn_json_init(&doc,native_vocab_fixture_json,strlen(native_vocab_fixture_json)));
    BnTokenizerVocab v={0};size_t budget=0;assert(!bn_tokenizer_vocab_plan(N,&budget)&&budget);
    assert(bn_tokenizer_vocab_plan(0,&budget)==-1&&bn_tokenizer_vocab_plan(N,NULL)==-1);
    assert(bn_tokenizer_vocab_init_json(&v,&doc,N,budget-1)==-1&&!v.state);
    assert(!bn_tokenizer_vocab_init_json(&v,&doc,N,budget)&&bn_tokenizer_vocab_size(&v)==N);void *state=v.state;
    assert(bn_tokenizer_vocab_init_json(&v,&doc,N,budget)==-1&&v.state==state);
    for(int i=0;i<N;i++) {
        const NativeVocabReference *r=native_vocab_reference+i;const char *raw=bn_tokenizer_vocab_raw(&v,i);
        assert(raw&&strlen(raw)==r->raw_bytes&&!memcmp(raw,r->raw,r->raw_bytes));
        BnTokenizerVocabDecodeInfo info={SIZE_MAX,-1};assert(!bn_tokenizer_vocab_decode_info(&v,i,&info)&&info.bytes==r->decoded_bytes&&info.contains_replacement==r->replacement);
        _Alignas(size_t) unsigned char out[128];unsigned char saved[128];memset(out,0xa5,sizeof(out));size_t written=SIZE_MAX;
        assert(!bn_tokenizer_vocab_decode(&v,i,out+1,sizeof(out)-2,&written)&&written==r->decoded_bytes&&!memcmp(out+1,r->decoded,written));
        assert(out[0]==0xa5);for(size_t j=written+1;j<sizeof(out);j++)assert(out[j]==0xa5);
        memcpy(saved,out,sizeof(out));written=SIZE_MAX;
        if(info.bytes)assert(bn_tokenizer_vocab_decode(&v,i,out+1,info.bytes-1,&written)==-1&&written==SIZE_MAX&&!memcmp(out,saved,sizeof(out)));
        else assert(!bn_tokenizer_vocab_decode(&v,i,NULL,0,&written)&&written==0);
        written=SIZE_MAX;
        if(info.bytes)assert(bn_tokenizer_vocab_decode(&v,i,(unsigned char*)raw,info.bytes,&written)==-1&&written==SIZE_MAX&&!memcmp(raw,r->raw,r->raw_bytes));
        if(info.bytes)assert(bn_tokenizer_vocab_decode(&v,i,out+1,sizeof(out)-2,(size_t*)(void*)out)==-1&&!memcmp(out,saved,sizeof(out)));
    }
    BnTokenizerVocabDecodeInfo info={123,7};assert(bn_tokenizer_vocab_decode_info(&v,-1,&info)==-1&&info.bytes==123&&info.contains_replacement==7);
    assert(!bn_tokenizer_vocab_raw(&v,N)&&!bn_tokenizer_vocab_raw(NULL,0));size_t written=99;unsigned char byte=0xa5;
    assert(bn_tokenizer_vocab_decode(&v,N,&byte,1,&written)==-1&&byte==0xa5&&written==99);
    bn_tokenizer_vocab_free(&v);bn_tokenizer_vocab_free(&v);assert(!v.state&&!bn_tokenizer_vocab_size(&v));
    assert(bn_tokenizer_vocab_init_json(&v,&doc,N+1,budget+128)==-1&&!v.state);
    bn_json_free(&doc);
    const char *bad[][2]={{"ByteLevel","Metaspace"},{"\\u00c3\":1","\\u00c3\":0"},{"\"id\":11","\"id\":12"},{"\"id\":10","\"id\":-1"}};
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++) {
        char *text=native_vocab_replace(native_vocab_fixture_json,bad[i][0],bad[i][1]);assert(!bn_json_init(&doc,text,strlen(text)));free(text);
        const BnJSONValue *root=bn_json_root(&doc);assert(bn_tokenizer_vocab_init_json(&v,&doc,N,budget)==-1&&!v.state&&bn_json_root(&doc)==root);bn_json_free(&doc);
    }
    const char *duplicates[]={
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0}},\"decoder\":{\"type\":\"ByteLevel\"},\"added_tokens\":[{\"id\":0,\"content\":\"a\"},{\"id\":0,\"content\":\"a\"}]}",
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0}},\"decoder\":{\"type\":\"ByteLevel\"},\"added_tokens\":[{\"id\":0,\"content\":\"different\"}]}"};
    for(size_t i=0;i<2;i++){assert(!bn_json_init(&doc,duplicates[i],strlen(duplicates[i])));assert(bn_tokenizer_vocab_init_json(&v,&doc,1,budget)==-1&&!v.state);bn_json_free(&doc);}
    puts("Native ByteLevel vocabulary: source UTF-8/NUL/replacement, budgets and atomic setup PASSED");
}
