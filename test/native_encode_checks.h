#include "native_encode_reference.h"
#include "native_encode_edges.h"
static void test_native_encode(void) {
    BnJSON doc={0};assert(!bn_json_init(&doc,native_encode_fixture,strlen(native_encode_fixture)));
    BnTokenizerEncoderPlan plan={0};assert(!bn_tokenizer_encoder_plan(&doc,315,&plan));
    BnTokenizerEncoder encoder={0};assert(bn_tokenizer_encoder_init_json(&encoder,&doc,315,plan.host_bytes-1)==-1&&!encoder.state);
    assert(!bn_tokenizer_encoder_init_json(&encoder,&doc,315,plan.host_bytes));void *identity=encoder.state;
    assert(bn_tokenizer_encoder_init_json(&encoder,&doc,315,plan.host_bytes)==-1&&encoder.state==identity);
    BnTokenizerEncodeScratch a={0},b={0};size_t sb=0;
    assert(!bn_tokenizer_encode_scratch_plan(128,&sb));
    assert(bn_tokenizer_encode_scratch_init(&a,128,sb-1)==-1&&!a.state);
    assert(!bn_tokenizer_encode_scratch_init(&a,128,sb)&&!bn_tokenizer_encode_scratch_init(&b,128,sb));
    assert(bn_tokenizer_encode_scratch_init(&a,128,sb)==-1);
    _Alignas(size_t) unsigned char storage[512],saved[512];int *out=(int*)(void*)storage;
    for(int run=0;run<8;run++)for(size_t i=0;i<sizeof(native_encode_cases)/sizeof(native_encode_cases[0]);i++) {
        size_t count=SIZE_MAX;memset(storage,0xa5,sizeof(storage));
        assert(!bn_tokenizer_encoder_encode(&encoder,run&1?&a:&b,native_encode_cases[i].text,native_encode_cases[i].bytes,out+1,64,&count));
        assert(count==native_encode_cases[i].count&&!memcmp(out+1,native_encode_cases[i].ids,count*sizeof(int)));
        assert(out[0]==(int)UINT32_C(0xa5a5a5a5)&&out[count+1]==(int)UINT32_C(0xa5a5a5a5));
        if(count) {
            memcpy(saved,storage,sizeof(storage));size_t written=SIZE_MAX;
            assert(bn_tokenizer_encoder_encode(&encoder,&a,native_encode_cases[i].text,native_encode_cases[i].bytes,out+1,count-1,&written)==-1&&written==SIZE_MAX&&!memcmp(saved,storage,sizeof(storage)));
        }
        count=SIZE_MAX;assert(!bn_tokenizer_encoder_pretokenize(&encoder,native_encode_cases[i].text,native_encode_cases[i].bytes,(uint32_t*)(void*)(out+1),64,&count));
        assert(count==native_encode_cases[i].pieces&&!memcmp(out+1,native_encode_cases[i].ends,count*4));
        if(count) {
            memcpy(saved,storage,sizeof(storage));size_t written=SIZE_MAX;
            assert(bn_tokenizer_encoder_pretokenize(&encoder,native_encode_cases[i].text,native_encode_cases[i].bytes,(uint32_t*)(void*)(out+1),count-1,&written)==-1&&written==SIZE_MAX&&!memcmp(saved,storage,sizeof(storage)));
        }
    }
    for(size_t i=0;i<sizeof(native_encode_edges)/sizeof(native_encode_edges[0]);i++) {
        size_t count=SIZE_MAX;
        assert(!bn_tokenizer_encoder_encode(&encoder,&a,native_encode_edges[i].text,native_encode_edges[i].bytes,out,64,&count));
        assert(count==native_encode_edges[i].count&&!memcmp(out,native_encode_edges[i].ids,count*sizeof(int)));
        count=SIZE_MAX;
        assert(!bn_tokenizer_encoder_pretokenize(&encoder,native_encode_edges[i].text,native_encode_edges[i].bytes,(uint32_t*)(void*)out,64,&count));
        assert(count==native_encode_edges[i].pieces&&!memcmp(out,native_encode_edges[i].ends,count*sizeof(uint32_t)));
    }
    const unsigned char bad[][4]={{0xc0,0x80,0,0},{0xed,0xa0,0x80,0},{0xf4,0x90,0x80,0x80},{'a',0xe2,0x82,0}};
    const size_t lengths[]={2,3,4,3};
    for(int i=0;i<4;i++) {
        memset(storage,0xa5,sizeof(storage));memcpy(saved,storage,sizeof(storage));size_t n=SIZE_MAX;
        assert(bn_tokenizer_encoder_encode(&encoder,&a,bad[i],lengths[i],out,64,&n)==-1&&n==SIZE_MAX&&!memcmp(saved,storage,sizeof(storage)));
        assert(bn_tokenizer_encoder_pretokenize(&encoder,bad[i],lengths[i],(uint32_t*)(void*)out,64,&n)==-1&&n==SIZE_MAX&&!memcmp(saved,storage,sizeof(storage)));
    }
    size_t n=SIZE_MAX;const unsigned char text[]="abc";
    memset(storage,0xa5,sizeof(storage));memcpy(saved,storage,sizeof(storage));
    assert(bn_tokenizer_encoder_encode(&encoder,&a,text,3,out,64,(size_t*)(void*)storage)==-1&&!memcmp(saved,storage,sizeof(storage)));
    _Alignas(int) unsigned char alias[16]="abc";unsigned char original[16];memcpy(original,alias,sizeof(alias));
    assert(bn_tokenizer_encoder_encode(&encoder,&a,alias,3,(int*)(void*)alias,4,&n)==-1&&n==SIZE_MAX&&!memcmp(alias,original,sizeof(alias)));
    assert(bn_tokenizer_encoder_encode(&encoder,&a,text,3,(int*)a.state,1,&n)==-1&&n==SIZE_MAX);
    assert(bn_tokenizer_encoder_encode(&encoder,&a,text,3,(int*)encoder.state,1,&n)==-1&&n==SIZE_MAX);
    assert(bn_tokenizer_encoder_encode(&encoder,&a,text,3,NULL,1,&n)==-1&&n==SIZE_MAX);
    assert(bn_tokenizer_encoder_encode(&encoder,&a,text,3,out,SIZE_MAX,&n)==-1&&n==SIZE_MAX);
    assert(bn_tokenizer_encoder_encode(&encoder,&a,text,129,out,64,&n)==-1&&n==SIZE_MAX);
    assert(!bn_tokenizer_encoder_encode(&encoder,&a,NULL,0,NULL,0,&n)&&n==0);
    assert(bn_tokenizer_encode_scratch_plan(SIZE_MAX,&n)==-1&&n==0);
    bn_tokenizer_encode_scratch_free(&a);bn_tokenizer_encode_scratch_free(&a);bn_tokenizer_encode_scratch_free(&b);
    bn_tokenizer_encoder_free(&encoder);bn_tokenizer_encoder_free(&encoder);bn_json_free(&doc);
    const char *changes[][2]={
        {"\"byte_fallback\":false","\"byte_fallback\":true"},
        {"\"single_word\":false","\"single_word\":true"},
        {"\"lstrip\":false","\"lstrip\":true"},
        {"\"normalizers\":[]","\"normalizers\":[{\"type\":\"NFC\"}]"},
        {"\"use_regex\":false","\"use_regex\":true"},
        {"\"behavior\":\"Isolated\"","\"behavior\":\"Removed\""},
        {"\"a b\"","\"a z\""},
        {"\"a b\"","[1,\"2\"]"},
        {"\"a b\"","\"a b\",\"a b\""}};
    for(size_t i=0;i<sizeof(changes)/sizeof(changes[0]);i++) {
        char *json=native_vocab_replace(native_encode_fixture,changes[i][0],changes[i][1]);assert(!bn_json_init(&doc,json,strlen(json)));free(json);
        assert(bn_tokenizer_encoder_init_json(&encoder,&doc,315,1024*1024)==-1&&!encoder.state);bn_json_free(&doc);
    }
    char *json=native_vocab_replace(native_encode_fixture,"\"a b\"","[\"a\",\"b\"]");assert(!bn_json_init(&doc,json,strlen(json)));free(json);
    assert(!bn_tokenizer_encoder_init_json(&encoder,&doc,315,1024*1024));bn_tokenizer_encoder_free(&encoder);bn_json_free(&doc);
    puts("Native JSON BPE: source ranks/splits/added/NUL, budgets, failure atomicity and isolated workspace PASSED");
}
