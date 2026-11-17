#include "native_stream_reference.h"
#include "native_stream_error_reference.h"
#include "native_stream_fallback_reference.h"
static void native_stream_case_check(const BnTokenizerDecoder *decoder,void *identity,const int *ids,size_t count,int skip,
    const unsigned char *const *chunks,const size_t *lengths,const unsigned char (*pending)[4]) {
    _Alignas(size_t) unsigned char out[256],saved[256];
        BnTokenizerDecodeStream state={0};assert(!bn_tokenizer_decode_stream_init(&state,decoder,skip));
        for(size_t step=0;step<=count;step++) {
            size_t wanted=lengths[step],n=SIZE_MAX;int finish=step==count;
            BnTokenizerDecodeStream before=state;memset(out,0xa5,sizeof(out));memcpy(saved,out,sizeof(out));
            const int *step_ids=finish?NULL:ids+step;
            if(wanted)assert(bn_tokenizer_decode_stream_push(decoder,&state,step_ids,finish?0:1,finish,out+1,wanted-1,&n)==-1&&n==SIZE_MAX&&!memcmp(&state,&before,sizeof(state))&&!memcmp(out,saved,sizeof(out)));
            assert(!bn_tokenizer_decode_stream_push(decoder,&state,step_ids,finish?0:1,finish,out+1,wanted,&n));
            assert(n==wanted&&!memcmp(out+1,chunks[step],n)&&out[0]==0xa5&&out[n+1]==0xa5);
            assert(state.count==pending[step][0]&&!memcmp(state.pending,pending[step]+1,3)&&!state.pending[3]);
            assert(state.finished==finish);
        }
        size_t n=SIZE_MAX;BnTokenizerDecodeStream before=state;
        assert(bn_tokenizer_decode_stream_push(decoder,&state,NULL,0,0,out,sizeof(out),&n)==-1&&n==SIZE_MAX&&!memcmp(&state,&before,sizeof(state)));
        assert(!bn_tokenizer_decode_stream_reset(&state)&&!state.finished&&!state.count&&state.owner==identity);
        bn_tokenizer_decode_stream_free(&state);bn_tokenizer_decode_stream_free(&state);assert(!state.owner);
}
static void test_gguf_stream_bridge(void) {
    char *pieces[]={"Ã","©","Ā","<stop>","<user>"};
    BnTokenizer t={0};t.vocab=pieces;t.vocab_size=5;
    int32_t types[]={1,1,1,3,4};BnGGUFKeyValue kv={0};
    kv.key="tokenizer.ggml.token_type";kv.type=BN_GGUF_TYPE_ARRAY;
    kv.value.arr.elem_type=BN_GGUF_TYPE_INT32;kv.value.arr.n=5;kv.value.arr.data=types;
    BnGGUFFile f={0};f.n_kv=1;f.kvs=&kv;size_t budget;
    assert(!bn_tokenizer_decoder_plan(5,&budget));BnTokenizerDecoder d={0};
    assert(bn_tokenizer_decoder_init_gguf(&d,&t,&f,budget-1)==-1&&!d.state);
    assert(!bn_tokenizer_decoder_init_gguf(&d,&t,&f,budget));
    BnTokenizerDecodeStream a={0},b={0};assert(!bn_tokenizer_decode_stream_init(&a,&d,1));
    assert(!bn_tokenizer_decode_stream_init(&b,&d,0));unsigned char out[64];size_t n;int lead=0,user=4;
    assert(!bn_tokenizer_decode_stream_push(&d,&a,&lead,1,0,out,sizeof(out),&n)&&!n&&a.count==1);
    assert(!bn_tokenizer_decode_stream_push(&d,&b,&user,1,1,out,sizeof(out),&n)&&n==6&&!memcmp(out,"<user>",6));
    int tail[]={1,2,3,4};const unsigned char expected[]={195,169,0,'<','u','s','e','r','>'};
    assert(!bn_tokenizer_decode_stream_push(&d,&a,tail,4,1,out,sizeof(out),&n)&&n==sizeof(expected)&&!memcmp(out,expected,n));
    bn_tokenizer_decoder_free(&d);t.metaspace=1;
    assert(bn_tokenizer_decoder_init_gguf(&d,&t,&f,budget)==-1&&!d.state);t.metaspace=0;
    kv.value.arr.elem_type=BN_GGUF_TYPE_FLOAT32;
    assert(bn_tokenizer_decoder_init_gguf(&d,&t,&f,budget)==-1&&!d.state);
    f.n_kv=0;assert(!bn_tokenizer_decoder_init_gguf(&d,&t,&f,budget));
    a=(BnTokenizerDecodeStream){0};assert(!bn_tokenizer_decode_stream_init(&a,&d,1));int stop=3;
    assert(!bn_tokenizer_decode_stream_push(&d,&a,&stop,1,1,out,sizeof(out),&n)&&n==6&&!memcmp(out,"<stop>",6));
    bn_tokenizer_decoder_free(&d);puts("GGUF ByteLevel streams: private UTF8 carry/NUL/control versus user-defined policy PASSED");
}
static void test_native_stream(void) {
    test_gguf_stream_bridge();
    BnJSON doc={0};assert(!bn_json_init(&doc,native_stream_fixture,strlen(native_stream_fixture)));
    BnTokenizerDecoder decoder={0},other={0};size_t budget=0;assert(!bn_tokenizer_decoder_plan(261,&budget));
    assert(bn_tokenizer_decoder_init_json(&decoder,&doc,261,budget-1)==-1&&!decoder.state);
    assert(!bn_tokenizer_decoder_init_json(&decoder,&doc,261,budget));void *identity=decoder.state;
    assert(bn_tokenizer_decoder_init_json(&decoder,&doc,261,budget)==-1&&decoder.state==identity);
    assert(!bn_tokenizer_decoder_init_json(&other,&doc,261,budget));
    _Alignas(size_t) unsigned char out[256],saved[256];
    for(size_t i=0;i<sizeof(native_stream_cases)/sizeof(native_stream_cases[0]);i++)native_stream_case_check(&decoder,identity,native_stream_cases[i].ids,native_stream_cases[i].count,native_stream_cases[i].skip,native_stream_cases[i].chunks,native_stream_cases[i].lengths,native_stream_cases[i].pending);
    for(size_t i=0;i<sizeof(native_stream_error_cases)/sizeof(native_stream_error_cases[0]);i++)native_stream_case_check(&decoder,identity,native_stream_error_cases[i].ids,native_stream_error_cases[i].count,native_stream_error_cases[i].skip,native_stream_error_cases[i].chunks,native_stream_error_cases[i].lengths,native_stream_error_cases[i].pending);
    BnTokenizerDecodeStream state={0};assert(!bn_tokenizer_decode_stream_init(&state,&decoder,0));int lead=0xc3,tail=0xa9,empty=256;
    size_t n=99;assert(!bn_tokenizer_decode_stream_push(&decoder,&state,&lead,1,0,NULL,0,&n)&&!n&&state.count==1);
    BnTokenizerDecodeStream before=state;
    assert(!bn_tokenizer_decode_stream_push(&decoder,&state,&empty,1,0,NULL,0,&n)&&!n&&!memcmp(&state,&before,sizeof(state)));
    BnTokenizerDecodeStream checkpoint=state;unsigned char first[16];size_t first_n;
    assert(!bn_tokenizer_decode_stream_push(&decoder,&state,&tail,1,1,first,sizeof(first),&first_n));
    assert(!bn_tokenizer_decode_stream_push(&decoder,&checkpoint,&tail,1,1,out,sizeof(out),&n)&&n==first_n&&!memcmp(out,first,n)&&!memcmp(&state,&checkpoint,sizeof(state)));
    assert(!bn_tokenizer_decode_stream_reset(&state));before=state;memset(out,0xa5,sizeof(out));memcpy(saved,out,sizeof(out));n=SIZE_MAX;
    int bad[]={65,-1},bad2[]={65,261};
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,bad,2,1,out,sizeof(out),&n)==-1&&n==SIZE_MAX&&!memcmp(out,saved,sizeof(out))&&!memcmp(&state,&before,sizeof(state)));
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,bad2,2,0,out,sizeof(out),&n)==-1&&n==SIZE_MAX&&!memcmp(&state,&before,sizeof(state)));
    assert(bn_tokenizer_decode_stream_push(&other,&state,&lead,1,0,out,sizeof(out),&n)==-1&&n==SIZE_MAX);
    assert(bn_tokenizer_decode_stream_init(&state,&decoder,1)==-1&&!memcmp(&state,&before,sizeof(state)));
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,&lead,1,0,out,sizeof(out),(size_t*)(void*)out)==-1&&!memcmp(out,saved,sizeof(out)));
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,&lead,1,0,(unsigned char*)&state,sizeof(state),&n)==-1&&!memcmp(&state,&before,sizeof(state)));
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,(int*)(void*)&state,1,0,out,sizeof(out),&n)==-1&&!memcmp(&state,&before,sizeof(state)));
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,&lead,SIZE_MAX,0,out,sizeof(out),&n)==-1&&n==SIZE_MAX);
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,&lead,1,0,out,SIZE_MAX,&n)==-1&&n==SIZE_MAX);
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,NULL,1,0,out,sizeof(out),&n)==-1&&n==SIZE_MAX);
    state.count=4;BnTokenizerDecodeStream corrupt=state;
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,&lead,1,0,out,sizeof(out),&n)==-1&&!memcmp(&state,&corrupt,sizeof(state)));assert(!bn_tokenizer_decode_stream_reset(&state));
    state.count=1;state.pending[0]='A';corrupt=state;
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,&lead,1,0,out,sizeof(out),&n)==-1&&!memcmp(&state,&corrupt,sizeof(state)));assert(!bn_tokenizer_decode_stream_reset(&state));
    const char *raw=bn_json_text(bn_json_get(bn_json_at(bn_json_get(bn_json_root(&doc),"added_tokens"),0),"content"));
    int mixed[]={257,65};bn_tokenizer_decode_stream_free(&state);assert(!bn_tokenizer_decode_stream_init(&state,&decoder,1));before=state;
    assert(bn_tokenizer_decode_stream_push(&decoder,&state,mixed,2,1,(unsigned char*)raw,1,&n)==-1&&!strcmp(raw,"<s>")&&!memcmp(&state,&before,sizeof(state)));
    bn_tokenizer_decode_stream_free(&state);bn_tokenizer_decoder_free(&other);bn_tokenizer_decoder_free(&decoder);bn_tokenizer_decoder_free(&decoder);bn_json_free(&doc);
    const char *changes[][2]={{"\"special\":true","\"special\":\"bad\""},{",\"special\":true",""}};
    for(size_t i=0;i<2;i++) {
        char *json=native_vocab_replace(native_stream_fixture,changes[i][0],changes[i][1]);assert(!bn_json_init(&doc,json,strlen(json)));free(json);
        assert(bn_tokenizer_decoder_init_json(&decoder,&doc,261,budget)==-1&&!decoder.state);
        BnTokenizerVocab legacy={0};size_t vb;assert(!bn_tokenizer_vocab_plan(261,&vb)&&!bn_tokenizer_vocab_init_json(&legacy,&doc,261,vb));bn_tokenizer_vocab_free(&legacy);bn_json_free(&doc);
    }
    for(size_t i=0;i<sizeof(native_stream_fallback_cases)/sizeof(native_stream_fallback_cases[0]);i++) {
        char *json=native_vocab_replace(native_stream_fixture,"\"\":256",native_stream_fallback_cases[i].replacement);
        if(strstr(native_stream_fallback_cases[i].replacement,"\\u0000")) {
            /* Bounded JSON strings reject embedded NUL; decoded ByteLevel NUL
             * remains supported through its mapped alphabet character. */
            assert(bn_json_init(&doc,json,strlen(json))==-1);free(json);continue;
        }
        assert(!bn_json_init(&doc,json,strlen(json)));free(json);
        assert(!bn_tokenizer_decoder_init_json(&decoder,&doc,261,budget));
        native_stream_case_check(&decoder,decoder.state,native_stream_fallback_cases[i].ids,native_stream_fallback_cases[i].count,0,
            native_stream_fallback_cases[i].chunks,native_stream_fallback_cases[i].lengths,native_stream_fallback_cases[i].pending);
        bn_tokenizer_decoder_free(&decoder);bn_json_free(&doc);
    }
    puts("Native stream decode: source UTF8 chunks/pending, whole-token fallback, skip, flush/reset/checkpoint, atomic failures PASSED");
}
