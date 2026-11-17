#include "json.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void reject(const char *s) {
    BnJSON d={0};assert(bn_json_init(&d,s,strlen(s))==-1&&!d.state);bn_json_free(&d);
}
static void test_explicit_limits(void) {
    BnJSON d={0};BnJSONLimits limits={3,2};
    assert(!bn_json_init_limits(&d,"[0]",3,&limits));bn_json_free(&d);limits.max_nodes=1;
    assert(bn_json_init_limits(&d,"[0]",3,&limits)==-1&&!d.state);limits.max_nodes=2;limits.max_bytes=2;
    assert(bn_json_init_limits(&d,"[0]",3,&limits)==-1&&!d.state);
    limits=(BnJSONLimits){SIZE_MAX,SIZE_MAX};assert(bn_json_init_limits(&d,"0",SIZE_MAX,&limits)==-1);
    assert(!bn_json_init_limits(&d,"{}",2,&limits));bn_json_free(&d);
    limits=(BnJSONLimits){3,0};assert(bn_json_init_limits(&d,"[0]",3,&limits)==-1);
    assert(bn_json_init_limits(&d,"0",1,NULL)==-1);
    size_t elements=BN_JSON_MAX_NODES+37,bytes=2*elements+1;char *large=malloc(bytes);assert(large);large[0]='[';
    for(size_t i=0;i<elements;i++){large[1+2*i]='0';large[2+2*i]=i+1==elements?']':',';}
    limits=(BnJSONLimits){bytes,elements};assert(bn_json_init_limits(&d,large,bytes,&limits)==-1&&!d.state);
    limits.max_nodes++;assert(!bn_json_init_limits(&d,large,bytes,&limits)&&bn_json_size(bn_json_root(&d))==elements);bn_json_free(&d);
    assert(bn_json_init(&d,large,bytes)==-1&&!d.state);free(large);
    const char object[]= "{\"z\":0,\"a\":1}";assert(!bn_json_init(&d,object,strlen(object)));
    const char *key="unchanged";assert(bn_json_member(bn_json_root(&d),0,&key)&&!strcmp(key,"a"));
    assert(bn_json_member(bn_json_root(&d),1,&key)&&!strcmp(key,"z"));
    const char *old=key;assert(!bn_json_member(bn_json_root(&d),2,&key)&&key==old);
    assert(!bn_json_member(bn_json_root(&d),0,NULL)&&!bn_json_member(NULL,0,&key)&&key==old);bn_json_free(&d);
}
int main(void) {
    test_explicit_limits();
    char input[]="{\"z\":null,\"a\":[true,false,18446744073709551615,-0,1.25e-20,\"\\uD83D\\uDE00\"],\"\\u0062\":{\"\":\"value\"}}";
    BnJSON d={0};assert(!bn_json_init(&d,input,strlen(input)));memset(input,'?',strlen(input));
    const BnJSONValue *root=bn_json_root(&d),*a=bn_json_get(root,"a");assert(bn_json_type(root)==BN_JSON_OBJECT&&bn_json_size(root)==3&&bn_json_size(a)==6);
    assert(bn_json_type(bn_json_get(root,"z"))==BN_JSON_NULL&&!bn_json_get(root,"missing"));int boolean=7;assert(!bn_json_bool(bn_json_at(a,0),&boolean)&&boolean==1);assert(!bn_json_bool(bn_json_at(a,1),&boolean)&&boolean==0);
    uint64_t n=99;assert(!bn_json_u64(bn_json_at(a,2),&n)&&n==UINT64_MAX);n=99;assert(bn_json_u64(bn_json_at(a,3),&n)==-1&&n==99);assert(bn_json_u64(bn_json_at(a,4),&n)==-1&&n==99);
    assert(!strcmp(bn_json_text(bn_json_at(a,4)),"1.25e-20"));assert(!strcmp(bn_json_text(bn_json_at(a,5)),"\xf0\x9f\x98\x80"));assert(!strcmp(bn_json_text(bn_json_get(bn_json_get(root,"b"),"")),"value"));
    assert(!bn_json_get(a,"a")&&!bn_json_at(root,0)&&!bn_json_at(a,6)&&bn_json_type(NULL)==BN_JSON_INVALID&&!bn_json_text(bn_json_get(root,"z")));
    void *state=d.state;assert(bn_json_init(&d,"null",4)==-1&&state==d.state);bn_json_free(&d);bn_json_free(&d);assert(!bn_json_root(&d));
    const char *bad[]={""," ","{}{}","[1,]","{\"a\":1,}","{\"a\":1,\"\\u0061\":2}","{\"x\":{\"a\":1,\"a\":2}}","[truefalse]","01","-01","+1",".5","1.","1e","1e+","--1","nan","Infinity","\xef\xbb\xbf{}","\"\\u0000\"","\"\\uD800\"","\"\\uDC00\"","\"\\uD800\\u0041\"","\"\xc0\x80\"","\"\xed\xa0\x80\"","\"\xf4\x90\x80\x80\"","\"\x01\"","\"\\x41\"","[","{","{\"a\"}","[null null]"};
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++)reject(bad[i]);
    assert(!bn_json_init(&d,"18446744073709551616",20));n=99;assert(bn_json_u64(bn_json_root(&d),&n)==-1&&n==99);bn_json_free(&d);
    assert(!bn_json_init(&d,"1e999999999999999999999",23));assert(!strcmp(bn_json_text(bn_json_root(&d)),"1e999999999999999999999"));bn_json_free(&d);
    char nested[140];for(unsigned depth=64;depth<=65;depth++){memset(nested,'[',depth);nested[depth]='0';memset(nested+depth+1,']',depth);size_t bytes=depth*2+1;int result=bn_json_init(&d,nested,bytes);assert(result==(depth==64?0:-1));bn_json_free(&d);}
    for(size_t elements=BN_JSON_MAX_NODES-1;elements<=BN_JSON_MAX_NODES;elements++) {
        size_t bytes=elements*2+1;char *large=malloc(bytes);assert(large);large[0]='[';for(size_t i=0;i<elements;i++){large[1+i*2]='0';large[2+i*2]=i+1==elements?']':',';}
        int result=bn_json_init(&d,large,bytes);assert(result==(elements==BN_JSON_MAX_NODES-1?0:-1));if(!result)assert(bn_json_size(bn_json_root(&d))==elements);bn_json_free(&d);free(large);
    }
    assert(bn_json_init(&d,"0",(size_t)BN_JSON_MAX_BYTES+1)==-1);assert(bn_json_init(NULL,"0",1)==-1&&bn_json_init(&d,NULL,1)==-1);
    puts("JSON owned configuration, syntax, Unicode, duplicate and resource-limit tests PASSED");return 0;
}
