#include "safetensors.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __EMSCRIPTEN__
#include <unistd.h>
#ifdef __linux__
#include <dirent.h>
#endif
#endif

static unsigned char *container(const char *header,size_t payload,size_t *size) {
    size_t n=strlen(header);*size=8+n+payload;
    unsigned char *data=malloc(*size);assert(data);
    for(int i=0;i<8;i++)data[i]=(unsigned char)((uint64_t)n>>(i*8));
    memcpy(data+8,header,n);
    for(size_t i=0;i<payload;i++)data[8+n+i]=(unsigned char)(i*37+3);
    return data;
}
static void rejects(const char *header,size_t payload) {
    size_t size;unsigned char *data=container(header,payload,&size);BnSafetensors f={0};
    assert(bn_safetensors_init_buffer(&f,data,size)==-1&&!f.state);
    bn_safetensors_free(&f);free(data);
}
static void valid_io(void) {
    const char *header="{\"z\":{\"shape\":[4],\"data_offsets\":[4,8],\"dtype\":\"U8\"},"
        "\"a\":{\"dtype\":\"I16\",\"shape\":[2],\"data_offsets\":[0,4]},"
        "\"empty\":{\"dtype\":\"F32\",\"shape\":[0,2],\"data_offsets\":[8,8]},"
        "\"__metadata__\":{\"format\":\"pt\",\"escaped\":\"line\\n\\t\\\\\\\"\"}}  ";
    size_t size;unsigned char *data=container(header,8,&size);BnSafetensors f={0};
    assert(!bn_safetensors_init_buffer(&f,data,size));assert(bn_safetensors_count(&f)==3);
    assert(bn_safetensors_find(&f,"a")==0&&bn_safetensors_find(&f,"empty")==1&&bn_safetensors_find(&f,"z")==2);
    assert(bn_safetensors_find(&f,"missing")==-1&&!bn_safetensors_tensor(&f,3));
    const BnSafetensorsTensor *t=bn_safetensors_tensor(&f,0);
    assert(t->dtype==BN_SAFETENSORS_I16&&t->n_dims==1&&t->shape[0]==2&&t->bytes==4&&t->offset==0);
    assert(!strcmp(bn_safetensors_metadata(&f,"format"),"pt"));
    assert(!strcmp(bn_safetensors_metadata(&f,"escaped"),"line\n\t\\\""));
    assert(!bn_safetensors_metadata(&f,"missing"));
    unsigned char guard[10];memset(guard,0xa5,sizeof(guard));
    assert(!bn_safetensors_read(&f,2,1,guard+1,3));
    assert(guard[0]==0xa5&&guard[4]==0xa5&&!memcmp(guard+1,data+8+strlen(header)+5,3));
    unsigned char before[10];memcpy(before,guard,sizeof(guard));
    assert(bn_safetensors_read(&f,2,3,guard,2)==-1&&!memcmp(before,guard,sizeof(guard)));
    assert(bn_safetensors_read(&f,2,UINT64_MAX,guard,1)==-1);
    assert(bn_safetensors_read(&f,3,0,guard,0)==-1);
    assert(bn_safetensors_read(&f,0,0,NULL,1)==-1);
    assert(!bn_safetensors_read(&f,1,0,NULL,0));
    void *state=f.state;assert(bn_safetensors_init_buffer(&f,data,size)==-1&&f.state==state);
    assert(bn_safetensors_init_file(&f,"/does-not-exist")==-1&&f.state==state);
    bn_safetensors_free(&f);bn_safetensors_free(&f);assert(!f.state);
#ifndef __EMSCRIPTEN__
    char path[]="/tmp/bn-safetensors-XXXXXX";int fd=mkstemp(path);assert(fd>=0);
    assert(write(fd,data,size)==(ssize_t)size);assert(!close(fd));
    assert(!bn_safetensors_init_file(&f,path));
    assert(!bn_safetensors_read(&f,0,0,guard,4)&&!memcmp(guard,data+8+strlen(header),4));
    /* Inject a short source after admission: pread reports failure without
     * touching mapped payload pages or publishing a successful short read. */
    FILE *writer=fopen(path,"wb");assert(writer);
    size_t short_size=8+strlen(header)+1;
    assert(fwrite(data,1,short_size,writer)==short_size&&!fclose(writer));
    memset(guard,0xa5,sizeof(guard));
    assert(bn_safetensors_read(&f,0,0,guard,4)==-1);
    assert(guard[0]==data[8+strlen(header)]&&guard[1]==0xa5&&guard[4]==0xa5);
    assert(!unlink(path));bn_safetensors_free(&f);
    /* File reads retain their own descriptor after unlink. */
    memcpy(path,"/tmp/bn-safetensors-XXXXXX",sizeof(path));fd=mkstemp(path);assert(fd>=0);
    assert(write(fd,data,size)==(ssize_t)size&&!close(fd));
    assert(!bn_safetensors_init_file(&f,path)&&!unlink(path));
    assert(!bn_safetensors_read(&f,2,0,guard,4));bn_safetensors_free(&f);
#endif
    free(data);
    assert(bn_safetensors_init_file(&f,"/does-not-exist")==-1&&!f.state);
    assert(!bn_safetensors_count(NULL)&&!bn_safetensors_tensor(NULL,0));
    assert(bn_safetensors_read(NULL,0,0,NULL,0)==-1);
    puts("Safetensors buffer/file ownership, lookup and bounded reads PASSED");
}
static void types_and_unicode(void) {
    for(int i=0;i<BN_SAFETENSORS_DTYPE_COUNT;i++) {
        const char *name=bn_safetensors_dtype_name((BnSafetensorsDtype)i);
        unsigned bits=bn_safetensors_dtype_bits((BnSafetensorsDtype)i);assert(name&&bits);
        char header[256];snprintf(header,sizeof(header),"{\"x\":{\"dtype\":\"%s\",\"shape\":[4],\"data_offsets\":[0,%u]}}",name,bits/2);
        size_t size;unsigned char *data=container(header,bits/2,&size);BnSafetensors f={0};
        assert(!bn_safetensors_init_buffer(&f,data,size));
        assert(bn_safetensors_tensor(&f,0)->dtype==(BnSafetensorsDtype)i);
        bn_safetensors_free(&f);free(data);
    }
    assert(!bn_safetensors_dtype_bits((BnSafetensorsDtype)-1));
    assert(!bn_safetensors_dtype_name((BnSafetensorsDtype)999));
    const char *header="{\"\\u00e9\\ud83d\\ude00\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}";
    size_t size;unsigned char *data=container(header,4,&size);BnSafetensors f={0};
    assert(!bn_safetensors_init_buffer(&f,data,size));
    assert(bn_safetensors_find(&f,"\xc3\xa9\xf0\x9f\x98\x80")==0);
    assert(bn_safetensors_tensor(&f,0)->n_dims==0);bn_safetensors_free(&f);free(data);
    data=container("{}",0,&size);assert(!bn_safetensors_init_buffer(&f,data,size)&&!bn_safetensors_count(&f));
    bn_safetensors_free(&f);free(data);
    char ranked[512];size_t used=(size_t)snprintf(ranked,sizeof(ranked),"{\"rank\":{\"dtype\":\"U8\",\"shape\":[");
    for(int i=0;i<BN_SAFETENSORS_MAX_DIMS;i++)used+=(size_t)snprintf(ranked+used,sizeof(ranked)-used,"%s1",i?",":"");
    snprintf(ranked+used,sizeof(ranked)-used,"],\"data_offsets\":[0,1]}}");
    data=container(ranked,1,&size);assert(!bn_safetensors_init_buffer(&f,data,size));
    assert(bn_safetensors_tensor(&f,0)->n_dims==BN_SAFETENSORS_MAX_DIMS);
    bn_safetensors_free(&f);free(data);
    snprintf(ranked+used,sizeof(ranked)-used,",1],\"data_offsets\":[0,1]}}");rejects(ranked,1);
    puts("Safetensors storage types, packed extents, Unicode and scalar/empty containers PASSED");
}
static void invalid_headers(void) {
    const char *bad[]={
        "", "[]", " {}", "{}x", "{,}",
        "{\"x\":{}}", "{\"x\":{\"dtype\":\"UNKNOWN\",\"shape\":[],\"data_offsets\":[0,0]}}",
        "{\"x\":{\"dtype\":\"F4\",\"shape\":[],\"data_offsets\":[0,0]}}",
        "{\"x\":{\"dtype\":\"U8\",\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]}}",
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[0],\"unknown\":0,\"data_offsets\":[0,0]}}",
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[-1],\"data_offsets\":[0,0]}}",
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[00],\"data_offsets\":[0,0]}}",
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[1e0],\"data_offsets\":[0,1]}}",
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[1.0],\"data_offsets\":[0,1]}}",
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[18446744073709551616],\"data_offsets\":[0,0]}}",
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[0,18446744073709551615,2],\"data_offsets\":[0,0]}}",
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0]}}",
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[1,0]}}",
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]},\"\\u0078\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]}}",
        "{\"__metadata__\":{\"x\":\"a\",\"\\u0078\":\"b\"}}",
        "{\"__metadata__\":{},\"__metadata__\":{}}",
        "{\"__metadata__\":{\"x\":1}}", "{\"__metadata__\":null}",
        "{\"__metadata__\":{\"\\u0000\":\"x\"}}",
        "{\"__metadata__\":{\"\\ud800\":\"x\"}}",
        "{\"__metadata__\":{\"\\udc00\":\"x\"}}",
        "{\"__metadata__\":{\"\\ud800\\u0041\":\"x\"}}",
        "{\"__metadata__\":{\"\\q\":\"x\"}}",
        "{\"__metadata__\":{\"\xc0\xaf\":\"x\"}}",
        "{\"__metadata__\":{\"\xed\xa0\x80\":\"x\"}}",
        "{\"__metadata__\":{\"\xf4\x90\x80\x80\":\"x\"}}",
        "{\"__metadata__\":{\"raw\nnewline\":\"x\"}}",
        "{\"__metadata__\":{\"x\":\"unterminated}}",
    };
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);i++)rejects(bad[i],0);
    rejects("{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}",2); /* gap */
    rejects("{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",2); /* tail */
    rejects("{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]},\"y\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}",2);
    puts("Safetensors malformed JSON/Unicode, duplicates, overflow and payload coverage rejection PASSED");
}
static void mutations(void) {
    const char *header="{\"x\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}";
    size_t size;unsigned char *data=container(header,8,&size);
    for(size_t n=0;n<size;n++){BnSafetensors f={0};assert(bn_safetensors_init_buffer(&f,data,n)==-1&&!f.state);}
    unsigned seed=20261111;
    for(int i=0;i<10000;i++) {
        seed=seed*1664525u+1013904223u;size_t pos=seed%(size-8);
        unsigned char old=data[pos];data[pos]=(unsigned char)(seed>>24);BnSafetensors f={0};
        if(!bn_safetensors_init_buffer(&f,data,size)) {
            uint64_t header_size=0;for(int k=0;k<8;k++)header_size|=(uint64_t)data[k]<<(k*8);
            for(size_t k=0;k<bn_safetensors_count(&f);k++) {
                const BnSafetensorsTensor *t=bn_safetensors_tensor(&f,k);assert(t->bytes<=size-8-header_size);
                assert(bn_safetensors_find(&f,t->name)==(int)k);
                unsigned char copy[128];assert(t->bytes<=sizeof(copy));
                assert(!bn_safetensors_read(&f,k,0,copy,(size_t)t->bytes));
                assert(!memcmp(copy,data+8+(size_t)header_size+(size_t)t->offset,(size_t)t->bytes));
            }
        }
        bn_safetensors_free(&f);data[pos]=old;
    }
    free(data);puts("Safetensors all truncations and 10000 bounded mutation cases PASSED");
}

#ifndef __EMSCRIPTEN__
static void write_bytes(const char *path,const void *data,size_t size) {
    FILE *file=fopen(path,"wb");assert(file);
    assert(fwrite(data,1,size,file)==size&&!fclose(file));
}
static void write_text(const char *path,const char *text) {write_bytes(path,text,strlen(text));}
static int descriptor_count(void) {
#ifdef __linux__
    DIR *dir=opendir("/proc/self/fd");assert(dir);int count=0;
    struct dirent *entry;
    while((entry=readdir(dir)))if(strcmp(entry->d_name,".")&&strcmp(entry->d_name,".."))count++;
    assert(!closedir(dir));return count;
#else
    return 0;
#endif
}
static void index_binding(void) {
    char directory[]="/tmp/bn-safetensors-index-XXXXXX";assert(mkdtemp(directory));
    char first[512],second[512],manifest[512];
    snprintf(first,sizeof(first),"%s/one.safetensors",directory);
    snprintf(second,sizeof(second),"%s/two.safetensors",directory);
    snprintf(manifest,sizeof(manifest),"%s/model.safetensors.index.json",directory);
    size_t na,nb;unsigned char *a=container("{\"a\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}",4,&na);
    unsigned char *b=container("{\"b\":{\"dtype\":\"I8\",\"shape\":[3],\"data_offsets\":[0,3]}}",3,&nb);
    b[nb-3]=0xe0;b[nb-2]=0xe1;b[nb-1]=0xe2;
    write_bytes(first,a,na);write_bytes(second,b,nb);
    const char *good="{\"weight_map\":{\"b\":\"two.safetensors\",\"a\":\"one.safetensors\"},\"metadata\":{\"total_size\":7}}";
    write_text(manifest,good);BnSafetensorsIndex index={0};int before=descriptor_count();
    assert(!bn_safetensors_index_init(&index,manifest));
    assert(bn_safetensors_index_count(&index)==2&&bn_safetensors_index_shards(&index)==2&&bn_safetensors_index_bytes(&index)==7);
    assert(bn_safetensors_index_find(&index,"a")==0&&bn_safetensors_index_find(&index,"b")==1);
    assert(bn_safetensors_index_find(&index,"c")==-1&&!bn_safetensors_index_tensor(&index,2));
    assert(bn_safetensors_index_tensor(&index,0)->dtype==BN_SAFETENSORS_F32);
    assert(!strcmp(bn_safetensors_index_shard_name(&index,0),"one.safetensors"));
    assert(!strcmp(bn_safetensors_index_shard_name(&index,1),"two.safetensors"));
    unsigned char output[4]={0xa5,0xa5,0xa5,0xa5};
    assert(!bn_safetensors_index_read(&index,1,1,output+1,2));
    assert(output[0]==0xa5&&output[1]==0xe1&&output[2]==0xe2&&output[3]==0xa5);
    assert(bn_safetensors_index_read(&index,1,2,output,2)==-1&&output[0]==0xa5);
    assert(bn_safetensors_index_read(&index,2,0,output,0)==-1);
    void *state=index.state;assert(bn_safetensors_index_init(&index,"/missing")==-1&&index.state==state);
    bn_safetensors_index_free(&index);bn_safetensors_index_free(&index);
    assert(!index.state&&descriptor_count()==before);
    const char *bad[]={
        "{}", "[]", "{\"weight_map\":{}}", "{\"metadata\":{\"total_size\":0}}",
        "{\"metadata\":{},\"weight_map\":{}}",
        "{\"metadata\":{\"total_size\":0,\"total_size\":0},\"weight_map\":{}}",
        "{\"metadata\":{\"total_size\":0},\"weight_map\":{},\"weight_map\":{}}",
        "{\"metadata\":{\"total_size\":0},\"weight_map\":{},\"unknown\":0}",
        "{\"metadata\":{\"total_size\":0},\"weight_map\":{\"a\":1}}",
        "{\"metadata\":{\"total_size\":0},\"weight_map\":{\"a\":\"../one.safetensors\"}}",
        "{\"metadata\":{\"total_size\":0},\"weight_map\":{\"a\":\"/one.safetensors\"}}",
        "{\"metadata\":{\"total_size\":0},\"weight_map\":{\"a\":\"..\\\\one.safetensors\"}}",
        "{\"metadata\":{\"total_size\":0},\"weight_map\":{\"a\":\"..\\u002fone.safetensors\"}}",
        "{\"metadata\":{\"total_size\":0},\"weight_map\":{\"a\":\"one.bin\"}}",
        "{\"metadata\":{\"total_size\":4},\"weight_map\":{\"a\":\"missing.safetensors\"}}",
        "{\"metadata\":{\"total_size\":7},\"weight_map\":{\"a\":\"one.safetensors\",\"b\":\"zz-missing.safetensors\"}}",
        "{\"metadata\":{\"total_size\":8},\"weight_map\":{\"a\":\"one.safetensors\",\"b\":\"two.safetensors\"}}",
        "{\"metadata\":{\"total_size\":7},\"weight_map\":{\"b\":\"one.safetensors\",\"a\":\"two.safetensors\"}}",
        "{\"metadata\":{\"total_size\":7},\"weight_map\":{\"a\":\"one.safetensors\",\"b\":\"one.safetensors\"}}",
        "{\"metadata\":{\"total_size\":4},\"weight_map\":{\"a\":\"one.safetensors\",\"\\u0061\":\"one.safetensors\"}}",
        "{\"metadata\":{\"total_size\":4},\"weight_map\":{\"missing\":\"one.safetensors\"}}",
        "{\"metadata\":{\"total_size\":18446744073709551616},\"weight_map\":{}}",
        "{\"metadata\":{\"total_size\":-1},\"weight_map\":{}}",
    };
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);i++) {
        write_text(manifest,bad[i]);assert(bn_safetensors_index_init(&index,manifest)==-1&&!index.state);
        assert(descriptor_count()==before);
    }
    /* Repeated partial open failures release every descriptor/map. */
    write_text(manifest,bad[15]);
    for(int i=0;i<200;i++)assert(bn_safetensors_index_init(&index,manifest)==-1&&!index.state);
    assert(descriptor_count()==before);
    /* Only referenced files matter: an unrelated complete export may coexist. */
    write_text(manifest,"{\"metadata\":{\"total_size\":4},\"weight_map\":{\"a\":\"one.safetensors\"}}");
    assert(!bn_safetensors_index_init(&index,manifest)&&bn_safetensors_index_count(&index)==1);
    bn_safetensors_index_free(&index);
    /* Duplicate actual names across shards reject, even with the right size. */
    size_t changed_size;unsigned char *changed=container("{\"a\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[0,3]}}",3,&changed_size);
    write_bytes(second,changed,changed_size);write_text(manifest,good);
    assert(bn_safetensors_index_init(&index,manifest)==-1&&!index.state);free(changed);write_bytes(second,b,nb);
    /* Unindexed tensors in a referenced shard reject even when total size agrees. */
    changed=container("{\"a\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[0,3]},\"orphan\":{\"dtype\":\"U8\",\"shape\":[],\"data_offsets\":[3,4]}}",4,&changed_size);
    write_bytes(first,changed,changed_size);assert(bn_safetensors_index_init(&index,manifest)==-1&&!index.state);
    free(changed);write_bytes(first,a,na);
    assert(!bn_safetensors_index_init(&index,manifest));
    assert(!unlink(first)&&!unlink(second)&&!unlink(manifest));
    assert(!bn_safetensors_index_read(&index,1,0,output,3)&&output[0]==0xe0);
    bn_safetensors_index_free(&index);assert(descriptor_count()==before);
    write_text(manifest,"{\"metadata\":{\"total_size\":0},\"weight_map\":{}}");
    assert(!bn_safetensors_index_init(&index,manifest)&&!bn_safetensors_index_count(&index)&&!bn_safetensors_index_shards(&index));
    assert(bn_safetensors_index_read(&index,0,0,NULL,0)==-1);
    bn_safetensors_index_free(&index);assert(!unlink(manifest)&&!rmdir(directory));
    free(a);free(b);puts("Safetensors index binding, coverage, path validation and rollback PASSED");
}
#endif

int main(void) {valid_io();types_and_unicode();invalid_headers();
#ifndef __EMSCRIPTEN__
    index_binding();
#endif
    mutations();puts("Safetensors PASSED");return 0;}
