#include "trellis.h"
#include "quant.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __EMSCRIPTEN__
#include <unistd.h>
static void fixture(const char *dir,int marker_ok,int shape_ok) {
    char path[256],header[1024];snprintf(path,sizeof(path),"%s/one.safetensors",dir);
    int count=snprintf(header,sizeof(header),"{\"w.mul1\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]},\"w.trellis\":{\"dtype\":\"I16\",\"shape\":[8,8,48],\"data_offsets\":[4,6148]},\"w.suh\":{\"dtype\":\"F16\",\"shape\":%s,\"data_offsets\":[6148,6404]},\"w.svh\":{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[6404,6660]}}",shape_ok?"[128]":"[16,8]");
    FILE*f=fopen(path,"wb");assert(f);unsigned char length[8];for(int i=0;i<8;i++)length[i]=(unsigned char)((uint64_t)count>>(i*8));assert(fwrite(length,1,8,f)==8);assert(fwrite(header,1,(size_t)count,f)==(size_t)count);
    unsigned char payload[6660];for(int i=0;i<6660;i++)payload[i]=(unsigned char)(i*37+11);
    uint32_t marker=marker_ok?UINT32_C(0x83dcd12d):UINT32_C(0xcbac1fed);for(int i=0;i<4;i++)payload[i]=(unsigned char)(marker>>(i*8));assert(fwrite(payload,1,sizeof(payload),f)==sizeof(payload));assert(!fclose(f));
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);f=fopen(path,"w");assert(f);fputs("{\"metadata\":{\"total_size\":6660},\"weight_map\":{\"w.mul1\":\"one.safetensors\",\"w.trellis\":\"one.safetensors\",\"w.suh\":\"one.safetensors\",\"w.svh\":\"one.safetensors\"}}",f);assert(!fclose(f));
}
int main(void) {
    char dir[]="/tmp/bn-trellis-XXXXXX";assert(mkdtemp(dir));fixture(dir,1,1);char path[256];snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);
    BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    size_t ids[4];const char*names[]={"w.trellis","w.suh","w.svh","w.mul1"};for(int i=0;i<4;i++){int id=bn_safetensors_index_find(&index,names[i]);assert(id>=0);ids[i]=(size_t)id;}
    BnTrellisSource source={0};assert(!bn_trellis_source_init(&source,&index,BN_QUANT_FORMAT_EXL3_MUL1_3,ids));const BnTrellisLayout*l=bn_trellis_source_layout(&source);assert(l&&l->k==128&&l->n==128&&l->packed_bytes==6144&&l->pre_offset==6144&&l->post_offset==6400&&l->bundle_bytes==6656&&l->marker==UINT32_C(0x83dcd12d));
    unsigned char data[6658];memset(data,0xa5,sizeof(data));assert(!bn_trellis_source_read(&source,data+1,6656));assert(data[0]==0xa5&&data[6657]==0xa5);for(int i=0;i<6656;i++)assert(data[i+1]==(unsigned char)((i+4)*37+11));
    BnTrellisBundle view={0};assert(!bn_trellis_bundle(&source,data+1,6656,&view));assert(view.packed==data+1&&view.pre==data+6145&&view.post==data+6401);BnTrellisBundle saved=view;assert(bn_trellis_bundle(&source,data+1,6655,&view)==-1&&!memcmp(&view,&saved,sizeof(view)));
    memset(data,0xa5,sizeof(data));assert(bn_trellis_source_read(&source,data,6655)==-1);assert(bn_trellis_source_read(&source,data,6657)==-1);assert(bn_trellis_source_read(&source,NULL,6656)==-1);for(int i=0;i<6658;i++)assert(data[i]==0xa5);
    void *state=source.state;assert(bn_trellis_source_init(&source,&index,BN_QUANT_FORMAT_EXL3_MUL1_3,ids)==-1&&source.state==state);
    BnTrellisSource bad={0};assert(bn_trellis_source_init(&bad,&index,BN_GGUF_TENSOR_Q3_K,ids)==-1&&!bad.state);size_t wrong[4];memcpy(wrong,ids,sizeof(wrong));wrong[2]=wrong[1];assert(bn_trellis_source_init(&bad,&index,BN_QUANT_FORMAT_EXL3_MUL1_3,wrong)==-1);wrong[2]=SIZE_MAX;assert(bn_trellis_source_init(&bad,&index,BN_QUANT_FORMAT_EXL3_MUL1_3,wrong)==-1);
    /* Admission keeps the fd; a later short read must fail, not publish success. */
    snprintf(path,sizeof(path),"%s/one.safetensors",dir);FILE*f=fopen(path,"wb");assert(f);assert(!fclose(f));assert(bn_trellis_source_read(&source,data+1,6656)==-1);bn_safetensors_index_free(&index);assert(!bn_trellis_source_layout(&source)&&bn_trellis_source_read(&source,data+1,6656)==-1);bn_trellis_source_free(&source);bn_trellis_source_free(&source);
    fixture(dir,0,1);snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!bn_safetensors_index_init(&index,path));assert(bn_trellis_source_init(&bad,&index,BN_QUANT_FORMAT_EXL3_MUL1_3,ids)==-1&&!bad.state);bn_safetensors_index_free(&index);
    fixture(dir,1,0);assert(!bn_safetensors_index_init(&index,path));assert(bn_trellis_source_init(&bad,&index,BN_QUANT_FORMAT_EXL3_MUL1_3,ids)==-1&&!bad.state);bn_safetensors_index_free(&index);
    assert(!unlink(path));snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!unlink(path));assert(!rmdir(dir));puts("Trellis immutable native admission/bundle tests PASSED");return 0;
}
#else
int main(void){puts("Trellis file fixtures skipped on WASM");return 0;}
#endif
