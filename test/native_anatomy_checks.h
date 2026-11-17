#ifndef __EMSCRIPTEN__
#include "native_anatomy_reference.h"
#include <limits.h>
typedef struct {char name[160];BnSafetensorsDtype dtype;int rank;uint64_t shape[3],offset,bytes;} NativeFixtureTensor;
static void native_anatomy_fixture_geometry(const char *dir,int variant,int index_heads,int index_dim,int heads,int head_dim,int groups) {
    NativeFixtureTensor *spec=calloc(4096,sizeof(*spec));assert(spec);size_t count=0;
    for(size_t i=0;i<sizeof(native_core_reference)/sizeof(native_core_reference[0]);i++) {
        const NativeCoreReference *r=&native_core_reference[i];
        if(variant==1&&!strcmp(r->name,"layers.39.ffn_norm.weight"))continue;
        NativeFixtureTensor *s=&spec[count++];snprintf(s->name,sizeof(s->name),"%s",r->name);s->dtype=r->dtype;s->rank=r->rank;s->shape[0]=r->shape[0];s->shape[1]=r->shape[1];
        if(variant==2&&!strcmp(s->name,"layers.39.attn.wkv.weight"))s->dtype=BN_SAFETENSORS_F8_E5M2;
        if(variant==3&&!strcmp(s->name,"layers.39.hc_ffn_fn"))s->shape[1]/=2;
        if(variant==5&&!strcmp(s->name,"layers.39.attn.wo_b.scale"))s->shape[1]++;
    }
    for(size_t i=0;i<count;i++){
        NativeFixtureTensor*t=&spec[i];
        if(heads!=8||head_dim!=128||groups!=2) {
        int block=t->dtype==BN_SAFETENSORS_F8_E8M0?32:1;
        if(strstr(t->name,"attn.attn_sink"))t->shape[0]=heads;
        if(strstr(t->name,"attn.kv_norm.weight")||strstr(t->name,"attn.compressor.norm.weight")||
           strstr(t->name,"attn.compressor.wgate.weight")||strstr(t->name,"attn.compressor.wkv.weight"))t->shape[0]=head_dim;
        if(strstr(t->name,"attn.wkv."))t->shape[0]=(head_dim+block-1)/block;
        if(strstr(t->name,"attn.wq_b."))t->shape[0]=((uint64_t)heads*head_dim+block-1)/block;
        if(strstr(t->name,"attn.wo_a.")){t->shape[0]=((uint64_t)groups*128+block-1)/block;t->shape[1]=((uint64_t)(heads/groups)*head_dim+block-1)/block;}
        if(strstr(t->name,"attn.wo_b."))t->shape[1]=((uint64_t)groups*128+block-1)/block;
        if(strstr(t->name,"attn.indexer.wk.weight"))t->shape[1]=head_dim;
        }
        if(strstr(t->name,"attn.indexer.wq_b.weight"))t->shape[0]=(uint64_t)index_heads*index_dim;
        if(strstr(t->name,"attn.indexer.wq_b.scale"))t->shape[0]=((uint64_t)index_heads*index_dim+31)/32;
        if(strstr(t->name,"attn.indexer.weights_proj.weight"))t->shape[0]=index_heads;
        if(strstr(t->name,"attn.indexer.wk.weight")||strstr(t->name,"attn.indexer.k_norm.weight"))t->shape[0]=index_dim;
    }
    const char *roles[]={"w1","w3","w2"},*parts[]={"trellis","suh","svh","mul1"};
    for(int l=0;l<40;l++)for(int e=0;e<2;e++)for(int p=0;p<3;p++)for(int c=0;c<4;c++) {
        NativeFixtureTensor *s=&spec[count++];snprintf(s->name,sizeof(s->name),"layers.%d.ffn.experts.%d.%s.%s",l,e,roles[p],parts[c]);
        s->dtype=c==0?BN_SAFETENSORS_I16:c==3?BN_SAFETENSORS_I32:BN_SAFETENSORS_F16;s->rank=c==0?3:c==3?0:1;s->shape[0]=c==0?8:128;s->shape[1]=8;s->shape[2]=48;
    }
    for(int i=0;i<2+(variant==4);i++) {
        NativeFixtureTensor *s=&spec[count++];snprintf(s->name,sizeof(s->name),"%s",i==0?"mtp.0.unused":i==1?"vision.unused":"layers.0.attn.unrecognized");s->dtype=BN_SAFETENSORS_F32;s->rank=1;s->shape[0]=1;
    }
    assert(count<4096);uint64_t payload=0;char *header=malloc(2*1024*1024);assert(header);size_t used=0;
    header[used++]='{';
    for(size_t i=0;i<count;i++) {
        NativeFixtureTensor *s=&spec[i];uint64_t elements=1;char shape[128];size_t sp=0;shape[sp++]='[';
        for(int d=0;d<s->rank;d++){elements*=s->shape[d];int n=snprintf(shape+sp,sizeof(shape)-sp,"%s%llu",d?",":"",(unsigned long long)s->shape[d]);assert(n>0&&(size_t)n<sizeof(shape)-sp);sp+=(size_t)n;}
        shape[sp++]=']';shape[sp]=0;s->offset=payload;s->bytes=elements*bn_safetensors_dtype_bits(s->dtype)/8;payload+=s->bytes;
        int n=snprintf(header+used,2*1024*1024-used,"%s\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[%llu,%llu]}",i?",":"",s->name,bn_safetensors_dtype_name(s->dtype),shape,(unsigned long long)s->offset,(unsigned long long)payload);assert(n>0&&(size_t)n<2*1024*1024-used);used+=(size_t)n;
    }
    header[used++]='}';char path[256];snprintf(path,sizeof(path),"%s/one.safetensors",dir);FILE *f=fopen(path,"wb");assert(f);unsigned char length[8];for(int i=0;i<8;i++)length[i]=(unsigned char)((uint64_t)used>>(8*i));assert(fwrite(length,1,8,f)==8&&fwrite(header,1,used,f)==used);uint64_t base=8+used;assert(base+payload<LONG_MAX);assert(!fseek(f,(long)(base+payload-1),SEEK_SET));assert(fputc(0,f)!=EOF);
    for(size_t i=0;i<count;i++)if(strstr(spec[i].name,".mul1")){unsigned char marker[]={0x2d,0xd1,0xdc,0x83};assert(!fseek(f,(long)(base+spec[i].offset),SEEK_SET));assert(fwrite(marker,1,4,f)==4);}
    if(variant==7)for(size_t i=0;i<count;i++) {
        NativeFixtureTensor*t=&spec[i];
        if((!strncmp(t->name,"layers.0.",9)||!strncmp(t->name,"layers.2.",9))&&
           (strstr(t->name,"attn.wo_a.")||strstr(t->name,"attn.wo_b."))) {
            unsigned char block[4096];memset(block,t->dtype==BN_SAFETENSORS_F8_E8M0?127:32,sizeof(block));
            assert(!fseek(f,(long)(base+t->offset),SEEK_SET));
            for(uint64_t left=t->bytes;left;){size_t n=left<sizeof(block)?(size_t)left:sizeof(block);assert(fwrite(block,1,n,f)==n);left-=n;}
        }
    }
    if(variant==8)for(size_t i=0;i<count;i++) {
        NativeFixtureTensor*t=&spec[i];int selected=!strncmp(t->name,"layers.0.",9)||!strncmp(t->name,"layers.2.",9)||!strncmp(t->name,"layers.3.",9)||!strncmp(t->name,"layers.20.",10);
        if(!selected||!strstr(t->name,"attn."))continue;
        int projection=strstr(t->name,"attn.wq_")||strstr(t->name,"attn.wkv.")||strstr(t->name,"attn.wo_");
        int gamma=strstr(t->name,".q_norm.")||strstr(t->name,".kv_norm.")||strstr(t->name,"compressor.norm.")||strstr(t->name,"indexer.k_norm.");
        int dense=strstr(t->name,"compressor.wkv.")||strstr(t->name,"compressor.wgate.")||strstr(t->name,"indexer.wk.");
        if(!projection&&!gamma&&!dense)continue;
        unsigned char block[4096];if(projection)memset(block,t->dtype==BN_SAFETENSORS_F8_E8M0?127:32,sizeof(block));
        else for(size_t j=0;j<sizeof(block);j+=2){block[j]=gamma?128:0;block[j+1]=gamma?63:62;}
        assert(!fseek(f,(long)(base+t->offset),SEEK_SET));for(uint64_t left=t->bytes;left;){size_t n=left<sizeof(block)?(size_t)left:sizeof(block);assert(fwrite(block,1,n,f)==n);left-=n;}
    }
    assert(!fclose(f));snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);f=fopen(path,"w");assert(f);fprintf(f,"{\"metadata\":{\"total_size\":%llu},\"weight_map\":{",(unsigned long long)payload);for(size_t i=0;i<count;i++)fprintf(f,"%s\"%s\":\"one.safetensors\"",i?",":"",spec[i].name);fputs("}}",f);assert(!fclose(f));free(header);free(spec);
}
static void native_anatomy_fixture_shape(const char *dir,int variant,int index_heads,int index_dim) {native_anatomy_fixture_geometry(dir,variant,index_heads,index_dim,8,128,2);}
static void native_anatomy_fixture(const char*dir,int variant){native_anatomy_fixture_shape(dir,variant,2,32);}
static void native_plan_release_check(void *context) {
    BnModel *model=context;assert(bn_model_native_anatomy(model)&&bn_model_native_config(model)&&bn_model_trellis_expert_reader(model));
}
static void test_native_anatomy(void) {
    char dir[]="/tmp/bn-native-anatomy-XXXXXX",path[256];assert(mkdtemp(dir));snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);
    BnModelNativeAnatomy *plan=malloc(sizeof(*plan)),*saved=malloc(sizeof(*saved));assert(plan&&saved);BnJSON doc={0};assert(!bn_json_init(&doc,native_anatomy_config,strlen(native_anatomy_config)));
    for(int variant=0;variant<=5;variant++) {
        native_anatomy_fixture(dir,variant);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));memset(plan,0xa5,sizeof(*plan));memcpy(saved,plan,sizeof(*plan));int result=bn_model_arch_native_anatomy(&doc,&index,plan);
        if(variant){assert(result==-1&&!memcmp(plan,saved,sizeof(*plan)));BnModel failed={0};assert(!bn_model_load_trellis_config(&failed,&index,native_anatomy_config,strlen(native_anatomy_config)));BnModelRuntime *runtime=failed.runtime;assert(bn_model_prepare_native_anatomy(&failed)==-1&&failed.runtime==runtime&&!bn_model_native_anatomy(&failed)&&bn_model_trellis_expert_reader(&failed));bn_model_free(&failed);}
        else {
            assert(!result&&plan->core_format==BN_QUANT_FORMAT_E4M3_UE8M0_32);size_t bound=3;for(int l=0;l<40;l++)for(int r=0;r<BN_NATIVE_ROLE_COUNT;r++)if(plan->layers[l].tensor[r]>=0)bound++;assert(bound==1258);
            assert(plan->layers[0].kv_source==-1&&plan->layers[0].index_source==-1&&plan->layers[3].kv_source==2&&plan->layers[3].index_source==2&&plan->layers[25].kv_source==20&&plan->layers[25].index_source==24);
            assert(plan->layers[20].candidate_mode==BN_NATIVE_CANDIDATE_SOURCE&&plan->layers[24].candidate_mode==BN_NATIVE_CANDIDATE_USE&&plan->layers[25].candidate_mode==BN_NATIVE_CANDIDATE_NONE);
            assert(plan->layers[2].tensor[BN_NATIVE_COMP_GATE]>=0&&plan->layers[20].tensor[BN_NATIVE_COMP_GATE]==-1&&plan->layers[24].tensor[BN_NATIVE_INDEX_K]==-1&&plan->layers[1].engram_rows==64&&plan->layers[14].engram_rows==96);
            BnModel model={0};assert(!bn_model_load_trellis_config(&model,&index,native_anatomy_config,strlen(native_anatomy_config)));assert(!bn_model_prepare_native_anatomy(&model));const BnModelNativeAnatomy *owned=bn_model_native_anatomy(&model);assert(owned&&owned->embedding==plan->embedding&&owned->output==plan->output);for(int l=0;l<40;l++)for(int r=0;r<BN_NATIVE_ROLE_COUNT;r++)assert(owned->layers[l].tensor[r]==plan->layers[l].tensor[r]);assert(bn_model_prepare_native_anatomy(&model)==-1&&owned==bn_model_native_anatomy(&model));assert(!bn_model_ensure_backend(&model));assert(!bn_backend_model_attach_gpu_resources(bn_model_backend(&model),&model,&model,native_plan_release_check));bn_model_free(&model);assert(!model.runtime);
        }
        bn_safetensors_index_free(&index);
    }
    native_anatomy_fixture(dir,0);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    const char *changes[][2]={{"\"kv_source_layer_ids\":[2,8,14,20]","\"kv_source_layer_ids\":[2,8,14,21]"},{"\"index_source_layer_ids\":[2,8,14,20,24,28,32,36]","\"index_source_layer_ids\":[2,2,14,20,24,28,32,36]"},{"\"candidate_topk_blocks\":2048","\"candidate_topk_blocks\":1"},{"\"o_groups\":2","\"o_groups\":3"},{"\"hc_mult\":4","\"hc_mult\":17"}};
    for(size_t i=0;i<sizeof(changes)/sizeof(changes[0]);i++) {
        const char *match=strstr(native_anatomy_config,changes[i][0]);assert(match);char input[8192];int n=snprintf(input,sizeof(input),"%.*s%s%s",(int)(match-native_anatomy_config),native_anatomy_config,changes[i][1],match+strlen(changes[i][0]));assert(n>0&&(size_t)n<sizeof(input));BnJSON invalid={0};assert(!bn_json_init(&invalid,input,(size_t)n));memset(plan,0xa5,sizeof(*plan));memcpy(saved,plan,sizeof(*plan));assert(bn_model_arch_native_anatomy(&invalid,&index,plan)==-1&&!memcmp(plan,saved,sizeof(*plan)));bn_json_free(&invalid);
    }
    bn_safetensors_index_free(&index);bn_json_free(&doc);free(plan);free(saved);assert(!remove(path));snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!remove(path));assert(!remove(dir));puts("Native text storage anatomy, sharing schedule and atomic rejection PASSED");
}
#else
static void test_native_anatomy(void) {}
#endif
