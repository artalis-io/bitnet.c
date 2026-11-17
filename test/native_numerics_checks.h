#ifndef __EMSCRIPTEN__
#include <fenv.h>
static char *native_numerics_replace(const char *source,const char *from,const char *to) {
    const char *at=strstr(source,from);assert(at);size_t prefix=(size_t)(at-source),tail=strlen(at+strlen(from));
    char *s=malloc(prefix+strlen(to)+tail+1);assert(s);memcpy(s,source,prefix);memcpy(s+prefix,to,strlen(to));memcpy(s+prefix+strlen(to),at+strlen(from),tail+1);return s;
}
static void native_numerics_check(const char *config,int accepted) {
    BnJSON doc={0};assert(!bn_json_init(&doc,config,strlen(config)));BnModelNativeNumerics p,saved;
    memset(&p,0xa5,sizeof(p));saved=p;int result=bn_model_arch_native_numerics(&doc,&p);
    if(!accepted)assert(result==-1&&!memcmp(&p,&saved,sizeof(p)));
    else {
        assert(!result&&p.hc_carry_pre==1&&p.hc_streams==4&&p.hc_iterations==20&&p.norm_topk==1);
        uint32_t bits;memcpy(&bits,&p.rms_epsilon,4);assert(bits==UINT32_C(0x1e3ce508));memcpy(&bits,&p.hc_epsilon,4);assert(bits==UINT32_C(0x358637bd));
        assert(p.swiglu_limit==10&&p.routed_scale==1.5f&&p.rope_theta==10000&&p.compress_rope_theta==160000&&p.yarn_factor==16&&p.yarn_beta_fast==32&&p.yarn_beta_slow==1);
        assert(p.yarn_original_context==65536&&p.context==1048576&&p.bos_token==0&&p.eos_token==1&&p.pad_token==2&&p.engram_vocab==16000000&&p.engram_compressed_vocab==256&&p.engram_pad_token==2);
        assert(p.rope_attention_factor==1&&!p.main_rope_yarn&&p.compress_rope_yarn&&p.yarn_truncate);
    }
    bn_json_free(&doc);
}
static void native_numerics_release(void *context) {
    BnModel *m=context;assert(bn_model_native_numerics(m)&&bn_model_native_anatomy(m)&&bn_model_native_config(m));
}
static void native_frequency_metadata_check(void) {
    int original=fegetround();assert(!fesetround(FE_TONEAREST));
    BnModelNativeNumerics p={0};p.rope_theta=10000;p.compress_rope_theta=160000;p.compress_rope_yarn=1;
    p.yarn_factor=16;p.yarn_beta_fast=32;p.yarn_beta_slow=1;p.yarn_original_context=65536;p.yarn_truncate=1;
    BnModelNativeRopeFrequency f,saved;memset(&saved,0xa5,sizeof(saved));f=saved;
    assert(!bn_model_arch_native_rope_frequency(64,&p,0,&f)&&f.rotary_dim==64&&!f.yarn&&f.base==10000&&f.factor==1&&f.ramp_start==0&&f.ramp_scale==1);
    assert(!bn_model_arch_native_rope_frequency(64,&p,1,&f)&&f.yarn&&f.base==160000&&f.factor==16&&f.ramp_start==15&&f.ramp_scale==.1f);
    f=saved;assert(bn_model_arch_native_rope_frequency(63,&p,1,&f)==-1&&!memcmp(&f,&saved,sizeof(f)));
    const int modes[]={FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    for(int i=0;i<3;i++){assert(!fesetround(modes[i]));assert(bn_model_arch_native_rope_frequency(64,&p,1,&f)==-1&&fegetround()==modes[i]&&!memcmp(&f,&saved,sizeof(f)));}assert(!fesetround(FE_TONEAREST));
    p.yarn_factor=0;assert(bn_model_arch_native_rope_frequency(64,&p,1,&f)==-1&&!memcmp(&f,&saved,sizeof(f)));assert(!fesetround(original));
}
static void test_native_numerics(void) {
    native_frequency_metadata_check();
    char *base=native_numerics_replace(native_anatomy_config,"\"engram_compressed_vocab_size\":99092","\"engram_compressed_vocab_size\":256");
    const char *equivalent[]={"1e-20","1.000000e-20","10e-21","1000e-23","0.00000000000000000001","0.0000000000000000000100","1E-00000000000000000000000020"};
    int original=fegetround();const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    for(size_t m=0;m<4;m++){assert(!fesetround(modes[m]));for(size_t i=0;i<sizeof(equivalent)/sizeof(equivalent[0]);i++) {
        char replacement[256];snprintf(replacement,sizeof(replacement),"\"rms_norm_eps\":%s",equivalent[i]);char *s=native_numerics_replace(base,"\"rms_norm_eps\":1e-20",replacement);native_numerics_check(s,1);free(s);
    }}assert(!fesetround(original));
    const char *bad[][2]={
        {"\"rms_norm_eps\":1e-20","\"rms_norm_eps\":1e-19"},{"\"rms_norm_eps\":1e-20","\"rms_norm_eps\":-1e-20"},
        {"\"rms_norm_eps\":1e-20","\"rms_norm_eps\":1e-200000000000000000000"},{"\"rms_norm_eps\":1e-20","\"rms_norm_eps\":1e20000000000000000000"},
        {"\"rms_norm_eps\":1e-20","\"rms_norm_eps\":1.00000000000000000000001e-20"},{"\"rms_norm_eps\":1e-20","\"rms_norm_eps\":\"1e-20\""},
        {"\"hc_eps\":1e-06","\"hc_eps\":0"},{"\"hc_mult\":4","\"hc_mult\":3"},{"\"hc_sinkhorn_iters\":20","\"hc_sinkhorn_iters\":19"},
        {"\"swiglu_limit\":10.0","\"swiglu_limit\":0"},{"\"routed_scaling_factor\":1.5","\"routed_scaling_factor\":1"},
        {"\"rope_theta\":10000","\"rope_theta\":100000"},{"\"compress_rope_theta\":160000","\"compress_rope_theta\":10000"},
        {"\"factor\":16","\"factor\":8"},{"\"beta_fast\":32","\"beta_fast\":16"},{"\"beta_slow\":1","\"beta_slow\":2"},
        {"\"original_max_position_embeddings\":65536","\"original_max_position_embeddings\":32768"},
        {"\"rope_type\":\"yarn\"","\"rope_type\":\"linear\""},{"\"rope_type\":\"yarn\"","\"rope_type\":\"yarn\",\"attention_factor\":1"},
        {"\"attention_dropout\":0.0","\"attention_dropout\":0.1"},{"\"use_cache\":true","\"use_cache\":false"},
        {"\"norm_topk_prob\":true","\"norm_topk_prob\":false"},{"\"dtype\":\"bfloat16\"","\"dtype\":\"float16\""},
        {"\"attention_bias\":false","\"attention_bias\":true"},{"\"tie_word_embeddings\":false","\"tie_word_embeddings\":true"},
        {"\"engram_compressed_vocab_size\":256","\"engram_compressed_vocab_size\":0"},
        {"\"engram_pad_token_id\":2","\"engram_pad_token_id\":256"},{"\"bos_token_id\":0","\"bos_token_id\":256"},
        {"\"max_position_embeddings\":1048576","\"max_position_embeddings\":0"},
        {"\"hc_mult\":4","\"num_hash_layers\":1,\"hc_mult\":4"}
    };
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++){char *s=native_numerics_replace(base,bad[i][0],bad[i][1]);native_numerics_check(s,0);free(s);}
    char *zero=native_numerics_replace(base,"\"attention_dropout\":0.0","\"attention_dropout\":-0e999999999999999999999");native_numerics_check(zero,1);free(zero);
    char dir[]="/tmp/bn-native-numerics-XXXXXX",path[256];assert(mkdtemp(dir));native_anatomy_fixture(dir,0);snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    BnModel model={0};assert(!bn_model_native_numerics(NULL)&&bn_model_prepare_native_numerics(&model)==-1);assert(!bn_model_load_trellis_config(&model,&index,base,strlen(base)));assert(bn_model_prepare_native_numerics(&model)==-1&&!bn_model_native_numerics(&model));assert(!bn_model_prepare_native_anatomy(&model));assert(!bn_model_prepare_native_numerics(&model));
    BnModelNativeRopeFrequency frequency;assert(!bn_model_native_rope_frequency(&model,0,&frequency)&&frequency.rotary_dim==32&&!frequency.yarn&&frequency.base==10000);
    const BnModelNativeNumerics *owned=bn_model_native_numerics(&model);assert(owned&&owned->hc_carry_pre);assert(bn_model_prepare_native_numerics(&model)==-1&&owned==bn_model_native_numerics(&model));assert(!bn_model_ensure_backend(&model));assert(!bn_backend_model_attach_gpu_resources(bn_model_backend(&model),&model,&model,native_numerics_release));bn_model_free(&model);assert(!bn_model_native_numerics(&model));
    char *invalid=native_numerics_replace(base,"\"rms_norm_eps\":1e-20","\"rms_norm_eps\":1e-6");assert(!bn_model_load_trellis_config(&model,&index,invalid,strlen(invalid)));assert(!bn_model_prepare_native_anatomy(&model));const BnModelNativeAnatomy *anatomy=bn_model_native_anatomy(&model);assert(bn_model_prepare_native_numerics(&model)==-1&&!bn_model_native_numerics(&model)&&anatomy==bn_model_native_anatomy(&model)&&bn_model_trellis_expert_reader(&model));bn_model_free(&model);free(invalid);
    bn_safetensors_index_free(&index);snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!remove(path));snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!remove(path));assert(!rmdir(dir));free(base);puts("Native numerical profile/decimal identity/lifecycle PASSED");
}
#else
static void test_native_numerics(void) {}
#endif
