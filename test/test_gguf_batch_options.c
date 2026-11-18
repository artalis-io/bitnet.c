#include "../src/main_native_options.h"
#include "../src/main_gguf_batch_options.h"
#include "transformer/bf16_internal.h"
#include "json.h"
#include "../src/main_bf16_policy.h"
#include <assert.h>
static void options(void) {
    GGUFBatchOptions o;
    char *args[]={"bitnet","--cuda-batch","model","--profile","dense-bf16","--batch-prompts","p.json","--batch-live","16","--prefill-tokens","2048"};
    assert(!gguf_batch_parse(11,args,&o)&&o.live==16&&o.budget==8192&&o.prefill==2048&&o.threads==8);
    GGUFBatchOptions before=o;
    char *bad[][14]={
        {"b","--cuda-batch","m","--profile","unknown","--batch-prompts","x"},
        {"b","--cuda-batch","m","--profile","dense-bf16","--batch-prompts","x","--batch-live","17"},
        {"b","--cuda-batch","m","--profile","dense-bf16","--batch-prompts","x","--batch-token-budget","8193"},
        {"b","--cuda-batch","m","--profile","dense-bf16","--batch-prompts","x","--gpu-cache-mb","1"},
        {"b","--cuda-batch","m","--profile","glm-recurrent","--batch-prompts","x","--policy","p"},
        {"b","--cuda-batch","m","--profile","flash-bf16","--batch-prompts","x","--kv-budget-mb","1"},
        {"b","--cuda-batch","m","--profile","dense-bf16","--batch-prompts","x","--request-device-mb","0"},
        {"b","--cuda-batch","m","--profile","dense-bf16","--batch-prompts","x","--batch-live","1","--batch-live","2"},
        {"b","--cuda-batch","m","--profile","dense-bf16","--batch-prompts","x","-n","0"},
        {"b","--cuda-batch","m","--profile","dense-bf16","--batch-prompts","x","-t","0"},
    };
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++) {
        int n=0;while(n<14&&bad[i][n])n++;
        assert(gguf_batch_parse(n,bad[i],&o)==-1&&!memcmp(&o,&before,sizeof(o)));
    }
    args[4]="flash-bf16";assert(!gguf_batch_parse(11,args,&o)&&o.budget==32768);
    args[4]="glm-recurrent";assert(!gguf_batch_parse(11,args,&o)&&o.budget==32768);
    char *memory[]={"b","--cuda-batch","m","--profile","flash-bf16","--batch-prompts","x","--serving-host-mb","700","--serving-device-mb","14000"};
    assert(!gguf_batch_parse(11,memory,&o)&&o.serving_host_mb==700&&o.serving_device_mb==14000);
    memory[4]="dense-bf16";assert(!gguf_batch_parse(11,memory,&o)&&o.serving_host_mb==700&&o.serving_device_mb==14000);before=o;
    memory[4]="glm-recurrent";memory[8]="0";assert(gguf_batch_parse(11,memory,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    char *cache[]={"b","--cuda-batch","m","--profile","glm-recurrent","--batch-prompts","x","--gpu-cache-mb","32"};
    assert(!gguf_batch_parse(9,cache,&o)&&o.gpu_mb==32);
    FILE *f=tmpfile();assert(f);gguf_batch_usage(f,"bitnet");assert(ftell(f)>0&&!fclose(f));
}
static void policy(void) {
    char data[4096];size_t n=(size_t)snprintf(data,sizeof(data),"{\"arithmetic\":5,\"ffn_arithmetic\":2,\"input_norm_reductions\":[");
    for(int i=0;i<65;i++)n+=(size_t)snprintf(data+n,sizeof(data)-n,"%s0",i?",":"");
    n+=(size_t)snprintf(data+n,sizeof(data)-n,"],\"post_norm_reductions\":[");
    for(int i=0;i<64;i++)n+=(size_t)snprintf(data+n,sizeof(data)-n,"%s0",i?",":"");
    n+=(size_t)snprintf(data+n,sizeof(data)-n,"],\"attention_split_policy\":1,\"gdn_prefill_selection\":1}");
    BnTransformerBF16Policy p={0};assert(!gguf_batch_policy_parse(data,n,&p)&&p.arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32&&p.ffn_arithmetic==2&&p.gdn_prefill_selection==1);
    BnTransformerBF16Policy before=p;
    assert(gguf_batch_policy_parse(data,n-1,&p)==-1&&!memcmp(&p,&before,sizeof(p)));
    assert(gguf_batch_policy_parse("{}",2,&p)==-1&&!memcmp(&p,&before,sizeof(p)));
    assert(gguf_batch_policy_parse(data,16385,&p)==-1&&!memcmp(&p,&before,sizeof(p)));
}
static void serving_options(void) {
    GGUFBatchOptions o;char *args[]={"b","--cuda-batch","m","--profile","dense-bf16","--serve-stdio"};
    const char *profiles[]={"dense-bf16","flash-bf16","glm-recurrent"};
    for(int i=0;i<3;i++){args[4]=(char*)profiles[i];assert(!gguf_batch_parse(6,args,&o)&&o.serve&&!o.file&&o.tokens==32&&o.live==2);}
    GGUFBatchOptions before=o;
    char *bad[][10]={
        {"b","--cuda-batch","m","--profile","dense-bf16","--serve-stdio","--batch-prompts","x"},
        {"b","--cuda-batch","m","--profile","dense-bf16","--serve-stdio","--serve-stdio"},
        {"b","--cuda-batch","m","--profile","dense-bf16","--serve-stdio","-n","0"},
    };
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++){int n=0;while(n<10&&bad[i][n])n++;assert(gguf_batch_parse(n,bad[i],&o)==-1&&!memcmp(&before,&o,sizeof(o)));}
}

static void mixed_options(void) {
    GGUFBatchOptions o;
    char *args[]={"b","--cuda-batch","m","--profile","dense-bf16","--serve-stdio","--batch-live","4","--batch-token-budget","16","--mixed-token-budget","8"};
    const char *profiles[]={"dense-bf16","flash-bf16","glm-recurrent"};
    for(size_t i=0;i<3;i++){args[4]=(char*)profiles[i];assert(!gguf_batch_parse(12,args,&o)&&o.mixed_budget==8);}
    const char *good[]={"0","4","16"};
    for(size_t i=0;i<sizeof(good)/sizeof(good[0]);i++){args[11]=(char*)good[i];assert(!gguf_batch_parse(12,args,&o)&&o.mixed_budget==(i?i==1?4:16:0));}
    GGUFBatchOptions before=o;
    const char *bad[]={"-1","3","17","131073","1x"};
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++){args[11]=(char*)bad[i];assert(gguf_batch_parse(12,args,&o)==-1&&!memcmp(&before,&o,sizeof(o)));}
    args[11]="8";assert(gguf_batch_parse(11,args,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    char *duplicate[]={"b","--cuda-batch","m","--profile","dense-bf16","--serve-stdio","--mixed-token-budget","8","--mixed-token-budget","8"};
    assert(gguf_batch_parse(10,duplicate,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    char *file[]={"b","--cuda-batch","m","--profile","dense-bf16","--batch-prompts","p.json","--mixed-token-budget","8"};
    assert(!gguf_batch_parse(9,file,&o)&&o.mixed_budget==8);
}

int main(void){options();policy();serving_options();mixed_options();(void)native_cli_parse;(void)native_cli_usage;puts("GGUF batch options/profile limits and atomic arithmetic metadata PASSED");return 0;}
