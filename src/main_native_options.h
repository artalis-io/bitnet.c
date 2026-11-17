/* Main-owned options and bounded arithmetic; independent of model/backend. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "generate_batch_plan.h"
typedef struct {
    const char *directory,*profile,*prompt,*prompt_file,*trace_directory,*batch_file;
    int tokens,prefill,maxseq,ram_mb,gpu_mb,temporary_experts,window;
    int host_mb,device_mb,admit,pinned,keep_special,token_ids,prepare,predict,ram_admit;
    int batch_live,batch_budget,mixed_budget,serve;
    int host_budget_set,device_budget_set;
    int ram_system_pages;
} NativeCLIOptions;
static inline int native_cli_batch_mode(const NativeCLIOptions *o) {return o->batch_file||o->serve;}
static inline int native_cli_context(size_t prompt,int outputs,int predict,int capacity) {
    if(!prompt||outputs<0||(predict!=0&&predict!=1)||capacity<1||prompt>(size_t)capacity)return -1;
    size_t commits=(size_t)(outputs-(predict&&outputs>0));
    return commits<=(size_t)capacity-prompt?0:-1;
}
static inline int native_cli_add(size_t *sum,size_t value) {
    if(value>SIZE_MAX-*sum)return -1;
    *sum+=value;return 0;
}
static inline int native_cli_mul(size_t a,size_t b,size_t *out) {
    if(a&&b>SIZE_MAX/a)return -1;
    *out=a*b;return 0;
}
static inline int native_cli_piece_bound(size_t bytes,size_t *out) {
    if(bytes>(SIZE_MAX-6)/3)return -1;
    *out=6+3*bytes;return 0;
}
static inline int native_cli_cold_experts(int batch,int active,int experts,int window,int *out) {
    if(batch<1||active<1||active>INT_MAX/8||experts<active||window<1)return -1;
    int rows=batch<8?batch:8,need=rows*active;
    if(batch>8){int group=window<experts?window:experts;if(group>need)need=group;}
    *out=need;return 0;
}
static int native_cli_number(const char *text,int low,int high,int *out) {
    if(!text||!*text)return -1;
    unsigned n=0,limit=(unsigned)high;
    for(const unsigned char *p=(const unsigned char*)text;*p;p++) {
        if(*p<'0'||*p>'9')return -1;
        unsigned d=(unsigned)(*p-'0');
        if(n>limit/10||(n==limit/10&&d>limit%10))return -1;
        n=n*10+d;
    }
    if(n<(unsigned)low)return -1;
    *out=(int)n;return 0;
}
static void native_cli_usage(FILE *out,const char *program) {
    fprintf(out,"Usage: %s --native-text MODEL_DIR --profile exl3-int4-f16 (-p TEXT | --prompt-file FILE | --batch-prompts FILE | --serve-stdio) [options]\n",program);
    fputs("Raw UTF-8, greedy CUDA generation. No chat template, BOS insertion or automatic EOS stop.\n"
        "Profile: EXL3 experts, INT4 BF16 row lookup adapted to F16 Engram math; not original Diffbot full-runtime equivalence.\n"
        "  -n N                   New tokens (default 32)\n"
        "  --prefill-tokens N      Batch size, 1..16384 (default 512)\n"
        "  --maxseq N              Context capacity, multiple of 256 (default 4096)\n"
        "  --ram-cache-mb N        RAM expert payload MiB (default 640; 0 disables)\n"
        "  --gpu-cache-mb N        Decode cache payload MiB (default 640; 0 disables)\n"
        "  --temporary-experts N  Temporary expert slots, 48..4096 (default 48)\n"
        "  --fused-window N        Prefill upload window, 1..32 (default 32)\n"
        "  --host-budget-mb N      Aggregate host payload cap; batch default16384MiB\n"
        "  --device-budget-mb N    Aggregate device cap; batch default90% free at setup\n"
        "                         Explicit0 disables a cap; single-request defaults stay uncapped.\n"
        "  --admit-decode          Admit decode experts into the GPU cache\n"
        "  --ram-admit-misses      Admit RAM misses independently of GPU admission\n"
        "  --ram-page-policy MODE  huge (default): best-effort huge-page advice;\n"
        "                         system: omit advice and inherit OS page policy.\n"
        "                         Explicit policy requires a nonzero RAM cache.\n"
        "  --batch-prompts FILE    JSON array of 1..64 raw UTF-8 prompts; NDJSON prediction output\n"
        "  --serve-stdio           Persistent NDJSON submit/cancel/shutdown commands on stdin\n"
        "                         Records up to1MiB; EOF/shutdown drain accepted work.\n"
        "                         -n is the per-request ceiling; submit may set max_outputs.\n"
        "  --batch-live N          Maximum concurrent batch requests, 1..16 (default 2)\n"
        "  --mixed-token-budget N  Mixed prefill/decode step cap; 0 inherits full budget (default0)\n"
        "                         Positive values: batch-live..batch-token-budget; pure prefill unchanged.\n"
        "  --batch-token-budget N  Per-step tokens, up to 131072 (default prefill-tokens * batch-live)\n"
        "                         Batch prefill chunks are limited to 8192 tokens per request.\n"
        "  --pinned-staging        Use bounded pinned expert staging\n"
        "  --keep-special          Include special-token text in output\n"
        "  --token-ids             Report generated IDs on stderr\n"
        "  --prediction-first      Stream predictions before decode; skip the final decode\n"
        "  --prepare-only          Load/prepare bounded CUDA state and print payload JSON; no inference\n"
        "  --trace-dir DIR         Write exclusive diagnostic input/logit files in an existing directory\n"
        "Prompt files are counted bytes (up to 16 MiB); FILE=- reads stdin. Output is counted UTF-8 without an added newline.\n"
        "Payload caps exclude JSON/index/allocator/driver/library overhead and OS file cache.\n",out);
}
static int native_cli_parse(int argc,char **argv,NativeCLIOptions *out) {
    NativeCLIOptions o={0};o.tokens=32;o.prefill=512;o.maxseq=4096;o.ram_mb=o.gpu_mb=640;
    o.temporary_experts=48;o.window=32;uint32_t seen=0;
    o.batch_live=2;
    if(argc<3)return -1;
    if(!strcmp(argv[2],"--help"))return 1;
    o.directory=argv[2];if(!*o.directory)return -1;
    const char *names[]={"--profile","-p","--prompt-file","--trace-dir","-n","--prefill-tokens","--maxseq",
        "--ram-cache-mb","--gpu-cache-mb","--temporary-experts","--fused-window","--host-budget-mb","--device-budget-mb",
        "--admit-decode","--pinned-staging","--keep-special","--token-ids","--prepare-only","--prediction-first","--ram-admit-misses",
        "--batch-prompts","--batch-live","--batch-token-budget","--serve-stdio","--mixed-token-budget","--ram-page-policy"};
    for(int i=3;i<argc;i++) {
        if(!strcmp(argv[i],"--help"))return 1;
        unsigned key=0;for(;key<sizeof(names)/sizeof(names[0]);key++)if(!strcmp(argv[i],names[key]))break;
        if(key==sizeof(names)/sizeof(names[0])||(seen&(UINT32_C(1)<<key)))return -1;
        seen|=UINT32_C(1)<<key;
        if(key==23){o.serve=1;continue;}
        const char *value=NULL;
        if(key<13||key>=20){if(++i==argc||!*argv[i])return -1;value=argv[i];}
        if(key<4) {
            if(key==0)o.profile=value;else if(key==1)o.prompt=value;else if(key==2)o.prompt_file=value;else o.trace_directory=value;
        }else if(key<13) {
            int *fields[]={&o.tokens,&o.prefill,&o.maxseq,&o.ram_mb,&o.gpu_mb,&o.temporary_experts,&o.window,&o.host_mb,&o.device_mb};
            const int low[]={0,1,256,0,0,48,1,0,0},high[]={INT_MAX,16384,INT_MAX,INT_MAX,INT_MAX,4096,32,INT_MAX,INT_MAX};
            if(native_cli_number(value,low[key-4],high[key-4],fields[key-4]))return -1;
            if(key==11)o.host_budget_set=1;
            if(key==12)o.device_budget_set=1;
        }else if(key<20) {
            int *flags[]={&o.admit,&o.pinned,&o.keep_special,&o.token_ids,&o.prepare,&o.predict,&o.ram_admit};*flags[key-13]=1;
        }else if(key==20)o.batch_file=value;
        else if(key==24){if(native_cli_number(value,0,BN_GENERATE_BATCH_MAX_TOKENS,&o.mixed_budget))return -1;}
        else if(key==25){
            if(!strcmp(value,"huge"))o.ram_system_pages=0;
            else if(!strcmp(value,"system"))o.ram_system_pages=1;
            else return -1;
        }
        else if(native_cli_number(value,1,key==21?BN_GENERATE_BATCH_MAX_REQUESTS:BN_GENERATE_BATCH_MAX_TOKENS,key==21?&o.batch_live:&o.batch_budget))return -1;
    }
    int sources=(o.prompt!=NULL)+(o.prompt_file!=NULL)+(o.batch_file!=NULL)+o.serve;
    if(!o.profile||strcmp(o.profile,"exl3-int4-f16")||sources!=1||o.maxseq%256||(o.admit&&!o.gpu_mb)||(o.ram_admit&&!o.ram_mb))return -1;
    if((seen&(UINT32_C(1)<<25))&&!o.ram_mb)return -1;
    if(native_cli_batch_mode(&o)) {
        if(!o.tokens||o.prepare||o.token_ids||o.prefill>BN_GENERATE_BATCH_MAX_CHUNK_TOKENS)return -1;
        if(!o.batch_budget)o.batch_budget=o.prefill*o.batch_live;
        if(o.batch_budget<o.batch_live||(o.mixed_budget&&(o.mixed_budget<o.batch_live||o.mixed_budget>o.batch_budget)))return -1;
    }else if(seen&((UINT32_C(1)<<21)|(UINT32_C(1)<<22)|(UINT32_C(1)<<24)))return -1;
    *out=o;return 0;
}
