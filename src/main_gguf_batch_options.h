/* Strict product options, shared scheduler limits; no model/backend work. */
typedef struct {
    const char *model,*profile,*file,*trace,*policy;
    int live,prefill,budget,maxseq,tokens,kv_mb,ram_mb,gpu_mb,device_mb,host_mb,bos,threads,serving_host_mb,serving_device_mb,mixed_budget,serve;
    unsigned seen;
} GGUFBatchOptions;
static int gguf_batch_parse(int argc,char **argv,GGUFBatchOptions *out) {
    GGUFBatchOptions o={0};o.live=2;o.prefill=512;o.maxseq=4096;o.tokens=32;
    o.serving_host_mb=16384;
    o.kv_mb=4096;o.ram_mb=2048;o.device_mb=16384;o.host_mb=256;o.threads=8;
    if(argc<3)return -1;
    if(!strcmp(argv[2],"--help"))return 1;
    o.model=argv[2];if(!*o.model)return -1;
    const char *names[]={"--profile","--batch-prompts","--trace-dir","--policy","--batch-live",
        "--prefill-tokens","--batch-token-budget","--maxseq","-n","--kv-budget-mb","--ram-cache-mb",
        "--gpu-cache-mb","--request-device-mb","--request-host-mb","--bos","-t","--serving-host-mb","--serving-device-mb","--serve-stdio","--mixed-token-budget"};
    for(int i=3;i<argc;i++) {
        if(!strcmp(argv[i],"--help"))return 1;
        unsigned key=0;for(;key<20;key++)if(!strcmp(argv[i],names[key]))break;
        if(key==20||(o.seen&(1u<<key)))return -1;
        o.seen|=1u<<key;
        if(key==18){o.serve=1;continue;}
        if(key==14){o.bos=1;continue;}
        if(++i==argc||!*argv[i])return -1;
        if(key==19){if(native_cli_number(argv[i],0,BN_GENERATE_BATCH_MAX_TOKENS,&o.mixed_budget))return -1;continue;}
        if(key==15){if(native_cli_number(argv[i],1,256,&o.threads))return -1;continue;}
        if(key>=16){int *field=key==16?&o.serving_host_mb:&o.serving_device_mb;
            if(native_cli_number(argv[i],1,INT_MAX,field))return -1;
            continue;}
        if(key<4){const char **fields[]={&o.profile,&o.file,&o.trace,&o.policy};*fields[key]=argv[i];}
        else {
            int *fields[]={&o.live,&o.prefill,&o.budget,&o.maxseq,&o.tokens,&o.kv_mb,&o.ram_mb,&o.gpu_mb,&o.device_mb,&o.host_mb};
            const int low[]={1,1,1,1,1,1,0,0,1,1};
            const int high[]={16,BN_GENERATE_BATCH_MAX_CHUNK_TOKENS,BN_GENERATE_BATCH_MAX_TOKENS,INT_MAX,INT_MAX,INT_MAX,INT_MAX,INT_MAX,INT_MAX,INT_MAX};
            if(native_cli_number(argv[i],low[key-4],high[key-4],fields[key-4]))return -1;
        }
    }
    if(!o.profile||((o.file!=NULL)+o.serve)!=1)return -1;
    int dense=!strcmp(o.profile,"dense-bf16"),flash=!strcmp(o.profile,"flash-bf16"),glm=!strcmp(o.profile,"glm-recurrent");
    if(!dense&&!flash&&!glm)return -1;
    if(!o.budget){o.budget=o.prefill*o.live;if(dense&&o.budget>8192)o.budget=8192;}
    if(o.budget<o.live||(dense&&o.budget>8192)||(o.mixed_budget&&(o.mixed_budget<o.live||o.mixed_budget>o.budget)))return -1;
    if(!dense&&(o.policy||(o.seen&(1u<<9))))return -1;
    if(!flash&&o.gpu_mb)return -1;
    *out=o;return 0;
}
static void gguf_batch_usage(FILE *out,const char *name) {
    fprintf(out,"Usage: %s --cuda-batch MODEL.gguf --profile dense-bf16|flash-bf16|glm-recurrent (--batch-prompts FILE | --serve-stdio) [options]\n",name);
    fputs("JSON array of1..64 raw UTF8 prompts; FILE=- reads stdin. Length-bounded greedy NDJSON output.\n"
        "  --serve-stdio           Persistent NDJSON submit/cancel/shutdown commands on stdin\n"
        "                          Records up to1MiB; EOF/shutdown drain accepted work.\n"
        "                          -n is the per-request ceiling; submit may set max_outputs.\n"
        "  --batch-live N           Concurrent requests1..16 (default2)\n"
        "  --prefill-tokens N       Chunk ceiling1..8192 (default512)\n"
        "  --batch-token-budget N   Per-step tokens; dense packed maximum8192\n"
        "  --mixed-token-budget N   Mixed prefill/decode cap; 0 inherits full budget (default0)\n"
        "                          Positive values: batch-live..batch-token-budget; pure prefill unchanged.\n"
        "  --maxseq N               Context cap (default4096)\n"
        "  -n N                     Predictions, positive (default32); last is not forwarded\n"
        "  --kv-budget-mb N         Dense pool KV payload cap, idle included (default4096)\n"
        "  --ram-cache-mb N         Shared MoE source payload cache (default2048)\n"
        "  --gpu-cache-mb N         Flash expert GPU payload cache (default0)\n"
        "  --serving-host-mb N      Shared+retained serving payload cap (default16384)\n"
        "  --serving-device-mb N    Shared+retained device cap (default90% free at setup)\n"
        "  --request-device-mb N    Request device payload cap (default16384)\n"
        "  --request-host-mb N      Request host/staging cap (default256)\n"
        "  --policy FILE            Dense explicit arithmetic/reduction JSON contract\n"
        "  --trace-dir DIR          Existing directory, exclusive per-request F32/input traces\n"
        "  -t N                     CPU preparation workers (default8)\n"
        "  --bos                    Honor GGUF add_bos metadata; otherwise raw encoding\n"
        "Dense uses packed GPU execution/pooling. Flash/GLM cohorts share expert uploads and joint FFN GPU launches.\n"
        "All profiles retain healthy idle requests within aggregate budgets. Singletons use counted per-session forward fallback. Flash GPU caches share union leases across cohorts.\n"
        "No chat template, EOS stop, sampling penalty or implicit CPU fallback. Payload caps are not total RSS/VRAM bounds.\n",out);
}
