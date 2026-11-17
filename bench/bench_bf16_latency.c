/* Loaded-model greedy token delivery timings. Tokenization, allocation/setup,
 * and JSON output are excluded. TTFT includes request restart, prefill, readback and
 * greedy sampling; decode excludes the first prefill-sampled output token. */
#include "model.h"
#include "model_internal.h"
#include "session.h"
#include "session_internal.h"
#include "generate.h"
#include "transformer.h"
#include "platform.h"
#include "backend_layout.h"
#include "bf16_policy_io.h"
#ifdef BN_ENABLE_CUDA
#include "gpu_cuda.h"
#endif
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef BN_ENABLE_CUDA
static int number(const char *text,int minimum,int *out) {
    char *end=NULL;errno=0;long value=strtol(text,&end,10);
    if(errno||end==text||*end||value<minimum||value>INT_MAX/2)return -1;
    *out=(int)value;return 0;
}
static int greedy(const float *logits,int n) {
    if(!logits)return -1;
    int id=-1;float best=-INFINITY;
    for(int i=0;i<n;i++)if(logits[i]>best){best=logits[i];id=i;}
    return isfinite(best)?id:-1;
}
static int reject_cpu(void *ctx) {
    (void)ctx;fprintf(stderr,"native benchmark attempted CPU fallback\n");return -1;
}
static int *read_prompt(const char *path,int *count) {
    FILE *f=fopen(path,"rb");if(!f)return NULL;
    int *ids=NULL;
    if(fseek(f,0,SEEK_END))goto done;
    long bytes=ftell(f);
    if(bytes<=0||bytes%4||bytes/4>INT_MAX/2||fseek(f,0,SEEK_SET))goto done;
    *count=(int)(bytes/4);ids=malloc((size_t)*count*sizeof(*ids));if(!ids)goto done;
    for(int i=0;i<*count;i++) {
        unsigned char b[4];
        if(fread(b,1,4,f)!=4){free(ids);ids=NULL;goto done;}
        uint32_t value=(uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;
        if(value>INT_MAX){free(ids);ids=NULL;goto done;}ids[i]=(int)value;
    }
done:
    fclose(f);return ids;
}
static void weight(const BnGPUBF16LinearWeight *w,const char *role,int layer) {
    if(!w->raw&&!w->decoded)return;
    printf("{\"kind\":\"weight\",\"layer\":%d,\"role\":\"%s\",\"format\":%d,\"raw\":%s,\"decoded\":%s}\n",
        layer,role,w->type,w->raw?"true":"false",w->decoded?"true":"false");
}
#endif

int main(int argc,char **argv) {
#ifndef BN_ENABLE_CUDA
    (void)argc;(void)argv;fprintf(stderr,"requires BN_ENABLE_CUDA=1\n");return 2;
#else
    int output=0,runs=0,warmups=0,chunk=0,capacity=0,np=0,status=1;
    BnGGUFFile *file=NULL;BnGPUBackend *gpu=NULL;BnModel model={0};BnSession *session=NULL;
    int *prompt=NULL,*ids=NULL;double *steps=NULL;
    if((argc!=8&&argc!=9&&argc!=10)||number(argv[3],2,&output)||number(argv[4],1,&runs)||
       number(argv[5],0,&warmups)||number(argv[6],1,&chunk)||number(argv[7],1,&capacity)||
       runs>INT_MAX-warmups||chunk>capacity) {
        fprintf(stderr,"usage: %s MODEL.gguf PROMPT.i32 OUTPUTS RUNS WARMUPS PREFILL_CHUNK MAXSEQ [eager|fp32-intermediates|tiled-fp32|cp-fp32|cp-fp32-gated-r2|noncp-fp32] [LOOPED_ADD_MASK_HEX] | --policy POLICY.json\n",argv[0]);return 2;
    }
    BnTransformerBF16Arithmetic arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_EAGER;
    int policy_file=argc>=9&&!strcmp(argv[8],"--policy");
    if(policy_file&&argc!=10){fprintf(stderr,"--policy requires one file\n");return 2;}
    if(argc>=9&&!policy_file) {
        if(!strcmp(argv[8],"fp32-intermediates"))arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_FP32_INTERMEDIATES;
        else if(!strcmp(argv[8],"cp-fp32"))arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32;
        else if(!strcmp(argv[8],"cp-fp32-gated-r2"))arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32_GATED_R2;
        else if(!strcmp(argv[8],"noncp-fp32"))arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32;
        else if(!strcmp(argv[8],"tiled-fp32"))arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32;
        else if(strcmp(argv[8],"eager")){fprintf(stderr,"unknown arithmetic profile\n");return 2;}
    }
    uint64_t norm_loop_mask=0;
    if(argc==10&&!policy_file) {
        char *end=NULL;errno=0;
        unsigned long long value=strtoull(argv[9],&end,16);
        if(errno||end==argv[9]||*end||argv[9][0]=='-')return 2;
        norm_loop_mask=(uint64_t)value;
        if(norm_loop_mask&&arithmetic<BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32)return 2;
    }
    BnTransformerBF16Policy policy={0};policy.arithmetic=arithmetic;
    for(int l=0;l<64;l++)if(norm_loop_mask&(UINT64_C(1)<<l))
        policy.post_norm_reductions[l]=BN_GPU_RMSNORM_REDUCTION_TILED_5120_LOOPED_ADD_FMA;
    if(policy_file) {
        if(bench_policy_read(argv[9],&policy)){fprintf(stderr,"invalid BF16 policy file\n");return 2;}
        arithmetic=policy.arithmetic;
    }
    prompt=read_prompt(argv[2],&np);
    if(!prompt||np>capacity||output>capacity-np)goto done;
    ids=malloc((size_t)output*sizeof(*ids));steps=malloc((size_t)(output-1)*sizeof(*steps));
    if(!ids||!steps)goto done;
    file=bn_gguf_open_file(argv[1]);if(!file)goto done;
    if(bn_model_load_with_cpu_preparation(&model,file,capacity,0,0,0))goto done;
    for(int i=0;i<np;i++)if(prompt[i]>=model.config.vocab_size)goto done;
    gpu=bn_gpu_cuda_create();if(!gpu)goto done;
    if(bn_backend_layout_prepare_bf16_with_math(bn_model_backend(&model),gpu,&model.config,
       &model.weights,chunk,BN_BACKEND_BF16_MATH_MKL_VML_HA))goto done;
    const BnBackendBF16Model *layout=bn_backend_layout_bf16(bn_model_backend(&model));
    if(!layout)goto done;
    session=bn_session_create_backend(&model,NULL);if(!session||!session->backend_only)goto done;
    if(bn_transformer_bf16_request_init_with_policy(&model,gpu,bn_session_backend(session),chunk,capacity,NULL,0,&policy))goto done;
    BnTransformerBF16Policy admitted={0};
    if(bn_transformer_bf16_policy(bn_session_backend(session),&admitted))goto done;
    gpu->prepare_cpu_operations=reject_cpu;
    printf("{\"kind\":\"config\",\"prompt_tokens\":%d,\"output_tokens\":%d,\"layers\":%d,\"maxseq\":%d,\"prefill_chunk\":%d,\"backend_only\":true,\"math\":\"MKL_VML_HA\",\"bf16_gemm_reduction\":\"fp32\",\"arithmetic_profile\":\"%s\",\"clock\":\"CLOCK_MONOTONIC\"}\n",np,output,layout->n_layers,capacity,chunk,arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_EAGER?"eager":arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_FP32_INTERMEDIATES?"fp32-intermediates":arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_TILED_FP32?"tiled-fp32":arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32?"cp-fp32":arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32_GATED_R2?"cp-fp32-gated-r2":"noncp-fp32");
    fputs("{\"kind\":\"bf16_policy\",\"admitted\":",stdout);
    if(bench_policy_write(stdout,&admitted))goto done;
    puts("}");
    weight(&layout->embedding,"embedding",-1);weight(&layout->output,"output",-1);
    for(int i=0;i<layout->n_layers;i++) {
        const BnBackendBF16Layer *l=&layout->layers[i];
        weight(&l->qkv,"qkv",i);weight(&l->z,"z",i);weight(&l->ba,"ba",i);weight(&l->o,"o",i);
        weight(&l->q,"q",i);weight(&l->k,"k",i);weight(&l->v,"v",i);
        weight(&l->ffn.gate,"ffn_gate",i);weight(&l->ffn.up,"ffn_up",i);weight(&l->ffn.down,"ffn_down",i);
    }
    for(int run=0;run<warmups+runs;run++) {
        double start=bn_platform_time_ms();
        if(run&&bn_transformer_bf16_session_restart(&model,session))goto done;
        ids[0]=greedy(bn_prefill(&model,session,prompt,np,0,0),model.config.vocab_size);
        if(ids[0]<0)goto done;
        double first=bn_platform_time_ms(),last=first;
        for(int i=1;i<output;i++) {
            ids[i]=greedy(bn_transformer_forward(&model,session,ids[i-1],np+i-1),model.config.vocab_size);
            if(ids[i]<0)goto done;
            double now=bn_platform_time_ms();steps[i-1]=now-last;last=now;
        }
        if(bn_transformer_bf16_position(bn_session_backend(session))!=np+output-1)goto done;
        printf("{\"kind\":\"sample\",\"run\":%d,\"warmup\":%s,\"ttft_ms\":%.9f,\"decode_mean_ms\":%.9f,\"decode_total_ms\":%.9f,\"tokens\":[",
            run<warmups?run:run-warmups,run<warmups?"true":"false",first-start,(last-first)/(output-1),last-first);
        for(int i=0;i<output;i++)printf("%s%d",i?",":"",ids[i]);
        printf("],\"decode_steps_ms\":[");
        for(int i=0;i<output-1;i++)printf("%s%.9f",i?",":"",steps[i]);
        puts("]}");fflush(stdout);
    }
    status=0;
done:
    if(status)fprintf(stderr,"native BF16 latency benchmark failed\n");
    bn_session_free(session,NULL);bn_model_free(&model);
    if(gpu)bn_gpu_cuda_destroy(gpu);
    if(file)bn_gguf_free(file);
    free(prompt);free(ids);free(steps);return status;
#endif
}
