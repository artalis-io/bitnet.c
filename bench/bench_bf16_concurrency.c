/* Serialized multi-request CUDA baseline. Sessions remain concurrently live,
 * but each forward call executes separately: this is NOT fused decode batching.
 * All requests arrive at time zero. Compare every token to isolated execution
 * with identical chunk boundaries before reporting queue-inclusive timings. */
#include "model.h"
#include "model_internal.h"
#include "session.h"
#include "session_internal.h"
#include "generate.h"
#include "transformer.h"
#include "platform.h"
#include "backend_layout.h"
#include "../src/transformer/bf16_internal.h"
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
typedef struct {
    BnSession *session;
    int *prompt, *reference, *tokens;
    double *delivery;
    int prefilled, produced, n_prompt;
    double admitted_ms;
} Request;

static BnSession *prepare(BnModel *model, BnGPUBackend *gpu, int chunk, int capacity) {
    BnSession *s=bn_session_create_backend(model,NULL);
    if(!s)return NULL;
    if(bn_transformer_bf16_request_init_with_arithmetic(model,gpu,bn_session_backend(s),
        chunk,capacity,NULL,0,BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32)) {
        bn_session_free(s,NULL);return NULL;
    }
    return s;
}

/* One bounded prefill chunk or one decode token. */
static int advance(BnModel *model, Request *r, int np, int chunk, int output) {
    if(!r->session||r->produced>=output)return -1;
    float *logits=NULL;
    if(r->prefilled<np) {
        int count=np-r->prefilled;if(count>chunk)count=chunk;
        if(r->prefilled+count<np) {
            if(bn_transformer_prefill_no_logits(model,r->session,r->prompt+r->prefilled,count,r->prefilled))return -1;
        } else {
            logits=bn_transformer_prefill(model,r->session,r->prompt+r->prefilled,count,r->prefilled);
            if(!logits)return -1;
        }
        r->prefilled+=count;
    } else {
        logits=bn_transformer_forward(model,r->session,r->tokens[r->produced-1],np+r->produced-1);
        if(!logits)return -1;
    }
    if(logits) {
        int token=greedy(logits,model->config.vocab_size);if(token<0)return -1;
        r->tokens[r->produced]=token;r->delivery[r->produced++]=bn_platform_time_ms();
    }
    return 0;
}
#endif

int main(int argc,char **argv) {
#ifndef BN_ENABLE_CUDA
    (void)argc;(void)argv;fprintf(stderr,"requires BN_ENABLE_CUDA=1\n");return 2;
#else
    int outputs=0,count=0,limit=0,chunk=0,capacity=0,np=0,status=1,mixed=0;
    BnGGUFFile *file=NULL;BnGPUBackend *gpu=NULL;BnModel model={0};
    Request *requests=NULL;int *prompt=NULL;
    if((argc!=8&&argc!=9)||number(argv[3],2,&outputs)||number(argv[4],1,&count)||
       number(argv[5],1,&limit)||number(argv[6],1,&chunk)||number(argv[7],1,&capacity)||
       outputs>256||count>64||limit>16||limit>count||chunk>capacity) {
        fprintf(stderr,"usage: %s MODEL.gguf PROMPT.i32 OUTPUTS REQUESTS MAX_LIVE PREFILL_CHUNK MAXSEQ [uniform|mixed]\n",argv[0]);return 2;
    }
    if(argc==9) {
        if(!strcmp(argv[8],"mixed"))mixed=1;
        else if(strcmp(argv[8],"uniform"))return 2;
    }
    prompt=read_prompt(argv[2],&np);
    if(!prompt||np<2||np>capacity||outputs>capacity-np)goto done;
    file=bn_gguf_open_file(argv[1]);if(!file)goto done;
    if(bn_model_load_with_cpu_preparation(&model,file,capacity,0,0,0))goto done;
    for(int i=0;i<np;i++)if(prompt[i]>=model.config.vocab_size)goto done;
    gpu=bn_gpu_cuda_create();if(!gpu)goto done;
    if(bn_backend_layout_prepare_bf16_with_math(bn_model_backend(&model),gpu,&model.config,
        &model.weights,chunk,BN_BACKEND_BF16_MATH_MKL_VML_HA))goto done;
    gpu->prepare_cpu_operations=reject_cpu;
    requests=calloc((size_t)count,sizeof(*requests));if(!requests)goto done;
    /* Distinct controlled prompts: swap valid IDs in each request. */
    for(int i=0;i<count;i++) {
        Request *r=&requests[i];
        r->prompt=malloc((size_t)np*sizeof(int));r->reference=malloc((size_t)outputs*sizeof(int));
        r->tokens=malloc((size_t)outputs*sizeof(int));r->delivery=malloc((size_t)outputs*sizeof(double));
        if(!r->prompt||!r->reference||!r->tokens||!r->delivery)goto done;
        memcpy(r->prompt,prompt,(size_t)np*sizeof(int));
        r->n_prompt=mixed&&(i&1)&&np>5?5:np;
        int j=1+i%(r->n_prompt-1),temp=r->prompt[1];r->prompt[1]=r->prompt[j];r->prompt[j]=temp;
        r->session=prepare(&model,gpu,chunk,capacity);if(!r->session)goto done;
        while(r->produced<outputs)if(advance(&model,r,r->n_prompt,chunk,outputs))goto done;
        memcpy(r->reference,r->tokens,(size_t)outputs*sizeof(int));
        bn_session_free(r->session,NULL);r->session=NULL;r->prefilled=0;r->produced=0;
    }
    int admitted=0,live=0,completed=0,peak=0,cursor=0;
    size_t decode_calls=0,prefill_calls=0;
    double start=bn_platform_time_ms();
    while(completed<count) {
        while(admitted<count&&live<limit) {
            Request *r=&requests[admitted++];
            r->session=prepare(&model,gpu,chunk,capacity);if(!r->session)goto done;
            r->admitted_ms=bn_platform_time_ms()-start;live++;if(live>peak)peak=live;
        }
        /* One decode per ready session, then one round-robin prefill chunk.
         * This prevents a queued long prompt from indefinitely starving. */
        for(int i=0;i<admitted;i++) {
            Request *r=&requests[i];
            if(!r->session||r->prefilled<r->n_prompt)continue;
            if(advance(&model,r,r->n_prompt,chunk,outputs))goto done;
            decode_calls++;
            if(r->produced==outputs) {
                if(bn_transformer_bf16_position(bn_session_backend(r->session))!=r->n_prompt+outputs-1)goto done;
                bn_session_free(r->session,NULL);r->session=NULL;live--;completed++;
            }
        }
        for(int n=0;n<admitted;n++) {
            int i=(cursor+n)%admitted;Request *r=&requests[i];
            if(!r->session||r->prefilled==r->n_prompt)continue;
            if(advance(&model,r,r->n_prompt,chunk,outputs))goto done;
            prefill_calls++;cursor=(i+1)%admitted;break;
        }
    }
    double elapsed=bn_platform_time_ms()-start;
    for(int i=0;i<count;i++)if(memcmp(requests[i].tokens,requests[i].reference,(size_t)outputs*sizeof(int))) {
        fprintf(stderr,"request %d differs from isolated execution\n",i);goto done;
    }
    printf("{\"kind\":\"config\",\"requests\":%d,\"max_live\":%d,\"peak_live\":%d,\"prompt_tokens\":%d,\"output_tokens\":%d,\"prefill_chunk\":%d,\"maxseq\":%d,\"reserved_context_slots_peak\":%zu,\"mixed_lengths\":%s,\"serialized_execution\":true,\"all_arrive_at_zero\":true,\"matches_isolated\":true,\"arithmetic\":\"cp-fp32\",\"math\":\"MKL_VML_HA\"}\n",
        count,limit,peak,np,outputs,chunk,capacity,(size_t)peak*capacity,mixed?"true":"false");
    for(int i=0;i<count;i++) {
        Request *r=&requests[i];
        printf("{\"kind\":\"request\",\"id\":%d,\"prompt_tokens\":%d,\"admitted_ms\":%.6f,\"ttft_ms\":%.6f,\"completed_ms\":%.6f,\"tokens\":[",i,r->n_prompt,r->admitted_ms,r->delivery[0]-start,r->delivery[outputs-1]-start);
        for(int j=0;j<outputs;j++)printf("%s%d",j?",":"",r->tokens[j]);
        printf("],\"delivery_ms\":[");
        for(int j=0;j<outputs;j++)printf("%s%.6f",j?",":"",r->delivery[j]-start);
        puts("]}");
    }
    printf("{\"kind\":\"summary\",\"elapsed_ms\":%.6f,\"output_tokens_per_second\":%.6f,\"prefill_calls\":%zu,\"decode_calls\":%zu}\n",elapsed,(double)count*outputs*1000/elapsed,prefill_calls,decode_calls);
    status=0;
done:
    if(status)fprintf(stderr,"multi-request benchmark failed\n");
    if(requests)for(int i=0;i<count;i++) {
        bn_session_free(requests[i].session,NULL);free(requests[i].prompt);free(requests[i].reference);
        free(requests[i].tokens);free(requests[i].delivery);
    }
    free(requests);free(prompt);bn_model_free(&model);
    if(gpu)bn_gpu_cuda_destroy(gpu);
    if(file)bn_gguf_free(file);
    return status;
#endif
}
