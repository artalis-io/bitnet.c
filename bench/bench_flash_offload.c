/* Flash offload diagnostics: caches persist across restarted requests. Per-phase
 * counters are observed between timed calls; no extra activation readbacks.
 * Source wrapper/read/metadata/retry spans are not isolated GPU DMA stalls.
 * Filesystem-cache state is not controlled. See docs/benchmarks.md. */
#include "model.h"
#include "model_internal.h"
#include "session.h"
#include "session_internal.h"
#include "generate.h"
#include "moe.h"
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
#endif

int main(int argc,char **argv) {
#ifndef BN_ENABLE_CUDA
    (void)argc;(void)argv;fprintf(stderr,"requires BN_ENABLE_CUDA=1\n");return 2;
#else
    int output=0,runs=0,warmups=0,chunk=0,capacity=0,np=0,status=1;
    int budget_mb=0,source_mb=0;
    BnGGUFFile *file=NULL;BnGPUBackend *gpu=NULL;BnModel model={0};BnSession *session=NULL;
    int *prompt=NULL,*ids=NULL;double *steps=NULL;
    if(argc!=10||number(argv[3],2,&output)||number(argv[4],1,&runs)||
       number(argv[5],0,&warmups)||number(argv[6],1,&chunk)||number(argv[7],1,&capacity)||
       number(argv[8],0,&budget_mb)||number(argv[9],0,&source_mb)||
       runs>INT_MAX-warmups||chunk>capacity||output>256) {
        fprintf(stderr,"usage: %s MODEL.gguf PROMPT.i32 OUTPUTS RUNS WARMUPS PREFILL_CHUNK MAXSEQ GPU_CACHE_MIB RAM_CACHE_MIB\n",argv[0]);return 2;
    }
    BnTransformerBF16FlashFFNInfo snapshots[257];
    prompt=read_prompt(argv[2],&np);
    if(!prompt||np>capacity||output>capacity-np)goto done;
    ids=malloc((size_t)output*sizeof(*ids));steps=malloc((size_t)(output-1)*sizeof(*steps));
    if(!ids||!steps)goto done;
    file=bn_gguf_open_file(argv[1]);if(!file)goto done;
    if(bn_model_load_with_cpu_preparation(&model,file,capacity,0,0,0))goto done;
    if(file->n_shards>1) {
        if(bn_model_set_moe_shard_files(&model,file->owned_maps,file->n_shards))goto done;
    } else {
        const BnMappedFile *mapped=bn_gguf_primary_file(file);
        if(!mapped||mapped->fd<0)goto done;
        bn_model_set_moe_fd(&model,mapped->fd);
    }
    bn_model_set_moe_mmap_base(&model,NULL);bn_model_set_moe_mmap_shards(&model,NULL,0);
    size_t max_bytes[3]={0};
    for(int layer=0;layer<model.config.n_layers;layer++) {
        const BnMoEExpertMap *map=&model.weights.layers[layer].moe.expert_map;
        size_t values[]={map->expert_gate_bytes,map->expert_up_bytes,map->expert_down_bytes};
        for(int role=0;role<3;role++)if(values[role]>max_bytes[role])max_bytes[role]=values[role];
    }
    void *cache=source_mb?bn_moe_cache_create((size_t)source_mb*1024*1024,max_bytes[0],max_bytes[1],max_bytes[2]):NULL;
    if(source_mb&&!cache)goto done;
    bn_model_set_moe_cache(&model,cache);
    for(int i=0;i<np;i++)if(prompt[i]>=model.config.vocab_size)goto done;
    gpu=bn_gpu_cuda_create();if(!gpu)goto done;
    if(bn_backend_layout_prepare_bf16_flash_model_with_cache(bn_model_backend(&model),gpu,&model.config,
       &model.weights,(size_t)12*1024*1024*1024,64u*1024*1024,(size_t)budget_mb*1024*1024))goto done;
    session=bn_session_create_backend(&model,NULL);if(!session||!session->backend_only)goto done;
    if(bn_transformer_bf16_flash_prepare_context(&model,gpu,bn_session_backend(session),chunk,capacity,
       (size_t)4*1024*1024*1024,256u*1024*1024))goto done;
    gpu->prepare_cpu_operations=reject_cpu;
    if(bn_transformer_bf16_flash_ffn_info(bn_session_backend(session),&snapshots[0]))goto done;
    printf("{\"kind\":\"config\",\"prompt_tokens\":%d,\"output_tokens\":%d,\"runs\":%d,\"warmups\":%d,\"maxseq\":%d,\"prefill_chunk\":%d,\"gpu_cache_mib\":%d,\"ram_cache_mib\":%d,\"reserved_expert_bytes\":%zu,\"initial_resident_expert_bytes\":%zu,\"backend_only\":true,\"cache_state_retained\":true,\"filesystem_cache_controlled\":false}\n",
        np,output,runs,warmups,capacity,chunk,budget_mb,source_mb,
        snapshots[0].cache.reserved_device_bytes,snapshots[0].cache.resident_bytes);
    for(int run=0;run<warmups+runs;run++) {
        double start=bn_platform_time_ms();
        if(run&&bn_transformer_bf16_session_restart(&model,session))goto done;
        double reset_ms=bn_platform_time_ms()-start;
        if(bn_transformer_bf16_flash_ffn_info(bn_session_backend(session),&snapshots[0]))goto done;
        start=bn_platform_time_ms();
        ids[0]=greedy(bn_prefill(&model,session,prompt,np,0,0),model.config.vocab_size);
        if(ids[0]<0)goto done;
        double first=bn_platform_time_ms();
        if(bn_transformer_bf16_flash_ffn_info(bn_session_backend(session),&snapshots[1]))goto done;
        for(int i=1;i<output;i++) {
            double phase_start=bn_platform_time_ms();
            ids[i]=greedy(bn_transformer_forward(&model,session,ids[i-1],np+i-1),model.config.vocab_size);
            if(ids[i]<0)goto done;
            double now=bn_platform_time_ms();steps[i-1]=now-phase_start;
            if(bn_transformer_bf16_flash_ffn_info(bn_session_backend(session),&snapshots[i+1]))goto done;
        }

        double decode_total=0;
        for(int i=0;i<output-1;i++)decode_total+=steps[i];
        printf("{\"kind\":\"sample\",\"run\":%d,\"warmup\":%s,\"ttft_ms\":%.9f,\"decode_mean_ms\":%.9f,\"decode_total_ms\":%.9f,\"tokens\":[",
            run,run<warmups?"true":"false",reset_ms+first-start,decode_total/(output-1),decode_total);
        for(int i=0;i<output;i++)printf("%s%d",i?",":"",ids[i]);
        printf("],\"decode_steps_ms\":[");
        for(int i=0;i<output-1;i++)printf("%s%.9f",i?",":"",steps[i]);
        puts("]}");fflush(stdout);
        for(int phase=0;phase<output;phase++) {
            const BnTransformerBF16FlashFFNInfo *a=&snapshots[phase],*b=&snapshots[phase+1];
            printf("{\"kind\":\"phase\",\"run\":%d,\"phase\":%d,\"prefill\":%s,\"ms\":%.6f,\"gpu_hits\":%zu,\"gpu_misses\":%zu,\"miss_bytes\":%zu,\"upload_bytes\":%zu,\"fill_copy_bytes\":%zu,\"source_io_bytes\":%zu,\"source_io_ms\":%.6f,\"source_wrapper_ms\":%.6f,\"bank_retry_ms\":%.6f,\"metadata_ms\":%.6f,\"ram_hits\":%zu,\"ram_misses\":%zu}\n",
                run,phase,phase?"false":"true",phase?steps[phase-1]:reset_ms+first-start,
                b->cache.hits-a->cache.hits,
                budget_mb?b->cache.misses-a->cache.misses:b->banks.submitted_experts-a->banks.submitted_experts,
                budget_mb?b->cache.miss_bytes-a->cache.miss_bytes:b->banks.upload_bytes-a->banks.upload_bytes,
                b->banks.upload_bytes-a->banks.upload_bytes,b->cache.fill_copy_bytes-a->cache.fill_copy_bytes,
                b->source.io_bytes-a->source.io_bytes,b->source.io_time_ms-a->source.io_time_ms,
                b->source_ms-a->source_ms,b->bank_retry_ms-a->bank_retry_ms,b->metadata_ms-a->metadata_ms,
                b->source.gpu_host_cache_hits-a->source.gpu_host_cache_hits,b->source.gpu_host_cache_misses-a->source.gpu_host_cache_misses);
        }

    }
    status=0;
done:
    if(status)fprintf(stderr,"Flash offload benchmark failed\n");
    bn_session_free(session,NULL);bn_moe_cache_free(bn_model_moe_cache(&model));bn_model_set_moe_cache(&model,NULL);bn_model_free(&model);
    if(gpu)bn_gpu_cuda_destroy(gpu);
    if(file)bn_gguf_free(file);
    free(prompt);free(ids);free(steps);return status;
#endif
}
