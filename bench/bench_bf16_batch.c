/* Whole-model batched serving benchmark. Default prefill uses one round-robin
 * chunk at a time; --packed-reference batches pending chunks and checks every
 * greedy stream against externally supplied request-major int32 tokens.
 * Default streams are checked against isolated execution before reporting.
 * This strict diagnostic may fail for larger shapes: independent GGUF GEMM
 * arithmetic itself changes with batch size. Full-batch reference qualification
 * is required before interpreting such cases as serving performance. */
#include "model.h"
#include "model_internal.h"
#include "session.h"
#include "session_internal.h"
#include "generate.h"
#include "transformer.h"
#include "platform.h"
#include "backend_layout.h"
#include "bf16_policy_io.h"
#include "bf16_batch_schedule.h"
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
static int greedy(const float *logits,int n) {return bn_bench_batch_greedy(logits,n);}
static int reject_cpu(void *ctx) {
    (void)ctx;fprintf(stderr,"native benchmark attempted CPU fallback\n");return -1;
}
static int *read_prompt(const char *path,int *count,int maximum) {
    FILE *f=fopen(path,"rb");if(!f)return NULL;
    int *ids=NULL;
    if(fseek(f,0,SEEK_END))goto done;
    long bytes=ftell(f);
    if(bytes<=0||bytes%4||bytes/4>maximum||fseek(f,0,SEEK_SET))goto done;
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
/* Untimed diagnostic mode only: preserve every already-published BF16 word.
 * Explicit little-endian output is independent of host byte order. */
static int dump_logits(const char *directory,int request,int step,const float *logits,
    int vocab,unsigned char *buffer) {
    if(!directory||!logits||vocab<1||!buffer)return -1;
    for(int i=0;i<vocab;i++) {
        uint32_t word;memcpy(&word,logits+i,sizeof(word));
        if(word&65535u)return -1;
        buffer[2*(size_t)i]=(unsigned char)(word>>16);
        buffer[2*(size_t)i+1]=(unsigned char)(word>>24);
    }
    char path[4096];int n=snprintf(path,sizeof(path),"%s/request-%d-logits-%d.bin",directory,request,step);
    if(n<1||(size_t)n>=sizeof(path))return -1;
    FILE *f=fopen(path,"wb");if(!f)return -1;
    int failed=fwrite(buffer,2,(size_t)vocab,f)!=(size_t)vocab;
    if(fclose(f))failed=1;
    return failed?-1:0;
}
typedef BnBenchBatchRequest Request;

static BnSession *prepare(BnModel *model, BnGPUBackend *gpu, int chunk, int capacity, const BnTransformerBF16Policy *policy) {
    BnSession *s=bn_session_create_backend(model,NULL);
    if(!s)return NULL;
    if(bn_transformer_bf16_request_init_with_policy(model,gpu,bn_session_backend(s),
        chunk,capacity,NULL,0,policy)) {
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
static int retire(BnTransformerBF16Pool *pool,Request *request) {
    if(pool) {
        if(bn_transformer_bf16_pool_release(pool,request->session))return -1;
    } else bn_session_free(request->session,NULL);
    request->session=NULL;return 0;
}
static int injected_ffn_failure(void *ctx,void *request,const BnGPUFFNResidentPlan *plan) {
    (void)ctx;(void)request;(void)plan;return -1;
}
/* Fail after the first grouped SSM write. No output may be published and all
 * participants/coordinator must become unusable until reset. Outside timing. */
static int batch_failure_selftest(BnModel *m,BnGPUBackend *gpu,BnBackendSession *coordinator,
    Request *first,int chunk,int capacity,int limit,const BnTransformerBF16Policy *policy) {
    int n=limit>1?2:1,result=-1,tokens[2][2]={{0}};double delivery[2][2];Request r[2]={{0}};
    BnTransformerBF16DecodeRow rows[2];size_t bytes=(size_t)m->config.vocab_size*sizeof(float);
    float *snapshot=malloc((size_t)n*bytes);if(!snapshot)return -1;
    for(int i=0;i<n;i++) {
        r[i].prompt=first->prompt;r[i].n_prompt=first->n_prompt;r[i].tokens=tokens[i];r[i].delivery=delivery[i];
        r[i].session=prepare(m,gpu,chunk,capacity,policy);if(!r[i].session)goto done;
        while(!r[i].produced)if(advance(m,&r[i],r[i].n_prompt,chunk,2))goto done;
        rows[i]=(BnTransformerBF16DecodeRow){bn_session_backend(r[i].session),tokens[i][0],r[i].n_prompt,r[i].session->state.logits,(size_t)m->config.vocab_size};
        memcpy((char*)snapshot+(size_t)i*bytes,rows[i].logits,bytes);
    }
    BnTransformerBF16BatchInfo before,after;if(bn_transformer_bf16_batch_info(coordinator,&before))goto done;
    int (*original)(void*,void*,const BnGPUFFNResidentPlan*)=gpu->dense_ffn_bf16_resident;
    gpu->dense_ffn_bf16_resident=injected_ffn_failure;
    int st=bn_transformer_bf16_batch_decode(m,gpu,coordinator,rows,n);
    gpu->dense_ffn_bf16_resident=original;
    if(st!=-1||bn_transformer_bf16_batch_info(coordinator,&after)||!after.failed||after.decode_calls!=before.decode_calls)goto done;
    for(int i=0;i<n;i++)if(bn_transformer_bf16_position(rows[i].session)!=-1||
        memcmp((char*)snapshot+(size_t)i*bytes,rows[i].logits,bytes)||
        bn_transformer_forward(m,r[i].session,rows[i].token,rows[i].position))goto done;
    if(bn_transformer_bf16_batch_decode(m,gpu,coordinator,rows,n)!=-1)goto done;
    result=0;
done:
    for(int i=0;i<n;i++)bn_session_free(r[i].session,NULL);
    free(snapshot);return result;
}

/* Full-logit batch-of-one oracle plus rejection atomicity, outside timing. */
static int batch_selftest(BnModel *m,BnGPUBackend *gpu,BnBackendSession *coordinator,
    Request *first,int chunk,int capacity,const float *expected,const BnTransformerBF16Policy *policy) {
    Request r={0};r.prompt=first->prompt;r.n_prompt=first->n_prompt;
    int tokens[2]={0};double delivery[2];r.tokens=tokens;r.delivery=delivery;
    r.session=prepare(m,gpu,chunk,capacity,policy);if(!r.session)return -1;
    int result=-1;float *snapshot=NULL;size_t bytes=(size_t)m->config.vocab_size*sizeof(float);
    while(!r.produced)if(advance(m,&r,r.n_prompt,chunk,2))goto done;
    if(tokens[0]!=first->reference[0])goto done;
    snapshot=malloc(bytes);if(!snapshot)goto done;memcpy(snapshot,r.session->state.logits,bytes);
    BnTransformerBF16DecodeRow row={bn_session_backend(r.session),tokens[0],r.n_prompt,r.session->state.logits,(size_t)m->config.vocab_size};
    BnTransformerBF16BatchInfo before,after;if(bn_transformer_bf16_batch_info(coordinator,&before))goto done;
    for(int reject=0;reject<9;reject++) {
        BnTransformerBF16DecodeRow bad[2]={row,row};int n=1;
        switch(reject) {
            case 0:bad[0].token=-1;break;case 1:bad[0].position++;break;
            case 2:bad[0].logits_count--;break;case 3:bad[0].logits=NULL;break;
            case 4:bad[0].session=coordinator;break;case 5:bad[0].session=NULL;break;
            case 6:n=2;break;case 7:n=2;bad[1].session=NULL;break;
            default:n=0;break;
        }
        if(bn_transformer_bf16_batch_decode(m,gpu,coordinator,bad,n)!=-1||
           bn_transformer_bf16_position(row.session)!=row.position||memcmp(snapshot,row.logits,bytes)||
           bn_transformer_bf16_batch_info(coordinator,&after)||after.failed||
           before.decode_calls!=after.decode_calls||before.busy_fences!=after.busy_fences)goto done;
    }
    if(bn_transformer_bf16_batch_decode(m,gpu,coordinator,&row,1)||
       bn_transformer_bf16_position(row.session)!=row.position+1||memcmp(expected,row.logits,bytes)) {
        fprintf(stderr,"batch-of-one full logits differ from isolated decode\n");goto done;
    }
    result=0;
done:
    free(snapshot);bn_session_free(r.session,NULL);return result;
}
#endif

int main(int argc,char **argv) {
#ifndef BN_ENABLE_CUDA
    (void)argc;(void)argv;fprintf(stderr,"requires BN_ENABLE_CUDA=1\n");return 2;
#else
    int outputs=0,count=0,limit=0,chunk=0,capacity=0,np=0,status=1,mixed=0,fused=0,pool_mib=0;
    BnBackendSession *coordinator=NULL;BnTransformerBF16Pool pool_storage={0},*pool=NULL;
    BnGGUFFile *file=NULL;BnGPUBackend *gpu=NULL;BnModel model={0};
    Request *requests=NULL;int *prompt=NULL,*reference_tokens=NULL;float *single_logits=NULL;
    const char *packed_reference=NULL,*dump_directory=NULL;int n_reference=0,packed_rows=0;
    BnBenchMixedStep *step_trace=NULL;size_t step_count=0,step_capacity=0;
    unsigned char *dump_buffer=NULL;int dumped[64]={0};
    if((argc<8||argc>18)||number(argv[3],2,&outputs)||number(argv[4],1,&count)||
       number(argv[5],1,&limit)||number(argv[6],1,&chunk)||number(argv[7],1,&capacity)||
       outputs>256||count>64||limit>16||limit>count||chunk>capacity) {
        fprintf(stderr,"usage: %s MODEL.gguf PROMPT.i32 OUTPUTS REQUESTS MAX_LIVE PREFILL_CHUNK MAXSEQ [uniform|mixed] [--policy POLICY.json] [--packed-reference TOKENS.i32] [--dump-logits DIRECTORY] [--fused-steps] [--kv-pool-mib N]\n",argv[0]);return 2;
    }
    BnTransformerBF16Policy policy={0};policy.arithmetic=BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32;
    int option=8,have_policy=0,have_mode=0;
    while(option<argc) {
        const char *arg=argv[option++];
        if(!strcmp(arg,"--policy")) {
            if(have_policy++||option==argc||bench_policy_read(argv[option++],&policy))return 2;
        } else if(!strcmp(arg,"--packed-reference")) {
            if(packed_reference||option==argc)return 2;
            packed_reference=argv[option++];
        } else if(!strcmp(arg,"--dump-logits")) {
            if(dump_directory||option==argc||!argv[option][0])return 2;
            dump_directory=argv[option++];
        } else if(!strcmp(arg,"--kv-pool-mib")) {
            if(pool_mib||option==argc||number(argv[option++],1,&pool_mib)||(size_t)pool_mib>SIZE_MAX/1048576)return 2;
        } else if(!strcmp(arg,"--fused-steps")) {
            if(fused++)return 2;
        } else if(!strcmp(arg,"mixed")||!strcmp(arg,"uniform")) {
            if(have_mode++)return 2;
            mixed=!strcmp(arg,"mixed");
        } else return 2;
    }
    if((dump_directory||fused)&&!packed_reference)return 2;
    if(fused&&capacity>8192)return 2;
    if(packed_reference) {
        if(chunk>8192/limit){fprintf(stderr,"packed prefill budget exceeds 8192 tokens\n");return 2;}
        packed_rows=chunk*limit;
        reference_tokens=read_prompt(packed_reference,&n_reference,count*outputs);
        if(!reference_tokens||n_reference!=count*outputs){fprintf(stderr,"reference must contain REQUESTS*OUTPUTS int32 tokens\n");goto done;}
    }
    prompt=read_prompt(argv[2],&np,capacity);
    if(!prompt||np<2||np>capacity||outputs>capacity-np)goto done;
    file=bn_gguf_open_file(argv[1]);if(!file)goto done;
    if(bn_model_load_with_cpu_preparation(&model,file,capacity,0,0,0))goto done;
    for(int i=0;i<np;i++)if(prompt[i]>=model.config.vocab_size)goto done;
    for(int i=0;i<n_reference;i++)if(reference_tokens[i]>=model.config.vocab_size)goto done;
    if(dump_directory){dump_buffer=malloc((size_t)model.config.vocab_size*2);if(!dump_buffer)goto done;}
    gpu=bn_gpu_cuda_create();if(!gpu)goto done;
    if(bn_backend_layout_prepare_bf16_with_math(bn_model_backend(&model),gpu,&model.config,
        &model.weights,packed_reference?packed_rows:(chunk>limit?chunk:limit),BN_BACKEND_BF16_MATH_MKL_VML_HA))goto done;
    gpu->prepare_cpu_operations=reject_cpu;
    requests=calloc((size_t)count,sizeof(*requests));if(!requests)goto done;
    single_logits=malloc((size_t)model.config.vocab_size*sizeof(float));if(!single_logits)goto done;
    /* Distinct controlled prompts: swap valid IDs in each request. */
    for(int i=0;i<count;i++) {
        Request *r=&requests[i];
        r->prompt=malloc((size_t)np*sizeof(int));r->reference=malloc((size_t)outputs*sizeof(int));
        r->tokens=malloc((size_t)outputs*sizeof(int));r->delivery=malloc((size_t)outputs*sizeof(double));
        if(!r->prompt||!r->reference||!r->tokens||!r->delivery)goto done;
        memcpy(r->prompt,prompt,(size_t)np*sizeof(int));
        r->n_prompt=mixed&&(i&1)&&np>5?5:np;
        int j=1+i%(r->n_prompt-1),temp=r->prompt[1];r->prompt[1]=r->prompt[j];r->prompt[j]=temp;
        r->session=prepare(&model,gpu,chunk,capacity,&policy);if(!r->session)goto done;
        while(r->produced<outputs) {
            if(advance(&model,r,r->n_prompt,chunk,outputs))goto done;
            if(!i&&r->produced==2)memcpy(single_logits,r->session->state.logits,(size_t)model.config.vocab_size*sizeof(float));
        }
        memcpy(r->reference,r->tokens,(size_t)outputs*sizeof(int));
        bn_session_free(r->session,NULL);r->session=NULL;r->prefilled=0;r->produced=0;
    }
    coordinator=bn_backend_session_create();if(!coordinator)goto done;
    if(bn_transformer_bf16_batch_prepare_with_policy(&model,gpu,coordinator,limit,&policy))goto done;
    if(batch_selftest(&model,gpu,coordinator,&requests[0],chunk,capacity,single_logits,&policy))goto done;
    if(batch_failure_selftest(&model,gpu,coordinator,&requests[0],chunk,capacity,limit,&policy))goto done;
    bn_backend_session_free(coordinator);coordinator=bn_backend_session_create();if(!coordinator)goto done;
    if(packed_reference) {
        if(bn_transformer_bf16_batch_prepare_packed_with_policy(&model,gpu,coordinator,limit,packed_rows,&policy))goto done;
        for(int i=0;i<count;i++)memcpy(requests[i].reference,reference_tokens+(size_t)i*outputs,(size_t)outputs*sizeof(int));
    } else if(bn_transformer_bf16_batch_prepare_with_policy(&model,gpu,coordinator,limit,&policy))goto done;
    BnTransformerBF16Policy admitted_policy={0};
    if(bn_transformer_bf16_policy(coordinator,&admitted_policy))goto done;
    fputs("{\"kind\":\"bf16_policy\",\"admitted\":",stdout);
    if(bench_policy_write(stdout,&admitted_policy))goto done;
    puts("}");
    BnTransformerBF16BatchInfo initial_info;if(bn_transformer_bf16_batch_info(coordinator,&initial_info))goto done;
    int admitted=0,live=0,completed=0,peak=0,cursor=0;
    size_t decode_calls=0,prefill_calls=0,prefill_tokens=0;
    double prefill_ms=0,decode_ms=0,mixed_ms=0;
    size_t mixed_phase_calls=0;
    if(fused) {
        step_capacity=(size_t)count*((np+chunk-1)/chunk+outputs);
        step_trace=calloc(step_capacity,sizeof(*step_trace));if(!step_trace)goto done;
    }
    if(pool_mib) {
        if(bn_transformer_bf16_pool_init(&pool_storage,&model,gpu,limit,chunk,capacity,(size_t)pool_mib*1048576,&policy))goto done;
        pool=&pool_storage;
    }
    double start=bn_platform_time_ms();
    while(completed<count) {
        while(admitted<count&&live<limit) {
            Request *r=&requests[admitted];
            if(pool) {
                int acquired=bn_transformer_bf16_pool_acquire(pool,&r->session);
                if(acquired<0)goto done;
                if(acquired==1)break;
            } else {r->session=prepare(&model,gpu,chunk,capacity,&policy);if(!r->session)goto done;}
            admitted++;
            r->admitted_ms=bn_platform_time_ms()-start;live++;if(live>peak)peak=live;
        }
        if(!live)goto done;
        if(fused) {
            double step_start=bn_platform_time_ms();int has_prefill=0;
            if(step_count>=step_capacity)goto done;
            if(bn_bench_mixed_cohort(&model,gpu,coordinator,requests,admitted,limit,chunk,outputs,
                &prefill_tokens,&decode_calls,&has_prefill,&step_trace[step_count])!=1)goto done;
            if(dump_directory)for(int i=0;i<admitted;i++) {
                Request *r=&requests[i];if(!r->session||r->produced==dumped[i])continue;
                if(r->produced!=dumped[i]+1||dump_logits(dump_directory,i,dumped[i],r->session->state.logits,model.config.vocab_size,dump_buffer))goto done;
                dumped[i]++;
            }
            /* The complete cohort has published before any session is freed. */
            for(int i=0;i<admitted;i++) {
                Request *r=&requests[i];if(!r->session||r->produced!=outputs)continue;
                if(bn_transformer_bf16_position(bn_session_backend(r->session))!=r->n_prompt+outputs-1)goto done;
                if(retire(pool,r))goto done;
                live--;completed++;
            }
            double duration=bn_platform_time_ms()-step_start;
            step_trace[step_count++].duration_ms=duration;
            if(has_prefill){mixed_ms+=duration;mixed_phase_calls++;prefill_calls++;}
            else decode_ms+=duration;
            continue;
        }
        /* Decode ready sessions first, then one bounded prefill cohort (or
         * a single legacy chunk), so long prompts cannot starve decode. */
        BnTransformerBF16DecodeRow rows[16];Request *ready[16];int nr=0;
        for(int i=0;i<admitted;i++) {
            Request *r=&requests[i];if(!r->session||r->prefilled<r->n_prompt)continue;
            rows[nr].session=bn_session_backend(r->session);rows[nr].token=r->tokens[r->produced-1];
            rows[nr].position=r->n_prompt+r->produced-1;rows[nr].logits=r->session->state.logits;rows[nr].logits_count=(size_t)model.config.vocab_size;
            ready[nr++]=r;
        }
        double decode_start=bn_platform_time_ms();
        if(nr&&bn_transformer_bf16_batch_decode(&model,gpu,coordinator,rows,nr))goto done;
        for(int i=0;i<nr;i++) {
            Request *r=ready[i];int token=greedy(r->session->state.logits,model.config.vocab_size);if(token<0)goto done;
            if(dump_directory) {
                int id=(int)(r-requests);if(dumped[id]!=r->produced||dump_logits(dump_directory,id,r->produced,r->session->state.logits,model.config.vocab_size,dump_buffer))goto done;
                dumped[id]++;
            }
            r->tokens[r->produced]=token;r->delivery[r->produced++]=bn_platform_time_ms();decode_calls++;
            if(r->produced==outputs) {
                if(bn_transformer_bf16_position(bn_session_backend(r->session))!=r->n_prompt+outputs-1)goto done;
            }
        }
        /* Every row is ready after the grouped call. Publish the entire
         * cohort before reclamation can delay another request's final token. */
        for(int i=0;i<nr;i++) {
            Request *r=ready[i];
            if(r->produced==outputs) {
                if(retire(pool,r))goto done;
                live--;completed++;
            }
        }
        if(nr)decode_ms+=bn_platform_time_ms()-decode_start;
        double prefill_start=bn_platform_time_ms();size_t calls_before=prefill_calls;
        if(packed_reference) {
            int submitted=bn_bench_prefill_cohort(&model,gpu,coordinator,requests,admitted,limit,chunk,&cursor,&prefill_tokens);
            if(submitted<0)goto done;
            if(submitted)prefill_calls++;
            if(submitted&&dump_directory)for(int i=0;i<admitted;i++) {
                Request *r=&requests[i];if(r->produced!=1||dumped[i])continue;
                if(!r->session||dump_logits(dump_directory,i,0,r->session->state.logits,model.config.vocab_size,dump_buffer))goto done;
                dumped[i]=1;
            }
        } else for(int n=0;n<admitted;n++) {
            int i=(cursor+n)%admitted;Request *r=&requests[i];
            if(!r->session||r->prefilled==r->n_prompt)continue;
            int before=r->prefilled;
            if(advance(&model,r,r->n_prompt,chunk,outputs))goto done;
            prefill_tokens+=(size_t)(r->prefilled-before);
            prefill_calls++;cursor=(i+1)%admitted;break;
        }
        if(prefill_calls!=calls_before)prefill_ms+=bn_platform_time_ms()-prefill_start;
    }
    double elapsed=bn_platform_time_ms()-start;
    if(dump_directory)for(int i=0;i<count;i++)if(dumped[i]!=outputs)goto done;
    for(int i=0;i<count;i++)if(memcmp(requests[i].tokens,requests[i].reference,(size_t)outputs*sizeof(int))) {
        int first=0;while(first<outputs&&requests[i].tokens[first]==requests[i].reference[first])first++;
        fprintf(stderr,"request %d differs from %s reference at output %d: got %d expected %d\n",i,packed_reference?"external":"isolated",first,requests[i].tokens[first],requests[i].reference[first]);
        fprintf(stderr,"candidate:");for(int j=0;j<outputs;j++)fprintf(stderr," %d",requests[i].tokens[j]);
        fprintf(stderr,"\nreference:");for(int j=0;j<outputs;j++)fprintf(stderr," %d",requests[i].reference[j]);fprintf(stderr,"\n");goto done;
    }
    printf("{\"kind\":\"config\",\"requests\":%d,\"max_live\":%d,\"peak_live\":%d,\"prompt_tokens\":%d,\"output_tokens\":%d,\"prefill_chunk\":%d,\"maxseq\":%d,\"reserved_context_slots_peak\":%zu,\"mixed_lengths\":%s,\"serialized_execution\":false,\"all_arrive_at_zero\":true,\"matches_isolated\":%s,\"matches_reference\":true,\"reference_kind\":\"%s\",\"kv_pool_enabled\":%s,\"fused_steps\":%s,\"packed_prefill\":%s,\"packed_token_budget\":%d,\"batch_one_logits_exact\":true,\"invalid_cohort_atomic\":true,\"partial_submit_failure_poisoned\":true,\"diagnostic_logits_dump\":%s,\"arithmetic\":%d,\"math\":\"MKL_VML_HA\"}\n",
        count,limit,peak,np,outputs,chunk,capacity,(size_t)peak*capacity,mixed?"true":"false",packed_reference?"null":"true",packed_reference?"external_tokens":"isolated",pool?"true":"false",fused?"true":"false",packed_reference?"true":"false",packed_rows,dump_directory?"true":"false",admitted_policy.arithmetic);
    for(int i=0;i<count;i++) {
        Request *r=&requests[i];
        printf("{\"kind\":\"request\",\"id\":%d,\"prompt_tokens\":%d,\"admitted_ms\":%.6f,\"ttft_ms\":%.6f,\"completed_ms\":%.6f,\"tokens\":[",i,r->n_prompt,r->admitted_ms,r->delivery[0]-start,r->delivery[outputs-1]-start);
        for(int j=0;j<outputs;j++)printf("%s%d",j?",":"",r->tokens[j]);
        printf("],\"delivery_ms\":[");
        for(int j=0;j<outputs;j++)printf("%s%.6f",j?",":"",r->delivery[j]-start);
        puts("]}");
    }
    for(size_t c=0;c<step_count;c++) {
        BnBenchMixedStep *t=&step_trace[c];
        printf("{\"kind\":\"mixed_step\",\"call\":%zu,\"has_prefill\":%s,\"duration_ms\":%.6f,\"rows\":[",c,t->has_prefill?"true":"false",t->duration_ms);
        for(int i=0;i<t->count;i++)printf("%s{\"request\":%d,\"position\":%d,\"tokens\":%d,\"role\":%d}",i?",":"",t->request[i],t->position[i],t->tokens[i],t->role[i]);
        puts("]}");
    }
    printf("{\"kind\":\"summary\",\"elapsed_ms\":%.6f,\"output_tokens_per_second\":%.6f,\"prefill_calls\":%zu,\"decode_calls\":%zu,\"prefill_tokens\":%zu,\"prefill_phase_ms\":%.6f,\"decode_phase_ms\":%.6f,\"mixed_phase_ms\":%.6f,\"mixed_phase_calls\":%zu}\n",elapsed,(double)count*outputs*1000/elapsed,prefill_calls,decode_calls,prefill_tokens,prefill_ms,decode_ms,mixed_ms,mixed_phase_calls);
    BnTransformerBF16BatchInfo info;
    if(bn_transformer_bf16_batch_info(coordinator,&info)||info.failed)goto done;
    printf("{\"kind\":\"batch\",\"decode_batches\":%llu,\"prefill_batches\":%llu,\"mixed_batches\":%llu,\"metadata_busy_fences\":%llu}\n",(unsigned long long)(info.decode_calls-initial_info.decode_calls),(unsigned long long)(info.prefill_calls-initial_info.prefill_calls),(unsigned long long)(info.mixed_calls-initial_info.mixed_calls),(unsigned long long)(info.busy_fences-initial_info.busy_fences));
    if(pool) {
        BnTransformerBF16PoolInfo pi;if(bn_transformer_bf16_pool_info(pool,&pi)||pi.leased_slots||pi.reserved_kv_bytes>pi.kv_budget_bytes)goto done;
        printf("{\"kind\":\"kv_pool\",\"scope\":\"kv_payload_only\",\"budget_bytes\":%zu,\"kv_bytes_per_slot\":%zu,\"reserved_kv_bytes\":%zu,\"peak_kv_bytes\":%zu,\"allocated_slots\":%d,\"leased_slots\":%d,\"created\":%llu,\"reused\":%llu,\"discarded\":%llu}\n",
            pi.kv_budget_bytes,pi.kv_bytes_per_slot,pi.reserved_kv_bytes,pi.peak_kv_bytes,pi.allocated_slots,pi.leased_slots,
            (unsigned long long)pi.created,(unsigned long long)pi.reused,(unsigned long long)pi.discarded);
    }
    status=0;
done:
    if(status)fprintf(stderr,"batched decode benchmark failed\n");
    if(requests)for(int i=0;i<count;i++) {
        if(!pool)bn_session_free(requests[i].session,NULL);
        free(requests[i].prompt);free(requests[i].reference);
        free(requests[i].tokens);free(requests[i].delivery);
    }
    bn_transformer_bf16_pool_free(pool);free(step_trace);free(dump_buffer);free(requests);free(prompt);free(reference_tokens);free(single_logits);bn_backend_session_free(coordinator);bn_model_free(&model);
    if(gpu)bn_gpu_cuda_destroy(gpu);
    if(file)bn_gguf_free(file);
    return status;
#endif
}
