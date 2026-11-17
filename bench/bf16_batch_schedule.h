#ifndef BN_BENCH_BF16_BATCH_SCHEDULE_H
#define BN_BENCH_BF16_BATCH_SCHEDULE_H

#include "model_internal.h"
#include "session_internal.h"
#include "platform.h"
#include "../src/transformer/bf16_internal.h"
#include <math.h>

typedef struct {
    BnSession *session;
    int *prompt, *reference, *tokens;
    double *delivery;
    int prefilled, produced, n_prompt;
    double admitted_ms;
} BnBenchBatchRequest;

static inline int bn_bench_batch_greedy(const float *logits,int n) {
    if(!logits)return -1;
    int id=-1;float best=-INFINITY;
    for(int i=0;i<n;i++)if(logits[i]>best){best=logits[i];id=i;}
    return isfinite(best)?id:-1;
}

/* One bounded cohort in circular request order. Return 1 for a submitted
 * cohort, 0 when none is pending, or -1 on rejection/execution failure.
 * Publish host progress only after successful execution and valid sampling.
 * On error the benchmark discards the cohort; this does not roll back GPU state. */
static inline int bn_bench_prefill_cohort(const BnModel *model,BnGPUBackend *gpu,
    BnBackendSession *coordinator,BnBenchBatchRequest *requests,int admitted,
    int limit,int chunk,int *cursor,size_t *token_count) {
    if(!model||!gpu||!coordinator||!requests||!cursor||!token_count||admitted<1||admitted>64||
       limit<1||limit>16||chunk<1||chunk>8192/limit||*cursor<0||*cursor>=admitted||model->config.vocab_size<1)return -1;
    BnTransformerBF16PrefillRow pending[16];BnBenchBatchRequest *members[16];
    int selected[16]={0},count=0,last=*cursor;
    for(int n=0;n<admitted&&count<limit;n++) {
        int i=(*cursor+n)%admitted;BnBenchBatchRequest *r=&requests[i];
        if(!r->session)continue;
        if(r->n_prompt<1||r->prefilled<0||r->prefilled>r->n_prompt)return -1;
        if(r->prefilled==r->n_prompt)continue;
        int nt=r->n_prompt-r->prefilled;if(nt>chunk)nt=chunk;
        int final=r->prefilled+nt==r->n_prompt;
        if(!r->prompt||r->produced||(final&&(!r->session->state.logits||!r->tokens||!r->delivery)))return -1;
        pending[count]=(BnTransformerBF16PrefillRow){bn_session_backend(r->session),r->prompt+r->prefilled,
            nt,r->prefilled,final?r->session->state.logits:NULL,final?(size_t)model->config.vocab_size:0};
        members[count++]=r;last=i;
    }
    if(!count)return 0;
    if(bn_transformer_bf16_batch_prefill(model,gpu,coordinator,pending,count,NULL,NULL))return -1;
    for(int i=0;i<count;i++)if(pending[i].logits) {
        selected[i]=bn_bench_batch_greedy(pending[i].logits,model->config.vocab_size);
        if(selected[i]<0)return -1;
    }
    for(int i=0;i<count;i++) {
        BnBenchBatchRequest *r=members[i];r->prefilled+=pending[i].n_tokens;
        *token_count+=(size_t)pending[i].n_tokens;
        if(pending[i].logits) {
            r->tokens[0]=selected[i];r->delivery[0]=bn_platform_time_ms();r->produced=1;
        }
    }
    *cursor=(last+1)%admitted;return 1;
}
typedef struct {
    int count,request[16],position[16],tokens[16],role[16],has_prefill;
    double duration_ms;
} BnBenchMixedStep;

/* Mixed geometry follows the original scheduler: ascending query lengths,
 * stable request order, then a contiguous decode prefix and prefill tail.
 * Logical prompt progress remains independent of the recurrent operator role. */
static inline int bn_bench_mixed_cohort(const BnModel *model,BnGPUBackend *gpu,
    BnBackendSession *coordinator,BnBenchBatchRequest *requests,int admitted,
    int limit,int chunk,int outputs,size_t *prompt_tokens,size_t *decode_tokens,
    int *has_prefill,BnBenchMixedStep *trace) {
    if(!model||!gpu||!coordinator||!requests||!prompt_tokens||!decode_tokens||!has_prefill||
       admitted<1||admitted>64||limit<1||limit>16||chunk<1||chunk>8192/limit||
       outputs<2||model->config.vocab_size<1)return -1;
    BnTransformerBF16StepRow rows[16];BnBenchBatchRequest *members[16];
    int count=0,selected[16],prompt[16],sample[16];
    for(int i=0;i<admitted;i++) {
        BnBenchBatchRequest *r=&requests[i];if(!r->session)continue;
        if(count==limit||!r->prompt||!r->tokens||!r->delivery||!r->session->state.logits||
           r->n_prompt<1||r->prefilled<0||r->prefilled>r->n_prompt||r->produced<0||r->produced>=outputs||
           (r->prefilled<r->n_prompt?r->produced!=0:r->produced<1))return -1;
        int pending=r->prefilled<r->n_prompt,nt=pending?r->n_prompt-r->prefilled:1;
        if(nt>chunk)nt=chunk;
        BnTransformerBF16StepRow row={{bn_session_backend(r->session),pending?r->prompt+r->prefilled:r->tokens+r->produced-1,
            nt,pending?r->prefilled:r->n_prompt+r->produced-1,r->session->state.logits,(size_t)model->config.vocab_size},BN_TRANSFORMER_BF16_STEP_PREFILL};
        int j=count;while(j>0&&rows[j-1].row.n_tokens>nt){rows[j]=rows[j-1];members[j]=members[j-1];j--;}
        rows[j]=row;members[j]=r;count++;
    }
    if(!count)return 0;
    int tail=0,any=0;
    for(int i=0;i<count;i++) {
        BnBenchBatchRequest *r=members[i];prompt[i]=r->prefilled<r->n_prompt;
        if(prompt[i]){tail=1;any=1;}
        rows[i].kind=tail?BN_TRANSFORMER_BF16_STEP_PREFILL:BN_TRANSFORMER_BF16_STEP_DECODE;
        sample[i]=!prompt[i]||r->prefilled+rows[i].row.n_tokens==r->n_prompt;
    }
    if(bn_transformer_bf16_batch_step(model,gpu,coordinator,rows,count,NULL,NULL))return -1;
    for(int i=0;i<count;i++)if(sample[i]) {
        selected[i]=bn_bench_batch_greedy(rows[i].row.logits,model->config.vocab_size);
        if(selected[i]<0)return -1;
    }
    for(int i=0;i<count;i++) {
        BnBenchBatchRequest *r=members[i];
        if(prompt[i]){r->prefilled+=rows[i].row.n_tokens;*prompt_tokens+=(size_t)rows[i].row.n_tokens;}
        else (*decode_tokens)++;
        if(sample[i]){r->tokens[r->produced]=selected[i];r->delivery[r->produced++]=bn_platform_time_ms();}
    }
    if(trace) {
        trace->count=count;trace->has_prefill=any;
        for(int i=0;i<count;i++) {
            trace->request[i]=(int)(members[i]-requests);trace->position[i]=rows[i].row.position;
            trace->tokens[i]=rows[i].row.n_tokens;trace->role[i]=(int)rows[i].kind;
        }
    }
    *has_prefill=any;return 1;
}
#endif
