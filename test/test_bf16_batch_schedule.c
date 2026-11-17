/* CPU-only scheduling checks; numerical kernels have independent CUDA tests. */
#include "../bench/bf16_batch_schedule.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static int calls,seen_count,fail_submit,invalid_logits;
static double ticks;
static BnTransformerBF16PrefillRow seen[16];
BnBackendSession *bn_session_backend(const BnSession *session) {
    return (BnBackendSession *)(uintptr_t)session;
}
double bn_platform_time_ms(void) {return ++ticks;}
int bn_transformer_bf16_batch_prefill(const BnModel *model,BnGPUBackend *gpu,
    BnBackendSession *coordinator,const BnTransformerBF16PrefillRow *rows,int count,
    BnTransformerBF16Observe observe,void *context) {
    assert(model&&gpu&&coordinator&&!observe&&!context&&count>0&&count<=16);
    calls++;seen_count=count;memcpy(seen,rows,(size_t)count*sizeof(*rows));
    if(fail_submit)return -1;
    for(int i=0;i<count;i++)if(rows[i].logits) {
        assert(rows[i].logits_count==3);
        for(int j=0;j<3;j++)rows[i].logits[j]=invalid_logits&&i==count-1?NAN:(float)j;
    }
    return 0;
}

static BnTransformerBF16StepRow mixed_seen[16];
int bn_transformer_bf16_batch_step(const BnModel *model,BnGPUBackend *gpu,
    BnBackendSession *coordinator,const BnTransformerBF16StepRow *rows,int count,
    BnTransformerBF16Observe observe,void *context) {
    BnTransformerBF16PrefillRow plain[16];
    memcpy(mixed_seen,rows,(size_t)count*sizeof(*rows));
    for(int i=0;i<count;i++)plain[i]=rows[i].row;
    return bn_transformer_bf16_batch_prefill(model,gpu,coordinator,plain,count,observe,context);
}
static void test_mixed(void) {
    BnModel model={0};model.config.vocab_size=3;BnGPUBackend gpu={0};
    BnSession sessions[4]={0};float logits[4][3];int prompts[4][5]={{0}},tokens[4][64]={{0}};
    double delivery[4][64]={{0}};BnBenchBatchRequest requests[4];
    BnBackendSession *co=(BnBackendSession *)(uintptr_t)&gpu;
    for(int i=0;i<4;i++) {
        sessions[i].state.logits=logits[i];
        requests[i]=(BnBenchBatchRequest){&sessions[i],prompts[i],NULL,tokens[i],delivery[i],0,0,i%2?2:5,0};
    }
    BnBenchMixedStep trace={0};
    size_t prompt=0,decode=0;int has=0;invalid_logits=fail_submit=0;
    assert(bn_bench_mixed_cohort(&model,&gpu,co,requests,4,4,2,64,&prompt,&decode,&has,&trace)==1);
    assert(has&&prompt==8&&!decode&&seen_count==4);
    assert(mixed_seen[0].row.session==bn_session_backend(&sessions[0])); /* Equal widths retain request order. */
    for(int i=0;i<4;i++)assert(mixed_seen[i].kind==BN_TRANSFORMER_BF16_STEP_PREFILL);
    assert(requests[1].produced==1&&requests[3].produced==1&&!requests[0].produced);
    assert(bn_bench_mixed_cohort(&model,&gpu,co,requests,4,4,2,64,&prompt,&decode,&has,&trace)==1);
    assert(has&&prompt==12&&decode==2);
    assert(trace.count==4&&trace.has_prefill&&trace.request[0]==1&&trace.position[0]==2&&trace.tokens[0]==1&&trace.role[0]==1);
    assert(mixed_seen[0].row.session==bn_session_backend(&sessions[1])&&mixed_seen[0].kind==BN_TRANSFORMER_BF16_STEP_DECODE);
    assert(mixed_seen[1].row.session==bn_session_backend(&sessions[3])&&mixed_seen[1].kind==BN_TRANSFORMER_BF16_STEP_DECODE);
    assert(mixed_seen[2].kind==BN_TRANSFORMER_BF16_STEP_PREFILL&&mixed_seen[3].kind==BN_TRANSFORMER_BF16_STEP_PREFILL);
    assert(bn_bench_mixed_cohort(&model,&gpu,co,requests,4,4,2,64,&prompt,&decode,&has,&trace)==1);
    assert(has&&prompt==14&&decode==4);
    for(int i=0;i<4;i++)assert(mixed_seen[i].kind==BN_TRANSFORMER_BF16_STEP_PREFILL); /* One-token tail. */
    assert(bn_bench_mixed_cohort(&model,&gpu,co,requests,4,4,2,64,&prompt,&decode,&has,&trace)==1&&!has);
    for(int i=0;i<4;i++)assert(mixed_seen[i].kind==BN_TRANSFORMER_BF16_STEP_DECODE);
    BnBenchBatchRequest saved[4];memcpy(saved,requests,sizeof(saved));size_t p=prompt,d=decode;
    fail_submit=1;
    assert(bn_bench_mixed_cohort(&model,&gpu,co,requests,4,4,2,64,&prompt,&decode,&has,&trace)==-1);
    fail_submit=0;invalid_logits=1;
    assert(bn_bench_mixed_cohort(&model,&gpu,co,requests,4,4,2,64,&prompt,&decode,&has,&trace)==-1);
    assert(!memcmp(saved,requests,sizeof(saved))&&p==prompt&&d==decode);
    invalid_logits=0;requests[3].prefilled=-1;int before=calls;
    assert(bn_bench_mixed_cohort(&model,&gpu,co,requests,4,4,2,64,&prompt,&decode,&has,&trace)==-1&&calls==before);
}

int main(void) {
    BnModel model={0};model.config.vocab_size=3;BnGPUBackend gpu={0};
    BnSession sessions[4]={0};float logits[4][3];
    int prompts[4][5],tokens[4];double delivery[4];BnBenchBatchRequest requests[4];
    BnBackendSession *coordinator=(BnBackendSession *)(uintptr_t)&gpu;
    for(int i=0;i<4;i++) {
        for(int j=0;j<5;j++)prompts[i][j]=j%3;
        sessions[i].state.logits=logits[i];tokens[i]=-123;delivery[i]=-123;
        requests[i]=(BnBenchBatchRequest){&sessions[i],prompts[i],NULL,&tokens[i],&delivery[i],0,0,5,0};
    }
    int cursor=0;size_t total=0;requests[1].n_prompt=2;
    assert(bn_bench_prefill_cohort(&model,&gpu,coordinator,requests,2,2,3,&cursor,&total)==1);
    assert(calls==1&&seen_count==2&&seen[0].n_tokens==3&&seen[0].position==0);
    assert(!seen[0].logits&&!seen[0].logits_count);
    assert(seen[1].n_tokens==2&&seen[1].logits==logits[1]&&seen[1].logits_count==3);
    assert(requests[0].prefilled==3&&!requests[0].produced&&tokens[0]==-123);
    assert(requests[1].prefilled==2&&requests[1].produced==1&&tokens[1]==2&&total==5);
    assert(bn_bench_prefill_cohort(&model,&gpu,coordinator,requests,2,2,3,&cursor,&total)==1);
    assert(calls==2&&seen_count==1&&seen[0].tokens==prompts[0]+3&&seen[0].position==3&&seen[0].n_tokens==2);
    assert(requests[0].prefilled==5&&requests[0].produced==1&&tokens[0]==2&&total==7&&cursor==1);
    assert(bn_bench_prefill_cohort(&model,&gpu,coordinator,requests,2,2,3,&cursor,&total)==0&&calls==2&&total==7);

    /* Circular selection remains bounded and cannot skip a pending request. */
    for(int i=0;i<4;i++){requests[i].n_prompt=5;requests[i].prefilled=requests[i].produced=0;tokens[i]=-123;delivery[i]=-123;}
    cursor=0;total=0;
    for(int turn=0;turn<10;turn++) {
        assert(bn_bench_prefill_cohort(&model,&gpu,coordinator,requests,4,2,1,&cursor,&total)==1);
        int first=turn%2*2;assert(seen_count==2);
        assert(seen[0].session==bn_session_backend(&sessions[first]));
        assert(seen[1].session==bn_session_backend(&sessions[first+1]));
        assert(seen[0].position==turn/2&&seen[0].n_tokens==1);
    }
    assert(total==20);
    for(int i=0;i<4;i++)assert(requests[i].prefilled==5&&requests[i].produced==1&&tokens[i]==2);

    /* Reject a malformed later selected row before any submission. */
    for(int i=0;i<2;i++){requests[i].n_prompt=2;requests[i].prefilled=requests[i].produced=0;tokens[i]=-123;delivery[i]=-123;}
    cursor=0;total=0;int before=calls;requests[1].prefilled=-1;
    assert(bn_bench_prefill_cohort(&model,&gpu,coordinator,requests,2,2,3,&cursor,&total)==-1&&calls==before);
    requests[1].prefilled=0;
    assert(bn_bench_prefill_cohort(&model,&gpu,coordinator,requests,2,17,3,&cursor,&total)==-1&&calls==before);
    assert(bn_bench_prefill_cohort(&model,&gpu,coordinator,requests,2,2,4097,&cursor,&total)==-1&&calls==before);
    fail_submit=1;
    assert(bn_bench_prefill_cohort(&model,&gpu,coordinator,requests,2,2,3,&cursor,&total)==-1&&calls==before+1);
    fail_submit=0;invalid_logits=1;
    assert(bn_bench_prefill_cohort(&model,&gpu,coordinator,requests,2,2,3,&cursor,&total)==-1);
    assert(cursor==0&&total==0);
    for(int i=0;i<2;i++)assert(!requests[i].prefilled&&!requests[i].produced&&tokens[i]==-123&&delivery[i]==-123);
    test_mixed();
    puts("BF16 packed cohort schedule PASSED");return 0;
}
