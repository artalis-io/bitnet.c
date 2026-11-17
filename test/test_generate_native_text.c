#include "generate.h"
#include "model.h"
#include "session_internal.h"
#include "transformer_internal.h"
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "native_encode_reference.h"
#include "native_text_control_reference.h"
#ifdef BN_TEST_NATIVE_TEXT_HEADER
#include "../src/generate_native_text.h"
#include "../src/generate_native_schedule.h"
#endif
#ifdef BN_TEST_DENY_HEAP
static _Thread_local int control_hot,control_heap_ops;
extern void *__real_malloc(size_t);extern void *__real_calloc(size_t,size_t);extern void *__real_realloc(void*,size_t);extern void __real_free(void*);
void *__wrap_malloc(size_t n){if(control_hot){control_heap_ops++;return NULL;}return __real_malloc(n);}
void *__wrap_calloc(size_t n,size_t k){if(control_hot){control_heap_ops++;return NULL;}return __real_calloc(n,k);}
void *__wrap_realloc(void *p,size_t n){if(control_hot){control_heap_ops++;return NULL;}return __real_realloc(p,n);}
void __wrap_free(void *p){if(control_hot)control_heap_ops++;__real_free(p);}
#endif
/* These stubs test orchestration, not model arithmetic. Real-weight source
 * comparisons exercise the same public API through bench_native_forward. */
typedef struct Control {
    struct Control *peer;
    BnModel model;BnSession session;BnModelNativeAnatomy anatomy;float logits[315];
    const int *prompt,*sequence;int prompt_count,sequence_count,generated;
    int pending,raw[64],stage_calls,forward_calls,completed,busy_each,busy_left;
    int fail_stage,fail_forward,stale_ticket,capacity,max_batch,admit,tie;
    size_t callbacks,flushes,observations,output_bytes;unsigned char output[128];
    int stop_at,nonfinite,predict,resumable;
    BnGenerateNativeTextRequest *request;
} Control;
static Control *control(const BnModel *m){return m?(Control*)(void*)m->runtime:NULL;}
static Control *control_session(const BnModel *m,const BnSession *s) {
    Control *c=control(m);
    if(c&&s==&c->session)return c;
    return c&&c->peer&&s==&c->peer->session?c->peer:NULL;
}
const BnModelNativeAnatomy *bn_model_native_anatomy(const BnModel *m) {
    Control *c=control(m);return c?&c->anatomy:NULL;
}
int bn_session_native_history_info(const BnSession *s,const BnModel *m,BnSessionNativeHistoryInfo *out) {
    Control *c=control_session(m,s);if(!c||s!=&c->session||!out)return -1;
    *out=(BnSessionNativeHistoryInfo){c->max_batch,c->capacity,s->pos,c->pending,0};return 0;
}
int bn_session_native_history_stage(BnSession *s,const BnModel *m,const int *tokens,int n) {
    Control *c=control_session(m,s);assert(c&&s==&c->session&&!c->pending&&n>0&&n<=c->max_batch&&s->pos<=c->capacity-n);
    c->stage_calls++;if(c->stage_calls==c->fail_stage)return -1;
    if(s->pos<c->prompt_count) {
        assert(n<=c->prompt_count-s->pos&&!memcmp(tokens,c->prompt+s->pos,(size_t)n*sizeof(int)));
    }else assert(n==1&&c->generated<c->sequence_count&&tokens[0]==c->sequence[c->generated]);
    memcpy(c->raw,tokens,(size_t)n*sizeof(int));c->pending=n;c->busy_left=c->busy_each;return 0;
}
int bn_transformer_trellis_forward_run(const BnModel *m,BnSession *s,int admit,BnTransformerTrellisForwardResult *out) {
    Control *c=control_session(m,s);assert(c&&s==&c->session&&c->pending&&out);
    assert(admit==(s->pos<c->prompt_count?0:c->admit));c->forward_calls++;
    if(c->busy_left){c->busy_left--;return BN_GPU_UPLOAD_BUSY;}
    if(c->stage_calls==c->fail_forward)return -1;
    int position=s->pos,n=c->pending;s->pos+=n;c->pending=0;c->completed++;
    if(position>=c->prompt_count)c->generated++;
    for(int i=0;i<315;i++)c->logits[i]=0;
    if(c->generated<c->sequence_count)c->logits[c->sequence[c->generated]]=10;
    if(c->tie){c->logits[0]=10;c->logits[195]=10;}
    if(c->nonfinite)c->logits[1]=NAN;
    *out=(BnTransformerTrellisForwardResult){c,s,1,(uint64_t)c->completed,position,n,1,315,{0},c->logits};return 0;
}
int bn_transformer_trellis_forward_current(const BnModel *m,BnSession *s,const BnTransformerTrellisForwardResult *out) {
    Control *c=control_session(m,s);return c&&s==&c->session&&out&&out->request==c&&out->epoch==(uint64_t)c->completed&&!c->stale_ticket;
}
static void control_init(Control *c,const int *prompt,int count) {
    memset(c,0,sizeof(*c));c->model.runtime=(BnModelRuntime*)(void*)c;c->session.backend_only=1;
    c->session.state.logits=c->logits;c->anatomy.vocab=315;c->prompt=prompt;c->prompt_count=count;
    c->sequence=native_text_control_ids;c->sequence_count=6;c->capacity=128;c->max_batch=64;c->admit=1;
}
static int control_token(const unsigned char *piece,size_t bytes,int id,void *user) {
    Control *c=user;
    if(c->request) {
        assert(bn_generate_native_text_request_step(c->request)==BN_NATIVE_TEXT_ERROR);
        assert(bn_generate_native_text_request_cancel(c->request)==BN_NATIVE_TEXT_ERROR);
        bn_generate_native_text_request_free(c->request);assert(c->request->owner==c->request);
    }
    assert(bytes<=sizeof(c->output)-c->output_bytes);
    if(bytes){memcpy(c->output+c->output_bytes,piece,bytes);}c->output_bytes+=bytes;
    if(id<0){assert(id==-1);c->flushes++;return 0;}
    assert(c->callbacks<(size_t)c->sequence_count&&id==c->sequence[c->callbacks]);c->callbacks++;
    assert(c->session.pos==c->prompt_count+(int)c->callbacks-c->predict&&
        c->generated==(int)c->callbacks-c->predict&&!c->pending);
    return c->stop_at>0&&(int)c->callbacks==c->stop_at;
}
static void control_observe(const int *tokens,int n,int pos,const float *logits,int vocab,void *user) {
    Control *c=user;assert(n>0&&tokens&&logits==c->logits&&vocab==315&&pos+n==c->session.pos);c->observations++;
}
static void control_request_checks(const BnTokenizerEncoder *encoder,const BnTokenizerDecoder *decoder,
    const unsigned char *text,size_t bytes,const int *prompt,int count,
    const unsigned char *other,size_t other_bytes,const int *other_prompt,int other_count) {
    Control *c=calloc(2,sizeof(*c));assert(c);
    BnTokenizerEncodeScratch scratch[2]={{0},{0}};size_t sb;
    int tokens[2][64];unsigned char piece[2][128];
    BnGenerateNativeTextWorkspace w[2];
    assert(!bn_tokenizer_encode_scratch_plan(128,&sb));
    for(int i=0;i<2;i++) {
        assert(!bn_tokenizer_encode_scratch_init(scratch+i,128,sb));
        w[i]=(BnGenerateNativeTextWorkspace){encoder,decoder,scratch+i,tokens[i],64,piece[i],128};
    }
    /* Same immutable model and tokenizer, independent histories, UTF8 state,
     * batches and stop policies. Alternate even when one session returns BUSY. */
    for(int predict=0;predict<2;predict++) {
        control_init(c,prompt,count);control_init(c+1,other_prompt,other_count);c->peer=c+1;
        BnGenerateNativeTextRequest r[2]={{0},{0}};
        for(int i=0;i<2;i++) {
            c[i].predict=predict;c[i].busy_each=i+1;c[i].stop_at=i?3:6;
            BnGenerateNativeTextOptions o={i?1:3,6,1,i};
            BnGenerateNativeTextCallbacks cb={control_token,control_observe,c+i};
            assert(!bn_generate_native_text_request_init(r+i,&c->model,&c[i].session,w+i,
                i?other:text,i?other_bytes:bytes,&o,&cb,predict));
            assert(!c[i].stage_calls);c[i].request=r+i;
        }
        int done[2]={0},turns=0;
        while(!done[0]||!done[1]) {
            int i=turns++%2;assert(turns<256);
            int calls=c[i].forward_calls,stages=c[i].stage_calls;
            BnGenerateNativeTextStatus status=bn_generate_native_text_request_step(r+i);
            assert(c[i].forward_calls-calls<=1&&c[i].stage_calls-stages<=1);
            assert(status==BN_NATIVE_TEXT_PROGRESS||status==BN_NATIVE_TEXT_BUSY||status==BN_NATIVE_TEXT_DONE);
            done[i]=status==BN_NATIVE_TEXT_DONE;
        }
        for(int i=0;i<2;i++) {
            int limit=c[i].stop_at,found=0;
            assert(r[i].generated==limit&&c[i].callbacks==(size_t)limit&&c[i].flushes==1&&
                c[i].session.pos==c[i].prompt_count+limit-predict&&!c[i].pending);
            for(size_t k=0;k<sizeof(native_text_control_cases)/sizeof(native_text_control_cases[0]);k++) {
                if(native_text_control_cases[k].skip==i&&native_text_control_cases[k].limit==limit) {
                    assert(c[i].output_bytes==native_text_control_cases[k].count&&
                        !memcmp(c[i].output,native_text_control_cases[k].bytes,c[i].output_bytes));found=1;
                }
            }
            assert(found);
            int calls=c[i].forward_calls;
            assert(bn_generate_native_text_request_cancel(r+i)==BN_NATIVE_TEXT_DONE);
            assert(bn_generate_native_text_request_step(r+i)==BN_NATIVE_TEXT_DONE&&calls==c[i].forward_calls);
            bn_generate_native_text_request_free(r+i);assert(!r[i].owner);
        }
    }
    for(int predict=0;predict<2;predict++)for(int phase=0;phase<3;phase++) {
        control_init(c,prompt,count);c->predict=predict;
        BnGenerateNativeTextRequest r={0};BnGenerateNativeTextOptions o={3,6,1,1};
        BnGenerateNativeTextCallbacks cb={control_token,control_observe,c};
        assert(!bn_generate_native_text_request_init(&r,&c->model,&c->session,w,text,bytes,&o,&cb,predict));
        BnGenerateNativeTextRequest copy=r;
        assert(bn_generate_native_text_request_step(&copy)==BN_NATIVE_TEXT_ERROR);
        assert(bn_generate_native_text_request_cancel(&copy)==BN_NATIVE_TEXT_ERROR);
        bn_generate_native_text_request_free(&copy);assert(copy.owner==&r);
        assert(bn_generate_native_text_request_init(&r,&c->model,&c->session,w,text,bytes,&o,&cb,predict)==-2);
        if(phase) {
            while(r.prefilled<r.prompt_tokens)assert(bn_generate_native_text_request_step(&r)==BN_NATIVE_TEXT_PROGRESS);
            c->busy_each=2;
            assert(bn_generate_native_text_request_step(&r)==BN_NATIVE_TEXT_BUSY);
            assert(c->callbacks==(size_t)predict&&c->pending==1&&r.generated==predict);
        }
        int calls=c->forward_calls,stages=c->stage_calls;size_t callbacks=c->callbacks;
        if(phase==2) {
            bn_generate_native_text_request_free(&r);assert(!r.owner);
            assert(bn_generate_native_text_request_step(&r)==BN_NATIVE_TEXT_ERROR);
        }else {
            assert(bn_generate_native_text_request_cancel(&r)==BN_NATIVE_TEXT_CANCELLED);
            assert(bn_generate_native_text_request_cancel(&r)==BN_NATIVE_TEXT_CANCELLED);
            assert(bn_generate_native_text_request_step(&r)==BN_NATIVE_TEXT_CANCELLED);
            bn_generate_native_text_request_free(&r);
        }
        assert(c->forward_calls==calls&&c->stage_calls==stages&&c->callbacks==callbacks&&!c->flushes);
        assert(c->pending==(phase?1:0)); /* Reset/free of session owns reclamation. */
    }
    /* Error is terminal, and a BUSY retry may fail after output in prediction mode. */
    for(int predict=0;predict<2;predict++)for(int failure=0;failure<4;failure++) {
        control_init(c,prompt,count);c->predict=predict;c->busy_each=2;
        int prefill=(count+2)/3;
        if(failure==0)c->fail_stage=prefill+1;
        if(failure==1)c->fail_forward=prefill+1;
        if(failure==2)c->stale_ticket=1;
        if(failure==3)c->nonfinite=1;
        BnGenerateNativeTextRequest r={0};BnGenerateNativeTextOptions o={3,6,1,1};
        BnGenerateNativeTextCallbacks cb={control_token,control_observe,c};
        assert(!bn_generate_native_text_request_init(&r,&c->model,&c->session,w,text,bytes,&o,&cb,predict));
        BnGenerateNativeTextStatus status;int turns=0;
        do{status=bn_generate_native_text_request_step(&r);assert(++turns<128);}
        while(status==BN_NATIVE_TEXT_PROGRESS||status==BN_NATIVE_TEXT_BUSY);
        assert(status==BN_NATIVE_TEXT_ERROR&&!c->flushes&&c->callbacks==(size_t)(predict&&failure<2));
        int calls=c->forward_calls,stages=c->stage_calls;
        assert(bn_generate_native_text_request_step(&r)==BN_NATIVE_TEXT_ERROR);
        assert(bn_generate_native_text_request_cancel(&r)==BN_NATIVE_TEXT_ERROR);
        assert(c->forward_calls==calls&&c->stage_calls==stages);
        bn_generate_native_text_request_free(&r);
    }
    control_init(c,prompt,count);
    BnGenerateNativeTextRequest r={0};BnGenerateNativeTextOptions o={3,6,1,1};
    BnGenerateNativeTextWorkspace alias=w[0];alias.piece=(unsigned char*)&r;
    assert(bn_generate_native_text_request_init(&r,&c->model,&c->session,&alias,text,bytes,&o,NULL,0)==-2&&!r.owner);
    alias=w[0];alias.tokens=(int*)(void*)&r;
    assert(bn_generate_native_text_request_init(&r,&c->model,&c->session,&alias,text,bytes,&o,NULL,0)==-2&&!r.owner);
    for(int predict=-1;predict<=2;predict+=3)
        assert(bn_generate_native_text_request_init(&r,&c->model,&c->session,w,text,bytes,&o,NULL,predict)==-2&&!r.owner);
    assert(!bn_generate_native_text_request_init(&r,&c->model,&c->session,w,text,bytes,&o,NULL,0));
    c->session.pos=1;assert(bn_generate_native_text_request_step(&r)==BN_NATIVE_TEXT_ERROR&&!c->stage_calls);
    bn_generate_native_text_request_free(&r);
    for(int i=0;i<2;i++)bn_tokenizer_encode_scratch_free(scratch+i);
    free(c);
}

static int invoke(Control *c,BnGenerateNativeTextWorkspace *w,const unsigned char *text,size_t bytes,
    BnGenerateNativeTextOptions *o,BnGenerateNativeTextCallbacks *cb) {
#ifdef BN_TEST_DENY_HEAP
    control_hot=1;control_heap_ops=0;
#endif
    int rc;
    if(c->resumable) {
        BnGenerateNativeTextRequest r={0};int attempts=c->forward_calls;
        rc=bn_generate_native_text_request_init(&r,&c->model,&c->session,w,text,bytes,o,cb,c->predict);
        assert(c->forward_calls==attempts);
        if(!rc) {
            BnGenerateNativeTextStatus status;
            do {
                int before=c->forward_calls,stages=c->stage_calls,done=c->completed;
                status=bn_generate_native_text_request_step(&r);
                assert(c->forward_calls-before<=1&&c->stage_calls-stages<=1&&c->completed-done<=1);
            }while(status==BN_NATIVE_TEXT_PROGRESS||status==BN_NATIVE_TEXT_BUSY);
            rc=status==BN_NATIVE_TEXT_DONE?r.generated:-2;
            int before=c->forward_calls;size_t tokens=c->callbacks,flushes=c->flushes;
            assert(bn_generate_native_text_request_step(&r)==status);
            assert(c->forward_calls==before&&c->callbacks==tokens&&c->flushes==flushes);
        }
        bn_generate_native_text_request_free(&r);
    }else rc=c->predict?bn_generate_native_text_predict(&c->model,&c->session,w,text,bytes,o,cb):
        bn_generate_native_text(&c->model,&c->session,w,text,bytes,o,cb);
#ifdef BN_TEST_DENY_HEAP
    control_hot=0;assert(!control_heap_ops);
#endif
    return rc;
}
typedef struct {const BnTokenizerEncoder *encoder;const BnTokenizerDecoder *decoder;size_t cases;} Worker;
static void *control_worker(void *arg) {
    Worker *worker=arg;BnTokenizerEncodeScratch scratch={0};size_t sb;
    assert(!bn_tokenizer_encode_scratch_plan(128,&sb)&&!bn_tokenizer_encode_scratch_init(&scratch,128,sb));
    int tokens[64];unsigned char piece[129];Control *c=calloc(1,sizeof(*c));assert(c);
    BnGenerateNativeTextWorkspace w={worker->encoder,worker->decoder,&scratch,tokens,64,piece,128};
    const unsigned char *text=(const unsigned char*)"abc123def456";const int *prompt=NULL;int count=0;
    for(size_t i=0;i<sizeof(native_encode_cases)/sizeof(native_encode_cases[0]);i++) {
        if(native_encode_cases[i].bytes==12&&!memcmp(native_encode_cases[i].text,text,12)) {
            prompt=native_encode_cases[i].ids;count=(int)native_encode_cases[i].count;break;
        }
    }
    assert(prompt&&count>3);
    for(int resumable=0;resumable<=1;resumable++)for(int predict=0;predict<=1;predict++)for(size_t k=0;k<sizeof(native_text_control_cases)/sizeof(native_text_control_cases[0]);k++)for(int busy=0;busy<=2;busy+=2)for(int batch=1;batch<=64;batch=batch==1?3:64) {
        control_init(c,prompt,count);c->resumable=resumable;c->predict=predict;c->busy_each=busy;c->stop_at=native_text_control_cases[k].limit;
        BnGenerateNativeTextOptions o={batch,c->stop_at?6:0,1,native_text_control_cases[k].skip};
        BnGenerateNativeTextCallbacks cb={control_token,control_observe,c};memset(piece,0xa5,sizeof(piece));
        int rc=invoke(c,&w,text,12,&o,&cb),limit=native_text_control_cases[k].limit;
        int commits=limit-(predict&&limit>0);
        assert(rc==limit&&c->generated==commits&&c->callbacks==(size_t)limit&&c->flushes==1);
        assert(c->output_bytes==native_text_control_cases[k].count&&!memcmp(c->output,native_text_control_cases[k].bytes,c->output_bytes));
        int prefill=(count+batch-1)/batch;
        assert(c->stage_calls==prefill+commits&&c->completed==c->stage_calls&&c->forward_calls==c->stage_calls*(busy+1));
        assert(c->observations==(size_t)c->completed&&piece[128]==0xa5);worker->cases++;
        if(batch==64)break;
    }
    BnGenerateNativeTextOptions o={3,6,1,1};BnGenerateNativeTextCallbacks cb={control_token,control_observe,c};
    /* Failed decode publishes no token, and BUSY never restages it. */
    control_init(c,prompt,count);c->busy_each=2;c->fail_forward=(count+2)/3+2;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&c->callbacks==1&&!c->flushes&&c->session.pos==count+1);
    control_init(c,prompt,count);c->fail_stage=2;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->callbacks&&!c->flushes&&c->session.pos==3);
    control_init(c,prompt,count);c->stale_ticket=1;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->callbacks&&!c->observations);
    control_init(c,prompt,count);c->capacity=count+5;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls);
    control_init(c,prompt,count);c->max_batch=2;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls);
    control_init(c,prompt,count);w.token_capacity=1;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls);w.token_capacity=64;
    control_init(c,prompt,count);w.piece_capacity=5;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls);w.piece_capacity=128;
    control_init(c,prompt,count);o.skip_special=0;w.piece_capacity=6;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&c->generated==1&&c->callbacks==1&&!c->flushes&&!c->pending);w.piece_capacity=128;o.skip_special=1;
    control_init(c,prompt,count);unsigned char *old_piece=w.piece;w.piece=(unsigned char*)tokens;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls);w.piece=old_piece;
    control_init(c,prompt,count);const unsigned char bad[]={0xc0,0x80};
    assert(invoke(c,&w,bad,sizeof(bad),&o,&cb)==-2&&!c->stage_calls);
    control_init(c,prompt,count);assert(invoke(c,&w,NULL,0,&o,&cb)==-2&&!c->stage_calls);
    control_init(c,prompt,count);c->session.pos=1;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls);
    control_init(c,prompt,count);c->session.backend_only=0;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls);
    /* Greedy ties choose ID zero and NUL survives counted output. */
    control_init(c,prompt,count);const int tie[]={0};c->sequence=tie;c->sequence_count=1;c->tie=1;o.max_new_tokens=1;
    assert(invoke(c,&w,text,12,&o,&cb)==1&&c->output_bytes==1&&!c->output[0]&&c->flushes==1);
    control_init(c,prompt,count);o.max_new_tokens=0;
    assert(invoke(c,&w,text,12,&o,NULL)==0&&c->session.pos==count);
    control_init(c,prompt,count);c->nonfinite=1;o.max_new_tokens=1;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->callbacks&&c->generated==0&&c->session.pos==count);
    BnGenerateNativeTextOptions invalids[]={{0,1,1,1},{3,-1,1,1},{3,1,2,1},{3,1,1,2}};
    for(size_t i=0;i<sizeof(invalids)/sizeof(invalids[0]);i++) {
        control_init(c,prompt,count);assert(invoke(c,&w,text,12,invalids+i,&cb)==-2&&!c->stage_calls);
    }
    /* Counted prompt bytes include NUL and cannot pass through strlen. */
    const unsigned char nul_text[]={0,'a',0};const int *nul_ids=NULL;int nul_count=0;
    for(size_t i=0;i<sizeof(native_encode_cases)/sizeof(native_encode_cases[0]);i++) {
        if(native_encode_cases[i].bytes==sizeof(nul_text)&&!memcmp(native_encode_cases[i].text,nul_text,sizeof(nul_text))) {
            nul_ids=native_encode_cases[i].ids;nul_count=(int)native_encode_cases[i].count;break;
        }
    }
    assert(nul_ids&&nul_count);control_init(c,nul_ids,nul_count);o.max_new_tokens=0;
    assert(invoke(c,&w,nul_text,sizeof(nul_text),&o,&cb)==0&&c->session.pos==nul_count&&c->flushes==1);
    control_init(c,prompt,count);o.max_new_tokens=INT_MAX;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls);o.max_new_tokens=0;
    control_init(c,prompt,count);w.token_capacity=SIZE_MAX;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls);w.token_capacity=64;
    control_init(c,prompt,count);w.piece_capacity=SIZE_MAX;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls);w.piece_capacity=128;
    /* Prediction-first output precedes staging, and a later failure can
     * follow visible output. Capacity covers committed inputs only. */
    int prefill=(count+2)/3;
    control_init(c,prompt,count);c->predict=1;c->capacity=count;o.max_new_tokens=1;
    assert(invoke(c,&w,text,12,&o,&cb)==1&&c->callbacks==1&&c->flushes==1&&
        c->stage_calls==prefill&&c->session.pos==count&&!c->generated);
    control_init(c,prompt,count);c->predict=1;c->capacity=count;o.max_new_tokens=0;
    assert(invoke(c,&w,text,12,&o,&cb)==0&&!c->callbacks&&c->flushes==1&&c->session.pos==count);
    control_init(c,prompt,count);c->predict=1;c->capacity=count+5;o.max_new_tokens=6;
    assert(invoke(c,&w,text,12,&o,&cb)==6&&c->callbacks==6&&c->generated==5&&
        c->flushes==1&&c->session.pos==count+5);
    control_init(c,prompt,count);c->predict=1;c->capacity=count+4;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->stage_calls&&!c->callbacks);
    control_init(c,prompt,count);c->predict=1;c->stop_at=1;
    assert(invoke(c,&w,text,12,&o,&cb)==1&&c->callbacks==1&&c->flushes==1&&c->stage_calls==prefill);
    control_init(c,prompt,count);c->predict=1;c->fail_stage=prefill+1;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&c->callbacks==1&&!c->flushes&&c->session.pos==count);
    control_init(c,prompt,count);c->predict=1;c->busy_each=2;c->fail_forward=prefill+2;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&c->callbacks==2&&!c->flushes&&c->session.pos==count+1);
    control_init(c,prompt,count);c->predict=1;c->nonfinite=1;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&!c->callbacks&&c->session.pos==count);
    control_init(c,prompt,count);c->predict=1;o.max_new_tokens=1;
    assert(invoke(c,&w,text,12,&o,NULL)==1&&c->session.pos==count&&c->stage_calls==prefill);
    control_init(c,prompt,count);c->predict=1;o.max_new_tokens=6;o.skip_special=0;w.piece_capacity=6;
    assert(invoke(c,&w,text,12,&o,&cb)==-2&&c->callbacks==1&&c->generated==1&&!c->flushes&&!c->pending);
    w.piece_capacity=128;
    control_request_checks(worker->encoder,worker->decoder,text,12,prompt,count,nul_text,sizeof(nul_text),nul_ids,nul_count);
    free(c);bn_tokenizer_encode_scratch_free(&scratch);return NULL;
}
#include "native_schedule_checks.h"

int main(void) {
    BnJSON doc={0};assert(!bn_json_init(&doc,native_encode_fixture,strlen(native_encode_fixture)));
    BnTokenizerEncoder encoder={0};BnTokenizerDecoder decoder={0};BnTokenizerEncoderPlan ep;size_t db;
    assert(!bn_tokenizer_encoder_plan(&doc,315,&ep)&&!bn_tokenizer_encoder_init_json(&encoder,&doc,315,ep.host_bytes));
    assert(!bn_tokenizer_decoder_plan(315,&db)&&!bn_tokenizer_decoder_init_json(&decoder,&doc,315,db));
    native_schedule_checks(&encoder,&decoder);
    Worker workers[2]={{&encoder,&decoder,0},{&encoder,&decoder,0}};pthread_t threads[2];
    for(int i=0;i<2;i++)assert(!pthread_create(threads+i,NULL,control_worker,workers+i));
    for(int i=0;i<2;i++)assert(!pthread_join(threads[i],NULL));
    assert(workers[0].cases==336&&workers[1].cases==336);
    bn_tokenizer_decoder_free(&decoder);bn_tokenizer_encoder_free(&encoder);bn_json_free(&doc);
    puts("Native text generation: 672 blocking/resumable committed/prediction source-prefix/stop/batch/BUSY cases, failures, counted UTF8/NUL, shared tokenizer models PASSED");return 0;
}
