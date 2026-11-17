#include <assert.h>
#include <stdio.h>
#include <string.h>
static double clock_value;
static double fake_clock(void){return clock_value;}
#define BN_TEXT_TIMING_CLOCK() fake_clock()
#include "../bench/native_text_timing.h"
#include "../bench/native_schedule_metrics.h"
typedef struct {
    BnTextTiming recorder;
    BnTextTimingFrame frames[8];BnTextTimingToken tokens[8];
    int inputs[16];unsigned char bytes[16];
} Case;
static const int prompt[]={3,4,5},selected[]={1,2};
static const float logits[16]={0};
static void init(Case *c,int predict,int outputs) {
    memset(c,0,sizeof(*c));
    c->recorder=(BnTextTiming){.prompt=3,.max_new=outputs,.vocabulary=16,.predict=predict,
        .frames=c->frames,.frame_capacity=8,.tokens=c->tokens,.token_capacity=8,
        .inputs=c->inputs,.input_capacity=16,.bytes=c->bytes,.byte_capacity=16};
    clock_value=10;assert(!native_text_timing_start(&c->recorder));
}
static void prefill(Case *c) {
    clock_value=12;native_text_timing_forward(prompt,2,0,logits,16,&c->recorder);
    clock_value=20;native_text_timing_forward(prompt+2,1,2,logits,16,&c->recorder);
    assert(!c->recorder.bad&&c->recorder.prefill==20);
}
static void valid(int predict) {
    Case c;init(&c,predict,2);prefill(&c);BnTextTiming *r=&c.recorder;
    for(int i=0;i<2;i++) {
        if(!predict){clock_value+=7;native_text_timing_forward(selected+i,1,3+i,logits,16,r);}
        clock_value+=1;assert(!native_text_timing_token((const unsigned char*)"ab"+i,1,selected[i],r));
        if(predict&&i==0){clock_value+=7;native_text_timing_forward(selected+i,1,3+i,logits,16,r);}
    }
    clock_value+=1;assert(!native_text_timing_token(NULL,0,-1,r));
    clock_value+=1;assert(!native_text_timing_finish(r,2));
    assert(r->first_token-r->prefill==(predict?1:8)&&r->first_byte==r->first_token);
    assert(r->frame_count==(size_t)(predict?3:4)&&r->input_count==(size_t)(predict?4:5));
    assert(r->position==(predict?4:5)&&r->generated==2&&r->decoded==(predict?1:2));
    assert(r->token_count==3&&r->byte_count==2&&!memcmp(r->bytes,"ab",2));
    assert(!memcmp(r->inputs,prompt,sizeof(prompt))&&r->inputs[3]==1);
    if(!predict)assert(r->inputs[4]==2);
    assert(r->end-r->start==(predict?21:28));
}
int main(void) {
    valid(0);valid(1);Case c;BnTextTiming *r=&c.recorder;
    for(int mode=0;mode<=1;mode++) {
        init(&c,mode,0);prefill(&c);clock_value=21;
        assert(!native_text_timing_token(NULL,0,-1,r));clock_value=22;
        assert(!native_text_timing_finish(r,0)&&r->first_token==-1&&r->first_byte==-1&&r->position==3);
    }
    /* Empty token chunks and a UTF8 flush have distinct availability times. */
    init(&c,1,1);prefill(&c);clock_value=21;
    assert(!native_text_timing_token(NULL,0,1,r));clock_value=22;
    const unsigned char tail[]={0xef,0xbf,0xbd};
    assert(!native_text_timing_token(tail,3,-1,r));clock_value=23;
    assert(!native_text_timing_finish(r,1)&&r->first_token==21&&r->first_byte==22&&r->byte_count==3);
    /* Callback ordering, clocks, geometry, and bounded storage must reject
     * malformed observations rather than silently producing latency metrics. */
    init(&c,0,1);prefill(&c);clock_value=21;
    assert(native_text_timing_token(NULL,0,1,r)==1&&r->bad);
    init(&c,1,1);prefill(&c);clock_value=21;
    native_text_timing_forward(selected,1,3,logits,16,r);assert(r->bad);
    init(&c,1,2);prefill(&c);clock_value=21;
    native_text_timing_forward(selected,1,3,logits,16,r);assert(r->bad);
    init(&c,1,2);clock_value=11;
    assert(native_text_timing_token(NULL,0,1,r)==1&&r->bad);
    init(&c,1,2);clock_value=9;
    native_text_timing_forward(prompt,2,0,logits,16,r);assert(r->bad);
    init(&c,1,2);clock_value=NAN;
    native_text_timing_forward(prompt,2,0,logits,16,r);assert(r->bad);
    init(&c,1,2);clock_value=11;
    native_text_timing_forward(prompt,2,1,logits,16,r);assert(r->bad);
    init(&c,1,2);clock_value=11;
    native_text_timing_forward(prompt,2,0,logits,15,r);assert(r->bad);
    init(&c,1,2);r->frame_capacity=1;clock_value=11;
    memset(c.frames+1,0xa5,sizeof(c.frames[1]));unsigned char guard[sizeof(c.frames[1])];memcpy(guard,c.frames+1,sizeof(guard));
    native_text_timing_forward(prompt,2,0,logits,16,r);clock_value=12;
    native_text_timing_forward(prompt+2,1,2,logits,16,r);
    assert(r->bad&&!memcmp(guard,c.frames+1,sizeof(guard)));
    init(&c,1,1);prefill(&c);r->byte_capacity=0;clock_value=21;
    c.bytes[0]=0xa5;assert(native_text_timing_token((const unsigned char*)"x",1,1,r)==1&&r->bad&&c.bytes[0]==0xa5);
    init(&c,1,1);prefill(&c);clock_value=21;
    assert(native_text_timing_token(NULL,0,16,r)==1&&r->bad);
    init(&c,1,1);prefill(&c);clock_value=21;
    assert(!native_text_timing_token(NULL,0,1,r));clock_value=22;
    assert(native_text_timing_finish(r,1)==-1&&r->bad); /* missing flush */
    init(&c,1,1);prefill(&c);clock_value=21;
    assert(!native_text_timing_token(NULL,0,1,r));clock_value=22;
    assert(!native_text_timing_token(NULL,0,-1,r));clock_value=23;
    assert(native_text_timing_token(NULL,0,-1,r)==1&&r->bad);
    init(&c,1,1);r->input_capacity=2;assert(native_text_timing_start(r)==-1);
    init(&c,1,1);r->predict=2;assert(native_text_timing_start(r)==-1);
    init(&c,1,0);prefill(&c);clock_value=21;
    assert(!native_text_timing_token(NULL,0,-1,r));clock_value=22;
    assert(!native_text_timing_finish(r,0));clock_value=23;
    assert(native_text_timing_finish(r,0)==-1&&r->bad&&r->end==22);
    /* End-to-end tails use nearest rank, exclude missing observations, and
     * cannot mutate caller records used for per-request reports. */
    const double samples[]={50,10,30,20,40,60};BnNativeServingDistribution distribution;
    assert(!native_serving_distribution(samples,6,&distribution));
    assert(distribution.count==6&&distribution.minimum==10&&distribution.p50==30&&
        distribution.p95==60&&distribution.p99==60&&distribution.maximum==60&&samples[0]==50);
    assert(!native_serving_distribution(samples,1,&distribution)&&distribution.p99==50);
    double invalid[]={NAN};assert(native_serving_distribution(invalid,1,&distribution)==-1);
    invalid[0]=-1;assert(native_serving_distribution(invalid,1,&distribution)==-1);
    assert(native_serving_distribution(samples,0,&distribution)==-1);
    assert(native_serving_distribution(samples,101,&distribution)==-1);
    puts("Prepared text timing: callback order, clocks, UTF8 availability, geometry and bounded recording PASSED");
    return 0;
}
