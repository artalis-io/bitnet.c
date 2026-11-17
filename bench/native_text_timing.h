#ifndef BN_BENCH_NATIVE_TEXT_TIMING_H
#define BN_BENCH_NATIVE_TEXT_TIMING_H
#include "platform.h"
#include <limits.h>
#include <math.h>
#include <string.h>
#ifndef BN_TEXT_TIMING_CLOCK
#define BN_TEXT_TIMING_CLOCK() bn_platform_time_ms()
#endif
/* Caller-owned bounded recording only: no heap, file IO or GPU operations. */
typedef struct {int position,rows;size_t offset;double time;} BnTextTimingFrame;
typedef struct {int id;size_t bytes;double time;} BnTextTimingToken;
typedef struct {
    int prompt,max_new,vocabulary,predict,position,decoded,generated,flushed,bad;
    BnTextTimingFrame *frames;size_t frame_capacity,frame_count;
    BnTextTimingToken *tokens;size_t token_capacity,token_count;
    int *inputs;size_t input_capacity,input_count;
    unsigned char *bytes;size_t byte_capacity,byte_count;
    double start,last,prefill,first_token,first_byte,end;
} BnTextTiming;
static inline int native_text_timing_start(BnTextTiming *r) {
    if(!r||r->prompt<1||r->max_new<0||r->vocabulary<1||
        (r->predict!=0&&r->predict!=1))return -1;
    int commits=r->max_new-(r->predict&&r->max_new>0);
    if(r->prompt>INT_MAX-commits||!r->frames||!r->frame_capacity||
        !r->tokens||r->token_capacity<(size_t)r->max_new+1||!r->inputs||
        r->input_capacity<(size_t)r->prompt+(size_t)commits||
        (r->byte_capacity&&!r->bytes))return -1;
    double now=BN_TEXT_TIMING_CLOCK();if(!isfinite(now)||now<0)return -1;
    r->position=r->decoded=r->generated=r->flushed=r->bad=0;
    r->frame_count=r->token_count=r->input_count=r->byte_count=0;
    r->start=r->last=now;r->prefill=r->first_token=r->first_byte=r->end=-1;return 0;
}
static inline int native_text_timing_tick(BnTextTiming *r,double *now) {
    *now=BN_TEXT_TIMING_CLOCK();
    if(r->bad||!isfinite(*now)||*now<r->last){r->bad=1;return -1;}
    r->last=*now;return 0;
}
static inline void native_text_timing_forward(const int *ids,int rows,int pos,
    const float *logits,int vocabulary,void *user) {
    BnTextTiming *r=user;double now;
    if(native_text_timing_tick(r,&now))return;
    if(r->flushed||!ids||!logits||vocabulary!=r->vocabulary||rows<1||
        pos!=r->position||pos>INT_MAX-rows||r->frame_count>=r->frame_capacity||
        (size_t)rows>r->input_capacity-r->input_count){r->bad=1;return;}
    if(pos<r->prompt) {
        if(rows>r->prompt-pos||r->generated){r->bad=1;return;}
    }else {
        int limit=r->max_new-(r->predict&&r->max_new>0);
        if(rows!=1||r->decoded>=limit||r->generated!=r->decoded+r->predict){r->bad=1;return;}
        r->decoded++;
    }
    r->frames[r->frame_count++]=(BnTextTimingFrame){pos,rows,r->input_count,now};
    memcpy(r->inputs+r->input_count,ids,(size_t)rows*sizeof(int));r->input_count+=(size_t)rows;
    r->position+=rows;if(r->position==r->prompt)r->prefill=now;
}
static inline int native_text_timing_token(const unsigned char *bytes,size_t n,int id,void *user) {
    BnTextTiming *r=user;double now;
    if(native_text_timing_tick(r,&now))return 1;
    if(r->flushed||r->prefill<0||(n&&!bytes)||id< -1||id>=r->vocabulary||
        r->token_count>=r->token_capacity||n>r->byte_capacity-r->byte_count){r->bad=1;return 1;}
    if(id>=0) {
        if(r->generated>=r->max_new||r->position!=r->prompt+r->generated+!r->predict){r->bad=1;return 1;}
        if(!r->generated)r->first_token=now;
        r->generated++;
    }else {
        if(r->decoded!=r->generated-(r->predict&&r->generated>0)){r->bad=1;return 1;}
        r->flushed=1;
    }
    if(n&&r->first_byte<0)r->first_byte=now;
    if(n)memcpy(r->bytes+r->byte_count,bytes,n);
    r->byte_count+=n;r->tokens[r->token_count++]=(BnTextTimingToken){id,n,now};return 0;
}
static inline int native_text_timing_finish(BnTextTiming *r,int generated) {
    if(r->end>=0){r->bad=1;return -1;}
    double now;if(native_text_timing_tick(r,&now))return -1;
    r->end=now;
    if(!r->flushed||r->prefill<0||generated!=r->generated||generated<0||
        r->decoded!=generated-(r->predict&&generated>0)||
        r->position!=r->prompt+r->decoded){r->bad=1;return -1;}
    return 0;
}
#endif
