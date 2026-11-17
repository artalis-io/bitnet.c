#ifndef BN_BENCH_NATIVE_TEXT_WORKLOAD_H
#define BN_BENCH_NATIVE_TEXT_WORKLOAD_H
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

/* Benchmark-only bounded round-robin inputs; no model/session ownership. */
#define BN_TEXT_WORKLOAD_MAX_REQUESTS 8
typedef struct {
    int max_rows,end;
    size_t frames;
} BnTextWorkloadShape;

static int native_text_workload_add(BnTextWorkloadShape *s,int rows,int position) {
    if(!s||rows<1||rows>16384||position<0||position>INT_MAX-rows||
       s->frames==SIZE_MAX)return -1;
    if(rows>s->max_rows)s->max_rows=rows;
    if(position+rows>s->end)s->end=position+rows;
    s->frames++;return 0;
}
static int native_text_workload_finish(const BnTextWorkloadShape *s,
    int *prefill,int *context) {
    if(!s||!prefill||!context||!s->frames||s->max_rows<1||
       s->max_rows>16384||s->end<s->max_rows||s->end>=INT_MAX-255)return -1;
    *prefill=s->max_rows;*context=(s->end+255)/256*256;return 0;
}
static int native_text_workload_pick(int run,int count) {
    if(run<0||count<1||count>BN_TEXT_WORKLOAD_MAX_REQUESTS)return -1;
    return run%count;
}
static int native_text_workload_skip(int run,int count) {
    if(native_text_workload_pick(run,count)<0)return -1;
    /* Both skip modes occur for every input, independent of list position. */
    return (run/count)&1;
}
#endif
