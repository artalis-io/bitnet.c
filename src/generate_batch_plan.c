#include "generate_batch_plan.h"
#include <stdio.h>

int bn_generate_batch_plan_with_prefill_policy(const BnGenerateBatchLimits *l,
    const BnGenerateBatchCandidate *c,int count,BnGenerateBatchPrefillPolicy policy,
    BnGenerateBatchPlan *out) {
    if((policy!=BN_GENERATE_BATCH_PREFILL_FIXED&&policy!=BN_GENERATE_BATCH_PREFILL_RECLAIM)||
       !l||!out||count<0||count>BN_GENERATE_BATCH_MAX_REQUESTS||(!c&&count)||
       l->max_requests<1||l->max_requests>BN_GENERATE_BATCH_MAX_REQUESTS||
       l->max_prefills<1||l->max_prefills>l->max_requests||
       l->max_chunk<1||l->max_chunk>BN_GENERATE_BATCH_MAX_CHUNK_TOKENS||
       l->token_budget<1||l->token_budget>BN_GENERATE_BATCH_MAX_TOKENS||
       l->capacity<1||l->mixed_token_budget<0||
       (l->mixed_token_budget&&(l->mixed_token_budget<l->max_requests||
        l->mixed_token_budget>l->token_budget)))goto invalid;
    for(int i=0;i<count;i++) {
        if(c[i].prompt_tokens<1||c[i].prefilled<0||c[i].prefilled>c[i].prompt_tokens||
           c[i].generated<0||c[i].max_outputs<1||c[i].generated>c[i].max_outputs||
           c[i].prompt_tokens>l->capacity||c[i].max_outputs-1>l->capacity-c[i].prompt_tokens||
           (c[i].prefilled<c[i].prompt_tokens?c[i].generated!=0:c[i].generated<1))goto invalid;
    }
    BnGenerateBatchPlan plan={0};int selected[BN_GENERATE_BATCH_MAX_REQUESTS],prefills=0;
    for(int i=0;i<count&&plan.count<l->max_requests&&plan.count<l->token_budget;i++) {
        if(c[i].prefilled==c[i].prompt_tokens&&c[i].generated<c[i].max_outputs) {
            plan.rows[plan.count++]=(BnGenerateBatchRow){i,1,c[i].prompt_tokens+(c[i].generated-1),0,1,1};
            plan.total_tokens++;
        }
    }
    int budget=plan.count&&l->mixed_token_budget?l->mixed_token_budget:l->token_budget;
    int remaining=budget-plan.total_tokens;
    for(int i=0;i<count&&prefills<l->max_prefills&&plan.count+prefills<l->max_requests&&prefills<remaining;i++)
        if(c[i].prefilled<c[i].prompt_tokens)selected[prefills++]=i;
    int share=prefills?remaining/prefills:0;if(share>l->max_chunk)share=l->max_chunk;
    if(prefills&&policy==BN_GENERATE_BATCH_PREFILL_RECLAIM) {
        int high=l->max_chunk;
        while(share<high) {
            int trial=share+(high-share+1)/2,used=0;
            for(int j=0;j<prefills;j++) {
                int i=selected[j],n=c[i].prompt_tokens-c[i].prefilled;
                used+=n<trial?n:trial;
            }
            if(used<=remaining)share=trial;else high=trial-1;
        }
    }
    for(int j=0;j<prefills;j++) {
        int i=selected[j],n=c[i].prompt_tokens-c[i].prefilled;if(n>share)n=share;
        plan.rows[plan.count++]=(BnGenerateBatchRow){i,n,c[i].prefilled,1,n==c[i].prompt_tokens-c[i].prefilled,0};
        plan.total_tokens+=n;
    }
    /* Restore admission order for ties after selecting decode capacity first. */
    for(int i=1;i<plan.count;i++) {
        BnGenerateBatchRow row=plan.rows[i];int j=i;
        while(j&&(plan.rows[j-1].n_tokens>row.n_tokens||
            (plan.rows[j-1].n_tokens==row.n_tokens&&plan.rows[j-1].candidate>row.candidate))) {
            plan.rows[j]=plan.rows[j-1];j--;
        }
        plan.rows[j]=row;
    }
    int tail=0;
    for(int i=0;i<plan.count;i++) {
        if(plan.rows[i].is_prefill)tail=1;
        plan.rows[i].decode_role=!tail;
    }
    *out=plan;return plan.count?1:0;
invalid:
    fprintf(stderr,"generate batch plan: invalid limits or request progress\n");return -1;
}

int bn_generate_batch_plan(const BnGenerateBatchLimits *l,
    const BnGenerateBatchCandidate *c,int count,BnGenerateBatchPlan *out) {
    return bn_generate_batch_plan_with_prefill_policy(l,c,count,
        BN_GENERATE_BATCH_PREFILL_FIXED,out);
}

static int memory_quote_valid(BnGenerateBatchMemoryQuote q) {
    return q.peak.host_bytes>=q.retained.host_bytes&&q.peak.device_bytes>=q.retained.device_bytes;
}
static int memory_fits(BnGenerateBatchMemory have,BnGenerateBatchMemory add,BnGenerateBatchMemory cap) {
    return have.host_bytes<=cap.host_bytes&&have.device_bytes<=cap.device_bytes&&
        add.host_bytes<=cap.host_bytes-have.host_bytes&&add.device_bytes<=cap.device_bytes-have.device_bytes;
}
static void memory_add(BnGenerateBatchMemory *a,BnGenerateBatchMemory b) {
    a->host_bytes+=b.host_bytes;a->device_bytes+=b.device_bytes;
}
static void memory_sub(BnGenerateBatchMemory *a,BnGenerateBatchMemory b) {
    a->host_bytes-=b.host_bytes;a->device_bytes-=b.device_bytes;
}
static void memory_peak(BnGenerateBatchMemoryLedger *l) {
    if(l->used.host_bytes>l->peak.host_bytes)l->peak.host_bytes=l->used.host_bytes;
    if(l->used.device_bytes>l->peak.device_bytes)l->peak.device_bytes=l->used.device_bytes;
}
static BnGenerateBatchMemorySlot *memory_ticket(BnGenerateBatchMemoryLedger *l,BnGenerateBatchMemoryTicket t) {
    if(!l||!l->initialized||t.owner!=l||t.slot<0||t.slot>=BN_GENERATE_BATCH_MAX_REQUESTS||!t.generation)return NULL;
    BnGenerateBatchMemorySlot *s=l->slots+t.slot;
    return s->state&&s->generation==t.generation?s:NULL;
}
int bn_generate_batch_memory_init(BnGenerateBatchMemoryLedger *l,
    BnGenerateBatchMemory cap,BnGenerateBatchMemoryQuote shared) {
    if(!l||l->initialized||!memory_quote_valid(shared)||
       !memory_fits((BnGenerateBatchMemory){0},shared.peak,cap))return -1;
    BnGenerateBatchMemoryLedger next={0};next.initialized=1;next.limit=cap;next.shared=shared.retained;
    next.used=shared.peak;next.peak=shared.peak;next.shared_preparing=1;*l=next;return 0;
}
int bn_generate_batch_memory_shared_commit_as(BnGenerateBatchMemoryLedger *l,BnGenerateBatchMemory retained) {
    if(!l||!l->initialized||!l->shared_preparing||retained.host_bytes>l->shared.host_bytes||
       retained.device_bytes>l->shared.device_bytes)return -1;
    l->shared=retained;l->used=retained;l->shared_preparing=0;return 0;
}
int bn_generate_batch_memory_shared_commit(BnGenerateBatchMemoryLedger *l) {
    if(!l)return -1;
    return bn_generate_batch_memory_shared_commit_as(l,l->shared);
}
int bn_generate_batch_memory_shared_prepare_as(BnGenerateBatchMemoryLedger *l,
    BnGenerateBatchMemoryQuote q) {
    if(!l||!l->initialized||!l->shared_preparing||!memory_quote_valid(q)||
       !memory_fits((BnGenerateBatchMemory){0},q.peak,l->limit))return -1;
    l->shared=q.retained;l->used=q.peak;memory_peak(l);return 0;
}
int bn_generate_batch_memory_reserve(BnGenerateBatchMemoryLedger *l,
    BnGenerateBatchMemoryQuote q,BnGenerateBatchMemoryTicket *out) {
    if(!l||!l->initialized||l->shared_preparing||!out||!memory_quote_valid(q)||
       !memory_fits(l->shared,q.peak,l->limit))return -1;
    int slot=-1;
    for(int i=0;i<BN_GENERATE_BATCH_MAX_REQUESTS;i++)if(!l->slots[i].state&&l->slots[i].generation!=UINT64_MAX){slot=i;break;}
    if(slot<0){int future=0;for(int i=0;i<BN_GENERATE_BATCH_MAX_REQUESTS;i++)future|=l->slots[i].state&&l->slots[i].generation<UINT64_MAX;
        if(!future)return -1;}
    if(slot<0||!memory_fits(l->used,q.peak,l->limit)){if(l->busy_returns!=UINT64_MAX)l->busy_returns++;return 1;}
    BnGenerateBatchMemorySlot *s=l->slots+slot;s->quote=q;s->state=1;s->generation++;
    memory_add(&l->used,q.peak);memory_peak(l);if(l->reservations!=UINT64_MAX)l->reservations++;
    *out=(BnGenerateBatchMemoryTicket){slot,s->generation,l};return 0;
}
int bn_generate_batch_memory_commit(BnGenerateBatchMemoryLedger *l,BnGenerateBatchMemoryTicket t) {
    BnGenerateBatchMemorySlot *s=memory_ticket(l,t);if(!s||s->state!=1)return -1;
    memory_sub(&l->used,s->quote.peak);memory_add(&l->used,s->quote.retained);s->state=2;return 0;
}
int bn_generate_batch_memory_check_active(const BnGenerateBatchMemoryLedger *l,
    BnGenerateBatchMemoryQuote q) {
    if(!l||!l->initialized||l->shared_preparing||!memory_quote_valid(q)||
       !memory_fits(l->shared,q.peak,l->limit))return -1;
    BnGenerateBatchMemory active=l->shared;int count=0;
    for(int i=0;i<BN_GENERATE_BATCH_MAX_REQUESTS;i++) {
        const BnGenerateBatchMemorySlot *s=l->slots+i;
        if(s->state==1||s->state==2) {
            memory_add(&active,s->state==1?s->quote.peak:s->quote.retained);count++;
        }
    }
    return count==BN_GENERATE_BATCH_MAX_REQUESTS||!memory_fits(active,q.peak,l->limit)?1:0;
}
int bn_generate_batch_memory_check_shared(const BnGenerateBatchMemoryLedger *l,
    BnGenerateBatchMemoryQuote q) {
    return !l||!l->initialized||!memory_quote_valid(q)||
        !memory_fits(l->shared,q.peak,l->limit)?-1:0;
}
int bn_generate_batch_memory_idle(BnGenerateBatchMemoryLedger *l,BnGenerateBatchMemoryTicket t) {
    BnGenerateBatchMemorySlot *s=memory_ticket(l,t);if(!s||s->state!=2)return -1;s->state=3;return 0;
}
int bn_generate_batch_memory_reuse(BnGenerateBatchMemoryLedger *l,BnGenerateBatchMemoryTicket t,
    BnGenerateBatchMemoryTicket *out) {
    BnGenerateBatchMemorySlot *s=memory_ticket(l,t);if(!s||s->state!=3||!out||s->generation==UINT64_MAX)return -1;
    s->generation++;s->state=2;if(l->reuses!=UINT64_MAX)l->reuses++;
    *out=(BnGenerateBatchMemoryTicket){t.slot,s->generation,l};return 0;
}
int bn_generate_batch_memory_drop(BnGenerateBatchMemoryLedger *l,BnGenerateBatchMemoryTicket t) {
    BnGenerateBatchMemorySlot *s=memory_ticket(l,t);if(!s)return -1;
    memory_sub(&l->used,s->state==1?s->quote.peak:s->quote.retained);s->state=0;return 0;
}

int bn_generate_batch_memory_replace_idle(BnGenerateBatchMemoryLedger *l,
    BnGenerateBatchMemoryTicket t,BnGenerateBatchMemoryQuote q,
    BnGenerateBatchMemoryTicket *out) {
    BnGenerateBatchMemorySlot *s=memory_ticket(l,t);
    if(!s||s->state!=3||!out||s->generation==UINT64_MAX||!memory_quote_valid(q)||
       !memory_fits(l->shared,q.peak,l->limit))return -1;
    /* Old storage remains charged while the complete replacement is prepared;
     * this includes backends that allocate before releasing old workspace. */
    if(!memory_fits(l->used,q.peak,l->limit)) {
        if(l->busy_returns!=UINT64_MAX)l->busy_returns++;
        return 1;
    }
    memory_add(&q.peak,s->quote.retained);
    memory_sub(&l->used,s->quote.retained);memory_add(&l->used,q.peak);
    s->quote=q;s->state=1;s->generation++;memory_peak(l);
    if(l->reservations!=UINT64_MAX)l->reservations++;
    *out=(BnGenerateBatchMemoryTicket){t.slot,s->generation,l};return 0;
}
