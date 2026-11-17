#include "generate_batch_plan.h"
#include "generate_batch_plan_reference.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

static void captured_staggered_schedule(void) {
    BnGenerateBatchCandidate requests[12];
    for(int i=0;i<12;i++)requests[i]=(BnGenerateBatchCandidate){i>=4&&i<8?5:129,0,0,64};
    const int arrivals[]={0,8,68};BnGenerateBatchLimits limits={8,4,64,256,8192,0};
    int sampled=0,total=0;
    for(size_t step=0;step<sizeof(generate_batch_trace)/sizeof(generate_batch_trace[0]);step++) {
        BnGenerateBatchCandidate active[16],before[16];int ids[16],count=0;
        for(int i=0;i<12;i++)if(arrivals[i/4]<=(int)step&&requests[i].generated<64) {
            ids[count]=i;active[count++]=requests[i];
        }
        memcpy(before,active,(size_t)count*sizeof(*active));
        BnGenerateBatchPlan plan;assert(bn_generate_batch_plan(&limits,active,count,&plan)==1);
        assert(!memcmp(before,active,(size_t)count*sizeof(*active)));
        const BnGenerateBatchTrace *ref=&generate_batch_trace[step];assert(plan.count==ref->count);
        int sum=0;
        for(int j=0;j<plan.count;j++) {
            BnGenerateBatchRow *row=&plan.rows[j];int id=ids[row->candidate];
            assert(id==ref->id[j]&&row->n_tokens==ref->tokens[j]&&row->position==ref->position[j]&&row->decode_role==ref->role[j]);
            BnGenerateBatchCandidate *r=&requests[id];
            assert(row->is_prefill==(r->prefilled<r->prompt_tokens));
            if(row->is_prefill)r->prefilled+=row->n_tokens;
            if(row->emit_token){r->generated++;sampled++;}
            sum+=row->n_tokens;
        }
        assert(sum==plan.total_tokens&&sum<=256&&count<=8);total+=sum;
    }
    for(int i=0;i<12;i++)assert(requests[i].generated==64&&requests[i].prefilled==requests[i].prompt_tokens);
    assert(sampled==768&&total==1808);
    puts("Generate batch planner: all 134 independent staggered cohorts PASSED");
}

static void limits_and_priority(void) {
    BnGenerateBatchLimits limits={8,4,64,256,8192,0};
    BnGenerateBatchCandidate c[]={{129,0,0,64},{129,129,1,64},{5,0,0,64},{129,129,1,64}};
    BnGenerateBatchPlan plan;assert(bn_generate_batch_plan(&limits,c,4,&plan)==1);
    assert(plan.count==4&&plan.total_tokens==71);
    assert(plan.rows[0].candidate==1&&plan.rows[0].decode_role);
    assert(plan.rows[1].candidate==3&&plan.rows[1].decode_role);
    assert(plan.rows[2].candidate==2&&plan.rows[2].n_tokens==5&&plan.rows[2].emit_token);
    assert(plan.rows[3].candidate==0&&plan.rows[3].n_tokens==64&&!plan.rows[3].emit_token);
    limits.token_budget=3;assert(bn_generate_batch_plan(&limits,c,4,&plan)==1);
    assert(plan.count==3&&plan.total_tokens==3);
    /* Equal query lengths retain admission order, so a generation row behind
     * the first logical prefill belongs to the recurrent prefill tail. */
    assert(plan.rows[0].candidate==0&&!plan.rows[0].decode_role);
    assert(plan.rows[1].candidate==1&&!plan.rows[1].decode_role);
    assert(plan.rows[2].candidate==3&&!plan.rows[2].decode_role);
    limits.token_budget=1;assert(bn_generate_batch_plan(&limits,c,4,&plan)==1);
    assert(plan.count==1&&plan.rows[0].candidate==1&&plan.total_tokens==1);
    limits.capacity=INT_MAX;limits.token_budget=256;
    BnGenerateBatchCandidate edge={INT_MAX-1,INT_MAX-1,1,2};
    assert(bn_generate_batch_plan(&limits,&edge,1,&plan)==1&&plan.rows[0].position==INT_MAX-1);
    edge.generated=2;assert(bn_generate_batch_plan(&limits,&edge,1,&plan)==0&&!plan.count);
    assert(bn_generate_batch_plan(&limits,NULL,0,&plan)==0&&!plan.count);
    BnGenerateBatchPlan unchanged;memset(&unchanged,0x5a,sizeof(unchanged));
    for(int bad=0;bad<13;bad++) {
        BnGenerateBatchLimits l=limits;BnGenerateBatchCandidate request=c[0];plan=unchanged;
        switch(bad) {
            case 0:l.max_requests=17;break;case 1:l.max_prefills=0;break;
            case 2:l.max_chunk=0;break;case 3:l.token_budget=BN_GENERATE_BATCH_MAX_TOKENS+1;break;
            case 4:l.capacity=128;break;case 5:request.prompt_tokens=0;break;
            case 6:request.prefilled=-1;break;case 7:request.prefilled=130;break;
            case 8:request.generated=1;break;case 9:request.max_outputs=0;break;
            case 10:request.generated=-1;break;
            case 11:request.prefilled=129;break;
            default:request.max_outputs=INT_MAX;break;
        }
        assert(bn_generate_batch_plan(&l,&request,1,&plan)==-1);
        assert(!memcmp(&plan,&unchanged,sizeof(plan)));
    }
    assert(bn_generate_batch_plan(NULL,c,4,&plan)==-1);
    assert(bn_generate_batch_plan(&limits,NULL,1,&plan)==-1);
    assert(bn_generate_batch_plan(&limits,c,17,&plan)==-1);
}
static void mixed_budget(void) {
    BnGenerateBatchLimits limits={8,4,64,256,8192,8};
    BnGenerateBatchCandidate c[8];
    for(int i=0;i<8;i++)c[i]=(BnGenerateBatchCandidate){129,i<4?129:0,i<4?1:0,64};
    BnGenerateBatchPlan plan;
    /* Long prefills make progress on every mixed step while all decodes run. */
    for(int step=0;step<3;step++) {
        assert(bn_generate_batch_plan(&limits,c,8,&plan)==1);
        assert(plan.count==8&&plan.total_tokens==8);
        for(int j=0;j<8;j++) {
            BnGenerateBatchRow row=plan.rows[j];assert(row.candidate==j&&row.n_tokens==1);
            assert(row.is_prefill==(j>=4)&&row.decode_role==(j<4));
            if(row.is_prefill)c[j].prefilled++;
            else c[j].generated++;
        }
    }
    /* Completed decodes no longer reduce the bulk prefill budget. */
    for(int i=0;i<4;i++)c[i].generated=64;
    assert(bn_generate_batch_plan(&limits,c,8,&plan)==1);
    assert(plan.count==4&&plan.total_tokens==256);
    for(int i=0;i<4;i++)assert(plan.rows[i].n_tokens==64);
    /* Pure decodes fit even when every request is active. */
    for(int i=0;i<8;i++)c[i]=(BnGenerateBatchCandidate){129,129,1,64};
    assert(bn_generate_batch_plan(&limits,c,8,&plan)==1&&plan.count==8&&plan.total_tokens==8);
    BnGenerateBatchPlan before=plan;
    const int invalid[]={-1,1,7,257};
    for(size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++) {
        limits.mixed_token_budget=invalid[i];
        assert(bn_generate_batch_plan(&limits,c,8,&plan)==-1);
        assert(!memcmp(&plan,&before,sizeof(plan)));
    }
}
static void reclaimed_quota(void) {
    BnGenerateBatchLimits limits={8,4,64,256,8192,256};
    BnGenerateBatchCandidate c[8];
    for(int i=0;i<8;i++)c[i]=(BnGenerateBatchCandidate){i<4?129:i<6?5:129,i<4?129:0,i<4?1:0,192};
    BnGenerateBatchPlan fixed,reclaim,before;
    assert(bn_generate_batch_plan(&limits,c,8,&fixed)==1&&fixed.total_tokens==140);
    assert(bn_generate_batch_plan_with_prefill_policy(&limits,c,8,BN_GENERATE_BATCH_PREFILL_RECLAIM,&reclaim)==1);
    assert(reclaim.count==8&&reclaim.total_tokens==142);
    for(int i=0;i<8;i++) {
        BnGenerateBatchRow row=reclaim.rows[i];
        assert(row.candidate==i&&row.n_tokens==(i<4?1:i<6?5:64));
        assert(row.decode_role==(i<4)&&row.is_prefill==(i>=4));
    }
    /* Six decodes leave ten tokens: clip the short lane to two, grant eight. */
    for(int i=0;i<8;i++)c[i]=(BnGenerateBatchCandidate){i==6?2:129,i<6?129:0,i<6?1:0,192};
    limits.mixed_token_budget=16;
    assert(bn_generate_batch_plan_with_prefill_policy(&limits,c,8,BN_GENERATE_BATCH_PREFILL_RECLAIM,&reclaim)==1);
    assert(reclaim.total_tokens==16&&reclaim.rows[6].n_tokens==2&&reclaim.rows[7].n_tokens==8);
    before=reclaim;
    assert(bn_generate_batch_plan_with_prefill_policy(&limits,c,8,(BnGenerateBatchPrefillPolicy)2,&reclaim)==-1);
    assert(!memcmp(&before,&reclaim,sizeof(before)));
    /* Independent exhaustive quota enumeration; include small budgets, clipped
     * lanes, nonzero positions, completed candidates and reordered role ties. */
    unsigned seed=20261111;
    for(int trial=0;trial<10000;trial++) {
        seed=seed*1664525u+1013904223u;int count=1+(int)(seed%16u);
        limits=(BnGenerateBatchLimits){count,1+(int)((seed>>8)%(unsigned)count),1+(int)((seed>>16)%97u),1+(int)((seed>>24)%128u),8192,0};
        BnGenerateBatchCandidate items[16];int decodes=0,pending[16],np=0;
        for(int i=0;i<count;i++) {
            seed=seed*1664525u+1013904223u;int n=1+(int)(seed%257u),done=(int)((seed>>12)%3u);
            int pos=done?n:(int)((seed>>16)%(unsigned)n);
            items[i]=(BnGenerateBatchCandidate){n,pos,done?done:0,2};
            if(done==1&&decodes<limits.token_budget)decodes++;
        }
        int budget=limits.token_budget-decodes;
        for(int i=0;i<count&&np<limits.max_prefills&&decodes+np<count&&np<budget;i++)
            if(items[i].prefilled<items[i].prompt_tokens)pending[np++]=i;
        int quota=0;
        for(int q=1;q<=limits.max_chunk;q++) {
            int used=0;for(int j=0;j<np;j++) {
                int i=pending[j],n=items[i].prompt_tokens-items[i].prefilled;used+=n<q?n:q;
            }
            if(used<=budget)quota=q;
        }
        int rc=bn_generate_batch_plan_with_prefill_policy(&limits,items,count,BN_GENERATE_BATCH_PREFILL_RECLAIM,&reclaim);
        assert(rc==(decodes+np?1:0)&&reclaim.count==decodes+np);
        int total=0,tail=0,last_n=0,last_id=-1,seen=0;
        for(int j=0;j<reclaim.count;j++) {
            BnGenerateBatchRow row=reclaim.rows[j];BnGenerateBatchCandidate item=items[row.candidate];
            assert(!(seen&(1<<row.candidate)));seen|=1<<row.candidate;
            assert(row.n_tokens>last_n||(row.n_tokens==last_n&&row.candidate>last_id));
            last_n=row.n_tokens;last_id=row.candidate;
            if(row.is_prefill) {
                int n=item.prompt_tokens-item.prefilled;
                assert(row.n_tokens==(n<quota?n:quota)&&row.position==item.prefilled);
                int selected=0;for(int k=0;k<np;k++)selected|=pending[k]==row.candidate;assert(selected);
                assert(row.emit_token==(row.n_tokens==n));tail=1;
            } else assert(row.n_tokens==1&&item.prefilled==item.prompt_tokens&&item.generated==1);
            assert(row.decode_role==!tail);total+=row.n_tokens;
        }
        assert(total==reclaim.total_tokens&&total<=limits.token_budget);
    }
    puts("Reclaimed quota: clipping, atomic rejection and 10000 exhaustive host cases PASSED");
}
static void large_aggregate(void) {
    BnGenerateBatchLimits l={16,16,2048,32768,2304,0};BnGenerateBatchCandidate c[16];
    for(int i=0;i<16;i++)c[i]=(BnGenerateBatchCandidate){i&1?1026:2050,0,0,5};
    BnGenerateBatchPlan p;assert(bn_generate_batch_plan(&l,c,16,&p)==1&&p.count==16&&p.total_tokens==24592);
    for(int i=0;i<16;i++) {
        int id=i<8?2*i+1:2*(i-8);BnGenerateBatchRow r=p.rows[i];
        assert(r.candidate==id&&r.n_tokens==(i<8?1026:2048)&&!r.position&&r.is_prefill&&!r.decode_role);
        assert(r.emit_token==(i<8));
    }
    for(int i=0;i<4;i++){c[i].prefilled=c[i].prompt_tokens;c[i].generated=1;}
    assert(bn_generate_batch_plan(&l,c,16,&p)==1&&p.count==16&&p.total_tokens==18448);
    for(int i=0;i<4;i++)assert(p.rows[i].candidate==i&&p.rows[i].n_tokens==1&&p.rows[i].decode_role);
    l=(BnGenerateBatchLimits){16,16,BN_GENERATE_BATCH_MAX_CHUNK_TOKENS,BN_GENERATE_BATCH_MAX_TOKENS,8192,0};
    for(int i=0;i<16;i++)c[i]=(BnGenerateBatchCandidate){8192,0,0,1};
    for(int policy=0;policy<=1;policy++) {
        assert(bn_generate_batch_plan_with_prefill_policy(&l,c,16,(BnGenerateBatchPrefillPolicy)policy,&p)==1);
        assert(p.count==16&&p.total_tokens==131072);
        for(int i=0;i<16;i++)assert(p.rows[i].candidate==i&&p.rows[i].n_tokens==8192&&p.rows[i].emit_token);
    }
    l.token_budget=8193;assert(bn_generate_batch_plan(&l,c,1,&p)==1&&p.total_tokens==8192);
    BnGenerateBatchPlan before=p;l.max_chunk=BN_GENERATE_BATCH_MAX_CHUNK_TOKENS+1;
    assert(bn_generate_batch_plan(&l,c,1,&p)==-1&&!memcmp(&p,&before,sizeof(p)));
    l.max_chunk=8192;l.token_budget=BN_GENERATE_BATCH_MAX_TOKENS+1;
    assert(bn_generate_batch_plan(&l,c,1,&p)==-1&&!memcmp(&p,&before,sizeof(p)));
    puts("Large aggregate planner: mixed16 histories,32768 budget,131072 boundary, unchanged per-request cap PASSED");
}
static void memory_ledger(void) {
    BnGenerateBatchMemoryLedger l={0},foreign={0};
    BnGenerateBatchMemory cap={1000,2000};BnGenerateBatchMemoryQuote shared={{100,200},{150,300}};
    assert(!bn_generate_batch_memory_init(&l,cap,shared));assert(l.used.host_bytes==150&&l.used.device_bytes==300);
    BnGenerateBatchMemoryQuote q={{200,400},{300,600}};BnGenerateBatchMemoryTicket t[3],sentinel,bad;
    memset(&sentinel,0x5a,sizeof(sentinel));bad=sentinel;
    assert(bn_generate_batch_memory_reserve(&l,q,&bad)==-1&&!memcmp(&bad,&sentinel,sizeof(bad)));
    assert(!bn_generate_batch_memory_shared_commit(&l));assert(bn_generate_batch_memory_shared_commit(&l)==-1);
    for(int i=0;i<3;i++) {
        assert(!bn_generate_batch_memory_reserve(&l,q,t+i));assert(!bn_generate_batch_memory_commit(&l,t[i]));
    }
    assert(l.used.host_bytes==700&&l.used.device_bytes==1400&&l.peak.host_bytes==800&&l.peak.device_bytes==1600);
    assert(!bn_generate_batch_memory_idle(&l,t[0]));assert(l.used.host_bytes==700&&l.used.device_bytes==1400);
    /* Retained idle storage still occupies the budget. Peak must fit even
     * when the new request's eventual retained state would have fit. */
    BnGenerateBatchMemoryQuote larger={{100,100},{301,601}};bad=sentinel;
    assert(bn_generate_batch_memory_reserve(&l,larger,&bad)==1&&!memcmp(&bad,&sentinel,sizeof(bad)));
    assert(l.used.host_bytes==700&&l.used.device_bytes==1400&&l.busy_returns==1);
    BnGenerateBatchMemoryTicket renewed;
    assert(!bn_generate_batch_memory_reuse(&l,t[0],&renewed));assert(renewed.generation!=t[0].generation);
    assert(bn_generate_batch_memory_drop(&l,t[0])==-1);t[0]=renewed;
    assert(!bn_generate_batch_memory_init(&foreign,cap,shared)&&!bn_generate_batch_memory_shared_commit(&foreign));
    assert(bn_generate_batch_memory_drop(&foreign,t[0])==-1);
    for(int i=0;i<3;i++)assert(!bn_generate_batch_memory_drop(&l,t[i]));
    assert(l.used.host_bytes==100&&l.used.device_bytes==200);
    assert(!bn_generate_batch_memory_reserve(&l,q,t));assert(!bn_generate_batch_memory_drop(&l,t[0])); /* Allocation rollback. */
    assert(l.used.host_bytes==100&&l.used.device_bytes==200);
    BnGenerateBatchMemoryLedger before=l;bad=sentinel;
    BnGenerateBatchMemoryQuote impossible={{0,0},{901,1}};
    assert(bn_generate_batch_memory_reserve(&l,impossible,&bad)==-1&&!memcmp(&before,&l,sizeof(l))&&!memcmp(&bad,&sentinel,sizeof(bad)));
    impossible=(BnGenerateBatchMemoryQuote){{10,10},{9,20}};
    assert(bn_generate_batch_memory_reserve(&l,impossible,&bad)==-1&&!memcmp(&before,&l,sizeof(l)));
    /* Overflow-safe subtraction at the exact size_t boundary. */
    BnGenerateBatchMemoryLedger edge={0};BnGenerateBatchMemoryQuote near={{SIZE_MAX-10,0},{SIZE_MAX-10,0}};
    assert(!bn_generate_batch_memory_init(&edge,(BnGenerateBatchMemory){SIZE_MAX,0},near));
    assert(!bn_generate_batch_memory_shared_commit(&edge));
    BnGenerateBatchMemoryQuote ten={{10,0},{10,0}};assert(!bn_generate_batch_memory_reserve(&edge,ten,t));
    assert(!bn_generate_batch_memory_commit(&edge,t[0])&&edge.used.host_bytes==SIZE_MAX);
    assert(bn_generate_batch_memory_reserve(&edge,ten,&bad)==1);
    assert(!bn_generate_batch_memory_drop(&edge,t[0]));
    assert(bn_generate_batch_memory_reserve(&edge,(BnGenerateBatchMemoryQuote){{11,0},{11,0}},&bad)==-1);
    BnGenerateBatchMemoryLedger slots={0};assert(!bn_generate_batch_memory_init(&slots,(BnGenerateBatchMemory){0},(BnGenerateBatchMemoryQuote){0}));
    assert(!bn_generate_batch_memory_shared_commit(&slots));BnGenerateBatchMemoryTicket tickets[16];
    for(int i=0;i<16;i++)assert(!bn_generate_batch_memory_reserve(&slots,(BnGenerateBatchMemoryQuote){0},tickets+i));
    assert(bn_generate_batch_memory_reserve(&slots,(BnGenerateBatchMemoryQuote){0},&bad)==1);
    for(int i=0;i<16;i++)assert(!bn_generate_batch_memory_drop(&slots,tickets[i]));
    for(int i=0;i<16;i++)slots.slots[i].generation=UINT64_MAX;
    assert(bn_generate_batch_memory_reserve(&slots,(BnGenerateBatchMemoryQuote){0},&bad)==-1);
    BnGenerateBatchMemoryLedger shrink={0};
    assert(!bn_generate_batch_memory_init(&shrink,cap,shared));BnGenerateBatchMemoryLedger saved=shrink;
    assert(bn_generate_batch_memory_shared_commit_as(&shrink,(BnGenerateBatchMemory){101,200})==-1&&!memcmp(&saved,&shrink,sizeof(shrink)));
    assert(!bn_generate_batch_memory_shared_commit_as(&shrink,(BnGenerateBatchMemory){50,100}));
    assert(shrink.used.host_bytes==50&&shrink.used.device_bytes==100&&shrink.peak.host_bytes==150&&shrink.peak.device_bytes==300);
    puts("Generate batch aggregate memory: shared setup, preparation peaks, retained idle slots, reuse epochs, foreign/stale rejection, rollback, pressure, overflow and slot exhaustion PASSED");
}

static void memory_replacement(void) {
    BnGenerateBatchMemoryLedger l={0},before;
    BnGenerateBatchMemoryQuote initial={{10,20},{20,30}},next={{10,20},{60,70}};
    assert(!bn_generate_batch_memory_init(&l,(BnGenerateBatchMemory){100,100},initial));
    before=l;
    assert(!bn_generate_batch_memory_check_shared(&l,(BnGenerateBatchMemoryQuote){{80,70},{90,80}}));
    assert(bn_generate_batch_memory_check_shared(&l,(BnGenerateBatchMemoryQuote){{91,0},{91,0}})==-1);
    assert(!memcmp(&l,&before,sizeof(l))); /* Future admission does not publish setup. */
    before=l;
    assert(bn_generate_batch_memory_shared_prepare_as(&l,(BnGenerateBatchMemoryQuote){{1,1},{101,1}})==-1);
    assert(!memcmp(&l,&before,sizeof(l)));
    assert(!bn_generate_batch_memory_shared_prepare_as(&l,next)&&l.used.host_bytes==60&&l.peak.device_bytes==70);
    assert(!bn_generate_batch_memory_shared_commit(&l));before=l;
    assert(bn_generate_batch_memory_shared_prepare_as(&l,initial)==-1&&!memcmp(&l,&before,sizeof(l)));
    BnGenerateBatchMemoryTicket old,renewed,sentinel;
    assert(!bn_generate_batch_memory_reserve(&l,(BnGenerateBatchMemoryQuote){{20,30},{40,50}},&old));
    before=l;
    assert(bn_generate_batch_memory_check_active(&l,(BnGenerateBatchMemoryQuote){{80,60},{80,60}})==1&&!memcmp(&l,&before,sizeof(l)));
    assert(!bn_generate_batch_memory_commit(&l,old)&&!bn_generate_batch_memory_idle(&l,old));
    before=l;
    assert(!bn_generate_batch_memory_check_active(&l,(BnGenerateBatchMemoryQuote){{80,60},{80,60}})&&!memcmp(&l,&before,sizeof(l)));
    assert(bn_generate_batch_memory_check_active(&l,(BnGenerateBatchMemoryQuote){{101,0},{101,0}})==-1&&!memcmp(&l,&before,sizeof(l)));
    memset(&renewed,0x5a,sizeof(renewed));sentinel=renewed;before=l;
    assert(bn_generate_batch_memory_replace_idle(&l,old,(BnGenerateBatchMemoryQuote){{80,60},{80,60}},&renewed)==1);
    assert(!memcmp(&renewed,&sentinel,sizeof(renewed)));before.busy_returns++;
    assert(!memcmp(&l,&before,sizeof(l)));
    assert(!bn_generate_batch_memory_replace_idle(&l,old,(BnGenerateBatchMemoryQuote){{40,20},{50,40}},&renewed));
    assert(l.used.host_bytes==80&&l.used.device_bytes==90&&renewed.generation==old.generation+1);
    assert(bn_generate_batch_memory_drop(&l,old)==-1);
    assert(!bn_generate_batch_memory_commit(&l,renewed)&&l.used.host_bytes==50&&l.used.device_bytes==40);
    assert(!bn_generate_batch_memory_idle(&l,renewed));old=renewed;
    assert(!bn_generate_batch_memory_replace_idle(&l,old,(BnGenerateBatchMemoryQuote){{30,30},{30,30}},&renewed));
    /* Preparation failure destroys the whole slot before releasing its combined reservation. */
    assert(!bn_generate_batch_memory_drop(&l,renewed)&&l.used.host_bytes==10&&l.used.device_bytes==20);
    puts("Memory staged shared setup and idle replacement: overlap, BUSY atomicity, ownership, commit/abort PASSED");
}

int main(void) {
    memory_replacement();
    memory_ledger();
    captured_staggered_schedule();limits_and_priority();mixed_budget();reclaimed_quota();large_aggregate();puts("Generate batch planner PASSED");return 0;
}
