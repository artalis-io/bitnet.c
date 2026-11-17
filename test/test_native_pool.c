#include "model.h"
#include "session.h"
#include "transformer_internal.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int creates,frees,restarts,fail_create,fail_prepare,fail_restart;
int bn_transformer_trellis_request_plan(const BnModel *m,const BnTransformerTrellisRequestConfig *c,BnTransformerTrellisRequestPlan *out) {
    if(!m||!c||!out||c->max_tokens<1||c->max_tokens>16||c->maxseq<256||c->maxseq>1024||c->maxseq%256||c->max_tokens>c->maxseq)return -1;
    *out=(BnTransformerTrellisRequestPlan){(size_t)c->maxseq*2+(size_t)c->max_tokens*10,
        (size_t)c->maxseq+(size_t)c->max_tokens+64,(size_t)c->max_tokens*3,(size_t)c->maxseq*2,64};return 0;
}
BnSession *bn_session_create_backend(const BnModel *m,BnAllocator *a) {
    assert(m&&!a);creates++;if(fail_create)return NULL;return calloc(1,sizeof(BnSession));
}
void bn_session_free(BnSession *s,BnAllocator *a) {assert(!a);if(s){frees++;free(s);}}
int bn_transformer_trellis_prepare_request(const BnModel *m,BnSession *s,const BnTransformerTrellisRequestConfig *c) {
    assert(m&&s&&c);return fail_prepare?-1:0;
}
int bn_transformer_trellis_session_restart(const BnModel *m,BnSession *s) {
    assert(m&&s);restarts++;if(fail_restart)return -1;s->pos=0;return 0;
}
#include "../src/transformer/trellis_pool.h"
static BnModel model;
static BnTransformerTrellisRequestConfig ceiling={16,1024,32,0,0};
static void start(BnTransformerTrellisPool *pool,size_t device,size_t host,size_t pinned,int slots) {
    int old=creates;
    assert(!bn_transformer_trellis_pool_init(pool,&model,&ceiling,slots,device,host,pinned));
    assert(old==creates);
}
static void info(BnTransformerTrellisPool *pool,int allocated,int leased,size_t device,size_t host,size_t pinned) {
    BnTransformerTrellisPoolInfo p;assert(!bn_transformer_trellis_pool_info(pool,&p));
    assert(p.allocated_slots==allocated&&p.leased_slots==leased&&p.reserved_device==device&&p.reserved_host==host&&p.reserved_pinned==pinned);
    assert(p.reserved_device<=p.device_budget&&p.reserved_host<=p.host_budget&&p.reserved_pinned<=p.pinned_budget);
}
int main(void) {
    BnTransformerTrellisPool pool={0};BnSession *a,*b,*c;BnTransformerTrellisPoolInfo p;
    BnTransformerTrellisRequestConfig x=ceiling;x.max_tokens=4;x.maxseq=256;BnTransformerTrellisRequestPlan small,large;
    assert(!bn_transformer_trellis_request_plan(&model,&x,&small));x.max_tokens=8;x.maxseq=512;
    assert(!bn_transformer_trellis_request_plan(&model,&x,&large));
    start(&pool,10000,10000,10000,2);
    assert(!bn_transformer_trellis_pool_acquire(&pool,4,20,&a));a->pos=20;
    info(&pool,1,1,small.device_bytes,small.host_bytes,small.pinned_bytes);
    assert(!bn_transformer_trellis_pool_release(&pool,a,0)&&a->pos==0);
    info(&pool,1,0,small.device_bytes,small.host_bytes,small.pinned_bytes);
    int count=creates;assert(!bn_transformer_trellis_pool_acquire(&pool,4,200,&b)&&b==a&&creates==count);
    assert(!bn_transformer_trellis_pool_acquire(&pool,8,300,&c)&&c!=b);
    assert(bn_transformer_trellis_pool_acquire(&pool,4,1,&a)==1&&!a&&creates==count+1);
    assert(!bn_transformer_trellis_pool_release(&pool,b,0));
    assert(bn_transformer_trellis_pool_release(&pool,b,0)==-1);
    assert(!bn_transformer_trellis_pool_release(&pool,c,1));
    assert(!bn_transformer_trellis_pool_info(&pool,&p)&&p.created==2&&p.reused==1&&p.discarded==1&&p.released==3);
    BnTransformerTrellisPool copy=pool;assert(bn_transformer_trellis_pool_info(&copy,&p)==-1);
    bn_transformer_trellis_pool_free(&copy);assert(pool.state);bn_transformer_trellis_pool_free(&pool);assert(!pool.state);
    /* Each budget independently blocks live growth and evicts only idle data. */
    for(int dimension=0;dimension<3;dimension++) {
        size_t d=dimension==0?large.device_bytes:10000,h=dimension==1?large.host_bytes:10000,n=dimension==2?large.pinned_bytes:10000;
        start(&pool,d,h,n,2);assert(!bn_transformer_trellis_pool_acquire(&pool,4,1,&a));
        int old=frees;assert(bn_transformer_trellis_pool_acquire(&pool,8,300,&b)==1&&!b&&frees==old);
        info(&pool,1,1,small.device_bytes,small.host_bytes,small.pinned_bytes);
        assert(!bn_transformer_trellis_pool_release(&pool,a,0));
        assert(!bn_transformer_trellis_pool_acquire(&pool,8,300,&b)&&frees==old+1);
        info(&pool,1,1,large.device_bytes,large.host_bytes,large.pinned_bytes);
        assert(!bn_transformer_trellis_pool_info(&pool,&p)&&p.evicted==1);
        assert(!bn_transformer_trellis_pool_release(&pool,b,0));bn_transformer_trellis_pool_free(&pool);
    }
    /* A slot-count conflict also keeps the active lease untouched. */
    start(&pool,10000,10000,10000,1);assert(!bn_transformer_trellis_pool_acquire(&pool,4,1,&a));
    count=frees;assert(bn_transformer_trellis_pool_acquire(&pool,8,300,&b)==1&&!b&&frees==count);
    assert(!bn_transformer_trellis_pool_release(&pool,a,0));
    assert(!bn_transformer_trellis_pool_acquire(&pool,8,300,&b)&&frees==count+1);
    fail_restart=1;assert(!bn_transformer_trellis_pool_release(&pool,b,0));fail_restart=0;
    assert(!bn_transformer_trellis_pool_info(&pool,&p)&&p.restart_failures==1&&p.discarded==1&&!p.reserved_device&&!p.allocated_slots);
    bn_transformer_trellis_pool_free(&pool);
    for(int failure=0;failure<2;failure++) {
        start(&pool,10000,10000,10000,2);fail_create=failure==0;fail_prepare=failure==1;count=frees;
        assert(bn_transformer_trellis_pool_acquire(&pool,4,1,&a)==-1&&!a);
        info(&pool,0,0,0,0,0);assert(frees==count+(failure==1));fail_create=fail_prepare=0;
        assert(!bn_transformer_trellis_pool_acquire(&pool,4,1,&a));
        assert(!bn_transformer_trellis_pool_release(&pool,a,1));bn_transformer_trellis_pool_free(&pool);
    }
    start(&pool,small.device_bytes-1,10000,10000,2);count=creates;
    assert(bn_transformer_trellis_pool_acquire(&pool,4,1,&a)==-1&&!a&&creates==count);bn_transformer_trellis_pool_free(&pool);
    start(&pool,10000,10000,10000,2);
    assert(bn_transformer_trellis_pool_acquire(&pool,17,1,&a)==-1&&!a);
    assert(bn_transformer_trellis_pool_acquire(&pool,4,1025,&a)==-1&&!a);
    assert(bn_transformer_trellis_pool_acquire(&pool,4,0,&a)==-1&&!a);
    assert(!bn_transformer_trellis_pool_acquire(&pool,4,1024,&a));
    assert(!bn_transformer_trellis_pool_info(&pool,&p)&&p.capacity[0]==1024&&p.max_tokens[0]==4);
    BnSession foreign={0};assert(bn_transformer_trellis_pool_release(&pool,&foreign,1)==-1);
    assert(bn_transformer_trellis_pool_release(&pool,a,2)==-1);bn_transformer_trellis_pool_free(&pool);
    assert(creates==frees+1); /* One deliberate create allocation failure. */
    puts("Native pool: exact-shape reuse, retained idle charges, three budgets, BUSY no eviction, failure/discard/epoch cleanup PASSED");return 0;
}
