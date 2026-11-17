/* Model-free ownership, budget and failed-reuse checks for the dense KV pool. */
#include "../src/transformer/bf16_internal.h"
#include "model_internal.h"
#include "session.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static BnGPUBackend gpu;
static int created,freed,prepared,restarted,fail_create,fail_prepare,fail_restart;
static int expected_capacity=128;
static int resized,fail_resize,last_resize;
BnGPUBackend *bn_model_gpu(const BnModel *model){(void)model;return &gpu;}
BnBackendSession *bn_session_backend(const BnSession *s){return (BnBackendSession *)(uintptr_t)s;}
BnSession *bn_session_create_backend(const BnModel *model,BnAllocator *allocator) {
    (void)model;assert(!allocator);if(fail_create)return NULL;
    created++;return calloc(1,sizeof(BnSession));
}
void bn_session_free(BnSession *s,BnAllocator *allocator){assert(!allocator);if(s){freed++;free(s);}}
int bn_transformer_bf16_request_init_with_policy(const BnModel *model,BnGPUBackend *backend,
    BnBackendSession *session,int max_tokens,int capacity,void *rotary,int rotary_capacity,
    const BnTransformerBF16Policy *policy) {
    assert(model&&backend==&gpu&&session&&max_tokens==4&&capacity==expected_capacity&&!rotary&&!rotary_capacity&&policy&&policy->arithmetic==BN_TRANSFORMER_BF16_ARITHMETIC_EAGER);
    prepared++;return fail_prepare?-1:0;
}
int bn_transformer_bf16_session_restart(const BnModel *model,BnSession *s) {
    assert(model&&s);restarted++;if(fail_restart)return -1;s->pos=0;return 0;
}
int bn_transformer_bf16_position(const BnBackendSession *s){return ((const BnSession *)(uintptr_t)s)->pos;}
int bn_transformer_bf16_resize_idle_kv(const BnModel *model,BnSession *s,int capacity) {
    assert(model&&s&&!s->pos);resized++;last_resize=capacity;return fail_resize?-1:0;
}
static int resize_capability(void *r,BnGPUResidentView *v,size_t n){(void)r;(void)v;(void)n;assert(0);return -1;}

static void check_slots(BnTransformerBF16Pool *pool,size_t unit) {
    BnTransformerBF16PoolInfo info;assert(!bn_transformer_bf16_pool_info(pool,&info));
    size_t bytes=0;int slots=0,leased=0;
    for(int i=0;i<16;i++) {
        bytes+=(size_t)info.slot_capacity[i]*unit;slots+=info.slot_capacity[i]!=0;
        leased+=info.slot_leased[i];assert(!info.slot_leased[i]||info.slot_capacity[i]);
    }
    assert(bytes==info.reserved_kv_bytes&&bytes<=info.kv_budget_bytes);
    assert(slots==info.allocated_slots&&leased==info.leased_slots);
}

static void unchanged(const BnTransformerBF16PoolInfo *a,const BnTransformerBF16PoolInfo *b) {
    assert(a->kv_budget_bytes==b->kv_budget_bytes&&a->kv_bytes_per_slot==b->kv_bytes_per_slot&&
        a->reserved_kv_bytes==b->reserved_kv_bytes&&a->peak_kv_bytes==b->peak_kv_bytes&&
        a->max_slots==b->max_slots&&a->allocated_slots==b->allocated_slots&&a->leased_slots==b->leased_slots&&
        a->created==b->created&&a->reused==b->reused&&a->discarded==b->discarded&&a->evicted==b->evicted&&a->resized==b->resized);
    assert(!memcmp(a->slot_capacity,b->slot_capacity,sizeof(a->slot_capacity)));
    assert(!memcmp(a->slot_leased,b->slot_leased,sizeof(a->slot_leased)));
}

static void sized_admission(BnModel *model) {
    const size_t unit=65536;BnTransformerBF16Pool pool={0};BnTransformerBF16Policy policy={0};
    assert(!bn_transformer_bf16_pool_init(&pool,model,&gpu,3,4,128,128*unit,&policy));
    BnSession *a,*b,*c,*d=(BnSession *)(uintptr_t)1;BnTransformerBF16PoolInfo info,before;
    for(int bad=-1;bad<=129;bad+=130)
        assert(bn_transformer_bf16_pool_acquire_capacity(&pool,bad,&d)==-1&&!d);
    assert(bn_transformer_bf16_pool_acquire_capacity(&pool,0,&d)==-1&&!d);
    expected_capacity=16;assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,9,&a));
    expected_capacity=32;assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,17,&b));
    expected_capacity=64;assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,33,&c));
    c->pos=23;check_slots(&pool,unit);
    assert(!bn_transformer_bf16_pool_info(&pool,&before)&&before.reserved_kv_bytes==112*unit);
    assert(bn_transformer_bf16_pool_acquire_capacity(&pool,1,&d)==1&&!d);
    assert(!bn_transformer_bf16_pool_info(&pool,&info));unchanged(&before,&info);
    a->pos=11;b->pos=19;
    assert(!bn_transformer_bf16_pool_release(&pool,a)&&!bn_transformer_bf16_pool_release(&pool,b));
    BnSession *small;
    assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,9,&small)&&small==a&&small->pos==0&&b->pos==19&&c->pos==23);
    assert(!bn_transformer_bf16_pool_release(&pool,small));
    /* One active64 lease plus idle16+32: reclaim both idle entries to fit64. */
    assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,33,&d)&&c->pos==23);
    check_slots(&pool,unit);assert(!bn_transformer_bf16_pool_info(&pool,&before));
    assert(before.evicted==2&&before.created==4&&before.reserved_kv_bytes==128*unit);
    assert(bn_transformer_bf16_pool_acquire(&pool,&a)==1&&!a);
    assert(!bn_transformer_bf16_pool_info(&pool,&info));unchanged(&before,&info);
    assert(!bn_transformer_bf16_pool_release(&pool,d));
    assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,33,&a)&&a==d&&a->pos==0&&c->pos==23);
    assert(!bn_transformer_bf16_pool_release(&pool,a)&&!bn_transformer_bf16_pool_release(&pool,c));
    expected_capacity=128;assert(!bn_transformer_bf16_pool_acquire(&pool,&a));
    check_slots(&pool,unit);assert(!bn_transformer_bf16_pool_info(&pool,&info));
    assert(info.evicted==4&&info.discarded==0&&info.allocated_slots==1&&info.peak_kv_bytes==128*unit);
    assert(!bn_transformer_bf16_pool_release(&pool,a));
    /* Oversized reuse keeps the actual allocation charged, not the request. */
    assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,1,&a));
    assert(!bn_transformer_bf16_pool_info(&pool,&info)&&info.reserved_kv_bytes==128*unit);
    assert(bn_transformer_bf16_pool_acquire_capacity(&pool,1,&d)==1&&!d);
    bn_transformer_bf16_pool_free(&pool);

    assert(!bn_transformer_bf16_pool_init(&pool,model,&gpu,3,4,128,128*unit,&policy));
    expected_capacity=64;assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,33,&a));
    expected_capacity=16;assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,9,&b));
    assert(!bn_transformer_bf16_pool_release(&pool,b));
    assert(!bn_transformer_bf16_pool_info(&pool,&before));
    assert(bn_transformer_bf16_pool_acquire_capacity(&pool,128,&d)==1&&!d);
    assert(!bn_transformer_bf16_pool_info(&pool,&info));unchanged(&before,&info);
    bn_transformer_bf16_pool_free(&pool);

    assert(!bn_transformer_bf16_pool_init(&pool,model,&gpu,2,4,127,128*unit,&policy));
    expected_capacity=4;assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,1,&a));
    assert(!bn_transformer_bf16_pool_release(&pool,a));
    expected_capacity=127;fail_prepare=1;
    assert(bn_transformer_bf16_pool_acquire_capacity(&pool,127,&a)==-1&&!a);fail_prepare=0;
    check_slots(&pool,unit);assert(!bn_transformer_bf16_pool_info(&pool,&info));
    assert(info.evicted==1&&!info.allocated_slots&&!info.reserved_kv_bytes);
    assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,127,&a));
    assert(!bn_transformer_bf16_pool_release(&pool,a));fail_restart=1;
    assert(bn_transformer_bf16_pool_acquire_capacity(&pool,1,&a)==-1&&!a);fail_restart=0;
    check_slots(&pool,unit);assert(!bn_transformer_bf16_pool_info(&pool,&info)&&info.discarded==1);
    bn_transformer_bf16_pool_free(&pool);expected_capacity=128;
}

static void resized_admission(BnModel *model) {
    const size_t unit=65536;BnTransformerBF16Pool pool={0};BnTransformerBF16Policy policy={0};
    BnSession *a,*b,*out;BnTransformerBF16PoolInfo info,before;
    gpu.resident_resize_discard=resize_capability;
    assert(!bn_transformer_bf16_pool_init(&pool,model,&gpu,3,4,128,128*unit,&policy));
    expected_capacity=16;assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,9,&a));
    a->pos=10;assert(!bn_transformer_bf16_pool_release(&pool,a));
    assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,33,&out)&&out==a&&last_resize==64&&resized==1);
    expected_capacity=16;assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,9,&b));
    assert(!bn_transformer_bf16_pool_release(&pool,b));
    assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,17,&out)&&out==b&&last_resize==32&&resized==2);
    assert(!bn_transformer_bf16_pool_info(&pool,&before));
    assert(bn_transformer_bf16_pool_acquire_capacity(&pool,128,&out)==1&&!out);
    assert(!bn_transformer_bf16_pool_info(&pool,&info));unchanged(&before,&info);
    assert(!bn_transformer_bf16_pool_release(&pool,a)&&!bn_transformer_bf16_pool_release(&pool,b));
    assert(!bn_transformer_bf16_pool_acquire(&pool,&out)&&out==a&&last_resize==128);
    check_slots(&pool,unit);assert(!bn_transformer_bf16_pool_info(&pool,&info));
    assert(info.created==2&&info.reused==3&&info.evicted==1&&info.resized==3&&info.reserved_kv_bytes==128*unit);
    bn_transformer_bf16_pool_free(&pool);
    assert(!bn_transformer_bf16_pool_init(&pool,model,&gpu,1,4,128,128*unit,&policy));
    assert(!bn_transformer_bf16_pool_acquire_capacity(&pool,9,&a));
    assert(!bn_transformer_bf16_pool_release(&pool,a));fail_resize=1;
    assert(bn_transformer_bf16_pool_acquire_capacity(&pool,33,&out)==-1&&!out);fail_resize=0;
    check_slots(&pool,unit);assert(!bn_transformer_bf16_pool_info(&pool,&info));
    assert(info.discarded==1&&!info.reserved_kv_bytes&&!info.resized&&!info.leased_slots);
    bn_transformer_bf16_pool_free(&pool);gpu.resident_resize_discard=NULL;expected_capacity=128;
}

int main(void) {
    BnModel model={0};BnLayerWeights layers[64]={0};model.weights.layers=layers;
    model.config.n_layers=64;model.config.seq_len=256;model.config.bf16_layout=BN_MODEL_BF16_LAYOUT_HYBRID_OFFSET_NORM;
    for(int i=0;i<64;i++)layers[i].block_kind=i%4==3?BN_LAYER_BLOCK_ATTENTION:BN_LAYER_BLOCK_SSM;
    BnTransformerBF16Policy policy={0};size_t per=128u*1024*2*2*16;
    BnTransformerBF16Pool storage={0},*p=&storage;
    assert(bn_transformer_bf16_pool_init(p,&model,&gpu,2,4,128,per-1,&policy)==-1&&!p->state);
    assert(bn_transformer_bf16_pool_init(p,&model,&gpu,17,4,128,per*2,&policy)==-1&&!p->state);
    assert(bn_transformer_bf16_pool_init(p,&model,&gpu,2,129,128,per*2,&policy)==-1&&!p->state);
    assert(!created&&!prepared);
    assert(!bn_transformer_bf16_pool_init(p,&model,&gpu,3,4,128,per*2+17,&policy));
    assert(bn_transformer_bf16_pool_init(p,&model,&gpu,3,4,128,per*2+17,&policy)==-1);policy.arithmetic=(BnTransformerBF16Arithmetic)99; /* Pool owns its original copy. */
    BnTransformerBF16PoolInfo info;assert(!bn_transformer_bf16_pool_info(p,&info));
    assert(info.kv_bytes_per_slot==per&&!info.reserved_kv_bytes&&!info.allocated_slots);
    BnSession *a=NULL,*b=NULL,*out=(BnSession *)(uintptr_t)1;
    assert(!bn_transformer_bf16_pool_acquire(p,&a)&&!bn_transformer_bf16_pool_acquire(p,&b)&&a!=b);
    assert(bn_transformer_bf16_pool_acquire(p,&out)==1&&!out&&created==2&&prepared==2&&!restarted);
    assert(!bn_transformer_bf16_pool_info(p,&info)&&info.reserved_kv_bytes==2*per&&info.leased_slots==2);
    BnSession foreign={0};assert(bn_transformer_bf16_pool_release(p,&foreign)==-1);
    a->pos=73;assert(!bn_transformer_bf16_pool_release(p,a));
    assert(bn_transformer_bf16_pool_release(p,a)==-1);
    assert(!bn_transformer_bf16_pool_info(p,&info)&&info.reserved_kv_bytes==2*per&&info.leased_slots==1);
    assert(!bn_transformer_bf16_pool_acquire(p,&out)&&out==a&&out->pos==0&&restarted==1&&created==2);
    a->pos=-1;assert(!bn_transformer_bf16_pool_release(p,a)&&freed==1);
    assert(!bn_transformer_bf16_pool_info(p,&info)&&info.reserved_kv_bytes==per&&info.discarded==1);
    fail_create=1;assert(bn_transformer_bf16_pool_acquire(p,&out)==-1&&!out);fail_create=0;
    fail_prepare=1;assert(bn_transformer_bf16_pool_acquire(p,&out)==-1&&!out&&freed==2);fail_prepare=0;
    assert(!bn_transformer_bf16_pool_info(p,&info)&&info.reserved_kv_bytes==per&&info.allocated_slots==1);
    assert(!bn_transformer_bf16_pool_release(p,b));fail_restart=1;
    assert(bn_transformer_bf16_pool_acquire(p,&out)==-1&&!out&&freed==3);fail_restart=0;
    assert(!bn_transformer_bf16_pool_info(p,&info)&&!info.reserved_kv_bytes&&!info.allocated_slots&&info.peak_kv_bytes==2*per);
    assert(!bn_transformer_bf16_pool_acquire(p,&out));
    bn_transformer_bf16_pool_free(p);assert(freed==created&&!p->state);
    bn_transformer_bf16_pool_free(p);assert(freed==created);
    assert(bn_transformer_bf16_pool_info(p,&info)==-1);
    assert(!bn_transformer_bf16_pool_init(p,&model,&gpu,1,4,128,per,&(BnTransformerBF16Policy){0}));
    bn_transformer_bf16_pool_free(p);
    bn_transformer_bf16_pool_free(NULL);assert(bn_transformer_bf16_pool_acquire(NULL,&out)==-1&&!out);
    assert(bn_transformer_bf16_pool_info(NULL,&info)==-1);
    sized_admission(&model);assert(freed==created);
    resized_admission(&model);assert(freed==created);
    puts("BF16 KV pool ownership/budget PASSED");return 0;
}
