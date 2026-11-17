#include "gpu_moe_cache.h"
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
typedef struct {size_t active,peak,created,destroyed;} Fixture;
typedef struct {Fixture *owner;int value;size_t bytes;} Buffer;
static void destroy(void*ctx,void*ptr){if(!ptr)return;Fixture*f=ctx;Buffer*b=ptr;assert(b->owner==f&&f->active>=b->bytes);f->active-=b->bytes;f->destroyed++;free(b);}
static void *make(Fixture*f,int value){Buffer*b=malloc(sizeof(*b));assert(b);*b=(Buffer){f,value,4};f->active+=4;if(f->active>f->peak)f->peak=f->active;f->created++;return b;}
static void publish(BnGPUMoEReservation*r,Fixture*f,int value,BnGPUMoELease*out){void*g=make(f,value),*u=make(f,value+1),*d=make(f,value+2);assert(!bn_gpu_moe_cache_commit(r,g,u,d,out));}
static void test_leases(void){Fixture f={0};BnGPUBackend gpu={0};gpu.ctx=&f;gpu.buffer_destroy=destroy;BnGPUMoECache*c=bn_gpu_moe_cache_create(24,12,2,&gpu);assert(c);BnGPUMoEReservation a={0},b={0},busy={0},snapshot;BnGPUMoELease x={0},y={0},z={0};
 assert(!bn_gpu_moe_cache_reserve(c,0,0,&a));snapshot=a;assert(bn_gpu_moe_cache_close(c)==BN_GPU_MOE_CACHE_BUSY);assert(bn_gpu_moe_cache_acquire(c,0,0,1,&x)==0&&!x.cache);assert(bn_gpu_moe_cache_reserve(c,0,0,&busy)==BN_GPU_MOE_CACHE_BUSY);publish(&a,&f,10,&x);assert(bn_gpu_moe_cache_abort(&snapshot)==-1);
 assert(!bn_gpu_moe_cache_reserve(c,0,1,&b));publish(&b,&f,20,&y);assert(f.active==24);memset(&busy,0xa5,sizeof(busy));snapshot=busy;assert(bn_gpu_moe_cache_reserve(c,0,2,&busy)==BN_GPU_MOE_CACHE_BUSY&&!memcmp(&busy,&snapshot,sizeof(busy)));assert(bn_gpu_moe_cache_close(c)==BN_GPU_MOE_CACHE_BUSY);
 assert(bn_gpu_moe_cache_acquire(c,0,0,0,&z)==1);assert(!bn_gpu_moe_cache_release(&z));assert(!bn_gpu_moe_cache_release(&y));assert(!bn_gpu_moe_cache_reserve(c,0,2,&b));assert(f.active==12&&((Buffer*)x.gate)->value==10);publish(&b,&f,30,&y);assert(f.active==24&&f.peak==24);
 assert(!bn_gpu_moe_cache_release(&x));assert(!bn_gpu_moe_cache_release(&x));assert(!bn_gpu_moe_cache_release(&y));
 /* Prefill acquisition must leave LRU unchanged: oldest0 remains victim. */
 int head=c->lru_head,tail=c->lru_tail;assert(bn_gpu_moe_cache_acquire(c,0,0,0,&z)==1);assert(c->lru_head==head&&c->lru_tail==tail);assert(!bn_gpu_moe_cache_release(&z));assert(!bn_gpu_moe_cache_reserve(c,0,3,&a));
 BnGPUMoELease untouched={0},save=untouched;assert(bn_gpu_moe_cache_acquire(c,0,0,0,&untouched)==0&&!memcmp(&save,&untouched,sizeof(save)));
 /* Simulated upload failure: destroy partial caller-owned payload before abort. */
 void*partial=make(&f,77);assert(bn_gpu_moe_cache_commit(&a,NULL,NULL,NULL,&z)==-1);destroy(&f,partial);snapshot=a;assert(!bn_gpu_moe_cache_abort(&a));assert(bn_gpu_moe_cache_abort(&snapshot)==-1);
 assert(!bn_gpu_moe_cache_reserve(c,1,4,&a));assert(bn_gpu_moe_cache_commit(&snapshot,(void*)1,(void*)2,(void*)3,&z)==-1);publish(&a,&f,40,&z);BnGPUMoELease stale=z;assert(!bn_gpu_moe_cache_release(&z));assert(!bn_gpu_moe_cache_reserve(c,1,5,&a));assert(!bn_gpu_moe_cache_abort(&a));/* may evict other unpinned entry */
 stale.generation--;assert(bn_gpu_moe_cache_release(&stale)==-1);
 BnGPUMoECacheStats stats={0};assert(!bn_gpu_moe_cache_stats(c,&stats));assert(stats.capacity_payload==24&&stats.charged_payload<=24&&stats.pending==0&&stats.leases==0&&stats.aborts==2);
 assert(!bn_gpu_moe_cache_close(c));assert(!f.active&&f.created==f.destroyed);
}
static void test_churn(void){Fixture f={0};BnGPUBackend gpu={0};gpu.ctx=&f;gpu.buffer_destroy=destroy;BnGPUMoECache*c=bn_gpu_moe_cache_create(48,12,4,&gpu);assert(c);BnGPUMoEReservation r={0};BnGPUMoELease anchor={0},lease={0};assert(!bn_gpu_moe_cache_reserve(c,7,999,&r));publish(&r,&f,1234,&anchor);
 for(int i=0;i<5000;i++){int layer=i%3,key=(i*17)%97;int hit=bn_gpu_moe_cache_acquire(c,layer,key,i%2,&lease);assert(hit>=0);if(hit){assert(((Buffer*)lease.gate)->value==layer*1000+key*3);assert(!bn_gpu_moe_cache_release(&lease));}else{assert(!bn_gpu_moe_cache_reserve(c,layer,key,&r));if(i%7==0){void*p=make(&f,0);destroy(&f,p);assert(!bn_gpu_moe_cache_abort(&r));}else{publish(&r,&f,layer*1000+key*3,&lease);assert(!bn_gpu_moe_cache_release(&lease));}}
  BnGPUMoECacheStats s;assert(!bn_gpu_moe_cache_stats(c,&s));assert(s.pending==0&&s.leases==1&&s.pinned_entries==1&&s.charged_payload<=48&&f.active==s.published*12&&f.peak<=48);assert(((Buffer*)anchor.gate)->value==1234);
  for(int j=0;j<c->n_slots;j++)if(c->entries[j].published){void*g=NULL,*u=NULL,*d=NULL;assert(bn_gpu_moe_cache_lookup_prefill(c,c->entries[j].layer,c->entries[j].expert_idx,&g,&u,&d));assert(g==c->entries[j].gate_gpu&&u==c->entries[j].up_gpu&&d==c->entries[j].down_gpu);}
 }
 assert(!bn_gpu_moe_cache_release(&anchor));assert(!bn_gpu_moe_cache_close(c));assert(!f.active&&f.created==f.destroyed);
}
static void test_bounds(void){Fixture f={0};BnGPUBackend gpu={0};gpu.ctx=&f;gpu.buffer_destroy=destroy;assert(!bn_gpu_moe_cache_create(11,12,1,&gpu));assert(!bn_gpu_moe_cache_create(24,12,-1,&gpu));BnGPUMoECache*c=bn_gpu_moe_cache_create(SIZE_MAX,12,2,&gpu);assert(c&&c->n_slots==2);BnGPUMoEReservation r={0};assert(bn_gpu_moe_cache_reserve(c,-1,0,&r)==-1);assert(bn_gpu_moe_cache_reserve(c,0,-1,&r)==-1);c->entries[0].generation=UINT64_MAX;assert(!bn_gpu_moe_cache_reserve(c,0,0,&r)&&r.slot==1);assert(!bn_gpu_moe_cache_abort(&r));c->entries[1].generation=UINT64_MAX;assert(bn_gpu_moe_cache_reserve(c,0,1,&r)==BN_GPU_MOE_CACHE_BUSY);assert(!bn_gpu_moe_cache_close(c));}
static void test_recycled_reservations(void) {
 Fixture f={0};BnGPUBackend gpu={.ctx=&f,.buffer_destroy=destroy};
 BnGPUMoECache*c=bn_gpu_moe_cache_create(24,12,2,&gpu);assert(c);
 BnGPUMoEReservation r={0},stale={0};BnGPUMoELease anchor={0},lease={0},miss={0};
 assert(!bn_gpu_moe_cache_reserve_reuse(c,7,99,&r));publish(&r,&f,900,&anchor);
 assert(!bn_gpu_moe_cache_reserve_reuse(c,0,0,&r));publish(&r,&f,0,&lease);
 assert(!bn_gpu_moe_cache_release(&lease));
 for(int i=1;i<=1000;i++) {
  assert(!bn_gpu_moe_cache_reserve_reuse(c,i%3,i,&r));stale=r;
  void*g=NULL,*u=NULL,*d=NULL;
  int retained=bn_gpu_moe_cache_reserved_buffers(&r,&g,&u,&d);assert(retained>=0);
  assert(!bn_gpu_moe_cache_acquire(c,i%3,i,0,&miss)&&!miss.cache);
  BnGPUMoEReservation busy={0};assert(bn_gpu_moe_cache_reserve_reuse(c,i%3,i,&busy)==BN_GPU_MOE_CACHE_BUSY);
  assert(bn_gpu_moe_cache_close(c)==BN_GPU_MOE_CACHE_BUSY);
  if(retained) {
   assert(f.active==24&&((Buffer*)anchor.gate)->value==900);
   assert(bn_gpu_moe_cache_commit(&r,(void*)1,(void*)2,(void*)3,&lease)==-1);
   ((Buffer*)g)->value=i;((Buffer*)u)->value=i+1;((Buffer*)d)->value=i+2;
  }
  if(i%7==0) {
   assert(!bn_gpu_moe_cache_abort(&r)); /* Includes poisoned retained payload. */
  } else if(i%11==0) {
   assert(!bn_gpu_moe_cache_discard_reserved_buffers(&r));
   assert(!bn_gpu_moe_cache_discard_reserved_buffers(&r));
   assert(f.active==12);publish(&r,&f,i,&lease);assert(!bn_gpu_moe_cache_release(&lease));
  } else {
   if(retained)assert(!bn_gpu_moe_cache_commit(&r,g,u,d,&lease));
   else publish(&r,&f,i,&lease);
   assert(((Buffer*)lease.gate)->value==i);assert(!bn_gpu_moe_cache_release(&lease));
  }
  assert(bn_gpu_moe_cache_abort(&stale)==-1);
  assert(bn_gpu_moe_cache_reserved_buffers(&stale,&g,&u,&d)==-1);
  assert(bn_gpu_moe_cache_discard_reserved_buffers(&stale)==-1);
  BnGPUMoECacheStats stats;assert(!bn_gpu_moe_cache_stats(c,&stats));
  assert(!stats.pending&&stats.leases==1&&stats.charged_payload==f.active&&f.peak==24);
  assert(((Buffer*)anchor.gate)->value==900);
 }
 assert(!bn_gpu_moe_cache_release(&anchor));assert(!bn_gpu_moe_cache_close(c));
 assert(!f.active&&f.created==f.destroyed);
 puts("Recycled LRU reservations: pinned isolation, pending ownership, poison, discard, stale tickets PASSED");
}
int main(void){test_leases();test_churn();test_bounds();test_recycled_reservations();puts("GPU expert reservations, asynchronous leases, bounded payload and LRU PASSED");return 0;}
