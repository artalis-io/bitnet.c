#include "backend_model.h"
#include "trellis.h"
#include "platform.h"
#include <stdlib.h>
#include <string.h>

#define NONE SIZE_MAX
typedef struct { uint64_t row; size_t prev,next; } RowEntry;
typedef struct {
    const BnTrellisRowSource *source;
    const void *source_state;
    BnTrellisRowLayout layout;
    BnBackendRowCacheStats stats;
    size_t slots,batch_slots,head,tail;
    RowEntry *entries;
    size_t *hash,*batch_hash;
    unsigned char *rows,*staging,*scratch;
} RowCache;

static RowCache *get_cache(const BnBackendRowCache *cache) {
    RowCache *r=cache?cache->state:NULL;
    return r&&r->source->state==r->source_state&&
        bn_trellis_row_source_layout(r->source)?r:NULL;
}
static int cost(size_t base,size_t bytes,size_t capacity,size_t *total,size_t *slots) {
    if(!capacity||capacity>SIZE_MAX/2)return -1;
    size_t n=1;
    while(n<capacity*2){if(n>SIZE_MAX/2)return -1;n*=2;}
    if(bytes>SIZE_MAX-sizeof(RowEntry))return -1;
    size_t each=bytes+sizeof(RowEntry);
    if(capacity>(SIZE_MAX-base)/each)return -1;
    size_t used=base+capacity*each;
    if(n>(SIZE_MAX-used)/sizeof(size_t))return -1;
    *total=used+n*sizeof(size_t);*slots=n;return 0;
}
int bn_backend_model_row_cache_init(BnBackendRowCache *cache,
    const BnTrellisRowSource *source,size_t budget,size_t max_rows) {
    if(!cache||cache->state||!max_rows)return -1;
    const BnTrellisRowLayout *l=bn_trellis_row_source_layout(source);
    if(!l||!l->bundle_bytes||max_rows>SIZE_MAX/sizeof(uint64_t)||
        max_rows>SIZE_MAX/l->bundle_bytes-1)return -1;
    size_t base=(max_rows+1)*l->bundle_bytes;
    if(base>SIZE_MAX-sizeof(RowCache))return -1;
    base+=sizeof(RowCache);
    size_t batch_slots=1;
    while(batch_slots<max_rows*2){if(batch_slots>SIZE_MAX/2)return -1;batch_slots*=2;}
    if(batch_slots>(SIZE_MAX-base)/sizeof(size_t))return -1;
    base+=batch_slots*sizeof(size_t);
    if(base>budget||l->bundle_bytes>SIZE_MAX-sizeof(RowEntry))return -1;
    size_t upper=(budget-base)/(l->bundle_bytes+sizeof(RowEntry)),lower=0;
    if(l->rows<(uint64_t)upper)upper=(size_t)l->rows;
    while(lower<upper) {
        size_t middle=lower+(upper-lower+1)/2,total,slots;
        if(!cost(base,l->bundle_bytes,middle,&total,&slots)&&total<=budget)lower=middle;
        else upper=middle-1;
    }
    size_t total,slots;
    if(!lower||cost(base,l->bundle_bytes,lower,&total,&slots))return -1;
    RowCache *r=malloc(total);if(!r)return -1;
    memset(r,0,sizeof(*r));r->source=source;r->source_state=source->state;r->layout=*l;
    r->stats.budget_bytes=budget;r->stats.allocated_bytes=total;
    r->stats.capacity=lower;r->stats.max_gather_rows=max_rows;r->slots=slots;
    r->batch_slots=batch_slots;r->head=r->tail=NONE;
    r->entries=(RowEntry *)(r+1);r->hash=(size_t *)(r->entries+lower);
    memset(r->hash,0,slots*sizeof(size_t));r->batch_hash=r->hash+slots;
    r->rows=(unsigned char *)(r->batch_hash+batch_slots);
    r->staging=r->rows+lower*l->bundle_bytes;r->scratch=r->staging+max_rows*l->bundle_bytes;
    cache->state=r;return 0;
}
void bn_backend_model_row_cache_free(BnBackendRowCache *cache) {
    if(cache){free(cache->state);cache->state=NULL;}
}
int bn_backend_model_row_cache_stats(const BnBackendRowCache *cache,BnBackendRowCacheStats *out) {
    const RowCache *r=get_cache(cache);if(!r||!out)return -1;
    *out=r->stats;return 0;
}
const BnTrellisRowLayout *bn_backend_model_row_cache_layout(const BnBackendRowCache *cache) {
    const RowCache *r=get_cache(cache);return r?&r->layout:NULL;
}
static size_t row_hash(uint64_t row) {
    row^=row>>30;row*=UINT64_C(0xbf58476d1ce4e5b9);
    row^=row>>27;row*=UINT64_C(0x94d049bb133111eb);row^=row>>31;
    return (size_t)row;
}
static size_t home(const RowCache *r,uint64_t row) {
    return row_hash(row)&(r->slots-1);
}
static size_t find(const RowCache *r,uint64_t row) {
    size_t at=home(r,row);
    while(r->hash[at]) {
        if(r->entries[r->hash[at]-1].row==row)return r->hash[at]-1;
        at=(at+1)&(r->slots-1);
    }
    return NONE;
}
static void insert(RowCache *r,size_t slot) {
    size_t at=home(r,r->entries[slot].row);
    while(r->hash[at])at=(at+1)&(r->slots-1);
    r->hash[at]=slot+1;
}
/* Backshift deletion keeps empty-bucket lookup valid without tombstone growth. */
static void erase(RowCache *r,uint64_t row) {
    size_t at=home(r,row),mask=r->slots-1;
    while(r->entries[r->hash[at]-1].row!=row)at=(at+1)&mask;
    size_t hole=at;
    for(size_t next=(at+1)&mask;r->hash[next];next=(next+1)&mask) {
        size_t start=home(r,r->entries[r->hash[next]-1].row);
        if(((next-start)&mask)>=((next-hole)&mask)) {
            r->hash[hole]=r->hash[next];hole=next;
        }
    }
    r->hash[hole]=0;
}
static void unlink_entry(RowCache *r,size_t slot) {
    RowEntry *e=&r->entries[slot];
    if(e->prev!=NONE)r->entries[e->prev].next=e->next;else r->head=e->next;
    if(e->next!=NONE)r->entries[e->next].prev=e->prev;else r->tail=e->prev;
}
static void front(RowCache *r,size_t slot) {
    RowEntry *e=&r->entries[slot];e->prev=NONE;e->next=r->head;
    if(r->head!=NONE)r->entries[r->head].prev=slot;else r->tail=slot;
    r->head=slot;
}
static void add(uint64_t *value,uint64_t increment) {
    *value=increment>UINT64_MAX-*value?UINT64_MAX:*value+increment;
}
static int disjoint(const void *a,size_t n,const void *b,size_t m) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(n>UINTPTR_MAX-x||m>UINTPTR_MAX-y)return 0;
    return x+n<=y||y+m<=x;
}
int bn_backend_model_row_cache_gather(BnBackendRowCache *cache,
    const uint64_t *ids,size_t count,void *weights,size_t wb,void *scales,size_t sb) {
    RowCache *r=get_cache(cache);
    if(!r||count>r->stats.max_gather_rows||wb!=count*r->layout.weight_bytes||
        sb!=count*r->layout.scale_bytes)return -1;
    if(!count)return 0;
    size_t ib=count*sizeof(*ids);
    if(!ids||!weights||!scales||!disjoint(weights,wb,scales,sb)||
        !disjoint(ids,ib,weights,wb)||!disjoint(ids,ib,scales,sb)||
        !disjoint(r,r->stats.allocated_bytes,weights,wb)||
        !disjoint(r,r->stats.allocated_bytes,scales,sb)||
        !disjoint(r,r->stats.allocated_bytes,ids,ib))return -1;
    for(size_t i=0;i<count;i++)if(ids[i]>=r->layout.rows)return -1;
    size_t bytes=r->layout.bundle_bytes;
    memset(r->batch_hash,0,r->batch_slots*sizeof(size_t));
    for(size_t i=0;i<count;i++) {
        size_t batch=row_hash(ids[i])&(r->batch_slots-1);
        while(r->batch_hash[batch]&&ids[r->batch_hash[batch]-1]!=ids[i])batch=(batch+1)&(r->batch_slots-1);
        size_t slot=find(r,ids[i]);
        if(slot!=NONE){add(&r->stats.hits,1);unlink_entry(r,slot);}
        else {
            add(&r->stats.misses,1);
            const unsigned char *data=r->scratch;
            if(r->batch_hash[batch]) {
                data=r->staging+(r->batch_hash[batch]-1)*bytes;
                add(&r->stats.staging_hits,1);
            } else {
                add(&r->stats.source_bytes_requested,bytes);
                double start=bn_platform_time_ms();
                int result=bn_trellis_row_source_read(r->source,ids[i],r->scratch,bytes);
                double elapsed=bn_platform_time_ms()-start;
                if(elapsed>0)r->stats.source_read_ms+=elapsed;
                if(result){add(&r->stats.io_failures,1);return -1;}
            }
            if(r->stats.resident_rows<r->stats.capacity)slot=r->stats.resident_rows++;
            else {
                slot=r->tail;erase(r,r->entries[slot].row);unlink_entry(r,slot);add(&r->stats.evictions,1);
            }
            r->entries[slot].row=ids[i];memcpy(r->rows+slot*bytes,data,bytes);insert(r,slot);
        }
        front(r,slot);memcpy(r->staging+i*bytes,r->rows+slot*bytes,bytes);
        if(!r->batch_hash[batch])r->batch_hash[batch]=i+1;
    }
    for(size_t i=0;i<count;i++) {
        memcpy((unsigned char *)weights+i*r->layout.weight_bytes,r->staging+i*bytes,r->layout.weight_bytes);
        memcpy((unsigned char *)scales+i*r->layout.scale_bytes,r->staging+i*bytes+r->layout.weight_bytes,r->layout.scale_bytes);
    }
    return 0;
}
