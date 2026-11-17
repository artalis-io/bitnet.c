/* Main adapter lifetime checks with deterministic payloads and failed storage.
 * Actual model allocation plans and numerical output are qualified separately. */
#include "../src/main_native_options.h"
#include "../src/main_gguf_batch_options.h"
#include "../src/transformer/bf16_internal.h"
#include "model_internal.h"
#include "session.h"
#include "session_internal.h"
#include "json.h"
#include <assert.h>
#include <stdlib.h>

typedef struct {
    BnModel model;BnGPUBackend *gpu;GGUFBatchOptions options;BnTransformerBF16Policy policy;
    struct {int count;} input;struct {int n;} jobs[64];
    int dense,chunk,failed;size_t piece_capacity,frontend_host,model_host_budget,model_device_budget;
    BnGenerateBatchMemoryLedger memory;BnGenerateBatchMemoryQuote request_quote;
    BnTransformerBF16RequestMemory backend_quote,coordinator_quote;BnBackendBF16MemoryPlan model_quote;
    struct {BnSession *session;BnGenerateBatchMemoryTicket ticket;int leased,capacity;} memory_slots[16];
    uint64_t created,reused,evicted,resized,discarded;size_t kv_reserved,kv_peak,kv_limit;
} GGUFBatch;
#include "../src/main_gguf_memory.h"
typedef struct {BnSession session;int capacity,healthy;} FakeSession;
static int creates,frees,prepares,restarts,resizes,fail_prepare,fail_restart,fail_resize;
BnBackendModel *bn_model_backend(const BnModel *m){return (BnBackendModel *)(uintptr_t)m;}
BnBackendSession *bn_session_backend(const BnSession *s){return (BnBackendSession *)(uintptr_t)s;}
BnSession *bn_session_create_backend(const BnModel *m,BnAllocator *a) {
    assert(m&&!a);FakeSession *s=calloc(1,sizeof(*s));assert(s);s->healthy=1;creates++;return &s->session;
}
void bn_session_free(BnSession *s,BnAllocator *a){assert(!a);if(s){frees++;free(s);}}
size_t bn_transformer_bf16_dense_kv_bytes(const BnModel *m,int capacity){assert(m&&capacity>0);return (size_t)capacity*10;}
int bn_transformer_bf16_dense_memory_plan(const BnModel *m,BnGPUBackend *g,int tokens,int capacity,
    const BnTransformerBF16Policy *policy,BnTransformerBF16RequestMemory *out) {
    assert(m&&g&&tokens==4&&policy&&out&&capacity>=4&&capacity<=256);
    *out=(BnTransformerBF16RequestMemory){100+(size_t)capacity*10,32,32};return 0;
}
int bn_transformer_bf16_request_init_with_policy(const BnModel *m,BnGPUBackend *g,BnBackendSession *session,
    int tokens,int capacity,void *cs,int rotary,const BnTransformerBF16Policy *policy) {
    assert(m&&g&&session&&tokens==4&&capacity>=4&&!cs&&!rotary&&policy);prepares++;
    ((FakeSession *)session)->capacity=capacity;return fail_prepare?-1:0;
}
int bn_transformer_bf16_session_restart(const BnModel *m,BnSession *s) {
    assert(m&&s);restarts++;s->pos=0;return fail_restart?-1:0;
}
int bn_transformer_bf16_resize_idle_kv(const BnModel *m,BnSession *s,int capacity) {
    assert(m&&s&&!s->pos&&capacity>((FakeSession *)s)->capacity);resizes++;
    if(fail_resize){((FakeSession *)s)->healthy=0;return -1;}
    ((FakeSession *)s)->capacity=capacity;return 0;
}
int bn_transformer_bf16_position(const BnBackendSession *s){return ((const FakeSession *)s)->healthy?((const FakeSession *)s)->session.pos:-1;}
const BnBackendBF16GLMModel *bn_backend_layout_bf16_glm_model(const BnBackendModel *m){(void)m;return NULL;}
const BnBackendBF16FlashIOModel *bn_backend_layout_bf16_flash_model(const BnBackendModel *m){(void)m;return NULL;}
int bn_transformer_bf16_glm_memory_plan(const BnModel *m,int n,int c,BnTransformerBF16RequestMemory *out){(void)m;(void)n;(void)c;(void)out;assert(0);return -1;}
int bn_transformer_bf16_flash_memory_plan(const BnModel *m,BnGPUBackend *g,int n,int c,BnTransformerBF16RequestMemory *out){(void)m;(void)g;(void)n;(void)c;(void)out;assert(0);return -1;}
int bn_transformer_bf16_glm_prepare_context(const BnModel *m,BnSession *s,int n,int c,size_t d,size_t h){(void)m;(void)s;(void)n;(void)c;(void)d;(void)h;assert(0);return -1;}
int bn_transformer_bf16_flash_prepare_context(const BnModel *m,BnGPUBackend *g,BnBackendSession *s,int n,int c,size_t d,size_t h){(void)m;(void)g;(void)s;(void)n;(void)c;(void)d;(void)h;assert(0);return -1;}
int bn_transformer_bf16_glm_info(const BnBackendSession *s,BnTransformerBF16GLMRequestInfo *out){(void)s;(void)out;assert(0);return -1;}
int bn_transformer_bf16_flash_info(const BnBackendSession *s,BnTransformerBF16FlashInfo *out){(void)s;(void)out;assert(0);return -1;}
static int resize_capability(void *r,BnGPUResidentView *v,size_t n){(void)r;(void)v;(void)n;assert(0);return -1;}
static void setup(GGUFBatch *b,BnGPUBackend *g,size_t device,int live,size_t kv) {
    memset(b,0,sizeof(*b));b->gpu=g;b->dense=1;b->chunk=4;b->kv_limit=kv;
    b->model.config.seq_len=256;b->model.config.vocab_size=8;
    b->options.live=live;b->options.device_mb=b->options.host_mb=1;
    assert(!bn_generate_batch_memory_init(&b->memory,(BnGenerateBatchMemory){SIZE_MAX,device},
        (BnGenerateBatchMemoryQuote){{0,100},{0,100}}));
    assert(!bn_generate_batch_memory_shared_commit(&b->memory));
    fail_prepare=fail_restart=fail_resize=0;
}
static void finish(GGUFBatch *b) {
    gguf_memory_free(b);assert(!b->failed&&b->memory.used.device_bytes==100&&b->memory.used.host_bytes==0&&!b->kv_reserved);
}
int main(void) {
    BnGPUBackend g={0};g.resident_resize_discard=resize_capability;GGUFBatch b;BnSession *a,*c,*out;
    setup(&b,&g,1000,2,2560);
    assert(!gguf_memory_acquire(&b,9,&a)&&!gguf_memory_acquire(&b,9,&c));a->pos=7;c->pos=13;
    assert(!gguf_memory_release(&b,a,0));int before_frees=frees,before_restarts=restarts;
    assert(gguf_memory_acquire(&b,33,&out)==1&&!out&&frees==before_frees&&restarts==before_restarts&&a->pos==7&&c->pos==13);
    assert(!gguf_memory_release(&b,c,0));
    assert(!gguf_memory_acquire(&b,33,&out));
    assert(((FakeSession *)out)->capacity==64&&b.created==3&&b.evicted==2&&!b.resized&&b.memory.used.device_bytes==840);
    assert(!gguf_memory_release(&b,out,0));a=out;before_restarts=restarts;
    assert(!gguf_memory_acquire(&b,9,&out)&&out==a&&restarts==before_restarts+1&&b.kv_reserved==640);
    finish(&b);
    setup(&b,&g,2000,1,2560);assert(!gguf_memory_acquire(&b,9,&a));assert(!gguf_memory_release(&b,a,0));
    before_frees=frees;int before_creates=creates,before_resizes=resizes;
    assert(!gguf_memory_acquire(&b,33,&out)&&out==a&&creates==before_creates&&frees==before_frees&&resizes==before_resizes+1);
    assert(b.resized==1&&b.reused==1&&b.kv_reserved==640&&b.memory.used.device_bytes==840&&b.memory.peak.device_bytes==1100);
    ((FakeSession *)out)->healthy=0;assert(!gguf_memory_release(&b,out,0)&&b.discarded==1);finish(&b);
    setup(&b,&g,2000,1,2560);assert(!gguf_memory_acquire(&b,9,&a)&&!gguf_memory_release(&b,a,0));fail_resize=1;
    assert(gguf_memory_acquire(&b,33,&out)==-1&&!out&&b.discarded==1);finish(&b);
    setup(&b,&g,2000,1,2560);fail_prepare=1;
    assert(gguf_memory_acquire(&b,9,&out)==-1&&!out&&!b.created);finish(&b);
    setup(&b,&g,2000,1,2560);assert(!gguf_memory_acquire(&b,9,&a)&&!gguf_memory_release(&b,a,0));fail_restart=1;
    assert(gguf_memory_acquire(&b,9,&out)==-1&&!out&&b.discarded==1);finish(&b);
    setup(&b,&g,2000,2,320);assert(!gguf_memory_acquire(&b,9,&a)&&!gguf_memory_acquire(&b,9,&c)&&!gguf_memory_release(&b,a,0));
    before_frees=frees;assert(gguf_memory_acquire(&b,17,&out)==1&&!out&&frees==before_frees&&b.kv_reserved==320);finish(&b);
    assert(creates==frees);
    (void)native_cli_parse;(void)native_cli_usage;(void)gguf_batch_parse;(void)gguf_batch_usage;
    (void)gguf_memory_frontend;(void)gguf_memory_init;(void)gguf_memory_coordinator;(void)gguf_memory_ready;(void)gguf_memory_report;
    puts("GGUF managed pool: active-pressure atomicity, charged idle reuse, growth overlap, tight-budget eviction and failed prepare/restart/resize cleanup PASSED");
    return 0;
}
