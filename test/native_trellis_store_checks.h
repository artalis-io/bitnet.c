#ifndef __EMSCRIPTEN__
typedef struct {
    BnGPUMoETrellisStore *store;
    size_t live, peak, pinned;
    int batches, pin_fail, host_copies;
    int calls, frees, fail_at, writes, write_fail_at, reusable;
} TrellisStoreTestGPU;
static void *trellis_store_test_upload(void *context, const void *data,
    size_t bytes, int format, int k, int n, uint32_t marker) {
    TrellisStoreTestGPU *t = context;
    assert(data && bytes == 13056 && format == BN_QUANT_FORMAT_EXL3_MUL1_3 &&
        marker == UINT32_C(0x83dcd12d) &&
        ((k == 128 && n == 256) || (k == 256 && n == 128)));
    BnGPUMoETrellisStats stats;
    assert(!bn_gpu_moe_trellis_stats(t->store, &stats));
    assert(stats.decode.pending + stats.temporary.pending >= 1);
    assert(t->reusable || stats.decode.pending + stats.temporary.pending == 1);
    assert(stats.decode.charged_payload <= 39168 &&
           stats.temporary.charged_payload <= (size_t)(t->reusable?78336:39168));
    if (++t->calls == t->fail_at) return NULL;
    void *out = malloc(bytes); assert(out); memcpy(out, data, bytes);
    t->live += bytes;
    if (t->live > t->peak) t->peak = t->live;
    return out;
}
static void trellis_store_test_destroy(void *context, void *buffer) {
    TrellisStoreTestGPU *t = context;
    assert(buffer && t->live >= 13056);
    t->frees++; t->live -= 13056; free(buffer);
}
static void trellis_store_test_bytes(const BnGPUMoETrellisLease *lease,
    int layer, int expert) {
    BnMoEHostExpert host = {0};
    host.projection[0] = lease->buffers.gate;
    host.projection[1] = lease->buffers.up;
    host.projection[2] = lease->buffers.down;
    check_catalog_bytes(&host, layer, expert);
}
static int trellis_store_test_write(void *context,void *buffer,const void *data,
    size_t bytes,int format,int k,int n,uint32_t marker) {
    TrellisStoreTestGPU *t=context;
    assert(buffer&&data&&bytes==13056&&format==BN_QUANT_FORMAT_EXL3_MUL1_3&&
        marker==UINT32_C(0x83dcd12d)&&
        ((k==128&&n==256)||(k==256&&n==128)));
    if(++t->writes==t->write_fail_at) {
        memset(buffer,0xcc,bytes); /* A failed DMA may have overwritten data. */
        return -1;
    }
    memcpy(buffer,data,bytes);return 0;
}
typedef struct { TrellisStoreTestGPU *owner; void *host; size_t bytes; } TrellisTestUpload;
static void *trellis_test_upload_create(void *context,size_t bytes,void **host) {
    TrellisStoreTestGPU *t=context;
    assert(!t->pinned&&bytes==39168);
    if(t->pin_fail)return NULL;
    TrellisTestUpload *u=calloc(1,sizeof(*u));assert(u);
    u->owner=t;u->host=malloc(bytes);assert(u->host);u->bytes=bytes;
    t->pinned=bytes;*host=u->host;return u;
}
static void trellis_test_upload_destroy(void *context,void *state) {
    TrellisTestUpload *u=state;TrellisStoreTestGPU *t=context;
    assert(u->owner==t&&t->pinned==u->bytes);
    t->pinned=0;free(u->host);free(u);
}
static int trellis_test_upload_batch(void *context,void *state,
    BnGPUTrellisUploadItem *items,int count) {
    TrellisTestUpload *u=state;TrellisStoreTestGPU *t=context;
    assert(u->owner==t&&count==3);t->batches++;
    void *buffers[3]={0};int created[3]={0};
    for(int p=0;p<count;p++) {
        BnGPUTrellisUploadItem *v=&items[p];
        assert(v->offset==(size_t)p*13056&&v->bytes==13056&&
            v->offset+v->bytes<=u->bytes);
        void *data=(unsigned char*)u->host+v->offset;
        if(v->source) {
            assert(v->source!=data);
            memcpy(data,v->source,v->bytes);t->host_copies++;
        }
        buffers[p]=v->buffer;
        if(buffers[p]) {
            if(trellis_store_test_write(context,buffers[p],data,v->bytes,
                v->format,v->k,v->n,v->marker))goto fail;
        } else {
            buffers[p]=trellis_store_test_upload(context,data,v->bytes,
                v->format,v->k,v->n,v->marker);
            if(!buffers[p])goto fail;
            created[p]=1;
        }
    }
    for(int p=0;p<count;p++)items[p].buffer=buffers[p];
    return 0;
fail:
    for(int p=0;p<count;p++)if(created[p])trellis_store_test_destroy(context,buffers[p]);
    return -1;
}
static void test_native_trellis_pinned(BnModel *model) {
    BnGPUMoETrellisStore store={0};
    TrellisStoreTestGPU t={.store=&store,.reusable=1};
    BnGPUBackend gpu={.ctx=&t,.trellis_buffer_create=trellis_store_test_upload,
        .trellis_buffer_write=trellis_store_test_write,.buffer_destroy=trellis_store_test_destroy,
        .trellis_upload_create=trellis_test_upload_create,
        .trellis_upload_batch=trellis_test_upload_batch,
        .trellis_upload_destroy=trellis_test_upload_destroy};
    assert(bn_gpu_moe_trellis_store_init_pinned(&store,model,&gpu,0,39168,0,39168,39167)==-1);
    gpu.trellis_upload_batch=NULL;
    assert(bn_gpu_moe_trellis_store_init_pinned(&store,model,&gpu,0,39168,0,39168,39168)==-1);
    gpu.trellis_upload_batch=trellis_test_upload_batch;t.pin_fail=1;
    assert(bn_gpu_moe_trellis_store_init_pinned(&store,model,&gpu,0,39168,0,39168,39168)==-1);
    assert(!store.state&&!t.pinned);t.pin_fail=0;
    for(int admit=0;admit<=1;admit++) {
        t=(TrellisStoreTestGPU){.store=&store,.reusable=1};
        assert(!bn_gpu_moe_trellis_store_init_pinned(&store,model,&gpu,
            admit?39168:0,admit?39168:0,admit?0:39168,39168,39168));
        BnGPUMoETrellisLease a={0},b={0};BnGPUMoETrellisStats stats;
        assert(!bn_gpu_moe_trellis_acquire(&store,0,0,admit,&a));
        trellis_store_test_bytes(&a,0,0);
        assert(bn_gpu_moe_trellis_acquire(&store,1,1,admit,&b)==BN_GPU_MOE_CACHE_BUSY);
        assert(t.batches==1&&t.calls==3&&!t.writes&&t.host_copies==admit*3);
        assert(!bn_gpu_moe_trellis_release(&store,&a));
        assert(!bn_gpu_moe_trellis_acquire(&store,1,1,admit,&a));
        trellis_store_test_bytes(&a,1,1);
        assert(t.batches==2&&t.calls==3&&t.writes==3&&t.host_copies==admit*6);
        assert(!bn_gpu_moe_trellis_release(&store,&a));
        t.write_fail_at=t.writes+2;
        assert(bn_gpu_moe_trellis_acquire(&store,0,1,admit,&a)==-1&&!a.owner);
        assert(!bn_gpu_moe_trellis_stats(&store,&stats)&&!stats.live_leases&&
            !stats.decode.pending&&!stats.temporary.pending&&!t.live);
        t.write_fail_at=0;
        assert(!bn_gpu_moe_trellis_acquire(&store,0,1,admit,&a));
        trellis_store_test_bytes(&a,0,1);
        assert(!bn_gpu_moe_trellis_release(&store,&a));
        assert(!bn_gpu_moe_trellis_stats(&store,&stats)&&stats.upload_batches==3&&
            stats.pinned_staging_payload==39168&&stats.staging_payload==39168&&
            stats.buffer_create_count==6&&stats.buffer_reuse_count==3);
        assert(!bn_gpu_moe_trellis_store_close(&store)&&!t.live&&!t.pinned&&t.frees==6);
    }
    for(int fail=1;fail<=3;fail++) {
        t=(TrellisStoreTestGPU){.store=&store,.reusable=1,.fail_at=fail};
        assert(!bn_gpu_moe_trellis_store_init_pinned(&store,model,&gpu,0,39168,0,39168,39168));
        BnGPUMoETrellisLease a={0};
        assert(bn_gpu_moe_trellis_acquire(&store,0,0,1,&a)==-1&&!a.owner&&!t.live);
        assert(!bn_gpu_moe_trellis_store_close(&store)&&!t.pinned&&t.frees==fail-1);
    }
    puts("Native bounded pinned triplet upload, admission, poison and cleanup PASSED");
}
static void test_native_trellis_reuse(BnModel *model) {
    BnGPUMoETrellisStore store={0};
    TrellisStoreTestGPU t={.store=&store,.reusable=1};
    BnGPUBackend gpu={.ctx=&t,.trellis_buffer_create=trellis_store_test_upload,
        .trellis_buffer_write=trellis_store_test_write,.buffer_destroy=trellis_store_test_destroy};
    assert(!bn_gpu_moe_trellis_store_init(&store,model,&gpu,0,39168,78336,39168));
    BnGPUMoETrellisLease a={0},b={0},c={0},cached={0};
    assert(!bn_gpu_moe_trellis_acquire(&store,0,0,0,&a));
    assert(!bn_gpu_moe_trellis_acquire(&store,0,1,0,&b));
    BnGPUMoETrellisLease stale=a;void *first=a.buffers.gate;
    assert(bn_gpu_moe_trellis_acquire(&store,0,0,0,&c)==BN_GPU_MOE_CACHE_BUSY);
    assert(bn_gpu_moe_trellis_acquire(&store,1,1,0,&c)==BN_GPU_MOE_CACHE_BUSY);
    assert(!c.owner&&t.calls==6&&!t.writes);
    assert(bn_gpu_moe_trellis_store_close(&store)==BN_GPU_MOE_CACHE_BUSY);
    assert(!bn_gpu_moe_trellis_release(&store,&a)&&t.live==78336);
    assert(!bn_gpu_moe_trellis_acquire(&store,1,1,0,&c));
    assert(c.buffers.gate==first&&t.calls==6&&t.writes==3);
    trellis_store_test_bytes(&c,1,1);trellis_store_test_bytes(&b,0,1);
    assert(bn_gpu_moe_trellis_release(&store,&stale)==-1);
    c.buffers.gate=b.buffers.gate;
    assert(bn_gpu_moe_trellis_release(&store,&c)==-1);
    c.buffers.gate=first;
    assert(!bn_gpu_moe_trellis_release(&store,&c));
    assert(!bn_gpu_moe_trellis_release(&store,&b));
    assert(!bn_gpu_moe_trellis_acquire(&store,1,0,1,&cached));
    assert(!bn_gpu_moe_trellis_acquire(&store,1,0,0,&a));
    assert(a.cached.cache&&a.buffers.gate==cached.buffers.gate&&t.calls==9&&t.writes==3);
    assert(!bn_gpu_moe_trellis_release(&store,&a));
    assert(!bn_gpu_moe_trellis_release(&store,&cached));
    t.write_fail_at=t.writes+2;
    assert(bn_gpu_moe_trellis_acquire(&store,0,0,0,&a)==-1&&!a.owner);
    BnGPUMoETrellisStats stats;
    assert(!bn_gpu_moe_trellis_stats(&store,&stats));
    assert(!stats.live_leases&&!stats.temporary.pending&&
        stats.temporary_allocated_payload==39168&&t.live==78336);
    t.write_fail_at=0;
    assert(!bn_gpu_moe_trellis_acquire(&store,0,0,0,&a)&&t.calls==12);
    trellis_store_test_bytes(&a,0,0);
    assert(!bn_gpu_moe_trellis_release(&store,&a));
    assert(!bn_gpu_moe_trellis_stats(&store,&stats));
    assert(stats.temporary_allocated_payload==78336&&stats.temporary_peak_payload==78336&&
        stats.temporary_allocated_payload<=stats.temporary.capacity_payload&&
        stats.buffer_create_count==12&&stats.buffer_reuse_count==4&&
        !stats.live_leases&&!stats.temporary.charged_payload&&t.peak==3*39168);
    assert(!bn_gpu_moe_trellis_store_close(&store)&&!t.live&&t.frees==12);
    for(int fail=1;fail<=3;fail++) {
        t=(TrellisStoreTestGPU){.store=&store,.reusable=1,.fail_at=fail};
        assert(!bn_gpu_moe_trellis_store_init(&store,model,&gpu,0,0,39168,39168));
        assert(bn_gpu_moe_trellis_acquire(&store,0,0,0,&a)==-1&&!a.owner);
        assert(!bn_gpu_moe_trellis_stats(&store,&stats)&&
            !stats.temporary_allocated_payload&&!stats.temporary.pending&&!t.live);
        assert(!bn_gpu_moe_trellis_store_close(&store)&&t.frees==fail-1);
    }
    puts("Native reusable expert payload bounds, pinned isolation, poison and teardown PASSED");
}
static void test_native_trellis_decode_reuse(BnModel *model) {
    BnGPUMoETrellisStore store={0};
    TrellisStoreTestGPU t={.store=&store,.reusable=1};
    BnGPUBackend gpu={.ctx=&t,.trellis_buffer_create=trellis_store_test_upload,
        .trellis_buffer_write=trellis_store_test_write,.buffer_destroy=trellis_store_test_destroy};
    assert(!bn_gpu_moe_trellis_store_init(&store,model,&gpu,0,39168,0,39168));
    BnGPUMoETrellisLease a={0},b={0};BnGPUMoETrellisStats stats;
    assert(!bn_gpu_moe_trellis_acquire(&store,0,0,1,&a));
    assert(bn_gpu_moe_trellis_acquire(&store,0,1,1,&b)==BN_GPU_MOE_CACHE_BUSY);
    assert(t.calls==3&&!t.writes);
    void *buffer=a.buffers.gate;assert(!bn_gpu_moe_trellis_release(&store,&a));
    assert(!bn_gpu_moe_trellis_acquire(&store,1,1,1,&b));
    assert(b.buffers.gate==buffer&&t.calls==3&&t.writes==3&&t.live==39168);
    trellis_store_test_bytes(&b,1,1);assert(!bn_gpu_moe_trellis_release(&store,&b));
    t.write_fail_at=t.writes+2;
    assert(bn_gpu_moe_trellis_acquire(&store,0,1,1,&a)==-1&&!a.owner);
    assert(!bn_gpu_moe_trellis_stats(&store,&stats)&&!stats.decode.pending&&
        !stats.decode.published&&!stats.live_leases&&!t.live&&t.frees==3);
    t.write_fail_at=0;
    assert(!bn_gpu_moe_trellis_acquire(&store,0,1,1,&a)&&t.calls==6);
    trellis_store_test_bytes(&a,0,1);assert(!bn_gpu_moe_trellis_release(&store,&a));
    assert(!bn_gpu_moe_trellis_store_close(&store)&&!t.live&&t.frees==6);
    puts("Native decode LRU refill: pinned refusal, exact bytes, poisoned replacement and retry PASSED");
}
static void test_native_trellis_ram_policy(const BnModel *model) {
    const size_t entry=39168;
    BnGPUMoETrellisStore store={0};TrellisStoreTestGPU t={.store=&store,.reusable=1};
    BnGPUBackend gpu={.ctx=&t,.trellis_buffer_create=trellis_store_test_upload,
        .trellis_buffer_write=trellis_store_test_write,.buffer_destroy=trellis_store_test_destroy};
    BnGPUMoETrellisPolicy policy={.ram_admit_misses=1};BnGPUMoETrellisLease a={0},b={0};BnGPUMoETrellisStats before,after;
    assert(bn_gpu_moe_trellis_store_init_policy(&store,model,&gpu,0,entry,entry,entry,0,&policy)==-1&&!store.state);
    policy.ram_admit_misses=2;
    assert(bn_gpu_moe_trellis_store_init_policy(&store,model,&gpu,entry,entry,entry,entry,0,&policy)==-1&&!store.state);
    policy.ram_admit_misses=0;
    policy.ram_pages=(BnMoECachePagePolicy)-1;
    assert(bn_gpu_moe_trellis_store_init_policy(&store,model,&gpu,entry,entry,entry,entry,0,&policy)==-1&&!store.state);
    policy.ram_pages=(BnMoECachePagePolicy)2;
    assert(bn_gpu_moe_trellis_store_init_policy(&store,model,&gpu,entry,entry,entry,entry,0,&policy)==-1&&!store.state);
    policy.ram_pages=BN_MOE_CACHE_PAGES_SYSTEM;
    assert(bn_gpu_moe_trellis_store_init_policy(&store,model,&gpu,0,entry,entry,entry,0,&policy)==-1&&!store.state&&!t.calls);
    policy.ram_admit_misses=1;
    assert(!bn_gpu_moe_trellis_store_init_policy(&store,model,&gpu,entry,entry,entry,entry,0,&policy));
    policy.ram_pages=BN_MOE_CACHE_PAGES_HUGE_HINT;
    policy.ram_admit_misses=0; /* Store copied the policy, not this pointer. */
    assert(!bn_gpu_moe_trellis_acquire(&store,0,0,0,&a));trellis_store_test_bytes(&a,0,0);
    assert(!bn_gpu_moe_trellis_stats(&store,&before)&&before.host.io_count==3&&!before.decode.published&&before.ram_pages==BN_MOE_CACHE_PAGES_SYSTEM);
    assert(bn_gpu_moe_trellis_acquire(&store,1,1,0,&b)==BN_GPU_MOE_CACHE_BUSY&&!b.owner);
    assert(!bn_gpu_moe_trellis_stats(&store,&after)&&!memcmp(&before.host,&after.host,sizeof(before.host)));
    assert(!bn_gpu_moe_trellis_release(&store,&a));
    assert(!bn_gpu_moe_trellis_acquire(&store,0,0,0,&a));trellis_store_test_bytes(&a,0,0);
    assert(!bn_gpu_moe_trellis_release(&store,&a));
    assert(!bn_gpu_moe_trellis_stats(&store,&after)&&after.host.io_count==3&&
        after.host.gpu_host_cache_hits==1&&after.upload_bytes==entry*2&&!after.decode.published);
    /* GPU admission can reuse RAM, while a later RAM-only scan leaves that
     * GPU entry intact even after its host copy is evicted. */
    assert(!bn_gpu_moe_trellis_acquire(&store,0,0,1,&a));trellis_store_test_bytes(&a,0,0);
    assert(!bn_gpu_moe_trellis_release(&store,&a));
    assert(!bn_gpu_moe_trellis_acquire(&store,1,1,0,&b));trellis_store_test_bytes(&b,1,1);
    assert(!bn_gpu_moe_trellis_release(&store,&b));
    assert(!bn_gpu_moe_trellis_stats(&store,&before)&&before.host.io_count==6&&before.decode.published==1&&!before.decode.evictions);
    assert(!bn_gpu_moe_trellis_acquire(&store,0,0,0,&a));trellis_store_test_bytes(&a,0,0);
    assert(!bn_gpu_moe_trellis_release(&store,&a));
    assert(!bn_gpu_moe_trellis_stats(&store,&after)&&after.upload_count==before.upload_count&&after.host.io_count==before.host.io_count);
    assert(!bn_gpu_moe_trellis_store_close(&store)&&!t.live);
    /* Failed device uploads may leave a valid admitted RAM entry. Retrying
     * consumes it without another read; reservations and leases are released. */
    memset(&t,0,sizeof(t));t.store=&store;t.reusable=1;t.fail_at=2;
    policy.ram_admit_misses=1;
    assert(!bn_gpu_moe_trellis_store_init_policy(&store,model,&gpu,entry,entry,entry,entry,0,&policy));
    assert(bn_gpu_moe_trellis_acquire(&store,0,1,0,&a)==-1&&!a.owner&&!t.live);
    assert(!bn_gpu_moe_trellis_stats(&store,&after)&&after.host.io_count==3&&!after.temporary.pending&&!after.live_leases);
    t.fail_at=0;assert(!bn_gpu_moe_trellis_acquire(&store,0,1,0,&a));trellis_store_test_bytes(&a,0,1);
    assert(!bn_gpu_moe_trellis_release(&store,&a));
    assert(!bn_gpu_moe_trellis_stats(&store,&after)&&after.host.io_count==3&&after.host.gpu_host_cache_hits==1);
    assert(!bn_gpu_moe_trellis_store_close(&store)&&!t.live);
    /* Pinned staging still has its one existing bundle charge. */
    memset(&t,0,sizeof(t));t.store=&store;t.reusable=1;
    gpu.trellis_upload_create=trellis_test_upload_create;gpu.trellis_upload_batch=trellis_test_upload_batch;
    gpu.trellis_upload_destroy=trellis_test_upload_destroy;
    assert(!bn_gpu_moe_trellis_store_init_policy(&store,model,&gpu,entry,entry,entry,entry,entry,&policy));
    for(int i=0;i<2;i++) {
        assert(!bn_gpu_moe_trellis_acquire(&store,0,0,0,&a));trellis_store_test_bytes(&a,0,0);
        assert(!bn_gpu_moe_trellis_release(&store,&a));
    }
    assert(!bn_gpu_moe_trellis_stats(&store,&after)&&after.host.io_count==3&&after.host.gpu_host_cache_hits==1&&
        after.staging_payload==entry&&after.pinned_staging_payload==entry&&!after.decode.published);
    assert(t.host_copies==6&&t.batches==2);
    assert(!bn_gpu_moe_trellis_store_close(&store)&&!t.live&&!t.pinned);
    puts("Native independent RAM admission: bounded reuse, BUSY no IO, GPU protection, failed upload retry and pinned staging PASSED");
}
static void test_native_trellis_store(void) {
    char dir[] = "/tmp/bn-trellis-store-XXXXXX", path[256];
    assert(mkdtemp(dir)); trellis_catalog_fixture(dir, 0, 0);
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    BnSafetensorsIndex index = {0}; BnModel model = {0};
    assert(!bn_safetensors_index_init(&index, path));
    assert(!bn_model_bind_trellis_experts(&model, &index, "deepseek_v41", 2, 2, 128, 256));
    BnGPUMoETrellisStore store = {0}, other = {0};
    TrellisStoreTestGPU t = {.store = &store};
    BnGPUBackend gpu = {.ctx = &t, .trellis_buffer_create = trellis_store_test_upload,
                        .buffer_destroy = trellis_store_test_destroy};
    assert(bn_gpu_moe_trellis_store_init(&store, &model, &gpu, 0, 39168, 0, 39167) == -1);
    assert(bn_gpu_moe_trellis_store_init(&store, &model, &gpu, 1, 39168, 0, 39168) == -1);
    assert(bn_gpu_moe_trellis_store_init(&store, &model, &gpu, 0, 0, 0, 39168) == -1);
    assert(!store.state);
    assert(!bn_gpu_moe_trellis_store_init(&store, &model, &gpu, 39168, 39168, 39168, 39168));
    assert(bn_gpu_moe_trellis_store_init(&store, &model, &gpu, 39168, 39168, 39168, 39168) == -1);
    BnGPUMoETrellisLease a = {0}, b = {0}, c = {0}, empty = {0};
    assert(bn_gpu_moe_trellis_acquire(&store, -1, 0, 1, &a) == -1 && !a.owner);
    assert(bn_gpu_moe_trellis_acquire(&store, 0, 2, 1, &a) == -1 && !a.owner);
    assert(bn_gpu_moe_trellis_acquire(&store, 0, 0, 2, &a) == -1 && !a.owner);
    assert(!bn_gpu_moe_trellis_acquire(&store, 0, 0, 1, &a));
    trellis_store_test_bytes(&a, 0, 0);
    assert(!bn_gpu_moe_trellis_acquire(&store, 0, 0, 0, &b));
    assert(a.buffers.gate == b.buffers.gate && t.calls == 3);
    BnGPUMoETrellisLease saved = a;
    assert(bn_gpu_moe_trellis_acquire(&store, 1, 1, 1, &a) == -1 && !memcmp(&a, &saved, sizeof(a)));
    assert(bn_gpu_moe_trellis_store_close(&store) == BN_GPU_MOE_CACHE_BUSY);
    assert(bn_gpu_moe_trellis_release(&other, &a) == -1);
    assert(bn_gpu_moe_trellis_acquire(&store, 1, 1, 1, &c) == BN_GPU_MOE_CACHE_BUSY && !c.owner);
    assert(t.calls == 3);
    assert(!bn_gpu_moe_trellis_acquire(&store, 1, 1, 0, &c));
    trellis_store_test_bytes(&c, 1, 1);
    assert(t.live == 2 * 39168 && t.peak == 2 * 39168);
    assert(bn_gpu_moe_trellis_acquire(&store, 1, 1, 0, &empty) == BN_GPU_MOE_CACHE_BUSY);
    assert(bn_gpu_moe_trellis_acquire(&store, 0, 1, 0, &empty) == BN_GPU_MOE_CACHE_BUSY);
    assert(!empty.owner && t.calls == 6);
    uint64_t generation = c.temporary.generation;
    c.temporary.generation++;
    assert(bn_gpu_moe_trellis_release(&store, &c) == -1 && t.live == 2 * 39168);
    c.temporary.generation = generation;
    assert(!bn_gpu_moe_trellis_release(&store, &c) && t.live == 39168);
    assert(!bn_gpu_moe_trellis_release(&store, &b));
    assert(!bn_gpu_moe_trellis_release(&store, &a));
    assert(!bn_gpu_moe_trellis_release(&store, &a));
    assert(!bn_gpu_moe_trellis_acquire(&store, 0, 0, 0, &a) && t.calls == 6);
    assert(!bn_gpu_moe_trellis_release(&store, &a));
    BnGPUMoETrellisStats stats;
    assert(!bn_gpu_moe_trellis_stats(&store, &stats));
    assert(stats.host.io_count == 6 && stats.host.io_bytes == 2 * 39168 &&
        stats.upload_count == 6 && stats.upload_bytes == 2 * 39168 &&
        stats.staging_payload == 39168 && !stats.live_leases &&
        stats.decode.published == 1 && !stats.temporary.pending &&
        stats.decode.evictions == 0 && stats.temporary.aborts == 1);
    assert(!bn_gpu_moe_trellis_store_close(&store) && !t.live && t.frees == 6);
    assert(!bn_gpu_moe_trellis_store_close(&store));
    for (int fail = 1; fail <= 3; fail++) {
        t.calls = t.frees = 0; t.fail_at = fail;
        assert(!bn_gpu_moe_trellis_store_init(&store, &model, &gpu, 0, 39168, 39168, 39168));
        assert(bn_gpu_moe_trellis_acquire(&store, 0, 0, fail % 2, &a) == -1 && !a.owner);
        assert(!t.live && t.frees == fail - 1);
        assert(!bn_gpu_moe_trellis_stats(&store, &stats) && !stats.live_leases &&
               !stats.decode.pending && !stats.temporary.pending);
        t.fail_at = 0;
        assert(!bn_gpu_moe_trellis_acquire(&store, 0, 0, fail % 2, &a));
        assert(!bn_gpu_moe_trellis_release(&store, &a));
        assert(!bn_gpu_moe_trellis_store_close(&store) && !t.live);
    }
    assert(!bn_gpu_moe_trellis_store_init(&store, &model, &gpu, 0, 0, 39168, 39168));
    assert(bn_gpu_moe_trellis_acquire(&store, 0, 0, 1, &a) == BN_GPU_MOE_CACHE_BUSY);
    assert(!bn_gpu_moe_trellis_acquire(&store, 0, 0, 0, &a));
    assert(!bn_gpu_moe_trellis_release(&store, &a));
    assert(!bn_gpu_moe_trellis_store_close(&store) && !t.live);
    test_native_trellis_pinned(&model);
    test_native_trellis_reuse(&model);
    test_native_trellis_decode_reuse(&model);
    test_native_trellis_ram_policy(&model);
    t.calls = 0;
    assert(!bn_gpu_moe_trellis_store_init(&store, &model, &gpu, 0, 39168, 0, 39168));
    assert(bn_gpu_moe_trellis_acquire(&store, 0, 0, 0, &a) == BN_GPU_MOE_CACHE_BUSY);
    snprintf(path, sizeof(path), "%s/one.safetensors", dir);
    FILE *f = fopen(path, "wb"); assert(f && !fclose(f));
    assert(bn_gpu_moe_trellis_acquire(&store, 0, 0, 1, &a) == -1 && !t.calls && !a.owner);
    assert(!bn_gpu_moe_trellis_stats(&store, &stats) && !stats.decode.pending);
    BnGPUMoETrellisStore bad_source={0};TrellisStoreTestGPU failed={.store=&bad_source};
    BnGPUBackend bad_gpu={.ctx=&failed,.trellis_buffer_create=trellis_store_test_upload,.buffer_destroy=trellis_store_test_destroy};
    BnGPUMoETrellisPolicy ram_policy={.ram_admit_misses=1};
    assert(!bn_gpu_moe_trellis_store_init_policy(&bad_source,&model,&bad_gpu,39168,0,39168,39168,0,&ram_policy));
    for(int i=0;i<2;i++) {
        assert(bn_gpu_moe_trellis_acquire(&bad_source,0,0,0,&b)==-1&&!b.owner&&!failed.calls);
        assert(!bn_gpu_moe_trellis_stats(&bad_source,&stats)&&!stats.live_leases&&!stats.temporary.pending&&
            stats.host.gpu_host_cache_hits==0&&stats.host.io_count==0);
    }
    assert(!bn_gpu_moe_trellis_store_close(&bad_source));
    bn_safetensors_index_free(&index);
    assert(bn_gpu_moe_trellis_acquire(&store, 0, 0, 1, &a) == -1);
    assert(!bn_gpu_moe_trellis_store_close(&store));
    bn_model_free(&model);
    assert(!remove(path)); snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    assert(!remove(path) && !remove(dir));
    puts("Native expert placement budgets, policies, leases and rollback PASSED");
}
#else
static void test_native_trellis_store(void) {}
#endif
