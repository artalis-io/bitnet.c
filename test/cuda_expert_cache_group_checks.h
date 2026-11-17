#ifndef BN_TEST_CUDA_EXPERT_CACHE_GROUP_CHECKS_H
#define BN_TEST_CUDA_EXPERT_CACHE_GROUP_CHECKS_H
static void cache_group_round(BnGPUBackend *g,Request *r,Request *control,int gt,int dt,
    int layer,const int *bases,const int *prefill,int expected_hits,int teardown) {
    int off[2][513],ids[20],count=0;BnGPUExpertCacheGroupRow rows[2];
    for(int i=0;i<2;i++) {
        execute(control+i,g,gt,dt,bases[i],NULL);offsets(off[i],bases[i],r[i].nt);
        rows[i]=(BnGPUExpertCacheGroupRow){r[i].request,off[i],r[i].nt,prefill[i]};
        assert(!g->moe_bank_prepare_input(g->ctx,r[i].request,&r[i].input,&r[i].bank.prepared_input));
    }
    for(int e=0;e<512;e++)if(off[0][e+1]>off[0][e]||off[1][e+1]>off[1][e])ids[count++]=e;
    BnGPUExpertCacheLayer ticket;int status;
    do {status=g->expert_cache_begin_group(r[0].request,layer,rows,2,&ticket);if(status==BN_GPU_MOE_BANK_BUSY)thrd_yield();}while(status==BN_GPU_MOE_BANK_BUSY);
    assert(!status);int hits=0;for(int e=0;e<512;e++)hits+=(int)((ticket.hits[e/64]>>(e%64))&1);
    assert(hits==expected_hits);
    BnGPUMoEBankStats before,after;assert(!g->moe_bank_stats(r[0].request,&before));
    size_t expert_bytes=0;
    for(int first=0;first<count;first+=8) {
        int n=count-first;if(n>8)n=8;
        BnGPUMoEBankGroupPlan p={0};p.cache_generation=ticket.generation;p.n_experts=n;
        do {status=g->moe_bank_acquire(r[0].request,(unsigned)(first/8)%2,gt,dt,&p.lease);if(status==BN_GPU_MOE_BANK_BUSY)thrd_yield();}while(status==BN_GPU_MOE_BANK_BUSY);
        assert(!status);expert_bytes=p.lease.expert_bytes;
        memset(p.lease.data,0,(size_t)n*expert_bytes); /* Hits must ignore poisoned staging. */
        for(int j=0;j<n;j++) {
            int e=ids[first+j];if((ticket.hits[e/64]>>(e%64))&1)continue;
            unsigned char *raw=(unsigned char*)p.lease.data+(size_t)j*expert_bytes;
            fill(raw,p.lease.gate_bytes,gt,e);fill(raw+p.lease.gate_bytes,p.lease.gate_bytes,gt,e);
            fill(raw+2*p.lease.gate_bytes,p.lease.down_bytes,dt,e);
        }
        BnGPUMoEBankItem items[2][8];BnGPUMoEBankConsumerPlan consumers[2];int used=0;
        for(int i=0;i<2;i++) {
            int active=0;
            for(int j=0;j<n;j++){int e=ids[first+j];items[used][j]=(BnGPUMoEBankItem){e,off[i][e],off[i][e+1]-off[i][e]};active+=items[used][j].n_rows>0;}
            if(!active)continue;
            consumers[used]=(BnGPUMoEBankConsumerPlan){.request=r[i].request,.workspace_slot=(unsigned)(first/8)%2,
                .total_tokens=r[i].nt,.experts=items[used],.x=r[i].bank.x,.gate=r[i].bank.gate,
                .up=r[i].bank.up,.out=r[i].bank.out,.prepared_input=r[i].bank.prepared_input};used++;
        }
        p.n_consumers=used;p.consumers=consumers;
        if(!first) {
            BnGPUMoEBankGroupPlan bad=p;bad.cache_generation++;assert(g->moe_bank_submit_group(g->ctx,r[0].request,&bad)==-1);
            bad=p;if(used==2){bad.n_consumers=1;assert(g->moe_bank_submit_group(g->ctx,r[0].request,&bad)==-1);}
            assert(g->expert_cache_end(r[0].request,&ticket)==-1); /* Not all union experts queued. */
            assert(!g->moe_bank_stats(r[0].request,&after)&&!memcmp(&before,&after,sizeof(before)));
        }
        do {status=g->moe_bank_submit_group(g->ctx,r[0].request,&p);if(status==BN_GPU_MOE_BANK_BUSY)thrd_yield();}while(status==BN_GPU_MOE_BANK_BUSY);
        assert(!status);
    }
    if(teardown) {int i=teardown-1;bn_backend_session_free(r[i].session);r[i].session=NULL;}
    else assert(!g->expert_cache_end(r[0].request,&ticket));
    for(int i=0;i<2;i++)if(r[i].session) {
        size_t bytes=(size_t)r[i].nt*10*256*2;uint16_t *actual=malloc(bytes),*expected=malloc(bytes);assert(actual&&expected);
        assert(!g->resident_read(r[i].request,r[i].bank.out,actual)&&!g->resident_read(control[i].request,control[i].bank.out,expected));
        assert(!memcmp(actual,expected,bytes));free(actual);free(expected);
    }
    if(!teardown) {
        assert(!g->moe_bank_stats(r[0].request,&after));
        assert(after.upload_bytes-before.upload_bytes==(size_t)(count-hits)*expert_bytes);
        assert(after.joint_kernel_launches>before.joint_kernel_launches);
        BnGPUExpertCacheStats stats;assert(!g->expert_cache_stats(r[0].request,&stats));assert(stats.resident_bytes<=stats.reserved_device_bytes);
    }
}
static void cache_group_cases(BnGPUBackend *g,int gt,int dt,int matrix,int capacity) {
    Request control[2],r[2];int nt[]={matrix?65:1,matrix?1:65};
    for(int i=0;i<2;i++)init(control+i,g,nt[i],NULL);
    BnGPUMoEBankLease lease;assert(!g->moe_bank_acquire(control[0].request,0,gt,dt,&lease));
    size_t bytes=lease.expert_bytes;assert(!g->moe_bank_cancel(control[0].request,&lease));
    int gate[]={gt,gt},down[]={dt,dt};BnGPUExpertCacheConfig cfg={2,512,256,32,gate,down,(size_t)capacity*bytes};
    void *cache=g->expert_cache_create(g->ctx,&cfg);assert(cache);
    for(int i=0;i<2;i++)init(r+i,g,nt[i],cache);
    int off[2][513];for(int i=0;i<2;i++)offsets(off[i],i*5,nt[i]);
    BnGPUExpertCacheGroupRow rows[]={{r[0].request,off[0],nt[0],1},{r[1].request,off[1],nt[1],0}},bad[2];
    BnGPUExpertCacheLayer out,sentinel;memset(&sentinel,0x5a,sizeof(sentinel));out=sentinel;
    assert(g->expert_cache_begin_group(r[0].request,0,rows,0,&out)==-1&&!memcmp(&out,&sentinel,sizeof(out)));
    assert(g->expert_cache_begin_group(r[0].request,0,rows,17,&out)==-1);
    memcpy(bad,rows,sizeof(bad));bad[1].request=r[0].request;assert(g->expert_cache_begin_group(r[0].request,0,bad,2,&out)==-1);
    memcpy(bad,rows,sizeof(bad));bad[1].prefill=2;assert(g->expert_cache_begin_group(r[0].request,0,bad,2,&out)==-1);
    off[1][512]--;assert(g->expert_cache_begin_group(r[0].request,0,rows,2,&out)==-1);off[1][512]++;
    assert(!memcmp(&out,&sentinel,sizeof(out)));
    assert(!g->expert_cache_begin_group(r[0].request,0,rows,2,&out));
    assert(g->expert_cache_begin(r[1].request,1,0,off[1],nt[1],&sentinel)==-1);
    assert(!g->expert_cache_cancel(r[1].request));assert(g->expert_cache_end(r[0].request,&out)==-1);
    int bases[]={0,5},prefill[]={1,0},borrow[]={1,1},decode[]={0,0};
    cache_group_round(g,r,control,gt,dt,0,bases,prefill,0,0);
    int retained=capacity<10?capacity:10;
    cache_group_round(g,r,control,gt,dt,0,bases,borrow,retained,0);
    cache_group_round(g,r,control,gt,dt,0,bases,decode,retained,0);
    retained=capacity<15?capacity:15;
    cache_group_round(g,r,control,gt,dt,0,bases,decode,retained,0);
    BnGPUExpertCacheStats stats;assert(!g->expert_cache_stats(r[0].request,&stats));
    if(capacity<15)assert(stats.bypasses);else assert(stats.resident_experts==15);
    int other[]={30,35};cache_group_round(g,r,control,gt,dt,1,other,decode,0,0);
    assert(!g->expert_cache_stats(r[0].request,&stats)&&stats.evictions);
    cache_group_round(g,r,control,gt,dt,0,bases,borrow,capacity==20?5:0,0);
    for(int i=0;i<2;i++)bn_backend_session_free(r[i].session);
    /* Participant-first and owner-first teardown of an open, submitted union
     * must release all pins while retaining the model cache for later use. */
    for(int order=1;order<=2;order++) {
        for(int i=0;i<2;i++)init(r+i,g,nt[i],cache);
        int unseen[]={80+order*30,85+order*30};cache_group_round(g,r,control,gt,dt,0,unseen,decode,0,order);
        int alive=order==1?1:0;int single[513];offsets(single,unseen[alive],nt[alive]);BnGPUExpertCacheLayer ticket;int status;
        do {status=g->expert_cache_begin(r[alive].request,0,1,single,nt[alive],&ticket);if(status==BN_GPU_MOE_BANK_BUSY)thrd_yield();}while(status==BN_GPU_MOE_BANK_BUSY);
        assert(!status&&!g->expert_cache_cancel(r[alive].request));bn_backend_session_free(r[alive].session);
    }
    for(int i=0;i<2;i++)bn_backend_session_free(control[i].session);
    g->expert_cache_destroy(cache);
}
static void cache_group_checks(BnGPUBackend *g) {
    assert(g->expert_cache_begin_group);
    int gate[]={BN_GGUF_TENSOR_Q4_K,BN_GGUF_TENSOR_Q5_K,BN_GGUF_TENSOR_Q4_K};
    int down[]={BN_GGUF_TENSOR_Q5_1,BN_GGUF_TENSOR_Q8_0,BN_GGUF_TENSOR_Q8_0};
    for(int f=0;f<3;f++)for(int matrix=0;matrix<2;matrix++)for(int capacity=5;capacity<=20;capacity+=15)
        cache_group_cases(g,gate[f],down[f],matrix,capacity);
    puts("CUDA joint cache PASSED:12 format/geometry/capacity cases, union pins, mixed policies, poisoned warm staging, zero warm H2D, eviction, stale/invalid leases and both teardown orders");
}
#endif
