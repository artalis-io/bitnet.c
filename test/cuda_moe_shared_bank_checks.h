#ifndef BN_TEST_MOE_SHARED_BANK_CHECKS_H
#define BN_TEST_MOE_SHARED_BANK_CHECKS_H
/* Nonzero differential and asynchronous source-lifetime tests. Each request
 * has different inputs, grouping and vector/matrix geometry. */
typedef struct {
    BnBackendSession *session;
    void *request;
    BnGPUMoEBankConsumerPlan consumer;
    BnGPUMoEBankItem items[3];
    uint16_t *data, *expected;
    size_t count;
} SharedBankCase;
static void shared_payload(BnGPUMoEBankLease lease,const int *ids,int count,int generation) {
    for(int e=0;e<count;e++) {
        unsigned char *raw=(unsigned char*)lease.data+(size_t)e*lease.expert_bytes;
        int seed=ids[e]+generation*3;
        matrix_payload(raw,lease.gate_bytes,lease.gate_type,seed);
        matrix_payload(raw+lease.gate_bytes,lease.gate_bytes,lease.gate_type,seed+1);
        matrix_payload(raw+2*lease.gate_bytes,lease.down_bytes,lease.down_type,seed+2);
    }
}
static void shared_baseline(BnGPUBackend *g,SharedBankCase *c,int gt,int dt,int generation,uint16_t *out) {
    BnGPUMoEBankResidentPlan p={0};BnGPUMoEBankItem items[3];int ids[3],count=0;
    for(int e=0;e<3;e++)if(c->items[e].n_rows){items[count]=c->items[e];ids[count++]=c->items[e].expert;}
    int status;
    do {status=g->moe_bank_acquire(c->request,0,gt,dt,&p.lease);if(status==BN_GPU_MOE_BANK_BUSY)thrd_yield();}while(status==BN_GPU_MOE_BANK_BUSY);
    assert(!status);shared_payload(p.lease,ids,count,generation);
    p.total_tokens=c->consumer.total_tokens;p.n_experts=count;p.experts=items;
    p.x=c->consumer.x;p.gate=c->consumer.gate;p.up=c->consumer.up;p.out=c->consumer.out;
    p.grouped=1;p.prepared_input=c->consumer.prepared_input;p.clamped=c->consumer.clamped;p.clamp_limit=c->consumer.clamp_limit;
    for(size_t i=0;i<c->count;i++)c->data[i]=0x5a5a;
    assert(!g->resident_write(c->request,p.out,c->data));
    assert(!g->moe_bank_submit(g->ctx,c->request,&p));
    assert(!g->resident_read(c->request,p.out,out));
}
static void shared_banks(BnGPUBackend *g) {
    assert(g->moe_bank_submit_group);int cases=0;
    for(int profile=0;profile<2;profile++)for(int gt=0;gt<2;gt++)for(int dt=0;dt<4;dt++) {
        int experts=profile?288:512,k=profile?8:10,hidden=256;
        int gate=gt?BN_GGUF_TENSOR_Q5_K:BN_GGUF_TENSOR_Q4_K;
        int down=dt==3?BN_GGUF_TENSOR_Q6_K:dt==2?BN_GGUF_TENSOR_Q5_K:dt?BN_GGUF_TENSOR_Q8_0:BN_GGUF_TENSOR_Q5_1;
        int ids[]={7,experts-1,64},tokens[]={1,65,9,65,65};SharedBankCase c[5]={0};
        for(int i=0;i<5;i++) {
            c[i].session=bn_backend_session_create();assert(c[i].session);
            void *r=c[i].request=bn_gpu_backend_resident_request(g,c[i].session);assert(r);
            size_t payload=(size_t)hidden*624*3*2;
            BnGPUMoEBankConfig cfg={2,3,256,hidden,payload,payload,experts,k};
            assert(!g->moe_bank_reserve(g->ctx,r,&cfg));
            assert(!g->moe_bank_input_reserve(g->ctx,r,tokens[i],(size_t)tokens[i]*(k+1)*8*36));
            assert(!g->moe_bank_grouped_reserve(g->ctx,r,tokens[i],72,(size_t)tokens[i]*k*8*36+72));
            assert(!g->linear_a8_bf16_reserve(g->ctx,r,(size_t)tokens[i]*256));
            int rows=tokens[i]*k;c[i].count=(size_t)rows*256;
            c[i].data=malloc(c[i].count*2);c[i].expected=malloc(c[i].count*2);assert(c[i].data&&c[i].expected);
            for(size_t j=0;j<c[i].count;j++)c[i].data[j]=(uint16_t)((j%5?0x3f80:0xbf80)+(j+i*7)%19);
            BnGPUMoEBankInputPlan prep={0};prep.n_tokens=tokens[i];prep.x=alloc(g,r,(size_t)tokens[i]*256);
            assert(!g->resident_write(r,prep.x,c[i].data));
            assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,rows,&prep.slots));
            assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,1,&prep.status));
            int32_t *slots=malloc((size_t)rows*4),zero=0;assert(slots);
            for(int j=0;j<rows;j++)slots[j]=rows-1-j;
            assert(!g->resident_write(r,prep.slots,slots)&&!g->resident_write(r,prep.status,&zero));free(slots);
            uint64_t generation;assert(!g->moe_bank_prepare_input(g->ctx,r,&prep,&generation));
            BnGPUMoEBankConsumerPlan *v=&c[i].consumer;
            v->request=r;v->total_tokens=tokens[i];v->prepared_input=generation;v->experts=c[i].items;
            v->clamped=i&1;v->clamp_limit=10;v->x=alloc(g,r,c[i].count);v->out=alloc(g,r,c[i].count);
            v->gate=alloc(g,r,c[i].count);v->up=alloc(g,r,c[i].count);
            for(int e=0;e<3;e++)c[i].items[e]=(BnGPUMoEBankItem){ids[e],0,0};
        }
        c[0].items[0]=(BnGPUMoEBankItem){ids[0],0,1};c[0].items[2]=(BnGPUMoEBankItem){ids[2],3,1};
        c[1].items[0]=(BnGPUMoEBankItem){ids[0],3,25};c[1].items[1]=(BnGPUMoEBankItem){ids[1],77,9};
        c[2].items[1]=(BnGPUMoEBankItem){ids[1],2,3};c[2].items[2]=(BnGPUMoEBankItem){ids[2],22,7};
        c[3].items[0]=(BnGPUMoEBankItem){ids[0],5,8};
        c[4].items[0]=(BnGPUMoEBankItem){ids[0],11,24};
        for(int i=0;i<5;i++)shared_baseline(g,c+i,gate,down,0,c[i].expected);
        BnGPUMoEBankConsumerPlan consumers[]={c[0].consumer,c[1].consumer,c[2].consumer,c[3].consumer,c[4].consumer};
        BnGPUMoEBankGroupPlan p={0};p.n_experts=3;p.n_consumers=5;p.consumers=consumers;
        assert(!g->moe_bank_acquire(c[0].request,0,gate,down,&p.lease));shared_payload(p.lease,ids,3,0);
        BnGPUMoEBankStats before,after;assert(!g->moe_bank_stats(c[0].request,&before));
#define SHARED_REJECT(change) do { BnGPUMoEBankGroupPlan bad=p;change;assert(g->moe_bank_submit_group(g->ctx,c[0].request,&bad)==-1); }while(0)
        SHARED_REJECT(bad.n_consumers=0);SHARED_REJECT(bad.n_consumers=17);SHARED_REJECT(bad.n_experts=4);
        SHARED_REJECT(bad.lease.generation++);SHARED_REJECT(bad.lease.data=NULL);SHARED_REJECT(bad.consumers=NULL);
        BnGPUMoEBankConsumerPlan saved=consumers[1];
        consumers[1].prepared_input++;SHARED_REJECT((void)0);consumers[1]=saved;
        consumers[1].request=c[0].request;SHARED_REJECT((void)0);consumers[1]=saved;
        consumers[1].out=consumers[1].x;SHARED_REJECT((void)0);consumers[1]=saved;
        consumers[1].clamped=1;consumers[1].clamp_limit=NAN;SHARED_REJECT((void)0);consumers[1]=saved;
        c[1].items[1].expert--;SHARED_REJECT((void)0);c[1].items[1].expert++;
        c[1].items[1].first_row=3;SHARED_REJECT((void)0);c[1].items[1].first_row=77;
#undef SHARED_REJECT
        BnGPUMoEBankLease blocked;assert(!g->moe_bank_acquire(c[1].request,0,gate,down,&blocked));
        assert(g->moe_bank_submit_group(g->ctx,c[0].request,&p)==BN_GPU_MOE_BANK_BUSY);
        assert(!g->moe_bank_stats(c[0].request,&after)&&!memcmp(&before,&after,sizeof(before)));
        assert(!g->moe_bank_cancel(c[1].request,&blocked));
        assert(!g->moe_bank_submit_group(g->ctx,c[0].request,&p));
        assert(g->moe_bank_cancel(c[0].request,&p.lease)==-1);
        for(int i=0;i<5;i++) {
            assert(!g->resident_read(c[i].request,c[i].consumer.out,c[i].data));
            assert(!memcmp(c[i].data,c[i].expected,c[i].count*2));
        }
        assert(!g->moe_bank_stats(c[0].request,&after));
        assert(after.upload_bytes-before.upload_bytes==3*p.lease.expert_bytes&&after.shared_uploads==1&&after.shared_consumers==5);
        assert(after.shared_upload_bytes_avoided==5*p.lease.expert_bytes);
        assert(after.joint_kernel_launches==10&&after.joint_projection_launches==8&&after.joint_launches_avoided==10);
        /* Source owner need not consume its own bank. Peer outputs must still
         * be ordered before later work on their private streams. */
        assert(!g->moe_bank_acquire(c[0].request,1,gate,down,&p.lease));shared_payload(p.lease,ids,3,0);
        p.n_consumers=4;p.consumers=consumers+1;
        assert(!g->moe_bank_submit_group(g->ctx,c[0].request,&p));
        for(int i=1;i<5;i++) {
            assert(!g->resident_read(c[i].request,c[i].consumer.out,c[i].data));
            assert(!memcmp(c[i].data,c[i].expected,c[i].count*2));
        }
        /* Reuse a source across alternating cohorts, snapshotting outputs on
         * each consumer stream before any host readback. */
        enum { STEPS=4 };BnGPUResidentView snapshots[STEPS][2];uint16_t *gold[STEPS];
        for(int step=0;step<STEPS;step++) {
            int i=1+(step&1);gold[step]=malloc(c[i].count*2);assert(gold[step]);
            shared_baseline(g,c+i,gate,down,step+1,gold[step]);
            snapshots[step][0]=alloc(g,c[i].request,c[i].count/2);snapshots[step][1]=alloc(g,c[i].request,c[i].count/2);
        }
        for(int step=0;step<STEPS;step++) {
            int i=1+(step&1),status;BnGPUMoEBankConsumerPlan pair[]={c[0].consumer,c[i].consumer};
            do {status=g->moe_bank_acquire(c[0].request,0,gate,down,&p.lease);if(status==BN_GPU_MOE_BANK_BUSY)thrd_yield();}while(status==BN_GPU_MOE_BANK_BUSY);
            assert(!status);shared_payload(p.lease,ids,3,step+1);p.n_consumers=2;p.consumers=pair;
            do {status=g->moe_bank_submit_group(g->ctx,c[0].request,&p);if(status==BN_GPU_MOE_BANK_BUSY)thrd_yield();}while(status==BN_GPU_MOE_BANK_BUSY);
            assert(!status);
            BnGPUSplitColumnsResidentPlan split={.rows=tokens[i]*k,.cols=256,.left_cols=128,.x=c[i].consumer.out,.left=snapshots[step][0],.right=snapshots[step][1]};
            assert(!g->split_columns_resident(g->ctx,c[i].request,&split));
        }
        /* Source teardown must drain every borrowing request, even though the
         * most recent source request has only one token. */
        bn_backend_session_free(c[0].session);c[0].session=NULL;
        for(int step=0;step<STEPS;step++) {
            int i=1+(step&1);
            for(int half=0;half<2;half++) {
                assert(!g->resident_read(c[i].request,snapshots[step][half],c[i].data));
                for(size_t row=0;row<c[i].count/256;row++)
                    assert(!memcmp(c[i].data+row*128,gold[step]+row*256+half*128,256));
            }
            free(gold[step]);
        }
        for(int i=0;i<5;i++){if(c[i].session)bn_backend_session_free(c[i].session);free(c[i].data);free(c[i].expected);}
        cases++;
    }
    printf("CUDA shared MoE banks PASSED: %d format/geometry cases, independent inputs, BUSY, alternating cohorts and source-first teardown\n",cases);
}
#endif
