/* Joint execution must match unchanged source-qualified single submissions,
 * including untouched output slots, private guards, dirty scheduler reuse,
 * asymmetric residency groups and either stream as coordinator. */
static void fused_exl3_group_compare(BnGPUBackend *g,void **requests,
    BnGPUFusedEXL3ResidentPlan *plans,const unsigned char *poison,size_t bytes) {
    assert(g->trellis_fused_group_resident);
    unsigned char *expected[2]={malloc(bytes),malloc(bytes)},*actual=malloc(bytes);
    assert(expected[0]&&expected[1]&&actual);
    for(int i=0;i<2;i++) {
        assert(!g->trellis_fused_resident(g->ctx,requests[i],plans+i));
        assert(!g->resident_read(requests[i],plans[i].slots,expected[i]));
        assert(!g->resident_write(requests[i],plans[i].slots,poison));
    }
    BnGPUFusedEXL3Consumer consumers[2]={{requests[0],plans},{requests[1],plans+1}};
    BnGPUFusedEXL3GroupPlan group={2,consumers},bad_group;
    BnGPUFusedEXL3GroupStats stats,canary;memset(&canary,0xa5,sizeof(canary));
#define FUSED_GROUP_BAD(BAD) do { stats=canary;assert(g->trellis_fused_group_resident(g->ctx,requests[0],&(BAD),&stats)==-1);assert(!memcmp(&stats,&canary,sizeof(stats))); } while(0)
    bad_group=group;bad_group.n_consumers=1;FUSED_GROUP_BAD(bad_group);
    bad_group=group;bad_group.n_consumers=BN_GPU_FUSED_EXL3_MAX_CONSUMERS+1;FUSED_GROUP_BAD(bad_group);
    bad_group=group;bad_group.consumers=NULL;FUSED_GROUP_BAD(bad_group);
    consumers[1]=consumers[0];FUSED_GROUP_BAD(group);
    consumers[1]=(BnGPUFusedEXL3Consumer){requests[1],plans+1};
    BnGPUFusedEXL3ResidentPlan bad=plans[1];bad.weight[2][bad.resident_count-1]=NULL;
    consumers[1].plan=&bad;FUSED_GROUP_BAD(group);
    bad=plans[1];bad.x=plans[0].x;FUSED_GROUP_BAD(group);
    consumers[1].plan=plans+1;
#undef FUSED_GROUP_BAD
    for(int i=0;i<2;i++) {
        assert(!g->resident_read(requests[i],plans[i].slots,actual));assert(!memcmp(actual,poison,bytes));
    }
    for(int owner=0;owner<2;owner++) {
        /* Writes are queued on the private streams. The joint owner must
         * wait for both, and both readers must wait for the joint output. */
        for(int i=0;i<2;i++)assert(!g->resident_write(requests[i],plans[i].slots,poison));
        assert(!g->trellis_fused_group_resident(g->ctx,requests[owner],&group,&stats));
        assert(stats.consumers==2&&stats.kernel_launches==1&&stats.joint_kernel_launches==1);
        for(int i=0;i<2;i++) {
            assert(!g->resident_read(requests[i],plans[i].slots,actual));assert(!memcmp(actual,expected[i],bytes));
        }
    }
    for(int i=0;i<2;i++)assert(!g->resident_write(requests[i],plans[i].slots,poison));
    free(actual);free(expected[1]);free(expected[0]);
}

static void fused_exl3_group_wide(BnGPUBackend *g,const BnGPUFusedEXL3ResidentPlan *source) {
    enum { N=16, V=10 };
    void *requests[N];BnGPUFusedEXL3ResidentPlan plans[N];BnGPUResidentView guards[N][V];
    BnGPUFusedEXL3Consumer consumers[N];unsigned char *expected[N];
    const BnGPUResidentView inputs[]={source->x,source->counts,source->order,source->weights,source->slot_base};
    void *host[5];
    for(int j=0;j<5;j++) {
        size_t width=inputs[j].type==BN_GGUF_TENSOR_F16?2:4;
        host[j]=malloc(inputs[j].count*width);assert(host[j]);
        assert(!g->resident_read(inputs[j].request,inputs[j],host[j]));
    }
    size_t bytes=source->slots.count*4;unsigned char *actual=malloc(bytes),*poison=malloc(bytes);
    assert(actual&&poison);memset(poison,0xa5,bytes);
    for(int i=0;i<N;i++) {
        requests[i]=g->resident_request_create(g->ctx);assert(requests[i]);
        assert(!g->trellis_fused_reserve(g->ctx,requests[i]));plans[i]=*source;
        plans[i].group_width=i<8?8:32;
        BnGPUResidentView *views[]={&plans[i].x,&plans[i].counts,&plans[i].order,&plans[i].weights,&plans[i].slot_base,
            &plans[i].state_g,&plans[i].state_u,&plans[i].intermediate_g,&plans[i].intermediate_u,&plans[i].slots};
        for(int j=0;j<V;j++) {
            BnGPUResidentView old=*views[j];void *data=j<5?host[j]:NULL;uint16_t *private_x=NULL;
            if(!j) {
                private_x=malloc(old.count*2);assert(private_x);
                for(size_t k=0;k<old.count;k++)private_x[k]=bn_fp32_to_fp16(
                    bn_fp16_to_fp32(((uint16_t*)host[0])[k])*(i&1?-1.0f:1.0f+(float)i/16.0f));
                data=private_x;
            }
            *views[j]=native_router_guard(g,requests[i],old.type,old.count,data,&guards[i][j]);free(private_x);
        }
        expected[i]=malloc(bytes);assert(expected[i]);
        assert(!g->trellis_fused_resident(g->ctx,requests[i],plans+i));
        assert(!g->resident_read(requests[i],plans[i].slots,expected[i]));
        assert(!g->resident_write(requests[i],plans[i].slots,poison));
        consumers[i]=(BnGPUFusedEXL3Consumer){requests[i],plans+i};
    }
    BnGPUFusedEXL3GroupPlan group={N,consumers};BnGPUFusedEXL3GroupStats stats,canary;
    memset(&canary,0xa5,sizeof(canary));stats=canary;
    BnGPUFusedEXL3ResidentPlan bad=plans[N-1];bad.weight[2][bad.resident_count-1]=NULL;
    consumers[N-1].plan=&bad;
    assert(g->trellis_fused_group_resident(g->ctx,requests[N-1],&group,&stats)==-1);
    assert(!memcmp(&stats,&canary,sizeof(stats)));
    assert(!g->resident_read(requests[0],plans[0].slots,actual)&&!memcmp(actual,poison,bytes));
    consumers[N-1].plan=plans+N-1;
    for(int pass=0;pass<2;pass++) {
        for(int i=0;i<N;i++)assert(!g->resident_write(requests[i],plans[i].slots,poison));
        assert(!g->trellis_fused_group_resident(g->ctx,requests[pass?0:N-1],&group,&stats));
        /* Width8: one batch of8. Width32: residency permits5 then3.
         * A private width/tile never changes to fit the joint launch. */
        assert(stats.consumers==N&&stats.kernel_launches==3&&stats.joint_kernel_launches==3);
        for(int i=0;i<N;i++) {
            assert(!g->resident_read(requests[i],plans[i].slots,actual));assert(!memcmp(actual,expected[i],bytes));
        }
    }
    for(int i=0;i<N;i++) {
        for(int j=0;j<V;j++) {
            BnGPUResidentView v=guards[i][j];size_t width=v.type==BN_GGUF_TENSOR_F16?2:4;
            unsigned char *raw=malloc(v.count*width);assert(raw&&!g->resident_read(requests[i],v,raw));
            for(size_t k=0;k<16*width;k++)assert(raw[k]==0xa5&&raw[(v.count-16)*width+k]==0xa5);
            free(raw);
        }
    }
    for(int i=0;i<N;i++){free(expected[i]);g->resident_request_destroy(requests[i]);}
    for(int j=0;j<5;j++)free(host[j]);
    free(poison);free(actual);
    puts("CUDA joint EXL3:16 private histories, mixed widths, residency partitions, dirty reuse and rejection atomicity PASSED");
}
