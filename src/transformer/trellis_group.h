/* Layer-coordinated native requests. Activations/KV never change owner;
 * immutable expert handles are leased once for all consumers in a wave. */
typedef struct {
    TrellisRequest *r;
    int tokens,width[3],sms[3];
    BnGPURouterResidentPlan router;
    BnGPUNativeMoEIndexPlan index;
    BnGPUResidentView x,base,kind,slots,routed;
} TrellisGroupWork;
static int trellis_group_drain(TrellisGroupWork *w,int count,BnGPUMoETrellisLease *leases,int acquired) {
    int failed=0;
    for(int i=0;i<count;i++) {
        TrellisRequest *r=w[i].r;
        if(!r->request){failed=1;continue;}
        if(r->gpu->resident_synchronize(r->request)){r->destroy(r->request);r->request=NULL;failed=1;}
    }
    /* Every consumer is drained before any shared handle can be recycled. */
    for(int i=0;i<acquired;i++)if(bn_gpu_moe_trellis_release(w[0].r->store,leases+i))failed=1;
    return failed?-1:0;
}
static int trellis_group_router(TrellisGroupWork *w,TrellisRequest *r,int layer,int tokens) {
    const BnModelNativeNumerics *n=bn_model_native_numerics(r->model);BnGPUBackend *g=r->gpu;
    int D=r->anatomy->experts.dim,E=r->anatomy->experts.experts,K=r->anatomy->experts.active_experts;
    size_t A=(size_t)tokens*K;BnGPURouterWorkspace rw;
    void *weight=bn_backend_layout_native_router_weight(bn_model_backend(r->model),layer);
    if(!n||!weight||bn_gpu_router_workspace(tokens,D,E,&rw))return -1;
    memset(w,0,sizeof(*w));w->r=r;w->tokens=tokens;w->x=trellis_slice(r,TR_X,(size_t)tokens*D);
    w->router=(BnGPURouterResidentPlan){tokens,D,E,K,n->routed_scale,weight,w->x,
        trellis_slice(r,TR_LOGITS,(size_t)tokens*E),trellis_slice(r,TR_IDS,A),trellis_slice(r,TR_WEIGHTS,A),
        trellis_slice(r,TR_AHI,rw.a_count),trellis_slice(r,TR_ALO,rw.a_count),
        trellis_slice(r,TR_ASCALE,rw.scale_count),trellis_slice(r,TR_PARTIALS,rw.part_count)};
    return g->router_sqrtsp_resident(g->ctx,r->request,&w->router);
}
static int trellis_group_route(TrellisGroupWork *w,TrellisRequest *r,int layer,int tokens) {
    if(trellis_group_router(w,r,layer,tokens))return -1;
    BnGPUBackend *g=r->gpu;int D=r->anatomy->experts.dim,E=r->anatomy->experts.experts;
    int K=r->anatomy->experts.active_experts;size_t A=(size_t)tokens*K;
    w->index=(BnGPUNativeMoEIndexPlan){tokens,K,E,w->router.selected,w->router.weights,
        trellis_slice(r,TR_KEYS,A),trellis_slice(r,TR_VALUES,A),trellis_slice(r,TR_SORTED_KEYS,A),
        trellis_slice(r,TR_ORDER,A),trellis_slice(r,TR_INVERSE,A),trellis_slice(r,TR_SORTED_WEIGHTS,A),
        trellis_slice(r,TR_COUNTS,(size_t)E+1),trellis_slice(r,TR_STARTS,(size_t)E+1)};
    if(g->native_moe_index_resident(g->ctx,r->request,&w->index)||
       g->resident_read(r->request,w->index.counts,r->counts)||
       bn_moe_exl3_plan_groups(&r->plan,r->counts,E,tokens,K,D,r->anatomy->experts.hidden,256,256u*1024u*1024u)||
       r->plan.slot_count>r->workspace.slot_rows)return -1;
    w->base=trellis_slice(r,TR_BASE,E);w->kind=trellis_slice(r,TR_KIND,E);
    w->slots=trellis_slice(r,TR_SLOTS,(size_t)r->plan.slot_count*D);
    w->routed=trellis_slice(r,TR_ROUTED,(size_t)tokens*D);
    if(g->resident_write(r->request,w->base,r->plan.slot_base)||
       g->resident_write(r->request,w->kind,r->plan.slot_kind)||g->resident_zero(r->request,w->routed))return -1;
    const int low[]={33,17,1},high[]={256,32,16},tiles[]={64,32,16};
    for(int tier=0;tier<3;tier++) {
        int active=0;for(int e=0;e<E;e++)if(r->counts[e]>=low[tier]&&r->counts[e]<=high[tier])active++;
        if(!active)continue;
        BnGPUFusedEXL3LaunchInfo info;int tn=tiles[tier]==16&&!(D%256)&&!(r->anatomy->experts.hidden%256)?256:128;
        if(g->trellis_fused_launch_info(g->ctx,r->request,tiles[tier],tn,&info)||
           info.multiprocessors<8||info.blocks_per_sm!=1||info.max_groups<1)return -1;
        int groups=info.max_groups<active?info.max_groups:active;
        w->width[tier]=info.multiprocessors/groups;if(w->width[tier]>32)w->width[tier]=32;
        w->sms[tier]=info.multiprocessors;
    }
    memset(r->eligible,0,(size_t)E*sizeof(*r->eligible));return 0;
}
static int trellis_group_fused_plan(TrellisGroupWork *w,const int *experts,
    const BnGPUMoETrellisLease *leases,int count,int tier,BnGPUFusedEXL3ResidentPlan *p) {
    TrellisRequest *r=w->r;int D=r->anatomy->experts.dim,H=r->anatomy->experts.hidden;
    const int low[]={33,17,1},high[]={256,32,16},tiles[]={64,32,16};
    int selected[32],active=0;
    for(int j=0;j<count&&active<r->options.fused_window;j++) {
        int e=experts[j];
        if(!r->eligible[e]&&r->counts[e]>=low[tier]&&r->counts[e]<=high[tier])selected[active++]=j;
    }
    if(!active)return 0;
    int groups=w->sms[tier]/w->width[tier];if(groups>active)groups=active;
    BnGPUFusedEXL3Workspace ws;if(bn_gpu_fused_exl3_workspace(groups,256,D,H,&ws))return -1;
    memset(p,0,sizeof(*p));p->format=BN_QUANT_FORMAT_EXL3_MUL1_3;p->codebook_marker=UINT32_C(0x83dcd12d);
    p->tokens=w->tokens;p->topk=r->anatomy->experts.active_experts;p->experts=r->anatomy->experts.experts;
    p->dim=D;p->hidden=H;p->capacity=256;p->groups=groups;p->group_width=w->width[tier];
    p->tile_m=tiles[tier];p->tile_n=tiles[tier]==16&&!(D%256)&&!(H%256)?256:128;
    p->resident_count=active;p->slot_rows=r->plan.slot_count;p->act_limit=bn_model_native_numerics(r->model)->swiglu_limit;
    p->x=w->x;p->counts=w->index.counts;p->order=w->index.order;p->weights=w->index.sorted_weights;p->slot_base=w->base;p->slots=w->slots;
    p->state_g=trellis_slice(r,TR_STATE_G,ws.state_count);p->state_u=trellis_slice(r,TR_STATE_U,ws.state_count);
    p->intermediate_g=trellis_slice(r,TR_INTER_G,ws.intermediate_count);p->intermediate_u=trellis_slice(r,TR_INTER_U,ws.intermediate_count);
    for(int b=0;b<active;b++) {
        int j=selected[b];p->expert_ids[b]=experts[j];p->weight[0][b]=leases[j].buffers.gate;
        p->weight[1][b]=leases[j].buffers.up;p->weight[2][b]=leases[j].buffers.down;
    }
    return 1;
}
static int trellis_group_fused(TrellisGroupWork *work,int members,const int *experts,
    const BnGPUMoETrellisLease *leases,int count,BnTransformerTrellisGroupStats *stats) {
    for(int tier=0;tier<3;tier++)for(;;) {
        BnGPUFusedEXL3ResidentPlan plans[BN_TRANSFORMER_TRELLIS_GROUP_MAX];
        BnGPUFusedEXL3Consumer consumers[BN_TRANSFORMER_TRELLIS_GROUP_MAX];
        TrellisGroupWork *owners[BN_TRANSFORMER_TRELLIS_GROUP_MAX];int n=0;
        for(int i=0;i<members;i++) {
            int status=trellis_group_fused_plan(work+i,experts,leases,count,tier,plans+n);
            if(status<0)return -1;
            if(status){owners[n]=work+i;consumers[n]=(BnGPUFusedEXL3Consumer){work[i].r->request,plans+n};n++;}
        }
        if(!n)break;
        BnGPUBackend *g=owners[0]->r->gpu;
        if(n>1&&g->trellis_fused_group_resident) {
            BnGPUFusedEXL3GroupPlan group={n,consumers};BnGPUFusedEXL3GroupStats result;
            if(g->trellis_fused_group_resident(g->ctx,consumers[0].request,&group,&result))return -1;
            if(result.consumers!=(uint64_t)n||!result.kernel_launches||result.kernel_launches>(uint64_t)n||
               result.joint_kernel_launches>result.kernel_launches)return -1;
            stats->fused_launches+=result.kernel_launches;
            stats->joint_fused_launches+=result.joint_kernel_launches;
            stats->fused_launches_avoided+=result.consumers-result.kernel_launches;
            stats->individual_fused_launches+=result.kernel_launches-result.joint_kernel_launches;
        }else for(int i=0;i<n;i++) {
            TrellisRequest *r=owners[i]->r;
            stats->fused_launches++;stats->individual_fused_launches++;
            if(r->gpu->trellis_fused_resident(r->gpu->ctx,r->request,plans+i))return -1;
        }
        for(int i=0;i<n;i++)for(int b=0;b<plans[i].resident_count;b++)
            owners[i]->r->eligible[plans[i].expert_ids[b]]=1;
    }
    return 0;
}
static int trellis_group_folded(TrellisGroupWork *w,const BnMoEEXL3Group *group,
    const BnGPUMoETrellisLease *leases,BnTransformerTrellisGroupStats *stats) {
    TrellisRequest *r=w->r;BnGPUBackend *g=r->gpu;
    int D=r->anatomy->experts.dim,H=r->anatomy->experts.hidden,B=group->count,M=group->rows;
    if(B>r->workspace.batch_cap||B*M>r->workspace.group_rows)return -1;
    BnGPUNativeMoEGroupPlan gather={0};gather.tokens=w->tokens;gather.topk=r->anatomy->experts.active_experts;
    gather.dim=D;gather.batch=B;gather.rows=M;gather.x=w->x;gather.order=w->index.order;
    gather.out=trellis_slice(r,TR_GROUP_X,(size_t)B*M*D);
    for(int b=0;b<B;b++){int e=group->experts[b];gather.starts[b]=r->plan.starts[e];gather.counts[b]=r->counts[e];}
    BnGPUFoldedEXL3FFNResidentPlan p={0};p.activation_profile=BN_GPU_TRELLIS_FFN_SILU_HALF;
    p.act_limit=bn_model_native_numerics(r->model)->swiglu_limit;
    BnGPUFoldedEXL3ResidentPlan *projections[]={&p.gate,&p.up,&p.down};
    for(int pr=0;pr<3;pr++) {
        BnGPUFoldedEXL3ResidentPlan *v=projections[pr];v->format=BN_QUANT_FORMAT_EXL3_MUL1_3;v->codebook_marker=UINT32_C(0x83dcd12d);
        v->batch=B;v->n_tokens=M;v->k=pr==2?H:D;v->n=pr==2?D:H;
        v->decoded=trellis_slice(r,TR_DECODED,(size_t)B*D*H);
        for(int b=0;b<B;b++)v->weights[b]=pr==0?leases[b].buffers.gate:pr==1?leases[b].buffers.up:leases[b].buffers.down;
    }
    p.gate.x=p.up.x=gather.out;p.gate.out=trellis_slice(r,TR_GROUP_G,(size_t)B*M*H);
    p.up.out=p.down.x=trellis_slice(r,TR_GROUP_U,(size_t)B*M*H);
    p.down.out=bn_gpu_resident_view_slice(w->slots,(size_t)group->slot*D,(size_t)B*M*D);
    if(g->native_moe_group_resident(g->ctx,r->request,&gather))return -1;
    stats->folded_launches++;
    if(g->trellis_folded_ffn_resident(g->ctx,r->request,&p))return -1;
    for(int b=0;b<B;b++)r->eligible[group->experts[b]]=1;
    return 0;
}
static int trellis_group_acquire(TrellisGroupWork *w,int layer,int admit,const int *experts,int count,
    BnGPUMoETrellisLease *leases,int *acquired,BnTransformerTrellisGroupStats *stats) {
    *acquired=0;
    for(int j=0;j<count;j++) {
        int status=bn_gpu_moe_trellis_acquire(w->r->store,layer,experts[j],admit,leases+j);
        if(status)return status;
        (*acquired)++;stats->expert_borrows++;
    }
    return 0;
}
typedef struct {
    int experts[32],count,acquired;
    BnGPUMoETrellisLease leases[32];
} TrellisGroupWave;
static void trellis_group_next_wave(TrellisGroupWork *work,int members,int experts,
    int window,int *first,TrellisGroupWave *wave) {
    *wave=(TrellisGroupWave){0};
    while(*first<experts&&wave->count<window) {
        int selected=0,heavy=0;
        for(int i=0;i<members;i++) {
            int count=work[i].r->counts[*first];selected|=count>0;heavy|=count>256;
        }
        if(selected&&!heavy)wave->experts[wave->count++]=*first;
        (*first)++;
    }
}
static int trellis_group_load_wave(TrellisGroupWork *work,int layer,int admit,
    TrellisGroupWave *wave,int lookahead,BnTransformerTrellisGroupStats *stats) {
    while(wave->acquired<wave->count) {
        int j=wave->acquired;
        int status=bn_gpu_moe_trellis_acquire(work[0].r->store,layer,wave->experts[j],admit,wave->leases+j);
        if(status) {
            if(lookahead&&status==BN_GPU_UPLOAD_BUSY){stats->lookahead_busy_stops++;return 0;}
            return status;
        }
        wave->acquired++;stats->expert_borrows++;
        if(lookahead)stats->lookahead_borrows++;
    }
    return 0;
}
static int trellis_group_fused_waves(TrellisGroupWork *work,int members,int layer,
    int admit,int window,BnTransformerTrellisGroupStats *stats) {
    TrellisGroupWave waves[2]={0};int first=0,current=0,status;
    int experts=work[0].r->anatomy->experts.experts;
    trellis_group_next_wave(work,members,experts,window,&first,waves);
    while(waves[current].count) {
        TrellisGroupWave *wave=waves+current,*next=waves+(1-current);
        status=trellis_group_load_wave(work,layer,admit,wave,0,stats);
        if(status)goto failed;
        status=trellis_group_fused(work,members,wave->experts,wave->leases,wave->count,stats);
        if(status)goto failed;
        /* Keep the original next wave intact: only its ready prefix is loaded
         * while current consumers run. Reservations bound the prefix, and a
         * full cache defers the remainder until current leases are released. */
        trellis_group_next_wave(work,members,experts,window,&first,next);
        status=trellis_group_load_wave(work,layer,admit,next,1,stats);
        if(status)goto failed;
        int drained=trellis_group_drain(work,members,wave->leases,wave->acquired);
        wave->acquired=0;
        if(drained){status=-1;goto failed;}
        current=1-current;
    }
    return 0;
failed:
    /* Either array may be current. Drain every consumer before releasing
     * either set, including prefixes that have never reached compute. */
    if(trellis_group_drain(work,members,waves[0].leases,waves[0].acquired))status=-1;
    if(trellis_group_drain(work,members,waves[1].leases,waves[1].acquired))status=-1;
    return status;
}
static int trellis_group_shared(TrellisGroupWork *w,int layer) {
    TrellisRequest *r=w->r;BnGPUBackend *g=r->gpu;
    int D=r->anatomy->experts.dim,H=r->anatomy->experts.hidden;
    const int roles[]={BN_NATIVE_SHARED_GATE,BN_NATIVE_SHARED_UP,BN_NATIVE_SHARED_DOWN};void *weights[3];
    for(int p=0;p<3;p++) {
        const BnBackendNativeF16Weight *v=bn_backend_layout_native_core_weight(bn_model_backend(r->model),layer,roles[p]);
        if(!v||v->rows!=(p==2?D:H)||v->cols!=(p==2?H:D)){return -1;}
        weights[p]=v->weight;
    }
    size_t nx=(size_t)w->tokens*D,nh=(size_t)w->tokens*H;
    BnGPUResidentView sg=trellis_slice(r,TR_SHARED_G,nh),su=trellis_slice(r,TR_SHARED_U,nh),sa=trellis_slice(r,TR_SHARED_ACT,nh);
    BnGPUResidentView shared=trellis_slice(r,TR_SHARED,nx),out=trellis_slice(r,TR_OUT,nx);
    const BnModelNativeNumerics *numeric=bn_model_native_numerics(r->model);
    if(!numeric)return -1;
    BnGPUF16FFNResidentPlan dense={{weights[0],D,H,w->tokens,w->x,sg},{weights[1],D,H,w->tokens,w->x,su},
        {weights[2],H,D,w->tokens,sa,shared},BN_GPU_F16_FFN_SILU_HALF,numeric->swiglu_limit};
    BnGPUAddF32ResidentPlan add={nx,w->routed,shared,out};
    if(g->ffn_f16_resident(g->ctx,r->request,&dense)||g->add_f32_resident(g->ctx,r->request,&add)) {
        return -1;
    }
    return 0;
}
static int trellis_group_ffn(TrellisRequest **requests,int count,int layer,int admit,
    BnTransformerTrellisGroupStats *stats) {
    TrellisGroupWork work[BN_TRANSFORMER_TRELLIS_GROUP_MAX];int E=requests[0]->anatomy->experts.experts;
    memset(work,0,sizeof(work));
    for(int i=0;i<count;i++)work[i].r=requests[i];
    for(int i=0;i<count;i++)if(trellis_group_route(work+i,requests[i],layer,requests[i]->hc->tokens)) {
        (void)trellis_group_drain(work,count,NULL,0);return -1;
    }
    int window=32;for(int i=0;i<count;i++)if(requests[i]->options.fused_window<window)window=requests[i]->options.fused_window;
    int fused_status=trellis_group_fused_waves(work,count,layer,admit,window,stats);
    if(fused_status)return fused_status;
    /* Preserve original folded groups and B/M arithmetic. Fused consumers in
     * peers reuse these same handles; folded/folded overlaps may reload. */
    for(int owner=0;owner<count;owner++)for(int j=0;j<requests[owner]->plan.group_count;j++) {
        BnMoEEXL3Group *group=requests[owner]->plan.groups+j;
        BnGPUMoETrellisLease leases[32]={{0}};int acquired;
        int status=trellis_group_acquire(work,layer,admit,group->experts,group->count,leases,&acquired,stats);
        if(!status)status=trellis_group_folded(work+owner,group,leases,stats);
        if(!status)status=trellis_group_fused(work,count,group->experts,leases,group->count,stats);
        if(trellis_group_drain(work,count,leases,acquired))return -1;
        if(status)return status;
    }
    for(int i=0;i<count;i++) {
        TrellisGroupWork *w=work+i;TrellisRequest *r=requests[i];BnGPUBackend *g=r->gpu;
        int D=r->anatomy->experts.dim;
        for(int e=0;e<E;e++)if(r->counts[e]&&!r->eligible[e]){(void)trellis_group_drain(work,count,NULL,0);return -1;}
        BnGPUNativeMoEReducePlan reduce={w->tokens,r->anatomy->experts.active_experts,D,E,r->plan.slot_count,
            w->index.indices,w->index.inverse,w->index.starts,w->base,w->kind,w->index.sorted_weights,w->slots,w->routed};
        if(g->native_moe_reduce_resident(g->ctx,r->request,&reduce)||trellis_group_shared(w,layer)) {
            (void)trellis_group_drain(work,count,NULL,0);return -1;
        }
    }
    return trellis_group_drain(work,count,NULL,0);
}
static int trellis_group_decode_cohort(const BnGPUDecodeEXL3ResidentPlan *plans,int count,int first,
    int *experts,int *unique,int *bindings) {
    int n=0,end=first,references=0;
    for(;end<count;end++) {
        const BnGPUDecodeEXL3ResidentPlan *p=plans+end;int extra=0;
        for(int j=0;j<p->resident_count;j++) {
            int k=0;while(k<n&&experts[k]!=p->expert_ids[j])k++;
            if(k==n)extra++;
        }
        if(n+extra>BN_GPU_DECODE_EXL3_MAX_EXPERTS)break;
        references+=p->resident_count;
        for(int j=0;j<p->resident_count;j++) {
            int k=0;while(k<n&&experts[k]!=p->expert_ids[j])k++;
            if(k==n)experts[n++]=p->expert_ids[j];
        }
    }
    *unique=n;*bindings=references;return end;
}
/* Keep every source decode plan intact. Only immutable expert leases cross
 * request boundaries. A cohort union never exceeds the existing singleton
 * bound; overflow starts the next whole-request cohort, never a partial row. */
static int trellis_group_decode(TrellisRequest **requests,int count,int layer,int admit,
    BnTransformerTrellisGroupStats *stats) {
    TrellisGroupWork work[BN_TRANSFORMER_TRELLIS_GROUP_MAX]={0};
    BnGPUDecodeEXL3ResidentPlan plans[BN_TRANSFORMER_TRELLIS_GROUP_MAX];
    for(int i=0;i<count;i++)work[i].r=requests[i];
    for(int i=0;i<count;i++) {
        TrellisRequest *r=requests[i];int tokens=r->hc->tokens;
        const BnModelNativeNumerics *numeric=bn_model_native_numerics(r->model);
        if(!numeric||trellis_group_router(work+i,r,layer,tokens))goto failed;
        work[i].routed=trellis_slice(r,TR_ROUTED,(size_t)tokens*r->anatomy->experts.dim);
        if(trellis_decode_plan(r,tokens,&work[i].router,work[i].routed,numeric->swiglu_limit,plans+i))goto failed;
    }
    for(int first=0;first<count;) {
        int experts[BN_GPU_DECODE_EXL3_MAX_EXPERTS],n=0,references=0;
        int end=trellis_group_decode_cohort(plans,count,first,experts,&n,&references);
        if(end==first)goto failed;
        BnGPUMoETrellisLease leases[BN_GPU_DECODE_EXL3_MAX_EXPERTS]={{0}};int acquired=0,status=0;
        for(int j=0;j<n;j++) {
            status=bn_gpu_moe_trellis_acquire(requests[0]->store,layer,experts[j],admit,leases+j);
            if(status)break;
            acquired++;stats->expert_borrows++;
        }
        for(int i=first;!status&&i<end;i++) {
            TrellisRequest *r=requests[i];BnGPUDecodeEXL3ResidentPlan *p=plans+i;
            for(int j=0;j<p->resident_count;j++) {
                int k=0;while(k<n&&experts[k]!=p->expert_ids[j])k++;
                if(k==n){status=-1;break;}
                p->weight[0][j]=leases[k].buffers.gate;p->weight[1][j]=leases[k].buffers.up;p->weight[2][j]=leases[k].buffers.down;
            }
            if(!status) {
                status=p->resident_count?r->gpu->trellis_decode_resident(r->gpu->ctx,r->request,p):
                    r->gpu->resident_zero(r->request,work[i].routed);
                if(!status&&p->resident_count)stats->decode_submissions++;
            }
            if(!status)status=trellis_group_shared(work+i,layer);
        }
        if(!status){stats->decode_cohorts++;stats->decode_expert_reuses+=(uint64_t)(references-n);}
        if(trellis_group_drain(work,count,leases,acquired))return -1;
        if(status)return status;
        first=end;
    }
    return 0;
failed:
    (void)trellis_group_drain(work,count,NULL,0);return -1;
}
static int trellis_group_members(const BnModel *m,BnSession *const *sessions,int count,TrellisRequest **r) {
    if(!m||!sessions||count<2||count>BN_TRANSFORMER_TRELLIS_GROUP_MAX)return -1;
    for(int i=0;i<count;i++) {
        r[i]=trellis_request_get(m,sessions[i]);
        if(!r[i]||!r[i]->forward||!r[i]->hc||!r[i]->index||!r[i]->kv||!sessions[i]->state.logits)return -1;
        for(int j=0;j<i;j++)if(sessions[i]==sessions[j]||r[i]->request==r[j]->request)return -1;
    }
    return 0;
}
static int trellis_group_bound(BnSession *const *sessions,TrellisRequest **r,int count) {
    TrellisForward *first=r[0]->forward;
    if(first->group_owner!=sessions[0]||first->group_count!=count)return -1;
    for(int i=0;i<count;i++) {
        TrellisForward *f=r[i]->forward;
        if(f->group_owner!=sessions[0]||first->group_members[i]!=sessions[i]||
           first->group_generations[i]!=r[i]->generation||(f->phase!=4&&f->phase!=5)||
           f->layer!=first->layer||f->admit!=first->admit)return -1;
    }
    return 0;
}
static int trellis_group_discard(BnSession *const *sessions,int count) {
    for(int i=0;i<count;i++)bn_backend_session_release_gpu_resources(bn_session_backend(sessions[i]));
    return -1;
}
int bn_transformer_trellis_forward_group_cancel(const BnModel *m,BnSession *const *sessions,int count) {
    TrellisRequest *r[BN_TRANSFORMER_TRELLIS_GROUP_MAX];
    if(trellis_group_members(m,sessions,count,r)||trellis_group_bound(sessions,r,count))return -1;
    (void)trellis_group_discard(sessions,count);return 0;
}
int bn_transformer_trellis_forward_group(const BnModel *m,BnSession *const *sessions,int count,
    int admit,BnTransformerTrellisForwardResult *results,BnTransformerTrellisGroupStats *stats) {
    TrellisRequest *r[BN_TRANSFORMER_TRELLIS_GROUP_MAX];BnSessionNativeHistoryView history[BN_TRANSFORMER_TRELLIS_GROUP_MAX];
    if(!results||(admit!=0&&admit!=1)||trellis_group_members(m,sessions,count,r))return -1;
    int bound=r[0]->forward->group_owner!=NULL;
    if(bound&&trellis_group_bound(sessions,r,count))return -1;
    for(int i=0;i<count;i++) {
        TrellisForward *f=r[i]->forward;TrellisHC *h=r[i]->hc;
        if(bn_session_native_history_view(sessions[i],m,history+i))return -1;
        if(bound) {
            if(admit!=f->admit||history[i].epoch!=f->history.epoch||history[i].tokens!=f->history.tokens||
               history[i].position!=f->history.position||history[i].raw!=f->history.raw||history[i].mapped!=f->history.mapped)return -1;
        }else if(f->group_owner||f->phase||f->epoch==UINT64_MAX||r[i]->block_layer||
            history[i].tokens<1||history[i].tokens>f->workspace.max_tokens||
            history[i].tokens>r[i]->index->workspace.max_tokens||history[i].position>r[i]->kv->maxseq-history[i].tokens||
            r[i]->index->generation==UINT64_MAX||(h->phase!=0&&h->phase!=3)||
            h->epoch>UINT64_MAX-(uint64_t)(2*h->workspace.layers+1))return -1;
    }
    TrellisForward *first=r[0]->forward;
    if(!bound) {
        first->group_count=count;first->group_stats=(BnTransformerTrellisGroupStats){0};
        for(int i=0;i<count;i++) {
            TrellisForward *f=r[i]->forward;f->history=history[i];f->epoch++;f->admit=admit;f->layer=0;f->phase=4;
            f->group_owner=sessions[0];first->group_members[i]=sessions[i];first->group_generations[i]=r[i]->generation;
        }
    }
    first->group_stats.calls++;
    for(int i=0;i<count;i++)if(r[i]->forward->phase==4) {
        int status=bn_transformer_trellis_hc_begin(m,sessions[i]);
        if(status==BN_GPU_UPLOAD_BUSY)goto busy;
        if(status)goto fail;
        if(bn_transformer_trellis_index_begin(m,sessions[i],history[i].position,history[i].tokens))goto fail;
        r[i]->forward->phase=5;
    }
    while(first->layer<r[0]->hc->workspace.layers) {
        int layer=first->layer,shared=1,decode=1;
        for(int i=0;i<count;i++) {
            int status=trellis_block_prepare(m,sessions[i],layer,admit);
            if(status==BN_GPU_UPLOAD_BUSY)goto busy;
            if(status)goto fail;
            if(r[i]->hc->tokens<=8)shared=0;else decode=0;
        }
        if(shared) {
            int status=trellis_group_ffn(r,count,layer,admit,&first->group_stats);
            if(status==BN_GPU_UPLOAD_BUSY)goto busy;
            if(status)goto fail;
            first->group_stats.shared_layers++;
        }else if(decode) {
            int status=trellis_group_decode(r,count,layer,admit,&first->group_stats);
            if(status==BN_GPU_UPLOAD_BUSY)goto busy;
            if(status)goto fail;
            first->group_stats.shared_layers++;first->group_stats.decode_layers++;
        }else {
            for(int i=0;i<count;i++) {
                BnTransformerTrellisResult f;
                int status=bn_transformer_trellis_run_learned(m,sessions[i],layer,r[i]->hc->tokens,admit,&f);
                if(status==BN_GPU_UPLOAD_BUSY)goto busy;
                if(status)goto fail;
            }
            first->group_stats.individual_layers++;
        }
        for(int i=0;i<count;i++) {
            TrellisHC *h=r[i]->hc;BnTransformerTrellisHCResult v=trellis_hc_result(r[i],sessions[i]);
            BnGPUResidentView out=trellis_slice(r[i],TR_OUT,(size_t)h->tokens*r[i]->anatomy->experts.dim);
            if(bn_transformer_trellis_hc_apply(m,sessions[i],&v,out))goto fail;
            r[i]->block_layer=0;
        }
        for(int i=0;i<count;i++)r[i]->forward->layer++;
    }
    for(int i=0;i<count;i++) {
        TrellisForward *f=r[i]->forward;BnGPUResidentView collapsed,normalized;
        if(bn_transformer_trellis_hc_finish(m,sessions[i],&collapsed,&normalized))goto fail;
        int D=r[i]->anatomy->experts.dim,V=f->workspace.vocab,R=f->history.tokens,rows=f->all_logits?R:1;
        if(!f->all_logits)normalized=bn_gpu_resident_view_slice(normalized,(size_t)(R-1)*D,(size_t)D);
        BnGPUResidentView logits=trellis_forward_logits(f),last=bn_gpu_resident_view_slice(logits,(size_t)(rows-1)*V,(size_t)V);
        BnGPUF16LinearResidentPlan head={f->weight,D,V,rows,normalized,logits};
        if(r[i]->gpu->linear_f16_resident(r[i]->gpu->ctx,r[i]->request,&head)||
           r[i]->gpu->resident_read(r[i]->request,last,f->last))goto fail;
        for(int k=0;k<V;k++)if((f->last[k]&UINT16_C(0x7c00))==UINT16_C(0x7c00))goto fail;
    }
    /* The session module validates every ticket before publishing any one. */
    if(bn_session_native_history_commit_group(m,sessions,history,count))goto fail;
    for(int i=0;i<count;i++) {
        TrellisForward *f=r[i]->forward;int V=f->workspace.vocab;
        for(int k=0;k<V;k++)sessions[i]->state.logits[k]=bn_fp16_to_fp32(f->last[k]);
        f->phase=0;f->completed=f->epoch;f->group_owner=NULL;f->group_count=0;
        memset(f->group_members,0,sizeof(f->group_members));memset(f->group_generations,0,sizeof(f->group_generations));
    }
    for(int i=0;i<count;i++) {
        TrellisForward *f=r[i]->forward;
        results[i]=(BnTransformerTrellisForwardResult){r[i]->request,sessions[i],r[i]->generation,f->epoch,f->history.position,
            f->history.tokens,f->all_logits?f->history.tokens:1,f->workspace.vocab,trellis_forward_logits(f),sessions[i]->state.logits};
    }
    if(stats)*stats=first->group_stats;
    return 0;
busy:
    /* A failed helper may already have destroyed one participant. */
    for(int i=0;i<count;i++)if(trellis_request_get(m,sessions[i])!=r[i])goto fail;
    first->group_stats.waits++;if(stats)*stats=first->group_stats;return BN_GPU_UPLOAD_BUSY;
fail: {
    /* Do not dereference cached request pointers: a helper may have freed one. */
    (void)trellis_group_discard(sessions,count);return -1;
}
}
