#ifndef BN_TRANSFORMER_BF16_SHARED_H
#define BN_TRANSFORMER_BF16_SHARED_H
/* Transformer-owned cohort orchestration; bank resources remain backend-owned.
 * Family adapters supply routing views and CPU sources. Union ordering, bank
 * lifetimes, backpressure and accounting are common to both adapters. */
typedef struct {
    void *request, *source;
    const int32_t *offsets;
    BnGPUMoEBankConsumerPlan plan;
    int prefill;
    size_t banks, experts;
} BF16SharedConsumer;
typedef struct {
    int n_experts, capacity, gate_type, down_type;
    size_t gate_bytes, down_bytes;
    int (*read)(void *source,int expert,const BnGPUMoEBankLease *lease,void *dst);
    const uint64_t *cache_hits;
    uint64_t cache_generation;
} BF16SharedSource;
static int bf16_shared_consumers_fence(BnGPUBackend *g,BF16SharedConsumer *c,int count,
    BnTransformerBF16SharedInfo *info) {
    info->busy_fences++;
    for(int i=0;i<count;i++)if(g->resident_synchronize(c[i].request))return -1;
    return 0;
}
static int bf16_shared_banks(BnGPUBackend *g,const BF16SharedSource *source,
    BF16SharedConsumer *c,int count,BnTransformerBF16SharedInfo *info) {
    int capacity=source->capacity,expert=0,bank_index=0;
    if(!g->moe_bank_submit_group||source->n_experts<1||capacity<1||capacity>64||count<2||count>16)return -1;
    while(expert<source->n_experts) {
        int ids[64],n=0;
        while(expert<source->n_experts&&n<capacity) {
            int active=0;for(int i=0;i<count;i++)active|=c[i].offsets[expert+1]>c[i].offsets[expert];
            if(active)ids[n++]=expert;
            expert++;
        }
        if(!n)continue;
        BnGPUMoEBankLease lease={0};int status=-1;
        for(int retry=0;retry<8;retry++) {
            status=g->moe_bank_acquire(c[0].request,(unsigned)bank_index%2,
                source->gate_type,source->down_type,&lease);
            if(status!=BN_GPU_MOE_BANK_BUSY)break;
            if(bf16_shared_consumers_fence(g,c,count,info))return -1;
        }
        if(status)return -1;
        if(lease.capacity<(uint32_t)n||lease.gate_bytes!=source->gate_bytes||lease.down_bytes!=source->down_bytes||
           lease.expert_bytes!=2*source->gate_bytes+source->down_bytes) {
            (void)g->moe_bank_cancel(c[0].request,&lease);return -1;
        }
        for(int e=0;e<n;e++) {
            if(source->cache_hits&&((source->cache_hits[ids[e]/64]>>(ids[e]%64))&1))continue;
            int reader=-1;
            for(int i=0;i<count;i++)if(c[i].offsets[ids[e]+1]>c[i].offsets[ids[e]]) {
                if(reader<0)reader=i;
                if(!c[i].prefill){reader=i;break;}
            }
            if(reader<0||source->read(c[reader].source,ids[e],&lease,
                (unsigned char*)lease.data+(size_t)e*lease.expert_bytes)) {
                (void)g->moe_bank_cancel(c[0].request,&lease);return -1;
            }
        }
        BnGPUMoEBankItem items[16][64];BnGPUMoEBankConsumerPlan consumers[16];int indices[16],used=0;
        for(int i=0;i<count;i++) {
            int active=0;
            for(int e=0;e<n;e++) {
                int rows=c[i].offsets[ids[e]+1]-c[i].offsets[ids[e]];
                items[used][e]=(BnGPUMoEBankItem){ids[e],c[i].offsets[ids[e]],rows};active+=rows>0;
            }
            if(!active)continue;
            indices[used]=i;consumers[used]=c[i].plan;consumers[used].request=c[i].request;
            consumers[used].workspace_slot=(unsigned)bank_index%2;consumers[used].experts=items[used];used++;
        }
        BnGPUMoEBankGroupPlan group={.lease=lease,.n_experts=n,.n_consumers=used,.consumers=consumers,.cache_generation=source->cache_generation};
        for(int retry=0;retry<8;retry++) {
            status=g->moe_bank_submit_group(g->ctx,c[0].request,&group);
            if(status!=BN_GPU_MOE_BANK_BUSY)break;
            if(bf16_shared_consumers_fence(g,c,count,info)){status=-1;break;}
        }
        if(status){(void)g->moe_bank_cancel(c[0].request,&lease);return -1;}
        for(int j=0;j<used;j++) {
            BF16SharedConsumer *v=c+indices[j];v->banks++;
            for(int e=0;e<n;e++)v->experts+=(size_t)(items[j][e].n_rows>0);
        }
        bank_index++;
    }
    info->shared_layers++;return 0;
}
static int bf16_shared_glm_read(void *source,int expert,const BnGPUMoEBankLease *lease,void *dst) {
    (void)lease;return bf16_moe_source_copy(source,expert,dst);
}
static int bf16_shared_moe(BF16GLMRequest **requests,BnTransformerBF16MoEWork **work,
    int count,BnTransformerBF16SharedInfo *info) {
    const BnTransformerBF16MoEPlan *first=&work[0]->active;
    const BnBackendBF16MoEWeights *weights=first->weights;BnGPUBackend *g=weights->gpu;
    BF16SharedConsumer c[16]={0};
    BF16SharedSource source={weights->n_experts,first->bank_capacity,weights->gate_type,
        weights->down_type,weights->gate_bytes,weights->down_bytes,bf16_shared_glm_read,NULL,0};
    if(weights->n_experts!=288)return -1;
    for(int i=0;i<count;i++) {
        BnTransformerBF16MoEWork *s=work[i];const BnTransformerBF16MoEPlan *p=&s->active;
        if(!s->pending||s->failed||s->next_expert||p->weights!=weights||p->source!=first->source||
            p->bank_capacity!=first->bank_capacity)return -1;
        c[i].request=requests[i]->request;c[i].source=&s->active;c[i].offsets=s->offsets;c[i].prefill=p->prefill;
        c[i].plan=(BnGPUMoEBankConsumerPlan){.total_tokens=p->n_tokens,.x=p->gathered,
            .gate=p->gate,.up=p->up,.out=p->down,.prepared_input=s->generation,.clamped=1,.clamp_limit=weights->clamp_limit};
    }
    if(bf16_shared_banks(g,&source,c,count,info))return -1;
    for(int i=0;i<count;i++) {
        work[i]->next_expert=weights->n_experts;work[i]->bank+=(unsigned)c[i].banks;
        work[i]->submitted_banks+=c[i].banks;work[i]->submitted_experts+=c[i].experts;
    }
    return 0;
}
int bn_transformer_bf16_glm_forward_group_session(BnModel *model,
    const BnTransformerBF16SessionRow *rows,int count,BnTransformerBF16SharedInfo *out) {
    if(!model||!rows||count<1||count>16)return -1;
    BF16GLMRequest *requests[16];BnTransformerBF16SharedInfo info={0};
    for(int i=0;i<count;i++) {
        const BnTransformerBF16SessionRow *p=rows+i;
        if(!p->session||!p->tokens)return -1;
        BF16GLMRequest *r=bf16_glm_request(bn_session_backend(p->session));requests[i]=r;
        if(!bf16_glm_context_current(r)||r->context_model!=model||r->context_session!=p->session||
           r->active||p->n_tokens<1||p->n_tokens>r->max_tokens||p->position!=r->position||
           p->n_tokens>r->capacity-p->position||(p->prefill!=0&&p->prefill!=1)||(!p->prefill&&p->n_tokens!=1)||
           !p->session->state.logits)return -1;
        for(int j=0;j<i;j++)if(rows[j].session==p->session||requests[j]->request==r->request)return -1;
        for(int j=0;j<p->n_tokens;j++)if(p->tokens[j]<0||p->tokens[j]>=r->weights->vocab_size)return -1;
        if(i&&(r->weights!=requests[0]->weights||r->gpu!=requests[0]->gpu||r->source!=requests[0]->source))return -1;
    }
    if(count==1) {
        int status=bf16_glm_runtime_run(model,rows[0].session,rows[0].tokens,rows[0].n_tokens,rows[0].position,1,rows[0].prefill);
        if(!status&&out)*out=info;
        return status;
    }
    BnGPUBackend *g=requests[0]->gpu;BnGPUMoEBankStats before[16];
    if(!g->moe_bank_submit_group||!g->moe_bank_stats)return -1;
    for(int i=0;i<count;i++)if(g->moe_bank_stats(requests[i]->request,before+i))return -1;
    for(int i=0;i<count;i++)if(requests[i]->context_tokens!=rows[i].tokens)
        memcpy(requests[i]->context_tokens,rows[i].tokens,(size_t)rows[i].n_tokens*sizeof(int));
    for(int i=0;i<count;i++)if(bn_transformer_bf16_glm_begin(bn_session_backend(rows[i].session),
        requests[i]->context_tokens,rows[i].n_tokens,rows[i].position,rows[i].prefill))goto failed;
    for(int layer=0;layer<45;layer++) {
        BnTransformerBF16MoEWork *work[16];
        for(int i=0;i<count;i++) {
            BF16GLMRequest *r=requests[i];if(r->layer!=layer)goto failed;
            int status=bf16_glm_layer_submit(r,layer>=3);
            if(status!=(layer>=3?1:0))goto failed;
            if(layer>=3)work[i]=r->weights->layers[layer].kda.gpu?&r->kda_work.moe:&r->mla_work.moe;
        }
        if(layer<3)continue;
        if(bf16_shared_moe(requests,work,count,&info))goto failed;
        for(int i=0;i<count;i++)if(bf16_glm_layer_submit(requests[i],0))goto failed;
    }
    for(int i=0;i<count;i++)if(bf16_glm_head_submit(requests[i]))goto failed;
    for(int i=0;i<count;i++)if(g->resident_read(requests[i]->request,requests[i]->logits,requests[i]->host_logits))goto failed;
    for(int i=0;i<count;i++)if(g->resident_synchronize(requests[i]->request))goto failed;
    for(int i=0;i<count;i++) {
        BnGPUMoEBankStats after;
        if(g->moe_bank_stats(requests[i]->request,&after)||after.shared_uploads<before[i].shared_uploads||
           after.shared_consumers<before[i].shared_consumers||
           after.shared_upload_bytes_avoided<before[i].shared_upload_bytes_avoided||after.upload_bytes<before[i].upload_bytes||
           after.joint_kernel_launches<before[i].joint_kernel_launches||
           after.joint_projection_launches<before[i].joint_projection_launches||
           after.joint_launches_avoided<before[i].joint_launches_avoided)goto failed;
        info.weight_upload_bytes+=after.upload_bytes-before[i].upload_bytes;
        info.joint_kernel_launches+=(uint64_t)(after.joint_kernel_launches-before[i].joint_kernel_launches);
        info.joint_projection_launches+=(uint64_t)(after.joint_projection_launches-before[i].joint_projection_launches);
        info.joint_launches_avoided+=(uint64_t)(after.joint_launches_avoided-before[i].joint_launches_avoided);
        info.shared_banks+=(uint64_t)(after.shared_uploads-before[i].shared_uploads);
        info.shared_consumers+=(uint64_t)(after.shared_consumers-before[i].shared_consumers);
        size_t bytes=after.shared_upload_bytes_avoided-before[i].shared_upload_bytes_avoided;
        info.shared_upload_bytes_avoided=bytes>SIZE_MAX-info.shared_upload_bytes_avoided?SIZE_MAX:info.shared_upload_bytes_avoided+bytes;
    }
    /* No fallible operation remains: publish the cohort together. */
    for(int i=0;i<count;i++) {
        BF16GLMRequest *r=requests[i];BnSession *session=rows[i].session;
        r->position+=r->nt;r->parity^=1;r->active=0;
        for(int j=0;j<r->weights->vocab_size;j++){uint32_t bits=(uint32_t)r->host_logits[j]<<16;memcpy(session->state.logits+j,&bits,4);}
        for(int j=0;j<rows[i].n_tokens;j++)session->state.token_history[rows[i].position+j]=r->context_tokens[j];
        session->pos=r->position;
    }
    if(out)*out=info;
    return 0;
failed:
    for(int i=0;i<count;i++)requests[i]->failed=1;
    return -1;
}
static int bf16_shared_flash_read(void *source,int expert,const BnGPUMoEBankLease *lease,void *dst) {
    BF16PLERequest *r=source;BF16FlashFFNRequest *s=&r->flash->ffn;
    const BnMoEExpertMap *map=&s->weights->layers[s->layer].expert_map;
    BnMoEState *state=&s->source;
    void *saved[3]={state->buf,state->buf2,state->buf5};
    size_t sizes[3]={state->buf_size,state->buf2_size,state->buf5_size};
    unsigned char *raw=dst;double start=bn_platform_time_ms();
    state->buf=raw;state->buf_size=lease->gate_bytes;
    state->buf2=raw+lease->gate_bytes;state->buf2_size=lease->gate_bytes;
    state->buf5=raw+2*lease->gate_bytes;state->buf5_size=lease->down_bytes;
    BnMoEHostExpert host={0};
    int status=bn_moe_host_expert_acquire(bn_model_moe_io_const(r->model),state,map,
        s->layer,expert,!r->flash->prefill,&host);
    if(!status)for(int role=0;role<3;role++) {
        void *target=raw+(size_t)role*lease->gate_bytes;
        if(target!=host.projection[role])memcpy(target,host.projection[role],role==2?lease->down_bytes:lease->gate_bytes);
    }
    bn_moe_host_expert_release(&host);
    state->buf=saved[0];state->buf2=saved[1];state->buf5=saved[2];
    state->buf_size=sizes[0];state->buf2_size=sizes[1];state->buf5_size=sizes[2];
    s->source_ms+=bn_platform_time_ms()-start;return status;
}
static int bf16_shared_flash_moe(BF16PLERequest **requests,int count,
    BnTransformerBF16SharedInfo *info) {
    BF16FlashFFNRequest *first=&requests[0]->flash->ffn;
    const BnMoEExpertMap *map=&first->weights->layers[first->layer].expert_map;
    BF16SharedSource source={512,8,map->gate_type,map->down_type,
        map->expert_gate_bytes,map->expert_down_bytes,bf16_shared_flash_read,NULL,0};
    BF16SharedConsumer c[16]={0};
    if(map->expert_gate_bytes!=map->expert_up_bytes)return -1;
    for(int i=0;i<count;i++) {
        BF16PLERequest *r=requests[i];BF16FlashRequest *f=r->flash;BF16FlashFFNRequest *s=&f->ffn;
        if(!s->pending||s->next_expert||s->cache_started||s->layer!=first->layer||s->weights!=first->weights)return -1;
        size_t n=(size_t)f->n_tokens;
        c[i].request=r->request;c[i].source=r;c[i].offsets=s->offsets;c[i].prefill=f->prefill;
        c[i].plan=(BnGPUMoEBankConsumerPlan){.total_tokens=f->n_tokens,
            .x=bf16_slice(s->v.gathered,n*10*2560),.gate=bf16_slice(s->v.gate,n*10*640),
            .up=bf16_slice(s->v.up,n*10*640),.out=bf16_slice(s->v.down,n*10*2560),.prepared_input=s->generation};
    }
    if(first->weights->expert_cache) {
        BnGPUBackend *g=requests[0]->gpu;BnGPUExpertCacheGroupRow rows[16];
        for(int i=0;i<count;i++)rows[i]=(BnGPUExpertCacheGroupRow){requests[i]->request,c[i].offsets,
            requests[i]->flash->n_tokens,c[i].prefill};
        int status=-1;
        for(int retry=0;retry<8;retry++) {
            status=g->expert_cache_begin_group(requests[0]->request,first->layer,rows,count,&first->cache_layer);
            if(status!=BN_GPU_MOE_BANK_BUSY)break;
            if(bf16_shared_consumers_fence(g,c,count,info))return -1;
        }
        if(status)return -1;
        source.cache_hits=first->cache_layer.hits;source.cache_generation=first->cache_layer.generation;
    }
    if(bf16_shared_banks(requests[0]->gpu,&source,c,count,info))return -1;
    for(int i=0;i<count;i++) {
        requests[i]->flash->ffn.next_expert=512;
        requests[i]->flash->ffn.bank+=(unsigned)c[i].banks;
    }
    return 0;
}
int bn_transformer_bf16_flash_forward_group_session(BnModel *model,
    const BnTransformerBF16SessionRow *rows,int count,BnTransformerBF16SharedInfo *out) {
    if(!model||!rows||count<1||count>16)return -1;
    BF16PLERequest *requests[16];BF16SharedConsumer c[16]={0};BnTransformerBF16SharedInfo info={0};
    for(int i=0;i<count;i++) {
        const BnTransformerBF16SessionRow *p=rows+i;
        if(!p->session||!p->tokens||!p->session->backend_only||!bn_session_matches_model(p->session,model))return -1;
        BF16PLERequest *r=bf16_ple_request(bn_session_backend(p->session));requests[i]=r;
        if(!bf16_flash_runtime_current(r)||r->model!=model||!r->flash->ffn.weights||
           r->flash->active||r->flash->model_running||
           p->n_tokens<1||p->n_tokens>r->info.max_tokens||p->position!=r->flash->position||
           p->position!=p->session->pos||p->n_tokens>r->flash->capacity-p->position||
           (p->prefill!=0&&p->prefill!=1)||(!p->prefill&&p->n_tokens!=1)||!p->session->state.logits)return -1;
        for(int j=0;j<i;j++)if(rows[j].session==p->session||requests[j]->request==r->request)return -1;
        for(int j=0;j<p->n_tokens;j++)if(p->tokens[j]<0||p->tokens[j]>=r->flash->io_weights->vocab_size)return -1;
        if(i&&(r->gpu!=requests[0]->gpu||r->flash->ffn.weights!=requests[0]->flash->ffn.weights))return -1;
        c[i].request=r->request;
    }
    if(count==1) {
        int status=bf16_flash_runtime_run(model,rows[0].session,rows[0].tokens,rows[0].n_tokens,rows[0].position,1,rows[0].prefill);
        if(!status&&out)*out=info;
        return status;
    }
    BnGPUBackend *g=requests[0]->gpu;BnGPUMoEBankStats before[16];
    BnGPUExpertCacheStats cache_before[16];int cached=requests[0]->flash->ffn.weights->expert_cache!=NULL;
    if(cached&&(!g->expert_cache_begin_group||!g->expert_cache_end||!g->expert_cache_stats))return -1;
    if(cached)for(int i=0;i<count;i++)if(g->expert_cache_stats(requests[i]->request,cache_before+i))return -1;
    if(!g->moe_bank_submit_group||!g->moe_bank_stats)return -1;
    for(int i=0;i<count;i++)if(g->moe_bank_stats(requests[i]->request,before+i))return -1;
    for(int i=0;i<count;i++) {
        int status=-1;
        for(int retry=0;retry<8;retry++) {
            status=bn_transformer_bf16_flash_model_begin(bn_session_backend(rows[i].session),
                (const int32_t*)rows[i].tokens,rows[i].n_tokens,rows[i].position,rows[i].prefill);
            if(status!=BN_GPU_MOE_BANK_BUSY)break;
            if(bf16_shared_consumers_fence(g,c,count,&info))goto failed;
        }
        if(status)goto failed;
    }
    for(int i=0;i<count;i++) {
        BF16FlashRequest *f=requests[i]->flash;
        if(bn_transformer_bf16_flash_tokens_submit(bn_session_backend(rows[i].session),f->model_tokens,f->n_tokens))goto failed;
        f->model_phase=1;
    }
    for(int layer=0;layer<requests[0]->flash->weights->n_layers;layer++) {
        for(int i=0;i<count;i++) {
            BF16PLERequest *r=requests[i];BF16FlashRequest *f=r->flash;BnBackendSession *b=bn_session_backend(rows[i].session);
            if((layer==r->weights->layer&&bn_transformer_bf16_flash_ple_apply(b,f->model_tokens,f->n_tokens,f->position))||
               bn_transformer_bf16_flash_attention_submit(b,layer))goto failed;
            f->model_phase=2;
            if(bf16_flash_ffn_prepare(b,layer))goto failed;
        }
        if(bf16_shared_flash_moe(requests,count,&info))goto failed;
        for(int i=0;i<count;i++) {
            if(bn_transformer_bf16_flash_ffn_submit(bn_session_backend(rows[i].session),layer))goto failed;
            requests[i]->flash->model_layer++;requests[i]->flash->model_phase=1;
        }
        if(cached&&g->expert_cache_end(requests[0]->request,&requests[0]->flash->ffn.cache_layer))goto failed;
    }
    for(int i=0;i<count;i++) {
        if(bn_transformer_bf16_flash_head_submit(bn_session_backend(rows[i].session)))goto failed;
        requests[i]->flash->model_phase=4;
        if(bf16_flash_can_complete(requests[i]))goto failed;
    }
    for(int i=0;i<count;i++)if(g->resident_read(requests[i]->request,requests[i]->flash->logits,requests[i]->flash->host_logits))goto failed;
    for(int i=0;i<count;i++)if(g->resident_synchronize(requests[i]->request))goto failed;
    for(int i=0;i<count;i++) {
        BnGPUMoEBankStats after;
        if(g->moe_bank_stats(requests[i]->request,&after)||after.shared_uploads<before[i].shared_uploads||
           after.shared_consumers<before[i].shared_consumers||after.shared_upload_bytes_avoided<before[i].shared_upload_bytes_avoided||after.upload_bytes<before[i].upload_bytes||
           after.joint_kernel_launches<before[i].joint_kernel_launches||
           after.joint_projection_launches<before[i].joint_projection_launches||
           after.joint_launches_avoided<before[i].joint_launches_avoided)goto failed;
        info.weight_upload_bytes+=after.upload_bytes-before[i].upload_bytes;
        info.joint_kernel_launches+=(uint64_t)(after.joint_kernel_launches-before[i].joint_kernel_launches);
        info.joint_projection_launches+=(uint64_t)(after.joint_projection_launches-before[i].joint_projection_launches);
        info.joint_launches_avoided+=(uint64_t)(after.joint_launches_avoided-before[i].joint_launches_avoided);
        info.shared_banks+=(uint64_t)(after.shared_uploads-before[i].shared_uploads);
        info.shared_consumers+=(uint64_t)(after.shared_consumers-before[i].shared_consumers);
        size_t bytes=after.shared_upload_bytes_avoided-before[i].shared_upload_bytes_avoided;
        info.shared_upload_bytes_avoided=bytes>SIZE_MAX-info.shared_upload_bytes_avoided?SIZE_MAX:info.shared_upload_bytes_avoided+bytes;
    }
    if(cached)for(int i=0;i<count;i++) {
        BnGPUExpertCacheStats after;const BnGPUExpertCacheStats *before_cache=cache_before+i;
        if(g->expert_cache_stats(requests[i]->request,&after)||after.hits<before_cache->hits||
           after.misses<before_cache->misses||after.hit_bytes<before_cache->hit_bytes||
           after.miss_bytes<before_cache->miss_bytes||after.evictions<before_cache->evictions||
           after.bypasses<before_cache->bypasses||
           after.fill_copy_bytes<before_cache->fill_copy_bytes)goto failed;
        info.cache_hits+=after.hits-before_cache->hits;info.cache_misses+=after.misses-before_cache->misses;
        info.cache_hit_bytes+=after.hit_bytes-before_cache->hit_bytes;
        info.cache_miss_bytes+=after.miss_bytes-before_cache->miss_bytes;
        info.cache_evictions+=after.evictions-before_cache->evictions;
        info.cache_bypasses+=after.bypasses-before_cache->bypasses;
        info.cache_fill_copy_bytes+=after.fill_copy_bytes-before_cache->fill_copy_bytes;
        if(after.reserved_device_bytes>info.cache_reserved_device_bytes)info.cache_reserved_device_bytes=after.reserved_device_bytes;
    }
    for(int i=0;i<count;i++) {
        BF16PLERequest *r=requests[i];BF16FlashRequest *f=r->flash;BnSession *s=rows[i].session;
        bf16_flash_commit(r);
        for(int j=0;j<f->io_weights->vocab_size;j++){uint32_t bits=(uint32_t)f->host_logits[j]<<16;memcpy(s->state.logits+j,&bits,4);}
        for(int j=0;j<rows[i].n_tokens;j++)s->state.token_history[rows[i].position+j]=f->model_tokens[j];
        s->pos=f->position;
    }
    if(out)*out=info;
    return 0;
failed:
    if(g->expert_cache_cancel)for(int i=0;i<count;i++)(void)g->expert_cache_cancel(requests[i]->request);
    for(int i=0;i<count;i++)requests[i]->info.failed=1;
    return -1;
}
#endif
