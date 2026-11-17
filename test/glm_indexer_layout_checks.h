/* Synthetic source ownership, preparation rollback and whole-plan preflight. */
typedef struct {int calls,fail,freed;} IndexerProbe;
static void *indexer_upload(void *ctx,const void *src,size_t bytes,int type,int rows,int cols){
    IndexerProbe *p=ctx;assert(src&&bytes&&rows>0&&cols>0&&(type==BN_GGUF_TENSOR_Q8_0||type==BN_GGUF_TENSOR_F32));
    if(++p->calls==p->fail)return NULL;
    void *out=malloc(bytes);assert(out);memcpy(out,src,bytes);return out;
}
static void indexer_destroy(void *ctx,void *buffer){IndexerProbe *p=ctx;assert(buffer);p->freed++;free(buffer);}
static void test_indexer_preparation(void){
    BnModelAnatomy *a=calloc(1,sizeof(*a));assert(a);a->n_layers=1;a->dim=a->q_rank=256;a->indexer_heads=32;a->indexer_dim=128;a->indexer_pool=4;a->indexer_norm_eps=1e-6f;a->layers[0].kind=BN_MODEL_ANATOMY_INDEXED_MLA;
    for(int i=0;i<BN_MODEL_TENSOR_ROLE_COUNT;++i)a->layers[0].tensor[i]=-1;
    BnModelTensorRole roles[]={BN_MODEL_TENSOR_INDEXER_Q,BN_MODEL_TENSOR_INDEXER_K,BN_MODEL_TENSOR_INDEXER_POOL_GATE,BN_MODEL_TENSOR_INDEXER_HEAD_WEIGHTS,BN_MODEL_TENSOR_INDEXER_K_NORM,BN_MODEL_TENSOR_INDEXER_K_BIAS,BN_MODEL_TENSOR_INDEXER_POOL_APE};
    int rows[]={4096,128,128,32,1,1,4},cols[]={256,256,256,256,128,128,128};
    BnGGUFTensorInfo ts[7]={0};size_t sizes[7],total=0;
    for(int i=0;i<7;++i){a->layers[0].tensor[roles[i]]=i;ts[i].type=i<3?BN_GGUF_TENSOR_Q8_0:BN_GGUF_TENSOR_F32;ts[i].n_dims=i==4||i==5?1:2;ts[i].dims[0]=cols[i];ts[i].dims[1]=rows[i];ts[i].offset=total;assert(bn_gguf_tensor_size(ts[i].type,(uint64_t)rows[i]*cols[i],&sizes[i]));total+=sizes[i];}
    unsigned char *raw=calloc(1,total),*saved=malloc(total);assert(raw&&saved);float value=1.00390625f;
    for(int i=3;i<7;++i)memcpy(raw+ts[i].offset,&value,4);
    memcpy(saved,raw,total);BnGGUFFile f={.raw=raw,.raw_size=total,.tensors=ts,.n_tensors=7};
    IndexerProbe q={0};BnGPUBackend g={.ctx=&q,.buffer_create_quant_only=indexer_upload,.buffer_destroy=indexer_destroy};BnBackendBF16IndexerWeights w={0};
    assert(!bn_backend_layout_bf16_indexer_init(&w,&g,&f,a,0,total));assert(q.calls==7&&w.bytes==total&&w.heads==32&&w.weight_scale==.015625f&&w.norm_eps==1e-6f);
    void *ptr[]={w.query.raw,w.key.raw,w.gate.raw,w.head_weights,w.norm,w.bias,w.ape};for(int i=0;i<7;++i)assert(!memcmp(ptr[i],raw+ts[i].offset,sizes[i]));
    assert(bn_backend_layout_bf16_indexer_init(&w,&g,&f,a,0,total)==-1);bn_backend_layout_bf16_indexer_free(&w);bn_backend_layout_bf16_indexer_free(&w);assert(q.freed==7);
    for(int fail=1;fail<=7;++fail){q=(IndexerProbe){.fail=fail};BnBackendBF16IndexerWeights before=w;assert(bn_backend_layout_bf16_indexer_init(&w,&g,&f,a,0,total)==-1&&q.calls==fail&&q.freed==fail-1&&!memcmp(&before,&w,sizeof(w)));}
    q=(IndexerProbe){0};assert(bn_backend_layout_bf16_indexer_init(&w,&g,&f,a,0,total-1)==-1);
    a->rope_dim=64;assert(bn_backend_layout_bf16_indexer_init(&w,&g,&f,a,0,total)==-1);a->rope_dim=0;
    a->indexer_norm_eps=NAN;assert(bn_backend_layout_bf16_indexer_init(&w,&g,&f,a,0,total)==-1);a->indexer_norm_eps=1e-6f;
    a->layers[0].kind=BN_MODEL_ANATOMY_KDA;assert(bn_backend_layout_bf16_indexer_init(&w,&g,&f,a,0,total)==-1);a->layers[0].kind=BN_MODEL_ANATOMY_INDEXED_MLA;
    for(int i=0;i<7;++i){uint64_t offset=ts[i].offset;ts[i].offset=total;assert(bn_backend_layout_bf16_indexer_init(&w,&g,&f,a,0,total)==-1);ts[i].offset=offset;
        uint64_t col=ts[i].dims[0];ts[i].dims[0]++;assert(bn_backend_layout_bf16_indexer_init(&w,&g,&f,a,0,total)==-1);ts[i].dims[0]=col;}
    for(int i=3;i<7;++i){float invalid=INFINITY;memcpy(raw+ts[i].offset,&invalid,4);assert(bn_backend_layout_bf16_indexer_init(&w,&g,&f,a,0,total)==-1);memcpy(raw+ts[i].offset,&value,4);}
    assert(!q.calls&&!w.gpu&&!memcmp(raw,saved,total));free(raw);free(saved);free(a);puts("  indexer raw Q8/F32 preparation, source bounds, budget and upload rollback PASSED");
}
#define INDEXER_CALLBACK(NAME,TYPE) static int NAME(void *ctx,void *r,const TYPE *p){IndexerProbe *q=ctx;assert(ctx==r&&p);return ++q->calls==q->fail?-1:0;}
INDEXER_CALLBACK(indexer_linear,BnGPUBF16LinearResidentPlan)
INDEXER_CALLBACK(indexer_head,BnGPUBF16F32LinearResidentPlan)
INDEXER_CALLBACK(indexer_norm,BnGPULayerNorm128F32BF16ResidentPlan)
INDEXER_CALLBACK(indexer_fp8,BnGPUFWHT128FP8ResidentPlan)
INDEXER_CALLBACK(indexer_scale,BnGPUIndexerWeightScaleResidentPlan)
INDEXER_CALLBACK(indexer_pool,BnGPUKPoolFP8ResidentPlan)
INDEXER_CALLBACK(indexer_append,BnGPUIndexerFP8AppendResidentPlan)
INDEXER_CALLBACK(indexer_gather,BnGPUIndexerFP8GatherResidentPlan)
INDEXER_CALLBACK(indexer_score,BnGPUIndexerFP8ScoreResidentPlan)
INDEXER_CALLBACK(indexer_select,BnGPUIndexerSelectResidentPlan)
INDEXER_CALLBACK(indexer_paged_score,BnGPUIndexerFP8PagedScoreResidentPlan)
#undef INDEXER_CALLBACK
static void test_indexer_orchestration(void){
    IndexerProbe q={0};BnGPUBackend g={.ctx=&q,.linear_a8_bf16_resident=indexer_linear,.linear_bf16_f32_weight_resident=indexer_head,.layernorm128_f32_bf16_resident=indexer_norm,.fwht128_fp8_resident=indexer_fp8,.indexer_weight_scale_resident=indexer_scale,.kpool_fp8_resident=indexer_pool};
    BnBackendBF16IndexerWeights w={.gpu=&g,.head_weights=&g,.norm=&g,.bias=&g,.ape=&g,.dim=128,.q_rank=128,.heads=32,.head_dim=128,.pool_size=4,.norm_eps=1e-6f,.weight_scale=.015625f};w.query=w.key=w.gate=(BnGPUBF16LinearWeight){.raw=&g,.type=BN_GGUF_TENSOR_Q8_0};
    size_t counts[]={256,256,1024,1024,8192,256,256,256,256,64,8192,64,64,128,1};BnGPUResidentView v[15];
    for(int i=0;i<15;++i)v[i]=(BnGPUResidentView){.request=&q,.slot=(uint32_t)i,.count=counts[i],.type=i==8||i==9||i==11||i==12||i==14?BN_GGUF_TENSOR_F32:i==10||i==13?BN_GGUF_TENSOR_I8:BN_GGUF_TENSOR_BF16};
    BnTransformerBF16IndexerPlan p={.weights=&w,.n_tokens=2,.position=3,.x=v[0],.qr=v[1],.initial_tail=v[2],.next_tail=v[3],.q=v[4],.key_raw=v[5],.key=v[6],.gate=v[7],.cast=v[8],.head_raw=v[9],.q_fp8=v[10],.q_scales=v[11],.head_weights=v[12],.compressed=v[13],.compressed_scales=v[14]};
    for(int fail=0;fail<=8;++fail){q=(IndexerProbe){.fail=fail};assert(bn_transformer_bf16_indexer(&q,&p)==(fail?-1:0)&&q.calls==(fail?fail:8));}
    q=(IndexerProbe){0};BnTransformerBF16IndexerPlan bad=p;bad.next_tail=bad.initial_tail;assert(bn_transformer_bf16_indexer(&q,&bad)==-1);
    bad=p;bad.head_weights=bad.head_raw;assert(bn_transformer_bf16_indexer(&q,&bad)==-1);bad=p;bad.q_fp8.count--;assert(bn_transformer_bf16_indexer(&q,&bad)==-1);
    bad=p;bad.layer=1;assert(bn_transformer_bf16_indexer(&q,&bad)==-1);bad=p;bad.position=INT_MAX;assert(bn_transformer_bf16_indexer(&q,&bad)==-1);bad=p;bad.position=0;assert(bn_transformer_bf16_indexer(&q,&bad)==-1);assert(!q.calls);
    g.indexer_fp8_append_resident=indexer_append;g.indexer_fp8_gather_resident=indexer_gather;
    BnGPUResidentView extra[6];size_t ec[]={8448,1,1,128,1,1};int et[]={BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32};
    for(int j=0;j<6;++j)extra[j]=(BnGPUResidentView){.request=&q,.slot=(uint32_t)(15+j),.type=et[j],.count=ec[j]};
    BnTransformerBF16IndexerCachePlan cp={.indexer=p,.num_cache_blocks=1,.logical_blocks=1,.cache=extra[0],.block_table=extra[1],.slots=extra[2],.gathered=extra[3],.scales=extra[4],.status=extra[5]};
    for(int fail=0;fail<=10;++fail){q=(IndexerProbe){.fail=fail};assert(bn_transformer_bf16_indexer_cache(&q,&cp)==(fail?-1:0)&&q.calls==(fail?fail:10));}
    q=(IndexerProbe){0};BnTransformerBF16IndexerCachePlan cb=cp;cb.gathered=p.compressed;assert(bn_transformer_bf16_indexer_cache(&q,&cb)==-1);
    cb=cp;cb.status=cb.slots;assert(bn_transformer_bf16_indexer_cache(&q,&cb)==-1);cb=cp;cb.cache.offset=1;assert(bn_transformer_bf16_indexer_cache(&q,&cb)==-1);
    cb=cp;cb.slots.count++;assert(bn_transformer_bf16_indexer_cache(&q,&cb)==-1);cb=cp;cb.indexer.compressed.offset=1;assert(bn_transformer_bf16_indexer_cache(&q,&cb)==-1);assert(!q.calls);
    g.indexer_fp8_score_resident=indexer_score;
    BnTransformerBF16IndexerScoresPlan sp={.cache=cp,.scores={.request=&q,.slot=21,.type=BN_GGUF_TENSOR_F32,.count=2}};
    for(int fail=0;fail<=11;++fail){q=(IndexerProbe){.fail=fail};assert(bn_transformer_bf16_indexer_scores(&q,&sp)==(fail?-1:0)&&q.calls==(fail?fail:11));}
    q=(IndexerProbe){0};BnTransformerBF16IndexerScoresPlan sb;
#define SCORE_REJECT(change) do{sb=sp;change;assert(bn_transformer_bf16_indexer_scores(&q,&sb)==-1&&!q.calls);}while(0)
    SCORE_REJECT(sb.scores.count++);SCORE_REJECT(sb.scores.request=&g);SCORE_REJECT(sb.scores.type=BN_GGUF_TENSOR_I32);
    SCORE_REJECT(sb.scores.offset=SIZE_MAX);SCORE_REJECT(sb.cache.indexer.q_fp8.offset=1);
    BnGPUResidentView all[]={p.x,p.qr,p.initial_tail,p.next_tail,p.q,p.key_raw,p.key,p.gate,p.cast,p.head_raw,p.q_fp8,p.q_scales,p.head_weights,p.compressed,p.compressed_scales,cp.cache,cp.block_table,cp.slots,cp.gathered,cp.scales,cp.status};
    for(int j=0;j<21;++j){SCORE_REJECT(sb.scores.slot=all[j].slot;sb.scores.offset=all[j].offset);}
    SCORE_REJECT(sb.cache.slots.count++);SCORE_REJECT(sb.cache.indexer.next_tail=p.initial_tail);
    g.indexer_fp8_score_resident=NULL;SCORE_REJECT((void)0);g.indexer_fp8_score_resident=indexer_score;
#undef SCORE_REJECT
    /* No complete pool still validates the global plan and submits all APIs;
     * the backend's empty append/gather/score contracts perform no work. */
    sb=sp;sb.cache.indexer.position=0;sb.cache.indexer.compressed=(BnGPUResidentView){0};sb.cache.indexer.compressed_scales=(BnGPUResidentView){0};sb.cache.slots=(BnGPUResidentView){0};sb.cache.gathered=(BnGPUResidentView){0};sb.cache.scales=(BnGPUResidentView){0};sb.scores=(BnGPUResidentView){0};
    q=(IndexerProbe){0};assert(!bn_transformer_bf16_indexer_scores(&q,&sb)&&q.calls==11);
    g.indexer_select_resident=indexer_select;
    BnTransformerBF16IndexerSelectPlan sel={.indexed=sp,.pools={.request=&q,.slot=22,.type=BN_GGUF_TENSOR_I32,.count=1024},.out={.request=&q,.slot=23,.type=BN_GGUF_TENSOR_I32,.count=4102}};
    for(int fail=0;fail<=12;++fail){q=(IndexerProbe){.fail=fail};assert(bn_transformer_bf16_indexer_select(&q,&sel)==(fail?-1:0)&&q.calls==(fail?fail:12));}
    q=(IndexerProbe){0};BnTransformerBF16IndexerSelectPlan bad_sel;
#define SELECT_REJECT(change) do{bad_sel=sel;change;assert(bn_transformer_bf16_indexer_select(&q,&bad_sel)==-1&&!q.calls);}while(0)
    SELECT_REJECT(bad_sel.out.count--);SELECT_REJECT(bad_sel.pools.request=&g);SELECT_REJECT(bad_sel.out.type=BN_GGUF_TENSOR_F32);
    SELECT_REJECT(bad_sel.out.offset=SIZE_MAX);SELECT_REJECT(bad_sel.short_prefill=2);SELECT_REJECT(bad_sel.out.slot=sel.pools.slot);
    for(int j=0;j<21;++j){SELECT_REJECT(bad_sel.out.slot=all[j].slot);SELECT_REJECT(bad_sel.pools.slot=all[j].slot);}
    SELECT_REJECT(bad_sel.out.slot=sp.scores.slot);SELECT_REJECT(bad_sel.pools.slot=sp.scores.slot);
    g.indexer_select_resident=NULL;SELECT_REJECT((void)0);g.indexer_select_resident=indexer_select;
#undef SELECT_REJECT
    sel.short_prefill=1;sel.indexed.scores=(BnGPUResidentView){0};g.indexer_fp8_score_resident=NULL;
    for(int fail=0;fail<=11;++fail){q=(IndexerProbe){.fail=fail};assert(bn_transformer_bf16_indexer_select(&q,&sel)==(fail?-1:0)&&q.calls==(fail?fail:11));}
    g.indexer_fp8_score_resident=indexer_score;
    g.indexer_fp8_paged_score_resident=indexer_paged_score;
    BnTransformerBF16IndexerDecodePlan dp={.indexer=p,.num_cache_blocks=1,.logical_blocks=1,.cache=cp.cache,.block_table=cp.block_table,.slots=cp.slots,.scores=sp.scores,.pools=sel.pools,.out=sel.out,.status=cp.status};
    /* No gather or contiguous scorer is required by direct paged decode. */
    g.indexer_fp8_gather_resident=NULL;g.indexer_fp8_score_resident=NULL;
    for(int fail=0;fail<=11;++fail){q=(IndexerProbe){.fail=fail};assert(bn_transformer_bf16_indexer_decode(&q,&dp)==(fail?-1:0)&&q.calls==(fail?fail:11));}
    q=(IndexerProbe){0};BnTransformerBF16IndexerDecodePlan db;
#define DECODE_REJECT(change) do{db=dp;change;assert(bn_transformer_bf16_indexer_decode(&q,&db)==-1&&!q.calls);}while(0)
    DECODE_REJECT(db.cache.offset=1);DECODE_REJECT(db.indexer.q_fp8.offset=1);DECODE_REJECT(db.short_decode=2);DECODE_REJECT(db.logical_blocks=0);DECODE_REJECT(db.scores.count++);DECODE_REJECT(db.out.count--);DECODE_REJECT(db.status.request=&g);
    BnGPUResidentView dv[]={dp.indexer.x,dp.indexer.qr,dp.indexer.initial_tail,dp.indexer.next_tail,dp.indexer.q,dp.indexer.key_raw,dp.indexer.key,dp.indexer.gate,dp.indexer.cast,dp.indexer.head_raw,dp.indexer.q_fp8,dp.indexer.q_scales,dp.indexer.head_weights,dp.indexer.compressed,dp.indexer.compressed_scales,dp.cache,dp.block_table,dp.slots,dp.scores,dp.pools,dp.out,dp.status};
    for(int a=15;a<22;++a)for(int b=0;b<a;++b){DECODE_REJECT(((BnGPUResidentView*[]){&db.cache,&db.block_table,&db.slots,&db.scores,&db.pools,&db.out,&db.status})[a-15]->slot=dv[b].slot);}
    g.indexer_fp8_paged_score_resident=NULL;DECODE_REJECT((void)0);
#undef DECODE_REJECT
    dp.short_decode=1;dp.scores=(BnGPUResidentView){0};
    for(int fail=0;fail<=10;++fail){q=(IndexerProbe){.fail=fail};assert(bn_transformer_bf16_indexer_decode(&q,&dp)==(fail?-1:0)&&q.calls==(fail?fail:10));}
    puts("  indexer direct paged decode preflight, all aliases, score bypass and submission failures PASSED");
}
