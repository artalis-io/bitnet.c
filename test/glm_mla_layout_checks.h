/* MLA backend ownership, conversion budgets and submission failure coverage. */
typedef struct {int calls,fail,freed,status,busy,acquire_fail,cancel;unsigned char bank[96];} MlaProbe;
typedef struct {size_t bytes;int rows,cols,type;unsigned char data[];} MlaMockBuffer;
static void*mla_upload(void*ctx,const void*data,size_t bytes,int type,int rows,int cols){MlaProbe*p=ctx;if(++p->calls==p->fail)return NULL;MlaMockBuffer*b=malloc(sizeof(*b)+bytes);assert(b&&data);*b=(MlaMockBuffer){.bytes=bytes,.rows=rows,.cols=cols,.type=type};memcpy(b->data,data,bytes);return b;}
static void*mla_convert(void*ctx,void*raw){MlaProbe*p=ctx;MlaMockBuffer*src=raw;if(++p->calls==p->fail)return NULL;size_t bytes=(size_t)src->rows*src->cols*2;MlaMockBuffer*b=calloc(1,sizeof(*b)+bytes);assert(b);*b=(MlaMockBuffer){.bytes=bytes,.rows=src->rows,.cols=src->cols,.type=BN_GGUF_TENSOR_BF16};return b;}
static void mla_destroy(void*ctx,void*ptr){MlaProbe*p=ctx;assert(ptr);p->freed++;free(ptr);}
static void test_mla_preparation(void){
 BnModelAnatomy*a=calloc(1,sizeof(*a));assert(a);a->n_layers=1;a->dim=64;a->q_rank=64;a->kv_rank=64;a->attention_heads=2;a->mla_key_dim=32;a->mla_value_dim=32;a->norm_eps=1e-5f;a->layers[0].kind=BN_MODEL_ANATOMY_INDEXED_MLA;for(int j=0;j<BN_MODEL_TENSOR_ROLE_COUNT;++j)a->layers[0].tensor[j]=-1;
 BnModelTensorRole roles[]={BN_MODEL_TENSOR_MLA_Q_A,BN_MODEL_TENSOR_MLA_Q_B,BN_MODEL_TENSOR_MLA_KV_A,BN_MODEL_TENSOR_ATTN_OUTPUT,BN_MODEL_TENSOR_MLA_Q_A_NORM,BN_MODEL_TENSOR_MLA_KV_A_NORM,BN_MODEL_TENSOR_MLA_K_B,BN_MODEL_TENSOR_MLA_V_B};
 int rows[]={64,64,64,64,1,1,128,64},cols[]={64,64,64,64,64,64,32,64};BnGGUFTensorInfo ts[8]={0};size_t sizes[8],total=0,payload=0,peak=0;
 for(int j=0;j<8;++j){a->layers[0].tensor[roles[j]]=j;ts[j].type=j==4||j==5?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_Q8_0;ts[j].n_dims=j==4||j==5?1:j>=6?3:2;ts[j].dims[0]=cols[j];ts[j].dims[1]=j>=6?rows[j]/2:rows[j];ts[j].dims[2]=j>=6?2:1;ts[j].offset=total;assert(bn_gguf_tensor_size(ts[j].type,(uint64_t)rows[j]*cols[j],&sizes[j]));total+=sizes[j];size_t keep=j<4?sizes[j]:(size_t)rows[j]*cols[j]*2;size_t next=payload+keep;size_t transient=j<4?next:next+sizes[j];if(transient>peak)peak=transient;payload=next;}
 unsigned char*raw=calloc(1,total),*saved=malloc(total);assert(raw&&saved);float gamma=1.00390625f;for(int j=4;j<6;++j)memcpy(raw+ts[j].offset,&gamma,4);memcpy(saved,raw,total);BnGGUFFile f={.raw=raw,.raw_size=total,.tensors=ts,.n_tensors=8};MlaProbe probe={0};BnGPUBackend g={.ctx=&probe,.buffer_create_quant_only=mla_upload,.buffer_create_bf16_decoded=mla_convert,.buffer_create_bf16_identity_materialized=mla_convert,.buffer_destroy=mla_destroy};BnBackendBF16MLAWeights w={0};
 assert(!bn_backend_layout_bf16_mla_init(&w,&g,&f,a,0,peak)&&probe.calls==12&&w.bytes==payload&&w.peak_bytes==peak&&probe.freed==4);assert(w.scale==1.f/sqrtf(32.f));assert(bn_backend_layout_bf16_mla_init(&w,&g,&f,a,0,peak)==-1);bn_backend_layout_bf16_mla_free(&w);bn_backend_layout_bf16_mla_free(&w);assert(probe.freed==12);
 for(int fail=1;fail<=12;++fail){probe=(MlaProbe){.fail=fail};BnBackendBF16MLAWeights before=w;assert(bn_backend_layout_bf16_mla_init(&w,&g,&f,a,0,peak)==-1&&probe.calls==fail&&probe.freed==fail-1&&!memcmp(&before,&w,sizeof(w)));}
 probe=(MlaProbe){0};assert(bn_backend_layout_bf16_mla_init(&w,&g,&f,a,0,peak-1)==-1);a->rope_dim=1;assert(bn_backend_layout_bf16_mla_init(&w,&g,&f,a,0,peak)==-1);a->rope_dim=0;
 for(int j=0;j<8;++j){uint64_t dim=ts[j].dims[0],off=ts[j].offset;ts[j].dims[0]++;assert(bn_backend_layout_bf16_mla_init(&w,&g,&f,a,0,peak)==-1);ts[j].dims[0]=dim;ts[j].offset=total;assert(bn_backend_layout_bf16_mla_init(&w,&g,&f,a,0,peak)==-1);ts[j].offset=off;}
 for(int j=4;j<6;++j){float bad=INFINITY;memcpy(raw+ts[j].offset,&bad,4);assert(bn_backend_layout_bf16_mla_init(&w,&g,&f,a,0,peak)==-1);memcpy(raw+ts[j].offset,&gamma,4);}
 g.buffer_create_bf16_identity_materialized=NULL;assert(bn_backend_layout_bf16_mla_init(&w,&g,&f,a,0,peak)==-1);assert(!probe.calls&&!memcmp(saved,raw,total));free(raw);free(saved);free(a);puts("  MLA source ownership, identity layout budgets and all conversion rollback failures PASSED");
}
#define MLA_CALLBACK(NAME,TYPE) static int NAME(void*ctx,void*r,const TYPE*p){MlaProbe*q=ctx;assert(ctx==r&&p);return ++q->calls==q->fail?-1:0;}
MLA_CALLBACK(mla_linear,BnGPUBF16LinearResidentPlan)
MLA_CALLBACK(mla_norm,BnGPURMSNormResidentPlan)
MLA_CALLBACK(mla_head,BnGPUHeadLinearBF16ResidentPlan)
MLA_CALLBACK(mla_append,BnGPUMlaFP8CacheAppendResidentPlan)
MLA_CALLBACK(mla_attention,BnGPUMlaFP8AttentionResidentPlan)
MLA_CALLBACK(mla_idx_head,BnGPUBF16F32LinearResidentPlan)
MLA_CALLBACK(mla_idx_norm,BnGPULayerNorm128F32BF16ResidentPlan)
MLA_CALLBACK(mla_idx_fp8,BnGPUFWHT128FP8ResidentPlan)
MLA_CALLBACK(mla_idx_scale,BnGPUIndexerWeightScaleResidentPlan)
MLA_CALLBACK(mla_idx_pool,BnGPUKPoolFP8ResidentPlan)
MLA_CALLBACK(mla_idx_append,BnGPUIndexerFP8AppendResidentPlan)
MLA_CALLBACK(mla_idx_score,BnGPUIndexerFP8PagedScoreResidentPlan)
MLA_CALLBACK(mla_idx_select,BnGPUIndexerSelectResidentPlan)
MLA_CALLBACK(mla_block_mhc,BnGPUMHCFusedResidentPlan)
MLA_CALLBACK(mla_block_pre,BnGPUMHCPreResidentPlan)
MLA_CALLBACK(mla_block_top,BnGPUBiasedSigmoidF32ResidentPlan)
MLA_CALLBACK(mla_block_group,BnGPUMoEGroupBF16ResidentPlan)
MLA_CALLBACK(mla_block_clamp,BnGPUClampedSwiGLUResidentPlan)
MLA_CALLBACK(mla_block_bank,BnGPUMoEBankResidentPlan)
MLA_CALLBACK(mla_block_reduce,BnGPUMoEReduceBF16ResidentPlan)
MLA_CALLBACK(mla_block_add,BnGPUAddBF16ResidentPlan)
#undef MLA_CALLBACK
static int mla_block_splits(void *ctx,int nt,int dim){assert(ctx&&nt==2&&dim==4096);return 1;}
static int mla_block_zero(void *r,BnGPUResidentView v){MlaProbe*q=r;assert(v.count==293);return ++q->calls==q->fail?-1:0;}
static int mla_block_read(void *r,BnGPUResidentView v,void *out){MlaProbe*q=r;if(++q->calls==q->fail)return -1;
 if(v.count==1){*(int32_t*)out=q->status;return 0;}assert(v.count==293);int32_t*p=out;memset(p,0,293*4);for(int e=0;e<=288;++e)p[e]=(e<8?e:8)*2;return 0;}
static int mla_block_input(void *ctx,void*r,const BnGPUMoEBankInputPlan*p,uint64_t*generation){MlaProbe*q=r;assert(ctx==r&&p);if(++q->calls==q->fail)return -1;*generation=1;return 0;}
static int mla_block_acquire(void *r,unsigned slot,int gt,int dt,BnGPUMoEBankLease*out){MlaProbe*q=r;assert(slot<2&&gt==BN_GGUF_TENSOR_Q4_K&&dt==BN_GGUF_TENSOR_Q5_K);if(++q->calls==q->fail||q->acquire_fail)return -1;if(q->busy){q->busy=0;return BN_GPU_MOE_BANK_BUSY;}*out=(BnGPUMoEBankLease){.capacity=2,.data=q->bank,.gate_bytes=16,.down_bytes=16,.expert_bytes=48};return 0;}
static int mla_block_cancel(void*r,const BnGPUMoEBankLease*l){MlaProbe*q=r;assert(l->data==q->bank);q->cancel++;return 0;}
static void test_mla_moe_composition(MlaProbe*q,BnGPUBackend*g,BnTransformerBF16IndexedMLAPlan indexed,
 BnBackendBF16MLAWeights*w,BnBackendBF16IndexerWeights*iw,int mode){
 BnGPUBackend saved=*g;w->dim=iw->dim=4096;indexed.prepare.prepare.x.count=8192;indexed.indexer.indexer.x=indexed.prepare.prepare.x;indexed.indexer.indexer.cast.count=8192;indexed.output.out.count=8192;
 g->mhc_fused_bf16_split_count=g->mhc_pre_bf16_split_count=mla_block_splits;g->mhc_fused_post_pre_bf16_resident=mla_block_mhc;g->mhc_pre_bf16_resident=mla_block_pre;
 g->resident_read=mla_block_read;g->resident_zero=mla_block_zero;g->moe_biased_sigmoid_f32_resident=mla_block_top;g->moe_group_bf16_resident=mla_block_group;g->clamped_swiglu_bf16_resident=mla_block_clamp;g->moe_bank_prepare_input=mla_block_input;g->moe_bank_acquire=mla_block_acquire;g->moe_bank_submit=mla_block_bank;g->moe_bank_cancel=mla_block_cancel;g->moe_reduce_bf16_resident=mla_block_reduce;g->add_bf16_resident=mla_block_add;
 BnGPUResidentView v[37]={0};size_t sizes[]={32768,48,2,8,8,32,8192,8192,32768,8192,8,32,32768,48,2,8,8,32,8192,8192,8192,576,16,16,293,288,16,16,65536,4096,4096,65536,65536,8192,1024,512,8192};
 for(int j=0;j<37;++j)v[j]=(BnGPUResidentView){.request=q,.slot=(uint32_t)(40+j),.count=sizes[j],.type=(j>=1&&j<=5)||j==10||j==11||(j>=13&&j<=17)||(j>=20&&j<=22)?BN_GGUF_TENSOR_F32:j>=23&&j<=27?BN_GGUF_TENSOR_I32:BN_GGUF_TENSOR_BF16};
 BnGPUMHCPreResidentPlan pre={.n_tokens=2,.dim=4096,.streams=4,.n_splits=1,.iterations=20,.rms_eps=1e-5f,.pre_eps=1e-6f,.sinkhorn_eps=1e-6f,.post_mult=2,.norm_eps=1e-5f,.fn=g,.scale=g,.base=g,.norm=g,.residual=v[0],.dot=v[1],.sqrsum=v[2],.pre=v[3],.post=v[4],.comb=v[5],.collapsed=v[6],.out=indexed.prepare.prepare.x},ffn=pre;ffn.residual=v[12];ffn.dot=v[13];ffn.sqrsum=v[14];ffn.pre=v[15];ffn.post=v[16];ffn.comb=v[17];ffn.collapsed=v[18];ffn.out=v[19];
 /* Opaque mock payloads test continuation; numerical expert geometry is
  * independently covered by actual checkpoint replay. */
 unsigned char raw[288*48]={0};BnMoEExpertMap map={.gate_type=BN_GGUF_TENSOR_Q4_K,.up_type=BN_GGUF_TENSOR_Q4_K,.down_type=BN_GGUF_TENSOR_Q5_K,.expert_gate_bytes=16,.expert_up_bytes=16,.expert_down_bytes=16,.gate_stride=16,.up_stride=16,.down_stride=16,.up_offset=288*16,.down_offset=576*16};BnMoEIO io={.fd=-1,.mmap_base=raw};BnMoEState host={0};
 BnBackendBF16MoEWeights mw={.gpu=g,.router=g,.bias=g,.source_map=&map,.dim=4096,.hidden_dim=256,.n_experts=288,.k=8,.gate_type=BN_GGUF_TENSOR_Q4_K,.down_type=BN_GGUF_TENSOR_Q5_K,.gate_bytes=16,.down_bytes=16,.route_scale=2.5f,.clamp_limit=10};mw.shared=(BnBackendBF16DenseFFNWeights){.gpu=g,.dim=4096,.hidden_dim=256,.clamp_limit=10,.gate_up=w->q_a,.down=w->q_a};
 BnTransformerBF16MLAMoEBlockPlan p={.pre=pre,.ffn=ffn,.has_previous=mode,.previous_residual=v[8],.previous_branch=v[9],.previous_post=v[10],.previous_comb=v[11],.attention=indexed,.moe={.weights=&mw,.source=&io,.host_state=&host,.n_tokens=2,.prefill=1,.bank_capacity=2,.x=v[19],.cast=v[20],.logits=v[21],.route_weights=v[22],.indices=v[23],.metadata=v[24],.cursors=v[25],.slots=v[26],.inverse=v[27],.gathered=v[28],.gate=v[29],.up=v[30],.down=v[31],.weighted=v[32],.routed=v[33],.shared_gate_up=v[34],.shared_activated=v[35],.shared_out=v[36],.out={.request=q,.slot=77,.count=8192,.type=BN_GGUF_TENSOR_BF16}}},bad;
 BnTransformerBF16MLAMoEBlockWork work={0};int calls=mode?33:32;
 *q=(MlaProbe){0};
#define BLOCK_REJECT(change) do{bad=p;change;assert(bn_transformer_bf16_mla_moe_begin(q,&bad,&work)==-1&&!work.pending&&!work.failed&&!q->calls);}while(0)
 BLOCK_REJECT(bad.has_previous=2);BLOCK_REJECT(bad.ffn.n_splits=2);BLOCK_REJECT(bad.moe.x.slot=80);BLOCK_REJECT(bad.pre.out=bad.pre.collapsed);BLOCK_REJECT(bad.moe.out.slot=bad.attention.prepare.prepare.qr.slot);BLOCK_REJECT(bad.moe.shared_out.slot=bad.attention.output.out.slot);BLOCK_REJECT(bad.ffn.collapsed.slot=bad.pre.residual.slot);if(mode)BLOCK_REJECT(bad.previous_residual=bad.pre.residual);
#undef BLOCK_REJECT
 for(int fail=1;fail<=calls;++fail){*q=(MlaProbe){.fail=fail};work=(BnTransformerBF16MLAMoEBlockWork){0};assert(bn_transformer_bf16_mla_moe_begin(q,&p,&work)==-1&&work.failed&&q->calls==fail);assert(bn_transformer_bf16_mla_moe_submit(q,&work)==-1&&q->calls==fail);}
 *q=(MlaProbe){.status=1};work=(BnTransformerBF16MLAMoEBlockWork){0};assert(bn_transformer_bf16_mla_moe_begin(q,&p,&work)==-1&&work.failed&&!work.moe.pending&&work.status_read_bytes==4&&q->calls==calls-9);
 *q=(MlaProbe){0};work=(BnTransformerBF16MLAMoEBlockWork){0};assert(!bn_transformer_bf16_mla_moe_begin(q,&p,&work)&&q->calls==calls&&work.status_read_bytes==4);assert(bn_transformer_bf16_mla_moe_begin(q,&p,&work)==-1&&q->calls==calls);q->busy=1;assert(bn_transformer_bf16_mla_moe_submit(q,&work)==BN_GPU_MOE_BANK_BUSY&&work.pending&&!work.failed&&q->calls==calls+1);q->acquire_fail=1;assert(bn_transformer_bf16_mla_moe_submit(q,&work)==-1&&work.failed);
 *q=(MlaProbe){0};work=(BnTransformerBF16MLAMoEBlockWork){0};assert(!bn_transformer_bf16_mla_moe_begin(q,&p,&work));assert(!bn_transformer_bf16_mla_moe_submit(q,&work)&&!work.pending&&!work.failed&&work.moe.submitted_banks==4&&work.moe.submitted_experts==8);assert(work.moe.metadata_read_bytes==1172&&!host.buf&&!host.buf2&&!host.buf5);assert(bn_transformer_bf16_mla_moe_submit(q,&work)==-1);
 *g=saved;w->dim=iw->dim=64;puts("  MLA/MoE block preflight, all begin failures, status gate and BUSY continuation PASSED");
}
static void test_indexed_mla_orchestration(void){
 MlaProbe probe={0};BnGPUBackend g={.ctx=&probe,.linear_a8_bf16_resident=mla_linear,.rmsnorm_bf16_resident=mla_norm,.head_linear_bf16_resident=mla_head,.mla_fp8_cache_append_resident=mla_append,.mla_fp8_attention_resident=mla_attention,.linear_bf16_f32_weight_resident=mla_idx_head,.layernorm128_f32_bf16_resident=mla_idx_norm,.fwht128_fp8_resident=mla_idx_fp8,.indexer_weight_scale_resident=mla_idx_scale,.kpool_fp8_resident=mla_idx_pool,.indexer_fp8_append_resident=mla_idx_append,.indexer_fp8_paged_score_resident=mla_idx_score,.indexer_select_resident=mla_idx_select};
 BnBackendBF16MLAWeights w={.gpu=&g,.q_norm=&g,.kv_norm=&g,.key=&g,.value=&g,.dim=64,.q_rank=1536,.kv_rank=512,.heads=64,.key_dim=256,.value_dim=256,.norm_eps=1e-5f,.scale=.0625f};w.q_a=w.q_b=w.kv_a=w.output=(BnGPUBF16LinearWeight){.raw=&g,.type=BN_GGUF_TENSOR_Q8_0};
 BnBackendBF16IndexerWeights iw={.gpu=&g,.head_weights=&g,.norm=&g,.bias=&g,.ape=&g,.dim=64,.q_rank=1536,.heads=32,.head_dim=128,.pool_size=4,.norm_eps=1e-6f,.weight_scale=.015625f};iw.query=iw.key=iw.gate=w.q_a;
 for(int mode=0;mode<2;++mode){size_t n=2;int pos=mode?4096:3,mb=mode?65:1,ib=mode?16:1;size_t end=(pos+2)/4,groups=end-pos/4,split=n*64*34;
  size_t counts[]={n*64,n*1536,n*1536,n*512,n*512,n*64*256,n*64*512,n,(size_t)mb*64*656,1024,1024,n*4096,n*128,n*128,n*128,n*64,n*32,n*4096,n*32,n*32,groups*128,groups,(size_t)ib*8448,(size_t)ib,groups,mode?n*end:0,n*512,n*2051,1,(size_t)mb,n*2176,n*64*512,n*64,split*512,split,n*64*256,n*64};
  int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16};
  BnGPUResidentView v[37]={0};for(int j=0;j<37;++j)if(counts[j])v[j]=(BnGPUResidentView){.request=&probe,.slot=(uint32_t)j,.count=counts[j],.type=types[j]};
  BnTransformerBF16IndexedMLAPlan p={.prepare={.prepare={.weights=&w,.n_tokens=2,.x=v[0],.q_raw=v[1],.qr=v[2],.kv_raw=v[3],.kv=v[4],.q=v[5],.query=v[6]},.num_cache_blocks=mb,.slots=v[7],.cache=v[8]},.indexer={.indexer={.weights=&iw,.n_tokens=2,.position=pos,.x=v[0],.qr=v[2],.initial_tail=v[9],.next_tail=v[10],.q=v[11],.key_raw=v[12],.key=v[13],.gate=v[14],.cast=v[15],.head_raw=v[16],.q_fp8=v[17],.q_scales=v[18],.head_weights=v[19],.compressed=v[20],.compressed_scales=v[21]},.num_cache_blocks=ib,.logical_blocks=ib,.short_decode=!mode,.cache=v[22],.block_table=v[23],.slots=v[24],.scores=v[25],.pools=v[26],.out=v[27],.status=v[28]},.output={.weights=&w,.attention={.n_tokens=2,.position=pos,.num_cache_blocks=mb,.logical_blocks=mb,.query=v[6],.cache=v[8],.selection=v[27],.status=v[28],.block_table=v[29],.physical=v[30],.out=v[31],.lse=v[32],.partial=v[33],.partial_lse=v[34]},.values=v[35],.out=v[36]}},bad;
  int calls=mode?21:20;for(int fail=0;fail<=calls;++fail){probe=(MlaProbe){.fail=fail};assert(bn_transformer_bf16_indexed_mla(&probe,&p)==(fail?-1:0)&&probe.calls==(fail?fail:calls));}probe=(MlaProbe){0};
#define INDEXED_REJECT(change) do{bad=p;change;assert(bn_transformer_bf16_indexed_mla(&probe,&bad)==-1&&!probe.calls);}while(0)
  INDEXED_REJECT(bad.indexer.indexer.qr=bad.prepare.prepare.q_raw);INDEXED_REJECT(bad.output.attention.query.offset=8);INDEXED_REJECT(bad.output.attention.selection.slot=40);INDEXED_REJECT(bad.output.attention.status.slot=41);INDEXED_REJECT(bad.indexer.indexer.key_raw.slot=1);INDEXED_REJECT(bad.indexer.cache.slot=8);INDEXED_REJECT(bad.indexer.indexer.next_tail.slot=4);INDEXED_REJECT(bad.output.attention.position++);INDEXED_REJECT(bad.indexer.scores.count++);INDEXED_REJECT(bad.output.out.count--);
  for(int j=0;j<36;++j)if(v[j].count)INDEXED_REJECT(bad.output.out.slot=v[j].slot);
  g.indexer_select_resident=NULL;INDEXED_REJECT((void)0);g.indexer_select_resident=mla_idx_select;
#undef INDEXED_REJECT
 test_mla_moe_composition(&probe,&g,p,&w,&iw,mode);
 }
 puts("  Indexed MLA exact links, cross-stage aliases and all 20/21 submission failures PASSED");
}
static void test_mla_attention_orchestration(void){
 MlaProbe probe={0};BnGPUBackend g={.ctx=&probe,.mla_fp8_attention_resident=mla_attention,.linear_a8_bf16_resident=mla_linear,.head_linear_bf16_resident=mla_head};
 BnBackendBF16MLAWeights w={.gpu=&g,.q_norm=&g,.kv_norm=&g,.key=&g,.value=&g,.dim=64,.q_rank=1536,.kv_rank=512,.heads=64,.key_dim=256,.value_dim=256,.norm_eps=1e-5f,.scale=.0625f};w.q_a=w.q_b=w.kv_a=w.output=(BnGPUBF16LinearWeight){.raw=&g,.type=BN_GGUF_TENSOR_Q8_0};
 for(int mode=0;mode<2;++mode){size_t n=mode?65:2,split=n<=64?n*64*34:0;
  size_t counts[]={n*64*512,2*64*656,n*2051,2,n*2176,n*64*512,n*64,split*512,split,1,n*64*256,n*64};
  int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16};BnGPUResidentView v[12]={0};for(int j=0;j<12;++j)if(counts[j])v[j]=(BnGPUResidentView){.request=&probe,.slot=(uint32_t)j,.count=counts[j],.type=types[j]};
  BnTransformerBF16MLAAttentionPlan p={.weights=&w,.attention={.n_tokens=(int)n,.num_cache_blocks=2,.logical_blocks=2,.query=v[0],.cache=v[1],.selection=v[2],.block_table=v[3],.physical=v[4],.out=v[5],.lse=v[6],.partial=v[7],.partial_lse=v[8],.status=v[9]},.values=v[10],.out=v[11]},bad;
  for(int fail=0;fail<=3;++fail){probe=(MlaProbe){.fail=fail};assert(bn_transformer_bf16_mla_attention(&probe,&p)==(fail?-1:0)&&probe.calls==(fail?fail:3));}
  probe=(MlaProbe){0};
#define MLA_ATT_REJECT(change) do{bad=p;change;assert(bn_transformer_bf16_mla_attention(&probe,&bad)==-1&&!probe.calls);}while(0)
  MLA_ATT_REJECT(bad.attention.n_tokens=0);MLA_ATT_REJECT(bad.attention.position=-1);MLA_ATT_REJECT(bad.attention.position=INT_MAX);MLA_ATT_REJECT(bad.attention.logical_blocks=0);MLA_ATT_REJECT(bad.attention.num_cache_blocks=INT_MAX);MLA_ATT_REJECT(bad.attention.physical.count--);MLA_ATT_REJECT(bad.attention.query.offset=1);MLA_ATT_REJECT(bad.attention.status.request=&g);MLA_ATT_REJECT(bad.out=bad.attention.query);MLA_ATT_REJECT(bad.values=bad.attention.out);
  for(int j=0;j<10;++j)if(v[j].count){MLA_ATT_REJECT(bad.out.slot=v[j].slot);MLA_ATT_REJECT(bad.values.slot=v[j].slot);}
  g.mla_fp8_attention_resident=NULL;MLA_ATT_REJECT((void)0);g.mla_fp8_attention_resident=mla_attention;
  w.scale=1.f;MLA_ATT_REJECT((void)0);w.scale=.0625f;
  if(mode){MLA_ATT_REJECT(bad.attention.partial.request=&probe);MLA_ATT_REJECT(bad.attention.partial_lse.count=1);}
#undef MLA_ATT_REJECT
 }
 puts("  MLA attention/output global preflight, decode/prefill scratch and submission failures PASSED");
}
static void test_mla_orchestration(void){
 test_indexed_mla_orchestration();
 test_mla_attention_orchestration();
 MlaProbe probe={0};BnGPUBackend g={.ctx=&probe,.linear_a8_bf16_resident=mla_linear,.rmsnorm_bf16_resident=mla_norm,.head_linear_bf16_resident=mla_head};BnBackendBF16MLAWeights w={.gpu=&g,.q_norm=&g,.kv_norm=&g,.key=&g,.value=&g,.dim=64,.q_rank=1536,.kv_rank=512,.heads=2,.key_dim=32,.value_dim=32,.norm_eps=1e-5f};w.q_a=w.q_b=w.kv_a=w.output=(BnGPUBF16LinearWeight){.raw=&g,.type=BN_GGUF_TENSOR_Q8_0};size_t counts[]={128,3072,3072,1024,1024,128,2048};BnGPUResidentView v[7];for(int j=0;j<7;++j)v[j]=(BnGPUResidentView){.request=&probe,.slot=(uint32_t)j,.count=counts[j],.type=BN_GGUF_TENSOR_BF16};BnTransformerBF16MLAPreparePlan p={.weights=&w,.n_tokens=2,.x=v[0],.q_raw=v[1],.qr=v[2],.kv_raw=v[3],.kv=v[4],.q=v[5],.query=v[6]};
 for(int fail=0;fail<=6;++fail){probe=(MlaProbe){.fail=fail};assert(bn_transformer_bf16_mla_prepare(&probe,&p)==(fail?-1:0)&&probe.calls==(fail?fail:6));}
 probe=(MlaProbe){0};BnTransformerBF16MLAPreparePlan b=p;w.q_rank=1024;assert(bn_transformer_bf16_mla_prepare(&probe,&p)==-1&&!probe.calls);w.q_rank=1536;b.query.count--;assert(bn_transformer_bf16_mla_prepare(&probe,&b)==-1);b=p;b.qr.offset=1;assert(bn_transformer_bf16_mla_prepare(&probe,&b)==-1);b=p;b.q_raw.request=&g;assert(bn_transformer_bf16_mla_prepare(&probe,&b)==-1);
 for(int j=0;j<6;++j){b=p;b.query.slot=v[j].slot;assert(bn_transformer_bf16_mla_prepare(&probe,&b)==-1&&!probe.calls);}
 BnTransformerBF16MLACachePlan cache={.prepare=p,.num_cache_blocks=1,.slots={.request=&probe,.slot=7,.count=2,.type=BN_GGUF_TENSOR_I32},.cache={.request=&probe,.slot=8,.count=64*656,.type=BN_GGUF_TENSOR_I8}},bad;
 g.mla_fp8_cache_append_resident=mla_append;
 for(int fail=0;fail<=7;++fail){probe=(MlaProbe){.fail=fail};assert(bn_transformer_bf16_mla_cache(&probe,&cache)==(fail?-1:0)&&probe.calls==(fail?fail:7));}
 probe=(MlaProbe){0};
#define MLA_CACHE_REJECT(change) do{bad=cache;change;assert(bn_transformer_bf16_mla_cache(&probe,&bad)==-1&&!probe.calls);}while(0)
 MLA_CACHE_REJECT(bad.num_cache_blocks=0);MLA_CACHE_REJECT(bad.num_cache_blocks=INT_MAX);MLA_CACHE_REJECT(bad.cache.offset=1);MLA_CACHE_REJECT(bad.slots.count--);MLA_CACHE_REJECT(bad.cache.count--);MLA_CACHE_REJECT(bad.slots.request=&g);MLA_CACHE_REJECT(bad.cache.type=BN_GGUF_TENSOR_F32);
 for(int j=0;j<7;++j){MLA_CACHE_REJECT(bad.cache.slot=v[j].slot);MLA_CACHE_REJECT(bad.slots.slot=v[j].slot);}MLA_CACHE_REJECT(bad.slots.slot=bad.cache.slot);
 g.mla_fp8_cache_append_resident=NULL;MLA_CACHE_REJECT((void)0);g.mla_fp8_cache_append_resident=mla_append;
#undef MLA_CACHE_REJECT
 BnTransformerBF16MLAOutputPlan out={.weights=&w,.n_tokens=2,.context=v[6],.values=v[5],.out=v[0]};for(int fail=0;fail<=2;++fail){probe=(MlaProbe){.fail=fail};assert(bn_transformer_bf16_mla_output(&probe,&out)==(fail?-1:0)&&probe.calls==(fail?fail:2));}
 probe=(MlaProbe){0};out.out=out.values;assert(bn_transformer_bf16_mla_output(&probe,&out)==-1&&!probe.calls);puts("  MLA prepare/output global aliases and all submission failures PASSED");
}
