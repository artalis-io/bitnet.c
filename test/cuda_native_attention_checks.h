#include "cuda_native_attention_reference.h"
static uint16_t* native_attn_data(size_t count,unsigned seed,int shift) {
    uint16_t*b=malloc(count*2);assert(b);
    for(size_t i=0;i<count;i++)b[i]=bn_fp32_to_fp16((float)((int)((i*seed+seed*13)%1021)-510)/(float)(1u<<shift));
    return b;
}
static void run_native_attention_reference(BnGPUBackend*g,int bounded) {
    assert(g->latent_attention_f16_resident);size_t calls=0,words=0;
    float sinks[64],freq[32];memcpy(sinks,native_attn_sinks,sizeof(sinks));memcpy(freq,native_attn_freq,sizeof(freq));
    NativeNormWeightSource ss={(const unsigned char*)sinks,64,4};BnGPURawRowsPlan upload={BN_GGUF_TENSOR_F32,1,64,256,256,&ss,native_norm_read_weight};
    void*weight=g->buffer_create_raw_rows(g->ctx,&upload);assert(weight);upload.type=BN_GGUF_TENSOR_F16;upload.device_budget_bytes=upload.host_budget_bytes=128;
    ss.width=2;void*badweight=g->buffer_create_raw_rows(g->ctx,&upload);assert(badweight);
    for(size_t ci=0;ci<sizeof(native_attn_fixtures)/sizeof(*native_attn_fixtures);ci++) {
        if(bounded&&ci!=0&&ci!=1&&ci!=2&&ci!=9&&ci!=10)continue;
        int R=native_attn_fixtures[ci].R,T=native_attn_fixtures[ci].T,rate=native_attn_fixtures[ci].rate,dense=native_attn_fixtures[ci].dense,W=native_attn_fixtures[ci].W,pos=native_attn_fixtures[ci].pos,epp=native_attn_fixtures[ci].epp,pages=native_attn_fixtures[ci].pages;
        size_t capacity=(size_t)epp*pages;BnGPUF16LatentAttentionWorkspace ws;assert(!bn_gpu_latent_attention_workspace(R,T,rate,dense,W,pos,&ws));
        uint16_t*q=native_attn_data((size_t)R*64*512,37,10),*ring=W?native_attn_data(768*512,17,9):NULL,*kv=W?native_attn_data((size_t)R*512,73,9):NULL,*pc=native_attn_data(capacity*448,29,9),*pr=native_attn_data(capacity*64,41,9);
        int32_t*bt=malloc((size_t)pages*4),*ids=dense?NULL:malloc((size_t)R*512*4);assert(bt&&(dense||ids));for(int i=0;i<pages;i++)bt[i]=pages-1-i;
        if(ids)for(int row=0;row<R;row++)for(int col=0;col<512;col++){int i=row*512+col,bound=(pos+row+1)/rate;if(bound>T)bound=T;ids[i]=i%11==0||!bound?-1:(i*47+13)%bound;}
        void*r[2];BnGPUResidentView roots[2][11];BnGPUF16LatentAttentionResidentPlan plan[2];
        size_t n[]={(size_t)R*64*512,W?768*512:0,W?(size_t)R*512:0,capacity*448,capacity*64,(size_t)pages,dense?0:(size_t)R*512,native_attn_fixtures[ci].derot?32:0,(size_t)R*64*512,ws.ml_count,ws.acc_count};
        int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32};
        const void*data[]={q,ring,kv,pc,pr,bt,ids,freq,NULL,NULL,NULL};
        for(int s=0;s<2;s++) {
            r[s]=g->resident_request_create(g->ctx);assert(r[s]);BnGPUResidentView v[11]={{0}};
            for(int j=0;j<11;j++)if(n[j])v[j]=native_router_guard(g,r[s],types[j],n[j],j<8?data[j]:NULL,&roots[s][j]);
            plan[s]=(BnGPUF16LatentAttentionResidentPlan){R,T,rate,dense,W,pos,native_attn_fixtures[ci].beg,native_attn_fixtures[ci].floor,W?768:0,pages,epp,native_attn_fixtures[ci].sinks?weight:NULL,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10]};
            BnGPUF16LatentAttentionResidentPlan bad;
#define ATTN_BAD(F,V) do{bad=plan[s];bad.F=(V);assert(g->latent_attention_f16_resident(g->ctx,r[s],&bad)==-1);}while(0)
            ATTN_BAD(n_tokens,0);ATTN_BAD(pool_entries,-1);ATTN_BAD(rate,0);ATTN_BAD(dense_pool,2);ATTN_BAD(window,129);ATTN_BAD(pos0,INT_MAX);ATTN_BAD(ring_beg,pos+1);ATTN_BAD(win_floor,pos+1);ATTN_BAD(ring_rows,1048577);ATTN_BAD(pages,0);ATTN_BAD(entries_per_page,3);ATTN_BAD(sinks,badweight);ATTN_BAD(out,plan[s].queries);ATTN_BAD(queries.count,n[0]-1);ATTN_BAD(queries.slot,plan[s].queries.slot^UINT32_C(0x10000));ATTN_BAD(inv_freq.type,BN_GGUF_TENSOR_F16);
            if(n[10]){ATTN_BAD(scratch_acc.count,n[10]-1);}else{ATTN_BAD(scratch_acc,plan[s].out);}
            if(dense){ATTN_BAD(indices,plan[s].block_table);}else{ATTN_BAD(indices.type,BN_GGUF_TENSOR_F16);}
#undef ATTN_BAD
            native_router_guard_check(g,r[s],roots[s][8],n[8],types[8],folded_test_unwritten(n[8]*2));
        }
        BnGPUF16LatentAttentionResidentPlan bad=plan[0];bad.queries=plan[1].queries;assert(g->latent_attention_f16_resident(g->ctx,r[0],&bad)==-1);
        for(int rep=0;rep<2;rep++)for(int s=0;s<2;s++) {
            assert(!g->latent_attention_f16_resident(g->ctx,r[s],plan+s));native_router_guard_check(g,r[s],roots[s][8],n[8],types[8],native_attn_fixtures[ci].hash);
            for(int j=0;j<8;j++)if(n[j])native_router_guard_check(g,r[s],roots[s][j],n[j],types[j],native_norm_hash(data[j],n[j]*(types[j]==BN_GGUF_TENSOR_F16?2:4)));
            for(int j=9;j<11;j++)if(n[j])native_topk_guard_only(g,r[s],roots[s][j],n[j],types[j]);
            words+=n[8];calls++;
        }
        if(ci==0)for(int s=0;s<2;s++) {
            BnGPUF16LatentAttentionResidentPlan empty=plan[s];empty.pages=empty.entries_per_page=empty.ring_rows=0;
            empty.pool_c=empty.pool_r=empty.block_table=empty.ring=(BnGPUResidentView){0};
            assert(!g->latent_attention_f16_resident(g->ctx,r[s],&empty));
            native_router_guard_check(g,r[s],roots[s][8],n[8],types[8],native_attn_fixtures[ci].hash);
        }
        for(int s=0;s<2;s++){g->resident_request_destroy(r[s]);}free(q);free(ring);free(kv);free(pc);free(pr);free(bt);free(ids);
    }
    g->buffer_destroy(g->ctx,badweight);g->buffer_destroy(g->ctx,weight);
    printf("CUDA source latent attention PASSED: %zu calls, %zu output words, bounded%d\n",calls,words,bounded);
}
