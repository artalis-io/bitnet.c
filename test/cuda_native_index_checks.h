#include "cuda_native_index_reference.h"
static void run_native_index_reference(BnGPUBackend*g) {
    assert(g->index_query_score_f16_resident&&g->index_score_f16_resident);void*w[2];
    for(int j=0;j<2;j++){F16FFNSource source={128,j?32:4096,j?17:13,j};size_t tile=32*128*2;BnGPUF16RowsPlan p={source.n,128,(size_t)source.n*128*2+tile,tile,&source,f16_ffn_read};w[j]=g->buffer_create_f16_rows(g->ctx,&p);assert(w[j]);}
    size_t calls=0;
    for(size_t c=0;c<sizeof(native_index_fixtures)/sizeof(*native_index_fixtures);c++) {
        int R=native_index_fixtures[c].R,T=native_index_fixtures[c].T,pages=native_index_fixtures[c].pages,epp=native_index_fixtures[c].epp,stride=native_index_fixtures[c].stride;
        size_t K=pages?(size_t)pages*epp:(size_t)T;uint16_t*x=malloc((size_t)R*128*2),*qr=malloc((size_t)R*128*2),*keys=malloc(K*128*2);int32_t*bt=pages?malloc((size_t)pages*4):NULL;assert(x&&qr&&keys&&(!pages||bt));
        for(size_t i=0;i<(size_t)R*128;i++){x[i]=bn_fp32_to_fp16((float)((int)((i*37+41)%1021)-510)/1024);qr[i]=bn_fp32_to_fp16((float)((int)((i*73+19)%1021)-510)/1024);}
        memset(keys,0xa5,K*128*2);for(int i=0;i<pages;i++)bt[i]=pages-1-i;
        /* The source recipe applies the modulus after adding11. */
        for(int n=0;n<T;n++)for(int d=0;d<128;d++){size_t row=pages?(size_t)bt[n/epp]*epp+n%epp:(size_t)n;keys[row*128+d]=bn_fp32_to_fp16((float)(((n*128+d)*17+11)%257-128)/64);}
        size_t key_shift=(c%2)?2:0;uint16_t*shifted_keys=malloc((K*128+key_shift)*2);assert(shifted_keys);
        memset(shifted_keys,0xa5,key_shift*2);memcpy(shifted_keys+key_shift,keys,K*128*2);
        void*r[2];BnGPUF16IndexQueryScoreResidentPlan p[2];BnGPUResidentView roots[2][8];size_t counts[]={(size_t)R*128,(size_t)R*128,(size_t)R*4096,(size_t)R*32,K*128,(size_t)pages,32,(size_t)R*stride};
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[8];const void*data[]={qr,x,NULL,NULL,shifted_keys,bt,native_rope_freq1,NULL};
            for(int j=0;j<8;j++)v[j]=counts[j]?native_router_guard(g,r[q],j==5?BN_GGUF_TENSOR_I32:j==6?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_F16,counts[j]+(j==4?key_shift:0),data[j],&roots[q][j]):(BnGPUResidentView){0};
            if(key_shift)v[4]=bn_gpu_resident_view_slice(v[4],key_shift,counts[4]);
            p[q]=(BnGPUF16IndexQueryScoreResidentPlan){{{w[0],128,4096,R,v[0],v[2]},{w[1],128,32,R,v[1],v[3]},
                {R,32,64,128,64,64,0,native_index_fixtures[c].pos,1e-20f,NULL,v[2],v[6],{0}}},
                {R,T,native_index_fixtures[c].pos,native_index_fixtures[c].rate,stride,pages,epp,v[2],v[3],v[4],v[5],v[7]}};
            assert(g->index_query_score_f16_resident(g->ctx,r[q],p+q)==-1);assert(!g->linear_f16_reserve(g->ctx,r[q]));BnGPUF16IndexQueryScoreResidentPlan bad;
#define INDEX_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->index_query_score_f16_resident(g->ctx,r[q],&bad)==-1);}while(0)
            INDEX_BAD(score.keys.offset,p[q].score.keys.offset+1);INDEX_BAD(score.n_keys,0);INDEX_BAD(score.pos0,INT_MAX);INDEX_BAD(score.score_stride,0);INDEX_BAD(score.rate,0);INDEX_BAD(score.scores.count,counts[7]-1);INDEX_BAD(query.query_rope.norm,w[0]);INDEX_BAD(query.query_rope.slice_offset,0);INDEX_BAD(query.weights.out,p[q].query.query.out);INDEX_BAD(score.keys,p[q].score.queries);INDEX_BAD(score.block_table.count,(size_t)pages+1);
#undef INDEX_BAD
            for(int j=2;j<4;j++)native_router_guard_check(g,r[q],roots[q][j],counts[j],BN_GGUF_TENSOR_F16,folded_test_unwritten(counts[j]*2));
            native_router_guard_check(g,r[q],roots[q][7],counts[7],BN_GGUF_TENSOR_F16,folded_test_unwritten(counts[7]*2));
        }
        BnGPUF16IndexQueryScoreResidentPlan bad=p[0];bad.score.keys=p[1].score.keys;assert(g->index_query_score_f16_resident(g->ctx,r[0],&bad)==-1);
        for(int q=0;q<2;q++) {
            assert(!g->index_query_score_f16_resident(g->ctx,r[q],p+q));
            for(int j=0;j<3;j++)native_router_guard_check(g,r[q],roots[q][j==2?7:2+j],counts[j==2?7:2+j],BN_GGUF_TENSOR_F16,native_index_fixtures[c].hash[j]);
            native_router_guard_check(g,r[q],roots[q][0],counts[0],BN_GGUF_TENSOR_F16,native_norm_hash(qr,counts[0]*2));native_router_guard_check(g,r[q],roots[q][1],counts[1],BN_GGUF_TENSOR_F16,native_norm_hash(x,counts[1]*2));native_router_guard_check(g,r[q],roots[q][4],counts[4]+key_shift,BN_GGUF_TENSOR_F16,native_norm_hash(shifted_keys,(counts[4]+key_shift)*2));
            if(pages){native_router_guard_check(g,r[q],roots[q][5],pages,BN_GGUF_TENSOR_I32,native_norm_hash(bt,(size_t)pages*4));}calls++;
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}free(x);free(qr);free(keys);free(shifted_keys);free(bt);
    }
    for(int j=0;j<2;j++){g->buffer_destroy(g->ctx,w[j]);}printf("CUDA source F16 index query/score PASSED: %zu calls\n",calls);
}
