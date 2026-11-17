#include "cuda_native_compress_reference.h"
static uint16_t *native_compress_input(int pos,int R,int D,int gate) {
    uint16_t*x=malloc((size_t)R*D*2);assert(x);
    for(int row=0;row<R;row++)for(int c=0;c<D;c++) {
        int v=gate?((pos+row)*23+c*41+3)%509-254:((pos+row)*17+c*73+19)%1021-510;
        x[(size_t)row*D+c]=bn_fp32_to_fp16((float)v/(gate?128:1024));
    }
    return x;
}
static void run_native_compress_reference(BnGPUBackend*g) {
    assert(g->compress_f16_resident&&g->compressor_f16_resident);size_t calls=0;
    for(int config=0;config<8;config++) {
        const size_t first=(size_t)config*12;int D=native_compress_fixtures[first].D,m=native_compress_fixtures[first].m;
        int B=256+m,split=native_compress_fixtures[first].split,relative=config%2,projected=config>=4;
        uint16_t*zero=calloc((size_t)B*D,2),*norm=malloc((size_t)D*2);float*ape=malloc((size_t)m*D*4),freq[16];assert(zero&&norm&&ape);
        for(int c=0;c<D;c++)norm[c]=bn_fp32_to_fp16((float)((c*7+11)%33-16)/16);
        for(int c=0;c<m*D;c++)ape[c]=(float)((c*11+5)%29-14)/32;
        for(int c=0;c<16;c++)freq[c]=config%3?1.0f/(float)(1<<(c%8)):0;
        void*w[2]={NULL,NULL};
        if(projected)for(int role=0;role<(m>1?2:1);role++) {
            F16FFNSource source={128,D,13,role};size_t tile=(size_t)32*128*2;
            BnGPUF16RowsPlan wp={D,128,(size_t)D*128*2+tile,tile,&source,f16_ffn_read};w[role]=g->buffer_create_f16_rows(g->ctx,&wp);assert(w[role]);
        }
        void*r[2];BnGPUCompressF16ResidentPlan p[2];BnGPUResidentView state[2][7];
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);if(projected)assert(!g->linear_f16_reserve(g->ctx,r[q]));
            BnGPUResidentView v[7];size_t counts[]={(size_t)B*D,(size_t)B*D,(size_t)m*D,(size_t)D,16,(size_t)1024*split,(size_t)1024*(D-split)};
            int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16};const void*data[]={zero,zero,ape,norm,freq,NULL,NULL};
            for(int j=0;j<7;j++)v[j]=counts[j]?native_router_guard(g,r[q],types[j],counts[j],data[j],&state[q][j]):(BnGPUResidentView){0};
            p[q]=(BnGPUCompressF16ResidentPlan){1,0,m,D,32,B,1024,split,relative,1e-20f,{0},{0},v[0],v[1],v[2],v[3],v[4],v[5],v[6]};
        }
        for(size_t f=first;f<first+12;f++) {
            int R=native_compress_fixtures[f].R,pos=native_compress_fixtures[f].pos;size_t n=(size_t)R*D;
            uint16_t*x=native_compress_input(pos,R,projected?128:D,0),*gate=projected?NULL:native_compress_input(pos,R,D,1);
            for(int q=0;q<2;q++) {
                BnGPUResidentView sx,sk,sg;BnGPUResidentView xv=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,(size_t)R*(projected?128:D),x,&sx);
                p[q].n_tokens=R;p[q].pos0=pos;
                p[q].kv=projected?native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,n,NULL,&sk):xv;
                p[q].gate=projected?(m>1?native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,n,NULL,&sg):p[q].kv):native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,n,gate,&sg);
                BnGPUF16CompressorResidentPlan cp={{w[0],128,D,R,xv,p[q].kv},{0},p[q]};
                if(projected&&m>1)cp.gate=(BnGPUF16LinearResidentPlan){w[1],128,D,R,xv,p[q].gate};
                BnGPUCompressF16ResidentPlan bad=p[q];bad.dest_rows=0;assert(g->compress_f16_resident(g->ctx,r[q],&bad)==-1);
                bad=p[q];bad.ring_gate=bad.ring_kv;assert(g->compress_f16_resident(g->ctx,r[q],&bad)==-1);
                bad=p[q];bad.norm=p[1-q].norm;assert(g->compress_f16_resident(g->ctx,r[q],&bad)==-1);
                if(projected) {
                    BnGPUF16CompressorResidentPlan cb=cp;cb.compress.dest_a.count--;assert(g->compressor_f16_resident(g->ctx,r[q],&cb)==-1);
                    native_router_guard_check(g,r[q],sk,n,BN_GGUF_TENSOR_F16,folded_test_unwritten(n*2));
                    assert(!g->compressor_f16_resident(g->ctx,r[q],&cp));
                } else assert(!g->compress_f16_resident(g->ctx,r[q],p+q));
                native_router_guard_check(g,r[q],projected?sk:sx,n,BN_GGUF_TENSOR_F16,native_compress_fixtures[f].hash[0]);
                native_router_guard_check(g,r[q],projected&&m==1?sk:sg,n,BN_GGUF_TENSOR_F16,native_compress_fixtures[f].hash[1]);
                for(int j=0;j<2;j++)native_router_guard_check(g,r[q],state[q][j],(size_t)B*D,BN_GGUF_TENSOR_F16,native_compress_fixtures[f].hash[2+j]);
                native_router_guard_check(g,r[q],state[q][5],(size_t)1024*split,BN_GGUF_TENSOR_F16,native_compress_fixtures[f].hash[4]);
                if(split<D)native_router_guard_check(g,r[q],state[q][6],(size_t)1024*(D-split),BN_GGUF_TENSOR_F16,native_compress_fixtures[f].hash[5]);
                native_router_guard_check(g,r[q],state[q][2],(size_t)m*D,BN_GGUF_TENSOR_F32,native_norm_hash(ape,(size_t)m*D*4));
                native_router_guard_check(g,r[q],state[q][3],D,BN_GGUF_TENSOR_F16,native_norm_hash(norm,(size_t)D*2));
                native_router_guard_check(g,r[q],state[q][4],16,BN_GGUF_TENSOR_F32,native_norm_hash(freq,64));calls++;
            }
            free(x);free(gate);
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}
        for(int role=0;role<2;role++){if(w[role])g->buffer_destroy(g->ctx,w[role]);}
        free(zero);free(norm);free(ape);
    }
    printf("CUDA source stateful compressor reference PASSED: %zu calls\n",calls);
}
