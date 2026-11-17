#include "cuda_decode_exl3_reference.h"
static void decode_exl3_check(BnGPUBackend*g,void*r,BnGPUResidentView storage,size_t used,uint64_t hash) {
    size_t width=storage.type==BN_GGUF_TENSOR_I8?1:storage.type==BN_GGUF_TENSOR_F16?2:4;
    unsigned char*b=malloc(storage.count*width);assert(b&&!g->resident_read(r,storage,b));
    if(hash)assert(native_norm_hash(b+16*width,used*width)==hash);
    for(size_t i=0;i<16*width;i++)assert(b[i]==0xa5&&b[(storage.count-16)*width+i]==0xa5);
    free(b);
}
static void run_decode_exl3_reference(BnGPUBackend*g) {
    assert(g->trellis_decode_resident);size_t executions=0;
    BnGPUDecodeEXL3Workspace untouched={0},badws=untouched;
    assert(bn_gpu_decode_exl3_workspace(9,6,8,128,128,&badws)==-1&&!memcmp(&badws,&untouched,sizeof(badws)));
    assert(bn_gpu_decode_exl3_workspace(1,33,8,128,128,&badws)==-1);
    assert(bn_gpu_decode_exl3_workspace(1,6,5,128,128,&badws)==-1);
    assert(bn_gpu_decode_exl3_workspace(1,6,8,129,128,&badws)==-1);
    void*w[3][8];for(int e=0;e<8;e++)for(int pr=0;pr<3;pr++)w[pr][e]=folded_test_weight(g,128,128,3+14*e+44*pr);
    for(size_t f=0;f<sizeof(decode_exl3_fixtures)/sizeof(*decode_exl3_fixtures);f++) {
        int R=decode_exl3_fixtures[f].R,K=6,D=128,E=8;size_t A=(size_t)R*K;
        BnGPUDecodeEXL3Workspace ws;assert(!bn_gpu_decode_exl3_workspace(R,K,E,D,D,&ws));
        uint16_t*x=folded_test_input(1,R,D),weights[48];int32_t ids[48];
        for(int row=0;row<R;row++)for(int k=0;k<K;k++){int i=row*K+k;ids[i]=(row*3+k)%E;weights[i]=bn_fp32_to_fp16((float)((i*11+3)%127+1)/256);}
        BnGPUDecodeEXL3ResidentPlan p[2];BnGPUResidentView storage[2][13];void*r[2];
        size_t counts[]={(size_t)R*D,A,A,ws.table_bytes,ws.indices64_bytes,ws.had_count,ws.had_count,ws.intermediate_count,ws.intermediate_count,ws.intermediate_count,ws.down_count,ws.counter_count,(size_t)R*D};
        int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F32};
        int32_t*zero=calloc(ws.counter_count,4);assert(zero);
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[13];
            for(int i=0;i<13;i++)v[i]=native_router_guard(g,r[q],types[i],counts[i],i==0?(void*)x:i==1?(void*)ids:i==2?(void*)weights:i==11?(void*)zero:NULL,&storage[q][i]);
            p[q]=(BnGPUDecodeEXL3ResidentPlan){0};p[q].format=BN_QUANT_FORMAT_EXL3_MUL1_3;p[q].codebook_marker=UINT32_C(0x83dcd12d);p[q].tokens=R;p[q].topk=K;p[q].experts=E;p[q].dim=D;p[q].hidden=D;p[q].resident_count=E;p[q].act_limit=(float)decode_exl3_fixtures[f].clip;
            for(int e=0;e<E;e++){p[q].expert_ids[e]=e;for(int pr=0;pr<3;pr++)p[q].weight[pr][e]=w[pr][e];}
            p[q].x=v[0];p[q].indices=v[1];p[q].weights=v[2];p[q].tables=v[3];p[q].indices64=v[4];p[q].had_gate=v[5];p[q].had_up=v[6];p[q].gate=v[7];p[q].up=v[8];p[q].act=v[9];p[q].down=v[10];p[q].counters=v[11];p[q].out=v[12];
            BnGPUDecodeEXL3ResidentPlan bad;
#define DECODE_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->trellis_decode_resident(g->ctx,r[q],&bad)==-1);}while(0)
            DECODE_BAD(tokens,9);DECODE_BAD(topk,33);DECODE_BAD(codebook_marker,0);DECODE_BAD(format,BN_GGUF_TENSOR_Q3_K);DECODE_BAD(resident_count,49);DECODE_BAD(expert_ids[1],0);DECODE_BAD(weight[2][7],NULL);DECODE_BAD(x.count,counts[0]-1);DECODE_BAD(x.offset,p[q].x.offset+1);DECODE_BAD(counters.count,ws.counter_count-1);DECODE_BAD(out,p[q].x);DECODE_BAD(up,p[q].gate);DECODE_BAD(act_limit,-1);
#undef DECODE_BAD
            native_router_guard_check(g,r[q],storage[q][12],counts[12],BN_GGUF_TENSOR_F32,folded_test_unwritten(counts[12]*4));
        }
        BnGPUDecodeEXL3ResidentPlan bad=p[0];bad.x=p[1].x;assert(g->trellis_decode_resident(g->ctx,r[0],&bad)==-1);
        for(int q=0;q<2;q++)for(int repeat=0;repeat<2;repeat++) {
            assert(!g->trellis_decode_resident(g->ctx,r[q],&p[q]));
            const uint64_t*h=decode_exl3_fixtures[f].hash;
            for(int i=0;i<13;i++){uint64_t expected=i>=7&&i<=10?h[i-7]:i==12?h[4]:i==5&&R>1?h[5]:i==6&&R>1?h[6]:0;size_t used=i>=7&&i<=10?A*D:i==12?(size_t)R*D:i==5||i==6?A*D:0;decode_exl3_check(g,r[q],storage[q][i],used,expected);}
            executions++;
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}
        free(zero);free(x);
    }
    for(int e=0;e<8;e++)for(int pr=0;pr<3;pr++)g->buffer_destroy(g->ctx,w[pr][e]);
    printf("CUDA small-batch EXL3 decode reference PASSED: %zu executions\n",executions);
}
