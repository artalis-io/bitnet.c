#include "cuda_native_qkv_reference.h"
static void *native_qkv_gamma(BnGPUBackend*g,int D,int bf16) {
    uint16_t*b=malloc((size_t)D*2);assert(b);
    for(int i=0;i<D;i++){float v=(float)((i*7+11)%33-16)/16;uint32_t bits;memcpy(&bits,&v,4);b[i]=bf16?(uint16_t)(bits>>16):bn_fp32_to_fp16(v);}
    NativeNormWeightSource source={(const unsigned char*)b,D,2};BnGPURawRowsPlan p={bf16?BN_GGUF_TENSOR_BF16:BN_GGUF_TENSOR_F16,1,D,(size_t)D*2,(size_t)D*2,&source,native_norm_read_weight};
    void*w=g->buffer_create_raw_rows(g->ctx,&p);free(b);assert(w);return w;
}
static void run_native_rope_reference(BnGPUBackend*g) {
    assert(g->rope_gptj_f16_resident);size_t executions=0;
    for(size_t c=0;c<sizeof(native_rope_fixtures)/sizeof(*native_rope_fixtures);c++) {
        const int R=native_rope_fixtures[c].R,H=native_rope_fixtures[c].H,D=native_rope_fixtures[c].D,S=native_rope_fixtures[c].S;
        size_t n=(size_t)R*H*S;uint16_t*x=folded_test_input(1,R,H*S);assert(native_norm_hash(x,n*2)==native_rope_fixtures[c].input);
        void*weight=native_rope_fixtures[c].weight?native_qkv_gamma(g,D,native_rope_fixtures[c].weight==2):NULL;
        int32_t*ids=malloc((size_t)R*4);assert(ids);for(int i=0;i<R;i++)ids[i]=2147483000-i*7919;
        void*r[2];BnGPUGPTJF16ResidentPlan p[2];BnGPUResidentView storage[2][3];
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);p[q]=(BnGPUGPTJF16ResidentPlan){R,H,D,S,native_rope_fixtures[c].offset,native_rope_fixtures[c].rotary,native_rope_fixtures[c].rotate_offset,native_rope_fixtures[c].pos,1e-6f,weight,{0},{0},{0}};
            p[q].x=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,n,x,&storage[q][0]);
            const uint32_t*f=native_rope_fixtures[c].freq?native_rope_freq1:native_rope_freq0;
            p[q].inv_freq=native_router_guard(g,r[q],BN_GGUF_TENSOR_F32,(size_t)p[q].rotary_dim/2,f,&storage[q][1]);
            if(native_rope_fixtures[c].explicit_ids)p[q].positions=native_router_guard(g,r[q],BN_GGUF_TENSOR_I32,R,ids,&storage[q][2]);
            BnGPUGPTJF16ResidentPlan bad;
#define ROPE_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->rope_gptj_f16_resident(g->ctx,r[q],&bad)==-1);}while(0)
            ROPE_BAD(dim,0);ROPE_BAD(dim,INT_MAX);ROPE_BAD(head_stride,INT_MIN);ROPE_BAD(slice_offset,1);ROPE_BAD(rotary_dim,D+2);ROPE_BAD(rotate_offset,INT_MAX);ROPE_BAD(pos0,-1);ROPE_BAD(eps,0);ROPE_BAD(eps,NAN);ROPE_BAD(x.count,n-1);ROPE_BAD(x.offset,p[q].x.offset+1);ROPE_BAD(inv_freq,p[q].x);ROPE_BAD(inv_freq.count,p[q].inv_freq.count+1);ROPE_BAD(positions.count,(size_t)R+1);
#undef ROPE_BAD
            native_router_guard_check(g,r[q],storage[q][0],n,BN_GGUF_TENSOR_F16,native_rope_fixtures[c].input);
        }
        BnGPUGPTJF16ResidentPlan bad=p[0];bad.inv_freq=p[1].inv_freq;assert(g->rope_gptj_f16_resident(g->ctx,r[0],&bad)==-1);
        for(int q=0;q<2;q++)for(int repeat=0;repeat<2;repeat++) {
            assert(!g->resident_write(r[q],p[q].x,x)&&!g->rope_gptj_f16_resident(g->ctx,r[q],&p[q]));
            native_router_guard_check(g,r[q],storage[q][0],n,BN_GGUF_TENSOR_F16,native_rope_fixtures[c].output);
            const uint32_t*f=native_rope_fixtures[c].freq?native_rope_freq1:native_rope_freq0;
            native_router_guard_check(g,r[q],storage[q][1],(size_t)p[q].rotary_dim/2,BN_GGUF_TENSOR_F32,native_norm_hash(f,(size_t)p[q].rotary_dim*2));
            if(p[q].positions.request)native_router_guard_check(g,r[q],storage[q][2],R,BN_GGUF_TENSOR_I32,native_norm_hash(ids,(size_t)R*4));
            executions++;
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}
        if(weight){g->buffer_destroy(g->ctx,weight);}
        free(ids);free(x);
    }
    printf("CUDA source GPT-J strided/pair-norm reference PASSED: %zu executions\n",executions);
}
static void run_native_qkv_reference(BnGPUBackend*g) {
    assert(g->latent_qkv_f16_resident);size_t executions=0;void*weights[3];
    for(int role=0;role<3;role++){int N=role==1?384:128;F16FFNSource source={128,N,13,role};size_t tile=(size_t)32*128*2;BnGPUF16RowsPlan p={N,128,(size_t)N*128*2+tile,tile,&source,f16_ffn_read};weights[role]=g->buffer_create_f16_rows(g->ctx,&p);assert(weights[role]);}
    void*gamma=native_qkv_gamma(g,128,1);
    for(size_t c=0;c<sizeof(native_qkv_fixtures)/sizeof(*native_qkv_fixtures);c++) {
        int R=native_qkv_fixtures[c].R;size_t nx=(size_t)R*128,nq=(size_t)R*384;uint16_t*x=folded_test_input(1,R,128);int32_t*ids=malloc((size_t)R*4);assert(ids);for(int i=0;i<R;i++)ids[i]=1048575-i*7919;
        void*r[2];BnGPUF16LatentQKVResidentPlan p[2];BnGPUResidentView storage[2][7];
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[7];
            for(int j=0;j<5;j++)v[j]=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,j==3?nq:nx,j==0?x:NULL,&storage[q][j]);
            v[5]=native_router_guard(g,r[q],BN_GGUF_TENSOR_F32,16,native_rope_freq1,&storage[q][5]);v[6]=(BnGPUResidentView){0};
            if(native_qkv_fixtures[c].explicit_ids)v[6]=native_router_guard(g,r[q],BN_GGUF_TENSOR_I32,R,ids,&storage[q][6]);
            p[q]=(BnGPUF16LatentQKVResidentPlan){{weights[0],128,128,R,v[0],v[1]},{weights[1],128,384,R,v[2],v[3]},{weights[2],128,128,R,v[0],v[4]},
                {R,128,1,1e-6f,0,1,gamma,v[1],v[2]},
                {R,3,32,128,96,32,0,65535,1e-6f,NULL,v[3],v[5],v[6]},
                {R,1,128,128,0,32,96,65535,1e-6f,gamma,v[4],v[5],v[6]}};
            assert(g->latent_qkv_f16_resident(g->ctx,r[q],p+q)==-1);assert(!g->linear_f16_reserve(g->ctx,r[q]));
            BnGPUF16LatentQKVResidentPlan bad;
#define QKV_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->latent_qkv_f16_resident(g->ctx,r[q],&bad)==-1);}while(0)
            QKV_BAD(kv_rope.norm,NULL);QKV_BAD(q_rope.norm,gamma);QKV_BAD(q_rope.head_stride,INT_MIN);QKV_BAD(kv_rope.rotate_offset,INT_MAX);QKV_BAD(q_norm.constant_scale,2);QKV_BAD(q_norm.out,p[q].q_a.out);QKV_BAD(q_norm.eps,1e-5f);QKV_BAD(q_b.n,383);QKV_BAD(kv.weight,NULL);QKV_BAD(kv_rope.pos0,1);QKV_BAD(q_rope.inv_freq,p[q].q_a.x);
#undef QKV_BAD
            for(int j=1;j<5;j++)native_router_guard_check(g,r[q],storage[q][j],j==3?nq:nx,BN_GGUF_TENSOR_F16,folded_test_unwritten((j==3?nq:nx)*2));
        }
        BnGPUF16LatentQKVResidentPlan bad=p[0];bad.kv.x=p[1].kv.x;assert(g->latent_qkv_f16_resident(g->ctx,r[0],&bad)==-1);
        for(int q=0;q<2;q++)for(int repeat=0;repeat<2;repeat++) {
            assert(!g->latent_qkv_f16_resident(g->ctx,r[q],p+q));
            for(int j=1;j<5;j++)native_router_guard_check(g,r[q],storage[q][j],j==3?nq:nx,BN_GGUF_TENSOR_F16,native_qkv_fixtures[c].hash[j-1]);
            native_router_guard_check(g,r[q],storage[q][0],nx,BN_GGUF_TENSOR_F16,native_norm_hash(x,nx*2));
            native_router_guard_check(g,r[q],storage[q][5],16,BN_GGUF_TENSOR_F32,native_norm_hash(native_rope_freq1,64));
            if(p[q].q_rope.positions.request)native_router_guard_check(g,r[q],storage[q][6],R,BN_GGUF_TENSOR_I32,native_norm_hash(ids,(size_t)R*4));
            executions++;
        }
        for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}free(ids);free(x);
    }
    for(int j=0;j<3;j++){g->buffer_destroy(g->ctx,weights[j]);}g->buffer_destroy(g->ctx,gamma);
    printf("CUDA source latent QKV composition reference PASSED: %zu executions\n",executions);
}
