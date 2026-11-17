#include "cuda_f16_ffn_reference.h"
static uint64_t f16_ffn_hash(const void *data,size_t n) {
    const unsigned char *b=data;uint64_t h=UINT64_C(1469598103934665603);
    for(size_t i=0;i<n;i++)h=(h^b[i])*UINT64_C(1099511628211);
    return h;
}
typedef struct {int k,n,seed,role;} F16FFNSource;
static int f16_ffn_read(void *context,int row,int count,uint16_t *out,size_t cap) {
    F16FFNSource*s=context;assert(cap>=(size_t)count*s->k);
    for(int r=0;r<count;r++)for(int k=0;k<s->k;k++) {
        size_t i=(size_t)k*s->n+row+r;
        int v=(int)((i*(37+s->role*2)+(size_t)s->seed*53)%257)-128;
        out[(size_t)r*s->k+k]=bn_fp32_to_fp16((float)v/4096);
    }
    return 0;
}
static BnGPUResidentView f16_ffn_guard(BnGPUBackend*g,void*r,int type,size_t n,
    const void*data,BnGPUResidentView *storage) {
    size_t w=type==BN_GGUF_TENSOR_F32?4:2;unsigned char*b=malloc((n+16)*w);assert(b);
    memset(b,0xa5,(n+16)*w);if(data)memcpy(b+8*w,data,n*w);
    *storage=cuda_test_resident_alloc(g,r,type,n+16,b);free(b);
    return bn_gpu_resident_view_slice(*storage,8,n);
}
static void run_f16_ffn_reference(BnGPUBackend*g) {
    assert(g->ffn_f16_resident&&g->linear_f16_reserve);size_t words=0;
    for(size_t c=0;c<sizeof(f16_ffn_fixtures)/sizeof(f16_ffn_fixtures[0]);c++) {
        const F16FFNFixture*f=&f16_ffn_fixtures[c];int D=f->dim,H=f->hidden,R=f->rows;
        size_t nx=(size_t)R*D,nh=(size_t)R*H;void*weights[3];
        for(int pr=0;pr<3;pr++) {
            int k=pr==2?H:D,n=pr==2?D:H;F16FFNSource source={k,n,f->seed,pr};
            size_t tile=(size_t)(n<32?n:32)*k*2;
            BnGPUF16RowsPlan upload={n,k,(size_t)n*k*2+tile,tile,&source,f16_ffn_read};
            weights[pr]=g->buffer_create_f16_rows(g->ctx,&upload);assert(weights[pr]);
        }
        uint16_t*x=malloc(nx*2);assert(x);
        for(size_t i=0;i<nx;i++) {
            float v=(float)((int)((i*73+19)%1021)-510)/1024;
            if(i/(size_t)D==1)v=0;
            if(i/(size_t)D==2)v*=0x1p-14f;
            v*=f->scale;
            x[i]=bn_fp32_to_fp16(v);
        }
        assert(f16_ffn_hash(x,nx*2)==f->input);
        void*r[2];BnGPUF16FFNResidentPlan p[2];BnGPUResidentView stores[2][5];
        const size_t counts[]={nx,nh,nh,nh,nx};
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[5];
            for(int j=0;j<5;j++)v[j]=f16_ffn_guard(g,r[q],j==4?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_F16,counts[j],j?NULL:x,&stores[q][j]);
            p[q]=(BnGPUF16FFNResidentPlan){
                {weights[0],D,H,R,v[0],v[1]},
                {weights[1],D,H,R,v[0],v[2]},
                {weights[2],H,D,R,v[3],v[4]},BN_GPU_F16_FFN_SILU_HALF,f->limit};
            assert(g->ffn_f16_resident(g->ctx,r[q],&p[q])==-1);
            assert(!g->linear_f16_reserve(g->ctx,r[q])&&!g->linear_f16_reserve(g->ctx,r[q]));
            BnGPUF16FFNResidentPlan bad;
#define F16_FFN_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->ffn_f16_resident(g->ctx,r[q],&bad)==-1);}while(0)
            F16_FFN_BAD(activation_profile,(BnGPUF16FFNActivationProfile)1);
            F16_FFN_BAD(act_limit,-1);F16_FFN_BAD(act_limit,NAN);F16_FFN_BAD(act_limit,65505);
            F16_FFN_BAD(down.weight,NULL);F16_FFN_BAD(down.n,D+1);
            F16_FFN_BAD(up.x,p[q].down.x);F16_FFN_BAD(up.out,p[q].gate.out);
            F16_FFN_BAD(down.x,p[q].up.out);F16_FFN_BAD(down.out.type,BN_GGUF_TENSOR_F16);
            F16_FFN_BAD(gate.out.offset,p[q].gate.out.offset+1);
            F16_FFN_BAD(down.n_tokens,R+1);
#undef F16_FFN_BAD
            for(int j=1;j<5;j++) {
                size_t n=(counts[j]+16)*(j==4?4:2);unsigned char*b=malloc(n);assert(b&&!g->resident_read(r[q],stores[q][j],b));
                for(size_t i=0;i<n;i++)assert(b[i]==0xa5);
                free(b);
            }
        }
        BnGPUF16FFNResidentPlan bad=p[0];bad.up.x=p[1].up.x;
        assert(g->ffn_f16_resident(g->ctx,r[0],&bad)==-1);
        for(int repeat=0;repeat<2;repeat++)for(int q=0;q<2;q++)assert(!g->ffn_f16_resident(g->ctx,r[q],&p[q]));
        for(int q=0;q<2;q++) {
            for(int j=0;j<5;j++) {
                size_t w=j==4?4:2;unsigned char*b=malloc((counts[j]+16)*w);assert(b&&!g->resident_read(r[q],stores[q][j],b));
                for(size_t i=0;i<8*w;i++)assert(b[i]==0xa5&&b[(counts[j]+8)*w+i]==0xa5);
                uint64_t h=f16_ffn_hash(b+8*w,counts[j]*w);
                if(h!=(j?f->output[j-1]:f->input)){fprintf(stderr,"F16 FFN fixture%zu q%d stage%d mismatch\n",c,q,j);abort();}
                free(b);words+=counts[j];
            }
            g->resident_request_destroy(r[q]);
        }
        for(int pr=0;pr<3;pr++)g->buffer_destroy(g->ctx,weights[pr]);
        free(x);
    }
    printf("CUDA source F16 shared FFN exact boundaries/guards/two-request isolation PASSED: %zu words\n",words);
}
