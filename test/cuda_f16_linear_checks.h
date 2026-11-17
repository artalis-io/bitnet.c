/* Dyadic fixtures: every dot product is an exactly representable integer/128.
 * Expected values are independent scalar integer sums, including odd shapes. */
typedef struct { int k; } F16LinearTestSource;
static int f16_linear_test_read(void *context,int row,int count,uint16_t *out,size_t capacity) {
    F16LinearTestSource *s=context;assert(capacity>=(size_t)count*s->k);
    for(int r=0;r<count;r++)for(int c=0;c<s->k;c++)
        out[(size_t)r*s->k+c]=bn_fp32_to_fp16((float)(((row+r)*7+c*3)%9-4)/16);
    return 0;
}
static void run_f16_linear_reference(BnGPUBackend *gpu) {
    assert(gpu->linear_f16_resident&&gpu->linear_f16_reserve&&gpu->linear_f16_workspace_size);
    assert(gpu->linear_f16_workspace_size(gpu->ctx)==48u*1024u*1024u);
    assert(!gpu->linear_f16_workspace_size(NULL));
    const int shapes[][2]={{1,1},{35,67},{128,128}},tokens[]={1,7,64,256,1024};size_t words=0;
    for(int a=0;a<3;a++) {
        int n=shapes[a][0],k=shapes[a][1];F16LinearTestSource source={k};
        size_t tile=(size_t)(n<32?n:32)*k*2;
        BnGPUF16RowsPlan upload={n,k,(size_t)n*k*2+tile,tile,&source,f16_linear_test_read};
        void *weight=gpu->buffer_create_f16_rows(gpu->ctx,&upload);assert(weight);
        for(int b=0;b<5;b++)for(int type_id=0;type_id<2;type_id++) {
            int m=tokens[b],type=type_id?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_F16;
            size_t nx=(size_t)m*k,ny=(size_t)m*n,width=type_id?4:2;
            uint16_t *x=malloc(nx*2);unsigned char *host=malloc((ny+16)*width);assert(x&&host);
            void *requests[2];BnGPUF16LinearResidentPlan plans[2];BnGPUResidentView storage[2];
            for(int q=0;q<2;q++) {
                for(int t=0;t<m;t++)for(int c=0;c<k;c++)x[(size_t)t*k+c]=bn_fp32_to_fp16((float)((t*5+c*11+q)%7-3)/8);
                memset(host,0x5a,(ny+16)*width);requests[q]=gpu->resident_request_create(gpu->ctx);assert(requests[q]);
                plans[q]=(BnGPUF16LinearResidentPlan){weight,k,n,m,
                    cuda_test_resident_alloc(gpu,requests[q],BN_GGUF_TENSOR_F16,nx,x),{0}};
                storage[q]=cuda_test_resident_alloc(gpu,requests[q],type,ny+16,host);
                plans[q].out=bn_gpu_resident_view_slice(storage[q],8,ny);
                assert(gpu->linear_f16_resident(gpu->ctx,requests[q],&plans[q])==-1);
                assert(!gpu->linear_f16_reserve(gpu->ctx,requests[q]));assert(!gpu->linear_f16_reserve(gpu->ctx,requests[q]));
                BnGPUF16LinearResidentPlan bad;
#define F16_LINEAR_BAD(field,value) do {bad=plans[q];bad.field=(value);assert(gpu->linear_f16_resident(gpu->ctx,requests[q],&bad)==-1);} while(0)
                F16_LINEAR_BAD(k,0);F16_LINEAR_BAD(n,n+1);F16_LINEAR_BAD(n_tokens,INT_MAX);
                F16_LINEAR_BAD(weight,NULL);F16_LINEAR_BAD(x.type,BN_GGUF_TENSOR_F32);
                F16_LINEAR_BAD(x.count,nx+1);F16_LINEAR_BAD(out.type,BN_GGUF_TENSOR_I8);
                F16_LINEAR_BAD(out.count,ny+1);F16_LINEAR_BAD(out,plans[q].x);
                int dummy;assert(gpu->linear_f16_resident(&dummy,requests[q],&plans[q])==-1);
#undef F16_LINEAR_BAD
                assert(!gpu->resident_read(requests[q],storage[q],host));
                for(size_t i=0;i<(ny+16)*width;i++)assert(host[i]==0x5a);
            }
            BnGPUF16LinearResidentPlan bad=plans[0];bad.x=plans[1].x;
            assert(gpu->linear_f16_resident(gpu->ctx,requests[0],&bad)==-1);
            for(int q=0;q<2;q++)assert(!gpu->linear_f16_resident(gpu->ctx,requests[q],&plans[q]));
            for(int q=0;q<2;q++) {
                assert(!gpu->resident_read(requests[q],storage[q],host));
                for(size_t i=0;i<8*width;i++)assert(host[i]==0x5a&&host[(ny+8)*width+i]==0x5a);
                for(int t=0;t<m;t++)for(int r=0;r<n;r++) {
                    int sum=0;for(int c=0;c<k;c++)sum+=((r*7+c*3)%9-4)*((t*5+c*11+q)%7-3);
                    size_t index=8+(size_t)t*n+r;float expected=(float)sum/128;
                    if(type_id)assert(((float*)host)[index]==expected);
                    else assert(((uint16_t*)host)[index]==bn_fp32_to_fp16(expected));
                }
                words+=ny;gpu->resident_request_destroy(requests[q]);
            }
            free(x);free(host);
        }
        gpu->buffer_destroy(gpu->ctx,weight);
    }
    printf("CUDA resident F16 linear exact dyadic/guards/two-request isolation PASSED: %zu words\n",words);
}
