/* Primary numerical golden: unchanged original Diffbot Triton lookup,
 * all finite nonnegative FP16 scales, every nibble, captured before native.
 * See docs/deepseek-v41-1017-native-engram-rows.json. */
static size_t scaled_rows_width(int type) {
    assert(type==BN_GGUF_TENSOR_I32||type==BN_GGUF_TENSOR_F16||type==BN_GGUF_TENSOR_BF16);
    return type==BN_GGUF_TENSOR_I32?4:2;
}
static BnGPUResidentView scaled_rows_guard(BnGPUBackend *g,void *r,int type,size_t n,
    const void *input,BnGPUResidentView *root) {
    size_t width=scaled_rows_width(type);unsigned char *host=malloc((n+34)*width);assert(host);
    memset(host,0xa5,(n+34)*width);if(input)memcpy(host+17*width,input,n*width);
    *root=cuda_test_resident_alloc(g,r,type,n+34,host);free(host);
    return bn_gpu_resident_view_slice(*root,17,n);
}
static void scaled_rows_check(BnGPUBackend *g,void *r,BnGPUResidentView root,size_t n,uint64_t expected) {
    size_t width=scaled_rows_width(root.type);unsigned char *host=malloc((n+34)*width);assert(host);
    assert(!g->resident_read(r,root,host));
    assert(native_norm_hash(host+17*width,n*width)==expected);
    for(size_t i=0;i<17*width;i++)assert(host[i]==0xa5&&host[(n+17)*width+i]==0xa5);
    free(host);
}
static void run_scaled_rows_reference(BnGPUBackend *g) {
    const int cases[][2]={{31744,32},{1,32},{2,64},{17,128},{129,256},{1,512},{3,1024},{2,2048},{1,65536}};
    size_t calls=0,words=0;
    for(size_t c=0;c<sizeof(cases)/sizeof(*cases);c++) {
        int rows=cases[c][0],cols=cases[c][1];BnGPUScaledRowsWorkspace ws;
        assert(!bn_gpu_scaled_rows_shape(rows,cols,&ws));
        size_t wb=ws.packed_i32*4,sb=ws.scales_f16*2,ob=ws.out_bf16*2;
        unsigned char *w=malloc(wb);uint16_t *s=malloc(sb);assert(w&&s);
        memset(w,0x88,wb);for(size_t i=0;i<ws.scales_f16;i++)s[i]=0x3c00;
        uint64_t golden;
        if(!c) {
            for(int i=0;i<rows;i++){s[i]=(uint16_t)i;for(int j=0;j<16;j++)w[(size_t)i*16+j]=(unsigned char)(0x10+(j%8)*0x22);}
            golden=UINT64_C(0xba104b93f11abf25);
        } else {
            golden=UINT64_C(14695981039346656037);
            for(size_t i=0;i<ob;i++)golden*=UINT64_C(1099511628211);
        }
        void *r[2];BnGPUResidentView roots[2][3];BnGPUQuantRowsBF16ResidentPlan p[2];
        for(int q=0;q<2;q++) {
            r[q]=g->resident_request_create(g->ctx);assert(r[q]);memset(&p[q],0,sizeof(p[q]));
            p[q].rows=rows;p[q].cols=cols;p[q].type=BN_QUANT_FORMAT_INT4_FP16S32;p[q].row_stride_bytes=(size_t)cols/2;
            p[q].packed=scaled_rows_guard(g,r[q],BN_GGUF_TENSOR_I32,ws.packed_i32,w,&roots[q][0]);
            p[q].scales=scaled_rows_guard(g,r[q],BN_GGUF_TENSOR_F16,ws.scales_f16,s,&roots[q][1]);
            p[q].out=scaled_rows_guard(g,r[q],BN_GGUF_TENSOR_BF16,ws.out_bf16,NULL,&roots[q][2]);
            BnGPUQuantRowsBF16ResidentPlan bad;
#define SCALED_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->quant_rows_bf16_resident(g->ctx,r[q],&bad)==-1);}while(0)
            SCALED_BAD(rows,0);SCALED_BAD(cols,cols+1);SCALED_BAD(type,BN_GGUF_TENSOR_Q4_0);SCALED_BAD(row_stride_bytes,(size_t)cols/2+16);
            SCALED_BAD(packed.count,ws.packed_i32-1);SCALED_BAD(scales.count,ws.scales_f16+1);SCALED_BAD(out.count,ws.out_bf16-1);
            SCALED_BAD(packed.type,BN_GGUF_TENSOR_F32);SCALED_BAD(scales.type,BN_GGUF_TENSOR_BF16);SCALED_BAD(out.type,BN_GGUF_TENSOR_F16);
            SCALED_BAD(packed.slot,p[q].packed.slot^UINT32_C(0x10000));SCALED_BAD(scales.offset,p[q].scales.offset+35);
            SCALED_BAD(scales,p[q].out);SCALED_BAD(out,p[q].packed);
#undef SCALED_BAD
            assert(g->quant_rows_bf16_resident(NULL,r[q],p+q)==-1);
            assert(g->quant_rows_bf16_resident(g->ctx,NULL,p+q)==-1);
            assert(g->quant_rows_bf16_resident(g->ctx,r[q],NULL)==-1);
            scaled_rows_check(g,r[q],roots[q][2],ws.out_bf16,folded_test_unwritten(ob));
        }
        BnGPUQuantRowsBF16ResidentPlan bad=p[0];bad.scales=p[1].scales;
        assert(g->quant_rows_bf16_resident(g->ctx,r[0],&bad)==-1);
        bad=p[0];bad.packed=p[1].packed;assert(g->quant_rows_bf16_resident(g->ctx,r[0],&bad)==-1);
        bad=p[0];bad.out=p[1].out;assert(g->quant_rows_bf16_resident(g->ctx,r[0],&bad)==-1);
        for(int rep=0;rep<2;rep++)for(int q=0;q<2;q++) {
            unsigned char *poison=malloc((ws.out_bf16+34)*2);assert(poison);memset(poison,0xa5,(ws.out_bf16+34)*2);
            assert(!g->resident_write(r[q],roots[q][2],poison));free(poison);
            assert(!g->quant_rows_bf16_resident(g->ctx,r[q],p+q));
            scaled_rows_check(g,r[q],roots[q][2],ws.out_bf16,golden);calls++;words+=ws.out_bf16;
        }
        for(int q=0;q<2;q++) {
            scaled_rows_check(g,r[q],roots[q][0],ws.packed_i32,native_norm_hash(w,wb));
            scaled_rows_check(g,r[q],roots[q][1],ws.scales_f16,native_norm_hash(s,sb));
            assert(!g->resident_resize_discard(r[q],&roots[q][0],ws.packed_i32+35));
            assert(g->quant_rows_bf16_resident(g->ctx,r[q],p+q)==-1);
            scaled_rows_check(g,r[q],roots[q][2],ws.out_bf16,golden);g->resident_request_destroy(r[q]);
        }
        free(s);free(w);
    }
    printf("CUDA separate-scale row decode source golden/guards/ownership PASSED: %zu calls, %zu words\n",calls,words);
}
