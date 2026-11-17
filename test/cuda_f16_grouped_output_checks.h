typedef struct {int k,down;} F16GroupedSource;
static int f16_grouped_read(void*ctx,int row,int count,uint16_t*out,size_t cap) {
    F16GroupedSource*s=ctx;assert(cap>=(size_t)count*s->k);
    for(int r=0;r<count;r++)for(int k=0;k<s->k;k++) {
        int v=s->down?((row+r)*5+k*7)%11-5:((row+r)*7+k*3)%9-4;
        out[(size_t)r*s->k+k]=bn_fp32_to_fp16((float)v/16);
    }
    return 0;
}
static void run_f16_grouped_output_reference(BnGPUBackend*g) {
    assert(g->grouped_output_f16_resident);const int shapes[][3]={{1,1,1},{3,35,17},{8,128,128}},rows[]={1,7,33,273};size_t calls=0,words=0;
    for(int si=0;si<3;si++) {
        int G=shapes[si][0],K=shapes[si][1],N=shapes[si][2],D=K+3,W=G*N;void*w[2];F16GroupedSource source[2]={{K,0},{W,1}};
        for(int j=0;j<2;j++){int n=j?D:W,k=j?W:K;size_t tile=(size_t)(n<32?n:32)*k*2;BnGPUF16RowsPlan upload={n,k,(size_t)n*k*2+tile,tile,source+j,f16_grouped_read};w[j]=g->buffer_create_f16_rows(g->ctx,&upload);assert(w[j]);}
        for(int ri=0;ri<4;ri++)for(int oi=0;oi<2;oi++) {
            int R=rows[ri],type=oi?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_F16;BnGPUF16GroupedOutputWorkspace ws;assert(!bn_gpu_f16_grouped_output_workspace(G,K,N,R,&ws));size_t counts[]={ws.input_count,ws.concat_count,(size_t)R*D};int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,type};
            void*r[2];BnGPUResidentView roots[2][3];BnGPUF16GroupedOutputResidentPlan p[2];uint16_t*x[2],*cat[2];void*out[2];
            for(int q=0;q<2;q++) {
                x[q]=malloc(counts[0]*2);cat[q]=malloc(counts[1]*2);out[q]=malloc(counts[2]*(oi?4:2));assert(x[q]&&cat[q]&&out[q]);
                for(int h=0;h<G;h++)for(int row=0;row<R;row++)for(int k=0;k<K;k++)x[q][((size_t)h*R+row)*K+k]=bn_fp32_to_fp16((float)(((h*R+row)*5+k*11+q)%7-3)/8);
                for(int h=0;h<G;h++)for(int row=0;row<R;row++)for(int n=0;n<N;n++){int sum=0;for(int k=0;k<K;k++)sum+=(((h*N+n)*7+k*3)%9-4)*(((h*R+row)*5+k*11+q)%7-3);uint16_t v=bn_fp32_to_fp16((float)sum/128);cat[q][(size_t)row*W+h*N+n]=v;}
                for(int row=0;row<R;row++)for(int d=0;d<D;d++){float sum=0;for(int k=0;k<W;k++)sum+=bn_fp16_to_fp32(cat[q][(size_t)row*W+k])*(float)((d*5+k*7)%11-5)/16;if(oi)((float*)out[q])[(size_t)row*D+d]=sum;else((uint16_t*)out[q])[(size_t)row*D+d]=bn_fp32_to_fp16(sum);}
                r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[3];for(int j=0;j<3;j++)v[j]=native_router_guard(g,r[q],types[j],counts[j],j==0?x[q]:NULL,&roots[q][j]);
                p[q]=(BnGPUF16GroupedOutputResidentPlan){w[0],G,K,N,R,v[0],{w[1],W,D,R,v[1],v[2]}};
                assert(g->grouped_output_f16_resident(g->ctx,r[q],p+q)==-1);assert(!g->linear_f16_reserve(g->ctx,r[q]));BnGPUF16GroupedOutputResidentPlan bad;
#define GROUP_BAD(F,V) do{bad=p[q];bad.F=(V);int status=g->grouped_output_f16_resident(g->ctx,r[q],&bad);if(status!=-1)fprintf(stderr,"Grouped rejection %s failed G%d K%d N%d R%d\n",#F,G,K,N,R);assert(status==-1);}while(0)
                GROUP_BAD(groups,65);GROUP_BAD(k,0);GROUP_BAD(n,N+1);GROUP_BAD(n_tokens,0);GROUP_BAD(group_weight,NULL);GROUP_BAD(group_weight,w[1]);GROUP_BAD(x.type,BN_GGUF_TENSOR_F32);GROUP_BAD(x.count,counts[0]-1);GROUP_BAD(down.x.count,counts[1]+1);GROUP_BAD(down.x,p[q].x);GROUP_BAD(down.k,W+1);GROUP_BAD(down.n_tokens,R+1);GROUP_BAD(down.out.count,counts[2]-1);if(counts[2]<=counts[1]&&!oi){GROUP_BAD(down.out,bn_gpu_resident_view_slice(roots[q][1],16,counts[2]));}
#undef GROUP_BAD
                for(int j=1;j<3;j++)native_router_guard_check(g,r[q],roots[q][j],counts[j],types[j],folded_test_unwritten(counts[j]*(types[j]==BN_GGUF_TENSOR_F32?4:2)));
            }
            BnGPUF16GroupedOutputResidentPlan bad=p[0];bad.x=p[1].x;assert(g->grouped_output_f16_resident(g->ctx,r[0],&bad)==-1);bad=p[0];bad.down.x=p[1].down.x;assert(g->grouped_output_f16_resident(g->ctx,r[0],&bad)==-1);
            for(int rep=0;rep<2;rep++)for(int q=0;q<2;q++) {
                assert(!g->grouped_output_f16_resident(g->ctx,r[q],p+q));const void*expected[]={x[q],cat[q],out[q]};for(int j=0;j<3;j++)native_router_guard_check(g,r[q],roots[q][j],counts[j],types[j],native_norm_hash(expected[j],counts[j]*(types[j]==BN_GGUF_TENSOR_F32?4:2)));calls++;words+=counts[2];

            }
            for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);free(x[q]);free(cat[q]);free(out[q]);}
        }
        for(int j=0;j<2;j++){g->buffer_destroy(g->ctx,w[j]);}
    }
    printf("CUDA F16 grouped output exact dyadic/views/guards PASSED: %zu calls, %zu output words\n",calls,words);
}
