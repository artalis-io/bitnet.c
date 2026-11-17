#ifndef BN_GPU_CUDA_MHC_CUH
#define BN_GPU_CUDA_MHC_CUH

namespace bn_cuda_mhc {
using bn_cuda_gdn::ResidentRequest;
using bn_cuda_gdn::view_data;
using bn_cuda_gdn::overlap;
using bn_cuda_gdn::set_device;

/* SM120 split-K TF32 projection and fixed four-stream mHC arithmetic. */
static __global__ void mhc_project_kernel(float *dot,float *sq,
    const __nv_bfloat16 *x,const float *weight,int nt,int dim,int splits) {
    int lane=threadIdx.x,group=lane/4,col=lane%4;
    int r=blockIdx.x*16+group,split=blockIdx.y,kblocks=dim*4/64;
    int blocks=kblocks/splits+(split<kblocks%splits);
    int start=(split*(kblocks/splits)+min(split,kblocks%splits))*64;
    float d[3][4]={{0}},s0=0,s1=0;
    for(int k=start;k<start+blocks*64;k+=8) {
        int ki=k+col;float a0=r<nt?__bfloat162float(x[(size_t)r*dim*4+ki]):0;
        float a1=r+8<nt?__bfloat162float(x[(size_t)(r+8)*dim*4+ki]):0;
        float a2=r<nt?__bfloat162float(x[(size_t)r*dim*4+ki+4]):0;
        float a3=r+8<nt?__bfloat162float(x[(size_t)(r+8)*dim*4+ki+4]):0;
        s0=__fadd_rn(s0,fmaf(a0,a0,__fmul_rn(a2,a2)));
        s1=__fadd_rn(s1,fmaf(a1,a1,__fmul_rn(a3,a3)));
        #pragma unroll
        for(int tile=0;tile<3;++tile) {
            unsigned b0,b1;float f0=weight[(size_t)(tile*8+group)*dim*4+ki],f1=weight[(size_t)(tile*8+group)*dim*4+ki+4];
            asm("cvt.rn.tf32.f32 %0, %1;":"=r"(b0):"f"(f0));
            asm("cvt.rn.tf32.f32 %0, %1;":"=r"(b1):"f"(f1));
            unsigned u0=__float_as_uint(a0),u1=__float_as_uint(a1),u2=__float_as_uint(a2),u3=__float_as_uint(a3);
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                :"+f"(d[tile][0]),"+f"(d[tile][1]),"+f"(d[tile][2]),"+f"(d[tile][3])
                :"r"(u0),"r"(u1),"r"(u2),"r"(u3),"r"(b0),"r"(b1));
        }
    }
    for(int delta=2;delta;delta/=2) {
        s0=__fadd_rn(s0,__shfl_xor_sync(0xffffffff,s0,delta));
        s1=__fadd_rn(s1,__shfl_xor_sync(0xffffffff,s1,delta));
    }
    if(col==0) {if(r<nt)sq[(size_t)split*nt+r]=s0;if(r+8<nt)sq[(size_t)split*nt+r+8]=s1;}
    #pragma unroll
    for(int tile=0;tile<3;++tile) {
        int c=tile*8+col*2;
        if(r<nt) {dot[((size_t)split*nt+r)*24+c]=d[tile][0];dot[((size_t)split*nt+r)*24+c+1]=d[tile][1];}
        if(r+8<nt) {dot[((size_t)split*nt+r+8)*24+c]=d[tile][2];dot[((size_t)split*nt+r+8)*24+c+1]=d[tile][3];}
    }
}
static __global__ void mhc_coeff_kernel(float *pre,float *post,float *comb,
    const float *dot,const float *sq,const float *scale,const float *base,
    int nt,int dim,int splits,int iterations,float rms_eps,float pre_eps,float sink_eps,float mult) {
    int t=blockIdx.x,lane=threadIdx.x;float sum=0,mix=0;
    for(int s=0;s<splits;++s) {sum=__fadd_rn(sum,sq[(size_t)s*nt+t]);mix=__fadd_rn(mix,dot[((size_t)s*nt+t)*24+lane%24]);}
    float inv=rsqrtf(__fadd_rn(sum/(dim*4),rms_eps));mix=__fmul_rn(mix,inv);
    float score=fmaf(mix,scale[0],base[lane%24]);
    if(lane<4)pre[(size_t)t*4+lane]=__fadd_rn(1.0f/(1.0f+expf(-score)),pre_eps);
    if(lane>=4&&lane<8) {score=fmaf(mix,scale[1],base[lane]);post[(size_t)t*4+lane-4]=__fmul_rn(1.0f/(1.0f+expf(-score)),mult);}
    float logit=fmaf(__shfl_sync(0xffffffff,mix,8+(lane&15)),scale[2],base[8+(lane&15)]);
    float maximum=fmaxf(logit,__shfl_xor_sync(0xffffffff,logit,2));maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1));
    float value=expf(logit-maximum),denom=value;
    for(int d=2;d;d/=2)denom=__fadd_rn(denom,__shfl_xor_sync(0xffffffff,denom,d));
    value=__fadd_rn(value/denom,sink_eps);denom=value;
    for(int d=8;d>=4;d/=2)denom=__fadd_rn(denom,__shfl_xor_sync(0xffffffff,denom,d));
    value=value/__fadd_rn(denom,sink_eps);
    if(inv>=0)for(int i=1;i<iterations;++i) {
        denom=value;for(int d=2;d;d/=2)denom=__fadd_rn(denom,__shfl_xor_sync(0xffffffff,denom,d));value=value/__fadd_rn(denom,sink_eps);
        denom=value;for(int d=8;d>=4;d/=2)denom=__fadd_rn(denom,__shfl_xor_sync(0xffffffff,denom,d));value=value/__fadd_rn(denom,sink_eps);
    }
    if(lane<16)comb[(size_t)t*16+lane]=value;
}
static __global__ void mhc_collapse_kernel(__nv_bfloat16 *raw,__nv_bfloat16 *out,
    const __nv_bfloat16 *res,const float *pre,const __nv_bfloat16 *norm,int dim,float eps) {
    int tid=threadIdx.x,t=blockIdx.x;float sums[16]={0};__shared__ float partial[64];
    for(int b=0;b<dim;b+=1024) {
        #pragma unroll
        for(int j=0;j<16;++j) {
            int k=b+tid*8+(j/8)*512+j%8;float a=0;
            #pragma unroll
            for(int h=0;h<4;++h)a=fmaf(pre[(size_t)t*4+h],__bfloat162float(res[((size_t)t*4+h)*dim+k]),a);
            sums[j]=fmaf(a,a,sums[j]);raw[(size_t)t*dim+k]=__float2bfloat16_rn(a);
        }
    }
    float sum=0;for(int j=0;j<16;++j)sum=__fadd_rn(sum,sums[(j&1)*8+j/2]);
    partial[tid]=sum;__syncthreads();sum=__fadd_rn(sum,partial[tid^32]);
    for(int d=16;d;d/=2)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,d));
    float inv=rsqrtf(__fadd_rn(sum/dim,eps));
    for(int k=tid;k<dim;k+=64) {
        float a=__bfloat162float(raw[(size_t)t*dim+k]);
        out[(size_t)t*dim+k]=norm?__float2bfloat16_rn(__fmul_rn(__fmul_rn(a,inv),__bfloat162float(norm[k]))):__float2bfloat16_rn(a);
    }
}
static __global__ void mhc_post_kernel(__nv_bfloat16 *out,const __nv_bfloat16 *branch,
    const __nv_bfloat16 *res,const float *post,const float *comb,int dim) {
    int t=blockIdx.x;for(int k=threadIdx.x;k<dim*4;k+=blockDim.x) {
        int h=k/dim,c=k%dim;float a=fmaf(post[(size_t)t*4+h],__bfloat162float(branch[(size_t)t*dim+c]),__fmul_rn(comb[(size_t)t*16+h],__bfloat162float(res[(size_t)t*4*dim+c])));
        #pragma unroll
        for(int i=1;i<4;++i)a=fmaf(comb[(size_t)t*16+i*4+h],__bfloat162float(res[((size_t)t*4+i)*dim+c]),a);
        out[(size_t)t*dim*4+k]=__float2bfloat16_rn(a);
    }
}

template<int Tile,int Threads>
static __global__ void mhc_fused_project_kernel(float *dot,float *sq,__nv_bfloat16 *out,
    const __nv_bfloat16 *res,const __nv_bfloat16 *branch,const float *post,
    const float *comb,const float *weight,int nt,int dim,int splits) {
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,t=blockIdx.x,tile=blockIdx.y,s=blockIdx.z;
    int per=dim/splits;float a[Tile]={0},square=0;
    for(int k=s*per+tid;k<(s+1)*per;k+=Threads) {
        float x=__bfloat162float(branch[(size_t)t*dim+k]),r[4];
        #pragma unroll
        for(int j=0;j<4;++j) {
            r[j]=fmaf(post[(size_t)t*4+j],x,__fmul_rn(comb[(size_t)t*16+j],__bfloat162float(res[(size_t)t*4*dim+k])));
            #pragma unroll
            for(int i=1;i<4;++i)r[j]=fmaf(comb[(size_t)t*16+i*4+j],__bfloat162float(res[((size_t)t*4+i)*dim+k]),r[j]);
            if(tile==0){out[((size_t)t*4+j)*dim+k]=__float2bfloat16_rn(r[j]);square=fmaf(r[j],r[j],square);}
        }
        #pragma unroll
        for(int n=0;n<Tile;++n) {
            #pragma unroll
            for(int j=0;j<4;++j)a[n]=fmaf(weight[((size_t)(tile*Tile+n)*4+j)*dim+k],r[j],a[n]);
        }
    }
    for(int d=16;d;d/=2) {
        #pragma unroll
        for(int n=0;n<Tile;++n)a[n]=__fadd_rn(a[n],__shfl_xor_sync(0xffffffff,a[n],d));
        if(tile==0)square=__fadd_rn(square,__shfl_xor_sync(0xffffffff,square,d));
    }
    __shared__ float partial[Threads/32][Tile+1];
    if(lane==0) {
        #pragma unroll
        for(int n=0;n<Tile;++n)partial[warp][n]=a[n];
        if(tile==0)partial[warp][Tile]=square;
    }
    __syncthreads();
    if(warp==0) {
        if(lane<Tile){float sum=0;for(int w=0;w<Threads/32;++w)sum=__fadd_rn(sum,partial[w][lane]);dot[((size_t)s*nt+t)*24+tile*Tile+lane]=sum;}
        if(lane==0&&tile==0){float sum=0;for(int w=0;w<Threads/32;++w)sum=__fadd_rn(sum,partial[w][Tile]);sq[(size_t)s*nt+t]=sum;}
    }
}

static int geometry(int nt,int dim,int streams) {
    return nt>0 && dim>0 && dim%1024==0 && dim<=INT_MAX/4 &&
        nt<=INT_MAX/(dim*4) && streams==4;
}
static int split_count(void *ptr,int nt,int dim) {
    BnCudaCtx *ctx=(BnCudaCtx*)ptr;int sms=0;
    if(!ctx || ctx->compute_capability!=1200 || !geometry(nt,dim,4) ||
       cuda_ctx_set_device(ctx) || cudaDeviceGetAttribute(&sms,
           cudaDevAttrMultiProcessorCount,ctx->device)!=cudaSuccess) return -1;
    int count=sms/((nt+63)/64),limit=(dim*4/64)/4;
    if(count>limit)count=limit;
    return count>0?count:1;
}
static const BnCudaBuffer *weight(void *ptr,int rows,int cols,int type) {
    const BnCudaBuffer *w=(const BnCudaBuffer*)ptr;
    size_t size=(size_t)rows*cols*(type==BN_GGUF_TENSOR_F32?4:2);
    return w&&w->data&&w->rows==rows&&w->cols==cols&&w->type==type&&w->size==size?w:NULL;
}
static int views(ResidentRequest*r,const BnGPUResidentView *v,
    const size_t *counts,const int *types,void **data,int n,int writes_from) {
    for(int i=0;i<n;++i) {
        if(v[i].request!=r || v[i].count!=counts[i] || v[i].type!=types[i] ||
            !(data[i]=view_data(r,v[i])))return -1;
        for(int j=0;j<i;++j)
            if((i>=writes_from||j>=writes_from)&&overlap(v[i],v[j]))return -1;
    }
    return 0;
}
static int validate_pre(void *ctx_ptr,void *ptr,const BnGPUMHCPreResidentPlan *p,
    void **data,const BnCudaBuffer **weights) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||
        !geometry(p->n_tokens,p->dim,p->streams)||p->n_splits<1||
        p->n_splits>p->dim*4/64||p->n_splits>65535||p->iterations<1||p->iterations>128||
        !isfinite(p->rms_eps)||p->rms_eps<=0||!isfinite(p->pre_eps)||p->pre_eps<0||
        !isfinite(p->sinkhorn_eps)||p->sinkhorn_eps<=0||!isfinite(p->post_mult)||p->post_mult<0||
        (p->norm&&(!isfinite(p->norm_eps)||p->norm_eps<=0))||set_device(r))return -1;
    const BnCudaBuffer *fn=weight(p->fn,24,p->dim*4,BN_GGUF_TENSOR_F32),
        *scale=weight(p->scale,1,3,BN_GGUF_TENSOR_F32),
        *base=weight(p->base,1,24,BN_GGUF_TENSOR_F32),
        *norm=p->norm?weight(p->norm,1,p->dim,BN_GGUF_TENSOR_BF16):NULL;
    if(!fn||!scale||!base||(p->norm&&!norm))return -1;
    size_t n=p->n_tokens,ns=n*p->n_splits;
    if(ns>SIZE_MAX/24/sizeof(float))return -1;
    BnGPUResidentView v[]={p->residual,p->dot,p->sqrsum,p->pre,p->post,p->comb,p->collapsed,p->out};
    size_t counts[]={n*p->dim*4,ns*24,ns,n*4,n*4,n*16,n*p->dim,n*p->dim};
    int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,
        BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16};
    if(views(r,v,counts,types,data,8,1))return -1;
    weights[0]=fn;weights[1]=scale;weights[2]=base;weights[3]=norm;return 0;
}
static int pre_tail(ResidentRequest*r,const BnGPUMHCPreResidentPlan*p,void **data,const BnCudaBuffer **weights) {
    const BnCudaBuffer *scale=weights[1],*base=weights[2],*norm=weights[3];
    mhc_coeff_kernel<<<p->n_tokens,32,0,r->stream>>>((float*)data[3],(float*)data[4],(float*)data[5],
        (const float*)data[1],(const float*)data[2],(const float*)scale->data,(const float*)base->data,
        p->n_tokens,p->dim,p->n_splits,p->iterations,p->rms_eps,p->pre_eps,p->sinkhorn_eps,p->post_mult);
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    mhc_collapse_kernel<<<p->n_tokens,64,0,r->stream>>>((__nv_bfloat16*)data[6],(__nv_bfloat16*)data[7],
        (const __nv_bfloat16*)data[0],(const float*)data[3],norm?(const __nv_bfloat16*)norm->data:NULL,p->dim,p->norm_eps);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}
static int pre(void *ctx_ptr,void *ptr,const BnGPUMHCPreResidentPlan *p) {
    ResidentRequest*r=(ResidentRequest*)ptr;void *data[8];const BnCudaBuffer *weights[4];
    if(validate_pre(ctx_ptr,ptr,p,data,weights))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    mhc_project_kernel<<<dim3((p->n_tokens+15)/16,p->n_splits),32,0,r->stream>>>(
        (float*)data[1],(float*)data[2],(const __nv_bfloat16*)data[0],
        (const float*)weights[0]->data,p->n_tokens,p->dim,p->n_splits);
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    return pre_tail(r,p,data,weights);
}
static __device__ __forceinline__ float stream_mean_add(float a,float b){float out;asm("add.rn.f32 %0,%1,%2;":"=f"(out):"f"(a),"f"(b));return out;}
static __global__ void stream_mean_kernel(__nv_bfloat16 *out,const __nv_bfloat16 *x,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    size_t base=(size_t)(i/4096)*16384+i%4096;float sum=0;
    for(int s=0;s<4;++s)sum=stream_mean_add(sum,__bfloat162float(x[base+s*4096]));
    float mean;asm("mul.rn.f32 %0,%1,0f3e800000;":"=f"(mean):"f"(sum));out[i]=__float2bfloat16_rn(mean);
}
static int stream_mean(void *ctx_ptr,void *ptr,const BnGPUStreamMeanResidentPlan *p){
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||p->dim!=4096||p->streams!=4||
       !geometry(p->n_tokens,p->dim,p->streams)||set_device(r))return -1;
    size_t n=(size_t)p->n_tokens*p->dim;BnGPUResidentView v[]={p->x,p->out};size_t counts[]={n*4,n};
    int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16};void *data[2];if(views(r,v,counts,types,data,2,1))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    stream_mean_kernel<<<(unsigned)((n+255)/256),256,0,r->stream>>>((__nv_bfloat16*)data[1],(const __nv_bfloat16*)data[0],(int)n);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

static int post(void *ctx_ptr,void *ptr,const BnGPUMHCPostResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest*r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||
        !geometry(p->n_tokens,p->dim,p->streams)||set_device(r))return -1;
    size_t n=p->n_tokens;
    BnGPUResidentView v[]={p->residual,p->branch,p->post,p->comb,p->out};
    size_t counts[]={n*p->dim*4,n*p->dim,n*4,n*16,n*p->dim*4};
    int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_BF16};
    void *data[5];if(views(r,v,counts,types,data,5,4))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    mhc_post_kernel<<<p->n_tokens,128,0,r->stream>>>((__nv_bfloat16*)data[4],
        (const __nv_bfloat16*)data[1],(const __nv_bfloat16*)data[0],
        (const float*)data[2],(const float*)data[3],p->dim);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}
static int fused_split_count(void *ctx_ptr,int nt,int dim) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;
    if(!ctx||ctx->compute_capability!=1200||dim!=4096||!geometry(nt,dim,4))return -1;
    return nt<=4?16:(nt<=32?8:split_count(ctx_ptr,nt,dim));
}
static int fused(void *ctx_ptr,void *ptr,const BnGPUMHCFusedResidentPlan *p) {
    if(!p)return -1;
    const BnGPUMHCPreResidentPlan *q=&p->next;ResidentRequest*r=(ResidentRequest*)ptr;
    void *data[8];const BnCudaBuffer *weights[4];
    if(q->dim!=4096||(q->n_tokens<=32&&q->n_splits!=(q->n_tokens<=4?16:8))||
        validate_pre(ctx_ptr,ptr,q,data,weights))return -1;
    size_t n=q->n_tokens,ns=n*q->n_splits;
    BnGPUResidentView v[]={p->previous_residual,p->previous_branch,p->previous_post,p->previous_comb,
        q->residual,q->dot,q->sqrsum,q->pre,q->post,q->comb,q->collapsed,q->out};
    size_t counts[]={n*q->dim*4,n*q->dim,n*4,n*16,n*q->dim*4,ns*24,ns,n*4,n*4,n*16,n*q->dim,n*q->dim};
    int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,
        BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16};
    void *all[12];if(views(r,v,counts,types,all,12,4))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    if(q->n_tokens>32) {
        mhc_post_kernel<<<q->n_tokens,128,0,r->stream>>>((__nv_bfloat16*)data[0],
            (const __nv_bfloat16*)all[1],(const __nv_bfloat16*)all[0],(const float*)all[2],(const float*)all[3],q->dim);
        if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
        return pre(ctx_ptr,ptr,q);
    }
    dim3 grid(q->n_tokens,q->n_tokens<16?12:4,q->n_splits);
#define BN_MHC_FUSED_ARGS (float*)data[1],(float*)data[2],(__nv_bfloat16*)data[0], \
    (const __nv_bfloat16*)all[0],(const __nv_bfloat16*)all[1],(const float*)all[2], \
    (const float*)all[3],(const float*)weights[0]->data,q->n_tokens,q->dim,q->n_splits
    if(q->n_tokens<=4)mhc_fused_project_kernel<2,256><<<grid,256,0,r->stream>>>(BN_MHC_FUSED_ARGS);
    else if(q->n_tokens<16)mhc_fused_project_kernel<2,128><<<grid,128,0,r->stream>>>(BN_MHC_FUSED_ARGS);
    else mhc_fused_project_kernel<6,128><<<grid,128,0,r->stream>>>(BN_MHC_FUSED_ARGS);
#undef BN_MHC_FUSED_ARGS
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    return pre_tail(r,q,data,weights);
}
} /* namespace bn_cuda_mhc */
#endif
