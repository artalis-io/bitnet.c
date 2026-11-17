#ifndef BN_GPU_CUDA_SWA_RING_CUH
#define BN_GPU_CUDA_SWA_RING_CUH
namespace bn_cuda_swa_ring {
static __global__ void snapshot(const uint16_t*ring,uint16_t*tmp,size_t n,size_t start) {
    size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t<n)tmp[t]=ring[start+t];
}
static __global__ void update(const uint16_t*kv,uint16_t*ring,const uint16_t*tmp,
    int B,int D,int old_rows,int chunk_src,int chunk_rows,int chunk_dst) {
    size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(size_t)B*D)return;
    int row=(int)(t/D),col=(int)(t%D);
    if(row>=chunk_dst&&row<chunk_dst+chunk_rows)ring[t]=kv[(size_t)(chunk_src+row-chunk_dst)*D+col];
    else if(row<old_rows)ring[t]=tmp[t];
}
static int run(void*vctx,void*request,const BnGPUSWARingUpdateResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;BnGPUSWARingUpdateInfo info;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||p->dim<2||p->dim>2048||p->dim%2||
       bn_gpu_swa_ring_update_info(p->n_tokens,p->pos0,p->window_beg,p->window,p->ring_rows,&info))return -1;
    BnGPUResidentView v[]={p->kv,p->ring,p->scratch};size_t n[]={(size_t)p->n_tokens*p->dim,(size_t)p->ring_rows*p->dim,(size_t)p->ring_rows*p->dim};void*data[3];
    for(int i=0;i<3;i++) {
        if(v[i].type!=BN_GGUF_TENSOR_F16||v[i].count!=n[i])return -1;
        data[i]=bn_cuda_gdn::view_data(r,v[i]);if(!data[i]||((uintptr_t)data[i]&1u))return -1;
        for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(v[i],v[j]))return -1;
    }
    if(bn_cuda_gdn::set_device(r))return -1;
    size_t copied=(size_t)info.old_rows*p->dim;
    if(copied){snapshot<<<(copied+255)/256,256,0,r->stream>>>((const uint16_t*)data[1],(uint16_t*)data[2],copied,(size_t)info.ring_src*p->dim);if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}}
    update<<<(n[1]+255)/256,256,0,r->stream>>>((const uint16_t*)data[0],(uint16_t*)data[1],(const uint16_t*)data[2],p->ring_rows,p->dim,
        info.old_rows,info.chunk_src,info.chunk_rows,info.chunk_dst);
    if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
}
}
#endif
