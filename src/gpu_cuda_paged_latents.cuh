#ifndef BN_GPU_CUDA_PAGED_LATENTS_CUH
#define BN_GPU_CUDA_PAGED_LATENTS_CUH
namespace bn_cuda_paged_latents {
static __global__ void publish(const uint16_t*x,const uint16_t*k,const int*bt,
    uint16_t*c,uint16_t*r,uint16_t*i,int rows,int entry0,int epp,int D,int RD,int I) {
    size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x,total=(size_t)rows*(D+I);
    if(t>=total)return;int row=(int)(t/(D+I)),col=(int)(t%(D+I)),e=entry0+row;
    size_t dst=(size_t)bt[e/epp]*epp+e%epp;
    if(col<D-RD)c[dst*(D-RD)+col]=x[(size_t)row*D+col];
    else if(col<D)r[dst*RD+col-(D-RD)]=x[(size_t)row*D+col];
    else i[dst*I+col-D]=k[(size_t)row*I+col-D];
}
static int run(void*vctx,void*request,const BnGPUPagedLatentPublishResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||
       p->n_rows<1||p->n_rows>65535||p->entry0<0||p->entries_per_page<1||p->entries_per_page>65535||
       p->pages<1||p->pages>65535||p->pages>INT_MAX/p->entries_per_page||
       p->n_rows>p->pages*p->entries_per_page||p->entry0>p->pages*p->entries_per_page-p->n_rows||
       p->dim<4||p->dim>2048||p->dim%2||p->rotary_dim<2||p->rotary_dim%2||p->rotary_dim>=p->dim||
       p->index_dim<2||p->index_dim>2048||p->index_dim%2)return -1;
    size_t R=(size_t)p->n_rows,C=(size_t)p->pages*p->entries_per_page;
    BnGPUResidentView v[]={p->latents,p->keys,p->block_table,p->pool_c,p->pool_r,p->pool_idx};
    size_t counts[]={R*p->dim,R*p->index_dim,(size_t)p->pages,C*(p->dim-p->rotary_dim),C*p->rotary_dim,C*p->index_dim};void*data[6];
    for(int j=0;j<6;j++) {
        if(counts[j]>SIZE_MAX/4||v[j].type!=(j==2?BN_GGUF_TENSOR_I32:BN_GGUF_TENSOR_F16)||v[j].count!=counts[j])return -1;
        data[j]=bn_cuda_gdn::view_data(r,v[j]);if(!data[j]||((uintptr_t)data[j]&(j==2?3u:1u)))return -1;
        for(int k=0;k<j;k++)if(bn_cuda_gdn::overlap(v[j],v[k]))return -1;
    }
    if(bn_cuda_gdn::set_device(r))return -1;
    size_t n=R*(p->dim+p->index_dim);
    publish<<<(n+255)/256,256,0,r->stream>>>((const uint16_t*)data[0],(const uint16_t*)data[1],(const int*)data[2],
        (uint16_t*)data[3],(uint16_t*)data[4],(uint16_t*)data[5],p->n_rows,p->entry0,p->entries_per_page,p->dim,p->rotary_dim,p->index_dim);
    if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
}
}
#endif
