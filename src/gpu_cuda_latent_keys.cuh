#ifndef BN_GPU_CUDA_LATENT_KEYS_CUH
#define BN_GPU_CUDA_LATENT_KEYS_CUH
/* Source order: exllamav3 bd1f5191 modules/dsv4.py::_indexer_keys (MIT). */
namespace bn_cuda_latent_keys {
static __global__ void positions(int*out,int R,int entry0,int rate) {
    int row=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if(row<R)out[row]=(entry0+row)*rate;
}
static int run(void*ctx,void*request,const BnGPUF16LatentKeysResidentPlan*p) {
    if(!p||p->entry0<0||p->rate<1||p->project.n_tokens<1||
       p->entry0>INT_MAX/p->rate||p->project.n_tokens-1>INT_MAX/p->rate-p->entry0)return -1;
    void*linear[2],*norm[3],*rope[4];int type;
    if(bn_cuda_f16_rows::validate_linear(ctx,request,&p->project,linear)||
       bn_cuda_native_norm::validate(ctx,request,&p->norm,norm,&type)||
       bn_cuda_rope::validate(ctx,request,&p->keys_rope,rope,&type)||
       bn_cuda_rope::validate(ctx,request,&p->latents_rope,rope,&type))return -1;
    if(p->project.n_tokens!=p->norm.n_tokens||p->project.n_tokens!=p->keys_rope.n_tokens||
       p->project.n_tokens!=p->latents_rope.n_tokens||p->project.n!=p->norm.dim||
       p->project.out.type!=BN_GGUF_TENSOR_F16||p->norm.out.type!=BN_GGUF_TENSOR_F16||
       !p->norm.weight||p->norm.weight_groups!=1||p->norm.constant_bias!=0.f||p->norm.constant_scale!=1.f||
       !bn_cuda_exl3::same_view(p->project.out,p->norm.x)||
       !bn_cuda_exl3::same_view(p->norm.out,p->keys_rope.x)||
       !bn_cuda_exl3::same_view(p->project.x,p->latents_rope.x)||
       !bn_cuda_exl3::same_view(p->keys_rope.inv_freq,p->latents_rope.inv_freq)||
       !p->keys_rope.positions.request||!bn_cuda_exl3::same_view(p->keys_rope.positions,p->latents_rope.positions)||
       p->keys_rope.norm||p->latents_rope.norm||p->keys_rope.heads!=1||p->latents_rope.heads!=1||
       p->keys_rope.head_stride!=p->project.n||p->latents_rope.head_stride!=p->project.k||
       p->keys_rope.dim!=p->keys_rope.rotary_dim||p->latents_rope.dim!=p->keys_rope.rotary_dim||
       p->latents_rope.rotary_dim!=p->keys_rope.rotary_dim||
       p->keys_rope.slice_offset!=p->keys_rope.head_stride-p->keys_rope.dim||
       p->latents_rope.slice_offset!=p->latents_rope.head_stride-p->latents_rope.dim||
       p->keys_rope.rotate_offset||p->latents_rope.rotate_offset||p->keys_rope.pos0||p->latents_rope.pos0)return -1;
    BnGPUResidentView live[]={p->project.x,p->project.out,p->norm.out,p->keys_rope.positions,p->keys_rope.inv_freq};
    for(int i=0;i<5;i++)for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(live[i],live[j]))return -1;
    bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
    int*ids=(int*)bn_cuda_gdn::view_data(r,p->keys_rope.positions);
    positions<<<(p->project.n_tokens+255)/256,256,0,r->stream>>>(ids,p->project.n_tokens,p->entry0,p->rate);
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    if(bn_cuda_f16_rows::linear(ctx,request,&p->project)||bn_cuda_native_norm::run(ctx,request,&p->norm)||
       bn_cuda_rope::run(ctx,request,&p->keys_rope)||bn_cuda_rope::run(ctx,request,&p->latents_rope))return -1;
    return 0;
}
}
#endif
