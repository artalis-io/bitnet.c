#ifndef BN_GPU_CUDA_MOE_JOINT_CUH
#define BN_GPU_CUDA_MOE_JOINT_CUH
/* Bounded by-value launch descriptors: no device allocation, persistent
 * foreign pointers or additional managed payload. Grid-constant parameters
 * avoid a per-thread local copy when indexing the cohort. */
struct JointGroupedBatch {
    GroupedBatch batch[16];
    int count, max_rows;
};
static_assert(sizeof(JointGroupedBatch) < 3072, "joint launch parameter bound");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define BN_MOE_GRID_CONSTANT __grid_constant__
#else
#define BN_MOE_GRID_CONSTANT
#endif

template<typename W,int Roles,bool Down>
static __global__ void bank_joint_vector(BN_MOE_GRID_CONSTANT const JointGroupedBatch group) {
    int entry=(int)blockIdx.z/Roles,experts=group.batch[0].experts;
    const GroupedBatch &b=group.batch[entry/experts];entry%=experts;
    BnGPUMoEBankItem item=b.items[entry];int token=(int)blockIdx.y,role=(int)blockIdx.z%Roles;
    if(token>=item.n_rows)return;
    int rows=Down?b.dim:b.hidden,cols=Down?b.hidden:b.dim;
    const unsigned char *base=b.raw_ptrs?b.raw_ptrs[entry]:b.raw+(size_t)entry*b.stride;
    const W *w=(const W*)(base+(Down?2:role)*b.gate_bytes);
    const BnCudaBlockQ8_1 *x=(Down?b.down_input:b.input)+(size_t)item.first_row*(cols/32);
    __nv_bfloat16 *out=(Down?b.out:role?b.up:b.gate)+(size_t)item.first_row*rows;
    grouped_vec_core(out,w,x,rows,cols,(int)blockIdx.x,token);
}
template<typename W,int Roles,bool Down,int Tokens>
static __global__ void bank_joint_matrix(BN_MOE_GRID_CONSTANT const JointGroupedBatch group) {
    const GroupedBatch &b=group.batch[blockIdx.z/Roles];
    BnGPUMoEBankItem item=b.items[blockIdx.y];int role=(int)blockIdx.z%Roles;
    int tiles=(group.max_rows-1)/Tokens+1,token0=((int)blockIdx.x%tiles)*Tokens;
    if(token0>=item.n_rows)return; /* Uniform before every MMA/shared barrier. */
    int rows=Down?b.dim:b.hidden,cols=Down?b.hidden:b.dim;
    const unsigned char *base=b.raw_ptrs?b.raw_ptrs[blockIdx.y]:b.raw+(size_t)blockIdx.y*b.stride;
    const W *w=(const W*)(base+(Down?2:role)*b.gate_bytes);
    const BnCudaBlockQ8_1 *x=(Down?b.down_input:b.input)+(size_t)item.first_row*(cols/32);
    __nv_bfloat16 *out=(Down?b.out:role?b.up:b.gate)+(size_t)item.first_row*rows;
    grouped_mma_core<Tokens>(out,w,x,rows,cols,item.n_rows,((int)blockIdx.x/tiles)*16,token0);
}
static __global__ void bank_joint_activate(BN_MOE_GRID_CONSTANT const JointGroupedBatch group) {
    const GroupedBatch &b=group.batch[blockIdx.z];BnGPUMoEBankItem item=b.items[blockIdx.y];
    size_t first=(size_t)item.first_row*b.hidden,i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(b.clamped) {
        if(i>=(size_t)item.n_rows*b.hidden)return;i+=first;
        float g=fminf(__bfloat162float(b.gate[i]),b.clamp_limit);
        float u=fmaxf(-b.clamp_limit,fminf(__bfloat162float(b.up[i]),b.clamp_limit));
        float s=__bfloat162float(__float2bfloat16_rn(g/(1.f+expf(-g))));
        b.gate[i]=__float2bfloat16_rn(__fmul_rn(s,__fadd_rn(u,0.f)));
    }else silu_mul_bf16_core(b.gate+first,b.up+first,(size_t)item.n_rows*b.hidden,i);
}
static __global__ void bank_joint_quantize(BN_MOE_GRID_CONSTANT const JointGroupedBatch group) {
    const GroupedBatch &b=group.batch[blockIdx.z];BnGPUMoEBankItem item=b.items[blockIdx.y];
    if(blockIdx.x>=(size_t)item.n_rows*(b.hidden/32))return;
    quantize_bf16_q8_affine_batch_core(b.down_input+(size_t)item.first_row*(b.hidden/32),
        b.gate+(size_t)item.first_row*b.hidden,blockIdx.x);
}
/* Preserve the private per-bank precision bucket, including mixed prefill and
 * decode. Combining row counts must never change MMVQ/MMQ or MMA tile math. */
static int joint_precision_bucket(const GroupedBatch &b) {
    return !b.matrix?0:b.max_rows<=8?8:b.max_rows<=24?24:64;
}
template<typename W,int Roles,bool Down,int Tokens>
static void joint_matrix_launch(ResidentRequest *r,const JointGroupedBatch &group) {
    int rows=Down?group.batch[0].dim:group.batch[0].hidden;
    unsigned tiles=(unsigned)(((size_t)rows+15)/16*(((size_t)group.max_rows+Tokens-1)/Tokens));
    bank_joint_matrix<W,Roles,Down,Tokens><<<dim3(tiles,group.batch[0].experts,group.count*Roles),Tokens*4,0,r->stream>>>(group);
}
template<typename W,int Roles,bool Down>
static int joint_project_launch(ResidentRequest *r,const JointGroupedBatch &all,size_t *launches) {
    const int buckets[]={0,8,24,64};
    for(int kind=0;kind<4;kind++) {
        JointGroupedBatch group={};
        for(int i=0;i<all.count;i++)if(joint_precision_bucket(all.batch[i])==buckets[kind]) {
            group.batch[group.count++]=all.batch[i];
            if(all.batch[i].max_rows>group.max_rows)group.max_rows=all.batch[i].max_rows;
        }
        if(!group.count)continue;
        if(!kind) {
            int rows=Down?group.batch[0].dim:group.batch[0].hidden;
            bank_joint_vector<W,Roles,Down><<<dim3(rows,group.max_rows,group.count*group.batch[0].experts*Roles),32,0,r->stream>>>(group);
        }else if(kind==1)joint_matrix_launch<W,Roles,Down,8>(r,group);
        else if(kind==2)joint_matrix_launch<W,Roles,Down,24>(r,group);
        else joint_matrix_launch<W,Roles,Down,64>(r,group);
        if(cudaGetLastError()!=cudaSuccess)return -1;
        (*launches)++;
    }
    return 0;
}
static int joint_grouped_execute(ResidentRequest *r,const JointGroupedBatch &all,
    BnQuantBF16MoEKind gate_kind,BnQuantBF16MoEKind down_kind,size_t *projections) {
    if(gate_kind==BN_QUANT_BF16_MOE_Q4_K) {
        if(joint_project_launch<BnBlockQ4K,2,false>(r,all,projections))return -1;
    }else if(gate_kind==BN_QUANT_BF16_MOE_Q5_K) {
        if(joint_project_launch<BnBlockQ5K,2,false>(r,all,projections))return -1;
    }else return -1;
    int experts=all.batch[0].experts,hidden=all.batch[0].hidden;
    bank_joint_activate<<<dim3((unsigned)(((size_t)all.max_rows*hidden+255)/256),experts,all.count),256,0,r->stream>>>(all);
    if(cudaGetLastError()!=cudaSuccess)return -1;
    bank_joint_quantize<<<dim3((unsigned)((size_t)all.max_rows*(hidden/32)),experts,all.count),32,0,r->stream>>>(all);
    if(cudaGetLastError()!=cudaSuccess)return -1;
    if(down_kind==BN_QUANT_BF16_MOE_Q5_1)return joint_project_launch<BnBlockQ5_1,1,true>(r,all,projections);
    if(down_kind==BN_QUANT_BF16_MOE_Q8_0)return joint_project_launch<BnBlockQ8_0,1,true>(r,all,projections);
    if(down_kind==BN_QUANT_BF16_MOE_Q5_K)return joint_project_launch<BnBlockQ5K,1,true>(r,all,projections);
    if(down_kind==BN_QUANT_BF16_MOE_Q6_K)return joint_project_launch<BnBlockQ6K,1,true>(r,all,projections);
    return -1;
}
#undef BN_MOE_GRID_CONSTANT
#endif
