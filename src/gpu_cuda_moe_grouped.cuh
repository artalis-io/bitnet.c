#ifndef BN_GPU_CUDA_MOE_GROUPED_CUH
#define BN_GPU_CUDA_MOE_GROUPED_CUH
/* Included inside bn_cuda_moe_resident, after the qualified format cores.
 * Each expert retains independent rowwise arithmetic and the original total-
 * batch MMVQ/MMQ precision selection. Matrix Q4_K/Q5_K/Q5_1 use integer MMA;
 * Q8_0 down uses its separate F32 scale contract. Vector projections retain
 * their DP4A cores. */

static __device__ __forceinline__ void grouped_vec_core(__nv_bfloat16 *out,
    const BnBlockQ6K *w, const BnCudaBlockQ8_1 *x, int rows, int cols, int row, int token) {
    q6k_bf16_mmvq_core(out,w,x,rows,cols,row,token);
}
static __device__ __forceinline__ void grouped_mat_core(__nv_bfloat16 *out,
    const BnBlockQ6K *w, const BnCudaBlockQ8_1 *x, int rows, int cols, int nt,
    int row, int token0, BnCudaBlockQ8_1 activation[4][8]) {
    q6k_bf16_mmq_core(out,w,x,rows,cols,nt,row,token0,activation);
}

static __device__ __forceinline__ void grouped_vec_core(__nv_bfloat16 *out,
    const BnBlockQ4K *w, const BnCudaBlockQ8_1 *x, int rows, int cols, int row, int token) {
    q4k_bf16_mmvq_core(out, w, x, rows, cols, row, token);
}
static __device__ __forceinline__ void grouped_mat_core(__nv_bfloat16 *out,
    const BnBlockQ4K *w, const BnCudaBlockQ8_1 *x, int rows, int cols, int nt,
    int row, int token0, BnCudaBlockQ8_1 activation[4][8]) {
    q4k_bf16_mmq_core(out, w, x, rows, cols, nt, row, token0, activation);
}

static __device__ __forceinline__ void grouped_vec_core(__nv_bfloat16 *out,
    const BnBlockQ5K *w, const BnCudaBlockQ8_1 *x, int rows, int cols, int row, int token) {
    q5k_bf16_mmvq_core(out, w, x, rows, cols, row, token);
}
static __device__ __forceinline__ void grouped_mat_core(__nv_bfloat16 *out,
    const BnBlockQ5K *w, const BnCudaBlockQ8_1 *x, int rows, int cols, int nt,
    int row, int token0, BnCudaBlockQ8_1 activation[4][8]) {
    q5k_bf16_mmq_core(out, w, x, rows, cols, nt, row, token0, activation);
}

static __device__ __forceinline__ void grouped_vec_core(__nv_bfloat16 *out,
    const BnBlockQ5_1 *w, const BnCudaBlockQ8_1 *x, int rows, int cols, int row, int token) {
    q5_1_bf16_mmvq_core(out, w, x, rows, cols, row, token);
}
static __device__ __forceinline__ void grouped_mat_core(__nv_bfloat16 *out,
    const BnBlockQ5_1 *w, const BnCudaBlockQ8_1 *x, int rows, int cols, int nt,
    int row, int token0, BnCudaBlockQ8_1 activation[4][8]) {
    q5_1_bf16_mmq_core(out, w, x, rows, cols, nt, row, token0, &activation[0][0]);
}

static __device__ __forceinline__ void grouped_vec_core(__nv_bfloat16 *out,
    const BnBlockQ8_0 *w, const BnCudaBlockQ8_1 *x, int rows, int cols, int row, int token) {
    q8_bf16_mmvq_core(out, w, x, rows, cols, row, token);
}
static __device__ __forceinline__ void grouped_mat_core(__nv_bfloat16 *out,
    const BnBlockQ8_0 *w, const BnCudaBlockQ8_1 *x, int rows, int cols, int nt,
    int row, int token0, BnCudaBlockQ8_1 activation[4][8]) {
    q8_bf16_mmq_core(out, w, x, rows, cols, nt, row, token0, &activation[0][0]);
}


template<typename W, int Roles, bool Down, bool Matrix>
static __global__ void bank_project_grouped(__nv_bfloat16 *out0, __nv_bfloat16 *out1,
    const unsigned char *raw, const unsigned char *const *raw_ptrs, size_t stride, size_t gate_bytes, const BnGPUMoEBankItem *items,
    const BnCudaBlockQ8_1 *input, int rows, int cols, int max_rows) {
    int entry = Matrix ? blockIdx.y : blockIdx.z / Roles;
    int role = Matrix ? blockIdx.z : blockIdx.z % Roles;
    BnGPUMoEBankItem item = items[entry];
    int tiles = (max_rows - 1) / 4 + 1;
    int token0 = Matrix ? (blockIdx.x % tiles) * 4 : blockIdx.y;
    if (token0 >= item.n_rows) return; /* Uniform before any shared barrier. */
    const unsigned char *base = raw_ptrs ? raw_ptrs[entry] : raw + (size_t)entry * stride;
    const W *w = (const W *)(base + (Down ? 2 : role) * gate_bytes);
    const BnCudaBlockQ8_1 *x = input + (size_t)item.first_row * (cols / 32);
    __nv_bfloat16 *out = (role ? out1 : out0) + (size_t)item.first_row * rows;
    if (Matrix) {
        __shared__ BnCudaBlockQ8_1 activation[4][8];
        int row = (blockIdx.x / tiles) * 128 + threadIdx.x;
        grouped_mat_core(out, w, x, rows, cols, item.n_rows, row, token0, activation);
    } else {
        grouped_vec_core(out, w, x, rows, cols, blockIdx.x, token0);
    }
}

static __global__ void bank_activate_grouped(__nv_bfloat16 *gate, const __nv_bfloat16 *up,
    const BnGPUMoEBankItem *items, int hidden,int clamped,float limit) {
    BnGPUMoEBankItem item = items[blockIdx.y];
    size_t first = (size_t)item.first_row * hidden;
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(clamped){if(i>=(size_t)item.n_rows*hidden)return;i+=first;
        float g=fminf(__bfloat162float(gate[i]),limit),u=fmaxf(-limit,fminf(__bfloat162float(up[i]),limit));
        float s=__bfloat162float(__float2bfloat16_rn(g/(1.f+expf(-g))));
        gate[i]=__float2bfloat16_rn(__fmul_rn(s,__fadd_rn(u,0.f)));
    }else silu_mul_bf16_core(gate+first,up+first,(size_t)item.n_rows*hidden,i);
}

static __global__ void bank_quantize_grouped(BnCudaBlockQ8_1 *out, const __nv_bfloat16 *gate,
    const BnGPUMoEBankItem *items, int hidden) {
    BnGPUMoEBankItem item = items[blockIdx.y];
    if (blockIdx.x >= (size_t)item.n_rows * (hidden / 32)) return;
    quantize_bf16_q8_affine_batch_core(out + (size_t)item.first_row * (hidden / 32),
        gate + (size_t)item.first_row * hidden, blockIdx.x);
}

struct GroupedBatch {
    const unsigned char *raw;
    const unsigned char *const *raw_ptrs;
    size_t stride, gate_bytes;
    const BnGPUMoEBankItem *items;
    const BnCudaBlockQ8_1 *input;
    BnCudaBlockQ8_1 *down_input;
    __nv_bfloat16 *gate, *up, *out;
    int dim, hidden, max_rows, experts, matrix, clamped;
    float clamp_limit;
};

template<int Tokens,typename W>
static __device__ __forceinline__ void grouped_mma_core(__nv_bfloat16 *out,const W *w,
    const BnCudaBlockQ8_1 *x,int rows,int cols,int nt,int row0,int token0) {
    cuda_bf16_affine_mma_core<W,Tokens>(out,w,x,rows,cols,nt,row0,token0);
}
template<int Tokens>
static __device__ __forceinline__ void grouped_mma_core(__nv_bfloat16 *out,const BnBlockQ5_1 *w,
    const BnCudaBlockQ8_1 *x,int rows,int cols,int nt,int row0,int token0) {
    cuda_bf16_q51_mma_core<Tokens>(out,w,x,rows,cols,nt,row0,token0);
}
template<int Tokens>
static __device__ __forceinline__ void grouped_mma_core(__nv_bfloat16 *out,const BnBlockQ8_0 *w,
    const BnCudaBlockQ8_1 *x,int rows,int cols,int nt,int row0,int token0) {
    cuda_bf16_q8_mma_core<Tokens>(out,w,x,rows,cols,nt,row0,token0);
}
template<int Tokens>
static __device__ __forceinline__ void grouped_mma_core(__nv_bfloat16 *out,const BnBlockQ6K *w,
    const BnCudaBlockQ8_1 *x,int rows,int cols,int nt,int row0,int token0) {
    cuda_bf16_q6_mma_core<Tokens>(out,w,x,rows,cols,nt,row0,token0);
}
template<typename W,int Roles,bool Down,int Tokens>
static __global__ void bank_project_grouped_mma(__nv_bfloat16 *out0,__nv_bfloat16 *out1,
    const unsigned char *raw,const unsigned char *const *raw_ptrs,size_t stride,size_t gate_bytes,
    const BnGPUMoEBankItem *items,const BnCudaBlockQ8_1 *input,int rows,int cols,int max_rows) {
    BnGPUMoEBankItem item=items[blockIdx.y];
    int tiles=(max_rows-1)/Tokens+1,token0=(blockIdx.x%tiles)*Tokens;
    if(token0>=item.n_rows)return; /* Uniform before shared barriers/MMA. */
    const unsigned char *base=raw_ptrs?raw_ptrs[blockIdx.y]:raw+(size_t)blockIdx.y*stride;
    const W *w=(const W*)(base+(Down?2:blockIdx.z)*gate_bytes);
    const BnCudaBlockQ8_1 *x=input+(size_t)item.first_row*(cols/32);
    __nv_bfloat16 *out=(blockIdx.z?out1:out0)+(size_t)item.first_row*rows;
    grouped_mma_core<Tokens>(out,w,x,rows,cols,item.n_rows,(blockIdx.x/tiles)*16,token0);
}
template<typename W,int Roles,bool Down,int Tokens>
static void grouped_mma_tile_launch(ResidentRequest *r,const GroupedBatch *b) {
    int rows=Down?b->dim:b->hidden,cols=Down?b->hidden:b->dim;
    unsigned tiles=(unsigned)(((size_t)rows+15)/16*(((size_t)b->max_rows+Tokens-1)/Tokens));
    bank_project_grouped_mma<W,Roles,Down,Tokens><<<dim3(tiles,b->experts,Roles),Tokens*4,0,r->stream>>>(
        Down?b->out:b->gate,Down?NULL:b->up,b->raw,b->raw_ptrs,b->stride,b->gate_bytes,
        b->items,Down?b->down_input:b->input,rows,cols,b->max_rows);
}
template<typename W,int Roles,bool Down>
static void grouped_mma_project_launch(ResidentRequest *r,const GroupedBatch *b) {
    if(b->max_rows<=8)grouped_mma_tile_launch<W,Roles,Down,8>(r,b);
    else if(b->max_rows<=24)grouped_mma_tile_launch<W,Roles,Down,24>(r,b);
    else grouped_mma_tile_launch<W,Roles,Down,64>(r,b);
}

template<typename W, int Roles, bool Down>
static void grouped_project_launch(ResidentRequest *r, const GroupedBatch *b) {
    int rows = Down ? b->dim : b->hidden, cols = Down ? b->hidden : b->dim;
    __nv_bfloat16 *out0 = Down ? b->out : b->gate, *out1 = Down ? NULL : b->up;
    const BnCudaBlockQ8_1 *input = Down ? b->down_input : b->input;
    if (b->matrix) {
        unsigned tiles = (unsigned)(((size_t)rows + 127) / 128 * (((size_t)b->max_rows + 3) / 4));
        bank_project_grouped<W, Roles, Down, true><<<dim3(tiles, b->experts, Roles), 128, 0, r->stream>>>(
            out0, out1, b->raw, b->raw_ptrs, b->stride, b->gate_bytes, b->items, input, rows, cols, b->max_rows);
    } else {
        bank_project_grouped<W, Roles, Down, false><<<dim3(rows, b->max_rows, b->experts * Roles), 32, 0, r->stream>>>(
            out0, out1, b->raw, b->raw_ptrs, b->stride, b->gate_bytes, b->items, input, rows, cols, b->max_rows);
    }
}

static int grouped_execute(ResidentRequest *r, const GroupedBatch *b,
    BnQuantBF16MoEKind gate_kind, BnQuantBF16MoEKind down_kind) {
    if (gate_kind == BN_QUANT_BF16_MOE_Q4_K) {
        if(b->matrix)grouped_mma_project_launch<BnBlockQ4K,2,false>(r,b);
        else grouped_project_launch<BnBlockQ4K,2,false>(r,b);
    } else if (gate_kind == BN_QUANT_BF16_MOE_Q5_K) {
        if(b->matrix)grouped_mma_project_launch<BnBlockQ5K,2,false>(r,b);
        else grouped_project_launch<BnBlockQ5K,2,false>(r,b);
    }
    else return -1;
    if (cudaGetLastError() != cudaSuccess) return -1;
    bank_activate_grouped<<<dim3((unsigned)(((size_t)b->max_rows * b->hidden + 255) / 256), b->experts),
        256, 0, r->stream>>>(b->gate, b->up, b->items, b->hidden,b->clamped,b->clamp_limit);
    if (cudaGetLastError() != cudaSuccess) return -1;
    bank_quantize_grouped<<<dim3((unsigned)((size_t)b->max_rows * (b->hidden / 32)), b->experts),
        32, 0, r->stream>>>(b->down_input, b->gate, b->items, b->hidden);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (down_kind == BN_QUANT_BF16_MOE_Q5_1) {
        if(b->matrix)grouped_mma_project_launch<BnBlockQ5_1,1,true>(r,b);
        else grouped_project_launch<BnBlockQ5_1,1,true>(r,b);
    }
    else if (down_kind == BN_QUANT_BF16_MOE_Q8_0) {
        if(b->matrix)grouped_mma_project_launch<BnBlockQ8_0,1,true>(r,b);
        else grouped_project_launch<BnBlockQ8_0,1,true>(r,b);
    }
    else if(down_kind==BN_QUANT_BF16_MOE_Q5_K){
        if(b->matrix)grouped_mma_project_launch<BnBlockQ5K,1,true>(r,b);
        else grouped_project_launch<BnBlockQ5K,1,true>(r,b);
    }
    else if(down_kind==BN_QUANT_BF16_MOE_Q6_K){
        if(b->matrix)grouped_mma_project_launch<BnBlockQ6K,1,true>(r,b);
        else grouped_project_launch<BnBlockQ6K,1,true>(r,b);
    }
    else return -1;
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
#endif
