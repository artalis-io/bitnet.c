#ifndef BN_GPU_CUDA_ROWS_CUH
#define BN_GPU_CUDA_ROWS_CUH

namespace bn_cuda_rows {
using bn_cuda_gdn::ResidentRequest;
using bn_cuda_gdn::set_device;
using bn_cuda_gdn::view_data;
using bn_cuda_gdn::overlap;

/* Four adjacent outputs share two packed bytes and one FP16 scale. BF16
 * stores remain scalar so guarded slices need only their natural alignment. */
static __global__ void int4_fp16s32_kernel(__nv_bfloat16 *out,
    const unsigned char *raw,const __half *scales,size_t count,int cols) {
    size_t i=((size_t)blockIdx.x*blockDim.x+threadIdx.x)*4;
    if(i>=count)return;
    size_t row=i/(size_t)cols,col=i%(size_t)cols;
    unsigned packed=*(const unsigned short *)(raw+row*(cols/2)+col/2);
    float scale=__half2float(scales[row*(cols/32)+col/32]);
    for(int j=0;j<4;j++) {
        int q=(int)((packed>>(j*4))&15)-8;
        out[i+j]=__float2bfloat16_rn(__fmul_rn((float)q,scale));
    }
}

static int decode_scaled(ResidentRequest *r,const BnGPUQuantRowsBF16ResidentPlan *p,
    const BnQuantRowScaledOps *ops) {
    BnGPUScaledRowsWorkspace shape;
    if(ops->gpu_decode!=BN_QUANT_ROW_GPU_INT4_FP16S32_BF16||
        ops->block_cols!=32||ops->weight_bits!=4||ops->scale_element_bytes!=2||
        bn_gpu_scaled_rows_shape(p->rows,p->cols,&shape)||
        p->row_stride_bytes!=(size_t)p->cols/2||
        p->packed.type!=BN_GGUF_TENSOR_I32||p->packed.count!=shape.packed_i32||
        p->scales.type!=BN_GGUF_TENSOR_F16||p->scales.count!=shape.scales_f16||
        !bn_quant_format_is_bf16(p->out.type)||p->out.count!=shape.out_bf16||
        overlap(p->packed,p->scales)||overlap(p->packed,p->out)||overlap(p->scales,p->out))return -1;
    void *raw=view_data(r,p->packed),*scales=view_data(r,p->scales),*out=view_data(r,p->out);
    if(!raw||!scales||!out)return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    int4_fp16s32_kernel<<<(shape.out_bf16/4+127)/128,128,0,r->stream>>>(
        (__nv_bfloat16 *)out,(const unsigned char *)raw,(const __half *)scales,shape.out_bf16,p->cols);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

static __global__ void iq4nl_kernel(__nv_bfloat16 *out, const unsigned char *raw,
    size_t count, int cols, size_t stride) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    size_t row = i / (size_t)cols, col = i % (size_t)cols;
    const BnBlockIQ4NL *block = (const BnBlockIQ4NL *)(raw + row * stride) + col / 32;
    unsigned char q = block->qs[col % 16];
    int code = col % 32 < 16 ? q & 15 : q >> 4;
    out[i] = __float2bfloat16_rn(__fmul_rn(cuda_fp16_to_fp32(block->d),
                                        (float)bn_kvalues_iq4nl[code]));
}

static int decode(void *ctx_ptr, void *request, const BnGPUQuantRowsBF16ResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)request;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->rows <= 0 || p->cols <= 0 || p->rows > INT_MAX / p->cols || set_device(r))
        return -1;
    const BnQuantRowScaledOps *row = bn_quant_format_row_scaled_ops(p->type);
    if(row)return decode_scaled(r,p,row);
    if(p->scales.count||p->scales.request)return -1;
    const BnQuantFormatOps *f = bn_quant_format_ops(p->type);
    if (!f || f->bf16_decode != BN_QUANT_BF16_DECODE_IQ4_NL ||
        f->block_elems <= 0 || !f->bytes_per_block || p->cols % f->block_elems ||
        (size_t)(p->cols / f->block_elems) > (SIZE_MAX - 15) / f->bytes_per_block)
        return -1;
    size_t bytes = (size_t)(p->cols / f->block_elems) * f->bytes_per_block;
    size_t stride = (bytes + 15) & ~(size_t)15;
    if (p->row_stride_bytes != stride || (size_t)p->rows > SIZE_MAX / stride)
        return -1;
    size_t count = (size_t)p->rows * p->cols;
    void *raw = view_data(r, p->packed), *out = view_data(r, p->out);
    if (!raw || !out || p->packed.type != BN_GGUF_TENSOR_I32 ||
        p->packed.count != (size_t)p->rows * stride / sizeof(int32_t) ||
        !bn_quant_format_is_bf16(p->out.type) || p->out.count != count ||
        overlap(p->packed, p->out)) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    iq4nl_kernel<<<(count + 255) / 256, 256, 0, r->stream>>>(
        (__nv_bfloat16 *)out, (const unsigned char *)raw, count, p->cols, stride);
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}
} // namespace bn_cuda_rows
#endif
