#ifndef BN_QUANT_Q8_ROUTED_CUDA_CUH
#define BN_QUANT_Q8_ROUTED_CUDA_CUH
/* Borrowed single-expert Q8_0 projections. The backend owns activation
 * quantization and the immutable global MMQ interval schedule. */
static __global__ void q8_borrowed_mmvq_kernel(float *out,
    const BnBlockQ8_0 *weights, const BnCudaBlockQ8_1 *input,
    int rows, int cols, int logical_tokens) {
    int lane=threadIdx.x&31, row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    int token=blockIdx.y, nb=cols/32;
    if(row>=rows)return;
    float sum=q8_0_prepared_row_sum(weights+(size_t)row*nb,
        input+(size_t)token*nb,nb,lane,logical_tokens==1?4:1);
    if(!lane)out[(size_t)token*rows+row]=sum;
}
static __global__ void q8_borrowed_mmq_kernel(float *out,
    const BnBlockQ8_0 *weights, const BnCudaBlockQ8_0F32 *input,
    const int *schedule, int rows, int cols, int experts, int expert,
    int jwidth, int token_tiles) {
    int row=blockIdx.x*blockDim.x+threadIdx.x, token=blockIdx.y;
    if(row>=rows)return;
    int nb=cols/32;
    int tile=((row/128)*experts+expert)*token_tiles+token/jwidth;
    out[(size_t)token*rows+row]=q8_mmq_row(weights+(size_t)row*nb,
        input+(size_t)token*nb,schedule,tile);
}
#endif
