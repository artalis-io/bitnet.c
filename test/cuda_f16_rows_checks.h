#ifdef BN_ENABLE_CUDA
typedef struct {int rows,cols,calls,fail_row;} F16RowsTestSource;
static uint16_t f16_rows_test_value(int row,int col,int cols){return(uint16_t)(((size_t)row*cols+col)*37+11);}
static int f16_rows_test_read(void *context,int row,int count,uint16_t *out,size_t capacity) {
    F16RowsTestSource *s=context;assert(row==s->calls*32&&count>0&&count<=32&&count<=s->rows-row&&capacity>=(size_t)count*s->cols);s->calls++;if(row==s->fail_row)return -1;
    for(int i=0;i<count;i++)for(int j=0;j<s->cols;j++)out[(size_t)i*s->cols+j]=f16_rows_test_value(row+i,j,s->cols);
    return 0;
}
static void run_f16_rows_reference(BnGPUBackend *gpu) {
    assert(gpu->buffer_create_f16_rows);BnCudaWeightPoolStats before,after;assert(!bn_gpu_cuda_weight_pool_stats(gpu,&before));
    const int shapes[][2]={{1,1},{35,67},{32,32},{256,256}};size_t words=0;
    for(size_t i=0;i<sizeof(shapes)/sizeof(shapes[0]);i++) {
        int rows=shapes[i][0],cols=shapes[i][1];size_t n=(size_t)rows*cols,tile=(size_t)(rows<32?rows:32)*cols*2;F16RowsTestSource s={rows,cols,0,-1};BnGPUF16RowsPlan plan={rows,cols,n*2+tile,tile,&s,f16_rows_test_read};
        plan.device_budget_bytes--;assert(!gpu->buffer_create_f16_rows(gpu->ctx,&plan)&&!s.calls);plan.device_budget_bytes++;plan.host_budget_bytes--;assert(!gpu->buffer_create_f16_rows(gpu->ctx,&plan)&&!s.calls);plan.host_budget_bytes++;
        void *weight=gpu->buffer_create_f16_rows(gpu->ctx,&plan);assert(weight&&s.calls==(rows+31)/32);uint16_t *out=malloc((n+2)*2);assert(out);for(size_t j=0;j<n+2;j++)out[j]=0xa55a;
        assert(bn_gpu_cuda_read_f16_rows(gpu,weight,out+1,n-1)==-1);for(size_t j=0;j<n+2;j++)assert(out[j]==0xa55a);
        BnGPUBackend wrong=*gpu;int dummy;wrong.ctx=&dummy;assert(bn_gpu_cuda_read_f16_rows(&wrong,weight,out+1,n)==-1);
        assert(!bn_gpu_cuda_read_f16_rows(gpu,weight,out+1,n));assert(out[0]==0xa55a&&out[n+1]==0xa55a);for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)assert(out[1+(size_t)c*rows+r]==f16_rows_test_value(r,c,cols));
        float *x=calloc((size_t)cols,sizeof(float)),*y=calloc((size_t)rows,sizeof(float));assert(x&&y);assert(gpu->matvec(gpu->ctx,y,weight,x,rows,cols,BN_GGUF_TENSOR_F16)==-1);assert(gpu->matmul(gpu->ctx,y,weight,x,rows,cols,1,BN_GGUF_TENSOR_F16)==-1);free(x);free(y);
        gpu->buffer_destroy(gpu->ctx,weight);free(out);words+=n;
    }
    for(int attempt=0;attempt<3;attempt++) {
        F16RowsTestSource s={35,67,0,32};size_t tile=32*67*2;BnGPUF16RowsPlan plan={35,67,35*67*2+tile,tile,&s,f16_rows_test_read};assert(!gpu->buffer_create_f16_rows(gpu->ctx,&plan)&&s.calls==2);s.calls=0;s.fail_row=-1;void *weight=gpu->buffer_create_f16_rows(gpu->ctx,&plan);assert(weight);gpu->buffer_destroy(gpu->ctx,weight);
    }
    assert(!bn_gpu_cuda_weight_pool_stats(gpu,&after));assert(before.idle_bytes==after.idle_bytes&&before.idle_entries==after.idle_entries);printf("CUDA bounded F16 row materialization/transpose/rollback PASSED: %zu exact words\n",words);
}
#endif
