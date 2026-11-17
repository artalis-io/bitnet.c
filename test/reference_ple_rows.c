/* Native actual-checkpoint batched PLE IDs and bounded raw row staging.
 * Link like test_bf16_runtime; run: probe FIRST_SHARD OUTPUT_DIRECTORY. */
#include "gguf.h"
#include "backend_quant.h"
#include "model.h"
#include "gpu_cuda.h"
#include "gpu_backend.h"
#include "backend_session.h"
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif
#include "transformer_prefill_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void save(const char *dir,int nt,const char *suffix,const void *data,size_t bytes) {
    char path[2048]; int len=snprintf(path,sizeof(path),"%s/%d-%s.bin",dir,nt,suffix);
    assert(len>0&&(size_t)len<sizeof(path));FILE*f=fopen(path,"wb");assert(f);
    assert(fwrite(data,1,bytes,f)==bytes);fclose(f);
}
int main(int argc,char **argv) {
    assert(argc==3||argc==4);BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);
    BnModel model = {0};
    assert(!bn_model_load_with_cpu_preparation(&model, f, 256, 0, 0, 0));
    BnConfig c = model.config;
    assert(c.ple_ngram_size == 3 && c.ple_heads_per_ngram == 8 &&
        c.ple_head_count == 16 && c.ple_eos_token_id == 248044);
    BnQWeight w = model.weights.ple_token_embd;
    assert(w.cols == 160 && w.rows == 320001536 && w.type == BN_GGUF_TENSOR_IQ4_NL);
    size_t stride=bn_backend_quant_host_row_stride(&w);assert(stride==96);
    void *stage=argc==3?malloc(129u*16*stride):NULL;assert(argc==4||stage);
    BnGPUBackend *gpu = NULL; BnBackendSession *session = NULL; void *request = NULL;
    BnGPUResidentView raw = {0}, output = {0};
    uint16_t *decoded = NULL; size_t words = 0;
    if (argc == 4) {
        gpu = bn_gpu_cuda_create(); session = bn_backend_session_create(); assert(gpu && session);
        request = bn_gpu_backend_resident_request(gpu, session); assert(request);
        assert(!gpu->resident_upload_reserve(request, 2, 129u*16*stride, 2u*129*16*stride));
        assert(!gpu->resident_alloc(request, BN_GGUF_TENSOR_I32, 129u*16*stride/4, &raw));
        assert(!gpu->resident_alloc(request, BN_GGUF_TENSOR_BF16, 129u*16*160, &output));
        decoded = malloc(129u*16*160*2); assert(decoded);
        puts("PLE_PINNED_POOL slots2 capacity198144 payload396288");
    }
    const int sizes[]={1,5,17,129};
    for(int ci=0;ci<4;ci++) {
        int nt=sizes[ci],nr=nt>=5?4:1;int32_t tokens[129],starts[5]={0},ctx[8],ids[129*16];
        for(int i=0;i<nt;i++)tokens[i]=i%11==5?248044:(i*37+97)%248000;
        starts[nr]=nt;if(nr==4){starts[1]=0;starts[2]=nt/3;starts[3]=nt/2;}
        for(int i=0;i<nr*2;i++)ctx[i]=i%3==0?248044:(i*19+97)%251;
        assert(!bn_transformer_prefill_ple_ids(&c,tokens,nt,starts,nr,ctx,(size_t)nr*2,ids,(size_t)nt*16));
        BnGPUHostUploadLease lease = {0};
        if (gpu) {
            assert(!gpu->resident_upload_acquire(request, (unsigned)ci % 2, &lease));
            assert(lease.capacity == 129u*16*stride); stage = lease.data;
        }
        assert(!bn_backend_quant_gather_host_rows(&w,ids,(size_t)nt*16,stage,(size_t)nt*16*stride));
        save(argv[2],nt,"ids",ids,(size_t)nt*16*sizeof(*ids));save(argv[2],nt,"packed",stage,(size_t)nt*16*stride);
        if (gpu) {
            BnGPUQuantRowsBF16ResidentPlan plan = {0}; plan.rows = nt*16; plan.cols = 160;
            plan.type = w.type; plan.row_stride_bytes = stride;
            plan.packed = bn_gpu_resident_view_slice(raw, 0, (size_t)nt*16*stride/4);
            plan.out = bn_gpu_resident_view_slice(output, 0, (size_t)nt*16*160);
            assert(!gpu->resident_upload_submit(request, &lease, plan.packed));
#ifdef BN_CAPTURE_CUDA_PROFILER
            if (nt == 5) {
                for (int warm = 0; warm < 3; warm++) {
                    assert(!gpu->quant_rows_bf16_resident(gpu->ctx, request, &plan));
                    assert(!gpu->resident_synchronize(request));
                }
                assert(cudaProfilerStart() == cudaSuccess);
                /* Reacquire only after the warm-up completion established DMA safety. */
                assert(!gpu->resident_upload_acquire(request, (unsigned)ci % 2, &lease));
                assert(!gpu->resident_upload_submit(request, &lease, plan.packed));
            }
#endif
            assert(!gpu->quant_rows_bf16_resident(gpu->ctx, request, &plan));
            assert(!gpu->resident_synchronize(request));
#ifdef BN_CAPTURE_CUDA_PROFILER
            if (nt == 5) assert(cudaProfilerStop() == cudaSuccess);
#endif
            assert(!gpu->resident_read(request, plan.out, decoded));
            char path[2048]; int len=snprintf(path,sizeof(path),"%s/%d-bf16.bin",argv[3],nt);
            assert(len>0&&(size_t)len<sizeof(path)); FILE *gold=fopen(path,"rb");assert(gold);
            size_t count=(size_t)nt*16*160; uint16_t *expected=malloc(count*2);assert(expected);
            assert(fread(expected,2,count,gold)==count&&fgetc(gold)==EOF);fclose(gold);
            assert(!memcmp(decoded,expected,count*2));free(expected);
            save(argv[2],nt,"bf16",decoded,count*2);words+=count;
        }

    }
    bn_backend_session_free(session); if(gpu)bn_gpu_cuda_destroy(gpu);free(decoded);
    printf("PLE_GPU_ROWS exact_words%zu\n",words);
    if (argc == 3) free(stage);
    bn_model_free(&model);
    bn_gguf_free(f);
    puts("PLE_RAM_ROWS PASSED: maximum staging 198144 bytes");
    return 0;
}
