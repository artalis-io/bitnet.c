/* Independent pinned GGML Q4_K packed CPU oracle; links no bitnet objects.
 * Requires ggml_backend_cpu_repack_buffer_type from the pinned CPU library.
 * Build as C++17 with ggml/include and versioned ggml/base/cpu libraries.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
// Internal C++ entry point: intentionally pinned, not a stable GGML API.
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);
static uint64_t hash_values(const float *x, size_t n) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) {
        uint32_t bits; std::memcpy(&bits, x+i, sizeof(bits));
        hash = (hash ^ bits) * UINT64_C(1099511628211);
    }
    return hash;
}
int main(int argc, char **argv) {
    const int rows=argc>1 ? std::atoi(argv[1]) : 136;
    assert(argc<=2 && (rows==128 || rows==136));
    constexpr int nb=3, cols=nb*256, tokens=33;
    std::vector<uint8_t> weights(rows*nb*144);
    std::vector<float> input(tokens*cols), reference(tokens*rows), output(tokens*rows);
    assert(ggml_type_size(GGML_TYPE_Q4_K)==144);
    for (int b=0;b<rows*nb;b++) {
        uint8_t *p=weights.data()+b*144;
        ggml_fp16_t d=ggml_fp32_to_fp16(0.002f*(b%19+1));
        ggml_fp16_t m=ggml_fp32_to_fp16(0.001f*(b%13+1));
        std::memcpy(p,&d,2);std::memcpy(p+2,&m,2);
        for(int i=0;i<12;i++)p[4+i]=(uint8_t)(b*17+i*23);
        for(int i=0;i<128;i++)p[16+i]=(uint8_t)(b*31+i*37);
    }
    for(int i=0;i<tokens*cols;i++)input[i]=sinf((float)i*0.13f);
    ggml_backend_t backend=ggml_backend_cpu_init();assert(backend);
    ggml_backend_cpu_set_n_threads(backend,4);
    ggml_init_params params={4*1024*1024,nullptr,true};
    ggml_context *wc=ggml_init(params);assert(wc);
    ggml_tensor *w=ggml_new_tensor_2d(wc,GGML_TYPE_Q4_K,cols,rows);
    auto buft=ggml_backend_cpu_repack_buffer_type();assert(buft);
    auto wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,buft);assert(wb);
    ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(w,weights.data(),0,weights.size());
    std::printf("buffer=%s rows=%d cols=%d\n",ggml_backend_buft_name(buft),rows,cols);
    const int counts[]={33,1,2,3,4,5,6,7,8,9,17};
    for(int count:counts) {
        ggml_context *ctx=ggml_init(params);assert(ctx);
        auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,count);
        auto y=ggml_mul_mat(ctx,w,x);
        assert(ggml_backend_supports_op(backend,y));
        auto graph=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(graph,y);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);assert(buffer);
        ggml_backend_tensor_set(x,input.data(),0,(size_t)count*cols*sizeof(float));
        assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(y,output.data(),0,(size_t)count*rows*sizeof(float));
        if(count==33)reference=output;
        int different=0;float max_diff=0;
        for(int i=0;i<count*rows;i++) {
            different+=std::memcmp(output.data()+i,reference.data()+i,sizeof(float))!=0;
            max_diff=std::fmax(max_diff,std::fabs(output[i]-reference[i]));
        }
        std::printf("{%d, UINT64_C(0x%016llx)}, changed=%d max_diff=%.9g\n",count,
            (unsigned long long)hash_values(output.data(),(size_t)count*rows),different,max_diff);
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_buffer_free(wb);ggml_free(wc);ggml_backend_free(backend);
}
