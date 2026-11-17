/* Private sparse-attention merge qualification. No production fallback.
 * Usage: probe CASE_INDEX. Rows: partial lse gate expected rows splits. */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../src/gpu_cuda_qsa_attention.cuh"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static void check(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));abort();} }
static void *load(const char *path,size_t n) {
    FILE*f=fopen(path,"rb");assert(f);void*p=malloc(n);assert(p);
    assert(fread(p,1,n,f)==n&&fgetc(f)==EOF);fclose(f);return p;
}
static void *upload(const void*p,size_t n) {void*d=NULL;check(cudaMalloc(&d,n));check(cudaMemcpy(d,p,n,cudaMemcpyHostToDevice));return d;}
int main(int argc,char **argv) {
    assert(argc==2);check(cudaSetDevice(0));FILE*f=fopen(argv[1],"r");assert(f);
    char pp[4096],lp[4096],gp[4096],ep[4096];int rows,splits,cases=0;size_t total=0,errors=0;
    while(fscanf(f,"%4095s %4095s %4095s %4095s %d %d",pp,lp,gp,ep,&rows,&splits)==6) {
        assert(rows>0&&(splits==4||splits==8||splits==16||splits==64));
        size_t n=(size_t)rows*24*256,np=n*splits,nl=(size_t)rows*24*splits;
        void*p=load(pp,np*4),*l=load(lp,nl*4),*g=load(gp,n*2),*e=load(ep,n*2);
        float*dp=(float*)upload(p,np*4),*dl=(float*)upload(l,nl*4);
        __nv_bfloat16*dg=(__nv_bfloat16*)upload(g,n*2),*out=NULL;
        check(cudaMalloc((void**)&out,n*2));
        #define RUN(S) bn_cuda_qsa_attention::merge_kernel<S><<<dim3(rows,24),256>>>(dp,dl,dg,out,rows)
        switch(splits){case 4:RUN(4);break;case 8:RUN(8);break;case 16:RUN(16);break;case 64:RUN(64);break;}
        #undef RUN
        check(cudaGetLastError());unsigned short*actual=(unsigned short*)malloc(n*2);assert(actual);
        check(cudaMemcpy(actual,out,n*2,cudaMemcpyDeviceToHost));
        size_t count=0;for(size_t i=0;i<n;i++)if(actual[i]!=((unsigned short*)e)[i]) {
            if(count<3)fprintf(stderr,"case%d split%d row%d word%zu got%04x expected%04x\n",cases,splits,rows,i,actual[i],((unsigned short*)e)[i]);count++;
        }
        printf("case%d rows%d splits%d words%zu differences%zu\n",cases,rows,splits,n,count);fflush(stdout);total+=n;errors+=count;cases++;
        free(actual);free(p);free(l);free(g);free(e);check(cudaFree(dp));check(cudaFree(dl));check(cudaFree(dg));check(cudaFree(out));
    }
    fclose(f);printf("QSA_MERGE cases%d words%zu differences%zu\n",cases,total,errors);return errors?1:0;
}
