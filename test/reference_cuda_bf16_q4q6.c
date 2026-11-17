/* Generate synthetic raw weights for reference_cuda_bf16_q4q6.py.
 * cc -std=c11 -Iinclude -Itest test/reference_cuda_bf16_q4q6.c -o /tmp/bf16-weights
 * /tmp/bf16-weights <existing-output-directory>
 */
#include "cuda_signed_mmq_fixture.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
static void save(const char *directory,int type,int rows,int cols,int seed) {
 size_t bytes=signed_mmq_fixture_bytes(type,(size_t)rows*cols);void*w=malloc(bytes);assert(w);signed_mmq_fixture_weights(w,type,(size_t)rows*cols,seed);
 char f[512];snprintf(f,sizeof(f),"%s/w-%d-%d-%d-%d.bin",directory,type,rows,cols,seed);FILE*out=fopen(f,"wb");assert(out&&fwrite(w,1,bytes,out)==bytes);fclose(out);free(w);
}
int main(int argc,char **argv){assert(argc==2);(void)signed_mmq_fixture_input;int types[]={12,14};for(int t=0;t<2;t++){for(int r=0;r<2;r++)for(int s=0;s<2;s++)save(argv[1],types[t],r?129:33,256,s);save(argv[1],types[t],1,256,0);save(argv[1],types[t],5120,256,0);save(argv[1],types[t],5121,256,0);save(argv[1],types[t],33,768,0);save(argv[1],types[t],33,5120,0);save(argv[1],types[t],33,6144,0);}return 0;}
