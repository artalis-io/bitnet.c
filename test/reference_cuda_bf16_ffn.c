/* Generate synthetic raw weights for reference_cuda_bf16_ffn.py.
 * cc -std=c11 -Iinclude -Itest test/reference_cuda_bf16_ffn.c -o /tmp/bf16-ffn-weights
 * /tmp/bf16-ffn-weights <existing-output-directory>
 */
#include "cuda_signed_mmq_fixture.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
static void save(const char*dir,int type,int rows,int cols,int seed){size_t bytes=signed_mmq_fixture_bytes(type,(size_t)rows*cols);void*w=malloc(bytes);assert(w);signed_mmq_fixture_weights(w,type,(size_t)rows*cols,seed);char path[512];snprintf(path,sizeof(path),"%s/w-%d-%d-%d-%d.bin",dir,type,rows,cols,seed);FILE*f=fopen(path,"wb");assert(f&&fwrite(w,1,bytes,f)==bytes);fclose(f);free(w);}
int main(int argc,char**argv){assert(argc==2);(void)signed_mmq_fixture_input;int dims[]={256,768},hidden[]={256,512,5376};int types[]={11,20,23};for(int d=0;d<2;d++)for(int h=0;h<3;h++)for(int t=0;t<3;t++)for(int seed=0;seed<2;seed++){save(argv[1],types[t],hidden[h],dims[d],seed);save(argv[1],types[t],dims[d],hidden[h],seed);}return 0;}
