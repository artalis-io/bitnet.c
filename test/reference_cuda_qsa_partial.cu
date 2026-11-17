/* Private tiled QSA partial + merge qualification. Explicit captured Q/K/V,
 * canonical selections and gates are stage inputs, not full-model qualification. */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../src/gpu_cuda_qsa_attention.cuh"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));abort();}}
static void *load(const char*path,size_t *bytes){FILE*f=fopen(path,"rb");assert(f);assert(!fseek(f,0,SEEK_END));long end=ftell(f);assert(end>0&&!fseek(f,0,SEEK_SET));*bytes=(size_t)end;void*p=malloc(*bytes);assert(p&&fread(p,1,*bytes,f)==*bytes&&fgetc(f)==EOF);fclose(f);return p;}
static void *upload(const void*p,size_t n){void*d=NULL;check(cudaMalloc(&d,n));if(p)check(cudaMemcpy(d,p,n,cudaMemcpyHostToDevice));else check(cudaMemset(d,0,n));return d;}
static void *file_buffer(const char*path,size_t expected){size_t n;void*p=load(path,&n);assert(n==expected);void*d=upload(p,n);free(p);return d;}
static size_t compare(const char*path,void*device,size_t count,int width,int ci,const char*stage){
 size_t bytes;void*gold=load(path,&bytes);assert(bytes==count*width);void*out=malloc(bytes);assert(out);check(cudaMemcpy(out,device,bytes,cudaMemcpyDeviceToHost));size_t differences=0;double maximum=0;
 for(size_t i=0;i<count;i++)if(memcmp((char*)out+i*width,(char*)gold+i*width,width)){
  if(width==4){float a=((float*)out)[i],b=((float*)gold)[i];if(isfinite(a)&&isfinite(b))maximum=fmax(maximum,fabs((double)a-b));if(differences<2)fprintf(stderr,"case%d %s word%zu %.9g expected%.9g\n",ci,stage,i,a,b);}
  else if(differences<2)fprintf(stderr,"case%d %s word%zu %04x expected%04x\n",ci,stage,i,((unsigned short*)out)[i],((unsigned short*)gold)[i]);
  differences++;
 }
 printf("case%d %s words%zu differences%zu max_abs%.9g\n",ci,stage,count,differences,maximum);fflush(stdout);free(gold);free(out);return differences;
}
template<int BN,int W,int S>
static void launch(const __nv_bfloat16*q,const __nv_bfloat16*k,const __nv_bfloat16*v,const int*indices,const int*table,const int*requests,float*partial,float*lse,const __nv_bfloat16*gate,__nv_bfloat16*out,int rows,int pages,int blocks,int reqs,cudaStream_t stream){
 using namespace bn_cuda_qsa_attention;
 size_t shared=sizeof(PartialShared<BN>);
 partial_kernel<BN,W,S><<<dim3(rows,2,S),128,shared,stream>>>(q,k,v,indices,table,requests,partial,lse,gate,out,rows,pages,blocks,reqs);
 check(cudaGetLastError());
 if(S>1){merge_kernel<S><<<dim3(rows,24),256,0,stream>>>(partial,lse,gate,out,rows);check(cudaGetLastError());}
}
/* Single-split specialization never launches a merge. */
template<int BN,int W>
static void launch_single(const __nv_bfloat16*q,const __nv_bfloat16*k,const __nv_bfloat16*v,const int*indices,const int*table,const int*requests,float*partial,float*lse,const __nv_bfloat16*gate,__nv_bfloat16*out,int rows,int pages,int blocks,int reqs,cudaStream_t stream){
 using namespace bn_cuda_qsa_attention;size_t shared=sizeof(PartialShared<BN>);
 partial_kernel<BN,W,1><<<dim3(rows,2),128,shared,stream>>>(q,k,v,indices,table,requests,partial,lse,gate,out,rows,pages,blocks,reqs);check(cudaGetLastError());
}
template<int BN,int W,int S> static void configure(){
 check(cudaFuncSetAttribute(bn_cuda_qsa_attention::partial_kernel<BN,W,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sizeof(bn_cuda_qsa_attention::PartialShared<BN>)));
}
int main(int argc,char**argv){
 assert(argc>=2&&argc<=4);bool paged=false,timing=false;
 for(int i=2;i<argc;i++){if(!strcmp(argv[i],"--paged"))paged=true;else if(!strcmp(argv[i],"--timing"))timing=true;else assert(!"unknown option");}
 check(cudaSetDevice(0));
 configure<32,4,64>();configure<32,1,16>();configure<32,1,8>();configure<32,1,4>();configure<64,2,4>();configure<32,1,1>();configure<64,2,1>();
 FILE*f=fopen(argv[1],"r");assert(f);
 char qp[4096],kp[4096],vp[4096],ip[4096],gp[4096],pp[4096],lp[4096],op[4096];int rows,bn,warps,splits,cases=0;size_t errors=0,partial_words=0,lse_words=0,out_words=0;
 while(fscanf(f,"%4095s %4095s %4095s %4095s %4095s %4095s %4095s %4095s %d %d %d %d",qp,kp,vp,ip,gp,pp,lp,op,&rows,&bn,&warps,&splits)==12){
  assert(rows>0);size_t n=(size_t)rows*24*256,raw_bytes;void*raw=load(kp,&raw_bytes);
  int pages,blocks,reqs=1;int *table,*requests;__nv_bfloat16*k,*v;
  if(paged) {
   char tp[4096],rp[4096];assert(fscanf(f,"%4095s %4095s %d %d",tp,rp,&pages,&reqs)==4);
   assert(raw_bytes%(16*1024)==0&&pages>0&&reqs>0);blocks=(int)(raw_bytes/(16*1024));
   k=(__nv_bfloat16*)upload(raw,raw_bytes);free(raw);size_t value_bytes;raw=load(vp,&value_bytes);assert(value_bytes==raw_bytes);v=(__nv_bfloat16*)upload(raw,value_bytes);free(raw);
   table=(int*)file_buffer(tp,(size_t)pages*reqs*4);requests=(int*)file_buffer(rp,(size_t)rows*4);
  } else {
   assert(raw_bytes%1024==0);int visible=(int)(raw_bytes/1024);pages=(visible+15)/16;blocks=pages+1;
   size_t cache_bytes=(size_t)blocks*16*1024;void*cache=calloc(1,cache_bytes);assert(cache);memcpy((char*)cache+16*1024,raw,raw_bytes);free(raw);
   k=(__nv_bfloat16*)upload(cache,cache_bytes);raw=load(vp,&raw_bytes);assert(raw_bytes==(size_t)visible*1024);memcpy((char*)cache+16*1024,raw,raw_bytes);free(raw);v=(__nv_bfloat16*)upload(cache,cache_bytes);free(cache);
   int*host_table=(int*)malloc(pages*sizeof(int));assert(host_table);for(int i=0;i<pages;i++)host_table[i]=i+1;table=(int*)upload(host_table,pages*sizeof(int));free(host_table);requests=(int*)upload(NULL,rows*sizeof(int));
  }
  auto*q=(__nv_bfloat16*)file_buffer(qp,n*2);auto*gate=(__nv_bfloat16*)file_buffer(gp,n*2);int*indices=(int*)file_buffer(ip,(size_t)rows*2052*4);
  auto*out=(__nv_bfloat16*)upload(NULL,n*2);float*partial=(float*)upload(NULL,n*splits*4),*lse=(float*)upload(NULL,(size_t)rows*24*splits*4);
  #define ARGS q,k,v,indices,table,requests,partial,lse,gate,out,rows,pages,blocks,reqs,stream
  auto submit=[&](cudaStream_t stream){
  if(bn==32&&warps==4&&splits==64)launch<32,4,64>(ARGS);
  else if(bn==32&&warps==1&&splits==16)launch<32,1,16>(ARGS);
  else if(bn==32&&warps==1&&splits==8)launch<32,1,8>(ARGS);
  else if(bn==32&&warps==1&&splits==4)launch<32,1,4>(ARGS);
  else if(bn==64&&warps==2&&splits==4)launch<64,2,4>(ARGS);
  else if(bn==32&&warps==1&&splits==1)launch_single<32,1>(ARGS);
  else if(bn==64&&warps==2&&splits==1)launch_single<64,2>(ARGS);
  else assert(!"unsupported reference profile");
  };
  submit(0);
  #undef ARGS
  if(splits>1){errors+=compare(pp,partial,n*splits,4,cases,"partial");partial_words+=n*splits;errors+=compare(lp,lse,(size_t)rows*24*splits,4,cases,"lse");lse_words+=(size_t)rows*24*splits;}
  errors+=compare(op,out,n,2,cases,"attention");out_words+=n;cases++;
  if(timing){
   cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
   cudaGraph_t graph;cudaGraphExec_t exec;cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
   check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));submit(stream);check(cudaStreamEndCapture(stream,&graph));
   size_t node_count=0;check(cudaGraphGetNodes(graph,NULL,&node_count));assert(node_count==(size_t)(splits>1?2:1));
   cudaGraphNode_t nodes[2];check(cudaGraphGetNodes(graph,nodes,&node_count));for(size_t j=0;j<node_count;j++){cudaGraphNodeType type;check(cudaGraphNodeGetType(nodes[j],&type));assert(type==cudaGraphNodeTypeKernel);}
   check(cudaGraphInstantiate(&exec,graph,NULL,NULL,0));
   for(int i=0;i<3;i++)check(cudaGraphLaunch(exec,stream));
   check(cudaEventRecord(start,stream));for(int i=0;i<100;i++)check(cudaGraphLaunch(exec,stream));check(cudaEventRecord(stop,stream));check(cudaEventSynchronize(stop));
   float ms;check(cudaEventElapsedTime(&ms,start,stop));printf("QSA_TIMING case%d rows%d bn%d warps%d splits%d execution_warps4 warm_graph_us%.6f iterations100 graph_kernel_nodes%zu graph_copy_nodes0 graph_allocation_nodes0\n",cases-1,rows,bn,warps,splits,ms*10,node_count);fflush(stdout);
   check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaEventDestroy(start));check(cudaEventDestroy(stop));check(cudaStreamDestroy(stream));
  }
  check(cudaFree(q));check(cudaFree(k));check(cudaFree(v));check(cudaFree(indices));check(cudaFree(table));check(cudaFree(requests));check(cudaFree(gate));check(cudaFree(partial));check(cudaFree(lse));check(cudaFree(out));
 }
 assert(cases>0&&feof(f));fclose(f);printf("QSA_PARTIAL cases%d partial_words%zu lse_words%zu out_words%zu differences%zu\n",cases,partial_words,lse_words,out_words,errors);return errors?1:0;
}
