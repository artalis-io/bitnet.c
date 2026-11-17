/* Actual Flash layer3 projection, pre-indexer, score and deterministic selection.
 * Link like test_cuda_backend; run FIRST_SHARD PRE_INDEXER_CAPTURES SELECTION_ROOT. */
#include "gpu_cuda.h"
#include "backend_session.h"
#include "model_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
static uint16_t bf16(float x){uint32_t u;memcpy(&u,&x,4);return(uint16_t)((u+0x7fff+((u>>16)&1))>>16);}
static BnGPUResidentView alloc(BnGPUBackend*g,void*r,size_t n,int zero){BnGPUResidentView v;assert(!g->resident_alloc(r,BN_GGUF_TENSOR_BF16,n,&v));if(zero)assert(!g->resident_zero(r,v));return v;}
static BnGPUResidentView slice(BnGPUResidentView v,size_t off,size_t n){return bn_gpu_resident_view_slice(v,off,n);}
static void*norm(BnGPUBackend*g,const float*x){uint16_t y[128];for(int i=0;i<128;i++)y[i]=bf16(x[i]-1.0f);void*w=g->buffer_create(g->ctx,y,sizeof(y),BN_GGUF_TENSOR_BF16,1,128);assert(w);return w;}
static size_t compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*prefix,const char*stage){
 char path[2048];snprintf(path,sizeof(path),"%s/%s-%s.bin",dir,prefix,stage);FILE*f=fopen(path,"rb");assert(f);
 uint16_t*a=malloc(v.count*2),*b=malloc(v.count*2);assert(a&&b&&fread(b,2,v.count,f)==v.count&&fgetc(f)==EOF);fclose(f);assert(!g->resident_read(r,v,a));
 size_t d=0;for(size_t i=0;i<v.count;i++)if(a[i]!=b[i]){if(d<3)fprintf(stderr,"%s %s word%zu %04x expected%04x\n",prefix,stage,i,a[i],b[i]);d++;}
 printf("%s %s words%zu differences%zu\n",prefix,stage,v.count,d);fflush(stdout);assert(!d);free(a);free(b);return v.count;
}
static void score_reference(BnGPUBackend*g,void*r,BnGPUResidentView v,int nt,int groups,int position,const char*root,const char*prefix,size_t*score_words,size_t*index_words,BnGPUResidentView packed){
 char path[2048];snprintf(path,sizeof(path),"%s/captures/%s-scores.bin",root,prefix);FILE*f=fopen(path,"rb");assert(f);assert(!fseek(f,0,SEEK_END));long bytes=ftell(f);assert(bytes>=0&&bytes%(nt*4)==0);int cols=(int)(bytes/(nt*4));rewind(f);
 float*gold=malloc((size_t)nt*cols*4),*actual=malloc(v.count*4);assert(gold&&actual&&fread(gold,4,(size_t)nt*cols,f)==(size_t)nt*cols);fclose(f);assert(!g->resident_read(r,v,actual));
 size_t d=0,visible_words=0;
 for(int row=0;row<nt;row++){int visible=(position+row+1)/4;visible_words+=visible;assert(visible<=cols&&visible<=groups);for(int j=0;j<groups;j++){
  uint32_t a,b;memcpy(&a,actual+(size_t)row*groups+j,4);if(j<visible)memcpy(&b,gold+(size_t)row*cols+j,4);else b=UINT32_C(0xff800000);
  if(a!=b){if(d<3)fprintf(stderr,"%s score row%d col%d %08x expected%08x\n",prefix,row,j,a,b);d++;}
 }}
 printf("%s score visible%zu differences%zu\n",prefix,visible_words,d);fflush(stdout);assert(!d);free(actual);free(gold);*score_words+=visible_words;
 snprintf(path,sizeof(path),"%s/canonical/%s-canonical.bin",root,prefix);f=fopen(path,"rb");assert(f);size_t n=(size_t)nt*2052;int32_t*a=malloc(n*4),*b=malloc(n*4);assert(a&&b&&fread(b,4,n,f)==n&&fgetc(f)==EOF);fclose(f);assert(!g->resident_read(r,packed,a));d=0;
 for(size_t i=0;i<n;i++)if(a[i]!=b[i]){if(d<3)fprintf(stderr,"%s packed %zu %d expected%d\n",prefix,i,a[i],b[i]);d++;}
 printf("%s packed words%zu differences%zu\n",prefix,n,d);fflush(stdout);assert(!d);free(a);free(b);*index_words+=n;
}
typedef struct{float score;int index;} Rank;
static int rank_compare(const void*aa,const void*bb){const Rank*a=aa,*b=bb;if(a->score>b->score)return-1;if(a->score<b->score)return 1;return(a->index>b->index)-(a->index<b->index);}
static int int_compare(const void*aa,const void*bb){int a=*(const int*)aa,b=*(const int*)bb;return(a>b)-(a<b);}
static void synthetic_selection(BnGPUBackend*g){
 int lengths[]={0,1,511,512,513,1024,1025,8192,8193,32769,65536};size_t checked=0;
 BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);
 for(unsigned c=0;c<sizeof(lengths)/sizeof(lengths[0]);c++)for(int pattern=0;pattern<4;pattern++){
  int n=lengths[c],cap=n?n:1,position=n?n*4-1:0;float*values=malloc((size_t)cap*4);Rank*ranks=malloc((size_t)cap*sizeof(Rank));assert(values&&ranks);
  for(int i=0;i<cap;i++){values[i]=pattern==3?(i%3==0?-0.0f:(float)(i%37)):(pattern==2?(float)i:(pattern?(float)((i*17+i/13)%37)/4.f:0));ranks[i]=(Rank){values[i],i};}
  qsort(ranks,n,sizeof(Rank),rank_compare);int chosen[512],count=n<512?n:512;for(int i=0;i<count;i++)chosen[i]=ranks[i].index;qsort(chosen,count,sizeof(int),int_compare);
  int32_t expected[2052],actual[2052];for(int i=0;i<2052;i++)expected[i]=-1;for(int i=0;i<count;i++)for(int j=0;j<4;j++)expected[i*4+j]=chosen[i]*4+j;
  int end=position+1,tail=end%4;for(int j=0;j<tail;j++)expected[count*4+j]=n*4+j;expected[2051]=count*4+tail;
  BnGPUResidentView scores,out;assert(!g->resident_alloc(r,BN_GGUF_TENSOR_F32,cap,&scores));assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,2052,&out));assert(!g->resident_write(r,scores,values));
  BnGPUQSASelectResidentPlan p={1,position,cap,scores,out};
  for(int repeat=0;repeat<3;repeat++){assert(!g->qsa_select_resident(g->ctx,r,&p));assert(!g->resident_read(r,out,actual));assert(!memcmp(actual,expected,sizeof(actual)));checked+=2052;}
  free(values);free(ranks);
 }
 bn_backend_session_free(s);printf("QSA_SELECTION_SYNTHETIC_PASSED words%zu ties, stable rank, max65536groups\n",checked);
}
static void synthetic_scoring(BnGPUBackend*g){
 int sizes[]={1,15,16,17,63,64,65,1025,65536};size_t checked=0;
 BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);
 for(unsigned c=0;c<sizeof(sizes)/sizeof(sizes[0]);c++){
  int groups=sizes[c],tokens=3,position=groups*4-3;uint16_t*q=malloc(3u*512*2),*k=malloc((size_t)groups*128*2);float*out=malloc((size_t)groups*3*4);assert(q&&k&&out);
  const float heads[]={-.5f,.25f,-.125f,.75f};for(int row=0;row<3;row++)for(int h=0;h<4;h++)for(int d=0;d<128;d++)q[row*512+h*128+d]=bf16(heads[h]*(row+1)/4.f);
  for(int col=0;col<groups;col++)for(int d=0;d<128;d++)k[(size_t)col*128+d]=bf16((float)((col*7+d*3)%17-8)/8.f);
  BnGPUResidentView qv,kv,sv;assert(!g->resident_alloc(r,BN_GGUF_TENSOR_BF16,3u*512,&qv));assert(!g->resident_alloc(r,BN_GGUF_TENSOR_BF16,(size_t)groups*128,&kv));assert(!g->resident_alloc(r,BN_GGUF_TENSOR_F32,(size_t)groups*3,&sv));assert(!g->resident_write(r,qv,q));assert(!g->resident_write(r,kv,k));
  BnGPUQSAScoreResidentPlan p={tokens,position,groups,qv,kv,sv};assert(!g->qsa_score_bf16_resident(g->ctx,r,&p));assert(!g->resident_read(r,sv,out));
  for(int row=0;row<3;row++)for(int col=0;col<groups;col++){
   float expected=-INFINITY;
   if(col<(position+row+1)/4){float sum=0;for(int d=0;d<128;d++)sum+=(float)((col*7+d*3)%17-8)/8.f;float v[4];for(int h=0;h<4;h++)v[h]=fmaxf(sum*heads[h]*(row+1)/4.f,0);expected=(v[0]+v[1])+(v[2]+v[3]);}
   uint32_t a,b;memcpy(&a,out+(size_t)row*groups+col,4);memcpy(&b,&expected,4);assert(a==b);checked++;
  }
  free(q);free(k);free(out);
 }
 bn_backend_session_free(s);printf("QSA_SCORE_SYNTHETIC_PASSED words%zu boundary tiles and max65536groups\n",checked);
}
int main(int argc,char**argv){
 assert(argc==4);BnGGUFFile*file=bn_gguf_open_file(argv[1]);assert(file);BnModel m={0};assert(!bn_model_load_with_cpu_preparation(&m,file,8192,0,0,0));
 assert(m.config.dim==2560&&m.config.indexer_head_count==4&&m.config.indexer_head_size==128&&m.config.attention_compress_ratios[3]==4);
 BnAttentionIndexerWeights*w=&m.weights.layers[3].indexer;
 assert(w->q.type==BN_GGUF_TENSOR_BF16&&w->k.type==BN_GGUF_TENSOR_BF16&&w->q.rows==512&&w->k.rows==128&&w->q.cols==2560&&w->k.cols==2560);
 BnGPUBackend*g=bn_gpu_cuda_create();assert(g&&g->qsa_indexer_bf16_resident&&g->qsa_score_bf16_resident&&g->qsa_select_resident);synthetic_selection(g);synthetic_scoring(g);
 size_t bytes=640u*2560*2;unsigned char*stack=malloc(bytes);assert(stack);memcpy(stack,w->q.data,512u*2560*2);memcpy(stack+512u*2560*2,w->k.data,128u*2560*2);
 void*projection=g->buffer_create(g->ctx,stack,bytes,BN_GGUF_TENSOR_BF16,640,2560);assert(projection);free(stack);
 void*qw=norm(g,w->q_norm),*kw=norm(g,w->k_norm);BnGPURotaryCachePlan rp={8192,64,10000000.f};void*cs=g->buffer_create_rotary_bf16(g->ctx,&rp);assert(cs);
 const int nts[]={1,3,4,5,17,129,129,129,129,2053,2053,4097},chunks[]={1,3,4,5,7,129,64,7,1,2053,257,4097};size_t words=0,score_words=0,index_words=0;int phases=0;
 for(int ci=0;ci<12;ci++){
  int nt=nts[ci],cap=chunks[ci];BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);assert(r);assert(!g->linear_bf16_reserve(g->ctx,r,0));
  BnGPUResidentView input=alloc(g,r,(size_t)cap*2560,0),merged=alloc(g,r,(size_t)cap*640,0),q=alloc(g,r,(size_t)cap*512,0);
  BnGPUResidentView history[2]={alloc(g,r,512,1),alloc(g,r,512,0)},compressed=alloc(g,r,(size_t)((nt+8)/4)*128,1);
  int cp_cap=(nt+8)/4;BnGPUResidentView score_buf,packed_buf;
  assert(!g->resident_alloc(r,BN_GGUF_TENSOR_F32,(size_t)cap*cp_cap,&score_buf));assert(!g->resident_alloc(r,BN_GGUF_TENSOR_I32,(size_t)cap*2052,&packed_buf));
  int pos=0,phase=0,parity=0;
  while(pos<nt+8){
   int take=pos<nt?(nt-pos<cap?nt-pos:cap):1;size_t n=(size_t)take;
   uint16_t*x=malloc(n*2560*2);assert(x);for(size_t i=0;i<n*2560;i++){size_t j=(size_t)pos*2560+i;x[i]=bf16((float)((int)((j*17+j/37)%509)-254)/128);}
   assert(!g->resident_write(r,slice(input,0,n*2560),x));free(x);
   BnGPUBF16LinearResidentPlan lp={0};lp.n_tokens=take;lp.rows=640;lp.cols=2560;lp.weight.type=BN_GGUF_TENSOR_BF16;lp.weight.decoded=projection;lp.x=slice(input,0,n*2560);lp.out=slice(merged,0,n*640);assert(!g->linear_bf16_resident(g->ctx,r,&lp));
   BnGPUQSAIndexerResidentPlan p={0};p.n_tokens=take;p.position=pos;p.eps=1e-6f;p.q_weight=qw;p.k_weight=kw;p.cos_sin=cs;p.projected=lp.out;p.initial_history=history[parity];p.next_history=history[1-parity];p.q=slice(q,0,n*512);
   int groups=(pos+take)/4-pos/4;if(groups)p.compressed=slice(compressed,(size_t)(pos/4)*128,(size_t)groups*128);
   BnGPUQSAIndexerResidentPlan bad=p;bad.next_history=p.initial_history;assert(g->qsa_indexer_bf16_resident(g->ctx,r,&bad)==-1);
   bad=p;bad.position=INT_MAX;assert(g->qsa_indexer_bf16_resident(g->ctx,r,&bad)==-1);
   bad=p;bad.eps=NAN;assert(g->qsa_indexer_bf16_resident(g->ctx,r,&bad)==-1);
   bad=p;bad.q_weight=projection;assert(g->qsa_indexer_bf16_resident(g->ctx,r,&bad)==-1);
   bad=p;if(groups)bad.compressed=(BnGPUResidentView){0};else bad.compressed=slice(compressed,0,128);assert(g->qsa_indexer_bf16_resident(g->ctx,r,&bad)==-1);
   assert(!g->qsa_indexer_bf16_resident(g->ctx,r,&p));
   int selected=(ci==0||ci==5||ci>=9)&&(phase==0||pos+take==nt+8);
   BnGPUQSAScoreResidentPlan sp={take,pos,cp_cap,p.q,compressed,slice(score_buf,0,n*cp_cap)};
   BnGPUQSASelectResidentPlan tp={take,pos,cp_cap,sp.scores,slice(packed_buf,0,n*2052)};
   if(selected){
    BnGPUQSAScoreResidentPlan sbad=sp;sbad.cache_groups=0;assert(g->qsa_score_bf16_resident(g->ctx,r,&sbad)==-1);
    sbad=sp;sbad.q=sp.compressed;assert(g->qsa_score_bf16_resident(g->ctx,r,&sbad)==-1);
    BnGPUQSASelectResidentPlan tbad=tp;tbad.position=INT_MAX;assert(g->qsa_select_resident(g->ctx,r,&tbad)==-1);
    tbad=tp;tbad.out.count--;assert(g->qsa_select_resident(g->ctx,r,&tbad)==-1);
    assert(!g->qsa_score_bf16_resident(g->ctx,r,&sp));assert(!g->qsa_select_resident(g->ctx,r,&tp));
   }
   assert(!g->resident_synchronize(r));parity=1-parity;
   char prefix[64];snprintf(prefix,sizeof(prefix),"%d-%d-%d",nt,cap,phase);
   words+=compare(g,r,lp.x,argv[2],prefix,"input");words+=compare(g,r,lp.out,argv[2],prefix,"projected");words+=compare(g,r,p.q,argv[2],prefix,"q");words+=compare(g,r,history[parity],argv[2],prefix,"history");
   int visible=(pos+take)/4;if(visible)words+=compare(g,r,slice(compressed,0,(size_t)visible*128),argv[2],prefix,"compressed");
   if(selected)score_reference(g,r,sp.scores,take,cp_cap,pos,argv[3],prefix,&score_words,&index_words,tp.out);
   pos+=take;phase++;phases++;
  }
  bn_backend_session_free(s);
 }
 g->buffer_destroy(g->ctx,cs);g->buffer_destroy(g->ctx,kw);g->buffer_destroy(g->ctx,qw);g->buffer_destroy(g->ctx,projection);bn_model_free(&m);bn_gpu_cuda_destroy(g);bn_gguf_free(file);
 printf("QSA_SELECTION_PASSED bf16_words%zu phases%d score_words%zu index_words%zu\n",words,phases,score_words,index_words);return 0;
}
