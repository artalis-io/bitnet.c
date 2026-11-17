#define main preindexer_existing_tests_main
#include "test_cuda_glm_preindexer.c"
#undef main
static int selection_i32_compare(const void *a,const void *b){int32_t x=*(const int32_t*)a,y=*(const int32_t*)b;return (x>y)-(x<y);}
static size_t selection_check(BnGPUBackend*g,void*r,const BnTransformerBF16IndexerDecodePlan*p,const char*dir){
    int nt=p->indexer.n_tokens,pos=p->indexer.position;size_t n=(size_t)nt;
    int32_t *pools=malloc(n*512*4),*out=malloc(n*2051*4),*expected=load(dir,"expanded",n*2051*4),*ep=NULL;assert(pools&&out);
    assert(!g->resident_read(r,p->pools,pools)&&!g->resident_read(r,p->out,out));if(!p->short_decode)ep=load(dir,"pools",n*512*4);
    for(int row=0;row<nt;++row){int end=pos+row+1,visible=end/4,take=visible<512?visible:512;int32_t*a=pools+(size_t)row*512,*o=out+(size_t)row*2051,*e=expected+(size_t)row*2051;
        for(int j=0;j<512;++j){if(j<take){assert(a[j]>=0&&a[j]<visible&&(j==0||a[j]>a[j-1]));}else assert(a[j]==-1);}
        if(p->short_decode){for(int j=0;j<take;++j)assert(a[j]==j);assert(!memcmp(o,e,2051*4));}
        else {for(int j=0;j<512;++j)for(int t=0;t<4;++t)assert(o[j*4+t]==(a[j]<0?-1:a[j]*4+t));for(int j=0;j<3;++j)assert(o[2048+j]==(j<(end&3)?visible*4+j:-1));
            int32_t *pe=ep+(size_t)row*512;qsort(pe,512,4,selection_i32_compare);qsort(a,512,4,selection_i32_compare);assert(!memcmp(a,pe,512*4));
            qsort(o,2051,4,selection_i32_compare);qsort(e,2051,4,selection_i32_compare);assert(!memcmp(o,e,2051*4));}
    }
    free(pools);free(out);free(expected);free(ep);return n*(512+2051);
}
int main(int argc,char **argv){assert(argc==6);BnGGUFFile *f=bn_gguf_open_file(argv[1]);assert(f);BnModelAnatomy *a=malloc(sizeof(*a));assert(a);assert(!bn_model_arch_plan_anatomy(f,a));BnGPUBackend *g=bn_gpu_cuda_create();assert(g);BnBackendBF16IndexerWeights w={0};assert(!bn_backend_layout_bf16_indexer_init(&w,g,f,a,3,8326144));g->indexer_fp8_gather_resident=NULL;g->indexer_fp8_score_resident=NULL;
 char schedule_path[1024];snprintf(schedule_path,sizeof(schedule_path),"%s/schedules.txt",argv[4]);FILE *schedule=fopen(schedule_path,"r");assert(schedule);int cases;assert(fscanf(schedule,"%d",&cases)==1&&cases>0&&cases<=128);uint16_t zero[1024]={0};size_t words=0,score_words=0,selection_words=0;
 for(int case_index=0;case_index<cases;++case_index){int case_id,ns,steps[64],maxnt=1,total=0;assert(fscanf(schedule,"%d%d",&case_id,&ns)==2&&case_id>=0&&ns>0&&ns<=64);for(int j=0;j<ns;++j){assert(fscanf(schedule,"%d",&steps[j])==1&&steps[j]>0&&steps[j]<=16384);total+=steps[j];if(steps[j]>maxnt)maxnt=steps[j];}
  int blocks=((total/4)+63)/64+2;size_t cache_bytes=(size_t)blocks*8448;unsigned char *sentinel=malloc(cache_bytes);int32_t *table=malloc((size_t)blocks*4);assert(sentinel&&table);memset(sentinel,0xa5,cache_bytes);for(int j=0;j<blocks;++j)table[j]=blocks-1-j;
  BnBackendSession *s=bn_backend_session_create();assert(s);void *r=bn_gpu_backend_resident_request(g,s);assert(r);assert(!g->linear_bf16_reserve(g->ctx,r,(size_t)maxnt*w.dim));BnGPUResidentView tail=alloc(g,r,BN_GGUF_TENSOR_BF16,1024,zero),next=alloc(g,r,BN_GGUF_TENSOR_BF16,1024,NULL),cache=alloc(g,r,BN_GGUF_TENSOR_I8,cache_bytes,sentinel),tv=alloc(g,r,BN_GGUF_TENSOR_I32,blocks,table);int32_t status=0;BnGPUResidentView statusv=alloc(g,r,BN_GGUF_TENSOR_I32,1,&status);int pos=0;
  for(int phase=0;phase<ns;++phase){int nt=steps[phase];size_t n=nt,visible=(pos+nt)/4,groups=visible-pos/4;char dir[1024],cd[1024];snprintf(dir,sizeof(dir),"%s/%d/%d",argv[2],case_id,phase);snprintf(cd,sizeof(cd),"%s/%d/%d",argv[3],case_id,phase);void *x=load(dir,"x",n*w.dim*2),*qr=load(dir,"qr",n*w.q_rank*2);size_t counts[]={n*w.dim,n*w.q_rank,1024,1024,n*4096,n*128,n*128,n*128,n*w.dim,n*32,n*4096,n*32,n*32,groups*128,groups};BnGPUResidentView v[15];
   for(int j=0;j<15;++j){int type=j==8||j==9||j==11||j==12||j==14?BN_GGUF_TENSOR_F32:j==10||j==13?BN_GGUF_TENSOR_I8:BN_GGUF_TENSOR_BF16;v[j]=j==2?tail:j==3?next:alloc(g,r,type,counts[j],j==0?x:j==1?qr:NULL);}
   int32_t *slots=malloc((groups?groups:1)*4);assert(slots);for(size_t j=0;j<groups;++j){size_t logical=(size_t)pos/4+j;slots[j]=table[logical/64]*64+(int32_t)(logical%64);}
   BnTransformerBF16IndexerDecodePlan p={.indexer={.weights=&w,.layer=3,.n_tokens=nt,.position=pos,.x=v[0],.qr=v[1],.initial_tail=v[2],.next_tail=v[3],.q=v[4],.key_raw=v[5],.key=v[6],.gate=v[7],.cast=v[8],.head_raw=v[9],.q_fp8=v[10],.q_scales=v[11],.head_weights=v[12],.compressed=v[13],.compressed_scales=v[14]},.num_cache_blocks=blocks,.logical_blocks=blocks,.cache=cache,.block_table=tv,.slots=alloc(g,r,BN_GGUF_TENSOR_I32,groups,slots),.status=statusv};
   int short_mode=pos+nt<=2048;p.short_decode=short_mode;p.scores=alloc(g,r,BN_GGUF_TENSOR_F32,short_mode?0:n*visible,NULL);p.pools=alloc(g,r,BN_GGUF_TENSOR_I32,n*512,NULL);p.out=alloc(g,r,BN_GGUF_TENSOR_I32,n*2051,NULL);
   BnTransformerBF16IndexerDecodePlan bad=p;bad.out.count--;assert(bn_transformer_bf16_indexer_decode(r,&bad)==-1);assert(!bn_transformer_bf16_indexer_decode(r,&p));char ref_dir[1024];snprintf(ref_dir,sizeof(ref_dir),"%s/%d/%d",argv[5],case_id,phase);selection_words+=selection_check(g,r,&p,ref_dir);
   char sd[1024];snprintf(sd,sizeof(sd),"%s/%d/%d",argv[4],case_id,phase);if(!short_mode)score_words+=check(g,r,p.scores,sd,"scores");assert(!g->resident_read(r,statusv,&status)&&!status);
   int indexes[]={4,5,6,7,9,10,11,12,13,14,3};const char *names[]={"q","key_raw","key","gate","head_raw","q_fp8","q_scales","head_weights","compressed","compressed_scales","next_tail"};for(int j=0;j<11;++j)words+=check(g,r,v[indexes[j]],dir,names[j]);words+=check(g,r,cache,cd,"cache");check(g,r,tail,dir,"initial_tail");BnGPUResidentView swap=tail;tail=next;next=swap;pos+=nt;free(slots);free(x);free(qr);
  }bn_backend_session_free(s);free(sentinel);free(table);printf("case%d native paged indexer chain PASSED tokens%d\n",case_id,pos);fflush(stdout);
 }assert(fscanf(schedule," %d",&cases)==EOF);fclose(schedule);printf("PASSED %zu exact pre-indexer/cache elements and %zu exact causal score elements, %zu selection I32 values checked\n",words,score_words,selection_words);bn_backend_layout_bf16_indexer_free(&w);bn_gpu_cuda_destroy(g);bn_gguf_free(f);free(a);return 0;}
