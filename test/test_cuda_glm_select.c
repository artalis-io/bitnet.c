#define main glm_indexer_parent_main
#include "test_cuda_glm_indexer.c"
#undef main
#include <float.h>
typedef struct {float score;int index;} ScoreEntry;
static int rank_compare(const void *a,const void *b){const ScoreEntry*x=a,*y=b;if(x->score>y->score)return -1;if(x->score<y->score)return 1;return (x->index>y->index)-(x->index<y->index);}
static int index_compare(const void *a,const void *b){int x=*(const int*)a,y=*(const int*)b;return (x>y)-(x<y);}
static float selection_input(size_t i,int kind){
    if(kind==1)return -(float)(i%1024+1);
    if(kind==2){uint32_t bits=(uint32_t)(i%3)|((i%2)?0x80000000u:0);float x;memcpy(&x,&bits,4);return x;}
    if(kind==3)return i%2?FLT_MAX:-FLT_MAX;
    return (float)((int)((i*37+i/13)%8191)-4095)/128;
}
static size_t selection_case(BnGPUBackend*g,int nt,int pos,int kind,int short_mode,int invalid){
    assert(nt>0&&nt<=16384&&pos>=0&&pos<=INT_MAX-nt);
    size_t n=(size_t)nt,k=(size_t)(nt+pos)/4;float *scores=malloc((n*k+1)*4);int32_t *pools=malloc(n*512*4),*out=malloc(n*2051*4),*ep=malloc(n*512*4),*eo=malloc(n*2051*4);ScoreEntry*entries=malloc((k+1)*sizeof(*entries));assert(scores&&pools&&out&&ep&&eo&&entries);
    for(size_t i=0;i<n*k;++i)scores[i]=selection_input(i,kind);
    /* Future NaNs must never participate; nonfinite visible scores fail closed. */
    for(int row=0;row<nt;++row)for(size_t i=(size_t)(pos+row+1)/4;i<k;++i)scores[(size_t)row*k+i]=NAN;
    if(invalid&&k)scores[0]=invalid==1?NAN:INFINITY;
    for(size_t i=0;i<n*512;++i)ep[i]=-1;
    for(size_t i=0;i<n*2051;++i)eo[i]=-1;
    int32_t status=0,expected_status=0;
    for(int row=0;row<nt;++row){int end=pos+row+1,visible=end/4,take=visible<512?visible:512,failed=0;
        if(!short_mode)for(int j=0;j<visible;++j){float v=scores[(size_t)row*k+j];entries[j]=(ScoreEntry){v,j};if(!isfinite(v))failed=1;}
        if(failed){expected_status=1;continue;}
        if(short_mode){for(int j=0;j<take;++j)ep[(size_t)row*512+j]=j;for(int j=0;j<end;++j)eo[(size_t)row*2051+j]=j;continue;}
        qsort(entries,visible,sizeof(*entries),rank_compare);for(int j=0;j<take;++j)ep[(size_t)row*512+j]=entries[j].index;
        qsort(ep+(size_t)row*512,take,4,index_compare);
        for(int j=0;j<take;++j)for(int a=0;a<4;++a)eo[(size_t)row*2051+j*4+a]=ep[(size_t)row*512+j]*4+a;
        for(int j=0;j<(end&3);++j)eo[(size_t)row*2051+2048+j]=visible*4+j;
    }
    BnBackendSession*s=bn_backend_session_create(),*other=bn_backend_session_create();assert(s&&other);void*r=bn_gpu_backend_resident_request(g,s),*peer=bn_gpu_backend_resident_request(g,other);assert(r&&peer);
    BnGPUIndexerSelectResidentPlan p={.n_tokens=nt,.position=pos,.n_keys=(int)k,.short_prefill=short_mode,.scores=alloc(g,r,BN_GGUF_TENSOR_F32,short_mode?0:n*k,scores),.pools=alloc(g,r,BN_GGUF_TENSOR_I32,n*512,NULL),.out=alloc(g,r,BN_GGUF_TENSOR_I32,n*2051,NULL),.status=alloc(g,r,BN_GGUF_TENSOR_I32,1,&status)};
    assert(!g->indexer_select_resident(g->ctx,r,&p));assert(!g->resident_read(r,p.pools,pools)&&!g->resident_read(r,p.out,out)&&!g->resident_read(r,p.status,&status));assert(status==expected_status);assert(!memcmp(pools,ep,n*512*4)&&!memcmp(out,eo,n*2051*4));
    BnGPUIndexerSelectResidentPlan bad;
#define REJECT(change) do{bad=p;change;assert(g->indexer_select_resident(g->ctx,r,&bad)==-1);}while(0)
    REJECT(bad.n_tokens=0);REJECT(bad.position=INT_MAX);REJECT(bad.position=-1);REJECT(bad.n_keys++);REJECT(bad.short_prefill=2);
    REJECT(bad.out.count--);REJECT(bad.pools.request=peer);REJECT(bad.status=p.pools);REJECT(bad.out.slot=p.pools.slot);REJECT(bad.pools.type=BN_GGUF_TENSOR_F32);REJECT(bad.out.offset=SIZE_MAX);
    if(!short_mode&&k)REJECT(bad.scores.count--);
    assert(!g->resident_read(r,p.out,out)&&!memcmp(out,eo,n*2051*4));
#undef REJECT
    status=1;assert(!g->resident_write(r,p.status,&status));assert(!g->indexer_select_resident(g->ctx,r,&p));assert(!g->resident_read(r,p.status,&status)&&status==1);
    bn_backend_session_free(other);bn_backend_session_free(s);free(scores);free(pools);free(out);free(ep);free(eo);free(entries);return n*(512+2051);
}
int main(void){BnGPUBackend*g=bn_gpu_cuda_create();assert(g&&g->indexer_select_resident);size_t count=0;int cases[][2]={{1,0},{2,1},{17,3},{65,0},{17,2047},{65,4095},{17,262143}};for(size_t i=0;i<sizeof(cases)/sizeof(cases[0]);++i)for(int kind=0;kind<4;++kind){int nt=cases[i][0],pos=cases[i][1];count+=selection_case(g,nt,pos,kind,0,0);if(nt+pos<=2048)count+=selection_case(g,nt,pos,kind,1,0);}count+=selection_case(g,2,4095,0,0,1);count+=selection_case(g,2,4095,0,0,2);bn_gpu_cuda_destroy(g);printf("GLM signed pool selection/expansion PASSED: %zu exact I32 values\n",count);return 0;}
