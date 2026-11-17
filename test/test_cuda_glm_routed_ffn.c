#include "gpu_cuda.h"
#include "backend_session.h"
#include "gguf.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
static void*load(const char*d,const char*n,size_t bytes){char p[1024];snprintf(p,sizeof(p),"%s/%s.bin",d,n);FILE*f=fopen(p,"rb");assert(f);void*b=malloc(bytes);assert(b);assert(fread(b,1,bytes,f)==bytes&&fgetc(f)==EOF);fclose(f);return b;}
static BnGPUResidentView allocv(BnGPUBackend*g,void*r,int type,size_t n,const void*b){BnGPUResidentView v={0};assert(!g->resident_alloc(r,type,n,&v));if(b)assert(!g->resident_write(r,v,b));return v;}
static size_t cmp(const uint16_t*a,const uint16_t*b,size_t n,const char*name){for(size_t i=0;i<n;++i)if(a[i]!=b[i]){fprintf(stderr,"%s word%zu got%04x expected%04x\n",name,i,a[i],b[i]);exit(3);}return n;}
int main(int argc,char**argv){assert(argc==3);BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnGPUBackend*g=bn_gpu_cuda_create();assert(g);
    enum{D=4096,H=2048,E=288,K=8};size_t gb=(size_t)D*H/256*144,db=(size_t)D*H/256*176,stride=2*gb+db,maxstride=(size_t)D*H/256*176*2+(size_t)D*H/32*34,words=0;
    const unsigned char*raw[3];const char*names[]={"gate","up","down"};for(int i=0;i<3;++i){char name[128];snprintf(name,sizeof(name),"blk.3.ffn_%s_exps.weight",names[i]);int ti=bn_gguf_find_tensor(f,name);assert(ti>=0);BnGGUFTensorInfo*t=&f->tensors[ti];assert(t->type==(i==2?BN_GGUF_TENSOR_Q5_K:BN_GGUF_TENSOR_Q4_K)&&t->n_dims==3&&t->dims[2]==E&&t->dims[0]==(i==2?H:D)&&t->dims[1]==(i==2?D:H));raw[i]=bn_gguf_tensor_data(f,ti);assert(raw[i]);}
    int tokens[]={1,2,9,64,65,129},capacities[]={1,3,8};
    for(int ci=0;ci<6;++ci){int nt=tokens[ci];size_t n=(size_t)nt*K;char dir[1024];snprintf(dir,sizeof(dir),"%s/%d",argv[2],ci);
        void*x=load(dir,"x",(size_t)nt*D*2),*ids=load(dir,"ids",n*4),*weights=load(dir,"weights",n*4);uint16_t*refgu=load(dir,"gate_up",n*H*4),*refact=load(dir,"activated",n*H*2),*refdown=load(dir,"down",n*D*2),*refweighted=load(dir,"weighted",n*D*2),*refout=load(dir,"out",(size_t)nt*D*2);
        for(int grouped=0;grouped<2;++grouped)for(int capi=0;capi<3;++capi){int cap=capacities[capi];BnBackendSession*session=bn_backend_session_create();assert(session);void*r=bn_gpu_backend_resident_request(g,session);assert(r);
            BnGPUMoEBankConfig cfg={.slots=2,.experts_per_slot=(unsigned)cap,.dim=D,.hidden_dim=H,.host_budget_bytes=2*cap*maxstride,.device_budget_bytes=2*cap*maxstride,.n_experts=E,.k=K};assert(!g->moe_bank_reserve(g->ctx,r,&cfg));assert(!g->linear_a8_bf16_reserve(g->ctx,r,(size_t)nt*D));
            size_t input=(size_t)nt*(D/32)*36*(K+1),meta=2u*cap*sizeof(BnGPUMoEBankItem),down=n*(H/32)*36;assert(!g->moe_bank_input_reserve(g->ctx,r,nt,input));assert(!g->moe_bank_grouped_reserve(g->ctx,r,nt,meta,down+meta));
            BnGPUResidentView xv=allocv(g,r,BN_GGUF_TENSOR_BF16,(size_t)nt*D,x),iv=allocv(g,r,BN_GGUF_TENSOR_I32,n,ids),wv=allocv(g,r,BN_GGUF_TENSOR_F32,n,weights),mv=allocv(g,r,BN_GGUF_TENSOR_I32,296,NULL);
            BnGPUMoEGroupBF16ResidentPlan group={.n_tokens=nt,.n_experts=E,.k=K,.dim=D,.indices=iv,.x=xv,.offsets=bn_gpu_resident_view_slice(mv,0,E+1),.status=bn_gpu_resident_view_slice(mv,292,1),.cursors=allocv(g,r,BN_GGUF_TENSOR_I32,E,NULL),.slots=allocv(g,r,BN_GGUF_TENSOR_I32,n,NULL),.inverse=allocv(g,r,BN_GGUF_TENSOR_I32,n,NULL),.gathered=allocv(g,r,BN_GGUF_TENSOR_BF16,n*D,NULL)};
            assert(!g->moe_group_bf16_resident(g->ctx,r,&group));int offsets[296];assert(!g->resident_read(r,mv,offsets));assert(!offsets[0]&&offsets[E]==(int)n&&!offsets[292]);uint64_t gen=0;BnGPUMoEBankInputPlan prep={nt,xv,group.slots,group.status};assert(!g->moe_bank_prepare_input(g->ctx,r,&prep,&gen));
            BnGPUResidentView gate=allocv(g,r,BN_GGUF_TENSOR_BF16,n*H,NULL),up=allocv(g,r,BN_GGUF_TENSOR_BF16,n*H,NULL),out=allocv(g,r,BN_GGUF_TENSOR_BF16,n*D,NULL),weighted=allocv(g,r,BN_GGUF_TENSOR_BF16,n*D,NULL),result=allocv(g,r,BN_GGUF_TENSOR_BF16,(size_t)nt*D,NULL);
            int e=0,bank=0,active=0;while(e<E){while(e<E&&offsets[e]==offsets[e+1])e++;if(e==E)break;BnGPUMoEBankLease lease={0};int status;while((status=g->moe_bank_acquire(r,bank%2,BN_GGUF_TENSOR_Q4_K,BN_GGUF_TENSOR_Q5_K,&lease))==BN_GPU_MOE_BANK_BUSY)thrd_yield();assert(!status&&lease.expert_bytes==stride);BnGPUMoEBankItem items[64];int count=0;
                while(e<E&&count<cap){int rows=offsets[e+1]-offsets[e];if(rows){items[count]=(BnGPUMoEBankItem){e,offsets[e],rows};unsigned char*dst=(unsigned char*)lease.data+(size_t)count*stride;for(int role=0;role<3;++role)memcpy(dst+(role==2?2*gb:role*gb),raw[role]+(size_t)e*(role==2?db:gb),role==2?db:gb);count++;active++;}e++;}
                BnGPUMoEBankResidentPlan plan={.lease=lease,.total_tokens=nt,.n_experts=count,.experts=items,.x=group.gathered,.gate=gate,.up=up,.out=out,.prepared_input=gen,.grouped=grouped,.clamped=1,.clamp_limit=10};assert(!g->moe_bank_submit(g->ctx,r,&plan));bank++;
            }
            BnGPUMoEReduceBF16ResidentPlan red={.n_tokens=nt,.k=K,.dim=D,.grouped=out,.inverse=group.inverse,.weights=wv,.weighted=weighted,.out=result,.status=group.status};assert(!g->moe_reduce_bf16_resident(g->ctx,r,&red));assert(!g->resident_synchronize(r));
            int32_t*inverse=malloc(n*4);uint16_t*ga=malloc(n*H*2),*ua=malloc(n*H*2),*oa=malloc(n*D*2),*wa=malloc(n*D*2),*ya=malloc((size_t)nt*D*2);assert(inverse&&ga&&ua&&oa&&wa&&ya);assert(!g->resident_read(r,group.inverse,inverse));assert(!g->resident_read(r,gate,ga));assert(!g->resident_read(r,up,ua));assert(!g->resident_read(r,out,oa));assert(!g->resident_read(r,weighted,wa));assert(!g->resident_read(r,result,ya));
            for(size_t slot=0;slot<n;++slot){assert(inverse[slot]>=0&&(size_t)inverse[slot]<n);size_t row=(size_t)inverse[slot];words+=cmp(ga+row*H,refact+slot*H,H,"activation");words+=cmp(ua+row*H,refgu+slot*H*2+H,H,"up");words+=cmp(oa+row*D,refdown+slot*D,D,"down");}words+=cmp(wa,refweighted,n*D,"weighted");words+=cmp(ya,refout,(size_t)nt*D,"sum");BnGPUMoEBankStats stats={0};assert(!g->moe_bank_stats(r,&stats)&&stats.upload_bytes==(size_t)active*stride&&stats.reserved_host_bytes==cfg.host_budget_bytes&&stats.reserved_device_bytes==cfg.device_budget_bytes);printf("case%d T%d capacity%d grouped%d PASSED, %zu upload bytes\n",ci,nt,cap,grouped,stats.upload_bytes);fflush(stdout);
            free(inverse);free(ga);free(ua);free(oa);free(wa);free(ya);bn_backend_session_free(session);
        }free(x);free(ids);free(weights);free(refgu);free(refact);free(refdown);free(refweighted);free(refout);
    }printf("PASSED %zu exact routed-execution words\n",words);bn_gpu_cuda_destroy(g);bn_gguf_free(f);return 0;
}
