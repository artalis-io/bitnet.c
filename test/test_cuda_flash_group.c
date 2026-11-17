/* Actual-checkpoint cohort failure/recovery. Input files contain raw I32 IDs;
 * no expected logits enter inference. Separate verification checks output. */
#include "model_internal.h"
#include "session_internal.h"
#include "gpu_cuda.h"
#include "moe.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NORMAL, BUSY_FOREVER, SOURCE_FAIL, SUBMIT_FAIL, HEAD_FAIL, FENCE_FAIL, BUSY_ONCE };
static int mode,acquires,submits,heads,writes;
static int (*real_acquire)(void*,unsigned,int,int,BnGPUMoEBankLease*);
static int (*real_submit)(void*,void*,const BnGPUMoEBankGroupPlan*);
static int (*real_read)(void*,BnGPUResidentView,void*);
static int (*real_write)(void*,BnGPUResidentView,const void*);
static int (*real_sync)(void*);
extern int __real_bn_moe_host_expert_acquire(const BnMoEIO*,BnMoEState*,const BnMoEExpertMap*,int,int,int,BnMoEHostExpert*);
int __wrap_bn_moe_host_expert_acquire(const BnMoEIO *io,BnMoEState *state,
    const BnMoEExpertMap *map,int layer,int expert,int admit,BnMoEHostExpert *out) {
    if(mode==SOURCE_FAIL)return -1;
    return __real_bn_moe_host_expert_acquire(io,state,map,layer,expert,admit,out);
}
static int acquire(void *r,unsigned slot,int gate,int down,BnGPUMoEBankLease *out) {
    acquires++;if(mode==BUSY_FOREVER||(mode==BUSY_ONCE&&acquires==1))return BN_GPU_MOE_BANK_BUSY;
    return real_acquire(r,slot,gate,down,out);
}
static int submit(void *c,void *r,const BnGPUMoEBankGroupPlan *p) {
    int result=real_submit(c,r,p);
    if(!result&&mode==SUBMIT_FAIL&&++submits==1)return -1;
    return result;
}
static int readback(void *r,BnGPUResidentView v,void *out) {
    int head=v.type==BN_GGUF_TENSOR_BF16&&v.count==248320;
    if(head&&mode==HEAD_FAIL&&heads==1)return -1;
    int result=real_read(r,v,out);if(!result&&head)heads++;
    return result;
}
static int writeback(void *r,BnGPUResidentView v,const void *data) {writes++;return real_write(r,v,data);}
static int synchronize(void *r) {return mode==FENCE_FAIL&&heads==2?-1:real_sync(r);}
static int *load_ids(const char *path,int *count) {
    FILE *f=fopen(path,"rb");assert(f);assert(!fseek(f,0,SEEK_END));long bytes=ftell(f);assert(bytes>0&&bytes<=516&&bytes%4==0);
    rewind(f);int *ids=malloc((size_t)bytes);assert(ids&&sizeof(*ids)==4&&fread(ids,1,(size_t)bytes,f)==(size_t)bytes&&!fclose(f));*count=(int)bytes/4;return ids;
}
static void prepare(BnModel *model,BnSession **sessions) {
    for(int i=0;i<2;i++) {
        if(sessions[i])bn_session_reset(sessions[i],model);
        else {sessions[i]=bn_session_create_backend(model,NULL);assert(sessions[i]);}
        BnTransformerBF16RequestMemory quote;
        int before_writes=writes;
        assert(!bn_transformer_bf16_flash_memory_plan(model,bn_model_gpu(model),129,256,&quote));
        assert(writes==before_writes);
        assert(!bn_transformer_bf16_flash_prepare_context(model,bn_model_gpu(model),bn_session_backend(sessions[i]),129,256,(size_t)16<<30,256u<<20));
        BnTransformerBF16FlashInfo prepared;assert(!bn_transformer_bf16_flash_info(bn_session_backend(sessions[i]),&prepared));
        assert(quote.device_bytes==prepared.gpu_bytes&&quote.host_bytes==prepared.host_bytes&&quote.host_peak_bytes==prepared.preparation_host_peak_bytes);
        for(int j=0;j<248320;j++)sessions[i]->state.logits[j]=123.25f;
        for(int j=0;j<256;j++)sessions[i]->state.token_history[j]=-7;
    }
}
static void unpublished(BnSession **sessions,int failed) {
    for(int i=0;i<2;i++) {
        assert(!sessions[i]->pos);
        BnTransformerBF16FlashInfo info;assert(!bn_transformer_bf16_flash_info(bn_session_backend(sessions[i]),&info));
        assert(!info.position&&info.failed==failed);
        for(int j=0;j<248320;j++)assert(sessions[i]->state.logits[j]==123.25f);
        for(int j=0;j<256;j++)assert(sessions[i]->state.token_history[j]==-7);
    }
}
int main(int argc,char **argv) {
    assert(argc==5||argc==6);if(argc==6)assert(!strcmp(argv[5],"--cache"));BnGGUFFile *file=bn_gguf_open_file(argv[1]);assert(file);BnModel model={0};
    assert(!bn_model_load_with_cpu_preparation(&model,file,256,0,0,0));BnGPUBackend *g=bn_gpu_cuda_create();assert(g&&g->moe_bank_submit_group);
    assert(!bn_backend_layout_prepare_bf16_flash_model_with_cache(bn_model_backend(&model),g,&model.config,&model.weights,(size_t)12<<30,64u<<20,argc==6?((size_t)4<<30):0));
    assert(!bn_model_set_moe_shard_files(&model,file->owned_maps,file->n_shards));
    const BnBackendBF16FlashFFNModel *weights=bn_backend_layout_bf16_flash_ffn_model(bn_model_backend(&model));
    size_t gate_bytes=0,down_bytes=0;
    for(int l=0;l<48;l++) {
        if(weights->layers[l].expert_map.expert_gate_bytes>gate_bytes)gate_bytes=weights->layers[l].expert_map.expert_gate_bytes;
        if(weights->layers[l].expert_map.expert_down_bytes>down_bytes)down_bytes=weights->layers[l].expert_map.expert_down_bytes;
    }
    BnMoECache *cache=bn_moe_cache_create(128u<<20,gate_bytes,gate_bytes,down_bytes);assert(cache);bn_model_set_moe_cache(&model,cache);
    int counts[2],*ids[]={load_ids(argv[2],counts),load_ids(argv[3],counts+1)};assert(counts[0]==5&&counts[1]==129);
    real_acquire=g->moe_bank_acquire;real_submit=g->moe_bank_submit_group;real_read=g->resident_read;real_write=g->resident_write;real_sync=g->resident_synchronize;
    g->moe_bank_acquire=acquire;g->moe_bank_submit_group=submit;g->resident_read=readback;g->resident_write=writeback;g->resident_synchronize=synchronize;
    BnSession *sessions[2]={0};prepare(&model,sessions);
    BnTransformerBF16SessionRow rows[2];for(int i=0;i<2;i++)rows[i]=(BnTransformerBF16SessionRow){sessions[i],ids[i],counts[i],0,1};
    BnTransformerBF16SharedInfo info,sentinel;memset(&sentinel,0x5a,sizeof(sentinel));info=sentinel;
    int before=writes,last=ids[1][counts[1]-1];ids[1][counts[1]-1]=248320;
    assert(bn_transformer_bf16_flash_forward_group_session(&model,rows,2,&info)==-1);ids[1][counts[1]-1]=last;
    assert(writes==before&&!memcmp(&info,&sentinel,sizeof(info)));unpublished(sessions,0);
    rows[1].session=sessions[0];assert(bn_transformer_bf16_flash_forward_group_session(&model,rows,2,&info)==-1);rows[1].session=sessions[1];
    BnModel copy=model;assert(bn_transformer_bf16_flash_forward_group_session(&copy,rows,2,&info)==-1);
    assert(bn_transformer_bf16_flash_forward_group_session(&model,rows,0,&info)==-1);
    assert(bn_transformer_bf16_flash_forward_group_session(&model,rows,17,&info)==-1);
    assert(writes==before&&!memcmp(&info,&sentinel,sizeof(info)));unpublished(sessions,0);
    for(int failure=BUSY_FOREVER;failure<=FENCE_FAIL;failure++) {
        if(failure!=BUSY_FOREVER)prepare(&model,sessions);
        mode=failure;acquires=submits=heads=0;info=sentinel;
        assert(bn_transformer_bf16_flash_forward_group_session(&model,rows,2,&info)==-1);
        assert(!memcmp(&info,&sentinel,sizeof(info)));unpublished(sessions,1);
        if(failure==BUSY_FOREVER)assert(acquires==8);
        if(failure==SUBMIT_FAIL)assert(submits==1);
        if(failure==HEAD_FAIL)assert(heads==1);
        if(failure==FENCE_FAIL)assert(heads==2);
        before=writes;assert(bn_transformer_bf16_flash_forward_group_session(&model,rows,2,&info)==-1&&writes==before);
        mode=NORMAL;
    }
    prepare(&model,sessions);mode=BUSY_ONCE;acquires=heads=0;
    assert(!bn_transformer_bf16_flash_forward_group_session(&model,rows,2,&info));mode=NORMAL;
    assert(info.shared_layers==48&&info.shared_banks&&info.shared_upload_bytes_avoided&&info.busy_fences);
    assert(info.joint_kernel_launches&&info.joint_projection_launches&&info.joint_launches_avoided);
    for(int i=0;i<2;i++) {
        assert(sessions[i]->pos==counts[i]);assert(!memcmp(sessions[i]->state.token_history,ids[i],(size_t)counts[i]*4));
        char path[4096];assert(snprintf(path,sizeof(path),"%s/request-%d-head.f32le",argv[4],i)>0);
        FILE *out=fopen(path,"wbx");assert(out&&fwrite(sessions[i]->state.logits,4,248320,out)==248320&&!fclose(out));
    }
    /* Cohort -> singleton -> cohort transitions retain chronological private
     * PLE/GDN/QSA state and public history. The first decode still has a source
     * oracle; subsequent request0 heads can be checked against its longer oracle. */
    int decode[2];
    for(int i=0;i<2;i++) {
        decode[i]=0;for(int j=1;j<248320;j++)if(sessions[i]->state.logits[j]>sessions[i]->state.logits[decode[i]])decode[i]=j;
        assert(bn_transformer_bf16_forward_session(&model,sessions[i],decode+i,1,counts[i],0));
        assert(sessions[i]->pos==counts[i]+1&&sessions[i]->state.token_history[counts[i]]==decode[i]);
        char path[4096];snprintf(path,sizeof(path),"%s/request-%d-decode.f32le",argv[4],i);
        FILE *out=fopen(path,"wbx");assert(out&&fwrite(sessions[i]->state.logits,4,248320,out)==248320&&!fclose(out));
    }
    for(int step=0;step<3;step++) {
        for(int i=0;i<2;i++) {
            decode[i]=0;for(int j=1;j<248320;j++)if(sessions[i]->state.logits[j]>sessions[i]->state.logits[decode[i]])decode[i]=j;
            rows[i]=(BnTransformerBF16SessionRow){sessions[i],decode+i,1,counts[i]+1+step,0};
        }
        assert(!bn_transformer_bf16_flash_forward_group_session(&model,rows,2,&info));
        for(int i=0;i<2;i++)assert(sessions[i]->pos==counts[i]+2+step&&sessions[i]->state.token_history[counts[i]+1+step]==decode[i]);
        char path[4096];snprintf(path,sizeof(path),"%s/request-0-phase%d.f32le",argv[4],step+2);
        FILE *out=fopen(path,"wbx");assert(out&&fwrite(sessions[0]->state.logits,4,248320,out)==248320&&!fclose(out));
    }
    for(int i=0;i<2;i++){bn_session_free(sessions[i],NULL);free(ids[i]);}
    g->moe_bank_acquire=real_acquire;g->moe_bank_submit_group=real_submit;g->resident_read=real_read;g->resident_write=real_write;g->resident_synchronize=real_sync;
    bn_moe_cache_free(cache);bn_model_set_moe_cache(&model,NULL);bn_model_free(&model);bn_gpu_cuda_destroy(g);bn_gguf_free(file);
    puts("Flash cohort validation/publication, bounded BUSY, I/O/partial-submit/read/fence failure, reset and recovery PASSED");return 0;
}
