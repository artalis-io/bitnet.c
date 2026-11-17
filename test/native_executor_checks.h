#include "transformer_internal.h"
#include "backend_quant.h"
#ifndef __EMSCRIPTEN__
typedef struct {
    NativeTrellisOwnerGPU owner;
    int created,destroyed,allocations,fail_at,math_calls,zeros,fail_zero;
} NativeExecFake;
typedef struct {NativeExecFake *owner;unsigned slots;} NativeExecRequest;
static void *native_exec_create(void *ctx) {NativeExecFake*t=ctx;NativeExecRequest*r=calloc(1,sizeof(*r));assert(r);r->owner=t;t->created++;return r;}
static void native_exec_destroy(void *ptr) {NativeExecRequest*r=ptr;r->owner->destroyed++;free(r);}
static size_t native_exec_size(void *ctx) {(void)ctx;return 8;}
static size_t native_exec_index_size(void *ctx,int n) {(void)ctx;(void)n;return 8;}
static int native_exec_reserve(void *ctx,void *ptr) {(void)ctx;(void)ptr;return 0;}
static int native_exec_reserve_int(void *ctx,void *ptr,int n) {(void)n;return native_exec_reserve(ctx,ptr);}
static int native_exec_info(void *ctx,void *ptr,int m,int n,BnGPUFusedEXL3LaunchInfo*out) {
    (void)ctx;(void)ptr;(void)m;(void)n;*out=(BnGPUFusedEXL3LaunchInfo){92160,512,1,188,23};return 0;
}
static int native_exec_alloc(void *ptr,int type,size_t count,BnGPUResidentView*out) {
    NativeExecRequest*r=ptr;NativeExecFake*t=r->owner;
    if(++t->allocations==t->fail_at)return -1;
    *out=(BnGPUResidentView){ptr,0,count,r->slots++,type};return 0;
}
static int native_exec_write(void *p,BnGPUResidentView v,const void *b) {(void)p;(void)v;(void)b;return -1;}
static int native_exec_read(void *p,BnGPUResidentView v,void *b) {(void)p;(void)v;(void)b;return -1;}
static int native_exec_zero(void *p,BnGPUResidentView v) {NativeExecRequest*r=p;(void)v;r->owner->zeros++;return r->owner->fail_zero?-1:0;}
static int native_exec_sync(void *p) {(void)p;return 0;}
#define NATIVE_EXEC_REJECT(N,T) static int N(void*c,void*r,const T*p){(void)r;(void)p;((NativeExecFake*)c)->math_calls++;return -1;}
NATIVE_EXEC_REJECT(native_exec_router,BnGPURouterResidentPlan)
NATIVE_EXEC_REJECT(native_exec_index,BnGPUNativeMoEIndexPlan)
NATIVE_EXEC_REJECT(native_exec_group,BnGPUNativeMoEGroupPlan)
NATIVE_EXEC_REJECT(native_exec_reduce,BnGPUNativeMoEReducePlan)
NATIVE_EXEC_REJECT(native_exec_fused,BnGPUFusedEXL3ResidentPlan)
NATIVE_EXEC_REJECT(native_exec_decode,BnGPUDecodeEXL3ResidentPlan)
NATIVE_EXEC_REJECT(native_exec_folded,BnGPUFoldedEXL3FFNResidentPlan)
NATIVE_EXEC_REJECT(native_exec_shared,BnGPUF16FFNResidentPlan)
NATIVE_EXEC_REJECT(native_exec_add,BnGPUAddF32ResidentPlan)
#undef NATIVE_EXEC_REJECT
static void test_native_executor_setup(void) {
    assert(bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16)==BN_GGUF_TENSOR_F16&&bn_backend_quant_scalar_bytes(BN_BACKEND_SCALAR_F16)==2);
    char dir[]="/tmp/bn-native-executor-XXXXXX",path[256];assert(mkdtemp(dir));native_anatomy_fixture(dir,0);
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    char *config=native_numerics_replace(native_anatomy_config,"\"engram_compressed_vocab_size\":99092","\"engram_compressed_vocab_size\":256");
    BnModel model={0};assert(!bn_model_load_trellis_config(&model,&index,config,strlen(config)));free(config);
    assert(!bn_model_prepare_native_anatomy(&model)&&!bn_model_prepare_native_numerics(&model));
    NativeExecFake t={0};t.owner.model=&model;BnGPUBackend g={.ctx=&t,.buffer_create_f16_rows=native_owner_core_create,
        .buffer_destroy=native_owner_destroy,.trellis_buffer_create=native_owner_expert_create,
        .resident_request_create=native_exec_create,.resident_request_destroy=native_exec_destroy,.resident_alloc=native_exec_alloc,
        .resident_write=native_exec_write,.resident_read=native_exec_read,.resident_zero=native_exec_zero,.resident_synchronize=native_exec_sync,
        .router_sqrtsp_resident=native_exec_router,.native_moe_index_workspace_size=native_exec_index_size,
        .native_moe_index_reserve=native_exec_reserve_int,.native_moe_index_resident=native_exec_index,
        .native_moe_group_resident=native_exec_group,.native_moe_reduce_resident=native_exec_reduce,
        .trellis_fused_workspace_size=native_exec_size,.trellis_fused_reserve=native_exec_reserve,
        .trellis_fused_launch_info=native_exec_info,.trellis_fused_resident=native_exec_fused,.trellis_decode_resident=native_exec_decode,
        .trellis_folded_workspace_size=native_exec_size,.trellis_folded_reserve=native_exec_reserve_int,
        .trellis_folded_ffn_resident=native_exec_folded,.linear_f16_workspace_size=native_exec_size,
        .linear_f16_reserve=native_exec_reserve,.ffn_f16_resident=native_exec_shared,.add_f32_resident=native_exec_add};
    BnBackendNativeCoreStats cs;assert(!bn_backend_layout_native_core_plan(&index,bn_model_native_anatomy(&model),&cs));
    assert(!bn_model_prepare_native_core(&model,&g,cs.device_peak_bytes,cs.host_staging_bytes));
    const BnMoEExpertReader*reader=bn_model_trellis_expert_reader(&model);size_t entry=reader->bytes[0]+reader->bytes[1]+reader->bytes[2];
    assert(!bn_gpu_moe_trellis_prepare_model(&model,entry*2,entry*2,entry*2,entry));
    BnTransformerTrellisOptions o={64,2,0,0,0};BnTransformerTrellisWorkspace w,saved;
    memset(&saved,0xa5,sizeof(saved));memcpy(&w,&saved,sizeof(w));
    assert(bn_transformer_trellis_session_plan(NULL,&o,&w)==-1&&!memcmp(&w,&saved,sizeof(w)));
    BnTransformerTrellisOptions bad=o;bad.max_tokens=0;assert(bn_transformer_trellis_session_plan(&model,&bad,&w)==-1&&!memcmp(&w,&saved,sizeof(w)));
    bad=o;bad.fused_window=33;assert(bn_transformer_trellis_session_plan(&model,&bad,&w)==-1);
    bad=o;bad.f16acc_enabled=2;assert(bn_transformer_trellis_session_plan(&model,&bad,&w)==-1);
    assert(!bn_transformer_trellis_session_plan(&model,&o,&w)&&w.device_bytes&&w.host_bytes&&!t.created);
    BnSession*s=bn_session_create_backend(&model,NULL);assert(s);
    o.device_budget=w.device_bytes;o.host_budget=w.host_bytes;
    bad=o;bad.device_budget--;assert(bn_transformer_trellis_prepare_session(&model,s,&bad)==-1&&!t.created);
    bad=o;bad.host_budget--;assert(bn_transformer_trellis_prepare_session(&model,s,&bad)==-1&&!t.created);
    t.fail_zero=1;
    assert(bn_transformer_trellis_prepare_session(&model,s,&o)==-1&&t.created==t.destroyed);
    t.fail_zero=0;
    int success=0;
    for(int fail=1;fail<52;fail++) {
        t.fail_at=fail;t.allocations=0;int rc=bn_transformer_trellis_prepare_session(&model,s,&o);
        if(!rc){success=1;assert(t.created==t.destroyed+1);break;}
        assert(t.created==t.destroyed&&!bn_backend_session_has_gpu_resources(bn_session_backend(s)));
    }
    assert(success&&t.created>20);t.fail_at=0;int created=t.created,zeros=t.zeros;
    assert(!bn_transformer_trellis_prepare_session(&model,s,&o)&&t.created==created&&t.zeros==zeros);
    bad=o;bad.f16acc_enabled=1;assert(bn_transformer_trellis_prepare_session(&model,s,&bad)==-1&&t.created==created);
    void *request=NULL;BnGPUResidentView input={0};assert(!bn_transformer_trellis_session_input(&model,s,64,&request,&input)&&request&&input.count==64u*128);
    assert(!bn_transformer_trellis_session_input(&model,s,1,&request,&input)&&input.count==128);
    assert(!bn_transformer_trellis_session_input(&model,s,64,&request,&input));
    void *keep=request;BnGPUResidentView old=input;
    assert(bn_transformer_trellis_session_input(&model,s,0,&request,&input)==-1&&request==keep&&!memcmp(&input,&old,sizeof(input)));
    assert(bn_transformer_trellis_session_input(&model,s,65,&request,&input)==-1&&request==keep&&!memcmp(&input,&old,sizeof(input)));
    BnTransformerTrellisResult result,unchanged;memset(&unchanged,0xa5,sizeof(unchanged));memcpy(&result,&unchanged,sizeof(result));
    assert(bn_transformer_trellis_run_learned(&model,s,-1,64,0,&result)==-1&&!memcmp(&result,&unchanged,sizeof(result)));
    /* No prepared router in this setup fixture: reject before GPU work. */
    assert(bn_transformer_trellis_run_learned(&model,s,39,64,0,&result)==-1&&!memcmp(&result,&unchanged,sizeof(result))&&!t.math_calls);
    assert(bn_transformer_trellis_run_learned(&model,s,39,1,0,&result)==-1&&!memcmp(&result,&unchanged,sizeof(result))&&!t.math_calls);
    BnModel copy=model;assert(bn_transformer_trellis_session_input(&copy,s,64,&request,&input)==-1);
    bn_session_reset(s,&model);assert(t.created==t.destroyed);
    assert(bn_transformer_trellis_session_input(&model,s,64,&request,&input)==-1);
    assert(!bn_transformer_trellis_prepare_session(&model,s,&o));bn_session_free(s,NULL);assert(t.created==t.destroyed);
    /* Decode-only requests neither require nor reserve prefill operators. */
    g.native_moe_index_workspace_size=NULL;g.native_moe_index_reserve=NULL;g.native_moe_index_resident=NULL;
    g.native_moe_group_resident=NULL;g.native_moe_reduce_resident=NULL;
    g.trellis_fused_workspace_size=NULL;g.trellis_fused_reserve=NULL;g.trellis_fused_launch_info=NULL;g.trellis_fused_resident=NULL;
    g.trellis_folded_workspace_size=NULL;g.trellis_folded_reserve=NULL;g.trellis_folded_ffn_resident=NULL;
    assert(bn_transformer_trellis_session_plan(&model,&o,&w)==-1);
    for(int max=1;max<=8;max+=7) {
        BnTransformerTrellisOptions small={max,1,0,0,0};
        assert(!bn_transformer_trellis_session_plan(&model,&small,&w)&&!w.slot_rows&&!w.group_rows&&!w.batch_cap&&w.device_bytes<o.device_budget);
        small.device_budget=w.device_bytes;small.host_budget=w.host_bytes;
        s=bn_session_create_backend(&model,NULL);assert(s);int done=0;
        for(int fail=1;fail<52;fail++) {
            t.allocations=0;t.fail_at=fail;
            if(!bn_transformer_trellis_prepare_session(&model,s,&small)){done=1;break;}
            assert(t.created==t.destroyed&&!bn_backend_session_has_gpu_resources(bn_session_backend(s)));
        }
        assert(done);t.fail_at=0;
        assert(!bn_transformer_trellis_session_input(&model,s,max,&request,&input)&&input.count==(size_t)max*128);
        bn_session_free(s,NULL);assert(t.created==t.destroyed);
        g.trellis_decode_resident=NULL;
        assert(bn_transformer_trellis_session_plan(&model,&small,&w)==-1);
        g.trellis_decode_resident=native_exec_decode;
    }
    bn_model_free(&model);bn_safetensors_index_free(&index);snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!remove(path));
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!remove(path)&&!rmdir(dir));
    puts("Native executor setup budget/rollback/idempotence/identity/lifetime PASSED");
}
#else
static void test_native_executor_setup(void) {
    assert(bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16)==BN_GGUF_TENSOR_F16&&bn_backend_quant_scalar_bytes(BN_BACKEND_SCALAR_F16)==2);}
#endif
