#include "cuda_native_candidate_use_reference.h"
static int native_use_kept(int row,int b,int B,int kind) {
    switch(kind){case 0:return b<2048;case 1:return b>=B-2048;case 2:return b==(row*17)%B||b==(row*17+31)%B||b==(row*17+62)%B;case 3:return(b+row*131)%B<2048;case 4:return b==B-1;case 5:return 0;case 6:return b%257==row%257;default:return b==0;}
}
static void run_native_candidate_use_reference(BnGPUBackend*g) {
    assert(g->index_candidate_use_f16_resident);size_t calls=0;
    for(size_t c=0;c<sizeof(native_use_fixtures)/sizeof(*native_use_fixtures);c++) {
        int R=native_use_fixtures[c].R,T=native_use_fixtures[c].T,pos=native_use_fixtures[c].pos,rate=native_use_fixtures[c].rate,kind=native_use_fixtures[c].kind,pages=native_use_fixtures[c].pages,epp=native_use_fixtures[c].epp,shift=native_use_fixtures[c].shift,cap=R*16384;if(cap>T)cap=T;int stride=((cap+127)/128)*128;
        BnGPUIndexCandidateUseWorkspace ws;assert(!bn_gpu_index_candidate_use_workspace(R,T,stride,pos,rate,&ws));size_t K=pages?(size_t)pages*epp:(size_t)T;
        size_t n[]={(size_t)R*4096,(size_t)R*32,K*128+shift,(size_t)pages,ws.drop_count,(size_t)cap,1,ws.gathered_count+shift,ws.scores_count,ws.output_count,(size_t)ws.blocks,(size_t)ws.blocks,(size_t)ws.tiles,ws.select.indices_count,ws.select.scores_count,ws.select.counts_count};int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32};
        uint16_t*qv=malloc(n[0]*2),*w=malloc(n[1]*2),*keys=malloc(n[2]*2);int32_t*bt=pages?malloc(n[3]*4):NULL;unsigned char*drop=malloc(n[4]);assert(qv&&w&&keys&&drop&&(!pages||bt));
        for(size_t i=0;i<n[0];i++)qv[i]=bn_fp32_to_fp16(i/4096==1?0:(float)((int)((i*17+13)%257)-128)/256);
        for(size_t i=0;i<n[1];i++)w[i]=bn_fp32_to_fp16((float)((int)((i*7+3)%31)-15)/32);
        memset(keys,0xa5,n[2]*2);for(int j=0;j<pages;j++)bt[j]=pages-1-j;
        for(int row=0;row<T;row++)for(int d=0;d<128;d++){size_t physical=pages?(size_t)bt[row/epp]*epp+row%epp:(size_t)row;size_t i=(size_t)row*128+d;keys[shift+physical*128+d]=bn_fp32_to_fp16((float)((int)((i*11+7)%257)-128)/64);}
        for(int row=0;row<R;row++)for(int b=0;b<ws.blocks;b++)drop[(size_t)row*ws.blocks+b]=!native_use_kept(row,b,ws.blocks,kind);
        int32_t*scan[3]={malloc(n[10]*4),malloc(n[11]*4),malloc(n[12]*4)};assert(scan[0]&&scan[1]&&scan[2]);int total=0,local=0;
        for(int b=0;b<ws.blocks;b++){if(b%256==0){scan[2][b/256]=total;local=0;}int kept=0;for(int row=0;row<R;row++)kept|=!drop[(size_t)row*ws.blocks+b];int count=kept?(T-b*8<8?T-b*8:8):0;scan[0][b]=count;scan[1][b]=local;local+=count;total+=count;}assert(total<=cap);
        void*r[2];BnGPUResidentView roots[2][16];BnGPUIndexCandidateUseResidentPlan p[2];const void*data[]={qv,w,keys,bt,drop};
        for(int j=0;j<2;j++) {
            r[j]=g->resident_request_create(g->ctx);assert(r[j]);BnGPUResidentView v[16]={{0}};
            for(int i=0;i<16;i++)if(n[i])v[i]=native_router_guard(g,r[j],types[i],n[i],i<5?data[i]:NULL,&roots[j][i]);
            v[2]=bn_gpu_resident_view_slice(v[2],shift,K*128);v[7]=bn_gpu_resident_view_slice(v[7],shift,ws.gathered_count);
            p[j]=(BnGPUIndexCandidateUseResidentPlan){R,T,stride,pos,rate,pages,epp,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15]};BnGPUIndexCandidateUseResidentPlan bad;
#define USE_BAD(F,V) do{bad=p[j];bad.F=(V);assert(g->index_candidate_use_f16_resident(g->ctx,r[j],&bad)==-1);}while(0)
            USE_BAD(n_tokens,273);USE_BAD(n_keys,16384);USE_BAD(score_stride,stride-1);USE_BAD(pos0,INT_MAX);USE_BAD(rate,0);USE_BAD(entry_count.count,2);USE_BAD(entry_count,bn_gpu_resident_view_slice(p[j].block_counts,0,1));USE_BAD(block_offsets,p[j].block_counts);USE_BAD(indices.count,ws.output_count-1);USE_BAD(block_table.count,(size_t)pages+1);
            if(ws.select.split_groups){USE_BAD(scratch_counts.count,ws.select.counts_count-1);}else{USE_BAD(scratch_counts,p[j].entry_count);}
#undef USE_BAD
            for(int i=5;i<10;i++)native_router_guard_check(g,r[j],roots[j][i],n[i],types[i],folded_test_unwritten(n[i]*(types[i]==BN_GGUF_TENSOR_F16?2:4)));
        }
        BnGPUIndexCandidateUseResidentPlan bad=p[0];bad.drop=p[1].drop;assert(g->index_candidate_use_f16_resident(g->ctx,r[0],&bad)==-1);
        for(int rep=0;rep<2;rep++)for(int j=0;j<2;j++) {
            assert(!g->index_candidate_use_f16_resident(g->ctx,r[j],p+j));
            for(int i=0;i<5;i++)if(n[i])native_router_guard_check(g,r[j],roots[j][i],n[i],types[i],native_norm_hash(data[i],n[i]*(types[i]==BN_GGUF_TENSOR_I8?1:types[i]==BN_GGUF_TENSOR_F16?2:4)));
            for(int i=5;i<10;i++){
                if(i==7&&shift){unsigned char*b=malloc((n[i]+32)*2);assert(b&&!g->resident_read(r[j],roots[j][i],b));assert(native_norm_hash(b+(16+shift)*2,ws.gathered_count*2)==native_use_fixtures[c].hash[2]);for(size_t k=0;k<(size_t)(16+shift)*2;k++)assert(b[k]==0xa5);for(size_t k=(16+n[i])*2;k<(n[i]+32)*2;k++)assert(b[k]==0xa5);free(b);}
                else native_router_guard_check(g,r[j],roots[j][i],n[i],types[i],native_use_fixtures[c].hash[i-5]);
            }
            for(int i=10;i<13;i++)native_router_guard_check(g,r[j],roots[j][i],n[i],types[i],native_norm_hash(scan[i-10],n[i]*4));
            for(int i=13;i<16;i++){if(n[i])native_topk_guard_only(g,r[j],roots[j][i],n[i],types[i]);}calls++;
        }
        for(int j=0;j<2;j++){g->resident_request_destroy(r[j]);}for(int j=0;j<3;j++)free(scan[j]);free(qv);free(w);free(keys);free(bt);free(drop);
    }
    printf("CUDA source candidate-use gather/score/remap PASSED: %zu calls\n",calls);
}
