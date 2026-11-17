#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../bench/native_text_workload.h"

int main(void) {
    BnTextWorkloadShape short_shape={0},long_shape={0};
    assert(!native_text_workload_add(&short_shape,1026,0));
    for(int i=0;i<5;i++)assert(!native_text_workload_add(&short_shape,1,1026+i));
    assert(!native_text_workload_add(&long_shape,2048,0));
    assert(!native_text_workload_add(&long_shape,2,2048));
    for(int i=0;i<5;i++)assert(!native_text_workload_add(&long_shape,1,2050+i));
    int batch=-1,context=-1;
    assert(!native_text_workload_finish(&short_shape,&batch,&context));
    assert(batch==1026&&context==1280&&short_shape.frames==6);
    assert(!native_text_workload_finish(&long_shape,&batch,&context));
    assert(batch==2048&&context==2304&&long_shape.frames==7);

    BnTextWorkloadShape empty={0},before=long_shape;
    batch=-1;context=-2;
    assert(native_text_workload_finish(&empty,&batch,&context)==-1);
    assert(batch==-1&&context==-2);
    assert(native_text_workload_add(NULL,1,0)==-1);
    const int bad[][2]={{0,0},{16385,0},{1,-1},{1,INT_MIN},{1,INT_MAX},{2,INT_MAX-1}};
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++) {
        assert(native_text_workload_add(&long_shape,bad[i][0],bad[i][1])==-1);
        assert(!memcmp(&long_shape,&before,sizeof(before)));
    }
    BnTextWorkloadShape full={1,1,SIZE_MAX};
    assert(native_text_workload_add(&full,1,1)==-1&&full.frames==SIZE_MAX);
    BnTextWorkloadShape limit={0};
    assert(!native_text_workload_add(&limit,1,INT_MAX-257));
    assert(!native_text_workload_finish(&limit,&batch,&context));
    assert(context==INT_MAX-255);
    assert(!native_text_workload_add(&limit,1,INT_MAX-256));
    batch=-1;context=-2;
    assert(native_text_workload_finish(&limit,&batch,&context)==-1);
    assert(batch==-1&&context==-2);
    assert(native_text_workload_finish(NULL,&batch,&context)==-1);
    assert(native_text_workload_finish(&short_shape,NULL,&context)==-1);
    assert(native_text_workload_finish(&short_shape,&batch,NULL)==-1);

    for(int count=1;count<=BN_TEXT_WORKLOAD_MAX_REQUESTS;count++) {
        int seen[BN_TEXT_WORKLOAD_MAX_REQUESTS][2]={{0}};
        for(int cycle=0;cycle<4;cycle++) {
            unsigned mask=0;
            for(int item=0;item<count;item++) {
                int run=cycle*count+item;
                int pick=native_text_workload_pick(run,count);
                int skip=native_text_workload_skip(run,count);
                assert(pick>=0&&pick<count&&skip==(cycle&1));
                assert(!(mask&(1u<<pick)));mask|=1u<<pick;seen[pick][skip]++;
            }
            assert(mask==(1u<<count)-1);
        }
        for(int i=0;i<count;i++)assert(seen[i][0]==2&&seen[i][1]==2);
    }
    for(int run=0;run<4;run++) {
        assert(native_text_workload_pick(run,1)==0);
        assert(native_text_workload_skip(run,1)==(run&1));
    }
    assert(native_text_workload_pick(-1,2)==-1);
    assert(native_text_workload_skip(INT_MIN,2)==-1);
    assert(native_text_workload_pick(0,0)==-1);
    assert(native_text_workload_skip(0,9)==-1);
    puts("Native text workload PASSED");
    return 0;
}
