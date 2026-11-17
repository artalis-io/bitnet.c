#include "../src/main_serve_io.h"
#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>
#include <time.h>

static double tick(void *context) {return (double)atomic_fetch_add((_Atomic unsigned*)context,1);}
static double invalid_clock(void *context) {(void)context;return NAN;}
static void put(int fd,const void *data,size_t bytes) {
    const unsigned char *p=data;
    while(bytes) {ssize_t n=write(fd,p,bytes);if(n<0&&errno==EINTR)continue;assert(n>0);p+=n;bytes-=(size_t)n;}
}
static void wait_state(ServeIngress *s,int input,int space,size_t records) {
    struct timespec until;assert(!clock_gettime(CLOCK_REALTIME,&until));until.tv_sec+=10;
    pthread_mutex_lock(&s->mutex);
    while((input&&!s->info.waiting_input)||(space&&!s->info.waiting_space)||s->info.records<records)
        assert(!pthread_cond_timedwait(&s->changed,&s->mutex,&until));
    pthread_mutex_unlock(&s->mutex);
}
static void parse(void) {
    BnJSON d={0};ServeCommand command={0};
    char text[]="{\"op\":\"submit\",\"id\":18446744073709551615,\"prompt\":\"hi\\n\\uD83D\\uDE00\",\"max_outputs\":2}";
    assert(!serve_command_parse(&d,text,strlen(text),256,3,&command));memset(text,'?',strlen(text));
    assert(command.kind==SERVE_SUBMIT&&command.id==UINT64_MAX&&command.max_outputs==2&&!strcmp(command.prompt,"hi\n\xf0\x9f\x98\x80"));bn_json_free(&d);
    const char *valid[]={"{\"op\":\"submit\",\"id\":0,\"prompt\":\"x\"}","{\"op\":\"cancel\",\"id\":4}","{\"op\":\"shutdown\"}"};
    for(int i=0;i<3;i++) {
        assert(!serve_command_parse(&d,valid[i],strlen(valid[i]),256,3,&command)&&command.kind==(ServeCommandKind)i);
        if(!i)assert(command.max_outputs==3&&command.id==0);
        bn_json_free(&d);
    }
    const char *bad[]={"", "[]", "{}", "{\"op\":1}","{\"op\":\"other\"}",
        "{\"op\":\"submit\",\"id\":1,\"prompt\":\"\"}","{\"op\":\"submit\",\"id\":1,\"prompt\":0}",
        "{\"op\":\"submit\",\"id\":1,\"prompt\":\"x\",\"max_outputs\":0}",
        "{\"op\":\"submit\",\"id\":1,\"prompt\":\"x\",\"max_outputs\":4}",
        "{\"op\":\"submit\",\"id\":1,\"prompt\":\"x\",\"max_outputs\":1.0}",
        "{\"op\":\"submit\",\"id\":1,\"prompt\":\"x\",\"extra\":true}",
        "{\"op\":\"submit\",\"id\":-1,\"prompt\":\"x\"}",
        "{\"op\":\"submit\",\"id\":18446744073709551616,\"prompt\":\"x\"}",
        "{\"op\":\"submit\",\"id\":1,\"\\u0069d\":2,\"prompt\":\"x\"}",
        "{\"op\":\"submit\",\"id\":1,\"prompt\":\"\\u0000\"}",
        "{\"op\":\"submit\",\"id\":1,\"prompt\":\"\xc0\x80\"}",
        "{\"op\":\"cancel\"}","{\"op\":\"cancel\",\"id\":\"1\"}",
        "{\"op\":\"cancel\",\"id\":1,\"prompt\":\"x\"}","{\"op\":\"shutdown\",\"id\":0}",
        "{\"op\":\"shutdown\"}{}"};
    ServeCommand before=command;
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++) {
        assert(serve_command_parse(&d,bad[i],strlen(bad[i]),256,3,&command)==-1&&!d.state);
        assert(!memcmp(&command,&before,sizeof(command)));
    }
    assert(serve_command_parse(&d,valid[0],strlen(valid[0]),4,3,&command)==-1&&!d.state);
    assert(serve_command_parse(&d,valid[0],strlen(valid[0]),256,0,&command)==-1&&!d.state);
    assert(!bn_json_init(&d,"{}",2));void *state=d.state;
    assert(serve_command_parse(&d,valid[0],strlen(valid[0]),256,3,&command)==-1&&d.state==state);bn_json_free(&d);
}
static void records(void) {
    int fds[2];assert(!pipe(fds));_Atomic unsigned clock=1;ServeIngress s={0};
    assert(!serve_ingress_init(&s,fds[0],8,tick,&clock));
    ServeRecord r={99,99,-1};unsigned char out[9];memset(out,0xa5,sizeof(out));
    assert(!serve_ingress_pop(&s,out,sizeof(out),&r,0)&&r.bytes==99&&out[0]==0xa5);
    put(fds[1],"12345678\n123456789\n\na",21);close(fds[1]);
    assert(serve_ingress_pop(&s,out,8,&r,1)==-1&&r.bytes==99);
    assert(serve_ingress_pop(&s,out,sizeof(out),&r,1)==1&&r.bytes==8&&!r.oversized&&!memcmp(out,"12345678",9));
    assert(serve_ingress_pop(&s,out,sizeof(out),&r,1)==1&&!r.bytes&&r.oversized&&!out[0]);
    assert(serve_ingress_pop(&s,out,sizeof(out),&r,1)==1&&!r.bytes&&!r.oversized&&!out[0]);
    assert(serve_ingress_pop(&s,out,sizeof(out),&r,1)==1&&r.bytes==1&&!strcmp((char*)out,"a"));
    assert(serve_ingress_pop(&s,out,sizeof(out),&r,1)==2);
    ServeIngressInfo info;assert(!serve_ingress_info(&s,&info)&&info.records==4&&info.oversized==1&&info.bytes==21&&info.peak<=8&&info.ended&&!info.failed);
    serve_ingress_free(&s);assert(fcntl(fds[0],F_GETFD)>=0);close(fds[0]);serve_ingress_free(&s);
}
static void arrivals(void) {
    int fds[2];assert(!pipe(fds));_Atomic unsigned clock=1;ServeIngress s={0};
    assert(!serve_ingress_init(&s,fds[0],64,tick,&clock));
    put(fds[1],"one\n",4);wait_state(&s,0,0,1);
    put(fds[1],"two\n",4);wait_state(&s,0,0,2);atomic_store(&clock,100);
    ServeRecord r;unsigned char out[65];
    assert(serve_ingress_pop(&s,out,sizeof(out),&r,0)==1&&r.received_ms==1&&!strcmp((char*)out,"one"));
    assert(serve_ingress_pop(&s,out,sizeof(out),&r,0)==1&&r.received_ms==2&&!strcmp((char*)out,"two"));
    wait_state(&s,1,0,2);ServeIngress copy;memcpy(&copy,&s,sizeof(copy));ServeIngressInfo info;
    assert(serve_ingress_info(&copy,&info)==-1);serve_ingress_free(&copy);
    serve_ingress_free(&s);assert(fcntl(fds[0],F_GETFD)>=0);close(fds[0]);close(fds[1]);
}
static void full_stop(void) {
    int fds[2];assert(!pipe(fds));_Atomic unsigned clock=1;ServeIngress s={0};
    assert(!serve_ingress_init(&s,fds[0],8,tick,&clock));
    put(fds[1],"0\n1\n2\n3\n4\n5\n6\n7\n8\n",18);wait_state(&s,0,1,8);
    ServeIngressInfo info;assert(!serve_ingress_info(&s,&info)&&info.queued==8&&info.peak==8&&info.waiting_space);
    assert(!serve_ingress_stop(&s)&&!serve_ingress_stop(&s));serve_ingress_free(&s);
    assert(fcntl(fds[0],F_GETFD)>=0);close(fds[0]);close(fds[1]);
}
static void split_and_binary(void) {
    int fds[2];assert(!pipe(fds));_Atomic unsigned clock=1;ServeIngress s={0};
    assert(!serve_ingress_init(&s,fds[0],10000,tick,&clock));
    unsigned char *data=malloc(10001),*out=malloc(10001);assert(data&&out);memset(data,'x',10000);data[10000]='\n';
    put(fds[1],data,10001);const char binary[]={'a',0,'b','\n'};put(fds[1],binary,sizeof(binary));close(fds[1]);ServeRecord r;
    assert(serve_ingress_pop(&s,out,10001,&r,1)==1&&r.bytes==10000&&!r.oversized&&!memcmp(data,out,10000)&&!out[10000]);
    assert(serve_ingress_pop(&s,out,10001,&r,1)==1&&r.bytes==3&&!memcmp(out,binary,3)&&!out[3]);
    assert(serve_ingress_pop(&s,out,10001,&r,1)==2);serve_ingress_free(&s);close(fds[0]);free(data);free(out);
}
static void errors(void) {
    ServeIngress s={0};size_t bytes=7;
    assert(serve_ingress_plan(0,&bytes)==-1&&bytes==7);
    assert(serve_ingress_plan(SIZE_MAX,&bytes)==-1&&bytes==7);
    assert(!serve_ingress_plan(SERVE_RECORD_BYTES,&bytes)&&bytes==sizeof(s)+8*(SERVE_RECORD_BYTES+1));
    _Atomic unsigned clock=1;assert(!serve_ingress_init(&s,INT_MAX,8,tick,&clock));
    unsigned char out[9];ServeRecord r;assert(serve_ingress_pop(&s,out,sizeof(out),&r,1)==-1);serve_ingress_free(&s);
    int fds[2];assert(!pipe(fds));assert(!serve_ingress_init(&s,fds[0],8,invalid_clock,NULL));put(fds[1],"x\n",2);
    assert(serve_ingress_pop(&s,out,sizeof(out),&r,1)==-1);serve_ingress_free(&s);close(fds[0]);close(fds[1]);
}
int main(void) {
    parse();records();arrivals();full_stop();split_and_binary();errors();
    puts("Bounded serving ingress: independent arrivals, strict commands, byte limits, full/idle stop, EOF and IO failures PASSED");return 0;
}
