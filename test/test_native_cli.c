#include "../src/main_native_options.h"
#include "../src/main_native_io.h"
#include "../src/main_native_batch_input.h"
#include <assert.h>
static void parser(void) {
    char *base[]={"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello"};NativeCLIOptions o;
    assert(!native_cli_parse(7,base,&o)&&o.tokens==32&&o.prefill==512&&o.maxseq==4096&&o.ram_mb==640&&o.gpu_mb==640&&!o.predict&&!o.ram_admit);
    NativeCLIOptions before=o;
    char *invalid[][12]={
        {"bitnet","--native-text","model","-p","hello"},
        {"bitnet","--native-text","model","--profile","wrong","-p","hello"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","-n","-1"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","-n","2147483648"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","-n","1x"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--maxseq","1025"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--prefill-tokens","0"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--temporary-experts","47"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--fused-window","33"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--profile","exl3-int4-f16"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--prompt-file","input"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--flash"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--token-ids","--token-ids"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--prediction-first","--prediction-first"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--ram-admit-misses","--ram-admit-misses"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--ram-admit-misses","--ram-cache-mb","0"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","-n"},
    };
    for(size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++) {
        int argc=0;while(argc<12&&invalid[i][argc])argc++;
        assert(native_cli_parse(argc,invalid[i],&o)==-1&&!memcmp(&o,&before,sizeof(o)));
    }
    char *all[]={"bitnet","--native-text","model","--profile","exl3-int4-f16","--prompt-file","-","-n","0","--prefill-tokens","1","--maxseq","256","--ram-cache-mb","0","--gpu-cache-mb","32","--temporary-experts","48","--fused-window","1","--host-budget-mb","1000","--device-budget-mb","2000","--admit-decode","--pinned-staging","--keep-special","--token-ids","--prepare-only","--trace-dir","trace"};
    assert(!native_cli_parse((int)(sizeof(all)/sizeof(all[0])),all,&o)&&!o.tokens&&o.admit&&o.pinned&&o.keep_special&&o.token_ids&&o.prepare&&o.host_mb==1000&&o.device_mb==2000&&!strcmp(o.prompt_file,"-"));
    assert(o.host_budget_set&&o.device_budget_set);
    char *uncapped[]={"bitnet","--native-text","model","--profile","exl3-int4-f16","--batch-prompts","x","--host-budget-mb","0","--device-budget-mb","0"};
    assert(!native_cli_parse(11,uncapped,&o)&&o.host_budget_set&&o.device_budget_set&&!o.host_mb&&!o.device_mb);
    char *predict[]={"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--prediction-first"};
    assert(!native_cli_parse(8,predict,&o)&&o.predict&&o.tokens==32);
    char *ram[]={"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--ram-admit-misses"};
    assert(!native_cli_parse(8,ram,&o)&&o.ram_admit&&!o.admit&&o.ram_mb==640);
    char *help[]={"bitnet","--native-text","--help"};assert(native_cli_parse(3,help,&o)==1);
    FILE *file=tmpfile();assert(file);native_cli_usage(file,"bitnet");assert(ftell(file)>0&&!fclose(file));
    int n=99;assert(!native_cli_number("2147483647",0,INT_MAX,&n)&&n==INT_MAX);
    for(const char **p=(const char*[]){"","-1","+1"," 1","1 ","1.0","9999999999999999999999999999",NULL};*p;p++) {
        n=99;assert(native_cli_number(*p,0,INT_MAX,&n)==-1&&n==99);
    }
    int cold=0;assert(!native_cli_cold_experts(512,6,384,32,&cold)&&cold==48);
    assert(!native_cli_cold_experts(1,6,384,32,&cold)&&cold==6);
    assert(!native_cli_cold_experts(8,16,384,32,&cold)&&cold==128);
    assert(native_cli_cold_experts(8,INT_MAX,INT_MAX,32,&cold)==-1&&cold==128);
    size_t value=12;assert(native_cli_add(&value,SIZE_MAX)==-1&&value==12);
    assert(native_cli_mul(SIZE_MAX,2,&value)==-1&&value==12);
    assert(!native_cli_mul(0,SIZE_MAX,&value)&&!value);
    assert(!native_cli_piece_bound(5,&value)&&value==21);
    assert(native_cli_piece_bound(SIZE_MAX,&value)==-1&&value==21);
    assert(!native_cli_context(1026,254,0,1280)&&native_cli_context(1026,255,0,1280)==-1);
    assert(!native_cli_context(1026,255,1,1280)&&native_cli_context(1026,256,1,1280)==-1);
    for(int mode=0;mode<=1;mode++) {
        assert(!native_cli_context(256,0,mode,256));
        assert(native_cli_context(256,1,mode,256)==(mode?0:-1));
        assert(native_cli_context(257,0,mode,256)==-1);
    }
    assert(!native_cli_context(1,INT_MAX,1,INT_MAX)&&native_cli_context(1,INT_MAX,0,INT_MAX)==-1);
    assert(native_cli_context(SIZE_MAX,0,1,INT_MAX)==-1&&native_cli_context(0,1,1,256)==-1);
    assert(native_cli_context(1,-1,1,256)==-1&&native_cli_context(1,1,2,256)==-1&&native_cli_context(1,1,1,0)==-1);
}
static void io(void) {
    const unsigned char raw[]={0,195,169,'x'};FILE *file=tmpfile();assert(file&&fwrite(raw,1,sizeof(raw),file)==sizeof(raw)&&!fseek(file,0,SEEK_SET));
    unsigned char *out=NULL;size_t bytes=SIZE_MAX;
    assert(!native_cli_read(file,sizeof(raw),&out,&bytes)&&bytes==sizeof(raw)&&!memcmp(out,raw,bytes));free(out);
    assert(!fseek(file,0,SEEK_SET));out=NULL;bytes=SIZE_MAX;
    assert(native_cli_read(file,sizeof(raw)-1,&out,&bytes)==-1&&!out&&bytes==SIZE_MAX);assert(!fclose(file));
    file=tmpfile();assert(file);out=NULL;bytes=99;
    assert(!native_cli_read(file,1,&out,&bytes)&&!bytes);free(out);assert(!fclose(file));
    file=tmpfile();assert(file);for(int i=0;i<10000;i++)assert(fputc(i&255,file)!=EOF);rewind(file);
    assert(!native_cli_read(file,10000,&out,&bytes)&&bytes==10000);
    for(size_t i=0;i<bytes;i++){assert(out[i]==(unsigned char)i);}free(out);assert(!fclose(file));
    file=tmpfile();assert(file);const int ids[]={0,1,0x12345678};const float floats[]={1.0f,-0.0f};
    assert(!native_cli_write(file,ids,3,0)&&!native_cli_write(file,floats,2,1));rewind(file);
    unsigned char got[20];const unsigned char expected[]={0,0,0,0,1,0,0,0,0x78,0x56,0x34,0x12,0,0,0x80,0x3f,0,0,0,0x80};
    assert(fread(got,1,sizeof(got),file)==sizeof(got)&&!memcmp(got,expected,sizeof(got))&&!fclose(file));
    char path[32];assert(!native_cli_path(path,sizeof(path),"dir","file")&&!strcmp(path,"dir/file"));
    assert(native_cli_path(path,3,"dir","file")==-1);
    (void)native_cli_read_path;(void)native_cli_dump;
}
static void batch(void) {
    NativeCLIOptions o;
    char *args[]={"bitnet","--native-text","model","--profile","exl3-int4-f16","--batch-prompts","requests.json","--batch-live","3","--prefill-tokens","513"};
    assert(!native_cli_parse(11,args,&o)&&o.batch_live==3&&o.batch_budget==1539&&!strcmp(o.batch_file,"requests.json"));
    char *wide[]={"bitnet","--native-text","model","--profile","exl3-int4-f16","--batch-prompts","x","--batch-live","16","--prefill-tokens","2048"};
    assert(!native_cli_parse(11,wide,&o)&&o.batch_live==16&&o.batch_budget==32768);
    NativeCLIOptions before=o;
    char *invalid[][14]={
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--batch-prompts","x"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","-p","hello","--batch-live","2"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","--batch-prompts","x","--batch-live","17"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","--batch-prompts","x","--batch-live","3","--batch-token-budget","2"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","--batch-prompts","x","-n","0"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","--batch-prompts","x","--prepare-only"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","--batch-prompts","x","--token-ids"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","--batch-prompts","x","--prefill-tokens","8193"},
        {"bitnet","--native-text","model","--profile","exl3-int4-f16","--batch-prompts","x","--batch-token-budget","131073"},
    };
    for(size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++) {
        int n=0;while(n<14&&invalid[i][n])n++;
        assert(native_cli_parse(n,invalid[i],&o)==-1&&!memcmp(&o,&before,sizeof(o)));
    }
    NativeCLIBatchInput input={0};const char *good="[\"hello\\nworld\",\"été 😀\"]";
    assert(!native_cli_batch_input_init(&input,good,strlen(good))&&input.count==2);
    assert(!strcmp(input.prompts[0],"hello\nworld")&&!strcmp(input.prompts[1],"été 😀"));
    assert(native_cli_batch_input_init(&input,good,strlen(good))==-1&&input.count==2);
    bn_json_free(&input.json);memset(&input,0,sizeof(input));
    const char *bad[]={"[]","{}","[1]","[\"\"]","[true]","[\"x\\u0000y\"]","[\"x\",null]"};
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++)assert(native_cli_batch_input_init(&input,bad[i],strlen(bad[i]))==-1&&!input.json.state&&!input.count);
    char many[1024]="[";for(int i=0;i<65;i++)strcat(many,i?",\"x\"":"\"x\"");strcat(many,"]");
    assert(native_cli_batch_input_init(&input,many,strlen(many))==-1&&!input.json.state);
    FILE *out=tmpfile();assert(out);const unsigned char text[]={'a','"','\\','\n',0,195,169};
    assert(!native_cli_batch_json_text(out,text,sizeof(text)));rewind(out);
    char encoded[128]={0};size_t n=fread(encoded,1,sizeof(encoded)-1,out);
    assert(n==strlen("\"a\\\"\\\\\\u000a\\u0000é\"")&&!strcmp(encoded,"\"a\\\"\\\\\\u000a\\u0000é\""));assert(!fclose(out));
}
static void serving_options(void) {
    NativeCLIOptions o;char *args[]={"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","--batch-live","3"};
    assert(!native_cli_parse(8,args,&o)&&o.serve&&native_cli_batch_mode(&o)&&!o.batch_file&&o.batch_live==3&&o.batch_budget==1536);
    NativeCLIOptions before=o;
    char *bad[][12]={
        {"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","--batch-prompts","x"},
        {"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","-p","x"},
        {"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","--serve-stdio"},
        {"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","--prepare-only"},
        {"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","--token-ids"},
        {"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","-n","0"},
        {"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","--prefill-tokens","8193"},
    };
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++){int n=0;while(n<12&&bad[i][n])n++;assert(native_cli_parse(n,bad[i],&o)==-1&&!memcmp(&before,&o,sizeof(o)));}
}

static void mixed_options(void) {
    NativeCLIOptions o;
    char *args[]={"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","--batch-live","4","--batch-token-budget","16","--mixed-token-budget","8"};
    assert(!native_cli_parse(12,args,&o)&&o.mixed_budget==8);
    const char *good[]={"0","4","16"};
    for(size_t i=0;i<sizeof(good)/sizeof(good[0]);i++){args[11]=(char*)good[i];assert(!native_cli_parse(12,args,&o)&&o.mixed_budget==(i?i==1?4:16:0));}
    NativeCLIOptions before=o;
    const char *bad[]={"-1","3","17","131073","1x"};
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++){args[11]=(char*)bad[i];assert(native_cli_parse(12,args,&o)==-1&&!memcmp(&before,&o,sizeof(o)));}
    args[11]="8";assert(native_cli_parse(11,args,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    char *duplicate[]={"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","--mixed-token-budget","8","--mixed-token-budget","8"};
    assert(native_cli_parse(10,duplicate,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    char *single[]={"b","--native-text","m","--profile","exl3-int4-f16","-p","hello","--mixed-token-budget","0"};
    assert(native_cli_parse(9,single,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    char *file[]={"b","--native-text","m","--profile","exl3-int4-f16","--batch-prompts","p.json","--mixed-token-budget","8"};
    assert(!native_cli_parse(9,file,&o)&&o.mixed_budget==8);
}

static void page_options(void) {
    NativeCLIOptions o;
    char *args[]={"b","--native-text","m","--profile","exl3-int4-f16","-p","hello","--ram-page-policy","system"};
    assert(!native_cli_parse(9,args,&o)&&o.ram_system_pages);
    args[8]="huge";assert(!native_cli_parse(9,args,&o)&&!o.ram_system_pages);
    assert(!native_cli_parse(7,args,&o)&&!o.ram_system_pages);
    NativeCLIOptions before=o;
    const char *bad[]={"base","always","SYSTEM","1",""};
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++) {
        args[8]=(char*)bad[i];
        assert(native_cli_parse(9,args,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    }
    assert(native_cli_parse(8,args,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    char *duplicate[]={"b","--native-text","m","--profile","exl3-int4-f16","-p","hello","--ram-page-policy","huge","--ram-page-policy","system"};
    assert(native_cli_parse(11,duplicate,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    char *disabled[]={"b","--native-text","m","--profile","exl3-int4-f16","-p","hello","--ram-page-policy","system","--ram-cache-mb","0"};
    assert(native_cli_parse(11,disabled,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    disabled[8]="huge";assert(native_cli_parse(11,disabled,&o)==-1&&!memcmp(&before,&o,sizeof(o)));
    char *serve[]={"b","--native-text","m","--profile","exl3-int4-f16","--serve-stdio","--ram-page-policy","system"};
    assert(!native_cli_parse(8,serve,&o)&&o.serve&&o.ram_system_pages);
    char *batch[]={"b","--native-text","m","--profile","exl3-int4-f16","--batch-prompts","requests.json","--ram-page-policy","system"};
    assert(!native_cli_parse(9,batch,&o)&&o.batch_file&&o.ram_system_pages);
}

int main(void){parser();io();batch();serving_options();mixed_options();page_options();puts("Native CLI options, batch input, bounded counted IO, payload arithmetic and LE traces PASSED");return 0;}
