#include <stdlib.h>
_Static_assert(sizeof(float)==4,"native traces require IEEE binary32");
/* Counted bounded reads, including NUL; result publication is atomic. */
static int native_cli_read(FILE *file,size_t limit,unsigned char **out,size_t *bytes) {
    if(!file||!out||!bytes||!limit)return -1;
    size_t cap=limit<4096?limit:4096,n=0;unsigned char *data=malloc(cap);if(!data)return -1;
    for(;;) {
        size_t got=fread(data+n,1,cap-n,file);n+=got;
        if(ferror(file)){free(data);return -1;}
        if(n<cap)break;
        int extra=fgetc(file);
        if(extra==EOF){if(ferror(file)){free(data);return -1;}break;}
        if(cap==limit){free(data);return -1;}
        size_t next=cap>limit-cap?limit:cap*2;
        unsigned char *grown=realloc(data,next);if(!grown){free(data);return -1;}
        data=grown;data[n++]=(unsigned char)extra;cap=next;
    }
    *out=data;*bytes=n;return 0;
}
static int native_cli_read_path(const char *path,size_t limit,unsigned char **out,size_t *bytes) {
    FILE *file=fopen(path,"rb");if(!file)return -1;
    unsigned char *data=NULL;size_t n=0;int status=native_cli_read(file,limit,&data,&n);
    if(fclose(file)&&!status){free(data);return -1;}
    if(status)return -1;
    *out=data;*bytes=n;return 0;
}
static int native_cli_path(char *out,size_t cap,const char *directory,const char *name) {
    int n=snprintf(out,cap,"%s/%s",directory,name);return n<0||(size_t)n>=cap?-1:0;
}
/* Diagnostic files never replace an existing path. Values are explicitly LE. */
static int native_cli_write(FILE *file,const void *values,size_t count,int floats) {
    unsigned char buffer[4096];size_t pos=0;int status=0;
    while(pos<count) {
        size_t n=count-pos;if(n>sizeof(buffer)/4)n=sizeof(buffer)/4;
        for(size_t i=0;i<n;i++) {
            uint32_t v;
            if(floats)memcpy(&v,(const float*)values+pos+i,4);else v=(uint32_t)((const int*)values)[pos+i];
            for(unsigned j=0;j<4;j++)buffer[i*4+j]=(unsigned char)(v>>(j*8));
        }
        if(fwrite(buffer,4,n,file)!=n){status=-1;break;}pos+=n;
    }
    return status;
}
static int native_cli_dump(const char *path,const void *values,size_t count,int floats) {
    FILE *file=fopen(path,"wbx");if(!file)return -1;
    int status=native_cli_write(file,values,count,floats);
    if(fclose(file))status=-1;
    return status;
}
