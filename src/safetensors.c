#include "safetensors.h"
#include "platform.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __EMSCRIPTEN__
#include <unistd.h>
#endif

/* Format: https://github.com/safetensors/safetensors (tensor.rs).
 * This parser owns strings, checks integer/UTF-8 syntax, then independently
 * validates dtype extents and sorted contiguous payload coverage. */
typedef struct {const char *name; unsigned bits;} Dtype;
static const Dtype dtypes[BN_SAFETENSORS_DTYPE_COUNT] = {
    {"BOOL",8},{"U8",8},{"I8",8},{"F8_E5M2",8},{"F8_E4M3",8},{"F8_E8M0",8},
    {"F8_E4M3FNUZ",8},{"F8_E5M2FNUZ",8},{"I16",16},{"U16",16},{"F16",16},
    {"BF16",16},{"I32",32},{"U32",32},{"F32",32},{"C64",64},{"F64",64},
    {"I64",64},{"U64",64},{"F4",4},{"F6_E2M3",6},{"F6_E3M2",6}
};
typedef struct {char *key,*value;} Metadata;
typedef struct {
    BnMappedFile map;
    size_t payload, count, capacity, metadata_count, metadata_capacity;
    BnSafetensorsTensor *tensors;
    Metadata *metadata;
} File;
typedef struct {const unsigned char *p,*end;} Json;

static int error(void) {fprintf(stderr,"safetensors: invalid or unsupported container/read\n");return -1;}
const char *bn_safetensors_dtype_name(BnSafetensorsDtype dtype) {
    return (unsigned)dtype<BN_SAFETENSORS_DTYPE_COUNT?dtypes[dtype].name:NULL;
}
unsigned bn_safetensors_dtype_bits(BnSafetensorsDtype dtype) {
    return (unsigned)dtype<BN_SAFETENSORS_DTYPE_COUNT?dtypes[dtype].bits:0;
}
static void whitespace(Json *j) {
    while(j->p<j->end&&(*j->p==' '||*j->p=='\t'||*j->p=='\r'||*j->p=='\n'))j->p++;
}
static int take(Json *j,unsigned char c) {
    whitespace(j);if(j->p==j->end||*j->p!=c)return -1;j->p++;return 0;
}
static int number(Json *j,uint64_t *out) {
    whitespace(j);if(j->p==j->end||*j->p<'0'||*j->p>'9')return -1;
    uint64_t v=0;int zero=*j->p=='0';const unsigned char *start=j->p;
    while(j->p<j->end&&*j->p>='0'&&*j->p<='9') {
        unsigned n=*j->p++-'0';if(v>(UINT64_MAX-n)/10)return -1;v=v*10+n;
    }
    if(zero&&j->p-start>1)return -1;
    *out=v;return 0;
}
static int hex4(Json *j,unsigned *value) {
    if((size_t)(j->end-j->p)<4)return -1;
    unsigned v=0;
    for(int i=0;i<4;i++) {
        unsigned c=*j->p++,n;
        if(c>='0'&&c<='9')n=c-'0';else if(c>='a'&&c<='f')n=c-'a'+10;
        else if(c>='A'&&c<='F')n=c-'A'+10;else return -1;
        v=(v<<4)|n;
    }
    *value=v;return 0;
}
static size_t encode(unsigned cp,char *dst) {
    if(cp<128){dst[0]=(char)cp;return 1;}
    if(cp<2048){dst[0]=(char)(0xc0|(cp>>6));dst[1]=(char)(0x80|(cp&63));return 2;}
    if(cp<65536){dst[0]=(char)(0xe0|(cp>>12));dst[1]=(char)(0x80|((cp>>6)&63));dst[2]=(char)(0x80|(cp&63));return 3;}
    dst[0]=(char)(0xf0|(cp>>18));dst[1]=(char)(0x80|((cp>>12)&63));
    dst[2]=(char)(0x80|((cp>>6)&63));dst[3]=(char)(0x80|(cp&63));return 4;
}
static int string(Json *j,char **out) {
    if(take(j,'"'))return -1;
    /* Allocate only the serialized string's span, not the entire header. */
    const unsigned char *scan=j->p;int escaped=0;
    while(scan<j->end) {
        unsigned c=*scan++;
        if(!escaped&&c=='"')break;
        if(!escaped&&c=='\\')escaped=1;else escaped=0;
    }
    if(scan==j->p||scan[-1]!='"')return -1;
    size_t capacity=(size_t)(scan-j->p);char *s=malloc(capacity);if(!s)return -1;
    size_t used=0;
    while(j->p<j->end) {
        unsigned cp=*j->p++;
        if(cp=='"'){s[used]=0;*out=s;return 0;}
        if(cp<32)goto invalid;
        if(cp=='\\') {
            if(j->p==j->end)goto invalid;
            unsigned c=*j->p++;
            switch(c) {
                case '"':case '\\':case '/':cp=c;break;
                case 'b':cp=8;break;case 'f':cp=12;break;case 'n':cp=10;break;
                case 'r':cp=13;break;case 't':cp=9;break;
                case 'u':
                    if(hex4(j,&cp))goto invalid;
                    if(cp>=0xd800&&cp<=0xdbff) {
                        unsigned low;
                        if((size_t)(j->end-j->p)<6||j->p[0]!='\\'||j->p[1]!='u')goto invalid;
                        j->p+=2;if(hex4(j,&low)||low<0xdc00||low>0xdfff)goto invalid;
                        cp=0x10000+((cp-0xd800)<<10)+(low-0xdc00);
                    } else if(cp>=0xdc00&&cp<=0xdfff)goto invalid;
                    if(!cp)goto invalid;
                    break;
                default:goto invalid;
            }
        } else if(cp>=128) {
            int n;unsigned minimum;
            if(cp>=0xc2&&cp<=0xdf){n=1;minimum=128;cp&=31;}
            else if(cp>=0xe0&&cp<=0xef){n=2;minimum=2048;cp&=15;}
            else if(cp>=0xf0&&cp<=0xf4){n=3;minimum=65536;cp&=7;}
            else goto invalid;
            if((size_t)(j->end-j->p)<(size_t)n)goto invalid;
            for(int i=0;i<n;i++){unsigned c=*j->p++;if((c&0xc0)!=0x80)goto invalid;cp=(cp<<6)|(c&63);}
            if(cp<minimum||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff))goto invalid;
        }
        char bytes[4];size_t n=encode(cp,bytes);
        if(n>=capacity-used)goto invalid;
        memcpy(s+used,bytes,n);used+=n;
    }
invalid:
    free(s);return -1;
}
static int array(Json *j,uint64_t *out,size_t maximum,size_t *count) {
    if(take(j,'['))return -1;
    size_t n=0;whitespace(j);
    if(j->p<j->end&&*j->p==']'){j->p++;*count=0;return 0;}
    for(;;) {
        if(n==maximum||number(j,&out[n++]))return -1;
        whitespace(j);if(j->p==j->end)return -1;
        if(*j->p==']'){j->p++;*count=n;return 0;}
        if(take(j,','))return -1;
    }
}
static int tensor(Json *j,BnSafetensorsTensor *t) {
    if(take(j,'{'))return -1;
    unsigned fields=0;
    for(;;) {
        char *key=NULL;if(string(j,&key))return -1;
        unsigned bit=!strcmp(key,"dtype")?1:!strcmp(key,"shape")?2:!strcmp(key,"data_offsets")?4:0;
        free(key);if(!bit||(fields&bit)||take(j,':'))return -1;fields|=bit;
        if(bit==1) {
            char *name=NULL;if(string(j,&name))return -1;
            unsigned i;for(i=0;i<BN_SAFETENSORS_DTYPE_COUNT;i++)if(!strcmp(name,dtypes[i].name))break;
            free(name);if(i==BN_SAFETENSORS_DTYPE_COUNT)return -1;t->dtype=(BnSafetensorsDtype)i;
        } else if(bit==2) {
            size_t n;if(array(j,t->shape,BN_SAFETENSORS_MAX_DIMS,&n))return -1;t->n_dims=(int)n;
        } else {
            uint64_t offsets[2];size_t n;
            if(array(j,offsets,2,&n)||n!=2||offsets[1]<offsets[0])return -1;
            t->offset=offsets[0];t->bytes=offsets[1]-offsets[0];
        }
        whitespace(j);if(j->p==j->end)return -1;
        if(*j->p=='}'){j->p++;return fields==7?0:-1;}
        if(take(j,','))return -1;
    }
}
static void *reserve(void *ptr,size_t *capacity,size_t count,size_t width) {
    if(count<*capacity)return ptr;
    if(count>=BN_SAFETENSORS_MAX_TENSORS)return NULL;
    size_t next=*capacity?*capacity*2:8;
    if(next>BN_SAFETENSORS_MAX_TENSORS)next=BN_SAFETENSORS_MAX_TENSORS;
    if(next>SIZE_MAX/width)return NULL;
    void *p=realloc(ptr,next*width);if(!p)return NULL;*capacity=next;return p;
}
static int metadata(Json *j,File *f) {
    if(take(j,'{'))return -1;
    whitespace(j);
    if(j->p<j->end&&*j->p=='}'){j->p++;return 0;}
    for(;;) {
        void *grown=reserve(f->metadata,&f->metadata_capacity,f->metadata_count,sizeof(*f->metadata));
        if(!grown)return -1;
        f->metadata=grown;
        Metadata *m=&f->metadata[f->metadata_count++];*m=(Metadata){0};
        if(string(j,&m->key)||take(j,':')||string(j,&m->value))return -1;
        whitespace(j);if(j->p==j->end)return -1;
        if(*j->p=='}'){j->p++;return 0;}
        if(take(j,','))return -1;
    }
}
static int by_name(const void *a,const void *b) {
    return strcmp(((const BnSafetensorsTensor*)a)->name,((const BnSafetensorsTensor*)b)->name);
}
static int by_metadata(const void *a,const void *b) {
    return strcmp(((const Metadata*)a)->key,((const Metadata*)b)->key);
}
static int by_offset(const void *a,const void *b) {
    const BnSafetensorsTensor *x=*(const BnSafetensorsTensor*const*)a,*y=*(const BnSafetensorsTensor*const*)b;
    if(x->offset!=y->offset)return x->offset<y->offset?-1:1;
    return x->bytes==y->bytes?0:x->bytes<y->bytes?-1:1;
}
static void destroy(File *f) {
    if(!f)return;
    for(size_t i=0;i<f->count;i++)free((void*)f->tensors[i].name);
    for(size_t i=0;i<f->metadata_count;i++){free(f->metadata[i].key);free(f->metadata[i].value);}
    free(f->metadata);free(f->tensors);bn_platform_unload_file(&f->map);free(f);
}
static int parse(File *f) {
    if(f->map.size<10)return -1;
    uint64_t header=0;for(int i=0;i<8;i++)header|=(uint64_t)f->map.data[i]<<(8*i);
    if(header<2||header>BN_SAFETENSORS_MAX_HEADER||header>f->map.size-8||f->map.data[8]!='{')return -1;
    f->payload=8+(size_t)header;
    Json j={f->map.data+8,f->map.data+f->payload};if(take(&j,'{'))return -1;
    int seen_metadata=0;whitespace(&j);
    if(j.p<j.end&&*j.p=='}')j.p++;
    else for(;;) {
        char *name=NULL;if(string(&j,&name))return -1;
        if(take(&j,':')){free(name);return -1;}
        if(!strcmp(name,"__metadata__")) {
            free(name);if(seen_metadata++||metadata(&j,f))return -1;
        } else {
            void *grown=reserve(f->tensors,&f->capacity,f->count,sizeof(*f->tensors));
            if(!grown){free(name);return -1;}
            f->tensors=grown;
            BnSafetensorsTensor *t=&f->tensors[f->count++];memset(t,0,sizeof(*t));t->name=name;
            if(tensor(&j,t))return -1;
        }
        whitespace(&j);if(j.p==j.end)return -1;
        if(*j.p=='}'){j.p++;break;}
        if(take(&j,','))return -1;
    }
    whitespace(&j);if(j.p!=j.end)return -1;
    if(f->metadata_count>1)qsort(f->metadata,f->metadata_count,sizeof(*f->metadata),by_metadata);
    for(size_t i=1;i<f->metadata_count;i++)if(!strcmp(f->metadata[i-1].key,f->metadata[i].key))return -1;
    if(f->count>1)qsort(f->tensors,f->count,sizeof(*f->tensors),by_name);
    for(size_t i=1;i<f->count;i++)if(!strcmp(f->tensors[i-1].name,f->tensors[i].name))return -1;
    const BnSafetensorsTensor **order=f->count?malloc(f->count*sizeof(*order)):NULL;
    if(f->count&&!order)return -1;
    int valid=1;uint64_t cursor=0;
    for(size_t i=0;i<f->count;i++) {
        BnSafetensorsTensor *t=&f->tensors[i];order[i]=t;
        uint64_t bits=bn_safetensors_dtype_bits(t->dtype);int empty=0;
        for(int k=0;k<t->n_dims;k++) {
            if(!t->shape[k])empty=1;
            else if(t->shape[k]>SIZE_MAX/bits){valid=0;break;}
            else bits*=t->shape[k];
        }
        if(empty)bits=0;
        if(bits%8||bits/8!=t->bytes)valid=0;
    }
    if(f->count>1)qsort(order,f->count,sizeof(*order),by_offset);
    for(size_t i=0;i<f->count;i++) {
        const BnSafetensorsTensor *t=order[i];
        if(t->offset!=cursor||t->bytes>UINT64_MAX-cursor){valid=0;break;}
        cursor+=t->bytes;
    }
    free(order);return valid&&cursor==f->map.size-f->payload?0:-1;
}
static int init(BnSafetensors *out,BnMappedFile map) {
    File *f=calloc(1,sizeof(*f));if(!f){bn_platform_unload_file(&map);return error();}
    f->map=map;if(!map.data||parse(f)){destroy(f);return error();}
    out->state=f;return 0;
}
int bn_safetensors_init_file(BnSafetensors *out,const char *path) {
    if(!out||out->state||!path)return error();
    return init(out,bn_platform_load_file(path));
}
int bn_safetensors_init_buffer(BnSafetensors *out,const void *data,size_t size) {
    if(!out||out->state||!data)return error();
    return init(out,bn_platform_load_buffer(data,size));
}
void bn_safetensors_free(BnSafetensors *file) {
    if(file){destroy(file->state);file->state=NULL;}
}
size_t bn_safetensors_count(const BnSafetensors *file) {
    const File *f=file?file->state:NULL;return f?f->count:0;
}
const BnSafetensorsTensor *bn_safetensors_tensor(const BnSafetensors *file,size_t index) {
    const File *f=file?file->state:NULL;return f&&index<f->count?&f->tensors[index]:NULL;
}
int bn_safetensors_find(const BnSafetensors *file,const char *name) {
    const File *f=file?file->state:NULL;if(!f||!name)return -1;
    size_t lo=0,hi=f->count;
    while(lo<hi){size_t mid=lo+(hi-lo)/2;int c=strcmp(name,f->tensors[mid].name);
        if(!c)return (int)mid;
        if(c<0)hi=mid;else lo=mid+1;}
    return -1;
}
const char *bn_safetensors_metadata(const BnSafetensors *file,const char *key) {
    const File *f=file?file->state:NULL;if(!f||!key)return NULL;
    size_t lo=0,hi=f->metadata_count;
    while(lo<hi){size_t mid=lo+(hi-lo)/2;int c=strcmp(key,f->metadata[mid].key);
        if(!c)return f->metadata[mid].value;
        if(c<0)hi=mid;else lo=mid+1;}
    return NULL;
}
int bn_safetensors_read(const BnSafetensors *file,size_t index,uint64_t offset,
    void *destination,size_t bytes) {
    const File *f=file?file->state:NULL;
    if(!f||index>=f->count||(!destination&&bytes))return error();
    const BnSafetensorsTensor *t=&f->tensors[index];
    if(offset>t->bytes||bytes>t->bytes-offset)return error();
    if(!bytes)return 0;
    size_t absolute=f->payload+(size_t)t->offset+(size_t)offset;
#ifndef __EMSCRIPTEN__
    if(f->map.fd>=0) {
        size_t done=0;
        while(done<bytes) {
            size_t chunk=bytes-done;if(chunk>((size_t)1<<30))chunk=(size_t)1<<30;
            ssize_t n=pread(f->map.fd,(unsigned char*)destination+done,chunk,(off_t)(absolute+done));
            if(n<0&&errno==EINTR)continue;
            if(n<=0)return error();
            done+=(size_t)n;
        }
        return 0;
    }
#endif
    memmove(destination,f->map.data+absolute,bytes);return 0;
}

typedef struct {
    size_t shard,local;
    const BnSafetensorsTensor *tensor;
} IndexTensor;
typedef struct {
    size_t shards,count;
    uint64_t bytes;
    BnSafetensors *files;
    char **names;
    IndexTensor *tensors;
} Index;

static void index_destroy(Index *index) {
    if(!index)return;
    for(size_t i=0;i<index->shards;i++) {
        if(index->files)bn_safetensors_free(&index->files[i]);
        if(index->names)free(index->names[i]);
    }
    free(index->files);free(index->names);free(index->tensors);free(index);
}
static int index_total(Json *j,uint64_t *bytes) {
    char *key=NULL;
    if(take(j,'{')||string(j,&key))return -1;
    int valid=!strcmp(key,"total_size");free(key);
    return valid&&!take(j,':')&&!number(j,bytes)&&!take(j,'}')?0:-1;
}
static int index_header(File *manifest,uint64_t *bytes) {
    if(!manifest->map.data||manifest->map.size<2||manifest->map.size>BN_SAFETENSORS_MAX_HEADER)return -1;
    Json j={manifest->map.data,manifest->map.data+manifest->map.size};unsigned fields=0;
    if(take(&j,'{'))return -1;
    for(;;) {
        char *key=NULL;if(string(&j,&key))return -1;
        unsigned bit=!strcmp(key,"metadata")?1:!strcmp(key,"weight_map")?2:0;
        free(key);if(!bit||(fields&bit)||take(&j,':'))return -1;fields|=bit;
        if(bit==1?index_total(&j,bytes):metadata(&j,manifest))return -1;
        whitespace(&j);if(j.p==j.end)return -1;
        if(*j.p=='}'){j.p++;break;}
        if(take(&j,','))return -1;
    }
    whitespace(&j);if(fields!=3||j.p!=j.end)return -1;
    if(manifest->metadata_count>1)qsort(manifest->metadata,manifest->metadata_count,sizeof(*manifest->metadata),by_metadata);
    for(size_t i=1;i<manifest->metadata_count;i++)
        if(!strcmp(manifest->metadata[i-1].key,manifest->metadata[i].key))return -1;
    return 0;
}
static int by_path(const void *a,const void *b) {
    return strcmp(*(char*const*)a,*(char*const*)b);
}
static int by_index_name(const void *a,const void *b) {
    return strcmp(((const IndexTensor*)a)->tensor->name,((const IndexTensor*)b)->tensor->name);
}
static int shard_basename(const char *name) {
    size_t n=strlen(name),suffix=sizeof(".safetensors")-1;
    return n>=suffix&&!strchr(name,'/')&&!strchr(name,'\\')&&
        !strcmp(name+n-suffix,".safetensors");
}
int bn_safetensors_index_init(BnSafetensorsIndex *out,const char *path) {
    if(!out||out->state||!path)return error();
    File *manifest=calloc(1,sizeof(*manifest));Index *index=calloc(1,sizeof(*index));
    char **paths=NULL;uint64_t expected=0;
    if(!manifest||!index)goto invalid;
    manifest->map=bn_platform_load_file(path);
    if(index_header(manifest,&expected))goto invalid;
    size_t count=manifest->metadata_count;
    if(count) {
        paths=malloc(count*sizeof(*paths));if(!paths)goto invalid;
        for(size_t i=0;i<count;i++) {
            paths[i]=manifest->metadata[i].value;
            if(!shard_basename(paths[i]))goto invalid;
        }
        if(count>1)qsort(paths,count,sizeof(*paths),by_path);
        for(size_t i=0;i<count;i++)if(!i||strcmp(paths[i],paths[i-1]))paths[index->shards++]=paths[i];
        if(index->shards>BN_SAFETENSORS_MAX_SHARDS)goto invalid;
        index->files=calloc(index->shards,sizeof(*index->files));
        index->names=calloc(index->shards,sizeof(*index->names));
        index->tensors=malloc(count*sizeof(*index->tensors));
        if(!index->files||!index->names||!index->tensors)goto invalid;
    }
    const char *slash=strrchr(path,'/');size_t prefix=slash?(size_t)(slash-path)+1:0;
    for(size_t i=0;i<index->shards;i++) {
        size_t n=strlen(paths[i]);
        if(prefix>SIZE_MAX-n-1)goto invalid;
        char *full=malloc(prefix+n+1);index->names[i]=malloc(n+1);
        if(!full||!index->names[i]){free(full);goto invalid;}
        memcpy(index->names[i],paths[i],n+1);memcpy(full,path,prefix);memcpy(full+prefix,paths[i],n+1);
        int opened=bn_safetensors_init_file(&index->files[i],full);free(full);
        if(opened)goto invalid;
        size_t tensors=bn_safetensors_count(&index->files[i]);
        if(tensors>count-index->count)goto invalid;
        for(size_t k=0;k<tensors;k++) {
            const BnSafetensorsTensor *t=bn_safetensors_tensor(&index->files[i],k);
            if(t->bytes>UINT64_MAX-index->bytes)goto invalid;
            index->bytes+=t->bytes;
            index->tensors[index->count++]=(IndexTensor){i,k,t};
        }
    }
    if(index->count!=count||index->bytes!=expected)goto invalid;
    if(count>1)qsort(index->tensors,count,sizeof(*index->tensors),by_index_name);
    for(size_t i=0;i<count;i++) {
        IndexTensor *t=&index->tensors[i];Metadata *m=&manifest->metadata[i];
        if(strcmp(t->tensor->name,m->key)||strcmp(index->names[t->shard],m->value))goto invalid;
    }
    free(paths);destroy(manifest);out->state=index;return 0;
invalid:
    free(paths);destroy(manifest);index_destroy(index);return error();
}
void bn_safetensors_index_free(BnSafetensorsIndex *index) {
    if(index){index_destroy(index->state);index->state=NULL;}
}
size_t bn_safetensors_index_count(const BnSafetensorsIndex *handle) {
    const Index *index=handle?handle->state:NULL;return index?index->count:0;
}
size_t bn_safetensors_index_shards(const BnSafetensorsIndex *handle) {
    const Index *index=handle?handle->state:NULL;return index?index->shards:0;
}
uint64_t bn_safetensors_index_bytes(const BnSafetensorsIndex *handle) {
    const Index *index=handle?handle->state:NULL;return index?index->bytes:0;
}
const BnSafetensorsTensor *bn_safetensors_index_tensor(const BnSafetensorsIndex *handle,size_t tensor_id) {
    const Index *index=handle?handle->state:NULL;
    return index&&tensor_id<index->count?index->tensors[tensor_id].tensor:NULL;
}
const char *bn_safetensors_index_shard_name(const BnSafetensorsIndex *handle,size_t tensor_id) {
    const Index *index=handle?handle->state:NULL;
    return index&&tensor_id<index->count?index->names[index->tensors[tensor_id].shard]:NULL;
}
int bn_safetensors_index_find(const BnSafetensorsIndex *handle,const char *name) {
    const Index *index=handle?handle->state:NULL;if(!index||!name)return -1;
    size_t lo=0,hi=index->count;
    while(lo<hi) {
        size_t mid=lo+(hi-lo)/2;int c=strcmp(name,index->tensors[mid].tensor->name);
        if(!c)return (int)mid;
        if(c<0)hi=mid;else lo=mid+1;
    }
    return -1;
}
int bn_safetensors_index_read(const BnSafetensorsIndex *handle,size_t tensor_id,
    uint64_t offset,void *destination,size_t bytes) {
    const Index *index=handle?handle->state:NULL;
    if(!index||tensor_id>=index->count)return error();
    const IndexTensor *t=&index->tensors[tensor_id];
    return bn_safetensors_read(&index->files[t->shard],t->local,offset,destination,bytes);
}
