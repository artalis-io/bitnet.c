#include "json.h"
#include <stdlib.h>
#include <string.h>
/* UTF-8/string decoding shares the strict storage reader's rules. */
typedef struct {const unsigned char *p,*end;} Json;
static void whitespace(Json *j) {
    while(j->p<j->end&&(*j->p==' '||*j->p=='\t'||*j->p=='\r'||*j->p=='\n'))j->p++;
}
static int take(Json *j,unsigned char c) {
    whitespace(j);if(j->p==j->end||*j->p!=c)return -1;j->p++;return 0;
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

struct BnJSONValue {
    BnJSONType type;
    char *key, *text;
    int boolean;
    size_t count, capacity;
    struct BnJSONValue **children;
};
typedef struct { Json json; size_t nodes,max_nodes; } Parser;
static void value_free(BnJSONValue *v) {
    if (!v) return;
    for (size_t i=0;i<v->count;i++) value_free(v->children[i]);
    free(v->children);free(v->key);free(v->text);free(v);
}
static int key_compare(const void *a,const void *b) {
    const BnJSONValue *const *x=a,*const *y=b;
    return strcmp((*x)->key,(*y)->key);
}
static int append(BnJSONValue *v,BnJSONValue *child,size_t limit) {
    if(v->count==v->capacity) {
        size_t capacity=v->capacity?(v->capacity>limit/2?limit:v->capacity*2):8;
        if(capacity>limit)capacity=limit;
        if(capacity<=v->count || capacity>SIZE_MAX/sizeof(*v->children))return -1;
        void *next=realloc(v->children,capacity*sizeof(*v->children));
        if(!next)return -1;
        v->children=next;v->capacity=capacity;
    }
    v->children[v->count++]=child;return 0;
}
static int numeric(Json *j,char **out) {
    const unsigned char *start=j->p;
    if(j->p<j->end&&*j->p=='-')j->p++;
    if(j->p==j->end)return -1;
    if(*j->p=='0')j->p++;
    else {
        if(*j->p<'1'||*j->p>'9')return -1;
        do {j->p++;}while(j->p<j->end&&*j->p>='0'&&*j->p<='9');
    }
    if(j->p<j->end&&*j->p=='.') {
        j->p++;const unsigned char *fraction=j->p;
        while(j->p<j->end&&*j->p>='0'&&*j->p<='9')j->p++;
        if(j->p==fraction)return -1;
    }
    if(j->p<j->end&&(*j->p=='e'||*j->p=='E')) {
        j->p++;if(j->p<j->end&&(*j->p=='+'||*j->p=='-'))j->p++;
        const unsigned char *exponent=j->p;
        while(j->p<j->end&&*j->p>='0'&&*j->p<='9')j->p++;
        if(j->p==exponent)return -1;
    }
    size_t bytes=(size_t)(j->p-start);char *text=malloc(bytes+1);if(!text)return -1;
    memcpy(text,start,bytes);text[bytes]=0;*out=text;return 0;
}
static BnJSONValue *parse_value(Parser *p,unsigned depth) {
    Json *j=&p->json;whitespace(j);
    if(depth>BN_JSON_MAX_DEPTH || p->nodes==p->max_nodes || j->p==j->end)return NULL;
    BnJSONValue *v=calloc(1,sizeof(*v));if(!v)return NULL;p->nodes++;
    unsigned char c=*j->p;
    if(c=='{'||c=='[') {
        int object=c=='{';unsigned char close=object?'}':']';j->p++;
        v->type=object?BN_JSON_OBJECT:BN_JSON_ARRAY;whitespace(j);
        if(j->p<j->end&&*j->p==close){j->p++;return v;}
        for(;;) {
            char *key=NULL;
            if(object&&(string(j,&key)||take(j,':'))){free(key);goto fail;}
            BnJSONValue *child=parse_value(p,depth+1);
            if(!child){free(key);goto fail;}
            child->key=key;
            if(append(v,child,p->max_nodes)){value_free(child);goto fail;}
            whitespace(j);if(j->p==j->end)goto fail;
            if(*j->p==close){j->p++;break;}
            if(take(j,','))goto fail;
        }
        if(object&&v->count>1) {
            qsort(v->children,v->count,sizeof(*v->children),key_compare);
            for(size_t i=1;i<v->count;i++)if(!strcmp(v->children[i-1]->key,v->children[i]->key))goto fail;
        }
    } else if(c=='"') {
        v->type=BN_JSON_STRING;if(string(j,&v->text))goto fail;
    } else if(c=='-'||(c>='0'&&c<='9')) {
        v->type=BN_JSON_NUMBER;if(numeric(j,&v->text))goto fail;
    } else {
        const char *word=c=='n'?"null":c=='t'?"true":c=='f'?"false":NULL;
        if(!word)goto fail;
        size_t n=strlen(word);if((size_t)(j->end-j->p)<n||memcmp(j->p,word,n))goto fail;
        j->p+=n;v->type=c=='n'?BN_JSON_NULL:BN_JSON_BOOL;v->boolean=c=='t';
    }
    return v;
fail:
    value_free(v);return NULL;
}
int bn_json_init_limits(BnJSON *document,const void *data,size_t bytes,const BnJSONLimits *limits) {
    if(!document||document->state||!limits||!limits->max_bytes||!limits->max_nodes||
        !data||!bytes||bytes>limits->max_bytes||bytes>PTRDIFF_MAX)return -1;
    Parser p={0};p.max_nodes=limits->max_nodes;
    p.json.p=data;p.json.end=p.json.p+bytes;
    BnJSONValue *root=parse_value(&p,0);if(!root)return -1;
    whitespace(&p.json);if(p.json.p!=p.json.end){value_free(root);return -1;}
    document->state=root;return 0;
}
int bn_json_init(BnJSON *document,const void *data,size_t bytes) {
    const BnJSONLimits limits={BN_JSON_MAX_BYTES,BN_JSON_MAX_NODES};
    return bn_json_init_limits(document,data,bytes,&limits);
}
void bn_json_free(BnJSON *document) {
    if(document){value_free(document->state);document->state=NULL;}
}
const BnJSONValue *bn_json_root(const BnJSON *document){return document?document->state:NULL;}
BnJSONType bn_json_type(const BnJSONValue *v){return v?v->type:BN_JSON_INVALID;}
const BnJSONValue *bn_json_get(const BnJSONValue *v,const char *key) {
    if(!v||v->type!=BN_JSON_OBJECT||!key)return NULL;
    size_t lo=0,hi=v->count;
    while(lo<hi){size_t mid=lo+(hi-lo)/2;int c=strcmp(key,v->children[mid]->key);
        if(c==0)return v->children[mid];
        if(c<0)hi=mid;else lo=mid+1;
    }
    return NULL;
}
const BnJSONValue *bn_json_at(const BnJSONValue *v,size_t i){return v&&v->type==BN_JSON_ARRAY&&i<v->count?v->children[i]:NULL;}
const BnJSONValue *bn_json_member(const BnJSONValue *v,size_t i,const char **key) {
    if(!v||v->type!=BN_JSON_OBJECT||i>=v->count||!key)return NULL;
    *key=v->children[i]->key;return v->children[i];
}
size_t bn_json_size(const BnJSONValue *v){return v&&(v->type==BN_JSON_ARRAY||v->type==BN_JSON_OBJECT)?v->count:0;}
const char *bn_json_text(const BnJSONValue *v){return v&&(v->type==BN_JSON_STRING||v->type==BN_JSON_NUMBER)?v->text:NULL;}
int bn_json_u64(const BnJSONValue *v,uint64_t *out) {
    if(!v||v->type!=BN_JSON_NUMBER||!out)return -1;
    const unsigned char *s=(const unsigned char*)v->text;uint64_t n=0;
    for(;*s;s++){if(*s<'0'||*s>'9')return -1;unsigned digit=*s-'0';if(n>(UINT64_MAX-digit)/10)return -1;n=n*10+digit;}
    *out=n;return 0;
}
int bn_json_bool(const BnJSONValue *v,int *out){if(!v||v->type!=BN_JSON_BOOL||!out)return -1;*out=v->boolean;return 0;}
