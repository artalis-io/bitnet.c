#include "tokenizer_map_data.h"
typedef struct {
    int encoded;
    const unsigned char *text;
    size_t bytes,position;
    TokenizerVocabBytes stream;
} TokenizerMapInput;
typedef struct {size_t bytes,chars,start,end;uint32_t first;} TokenizerMapTrim;
typedef struct {
    BnTokenizerMapInfo info;
    size_t slots;
    size_t *offsets;
    uint32_t *ids,*hash;
    unsigned char *keys;
} TokenizerMapState;
static TokenizerMapInput tokenizer_map_entry_input(const TokenizerVocabEntry *e) {
    TokenizerMapInput input={0};input.encoded=1;input.stream.raw=(const unsigned char*)e->raw;input.stream.length=e->raw_bytes;return input;
}
static int tokenizer_map_next(TokenizerMapInput *input,uint32_t *cp) {
    if(input->encoded) {
        unsigned char unit[4];unsigned n;int rc=tokenizer_vocab_unit(&input->stream,unit,&n);if(rc<=0)return rc;
        size_t pos=0;*cp=tokenizer_nfc_read(unit,n,&pos);return *cp<=0x10ffff?1:-1;
    }
    if(input->position==input->bytes)return 0;
    *cp=tokenizer_nfc_read(input->text,input->bytes,&input->position);return *cp<=0x10ffff?1:-1;
}
static void tokenizer_map_fold(uint32_t cp,uint32_t local[3],const uint32_t **values,unsigned *count) {
    if(cp>=0xac00&&cp<0xd7a4) {
        uint32_t n=cp-0xac00;local[0]=0x1100+n/588;local[1]=0x1161+n%588/28;*count=2;
        if(n%28){local[2]=0x11a7+n%28;*count=3;}*values=local;return;
    }
    size_t lo=0,hi=sizeof(tokenizer_map_folds)/sizeof(tokenizer_map_folds[0]);
    while(lo<hi){size_t mid=lo+(hi-lo)/2;if(tokenizer_map_folds[mid].cp<cp)lo=mid+1;else hi=mid;}
    if(lo<sizeof(tokenizer_map_folds)/sizeof(tokenizer_map_folds[0])&&tokenizer_map_folds[lo].cp==cp) {
        *values=tokenizer_map_fold_values+tokenizer_map_folds[lo].offset;*count=tokenizer_map_folds[lo].count;
    }else {local[0]=cp;*values=local;*count=1;}
}
static int tokenizer_map_white(uint32_t cp) {
    size_t lo=0,hi=sizeof(tokenizer_map_white_space)/sizeof(tokenizer_map_white_space[0]);
    while(lo<hi){size_t mid=lo+(hi-lo)/2;if(tokenizer_map_white_space[mid]<cp)lo=mid+1;else hi=mid;}
    return lo<sizeof(tokenizer_map_white_space)/sizeof(tokenizer_map_white_space[0])&&tokenizer_map_white_space[lo]==cp;
}
static unsigned tokenizer_map_utf8(uint32_t cp,unsigned char out[4]) {
    if(cp<0x80){out[0]=(unsigned char)cp;return 1;}
    if(cp<0x800){out[0]=(unsigned char)(0xc0|(cp>>6));out[1]=(unsigned char)(0x80|(cp&63));return 2;}
    if(cp<0x10000){out[0]=(unsigned char)(0xe0|(cp>>12));out[1]=(unsigned char)(0x80|((cp>>6)&63));out[2]=(unsigned char)(0x80|(cp&63));return 3;}
    out[0]=(unsigned char)(0xf0|(cp>>18));out[1]=(unsigned char)(0x80|((cp>>12)&63));out[2]=(unsigned char)(0x80|((cp>>6)&63));out[3]=(unsigned char)(0x80|(cp&63));return 4;
}
/* NFKD/mark removal leaves only canonical-class-zero characters. The generic
 * per-scalar fold therefore streams exactly; lowercase is per character.
 * Trimming precedes the final GLOBAL PUA replacement, including literal PUA. */
static int tokenizer_map_run(TokenizerMapInput input,unsigned char *out,size_t start,size_t end,TokenizerMapTrim *result) {
    TokenizerMapTrim t={0};t.start=SIZE_MAX;uint32_t cp;int rc,previous_space=0;
    while((rc=tokenizer_map_next(&input,&cp))>0) {
        uint32_t local[3];const uint32_t *values;unsigned count;tokenizer_map_fold(cp,local,&values,&count);
        for(unsigned i=0;i<count;i++) {
            uint32_t c=values[i];int ascii_space=c==32||c==9||c==10||c==13;
            if(ascii_space&&previous_space)continue;
            previous_space=ascii_space;if(ascii_space)c=32;
            unsigned char utf8[4];unsigned n=tokenizer_map_utf8(c==0xe000?32:c,utf8);
            if(t.bytes>SIZE_MAX-n)return -1;
            if(!t.chars)t.first=c;
            t.chars++;
            if(!tokenizer_map_white(c)){if(t.start==SIZE_MAX)t.start=t.bytes;t.end=t.bytes+n;}
            if(out&&t.bytes>=start&&t.bytes<end)memcpy(out+t.bytes-start,utf8,n);
            t.bytes+=n;
        }
    }
    if(rc<0)return -1;
    *result=t;return 0;
}
static int tokenizer_map_normalize(TokenizerMapInput input,unsigned char *out,size_t capacity,size_t *written) {
    TokenizerMapTrim t;if(!written||tokenizer_map_run(input,NULL,0,0,&t))return -1;
    int single_space=t.chars==1&&t.first==32;
    size_t n=single_space?1:t.start==SIZE_MAX?0:t.end-t.start;
    if(out) {
        if(capacity<n)return -1;
        if(single_space)out[0]=32;
        else if(n){TokenizerMapTrim check;if(tokenizer_map_run(input,out,t.start,t.end,&check))return -1;}
    }
    *written=n;return 0;
}
static int tokenizer_map_spec(const BnTokenizerVocab *v,const BnTokenizerMapSpec *s) {
    return v&&v->state&&s&&s->profile==BN_TOKENIZER_MAP_ENGRAM_V41&&s->vocab_size==bn_tokenizer_vocab_size(v)&&
        s->compressed_vocab>0&&s->compressed_vocab<=s->vocab_size&&s->pad_token>=0&&s->pad_token<s->vocab_size;
}
/* kind0: raw-token fallback, kind1: empty-normalization decoded fallback,
 * kind2: nonempty normalized bytes. Counted keys can contain embedded NUL. */
static int tokenizer_map_measure_key(const TokenizerVocabEntry *e,size_t *bytes,int *kind) {
    if(e->replacement){*bytes=e->raw_bytes;*kind=0;return 0;}
    size_t n;if(tokenizer_map_normalize(tokenizer_map_entry_input(e),NULL,0,&n))return -1;
    *kind=n?2:1;*bytes=n?n:e->decoded_bytes;return 0;
}
int bn_tokenizer_map_plan(const BnTokenizerVocab *v,const BnTokenizerMapSpec *spec,BnTokenizerMapWorkspace *out) {
    if(!out||!tokenizer_map_spec(v,spec))return -1;
    const TokenizerVocabState *vs=v->state;size_t n=(size_t)vs->count,key_bytes=0;
    for(size_t i=0;i<n;i++){size_t bytes;int kind;if(tokenizer_map_measure_key(vs->entries+i,&bytes,&kind)||bytes>SIZE_MAX-key_bytes)return -1;key_bytes+=bytes;}
    if(n>SIZE_MAX/2||n==SIZE_MAX)return -1;
    size_t slots=1;while(slots<2*n){if(slots>SIZE_MAX/2)return -1;slots*=2;}
    size_t host=sizeof(TokenizerMapState);
    if(n+1>(SIZE_MAX-host)/sizeof(size_t))return -1;
    host+=(n+1)*sizeof(size_t);
    if(n>(SIZE_MAX-host)/sizeof(uint32_t))return -1;
    host+=n*sizeof(uint32_t);
    if(slots>(SIZE_MAX-host)/sizeof(uint32_t))return -1;
    host+=slots*sizeof(uint32_t);
    if(key_bytes==SIZE_MAX||key_bytes+1>SIZE_MAX-host)return -1;
    host+=key_bytes+1;
    *out=(BnTokenizerMapWorkspace){host,key_bytes,slots};return 0;
}
static uint64_t tokenizer_map_hash(const unsigned char *key,size_t n) {
    uint64_t hash=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++){hash^=key[i];hash*=UINT64_C(1099511628211);}return hash;
}
int bn_tokenizer_map_init(BnTokenizerMap *map,const BnTokenizerVocab *v,const BnTokenizerMapSpec *spec,size_t budget) {
    BnTokenizerMapWorkspace w;if(!map||map->state||bn_tokenizer_map_plan(v,spec,&w)||budget<w.host_bytes)return -1;
    TokenizerMapState *s=calloc(1,w.host_bytes);if(!s)return -1;size_t n=(size_t)spec->vocab_size;
    s->offsets=(size_t*)(s+1);s->ids=(uint32_t*)(s->offsets+n+1);s->hash=s->ids+n;s->keys=(unsigned char*)(s->hash+w.hash_slots);s->slots=w.hash_slots;
    s->info=(BnTokenizerMapInfo){spec->vocab_size,0,0,w.host_bytes,w.key_bytes};const TokenizerVocabState *vs=v->state;
    size_t used=0;uint32_t unique=0;
    for(size_t i=0;i<n;i++) {
        const TokenizerVocabEntry *e=vs->entries+i;size_t bytes;int kind;
        if(tokenizer_map_measure_key(e,&bytes,&kind)||bytes>w.key_bytes-used)goto fail;
        s->offsets[i]=used;unsigned char *dst=s->keys+used;
        if(kind==0)memcpy(dst,e->raw,bytes);
        else if(kind==1){BnTokenizerVocabDecodeInfo info;if(tokenizer_vocab_decode_raw(e->raw,e->raw_bytes,dst,&info)||info.bytes!=bytes)goto fail;}
        else {size_t written;if(tokenizer_map_normalize(tokenizer_map_entry_input(e),dst,bytes,&written)||written!=bytes)goto fail;}
        used+=bytes;s->offsets[i+1]=used;size_t slot=(size_t)tokenizer_map_hash(dst,bytes)&(w.hash_slots-1);
        while(s->hash[slot]) {
            size_t prior=(size_t)s->hash[slot]-1,len=s->offsets[prior+1]-s->offsets[prior];
            if(len==bytes&&!memcmp(s->keys+s->offsets[prior],dst,bytes))break;
            slot=(slot+1)&(w.hash_slots-1);
        }
        if(!s->hash[slot]) {
            if(unique>=(uint32_t)spec->compressed_vocab)goto fail;
            s->hash[slot]=(uint32_t)i+1;s->ids[i]=unique++;
        }else s->ids[i]=s->ids[s->hash[slot]-1];
    }
    if(used!=w.key_bytes||unique!=(uint32_t)spec->compressed_vocab)goto fail;
    s->info.compressed_vocab=(int)unique;s->info.mapped_pad=s->ids[spec->pad_token];map->state=s;return 0;
fail:free(s);return -1;
}
void bn_tokenizer_map_free(BnTokenizerMap *map){if(map){free(map->state);map->state=NULL;}}
const uint32_t *bn_tokenizer_map_ids(const BnTokenizerMap *map){const TokenizerMapState *s=map?map->state:NULL;return s?s->ids:NULL;}
int bn_tokenizer_map_info(const BnTokenizerMap *map,BnTokenizerMapInfo *out) {
    const TokenizerMapState *s=map?map->state:NULL;if(!s||!out)return -1;*out=s->info;return 0;
}
int bn_tokenizer_map_key(const BnTokenizerMap *map,int token,BnTokenizerMapKey *out) {
    const TokenizerMapState *s=map?map->state:NULL;if(!s||!out||token<0||token>=s->info.vocab_size)return -1;
    *out=(BnTokenizerMapKey){s->keys+s->offsets[token],s->offsets[token+1]-s->offsets[token]};return 0;
}
