/* Private immutable ByteLevel vocabulary view; JSON retains string ownership. */
typedef struct {
    const char *raw;
    size_t raw_bytes,decoded_bytes;
    unsigned char replacement,added;
} TokenizerVocabEntry;
typedef struct {
    size_t allocated;
    int count;
    TokenizerVocabEntry entries[];
} TokenizerVocabState;
typedef struct {
    const unsigned char *raw;
    size_t length,position;
    unsigned char pending[4],look[4];
    unsigned pending_count,pending_pos,look_count;
} TokenizerVocabBytes;
static int tokenizer_vocab_byte(TokenizerVocabBytes *s) {
    if(s->pending_pos<s->pending_count)return s->pending[s->pending_pos++];
    if(s->position==s->length)return -1;
    size_t start=s->position;uint32_t cp=tokenizer_nfc_read(s->raw,s->length,&s->position);
    if(cp>0x10ffff)return -2;
    if(cp<128||(cp>=0xa1&&cp<=0xac)||(cp>=0xae&&cp<=0xff))return (int)cp;
    if(cp>=BN_BPE_UNICODE_OFFSET&&cp<=BN_BPE_UNICODE_END)return bpe_n2b[cp-BN_BPE_UNICODE_OFFSET];
    s->pending_count=(unsigned)(s->position-start);s->pending_pos=1;
    memcpy(s->pending,s->raw+start,s->pending_count);return s->pending[0];
}
static int tokenizer_vocab_unit(TokenizerVocabBytes *s,unsigned char out[4],unsigned *length) {
    while(s->look_count<4) {
        int byte=tokenizer_vocab_byte(s);if(byte==-1)break;if(byte<0)return -1;
        s->look[s->look_count++]=(unsigned char)byte;
    }
    if(!s->look_count)return 0;
    unsigned a=s->look[0],width=a<128?1:a>=0xc2&&a<=0xdf?2:a>=0xe0&&a<=0xef?3:a>=0xf0&&a<=0xf4?4:0;
    unsigned consumed=1;int valid=width!=0;
    for(unsigned k=1;valid&&k<width;k++) {
        if(k>=s->look_count){valid=0;consumed=k;break;}
        unsigned byte=s->look[k];
        if((byte&0xc0)!=0x80||(k==1&&((a==0xe0&&byte<0xa0)||(a==0xed&&byte>0x9f)||
            (a==0xf0&&byte<0x90)||(a==0xf4&&byte>0x8f)))){valid=0;consumed=k;}
    }
    if(valid){consumed=width;*length=width;memcpy(out,s->look,width);}
    else {out[0]=0xef;out[1]=0xbf;out[2]=0xbd;*length=3;}
    memmove(s->look,s->look+consumed,s->look_count-consumed);s->look_count-=consumed;return 1;
}
static int tokenizer_vocab_decode_raw(const char *raw,size_t raw_bytes,unsigned char *out,BnTokenizerVocabDecodeInfo *info) {
    TokenizerVocabBytes stream={0};stream.raw=(const unsigned char*)raw;stream.length=raw_bytes;
    size_t bytes=0;int replacement=0,status;unsigned char unit[4];unsigned n;
    while((status=tokenizer_vocab_unit(&stream,unit,&n))>0) {
        if(bytes>SIZE_MAX-n)return -1;
        if(n==3&&unit[0]==0xef&&unit[1]==0xbf&&unit[2]==0xbd)replacement=1;
        if(out)memcpy(out+bytes,unit,n);
        bytes+=n;
    }
    if(status<0)return -1;
    *info=(BnTokenizerVocabDecodeInfo){bytes,replacement};return 0;
}
int bn_tokenizer_vocab_plan(int count,size_t *out) {
    if(!out||count<1||(size_t)count>(SIZE_MAX-sizeof(TokenizerVocabState))/sizeof(TokenizerVocabEntry))return -1;
    *out=sizeof(TokenizerVocabState)+(size_t)count*sizeof(TokenizerVocabEntry);return 0;
}
int bn_tokenizer_vocab_init_json(BnTokenizerVocab *v,const BnJSON *document,int count,size_t budget) {
    size_t bytes;if(!v||v->state||bn_tokenizer_vocab_plan(count,&bytes)||budget<bytes)return -1;
    const BnJSONValue *root=bn_json_root(document),*model=bn_json_get(root,"model"),*decoder=bn_json_get(root,"decoder");
    const BnJSONValue *core=bn_json_get(model,"vocab"),*added=bn_json_get(root,"added_tokens");
    const char *model_type=bn_json_text(bn_json_get(model,"type")),*decoder_type=bn_json_text(bn_json_get(decoder,"type"));
    if(!model_type||strcmp(model_type,"BPE")||!decoder_type||strcmp(decoder_type,"ByteLevel")||
        bn_json_type(core)!=BN_JSON_OBJECT||!bn_json_size(core)||bn_json_size(core)>(size_t)count||
        bn_json_type(added)!=BN_JSON_ARRAY||bn_json_size(added)>(size_t)count)return -1;
    TokenizerVocabState *s=calloc(1,bytes);if(!s)return -1;s->allocated=bytes;s->count=count;
    for(size_t i=0;i<bn_json_size(core);i++) {
        const char *piece;const BnJSONValue *value=bn_json_member(core,i,&piece);uint64_t id;
        if(!value||bn_json_u64(value,&id)||id>=(uint64_t)count||s->entries[id].raw)goto fail;
        s->entries[id].raw=piece;
    }
    for(size_t i=0;i<bn_json_size(added);i++) {
        const BnJSONValue *item=bn_json_at(added,i),*content=bn_json_get(item,"content");uint64_t id;
        if(bn_json_type(content)!=BN_JSON_STRING||bn_json_u64(bn_json_get(item,"id"),&id)||id>=(uint64_t)count)goto fail;
        const char *piece=bn_json_text(content);TokenizerVocabEntry *e=s->entries+id;
        if(e->added||(e->raw&&strcmp(e->raw,piece)))goto fail;
        e->raw=piece;e->added=1;
    }
    for(int i=0;i<count;i++) {
        TokenizerVocabEntry *e=s->entries+i;if(!e->raw)goto fail;e->raw_bytes=strlen(e->raw);BnTokenizerVocabDecodeInfo info;
        if(tokenizer_vocab_decode_raw(e->raw,e->raw_bytes,NULL,&info))goto fail;
        e->decoded_bytes=info.bytes;e->replacement=(unsigned char)info.contains_replacement;
    }
    v->state=s;return 0;
fail:free(s);return -1;
}
void bn_tokenizer_vocab_free(BnTokenizerVocab *v) {if(v){free(v->state);v->state=NULL;}}
int bn_tokenizer_vocab_size(const BnTokenizerVocab *v) {const TokenizerVocabState *s=v?v->state:NULL;return s?s->count:0;}
static const TokenizerVocabEntry *tokenizer_vocab_entry(const BnTokenizerVocab *v,int token) {
    const TokenizerVocabState *s=v?v->state:NULL;return s&&token>=0&&token<s->count?s->entries+token:NULL;
}
const char *bn_tokenizer_vocab_raw(const BnTokenizerVocab *v,int token) {const TokenizerVocabEntry *e=tokenizer_vocab_entry(v,token);return e?e->raw:NULL;}
int bn_tokenizer_vocab_decode_info(const BnTokenizerVocab *v,int token,BnTokenizerVocabDecodeInfo *out) {
    const TokenizerVocabEntry *e=tokenizer_vocab_entry(v,token);if(!e||!out)return -1;
    *out=(BnTokenizerVocabDecodeInfo){e->decoded_bytes,e->replacement};return 0;
}
static int tokenizer_vocab_disjoint(const void *a,size_t n,const void *b,size_t m) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(n>UINTPTR_MAX-x||m>UINTPTR_MAX-y)return 0;
    return x+n<=y||y+m<=x;
}
int bn_tokenizer_vocab_decode(const BnTokenizerVocab *v,int token,unsigned char *out,size_t capacity,size_t *written) {
    const TokenizerVocabEntry *e=tokenizer_vocab_entry(v,token);if(!e||!written||capacity<e->decoded_bytes||(!out&&e->decoded_bytes))return -1;
    const TokenizerVocabState *s=v->state;size_t n=e->decoded_bytes;
    if(!tokenizer_vocab_disjoint(out,n,written,sizeof(*written))||
        !tokenizer_vocab_disjoint(out,n,s,s->allocated)||!tokenizer_vocab_disjoint(out,n,v,sizeof(*v))||
        !tokenizer_vocab_disjoint(out,n,e->raw,e->raw_bytes+1)||
        !tokenizer_vocab_disjoint(written,sizeof(*written),s,s->allocated)||
        !tokenizer_vocab_disjoint(written,sizeof(*written),v,sizeof(*v))||
        !tokenizer_vocab_disjoint(written,sizeof(*written),e->raw,e->raw_bytes+1))return -1;
    BnTokenizerVocabDecodeInfo info;
    if(tokenizer_vocab_decode_raw(e->raw,e->raw_bytes,out,&info))return -1;
    *written=info.bytes;return 0;
}
