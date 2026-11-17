/* Shared immutable vocabulary/skip policy, request-local bounded UTF8 state. */
typedef struct {BnTokenizerVocab vocab;size_t allocated;int count;unsigned char flags[];} TokenizerDecoderState;
int bn_tokenizer_decoder_plan(int count,size_t *out) {
    size_t vb;if(!out||bn_tokenizer_vocab_plan(count,&vb)||
        (size_t)count>SIZE_MAX-sizeof(TokenizerDecoderState)||vb>SIZE_MAX-sizeof(TokenizerDecoderState)-(size_t)count)return -1;
    *out=sizeof(TokenizerDecoderState)+(size_t)count+vb;return 0;
}
void bn_tokenizer_decoder_free(BnTokenizerDecoder *d) {
    if(d){TokenizerDecoderState *s=d->state;if(s){bn_tokenizer_vocab_free(&s->vocab);free(s);}d->state=NULL;}
}
int bn_tokenizer_decoder_init_gguf(BnTokenizerDecoder *d,const BnTokenizer *t,
    const BnGGUFFile *file,size_t budget) {
    size_t total,vb;
    if(!d||d->state||!t||!file||t->metaspace||!t->vocab||
       bn_tokenizer_decoder_plan(t->vocab_size,&total)||budget<total||
       bn_tokenizer_vocab_plan(t->vocab_size,&vb))return -1;
    const unsigned char *types=NULL;
    for(uint64_t i=0;i<file->n_kv;i++)if(!strcmp(file->kvs[i].key,"tokenizer.ggml.token_type")) {
        const BnGGUFKeyValue *kv=file->kvs+i;
        if(kv->type!=BN_GGUF_TYPE_ARRAY||kv->value.arr.elem_type!=BN_GGUF_TYPE_INT32||
           kv->value.arr.n<(uint64_t)t->vocab_size||!kv->value.arr.data)return -1;
        types=kv->value.arr.data;break;
    }
    TokenizerDecoderState *s=calloc(1,total-vb);if(!s)return -1;
    s->allocated=total-vb;s->count=t->vocab_size;BnTokenizerDecoder pending={s};
    TokenizerVocabState *v=calloc(1,vb);if(!v){free(s);return -1;}
    s->vocab.state=v;v->allocated=vb;v->count=t->vocab_size;
    for(int i=0;i<t->vocab_size;i++) {
        TokenizerVocabEntry *e=v->entries+i;BnTokenizerVocabDecodeInfo info;
        if(!t->vocab[i])goto fail;
        e->raw=t->vocab[i];e->raw_bytes=strlen(e->raw);
        if(tokenizer_vocab_decode_raw(e->raw,e->raw_bytes,NULL,&info))goto fail;
        e->decoded_bytes=info.bytes;e->replacement=(unsigned char)info.contains_replacement;
        if(types){int32_t kind;memcpy(&kind,types+(size_t)i*4,4);s->flags[i]=(unsigned char)(kind==3);}
        size_t pos=0;
        while(pos<e->raw_bytes) {
            uint32_t cp=tokenizer_nfc_read((const unsigned char*)e->raw,e->raw_bytes,&pos);
            if(!((cp>=33&&cp<=126)||(cp>=161&&cp<=172)||(cp>=174&&cp<=255)||
                (cp>=BN_BPE_UNICODE_OFFSET&&cp<=BN_BPE_UNICODE_END))){s->flags[i]|=2;break;}
        }
    }
    d->state=s;return 0;
fail:bn_tokenizer_decoder_free(&pending);return -1;
}
int bn_tokenizer_decoder_init_json(BnTokenizerDecoder *d,const BnJSON *doc,int count,size_t budget) {
    size_t total,vb;if(!d||d->state||bn_tokenizer_decoder_plan(count,&total)||budget<total||bn_tokenizer_vocab_plan(count,&vb))return -1;
    TokenizerDecoderState *s=calloc(1,total-vb);if(!s)return -1;
    s->allocated=total-vb;s->count=count;BnTokenizerDecoder pending={s};
    if(bn_tokenizer_vocab_init_json(&s->vocab,doc,count,vb))goto fail;
    const BnJSONValue *added=bn_json_get(bn_json_root(doc),"added_tokens");
    for(size_t i=0;i<bn_json_size(added);i++) {
        const BnJSONValue *item=bn_json_at(added,i);uint64_t id;int special;
        if(bn_json_u64(bn_json_get(item,"id"),&id)||id>=(uint64_t)count||bn_json_bool(bn_json_get(item,"special"),&special))goto fail;
        s->flags[id]=(unsigned char)special;
    }
    const TokenizerVocabState *v=s->vocab.state;
    for(int i=0;i<count;i++) {
        const TokenizerVocabEntry *e=v->entries+i;size_t pos=0;
        while(pos<e->raw_bytes) {
            uint32_t cp=tokenizer_nfc_read((const unsigned char*)e->raw,e->raw_bytes,&pos);
            /* Rust ByteLevel falls back for the WHOLE token if any character
             * is outside its exact 256-character alphabet. */
            if(!((cp>=33&&cp<=126)||(cp>=161&&cp<=172)||(cp>=174&&cp<=255)||
                (cp>=BN_BPE_UNICODE_OFFSET&&cp<=BN_BPE_UNICODE_END))) {s->flags[i]|=2;break;}
        }
    }
    d->state=s;return 0;
fail:bn_tokenizer_decoder_free(&pending);return -1;
}
static unsigned tokenizer_stream_width(unsigned b) {return b<128?1:b>=0xc2&&b<=0xdf?2:b>=0xe0&&b<=0xef?3:b>=0xf0&&b<=0xf4?4:0;}
static int tokenizer_stream_continuation(unsigned lead,unsigned index,unsigned byte) {
    return(byte&0xc0)==0x80&&(index!=1||!((lead==0xe0&&byte<0xa0)||(lead==0xed&&byte>0x9f)||
        (lead==0xf0&&byte<0x90)||(lead==0xf4&&byte>0x8f)));
}
static int tokenizer_stream_surrogate(const BnTokenizerDecodeStream *s) {
    return s->count==2&&s->pending[0]==0xed&&s->pending[1]>=0xa0&&s->pending[1]<=0xbf;
}
static int tokenizer_stream_valid(const BnTokenizerDecodeStream *s) {
    if(!s||!s->owner||s->count>3||(s->skip_special!=0&&s->skip_special!=1)||
        (s->finished!=0&&s->finished!=1)||(s->finished&&s->count))return 0;
    if(s->count) {
        unsigned width=tokenizer_stream_width(s->pending[0]);if(width<2||s->count>=width)return 0;
        if(tokenizer_stream_surrogate(s))return 1;
        for(unsigned i=1;i<s->count;i++)if(!tokenizer_stream_continuation(s->pending[0],i,s->pending[i]))return 0;
    }
    return 1;
}
int bn_tokenizer_decode_stream_init(BnTokenizerDecodeStream *stream,const BnTokenizerDecoder *d,int skip) {
    const TokenizerDecoderState *s=d?d->state:NULL;
    if(!stream||!s||(skip!=0&&skip!=1)||
        !tokenizer_vocab_disjoint(stream,sizeof(*stream),d,sizeof(*d))||
        !tokenizer_vocab_disjoint(stream,sizeof(*stream),s,s->allocated))return -1;
    const TokenizerVocabState *v=s->vocab.state;
    if(!tokenizer_vocab_disjoint(stream,sizeof(*stream),v,v->allocated)||stream->owner)return -1;
    BnTokenizerDecodeStream next={0};next.owner=s;next.skip_special=skip;*stream=next;return 0;
}
int bn_tokenizer_decode_stream_reset(BnTokenizerDecodeStream *stream) {
    if(!stream||!stream->owner||(stream->skip_special!=0&&stream->skip_special!=1))return -1;
    BnTokenizerDecodeStream next={0};next.owner=stream->owner;next.skip_special=stream->skip_special;*stream=next;return 0;
}
void bn_tokenizer_decode_stream_free(BnTokenizerDecodeStream *stream) {if(stream){BnTokenizerDecodeStream empty={0};*stream=empty;}}
static int tokenizer_stream_emit(unsigned char *out,size_t *n,const unsigned char *data,size_t bytes) {
    if(bytes>SIZE_MAX-*n)return -1;
    if(out)memcpy(out+*n,data,bytes);
    *n+=bytes;return 0;
}
static int tokenizer_stream_replacement(unsigned char *out,size_t *n) {
    static const unsigned char replacement[]={0xef,0xbf,0xbd};return tokenizer_stream_emit(out,n,replacement,3);
}
static int tokenizer_stream_byte(BnTokenizerDecodeStream *s,unsigned byte,unsigned char *out,size_t *n) {
    if(tokenizer_stream_surrogate(s)) {
        if(tokenizer_stream_replacement(out,n)||tokenizer_stream_replacement(out,n))return -1;
        memset(s->pending,0,sizeof(s->pending));s->count=0;
    }
    if(s->count) {
        /* The source incremental codec waits for a third byte before rejecting
         * ED A0..BF. Keep this bounded two-byte state even though it cannot
         * become a valid scalar; batch/flush still replace both invalid bytes. */
        if(s->count==1&&s->pending[0]==0xed&&byte>=0xa0&&byte<=0xbf) {
            s->pending[1]=(unsigned char)byte;s->count=2;return 0;
        }
        if(tokenizer_stream_continuation(s->pending[0],s->count,byte)) {
            s->pending[s->count++]=(unsigned char)byte;
            if(s->count==tokenizer_stream_width(s->pending[0])) {
                if(tokenizer_stream_emit(out,n,s->pending,s->count))return -1;
                memset(s->pending,0,sizeof(s->pending));s->count=0;
            }
            return 0;
        }
        if(tokenizer_stream_replacement(out,n))return -1;
        memset(s->pending,0,sizeof(s->pending));s->count=0;
        /* Invalid continuation starts a new unit; never discard that byte. */
    }
    unsigned width=tokenizer_stream_width(byte);
    if(width==1){unsigned char b=(unsigned char)byte;return tokenizer_stream_emit(out,n,&b,1);}
    if(!width)return tokenizer_stream_replacement(out,n);
    s->pending[0]=(unsigned char)byte;s->count=1;return 0;
}
static int tokenizer_stream_run(const TokenizerDecoderState *s,BnTokenizerDecodeStream *stream,
    const int *tokens,size_t count,int finish,unsigned char *out,size_t *written) {
    const TokenizerVocabState *v=s->vocab.state;size_t n=0;
    for(size_t i=0;i<count;i++) {
        int id=tokens[i];if(id<0||id>=s->count)return -1;
        if(stream->skip_special&&(s->flags[id]&1))continue;
        const TokenizerVocabEntry *e=v->entries+id;
        if(s->flags[id]&2) {
            for(size_t k=0;k<e->raw_bytes;k++)if(tokenizer_stream_byte(stream,(unsigned char)e->raw[k],out,&n))return -1;
            continue;
        }
        TokenizerVocabBytes bytes={0};bytes.raw=(const unsigned char*)e->raw;bytes.length=e->raw_bytes;int b;
        while((b=tokenizer_vocab_byte(&bytes))>=0)if(tokenizer_stream_byte(stream,(unsigned)b,out,&n))return -1;
        if(b!=-1)return -1;
    }
    if(finish) {
        if(stream->count&&tokenizer_stream_replacement(out,&n))return -1;
        if(tokenizer_stream_surrogate(stream)&&tokenizer_stream_replacement(out,&n))return -1;
        memset(stream->pending,0,sizeof(stream->pending));stream->count=0;stream->finished=1;
    }
    *written=n;return 0;
}
int bn_tokenizer_decode_stream_push(const BnTokenizerDecoder *d,BnTokenizerDecodeStream *stream,
    const int *tokens,size_t count,int finish,unsigned char *out,size_t cap,size_t *written) {
    const TokenizerDecoderState *s=d?d->state:NULL;
    if(!s||!stream||!written||(finish!=0&&finish!=1)||!tokenizer_stream_valid(stream)||
        stream->owner!=s||stream->finished||count>SIZE_MAX/sizeof(*tokens)||(!tokens&&count)||(!out&&cap))return -1;
    size_t input_bytes=count*sizeof(*tokens);const TokenizerVocabState *v=s->vocab.state;
    if(!tokenizer_vocab_disjoint(tokens,input_bytes,stream,sizeof(*stream))||
        !tokenizer_vocab_disjoint(out,cap,stream,sizeof(*stream))||!tokenizer_vocab_disjoint(written,sizeof(*written),stream,sizeof(*stream))||
        !tokenizer_vocab_disjoint(out,cap,tokens,input_bytes)||!tokenizer_vocab_disjoint(written,sizeof(*written),tokens,input_bytes)||
        !tokenizer_vocab_disjoint(out,cap,written,sizeof(*written))||
        !tokenizer_vocab_disjoint(out,cap,d,sizeof(*d))||!tokenizer_vocab_disjoint(written,sizeof(*written),d,sizeof(*d))||
        !tokenizer_vocab_disjoint(out,cap,s,s->allocated)||!tokenizer_vocab_disjoint(written,sizeof(*written),s,s->allocated)||
        !tokenizer_vocab_disjoint(out,cap,v,v->allocated)||!tokenizer_vocab_disjoint(written,sizeof(*written),v,v->allocated))return -1;
    for(size_t i=0;i<count;i++) {
        int id=tokens[i];if(id<0||id>=s->count)return -1;const TokenizerVocabEntry *e=v->entries+id;
        if(!tokenizer_vocab_disjoint(out,cap,e->raw,e->raw_bytes+1)||
            !tokenizer_vocab_disjoint(written,sizeof(*written),e->raw,e->raw_bytes+1))return -1;
    }
    BnTokenizerDecodeStream next=*stream;size_t n;
    if(tokenizer_stream_run(s,&next,tokens,count,finish,NULL,&n)||n>cap)return -1;
    next=*stream;size_t produced;
    if(tokenizer_stream_run(s,&next,tokens,count,finish,out,&produced))return -1;
    *stream=next;*written=produced;return 0;
}
