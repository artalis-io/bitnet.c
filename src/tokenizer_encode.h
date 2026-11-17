/* Native bounded immutable byte BPE; request-local heap workspace. */
#include "tokenizer_encode_unicode.h"
static const char *const tokenizer_encode_patterns[]={
"\\p{N}{1,3}",
"[一-龥぀-ゟ゠-ヿ]+",
"[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
};
enum { ENC_L=1,ENC_M=2,ENC_N=4,ENC_P=8,ENC_S=16,ENC_W=32 };
typedef struct {const char *raw;int id;} EncodeIndex;
typedef struct {int child,sibling,token;unsigned char byte;} EncodeTrie;
typedef struct {
    BnTokenizerVocab vocab;
    size_t allocated,trie_capacity,trie_count;
    int count,n_merges,bytes[256];
    EncodeIndex *index;
    struct BnTokenizerMerge *merges;
    EncodeTrie *trie;
} EncodeState;
typedef struct {int id,prev,next;uint32_t version;} EncodeNode;
typedef struct {int rank,left,right,token;uint32_t lv,rv;} EncodeEvent;
typedef struct {
    size_t allocated,capacity,heap_capacity,heap_count;
    EncodeNode *nodes;EncodeEvent *heap;int *result;
} EncodeScratch;
typedef struct {const EncodeState *s;const unsigned char *text;int length,pos,plain_end,span_end,added;} EncodeCursor;
static int encode_size_add(size_t *n,size_t count,size_t item) {
    if(count>(SIZE_MAX-*n)/item)return -1;
    *n+=count*item;return 0;
}
static int encode_json_type(const BnJSONValue *v,const char *type) {
    const char *s=bn_json_text(bn_json_get(v,"type"));return s&&!strcmp(s,type);
}
static int encode_json_false(const BnJSONValue *v,const char *key) {
    int b;return !bn_json_bool(bn_json_get(v,key),&b)&&!b;
}
static int encode_json_optional_null(const BnJSONValue *v,const char *key) {
    BnJSONType t=bn_json_type(bn_json_get(v,key));return t==BN_JSON_NULL||t==BN_JSON_INVALID;
}
static int encode_profile(const BnJSON *document) {
    const BnJSONValue *root=bn_json_root(document),*m=bn_json_get(root,"model"),*n=bn_json_get(root,"normalizer");
    if(!encode_json_type(m,"BPE")||!encode_json_optional_null(m,"dropout")||
        !encode_json_optional_null(m,"unk_token")||!encode_json_optional_null(m,"continuing_subword_prefix")||
        !encode_json_optional_null(m,"end_of_word_suffix")||!encode_json_false(m,"byte_fallback")||
        !encode_json_false(m,"fuse_unk"))return -1;
    const BnJSONValue *ignore=bn_json_get(m,"ignore_merges");int b;
    if(ignore&&(bn_json_bool(ignore,&b)||b))return -1;
    if(n&&bn_json_type(n)!=BN_JSON_NULL&&(!encode_json_type(n,"Sequence")||
        bn_json_type(bn_json_get(n,"normalizers"))!=BN_JSON_ARRAY||bn_json_size(bn_json_get(n,"normalizers"))))return -1;
    const BnJSONValue *pre=bn_json_get(root,"pre_tokenizer"),*steps=bn_json_get(pre,"pretokenizers");
    if(!encode_json_type(pre,"Sequence")||bn_json_type(steps)!=BN_JSON_ARRAY||bn_json_size(steps)!=4)return -1;
    for(int i=0;i<3;i++) {
        const BnJSONValue *step=bn_json_at(steps,(size_t)i);
        const char *regex=bn_json_text(bn_json_get(bn_json_get(step,"pattern"),"Regex"));
        const char *behavior=bn_json_text(bn_json_get(step,"behavior"));
        if(!encode_json_type(step,"Split")||!regex||strcmp(regex,tokenizer_encode_patterns[i])||
            !behavior||strcmp(behavior,"Isolated")||!encode_json_false(step,"invert"))return -1;
    }
    const BnJSONValue *byte=bn_json_at(steps,3);
    if(!encode_json_type(byte,"ByteLevel")||!encode_json_false(byte,"add_prefix_space")||
        !encode_json_false(byte,"use_regex")||bn_json_bool(bn_json_get(byte,"trim_offsets"),&b))return -1;
    const BnJSONValue *post=bn_json_get(root,"post_processor");
    if(post&&bn_json_type(post)!=BN_JSON_NULL&&!encode_json_type(post,"ByteLevel"))return -1;
    return 0;
}
int bn_tokenizer_encoder_plan(const BnJSON *doc,int count,BnTokenizerEncoderPlan *out) {
    if(!out||encode_profile(doc))return -1;
    size_t vocab_bytes;if(bn_tokenizer_vocab_plan(count,&vocab_bytes))return -1;
    const BnJSONValue *root=bn_json_root(doc),*added=bn_json_get(root,"added_tokens");
    const BnJSONValue *merges=bn_json_get(bn_json_get(root,"model"),"merges");
    size_t nm=bn_json_size(merges),nodes=1;
    if(bn_json_type(merges)!=BN_JSON_ARRAY||nm>INT_MAX||bn_json_type(added)!=BN_JSON_ARRAY||bn_json_size(added)>(size_t)count)return -1;
    for(size_t i=0;i<bn_json_size(added);i++) {
        const BnJSONValue *a=bn_json_at(added,i);const char *text=bn_json_text(bn_json_get(a,"content"));int b;
        if(bn_json_type(bn_json_get(a,"content"))!=BN_JSON_STRING||!text||!*text||!encode_json_false(a,"single_word")||!encode_json_false(a,"lstrip")||
            !encode_json_false(a,"rstrip")||bn_json_bool(bn_json_get(a,"normalized"),&b)||
            bn_json_bool(bn_json_get(a,"special"),&b)||strlen(text)>(size_t)INT_MAX-nodes)return -1;
        nodes+=strlen(text);
    }
    size_t total=sizeof(EncodeState);
    if(encode_size_add(&total,(size_t)count,sizeof(EncodeIndex))||encode_size_add(&total,nm,sizeof(struct BnTokenizerMerge))||
        encode_size_add(&total,nodes,sizeof(EncodeTrie))||encode_size_add(&total,1,vocab_bytes))return -1;
    *out=(BnTokenizerEncoderPlan){total,nodes,count,(int)nm};return 0;
}
static int encode_index_compare(const void *a,const void *b) {return strcmp(((const EncodeIndex*)a)->raw,((const EncodeIndex*)b)->raw);}
/* In-place construction sort keeps temporary allocation inside the plan. */
static void encode_swap(unsigned char *a,unsigned char *b,size_t width) {
    unsigned char tmp[sizeof(EncodeIndex)+sizeof(struct BnTokenizerMerge)];
    memcpy(tmp,a,width);memcpy(a,b,width);memcpy(b,tmp,width);
}
static void encode_sift(unsigned char *base,size_t n,size_t root,size_t width,int (*compare)(const void*,const void*)) {
    while(root*2+1<n) {
        size_t child=root*2+1;
        if(child+1<n&&compare(base+child*width,base+(child+1)*width)<0)child++;
        if(compare(base+root*width,base+child*width)>=0)break;
        encode_swap(base+root*width,base+child*width,width);root=child;
    }
}
static void encode_sort(void *array,size_t n,size_t width,int (*compare)(const void*,const void*)) {
    unsigned char *base=array;
    for(size_t i=n/2;i>0;i--)encode_sift(base,n,i-1,width,compare);
    while(n>1){encode_swap(base,base+(n-1)*width,width);n--;encode_sift(base,n,0,width,compare);}
}
static int encode_lookup(const EncodeState *s,const char *a,size_t na,const char *b,size_t nb) {
    if(na>SIZE_MAX-nb)return -1;
    size_t n=na+nb;int lo=0,hi=s->count;
    while(lo<hi) {
        int mid=lo+(hi-lo)/2;const char *raw=s->index[mid].raw;size_t nr=strlen(raw),part=nr<na?nr:na;
        int cmp=memcmp(raw,a,part);
        if(!cmp&&nr>=na){size_t rest=nr-na;part=rest<nb?rest:nb;if(part)cmp=memcmp(raw+na,b,part);}
        if(!cmp)cmp=(nr>n)-(nr<n);
        if(!cmp)return s->index[mid].id;
        if(cmp<0)lo=mid+1;else hi=mid;
    }
    return -1;
}
static void encode_byte_raw(unsigned byte,char out[3]) {
    unsigned cp=byte;
    if(byte<=32)cp=256+byte;
    else if(byte==127)cp=289;
    else if(byte>=128&&byte<=160)cp=290+byte-128;
    else if(byte==173)cp=323;
    if(cp<128){out[0]=(char)cp;out[1]=0;}
    else {out[0]=(char)(0xc0|(cp>>6));out[1]=(char)(0x80|(cp&63));out[2]=0;}
}
void bn_tokenizer_encoder_free(BnTokenizerEncoder *e) {
    if(e){EncodeState *s=e->state;if(s){bn_tokenizer_vocab_free(&s->vocab);free(s);}e->state=NULL;}
}
int bn_tokenizer_encoder_init_json(BnTokenizerEncoder *e,const BnJSON *doc,int count,size_t budget) {
    BnTokenizerEncoderPlan plan;size_t vb;
    if(!e||e->state||bn_tokenizer_encoder_plan(doc,count,&plan)||budget<plan.host_bytes||bn_tokenizer_vocab_plan(count,&vb))return -1;
    EncodeState *s=calloc(1,plan.host_bytes-vb);if(!s)return -1;
    s->allocated=plan.host_bytes-vb;s->count=count;s->n_merges=plan.merges;s->trie_capacity=plan.trie_nodes;
    s->index=(EncodeIndex*)(s+1);s->merges=(struct BnTokenizerMerge*)(s->index+count);s->trie=(EncodeTrie*)(s->merges+s->n_merges);
    BnTokenizerEncoder pending={s};
    if(bn_tokenizer_vocab_init_json(&s->vocab,doc,count,vb))goto fail;
    const TokenizerVocabState *v=s->vocab.state;
    for(int i=0;i<count;i++)s->index[i]=(EncodeIndex){v->entries[i].raw,i};
    encode_sort(s->index,(size_t)count,sizeof(*s->index),encode_index_compare);
    for(int i=1;i<count;i++)if(!strcmp(s->index[i-1].raw,s->index[i].raw))goto fail;
    const BnJSONValue *root=bn_json_root(doc),*model=bn_json_get(root,"model"),*core=bn_json_get(model,"vocab"),*merges=bn_json_get(model,"merges");
    for(unsigned i=0;i<256;i++) {
        char raw[3];encode_byte_raw(i,raw);s->bytes[i]=encode_lookup(s,raw,strlen(raw),NULL,0);
        if(s->bytes[i]<0||!bn_json_get(core,raw))goto fail;
    }
    for(int i=0;i<s->n_merges;i++) {
        const BnJSONValue *pair=bn_json_at(merges,(size_t)i);const char *left,*right;size_t nl,nr;
        if(bn_json_type(pair)==BN_JSON_STRING) {
            left=bn_json_text(pair);const char *sep=strchr(left,' ');if(!sep||sep==left||!sep[1]||strchr(sep+1,' '))goto fail;
            nl=(size_t)(sep-left);right=sep+1;nr=strlen(right);
        } else if(bn_json_type(pair)==BN_JSON_ARRAY&&bn_json_size(pair)==2) {
            if(bn_json_type(bn_json_at(pair,0))!=BN_JSON_STRING||bn_json_type(bn_json_at(pair,1))!=BN_JSON_STRING)goto fail;
            left=bn_json_text(bn_json_at(pair,0));right=bn_json_text(bn_json_at(pair,1));if(!left||!right)goto fail;nl=strlen(left);nr=strlen(right);
        } else goto fail;
        int l=encode_lookup(s,left,nl,NULL,0),r=encode_lookup(s,right,nr,NULL,0),t=encode_lookup(s,left,nl,right,nr);
        if(!nl||!nr||l<0||r<0||t<0||!bn_json_get(core,v->entries[l].raw)||!bn_json_get(core,v->entries[r].raw)||!bn_json_get(core,v->entries[t].raw))goto fail;
        s->merges[i]=(struct BnTokenizerMerge){l,r,t,i};
    }
    encode_sort(s->merges,(size_t)s->n_merges,sizeof(*s->merges),tokenizer_merge_compare);
    for(int i=1;i<s->n_merges;i++)if(s->merges[i-1].left==s->merges[i].left&&s->merges[i-1].right==s->merges[i].right)goto fail;
    for(size_t i=0;i<s->trie_capacity;i++)s->trie[i].token=-1;
    s->trie_count=1;
    for(int id=0;id<count;id++)if(v->entries[id].added) {
        int node=0;const unsigned char *text=(const unsigned char*)v->entries[id].raw;
        for(size_t i=0;text[i];i++) {
            int child=s->trie[node].child;
            while(child&&s->trie[child].byte!=text[i])child=s->trie[child].sibling;
            if(!child) {
                if(s->trie_count==s->trie_capacity)goto fail;
                child=(int)s->trie_count++;s->trie[child].byte=text[i];s->trie[child].sibling=s->trie[node].child;s->trie[node].child=child;
            }
            node=child;
        }
        if(s->trie[node].token>=0)goto fail;
        s->trie[node].token=id;
    }
    e->state=s;return 0;
fail:bn_tokenizer_encoder_free(&pending);return -1;
}
static unsigned encode_flags(uint32_t cp) {
    if(cp>0x10ffff)return 0;
    size_t lo=0,hi=sizeof(tokenizer_encode_unicode)/sizeof(tokenizer_encode_unicode[0]);
    while(lo+1<hi){size_t mid=lo+(hi-lo)/2;if(tokenizer_encode_unicode[mid].start<=cp)lo=mid;else hi=mid;}
    return tokenizer_encode_unicode[lo].flags;
}
static int encode_read(const unsigned char *text,int pos,int end,uint32_t *cp) {size_t p=(size_t)pos;*cp=tokenizer_nfc_read(text,(size_t)end,&p);return (int)p;}
static int encode_valid(const unsigned char *text,size_t bytes) {
    if(bytes>INT_MAX||(!text&&bytes))return 0;
    size_t pos=0;while(pos<bytes)if(tokenizer_nfc_read(text,bytes,&pos)>0x10ffff)return 0;
    return 1;
}
static int encode_cjk(uint32_t cp) {return(cp>=0x4e00&&cp<=0x9fa5)||(cp>=0x3040&&cp<=0x30ff);}
static int encode_ascii_punct(uint32_t cp) {return cp<127&&cp>=33&&((cp<=47)||(cp>=58&&cp<=64)||(cp>=91&&cp<=96)||cp>=123);}
static int encode_ascii_letter(uint32_t cp) {return(cp>='A'&&cp<='Z')||(cp>='a'&&cp<='z');}
static int encode_match(const unsigned char *text,int pos,int end) {
    uint32_t cp,next;int after=encode_read(text,pos,end,&cp),p=after;
    if(encode_ascii_punct(cp)) {
        while(p<end){int q=encode_read(text,p,end,&next);if(!encode_ascii_letter(next))break;p=q;}
        if(p>after)return p;
    }
    unsigned f=encode_flags(cp);
    p=pos;
    if(cp!='\r'&&cp!='\n'&&!(f&(ENC_L|ENC_P|ENC_S)))p=after;
    int begin=p;
    while(p<end){int q=encode_read(text,p,end,&next);if(!(encode_flags(next)&(ENC_L|ENC_M)))break;p=q;}
    if(p>begin)return p;
    if(f&(ENC_L|ENC_M))return after; /* Optional prefix backtracks for one mark. */
    p=cp==' '?after:pos;begin=p;
    while(p<end){int q=encode_read(text,p,end,&next);if(!(encode_flags(next)&(ENC_P|ENC_S)))break;p=q;}
    if(p>begin){while(p<end&&(text[p]=='\r'||text[p]=='\n'))p++;return p;}
    if(f&ENC_W) {
        p=pos;int last_newline=pos,previous=pos;
        while(p<end){int q=encode_read(text,p,end,&next);if(!(encode_flags(next)&ENC_W))break;previous=p;p=q;if(next=='\r'||next=='\n')last_newline=p;}
        if(last_newline>pos)return last_newline;
        if(p<end&&previous>pos)return previous;
        return p;
    }
    return pos;
}
static int encode_added(const EncodeState *s,const unsigned char *text,int pos,int end,int *stop) {
    int node=0,token=-1;*stop=pos;
    for(int p=pos;p<end;p++) {
        int child=s->trie[node].child;while(child&&s->trie[child].byte!=text[p])child=s->trie[child].sibling;
        if(!child)break;
        node=child;if(s->trie[node].token>=0){token=s->trie[node].token;*stop=p+1;}
    }
    return token;
}
static int encode_piece(EncodeCursor *c,int *token) {
    int start=c->pos;*token=-1;if(start==c->length)return start;
    if(c->added&&start>=c->plain_end) {
        int stop;*token=encode_added(c->s,c->text,start,c->length,&stop);
        if(*token>=0){c->pos=stop;c->span_end=c->plain_end=stop;return stop;}
        int p=start;uint32_t cp;
        while(p<c->length){if(p>start&&encode_added(c->s,c->text,p,c->length,&stop)>=0)break;p=encode_read(c->text,p,c->length,&cp);}
        c->plain_end=p;
    }
    if(!c->added)c->plain_end=c->length;
    if(start>=c->span_end) {
        uint32_t cp;int p=encode_read(c->text,start,c->plain_end,&cp);unsigned flags=encode_flags(cp);
        if(flags&ENC_N) {
            for(int n=1;n<3&&p<c->plain_end;n++){uint32_t next;int q=encode_read(c->text,p,c->plain_end,&next);if(!(encode_flags(next)&ENC_N))break;p=q;}
        } else if(encode_cjk(cp)) {
            while(p<c->plain_end){uint32_t next;int q=encode_read(c->text,p,c->plain_end,&next);if(!encode_cjk(next)||(encode_flags(next)&ENC_N))break;p=q;}
        } else {
            while(p<c->plain_end){uint32_t next;int q=encode_read(c->text,p,c->plain_end,&next);if(encode_cjk(next)||(encode_flags(next)&ENC_N))break;p=q;}
        }
        c->span_end=p;
    }
    int p=start,stop=encode_match(c->text,p,c->span_end);
    if(stop==p) {
        uint32_t cp;p=encode_read(c->text,p,c->span_end,&cp);
        while(p<c->span_end&&encode_match(c->text,p,c->span_end)==p)p=encode_read(c->text,p,c->span_end,&cp);
        stop=p;
    }
    c->pos=stop;return stop;
}
int bn_tokenizer_encode_scratch_plan(size_t cap,size_t *out) {
    if(!out||cap>INT_MAX||cap>(SIZE_MAX-1)/3)return -1;
    size_t bytes=sizeof(EncodeScratch);
    if(encode_size_add(&bytes,cap,sizeof(EncodeNode))||encode_size_add(&bytes,cap*3+1,sizeof(EncodeEvent))||encode_size_add(&bytes,cap,sizeof(int)))return -1;
    *out=bytes;return 0;
}
int bn_tokenizer_encode_scratch_init(BnTokenizerEncodeScratch *handle,size_t cap,size_t budget) {
    size_t bytes;if(!handle||handle->state||bn_tokenizer_encode_scratch_plan(cap,&bytes)||budget<bytes)return -1;
    EncodeScratch *s=calloc(1,bytes);if(!s)return -1;
    s->allocated=bytes;s->capacity=cap;s->heap_capacity=cap*3+1;s->nodes=(EncodeNode*)(s+1);s->heap=(EncodeEvent*)(s->nodes+cap);s->result=(int*)(s->heap+s->heap_capacity);handle->state=s;return 0;
}
void bn_tokenizer_encode_scratch_free(BnTokenizerEncodeScratch *s) {if(s){free(s->state);s->state=NULL;}}
static int encode_event_less(const EncodeEvent *a,const EncodeEvent *b) {return a->rank<b->rank||(a->rank==b->rank&&a->left<b->left);}
static int encode_push(const EncodeState *s,EncodeScratch *w,int left) {
    int right=w->nodes[left].next;if(right<0)return 0;
    BnTokenizer table={0};table.merges=s->merges;table.n_merges=s->n_merges;
    const struct BnTokenizerMerge *m=tokenizer_find_merge(&table,w->nodes[left].id,w->nodes[right].id);if(!m)return 0;
    if(w->heap_count==w->heap_capacity)return -1;
    EncodeEvent e={m->rank,left,right,m->token,w->nodes[left].version,w->nodes[right].version};size_t p=w->heap_count++;
    while(p){size_t parent=(p-1)/2;if(!encode_event_less(&e,&w->heap[parent]))break;w->heap[p]=w->heap[parent];p=parent;}w->heap[p]=e;return 0;
}
static EncodeEvent encode_pop(EncodeScratch *w) {
    EncodeEvent out=w->heap[0],last=w->heap[--w->heap_count];size_t p=0;
    while(p*2+1<w->heap_count){size_t child=p*2+1;if(child+1<w->heap_count&&encode_event_less(&w->heap[child+1],&w->heap[child]))child++;if(!encode_event_less(&w->heap[child],&last))break;w->heap[p]=w->heap[child];p=child;}
    if(w->heap_count)w->heap[p]=last;
    return out;
}
static int encode_plain(const EncodeState *s,EncodeScratch *w,const unsigned char *text,int length,size_t *count) {
    w->heap_count=0;
    for(int i=0;i<length;i++)w->nodes[i]=(EncodeNode){s->bytes[text[i]],i-1,i+1<length?i+1:-1,0};
    for(int i=0;i<length;i++)if(encode_push(s,w,i))return -1;
    while(w->heap_count) {
        EncodeEvent e=encode_pop(w);EncodeNode *left=w->nodes+e.left,*right=w->nodes+e.right;
        if(left->next!=e.right||left->version!=e.lv||right->version!=e.rv)continue;
        left->id=e.token;left->version++;left->next=right->next;right->version++;
        if(right->next>=0)w->nodes[right->next].prev=e.left;
        if(left->prev>=0&&encode_push(s,w,left->prev))return -1;
        if(encode_push(s,w,e.left))return -1;
    }
    for(int i=0;i>=0;i=w->nodes[i].next){if(*count==w->capacity)return -1;w->result[(*count)++]=w->nodes[i].id;}
    return 0;
}
static int encode_outputs(const EncodeState *s,const void *handle,size_t handle_bytes,const unsigned char *input,size_t bytes,
    void *out,size_t output_bytes,size_t *written) {
    if(!written||(!out&&output_bytes)||!encode_valid(input,bytes))return 0;
    const TokenizerVocabState *v=s->vocab.state;
    return tokenizer_vocab_disjoint(out,output_bytes,written,sizeof(*written))&&
        tokenizer_vocab_disjoint(out,output_bytes,input,bytes)&&tokenizer_vocab_disjoint(written,sizeof(*written),input,bytes)&&
        tokenizer_vocab_disjoint(out,output_bytes,s,s->allocated)&&tokenizer_vocab_disjoint(written,sizeof(*written),s,s->allocated)&&
        tokenizer_vocab_disjoint(out,output_bytes,v,v->allocated)&&tokenizer_vocab_disjoint(written,sizeof(*written),v,v->allocated)&&
        tokenizer_vocab_disjoint(out,output_bytes,handle,handle_bytes)&&tokenizer_vocab_disjoint(written,sizeof(*written),handle,handle_bytes);
}
int bn_tokenizer_encoder_encode(const BnTokenizerEncoder *e,BnTokenizerEncodeScratch *scratch,const unsigned char *input,size_t bytes,int *tokens,size_t cap,size_t *written) {
    const EncodeState *s=e?e->state:NULL;EncodeScratch *w=scratch?scratch->state:NULL;
    if(!s||!w||bytes>w->capacity||cap>SIZE_MAX/sizeof(*tokens)||!encode_outputs(s,e,sizeof(*e),input,bytes,tokens,cap*sizeof(*tokens),written)||
        !tokenizer_vocab_disjoint(input,bytes,w,w->allocated)||!tokenizer_vocab_disjoint(tokens,cap*sizeof(*tokens),w,w->allocated)||
        !tokenizer_vocab_disjoint(written,sizeof(*written),w,w->allocated)||!tokenizer_vocab_disjoint(tokens,cap*sizeof(*tokens),scratch,sizeof(*scratch))||
        !tokenizer_vocab_disjoint(written,sizeof(*written),scratch,sizeof(*scratch)))return -1;
    EncodeCursor cursor={s,input,(int)bytes,0,0,0,1};size_t count=0;
    while(cursor.pos<cursor.length) {
        int start=cursor.pos,token;int end=encode_piece(&cursor,&token);
        if(token>=0){if(count==w->capacity)return -1;w->result[count++]=token;}
        else if(encode_plain(s,w,input+start,end-start,&count))return -1;
    }
    if(count>cap)return -1;
    if(count)memcpy(tokens,w->result,count*sizeof(*tokens));
    *written=count;return 0;
}
int bn_tokenizer_encoder_pretokenize(const BnTokenizerEncoder *e,const unsigned char *input,size_t bytes,uint32_t *ends,size_t cap,size_t *written) {
    const EncodeState *s=e?e->state:NULL;
    if(!s||cap>SIZE_MAX/sizeof(*ends)||!encode_outputs(s,e,sizeof(*e),input,bytes,ends,cap*sizeof(*ends),written))return -1;
    size_t count=0;EncodeCursor cursor={s,input,(int)bytes,0,0,0,0};int token;
    while(cursor.pos<cursor.length){encode_piece(&cursor,&token);count++;}
    if(count>cap)return -1;
    cursor=(EncodeCursor){s,input,(int)bytes,0,0,0,0};count=0;
    while(cursor.pos<cursor.length)ends[count++]=(uint32_t)encode_piece(&cursor,&token);
    *written=count;return 0;
}
