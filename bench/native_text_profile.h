/* Public raw-text API control: reference payloads are assertions only. */
typedef struct {
    BnJSON tokenizer,reference;
    BnTokenizerEncoder encoder;
    BnTokenizerDecoder decoder;
    BnTokenizerEncodeScratch scratch;
    BnGenerateNativeTextWorkspace workspace;
    const BnJSONValue *frames,*hostframes,*output;
    const char *directory;
    unsigned char *prompt,*joined;
    uint32_t *expected_ids;
    size_t prompt_bytes,prompt_tokens,forward_count,chunk_count,joined_bytes;
    size_t encoded_checks,logit_bytes;
    int vocabulary,generated,skip,predict,request_index,request_count;
    size_t prefills;
} ProfileNativeText;
static void profile_text_init(ProfileNativeText *t,const char *directory,const char *tokenizer,
    int vocabulary,const BnJSONValue *frames,const BnJSONValue *hostframes) {
    t->directory=directory;t->vocabulary=vocabulary;t->frames=frames;t->hostframes=hostframes;
    profile_json(&t->tokenizer,".",tokenizer,1);profile_json(&t->reference,directory,"text-reference.json",0);
    const BnJSONValue *root=bn_json_root(&t->reference);
    t->prompt_tokens=(size_t)profile_number(root,"token_count");
    t->prompt=profile_file(directory,bn_json_text(bn_json_get(root,"prompt")),&t->prompt_bytes);
    assert(t->prompt_bytes==(size_t)profile_number(root,"prompt_bytes")&&t->prompt_tokens);
    size_t bytes;t->expected_ids=profile_file(directory,bn_json_text(bn_json_get(root,"ids")),&bytes);
    assert(t->prompt_tokens<=SIZE_MAX/sizeof(int)&&bytes==t->prompt_tokens*4);
    t->generated=(int)bn_json_size(bn_json_get(root,"generated_ids"));
    BnTokenizerEncoderPlan ep;size_t db,sb;
    assert(!bn_tokenizer_encoder_plan(&t->tokenizer,vocabulary,&ep)&&
        !bn_tokenizer_decoder_plan(vocabulary,&db)&&!bn_tokenizer_encode_scratch_plan(t->prompt_bytes,&sb));
    assert(!bn_tokenizer_encoder_init_json(&t->encoder,&t->tokenizer,vocabulary,ep.host_bytes)&&
        !bn_tokenizer_decoder_init_json(&t->decoder,&t->tokenizer,vocabulary,db)&&
        !bn_tokenizer_encode_scratch_init(&t->scratch,t->prompt_bytes,sb));
    t->workspace=(BnGenerateNativeTextWorkspace){&t->encoder,&t->decoder,&t->scratch,
        malloc(t->prompt_tokens*sizeof(int)),t->prompt_tokens,malloc(65536),65536};
    assert(t->workspace.tokens&&t->workspace.piece);t->joined=malloc(65536);assert(t->joined);
    fprintf(stderr,"native text encoder%zu decoder%zu scratch%zu token_buffer%zu piece_buffer65536\n",
        ep.host_bytes,db,sb,t->prompt_tokens*sizeof(int));
}
static void profile_text_forward(const int *tokens,int count,int position,const float *logits,int vocabulary,void *user) {
    ProfileNativeText *t=user;assert(vocabulary==t->vocabulary);
    if(!t->forward_count) {
        for(size_t i=0;i<t->prompt_tokens;i++)assert((uint32_t)t->workspace.tokens[i]==t->expected_ids[i]);
        t->encoded_checks++;
    }
    const BnJSONValue *frame=bn_json_at(t->frames,t->forward_count);
    assert(frame&&count==profile_number(frame,"rows")&&position==profile_number(frame,"position"));
    uint64_t *ids=profile_load(t->directory,frame,"tokens",(size_t)count*8);
    for(int i=0;i<count;i++){assert(ids[i]==(uint64_t)tokens[i]);}free(ids);
    const char *name=bn_json_text(bn_json_get(frame,"name"));assert(name);
    void *gold=profile_load(t->directory,bn_json_get(t->hostframes,name),"last_logits",(size_t)vocabulary*4);
    assert(!memcmp(logits,gold,(size_t)vocabulary*4));free(gold);
    t->forward_count++;t->logit_bytes+=(size_t)vocabulary*4;
}
static int profile_text_token(const unsigned char *piece,size_t bytes,int id,void *user) {
    ProfileNativeText *t=user;
    const BnJSONValue *root=bn_json_root(&t->reference),*ids=bn_json_get(root,"generated_ids");
    if(t->chunk_count<(size_t)t->generated) {
        assert(t->forward_count==t->prefills+t->chunk_count+(t->predict?0:1));
        uint64_t expected;assert(!bn_json_u64(bn_json_at(ids,t->chunk_count),&expected)&&id==(int)expected);
    }else assert(t->chunk_count==(size_t)t->generated&&id==-1);
    const BnJSONValue *chunk=bn_json_at(bn_json_get(t->output,"chunks"),t->chunk_count);
    assert(chunk&&bytes==(size_t)profile_number(chunk,"bytes"));size_t n;
    void *gold=profile_file(t->directory,bn_json_text(bn_json_get(chunk,"file")),&n);
    assert(n==bytes&&!memcmp(piece,gold,n));free(gold);
    assert(bytes<=65536-t->joined_bytes);memcpy(t->joined+t->joined_bytes,piece,bytes);
    t->joined_bytes+=bytes;t->chunk_count++;return 0;
}
static void profile_text_verify(ProfileNativeText *t,int generated,int position,int run) {
    size_t omitted=t->predict&&t->generated>0;
    assert(generated==t->generated&&position==(int)t->prompt_tokens+t->generated-(int)omitted);
    assert(t->encoded_checks==1&&t->forward_count==bn_json_size(t->frames)-omitted&&t->chunk_count==(size_t)t->generated+1);
    size_t n;void *gold=profile_file(t->directory,bn_json_text(bn_json_get(t->output,"full")),&n);
    assert(n==t->joined_bytes&&n==(size_t)profile_number(t->output,"bytes")&&!memcmp(gold,t->joined,n));free(gold);
    printf("VERIFIED TEXT run%d skip%d prompt_bytes%zu prompt_tokens%zu forwards%zu generated%d chunks%zu host_logit_bytes%zu output_bytes%zu position%d output_hex",
        run,t->skip,t->prompt_bytes,t->prompt_tokens,t->forward_count,t->generated,t->chunk_count,t->logit_bytes,t->joined_bytes,position);
    for(size_t i=0;i<t->joined_bytes;i++){printf("%02x",t->joined[i]);}
    if(t->request_count>1)printf(" request_index%d request_count%d",t->request_index,t->request_count);
    putchar('\n');fflush(stdout);
}
static void profile_text_execute(ProfileNativeText *t,const BnModel *model,BnSession *session,
    int run,int prefill,int admit) {
    t->skip=native_text_workload_skip(run,t->request_count?t->request_count:1);
    assert(t->skip>=0);t->forward_count=t->chunk_count=t->joined_bytes=t->encoded_checks=t->logit_bytes=0;
    t->output=bn_json_at(bn_json_get(bn_json_root(&t->reference),"outputs"),(size_t)t->skip);
    int skip;assert(t->output&&!bn_json_bool(bn_json_get(t->output,"skip_special"),&skip)&&skip==t->skip);
    memset(t->workspace.tokens,0xa5,t->prompt_tokens*sizeof(int));
    BnGenerateNativeTextOptions options={prefill,t->generated,admit,t->skip};
    BnGenerateNativeTextCallbacks callbacks={profile_text_token,profile_text_forward,t};
    t->prefills=(t->prompt_tokens+(size_t)prefill-1)/(size_t)prefill;
    int generated=t->predict?
        bn_generate_native_text_predict(model,session,&t->workspace,t->prompt,t->prompt_bytes,&options,&callbacks):
        bn_generate_native_text(model,session,&t->workspace,t->prompt,t->prompt_bytes,&options,&callbacks);
    profile_text_verify(t,generated,session->pos,run);
}

static void profile_text_free(ProfileNativeText *t) {
    bn_tokenizer_encode_scratch_free(&t->scratch);bn_tokenizer_decoder_free(&t->decoder);bn_tokenizer_encoder_free(&t->encoder);
    free(t->workspace.tokens);free(t->workspace.piece);free(t->prompt);free(t->expected_ids);free(t->joined);
    bn_json_free(&t->reference);bn_json_free(&t->tokenizer);
}
