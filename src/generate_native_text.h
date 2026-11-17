/* Native text orchestration stays above tokenizer/session/transformer. */
static int native_text_disjoint(const void *a,size_t n,const void *b,size_t m) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return n<=UINTPTR_MAX-x&&m<=UINTPTR_MAX-y&&(x+n<=y||y+m<=x);
}
int bn_generate_native_text_request_init(BnGenerateNativeTextRequest *r,
    const BnModel *m,BnSession *s,const BnGenerateNativeTextWorkspace *workspace,
    const unsigned char *text,size_t bytes,const BnGenerateNativeTextOptions *options,
    const BnGenerateNativeTextCallbacks *callbacks,int predict) {
    if(!r||r->owner||!m||!s||!workspace||!options||!s->backend_only||s->pos||!s->state.logits||
        (predict!=0&&predict!=1)||options->prefill_tokens<1||options->max_new_tokens<0||
        (options->admit_decode!=0&&options->admit_decode!=1)||
        (options->skip_special!=0&&options->skip_special!=1))return -2;
    BnGenerateNativeTextWorkspace w=*workspace;
    BnGenerateNativeTextOptions o=*options;
    int commits=o.max_new_tokens-(predict&&o.max_new_tokens>0);
    BnGenerateNativeTextCallbacks cb={0};if(callbacks)cb=*callbacks;
    if(!w.encoder||!w.decoder||!w.scratch||!w.tokens||!w.token_capacity||
        w.token_capacity>SIZE_MAX/sizeof(int)||!w.piece||w.piece_capacity<6)return -2;
    size_t token_bytes=w.token_capacity*sizeof(int);
    const void *owners[]={workspace,options,callbacks,m,s,w.encoder,w.decoder,w.scratch,r};
    const size_t sizes[]={sizeof(*workspace),sizeof(*options),callbacks?sizeof(*callbacks):0,
        sizeof(*m),sizeof(*s),sizeof(*w.encoder),sizeof(*w.decoder),sizeof(*w.scratch),sizeof(*r)};
    if(!native_text_disjoint(w.tokens,token_bytes,w.piece,w.piece_capacity)||
        !native_text_disjoint(w.piece,w.piece_capacity,text,bytes)||
        !native_text_disjoint(r,sizeof(*r),text,bytes))return -2;
    for(size_t i=0;i<sizeof(owners)/sizeof(owners[0]);i++) {
        if(!native_text_disjoint(w.tokens,token_bytes,owners[i],sizes[i])||
            !native_text_disjoint(w.piece,w.piece_capacity,owners[i],sizes[i])||
            (i+1<sizeof(owners)/sizeof(owners[0])&&!native_text_disjoint(r,sizeof(*r),owners[i],sizes[i])))return -2;
    }
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(m);
    BnSessionNativeHistoryInfo history;
    BnTokenizerDecodeStream stream={0};size_t count=0;
    if(!a||a->vocab<1||bn_session_native_history_info(s,m,&history)||history.committed||history.staged||
        bn_tokenizer_decode_stream_init(&stream,w.decoder,o.skip_special)||
        bn_tokenizer_encoder_encode(w.encoder,w.scratch,text,bytes,w.tokens,w.token_capacity,&count)||
        !count||count>(size_t)(INT_MAX-commits)||history.capacity<0||
        count+(size_t)commits>(size_t)history.capacity||
        history.max_tokens<1||(count<(size_t)o.prefill_tokens?(int)count:o.prefill_tokens)>history.max_tokens)return -2;
    for(size_t i=0;i<count;i++)if(w.tokens[i]<0||w.tokens[i]>=a->vocab)return -2;
    *r=(BnGenerateNativeTextRequest){.owner=r,.model=m,.session=s,.workspace=w,
        .options=o,.callbacks=cb,.stream=stream,.prompt_tokens=count,.phase=1,
        .predict=predict,.status=BN_NATIVE_TEXT_PROGRESS};
    return 0;
}
static BnGenerateNativeTextStatus native_text_terminal(BnGenerateNativeTextRequest *r,
    BnGenerateNativeTextStatus status) {
    r->status=status;r->phase=0;r->stepping=0;return status;
}
static BnGenerateNativeTextStatus native_text_finish(BnGenerateNativeTextRequest *r) {
    size_t n;
    if(bn_tokenizer_decode_stream_push(r->workspace.decoder,&r->stream,NULL,0,1,
        r->workspace.piece,r->workspace.piece_capacity,&n))return native_text_terminal(r,BN_NATIVE_TEXT_ERROR);
    if(r->callbacks.token)(void)r->callbacks.token(r->workspace.piece,n,-1,r->callbacks.user_data);
    return native_text_terminal(r,BN_NATIVE_TEXT_DONE);
}
static int native_text_publish(BnGenerateNativeTextRequest *r) {
    r->stream=r->pending;r->generated++;
    return r->callbacks.token&&r->callbacks.token(r->workspace.piece,r->piece_bytes,
        r->next,r->callbacks.user_data);
}
static BnGenerateNativeTextStatus native_text_step(BnGenerateNativeTextRequest *r) {
    BnSession *s=r->session;const BnModel *m=r->model;
    BnSessionNativeHistoryInfo h;
    if(bn_session_native_history_info(s,m,&h)||s->pos!=r->position||
        h.committed!=r->position||h.staged!=r->rows)return native_text_terminal(r,BN_NATIVE_TEXT_ERROR);
    if(r->phase==2) {
        const BnModelNativeAnatomy *a=bn_model_native_anatomy(m);
        if(!a||a->vocab<1||!s->state.logits)return native_text_terminal(r,BN_NATIVE_TEXT_ERROR);
        r->next=0;
        for(int id=0;id<a->vocab;id++) {
            if(!isfinite(s->state.logits[id]))return native_text_terminal(r,BN_NATIVE_TEXT_ERROR);
            if(s->state.logits[id]>s->state.logits[r->next])r->next=id;
        }
        r->pending=r->stream;
        if(bn_tokenizer_decode_stream_push(r->workspace.decoder,&r->pending,&r->next,1,0,
            r->workspace.piece,r->workspace.piece_capacity,&r->piece_bytes))return native_text_terminal(r,BN_NATIVE_TEXT_ERROR);
        if(r->predict) {
            int stop=native_text_publish(r);
            if(stop||r->generated==r->options.max_new_tokens)return native_text_finish(r);
        }
        r->phase=3;
    }
    const int *tokens=r->phase==1?r->workspace.tokens+r->prefilled:&r->next;
    if(!r->rows) {
        size_t left=r->prompt_tokens-r->prefilled;
        int rows=r->phase==1?(int)(left<(size_t)r->options.prefill_tokens?left:(size_t)r->options.prefill_tokens):1;
        if(bn_session_native_history_stage(s,m,tokens,rows))return native_text_terminal(r,BN_NATIVE_TEXT_ERROR);
        r->rows=rows;
    }
    BnTransformerTrellisForwardResult out;
    int status=bn_transformer_trellis_forward_run(m,s,r->phase==1?0:r->options.admit_decode,&out);
    if(status==BN_GPU_UPLOAD_BUSY){r->status=BN_NATIVE_TEXT_BUSY;r->stepping=0;return r->status;}
    if(status||!bn_transformer_trellis_forward_current(m,s,&out)||out.position!=r->position||
        out.tokens!=r->rows||s->pos!=r->position+r->rows)return native_text_terminal(r,BN_NATIVE_TEXT_ERROR);
    if(r->callbacks.forward)r->callbacks.forward(tokens,r->rows,r->position,out.host_logits,out.vocab,r->callbacks.user_data);
    r->position+=r->rows;
    if(r->phase==1) {
        r->prefilled+=(size_t)r->rows;r->rows=0;
        if(r->prefilled==r->prompt_tokens) {
            r->phase=2;
            if(!r->options.max_new_tokens)return native_text_finish(r);
        }
    }else {
        r->rows=0;r->phase=2;
        if(!r->predict) {
            int stop=native_text_publish(r);
            if(stop||r->generated==r->options.max_new_tokens)return native_text_finish(r);
        }
    }
    r->status=BN_NATIVE_TEXT_PROGRESS;r->stepping=0;return r->status;
}
BnGenerateNativeTextStatus bn_generate_native_text_request_step(BnGenerateNativeTextRequest *r) {
    if(!r||r->owner!=r||r->stepping)return BN_NATIVE_TEXT_ERROR;
    if(!r->phase)return r->status;
    r->stepping=1;return native_text_step(r);
}
BnGenerateNativeTextStatus bn_generate_native_text_request_cancel(BnGenerateNativeTextRequest *r) {
    if(!r||r->owner!=r||r->stepping)return BN_NATIVE_TEXT_ERROR;
    if(!r->phase)return r->status;
    return native_text_terminal(r,BN_NATIVE_TEXT_CANCELLED);
}
void bn_generate_native_text_request_free(BnGenerateNativeTextRequest *r) {
    if(!r||r->stepping||(r->owner&&r->owner!=r))return;
    *r=(BnGenerateNativeTextRequest){0};
}
static int native_text_generate(const BnModel *m,BnSession *s,
    const BnGenerateNativeTextWorkspace *workspace,const unsigned char *text,size_t bytes,
    const BnGenerateNativeTextOptions *options,const BnGenerateNativeTextCallbacks *callbacks,int predict) {
    BnGenerateNativeTextRequest r={0};
    if(bn_generate_native_text_request_init(&r,m,s,workspace,text,bytes,options,callbacks,predict))return -2;
    BnGenerateNativeTextStatus status;
    do{status=bn_generate_native_text_request_step(&r);}while(status==BN_NATIVE_TEXT_PROGRESS||status==BN_NATIVE_TEXT_BUSY);
    int result=status==BN_NATIVE_TEXT_DONE?r.generated:-2;
    bn_generate_native_text_request_free(&r);return result;
}
int bn_generate_native_text(const BnModel *m,BnSession *s,
    const BnGenerateNativeTextWorkspace *workspace,const unsigned char *text,size_t bytes,
    const BnGenerateNativeTextOptions *options,const BnGenerateNativeTextCallbacks *callbacks) {
    return native_text_generate(m,s,workspace,text,bytes,options,callbacks,0);
}
int bn_generate_native_text_predict(const BnModel *m,BnSession *s,
    const BnGenerateNativeTextWorkspace *workspace,const unsigned char *text,size_t bytes,
    const BnGenerateNativeTextOptions *options,const BnGenerateNativeTextCallbacks *callbacks) {
    return native_text_generate(m,s,workspace,text,bytes,options,callbacks,1);
}
