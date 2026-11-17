#ifndef __EMSCRIPTEN__
typedef struct {BnSession *session;int *released;int capacity;} NativeSessionRelease;
static void native_session_release(void *context) {
    NativeSessionRelease *r=context;
    assert(!bn_backend_session_has_gpu_resources(bn_session_backend(r->session)));
    assert(r->session->state.token_history[r->capacity-1]==37);
    (*r->released)++;free(r);
}
static void native_session_attach(BnSession *session,int *released,int capacity) {
    NativeSessionRelease *r=malloc(sizeof(*r));assert(r);
    *r=(NativeSessionRelease){session,released,capacity};
    assert(!bn_backend_session_attach_gpu_resources(bn_session_backend(session),
        released,r,native_session_release));
}
static void test_native_session_storage(void) {
    char dir[]="/tmp/bn-native-session-XXXXXX",path[256];assert(mkdtemp(dir));
    native_anatomy_fixture(dir,0);snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);
    BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));BnModel model={0};
    assert(!bn_model_load_trellis_config(&model,&index,native_anatomy_config,strlen(native_anatomy_config)));
    model.config.vocab_size=17;model.config.seq_len=23; /* A valid-looking legacy trap. */
    assert(!bn_session_create_backend(&model,NULL)&&!bn_session_create(&model,NULL));
    assert(!bn_model_prepare_native_anatomy(&model));
    BnModelNativeAnatomy *a=model.runtime->native_anatomy;assert(a&&a->vocab>17&&a->context>23);
    BnConfig saved=model.config;int vocab=a->vocab,capacity=a->context;
    BnSession *s[2]={bn_session_create_backend(&model,NULL),bn_session_create_backend(&model,NULL)};
    assert(s[0]&&s[1]&&s[0]->backend_only&&s[1]->backend_only&&!bn_session_create(&model,NULL));
    for(int q=0;q<2;q++) {
        assert(bn_session_matches_model(s[q],&model));
        assert(s[q]->backend_state->host_vocabulary==vocab&&s[q]->backend_state->host_capacity==capacity);
        assert(s[q]->backend_state->native_anatomy==a);
        assert(s[q]->state.logits&&s[q]->state.token_history&&!s[q]->state.x&&!s[q]->state.key_cache&&!s[q]->moe_state);
        s[q]->state.logits[vocab-1]=(float)(q+11);
        for(int i=0;i<capacity;i++)s[q]->state.token_history[i]=37;
        s[q]->pos=5;
    }
    assert(s[0]->state.logits!=s[1]->state.logits&&s[0]->state.token_history!=s[1]->state.token_history);
    int released[2]={0,0};for(int q=0;q<2;q++)native_session_attach(s[q],&released[q],capacity);
    BnModel copy=model;assert(!bn_session_matches_model(s[0],&copy));bn_session_reset(s[0],&copy);assert(s[0]->pos==5&&!released[0]);
    a->vocab++;assert(!bn_session_matches_model(s[0],&model));bn_session_reset(s[0],&model);assert(s[0]->pos==5&&!released[0]);a->vocab--;
    a->context++;assert(!bn_session_matches_model(s[0],&model));a->context--;
    BnModelNativeAnatomy *replacement=malloc(sizeof(*replacement));assert(replacement);*replacement=*a;model.runtime->native_anatomy=replacement;
    assert(!bn_session_matches_model(s[0],&model));bn_session_reset(s[0],&model);assert(!released[0]);model.runtime->native_anatomy=a;free(replacement);
    a->vocab=0;assert(!bn_session_create_backend(&model,NULL));a->vocab=vocab;
    a->context=-1;assert(!bn_session_create_backend(&model,NULL));a->context=capacity;
    model.config.vocab_size=1;model.config.seq_len=1;assert(bn_session_matches_model(s[0],&model));
    bn_session_reset(s[0],&model);assert(released[0]==1&&!s[0]->pos&&s[1]->pos==5&&!released[1]);
    for(int i=0;i<capacity;i++)assert(!s[0]->state.token_history[i]&&s[1]->state.token_history[i]==37);
    assert(s[0]->state.logits[vocab-1]==11&&s[1]->state.logits[vocab-1]==12);
    bn_session_reset(s[0],&model);assert(released[0]==1);
    /* Revoked borrowed source must not switch an existing session to legacy extents. */
    model.config.vocab_size=vocab;model.config.seq_len=capacity;
    bn_safetensors_index_free(&index);assert(!bn_session_matches_model(s[1],&model));
    assert(!bn_session_create_backend(&model,NULL)&&!bn_session_create(&model,NULL));
    bn_session_reset(s[1],&model);assert(s[1]->pos==5&&!released[1]);
    model.config=saved;bn_model_free(&model);
    bn_session_free(s[0],NULL);bn_session_free(s[1],NULL);assert(released[0]==1&&released[1]==1);
    snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!remove(path));
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!remove(path)&&!rmdir(dir));
    puts("Native session anatomy extents/identity/reset/revoked source/cleanup PASSED");
}
static void test_native_session_history(void) {
    char dir[]="/tmp/bn-native-history-XXXXXX",path[256];assert(mkdtemp(dir));
    native_anatomy_fixture(dir,0);snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);
    BnSafetensorsIndex index={0};assert(!bn_safetensors_index_init(&index,path));
    char *config=native_numerics_replace(native_anatomy_config,"\"engram_compressed_vocab_size\":99092","\"engram_compressed_vocab_size\":128");
    BnModel model={0};assert(!bn_model_load_trellis_config(&model,&index,config,strlen(config)));free(config);
    assert(!bn_model_prepare_native_anatomy(&model)&&!bn_model_prepare_native_numerics(&model));
    uint32_t map[256];for(int i=0;i<256;i++)map[i]=(uint32_t)i/2;
    size_t budget=99;assert(bn_session_native_history_plan(&model,4,&budget)==-1&&budget==99);
    assert(!bn_model_prepare_native_token_map(&model,map,256,sizeof(map)));
    assert(!bn_session_native_history_plan(&model,4,&budget));
    assert(bn_session_native_history_plan(&model,0,&budget)==-1);
    BnSession *s=bn_session_create_backend(&model,NULL),*other=bn_session_create_backend(&model,NULL);assert(s&&other);
    int tokens[]={8,10,2,14}; /* Compressed PAD is1, raw PAD is2. */
    BnSessionNativeHistoryView v,again;
    assert(bn_session_native_history_stage(s,&model,tokens,4)==-1);
    assert(bn_session_native_history_prepare(s,&model,4,budget-1)==-1&&!s->backend_state->native_history);
    assert(!bn_session_native_history_prepare(s,&model,4,budget));
    assert(bn_session_native_history_prepare(s,&model,4,budget)==-1);
    assert(!bn_session_native_history_prepare(other,&model,4,budget));
    assert(bn_session_native_history_view(s,&model,&v)==-1);
    assert(!bn_session_native_history_stage(s,&model,tokens,4));
    assert(!bn_session_native_history_view(s,&model,&v)&&v.prefix==3&&v.tokens==4&&v.position==0&&v.length==7&&v.host_bytes==budget);
    assert(v.raw&&v.raw!=tokens&&!memcmp(v.raw,tokens,4*sizeof(*tokens)));
    again=v;again.raw=tokens;assert(bn_session_native_history_commit(s,&model,&again)==-1&&!s->pos);
    uint32_t expected[]={1,1,1,4,5,1,7};assert(!memcmp(v.mapped,expected,sizeof(expected)));
    assert(!s->pos&&!s->state.token_history[0]&&!other->pos);
    assert(!bn_session_native_history_view(s,&model,&again)&&again.epoch==v.epoch&&again.mapped==v.mapped);
    assert(bn_session_native_history_stage(s,&model,tokens,1)==-1);
    again=v;again.epoch++;assert(bn_session_native_history_commit(s,&model,&again)==-1&&!s->pos);
    assert(!bn_session_native_history_abort(s,&model,&v)&&!s->pos);
    assert(bn_session_native_history_commit(s,&model,&v)==-1);
    assert(bn_session_native_history_stage(s,&model,(const int*)(const void*)v.mapped,1)==-1);
    assert(bn_session_native_history_stage(s,&model,tokens,0)==-1);
    assert(bn_session_native_history_stage(s,&model,tokens,5)==-1);
    int bad[]={8,10,2,-1};assert(bn_session_native_history_stage(s,&model,bad,4)==-1&&!s->pos);
    assert(!memcmp(v.mapped,expected,sizeof(expected)));
    assert(!bn_session_native_history_stage(s,&model,tokens,4));assert(!bn_session_native_history_view(s,&model,&again));
    assert(again.epoch!=v.epoch&&bn_session_native_history_commit(s,&model,&v)==-1);
    assert(!bn_session_native_history_commit(s,&model,&again)&&s->pos==4);
    for(int i=0;i<4;i++)assert(s->state.token_history[i]==tokens[i]);
    tokens[0]=18;assert(!bn_session_native_history_stage(s,&model,tokens,1));assert(!bn_session_native_history_view(s,&model,&v));
    uint32_t continued[]={5,1,7,9};assert(!memcmp(v.mapped,continued,sizeof(continued)));
    assert(!bn_session_native_history_abort(s,&model,&v));
    assert(bn_session_native_history_rewind(s,&model,2)==-1&&s->pos==4);
    s->pos=2; /* Simulated executor restoration; no claim of GPU rewind. */
    assert(!bn_session_native_history_rewind(s,&model,2));
    assert(!s->state.token_history[2]&&!s->state.token_history[3]);
    assert(!bn_session_native_history_stage(s,&model,tokens,1));assert(!bn_session_native_history_view(s,&model,&v));
    uint32_t rewound[]={1,4,5,9};assert(!memcmp(v.mapped,rewound,sizeof(rewound)));
    bn_session_reset(s,&model);assert(!s->pos&&bn_session_native_history_commit(s,&model,&v)==-1);
    assert(!bn_session_native_history_stage(s,&model,tokens,1));assert(!bn_session_native_history_view(s,&model,&again));
    assert(again.epoch!=v.epoch&&again.mapped[0]==1&&again.mapped[1]==1&&again.mapped[2]==1);
    BnModel copy=model;assert(bn_session_native_history_commit(s,&copy,&again)==-1);
    assert(!bn_session_native_history_abort(s,&model,&again));
    assert(!bn_session_native_history_stage(other,&model,tokens,1));assert(!bn_session_native_history_view(other,&model,&v));
    assert(bn_session_native_history_commit(s,&model,&v)==-1);
    assert(!bn_session_native_history_commit(other,&model,&v)&&other->pos==1&&!s->pos);
    BnSessionNativeHistoryInfo before_restart;
    assert(!bn_session_native_history_info(other,&model,&before_restart)&&before_restart.committed==1);
    assert(!bn_session_native_history_restart(other,&model)&&!other->pos);
    assert(bn_session_native_history_commit(other,&model,&v)==-1);
    assert(!bn_session_native_history_stage(other,&model,tokens,1));
    assert(!bn_session_native_history_view(other,&model,&again)&&again.epoch!=v.epoch);
    assert(bn_session_native_history_restart(other,&model)==-1&&!other->pos);
    assert(!bn_session_native_history_abort(other,&model,&again));
    BnSession *cohort[2]={s,other};BnSessionNativeHistoryView tickets[2];
    assert(!bn_session_native_history_stage(s,&model,tokens,2)&&!bn_session_native_history_view(s,&model,tickets));
    assert(!bn_session_native_history_stage(other,&model,tokens+2,1)&&!bn_session_native_history_view(other,&model,tickets+1));
    BnSessionNativeHistoryView invalid[2]={tickets[0],tickets[1]};invalid[1].epoch++;
    assert(bn_session_native_history_commit_group(&model,cohort,invalid,2)==-1&&!s->pos&&!other->pos);
    BnSession *duplicate[2]={s,s};BnSessionNativeHistoryView duplicate_tickets[2]={tickets[0],tickets[0]};
    assert(bn_session_native_history_commit_group(&model,duplicate,duplicate_tickets,2)==-1);
    assert(!bn_session_native_history_commit_group(&model,cohort,tickets,2)&&s->pos==2&&other->pos==1);
    assert(!memcmp(s->state.token_history,tokens,2*sizeof(int))&&other->state.token_history[0]==tokens[2]);
    assert(bn_session_native_history_commit_group(&model,cohort,tickets,2)==-1);
    assert(bn_session_native_history_commit_group(&model,cohort,tickets,17)==-1);
    bn_session_reset(s,&model);bn_session_reset(other,&model);
    for(int chunk=1;chunk<=4;chunk++) {
        bn_session_reset(s,&model);
        for(int pos=0;pos<67;) {
            int n=67-pos<chunk?67-pos:chunk,raw[4];
            for(int i=0;i<n;i++)raw[i]=((pos+i)*17+2)%256;
            assert(!bn_session_native_history_stage(s,&model,raw,n));
            assert(!bn_session_native_history_view(s,&model,&v));
            for(int i=0;i<v.prefix+n;i++) {
                int at=pos-v.prefix+i;
                uint32_t want=at<0?1:(uint32_t)((at*17+2)%256)/2;
                assert(v.mapped[i]==want);
            }
            /* Input lifetime ends at stage; commit uses its owned raw copy. */
            for(int i=0;i<n;i++)raw[i]=-1;
            assert(!bn_session_native_history_commit(s,&model,&v));pos+=n;
        }
        for(int i=0;i<67;i++)assert(s->state.token_history[i]==(i*17+2)%256);
    }
    bn_session_reset(s,&model);
    int cap=s->backend_state->host_capacity;
    for(int pos=0;pos<cap;) {
        int n=cap-pos<4?cap-pos:4,raw[4]={4,6,8,10};
        assert(!bn_session_native_history_stage(s,&model,raw,n));
        assert(!bn_session_native_history_view(s,&model,&v));
        assert(!bn_session_native_history_commit(s,&model,&v));pos+=n;
    }
    assert(s->pos==cap&&bn_session_native_history_stage(s,&model,tokens,1)==-1);
    bn_session_reset(s,&model);
    bn_session_free(s,NULL);bn_session_free(other,NULL);bn_model_free(&model);bn_safetensors_index_free(&index);
    snprintf(path,sizeof(path),"%s/one.safetensors",dir);assert(!remove(path));
    snprintf(path,sizeof(path),"%s/model.safetensors.index.json",dir);assert(!remove(path)&&!rmdir(dir));
    puts("Native session mapped history stage/abort/commit/padding/rewind/reset/isolation PASSED");
}

#else
static void test_native_session_history(void) {}
static void test_native_session_storage(void) {}
#endif
