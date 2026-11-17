/* Main-owned batch transport. Strings borrow the immutable input document. */
#include "json.h"
#define NATIVE_CLI_BATCH_REQUESTS 64
typedef struct {
    BnJSON json;
    int count;
    const char *prompts[NATIVE_CLI_BATCH_REQUESTS];
} NativeCLIBatchInput;
static inline int native_cli_batch_input_init(NativeCLIBatchInput *in,const void *data,size_t bytes) {
    if(!in||in->json.state||in->count||bn_json_init(&in->json,data,bytes))return -1;
    const BnJSONValue *root=bn_json_root(&in->json);size_t n=bn_json_size(root);
    const char *texts[NATIVE_CLI_BATCH_REQUESTS];
    if(bn_json_type(root)!=BN_JSON_ARRAY||!n||n>NATIVE_CLI_BATCH_REQUESTS)goto fail;
    for(size_t i=0;i<n;i++) {
        const BnJSONValue *v=bn_json_at(root,i);texts[i]=bn_json_text(v);
        if(bn_json_type(v)!=BN_JSON_STRING||!texts[i]||!*texts[i])goto fail;
    }
    memcpy(in->prompts,texts,n*sizeof(*texts));in->count=(int)n;return 0;
fail:
    bn_json_free(&in->json);return -1;
}
static inline int native_cli_batch_json_text(FILE *out,const unsigned char *text,size_t bytes) {
    if(fputc('"',out)==EOF)return -1;
    for(size_t i=0;i<bytes;i++) {
        unsigned c=text[i];
        if(c=='"'||c=='\\'){if(fputc('\\',out)==EOF||fputc((int)c,out)==EOF)return -1;}
        else if(c<32){if(fprintf(out,"\\u%04x",c)<0)return -1;}
        else if(fputc((int)c,out)==EOF)return -1;
    }
    return fputc('"',out)==EOF?-1:0;
}
