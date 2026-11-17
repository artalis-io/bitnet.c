#include "trellis.h"
#include "quant.h"
#include <limits.h>
#include <stdlib.h>

typedef struct {
    const BnSafetensorsIndex *index;
    const void *index_state;
    size_t tensors[3];
    BnTrellisLayout layout;
} Source;

static const Source *get(const BnTrellisSource *source) {
    const Source *s=source?source->state:NULL;
    return s && s->index->state==s->index_state ? s : NULL;
}
int bn_trellis_source_init(BnTrellisSource *source,
    const BnSafetensorsIndex *index,int format,const size_t ids[4]) {
    if(!source || source->state || !index || !index->state || !ids)return -1;
    const BnQuantTrellisOps *ops=bn_quant_format_trellis_ops(format);
    if(!ops || ops->projection!=BN_QUANT_TRELLIS_PROJECTION_EXL3_MUL1_3_F16_F32)return -1;
    const BnSafetensorsTensor *t[4];
    for(int i=0;i<4;i++) {
        t[i]=bn_safetensors_index_tensor(index,ids[i]);if(!t[i])return -1;
        for(int j=0;j<i;j++)if(ids[i]==ids[j])return -1;
    }
    if(t[0]->dtype!=BN_SAFETENSORS_I16 || t[0]->n_dims!=3 ||
       !t[0]->shape[0] || !t[0]->shape[1] ||
       t[0]->shape[0]>(uint64_t)INT_MAX/16 || t[0]->shape[1]>(uint64_t)INT_MAX/16 ||
       t[0]->shape[2]!=ops->tile_bytes/2 || ops->tile_rows!=16 || ops->tile_cols!=16 ||
       ops->tile_bytes!=96 || t[0]->bytes>SIZE_MAX ||
       t[1]->dtype!=BN_SAFETENSORS_F16 || t[1]->n_dims!=1 ||
       t[2]->dtype!=BN_SAFETENSORS_F16 || t[2]->n_dims!=1 ||
       t[1]->shape[0]!=t[0]->shape[0]*16 || t[2]->shape[0]!=t[0]->shape[1]*16 ||
       t[1]->shape[0]%128 || t[2]->shape[0]%128 ||
       t[3]->dtype!=BN_SAFETENSORS_I32 || t[3]->n_dims!=0 || t[3]->bytes!=4 ||
       t[1]->bytes>SIZE_MAX || t[2]->bytes>SIZE_MAX)return -1;
    size_t packed=(size_t)t[0]->bytes,pre=(size_t)t[1]->bytes,post=(size_t)t[2]->bytes;
    if(packed>SIZE_MAX-pre || packed+pre>SIZE_MAX-post)return -1;
    unsigned char marker[4];
    if(bn_safetensors_index_read(index,ids[3],0,marker,4))return -1;
    uint32_t m=(uint32_t)marker[0]|((uint32_t)marker[1]<<8)|
               ((uint32_t)marker[2]<<16)|((uint32_t)marker[3]<<24);
    if(m!=ops->codebook_marker)return -1;
    Source *s=calloc(1,sizeof(*s));if(!s)return -1;
    s->index=index;s->index_state=index->state;
    for(int i=0;i<3;i++)s->tensors[i]=ids[i];
    s->layout=(BnTrellisLayout){format,(int)t[1]->shape[0],(int)t[2]->shape[0],m,
        packed,packed,packed+pre,packed+pre+post};
    source->state=s;return 0;
}
void bn_trellis_source_free(BnTrellisSource *source) {
    if(source){free(source->state);source->state=NULL;}
}
const BnTrellisLayout *bn_trellis_source_layout(const BnTrellisSource *source) {
    const Source*s=get(source);return s?&s->layout:NULL;
}
int bn_trellis_source_read(const BnTrellisSource *source,void *destination,size_t bytes) {
    const Source*s=get(source);
    if(!s || !destination || bytes!=s->layout.bundle_bytes)return -1;
    const size_t offsets[]={0,s->layout.pre_offset,s->layout.post_offset,bytes};
    for(int i=0;i<3;i++)
        if(bn_safetensors_index_read(s->index,s->tensors[i],0,
            (unsigned char*)destination+offsets[i],offsets[i+1]-offsets[i]))return -1;
    return 0;
}
int bn_trellis_bundle(const BnTrellisSource *source,const void *data,size_t bytes,BnTrellisBundle *out) {
    const Source*s=get(source);if(!s || !data || !out || bytes!=s->layout.bundle_bytes)return -1;
    BnTrellisBundle b={data,(const unsigned char*)data+s->layout.pre_offset,
        (const unsigned char*)data+s->layout.post_offset};*out=b;return 0;
}

typedef struct {
    const BnSafetensorsIndex *index;
    const void *index_state;
    size_t tensors[2];
    BnTrellisRowLayout layout;
} RowSource;

static const RowSource *row_get(const BnTrellisRowSource *source) {
    const RowSource *s=source?source->state:NULL;
    return s&&s->index->state==s->index_state?s:NULL;
}
int bn_trellis_row_source_init(BnTrellisRowSource *source,
    const BnSafetensorsIndex *index,int format,const size_t ids[2]) {
    if(!source||source->state||!index||!index->state||!ids||ids[0]==ids[1])return -1;
    const BnQuantRowScaledOps *ops=bn_quant_format_row_scaled_ops(format);
    if(!ops||ops->block_cols!=32||ops->weight_bits!=4||ops->scale_element_bytes!=2)return -1;
    const BnSafetensorsTensor *w=bn_safetensors_index_tensor(index,ids[0]);
    const BnSafetensorsTensor *s=bn_safetensors_index_tensor(index,ids[1]);
    if(!w||!s||w->dtype!=BN_SAFETENSORS_U8||s->dtype!=BN_SAFETENSORS_F16||
        w->n_dims!=2||s->n_dims!=2||!w->shape[0]||!w->shape[1]||
        w->shape[1]>(uint64_t)INT_MAX/2||w->shape[1]%16||
        s->shape[0]!=w->shape[0]||s->shape[1]!=w->shape[1]/16)return -1;
    size_t wb=(size_t)w->shape[1],sb=wb/8;
    if(wb>SIZE_MAX-sb||w->shape[0]>UINT64_MAX/wb||w->shape[0]>UINT64_MAX/sb||
        w->bytes!=w->shape[0]*wb||s->bytes!=w->shape[0]*sb)return -1;
    RowSource *r=calloc(1,sizeof(*r));if(!r)return -1;
    r->index=index;r->index_state=index->state;r->tensors[0]=ids[0];r->tensors[1]=ids[1];
    r->layout=(BnTrellisRowLayout){format,(int)(wb*2),w->shape[0],wb,sb,wb+sb};
    source->state=r;return 0;
}
void bn_trellis_row_source_free(BnTrellisRowSource *source) {
    if(source){free(source->state);source->state=NULL;}
}
const BnTrellisRowLayout *bn_trellis_row_source_layout(const BnTrellisRowSource *source) {
    const RowSource *s=row_get(source);return s?&s->layout:NULL;
}
int bn_trellis_row_source_read(const BnTrellisRowSource *source,
    uint64_t row,void *destination,size_t bytes) {
    const RowSource *s=row_get(source);
    if(!s||row>=s->layout.rows||!destination||bytes!=s->layout.bundle_bytes)return -1;
    if(bn_safetensors_index_read(s->index,s->tensors[0],row*s->layout.weight_bytes,
        destination,s->layout.weight_bytes))return -1;
    return bn_safetensors_index_read(s->index,s->tensors[1],row*s->layout.scale_bytes,
        (unsigned char *)destination+s->layout.weight_bytes,s->layout.scale_bytes);
}
