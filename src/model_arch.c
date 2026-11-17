#include "model_arch.h"
#include "model_tokenizer_policy.h"
#include "quant.h"
#include <limits.h>
#include <math.h>
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *bn_model_arch_prefix(const char *arch) {
    return arch && arch[0] ? arch : "llama";
}

static int native_config_string(const BnJSONValue *object, const char *key,
    const char *expected) {
    const BnJSONValue *v = bn_json_get(object, key);
    return bn_json_type(v) == BN_JSON_STRING && !strcmp(bn_json_text(v), expected);
}

static int native_config_integer(const BnJSONValue *object, const char *key,
    int minimum, int maximum, int *out) {
    uint64_t n;
    if (bn_json_u64(bn_json_get(object, key), &n) ||
        n < (uint64_t)minimum || n > (uint64_t)maximum) return -1;
    *out = (int)n;
    return 0;
}

int bn_model_arch_native_expert_config(const BnJSON *document,
    BnModelNativeExpertConfig *out) {
    if (!out) return -1;
    const BnJSONValue *root = bn_json_root(document);
    const BnJSONValue *text = bn_json_get(root, "text_config");
    const BnJSONValue *quant = bn_json_get(root, "quantization_config");
    const BnJSONValue *architectures = bn_json_get(root, "architectures");
    const BnJSONValue *architecture = bn_json_at(architectures, 0);
    if (!native_config_string(root, "model_type", "deepseek_v41") ||
        !native_config_string(text, "model_type", "deepseek_v41_text") ||
        bn_json_size(architectures) != 1 || bn_json_type(architecture) != BN_JSON_STRING ||
        strcmp(bn_json_text(architecture), "DeepseekV41ForCausalLM") ||
        !native_config_string(quant, "quant_method", "exl3") ||
        !native_config_string(quant, "codebook", "mul1")) return -1;
    BnModelNativeExpertConfig c = {0};
    int bits;
    if (native_config_integer(quant, "bits", 3, 3, &bits) ||
        native_config_integer(text, "num_hidden_layers", 1, BN_MODEL_ANATOMY_MAX_LAYERS, &c.layers) ||
        native_config_integer(text, "n_routed_experts", 1, INT_MAX, &c.experts) ||
        native_config_integer(text, "num_experts_per_tok", 1, c.experts, &c.active_experts) ||
        native_config_integer(text, "hidden_size", 128, INT_MAX, &c.dim) ||
        native_config_integer(text, "moe_intermediate_size", 128, INT_MAX, &c.hidden) ||
        native_config_integer(text, "num_nextn_predict_layers", 0, BN_MODEL_ANATOMY_MAX_LAYERS, &c.nextn_layers) ||
        c.dim % 128 || c.hidden % 128) return -1;
    const BnJSONValue *layer_bits = bn_json_get(quant, "layer_bits");
    if (bn_json_type(layer_bits) != BN_JSON_OBJECT) return -1;
    for (int l = 0; l < c.layers; l++) {
        char key[32];
        snprintf(key, sizeof(key), "%d", l);
        if (native_config_integer(layer_bits, key, 3, 3, &bits)) return -1;
    }
    memcpy(c.architecture, "deepseek_v41", sizeof("deepseek_v41"));
    c.format = BN_QUANT_FORMAT_EXL3_MUL1_3;
    *out = c;
    return 0;
}

static int native_product(int a,int b,int *out) {
    if(a<=0||b<=0||a>INT_MAX/b)return -1;
    *out=a*b;return 0;
}
static int native_flags(const BnJSONValue *text,const char *key,int layers,int *flags) {
    const BnJSONValue *a=bn_json_get(text,key);
    if(bn_json_type(a)!=BN_JSON_ARRAY||bn_json_size(a)>(size_t)layers)return -1;
    for(size_t i=0;i<bn_json_size(a);i++) {
        uint64_t v;if(bn_json_u64(bn_json_at(a,i),&v)||v>=(uint64_t)layers||
            flags[v])return -1;
        flags[v]=1;
    }
    return 0;
}
static int native_bool_setting(const BnJSONValue *object,const char *key,int expected) {
    int v;return bn_json_bool(bn_json_get(object,key),&v)||v!=expected?-1:0;
}
/* Compare a parsed JSON number to digits * 10^power without binary rounding.
 * Significant digits exclude decimal point and exterior zeroes. Exponents too
 * large to cancel a bounded document's digits cannot match this profile. */
static int native_decimal(const BnJSONValue *object,const char *key,
    const char *digits,int power) {
    const BnJSONValue *v=bn_json_get(object,key);
    if(bn_json_type(v)!=BN_JSON_NUMBER)return 0;
    const char *s=bn_json_text(v),*p=s;int negative=*p=='-';if(negative)p++;
    const char *begin=p;int total=0,fraction=0,dot=0,first=-1,last=-1;
    for(;*p&&*p!='e'&&*p!='E';p++) {
        if(*p=='.'){dot=1;continue;}
        if(*p!='0'){if(first<0)first=total;last=total;}
        total++;fraction+=dot;
    }
    if(first<0)return !strcmp(digits,"0");
    if(negative||last-first+1!=(int)strlen(digits))return 0;
    int exponent=0,sign=1;
    if(*p){p++;if(*p=='-'||*p=='+'){if(*p=='-')sign=-1;p++;}
        for(;*p;p++) {
            if(exponent>(int)(2u*BN_JSON_MAX_BYTES+128u)/10)return 0;
            exponent=exponent*10+(*p-'0');
        }
    }
    if(sign*exponent-fraction+total-last-1!=power)return 0;
    int index=0,matched=0;
    for(p=begin;*p&&*p!='e'&&*p!='E';p++)if(*p!='.') {
        if(index>=first&&index<=last&&*p!=digits[matched++])return 0;
        index++;
    }
    return 1;
}

int bn_model_arch_native_numerics(const BnJSON *document,BnModelNativeNumerics *out) {
    if(!out)return -1;
    BnModelNativeExpertConfig experts;
    if(bn_model_arch_native_expert_config(document,&experts))return -1;
    const BnJSONValue *root=bn_json_root(document),*text=bn_json_get(root,"text_config");
    const BnJSONValue *rope=bn_json_get(text,"rope_scaling");
    BnModelNativeNumerics p={0};int vocab,unused;
    if(!native_config_string(root,"dtype","bfloat16")||
        !native_config_string(text,"hidden_act","silu")||
        !native_config_string(text,"scoring_func","sqrtsoftplus")||
        !native_config_string(text,"topk_method","noaux_tc")||
        !native_config_string(rope,"rope_type","yarn")||
        native_bool_setting(text,"norm_topk_prob",1)||
        native_bool_setting(text,"attention_bias",0)||
        native_bool_setting(text,"tie_word_embeddings",0)||
        native_bool_setting(text,"use_cache",1)||
        !native_decimal(text,"attention_dropout","0",0)||
        !native_decimal(text,"rms_norm_eps","1",-20)||
        !native_decimal(text,"hc_eps","1",-6)||
        !native_decimal(text,"swiglu_limit","1",1)||
        !native_decimal(text,"routed_scaling_factor","15",-1)||
        !native_decimal(text,"rope_theta","1",4)||
        !native_decimal(text,"compress_rope_theta","16",4)||
        !native_decimal(rope,"factor","16",0)||
        !native_decimal(rope,"beta_fast","32",0)||
        !native_decimal(rope,"beta_slow","1",0)||
        native_config_integer(text,"hc_mult",4,4,&p.hc_streams)||
        native_config_integer(text,"hc_sinkhorn_iters",20,20,&p.hc_iterations)||
        native_config_integer(rope,"original_max_position_embeddings",65536,65536,&p.yarn_original_context)||
        native_config_integer(text,"max_position_embeddings",1,INT_MAX,&p.context)||
        native_config_integer(text,"vocab_size",1,INT_MAX,&vocab)||
        native_config_integer(root,"bos_token_id",0,vocab-1,&p.bos_token)||
        native_config_integer(root,"eos_token_id",0,vocab-1,&p.eos_token)||
        native_config_integer(root,"pad_token_id",0,vocab-1,&p.pad_token))return -1;
    /* No alternate hash-routing path in this execution profile. */
    const BnJSONValue *hash=bn_json_get(text,"num_hash_layers");uint64_t hash_layers;
    if(hash&&(bn_json_u64(hash,&hash_layers)||hash_layers))return -1;
    const BnJSONValue *engram=bn_json_get(text,"engram_layer_ids");
    if(bn_json_type(engram)!=BN_JSON_ARRAY)return -1;
    if(bn_json_size(engram)&&(
        native_config_integer(text,"engram_vocab_size",2,INT_MAX,&p.engram_vocab)||
        native_config_integer(text,"engram_compressed_vocab_size",1,vocab,&p.engram_compressed_vocab)||
        native_config_integer(text,"engram_pad_token_id",0,vocab-1,&p.engram_pad_token)))return -1;
    /* This profile has exactly the five validated YaRN settings. */
    if(bn_json_size(rope)!=5)return -1;
    if(native_config_integer(text,"num_key_value_heads",1,1,&unused))return -1;
    p.rms_epsilon=1e-20f;p.hc_epsilon=1e-6f;p.swiglu_limit=10.0f;p.routed_scale=1.5f;
    p.rope_theta=10000.0f;p.compress_rope_theta=160000.0f;
    p.yarn_factor=16.0f;p.yarn_beta_fast=32.0f;p.yarn_beta_slow=1.0f;
    p.rope_attention_factor=1.0f;p.main_rope_yarn=0;p.compress_rope_yarn=1;p.yarn_truncate=1;
    p.hc_carry_pre=1;p.norm_topk=1;
    *out=p;return 0;
}

static int native_tensor(const BnSafetensorsIndex *index,const char *prefix,
    const char *suffix,BnSafetensorsDtype dtype,int rank,uint64_t d0,uint64_t d1) {
    char name[160];int n=snprintf(name,sizeof(name),"%s%s",prefix,suffix);
    if(n<0||(size_t)n>=sizeof(name))return -1;
    int id=bn_safetensors_index_find(index,name);if(id<0)return -1;
    const BnSafetensorsTensor *t=bn_safetensors_index_tensor(index,(size_t)id);
    if(!t||t->dtype!=dtype||t->n_dims!=rank||t->shape[0]!=d0||
        (rank==2&&t->shape[1]!=d1))return -1;
    return id;
}
static int native_fp8_pair(const BnSafetensorsIndex *index,const char *prefix,
    const char *key,int rows,int cols,int *weight,int *scale,
    BnModelNativeAnatomy *plan,int layer,int weight_role,int scale_role) {
    const BnQuantBlockScaledOps *ops=bn_quant_format_block_scaled_ops(BN_QUANT_FORMAT_E4M3_UE8M0_32);
    if(!ops||ops->decode!=BN_QUANT_BLOCK_SCALED_E4M3_UE8M0_32_F16||
        ops->block_rows<=0||ops->block_cols<=0||ops->weight_element_bytes!=1||
        ops->scale_element_bytes!=1)return -1;
    char suffix[96];int n=snprintf(suffix,sizeof(suffix),"%s.weight",key);
    if(n<0||(size_t)n>=sizeof(suffix))return -1;
    int w=native_tensor(index,prefix,suffix,BN_SAFETENSORS_F8_E4M3,2,(uint64_t)rows,(uint64_t)cols);
    n=snprintf(suffix,sizeof(suffix),"%s.scale",key);
    if(n<0||(size_t)n>=sizeof(suffix))return -1;
    int s=native_tensor(index,prefix,suffix,BN_SAFETENSORS_F8_E8M0,2,
        ((uint64_t)rows+(unsigned)ops->block_rows-1)/(unsigned)ops->block_rows,
        ((uint64_t)cols+(unsigned)ops->block_cols-1)/(unsigned)ops->block_cols);
    if(w<0||s<0)return -1;
    if(plan->core_projection_count>=BN_MODEL_NATIVE_MAX_CORE_PROJECTIONS)return -1;
    plan->core_projections[plan->core_projection_count++]=(BnModelNativeCoreProjection){
        layer,weight_role,scale_role,w,s,rows,cols};
    *weight=w;*scale=s;return 0;
}
int bn_model_arch_native_rope_frequency(int dim,const BnModelNativeNumerics*p,
    int compressed,BnModelNativeRopeFrequency*out) {
    if(!p||!out||fegetround()!=FE_TONEAREST||dim<2||dim>2048||dim%2||(compressed!=0&&compressed!=1))return -1;
    int yarn=compressed?p->compress_rope_yarn:p->main_rope_yarn;
    float base=compressed?p->compress_rope_theta:p->rope_theta;
    if((yarn!=0&&yarn!=1)||!(base>1.f&&base<=1e9f))return -1;
    BnModelNativeRopeFrequency w={dim,yarn,base,1.f,0.f,1.f};
    if(yarn) {
        if(!(p->yarn_factor>=1.f&&p->yarn_factor<=65536.f)||p->yarn_original_context<1||
           !(p->yarn_beta_fast>0.f&&p->yarn_beta_fast<=1e9f)||
           !(p->yarn_beta_slow>0.f&&p->yarn_beta_slow<=p->yarn_beta_fast)||
           (p->yarn_truncate!=0&&p->yarn_truncate!=1))return -1;
        const double pi=3.14159265358979323846;
        double den=2.0*log((double)base),low=(dim*log(p->yarn_original_context/((double)p->yarn_beta_fast*2.0*pi)))/den,
            high=(dim*log(p->yarn_original_context/((double)p->yarn_beta_slow*2.0*pi)))/den;
        if(p->yarn_truncate){low=floor(low);high=ceil(high);}
        if(low<0){low=0;}if(high>dim-1){high=dim-1;}
        if(low==high)high+=0.001;
        if(!isfinite(low)||!isfinite(high)||low>dim-1||high<=low)return -1;
        w.factor=p->yarn_factor;w.ramp_start=(float)low;w.ramp_scale=(float)(1.0/(high-low));
        if(!(w.ramp_scale>0.f&&w.ramp_scale<=1e9f))return -1;
    }
    *out=w;return 0;
}

static int native_parameter(BnModelNativeAnatomy *p,const BnSafetensorsIndex *index,
    int layer,int role,int id) {
    const BnSafetensorsTensor *t=bn_safetensors_index_tensor(index,(size_t)id);
    if(!t)return -1;
    if(t->dtype!=BN_SAFETENSORS_BF16&&t->dtype!=BN_SAFETENSORS_F32)return 0;
    if(p->parameter_count>=BN_MODEL_NATIVE_MAX_PARAMETERS||t->n_dims<1||t->n_dims>2||
        !t->shape[0]||t->shape[0]>INT_MAX||
        (t->n_dims==2&&(!t->shape[1]||t->shape[1]>INT_MAX)))return -1;
    int linear=layer<0?role==BN_NATIVE_GLOBAL_OUTPUT:
        role==BN_NATIVE_ROUTER||role==BN_NATIVE_COMP_KV||role==BN_NATIVE_COMP_GATE||
        role==BN_NATIVE_INDEX_K||role==BN_NATIVE_INDEX_WEIGHTS;
    if(linear&&(t->dtype!=BN_SAFETENSORS_BF16||t->n_dims!=2))return -1;
    p->parameters[p->parameter_count++]=(BnModelNativeParameter){layer,role,id,
        linear?BN_GGUF_TENSOR_F16:t->dtype==BN_SAFETENSORS_BF16?BN_GGUF_TENSOR_BF16:BN_GGUF_TENSOR_F32,
        t->n_dims==2?(int)t->shape[0]:1,(int)t->shape[t->n_dims-1],
        linear?BN_NATIVE_PARAMETER_F16_LINEAR:BN_NATIVE_PARAMETER_RAW};
    return 0;
}

int bn_model_arch_native_anatomy(const BnJSON *document,
    const BnSafetensorsIndex *index,BnModelNativeAnatomy *out) {
    if(!index||!index->state||!out)return -1;
    BnModelNativeAnatomy p={0};
    p.core_format=BN_QUANT_FORMAT_E4M3_UE8M0_32;
    const BnQuantBlockScaledOps *core=bn_quant_format_block_scaled_ops(p.core_format);
    if(!core)return -1;
    if(bn_model_arch_native_expert_config(document,&p.experts))return -1;
    const BnJSONValue *root=bn_json_root(document),*text=bn_json_get(root,"text_config");
    const BnJSONValue *quant=bn_json_get(root,"quantization_config");
#define NI(F,K,MIN,MAX) if(native_config_integer(text,K,MIN,MAX,&p.F))return -1
    NI(vocab,"vocab_size",1,INT_MAX);NI(context,"max_position_embeddings",1,INT_MAX);
    NI(heads,"num_attention_heads",1,INT_MAX);NI(head_dim,"head_dim",1,INT_MAX);
    NI(rope_dim,"qk_rope_head_dim",2,INT_MAX);NI(q_rank,"q_lora_rank",1,INT_MAX);
    NI(o_groups,"o_groups",1,INT_MAX);NI(o_rank,"o_lora_rank",1,INT_MAX);
    NI(streams,"hc_mult",1,16);NI(index_heads,"index_n_heads",1,INT_MAX);
    NI(index_dim,"index_head_dim",1,INT_MAX);NI(index_topk,"index_topk",1,INT_MAX);
    NI(sliding_window,"sliding_window",1,INT_MAX);
    NI(candidate_blocks,"candidate_topk_blocks",1,INT_MAX);
    NI(candidate_block_size,"candidate_block_size",1,INT_MAX);
#undef NI
    int kv_heads,shared,shared_hidden,q_rows,o_rows,o_cols,index_rows,streams_dim;
    if(native_config_integer(text,"num_key_value_heads",1,1,&kv_heads)||
        native_config_integer(text,"n_shared_experts",1,INT_MAX,&shared)||
        native_product(shared,p.experts.hidden,&shared_hidden)||
        native_product(p.heads,p.head_dim,&q_rows)||p.heads%p.o_groups||
        native_product(p.o_groups,p.o_rank,&o_rows)||
        native_product(p.heads/p.o_groups,p.head_dim,&o_cols)||
        native_product(p.index_heads,p.index_dim,&index_rows)||
        native_product(p.streams,p.experts.dim,&streams_dim)||
        p.rope_dim>p.head_dim||p.rope_dim%2||
        !native_config_string(text,"hidden_act","silu")||
        !native_config_string(text,"scoring_func","sqrtsoftplus")||
        !native_config_string(text,"topk_method","noaux_tc")||
        native_bool_setting(text,"norm_topk_prob",1)||
        native_bool_setting(text,"attention_bias",0)||
        native_bool_setting(text,"tie_word_embeddings",0))return -1;
    uint64_t hash_layers=0;
    const BnJSONValue *hash=bn_json_get(text,"num_hash_layers");
    if(hash&&(bn_json_u64(hash,&hash_layers)||hash_layers))return -1;
    const BnJSONValue *nonrouted=bn_json_get(quant,"non_routed_quantization");
    const BnJSONValue *block=bn_json_get(nonrouted,"weight_block_size");
    uint64_t b0,b1;
    if(!native_config_string(nonrouted,"quant_method","deepseek_v4_fp8")||
        !native_config_string(nonrouted,"fmt","e4m3")||
        !native_config_string(nonrouted,"scale_fmt","ue8m0")||
        bn_json_size(block)!=2||bn_json_u64(bn_json_at(block,0),&b0)||
        bn_json_u64(bn_json_at(block,1),&b1)||b0!=(uint64_t)core->block_rows||
        b1!=(uint64_t)core->block_cols)return -1;
    int kv_flags[BN_MODEL_ANATOMY_MAX_LAYERS]={0},idx_flags[BN_MODEL_ANATOMY_MAX_LAYERS]={0};
    if(native_flags(text,"kv_source_layer_ids",p.experts.layers,kv_flags)||
        native_flags(text,"index_source_layer_ids",p.experts.layers,idx_flags))return -1;
    const BnJSONValue *ratios=bn_json_get(text,"compress_ratios");
    if(bn_json_type(ratios)!=BN_JSON_ARRAY||bn_json_size(ratios)<(size_t)p.experts.layers)return -1;
    for(int l=0;l<BN_MODEL_ANATOMY_MAX_LAYERS;l++) {
        p.layers[l].kv_source=p.layers[l].index_source=-1;
        for(int r=0;r<BN_NATIVE_ROLE_COUNT;r++)p.layers[l].tensor[r]=-1;
    }
    int latest_kv=-1,latest_index=-1;
    for(int l=0;l<p.experts.layers;l++) {
        uint64_t ratio;if(bn_json_u64(bn_json_at(ratios,(size_t)l),&ratio)||ratio>INT_MAX)return -1;
        BnModelNativeLayerAnatomy *v=&p.layers[l];v->compress_ratio=(int)ratio;
        if(!ratio){if(kv_flags[l]||idx_flags[l])return -1;continue;}
        if(kv_flags[l])latest_kv=l;
        v->kv_source=latest_kv<0?l:latest_kv;
        if(p.layers[v->kv_source].compress_ratio!=v->compress_ratio)return -1;
        if(idx_flags[l])latest_index=l;
        if(latest_index<0||p.layers[latest_index].kv_source!=v->kv_source||
            (v->kv_source==l&&!idx_flags[l]))return -1;
        v->index_source=latest_index;
    }
    p.candidate_source=-1;
    const BnJSONValue *candidate=bn_json_get(text,"candidate_source_layer_id");
    if(!candidate)return -1;
    if(bn_json_type(candidate)!=BN_JSON_NUMBER||strcmp(bn_json_text(candidate),"-1")) {
        if(native_config_integer(text,"candidate_source_layer_id",0,p.experts.layers-1,&p.candidate_source))return -1;
    }
    if(p.candidate_source>=0) {
        int c=p.candidate_source;
        if(!idx_flags[c]||!p.layers[c].compress_ratio||
            (uint64_t)p.candidate_blocks*(uint64_t)p.candidate_block_size<(uint64_t)p.index_topk)return -1;
        for(int l=c;l<p.experts.layers;l++)if(idx_flags[l]) {
            p.layers[l].candidate_mode=l==c?BN_NATIVE_CANDIDATE_SOURCE:BN_NATIVE_CANDIDATE_USE;
        }
    }
    const BnJSONValue *engram=bn_json_get(text,"engram_layer_ids"),*rows=bn_json_get(text,"engram_num_embeddings");
    if(bn_json_type(engram)!=BN_JSON_ARRAY||bn_json_type(rows)!=BN_JSON_ARRAY||
        bn_json_size(engram)!=bn_json_size(rows)||bn_json_size(engram)>(size_t)p.experts.layers)return -1;
    int engram_width=0,engram_output=0;
    if(bn_json_size(engram)) {
        p.engram_format=BN_QUANT_FORMAT_INT4_FP16S32;
        if(!native_config_string(quant,"engram_embed_format","int4_fp16s32")||
            native_config_integer(text,"engram_max_ngram_size",2,INT_MAX,&p.engram_ngram)||
            native_config_integer(text,"engram_n_heads",1,INT_MAX,&p.engram_heads)||
            native_config_integer(text,"engram_head_dim",32,INT_MAX,&p.engram_dim)||p.engram_dim%32||
            native_product(p.engram_ngram-1,p.engram_heads,&engram_width)||
            native_product(engram_width,p.engram_dim,&engram_width)||
            native_product(p.streams+1,p.experts.dim,&engram_output))return -1;
        for(size_t i=0;i<bn_json_size(engram);i++) {
            uint64_t id,n;if(bn_json_u64(bn_json_at(engram,i),&id)||id>=(uint64_t)p.experts.layers||
                p.layers[id].engram_rows||bn_json_u64(bn_json_at(rows,i),&n)||!n)return -1;
            p.layers[id].engram_rows=n;
        }
    }
    p.embedding=native_tensor(index,"","embed.weight",BN_SAFETENSORS_BF16,2,p.vocab,p.experts.dim);
    p.output=native_tensor(index,"","head.weight",BN_SAFETENSORS_BF16,2,p.vocab,p.experts.dim);
    p.output_norm=native_tensor(index,"","norm.weight",BN_SAFETENSORS_BF16,1,p.experts.dim,0);
    if(p.embedding<0||p.output<0||p.output_norm<0)return -1;
    if(native_parameter(&p,index,-1,BN_NATIVE_GLOBAL_EMBEDDING,p.embedding)||
        native_parameter(&p,index,-1,BN_NATIVE_GLOBAL_OUTPUT,p.output)||
        native_parameter(&p,index,-1,BN_NATIVE_GLOBAL_OUTPUT_NORM,p.output_norm))return -1;
    for(int l=0;l<p.experts.layers;l++) {
        BnModelNativeLayerAnatomy *v=&p.layers[l];int *t=v->tensor;char prefix[32];snprintf(prefix,sizeof(prefix),"layers.%d.",l);
#define B(R,K,D,N,A,Z) do {t[BN_NATIVE_##R]=native_tensor(index,prefix,K,BN_SAFETENSORS_##D,N,(uint64_t)(A),(uint64_t)(Z));if(t[BN_NATIVE_##R]<0||native_parameter(&p,index,l,BN_NATIVE_##R,t[BN_NATIVE_##R]))return -1;}while(0)
#define F(R,K,A,Z) do {if(native_fp8_pair(index,prefix,K,A,Z,&t[BN_NATIVE_##R],&t[BN_NATIVE_##R##_SCALE],&p,l,BN_NATIVE_##R,BN_NATIVE_##R##_SCALE))return -1;}while(0)
        B(ATTN_NORM,"attn_norm.weight",BF16,1,p.experts.dim,0);B(FFN_NORM,"ffn_norm.weight",BF16,1,p.experts.dim,0);
        int hc_rows=p.streams*p.streams+2*p.streams;
        B(HC_ATTN_FN,"hc_attn_fn",F32,2,hc_rows,streams_dim);B(HC_ATTN_BASE,"hc_attn_base",F32,1,hc_rows,0);B(HC_ATTN_SCALE,"hc_attn_scale",F32,1,3,0);
        B(HC_FFN_FN,"hc_ffn_fn",F32,2,hc_rows,streams_dim);B(HC_FFN_BASE,"hc_ffn_base",F32,1,hc_rows,0);B(HC_FFN_SCALE,"hc_ffn_scale",F32,1,3,0);
        F(Q_A,"attn.wq_a",p.q_rank,p.experts.dim);F(Q_B,"attn.wq_b",q_rows,p.q_rank);F(KV,"attn.wkv",p.head_dim,p.experts.dim);F(O_A,"attn.wo_a",o_rows,o_cols);F(O_B,"attn.wo_b",p.experts.dim,o_rows);
        B(Q_NORM,"attn.q_norm.weight",BF16,1,p.q_rank,0);B(KV_NORM,"attn.kv_norm.weight",BF16,1,p.head_dim,0);B(ATTN_SINK,"attn.attn_sink",F32,1,p.heads,0);
        B(ROUTER,"ffn.gate.weight",BF16,2,p.experts.experts,p.experts.dim);B(ROUTER_BIAS,"ffn.gate.bias",F32,1,p.experts.experts,0);B(ROUTER_BIAS_VL,"ffn.gate.bias_vl",F32,1,p.experts.experts,0);
        F(SHARED_GATE,"ffn.shared_experts.w1",shared_hidden,p.experts.dim);F(SHARED_UP,"ffn.shared_experts.w3",shared_hidden,p.experts.dim);F(SHARED_DOWN,"ffn.shared_experts.w2",p.experts.dim,shared_hidden);
        if(v->kv_source==l) {
            B(COMP_KV,"attn.compressor.wkv.weight",BF16,2,p.head_dim,p.experts.dim);B(COMP_NORM,"attn.compressor.norm.weight",BF16,1,p.head_dim,0);
            if(v->compress_ratio>1){B(COMP_GATE,"attn.compressor.wgate.weight",BF16,2,p.head_dim,p.experts.dim);}
            B(INDEX_K,"attn.indexer.wk.weight",BF16,2,p.index_dim,p.head_dim);B(INDEX_K_NORM,"attn.indexer.k_norm.weight",BF16,1,p.index_dim,0);
        }
        if(v->index_source==l) {
            F(INDEX_Q,"attn.indexer.wq_b",index_rows,p.q_rank);
            B(INDEX_WEIGHTS,"attn.indexer.weights_proj.weight",BF16,2,p.index_heads,p.experts.dim);
        }
        if(v->engram_rows) {
            B(ENGRAM_EMBED,"engram.embed.weight",U8,2,v->engram_rows,p.engram_dim/2);B(ENGRAM_SCALE,"engram.embed.scale",F16,2,v->engram_rows,p.engram_dim/32);
            F(ENGRAM_WKV,"engram.wkv",engram_output,engram_width);
            B(ENGRAM_K,"engram.k_weight",BF16,2,p.streams,p.experts.dim);B(ENGRAM_Q,"engram.q_weight",BF16,2,p.streams,p.experts.dim);
        }
#undef B
#undef F
    }
    size_t count=bn_safetensors_index_count(index);unsigned char *seen=calloc(count,1);if(!seen)return -1;
    seen[p.embedding]=seen[p.output]=seen[p.output_norm]=1;
    for(int l=0;l<p.experts.layers;l++)for(int r=0;r<BN_NATIVE_ROLE_COUNT;r++)if(p.layers[l].tensor[r]>=0)seen[p.layers[l].tensor[r]]=1;
    int invalid=0;
    for(size_t i=0;i<count;i++) {
        const BnSafetensorsTensor *t=bn_safetensors_index_tensor(index,i);
        if(!t||(!strncmp(t->name,"layers.",7)&&!strstr(t->name,".ffn.experts.")&&!seen[i])){invalid=1;break;}
    }
    free(seen);if(invalid)return -1;
    *out=p;return 0;
}

static int bn_model_arch_is_gemma4(const char *arch) {
    return arch && strcmp(arch, "gemma4") == 0;
}

int bn_model_arch_activation(const char *arch) {
    const BnModelArchOps *ops = bn_model_arch_ops_for(arch);
    if (ops && ops->activation)
        return ops->activation(arch);
    return BN_MODEL_ACTIVATION_SILU;
}

int bn_model_arch_activation_is_silu(int act_type) {
    return act_type == BN_MODEL_ACTIVATION_SILU;
}

int bn_model_arch_activation_is_relu2(int act_type) {
    return act_type == BN_MODEL_ACTIVATION_RELU2;
}

int bn_model_arch_activation_is_gelu(int act_type) {
    return act_type == BN_MODEL_ACTIVATION_GELU;
}

int bn_model_arch_activation_uses_silu_path(int act_type) {
    return !bn_model_arch_activation_is_relu2(act_type) &&
           !bn_model_arch_activation_is_gelu(act_type);
}

int bn_model_arch_attention_value_shares_key(const char *arch) {
    const BnModelArchOps *ops = bn_model_arch_ops_for(arch);
    return ops && ops->attention_value_shares_key &&
           ops->attention_value_shares_key(arch);
}

const char *bn_model_arch_tensor_suffix(BnModelTensorRole role) {
    switch (role) {
        case BN_MODEL_TENSOR_ATTN_NORM:   return "attn_norm.weight";
        case BN_MODEL_TENSOR_ATTN_Q:      return "attn_q.weight";
        case BN_MODEL_TENSOR_ATTN_K:      return "attn_k.weight";
        case BN_MODEL_TENSOR_ATTN_V:      return "attn_v.weight";
        case BN_MODEL_TENSOR_ATTN_OUTPUT: return "attn_output.weight";
        case BN_MODEL_TENSOR_ATTN_Q_BIAS: return "attn_q.bias";
        case BN_MODEL_TENSOR_ATTN_K_BIAS: return "attn_k.bias";
        case BN_MODEL_TENSOR_ATTN_V_BIAS: return "attn_v.bias";
        case BN_MODEL_TENSOR_ATTN_Q_NORM: return "attn_q_norm.weight";
        case BN_MODEL_TENSOR_ATTN_K_NORM: return "attn_k_norm.weight";
        case BN_MODEL_TENSOR_ATTN_SUB_NORM: return "attn_sub_norm.weight";
        case BN_MODEL_TENSOR_ATTN_POST_NORM: return "post_attention_norm.weight";
        case BN_MODEL_TENSOR_SSM_QKV:     return "attn_qkv.weight";
        case BN_MODEL_TENSOR_SSM_GATE:    return "attn_gate.weight";
        case BN_MODEL_TENSOR_SSM_A:       return "ssm_a";
        case BN_MODEL_TENSOR_SSM_ALPHA:   return "ssm_alpha.weight";
        case BN_MODEL_TENSOR_SSM_BETA:    return "ssm_beta.weight";
        case BN_MODEL_TENSOR_SSM_CONV1D:  return "ssm_conv1d.weight";
        case BN_MODEL_TENSOR_SSM_DT_BIAS: return "ssm_dt.bias";
        case BN_MODEL_TENSOR_SSM_NORM:    return "ssm_norm.weight";
        case BN_MODEL_TENSOR_SSM_OUT:     return "ssm_out.weight";
        case BN_MODEL_TENSOR_FFN_NORM:    return "ffn_norm.weight";
        case BN_MODEL_TENSOR_FFN_POST_ATTN_NORM: return "post_attention_norm.weight";
        case BN_MODEL_TENSOR_FFN_SUB_NORM: return "ffn_sub_norm.weight";
        case BN_MODEL_TENSOR_FFN_POST_NORM: return "post_ffw_norm.weight";
        case BN_MODEL_TENSOR_FFN_GATE:    return "ffn_gate.weight";
        case BN_MODEL_TENSOR_FFN_UP:      return "ffn_up.weight";
        case BN_MODEL_TENSOR_FFN_DOWN:    return "ffn_down.weight";
        case BN_MODEL_TENSOR_MOE_ROUTER:  return "ffn_gate_inp.weight";
        case BN_MODEL_TENSOR_MOE_GATE_EXPS: return "ffn_gate_exps.weight";
        case BN_MODEL_TENSOR_MOE_UP_EXPS: return "ffn_up_exps.weight";
        case BN_MODEL_TENSOR_MOE_GATE_UP_EXPS: return "ffn_gate_up_exps.weight";
        case BN_MODEL_TENSOR_MOE_DOWN_EXPS: return "ffn_down_exps.weight";
        case BN_MODEL_TENSOR_SHARED_FFN_GATE: return "ffn_gate_shexp.weight";
        case BN_MODEL_TENSOR_SHARED_FFN_UP: return "ffn_up_shexp.weight";
        case BN_MODEL_TENSOR_SHARED_FFN_DOWN: return "ffn_down_shexp.weight";
        case BN_MODEL_TENSOR_SHARED_FFN_ROUTER: return "ffn_gate_inp_shexp.weight";
        case BN_MODEL_TENSOR_LAYER_OUTPUT_SCALE: return "layer_output_scale.weight";
        case BN_MODEL_TENSOR_HC_ATTN_NORM: return "hc_attn_norm.weight";
        case BN_MODEL_TENSOR_HC_ATTN_DOWN: return "hc_attn_down.weight";
        case BN_MODEL_TENSOR_HC_ATTN_UP: return "hc_attn_up.weight";
        case BN_MODEL_TENSOR_HC_ATTN_INJECT: return "hc_attn_inject.weight";
        case BN_MODEL_TENSOR_HC_FFN_NORM: return "hc_ffn_norm.weight";
        case BN_MODEL_TENSOR_HC_FFN_DOWN: return "hc_ffn_down.weight";
        case BN_MODEL_TENSOR_HC_FFN_UP: return "hc_ffn_up.weight";
        case BN_MODEL_TENSOR_HC_FFN_INJECT: return "hc_ffn_inject.weight";
        case BN_MODEL_TENSOR_INDEXER_Q: return "indexer.q_proj.weight";
        case BN_MODEL_TENSOR_INDEXER_K: return "indexer.k_proj.weight";
        case BN_MODEL_TENSOR_INDEXER_Q_NORM: return "indexer.q_norm.weight";
        case BN_MODEL_TENSOR_INDEXER_K_NORM: return "indexer.k_norm.weight";
        case BN_MODEL_TENSOR_PLE_KEY: return "ple_key.weight";
        case BN_MODEL_TENSOR_PLE_VALUE: return "ple_value.weight";
        case BN_MODEL_TENSOR_PLE_NORM_KEY: return "ple_norm_key.weight";
        case BN_MODEL_TENSOR_PLE_NORM_QUERY: return "ple_norm_query.weight";
        case BN_MODEL_TENSOR_PLE_NORM_CONV: return "ple_norm_conv.weight";
        case BN_MODEL_TENSOR_PLE_CONV1D: return "ple_conv1d.weight";
        default:                          return NULL;
    }
}

int bn_model_arch_default_tensor_name(char *out,
                                      size_t out_size,
                                      int layer,
                                      int role) {
    const char *suffix = bn_model_arch_tensor_suffix((BnModelTensorRole)role);
    if (!out || out_size == 0 || layer < 0 || !suffix) return -1;
    int n = snprintf(out, out_size, "blk.%d.%s", layer, suffix);
    return (n < 0 || (size_t)n >= out_size) ? -1 : 0;
}

static int bn_model_arch_qwen4exp_tensor_name(char *out, size_t out_size,
                                               int layer, int role) {
    const char *name = NULL;
    switch ((BnModelTensorRole)role) {
        case BN_MODEL_TENSOR_HC_OUTPUT_NORM: name = "output_hc_norm.weight"; break;
        case BN_MODEL_TENSOR_HC_OUTPUT_DOWN: name = "output_hc_down.weight"; break;
        case BN_MODEL_TENSOR_HC_OUTPUT_UP: name = "output_hc_up.weight"; break;
        default:
            return bn_model_arch_default_tensor_name(out, out_size, layer, role);
    }
    if (!out || out_size == 0) return -1;
    int n = snprintf(out, out_size, "%s", name);
    return (n < 0 || (size_t)n >= out_size) ? -1 : 0;
}

int bn_model_arch_tensor_name_for(const BnModelArchOps *ops,
                                  char *out,
                                  size_t out_size,
                                  int layer,
                                  BnModelTensorRole role) {
    if (ops && ops->tensor_name)
        return ops->tensor_name(out, out_size, layer, (int)role);
    return bn_model_arch_default_tensor_name(out, out_size, layer, (int)role);
}

int bn_model_arch_expert_component_name(const char *architecture,
    char *out, size_t out_size, int layer, int expert,
    BnModelTensorRole role, BnModelExpertComponent component) {
    if (!architecture || strcmp(architecture, "deepseek_v41") ||
        !out || !out_size || layer < 0 || expert < 0) return -1;
    const char *projection;
    switch (role) {
        case BN_MODEL_TENSOR_MOE_GATE_EXPS: projection = "w1"; break;
        case BN_MODEL_TENSOR_MOE_UP_EXPS: projection = "w3"; break;
        case BN_MODEL_TENSOR_MOE_DOWN_EXPS: projection = "w2"; break;
        default: return -1;
    }
    const char *suffix;
    switch (component) {
        case BN_MODEL_EXPERT_COMPONENT_PACKED: suffix = "trellis"; break;
        case BN_MODEL_EXPERT_COMPONENT_PRE: suffix = "suh"; break;
        case BN_MODEL_EXPERT_COMPONENT_POST: suffix = "svh"; break;
        case BN_MODEL_EXPERT_COMPONENT_MARKER: suffix = "mul1"; break;
        default: return -1;
    }
    char name[128];
    int n = snprintf(name, sizeof(name), "layers.%d.ffn.experts.%d.%s.%s",
                     layer, expert, projection, suffix);
    if (n < 0 || (size_t)n >= sizeof(name) || (size_t)n >= out_size)
        return -1;
    memcpy(out, name, (size_t)n + 1);
    return 0;
}

int bn_model_arch_tensor_scale_name_for(const BnModelArchOps *ops,
                                        char *out,
                                        size_t out_size,
                                        int layer,
                                        BnModelTensorRole role) {
    char weight_name[128];
    if (bn_model_arch_tensor_name_for(ops, weight_name, sizeof(weight_name),
                                      layer, role) != 0)
        return -1;
    const char *suffix = ".weight";
    size_t len = strlen(weight_name);
    size_t suffix_len = strlen(suffix);
    if (len < suffix_len ||
        strcmp(weight_name + len - suffix_len, suffix) != 0)
        return -1;
    if (len - suffix_len + strlen(".scale") + 1 > out_size)
        return -1;
    memcpy(out, weight_name, len - suffix_len);
    memcpy(out + len - suffix_len, ".scale", strlen(".scale") + 1);
    return 0;
}

void bn_model_arch_apply_config(BnConfig *c, const BnModelArchOps *ops) {
    if (c) {
        c->policy_flags = ops ? ops->policy_flags : 0;
        c->bf16_layout = ops ? ops->bf16_layout : BN_MODEL_BF16_LAYOUT_NONE;
        c->ssm_qk_norm = ops ? ops->ssm_qk_norm : BN_SSM_QK_NORM_CLAMPED_L2;
    }
}

/* Populate shared admission/source metadata without fabricating legacy GQA
 * attention or CPU weight bindings for heterogeneous planned topologies. */
int bn_model_arch_apply_anatomy_config(BnConfig *config,
    const BnModelAnatomy *a, int max_seq_len) {
    if (!config || !a || max_seq_len < 0 || a->dim < 1 ||
        a->n_layers < 1 || a->n_layers > BN_MODEL_ANATOMY_MAX_LAYERS ||
        a->vocab_size < 1 || a->context_length < 1) return -1;
    BnConfig next = *config;
    BnConfig *c = &next;
    c->dim = a->dim; c->hidden_dim = a->dense_hidden;
    c->n_layers = a->n_layers; c->nextn_predict_layers = a->nextn_layers;
    c->vocab_size = a->vocab_size; c->seq_len = a->context_length;
    if (max_seq_len > 0 && max_seq_len < c->seq_len) c->seq_len = max_seq_len;
    c->norm_eps = a->norm_eps;
    c->n_experts = a->experts; c->n_experts_active = a->active_experts;
    c->moe_intermediate_size = a->expert_hidden;
    c->moe_norm_topk_prob = a->normalize_expert_weights;
    c->moe_expert_weights_scale = a->expert_weights_scale;
    c->has_shared_expert = a->shared_experts > 0;
    c->shared_expert_intermediate_size = a->shared_hidden;
    *config = next;
    return 0;
}

int bn_model_arch_requires_float_kquant_fallback(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_REQUIRES_FLOAT_KQUANT_FALLBACK) != 0);
}

float bn_model_arch_attention_scale(const BnConfig *c, int head_size) {
    if (c && (c->policy_flags & BN_MODEL_ARCH_POLICY_UNIT_ATTENTION_SCALE))
        return 1.0f;
    return 1.0f / sqrtf((float)head_size);
}

BnModelArchRMSNormMode bn_model_arch_rmsnorm_mode(const BnConfig *c) {
    if (c && (c->policy_flags & BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER))
        return BN_MODEL_ARCH_RMSNORM_REFERENCE_ORDER;
    return BN_MODEL_ARCH_RMSNORM_BACKEND_ORDER;
}

int bn_model_arch_rmsnorm_uses_reference_order(
    const BnConfig *c) {
    return bn_model_arch_rmsnorm_mode(c) ==
           BN_MODEL_ARCH_RMSNORM_REFERENCE_ORDER;
}

float bn_model_arch_norm_epsilon(const BnConfig *c) {
    return c ? c->norm_eps : 0.0f;
}

float bn_model_arch_final_logit_softcap(const BnConfig *c) {
    return c ? c->final_logit_softcap : 0.0f;
}

int bn_model_arch_attention_value_shares_key_config(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY) != 0);
}

int bn_model_arch_attention_qk_norm_stride(const BnConfig *c,
                                           int head_size) {
    return c && c->qk_norm_per_head ? head_size : 0;
}

int bn_model_arch_attention_uses_per_head_qk_norm(const BnConfig *c) {
    return c ? c->qk_norm_per_head : 0;
}

int bn_model_arch_uses_per_layer_embedding(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT) != 0);
}

int bn_model_arch_uses_attention_post_norm(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_ATTENTION_POST_NORM) != 0);
}

int bn_model_arch_uses_ffn_post_norm(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_FFN_POST_NORM) != 0);
}

int bn_model_arch_uses_layer_output_scale(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_LAYER_OUTPUT_SCALE) != 0);
}

int bn_model_arch_has_ffn_gate(const BnConfig *c) {
    return c ? c->has_ffn_gate : 0;
}

int bn_model_arch_config_activation(const BnConfig *c) {
    return c ? c->act_type : BN_MODEL_ACTIVATION_SILU;
}

int bn_model_arch_uses_hybrid_layer_layout(const BnConfig *c) {
    return c && c->full_attn_interval > 0;
}

int bn_model_arch_uses_hybrid_ssm(const BnConfig *c) {
    return bn_model_arch_uses_hybrid_layer_layout(c) && c->ssm_inner_size > 0;
}

int bn_model_arch_uses_dense_attention_only(const BnConfig *c) {
    return c &&
           c->n_experts <= 0 &&
           !bn_model_arch_uses_hybrid_layer_layout(c);
}

int bn_model_arch_uses_large_dense_shape(const BnConfig *c) {
    return c &&
           c->n_experts <= 0 &&
           c->dim >= 4096;
}

int bn_model_arch_uses_large_dense_hybrid_ssm(const BnConfig *c) {
    return bn_model_arch_uses_hybrid_ssm(c) &&
           bn_model_arch_uses_large_dense_shape(c);
}

static int model_arch_gemma4_divides_rope_freqs(const BnConfig *c, int layer) {
    if (!c || ((c->policy_flags & BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT) == 0))
        return 0;
    if (c->per_layer_input_dim > 0)
        return 1;
    if (layer >= 0 &&
        layer < c->n_layers &&
        layer < (int)(sizeof(c->sliding_window_pattern) /
                      sizeof(c->sliding_window_pattern[0])))
        return !c->sliding_window_pattern[layer];
    return 0;
}

int bn_model_arch_divides_rope_freqs(const BnConfig *c, int layer) {
    return model_arch_gemma4_divides_rope_freqs(c, layer);
}

int bn_model_arch_uses_swa_rope(const BnConfig *c, int layer_head_size) {
    return c && c->rope_theta_swa > 0.0f &&
           layer_head_size < c->head_size;
}

int bn_model_arch_rope_dims_for_head(const BnConfig *c,
                                     int layer_head_size) {
    if (!c || layer_head_size <= 0)
        return 0;
    int base_rope_dims = c->rope_text_dims > 0
        ? c->rope_text_dims
        : (c->rope_dim_count > 0 ? c->rope_dim_count : layer_head_size);
    int rope_dims = bn_model_arch_uses_swa_rope(c, layer_head_size) &&
                    c->rope_dim_count_swa > 0
        ? c->rope_dim_count_swa
        : base_rope_dims;
    if (rope_dims > layer_head_size)
        rope_dims = layer_head_size;
    return rope_dims > 0 ? rope_dims : 0;
}

float bn_model_arch_rope_theta_for_head(const BnConfig *c,
                                        int layer_head_size) {
    if (!c)
        return 0.0f;
    return bn_model_arch_uses_swa_rope(c, layer_head_size)
        ? c->rope_theta_swa : c->rope_theta;
}

float bn_model_arch_rope_base_theta(const BnConfig *c) {
    return c ? c->rope_theta : 0.0f;
}

int bn_model_arch_rope_uses_base_frequency(const BnConfig *c,
                                           int layer_head_size) {
    return !bn_model_arch_uses_swa_rope(c, layer_head_size);
}

void bn_model_arch_init_rope_frequencies(const BnConfig *c,
                                         float *freqs,
                                         int capacity_pairs) {
    if (!c || !freqs || capacity_pairs <= 0)
        return;
    int rope_dims = c->rope_dim_count > 0
        ? c->rope_dim_count : c->head_size;
    int half_rope = rope_dims / 2;
    if (half_rope > capacity_pairs)
        half_rope = capacity_pairs;
    bn_model_arch_init_rope_frequencies_for_theta(
        c->rope_theta, rope_dims, freqs, half_rope);
    if (c->rope_text_dims > 0) {
        int text_pairs = c->rope_text_dims / 2;
        if (text_pairs < 0)
            text_pairs = 0;
        for (int i = text_pairs; i < half_rope; i++)
            freqs[i] = 0.0f;
    }
}

void bn_model_arch_init_rope_frequencies_for_theta(float theta,
                                                   int rope_dims,
                                                   float *freqs,
                                                   int capacity_pairs) {
    if (!freqs || rope_dims <= 0 || capacity_pairs <= 0)
        return;
    int half_rope = rope_dims / 2;
    if (half_rope > capacity_pairs)
        half_rope = capacity_pairs;
    float freq_scale = powf(theta, -2.0f / (float)rope_dims);
    float freq = 1.0f;
    for (int i = 0; i < half_rope; i++) {
        freqs[i] = freq;
        freq *= freq_scale;
    }
}

void bn_model_arch_init_rope_angles_for_theta(float theta,
                                              int rope_dims,
                                              int position,
                                              float *angles,
                                              int capacity_pairs) {
    if (!angles || rope_dims <= 0 || capacity_pairs <= 0)
        return;
    int half_rope = rope_dims / 2;
    if (half_rope > capacity_pairs)
        half_rope = capacity_pairs;
    float theta_scale = powf(theta, -2.0f / (float)rope_dims);
    float angle = (float)position;
    for (int i = 0; i < half_rope; i++) {
        angles[i] = angle;
        angle *= theta_scale;
    }
}

int bn_model_arch_per_layer_embedding_dim(const BnConfig *c) {
    if (!bn_model_arch_uses_per_layer_embedding(c) ||
        c->per_layer_input_dim <= 0)
        return 0;
    return c->per_layer_input_dim;
}

int bn_model_arch_allows_small_dense_prefill_decode_fallback(
    const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK) != 0) &&
           bn_model_arch_uses_dense_attention_only(c) &&
           c->dim <= 2560;
}

int bn_model_arch_prefill_uses_decode_for_parity(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY) != 0);
}

int bn_model_arch_moe_requires_float_kquant_gateup_fallback(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP_FALLBACK) != 0);
}

int bn_model_arch_moe_requires_reference_attention(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ATTENTION) != 0);
}

int bn_model_arch_requires_reference_attention(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION) != 0);
}

int bn_model_arch_reference_attention_from_layer(const BnConfig *c) {
    if (!c || c->n_experts > 0 || c->n_layers <= 0 ||
        !(c->policy_flags &
          BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION_LAST_THIRD))
        return -1;
    return (2 * c->n_layers) / 3;
}

int bn_model_arch_requires_reference_recurrent(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT) != 0);
}

int bn_model_arch_uses_hyper_connections(const BnConfig *c) {
    return c && (c->policy_flags & BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS) != 0;
}

int bn_model_arch_uses_query_sparse_attention(const BnConfig *c) {
    return c &&
           (c->policy_flags & BN_MODEL_ARCH_POLICY_QUERY_SPARSE_ATTENTION) != 0;
}

int bn_model_arch_ssm_uses_sigmoid_gate(const BnConfig *c) {
    return c && (c->policy_flags & BN_MODEL_ARCH_POLICY_SSM_SIGMOID_GATE) != 0;
}

int bn_model_arch_uses_positional_layer_embedding(const BnConfig *c) {
    return c && (c->policy_flags &
                 BN_MODEL_ARCH_POLICY_POSITIONAL_LAYER_EMBEDDING) != 0;
}

int bn_model_arch_moe_uses_scaled_router_input(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT) != 0);
}

int bn_model_arch_moe_uses_reference_router_accumulation(
    const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION) != 0);
}

int bn_model_arch_attention_uses_padded_weighted_v_reduction(
    const BnConfig *c) {
    return c && c->n_experts > 0 &&
           (c->policy_flags &
            BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION) != 0 &&
           (c->policy_flags &
            BN_MODEL_ARCH_POLICY_MOE_DENSE_RESIDUAL_BRANCH) == 0;
}

int bn_model_arch_moe_uses_separate_router_topk(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_MOE_SEPARATE_ROUTER_TOPK) != 0);
}

int bn_model_arch_moe_uses_dense_residual_branch(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_DENSE_RESIDUAL_BRANCH) != 0);
}

int bn_model_arch_uses_moe(const BnConfig *c) {
    return c && c->n_experts > 0;
}

static int bn_model_arch_gguf_key(BnGGUFFile *f,
                                  const char *suffix,
                                  char *key,
                                  size_t key_size) {
    if (!f || !suffix || !key || key_size == 0) return -1;
    const char *arch = bn_gguf_get_str(f, "general.architecture");
    const BnModelArchOps *ops = bn_model_arch_ops_for(arch);
    const char *prefix = ops && ops->prefix
        ? ops->prefix(arch)
        : bn_model_arch_prefix(arch);
    int n = snprintf(key, key_size, "%s.%s", prefix, suffix);
    return (n < 0 || (size_t)n >= key_size) ? -1 : 0;
}

int bn_model_arch_gguf_u32(BnGGUFFile *f, const char *suffix) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0) return 0;
    return (int)bn_gguf_get_u32(f, key);
}

int bn_model_arch_gguf_u32_or_i32_array(BnGGUFFile *f,
                                        const char *suffix,
                                        int idx) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0) return 0;
    int ki = bn_gguf_find_key(f, key);
    if (ki < 0) return 0;
    BnGGUFKeyValue *kv = &f->kvs[ki];
    if (kv->type == BN_GGUF_TYPE_UINT32) return (int)kv->value.u32;
    if (kv->type == BN_GGUF_TYPE_INT32) return kv->value.i32;
    if (kv->type != BN_GGUF_TYPE_ARRAY || idx < 0) return 0;
    BnGGUFArray *a = &kv->value.arr;
    if ((uint64_t)idx >= a->n || !a->data) return 0;
    if (a->elem_type == BN_GGUF_TYPE_INT32) {
        int32_t v;
        memcpy(&v, (const uint8_t *)a->data + (size_t)idx * sizeof(v),
               sizeof(v));
        return v;
    }
    if (a->elem_type == BN_GGUF_TYPE_UINT32) {
        uint32_t v;
        memcpy(&v, (const uint8_t *)a->data + (size_t)idx * sizeof(v),
               sizeof(v));
        return (int)v;
    }
    return 0;
}

float bn_model_arch_gguf_f32(BnGGUFFile *f, const char *suffix) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0) return 0.0f;
    return bn_gguf_get_f32(f, key);
}

uint64_t bn_model_arch_gguf_arr_n(BnGGUFFile *f, const char *suffix) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0) return 0;
    return bn_gguf_get_arr_n(f, key);
}

const void *bn_model_arch_gguf_arr_data(BnGGUFFile *f, const char *suffix) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0)
        return NULL;
    return bn_gguf_get_arr_data(f, key);
}

int bn_model_arch_gguf_bool_array(BnGGUFFile *f,
                                  const char *suffix,
                                  int idx) {
    char key[128];
    if (bn_model_arch_gguf_key(f, suffix, key, sizeof(key)) != 0 ||
        idx < 0)
        return 0;
    int ki = bn_gguf_find_key(f, key);
    if (ki < 0) return 0;
    BnGGUFKeyValue *kv = &f->kvs[ki];
    if (kv->type == BN_GGUF_TYPE_BOOL) return kv->value.b ? 1 : 0;
    if (kv->type != BN_GGUF_TYPE_ARRAY) return 0;
    BnGGUFArray *a = &kv->value.arr;
    if ((uint64_t)idx >= a->n || !a->data) return 0;
    if (a->elem_type == BN_GGUF_TYPE_BOOL)
        return ((const uint8_t *)a->data)[idx] ? 1 : 0;
    if (a->elem_type == BN_GGUF_TYPE_INT32) {
        int32_t v;
        memcpy(&v, (const uint8_t *)a->data + (size_t)idx * sizeof(v),
               sizeof(v));
        return v != 0;
    }
    if (a->elem_type == BN_GGUF_TYPE_UINT32) {
        uint32_t v;
        memcpy(&v, (const uint8_t *)a->data + (size_t)idx * sizeof(v),
               sizeof(v));
        return v != 0;
    }
    return 0;
}

int bn_model_arch_gguf_uses_moe(BnGGUFFile *f) {
    return bn_model_arch_gguf_u32(f, "expert_count") > 0;
}

int bn_model_arch_uses_non_hybrid_moe(const BnConfig *c) {
    return bn_model_arch_uses_moe(c) && c->full_attn_interval <= 0;
}

int bn_model_arch_uses_hybrid_moe(const BnConfig *c) {
    return bn_model_arch_uses_moe(c) && c->full_attn_interval > 0;
}

int bn_model_arch_uses_two_expert_all_active_moe(const BnConfig *c) {
    return c &&
           c->n_experts == 2 &&
           c->n_experts_active == 2;
}

int bn_model_arch_uses_more_than_two_expert_moe(const BnConfig *c) {
    return c && c->n_experts > 2;
}

int bn_model_arch_moe_total_experts(const BnConfig *c) {
    return c ? c->n_experts : 0;
}

int bn_model_arch_moe_active_experts(const BnConfig *c) {
    return c ? c->n_experts_active : 0;
}

int bn_model_arch_moe_expert_hidden_dim(const BnConfig *c) {
    return c ? c->moe_intermediate_size : 0;
}

int bn_model_arch_moe_route_shape_valid(const BnConfig *c) {
    return bn_model_arch_moe_active_experts(c) > 0 &&
           bn_model_arch_moe_expert_hidden_dim(c) > 0;
}

int bn_model_arch_moe_normalizes_topk_route_weights(const BnConfig *c) {
    return c ? c->moe_norm_topk_prob : 0;
}

float bn_model_arch_moe_expert_weights_scale(const BnConfig *c) {
    return c ? c->moe_expert_weights_scale : 0.0f;
}

int bn_model_arch_moe_uses_reference_silu(const BnConfig *c) {
    return c ? c->moe_uses_reference_silu : -1;
}

int bn_model_arch_config_has_shared_expert(const BnConfig *c) {
    return c ? c->has_shared_expert : 0;
}

int bn_model_arch_shared_expert_hidden_dim(const BnConfig *c) {
    if (!bn_model_arch_config_has_shared_expert(c) ||
        c->shared_expert_intermediate_size <= 0)
        return 0;
    return c->shared_expert_intermediate_size;
}

int bn_model_arch_moe_prefill_uses_matvec_router(const BnConfig *c) {
    return c && (c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_PREFILL_MATVEC_ROUTER) != 0;
}

int bn_model_arch_moe_prefill_requires_matvec(const BnConfig *c) {
    return bn_model_arch_uses_two_expert_all_active_moe(c) &&
           c->has_shared_expert;
}

int bn_model_arch_uses_all_active_two_expert_moe(const BnConfig *c,
                                                 int dim) {
    return bn_model_arch_uses_two_expert_all_active_moe(c) &&
           c->moe_intermediate_size >= 4096 &&
           dim <= 2048;
}

int bn_model_arch_attention_window(const BnConfig *c, int layer) {
    if (!c || c->sliding_window <= 0 || layer < 0 || layer >= c->n_layers ||
        layer >= (int)(sizeof(c->sliding_window_pattern) /
                       sizeof(c->sliding_window_pattern[0])))
        return 0;
    return c->sliding_window_pattern[layer] ? c->sliding_window : 0;
}

int bn_model_arch_loads_extra_metadata(const BnConfig *c) {
    return bn_model_arch_uses_per_layer_embedding(c);
}

int bn_model_arch_loads_per_layer_input_weights(const BnConfig *c) {
    return bn_model_arch_uses_per_layer_embedding(c) &&
           c->per_layer_input_dim > 0;
}

int bn_model_arch_layer_reuses_kv(const BnConfig *c, int layer) {
    return bn_model_arch_attention_value_shares_key_config(c) &&
           c->kv_unique_layer_count > 0 &&
           layer >= c->kv_unique_layer_count;
}

int bn_model_arch_kv_reuse_layer(const BnConfig *c, int layer) {
    if (!bn_model_arch_layer_reuses_kv(c, layer))
        return -1;
    int is_swa = (layer >= 0 &&
                  layer < (int)(sizeof(c->sliding_window_pattern) /
                                sizeof(c->sliding_window_pattern[0])))
        ? c->sliding_window_pattern[layer]
        : 0;
    int reuse_layer = c->kv_unique_layer_count - (is_swa ? 2 : 1);
    return reuse_layer < 0 ? 0 : reuse_layer;
}

int bn_model_arch_loads_extra_ffn_post_norms(const BnConfig *c) {
    return bn_model_arch_uses_ffn_post_norm(c);
}

int bn_model_arch_loads_moe_aux_weights(const BnConfig *c) {
    return bn_model_arch_moe_uses_scaled_router_input(c);
}

int bn_model_arch_separates_rope_norm(const BnConfig *c) {
    return c && (c->policy_flags & BN_MODEL_ARCH_POLICY_SEPARATE_ROPE_NORM) != 0;
}

int bn_model_arch_config_uses_full_rope_text_dims(const BnConfig *c) {
    return c &&
           ((c->policy_flags & BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS) != 0);
}

int bn_model_arch_tokenizer_uses_metaspace(const char *tokenizer_model) {
    return bn_model_arch_is_gemma4(tokenizer_model);
}

BnTokenizerPre bn_model_tokenizer_pretokenizer(const char *tokenizer_pre) {
    if (tokenizer_pre && strcmp(tokenizer_pre, "qwen2") == 0)
        return BN_TOKENIZER_PRE_LETTERS;
    if (tokenizer_pre && strcmp(tokenizer_pre, "qwen35") == 0)
        return BN_TOKENIZER_PRE_LETTERS_MARKS;
    if (tokenizer_pre && strcmp(tokenizer_pre, "gemma4") == 0)
        return BN_TOKENIZER_PRE_NEWLINES;
    if (tokenizer_pre && strcmp(tokenizer_pre, "glm4") == 0)
        return BN_TOKENIZER_PRE_GLM4;
    return BN_TOKENIZER_PRE_NONE;
}

const char *bn_model_tokenizer_extra_eog(const char *tokenizer_pre, int index) {
    static const char *const glm_eog[] = {"<|user|>", "<|observation|>"};
    if (!tokenizer_pre || strcmp(tokenizer_pre, "glm4") != 0 || index < 0 || index >= 2)
        return NULL;
    return glm_eog[index];
}

int bn_model_tokenizer_normalizes_nfc(const char *tokenizer_pre) {
    return tokenizer_pre && strcmp(tokenizer_pre, "qwen35") == 0;
}

int bn_model_tokenizer_uses_metaspace(const char *tokenizer_model) {
    return bn_model_arch_tokenizer_uses_metaspace(tokenizer_model);
}

int bn_model_tokenizer_default_add_bos(const char *tokenizer_pre,
                                       int has_bos_token) {
    if (tokenizer_pre && strcmp(tokenizer_pre, "qwen35") == 0)
        return 0;
    return has_bos_token != 0;
}

int bn_model_arch_uses_small_dense_shape(const BnConfig *c) {
    return bn_model_arch_uses_dense_attention_only(c) &&
           c->dim <= 2560;
}

int bn_model_arch_dense_batch_prefill_shape_allowed(
    const BnConfig *c,
    int supports_large_dense_batch_prefill) {
    if (!bn_model_arch_uses_dense_attention_only(c))
        return 0;
    return c->dim <= (supports_large_dense_batch_prefill ? 8192 : 2560);
}

int bn_model_arch_dense_logits_argmax_shape_allowed(const BnConfig *c,
                                                    int logits_rows) {
    return bn_model_arch_uses_dense_attention_only(c) &&
           logits_rows > 262144;
}

int bn_model_arch_moe_logits_mmvq_argmax_shape_allowed(const BnConfig *c,
                                                       int logits_cols) {
    return bn_model_arch_uses_moe(c) &&
           logits_cols == 1536;
}

int bn_model_arch_prefill_uses_reference_activation(const BnConfig *c) {
    return c && ((c->policy_flags &
                  BN_MODEL_ARCH_POLICY_PREFILL_REFERENCE_ACTIVATION) != 0);
}

int bn_model_arch_ffn_uses_reference_activation(const BnConfig *c) {
    return c && ((c->policy_flags & BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION) != 0);
}

int bn_model_arch_rope_text_dims(int rope_dim_count,
                                 const int32_t *sections,
                                 uint64_t n_sections) {
    if (!sections || n_sections == 0 || rope_dim_count <= 0 || sections[0] <= 0)
        return 0;
    return sections[0] * 2;
}

int bn_model_arch_is_ssm_layer(const BnConfig *c, int layer) {
    return bn_model_arch_uses_hybrid_layer_layout(c) &&
           ((layer + 1) % c->full_attn_interval != 0);
}

int bn_model_arch_is_attention_layer(const BnConfig *c, int layer) {
    return c && !bn_model_arch_is_ssm_layer(c, layer);
}

int bn_model_arch_attention_layer_index(const BnConfig *c, int layer) {
    if (!c) return -1;
    return bn_model_arch_uses_hybrid_layer_layout(c)
        ? (layer + 1) / c->full_attn_interval - 1
        : layer;
}

int bn_model_arch_ssm_layer_index(const BnConfig *c, int layer) {
    if (!bn_model_arch_uses_hybrid_layer_layout(c)) return -1;
    return layer - (layer + 1) / c->full_attn_interval;
}

int bn_model_arch_attention_layer_count(const BnConfig *c) {
    if (!c || c->n_layers <= 0) return 0;
    return bn_model_arch_uses_hybrid_layer_layout(c)
        ? c->n_layers / c->full_attn_interval
        : c->n_layers;
}

int bn_model_arch_ssm_layer_count(const BnConfig *c) {
    if (!c || c->n_layers <= 0) return 0;
    int n_attn_layers = bn_model_arch_attention_layer_count(c);
    int n_ssm_layers = c->n_layers - n_attn_layers;
    return n_ssm_layers > 0 ? n_ssm_layers : 0;
}

int bn_model_arch_infer_moe_hidden(BnGGUFFile *f,
                                   const BnModelArchOps *ops) {
    char name[128];
    if (bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0,
                                      BN_MODEL_TENSOR_MOE_GATE_EXPS) != 0)
        return 0;
    int ti = bn_gguf_find_tensor(f, name);
    if (ti >= 0 && f->tensors[ti].n_dims >= 3)
        return (int)f->tensors[ti].dims[1];
    if (bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0,
                                      BN_MODEL_TENSOR_MOE_GATE_UP_EXPS) != 0)
        return 0;
    ti = bn_gguf_find_tensor(f, name);
    if (ti >= 0 && f->tensors[ti].n_dims >= 3)
        return (int)(f->tensors[ti].dims[1] / 2);
    return 0;
}

int bn_model_arch_has_shared_expert(BnGGUFFile *f,
                                    const BnModelArchOps *ops) {
    char name[128];
    if (bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0,
                                      BN_MODEL_TENSOR_SHARED_FFN_GATE) != 0)
        return 0;
    return bn_gguf_find_tensor(f, name) >= 0;
}

int bn_model_arch_infer_shared_expert_hidden(BnGGUFFile *f,
                                             const BnModelArchOps *ops) {
    char name[128];
    if (bn_model_arch_tensor_name_for(ops, name, sizeof(name), 0,
                                      BN_MODEL_TENSOR_SHARED_FFN_GATE) != 0)
        return 0;
    int ti = bn_gguf_find_tensor(f, name);
    if (ti >= 0 && f->tensors[ti].n_dims >= 2)
        return (int)f->tensors[ti].dims[1];
    return 0;
}

void bn_model_arch_load_moe_config(BnConfig *c,
                                   BnGGUFFile *f,
                                   const BnModelArchOps *ops,
                                   const char *prefix) {
    char key[128];
    snprintf(key, sizeof(key), "%s.expert_count", prefix);
    c->n_experts = (int)bn_gguf_get_u32(f, key);
    snprintf(key, sizeof(key), "%s.expert_used_count", prefix);
    c->n_experts_active = (int)bn_gguf_get_u32(f, key);

    if (c->n_experts <= 0) return;
    if (ops)
        c->policy_flags |= ops->moe_policy_flags;
    c->moe_norm_topk_prob =
        (c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_UNNORMALIZED_TOPK) == 0;
    c->moe_uses_reference_silu =
        (c->policy_flags & BN_MODEL_ARCH_POLICY_MOE_REFERENCE_SILU) != 0;
    snprintf(key, sizeof(key), "%s.expert_weights_scale", prefix);
    c->moe_expert_weights_scale = bn_gguf_get_f32(f, key);

    snprintf(key, sizeof(key), "%s.expert_feed_forward_length", prefix);
    c->moe_intermediate_size = (int)bn_gguf_get_u32(f, key);
    if (c->moe_intermediate_size == 0)
        c->moe_intermediate_size = bn_model_arch_infer_moe_hidden(f, ops);

    c->has_shared_expert = bn_model_arch_has_shared_expert(f, ops);
    if (c->has_shared_expert) {
        snprintf(key, sizeof(key), "%s.expert_shared_feed_forward_length", prefix);
        c->shared_expert_intermediate_size = (int)bn_gguf_get_u32(f, key);
        if (c->shared_expert_intermediate_size == 0)
            c->shared_expert_intermediate_size =
                bn_model_arch_infer_shared_expert_hidden(f, ops);
    }
}

static void bn_model_arch_apply_gemma4_shapes(BnConfig *c,
                                              int max_head_size,
                                              int max_kv_dim) {
    if (!c) return;
    c->head_size = max_head_size;
    c->kv_dim = max_kv_dim;
    c->kv_mul = c->n_heads / c->n_kv_heads;
}

static int bn_model_arch_gemma4_tensor_name(char *out, size_t out_size,
                                            int layer, int role) {
    const char *suffix = NULL;
    switch ((BnModelTensorRole)role) {
        case BN_MODEL_TENSOR_FFN_SUB_NORM:
            suffix = "pre_ffw_norm_2.weight";
            break;
        default:
            return bn_model_arch_default_tensor_name(out, out_size, layer, role);
    }
    if (!out || out_size == 0 || layer < 0) return -1;
    int n = snprintf(out, out_size, "blk.%d.%s", layer, suffix);
    return (n < 0 || (size_t)n >= out_size) ? -1 : 0;
}

static void bn_model_arch_apply_default_shapes(BnConfig *c,
                                               int max_head_size,
                                               int max_kv_dim) {
    (void)c;
    (void)max_head_size;
    (void)max_kv_dim;
}

static int bn_model_arch_default_activation(const char *arch) {
    (void)arch;
    return BN_MODEL_ACTIVATION_SILU;
}

static int bn_model_arch_bitnet_activation(const char *arch) {
    (void)arch;
    return BN_MODEL_ACTIVATION_RELU2;
}

static int bn_model_arch_gemma4_activation(const char *arch) {
    (void)arch;
    return BN_MODEL_ACTIVATION_GELU;
}

static int bn_model_arch_attention_value_unique(const char *arch) {
    (void)arch;
    return 0;
}

static int bn_model_arch_attention_value_shared(const char *arch) {
    (void)arch;
    return 1;
}

static int bn_model_arch_match_gemma4(const char *arch) {
    return bn_model_arch_is_gemma4(arch);
}

static int bn_model_arch_match_qwen(const char *arch) {
    return arch && strncmp(arch, "qwen", 4) == 0;
}

static int bn_model_arch_match_qwen2(const char *arch) {
    return arch && strncmp(arch, "qwen2", 5) == 0;
}

static int bn_model_arch_match_qwen3(const char *arch) {
    return arch && (strcmp(arch, "qwen3") == 0 ||
                    strcmp(arch, "qwen3moe") == 0);
}

static int bn_model_arch_match_qwen35(const char *arch) {
    return arch && (strcmp(arch, "qwen35") == 0 ||
                    strcmp(arch, "qwen35moe") == 0);
}

static int bn_model_arch_match_qwen4exp(const char *arch) {
    return arch && strcmp(arch, "qwen4exp") == 0;
}

static int bn_model_arch_match_bitnet(const char *arch) {
    return arch && strncmp(arch, "bitnet", 6) == 0;
}

static int bn_model_arch_match_default(const char *arch) {
    (void)arch;
    return 1;
}

static int glm5next_match(const char *arch) {
    return arch && strcmp(arch, "glm5next") == 0;
}

static int glm5next_tensor_name(char *out, size_t capacity, int layer, int role) {
    const char *suffix = NULL;
    switch ((BnModelTensorRole)role) {
        case BN_MODEL_TENSOR_TOKEN_EMBD: suffix = "token_embd.weight"; break;
        case BN_MODEL_TENSOR_OUTPUT: suffix = "output.weight"; break;
        case BN_MODEL_TENSOR_OUTPUT_NORM: suffix = "output_norm.weight"; break;
        case BN_MODEL_TENSOR_SSM_OUT: suffix = "attn_output.weight"; break;
        case BN_MODEL_TENSOR_SSM_CONV1D_Q: suffix = "ssm_conv1d_q.weight"; break;
        case BN_MODEL_TENSOR_SSM_CONV1D_K: suffix = "ssm_conv1d_k.weight"; break;
        case BN_MODEL_TENSOR_SSM_CONV1D_V: suffix = "ssm_conv1d_v.weight"; break;
        case BN_MODEL_TENSOR_SSM_F_A: suffix = "ssm_f_a.weight"; break;
        case BN_MODEL_TENSOR_SSM_F_B: suffix = "ssm_f_b.weight"; break;
        case BN_MODEL_TENSOR_SSM_G_A: suffix = "ssm_g_a.weight"; break;
        case BN_MODEL_TENSOR_SSM_G_B: suffix = "ssm_g_b.weight"; break;
        case BN_MODEL_TENSOR_MHC_ATTN_FN: suffix = "hc_attn_fn.weight"; break;
        case BN_MODEL_TENSOR_MHC_ATTN_BASE: suffix = "hc_attn_base.weight"; break;
        case BN_MODEL_TENSOR_MHC_ATTN_SCALE: suffix = "hc_attn_scale.weight"; break;
        case BN_MODEL_TENSOR_MHC_FFN_FN: suffix = "hc_ffn_fn.weight"; break;
        case BN_MODEL_TENSOR_MHC_FFN_BASE: suffix = "hc_ffn_base.weight"; break;
        case BN_MODEL_TENSOR_MHC_FFN_SCALE: suffix = "hc_ffn_scale.weight"; break;
        case BN_MODEL_TENSOR_MLA_Q_A: suffix = "attn_q_a.weight"; break;
        case BN_MODEL_TENSOR_MLA_Q_A_NORM: suffix = "attn_q_a_norm.weight"; break;
        case BN_MODEL_TENSOR_MLA_Q_B: suffix = "attn_q_b.weight"; break;
        case BN_MODEL_TENSOR_MLA_KV_A: suffix = "attn_kv_a_mqa.weight"; break;
        case BN_MODEL_TENSOR_MLA_KV_A_NORM: suffix = "attn_kv_a_norm.weight"; break;
        case BN_MODEL_TENSOR_MLA_K_B: suffix = "attn_k_b.weight"; break;
        case BN_MODEL_TENSOR_MLA_V_B: suffix = "attn_v_b.weight"; break;
        case BN_MODEL_TENSOR_INDEXER_Q: suffix = "indexer.attn_q_b.weight"; break;
        case BN_MODEL_TENSOR_INDEXER_K: suffix = "indexer.attn_k.weight"; break;
        case BN_MODEL_TENSOR_INDEXER_K_NORM: suffix = "indexer.k_norm.weight"; break;
        case BN_MODEL_TENSOR_INDEXER_K_BIAS: suffix = "indexer.k_norm.bias"; break;
        case BN_MODEL_TENSOR_INDEXER_HEAD_WEIGHTS: suffix = "indexer.proj.weight"; break;
        case BN_MODEL_TENSOR_INDEXER_POOL_APE: suffix = "indexer_compressor_ape.weight"; break;
        case BN_MODEL_TENSOR_INDEXER_POOL_GATE: suffix = "indexer_compressor_gate.weight"; break;
        case BN_MODEL_TENSOR_MOE_ROUTER_BIAS: suffix = "exp_probs_b.bias"; break;
        case BN_MODEL_TENSOR_ATTN_NORM: case BN_MODEL_TENSOR_ATTN_Q:
        case BN_MODEL_TENSOR_ATTN_K: case BN_MODEL_TENSOR_ATTN_V:
        case BN_MODEL_TENSOR_ATTN_OUTPUT: case BN_MODEL_TENSOR_SSM_A:
        case BN_MODEL_TENSOR_SSM_BETA: case BN_MODEL_TENSOR_SSM_DT_BIAS:
        case BN_MODEL_TENSOR_SSM_NORM: case BN_MODEL_TENSOR_FFN_NORM:
        case BN_MODEL_TENSOR_FFN_GATE: case BN_MODEL_TENSOR_FFN_UP:
        case BN_MODEL_TENSOR_FFN_DOWN: case BN_MODEL_TENSOR_MOE_ROUTER:
        case BN_MODEL_TENSOR_MOE_GATE_EXPS: case BN_MODEL_TENSOR_MOE_UP_EXPS:
        case BN_MODEL_TENSOR_MOE_DOWN_EXPS: case BN_MODEL_TENSOR_SHARED_FFN_GATE:
        case BN_MODEL_TENSOR_SHARED_FFN_UP: case BN_MODEL_TENSOR_SHARED_FFN_DOWN:
            return bn_model_arch_default_tensor_name(out, capacity, layer, role);
        default: return -1; /* In particular, legacy GDN and HC roles are absent. */
    }
    if (!out || !capacity) return -1;
    int global = role == BN_MODEL_TENSOR_TOKEN_EMBD || role == BN_MODEL_TENSOR_OUTPUT ||
                 role == BN_MODEL_TENSOR_OUTPUT_NORM;
    if (!global && layer < 0) return -1;
    int n = global ? snprintf(out, capacity, "%s", suffix) :
                     snprintf(out, capacity, "blk.%d.%s", layer, suffix);
    return n < 0 || (size_t)n >= capacity ? -1 : 0;
}

static const BnGGUFKeyValue *glm5next_key(BnGGUFFile *f, const char *suffix) {
    char key[128];
    int n = snprintf(key, sizeof(key), "glm5next.%s", suffix);
    if (n < 0 || (size_t)n >= sizeof(key)) return NULL;
    const BnGGUFKeyValue *found = NULL;
    for (uint64_t i = 0; i < f->n_kv; i++)
        if (f->kvs[i].key && strcmp(f->kvs[i].key, key) == 0) {
            if (found) return NULL;
            found = &f->kvs[i];
        }
    return found;
}

static int glm5next_int(BnGGUFFile *f, const char *suffix) {
    const BnGGUFKeyValue *kv = glm5next_key(f, suffix);
    if (!kv) return -1;
    if (kv->type == BN_GGUF_TYPE_UINT32 && kv->value.u32 <= INT_MAX) return (int)kv->value.u32;
    if (kv->type == BN_GGUF_TYPE_INT32 && kv->value.i32 >= 0) return kv->value.i32;
    return -1;
}

static float glm5next_float(BnGGUFFile *f, const char *suffix) {
    const BnGGUFKeyValue *kv = glm5next_key(f, suffix);
    return kv && kv->type == BN_GGUF_TYPE_FLOAT32 ? kv->value.f32 : NAN;
}

static const BnGGUFArray *glm5next_array(BnGGUFFile *f, const char *suffix,
                                       uint32_t type, int count) {
    const BnGGUFKeyValue *kv = glm5next_key(f, suffix);
    return kv && kv->type == BN_GGUF_TYPE_ARRAY && kv->value.arr.elem_type == type &&
        kv->value.arr.n == (uint64_t)count && kv->value.arr.data ? &kv->value.arr : NULL;
}

static int glm5next_bind(BnGGUFFile *f, int layer, BnModelTensorRole role,
                         int nd, int d0, int d1, int d2, int f32) {
    char name[128];
    if (glm5next_tensor_name(name, sizeof(name), layer, role)) return -1;
    int found = -1;
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        if (f->tensors[i].name && strcmp(f->tensors[i].name, name) == 0) {
            if (found >= 0) return -1;
            found = (int)i;
        }
    }
    if (found < 0) return -1;
    const BnGGUFTensorInfo *t = &f->tensors[found];
    if (t->n_dims != (uint32_t)nd || t->dims[0] != (uint64_t)d0 ||
        (nd > 1 && t->dims[1] != (uint64_t)d1) ||
        (nd > 2 && t->dims[2] != (uint64_t)d2) || (f32 && t->type != BN_GGUF_TENSOR_F32))
        return -1;
    const BnQuantFormatOps *format = bn_quant_format_ops((int)t->type);
    if (!format || !(format->caps & BN_QUANT_CAP_LOADABLE) ||
        !(format->caps & BN_QUANT_CAP_CPU_MATVEC) || format->block_elems <= 0 ||
        !format->bytes_per_block || d0 % format->block_elems) return -1;
    uint64_t elements = 1;
    for (int i = 0; i < nd; i++) {
        if (!t->dims[i] || elements > UINT64_MAX / t->dims[i]) return -1;
        elements *= t->dims[i];
    }
    size_t bytes;
    return !bn_gguf_tensor_size(t->type, elements, &bytes) || !bytes ? -1 : found;
}

static int glm5next_plan_anatomy(BnGGUFFile *f, BnModelAnatomy *out) {
    BnModelAnatomy p = {0};
#define READ_INT(FIELD, KEY) p.FIELD = glm5next_int(f, KEY)
#define READ_FLOAT(FIELD, KEY) p.FIELD = glm5next_float(f, KEY)
    READ_INT(dim, "embedding_length"); READ_INT(stored_layers, "block_count");
    READ_INT(nextn_layers, "nextn_predict_layers"); READ_INT(vocab_size, "vocab_size");
    READ_INT(context_length, "context_length"); READ_INT(dense_hidden, "feed_forward_length");
    READ_INT(leading_dense_layers, "leading_dense_block_count");
    READ_INT(attention_heads, "attention.head_count");
    READ_INT(mla_key_dim, "attention.key_length_mla"); READ_INT(mla_value_dim, "attention.value_length_mla");
    READ_INT(q_rank, "attention.q_lora_rank"); READ_INT(kv_rank, "attention.kv_lora_rank");
    READ_INT(rope_dim, "rope.dimension_count"); READ_INT(kda_head_dim, "kda.head_dim");
    READ_INT(conv_kernel, "ssm.conv_kernel"); READ_INT(mhc_streams, "hyper_connection.count");
    READ_INT(mhc_iterations, "hyper_connection.sinkhorn_iterations");
    READ_INT(indexer_heads, "attention.indexer.head_count"); READ_INT(indexer_dim, "attention.indexer.key_length");
    READ_INT(indexer_top_k, "attention.indexer.top_k"); READ_INT(indexer_pool, "attention.indexer.kpool");
    READ_INT(experts, "expert_count"); READ_INT(active_experts, "expert_used_count");
    READ_INT(expert_hidden, "expert_feed_forward_length");
    READ_INT(shared_hidden, "expert_shared_feed_forward_length"); READ_INT(shared_experts, "expert_shared_count");
    READ_FLOAT(norm_eps, "attention.layer_norm_rms_epsilon");
    READ_FLOAT(indexer_norm_eps, "attention.layer_norm_epsilon");
    READ_FLOAT(mhc_eps, "hyper_connection.epsilon"); READ_FLOAT(kda_lower_bound, "kda.gate_lower_bound");
    READ_FLOAT(expert_weights_scale, "expert_weights_scale");
#undef READ_INT
#undef READ_FLOAT
    if (p.dim <= 0 || p.stored_layers < 1 || p.stored_layers > BN_MODEL_ANATOMY_MAX_LAYERS ||
        p.nextn_layers < 0 || p.nextn_layers >= p.stored_layers || p.vocab_size <= 0 || p.context_length <= 0 ||
        p.dense_hidden <= 0 || p.leading_dense_layers < 0 || p.attention_heads <= 0 ||
        p.mla_key_dim <= 0 || p.mla_value_dim <= 0 || p.q_rank <= 0 || p.kv_rank <= 0 ||
        p.rope_dim != 0 || p.kda_head_dim <= 0 || p.conv_kernel < 2 || p.conv_kernel > 4 ||
        p.mhc_streams < 1 || p.mhc_streams > 16 || p.mhc_iterations < 1 || p.mhc_iterations > 128 ||
        p.dim > INT_MAX / p.mhc_streams || p.attention_heads > INT_MAX / p.mla_key_dim ||
        p.attention_heads > INT_MAX / p.mla_value_dim || p.indexer_heads <= 0 || p.indexer_dim != 128 ||
        p.indexer_heads > INT_MAX / p.indexer_dim || p.indexer_pool < 2 || p.indexer_top_k <= 0 ||
        p.indexer_top_k % p.indexer_pool || p.experts <= 0 || p.active_experts <= 0 ||
        p.active_experts > p.experts || p.expert_hidden <= 0 || p.shared_hidden <= 0 || p.shared_experts != 1 ||
        !isfinite(p.norm_eps) || p.norm_eps <= 0 || !isfinite(p.indexer_norm_eps) || p.indexer_norm_eps <= 0 ||
        !isfinite(p.mhc_eps) || p.mhc_eps <= 0 || !isfinite(p.kda_lower_bound) || p.kda_lower_bound >= 0 ||
        !isfinite(p.expert_weights_scale) || p.expert_weights_scale <= 0 ||
        glm5next_int(f, "attention.key_length") != p.kv_rank ||
        glm5next_int(f, "attention.value_length") != p.kv_rank ||
        glm5next_int(f, "expert_gating_func") != 2 || glm5next_int(f, "expert_group_count") != 1 ||
        glm5next_int(f, "expert_group_used_count") != 1) goto invalid;
    p.n_layers = p.stored_layers - p.nextn_layers;
    if (p.leading_dense_layers > p.n_layers) goto invalid;
    const BnGGUFKeyValue *normalizes = glm5next_key(f, "expert_weights_norm");
    if (!normalizes || normalizes->type != BN_GGUF_TYPE_BOOL || normalizes->value.b > 1) goto invalid;
    p.normalize_expert_weights = normalizes->value.b;
    const BnGGUFKeyValue *kv = glm5next_key(f, "attention.head_count_kv");
    if (!kv || kv->type != BN_GGUF_TYPE_ARRAY || !kv->value.arr.data ||
        kv->value.arr.n != (uint64_t)p.stored_layers ||
        (kv->value.arr.elem_type != BN_GGUF_TYPE_UINT32 && kv->value.arr.elem_type != BN_GGUF_TYPE_INT32)) goto invalid;
    const BnGGUFArray *clamp = glm5next_array(f, "swiglu_clamp_exp", BN_GGUF_TYPE_FLOAT32, p.stored_layers);
    const BnGGUFArray *shared_clamp = glm5next_array(f, "swiglu_clamp_shexp", BN_GGUF_TYPE_FLOAT32, p.stored_layers);
    if (!clamp || !shared_clamp) goto invalid;
    p.kda_qk_eps = 1e-6f; p.kda_output_eps = 1e-5f; p.kda_parameter_is_negative_exp = 1;
    p.kda_gate_rank = p.kda_head_dim;
    for (int l = 0; l < BN_MODEL_ANATOMY_MAX_LAYERS; l++)
        for (int r = 0; r < BN_MODEL_TENSOR_ROLE_COUNT; r++) p.layers[l].tensor[r] = -1;
    p.embedding_tensor = glm5next_bind(f, -1, BN_MODEL_TENSOR_TOKEN_EMBD, 2, p.dim, p.vocab_size, 0, 0);
    p.output_tensor = glm5next_bind(f, -1, BN_MODEL_TENSOR_OUTPUT, 2, p.dim, p.vocab_size, 0, 0);
    p.output_norm_tensor = glm5next_bind(f, -1, BN_MODEL_TENSOR_OUTPUT_NORM, 1, p.dim, 0, 0, 1);
    if (p.embedding_tensor < 0 || p.output_tensor < 0 || p.output_norm_tensor < 0) goto invalid;
    for (int l = 0; l < p.stored_layers; l++) {
        uint32_t kind;
        memcpy(&kind, (const char *)kv->value.arr.data + (size_t)l * sizeof(kind), sizeof(kind));
        float ce, cs;
        memcpy(&ce, (const char *)clamp->data + (size_t)l * sizeof(ce), sizeof(ce));
        memcpy(&cs, (const char *)shared_clamp->data + (size_t)l * sizeof(cs), sizeof(cs));
        if ((kind != 0 && kind != 1) || !isfinite(ce) || ce <= 0 || !isfinite(cs) || cs <= 0) goto invalid;
        if (l >= p.n_layers) continue;
        BnModelLayerAnatomy *layer = &p.layers[l];
        layer->kind = kind ? BN_MODEL_ANATOMY_INDEXED_MLA : BN_MODEL_ANATOMY_KDA;
        layer->uses_moe = l >= p.leading_dense_layers;
        layer->swiglu_clamp_exp = ce; layer->swiglu_clamp_shexp = cs;
#define BIND(ROLE, ND, D0, D1, D2, F32) do { \
    int index = glm5next_bind(f, l, ROLE, ND, D0, D1, D2, F32); \
    if (index < 0) { goto invalid; } layer->tensor[ROLE] = index; \
} while (0)
        BIND(BN_MODEL_TENSOR_ATTN_NORM, 1, p.dim, 0, 0, 1);
        BIND(BN_MODEL_TENSOR_FFN_NORM, 1, p.dim, 0, 0, 1);
        int streams = p.mhc_streams, coeff = streams * (streams + 2);
        BIND(BN_MODEL_TENSOR_MHC_ATTN_FN, 2, p.dim * streams, coeff, 0, 0);
        BIND(BN_MODEL_TENSOR_MHC_ATTN_BASE, 1, coeff, 0, 0, 1);
        BIND(BN_MODEL_TENSOR_MHC_ATTN_SCALE, 1, 3, 0, 0, 1);
        BIND(BN_MODEL_TENSOR_MHC_FFN_FN, 2, p.dim * streams, coeff, 0, 0);
        BIND(BN_MODEL_TENSOR_MHC_FFN_BASE, 1, coeff, 0, 0, 1);
        BIND(BN_MODEL_TENSOR_MHC_FFN_SCALE, 1, 3, 0, 0, 1);
        if (!kind) {
            char name[128];
            if (glm5next_tensor_name(name, sizeof(name), l, BN_MODEL_TENSOR_ATTN_Q)) goto invalid;
            int qi = bn_gguf_find_tensor(f, name);
            if (qi < 0 || f->tensors[qi].n_dims != 2 || !f->tensors[qi].dims[1] ||
                f->tensors[qi].dims[1] > INT_MAX / 3 || f->tensors[qi].dims[1] % p.kda_head_dim) goto invalid;
            int projection = (int)f->tensors[qi].dims[1], heads = projection / p.kda_head_dim;
            if (p.kda_heads && p.kda_heads != heads) goto invalid;
            p.kda_heads = heads;
            BIND(BN_MODEL_TENSOR_ATTN_Q, 2, p.dim, projection, 0, 0);
            BIND(BN_MODEL_TENSOR_ATTN_K, 2, p.dim, projection, 0, 0);
            BIND(BN_MODEL_TENSOR_ATTN_V, 2, p.dim, projection, 0, 0);
            BIND(BN_MODEL_TENSOR_SSM_OUT, 2, projection, p.dim, 0, 0);
            BIND(BN_MODEL_TENSOR_SSM_BETA, 2, p.dim, heads, 0, 0);
            BIND(BN_MODEL_TENSOR_SSM_F_A, 2, p.dim, p.kda_gate_rank, 0, 0);
            BIND(BN_MODEL_TENSOR_SSM_G_A, 2, p.dim, p.kda_gate_rank, 0, 0);
            BIND(BN_MODEL_TENSOR_SSM_F_B, 2, p.kda_gate_rank, projection, 0, 0);
            BIND(BN_MODEL_TENSOR_SSM_G_B, 2, p.kda_gate_rank, projection, 0, 0);
            BIND(BN_MODEL_TENSOR_SSM_CONV1D_Q, 3, p.conv_kernel, 1, projection, 1);
            BIND(BN_MODEL_TENSOR_SSM_CONV1D_K, 3, p.conv_kernel, 1, projection, 1);
            BIND(BN_MODEL_TENSOR_SSM_CONV1D_V, 3, p.conv_kernel, 1, projection, 1);
            BIND(BN_MODEL_TENSOR_SSM_A, 1, heads, 0, 0, 1);
            BIND(BN_MODEL_TENSOR_SSM_DT_BIAS, 1, projection, 0, 0, 1);
            BIND(BN_MODEL_TENSOR_SSM_NORM, 1, p.kda_head_dim, 0, 0, 1);
        } else {
            BIND(BN_MODEL_TENSOR_MLA_Q_A, 2, p.dim, p.q_rank, 0, 0);
            BIND(BN_MODEL_TENSOR_MLA_Q_A_NORM, 1, p.q_rank, 0, 0, 1);
            BIND(BN_MODEL_TENSOR_MLA_Q_B, 2, p.q_rank, p.attention_heads * p.mla_key_dim, 0, 0);
            BIND(BN_MODEL_TENSOR_MLA_KV_A, 2, p.dim, p.kv_rank, 0, 0);
            BIND(BN_MODEL_TENSOR_MLA_KV_A_NORM, 1, p.kv_rank, 0, 0, 1);
            BIND(BN_MODEL_TENSOR_MLA_K_B, 3, p.mla_key_dim, p.kv_rank, p.attention_heads, 0);
            BIND(BN_MODEL_TENSOR_MLA_V_B, 3, p.kv_rank, p.mla_value_dim, p.attention_heads, 0);
            BIND(BN_MODEL_TENSOR_ATTN_OUTPUT, 2, p.attention_heads * p.mla_value_dim, p.dim, 0, 0);
            BIND(BN_MODEL_TENSOR_INDEXER_Q, 2, p.q_rank, p.indexer_heads * p.indexer_dim, 0, 0);
            BIND(BN_MODEL_TENSOR_INDEXER_K, 2, p.dim, p.indexer_dim, 0, 0);
            BIND(BN_MODEL_TENSOR_INDEXER_K_NORM, 1, p.indexer_dim, 0, 0, 1);
            BIND(BN_MODEL_TENSOR_INDEXER_K_BIAS, 1, p.indexer_dim, 0, 0, 1);
            BIND(BN_MODEL_TENSOR_INDEXER_HEAD_WEIGHTS, 2, p.dim, p.indexer_heads, 0, 1);
            BIND(BN_MODEL_TENSOR_INDEXER_POOL_APE, 2, p.indexer_dim, p.indexer_pool, 0, 1);
            BIND(BN_MODEL_TENSOR_INDEXER_POOL_GATE, 2, p.dim, p.indexer_dim, 0, 0);
        }
        if (!layer->uses_moe) {
            BIND(BN_MODEL_TENSOR_FFN_GATE, 2, p.dim, p.dense_hidden, 0, 0);
            BIND(BN_MODEL_TENSOR_FFN_UP, 2, p.dim, p.dense_hidden, 0, 0);
            BIND(BN_MODEL_TENSOR_FFN_DOWN, 2, p.dense_hidden, p.dim, 0, 0);
        } else {
            BIND(BN_MODEL_TENSOR_MOE_ROUTER, 2, p.dim, p.experts, 0, 1);
            BIND(BN_MODEL_TENSOR_MOE_ROUTER_BIAS, 1, p.experts, 0, 0, 1);
            BIND(BN_MODEL_TENSOR_MOE_GATE_EXPS, 3, p.dim, p.expert_hidden, p.experts, 0);
            BIND(BN_MODEL_TENSOR_MOE_UP_EXPS, 3, p.dim, p.expert_hidden, p.experts, 0);
            BIND(BN_MODEL_TENSOR_MOE_DOWN_EXPS, 3, p.expert_hidden, p.dim, p.experts, 0);
            BIND(BN_MODEL_TENSOR_SHARED_FFN_GATE, 2, p.dim, p.shared_hidden, 0, 0);
            BIND(BN_MODEL_TENSOR_SHARED_FFN_UP, 2, p.dim, p.shared_hidden, 0, 0);
            BIND(BN_MODEL_TENSOR_SHARED_FFN_DOWN, 2, p.shared_hidden, p.dim, 0, 0);
        }
#undef BIND
    }
    if (!p.kda_heads) goto invalid;
    *out = p;
    return 0;
invalid:
    fprintf(stderr, "GLM model anatomy: invalid metadata or tensor layout\n");
    return -1;
}

int bn_model_arch_plan_anatomy(BnGGUFFile *file, BnModelAnatomy *out) {
    if (!file || !out || !file->kvs || !file->tensors || !file->n_kv || !file->n_tensors ||
        file->n_kv > BN_GGUF_MAX_COUNT || file->n_tensors > BN_GGUF_MAX_COUNT) return -1;
    int architecture_keys = 0;
    for (uint64_t i = 0; i < file->n_kv; i++)
        if (file->kvs[i].key && !strcmp(file->kvs[i].key, "general.architecture")) architecture_keys++;
    if (architecture_keys != 1) return -1;
    const char *arch = bn_gguf_get_str(file, "general.architecture");
    const BnModelArchOps *ops = bn_model_arch_ops_for(arch);
    return ops && ops->plan_anatomy ? ops->plan_anatomy(file, out) : -1;
}

const BnModelArchOps *bn_model_arch_registry(size_t *count) {
    static const BnModelArchOps ops[] = {
        {
            .name = "glm5next",
            .is_match = glm5next_match,
            .prefix = bn_model_arch_prefix,
            .activation = bn_model_arch_default_activation,
            .attention_value_shares_key = bn_model_arch_attention_value_unique,
            .tensor_name = glm5next_tensor_name,
            .ssm_qk_norm = BN_SSM_QK_NORM_L2_SUM_EPS,
            .bf16_layout = BN_MODEL_BF16_LAYOUT_GLM_MHC,
            .plan_anatomy = glm5next_plan_anatomy,
        },
        {
            "qwen4exp",
            BN_MODEL_ARCH_POLICY_GGUF_ZERO_CENTERED_NORM_SHIFTED |
            BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION |
            BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT |
            BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
            BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION |
            BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER |
            BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS |
            BN_MODEL_ARCH_POLICY_SEPARATE_ROPE_NORM |
            BN_MODEL_ARCH_POLICY_HYPER_CONNECTIONS |
            BN_MODEL_ARCH_POLICY_QUERY_SPARSE_ATTENTION |
            BN_MODEL_ARCH_POLICY_SSM_SIGMOID_GATE |
            BN_MODEL_ARCH_POLICY_POSITIONAL_LAYER_EMBEDDING |
            BN_MODEL_ARCH_POLICY_MOE_SEPARATE_ROUTER_TOPK,
            BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ATTENTION |
            BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION,
            bn_model_arch_match_qwen4exp,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_qwen4exp_tensor_name,
            bn_model_arch_apply_default_shapes,
            BN_SSM_QK_NORM_RMS_SCALED,
            BN_MODEL_BF16_LAYOUT_FLASH_HC,
            NULL,
        },
        {
            "gemma4",
            BN_MODEL_ARCH_POLICY_UNIT_ATTENTION_SCALE |
            BN_MODEL_ARCH_POLICY_ATTENTION_VALUE_SHARES_KEY |
            BN_MODEL_ARCH_POLICY_PER_LAYER_INPUT |
            BN_MODEL_ARCH_POLICY_ATTENTION_POST_NORM |
            BN_MODEL_ARCH_POLICY_FFN_POST_NORM |
            BN_MODEL_ARCH_POLICY_LAYER_OUTPUT_SCALE |
            BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION |
            BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER |
            BN_MODEL_ARCH_POLICY_PREFILL_REFERENCE_ACTIVATION |
            BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION |
            BN_MODEL_ARCH_POLICY_MOE_SCALED_ROUTER_INPUT |
            BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION |
            BN_MODEL_ARCH_POLICY_MOE_DENSE_RESIDUAL_BRANCH,
            BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION,
            bn_model_arch_match_gemma4,
            bn_model_arch_prefix,
            bn_model_arch_gemma4_activation,
            bn_model_arch_attention_value_shared,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_gemma4_tensor_name,
            bn_model_arch_apply_gemma4_shapes,
            BN_SSM_QK_NORM_CLAMPED_L2,
            BN_MODEL_BF16_LAYOUT_NONE,
            NULL,
        },
        {
            "qwen3",
            BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION |
            BN_MODEL_ARCH_POLICY_SEPARATE_ROPE_NORM |
            BN_MODEL_ARCH_POLICY_PREFILL_REFERENCE_ACTIVATION |
            BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION |
            BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION_LAST_THIRD,
            BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ROUTER_ACCUMULATION |
            BN_MODEL_ARCH_POLICY_MOE_REFERENCE_SILU,
            bn_model_arch_match_qwen3,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
            BN_SSM_QK_NORM_CLAMPED_L2,
            BN_MODEL_BF16_LAYOUT_NONE,
            NULL,
        },
        {
            "qwen35",
            BN_MODEL_ARCH_POLICY_REFERENCE_ATTENTION |
            BN_MODEL_ARCH_POLICY_REFERENCE_RECURRENT |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK |
            BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION |
            BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER |
            BN_MODEL_ARCH_POLICY_FULL_ROPE_TEXT_DIMS |
            BN_MODEL_ARCH_POLICY_SEPARATE_ROPE_NORM,
            0,
            bn_model_arch_match_qwen35,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
            BN_SSM_QK_NORM_RMS_SCALED,
            BN_MODEL_BF16_LAYOUT_HYBRID_OFFSET_NORM,
            NULL,
        },
        {
            "qwen2",
            BN_MODEL_ARCH_POLICY_REFERENCE_RMSNORM_ORDER |
            BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK,
            BN_MODEL_ARCH_POLICY_MOE_REFERENCE_SILU |
            BN_MODEL_ARCH_POLICY_MOE_FLOAT_KQUANT_GATEUP_FALLBACK |
            BN_MODEL_ARCH_POLICY_MOE_REFERENCE_ATTENTION |
            BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
            BN_MODEL_ARCH_POLICY_MOE_UNNORMALIZED_TOPK,
            bn_model_arch_match_qwen2,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
            BN_SSM_QK_NORM_CLAMPED_L2,
            BN_MODEL_BF16_LAYOUT_NONE,
            NULL,
        },
        {
            "qwen",
            BN_MODEL_ARCH_POLICY_PREFILL_DECODE_PARITY |
            BN_MODEL_ARCH_POLICY_SMALL_DENSE_PREFILL_DECODE_FALLBACK |
            BN_MODEL_ARCH_POLICY_REFERENCE_FFN_ACTIVATION,
            0,
            bn_model_arch_match_qwen,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
            BN_SSM_QK_NORM_CLAMPED_L2,
            BN_MODEL_BF16_LAYOUT_NONE,
            NULL,
        },
        {
            "bitnet",
            0,
            0,
            bn_model_arch_match_bitnet,
            bn_model_arch_prefix,
            bn_model_arch_bitnet_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
            BN_SSM_QK_NORM_CLAMPED_L2,
            BN_MODEL_BF16_LAYOUT_NONE,
            NULL,
        },
        {
            "default",
            0,
            0,
            bn_model_arch_match_default,
            bn_model_arch_prefix,
            bn_model_arch_default_activation,
            bn_model_arch_attention_value_unique,
            bn_model_arch_is_ssm_layer,
            bn_model_arch_default_tensor_name,
            bn_model_arch_apply_default_shapes,
            BN_SSM_QK_NORM_CLAMPED_L2,
            BN_MODEL_BF16_LAYOUT_NONE,
            NULL,
        },
    };
    if (count) *count = sizeof(ops) / sizeof(ops[0]);
    return ops;
}

const BnModelArchOps *bn_model_arch_ops_for(const char *arch) {
    size_t n = 0;
    const BnModelArchOps *ops = bn_model_arch_registry(&n);
    for (size_t i = 0; i < n; i++) {
        if (ops[i].is_match && ops[i].is_match(arch)) return &ops[i];
    }
    return n > 0 ? &ops[n - 1] : NULL;
}

float bn_model_arch_zero_centered_norm_storage_offset(const BnConfig *c) {
    return c && (c->policy_flags & BN_MODEL_ARCH_POLICY_GGUF_ZERO_CENTERED_NORM_SHIFTED) ? 1.0f : 0.0f;
}

#include "model_arch_engram.h"
int bn_model_arch_native_engram_layout(const BnJSON *document,int layer,BnModelNativeEngramLayout *out) {
    BnModelNativeNumerics numeric;
    const BnJSONValue *text=bn_json_get(bn_json_root(document),"text_config");
    const BnJSONValue *ids=bn_json_get(text,"engram_layer_ids"),*rows=bn_json_get(text,"engram_num_embeddings");
    int layers,ngram,heads;
    if(!out||bn_model_arch_native_numerics(document,&numeric)||
        native_config_integer(text,"num_hidden_layers",1,BN_MODEL_ANATOMY_MAX_LAYERS,&layers)||layer<0||layer>=layers||
        native_config_integer(text,"engram_max_ngram_size",2,BN_MODEL_NATIVE_ENGRAM_MAX_NGRAM,&ngram)||
        native_config_integer(text,"engram_n_heads",1,BN_MODEL_NATIVE_ENGRAM_MAX_HEADS,&heads)||
        heads>BN_MODEL_NATIVE_ENGRAM_MAX_HEADS/(ngram-1)||numeric.engram_vocab<2||numeric.engram_compressed_vocab<1||
        bn_json_type(ids)!=BN_JSON_ARRAY||bn_json_type(rows)!=BN_JSON_ARRAY||
        !bn_json_size(ids)||bn_json_size(ids)!=(size_t)bn_json_size(rows)||bn_json_size(ids)>(size_t)layers)return -1;
    BnModelNativeEngramLayout result={0};unsigned char seen[BN_MODEL_ANATOMY_MAX_LAYERS]={0};int found=0;
    uint32_t prime=(uint32_t)numeric.engram_vocab-1;int total=(ngram-1)*heads;
    for(size_t i=0;i<bn_json_size(ids);i++) {
        uint64_t id,expected;if(bn_json_u64(bn_json_at(ids,i),&id)||id>=(uint64_t)layers||seen[id]||
            bn_json_u64(bn_json_at(rows,i),&expected)||!expected)return -1;
        seen[id]=1;uint64_t sum=0;
        for(int h=0;h<total;h++) {
            do{if(prime==INT_MAX)return -1;prime++;}while(!native_hash_prime(prime));
            if((int)id==layer){result.sizes[h]=prime;result.offsets[h]=sum;}
            sum+=prime;
        }
        if(sum!=expected)return -1;
        if((int)id==layer){found=1;result.rows=sum;}
    }
    if(!found)return -1;
    result.layer=layer;result.ngram=ngram;result.heads_per_ngram=heads;result.heads=total;
    result.compressed_vocab=numeric.engram_compressed_vocab;result.pad_token=numeric.engram_pad_token;
    uint64_t bound=INT64_MAX/(uint64_t)numeric.engram_compressed_vocab/2;
    NativeHashPCG rng;native_hash_seed(&rng,(uint32_t)(10007*layer));
    for(int i=0;i<ngram;i++)result.multipliers[i]=2*native_hash_bounded(&rng,bound)+1;
    *out=result;return 0;
}

static int native_engram_memory_disjoint(const void *a,size_t n,const void *b,size_t m) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(n>UINTPTR_MAX-x||m>UINTPTR_MAX-y)return 0;
    return x+n<=y||y+m<=x;
}
int bn_model_arch_native_engram_hash_rows(const BnModelNativeEngramLayout *p,uint32_t pad,
    const uint32_t *history,size_t batch,size_t history_len,size_t out_len,
    uint64_t *out,size_t out_count) {
    if(!p||p->layer<0||p->layer>=BN_MODEL_ANATOMY_MAX_LAYERS||p->ngram<2||
        p->ngram>BN_MODEL_NATIVE_ENGRAM_MAX_NGRAM||p->heads_per_ngram<1||
        p->heads_per_ngram>BN_MODEL_NATIVE_ENGRAM_MAX_HEADS/(p->ngram-1)||
        p->heads!=(p->ngram-1)*p->heads_per_ngram||p->compressed_vocab<1||
        pad>=(uint32_t)p->compressed_vocab||!batch||out_len>history_len)return -1;
    uint64_t sum=0;
    for(int h=0;h<p->heads;h++) {
        if(!p->sizes[h]||p->sizes[h]>INT_MAX||p->offsets[h]!=sum)return -1;
        sum+=p->sizes[h];
    }
    if(p->rows!=sum)return -1;
    uint64_t limit=INT64_MAX/(uint64_t)p->compressed_vocab;
    for(int i=0;i<p->ngram;i++)if(!p->multipliers[i]||!(p->multipliers[i]&1)||p->multipliers[i]>limit)return -1;
    if(out_len>SIZE_MAX/batch||batch*out_len>SIZE_MAX/(size_t)p->heads||
        out_count!=batch*out_len*(size_t)p->heads||out_count>SIZE_MAX/sizeof(*out))return -1;
    if(history_len>SIZE_MAX/batch||batch*history_len>SIZE_MAX/sizeof(*history))return -1;
    /* Defined empty scheduling operation; no source -0 slicing is performed. */
    if(!out_len)return 0;
    size_t hc=batch*history_len,hb=hc*sizeof(*history),ob=out_count*sizeof(*out);
    if(!history||!out||!native_engram_memory_disjoint(out,ob,history,hb)||
        !native_engram_memory_disjoint(out,ob,p,sizeof(*p)))return -1;
    /* Validate the complete input before writing any output, including prefix. */
    for(size_t i=0;i<hc;i++)if(history[i]>=(uint32_t)p->compressed_vocab)return -1;
    size_t first=history_len-out_len;
    for(size_t b=0;b<batch;b++) {
        const uint32_t *tokens=history+b*history_len;size_t segment=0;
        for(size_t t=0;t<first;t++)if(tokens[t]==pad)segment=t+1;
        for(size_t t=first;t<history_len;t++) {
            uint64_t mixed=(uint64_t)tokens[t]*p->multipliers[0];
            uint64_t *dest=out+(b*out_len+(t-first))*(size_t)p->heads;
            for(int shift=1;shift<p->ngram;shift++) {
                uint32_t token=t>=(size_t)shift&&t-(size_t)shift>=segment?tokens[t-(size_t)shift]:pad;
                mixed^=(uint64_t)token*p->multipliers[shift];
                int base=(shift-1)*p->heads_per_ngram;
                for(int h=0;h<p->heads_per_ngram;h++)dest[base+h]=mixed%p->sizes[base+h]+p->offsets[base+h];
            }
            /* A pad can end a gram. Only subsequent positions reset context. */
            if(tokens[t]==pad)segment=t+1;
        }
    }
    return 0;
}

int bn_model_arch_native_engram_map_spec(const BnModelNativeAnatomy *a,
    const BnModelNativeNumerics *n,BnTokenizerMapSpec *out) {
    if(!a||!n||!out||a->engram_ngram<2||a->engram_heads<1||a->vocab<1||
        n->engram_compressed_vocab<1||n->engram_compressed_vocab>a->vocab||
        n->engram_pad_token<0||n->engram_pad_token>=a->vocab)return -1;
    *out=(BnTokenizerMapSpec){BN_TOKENIZER_MAP_ENGRAM_V41,a->vocab,n->engram_compressed_vocab,n->engram_pad_token};return 0;
}

int bn_model_arch_native_head(const BnModelNativeAnatomy*a,const BnModelNativeNumerics*n,BnModelNativeHead*out) {
    if(!a||!n||!out||fegetround()!=FE_TONEAREST||a->core_format!=BN_QUANT_FORMAT_E4M3_UE8M0_32||
       a->experts.layers<1||a->experts.layers>BN_MODEL_ANATOMY_MAX_LAYERS||
       a->experts.dim<128||a->experts.dim>32768||a->experts.dim%128||a->vocab<1||
       a->output<0||a->output_norm<0||a->streams!=4||n->hc_streams!=4||!n->hc_carry_pre||n->rms_epsilon!=1e-20f)return -1;
    *out=(BnModelNativeHead){BN_NATIVE_HEAD_F16_LINEAR_F16,a->experts.dim,a->vocab};return 0;
}
int bn_model_arch_native_engram_math(const BnModelNativeAnatomy *a,
    const BnModelNativeNumerics *n,int layer,BnModelNativeEngramMath *out) {
    if(!a||!n||!out||fegetround()!=FE_TONEAREST||a->experts.layers<1||a->experts.layers>BN_MODEL_ANATOMY_MAX_LAYERS||
       layer<0||layer>=a->experts.layers||!a->layers[layer].engram_rows||
       a->engram_format!=BN_QUANT_FORMAT_INT4_FP16S32||a->core_format!=BN_QUANT_FORMAT_E4M3_UE8M0_32||
       a->experts.dim<4||a->experts.dim>32768||a->experts.dim%4||
       a->streams!=4||n->hc_streams!=a->streams||n->rms_epsilon!=1e-20f||
       a->engram_ngram<2||a->engram_ngram>BN_MODEL_NATIVE_ENGRAM_MAX_NGRAM||
       a->engram_heads<1||a->engram_heads>BN_MODEL_NATIVE_ENGRAM_MAX_HEADS/(a->engram_ngram-1)||
       a->engram_dim<32||a->engram_dim%32||a->layers[layer].tensor[BN_NATIVE_ENGRAM_WKV]<0||
       a->layers[layer].tensor[BN_NATIVE_ENGRAM_K]<0||a->layers[layer].tensor[BN_NATIVE_ENGRAM_Q]<0)return -1;
    int heads=(a->engram_ngram-1)*a->engram_heads;
    if(a->engram_dim>32768/heads)return -1;
    BnModelNativeEngramMath spec={BN_NATIVE_ENGRAM_MATH_BF16_LOOKUP_F16,layer,a->experts.dim,
        a->streams,heads,a->engram_dim,heads*a->engram_dim,n->rms_epsilon,
        (float)(1.0/sqrt((double)a->experts.dim))};
    *out=spec;return 0;
}
