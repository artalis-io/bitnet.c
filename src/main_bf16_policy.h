/* Explicit arithmetic metadata, not weights, expected outputs or token IDs. */
static int gguf_batch_policy_parse(const void *data,size_t bytes,BnTransformerBF16Policy *out) {
    BnJSON doc={0};if(!out||bytes>16384||bn_json_init(&doc,data,bytes))return -1;
    const BnJSONValue *root=bn_json_root(&doc);BnTransformerBF16Policy policy={0};int failed=1;
    const char *names[]={"arithmetic","ffn_arithmetic","input_norm_reductions","post_norm_reductions","attention_split_policy","gdn_prefill_selection"};
    if(bn_json_type(root)!=BN_JSON_OBJECT||bn_json_size(root)<4||bn_json_size(root)>6)goto done;
    for(size_t i=0;i<bn_json_size(root);i++) {
        const char *key=NULL;(void)bn_json_member(root,i,&key);int known=0;
        for(int j=0;j<6;j++)known|=!strcmp(key,names[j]);
        if(!known)goto done;
    }
    for(int i=0;i<6;i++) {
        const BnJSONValue *v=bn_json_get(root,names[i]);uint64_t n;
        if(i>=4&&!v)continue;
        if(i==2||i==3) {
            size_t count=i==2?65:64;BnGPURMSNormReduction *a=i==2?policy.input_norm_reductions:policy.post_norm_reductions;
            if(bn_json_type(v)!=BN_JSON_ARRAY||bn_json_size(v)!=count)goto done;
            for(size_t j=0;j<count;j++) {
                if(bn_json_u64(bn_json_at(v,j),&n)||n>BN_GPU_RMSNORM_REDUCTION_TILED_5120_PERSISTENT_FUSED_SQUARES)goto done;
                a[j]=(BnGPURMSNormReduction)n;
            }
        }else {
            unsigned limit=i==0?BN_TRANSFORMER_BF16_ARITHMETIC_NONCP_FP32:i==1?BN_TRANSFORMER_BF16_FFN_COMPILED:i==4?BN_GPU_ATTENTION_SPLIT_BATCH:BN_TRANSFORMER_BF16_GDN_PREFILL_SINGLE_CP;
            if(bn_json_u64(v,&n)||n>limit)goto done;
            if(i==0)policy.arithmetic=(BnTransformerBF16Arithmetic)n;
            else if(i==1)policy.ffn_arithmetic=(BnTransformerBF16FFNArithmetic)n;
            else if(i==4)policy.attention_split_policy=(BnGPUAttentionSplitPolicy)n;
            else policy.gdn_prefill_selection=(BnTransformerBF16GDNPrefillSelection)n;
        }
    }
    if(policy.arithmetic<BN_TRANSFORMER_BF16_ARITHMETIC_CP_FP32)goto done;
    *out=policy;failed=0;
done:bn_json_free(&doc);return failed?-1:0;
}
