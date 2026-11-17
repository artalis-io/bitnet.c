#ifndef __EMSCRIPTEN__
static void test_native_engram_math_spec(void) {
    BnModelNativeAnatomy*a=calloc(1,sizeof(*a));assert(a);a->experts.layers=40;a->experts.dim=128;
    a->layers[1].engram_rows=56;a->streams=4;a->engram_ngram=4;a->engram_heads=2;a->engram_dim=32;
    a->engram_format=BN_QUANT_FORMAT_INT4_FP16S32;a->core_format=BN_QUANT_FORMAT_E4M3_UE8M0_32;
    BnModelNativeNumerics n={0};n.rms_epsilon=1e-20f;n.hc_streams=4;
    BnModelNativeEngramMath out;assert(!bn_model_arch_native_engram_math(a,&n,1,&out));
    uint32_t bits;memcpy(&bits,&out.gate_scale,4);assert(bits==UINT32_C(0x3db504f3));
    assert(out.profile==BN_NATIVE_ENGRAM_MATH_BF16_LOOKUP_F16&&out.dim==128&&out.streams==4&&out.heads==6&&out.embedding_dim==192);
    BnModelNativeEngramMath saved=out;assert(bn_model_arch_native_engram_math(a,&n,0,&out)==-1&&!memcmp(&out,&saved,sizeof(out)));
    assert(bn_model_arch_native_engram_math(NULL,&n,1,&out)==-1&&bn_model_arch_native_engram_math(a,NULL,1,&out)==-1);
    a->experts.layers=INT_MAX;assert(bn_model_arch_native_engram_math(a,&n,1000,&out)==-1);a->experts.layers=40;
    int old=fegetround();if(!fesetround(FE_DOWNWARD)){assert(bn_model_arch_native_engram_math(a,&n,1,&out)==-1&&!memcmp(&out,&saved,sizeof(out)));assert(!fesetround(old));}
    a->experts.dim=5120;assert(!bn_model_arch_native_engram_math(a,&n,1,&out));memcpy(&bits,&out.gate_scale,4);assert(bits==UINT32_C(0x3c64f92e));
    n.rms_epsilon=1e-6f;assert(bn_model_arch_native_engram_math(a,&n,1,&out)==-1);free(a);
    BnModelNativeAnatomy*head_a=calloc(1,sizeof(*head_a));assert(head_a);head_a->core_format=BN_QUANT_FORMAT_E4M3_UE8M0_32;
    head_a->experts.layers=40;head_a->experts.dim=5120;head_a->vocab=129280;head_a->output=1;head_a->output_norm=2;head_a->streams=4;
    BnModelNativeNumerics head_n={0};head_n.hc_streams=4;head_n.hc_carry_pre=1;head_n.rms_epsilon=1e-20f;
    BnModelNativeHead head;assert(!bn_model_arch_native_head(head_a,&head_n,&head)&&head.profile==BN_NATIVE_HEAD_F16_LINEAR_F16&&head.dim==5120&&head.vocab==129280);
    BnModelNativeHead saved_head=head;head_n.hc_carry_pre=0;assert(bn_model_arch_native_head(head_a,&head_n,&head)==-1&&!memcmp(&head,&saved_head,sizeof(head)));free(head_a);
    puts("Native architecture-owned Engram math bridge/shape/scalar/atomic rejection PASSED");
}
#else
static void test_native_engram_math_spec(void) {}
#endif
