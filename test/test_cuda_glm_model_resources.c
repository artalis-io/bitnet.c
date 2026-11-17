/* Actual complete main-model resource ownership; no generation admission. */
#include "backend_layout.h"
#include "backend_model.h"
#include "gpu_cuda.h"
#include "model_arch.h"
#include "moe.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv) {
    assert(argc==2);BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);
    BnModelAnatomy*a=malloc(sizeof(*a));assert(a);assert(!bn_model_arch_plan_anatomy(f,a));
    assert(a->n_layers==45&&a->dim==4096&&a->vocab_size==154880);
    BnMoEExpertMap*maps=calloc((size_t)a->n_layers,sizeof(*maps));assert(maps);
    for(int i=0;i<a->n_layers;++i)if(a->layers[i].uses_moe){const BnModelLayerAnatomy*l=&a->layers[i];
        BnMoEExpertTensorNames names={.gate=f->tensors[l->tensor[BN_MODEL_TENSOR_MOE_GATE_EXPS]].name,
            .up=f->tensors[l->tensor[BN_MODEL_TENSOR_MOE_UP_EXPS]].name,.down=f->tensors[l->tensor[BN_MODEL_TENSOR_MOE_DOWN_EXPS]].name};
        assert(!bn_moe_load_expert_map(f,&names,288,2048,&maps[i]));}
    BnGPUBackend*g=bn_gpu_cuda_create();assert(g);BnBackendModel*b=bn_backend_model_create();assert(b);
    size_t budget=(size_t)12*1024*1024*1024,host=(size_t)256*1024*1024;
    assert(!bn_backend_layout_prepare_bf16_glm_model(b,g,f,a,maps,45,budget,host));
    const BnBackendBF16GLMModel*m=bn_backend_layout_bf16_glm_model(b);assert(m&&m->gpu==g&&bn_backend_model_gpu(b)==g);
    assert(m->anatomy!=a&&m->n_layers==45&&m->dim==4096&&m->vocab_size==154880);
    assert(m->embedding.raw&&m->output.raw&&m->embedding.raw!=m->output.raw&&m->output_norm);
    assert(m->embedding.type==BN_GGUF_TENSOR_Q8_0&&m->output.type==BN_GGUF_TENSOR_Q8_0&&!m->embedding.decoded&&!m->output.decoded);
    /* Independent payload constants from the separately qualified actual
     * component bindings. Routed expert source bytes are excluded. */
    size_t expected=2*((size_t)154880*4096/32*34)+4096*2;int nk=0,na=0,nm=0,nd=0;
    for(int i=0;i<45;++i){const BnBackendBF16GLMLayer*l=&m->layers[i];
        assert(l->attention_mhc.gpu==g&&l->ffn_mhc.gpu==g&&l->attention_mhc.bytes==1581164&&l->ffn_mhc.bytes==1581164);
        expected+=2*1581164;
        if(a->layers[i].kind==BN_MODEL_ANATOMY_KDA){nk++;assert(l->kda.gpu==g&&l->kda.layer==i&&l->kda.bytes==146653952&&!l->mla.gpu&&!l->indexer.gpu);expected+=146653952;}
        else{na++;assert(l->mla.gpu==g&&l->mla.layer==i&&l->mla.bytes==140513280&&l->indexer.gpu==g&&l->indexer.layer==i&&l->indexer.bytes==8326144&&!l->kda.gpu);expected+=140513280+8326144;}
        if(a->layers[i].uses_moe){nm++;assert(l->moe.gpu==g&&l->moe.layer==i&&l->moe.bytes==31458432&&!l->dense.gpu);
            assert(l->moe.source_map==&l->expert_map&&l->moe.source_map!=&maps[i]&&!l->expert_map.repacked);
            assert(l->moe.gate_type==(i==11?BN_GGUF_TENSOR_Q5_K:BN_GGUF_TENSOR_Q4_K));
            assert(l->moe.down_type==((i==11||i==12||i==44)?BN_GGUF_TENSOR_Q6_K:BN_GGUF_TENSOR_Q5_K));expected+=31458432;
        }else{nd++;assert(l->dense.gpu==g&&l->dense.bytes==160432128&&!l->moe.gpu);expected+=160432128;}
    }
    assert(nk==34&&na==11&&nm==42&&nd==3&&m->bytes==expected&&m->peak_bytes==expected&&m->host_staging_bytes==108347392);
    assert(m->peak_bytes<=budget&&m->host_staging_bytes<=host);
    const BnBackendBF16GLMModel*before=m;assert(bn_backend_layout_prepare_bf16_glm_model(b,g,f,a,maps,45,budget,host)==-1&&bn_backend_layout_bf16_glm_model(b)==before);
    printf("GLM model resources PASSED: layers45 KDA%d MLA%d MoE%d dense%d bytes%zu peak%zu host%zu\n",nk,na,nm,nd,m->bytes,m->peak_bytes,m->host_staging_bytes);fflush(stdout);
    bn_backend_model_release_gpu(b);assert(!bn_backend_layout_bf16_glm_model(b)&&!bn_backend_model_gpu_for_cpu_operations(b));
    assert(bn_backend_layout_prepare_bf16_glm_model(b,g,f,a,maps,45,expected-1,host)==-1);
    assert(!bn_backend_layout_bf16_glm_model(b)&&!bn_backend_model_gpu_for_cpu_operations(b));
    puts("GLM aggregate one-byte-short budget rollback PASSED");
    free(maps);free(a);bn_backend_model_free(b);bn_gpu_cuda_destroy(g);bn_gguf_free(f);return 0;
}
