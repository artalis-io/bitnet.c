/* Real-model execution-invariance check, not an independent numeric oracle.
 * Last-logit and all-logit requests must preserve identical KV/SSM/PLE state
 * and produce identical next-step logits on the same teacher-forced token.
 * Usage: test_prefill_output_rows dense.gguf [--cuda]
 */
#include "model.h"
#include "model_internal.h"
#include "session.h"
#include "transformer.h"
#include "tokenizer.h"
#include "threadpool.h"
#include "gpu_backend.h"
#include "gpu_graph_ir.h"
#ifdef BN_ENABLE_CUDA
#include "gpu_cuda.h"
#endif
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t state_hash(const void *data, size_t bytes) {
    const unsigned char *p = data;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; i++) hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

static uint64_t gpu_hash(BnGPUBackend *gpu, int buffer, size_t bytes) {
    unsigned char *data = malloc(bytes);
    assert(data && gpu->read_activation(gpu->ctx, buffer, data, bytes, 0) == 0);
    uint64_t hash = state_hash(data, bytes);
    free(data);
    return hash;
}

static void snapshot(const BnModel *model, const BnSession *session,
                     BnGPUBackend *gpu, uint64_t hashes[9]) {
    const BnConfig *c = &model->config;
    const BnRunState *s = &session->state;
    size_t attn = (size_t)bn_model_session_policy_attention_layer_count(c);
    size_t ssm = (size_t)bn_model_session_policy_ssm_layer_count(c);
    size_t kv_bytes = attn * c->seq_len * c->kv_dim * sizeof(float);
    size_t state_bytes = ssm * c->ssm_time_step_rank * c->ssm_state_size *
        (c->ssm_time_step_rank ? c->ssm_inner_size / c->ssm_time_step_rank : 0) * sizeof(float);
    size_t conv_bytes = ssm * (c->ssm_conv_kernel ? c->ssm_conv_kernel - 1 : 0) *
        (c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size) * sizeof(float);
    size_t ple_bytes = c->ple_head_count ? (size_t)(c->ple_conv_kernel - 1) *
        c->ple_ngram_size * c->hyper_connection_count * c->dim * sizeof(float) : 0;
    hashes[0] = state_hash(s->key_cache, kv_bytes);
    hashes[1] = state_hash(s->value_cache, kv_bytes);
    hashes[2] = state_hash(s->ssm_state, state_bytes);
    hashes[3] = state_hash(s->ssm_conv_state, conv_bytes);
    hashes[4] = state_hash(s->ple_conv_state, ple_bytes);
    hashes[5] = gpu && kv_bytes ? gpu_hash(gpu, BN_GPU_VALUE_KEY_CACHE, kv_bytes) : 0;
    hashes[6] = gpu && kv_bytes ? gpu_hash(gpu, BN_GPU_VALUE_VALUE_CACHE, kv_bytes) : 0;
    hashes[7] = gpu && state_bytes ? gpu_hash(gpu, BN_GPU_VALUE_SSM_STATE, state_bytes) : 0;
    hashes[8] = gpu && conv_bytes ? gpu_hash(gpu, BN_GPU_VALUE_SSM_CONV_STATE, conv_bytes) : 0;
}

int main(int argc, char **argv) {
    assert(argc == 2 || (argc == 3 && strcmp(argv[2], "--cuda") == 0));
    int cuda = argc == 3;
#ifndef BN_ENABLE_CUDA
    assert(!cuda);
#endif
    BnGGUFFile *file = bn_gguf_open_file(argv[1]); assert(file);
    BnModel model; assert(bn_model_load(&model, file, 256, 0, 0) == 0);
    /* This dense-model gate avoids requiring an external expert-cache budget. */
    assert(!bn_model_uses_moe(&model));
    BnThreadPool *pool = bn_tp_create(7); assert(pool);
    bn_model_set_thread_pool(&model, pool, 1);
    BnGPUBackend *gpu = NULL;
#ifdef BN_ENABLE_CUDA
    if (cuda) {
        gpu = bn_gpu_cuda_create(); assert(gpu);
        assert(bn_model_upload_weights(&model, gpu) == 0);
        assert(bn_model_init_gpu_activations(&model, gpu) == 0);
    }
#endif
    BnTokenizer tok; assert(bn_tokenizer_init(&tok, file) == 0);
    int tokens[128];
    int nt = bn_tokenizer_encode(&tok, "In the year 2020, the world", tok.add_bos, tokens, 128);
    assert(nt > 1 && nt < 128);
    BnSession *session = bn_session_create(&model, NULL); assert(session);
    size_t logits_bytes = (size_t)model.config.vocab_size * sizeof(float);
    float *all = malloc((size_t)nt * logits_bytes); assert(all);
    uint64_t saved_state[9], current_state[9], saved_decode = 0;
    for (int mode = 0; mode < 3; mode++) {
        bn_session_reset(session, &model);
        if (mode == 0) assert(bn_transformer_prefill(&model, session, tokens, nt, 0));
        else if (mode == 1) {
            for (size_t i = 0; i < (size_t)nt * model.config.vocab_size; i++) all[i] = NAN;
            assert(bn_transformer_prefill_all(&model, session, tokens, nt, 0, all) == 0);
            for (size_t i = 0; i < (size_t)nt * model.config.vocab_size; i++) assert(isfinite(all[i]));
        } else assert(bn_transformer_prefill_no_logits(&model, session, tokens, nt, 0) == 0);
        snapshot(&model, session, gpu, current_state);
        if (mode == 0) memcpy(saved_state, current_state, sizeof(saved_state));
        else for (int i = 0; i < 9; i++) {
            if (saved_state[i] != current_state[i]) fprintf(stderr, "State mismatch mode=%d component=%d\n", mode, i);
            assert(saved_state[i] == current_state[i]);
        }
        float *logits = bn_transformer_forward(&model, session, tokens[0], nt); assert(logits);
        uint64_t hash = state_hash(logits, logits_bytes);
        if (mode == 0) saved_decode = hash;
        else assert(hash == saved_decode);
    }
    printf("Prefill output-row state invariance PASSED: backend=%s tokens=%d requests=3 state_components=9\n", cuda ? "cuda" : "cpu", nt);
    free(all); bn_session_free(session, NULL); bn_tokenizer_free(&tok); bn_model_free(&model);
#ifdef BN_ENABLE_CUDA
    if (gpu) bn_gpu_cuda_destroy(gpu);
#endif
    bn_gguf_free(file);
    return 0;
}
